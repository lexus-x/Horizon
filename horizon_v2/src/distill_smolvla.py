#!/usr/bin/env python
"""Distill the closed-loop (h=10) teacher into a single-pass (h=50) student.

The student is the SAME frozen-backbone SmolVLA, expert-only light finetune (VLM
+ vision frozen). Target = the 50-step action chunk the *closed loop* actually
executed from each start state (collected by collect_teacher.py). Trained with the
stock flow-matching loss (policy.forward), so the student learns to emit, in ONE
open-loop forward pass, the trajectory the closed loop produced over 5 replans.

The teacher windows are recorded post-preprocessor (already normalized + tokenized)
so NO pre() is applied here -- batches go straight into policy.forward(), exactly
the space the flow-matching loss trains in.

Honest controls live in finetune_smolvla.py (vanilla FT on demos at matched steps);
the kill-gate is: distilled student at h=50 must (a) recover most of the closed-loop
gain and (b) beat vanilla-FT at matched compute. If it only ties vanilla-FT, the
distillation target added nothing.
"""
import argparse
import glob
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
CKPT = str(HARNESS / "ckpt_smolvla_metaworld")

OBS_IMAGE = "observation.image"
OBS_STATE = "observation.state"
ACTION = "action"


class TeacherWindows(Dataset):
    """All (obs, 50-step closed-loop action) windows from teacher shards, in RAM.

    success_only: distill on SUCCESSFUL closed-loop episodes only (filtered BC) --
    the point is to reproduce the closed loop's *wins*, not imitate its failures.
    Shards tagged success=False are dropped; success True/None (unlabeled) are kept.
    """

    def __init__(self, shard_dir, success_only=True):
        files = sorted(glob.glob(str(Path(shard_dir) / "ep_*.pt")))
        assert files, f"no shards in {shard_dir}"
        kept, n_skip = [], 0
        for f in files:
            p = torch.load(f, map_location="cpu")
            succ = p.pop("success", None)
            if success_only and succ is False:
                n_skip += 1
                continue
            kept.append(p)
        assert kept, f"no usable shards in {shard_dir} (success_only={success_only})"
        self.keys = [k for k in kept[0].keys()]
        self.data = {k: torch.cat([p[k] for p in kept], dim=0) for k in self.keys}
        self.n = self.data[ACTION].shape[0]
        print(f"loaded {len(kept)} shards (skipped {n_skip} failed) -> {self.n} windows; "
              f"img {tuple(self.data[OBS_IMAGE].shape)}, act {tuple(self.data[ACTION].shape)}",
              flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}


def build_policy(device, train_expert_only=True, freeze_vision=True):
    policy = SmolVLAPolicy.from_pretrained(CKPT)
    mwe = policy.model.vlm_with_expert
    mwe.train_expert_only = train_expert_only
    mwe.freeze_vision_encoder = freeze_vision
    mwe.set_requires_grad()
    policy.config.train_expert_only = train_expert_only
    policy.config.freeze_vision_encoder = freeze_vision
    policy.to(device)
    policy.train()
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in policy.parameters())
    print(f"trainable params: {n_train/1e6:.1f}M / {n_tot/1e6:.1f}M "
          f"({100*n_train/n_tot:.1f}%)", flush=True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="teacher shard dir from collect_teacher.py")
    ap.add_argument("--success_only", type=int, default=1, help="1=distill on successful teacher episodes only")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2_coeff", type=float, default=0.0,
                    help="L2-to-init regularization coefficient. Penalises ||w - w0||^2 "
                         "for trainable params to prevent catastrophic forgetting. "
                         "0=off (original behaviour). Try 0.01, 0.1, 1.0.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    print(f"=== distill student from {args.shards} for {args.steps} steps "
          f"(l2_coeff={args.l2_coeff}) ===", flush=True)
    policy = build_policy(args.device)

    # Snapshot initial trainable weights for L2-to-init penalty.
    init_params = {}
    if args.l2_coeff > 0.0:
        for name, p in policy.named_parameters():
            if p.requires_grad:
                init_params[name] = p.detach().clone()
        print(f"L2-to-init: snapshotted {len(init_params)} trainable param tensors", flush=True)

    ds = TeacherWindows(args.shards, success_only=bool(args.success_only))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                    drop_last=True, pin_memory=True)

    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    def to_device(batch):
        # Inputs must be float32 (matches what pre() produces): expert projections
        # are fp32 while the frozen VLM backbone is bf16 and casts internally.
        out = {}
        for k, v in batch.items():
            v = v.to(args.device)
            if v.is_floating_point():
                v = v.to(torch.float32)
            out[k] = v
        return out

    t0 = time.time()
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            batch = to_device(batch)
            loss_dict = policy.forward(batch)
            loss = loss_dict[0] if isinstance(loss_dict, tuple) else (
                loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict)

            if args.l2_coeff > 0.0:
                l2 = sum(
                    (p - init_params[n]).pow(2).sum()
                    for n, p in policy.named_parameters()
                    if p.requires_grad
                )
                loss = loss + args.l2_coeff * l2

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 10.0)
            opt.step()
            if step % args.log_every == 0:
                print(f"step {step:5d}  loss={loss.item():.4f}  "
                      f"({(time.time()-t0)/max(step,1)*1000:.0f}ms/step)", flush=True)
            if step > 0 and step % args.save_every == 0:
                policy.save_pretrained(args.out)
            step += 1
    policy.save_pretrained(args.out)
    print(f"saved student -> {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
