#!/usr/bin/env python
"""Single-task light finetune of frozen-backbone SmolVLA (expert-only) on one
MetaWorld task. Vanilla flow-matching loss now; NIAC consistency term added next.

PoC comparison (the honest controls): base  vs  vanilla-finetune  vs  NIAC-finetune.
Only the action expert trains; the VLM + vision encoder are frozen. The base was
already trained on metaworld_mt50, so vanilla-FT is the control NIAC must beat.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
CKPT = str(HARNESS / "ckpt_smolvla_metaworld")

# MetaWorld task -> (NL string in dataset, episode index range)
TASKS = {
    "push-v3": ("Push the puck to a goal", range(2000, 2100)),
    "drawer-open-v3": ("Open a drawer", range(950, 1000)),
    "plate-slide-v3": ("Slide a plate into a cabinet", range(1550, 1600)),
    "window-open-v3": ("Push and open a window", range(2400, 2450)),
}


def build_policy(device, train_expert_only=True, freeze_vision=True):
    # SmolVLAConfig.from_pretrained chokes on the `type` field, but the policy
    # loader works. Apply expert-only freezing on the inner model after load:
    # SmolVLMWithExpertModel.set_requires_grad() freezes the VLM, keeps the
    # action expert + projections trainable.
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
    ap.add_argument("--task", default="push-v3", choices=list(TASKS))
    ap.add_argument("--mode", default="vanilla", choices=["vanilla", "niac"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--niac_weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    nl, ep_range = TASKS[args.task]
    eps = list(ep_range)
    out = args.out or str(HARNESS / f"ft_{args.task}_{args.mode}")
    print(f"=== finetune {args.task} ({nl!r}) mode={args.mode} "
          f"eps={eps[0]}..{eps[-1]} steps={args.steps} ===", flush=True)

    policy = build_policy(args.device)
    chunk = policy.config.chunk_size  # 50
    fps = 80
    dt = {
        "action": [i / fps for i in range(chunk)],
        "observation.image": [0.0],
        "observation.state": [0.0],
    }
    ds = LeRobotDataset("lerobot/metaworld_mt50", episodes=eps, delta_timestamps=dt)
    print(f"dataset frames: {ds.num_frames}, episodes: {ds.num_episodes}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": {}}},
    )

    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    t0 = time.time()
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            batch = pre(batch)
            loss_dict = policy.forward(batch)
            # this lerobot's SmolVLA.forward returns (loss, info_dict)
            if isinstance(loss_dict, tuple):
                loss = loss_dict[0]
            elif isinstance(loss_dict, dict):
                loss = loss_dict["loss"]
            else:
                loss = loss_dict
            # NIAC consistency term added in the next increment (mode == "niac")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 10.0)
            opt.step()
            if step % args.log_every == 0:
                print(f"step {step:5d}  loss={loss.item():.4f}  "
                      f"({(time.time()-t0)/max(step,1)*1000:.0f}ms/step)", flush=True)
            if step > 0 and step % args.save_every == 0:
                policy.save_pretrained(out)
            step += 1
    policy.save_pretrained(out)
    print(f"saved -> {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
