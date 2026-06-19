#!/usr/bin/env python
"""Train-free test-time action-chunk selection for SmolVLA on MetaWorld.

Reuses lerobot's faithful eval pipeline (env + official pre/post processors +
eval_policy rollout loop) and only swaps the action-selection step via a
SmolVLAPolicy subclass. This keeps the base policy bit-for-bit faithful
(verified: reach-v3 = 15% with the stock select_action).

Modes
-----
- base        : K=1, identical to stock select_action (sanity / control).
- consensus   : best-of-N, pick the medoid chunk (min mean L2 distance to the
                other K-1 candidates). Train-free "self-consistency" verifier.
- proofreading: staged irreversible filtering with independent checks
                (consensus -> smoothness) then medoid of survivors.

All modes share seeds for an apples-to-apples comparison. Candidate dispersion
is logged per replan so we can later test the kinetic-proofreading
multiplicative-fidelity prediction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION, OBS_STATE
from replan_gate import ReplanGate, compute_prefix_feature
from a2c2_head import A2C2Head


# ----------------------------- selection logic ------------------------------
def _medoid_index(chunks_kbsa: torch.Tensor) -> torch.Tensor:
    """chunks_kbsa: (K, B, S, A). Return (B,) index of medoid candidate per env.

    Medoid = candidate with minimum mean L2 distance to the other candidates
    (flattened over S*A). Robust 'consensus' choice.
    """
    K, B = chunks_kbsa.shape[:2]
    flat = chunks_kbsa.reshape(K, B, -1)  # (K,B,D)
    # pairwise distances over K for each B: (K,K,B)
    diff = flat.unsqueeze(1) - flat.unsqueeze(0)  # (K,K,B,D)
    dist = diff.norm(dim=-1)  # (K,K,B)
    mean_dist = dist.mean(dim=1)  # (K,B) mean dist of each cand to all others
    return mean_dist.argmin(dim=0)  # (B,)


def _smoothness_cost(chunks_kbsa: torch.Tensor) -> torch.Tensor:
    """(K,B,S,A) -> (K,B) jerk cost: mean squared 2nd difference along time.

    Lower = physically smoother chunk (independent-ish signal from consensus).
    """
    d2 = chunks_kbsa[:, :, 2:] - 2 * chunks_kbsa[:, :, 1:-1] + chunks_kbsa[:, :, :-2]
    return (d2 ** 2).mean(dim=(2, 3))  # (K,B)


class CascadeSmolVLA(SmolVLAPolicy):
    """SmolVLA with train-free test-time chunk selection at each replan."""

    # configured after construction
    cascade_mode: str = "base"
    cascade_K: int = 1
    cascade_keep_frac: float = 0.5
    _disp_log: list | None = None
    # probe mode: act with the base draw, record per-episode candidate dispersion
    _ep_idx: int = -1
    _probe_records: list | None = None  # list of (episode_idx, dispersion)
    
    # gate properties
    gate_model: torch.nn.Module = None
    gate_threshold: float = 0.5
    _gate_replan_counts: list = None

    # a2c2 properties: open-loop base chunk + cheap per-step correction head
    a2c2_head: torch.nn.Module = None
    _a2c2_pos: int = 0

    def reset(self):
        super().reset()
        # eval_policy calls reset() once per rollout; with batch_size=1 that is
        # once per episode, so this counter aligns with per_episode index.
        if self.cascade_mode == "probe":
            self._ep_idx += 1

    @torch.no_grad()
    def select_action(self, batch, noise=None, **kwargs):
        self.eval()
        batch = self._prepare_batch(batch)
        from lerobot.policies.utils import populate_queues

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self.cascade_mode == "gate" and len(self._queues[ACTION]) > 0 and self.gate_model is not None:
            # Replan check
            t = self.config.chunk_size - len(self._queues[ACTION])
            # We must compute prefix feature. We don't cache because it's fast (no denoise loop)
            feat = compute_prefix_feature(self, dict(batch)) # (B, D)
            step_idx = torch.full((feat.shape[0], 1), t / self.config.chunk_size, device=feat.device)
            prob = self.gate_model(feat, step_idx).squeeze(-1) # (B,)
            
            if prob.max().item() > self.gate_threshold:
                self._queues[ACTION].clear()
                if self._gate_replan_counts is not None:
                    self._gate_replan_counts.append(1)

        if len(self._queues[ACTION]) == 0:
            if self.cascade_mode == "gate" and self._gate_replan_counts is not None:
                self._gate_replan_counts.append(1) # We planned a chunk
            chosen = self._plan_chunk(batch)  # (B,S,A)
            self._queues[ACTION].extend(chosen.transpose(0, 1)[: self.config.n_action_steps])
            self._a2c2_pos = 0  # fresh open-loop chunk -> reset chunk position

        a = self._queues[ACTION].popleft()
        if self.cascade_mode == "a2c2" and self.a2c2_head is not None:
            # cheap per-step correction from latest proprio state (no VLM rerun)
            state = batch[OBS_STATE]
            if state.ndim > 2:
                state = state[:, -1, :]
            pos = torch.full((a.shape[0], 1), self._a2c2_pos / self.config.chunk_size,
                             device=a.device, dtype=torch.float32)
            a = a + self.a2c2_head(state.float(), a.float(), pos)
            self._a2c2_pos += 1
        return a

    @torch.no_grad()
    def _plan_chunk(self, batch):
        K = self.cascade_K if self.cascade_mode != "base" else 1
        # Generate K candidate chunks. noise=None => fresh flow noise each call.
        cands = []
        for _ in range(K):
            # _get_action_chunk reads self._queues (already populated) for obs stacking.
            cands.append(self._get_action_chunk(dict(batch), noise=None))  # (B,S,A)
        chunks = torch.stack(cands, dim=0)  # (K,B,S,A)

        # log dispersion (mean pairwise dist) per env
        if self._disp_log is not None and K > 1:
            flat = chunks.reshape(K, chunks.shape[1], -1)
            d = (flat.unsqueeze(1) - flat.unsqueeze(0)).norm(dim=-1)  # (K,K,B)
            self._disp_log.append(float(d.mean().item()))

        if self.cascade_mode in ["base", "gate", "a2c2"]:
            return chunks[0]

        if self.cascade_mode == "probe":
            # NIAC PREMISE TEST: act with the faithful base draw (chunks[0]) but
            # record this state's candidate dispersion (spread of K draws) tagged
            # to the current episode. Downstream we test whether dispersion
            # predicts episode FAILURE (AUROC). If it does not, NIAC's
            # "noise-variance is harmful" premise has nothing to stand on.
            K_, B = chunks.shape[:2]
            flat = chunks.reshape(K_, B, -1)
            d = (flat.unsqueeze(1) - flat.unsqueeze(0)).norm(dim=-1)  # (K,K,B)
            per_env = d.sum(dim=(0, 1)) / max(K_ * (K_ - 1), 1)  # (B,) mean pairwise
            if self._probe_records is not None:
                for b in range(B):
                    self._probe_records.append((int(self._ep_idx), float(per_env[b].item())))
            return chunks[0]

        if self.cascade_mode == "mean":
            # NIAC premise probe: element-wise average over K noise draws.
            # This is the test-time shadow of a noise-invariance/consistency
            # objective -- it collapses the flow's noise-induced action variance
            # to the conditional-mean action. If success rises, the variance was
            # harmful jitter (not meaningful multimodality) -> NIAC is motivated
            # and amortizes this to a single forward pass at deploy time.
            return chunks.mean(dim=0)

        if self.cascade_mode == "consensus":
            idx = _medoid_index(chunks)  # (B,)
            return chunks[idx, torch.arange(chunks.shape[1])]

        if self.cascade_mode == "proofreading":
            # Stage 1 (irreversible): drop candidates farthest from consensus.
            K_, B = chunks.shape[:2]
            flat = chunks.reshape(K_, B, -1)
            mean_d = (flat.unsqueeze(1) - flat.unsqueeze(0)).norm(dim=-1).mean(dim=1)  # (K,B)
            keep = max(2, int(round(self.cascade_keep_frac * K_)))
            keep_idx = mean_d.argsort(dim=0)[:keep]  # (keep,B) closest-to-consensus survive
            # Stage 2 (independent check): among survivors pick lowest jerk.
            jerk = _smoothness_cost(chunks)  # (K,B)
            out = torch.empty_like(chunks[0])
            for b in range(B):
                surv = keep_idx[:, b]
                best = surv[jerk[surv, b].argmin()]
                out[b] = chunks[best, b]
            return out

        raise ValueError(self.cascade_mode)


def build(ckpt: str, task: str, batch_size: int, device: str, proc_ckpt: str = None):
    # proc_ckpt: where to load the pre/post processors (normalization stats) from.
    # A distilled/finetuned student dir only holds model+config; its processors are
    # identical to the base ckpt's (we train in normalized space), so load them from
    # the base ckpt when evaluating a student.
    proc_ckpt = proc_ckpt or ckpt
    env_cfg = MetaworldEnvConfig(task=task, obs_type="pixels_agent_pos")
    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)
    vec = envs[task][0]

    policy = CascadeSmolVLA.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=proc_ckpt,
        preprocessor_overrides={
            "device_processor": {"device": device},
            # Disable the checkpoint's stale rename map (observation.image ->
            # observation.images.camera1) exactly like lerobot-eval does, so the
            # env's observation.image matches the patched single-camera config.
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    return vec, policy, pre, post, env_pre, env_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld")
    ap.add_argument("--proc_ckpt", default=None,
                    help="load pre/post processors (norm stats) from here; defaults to --ckpt. "
                         "Point to base ckpt when --ckpt is a distilled/finetuned student dir.")
    ap.add_argument("--task", default="reach-v3")
    ap.add_argument("--mode", choices=["base", "mean", "consensus", "proofreading", "gate", "a2c2"], default="base")
    ap.add_argument("--gate_weights", default=None, help="Path to trained gate weights .pt file")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--a2c2_head", default=None, help="Path to trained A2C2 head .pt file")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--n_action_steps", type=int, default=None,
                    help="override execution horizon before replanning (Candidate-A pre-test)")
    ap.add_argument("--num_steps", type=int, default=None,
                    help="override flow denoise steps (default 10). Cheap-closed-loop probe: "
                         "fewer steps per inference so closed-loop costs ~1x open-loop.")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--videos_dir", default=None, help="if set, render rollout MP4s here")
    ap.add_argument("--max_render", type=int, default=0, help="number of episodes to render to video")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vec, policy, pre, post, env_pre, env_post = build(
        args.ckpt, args.task, args.batch_size, args.device, args.proc_ckpt
    )
    policy.cascade_mode = args.mode
    policy.cascade_K = args.K
    policy.cascade_keep_frac = args.keep_frac
    policy._disp_log = []
    policy._gate_replan_counts = []
    
    if args.mode == "gate":
        if not args.gate_weights:
            print("WARNING: --mode gate used but --gate_weights not provided.")
        else:
            sd = torch.load(args.gate_weights, map_location=args.device, weights_only=True)
            feature_dim = sd['net.0.weight'].shape[1] - 1
            gate = ReplanGate(feature_dim=feature_dim).to(args.device)
            gate.load_state_dict(sd)
            gate.eval()
            policy.gate_model = gate
            policy.gate_threshold = args.gate_threshold

    if args.mode == "a2c2":
        if not args.a2c2_head:
            print("WARNING: --mode a2c2 used but --a2c2_head not provided.")
        else:
            hd = torch.load(args.a2c2_head, map_location=args.device, weights_only=False)
            head = A2C2Head(state_dim=hd["state_dim"], action_dim=hd["action_dim"], hidden=hd["hidden"]).to(args.device)
            head.load_state_dict(hd["state_dict"]); head.eval()
            policy.a2c2_head = head
            
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps  # shorter open-loop horizon
        print(f"n_action_steps override -> {args.n_action_steps}", flush=True)
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps  # cheaper denoise per inference
        print(f"num_steps (denoise) override -> {args.num_steps}", flush=True)

    info = eval_policy(
        env=vec,
        policy=policy,
        env_preprocessor=env_pre,
        env_postprocessor=env_post,
        preprocessor=pre,
        postprocessor=post,
        n_episodes=args.n_episodes,
        max_episodes_rendered=args.max_render,
        videos_dir=Path(args.videos_dir) if args.videos_dir else None,
        start_seed=args.seed,
    )
    agg = info["aggregated"]
    disp = policy._disp_log
    replan_counts = policy._gate_replan_counts
    total_replans = sum(replan_counts) if replan_counts else 0
    avg_replans = total_replans / args.n_episodes if args.n_episodes > 0 else 0
    
    result = {
        "mode": args.mode,
        "K": args.K if args.mode not in ["base", "gate"] else 1,
        "task": args.task,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "pc_success": agg["pc_success"],
        "avg_max_reward": agg["avg_max_reward"],
        "eval_ep_s": agg["eval_ep_s"],
        "mean_candidate_dispersion": float(np.mean(disp)) if disp else None,
        "avg_replans_per_ep": avg_replans,
    }
    print("RESULT " + json.dumps(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"result": result, "per_episode": info["per_episode"]}, f, indent=2)


if __name__ == "__main__":
    main()
