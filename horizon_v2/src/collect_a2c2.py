#!/usr/bin/env python
"""Collect A2C2 training data: at each step of an OPEN-LOOP base rollout, record
(latest state, stale base action, chunk position, re-observed target action).

target = the action the base would take if it re-ran the VLM at the CURRENT state
(= fresh chunk[0]); base = the stale open-loop chunk action at this index. The head
learns correction = target - base from (state, base, position) WITHOUT re-running
the VLM. States come from the open-loop base's own rollout (deployment distribution).

Seeds 3000+ (disjoint from eval pool 1000-1049 and teacher pool 2000-2049).
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
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION, OBS_STATE

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
CKPT = str(HARNESS / "ckpt_smolvla_metaworld")
TASKS = {"push-v3": "Push the puck to a goal", "plate-slide-v3": "Slide a plate into a cabinet"}


class A2C2Collector(SmolVLAPolicy):
    """Open-loop base rollout; per step record (state, base_a, pos, re-observed target)."""
    _pos: int = 0
    _rec: list = None  # list of dict(state, base_a, pos_frac, target)

    @torch.no_grad()
    def select_action(self, batch, noise=None, **kwargs):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:                 # plan a fresh open-loop chunk
            chunk = self._get_action_chunk(batch, noise)   # (B,S,A)
            self._queues[ACTION].extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
            self._pos = 0

        # re-observed target = fresh chunk[0] at the CURRENT obs (one extra VLM pass, offline only)
        target = self._get_action_chunk(batch, None)[:, 0, :]   # (B,A)
        base_a = self._queues[ACTION].popleft()                  # (B,A) stale open-loop action

        if self._rec is not None:
            state = batch[OBS_STATE].detach().to("cpu")
            if state.ndim > 2:
                state = state[:, -1, :]
            self._rec.append({
                "state": state.float(),                     # (B,Ds)
                "base_a": base_a.detach().to("cpu").float(),  # (B,A)
                "pos_frac": torch.full((base_a.shape[0], 1), self._pos / self.config.chunk_size),
                "target": target.detach().to("cpu").float(),  # (B,A)
            })
        self._pos += 1
        return base_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3", choices=list(TASKS))
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"../results/a2c2data_{args.task}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env_cfg = MetaworldEnvConfig(task=args.task, obs_type="pixels_agent_pos")
    vec = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.task][0]
    policy = A2C2Collector.from_pretrained(CKPT)
    policy.to(args.device); policy.eval()
    policy.config.n_action_steps = policy.config.chunk_size  # OPEN-LOOP base (replan every 50)
    policy._rec = []

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    print(f"=== collect A2C2 data on {args.task}, {args.n_episodes} eps seed {args.seed} ===", flush=True)
    info = eval_policy(env=vec, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                       preprocessor=pre, postprocessor=post, n_episodes=args.n_episodes,
                       max_episodes_rendered=0, videos_dir=None, start_seed=args.seed)

    data = {k: torch.cat([r[k] for r in policy._rec], dim=0) for k in ["state", "base_a", "pos_frac", "target"]}
    data["meta"] = {"task": args.task, "n_episodes": args.n_episodes, "seed": args.seed,
                    "n_samples": int(data["state"].shape[0]),
                    "openloop_pc_success": info["aggregated"]["pc_success"]}
    torch.save(data, out)
    print(f"saved {data['state'].shape[0]} samples -> {out}  (open-loop base success "
          f"{info['aggregated']['pc_success']:.0f}%)", flush=True)
    print("META " + json.dumps(data["meta"]), flush=True)


if __name__ == "__main__":
    main()
