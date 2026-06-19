#!/usr/bin/env python
"""Collect closed-loop (h=10) teacher rollouts for trajectory distillation.

The Horizon goal: keep the closed-loop success gain (push 56->78, plate 46->100)
at ~1x inference. The gate route is dead (no leading replan signal; see
RESEARCH_LOG entry 4). This is the distillation route instead: run the *closed
loop* (replan every 10 steps) as a slow teacher, record at every env step the
(normalized obs the policy saw, normalized action it executed), then later train a
single h=50 open-loop student to reproduce the closed loop's executed actions
from the start state.

Why record inside select_action: lerobot's rollout applies the preprocessor
(normalize + tokenize + to-device) BEFORE select_action and the postprocessor
(un-normalize) AFTER it. So the `batch` select_action receives is exactly what
policy.forward() consumes (tokenized language included), and the value it returns
is the action in the policy's normalized output space. Recording here means the
distillation set needs NO re-normalization -- it lives in the same space the
flow-matching loss trains in.

Collection seeds (2000+) are disjoint from the eval pool (1000-1049).
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
from lerobot.utils.constants import ACTION

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
CKPT = str(HARNESS / "ckpt_smolvla_metaworld")

TASKS = {
    "push-v3": "Push the puck to a goal",
    "plate-slide-v3": "Slide a plate into a cabinet",
}


class TeacherRecorder(SmolVLAPolicy):
    """Base SmolVLA run as a closed-loop teacher, recording obs+executed action.

    Set `n_action_steps` low (e.g. 10) to make it closed-loop. Mirrors the stock
    select_action exactly, adding per-step recording. Episodes are delimited by
    reset() (eval_policy calls it once per episode at batch_size=1).
    """

    chunk_size_keep: int = 50      # length of the distillation target window
    snap_stride: int = 2           # snapshot obs every N steps (actions kept every step)
    _ep_actions: list = None       # current episode: (B,A) executed normalized actions, in order
    _ep_snaps: list = None         # current episode: (step_index, obs_snapshot_dict)
    _episodes: list = None         # finished episodes, in rollout order: [(snaps, actions), ...]

    def reset(self):
        super().reset()
        # episode boundary: stash the finished (non-empty) episode in rollout order.
        # eval_policy may fire extra resets before the first stepped episode; those
        # carry no snaps and are skipped, so _episodes stays aligned with
        # info["per_episode"] (used to attach the env success label after the run).
        if self._episodes is None:
            self._episodes = []
        if self._ep_snaps:
            self._episodes.append((self._ep_snaps, self._ep_actions))
        self._ep_actions = []
        self._ep_snaps = []

    def finalize(self):
        if self._ep_snaps:
            self._episodes.append((self._ep_snaps, self._ep_actions))
            self._ep_snaps, self._ep_actions = [], []

    @staticmethod
    def _snapshot(batch):
        snap = {}
        for k, v in batch.items():
            if k == ACTION or not torch.is_tensor(v):
                continue
            v = v.detach().to("cpu")
            if v.is_floating_point():
                v = v.to(torch.float16)
            snap[k] = v
        return snap

    def build_shard(self, snaps, acts):
        """(snaps, executed actions) for one episode -> shard dict of stacked
        (obs, 50-step closed-loop continuation) windows, or None if empty."""
        S = self.chunk_size_keep
        obs_keys = list(snaps[0][1].keys())
        cols = {k: [] for k in obs_keys}
        targets = []
        for step_i, snap in snaps:
            win = acts[step_i: step_i + S]            # closed-loop continuation
            if len(win) == 0:
                continue
            if len(win) < S:                          # pad tail with last action (hold)
                win = win + [win[-1]] * (S - len(win))
            targets.append(torch.stack(win, dim=1)[0])  # (S, A), batch_size 1
            for k in obs_keys:
                cols[k].append(snap[k][0])
        if not targets:
            return None
        shard = {k: torch.stack(v) for k, v in cols.items()}
        shard["action"] = torch.stack(targets)
        return shard

    @torch.no_grad()
    def select_action(self, batch, noise=None, **kwargs):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        step_i = len(self._ep_actions)
        if step_i % self.snap_stride == 0:
            self._ep_snaps.append((step_i, self._snapshot(batch)))

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)        # (B,S,A) normalized
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        a = self._queues[ACTION].popleft()                        # (B,A) normalized
        self._ep_actions.append(a.detach().to("cpu"))
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3", choices=list(TASKS))
    ap.add_argument("--n_action_steps", type=int, default=10, help="teacher horizon (closed loop)")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2000, help="collection seeds (disjoint from eval 1000-1049)")
    ap.add_argument("--snap_stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out or f"../results/teacher_{args.task}_h{args.n_action_steps}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_cfg = MetaworldEnvConfig(task=args.task, obs_type="pixels_agent_pos")
    vec = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.task][0]

    policy = TeacherRecorder.from_pretrained(CKPT)
    policy.to(args.device); policy.eval()
    policy.config.n_action_steps = args.n_action_steps
    policy.snap_stride = args.snap_stride

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "rename_observations_processor": {"rename_map": {}}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    print(f"=== collect teacher h{args.n_action_steps} on {args.task}, "
          f"{args.n_episodes} eps from seed {args.seed} ===", flush=True)
    info = eval_policy(
        env=vec, policy=policy,
        env_preprocessor=env_pre, env_postprocessor=env_post,
        preprocessor=pre, postprocessor=post,
        n_episodes=args.n_episodes, max_episodes_rendered=0, videos_dir=None,
        start_seed=args.seed,
    )
    policy.finalize()   # last episode (no trailing reset)

    # Align recorded episodes (rollout order) with the env per-episode success
    # labels, save one shard per episode tagged with success so distill can filter.
    per_ep = info["per_episode"]
    eps = policy._episodes
    if len(eps) != len(per_ep):
        print(f"WARN: recorded {len(eps)} episodes but info has {len(per_ep)}; "
              f"zipping to min and tagging extras success=None", flush=True)
    n = min(len(eps), len(per_ep))
    total_w = succ_w = n_succ = 0
    ep_meta = []
    for i in range(len(eps)):
        snaps, acts = eps[i]
        shard = policy.build_shard(snaps, acts)
        if shard is None:
            continue
        success = bool(per_ep[i]["success"]) if i < n else None
        shard["success"] = success
        torch.save(shard, out_dir / f"ep_{i:04d}.pt")
        w = shard["action"].shape[0]
        total_w += w
        if success:
            n_succ += 1; succ_w += w
        ep_meta.append({"ep": i, "success": success, "windows": w, "steps": len(acts)})
    print(f"saved {len(ep_meta)} shards: {total_w} windows "
          f"({succ_w} from {n_succ} successful eps)", flush=True)

    meta = {
        "task": args.task,
        "teacher_n_action_steps": args.n_action_steps,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "snap_stride": args.snap_stride,
        "n_windows_total": total_w,
        "n_windows_success": succ_w,
        "n_episodes_success": n_succ,
        "teacher_pc_success": info["aggregated"]["pc_success"],
        "chunk_size": policy.chunk_size_keep,
        "per_episode": ep_meta,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("META " + json.dumps({k: v for k, v in meta.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
