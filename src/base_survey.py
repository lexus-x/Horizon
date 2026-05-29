#!/usr/bin/env python
"""Base-policy success survey across MetaWorld tasks (Part A: find the mid-band).

Goal: locate (task) where the FROZEN base sits in a measurable success band
(~30-75%) so a later method's boost is reviewer-defensible (not floored at 20%,
not ceilinged near 100%). Reuses the faithful harness verbatim
(build_env_and_policy + lerobot eval_policy) so numbers stay bit-for-bit faithful.

For each task we run n_episodes per start-seed across several seeds, pool the
per-episode successes, and report a Wilson 95% CI. One bad task name is logged
and skipped (never kills the whole survey).
"""

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from harness_common import build_env_and_policy
from lerobot.scripts.lerobot_eval import eval_policy


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi) in %."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def _episode_success(ep: dict) -> bool:
    """Pull a success flag out of a per-episode record, defensively."""
    for key in ("success", "is_success", "pc_success"):
        if key in ep:
            v = ep[key]
            return bool(v) if not isinstance(v, (list, tuple)) else bool(v[-1])
    return False


def survey_task(ckpt, task, n_episodes, batch_size, seeds, device):
    """Run base-mode eval for one task across several start-seeds. Pool results."""
    vec, policy, pre, post, env_pre, env_post = build_env_and_policy(
        ckpt, task, batch_size=batch_size, device=device
    )
    per_seed = []
    pooled_succ = 0
    pooled_n = 0
    rewards = []
    for sd in seeds:
        info = eval_policy(
            env=vec,
            policy=policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=pre,
            postprocessor=post,
            n_episodes=n_episodes,
            max_episodes_rendered=0,
            videos_dir=None,
            start_seed=sd,
        )
        agg = info["aggregated"]
        eps = info.get("per_episode", [])
        n_succ = sum(_episode_success(e) for e in eps) if eps else round(
            agg["pc_success"] / 100.0 * n_episodes
        )
        n_tot = len(eps) if eps else n_episodes
        per_seed.append(
            {"seed": sd, "pc_success": agg["pc_success"], "n": n_tot,
             "avg_max_reward": agg.get("avg_max_reward")}
        )
        pooled_succ += n_succ
        pooled_n += n_tot
        if agg.get("avg_max_reward") is not None:
            rewards.append(agg["avg_max_reward"])
        # free per-task env after its last seed handled outside loop

    # clean up GPU/env between tasks
    try:
        vec.close()
    except Exception:
        pass
    del policy
    torch.cuda.empty_cache()

    lo, hi = wilson_ci(pooled_succ, pooled_n)
    seed_rates = [s["pc_success"] for s in per_seed]
    return {
        "task": task,
        "pooled_success_pct": 100.0 * pooled_succ / pooled_n if pooled_n else 0.0,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "pooled_n": pooled_n,
        "seed_rates_pct": seed_rates,
        "seed_rate_std": float(np.std(seed_rates)) if seed_rates else 0.0,
        "mean_avg_max_reward": float(np.mean(rewards)) if rewards else None,
        "per_seed": per_seed,
        "midband": 30.0 <= (100.0 * pooled_succ / pooled_n if pooled_n else 0.0) <= 75.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/user/Desktop/vla_projects/RESEARCH/active/"
        "_harness-faithful-smolvla/ckpt_smolvla_metaworld",
    )
    ap.add_argument(
        "--tasks",
        default="reach-v3,button-press-v3,drawer-open-v3,"
        "pick-place-v3,peg-insert-side-v3",
    )
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seeds", default="1000,2000,3000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        default="/home/user/Desktop/vla_projects/RESEARCH/active/"
        "_harness-faithful-smolvla/results/base_survey_metaworld.json",
    )
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Survey: {len(tasks)} tasks x {len(seeds)} seeds x "
          f"{args.n_episodes} eps = {len(tasks)*len(seeds)*args.n_episodes} rollouts")

    results = []
    t0 = time.time()
    for task in tasks:
        ts = time.time()
        print(f"\n=== {task} ===", flush=True)
        try:
            r = survey_task(args.ckpt, task, args.n_episodes, args.batch_size,
                            seeds, args.device)
            band = "MID-BAND" if r["midband"] else ""
            print(f"  success={r['pooled_success_pct']:.1f}% "
                  f"[{r['wilson95_lo']:.1f},{r['wilson95_hi']:.1f}] "
                  f"n={r['pooled_n']} seed_std={r['seed_rate_std']:.1f} "
                  f"({time.time()-ts:.0f}s) {band}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {task}: {e}", flush=True)
            traceback.print_exc()
            results.append({"task": task, "error": str(e)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"survey": results,
                   "config": {"tasks": tasks, "seeds": seeds,
                              "n_episodes": args.n_episodes,
                              "ckpt": args.ckpt},
                   "wall_s": time.time() - t0}, f, indent=2)

    print(f"\n===== SURVEY TABLE ({time.time()-t0:.0f}s) =====")
    print(f"{'task':<22} {'succ%':>7} {'95% CI':>16} {'seed_std':>9}  band")
    for r in results:
        if "error" in r:
            print(f"{r['task']:<22} {'ERR':>7}  {r['error'][:30]}")
            continue
        ci = f"[{r['wilson95_lo']:.0f},{r['wilson95_hi']:.0f}]"
        band = "<== MID" if r["midband"] else ""
        print(f"{r['task']:<22} {r['pooled_success_pct']:>6.1f}% {ci:>16} "
              f"{r['seed_rate_std']:>8.1f}%  {band}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
