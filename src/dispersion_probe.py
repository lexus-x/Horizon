#!/usr/bin/env python
"""NIAC PREMISE TEST: does the frozen base policy's per-episode sampling
dispersion predict episode FAILURE?

Acts with the faithful base single draw (chunks[0]) so success labels match the
real base policy, while measuring the dispersion of K=8 i.i.d. flow draws at each
replan. Per episode we summarize dispersion (mean / max / early) and test
AUROC(dispersion -> failure).

Decision: if even the best dispersion feature gives AUROC <= ~0.6 (i.e. it cannot
tell failing from succeeding episodes), NIAC's "noise-variance is harmful"
premise has no signal to stand on -> fully dead. AUROC > ~0.65 -> a real,
different premise survives ("the policy is uncertain exactly where it fails").
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from cascade.cascade_eval import build
from lerobot.scripts.lerobot_eval import eval_policy

HARNESS = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HARNESS / "ckpt_smolvla_metaworld"))
    ap.add_argument("--task", default="push-v3")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # batch_size=1 => one episode per rollout => reset()/_ep_idx aligns with per_episode index
    vec, policy, pre, post, env_pre, env_post = build(args.ckpt, args.task, 1, args.device)
    policy.cascade_mode = "probe"
    policy.cascade_K = args.K
    policy._probe_records = []
    policy._ep_idx = -1

    info = eval_policy(
        env=vec, policy=policy,
        env_preprocessor=env_pre, env_postprocessor=env_post,
        preprocessor=pre, postprocessor=post,
        n_episodes=args.n_episodes, max_episodes_rendered=0,
        videos_dir=None, start_seed=args.seed,
    )

    # group dispersion by episode index
    by_ep = defaultdict(list)
    for ep, d in policy._probe_records:
        by_ep[ep].append(d)

    pe = info["per_episode"]
    rows = []
    feats = {"mean": [], "max": [], "early": []}
    fail = []
    for i, e in enumerate(pe):
        ds = by_ep.get(i, [])
        if not ds:
            continue
        k = max(1, len(ds) // 3)
        fm, fx, fe = float(np.mean(ds)), float(np.max(ds)), float(np.mean(ds[:k]))
        feats["mean"].append(fm)
        feats["max"].append(fx)
        feats["early"].append(fe)
        fail.append(0 if e["success"] else 1)
        rows.append({"ep": i, "success": bool(e["success"]), "n_replans": len(ds),
                     "disp_mean": fm, "disp_max": fx, "disp_early": fe})

    y = np.array(fail)
    aurocs = {}
    for name, f in feats.items():
        if len(set(y.tolist())) < 2:
            aurocs[name] = None
        else:
            # AUROC of "higher dispersion => more likely failure"
            aurocs[name] = float(roc_auc_score(y, np.array(f)))

    n = len(fail)
    n_fail = int(y.sum())
    base_succ = 100.0 * (1 - y.mean()) if n else None
    # mean dispersion split by outcome (effect direction + size)
    fm = np.array(feats["mean"])
    disp_fail = float(fm[y == 1].mean()) if n_fail else None
    disp_succ = float(fm[y == 0].mean()) if (n - n_fail) else None

    best = max([v for v in aurocs.values() if v is not None], default=None)
    verdict = ("PREMISE-ALIVE (dispersion predicts failure)" if (best and best >= 0.65)
               else "WEAK (inconclusive)" if (best and best >= 0.60)
               else "PREMISE-DEAD (dispersion does NOT predict failure)")

    result = {
        "task": args.task, "K": args.K, "n_episodes_with_disp": n,
        "base_success_pct": base_succ, "n_fail": n_fail,
        "auroc_disp_to_failure": aurocs,
        "best_auroc": best,
        "mean_disp_failed_eps": disp_fail, "mean_disp_succeeded_eps": disp_succ,
        "verdict": verdict,
    }
    print("RESULT " + json.dumps(result))
    out = args.out or str(HARNESS / f"results/niac_probe/dispersion_{args.task}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"result": result, "per_episode": rows}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
