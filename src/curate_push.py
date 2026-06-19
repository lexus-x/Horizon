"""Reset-Curriculum stratifier (overnight, train-free).

Split the push closed-loop teacher's SUCCESSFUL episodes into a start-PREDICTABLE
stratum vs a start-UNPREDICTABLE stratum, matched in size, by a reactivity score =
how much the teacher re-plans away from its original chunk plan as it re-observes.

Low score  = executed trajectory ~ a function of the chunk-start state (SHAPE).
High score = the teacher changes its mind mid-chunk on live obs (REACTIVE / info-bound).

Premise of Reset-Curriculum: if distillation collapse is driven by the unpredictable
stratum, then distilling on ONLY the predictable stratum should reduce the push collapse
-> first evidence the shape/reactive boundary is MOVABLE by editing the init distribution.

Writes ep_*.pt symlinks into teacher_push_pred/ and teacher_push_unpred/ (matched size),
so distill_smolvla.py --shards <dir> works unchanged.
"""
import glob, os, json
from pathlib import Path
import numpy as np
import torch

SRC = Path("../results/teacher_push-v3_h10")
STRIDE = 2  # snap_stride used in collect_teacher
CHUNK = 50

def first_chunk(ep):
    """the executed first 50-step chunk from the episode start = action[0], flattened."""
    return ep["action"].float().numpy()[0].reshape(-1)  # [200]

def main():
    files = sorted(glob.glob(str(SRC / "ep_*.pt")))
    # pass 1: collect successful episodes' start chunks
    succ_rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d.get("success", None) is False:
            continue
        succ_rows.append((f, first_chunk(d), int(d["action"].shape[0])))
    chunks = np.stack([r[1] for r in succ_rows])
    mean_chunk = chunks.mean(0)
    # reactivity proxy = atypicality of the executed start-chunk vs the canonical motion.
    # Low = typical/start-predictable (shape-like); high = idiosyncratic correction (reactive).
    # (within-chunk replan-divergence was absent in stored targets, so we use trajectory
    #  typicality as the start-predictability proxy -- honest caveat in the report.)
    rows = [(f, float(np.linalg.norm(c - mean_chunk)), T) for (f, c, T) in succ_rows]
    rows.sort(key=lambda r: r[1])  # ascending atypicality
    n = len(rows)
    half = n // 2
    pred = rows[:half]              # low reactivity = start-predictable (shape-like)
    unpred = rows[-half:]           # high reactivity = info-bound (reactive)
    scores = np.array([r[1] for r in rows])
    print(f"[curate_push] {n} successful push teacher eps; reactivity "
          f"min {scores.min():.4f} med {np.median(scores):.4f} max {scores.max():.4f}")
    print(f"  predictable stratum: {len(pred)} eps, mean react {np.mean([r[1] for r in pred]):.4f}")
    print(f"  unpredictable stratum: {len(unpred)} eps, mean react {np.mean([r[1] for r in unpred]):.4f}")

    for name, group in [("teacher_push_pred", pred), ("teacher_push_unpred", unpred)]:
        out = Path("../results") / name
        if out.exists():
            for p in out.glob("ep_*.pt"):
                p.unlink()
        out.mkdir(exist_ok=True)
        for i, (f, sc, T) in enumerate(group):
            dst = out / f"ep_{i:04d}.pt"
            try:
                os.symlink(os.path.abspath(f), dst)
            except FileExistsError:
                pass
        print(f"  wrote {len(group)} shards -> {out}")

    Path("../results/curate_push_scores.json").write_text(json.dumps({
        "n_success": n, "reactivity_scores": [r[1] for r in rows],
        "files": [os.path.basename(r[0]) for r in rows],
        "split_index": half,
    }, indent=2))
    print("[curate_push] done")

if __name__ == "__main__":
    main()
