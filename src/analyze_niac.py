#!/usr/bin/env python
"""Analyze the NIAC test-time noise-averaging probe.

Loads results/niac_probe/*.json, computes Wilson 95% CIs, prints a table and an
honest verdict, and renders a proposal figure (success +/- CI per mode/task).

Headline question: does mean-K (noise-averaged action chunk) beat the single-draw
base on success? mean-K is the inference-time shadow of a noise-invariance
(consistency) objective. consensus-K (medoid) is the on-manifold control.
"""
import json
import math
from pathlib import Path

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
OUT = HARNESS / "results/niac_probe"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def load(tag):
    f = OUT / f"{tag}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    r = d["result"]
    n = r["n_episodes"]
    succ = r["pc_success"] / 100.0
    k = round(succ * n)
    center, lo, hi = wilson(k, n)
    r["_k_succ"] = k
    r["_wilson"] = (center, lo, hi)
    return r


def fmt(r):
    if r is None:
        return None
    c, lo, hi = r["_wilson"]
    disp = r.get("mean_candidate_dispersion")
    disp_s = f"{disp:.3f}" if disp else "  -  "
    return (f"{r['task']:18s} {r['mode']:10s} K={r['K']:<2d} n={r['n_episodes']:<3d} "
            f"succ={r['pc_success']:5.1f}%  CI[{lo:4.1f},{hi:4.1f}]  disp={disp_s}")


def overlap(a, b):
    """True if the two Wilson CIs overlap."""
    _, alo, ahi = a["_wilson"]
    _, blo, bhi = b["_wilson"]
    return not (ahi < blo or bhi < alo)


def main():
    tags = ["push_base", "push_mean_k2", "push_mean_k4", "push_mean_k8",
            "push_consensus_k4", "plate_base", "plate_mean_k4"]
    rs = {t: load(t) for t in tags}

    print("=" * 78)
    print("NIAC TEST-TIME NOISE-AVERAGING PROBE  -  results")
    print("=" * 78)
    for t in tags:
        line = fmt(rs[t])
        print(f"  {line}" if line else f"  {t:18s} (missing)")
    print("=" * 78)

    # ---- verdict logic ----
    print("\nVERDICT")
    pb, pm4 = rs.get("push_base"), rs.get("push_mean_k4")
    lb, lm4 = rs.get("plate_base"), rs.get("plate_mean_k4")

    def delta(base, mean):
        if base is None or mean is None:
            return None
        d = mean["pc_success"] - base["pc_success"]
        ov = overlap(base, mean)
        return d, ov

    notes = []
    pd = delta(pb, pm4)
    if pd:
        d, ov = pd
        notes.append(f"  push-v3:  mean-K4 - base = {d:+.1f} pp  "
                     f"(CIs {'OVERLAP' if ov else 'DISJOINT'})")
    ld = delta(lb, lm4)
    if ld:
        d, ov = ld
        notes.append(f"  plate-slide-v3: mean-K4 - base = {d:+.1f} pp  "
                     f"(CIs {'OVERLAP' if ov else 'DISJOINT'})")

    # dose-response on push
    seq = [rs.get(f"push_mean_k{k}") for k in (2, 4, 8)]
    if pb and all(seq):
        vals = [pb["pc_success"]] + [r["pc_success"] for r in seq]
        mono = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        notes.append(f"  push dose-response base->K2->K4->K8: "
                     f"{'->'.join(f'{v:.0f}' for v in vals)}  "
                     f"({'MONOTONE up' if mono else 'non-monotone'})")
    # mean vs consensus (mechanism)
    pc = rs.get("push_consensus_k4")
    if pm4 and pc:
        notes.append(f"  push mechanism: mean-K4 {pm4['pc_success']:.0f}% vs "
                     f"consensus/medoid-K4 {pc['pc_success']:.0f}%  "
                     f"(mean>medoid => variance-reduction; medoid>=mean => pick-central)")
    print("\n".join(notes) if notes else "  (insufficient results)")

    # ---- honest call ----
    print("\nCALL")
    if pd and ld:
        dp, ovp = pd
        dl, ovl = ld
        pos_both = dp > 0 and dl > 0
        strong = pos_both and (not ovp) and (not ovl)
        if strong:
            print("  STRONG: noise-averaging beats base on BOTH tasks w/ disjoint CIs.")
            print("  -> NIAC premise validated; a trained consistency objective is")
            print("     motivated (recovers this gain at 1x inference).")
        elif pos_both:
            print("  PRELIMINARY-POSITIVE: averaging helps directionally on both tasks,")
            print("  CIs overlap at this n. Honest 'promising preliminary signal' for")
            print("  a proposal; needs more seeds for significance.")
        else:
            print("  NULL/MIXED: noise variance is not the success bottleneck on these")
            print("  tasks (averaging does not consistently help). Report honestly --")
            print("  NIAC premise unsupported here; do NOT claim a boost.")
    else:
        print("  (results incomplete)")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [("push-v3", ["push_base", "push_mean_k2", "push_mean_k4",
                                "push_mean_k8", "push_consensus_k4"]),
                  ("plate-slide-v3", ["plate_base", "plate_mean_k4"])]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (title, tg) in zip(axes, groups):
            labels, cs, los, his = [], [], [], []
            for t in tg:
                r = rs.get(t)
                if r is None:
                    continue
                c, lo, hi = r["_wilson"]
                lab = r["mode"] + (f"-K{r['K']}" if r["mode"] != "base" else "")
                labels.append(lab)
                cs.append(c)
                los.append(c - lo)
                his.append(hi - c)
            x = range(len(labels))
            colors = ["#888"] + ["#2a7" if "mean" in l else "#27a"
                                 for l in labels[1:]]
            ax.bar(x, cs, yerr=[los, his], capsize=4, color=colors[:len(labels)])
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("success %")
            ax.set_ylim(0, 100)
            ax.set_title(title)
            for xi, c in zip(x, cs):
                ax.text(xi, c + 2, f"{c:.0f}", ha="center", fontsize=8)
        fig.suptitle("NIAC premise: test-time noise-averaging vs single-draw base "
                     "(95% Wilson CI)")
        fig.tight_layout()
        figp = OUT / "niac_probe_figure.png"
        fig.savefig(figp, dpi=130)
        print(f"\nfigure -> {figp}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")


if __name__ == "__main__":
    main()
