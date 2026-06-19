#!/usr/bin/env python
"""Paired McNemar between two eval result JSONs on a matched seed pool.

The Horizon stats discipline (METHODOLOGY.md): compare arms on ONE seed pool with
matched n, paired by seed, tested with exact McNemar -- NOT unpaired overlapping
CIs (that shortcut faked a +2pp once; see RESEARCH_LOG entry 3).

Usage: paired_mcnemar.py A.json B.json [--labels A B]
Reports success rates and the exact two-sided McNemar p for "does B differ from A".
"""
import argparse
import json
from math import comb


def load(path):
    d = json.load(open(path))
    per = d["per_episode"] if "per_episode" in d else d
    # map seed -> success (bool)
    out = {}
    for e in per:
        out[int(e["seed"])] = bool(e["success"])
    return out, d.get("result", {})


def exact_mcnemar_p(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=None)
    args = ap.parse_args()

    A, ra = load(args.a)
    B, rb = load(args.b)
    la, lb = args.labels or ("A", "B")

    seeds = sorted(set(A) & set(B))
    na, nb = len(A), len(B)
    if len(seeds) != na or len(seeds) != nb:
        print(f"WARN: A has {na}, B has {nb}, matched {len(seeds)} seeds")

    sa = sum(A[s] for s in seeds)
    sb = sum(B[s] for s in seeds)
    n = len(seeds)
    # discordant pairs
    b_ = sum(1 for s in seeds if A[s] and not B[s])   # A win, B lose
    c_ = sum(1 for s in seeds if B[s] and not A[s])   # B win, A lose  (B helped)
    p = exact_mcnemar_p(b_, c_)

    print(f"matched seeds: n={n}")
    print(f"{la:>18}: {sa}/{n} = {100*sa/n:.1f}%")
    print(f"{lb:>18}: {sb}/{n} = {100*sb/n:.1f}%")
    print(f"delta ({lb}-{la}): {100*(sb-sa)/n:+.1f}pp")
    print(f"{lb} helped (A0->B1): {c_}   {lb} hurt (A1->B0): {b_}")
    print(f"exact McNemar two-sided p = {p:.4g}")
    sig = "SIGNIFICANT" if p < 0.05 else "n.s."
    print(f"verdict: {sig} at alpha=0.05")


if __name__ == "__main__":
    main()
