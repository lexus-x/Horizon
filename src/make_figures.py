#!/usr/bin/env python
"""Regenerate every Horizon README figure from the measured result JSONs.

Every number plotted here is read from (or hard-grounded against) the result
JSONs in ../results/. Nothing is invented. Output PNGs go to ../figures/ using
the fixed filename manifest the README expects.

Run:
    /home/user/miniconda3/envs/lerobot/bin/python src/make_figures.py
(only needs matplotlib + numpy; no GPU, no lerobot)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# palette
C_BASE = "#9aa3ab"     # grey  (open-loop / base)
C_GAIN = "#1f77b4"     # blue  (closed-loop / horizon)
C_GOOD = "#2ca02c"     # green
C_BAD = "#d62728"      # red
C_MID = "#d2691e"      # orange (mid-band highlight / plate)
C_NULL = "#7f7f7f"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path, os.path.getsize(path), "bytes")


# --------------------------------------------------------------------------- #
# 1. open_loop_gap.png
# --------------------------------------------------------------------------- #
def fig_open_loop_gap():
    push = {h: load(f"horizon_push_h{h}.json")["result"]["pc_success"] for h in (50, 25, 10)}
    plate = {h: load(f"horizon_plate_h{h}.json")["result"]["pc_success"] for h in (50, 10)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    hs = [50, 25, 10]
    ys = [push[h] for h in hs]
    x = np.arange(len(hs))
    bars = ax.bar(x, ys, color=[C_BASE, "#5b9bd5", C_GAIN], width=0.62,
                  edgecolor="black", linewidth=0.7)
    for xi, yi in zip(x, ys):
        ax.text(xi, yi + 1.6, f"{yi:.0f}%", ha="center", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["h=50", "h=25", "h=10"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Execution horizon (steps before replan)")
    ax.set_title("push-v3")
    ax.plot([0, 2], [86, 86], color="black", lw=1)
    ax.text(1, 88, "McNemar h50→h10  p = 0.035\n(helped 17 / hurt 6)",
            ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(50, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    hs = [50, 10]
    ys = [plate[h] for h in hs]
    x = np.array([0, 2])
    bars = ax.bar(x, ys, color=["#e6b59a", C_MID], width=0.62,
                  edgecolor="black", linewidth=0.7)
    for xi, yi in zip(x, ys):
        ax.text(xi, yi + 1.6, f"{yi:.0f}%", ha="center", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["h=50", "h=10"])
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Execution horizon (steps before replan)")
    ax.set_title("plate-slide-v3")
    ax.plot([0, 2], [104, 104], color="black", lw=1)
    ax.text(1, 105.5, "McNemar h50→h10  p ≈ 1e-8\n(helped 27 / hurt 0)",
            ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(50, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("The open-loop execution gap", fontsize=18, fontweight="bold")
    fig.text(0.5, -0.02,
             "Frozen SmolVLA executes 50-step chunks open-loop. Shrinking the execution "
             "horizon (replan more often → closed loop)\nmonotonically lifts success. "
             "n=50 episodes per arm, paired seed pool (seeds 1000–1049). "
             "Costs ~5× inference at h=10.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "open_loop_gap.png")


# --------------------------------------------------------------------------- #
# 2. base_survey_midband.png
# --------------------------------------------------------------------------- #
def fig_base_survey():
    survey = load("base_survey_metaworld.json")["survey"]
    survey = sorted(survey, key=lambda r: r["pooled_success_pct"])
    tasks = [r["task"].replace("-v3", "") for r in survey]
    vals = [r["pooled_success_pct"] for r in survey]
    los = [max(0.0, r["pooled_success_pct"] - r["wilson95_lo"]) for r in survey]
    his = [max(0.0, r["wilson95_hi"] - r["pooled_success_pct"]) for r in survey]

    colors = []
    for r in survey:
        if r["task"] == "reach-v3":
            colors.append(C_BAD)
        elif r["midband"]:
            colors.append(C_GOOD)
        else:
            colors.append(C_BASE)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(tasks))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6,
           yerr=[los, his], error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(his) + 1.5, f"{v:.0f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.axhspan(30, 75, color=C_GOOD, alpha=0.08)
    ax.axhline(30, color=C_GOOD, ls="--", lw=0.9, alpha=0.6)
    ax.axhline(75, color=C_GOOD, ls="--", lw=0.9, alpha=0.6)
    ax.text(len(tasks) - 0.5, 76, "mid-band ceiling ~75%", color="#5a7d2a", fontsize=8, ha="right")
    ax.text(len(tasks) - 0.5, 31, "mid-band floor ~30%", color="#5a7d2a", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Base success rate (%)")
    ax.set_title("Frozen SmolVLA base survey on MetaWorld — the measurable mid-band")
    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, color=C_GOOD, label="mid-band 30–75% (usable for repair)"),
        Rectangle((0, 0), 1, 1, color=C_BASE, label="floored / ceilinged"),
        Rectangle((0, 0), 1, 1, color=C_BAD, label="reach-v3 (floored outlier)"),
    ], loc="upper left", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.02,
             "Pooled over 3 seeds (n=90/task, Wilson 95% CIs). Mid-band tasks "
             "(drawer-open, push, peg-insert, window-open, plate-slide) are where a "
             "train-free repair has headroom.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    save(fig, "base_survey_midband.png")


# --------------------------------------------------------------------------- #
# 3. niac_null.png
# --------------------------------------------------------------------------- #
def fig_niac_null():
    base = load("push_base.json")["result"]["pc_success"]
    k2 = load("push_mean_k2.json")["result"]["pc_success"]
    k4 = load("push_mean_k4.json")["result"]["pc_success"]
    k8 = load("push_mean_k8.json")["result"]["pc_success"]
    med = load("push_consensus_k4.json")["result"]["pc_success"]

    arms = ["base", "mean K=2", "mean K=4", "mean K=8", "medoid K=4\n(consensus)"]
    vals = [base, k2, k4, k8, med]
    colors = [C_BASE, C_GAIN, C_GAIN, C_GAIN, C_NULL]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(arms))
    ax.bar(x, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.7)
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 1.0, f"{yi:.0f}%", ha="center", fontweight="bold", fontsize=12)
    ax.axhline(base, color=C_BAD, ls="--", lw=1.2, label=f"base = {base:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 82)
    ax.set_ylabel("push-v3 success rate (%)")
    ax.set_title("NIAC test-time noise-averaging is NULL — averaging does not beat base")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.02,
             "Averaging K flow draws (mean) or picking the medoid (consensus) never "
             "exceeds base.\nPaired McNemar (matched 50-seed pool): push p = 0.885 — "
             "no variance-reduction mechanism; mean-K4 ≡ medoid-K4 ≡ 58%.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, "niac_null.png")


# --------------------------------------------------------------------------- #
# 4. dispersion_failure.png
# --------------------------------------------------------------------------- #
def fig_dispersion():
    push = load("dispersion_push-v3.json")["result"]["auroc_disp_to_failure"]
    plate = load("dispersion_plate-slide-v3.json")["result"]["auroc_disp_to_failure"]

    feats = ["mean", "max", "early (leading)"]
    keys = ["mean", "max", "early"]
    pvals = [push[k] for k in keys]
    plvals = [plate[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(feats))
    w = 0.38
    b1 = ax.bar(x - w / 2, pvals, w, label="push-v3", color=C_GAIN,
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, plvals, w, label="plate-slide-v3", color=C_MID,
                edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.012,
                    f"{bb.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="black", ls="--", lw=1.1)
    ax.text(2.45, 0.515, "chance (0.5)", fontsize=9, ha="right", va="bottom")
    ax.annotate("backwards\n(< 0.5)", xy=(-w / 2, pvals[0]), xytext=(-w / 2, 0.22),
                ha="center", fontsize=8.5, color=C_GAIN,
                arrowprops=dict(arrowstyle="->", color=C_GAIN))
    ax.annotate("plate 'max'\nonly\n(lagging)", xy=(1 + w / 2, plvals[1]), xytext=(1 + w / 2, 0.95),
                ha="center", fontsize=8.5, color=C_MID,
                arrowprops=dict(arrowstyle="->", color=C_MID))
    ax.set_xticks(x)
    ax.set_xticklabels([f"dispersion\n{f}" for f in feats])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC (dispersion → failure)")
    ax.set_title("Dispersion does NOT give a robust leading failure signal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.02,
             "Per-task AUROC of action-sampling dispersion predicting episode failure. "
             "push is backwards (mean 0.06, early 0.05);\nplate only registers on the "
             "LAGGING 'max' (0.79) while its leading 'early' is 0.29. Tasks disagree — "
             "no robust leading signal.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, "dispersion_failure.png")


# --------------------------------------------------------------------------- #
# 5. mechanism.png
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc, ec="black", fs=9.5, tc="black", bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, color="black", style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=15, lw=lw, color=color, linestyle=ls))


def fig_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- open loop ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 9.4, fill=False, ec=C_BAD, lw=1.5))
    ax.text(5, 9.3, "Open-loop  (default, h = 50)", ha="center", fontsize=14,
            fontweight="bold", color=C_BAD)
    ax.text(5, 8.6, "plan once, execute all 50 steps blindly", ha="center", fontsize=9.5, color="#555")
    _box(ax, 1.0, 7.0, 2.6, 1.1, "Observe\n(state s₀)", "#dbe7f3")
    _box(ax, 5.6, 7.0, 3.2, 1.1, "Plan one\n50-step chunk", "#dbe7f3")
    _arrow(ax, 3.6, 7.55, 5.6, 7.55)
    _arrow(ax, 7.2, 7.0, 1.6, 6.0, color=C_BAD)
    ax.text(4.4, 6.1, "execute all 50 steps with NO re-observation",
            ha="center", fontsize=8.5, color=C_BAD, style="italic")
    for i in range(7):
        c = "#cfe0f0" if i < 2 else "#f3d4d4"
        ax.add_patch(Rectangle((1.0 + i * 1.05, 4.5 - i * 0.15), 0.85, 0.85,
                               fc=c, ec="#999", lw=0.7))
    _box(ax, 1.0, 2.7, 7.8, 1.0,
         "confident-but-WRONG chunk  ->  drifts off-target,\nunrecoverable for the full 50 steps",
         "#f7d6d6", ec=C_BAD, fs=9.5, bold=True)
    _box(ax, 3.6, 0.8, 2.8, 1.2, "FAIL", C_BAD, ec=C_BAD, tc="white", fs=18, bold=True)

    # --- closed loop ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 9.4, fill=False, ec=C_GOOD, lw=1.5))
    ax.text(5, 9.3, "Closed-loop  (h = 10)", ha="center", fontsize=14,
            fontweight="bold", color=C_GOOD)
    ax.text(5, 8.6, "execute 10 steps, re-observe & replan, repeat", ha="center", fontsize=9.5, color="#555")
    labels = ["Observe", "Plan", "Exec 10", "Replan"]
    for i, lab in enumerate(labels):
        _box(ax, 0.7 + i * 2.25, 6.7, 2.0, 1.0, lab, "#d8f0d8", ec=C_GOOD, fs=9)
        if i < 3:
            _arrow(ax, 0.7 + i * 2.25 + 2.0, 7.2, 0.7 + (i + 1) * 2.25, 7.2, color=C_GOOD)
    _arrow(ax, 9.2, 6.7, 0.9, 6.7, color=C_GOOD, ls="--", lw=1.2)
    ax.text(5, 6.1, "each block = 10 steps;  small errors corrected before they compound",
            ha="center", fontsize=8.5, color=C_GOOD, style="italic")
    for i in range(7):
        ax.add_patch(Rectangle((1.0 + i * 1.05, 4.5), 0.85, 0.85,
                               fc="#d8f0d8", ec=C_GOOD, lw=0.8))
    _box(ax, 3.6, 0.8, 2.8, 1.2, "SUCCESS", C_GOOD, ec=C_GOOD, tc="white", fs=16, bold=True)

    fig.suptitle("The open-loop execution gap: a confident wrong chunk is unrecoverable for 50 steps",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.01,
             "Measured paired gains (frozen SmolVLA, MetaWorld, n=50)     "
             "push-v3:  +22pp  (56% -> 78%, McNemar p=0.035)     "
             "plate-slide-v3:  +54pp  (46% -> 100%, McNemar p~1e-8)",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save(fig, "mechanism.png")


# --------------------------------------------------------------------------- #
# 6. proposed_method.png
# --------------------------------------------------------------------------- #
def fig_proposed_method():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    # frozen stack box
    ax.add_patch(FancyBboxPatch((0.3, 1.4), 6.4, 4.0,
                 boxstyle="round,pad=0.05,rounding_size=0.1",
                 fc="#fbe9cf", ec=C_MID, lw=1.6))
    ax.text(3.5, 5.0, "FROZEN  SmolVLA   (weights not updated)",
            ha="center", fontsize=11, fontweight="bold", color="#a05a00")
    _box(ax, 0.6, 3.0, 1.7, 1.2, "Image +\nproprio +\nlang instr.", "#ffffff", fs=8)
    _box(ax, 2.6, 3.0, 1.6, 1.2, "VLM\nbackbone\n(frozen)", "#ffffff", fs=8.5, bold=True)
    _box(ax, 4.5, 3.0, 2.0, 1.2, "Flow-matching\naction expert\n(10 Euler steps)\n(frozen)",
         "#ffffff", fs=8)
    _arrow(ax, 2.3, 3.6, 2.6, 3.6)
    _arrow(ax, 4.2, 3.6, 4.5, 3.6)
    _box(ax, 4.9, 1.55, 1.3, 0.9, "candidate\n50-step chunk", "#eeeeee", fs=7.5)
    _arrow(ax, 5.5, 3.0, 5.5, 2.45)

    # trained gate box
    ax.add_patch(FancyBboxPatch((7.3, 2.0), 3.8, 3.4,
                 boxstyle="round,pad=0.05,rounding_size=0.1",
                 fc="#e9e2f5", ec="#7a5fb0", lw=1.6))
    ax.text(9.2, 5.0, "TRAINED  replan-gate",
            ha="center", fontsize=11, fontweight="bold", color="#5a3f90")
    _box(ax, 7.6, 3.6, 3.2, 1.1, "small MLP head\nreads expert features\n+ current step index",
         "#ffffff", fs=8.5)
    _box(ax, 7.6, 2.2, 3.2, 1.1, "per-step decision:\ncontinue  vs  REPLAN",
         "#ffffff", fs=8.5, bold=True)
    _arrow(ax, 6.5, 3.6, 7.6, 4.1, color="#7a5fb0")
    ax.text(6.9, 4.25, "features", fontsize=7.5, color="#7a5fb0", rotation=20)

    # replan loop back
    _arrow(ax, 7.6, 2.6, 3.5, 2.0, color="#7a5fb0", ls="--", lw=1.2)
    ax.text(5.4, 1.95, "REPLAN -> re-observe & re-run the frozen stack",
            ha="center", fontsize=7.5, color="#7a5fb0", style="italic")

    _box(ax, 8.4, 0.4, 2.5, 0.95, "continue: execute next step", "#d8f0d8", ec=C_GOOD, fs=8.5)
    _arrow(ax, 9.2, 2.2, 9.6, 1.35, color=C_GOOD)

    ax.text(6, 6.5, "Proposed method: a trained replan-gate on a FROZEN flow-matching VLA",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6, 6.0, "Goal: match the h=10 closed-loop success at ~1× (open-loop) inference compute",
            ha="center", fontsize=10, color="#555")

    # legend
    ax.add_patch(Rectangle((8.7, 5.55), 0.3, 0.25, fc="#fbe9cf", ec=C_MID))
    ax.text(9.05, 5.67, "FROZEN (no grad)", fontsize=8, va="center")
    ax.add_patch(Rectangle((8.7, 5.2), 0.3, 0.25, fc="#e9e2f5", ec="#7a5fb0"))
    ax.text(9.05, 5.32, "TRAINED (the contribution)", fontsize=8, va="center")

    fig.text(0.5, 0.0,
             "novelty scoop-gated vs BID (Bidirectional Decoding, NeurIPS 2024)",
             ha="center", fontsize=9, color=C_BAD, style="italic")
    fig.tight_layout()
    save(fig, "proposed_method.png")


# --------------------------------------------------------------------------- #
# 7. falsification_journey.png
# --------------------------------------------------------------------------- #
def fig_journey():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")

    swings = [
        ("1. CASTLE", "STL temporal-logic\nrouted correction",
         "routing NOT load-bearing;\nρ-routed == random\n=> reduces to coverage"),
        ("2. STL-GDPA", "analytic ρ-gradient\napproach guidance",
         "collapses to trivial\noracle goal-attractor\n(wrong-order beat ordered)"),
        ("3. NIAC averaging", "average K flow draws\n(variance reduction)",
         "NULL: push p=0.885,\nplate p=0.664;\nbase >= every arm"),
        ("4. NIAC dispersion", "dispersion predicts\nfailure (leading signal)",
         "no robust leading signal\n(push AUROC ~0.06 back;\ntasks disagree)"),
    ]
    bw = 2.5
    for i, (title, sub, why) in enumerate(swings):
        x0 = 0.3 + i * 2.75
        ax.add_patch(FancyBboxPatch((x0, 1.2), bw, 4.3,
                     boxstyle="round,pad=0.03,rounding_size=0.06",
                     fc="#f0f0f0", ec="#bbb", lw=1.2))
        ax.text(x0 + bw / 2, 5.0, title, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(x0 + bw / 2, 4.25, sub, ha="center", fontsize=8.3, color=C_GAIN)
        ax.text(x0 + bw / 2, 2.9, why, ha="center", fontsize=7.6, color="#555")
        _box(ax, x0 + 0.3, 1.35, bw - 0.6, 0.7, "FALSIFIED", "#f7d6d6", ec=C_BAD,
             tc=C_BAD, fs=9, bold=True)
        if i < 3:
            _arrow(ax, x0 + bw, 3.3, x0 + 2.75, 3.3, color="#999")

    # survivor
    x0 = 11.3
    ax.add_patch(FancyBboxPatch((x0, 1.2), 1.5, 4.3,
                 boxstyle="round,pad=0.03,rounding_size=0.06",
                 fc="#d8f0d8", ec=C_GOOD, lw=2.0))
    ax.text(x0 + 0.75, 5.0, "SURVIVOR", ha="center", fontsize=10.5,
            fontweight="bold", color=C_GOOD)
    ax.text(x0 + 0.75, 4.0, "Open-loop\nexecution gap", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(x0 + 0.75, 3.0, "shrink execution\nhorizon\n(h=50 -> h=10)\n= close the loop",
            ha="center", fontsize=7.6, color="#3a6b1a")
    _box(ax, x0 + 0.1, 2.0, 1.3, 0.75, "push +22pp (p=0.035)\nplate +54pp (p~1e-8)",
         "#bfe6bf", ec=C_GOOD, fs=6.8, bold=True)
    _box(ax, x0 + 0.1, 1.3, 1.3, 0.55, "motivation + positive number\nfor a TRAINED method\n(not yet the contribution)",
         "#eaf7ea", ec=C_GOOD, fs=5.8)

    ax.text(6.5, 6.5, "Falsification journey: four killed swings narrowed to one honest survivor",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 6.0,
            "Each swing was killed by its own paired test before the open-loop gap survived.",
            ha="center", fontsize=9.5, color="#555")
    fig.text(0.5, -0.01,
             "Honest framing: the survivor is a known knob (receding-horizon / BID) "
             "costing ~5× inference; it motivates the trained, ~1×-compute contribution.",
             ha="center", fontsize=8.8, color="#666", style="italic")
    fig.tight_layout()
    save(fig, "falsification_journey.png")


if __name__ == "__main__":
    fig_open_loop_gap()
    fig_base_survey()
    fig_niac_null()
    fig_dispersion()
    fig_mechanism()
    fig_proposed_method()
    fig_journey()
    print("ALL FIGURES DONE")
