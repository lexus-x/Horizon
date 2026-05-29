#!/usr/bin/env python
"""Horizon figure generator.

Reads the REAL result JSONs produced by the faithful smolvla_metaworld harness
and renders the four publication figures used by the Horizon README.

Run with the lerobot env python:
    /home/user/miniconda3/envs/lerobot/bin/python make_figures.py

All figures land in /home/user/Desktop/vla_projects/Horizon/figures/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS = "/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/results"
NIAC = os.path.join(RESULTS, "niac_probe")
FIGDIR = "/home/user/Desktop/vla_projects/Horizon/figures"
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Small palette (colour-blind friendly-ish, consistent across figures)
# ---------------------------------------------------------------------------
C_PUSH = "#2c6fbb"   # blue
C_PLATE = "#d1622b"  # orange
C_BASE = "#9aa0a6"   # grey (base / chance)
C_MID = "#2e8b57"    # mid-band green
C_OUT = "#b03a2e"    # outlier red
C_NULL = "#7d8a96"   # null grey-blue


def _load(path):
    with open(path) as f:
        return json.load(f)


def _pc(path):
    """pc_success from a {'result': {...}} eval JSON."""
    return _load(path)["result"]["pc_success"]


# ===========================================================================
# 1. open_loop_gap.png  -- the hero figure
# ===========================================================================
def fig_open_loop_gap():
    push = {
        50: _pc(os.path.join(NIAC, "horizon_push_h50.json")),
        25: _pc(os.path.join(NIAC, "horizon_push_h25.json")),
        10: _pc(os.path.join(NIAC, "horizon_push_h10.json")),
    }
    plate = {
        50: _pc(os.path.join(NIAC, "horizon_plate_h50.json")),
        10: _pc(os.path.join(NIAC, "horizon_plate_h10.json")),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)

    # ----- push panel (3 horizons) -----
    ax = axes[0]
    hs = [50, 25, 10]
    xs = range(len(hs))
    vals = [push[h] for h in hs]
    bars = ax.bar(xs, vals, width=0.62, color=C_PUSH, edgecolor="black", linewidth=0.8)
    # gradient cue: lighter at h50 -> solid at h10
    for b, alpha in zip(bars, [0.45, 0.7, 1.0]):
        b.set_alpha(alpha)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=15, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"h={h}" for h in hs], fontsize=13)
    ax.set_title("push-v3", fontsize=16, fontweight="bold")
    ax.set_ylabel("Success rate (%)", fontsize=14)
    ax.set_xlabel("Execution horizon (steps before replan)", fontsize=12)
    # significance bracket h50 -> h10
    _sig_bracket(ax, 0, 2, max(vals) + 7,
                 "McNemar h50→h10  p = 0.035\n(helped 17 / hurt 6)")

    # ----- plate panel (2 horizons) -----
    ax = axes[1]
    hs = [50, 10]
    xs = range(len(hs))
    vals = [plate[h] for h in hs]
    bars = ax.bar(xs, vals, width=0.5, color=C_PLATE, edgecolor="black", linewidth=0.8)
    for b, alpha in zip(bars, [0.5, 1.0]):
        b.set_alpha(alpha)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=15, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"h={h}" for h in hs], fontsize=13)
    ax.set_title("plate-slide-v3", fontsize=16, fontweight="bold")
    ax.set_xlabel("Execution horizon (steps before replan)", fontsize=12)
    _sig_bracket(ax, 0, 1, max(vals) + 7,
                 "McNemar h50→h10  p ≈ 1e-8\n(helped 27 / hurt 0)")

    for ax in axes:
        ax.set_ylim(0, 118)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(50, color="grey", lw=0.8, ls=":", alpha=0.5)

    fig.suptitle("The open-loop execution gap", fontsize=21, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Frozen SmolVLA executes 50-step chunks open-loop. Shrinking the execution horizon "
             "(replan more often → closed loop)\nmonotonically lifts success. n=50 episodes per arm, "
             "paired seed pool (seeds 1000–1049). Costs ~5× inference at h=10.",
             ha="center", fontsize=10.5, color="#333333")
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    out = os.path.join(FIGDIR, "open_loop_gap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _sig_bracket(ax, x1, x2, y, label):
    """Draw a significance bracket between x1 and x2 at height y."""
    h = 2.0
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, color="black")
    ax.text((x1 + x2) / 2.0, y + h + 1.0, label, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#111111")


# ===========================================================================
# 2. base_survey_midband.png
# ===========================================================================
def fig_base_survey():
    survey = _load(os.path.join(RESULTS, "base_survey_metaworld.json"))["survey"]
    # sort ascending by success for a clean ramp
    survey = sorted(survey, key=lambda r: r["pooled_success_pct"])

    tasks = [r["task"].replace("-v3", "") for r in survey]
    vals = [r["pooled_success_pct"] for r in survey]
    los = [max(0.0, r["pooled_success_pct"] - r["wilson95_lo"]) for r in survey]
    his = [max(0.0, r["wilson95_hi"] - r["pooled_success_pct"]) for r in survey]
    midband = [r["midband"] for r in survey]

    fig, ax = plt.subplots(figsize=(13, 6.8))
    xs = list(range(len(tasks)))

    # shaded mid-band region 30-75%
    ax.axhspan(30, 75, color=C_MID, alpha=0.10, zorder=0)
    ax.axhline(30, color=C_MID, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(75, color=C_MID, lw=1.0, ls="--", alpha=0.6)
    ax.text(len(tasks) - 0.4, 76.5, "mid-band ceiling 75%", ha="right",
            fontsize=9.5, color=C_MID)
    ax.text(len(tasks) - 0.4, 25.0, "mid-band floor 30%", ha="right",
            fontsize=9.5, color=C_MID)

    colors = []
    for r, mb in zip(survey, midband):
        if r["task"] == "reach-v3":
            colors.append(C_OUT)          # floored outlier highlight
        elif mb:
            colors.append(C_MID)          # mid-band (the usable tasks)
        else:
            colors.append(C_BASE)         # floored/ceilinged

    bars = ax.bar(xs, vals, width=0.7, color=colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor="black",
                elinewidth=1.1, capsize=3.5, alpha=0.7)

    for x, v in zip(xs, vals):
        ax.text(x, v + max(his) + 0.6, f"{v:.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")

    # mark reach-v3 as a floored OUTLIER
    reach_ix = tasks.index("reach")
    ax.annotate("floored outlier\n(every prior “dead” verdict used this task)",
                xy=(reach_ix, vals[reach_ix]),
                xytext=(reach_ix + 1.6, vals[reach_ix] + 28),
                fontsize=10.5, fontweight="bold", color=C_OUT,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.6))

    ax.set_xticks(list(xs))
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=11)
    ax.set_ylabel("Base success rate (%)", fontsize=14)
    ax.set_ylim(0, 112)
    ax.set_title("Frozen SmolVLA base survey on MetaWorld — the measurable mid-band",
                 fontsize=17, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    legend = [
        Patch(facecolor=C_MID, edgecolor="black", label="mid-band 30–75% (usable for repair)"),
        Patch(facecolor=C_BASE, edgecolor="black", label="floored / ceilinged"),
        Patch(facecolor=C_OUT, edgecolor="black", label="reach-v3 (floored outlier)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10.5, framealpha=0.9)

    fig.text(0.5, 0.005,
             "Pooled over 3 seeds (n=90/task, Wilson 95% CIs). Mid-band tasks "
             "(drawer-open, push, peg-insert, window-open, plate-slide) are where a train-free "
             "repair has headroom.",
             ha="center", fontsize=10, color="#333333")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIGDIR, "base_survey_midband.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ===========================================================================
# 3. niac_null.png
# ===========================================================================
def fig_niac_null():
    arms = [
        ("base", "push_base.json", C_BASE),
        ("mean K=2", "push_mean_k2.json", C_PUSH),
        ("mean K=4", "push_mean_k4.json", C_PUSH),
        ("mean K=8", "push_mean_k8.json", C_PUSH),
        ("medoid K=4\n(consensus)", "push_consensus_k4.json", C_NULL),
    ]
    labels, vals, colors = [], [], []
    for lab, fn, col in arms:
        labels.append(lab)
        vals.append(_pc(os.path.join(NIAC, fn)))
        colors.append(col)

    base_val = vals[0]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    xs = range(len(labels))
    bars = ax.bar(xs, vals, width=0.66, color=colors, edgecolor="black", linewidth=0.8)

    # base reference line
    ax.axhline(base_val, color=C_OUT, lw=1.6, ls="--", alpha=0.85,
               label=f"base = {base_val:.0f}%")

    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    # highlight mean-K4 == medoid-K4 == 58 equivalence callout
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("push-v3 success rate (%)", fontsize=14)
    ax.set_ylim(0, max(vals) + 16)
    ax.set_title("NIAC test-time noise-averaging is NULL — averaging does not beat base",
                 fontsize=16, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=11)

    fig.text(0.5, 0.015,
             "Averaging K flow draws (mean) or picking the medoid (consensus) never exceeds base.\n"
             "Paired McNemar (matched 50-seed pool): push p = 0.885 — no variance-reduction "
             "mechanism; mean-K4 ≡ medoid-K4 ≡ 58%.",
             ha="center", fontsize=10, color="#333333")
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    out = os.path.join(FIGDIR, "niac_null.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ===========================================================================
# 4. dispersion_failure.png
# ===========================================================================
def fig_dispersion():
    push = _load(os.path.join(NIAC, "dispersion_push-v3.json"))["result"]
    plate = _load(os.path.join(NIAC, "dispersion_plate-slide-v3.json"))["result"]

    feats = ["mean", "max", "early"]
    push_auroc = [push["auroc_disp_to_failure"][f] for f in feats]
    plate_auroc = [plate["auroc_disp_to_failure"][f] for f in feats]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = range(len(feats))
    w = 0.36

    b1 = ax.bar([i - w / 2 for i in x], push_auroc, width=w,
                color=C_PUSH, edgecolor="black", linewidth=0.8, label="push-v3")
    b2 = ax.bar([i + w / 2 for i in x], plate_auroc, width=w,
                color=C_PLATE, edgecolor="black", linewidth=0.8, label="plate-slide-v3")

    for bars, vals in [(b1, push_auroc), (b2, plate_auroc)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    # chance line at 0.5
    ax.axhline(0.5, color="black", lw=1.4, ls="--", alpha=0.8)
    ax.text(len(feats) - 0.55, 0.515, "chance (0.5)", ha="right", fontsize=10, color="black")

    # annotate the two failure modes
    ax.annotate("backwards\n(< 0.5)", xy=(0 - w / 2, push_auroc[0]),
                xytext=(0 - w / 2, 0.18), ha="center", fontsize=9.5, color=C_PUSH,
                arrowprops=dict(arrowstyle="->", color=C_PUSH, lw=1.3))
    ax.annotate("plate 'max'\nonly\n(lagging)", xy=(1 + w / 2, plate_auroc[1]),
                xytext=(1 + w / 2 + 0.05, 0.93), ha="center", fontsize=9.5, color=C_PLATE,
                arrowprops=dict(arrowstyle="->", color=C_PLATE, lw=1.3))

    ax.set_xticks(list(x))
    ax.set_xticklabels(["dispersion\nmean", "dispersion\nmax", "dispersion\nearly (leading)"],
                       fontsize=12)
    ax.set_ylabel("AUROC (dispersion → failure)", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title("Dispersion does NOT give a robust leading failure signal",
                 fontsize=16, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=12)

    fig.text(0.5, 0.015,
             "Per-task AUROC of action-sampling dispersion predicting episode failure. push is "
             "backwards (mean 0.06, early 0.05);\nplate only registers on the LAGGING 'max' (0.79) "
             "while its leading 'early' is 0.29. Tasks disagree — no robust leading signal.",
             ha="center", fontsize=9.8, color="#333333")
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    out = os.path.join(FIGDIR, "dispersion_failure.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ===========================================================================
if __name__ == "__main__":
    outs = []
    outs.append(fig_open_loop_gap())
    outs.append(fig_base_survey())
    outs.append(fig_niac_null())
    outs.append(fig_dispersion())
    print("WROTE:")
    for o in outs:
        sz = os.path.getsize(o)
        print(f"  {o}  ({sz} bytes)")
