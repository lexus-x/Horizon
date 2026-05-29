#!/usr/bin/env python
"""Horizon project schematic diagrams.

Generates three publication-style schematic PNGs (no external data):
  - mechanism.png          : open-loop (h=50) commit-and-fail  vs  closed-loop (h=10) replan-and-correct
  - proposed_method.png    : frozen SmolVLA + trained replan-gate head, ~1x compute
  - falsification_journey.png : 4 falsified swings -> the open-loop-execution-gap survivor

Run with:
  /home/user/miniconda3/envs/lerobot/bin/python /home/user/Desktop/vla_projects/Horizon/make_diagrams.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Shared palette + style
# ----------------------------------------------------------------------------
FIG_DIR = "/home/user/Desktop/vla_projects/Horizon/figures"
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    "ink":     "#1f2933",   # near-black text
    "muted":   "#6b7280",   # secondary text
    "blue":    "#2b6cb0",   # process / neutral box
    "blue_bg": "#ebf2fb",
    "green":   "#2f855a",   # success
    "green_bg":"#e6f4ec",
    "red":     "#c53030",   # failure
    "red_bg":  "#fdecec",
    "amber":   "#b7791f",   # frozen / locked
    "amber_bg":"#fdf3e0",
    "violet":  "#6b46c1",   # trained / proposed
    "violet_bg":"#f0ebfa",
    "grey":    "#9aa3ad",   # falsified / dead
    "grey_bg": "#eef0f2",
    "line":    "#cbd2d9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": PALETTE["line"],
    "savefig.dpi": 200,
})


def box(ax, xy, w, h, text, facecolor, edgecolor, textcolor="#1f2933",
        fontsize=9.5, fontweight="normal", rounding=0.04, lw=1.6, zorder=2,
        ha="center", va="center"):
    """Draw a rounded box with centered (or aligned) text. xy = lower-left."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if ha == "center" else (x + 0.04 if ha == "left" else x + w - 0.04)
    ty = y + h / 2 if va == "center" else (y + h - 0.05 if va == "top" else y + 0.05)
    ax.text(tx, ty, text, ha=ha, va=va, fontsize=fontsize, color=textcolor,
            fontweight=fontweight, zorder=zorder + 1, wrap=True)
    return patch


def arrow(ax, start, end, color="#1f2933", lw=1.8, style="-|>", mut=14,
          connectionstyle="arc3,rad=0.0", zorder=3, ls="-"):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        color=color, connectionstyle=connectionstyle, zorder=zorder,
        linestyle=ls, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a


def clean_ax(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal", adjustable="box")


# ============================================================================
# 1) mechanism.png
# ============================================================================
def make_mechanism():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax in (axL, axR):
        clean_ax(ax, (0, 10), (0, 10))

    # ---------------- LEFT: open-loop (default h=50), FAILS --------------
    axL.add_patch(Rectangle((0.15, 0.2), 9.7, 9.6, facecolor="#ffffff",
                            edgecolor=PALETTE["red"], lw=1.4, zorder=0, alpha=0.6))
    axL.text(5.0, 9.45, "Open-loop  (default, h = 50)", ha="center", va="center",
             fontsize=14, fontweight="bold", color=PALETTE["red"])
    axL.text(5.0, 8.85, "plan once, execute all 50 steps blindly",
             ha="center", va="center", fontsize=10, color=PALETTE["muted"])

    box(axL, (1.4, 7.3), 2.4, 1.0, "Observe\n(state $s_0$)",
        PALETTE["blue_bg"], PALETTE["blue"], fontsize=10)
    box(axL, (6.0, 7.3), 2.6, 1.0, "Plan one\n50-step chunk",
        PALETTE["blue_bg"], PALETTE["blue"], fontsize=10)
    arrow(axL, (3.8, 7.8), (6.0, 7.8), color=PALETTE["ink"])

    # the chunk executed blindly: a row of tiles, drifting off
    n = 10
    x0, xw, gap = 1.0, 0.72, 0.06
    base_y = 5.3
    drift = 0.0
    axL.text(5.0, 6.55, "execute all 50 steps with NO re-observation",
             ha="center", va="center", fontsize=9.5, color=PALETTE["red"],
             fontstyle="italic")
    for i in range(n):
        # progressively drift downward to depict open-loop drift
        drift = -0.16 * (i ** 1.25) / 2.0
        col = PALETTE["blue_bg"] if i < 3 else PALETTE["red_bg"]
        edg = PALETTE["blue"] if i < 3 else PALETTE["red"]
        xx = x0 + i * (xw + gap)
        box(axL, (xx, base_y + drift), xw, 0.6, "", col, edg, rounding=0.02, lw=1.2)
    # connect plan -> chunk
    arrow(axL, (7.3, 7.3), (1.3 + 0.36, base_y + 0.6 + 0.05),
          color=PALETTE["ink"], connectionstyle="arc3,rad=-0.25")
    axL.text(1.0, base_y + 0.95, "step 1", ha="left", va="center",
             fontsize=8, color=PALETTE["muted"])
    axL.text(x0 + (n - 1) * (xw + gap) + xw, base_y - 1.5, "step 50",
             ha="right", va="center", fontsize=8, color=PALETTE["muted"])

    # commit annotation
    box(axL, (1.0, 3.0), 8.0, 1.05,
        "Confident-but-WRONG chunk  ->  drifts off-target,\nunrecoverable for the full 50 steps",
        PALETTE["red_bg"], PALETTE["red"], textcolor=PALETTE["red"],
        fontsize=10, fontweight="bold")

    box(axL, (3.3, 1.0), 3.4, 1.1, "FAIL", PALETTE["red"], PALETTE["red"],
        textcolor="white", fontsize=16, fontweight="bold")
    arrow(axL, (5.0, 3.0), (5.0, 2.1), color=PALETTE["red"], lw=2.2)

    # ---------------- RIGHT: closed-loop (h=10), SUCCEEDS ---------------
    axR.add_patch(Rectangle((0.15, 0.2), 9.7, 9.6, facecolor="#ffffff",
                            edgecolor=PALETTE["green"], lw=1.4, zorder=0, alpha=0.6))
    axR.text(5.0, 9.45, "Closed-loop  (h = 10)", ha="center", va="center",
             fontsize=14, fontweight="bold", color=PALETTE["green"])
    axR.text(5.0, 8.85, "execute 10 steps, re-observe & replan, repeat",
             ha="center", va="center", fontsize=10, color=PALETTE["muted"])

    # cycle of 4 short bursts that stay on track
    cx = [1.6, 3.85, 6.1, 8.35]
    cy = 6.6
    labels = ["Observe", "Plan", "Exec 10", "Replan"]
    for i, (x, lab) in enumerate(zip(cx, labels)):
        fb = PALETTE["green_bg"]
        ec = PALETTE["green"]
        box(axR, (x - 0.95, cy - 0.5), 1.9, 1.0, lab, fb, ec, fontsize=9.5)
        if i < 3:
            arrow(axR, (x + 0.95, cy), (cx[i + 1] - 0.95, cy), color=PALETTE["ink"])
    # loop-back arrow Replan -> Observe
    arrow(axR, (8.35, cy - 0.5), (1.6, cy - 0.5), color=PALETTE["green"],
          lw=1.8, connectionstyle="arc3,rad=0.32", ls=(0, (4, 2)))
    axR.text(5.0, 4.95, "each block = 10 steps;  small errors corrected before they compound",
             ha="center", va="center", fontsize=9.5, color=PALETTE["green"],
             fontstyle="italic")

    # on-track tiles
    n2 = 10
    base_y2 = 3.55
    for i in range(n2):
        # correction bumps: stays near baseline
        wig = 0.10 * ((i % 3) - 1)
        xx = x0 + i * (xw + gap)
        box(axR, (xx, base_y2 + wig), xw, 0.6, "",
            PALETTE["green_bg"], PALETTE["green"], rounding=0.02, lw=1.2)

    box(axR, (3.3, 1.0), 3.4, 1.1, "SUCCESS", PALETTE["green"], PALETTE["green"],
        textcolor="white", fontsize=15, fontweight="bold")
    arrow(axR, (5.0, 3.4), (5.0, 2.1), color=PALETTE["green"], lw=2.2)

    fig.suptitle("The open-loop execution gap: a confident wrong chunk is unrecoverable for 50 steps",
                 fontsize=13, fontweight="bold", color=PALETTE["ink"], y=0.985)
    fig.text(0.5, 0.045,
             "Measured paired gains (frozen SmolVLA, MetaWorld, n=50)",
             ha="center", va="center", fontsize=10.5, color=PALETTE["muted"])
    fig.text(0.5, 0.012,
             "push-v3:  +22pp  (56% -> 78%, McNemar p=0.035)        "
             "plate-slide-v3:  +54pp  (46% -> 100%, McNemar p~1e-8)",
             ha="center", va="center", fontsize=11, color=PALETTE["ink"],
             fontweight="bold")
    fig.tight_layout(rect=[0, 0.085, 1, 0.95])
    out = os.path.join(FIG_DIR, "mechanism.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ============================================================================
# 2) proposed_method.png
# ============================================================================
def make_proposed():
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    clean_ax(ax, (0, 14), (0, 8))

    ax.text(7.0, 7.6, "Proposed method: a trained replan-gate on a FROZEN flow-matching VLA",
            ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    ax.text(7.0, 7.08, "Goal: match the h=10 closed-loop success at ~1x (open-loop) inference compute",
            ha="center", va="center", fontsize=10.5, color=PALETTE["muted"])

    # ---- FROZEN SmolVLA container (locked) ----
    frozen = FancyBboxPatch((0.4, 1.6), 8.4, 4.7,
                            boxstyle="round,pad=0.0,rounding_size=0.12",
                            linewidth=2.0, edgecolor=PALETTE["amber"],
                            facecolor=PALETTE["amber_bg"], zorder=1)
    ax.add_patch(frozen)
    # a small lock glyph drawn from primitives (left of the label)
    lx, ly = 0.72, 5.95
    ax.add_patch(Rectangle((lx, ly - 0.14), 0.26, 0.20, facecolor=PALETTE["amber"],
                           edgecolor=PALETTE["amber"], zorder=3))
    ax.add_patch(plt.matplotlib.patches.Arc((lx + 0.13, ly + 0.06), 0.20, 0.24,
                 theta1=0, theta2=180, lw=2.0, edgecolor=PALETTE["amber"], zorder=3))
    ax.text(1.12, 5.95, "FROZEN  SmolVLA   (weights not updated)",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color=PALETTE["amber"])

    # inputs
    box(ax, (0.9, 3.6), 2.0, 1.1, "Image +\nproprio +\nlang instr.",
        "#ffffff", PALETTE["amber"], fontsize=9.5)
    # VLM backbone
    box(ax, (3.4, 3.55), 2.3, 1.2, "VLM\nbackbone\n(frozen)",
        "#ffffff", PALETTE["amber"], fontsize=10, fontweight="bold")
    # flow-matching action expert
    box(ax, (6.1, 3.4), 2.4, 1.5, "Flow-matching\naction expert\n(10 Euler steps)\n(frozen)",
        "#ffffff", PALETTE["amber"], fontsize=9.5, fontweight="bold")
    arrow(ax, (2.9, 4.15), (3.4, 4.15), color=PALETTE["ink"])
    arrow(ax, (5.7, 4.15), (6.1, 4.15), color=PALETTE["ink"])

    # expert output: 50-step chunk
    box(ax, (6.1, 2.05), 2.4, 0.85, "candidate\n50-step chunk",
        "#ffffff", PALETTE["amber"], fontsize=9)
    arrow(ax, (7.3, 3.4), (7.3, 2.9), color=PALETTE["ink"])

    # ---- TRAINED replan-gate head ----
    gate = FancyBboxPatch((9.6, 2.6), 3.9, 3.1,
                          boxstyle="round,pad=0.0,rounding_size=0.12",
                          linewidth=2.2, edgecolor=PALETTE["violet"],
                          facecolor=PALETTE["violet_bg"], zorder=1)
    ax.add_patch(gate)
    ax.text(11.55, 5.4, "TRAINED  replan-gate", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=PALETTE["violet"])
    box(ax, (9.9, 3.95), 3.3, 1.1,
        "small MLP head\nreads expert features\n+ current step index",
        "#ffffff", PALETTE["violet"], fontsize=9.5)
    box(ax, (9.9, 2.85), 3.3, 0.95,
        "per-step decision:\ncontinue  vs  REPLAN",
        "#ffffff", PALETTE["violet"], fontsize=9.5, fontweight="bold")

    # tap expert features into gate
    arrow(ax, (8.5, 4.15), (9.9, 4.5), color=PALETTE["violet"], lw=1.8,
          connectionstyle="arc3,rad=-0.12")
    ax.text(9.05, 4.78, "features", ha="center", va="center", fontsize=8.5,
            color=PALETTE["violet"], fontstyle="italic")

    # gate decisions feed back
    # continue -> execute next step
    box(ax, (10.1, 0.55), 2.9, 0.95, "continue: execute next step",
        PALETTE["green_bg"], PALETTE["green"], fontsize=9, textcolor=PALETTE["green"])
    arrow(ax, (11.0, 2.85), (11.0, 1.5), color=PALETTE["green"], lw=1.8)
    # replan -> re-run frozen stack (loop back)
    arrow(ax, (9.9, 3.0), (4.55, 2.05), color=PALETTE["violet"], lw=1.8,
          connectionstyle="arc3,rad=0.28", ls=(0, (4, 2)))
    ax.text(6.7, 1.35, "REPLAN -> re-observe & re-run the frozen stack",
            ha="center", va="center", fontsize=9, color=PALETTE["violet"],
            fontstyle="italic")

    # compute note
    box(ax, (0.9, 0.45), 6.2, 0.95,
        "Replans only when the gate fires  ->  far fewer than every-10-step;\n"
        "targets h=10 success at ~1x open-loop compute",
        PALETTE["blue_bg"], PALETTE["blue"], fontsize=9.5, textcolor=PALETTE["blue"])

    # legend frozen vs trainable
    leg = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["amber_bg"],
               markeredgecolor=PALETTE["amber"], markersize=14, label="FROZEN (no grad)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["violet_bg"],
               markeredgecolor=PALETTE["violet"], markersize=14, label="TRAINED (the contribution)"),
    ]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.998, 0.86),
              frameon=True, fontsize=9.5, handletextpad=0.6)

    # scoop-gate note
    ax.text(13.55, 0.25,
            "novelty scoop-gated vs BID (Bidirectional Decoding, NeurIPS 2024)",
            ha="right", va="center", fontsize=9, color=PALETTE["red"],
            fontstyle="italic")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "proposed_method.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ============================================================================
# 3) falsification_journey.png
# ============================================================================
def make_journey():
    W = 18.0
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    clean_ax(ax, (0, W), (0, 10))

    ax.text(W / 2, 9.5, "Falsification journey: four killed swings narrowed to one honest survivor",
            ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    ax.text(W / 2, 8.95,
            "Each swing was killed by its own paired test before the open-loop gap survived.",
            ha="center", va="center", fontsize=10.5, color=PALETTE["muted"])

    # four falsified stages as a narrowing funnel (decreasing height)
    stages = [
        ("1. CASTLE",
         "STL temporal-logic\nrouted correction",
         "routing NOT load-bearing:\nrho-routed == random\n=> reduces to coverage",
         "FALSIFIED"),
        ("2. STL-GDPA",
         "analytic rho-gradient\napproach guidance",
         "collapses to trivial\noracle goal-attractor\n(wrong-order beat ordered)",
         "FALSIFIED"),
        ("3. NIAC averaging",
         "average K flow draws\n(variance reduction)",
         "NULL: push p=0.885,\nplate p=0.664;\nbase >= every arm",
         "FALSIFIED"),
        ("4. NIAC dispersion",
         "dispersion predicts\nfailure (leading signal)",
         "no robust leading signal\n(push AUROC ~0.06 back;\ntasks disagree)",
         "FALSIFIED"),
    ]

    x0 = 0.55
    w = 2.75
    gap = 0.32
    h = 5.3
    ymid = 4.5
    y = ymid - h / 2
    for i, (title, idea, kill, tag) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(ax, (x, y), w, h, "", PALETTE["grey_bg"], PALETTE["grey"],
            rounding=0.06, lw=1.6)
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=PALETTE["ink"])
        ax.text(x + w / 2, y + h - 1.5, idea, ha="center", va="center",
                fontsize=9, color=PALETTE["blue"])
        ax.text(x + w / 2, y + h - 3.0, kill, ha="center", va="center",
                fontsize=8.5, color=PALETTE["muted"])
        # red FALSIFIED stamp at bottom
        box(ax, (x + 0.45, y + 0.28), w - 0.9, 0.55, tag,
            PALETTE["red_bg"], PALETTE["red"], textcolor=PALETTE["red"],
            fontsize=9, fontweight="bold", rounding=0.05)
        # arrow to next
        if i < len(stages) - 1:
            arrow(ax, (x + w, ymid), (x + w + gap, ymid),
                  color=PALETTE["grey"], lw=2.0)

    # survivor box on the right (green, full height)
    sx = x0 + 4 * (w + gap) + 0.25
    sw = W - sx - 0.45
    sy, sh = 1.55, 5.9
    surv = FancyBboxPatch((sx, sy), sw, sh,
                          boxstyle="round,pad=0.0,rounding_size=0.10",
                          linewidth=2.6, edgecolor=PALETTE["green"],
                          facecolor=PALETTE["green_bg"], zorder=2)
    ax.add_patch(surv)
    arrow(ax, (sx - 0.55, ymid), (sx, ymid), color=PALETTE["green"], lw=2.4)

    ax.text(sx + sw / 2, sy + sh - 0.55, "SURVIVOR", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=PALETTE["green"])
    ax.text(sx + sw / 2, sy + sh - 1.5, "Open-loop\nexecution gap",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=PALETTE["ink"])
    ax.text(sx + sw / 2, sy + sh - 2.95,
            "shrink execution horizon\n(h=50 -> h=10) = close the loop",
            ha="center", va="center", fontsize=9.5, color=PALETTE["green"])
    box(ax, (sx + 0.35, sy + 1.35), sw - 0.7, 1.05,
        "push  +22pp  (p=0.035)\nplate +54pp  (p~1e-8)",
        "#ffffff", PALETTE["green"], textcolor=PALETTE["green"],
        fontsize=10.5, fontweight="bold")
    box(ax, (sx + 0.35, sy + 0.2), sw - 0.7, 0.95,
        "motivation + positive number\nfor a TRAINED method\n(not yet the contribution)",
        PALETTE["green_bg"], PALETTE["green"], textcolor=PALETTE["green"],
        fontsize=8.5, rounding=0.05)

    fig.text(0.5, 0.02,
             "Honest framing: the survivor is a known knob (receding-horizon / BID) costing ~5x inference; "
             "it motivates the trained, ~1x-compute contribution.",
             ha="center", va="center", fontsize=9.5, color=PALETTE["muted"],
             fontstyle="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG_DIR, "falsification_journey.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    outs = [make_mechanism(), make_proposed(), make_journey()]
    for o in outs:
        sz = os.path.getsize(o) if os.path.exists(o) else -1
        print(f"WROTE {o}  ({sz} bytes)")
