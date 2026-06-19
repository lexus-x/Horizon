#!/usr/bin/env python
"""Paper-style architecture diagram for Horizon v2 (matplotlib render).
Frozen SmolVLA + the four deployment regimes (open-loop / closed-loop / distilled
student / A2C2 head), with compute cost and shape-vs-reactive outcome tags.
A compile-ready TikZ version is in deck/architecture.tex."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

BLUE="#1a73e8"; GREEN="#34a853"; GREY="#9aa0a6"; PURPLE="#9c27b0"; GOLD="#f4b400"; INK="#202124"
FROZEN="#e8f0fe"; TRAIN="#fce8e6"

fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc="white", ec=INK, lw=1.4, fs=10.5, weight="normal", tc=INK, style="round,pad=0.02,rounding_size=2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc, zorder=3)

def arrow(x1,y1,x2,y2, color=INK, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, zorder=1, shrinkA=2, shrinkB=2))

# ---------- top: frozen base pipeline ----------
ax.text(50, 97, "Horizon: frozen flow-matching VLA — four ways to deploy the action chunk",
        ha="center", fontsize=14, weight="bold", color=INK)

box(2, 78, 15, 12, "MetaWorld obs\nimage + proprio state", fc="white", fs=10)
# frozen base group
ax.add_patch(FancyBboxPatch((20, 74.5), 42, 19, boxstyle="round,pad=0.02,rounding_size=2",
                            fc=FROZEN, ec=BLUE, lw=1.6, zorder=1))
ax.text(41, 91, "FROZEN  SmolVLA   (no gradient)", ha="center", fontsize=10.5, weight="bold", color=BLUE)
box(23, 78, 16, 10, "VLM backbone\n(image+lang+state)", fc="white", ec=BLUE, fs=9.5)
box(43, 78, 16, 10, "flow-matching\naction expert\n(10 Euler steps)", fc="white", ec=BLUE, fs=9.5)
box(66, 78, 16, 12, "50-step\naction chunk\n$a_{1:50}$", fc="white", ec=INK, fs=10)
arrow(17, 84, 23, 83); arrow(39, 83, 43, 83); arrow(59, 83, 66, 84)
ax.text(63.5, 71.5, "the chunk is consumed differently by each regime below ↓",
        ha="center", fontsize=9, style="italic", color="#555")

# ---------- four regimes ----------
yb = 40; hb = 24; wb = 21.5; gap = 3.2
xs = [2 + i*(wb+gap) for i in range(4)]

def regime(x, title, mech, compute, tag, tagcol, edge, trained=False, result=""):
    fc = TRAIN if trained else "white"
    ax.add_patch(FancyBboxPatch((x, yb), wb, hb, boxstyle="round,pad=0.02,rounding_size=2",
                                fc=fc, ec=edge, lw=2.0, zorder=2))
    ax.text(x+wb/2, yb+hb-2.4, title, ha="center", fontsize=11, weight="bold", color=edge)
    ax.text(x+wb/2, yb+hb-8.2, mech, ha="center", fontsize=8.6, color=INK)
    ax.text(x+wb/2, yb+5.2, compute, ha="center", fontsize=9.2, weight="bold", color=INK)
    ax.text(x+wb/2, yb+1.8, ("TRAINED head/expert" if trained else "train-free"),
            ha="center", fontsize=8, style="italic", color=("#b00020" if trained else "#137333"))
    # result tag pill
    ax.add_patch(FancyBboxPatch((x+wb/2-9, yb+hb+1.5), 18, 4.2, boxstyle="round,pad=0.02,rounding_size=2",
                                fc=tagcol, ec="none", zorder=3))
    ax.text(x+wb/2, yb+hb+3.6, tag, ha="center", fontsize=8.8, weight="bold", color="white", zorder=4)
    arrow(74, 78, x+wb/2, yb+hb+6, color="#bbb", lw=1.2)

regime(xs[0], "Open-loop  h=50", "commit all 50 steps\nno re-observation", "compute: 1×",
       "push 56 · plate 46", GREY, GREY, result="base")
regime(xs[1], "Closed-loop  h=10", "replan every 10 steps\nre-observe + correct", "compute: ~5×",
       "push 78 · plate 100", GREEN, GREEN, result="loop")
regime(xs[2], "Distilled student", "one open-loop pass,\ntrained on closed-loop\ntrajectories", "compute: 1×",
       "plate 86 ✓ · push 28 ✗", BLUE, BLUE, trained=True)
regime(xs[3], "A2C2 head  (ours)", "open-loop base chunk\n+ cheap per-step\ncorrection (proprio)", "compute: ~1×",
       "reactive fix — testing", PURPLE, PURPLE, trained=True)

# ---------- bottom takeaway ----------
ax.add_patch(FancyBboxPatch((2, 6), 96, 9, boxstyle="round,pad=0.02,rounding_size=2",
                            fc="#f1f3f4", ec=INK, lw=1.2, zorder=1))
ax.text(50, 12.4, "Finding:  the closed-loop gain is recoverable at 1× compute  IFF it is trajectory-SHAPE (plate ✓)  not reactive re-observation (push ✗)",
        ha="center", fontsize=10.6, weight="bold", color=INK)
ax.text(50, 8.4, "Distillation removes the loop (works only for shape).  A2C2 keeps cheap re-observation every step (targets reactive) — no when-to-replan signal needed.",
        ha="center", fontsize=9, color="#444")

fig.tight_layout()
fig.savefig(FIG/"fig0_architecture.png", dpi=170, bbox_inches="tight")
print("wrote", FIG/"fig0_architecture.png")
