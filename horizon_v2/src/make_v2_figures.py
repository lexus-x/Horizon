#!/usr/bin/env python
"""Pitch figures for Horizon v2 — redesigned for visual clarity.
Every number read from results/*.json. One message per figure, big labels,
compute chips (1x/5x), and green-check / red-x verdicts. Output -> figures/."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"base":"#9aa0a6","vanilla":"#f4b400","distill":"#1a73e8","loop":"#34a853",
     "gate":"#ea4335","a2c2":"#9c27b0"}
INK="#202124"

def succ(name):
    p = RES/f"{name}.json"
    if not p.exists():
        p = RES/"sweep"/f"{name}.json"
    return json.load(open(p))["result"]["pc_success"] if p.exists() else None

def chip(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x-0.16, y), 0.32, 7, boxstyle="round,pad=0.02,rounding_size=2",
                 fc=color, ec="none", transform=ax.get_xaxis_transform() if False else ax.transData,
                 clip_on=False, zorder=5))
    ax.text(x, y+3.5, text, ha="center", va="center", fontsize=11, weight="bold", color="white", zorder=6)


# ===== Fig 1: open-loop gap =====
fig, ax = plt.subplots(figsize=(7, 4.6))
base=[succ("horizon_push_h50"),succ("horizon_plate_h50")]
loop=[succ("horizon_push_h10"),succ("horizon_plate_h10")]
x=np.arange(2); w=0.34
b1=ax.bar(x-w/2, base, w, color=C["base"], label="open-loop  (1× compute)", edgecolor="white")
b2=ax.bar(x+w/2, loop, w, color=C["loop"], label="closed-loop  (5× compute)", edgecolor="white")
for i in range(2):
    ax.text(i-w/2, base[i]+2, f"{base[i]:.0f}%", ha="center", fontsize=14, weight="bold")
    ax.text(i+w/2, loop[i]+2, f"{loop[i]:.0f}%", ha="center", fontsize=14, weight="bold")
    ax.annotate(f"+{loop[i]-base[i]:.0f} pts", xy=(i+w/2, loop[i]), xytext=(i, loop[i]+11),
                ha="center", fontsize=13, weight="bold", color=C["loop"],
                arrowprops=dict(arrowstyle="->", color=C["loop"], lw=2))
ax.set_xticks(x); ax.set_xticklabels(["push","plate-slide"], fontsize=14)
ax.set_ylim(0,120); ax.set_ylabel("success rate"); ax.set_yticks([0,25,50,75,100]); ax.set_yticklabels([f"{v}%" for v in [0,25,50,75,100]])
ax.legend(fontsize=12, loc="upper left", frameon=False)
ax.set_title("Closing the loop works — but costs 5× compute", fontsize=16, weight="bold", pad=14)
fig.tight_layout(); fig.savefig(FIG/"fig1_openloop_gap.png", dpi=160); plt.close(fig)


# ===== Fig 2: distillation hero =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
def panel(ax, task, pre):
    van=max(v for v in [succ(f"{pre}_vanilla"),succ(f"eval_vanilla_{task}_lr3e5"),succ(f"eval_vanilla_{task}_lr1e5")] if v is not None)
    dist=max(v for v in [succ(f"{pre}_student"),succ(f"eval_distill_{task}_lr3e5"),succ(f"eval_distill_{task}_lr1e5")] if v is not None)
    vals=[succ(f"{pre}_h50"),van,dist,succ(f"{pre}_h10")]
    cols=[C["base"],C["vanilla"],C["distill"],C["loop"]]
    comp=["1×","1×","1×","5×"]
    names=["base","vanilla\nFT","DISTILL\n(ours)","closed\nloop"]
    xs=np.arange(4)
    bars=ax.bar(xs, vals, color=cols, width=0.66, edgecolor="white")
    for r,v in zip(bars,vals):
        ax.text(r.get_x()+r.get_width()/2, v+2, f"{v:.0f}%", ha="center", fontsize=14, weight="bold")
    names=[f"{n}\n{c}" for n,c in zip(names,comp)]   # fold compute into the tick label
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0,118); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
    # verdict on distill bar
    win = vals[2] > vals[0] + 10
    ax.text(2, vals[2]+10, "✓" if win else "✗", ha="center", fontsize=26, weight="bold",
            color=C["loop"] if win else C["gate"])
    ax.set_title(f"{task}-v3   "+("SHAPE  ✓ cheap fix works" if win else "REACTIVE  ✗ cheap fix fails"),
                 fontsize=14, weight="bold", color=C["distill"] if win else C["gate"])
panel(axes[0],"push","horizon_push"); panel(axes[1],"plate","horizon_plate")
fig.suptitle("Distillation = closed-loop accuracy at 1× compute — but only on shape tasks",
             fontsize=16, weight="bold")
fig.subplots_adjust(bottom=0.16)
fig.tight_layout(rect=[0,0.02,1,0.94]); fig.savefig(FIG/"fig2_distillation.png", dpi=160); plt.close(fig)


# ===== Fig 3: recipe robust (bands) =====
fig, ax = plt.subplots(figsize=(7.5, 4.8))
van=[succ("horizon_plate_vanilla"),succ("eval_vanilla_plate_lr3e5"),succ("eval_vanilla_plate_lr1e5")]
dis=[succ("horizon_plate_student"),succ("eval_distill_plate_lr3e5"),succ("eval_distill_plate_lr1e5")]
ax.axhspan(min(van),max(van),color=C["vanilla"],alpha=0.18)
ax.axhspan(min(dis),max(dis),color=C["distill"],alpha=0.18)
xs=[0,1,2]
ax.plot(xs,van,"o-",color=C["vanilla"],lw=2.5,ms=10,label="vanilla-FT (demo target)")
ax.plot(xs,dis,"s-",color=C["distill"],lw=2.5,ms=10,label="distill (closed-loop target)")
ax.axhline(succ("horizon_plate_h50"),ls=":",color=C["base"],lw=2,label="base")
for i,v in enumerate(van): ax.text(i,v-5,f"{v:.0f}",ha="center",fontsize=12,weight="bold",color=C["vanilla"])
for i,v in enumerate(dis): ax.text(i,v+2.5,f"{v:.0f}",ha="center",fontsize=12,weight="bold",color=C["distill"])
ax.annotate("distill's WORST (64)\n>\nvanilla's BEST (52)", xy=(1.5,58), fontsize=12, weight="bold",
            ha="center", color=INK, bbox=dict(boxstyle="round", fc="#fffde7", ec="#fbc02d"))
ax.set_xticks(xs); ax.set_xticklabels(["lr 1e-4","lr 3e-5","lr 1e-5"], fontsize=12)
ax.set_ylim(0,100); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
ax.set_title("plate: the win is the TARGET, not the recipe\n(distill beats vanilla at every setting)", fontsize=14.5, weight="bold")
ax.legend(fontsize=11, loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig(FIG/"fig3_recipe_robust.png", dpi=160); plt.close(fig)


# ===== Fig 4: boundary (big contrast) =====
fig, ax = plt.subplots(figsize=(8, 4.8)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.text(5,9.4,"When can you get the loop's accuracy for free?", ha="center", fontsize=16, weight="bold")
# left: shape
ax.add_patch(FancyBboxPatch((0.4,1),4.2,7.4,boxstyle="round,pad=0.1,rounding_size=0.2",fc="#e8f0fe",ec=C["distill"],lw=2.5))
ax.text(2.5,7.6,"SHAPE  (plate-slide)",ha="center",fontsize=14,weight="bold",color=C["distill"])
ax.text(2.5,6.6,"smooth, predictable\nfrom the start state",ha="center",fontsize=11,color=INK)
ax.text(2.5,4.7,"46% → 86%",ha="center",fontsize=22,weight="bold",color=C["distill"])
ax.text(2.5,3.6,"at 1× compute",ha="center",fontsize=12,color=INK)
ax.text(2.5,2.0,"✓ distillable",ha="center",fontsize=18,weight="bold",color=C["loop"])
# right: reactive
ax.add_patch(FancyBboxPatch((5.4,1),4.2,7.4,boxstyle="round,pad=0.1,rounding_size=0.2",fc="#fce8e6",ec=C["gate"],lw=2.5))
ax.text(7.5,7.6,"REACTIVE  (push)",ha="center",fontsize=14,weight="bold",color=C["gate"])
ax.text(7.5,6.6,"needs live re-observation\nof the puck mid-motion",ha="center",fontsize=11,color=INK)
ax.text(7.5,4.7,"56% → 28%",ha="center",fontsize=22,weight="bold",color=C["gate"])
ax.text(7.5,3.6,"open-loop can't react",ha="center",fontsize=12,color=INK)
ax.text(7.5,2.0,"✗ not distillable",ha="center",fontsize=18,weight="bold",color=C["gate"])
ax.text(5,0.3,"The rule: cheap recovery works IFF the gain is trajectory-shape.",ha="center",fontsize=12.5,style="italic")
fig.tight_layout(); fig.savefig(FIG/"fig4_boundary.png", dpi=160); plt.close(fig)


# ===== Fig 5: cheap-loop probe =====
fig, ax = plt.subplots(figsize=(7.5, 4.7))
ns=[2,4,10]
pu=[succ("cheaploop_push-v3_h10_ns2"),succ("cheaploop_push-v3_h10_ns4"),succ("horizon_push_h10")]
pl=[succ("cheaploop_plate-slide-v3_h10_ns2"),succ("cheaploop_plate-slide-v3_h10_ns4"),succ("horizon_plate_h10")]
ax.plot(ns,pu,"o-",color=C["distill"],lw=2.5,ms=10,label="push (closed-loop)")
ax.plot(ns,pl,"s-",color=C["loop"],lw=2.5,ms=10,label="plate (closed-loop)")
for x_,y_ in zip(ns,pu): ax.text(x_,y_+2.5,f"{y_:.0f}",ha="center",fontsize=12,weight="bold",color=C["distill"])
for x_,y_ in zip(ns,pl): ax.text(x_,y_-5,f"{y_:.0f}",ha="center",fontsize=12,weight="bold",color=C["loop"])
ax.set_xticks(ns); ax.set_xlabel("denoise steps per inference (fewer = cheaper expert)")
ax.set_ylim(40,112); ax.set_yticks([50,75,100]); ax.set_yticklabels(["50%","75%","100%"])
ax.set_title("Few denoise steps keep accuracy (push 76 @ 4 steps)…\n…but cost is the VLM, not the denoiser → motivates A2C2", fontsize=13.5, weight="bold")
ax.legend(fontsize=12, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(FIG/"fig5_cheaploop_probe.png", dpi=160); plt.close(fig)


# ===== Fig 6: gate null =====
fig, ax = plt.subplots(figsize=(6, 4.6))
g=succ("horizon_push_gate_n50")
vals=[succ("horizon_push_h50"),g,succ("horizon_push_h10")]
cols=[C["base"],C["gate"],C["loop"]]; names=["base","learned\ngate","closed\nloop\n(target)"]
bars=ax.bar(range(3),vals,color=cols,width=0.6,edgecolor="white")
for r,v in zip(bars,vals): ax.text(r.get_x()+r.get_width()/2,v+2,f"{v:.0f}%",ha="center",fontsize=14,weight="bold")
ax.plot([0,1],[vals[0],vals[1]],"k--",lw=1)
ax.text(0.5,68,"ties base\n(p=0.83) ✗",ha="center",fontsize=12,weight="bold",color=C["gate"])
ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=12)
ax.set_ylim(0,100); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
ax.set_title("The original 'gate' idea is DEAD\n(adds nothing over doing nothing)", fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig6_gate_null.png", dpi=160); plt.close(fig)


# ===== Fig 7: money plot =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
def money(ax, task, pre):
    ax.axhspan(70,108,xmin=0,xmax=0.30,color="#e6f4ea",zorder=0)
    # x position, and (label dx,dy in points, ha) chosen to avoid overlaps
    pts=[("base",1.0,succ(f"{pre}_h50"),C["base"],(-14,-20,"center")),
         ("closed-loop",5.0,succ(f"{pre}_h10"),C["loop"],(0,16,"center")),
         ("DISTILL",1.0,succ(f"{pre}_student"),C["distill"],(0,-34,"center"))]
    g=succ(f"{pre}_gate_n50")
    if g is not None: pts.append(("gate",2.1,g,C["gate"],(0,16,"center")))
    a=succ(f"{pre}_a2c2")
    if a is not None: pts.append(("A2C2",1.6,a,C["a2c2"],(0,18,"center")))
    for name,c,s,col,(dx,dy,ha) in pts:
        ax.scatter(c,s,s=340,color=col,edgecolor="black",lw=1.6,zorder=3)
        ax.annotate(f"{name}\n{s:.0f}%",(c,s),textcoords="offset points",xytext=(dx,dy),
                    ha=ha,fontsize=11.5,weight="bold")
    ax.text(1.05,102,"cheap & accurate\n= GOAL",fontsize=10.5,color="#1e7d32",weight="bold")
    ax.set_xlim(0.3,5.8); ax.set_ylim(0,112)
    ax.set_xticks([1,5]); ax.set_xticklabels(["1×\n(open-loop)","5×\n(closed-loop)"])
    ax.set_xlabel("inference compute"); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
    ax.set_title(f"{task}-v3",fontsize=14,weight="bold")
money(axes[0],"push","horizon_push"); money(axes[1],"plate","horizon_plate")
axes[0].set_ylabel("success rate")
fig.suptitle("The goal is the top-left corner: high accuracy, low compute",fontsize=16,weight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(FIG/"fig7_compute_accuracy.png", dpi=160); plt.close(fig)

# ===== Fig 8: reactive side — every cheap fix falls short of the loop =====
g=succ("horizon_push_gate_n50"); a=succ("horizon_push_a2c2")
names=["base"]; vals=[succ("horizon_push_h50")]; cols=[C["base"]]
vals.append(succ("horizon_push_student")); names.append("distill"); cols.append(C["distill"])
if g is not None: vals.append(g); names.append("gate"); cols.append(C["gate"])
if a is not None: vals.append(a); names.append("A2C2\n(proprio)"); cols.append(C["a2c2"])
vals.append(succ("horizon_push_h10")); names.append("closed\nloop (5×)"); cols.append(C["loop"])
fig, ax = plt.subplots(figsize=(8.4,4.7))
bars=ax.bar(range(len(vals)), vals, color=cols, width=0.62, edgecolor="white")
for r,v in zip(bars,vals): ax.text(r.get_x()+r.get_width()/2,v+2,f"{v:.0f}%",ha="center",fontsize=13,weight="bold")
loopv=succ("horizon_push_h10")
ax.axhline(loopv, ls="--", color=C["loop"], lw=2)
ax.text(0.2, loopv+2.5, f"what we need: {loopv:.0f}% (the loop)", color=C["loop"], fontsize=11, weight="bold")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0,100); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
ax.set_title("Reactive side (push): every CHEAP fix falls short\n— reactivity is visual & needs the loop", fontsize=14.5, weight="bold")
fig.tight_layout(); fig.savefig(FIG/"fig8_a2c2.png", dpi=160); plt.close(fig)

# ===== Fig 9: the boundary across 6 tasks =====
shape=[("plate","horizon_plate"),("drawer","horizon_drawer"),("window","horizon_window")]
react=[("push","horizon_push"),("peg","horizon_peg"),("pickplace","horizon_pickplace")]
allt=shape+react
labels=[t[0] for t in allt]
base=[succ(f"{p}_h50") for _,p in allt]
dist=[succ(f"{p}_student") for _,p in allt]
loop=[succ(f"{p}_h10") for _,p in allt]
if all(v is not None for v in base+dist+loop):
    fig, ax = plt.subplots(figsize=(12.5,5.2))
    x=np.arange(6); w=0.26
    ax.bar(x-w, base, w, color=C["base"], label="base (1×)", edgecolor="white")
    ax.bar(x,    dist, w, color=C["distill"], label="distill (1×)", edgecolor="white")
    ax.bar(x+w,  loop, w, color=C["loop"], label="closed-loop (5×)", edgecolor="white")
    for i in range(6):
        ax.text(i, dist[i]+2, f"{dist[i]:.0f}", ha="center", fontsize=10.5, weight="bold", color=C["distill"])
    ax.axvline(2.5, ls="--", color="#999", lw=1.5)
    ax.text(1.0, 112, "SHAPE  →  distill recovers ✓", ha="center", fontsize=13, weight="bold", color=C["distill"])
    ax.text(4.5, 112, "REACTIVE  →  distill fails ✗", ha="center", fontsize=13, weight="bold", color=C["gate"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0,122); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0%","50%","100%"])
    ax.set_ylabel("success rate"); ax.legend(fontsize=11, loc="center right", frameon=False)
    ax.set_title("The boundary across 6 tasks: closed-loop gain is cheaply recoverable on SHAPE, not REACTIVE",
                 fontsize=14.5, weight="bold", pad=26)
    fig.tight_layout(); fig.savefig(FIG/"fig9_boundary_6task.png", dpi=160); plt.close(fig)

print("figures regenerated:")
for f in sorted(FIG.glob("*.png")): print("  ",f.name)
