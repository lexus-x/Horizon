"""Assemble the morning report from all result JSONs. Paired McNemar by seed."""
import json, math
from pathlib import Path

R = Path("../results")

def load(name):
    p = R / name
    if not p.exists():
        return None
    d = json.load(open(p))
    pc = d["result"].get("pc_success")
    seedmap = {e["seed"]: bool(e["success"]) for e in d.get("per_episode", [])}
    return {"pc": pc, "seedmap": seedmap, "n": len(seedmap)}

def mcnemar(a, b):
    """two-sided exact McNemar on paired seeds; returns (delta_pp, b_helped, c_hurt, p)."""
    if not a or not b:
        return None
    seeds = sorted(set(a["seedmap"]) & set(b["seedmap"]))
    if not seeds:
        return None
    bb = sum(1 for s in seeds if b["seedmap"][s] and not a["seedmap"][s])   # b better than a
    cc = sum(1 for s in seeds if a["seedmap"][s] and not b["seedmap"][s])   # a better than b
    n = bb + cc
    k = min(bb, cc)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    delta = 100.0 * (sum(b["seedmap"][s] for s in seeds) - sum(a["seedmap"][s] for s in seeds)) / len(seeds)
    return (delta, bb, cc, p, len(seeds))

def fmt(name):
    r = load(name)
    return f"{r['pc']:.0f}% (n={r['n']})" if r else "—(missing)"

lines = ["# Overnight Report — 2026-06-11 night\n",
         "Autonomous run. Base = frozen public smolvla_metaworld. All eval seeds 1000-1049, h50 (1x), paired McNemar.\n"]

lines += ["## Job 1 — open-loop vs closed-loop teacher distillation (the kill-shot)\n",
          "Question: is Horizon's distillation gain the CLOSED-LOOP SIGNAL, or just BC on rollout-shaped windows?",
          "If open-loop-teacher-distill ~ base while closed-loop-teacher-distill is far above, the closed-loop signal is load-bearing.\n",
          "| task | base h50 | loop h10 | **distill CL (Horizon)** | distill OL (control) | verdict |",
          "|---|---|---|---|---|---|"]
for short, base, loop, cl in [("plate", "horizon_plate_h50.json", "horizon_plate_h10.json", "horizon_plate_student.json"),
                              ("push",  "horizon_push_h50.json",  "horizon_push_h10.json",  "horizon_push_student.json")]:
    ol = f"horizon_{short}_studentOL.json"
    a = load(base); o = load(ol); c = load(cl)
    verd = "—"
    if a and o and c:
        ol_vs_base = mcnemar(a, o); cl_vs_ol = mcnemar(o, c)
        if ol_vs_base and cl_vs_ol:
            verd = f"OL vs base {ol_vs_base[0]:+.0f}pp p={ol_vs_base[3]:.2g}; CL vs OL {cl_vs_ol[0]:+.0f}pp p={cl_vs_ol[3]:.2g}"
    lines.append(f"| {short} | {fmt(base)} | {fmt(loop)} | {fmt(cl)} | {fmt(ol)} | {verd} |")

lines += ["\n## Job 2 — push curated distillation (is the shape/reactive boundary movable?)\n",
          "Size-matched (21 eps each). Predictable = typical start-chunk; unpredictable = idiosyncratic.",
          "CAVEAT: split is by trajectory-typicality proxy (within-chunk replan signal was absent in stored targets); exploratory.\n",
          "| arm | SR | vs base(56) | vs full-CL push student(28) |",
          "|---|---|---|---|"]
base_push = load("horizon_push_h50.json"); full_push = load("horizon_push_student.json")
for strat in ["pred", "unpred"]:
    s = load(f"horizon_push_student_{strat}.json")
    vb = mcnemar(base_push, s) if s else None
    vf = mcnemar(full_push, s) if s else None
    vbs = f"{vb[0]:+.0f}pp p={vb[3]:.2g}" if vb else "—"
    vfs = f"{vf[0]:+.0f}pp p={vf[3]:.2g}" if vf else "—"
    lines.append(f"| push curated-{strat} | {fmt(f'horizon_push_student_{strat}.json')} | {vbs} | {vfs} |")

lines += ["\n## How to read",
          "- Job1: expect OL-distill ~ base (forgetting wall / self-distill) << CL-distill -> confirms the closed-loop signal is what distillation captures (bulletproofs Horizon's core claim).",
          "- Job2: if curated-pred > curated-unpred (and pred less collapsed vs base 56), that's first evidence the boundary is movable via init curation; if both ~equal/collapsed, the typicality proxy didn't move it (premise needs the real reset-curated re-collect, not a post-hoc split).",
          "\n(References for reactivity/predictability scoop: RFCL 2405.03379, IL-horizon 2407.15007, PACE 2606.00537, DEFLECT 2605.19294.)"]

out = R / "MORNING_REPORT.md"
out.write_text("\n".join(lines))
print("wrote", out)
print("\n".join(lines))
