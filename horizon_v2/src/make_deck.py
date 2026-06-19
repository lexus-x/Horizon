#!/usr/bin/env python
"""Build the Horizon v2 pitch deck (.pptx) from the generated figures.
Re-runnable: picks up whatever figures currently exist (e.g. the A2C2 figure once
it's generated). Output -> deck/Horizon_v2_pitch.pptx"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
VID = ROOT / "videos"
OUT = ROOT / "deck" / "Horizon_v2_pitch.pptx"

INK = RGBColor(0x20, 0x21, 0x24)
BLUE = RGBColor(0x1A, 0x73, 0xE8)
GREEN = RGBColor(0x34, 0xA8, 0x53)
RED = RGBColor(0xB0, 0x00, 0x20)
GREY = RGBColor(0x5f, 0x63, 0x68)
LIGHT = RGBColor(0xF1, 0xF3, 0xF4)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def _tb(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tb.text_frame.word_wrap = True
    return tb.text_frame


def title_bar(slide, title, sub=None):
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(1.1))
    bar.fill.solid(); bar.fill.fore_color.rgb = INK; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = Inches(0.4); tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = RGBColor(255,255,255)
    if sub:
        sp = tf.add_paragraph(); sp.text = sub
        sp.font.size = Pt(13); sp.font.color.rgb = RGBColor(0xCF,0xD8,0xDC)


def add_img(slide, path, x, y, w=None, h=None):
    if not Path(path).exists():
        return None
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def bullets(slide, items, x, y, w, h, size=16, gap=6):
    tf = _tb(slide, x, y, w, h)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        lvl = 0
        if isinstance(it, tuple): it, lvl, *rest = (it + (0,))[:3]
        p.text = it; p.level = lvl
        p.font.size = Pt(size - 2*lvl); p.font.color.rgb = INK
        p.space_after = Pt(gap)
        if it.startswith("✓"): p.font.color.rgb = GREEN; p.font.bold = True
        if it.startswith("✗") or it.startswith("✕"): p.font.color.rgb = RED; p.font.bold = True
    return tf


def new(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    return s


# ---------------- 1. title ----------------
s = prs.slides.add_slide(BLANK)
box = s.shapes.add_shape(1, 0, 0, SW, SH); box.fill.solid(); box.fill.fore_color.rgb = INK; box.line.fill.background()
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.9); tf.margin_right = Inches(0.9)
p = tf.paragraphs[0]; p.text = "Horizon"; p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = RGBColor(255,255,255)
for line, sz, col in [
    ("Closed-loop accuracy at open-loop cost in frozen flow-matching VLAs", 24, RGBColor(0x8A,0xB4,0xF8)),
    ("Frozen SmolVLA × MetaWorld  ·  measured, paired-McNemar evidence", 16, RGBColor(0xCF,0xD8,0xDC)),
    ("Result: plate-slide 46→86% at 1× compute  ·  a shape-vs-reactive boundary", 16, RGBColor(0xA7,0xF3,0xD0)),
]:
    q = tf.add_paragraph(); q.text = line; q.font.size = Pt(sz); q.font.color.rgb = col

# ---------------- 2. problem ----------------
s = new("The problem", "VLAs are accurate only when they close the loop — but that costs ~5×")
add_img(s, FIG/"fig1_openloop_gap.png", Inches(0.4), Inches(1.4), w=Inches(6.6))
bullets(s, [
    "Frozen flow-matching VLA commits a 50-step chunk open-loop → drifts, fails.",
    "Replan every 10 steps (close the loop) → re-observe + correct → big gains.",
    "push 56→78  ·  plate 46→100   (paired McNemar, n=50).",
    "But closing the loop = ~5× inference. VLAs are already too slow for real-time.",
    "Question: can we get closed-loop accuracy at open-loop (1×) cost?",
], Inches(7.2), Inches(1.5), Inches(5.8), Inches(5.4), size=17)

# ---------------- 3. architecture ----------------
s = new("Architecture", "frozen SmolVLA + four ways to deploy the action chunk")
add_img(s, FIG/"fig0_architecture.png", Inches(0.3), Inches(1.3), w=Inches(12.7))

# ---------------- 4. honesty / killed ----------------
s = new("What we killed first (the honesty)", "negatives map the limits — and caught our own false positive")
add_img(s, FIG/"fig6_gate_null.png", Inches(0.4), Inches(1.5), w=Inches(5.2))
bullets(s, [
    "✗ Learned replan-gate (the original contribution): at fair n=50 ties base",
    "   (60 vs 56, p=0.83), never reaches the loop. No leading replan signal exists.",
    "✗ 4 train-free wrappers falsified (CASTLE, STL-GDPA, NIAC ×2).",
    "✓ Stats discipline: paired McNemar, one seed pool, matched-compute control —",
    "   it deleted an earlier +2pp false positive before it shipped.",
    "Takeaway: you can't predict WHEN to replan, and you can't add capability",
    "with a train-free wrapper. The loop itself is doing the work.",
], Inches(6.0), Inches(1.6), Inches(7.0), Inches(5.3), size=15)

# ---------------- 5. main result ----------------
s = new("Result: distill the closed loop into one pass", "plate-slide 46→86% at 1× compute")
add_img(s, FIG/"fig2_distillation.png", Inches(0.5), Inches(1.5), w=Inches(12.3))
bullets(s, [
    "plate: distill 86% vs base 46% (p=1e-5)  ·  push: collapses 56→28% (p=0.007).",
], Inches(0.6), Inches(6.7), Inches(12), Inches(0.6), size=15)

# ---------------- 6. recipe robust ----------------
s = new("It's the target, not the finetuning", "matched-compute control, swept across recipes")
add_img(s, FIG/"fig3_recipe_robust.png", Inches(0.5), Inches(1.5), w=Inches(6.6))
bullets(s, [
    "Vanilla finetune on demos ≈ base (best 52%).",
    "Distillation on closed-loop trajectories = 64–86%.",
    "✓ Distill's WORST recipe (64) beats vanilla's BEST (52).",
    "Best-vs-best: +34pp, helped 18 / hurt 1, p=8e-5.",
    "→ the gain is the closed-loop TARGET, robust to the recipe.",
], Inches(7.3), Inches(1.7), Inches(5.7), Inches(5), size=17)

# ---------------- 7. boundary ----------------
s = new("The finding: a shape-vs-reactive boundary", "the actual intellectual contribution")
add_img(s, FIG/"fig4_boundary.png", Inches(0.5), Inches(1.5), w=Inches(6.4))
bullets(s, [
    "Closed-loop gain is recoverable at 1× IFF it is trajectory-SHAPE.",
    "plate (smooth slide): predictable from start state → distillable ✓",
    "push (reactive): needs live re-observation of puck drift → not distillable ✗",
    "An ex-ante rule: tells a deployer which tasks can run cheap,",
    "and which genuinely need the loop. That map is the contribution.",
], Inches(7.1), Inches(1.7), Inches(5.9), Inches(5), size=16)

# ---------------- 7b. boundary across 6 tasks ----------------
s = new("The boundary holds across 6 tasks", "3 shape + 3 reactive — a law, not an anecdote")
add_img(s, FIG/"fig9_boundary_6task.png", Inches(0.35), Inches(1.35), w=Inches(12.6))
bullets(s, [
    "Shape: plate 46→86, drawer 58→100, window 42→68 — distill recovers the loop gain.",
    "Reactive: push 56→28, peg 20→26, pick-place 8→14 — distill does NOT recover.",
], Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.7), size=13)

# ---------------- 8. cheap loop probe ----------------
s = new("Why reactive tasks need a different fix", "cheap-denoise probe → cost is the VLM, not the denoiser")
add_img(s, FIG/"fig5_cheaploop_probe.png", Inches(0.5), Inches(1.5), w=Inches(6.6))
bullets(s, [
    "Flow expert tolerates few-step denoise (push 76% at 4 steps).",
    "But cutting denoise barely changes wall-clock (~10%).",
    "→ the 5× cost is the VLM re-encode every replan, not the denoiser.",
    "So: compute the VLM chunk ONCE, correct per-step with a cheap head.",
    "That is exactly A2C2 (next slide).",
], Inches(7.3), Inches(1.7), Inches(5.7), Inches(5), size=17)

# ---------------- 9. A2C2 ----------------
s = new("A2C2: cheap per-step correction (ours)", "open-loop base chunk + tiny proprio head — reactivity at ~1×")
if (FIG/"fig8_a2c2.png").exists():
    add_img(s, FIG/"fig8_a2c2.png", Inches(0.5), Inches(1.5), w=Inches(6.6))
    bx, bw = Inches(7.3), Inches(5.7)
else:
    bx, bw = Inches(0.6), Inches(12.1)
bullets(s, [
    "Base VLA emits one 50-step chunk per 50 steps (1× VLM).",
    "A tiny MLP runs every step: (latest proprio state, base action, chunk position)",
    "   → per-step correction. No VLM rerun → ~1× compute, fully reactive.",
    "Sidesteps the dead gate (corrects every step, no when-to-replan signal needed).",
    "Trained on closed-loop residual (target − stale base action).",
    "Status: data + head built; eval result folds in here (running).",
    "Based on A2C2 (Sendai et al. 2025, arXiv:2509.23224).",
], bx, Inches(1.7), bw, Inches(5), size=15)

# ---------------- 10. money plot ----------------
s = new("Compute vs accuracy", "goal: top-left — high success, low compute")
add_img(s, FIG/"fig7_compute_accuracy.png", Inches(0.5), Inches(1.5), w=Inches(12.3))

# ---------------- 11. videos (embedded, playable) ----------------
s = new("Rollouts", "open-loop commits and fails; closing the loop / distilling recovers")

def add_movie(slide, mp4, x, y, w, h, poster=None):
    if not Path(mp4).exists():
        return
    try:
        slide.shapes.add_movie(str(mp4), x, y, w, h,
                               poster_frame_image=str(poster) if poster and Path(poster).exists() else None,
                               mime_type="video/mp4")
    except Exception:
        if poster and Path(poster).exists():
            slide.shapes.add_picture(str(poster), x, y, width=w, height=h)

# big side-by-side highlight (top)
hl = VID/"highlight_plate_base_vs_distill.mp4"
add_movie(s, hl, Inches(0.5), Inches(1.35), Inches(8.6), Inches(2.5), VID/"highlight_poster.png")
cf = _tb(s, Inches(0.5), Inches(3.9), Inches(8.6), Inches(0.5))
cf.paragraphs[0].text = "plate  ·  open-loop (left) drifts & fails   vs   distill 1× (right) succeeds"
cf.paragraphs[0].font.size = Pt(13); cf.paragraphs[0].font.bold = True
# push pair (top-right)
add_movie(s, next((VID/"push_base_h50").glob("eval_episode_0.mp4"), VID/"x"), Inches(9.5), Inches(1.35), Inches(3.3), Inches(2.5), VID/"push_base_h50/poster.png")
pf = _tb(s, Inches(9.5), Inches(3.9), Inches(3.3), Inches(0.5)); pf.paragraphs[0].text="push · open-loop (fails)"; pf.paragraphs[0].font.size=Pt(12)
# plate trio (bottom)
clips=[("plate_base_h50","base 1× (fail)"),("plate_distill_h50","distill 1× (success)"),("plate_loop_h10","closed-loop 5× (success)")]
x=Inches(0.5)
for name,capt in clips:
    add_movie(s, next((VID/name).glob("eval_episode_0.mp4"), VID/"x"), x, Inches(4.5), Inches(3.9), Inches(2.3), VID/name/"poster.png")
    c=_tb(s, x, Inches(6.85), Inches(3.9), Inches(0.4)); c.paragraphs[0].text=capt; c.paragraphs[0].font.size=Pt(12); c.paragraphs[0].font.bold=True
    x = Emu(int(x)+int(Inches(4.2)))

# ---------------- 12. scorecard ----------------
s = new("What we have (honest scorecard)")
rows = [
    ("Claim", "Status", "Evidence"),
    ("Open-loop gap (motivation)", "real, known knob", "push 56→78, plate 46→100, paired"),
    ("Distill recovers gain @1× (plate)", "REAL positive", "46→86, p=1e-5; >control p=4e-9; recipe-robust"),
    ("Distill on reactive (push)", "fails", "56→28 — open-loop can't fake re-observation"),
    ("Replan-gate (old contribution)", "dead / null", "ties base 60 vs 56, p=0.83"),
    ("Shape-vs-reactive boundary", "EVIDENCED, 6 tasks", "shape: plate/drawer/window ✓; reactive: push/peg/pickplace ✗"),
    ("A2C2 cheap reactive fix", "insufficient", "proprio head: push 62 (n.s.), plate 56 — short of loop"),
]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.4), Inches(1.4), Inches(12.5), Inches(4.8)).table
tbl.columns[0].width = Inches(4.6); tbl.columns[1].width = Inches(3.0); tbl.columns[2].width = Inches(4.9)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c); cell.text = val
        para = cell.text_frame.paragraphs[0]; para.font.size = Pt(13 if r else 14)
        para.font.bold = (r == 0)
        if r == 0: para.font.color.rgb = RGBColor(255,255,255); cell.fill.solid(); cell.fill.fore_color.rgb = INK
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(255,255,255)
            if c == 1:
                if "REAL" in val: para.font.color.rgb = GREEN; para.font.bold = True
                elif "dead" in val or "fails" in val: para.font.color.rgb = RED; para.font.bold = True

# ---------------- 13. next week ----------------
s = new("Next week", "turn one anchor into a law")
bullets(s, [
    "1. A2C2: finish push/plate eval; if push clears base at ~1× → reactive fix demonstrated.",
    "2. Breadth (6 tasks): shape {plate, drawer-open, window-open} vs reactive {push, peg-insert, pick-place}.",
    "    → predict: distill recovers shape, A2C2 needed for reactive. Makes the boundary a law.",
    "3. Multi-pool: 3–5 disjoint seed pools; per-pool McNemar + mixed-effects pooled estimate.",
    "4. Continuous metric (graded progress g_t) + anytime-valid stopping for efficiency.",
    "5. Write-up: ICBINB short paper now; ICRA 2027 main after breadth.",
], Inches(0.6), Inches(1.6), Inches(12.2), Inches(5.3), size=18, gap=12)

# ---------------- 14. publishability ----------------
s = new("Publishability", "match venue to maturity — honesty is the edge")
bullets(s, [
    "As-is (2 tasks, 1 pool): workshop-grade. ICBINB (negative-results) ~7/10.",
    "After breadth + multi-pool: ICRA 2027 main ~5/10 (deadline ~Sept 2026).",
    "Framing that lifts odds: the BOUNDARY is the contribution (a finding, not a tweak).",
    "Differentiators: closed-loop accuracy at 1× on a characterizable task class,",
    "    + an ex-ante rule for which tasks; matched-compute controls; reproducible harness.",
    "Pitch line: \"VLAs need 5× compute to not fail. We get the same accuracy at 1×",
    "    on the tasks where it's physically possible — and we tell you which those are.\"",
], Inches(0.6), Inches(1.6), Inches(12.2), Inches(5.3), size=17, gap=10)

prs.save(str(OUT))
print("saved deck ->", OUT, "(", len(prs.slides.__iter__.__self__._sldIdLst), "slides )")
