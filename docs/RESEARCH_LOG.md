# Horizon — Research Log

The honest decision trail for the Horizon project: *closing the loop on frozen
flow-matching VLAs*. Each entry is written as **Question → What we found (measured)
→ Decision**. Numbers are taken from result JSONs in
`RESEARCH/active/_harness-faithful-smolvla/results/`; where a number is a
single-seed-pool preliminary, it is flagged as such. Newest entry is at the
**bottom**.

Base model throughout: **FROZEN** SmolVLA (`smolvla_metaworld` checkpoint),
flow-matching action expert, run through lerobot on MetaWorld. Default execution
is chunk size 50, `n_action_steps=50` (open-loop), 10 Euler denoise steps.

Five swings appear below: one premise check, four falsifications, and the
positive anchor they produced. We keep the dead ends in full because the honesty
narrative *is* the contribution context — every "no" sharpened the eventual "yes."

---

## 0. Premise — is there a *publishable, train-free* frozen-VLA method here at all?

**Question.** Can a train-free, test-time wrapper around a frozen SmolVLA produce
a real, reviewer-defensible capability gain on MetaWorld — enough to be a paper,
not just a knob?

**What we found (measured).** First we had to earn a *measurable* testbed (Gate A).
A faithful lerobot harness (official pre/post processors + the rename-map fix; see
`METHODOLOGY.md`) gives a 12-task base survey at 3 seeds with Wilson CIs
(`results/base_survey_metaworld.json`). The base is **not** uniformly floored: five
tasks sit in a detectable 40–67% mid-band —

| task | base success (pooled, n=90) | 95% Wilson CI | band |
|---|---|---|---|
| drawer-open-v3 | 66.7% | [56.4, 75.5] | mid |
| push-v3 | 57.8% | [47.5, 67.5] | mid |
| window-open-v3 | 50.0% | [39.9, 60.1] | mid |
| plate-slide-v3 | 46.7% | [36.7, 56.9] | mid |
| peg-insert-side-v3 | 40.0% | [30.5, 50.3] | mid |
| reach-v3 | 27.8% | [19.6, 37.8] | floored outlier |
| pick-place-v3 | 7.8% | [3.8, 15.2] | floored outlier |
| button-press-v3 | 93.3% | [86.2, 96.9] | ceilinged |
| handle-press-v3 | 98.9% | [94.0, 99.8] | ceilinged |
| coffee-button-v3 | 100.0% | [95.9, 100.0] | ceilinged |

Crucially, **reach-v3 — the task every prior "this base is dead" verdict leaned
on — is a floored outlier (27.8%), not representative.** The headroom is real and
sits in the mid-band, so a gain is detectable.

**Decision.** Proceed: there *is* measurable headroom. Hunt for a train-free
test-time method that closes it, evaluated under strict paired stats so we cannot
fool ourselves. (Spoiler the next four entries deliver: every pure-wrapper idea
falsified. The premise as stated — *train-free* — does not hold.)

---

## 1. CASTLE — does STL-temporal-logic *routing* of corrections carry the gain?

**Question.** If we route corrective interventions to a frozen VLA using a signal
temporal logic (STL) robustness score ρ — intervene where/when the trajectory most
violates the task spec — does the *routing* (the placement of the correction) do
the work, or is it just spending an intervention budget?

**What we found (measured).** A budget-controlled kill-test: ρ-routed correction
vs. *random* correction at a **matched intervention budget**. At matched budget the
two were indistinguishable — **ρ-routed ≈ random**. The ~+30pp residual that CASTLE
produced was driven entirely by *coverage* (how many steps get a correction), not
by *where* STL placed them. Routing was **not load-bearing**.

**Decision.** **FALSIFIED** as a method. STL routing reduces to coverage. CASTLE
survives only as a *failure detector* (the ρ signal flags failing episodes,
AUROC ~0.87) — a diagnostic, not a capability method. Drop it as the lead.

---

## 2. STL-GDPA — does analytic ρ-gradient "approach guidance" steer the policy?

**Question.** Replace discrete routing with a continuous nudge: follow the analytic
gradient of the STL robustness ρ to guide the action expert toward satisfying the
task spec (gradient-directed predictive approach). Does spec-aware guidance beat a
naive attractor?

**What we found (measured).** The guidance **collapsed to a trivial oracle
goal-attractor**. The tell: a **wrong-order** nudge (guide toward subgoals in the
*wrong* sequence) performed **as well as or better than** the correctly-ordered,
spec-faithful nudge. If scrambling the temporal structure does not hurt, the STL
structure is doing nothing — all the lift came from "pull toward the goal," which
needs privileged goal state (an oracle), not the logic.

**Decision.** **FALSIFIED.** The method is an oracle goal-attractor wearing STL
clothing; the temporal-logic content is inert. Abandon analytic-ρ guidance.

---

## 3. NIAC averaging — does test-time noise-averaging over K flow draws help?

**Question.** A flow-matching expert maps a fresh noise sample to an action chunk.
Averaging K independent draws (`mean-K`) is the inference-time shadow of a
noise-invariance / consistency objective: if sampling noise is the bottleneck,
variance reduction should raise success. The on-manifold control is `consensus-K`
(medoid — pick the most central *real* draw); if medoid ≥ mean, the win is "pick a
central real sample," not "average." All arms share one seed pool so episodes are
**paired** (`run_niac_probe.sh`, `start_seed=1000`).

**What we found (measured).** **Null on both tasks** (`results/niac_probe/`):

| task | base | mean-K4 | medoid-K4 | dose-response (base→K2→K4→K8) | paired McNemar p |
|---|---|---|---|---|---|
| push-v3 | 64% (n=100) | 66% (n=100) | 58% (n=50) | 64 → 62 → 66 → 60 (non-monotone) | 0.885 |
| plate-slide-v3 | 58.3% (n=60) | 63.3% (n=60) | — | — | 0.664 |

No dose-response, no mechanism: **mean-K4 == medoid-K4 == 58%** on push at matched
n=50 (no variance-reduction signature — if averaging helped, mean would beat
medoid). On the matched 50-seed pool, **base (66%) ≥ every averaging arm**.

**The bug this run caught (and why we now trust the null).** An *earlier* pass had
reported a "+2pp" for averaging. It was an artifact of (a) comparing **unpaired,
overlapping Wilson CIs** and (b) **mismatched n** across arms (base at one n,
mean-K at another). Re-running with **one seed pool, matched n, and paired
McNemar** erased the +2pp entirely. This is the motivating example for the stats
discipline in `METHODOLOGY.md`: the false positive came from *exactly* the two
shortcuts (unpaired CIs, mismatched n) that the locked protocol now forbids.

**Decision.** **FALSIFIED.** Sampling-noise variance is not the success bottleneck
on these tasks; averaging does not add capability. Do not claim a boost. Keep the
stats fix as a permanent guardrail.

---

## 4. NIAC dispersion → failure — is action-sampling dispersion a *leading* signal?

**Question.** Even if averaging does not help, does the **dispersion** of the K
candidate chunks *predict* failure early enough to act on (a leading signal we
could gate on)? If high dispersion → impending failure, we have a usable trigger.

**What we found (measured).** **No robust leading signal; the two tasks disagree**
(`dispersion_push-v3.json`, `dispersion_plate-slide-v3.json`):

| task | AUROC(disp→failure) mean | best feature | early-window AUROC | verdict in JSON |
|---|---|---|---|---|
| push-v3 | 0.06 (backwards) | 0.53 (max) | 0.05 (backwards) | PREMISE-DEAD |
| plate-slide-v3 | 0.57 | 0.79 (max) | 0.29 (backwards) | "PREMISE-ALIVE" |

Read honestly: on push, the mean-feature AUROC of **0.06** is *backwards* (low
dispersion predicts failure, not high), and the best feature barely clears chance
at 0.53. Plate's headline 0.79 is only on the late-window `max` feature, while its
**early** window is **0.29 — backwards** — i.e. any predictive power is a **lagging
symptom** of a failure already underway, not a leading trigger. The tasks point in
opposite directions. (Note: the *success direction* of the same dispersion signal
is strong on push — successful episodes show **higher** mean dispersion than failed
ones, 1.02 vs 0.77, AUROC 0.94 *for success* — but that is a descriptor of the
failure mode, not an actionable leading predictor.)

**Decision.** **FALSIFIED** as a leading predictor. No reliable early dispersion
trigger exists across tasks. *But* the success-direction observation — failed
episodes are the **low-diversity, over-confident** ones — is the mechanistic seed
for the next swing.

---

## 5. PIVOT — the open-loop execution gap (the positive anchor)

**Question.** Four wrappers failed. The recurring lesson: a test-time wrapper can
only *reselect* the base's own samples — it cannot add capability — and the failure
mode is **confident open-loop commitment** (the base locks onto a confident, low-
diversity 50-step chunk and rides it into the wall). So: what if the problem is not
*which* chunk but *how long we commit to it*? SmolVLA executes the whole 50-step
chunk **open-loop** (`n_action_steps=50`). Shrink the execution horizon — replan
more often, i.e. close the loop — and does success rise?

**What we found (measured).** A horizon sweep via the `--n_action_steps` override
(`cascade_eval.py`), all arms on one paired seed pool (seeds 1000–1049, n=50)
(`results/niac_probe/horizon_*.json`):

| task | h=50 (open-loop default) | h=25 | h=10 (closed-loop) | paired McNemar (h50→h10) |
|---|---|---|---|---|
| push-v3 | 56% | 76% | 78% (**+22pp**) | helped 17 / hurt 6, p = 0.035 |
| plate-slide-v3 | 46% | — | 100% (**+54pp**) | helped 27 / hurt 0, p ≈ 1e-8 |

The gain is **large, paired, replicated on both tasks, and monotone in horizon**
(push 56 → 76 → 78 as h goes 50 → 25 → 10). The mechanism matches entry 4 exactly:
a confident-but-wrong 50-step chunk is unrecoverable for 50 steps; closing the loop
lets the base observe and correct.

**Decision.** **POSITIVE — this is the repo's empirical anchor.** Closing the
execution loop is a real, paired, replicated capability lever on the frozen base.
This becomes the motivation and the preliminary number for the Horizon project.

---

## 6. Current verdict (honest ceiling + what is and isn't the contribution)

**Question.** Is the open-loop-gap result a paper by itself?

**What we found (measured + literature).** The result is verified, but the
train-free lever is a **KNOWN knob**: receding-horizon control / ACT temporal
ensembling / **Bidirectional Decoding (BID, NeurIPS 2024 — closed-loop action-chunk
resampling)** / real-time async chunking. And it is not free: it costs **~5×
inference** (replan every 10 steps instead of 50). So the train-free version is the
*motivation plus a positive preliminary number*, not a novel contribution.

**Decision (current standing).**
- **Keep** the open-loop-gap as the positive anchor and the motivation. State its
  ceiling plainly; do not overclaim a known knob as new.
- The **proposed contribution** is a *trained, compute-efficient* version: a learned
  replan-gate on frozen-expert features (fire a replan only when the current chunk
  is going off), or distilling the closed-loop (h=10) policy into a single h=50
  forward pass — recovering most of the +22/+54pp at near-1× compute. Evaluated
  against a **vanilla-finetune control at matched compute**, so the claim is
  "method > same compute spent naively," not "method > base."
- This contribution is **SCOOP-GATED**: BID is the likely nearest prior art, and
  the space (async chunking, learned termination, adaptive horizon) is crowded.
  Novelty hinges entirely on the delegated web audit (PROPOSAL §6). **No method is
  claimed novel until that audit returns UNCLAIMED.**
- Reach-v3 stays banned as a headline task (floored outlier). All future numbers
  must use the paired-McNemar / one-seed-pool / matched-n / vanilla-FT-control
  discipline that caught our own +2pp false positive (entry 3).

**Status: positive anchor verified; trained method proposed but scoop-gated;
train-free knob honestly disclosed as known.**
