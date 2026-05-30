# Horizon replan-gate v2 — design (post-mortem-driven)

v1 verdict: KILL-GATE NOT PASSED (0/2). Diagnosis: the target was wrong, and the
testbed had no headroom. v2 fixes both. This is a design, not a run.

## What killed v1 (precise)
1. **Wrong target.** v1 trained on *action staleness* `‖E[fresh] − committed‖`.
   Highly predictable (val_corr 0.79) but orthogonal to *where replanning changes
   the outcome*. Plate q0.95 proved it: gate placed replans by staleness → 45% vs
   fixed 100% at the same budget (p=0.0001, 0/22 discordant).
2. **No headroom in the arena.** plate ceilings at **100% with fixed h=25 @ 3
   replans** — unbeatable by timing. push fixed curve is **non-monotonic** (h25
   72.5% > h15 65% > h10 62.5%) — "replan more at the right time" is incoherent
   there. A timing gate can only shine on a task where fixed-frequency is on a
   smooth, sub-ceiling success-vs-budget curve.

## Feasibility constraints (verified this session, not assumed)
- **Counterfactual branching is NOT available.** lerobot's `MetaworldEnv`
  unwrapped exposes no `sim/data/model/get_env_state/set_env_state` — cannot save
  a MuJoCo state and try replan-vs-not. So the "oracle: did replanning here help?"
  target requires either (a) forking the env to expose `mj` state, or (b) a
  *retrospective* offline label instead. v2 uses (b).
- **Dense reward + success ARE available** every `env.step` (already logged:
  `sum_reward`, `max_reward` per episode). MetaWorld's dense reward is shaped and
  progress-tracking → usable as the outcome signal **without** privileged geometry
  or sim access. This is the key unlock.

## v2 core change: an OUTCOME-LINKED, RETROSPECTIVE target
Replace staleness with a label computed *after* the episode, from the reward
trace — "was the policy about to stop making progress / head toward failure here?"

**Target A — progress-stall (regression).** At step t with committed chunk,
`y_t = max(0, r_t − max_{k≤H} r_{t+k})` over the open-loop window: how much future
shaped-reward is *lost* by committing the current chunk vs. its own near-term
peak. High = the chunk stalls/regresses → replanning is worth it.

**Target B — failure-attribution (classification, stronger).** Roll the frozen
policy fully open-loop (h=50) AND fully closed (h=10) on the SAME seed pool
(already have this data from the anchor). Label step t positive iff: open-loop
episode FAILS, closed-loop SUCCEEDS, and t falls in the window where the two
trajectories diverge (reward gap opens). This is the closest *cheap* proxy to "a
replan here flips the outcome" — it directly learns the moments closing the loop
rescued.

Both are computable from data we can already collect; neither needs sim branching.

## v2 secondary fixes
- **Budget = replans-per-executed-step**, not per-episode. v1's `avg_replans` was
  confounded by episode length (success ends early → fewer replans). Re-define the
  matched-budget axis as replan *rate*; recompute the kill-gate on it.
- **Pick a headroom task.** From the base survey, choose a task whose
  fixed-frequency curve is smooth and sub-ceiling in the comparable budget band.
  Candidates to screen first: window-open-v3 (~50%), drawer-open-v3 (~67%),
  and the floored multi-step ones (pick-place 8%) where there's the most room —
  run a 2-point fixed screen (h=50 vs h=10) and KEEP ONLY tasks where the gap is
  large AND monotonic. Drop plate (ceiling) and push (non-monotonic) as primary.
- **Budget regularizer at train OR eval.** Add a τ chosen to *hit* a target
  replan-rate inside the fixed-curve range, so every gate point is comparable
  (v1 had 4/8 gate points land OUT of range).

## v2 kill-gate (unchanged bar, cleaner axis)
On a headroom task, the gate must beat fixed-frequency at a matched **replan-rate**
with a significant paired McNemar (p<0.05). Same falsifiable discipline; if it
ties, still dead — but now it's a fair test of the *direction*, not of a bad
target on a bad arena.

## Effort / risk
- Reuses all v1 infra (`gate.py` collect/train/sweep/verdict, the harness, the
  parallel sweep). New code: ~1 target-labeling function + the rate-based budget
  axis in `verdict`. ~half a day.
- **Honest risk:** even with the right target, if no MetaWorld task gives a
  smooth sub-ceiling fixed curve, the *direction* is unfalsifiable here and the
  real conclusion is "this question needs a different benchmark" — which is itself
  a legitimate finding, not a failure to hide.

## Recommendation
Run **Target B on a screened headroom task** first (it's the most direct test and
reuses the anchor's open/closed rollouts). If B can't beat fixed-frequency at
matched rate on a task that *has* headroom, the trained-gate direction is dead for
real — and v1+v2 together make a clean, publishable negative ("the gap is real;
learned when-to-replan does not recover it cheaply; here's the evidence and why").
