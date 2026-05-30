# Horizon replan-gate — built, evaluated. KILL-GATE: **NOT PASSED (0/2 tasks).**

The gate proposed in `docs/PROPOSAL.md` §4 (Candidate A) — previously only a
scaffold — is now implemented (`src/gate.py`) and run end-to-end on the frozen
`smolvla_metaworld` policy via the faithful lerobot harness. **Every number below
is copied from `verdict_n40.json`** (n=40 paired episodes/arm, seed pool @1000).
The 18-arm sweep finished in **~23 min** wall-clock by launching every arm as a
concurrent process (GPU 25%→99%, VRAM 2.7→36 GB of 80 — the headroom that made
the fan-out possible).

## What was built
A trained adaptive **replan-gate** on the FROZEN flow-matching SmolVLA:
- **collect** — roll the frozen policy under a randomized replan schedule; record
  cheap proprio features `x_t` (phase, committed action, proprio drift, open-loop
  tracking residual, velocity — 21-d, **no extra VLM forward**) and target
  `d_t = ||E[fresh-replan action] − committed-stale action||` (chunk *staleness*).
- **train** — a tiny MLP `x_t → d_t`.
- **eval/sweep** — `base` (h=50 open-loop), `fixed` (replan every H∈{25,15,10,8}),
  `gate` (replan when `MLP(x_t) > τ`, τ at quantiles {.80,.90,.95}), paired on one
  seed pool, batch_size=1 (exact per-episode replan counting).
- **verdict** — the pre-registered kill-gate: does the gate **beat fixed-frequency
  at a matched replan budget**? Only gate points whose budget is *within* the
  measured fixed-curve range are eligible; a PASS needs Δ>0 **and** a significant
  paired exact-McNemar win (p<0.05).

## Signal is real — the gate does not clear the bar
Staleness head is highly predictive: held-out `val_corr = 0.789 (push)`,
`0.777 (plate)`.

### push-v3 (val_corr 0.789) — beats_fixed = **False**
| arm | replan budget | success |
|---|---|---|
| base h=50 | 5.85 | 55.0% |
| **fixed h=25** | **7.78** | **72.5%** |
| fixed h=15 | 14.97 | 65.0% |
| fixed h=10 | 22.90 | 62.5% |
| fixed h=8 | 27.82 | 65.0% |
| gate τ_q0.95 | 14.70 | 60.0% — in-range, Δ vs fixed curve **−5.3pp**, McNemar p=0.80 (g7/f9) |
| gate τ_q0.90 | 17.02 | 75.0% — in-range, Δ vs fixed curve **+10.7pp**, McNemar p=0.42 (g9/f5) |
| gate τ_q0.80 | 70.05 | 57.5% — OUT of range (overspends) |

Best in-range gate (q0.90): **+10.7pp but p=0.42 (NOT significant)** → fails the
kill-gate. Note the gate's nominal +10.7pp is only vs the *interpolated* fixed
curve at budget 17; the **best actual fixed arm (h=25) already gets 72.5% at 7.8
replans** — so a practitioner just picks h=25 and matches the gate at <½ the cost.

### plate-slide-v3 (val_corr 0.777) — beats_fixed = **False**
| arm | replan budget | success |
|---|---|---|
| base h=50 | 6.00 | 50.0% |
| **fixed h=25** | **3.00** | **100.0%** |
| fixed h=15 | 4.33 | 100.0% |
| fixed h=10 | 6.35 | 100.0% |
| fixed h=8 | 8.00 | 100.0% |
| gate τ_q0.95 | 6.53 | 45.0% — in-range, Δ vs fixed curve **−55pp**, McNemar p=0.0001 (g0/f22) |
| gate τ_q0.90 | 8.38 | 97.5% — OUT of range |
| gate τ_q0.80 | 26.65 | 92.5% — OUT (3× the budget for *worse* than fixed) |

Best in-range gate (q0.95): **−55pp, p=0.0001** — the gate, when held to a tight
budget, actively **misplaces** replans and tanks to 45% where fixed gets 100% at
3 replans. Clear loss.

## Verdict (verbatim from verdict_n40.json)
> **Gate beats fixed-frequency at matched budget on 0/2 tasks. KILL-GATE NOT
> PASSED — where it only ties, the gate collapses to the known receding-horizon
> knob.**

The motivation (open-loop execution gap) remains real and replicated — base→best
fixed: push +17.5pp, plate +50pp. But the *trained contribution* on top does
**not** beat fixed-frequency at equal compute on either task.

## Two honest findings the real data forced
1. **Push fixed curve is NON-MONOTONIC:** h25=72.5% > h15=65% > h10=62.5%.
   Replanning *more* than h=25 actively hurts on push at this seed. So the gate is
   chasing a curve where "more replans" already isn't the win — and at low τ it
   just overspends (q0.80 = 70 replans, 57.5%).
2. **`avg_replans` is confounded with episode length.** Success terminates an
   episode early (fewer replans); failures run to the 500-step cap (more). So a
   higher-success arm can show a *lower* budget (e.g. plate fixed-h25 = 100% at
   only 3.0 replans, vs base = 50% at 6.0). The matched-budget interpolation still
   holds *within* the fixed-frequency family (same policy), and the verdict
   (gate ties/loses) is robust to it, but absolute budgets across arms of
   different success rates should be read as replans-per-*episode*, not effort.

## Why it failed (mechanism)
The gate's target — action **staleness** ‖fresh−committed‖ — is highly
predictable but is **not where replanning changes the outcome**. The plate q0.95
result (0/22 discordant pairs, p=0.0001) shows it places replans in the wrong
moments under a tight budget.

## What a v2 would change
- Retarget on an **outcome-linked** signal (predicted task-progress stall / value
  drop), not action distance.
- Report budget as **replans-per-executed-step** (removes the length confound).
- Add a budget regularizer so gate points land *inside* the fixed-curve range
  (push q0.90 and plate q0.90 currently straddle the boundary).
- Test on longer-horizon, multi-stage tasks where replan *timing* has leverage.

## Reproduce
```
cd src
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# fast: all 18 arms as concurrent processes (uses VRAM headroom) -> verdict:
NEP=40 bash parallel_sweep.sh        # writes gate_artifacts/{sweep_*,verdict}.json, DONE3
# or one single-process sweep per task:
python3 gate.py sweep --task push-v3 --gate gate_artifacts/push-v3_gate.pt \
  --fixed 25,15,10,8 --gate_q 0.80,0.90,0.95 --n_episodes 40 --seed 1000 \
  --out gate_artifacts/sweep_push-v3.json
python3 gate.py verdict --sweeps gate_artifacts/sweep_*.json --out gate_artifacts/verdict.json
# from scratch (collect + train + sweep + verdict): bash orchestrate.sh
```
Artifacts: `results/gate/{verdict_n40.json, sweep_push_n40.json,
sweep_plate_n40.json}`; data `src/gate_artifacts/*.npz`; gates
`src/gate_artifacts/{push-v3,plate-slide-v3}_gate.pt`.

**Run faithfulness:** 16/16 scored arms (the 2 lowest-τ arms timed out earlier and
were re-run at high τ where they're budget-comparable), all n=40, paired on seed
pool @1000; each arm a separate process writing its own json (no shared-file
races); verdict recomputed from the assembled sweep files.
