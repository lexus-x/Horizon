# Horizon — Methodology

How every number in this repo was produced, and **why the protocol is built the
way it is**. The short version: a faithful evaluation harness so the numbers are
reproducible, and a statistics discipline so a real effect cannot be confused with
noise. The motivating cautionary tale runs through this whole document — we once
reported a "+2pp" gain that was a pure artifact, and the protocol below is the set
of guardrails that killed it. See `RESEARCH_LOG.md` §3 for that incident.

Source files (all in `RESEARCH/active/_harness-faithful-smolvla/`):
`harness_common.py`, `cascade/cascade_eval.py`, `analyze_niac.py`,
`dispersion_probe.py`, `base_survey.py`, `finetune_smolvla.py`,
`run_niac_probe.sh`.

---

## 1. Faithful evaluation harness

**Goal: reproduce lerobot's own SmolVLA numbers, so our baseline is the *real*
base and not a misconfigured strawman.** A weak harness manufactures fake headroom
— every "method" then looks good against a base we accidentally broke. The harness
exists to remove that failure mode before any method is tested.

We evaluate through lerobot's own machinery rather than a hand-rolled loop:

- **`lerobot.scripts.lerobot_eval.eval_policy`** drives the rollouts and aggregates
  `pc_success` / `avg_max_reward`. Using the official evaluator means our success
  counting matches lerobot's exactly (same success-latch semantics, same
  aggregation).
- **Official pre/post processors** via `make_pre_post_processors(...)` (policy side)
  and `make_env_pre_post_processors(...)` (env side). The action expert is
  extremely sensitive to input normalization and action de-normalization; using the
  checkpoint's *own* processors is what makes the policy behave as trained.
- **The rename-map fix (the load-bearing faithfulness correction).** The checkpoint
  ships a stale `rename_observations_processor` rename map that silently mis-wires
  observation keys for this MetaWorld setup. We override it to empty:

  ```python
  preprocessor_overrides={
      "device_processor": {"device": device},
      # Disable stale rename map (documented faithfulness fix)
      "rename_observations_processor": {"rename_map": {}},
  }
  ```

  *Why it matters:* with the stale map active, observations are routed to the wrong
  channels and success craters — that was the source of earlier "this base is dead"
  verdicts. With the map disabled, the harness reproduces lerobot's reported
  numbers. This single override is the difference between a reproducible baseline
  and a fictional one. (`harness_common.py:build_env_and_policy`.)

**Policy is FROZEN.** For all base/survey/probe/horizon results the SmolVLA weights
are loaded `from_pretrained` and run in `eval()`; nothing is trained. The only
"intervention" knobs are test-time (chunk selection mode, execution horizon),
implemented as a thin `SmolVLAPolicy` subclass in `cascade/cascade_eval.py` that
overrides action selection without touching weights.

---

## 2. MetaWorld setup

- **Env:** `MetaworldEnv(task=..., obs_type="pixels_agent_pos")` built with
  `make_env(..., use_async_envs=False)`. Synchronous vec env so seeding is exactly
  reproducible (async ordering would break the paired-seed guarantee in §3).
- **Tasks (`*-v3`):** the 12-task base survey covers
  reach, button-press, drawer-open, pick-place, peg-insert-side, push, door-open,
  window-open, handle-press, coffee-button, plate-slide, faucet-open. Method work
  focuses on the **mid-band** (40–67%): drawer-open, push, window-open, plate-slide,
  peg-insert. (`base_survey.py`, `results/base_survey_metaworld.json`.)
- **reach-v3 is a banned headline task.** It floors at 27.8% (Wilson [19.6, 37.8])
  — a floored *outlier*, yet it was the task every prior "dead base" verdict used.
  Method claims must live in the mid-band where a gain is actually detectable, not
  on a floored or ceilinged task.
- **Execution horizon.** Chunk size 50; default `n_action_steps=50` = fully
  **open-loop** (commit the whole chunk before replanning); 10 Euler denoise steps.
  The horizon sweep overrides `policy.config.n_action_steps` (e.g. 50 → 25 → 10) to
  vary how often the loop closes (`cascade_eval.py`, `--n_action_steps`).
- **Privileged state / continuous metric.** MetaWorld's 39-dim fully-observable
  state gives EE, object, and goal positions; `compute_graded_progress(...)` turns
  these into a graded `g_t ∈ [0,1]` (reach: normalized EE→goal; manipulation:
  0.3·approach + 0.7·place). This is the continuous metric of §3.

---

## 3. Statistics discipline (the guardrails, and what each one prevents)

This is the selling point of the repo: the same discipline that gates our claims
**caught our own false positive** before it left the building. Each rule below is
paired with the specific way a result can lie, and the NIAC bug is the worked
example.

**The motivating bug (NIAC averaging, `RESEARCH_LOG.md` §3).** An early pass
reported test-time noise-averaging as "+2pp" over base. It was an artifact of two
shortcuts: (a) the arms were compared via **unpaired, overlapping Wilson CIs**, and
(b) they were run at **mismatched n** (base at one episode count, mean-K at
another). Re-running under the protocol below — one seed pool, matched n, paired
McNemar — the +2pp **vanished** (paired McNemar push p = 0.885, plate p = 0.664;
base ≥ every averaging arm on the matched pool). Every rule here is a direct
descendant of that failure.

### 3.1 Paired McNemar / paired bootstrap — NOT unpaired overlapping CIs
We compare arms episode-by-episode on the **same seeds** and test the *paired*
disagreement (McNemar on the helped/hurt counts), e.g. the headline horizon result
push h50→h10 = **helped 17 / hurt 6, p = 0.035**; plate **helped 27 / hurt 0,
p ≈ 1e-8**.
*Why:* MetaWorld seeds vary enormously in difficulty, so most of the variance is
*between seeds*, shared across arms. Unpaired CIs throw that shared structure away
and can both (i) **hide** a real effect (overlapping CIs on a true win) and
(ii) **manufacture** a fake one — the +2pp artifact was an unpaired-CI mirage.
Pairing removes the between-seed variance that the unpaired test mistakes for
signal.

### 3.2 One seed pool, matched n across all arms
Every arm runs the **same start seed and the same number of episodes**
(`run_niac_probe.sh`, `start_seed=1000`; all modes share the pool by construction
in `cascade_eval.py`).
*Why:* mismatched n was the *second* half of the +2pp artifact — an arm that ran
on an easier or smaller seed slice looks better for free. Matched n on one pool
makes every comparison apples-to-apples; without it, "improvement" can be pure
sampling luck. The consensus/medoid control in the NIAC probe relies on this:
**mean-K4 == medoid-K4 == 58%** at matched n=50 is only interpretable because the
two arms saw identical episodes.

### 3.3 Vanilla-finetune control at matched compute
A trained method is compared not against the base but against a **vanilla finetune
that spent the same compute** (`finetune_smolvla.py`, expert-only, VLM frozen,
~22% params trainable).
*Why:* "method beats base" conflates *the method* with *the extra training*. The
honest question is "method > the same compute spent naively." A method that only
matches the vanilla control is "more training," not a contribution — that is the
kill-gate (PROPOSAL §7). This is the one guardrail that does not yet have a *trained*
result behind it; it is locked in for the proposed method so the eventual claim is
attributable to the method and not to the gradient steps.

### 3.4 Continuous metric (success + graded progress), not a threshold straddle
Alongside binary success we report `avg_max_reward` and the graded `g_t` progress
(§2).
*Why:* binary success on a task sitting near a success threshold is a step function
— a method can flip a handful of borderline episodes and look transformative, or do
real partial-progress work and look like nothing. The continuous metric detects
movement that the binary latch hides, and prevents threshold-straddling from
inflating or erasing an effect.

### 3.5 Powered, and honest about it (≥5 seed blocks; ~1,400 eps/arm for +5pp)
We report between-seed variance (e.g. base survey seed-rate std per task) and size
the run to the effect. At 80% power, detecting a +5pp difference needs ≈1,400
episodes per arm.
*Why:* a small effect on a small n is unfalsifiable — it neither confirms nor
denies the claim. Anything under-powered is labeled **preliminary** rather than
asserted. The horizon anchor is explicitly flagged this way: n=50 on a single seed
pool, so the *direction and size* are overwhelming (especially plate, helped 27 /
hurt 0) but the exact percentages (e.g. plate 100%) will move with the pool. We
state that, rather than banking the 100%.

---

## 4. One-paragraph summary

Evaluate the **frozen** base through lerobot's own `eval_policy` and processors,
with the rename-map override that makes the numbers reproduce; work in the
**mid-band** MetaWorld tasks where a gain is detectable (never on floored reach-v3);
and judge every comparison with **paired McNemar on one matched seed pool, against a
vanilla-finetune control, on a continuous metric, with honest power**. The proof
that this discipline earns its keep is that it deleted our own +2pp false positive
before we shipped it.
