# Proposal scaffold — a trained intervention for frozen-pretrained flow-matching VLAs

**Status: scaffold for a proposal (future work). The METHOD is a hypothesis with
thin-but-present novelty (§6), not a validated result.** Honesty rules: every claim
below is either (a) measured this session, or (b) explicitly flagged as
proposed/unverified. Prior-art citations (§6) were verified against live arXiv
abstracts.

---

## 1. Problem

Pretrained VLAs (SmolVLA, flow-matching action expert) are strong but plateau on
manipulation: on a faithfully-evaluated `smolvla_metaworld` base, success sits in
a **measurable mid-band** (push 58–64%, drawer 67%, window 50%, plate-slide
47–58%), with floored outliers (reach 28%, pick-place 8%). The headroom is real
and reviewer-defensible. **Train-free test-time fixes do not close it** (see §3).

## 2. Preliminary findings (measured this session — the empirical anchor)

1. **Measurable testbed (Gate A passed).** Faithful lerobot harness; 12-task base
   survey, 3 seeds, Wilson CIs (`results/base_survey_metaworld.json`). 5 tasks in
   the 40–67% mid-band where a gain is detectable. reach-v3 (the task every prior
   "dead" verdict used) is a floored outlier.
2. **Train-free interventions are exhausted (4 falsifications).** STL-routed
   correction (routing not load-bearing; reduces to coverage), STL-gradient
   guidance (collapses to an oracle goal-attractor), test-time noise-averaging
   (paired McNemar p=0.885 push / 0.664 plate — null; mean = medoid), and
   dispersion→failure prediction (no robust leading signal). → **A capability gain
   needs *training*, not a wrapper.**
3. **Failure-mode signal — confident open-loop commitment.** On **push-v3**,
   failed episodes show **lower** action-sampling diversity than successful ones
   (AUROC 0.94 in the success direction): the base fails by *confident
   commitment*. (Weaker on plate's mean feature — task-dependent as a *predictor*.)
4. **THE ANCHOR — the open-loop execution gap (verified, both tasks).** SmolVLA
   executes 50-step chunks **open-loop** (`n_action_steps=50`). Shrinking the
   execution horizon (replan more often; closed-loop) gives a large, **paired**,
   replicated gain:

   | task | h=50 (default) | h=10 | helped/hurt | McNemar p |
   |---|---|---|---|---|
   | push-v3 | 56% | 78% (+22pp) | 17 / 6 | 0.035 |
   | plate-slide-v3 | 46% | 100% (+54pp) | 27 / 0 | ~1e-8 |

   Monotone in horizon (push 56→76→78 at h=50/25/10). Mechanism matches §2.3:
   confident wrong chunks are unrecoverable for 50 open-loop steps; closing the
   loop lets the base correct. *(n=50, single seed pool — to be powered; plate
   100% likely varies by pool but direction/size are overwhelming.)*

   **Honest ceiling:** this closed-loop knob is KNOWN — replanning more often is
   plain receding-horizon / temporal ensembling, and BID (arXiv:2408.17355) and
   RTC (arXiv:2506.07339) already exploit chunk overlap test-time-free. It is the
   **motivation + a positive preliminary number**, NOT the contribution. The cost
   is **5× inference** (replan every 10 vs 50 steps) — which is exactly what a
   trained gate must remove. The raw success gain from closing the loop is a known
   knob; the contribution is a *learned gate that beats fixed-frequency replanning
   at a matched replan budget* (§4, §6).

## 3. Why train-free is not enough (the negative result that motivates training)

A wrapper/router/re-ranker can only *reselect* the base's own samples; it cannot
add capability. We measured this directly (proofreading/consensus re-ranking
under-performed base; STL routing == random at matched budget). The gain has to
come from **changing the weights**, cheaply (expert-only, frozen VLM).

## 4. Proposed method (novelty thin-but-present — positioned in §6)

All candidates: **light, expert-only finetune of the frozen-VLM SmolVLA** (≈22%
of params trainable — pipeline already stood up), single-task or small-task-set,
evaluated against a **vanilla-finetune control** (so the claim is "method > same
compute spent vanilla", not "method > base").

- **Candidate A — Efficient closed-loop / adaptive replanning (LEAD; pre-test
  PASSED).** The §2.4 pre-test already confirmed the open-loop gap is a large,
  replicated lever (+22pp push, +54pp plate) — but at **5× inference**. The
  contribution is to capture that gain **without** the 5× cost. Two trainable
  forms: (i) a **learned replan gate** — a small head on the frozen expert's
  features that fires a replan only when it predicts the current chunk is going
  off (target: match h=10 success at near-h=50 compute); (ii) **distill** the
  closed-loop (h=10) policy into a single h=50 forward pass. The
  **fixed-short-horizon (h=10) result is the baseline the method must match at
  lower compute**, not the contribution itself. The **raw closed-loop success gain
  (+22pp push, +54pp plate) is a KNOWN knob** (receding-horizon / temporal
  ensembling) and is NOT the contribution — the contribution is that a **learned
  gate must beat fixed-frequency replanning at a matched replan budget** (smarter
  *when-to-replan*, not just *more-often*). If it only ties fixed-frequency, it
  collapses to the known knob — that is the falsifiable kill-gate. **Reviewer
  attack / nearest prior art:** this is a crowded space, but the four nearest
  papers differ (see §6) — DCDP (arXiv:2603.01953) trains modules that *correct
  the actions*, not the replan timing; BID (arXiv:2408.17355) and RTC
  (arXiv:2506.07339) are *training-free* test-time chunking; Legato
  (arXiv:2602.12978) is *training-time* boundary smoothness. Novelty is **thin but
  present** and positioned in §6.
- **Candidate B — Compositional / sequential-task finetuning.** The base is
  floored exactly on multi-step tasks (pick-place 8%). Finetune toward
  compositional success (subtask-conditioned or curriculum). Blends with the one
  *verified-novel* cell we own (compositional MetaWorld×SmolVLA). **Reviewer
  attack:** hierarchical/language-subgoal VLA is crowded — scoop hard.
- **Candidate C — (only if §6 finds an empty cell)** a flow-matching-expert
  training objective not yet ported to VLAs. **Reviewer attack:** MeanFlow/
  consistency/shortcut variants are already scooped — high risk.

> The proposal commits to ONE candidate (Candidate A is the lead) and positions it
> against the four nearest papers in §6. Novelty is **thin but present** — narrow,
> but not scooped by DCDP / BID / RTC / Legato.

## 5. Feasibility (real, on our infra)

- Faithful eval harness + processors (rename-map fix) — reproduces lerobot's
  numbers. `harness_common.py`, `cascade/cascade_eval.py`.
- Expert-only finetune pipeline (`finetune_smolvla.py`) — VLM frozen, ~22% params
  trainable. *(One bug to fix: `policy.forward` returns a tuple, not a dict —
  trivial.)*
- `lerobot/metaworld_mt50` dataset; single-GPU; runs in minutes.

## 6. Prior-art positioning (verified against live abstracts)

The novelty here is **thin but present**: the action-chunking / replanning space is
crowded and active, but none of the four nearest papers do *trained, compute-
efficient, when-to-replan gating on a frozen VLA*. The four IDs below were checked
against **live arXiv abstracts** — a prior delegated scoop had hallucinated paper
IDs (including a fabricated "Dyno" citation), which is why every ID here resolves.

**Nearest *trained* neighbor — the cite-and-differentiate paper:**

- **DCDP** — arXiv:2603.01953, *Closed-Loop Action Chunks with Dynamic Corrections
  for Training-Free Diffusion Policy*. Despite "training-free" in its title for the
  *base* policy, DCDP **trains** correction modules (a self-supervised dynamics
  feature encoder + cross-attention + an asymmetric encoder-decoder) as a wrapper
  on a frozen diffusion policy. Its mechanism is **learned action-correction**: it
  *corrects the actions* using dynamics features. **Differentiator:** Horizon's gate
  decides **replan *timing*** (when to re-invoke the frozen policy) and is
  success/compute-oriented — it does not edit the actions. Different lever, same
  frozen-policy-wrapper setting; this is the paper a reviewer cites and the one to
  cite-and-differentiate.

**Training-free test-time methods (Horizon *trains* a gate; these do not):**

- **BID** — arXiv:2408.17355, *Bidirectional Decoding: Improving Action Chunking via
  Guided Test-Time Sampling* (ICLR 2025). **Training-free** forward-backward
  resampling of the action chunk at test time. No trained component.
- **RTC** — arXiv:2506.07339, *Real-Time Execution of Action Chunking Flow Policies*
  (Real-Time Chunking). **Training-free** asynchronous execution scheduling /
  inpainting the overlap region. No trained component.

**Training-time method (Horizon adds a frozen-policy gate; this reshapes training):**

- **Legato** — arXiv:2602.12978, *Learning Native Continuation for Action Chunking
  Flow Policies*. A **training-time** method that reshapes flow training for
  chunk-boundary **smoothness / continuity** — not success-rate recovery, not a
  frozen-policy add-on gate.

**Verdict: UNCLAIMED by these four, but narrow.** Horizon's lever (learn *when* to
replan on a frozen flow-matching VLA, judged against fixed-frequency replanning at a
matched budget) is not done by DCDP (corrects actions), BID/RTC (training-free), or
Legato (training-time smoothness). The area is crowded and moving fast, so the
novelty bar is thin — which is why the §7 kill-gate (beat fixed-frequency at matched
budget) is the load-bearing claim, not the raw closed-loop gain. For broader context
also note Recovery RL (arXiv:2010.15920) and A2C2 (arXiv:2509.23224); Consistency
Policy (Prasad 2024) is relevant only to the dead NIAC averaging idea (§3), not to
the replan-gate.

## 7. Evaluation protocol (locked — the stats discipline that caught our own bug)

- **Paired** seeds; **McNemar / paired bootstrap**, NOT unpaired overlapping CIs.
- **One seed pool**, matched n across all arms (the bug that faked our +2pp).
- **Powered**: ≥5 seed blocks; report between-seed variance. (+5pp needs ~1,400
  eps/arm at 80% power — size accordingly or report as preliminary.)
- **Controls**: base, **vanilla-finetune at matched compute**, method-finetune.
  Continuous metric (success + graded progress), not threshold-straddling.
- **Kill-gate (primary)**: the learned gate must **beat FIXED-FREQUENCY replanning
  at a matched replan budget** (same number of replans, smarter placement). The raw
  closed-loop gain is a known knob (receding-horizon); only *smarter when-to-replan*
  is the contribution. If the gate merely ties fixed-frequency, it collapses to the
  known knob — kill it.
- **Kill-gate (secondary)**: method must also beat the vanilla-finetune control with
  non-overlapping CIs, else it's "more training," not a method.

## 8. Risks (honest)

- **Novelty**: the dominant risk. Novelty is **thin but present** (§6) — not
  scooped by the four nearest papers (DCDP / BID / RTC / Legato), but the area is
  crowded and active, so the margin is narrow. The contribution stands or falls on
  the §7 kill-gate (the learned gate must beat fixed-frequency replanning at a
  matched budget); if it only ties, it collapses to the known closed-loop knob and
  this path does not yield a publishable method.
- **Effect size**: prior probes returned null/small; the trained gain may also be
  small. Powered eval or it's unfalsifiable.
- **Single-base generality**: results on `smolvla_metaworld` may not transfer.
