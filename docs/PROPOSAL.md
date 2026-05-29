# Proposal scaffold — a trained intervention for frozen-pretrained flow-matching VLAs

**Status: scaffold for a proposal (future work). The METHOD is a scoop-gated
hypothesis, not a validated result.** Honesty rules: every claim below is either
(a) measured this session, or (b) explicitly flagged as proposed/unverified.

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

## 6. Novelty gate (DELEGATED — run on a separate web AI before committing)

> "Adversarial novelty audit, live web + arXiv. CONTEXT (measured): a frozen
> flow-matching VLA (SmolVLA) executing 50-step action chunks open-loop loses
> +22pp (push) / +54pp (plate, MetaWorld) vs replanning every 10 steps — but that
> costs 5× inference. QUESTION: is it SCOOPED or UNCLAIMED to (A) train a
> **lightweight LEARNED REPLAN GATE / adaptive closed-loop controller** on a frozen
> pretrained VLA's expert features that recovers most of the closed-loop success
> gain at **near-open-loop (1×) compute** (i.e. learn *when* to replan, or distill
> the frequent-replan policy into one forward pass); (B) lightly finetune the
> expert for **compositional/sequential** MetaWorld success. For (A) you MUST
> directly classify and quote: **Bidirectional Decoding / BID (closed-loop action-
> chunk resampling)**, **real-time / asynchronous action chunking**, **temporal
> action ensembling (ACT)**, **learned early-termination / adaptive-horizon
> policies**, and any **VLA inference-time replanning** work — does any already do
> *trained, compute-efficient* adaptive replanning on a frozen VLA? For each: arXiv
> ID (verify it resolves), and the single nearest prior art a reviewer cites. Also
> position vs Recovery RL (2010.15920), A2C2 (2509.23224), Consistency Policy
> (Prasad 2024), DAgger, SmolVLA/OpenVLA/π0. Do not invent IDs. END WITH: UNCLAIMED
> or SCOOPED, and if SCOOPED name the exact paper that kills it."

## 7. Evaluation protocol (locked — the stats discipline that caught our own bug)

- **Paired** seeds; **McNemar / paired bootstrap**, NOT unpaired overlapping CIs.
- **One seed pool**, matched n across all arms (the bug that faked our +2pp).
- **Powered**: ≥5 seed blocks; report between-seed variance. (+5pp needs ~1,400
  eps/arm at 80% power — size accordingly or report as preliminary.)
- **Controls**: base, **vanilla-finetune at matched compute**, method-finetune.
  Continuous metric (success + graded progress), not threshold-straddling.
- **Kill-gate**: method must beat the vanilla-finetune control with non-overlapping
  CIs, else it's "more training," not a method.

## 8. Risks (honest)

- **Novelty**: the dominant risk. The method is unproven-novel until §6. If §6
  returns SCOOPED on A and B, this path does not yield a publishable method and we
  must relax a different constraint.
- **Effect size**: prior probes returned null/small; the trained gain may also be
  small. Powered eval or it's unfalsifiable.
- **Single-base generality**: results on `smolvla_metaworld` may not transfer.
