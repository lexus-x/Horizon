# Horizon — Full Session Record (`chat.md`)

> **What this is.** A curated, faithful record of the research session that produced the **Horizon** project — the conversation narrative (all user messages verbatim) plus every result, decision, and piece of information gathered. Written as the chat was being deleted, so this is the durable archive.
>
> **Date:** 2026‑05‑30 · **Researcher:** NAODE Research (islab.cwnu@gmail.com) · **Assistant:** Claude (Opus 4.8, 1M context).
>
> **Security note.** A GitHub Personal Access Token was pasted into the live chat. It is **redacted** here as `[GitHub PAT — REDACTED]`. Never commit live tokens.

---

## 0. One‑paragraph summary

The session began as a hunt for a **publishable, novel VLA method that improves success rate** on a frozen SmolVLA (flow‑matching) policy evaluated faithfully on MetaWorld. Four method ideas were built and **falsified** with cheap kill‑gates (CASTLE STL‑routing, STL‑GDPA guidance, NIAC test‑time averaging, NIAC dispersion‑predicts‑failure). The session then found its **first verified positive**: the **open‑loop execution gap** — a frozen SmolVLA executes 50‑step action chunks open‑loop, and simply replanning more often (closing the loop) lifts success by **+22pp (push)** and **+54pp (plate)**, paired and significant. That gain is from a *known* knob, so the proposed *contribution* is a **trained, compute‑efficient adaptive replan‑gate** whose honest bar is to **beat fixed‑frequency replanning at a matched replan budget**. A delegated novelty scoop returned **fabricated arXiv IDs**; live verification showed the named "killer" (Dyno) does not exist and the real neighbors (BID, RTC, DCDP, Legato) each differ on a real axis — novelty is **thin but present**, not scooped. Everything was packaged into the GitHub repo **github.com/lexus-x/Horizon**.

---

## 1. Conversation narrative (chronological)

### Phase A — pre‑compaction (NIAC single‑task PoC)
The session arrived (via context compaction) mid‑work on **NIAC** (Noise‑Invariant Action Consistency): the idea that a frozen flow VLA's noise‑induced action‑chunk variance is harmful, and a trained consistency loss would reduce it → higher success. A light **expert‑only finetune** pipeline (`finetune_smolvla.py`, ~22% params trainable, VLM frozen) had been stood up. The user asked: *"is there any ways that i can get the result of the NIAC now? i need to attach proof for proposal."* The assistant refused to fabricate a result and laid out the fastest honest path.

**User:** *"i have less than 2 hours left"*
→ Pivoted to the fastest honest test of NIAC's premise that needs **no training**: a **test‑time noise‑averaging probe** (denoise K times, average the chunks; if averaging out the noise helps, the premise holds).

**User:** *"use dynamic workflow"*
→ Re‑ran the probe as a parallel **dynamic Workflow**: 7 eval configs → analyze (Wilson CIs) → 3 adversarial verify lenses → honest synthesis.

**Result — NIAC test‑time: NULL.** The workflow caught a real stats bug in the analysis (an unpaired‑CI + mismatched‑n comparison had faked a "+2pp"). Corrected **paired McNemar**: push **p=0.885**, plate **p=0.664**. On a matched 50‑seed pool, base (66%) ≥ every averaging arm, and **mean‑K4 == medoid‑K4 == 58%** (no variance‑reduction mechanism). All 3 lenses rejected the claim.

**User:** *"what is the written proposal?"* → shown the honest‑null proposal doc.
**User:** *"what about the novelty and contribution?"* → honest: trained NIAC ≈ Consistency Policy (thin), and the premise was now unsupported.
**User:** *"is there any ways…"* → assistant proposed the decisive go/no‑go.

**User (decision):** *Run the decisive premise test.*
→ **Dispersion → failure AUROC probe** (`dispersion_probe.py`): act with the faithful base draw, measure K=8 candidate dispersion per episode, test whether dispersion predicts failure.

**Result — NIAC premise: DEAD.** push AUROC mean **0.06** (backwards — high spread = *success*), best 0.53; plate mean 0.57, max 0.79 but `early`=0.29 (a *lagging* symptom, not a leading signal); tasks disagree. No robust, leading, generalizable signal. NIAC fully falsified (4th swing).

**User (decision):** *Relax the schedule — propose a trained method as a plan.*
→ Wrote `PROPOSAL_trained_method.md` (scaffold). Candidate A (recoverability/replanning) had a **cheap train‑free pre‑test**: does shrinking the execution horizon (`n_action_steps`) raise success?

**THE POSITIVE.** Horizon sweep on push: **h50=56% → h25=76% → h10=78%** (monotone). Paired McNemar h50→h10 **p=0.035** (17 helped / 6 hurt — asymmetric, real). Generalization on plate: **h50=46% → h10=100%** (27 helped / 0 hurt, **p≈1e‑8**). Verified the override was correct per file (not a labeling bug). First positive of the session.

**User:** *"i need a fully documented github repo … architecture images and other visualizations … use dynamic workflow … github.com/lexus-x/Horizon"*
→ Launched a **dynamic Workflow** (5 parallel build agents + assemble) that produced README, docs, 7 figures, curated src/results, and a local git commit.

**User (pasted a delegated scoop):** verdict **"SCOOPED by Dyno (arXiv:2409.10549)"**, *"BUT can't I use that as my reference paper even if the novelty is thin?"*
→ Assistant **verified the arXiv IDs live** (memory: this external AI has fabricated IDs before). **2409.10549 = a stakeholder‑policy paper (not "Dyno"); 2409.14408 = a quantum‑field‑theory paper (not BID).** The "killer" was **hallucinated**. Real neighbors found and read: **BID 2408.17355** (training‑free), **RTC 2506.07339** (training‑free async), **DCDP 2603.01953** (trains action‑*correction* modules), **Legato 2602.12978** (training‑time *smoothness*). None is the exact "trained replan‑gate on a frozen VLA."

**User:** *"so you're saying this is a truly novel contribution that I can propose with proof?"*
→ Honest correction: **No.** Not scooped ≠ novel. Status = **verified motivation, thin‑but‑present novelty, unbuilt method.** Proof exists for the *gap* (known knob), not for the *method*.

**User (second scoop, AI admits it can't verify and guesses from acronyms):** verdict *"SCOOPED by DCDP."*
→ Assistant read DCDP's and Legato's **actual abstracts**: DCDP **corrects the actions** (dynamics encoder + cross‑attention), it does **not** learn a *when‑to‑replan* gate; Legato is **training‑time smoothness**. The "SCOOPED by DCDP" verdict was based on a wrong guess. **Not scooped by the four.**

**User (decision):** *"i don't care if the novelty is thin as long as it's barely novel and contributes to the success rate."*
→ Locked the honest contribution bar: **a learned/adaptive replan‑gate must beat *fixed‑frequency* replanning at a *matched replan budget*** (smarter *when*, not just *more often*). Corrected all repo citations to the verified IDs; framed DCDP as the cite‑and‑differentiate nearest neighbor; removed the fabricated "Dyno."

**User:** provided the PAT (`[GitHub PAT — REDACTED]`) and *"push asap."*
→ Pushed to **github.com/lexus-x/Horizon** (remote had an auto‑init README; merged with ours winning; token used transiently, **not** persisted in `.git/config`, no remote stored).

**User:** *"make a chat.md … push it under a folder named 'con:' … the whole chat and information gathered."*
→ This file.

---

## 2. Information gathered (the durable knowledge)

### 2.1 Setup / base
- **Model:** frozen **SmolVLA** checkpoint `smolvla_metaworld`, flow‑matching action expert (chunk_size=50, `n_action_steps=50` default = open‑loop, 10 Euler denoise steps, max_action_dim=32, real action 4‑dim), run via **lerobot** on **MetaWorld**.
- **Faithful harness:** `harness_common.py` `build_env_and_policy()` bakes the rename‑map fix (`preprocessor_overrides={"rename_observations_processor":{"rename_map":{}}}`) so numbers reproduce lerobot's. `MUJOCO_GL=egl`; python `/home/user/miniconda3/envs/lerobot/bin/python`.

### 2.2 Base survey (Gate A — measurable mid‑band)
Frozen `smolvla_metaworld`, 3 seeds, Wilson CIs (`results/base_survey_metaworld.json`):

| task | base success | band |
|---|---|---|
| drawer‑open‑v3 | 66.7% | mid |
| push‑v3 | 57.8% | mid (stable, primary) |
| window‑open‑v3 | 50.0% | mid |
| plate‑slide‑v3 | 46.7% | mid (stable, primary) |
| peg‑insert‑side‑v3 | 40.0% | mid (contact‑rich) |
| reach‑v3 | 27.8% | **floored outlier** |
| pick‑place‑v3 | 7.8% | floored |
| button / handle‑press / coffee‑button | 93 / 99 / 100% | ceilinged |

**Key reframe:** reach‑v3 (the task every prior "dead" verdict used) is a floored outlier; the same checkpoint sits 40–67% on 5 other tasks.

### 2.3 THE POSITIVE — open‑loop execution gap (verified)
Shrinking the execution horizon (`--n_action_steps`, replan more often → closed loop). n=50, single seed pool (1000–1049), paired:

| task | h=50 | h=25 | h=10 | paired McNemar (h50→h10) |
|---|---|---|---|---|
| push‑v3 | 56% | 76% | **78% (+22pp)** | helped 17 / hurt 6, **p=0.035** |
| plate‑slide‑v3 | 46% | — | **100% (+54pp)** | helped 27 / hurt 0, **p≈1e‑8** |

Monotone in horizon. Verified per‑file (override line confirmed; successes recounted from `per_episode`). **Mechanism:** the base fails by **confident open‑loop commitment** — a confident‑but‑wrong 50‑step chunk is unrecoverable for 50 steps; closing the loop lets it correct. Files: `results/horizon_{push,plate}_h{50,25,10}.json`.

**Honest ceiling:** this train‑free knob is **known** (receding‑horizon / ACT temporal ensembling / BID / RTC); it costs **~5× inference**. It is the *motivation + a positive number*, **not** the contribution.

### 2.4 The four falsified swings (the honesty trail)
1. **CASTLE** (STL‑temporal‑logic‑routed residual correction): routing **not load‑bearing** — at matched budget ρ‑routed == random; the residual's *coverage* drove the +30pp, not *where* STL placed it. Survives only as a failure **detector** (AUROC 0.87). FALSIFIED.
2. **STL‑GDPA** (analytic ρ‑gradient approach guidance): the win reproduced (+30pp) but the **wrong‑order** nudge (ignore "engage first," always go to goal) *beat* the ordered one → it's a trivial **oracle goal‑attractor / potential field**, nothing to do with temporal logic. FALSIFIED.
3. **NIAC test‑time averaging**: **NULL.** Paired McNemar push **p=0.885**, plate **p=0.664**; matched 50‑seed base (66%) ≥ all averaging arms; **mean‑K4 == medoid‑K4 == 58%**. Also caught a real **unpaired‑CI + mismatched‑n stats bug** that had faked a "+2pp." FALSIFIED.
4. **NIAC dispersion‑predicts‑failure**: push AUROC mean **0.06** (backwards), best 0.53; plate max 0.79 but a **lagging** symptom (early 0.29 backwards); tasks disagree. No robust leading signal. FALSIFIED.

→ Lesson that produced Horizon: train‑free wrappers can only *reselect* the base's own samples; a real capability gain needs the **execution loop** (and ultimately training).

### 2.5 Prior‑art verification (the fabricated‑scoop incident)
A delegated AI returned **"SCOOPED by Dyno (arXiv:2409.10549)."** Live checks:
- **2409.10549** → *"Confronting Conflicts to Yes: Untangling Wicked Problems with Open Design Systems"* (a stakeholder/policy paper). **"Dyno" does not exist** — fabricated.
- **2409.14408** (claimed "BID") → *"A Bekenstein‑type bound in QFT."* Wrong.

**Real, verified neighbors (abstracts read):**

| paper | arXiv | trained? | what it does | scoops the gate? |
|---|---|---|---|---|
| **BID** (Bidirectional Decoding) | 2408.17355 (ICLR 2025) | ❌ training‑free | test‑time forward‑backward chunk resampling | No |
| **RTC** (Real‑Time Chunking) | 2506.07339 | ❌ training‑free | async execution scheduling / inpainting | No |
| **DCDP** | 2603.01953 | ✅ trains modules | learned **action‑correction** (dynamics encoder + cross‑attn) on a frozen policy | No — different mechanism (corrects actions, not replan *timing*) |
| **Legato** (Native Continuation) | 2602.12978 | ✅ training‑time | reshapes flow training for chunk‑boundary **smoothness** | No |

→ **Not scooped** by these four. **DCDP** is the nearest *trained* neighbor and the cite‑and‑differentiate target. Novelty = **thin but present**, in a crowded/active area.

### 2.6 Proposed method + the honest bar
- **Contribution:** a **trained, lightweight adaptive replan‑gate** on the **frozen** flow‑matching VLA's expert features that decides **when** to replan, recovering the open‑loop‑gap success at **~1× compute**.
- **Differentiator:** vs DCDP (corrects actions) → Horizon **gates replan timing**; vs BID/RTC (training‑free) → Horizon **trains** a gate; vs Legato (training‑time smoothness) → Horizon targets **success at ~1× compute on a frozen policy**.
- **Kill‑gate (the bar the user set):** the learned gate must **beat fixed‑frequency replanning at a matched replan budget**. If it only ties → it collapses to the known knob, and the repo says so.
- **Status:** method **unbuilt**. Proof exists for the *gap* (motivation), not for the *method*.

### 2.7 Stats discipline (a selling point — it caught our own false positive)
Paired **McNemar** / bootstrap (not unpaired overlapping CIs), **one seed pool**, **matched n**, a **vanilla‑finetune control**, continuous metric (success + graded progress), and **power** (≥5 seed blocks; a true +5pp needs ~1,400 eps/arm at 80% power). The NIAC "+2pp" mirage is the cautionary example.

---

## 3. Repository map (github.com/lexus-x/Horizon)

```
Horizon/
├── README.md                  hero figure + honest TL;DR + Mermaid architecture
├── docs/
│   ├── PROPOSAL.md            the trained‑method proposal (scaffold)
│   ├── RESEARCH_LOG.md        the honest 4‑falsification decision trail
│   ├── METHODOLOGY.md         faithful harness + the paired‑stats discipline
│   └── ARCHITECTURE.md        Mermaid system diagram
├── figures/                   open_loop_gap, base_survey_midband, niac_null,
│                              dispersion_failure, mechanism, proposed_method,
│                              falsification_journey  (.png)
├── src/                       harness_common, cascade_eval, dispersion_probe,
│                              analyze_niac, base_survey, finetune_smolvla (.py)
├── results/                   all evidence JSONs (horizon_*, dispersion_*, base_survey, niac probe)
├── make_figures.py / make_diagrams.py
├── requirements.txt · LICENSE (MIT) · CITATION.cff · .gitignore
└── con:/chat.md               this record
```

**Reusable assets:** the faithful eval harness (`harness_common.py`, the rename‑map fix), the `cascade_eval.py` test‑time hooks (`--n_action_steps`, `mean`/`probe` modes), the dispersion‑AUROC probe, and the kill‑gate discipline that caught a real stats bug in real time.

---

## 4. Honest final verdict

> **Verified motivation, thin‑but‑present novelty, method unbuilt.** The open‑loop execution gap is real, paired, and replicated (+22pp / +54pp). It is *proposable* — framed as a measured gap on a language‑conditioned flow VLA plus a trained, compute‑efficient adaptive replan‑gate, positioned against DCDP/BID/RTC/Legato — but the word **"proof"** applies only to the gap until the gate is built and shown to beat fixed‑frequency replanning at a matched budget.

*End of record.*
