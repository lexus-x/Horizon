# Horizon — source

Curated copies of the faithful SmolVLA / MetaWorld harness used to produce the
Horizon evidence. The base model is a **frozen** SmolVLA flow-matching policy
(checkpoint `smolvla_metaworld`), run through lerobot on MetaWorld. None of these
scripts modify the base weights (except `finetune_smolvla.py`, which trains the
action expert only and is the proposed-method scaffold, not the frozen baseline).

## Files

| File | Role |
| --- | --- |
| `harness_common.py` | Faithful env + policy builder shared by every experiment. `build_env_and_policy()` constructs the MetaWorld vec-env and SmolVLA policy with the official lerobot pre/post-processors, including the documented faithfulness fix (empty `rename_map` override so the env's `observation.image` matches the patched single-camera config). Also provides MetaWorld privileged-state extractors (`get_ee_position_from_obs`, goal/object positions), a graded-progress metric `compute_graded_progress`, and `PerStepLogger` for offline analysis. |
| `cascade_eval.py` | Train-free test-time action-chunk selection driver. Defines `CascadeSmolVLA`, a `SmolVLAPolicy` subclass that swaps only the action-selection step (base / mean / consensus-medoid / proofreading / probe modes) while keeping the rollout loop bit-for-bit faithful. **This is also the script that owns the headline `--n_action_steps` execution-horizon override** (replan every N steps instead of the default 50 → open-loop vs closed-loop). Logs per-replan candidate dispersion. |
| `dispersion_probe.py` | NIAC premise test: acts with the faithful base draw but measures the per-episode dispersion of K i.i.d. flow draws and tests AUROC(dispersion → episode failure). Produces the `dispersion_*.json` evidence showing **no robust leading failure signal**. Imports `build` from `cascade_eval` (see note below). |
| `analyze_niac.py` | Loads the NIAC averaging result JSONs, computes Wilson 95% CIs, prints an honest verdict table (base vs mean-K vs consensus/medoid), and renders the NIAC-null figure. This is the analysis that recorded the **null** for test-time noise-averaging. |
| `base_survey.py` | Gate-A base-policy success survey across MetaWorld tasks. Pools per-episode successes across seeds and reports Wilson CIs to find the measurable mid-band (~30–75%) where a method's boost is reviewer-defensible. Produces `base_survey_metaworld.json`. |
| `finetune_smolvla.py` | Expert-only light finetune scaffold (VLM + vision encoder frozen, only the action expert trains). Vanilla flow-matching loss today; this is the control + starting point for the **proposed trained, compute-efficient closed-loop method**, not part of the frozen baseline. |

## Note on import paths

In the original research tree `cascade_eval.py` lived in a `cascade/`
subpackage, so `dispersion_probe.py` imports it as `from cascade.cascade_eval
import build`. In this curated layout both files sit flat in `src/`. To run
`dispersion_probe.py` here, change that import to `from cascade_eval import
build` (or run from a parent that still exposes the `cascade` package). The file
is kept verbatim to preserve provenance.

## Reproducing the headline (open-loop execution gap)

```bash
# default open-loop 50-step chunk vs closed-loop replan-every-10
python cascade_eval.py --task push-v3        --mode base --n_action_steps 50 --out ../results/horizon_push_h50.json
python cascade_eval.py --task push-v3        --mode base --n_action_steps 10 --out ../results/horizon_push_h10.json
python cascade_eval.py --task plate-slide-v3 --mode base --n_action_steps 50 --out ../results/horizon_plate_h50.json
python cascade_eval.py --task plate-slide-v3 --mode base --n_action_steps 10 --out ../results/horizon_plate_h10.json
```

Shrinking the execution horizon (more frequent replanning) yields the large
paired gain that motivates Horizon. Train-free this is a known knob (receding
horizon / temporal ensembling / Bidirectional Decoding) costing ~5× inference;
the proposed contribution is a trained, near-1× version.
