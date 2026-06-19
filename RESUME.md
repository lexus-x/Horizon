# Horizon Project — Resume Context
**Last updated: 2026-06-20**

## What this project is
Frozen SmolVLA-0.45B × MetaWorld. Studies execution horizon (chunk size) effects.
**Core finding**: closed-loop distillation (teacher h=10 → student h=50) improves success
on *shape* tasks but collapses on *reactive* tasks.

## Environment setup
```bash
conda activate lerobot   # /home/user/miniconda3/envs/lerobot
pip install draccus==0.10.0   # required — was missing, caused ModuleNotFoundError
cd /path/to/horizon/src
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
```

## Base checkpoint
**Required on the new machine:**
`/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld`

This is the frozen SmolVLA checkpoint. Transfer it or pull from HuggingFaceVLA/smolvla_metaworld.
Set `CKPT` in `src/distill_smolvla.py` and `BASE` in all shell scripts to point to it.

## Current results (as of 2026-06-20)

### Completed distillation (6 tasks)
| Task               | Base h50 | Loop h10 | Distill h50 | Status |
|--------------------|----------|----------|-------------|--------|
| plate-slide-v3     | 46%      | ~86%     | 86%         | GREEN  |
| drawer-open-v3     | 58%      | 100%     | 100%        | GREEN  |
| window-open-v3     | 42%      | ~70%     | 68%         | GREEN  |
| peg-insert-side-v3 | 20%      | ~40%     | 26%         | GREEN  |
| pick-place-v3      | 8%       | ~20%     | 14%         | GREEN  |
| push-v3            | 56%      | 78%      | 28%         | RED    |

Result files: `results/horizon_<short>_h50.json`, `results/horizon_<short>_student.json`

### In-progress distillation (14 tasks — running on old machine)
Pipeline script: `src/breadth_new14.sh`
Running as background job. Each task takes ~2 hrs (collect → distill → eval).
These are **midband tasks** (base SR 30–75%) found by the full MetaWorld survey.

| Task               | Base SR | Survey result |
|--------------------|---------|---------------|
| window-close-v3    | 73%     | results/base_survey_remaining.json |
| door-lock-v3       | 65%     | |
| sweep-into-v3      | 63%     | |
| reach-wall-v3      | 63%     | |
| disassemble-v3     | 58%     | |
| sweep-v3           | 53%     | |
| push-back-v3       | 43%     | |
| hammer-v3          | 43%     | |
| bin-picking-v3     | 45%     | |
| shelf-place-v3     | 33%     | |
| coffee-pull-v3     | 33%     | |
| peg-unplug-side-v3 | 38%     | |
| box-close-v3       | 35%     | |
| assembly-v3        | 35%     | |

### Full MetaWorld base survey (all 50 tasks)
- Original 12 tasks: `results/base_survey_metaworld.json`
- Remaining 38 tasks: `results/base_survey_remaining.json`
- Total midband tasks found: 19 (5 original + 14 new)

## How to resume the 14-task distillation on a new machine

### Step 1: Transfer or re-collect teacher data
Teacher data is NOT in git (too large). Either:
- `rsync -avz old_machine:results/teacher_*_h10/ results/` for each task, OR
- Re-collect on new machine (takes ~20 min per task):
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
python -u collect_teacher.py --task window-close-v3 --n_action_steps 10 \
  --n_episodes 50 --seed 2000 --snap_stride 2 \
  --out results/teacher_window-close-v3_h10
```

### Step 2: Run distillation (per task)
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
python -u distill_smolvla.py \
  --shards results/teacher_window-close-v3_h10 \
  --success_only 1 --steps 1500 --batch_size 16 --lr 1e-4 \
  --out results/student_winclose
```

### Step 3: Eval student
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
python -u cascade_eval.py \
  --ckpt results/student_winclose \
  --proc_ckpt /path/to/ckpt_smolvla_metaworld \
  --task window-close-v3 --mode base \
  --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 \
  --out results/horizon_winclose_student.json
```

### Or run all 14 at once
```bash
bash src/breadth_new14.sh
```
(Runs 2 pipelines concurrently, 7 waves total)

## Key files
- `src/distill_smolvla.py` — student training (supports --l2_coeff regularization)
- `src/collect_teacher.py` — collect closed-loop teacher trajectories
- `src/cascade_eval.py` — eval any policy on MetaWorld (requires --proc_ckpt for students)
- `src/base_survey.py` — survey base SR across MetaWorld tasks
- `src/breadth_new14.sh` — full pipeline for 14 pending tasks
- `src/paired_mcnemar.py` — statistical tests

## Key findings so far
1. **Shape tasks**: distillation recovers closed-loop gain in single open-loop pass
2. **Reactive tasks** (push): distillation collapses (28% vs base 56%)
3. **OL-teacher control** (plate): CL teacher +42pp over OL teacher (p=5.7e-6)
   → confirms closed-loop *content*, not just trajectory length, is what's learned
4. **L2 regularization failed**: l2_coeff=0.1 gave 6% on push (worse than no-L2 28%)
   → penalty ~45× larger than task loss; reactive collapse is fundamental, not fixable by reg

## Notes on cascade_eval.py
Student checkpoints do NOT contain normalization stats. Always pass:
```
--proc_ckpt /path/to/ckpt_smolvla_metaworld
```
without this you get `ProcessorMigrationError: Config file 'policy_preprocessor.json' not found`.
