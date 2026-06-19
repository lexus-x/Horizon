#!/bin/bash
# Overnight autonomous run (2026-06-11 night). Sequential, GPU-polite (co-resident job ~33GB).
# Job 1 (kill-shot, robust): open-loop-teacher vs closed-loop-teacher distillation, plate + push.
#         Tests whether Horizon's gain is the CLOSED-LOOP SIGNAL or just BC-on-rollout-windows.
# Job 2 (novel probe): push curated distillation, start-predictable vs unpredictable stratum
#         (size-matched, 21 eps each) -> first look at "is the shape/reactive boundary movable".
set -u
ROOT=/home/user/Desktop/vla_projects/RESEARCH/active/horizon
SRC=$ROOT/src
BASE=/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld
P=/home/user/miniconda3/envs/lerobot/bin/python
EGL="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1"
LOG=$ROOT/logs/overnight.log
cd "$SRC"
echo "===== OVERNIGHT START $(date) =====" | tee "$LOG"

run () { echo ">>> [$(date +%H:%M)] $*" | tee -a "$LOG"; env $EGL $P -u "$@" >> "$LOG" 2>&1 || echo "!!! FAILED: $*" | tee -a "$LOG"; }

# ---------- JOB 1: open-loop-teacher control (plate, push) ----------
for pair in "plate-slide-v3:plate" "push-v3:push"; do
  TASK=${pair%%:*}; SHORT=${pair##*:}
  echo "===== JOB1 OL-teacher $SHORT $(date +%H:%M) =====" | tee -a "$LOG"
  # open-loop teacher = run base with full 50-step chunk (n_action_steps=50), record its windows
  run collect_teacher.py --task $TASK --n_action_steps 50 --n_episodes 50 --seed 2000 --snap_stride 2 --out ../results/teacher_${TASK}_OL
  run distill_smolvla.py --shards ../results/teacher_${TASK}_OL --success_only 1 --steps 1500 --batch_size 16 --lr 1e-4 --out ../results/student_${SHORT}_OL
  run cascade_eval.py --ckpt ../results/student_${SHORT}_OL --proc_ckpt $BASE --task $TASK --mode base --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 --out ../results/horizon_${SHORT}_studentOL.json
done

# ---------- JOB 2: push curated distillation (predictable vs unpredictable) ----------
for STRAT in pred unpred; do
  echo "===== JOB2 push curated $STRAT $(date +%H:%M) =====" | tee -a "$LOG"
  run distill_smolvla.py --shards ../results/teacher_push_${STRAT} --success_only 1 --steps 1500 --batch_size 16 --lr 1e-4 --out ../results/student_push_${STRAT}
  run cascade_eval.py --ckpt ../results/student_push_${STRAT} --proc_ckpt $BASE --task push-v3 --mode base --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 --out ../results/horizon_push_student_${STRAT}.json
done

echo "===== ALL JOBS DONE $(date) =====" | tee -a "$LOG"
$P -u summarize_overnight.py >> "$LOG" 2>&1
echo "===== SUMMARY WRITTEN $(date) =====" | tee -a "$LOG"
