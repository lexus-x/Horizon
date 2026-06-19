#!/bin/bash
# Breadth sweep: test the distillation shape-vs-reactive boundary across 4 more tasks.
# Per task: base h50, loop h10 (the gap), teacher collect, distill, distill eval h50.
# Runs 2 task-pipelines concurrently (GPU budget). Logs -> logs/breadth_<short>.log
set -u
ROOT=/home/user/Desktop/vla_projects/RESEARCH/active/horizon
SRC=$ROOT/src
BASE=/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld
P=/home/user/miniconda3/envs/lerobot/bin/python
EGL="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1"
cd "$SRC"

pipeline () {
  local TASK=$1 SHORT=$2
  local LOG=$ROOT/logs/breadth_${SHORT}.log
  echo "=== [$SHORT] START $(date +%H:%M) ===" > "$LOG"
  # 1. base h50 (1x)
  env $EGL $P -u cascade_eval.py --ckpt $BASE --task $TASK --mode base \
     --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out ../results/horizon_${SHORT}_h50.json >> "$LOG" 2>&1
  # 2. loop h10 (5x) — the gap
  env $EGL $P -u cascade_eval.py --ckpt $BASE --task $TASK --mode base \
     --n_action_steps 10 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out ../results/horizon_${SHORT}_h10.json >> "$LOG" 2>&1
  # 3. teacher collect (closed-loop targets)
  env $EGL $P -u collect_teacher.py --task $TASK --n_action_steps 10 \
     --n_episodes 50 --seed 2000 --snap_stride 2 \
     --out ../results/teacher_${TASK}_h10 >> "$LOG" 2>&1
  # 4. distill student (1x), success-filtered
  env $EGL $P -u distill_smolvla.py --shards ../results/teacher_${TASK}_h10 \
     --success_only 1 --steps 1500 --batch_size 16 --lr 1e-4 \
     --out ../results/student_${SHORT} >> "$LOG" 2>&1
  # 5. distill eval h50 (1x)
  env $EGL $P -u cascade_eval.py --ckpt ../results/student_${SHORT} --proc_ckpt $BASE \
     --task $TASK --mode base --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out ../results/horizon_${SHORT}_student.json >> "$LOG" 2>&1
  echo "=== [$SHORT] DONE $(date +%H:%M) ===" >> "$LOG"
  echo "[$SHORT] pipeline done"
}

# Wave 1: shape candidates
pipeline drawer-open-v3 drawer &
pipeline window-open-v3 window &
wait
# Wave 2: reactive candidates
pipeline peg-insert-side-v3 peg &
pipeline pick-place-v3 pickplace &
wait
echo "BREADTH ALL DONE"
