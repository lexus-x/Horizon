#!/bin/bash
# Distillation pipeline for 14 new midband tasks found in remaining survey.
# Runs 2 pipelines concurrently per wave. GPU memory is sufficient to share.
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
  mkdir -p "$ROOT/logs"
  echo "=== [$SHORT] START $(date +%H:%M) ===" > "$LOG"

  # 1. base h50
  env $EGL $P -u cascade_eval.py --ckpt $BASE --task $TASK --mode base \
     --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out $ROOT/results/horizon_${SHORT}_h50.json >> "$LOG" 2>&1

  # 2. loop h10
  env $EGL $P -u cascade_eval.py --ckpt $BASE --task $TASK --mode base \
     --n_action_steps 10 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out $ROOT/results/horizon_${SHORT}_h10.json >> "$LOG" 2>&1

  # 3. collect teacher
  env $EGL $P -u collect_teacher.py --task $TASK --n_action_steps 10 \
     --n_episodes 50 --seed 2000 --snap_stride 2 \
     --out $ROOT/results/teacher_${TASK}_h10 >> "$LOG" 2>&1

  # 4. distill
  env $EGL $P -u distill_smolvla.py --shards $ROOT/results/teacher_${TASK}_h10 \
     --success_only 1 --steps 1500 --batch_size 16 --lr 1e-4 \
     --out $ROOT/results/student_${SHORT} >> "$LOG" 2>&1

  # 5. eval student
  env $EGL $P -u cascade_eval.py --ckpt $ROOT/results/student_${SHORT} --proc_ckpt $BASE \
     --task $TASK --mode base --n_action_steps 50 --n_episodes 50 --seed 1000 --batch_size 10 \
     --out $ROOT/results/horizon_${SHORT}_student.json >> "$LOG" 2>&1

  echo "=== [$SHORT] DONE $(date +%H:%M) ===" >> "$LOG"
  echo "[$SHORT] DONE"
}

# Wave 1: top shape candidates by base SR
pipeline window-close-v3    winclose   &
pipeline sweep-into-v3      sweepinto  &
wait

# Wave 2
pipeline sweep-v3           sweep      &
pipeline reach-wall-v3      reachwall  &
wait

# Wave 3
pipeline door-lock-v3       doorlock   &
pipeline disassemble-v3     disassemble &
wait

# Wave 4
pipeline hammer-v3          hammer     &
pipeline push-back-v3       pushback   &
wait

# Wave 5
pipeline bin-picking-v3     binpick    &
pipeline box-close-v3       boxclose   &
wait

# Wave 6
pipeline shelf-place-v3     shelfplace &
pipeline coffee-pull-v3     coffeepull &
wait

# Wave 7
pipeline peg-unplug-side-v3 pegunplug  &
pipeline assembly-v3        assembly   &
wait

echo "ALL 14 DONE"
