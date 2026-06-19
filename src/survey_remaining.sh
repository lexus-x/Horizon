#!/bin/bash
# Survey the 38 MetaWorld tasks not yet in base_survey_metaworld.json
# 2 seeds x 20 eps per task = faster screen; saves to base_survey_remaining.json
set -u
ROOT=/home/user/Desktop/vla_projects/RESEARCH/active/horizon
SRC=$ROOT/src
BASE=/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla/ckpt_smolvla_metaworld
P=/home/user/miniconda3/envs/lerobot/bin/python
LOG=$ROOT/results/survey_remaining.log

cd "$SRC"
echo "=== REMAINING SURVEY START $(date) ===" > "$LOG"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
$P -u base_survey.py \
  --ckpt "$BASE" \
  --tasks "assembly-v3,basketball-v3,bin-picking-v3,box-close-v3,button-press-topdown-v3,button-press-topdown-wall-v3,button-press-wall-v3,coffee-pull-v3,coffee-push-v3,dial-turn-v3,disassemble-v3,door-close-v3,door-lock-v3,door-unlock-v3,drawer-close-v3,faucet-close-v3,hammer-v3,hand-insert-v3,handle-press-side-v3,handle-pull-side-v3,handle-pull-v3,lever-pull-v3,peg-unplug-side-v3,pick-out-of-hole-v3,pick-place-wall-v3,plate-slide-back-side-v3,plate-slide-back-v3,plate-slide-side-v3,push-back-v3,push-wall-v3,reach-wall-v3,shelf-place-v3,soccer-v3,stick-pull-v3,stick-push-v3,sweep-into-v3,sweep-v3,window-close-v3" \
  --n_episodes 20 \
  --seeds "1000,2000" \
  --out "$ROOT/results/base_survey_remaining.json" \
  >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
