#!/bin/bash
# Wait for all 6 L2-sweep checkpoints, then eval each with cascade_eval.py

RESULTS=/home/user/Desktop/vla_projects/RESEARCH/active/horizon/results
SRC=/home/user/Desktop/vla_projects/RESEARCH/active/horizon/horizon_v2/src
PY=/home/user/miniconda3/envs/lerobot/bin/python

declare -A TASK_MAP=(
  [push_v3]=push-v3
  [peg_insert_side_v3]=peg-insert-side-v3
  [pick_place_v3]=pick-place-v3
)

for SLUG in push_v3 peg_insert_side_v3 pick_place_v3; do
  TASK=${TASK_MAP[$SLUG]}
  for L2 in 0p1 1p0; do
    TAG="${SLUG}_l2_${L2}"
    CKPT="$RESULTS/student_${TAG}"

    # Wait for checkpoint
    echo "[$(date)] waiting for $CKPT ..."
    until [ -f "$CKPT/config.json" ]; do sleep 30; done
    echo "[$(date)] $CKPT ready — launching eval"

    OUT_JSON="$RESULTS/horizon_${SLUG}_l2_${L2}.json"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 \
      $PY $SRC/cascade_eval.py \
        --task ${TASK} \
        --mode base \
        --n_action_steps 50 \
        --n_episodes 50 \
        --seed 1000 \
        --ckpt "$CKPT" \
        --out "$OUT_JSON" \
      && echo "[$(date)] done -> $OUT_JSON" \
      || echo "[$(date)] FAILED: $TAG"
  done
done

echo "=== ALL EVALS DONE ==="
