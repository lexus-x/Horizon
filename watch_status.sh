#!/bin/bash
RESULTS=/home/user/Desktop/vla_projects/RESEARCH/active/horizon/results

while true; do
  clear
  echo "=== HORIZON STATUS  $(date '+%H:%M:%S') ==="
  echo ""

  # Training
  echo "[ TRAINING ]"
  if [ -f "$RESULTS/student_push_l2_0p1_s300/config.json" ]; then
    echo "  push L2=0.1  300 steps   DONE"
  else
    LAST=$(grep "step " "$RESULTS/distill_push_l2_0p1_s300.log" 2>/dev/null | tail -1)
    STEP=$(echo "$LAST" | grep -oP 'step\s+\K[0-9]+')
    LOSS=$(echo "$LAST" | grep -oP 'loss=\K[0-9.]+')
    SPEED=$(echo "$LAST" | grep -oP '\((\K[0-9.]+)(?=ms/step)')
    if [ -n "$STEP" ] && [ -n "$SPEED" ]; then
      REM=$(( (300 - STEP) * SPEED / 1000 ))
      echo "  push L2=0.1  step $STEP/300  loss=$LOSS  ETA ~${REM}s"
    else
      echo "  push L2=0.1  loading..."
    fi
  fi

  echo ""
  echo "[ EVAL ]"
  RFILE="$RESULTS/horizon_push_l2_0p1_s300.json"
  ELOG="$RESULTS/eval_push_l2_0p1_s300.log"
  if [ -f "$RFILE" ]; then
    SR=$(python3 -c "import json; d=json.load(open('$RFILE')); print(f\"{d['result']['pc_success']:.0f}\")" 2>/dev/null)
    echo "  push L2=0.1  DONE  SR=${SR}%"
  elif [ -f "$ELOG" ]; then
    # count completed episodes from tqdm-style lines
    EP=$(grep -oP 'running_success_rate' "$ELOG" 2>/dev/null | wc -l)
    SR_NOW=$(grep -oP 'running_success_rate=\K[0-9.]+' "$ELOG" 2>/dev/null | tail -1)
    echo "  push L2=0.1  ep ~${EP}/50  live SR=${SR_NOW}%"
  else
    echo "  push L2=0.1  waiting..."
  fi

  echo ""
  echo "[ TARGET ]"
  echo "  push base   56%   <- must stay >= this"
  echo "  push noL2   28%   (collapsed — what we are fixing)"
  echo "  push loop   78%   (5x compute ceiling)"
  echo ""
  echo "  GOAL: L2 result >= 56%"

  sleep 5
done
