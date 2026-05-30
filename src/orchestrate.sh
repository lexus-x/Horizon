#!/usr/bin/env bash
# Self-contained autonomous Horizon replan-gate pipeline:
#   collect -> train -> sweep (base / fixed-H / gate-tau) -> matched-budget verdict.
# Writes gate_artifacts/STATUS (human log) and gate_artifacts/DONE (completion
# marker) so a caller can wait without polling. Idempotent: wipes prior artifacts.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
A=gate_artifacts
L=$A/logs
S=$A/STATUS

# ---- knobs (env-overridable) ----
CEP=${CEP:-10}                 # collect episodes per task
CM=${CM:-2}                    # fresh draws averaged for staleness target
NEP=${NEP:-40}                 # eval episodes per arm
FIXED=(25 15 10)
GATEQ=(0.4 0.5 0.6 0.7)
TASKS=(push-v3 plate-slide-v3)

pkill -f "gate.py" 2>/dev/null; sleep 2
rm -rf "$A"; mkdir -p "$L"
echo "$(date +%T) START CEP=$CEP CM=$CM NEP=$NEP" > "$S"

# ---- Phase 1: collect (both tasks in parallel) ----
echo "$(date +%T) PHASE collect" >> "$S"
cpids=()
for t in "${TASKS[@]}"; do
  python3 gate.py collect --task "$t" --n_episodes "$CEP" --M "$CM" --collect_p 0.25 \
    --seed 1000 --out "$A/${t}.npz" > "$L/collect_${t}.log" 2>&1 &
  cpids+=($!)
done
wait "${cpids[@]}"
for t in "${TASKS[@]}"; do
  echo "$(date +%T) $(grep -a '^COLLECT' $L/collect_${t}.log || echo "COLLECT $t FAILED")" >> "$S"
done

# ---- Phase 2+3: per-task train then sweep arms (tasks parallel; arms parallel) ----
run_task() {
  local t="$1"
  [ -f "$A/${t}.npz" ] || { echo "$(date +%T) SKIP $t (no npz)" >> "$S"; return 1; }
  python3 gate.py train --data "$A/${t}.npz" --out "$A/${t}_gate.pt" \
    --steps 8000 --log_every 4000 > "$L/train_${t}.log" 2>&1
  echo "$(date +%T) $(grep -a '^TRAIN' $L/train_${t}.log)" >> "$S"

  local pids=()
  python3 gate.py eval --task "$t" --mode base --n_episodes "$NEP" --seed 1000 \
    --out "$A/arm_${t}_base.json" > "$L/arm_${t}_base.log" 2>&1 & pids+=($!)
  for h in "${FIXED[@]}"; do
    python3 gate.py eval --task "$t" --mode fixed --horizon "$h" --n_episodes "$NEP" --seed 1000 \
      --out "$A/arm_${t}_fixed_h${h}.json" > "$L/arm_${t}_fixed_h${h}.log" 2>&1 & pids+=($!)
  done
  for q in "${GATEQ[@]}"; do
    python3 gate.py eval --task "$t" --mode gate --gate "$A/${t}_gate.pt" --tau_q "$q" \
      --n_episodes "$NEP" --seed 1000 --out "$A/arm_${t}_gate_q${q}.json" \
      > "$L/arm_${t}_gate_q${q}.log" 2>&1 & pids+=($!)
  done
  wait "${pids[@]}"
  echo "$(date +%T) $t arms done: $(ls $A/arm_${t}_*.json 2>/dev/null | wc -l)/$(( 1 + ${#FIXED[@]} + ${#GATEQ[@]} ))" >> "$S"
}

echo "$(date +%T) PHASE train+sweep" >> "$S"
tpids=()
for t in "${TASKS[@]}"; do run_task "$t" & tpids+=($!); done
wait "${tpids[@]}"

# ---- Phase 4: assemble per-task sweep files + matched-budget verdict ----
echo "$(date +%T) PHASE verdict" >> "$S"
python3 - "$A" "${TASKS[@]}" >> "$S" 2>&1 <<'PY'
import json, sys, glob, time
A = sys.argv[1]; tasks = sys.argv[2:]
for task in tasks:
    arms = []
    for f in sorted(glob.glob(f"{A}/arm_{task}_*.json")):
        d = json.load(open(f)); arms.append(d.get("result", d))
    if not arms:
        print(f"{time.strftime('%H:%M:%S')} no arms for {task}"); continue
    vc = next((a.get("val_corr") for a in arms if a.get("val_corr") is not None), None)
    json.dump({"task": task, "val_corr": vc, "arms": arms},
              open(f"{A}/sweep_{task}.json", "w"), indent=2)
    print(f"{time.strftime('%H:%M:%S')} sweep_{task}.json: {len(arms)} arms")
PY

python3 gate.py verdict --sweeps "$A"/sweep_*.json --out "$A/verdict.json" \
  > "$L/verdict.log" 2>&1 || echo "$(date +%T) VERDICT cmd failed (see logs/verdict.log)" >> "$S"
grep -a '"summary"' "$A/verdict.json" >> "$S" 2>/dev/null

echo "$(date +%T) ALL_DONE" >> "$S"
touch "$A/DONE"
