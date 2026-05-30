#!/usr/bin/env bash
# Fully-parallel sweep: one process per (task, arm). 80GB VRAM fits all of them
# at once (~2.7GB each). Each writes its OWN per-arm json -> no shared-file races.
# Then assemble per-task sweep_*.json and the matched-budget verdict.
# Writes gate_artifacts/DONE3 when finished.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
A=gate_artifacts
L=$A/logs
mkdir -p "$L"
rm -f "$A"/DONE3 "$A"/parm_*.json
S=$A/STATUS3
NEP=${NEP:-40}
FIXED=(25 15 10 8)
GATEQ=(0.80 0.85 0.90 0.95)
TASKS=(push-v3 plate-slide-v3)

echo "$(date +%T) START parallel NEP=$NEP" > "$S"

launch() {  # task, label, args...
  local t="$1" lab="$2"; shift 2
  python3 gate.py eval --task "$t" --n_episodes "$NEP" --seed 1000 \
    --out "$A/parm_${t}_${lab}.json" "$@" > "$L/parm_${t}_${lab}.log" 2>&1 &
}

pids=()
for t in "${TASKS[@]}"; do
  launch "$t" base --mode base;                         pids+=($!)
  for h in "${FIXED[@]}"; do
    launch "$t" "fixed_h${h}" --mode fixed --horizon "$h"; pids+=($!)
  done
  for q in "${GATEQ[@]}"; do
    launch "$t" "gate_q${q}" --mode gate --gate "$A/${t}_gate.pt" --tau_q "$q"; pids+=($!)
  done
done
echo "$(date +%T) launched ${#pids[@]} arm processes" >> "$S"
wait "${pids[@]}"
echo "$(date +%T) all arms done: $(ls $A/parm_*.json 2>/dev/null | wc -l)" >> "$S"

# assemble per-task sweep files + verdict
python3 - "$A" "${TASKS[@]}" >> "$S" 2>&1 <<'PY'
import json, sys, glob, time
A = sys.argv[1]; tasks = sys.argv[2:]
for task in tasks:
    arms = []
    for f in sorted(glob.glob(f"{A}/parm_{task}_*.json")):
        d = json.load(open(f)); arms.append(d.get("result", d))
    if not arms:
        print(f"{time.strftime('%H:%M:%S')} no arms for {task}"); continue
    vc = next((a.get("val_corr") for a in arms if a.get("val_corr") is not None), None)
    json.dump({"task": task, "val_corr": vc, "arms": arms},
              open(f"{A}/sweep_{task}.json", "w"), indent=2)
    print(f"{time.strftime('%H:%M:%S')} sweep_{task}.json: {len(arms)} arms")
PY

python3 gate.py verdict --sweeps "$A"/sweep_*.json --out "$A/verdict.json" \
  > "$L/verdict.log" 2>&1 || echo "$(date +%T) VERDICT FAILED" >> "$S"
grep -a '"summary"' "$A/verdict.json" >> "$S" 2>/dev/null
echo "$(date +%T) ALL_DONE" >> "$S"
touch "$A/DONE3"
