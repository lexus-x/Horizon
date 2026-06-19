#!/bin/bash
# Watch the remaining-task survey progress + full results table
RDIR=/home/user/Desktop/vla_projects/RESEARCH/active/horizon/results
LOG=$RDIR/survey_remaining.log
P=/home/user/miniconda3/envs/lerobot/bin/python3

while true; do
  clear
  echo "=== HORIZON SURVEY  $(date '+%H:%M:%S') ==="
  echo ""

  # How many tasks completed so far
  DONE=$(grep -c "^  success=" "$LOG" 2>/dev/null || echo 0)
  LAST=$(grep "^===" "$LOG" 2>/dev/null | tail -1)
  echo "[ SURVEY PROGRESS ]  $DONE / 38 tasks done"
  echo "  Last: $LAST"
  echo ""

  # Show midband hits found so far
  if [ -f "$RDIR/base_survey_remaining.json" ]; then
    echo "[ MIDBAND (20-80%) — new tasks ]"
    $P -c "
import json
d = json.load(open('$RDIR/base_survey_remaining.json'))
mb = [t for t in d['survey'] if t.get('midband')]
if not mb:
    print('  none yet')
for t in sorted(mb, key=lambda x: x['pooled_success_pct'], reverse=True):
    print(f\"  {t['task']:<35} {t['pooled_success_pct']:>5.1f}%\")
" 2>/dev/null
    echo ""
  fi

  # Known results table
  echo "[ ALL GREEN/RED (distill vs base) ]"
  $P -c "
import json, os
RDIR='$RDIR'
rows = [
  ('plate-slide-v3',   'plate',     'horizon_plate_h50.json',     'horizon_plate_student.json'),
  ('drawer-open-v3',   'drawer',    'horizon_drawer_h50.json',    'horizon_drawer_student.json'),
  ('window-open-v3',   'window',    'horizon_window_h50.json',    'horizon_window_student.json'),
  ('push-v3',          'push',      'horizon_push_h50.json',      'horizon_push_student.json'),
  ('peg-insert-side',  'peg',       'horizon_peg_h50.json',       'horizon_peg_student.json'),
  ('pick-place-v3',    'pick',      'horizon_pickplace_h50.json', 'horizon_pickplace_student.json'),
]
def sr(f):
    try:
        d=json.load(open(os.path.join(RDIR,f)))
        return d['result']['pc_success']
    except: return None

print(f\"  {'Task':<20} {'Base':>6}  {'Distill':>8}  Status\")
print('  '+'-'*48)
for name, short, bf, sf in rows:
    b=sr(bf); s=sr(sf)
    if b is None or s is None: continue
    ok = 'GREEN' if s>=b else 'RED'
    print(f\"  {name:<20} {b:>5.0f}%  {s:>7.0f}%  {ok}\")
" 2>/dev/null

  sleep 5
done
