#!/usr/bin/env bash
# Launch one pinned GPU. The process group killed on the deadline is this job only.
# Do not edit the shared bashrc. That file sets CUDA_VISIBLE_DEVICES to the
# least-used index on every shell; this script overrides it for this process.
set -u
MODE="${1:-smoke}"
JOB=/lfs/mercury2/0/brando9/brier_ce_pilot_076e
CODE="$JOB/code"
PY=/lfs/mercury2/0/brando9/uv_envs/veribench/bin/python
OUT="$JOB/$MODE"
DATA="$JOB/data"
mkdir -p "$OUT" "$DATA"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "START $(date -Is) mode=$MODE host=$(hostname) pid=$$"

if [ "$MODE" = "full" ] && tmux has-session -t brier-ce-pilot-076e 2>/dev/null; then
  # The caller may already be inside that session. Only refuse a second full job.
  if [ "${INSIDE_BRIER_TMUX:-}" != "1" ]; then
    echo "BLOCKED full session brier-ce-pilot-076e already exists"
    exit 3
  fi
fi

UUID=$(
  "$PY" - << 'PY'
import subprocess, sys
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
    text=True,
)
busy = set()
for line in apps.splitlines():
    if line.strip():
        busy.add(line.split(",")[0].strip())
for line in gpus.splitlines():
    _idx, uuid, mem, util = [part.strip() for part in line.split(",")]
    if uuid not in busy and int(mem) < 200 and int(util) == 0:
        print(uuid)
        sys.exit(0)
sys.exit(2)
PY
)
if [ -z "${UUID:-}" ]; then
  echo "BLOCKED no idle GPU"
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$UUID"
export GPU_UUID="$UUID"
echo "PINNED_GPU $UUID"

AVAIL=$(df -B1 -P /lfs/mercury2/0 | awk 'NR==2 {print $4}')
echo "FREE_BYTES $AVAIL"
if [ "$AVAIL" -lt 8000000000 ]; then
  echo "BLOCKED free bytes $AVAIL"
  exit 3
fi

if [ "$MODE" = "smoke" ]; then
  LIMIT=1500
else
  LIMIT=44400
fi

cd "$CODE"
"$PY" -u train.py --mode "$MODE" --out "$OUT" --data "$DATA" --job-root "$JOB" >> "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/train.pid"
echo "$PID" > "$OUT/train.pgid"
echo "TRAIN_PID $PID"

(
  while kill -0 "$PID" 2>/dev/null; do
    date -Is
    nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader | grep "$UUID" || true
    sleep 60
  done
) >> "$OUT/gpu_samples.log" 2>&1 &
SAMP=$!
echo "$SAMP" > "$OUT/sampler.pid"

(
  sleep "$LIMIT"
  echo "WATCHDOG killing $PID"
  kill -TERM "$PID" 2>/dev/null || true
  sleep 30
  kill -KILL "$PID" 2>/dev/null || true
) &
WATCH=$!

set +e
wait "$PID"
CODE_EXIT=$?
set -u
kill "$WATCH" 2>/dev/null || true
kill "$SAMP" 2>/dev/null || true
echo "END $(date -Is) exit=$CODE_EXIT"
exit "$CODE_EXIT"
