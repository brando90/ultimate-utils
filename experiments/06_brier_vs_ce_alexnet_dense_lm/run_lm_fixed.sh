#!/usr/bin/env bash
# One pinned GPU. The watchdog is the protocol's throughput-derived limit.
# It is longer than every planned run, and it kills only this process.
set -u
MODE="${1:-all}"
JOB=/lfs/mercury2/0/brando9/brier_ce_pilot_076e
CODE="$JOB/code"
PY=/lfs/mercury2/0/brando9/uv_envs/veribench/bin/python
OUT="$JOB/lm_fixed"
DATA="$JOB/data/wikitext103_hf"
mkdir -p "$OUT" "$DATA"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "START $(date -Is) mode=$MODE host=$(hostname) pid=$$"

if tmux has-session -t brier-ce-lm-076e 2>/dev/null; then
  if [ "${INSIDE_BRIER_LM:-}" != "1" ]; then
    echo "BLOCKED session brier-ce-lm-076e already exists"
    exit 3
  fi
fi
if pgrep -f "$CODE/train_lm_fixed.py" >/dev/null 2>&1; then
  echo "BLOCKED train_lm_fixed.py already running"
  exit 3
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

cd "$CODE"
run_py() {
  "$PY" -u train_lm_fixed.py --stage "$1" --out "$OUT" --data "$DATA" >> "$OUT/train.log" 2>&1 < /dev/null &
  echo $! > "$OUT/train.pid"
  echo "TRAIN_PID $!"
  wait $!
}

if [ "$MODE" = "all" ] || [ "$MODE" = "prepare" ]; then
  run_py prepare
  PREP=$?
  echo "PREPARE_EXIT $PREP"
  if [ "$PREP" -ne 0 ]; then
    exit "$PREP"
  fi
fi

if [ "$MODE" = "prepare" ]; then
  exit 0
fi

LIMIT=$("$PY" -c "import json; print(int(json.load(open('$OUT/protocol.json'))['watchdog_seconds']))")
echo "WATCHDOG_SECONDS $LIMIT"
"$PY" -u train_lm_fixed.py --stage train --out "$OUT" --data "$DATA" >> "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/train.pid"
echo "TRAIN_PID $PID"
(
  while kill -0 "$PID" 2>/dev/null; do
    date -Is
    nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader | grep "$UUID" || true
    sleep 60
  done
) >> "$OUT/gpu_samples.log" 2>&1 &
SAMP=$!
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
