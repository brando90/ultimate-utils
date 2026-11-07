#!/usr/bin/env bash
# CIFAR-10 stability follow-up on one idle GPU.
# Never the WikiText GPU, and never a GPU that already has a compute process.
set -u
MODE="${1:-all}"
JOB=/lfs/mercury1/0/brando9/brier_ce_pilot_076e
CODE="$JOB/code"
PY=/lfs/mercury1/0/brando9/uv_envs/veribench/bin/python
OUT="$JOB/stability_cifar"
DATA="$JOB/data"
FORBIDDEN=GPU-4cdf8223-cd3c-4c19-fe22-dd070b82e697
PREFER=GPU-071c409d-a1bb-9fd7-1239-a271b8ce1889
mkdir -p "$OUT" "$DATA"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "START $(date -Is) mode=$MODE host=$(hostname) pid=$$"

if tmux has-session -t brier-ce-stability-076e 2>/dev/null; then
  if [ "${INSIDE_BRIER_STABILITY:-}" != "1" ]; then
    echo "BLOCKED session brier-ce-stability-076e already exists"
    exit 3
  fi
fi
if pgrep -f "$CODE/train_stability.py" >/dev/null 2>&1; then
  echo "BLOCKED train_stability.py already running"
  exit 3
fi

UUID=$(
  PREFER="$PREFER" FORBIDDEN="$FORBIDDEN" "$PY" - << 'PY'
import os, subprocess, sys
prefer = os.environ["PREFER"]
forbidden = os.environ["FORBIDDEN"]
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
idle = []
for line in gpus.splitlines():
    _idx, uuid, mem, util = [part.strip() for part in line.split(",")]
    if uuid == forbidden or uuid in busy or int(mem) >= 200 or int(util) != 0:
        continue
    idle.append(uuid)
if prefer in idle:
    print(prefer)
    sys.exit(0)
if idle:
    print(idle[0])
    sys.exit(0)
sys.exit(2)
PY
)
if [ -z "${UUID:-}" ]; then
  echo "BLOCKED no idle GPU"
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$UUID"
echo "PINNED_GPU $UUID"
AVAIL=$(df -B1 -P /lfs/mercury1/0 | awk 'NR==2 {print $4}')
echo "FREE_BYTES $AVAIL"
if [ "$AVAIL" -lt 8000000000 ]; then
  echo "BLOCKED free bytes $AVAIL"
  exit 3
fi

cd "$CODE"
if [ "$MODE" = "all" ] || [ "$MODE" = "smoke" ]; then
  if [ ! -f "$OUT/protocol.json" ]; then
    SMOKE=1
    for TRY in 1 2 3; do
      "$PY" -u train_stability.py --stage smoke --out "$OUT" --data "$DATA" >> "$OUT/train.log" 2>&1 < /dev/null
      SMOKE=$?
      echo "SMOKE_EXIT $SMOKE try=$TRY"
      if [ "$SMOKE" -eq 0 ]; then
        break
      fi
      sleep $((15 * TRY))
    done
    if [ "$SMOKE" -ne 0 ]; then
      exit "$SMOKE"
    fi
  fi
fi
if [ "$MODE" = "smoke" ]; then
  exit 0
fi

LIMIT=$("$PY" -c "import json; print(int(json.load(open('$OUT/protocol.json'))['watchdog_seconds']))")
echo "WATCHDOG_SECONDS $LIMIT"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "TRAIN_ATTEMPT $ATTEMPT"
  "$PY" -u train_stability.py --stage train --out "$OUT" --data "$DATA" >> "$OUT/train.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$OUT/train.pid"
  echo "TRAIN_PID $PID"
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
  if [ "$CODE_EXIT" -eq 0 ]; then
    echo "END $(date -Is) exit=0 attempts=$ATTEMPT"
    exit 0
  fi
  if [ "$ATTEMPT" -ge 4 ]; then
    echo "END $(date -Is) exit=$CODE_EXIT attempts=$ATTEMPT"
    exit "$CODE_EXIT"
  fi
  SLEEP=$((30 * ATTEMPT))
  echo "RETRY attempt=$ATTEMPT exit=$CODE_EXIT sleep_seconds=$SLEEP"
  sleep "$SLEEP"
done
