#!/usr/bin/env bash
# Resume an unfinished dataset generation after the server restarted. Run on the HOST (from cron @reboot, or by hand to test).
#
#   scripts/resume_on_boot.sh            do it (waits up to 30 min for Docker, the container and the GPUs)
#   DRY_RUN=1 scripts/resume_on_boot.sh  only print what it would do
#
# It does nothing unless ALL of these hold:
#   * the file logs/AUTO_RESUME exists (delete it to switch the automatic resume off; no crontab edit needed)
#   * the container is running and nvidia-smi answers
#   * the dataset's index.json does not say "complete"
#   * no generation worker of the experiment is already running
#   * at least MIN_FREE_GB of RAM is available (the workers use about 10 GB each)
# Then it starts scripts/generate_parallel.sh in the container (it re-plans, starts the workers with --resume, merges and validates) and
# restarts the dataset report page. Settings below can be overridden by environment variables.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO" || exit 1
CONTAINER="${CONTAINER:-isaac-sim}"
EXPERIMENT="${EXPERIMENT:-pilot_v7}"
PHYSICS="${PHYSICS:-isaac_200hz_m20}"
DATASET="${DATASET:-data/datasets/physiworld_pilot_v7_v0.7.0}"
WORKERS="${WORKERS:-9}"; GPUS="${GPUS:-0,1,2}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
REPORT_PORT="${REPORT_PORT:-60193}"
# the report page listens on 0.0.0.0 (reachable from the campus network, no login) only while the file logs/REPORT_PUBLIC exists; else 127.0.0.1
if [ -f logs/REPORT_PUBLIC ]; then REPORT_HOST="${REPORT_HOST:-0.0.0.0}"; else REPORT_HOST="${REPORT_HOST:-127.0.0.1}"; fi
say() { echo "[$(date '+%F %T')] $*"; }
mkdir -p logs
exec 9>"$REPO/logs/.resume_on_boot.lock"; flock -n 9 || { say "another resume_on_boot is running; exit"; exit 0; }

[ -f logs/AUTO_RESUME ] || { say "logs/AUTO_RESUME does not exist: automatic resume is off"; exit 0; }
# 1. wait for Docker, the container and the GPUs (a boot takes a while)
for i in $(seq 1 90); do
  st="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
  if [ "$st" = running ] && nvidia-smi -L >/dev/null 2>&1; then break; fi
  if [ "$st" = exited ] || [ "$st" = created ]; then docker start "$CONTAINER" >/dev/null 2>&1 && say "started container $CONTAINER"; fi
  [ "$i" -eq 1 ] && say "waiting for Docker / container $CONTAINER / GPUs (state: ${st:-unknown})"
  sleep 20
done
[ "$st" = running ] && nvidia-smi -L >/dev/null 2>&1 || { say "container or GPUs not available after 30 min; giving up"; exit 1; }

# 2. is there anything to do?
complete="$(python3 - "$DATASET/index.json" <<'PY'
import json, sys
try:
    print("yes" if json.load(open(sys.argv[1])).get("complete") else "no")
except Exception:
    print("no")
PY
)"
if [ "$complete" = yes ]; then say "$DATASET is complete; nothing to do"; exit 0; fi
if ps -eo args | grep -E "generate run" | grep -F -- "--shard" | grep -F "experiment=$EXPERIMENT" | grep -qv grep; then
  say "generation workers of $EXPERIMENT are already running; nothing to do"; exit 0
fi
free_gb="$(free -g | awk '/^Mem:/{print $7}')"
if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then say "only ${free_gb} GB RAM available (< $MIN_FREE_GB GB); not starting. Run again later."; exit 2; fi

# 3. start (or describe)
STAMP="$(date +%Y%m%d_%H%M%S)"
CMD="cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=\$PWD/src:\$PYTHONPATH OMP_NUM_THREADS=4 HDF5_USE_FILE_LOCKING=FALSE && scripts/generate_parallel.sh --workers $WORKERS --gpus $GPUS -- experiment=$EXPERIMENT physics=$PHYSICS > logs/gen_${EXPERIMENT}_boot_$STAMP.log 2>&1"
if [ "${DRY_RUN:-0}" = 1 ]; then
  say "DRY RUN: would run in $CONTAINER: $CMD"
  say "DRY RUN: would (re)start the dataset report page on $REPORT_HOST:$REPORT_PORT if nothing listens there"
  exit 0
fi
say "resuming $EXPERIMENT: $WORKERS workers on GPUs $GPUS (${free_gb} GB RAM available)"
docker exec -d "$CONTAINER" bash -c "$CMD"
if ! ss -ltn 2>/dev/null | grep -q ":$REPORT_PORT "; then
  PYTHONPATH="$REPO/src" nohup python3 -m physiworld_uav.visualization.dataset_report --dataset "$DATASET" --serve --host "$REPORT_HOST" --port "$REPORT_PORT" \
      > logs/dataset_report_serve.log 2>&1 &
  say "dataset report page restarted on $REPORT_HOST:$REPORT_PORT"
fi
say "started; logs/gen_${EXPERIMENT}_boot_$STAMP.log"
