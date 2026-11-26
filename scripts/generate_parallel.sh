#!/usr/bin/env bash
# Parallel dataset generation: W worker processes, spread over the given GPUs, then one merge (validate + index.json).
# Run INSIDE the container (activated env, PYTHONPATH including src), ideally in tmux or `docker exec -d`:
#
#   scripts/generate_parallel.sh --workers 8 --gpus 0,1,2 -- experiment=pilot
#
# Why several workers per GPU: one generator process is CPU-bound (about 4–5 busy cores, ~3 % GPU utilisation, ~55–60
# physics steps/s). The GPU is idle most of the time, so the number of workers is limited by free CPU cores and RAM, not by
# the number of GPUs. Worker w uses GPU gpus[w mod n_gpus] through CUDA_VISIBLE_DEVICES (the config, hence the config hash,
# is not touched). Isaac may print renderer allocation errors when a non-zero GPU is selected; the simulation does not render.
#
# Keep --workers the SAME when you restart an unfinished dataset: the split of episodes over workers depends on it and each worker's --resume
# only reads its own shard files, so a different count regenerates episodes that exist in another worker's shard (duplicate ids, merge fails).
#
# Safe after crashes: every attempt starts the workers with --resume (episodes already written are skipped) and ends with a
# merge. Stops when index.json says complete, after MAX_ATTEMPTS, or when an attempt writes nothing new.
set -u
WORKERS=2; GPUS="0"; MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done
[ $# -ge 1 ] || { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
python -m physiworld_uav.experiments.generate plan "$@" || exit 1          # writes manifest / config / provenance once
last_written=-1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[$(date +%T)] attempt $attempt: $WORKERS workers on GPUs $GPUS"
  pids=()
  for w in $(seq 0 $((WORKERS - 1))); do
    gpu="${GPU_LIST[$((w % ${#GPU_LIST[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m physiworld_uav.experiments.generate run "$@" --shard "$w/$WORKERS" --resume \
        > "logs/gen_${STAMP}_a${attempt}_w${w}.log" 2>&1 &
    pids+=($!)
    sleep 5                                                                 # stagger the Isaac start-ups
  done
  for pid in "${pids[@]}"; do wait "$pid" || echo "worker pid $pid exited with an error (see logs/)"; done
  python -m physiworld_uav.experiments.generate merge "$@" || echo "merge refused (a worker still reports in_progress?)"
  status=$(python - "$@" <<'PY'
import json, sys
from physiworld_uav.config import compose, to_plain
from physiworld_uav.experiments.generate import dataset_dir
try:
    ov = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]     # Hydra overrides only (not --limit etc.)
    d = json.loads((dataset_dir(to_plain(compose(ov))) / "index.json").read_text())
    print(("complete" if d["complete"] else "incomplete"), d["n_written"], d.get("n_planned"))
except Exception:
    print("noindex 0 0")
PY
)
  echo "[$(date +%T)] attempt $attempt: $status"
  case "$status" in complete*) exit 0;; esac
  written=${status#* }; written=${written%% *}
  [ "$last_written" = "$written" ] && { echo "no progress since the previous attempt; stopping"; exit 1; }
  last_written=$written
done
exit 1
