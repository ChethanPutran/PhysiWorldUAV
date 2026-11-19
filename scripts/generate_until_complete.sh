#!/usr/bin/env bash
# Keep (re)starting a generation until the dataset is complete. Safe after crashes, reboots and
# container restarts because `--resume` skips episodes already on disk.
# Run INSIDE the container (activated env, PYTHONPATH including src), ideally in tmux or `docker exec -d`:
#   scripts/generate_until_complete.sh experiment=pilot
# Stops when index.json says complete, after MAX_ATTEMPTS, or when an attempt writes nothing new.
set -u
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  python -m physiworld_uav.experiments.generate run "$@" --resume
  status=$(python - "$@" <<'PY'
import json, sys
from physiworld_uav.config import compose, to_plain
from physiworld_uav.experiments.generate import dataset_dir
idx = dataset_dir(to_plain(compose(sys.argv[1:]))) / "index.json"
try:
    d = json.loads(idx.read_text())
    print(("complete" if d["complete"] else "incomplete"), d["n_written"], d.get("n_planned"))
except Exception:
    print("noindex 0 0")
PY
)
  echo "attempt $attempt: $status"
  case "$status" in complete*) exit 0;; esac
  written=${status#* }; written=${written%% *}
  [ "${last_written:-x}" = "$written" ] && { echo "no progress since the previous attempt; stopping"; exit 1; }
  last_written=$written
  sleep 30
done
exit 1
