#!/usr/bin/env bash
# After sweep fix_v1: wait until every job has finished, evaluate every new run on every role (3 GPUs in parallel), then summarise.
# Run INSIDE the container (detached):   scripts/post_sweep.sh
# Output: outputs/eval/fx_<ds>_<chunk>.json, outputs/reports/sweep_fix_v1/{tables,figures,SUMMARY.md}, log logs/post_sweep.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SW=outputs/sweeps/fix_v1; EXP=outputs/experiments
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
while python - <<PY
import json,sys
st=json.load(open("$SW/status.json")); sys.exit(0 if any(v["state"] in ("queued","running") for v in st.values()) else 1)
PY
do sleep 20; done
echo "[$(date +%T)] sweep finished; evaluating"
declare -A DS=( [v05]=data/datasets/physiworld_pilot_v0.5.0 [v06]=data/datasets/physiworld_pilot_v0.6.0 )
mkdir -p outputs/eval
for d in v05 v06; do
  M=()
  for r in $(ls -d $EXP/fx_${d}_*/ 2>/dev/null | xargs -n1 basename); do
    [ -f "$EXP/$r/checkpoints/best.pt" ] && M+=("$r=$EXP/$r/checkpoints/best.pt")
  done
  [ "$d" = v06 ] && M+=("v06_b3=$EXP/v06_b3/checkpoints/best.pt")                  # the seed-0 B3 of the first comparison (not affected by the bug)
  n=${#M[@]}; chunks=3; i=0
  for c in 0 1 2; do
    sub=(); for ((k=c; k<n; k+=chunks)); do sub+=("${M[$k]}"); done
    [ ${#sub[@]} -gt 0 ] || continue
    extra=(); [ $c -eq 0 ] && extra=(b0 b1 b6a_future)
    python -m physiworld_uav.evaluation.run --dataset "${DS[$d]}" --models "${extra[@]}" "${sub[@]}" --horizon 20 --noise nominal \
       --device "cuda:$c" --out "outputs/eval/fx_${d}_c${c}.json" > "logs/eval_fx_${d}_c${c}.log" 2>&1 &
  done
done
wait
echo "[$(date +%T)] evaluation finished; summarising"
python scripts/summarize_sweep.py > logs/summarize_sweep.log 2>&1 && echo "[$(date +%T)] done" || echo "[$(date +%T)] summary FAILED (logs/summarize_sweep.log)"
