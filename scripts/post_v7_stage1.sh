#!/usr/bin/env bash
# After the v7 stage-1 sweep: evaluate every run (best.pt) and the references (B0, B1, B6a oracle) on all roles of pilot v7, in 3 parallel
# chunks on GPUs 0-2, then the constant-velocity baseline and the summary. Run INSIDE the container (detached):  scripts/post_v7_stage1.sh
# Output: outputs/eval/st_v07_c{0,1,2}.json, outputs/eval/constant_velocity_v07.json, outputs/reports/v7_stage1/SUMMARY.md; log logs/post_v7_stage1.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DS=data/datasets/physiworld_pilot_v7_v0.7.0; E=outputs/experiments
m() { echo "st_v07_$1_s0=$E/st_v07_$1_s0/checkpoints/best.pt"; }
run() { local c="$1" g="$2"; shift 2
  python -m physiworld_uav.evaluation.run --dataset "$DS" --models "$@" --horizon 20 --noise nominal --device "cuda:$g" \
     --out "outputs/eval/st_v07_c$c.json" > "logs/eval_st_v07_c$c.log" 2>&1 && echo "[$(date +%T)] chunk $c done" || echo "[$(date +%T)] chunk $c FAILED"; }
run 0 0 b0 b1 b6a_future b6a_current "$(m b3)" "$(m b3_small)" &
run 1 1 "$(m b4)" "$(m b4_rw)" "$(m b2)" &
run 2 2 "$(m b5a)" "$(m b5b)" &
python scripts/constant_velocity_baseline.py --dataset "$DS" --out outputs/eval/constant_velocity_v07.json > logs/eval_cv_v07.log 2>&1 &
wait
python scripts/summarize_v7_stage1.py > logs/summarize_v7_stage1.log 2>&1 && echo "[$(date +%T)] summary done" || echo "[$(date +%T)] summary FAILED"
python -m physiworld_uav.notify "Evaluation of the 7 trained runs on all roles finished" --title "Evaluation" --tags white_check_mark >/dev/null 2>&1 || true
