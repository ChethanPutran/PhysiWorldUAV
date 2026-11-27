#!/usr/bin/env bash
# After training: evaluate, cross-evaluate, analyse and build the report for two run sets. Run INSIDE the container.
#
#   scripts/post_training_report.sh                 # v05 (pilot v0.5.0) vs v06 (pilot v0.6.0), report in outputs/reports/training_compare_v05_v06
# Waits until all 16 runs have a summary.json (or WAIT=0 to skip the wait), then:
#   1. scripts/eval_baselines.sh on each dataset with its own models, and the two cross-evaluations (v05 models on v06 data and back)
#   2. evaluation.analysis regime / anchor / residual / shuffle for each tag on its dataset (JSON in outputs/eval/)
#   3. evaluation.report -> figures, tables, REPORT.md and report.html
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D5=data/datasets/physiworld_pilot_v0.5.0; D6=data/datasets/physiworld_pilot_v0.6.0
OUTDIR="${OUTDIR:-outputs/reports/training_compare_v05_v06}"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [ "${WAIT:-1}" = "1" ]; then
  while [ "$(ls outputs/experiments/v0[56]_*/summary.json 2>/dev/null | wc -l)" -lt 16 ]; do sleep 30; done
fi
echo "[$(date +%T)] training finished; evaluating"
scripts/eval_baselines.sh "$D5" v05 nominal 20 cuda:0 v05 > logs/post_eval_v05.log 2>&1 &
scripts/eval_baselines.sh "$D6" v06 nominal 20 cuda:1 v06 > logs/post_eval_v06.log 2>&1 &
scripts/eval_baselines.sh "$D6" v05 nominal 20 cuda:2 v05_on_v06 > logs/post_eval_v05_on_v06.log 2>&1 &
wait
scripts/eval_baselines.sh "$D5" v06 nominal 20 cuda:0 v06_on_v05 > logs/post_eval_v06_on_v05.log 2>&1
echo "[$(date +%T)] analyses"
analyse() {                                         # tag dataset gpu
  local tag="$1" ds="$2" gpu="$3" E=outputs/experiments
  local M=(b1 b6a_future "b2=$E/${tag}_b2/checkpoints/best.pt" "b3=$E/${tag}_b3/checkpoints/best.pt" "b4=$E/${tag}_b4/checkpoints/best.pt"
           "b5b=$E/${tag}_b5b/checkpoints/best.pt" "b6b_future=$E/${tag}_b6b_future/checkpoints/best.pt")
  local A="python -m physiworld_uav.evaluation.analysis"
  $A regime   --dataset "$ds" --models "${M[@]}" --roles val_id test_id --gap b3:b6a_future b3:b4 b3:b2 --noise nominal --device "cuda:$gpu" --out "outputs/eval/analysis_regime_${tag}.json" > "logs/analysis_regime_${tag}.log" 2>&1
  $A anchor   --dataset "$ds" --models "b2=$E/${tag}_b2/checkpoints/best.pt" "b3=$E/${tag}_b3/checkpoints/best.pt" "b5b=$E/${tag}_b5b/checkpoints/best.pt" --roles val_id --every 1 5 10 --mode h --noise nominal --device "cuda:$gpu" --out "outputs/eval/analysis_anchor_${tag}.json" > "logs/analysis_anchor_${tag}.log" 2>&1
  $A residual --dataset "$ds" --models "b3=$E/${tag}_b3/checkpoints/best.pt" --roles val_id test_id --noise nominal --device "cuda:$gpu" --out "outputs/eval/analysis_residual_${tag}.json" > "logs/analysis_residual_${tag}.log" 2>&1
  $A shuffle  --dataset "$ds" --models b1 "b2=$E/${tag}_b2/checkpoints/best.pt" "b3=$E/${tag}_b3/checkpoints/best.pt" "b4=$E/${tag}_b4/checkpoints/best.pt" --roles train val_id test_id --noise nominal --device "cuda:$gpu" --out "outputs/eval/analysis_shuffle_${tag}.json" > "logs/analysis_shuffle_${tag}.log" 2>&1
}
analyse v05 "$D5" 0 & analyse v06 "$D6" 1 &
wait
echo "[$(date +%T)] report"
python -m physiworld_uav.evaluation.report --tags v05 v06 --out "$OUTDIR" --eval outputs/eval --resources "$OUTDIR" > logs/post_report.log 2>&1
tail -2 logs/post_report.log
echo "[$(date +%T)] done: $OUTDIR/report.html"
