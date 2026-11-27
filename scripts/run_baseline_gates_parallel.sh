#!/usr/bin/env bash
# Parallel version of run_baseline_gates.sh (run INSIDE the container): the independent training jobs run at the same
# time on different GPUs, then the evaluation runs once all have finished.
# Usage: scripts/run_baseline_gates_parallel.sh [dataset_dir] [gpu_a] [gpu_b] [gpu_c]
set -uo pipefail
DS="${1:-data/datasets/physiworld_pilot_v0.5.0}"; GA="${2:-cuda:0}"; GB="${3:-cuda:1}"; GC="${4:-cuda:2}"
EXP=outputs/experiments; OUT=outputs/eval
mkdir -p "$EXP" "$OUT" logs
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
tiny() { python -m physiworld_uav.training.run --dataset "$DS" --name "g1_$1_tiny" --model "$1" --tiny 8 --device "$2" \
           --set epochs=150 lr=0.003 batch_size=128 patience=1000 horizon_stages=[20] warmup_steps=20 > "logs/g1_$1_tiny.log" 2>&1; }
full() { python -m physiworld_uav.training.run --dataset "$DS" --name "pilot_$1" --model "$1" --device "$2" \
           --set epochs=60 patience=8 horizon_stages=[1,5,10,20] stage_patience=3 stage_max_epochs=8 > "logs/pilot_$1.log" 2>&1; }
echo "[$(date +%T)] launching jobs"
[ -f $EXP/g1_b2_tiny/summary.json ] || tiny b2 "$GA" &
tiny b3 "$GB" &
tiny b4 "$GC" &
full b2 "$GA" &
full b3 "$GB" &
full b4 "$GC" &
wait
echo "[$(date +%T)] training finished; evaluating"
python -m physiworld_uav.evaluation.run --dataset "$DS" --device "$GA" --roles train --limit-episodes 8 --horizon 20 \
    --models b0 b1 b2=$EXP/g1_b2_tiny/checkpoints/last.pt b3=$EXP/g1_b3_tiny/checkpoints/last.pt b4=$EXP/g1_b4_tiny/checkpoints/last.pt --out $OUT/g1_tiny.json > logs/eval_g1.log 2>&1
python -m physiworld_uav.evaluation.run --dataset "$DS" --device "$GA" --horizon 20 \
    --models b0 b1 b2=$EXP/pilot_b2/checkpoints/best.pt b3=$EXP/pilot_b3/checkpoints/best.pt b4=$EXP/pilot_b4/checkpoints/best.pt --out $OUT/pilot_gates.json > logs/eval_pilot.log 2>&1
echo "[$(date +%T)] gates finished"
