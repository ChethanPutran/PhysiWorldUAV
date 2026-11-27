#!/usr/bin/env bash
# Run the pilot baseline gates INSIDE the container (docs/specs/model_specification.md §8):
#   G1 tiny-overfit (B2, B3, B4), G2 full-pilot training with validation (B2, B3, B4), G3/G10 evaluation vs B0/B1.
# Usage: scripts/run_baseline_gates.sh [dataset_dir] [device]
set -euo pipefail
DS="${1:-data/datasets/physiworld_pilot_v0.5.0}"; DEV="${2:-cuda:0}"
EXP=outputs/experiments; OUT=outputs/eval
mkdir -p "$EXP" "$OUT" logs
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4
run() { echo "[$(date +%T)] $*"; "$@"; }
for M in b2 b3 b4; do
  run python -m physiworld_uav.training.run --dataset "$DS" --name "g1_${M}_tiny" --model "$M" --tiny 8 --device "$DEV" \
      --set epochs=150 lr=0.003 batch_size=128 patience=1000 horizon_stages=[20] warmup_steps=20
done
run python -m physiworld_uav.evaluation.run --dataset "$DS" --device "$DEV" --roles train --limit-episodes 8 --horizon 20 \
    --models b0 b1 b2=$EXP/g1_b2_tiny/checkpoints/last.pt b3=$EXP/g1_b3_tiny/checkpoints/last.pt b4=$EXP/g1_b4_tiny/checkpoints/last.pt --out $OUT/g1_tiny.json
for M in b2 b3 b4; do
  run python -m physiworld_uav.training.run --dataset "$DS" --name "pilot_${M}" --model "$M" --device "$DEV" \
      --set epochs=60 patience=8 horizon_stages=[1,5,10,20] stage_patience=3 stage_max_epochs=8
done
run python -m physiworld_uav.evaluation.run --dataset "$DS" --device "$DEV" --horizon 20 \
    --models b0 b1 b2=$EXP/pilot_b2/checkpoints/best.pt b3=$EXP/pilot_b3/checkpoints/best.pt b4=$EXP/pilot_b4/checkpoints/best.pt --out $OUT/pilot_gates.json
echo "[$(date +%T)] gates finished"
