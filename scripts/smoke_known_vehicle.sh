#!/usr/bin/env bash
# Smoke test of the known-vehicle option: tiny overfit run per model, then evaluate the checkpoints on a few validation episodes.
# Run inside the container. Usage: scripts/smoke_known_vehicle.sh <dataset> [gpu]
set -euo pipefail
D=${1:?dataset}; GPU=${2:-3}
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
CFG="config/training/default.yaml config/training/v7_openloop.yaml config/training/known_vehicle.yaml"
for M in b3 b2 b4; do
  python -m physiworld_uav.training.run --dataset "$D" --name smoke_kv_$M --model $M --tiny 6 --config $CFG --set epochs=3 'horizon_stages=[1,5]' val_horizon=5 --force
done
python -m physiworld_uav.evaluation.run --dataset "$D" --roles val_id --limit-episodes 5 --horizon 5 --device cuda:0 \
  --models b1_kv b6a_future_kv smoke_kv_b3=outputs/experiments/smoke_kv_b3/checkpoints/best.pt smoke_kv_b2=outputs/experiments/smoke_kv_b2/checkpoints/best.pt \
  smoke_kv_b4=outputs/experiments/smoke_kv_b4/checkpoints/best.pt --out outputs/eval/smoke_kv.json
echo SMOKE_DONE
