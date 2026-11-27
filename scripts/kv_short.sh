#!/usr/bin/env bash
# Short real run of the known-vehicle option: one model on the full training set, 8 epochs (2 per curriculum stage), then val_id evaluation.
# Inside the container. Usage: scripts/kv_short.sh <model> <gpu>
set -euo pipefail
M=${1:?model}; GPU=${2:?gpu}
D=data/datasets/physiworld_pilot_v7_v0.7.0
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
python -m physiworld_uav.training.run --dataset $D --name kv_short_${M}_s0 --model $M \
  --config config/training/default.yaml config/training/compare.yaml config/training/v7_openloop.yaml config/training/known_vehicle.yaml \
  --set epochs=8 stage_max_epochs=2 seed=0
echo KV_SHORT_DONE $M
