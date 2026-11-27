#!/usr/bin/env bash
# Evaluate the baselines trained by scripts/train_all_baselines.sh (plus the untrained b0, b1, b6a_*) on every split role of a dataset.
# Run INSIDE the container:  scripts/eval_baselines.sh <dataset_dir> <tag> [noise=nominal] [horizon=20] [device=cuda:0] [out_name=<tag>]
# Cross-evaluation: the models of <tag> on another dataset, e.g.  scripts/eval_baselines.sh data/datasets/physiworld_pilot_v0.6.0 v05 nominal 20 cuda:0 v05_on_v06
set -uo pipefail
DS="${1:?dataset dir}"; TAG="${2:?tag}"; NOISE="${3:-nominal}"; H="${4:-20}"; DEV="${5:-cuda:0}"; OUT="${6:-$TAG}"
EXP=outputs/experiments; mkdir -p outputs/eval logs
MODELS=(b0 b1 b6a_future b6a_current)
for k in b2 b3 b4 b4_rw b5a b5b b6b_future b6b_current; do
  [ -f "$EXP/${TAG}_${k}/checkpoints/best.pt" ] && MODELS+=("${k}=$EXP/${TAG}_${k}/checkpoints/best.pt") || echo "missing checkpoint for $k (skipped)"
done
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python -m physiworld_uav.evaluation.run --dataset "$DS" --models "${MODELS[@]}" --horizon "$H" --noise "$NOISE" --device "$DEV" \
    --out "outputs/eval/${OUT}.json" 2>&1 | tee "logs/eval_${OUT}.log" | tail -60
