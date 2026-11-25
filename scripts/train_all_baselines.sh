#!/usr/bin/env bash
# Train all trainable baselines on one dataset, spread over GPUs (one job per GPU at a time). Run INSIDE the container.
#
#   scripts/train_all_baselines.sh <dataset_dir> <tag> [gpus=0,1,2] [extra training YAMLs...]
#   e.g. scripts/train_all_baselines.sh data/datasets/physiworld_pilot_v0.6.0 v06 0,1,2
#
# Kinds: b2 b3 b4 b4_rw b5a b5b b6b_future b6b_current (b0, b1, b6a_* need no training: see scripts/eval_baselines.sh).
# Run names: <tag>_<kind>, output outputs/experiments/<tag>_<kind>/, logs logs/train_<tag>_<kind>.log.
# Config: config/training/default.yaml + config/training/compare.yaml unless other YAMLs are given. Same seed and settings for every kind.
set -uo pipefail
DS="${1:?dataset dir}"; TAG="${2:?tag}"; GPUS="${3:-0,1,2}"; shift 3 2>/dev/null || shift $#
YAMLS=("$@"); [ ${#YAMLS[@]} -gt 0 ] || YAMLS=(config/training/default.yaml config/training/compare.yaml)
KINDS=(b2 b3 b4 b4_rw b5a b5b b6b_future b6b_current)
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p logs
run_queue() {                                     # one GPU, its kinds in sequence
  local g="$1"; shift
  for k in "$@"; do
    echo "[$(date +%T)] gpu $g: start $k"
    python -m physiworld_uav.training.run --dataset "$DS" --name "${TAG}_${k}" --model "$k" --device "cuda:$g" \
        --config "${YAMLS[@]}" > "logs/train_${TAG}_${k}.log" 2>&1 \
      && echo "[$(date +%T)] gpu $g: done  $k" || echo "[$(date +%T)] gpu $g: FAILED $k (see logs/train_${TAG}_${k}.log)"
  done
}
for i in "${!GPU_LIST[@]}"; do
  q=(); for j in "${!KINDS[@]}"; do [ $((j % ${#GPU_LIST[@]})) -eq "$i" ] && q+=("${KINDS[$j]}"); done
  run_queue "${GPU_LIST[$i]}" "${q[@]}" &
done
wait
echo "[$(date +%T)] all jobs finished"
