#!/usr/bin/env bash
# Wind-estimator probe (evaluation/probe.py) on trained latent models of sweep fix_v1, 3 lanes on GPUs 0-2. Run INSIDE the container.
# Output: outputs/eval/probe_<run>.json, logs/probe_<run>.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DS=data/datasets/physiworld_pilot_v0.6.0; E=outputs/experiments
lane() { local g="$1"; shift
  for r in "$@"; do
    echo "[$(date +%T)] gpu $g: $r"
    python -m physiworld_uav.evaluation.probe --dataset "$DS" --checkpoint "$E/$r/checkpoints/best.pt" --noise nominal \
       --roles val_id test_id test_ood_magnitude test_ood_direction test_ood_gust --device "cuda:$g" \
       --out "outputs/eval/probe_${r}.json" > "logs/probe_${r}.log" 2>&1 && echo "[$(date +%T)] done $r" || echo "[$(date +%T)] FAILED $r"
  done; }
lane 0 v06_b3 fx_v06_b3_b3a_s0 fx_v06_b2_s0 &
lane 1 fx_v06_b3_s1 fx_v06_b3_b3b_s0 fx_v06_b5b_s0 &
lane 2 fx_v06_b3_s2 fx_v06_b3_b3c_s0 &
wait
echo "[$(date +%T)] all probes finished"
