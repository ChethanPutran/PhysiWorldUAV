#!/usr/bin/env bash
# Upload datasets and outputs to Google Drive with rclone (run on the HOST, not in the container).
#   scripts/upload_to_gdrive.sh [--dry-run] [--remote gdrive] [--dest PhysiWorldUAV] [--all-datasets]
# Default uploads: pilot v7 dataset, trained runs (best.pt + last.pt, no per-epoch files), evaluation, reports, sweeps.
# Uses `rclone copy` (never deletes on Drive); re-running skips what is already there. Excludes logs, secrets and caches.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
REMOTE=gdrive; DEST=PhysiWorldUAV; DRY=""; ALL=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY="--dry-run";; --remote) REMOTE="$2"; shift;; --dest) DEST="$2"; shift;; --all-datasets) ALL=1;;
  *) echo "unknown option $1"; exit 2;; esac; shift; done
command -v rclone >/dev/null || { echo "rclone not found (expected in ~/bin)"; exit 1; }
rclone listremotes | grep -q "^${REMOTE}:" || { echo "remote '${REMOTE}:' not configured: run 'rclone config' (see docs)"; exit 1; }
COMMON=(-P --transfers 4 --checkers 8 --drive-chunk-size 64M --retries 5 --low-level-retries 20 --exclude ".DS_Store" --exclude "__pycache__/**" --exclude "*.pyc" $DRY)
up() { echo "==> $1 -> ${REMOTE}:${DEST}/$2"; shift; rclone copy "$1" "${REMOTE}:${DEST}/$2" "${COMMON[@]}" "${@:3}"; }
echo "==> datasets"
if [ "$ALL" = 1 ]; then
  for d in data/datasets/physiworld_pilot_v0.5.0 data/datasets/physiworld_pilot_v0.6.0; do rclone copy "$d" "${REMOTE}:${DEST}/datasets/$(basename "$d")" "${COMMON[@]}"; done
fi
rclone copy data/datasets/physiworld_pilot_v7_v0.7.0 "${REMOTE}:${DEST}/datasets/physiworld_pilot_v7_v0.7.0" "${COMMON[@]}"
echo "==> trained runs (best.pt, last.pt, configs, metrics; no epoch_*.pt)"
rclone copy outputs/experiments "${REMOTE}:${DEST}/outputs/experiments" "${COMMON[@]}" --exclude "**/epoch_*.pt"
echo "==> evaluation, reports, sweeps"
rclone copy outputs/eval    "${REMOTE}:${DEST}/outputs/eval"    "${COMMON[@]}"
rclone copy outputs/reports "${REMOTE}:${DEST}/outputs/reports" "${COMMON[@]}"
rclone copy outputs/sweeps  "${REMOTE}:${DEST}/outputs/sweeps"  "${COMMON[@]}"
echo "==> design notes (.idea)"
rclone copy .idea "${REMOTE}:${DEST}/idea" "${COMMON[@]}"
echo "==> slides and report"
rclone copy report/slides/PhysiWorldUAV_midterm.pptx "${REMOTE}:${DEST}/report" "${COMMON[@]}" || true
rclone copy report/slides/PhysiWorldUAV_midterm.pdf  "${REMOTE}:${DEST}/report" "${COMMON[@]}" || true
rclone copy report/midterm_report.pdf                "${REMOTE}:${DEST}/report" "${COMMON[@]}" || true
echo "==> verify the dataset (size and checksum)"
[ -n "$DRY" ] || rclone check data/datasets/physiworld_pilot_v7_v0.7.0 "${REMOTE}:${DEST}/datasets/physiworld_pilot_v7_v0.7.0" --one-way
echo "done"
