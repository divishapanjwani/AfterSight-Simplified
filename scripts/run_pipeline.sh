#!/usr/bin/env bash
# AfterSight: build everything from the xBD Parquet shards, in order.
#   inspect (+ scene cache) -> features -> splits -> train (+ K-Means, confounding) -> review thresholds
#   -> Phase 3b experiments -> demo cache
# Usage (from anywhere):  scripts/run_pipeline.sh            # full run (~10 min on a laptop)
#                         scripts/run_pipeline.sh --no-experiments
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv at ./.venv — create it first (README → Quick start)." >&2; exit 1
fi
if ! compgen -G "data/raw/xbd_parquet/*.parquet" > /dev/null; then
  echo "No Parquet shards in data/raw/xbd_parquet/ — see data/README.md for where to put xBD." >&2; exit 1
fi
EXPERIMENTS=1
[[ "${1:-}" == "--no-experiments" ]] && EXPERIMENTS=0

PY=.venv/bin/python
started=$(date +%s)
step() {
  local label=$1; shift
  echo; echo "== $label"
  local t=$(date +%s)
  "$@"
  echo "   ($(( $(date +%s) - t )) s)"
}

step "1/7 inspect the shards and fill the scene cache" $PY scripts/inspect_dataset.py --alignment
step "2/7 per-building features"                       $PY scripts/build_features.py --workers 4
step "3/7 scene-grouped split manifests"               $PY scripts/make_splits.py
step "4/7 train + evaluate models (test sets once)"    $PY scripts/train.py
step "5/7 review thresholds (validation)"              $PY scripts/calibrate_review.py
if [[ $EXPERIMENTS == 1 ]]; then
  step "6/7 feature-set experiments (Phase 3b)"     $PY scripts/run_phase3b.py
else
  echo; echo "== 6/7 feature-set experiments skipped (--no-experiments)"
fi
step "7/7 pre-cache the demo scenes"                   $PY scripts/prepare_demo.py

echo; echo "Pipeline finished in $(( $(date +%s) - started )) s. Start the app with: ./start.sh"
