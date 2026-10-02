#!/bin/bash
# ==============================================================================
# Stock Market Recommendation - Daily Pipeline Execution Script
# ==============================================================================
set -e

MODE="${1:-auto}"

echo "----------------------------------------------------------------------"
echo "[Stock Market Recommendation] Two-Phase Pipeline Execution (Mode: $MODE)"
echo "Timestamp: $(date)"
echo "----------------------------------------------------------------------"

python -m src.pipeline_runner --mode "$MODE"

echo "----------------------------------------------------------------------"
echo "[Stock Market Recommendation] Pipeline Execution Completed Successfully!"
echo "----------------------------------------------------------------------"
