#!/bin/bash
# ==============================================================================
# AlphaTech Asynchronous Alpha Generation - Daily Pipeline Execution Script
# ==============================================================================
set -e

echo "----------------------------------------------------------------------"
echo "[AlphaTech] Starting Daily Quantitative Pipeline Execution..."
echo "Timestamp: $(date)"
echo "----------------------------------------------------------------------"

# Step 1: Ingest OHLCV Raw Market Data
echo "[Step 1/3] Running Data Ingestion Engine (src/01_data_ingestion.py)..."
python -m src.01_data_ingestion

# Step 2: Feature Engineering & Alpha Signal Extraction
echo "[Step 2/3] Running Feature Engineering Pipeline (src/02_feature_eng.py)..."
python -m src.02_feature_eng

# Step 3: Model Training & Real-Time Alpha Inference
echo "[Step 3/4] Running Model Inference & Recommendation Engine (src/03_model_inference.py)..."
python -m src.03_model_inference

# Step 4: Institutional Daily Morning Brief
echo "[Step 4/4] Generating Institutional Morning Brief (src/morning_brief.py)..."
python -m src.morning_brief

echo "----------------------------------------------------------------------"
echo "[AlphaTech] Daily Pipeline Execution Completed Successfully!"
echo "----------------------------------------------------------------------"
