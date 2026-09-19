# ==============================================================================
# Stock Market Recommendation - Daily Pipeline Execution (PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "[Stock Market Recommendation] Starting Daily Quantitative Pipeline Execution..." -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan

# Step 1: Ingest Market Data
Write-Host "[Step 1/3] Running Data Ingestion Engine (src/01_data_ingestion.py)..." -ForegroundColor Yellow
python -m src.01_data_ingestion

# Step 2: Feature Engineering
Write-Host "[Step 2/3] Running Feature Engineering Pipeline (src/02_feature_eng.py)..." -ForegroundColor Yellow
python -m src.02_feature_eng

# Step 3: Model Training & Inference
Write-Host "[Step 3/4] Running Model Inference Engine (src/03_model_inference.py)..." -ForegroundColor Yellow
python -m src.03_model_inference

# Step 4: Automated Morning Market Brief
Write-Host "[Step 4/4] Generating Institutional Daily Morning Brief (src/morning_brief.py)..." -ForegroundColor Yellow
python -m src.morning_brief

Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
Write-Host "[Stock Market Recommendation] Daily Pipeline Execution Completed Successfully!" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
