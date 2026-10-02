# ==============================================================================
# Stock Market Recommendation - Daily Pipeline Execution (PowerShell)
# ==============================================================================
param (
    [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "[Stock Market Recommendation] Two-Phase Pipeline Execution (Mode: $Mode)" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan

python -m src.pipeline_runner --mode $Mode

Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
Write-Host "[Stock Market Recommendation] Pipeline Execution Completed Successfully!" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------" -ForegroundColor Green
