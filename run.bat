@echo off
title Stock Market Recommendation Pipeline Engine
color 0A
echo ======================================================================
echo [Stock Market Recommendation] Starting Daily Quantitative Pipeline Execution...
echo ======================================================================
powershell -ExecutionPolicy Bypass -File "%~dp0run_daily.ps1"
echo.
echo Press any key to exit...
pause > nul
