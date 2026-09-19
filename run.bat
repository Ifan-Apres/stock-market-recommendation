@echo off
title AlphaTech Quantitative Pipeline Engine
color 0A
echo ======================================================================
echo [AlphaTech] Starting Daily Quantitative Pipeline Execution...
echo ======================================================================
powershell -ExecutionPolicy Bypass -File "%~dp0run_daily.ps1"
echo.
echo Press any key to exit...
pause > nul
