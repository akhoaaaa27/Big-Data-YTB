@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
echo ===== %date% %time% ===== >> log.txt
python ytb_data.py >> log.txt 2>&1