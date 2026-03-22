@echo off
setlocal
cd /d "%~dp0"
cmd /d /c python -m scripts.claw_team_control run
if %errorlevel% neq 0 (
  echo.
  echo Mustard Claw Team launch failed.
  pause
  exit /b 1
)
exit /b 0
