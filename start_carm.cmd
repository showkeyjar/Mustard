@echo off
setlocal
cd /d "%~dp0"
cmd /d /c python -m scripts.desktop_agent_control launch
if %errorlevel% neq 0 (
  echo.
  echo CARM launch failed.
  pause
  exit /b 1
)
exit /b 0
