@echo off
rem 启动 GUI，或直接跑某个脚本：run.bat train.py --steps 1000
rem
rem 训练用的解释器优先级：
rem   1. 环境变量 VOICEFILTER_PYTHON
rem   2. 同目录下的 python_path.txt（写一行解释器路径，已被 .gitignore 忽略）
rem   3. PATH 里的 python
set "PY=%VOICEFILTER_PYTHON%"
if not defined PY if exist "%~dp0python_path.txt" set /p PY=<"%~dp0python_path.txt"
if not defined PY set "PY=python"

if "%~1"=="" (
  "%PY%" "%~dp0app.py"
) else (
  "%PY%" "%~dp0%~1" %2 %3 %4 %5 %6 %7 %8 %9
)
pause
