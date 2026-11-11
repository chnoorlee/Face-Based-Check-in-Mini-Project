@echo off
REM ============================================
REM  课堂人脸签到 · 一键启动（双击本文件即可）
REM  打开后直接调用摄像头，人站到镜头前自动签到
REM ============================================
chcp 65001 >nul
cd /d "%~dp0"

set "PYW=.venv\Scripts\pythonw.exe"
if exist "%PYW%" goto run_venv

REM 没有虚拟环境时退回系统 Python
where pythonw >nul 2>nul
if errorlevel 1 (
    echo 未找到 Python 环境。
    echo 请先按 README 安装依赖，或确认 .venv 已创建。
    pause
    exit /b 1
)
start "" pythonw start.py
exit /b 0

:run_venv
start "" "%PYW%" start.py
exit /b 0
