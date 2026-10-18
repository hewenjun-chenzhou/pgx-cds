@echo off
chcp 65001 >nul
title PGx CDS · Streamlit

REM ============================================
REM  PGx CDS 一键启动
REM  - 双击本文件即可启动
REM  - 关闭此窗口 = 停止服务
REM ============================================

cd /d "%~dp0"

REM 检查 Python
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 如果 8501 端口已被占用，先停掉旧实例
echo 检查端口 8501 ...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo 发现旧实例 PID=%%a，正在停止...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 >nul
)

REM 自动打开浏览器（3 秒后）
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8501"

REM 启动 streamlit（阻塞，关闭本窗口 = 停止）
echo ============================================
echo  PGx CDS 启动中...
echo  地址: http://127.0.0.1:8501
echo  关闭本窗口即可停止
echo ============================================
python -m streamlit run app.py --server.port 8501 --server.headless false --server.address 127.0.0.1 --browser.gatherUsageStats false

pause
