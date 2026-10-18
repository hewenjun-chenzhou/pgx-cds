@echo off
chcp 65001 >nul
title PGx CDS · Stop

echo 正在停止 PGx CDS（端口 8501）...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo 已停止 PID %%a
)

REM 兜底：杀掉所有 streamlit 进程
for /f "tokens=1" %%p in ('wmic process where "name='python.exe' and commandline like '%%streamlit run%%'" get processid ^| findstr /r "[0-9]"') do (
    taskkill /F /PID %%p >nul 2>&1
)

echo 完成。
timeout /t 3 >nul
