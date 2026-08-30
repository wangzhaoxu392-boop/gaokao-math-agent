@echo off
chcp 65001 >nul
title 高考数学Agent 网页版（稳定版·自动重启）
cd /d "C:\Users\Administrator\Desktop\gaokao_math_agent"

echo ==============================================
echo   高考数学一体化Agent - 网页版稳定启动
echo   （服务崩溃后自动重启，批量做题不中断）
echo ==============================================
echo.

REM ---- 检查是否已在运行 ----
curl -s -o nul -w "%%{http_code}" -m 4 http://127.0.0.1:7860 > _web_check.txt
set /p WEB_CODE=< _web_check.txt
del _web_check.txt >nul 2>&1
if "%WEB_CODE%"=="200" (
    echo [OK] 网页服务已在运行，正在打开浏览器...
    start "" "http://127.0.0.1:7860"
    exit /b 0
)

REM ---- 清理残留死进程 ----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo [清理] 已清理残留的网页进程

REM ---- 检查 Ollama ----
curl -s -o nul -m 3 http://127.0.0.1:11434/api/tags
if errorlevel 1 goto start_ollama
echo [1/2] Ollama 正在运行中。
goto after_ollama

:start_ollama
echo [1/2] 正在启动 Ollama...
start "" "C:\Users\Administrator\AppData\Local\Programs\Ollama\ollama app.exe"
set /a n=0

:wait_ollama
set /a n+=1
curl -s -o nul -m 3 http://127.0.0.1:11434/api/tags
if not errorlevel 1 goto ollama_ready
if %n% geq 60 (
    echo [错误] Ollama 启动超时，请手动打开 Ollama 后再试。
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_ollama

:ollama_ready
echo [1/2] Ollama 已经就绪

:after_ollama
echo [2/2] 正在启动网页服务（崩溃自动重启）...
echo.
echo 提示：关闭本窗口即停止服务。
echo       批量做题中断后重新运行会自动续跑（断点续跑）。
echo.

REM ---- 自动重启循环：服务退出后等待2秒重启 ----
set /a crash_count=0
:restart_loop
set /a crash_count+=1
if %crash_count% gtr 1 (
    echo.
    echo [自动重启] 检测到服务退出，第 %crash_count% 次重启中...
    echo [自动重启] 批量做题会从断点续跑，已完成的题不会重做
    timeout /t 2 /nobreak >nul
)

REM 启动服务，首次就绪后打开浏览器
if %crash_count%==1 (
    start "高考数学Agent网页版" cmd /k ""C:\Users\Administrator\Desktop\gaokao_math_agent\.venv\Scripts\python.exe" -X utf8 math_agent_web.py"
    REM 等待服务就绪后打开浏览器
    set /a n=0
    :wait_web
    set /a n+=1
    curl -s -o nul -w "%%{http_code}" -m 3 http://127.0.0.1:7860 > _web_check2.txt
    set /p WEB_CODE2=< _web_check2.txt
    del _web_check2.txt >nul 2>&1
    if "%WEB_CODE2%"=="200" goto web_ready
    if %n% geq 25 (
        echo [提示] 服务启动较慢，稍后请手动访问 http://127.0.0.1:7860
        goto done
    )
    timeout /t 2 /nobreak >nul
    goto wait_web

    :web_ready
    start "" "http://127.0.0.1:7860"
)

REM 监控服务进程：如果 python 进程不存在，触发重启
echo [监控中] 服务运行中，正在监控进程状态...
:monitor_loop
timeout /t 5 /nobreak >nul
tasklist /fi "imagename eq python.exe" 2>nul | findstr /i "python.exe" >nul
if errorlevel 1 (
    echo [监控] 检测到 python 进程已退出，触发自动重启...
    goto restart_loop
)
goto monitor_loop

:done
echo.
echo 启动完成！
echo - 网页服务窗口请保持打开，关闭即停止服务
echo - 如果浏览器未自动打开，请手动访问 http://127.0.0.1:7860
echo - 批量做题支持断点续跑，中断后重新运行即可继续
echo.
pause
