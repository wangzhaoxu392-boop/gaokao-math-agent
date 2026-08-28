@echo off
chcp 65001 >nul
title 高考数学Agent 网页版 - 一键启动
cd /d "C:\Users\Administrator\Desktop\gaokao_math_agent"

echo ==============================================
echo   高考数学一体化Agent - 网页版一键启动
echo ==============================================
echo.

REM ---- 第一步：检查网页是否真的可访问（HTTP 200）----
curl -s -o nul -w "%%{http_code}" -m 4 http://127.0.0.1:7860 > _web_check.txt
set /p WEB_CODE=< _web_check.txt
del _web_check.txt >nul 2>&1
if "%WEB_CODE%"=="200" (
    echo [OK] 网页服务已在正常运行，正在打开浏览器...
    start "" "http://127.0.0.1:7860"
    exit /b 0
)

REM ---- 第二步：清理可能残留的死进程（端口被占但页面不可用）----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo [清理] 已清理残留的网页进程

REM ---- 第三步：检查 Ollama，未运行则自动启动 ----
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
echo [2/2] 正在启动网页服务（新窗口），关闭该窗口即停止服务...
start "高考数学Agent网页版" cmd /k ""C:\Users\Administrator\Desktop\gaokao_math_agent\.venv\Scripts\python.exe" math_agent_web.py"

REM ---- 第四步：等待服务真正就绪（最多 40 秒）再打开浏览器 ----
set /a n=0
:wait_web
set /a n+=1
curl -s -o nul -w "%%{http_code}" -m 3 http://127.0.0.1:7860 > _web_check2.txt
set /p WEB_CODE2=< _web_check2.txt
del _web_check2.txt >nul 2>&1
if "%WEB_CODE2%"=="200" goto web_ready
if %n% geq 20 (
    echo [提示] 服务启动较慢，稍后请手动访问 http://127.0.0.1:7860
    start "" "http://127.0.0.1:7860"
    goto done
)
timeout /t 2 /nobreak >nul
goto wait_web

:web_ready
start "" "http://127.0.0.1:7860"

:done
echo.
echo 启动完成！
echo - 网页服务窗口请保持打开，关闭即停止服务
echo - 如果浏览器未自动打开，请手动访问 http://127.0.0.1:7860
echo.
pause
