@echo off
title 高考数学Agent 网页版 - 一键启动
cd /d "C:\Users\Administrator\Desktop\gaokao_math_agent"

echo ==============================================
echo   高考数学一体化Agent · 网页版一键启动
echo ==============================================
echo.

REM 若网页服务已在运行，直接打开浏览器
curl -s -o nul -m 3 http://127.0.0.1:7860
if not errorlevel 1 (
    echo 网页服务已在运行，正在打开浏览器...
    start "" "http://127.0.0.1:7860"
    exit /b 0
)

REM 检查 Ollama 是否已运行，未运行则尝试启动
curl -s -o nul -m 3 http://127.0.0.1:11434/api/tags
if errorlevel 1 goto start_ollama
echo [1/2] Ollama 已在运行。
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
    echo  Ollama 启动超时，请手动打开 Ollama 后再重试。
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_ollama

:ollama_ready
echo [1/2] Ollama 已就绪。

:after_ollama
echo [2/2] 正在启动网页服务（新窗口，关闭该窗口即停止服务）...
start "高考数学Agent网页版" cmd /k ""C:\Users\Administrator\Desktop\gaokao_math_agent\.venv\Scripts\python.exe" math_agent_web.py"

timeout /t 8 /nobreak >nul
start "" "http://127.0.0.1:7860"
echo.
echo 启动完成！
echo - 网页服务窗口请保持打开，关闭即停止服务。
echo - 若浏览器未自动打开，请手动访问 http://127.0.0.1:7860
echo.
pause
