$ErrorActionPreference = "SilentlyContinue"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  高考数学一体化Agent - 网页版启动" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已在运行
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) {
        Write-Host "[OK] 网页服务已在运行，正在打开浏览器..." -ForegroundColor Green
        Start-Process "http://127.0.0.1:7860"
        exit 0
    }
} catch {
    # 未运行，继续启动
}

# 清理残留进程
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "[清理] 已清理残留进程"

# 启动网页服务（独立窗口）
Write-Host "[启动] 正在启动网页服务（首次加载约10-30秒）..." -ForegroundColor Yellow
Start-Process -FilePath "powershell.exe" -ArgumentList '-NoProfile','-Command','cd "C:\Users\Administrator\Desktop\gaokao_math_agent"; .\.venv\Scripts\python.exe -X utf8 math_agent_web.py' -WorkingDirectory "C:\Users\Administrator\Desktop\gaokao_math_agent"

# 等待服务就绪
Write-Host "[等待] 等待服务就绪..." -ForegroundColor Yellow
$n = 0
while ($n -lt 40) {
    $n++
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Host "[OK] 网页服务已就绪！" -ForegroundColor Green
            Start-Process "http://127.0.0.1:7860"
            exit 0
        }
    } catch {
        # 继续等待
    }
}

Write-Host "[提示] 服务启动较慢，请手动访问 http://127.0.0.1:7860" -ForegroundColor Yellow
exit 0
