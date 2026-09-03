# -*- coding: utf-8 -*-
"""
一键启动网页服务（快速版 v2）
- 检查服务是否已运行 → 是则直接打开浏览器
- 否则精确清理占用端口的残留进程 → 启动服务 → 快速轮询就绪 → 打开浏览器
"""
import os
import sys
import time
import socket
import webbrowser
import subprocess

BASE_DIR = r"C:\Users\Administrator\Desktop\gaokao_math_agent"
PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
WEB_PY = os.path.join(BASE_DIR, "math_agent_web.py")
URL = "http://127.0.0.1:7860"
PORT = 7860


def check(url, timeout=1):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def kill_port_owner(port):
    """精确杀掉占用指定端口的进程（不误杀自己）"""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    if p.pid != os.getpid():
                        p.kill()
                        print(f"[清理] 已结束残留进程 PID {p.pid}")
                except Exception:
                    pass
    except ImportError:
        pass


def main():
    print("=" * 46)
    print("  高考数学一体化Agent - 网页版启动（快速版）")
    print("=" * 46)
    print()

    # 1. 已在运行 → 直接打开浏览器
    if check(URL):
        print("[OK] 网页服务已在运行，正在打开浏览器...")
        webbrowser.open(URL)
        return 0

    # 2. 精确清理占用 7860 端口的残留进程
    kill_port_owner(PORT)
    time.sleep(1)

    # 3. 启动网页服务（新窗口）
    print("[启动] 正在启动网页服务（首次加载约5-15秒）...")
    try:
        subprocess.Popen(
            [PYTHON, "-X", "utf8", WEB_PY],
            cwd=BASE_DIR,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    except Exception as e:
        print(f"[错误] 启动失败: {e}")
        return 1

    # 4. 快速轮询等待就绪
    print("[等待] 等待服务就绪...")
    n = 0
    while n < 60:
        n += 1
        if check(URL, timeout=1):
            print("[OK] 网页服务已就绪，正在打开浏览器！")
            webbrowser.open(URL)
            return 0
        time.sleep(1)

    print("[提示] 服务启动较慢，请手动访问 " + URL)
    webbrowser.open(URL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
