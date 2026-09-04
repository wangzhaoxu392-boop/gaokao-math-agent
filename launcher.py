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


def kill_stale_web_processes():
    """清理所有残留的 math_agent_web.py 进程（含卡死/未监听/重复实例），不误杀自己。
    防止多次启动后进程堆积导致端口竞争、服务起不来、网页打不开。"""
    try:
        import psutil
        my_pid = os.getpid()
        killed = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if p.info["name"] and p.info["name"].lower().startswith("python"):
                    cmd = " ".join(p.info["cmdline"] or [])
                    if "math_agent_web.py" in cmd and p.info["pid"] != my_pid:
                        # 跳过当前 launcher 自身
                        p.kill()
                        killed.append(p.info["pid"])
            except Exception:
                continue
        if killed:
            print(f"[清理] 已结束 {len(killed)} 个残留服务进程: {killed}")
    except ImportError:
        pass


def main():
    print("=" * 46)
    print("  高考数学一体化Agent - 网页版启动（快速版）")
    print("=" * 46)
    print()

    # 0. 先清理所有残留的 math_agent_web.py 进程（防止重复实例堆积）
    kill_stale_web_processes()
    time.sleep(1)

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
