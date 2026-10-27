# -*- coding: utf-8 -*-
r"""DeepSeek API Key 安全轮换工具
================================
用途：把 .env 里的 DEEPSEEK_API_KEY 换成一个新 Key。

特点：
- 输入用 getpass 隐藏（不会显示在屏幕上，也不会进入命令历史）
- 先用新 Key 调一次余额接口校验，**校验通过才写回**，避免把服务写坏
- 写回前自动备份成 .env.bak.YYYYmmdd-HHMMSS
- 只改 DEEPSEEK_API_KEY 这一行，其余配置原样保留

用法（在项目目录下）：
    .\.venv\Scripts\python.exe rotate_key.py
"""
import os
import re
import sys
import time
import shutil
import getpass

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE, ".env")
KEY_LINE = "DEEPSEEK_API_KEY="


def mask(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:5]}***{key[-4:]}"


def validate(key: str):
    """用新 Key 查一次余额，返回 (ok, 说明)"""
    try:
        r = requests.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {key}"}, timeout=12)
    except Exception as e:
        return False, f"网络异常：{type(e).__name__}: {str(e)[:80]}"
    if r.status_code == 200:
        try:
            bi = (r.json().get("balance_infos") or [{}])[0]
            return True, f"余额 ¥{bi.get('total_balance', '?')}（{bi.get('currency', 'CNY')}）"
        except Exception:
            return True, "校验通过"
    if r.status_code in (401, 403):
        return False, f"Key 无效或已失效（HTTP {r.status_code}）"
    return False, f"校验失败（HTTP {r.status_code}）"


def rewrite_env(new_key: str) -> str:
    """把 .env 中的 DEEPSEEK_API_KEY 换成新值，返回备份文件路径"""
    if os.path.exists(ENV_PATH):
        text = open(ENV_PATH, "r", encoding="utf-8").read()
    else:
        text = ""
    backup = ENV_PATH + time.strftime(".bak.%Y%m%d-%H%M%S")
    if os.path.exists(ENV_PATH):
        shutil.copy2(ENV_PATH, backup)
    pattern = re.compile(r"(?m)^\s*" + re.escape(KEY_LINE) + r".*$")
    if pattern.search(text):
        text = pattern.sub(KEY_LINE + new_key, text, count=1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += KEY_LINE + new_key + "\n"
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return backup


def main() -> int:
    print("=" * 52)
    print("  DeepSeek API Key 轮换")
    print("=" * 52)
    print("1) 打开 https://platform.deepseek.com/api_keys 新建一个 Key")
    print("2) 复制新 Key，回到这里粘贴（输入不会显示）")
    print()

    if sys.stdin.isatty():
        key = getpass.getpass("新 Key：").strip()
    else:
        # 支持管道/脚本传入，便于无人值守与自检
        key = sys.stdin.readline().strip()
    if not key:
        print("未输入，已取消。")
        return 1
    if not key.startswith("sk-"):
        print("⚠️ 格式提示：DeepSeek 的 Key 一般以 sk- 开头，请确认没有复制错。")

    print("正在校验……")
    ok, msg = validate(key)
    if not ok:
        print(f"❌ {msg}")
        print("   未写入 .env，旧 Key 仍在使用。")
        return 1
    print(f"✅ 校验通过：{msg}")

    backup = rewrite_env(key)
    print(f"✅ 已写入 .env：{KEY_LINE}{mask(key)}")
    print(f"   原文件备份：{os.path.basename(backup)}")
    print()
    print("下一步：重启网页服务（双击「启动网页版_稳定版.bat」或重启 python math_agent_web.py），")
    print("        旧 Key 可到 DeepSeek 后台删除。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
