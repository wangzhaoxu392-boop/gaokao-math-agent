# -*- coding: utf-8 -*-
r"""
高考数学一体化Agent —— 网页操作版（Gradio）
==============================================
启动方式：
    cd C:\Users\Administrator\Desktop\gaokao_math_agent
    .\.venv\Scripts\python.exe math_agent_web.py
访问地址： http://127.0.0.1:7860
"""
import os
import io
import json
import time
import shutil
import threading
from pathlib import Path
from contextlib import redirect_stdout
import gradio as gr

# 无论从哪里启动，都切到脚本所在目录，保证相对路径正确
os.chdir(Path(__file__).resolve().parent)

# 复用原有全部核心逻辑（math_agent_all 的主菜单在 __main__ 里，import 不会触发）
import math_agent_all as maa

OUTPUT_FOLDER = Path(maa.OUTPUT_FOLDER)
OUT_RESEARCH_FOLDER = Path(maa.OUT_RESEARCH_FOLDER)
KNOWLEDGE_FOLDER = Path(maa.KNOWLEDGE_FOLDER)
RAW_EXAM_FOLDER = Path(maa.RAW_EXAM_FOLDER)
BATCH_INPUT_FOLDER = Path(maa.BATCH_INPUT_FOLDER)
ALLOWED_BATCH = (".txt", ".md", ".docx", ".pdf")

# 批量做题的"模型档位"：只覆盖数学模型，推理模型不受影响
_MATH_TIERS = [("标准（" + maa._MATH_MODEL_NAME + "）", "")]
if maa._MATH_PROVIDER == "deepseek":
    _MATH_TIERS.append(("省钱（deepseek-flash）", "deepseek-flash"))
elif maa._MATH_PROVIDER == "siliconflow":
    _MATH_TIERS.append(("省钱（Qwen2.5-7B-Instruct）", "Qwen/Qwen2.5-7B-Instruct"))

# ==================== 长任务协作式停止 ====================
_CANCEL = {"batch": False, "grade": False, "research": False, "fine": False, "wiki": False}


def _reset_cancel(key):
    _CANCEL[key] = False


def _should_stop(key):
    return bool(_CANCEL.get(key, False))


def _make_stop(key, label):
    def _stop():
        _CANCEL[key] = True
        return f"⏹ 已收到停止请求：{label}将在当前这一步结束后停下，已完成的结果会保留。"
    _stop.__name__ = f"stop_{key}"
    return _stop


stop_batch = _make_stop("batch", "批量做题")
stop_grade = _make_stop("grade", "判分")
stop_research = _make_stop("research", "真题教研")
stop_fine = _make_stop("fine", "讲义生成")
stop_wiki = _make_stop("wiki", "Wiki 编译")

# 界面文案按 .env 实际配置动态生成，避免"写着本地 Ollama、实际走云端 API"的误导
_LOCAL_LLM = os.getenv("LLM_MODEL", "deepseek-r1:14b")
_FAST_LLM = os.getenv("FAST_MODEL", "qwen2.5:7b")
_REASON_LABEL = f"推理模型（本地 {_LOCAL_LLM}）"
if maa._MATH_PROVIDER in ("deepseek", "siliconflow"):
    _MATH_LABEL = f"数学模型（云端 {maa._MATH_PROVIDER}: {maa._MATH_MODEL_NAME}）"
    _ENGINE_DESC = (
        f"推理走本地 Ollama（{_LOCAL_LLM} + {_FAST_LLM}），"
        f"数学计算走云端 {maa._MATH_PROVIDER} API（{maa._MATH_MODEL_NAME}）。"
        f"简单题约 1 分钟，大题约 1~5 分钟。")
else:
    _MATH_LABEL = f"数学模型（本地 {maa._MATH_MODEL_NAME}）"
    _ENGINE_DESC = (
        f"全部本地 Ollama 驱动（{_LOCAL_LLM} + {_FAST_LLM} + {maa._MATH_MODEL_NAME}）。"
        f"简单题约 1 分钟，大题约 3~8 分钟。")


def _run_with_console(fn, *args, **kwargs):
    """运行函数并捕获其 print 输出，返回 (输出文本, 函数返回值)"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


def _save_uploads(files, folder):
    if not files:
        return "未选择文件。"
    saved = []
    for f in files:
        name = Path(f.name).name
        shutil.copy(f, folder / name)
        saved.append(name)
    return f"已上传到 {folder.name}/：{'、'.join(saved)}"


def _list_files(folder, exts=None):
    if not folder.exists():
        return []
    return sorted(
        p.name for p in folder.iterdir()
        if p.is_file() and (exts is None or p.suffix.lower() in exts)
    )


# ==================== 功能1：单题交互 ====================
# 固定会话ID实现跨题记忆（追问可衔接）；记忆上限由 solve_agent 的记忆窗口控制，不会无限膨胀
_WEB_THREAD = "web_chat"

def clear_memory():
    try:
        maa.graph_chat.update_state(
            {"configurable": {"thread_id": _WEB_THREAD}, "recursion_limit": maa._RECURSION_LIMIT},
            {"messages": []})
        return "✅ 记忆已清空，开始全新对话"
    except Exception as e:
        return f"清空失败：{e}"

def solve_one(question, model_choice, math_model=""):
    if not question or not question.strip():
        yield "请输入题目。", None, ""
        return
    model_label = {"auto": "自动（按题型）", "reasoning": _REASON_LABEL, "math": _MATH_LABEL}.get(model_choice, model_choice)
    yield (f"⏳ 正在解题中（模型：{model_label}，简单题约1分钟，大题约3~5分钟），"
           f"请勿关闭页面，耐心等待...", None, "")
    cfg = {"configurable": {"thread_id": _WEB_THREAD}, "recursion_limit": maa._RECURSION_LIMIT}
    plot = OUTPUT_FOLDER / "plot_out.png"
    before = plot.stat().st_mtime if plot.exists() else None
    t0 = time.time()
    try:
        ans, model_used = maa.run_question(maa.graph_chat, question, cfg, model_choice=model_choice,
                                           math_model=(math_model or None))
    except Exception as e:
        emsg = str(e)
        elow = emsg.lower()
        if any(k in elow for k in ["401", "403", "authentication", "invalid api key", "余额", "insufficient", "402"]):
            if maa._MATH_PROVIDER == "deepseek":
                hint = "API 密钥无效或余额不足。请到 https://platform.deepseek.com/usage 查看余额，或在 .env 检查 DEEPSEEK_API_KEY。"
            else:
                hint = "API 密钥无效或余额不足。请到 https://siliconflow.cn/account/fee 查看余额并充值，或在 .env 检查 SILICONFLOW_API_KEY。"
        elif "预算" in emsg or "DAILY_BUDGET" in emsg:
            hint = emsg
        elif any(k in elow for k in ["timeout", "timed out", "connection", "network", "连接", "超时"]):
            hint = "网络连接超时或中断。请检查网络后重试；若持续出现，可稍后再试。"
        elif any(k in elow for k in ["ollama", "connection refused", "拒绝连接"]):
            hint = "本地 Ollama 服务未启动。请先在终端运行 ollama serve 或启动 Ollama 应用。"
        else:
            hint = f"解题出错（{type(e).__name__}）：{emsg[:200]}"
        yield f"❌ {hint}", None, ""
        return
    dt = time.time() - t0
    after = plot.stat().st_mtime if plot.exists() else None
    img = str(plot) if (after and after != before) else None
    st = maa._LAST_RUN_STATS
    stats = (f"⏱ 本题 {dt:.0f} 秒 · 模型 `{model_used}` · "
             f"输入 {st.get('prompt', 0):,} / 输出 {st.get('completion', 0):,} token · "
             f"约 ¥{st.get('cost', 0):.4f}")
    try:
        _append_history(question, ans, model_used, st, dt)
    except Exception:
        pass
    yield f"（本题用时 {dt:.0f} 秒，使用模型：{model_used}）\n\n{ans}", img, stats


# ==================== 功能2：LLM Wiki 知识库（Karpathy 改造） ====================
def build_kb():
    """（备选）旧版 RAG 向量库构建，未编译 Wiki 时作为回退"""
    files = _list_files(KNOWLEDGE_FOLDER)
    info = f"📁 knowledge_docs 现有文件：{files if files else '（空）'}\n\n"
    log, _ = _run_with_console(maa.build_rag_vector_db)
    return info + log


def upload_kb(files):
    msg = _save_uploads(files, KNOWLEDGE_FOLDER)
    return msg + "\n上传后请点击【编译 Wiki 知识库】使其生效。", (
        "、".join(_list_files(KNOWLEDGE_FOLDER)) or "（空）")


def _wiki_status():
    try:
        return maa.wiki_summary()
    except Exception:
        return "Wiki 概况读取失败"


def run_build_wiki(model_choice="auto"):
    yield "⏳ Wiki 编译启动中（按资料量数分钟到数十分钟），实时进度如下：\n", None
    _reset_cancel("wiki")
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.build_wiki(progress_cb=progress.append, model_choice=model_choice,
                               should_stop=lambda: _should_stop("wiki"))
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ Wiki 编译中，实时进度如下：\n"
    seen = 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield acc, None
        time.sleep(1.5)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    log = result.get("log", "")
    yield f"✅ Wiki 编译结束！完整日志：\n{log}", _wiki_status()


def run_wiki_lint():
    yield "⏳ Wiki 自检中...\n", None
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.wiki_lint(progress_cb=progress.append)
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ 自检中...\n"
    seen = 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield acc, None
        time.sleep(1.0)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    yield result.get("log", "自检完成")


def run_wiki_heal():
    yield "⏳ Wiki 修复中（自动重定向断裂链接/降级为纯文本）...\n", None
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.wiki_heal(progress_cb=progress.append)
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ 修复中...\n"
    seen = 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield acc, None
        time.sleep(1.0)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    yield result.get("log", "修复完成"), _wiki_status()


# ==================== 功能3：批量做题 ====================
def refresh_batch():
    return gr.update(choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))


def upload_batch(files):
    msg = _save_uploads(files, BATCH_INPUT_FOLDER)
    return msg, gr.update(choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))


def run_batch(files, model_choice="auto", math_model=""):
    if not files:
        yield "请先选择试卷文件。", None, None
        return
    tier = math_model or "标准"
    _reset_cancel("batch")
    yield f"⏳ 批量做题启动中（模型档位：{tier}，每题约需数分钟），下方实时显示进度...\n", None, None
    progress = []
    done = threading.Event()
    result = {}
    rows = []

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                for fname in files:
                    progress.append(f"=== 开始处理文件：{fname} ===")
                    full_text, docx_path = maa.batch_solve_file(
                        fname, progress_cb=progress.append, model_choice=model_choice,
                        math_model=(math_model or None),
                        should_stop=lambda: _should_stop("batch"))
                    pdf_path = OUTPUT_FOLDER / (Path(fname).stem + "_result.pdf")
                    maa.text_to_pdf(full_text, str(pdf_path))
                    st = dict(maa.LAST_BATCH_STATS or {})
                    for it in st.get("items", []):
                        rows.append([
                            Path(fname).name,
                            it.get("no"),
                            "✅" if it.get("success") else "❌",
                            (it.get("model") or "").split("（")[0][:18],
                            f"{float(it.get('seconds') or 0):.0f}s",
                            f"¥{float(it.get('cost') or 0):.4f}",
                            f"{int(it.get('prompt') or 0):,}/{int(it.get('completion') or 0):,}",
                        ])
                    progress.append(
                        f"=== {fname} 完成：{st.get('success', '?')} 题成功 / "
                        f"{st.get('failed', '?')} 题失败，约 ¥{float(st.get('cost') or 0):.4f} ===")
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ 批量做题进行中，实时进度如下：\n"
    seen = 0
    t_start = time.time()
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        elapsed = int(time.time() - t_start)
        if elapsed >= 60:
            acc += f"⏱ 已运行 {elapsed//60} 分 {elapsed%60} 秒，仍在处理中请耐心等待\n"
        yield acc, None, None
        time.sleep(1.0)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    elapsed = int(time.time() - t_start)
    acc += f"⏱ 总耗时 {elapsed//60} 分 {elapsed%60} 秒\n"
    log = result.get("log", "")
    outs = []
    for fname in files:
        for p in (OUTPUT_FOLDER / (Path(fname).stem + "_result.docx"),
                  OUTPUT_FOLDER / (Path(fname).stem + "_result.pdf")):
            if Path(p).exists():
                outs.append(str(p))
    total_cost = sum(float(r[5].lstrip("¥")) for r in rows) if rows else 0.0
    yield (f"✅ 批量做题完成！共 {len(rows)} 题，累计约 ¥{total_cost:.4f}。完整日志：\n{log}",
           (outs or None), (rows or None))


# ==================== 功能4：真题教研 ====================
def refresh_exam():
    return gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def upload_exam(files):
    msg = _save_uploads(files, RAW_EXAM_FOLDER)
    return msg, gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def run_research(files, model_choice="auto"):
    if not files:
        yield "请先选择要教研的真题文件。", None
        return
    yield "⏳ 教研启动中（每题约需数分钟，全程可能 1~4 小时），下方会实时显示进度...\n", None
    _reset_cancel("research")
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.research_workflow(only_files=list(files),
                                      progress_cb=progress.append,
                                      model_choice=model_choice,
                                      should_stop=lambda: _should_stop("research"))
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ 教研进行中，实时进度如下：\n"
    seen = 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield acc, None
        time.sleep(1.5)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    log = result.get("log", "")
    outs = [
        str(p) for p in (
            OUT_RESEARCH_FOLDER / "高考数学教研汇总.docx",
            OUT_RESEARCH_FOLDER / "高考数学教研汇总.pdf",
        ) if p.exists()
    ]
    yield f"✅ 教研完成！完整日志：\n{log}", (outs or None)


# ==================== 功能5：文件转MD ====================
def convert_to_md_ui(file_obj):
    if file_obj is None:
        return "请先上传文件。", None
    try:
        fpath = Path(file_obj.name) if hasattr(file_obj, "name") else Path(str(file_obj))
        md_text, md_path = maa.convert_file_to_md(fpath)
        preview = f"✅ 转换成功！共 {len(md_text)} 字符\n\n" + md_text[:8000]
        if len(md_text) > 8000:
            preview += "\n\n...（内容较长，完整内容请下载MD文件查看）"
        return preview, md_path
    except Exception as e:
        return f"❌ 转换失败：{type(e).__name__}: {str(e)[:300]}", None


# ==================== 功能5辅助：MD转Word ====================
def convert_md_to_docx_ui(file_obj):
    if file_obj is None:
        return "请先上传 MD 文件。", None
    try:
        fpath = Path(file_obj.name) if hasattr(file_obj, "name") else Path(str(file_obj))
        out_path, msg = maa.convert_md_to_docx(fpath)
        preview = f"✅ {msg}\n\n路径：{out_path}"
        return preview, out_path
    except Exception as e:
        return f"❌ 转换失败：{type(e).__name__}: {str(e)}", None


# ==================== 功能6：精细化内容生成 ====================
def fine_subjects():
    """返回学科列表"""
    return list(maa.FINE_TOPICS.keys())


def fine_categories(subject):
    """根据学科返回篇章列表"""
    if subject in maa.FINE_TOPICS:
        return list(maa.FINE_TOPICS[subject].keys())
    return []


def fine_topics(subject, category):
    """根据学科+篇章返回专题列表"""
    if subject in maa.FINE_TOPICS and category in maa.FINE_TOPICS[subject]:
        return maa.FINE_TOPICS[subject][category]
    return []


def _fine_update_cats(subject):
    cats = fine_categories(subject)
    return gr.update(choices=cats, value=(cats[0] if cats else None))


def _fine_update_topics(subject, category):
    tops = fine_topics(subject, category)
    return gr.update(choices=tops, value=(tops[0] if tops else None))


def run_fine(subject, category, topic, model_choice):
    if not topic:
        yield "请选择专题。", None
        return
    yield f"⏳ 正在生成「{subject} · {topic}」精细化内容（模型：{model_choice}），约 1~3 分钟，请耐心等待...\n", None
    _reset_cancel("fine")
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        ok, msg, path = maa.fine_generate_and_save(
            subject, topic, model_choice,
            out_folder=Path(maa.OUT_RESEARCH_FOLDER) / "精细化内容",
            progress_cb=progress.append,
            should_stop=lambda: _should_stop("fine"))
        result["ok"], result["msg"], result["path"] = ok, msg, path
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = ""
    seen = 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield acc, None
        time.sleep(1.0)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    if result.get("ok"):
        extra = ""
        try:
            usage = maa._load_usage()
            extra = f"\n\n📊 累计：API 解题 {usage.get('api_problems_solved',0)} 次，预估费用 ¥{usage.get('estimated_cost_cny',0):.4f}"
        except Exception:
            extra = ""
        yield acc + "\n" + result.get("msg", "") + extra, result.get("path")
    else:
        yield acc + "\n" + result.get("msg", "生成失败"), None


# ==================== 功能7：题目档案 · 命题人变式 ====================
def _run_archive(q, model):
    if not q or len(q.strip()) < 8:
        return "请先输入完整题目（建议粘贴完整题面，含参数与多小问）。"
    text, label, err = maa.generate_question_archive(q, model)
    if text:
        return text + f"\n\n—— 分析模型：{label}"
    return f"⚠️ {err or '未生成'}"


def _run_variant(q, model):
    if not q or len(q.strip()) < 8:
        return "请先输入完整题目（建议粘贴完整题面，含参数与多小问）。"
    text, label, err = maa.generate_proposer_variants(q, model)
    if text:
        return text + f"\n\n—— 分析模型：{label}"
    return f"⚠️ {err or '未生成'}"


def _run_archive_both(q, model):
    if not q or len(q.strip()) < 8:
        return "请先输入完整题目（建议粘贴完整题面，含参数与多小问）。", None
    text, label, err = maa.archive_and_variants(q, model)
    if not text:
        return f"⚠️ {err or '未生成'}", None
    out_dir = Path(maa.OUT_RESEARCH_FOLDER) / "题目档案"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "题目档案_" + q.strip()[:12].replace("\n", "").replace(" ", "").replace("/", "_") + ".docx"
    try:
        docx_path, info = maa.convert_md_to_docx(text, output_folder=str(out_dir), out_name=name)
        return text + f"\n\n—— 分析模型：{label}\n—— 已导出 Word：{docx_path}", str(docx_path)
    except Exception as e:
        return text + f"\n\n—— 分析模型：{label}\n（Word 导出失败：{type(e).__name__}: {str(e)[:100]}）", None


# ==================== 用量看板（顶部仪表盘） ====================
_BAL_CACHE = {"ts": 0.0, "data": (False, {})}


def _balance_cached(max_age=300):
    """余额查询带缓存，避免看板轮询把接口打爆（手动刷新会强制失效）"""
    now = time.time()
    if now - _BAL_CACHE["ts"] < max_age:
        return _BAL_CACHE["data"]
    try:
        data = maa.api_balance_info()
    except Exception as e:
        data = (False, {"error": f"{type(e).__name__}: {str(e)[:60]}"})
    _BAL_CACHE.update({"ts": now, "data": data})
    return data


def _num(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "0"


def _money(v):
    try:
        return f"¥{float(v):.2f}"
    except (TypeError, ValueError):
        return "¥—"


_DASH_CSS = """
<style>
.dsh{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
     gap:10px;margin:4px 0 6px 0}
.dsh .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;
     padding:10px 13px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.dsh .k{font-size:12px;color:#6b7280;letter-spacing:.3px}
.dsh .v{font-size:20px;font-weight:650;color:#0f172a;margin-top:2px;
     font-variant-numeric:tabular-nums}
.dsh .s{font-size:11px;color:#9ca3af;margin-top:3px;line-height:1.45}
.dsh .ok .v{color:#0f766e}
.dsh .warn .v{color:#b45309}
.dsh .bar{height:5px;background:#eef2f7;border-radius:99px;overflow:hidden;margin-top:6px}
.dsh .bar span{display:block;height:100%;background:linear-gradient(90deg,#34d399,#f59e0b);border-radius:99px}
.dsh .bar span.hi{background:linear-gradient(90deg,#f59e0b,#ef4444)}
</style>
"""


def _dashboard_html():
    s = maa.usage_snapshot()
    ok, bal = _balance_cached()

    # 余额卡
    if ok:
        bal_v = _money(bal.get("total"))
        bal_s = f"充值 {_money(bal.get('topped'))} · 赠送 {_money(bal.get('granted'))}"
        try:
            remain_balance = float(bal.get("total"))
        except (TypeError, ValueError):
            remain_balance = None
    elif bal.get("local"):
        bal_v, bal_s, remain_balance = "本地模式", "不产生 API 费用", None
    else:
        bal_v, bal_s = "查询失败", (bal.get("error") or "点「刷新」重试")
        remain_balance = None

    # 今日预算卡
    limit = s["daily_limit"]
    spent = s["daily_cost"]
    if limit > 0:
        pct = max(0.0, min(100.0, spent / limit * 100))
        bar_cls = "hi" if pct >= 80 else ""
        bar = f'<div class="bar"><span class="{bar_cls}" style="width:{pct:.1f}%"></span></div>'
        budget_v = f"{_money(spent)} / {_money(limit)}"
        remain_budget = max(0.0, limit - spent)
        budget_s = f"剩余 {_money(remain_budget)} · 今日 {s['daily_problems']} 题"
        card_cls = "warn" if pct >= 80 else ""
    else:
        bar, budget_v = "", _money(spent)
        remain_budget = None
        budget_s = "未设每日上限（DAILY_BUDGET_CNY=0）"
        card_cls = ""

    # 还能跑多少题
    caps = [x for x in (remain_budget, remain_balance) if x is not None]
    remain = min(caps) if caps else None
    avg = s["avg_cost"]
    if remain is not None and avg > 0:
        can_run = f"≈ {int(remain / avg)} 题"
        can_s = f"按均价 {_money(avg)}/题 估算"
    else:
        can_run, can_s = "—", "需要云端 API 与历史用量"

    cards = [
        f'<div class="card ok"><div class="k">账户余额</div>'
        f'<div class="v">{bal_v}</div><div class="s">{bal_s}</div></div>',
        f'<div class="card {card_cls}"><div class="k">今日花费 / 预算</div>'
        f'<div class="v">{budget_v}</div>{bar}<div class="s">{budget_s}</div></div>',
        f'<div class="card"><div class="k">今日 token（输入/输出）</div>'
        f'<div class="v">{_num(s["daily_prompt_tokens"])} / {_num(s["daily_completion_tokens"])}</div>'
        f'<div class="s">累计 {_num(s["total_prompt_tokens"])} / {_num(s["total_completion_tokens"])}</div></div>',
        f'<div class="card"><div class="k">累计解题</div>'
        f'<div class="v">{_num(s["total_problems"])} 题</div>'
        f'<div class="s">累计 {_money(s["total_cost"])} · 均价 {_money(avg)}</div></div>',
        f'<div class="card"><div class="k">当前数学模型</div>'
        f'<div class="v" style="font-size:15px">{s["model"]}</div>'
        f'<div class="s">{s["provider"]} · 推理 {s["local_llm"]}（本地）</div></div>',
        f'<div class="card"><div class="k">预计还能跑</div>'
        f'<div class="v">{can_run}</div><div class="s">{can_s}</div></div>',
    ]
    return _DASH_CSS + '<div class="dsh">' + "".join(cards) + "</div>"


def refresh_dashboard():
    _BAL_CACHE["ts"] = 0.0
    return _dashboard_html()


def balance_detail():
    _BAL_CACHE["ts"] = 0.0
    try:
        ok, msg = maa.query_siliconflow_balance()
    except Exception as e:
        msg = f"⚠️ 查询出错：{type(e).__name__}: {str(e)[:150]}"
    return msg


def export_answer(answer_text, question):
    """把单题解答导出为 Word（复用项目的 MD→Word 转换）"""
    text = (answer_text or "").strip()
    if len(text) < 20:
        return None
    out_dir = OUTPUT_FOLDER / "单题解答"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "解答_" + time.strftime("%Y%m%d_%H%M%S") + ".docx"
    md = f"# 题目\n\n{(question or '').strip()}\n\n# 解答\n\n{text}\n"
    try:
        path, _info = maa.convert_md_to_docx(md, output_folder=str(out_dir), out_name=name)
        return str(path)
    except Exception:
        return None


# ==================== 解题历史（持久化到 output/解题历史.json） ====================
_HISTORY_FILE = OUTPUT_FOLDER / "解题历史.json"
_HISTORY_MAX = 100
_HIST_MAP = {}


def _load_history():
    try:
        if _HISTORY_FILE.exists():
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _append_history(question, answer, model, stats, seconds):
    items = _load_history()
    items.append({
        "time": time.strftime("%m-%d %H:%M"),
        "question": (question or "").strip(),
        "answer": answer,
        "model": model,
        "seconds": round(float(seconds or 0), 1),
        "prompt": int(stats.get("prompt") or 0),
        "completion": int(stats.get("completion") or 0),
        "cost": float(stats.get("cost") or 0),
    })
    try:
        _HISTORY_FILE.write_text(
            json.dumps(items[-_HISTORY_MAX:], ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def history_choices():
    global _HIST_MAP
    rev = list(reversed(_load_history()))
    _HIST_MAP = {str(i): it for i, it in enumerate(rev)}
    choices = []
    for i, it in enumerate(rev):
        q = (it.get("question") or "").replace("\n", " ")[:26]
        choices.append((f"{it.get('time', '')} · {q} · {it.get('seconds', 0)}s", str(i)))
    return choices, f"共 {len(rev)} 条历史记录（最多保留 {_HISTORY_MAX} 条）"


def history_load(value):
    it = _HIST_MAP.get(str(value)) if value is not None else None
    if not it:
        return gr.update(), gr.update(), "请先在下拉框里选择一条历史记录"
    stats = (f"📚 历史记录 · 用时 {it.get('seconds', 0)} 秒 · 模型 `{it.get('model', '')}` · "
             f"输入 {int(it.get('prompt') or 0):,} / 输出 {int(it.get('completion') or 0):,} token · "
             f"约 ¥{float(it.get('cost') or 0):.4f}")
    return it.get("question", ""), it.get("answer", ""), stats


def history_clear():
    try:
        _HISTORY_FILE.write_text("[]", encoding="utf-8")
    except Exception:
        pass
    return gr.update(choices=[], value=None), "已清空解题历史"


def refresh_side():
    """看板 + 历史列表一起刷新（供 15 秒定时器调用）"""
    choices, info = history_choices()
    return _dashboard_html(), gr.update(choices=choices), info


# ==================== 精细化版本对比 / 批量判分 ====================
_FINE_OUT = Path(maa.OUT_RESEARCH_FOLDER) / "精细化内容"


def fine_versions_update(subject, topic):
    vs = maa.fine_versions(subject, topic, out_folder=_FINE_OUT)
    if not vs:
        return (gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                "该专题还没有历史版本（生成一次后会出现 v1）")
    choices = [(f"v{v['v']} · {v['time']} · {v['chars']}字 · {v['sections']}板块", str(v["v"]))
               for v in vs]
    latest = str(vs[-1]["v"])
    prev = str(vs[-2]["v"]) if len(vs) > 1 else latest
    return (gr.update(choices=choices, value=latest),
            gr.update(choices=choices, value=prev),
            f"共 {len(vs)} 个版本（A/B 任选两个对比，下载后逐段择优）")


def fine_compare_ui(subject, topic, a, b):
    if not a or not b:
        return "请先选择两个版本。", None, None
    text, pa, pb = maa.fine_compare(subject, topic, a, b, out_folder=_FINE_OUT)
    return text, pa, pb


def run_grade(paper, student_file, model_choice):
    if isinstance(paper, (list, tuple)):
        paper = paper[0] if paper else ""
    if not paper:
        yield "请先在「批量做题」里选一份已做过的试卷。", None, None
        return
    if student_file is None:
        yield "请上传学生答案文件（按「1. … 2. …」编号）。", None, None
        return
    yield "⏳ 正在判分（每题一次模型调用），请稍候……", None, None
    _reset_cancel("grade")
    try:
        text = maa.read_batch_file(Path(student_file.name))
    except Exception as e:
        yield f"❌ 学生答案读取失败：{e}", None, None
        return

    progress, done, result = [], threading.Event(), {}

    def worker():
        try:
            md, rows, total, total_max, err = maa.grade_student_file(
                paper, text, model_choice=model_choice, progress_cb=progress.append,
                should_stop=lambda: _should_stop("grade"))
            result.update({"md": md, "rows": rows, "total": total,
                           "max": total_max, "err": err})
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc, seen = "", 0
    while not done.is_set():
        while seen < len(progress):
            acc += progress[seen] + "\n"
            seen += 1
        yield (acc or "⏳ 判分中……"), None, None
        time.sleep(1.0)
    while seen < len(progress):
        acc += progress[seen] + "\n"
        seen += 1
    if result.get("err"):
        yield f"❌ {result['err']}", None, None
        return

    md, rows = result["md"], result["rows"]
    path = None
    try:
        path, _info = maa.convert_md_to_docx(
            md, output_folder=str(OUTPUT_FOLDER), out_name=f"{Path(paper).stem}_判分报告.docx")
    except Exception:
        pass
    yield (f"✅ 判分完成：**{result['total']:g} / {result['max']:g}**\n\n{acc}",
           (rows or None), (str(path) if path else None))


def refresh_profile():
    rows, wrong, summary = maa.learning_profile()
    return summary, (rows or None), (wrong or None)


def export_wrongbook_ui():
    try:
        path = maa.export_wrongbook()
    except Exception as e:
        return None, f"❌ 导出失败：{type(e).__name__}: {e}"
    if not path:
        return None, "暂无错题可导出（需先判分，且存在得分率低于 60% 的题）。"
    return str(path), f"✅ 已导出错题本：{Path(path).name}"


# ==================== 网页界面 ====================
with gr.Blocks(title="高考数学一体化Agent · 网页版") as demo:
    gr.Markdown(f"""# 🧮 高考数学一体化 Agent
{_ENGINE_DESC}""")

    # ---------- 顶部：用量看板 ----------
    dashboard = gr.HTML(_dashboard_html())
    with gr.Row():
        btn_refresh = gr.Button("🔄 刷新用量", scale=0, min_width=120)
        btn_balance = gr.Button("📄 余额明细", scale=0, min_width=140)
    with gr.Accordion("用量明细（API 余额 / 累计 token / 官网对账入口）", open=False):
        usage_detail = gr.Textbox(label="", lines=9, interactive=False, buttons=["copy"])
    btn_refresh.click(refresh_dashboard, outputs=[dashboard])
    btn_balance.click(balance_detail, outputs=[usage_detail])

    with gr.Tabs():
        # ========== ① 单题解题 ==========
        with gr.Tab("① 单题解题"):
            gr.Markdown("输入一道题 → 标准答案 + 逐步推导；支持追问、复制、导出 Word、查看历史。")
            q_input = gr.Textbox(
                label="题目（可含多小问；追问直接写“那它的极值呢”）",
                lines=4, buttons=["copy"],
                placeholder="例如：求函数 y = x^2 - 4x + 3 的单调区间和极值")
            with gr.Row():
                model_choice = gr.Dropdown(
                    label="模型", scale=3, value="auto",
                    choices=[("自动（按题型切换）", "auto"),
                             (_REASON_LABEL, "reasoning"),
                             (_MATH_LABEL, "math")])
                solve_tier = gr.Dropdown(label="档位（仅数学模型）", scale=1, value="", choices=_MATH_TIERS)
                btn_solve = gr.Button("开始解答", variant="primary", scale=1)
                btn_clear = gr.Button("清空记忆", scale=1)
            run_stats = gr.Markdown("")
            out_text = gr.Textbox(label="解答结果", lines=18, interactive=False, buttons=["copy"])
            with gr.Row():
                btn_export = gr.Button("📄 导出 Word", scale=1, min_width=120)
                export_file = gr.File(label="下载", scale=3)
            out_plot = gr.Image(label="函数图像（题目需要绘图时自动出现）", type="filepath")
            with gr.Accordion("📚 解题历史（最近 100 条，选中可载入重看）", open=False):
                hist_dd = gr.Dropdown(label="历史记录", choices=history_choices()[0],
                                      value=None, allow_custom_value=True)
                with gr.Row():
                    btn_hist_load = gr.Button("载入选中", scale=1)
                    btn_hist_refresh = gr.Button("刷新列表", scale=1)
                    btn_hist_clear = gr.Button("清空历史", scale=1)
                hist_info = gr.Markdown("")

            btn_solve.click(solve_one, inputs=[q_input, model_choice, solve_tier],
                            outputs=[out_text, out_plot, run_stats],
                            concurrency_id="chat", concurrency_limit=1)
            btn_clear.click(clear_memory, outputs=[out_text])
            btn_export.click(export_answer, inputs=[out_text, q_input], outputs=[export_file])
            btn_hist_load.click(history_load, inputs=[hist_dd],
                                outputs=[q_input, out_text, run_stats])
            btn_hist_refresh.click(history_choices, outputs=[hist_dd, hist_info])
            btn_hist_clear.click(history_clear, outputs=[hist_dd, hist_info])

        # ========== ② 题目档案 ==========
        with gr.Tab("② 题目档案"):
            gr.Markdown("把一道题升级成题型：**11 字段档案**（考点/教材原型/两种解法/易错点/变式方向）"
                        "+ **命题人视角 4 个变式**，可导出 Word。")
            archive_q = gr.Textbox(
                label="题目（建议粘贴完整题面，含参数与多小问）", lines=6, buttons=["copy"],
                placeholder="例如：已知函数 f(x)=x³-3x²+ax，讨论其单调性与极值。")
            with gr.Row():
                archive_model = gr.Dropdown(
                    label="分析模型", scale=3, value="math",
                    choices=[(_MATH_LABEL, "math"), (_REASON_LABEL, "reasoning"), ("自动", "auto")])
                btn_archive = gr.Button("① 题目档案", variant="primary", scale=1)
                btn_variant = gr.Button("② 命题人变式", variant="primary", scale=1)
                btn_archive_both = gr.Button("③ 档案+变式（推荐）", variant="secondary", scale=1)
            archive_out = gr.Textbox(label="分析结果", lines=22, interactive=False, buttons=["copy"])
            archive_dl = gr.File(label="下载 Word 文档（仅「档案+变式」生成）")
            btn_archive.click(_run_archive, inputs=[archive_q, archive_model], outputs=[archive_out],
                              concurrency_id="archive", concurrency_limit=1)
            btn_variant.click(_run_variant, inputs=[archive_q, archive_model], outputs=[archive_out],
                              concurrency_id="archive", concurrency_limit=1)
            btn_archive_both.click(_run_archive_both, inputs=[archive_q, archive_model],
                                   outputs=[archive_out, archive_dl],
                                   concurrency_id="archive", concurrency_limit=1)

        # ========== ③ 批量做题 ==========
        with gr.Tab("③ 批量做题"):
            gr.Markdown("整卷逐题批改 → 输出 Word + PDF，并给出**逐题统计**（用时/费用/token）。"
                        "余额有限时把「模型档位」切到省钱档。")
            batch_file = gr.Dropdown(
                label="选择 batch_input 中的试卷（可多选）", multiselect=True,
                allow_custom_value=True,
                choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))
            with gr.Row():
                batch_model = gr.Dropdown(
                    label="解题模型", scale=2, value="auto",
                    choices=[("自动（按题型切换）", "auto"),
                             (_REASON_LABEL, "reasoning"),
                             (_MATH_LABEL, "math")])
                batch_tier = gr.Dropdown(label="模型档位（仅数学模型）", scale=1, choices=_MATH_TIERS, value="")
                btn_refresh_b = gr.Button("刷新", scale=1, min_width=90)
                btn_batch = gr.Button("开始批量做题", variant="primary", scale=1)
                btn_stop_b = gr.Button("⏹ 停止", scale=1, min_width=90)
            batch_upload = gr.File(label="上传试卷到 batch_input（txt/md/docx/pdf）", file_count="multiple")
            batch_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
            batch_log = gr.Textbox(label="处理日志", lines=10, interactive=False)
            batch_table = gr.Dataframe(
                label="逐题统计", interactive=False, wrap=True,
                headers=["文件", "题号", "状态", "模型", "用时", "费用", "token 入/出"])
            batch_dl = gr.File(label="下载解答文档（Word + PDF）")
            with gr.Accordion("📝 判分（可选：上传学生答案，逐题给分）", open=False):
                gr.Markdown("先对上面的试卷跑一次「批量做题」（会自动保存标准答案），"
                            "再上传按「1. … 2. …」编号的学生答案，即可逐题给分并生成报告。")
                student_file = gr.File(label="学生答案文件（txt / md / docx）", file_count="single")
                with gr.Row():
                    grade_model = gr.Dropdown(
                        label="判分模型", scale=3, value="math",
                        choices=[(_MATH_LABEL, "math"), (_REASON_LABEL, "reasoning")])
                    btn_grade = gr.Button("开始判分", variant="primary", scale=1)
                    btn_stop_g = gr.Button("⏹ 停止", scale=1, min_width=90)
                grade_log = gr.Textbox(label="判分日志", lines=6, interactive=False)
                grade_table = gr.Dataframe(
                    label="逐题得分", interactive=False, wrap=True,
                    headers=["题号", "得分", "满分", "考点", "主要问题", "评语"])
                grade_dl = gr.File(label="下载判分报告（Word）")
            batch_upload.upload(upload_batch, inputs=[batch_upload], outputs=[batch_msg, batch_file])
            btn_refresh_b.click(refresh_batch, outputs=[batch_file])
            btn_batch.click(run_batch, inputs=[batch_file, batch_model, batch_tier],
                            outputs=[batch_log, batch_dl, batch_table],
                            concurrency_id="batch", concurrency_limit=1)
            btn_grade.click(run_grade, inputs=[batch_file, student_file, grade_model],
                            outputs=[grade_log, grade_table, grade_dl],
                            concurrency_id="grade", concurrency_limit=1)
            btn_stop_b.click(stop_batch, outputs=[batch_log], concurrency_id="stop")
            btn_stop_g.click(stop_grade, outputs=[grade_log], concurrency_id="stop")

        # ========== ④ 真题教研 ==========
        with gr.Tab("④ 真题教研"):
            gr.Markdown("真题 → 教研资料：标准答案 / 详细解答 / 第二种解法 / 易错点 / 命题特点 / 一题三变。"
                        "整份试卷约 1~4 小时，**每分析完一段都会落一份过程稿**到 "
                        "`output_research/教研过程/`，中途停止或中断也不丢已分析内容。")
            exam_file = gr.Dropdown(
                label="选择 raw_exam_files 中的真题（可多选）", multiselect=True,
                allow_custom_value=True,
                choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))
            with gr.Row():
                research_model = gr.Dropdown(
                    label="分析模型", scale=3, value="auto",
                    choices=[("自动（推理模型）", "auto"),
                             (_REASON_LABEL, "reasoning"),
                             (_MATH_LABEL, "math")])
                btn_refresh_e = gr.Button("刷新", scale=1, min_width=90)
                btn_research = gr.Button("开始真题教研", variant="primary", scale=1)
                btn_stop_r = gr.Button("⏹ 停止", scale=1, min_width=90)
            exam_upload = gr.File(label="上传真题到 raw_exam_files（pdf/docx）", file_count="multiple")
            exam_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
            exam_log = gr.Textbox(label="教研日志", lines=10, interactive=False)
            exam_dl = gr.File(label="下载教研资料（Word + PDF）")
            exam_upload.upload(upload_exam, inputs=[exam_upload], outputs=[exam_msg, exam_file])
            btn_refresh_e.click(refresh_exam, outputs=[exam_file])
            btn_research.click(run_research, inputs=[exam_file, research_model],
                               outputs=[exam_log, exam_dl],
                               concurrency_id="research", concurrency_limit=1)
            btn_stop_r.click(stop_research, outputs=[exam_log], concurrency_id="stop")

        # ========== ⑤ 精细化讲义 ==========
        with gr.Tab("⑤ 精细化讲义"):
            gr.Markdown("按 **12 板块模板** 生成专题精讲讲义（概念/公式/四级题型/易错点/重难点/真题映射），"
                        "自动导出 Word。学科 → 篇章 → 专题 三级联动。")
            with gr.Row():
                fine_subject = gr.Dropdown(label="学科", choices=fine_subjects(), value="数学", scale=1)
                fine_category = gr.Dropdown(label="篇章", choices=fine_categories("数学"), value=None, scale=1)
                fine_topic = gr.Dropdown(
                    label="专题",
                    choices=fine_topics("数学", list(maa.FINE_TOPICS["数学"].keys())[0]),
                    value=None, scale=2)
            with gr.Row():
                fine_model = gr.Dropdown(
                    label="生成模型", scale=3, value="math",
                    choices=[(_MATH_LABEL, "math"), (_REASON_LABEL, "reasoning"), ("自动", "auto")])
                btn_fine = gr.Button("生成讲义", variant="primary", scale=1)
                btn_stop_f = gr.Button("⏹ 停止", scale=1, min_width=90)
            fine_log = gr.Textbox(label="生成日志", lines=14, interactive=False)
            fine_dl = gr.File(label="下载 Word 讲义")
            with gr.Accordion("🕘 历史版本与对比（同专题重复生成保留 v1 / v2 / …，不覆盖旧稿）", open=False):
                gr.Markdown("每次生成都是新版本。选定两个版本点「对比」可看板块级字数变化，"
                            "两份 Word 都能下载，自己逐段择优。")
                with gr.Row():
                    ver_a = gr.Dropdown(label="版本 A", choices=[], scale=1, allow_custom_value=True)
                    ver_b = gr.Dropdown(label="版本 B", choices=[], scale=1, allow_custom_value=True)
                    btn_ver_refresh = gr.Button("刷新版本", scale=1, min_width=100)
                    btn_ver_cmp = gr.Button("对比", variant="primary", scale=1, min_width=90)
                ver_info = gr.Markdown("")
                ver_md = gr.Markdown("")
                with gr.Row():
                    ver_file_a = gr.File(label="下载版本 A")
                    ver_file_b = gr.File(label="下载版本 B")
            fine_subject.change(_fine_update_cats, inputs=[fine_subject], outputs=[fine_category])
            fine_category.change(_fine_update_topics, inputs=[fine_subject, fine_category], outputs=[fine_topic])
            fine_topic.change(fine_versions_update, inputs=[fine_subject, fine_topic],
                              outputs=[ver_a, ver_b, ver_info])
            btn_ver_refresh.click(fine_versions_update, inputs=[fine_subject, fine_topic],
                                  outputs=[ver_a, ver_b, ver_info])
            btn_ver_cmp.click(fine_compare_ui, inputs=[fine_subject, fine_topic, ver_a, ver_b],
                              outputs=[ver_md, ver_file_a, ver_file_b],
                              concurrency_id="fine", concurrency_limit=1)
            btn_fine.click(run_fine, inputs=[fine_subject, fine_category, fine_topic, fine_model],
                           outputs=[fine_log, fine_dl],
                           concurrency_id="fine", concurrency_limit=1)
            btn_stop_f.click(stop_fine, outputs=[fine_log], concurrency_id="stop")

        # ========== ⑥ 知识库 ==========
        with gr.Tab("⑥ 知识库"):
            gr.Markdown("把 `knowledge_docs/` 里的资料**编译成 Wiki 词条**（分类/要点/交叉链接/索引），"
                        "解题时自动检索；未编译则回退旧版向量库。")
            wiki_status = gr.Textbox(label="Wiki 概况", value=_wiki_status(), interactive=False, lines=2)
            kb_files = gr.Textbox(
                label="knowledge_docs 现有文件", interactive=False,
                value="、".join(_list_files(KNOWLEDGE_FOLDER)) or "（空）")
            kb_upload = gr.File(label="上传资料（txt/md/docx/pdf）", file_count="multiple")
            kb_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
            with gr.Row():
                wiki_model = gr.Dropdown(
                    label="编译模型", scale=3, value="auto",
                    choices=[("自动（快速模型）", "auto"),
                             (_REASON_LABEL, "reasoning"),
                             (_MATH_LABEL, "math")])
                btn_wiki_build = gr.Button("编译 Wiki", variant="primary", scale=1)
                btn_stop_w = gr.Button("⏹ 停止", scale=1, min_width=90)
                btn_wiki_lint = gr.Button("自检", scale=1)
                btn_wiki_heal = gr.Button("修复断链", scale=1)
            kb_log = gr.Textbox(label="日志", lines=12, interactive=False)
            with gr.Accordion("备选：重建旧版 RAG 向量库（未编译 Wiki 时回退用）", open=False):
                btn_build_legacy = gr.Button("重建旧版向量库")
            kb_upload.upload(upload_kb, inputs=[kb_upload], outputs=[kb_msg, kb_files])
            btn_wiki_build.click(run_build_wiki, inputs=[wiki_model], outputs=[kb_log, wiki_status],
                                 concurrency_id="wiki", concurrency_limit=1)
            btn_wiki_lint.click(run_wiki_lint, outputs=[kb_log])
            btn_wiki_heal.click(run_wiki_heal, outputs=[kb_log, wiki_status])
            btn_build_legacy.click(build_kb, outputs=[kb_log])
            btn_stop_w.click(stop_wiki, outputs=[kb_log], concurrency_id="stop")

        # ========== ⑦ 学情画像 ==========
        with gr.Tab("⑦ 学情画像"):
            gr.Markdown("汇总历次**判分记录**：按考点看平均得分率，定位薄弱项；错题可一键导出 Word 打印。"
                        "（数据来自 ③ 批量做题里的「判分」，判一次就会累积一次）")
            with gr.Row():
                btn_profile = gr.Button("🔄 刷新画像", variant="primary", scale=1)
                btn_wrongbook = gr.Button("📄 导出错题本", scale=1)
            profile_summary = gr.Markdown("点击「刷新画像」加载。")
            profile_table = gr.Dataframe(
                label="考点画像（按得分率升序，最薄弱的在最上面）", interactive=False, wrap=True,
                headers=["考点", "题数", "平均得分率", "评价"])
            wrong_table = gr.Dataframe(
                label="错题清单（得分率 < 60%）", interactive=False, wrap=True,
                headers=["时间", "试卷", "题号", "考点", "得分", "主要问题"])
            wrongbook_file = gr.File(label="下载错题本")
            btn_profile.click(refresh_profile,
                              outputs=[profile_summary, profile_table, wrong_table])
            btn_wrongbook.click(export_wrongbook_ui,
                                outputs=[wrongbook_file, profile_summary])

        # ========== ⑧ 格式转换 ==========
        with gr.Tab("⑧ 格式转换"):
            gr.Markdown("docx / pdf / txt → Markdown，Markdown → Word。适合把外部资料转成可入库/可教学的格式。")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**① 转 Markdown**")
                    md_upload = gr.File(label="上传 docx / pdf / txt", file_count="single")
                    btn_md_convert = gr.Button("转换为 MD", variant="primary")
                    md_preview = gr.Textbox(label="预览", lines=10, interactive=False)
                    md_dl = gr.File(label="下载 MD")
                with gr.Column():
                    gr.Markdown("**② 转 Word**")
                    docx_upload = gr.File(label="上传 .md 文件", file_count="single")
                    btn_docx_convert = gr.Button("转换为 Word", variant="primary")
                    docx_preview = gr.Textbox(label="结果", lines=10, interactive=False)
                    docx_dl = gr.File(label="下载 Word")
            btn_md_convert.click(convert_to_md_ui, inputs=[md_upload], outputs=[md_preview, md_dl])
            btn_docx_convert.click(convert_md_to_docx_ui, inputs=[docx_upload],
                                   outputs=[docx_preview, docx_dl])

    # 15 秒轮询：更新看板 + 解题历史列表（只读本地，余额走 300 秒缓存）
    _timer = gr.Timer(15)
    _timer.tick(refresh_side, outputs=[dashboard, hist_dd, hist_info])


if __name__ == "__main__":
    # 并发设为 1，避免与本地 Ollama 单实例推理冲突
    # inbrowser=False：避免和启动脚本自动打开浏览器冲突（否则会开两个网页）
    # 队列：默认并发 4，各功能用独立 concurrency_id 互不阻塞
    # （单题/批量/教研/讲义/知识库可同时跑；同一功能内部仍串行，避免抢 Ollama 与 API）
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False,
        theme=gr.themes.Soft())
