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
            {"configurable": {"thread_id": _WEB_THREAD}, "recursion_limit": 10},
            {"messages": []})
        return "✅ 记忆已清空，开始全新对话"
    except Exception as e:
        return f"清空失败：{e}"

def solve_one(question, model_choice):
    if not question or not question.strip():
        yield "请输入题目。", None
        return
    model_label = {"auto": "自动（按题型）", "reasoning": "DeepSeek推理", "math": "Qwen数学"}.get(model_choice, model_choice)
    yield f"⏳ 正在解题中（模型：{model_label}，简单题约1分钟，大题约3~5分钟），请勿关闭页面，耐心等待...", None
    cfg = {"configurable": {"thread_id": _WEB_THREAD}, "recursion_limit": 10}
    plot = OUTPUT_FOLDER / "plot_out.png"
    before = plot.stat().st_mtime if plot.exists() else None
    t0 = time.time()
    ans, model_used = maa.run_question(maa.graph_chat, question, cfg, model_choice=model_choice)
    dt = time.time() - t0
    after = plot.stat().st_mtime if plot.exists() else None
    img = str(plot) if (after and after != before) else None
    yield f"（本题用时 {dt:.0f} 秒，使用模型：{model_used}）\n\n{ans}", img


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


def run_build_wiki():
    yield "⏳ Wiki 编译启动中（按资料量数分钟到数十分钟），实时进度如下：\n", None
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.build_wiki(progress_cb=progress.append)
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


def run_batch(files):
    if not files:
        yield "请先选择试卷文件。", None
        return
    yield "⏳ 批量做题启动中（每题约需数分钟），下方实时显示进度...\n", None
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                for fname in files:
                    progress.append(f"=== 开始处理文件：{fname} ===")
                    full_text, docx_path = maa.batch_solve_file(
                        fname, progress_cb=progress.append)
                    pdf_path = OUTPUT_FOLDER / (Path(fname).stem + "_result.pdf")
                    maa.text_to_pdf(full_text, str(pdf_path))
                    progress.append(f"=== {fname} 完成，已生成 Word + PDF ===")
        except Exception as e:
            progress.append(f"[错误] {e}")
        finally:
            result["log"] = buf.getvalue()
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    acc = "⏳ 批量做题进行中，实时进度如下：\n"
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
    outs = []
    for fname in files:
        for p in (OUTPUT_FOLDER / (Path(fname).stem + "_result.docx"),
                  OUTPUT_FOLDER / (Path(fname).stem + "_result.pdf")):
            if Path(p).exists():
                outs.append(str(p))
    yield f"✅ 批量做题完成！完整日志：\n{log}", (outs or None)


# ==================== 功能4：真题教研 ====================
def refresh_exam():
    return gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def upload_exam(files):
    msg = _save_uploads(files, RAW_EXAM_FOLDER)
    return msg, gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def run_research(files):
    if not files:
        yield "请先选择要教研的真题文件。", None
        return
    yield "⏳ 教研启动中（每题约需数分钟，全程可能 1~4 小时），下方会实时显示进度...\n", None
    progress = []
    done = threading.Event()
    result = {}

    def worker():
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                maa.research_workflow(only_files=list(files),
                                      progress_cb=progress.append)
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


# ==================== 网页界面 ====================
with gr.Blocks(title="高考数学一体化Agent · 网页版",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🧮 高考数学一体化Agent · 网页操作版
本地 Ollama（deepseek-r1:14b + qwen2.5:7b）驱动。注意：大题推理较慢（约 1~5 分钟/题），
提交后请耐心等待页面返回；页面刷新后显示结果即为完成。
""")

    # ---------- 功能1 ----------
    with gr.Tab("① 单题交互"):
        q_input = gr.Textbox(
            label="输入题目（可含多小问，如“已知 f(x)=... (1)求... (2)证明...”），支持追问，如“那它的极值呢”",
            lines=4, placeholder="例如：求函数 y = x^2 - 4x + 3 的单调区间和极值")
        with gr.Row():
            model_choice = gr.Dropdown(
                label="解题模型",
                choices=[("自动（按题型切换）", "auto"), ("DeepSeek-R1 推理", "reasoning"), ("Qwen2.5-Math 数学", "math")],
                value="auto",
                scale=2)
            btn_solve = gr.Button("开始解答", variant="primary", scale=1)
            btn_clear = gr.Button("清空记忆", scale=1)
        out_text = gr.Textbox(label="解答结果", lines=20, interactive=False)
        out_plot = gr.Image(label="函数图像（如有）", type="filepath", visible=True)
        btn_solve.click(solve_one, inputs=[q_input, model_choice], outputs=[out_text, out_plot])
        btn_clear.click(clear_memory, outputs=[out_text])

    # ---------- 功能2 ----------
    with gr.Tab("② LLM Wiki 知识库"):
        gr.Markdown("""
**Karpathy LLM Wiki 模式**：`knowledge_docs/` 是原始资料层（只进不改）。
点击「编译 Wiki 知识库」后，LLM 会把资料**编译成结构化的 Markdown 词条**（分类 / 要点 / 交叉链接 / 总索引），
解题时直接从编译好的 Wiki 检索（更快、知识可积累）；未编译时自动回退旧 RAG。
""")
        kb_files = gr.Textbox(
            label="knowledge_docs 当前文件（原始资料层）",
            value="、".join(_list_files(KNOWLEDGE_FOLDER)) or "（空）",
            interactive=False)
        kb_upload = gr.File(label="上传资料到 knowledge_docs（txt/md/docx/pdf）",
                            file_count="multiple")
        kb_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
        with gr.Row():
            btn_wiki_build = gr.Button("编译 Wiki 知识库", variant="primary")
            btn_wiki_lint = gr.Button("Wiki 自检")
            btn_wiki_heal = gr.Button("Wiki 修复（自动修断裂链接）")
        wiki_status = gr.Textbox(
            label="Wiki 概况", value=_wiki_status(), interactive=False, lines=2)
        kb_log = gr.Textbox(label="Wiki 日志", lines=12, interactive=False)
        btn_build_legacy = gr.Button("重建旧版 RAG 向量库（备选，未编译 Wiki 时回退用）")
        kb_upload.upload(upload_kb, inputs=[kb_upload],
                         outputs=[kb_msg, kb_files])
        btn_wiki_build.click(run_build_wiki, outputs=[kb_log, wiki_status])
        btn_wiki_lint.click(run_wiki_lint, outputs=[kb_log])
        btn_wiki_heal.click(run_wiki_heal, outputs=[kb_log, wiki_status])
        btn_build_legacy.click(build_kb, outputs=[kb_log])

    # ---------- 功能3 ----------
    with gr.Tab("③ 批量做题"):
        batch_file = gr.Dropdown(
            label="选择 batch_input 中的试卷（可多选，一次处理多份）",
            multiselect=True,
            choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))
        with gr.Row():
            btn_refresh_b = gr.Button("刷新列表")
            btn_batch = gr.Button("开始批量做题", variant="primary")
        batch_upload = gr.File(
            label="上传试卷到 batch_input（txt/md/docx/pdf）", file_count="multiple")
        batch_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
        batch_log = gr.Textbox(label="处理日志", lines=12, interactive=False)
        batch_dl = gr.File(label="下载解答文档（Word + PDF，可多份）")
        batch_upload.upload(upload_batch, inputs=[batch_upload],
                            outputs=[batch_msg, batch_file])
        btn_refresh_b.click(refresh_batch, outputs=[batch_file])
        btn_batch.click(run_batch, inputs=[batch_file],
                        outputs=[batch_log, batch_dl])

    # ---------- 功能4 ----------
    with gr.Tab("④ 真题教研"):
        exam_file = gr.Dropdown(
            label="选择 raw_exam_files 中的真题（可多选，只教研选中的文件）",
            multiselect=True,
            choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))
        with gr.Row():
            btn_refresh_e = gr.Button("刷新列表")
            btn_research = gr.Button("开始真题教研", variant="primary")
        exam_upload = gr.File(
            label="上传真题到 raw_exam_files（pdf/docx）", file_count="multiple")
        exam_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
        exam_log = gr.Textbox(label="教研日志", lines=12, interactive=False)
        exam_dl = gr.File(label="下载教研资料（Word + PDF）")
        exam_upload.upload(upload_exam, inputs=[exam_upload],
                           outputs=[exam_msg, exam_file])
        btn_refresh_e.click(refresh_exam, outputs=[exam_file])
        btn_research.click(run_research, inputs=[exam_file],
                           outputs=[exam_log, exam_dl])

if __name__ == "__main__":
    # 并发设为 1，避免与本地 Ollama 单实例推理冲突
    # inbrowser=False：避免和启动脚本自动打开浏览器冲突（否则会开两个网页）
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False)
