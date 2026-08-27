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
def solve_one(question):
    if not question or not question.strip():
        return "请输入题目。", None
    cfg = {"configurable": {"thread_id": "web_chat"}, "recursion_limit": 10}
    plot = OUTPUT_FOLDER / "plot_out.png"
    before = plot.stat().st_mtime if plot.exists() else None
    t0 = time.time()
    ans = maa.run_question(maa.graph_chat, question, cfg)
    dt = time.time() - t0
    after = plot.stat().st_mtime if plot.exists() else None
    img = str(plot) if (after and after != before) else None
    return f"（本题用时 {dt:.0f} 秒）\n\n{ans}", img


# ==================== 功能2：RAG 知识库 ====================
def build_kb():
    files = _list_files(KNOWLEDGE_FOLDER)
    info = f"📁 knowledge_docs 现有文件：{files if files else '（空）'}\n\n"
    log, _ = _run_with_console(maa.build_rag_vector_db)
    return info + log


def upload_kb(files):
    msg = _save_uploads(files, KNOWLEDGE_FOLDER)
    return msg + "\n上传后请点击【构建/更新知识库】使其生效。", gr.update(
        choices=_list_files(KNOWLEDGE_FOLDER))


# ==================== 功能3：批量做题 ====================
def refresh_batch():
    return gr.update(choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))


def upload_batch(files):
    msg = _save_uploads(files, BATCH_INPUT_FOLDER)
    return msg, gr.update(choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))


def run_batch(filename):
    if not filename:
        return "请先选择或上传试卷文件。", None
    buf = io.StringIO()
    with redirect_stdout(buf):
        full_text, docx_path = maa.batch_solve_file(filename)
        pdf_path = OUTPUT_FOLDER / (Path(filename).stem + "_result.pdf")
        maa.text_to_pdf(full_text, str(pdf_path))
    log = buf.getvalue()
    outs = [p for p in (docx_path, str(pdf_path)) if Path(p).exists()]
    return log, (outs or None)


# ==================== 功能4：真题教研 ====================
def refresh_exam():
    return gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def upload_exam(files):
    msg = _save_uploads(files, RAW_EXAM_FOLDER)
    return msg, gr.update(choices=_list_files(RAW_EXAM_FOLDER, (".pdf", ".docx")))


def run_research():
    buf = io.StringIO()
    with redirect_stdout(buf):
        maa.research_workflow()
    log = buf.getvalue()
    outs = [
        str(p) for p in (
            OUT_RESEARCH_FOLDER / "高考数学教研汇总.docx",
            OUT_RESEARCH_FOLDER / "高考数学教研汇总.pdf",
        ) if p.exists()
    ]
    return log, (outs or None)


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
            label="输入题目（可含多小问，如“已知 f(x)=... (1)求... (2)证明...”）",
            lines=4, placeholder="例如：求函数 y = x^2 - 4x + 3 的单调区间和极值")
        with gr.Row():
            btn_solve = gr.Button("开始解答", variant="primary")
            btn_clear = gr.Button("清空记忆")
        out_text = gr.Textbox(label="解答结果", lines=20, interactive=False)
        out_plot = gr.Image(label="函数图像（如有）", type="filepath", visible=True)
        btn_solve.click(solve_one, inputs=[q_input], outputs=[out_text, out_plot])
        btn_clear.click(
            lambda: (maa.graph_chat.update_state(
                {"configurable": {"thread_id": "web_chat"}, "recursion_limit": 10},
                {"messages": []}), "✅ 记忆已清空")[1],
            outputs=[out_text])

    # ---------- 功能2 ----------
    with gr.Tab("② RAG 知识库"):
        kb_files = gr.Textbox(
            label="knowledge_docs 当前文件",
            value="、".join(_list_files(KNOWLEDGE_FOLDER)) or "（空）",
            interactive=False)
        kb_upload = gr.File(label="上传资料到 knowledge_docs（txt/md/docx/pdf）",
                            file_count="multiple")
        kb_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
        btn_build = gr.Button("构建/更新知识库", variant="primary")
        kb_log = gr.Textbox(label="构建日志", lines=12, interactive=False)
        kb_upload.upload(upload_kb, inputs=[kb_upload],
                         outputs=[kb_msg, kb_files])
        btn_build.click(build_kb, outputs=[kb_log])

    # ---------- 功能3 ----------
    with gr.Tab("③ 批量做题"):
        batch_file = gr.Dropdown(
            label="选择 batch_input 中的试卷",
            choices=_list_files(BATCH_INPUT_FOLDER, ALLOWED_BATCH))
        with gr.Row():
            btn_refresh_b = gr.Button("刷新列表")
            btn_batch = gr.Button("开始批量做题", variant="primary")
        batch_upload = gr.File(
            label="上传试卷到 batch_input（txt/md/docx/pdf）", file_count="multiple")
        batch_msg = gr.Textbox(label="上传状态", interactive=False, lines=2)
        batch_log = gr.Textbox(label="处理日志", lines=12, interactive=False)
        batch_dl = gr.File(label="下载解答文档（Word + PDF）")
        batch_upload.upload(upload_batch, inputs=[batch_upload],
                            outputs=[batch_msg, batch_file])
        btn_refresh_b.click(refresh_batch, outputs=[batch_file])
        btn_batch.click(run_batch, inputs=[batch_file],
                        outputs=[batch_log, batch_dl])

    # ---------- 功能4 ----------
    with gr.Tab("④ 真题教研"):
        exam_file = gr.Dropdown(
            label="raw_exam_files 中的真题（可选，教研会自动处理全部文件）",
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
        btn_research.click(run_research, outputs=[exam_log, exam_dl])

if __name__ == "__main__":
    # 并发设为 1，避免与本地 Ollama 单实例推理冲突
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=True)
