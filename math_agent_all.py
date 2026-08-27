from dotenv import load_dotenv
import os
import re
import json
import time
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import sympy
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import pdfplumber
from docx import Document

plt.switch_backend('Agg')
load_dotenv()

# ========================全局路径配置========================
CHROMA_DB_PATH = "./db_chroma"
KNOWLEDGE_FOLDER = "./knowledge_docs"
BATCH_INPUT_FOLDER = "./batch_input"
OUTPUT_FOLDER = "./output"
RAW_EXAM_FOLDER = "./raw_exam_files"
OUT_RESEARCH_FOLDER = "./output_research"

for p in [CHROMA_DB_PATH, KNOWLEDGE_FOLDER, BATCH_INPUT_FOLDER, OUTPUT_FOLDER, RAW_EXAM_FOLDER, OUT_RESEARCH_FOLDER]:
    os.makedirs(p, exist_ok=True)

llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=os.getenv("LLM_MODEL"),
    temperature=0.2
)

# 快速模型：用于拆题等轻量环节，加快功能1速度（可用 .env 的 FAST_MODEL 覆盖）
llm_fast = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=os.getenv("FAST_MODEL", "qwen2.5:7b"),
    temperature=0.1
)

CATEGORY_LIST = ["集合", "函数", "导数", "三角", "数列", "立体几何", "圆锥曲线", "概率统计"]

# ========================工具定义========================
@tool
def sympy_math_calc(code: str) -> str:
    """符号数学计算：求导、解方程、不等式、数列求和"""
    try:
        namespace = {"sympy": sympy}
        res = eval(code, namespace)
        return f"计算结果：{str(res)} \nLaTeX公式：{sympy.latex(res)}"
    except Exception as e:
        return f"计算工具错误：{str(e)}"

@tool
def plot_function(expr_str: str, x_range: tuple = (-10, 10)) -> str:
    """绘制函数图像，保存图片到output文件夹"""
    x = sympy.Symbol('x')
    expr = sympy.parse_expr(expr_str)
    f_lambda = sympy.lambdify(x, expr, "matplotlib")
    import numpy as np
    xs = np.linspace(x_range[0], x_range[1], 500)
    ys = f_lambda(xs)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.grid(True)
    plt.title(f"y = {expr_str}")
    img_path = os.path.join(OUTPUT_FOLDER, "plot_out.png")
    plt.savefig(img_path, dpi=120)
    plt.close()
    return f"图像已保存 {img_path}"

tools = [sympy_math_calc, plot_function]
llm_with_tools = llm.bind_tools(tools)

# ========================B RAG知识库模块【升级支持pdf、docx】========================
def read_knowledge_file(filepath: Path) -> str:
    """读取知识库文件：txt md docx pdf"""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        text_buf = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                pt = page.extract_text()
                if pt:
                    text_buf.append(pt)
        return "\n".join(text_buf)
    elif suffix == ".docx":
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    elif suffix in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def build_rag_vector_db():
    all_text = []
    support_exts = (".txt", ".md", ".docx", ".pdf")
    for fname in os.listdir(KNOWLEDGE_FOLDER):
        fp = Path(KNOWLEDGE_FOLDER) / fname
        if fp.suffix.lower() in support_exts:
            print(f"正在读取知识库文件：{fname}")
            content = read_knowledge_file(fp)
            if content.strip():
                all_text.append(content)
    if not all_text:
        print("knowledge_docs文件夹没有有效文档，跳过构建向量库")
        return None
    full_doc = "\n".join(all_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(full_doc)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
    print("✅RAG向量知识库构建完成，支持格式：txt/md/docx/pdf")
    return db

def load_rag_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        return db
    return None

def rag_retrieve(query: str):
    db = load_rag_db()
    if db is None:
        return "【RAG】无知识库文档"
    docs = db.similarity_search(query, k=3)
    context = "\n====参考知识点片段====\n"
    for d in docs:
        context += d.page_content + "\n"
    return context

# ========================D 大题自动拆解节点========================
def _looks_like_big_problem(user_q: str) -> bool:
    """粗略判断是否为需要拆解的大题：题内含小问标记（(1)/(1)、1. 等）视为大题"""
    return bool(re.search(r"[（(]\s*\d+\s*[)）]|\d+\s*[.、．]", user_q))

def decompose_problem(state: MessagesState):
    user_q = state["messages"][-1].content
    # 简单题直接跳过拆解（省一次大模型调用，加快速度）
    if not _looks_like_big_problem(user_q):
        return {"messages":[{"role":"system","content":"【题目拆解】无需拆解"}]}
    decompose_prompt = f"""
请把下面这道高考数学大题拆解成若干个子小问题，按顺序输出子问题列表。
如果是简单小题不需要拆解，直接返回【无需拆解】。
题目：{user_q}
输出格式：
是否拆解：是/否
子问题列表：
1. xxx
2. xxx
"""
    resp = llm_fast.invoke([{"role":"user","content":decompose_prompt}])
    return {"messages":[{"role":"system","content":"【题目拆解】"+resp.content}]}

def solve_agent(state: MessagesState):
    rag_ctx = rag_retrieve(state["messages"][-1].content)
    system_prompt = f"""
你是高考数学解题专家，请输出一份「完整、规范、精炼」的高考标准答案（类似官方评分标准答案）。

参考知识库信息（仅作知识点参考）：
{rag_ctx}

硬性规则：
1. 计算题、求导、解方程、求取值范围等应调用 sympy_math_calc 工具得到准确结果，禁止徒手心算；若调用工具失败，则直接给出你计算出的结果，不要反复重试工具。
2. 用户需要图像时调用 plot_function 绘图。
3. 禁止把 python 代码写在回答文本中，代码只能出现在工具调用里。
4. 【重要】禁止输出 Markdown 和 LaTeX 语法（不要出现 \\[、\\]、\\(、\\)、\\frac、\\dfrac、\\boxed、\\sqrt、**、#、\\ln 等）。
   数学式一律用普通文本 + Unicode 表示，例如：
   - 导数写为：f'(x) = ln x + 2 - 2a x
   - 分数写为：(a)/(b) 或 a/b
   - 幂/下标写为：x^2、x1、x2、e/2
   - 取值范围写为：0 < a < e/2
5. 输出正文就是标准答案本身，不要出现“解题思路”“考点总结”“教学点评”等多余板块，不要自我重复。
6. 【确定正确】禁止出现“似乎”“可能”“也许”“需要进一步验证”“这没有直接帮助”等探索性或不确定表述；每一步都要给出确定、正确的结论，若有疑问就独立重新推导出确定结果。
7. 【详略得当】步骤要完整、详细：不跳步，写清楚每一步的依据（如“由求导法则”“由对数性质”“令导数等于零”等）和中间结果；简单题也必须写完整推导，禁止只写结论；但不要写与解题无关的解释性文字。

输出结构（按题目小问顺序）：
每小问：【解】以编号步骤逐条完整推导——写成“步骤1：… 步骤2：…”，每一步写明依据、公式变换和中间结果，逐步推进到结论；步骤详略适中（一般 3～8 步）。
最后：【答案】汇总各小问的最终结论，简洁明确。
参考拆解出的子问题，分小问依次完整作答。
"""
    messages = [{"role":"system","content":system_prompt}] + state["messages"]
    ai_msg = llm_with_tools.invoke(messages)
    return {"messages":[ai_msg]}

def checker_agent(state: MessagesState):
    # 提取用户题目与解题 Agent 的初步解答（取最后一个无工具调用的 AI 回答）
    user_q = ""
    last_solve = ""
    for m in state["messages"]:
        if getattr(m, "type", "") == "human":
            user_q = m.content
        elif getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None):
            last_solve = m.content
    check_prompt = f"""
你是数学阅卷老师。下面是题目和初步解答，请独立重新演算，检查计算错误和逻辑漏洞。

【题目】
{user_q}

【初步解答】
{last_solve}

规则：
- 只输出最终的标准答案正文（复核修正后的完整标准答案），你是最终标准答案的权威输出者。
- 若初步解答有误，输出修正后完整、正确的标准答案；若正确，直接输出同一份标准答案。
- 禁止输出任何“复核”“校验”“检查过程”“评语”，禁止重复上面的题目和初步解答原文。
- 禁止 Markdown 和 LaTeX 语法（不要出现 \\[、\\]、\\(、\\)、\\frac、\\boxed、**、# 等），数学式一律用普通文本 + Unicode 表示。
- 按题目小问顺序给出【解】和完整的逐步推导：用“步骤1：… 步骤2：…”逐条编号，每一步写明依据、公式变换和中间结果，不跳步；简单题也必须写完整推导，禁止只写结论。
- 每条步骤必须确定、正确：禁止出现“似乎”“可能”“需要进一步验证”等探索性或不确定表述；若有疑问，独立重新计算得出确定结论，不要写出探索过程。
- 步骤详略适中（一般 3～8 步），不写与解题无关的文字。最后给出【答案】汇总各小问结论。
"""
    resp = llm.invoke([{"role": "user", "content": check_prompt}])
    return {"messages": [resp]}

def clean_model_answer(text: str) -> str:
    """清理模型输出中的 DeepSeek 特殊标记与 Markdown/LaTeX 残留，避免 Word 文档出现乱码和重复"""    # 1) 去掉 deepseek-r1 的 response / thinking 特殊标记；
    #    若内容中存在多个 response 标记（模型复读导致答案重复），只保留最后一段
    parts = re.split(r'\s*<\|?/?response\|?>?\s*', text, flags=re.IGNORECASE)
    text = parts[-1] if len(parts) > 1 else text
    text = re.sub(r'^\s*<\|?/?think(ing)?\|?>?\s*', '', text, flags=re.IGNORECASE)
    # 2) 去掉 Markdown 标题符号（行首 # / ## 等）和强调符号 **
    text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
    text = text.replace('**', '')
    # 3) 去掉 LaTeX 公式定界符残留
    text = text.replace('\\[', '').replace('\\]', '')
    text = text.replace('\\(', '').replace('\\)', '')
    # 4) LaTeX 数学符号 → 纯文本（供 Word 正常显示）
    text = latex_to_plain(text)
    # 5) 清理每行首尾空白
    lines = [ln.rstrip() for ln in text.split('\n')]
    return '\n'.join(lines).strip()


def run_question(graph, question: str, config, max_tries: int = 2) -> str:
    """统一答题入口：调用图解题并清洗输出；若模型只输出推理、正文为空，自动重试一次"""
    ans = ""
    for attempt in range(max_tries):
        res = graph.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
        ans = clean_model_answer(res["messages"][-1].content)
        if len(ans) >= 20:
            return ans
        print(f"⚠️ 本次未生成有效答案（模型可能只输出了推理过程），第{attempt + 1}次重试中...")
    return ans


def _find_matching_brace(s: str, start: int) -> int:
    """s[start] == '{'，返回与之匹配的 '}' 下标（用于 LaTeX 花括号配对）"""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


# LaTeX 简单命令 → Unicode/纯文本 映射
_LATEX_SYMBOL_MAP = {
    'ln': 'ln', 'log': 'log', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'ge': '≥', 'le': '≤', 'neq': '≠', 'to': '→', 'rightarrow': '→',
    'in': '∈', 'pi': 'π', 'pm': '±', 'times': '×', 'div': '÷',
    'mid': '|', 'cdot': '*', 'implies': '=>', 'infty': '∞',
    'text': '', 'mathrm': '', 'operatorname': '', 'quad': '  ', 'qquad': '  ',
    'left': '', 'right': '', 'big': '', 'Big': '', 'bigg': '', 'Bigg': '',
}


def latex_to_plain(s: str) -> str:
    """把常用 LaTeX 数学式转成可读纯文本（供 Word 无乱码显示）"""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            cmd = s[i + 1:j]
            low = cmd.lower()
            # \frac{a}{b} / \dfrac{a}{b} / \tfrac{a}{b}
            if low in ('frac', 'dfrac', 'tfrac', 'cfrac'):
                num = den = ''
                k = j
                while k < n and s[k] == ' ': k += 1
                if k < n and s[k] == '{':
                    e = _find_matching_brace(s, k)
                    num = latex_to_plain(s[k + 1:e])
                    k = e + 1
                while k < n and s[k] == ' ': k += 1
                if k < n and s[k] == '{':
                    e = _find_matching_brace(s, k)
                    den = latex_to_plain(s[k + 1:e])
                    k = e + 1
                out.append(f'({num})/({den})')
                i = k
                continue
            # \sqrt{...}
            if low == 'sqrt':
                k = j
                while k < n and s[k] == ' ': k += 1
                if k < n and s[k] == '{':
                    e = _find_matching_brace(s, k)
                    out.append('sqrt(' + latex_to_plain(s[k + 1:e]) + ')')
                    i = e + 1
                else:
                    out.append('sqrt')
                    i = k
                continue
            # \boxed{...} 保留内部内容
            if low == 'boxed':
                k = j
                while k < n and s[k] == ' ': k += 1
                if k < n and s[k] == '{':
                    e = _find_matching_brace(s, k)
                    out.append(latex_to_plain(s[k + 1:e]))
                    i = e + 1
                    continue
            # 其它已知命令
            if low in _LATEX_SYMBOL_MAP:
                out.append(_LATEX_SYMBOL_MAP[low])
                i = j
                continue
            # 未知命令：丢弃
            i = j
            continue
        elif c == '{' or c == '}':
            i += 1
            continue
        elif c == '^':
            # 上标：x^{2} -> x^2；e^{-1} -> e^(-1)
            k = i + 1
            while k < n and s[k] == ' ': k += 1
            if k < n and s[k] == '{':
                e = _find_matching_brace(s, k)
                sup = latex_to_plain(s[k + 1:e])
                i = e + 1
            else:
                sup = s[k] if k < n else ''
                i = k + 1
            if any(ch in sup for ch in '+-*/()'):
                out.append('^(' + sup + ')')
            else:
                out.append('^' + sup)
            continue
        elif c == '_':
            # 下标：x_1 -> x1
            k = i + 1
            while k < n and s[k] == ' ': k += 1
            if k < n and s[k] == '{':
                e = _find_matching_brace(s, k)
                out.append(latex_to_plain(s[k + 1:e]))
                i = e + 1
            elif k < n:
                out.append(s[k])
                i = k + 1
            else:
                i += 1
            continue
        else:
            out.append(c)
            i += 1
    return ''.join(out)

tool_node = ToolNode(tools)

def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "checker"

builder = StateGraph(MessagesState)
builder.add_node("decompose", decompose_problem)
builder.add_node("solve_agent", solve_agent)
builder.add_node("tools", tool_node)
builder.add_node("checker_agent", checker_agent)
builder.set_entry_point("decompose")
builder.add_edge("decompose", "solve_agent")
builder.add_conditional_edges("solve_agent", should_continue, {"tools":"tools", "checker":"checker_agent"})
builder.add_edge("tools", "solve_agent")
builder.add_edge("checker_agent", "__end__")

# langgraph-checkpoint-sqlite 3.x 中 from_conn_string 已改为上下文管理器，
# 这里直接构造 SqliteSaver(conn)，连接在进程生命周期内保持，实现跨会话记忆持久化。
conn = sqlite3.connect("./all_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph_chat = builder.compile(checkpointer=memory)

# ======================== 批量做题模块｜支持PDF输入，输出docx，不再输出md ========================
def read_batch_file(filepath:Path):
    """读取batch_input支持：txt md docx pdf"""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        text_all = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                pt = page.extract_text()
                if pt:
                    text_all += pt + "\n"
        return text_all
    elif suffix == ".docx":
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    elif suffix in (".txt",".md"):
        with open(filepath,"r",encoding="utf-8") as f:
            return f.read()
    else:
        raise Exception(f"不支持文件后缀 {suffix}")

def batch_solve_file(filename: str):
    file_path = Path(BATCH_INPUT_FOLDER)/filename
    content = read_batch_file(file_path)
    # 按“行首编号”切分多道习题：支持 1. / 1、/ 1． 三种编号；
    # 只匹配行首的数字编号，避免把题内小问 (1)(2) 或文本中的小数误拆。
    pattern = re.compile(r"(?m)^\s*\d+\s*[.、．]\s*")
    parts = pattern.split(content)
    q_list = [p.strip() for p in parts if len(p.strip())>3]
    all_out_items = []
    thread_id = "batch_session"
    config = {"configurable":{"thread_id":thread_id}, "recursion_limit": 10}
    graph_chat.update_state(config, {"messages":[]})

    for idx, q in enumerate(q_list):
        print(f"\n====批量处理第{idx+1}题====\n题目：{q[:80]}")
        ans_text = run_question(graph_chat, q, config)
        all_out_items.append({
            "no": idx+1,
            "question": q,
            "answer": ans_text
        })

    # 直接生成docx，不再生成md
    doc = Document()
    doc.add_heading(f"试卷解答：{filename}", level=1)
    for item in all_out_items:
        doc.add_heading(f"第{item['no']}题", level=2)
        doc.add_paragraph(f"【题目】{item['question']}")
        doc.add_paragraph("【解答】")
        for line in item["answer"].split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())
        doc.add_paragraph("-"*60)

    docx_save = Path(OUTPUT_FOLDER) / f"{Path(filename).stem}_result.docx"
    doc.save(str(docx_save))
    print(f"✅批量解答保存docx：{docx_save.resolve()}")

    # 返回纯文本，用于后续生成pdf
    full_text_buf = []
    for it in all_out_items:
        full_text_buf.append(f"## 第{it['no']}题\n【题目】{it['question']}\n【解答】\n{it['answer']}\n\n")
    full_text = "\n".join(full_text_buf)
    return full_text, str(docx_save)

def text_to_pdf(text:str, pdf_save_path:str):
    c = canvas.Canvas(pdf_save_path, pagesize=A4)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c.setFont("STSong-Light",11)
    width, height = A4
    y_pos = height-40
    lines = text.split("\n")
    for line in lines:
        if y_pos < 40:
            c.showPage()
            c.setFont("STSong-Light",11)
            y_pos = height-40
        c.drawString(30, y_pos, line[:120])
        y_pos -=18
    c.save()
    print(f"✅PDF输出：{pdf_save_path}")

# ========================教研模块：PDF/Word真题自动教研========================
def read_pdf(filepath: Path) -> str:
    text_all = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_all += page_text + "\n"
    return text_all

def read_docx(filepath: Path) -> str:
    doc = Document(filepath)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def load_raw_exam_files() -> list[tuple[str,str]]:
    res = []
    for f in Path(RAW_EXAM_FOLDER).glob("*"):
        suffix = f.suffix.lower()
        if suffix == ".pdf":
            content = read_pdf(f)
            res.append((f.name, content))
        elif suffix == ".docx":
            content = read_docx(f)
            res.append((f.name, content))
    return res

EXTRACT_PROMPT = """
下面是高考数学试卷文本，请把里面所有数学题目完整提取出来。
注意：同一大题的多个小问（如 (1)(2)(3)）必须合并为同一条记录，题目的 question 字段要完整包含该大题题干及其全部小问。
输出格式JSON数组，每一条：
{{
  "question": "完整原题文本（含该大题全部小问）",
  "category": "从【集合、函数、导数、三角、数列、立体几何、圆锥曲线、概率统计】选择最合适的一个分类"
}}
只输出JSON，不要多余解释。
试卷文本：
{text_chunk}
"""

RESEARCH_PROMPT = """
你是高中数学教研专家。给定一道高考真题，请完整输出下面全部模块，不要省略。
【原题】
{q_text}

输出严格包含下面全部板块：
1.【标准答案】
2.【详细解答】分步完整推导
3.【解题方法】本题用到的核心解题思路与方法
4.【第二种解法】提供另外一套不同思路的完整解法，如果没有写“本题无第二种可行解法”
5.【易错点】学生高频踩坑、容易错在哪里
6.【命题特点】高考命题角度、考察知识点、能力要求
7.【一题三变】给出3道变式题，改变条件，同源考点，用于训练。
"""

def research_workflow():
    print(f"\n====教研模块，读取文件夹 {RAW_EXAM_FOLDER}====")
    file_list = load_raw_exam_files()
    if len(file_list)==0:
        print("请把PDF/docx真题放入raw_exam_files")
        return
    print(f"读取到 {len(file_list)} 份试卷")
    all_research_items = []
    for fname,full_text in file_list:
        print(f"\n处理文件：{fname}")
        chunks = re.split(r"\n{3,}",full_text)
        for ck in chunks:
            ck = ck.strip()
            if len(ck)<80:
                continue
            resp_ext = llm.invoke([{"role":"user","content":EXTRACT_PROMPT.format(text_chunk=ck)}])
            raw = resp_ext.content
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                continue
            try:
                q_arr = json.loads(json_match.group())
                for item in q_arr:
                    q = item.get("question","")
                    cat = item.get("category","未知分类")
                    print(f" >>教研题目 {q[:50]}... 分类:{cat}")
                    ana_resp = llm.invoke([{"role":"user","content":RESEARCH_PROMPT.format(q_text=q)}])
                    all_research_items.append({
                        "category":cat,
                        "question":q,
                        "research_content":clean_model_answer(ana_resp.content)
                    })
            except Exception as e:
                print(f"片段解析跳过 {e}")
    #导出word
    doc = Document()
    doc.add_heading("高考数学真题教研资料", level=1)
    group = {c:[] for c in CATEGORY_LIST}
    group["其他"]=[]
    for it in all_research_items:
        c=it["category"]
        if c in group:
            group[c].append(it)
        else:
            group["其他"].append(it)
    for cat_name,qlist in group.items():
        if not qlist:continue
        doc.add_heading(f"▶ {cat_name}",level=2)
        for qi in qlist:
            doc.add_paragraph(f"【原题】{qi['question']}")
            for para in qi["research_content"].split("\n"):
                if para.strip():
                    doc.add_paragraph(para.strip())
            doc.add_paragraph("‑"*60)
    word_path = Path(OUT_RESEARCH_FOLDER)/"高考数学教研汇总.docx"
    doc.save(str(word_path))
    #导出pdf
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c = canvas.Canvas(str(Path(OUT_RESEARCH_FOLDER)/"高考数学教研汇总.pdf"),pagesize=A4)
    c.setFont("STSong-Light",11)
    w,h = A4
    y = h-40
    def wline(txt):
        nonlocal y
        if y<40:
            c.showPage()
            c.setFont("STSong-Light",11)
            y=h-40
        c.drawString(30,y,txt[:110])
        y-=16
    wline("====================高考数学真题教研资料====================")
    for it in all_research_items:
        wline(f"【分类】{it['category']}")
        wline(f"【原题】{it['question'][:120]}")
        for line in it["research_content"].split("\n"):
            if line.strip():
                wline(line.strip())
        wline("="*70)
    c.save()
    print(f"\n✅教研完成！")
    print(f"Word：{word_path.resolve()}")
    print(f"PDF：{(Path(OUT_RESEARCH_FOLDER)/'高考数学教研汇总.pdf').resolve()}")

# ========================主菜单========================
def print_main_menu():
    print("""
==================== 高考数学一体化Agent 主菜单 ====================
1 ｜单题交互模式（记忆+RAG+大题拆解+绘图+校验复核）
2 ｜构建/更新RAG知识库【支持 txt/md/docx/pdf】
3 ｜批量做题（batch_input支持 txt / md / docx / pdf，输出docx+pdf）
4 ｜真题教研模块：批量读取raw_exam_files中PDF/Word，输出教研Word+PDF
q ｜退出程序
===================================================================
""")

if __name__ == "__main__":
    while True:
        print_main_menu()
        opt = input("请输入功能数字：").strip()
        if opt.lower()=="q":
            print("程序结束")
            break
        if opt=="1":
            thread_id = "chat_session01"
            cfg = {"configurable":{"thread_id":thread_id}, "recursion_limit": 10}
            print("\n====单题交互，clear清空记忆；exit退回主菜单====\n")
            while True:
                user_in = input("请输入题目：").strip()
                if user_in.lower()=="exit":
                    break
                if user_in.lower()=="clear":
                    graph_chat.update_state(cfg,{"messages":[]})
                    print("✅记忆清空")
                    continue
                t0 = time.time()
                ans_text = run_question(graph_chat, user_in, cfg)
                print(f"（本题用时 {time.time()-t0:.0f} 秒）")
                print("\n【输出】")
                print(ans_text)
                print("‑"*80)
        elif opt=="2":
            build_rag_vector_db()
        elif opt=="3":
            exts = (".txt",".md",".docx",".pdf")
            files = [f for f in os.listdir(BATCH_INPUT_FOLDER) if Path(f).suffix.lower() in exts]
            if not files:
                print(f"batch_input文件夹没有可识别试卷，支持格式：{exts}")
                continue
            for idx,f in enumerate(files):
                print(f"{idx}: {f}")
            sel = int(input("选择试卷序号："))
            target = files[sel]
            full_text,_ = batch_solve_file(target)
            pdf_out = os.path.join(OUTPUT_FOLDER, Path(target).stem+"_result.pdf")
            text_to_pdf(full_text, pdf_out)
        elif opt=="4":
            research_workflow()
        else:
            print("无效输入，请重新选择")
