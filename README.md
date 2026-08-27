# 🧮 高考数学一体化 Agent

基于 **LangGraph + 本地 Ollama** 的数学教研与解题智能体，四个功能一体化：
单题深度讲解、知识库问答（RAG）、批量试卷批改、真题教研资料生成。全程本地运行，数据不出本机。

## ✨ 功能一览

| 功能 | 说明 |
|---|---|
| **① 单题交互** | 输入一道数学题（可含多小问），自动拆题 → 逐步推理 → 输出标准答案 + 详细步骤 + 解题方法，需要时可自动绘制函数图像 |
| **② RAG 知识库** | 把讲义 / 知识点文档放入 `knowledge_docs`，一键构建向量库，解答时可检索参考 |
| **③ 批量做题** | 把整份试卷（txt/md/docx/pdf）放入 `batch_input`，自动逐题批改，输出 Word + PDF |
| **④ 真题教研** | 把历年真题（pdf/docx）放入 `raw_exam_files`，自动提取题目并生成教研资料（标准答案 / 详细解答 / 解题方法 / 第二种解法 / 易错点 / 命题特点 / 一题三变），输出 Word + PDF |

提供两种操作界面：
- 🌐 **网页操作版**（推荐）：Gradio 网页界面，浏览器操作，支持文件上传与结果下载
- 💻 **命令行版**：`math_agent_all.py` 终端菜单

## 🔧 环境要求

- Windows / macOS / Linux
- Python 3.10+
- [Ollama](https://ollama.com/)（本地大模型运行时）

## 📦 安装步骤

### 1. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama 后，拉取两个模型（约 10GB）
ollama pull deepseek-r1:14b    # 主模型：解题推理
ollama pull qwen2.5:7b         # 快速模型：拆题提速
```

### 2. 配置 Python 环境

```bash
# 创建虚拟环境（示例为 Windows）
python -m venv .venv

# 激活（Windows）
.\.venv\Scripts\activate
# 激活（macOS / Linux）
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env     # Windows: copy .env.example .env
```
`.env` 中默认配置即为本地默认，无需修改；如需更换模型可编辑 `LLM_MODEL` / `FAST_MODEL`。

## 🚀 启动

### 网页操作版（推荐）

```bash
python math_agent_web.py
```
浏览器访问 <http://127.0.0.1:7860>。

> Windows 下也可双击 `启动网页版.bat`（自动检查并启动 Ollama、拉起服务并打开浏览器）。

### 命令行版

```bash
python math_agent_all.py
```
按菜单提示输入序号即可。

## 📁 目录结构

```
gaokao_math_agent/
├── math_agent_all.py        # 核心逻辑 + 命令行入口
├── math_agent_web.py        # 网页操作版（Gradio）
├── 启动网页版.bat            # Windows 一键启动脚本
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量示例
├── knowledge_docs/          # ② 知识库资料（放入文档）
├── raw_exam_files/          # ④ 真题文件（放入 pdf/docx）
├── batch_input/             # ③ 批量试卷（放入文件）
├── output/                  # ③ 批量做题输出
├── output_research/         # ④ 真题教研输出
└── db_chroma/               # ② 向量库（自动生成）
```

## 🛠 工作原理

- 基于 **LangGraph** 构建带状态的有向图 Agent，集成工具：
  - `sympy_math_calc`：符号计算（求导 / 解方程 / 数列求和等）
  - `plot_function`：绘制函数图像
  - `RAG`：基于 Chroma 的知识库检索
- 所有推理均在本地 Ollama 完成，不依赖任何云端 API，无需联网（除首次拉取模型）。
- 主模型负责推理，快速模型负责拆题等轻量环节，兼顾质量与速度。

## ⚠️ 说明

- 功能①依赖本地模型算力，**简单题约 1 分钟、大题约 5 分钟**，属正常现象；网页版提交后请耐心等待。
- 本项目生成的教研内容仅供教学参考，请以教材与考试说明为准。

## 📄 许可证

[MIT License](LICENSE)
