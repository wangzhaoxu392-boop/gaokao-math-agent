# 🧮 高考数学一体化 Agent

基于 **LangGraph + Ollama** 的数学教研与解题智能体，覆盖「做一道题 → 研究一道题 → 做整卷 → 整卷教研 → 备课讲义」的全流程。

支持**混合部署**：推理/拆题默认走本地 Ollama，数学模型可按 `.env` 切换为
本地 Ollama（数据不出本机）或云端 API（DeepSeek 官方 / 硅基流动，速度与准确率更高）。

## ✨ 功能一览（网页版七个页签，按使用顺序排列）

| 页签 | 说明 |
|---|---|
| **① 单题解题** | 输入一道题（可含多小问）→ 标准答案 + 逐步推导；支持追问、一键复制、导出 Word、查看历史记录；需要时自动画函数图像 |
| **② 题目档案** | 把一道题升级成题型：11 字段档案（考点 / 教材原型 / 两种解法 / 易错点 / 变式方向）+ 命题人视角 4 个完整变式，可导出 Word |
| **③ 批量做题** | 整卷（txt/md/docx/pdf）逐题批改 → Word + PDF，并给出**逐题统计**（用时 / 费用 / token）；支持断点续跑、自动重做错题、省钱模型档；可上传学生答案**逐题判分**并生成报告 |
| **④ 真题教研** | 历年真题 → 教研资料（标准答案 / 详细解答 / 第二解法 / 易错点 / 命题特点 / 一题三变），输出 Word + PDF |
| **⑤ 精细化讲义** | 按 12 板块模板生成专题精讲讲义（概念 / 公式 / 四级题型 / 易错点 / 重难点 / 真题映射），自动导出 Word；**同专题重复生成保留 v1/v2 版本**，可板块级对比后自行择优 |
| **⑥ 知识库** | 把 `knowledge_docs/` 的资料编译成结构化 Wiki 词条，解题时自动检索；未编译时回退旧版向量库 |
| **⑦ 学情画像** | 汇总历次判分：按考点算平均得分率、定位薄弱项；错题一键导出 Word 打印 |
| **⑧ 格式转换** | docx / pdf / txt → Markdown，Markdown → Word |

顶部常驻**用量看板**：账户余额、今日花费 / 预算（进度条）、今日 token、累计解题与均价、当前模型、预计还能跑多少题；15 秒自动刷新。

各功能使用独立并发通道，**可以同时运行**：批量做题跑着的同时，单题解题、判分、讲义生成互不阻塞（同一功能内部仍串行，避免抢占本地模型与 API）。

## 🎛 模型怎么选（实测）

| 模型 | 位置 | 单题耗时 | 单题费用 | 适合场景 |
|---|---|---|---|---|
| `deepseek-v4-pro` | 云端（默认） | ~50 秒 | ~¥0.043 | 压轴题、证明题、讲义/教研长文生成 |
| `deepseek-flash` | 云端（省钱档） | ~11 秒 | ~¥0.028 | 日常刷题、整卷批改 |
| `deepseek-r1:14b` | 本机 Ollama | ~59 秒 | ¥0 | 断网、不想花钱；质量次之 |

- **云端模型按 token 计费**，从账户余额扣；题目越长、工具轮次越多，输入 token 越大
- **真正免费的只有本地 Ollama 模型**：把 `.env` 的 `MATH_PROVIDER` 改成 `ollama` 即可，不产生任何 API 费用
- 网页版 ①③ 页签的「档位」下拉可随时切标准/省钱，只影响数学模型，推理模型不变

## 💸 哪些功能不花钱

判断标准只有一条：**是否调用云端大模型**。

| 功能 | 花钱吗 | 说明 |
|---|---|---|
| 单题解题、题目档案、批量做题、判分、真题教研、精细化讲义、Wiki 编译 | 💰 按 token | 约 ¥0.01~0.05／题（档位不同） |
| 解题历史载入、导出 Word、一键复制 | 🆓 免费 | 纯本地读写 |
| 看板刷新、余额查询 | 🆓 免费 | 余额接口不计费 |
| 上传资料、Wiki 检索 | 🆓 免费 | 本地 Ollama embedding（nomic-embed-text） |
| 学情画像、错题本导出 | 🆓 免费 | 本地聚合 + python-docx |
| 格式转换（docx/pdf/txt ⇄ Markdown ⇄ Word） | 🆓 免费 | PyMuPDF / python-docx |
| 全部 LLM 功能（`MATH_PROVIDER=ollama`） | 🆓 免费 | 换本机模型，慢一些、质量次之 |

长任务（批量 / 判分 / 教研 / 讲义 / Wiki）都可以点「⏹ 停止」：在下一题或下一批的边界停下，
已完成的结果保留，批量还能断点续跑。

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
- **模型路由**：`MATH_PROVIDER=ollama` 时全部本地推理；`deepseek` / `siliconflow` 时
  推理仍走本地，数学计算走云端 API（此时题目文本会发送到对应服务商）。

## ⚠️ 说明

- 功能①依赖本地模型算力，**简单题约 1 分钟、大题约 5 分钟**，属正常现象；网页版提交后请耐心等待。
- 解题图有收敛保护：`MAX_TOOL_ROUNDS`（单题最多几轮工具调用，默认 3）与
  `RECURSION_LIMIT`（步数上限，默认 30）。模型"反复验算停不下来"时靠这两个参数兜底。
- `NUM_CTX` 控制本地模型上下文（默认 8192）。不设置时 Ollama 只给 4096，长题或知识库
  提示词会被静默截断，表现为"答非所问/漏条件"。
- `ENABLE_CHECKER=on` 时解题后会再复核一遍（质量更稳，耗时约翻倍）；追求速度可设 `off`。
- 本项目生成的教研内容仅供教学参考，请以教材与考试说明为准。

## 📄 许可证

[MIT License](LICENSE)
