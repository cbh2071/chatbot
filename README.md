# 🧬 智能蛋白质功能预测助手 (LLM + MCP) / Intelligent Protein Function Prediction Assistant (LLM + MCP)

**[项目状态：进行中]** 
## 简介 (Introduction)

本项目旨在构建一个**智能的生物信息学助手**，通过自然语言聊天界面与用户交互，以预测蛋白质的功能。它利用**大型语言模型 (LLM)** 理解用户复杂的请求，并借助**模型上下文协议 (MCP)** 作为标准接口，协调调用后台工具，包括我们团队自行构建的蛋白质功能预测模型、UniProt 数据库查询工具等。

与简单的“输入序列->输出预测”工具不同，本项目的目标是创建一个能够理解上下文、规划任务、主动获取数据并以友好方式呈现结果的智能代理 (Agent)。

This project aims to build an **intelligent bioinformatics assistant** that interacts with users via a natural language chat interface to predict protein functions. It leverages **Large Language Models (LLMs)** to understand complex user requests and utilizes the **Model Context Protocol (MCP)** as a standardized interface to coordinate calls to backend tools, including our team's custom protein function prediction model, UniProt database query tools, and potentially others.

Unlike simple "input sequence -> output prediction" tools, the goal is to create an intelligent agent capable of understanding context, planning tasks, proactively fetching data, and presenting results in a user-friendly manner.

## 主要功能 (Features)

*   **自然语言交互 (Natural Language Interaction):** 通过聊天界面理解用户的指令和问题。
*   **蛋白质信息查询 (Protein Information Retrieval):** 根据 UniProt ID 或名称获取蛋白质序列、物种等信息。
*   **数据库搜索 (Database Search):** 根据用户提供的关键词（如基因名、功能描述、物种）在 UniProt 中搜索蛋白质。
*   **功能预测 (Function Prediction):** 利用团队构建的机器学习模型（支持多分类/多标签）预测给定蛋白质序列的功能。
*   **上下文理解与对话管理 (Context Understanding & Dialogue Management):** 处理多轮对话，并在信息不足时进行澄清。
*   **任务规划 (Task Planning):** (由 LLM 辅助) 将复杂请求分解为调用不同 MCP 工具的步骤。
*   **(可选) 结果可视化 (Result Visualization):** (若实现) 以图表形式展示预测结果的分布等。

## 系统架构 (Architecture)

本项目采用基于 LLM 的 Agent 架构，通过 MCP 连接前后端组件：

```mermaid
graph LR
    User[用户] -- 自然语言 --> ChatUI[聊天界面 (Gradio/Streamlit)]
    ChatUI -- 用户输入/事件 --> AgentCore[智能代理核心 (Python + LLM API)]

    subgraph Agent Core Logic
        AgentCore -- 理解/规划/生成 --> LLM_API[LLM API (OpenAI/Claude/Gemini...)]
        AgentCore -- 状态/历史 --> DialogMgr[对话管理器]
        AgentCore -- 工具调用 --> MCPClient[MCP 客户端]
    end

    subgraph Backend Tools via MCP
        MCPClient -- MCP协议 --> PredServer[预测模型 MCP 服务器]
        MCPClient -- MCP协议 --> UniProtServer[UniProt查询 MCP 服务器]
        MCPClient -- MCP协议 --> PlotServer[(可选) 绘图 MCP 服务器]
    end

    subgraph MCP Servers Implementation
        PredServer -- 调用 --> PredictionModel[团队构建的预测模型 (TF/PyTorch)]
        UniProtServer -- API调用 --> UniProtAPI[UniProt API]
        PlotServer -- 调用 --> PlotLib[绘图库 (Matplotlib/Plotly)]
    end

    AgentCore -- 格式化回复 --> ChatUI
    ChatUI -- 文本/图表 --> User
```

*   **聊天界面 (ChatUI):** 用户交互的前端。
*   **智能代理核心 (AgentCore):** 由 LLM 驱动，负责 NLU、对话管理、任务规划和回复生成。
*   **MCP 客户端 (MCPClient):** AgentCore 内的模块，负责与 MCP 服务器通信。
*   **MCP 服务器 (MCPServers):** 封装具体功能的后端服务，遵循 MCP 协议。
    *   **预测服务器 (PredServer):** 包装团队训练的预测模型。
    *   **UniProt 服务器 (UniProtServer):** 提供 UniProt 数据库查询和数据获取功能。
    *   **(可选) 绘图服务器 (PlotServer):** 提供数据可视化功能。
*   **LLM API:** 提供核心的自然语言处理和推理能力。

## 技术栈 (Technology Stack)

*   **核心逻辑与 Agent:** Python 3.x
*   **LLM 集成:** OpenAI API (GPT-4/3.5), Anthropic API (Claude 3), Google Gemini API (或选择其一/多)
*   **MCP 实现:** `mcp[cli]` Python SDK
*   **MCP 服务器:** Python, FastAPI / Stdio
*   **预测模型:** TensorFlow / Keras / PyTorch 
*   **数据交互:** `httpx` (异步 HTTP 请求)
*   **前端界面:** Gradio / Streamlit 
*   **(可选) 绘图:** Matplotlib / Seaborn / Plotly

## 安装与设置 (Setup & Installation)

1.  **克隆仓库 (Clone Repository):**
    ```bash
    git clone https://github.com/cbh2071/chatbot.git
    cd chatbot
    ```

2.  **创建并激活 Python 环境 (Create & Activate Environment):** (推荐使用 `venv` 或 `conda`)
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖 (Install Dependencies):**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量 (Configure Environment Variables):**
    *   复制 `.env.example` 文件为 `.env`。
    *   编辑 `.env` 文件，填入必要的 API 密钥：
        *   `OPENAI_API_KEY=` (如果使用 OpenAI)
        *   `ANTHROPIC_API_KEY=` (如果使用 Anthropic)
        *   `GOOGLE_API_KEY=` (如果使用 Google Gemini)
        *   (其他可能需要的密钥或配置)
    *   **重要:** `.env` 文件不应提交到版本控制中 (已在 `.gitignore` 中配置)。

5.  **放置模型文件 (Place Model Files):**
    *   如果预测模型文件较大，请根据说明将其放置在指定目录（例如 `models/`）。

## 如何运行 (Usage)

1.  **启动应用 (Start the Application):**
    ```bash
    # 假设主应用入口是 app_with_agent.py
    python app_with_agent.py
    ```
    或者根据你使用的 UI 框架启动，例如 `gradio app_with_agent.py`。

2.  **打开浏览器 (Open Browser):**
    *   应用启动后，通常会在终端显示一个本地 URL (例如 `http://127.0.0.1:7860`)。在浏览器中打开此 URL。

3.  **开始对话 (Start Chatting):**
    *   在聊天框中输入你的请求，例如：
        *   `你好`
        *   `预测一下蛋白质 P00533 的功能。`
        *   `EGFR_HUMAN 的序列是什么？`
        *   `搜索一下人类的酪氨酸激酶。`
        *   `请查找与阿尔茨海默病相关的人类蛋白质，并预测它们的功能。`

## 项目结构 (Project Structure)

```
.
├── .env.example          # 环境变量示例文件
├── .gitignore            # Git 忽略文件配置
├── README.md             # 项目说明文件 (本文档)
├── requirements.txt      # Python 依赖列表
├── config.py             # 配置文件 (API密钥路径, 模型设置, MCP工具定义等)
├── agent_core.py         # 智能代理核心逻辑
├── llm_clients.py        # 与 LLM API 交互的客户端
├── mcp_clients.py        # MCP 客户端逻辑 (如果独立)
├── mcp_server_prediction.py # 预测模型的 MCP 服务器
├── mcp_server_uniprot.py    # UniProt 查询的 MCP 服务器
├── model_predictor.py    # 实际加载和调用预测模型的逻辑
├── protein_utils.py      # 蛋白质序列验证、UniProt 数据获取等工具函数
├── app_with_agent.py     # Gradio/Streamlit 应用入口文件
├── models/                 # (可选) 存放模型权重文件的目录
└── data/                   # (可选) 存放示例数据或缓存的目录
```

## 未来改进方向 (Future Improvements)

*   [ ] 增加更多的 MCP 工具 (例如：本地 BLAST、其他生物信息学数据库接口)。
*   [ ] 提升 Agent 的任务规划能力，处理更复杂的嵌套任务。
*   [ ] 集成模型可解释性工具，让 Agent 能解释预测依据。
*   [ ] 增加更丰富的可视化输出。
*   [ ] 支持更多类型的输入（例如 PDB 文件 ID、基因名称）。
*   [ ] 实现结果缓存以提高效率。
*   [ ] 优化错误处理和用户反馈。

## 贡献 (Contributing)

本项目为课程大作业，目前主要由团队成员开发。如果您发现 Bug 或有改进建议，欢迎提出 Issue。

## 作者 (Authors)

*   [你的名字/团队成员1] - [职责，例如：Agent 核心开发]
*   [团队成员2] - [职责，例如：MCP 预测服务器]
*   [团队成员3] - [职责，例如：MCP UniProt 服务器 & UI]


