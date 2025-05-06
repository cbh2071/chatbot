# 🧬 智能蛋白质功能预测助手/ Intelligent Protein Function Prediction Assistant

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
    User[用户] -- 自然语言 --> ChatUI[聊天界面 (Gradio)] # 更新为 Gradio
    ChatUI -- 用户输入/事件 --> AgentCore[智能代理核心 (Python + LLM API)]

    subgraph Agent Core Logic
        AgentCore -- 理解/规划/生成 --> LLM_API[LLM API (OpenAI/Claude/Gemini...)]
        AgentCore -- 状态/历史 --> DialogMgr[对话管理器]
        AgentCore -- 工具调用 --> MCPClient[MCP 客户端 (in AgentCore)] # MCP Client 在 AgentCore 内部
    end

    subgraph Backend Tools via MCP
        AgentCore -- MCP协议 (stdio) --> MCPServer[MCP 服务器 (mcp_server.py)] # AgentCore 直接通过 stdio 启动和通信
    end

    subgraph MCP Server Implementation
        MCPServer -- 导入/调用 --> PredictionModel[预测模型逻辑 (model_predictor.py)]
        MCPServer -- 导入/调用 --> UniProtUtils[UniProt工具逻辑 (protein_utils.py)]
        # PlotServer 可类似地被 MCPServer 导入
    end

    AgentCore -- 格式化回复 --> ChatUI
    ChatUI -- 文本/图表 --> User
```
*   **聊天界面 (ChatUI):** 用户交互的前端 (当前使用 Gradio)。
*   **智能代理核心 (AgentCore):** 由 LLM 驱动，负责 NLU、对话管理、任务规划、回复生成，并内置 MCP 客户端逻辑与 MCP 服务器通过 stdio 通信。
*   **MCP 服务器 (MCPServer):** 单个 Python 进程 (`mcp_server.py`)，使用 `mcp` SDK (FastMCP) 暴露工具接口，由 AgentCore 作为子进程启动和管理。
    *   **预测功能:** 调用 `model_predictor.py` 中的逻辑。
    *   **UniProt 功能:** 调用 `protein_utils.py` 中的逻辑。
*   **LLM API:** 提供核心的自然语言处理和推理能力。

## 技术栈 (Technology Stack)

*   **核心逻辑与 Agent:** Python 3.x
*   **LLM 集成:** OpenAI API, Anthropic API, Google Gemini API (通过 `llm_clients.py`)
*   **MCP 实现:** `mcp[cli]` Python SDK (版本 >= 1.7.1 推荐)
*   **MCP 服务器:** Python, `mcp` SDK (FastMCP, stdio transport)
*   **预测模型:** TensorFlow / Keras / PyTorch (在 `model_predictor.py` 中调用)
*   **数据交互:** `httpx` (异步 HTTP 请求)
*   **前端界面:** Gradio
*   **环境管理:** `uv` (推荐)
*   **配置加载:** `python-dotenv`

## 安装与设置 (Setup & Installation using uv - Recommended)

[uv](https://github.com/astral-sh/uv) 是一个极快的 Python 包安装器和解析器。

1.  **安装 uv (Install uv):**
    *   请遵循官方安装指南：<https://astral.sh/uv#installation>
    *   通常涉及通过 `curl` (Linux/macOS) 或 `pipx` (跨平台) 或 PowerShell (Windows) 运行命令。安装后请**重启终端**。

2.  **克隆仓库 (Clone Repository):**
    ```bash
    git clone https://github.com/cbh2071/chatbot.git
    cd chatbot
    ```

3.  **创建并激活 Python 环境 (Create & Activate Environment using uv):**
    ```bash
    # 创建虚拟环境 (默认名为 .venv)
    uv venv
    # 激活环境
    # Linux/macOS:
    source .venv/bin/activate
    # Windows (Command Prompt):
    # .venv\Scripts\activate.bat
    # Windows (PowerShell):
    # .venv\Scripts\Activate.ps1
    ```

4.  **安装依赖 (Install Dependencies using uv):**
    *   安装核心依赖：
        ```bash
        # 确保你的 requirements.txt 是最新的 (见下一节)
        uv pip install -r requirements.txt
        # 或者直接指定核心包 (推荐固定版本)
        # uv pip install "mcp[cli]>=1.7.1" httpx gradio python-dotenv
        ```
    *   根据需要安装可选的 LLM 客户端：
        ```bash
        # uv pip install openai
        # uv pip install anthropic
        # uv pip install google-generativeai
        ```

5.  **配置环境变量 (Configure Environment Variables):**
    *   将项目根目录下的 `env.example` 文件复制为 `.env`。
        ```bash
        # Linux/macOS
        cp env.example .env
        # Windows
        # copy env.example .env
        ```
    *   **编辑 `.env` 文件**，填入你需要的 API 密钥 (参考 `config.py` 来确定需要哪些)：
        ```dotenv
        # .env file example
        DEFAULT_LLM_PROVIDER=openai # 或 anthropic, google, aihubmix, deepseek, ark
        DEFAULT_LLM_MODEL= # 可选, 填入特定模型名, 如 gpt-4o-mini

        OPENAI_API_KEY=sk-YourOpenAIKey...
        ANTHROPIC_API_KEY=sk-ant-YourAnthropicKey...
        GOOGLE_API_KEY=AIzaSyYourGoogleKey...
        AIHUBMIX_API_KEY=YourAiHubMixKey...
        DEEPSEEK_API_KEY=sk-YourDeepSeekKey...
        ARK_API_KEY=YourArkKey... # 火山方舟
        ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/ # 火山方舟 Base URL (如果需要修改)

        # 其他可能的配置...
        ```
    *   **重要:** `.env` 文件不应提交到版本控制中 (已在 `.gitignore` 中配置)。

6.  **模型文件 (Model Files):**
    *   如果你的蛋白质功能预测模型 (`model_predictor.py` 需要加载的文件) 不是随代码库一起提供，请按照指示将其放置在正确的目录（例如，如果代码期望在 `models/` 目录下找到它们）。

## 如何运行 (Usage)

**确保你已经激活了使用 `uv` 创建的虚拟环境。**

1.  **启动 Gradio Web UI 应用:**
    *   在终端中运行：
        ```bash
        python app_with_agent.py
        ```
    *   *(或者使用 `uv run`: `uv run python app_with_agent.py`)*
    *   **注意:** `app_with_agent.py` 内部的 `AgentCore` 会自动启动 `mcp_server.py` 作为子进程，并通过 stdio 与其通信。**你不需要单独启动 `mcp_server.py`。**

2.  **打开浏览器 (Open Browser):**
    *   应用启动后，终端会显示一个本地 URL (通常是 `http://127.0.0.1:7860` 或类似地址)。在你的网页浏览器中打开这个 URL。

3.  **开始对话 (Start Chatting):**
    *   在 Gradio 界面的聊天框中输入你的请求，例如：
        *   `你好`
        *   `预测一下蛋白质 P00533 的功能。`
        *   `EGFR_HUMAN 的序列是什么？`
        *   `搜索一下人类的酪氨酸激酶。` (Search for human tyrosine kinases.)
        *   `请查找与阿尔茨海默病相关的人类蛋白质，并预测它们的功能。` (Find human proteins related to Alzheimer's disease and predict their functions.)

## 项目结构 (Project Structure)

```
.
├── .env                # 实际的环境变量文件 (不提交)
├── .env.example        # 环境变量示例文件
├── .gitignore          # Git 忽略文件配置
├── README.md           # 项目说明文件 (本文档)
├── requirements.txt    # Python 依赖列表 (uv 可读)
├── config.py           # 配置文件 (API密钥加载, 模型设置, MCP工具定义等)
├── agent_core.py       # 智能代理核心逻辑 (包含 MCP 客户端)
├── llm_clients.py      # 与 LLM API 交互的客户端
├── mcp_server.py       # MCP 服务器实现 (包含所有工具)
├── model_predictor.py  # 实际加载和调用预测模型的逻辑
├── protein_utils.py    # 蛋白质序列验证、UniProt 数据获取/搜索等工具函数
├── app_with_agent.py   # Gradio 应用入口文件
├── models/             # (可选) 存放模型权重文件的目录
└── data/               # (可选) 存放示例数据或缓存的目录
```


## 未来改进方向 (Future Improvements)

*   [ ] 增加更多的 MCP 工具 (例如：本地 BLAST、其他生物信息学数据库接口)。
*   [ ] 提升 Agent 的任务规划能力，处理更复杂的嵌套任务。
*   [ ] 集成模型可解释性工具，让 Agent 能解释预测依据。
*   [ ] 增加更丰富的可视化输出。
*   [ ] 支持更多类型的输入（例如 PDB 文件 ID、基因名称）。
*   [ ] 实现结果缓存以提高效率。
*   [ ] 优化错误处理和用户反馈。

