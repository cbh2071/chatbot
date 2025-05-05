# config.py
import os
import json
from dotenv import load_dotenv

# 从 .env 文件加载环境变量 (如果存在)
load_dotenv()

# LLM 配置
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL")

# 获取 API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AIHUBMIX_API_KEY = os.getenv("AIHUBMIX_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ARK_API_KEY = os.getenv("ARK_API_KEY")

# 火山方舟 Base URL
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/")

# MCP 服务器脚本路径
MCP_SERVER_SCRIPT = "mcp_server.py"

# 定义可用的 MCP 工具
AVAILABLE_MCP_TOOLS = {
    "predict_protein_function_tool": {
        "description": "根据蛋白质的氨基酸序列预测其生物学功能。需要提供序列，可选提供来源物种信息。",
        "parameters": {
            "sequence": {"type": "string", "description": "必需，蛋白质的氨基酸序列。", "required": True},
            "organism": {"type": "string", "description": "可选，蛋白质来源的物种科学名称。", "required": False}
        },
        "returns": {"type": "object", "description": "包含 'predicted_function', 'confidence', 'model_version', 'processing_time_sec' 的字典，或包含 'error' 的字典。"}
    },
    "get_protein_data": {
        "description": "根据 UniProt 登录号或入口名称获取蛋白质的详细信息，包括序列、物种等。",
        "parameters": {
            "identifier": {"type": "string", "description": "必需，UniProt 登录号 (如 P00533) 或入口名称 (如 INS_HUMAN)。", "required": True}
        },
        "returns": {"type": "object", "description": "包含 'sequence', 'organism', 'id' 等信息的字典，或包含 'error' 的字典，或 null。"}
    },
    "search_proteins": {
        "description": "根据关键词、物种等条件在 UniProt 数据库中搜索蛋白质。",
        "parameters": {
            "query": {"type": "string", "description": "必需，搜索关键词（如基因名、功能描述等）。", "required": True},
            "species_filter": {"type": "string", "description": "可选，物种科学名称或 NCBI 分类 ID 进行过滤。", "required": False},
            "keyword_filter": {"type": "string", "description": "可选，使用 UniProt 关键字进行过滤。", "required": False},
            "limit": {"type": "integer", "description": "可选，限制返回结果的数量，默认为 10。", "required": False}
        },
        "returns": {"type": "array", "description": "包含蛋白质列表的数组，每个蛋白质是一个包含 'id', 'name', 'organism' 等信息的对象，或包含 'error' 的字典。"}
    }
}

# 将工具描述转换为 JSON 字符串
AVAILABLE_MCP_TOOLS_JSON = json.dumps(AVAILABLE_MCP_TOOLS, indent=2, ensure_ascii=False)

print("DEBUG: config.DEFAULT_LLM_PROVIDER =", DEFAULT_LLM_PROVIDER)
print("DEBUG: config.DEFAULT_LLM_MODEL =", DEFAULT_LLM_MODEL)