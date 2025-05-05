# mcp_server.py
import sys     # 导入sys库，用于访问标准输入/输出/错误流和退出程序
import logging # 导入logging库
import asyncio # 导入asyncio库
import os      # 导入os库，用于获取进程ID
from typing import Dict, Any, List

def validate_sequence(sequence: str) -> bool:
    """验证蛋白质序列是否只包含有效的氨基酸字符"""
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY-")
    return all(char in valid_chars for char in sequence.upper())

# 确保根目录在 sys.path 中，以便导入其他模块
# 如果你的项目结构需要，取消下面的注释并调整路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# 尝试导入必要的模块
try:
    # 从mcp库导入FastMCP类，用于快速创建MCP服务器
    from mcp.server.fastmcp import FastMCP
    # 从我们自己的模块导入实际的预测逻辑函数
    from model_predictor import predict_protein_function as predict_tool_impl
    # 从我们自己的模块导入序列验证函数
    from protein_utils import fetch_protein_data as get_data_impl
    from protein_utils import search_proteins as search_tool_impl
    # 从mcp.types导入用于类型提示的模块
    import mcp.types as types
except ImportError as e:
    # 如果导入失败（例如，依赖未安装或路径问题），向标准错误输出错误信息
    # 因为标准输出可能被MCP协议占用，关键错误信息应输出到stderr
    print(f"导入模块时出错: {e}。请确保 mcp, model_predictor.py, 和 protein_utils.py 可访问且依赖已安装。", file=sys.stderr)
    # 退出程序，返回错误码1
    sys.exit(1)

# 配置日志记录器，将日志输出到标准错误流(stderr)
# 这是因为MCP通过标准输出(stdout)进行协议通信，不能被日志干扰
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s [MCP Server] %(levelname)s: %(message)s')
# 获取名为__name__（即当前模块名'mcp_server'）的日志记录器实例
logger = logging.getLogger(__name__)

# 初始化 FastMCP 服务器实例
# 'name' 参数定义了服务器的名称，如果与MCP客户端（如Claude Desktop）集成，这个名称需要匹配客户端配置中的名称
# 'version' 参数定义了服务器的版本
mcp = FastMCP(name="protein_tools_server")

# 使用 @mcp.tool() 装饰器将一个异步函数注册为MCP工具
# MCP客户端可以通过调用这个工具来执行相应的操作
@mcp.tool()
async def predict_protein_function_tool(
    sequence: str,      # 必需参数：蛋白质序列字符串
    organism: str = ""  # 可选参数：来源物种的科学名称，提供额外上下文
) -> Dict[str, Any]:
    """
    MCP 工具：根据蛋白质的氨基酸序列预测其生物学功能。

    输入参数:
        sequence (str): 必需，有效的蛋白质序列字符串。
        organism (str): 可选，提供来源物种信息。

    返回:
        dict: 包含预测结果的字典，成功时有 'predicted_function', 'confidence', 'model_version', 'processing_time_sec' 键。
              如果预测失败，则包含 'error' 键及错误描述。
    """
    logger.info(f"MCP 工具 'predict_protein_function_tool' 被调用。")

    # 检查必需的 sequence 参数是否存在且非空
    if not sequence:
        logger.error("工具错误：输入的序列参数缺失或为空。")
        # 返回一个包含错误信息的字典，符合MCP错误响应格式
        return {"error": "参数 'sequence' 是必需的，不能为空。"}

    # 在调用模型之前，先验证输入序列的格式
    if not validate_sequence(sequence):
         logger.error(f"工具错误：检测到无效的序列格式。序列开头: {sequence[:30]}...")
         # 返回具体的错误信息
         return {"error": "无效的序列格式。只允许标准的氨基酸字符 (ACDEFGHIKLMNPQRSTVWY-)。"}

    logger.info(f"序列 (长度={len(sequence)}) 已通过验证。开始调用预测逻辑...")
    try:
        # 调用从 model_predictor 模块导入的实际预测函数
        result = await predict_tool_impl(sequence, organism)

        # 检查预测函数本身是否返回了错误（在model_predictor内部处理的错误）
        if "error" in result:
            logger.error(f"预测逻辑返回错误: {result['error']}")
            # 将内部错误直接传递给MCP客户端
            return result
        else:
            # 预测逻辑成功完成
            logger.info("预测逻辑成功完成。")
            # 返回包含预测结果的字典
            return result

    # 处理任务被取消的异常
    except asyncio.CancelledError:
         logger.warning("预测任务被取消。")
         return {"error": "预测被取消。"}
    # 捕获所有其他在预测过程中可能发生的未预料异常
    except Exception as e:
        # 使用 logger.exception 记录完整的错误信息和堆栈跟踪
        logger.exception("工具错误：预测过程中发生意外错误。")
        # 向客户端返回一个用户友好的内部错误消息
        return {"error": f"预测服务器内部发生错误: {type(e).__name__}"}

@mcp.tool()
async def get_protein_data(identifier: str) -> Dict[str, Any] | None:
    """
    根据 UniProt 登录号或入口名称获取蛋白质的详细信息，包括序列、物种等。

    Args:
        identifier: 必需，UniProt 登录号 (如 P00533) 或入口名称 (如 INS_HUMAN)。

    Returns:
        包含 'sequence', 'organism', 'id' 等信息的字典，或在未找到时返回 null，或在出错时包含 'error' 的字典。
    """
    logger.info(f"Tool 'get_protein_data' called with identifier: {identifier}")
    try:
        # 注意：fetch_protein_data 已经设计为在找不到或出错时返回 None
        # 但 MCP 工具最好返回包含 error 的字典或引发异常，让 FastMCP 处理
        result = await get_data_impl(identifier)
        if result is None:
            # 如果 get_data_impl 内部处理了 not found 并返回 None，我们将其转换
            # 但更好的做法是让 get_data_impl 在找不到时引发特定异常，或直接返回含 error 的 dict
             logger.warning(f"Identifier '{identifier}' not found or failed to fetch.")
             # 返回 None 在 JSON-RPC 中是合法的 null 结果
             return None # 或者 return {"error": f"Identifier '{identifier}' not found."}
        logger.info(f"Data fetched for {identifier}: Found ID {result.get('id')}")
        return result
    except Exception as e:
        logger.exception("Error in get_protein_data tool")
        return {"error": f"Internal server error fetching data for {identifier}: {type(e).__name__}"}

@mcp.tool()
async def search_proteins(query: str, species_filter: str | None = None, keyword_filter: str | None = None, limit: int = 10) -> List[Dict[str, Any]] | Dict[str, str]:
    """
    根据关键词、物种等条件在 UniProt 数据库中搜索蛋白质。

    Args:
        query: 必需，搜索关键词（如基因名、功能描述等）。
        species_filter: 可选，物种科学名称或 NCBI 分类 ID 进行过滤。
        keyword_filter: 可选，使用 UniProt 关键字进行过滤。
        limit: 可选，限制返回结果的数量，默认为 10。

    Returns:
        包含蛋白质列表的数组，每个蛋白质是一个包含 'id', 'name', 'organism' 等信息的对象；或在出错时包含 'error' 的字典。
    """
    logger.info(f"Tool 'search_proteins' called with query: '{query}', species: {species_filter}, limit: {limit}")
    # --- 实现 search_tool_impl ---
    # 你需要在 protein_utils.py 或这里实现 search_tool_impl 函数
    # 它应该调用 UniProt 的搜索 API
    # 例如: https://rest.uniprot.org/uniprotkb/search?query=your_query&fields=accession,id,organism_name&size=limit
    # 需要处理查询构建、API 调用、错误处理和结果格式化
    # --- --------------------- ---
    try:
        # 确保 limit 有一个合理的上限，防止请求过多数据
        safe_limit = min(limit, 50) # 例如，最多返回 50 条
        results = await search_tool_impl(query, species_filter, keyword_filter, safe_limit)
        logger.info(f"Search returned {len(results)} results.")
        return results
    except Exception as e:
        logger.exception("Error in search_proteins tool")
        return {"error": f"Internal server error during protein search: {type(e).__name__}"}

# 当这个脚本作为主程序运行时（即 `python mcp_server.py`）
if __name__ == "__main__":
    # 记录服务器启动信息，包括进程ID (PID) 和通信方式 (stdio)
    logger.info(f"启动蛋白质功能预测 MCP 服务器 (PID: {os.getpid()})，使用 stdio 传输...")
    try:
        # 运行 MCP 服务器，指定 transport='stdio' 表示使用标准输入/输出进行通信
        # 这个调用会阻塞，直到服务器被终止
        mcp.run(transport='stdio')
    except Exception as e:
        # 如果服务器启动或运行过程中发生错误，记录异常并退出
        logger.exception("MCP 服务器运行失败。")
        sys.exit(1)
    finally:
        # 当服务器停止时（正常或异常终止），记录一条消息
        logger.info("蛋白质功能预测 MCP 服务器已停止。")