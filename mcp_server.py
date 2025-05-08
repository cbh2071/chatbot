# mcp_server.py
import logging
import sys
import os
import asyncio
from typing import Dict, Any, List, Optional

# 确保根目录在 sys.path 中，以便导入其他模块 (如果需要)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# 导入官方 MCP Server 实现
from mcp.server.fastmcp import FastMCP
# 导入 mcp.types 用于类型提示 (如果工具函数需要返回特定类型)
# import mcp.types as types # 在这个例子中暂时不需要

# 导入你实际的工具函数实现
try:
    # 从 model_predictor 导入预测函数 (现在我们直接用这个名字)
    from model_predictor import predict_protein_function
    # 从 protein_utils 导入获取数据的函数
    from protein_utils import fetch_protein_data
    # 我们将在这里实现 search_proteins 的核心逻辑，所以不需要从外部导入
except ImportError as e:
    logging.error(f"无法导入工具实现所需的函数 (predict_protein_function 或 fetch_protein_data): {e}")
    logging.error("请确保 model_predictor.py 和 protein_utils.py 在 Python 路径中，并包含正确的函数。")
    # 定义临时的 placeholder 函数，以便服务器至少能启动
    async def predict_protein_function(sequence: str, organism: str = "") -> Dict[str, Any]: return {"error": "Predict tool implementation not loaded"}
    async def fetch_protein_data(identifier: str) -> Dict[str, Any] | None: return {"error": "Get data tool implementation not loaded"}

# 导入 httpx 用于 UniProt API 调用
import httpx
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [MCP Server] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 创建 FastMCP 实例 ---
mcp = FastMCP("protein_tools_server")

# --- 工具实现 ---

@mcp.tool()
async def predict_protein_function_tool(sequence: str, organism: str = "") -> Dict[str, Any]:
    """
    根据蛋白质的氨基酸序列预测其生物学功能。

    Args:
        sequence: 必需，蛋白质的氨基酸序列。
        organism: 可选，蛋白质来源的物种科学名称。

    Returns:
        包含 'predicted_function', 'confidence', 'model_version', 'processing_time_sec' 的字典，或包含 'error' 的字典。
    """
    logger.info(f"Tool 'predict_protein_function_tool' called with sequence length {len(sequence)}")
    try:
        # 直接调用导入的预测函数
        result = await predict_protein_function(sequence, organism)
        # 假设 predict_protein_function 内部会处理模拟/真实预测并返回字典
        if "error" in result:
             logger.error(f"Prediction failed: {result.get('error')}")
        else:
             logger.info(f"Prediction successful: {result.get('predicted_function')}")
        return result
    except Exception as e:
        logger.exception("Unexpected error in predict_protein_function_tool")
        return {"error": f"Internal server error during prediction: {type(e).__name__}"}

@mcp.tool()
async def get_protein_data(identifier: str) -> Dict[str, Any] | None:
    """
    根据 UniProt 登录号或入口名称获取蛋白质的详细信息，包括序列、物种等。

    Args:
        identifier: 必需，UniProt 登录号 (如 P00533) 或入口名称 (如 INS_HUMAN)。

    Returns:
        包含 'sequence', 'organism', 'id' 等信息的字典，或在未找到时返回包含 'error' 的字典，或在其他错误时也返回错误字典。
    """
    logger.info(f"Tool 'get_protein_data' called with identifier: {identifier}")
    try:
        result = await fetch_protein_data(identifier)
        if result is None:
            # fetch_protein_data 在找不到或其内部出错时返回 None
            logger.warning(f"Identifier '{identifier}' not found or failed to fetch in protein_utils.")
            # 返回明确的错误给 Agent，而不是 None/null
            return {"error": f"Identifier '{identifier}' not found or data could not be retrieved."}
        else:
             logger.info(f"Data fetched for {identifier}: Found ID {result.get('id')}")
             return result # 返回获取到的字典
    except Exception as e:
        logger.exception(f"Unexpected error in get_protein_data tool for identifier {identifier}")
        return {"error": f"Internal server error fetching data for {identifier}: {type(e).__name__}"}

# --- search_proteins 的核心实现 ---
async def _perform_uniprot_search(query: str, species_filter: Optional[str], keyword_filter: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """
    实际执行 UniProt 搜索的内部函数。
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    # 构建查询字符串 - 注意 UniProt 查询语法
    query_parts = [f"({query})"] # 将主查询放入括号，应对可能的复杂查询
    if species_filter:
        # 尝试智能判断是物种名还是分类ID
        if species_filter.isdigit():
            query_parts.append(f"taxonomy_id:{species_filter}")
        else:
            # 对于物种名，可能需要精确匹配或加引号，这里简单处理
            # 注意：UniProt API 对物种名的查询可能需要更精确的字段，如 organism_name:"Homo sapiens"
            query_parts.append(f"organism_name:\"{species_filter}\"") # 尝试加引号
    if keyword_filter:
        query_parts.append(f"keyword:{keyword_filter}")

    full_query = " AND ".join(query_parts)
    # 请求需要的字段
    fields = "accession,id,protein_name,organism_name,length" # protein_name 可能比 id 更易读

    params = {
        "query": full_query,
        "fields": fields,
        "format": "json",
        "size": limit
    }

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client: # 增加超时
        try:
            logger.debug(f"UniProt search query: {params['query']}, fields: {params['fields']}, size: {params['size']}")
            response = await client.get(base_url, params=params)
            response.raise_for_status() # 检查 HTTP 错误 (4xx, 5xx)
            data = response.json()

            results_list = []
            if "results" in data:
                for entry in data["results"]:
                    protein_info = {
                        "id": entry.get("primaryAccession"), # 使用 Accession 作为主要 ID
                        "entry_name": entry.get("uniProtkbId"), # UniProt ID (e.g., EGFR_HUMAN)
                        "name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName",{}).get("value", "N/A"), # 尝试获取推荐全名
                        "organism": entry.get("organism", {}).get("scientificName", "N/A"),
                        "length": entry.get("sequence", {}).get("length", 0)
                    }
                    # 如果推荐名没有，尝试获取提交名
                    if protein_info["name"] == "N/A" and "submissionNames" in entry.get("proteinDescription", {}):
                         submitted_names = entry["proteinDescription"]["submissionNames"]
                         if submitted_names:
                              protein_info["name"] = submitted_names[0].get("fullName", {}).get("value", "N/A")

                    results_list.append(protein_info)
            return results_list

        except httpx.HTTPStatusError as e:
            error_message = f"UniProt API returned status {e.response.status_code}."
            try:
                # 尝试解析错误响应体
                error_data = e.response.json()
                error_detail = error_data.get("messages", [str(e)])
                error_message += f" Details: {error_detail}"
            except ValueError: # JSONDecodeError
                error_message += f" Response body: {e.response.text[:200]}" # 显示部分原始响应
            logger.error(error_message)
            # 将 API 错误包装后重新抛出，由外层捕获
            raise ValueError(f"UniProt search failed: {error_message}") from e
        except httpx.RequestError as e:
            logger.error(f"Network error during UniProt search: {e}")
            raise ValueError(f"Network error contacting UniProt: {e}") from e
        except json.JSONDecodeError as e:
             logger.error(f"Failed to parse UniProt search response: {e}")
             raise ValueError("Failed to parse response from UniProt.") from e
# ---------------------------------------

@mcp.tool()
async def search_proteins(query: str, species_filter: Optional[str] = None, keyword_filter: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]] | Dict[str, Any]:
    """
    搜索 UniProt 数据库中的蛋白质。
    :param query: 搜索关键词
    :param species_filter: 可选的物种过滤条件
    :param keyword_filter: 可选的额外关键词过滤
    :param limit: 返回结果的最大数量
    :return: 包含蛋白质信息的列表
    """
    # 添加紧急调试打印
    print(f"DEBUG: ENTER search_proteins: query='{query}', species='{species_filter}', keyword='{keyword_filter}', limit={limit}", file=sys.stderr, flush=True)
    
    logger.info(f"Tool 'search_proteins' called with query: '{query}', species: {species_filter}, keyword: {keyword_filter}, limit: {limit}")
    try:
        # 添加基本的输入验证和限制
        safe_limit = max(1, min(limit, 50)) # 确保 limit 在 1 到 50 之间
        if not query or not query.strip():
            return {"error": "Search query cannot be empty."}

        # 调用内部实现的搜索函数
        results = await _perform_uniprot_search(query, species_filter, keyword_filter, safe_limit)
        logger.info(f"Search returned {len(results)} results (limit was {safe_limit}).")
        return results

    except ValueError as ve: # 捕获由 _perform_uniprot_search 抛出的特定错误
         logger.error(f"Search failed due to value error: {ve}")
         return {"error": str(ve)} # 将 ValueError 的消息直接作为错误信息返回
    except Exception as e:
        logger.exception("Unexpected error in search_proteins tool")
        return {"error": f"Internal server error during protein search: {type(e).__name__}"}

# --- 启动服务器 ---
if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport...")
    mcp.run(transport='stdio')