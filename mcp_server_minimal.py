# mcp_server_minimal.py
import logging
from mcp.server.fastmcp import FastMCP
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [MCP Server Minimal] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 创建 FastMCP 实例，不注册任何工具
mcp = FastMCP("minimal_server")

# 添加一个极其简单的 echo 工具用于测试
@mcp.tool()
def echo(message: str) -> str:
    logger.info(f"Minimal echo tool called with: {message}")
    print(f"DEBUG: ENTER minimal echo tool: message='{message}'", file=sys.stderr, flush=True)
    return f"You said: {message}"

if __name__ == "__main__":
    logger.info("Starting Minimal MCP server with stdio transport...")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.exception("Minimal server run failed!")