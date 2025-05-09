import asyncio
from mcp import ClientSession, stdio_client, StdioServerParameters

async def test_call_tool():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],

    )
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            init_result = await session.initialize()
            print(f"MCP mcp_session {session}")
            server_caps = init_result.capabilities
            server_info = init_result.serverInfo
            print(f"MCP 连接成功。服务器: {server_info.name} v{server_info.version}, 能力: {server_caps}")
            result = await session.call_tool(
                name="get_protein_data",
                arguments={"identifier": "P00533"}
            )
            print("工具返回：", result)

if __name__ == "__main__":
    asyncio.run(test_call_tool())