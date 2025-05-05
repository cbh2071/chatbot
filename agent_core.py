# agent_core.py
import logging
import json
import asyncio
import subprocess # 需要启动子进程
import sys
import os
import contextlib # 用于 AsyncExitStack
from typing import List, Dict, Any, Optional, Tuple

import config                   # 导入配置
from llm_clients import get_llm_client, BaseLLMClient # 导入 LLM 客户端工厂和基类

# 导入官方 MCP Client 相关库
from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class AgentCore:
    """智能代理核心类，使用官方 MCP SDK 与后台工具交互。"""

    def __init__(self, llm_provider: str = config.DEFAULT_LLM_PROVIDER):
        self.llm_client: BaseLLMClient | None = get_llm_client(llm_provider)
        if not self.llm_client:
            raise ValueError(f"无法初始化 LLM 客户端: {llm_provider}")

        # MCP 相关状态
        self.mcp_process: Optional[subprocess.Popen] = None
        self.mcp_session: Optional[ClientSession] = None
        # AsyncExitStack 用于优雅地管理异步上下文资源 (如 stdio_client)
        self.mcp_exit_stack: contextlib.AsyncExitStack = contextlib.AsyncExitStack()
        self.mcp_server_script = config.MCP_SERVER_SCRIPT # 从配置中获取脚本路径

        self.conversation_history: List[Dict[str, str]] = []
        self._mcp_ready = asyncio.Event() # 用于指示 MCP 客户端是否准备就绪

        logger.info(f"AgentCore 初始化，使用 LLM: {llm_provider}，准备启动 MCP 客户端...")

    async def start(self):
        """启动 Agent Core，包括启动和连接 MCP 客户端。"""
        if self.mcp_session and not self.mcp_session.is_closing:
            logger.info("MCP 客户端已在运行。")
            return True

        logger.info("正在启动 MCP 客户端...")
        try:
            server_params = StdioServerParameters(
                command=sys.executable, # 使用当前 python 解释器
                args=[self.mcp_server_script], # 要运行的服务器脚本
                cwd=os.path.dirname(os.path.abspath(__file__)), # 在当前目录运行
            )

            # 使用 AsyncExitStack 管理 stdio_client 的上下文
            streams = await self.mcp_exit_stack.enter_async_context(stdio_client(server_params))

            # 使用获取到的流创建 ClientSession
            self.mcp_session = await self.mcp_exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1]) # streams[0] is read, streams[1] is write
            )

            # 初始化 MCP 连接 (握手)，并捕获返回结果
            init_result = await self.mcp_session.initialize()

            # 从返回结果中获取服务器能力和信息
            server_caps = init_result.capabilities
            server_info = init_result.serverInfo # 也可以获取服务器信息

            logger.info(f"MCP 连接成功。服务器: {server_info.name} v{server_info.version}, 能力: {server_caps}")
            self._mcp_ready.set() # 标记 MCP 已就绪
            return True

        except Exception as e:
            logger.exception("启动 MCP 客户端失败!")
            await self.stop() # 尝试清理
            self._mcp_ready.clear()
            return False

    async def stop(self):
        """停止 Agent Core，关闭 MCP 客户端和子进程。"""
        logger.info("正在停止 MCP 客户端...")
        self._mcp_ready.clear()
        await self.mcp_exit_stack.aclose()
        self.mcp_session = None
        if self.mcp_process and self.mcp_process.poll() is None:
             logger.warning("MCP 进程在 exit stack 清理后仍在运行，尝试强制终止。")
             try:
                 self.mcp_process.terminate()
                 self.mcp_process.wait(timeout=1)
             except:
                 self.mcp_process.kill()
        self.mcp_process = None
        logger.info("MCP 客户端已停止。")

    async def _ensure_mcp_ready(self) -> bool:
         """确保 MCP 客户端已就绪，如果未就绪则尝试启动。"""
         if not self._mcp_ready.is_set():
             logger.warning("MCP 客户端未就绪，尝试启动...")
             if not await self.start():
                 logger.error("无法启动 MCP 客户端。")
                 return False
             await asyncio.sleep(0.5)
         return self._mcp_ready.is_set()

    def _format_history(self) -> str:
        """将对话历史格式化为字符串，供 LLM prompt 使用。"""
        formatted = ""
        for turn in self.conversation_history:
            role = "用户" if turn["role"] == "user" else "助手"
            formatted += f"{role}: {turn['content']}\n"
        return formatted.strip()

    async def _plan_execution(self, user_input: str) -> dict | None:
        """
        使用 LLM 进行规划：分析用户意图，决定是否调用 MCP 工具及所需参数。
        """
        history_str = self._format_history()
        system_prompt = f"""你是一个生物信息学助手。你的任务是理解用户的请求，并决定下一步行动。
        你可以直接回答用户的问题，或者调用可用的工具来获取信息或执行操作。

        以下是你可以使用的 MCP 工具列表 (JSON 格式描述):
        ```json
        {config.AVAILABLE_MCP_TOOLS_JSON}
        ```

        根据当前的对话历史和用户的最新请求，分析用户的意图。
        你的决策应该是以下两种之一：
        1.  **直接回复:** 如果请求是闲聊、问候、简单问题，或者你认为不需要工具就能回答。
        2.  **调用工具:** 如果用户的请求需要通过调用上述某个工具来完成。

        **你的输出必须是严格的 JSON 格式，包含以下字段:**
        - `action`: 字符串，值为 "direct_response" 或 "call_tool"。
        - `tool_name`: 字符串，如果 action 是 "call_tool"，则为要调用的工具名称；否则为 null。
        - `arguments`: 字典，如果 action 是 "call_tool"，则为传递给工具的参数键值对；否则为 null。确保只包含工具定义中存在的参数，并符合类型要求。如果用户没有提供必需参数，你需要向用户提问而不是直接调用。
        - `explanation`: 字符串，简要解释你为什么做出这个决策。

        **对话历史:**
        {history_str}

        **用户最新请求:**
        {user_input}

        请根据用户最新请求进行规划，并以 JSON 格式输出你的决策。"""
        logger.info("Agent Core: 开始规划阶段...")
        logger.debug(f"规划 Prompt (部分): {system_prompt[:500]}...")

        plan = await self.llm_client.generate_json(
            prompt=f"用户最新请求: {user_input}",
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=512
        )

        if plan and isinstance(plan, dict) and "action" in plan:
            logger.info(f"Agent Core: 规划完成 - 决策: {plan.get('action')}, 工具: {plan.get('tool_name')}, 解释: {plan.get('explanation')}")
            return plan
        else:
            logger.error("Agent Core: LLM 未能生成有效的规划 JSON。")
            return None

    async def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        通过官方 MCP ClientSession 调用指定的工具。
        :param tool_name: 要调用的工具名称。
        :param arguments: 传递给工具的参数。
        :return: 工具返回的结果字典，或包含 'error' 的字典。
        """
        if not await self._ensure_mcp_ready():
             return {"error": "MCP 客户端不可用，无法执行工具。"}

        if tool_name not in config.AVAILABLE_MCP_TOOLS:
            logger.error(f"Agent Core: 尝试调用未定义的工具 '{tool_name}'。")
            return {"error": f"内部错误：工具 '{tool_name}' 未定义。"}

        logger.info(f"Agent Core: 开始执行工具 '{tool_name}'，参数: {arguments}")

        # 设置工具特定的超时时间
        tool_timeout = 90.0 if tool_name == "search_proteins" else 30.0
        logger.info(f"为工具 '{tool_name}' 设置超时: {tool_timeout} 秒")

        try:
            # 使用 asyncio.wait_for 包装 call_tool 调用
            result: mcp_types.CallToolResult = await asyncio.wait_for(
                self.mcp_session.call_tool(
                    name=tool_name,
                    arguments=arguments
                ),
                timeout=tool_timeout
            )
            logger.info(f"Agent Core: 工具 '{tool_name}' 执行完成。")

            if result.isError:
                error_content = result.content[0].text if result.content and isinstance(result.content[0], mcp_types.TextContent) else "未知工具错误"
                logger.error(f"MCP 工具 '{tool_name}' 返回错误: {error_content}")
                return {"error": f"工具执行错误: {error_content}"}
            else:
                if result.content and len(result.content) == 1:
                    content_item = result.content[0]
                    if isinstance(content_item, mcp_types.TextContent) and content_item.text:
                        try:
                            parsed_result = json.loads(content_item.text)
                            logger.debug(f"工具 '{tool_name}' 返回结果 (解析后): {parsed_result}")
                            # 确保返回的是 List[Dict] 或 Dict[str, Any]
                            if tool_name == "search_proteins" and not isinstance(parsed_result, list):
                                logger.warning(f"Search tool did not return a list: {type(parsed_result)}")
                                return {"error": "Search tool returned unexpected format."}
                            return parsed_result
                        except json.JSONDecodeError:
                            logger.warning(f"工具 '{tool_name}' 返回了非 JSON 文本，直接使用。")
                            return {"result_text": content_item.text}
                    elif hasattr(content_item, 'model_dump') and callable(content_item.model_dump):
                         logger.debug(f"工具 '{tool_name}' 返回结果 (Pydantic): {content_item.model_dump()}")
                         parsed_result = content_item.model_dump()
                         if tool_name == "search_proteins" and not isinstance(parsed_result, list):
                              logger.warning(f"Search tool did not return a list: {type(parsed_result)}")
                              return {"error": "Search tool returned unexpected format."}
                         return parsed_result
                    else:
                         logger.warning(f"工具 '{tool_name}' 返回了未预期的内容类型或空内容。")
                         return {"warning": "工具返回了非预期格式的内容。"}
                elif tool_name == "search_proteins" and not result.content:
                     logger.info(f"工具 '{tool_name}' 成功执行但未返回任何内容 (可能是搜索无结果)。")
                     return []
                else:
                    logger.warning(f"工具 '{tool_name}' 返回了多个内容项或无内容（非搜索工具）。")
                    return {"success": True, "raw_content": "Complex or empty content"}

        except asyncio.TimeoutError:
             logger.error(f"调用 MCP 工具 '{tool_name}' 超时（超过 {tool_timeout} 秒）。")
             return {"error": f"调用工具 '{tool_name}' 超时 (超过 {tool_timeout} 秒)。请尝试简化查询或稍后再试。"}
        except Exception as e:
            logger.exception(f"Agent Core: 调用 MCP 工具 '{tool_name}' 时发生意外错误。")
            return {"error": f"调用工具 '{tool_name}' 时发生内部错误: {type(e).__name__}"}

    async def _generate_final_response(self, user_input: str, plan: dict | None, tool_result: dict | None) -> str:
        """
        使用 LLM 生成最终给用户的回复。
        """
        history_str = self._format_history()
        system_prompt = "你是一个友好且专业的生物信息学助手。根据提供的上下文信息，生成一个清晰、准确且自然的回复给用户。"

        context = f"对话历史:\n{history_str}\n\n用户的最新请求:\n{user_input}\n\n"

        if plan:
             context += f"助手规划:\n行动: {plan.get('action')}\n工具: {plan.get('tool_name', '无')}\n参数: {plan.get('arguments', '无')}\n解释: {plan.get('explanation', '无')}\n\n"
        else:
             context += "助手规划阶段失败。\n\n"

        if tool_result:
            tool_result_str = json.dumps(tool_result, indent=2, ensure_ascii=False, default=str)
            context += f"工具执行结果:\n```json\n{tool_result_str}\n```\n\n"
        elif plan and plan.get("action") == "call_tool":
             context += "工具调用未执行或失败。\n\n"

        final_prompt = context + "请基于以上所有信息，生成给用户的最终回复。"

        logger.info("Agent Core: 开始生成最终回复...")
        logger.debug(f"最终回复 Prompt (部分): {final_prompt[:500]}...")

        response = await self.llm_client.generate_text(
            prompt=final_prompt,
            temperature=0.7,
            max_tokens=1024
        )
        logger.info("Agent Core: 最终回复生成完毕。")
        return response

    async def process_message(self, user_input: str) -> str:
        """
        处理单条用户消息的完整流程。
        """
        logger.info(f"Agent Core: 收到用户消息: {user_input}")
        if not await self._ensure_mcp_ready():
            return "抱歉，后台服务暂时遇到问题，请稍后再试。"

        self.conversation_history.append({"role": "user", "content": user_input})

        plan = await self._plan_execution(user_input)

        tool_result = None
        if plan and plan.get("action") == "call_tool":
            tool_name = plan.get("tool_name")
            arguments = plan.get("arguments")
            tool_info = config.AVAILABLE_MCP_TOOLS.get(tool_name)
            missing_required = []
            if tool_info and isinstance(tool_info.get("parameters"), dict):
                for param_name, param_details in tool_info["parameters"].items():
                    if param_details.get("required") and param_name not in arguments:
                        missing_required.append(param_name)

            if missing_required:
                 logger.warning(f"规划调用工具 '{tool_name}' 但缺少必需参数: {missing_required}。将要求用户提供。")
                 plan["action"] = "ask_user"
                 plan["missing_params"] = missing_required
                 tool_result = {"error": f"需要更多信息才能执行 '{tool_name}'。缺少参数: {', '.join(missing_required)}"}
            else:
                tool_result = await self._execute_tool(tool_name, arguments)

        elif not plan:
             logger.error("Agent Core: 规划阶段失败，无法确定下一步行动。")
             tool_result = {"error": "无法理解您的请求或制定计划。"}

        final_response = await self._generate_final_response(user_input, plan, tool_result)

        self.conversation_history.append({"role": "assistant", "content": final_response})

        return final_response

    def clear_history(self):
        """清空对话历史。"""
        self.conversation_history = []
        logger.info("Agent Core: 对话历史已清空。")