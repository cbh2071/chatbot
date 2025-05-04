# agent_core.py
import logging
import json
import asyncio
from typing import List, Dict, Any, Tuple

import config                   # 导入配置
from llm_clients import get_llm_client, BaseLLMClient # 导入 LLM 客户端工厂和基类
from mcp_clients import get_mcp_client, SimpleMcpStdioClient # 导入 MCP 客户端工厂和实现

logger = logging.getLogger(__name__)

class AgentCore:
    """智能代理核心类，负责处理用户输入、调用LLM和MCP工具、生成回复。"""

    def __init__(self, llm_provider: str = config.DEFAULT_LLM_PROVIDER, mcp_client_instance: SimpleMcpStdioClient | None = None):
        """
        初始化 AgentCore。
        :param llm_provider: 要使用的 LLM 提供商名称 (例如 "openai", "anthropic")。
        :param mcp_client_instance: 外部传入的 MCP 客户端实例 (可选)。如果未提供，则内部创建一个。
        """
        self.llm_client: BaseLLMClient | None = get_llm_client(llm_provider)
        if not self.llm_client:
            raise ValueError(f"无法初始化 LLM 客户端: {llm_provider}")

        self.mcp_client = mcp_client_instance or get_mcp_client()
        # 确保 MCP 服务器在 Agent Core 启动时启动
        if not self.mcp_client.start():
             logger.warning("AgentCore 初始化时启动 MCP 客户端失败，后续工具调用可能失败。")

        self.conversation_history: List[Dict[str, str]] = [] # 存储对话历史 [{ "role": "user/assistant", "content": "..." }]

        logger.info(f"AgentCore 初始化完成，使用 LLM 提供商: {llm_provider}，MCP 客户端: {type(self.mcp_client).__name__}")

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
        :param user_input: 当前用户的输入。
        :return: 包含规划结果的字典 (例如 {"action": "call_tool", "tool_name": "...", "arguments": {...}} 或 {"action": "direct_response"})，
                 如果规划失败则返回 None。
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
        logger.debug(f"规划 Prompt (部分): {system_prompt[:500]}...") # 记录部分 prompt

        # 调用 LLM 生成 JSON 格式的规划
        plan = await self.llm_client.generate_json(
            prompt=f"用户最新请求: {user_input}", # 实际的用户输入放入 main prompt
            system_prompt=system_prompt,
            temperature=0.1, # 规划阶段需要更精确
            max_tokens=512 # 限制输出长度
        )

        if plan and isinstance(plan, dict) and "action" in plan:
            logger.info(f"Agent Core: 规划完成 - 决策: {plan.get('action')}, 工具: {plan.get('tool_name')}, 解释: {plan.get('explanation')}")
            return plan
        else:
            logger.error("Agent Core: LLM 未能生成有效的规划 JSON。")
            return None

    async def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        通过 MCP 客户端调用指定的工具。
        :param tool_name: 要调用的工具名称。
        :param arguments: 传递给工具的参数。
        :return: 工具返回的结果字典，或包含 'error' 的字典。
        """
        if tool_name not in config.AVAILABLE_MCP_TOOLS:
            logger.error(f"Agent Core: 尝试调用未定义的工具 '{tool_name}'。")
            return {"error": f"内部错误：工具 '{tool_name}' 未定义。"}

        logger.info(f"Agent Core: 开始执行工具 '{tool_name}'，参数: {arguments}")
        # **重要:** 在实际调用前，应该根据 AVAILABLE_MCP_TOOLS 中的定义严格验证参数是否存在、是否必需、类型是否匹配。
        # 这里暂时省略了详细的参数验证逻辑。

        try:
            # 调用 MCP 客户端的 call_tool 方法
            # 注意：这里的 timeout 可能需要根据工具的预期执行时间调整
            result = await self.mcp_client.call_tool(method=tool_name, params=arguments, timeout=60.0)
            logger.info(f"Agent Core: 工具 '{tool_name}' 执行完成。")
            logger.debug(f"工具 '{tool_name}' 返回结果: {result}")
            return result if isinstance(result, dict) else {"error": "工具返回了无效的格式。"}
        except Exception as e:
            logger.exception(f"Agent Core: 调用 MCP 工具 '{tool_name}' 时发生意外错误。")
            return {"error": f"调用工具 '{tool_name}' 时发生内部错误: {type(e).__name__}"}

    async def _generate_final_response(self, user_input: str, plan: dict | None, tool_result: dict | None) -> str:
        """
        使用 LLM 生成最终给用户的回复。
        :param user_input: 用户的原始输入。
        :param plan: 规划阶段的结果 (可能为 None)。
        :param tool_result: 工具执行的结果 (如果调用了工具，可能包含错误)。
        :return: 生成的自然语言回复。
        """
        history_str = self._format_history()
        system_prompt = "你是一个友好且专业的生物信息学助手。根据提供的上下文信息，生成一个清晰、准确且自然的回复给用户。"

        context = f"对话历史:\n{history_str}\n\n用户的最新请求:\n{user_input}\n\n"

        if plan:
             context += f"助手规划:\n行动: {plan.get('action')}\n工具: {plan.get('tool_name', '无')}\n参数: {plan.get('arguments', '无')}\n解释: {plan.get('explanation', '无')}\n\n"
        else:
             context += "助手规划阶段失败。\n\n"


        if tool_result:
            context += f"工具执行结果:\n```json\n{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n```\n\n"
        elif plan and plan.get("action") == "call_tool":
             context += "工具调用未执行或失败。\n\n"


        final_prompt = context + "请基于以上所有信息，生成给用户的最终回复。"

        logger.info("Agent Core: 开始生成最终回复...")
        logger.debug(f"最终回复 Prompt (部分): {final_prompt[:500]}...")

        response = await self.llm_client.generate_text(
            prompt=final_prompt, # 将完整上下文作为 prompt
            # system_prompt=system_prompt, # 或者保持 system_prompt 简洁，把指导放入主 prompt
            temperature=0.7, # 回复生成可以更有创造性一点
            max_tokens=1024
        )
        logger.info("Agent Core: 最终回复生成完毕。")
        return response

    async def process_message(self, user_input: str) -> str:
        """
        处理单条用户消息的完整流程。
        :param user_input: 用户输入的文本。
        :return: Agent 生成的回复文本。
        """
        logger.info(f"Agent Core: 收到用户消息: {user_input}")
        # 1. 将用户消息添加到历史记录
        self.conversation_history.append({"role": "user", "content": user_input})

        # 2. 规划阶段
        plan = await self._plan_execution(user_input)

        tool_result = None
        # 3. 执行阶段 (如果需要调用工具)
        if plan and plan.get("action") == "call_tool":
            tool_name = plan.get("tool_name")
            arguments = plan.get("arguments")
            if tool_name and isinstance(arguments, dict):
                tool_result = await self._execute_tool(tool_name, arguments)
            else:
                logger.error("Agent Core: 规划需要调用工具，但工具名称或参数无效。")
                tool_result = {"error": "内部规划错误：无法确定要调用的工具或参数。"}
                # 可以选择让 LLM 基于这个错误生成回复，或者直接返回错误
                # 这里我们让 LLM 尝试处理
        elif not plan:
             # 规划失败的情况
             logger.error("Agent Core: 规划阶段失败，无法确定下一步行动。")
             # 可以生成一个表示内部错误的回复，或者让 LLM 尝试通用回复
             # pass # 让 _generate_final_response 处理 plan is None 的情况

        # 4. 响应生成阶段
        final_response = await self._generate_final_response(user_input, plan, tool_result)

        # 5. 将助手回复添加到历史记录
        self.conversation_history.append({"role": "assistant", "content": final_response})

        # 6. 返回最终回复
        return final_response

    def clear_history(self):
        """清空对话历史。"""
        self.conversation_history = []
        logger.info("Agent Core: 对话历史已清空。")

    def shutdown(self):
         """关闭 Agent Core，停止 MCP 客户端。"""
         logger.info("Agent Core: 正在关闭...")
         self.mcp_client.stop()
         logger.info("Agent Core: 已关闭。")