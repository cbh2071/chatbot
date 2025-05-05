# app_with_agent.py
import gradio as gr
import logging
import asyncio
# import atexit # atexit 是同步的，对于异步清理可能不够用
from typing import List, Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("GradioAppWithBlocks")

# 导入 Agent Core
try:
    from agent_core import AgentCore
    import config # 需要配置来初始化AgentCore
except ImportError as e:
    logger.error(f"无法导入必要的模块: {e}")
    exit()

# --- 全局 Agent 实例 (应用级别单例) ---
try:
     # AgentCore 的初始化现在是同步的，但启动是异步的
     agent = AgentCore(llm_provider=config.DEFAULT_LLM_PROVIDER)
except ValueError as e:
     logger.error(f"初始化 AgentCore 失败: {e}")
     exit()

# --- Gradio 交互逻辑 ---
def convert_agent_history_to_gradio(agent_history: list) -> List[Tuple[Optional[str], Optional[str]]]:
    """将 Agent 的历史格式转换为 Gradio Chatbot 的格式"""
    gradio_history = []
    user_msg = None
    for turn in agent_history:
        if turn["role"] == "user":
            if user_msg is not None:
                 gradio_history.append((user_msg, None)) # 添加一个没有回复的用户消息
            user_msg = turn["content"]
        elif turn["role"] == "assistant":
            gradio_history.append((user_msg, turn["content"]))
            user_msg = None # 重置 user_msg
    # 如果最后一轮是用户消息，也添加，让其显示在聊天框中等待回复
    # 注意：这可能会导致用户看到自己的消息出现了两次，一次是输入时，一次是作为历史。需要权衡。
    # 通常的处理方式是不添加最后的用户消息到 history 输出，它会通过输入框清空来表示已处理。
    # 这里我们先不加最后的用户消息。
    return gradio_history

async def process_chat(message: str, history_list: List[Tuple[Optional[str], Optional[str]]]) -> Tuple[str, List[Tuple[Optional[str], Optional[str]]]]:
    """处理用户输入的核心函数"""
    print("-" * 20)
    print(f"DEBUG: process_chat - Received message: '{message}'")
    print(f"DEBUG: process_chat - Received history_list: {history_list}")
    print(f"DEBUG: process_chat - Current agent history len: {len(agent.conversation_history)}")
    print("-" * 20)

    logger.info(f"Gradio 收到消息: {message}")
    response = await agent.process_message(message)
    logger.info(f"Agent 回复: {response[:100]}...")

    updated_gradio_history = convert_agent_history_to_gradio(agent.conversation_history)
    print(f"DEBUG: process_chat - Returning updated history: {updated_gradio_history}")

    return "", updated_gradio_history

def clear_history_globally() -> List[Tuple[Optional[str], Optional[str]]]:
    """清空全局 agent 的历史记录"""
    global agent
    agent.clear_history()
    logger.info("聊天历史已通过自定义按钮清空 (全局 Agent)。")
    return []

# --- Gradio 应用定义 ---
title = "🧬 智能蛋白质功能预测助手 (LLM+MCP - Blocks - 单例 Agent)"
description = """
与智能助手对话来查询蛋白质信息、预测功能。助手由大型语言模型驱动，并通过 MCP 调用后端工具。
**示例:**
- `你好`
- `获取 P00533 的数据`
- `预测一下 P00533 的功能`
- `EGFR_HUMAN 是什么？`
- `帮我找找人类的酪氨酸激酶`
"""

with gr.Blocks(title=title, theme=gr.themes.Default()) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    chatbot = gr.Chatbot(label="对话窗口", height=600, scale=2)

    with gr.Row():
        # 输入文本框
        msg_textbox = gr.Textbox(
            label="输入消息", placeholder="请输入您的问题或指令...",
            show_label=False, container=False, scale=4
        )
        submit_btn = gr.Button("发送", scale=1)

    # 清空历史按钮
    clear_btn = gr.Button("清除对话历史")

    # 定义事件处理

    # 处理消息提交 (文本框回车 或 点击发送按钮)
    submit_action = msg_textbox.submit if hasattr(msg_textbox, 'submit') else submit_btn.click
    submit_action(
        fn=process_chat,
        inputs=[msg_textbox, chatbot],
        outputs=[msg_textbox, chatbot]
    )
    if submit_action != submit_btn.click:
        submit_btn.click(
            fn=process_chat,
            inputs=[msg_textbox, chatbot],
            outputs=[msg_textbox, chatbot]
        )

    # 处理清空按钮点击
    clear_btn.click(
        fn=clear_history_globally,
        inputs=[],
        outputs=[chatbot]
    )

# --- 应用生命周期 ---
async def startup_event():
    """应用启动时异步启动 Agent."""
    logger.info("Gradio 应用启动，正在异步启动 AgentCore MCP 客户端...")
    if not await agent.start():
         logger.error("严重错误: AgentCore 在应用启动时启动 MCP 失败。")
         # 可在 UI 中提示用户服务可能不可用

async def shutdown_event():
    """应用关闭时异步停止 Agent."""
    logger.info("Gradio 应用关闭，正在异步停止 AgentCore...")
    await agent.stop()

if __name__ == "__main__":
    async def main():
        await startup_event()
        try:
            await demo.launch(share=False)
        finally:
            await shutdown_event()

    logger.info("正在启动 Gradio Blocks 聊天界面...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，准备关闭...")
    logger.info("Gradio 界面已关闭。")