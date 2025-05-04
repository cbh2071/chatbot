# app.py
import gradio as gr               # 导入Gradio库，用于创建Web UI
import subprocess               # 导入subprocess库，用于创建和管理子进程（MCP Server）
import json                     # 导入json库，用于序列化和反序列化JSON数据（MCP通信）
import threading                # 导入threading库，用于创建后台线程读取子进程输出
import queue                    # 导入queue库，用于线程安全的消息队列
import sys                      # 导入sys库，用于访问Python解释器路径和退出程序
import asyncio                  # 导入asyncio库，用于支持异步操作（Gradio接口函数）
import os                       # 导入os库，用于文件路径操作和获取进程信息
import logging                  # 导入logging库
import time                     # 导入time库，用于计算时间和添加延迟
import re                       # 导入re库 (虽然这里没直接用，但依赖的protein_utils用了)
from typing import Optional, Dict, Any # 导入类型提示

# 为 Gradio 应用设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Gradio App] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 尝试从我们自己的模块导入工具函数
try:
    from protein_utils import fetch_protein_data, validate_sequence, is_potential_uniprot_id
except ImportError:
    logger.error("无法导入 protein_utils 模块。请确保它在同一目录或 PYTHONPATH 中。")
    sys.exit(1) # 导入失败则退出应用

# --- 简化的 MCP Stdio 客户端 ---
# 警告：这个实现对 MCP stdio 协议做了一些假设（例如，基于行的JSON，类似JSON-RPC的结构）。
# 它缺乏生产级客户端所具备的健壮的错误处理、初始化同步和协议细节处理。
# 仅用于演示目的。

class SimpleMcpStdioClient:
    """一个简单的类，用于启动、停止并通过stdio与MCP服务器子进程通信。"""
    def __init__(self, server_script="mcp_server.py"):
        """
        初始化客户端。
        :param server_script: MCP服务器脚本的文件名。
        """
        self.server_script = server_script # MCP服务器脚本路径
        self.process: Optional[subprocess.Popen] = None # 子进程对象，初始为None
        self.reader_thread_stdout: Optional[threading.Thread] = None # 读取stdout的线程
        self.reader_thread_stderr: Optional[threading.Thread] = None # 读取stderr的线程
        # 响应队列：用于存储从服务器stdout读取的、等待被处理的消息行
        self.response_queue: queue.Queue[str] = queue.Queue(maxsize=100)
        # 标准错误队列：用于存储从服务器stderr读取的最近行，主要用于调试
        self.stderr_lines: queue.Queue[str] = queue.Queue(maxsize=200)
        self._lock = threading.Lock() # 线程锁，用于保护共享资源（如进程状态、请求ID）
        self._request_id = 0          # 递增的请求ID，用于匹配请求和响应
        self.is_running = False       # 标记子进程是否应该在运行

    def _read_output(self, pipe, storage_queue, pipe_name):
        """
        后台线程函数，持续读取子进程的某个输出管道（stdout或stderr）。
        :param pipe: 子进程的输出管道 (例如 self.process.stdout)。
        :param storage_queue: 用于存储读取到的行的队列 (例如 self.response_queue)。
        :param pipe_name: 管道名称 ("stdout" 或 "stderr")，用于日志记录。
        """
        try:
            # 循环条件：进程存在、管道有效且未关闭
            while self.process and pipe and not pipe.closed:
                # 逐行读取输出（阻塞操作）
                line = pipe.readline()
                # 如果读到空行，通常表示管道已关闭（进程结束）
                if not line: break
                # 将读取到的字节解码为UTF-8字符串，替换无法解码的字符，并去除首尾空白
                decoded_line = line.decode('utf-8', errors='replace').strip()
                if decoded_line:
                    # 记录读取到的原始行（调试级别）
                    logger.debug(f"MCP 服务器 {pipe_name}: {decoded_line}")
                    try:
                        # 尝试将解码后的行放入对应的存储队列（非阻塞）
                        storage_queue.put_nowait(decoded_line)
                    except queue.Full:
                        # 如果队列已满的处理逻辑
                        if storage_queue is self.stderr_lines:
                            # 对于stderr队列，丢弃最旧的一条消息，再尝试放入新的
                            try: storage_queue.get_nowait()
                            except queue.Empty: pass # 如果在尝试丢弃时队列变空了，忽略
                            try: storage_queue.put_nowait(decoded_line)
                            except queue.Full: pass # 如果还是满的，就放弃这条消息
                        else: # 对于响应队列 (stdout)
                            # 记录警告，表示响应可能丢失
                            logger.warning(f"MCP 响应队列已满，丢弃消息: {decoded_line[:100]}...")

        except Exception as e:
             # 捕获读取过程中可能发生的任何异常
             logger.error(f"读取 {pipe_name} 的线程出错: {e}")
        finally:
             # 线程结束时记录日志
             logger.info(f"读取 {pipe_name} 的线程已结束。")
             # 这里可以添加信号通知主线程或其他逻辑，如果需要的话

    def start(self) -> bool:
        """
        启动MCP服务器子进程和读取线程。
        如果进程已在运行，则不执行任何操作。
        :return: 如果成功启动（或已在运行），返回True；否则返回False。
        """
        with self._lock: # 获取锁，保护进程启动过程
            # 检查进程是否已在运行 (self.is_running为True，进程对象存在，且进程尚未结束)
            if self.is_running and self.process and self.process.poll() is None:
                logger.info("MCP 服务器子进程已在运行。")
                return True

            logger.info(f"开始启动 MCP 服务器子进程: {sys.executable} {self.server_script}")
            try:
                # 获取当前Python解释器的路径
                python_executable = sys.executable
                # 使用subprocess.Popen启动子进程
                self.process = subprocess.Popen(
                    [python_executable, self.server_script], # 要执行的命令和参数
                    stdin=subprocess.PIPE,     # 将标准输入重定向到管道，以便发送数据
                    stdout=subprocess.PIPE,    # 将标准输出重定向到管道，以便读取响应
                    stderr=subprocess.PIPE,    # 将标准错误重定向到管道，以便捕获错误日志
                    bufsize=1,                 # 设置缓冲区大小为1，表示行缓冲
                    cwd=os.path.dirname(os.path.abspath(__file__)), # 设置子进程工作目录为当前脚本所在目录
                    universal_newlines=False   # 设置为False，表示以二进制模式读写管道
                )
                self.is_running = True # 标记进程已启动
                logger.info(f"MCP 服务器子进程已启动 (PID: {self.process.pid})。")

                # 清空上次运行可能残留的消息队列
                while not self.response_queue.empty():
                    try:
                        self.response_queue.get_nowait()
                    except queue.Empty:
                        break
                while not self.stderr_lines.empty():
                    try:
                        self.stderr_lines.get_nowait()
                    except queue.Empty:
                        break

                # 创建并启动读取stdout和stderr的后台线程
                # 设置为守护线程(daemon=True)，这样主程序退出时这些线程也会自动退出
                self.reader_thread_stdout = threading.Thread(target=self._read_output, args=(self.process.stdout, self.response_queue, "stdout"), daemon=True)
                self.reader_thread_stderr = threading.Thread(target=self._read_output, args=(self.process.stderr, self.stderr_lines, "stderr"), daemon=True)
                self.reader_thread_stdout.start()
                self.reader_thread_stderr.start()

                # 稍微等待一下，检查子进程是否立即退出了（表示启动失败）
                time.sleep(0.5)
                if self.process.poll() is not None:
                    # 如果进程已结束，抛出运行时错误
                    raise RuntimeError(f"MCP 服务器子进程立即终止。退出码: {self.process.poll()}。请检查stderr日志。")

                # 简化处理：假设服务器启动后很快就准备好了，没有实现复杂的握手协议
                # 这是一个风险点，实际的MCP协议可能有初始化阶段
                logger.info("假定 MCP 服务器已准备就绪 (未进行显式初始化确认)。")
                return True

            except Exception as e:
                # 捕获启动过程中发生的任何异常
                logger.exception(f"启动 MCP 服务器子进程失败: {e}")
                self.is_running = False # 标记为未运行
                self.process = None     # 清理进程对象
                self._log_stderr()      # 尝试记录stderr中可能存在的错误信息
                return False

    def stop(self):
        """停止MCP服务器子进程和相关线程。"""
        with self._lock: # 获取锁
            # 如果已标记为停止或进程对象不存在，则直接返回
            if not self.is_running or not self.process:
                logger.info("MCP 服务器已经停止。")
                return

            logger.info("正在停止 MCP 服务器子进程...")
            self.is_running = False # 设置标志，通知读取线程可以结束了（虽然它们是守护线程）
            try:
                # 检查进程是否仍在运行
                if self.process.poll() is None:
                    # 尝试优雅地终止进程 (发送SIGTERM)
                    self.process.terminate()
                    try:
                        # 等待最多3秒让进程自行退出
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # 如果超时仍未退出，则强制杀死进程 (发送SIGKILL)
                        logger.warning("MCP 服务器未能优雅终止，强制结束 (killing)。")
                        self.process.kill()
                logger.info("MCP 服务器子进程已停止。")
            except Exception as e:
                # 捕获停止过程中可能发生的错误
                logger.exception(f"停止 MCP 服务器子进程时出错: {e}")
            finally:
                # 清理进程对象
                self.process = None
                # 读取线程是守护线程，会在主线程退出时自动结束，无需显式join

    def _log_stderr(self):
         """将stderr_lines队列中缓存的最近错误日志输出到主应用的日志中。"""
         lines = []
         # 从队列中取出所有缓存的stderr行
         while not self.stderr_lines.empty():
             try: lines.append(self.stderr_lines.get_nowait())
             except queue.Empty: break
         if lines:
             # 如果有错误日志，将其合并并以ERROR级别记录
             logger.error("最近的 MCP 服务器 stderr 输出:\n" + "\n".join(lines))

    async def call_tool(self, method: str, params: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        异步调用MCP服务器上的一个工具方法。
        :param method: 要调用的工具名称 (与 mcp_server.py 中 @mcp.tool 定义的函数名对应)。
        :param params: 调用工具时传递的参数字典。
        :param timeout: 等待响应的超时时间（秒）。
        :return: 从MCP服务器收到的结果字典。如果发生错误或超时，返回包含 'error' 键的字典。
        """
        # 检查服务器进程是否正在运行，如果不在运行，尝试启动它
        if not self.is_running or not self.process or self.process.poll() is not None:
            logger.error("MCP 服务器未运行。尝试重新启动...")
            if not self.start(): # 调用start()方法尝试启动
                 # 如果启动失败，返回错误信息
                 return {"error": "MCP 预测服务不可用 (启动失败)。"}
            # 启动后稍微等待一下，让服务器有时间初始化
            await asyncio.sleep(1.0)

        # 使用锁保护请求ID的递增，确保每个请求ID唯一
        with self._lock:
            self._request_id += 1
            req_id = self._request_id

        # 构建符合JSON-RPC 2.0格式的请求消息体
        # 这是基于对MCP stdio协议的常见假设，实际协议可能不同
        request_message = {
            "jsonrpc": "2.0",
            "id": req_id,         # 请求ID，用于匹配响应
            "method": method,     # 要调用的方法名
            "params": params      # 方法参数
        }
        # 将请求字典序列化为JSON字符串，并在末尾添加换行符（基于行的协议）
        message_str = json.dumps(request_message) + '\n'
        logger.info(f"向 MCP 服务器发送请求 (ID {req_id}): {method}")
        # 记录详细的请求体（调试级别）
        logger.debug(f"请求体: {message_str.strip()}")

        try:
            # 将编码后的消息写入子进程的标准输入管道
            self.process.stdin.write(message_str.encode('utf-8'))
            # 刷新缓冲区，确保消息被立即发送
            self.process.stdin.flush()
        except Exception as e:
            # 如果写入stdin失败（例如管道已损坏），记录错误并停止客户端
            pid_info = self.process.pid if self.process else 'N/A'
            logger.exception(f"写入 MCP 服务器 stdin (PID: {pid_info}) 失败。正在停止客户端。")
            self.stop() # 停止客户端以反映服务不可用状态
            return {"error": "无法与预测服务通信。"}

        # 进入等待响应的循环
        start_wait = time.monotonic() # 记录开始等待的时间（使用单调时钟）
        processed_messages = 0 # 计数处理了多少条来自队列的消息
        while time.monotonic() - start_wait < timeout:
            try:
                # 尝试从响应队列中获取一条消息，设置短暂超时（0.1秒），避免完全阻塞
                response_line = self.response_queue.get(timeout=0.1)
                processed_messages += 1
                try:
                    # 尝试将读取到的行解析为JSON对象
                    response_data = json.loads(response_line)
                    # 检查解析后的数据是否为字典，且包含与请求匹配的 'id'
                    if isinstance(response_data, dict) and response_data.get("id") == req_id:
                        logger.info(f"收到 ID {req_id} 的响应。")
                        # 检查响应是成功结果还是错误
                        if "result" in response_data:
                            # 如果包含 'result' 键，表示成功，返回结果内容
                            return response_data["result"]
                        elif "error" in response_data:
                            # 如果包含 'error' 键，表示服务器返回了错误
                            err_info = response_data["error"]
                            # 提取错误消息
                            err_msg = err_info.get('message', '未知的 MCP 错误') if isinstance(err_info, dict) else str(err_info)
                            logger.error(f"MCP 工具为 ID {req_id} 返回错误: {err_msg}")
                            # 将错误信息包装后返回给调用者
                            return {"error": f"预测错误: {err_msg}"}
                        else:
                            # 如果响应格式不符合预期（既无result也无error）
                            logger.error(f"收到 ID {req_id} 的无效 MCP 响应格式: {response_data}")
                            return {"error": "从预测服务收到无效的响应格式。"}
                    else:
                         # 如果消息ID不匹配或根本不是响应消息（可能是通知等）
                         # 记录日志（调试级别）并忽略这条消息
                         logger.debug(f"忽略不匹配或非响应的消息: {response_line[:100]}...")
                         # 注意：这里简单地丢弃了不匹配的消息。如果MCP协议包含通知，需要额外处理。

                except json.JSONDecodeError:
                    # 如果读取到的行不是有效的JSON
                    logger.warning(f"从 MCP 服务器 stdout 收到非 JSON 行: {response_line[:100]}...")
                except Exception as e:
                     # 处理消息队列中的数据时发生其他错误
                     logger.exception(f"处理响应队列消息时出错: {e}")

            except queue.Empty:
                # 如果在0.1秒内队列为空，表示暂时没有新消息
                # 检查子进程是否在此期间意外终止
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.poll()
                    logger.error(f"在等待响应 ID {req_id} 时，MCP 服务器子进程意外终止。退出码: {exit_code}")
                    self.stop() # 更新客户端状态为停止
                    self._log_stderr() # 记录stderr日志，可能包含崩溃原因
                    return {"error": "预测服务意外终止。"}
                # 如果进程仍在运行且未超时，短暂休眠后继续循环等待
                await asyncio.sleep(0.05) # 使用asyncio.sleep避免阻塞事件循环

        # 如果循环结束仍未收到匹配的响应，表示超时
        proc_status = self.process.poll() if self.process else 'N/A'
        logger.error(f"等待请求 ID {req_id} 的响应超时 ({timeout}秒)。进程运行状态: {self.is_running}。进程退出码: {proc_status}")
        self._log_stderr() # 超时后也记录stderr，可能有助于诊断问题
        return {"error": f"预测请求在 {timeout} 秒后超时。"}


# --- 全局 MCP 客户端实例 ---
# 创建一个 SimpleMcpStdioClient 的全局实例，供 Gradio 应用使用
mcp_client = SimpleMcpStdioClient()

# --- Gradio 界面交互逻辑 ---
async def predict_interface(input_text: str, progress=gr.Progress()) -> str:
    """
    Gradio界面的核心处理函数。接收用户输入，调用后端预测，并格式化输出。
    使用了 Gradio 的 progress 对象来向用户显示处理进度。

    :param input_text: 用户在Gradio文本框中输入的内容。
    :param progress: Gradio提供的进度条对象。
    :return: 一个Markdown格式的字符串，包含处理结果或错误信息，以及处理日志。
    """
    input_text = input_text.strip() # 去除输入的首尾空白
    if not input_text:
        # 如果输入为空，直接返回错误信息
        return "⚠️ **错误:** 请输入 UniProt ID 或蛋白质序列。"

    sequence = None             # 存储最终用于预测的序列
    organism = "未知 / 未指定"  # 存储物种信息
    identifier_used = input_text # 存储用于显示的输入标识符
    status_updates = []         # 存储处理过程中的状态信息，用于生成日志

    # 使用 progress 对象更新进度条和描述信息
    progress(0.1, desc="正在分析输入...")
    status_updates.append("开始分析输入...")
    await asyncio.sleep(0.1) # 短暂等待，让UI有机会更新进度条

    # 1. 判断输入类型（UniProt ID 还是 序列）
    if is_potential_uniprot_id(input_text):
        # 如果输入看起来像 UniProt ID
        status_updates.append(f"输入 “{input_text}” 可能是 UniProt ID。正在尝试从 UniProt 获取数据...")
        logger.info(status_updates[-1])
        progress(0.3, desc=status_updates[-1]) # 更新进度
        # 调用 protein_utils 中的函数异步获取数据
        protein_data = await fetch_protein_data(input_text)
        if protein_data:
            # 如果成功获取数据
            sequence = protein_data["sequence"]     # 获取序列
            organism = protein_data["organism"]     # 获取物种
            identifier_used = protein_data["id"]    # 使用从API获取的规范ID作为标识
            status_updates.append(f"成功获取 {identifier_used} 的数据 (物种: {organism}, 序列长度: {len(sequence)})。")
            logger.info(status_updates[-1])
        else:
            # 如果获取数据失败
            status_updates.append(f"无法获取 UniProt ID “{input_text}” 的数据。将尝试按序列处理...")
            logger.warning(status_updates[-1])
            # 不设置 sequence，让后续逻辑按序列处理 input_text
    else:
         # 如果输入不像 UniProt ID
         status_updates.append("输入不像 UniProt ID。将直接按序列处理...")
         logger.info(status_updates[-1])

    # 2. 验证序列 (如果步骤1未获取到序列 或 获取失败)
    if sequence is None:
        # 如果 sequence 仍然是 None，说明需要将原始输入作为序列进行验证
        if validate_sequence(input_text):
            # 如果原始输入通过了序列验证
            sequence = input_text # 将原始输入赋值给 sequence
            status_updates.append(f"输入已验证为序列 (长度: {len(sequence)})。")
            logger.info(status_updates[-1])
        else:
             # 如果原始输入既不像ID，也未通过序列验证
             final_message = f"⚠️ **错误:** 输入 “{input_text}” 既不是有效的 UniProt ID，也不是有效的蛋白质序列。请检查输入。"
             status_updates.append("输入作为序列验证失败。")
             logger.warning(final_message)
             progress(1.0) # 完成进度条
             # 返回最终错误信息和日志
             return final_message + "\n\n**处理日志:**\n" + "\n".join(status_updates)

    # 3. 调用 MCP 工具进行预测 (前提是已获得有效序列)
    if sequence:
        status_updates.append(f"准备将序列 (长度 {len(sequence)}) 发送到预测服务...")
        logger.info(status_updates[-1])
        progress(0.6, desc="正在通过 MCP 调用预测功能...") # 更新进度

        # 调用全局 mcp_client 实例的 call_tool 方法
        # 传入工具名称、包含序列和物种的参数字典，以及较长的超时时间（例如60秒）
        result = await mcp_client.call_tool(
            method="predict_protein_function_tool", # 与 mcp_server.py 中定义的工具名一致
            params={"sequence": sequence, "organism": organism},
            timeout=60.0 # 设置较长超时，应对可能的长时间模型预测
        )

        progress(0.9, desc="正在处理预测结果...") # 更新进度
        await asyncio.sleep(0.1) # 短暂等待

        # 4. 格式化最终输出
        # 检查返回的 result 是否为字典且不包含 'error' 键
        if isinstance(result, dict) and "error" not in result:
            # 预测成功
            status_updates.append("预测成功完成。")
            logger.info(status_updates[-1])
            # 构建Markdown格式的成功消息
            final_message = f"""
            ✅ **预测成功**

            - **输入标识符:** {identifier_used}
            - **来源物种:** {organism}
            - **序列长度:** {len(sequence)} AA

            ---
            **模型预测结果:**
            - **预测功能:** `{result.get('predicted_function', 'N/A')}`
            - **置信度:** {result.get('confidence', -1.0):.3f}  (值范围通常0-1, N/A表示缺失)
            - **模型版本:** `{result.get('model_version', 'N/A')}`
            - **处理耗时:** {result.get('processing_time_sec', 'N/A')} 秒
            """
            # 对置信度进行格式化，如果不存在则显示 N/A
            conf_val = result.get('confidence')
            conf_str = f"{conf_val:.3f}" if isinstance(conf_val, (int, float)) else 'N/A'
            final_message = final_message.replace(f"{result.get('confidence', -1.0):.3f}", conf_str) # 替换置信度部分

        else:
            # 预测失败或返回错误
            # 提取错误详情
            error_detail = result.get("error", "未知的预测错误") if isinstance(result, dict) else "收到无效响应"
            status_updates.append(f"预测失败: {error_detail}")
            logger.error(status_updates[-1])
            # 构建Markdown格式的错误消息
            final_message = f"""
            ⚠️ **预测过程中发生错误**

            - **输入标识符:** {identifier_used}
            - **错误详情:** {error_detail}
            """
    else:
         # 这个分支理论上不应该被触发，因为前面的逻辑应该确保要么有sequence，要么已经返回错误
         final_message = "⚠️ **内部错误:** 未能获取用于预测的有效序列。"
         status_updates.append(final_message)
         logger.error(final_message)

    progress(1.0) # 完成进度条
    # 返回最终的Markdown消息，并附加上处理日志
    return final_message + "\n\n**处理日志:**\n" + "\n".join(status_updates)


# --- Gradio 应用定义 ---
# 应用标题
title = "🧬 蛋白质功能预测聊天机器人 (基于 MCP 后端)"
# 应用描述，支持Markdown格式
description = """
请输入一个 **UniProt ID** (例如 `P00533`, `INS_HUMAN`) 或直接粘贴 **蛋白质序列**。
系统将通过一个后台运行的 MCP 服务器（使用 stdio 通信）调用机器学习模型来预测蛋白质的功能。
**(注意：当前后端使用的是模拟模型进行演示)**
"""

# 定义输出组件为Markdown，方便格式化文本
output_markdown = gr.Markdown(label="结果与日志", elem_id="result-markdown")

# 自定义CSS样式，用于调整输出区域字体等
css = """
#result-markdown { font-family: monospace; } /* 设置输出区域使用等宽字体 */
#result-markdown code { background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; } /* code标签样式 */
"""

# 创建 Gradio 界面实例
iface = gr.Interface(
    fn=predict_interface, # 指定核心处理函数
    inputs=gr.Textbox(lines=5, label="UniProt ID 或 蛋白质序列", placeholder="输入 UniProt ID (例如 P00533) 或粘贴原始序列 ACDEF..."), # 输入组件：多行文本框
    outputs=output_markdown, # 输出组件：Markdown显示区域
    title=title,             # 设置界面标题
    description=description, # 设置界面描述
    allow_flagging="never", # 禁止Gradio自带的标记反馈功能
    css=css,                 # 应用自定义CSS
    # 提供一些示例输入，方便用户测试
    examples=[
        ["P00533"], # EGFR Human (ID)
        ["INS_HUMAN"], # Insulin Human (Entry Name)
        ["P69905"], # Hemoglobin subunit beta Human (ID)
        ["MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"], # Insulin Sequence (长序列)
        ["InvalidSequenceXYZ"], # 无效序列示例
        ["NonExistentID"] # 无效ID示例
    ]
)

# --- 应用生命周期管理 ---
def startup_event():
    """Gradio 应用启动时执行的函数。"""
    logger.info("Gradio 应用启动: 正在启动 MCP 客户端/服务器...")
    # 调用客户端的start方法启动子进程
    if not mcp_client.start():
         logger.error("严重错误: MCP 客户端在应用启动时启动失败。")
         # 注意：Gradio 可能没有很好的方式在UI层面提示这种启动失败
         # 界面可能仍会加载，但后续的预测调用会失败

def shutdown_event():
    """Gradio 应用关闭时执行的函数。"""
    logger.info("Gradio 应用关闭: 正在停止 MCP 客户端/服务器...")
    # 调用客户端的stop方法停止子进程
    mcp_client.stop()

# 当这个脚本作为主程序运行时 (`python app.py`)
if __name__ == "__main__":
    import atexit # 导入atexit库，用于注册程序退出时执行的函数

    # 在启动Gradio界面之前，手动调用启动函数
    startup_event()

    # 使用atexit注册关闭函数，确保在程序退出（包括Ctrl+C）时尝试停止子进程
    atexit.register(shutdown_event)

    # 启动 Gradio 应用的主循环
    logger.info("正在启动 Gradio 界面...")
    # iface.launch(share=True) # share=True 会创建一个公开链接，方便分享，本地测试时通常不需要
    iface.launch()
    # 程序会阻塞在这里，直到 Gradio 服务器被关闭
    logger.info("Gradio 界面已关闭。")