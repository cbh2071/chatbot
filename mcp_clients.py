import logging
import subprocess
import json
import threading
import queue
import sys
import asyncio
import os
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SimpleMcpStdioClient:
    def __init__(self, server_script="mcp_server.py"):
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.response_queue: queue.Queue = queue.Queue(maxsize=100)
        self._lock = threading.Lock()
        self._request_id = 0
        self.is_running = False
        self.stderr_lines = queue.Queue(maxsize=200)

    def _read_output(self, pipe, storage_queue, pipe_name):
        try:
            while self.process and pipe and not pipe.closed:
                line = pipe.readline()
                if not line: break
                decoded_line = line.decode('utf-8', errors='replace').strip()
                if decoded_line:
                    logger.debug(f"MCP Server {pipe_name}: {decoded_line}")
                    try:
                        storage_queue.put_nowait(decoded_line)
                    except queue.Full:
                        if storage_queue is self.stderr_lines:
                            try: storage_queue.get_nowait()
                            except queue.Empty: pass
                            try: storage_queue.put_nowait(decoded_line)
                            except queue.Full: pass
                        else:
                            logger.warning(f"MCP response queue overflow, discarding message: {decoded_line[:100]}...")
        except Exception as e:
            logger.error(f"Error in reader thread for {pipe_name}: {e}")
        finally:
            logger.info(f"Reader thread for {pipe_name} finished.")

    def start(self):
        with self._lock:
            if self.is_running and self.process and self.process.poll() is None:
                logger.info("MCP server process already running.")
                return True

            logger.info(f"Starting MCP server subprocess: {sys.executable} {self.server_script}")
            try:
                python_executable = sys.executable
                self.process = subprocess.Popen(
                    [python_executable, self.server_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    universal_newlines=False
                )
                self.is_running = True
                logger.info(f"MCP server process started (PID: {self.process.pid}).")

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

                self.reader_thread_stdout = threading.Thread(target=self._read_output, args=(self.process.stdout, self.response_queue, "stdout"), daemon=True)
                self.reader_thread_stderr = threading.Thread(target=self._read_output, args=(self.process.stderr, self.stderr_lines, "stderr"), daemon=True)
                self.reader_thread_stdout.start()
                self.reader_thread_stderr.start()

                time.sleep(0.5)
                if self.process.poll() is not None:
                    raise RuntimeError(f"MCP Server process terminated immediately. Exit code: {self.process.poll()}. Check stderr logs.")

                logger.info("MCP server assumed ready.")
                return True

            except Exception as e:
                logger.exception(f"Failed to start MCP server subprocess: {e}")
                self.is_running = False
                self.process = None
                self._log_stderr()
                return False

    def stop(self):
        with self._lock:
            if not self.is_running or not self.process:
                logger.info("MCP server already stopped.")
                return

            logger.info("Stopping MCP server process...")
            self.is_running = False
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("MCP server did not terminate gracefully, killing.")
                        self.process.kill()
                logger.info("MCP server process stopped.")
            except Exception as e:
                logger.exception(f"Error stopping MCP server process: {e}")
            finally:
                self.process = None

    def _log_stderr(self):
        lines = []
        while not self.stderr_lines.empty():
            try:
                lines.append(self.stderr_lines.get_nowait())
            except queue.Empty:
                break
        if lines:
            logger.error("Recent MCP Server stderr output:\n" + "\n".join(lines))

    async def call_tool(self, method: str, params: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        if not self.is_running or not self.process or self.process.poll() is not None:
            logger.error("MCP Server is not running. Attempting to restart...")
            if not self.start():
                return {"error": "MCP Prediction Service is unavailable (failed to start)."}
            await asyncio.sleep(1.0)

        with self._lock:
            self._request_id += 1
            req_id = self._request_id

        request_message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        message_str = json.dumps(request_message) + '\n'
        logger.info(f"Sending request (ID {req_id}) to MCP server: {method}")
        logger.debug(f"Request body: {message_str.strip()}")

        try:
            self.process.stdin.write(message_str.encode('utf-8'))
            self.process.stdin.flush()
        except Exception as e:
            logger.exception(f"Failed to write to MCP server stdin (PID: {self.process.pid if self.process else 'N/A'}). Stopping client.")
            self.stop()
            return {"error": "Failed to communicate with prediction service."}

        start_wait = time.monotonic()
        processed_messages = 0
        while time.monotonic() - start_wait < timeout:
            try:
                response_line = self.response_queue.get(timeout=0.1)
                processed_messages += 1
                try:
                    response_data = json.loads(response_line)
                    if isinstance(response_data, dict) and response_data.get("id") == req_id:
                        logger.info(f"Received response for ID {req_id}.")
                        if "result" in response_data:
                            return response_data["result"]
                        elif "error" in response_data:
                            err_info = response_data["error"]
                            err_msg = err_info.get('message', 'Unknown MCP error')
                            logger.error(f"MCP tool returned error for ID {req_id}: {err_msg}")
                            return {"error": f"Prediction Error: {err_msg}"}
                        else:
                            logger.error(f"Invalid MCP response format for ID {req_id}: {response_data}")
                            return {"error": "Received invalid response format from prediction service."}
                    else:
                        logger.debug(f"Ignoring non-matching/non-response message: {response_line[:100]}...")

                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON line from MCP server stdout: {response_line[:100]}...")
                except Exception as e:
                    logger.exception(f"Error processing message from response queue: {e}")

            except queue.Empty:
                if self.process and self.process.poll() is not None:
                    logger.error(f"MCP server process terminated unexpectedly while waiting for response ID {req_id}. Exit code: {self.process.poll()}")
                    self.stop()
                    self._log_stderr()
                    return {"error": "Prediction service terminated unexpectedly."}
                await asyncio.sleep(0.05)

        logger.error(f"Timeout waiting for response for request ID {req_id} after {timeout}s. Process running: {self.is_running}. Process exit code: {self.process.poll() if self.process else 'N/A'}")
        self._log_stderr()
        return {"error": f"Prediction request timed out after {timeout} seconds."}

def get_mcp_client() -> SimpleMcpStdioClient:
    """工厂函数：创建并返回一个新的 MCP 客户端实例。"""
    return SimpleMcpStdioClient() 