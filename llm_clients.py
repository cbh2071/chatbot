# llm_clients.py
import os
import logging
import json
from abc import ABC, abstractmethod
import asyncio

# 导入 OpenAI SDK
try:
    from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    APIError = RateLimitError = APIConnectionError = Exception

# 导入其他 SDK
try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    Anthropic = None
    AsyncAnthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import config

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    def __init__(self, api_key: str | None = None, default_model: str | None = None, base_url: str | None = None):
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = base_url
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, model: str | None = None, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
        pass

    async def generate_json(self, prompt: str, model: str | None = None, system_prompt: str | None = None, temperature: float = 0.2, max_tokens: int = 1024, retries: int = 2, **kwargs) -> dict | None:
        json_prompt = prompt + "\n\n请严格按照 JSON 格式返回结果，不要包含任何解释性文字或代码块标记。"
        response_text = ""
        for attempt in range(retries + 1):
            try:
                response_text = await self.generate_text(
                    prompt=json_prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                if response_text.startswith("错误："):
                    raise Exception(f"LLM API call failed: {response_text}")

                cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                json_output = json.loads(cleaned_text)
                return json_output
            except json.JSONDecodeError as e:
                logger.warning(f"LLM 返回的不是有效的 JSON (尝试 {attempt+1}/{retries+1}): {e}\n原始文本: {response_text[:500]}...")
                if attempt == retries:
                    return None
            except Exception as e:
                logger.exception(f"调用 LLM API 或处理 JSON 时发生错误 (尝试 {attempt+1}/{retries+1}): {e}")
                if attempt == retries:
                    return None
            if attempt < retries:
                await asyncio.sleep(1)
        return None

class CompatibleOpenAIClient(BaseLLMClient):
    def _initialize_client(self):
        if not AsyncOpenAI:
            logger.error("OpenAI SDK (>=1.0) 未安装。请运行 'pip install --upgrade openai'")
            return None
        if not self.api_key:
            logger.warning(f"当前服务商 ({self.base_url}) 的 API 密钥未在配置中提供。")

        try:
            logger.info(f"初始化 OpenAI 兼容客户端: base_url='{self.base_url}', 使用 API Key: {'是' if self.api_key else '否'}")
            return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            logger.exception(f"初始化 OpenAI 兼容客户端失败 (base_url={self.base_url}): {e}")
            return None

    async def generate_text(self, prompt: str, model: str | None = None, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
        if not self.client:
            return f"错误：OpenAI 兼容客户端 (URL: {self.base_url}) 未初始化。"
        target_model = model or self.default_model or "gpt-3.5-turbo"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"向 {self.base_url} 发送请求: model={target_model}, messages={messages}")
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            result = response.choices[0].message.content
            logger.debug(f"从 {self.base_url} 收到响应: {result[:100]}...")
            return result or ""
        except APIError as e:
            logger.exception(f"调用 OpenAI 兼容 API ({self.base_url}) 时发生 API 错误: {e.status_code} - {e.message}")
            return f"错误：API 调用失败 ({e.status_code}) - {e.message}"
        except RateLimitError as e:
            logger.exception(f"调用 OpenAI 兼容 API ({self.base_url}) 时触发速率限制。")
            return "错误：请求过于频繁，请稍后再试。"
        except APIConnectionError as e:
            logger.exception(f"无法连接到 OpenAI 兼容 API ({self.base_url})。")
            return f"错误：无法连接到服务 ({self.base_url})。"
        except Exception as e:
            logger.exception(f"调用 OpenAI 兼容 API ({self.base_url}) 时发生未知错误: {e}")
            return f"错误：调用 API 时发生未知错误 ({type(e).__name__})"

class AnthropicClient(BaseLLMClient):
    def _initialize_client(self):
        if not AsyncAnthropic:
            logger.error("Anthropic SDK 未安装。请运行 'pip install anthropic'")
            return None
        if not self.api_key:
            logger.error("未找到 Anthropic API 密钥 (ANTHROPIC_API_KEY 环境变量)。")
            return None
        try:
            return AsyncAnthropic(api_key=self.api_key)
        except Exception as e:
            logger.exception(f"初始化 Anthropic 客户端失败: {e}")
            return None

    async def generate_text(self, prompt: str, model: str | None = None, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
        if not self.client:
            return "错误：Anthropic 客户端未初始化。"
        target_model = model or self.default_model or "claude-3-haiku-20240307"

        try:
            logger.debug(f"向 Anthropic 发送请求: model={target_model}, system='{system_prompt}', prompt='{prompt[:100]}...'")
            response = await self.client.messages.create(
                model=target_model,
                system=system_prompt if system_prompt else " ",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                result = response.content[0].text
            else:
                result = ""
                logger.warning("Anthropic 响应格式不符合预期，未找到文本内容。")

            logger.debug(f"收到 Anthropic 响应: {result[:100]}...")
            return result
        except Exception as e:
            logger.exception(f"调用 Anthropic API 失败: {e}")
            return f"错误：调用 Anthropic API 失败 ({type(e).__name__})"

class GoogleClient(BaseLLMClient):
    def _initialize_client(self):
        if not genai:
            logger.error("Google Generative AI SDK 未安装。请运行 'pip install google-generativeai'")
            return None
        if not self.api_key:
            logger.error("未找到 Google API 密钥 (GOOGLE_API_KEY 环境变量)。")
            return None
        try:
            genai.configure(api_key=self.api_key)
            return True
        except Exception as e:
            logger.exception(f"配置 Google Generative AI 失败: {e}")
            return None

    async def generate_text(self, prompt: str, model: str | None = None, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
        if not self.client:
            return "错误：Google 客户端未配置。"
        target_model_name = model or self.default_model or 'gemini-pro'
        try:
            model_instance = genai.GenerativeModel(target_model_name)
        except Exception as e:
            logger.exception(f"无法获取 Google Gemini 模型实例 '{target_model_name}': {e}")
            return f"错误：无法获取 Google 模型 '{target_model_name}'"

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        try:
            logger.debug(f"向 Google Gemini 发送请求: model={target_model_name}, prompt='{full_prompt[:100]}...'")
            response = await model_instance.generate_content_async(
                full_prompt,
                generation_config=generation_config,
            )
            result = ""
            if hasattr(response, 'text'):
                result = response.text
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logger.warning(f"Google Gemini 请求被阻止: {block_reason}")
                result = f"错误：请求被内容安全策略阻止 ({block_reason})。"
            else:
                logger.warning("Google Gemini 响应格式不符合预期，未找到文本内容。")
                result = "错误：未能从 Google Gemini 获取有效响应。"

            logger.debug(f"收到 Google Gemini 响应: {result[:100]}...")
            return result
        except Exception as e:
            logger.exception(f"调用 Google Gemini API 失败: {e}")
            error_message = str(e)
            return f"错误：调用 Google Gemini API 失败 ({type(e).__name__}): {error_message}"

def get_llm_client(provider_name: str = config.DEFAULT_LLM_PROVIDER) -> BaseLLMClient | None:
    provider = provider_name.lower()
    api_key = None
    base_url = None
    default_model = None

    if provider == "openai":
        api_key = config.OPENAI_API_KEY
        base_url = None
        default_model = config.DEFAULT_LLM_MODEL or "gpt-3.5-turbo"
        return CompatibleOpenAIClient(api_key=api_key, default_model=default_model, base_url=base_url)

    elif provider == "aihubmix":
        api_key = config.AIHUBMIX_API_KEY
        base_url = "https://aihubmix.com/v1"
        default_model = config.DEFAULT_LLM_MODEL or "gpt-4o-mini"
        return CompatibleOpenAIClient(api_key=api_key, default_model=default_model, base_url=base_url)

    elif provider == "deepseek":
        api_key = config.DEEPSEEK_API_KEY
        base_url = "https://api.deepseek.com/v1"
        default_model = config.DEFAULT_LLM_MODEL or "deepseek-chat"
        return CompatibleOpenAIClient(api_key=api_key, default_model=default_model, base_url=base_url)

    elif provider == "ark" or provider == "volcengine":
        api_key = config.ARK_API_KEY
        base_url = config.ARK_BASE_URL or "https://ark.cn-beijing.volces.com/api/v3/"
        default_model = config.DEFAULT_LLM_MODEL or "<Your-ARK-Model-ID>"
        if default_model == "<Your-ARK-Model-ID>":
            logger.error("请在配置中为火山方舟 (ARK) 提供有效的模型 ID 或 Endpoint ID。")
            return None
        return CompatibleOpenAIClient(api_key=api_key, default_model=default_model, base_url=base_url)

    elif provider == "anthropic":
        return AnthropicClient(api_key=config.ANTHROPIC_API_KEY, default_model=config.DEFAULT_LLM_MODEL)

    elif provider == "google":
        return GoogleClient(api_key=config.GOOGLE_API_KEY, default_model=config.DEFAULT_LLM_MODEL)

    else:
        logger.error(f"不支持的 LLM 提供商: {provider_name}")
        return None