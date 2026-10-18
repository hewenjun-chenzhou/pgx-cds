"""
src/llm_client.py
DeepSeek API 统一调用接口
"""
from __future__ import annotations
import os
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取 DeepSeek API 客户端（单例）"""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ DEEPSEEK_API_KEY 未设置\n"
                "请在项目根目录的 .env 文件中添加：DEEPSEEK_API_KEY=sk-xxx"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int = 2000,
    model: str = "deepseek-chat",
) -> str:
    """
    调用 DeepSeek API

    Parameters
    ----------
    messages : list[dict]
        对话消息列表，格式：[{"role": "user", "content": "..."}]
    temperature : float
        温度参数，越低越稳定（0.1-0.3 推荐）
    json_mode : bool
        是否强制 JSON 格式输出
    max_tokens : int
        最大输出 token 数
    model : str
        模型名称，默认 deepseek-chat（DeepSeek-V3）

    Returns
    -------
    str : 模型回复内容
    """
    client = get_client()
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content