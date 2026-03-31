from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.settings import get_config


class BaseAgent:
    """智能体基类，封装 LLM 初始化等通用能力"""

    def __init__(self, model: str, temperature: float = 0.9) -> None:
        config = get_config()
        self.llm = ChatOpenAI(
            api_key=config.aliyun_api_key,
            base_url=config.aliyun_base_url,
            model=model,
            temperature=temperature,
        )