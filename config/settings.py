from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # 阿里百炼平台配置
    aliyun_api_key: str = ""
    aliyun_base_url: str = ""
    aliyun_default_model: str = ""

    # 各智能体独立模型配置
    generator_model_1: str = ""
    generator_model_2: str = ""
    generator_model_3: str = ""
    reviewer_model: str = ""
    selector_model: str = ""

    # Tavily 搜索工具
    tavily_api_key: str = ""

    # 全局运行参数
    max_retry_count: int = 15
    target_candidate_count: int = 3
    memory_storage_path: Path = Path("./memory/history")

    def __post_init__(self) -> None:
        self.aliyun_api_key = os.getenv("ALIYUN_API_KEY", "")
        self.aliyun_base_url = os.getenv(
            "ALIYUN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.aliyun_default_model = os.getenv("ALIYUN_DEFAULT_MODEL", "qwen-plus")
        self.generator_model_1 = os.getenv("GENERATOR_MODEL_1", self.aliyun_default_model)
        self.generator_model_2 = os.getenv("GENERATOR_MODEL_2", self.aliyun_default_model)
        self.generator_model_3 = os.getenv("GENERATOR_MODEL_3", self.aliyun_default_model)
        self.reviewer_model = os.getenv("REVIEWER_MODEL", self.aliyun_default_model)
        self.selector_model = os.getenv("SELECTOR_MODEL", self.aliyun_default_model)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.max_retry_count = int(os.getenv("MAX_RETRY_COUNT", "15"))
        self.target_candidate_count = int(os.getenv("TARGET_CANDIDATE_COUNT", "3"))
        self.memory_storage_path = Path(
            os.getenv("MEMORY_STORAGE_PATH", "./memory/history")
        )
        self._validate()
        self.memory_storage_path.mkdir(parents=True, exist_ok=True)

    def _validate(self) -> None:
        missing = []
        if not self.aliyun_api_key:
            missing.append("ALIYUN_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise EnvironmentError(
                f"缺少必要的环境变量，请在 .env 文件中配置：{', '.join(missing)}"
            )


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config