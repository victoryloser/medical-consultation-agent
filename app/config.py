import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings

load_dotenv()


def _interpolate_env(value: str) -> str:
    """替换 ${VAR_NAME} 为环境变量值"""
    def replacer(m):
        var = m.group(1)
        return os.environ.get(var, m.group(0))
    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _interpolate_dict(d):
    if isinstance(d, dict):
        return {k: _interpolate_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_interpolate_dict(i) for i in d]
    if isinstance(d, str):
        return _interpolate_env(d)
    return d


class AppConfig(BaseModel):
    title: str = "多模态医疗辅助问诊 Demo"
    version: str = "0.1.0"
    upload_dir: str = "tmp/uploads"
    max_image_size_mb: int = 10


class LLMConfig(BaseModel):
    text_provider: str = "ollama"
    vision_provider: str = "api"
    text_model: str = "qwen2.5:7b"
    vision_model: str = "qwen-vl-plus"
    ollama_base_url: str = "http://localhost:11434"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    timeout: int = 60
    max_tokens: int = 2048


class RAGConfig(BaseModel):
    collection_name: str = "medical_knowledge"
    persist_dir: str = "vector_store/chroma"
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80
    embedding_model: str = "BAAI/bge-small-zh-v1.5"


class RulesConfig(BaseModel):
    emergency_rules_path: str = "data/emergency_rules.json"
    department_mapping_path: str = "data/department_mapping.json"
    safety_keywords: list[str] = []


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/app.log"


class Config(BaseModel):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    rules: RulesConfig = RulesConfig()
    logging: LoggingConfig = LoggingConfig()


@lru_cache(maxsize=1)
def get_config(config_path: str = "configs/config.yaml") -> Config:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw = _interpolate_dict(raw)
        return Config(**raw)
    return Config()
