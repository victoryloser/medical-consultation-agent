from functools import lru_cache

from app.clients.base import BaseLLMClient
from app.clients.ollama_client import OllamaClient
from app.clients.api_client import ApiLLMClient
from app.config import Config, get_config


def get_text_client(config: Config | None = None) -> BaseLLMClient:
    cfg = config or get_config()
    if cfg.llm.text_provider == "ollama":
        return OllamaClient(cfg, use_vision_model=False)
    return ApiLLMClient(cfg, mode="text")


def get_vision_client(config: Config | None = None) -> BaseLLMClient:
    cfg = config or get_config()
    if cfg.llm.vision_provider == "ollama":
        return OllamaClient(cfg, use_vision_model=True)
    return ApiLLMClient(cfg, mode="vision")
