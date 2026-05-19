from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """文本对话"""
        ...

    @abstractmethod
    def vision_chat(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: str = "image/jpeg",
    ) -> str:
        """多模态对话，image_path 和 image_bytes 二选一"""
        ...
