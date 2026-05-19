import base64
from typing import Optional

import requests

from app.clients.base import BaseLLMClient
from app.config import Config


class OllamaClient(BaseLLMClient):

    def __init__(self, config: Config, use_vision_model: bool = False):
        self.base_url = config.llm.ollama_base_url
        self.text_model = config.llm.text_model
        self.vision_model = config.llm.vision_model if use_vision_model else config.llm.text_model
        self.timeout = config.llm.timeout
        self.max_tokens = config.llm.max_tokens

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.text_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": self.max_tokens},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError("无法连接 Ollama 服务，请确认 ollama serve 已启动（http://localhost:11434）")
        except Exception as e:
            raise RuntimeError(f"Ollama 文本调用失败: {e}")

    def vision_chat(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: str = "image/jpeg",
    ) -> str:
        b64 = self._encode_image(image_path, image_bytes)
        messages = [{"role": "user", "content": text, "images": [b64]}]
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.vision_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": self.max_tokens},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError("无法连接 Ollama 服务")
        except Exception as e:
            raise RuntimeError(f"Ollama 视觉调用失败: {e}")

    def _encode_image(self, image_path: Optional[str], image_bytes: Optional[bytes]) -> str:
        if image_bytes:
            return base64.b64encode(image_bytes).decode()
        if image_path:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        raise ValueError("image_path 和 image_bytes 不能同时为空")
