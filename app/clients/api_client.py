import base64
from typing import Optional

from openai import OpenAI

from app.clients.base import BaseLLMClient
from app.config import Config


class ApiLLMClient(BaseLLMClient):

    def __init__(self, config: Config, mode: str = "text"):
        self.client = OpenAI(
            base_url=config.llm.api_base_url,
            api_key=config.llm.api_key or "placeholder",
        )
        self.text_model = config.llm.text_model
        self.vision_model = config.llm.vision_model
        self.mode = mode
        self.max_tokens = config.llm.max_tokens
        self.timeout = config.llm.timeout

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"API 文本调用失败: {e}")

    def vision_chat(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_mime: str = "image/jpeg",
    ) -> str:
        b64 = self._encode_image(image_path, image_bytes)
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"API 视觉调用失败: {e}")

    def _encode_image(self, image_path: Optional[str], image_bytes: Optional[bytes]) -> str:
        if image_bytes:
            return base64.b64encode(image_bytes).decode()
        if image_path:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        raise ValueError("image_path 和 image_bytes 不能同时为空")
