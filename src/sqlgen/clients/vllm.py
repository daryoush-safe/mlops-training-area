from __future__ import annotations

from openai import OpenAI


class VLLMChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self._client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self._model = model

    def complete(
        self, messages: list[dict], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        content = response.choices[0].message.content
        return content or ""
