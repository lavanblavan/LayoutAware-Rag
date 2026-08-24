"""OpenAI-compatible LLM client — Groq (cloud) or Ollama (local)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from settings.Settings import Config

_llm_client = None


@dataclass
class _Message:
    content: str = ""
    reasoning: str = ""


@dataclass
class _Choice:
    message: _Message


@dataclass
class _ChatCompletion:
    choices: list[_Choice]


class _Completions:
    def __init__(self, base_url: str, api_key: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **_: Any,
    ) -> _ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if not response.ok:
            detail = response.text[:300]
            raise requests.HTTPError(
                f"{response.status_code} {response.reason}: {detail}",
                response=response,
            )
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return _ChatCompletion(
            choices=[
                _Choice(
                    message=_Message(
                        content=(message.get("content") or "").strip(),
                        reasoning=(message.get("reasoning") or "").strip(),
                    )
                )
            ]
        )


class _Chat:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.completions = _Completions(base_url, api_key)


class LLMClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.chat = _Chat(base_url, api_key)


# Backward-compatible alias
OllamaClient = LLMClient


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        if Config.LLM_PROVIDER == "groq":
            if not Config.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to settings/.env or set LLM_PROVIDER=ollama."
                )
            _llm_client = LLMClient(Config.GROQ_BASE_URL, Config.GROQ_API_KEY)
        else:
            _llm_client = LLMClient(Config.OLLAMA_BASE_URL)
    return _llm_client


def get_groq_client():
    """Backward-compatible alias."""
    return get_llm_client()
