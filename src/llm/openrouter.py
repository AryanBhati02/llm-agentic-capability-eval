"""
OpenRouter client — DeepSeek, Llama-4, Mistral, Qwen via unified API.

Uses the ``openai`` Python SDK with OpenRouter's base URL:
``https://openrouter.ai/api/v1``

Auth: OpenRouter API key via environment variable.
Supports custom routing and fallbacks as configured in OpenRouter dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import ExperimentConfig, ModelConfig

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(BaseLLMClient):
    """LLM client for OpenRouter models.

    Uses ``openai.AsyncOpenAI`` pointed at OpenRouter's endpoint.
    Covers DeepSeek-R1, Llama-4 Maverick, Qwen3-235B, Mistral Large,
    and any other model on OpenRouter.

    Args:
        model_name: Human-readable model name.
        config: ModelConfig (provider must be "openrouter").
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        super().__init__(model_name, config)
        self.base_url = config.base_url or _OPENROUTER_BASE_URL
        self._client = None

    def _get_client(self):
        """Lazy-initialize the AsyncOpenAI client for OpenRouter."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self.config.resolve_api_key()
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.config.timeout_seconds,
                default_headers={
                    "HTTP-Referer": "https://github.com/Aryann/llm-judge-task-decomposition",
                    "X-Title": "LLM Task Decomposition Research",
                },
            )
        return self._client

    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Call OpenRouter's /chat/completions endpoint."""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=experiment_config.temperature,
            top_p=experiment_config.top_p,
            max_tokens=self.config.max_tokens,
        )

        return response.model_dump()

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI-format chat completion response.

        OpenRouter returns standard OpenAI schema:
        {
            "choices": [{"message": {"content": "..."}}],
            "usage": {"prompt_tokens": N, "completion_tokens": N, ...},
            "model": "deepseek/deepseek-r1",
            ...
        }
        """
        text = ""
        choices = raw.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            text = message.get("content", "") or ""

        usage = raw.get("usage", {})

        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model_version_reported=raw.get("model"),
            raw_response=raw,
        )

    async def close(self) -> None:
        """Close the AsyncOpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
