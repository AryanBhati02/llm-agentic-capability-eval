"""
Anthropic client — Claude models via the Anthropic API.

Uses the official ``anthropic`` Python SDK for async message creation.
Auth: Anthropic API key via environment variable.
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import ExperimentConfig, ModelConfig

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """LLM client for Anthropic Claude models.

    Uses ``anthropic.AsyncAnthropic`` for async calls.

    Args:
        model_name: Human-readable model name.
        config: ModelConfig (provider must be "anthropic").
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        super().__init__(model_name, config)
        self._client = None

    def _get_client(self):
        """Lazy-initialize the AsyncAnthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            api_key = self.config.resolve_api_key()
            self._client = AsyncAnthropic(
                api_key=api_key,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Call Anthropic's /v1/messages endpoint."""
        client = self._get_client()

        response = await client.messages.create(
            model=self.model_id,
            max_tokens=self.config.max_tokens,
            temperature=experiment_config.temperature,
            top_p=experiment_config.top_p,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._response_to_dict(response)

    def _response_to_dict(self, response) -> dict[str, Any]:
        """Convert an Anthropic Message response to a plain dict."""
        text = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text += getattr(block, "text", "")

        usage = getattr(response, "usage", None)

        return {
            "text": text,
            "model": getattr(response, "model", None),
            "id": getattr(response, "id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            },
        }

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse the normalized Anthropic response dict."""
        usage = raw.get("usage", {})

        return LLMResponse(
            text=raw.get("text", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model_version_reported=raw.get("model"),
            raw_response=raw,
        )

    async def close(self) -> None:
        """Close the AsyncAnthropic client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
