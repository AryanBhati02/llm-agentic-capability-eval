"""
Google AI Studio client — Gemini models via the free tier.

Uses the official ``google-genai`` SDK for async generation.
Auth: Google AI Studio API key via environment variable.

Rate limits are project-specific and model-specific. Flash is generous;
Pro is heavily rate-limited on the free tier (~tens of RPD).
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import ExperimentConfig, ModelConfig

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """LLM client for Google Gemini models via AI Studio.

    Uses ``google.genai.Client`` with async generation.

    Args:
        model_name: Human-readable model name.
        config: ModelConfig (provider must be "google").
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        super().__init__(model_name, config)
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Google GenAI client."""
        if self._client is None:
            from google import genai

            api_key = self.config.resolve_api_key()
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Call Gemini's generate_content endpoint via the async API.

        Uses ``client.aio.models.generate_content()`` for true async.
        """
        from google.genai import types

        client = self._get_client()

        generation_config = types.GenerateContentConfig(
            temperature=experiment_config.temperature,
            top_p=experiment_config.top_p,
            max_output_tokens=self.config.max_tokens,
        )

        response = await client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=generation_config,
        )

        return self._response_to_dict(response)

    def _response_to_dict(self, response) -> dict[str, Any]:
        """Convert a google-genai response to a plain dict.

        The SDK response object isn't directly JSON-serializable,
        so we extract the fields we need.
        """
        result: dict[str, Any] = {
            "text": "",
            "model_version": getattr(response, "model_version", None),
            "candidates": [],
            "usage_metadata": {},
        }

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                result["text"] = candidate.content.parts[0].text or ""
            result["candidates"] = [
                {"finish_reason": str(getattr(candidate, "finish_reason", ""))}
            ]

        usage = getattr(response, "usage_metadata", None)
        if usage:
            result["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(
                    usage, "candidates_token_count", 0
                ),
                "total_token_count": getattr(usage, "total_token_count", 0),
            }

        return result

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse the normalized Gemini response dict."""
        usage = raw.get("usage_metadata", {})

        return LLMResponse(
            text=raw.get("text", ""),
            input_tokens=usage.get("prompt_token_count", 0),
            output_tokens=usage.get("candidates_token_count", 0),
            model_version_reported=raw.get("model_version"),
            raw_response=raw,
        )

    async def close(self) -> None:
        """Google GenAI client doesn't require explicit cleanup."""
        self._client = None
