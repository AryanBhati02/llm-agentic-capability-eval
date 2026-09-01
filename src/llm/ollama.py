"""
Ollama client — local models via Ollama's HTTP API.

Used exclusively for the pilot/smoke-test phase. Free, instant, unlimited.
Never used for research data collection.

Endpoint: ``POST http://localhost:11434/api/generate``
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.utils.config import ExperimentConfig, ModelConfig

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """LLM client for local Ollama models.

    Communicates via Ollama's REST API on localhost. No API key needed.
    Supports all models available in the local Ollama installation.

    Args:
        model_name: Human-readable model name.
        config: ModelConfig (provider must be "ollama").
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        super().__init__(model_name, config)
        self.base_url = config.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Call Ollama's /api/generate endpoint.

        Args:
            prompt: The prompt string.
            experiment_config: Shared inference parameters.

        Returns:
            Raw JSON response from Ollama.
        """
        client = await self._get_client()

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": experiment_config.temperature,
                "top_p": experiment_config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse Ollama's response format.

        Ollama returns:
        {
            "model": "llama3.2:3b",
            "response": "...",
            "total_duration": ...,
            "prompt_eval_count": ...,
            "eval_count": ...,
            ...
        }
        """
        return LLMResponse(
            text=raw.get("response", ""),
            input_tokens=raw.get("prompt_eval_count", 0),
            output_tokens=raw.get("eval_count", 0),
            model_version_reported=raw.get("model", self.model_id),
            raw_response=raw,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
