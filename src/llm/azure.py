"""
Azure AI Foundry client — GPT models via the Responses API.

Uses the ``openai`` Python SDK's Responses API (``client.responses.create``),
which is the modern interface for Azure AI Foundry endpoints:

    https://<resource>.services.ai.azure.com/openai/v1/responses

This is distinct from the classic Azure OpenAI API (``openai.azure.com``)
which uses ``chat.completions.create`` with versioned ``api-version`` query
parameters.  The Foundry Responses API does not require ``api-version``.

Auth: Azure API key via environment variable (AZURE_OPENAI_API_KEY).

Key API differences vs. classic Azure OpenAI:
  - Client: AsyncOpenAI(base_url=...) instead of AsyncAzureOpenAI(azure_endpoint=...)
  - Method: responses.create(input=...) instead of chat.completions.create(messages=...)
  - Response: response.output_text  /  response.usage.{input,output}_tokens
  - No api-version query parameter needed (Foundry v1 is always-latest)
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import ExperimentConfig, ModelConfig

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AzureAIFoundryClient(BaseLLMClient):
    """LLM client for Azure AI Foundry models via the Responses API.

    Targets the ``services.ai.azure.com/openai/v1`` endpoint using the
    standard ``openai.AsyncOpenAI`` client.  The Responses API is the
    recommended interface for GPT-5-series models deployed in AI Foundry.

    Args:
        model_name: Human-readable model name (the key from models.yaml).
        config: ModelConfig (provider must be ``"azure"``).
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        super().__init__(model_name, config)
        self._client = None

    def _get_client(self):
        """Lazy-initialize the AsyncOpenAI client for Azure AI Foundry.

        Uses ``AsyncOpenAI`` (not ``AsyncAzureOpenAI``) because the Foundry
        Responses API endpoint behaves like the standard OpenAI API — it
        expects a ``base_url`` ending in ``/openai/v1`` and does not need
        the ``api-version`` query parameter that ``AsyncAzureOpenAI`` injects.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = self.config.resolve_api_key()
            base_url = self.config.resolve_foundry_base_url()

            if not base_url:
                raise ValueError(
                    "Azure AI Foundry base URL not configured. Set "
                    "AZURE_OPENAI_ENDPOINT in .env (e.g. "
                    "https://<resource>.services.ai.azure.com/openai/v1) "
                    "or 'endpoint' in models.yaml."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Call the Azure AI Foundry Responses API.

        Uses ``client.responses.create()`` with the prompt as the ``input``
        parameter.  The deployment name comes from ``model_id`` in models.yaml
        (which must match the deployment name in Azure AI Foundry exactly).

        The Responses API returns a single ``Response`` object; we normalise
        it to a plain dict so that ``_parse_response`` stays independent of
        the SDK's type hierarchy.
        """
        client = self._get_client()

        deployment = self.model_id

        response = await client.responses.create(
            model=deployment,
            input=prompt,
            temperature=experiment_config.temperature,
            top_p=experiment_config.top_p,
            max_output_tokens=self.config.max_tokens,
        )

        return self._response_to_dict(response)

    def _response_to_dict(self, response) -> dict[str, Any]:
        """Convert an openai Responses API ``Response`` object to a plain dict.

        The ``Response`` object is not JSON-serialisable, so we extract the
        fields we need for ``_parse_response`` and for the stored raw record.
        """
        usage = getattr(response, "usage", None)

        return {
            "text": getattr(response, "output_text", "") or "",
            "model": getattr(response, "model", None),
            "id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            },
        }

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse the normalised Foundry Responses API dict.

        Responses API format (after ``_response_to_dict``):
        {
            "text": "...",
            "model": "gpt-5.4-mini",
            "usage": {"input_tokens": N, "output_tokens": N, ...},
            ...
        }
        """
        usage = raw.get("usage", {})

        return LLMResponse(
            text=raw.get("text", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model_version_reported=raw.get("model"),
            raw_response=raw,
        )

    async def close(self) -> None:
        """Close the underlying client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
