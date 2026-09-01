"""
Abstract base class for all LLM clients and the LLMResponse data container.

Every concrete provider client inherits from ``BaseLLMClient`` and implements
only ``_call_api()`` and ``_parse_response()``. The base class handles:
  - Timing / latency measurement
  - Retry with exponential backoff
  - Wrapping raw API output into a typed ``LLMResponse``

This is the template-method pattern: ``generate()`` is the public interface,
and it calls the two abstract hooks in sequence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import ExperimentConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Typed container for a single LLM generation result.

    Attributes:
        text: The generated text content.
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        latency_seconds: Wall-clock time for the API call.
        model_version_reported: Model version string returned by the API
            (if the provider reports one; None otherwise).
        raw_response: The full raw API response dict, preserved for
            debugging and provenance but never used by pipeline code.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    model_version_reported: str | None = None
    raw_response: dict = field(default_factory=dict)


class BaseLLMClient(ABC):
    """Abstract base for all LLM provider clients.

    Subclasses must implement:
        - ``_call_api(prompt, experiment_config)`` → raw API response dict
        - ``_parse_response(raw)`` → LLMResponse (without latency, which
          is measured by the base class)

    The base class provides:
        - ``generate(prompt, experiment_config)`` — the only public method,
          with retry/backoff and timing built in.

    Args:
        model_name: Human-readable model name (the key from models.yaml).
        config: The ModelConfig for this model.
    """

    def __init__(self, model_name: str, config: ModelConfig) -> None:
        self.model_name = model_name
        self.config = config
        self.provider = config.provider
        self.model_id = config.model_id

    async def generate(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> LLMResponse:
        """Generate a response for the given prompt.

        Handles retry with exponential backoff on transient failures.
        Measures wall-clock latency. This is the ONLY public method.

        Args:
            prompt: The full prompt string to send.
            experiment_config: Experiment-level parameters (temperature,
                top_p, retry settings). Applied identically to every model.

        Returns:
            An LLMResponse with the generated text and metadata.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        last_exception: Exception | None = None
        max_attempts = experiment_config.retry_count + 1

        for attempt in range(1, max_attempts + 1):
            try:
                start = time.perf_counter()
                raw = await self._call_api(prompt, experiment_config)
                elapsed = time.perf_counter() - start

                response = self._parse_response(raw)
                response = LLMResponse(
                    text=response.text,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_seconds=round(elapsed, 3),
                    model_version_reported=response.model_version_reported,
                    raw_response=response.raw_response,
                )

                logger.info(
                    "Generation complete",
                    extra={
                        "model": self.model_name,
                        "provider": self.provider,
                        "latency_seconds": response.latency_seconds,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
                return response

            except Exception as e:
                last_exception = e
                if attempt < max_attempts:
                    backoff = experiment_config.retry_backoff_base ** attempt
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {backoff:.1f}s...",
                        extra={
                            "model": self.model_name,
                            "provider": self.provider,
                            "error_type": type(e).__name__,
                            "retry_attempt": attempt,
                        },
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"All {max_attempts} attempts failed for "
                        f"{self.model_name}: {e}",
                        extra={
                            "model": self.model_name,
                            "provider": self.provider,
                            "error_type": type(e).__name__,
                        },
                    )

        assert last_exception is not None
        raise last_exception

    @abstractmethod
    async def _call_api(
        self,
        prompt: str,
        experiment_config: ExperimentConfig,
    ) -> dict[str, Any]:
        """Make the actual API call. Implemented by each provider.

        Args:
            prompt: The prompt string.
            experiment_config: Shared inference parameters.

        Returns:
            The raw API response as a dict.
        """
        ...

    @abstractmethod
    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse a raw API response into an LLMResponse.

        The ``latency_seconds`` field can be left as 0.0 — the base class
        overwrites it with its own timing measurement.

        Args:
            raw: The raw response dict from ``_call_api()``.

        Returns:
            A partially-filled LLMResponse (latency added by caller).
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_name={self.model_name!r}, "
            f"provider={self.provider!r}, model_id={self.model_id!r})"
        )
