"""
LLM client hierarchy — unified interface for all providers.

Usage::

    from src.llm import LLMClient

    client = LLMClient(provider="google", model="gemini-2.5-pro")
    response = await client.generate(prompt, experiment_config)
"""

from .base import BaseLLMClient, LLMResponse
from .factory import LLMClient, create_clients_from_config

__all__ = [
    "BaseLLMClient",
    "LLMClient",
    "LLMResponse",
    "create_clients_from_config",
]
