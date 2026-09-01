"""
LLMClient factory function.

Preserves the exact usage pattern from the spec::

    client = LLMClient(provider="ollama", model="llama3.2:3b")
    client = LLMClient(provider="azure", model="gpt-5")
    client = LLMClient(provider="google", model="gemini-2.5-pro")
    client = LLMClient(provider="openrouter", model="deepseek/deepseek-r1")
    client = LLMClient(provider="anthropic", model="claude-opus-4-8")
    response = await client.generate(prompt, experiment_config)

This is a function, not a class — it dispatches to the correct concrete
``BaseLLMClient`` subclass based on ``provider``.
"""

from __future__ import annotations

from src.utils.config import ModelConfig, ModelsConfig

from .base import BaseLLMClient


def _get_client_class(provider: str) -> type[BaseLLMClient]:
    """Lazily import and return the client class for a provider.

    This avoids importing heavy SDKs (openai, anthropic, google-genai)
    until they're actually needed.
    """
    if provider == "ollama":
        from .ollama import OllamaClient
        return OllamaClient
    elif provider == "azure":
        from .azure import AzureAIFoundryClient
        return AzureAIFoundryClient
    elif provider == "google":
        from .google import GoogleClient
        return GoogleClient
    elif provider == "openrouter":
        from .openrouter import OpenRouterClient
        return OpenRouterClient
    elif provider == "anthropic":
        from .anthropic import AnthropicClient
        return AnthropicClient
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Must be one of: ollama, azure, google, openrouter, anthropic"
        )


def LLMClient(
    provider: str,
    model: str,
    *,
    model_name: str | None = None,
    config: ModelConfig | None = None,
) -> BaseLLMClient:
    """Factory: create an LLM client for the given provider and model.

    Two usage modes:

    1. **Quick mode** (matches spec's factory pattern)::

           client = LLMClient(provider="google", model="gemini-2.5-pro")

       Uses sensible defaults for all config fields.

    2. **Config mode** (used by the experiment runner)::

           client = LLMClient(
               provider="google",
               model="gemini-2.5-pro",
               model_name="gemini-2.5-pro",
               config=model_config,
           )

       Uses the full ModelConfig from models.yaml.

    Args:
        provider: Provider identifier (ollama/azure/google/openrouter/anthropic).
        model: Model ID as expected by the provider's API.
        model_name: Human-readable model name (defaults to ``model``).
        config: Full ModelConfig. If None, a default is constructed.

    Returns:
        A concrete BaseLLMClient subclass instance.
    """
    if model_name is None:
        model_name = model

    if config is None:
        config = ModelConfig(provider=provider, model_id=model)

    cls = _get_client_class(provider)
    return cls(model_name=model_name, config=config)


def create_clients_from_config(
    models_config: ModelsConfig,
    pilot_mode: bool = False,
) -> dict[str, BaseLLMClient]:
    """Create all LLM clients from a ModelsConfig.

    Args:
        models_config: The loaded models configuration.
        pilot_mode: If True, only create clients for pilot models.

    Returns:
        A dict mapping model_name → client instance.
    """
    active = models_config.get_active_models(pilot_mode)
    clients: dict[str, BaseLLMClient] = {}

    for name, model_cfg in active.items():
        clients[name] = LLMClient(
            provider=model_cfg.provider,
            model=model_cfg.model_id,
            model_name=name,
            config=model_cfg,
        )

    return clients
