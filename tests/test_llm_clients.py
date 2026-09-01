"""
Mocked tests for the LLM client hierarchy.

No real API calls — all external I/O is mocked. Tests verify:
  - Factory dispatches to the correct client class
  - Each client constructs correct request payloads
  - Response parsing handles normal and edge cases
  - Retry logic works with exponential backoff
  - Error handling after all retries exhausted
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.base import BaseLLMClient, LLMResponse
from src.llm.factory import LLMClient, create_clients_from_config
from src.llm.ollama import OllamaClient
from src.llm.azure import AzureAIFoundryClient
from src.llm.google import GoogleClient
from src.llm.openrouter import OpenRouterClient
from src.llm.anthropic import AnthropicClient
from src.utils.config import ExperimentConfig, ModelConfig, ModelsConfig


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        temperature=0.2,
        top_p=1.0,
        retry_count=2,
        retry_backoff_base=1.0,
        dry_run=False,
        pilot_mode=False,
        randomize_order=False,
        random_seed=42,
    )


@pytest.fixture
def ollama_config() -> ModelConfig:
    return ModelConfig(
        provider="ollama",
        model_id="llama3.2:3b",
        max_tokens=512,
        timeout_seconds=30,
        api_key_env=None,
        concurrency_limit=2,
    )


@pytest.fixture
def google_config() -> ModelConfig:
    return ModelConfig(
        provider="google",
        model_id="gemini-2.5-flash",
        max_tokens=2048,
        timeout_seconds=60,
        api_key_env="GOOGLE_AI_STUDIO_API_KEY",
        concurrency_limit=5,
    )


@pytest.fixture
def openrouter_config() -> ModelConfig:
    return ModelConfig(
        provider="openrouter",
        model_id="deepseek/deepseek-r1",
        max_tokens=2048,
        timeout_seconds=60,
        api_key_env="OPENROUTER_API_KEY",
        concurrency_limit=5,
    )


@pytest.fixture
def azure_config() -> ModelConfig:
    return ModelConfig(
        provider="azure",
        model_id="gpt-5",
        model_snapshot="2026-06-01",
        max_tokens=4096,
        timeout_seconds=60,
        api_key_env="AZURE_OPENAI_API_KEY",
        concurrency_limit=5,
        endpoint="https://test.openai.azure.com/",
    )


@pytest.fixture
def anthropic_config() -> ModelConfig:
    return ModelConfig(
        provider="anthropic",
        model_id="claude-opus-4-8",
        max_tokens=4096,
        timeout_seconds=90,
        api_key_env="ANTHROPIC_API_KEY",
        concurrency_limit=3,
    )


class TestFactory:
    """Tests for the LLMClient factory function."""

    def test_factory_creates_ollama_client(self, ollama_config: ModelConfig):
        client = LLMClient(provider="ollama", model="llama3.2:3b", config=ollama_config)
        assert isinstance(client, OllamaClient)
        assert client.model_id == "llama3.2:3b"
        assert client.provider == "ollama"

    def test_factory_creates_google_client(self, google_config: ModelConfig):
        client = LLMClient(provider="google", model="gemini-2.5-flash", config=google_config)
        assert isinstance(client, GoogleClient)

    def test_factory_creates_openrouter_client(self, openrouter_config: ModelConfig):
        client = LLMClient(
            provider="openrouter", model="deepseek/deepseek-r1", config=openrouter_config
        )
        assert isinstance(client, OpenRouterClient)

    def test_factory_creates_azure_client(self, azure_config: ModelConfig):
        client = LLMClient(provider="azure", model="gpt-5", config=azure_config)
        assert isinstance(client, AzureAIFoundryClient)

    def test_factory_creates_anthropic_client(self, anthropic_config: ModelConfig):
        client = LLMClient(
            provider="anthropic", model="claude-opus-4-8", config=anthropic_config
        )
        assert isinstance(client, AnthropicClient)

    def test_factory_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="nonexistent", model="some-model")

    def test_factory_default_model_name(self, ollama_config: ModelConfig):
        client = LLMClient(provider="ollama", model="llama3.2:3b", config=ollama_config)
        assert client.model_name == "llama3.2:3b"

    def test_factory_custom_model_name(self, ollama_config: ModelConfig):
        client = LLMClient(
            provider="ollama",
            model="llama3.2:3b",
            model_name="my-llama",
            config=ollama_config,
        )
        assert client.model_name == "my-llama"


class TestCreateClientsFromConfig:
    """Tests for batch client creation from ModelsConfig."""

    def test_creates_research_models_only(self):
        config = ModelsConfig(
            models={
                "pilot-model": ModelConfig(
                    provider="ollama", model_id="test:latest"
                ),
                "research-model": ModelConfig(
                    provider="google", model_id="gemini-2.5-flash",
                    api_key_env="GOOGLE_AI_STUDIO_API_KEY",
                ),
            },
            pilot_models=["pilot-model"],
        )
        clients = create_clients_from_config(config, pilot_mode=False)
        assert "research-model" in clients
        assert "pilot-model" not in clients

    def test_creates_pilot_models_only(self):
        config = ModelsConfig(
            models={
                "pilot-model": ModelConfig(
                    provider="ollama", model_id="test:latest"
                ),
                "research-model": ModelConfig(
                    provider="google", model_id="gemini-2.5-flash",
                    api_key_env="GOOGLE_AI_STUDIO_API_KEY",
                ),
            },
            pilot_models=["pilot-model"],
        )
        clients = create_clients_from_config(config, pilot_mode=True)
        assert "pilot-model" in clients
        assert "research-model" not in clients


class TestOllamaClient:
    """Tests for OllamaClient."""

    @pytest.mark.asyncio
    async def test_generate_success(
        self, ollama_config: ModelConfig, experiment_config: ExperimentConfig
    ):
        client = OllamaClient(model_name="test-llama", config=ollama_config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3.2:3b",
            "response": "Step 1: Do this\nStep 2: Do that",
            "prompt_eval_count": 50,
            "eval_count": 30,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.generate("Test prompt", experiment_config)

        assert isinstance(result, LLMResponse)
        assert "Step 1" in result.text
        assert result.input_tokens == 50
        assert result.output_tokens == 30
        assert result.latency_seconds >= 0

    @pytest.mark.asyncio
    async def test_parse_response_empty(self, ollama_config: ModelConfig):
        client = OllamaClient(model_name="test", config=ollama_config)
        result = client._parse_response({})
        assert result.text == ""
        assert result.input_tokens == 0


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    def test_parse_response_standard(self, openrouter_config: ModelConfig):
        client = OpenRouterClient(model_name="test-ds", config=openrouter_config)
        raw = {
            "choices": [{"message": {"content": "Decomposition steps..."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
            "model": "deepseek/deepseek-r1",
        }
        result = client._parse_response(raw)
        assert result.text == "Decomposition steps..."
        assert result.input_tokens == 100
        assert result.output_tokens == 200

    def test_parse_response_empty_choices(self, openrouter_config: ModelConfig):
        client = OpenRouterClient(model_name="test", config=openrouter_config)
        result = client._parse_response({"choices": [], "usage": {}})
        assert result.text == ""


class TestAzureClient:
    """Tests for AzureAIFoundryClient."""

    def test_parse_response_standard(self, azure_config: ModelConfig):
        client = AzureAIFoundryClient(model_name="test-gpt", config=azure_config)
        raw = {
            "choices": [{"message": {"content": "Azure response text"}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 150},
            "model": "gpt-5-2026-06-01",
        }
        result = client._parse_response(raw)
        assert result.text == "Azure response text"
        assert result.input_tokens == 80
        assert result.output_tokens == 150
        assert result.model_version_reported == "gpt-5-2026-06-01"

    def test_parse_response_empty(self, azure_config: ModelConfig):
        client = AzureAIFoundryClient(model_name="test", config=azure_config)
        result = client._parse_response({})
        assert result.text == ""

    def test_uses_snapshot_as_deployment(self, azure_config: ModelConfig):
        client = AzureAIFoundryClient(model_name="test", config=azure_config)
        assert client.config.model_snapshot == "2026-06-01"


class TestGoogleClient:
    """Tests for GoogleClient."""

    def test_parse_response_standard(self, google_config: ModelConfig):
        client = GoogleClient(model_name="test-gemini", config=google_config)
        raw = {
            "text": "Gemini response text",
            "model_version": "gemini-2.5-flash-preview",
            "usage_metadata": {
                "prompt_token_count": 90,
                "candidates_token_count": 120,
                "total_token_count": 210,
            },
        }
        result = client._parse_response(raw)
        assert result.text == "Gemini response text"
        assert result.input_tokens == 90
        assert result.output_tokens == 120

    def test_parse_response_empty(self, google_config: ModelConfig):
        client = GoogleClient(model_name="test", config=google_config)
        result = client._parse_response({})
        assert result.text == ""


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    def test_parse_response_standard(self, anthropic_config: ModelConfig):
        client = AnthropicClient(model_name="test-claude", config=anthropic_config)
        raw = {
            "text": "Claude's thoughtful decomposition",
            "model": "claude-opus-4-8",
            "usage": {"input_tokens": 110, "output_tokens": 250},
        }
        result = client._parse_response(raw)
        assert result.text == "Claude's thoughtful decomposition"
        assert result.input_tokens == 110
        assert result.output_tokens == 250


class TestRetryLogic:
    """Tests for the base class retry mechanism."""

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(
        self, ollama_config: ModelConfig, experiment_config: ExperimentConfig
    ):
        client = OllamaClient(model_name="test", config=ollama_config)

        mock_response_ok = MagicMock()
        mock_response_ok.json.return_value = {
            "response": "success",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_response_ok.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[
                Exception("Connection reset"),
                mock_response_ok,
            ]
        )
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.generate("Test", experiment_config)
        assert result.text == "success"
        assert mock_http.post.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(
        self, ollama_config: ModelConfig, experiment_config: ExperimentConfig
    ):
        client = OllamaClient(model_name="test", config=ollama_config)

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("Persistent failure"))
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(Exception, match="Persistent failure"):
            await client.generate("Test", experiment_config)

        assert mock_http.post.call_count == 3


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_frozen(self):
        resp = LLMResponse(text="hello", input_tokens=10)
        with pytest.raises(AttributeError):
            resp.text = "modified"

    def test_defaults(self):
        resp = LLMResponse(text="test")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.latency_seconds == 0.0
        assert resp.model_version_reported is None
        assert resp.raw_response == {}
