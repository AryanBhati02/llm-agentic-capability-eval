"""
Test Azure AI Foundry connection and deployment using Responses API.

Usage:
    python scripts/test_azure_foundry.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from src.llm.azure import AzureAIFoundryClient
from src.utils.config import ExperimentConfig, ModelConfig

load_dotenv()


async def main() -> int:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    print(f"AZURE_OPENAI_KEY: {'Set (' + api_key[:4] + '...)' if api_key else 'NOT SET'}")
    print(f"AZURE_OPENAI_ENDPOINT: {endpoint or 'NOT SET'}")

    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY is not set in environment or .env")
        return 1

    config = ModelConfig(
        provider="azure",
        model_id="gpt-5.4-mini",
        max_tokens=256,
        timeout_seconds=30,
        api_key_env="AZURE_OPENAI_API_KEY",
        endpoint=endpoint,
    )

    exp_config = ExperimentConfig(
        temperature=0.2,
        top_p=1.0,
        retry_count=1,
    )

    client = AzureAIFoundryClient(model_name="test-gpt", config=config)

    prompt = "Decompose this task into 3 steps: Bake a chocolate cake from scratch."

    print(f"\nSending test prompt to Azure AI Foundry ({config.model_id})...")
    print(f"Prompt: {prompt}\n")

    try:
        response = await client.generate(prompt, exp_config)
        print("SUCCESS! Response received:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        print(f"Latency: {response.latency_seconds:.2f}s")
        print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")
        return 0
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
