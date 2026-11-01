from __future__ import annotations

from smartmeter.assistant.config import (
    AssistantConfig,
    LLMProvider,
    require_assistant_config,
)
from smartmeter.assistant.providers.errors import ProviderConfigurationError
from smartmeter.assistant.providers.protocol import LLMProviderClient


def create_provider(config: AssistantConfig | None = None) -> LLMProviderClient:
    """Create only the explicitly configured provider; never fall back to another."""
    selected = config if config is not None else require_assistant_config()
    if selected.provider == LLMProvider.OPENAI:
        from smartmeter.assistant.providers.openai_adapter import OpenAIProvider

        return OpenAIProvider(selected)
    if selected.provider == LLMProvider.OLLAMA:
        from smartmeter.assistant.providers.ollama_adapter import OllamaProvider

        return OllamaProvider(selected)
    raise ProviderConfigurationError("The configured LLM provider is not supported.")
