"""Provider-independent assistant configuration and contracts."""

from smartmeter.assistant.config import (
    AssistantConfig,
    AssistantConfigurationError,
    AssistantUnavailableError,
    LLMProvider,
    load_assistant_config,
    require_assistant_config,
)
from smartmeter.assistant.models import (
    CompletionReason,
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from smartmeter.assistant.providers.protocol import LLMProviderClient

__all__ = [
    "AssistantConfig",
    "AssistantConfigurationError",
    "AssistantUnavailableError",
    "CompletionReason",
    "LLMProvider",
    "LLMProviderClient",
    "MessageRole",
    "ProviderMessage",
    "ProviderRequest",
    "ProviderResponse",
    "ToolCall",
    "ToolDefinition",
    "load_assistant_config",
    "require_assistant_config",
]
