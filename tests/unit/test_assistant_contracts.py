from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from pydantic import ValidationError

from smartmeter.assistant.config import (
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


def _tool_definition(name: str = "get_energy_summary") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read an aggregate energy summary.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _provider_request() -> ProviderRequest:
    return ProviderRequest(
        model="test-model",
        messages=[ProviderMessage(role=MessageRole.USER, content="Summarize this period.")],
        tools=[_tool_definition()],
    )


def test_missing_provider_marks_assistant_unavailable():
    assert load_assistant_config({}) is None
    with pytest.raises(AssistantUnavailableError, match="No LLM provider is configured"):
        require_assistant_config({})


def test_valid_openai_configuration_keeps_key_redacted():
    config = load_assistant_config(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-not-for-use",
            "OPENAI_MODEL": "  configured-model  ",
        }
    )

    assert config is not None
    assert config.provider is LLMProvider.OPENAI
    assert config.openai_model == "configured-model"
    assert config.request_timeout == 30
    assert "test-key-not-for-use" not in repr(config)
    assert "test-key-not-for-use" not in str(config)
    assert "test-key-not-for-use" not in repr(config.openai_api_key)


def test_openai_requires_api_key_and_model_when_selected():
    with pytest.raises(AssistantConfigurationError, match="OPENAI_API_KEY is required"):
        load_assistant_config({"LLM_PROVIDER": "openai", "OPENAI_MODEL": "model"})
    with pytest.raises(AssistantConfigurationError, match="OPENAI_MODEL is required"):
        load_assistant_config({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "secret"})


def test_valid_ollama_configuration_uses_local_default_and_normalizes_url():
    config = load_assistant_config(
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "  local-model  ",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434/",
            "LLM_REQUEST_TIMEOUT": "45",
        }
    )

    assert config is not None
    assert config.provider is LLMProvider.OLLAMA
    assert config.ollama_model == "local-model"
    assert config.ollama_base_url == "http://127.0.0.1:11434"
    assert config.request_timeout == 45


def test_ollama_requires_a_model():
    with pytest.raises(AssistantConfigurationError, match="OLLAMA_MODEL is required"):
        load_assistant_config({"LLM_PROVIDER": "ollama"})


def test_invalid_provider_is_rejected_without_echoing_input():
    with pytest.raises(AssistantConfigurationError, match="LLM_PROVIDER") as error:
        load_assistant_config({"LLM_PROVIDER": "provider-secret-value"})
    assert "provider-secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost:11434",
        "http://user:password@localhost:11434",
        "http://localhost:11434?token=secret",
        "http://localhost:11434/#fragment",
        "http://localhost:99999",
        "not-a-url",
    ],
)
def test_invalid_ollama_url_is_rejected(url: str):
    with pytest.raises(AssistantConfigurationError, match="Invalid assistant configuration"):
        load_assistant_config(
            {
                "LLM_PROVIDER": "ollama",
                "OLLAMA_MODEL": "local-model",
                "OLLAMA_BASE_URL": url,
            }
        )


def test_invalid_ollama_url_does_not_echo_credentials():
    secret = "ollama-url-password"
    with pytest.raises(AssistantConfigurationError) as error:
        load_assistant_config(
            {
                "LLM_PROVIDER": "ollama",
                "OLLAMA_MODEL": "local-model",
                "OLLAMA_BASE_URL": f"http://user:{secret}@localhost:11434",
            }
        )

    assert secret not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "timeout",
    ["not-a-number", "nan", "inf", "0", "0.5", "-1", "120.1"],
)
def test_invalid_request_timeout_is_rejected(timeout: str):
    with pytest.raises(AssistantConfigurationError, match="LLM_REQUEST_TIMEOUT"):
        load_assistant_config({"LLM_PROVIDER": "ollama", "LLM_REQUEST_TIMEOUT": timeout})


def test_message_roles_and_tool_call_references_are_validated():
    assistant_message = ProviderMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=[
            ToolCall(
                id="call-1",
                name="get_energy_summary",
                arguments={"meter_identifier": "uci_household_1"},
            )
        ],
    )
    tool_message = ProviderMessage(
        role=MessageRole.TOOL,
        tool_call_id="call-1",
        content='{"reading_count": 10}',
    )

    assert assistant_message.tool_calls[0].arguments["meter_identifier"] == "uci_household_1"
    assert tool_message.tool_call_id == assistant_message.tool_calls[0].id
    with pytest.raises(ValidationError):
        ProviderMessage(role=MessageRole.USER)
    with pytest.raises(ValidationError):
        ProviderMessage(role=MessageRole.TOOL, content="result")
    with pytest.raises(ValidationError):
        ProviderMessage(role=MessageRole.ASSISTANT)


def test_tool_call_rejects_invalid_name_and_non_json_arguments():
    with pytest.raises(ValidationError):
        ToolCall(id="call-1", name="Not A Tool", arguments={})
    with pytest.raises(ValidationError):
        ToolCall(id="call-1", name="get_summary", arguments={"invalid": object()})


def test_provider_request_and_response_validate_bounds_and_normalize_completion():
    request = _provider_request()
    response = ProviderResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                name="get_energy_summary",
                arguments={"meter_identifier": "uci_household_1"},
            )
        ],
        finish_reason=CompletionReason.TOOL_CALLS,
        model="test-model",
        input_tokens=25,
        output_tokens=10,
    )

    assert request.model == response.model
    assert response.finish_reason is CompletionReason.TOOL_CALLS
    with pytest.raises(ValidationError):
        ProviderRequest(model="test-model", messages=[], max_output_tokens=9_000)
    with pytest.raises(ValidationError):
        ProviderResponse(text=None)


def test_provider_protocol_accepts_a_fake_implementation():
    class FakeProvider:
        def complete(self, request: ProviderRequest) -> ProviderResponse:
            assert request.model == "test-model"
            return ProviderResponse(text="Grounded response.", finish_reason=CompletionReason.STOP)

    provider: LLMProviderClient = FakeProvider()
    assert isinstance(provider, LLMProviderClient)
    assert provider.complete(_provider_request()).text == "Grounded response."


def test_shared_assistant_package_imports_without_provider_sdk_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name.split(".", maxsplit=1)[0] in {"openai", "ollama"}:
            raise AssertionError(f"Unexpected provider SDK import: {name}")
        return original_import(name, *args, **kwargs)

    for module_name in tuple(sys.modules):
        if module_name == "smartmeter.assistant" or module_name.startswith("smartmeter.assistant."):
            del sys.modules[module_name]
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    package = importlib.import_module("smartmeter.assistant")

    assert package.load_assistant_config({}) is None
    assert "openai" not in sys.modules
    assert "ollama" not in sys.modules


def test_fastapi_app_creation_does_not_require_assistant_configuration(monkeypatch):
    from smartmeter.api.main import create_app

    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "LLM_REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)

    application = create_app()

    assert application.title == "SmartMeter AI API"
    assert not any(path.startswith("/assistant") for path in application.openapi()["paths"])
