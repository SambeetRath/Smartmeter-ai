from __future__ import annotations

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from smartmeter.assistant.config import (
    AssistantUnavailableError,
    load_assistant_config,
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
from smartmeter.assistant.providers.errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderError,
)
from smartmeter.assistant.providers.factory import create_provider
from smartmeter.assistant.providers.ollama_adapter import OllamaProvider
from smartmeter.assistant.providers.openai_adapter import OpenAIProvider
from smartmeter.assistant.providers.protocol import LLMProviderClient

OPENAI_ENV = {
    "LLM_PROVIDER": "openai",
    "OPENAI_API_KEY": "test-key-not-for-use",
    "OPENAI_MODEL": "test-openai-model",
}
OLLAMA_ENV = {
    "LLM_PROVIDER": "ollama",
    "OLLAMA_MODEL": "test-ollama-model",
}
TOOL = ToolDefinition(
    name="get_energy_summary",
    description="Read an aggregate energy summary.",
    parameters={
        "type": "object",
        "properties": {"meter_identifier": {"type": "string"}},
        "required": ["meter_identifier"],
        "additionalProperties": False,
    },
)


def _config(environ: dict[str, str]):
    config = load_assistant_config(environ)
    assert config is not None
    return config


def _request(
    model: str,
    messages: list[ProviderMessage] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        messages=messages
        or [ProviderMessage(role=MessageRole.USER, content="Summarize the period.")],
        tools=tools if tools is not None else [TOOL],
        temperature=0.4,
        max_output_tokens=350,
    )


class FakeOpenAIClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.responses = SimpleNamespace(create=self._create)
        self.response = response
        self.error = error
        self.arguments: dict[str, object] | None = None
        self.closed = False

    def _create(self, **arguments: object) -> object:
        self.arguments = arguments
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeHTTPClient:
    def __init__(
        self,
        response: FakeHTTPResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.url: str | None = None
        self.body: dict[str, object] | None = None
        self.closed = False

    def post(self, url: str, *, json: dict[str, object]) -> FakeHTTPResponse:
        self.url = url
        self.body = json
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


def test_openai_responses_api_maps_text_and_bounded_request_parameters():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "model": "test-openai-model",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Usage increased."}],
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
    )
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    response = provider.complete(_request("test-openai-model"))

    assert isinstance(response, ProviderResponse)
    assert response.text == "Usage increased."
    assert response.finish_reason is CompletionReason.STOP
    assert response.input_tokens == 15
    assert response.output_tokens == 8
    assert client.arguments is not None
    assert client.arguments["model"] == "test-openai-model"
    assert client.arguments["temperature"] == 0.4
    assert client.arguments["max_output_tokens"] == 350
    assert client.arguments["tools"] == [
        {
            "type": "function",
            "name": TOOL.name,
            "description": TOOL.description,
            "parameters": TOOL.parameters,
        }
    ]


def test_openai_responses_api_preserves_multiple_function_call_ids():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-openai-1",
                    "name": "get_energy_summary",
                    "arguments": '{"meter_identifier":"meter-a"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call-openai-2",
                    "name": "get_energy_summary",
                    "arguments": '{"meter_identifier":"meter-b"}',
                },
            ],
        }
    )
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    response = provider.complete(_request("test-openai-model"))

    assert response.finish_reason is CompletionReason.TOOL_CALLS
    assert [call.id for call in response.tool_calls] == ["call-openai-1", "call-openai-2"]
    assert [call.arguments["meter_identifier"] for call in response.tool_calls] == [
        "meter-a",
        "meter-b",
    ]


def test_openai_rejects_function_calls_not_in_the_request_allowlist():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-unapproved",
                    "name": "delete_meter",
                    "arguments": "{}",
                }
            ],
        }
    )
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    with pytest.raises(ProviderError) as error:
        provider.complete(_request("test-openai-model"))

    assert error.value.code == "unapproved_tool_call"


def test_openai_conversation_translation_preserves_function_output_correlation():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }
    )
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)
    messages = [
        ProviderMessage(role=MessageRole.SYSTEM, content="Be precise."),
        ProviderMessage(role=MessageRole.USER, content="Read the summary."),
        ProviderMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id="call-preserved",
                    name="get_energy_summary",
                    arguments={"meter_identifier": "meter-a"},
                )
            ],
        ),
        ProviderMessage(
            role=MessageRole.TOOL,
            tool_call_id="call-preserved",
            content='{"reading_count":12}',
        ),
    ]

    provider.complete(_request("test-openai-model", messages))

    assert client.arguments is not None
    assert client.arguments["input"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Read the summary."},
        {
            "type": "function_call",
            "call_id": "call-preserved",
            "name": "get_energy_summary",
            "arguments": '{"meter_identifier":"meter-a"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-preserved",
            "output": '{"reading_count":12}',
        },
    ]


@pytest.mark.parametrize(
    ("exception_name", "expected_code"),
    [
        ("AuthenticationError", "authentication_error"),
        ("RateLimitError", "rate_limited"),
        ("APITimeoutError", "timeout"),
        ("APIConnectionError", "connection_error"),
        ("APIStatusError", "provider_error"),
    ],
)
def test_openai_errors_are_sanitized(exception_name: str, expected_code: str):
    error_type = type(exception_name, (RuntimeError,), {})
    secret_error = error_type("private-key-from-provider-error")
    client = FakeOpenAIClient(error=secret_error)
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    with pytest.raises(ProviderError) as error:
        provider.complete(_request("test-openai-model"))

    assert error.value.code == expected_code
    assert "private-key-from-provider-error" not in str(error.value)


@pytest.mark.parametrize(
    "output",
    [
        None,
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "get_energy_summary",
                "arguments": "[]",
            }
        ],
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "get_energy_summary",
                "arguments": "secret",
            }
        ],
        [{"type": "message", "content": [{"type": "output_text", "text": 4}]}],
    ],
)
def test_openai_malformed_responses_are_safe(output: object):
    client = FakeOpenAIClient({"status": "completed", "output": output})
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    with pytest.raises(ProviderError) as error:
        provider.complete(_request("test-openai-model"))

    assert error.value.code == "malformed_response"
    assert "secret" not in str(error.value)


def test_openai_refusal_is_normalized_as_content_filter():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that request."}],
                }
            ],
        }
    )

    response = OpenAIProvider(_config(OPENAI_ENV), client=client).complete(
        _request("test-openai-model")
    )

    assert response.finish_reason is CompletionReason.CONTENT_FILTER
    assert response.text == "I cannot help with that request."


def test_openai_incomplete_content_filter_reason_is_normalized():
    client = FakeOpenAIClient(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Filtered output."}],
                }
            ],
        }
    )

    response = OpenAIProvider(_config(OPENAI_ENV), client=client).complete(
        _request("test-openai-model")
    )

    assert response.finish_reason is CompletionReason.CONTENT_FILTER


def test_openai_requires_the_configured_model_and_closes_client():
    client = FakeOpenAIClient(
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        }
    )
    provider = OpenAIProvider(_config(OPENAI_ENV), client=client)

    with pytest.raises(ProviderConfigurationError, match="OPENAI_MODEL"):
        provider.complete(_request("caller-selected-model"))
    provider.close()

    assert client.arguments is None
    assert client.closed


def test_openai_client_creation_uses_secret_timeout_and_disables_retries(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSDK:
        @staticmethod
        def OpenAI(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        "smartmeter.assistant.providers.openai_adapter.importlib.import_module",
        lambda name: FakeSDK if name == "openai" else importlib.import_module(name),
    )
    env = {**OPENAI_ENV, "LLM_REQUEST_TIMEOUT": "45"}
    provider = OpenAIProvider(_config(env))

    assert isinstance(provider, OpenAIProvider)
    assert captured == {
        "api_key": "test-key-not-for-use",
        "timeout": 45,
        "max_retries": 0,
    }


def test_ollama_chat_maps_text_and_uses_configured_endpoint_and_bounds():
    client = FakeHTTPClient(
        FakeHTTPResponse(
            200,
            {
                "model": "test-ollama-model",
                "message": {"role": "assistant", "content": "Usage increased."},
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 9,
            },
        )
    )
    config = _config({**OLLAMA_ENV, "OLLAMA_BASE_URL": "http://127.0.0.1:11434/prefix"})
    provider = OllamaProvider(config, client=client)

    response = provider.complete(_request("test-ollama-model"))

    assert response.text == "Usage increased."
    assert response.finish_reason is CompletionReason.STOP
    assert response.input_tokens == 20
    assert response.output_tokens == 9
    assert client.url == "http://127.0.0.1:11434/prefix/api/chat"
    assert client.body is not None
    assert client.body["model"] == "test-ollama-model"
    assert client.body["stream"] is False
    assert client.body["options"] == {"temperature": 0.4, "num_predict": 350}
    assert client.body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": TOOL.name,
                "description": TOOL.description,
                "parameters": TOOL.parameters,
            },
        }
    ]


def test_ollama_tool_calls_get_safe_ids_and_support_multiple_calls():
    client = FakeHTTPClient(
        FakeHTTPResponse(
            200,
            {
                "model": "test-ollama-model",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_energy_summary",
                                "arguments": {"meter_identifier": "meter-a"},
                            }
                        },
                        {
                            "id": "ollama-provider-id",
                            "function": {
                                "name": "get_energy_summary",
                                "arguments": {"meter_identifier": "meter-b"},
                            },
                        },
                    ],
                },
                "done_reason": "stop",
            },
        )
    )
    response = OllamaProvider(_config(OLLAMA_ENV), client=client).complete(
        _request("test-ollama-model")
    )

    assert response.finish_reason is CompletionReason.TOOL_CALLS
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].id.startswith("ollama_")
    assert response.tool_calls[1].id == "ollama-provider-id"
    assert response.tool_calls[0].arguments["meter_identifier"] == "meter-a"


def test_ollama_rejects_tool_calls_not_in_the_request_allowlist():
    client = FakeHTTPClient(
        FakeHTTPResponse(
            200,
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "execute_sql",
                                "arguments": {"query": "SELECT 1"},
                            }
                        }
                    ],
                }
            },
        )
    )

    with pytest.raises(ProviderError) as error:
        OllamaProvider(_config(OLLAMA_ENV), client=client).complete(_request("test-ollama-model"))

    assert error.value.code == "unapproved_tool_call"


def test_ollama_request_maps_assistant_calls_and_correlated_tool_results():
    client = FakeHTTPClient(
        FakeHTTPResponse(
            200,
            {
                "message": {"role": "assistant", "content": "Complete."},
                "done_reason": "stop",
            },
        )
    )
    messages = [
        ProviderMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id="ollama-local-call-id",
                    name="get_energy_summary",
                    arguments={"meter_identifier": "meter-a"},
                )
            ],
        ),
        ProviderMessage(
            role=MessageRole.TOOL,
            tool_call_id="ollama-local-call-id",
            content='{"reading_count":10}',
        ),
    ]

    OllamaProvider(_config(OLLAMA_ENV), client=client).complete(
        _request("test-ollama-model", messages)
    )

    assert client.body is not None
    assert client.body["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_energy_summary",
                        "arguments": {"meter_identifier": "meter-a"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"reading_count":10}',
            "tool_name": "get_energy_summary",
        },
    ]


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_code"),
    [
        (401, {"error": "secret auth response"}, "authentication_error"),
        (404, {"error": "private model path"}, "model_unavailable"),
        (429, {"error": "secret rate data"}, "rate_limited"),
        (503, {"error": "internal host"}, "http_error"),
        (
            400,
            {"error": "This model does not support tools"},
            "tool_calling_unsupported",
        ),
    ],
)
def test_ollama_http_errors_are_safe(status_code: int, payload: object, expected_code: str):
    client = FakeHTTPClient(FakeHTTPResponse(status_code, payload))
    provider = OllamaProvider(_config(OLLAMA_ENV), client=client)

    with pytest.raises(ProviderError) as error:
        provider.complete(_request("test-ollama-model"))

    assert error.value.code == expected_code
    assert "secret" not in str(error.value)
    assert "private model path" not in str(error.value)
    assert "internal host" not in str(error.value)


def test_ollama_unsupported_tools_is_only_detected_when_tools_were_sent():
    client = FakeHTTPClient(FakeHTTPResponse(400, {"error": "The model does not support tools."}))
    provider = OllamaProvider(_config(OLLAMA_ENV), client=client)
    request = _request("test-ollama-model", tools=[])

    with pytest.raises(ProviderError) as error:
        provider.complete(request)

    assert error.value.code == "http_error"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"message": {"role": "user", "content": "wrong role"}},
        {"message": {"role": "assistant", "content": 10}},
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_energy_summary",
                            "arguments": ["not", "an", "object"],
                        }
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "not a valid tool name",
                            "arguments": {},
                        }
                    }
                ],
            }
        },
    ],
)
def test_ollama_malformed_responses_are_safe(payload: object):
    client = FakeHTTPClient(FakeHTTPResponse(200, payload))
    provider = OllamaProvider(_config(OLLAMA_ENV), client=client)

    with pytest.raises(ProviderError) as error:
        provider.complete(_request("test-ollama-model"))

    assert error.value.code == "malformed_response"
    assert "not a valid tool name" not in str(error.value)


def test_ollama_invalid_json_is_reported_as_malformed_response():
    client = FakeHTTPClient(FakeHTTPResponse(200, ValueError("private response body")))

    with pytest.raises(ProviderError) as error:
        OllamaProvider(_config(OLLAMA_ENV), client=client).complete(_request("test-ollama-model"))

    assert error.value.code == "malformed_response"
    assert "private response body" not in str(error.value)


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        (type("ReadTimeout", (TimeoutError,), {}), "timeout"),
        (type("ConnectError", (ConnectionError,), {}), "connection_error"),
    ],
)
def test_ollama_transport_errors_are_sanitized(
    error_type: type[Exception],
    expected_code: str,
):
    client = FakeHTTPClient(error=error_type("secret endpoint details"))

    with pytest.raises(ProviderError) as error:
        OllamaProvider(_config(OLLAMA_ENV), client=client).complete(_request("test-ollama-model"))

    assert error.value.code == expected_code
    assert "secret endpoint details" not in str(error.value)


def test_ollama_requires_the_configured_model_and_closes_client():
    client = FakeHTTPClient(FakeHTTPResponse(200, {}))
    provider = OllamaProvider(_config(OLLAMA_ENV), client=client)

    with pytest.raises(ProviderConfigurationError, match="OLLAMA_MODEL"):
        provider.complete(_request("caller-selected-model"))
    provider.close()

    assert client.url is None
    assert client.closed


def test_ollama_http_client_creation_uses_configured_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHTTPX:
        @staticmethod
        def Client(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        "smartmeter.assistant.providers.ollama_adapter.importlib.import_module",
        lambda name: FakeHTTPX if name == "httpx" else importlib.import_module(name),
    )
    config = _config({**OLLAMA_ENV, "LLM_REQUEST_TIMEOUT": "45"})
    provider = OllamaProvider(config)

    assert isinstance(provider, OllamaProvider)
    assert captured == {"timeout": 45, "follow_redirects": False}


@pytest.mark.parametrize(
    ("module_name", "provider_type", "config", "extra"),
    [
        ("openai", OpenAIProvider, _config(OPENAI_ENV), "assistant-openai"),
        ("httpx", OllamaProvider, _config(OLLAMA_ENV), "assistant-ollama"),
    ],
)
def test_missing_optional_provider_dependency_has_install_guidance(
    monkeypatch,
    module_name: str,
    provider_type: type[OpenAIProvider] | type[OllamaProvider],
    config,
    extra: str,
):
    original_import = importlib.import_module

    def missing_import(name: str):
        if name == module_name:
            raise ModuleNotFoundError(f"No module named '{module_name}'", name=module_name)
        return original_import(name)

    monkeypatch.setattr(
        f"{provider_type.__module__}.importlib.import_module",
        missing_import,
    )

    with pytest.raises(ProviderDependencyError) as error:
        provider_type(config)

    assert extra in str(error.value)
    assert module_name in str(error.value)


def test_factory_selects_only_the_explicitly_configured_provider(monkeypatch):
    openai_client = FakeOpenAIClient()
    ollama_client = FakeHTTPClient()
    original_import = importlib.import_module

    def fake_import(name: str):
        if name == "openai":
            return SimpleNamespace(OpenAI=lambda **_kwargs: openai_client)
        if name == "httpx":
            return SimpleNamespace(Client=lambda **_kwargs: ollama_client)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    openai_provider = create_provider(_config(OPENAI_ENV))
    ollama_provider = create_provider(_config(OLLAMA_ENV))

    assert isinstance(openai_provider, LLMProviderClient)
    assert isinstance(ollama_provider, LLMProviderClient)
    assert type(openai_provider).__name__ == "OpenAIProvider"
    assert type(ollama_provider).__name__ == "OllamaProvider"
    openai_provider.close()
    ollama_provider.close()
    assert openai_client.closed
    assert ollama_client.closed


def test_factory_requires_configuration_and_never_falls_back(monkeypatch):
    for name in (*OPENAI_ENV, *OLLAMA_ENV, "LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AssistantUnavailableError):
        create_provider()

    original_import = importlib.import_module
    requested: list[str] = []

    def missing_openai(name: str):
        requested.append(name)
        if name == "openai":
            raise ModuleNotFoundError("missing openai", name="openai")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_openai)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-for-use")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-model")
    with pytest.raises(ProviderDependencyError) as error:
        create_provider()

    assert error.value.extra == "assistant-openai"
    assert requested == ["openai"]
    assert not any(name == "httpx" for name in requested)


def test_factory_rejects_an_unsupported_provider_without_fallback():
    config = load_assistant_config(OPENAI_ENV)
    assert config is not None
    invalid_config = config.model_copy(update={"provider": "other"})

    with pytest.raises(ProviderConfigurationError, match="not supported"):
        create_provider(invalid_config)


def test_provider_construction_rejects_wrong_provider_configuration():
    with pytest.raises(ProviderConfigurationError):
        OpenAIProvider(_config(OLLAMA_ENV), client=FakeOpenAIClient())
    with pytest.raises(ProviderConfigurationError):
        OllamaProvider(_config(OPENAI_ENV), client=FakeHTTPClient())


def test_provider_adapters_do_not_import_optional_packages_at_module_import():
    script = """
import builtins
import importlib
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'openai', 'ollama', 'httpx'}:
        raise AssertionError(f'unexpected optional import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
for module in (
    'smartmeter.assistant',
    'smartmeter.assistant.providers.openai_adapter',
    'smartmeter.assistant.providers.ollama_adapter',
    'smartmeter.assistant.providers.factory',
):
    importlib.import_module(module)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
