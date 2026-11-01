from __future__ import annotations

import importlib
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from smartmeter.assistant.config import AssistantConfig, LLMProvider
from smartmeter.assistant.models import (
    CompletionReason,
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ToolCall,
)
from smartmeter.assistant.providers.errors import (
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderError,
    sanitize_provider_exception,
)


def _value(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _chat_messages(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        item: dict[str, Any] = {"role": message.role.value, "content": message.content or ""}
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            for call in message.tool_calls:
                tool_names[call.id] = call.name
            item["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        elif message.role is MessageRole.TOOL:
            tool_name = tool_names.get(message.tool_call_id or "")
            if tool_name is not None:
                item["tool_name"] = tool_name
        translated.append(item)
    return translated


def _ollama_tool_calls(raw_calls: object) -> list[ToolCall]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, (list, tuple)):
        raise ValueError("Ollama tool_calls must be a list.")

    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        function = _value(raw_call, "function")
        name = _value(function, "name")
        arguments = _value(function, "arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Ollama function arguments must be an object.")
        call_id = _value(raw_call, "id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"ollama_{uuid.uuid4().hex}"
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def _normalize_response(payload: object, requested_model: str) -> ProviderResponse:
    message = _value(payload, "message")
    if not isinstance(message, Mapping):
        raise ValueError("Ollama response message is missing.")
    if message.get("role") != "assistant":
        raise ValueError("Ollama response role is invalid.")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Ollama response content is invalid.")
    calls = _ollama_tool_calls(message.get("tool_calls"))
    done_reason = _value(payload, "done_reason")
    if calls:
        finish_reason = CompletionReason.TOOL_CALLS
    elif done_reason == "length":
        finish_reason = CompletionReason.LENGTH
    elif done_reason == "stop":
        finish_reason = CompletionReason.STOP
    elif done_reason == "content_filter":
        finish_reason = CompletionReason.CONTENT_FILTER
    else:
        finish_reason = CompletionReason.OTHER

    return ProviderResponse(
        text=content or None,
        tool_calls=calls,
        finish_reason=finish_reason,
        model=_value(payload, "model", requested_model),
        input_tokens=_value(payload, "prompt_eval_count"),
        output_tokens=_value(payload, "eval_count"),
    )


class OllamaProvider:
    """Synchronous adapter for Ollama's HTTP ``/api/chat`` endpoint."""

    def __init__(self, config: AssistantConfig, *, client: Any | None = None) -> None:
        if config.provider != LLMProvider.OLLAMA:
            raise ProviderConfigurationError("Ollama provider requires LLM_PROVIDER=ollama.")
        if config.ollama_model is None:
            raise ProviderConfigurationError("Ollama provider requires OLLAMA_MODEL.")
        self._model = config.ollama_model
        self._url = f"{config.ollama_base_url.rstrip('/')}/api/chat"
        self._client = client if client is not None else self._create_client(config)

    @staticmethod
    def _create_client(config: AssistantConfig) -> Any:
        try:
            httpx = importlib.import_module("httpx")
        except ModuleNotFoundError as error:
            if error.name == "httpx":
                raise ProviderDependencyError("httpx", "assistant-ollama") from None
            raise
        return httpx.Client(timeout=config.request_timeout, follow_redirects=False)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.model != self._model:
            raise ProviderConfigurationError(
                "Provider request model must match the configured OLLAMA_MODEL."
            )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": _chat_messages(request.messages),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        try:
            response = self._client.post(self._url, json=body)
        except Exception as error:
            raise sanitize_provider_exception(error, "Ollama") from None

        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            if status_code == 404:
                raise ProviderError(
                    "model_unavailable",
                    "Ollama endpoint or configured model was not found; verify OLLAMA_BASE_URL "
                    "and OLLAMA_MODEL.",
                )
            if status_code == 401:
                raise ProviderError(
                    "authentication_error",
                    "Ollama rejected the request; verify access to the configured endpoint.",
                )
            if status_code == 429:
                raise ProviderError(
                    "rate_limited",
                    "Ollama rate limit reached; retry the request later.",
                )
            if status_code == 400 and request.tools and _indicates_unsupported_tools(response):
                raise ProviderError(
                    "tool_calling_unsupported",
                    "The configured Ollama model or server does not support tool calling.",
                )
            raise ProviderError("http_error", "Ollama returned an unsuccessful HTTP response.")

        try:
            payload = response.json()
            normalized = _normalize_response(payload, self._model)
        except (TypeError, ValueError, ValidationError):
            raise ProviderError(
                "malformed_response",
                "Ollama returned a response that did not match the expected format.",
            ) from None
        allowed_tool_names = {tool.name for tool in request.tools}
        if any(call.name not in allowed_tool_names for call in normalized.tool_calls):
            raise ProviderError(
                "unapproved_tool_call",
                "Ollama returned a tool call that was not included in the request.",
            )
        return normalized

    def close(self) -> None:
        """Close the HTTP client when its owning application is shut down."""
        self._client.close()


def _indicates_unsupported_tools(response: object) -> bool:
    try:
        error_payload = response.json()
    except Exception:
        return False
    error_text = _value(error_payload, "error", "")
    if not isinstance(error_text, str):
        return False
    normalized = error_text.lower()
    return "tool" in normalized and ("support" in normalized or "unsupported" in normalized)
