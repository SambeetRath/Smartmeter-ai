from __future__ import annotations

import importlib
import json
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


def _responses_input(messages: list[ProviderMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {MessageRole.SYSTEM, MessageRole.USER}:
            items.append({"role": message.role.value, "content": message.content})
        elif message.role is MessageRole.ASSISTANT:
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            items.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                }
                for call in message.tool_calls
            )
        else:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
    return items


def _tool_calls(output_items: object) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if not isinstance(output_items, (list, tuple)):
        raise ValueError("Responses output must be a list.")
    for item in output_items:
        if _value(item, "type") != "function_call":
            continue
        arguments_value = _value(item, "arguments")
        if not isinstance(arguments_value, str):
            raise ValueError("Function arguments must be JSON text.")
        arguments = json.loads(arguments_value)
        if not isinstance(arguments, dict):
            raise ValueError("Function arguments must decode to an object.")
        calls.append(
            ToolCall(
                id=_value(item, "call_id"),
                name=_value(item, "name"),
                arguments=arguments,
            )
        )
    return calls


def _normalize_response(raw_response: object) -> ProviderResponse:
    output = _value(raw_response, "output")
    if output is None:
        raise ValueError("Responses output is missing.")

    text_parts: list[str] = []
    refusal_found = False
    for item in output:
        if _value(item, "type") != "message":
            continue
        content = _value(item, "content", [])
        if not isinstance(content, (list, tuple)):
            raise ValueError("Message content must be a list.")
        for part in content:
            part_type = _value(part, "type")
            if part_type == "output_text":
                text = _value(part, "text")
                if not isinstance(text, str):
                    raise ValueError("Output text must be a string.")
                text_parts.append(text)
            elif part_type == "refusal":
                refusal_found = True
                refusal = _value(part, "refusal")
                if isinstance(refusal, str):
                    text_parts.append(refusal)

    calls = _tool_calls(output)
    status = _value(raw_response, "status")
    incomplete = _value(raw_response, "incomplete_details")
    incomplete_reason = _value(incomplete, "reason")
    if calls:
        finish_reason = CompletionReason.TOOL_CALLS
    elif refusal_found:
        finish_reason = CompletionReason.CONTENT_FILTER
    elif status == "incomplete" and incomplete_reason == "content_filter":
        finish_reason = CompletionReason.CONTENT_FILTER
    elif status == "incomplete" and incomplete_reason == "max_output_tokens":
        finish_reason = CompletionReason.LENGTH
    elif status == "completed":
        finish_reason = CompletionReason.STOP
    else:
        finish_reason = CompletionReason.OTHER

    usage = _value(raw_response, "usage")
    return ProviderResponse(
        text="".join(text_parts) or None,
        tool_calls=calls,
        finish_reason=finish_reason,
        model=_value(raw_response, "model"),
        input_tokens=_value(usage, "input_tokens"),
        output_tokens=_value(usage, "output_tokens"),
    )


class OpenAIProvider:
    """Synchronous adapter for the official OpenAI Responses API."""

    def __init__(self, config: AssistantConfig, *, client: Any | None = None) -> None:
        if config.provider != LLMProvider.OPENAI:
            raise ProviderConfigurationError("OpenAI provider requires LLM_PROVIDER=openai.")
        if config.openai_api_key is None or config.openai_model is None:
            raise ProviderConfigurationError(
                "OpenAI provider requires OPENAI_API_KEY and OPENAI_MODEL."
            )
        self._model = config.openai_model
        self._client = client if client is not None else self._create_client(config)

    @staticmethod
    def _create_client(config: AssistantConfig) -> Any:
        try:
            openai = importlib.import_module("openai")
        except ModuleNotFoundError as error:
            if error.name == "openai":
                raise ProviderDependencyError("openai", "assistant-openai") from None
            raise
        return openai.OpenAI(
            api_key=config.openai_api_key.get_secret_value(),
            timeout=config.request_timeout,
            max_retries=0,
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.model != self._model:
            raise ProviderConfigurationError(
                "Provider request model must match the configured OPENAI_MODEL."
            )
        arguments: dict[str, Any] = {
            "model": self._model,
            "input": _responses_input(request.messages),
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.tools:
            arguments["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]

        try:
            raw_response = self._client.responses.create(**arguments)
        except Exception as error:
            raise sanitize_provider_exception(error, "OpenAI") from None

        try:
            normalized = _normalize_response(raw_response)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
            raise ProviderError(
                "malformed_response",
                "OpenAI returned a response that did not match the expected format.",
            ) from None
        allowed_tool_names = {tool.name for tool in request.tools}
        if any(call.name not in allowed_tool_names for call in normalized.tool_calls):
            raise ProviderError(
                "unapproved_tool_call",
                "OpenAI returned a tool call that was not included in the request.",
            )
        return normalized

    def close(self) -> None:
        """Close the SDK client when its owning application is shut down."""
        self._client.close()
