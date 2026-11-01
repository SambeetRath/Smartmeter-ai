from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class CompletionReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class ToolCall(BaseModel):
    """Provider-neutral tool call; arguments contain only JSON-compatible data."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, JsonValue]


class ProviderMessage(BaseModel):
    """Normalized conversation message, including explicit tool-call references."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    role: MessageRole
    content: str | None = Field(default=None, max_length=32_000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=32)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_role_content(self) -> ProviderMessage:
        if self.role in {MessageRole.SYSTEM, MessageRole.USER}:
            if self.content is None or not self.content.strip():
                raise ValueError(f"{self.role.value} messages require non-empty content.")
            if self.tool_calls or self.tool_call_id is not None:
                raise ValueError(f"{self.role.value} messages cannot contain tool-call data.")
        elif self.role is MessageRole.ASSISTANT:
            if (self.content is None or not self.content.strip()) and not self.tool_calls:
                raise ValueError("Assistant messages require content or at least one tool call.")
            if self.tool_call_id is not None:
                raise ValueError("Assistant messages cannot contain a tool_call_id.")
        else:
            if self.content is None:
                raise ValueError("Tool messages require content.")
            if self.tool_call_id is None:
                raise ValueError("Tool messages require a tool_call_id.")
            if self.tool_calls:
                raise ValueError("Tool messages cannot contain tool calls.")
        return self


class ToolDefinition(BaseModel):
    """Provider-neutral JSON Schema definition for an approved tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=1_000)
    parameters: dict[str, JsonValue]

    @model_validator(mode="after")
    def require_object_schema(self) -> ToolDefinition:
        if self.parameters.get("type") != "object":
            raise ValueError("Tool parameters must use a JSON Schema object.")
        return self


class ProviderRequest(BaseModel):
    """Bounded, provider-independent request passed to an LLM adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model: str = Field(min_length=1, max_length=128)
    messages: list[ProviderMessage] = Field(min_length=1, max_length=100)
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=32)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=1_024, ge=1, le=8_192)

    @model_validator(mode="after")
    def require_unique_tool_names(self) -> ProviderRequest:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Provider request tool names must be unique.")
        return self


class ProviderResponse(BaseModel):
    """Normalized completion; provider SDK response objects are never retained."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    text: str | None = Field(default=None, max_length=32_000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=32)
    finish_reason: CompletionReason = CompletionReason.OTHER
    model: str | None = Field(default=None, max_length=128)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_response_content(self) -> ProviderResponse:
        if (self.text is None or not self.text.strip()) and not self.tool_calls:
            raise ValueError("Provider response requires text or at least one tool call.")
        return self
