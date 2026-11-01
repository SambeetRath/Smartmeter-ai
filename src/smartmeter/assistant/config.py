from __future__ import annotations

import math
import os
from collections.abc import Mapping
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator


class AssistantConfigurationError(ValueError):
    """Raised when an explicitly selected assistant provider is misconfigured."""


class AssistantUnavailableError(RuntimeError):
    """Raised when assistant functionality is requested without a configured provider."""


class LLMProvider(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_REQUEST_TIMEOUT = 30.0
MIN_LLM_REQUEST_TIMEOUT = 1.0
MAX_LLM_REQUEST_TIMEOUT = 120.0


class AssistantConfig(BaseModel):
    """Validated provider settings; secrets use Pydantic's redacted SecretStr representation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    provider: LLMProvider
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str | None = None
    request_timeout: float = Field(
        default=DEFAULT_LLM_REQUEST_TIMEOUT,
        ge=MIN_LLM_REQUEST_TIMEOUT,
        le=MAX_LLM_REQUEST_TIMEOUT,
    )

    @field_validator("openai_model", "ollama_model")
    @classmethod
    def normalize_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Model name must not be empty.")
        if len(normalized) > 128:
            raise ValueError("Model name must not exceed 128 characters.")
        return normalized

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        candidate = value.strip()
        try:
            parsed = urlsplit(candidate)
            port = parsed.port
        except ValueError as error:
            raise ValueError("OLLAMA_BASE_URL must be a valid HTTP(S) URL.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must be an HTTP(S) URL without credentials, "
                "query parameters, or a fragment."
            )
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _parse_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError:
        raise AssistantConfigurationError(
            "LLM_REQUEST_TIMEOUT must be between 1 and 120 seconds."
        ) from None
    if (
        not math.isfinite(timeout)
        or not MIN_LLM_REQUEST_TIMEOUT <= timeout <= MAX_LLM_REQUEST_TIMEOUT
    ):
        raise AssistantConfigurationError("LLM_REQUEST_TIMEOUT must be between 1 and 120 seconds.")
    return timeout


def load_assistant_config(
    environ: Mapping[str, str] | None = None,
) -> AssistantConfig | None:
    """Load explicitly selected provider settings, or return None when none is selected."""
    variables = os.environ if environ is None else environ
    provider_value = variables.get("LLM_PROVIDER", "").strip().lower()
    if not provider_value:
        return None
    try:
        provider = LLMProvider(provider_value)
    except ValueError:
        raise AssistantConfigurationError(
            "LLM_PROVIDER must be either 'openai' or 'ollama'."
        ) from None

    timeout = _parse_timeout(
        variables.get("LLM_REQUEST_TIMEOUT", str(DEFAULT_LLM_REQUEST_TIMEOUT)).strip()
    )
    values: dict[str, object] = {
        "provider": provider,
        "request_timeout": timeout,
    }
    if provider is LLMProvider.OPENAI:
        api_key = variables.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise AssistantConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'."
            )
        model = variables.get("OPENAI_MODEL", "").strip()
        if not model:
            raise AssistantConfigurationError(
                "OPENAI_MODEL is required when LLM_PROVIDER is 'openai'."
            )
        values.update(openai_api_key=api_key, openai_model=model)
    else:
        model = variables.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise AssistantConfigurationError(
                "OLLAMA_MODEL is required when LLM_PROVIDER is 'ollama'."
            )
        values.update(
            ollama_base_url=variables.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            ollama_model=model,
        )

    try:
        return AssistantConfig.model_validate(values)
    except ValidationError:
        raise AssistantConfigurationError(
            "Invalid assistant configuration; check the selected provider URL, model, "
            "and request timeout."
        ) from None


def require_assistant_config(
    environ: Mapping[str, str] | None = None,
) -> AssistantConfig:
    """Load provider settings or fail clearly when the assistant is not configured."""
    config = load_assistant_config(environ)
    if config is None:
        raise AssistantUnavailableError(
            "No LLM provider is configured; assistant functionality is unavailable."
        )
    return config
