from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class ConfigurationError(ValueError):
    """Raised when dashboard configuration is invalid."""


@dataclass(frozen=True)
class DashboardConfig:
    api_base_url: str = DEFAULT_API_BASE_URL

    def __post_init__(self) -> None:
        value = self.api_base_url.strip()
        try:
            parsed = urlsplit(value)
        except ValueError as error:
            raise ConfigurationError("SMARTMETER_API_BASE_URL is not a valid URL.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError(
                "SMARTMETER_API_BASE_URL must be an HTTP(S) base URL without credentials, "
                "query parameters, or a fragment."
            )
        try:
            port = parsed.port
            hostname = parsed.hostname
        except ValueError as error:
            raise ConfigurationError("SMARTMETER_API_BASE_URL contains an invalid port.") from error
        if not hostname:
            raise ConfigurationError("SMARTMETER_API_BASE_URL must include a hostname.")
        if port is not None and not 1 <= port <= 65535:
            raise ConfigurationError("SMARTMETER_API_BASE_URL contains an invalid port.")
        normalized_path = parsed.path.rstrip("/")
        object.__setattr__(
            self,
            "api_base_url",
            urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", "")),
        )

    @classmethod
    def from_env(cls) -> DashboardConfig:
        return cls(
            api_base_url=os.getenv("SMARTMETER_API_BASE_URL", DEFAULT_API_BASE_URL),
        )
