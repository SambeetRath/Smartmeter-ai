from __future__ import annotations


class ProviderError(RuntimeError):
    """Safe provider failure suitable for handling at the assistant boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProviderDependencyError(ProviderError):
    """Raised when an explicitly selected provider's optional SDK is missing."""

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            "missing_dependency",
            f"The selected provider requires optional dependency '{package}'. "
            f"Install it with: python -m pip install -e .[{extra}]",
        )


class ProviderConfigurationError(ProviderError):
    """Raised when provider construction does not match validated configuration."""

    def __init__(self, message: str) -> None:
        super().__init__("configuration_error", message)


def sanitize_provider_exception(error: Exception, provider: str) -> ProviderError:
    """Map SDK/client failures to messages that do not include provider error content."""
    error_name = type(error).__name__
    status_code = getattr(error, "status_code", None)
    if error_name in {"AuthenticationError", "UnauthorizedError"} or status_code == 401:
        return ProviderError(
            "authentication_error",
            f"{provider} authentication failed; verify the selected provider credentials.",
        )
    if error_name in {"RateLimitError", "TooManyRequests"} or status_code == 429:
        return ProviderError(
            "rate_limited",
            f"{provider} rate limit reached; retry the request later.",
        )
    if "Timeout" in error_name or isinstance(error, TimeoutError):
        return ProviderError("timeout", f"{provider} request timed out.")
    if (
        "Connection" in error_name
        or "Connect" in error_name
        or isinstance(error, ConnectionError)
    ):
        return ProviderError(
            "connection_error",
            f"Could not connect to {provider}; verify provider availability and its configured URL.",
        )
    return ProviderError("provider_error", f"{provider} request failed.")
