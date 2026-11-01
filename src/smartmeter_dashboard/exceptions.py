class DashboardAPIError(RuntimeError):
    """Base exception for dashboard-to-API communication errors."""


class APIConnectionError(DashboardAPIError):
    """The API could not be reached within the configured timeout."""


class APIHTTPError(DashboardAPIError):
    """The API returned an unsuccessful HTTP status."""

    def __init__(self, status_code: int, endpoint: str, detail: str) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"API request to {endpoint} failed with HTTP {status_code}.")


class APIResponseError(DashboardAPIError):
    """The API response was empty, invalid JSON, or did not match its contract."""


class APIRequestTimeout(APIConnectionError):
    """The API connection or response exceeded its timeout."""
