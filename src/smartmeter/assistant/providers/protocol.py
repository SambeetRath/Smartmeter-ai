from __future__ import annotations

from typing import Protocol, runtime_checkable

from smartmeter.assistant.models import ProviderRequest, ProviderResponse


@runtime_checkable
class LLMProviderClient(Protocol):
    """Synchronous provider boundary matching the current synchronous FastAPI stack.

    Future adapters should implement ``complete`` using their synchronous SDK/client.
    FastAPI synchronous route functions run in its worker thread pool, avoiding blocking
    the event loop. An async adapter should be introduced only alongside an async route.
    """

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Submit a normalized request and return a normalized provider response."""
        ...
