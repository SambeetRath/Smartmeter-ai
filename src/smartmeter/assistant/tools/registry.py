from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from smartmeter.assistant.models import ToolDefinition
from smartmeter.assistant.tools.definitions import TOOL_SPECS, ToolSpec


class DuplicateToolError(ValueError):
    """Raised when a registry is constructed with duplicate tool names."""


class UnknownToolError(LookupError):
    """Raised when a tool name is not in the approved registry."""


class ToolRegistry:
    """Immutable allowlist of tools and their local argument validators.

    The registry has no execution or handler-dispatch operation. A future application
    integration must associate trusted handlers in code, never from tool-call arguments.
    """

    def __init__(self, tool_specs: Iterable[ToolSpec[Any]]) -> None:
        specs: dict[str, ToolSpec[Any]] = {}
        for spec in tool_specs:
            if spec.name in specs:
                raise DuplicateToolError(f"Duplicate analytics tool name: {spec.name}")
            spec.definition()
            specs[spec.name] = spec
        self._specs: Mapping[str, ToolSpec[Any]] = MappingProxyType(specs)

    def list_definitions(self) -> list[ToolDefinition]:
        """Return fresh provider-neutral definitions for registered tools."""
        return [spec.definition() for spec in self._specs.values()]

    def get_tool_spec(self, name: str) -> ToolSpec[Any]:
        """Return a registered tool spec by exact name."""
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownToolError("Unknown analytics tool name.") from None

    def get_definition(self, name: str) -> ToolDefinition:
        """Return a fresh definition for one exact registered tool name."""
        return self.get_tool_spec(name).definition()

    def validate_arguments(self, name: str, arguments: object) -> BaseModel:
        """Validate untrusted arguments with the exact registered Pydantic schema."""
        return self.get_tool_spec(name).validate_arguments(arguments)


APPROVED_TOOL_NAMES = tuple(spec.name for spec in TOOL_SPECS)
DEFAULT_TOOL_REGISTRY = ToolRegistry(TOOL_SPECS)
