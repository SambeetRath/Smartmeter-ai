from __future__ import annotations

import logging
from typing import Literal

from smartmeter.config import DEFAULT_LOG_LEVEL


def configure_logging(
    level: str | Literal["DEBUG", "INFO", "WARNING", "ERROR"] = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """Create a logger configured for predictable CLI output."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("smartmeter")
