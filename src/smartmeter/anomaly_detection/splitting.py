from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalDataSplit:
    """Chronologically ordered train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
    timestamp_column: str = "timestamp",
) -> TemporalDataSplit:
    """Split rows by chronological position without randomization or overlap.

    The source is not modified. The returned partitions are copies sorted by timestamp
    and retain the original index labels. All partitions must contain at least one row.
    Generate past-only features on the full ordered source before splitting when later
    periods need legitimate historical context from the preceding training period.
    Synthetic anomalies belong only in a copy of the held-out validation/test data.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Dataframe is missing timestamp column: {timestamp_column}")

    train_size = _validate_fraction(train_fraction, "train_fraction")
    validation_size = _validate_fraction(validation_fraction, "validation_fraction")
    if test_fraction is None:
        test_size = 1.0 - train_size - validation_size
    else:
        test_size = _validate_fraction(test_fraction, "test_fraction")
        if not math.isclose(train_size + validation_size + test_size, 1.0, abs_tol=1e-9):
            raise ValueError(
                "train_fraction, validation_fraction, and test_fraction must sum to 1."
            )
    if test_size <= 0 or not math.isfinite(test_size):
        raise ValueError("The resulting test_fraction must be greater than zero.")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_column} contains missing or invalid timestamps.")
    if timestamps.duplicated().any():
        raise ValueError(f"{timestamp_column} must be unique for a chronological split.")

    order = np.argsort(timestamps.array.asi8, kind="stable")
    ordered = frame.iloc[order]
    row_count = len(ordered)
    train_end = int(math.floor(row_count * train_size))
    validation_end = int(math.floor(row_count * (train_size + validation_size)))
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError(
            "The requested fractions must produce non-empty train, validation, and test sets."
        )

    split = TemporalDataSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
    logger.info(
        "Chronological split created: train=%d validation=%d test=%d.",
        len(split.train),
        len(split.validation),
        len(split.test),
    )
    return split


def _validate_fraction(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite fraction between 0 and 1.")
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction between 0 and 1.") from exc
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(f"{name} must be a finite fraction between 0 and 1.")
    return fraction
