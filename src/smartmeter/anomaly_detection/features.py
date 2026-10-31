from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

MEASUREMENT_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
)
DEFAULT_LAG_PERIODS = (1, 5, 15, 60)
DEFAULT_ROLLING_WINDOWS = (60,)

logger = logging.getLogger(__name__)


def _validate_periods(periods: Sequence[int], name: str) -> tuple[int, ...]:
    normalized = tuple(periods)
    if any(
        isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period <= 0
        for period in normalized
    ):
        raise ValueError(f"{name} must contain only positive integers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(int(period) for period in normalized)


def _measurement_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = (
        tuple(column for column in MEASUREMENT_COLUMNS if column in frame.columns)
        if columns is None
        else tuple(columns)
    )
    if not selected:
        raise ValueError("No measurement columns were selected or found.")
    if len(set(selected)) != len(selected):
        raise ValueError("Measurement columns must not contain duplicates.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataframe is missing measurement columns: {', '.join(missing)}")
    return selected


def _utc_timestamps(frame: pd.DataFrame, timestamp_column: str) -> pd.Series:
    if timestamp_column not in frame.columns:
        raise ValueError(f"Dataframe is missing timestamp column: {timestamp_column}")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    invalid_count = int(timestamps.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{timestamp_column} contains {invalid_count} missing or invalid timestamps."
        )
    if timestamps.duplicated().any():
        raise ValueError(f"{timestamp_column} must be unique for unambiguous temporal features.")
    return timestamps


def _finite_numeric_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def engineer_features(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    timestamp_column: str = "timestamp",
    lag_periods: Sequence[int] = DEFAULT_LAG_PERIODS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    rolling_min_periods: int = 1,
) -> pd.DataFrame:
    """Return original rows plus temporal, lag, and past-only rolling features.

    The function preserves the input's row order, index, and columns. Temporal and
    measurement-derived features are aligned back to each original row after stable
    chronological calculation. Source measurement columns remain unchanged. Missing
    or non-finite measurements are treated as unavailable in derived features, and
    ``<measurement>__is_missing`` makes that state explicit. No observations are
    dropped. Derived numeric columns use float32 to reduce memory use on large datasets.

    Rolling values are shifted before calculation, so the current observation never
    contributes to its own historical statistics. Compute features on the full
    chronological series before splitting: later split periods can use earlier
    observations as historical context, but no feature uses a future value.
    """
    selected_columns = _measurement_columns(frame, columns)
    lags = _validate_periods(lag_periods, "lag_periods")
    windows = _validate_periods(rolling_windows, "rolling_windows")
    if (
        isinstance(rolling_min_periods, bool)
        or not isinstance(rolling_min_periods, (int, np.integer))
        or rolling_min_periods <= 0
    ):
        raise ValueError("rolling_min_periods must be a positive integer.")
    if windows and rolling_min_periods > min(windows):
        raise ValueError("rolling_min_periods must not exceed the smallest rolling window.")
    if frame.empty:
        raise ValueError("Cannot engineer features for an empty dataframe.")

    timestamps = _utc_timestamps(frame, timestamp_column)
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0 + timestamps.dt.second / 3600.0
    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    day_of_week = timestamps.dt.day_of_week
    month = timestamps.dt.month

    result = frame.copy()
    result["hour_of_day"] = hour.to_numpy(dtype="float32")
    result["minute_of_day"] = minute_of_day.to_numpy(dtype="int16")
    result["day_of_week"] = day_of_week.to_numpy(dtype="int8")
    result["is_weekend"] = (day_of_week >= 5).to_numpy(dtype="uint8")
    result["month"] = month.to_numpy(dtype="int8")
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).to_numpy(dtype="float32")
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).to_numpy(dtype="float32")
    result["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0).to_numpy(dtype="float32")
    result["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0).to_numpy(dtype="float32")
    result["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0).to_numpy(dtype="float32")
    result["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0).to_numpy(dtype="float32")

    sort_positions = np.argsort(timestamps.array.asi8, kind="stable")
    inverse_positions = np.empty(len(sort_positions), dtype=np.int64)
    inverse_positions[sort_positions] = np.arange(len(sort_positions))

    for column in selected_columns:
        original_values = _finite_numeric_values(frame[column])
        sorted_values = original_values.to_numpy(dtype="float64", na_value=np.nan)[sort_positions]
        values = pd.Series(sorted_values)
        result[f"{column}__is_missing"] = original_values.isna().to_numpy(dtype="uint8")

        for period in lags:
            lagged = values.shift(period).to_numpy(dtype="float32")[inverse_positions]
            result[f"{column}__lag_{period}"] = lagged

        historical_values = values.shift(1)
        for window_size in windows:
            rolling = historical_values.rolling(
                window=window_size,
                min_periods=rolling_min_periods,
            )
            mean = rolling.mean()
            result[f"{column}__rolling_mean_{window_size}"] = mean.to_numpy(dtype="float32")[
                inverse_positions
            ]
            result[f"{column}__rolling_std_{window_size}"] = rolling.std(ddof=0).to_numpy(
                dtype="float32"
            )[inverse_positions]
            result[f"{column}__rolling_min_{window_size}"] = rolling.min().to_numpy(
                dtype="float32"
            )[inverse_positions]
            result[f"{column}__rolling_max_{window_size}"] = rolling.max().to_numpy(
                dtype="float32"
            )[inverse_positions]
            deviation = (values - mean).to_numpy(dtype="float32")[inverse_positions]
            result[f"{column}__deviation_from_rolling_mean_{window_size}"] = deviation

    added_features = len(result.columns) - len(frame.columns)
    logger.info(
        "Engineered %d temporal and measurement features for %d rows and %d measurements.",
        added_features,
        len(result),
        len(selected_columns),
    )
    return result
