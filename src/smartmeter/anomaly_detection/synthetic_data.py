from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from smartmeter.anomaly_detection.features import MEASUREMENT_COLUMNS

SUSTAINED_MODES = ("high", "low", "random")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticInjectionConfig:
    """Reproducible rates and physical bounds for controlled test anomalies.

    Point-anomaly rates specify the fraction of input rows to target. The sustained
    rate specifies a target row fraction, rounded down to complete fixed-length
    segments. The default policy prohibits any row from receiving multiple injections.
    """

    random_seed: int = 42
    consumption_spike_rate: float = 0.01
    low_consumption_rate: float = 0.01
    sustained_consumption_rate: float = 0.02
    voltage_anomaly_rate: float = 0.01
    sustained_duration: int = 5
    spike_multiplier: float = 2.5
    minimum_spike_increase_kw: float = 0.5
    low_consumption_factor: float = 0.1
    low_consumption_floor_kw: float = 0.05
    sustained_high_multiplier: float = 2.0
    sustained_low_factor: float = 0.25
    sustained_mode: Literal["high", "low", "random"] = "high"
    maximum_active_power_kw: float = 15.0
    voltage_deviation_volts: float = 20.0
    minimum_voltage_volts: float = 180.0
    maximum_voltage_volts: float = 260.0


def _validate_config(config: SyntheticInjectionConfig) -> None:
    if isinstance(config.random_seed, bool) or not isinstance(config.random_seed, int):
        raise ValueError("random_seed must be an integer.")
    for name in (
        "consumption_spike_rate",
        "low_consumption_rate",
        "sustained_consumption_rate",
        "voltage_anomaly_rate",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be a finite fraction between 0 and 1.")
    if (
        isinstance(config.sustained_duration, bool)
        or not isinstance(config.sustained_duration, int)
        or config.sustained_duration < 2
    ):
        raise ValueError("sustained_duration must be an integer of at least 2.")
    for name in (
        "spike_multiplier",
        "minimum_spike_increase_kw",
        "sustained_high_multiplier",
        "maximum_active_power_kw",
        "voltage_deviation_volts",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number.")
    for name in ("low_consumption_factor", "sustained_low_factor"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value < 1
        ):
            raise ValueError(f"{name} must be a finite fraction in [0, 1).")
    if config.spike_multiplier <= 1:
        raise ValueError("Consumption high multipliers must be greater than 1.")
    if (
        isinstance(config.low_consumption_floor_kw, bool)
        or not isinstance(config.low_consumption_floor_kw, Real)
        or not math.isfinite(config.low_consumption_floor_kw)
        or config.low_consumption_floor_kw < 0
    ):
        raise ValueError("low_consumption_floor_kw must be finite and non-negative.")
    if config.sustained_mode not in SUSTAINED_MODES:
        raise ValueError(f"sustained_mode must be one of {SUSTAINED_MODES}.")
    if (
        not isinstance(config.minimum_voltage_volts, Real)
        or not isinstance(config.maximum_voltage_volts, Real)
        or not math.isfinite(config.minimum_voltage_volts)
        or not math.isfinite(config.maximum_voltage_volts)
        or config.minimum_voltage_volts >= config.maximum_voltage_volts
    ):
        raise ValueError("Voltage bounds must be finite and minimum must be below maximum.")


def _row_target(row_count: int, rate: float) -> int:
    return int(math.floor(row_count * rate + 0.5))


def _choose_positions(
    candidates: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    category: str,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(
            f"Cannot inject {category}: requested {count} observations but only "
            f"{len(candidates)} are eligible."
        )
    return rng.permutation(candidates)[:count]


def _contiguous_segment_starts(eligible: np.ndarray, duration: int) -> np.ndarray:
    """Find starts of all all-eligible fixed-duration runs in linear vectorized time."""
    if len(eligible) < duration:
        return np.empty(0, dtype=np.int64)
    invalid_prefix = np.empty(len(eligible) + 1, dtype=np.int64)
    invalid_prefix[0] = 0
    np.cumsum(~eligible, dtype=np.int64, out=invalid_prefix[1:])
    invalid_counts = invalid_prefix[duration:] - invalid_prefix[:-duration]
    return np.flatnonzero(invalid_counts == 0)


def inject_synthetic_anomalies(
    frame: pd.DataFrame,
    *,
    config: SyntheticInjectionConfig | None = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Return a labeled synthetic copy while leaving all source values untouched.

    Original measurement columns and ``original_<measurement>`` copies remain
    unchanged. ``synthetic_<measurement>`` columns carry modified values (and unchanged
    copies for measurements not targeted on that row). Labels are explicitly named
    ``synthetic_anomaly_label`` and ``synthetic_anomaly_category``; they are controlled
    test labels, not real-world anomaly ground truth.

    Spike, low-consumption, sustained-consumption, and voltage events are drawn from a
    seeded generator. Rows are selected without overlap; infeasible requests raise
    ``ValueError`` instead of silently reducing the requested injection count.
    """
    selected_config = config or SyntheticInjectionConfig()
    _validate_config(selected_config)
    if frame.empty:
        raise ValueError("Cannot inject synthetic anomalies into an empty dataframe.")

    required = [timestamp_column, *MEASUREMENT_COLUMNS]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError("Dataframe is missing required columns: " + ", ".join(missing_columns))

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"{timestamp_column} contains missing or invalid timestamps.")
    if timestamps.duplicated().any():
        raise ValueError(f"{timestamp_column} must be unique for synthetic injection.")

    row_count = len(frame)
    point_counts = {
        "consumption_spike": _row_target(row_count, selected_config.consumption_spike_rate),
        "unusually_low_consumption": _row_target(row_count, selected_config.low_consumption_rate),
        "voltage_anomaly": _row_target(row_count, selected_config.voltage_anomaly_rate),
    }
    sustained_target = _row_target(
        row_count,
        selected_config.sustained_consumption_rate,
    )
    sustained_segments = sustained_target // selected_config.sustained_duration
    sustained_count = sustained_segments * selected_config.sustained_duration
    requested_count = sum(point_counts.values()) + sustained_count
    if requested_count > row_count:
        raise ValueError(
            "Requested synthetic injections overlap: the configured categories require "
            f"{requested_count} distinct rows but the dataframe has {row_count}."
        )

    order = np.argsort(timestamps.array.asi8, kind="stable")
    inverse_order = np.empty(row_count, dtype=np.int64)
    inverse_order[order] = np.arange(row_count)
    sorted_timestamps = timestamps.iloc[order].reset_index(drop=True)
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype="float64", na_value=np.nan)[order]
        for column in MEASUREMENT_COLUMNS
    }
    active_power = numeric["global_active_power"]
    voltage = numeric["voltage"]
    active_finite = np.isfinite(active_power) & (active_power >= 0)
    voltage_finite = np.isfinite(voltage)

    rng = np.random.default_rng(selected_config.random_seed)
    reserved = np.zeros(row_count, dtype=bool)
    categories = np.full(row_count, None, dtype=object)
    metadata: list[dict[str, object] | None] = [None] * row_count
    synthetic_values = {column: values.copy() for column, values in numeric.items()}
    event_number = 0

    def record_event(
        category: str,
        chronological_positions: np.ndarray,
        *,
        target_column: str,
        method: str,
        parameters: dict[str, object],
    ) -> None:
        nonlocal event_number
        event_number += 1
        source_positions = order[chronological_positions]
        start_position = int(chronological_positions[0])
        end_position = int(chronological_positions[-1])
        event_metadata = {
            "random_seed": selected_config.random_seed,
            "event_id": event_number,
            "category": category,
            "target_column": target_column,
            "method": method,
            "start_position": start_position,
            "end_position": end_position,
            "start_timestamp": sorted_timestamps.iloc[start_position].isoformat(),
            "end_timestamp": sorted_timestamps.iloc[end_position].isoformat(),
            "parameters": parameters,
        }
        categories[chronological_positions] = category
        reserved[chronological_positions] = True
        for chronological_position, source_position in zip(
            chronological_positions,
            source_positions,
            strict=True,
        ):
            metadata[int(chronological_position)] = {
                **event_metadata,
                "observation_position": int(chronological_position),
                "source_position": int(source_position),
            }

    spike_candidates = np.flatnonzero(
        active_finite & (active_power < selected_config.maximum_active_power_kw)
    )
    spike_positions = _choose_positions(
        spike_candidates,
        point_counts["consumption_spike"],
        rng=rng,
        category="consumption_spike",
    )
    for position in spike_positions:
        original = active_power[position]
        injected = min(
            max(
                original * selected_config.spike_multiplier,
                original + selected_config.minimum_spike_increase_kw,
            ),
            selected_config.maximum_active_power_kw,
        )
        synthetic_values["global_active_power"][position] = injected
        record_event(
            "consumption_spike",
            np.array([position]),
            target_column="global_active_power",
            method="bounded_multiplier",
            parameters={
                "original_value": float(original),
                "synthetic_value": float(injected),
                "maximum_active_power_kw": selected_config.maximum_active_power_kw,
            },
        )

    low_candidates = np.flatnonzero(
        active_finite & (active_power > selected_config.low_consumption_floor_kw) & ~reserved
    )
    low_positions = _choose_positions(
        low_candidates,
        point_counts["unusually_low_consumption"],
        rng=rng,
        category="unusually_low_consumption",
    )
    for position in low_positions:
        original = active_power[position]
        injected = max(
            original * selected_config.low_consumption_factor,
            selected_config.low_consumption_floor_kw,
        )
        synthetic_values["global_active_power"][position] = injected
        record_event(
            "unusually_low_consumption",
            np.array([position]),
            target_column="global_active_power",
            method="bounded_fraction",
            parameters={
                "original_value": float(original),
                "synthetic_value": float(injected),
                "fraction": selected_config.low_consumption_factor,
            },
        )

    duration = selected_config.sustained_duration
    if sustained_segments:
        high_eligible = (
            active_finite & (active_power < selected_config.maximum_active_power_kw) & ~reserved
        )
        low_eligible = (
            active_finite & (active_power > selected_config.low_consumption_floor_kw) & ~reserved
        )
        high_starts = _contiguous_segment_starts(high_eligible, duration)
        low_starts = _contiguous_segment_starts(low_eligible, duration)
        if selected_config.sustained_mode == "high":
            valid_starts = high_starts
        elif selected_config.sustained_mode == "low":
            valid_starts = low_starts
        else:
            valid_starts = np.union1d(high_starts, low_starts)

        if len(valid_starts) < sustained_segments:
            raise ValueError(
                "Cannot inject sustained abnormal consumption: not enough contiguous "
                "eligible observations remain after other injections."
            )

        chosen_starts: list[int] = []
        next_available_start = 0
        for start_value in valid_starts:
            start = int(start_value)
            if start < next_available_start:
                continue
            positions = np.arange(start, start + duration)
            if high_eligible[positions].all() and low_eligible[positions].all():
                mode = (
                    str(rng.choice(("high", "low")))
                    if (selected_config.sustained_mode == "random")
                    else selected_config.sustained_mode
                )
            elif high_eligible[positions].all():
                mode = "high"
            else:
                mode = "low"
            original = active_power[positions]
            if mode == "high":
                injected = np.minimum(
                    original * selected_config.sustained_high_multiplier,
                    selected_config.maximum_active_power_kw,
                )
                direction = "high"
            else:
                injected = np.maximum(
                    original * selected_config.sustained_low_factor,
                    selected_config.low_consumption_floor_kw,
                )
                direction = "low"
            synthetic_values["global_active_power"][positions] = injected
            record_event(
                "sustained_abnormal_consumption",
                positions,
                target_column="global_active_power",
                method="bounded_sustained_fraction",
                parameters={
                    "direction": direction,
                    "duration_observations": duration,
                    "factor": (
                        selected_config.sustained_high_multiplier
                        if mode == "high"
                        else selected_config.sustained_low_factor
                    ),
                },
            )
            chosen_starts.append(start)
            next_available_start = start + duration
            if len(chosen_starts) == sustained_segments:
                break

        if len(chosen_starts) != sustained_segments:
            raise ValueError("Could not place all requested sustained anomaly segments.")

    voltage_candidates = np.flatnonzero(voltage_finite & ~reserved)
    voltage_positions = _choose_positions(
        voltage_candidates,
        point_counts["voltage_anomaly"],
        rng=rng,
        category="voltage_anomaly",
    )
    for position in voltage_positions:
        original = voltage[position]
        lower = float(
            np.clip(
                original - selected_config.voltage_deviation_volts,
                selected_config.minimum_voltage_volts,
                selected_config.maximum_voltage_volts,
            )
        )
        upper = float(
            np.clip(
                original + selected_config.voltage_deviation_volts,
                selected_config.minimum_voltage_volts,
                selected_config.maximum_voltage_volts,
            )
        )
        available = np.array([value for value in (lower, upper) if value != original])
        if available.size == 0:
            raise ValueError(
                "Cannot inject voltage anomaly: selected voltage has no valid deviation."
            )
        injected = float(rng.choice(available))
        synthetic_values["voltage"][position] = injected
        record_event(
            "voltage_anomaly",
            np.array([position]),
            target_column="voltage",
            method="bounded_voltage_deviation",
            parameters={
                "original_value": float(original),
                "synthetic_value": injected,
                "deviation_volts": selected_config.voltage_deviation_volts,
                "minimum_voltage_volts": selected_config.minimum_voltage_volts,
                "maximum_voltage_volts": selected_config.maximum_voltage_volts,
            },
        )

    result = frame.copy()
    for column in MEASUREMENT_COLUMNS:
        original_values = frame[column].to_numpy(copy=True)
        result[f"original_{column}"] = original_values
        result[f"synthetic_{column}"] = synthetic_values[column][inverse_order]
    result["synthetic_anomaly_label"] = reserved.take(inverse_order).astype("uint8")
    result["synthetic_anomaly_category"] = categories.take(inverse_order)
    result["synthetic_injection_metadata"] = [
        metadata[int(position)] if metadata[int(position)] is not None else None
        for position in inverse_order
    ]

    logger.info(
        "Injected %d synthetic anomaly observations using seed %d.",
        int(result["synthetic_anomaly_label"].sum()),
        selected_config.random_seed,
    )
    return result
