from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_MEASUREMENT_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
)

_RESULT_COLUMNS = (
    "row_index",
    "timestamp",
    "column",
    "value",
    "baseline_mean",
    "baseline_std",
    "lower_bound",
    "upper_bound",
    "baseline_count",
    "is_suspicious_outlier",
    "classification",
    "confirmed_anomaly",
    "method",
)

logger = logging.getLogger(__name__)


def _validate_non_negative_finite(value: float, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return normalized


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _select_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        selected = tuple(column for column in DEFAULT_MEASUREMENT_COLUMNS if column in frame)
    else:
        selected = tuple(columns)

    if not selected:
        raise ValueError("No measurement columns were selected or found.")
    if len(set(selected)) != len(selected):
        raise ValueError("Measurement columns must not contain duplicates.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataframe is missing measurement columns: {', '.join(missing)}")
    return selected


def _finite_numeric_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _global_iqr_bounds(
    values: pd.Series,
    multiplier: float,
) -> tuple[float, float, int] | None:
    finite_values = values.dropna()
    if len(finite_values) < 4 or finite_values.nunique() < 2:
        return None
    first_quartile = finite_values.quantile(0.25)
    third_quartile = finite_values.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    if np.isclose(interquartile_range, 0.0):
        return None
    return (
        float(first_quartile - multiplier * interquartile_range),
        float(third_quartile + multiplier * interquartile_range),
        int(len(finite_values)),
    )


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=_RESULT_COLUMNS)


def _prepare_time_series(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise ValueError(f"Dataframe is missing timestamp column: {timestamp_column}")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_count = int(timestamps.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{timestamp_column} contains {invalid_count} missing or invalid timestamps."
        )
    if timestamps.duplicated().any():
        raise ValueError(
            f"{timestamp_column} must be unique so baselines use strictly earlier observations."
        )

    ordered = frame.copy()
    ordered["_detector_row_index"] = np.arange(len(ordered))
    ordered[timestamp_column] = timestamps
    return ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)


def _detection_rows(
    *,
    row_indices: np.ndarray,
    timestamps: pd.Series,
    column: str,
    values: pd.Series,
    means: pd.Series,
    standard_deviations: pd.Series,
    counts: pd.Series,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    lower_bounds = means - threshold * standard_deviations
    upper_bounds = means + threshold * standard_deviations
    eligible = values.notna() & means.notna() & standard_deviations.notna()
    flags = eligible & ((values < lower_bounds) | (values > upper_bounds))
    return pd.DataFrame(
        {
            "row_index": row_indices,
            "timestamp": timestamps.to_numpy(),
            "column": column,
            "value": values.to_numpy(),
            "baseline_mean": means.to_numpy(),
            "baseline_std": standard_deviations.to_numpy(),
            "lower_bound": lower_bounds.to_numpy(),
            "upper_bound": upper_bounds.to_numpy(),
            "baseline_count": counts.fillna(0).astype("int64").to_numpy(),
            "is_suspicious_outlier": flags.to_numpy(dtype=bool),
            "classification": np.where(flags, "suspicious_outlier", "not_flagged"),
            "confirmed_anomaly": False,
            "method": method,
        },
        columns=_RESULT_COLUMNS,
    )


def detect_global_iqr(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Flag global IQR outliers as suspicious, not as confirmed anomalies.

    Quantile bounds are calculated independently for each selected column over all
    finite observations. Because this baseline ignores time and seasonality, normal
    consumption peaks can be flagged. The input frame is not modified.
    """
    multiplier = _validate_non_negative_finite(iqr_multiplier, "iqr_multiplier")
    selected_columns = _select_columns(frame, columns)
    if frame.empty:
        return _empty_result()

    timestamps = (
        frame["timestamp"]
        if "timestamp" in frame.columns
        else pd.Series([pd.NaT] * len(frame), index=frame.index)
    )
    flagged_records: list[dict[str, object]] = []

    for column in selected_columns:
        values = _finite_numeric_values(frame[column])
        bounds = _global_iqr_bounds(values, multiplier)
        if bounds is None:
            continue
        lower_bound, upper_bound, baseline_count = bounds
        mask = values.notna() & ((values < lower_bound) | (values > upper_bound))
        for position in np.flatnonzero(mask.to_numpy()):
            flagged_records.append(
                {
                    "row_index": int(position),
                    "timestamp": timestamps.iloc[position],
                    "column": column,
                    "value": float(values.iloc[position]),
                    "baseline_mean": np.nan,
                    "baseline_std": np.nan,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "baseline_count": baseline_count,
                    "is_suspicious_outlier": True,
                    "classification": "suspicious_outlier",
                    "confirmed_anomaly": False,
                    "method": "global_iqr",
                }
            )

    result = (
        pd.DataFrame.from_records(flagged_records, columns=_RESULT_COLUMNS)
        if flagged_records
        else _empty_result()
    )
    logger.info(
        "Global IQR generated %d exploratory suspicious-outlier flags.",
        len(result),
    )
    return result


class GlobalIQRDetector:
    """Train-fitted global IQR detector using the shared exploratory threshold rule.

    Unlike :func:`detect_global_iqr`, this estimator learns quartile bounds from
    training observations and applies them unchanged to later partitions.
    """

    def __init__(
        self,
        *,
        columns: Sequence[str] | None = None,
        iqr_multiplier: float = 1.5,
    ) -> None:
        self.iqr_multiplier = _validate_non_negative_finite(iqr_multiplier, "iqr_multiplier")
        self.columns = None if columns is None else tuple(columns)

    def fit(self, training_frame: pd.DataFrame) -> GlobalIQRDetector:
        """Learn one set of finite-value quartile bounds per configured column."""
        selected_columns = _select_columns(training_frame, self.columns)
        if training_frame.empty:
            raise ValueError("training_frame must contain observations.")
        self.columns_ = selected_columns
        self.bounds_ = {
            column: _global_iqr_bounds(
                _finite_numeric_values(training_frame[column]),
                self.iqr_multiplier,
            )
            for column in selected_columns
        }
        logger.info(
            "Fitted train-only global IQR bounds for %d measurement columns.",
            sum(bounds is not None for bounds in self.bounds_.values()),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the existing long-form suspicious-outlier schema for new rows."""
        if not hasattr(self, "bounds_"):
            raise RuntimeError("GlobalIQRDetector must be fitted before prediction.")
        selected_columns = _select_columns(frame, self.columns_)
        timestamps = (
            frame["timestamp"]
            if "timestamp" in frame.columns
            else pd.Series([pd.NaT] * len(frame), index=frame.index)
        )
        records: list[dict[str, object]] = []
        for column in selected_columns:
            bounds = self.bounds_[column]
            if bounds is None:
                continue
            lower_bound, upper_bound, baseline_count = bounds
            values = _finite_numeric_values(frame[column])
            flags = values.notna() & ((values < lower_bound) | (values > upper_bound))
            for position in np.flatnonzero(flags.to_numpy()):
                records.append(
                    {
                        "row_index": int(position),
                        "timestamp": timestamps.iloc[position],
                        "column": column,
                        "value": float(values.iloc[position]),
                        "baseline_mean": np.nan,
                        "baseline_std": np.nan,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "baseline_count": baseline_count,
                        "is_suspicious_outlier": True,
                        "classification": "suspicious_outlier",
                        "confirmed_anomaly": False,
                        "method": "global_iqr",
                    }
                )
        return (
            pd.DataFrame.from_records(records, columns=_RESULT_COLUMNS)
            if records
            else _empty_result()
        )


def detect_rolling_outliers(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    window_size: int = 60,
    threshold: float = 3.0,
    minimum_history: int | None = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Compare each value with rolling statistics from strictly earlier rows.

    ``window_size`` counts prior observations, not elapsed time. The current row is
    excluded from the baseline. Missing/non-finite values do not contribute to it,
    and no row is evaluated until ``minimum_history`` finite prior values exist.
    A zero standard deviation yields equal bounds at the historical mean.
    """
    size = _validate_positive_integer(window_size, "window_size")
    sigma_threshold = _validate_non_negative_finite(threshold, "threshold")
    history_size = (
        size
        if minimum_history is None
        else _validate_positive_integer(minimum_history, "minimum_history")
    )
    if history_size > size:
        raise ValueError("minimum_history must not exceed window_size.")
    selected_columns = _select_columns(frame, columns)
    ordered = _prepare_time_series(frame, timestamp_column)
    if ordered.empty:
        return _empty_result()

    rows: list[pd.DataFrame] = []
    row_indices = ordered["_detector_row_index"].to_numpy(dtype="int64")
    timestamps = ordered[timestamp_column]
    for column in selected_columns:
        values = _finite_numeric_values(ordered[column]).reset_index(drop=True)
        past = values.shift(1)
        window = past.rolling(window=size, min_periods=history_size)
        means = window.mean()
        standard_deviations = window.std(ddof=0)
        counts = past.rolling(window=size, min_periods=1).count()
        rows.append(
            _detection_rows(
                row_indices=row_indices,
                timestamps=timestamps,
                column=column,
                values=values,
                means=means,
                standard_deviations=standard_deviations,
                counts=counts,
                threshold=sigma_threshold,
                method="rolling_mean_std",
            )
        )

    result = pd.concat(rows, ignore_index=True)
    logger.info(
        "Rolling statistical detector generated %d exploratory suspicious-outlier flags.",
        int(result["is_suspicious_outlier"].sum()),
    )
    return result


def detect_time_of_day_outliers(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    threshold: float = 3.0,
    minimum_history: int = 4,
    timestamp_column: str = "timestamp",
    group_by_day_of_week: bool = True,
) -> pd.DataFrame:
    """Compare readings with historical values from the same clock-time bucket.

    The default group is day-of-week plus minute-of-day. Only earlier timestamps
    contribute. ``group_by_day_of_week=False`` pools weekdays and weekends by
    minute-of-day. Baselines use expanding population standard deviation and are
    withheld until the requested number of finite historical values is available.
    """
    sigma_threshold = _validate_non_negative_finite(threshold, "threshold")
    history_size = _validate_positive_integer(minimum_history, "minimum_history")
    selected_columns = _select_columns(frame, columns)
    ordered = _prepare_time_series(frame, timestamp_column)
    if ordered.empty:
        return _empty_result()

    timestamps = ordered[timestamp_column]
    group_keys = pd.DataFrame(
        {
            "_minute_of_day": timestamps.dt.hour * 60 + timestamps.dt.minute,
        }
    )
    if group_by_day_of_week:
        group_keys["_day_of_week"] = timestamps.dt.day_of_week.to_numpy()
    groups = group_keys.groupby(
        list(group_keys.columns),
        sort=False,
        observed=True,
    ).indices.values()

    rows: list[pd.DataFrame] = []
    row_indices = ordered["_detector_row_index"].to_numpy(dtype="int64")
    for column in selected_columns:
        values = _finite_numeric_values(ordered[column]).reset_index(drop=True)
        means = np.full(len(ordered), np.nan)
        standard_deviations = np.full(len(ordered), np.nan)
        counts = np.zeros(len(ordered), dtype="int64")

        for positions in groups:
            group_values = values.iloc[positions]
            prior_values = group_values.shift(1)
            expanding = prior_values.expanding(min_periods=history_size)
            means[positions] = expanding.mean().to_numpy(dtype=float)
            standard_deviations[positions] = expanding.std(ddof=0).to_numpy(dtype=float)
            counts[positions] = (
                prior_values.expanding(min_periods=1).count().fillna(0).to_numpy(dtype="int64")
            )

        rows.append(
            _detection_rows(
                row_indices=row_indices,
                timestamps=timestamps,
                column=column,
                values=values,
                means=pd.Series(means),
                standard_deviations=pd.Series(standard_deviations),
                counts=pd.Series(counts),
                threshold=sigma_threshold,
                method="time_of_day_mean_std",
            )
        )

    result = pd.concat(rows, ignore_index=True)
    logger.info(
        "Time-of-day detector generated %d exploratory suspicious-outlier flags.",
        int(result["is_suspicious_outlier"].sum()),
    )
    return result
