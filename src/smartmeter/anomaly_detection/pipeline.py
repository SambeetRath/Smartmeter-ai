"""Chronological orchestration and reporting for the anomaly detectors."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from smartmeter.anomaly_detection.features import MEASUREMENT_COLUMNS, engineer_features
from smartmeter.anomaly_detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)
from smartmeter.anomaly_detection.statistical import (
    GlobalIQRDetector,
    detect_rolling_outliers,
    detect_time_of_day_outliers,
)
from smartmeter.config import DEFAULT_PROCESSED_DATASET_PATH, PROJECT_ROOT, REPORTS_DIR
from smartmeter.database.repositories import MeterRepository, ReadingRepository
from smartmeter.database.repositories.reading_repository import MAX_PAGE_SIZE
from smartmeter.database.session import SessionFactory, create_session_factory
from smartmeter.logging_config import configure_logging

logger = logging.getLogger(__name__)

DETECTOR_NAMES = (
    "global_iqr",
    "rolling_mean_std",
    "time_of_day",
    "isolation_forest",
)
DEFAULT_MODEL_FEATURES = (
    "global_active_power",
    "voltage",
    "hour_sin",
    "hour_cos",
    "global_active_power__lag_1",
    "global_active_power__rolling_mean_60",
)
_LABEL_COLUMNS = {
    "anomaly",
    "anomaly_category",
    "anomaly_label",
    "ground_truth",
    "is_anomaly",
    "label",
    "synthetic_anomaly_category",
    "synthetic_anomaly_label",
    "synthetic_ground_truth",
    "target",
    "y",
}
_RESULT_COLUMNS = (
    "run_id",
    "timestamp",
    "meter_identifier",
    *MEASUREMENT_COLUMNS,
    "detector_name",
    "anomaly_prediction",
    "prediction_available",
    "anomaly_score",
    "anomaly_category",
    "result_kind",
    "detector_metadata",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a repeatable, leakage-aware detection run.

    Only the chronologically held-out portion is emitted. Statistical baselines
    and Isolation Forest are fitted on the earlier training partition; causal
    statistical baselines may use preceding training observations as history.
    """

    source: Literal["parquet", "postgresql"] = "parquet"
    input_path: Path = DEFAULT_PROCESSED_DATASET_PATH
    results_path: Path = REPORTS_DIR / "anomaly_detections.csv"
    summary_path: Path = REPORTS_DIR / "anomaly_run.json"
    meter_identifier: str = "uci_household_1"
    detector_names: tuple[str, ...] = DETECTOR_NAMES
    measurement_columns: tuple[str, ...] = ("global_active_power",)
    train_fraction: float = 0.7
    iqr_multiplier: float = 1.5
    rolling_window_size: int = 60
    rolling_minimum_history: int = 30
    rolling_threshold: float = 3.0
    time_of_day_minimum_history: int = 4
    time_of_day_threshold: float = 3.0
    isolation_forest_config: IsolationForestConfig = field(
        default_factory=lambda: IsolationForestConfig(
            feature_columns=DEFAULT_MODEL_FEATURES,
        )
    )
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.source not in ("parquet", "postgresql"):
            raise ValueError("source must be 'parquet' or 'postgresql'.")
        if not isinstance(self.meter_identifier, str) or not self.meter_identifier.strip():
            raise ValueError("meter_identifier must be a non-empty string.")

        if isinstance(self.detector_names, str):
            raise ValueError("detector_names must be a sequence of detector names.")
        detectors = tuple(self.detector_names)
        if not detectors:
            raise ValueError("At least one detector must be selected.")
        if any(not isinstance(name, str) for name in detectors):
            raise ValueError("detector_names must contain only strings.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detector_names must not contain duplicates.")
        unknown = [name for name in detectors if name not in DETECTOR_NAMES]
        if unknown:
            raise ValueError("Unknown detector names: " + ", ".join(unknown))
        object.__setattr__(self, "detector_names", detectors)

        if isinstance(self.measurement_columns, str):
            raise ValueError("measurement_columns must be a sequence of column names.")
        columns = tuple(self.measurement_columns)
        if not columns:
            raise ValueError("measurement_columns must not be empty.")
        if any(not isinstance(column, str) for column in columns):
            raise ValueError("measurement_columns must contain only strings.")
        if len(set(columns)) != len(columns):
            raise ValueError("measurement_columns must not contain duplicates.")
        unknown_columns = [column for column in columns if column not in MEASUREMENT_COLUMNS]
        if unknown_columns:
            raise ValueError("Unknown measurement columns: " + ", ".join(unknown_columns))
        object.__setattr__(self, "measurement_columns", columns)

        if (
            isinstance(self.train_fraction, bool)
            or not isinstance(self.train_fraction, Real)
            or not math.isfinite(float(self.train_fraction))
            or not 0 < float(self.train_fraction) < 1
        ):
            raise ValueError("train_fraction must be a finite fraction between 0 and 1.")
        object.__setattr__(self, "train_fraction", float(self.train_fraction))
        if (
            isinstance(self.iqr_multiplier, bool)
            or not isinstance(self.iqr_multiplier, Real)
            or not math.isfinite(float(self.iqr_multiplier))
            or self.iqr_multiplier < 0
        ):
            raise ValueError("iqr_multiplier must be a finite non-negative number.")
        object.__setattr__(self, "iqr_multiplier", float(self.iqr_multiplier))
        for value, name in (
            (self.rolling_window_size, "rolling_window_size"),
            (self.rolling_minimum_history, "rolling_minimum_history"),
            (self.time_of_day_minimum_history, "time_of_day_minimum_history"),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if self.rolling_minimum_history > self.rolling_window_size:
            raise ValueError("rolling_minimum_history must not exceed rolling_window_size.")
        for value, name in (
            (self.rolling_threshold, "rolling_threshold"),
            (self.time_of_day_threshold, "time_of_day_threshold"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number.")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.isolation_forest_config, IsolationForestConfig):
            raise TypeError("isolation_forest_config must be an IsolationForestConfig.")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean.")
        for name in ("input_path", "results_path", "summary_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())


@dataclass(frozen=True)
class PipelineRunResult:
    """Sparse detection output and execution summary for one run."""

    results: pd.DataFrame
    summary: dict[str, Any]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_label_column(column: str) -> bool:
    normalized = column.casefold()
    return normalized in _LABEL_COLUMNS or normalized.endswith(
        ("_label", "_target", "_ground_truth")
    )


def _prepare_frame(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Input dataset contains no observations.")
    if not frame.columns.is_unique or any(not isinstance(column, str) for column in frame):
        raise ValueError("Input dataset must have unique string column names.")
    label_columns = [column for column in frame if _is_label_column(column)]
    if label_columns:
        raise ValueError(
            "Inference input must not contain ground-truth or target columns: "
            + ", ".join(label_columns)
        )

    required = {"timestamp", *MEASUREMENT_COLUMNS}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError("Input dataset is missing required columns: " + ", ".join(missing_columns))

    ordered = frame.copy()
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        count = int(timestamps.isna().sum())
        raise ValueError(f"Input dataset has {count} missing or invalid timestamps.")
    if timestamps.duplicated().any():
        raise ValueError("Input dataset contains duplicate timestamps.")
    ordered["timestamp"] = timestamps

    for column in MEASUREMENT_COLUMNS:
        values = pd.to_numeric(ordered[column], errors="coerce")
        invalid = ordered[column].notna() & values.isna()
        if invalid.any():
            raise ValueError(
                f"Measurement column '{column}' contains {int(invalid.sum())} non-numeric values."
            )
        ordered[column] = values

    return ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_parquet_dataset(path: str | Path = DEFAULT_PROCESSED_DATASET_PATH) -> pd.DataFrame:
    """Load the configured processed dataset from Parquet."""
    source = _resolve_path(Path(path).expanduser())
    if not source.is_file():
        raise FileNotFoundError(f"Processed Parquet dataset was not found: {source}")
    try:
        frame = pd.read_parquet(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read Parquet dataset '{source}': {exc}") from exc
    logger.info("Loaded %d rows from Parquet dataset %s.", len(frame), source)
    return frame


def load_postgresql_dataset(
    meter_identifier: str,
    *,
    session_factory: SessionFactory | None = None,
    page_size: int = 10_000,
) -> pd.DataFrame:
    """Load a meter's ordered readings in bounded database pages.

    The returned DataFrame is materialized in memory to support historical
    features and causal detectors. Page reads are bounded; the complete series
    is still required by this pipeline's current batch execution model.
    """
    if not meter_identifier.strip():
        raise ValueError("meter_identifier must not be empty.")
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= MAX_PAGE_SIZE
    ):
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")

    factory = session_factory or create_session_factory()
    meters = MeterRepository()
    readings = ReadingRepository()
    records: list[dict[str, Any]] = []
    with factory() as session:
        meter = meters.get_by_identifier(session, meter_identifier)
        if meter is None:
            raise FileNotFoundError(f"Meter '{meter_identifier}' was not found in PostgreSQL.")
        count = readings.count_for_meter(session, meter.id)
        for offset in range(0, count, page_size):
            page = readings.get_for_meter(
                session,
                meter.id,
                limit=page_size,
                offset=offset,
            )
            records.extend(
                {
                    "timestamp": row.timestamp,
                    **{column: getattr(row, column) for column in MEASUREMENT_COLUMNS},
                }
                for row in page
            )
    if not records:
        raise ValueError(f"Meter '{meter_identifier}' has no stored readings.")
    frame = pd.DataFrame.from_records(records)
    logger.info(
        "Loaded %d database readings for meter %s.",
        len(frame),
        meter_identifier,
    )
    return frame


def _feature_measurements(config: PipelineConfig) -> tuple[str, ...]:
    if config.isolation_forest_config.feature_columns is None:
        return MEASUREMENT_COLUMNS
    requested = set(config.measurement_columns)
    for feature in config.isolation_forest_config.feature_columns:
        measurement = feature.split("__", maxsplit=1)[0]
        if measurement in MEASUREMENT_COLUMNS and measurement != feature:
            requested.add(measurement)
    return tuple(column for column in MEASUREMENT_COLUMNS if column in requested)


def _feature_windows(config: PipelineConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    requested = config.isolation_forest_config.feature_columns
    if requested is None:
        return (1, 5, 15, 60), (60,)
    lag_periods = {1}
    rolling_windows = {60}
    for feature in requested:
        parts = feature.rsplit("__", maxsplit=1)
        if len(parts) != 2:
            continue
        kind, suffix = parts
        try:
            period = int(suffix)
        except ValueError:
            continue
        if kind.endswith("_lag"):
            lag_periods.add(period)
        elif kind.endswith(("_rolling_mean", "_rolling_std", "_rolling_min", "_rolling_max")):
            rolling_windows.add(period)
        elif kind.endswith("_deviation_from_rolling_mean"):
            rolling_windows.add(period)
    return tuple(sorted(lag_periods)), tuple(sorted(rolling_windows))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metadata_text(metadata: dict[str, Any]) -> str:
    return json.dumps(_jsonable(metadata), allow_nan=False, sort_keys=True, separators=(",", ":"))


def _make_result_records(
    *,
    run_id: str,
    meter_identifier: str,
    heldout: pd.DataFrame,
    detector_name: str,
    predictions: np.ndarray,
    available: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unavailable_reasons = metadata.pop("_unavailable_reasons", {})
    row_metadata = {
        "detector_summary_key": detector_name,
        "prediction_semantics": metadata["prediction_semantics"],
        "score_definition": metadata["score_definition"],
    }
    for position in range(len(heldout)):
        prediction_available = bool(available[position])
        is_anomaly = bool(predictions[position]) if prediction_available else None
        if prediction_available and not is_anomaly:
            continue
        reading = heldout.iloc[position]
        record: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": reading["timestamp"],
            "meter_identifier": meter_identifier,
        }
        for column in MEASUREMENT_COLUMNS:
            value = reading[column]
            record[column] = None if pd.isna(value) else value
        record.update(
            {
                "detector_name": detector_name,
                "anomaly_prediction": is_anomaly,
                "prediction_available": prediction_available,
                "anomaly_score": (
                    float(scores[position])
                    if prediction_available and math.isfinite(float(scores[position]))
                    else None
                ),
                "anomaly_category": None,
                "result_kind": "anomaly" if is_anomaly else "unavailable",
                "detector_metadata": _metadata_text(row_metadata),
            }
        )
        rows.append(record)

    summary = {
        "evaluated_observations": len(heldout),
        "available_predictions": int(available.sum()),
        "unavailable_predictions": int((~available).sum()),
        "coverage": float(available.mean()) if len(available) else 0.0,
        "anomaly_predictions": int(np.count_nonzero(predictions & available)),
        "normal_predictions_omitted_from_csv": int(np.count_nonzero(~predictions & available)),
        "unavailable_reasons": unavailable_reasons,
    }
    return rows, summary


def _empty_score_array(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float64)


def _run_global_iqr(
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    detector = GlobalIQRDetector(
        columns=config.measurement_columns,
        iqr_multiplier=config.iqr_multiplier,
    ).fit(training)
    detections = detector.predict(heldout)
    size = len(heldout)
    predictions = np.zeros(size, dtype=bool)
    available = np.zeros(size, dtype=bool)
    scores = _empty_score_array(size)
    any_finite_measurement = np.zeros(size, dtype=bool)
    threshold_bounds: dict[str, dict[str, float] | None] = {}

    for column in config.measurement_columns:
        values = pd.to_numeric(heldout[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.notna().to_numpy()
        any_finite_measurement |= finite
        bounds = detector.bounds_[column]
        if bounds is None:
            threshold_bounds[column] = None
            continue
        lower, upper, _ = bounds
        threshold_bounds[column] = {"lower": lower, "upper": upper}
        valid = finite
        available |= valid
        flagged = detections.loc[
            detections["column"].eq(column),
            "row_index",
        ].to_numpy(dtype=np.int64)
        predictions[flagged] = True
        iqr = (upper - lower) / (2 * detector.iqr_multiplier) if detector.iqr_multiplier else 0
        if iqr > 0:
            values_array = values.to_numpy(dtype=np.float64, na_value=np.nan)
            distance = np.maximum(values_array - upper, lower - values_array)
            column_scores = np.maximum(distance, 0) / iqr
            scores[flagged] = np.fmax(scores[flagged], column_scores[flagged])
    reasons = {
        "missing_measurement": int((~any_finite_measurement).sum()),
        "unfitted_threshold": int((any_finite_measurement & ~available).sum()),
    }
    return (
        predictions,
        available,
        scores,
        {
            "columns": list(config.measurement_columns),
            "iqr_multiplier": detector.iqr_multiplier,
            "threshold_fit_partition": "chronological_training",
            "training_rows": len(training),
            "threshold_bounds": threshold_bounds,
            "prediction_semantics": "exploratory_suspicious_outlier_flag",
            "score_definition": "distance beyond the IQR fence divided by the fitted IQR",
            "_unavailable_reasons": reasons,
        },
    )


def _run_statistical_detector(
    name: str,
    frame: pd.DataFrame,
    train_end: int,
    heldout: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if name == "rolling_mean_std":
        options = {
            "window_size": config.rolling_window_size,
            "minimum_history": config.rolling_minimum_history,
            "threshold": config.rolling_threshold,
        }
        detections = detect_rolling_outliers(
            frame,
            columns=config.measurement_columns,
            **options,
        )
    else:
        options = {
            "minimum_history": config.time_of_day_minimum_history,
            "threshold": config.time_of_day_threshold,
            "group_by_day_of_week": True,
        }
        detections = detect_time_of_day_outliers(
            frame,
            columns=config.measurement_columns,
            **options,
        )
    detections = detections.loc[detections["row_index"] >= train_end]
    size = len(heldout)
    predictions = np.zeros(size, dtype=bool)
    available = np.zeros(size, dtype=bool)
    scores = _empty_score_array(size)
    any_finite_measurement = np.zeros(size, dtype=bool)

    for column in config.measurement_columns:
        column_results = detections.loc[detections["column"].eq(column)]
        if column_results.empty:
            continue
        positions = column_results["row_index"].to_numpy(dtype=np.int64) - train_end
        values = pd.to_numeric(column_results["value"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        means = pd.to_numeric(column_results["baseline_mean"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        deviations = pd.to_numeric(
            column_results["baseline_std"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(values) & np.isfinite(means) & np.isfinite(deviations)
        any_finite_measurement[positions[np.isfinite(values)]] = True
        available[positions[valid]] = True
        flags = column_results["is_suspicious_outlier"].to_numpy(dtype=bool)
        predictions[positions[flags & valid]] = True
        nonzero_std = valid & (deviations > 0)
        column_scores = np.full(len(column_results), np.nan, dtype=np.float64)
        column_scores[nonzero_std] = (
            np.abs(values[nonzero_std] - means[nonzero_std]) / deviations[nonzero_std]
        )
        flagged = flags & valid
        scores[positions[flagged]] = np.fmax(
            scores[positions[flagged]],
            column_scores[flagged],
        )
    unavailable_reasons = {
        "missing_measurement": int(np.count_nonzero(~available & ~any_finite_measurement)),
        "insufficient_history": int(np.count_nonzero(~available & any_finite_measurement)),
    }
    metadata: dict[str, Any] = {
        "columns": list(config.measurement_columns),
        "training_rows": train_end,
        "prediction_semantics": "exploratory_suspicious_outlier_flag",
        "score_definition": (
            "absolute distance from the historical mean in population standard deviations; "
            "null when the baseline standard deviation is zero"
        ),
        "_unavailable_reasons": unavailable_reasons,
        **options,
    }
    return predictions, available, scores, metadata


def _run_isolation_forest(
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    detector = IsolationForestDetector(config.isolation_forest_config).fit(training)
    predicted_frame = detector.predict(heldout)
    selected_features = detector.feature_columns_
    feature_values = pd.DataFrame(index=heldout.index)
    for column in selected_features:
        values = pd.to_numeric(heldout[column], errors="coerce")
        feature_values[column] = values.replace([np.inf, -np.inf], np.nan)
    available = feature_values.notna().any(axis=1).to_numpy(dtype=bool)
    predictions = predicted_frame["isolation_forest_anomaly_label"].to_numpy(
        dtype=bool,
        copy=True,
    )
    scores = predicted_frame["isolation_forest_anomaly_score"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    metadata = {
        "configuration": asdict(config.isolation_forest_config),
        "feature_columns": list(selected_features),
        "training_rows": detector.n_training_rows_,
        "training_partition": "chronological_training",
        "prediction_semantics": "unsupervised_model_decision_not_verified_label",
        "score_definition": "negative Isolation Forest score_samples; larger is more unusual",
        "_unavailable_reasons": {
            "all_model_features_missing_or_non_finite": int((~available).sum())
        },
    }
    return predictions, available, scores, metadata


def run_anomaly_pipeline(
    frame: pd.DataFrame,
    *,
    config: PipelineConfig | None = None,
    meter_identifier: str | None = None,
) -> PipelineRunResult:
    """Run selected detectors on a chronological held-out partition.

    Input labels are rejected rather than being made available to inference.
    The output CSV is sparse: it contains positive anomaly predictions and
    unavailable predictions; available normal predictions are summarized but
    omitted. Statistical warm-up and missing-input predictions remain explicit.
    """
    selected_config = config or PipelineConfig()
    selected_meter = (
        selected_config.meter_identifier if meter_identifier is None else meter_identifier
    )
    if not isinstance(selected_meter, str) or not selected_meter.strip():
        raise ValueError("meter_identifier must be a non-empty string.")

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    source_rows = _prepare_frame(frame, selected_config)
    train_end = int(math.floor(len(source_rows) * selected_config.train_fraction))
    if train_end < 1 or train_end >= len(source_rows):
        raise ValueError(
            "train_fraction must produce non-empty chronological training and held-out sets."
        )
    if "isolation_forest" in selected_config.detector_names and train_end < 2:
        raise ValueError("Isolation Forest requires at least two training observations.")

    detector_frame = source_rows
    if "isolation_forest" in selected_config.detector_names:
        lag_periods, rolling_windows = _feature_windows(selected_config)
        detector_frame = engineer_features(
            source_rows,
            columns=_feature_measurements(selected_config),
            lag_periods=lag_periods,
            rolling_windows=rolling_windows,
        )

    training = detector_frame.iloc[:train_end].copy()
    heldout = detector_frame.iloc[train_end:].copy()
    run_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []
    detector_summaries: dict[str, Any] = {}

    runners = {
        "global_iqr": lambda: _run_global_iqr(training, heldout, selected_config),
        "rolling_mean_std": lambda: _run_statistical_detector(
            "rolling_mean_std",
            detector_frame,
            train_end,
            heldout,
            selected_config,
        ),
        "time_of_day": lambda: _run_statistical_detector(
            "time_of_day",
            detector_frame,
            train_end,
            heldout,
            selected_config,
        ),
        "isolation_forest": lambda: _run_isolation_forest(
            training,
            heldout,
            selected_config,
        ),
    }
    for detector_name in selected_config.detector_names:
        logger.info("Running detector %s on %d held-out rows.", detector_name, len(heldout))
        predictions, available, scores, metadata = runners[detector_name]()
        detector_rows, detector_summary = _make_result_records(
            run_id=run_id,
            meter_identifier=selected_meter,
            heldout=heldout,
            detector_name=detector_name,
            predictions=predictions,
            available=available,
            scores=scores,
            metadata=metadata,
        )
        results.extend(detector_rows)
        detector_summaries[detector_name] = {
            **detector_summary,
            "configuration": _jsonable(
                {key: value for key, value in metadata.items() if not key.startswith("_")}
            ),
        }

    result_frame = pd.DataFrame.from_records(results, columns=_RESULT_COLUMNS)
    if not result_frame.empty:
        result_frame = result_frame.sort_values(
            ["timestamp", "detector_name"],
            kind="stable",
        ).reset_index(drop=True)
    completed_at = datetime.now(timezone.utc)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "execution_seconds": round(time.perf_counter() - started, 6),
        "source": "dataframe",
        "meter_identifier": selected_meter,
        "input_rows": len(source_rows),
        "training_rows": len(training),
        "heldout_rows": len(heldout),
        "training_start": source_rows["timestamp"].iloc[0].isoformat(),
        "training_end": source_rows["timestamp"].iloc[train_end - 1].isoformat(),
        "heldout_start": source_rows["timestamp"].iloc[train_end].isoformat(),
        "heldout_end": source_rows["timestamp"].iloc[-1].isoformat(),
        "heldout_policy": (
            "Only chronological held-out rows are emitted. Training observations fit "
            "learned detectors; causal statistical detectors may use prior training "
            "observations as history. No synthetic labels are accepted or used."
        ),
        "prediction_semantics": (
            "Detector outputs are exploratory flags or unsupervised model decisions, "
            "not verified anomaly ground truth."
        ),
        "result_output_policy": (
            "CSV contains anomaly-positive and unavailable results. Available normal "
            "predictions are omitted and counted per detector."
        ),
        "detectors": detector_summaries,
        "result_rows": len(result_frame),
        "configuration": {
            "source": selected_config.source,
            "detectors": list(selected_config.detector_names),
            "measurement_columns": list(selected_config.measurement_columns),
            "train_fraction": selected_config.train_fraction,
            "iqr_multiplier": selected_config.iqr_multiplier,
        },
        "limitations": [
            "Global IQR uses training-period quartiles and does not model time-of-day "
            "or seasonal consumption patterns.",
            "Synthetic labels are not used in inference; this run does not estimate "
            "real-world accuracy.",
            "The database reader materializes all ordered readings in memory.",
            "Sparse CSV omits available normal predictions; detector counts in this "
            "summary include them.",
        ],
    }
    logger.info(
        "Detection run %s completed: input=%d held-out=%d result rows=%d in %.3f seconds.",
        run_id,
        len(source_rows),
        len(heldout),
        len(result_frame),
        summary["execution_seconds"],
    )
    return PipelineRunResult(result_frame, summary)


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Detection results already exist: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            temporary_path.rename(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def execute_anomaly_pipeline(
    config: PipelineConfig | None = None,
    *,
    session_factory: SessionFactory | None = None,
) -> PipelineRunResult:
    """Load configured input, run detectors, and safely write CSV and JSON outputs."""
    selected_config = config or PipelineConfig()
    results_path = _resolve_path(selected_config.results_path)
    summary_path = _resolve_path(selected_config.summary_path)
    input_path = _resolve_path(selected_config.input_path)
    if results_path.resolve() == summary_path.resolve():
        raise ValueError("CSV results and JSON summary paths must be different.")
    if selected_config.source == "parquet" and input_path.resolve() in (
        results_path.resolve(),
        summary_path.resolve(),
    ):
        raise ValueError("Output paths must not overwrite the input dataset.")
    if not selected_config.overwrite:
        existing = [path for path in (results_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
            )

    run_started = time.perf_counter()
    run_started_at = datetime.now(timezone.utc)
    if selected_config.source == "parquet":
        frame = load_parquet_dataset(input_path)
    else:
        frame = load_postgresql_dataset(
            selected_config.meter_identifier,
            session_factory=session_factory,
        )
    run = run_anomaly_pipeline(
        frame,
        config=selected_config,
        meter_identifier=selected_config.meter_identifier,
    )
    _write_csv(run.results, results_path, overwrite=selected_config.overwrite)
    run.summary["started_at"] = run_started_at.isoformat()
    run.summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    run.summary["source"] = selected_config.source
    run.summary["input_path"] = str(input_path) if selected_config.source == "parquet" else None
    run.summary["results_path"] = str(results_path)
    run.summary["result_file_size_bytes"] = results_path.stat().st_size
    run.summary["summary_path"] = str(summary_path)
    run.summary["execution_seconds"] = round(time.perf_counter() - run_started, 6)
    from smartmeter.anomaly_detection.evaluation import write_evaluation_report

    write_evaluation_report(
        run.summary,
        summary_path,
        overwrite=selected_config.overwrite,
    )
    logger.info(
        "Wrote %d sparse detection results to %s (%d bytes) and summary to %s.",
        len(run.results),
        results_path,
        results_path.stat().st_size,
        summary_path,
    )
    return run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chronological SmartMeter anomaly detection on Parquet or PostgreSQL."
    )
    parser.add_argument("--source", choices=("parquet", "postgresql"), default="parquet")
    parser.add_argument("--input", type=Path, default=DEFAULT_PROCESSED_DATASET_PATH)
    parser.add_argument("--results", type=Path, default=REPORTS_DIR / "anomaly_detections.csv")
    parser.add_argument("--summary", type=Path, default=REPORTS_DIR / "anomaly_run.json")
    parser.add_argument("--meter-identifier", default="uci_household_1")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTOR_NAMES,
        default=list(DETECTOR_NAMES),
    )
    parser.add_argument(
        "--measurement-columns",
        nargs="+",
        choices=MEASUREMENT_COLUMNS,
        default=["global_active_power"],
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--rolling-window-size", type=int, default=60)
    parser.add_argument("--rolling-minimum-history", type=int, default=30)
    parser.add_argument("--rolling-threshold", type=float, default=3.0)
    parser.add_argument("--time-of-day-minimum-history", type=int, default=4)
    parser.add_argument("--time-of-day-threshold", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the anomaly pipeline."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    configure_logging()
    config = PipelineConfig(
        source=arguments.source,
        input_path=arguments.input,
        results_path=arguments.results,
        summary_path=arguments.summary,
        meter_identifier=arguments.meter_identifier,
        detector_names=tuple(arguments.detectors),
        measurement_columns=tuple(arguments.measurement_columns),
        train_fraction=arguments.train_fraction,
        iqr_multiplier=arguments.iqr_multiplier,
        rolling_window_size=arguments.rolling_window_size,
        rolling_minimum_history=arguments.rolling_minimum_history,
        rolling_threshold=arguments.rolling_threshold,
        time_of_day_minimum_history=arguments.time_of_day_minimum_history,
        time_of_day_threshold=arguments.time_of_day_threshold,
        overwrite=arguments.overwrite,
    )
    try:
        execute_anomaly_pipeline(config)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except SQLAlchemyError:
        logger.exception("PostgreSQL anomaly-pipeline input failed.")
        return 1
    return 0
