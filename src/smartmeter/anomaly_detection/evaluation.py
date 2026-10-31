"""Leakage-aware comparison utilities for statistical and ML anomaly detectors."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from smartmeter.anomaly_detection.features import engineer_features
from smartmeter.anomaly_detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)
from smartmeter.anomaly_detection.splitting import (
    TemporalDataSplit,
    chronological_train_validation_test_split,
)
from smartmeter.anomaly_detection.statistical import (
    DEFAULT_MEASUREMENT_COLUMNS,
    GlobalIQRDetector,
    detect_rolling_outliers,
    detect_time_of_day_outliers,
)
from smartmeter.anomaly_detection.synthetic_data import (
    SyntheticInjectionConfig,
    inject_synthetic_anomalies,
)

logger = logging.getLogger(__name__)

SYNTHETIC_CATEGORIES = (
    "consumption_spike",
    "unusually_low_consumption",
    "sustained_abnormal_consumption",
    "voltage_anomaly",
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


@dataclass(frozen=True)
class EvaluationConfig:
    """Reproducible configuration for chronological detector comparison."""

    dataset_identifier: str = "unspecified"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    synthetic_config: SyntheticInjectionConfig = field(default_factory=SyntheticInjectionConfig)
    isolation_forest_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    contamination_candidates: tuple[float, ...] = ()
    rolling_window_size: int = 60
    rolling_minimum_history: int = 30
    rolling_threshold: float = 3.0
    time_of_day_minimum_history: int = 4
    time_of_day_threshold: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_identifier, str) or not self.dataset_identifier.strip():
            raise ValueError("dataset_identifier must be a non-empty string.")
        if (
            isinstance(self.rolling_window_size, bool)
            or not isinstance(self.rolling_window_size, int)
            or self.rolling_window_size < 1
        ):
            raise ValueError("rolling_window_size must be a positive integer.")
        if (
            isinstance(self.rolling_minimum_history, bool)
            or not isinstance(self.rolling_minimum_history, int)
            or not 1 <= self.rolling_minimum_history <= self.rolling_window_size
        ):
            raise ValueError("rolling_minimum_history must be between 1 and rolling_window_size.")
        if (
            isinstance(self.time_of_day_minimum_history, bool)
            or not isinstance(self.time_of_day_minimum_history, int)
            or self.time_of_day_minimum_history < 1
        ):
            raise ValueError("time_of_day_minimum_history must be a positive integer.")
        _validate_threshold(self.rolling_threshold, "rolling_threshold")
        _validate_threshold(self.time_of_day_threshold, "time_of_day_threshold")
        candidates = tuple(self.contamination_candidates)
        for value in candidates:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 0.5
            ):
                raise ValueError("contamination_candidates must contain finite values in (0, 0.5].")
        if len(set(float(value) for value in candidates)) != len(candidates):
            raise ValueError("contamination_candidates must not contain duplicates.")
        object.__setattr__(self, "contamination_candidates", candidates)


def _validate_threshold(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number.")


def _binary_array(values: Sequence[Any] | pd.Series, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if pd.isna(array).any():
        raise ValueError(f"{name} must not contain missing values.")
    if not all(value in (0, 1, False, True) for value in array):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return np.asarray(array, dtype=np.uint8)


def calculate_binary_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
) -> dict[str, Any]:
    """Calculate binary metrics with class order [normal=0, anomaly=1].

    Undefined precision, recall, F1, false-positive rate, or specificity is
    represented as 0.0 (zero-division policy), never NaN or infinity.
    """
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred Series must have identical ordered indexes.")
    truth = _binary_array(y_true, "y_true")
    prediction = _binary_array(y_pred, "y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have equal lengths.")

    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )
    return {
        "class_order": [{"label": 0, "name": "normal"}, {"label": 1, "name": "anomaly"}],
        "confusion_matrix": matrix.astype(int).tolist(),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1_score": float(f1_score(truth, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "false_positive_rate": _safe_ratio(false_positive, false_positive + true_negative),
        "specificity": _safe_ratio(true_negative, true_negative + false_positive),
        "zero_division_policy": "0.0 when the metric denominator is zero",
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    *,
    prediction_available: Sequence[bool] | pd.Series | None = None,
    categories: Sequence[Any] | pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate row-aligned predictions while excluding unavailable outputs explicitly.

    Missing ground-truth labels are an error. Missing predictions count as
    unavailable rather than normal. When both labels and predictions are Series,
    their ordered indexes must match exactly (duplicate index labels are supported
    because alignment remains positional).
    """
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred Series must have identical ordered indexes.")
    truth = _binary_array(y_true, "y_true")
    raw_prediction = np.asarray(y_pred, dtype=object)
    if raw_prediction.ndim != 1 or len(raw_prediction) != len(truth):
        raise ValueError("y_pred must be one-dimensional and aligned with y_true.")

    if prediction_available is None:
        available = ~pd.isna(raw_prediction)
    else:
        raw_available = np.asarray(prediction_available, dtype=object)
        if raw_available.ndim != 1 or len(raw_available) != len(truth):
            raise ValueError("prediction_available must align with y_true.")
        if pd.isna(raw_available).any() or not all(
            value in (0, 1, False, True) for value in raw_available
        ):
            raise ValueError("prediction_available must contain only boolean values.")
        available = np.asarray(raw_available, dtype=bool) & ~pd.isna(raw_prediction)

    invalid_prediction = [
        value for value in raw_prediction[available] if value not in (0, 1, False, True)
    ]
    if invalid_prediction:
        raise ValueError("Available predictions must contain only binary values 0 and 1.")

    aligned_categories: np.ndarray | None = None
    if categories is not None:
        if isinstance(y_true, pd.Series) and isinstance(categories, pd.Series):
            if not y_true.index.equals(categories.index):
                raise ValueError("categories must have the same ordered index as y_true.")
        aligned_categories = np.asarray(categories, dtype=object)
        if aligned_categories.ndim != 1 or len(aligned_categories) != len(truth):
            raise ValueError("categories must align with y_true.")
        if pd.isna(aligned_categories).any():
            raise ValueError("categories must not contain missing values.")
        if not np.array_equal(aligned_categories == "normal", truth == 0):
            raise ValueError(
                "categories must label every normal observation 'normal' and every "
                "anomaly with a non-'normal' category."
            )

    available_count = int(available.sum())
    metrics = (
        calculate_binary_metrics(truth[available], raw_prediction[available])
        if available_count
        else None
    )
    result: dict[str, Any] = {
        "total_observations": len(truth),
        "available_observations": available_count,
        "excluded_observations": len(truth) - available_count,
        "coverage": float(available_count / len(truth)),
        "metrics": metrics,
        "exclusion_policy": "Unavailable predictions are excluded, never imputed as normal.",
    }
    if aligned_categories is not None:
        result["category_metrics"] = _category_metrics(
            truth,
            raw_prediction,
            available,
            aligned_categories,
        )
    return result


def _category_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    available: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    normalized_predictions = np.zeros(len(predictions), dtype=np.uint8)
    normalized_predictions[available] = np.asarray(predictions[available], dtype=np.uint8)
    result: dict[str, Any] = {}
    for category in SYNTHETIC_CATEGORIES:
        target_category = categories == category
        other_anomaly = (categories != "normal") & (categories != category)
        normal = categories == "normal"
        category_truth = target_category.astype(np.uint8)
        category_evaluation = evaluate_predictions(
            category_truth,
            normalized_predictions,
            prediction_available=available,
        )
        if category_evaluation["metrics"] is not None:
            category_evaluation["metrics"]["class_order"] = [
                {"label": 0, "name": "non_target"},
                {"label": 1, "name": category},
            ]
        result[category] = {
            **category_evaluation,
            "target_category_support": int((target_category & available).sum()),
            "other_anomaly_support": int((other_anomaly & available).sum()),
            "normal_support": int((normal & available).sum()),
            "other_anomaly_predicted_positive": int(
                (other_anomaly & available & (normalized_predictions == 1)).sum()
            ),
            "normal_predicted_positive": int(
                (normal & available & (normalized_predictions == 1)).sum()
            ),
            "negative_class_note": (
                "Other anomaly categories are non-target negatives, separately counted; "
                "they are not treated as normal observations."
            ),
        }
    known_category = np.isin(categories, SYNTHETIC_CATEGORIES) | (categories == "normal")
    unknown_categories = categories[~known_category]
    if len(unknown_categories):
        result["unrecognized_category_counts"] = {
            str(category): int(np.count_nonzero(unknown_categories == category))
            for category in sorted(set(unknown_categories.tolist()), key=str)
        }
    return result


def run_anomaly_evaluation(
    frame: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Run validation tuning summaries and final test comparison on chronological data.

    Feature engineering runs once on the original series and uses only past values.
    The chronological splitter creates disjoint train/validation/test partitions.
    Synthetic anomalies are injected only into copies of validation and test rows.
    Isolation Forest preprocessing and fitting use training rows only. Optional
    contamination candidates are evaluated on validation only; the configured
    Isolation Forest is evaluated once on test, without automatic model selection.

    Global IQR bounds are fitted from training data with the same quartile rule as
    the existing exploratory function, then applied unchanged to held-out rows.
    """
    selected = config or EvaluationConfig()
    _validate_evaluation_frame(frame)
    started = time.perf_counter()
    source_bytes = int(frame.memory_usage(index=True, deep=True).sum())

    model_config = selected.isolation_forest_config
    if model_config.feature_columns is None:
        model_config = replace(model_config, feature_columns=DEFAULT_MODEL_FEATURES)
    engineered = engineer_features(
        frame,
        columns=("global_active_power", "voltage"),
    )
    missing_model_features = [
        column for column in model_config.feature_columns or () if column not in engineered
    ]
    if missing_model_features:
        raise ValueError(
            "Configured Isolation Forest features are not generated by this evaluator: "
            + ", ".join(missing_model_features)
        )
    split = chronological_train_validation_test_split(
        engineered,
        train_fraction=selected.train_fraction,
        validation_fraction=selected.validation_fraction,
    )
    validation_injected = _inject_eval_copy(split.validation, selected.synthetic_config)
    test_injected = _inject_eval_copy(split.test, selected.synthetic_config)
    validation_evaluation = _evaluate_partition(
        split,
        validation_injected,
        partition="validation",
        config=selected,
        model_config=model_config,
    )
    test_evaluation = _evaluate_partition(
        split,
        test_injected,
        partition="test",
        config=selected,
        model_config=model_config,
    )

    if selected.contamination_candidates:
        validation_evaluation["isolation_forest_contamination_sweep"] = (
            _evaluate_contamination_candidates(
                split.train,
                validation_injected,
                selected.contamination_candidates,
                model_config,
            )
        )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_identifier": selected.dataset_identifier,
        "synthetic_ground_truth_limitation": (
            "Labels represent controlled synthetic injections only; they are not "
            "verified real-world anomaly ground truth."
        ),
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "split_method": "chronological, sorted by timestamp, non-overlapping",
        "synthetic_injection_config": _json_safe(asdict(selected.synthetic_config)),
        "feature_engineering": {
            "method": "engineer_features on original chronological observations",
            "measurement_columns": ["global_active_power", "voltage"],
            "past_only": True,
            "model_feature_columns": list(model_config.feature_columns or ()),
        },
        "isolation_forest_training": {
            "training_rows": len(split.train),
            "fit_partitions": ["train"],
            "labels_used_for_fit": False,
            "validation_used_for_imputation_or_fit": False,
            "test_used_for_imputation_or_fit": False,
            "test_model_configuration": _json_safe(asdict(model_config)),
        },
        "comparison_policy": {
            "shared_test_rows": (
                "A separate intersection subset is reported for rows where every "
                "detector has an available prediction."
            ),
            "unavailable_predictions": (
                "Excluded from that detector's metrics, counted in coverage and "
                "exclusion reasons, never converted to normal."
            ),
            "global_iqr": (
                "Train-fitted quartile bounds using the same 1.5xIQR threshold rule as "
                "the existing exploratory detector; held-out rows do not fit thresholds."
            ),
            "category_analysis": (
                "One-vs-rest by synthetic category. Other anomaly categories are "
                "separately counted non-target negatives, not treated as normal."
            ),
        },
        "validation": validation_evaluation,
        "test": test_evaluation,
        "performance": {
            "evaluation_runtime_seconds": float(elapsed),
            "input_rows": len(frame),
            "source_frame_memory_bytes": source_bytes,
            "memory_note": (
                "Source DataFrame memory is measured with pandas memory_usage(deep=True); "
                "this is not peak process memory."
            ),
        },
        "limitations": [
            "Synthetic labels do not establish real-world detection accuracy.",
            "Global IQR is not time- or season-aware and its training distribution may "
            "not represent later consumption patterns.",
            "Rolling and time-of-day detectors require historical warm-up; coverage "
            "varies by detector and selected measurement availability.",
            "Isolation Forest does not model temporal ordering independently of supplied "
            "past-only features.",
            "No model is selected automatically from test performance.",
        ],
    }
    logger.info(
        "Completed anomaly evaluation for %d test rows in %.3f seconds.",
        len(split.test),
        elapsed,
    )
    return _json_safe(report)


def _validate_evaluation_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("frame must contain observations.")
    if not frame.columns.is_unique or any(not isinstance(name, str) for name in frame.columns):
        raise ValueError("frame must have unique string column names.")
    required = {"timestamp", *DEFAULT_MEASUREMENT_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("frame is missing required evaluation columns: " + ", ".join(missing))
    label_columns = [
        column
        for column in frame.columns
        if column.casefold() in _LABEL_COLUMNS
        or column.casefold().endswith(("_label", "_target", "_ground_truth"))
    ]
    if label_columns:
        raise ValueError(
            "Input frame must not contain ground-truth or target columns: "
            + ", ".join(label_columns)
        )


def _inject_eval_copy(
    partition: pd.DataFrame,
    synthetic_config: SyntheticInjectionConfig,
) -> pd.DataFrame:
    required = ["timestamp", *DEFAULT_MEASUREMENT_COLUMNS]
    injected = inject_synthetic_anomalies(
        partition.loc[:, required],
        config=synthetic_config,
    )
    result = partition.copy()
    for column in DEFAULT_MEASUREMENT_COLUMNS:
        result[column] = injected[f"synthetic_{column}"].to_numpy(copy=False)
    result["synthetic_anomaly_label"] = injected["synthetic_anomaly_label"].to_numpy(copy=False)
    result["synthetic_anomaly_category"] = injected["synthetic_anomaly_category"].to_numpy(
        copy=False
    )
    return result


def _evaluate_partition(
    split: TemporalDataSplit,
    injected: pd.DataFrame,
    *,
    partition: str,
    config: EvaluationConfig,
    model_config: IsolationForestConfig,
) -> dict[str, Any]:
    truth = injected["synthetic_anomaly_label"].to_numpy(dtype=np.uint8)
    categories = injected["synthetic_anomaly_category"].fillna("normal").to_numpy(dtype=object)
    truth_frame = pd.DataFrame(
        {
            "synthetic_anomaly_label": truth,
            "synthetic_anomaly_category": categories,
        }
    )
    detector_results: dict[str, dict[str, Any]] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    availability_arrays: dict[str, np.ndarray] = {}
    detector_configurations: dict[str, Any] = {}

    iqr_detector = GlobalIQRDetector(
        columns=DEFAULT_MEASUREMENT_COLUMNS,
        iqr_multiplier=1.5,
    ).fit(split.train)
    iqr_detections = iqr_detector.predict(injected)
    usable_iqr_columns = tuple(
        column for column, bounds in iqr_detector.bounds_.items() if bounds is not None
    )
    iqr_predictions, iqr_available = _global_iqr_predictions(
        iqr_detections,
        injected,
        usable_iqr_columns,
    )
    detector_results["global_iqr"] = _summarize_detector(
        truth_frame,
        iqr_predictions,
        iqr_available,
        categories,
        configuration={
            "columns": list(DEFAULT_MEASUREMENT_COLUMNS),
            "iqr_multiplier": 1.5,
            "threshold_fit_partition": "train",
        },
        extra={
            "evaluation_mode": "train_fitted_threshold",
            "excluded_reasons": _global_iqr_exclusions(injected, iqr_available),
        },
    )
    prediction_arrays["global_iqr"] = iqr_predictions
    availability_arrays["global_iqr"] = iqr_available
    detector_configurations["global_iqr"] = {
        "columns": list(DEFAULT_MEASUREMENT_COLUMNS),
        "iqr_multiplier": 1.5,
        "threshold_fit_partition": "train",
    }

    prior_frames = [split.train]
    if partition == "test":
        prior_frames.append(split.validation)
    history = pd.concat(
        [part.loc[:, ["timestamp", *DEFAULT_MEASUREMENT_COLUMNS]] for part in prior_frames]
        + [injected.loc[:, ["timestamp", *DEFAULT_MEASUREMENT_COLUMNS]]],
        ignore_index=True,
    )
    eval_start = len(history) - len(injected)
    for name, detector, detector_configuration in (
        (
            "rolling_mean_std",
            detect_rolling_outliers,
            {
                "window_size": config.rolling_window_size,
                "minimum_history": config.rolling_minimum_history,
                "threshold": config.rolling_threshold,
            },
        ),
        (
            "time_of_day",
            detect_time_of_day_outliers,
            {
                "minimum_history": config.time_of_day_minimum_history,
                "threshold": config.time_of_day_threshold,
                "group_by_day_of_week": True,
            },
        ),
    ):
        detections = detector(
            history,
            columns=DEFAULT_MEASUREMENT_COLUMNS,
            **detector_configuration,
        )
        predictions, available = _statistical_predictions(
            detections,
            eval_start,
            len(injected),
        )
        detector_results[name] = _summarize_detector(
            truth_frame,
            predictions,
            available,
            categories,
            configuration=detector_configuration | {"columns": list(DEFAULT_MEASUREMENT_COLUMNS)},
            extra={"excluded_reasons": _statistical_exclusions(injected, available)},
        )
        prediction_arrays[name] = predictions
        availability_arrays[name] = available
        detector_configurations[name] = detector_configuration | {
            "columns": list(DEFAULT_MEASUREMENT_COLUMNS)
        }

    fitted = IsolationForestDetector(model_config).fit(split.train)
    model_predictions_frame = fitted.predict(injected)
    model_predictions = model_predictions_frame["isolation_forest_anomaly_label"].to_numpy(
        dtype=np.uint8
    )
    model_features = model_config.feature_columns or ()
    model_input_available = np.zeros(len(injected), dtype=bool)
    for column in model_features:
        values = pd.to_numeric(injected[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        model_input_available |= values.notna().to_numpy()
    detector_results["isolation_forest"] = _summarize_detector(
        truth_frame,
        model_predictions,
        model_input_available,
        categories,
        configuration=_json_safe(asdict(model_config)),
        extra={
            "excluded_reasons": {"no_finite_model_features": int((~model_input_available).sum())},
            "training_rows": fitted.n_training_rows_,
        },
    )
    prediction_arrays["isolation_forest"] = model_predictions
    availability_arrays["isolation_forest"] = model_input_available
    detector_configurations["isolation_forest"] = _json_safe(asdict(model_config))

    return {
        "partition_rows": len(injected),
        "synthetic_label_counts": _label_counts(truth, categories),
        "detectors": detector_results,
        "shared_comparison": _shared_comparison(
            truth_frame,
            prediction_arrays,
            availability_arrays,
            categories,
        ),
        "detector_configurations": detector_configurations,
    }


def _global_iqr_predictions(
    detections: pd.DataFrame,
    frame: pd.DataFrame,
    usable_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros(len(frame), dtype=np.uint8)
    if not detections.empty:
        positions = detections["row_index"].to_numpy(dtype=np.int64)
        predictions[positions] = 1

    if usable_columns:
        available = (
            frame[list(usable_columns)]
            .apply(
                lambda column: (
                    pd.to_numeric(column, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .notna()
                )
            )
            .any(axis=1)
            .to_numpy()
        )
    else:
        available = np.zeros(len(frame), dtype=bool)
    return predictions, available


def _global_iqr_exclusions(
    frame: pd.DataFrame,
    available: np.ndarray,
) -> dict[str, int]:
    has_measurement = (
        frame.loc[:, list(DEFAULT_MEASUREMENT_COLUMNS)]
        .apply(
            lambda column: (
                pd.to_numeric(column, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
            )
        )
        .any(axis=1)
        .to_numpy()
    )
    return {
        "no_finite_measurement": int((~has_measurement).sum()),
        "no_usable_global_iqr_baseline": int((has_measurement & ~available).sum()),
    }


def _statistical_predictions(
    detections: pd.DataFrame,
    evaluation_start: int,
    evaluation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros(evaluation_count, dtype=np.uint8)
    available = np.zeros(evaluation_count, dtype=bool)
    if detections.empty:
        return predictions, available
    positions = detections["row_index"].to_numpy(dtype=np.int64)
    in_evaluation = (positions >= evaluation_start) & (
        positions < evaluation_start + evaluation_count
    )
    rows = detections.loc[in_evaluation].copy()
    offsets = rows["row_index"].to_numpy(dtype=np.int64) - evaluation_start
    flags = rows["is_suspicious_outlier"].to_numpy(dtype=bool)
    baseline_available = rows["baseline_mean"].notna().to_numpy()
    current_available = rows["value"].notna().to_numpy()
    np.logical_or.at(predictions, offsets, flags.astype(np.uint8))
    np.logical_or.at(available, offsets, baseline_available & current_available)
    return predictions, available


def _statistical_exclusions(
    frame: pd.DataFrame,
    available: np.ndarray,
) -> dict[str, int]:
    has_measurement = (
        frame.loc[:, list(DEFAULT_MEASUREMENT_COLUMNS)]
        .apply(
            lambda column: (
                pd.to_numeric(column, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
            )
        )
        .any(axis=1)
        .to_numpy()
    )
    return {
        "no_finite_measurement": int((~has_measurement).sum()),
        "insufficient_history": int((has_measurement & ~available).sum()),
    }


def _summarize_detector(
    truth_frame: pd.DataFrame,
    predictions: np.ndarray,
    available: np.ndarray,
    categories: np.ndarray,
    *,
    configuration: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = evaluate_predictions(
        truth_frame["synthetic_anomaly_label"],
        predictions,
        prediction_available=available,
        categories=categories,
    )
    summary["configuration"] = configuration
    summary.update(extra or {})
    return summary


def _shared_comparison(
    truth_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    categories: np.ndarray,
) -> dict[str, Any]:
    names = tuple(predictions)
    shared = np.logical_and.reduce([availability[name] for name in names])
    detector_metrics = {
        name: evaluate_predictions(
            truth_frame["synthetic_anomaly_label"].to_numpy(),
            predictions[name],
            prediction_available=shared,
            categories=categories,
        )
        for name in names
    }
    return {
        "detector_names": list(names),
        "shared_rows": int(shared.sum()),
        "excluded_rows": int(len(shared) - shared.sum()),
        "coverage": float(shared.mean()) if len(shared) else 0.0,
        "detector_metrics": detector_metrics,
        "note": (
            "All detector metrics here use the same available observations. The global IQR "
            "baseline still estimates its threshold from the evaluation partition."
        ),
    }


def _label_counts(truth: np.ndarray, categories: np.ndarray) -> dict[str, Any]:
    counts = {
        "normal": int(np.count_nonzero(truth == 0)),
        "anomaly": int(np.count_nonzero(truth == 1)),
        "categories": {
            category: int(np.count_nonzero(categories == category))
            for category in SYNTHETIC_CATEGORIES
        },
    }
    unknown = categories[~np.isin(categories, (*SYNTHETIC_CATEGORIES, "normal"))]
    if len(unknown):
        counts["unrecognized_categories"] = {
            str(category): int(np.count_nonzero(unknown == category))
            for category in sorted(set(unknown.tolist()), key=str)
        }
    return counts


def _evaluate_contamination_candidates(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    candidates: tuple[float, ...],
    model_config: IsolationForestConfig,
) -> dict[str, Any]:
    truth = validation["synthetic_anomaly_label"].to_numpy(dtype=np.uint8)
    categories = validation["synthetic_anomaly_category"].fillna("normal").to_numpy(dtype=object)
    truth_frame = pd.DataFrame(
        {
            "synthetic_anomaly_label": truth,
            "synthetic_anomaly_category": categories,
        }
    )
    output: dict[str, Any] = {
        "selection_policy": (
            "Validation-only comparison. No candidate is selected automatically; use "
            "validation results to specify the final test configuration explicitly."
        ),
        "candidates": {},
    }
    for contamination in candidates:
        candidate_config = replace(model_config, contamination=float(contamination))
        detector = IsolationForestDetector(candidate_config).fit(training)
        predicted_frame = detector.predict(validation)
        predicted = predicted_frame["isolation_forest_anomaly_label"].to_numpy(dtype=np.uint8)
        available = np.zeros(len(validation), dtype=bool)
        for column in candidate_config.feature_columns or ():
            values = pd.to_numeric(validation[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            available |= values.notna().to_numpy()
        output["candidates"][str(float(contamination))] = _summarize_detector(
            truth_frame,
            predicted,
            available,
            categories,
            configuration=_json_safe(asdict(candidate_config)),
            extra={"training_rows": detector.n_training_rows_},
        )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("Evaluation report cannot contain NaN or infinity.")
        return normalized
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evaluation report cannot contain NaN or infinity.")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_evaluation_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write strict JSON, refusing existing reports unless overwrite is explicit."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _json_safe(report),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Evaluation report already exists: {destination}")

    if overwrite:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.write("\n")
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    else:
        try:
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.write("\n")
        except FileExistsError as exc:
            raise FileExistsError(f"Evaluation report already exists: {destination}") from exc
    logger.info("Wrote anomaly evaluation report to %s.", destination)
    return destination
