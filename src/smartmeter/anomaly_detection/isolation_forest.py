"""Isolation Forest anomaly detection for engineered smart-meter features."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

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
_NON_FEATURE_COLUMNS = {"date", "time", "timestamp"}


@dataclass(frozen=True)
class IsolationForestConfig:
    """Isolation Forest and feature-selection options."""

    n_estimators: int = 100
    max_samples: int | float | Literal["auto"] = "auto"
    contamination: float | Literal["auto"] = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int | None = 42
    n_jobs: int | None = -1
    feature_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if self.max_samples != "auto":
            is_positive_count = (
                isinstance(self.max_samples, Integral)
                and not isinstance(self.max_samples, bool)
                and self.max_samples > 0
            )
            is_fraction = (
                isinstance(self.max_samples, Real)
                and not isinstance(self.max_samples, bool)
                and math.isfinite(self.max_samples)
                and 0 < self.max_samples <= 1
            )
            if not (is_positive_count or is_fraction):
                raise ValueError(
                    "max_samples must be 'auto', a positive integer, or a fraction <= 1."
                )
        if self.contamination != "auto":
            if (
                isinstance(self.contamination, bool)
                or not isinstance(self.contamination, Real)
                or not math.isfinite(self.contamination)
                or not 0 < self.contamination <= 0.5
            ):
                raise ValueError("contamination must be 'auto' or a fraction in (0, 0.5].")
        if (
            isinstance(self.max_features, bool)
            or not isinstance(self.max_features, Real)
            or not math.isfinite(self.max_features)
            or not 0 < self.max_features <= 1
        ):
            raise ValueError("max_features must be a finite fraction in (0, 1].")
        if isinstance(self.random_state, bool) or (
            self.random_state is not None and not isinstance(self.random_state, int)
        ):
            raise ValueError("random_state must be an integer or None.")
        if isinstance(self.n_jobs, bool) or (
            self.n_jobs is not None and not isinstance(self.n_jobs, int)
        ):
            raise ValueError("n_jobs must be an integer or None.")
        if self.feature_columns is not None:
            columns = tuple(self.feature_columns)
            if not columns or any(not isinstance(column, str) or not column for column in columns):
                raise ValueError("feature_columns must contain non-empty column names.")
            if len(set(columns)) != len(columns):
                raise ValueError("feature_columns must not contain duplicates.")
            if any(_is_label_column(column) for column in columns):
                raise ValueError("feature_columns must not include label or target columns.")
            object.__setattr__(self, "feature_columns", columns)


def _is_label_column(column: str) -> bool:
    normalized = column.casefold()
    return normalized in _LABEL_COLUMNS or normalized.endswith(
        ("_label", "_target", "_ground_truth")
    )


class IsolationForestDetector:
    """Fit an Isolation Forest on training data and predict aligned anomaly scores.

    Median imputation is fitted only during :meth:`fit`. ``predict`` transforms
    observations using that fitted imputer, then appends the sklearn prediction,
    a 0/1 anomaly flag, and a score whose larger values indicate greater
    abnormality. Label-like training columns are rejected to keep the unsupervised
    fit boundary explicit.
    """

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()

    def fit(self, training_frame: pd.DataFrame) -> IsolationForestDetector:
        """Fit imputation and the estimator from training rows only."""
        self._validate_frame(training_frame, "training_frame")
        label_columns = [column for column in training_frame if _is_label_column(column)]
        if label_columns:
            raise ValueError(
                "Training data must not contain label or target columns: "
                + ", ".join(label_columns)
            )

        columns = self._select_feature_columns(training_frame)
        matrix = self._feature_matrix(training_frame, columns)
        if len(matrix) < 2:
            raise ValueError("Isolation Forest requires at least two training observations.")
        if not np.isfinite(matrix).any():
            raise ValueError("Training data has no finite values in the selected features.")

        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = imputer.fit_transform(matrix)
        estimator = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        estimator.fit(imputed)

        self.feature_columns_ = columns
        self.imputer_ = imputer
        self.estimator_ = estimator
        self.n_training_rows_ = len(training_frame)
        logger.info(
            "Fitted Isolation Forest using %d training rows and %d features.",
            self.n_training_rows_,
            len(columns),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with aligned Isolation Forest labels and scores."""
        self._require_fitted()
        self._validate_frame(frame, "frame")
        missing_columns = [column for column in self.feature_columns_ if column not in frame]
        if missing_columns:
            raise ValueError(
                "Prediction data is missing fitted feature columns: " + ", ".join(missing_columns)
            )

        matrix = self._feature_matrix(frame, self.feature_columns_)
        imputed = self.imputer_.transform(matrix)
        predictions = self.estimator_.predict(imputed).astype(np.int8, copy=False)
        anomaly_scores = -self.estimator_.score_samples(imputed)

        result = frame.copy()
        result["isolation_forest_prediction"] = predictions
        result["isolation_forest_anomaly_label"] = (predictions == -1).astype(np.uint8)
        result["isolation_forest_anomaly_score"] = anomaly_scores
        return result

    def save(self, path: str | Path) -> Path:
        """Atomically save a fitted detector; the artifact remains pickle-based."""
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".joblib",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            dump(self, temporary_path)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        logger.info("Saved Isolation Forest artifact to %s.", destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> IsolationForestDetector:
        """Load a trusted joblib artifact.

        Joblib uses pickle under the hood. Never load a model artifact from an
        untrusted source.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Isolation Forest artifact does not exist: {source}")
        detector = load(source)
        if not isinstance(detector, cls) or not detector._is_fitted():
            raise ValueError(f"Artifact at {source} is not a fitted {cls.__name__}.")
        logger.info("Loaded Isolation Forest artifact from %s.", source)
        return detector

    def _select_feature_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        if self.config.feature_columns is not None:
            columns = self.config.feature_columns
            missing = [column for column in columns if column not in frame]
            if missing:
                raise ValueError(
                    "Training data is missing configured feature columns: " + ", ".join(missing)
                )
            return columns

        columns = tuple(
            column
            for column in frame.select_dtypes(include=[np.number, "bool"]).columns
            if column.casefold() not in _NON_FEATURE_COLUMNS
            and not column.casefold().startswith(("synthetic_", "original_"))
            and not _is_label_column(column)
        )
        if not columns:
            raise ValueError(
                "No numeric training features found; configure feature_columns explicitly."
            )
        return columns

    @staticmethod
    def _feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        feature_data = pd.DataFrame(index=frame.index)
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            feature_data[column] = values.replace([np.inf, -np.inf], np.nan)
        return feature_data.to_numpy(dtype=np.float64, na_value=np.nan)

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, name: str) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if frame.empty:
            raise ValueError(f"{name} must contain at least one observation.")
        if not frame.columns.is_unique:
            raise ValueError(f"{name} must have unique column names.")
        if any(not isinstance(column, str) for column in frame.columns):
            raise ValueError(f"{name} must use string column names.")

    def _is_fitted(self) -> bool:
        return all(
            hasattr(self, attribute)
            for attribute in ("feature_columns_", "imputer_", "estimator_", "n_training_rows_")
        )

    def _require_fitted(self) -> None:
        if not self._is_fitted():
            raise RuntimeError("IsolationForestDetector must be fitted before this operation.")
