"""Statistical anomaly detection baselines for smart-meter time series."""

from .evaluation import (
    EvaluationConfig,
    calculate_binary_metrics,
    evaluate_predictions,
    run_anomaly_evaluation,
    write_evaluation_report,
)
from .features import engineer_features
from .isolation_forest import IsolationForestConfig, IsolationForestDetector
from .pipeline import (
    DETECTOR_NAMES,
    PipelineConfig,
    PipelineRunResult,
    execute_anomaly_pipeline,
    load_parquet_dataset,
    load_postgresql_dataset,
    run_anomaly_pipeline,
)
from .splitting import TemporalDataSplit, chronological_train_validation_test_split
from .statistical import (
    GlobalIQRDetector,
    detect_global_iqr,
    detect_rolling_outliers,
    detect_time_of_day_outliers,
)
from .synthetic_data import SyntheticInjectionConfig, inject_synthetic_anomalies

__all__ = [
    "detect_global_iqr",
    "GlobalIQRDetector",
    "detect_rolling_outliers",
    "detect_time_of_day_outliers",
    "engineer_features",
    "IsolationForestConfig",
    "IsolationForestDetector",
    "DETECTOR_NAMES",
    "PipelineConfig",
    "PipelineRunResult",
    "run_anomaly_pipeline",
    "execute_anomaly_pipeline",
    "load_parquet_dataset",
    "load_postgresql_dataset",
    "TemporalDataSplit",
    "chronological_train_validation_test_split",
    "SyntheticInjectionConfig",
    "inject_synthetic_anomalies",
    "EvaluationConfig",
    "calculate_binary_metrics",
    "evaluate_predictions",
    "run_anomaly_evaluation",
    "write_evaluation_report",
]
