from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

METER_COLUMNS = (
    "meter_identifier",
    "reading_count",
    "first_reading_at",
    "last_reading_at",
)
TIMESERIES_COLUMNS = (
    "timestamp",
    "reading_count",
    "global_active_power_average",
    "global_active_power_minimum",
    "global_active_power_maximum",
    "global_reactive_power_average",
    "voltage_average",
    "global_intensity_average",
    "sub_metering_1_average",
    "sub_metering_2_average",
    "sub_metering_3_average",
)
ANOMALY_COLUMNS = (
    "run_id",
    "reading_id",
    "meter_identifier",
    "timestamp",
    "prediction_status",
    "algorithm",
    "anomaly_score",
    "anomaly_category",
)
DETECTION_RUN_COLUMNS = (
    "run_id",
    "meter_identifier",
    "algorithm",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "attempts",
    "training_readings",
    "readings_processed",
    "anomaly_count",
    "normal_count",
    "unavailable_count",
    "error_message",
)
SUMMARY_COLUMNS = (
    "meter_identifier",
    "start",
    "end",
    "reading_count",
    "active_power_average",
    "active_power_minimum",
    "active_power_maximum",
    "latest_active_power",
    "latest_active_power_at",
)
ELECTRICAL_METRICS = (
    ("global_active_power_average", "Active power", "kW", "Average household active power"),
    ("global_reactive_power_average", "Reactive power", "kW", "Average reactive power"),
    ("voltage_average", "Voltage", "V", "Average supply voltage"),
    ("global_intensity_average", "Current", "A", "Average current intensity"),
    (
        "sub_metering_1_average",
        "Sub-metering 1",
        "Wh / reading",
        "Average kitchen sub-meter energy per source reading",
    ),
    (
        "sub_metering_2_average",
        "Sub-metering 2",
        "Wh / reading",
        "Average laundry sub-meter energy per source reading",
    ),
    (
        "sub_metering_3_average",
        "Sub-metering 3",
        "Wh / reading",
        "Average climate-control sub-meter energy per source reading",
    ),
)


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        value = value.get("items", value.get("buckets", []))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Expected a sequence of API records.")
    records = []
    for item in value:
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json")
        if not isinstance(item, Mapping):
            raise TypeError("Each API record must be a mapping or Pydantic model.")
        records.append(dict(item))
    return records


def _frame(value: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    records = _records(value)
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_records(records)
    return frame.reindex(columns=columns)


def meters_dataframe(value: Any) -> pd.DataFrame:
    """Convert meter coverage responses into a chart/table-ready frame."""
    frame = _frame(value, METER_COLUMNS)
    for column in ("first_reading_at", "last_reading_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def timeseries_dataframe(value: Any) -> pd.DataFrame:
    """Convert API time buckets without recomputing aggregate values."""
    frame = _frame(value, TIMESERIES_COLUMNS)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def electrical_metrics_dataframe(value: Any) -> pd.DataFrame:
    """Convert aggregate buckets to long-form electrical metrics, preserving nulls."""
    frame = timeseries_dataframe(value)
    if frame.empty:
        return pd.DataFrame(
            columns=("timestamp", "metric", "value", "unit", "description", "reading_count")
        )
    metadata = {
        column: (name, unit, description) for column, name, unit, description in ELECTRICAL_METRICS
    }
    rows = []
    for column, (name, unit, description) in metadata.items():
        rows.extend(
            {
                "timestamp": row.timestamp,
                "metric": name,
                "value": getattr(row, column),
                "unit": unit,
                "description": description,
                "reading_count": row.reading_count,
            }
            for row in frame.itertuples(index=False)
        )
    return pd.DataFrame.from_records(rows)


def anomalies_dataframe(value: Any) -> pd.DataFrame:
    """Convert persisted anomaly API records into a tabular representation."""
    frame = _frame(value, ANOMALY_COLUMNS)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def detection_runs_dataframe(value: Any) -> pd.DataFrame:
    """Convert detection-run history without recomputing worker statistics."""
    frame = _frame(value, DETECTION_RUN_COLUMNS)
    for column in ("created_at", "started_at", "completed_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def missing_measurements_dataframe(value: Mapping[str, Any] | None) -> pd.DataFrame:
    """Convert backend-provided missing-value counts and rates into rows."""
    if not value:
        return pd.DataFrame(columns=("measurement", "missing_count", "missing_rate"))
    rows = [
        {
            "measurement": name,
            "missing_count": stats.model_dump(mode="json")["missing_count"]
            if hasattr(stats, "model_dump")
            else stats["missing_count"],
            "missing_rate": stats.model_dump(mode="json")["missing_rate"]
            if hasattr(stats, "model_dump")
            else stats["missing_rate"],
        }
        for name, stats in value.items()
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=("measurement", "missing_count", "missing_rate"),
    )


def analytics_summary_dataframe(value: Any) -> pd.DataFrame:
    """Represent summary fields returned by the backend without recalculating metrics."""
    if value is None:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError("Expected an analytics summary mapping or Pydantic model.")
    frame = pd.DataFrame.from_records([{key: value.get(key) for key in SUMMARY_COLUMNS}])
    for column in ("start", "end", "latest_active_power_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def quality_summary_dataframe(value: Any) -> pd.DataFrame:
    """Flatten database-observed quality fields for display without inferring new metrics."""
    if value is None:
        return pd.DataFrame(
            columns=(
                "meter_identifier",
                "reading_count",
                "database_metrics_scope",
                "expected_interval_minutes",
                "observed_gap_count",
                "estimated_missing_intervals",
                "boundary_gaps_assessed",
                "ingestion_validation_available",
            )
        )
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError("Expected a data-quality mapping or Pydantic model.")
    gaps = value.get("timestamp_gaps") or {}
    ingestion = value.get("ingestion_validation") or {}
    return pd.DataFrame.from_records(
        [
            {
                "meter_identifier": value.get("meter_identifier"),
                "reading_count": value.get("reading_count"),
                "database_metrics_scope": value.get("database_metrics_scope"),
                "expected_interval_minutes": gaps.get("expected_interval_minutes"),
                "observed_gap_count": gaps.get("observed_gap_count"),
                "estimated_missing_intervals": gaps.get("estimated_missing_intervals"),
                "boundary_gaps_assessed": gaps.get("boundary_gaps_assessed"),
                "ingestion_validation_available": ingestion.get("available"),
            }
        ]
    )
