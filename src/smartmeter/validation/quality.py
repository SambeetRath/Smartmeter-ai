from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from smartmeter.anomaly_detection.statistical import detect_global_iqr


def _empty_report() -> dict[str, Any]:
    return {
        "dataset_summary": {},
        "quality_summary": {},
        "issues": {
            "missing_measurements": [],
            "invalid_timestamps": [],
            "duplicate_timestamps": [],
            "duplicate_records": [],
            "missing_time_intervals": [],
            "negative_readings": [],
            "non_finite_values": [],
            "suspicious_outliers": [],
        },
        "terminal_summary": "",
        "status": "ok",
    }


def summarize_quality_report(report: dict[str, Any]) -> str:
    """Return a concise terminal summary for a validation pass."""
    summary = report.get("dataset_summary", {})
    total_rows = int(summary.get("rows", 0))
    issue_count = sum(len(items) for items in report.get("issues", {}).values())
    if issue_count == 0:
        return f"Validated {total_rows} records with no quality findings."

    lines = [f"Validated {total_rows} records with {issue_count} quality findings."]
    for key, issues in report.get("issues", {}).items():
        if issues:
            lines.append(f"- {key}: {len(issues)}")
    return "\n".join(lines)


def _evaluate_outliers(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if not columns:
        return []
    detections = detect_global_iqr(frame, columns=columns, iqr_multiplier=1.5)
    return [
        {
            "column": detection["column"],
            "row_index": int(frame.index[int(detection["row_index"])]),
            "value": float(detection["value"]),
            "lower_bound": float(detection["lower_bound"]),
            "upper_bound": float(detection["upper_bound"]),
            "method": "iqr_1.5x",
        }
        for detection in detections.to_dict(orient="records")
    ]


def validate_dataset(frame: pd.DataFrame, *, expected_frequency_minutes: int = 1) -> dict[str, Any]:
    """Validate a normalized smart-meter dataframe and report quality findings."""
    if frame.empty:
        report = _empty_report()
        report["dataset_summary"] = {"rows": 0, "status": "error", "columns": []}
        report["status"] = "error"
        report["terminal_summary"] = "No records available for validation."
        return report

    ordered_columns = list(frame.columns)
    numeric_columns = [
        column
        for column in ordered_columns
        if column.startswith("sub_metering_")
        or column in {"global_active_power", "global_reactive_power", "voltage", "global_intensity"}
    ]

    report = _empty_report()
    has_valid_timestamps = "timestamp" in frame.columns and not frame["timestamp"].isna().all()
    report["dataset_summary"] = {
        "rows": int(len(frame)),
        "columns": ordered_columns,
        "time_range_start": (
            frame["timestamp"].min().isoformat() if has_valid_timestamps else None
        ),
        "time_range_end": (frame["timestamp"].max().isoformat() if has_valid_timestamps else None),
        "expected_frequency_minutes": expected_frequency_minutes,
    }

    missing_measurements = []
    for column in numeric_columns:
        missing_count = int(frame[column].isna().sum())
        if missing_count:
            missing_measurements.append(
                {
                    "column": column,
                    "missing_count": missing_count,
                    "missing_pct": round((missing_count / len(frame)) * 100, 2),
                }
            )
    report["issues"]["missing_measurements"] = missing_measurements

    invalid_timestamps = []
    if "timestamp" in frame.columns:
        invalid_mask = frame["timestamp"].isna()
        if invalid_mask.any():
            invalid_timestamps = [
                {
                    "row_index": int(row_index),
                    "raw_value": (
                        str(frame.loc[row_index, "date"]) + " " + str(frame.loc[row_index, "time"])
                    ),
                }
                for row_index in frame.index[invalid_mask]
            ]
    report["issues"]["invalid_timestamps"] = invalid_timestamps

    duplicate_timestamps = []
    if "timestamp" in frame.columns:
        duplicate_series = frame["timestamp"].value_counts()
        for timestamp, count in duplicate_series.items():
            if count > 1:
                duplicate_timestamps.append({"timestamp": str(timestamp), "count": int(count)})
    report["issues"]["duplicate_timestamps"] = duplicate_timestamps

    duplicate_records = []
    duplicated_row_indices = frame.index[frame.duplicated(keep=False)].tolist()
    if duplicated_row_indices:
        duplicate_records = [{"row_indices": [int(index) for index in duplicated_row_indices]}]
    report["issues"]["duplicate_records"] = duplicate_records

    missing_intervals = []
    if "timestamp" in frame.columns:
        ordered_times = frame["timestamp"].dropna().sort_values().reset_index(drop=True)
        if len(ordered_times) > 1:
            deltas = ordered_times.diff().dt.total_seconds() / 60
            for idx, delta in deltas.items():
                if pd.isna(delta):
                    continue
                if delta > expected_frequency_minutes:
                    missing_intervals.append(
                        {
                            "from_timestamp": str(ordered_times.iloc[idx - 1]),
                            "to_timestamp": str(ordered_times.iloc[idx]),
                            "missing_minutes": float(round(delta - expected_frequency_minutes, 2)),
                        }
                    )
    report["issues"]["missing_time_intervals"] = missing_intervals

    negative_readings = []
    for column in numeric_columns:
        negative_mask = frame[column] < 0
        if negative_mask.any():
            negative_readings.append(
                {
                    "column": column,
                    "row_indices": [int(index) for index in frame.index[negative_mask]],
                    "count": int(negative_mask.sum()),
                }
            )
    report["issues"]["negative_readings"] = negative_readings

    non_finite_values = []
    for column in numeric_columns:
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        numeric_array = numeric_series.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = pd.Series(np.isfinite(numeric_array), index=numeric_series.index)
        invalid_mask = numeric_series.notna() & ~finite_mask
        if invalid_mask.any():
            non_finite_values.append(
                {
                    "column": column,
                    "row_indices": [int(index) for index in frame.index[invalid_mask]],
                    "count": int(invalid_mask.sum()),
                }
            )
    report["issues"]["non_finite_values"] = non_finite_values

    suspicious_outliers = _evaluate_outliers(frame, numeric_columns)
    report["issues"]["suspicious_outliers"] = suspicious_outliers

    outlier_counts = {
        column: sum(flag["column"] == column for flag in suspicious_outliers)
        for column in numeric_columns
    }
    negative_counts = {column: int((frame[column] < 0).sum()) for column in numeric_columns}
    non_finite_counts = {}
    for column in numeric_columns:
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        finite_mask = np.isfinite(numeric_series.to_numpy(dtype=float, na_value=np.nan))
        non_finite_counts[column] = int(
            (numeric_series.notna() & ~pd.Series(finite_mask, index=frame.index)).sum()
        )
    duplicate_timestamp_count = len(report["issues"]["duplicate_timestamps"])
    exact_duplicate_record_count = int(frame.duplicated(keep="first").sum())
    report["quality_summary"] = {
        "input_row_count": int(len(frame)),
        "processed_row_count": int(len(frame)),
        "output_file_path": None,
        "missing_measurement_counts": {
            column: int(frame[column].isna().sum()) for column in numeric_columns
        },
        "data_quality_violations": {
            "invalid_timestamp_count": len(invalid_timestamps),
            "duplicate_timestamp_count": duplicate_timestamp_count,
            "duplicate_timestamp_count_definition": (
                "Number of distinct timestamp values that occur more than once."
            ),
            "exact_duplicate_record_count": exact_duplicate_record_count,
            "exact_duplicate_record_count_definition": (
                "Number of repeated rows beyond the first identical row."
            ),
            "missing_interval_count": len(missing_intervals),
            "missing_interval_count_definition": (
                "Number of gaps between consecutive valid timestamps exceeding "
                "the expected frequency."
            ),
            "negative_measurement_counts": negative_counts,
            "negative_measurement_count": sum(negative_counts.values()),
            "non_finite_measurement_counts": non_finite_counts,
            "non_finite_measurement_count": sum(non_finite_counts.values()),
        },
        "exploratory_suspicious_outlier_flags": {
            "method": "global_iqr_1.5x",
            "iqr_multiplier": 1.5,
            "counts_by_column": outlier_counts,
            "total_flag_count": len(suspicious_outliers),
            "interpretation": (
                "Exploratory statistical flags only; global IQR does not account for "
                "time-of-day or seasonal consumption patterns and does not confirm anomalies."
            ),
        },
    }

    issue_count = sum(len(items) for items in report["issues"].values())
    report["status"] = "warning" if issue_count else "ok"
    report["terminal_summary"] = summarize_quality_report(report)
    return report
