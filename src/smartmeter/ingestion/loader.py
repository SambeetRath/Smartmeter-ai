from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from smartmeter.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_PROCESSED_DATASET_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from smartmeter.logging_config import configure_logging
from smartmeter.validation.quality import validate_dataset

DEFAULT_EXPECTED_COLUMNS = [
    "date",
    "time",
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

NUMERIC_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]


def normalize_column_name(column_name: str) -> str:
    """Normalize CSV headers to the expected schema names."""
    normalized = str(column_name).strip().lower()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in normalized)
    normalized = normalized.strip("_")
    return normalized


def _ensure_dataset_path(path: str | Path) -> Path:
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path


def load_dataset(
    path: str | Path | None = None,
    *,
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the raw UCI dataset, validate the schema, and normalize numeric columns."""
    dataset_path = _ensure_dataset_path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            "SmartMeter dataset not found. Download the UCI household power "
            f"consumption file and save it to '{dataset_path}' "
            "(or set SMARTMETER_DATASET_PATH)."
        )

    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: '{dataset_path}'")

    try:
        frame = pd.read_csv(
            dataset_path,
            sep=";",
            na_values=["?", ""],
            keep_default_na=True,
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file has no readable rows: '{dataset_path}'") from exc

    if frame.empty:
        raise ValueError(f"Dataset file contains no data rows: '{dataset_path}'")

    required_columns = list(expected_columns or DEFAULT_EXPECTED_COLUMNS)
    renamed_columns = {original: normalize_column_name(original) for original in frame.columns}
    frame = frame.rename(columns=renamed_columns)

    missing_columns = [col for col in required_columns if col not in frame.columns]
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if {"date", "time"}.issubset(frame.columns):
        combined_timestamp = (
            frame["date"].astype("string").str.strip()
            + " "
            + frame["time"].astype("string").str.strip()
        )
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(
            combined_timestamp,
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
    else:
        raise ValueError("Dataset does not include the expected date and time columns.")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def load_and_validate_dataset(
    path: str | Path | None = None,
    *,
    expected_columns: Sequence[str] | None = None,
):
    """Load a dataset and produce deterministic quality metrics."""
    frame = load_dataset(path, expected_columns=expected_columns)
    quality_report = validate_dataset(frame)
    return frame, quality_report


def save_processed_dataset(
    frame: pd.DataFrame,
    output_path: str | Path = DEFAULT_PROCESSED_DATASET_PATH,
) -> Path:
    """Write all normalized rows to Parquet, creating the parent directory if needed."""
    destination = _ensure_dataset_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, engine="pyarrow", index=False)
    return destination


def _write_quality_report(
    quality_report: dict,
    output_path: str | Path = REPORTS_DIR / "quality_report.json",
) -> Path:
    destination = _ensure_dataset_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(quality_report, handle, indent=2, allow_nan=False)
    return destination


def run_pipeline(
    dataset_path: str | Path | None = None,
    *,
    processed_path: str | Path = DEFAULT_PROCESSED_DATASET_PATH,
    report_path: str | Path = REPORTS_DIR / "quality_report.json",
) -> tuple[pd.DataFrame, dict]:
    """Load, validate, and write the processed dataset and quality report."""
    dataset, report = load_and_validate_dataset(dataset_path)
    processed_output = save_processed_dataset(dataset, processed_path)
    report["quality_summary"]["output_file_path"] = str(processed_output)
    _write_quality_report(report, report_path)
    return dataset, report


if __name__ == "__main__":
    logger = configure_logging()
    try:
        dataset, report = run_pipeline(DEFAULT_DATASET_PATH)
        processed_output = Path(report["quality_summary"]["output_file_path"])
        report_output = _ensure_dataset_path(REPORTS_DIR / "quality_report.json")
        logger.info(
            "Normalized %s rows and wrote Parquet output to %s (%s bytes)",
            len(dataset),
            processed_output,
            processed_output.stat().st_size,
        )
        logger.info("Wrote quality report to %s", report_output)
        logger.info(report["terminal_summary"])
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
