from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smartmeter.ingestion.loader import run_pipeline

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "synthetic_valid_household.csv"
)
MEASUREMENT_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]


def test_pipeline_writes_complete_parquet_and_quality_summary(tmp_path):
    source_path = tmp_path / "synthetic_source.csv"
    source = pd.read_csv(FIXTURE_PATH, sep=";", dtype=str)
    source.loc[2, "Global_active_power"] = "?"
    source.iloc[::-1].to_csv(source_path, sep=";", index=False)
    original_bytes = source_path.read_bytes()

    processed_path = tmp_path / "new-output" / "household.parquet"
    report_path = tmp_path / "new-reports" / "quality.json"
    dataset, report = run_pipeline(
        source_path,
        processed_path=processed_path,
        report_path=report_path,
    )

    saved = pd.read_parquet(processed_path, engine="pyarrow")
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert processed_path.is_file()
    assert len(saved) == len(dataset) == len(source)
    assert saved.columns.tolist() == dataset.columns.tolist()
    assert saved["timestamp"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(saved["timestamp"])
    assert all(
        pd.api.types.is_numeric_dtype(saved[column])
        for column in MEASUREMENT_COLUMNS
    )
    assert saved["global_active_power"].isna().sum() == 1
    assert report["quality_summary"]["input_row_count"] == len(source)
    assert report["quality_summary"]["processed_row_count"] == len(source)
    assert report["quality_summary"]["output_file_path"] == str(processed_path)
    assert report["quality_summary"]["missing_measurement_counts"]["global_active_power"] == 1
    assert saved_report["quality_summary"] == report["quality_summary"]
    assert source_path.read_bytes() == original_bytes
    assert report["quality_summary"]["exploratory_suspicious_outlier_flags"]["interpretation"]
