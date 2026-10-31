from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartmeter.ingestion.loader import load_and_validate_dataset, load_dataset
from smartmeter.validation.quality import validate_dataset

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_valid_household.csv"


def _sample_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
            "2024-01-01 00:02:00",
            "2024-01-01 00:03:00",
            "2024-01-01 00:04:00",
        ]
    )
    datetime_series = pd.Series(timestamps)
    return pd.DataFrame(
        {
            "date": datetime_series.dt.strftime("%d/%m/%Y"),
            "time": datetime_series.dt.strftime("%H:%M:%S"),
            "global_active_power": [1.0, 2.0, 3.0, 4.0, 5.0],
            "global_reactive_power": [0.1, 0.2, 0.3, 0.4, 0.5],
            "voltage": [220.0, 221.0, 222.0, 223.0, 224.0],
            "global_intensity": [10.0, 11.0, 12.0, 13.0, 14.0],
            "sub_metering_1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "sub_metering_2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sub_metering_3": [2.0, 3.0, 4.0, 5.0, 6.0],
            "timestamp": timestamps,
        }
    )


def test_package_import_does_not_trigger_module_execution_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.import_module("smartmeter.ingestion")


def test_load_valid_fixture_and_quality_report():
    frame, report = load_and_validate_dataset(FIXTURE_PATH)

    assert len(frame) == 5
    assert "timestamp" in frame.columns
    assert frame["timestamp"].is_monotonic_increasing
    assert report["status"] == "ok"


def test_missing_measurement_values_are_reported():
    frame = _sample_frame()
    frame.loc[0, "global_active_power"] = np.nan

    report = validate_dataset(frame)

    assert any(
        item["column"] == "global_active_power" for item in report["issues"]["missing_measurements"]
    )


def test_invalid_timestamps_are_reported():
    frame = _sample_frame()
    frame.loc[0, "timestamp"] = pd.NaT

    report = validate_dataset(frame)

    assert report["issues"]["invalid_timestamps"]


def test_duplicate_timestamps_and_records_are_detected():
    frame = _sample_frame()
    duplicated = pd.concat([frame.iloc[:2], frame.iloc[:2]], ignore_index=True)

    report = validate_dataset(duplicated)

    assert report["issues"]["duplicate_timestamps"]
    assert report["issues"]["duplicate_records"]


def test_missing_time_intervals_are_detected():
    frame = pd.DataFrame(
        {
            "date": ["01/01/2024", "01/01/2024", "01/01/2024"],
            "time": ["00:00:00", "00:02:00", "00:05:00"],
            "global_active_power": [1.0, 2.0, 3.0],
            "global_reactive_power": [0.2, 0.3, 0.4],
            "voltage": [220.0, 221.0, 222.0],
            "global_intensity": [10.0, 10.5, 11.0],
            "sub_metering_1": [0.0, 1.0, 2.0],
            "sub_metering_2": [1.0, 2.0, 3.0],
            "sub_metering_3": [2.0, 3.0, 4.0],
        }
    )
    frame["timestamp"] = pd.to_datetime(
        frame["date"] + " " + frame["time"],
        format="%d/%m/%Y %H:%M:%S",
    )

    report = validate_dataset(frame)

    assert report["issues"]["missing_time_intervals"]


def test_negative_values_and_non_finite_values_are_reported():
    frame = _sample_frame()
    frame.loc[0, "global_active_power"] = -8.0
    frame.loc[1, "global_active_power"] = float("inf")

    report = validate_dataset(frame)

    assert any(
        item["column"] == "global_active_power" for item in report["issues"]["negative_readings"]
    )
    assert any(
        item["column"] == "global_active_power" for item in report["issues"]["non_finite_values"]
    )


def test_outlier_detection_flags_values_without_removing_them():
    frame = _sample_frame()
    frame.loc[0, "global_active_power"] = 1000.0

    report = validate_dataset(frame)

    assert report["issues"]["suspicious_outliers"]
    assert frame.loc[0, "global_active_power"] == 1000.0


def test_missing_required_columns_raise_a_clear_error(tmp_path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("Date;Time;Voltage\n01/01/2024;00:00:00;220\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(bad_file)


def test_missing_dataset_raises_actionable_error():
    missing_path = Path("data/raw/definitely_missing.csv")

    with pytest.raises(
        FileNotFoundError,
        match="Download the UCI household power consumption file",
    ):
        load_dataset(missing_path)
