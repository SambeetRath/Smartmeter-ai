from __future__ import annotations

import pandas as pd
import pytest

from smartmeter.anomaly_detection import (
    detect_global_iqr,
    detect_rolling_outliers,
    detect_time_of_day_outliers,
)
from smartmeter.validation.quality import validate_dataset


def _series_frame(
    values: list[float | None],
    timestamps: list[str] | None = None,
) -> pd.DataFrame:
    if timestamps is None:
        timestamps = [f"2024-01-01 00:{minute:02d}:00" for minute in range(len(values))]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, errors="coerce"),
            "global_active_power": values,
            "voltage": [230.0] * len(values),
        }
    )


def test_global_iqr_is_configurable_and_marks_flags_as_suspicious_only():
    frame = pd.DataFrame(
        {
            "global_active_power": [1.0, 2.0, 2.0, 3.0, 100.0],
            "voltage": [220.0, 221.0, 222.0, 223.0, 224.0],
        }
    )

    flags = detect_global_iqr(
        frame,
        columns=["global_active_power"],
        iqr_multiplier=1.5,
    )

    assert flags["row_index"].tolist() == [4]
    assert flags["column"].tolist() == ["global_active_power"]
    assert flags["method"].tolist() == ["global_iqr"]
    assert flags["classification"].tolist() == ["suspicious_outlier"]
    assert not flags["confirmed_anomaly"].any()
    assert detect_global_iqr(
        frame,
        columns=["global_active_power"],
        iqr_multiplier=100,
    ).empty
    assert frame["global_active_power"].iloc[-1] == 100.0


def test_global_iqr_ignores_missing_and_non_finite_values():
    frame = pd.DataFrame({"global_active_power": [1.0, 2.0, 2.0, 3.0, 100.0, None, float("inf")]})

    flags = detect_global_iqr(frame, columns=["global_active_power"])

    assert flags["row_index"].tolist() == [4]
    assert flags.iloc[0]["baseline_count"] == 5


def test_quality_report_reuses_the_global_iqr_detector():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="min"),
            "global_active_power": [1.0, 2.0, 2.0, 3.0, 100.0],
        }
    )

    detector_flags = detect_global_iqr(frame, columns=["global_active_power"])
    report_flags = validate_dataset(frame)["issues"]["suspicious_outliers"]

    assert [flag["row_index"] for flag in report_flags] == detector_flags["row_index"].tolist()
    assert [flag["value"] for flag in report_flags] == detector_flags["value"].tolist()
    assert [flag["lower_bound"] for flag in report_flags] == detector_flags["lower_bound"].tolist()
    assert all(flag["method"] == "iqr_1.5x" for flag in report_flags)


@pytest.mark.parametrize(
    ("columns", "multiplier"),
    [
        (["not_a_measurement"], 1.5),
        (["global_active_power"], -1),
        (["global_active_power"], float("inf")),
        (["global_active_power", "global_active_power"], 1.5),
    ],
)
def test_global_iqr_rejects_invalid_configuration(columns, multiplier):
    with pytest.raises(ValueError):
        detect_global_iqr(
            pd.DataFrame({"global_active_power": [1.0, 2.0]}),
            columns=columns,
            iqr_multiplier=multiplier,
        )


def test_rolling_detector_uses_only_prior_observations_and_handles_zero_variance():
    frame = _series_frame([10.0, 10.0, 10.0, 100.0, 10.0])

    detections = detect_rolling_outliers(
        frame,
        columns=["global_active_power"],
        window_size=3,
        threshold=2,
    )
    spike = detections.loc[detections["row_index"] == 3].iloc[0]

    assert spike["baseline_mean"] == 10.0
    assert spike["baseline_std"] == 0.0
    assert spike["baseline_count"] == 3
    assert bool(spike["is_suspicious_outlier"])
    assert detections.loc[detections["row_index"] == 0, "baseline_mean"].isna().all()
    assert not detections["confirmed_anomaly"].any()


def test_rolling_detector_skips_missing_values_and_requires_minimum_history():
    frame = _series_frame([10.0, None, 10.0, 100.0])

    detections = detect_rolling_outliers(
        frame,
        columns=["global_active_power"],
        window_size=3,
        minimum_history=2,
    )

    first = detections.loc[detections["row_index"] == 1].iloc[0]
    spike = detections.loc[detections["row_index"] == 3].iloc[0]
    assert pd.isna(first["baseline_mean"])
    assert first["baseline_count"] == 1
    assert spike["baseline_count"] == 2
    assert spike["baseline_mean"] == 10.0
    assert bool(spike["is_suspicious_outlier"])
    assert detections.loc[detections["row_index"] == 1, "is_suspicious_outlier"].item() is False


def test_rolling_detector_is_invariant_to_future_observations():
    initial = _series_frame([10.0, 11.0, 9.0, 10.0])
    extended = pd.concat(
        [initial, _series_frame([1_000_000.0], ["2024-01-01 00:04:00"])],
        ignore_index=True,
    )
    options = {
        "columns": ["global_active_power"],
        "window_size": 3,
        "minimum_history": 2,
    }

    initial_result = detect_rolling_outliers(initial, **options)
    extended_result = detect_rolling_outliers(extended, **options)

    pd.testing.assert_frame_equal(
        initial_result.drop(columns="row_index"),
        extended_result.iloc[: len(initial_result)].drop(columns="row_index"),
        check_dtype=False,
    )


def test_rolling_detector_sorts_by_timestamp_and_keeps_source_row_indices():
    frame = _series_frame(
        [100.0, 10.0, 10.0],
        [
            "2024-01-01 00:02:00",
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
        ],
    )

    detections = detect_rolling_outliers(
        frame,
        columns=["global_active_power"],
        window_size=2,
        minimum_history=2,
    )

    assert detections["row_index"].tolist() == [1, 2, 0]
    assert bool(detections.iloc[-1]["is_suspicious_outlier"])


@pytest.mark.parametrize(
    ("window_size", "minimum_history"),
    [(0, None), (3, 0), (3, 4)],
)
def test_rolling_detector_rejects_invalid_history_configuration(window_size, minimum_history):
    with pytest.raises(ValueError):
        detect_rolling_outliers(
            _series_frame([1.0, 2.0]),
            columns=["global_active_power"],
            window_size=window_size,
            minimum_history=minimum_history,
        )


def test_rolling_detector_requires_unique_valid_timestamps():
    duplicate_times = _series_frame(
        [1.0, 2.0],
        ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
    )
    invalid_time = _series_frame([1.0, 2.0], ["2024-01-01 00:00:00", "not-a-time"])

    with pytest.raises(ValueError, match="must be unique"):
        detect_rolling_outliers(duplicate_times, columns=["global_active_power"])
    with pytest.raises(ValueError, match="invalid timestamps"):
        detect_rolling_outliers(invalid_time, columns=["global_active_power"])


def test_time_of_day_detector_uses_prior_matching_weekday_and_minute_only():
    frame = _series_frame(
        [10.0, 500.0, 10.0, 100.0],
        [
            "2024-01-01 08:15:00",
            "2024-01-01 08:16:00",
            "2024-01-08 08:15:00",
            "2024-01-15 08:15:00",
        ],
    )

    detections = detect_time_of_day_outliers(
        frame,
        columns=["global_active_power"],
        minimum_history=2,
        threshold=2,
    )
    evaluated = detections.loc[detections["row_index"] == 3].iloc[0]

    assert evaluated["baseline_count"] == 2
    assert evaluated["baseline_mean"] == 10.0
    assert bool(evaluated["is_suspicious_outlier"])
    assert detections.loc[detections["row_index"] == 1, "baseline_count"].item() == 0


def test_time_of_day_detector_excludes_future_matching_observations():
    history = _series_frame(
        [10.0, 10.0, 100.0],
        [
            "2024-01-01 08:15:00",
            "2024-01-08 08:15:00",
            "2024-01-15 08:15:00",
        ],
    )
    with_future = pd.concat(
        [
            history,
            _series_frame([1_000_000.0], ["2024-01-22 08:15:00"]),
        ],
        ignore_index=True,
    )
    options = {
        "columns": ["global_active_power"],
        "minimum_history": 2,
        "threshold": 2,
    }

    initial_result = detect_time_of_day_outliers(history, **options)
    extended_result = detect_time_of_day_outliers(with_future, **options)

    pd.testing.assert_frame_equal(
        initial_result.drop(columns="row_index"),
        extended_result.iloc[: len(initial_result)].drop(columns="row_index"),
        check_dtype=False,
    )
    assert initial_result.loc[2, "baseline_mean"] == 10.0


def test_time_of_day_detector_can_pool_weekdays_and_weekends():
    frame = _series_frame(
        [10.0, 10.0, 100.0],
        [
            "2024-01-01 08:15:00",
            "2024-01-06 08:15:00",
            "2024-01-07 08:15:00",
        ],
    )

    weekday_grouped = detect_time_of_day_outliers(
        frame,
        columns=["global_active_power"],
        minimum_history=1,
        threshold=2,
        group_by_day_of_week=True,
    )
    pooled = detect_time_of_day_outliers(
        frame,
        columns=["global_active_power"],
        minimum_history=1,
        threshold=2,
        group_by_day_of_week=False,
    )

    assert weekday_grouped.loc[2, "baseline_count"] == 0
    assert pooled.loc[2, "baseline_count"] == 2
    assert pooled.loc[2, "baseline_mean"] == 10.0


def test_time_of_day_detector_handles_missing_values_and_insufficient_history():
    frame = _series_frame(
        [None, 10.0, 10.0],
        [
            "2024-01-01 08:15:00",
            "2024-01-08 08:15:00",
            "2024-01-15 08:15:00",
        ],
    )

    detections = detect_time_of_day_outliers(
        frame,
        columns=["global_active_power"],
        minimum_history=1,
    )

    assert detections["baseline_count"].tolist() == [0, 0, 1]
    assert pd.isna(detections.loc[1, "baseline_mean"])
    assert detections["is_suspicious_outlier"].tolist() == [False, False, False]


def test_time_of_day_detector_requires_unique_valid_timestamps():
    duplicate_times = _series_frame(
        [1.0, 2.0],
        ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
    )
    with pytest.raises(ValueError, match="must be unique"):
        detect_time_of_day_outliers(
            duplicate_times,
            columns=["global_active_power"],
            minimum_history=1,
        )
