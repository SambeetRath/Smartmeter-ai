from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartmeter.anomaly_detection.features import engineer_features


def _frame(
    values: list[float | None],
    timestamps: list[str] | None = None,
) -> pd.DataFrame:
    if timestamps is None:
        timestamps = [f"2024-01-01 00:{minute:02d}:00" for minute in range(len(values))]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, errors="coerce"),
            "global_active_power": values,
            "global_reactive_power": [0.1] * len(values),
            "voltage": [230.0] * len(values),
            "global_intensity": [5.0] * len(values),
            "sub_metering_1": [0.0] * len(values),
            "sub_metering_2": [1.0] * len(values),
            "sub_metering_3": [2.0] * len(values),
        }
    )
    frame["source_marker"] = [f"source-{index}" for index in range(len(frame))]
    return frame


def test_temporal_features_cover_clock_weekday_weekend_and_month_cycles():
    frame = _frame(
        [1.0, 2.0, 3.0],
        [
            "2024-01-01 23:30:00",
            "2024-01-02 00:00:00",
            "2024-01-06 12:00:00",
        ],
    )

    features = engineer_features(
        frame,
        columns=["global_active_power"],
        lag_periods=(),
        rolling_windows=(),
    )

    assert features["hour_of_day"].tolist() == [23.5, 0.0, 12.0]
    assert features["minute_of_day"].tolist() == [1410, 0, 720]
    assert features["day_of_week"].tolist() == [0, 1, 5]
    assert features["is_weekend"].tolist() == [0, 0, 1]
    assert features["month"].tolist() == [1, 1, 1]
    assert features.loc[0, "hour_sin"] == pytest.approx(np.sin(2 * np.pi * 23.5 / 24))
    assert features.loc[0, "hour_cos"] == pytest.approx(np.cos(2 * np.pi * 23.5 / 24))
    assert features.loc[0, "month_sin"] == pytest.approx(0.0, abs=1e-6)
    assert features.loc[0, "month_cos"] == pytest.approx(1.0, abs=1e-6)


def test_lag_features_are_previous_observations_and_preserve_alignment():
    frame = _frame([10.0, 20.0, 30.0, 40.0, 50.0])
    original = frame.copy(deep=True)

    features = engineer_features(
        frame,
        columns=["global_active_power"],
        lag_periods=(1, 2, 4),
        rolling_windows=(),
    )

    assert pd.isna(features.loc[0, "global_active_power__lag_1"])
    assert features.loc[1, "global_active_power__lag_1"] == 10.0
    assert pd.isna(features.loc[0, "global_active_power__lag_2"])
    assert features.loc[2, "global_active_power__lag_2"] == 10.0
    assert features.loc[4, "global_active_power__lag_4"] == 10.0
    assert features["source_marker"].tolist() == frame["source_marker"].tolist()
    pd.testing.assert_frame_equal(frame, original)


def test_rolling_features_exclude_current_reading_and_use_configured_history():
    frame = _frame([10.0, 20.0, 30.0, 40.0])

    features = engineer_features(
        frame,
        columns=["global_active_power"],
        lag_periods=(),
        rolling_windows=(3,),
        rolling_min_periods=2,
    )

    assert pd.isna(features.loc[0, "global_active_power__rolling_mean_3"])
    assert features.loc[2, "global_active_power__rolling_mean_3"] == 15.0
    assert features.loc[2, "global_active_power__rolling_std_3"] == 5.0
    assert features.loc[2, "global_active_power__rolling_min_3"] == 10.0
    assert features.loc[2, "global_active_power__rolling_max_3"] == 20.0
    assert features.loc[2, "global_active_power__deviation_from_rolling_mean_3"] == 15.0
    assert features.loc[3, "global_active_power__rolling_mean_3"] == 20.0


def test_feature_values_at_existing_rows_do_not_change_when_future_rows_are_added():
    history = _frame([10.0, 11.0, 9.0, 12.0])
    extended = pd.concat(
        [
            history,
            _frame([1_000_000.0], ["2024-01-01 00:04:00"]),
        ],
        ignore_index=True,
    )
    options = {
        "columns": ["global_active_power"],
        "lag_periods": (1, 3),
        "rolling_windows": (3,),
        "rolling_min_periods": 2,
    }

    history_features = engineer_features(history, **options)
    extended_features = engineer_features(extended, **options)

    pd.testing.assert_frame_equal(
        history_features,
        extended_features.iloc[: len(history)],
        check_dtype=False,
    )


def test_missing_values_are_preserved_and_marked_without_dropping_rows():
    frame = _frame([10.0, None, float("inf"), 20.0])

    features = engineer_features(
        frame,
        columns=["global_active_power"],
        lag_periods=(1,),
        rolling_windows=(3,),
        rolling_min_periods=1,
    )

    assert len(features) == len(frame)
    assert features["global_active_power"].isna().sum() == 1
    assert features["global_active_power__is_missing"].tolist() == [0, 1, 1, 0]
    assert pd.isna(features.loc[3, "global_active_power__lag_1"])
    assert features.loc[3, "global_active_power__rolling_mean_3"] == 10.0


def test_unsorted_input_is_calculated_chronologically_but_returned_in_source_order():
    frame = _frame(
        [30.0, 10.0, 20.0],
        [
            "2024-01-01 00:02:00",
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
        ],
    )

    features = engineer_features(
        frame,
        columns=["global_active_power"],
        lag_periods=(1,),
        rolling_windows=(2,),
    )

    assert features["source_marker"].tolist() == frame["source_marker"].tolist()
    assert pd.isna(features.loc[1, "global_active_power__lag_1"])
    assert features.loc[2, "global_active_power__lag_1"] == 10.0
    assert features.loc[0, "global_active_power__lag_1"] == 20.0


def test_invalid_or_duplicate_timestamps_are_rejected():
    duplicate_times = _frame(
        [1.0, 2.0],
        ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
    )
    invalid_time = _frame([1.0, 2.0], ["2024-01-01 00:00:00", "not-a-time"])

    with pytest.raises(ValueError, match="must be unique"):
        engineer_features(duplicate_times)
    with pytest.raises(ValueError, match="invalid timestamps"):
        engineer_features(invalid_time)


@pytest.mark.parametrize(
    ("lag_periods", "rolling_windows", "minimum"),
    [
        ((0,), (3,), 1),
        ((1, 1), (3,), 1),
        ((1,), (0,), 1),
        ((1,), (3,), 0),
        ((1,), (3,), 4),
    ],
)
def test_invalid_feature_window_configuration_is_rejected(
    lag_periods,
    rolling_windows,
    minimum,
):
    with pytest.raises(ValueError):
        engineer_features(
            _frame([1.0, 2.0, 3.0]),
            lag_periods=lag_periods,
            rolling_windows=rolling_windows,
            rolling_min_periods=minimum,
        )


def test_feature_engineering_rejects_missing_measurement_columns():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2)})

    with pytest.raises(ValueError, match="measurement columns"):
        engineer_features(frame)
