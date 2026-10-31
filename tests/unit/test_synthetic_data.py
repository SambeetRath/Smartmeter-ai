from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartmeter.anomaly_detection.splitting import (
    chronological_train_validation_test_split,
)
from smartmeter.anomaly_detection.synthetic_data import (
    SyntheticInjectionConfig,
    inject_synthetic_anomalies,
)


def _frame(row_count: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=row_count, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": timestamps.strftime("%d/%m/%Y"),
            "time": timestamps.strftime("%H:%M:%S"),
            "global_active_power": np.linspace(1.0, 4.0, row_count),
            "global_reactive_power": np.linspace(0.1, 0.5, row_count),
            "voltage": np.full(row_count, 230.0),
            "global_intensity": np.linspace(5.0, 10.0, row_count),
            "sub_metering_1": np.linspace(0.0, 2.0, row_count),
            "sub_metering_2": np.linspace(1.0, 3.0, row_count),
            "sub_metering_3": np.linspace(2.0, 4.0, row_count),
            "source_marker": np.arange(row_count),
        }
    )


def _all_categories_config(**overrides) -> SyntheticInjectionConfig:
    values = {
        "random_seed": 7,
        "consumption_spike_rate": 0.1,
        "low_consumption_rate": 0.1,
        "sustained_consumption_rate": 0.2,
        "voltage_anomaly_rate": 0.1,
        "sustained_duration": 4,
    }
    values.update(overrides)
    return SyntheticInjectionConfig(**values)


def test_injection_generates_all_categories_and_correct_synthetic_labels():
    frame = _frame()

    injected = inject_synthetic_anomalies(frame, config=_all_categories_config())

    assert injected["synthetic_anomaly_label"].sum() == 20
    categories = injected.loc[
        injected["synthetic_anomaly_label"] == 1,
        "synthetic_anomaly_category",
    ]
    assert set(categories) == {
        "consumption_spike",
        "unusually_low_consumption",
        "sustained_abnormal_consumption",
        "voltage_anomaly",
    }
    normal_rows = injected["synthetic_anomaly_label"] == 0
    anomalous_rows = injected["synthetic_anomaly_label"] == 1
    assert injected.loc[normal_rows, "synthetic_anomaly_category"].isna().all()
    assert injected["synthetic_anomaly_label"].dtype == np.dtype("uint8")
    assert injected.loc[anomalous_rows, "synthetic_injection_metadata"].notna().all()
    assert injected.loc[normal_rows, "synthetic_injection_metadata"].isna().all()


def test_injection_retains_original_values_and_bounds_synthetic_values():
    frame = _frame()
    injected = inject_synthetic_anomalies(frame, config=_all_categories_config())

    for column in (
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
    ):
        pd.testing.assert_series_equal(
            injected[f"original_{column}"],
            frame[column],
            check_names=False,
        )
    assert injected["synthetic_global_active_power"].between(0, 15).all()
    assert injected["synthetic_voltage"].between(180, 260).all()
    assert (
        injected.loc[
            injected["synthetic_anomaly_category"] == "consumption_spike",
            "synthetic_global_active_power",
        ]
        > injected.loc[
            injected["synthetic_anomaly_category"] == "consumption_spike",
            "original_global_active_power",
        ]
    ).all()
    assert (
        injected.loc[
            injected["synthetic_anomaly_category"] == "unusually_low_consumption",
            "synthetic_global_active_power",
        ]
        < injected.loc[
            injected["synthetic_anomaly_category"] == "unusually_low_consumption",
            "original_global_active_power",
        ]
    ).all()
    voltage_rows = injected["synthetic_anomaly_category"] == "voltage_anomaly"
    assert (
        injected.loc[voltage_rows, "synthetic_voltage"]
        != injected.loc[voltage_rows, "original_voltage"]
    ).all()
    sustained_rows = injected["synthetic_anomaly_category"] == ("sustained_abnormal_consumption")
    assert (
        injected.loc[sustained_rows, "synthetic_global_active_power"]
        > injected.loc[sustained_rows, "original_global_active_power"]
    ).all()


def test_injection_never_mutates_the_source_dataframe():
    frame = _frame()
    before = frame.copy(deep=True)

    inject_synthetic_anomalies(frame, config=_all_categories_config())

    pd.testing.assert_frame_equal(frame, before)


def test_injection_is_reproducible_with_a_fixed_random_seed():
    frame = _frame()
    config = _all_categories_config(random_seed=123)

    first = inject_synthetic_anomalies(frame, config=config)
    second = inject_synthetic_anomalies(frame, config=config)

    pd.testing.assert_frame_equal(first, second)


def test_injection_rates_are_configurable_including_zero_injections():
    frame = _frame(20)
    config = SyntheticInjectionConfig(
        random_seed=3,
        consumption_spike_rate=0.25,
        low_consumption_rate=0,
        sustained_consumption_rate=0,
        voltage_anomaly_rate=0,
        sustained_duration=4,
    )

    injected = inject_synthetic_anomalies(frame, config=config)

    assert injected["synthetic_anomaly_label"].sum() == 5
    assert (injected["synthetic_anomaly_category"] == "consumption_spike").sum() == 5
    assert (
        injected["synthetic_global_active_power"] != injected["original_global_active_power"]
    ).sum() == 5


def test_sustained_injections_are_non_overlapping_contiguous_segments():
    frame = _frame(40)
    config = SyntheticInjectionConfig(
        random_seed=12,
        consumption_spike_rate=0,
        low_consumption_rate=0,
        sustained_consumption_rate=0.4,
        voltage_anomaly_rate=0,
        sustained_duration=4,
    )

    injected = inject_synthetic_anomalies(frame, config=config)
    sustained = injected.loc[
        injected["synthetic_anomaly_category"] == "sustained_abnormal_consumption"
    ]

    assert len(sustained) == 16
    segment_ids = sustained["synthetic_injection_metadata"].map(lambda value: value["event_id"])
    assert segment_ids.nunique() == 4
    for _, segment in sustained.assign(_event_id=segment_ids).groupby("_event_id"):
        positions = segment.index.to_numpy()
        assert np.array_equal(positions, np.arange(positions.min(), positions.max() + 1))


def test_sustained_low_consumption_mode_changes_a_continuous_segment_downward():
    frame = _frame(20)
    config = SyntheticInjectionConfig(
        random_seed=2,
        consumption_spike_rate=0,
        low_consumption_rate=0,
        sustained_consumption_rate=0.4,
        voltage_anomaly_rate=0,
        sustained_duration=4,
        sustained_mode="low",
    )

    injected = inject_synthetic_anomalies(frame, config=config)
    sustained = injected["synthetic_anomaly_category"] == ("sustained_abnormal_consumption")

    assert sustained.sum() == 8
    assert (
        injected.loc[sustained, "synthetic_global_active_power"]
        < injected.loc[sustained, "original_global_active_power"]
    ).all()
    assert {
        metadata["parameters"]["direction"]
        for metadata in injected.loc[sustained, "synthetic_injection_metadata"]
    } == {"low"}


def test_only_held_out_copy_receives_synthetic_labels_and_values():
    frame = _frame(40)
    before = frame.copy(deep=True)
    split = chronological_train_validation_test_split(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    config = SyntheticInjectionConfig(
        random_seed=1,
        consumption_spike_rate=0.5,
        low_consumption_rate=0,
        sustained_consumption_rate=0,
        voltage_anomaly_rate=0,
    )

    synthetic_test = inject_synthetic_anomalies(split.test, config=config)

    assert synthetic_test["synthetic_anomaly_label"].sum() == 4
    assert not any(column.startswith("synthetic_") for column in split.train.columns)
    assert not any(column.startswith("synthetic_") for column in split.validation.columns)
    pd.testing.assert_frame_equal(frame, before)


def test_overlapping_injection_request_is_rejected_instead_of_silently_truncated():
    config = SyntheticInjectionConfig(
        consumption_spike_rate=0.6,
        low_consumption_rate=0.6,
        sustained_consumption_rate=0,
        voltage_anomaly_rate=0,
    )

    with pytest.raises(ValueError, match="overlap"):
        inject_synthetic_anomalies(_frame(10), config=config)


def test_insufficient_eligible_measurements_raise_a_clear_error():
    frame = _frame(10)
    frame["global_active_power"] = np.nan
    config = SyntheticInjectionConfig(
        consumption_spike_rate=0.5,
        low_consumption_rate=0,
        sustained_consumption_rate=0,
        voltage_anomaly_rate=0,
    )

    with pytest.raises(ValueError, match="eligible"):
        inject_synthetic_anomalies(frame, config=config)


def test_sustained_request_smaller_than_one_segment_injects_nothing():
    config = SyntheticInjectionConfig(
        consumption_spike_rate=0,
        low_consumption_rate=0,
        sustained_consumption_rate=0.2,
        voltage_anomaly_rate=0,
        sustained_duration=5,
    )

    injected = inject_synthetic_anomalies(_frame(10), config=config)

    assert injected["synthetic_anomaly_label"].sum() == 0


@pytest.mark.parametrize(
    "config",
    [
        SyntheticInjectionConfig(consumption_spike_rate=-0.1),
        SyntheticInjectionConfig(low_consumption_factor=1.0),
        SyntheticInjectionConfig(sustained_duration=1),
        SyntheticInjectionConfig(sustained_mode="unknown"),
        SyntheticInjectionConfig(minimum_voltage_volts=260, maximum_voltage_volts=180),
    ],
)
def test_invalid_synthetic_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        inject_synthetic_anomalies(_frame(), config=config)


def test_chronological_split_sorts_data_and_keeps_periods_disjoint():
    frame = _frame(20).iloc[::-1].copy()
    original = frame.copy(deep=True)

    split = chronological_train_validation_test_split(
        frame,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )

    assert (len(split.train), len(split.validation), len(split.test)) == (10, 5, 5)
    assert split.train["timestamp"].is_monotonic_increasing
    assert split.validation["timestamp"].is_monotonic_increasing
    assert split.test["timestamp"].is_monotonic_increasing
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert set(split.train.index).isdisjoint(split.validation.index)
    assert set(split.validation.index).isdisjoint(split.test.index)
    pd.testing.assert_frame_equal(frame, original)


def test_chronological_split_uses_remainder_as_test_when_test_fraction_omitted():
    split = chronological_train_validation_test_split(
        _frame(20),
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert (len(split.train), len(split.validation), len(split.test)) == (12, 4, 4)


@pytest.mark.parametrize(
    ("row_count", "train_fraction", "validation_fraction", "test_fraction"),
    [
        (0, 0.7, 0.15, None),
        (2, 0.5, 0.25, 0.25),
        (10, 0.9, 0.05, 0.05),
        (10, 0.5, 0.5, 0),
        (10, 0.5, 0.2, 0.2),
    ],
)
def test_chronological_split_rejects_empty_or_inconsistent_partition_sizes(
    row_count,
    train_fraction,
    validation_fraction,
    test_fraction,
):
    with pytest.raises(ValueError):
        chronological_train_validation_test_split(
            _frame(row_count),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


def test_chronological_split_rejects_invalid_or_duplicate_timestamps():
    frame = _frame(5)
    frame.loc[frame.index[1], "timestamp"] = frame.loc[frame.index[0], "timestamp"]

    with pytest.raises(ValueError, match="unique"):
        chronological_train_validation_test_split(frame)
