from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smartmeter.anomaly_detection.features import MEASUREMENT_COLUMNS
from smartmeter.anomaly_detection.isolation_forest import IsolationForestConfig
from smartmeter.anomaly_detection.pipeline import (
    DETECTOR_NAMES,
    PipelineConfig,
    execute_anomaly_pipeline,
    load_postgresql_dataset,
    run_anomaly_pipeline,
)


def _frame(row_count: int = 240) -> pd.DataFrame:
    position = np.arange(row_count)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-06 08:15", periods=row_count, freq="7D"),
            "global_active_power": 1.0 + (position % 17) / 10.0,
            "global_reactive_power": 0.1 + (position % 7) / 100.0,
            "voltage": 230.0 + (position % 5) / 10.0,
            "global_intensity": 5.0 + (position % 11) / 10.0,
            "sub_metering_1": (position % 4) / 10.0,
            "sub_metering_2": 1.0 + (position % 3) / 10.0,
            "sub_metering_3": 2.0 + (position % 9) / 10.0,
        }
    )


def _config(**overrides) -> PipelineConfig:
    values = {
        "detector_names": DETECTOR_NAMES,
        "rolling_window_size": 5,
        "rolling_minimum_history": 3,
        "time_of_day_minimum_history": 2,
        "isolation_forest_config": IsolationForestConfig(
            n_estimators=16,
            contamination=0.1,
            random_state=27,
            n_jobs=1,
        ),
    }
    values.update(overrides)
    return PipelineConfig(**values)


def test_pipeline_runs_all_detectors_with_one_consistent_sparse_schema():
    frame = _frame()
    run = run_anomaly_pipeline(frame, config=_config())

    assert set(run.summary["detectors"]) == set(DETECTOR_NAMES)
    assert set(run.results["detector_name"]) <= set(DETECTOR_NAMES)
    assert run.results.columns.tolist() == [
        "run_id",
        "timestamp",
        "meter_identifier",
        *MEASUREMENT_COLUMNS,
        "detector_name",
        "anomaly_prediction",
        "prediction_available",
        "anomaly_score",
        "anomaly_category",
        "result_kind",
        "detector_metadata",
    ]
    assert run.results["run_id"].nunique() == 1
    assert run.results["meter_identifier"].eq("uci_household_1").all()
    assert run.results["result_kind"].isin(["anomaly", "unavailable"]).all()
    assert run.results["anomaly_category"].isna().all()
    assert run.results["timestamp"].is_monotonic_increasing
    assert run.summary["input_rows"] == len(frame)
    assert run.summary["training_rows"] == 168
    assert run.summary["heldout_rows"] == 72
    for detector_summary in run.summary["detectors"].values():
        assert (
            detector_summary["available_predictions"] + detector_summary["unavailable_predictions"]
            == detector_summary["evaluated_observations"]
        )
        assert (
            detector_summary["anomaly_predictions"]
            + detector_summary["normal_predictions_omitted_from_csv"]
            == detector_summary["available_predictions"]
        )


@pytest.mark.parametrize("detector_name", DETECTOR_NAMES)
def test_pipeline_runs_each_detector_independently(detector_name):
    run = run_anomaly_pipeline(
        _frame(),
        config=_config(detector_names=(detector_name,)),
    )

    assert run.summary["configuration"]["detectors"] == [detector_name]
    assert set(run.summary["detectors"]) == {detector_name}
    assert set(run.results["detector_name"]) <= {detector_name}


def test_pipeline_sorts_input_without_mutating_source_or_dropping_rows():
    source = _frame(80).iloc[::-1].reset_index(drop=True)
    untouched = source.copy(deep=True)

    run = run_anomaly_pipeline(
        source,
        config=_config(detector_names=("global_iqr",)),
    )

    pd.testing.assert_frame_equal(source, untouched)
    assert run.summary["input_rows"] == len(source)
    assert run.summary["heldout_rows"] == 24
    assert (
        run.summary["heldout_start"]
        == pd.Timestamp(
            "2021-02-01 08:15",
            tz="UTC",
        ).isoformat()
    )


def test_global_iqr_flags_heldout_spike_using_training_thresholds():
    frame = _frame(100)
    frame.loc[90, "global_active_power"] = 100.0
    run = run_anomaly_pipeline(
        frame,
        config=_config(detector_names=("global_iqr",)),
    )

    flagged = run.results.loc[run.results["result_kind"].eq("anomaly")]
    assert len(flagged) == 1
    assert flagged.iloc[0]["timestamp"] == pd.Timestamp(frame.loc[90, "timestamp"], tz="UTC")
    assert bool(flagged.iloc[0]["anomaly_prediction"])
    assert (
        run.summary["detectors"]["global_iqr"]["configuration"]["threshold_fit_partition"]
        == "chronological_training"
    )


def test_past_only_detector_exposes_time_of_day_warmup_as_unavailable():
    frame = _frame(80)
    frame["timestamp"] = pd.date_range("2024-01-01", periods=len(frame), freq="min")
    run = run_anomaly_pipeline(
        frame,
        config=_config(
            detector_names=("time_of_day",),
            time_of_day_minimum_history=2,
        ),
    )

    summary = run.summary["detectors"]["time_of_day"]
    assert summary["available_predictions"] == 0
    assert summary["unavailable_predictions"] == run.summary["heldout_rows"]
    assert run.results["result_kind"].eq("unavailable").all()
    assert run.results["anomaly_prediction"].isna().all()


def test_missing_measurements_are_unavailable_not_normal_predictions():
    frame = _frame(100)
    frame.loc[70:, "global_active_power"] = np.nan
    run = run_anomaly_pipeline(
        frame,
        config=_config(detector_names=("rolling_mean_std",)),
    )

    unavailable = run.results.loc[run.results["result_kind"].eq("unavailable")]
    assert len(unavailable) == 30
    assert unavailable["global_active_power"].isna().all()
    assert unavailable["anomaly_prediction"].isna().all()
    assert (
        run.summary["detectors"]["rolling_mean_std"]["unavailable_reasons"]["missing_measurement"]
        == 30
    )


def test_zero_variance_alert_does_not_borrow_a_score_from_a_non_flagged_column():
    frame = _frame(100)
    frame["global_active_power"] = 10.0
    frame.loc[70, "global_active_power"] = 100.0
    run = run_anomaly_pipeline(
        frame,
        config=_config(
            detector_names=("rolling_mean_std",),
            measurement_columns=("global_active_power", "voltage"),
        ),
    )

    active_power_alert = run.results.loc[
        run.results["timestamp"].eq(pd.Timestamp(frame.loc[70, "timestamp"], tz="UTC"))
    ]
    assert len(active_power_alert) == 1
    assert active_power_alert.iloc[0]["result_kind"] == "anomaly"
    assert pd.isna(active_power_alert.iloc[0]["anomaly_score"])


def test_pipeline_does_not_accept_synthetic_labels_for_inference():
    frame = _frame()
    frame["synthetic_anomaly_label"] = 0

    with pytest.raises(ValueError, match="ground-truth or target columns"):
        run_anomaly_pipeline(frame, config=_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.iloc[:0], "no observations"),
        (lambda frame: frame.assign(timestamp="invalid"), "invalid timestamps"),
        (
            lambda frame: frame.assign(global_active_power="not-a-number"),
            "non-numeric values",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate timestamps",
        ),
    ],
)
def test_pipeline_rejects_empty_or_invalid_input(mutation, message):
    with pytest.raises(ValueError, match=message):
        run_anomaly_pipeline(mutation(_frame()), config=_config())


def test_pipeline_rejects_insufficient_training_partition_for_isolation_forest():
    with pytest.raises(ValueError, match="at least two training"):
        run_anomaly_pipeline(
            _frame(2),
            config=_config(train_fraction=0.5),
        )


def test_pipeline_output_files_are_generated_and_existing_outputs_are_protected(tmp_path):
    source_path = tmp_path / "input.parquet"
    results_path = tmp_path / "nested" / "results.csv"
    summary_path = tmp_path / "nested" / "summary.json"
    _frame(100).to_parquet(source_path, index=False)
    config = _config(
        input_path=source_path,
        results_path=results_path,
        summary_path=summary_path,
    )

    run = execute_anomaly_pipeline(config)
    saved_results = pd.read_csv(results_path)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert results_path.is_file()
    assert summary_path.is_file()
    assert len(saved_results) == len(run.results)
    assert saved_results.columns.tolist() == run.results.columns.tolist()
    assert saved_summary["run_id"] == run.summary["run_id"]
    assert saved_summary["result_file_size_bytes"] == results_path.stat().st_size
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        execute_anomaly_pipeline(config)


def test_pipeline_allows_explicit_output_overwrite(tmp_path):
    source_path = tmp_path / "input.parquet"
    results_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.json"
    _frame(100).to_parquet(source_path, index=False)
    first = execute_anomaly_pipeline(
        _config(
            detector_names=("global_iqr",),
            input_path=source_path,
            results_path=results_path,
            summary_path=summary_path,
        )
    )
    second = execute_anomaly_pipeline(
        _config(
            detector_names=("global_iqr",),
            input_path=source_path,
            results_path=results_path,
            summary_path=summary_path,
            overwrite=True,
        )
    )

    assert first.summary["run_id"] != second.summary["run_id"]
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["run_id"] == second.summary["run_id"]
    )


def test_pipeline_reproducibility_for_fixed_isolation_forest_configuration():
    config = _config(detector_names=("isolation_forest",))

    first = run_anomaly_pipeline(_frame(), config=config)
    second = run_anomaly_pipeline(_frame(), config=config)

    pd.testing.assert_frame_equal(
        first.results.drop(columns=["run_id", "timestamp"]),
        second.results.drop(columns=["run_id", "timestamp"]),
        check_dtype=False,
    )


def test_postgresql_reader_uses_bounded_ordered_pages_and_preserves_nulls(monkeypatch):
    rows = [
        SimpleNamespace(
            timestamp=timestamp,
            **{
                column: (None if column == "global_active_power" and position == 2 else 1.0)
                for column in MEASUREMENT_COLUMNS
            },
        )
        for position, timestamp in enumerate(pd.date_range("2024-01-01", periods=5, freq="min"))
    ]
    page_offsets = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSessionFactory:
        def __call__(self):
            return FakeSession()

    class FakeMeterRepository:
        def get_by_identifier(self, _session, _identifier):
            return SimpleNamespace(id=12)

    class FakeReadingRepository:
        def count_for_meter(self, _session, _meter_id):
            return len(rows)

        def get_for_meter(self, _session, _meter_id, *, limit, offset):
            page_offsets.append(offset)
            return rows[offset : offset + limit]

    monkeypatch.setattr(
        "smartmeter.anomaly_detection.pipeline.MeterRepository",
        FakeMeterRepository,
    )
    monkeypatch.setattr(
        "smartmeter.anomaly_detection.pipeline.ReadingRepository",
        FakeReadingRepository,
    )

    loaded = load_postgresql_dataset(
        "test-meter",
        session_factory=FakeSessionFactory(),
        page_size=2,
    )

    assert len(loaded) == 5
    assert page_offsets == [0, 2, 4]
    assert loaded["timestamp"].is_monotonic_increasing
    assert pd.isna(loaded.loc[2, "global_active_power"])


def test_pipeline_config_rejects_invalid_detector_and_iqr_threshold():
    with pytest.raises(ValueError, match="Unknown detector"):
        PipelineConfig(detector_names=("unknown",))
    with pytest.raises(ValueError, match="iqr_multiplier"):
        PipelineConfig(iqr_multiplier=float("inf"))
