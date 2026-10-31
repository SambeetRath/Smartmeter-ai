from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smartmeter.anomaly_detection.evaluation import (
    EvaluationConfig,
    calculate_binary_metrics,
    evaluate_predictions,
    run_anomaly_evaluation,
    write_evaluation_report,
)
from smartmeter.anomaly_detection.features import MEASUREMENT_COLUMNS, engineer_features
from smartmeter.anomaly_detection.isolation_forest import IsolationForestConfig
from smartmeter.anomaly_detection.statistical import (
    GlobalIQRDetector,
    detect_global_iqr,
)
from smartmeter.anomaly_detection.synthetic_data import SyntheticInjectionConfig


def _frame(row_count: int = 240) -> pd.DataFrame:
    position = np.arange(row_count)
    timestamps = pd.date_range("2020-01-06 08:15", periods=row_count, freq="7D")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "global_active_power": 1.0 + (position % 17) / 10.0,
            "global_reactive_power": 0.1 + (position % 7) / 100.0,
            "voltage": 230.0 + (position % 5) / 10.0,
            "global_intensity": 5.0 + (position % 11) / 10.0,
            "sub_metering_1": (position % 4) / 10.0,
            "sub_metering_2": 1.0 + (position % 3) / 10.0,
            "sub_metering_3": 2.0 + (position % 9) / 10.0,
        },
        index=position % 13,
    )


def _evaluation_config(**overrides) -> EvaluationConfig:
    values = {
        "dataset_identifier": "synthetic-unit-fixture",
        "synthetic_config": SyntheticInjectionConfig(
            random_seed=11,
            consumption_spike_rate=0.05,
            low_consumption_rate=0.05,
            sustained_consumption_rate=0.1,
            voltage_anomaly_rate=0.05,
            sustained_duration=2,
        ),
        "isolation_forest_config": IsolationForestConfig(
            n_estimators=16,
            contamination=0.1,
            random_state=27,
            n_jobs=1,
        ),
        "contamination_candidates": (0.05, 0.1),
        "rolling_window_size": 5,
        "rolling_minimum_history": 3,
        "time_of_day_minimum_history": 2,
    }
    values.update(overrides)
    return EvaluationConfig(**values)


def test_binary_metrics_calculate_counts_and_metrics_with_fixed_class_order():
    result = calculate_binary_metrics(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
    )

    assert result["class_order"] == [
        {"label": 0, "name": "normal"},
        {"label": 1, "name": "anomaly"},
    ]
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert (result["true_positive"], result["true_negative"]) == (2, 2)
    assert (result["false_positive"], result["false_negative"]) == (1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)
    assert result["specificity"] == pytest.approx(2 / 3)


def test_zero_division_uses_documented_finite_zero_policy():
    result = calculate_binary_metrics([0, 0], [0, 0])

    assert result["precision"] == result["recall"] == result["f1_score"] == 0.0
    assert result["false_positive_rate"] == 0.0
    assert result["specificity"] == 1.0
    assert result["zero_division_policy"] == "0.0 when the metric denominator is zero"
    assert all(np.isfinite(value) for value in result.values() if isinstance(value, float))


def test_zero_normal_support_uses_zero_for_undefined_specificity_and_fpr():
    result = calculate_binary_metrics([1, 1], [1, 1])

    assert result["true_positive"] == 2
    assert result["false_positive_rate"] == 0.0
    assert result["specificity"] == 0.0


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        ([0, 2], [0, 1]),
        ([0, None], [0, 1]),
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
        ([], []),
    ],
)
def test_metrics_reject_invalid_binary_inputs(truth, prediction):
    with pytest.raises(ValueError):
        calculate_binary_metrics(truth, prediction)


def test_series_alignment_is_ordered_and_supports_duplicate_index_labels():
    truth = pd.Series([0, 1, 0], index=[7, 7, 2])
    prediction = pd.Series([0, 1, 1], index=[7, 7, 2])

    result = calculate_binary_metrics(truth, prediction)

    assert result["confusion_matrix"] == [[1, 1], [0, 1]]
    with pytest.raises(ValueError, match="identical ordered indexes"):
        calculate_binary_metrics(truth, prediction.iloc[::-1])


def test_evaluate_predictions_excludes_missing_or_unavailable_values_from_metrics():
    result = evaluate_predictions(
        [0, 1, 0, 1],
        [0, 1, pd.NA, 1],
        categories=["normal", "consumption_spike", "normal", "voltage_anomaly"],
    )

    assert result["total_observations"] == 4
    assert result["available_observations"] == 3
    assert result["excluded_observations"] == 1
    assert result["coverage"] == pytest.approx(0.75)
    assert result["metrics"]["true_positive"] == 2
    assert result["metrics"]["true_negative"] == 1


def test_missing_ground_truth_and_invalid_available_predictions_are_rejected():
    with pytest.raises(ValueError, match="y_true must not contain missing"):
        evaluate_predictions([0, None], [0, 1])
    with pytest.raises(ValueError, match="only binary values"):
        evaluate_predictions([0, 1], [0, 2])
    with pytest.raises(ValueError, match="only boolean values"):
        evaluate_predictions([0, 1], [0, 1], prediction_available=[True, None])


def test_unavailable_predictions_are_not_treated_as_normal_and_zero_coverage_is_explicit():
    result = evaluate_predictions([0, 1], [pd.NA, pd.NA])

    assert result["available_observations"] == 0
    assert result["coverage"] == 0.0
    assert result["metrics"] is None


def test_category_metrics_separate_other_anomalies_from_normal_observations():
    result = evaluate_predictions(
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        categories=[
            "normal",
            "consumption_spike",
            "voltage_anomaly",
            "unusually_low_consumption",
        ],
    )
    spike = result["category_metrics"]["consumption_spike"]

    assert spike["target_category_support"] == 1
    assert spike["other_anomaly_support"] == 2
    assert spike["normal_support"] == 1
    assert spike["other_anomaly_predicted_positive"] == 1
    assert spike["normal_predicted_positive"] == 1
    assert spike["metrics"]["class_order"] == [
        {"label": 0, "name": "non_target"},
        {"label": 1, "name": "consumption_spike"},
    ]
    assert "not treated as normal" in spike["negative_class_note"]
    assert spike["metrics"]["false_positive"] == 2


def test_unrecognized_anomaly_categories_are_counted_not_discarded():
    result = evaluate_predictions(
        [1, 0],
        [1, 0],
        categories=["new_synthetic_category", "normal"],
    )

    assert result["category_metrics"]["unrecognized_category_counts"] == {
        "new_synthetic_category": 1
    }


def test_category_labels_must_agree_with_binary_ground_truth():
    with pytest.raises(ValueError, match="every normal observation"):
        evaluate_predictions(
            [0, 1],
            [0, 1],
            categories=["voltage_anomaly", "normal"],
        )


def test_fitted_global_iqr_uses_training_thresholds_for_unseen_rows():
    training = pd.DataFrame({"global_active_power": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]})
    detector = GlobalIQRDetector(
        columns=["global_active_power"],
        iqr_multiplier=1.5,
    ).fit(training)
    training_bounds = detector.bounds_["global_active_power"]

    predictions = detector.predict(pd.DataFrame({"global_active_power": [2.5, 100.0]}))

    assert detector.bounds_["global_active_power"] == training_bounds
    assert predictions["row_index"].tolist() == [1]
    assert predictions["value"].tolist() == [100.0]
    assert predictions["method"].tolist() == ["global_iqr"]


def test_fitted_global_iqr_matches_existing_rule_when_fit_and_predict_share_data():
    frame = pd.DataFrame({"global_active_power": [1.0, 2.0, 2.0, 3.0, 100.0]})
    existing = detect_global_iqr(
        frame,
        columns=["global_active_power"],
        iqr_multiplier=1.5,
    )
    fitted = (
        GlobalIQRDetector(
            columns=["global_active_power"],
            iqr_multiplier=1.5,
        )
        .fit(frame)
        .predict(frame)
    )

    assert fitted["row_index"].tolist() == existing["row_index"].tolist()
    assert fitted["lower_bound"].tolist() == existing["lower_bound"].tolist()
    assert fitted["upper_bound"].tolist() == existing["upper_bound"].tolist()


def test_evaluation_config_rejects_invalid_contamination_and_history():
    with pytest.raises(ValueError, match="contamination_candidates"):
        EvaluationConfig(contamination_candidates=(0.1, 0.1))
    with pytest.raises(ValueError, match="rolling_minimum_history"):
        EvaluationConfig(rolling_window_size=3, rolling_minimum_history=4)
    with pytest.raises(ValueError, match="time_of_day_minimum_history"):
        EvaluationConfig(time_of_day_minimum_history=0)


def test_end_to_end_evaluation_keeps_splits_and_contamination_sweep_separate(
    monkeypatch,
):
    from smartmeter.anomaly_detection import evaluation as evaluation_module

    frame = _frame()
    before = frame.copy(deep=True)
    original_fit = evaluation_module.IsolationForestDetector.fit
    fit_calls: list[tuple[int, tuple[str, ...]]] = []

    def record_fit(detector, training_frame):
        fit_calls.append((len(training_frame), tuple(training_frame.columns)))
        return original_fit(detector, training_frame)

    monkeypatch.setattr(evaluation_module.IsolationForestDetector, "fit", record_fit)
    report = run_anomaly_evaluation(frame, config=_evaluation_config())

    assert report["split_sizes"] == {"train": 168, "validation": 36, "test": 36}
    assert report["test"]["partition_rows"] == 36
    assert report["test"]["synthetic_label_counts"]["anomaly"] > 0
    assert set(report["test"]["detectors"]) == {
        "global_iqr",
        "rolling_mean_std",
        "time_of_day",
        "isolation_forest",
    }
    assert report["test"]["shared_comparison"]["shared_rows"] > 0
    assert report["validation"]["isolation_forest_contamination_sweep"][
        "selection_policy"
    ].startswith("Validation-only")
    assert set(report["validation"]["isolation_forest_contamination_sweep"]["candidates"]) == {
        "0.05",
        "0.1",
    }
    assert "isolation_forest_contamination_sweep" not in report["test"]
    assert report["test"]["detectors"]["global_iqr"]["evaluation_mode"] == "train_fitted_threshold"
    assert all(
        training_rows == report["split_sizes"]["train"]
        and "synthetic_anomaly_label" not in columns
        and "synthetic_anomaly_category" not in columns
        for training_rows, columns in fit_calls
    )
    assert len(fit_calls) == 4  # Validation baseline, test model, and two validation candidates.
    pd.testing.assert_frame_equal(frame, before)


def test_warmup_unavailability_is_reported_instead_of_counted_as_normal():
    config = _evaluation_config(
        synthetic_config=SyntheticInjectionConfig(
            consumption_spike_rate=0,
            low_consumption_rate=0,
            sustained_consumption_rate=0,
            voltage_anomaly_rate=0,
        ),
        contamination_candidates=(),
        time_of_day_minimum_history=1_000,
    )
    report = run_anomaly_evaluation(_frame(60), config=config)
    time_of_day = report["test"]["detectors"]["time_of_day"]

    assert time_of_day["available_observations"] == 0
    assert time_of_day["excluded_observations"] == time_of_day["total_observations"]
    assert time_of_day["metrics"] is None
    assert (
        time_of_day["excluded_reasons"]["insufficient_history"]
        == (time_of_day["total_observations"])
    )


def test_evaluation_is_reproducible_except_for_run_metadata_and_runtime():
    frame = _frame()
    first = run_anomaly_evaluation(frame, config=_evaluation_config())
    second = run_anomaly_evaluation(frame, config=_evaluation_config())

    assert first["test"] == second["test"]
    assert first["validation"] == second["validation"]
    assert first["split_sizes"] == second["split_sizes"]
    assert first["performance"]["evaluation_runtime_seconds"] >= 0
    assert first["generated_at_utc"] != ""


def test_evaluation_rejects_preexisting_labels_duplicate_times_and_missing_columns():
    frame = _frame()
    frame["synthetic_anomaly_label"] = 0
    with pytest.raises(ValueError, match="must not contain ground-truth"):
        run_anomaly_evaluation(frame, config=_evaluation_config())

    duplicate_times = _frame()
    duplicate_times.iloc[1, duplicate_times.columns.get_loc("timestamp")] = duplicate_times.iloc[0][
        "timestamp"
    ]
    with pytest.raises(ValueError, match="must be unique"):
        run_anomaly_evaluation(duplicate_times, config=_evaluation_config())

    missing_column = _frame().drop(columns="voltage")
    with pytest.raises(ValueError, match="missing required evaluation columns"):
        run_anomaly_evaluation(missing_column, config=_evaluation_config())


def test_report_writer_creates_parent_directory_refuses_overwrite_and_rejects_nan(
    tmp_path,
):
    target = tmp_path / "nested" / "report.json"
    report = {
        "schema_version": "1.0",
        "metrics": {"precision": 0.0},
    }

    assert write_evaluation_report(report, target) == target
    assert json.loads(target.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_report(report, target)
    write_evaluation_report({"updated": True}, target, overwrite=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"updated": True}
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_evaluation_report({"metric": float("nan")}, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_report_writer_handles_nonfinite_metric_values_from_numpy(tmp_path):
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_evaluation_report(
            {"metric": np.float64(np.inf)},
            tmp_path / "invalid.json",
        )


def test_default_model_feature_names_are_in_engineered_feature_output():
    from smartmeter.anomaly_detection.evaluation import DEFAULT_MODEL_FEATURES

    frame = _frame(8)
    engineered = engineer_features(frame, columns=("global_active_power", "voltage"))

    assert set(DEFAULT_MODEL_FEATURES).issubset(engineered.columns)
    assert set(MEASUREMENT_COLUMNS).issubset(frame.columns)
