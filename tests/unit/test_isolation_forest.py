from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartmeter.anomaly_detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)


def _frame(
    values: list[float | None],
    *,
    start: str = "2024-01-01",
    index: list[int] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(values), freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "global_active_power": values,
            "voltage": [230.0] * len(values),
        },
        index=index,
    )


def _detector(**overrides) -> IsolationForestDetector:
    options = {
        "n_estimators": 32,
        "contamination": 0.1,
        "random_state": 17,
        "n_jobs": 1,
        "feature_columns": ("global_active_power", "voltage"),
    }
    options.update(overrides)
    return IsolationForestDetector(IsolationForestConfig(**options))


def test_fit_uses_only_training_rows_and_median_imputation_is_training_only():
    training = _frame([1.0, None, 2.0, 3.0])
    detector = _detector().fit(training)
    held_out = _frame([1_000_000.0, None], start="2024-02-01")

    result = detector.predict(held_out)

    assert detector.n_training_rows_ == len(training)
    assert detector.imputer_.statistics_.tolist() == [2.0, 230.0]
    assert result.loc[0, "global_active_power"] == 1_000_000.0
    assert pd.isna(result.loc[1, "global_active_power"])
    assert result["isolation_forest_prediction"].isin([-1, 1]).all()
    assert result["isolation_forest_anomaly_label"].isin([0, 1]).all()
    assert np.isfinite(result["isolation_forest_anomaly_score"]).all()


def test_predictions_preserve_input_rows_index_and_do_not_mutate_source():
    training = _frame(np.random.default_rng(4).normal(1.0, 0.05, 100).tolist())
    held_out = _frame([1.0, 20.0], start="2024-02-01", index=[12, 4])
    before = held_out.copy(deep=True)

    result = _detector().fit(training).predict(held_out)

    assert len(result) == len(held_out)
    assert result.index.tolist() == [12, 4]
    assert result["timestamp"].tolist() == held_out["timestamp"].tolist()
    assert result["isolation_forest_anomaly_label"].tolist() == [0, 1]
    pd.testing.assert_frame_equal(held_out, before)


def test_default_features_skip_timestamps_and_synthetic_labels():
    training = _frame([1.0, 2.0, 3.0, 4.0])
    training["synthetic_anomaly_label"] = np.array([0, 0, 0, 1], dtype=np.uint8)
    training["synthetic_global_active_power"] = [1.0, 2.0, 3.0, 100.0]

    with pytest.raises(ValueError, match="label or target columns"):
        IsolationForestDetector().fit(training)

    training = training.drop(columns="synthetic_anomaly_label")
    detector = IsolationForestDetector(
        IsolationForestConfig(n_estimators=8, random_state=1, n_jobs=1)
    ).fit(training)
    assert detector.feature_columns_ == ("global_active_power", "voltage")


def test_prediction_rejects_missing_fitted_feature_columns():
    detector = _detector().fit(_frame([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="missing fitted feature columns"):
        detector.predict(pd.DataFrame({"global_active_power": [1.0]}))


def test_all_missing_features_are_safe_when_another_training_feature_is_finite():
    training = _frame([1.0, 2.0, 3.0])
    training["empty_feature"] = np.nan
    detector = _detector(feature_columns=("global_active_power", "empty_feature")).fit(training)

    result = detector.predict(training)

    assert detector.imputer_.statistics_.tolist() == [2.0, 0.0]
    assert len(result) == len(training)


def test_infinite_measurements_are_treated_as_missing_for_imputation():
    training = _frame([1.0, np.inf, 3.0])
    detector = _detector(feature_columns=("global_active_power",)).fit(training)

    assert detector.imputer_.statistics_.tolist() == [2.0]
    assert np.isfinite(detector.predict(_frame([np.inf]))["isolation_forest_anomaly_score"]).all()


def test_fit_rejects_label_columns_and_requires_finite_training_data():
    labeled = _frame([1.0, 2.0, 3.0])
    labeled["synthetic_anomaly_label"] = [0, 0, 1]
    with pytest.raises(ValueError, match="label or target columns"):
        _detector().fit(labeled)

    no_finite_values = _frame([None, None, None])
    with pytest.raises(ValueError, match="no finite values"):
        _detector(feature_columns=("global_active_power",)).fit(no_finite_values)


def test_fit_requires_two_training_observations_and_predict_requires_fit():
    detector = _detector()
    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.predict(_frame([1.0]))
    with pytest.raises(ValueError, match="at least two"):
        detector.fit(_frame([1.0]))


def test_model_artifact_round_trip_preserves_predictions(tmp_path):
    training = _frame([1.0, 1.1, 0.9, 1.2, 20.0])
    evaluation = _frame([1.0, 20.0], start="2024-02-01")
    detector = _detector().fit(training)
    expected = detector.predict(evaluation)
    model_path = tmp_path / "nested" / "isolation_forest.joblib"

    saved_path = detector.save(model_path)
    loaded = IsolationForestDetector.load(saved_path)
    actual = loaded.predict(evaluation)

    assert saved_path == model_path
    pd.testing.assert_series_equal(
        actual["isolation_forest_anomaly_label"],
        expected["isolation_forest_anomaly_label"],
    )
    pd.testing.assert_series_equal(
        actual["isolation_forest_anomaly_score"],
        expected["isolation_forest_anomaly_score"],
    )


@pytest.mark.parametrize(
    "config_values",
    [
        {"n_estimators": 0},
        {"contamination": 0.0},
        {"contamination": 0.6},
        {"max_features": 0.0},
        {"feature_columns": ("synthetic_anomaly_label",)},
    ],
)
def test_invalid_configuration_is_rejected(config_values):
    with pytest.raises(ValueError):
        IsolationForestConfig(**config_values)
