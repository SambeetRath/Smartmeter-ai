from __future__ import annotations

from contextlib import nullcontext
from datetime import timezone
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from smartmeter.services import detection_service
from smartmeter.services.detection_service import LostDetectionLeaseError

UTC = timezone.utc


class FakeFactory:
    def __init__(self) -> None:
        self.session = object()

    def __call__(self):
        return nullcontext(self.session)

    def begin(self):
        return nullcontext(self.session)


def test_detection_service_maps_all_predictions_and_persists_atomically(monkeypatch):
    run_id = uuid4()
    lease_token = uuid4()
    timestamps = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "id": [101, 102],
            "timestamp": timestamps,
            "global_active_power": [1.0, np.nan],
            "global_reactive_power": [0.1, 0.1],
            "voltage": [230.0, 230.0],
            "global_intensity": [5.0, 5.0],
            "sub_metering_1": [0.0, 0.0],
            "sub_metering_2": [1.0, 1.0],
            "sub_metering_3": [2.0, 2.0],
        }
    )
    run = SimpleNamespace(
        id=run_id,
        status="running",
        lease_token=lease_token,
        meter=SimpleNamespace(meter_identifier="service-test-meter"),
        algorithm="global_iqr",
        start_timestamp=None,
        end_timestamp=None,
    )
    captured = {}

    class FakeDetectionRunRepository:
        def get_by_id(self, session, requested_id):
            assert requested_id == run_id
            return run

        def mark_completed(self, session, requested_id, requested_token, **kwargs):
            assert requested_id == run_id
            assert requested_token == lease_token
            captured.update(kwargs)
            return True

    def fake_pipeline_runner(data, *, config, meter_identifier):
        assert "id" not in data.columns
        assert config.include_normal_predictions is True
        assert config.detector_names == ("global_iqr",)
        assert meter_identifier == "service-test-meter"
        return SimpleNamespace(
            summary={
                "run_id": "pipeline-generated-id",
                "training_rows": 1,
                "configuration": {"detectors": ["global_iqr"]},
                "detectors": {
                    "global_iqr": {
                        "evaluated_observations": 2,
                        "anomaly_predictions": 0,
                        "normal_predictions_omitted_from_csv": 1,
                        "unavailable_predictions": 1,
                    }
                },
            },
            results=pd.DataFrame(
                [
                    {
                        "timestamp": timestamps[0],
                        "result_kind": "normal",
                        "anomaly_score": 0.0,
                        "anomaly_category": None,
                        "detector_metadata": '{"available": true}',
                    },
                    {
                        "timestamp": timestamps[1],
                        "result_kind": "unavailable",
                        "anomaly_score": np.nan,
                        "anomaly_category": None,
                        "detector_metadata": '{"reason": "missing_measurement"}',
                    },
                ]
            ),
        )

    monkeypatch.setattr(
        detection_service, "load_postgresql_dataset", lambda *_args, **_kwargs: frame
    )
    monkeypatch.setattr(
        detection_service,
        "DetectionRunRepository",
        FakeDetectionRunRepository,
    )

    detection_service.process_detection_run(
        run_id,
        lease_token,
        session_factory=FakeFactory(),
        pipeline_runner=fake_pipeline_runner,
    )

    assert captured["training_readings"] == 1
    assert captured["readings_processed"] == 2
    assert captured["summary"]["run_id"] == str(run_id)
    assert captured["normal_count"] == 1
    assert captured["unavailable_count"] == 1
    assert [row["reading_id"] for row in captured["results"]] == [101, 102]
    assert captured["results"][1]["anomaly_score"] is None
    assert captured["results"][1]["metadata_json"] == {"reason": "missing_measurement"}


def test_detection_service_rejects_lost_lease_before_running_pipeline(monkeypatch):
    run_id = uuid4()
    lease_token = uuid4()

    class FakeDetectionRunRepository:
        def get_by_id(self, _session, _run_id):
            return SimpleNamespace(
                id=run_id,
                status="running",
                lease_token=uuid4(),
                meter=SimpleNamespace(meter_identifier="service-test-meter"),
                algorithm="global_iqr",
                start_timestamp=None,
                end_timestamp=None,
            )

    monkeypatch.setattr(
        detection_service,
        "DetectionRunRepository",
        FakeDetectionRunRepository,
    )
    monkeypatch.setattr(
        detection_service,
        "load_postgresql_dataset",
        lambda *_args, **_kwargs: pytest.fail("A stale worker must not load data."),
    )

    with pytest.raises(LostDetectionLeaseError):
        detection_service.process_detection_run(
            run_id,
            lease_token,
            session_factory=FakeFactory(),
        )
