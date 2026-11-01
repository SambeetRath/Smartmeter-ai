from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from smartmeter_dashboard import detection_data
from smartmeter_dashboard.api_client import (
    AnomaliesPageResponse,
    AnomalyResponse,
    DataQualityResponse,
    DetectionRunResponse,
    DetectionRunsPageResponse,
    MeasurementStats,
    TimestampGapSummary,
)
from smartmeter_dashboard.components import common
from smartmeter_dashboard.dashboard_pages import (
    anomaly_detection,
    data_quality,
    detection_runs,
)
from smartmeter_dashboard.exceptions import APIConnectionError
from smartmeter_dashboard.state import DashboardFilters

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)
FILTERS = DashboardFilters("meter-a", START.date(), START.date())
RUN_ID = UUID("c0bfe534-57fc-4ed8-bd06-cbf482b47861")


class FakeColumn:
    def __init__(self, outputs, state):
        self.outputs = outputs
        self.state = state

    def selectbox(self, label, *, options, key=None, format_func=None, **_kwargs):
        value = self.state.get(key, options[0])
        self.outputs["selectboxes"].append((label, tuple(options), value))
        return value

    def button(self, label, *, disabled=False, key=None, **_kwargs):
        self.outputs["buttons"].append((label, disabled, key))
        return self.outputs["button_results"].get(label, False) and not disabled

    def text_input(self, _label, **_kwargs):
        return ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _install_mocks(monkeypatch, *, state=None, button_results=None):
    state = state if state is not None else {}
    outputs = {
        "kpis": [],
        "tables": [],
        "captions": [],
        "infos": [],
        "errors": [],
        "successes": [],
        "selectboxes": [],
        "buttons": [],
        "button_results": button_results or {},
        "reruns": [],
        "headers": [],
    }

    def columns(count):
        return [FakeColumn(outputs, state) for _ in range(count)]

    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "columns", columns)
    monkeypatch.setattr(st, "spinner", lambda _message: nullcontext())
    monkeypatch.setattr(st, "subheader", lambda _message: None)
    monkeypatch.setattr(st, "header", outputs["headers"].append)
    monkeypatch.setattr(st, "caption", outputs["captions"].append)
    monkeypatch.setattr(st, "info", outputs["infos"].append)
    monkeypatch.setattr(st, "error", outputs["errors"].append)
    monkeypatch.setattr(st, "success", outputs["successes"].append)
    monkeypatch.setattr(st, "warning", outputs["infos"].append)
    monkeypatch.setattr(st, "dataframe", lambda frame, **_kwargs: outputs["tables"].append(frame))
    monkeypatch.setattr(st, "text_input", lambda _label, **_kwargs: "")
    monkeypatch.setattr(st, "rerun", lambda: outputs["reruns"].append(True))

    def render_kpis(cards):
        outputs["kpis"].extend(cards)

    def status_indicator(status, **kwargs):
        outputs["successes"].append(f"{kwargs.get('label', '')}: {status}")

    for page in (anomaly_detection, detection_runs, data_quality):
        monkeypatch.setattr(page, "render_kpi_cards", render_kpis)
        monkeypatch.setattr(
            page, "render_data_table", lambda frame, **_kwargs: outputs["tables"].append(frame)
        )
        monkeypatch.setattr(page, "render_status_indicator", status_indicator, raising=False)
    monkeypatch.setattr(common, "render_status_indicator", status_indicator)
    return outputs, state


def _anomalies(*, status="anomaly", has_more=False):
    items = [
        AnomalyResponse(
            run_id=RUN_ID,
            reading_id=17,
            meter_identifier="meter-a",
            timestamp=START,
            prediction_status=status,
            algorithm="global_iqr",
            anomaly_score=2.5 if status == "anomaly" else None,
            anomaly_category="high" if status == "anomaly" else None,
            metadata=None,
        )
    ]
    return AnomaliesPageResponse(
        items=items,
        prediction_status=status,
        limit=100,
        offset=0,
        returned_count=1,
        has_more=has_more,
        next_offset=100 if has_more else None,
    )


def _run(*, status="completed"):
    return DetectionRunResponse(
        run_id=RUN_ID,
        meter_identifier="meter-a",
        algorithm="global_iqr",
        status=status,
        start=START,
        end=END,
        created_at=START,
        started_at=START,
        completed_at=END if status == "completed" else None,
        attempts=1,
        training_readings=10,
        readings_processed=5,
        anomaly_count=1,
        normal_count=3,
        unavailable_count=1,
        summary=None,
        configuration=None,
        error_message=None,
    )


def _runs(run):
    return DetectionRunsPageResponse(
        items=[run],
        limit=100,
        offset=0,
        returned_count=1,
        has_more=False,
        next_offset=None,
    )


def _quality(*, available=False):
    return DataQualityResponse(
        meter_identifier="meter-a",
        start=START,
        end=END,
        reading_count=2,
        missing_measurements={
            "global_active_power": MeasurementStats(missing_count=1, missing_rate=0.5),
            "voltage": MeasurementStats(missing_count=0, missing_rate=0.0),
        },
        timestamp_gaps=TimestampGapSummary(
            expected_interval_minutes=1,
            observed_gap_count=1,
            estimated_missing_intervals=3,
            first_observed_at=START,
            last_observed_at=END,
            boundary_gaps_assessed=False,
        ),
        database_metrics_scope="stored readings in the selected range",
        ingestion_validation={
            "available": available,
            "detail": "Historical report not stored.",
        },
    )


def test_anomaly_page_filters_with_shared_meter_and_date_and_formats_missing(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return _anomalies()

    anomaly_detection.render(FILTERS, "http://api.test", anomalies_loader=load)

    assert calls == [
        (
            ("http://api.test",),
            {
                "meter_identifier": "meter-a",
                "start": START,
                "end": END,
                "prediction_status": "anomaly",
                "algorithm": None,
                "category": None,
                "limit": 100,
                "offset": 0,
            },
        )
    ]
    frame = outputs["tables"][0]
    assert frame.loc[0, "timestamp"] == "2024-01-01 00:00:00 UTC"
    assert frame.loc[0, "anomaly_score"] == "2.5"


def test_anomaly_page_explicitly_labels_unavailable_predictions(monkeypatch):
    outputs, state = _install_mocks(monkeypatch)
    state["anomaly_prediction_status"] = "unavailable"
    anomaly_detection.render(
        FILTERS,
        "http://api.test",
        anomalies_loader=lambda *_args, **_kwargs: _anomalies(status="unavailable"),
    )

    assert any("not classified as normal" in message for message in outputs["infos"])
    assert outputs["tables"][0].loc[0, "anomaly_score"] == "Not provided"
    assert outputs["tables"][0].loc[0, "anomaly_category"] == "Not classified"


def test_anomaly_page_paginates_bounded_results_and_resets_on_filter_change(monkeypatch):
    outputs, state = _install_mocks(monkeypatch, button_results={"Next page": True})
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs["offset"])
        return _anomalies(has_more=True)

    anomaly_detection.render(FILTERS, "http://api.test", anomalies_loader=load)

    assert calls == [0]
    assert state[anomaly_detection.OFFSET_STATE_KEY] == 100
    assert outputs["reruns"] == [True]


def test_anomaly_page_shows_empty_and_api_error_states(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    empty = _anomalies().model_copy(update={"items": [], "returned_count": 0, "has_more": False})
    anomaly_detection.render(
        FILTERS,
        "http://api.test",
        anomalies_loader=lambda *_args, **_kwargs: empty,
    )
    assert any("No anomaly predictions" in message for message in outputs["infos"])

    outputs, _ = _install_mocks(monkeypatch)

    def fail(*_args, **_kwargs):
        raise APIConnectionError("offline")

    anomaly_detection.render(FILTERS, "http://api.test", anomalies_loader=fail)
    assert any("Could not load detection predictions" in message for message in outputs["errors"])


def test_detection_runs_show_historical_readings_run_created_later(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    calls = []
    run = _run().model_copy(
        update={
            "start": datetime(2010, 11, 13, tzinfo=UTC),
            "end": datetime(2010, 11, 20, tzinfo=UTC),
            "created_at": datetime(2026, 11, 1, tzinfo=UTC),
            "started_at": datetime(2026, 11, 1, tzinfo=UTC),
            "completed_at": datetime(2026, 11, 1, 0, 2, tzinfo=UTC),
            "readings_processed": 3_024,
            "anomaly_count": 26,
            "normal_count": 2_998,
            "unavailable_count": 0,
        }
    )

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return _runs(run)

    historical_reading_filter = DashboardFilters(
        "meter-a",
        datetime(2010, 11, 13, tzinfo=UTC).date(),
        datetime(2010, 11, 19, tzinfo=UTC).date(),
    )
    detection_runs.render(historical_reading_filter, "http://api.test", runs_loader=load)

    assert calls[0][1] == {
        "meter_identifier": "meter-a",
        "status": None,
        "algorithm": None,
        "limit": 100,
        "offset": 0,
    }
    table = outputs["tables"][0]
    assert table.loc[0, "created_at"] == "2026-11-01 00:00:00 UTC"
    assert table.loc[0, "readings_processed"] == 3_024
    assert table.loc[0, "anomaly_count"] == 26
    assert table.loc[0, "normal_count"] == 2_998
    assert table.loc[0, "unavailable_count"] == 0
    assert outputs["kpis"][:3] == [
        ("Runs in this page", "1", None),
        ("Completed", "1", None),
        ("Failed", "0", None),
    ]
    assert "Selected run status: completed" in outputs["successes"]
    assert (
        "Unavailable predictions",
        "0",
        "Unavailable is distinct from a normal prediction.",
    ) in outputs["kpis"]


def test_detection_runs_ignore_shared_reading_date_and_still_filter_by_meter(monkeypatch):
    observed_calls = []
    run = _runs(_run())

    def render_for(filters):
        outputs, _ = _install_mocks(monkeypatch)

        def load(*args, **kwargs):
            observed_calls.append(kwargs)
            return run

        detection_runs.render(filters, "http://api.test", runs_loader=load)
        assert outputs["tables"]

    render_for(DashboardFilters("meter-a", START.date(), START.date()))
    render_for(DashboardFilters("meter-a", END.date(), END.date()))
    render_for(DashboardFilters("meter-b", START.date(), START.date()))

    assert [call["meter_identifier"] for call in observed_calls] == [
        "meter-a",
        "meter-a",
        "meter-b",
    ]
    assert all("created_start" not in call and "created_end" not in call for call in observed_calls)


def test_detection_run_loader_omits_optional_creation_date_filters(monkeypatch):
    observed_filters = []

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def list_detection_runs(self, **filters):
            observed_filters.append(filters)
            return _runs(_run())

        def close(self):
            pass

    monkeypatch.setattr(detection_data, "SmartMeterAPIClient", FakeAPIClient)
    detection_data.fetch_detection_runs.clear()

    result = detection_data.fetch_detection_runs(
        "http://api.test",
        meter_identifier="meter-a",
        status=None,
        algorithm=None,
        limit=100,
        offset=0,
    )

    assert result.returned_count == 1
    assert observed_filters == [
        {
            "meter_identifier": "meter-a",
            "created_start": None,
            "created_end": None,
            "status": None,
            "algorithm": None,
            "limit": 100,
            "offset": 0,
        }
    ]


def test_detection_runs_are_available_when_shared_reading_date_is_invalid(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return _runs(_run())

    filters = DashboardFilters(
        "meter-a",
        None,
        None,
        date_range_error="Choose a date range of no more than 31 days.",
    )
    detection_runs.render(filters, "http://api.test", runs_loader=load)

    assert len(calls) == 1
    assert outputs["tables"]
    assert not outputs["infos"]


def test_detection_runs_empty_and_error_states(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    detection_runs.render(
        FILTERS,
        "http://api.test",
        runs_loader=lambda *_args, **_kwargs: _runs(_run()).model_copy(
            update={"items": [], "returned_count": 0}
        ),
    )
    assert any("No detection runs" in message for message in outputs["infos"])

    outputs, _ = _install_mocks(monkeypatch)

    def fail(*_args, **_kwargs):
        raise APIConnectionError("offline")

    detection_runs.render(FILTERS, "http://api.test", runs_loader=fail)
    assert any("Could not load detection runs" in message for message in outputs["errors"])


def test_quality_page_shows_stored_missing_and_unavailable_validation(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return _quality()

    data_quality.render(FILTERS, "http://api.test", quality_loader=load)

    assert calls[0][1] == {
        "meter_identifier": "meter-a",
        "start": START,
        "end": END,
    }
    assert outputs["kpis"][0] == ("Stored readings", "2", None)
    assert outputs["kpis"][2] == (
        "Estimated missing intervals",
        "3",
        "Based on an assumed 1-minute cadence.",
    )
    assert outputs["tables"][0].loc[0, "missing_rate"] == "50.0%"
    assert "Historical validation report: unavailable" in outputs["successes"]
    assert any("Historical report not stored" in message for message in outputs["infos"])


def test_quality_page_empty_missing_statistics_and_errors(monkeypatch):
    outputs, _ = _install_mocks(monkeypatch)
    result = _quality().model_copy(
        update={
            "reading_count": 0,
            "missing_measurements": {},
        }
    )
    data_quality.render(
        FILTERS,
        "http://api.test",
        quality_loader=lambda *_args, **_kwargs: result,
    )
    assert any("no per-measurement" in message for message in outputs["infos"])

    outputs, _ = _install_mocks(monkeypatch)

    def fail(*_args, **_kwargs):
        raise APIConnectionError("offline")

    data_quality.render(FILTERS, "http://api.test", quality_loader=fail)
    assert any("Could not load data-quality" in message for message in outputs["errors"])
