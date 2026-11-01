from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from smartmeter_dashboard import analytics_data
from smartmeter_dashboard.analytics_data import fetch_anomaly_summary, fetch_timeseries
from smartmeter_dashboard.api_client import (
    AnalyticsSummaryResponse,
    AnomalySummaryResponse,
    MeasurementStats,
    TimeSeriesBucket,
    TimeSeriesResponse,
)
from smartmeter_dashboard.components import common
from smartmeter_dashboard.dashboard_pages import (
    electrical_metrics,
    energy_analytics,
    overview,
)
from smartmeter_dashboard.exceptions import APIConnectionError, APIHTTPError
from smartmeter_dashboard.state import DashboardFilters
from smartmeter_dashboard.transformations import electrical_metrics_dataframe

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 3, tzinfo=UTC)
FILTERS = DashboardFilters("meter-a", date(2024, 1, 1), date(2024, 1, 2))


class FakeColumn:
    def __init__(self, metrics):
        self.metrics = metrics

    def metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _install_streamlit_mocks(monkeypatch):
    outputs = {"kpis": [], "charts": [], "captions": [], "infos": [], "errors": []}

    def columns(count):
        return [FakeColumn(outputs["kpis"]) for _ in range(count)]

    def chart(figure, **kwargs):
        outputs["charts"].append((figure, kwargs))

    def kpis(items):
        outputs["kpis"].extend(items)

    monkeypatch.setattr(st, "columns", columns)
    monkeypatch.setattr(st, "spinner", lambda _message: nullcontext())
    monkeypatch.setattr(st, "subheader", lambda _message: None)
    monkeypatch.setattr(st, "plotly_chart", chart)
    monkeypatch.setattr(
        st, "dataframe", lambda frame, **kwargs: outputs.setdefault("frames", []).append(frame)
    )
    monkeypatch.setattr(st, "caption", outputs["captions"].append)
    monkeypatch.setattr(st, "info", outputs["infos"].append)
    monkeypatch.setattr(st, "error", outputs["errors"].append)
    monkeypatch.setattr(st, "warning", lambda _message: None)
    monkeypatch.setattr(st, "selectbox", lambda _label, *, options, **_kwargs: options[0])
    for page in (overview, energy_analytics, electrical_metrics):
        monkeypatch.setattr(page, "render_chart", chart)
        if hasattr(page, "render_kpi_cards"):
            monkeypatch.setattr(page, "render_kpi_cards", kpis)
    monkeypatch.setattr(common, "render_chart", chart)
    monkeypatch.setattr(common, "render_kpi_cards", kpis)
    return outputs


def _summary() -> AnalyticsSummaryResponse:
    return AnalyticsSummaryResponse(
        meter_identifier="meter-a",
        start=START,
        end=END,
        reading_count=2,
        active_power_average=1.5,
        active_power_minimum=1.0,
        active_power_maximum=2.0,
        latest_active_power=2.0,
        latest_active_power_at=END,
        missing_measurements={
            name: MeasurementStats(missing_count=missing, missing_rate=rate)
            for name, missing, rate in (
                ("global_active_power", 0, 0.0),
                ("global_reactive_power", 1, 0.5),
                ("voltage", 0, 0.0),
                ("global_intensity", 0, 0.0),
                ("sub_metering_1", 0, 0.0),
                ("sub_metering_2", 0, 0.0),
                ("sub_metering_3", 0, 0.0),
            )
        },
    )


def _timeseries(interval: str = "hour") -> TimeSeriesResponse:
    buckets = [
        TimeSeriesBucket(
            timestamp=START,
            reading_count=2,
            global_active_power_average=1.25,
            global_active_power_minimum=1.0,
            global_active_power_maximum=1.5,
            global_reactive_power_average=None,
            voltage_average=230.0,
            global_intensity_average=5.0,
            sub_metering_1_average=0.1,
            sub_metering_2_average=0.2,
            sub_metering_3_average=0.3,
        ),
        TimeSeriesBucket(
            timestamp=END,
            reading_count=0,
            global_active_power_average=None,
            global_active_power_minimum=None,
            global_active_power_maximum=None,
            global_reactive_power_average=None,
            voltage_average=None,
            global_intensity_average=None,
            sub_metering_1_average=None,
            sub_metering_2_average=None,
            sub_metering_3_average=None,
        ),
    ]
    return TimeSeriesResponse(
        meter_identifier="meter-a",
        start=START,
        end=END,
        interval=interval,
        buckets=buckets,
        returned_count=len(buckets),
    )


def _anomaly_summary() -> AnomalySummaryResponse:
    return AnomalySummaryResponse(
        run_id="c0bfe534-57fc-4ed8-bd06-cbf482b47861",
        meter_identifier="meter-a",
        algorithm="global_iqr",
        run_status="completed",
        readings_processed=10,
        anomaly_count=2,
        normal_count=7,
        unavailable_count=1,
        anomaly_rate=0.2,
        counts_by_category={"high": 2},
        counts_by_algorithm={"global_iqr": 2},
    )


def test_overview_displays_backend_kpis_missing_stats_and_anomaly_summary(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    calls = {"summary": [], "series": [], "anomaly": []}

    def load_summary(*args):
        calls["summary"].append(args)
        return _summary()

    def load_series(*args):
        calls["series"].append(args)
        return _timeseries("day")

    def load_anomaly(*args):
        calls["anomaly"].append(args)
        return _anomaly_summary()

    overview.render(
        FILTERS,
        "http://api.test",
        summary_loader=load_summary,
        timeseries_loader=load_series,
        anomaly_loader=load_anomaly,
    )

    assert calls["summary"] == [("http://api.test", "meter-a", START, END)]
    assert calls["series"] == [("http://api.test", "meter-a", START, END, "day")]
    assert calls["anomaly"] == [("http://api.test", "meter-a")]
    assert [item[:2] for item in outputs["kpis"]] == [
        ("Total readings", "2"),
        ("Average active power", "1.50 kW"),
        ("Peak active power", "2.00 kW"),
        ("Latest active power", "2.00 kW"),
        ("Anomalies", "2"),
        ("Normal predictions", "7"),
        ("Unavailable predictions", "1"),
        ("Anomaly rate", "20.0%"),
    ]
    assert outputs["kpis"][3][2] == "Observed at 2024-01-03 00:00 UTC"
    assert outputs["frames"][0].loc[1, "missing_rate"] == "50.0%"
    assert len(outputs["charts"]) == 1
    assert outputs["charts"][0][0].data[0].y[0] == 1.25
    assert pd.isna(outputs["charts"][0][0].data[0].y[1])


def test_overview_handles_empty_results_without_fabricated_data(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    summary = _summary().model_copy(
        update={
            "reading_count": 0,
            "active_power_average": None,
            "active_power_minimum": None,
            "active_power_maximum": None,
            "latest_active_power": None,
            "latest_active_power_at": None,
        }
    )
    series = _timeseries("day").model_copy(update={"buckets": [], "returned_count": 0})

    overview.render(
        FILTERS,
        "http://api.test",
        summary_loader=lambda *_args: summary,
        timeseries_loader=lambda *_args: series,
        anomaly_loader=lambda *_args: _anomaly_summary(),
    )

    assert outputs["kpis"][1][1] == "—"
    assert outputs["charts"] == []
    assert any("No readings" in message for message in outputs["infos"])
    assert any("No daily consumption" in message for message in outputs["infos"])


def test_overview_surfaces_api_failure(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)

    def fail(*_args):
        raise APIConnectionError("offline")

    overview.render(FILTERS, "http://api.test", summary_loader=fail)

    assert any("Could not load overview analytics" in message for message in outputs["errors"])


def test_overview_handles_no_completed_anomaly_run(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    overview.render(
        FILTERS,
        "http://api.test",
        summary_loader=lambda *_args: _summary(),
        timeseries_loader=lambda *_args: _timeseries("day"),
        anomaly_loader=lambda *_args: None,
    )

    assert any("No completed anomaly-detection run" in text for text in outputs["infos"])
    assert outputs["errors"] == []


def test_overview_keeps_invalid_meter_summary_404_as_an_error(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)

    def unknown_meter(*_args):
        raise APIHTTPError(404, "/anomalies/summary", "Meter not found.")

    overview.render(
        FILTERS,
        "http://api.test",
        summary_loader=lambda *_args: _summary(),
        timeseries_loader=lambda *_args: _timeseries("day"),
        anomaly_loader=unknown_meter,
    )

    assert any("Could not load anomaly summary" in message for message in outputs["errors"])
    assert outputs["infos"] == []


def test_anomaly_summary_caches_expected_no_run_and_refreshes_to_new_summary(monkeypatch):
    calls = []
    current = {"summary": None}

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def anomaly_summary(self, *, meter_identifier):
            calls.append(meter_identifier)
            if current["summary"] is None:
                raise APIHTTPError(
                    404,
                    "/anomalies/summary",
                    analytics_data.NO_COMPLETED_DETECTION_RUN_DETAIL,
                )
            return current["summary"]

        def close(self):
            pass

    monkeypatch.setattr(analytics_data, "SmartMeterAPIClient", FakeAPIClient)
    analytics_data.clear_analytics_cache()
    try:
        assert fetch_anomaly_summary("http://cache.test", "meter-a") is None

        current["summary"] = _anomaly_summary()
        assert fetch_anomaly_summary("http://cache.test", "meter-a") is None
        assert calls == ["meter-a"]

        analytics_data.clear_analytics_cache()
        result = fetch_anomaly_summary("http://cache.test", "meter-a")
        assert result == _anomaly_summary()
        assert calls == ["meter-a", "meter-a"]
    finally:
        analytics_data.clear_analytics_cache()


def test_anomaly_summary_cache_has_short_ttl_and_does_not_cache_other_failures(
    monkeypatch,
):
    assert fetch_anomaly_summary._info.ttl == 60
    responses = [
        APIConnectionError("offline"),
        APIHTTPError(404, "/anomalies/summary", "Meter not found."),
        APIHTTPError(500, "/anomalies/summary", "Internal Server Error"),
        _anomaly_summary(),
    ]
    calls = []

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def anomaly_summary(self, *, meter_identifier):
            calls.append(meter_identifier)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            pass

    monkeypatch.setattr(analytics_data, "SmartMeterAPIClient", FakeAPIClient)
    analytics_data.clear_analytics_cache()
    try:
        with pytest.raises(APIConnectionError):
            fetch_anomaly_summary("http://failure.test", "meter-a")
        with pytest.raises(APIHTTPError) as error:
            fetch_anomaly_summary("http://failure.test", "meter-a")
        assert error.value.status_code == 404
        assert error.value.detail == "Meter not found."
        with pytest.raises(APIHTTPError) as error:
            fetch_anomaly_summary("http://failure.test", "meter-a")
        assert error.value.status_code == 500
        assert fetch_anomaly_summary("http://failure.test", "meter-a") == _anomaly_summary()
        assert calls == ["meter-a", "meter-a", "meter-a", "meter-a"]
    finally:
        analytics_data.clear_analytics_cache()


def test_energy_page_requests_hourly_and_daily_with_shared_filter_range(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    calls = []

    def load_summary(*args):
        calls.append(("summary", args))
        return _summary()

    def load_series(*args):
        calls.append((args[-1], args))
        return _timeseries(args[-1])

    energy_analytics.render(
        FILTERS,
        "http://api.test",
        summary_loader=load_summary,
        timeseries_loader=load_series,
    )

    assert [call[0] for call in calls] == ["summary", "hour", "day"]
    assert all(call[1][1:4] == ("meter-a", START, END) for call in calls)
    assert len(outputs["charts"]) == 2
    hourly = outputs["charts"][0][0]
    assert hourly.data[0].y[0] == 1.25
    assert pd.isna(hourly.data[0].y[1])
    assert list(hourly.data[1].y) == [2, 0]
    assert "hourly" in hourly.layout.title.text.lower()
    assert "daily" in outputs["charts"][1][0].layout.title.text.lower()


def test_energy_page_handles_empty_buckets_and_api_errors(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    empty = _timeseries().model_copy(update={"buckets": [], "returned_count": 0})
    energy_analytics.render(
        FILTERS,
        "http://api.test",
        summary_loader=lambda *_args: _summary(),
        timeseries_loader=lambda *_args: empty,
    )
    assert outputs["charts"] == []
    assert any("No aggregate buckets" in message for message in outputs["infos"])

    outputs = _install_streamlit_mocks(monkeypatch)

    def fail(*_args):
        raise APIConnectionError("offline")

    energy_analytics.render(FILTERS, "http://api.test", summary_loader=fail)
    assert any("Could not load energy analytics" in message for message in outputs["errors"])


def test_electrical_transformation_preserves_missing_aggregates():
    response = _timeseries()
    frame = electrical_metrics_dataframe(response)

    assert len(frame) == 14
    assert (
        frame.loc[(frame["metric"] == "Reactive power") & (frame["timestamp"] == START), "value"]
        .isna()
        .all()
    )
    assert frame.loc[frame["metric"] == "Voltage", "unit"].unique().tolist() == ["V"]
    assert frame.loc[frame["metric"] == "Sub-metering 1", "unit"].unique().tolist() == [
        "Wh / reading"
    ]


def test_electrical_page_charts_all_metrics_and_keeps_null_gaps(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    calls = []

    def load_series(*args):
        calls.append(args)
        return _timeseries(args[-1])

    electrical_metrics.render(
        FILTERS,
        "http://api.test",
        timeseries_loader=load_series,
    )

    assert calls == [("http://api.test", "meter-a", START, END, "hour")]
    assert len(outputs["charts"]) == 7
    active_power = outputs["charts"][0][0]
    assert isinstance(active_power, go.Figure)
    assert active_power.data[0].y[0] == 1.25
    assert pd.isna(active_power.data[0].y[1])
    assert active_power.layout.yaxis.title.text == "kW"
    assert any("without a value" in caption for caption in outputs["captions"])


def test_electrical_page_handles_empty_data_and_api_error(monkeypatch):
    outputs = _install_streamlit_mocks(monkeypatch)
    empty = _timeseries().model_copy(update={"buckets": [], "returned_count": 0})
    electrical_metrics.render(
        FILTERS,
        "http://api.test",
        timeseries_loader=lambda *_args: empty,
    )
    assert outputs["charts"] == []
    assert any("No electrical metric buckets" in message for message in outputs["infos"])

    outputs = _install_streamlit_mocks(monkeypatch)

    def fail(*_args):
        raise APIConnectionError("offline")

    electrical_metrics.render(FILTERS, "http://api.test", timeseries_loader=fail)
    assert any("Could not load electrical metrics" in message for message in outputs["errors"])


def test_successful_timeseries_responses_are_cached_and_cache_can_be_cleared(monkeypatch):
    import smartmeter_dashboard.analytics_data as data

    calls = []

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def analytics_timeseries(self, *args, **kwargs):
            calls.append((args, kwargs))
            return _timeseries(kwargs["interval"])

        def close(self):
            pass

    monkeypatch.setattr(data, "SmartMeterAPIClient", FakeAPIClient)
    fetch_timeseries.clear()
    try:
        first = fetch_timeseries("http://cache.test", "meter-a", START, END, "hour")
        second = fetch_timeseries("http://cache.test", "meter-a", START, END, "hour")
        assert first == second
        assert len(calls) == 1

        data.clear_analytics_cache()
        fetch_timeseries("http://cache.test", "meter-a", START, END, "hour")
        assert len(calls) == 2
    finally:
        data.clear_analytics_cache()
