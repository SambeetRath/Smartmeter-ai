from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timezone

import pytest
import streamlit as st

from smartmeter_dashboard.api_client import MeterCoverage
from smartmeter_dashboard.components.filters import (
    render_filter_panel,
)
from smartmeter_dashboard.exceptions import APIConnectionError
from smartmeter_dashboard.navigation import PAGE_NAMES, render_navigation
from smartmeter_dashboard.state import (
    DATE_METER_STATE_KEY,
    DATE_RANGE_STATE_KEY,
    METER_STATE_KEY,
    PAGE_STATE_KEY,
    default_date_range,
    validate_date_range,
)

UTC = timezone.utc


class FakeSidebar:
    def __init__(self, state: dict, *, button_result: bool = False, date_value=None):
        self.state = state
        self.button_result = button_result
        self.date_value = date_value
        self.messages: list[str] = []
        self.options: list[str] = []
        self.date_input_kwargs: dict | None = None

    def subheader(self, message):
        self.messages.append(message)

    def radio(self, _label, *, options, key, **_kwargs):
        self.options = list(options)
        self.state.setdefault(key, options[0])
        return self.state[key]

    def selectbox(self, _label, *, options, key, **_kwargs):
        self.options = list(options)
        self.state.setdefault(key, options[0])
        return self.state[key]

    def date_input(self, _label, *, key, **kwargs):
        self.date_input_kwargs = kwargs
        assert key in self.state
        if self.date_value is not None:
            self.state[key] = self.date_value
        return self.state[key]

    def button(self, _label, **_kwargs):
        return self.button_result

    def info(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)

    def success(self, message):
        self.messages.append(message)


@pytest.fixture
def fake_streamlit(monkeypatch):
    state: dict = {}
    sidebar = FakeSidebar(state)
    messages = []
    headers = []
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "sidebar", sidebar)
    monkeypatch.setattr(st, "title", messages.append)
    monkeypatch.setattr(st, "caption", messages.append)
    monkeypatch.setattr(st, "info", messages.append)
    monkeypatch.setattr(st, "error", messages.append)
    monkeypatch.setattr(st, "warning", messages.append)
    monkeypatch.setattr(st, "header", headers.append)
    monkeypatch.setattr(st, "spinner", lambda _message: nullcontext())
    monkeypatch.setattr(st, "set_page_config", lambda **_kwargs: None)
    return state, sidebar, messages, headers


def _meters() -> list[MeterCoverage]:
    return [
        MeterCoverage(
            meter_identifier="meter-a",
            reading_count=20,
            first_reading_at=datetime(2024, 1, 1, tzinfo=UTC),
            last_reading_at=datetime(2024, 1, 20, tzinfo=UTC),
        ),
        MeterCoverage(
            meter_identifier="meter-b",
            reading_count=10,
            first_reading_at=datetime(2024, 2, 1, tzinfo=UTC),
            last_reading_at=datetime(2024, 2, 10, tzinfo=UTC),
        ),
    ]


def test_navigation_offers_all_pages_and_preserves_selection(fake_streamlit):
    state, sidebar, _, _ = fake_streamlit

    assert render_navigation() == "Overview"
    assert tuple(sidebar.options) == PAGE_NAMES

    state[PAGE_STATE_KEY] = "Electrical Metrics"
    assert render_navigation() == "Electrical Metrics"


def test_all_six_sections_keep_the_same_meter_and_date_filter(fake_streamlit, monkeypatch):
    import smartmeter_dashboard.pages as pages

    state, _, _, headers = fake_streamlit
    expected_filters = None
    calls = []
    for page_name in PAGE_NAMES:
        state[PAGE_STATE_KEY] = page_name
        filters = render_filter_panel(_meters(), today=date(2026, 11, 1))
        if expected_filters is None:
            expected_filters = filters

        def capture(received_filters, api_base_url, selected_page=page_name):
            calls.append((selected_page, received_filters, api_base_url))

        module_name = {
            "Overview": "overview",
            "Energy Analytics": "energy_analytics",
            "Electrical Metrics": "electrical_metrics",
            "Anomaly Detection": "anomaly_detection",
            "Detection Runs": "detection_runs",
            "Data Quality": "data_quality",
        }[page_name]
        monkeypatch.setattr(getattr(pages, module_name), "render", capture)
        pages.render_page(page_name, filters, api_base_url="http://api.test")

    assert [name for name, *_ in calls] == list(PAGE_NAMES)
    assert all(received == expected_filters for _, received, _ in calls)
    assert all(api_url == "http://api.test" for _, _, api_url in calls)
    assert state[METER_STATE_KEY] == expected_filters.meter_identifier
    assert state[DATE_RANGE_STATE_KEY] == (
        expected_filters.start_date,
        expected_filters.end_date,
    )


def test_filters_persist_while_navigating_between_sections(fake_streamlit):
    state, _, _, _ = fake_streamlit
    state[PAGE_STATE_KEY] = "Anomaly Detection"
    state[METER_STATE_KEY] = "meter-b"
    state[DATE_METER_STATE_KEY] = "meter-b"
    selected_range = (date(2024, 2, 5), date(2024, 2, 8))
    state[DATE_RANGE_STATE_KEY] = selected_range

    assert render_navigation() == "Anomaly Detection"
    filters = render_filter_panel(_meters())

    assert filters.meter_identifier == "meter-b"
    assert (filters.start_date, filters.end_date) == selected_range
    assert state[PAGE_STATE_KEY] == "Anomaly Detection"


def test_filter_selection_and_defaults_use_selected_meter_coverage(fake_streamlit):
    state, sidebar, _, _ = fake_streamlit
    filters = render_filter_panel(_meters(), today=date(2026, 11, 1))

    assert filters.meter_identifier == "meter-a"
    assert (filters.start_date, filters.end_date) == (date(2024, 1, 14), date(2024, 1, 20))
    assert filters.start == datetime(2024, 1, 14, tzinfo=UTC)
    assert filters.end == datetime(2024, 1, 21, tzinfo=UTC)
    assert state[METER_STATE_KEY] == "meter-a"
    assert state[DATE_METER_STATE_KEY] == "meter-a"
    assert DATE_RANGE_STATE_KEY in state
    assert "meter-a" in sidebar.options
    assert sidebar.date_input_kwargs == {"format": "YYYY-MM-DD"}


def test_date_range_widget_uses_initialized_session_state_without_default_argument(
    fake_streamlit,
):
    state, sidebar, _, _ = fake_streamlit

    filters = render_filter_panel(_meters(), today=date(2026, 11, 1))

    assert filters.start_date == date(2024, 1, 14)
    assert filters.end_date == date(2024, 1, 20)
    assert state[DATE_RANGE_STATE_KEY] == (filters.start_date, filters.end_date)
    assert sidebar.date_input_kwargs is not None
    assert "value" not in sidebar.date_input_kwargs


def test_changing_meter_resets_default_dates_to_new_coverage(fake_streamlit):
    state, sidebar, _, _ = fake_streamlit
    state[METER_STATE_KEY] = "meter-b"
    state[DATE_METER_STATE_KEY] = "meter-a"
    state[DATE_RANGE_STATE_KEY] = (date(2024, 1, 2), date(2024, 1, 3))

    filters = render_filter_panel(_meters(), today=date(2026, 11, 1))

    assert filters.meter_identifier == "meter-b"
    assert (filters.start_date, filters.end_date) == (date(2024, 2, 4), date(2024, 2, 10))
    assert state[DATE_RANGE_STATE_KEY] == (date(2024, 2, 4), date(2024, 2, 10))
    assert "value" not in sidebar.date_input_kwargs


@pytest.mark.parametrize(
    ("start", "end", "expected_error"),
    [
        (date(2024, 1, 2), date(2024, 1, 1), "on or after"),
        (date(2024, 1, 1), date(2024, 2, 1), "no more than 31 days"),
        (date(2024, 1, 1), date(2024, 1, 31), None),
    ],
)
def test_date_range_validation(start, end, expected_error):
    actual_error = validate_date_range(start, end)
    if expected_error is None:
        assert actual_error is None
    else:
        assert expected_error in actual_error


def test_filter_panel_surfaces_invalid_date_range(fake_streamlit):
    _, sidebar, _, _ = fake_streamlit
    sidebar.date_value = (date(2024, 1, 20), date(2024, 1, 1))

    filters = render_filter_panel(_meters())

    assert filters.date_range_error == "The end date must be on or after the start date."
    assert any("end date must be" in message for message in sidebar.messages)


def test_filter_panel_rejects_incomplete_date_range(fake_streamlit):
    _, sidebar, _, _ = fake_streamlit
    sidebar.date_value = (None, None)

    filters = render_filter_panel(_meters())

    assert filters.date_range_error == "Select both a start date and an end date."
    assert "Select both a start date and an end date." in sidebar.messages


def test_empty_meter_list_has_no_selection_and_explains_state(fake_streamlit):
    _, sidebar, _, _ = fake_streamlit

    filters = render_filter_panel([])

    assert filters.meter_identifier is None
    assert any("No meter data" in message for message in sidebar.messages)


def test_app_empty_meter_state_and_navigation(fake_streamlit, monkeypatch):
    import smartmeter_dashboard.app as app

    _, _, messages, headers = fake_streamlit
    monkeypatch.setattr(app, "check_api", lambda _url: "smartmeter-ai")
    monkeypatch.setattr(app, "get_meters", lambda _url: [])

    app.main()

    assert any("No meters are registered" in message for message in messages)
    assert headers == ["Overview"]


def test_app_reports_api_connection_failure(fake_streamlit, monkeypatch):
    import smartmeter_dashboard.app as app

    _, _, messages, _ = fake_streamlit
    meter_calls = []

    def fail_health(_url):
        raise APIConnectionError("Could not connect.")

    monkeypatch.setattr(app, "check_api", fail_health)
    monkeypatch.setattr(app, "get_meters", lambda _url: meter_calls.append(_url))

    app.main()

    assert not meter_calls
    assert any("Could not reach the SmartMeter API" in message for message in messages)


def test_refresh_clears_meter_cache_and_reruns(monkeypatch, fake_streamlit):
    import smartmeter_dashboard.app as app

    state, sidebar, _, _ = fake_streamlit
    sidebar.button_result = True
    sidebar.date_value = None
    state[METER_STATE_KEY] = "meter-a"
    cleared = []
    reruns = []

    def meters(_url):
        return _meters()

    meters.clear = lambda: cleared.append(True)
    monkeypatch.setattr(app, "check_api", lambda _url: "smartmeter-ai")
    monkeypatch.setattr(app, "get_meters", meters)
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True))

    app.main()

    assert cleared == [True]
    assert reruns == [True]


def test_meter_list_uses_cache_until_explicit_refresh(monkeypatch):
    import smartmeter_dashboard.app as app

    calls = []

    class FakeAPIClient:
        def __init__(self, _config):
            pass

        def list_meters(self):
            calls.append("list")
            return _meters()

        def close(self):
            pass

    monkeypatch.setattr(app, "SmartMeterAPIClient", FakeAPIClient)
    app.get_meters.clear()
    try:
        assert app.get_meters("http://api.example.test") == _meters()
        assert app.get_meters("http://api.example.test") == _meters()
        assert calls == ["list"]

        app.get_meters.clear()
        assert app.get_meters("http://api.example.test") == _meters()
        assert calls == ["list", "list"]
    finally:
        app.get_meters.clear()


def test_default_range_falls_back_when_meter_has_no_coverage():
    meter = MeterCoverage(
        meter_identifier="empty",
        reading_count=0,
        first_reading_at=None,
        last_reading_at=None,
    )

    assert default_date_range(meter, today=date(2026, 11, 1)) == (
        date(2026, 10, 26),
        date(2026, 11, 1),
    )


def test_default_range_handles_inconsistent_meter_coverage():
    meter = MeterCoverage(
        meter_identifier="inconsistent",
        reading_count=2,
        first_reading_at=datetime(2024, 1, 3, tzinfo=UTC),
        last_reading_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    assert default_date_range(meter) == (date(2024, 1, 2), date(2024, 1, 2))
