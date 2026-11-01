from __future__ import annotations

import streamlit as st

from smartmeter_dashboard.analytics_data import clear_analytics_cache
from smartmeter_dashboard.api_client import MeterCoverage, SmartMeterAPIClient
from smartmeter_dashboard.components.common import (
    render_api_status,
    render_empty_state,
    render_error_state,
    render_header,
    render_loading,
    render_visual_styles,
)
from smartmeter_dashboard.components.filters import render_filter_panel
from smartmeter_dashboard.config import ConfigurationError, DashboardConfig
from smartmeter_dashboard.exceptions import DashboardAPIError
from smartmeter_dashboard.navigation import render_navigation
from smartmeter_dashboard.pages import render_page


@st.cache_data(ttl=300, show_spinner=False)
def get_meters(api_base_url: str) -> list[MeterCoverage]:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.list_meters()
    finally:
        client.close()


def check_api(api_base_url: str) -> str:
    client = SmartMeterAPIClient(DashboardConfig(api_base_url=api_base_url))
    try:
        return client.health().service
    finally:
        client.close()


def main() -> None:
    st.set_page_config(page_title="SmartMeter AI", page_icon="⚡", layout="wide")
    render_visual_styles()
    try:
        config = DashboardConfig.from_env()
    except ConfigurationError as error:
        render_error_state(f"Dashboard configuration error: {error}")
        return

    api_error = None
    service_name = None
    meters = []
    with render_loading("Connecting to the API and loading meter coverage..."):
        try:
            service_name = check_api(config.api_base_url)
        except DashboardAPIError as error:
            api_error = f"Could not reach the SmartMeter API: {error}"

        if service_name:
            try:
                meters = get_meters(config.api_base_url)
            except DashboardAPIError as error:
                api_error = f"Could not load meters from the SmartMeter API: {error}"

    page_name = render_navigation()
    filters = render_filter_panel(meters)
    if filters.refresh_requested:
        get_meters.clear()
        clear_analytics_cache()
        st.rerun()

    render_header(filters)
    render_api_status(
        available=service_name is not None and api_error is None,
        message=(
            f"API available: {service_name}"
            if service_name and api_error is None
            else api_error or "SmartMeter API is unavailable."
        ),
    )
    if api_error:
        render_error_state(api_error)
    elif not meters:
        render_empty_state("No meters are registered in the backend yet.")
    render_page(page_name, filters, api_base_url=config.api_base_url)


if __name__ == "__main__":
    main()
