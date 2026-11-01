from __future__ import annotations

import streamlit as st

from smartmeter_dashboard.pages import PAGE_PLACEHOLDER_MESSAGES
from smartmeter_dashboard.state import PAGE_STATE_KEY

PAGE_NAMES = tuple(PAGE_PLACEHOLDER_MESSAGES)


def render_navigation() -> str:
    st.sidebar.subheader("Navigation")
    return st.sidebar.radio(
        "Dashboard section",
        options=PAGE_NAMES,
        key=PAGE_STATE_KEY,
        label_visibility="collapsed",
    )
