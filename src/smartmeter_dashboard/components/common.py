from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from html import escape

import pandas as pd
import streamlit as st

from smartmeter_dashboard.state import DashboardFilters


def render_visual_styles() -> None:
    st.markdown(
        """
        <style>
        .sm-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
            gap: 0.75rem;
            margin: 0.35rem 0 1rem;
        }
        .sm-kpi-card {
            min-width: 0;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 0.65rem;
            background: rgba(128, 128, 128, 0.08);
        }
        .sm-kpi-label {
            color: inherit;
            opacity: 0.78;
            font-size: 0.86rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .sm-kpi-value {
            color: inherit;
            font-size: clamp(1.05rem, 2vw, 1.5rem);
            font-weight: 650;
            line-height: 1.3;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .sm-kpi-help {
            color: inherit;
            opacity: 0.68;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(
    items: Sequence[tuple[str, str, str | None]],
) -> None:
    cards = []
    for label, value, help_text in items:
        help_html = f'<div class="sm-kpi-help">{escape(help_text)}</div>' if help_text else ""
        cards.append(
            '<div class="sm-kpi-card">'
            f'<div class="sm-kpi-label">{escape(label)}</div>'
            f'<div class="sm-kpi-value">{escape(value)}</div>'
            f"{help_html}</div>"
        )
    st.markdown(
        f'<div class="sm-kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_data_table(frame: pd.DataFrame, *, height: int | str = "auto") -> None:
    column_widths = {
        column: st.column_config.Column(
            width=(
                "large"
                if any(token in str(column).lower() for token in ("time", "timestamp", "_id"))
                else "medium"
            )
        )
        for column in frame
    }
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=height,
        column_config=column_widths,
    )


def render_status_indicator(status: str, *, label: str | None = None) -> None:
    safe_status = escape(status.replace("_", " ").title())
    prefix = f"{escape(label)}: " if label else ""
    if status in {"completed", "normal", "available"}:
        st.success(f"{prefix}{safe_status}")
    elif status in {"failed", "unavailable", "error"}:
        st.error(f"{prefix}{safe_status}")
    else:
        st.info(f"{prefix}{safe_status}")


def render_chart(figure, *, height: int = 430) -> None:
    figure.update_layout(height=height)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"responsive": True, "displaylogo": False},
        theme=None,
    )


def render_header(filters: DashboardFilters) -> None:
    st.title("SmartMeter AI")
    st.caption("Energy analytics and anomaly detection")
    if filters.meter_identifier and filters.start_date and filters.end_date:
        st.caption(
            f"Meter: {filters.meter_identifier} · "
            f"Date range (UTC): {filters.start_date:%Y-%m-%d} "
            f"to {filters.end_date:%Y-%m-%d}"
        )


def render_api_status(*, available: bool, message: str) -> None:
    if available:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)


@contextmanager
def render_loading(message: str = "Loading data…") -> Iterator[None]:
    with st.spinner(message):
        yield


def render_empty_state(message: str) -> None:
    st.info(message)


def render_error_state(message: str) -> None:
    st.error(message)
