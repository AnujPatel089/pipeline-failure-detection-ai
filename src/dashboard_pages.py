"""Page-level composition for the three-workspace Streamlit dashboard.

Each render_*_page function only arranges existing components; none of it
touches inference, thresholds, or leakage protection.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import TIMESTAMP_COLUMN
from src.dashboard_components import (
    render_alert_timeline,
    render_architecture_panel,
    render_control_bar,
    render_evaluation_panel,
    render_feature_importance_panel,
    render_fault_intelligence,
    render_kpi_row,
    render_leakage_protection_panel,
    render_limitations_panel,
    render_mini_trend,
    render_model_summary_cards,
    render_observation_details,
    render_operations_header,
    render_page_title,
    render_pipeline_schematic,
    render_risk_banner,
    render_risk_gauge,
    render_service_status_panel,
    render_telemetry_cards,
    render_trend_tabs,
    render_validation_strategy_panel,
)
from src.dashboard_state import (
    DEFAULT_PAGE,
    EVALUATION_KEY,
    PAGE_KEY,
    PAGE_NAMES,
    THEME_KEY,
    recall,
    remember,
    resolve_current_prediction,
)
from src.dashboard_styles import DEFAULT_THEME, THEME_NAMES, get_theme, inject_global_styles


def render_sidebar_navigation() -> tuple[str, dict[str, str]]:
    """Render the navigation-first sidebar and return (active page, active theme)."""
    with st.sidebar:
        st.markdown(
            '<p class="scada-nav-title">Pipeline AI</p>'
            '<p class="scada-nav-subtitle">SCADA Intelligence Platform</p>',
            unsafe_allow_html=True,
        )

        page_kwargs: dict[str, Any] = {"key": PAGE_KEY}
        if PAGE_KEY not in st.session_state:
            page_kwargs["index"] = PAGE_NAMES.index(DEFAULT_PAGE)
        page = st.radio("Navigation", PAGE_NAMES, label_visibility="collapsed", **page_kwargs)

        st.markdown('<div class="scada-eyebrow-sub">Appearance</div>', unsafe_allow_html=True)
        theme_kwargs: dict[str, Any] = {"key": THEME_KEY}
        if THEME_KEY not in st.session_state:
            theme_kwargs["index"] = THEME_NAMES.index(DEFAULT_THEME)
        selected_theme_name = st.selectbox("Theme", THEME_NAMES, label_visibility="collapsed", **theme_kwargs)
        theme = get_theme(selected_theme_name)
        inject_global_styles(theme)

        st.markdown(
            '<div class="scada-sidebar-status"><span class="dot"></span>Model Service Ready</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="scada-badge">Portfolio Prototype</span>',
            unsafe_allow_html=True,
        )

    return page, theme


def render_operations_page(service: Any, segments: list[Any], theme: dict[str, str]) -> None:
    """Page 1: a 10-20 second read on current status. Preview, not investigation."""
    control = render_control_bar(service, segments)
    current, previous, prediction = resolve_current_prediction(
        service, control["records"], control["current_index"], control["mode"], control["threshold"]
    )

    render_operations_header()
    render_kpi_row(prediction)
    render_risk_banner(prediction, control["threshold"], control["mode"])

    left_col, right_col = st.columns(2)
    with left_col:
        render_pipeline_schematic(service, segments, control["segment_id"], control["threshold"], control["mode"])
    with right_col:
        render_telemetry_cards(current, previous)

    st.divider()
    mini_history = control["records"].iloc[: control["current_index"] + 1].set_index(TIMESTAMP_COLUMN)
    render_mini_trend(mini_history, theme)

    if st.button("Open Telemetry & Investigation →", key="goto_investigation"):
        st.session_state[PAGE_KEY] = "Telemetry & Investigation"
        st.rerun()


def render_investigation_page(service: Any, segments: list[Any], theme: dict[str, str]) -> None:
    """Page 2: deep interactive analysis — replay, full trends, gauge, history, evaluation."""
    control = render_control_bar(service, segments)
    current, previous, prediction = resolve_current_prediction(
        service, control["records"], control["current_index"], control["mode"], control["threshold"]
    )

    render_page_title("Telemetry & Investigation", "Detailed sensor trends, fault analysis, replay, and evaluation")

    history = control["records"].iloc[: control["current_index"] + 1].set_index(TIMESTAMP_COLUMN)
    render_trend_tabs(history, theme)

    st.divider()
    gauge_col, fault_col = st.columns(2)
    with gauge_col:
        render_risk_gauge(prediction["failure_probability"], control["threshold"], theme)
        st.caption(f"Monitoring mode: {control['mode']} · Active threshold: {control['threshold']:.0%}")
    with fault_col:
        render_fault_intelligence(prediction)

    st.divider()
    render_alert_timeline(theme)

    st.divider()
    evaluation_kwargs: dict[str, Any] = {"key": EVALUATION_KEY}
    if EVALUATION_KEY not in st.session_state:
        evaluation_kwargs["value"] = bool(recall(EVALUATION_KEY, False))
    evaluation_mode = st.checkbox("Evaluation Mode — Show Ground Truth", **evaluation_kwargs)
    remember(EVALUATION_KEY, evaluation_mode)
    if evaluation_mode:
        render_evaluation_panel(prediction)

    render_observation_details(current, prediction, control["current_index"], control["record_count"])


def render_model_system_page(theme: dict[str, str]) -> None:
    """Page 3: technical transparency — metrics, importance, validation, architecture, limits."""
    render_page_title("Model & System", "Architecture, validation, explainability, and platform details")
    render_model_summary_cards()
    st.divider()
    render_feature_importance_panel(theme)
    st.divider()
    render_validation_strategy_panel()
    st.divider()
    render_leakage_protection_panel()
    st.divider()
    render_architecture_panel()
    st.divider()
    render_service_status_panel()
    st.divider()
    render_limitations_panel()
