"""Streamlit entry point for the SCADA monitoring portfolio prototype.

Kept thin by design: page config, shared-resource loading, sidebar
navigation, and dispatch to one of three page renderers in
src/dashboard_pages.py. All UI composition lives there and in
src/dashboard_components.py.
"""
from __future__ import annotations

import streamlit as st

from src.dashboard_pages import (
    render_investigation_page,
    render_model_system_page,
    render_operations_page,
    render_sidebar_navigation,
)
from src.dashboard_service import DashboardService

st.set_page_config(
    page_title="Pipeline Failure Detection AI",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def get_service() -> DashboardService:
    """Load telemetry and both model pipelines once per Streamlit process."""
    return DashboardService()


service = get_service()
segments = service.available_segments()

page, theme = render_sidebar_navigation()

if page == "Operations Overview":
    render_operations_page(service, segments, theme)
elif page == "Telemetry & Investigation":
    render_investigation_page(service, segments, theme)
else:
    render_model_system_page(theme)
