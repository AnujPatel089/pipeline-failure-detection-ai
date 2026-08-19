"""Presentation-only UI building blocks for the Streamlit control-room dashboard.

Every function here only formats and displays data already produced by
DashboardService / src.monitoring. None of it participates in inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import FAULT_IMPORTANCE_PATH, FAULT_MODEL_PATH, FEATURE_IMPORTANCE_PATH, MODEL_FEATURES, MODEL_PATH
from src.dashboard_state import (
    HISTORY_KEY,
    HISTORY_KEYS_KEY,
    MODE_KEY,
    POSITION_KEY,
    REPLAY_SPEED_KEY,
    SEGMENT_KEY,
    format_observation,
    go_next,
    go_previous,
    go_skip,
    recall,
    remember,
    reset_record_position,
)
from src.monitoring import (
    CRITICAL_THRESHOLD,
    HIGH_SENSITIVITY_THRESHOLD,
    STANDARD_THRESHOLD,
    get_monitoring_threshold,
)

STATUS_CLASS = {"NORMAL": "normal", "WARNING": "warning", "CRITICAL": "critical"}


def apply_plotly_theme(
    figure: go.Figure, theme: dict[str, str], height: int | None = None, show_legend: bool = False
) -> go.Figure:
    """Style a Plotly figure with the active dashboard theme's palette.

    Plotly figures render outside the page's CSS cascade, so they need the
    active theme's colors passed in explicitly rather than picked up from CSS.
    """
    layout: dict[str, Any] = dict(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=theme["card"],
        plot_bgcolor=theme["card"],
        font=dict(color=theme["text_secondary"], size=12),
        title_font=dict(color=theme["text_primary"]),
        legend=dict(font=dict(color=theme["text_secondary"])),
        hoverlabel=dict(
            bgcolor=theme["card_elevated"], font_color=theme["text_primary"], bordercolor=theme["border"]
        ),
        showlegend=show_legend,
    )
    if height is not None:
        layout["height"] = height
    figure.update_layout(**layout)
    figure.update_xaxes(gridcolor=theme["grid"], zeroline=False, color=theme["text_secondary"])
    figure.update_yaxes(gridcolor=theme["grid"], zeroline=False, color=theme["text_secondary"])
    figure.update_annotations(font=dict(color=theme["text_primary"], size=12))
    return figure

FAULT_EXPLANATIONS = {
    "leak": "Pressure/flow behavior is consistent with a potential leak pattern.",
    "blockage": "Telemetry suggests restricted flow or blockage-like behavior.",
    "surge": "Telemetry indicates a possible surge condition.",
    "degradation": "Telemetry is consistent with gradual process/equipment degradation.",
}

TELEMETRY_NUMERIC_FIELDS = [
    ("Pressure", "pressure"),
    ("Flow Rate", "flow_rate"),
    ("Temperature", "temperature"),
    ("Pump Speed", "pump_speed"),
    ("Energy Consumption", "energy_consumption"),
]
TELEMETRY_STATE_FIELDS = [
    ("Valve Status", "valve_status"),
    ("Pump State", "pump_state"),
    ("Compressor State", "compressor_state"),
]


def render_operations_header() -> None:
    """Page 1's compact header: title, subtitle, and a short system-state strip."""
    st.markdown(
        '<p class="scada-title">PIPELINE FAILURE DETECTION <span class="accent">AI</span></p>'
        '<p class="scada-subtitle">SCADA Monitoring &bull; Failure Detection &bull; Fault Intelligence</p>',
        unsafe_allow_html=True,
    )
    items = [
        ("System Status", "Online", True),
        ("Model Service", "Ready", True),
        ("Build", "Portfolio Prototype", False),
    ]
    cells = "".join(
        f'<div class="scada-status-item"><div class="k">{label}</div>'
        f'<div class="v{" on" if on else ""}">{value}</div></div>'
        for label, value, on in items
    )
    st.markdown(f'<div class="scada-status-strip">{cells}</div>', unsafe_allow_html=True)


def render_page_title(title: str, subtitle: str) -> None:
    """Reusable header for Telemetry & Investigation and Model & System."""
    st.markdown(
        f'<p class="scada-title" style="font-size:1.7rem;">{title}</p>'
        f'<p class="scada-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_kpi_row(prediction: dict[str, Any]) -> None:
    status = prediction["pipeline_status"]
    css_status = STATUS_CLASS[status]
    fault_type = prediction.get("fault_type")
    fault_confidence = prediction.get("fault_confidence")

    cards = [
        ("Pipeline Status", status, "Current alert level", css_status),
        ("Failure Risk", f"{prediction['failure_probability']:.1%}", "Model 1 probability", css_status),
        ("Fault Type", fault_type.title() if fault_type else "N/A", "Model 2 classification", "info"),
        (
            "Fault Confidence",
            f"{fault_confidence:.1%}" if fault_confidence is not None else "N/A",
            "Classifier confidence",
            "info",
        ),
        ("Segment", f"Segment {prediction['segment_id']}", "Active pipeline segment", "info"),
    ]
    cells = "".join(
        f'<div class="scada-card scada-card-{cls}"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>'
        for label, value, sub, cls in cards
    )
    st.markdown(f'<div class="scada-card-grid">{cells}</div>', unsafe_allow_html=True)


def render_risk_banner(prediction: dict[str, Any], threshold: float, mode: str) -> None:
    status = prediction["pipeline_status"]
    css_status = STATUS_CLASS[status]
    fault_type = prediction.get("fault_type")
    fault_confidence = prediction.get("fault_confidence")

    if status == "NORMAL":
        title = "Normal Operating Condition"
    elif status == "CRITICAL" and fault_type:
        title = f"Critical — Potential {fault_type.title()} Condition"
    elif status == "CRITICAL":
        title = "Critical — Abnormal Condition"
    elif fault_type:
        title = f"Warning — Possible {fault_type.title()} Pattern"
    else:
        title = "Warning — Abnormal Condition"

    confidence_text = f"{fault_confidence:.1%}" if fault_confidence is not None else "N/A"
    st.markdown(
        f"""
        <div class="scada-banner scada-banner-{css_status}">
            <div class="eyebrow-label">AI Risk Assessment</div>
            <div class="title">{title}</div>
            <div class="msg">{prediction['operational_message']}</div>
            <div class="stats">
                <span>Failure probability &nbsp;<b>{prediction['failure_probability']:.1%}</b></span>
                <span>Fault confidence &nbsp;<b>{confidence_text}</b></span>
                <span>Active threshold &nbsp;<b>{threshold:.0%}</b></span>
                <span>Monitoring mode &nbsp;<b>{mode}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_schematic(service: Any, segments: list[Any], segment_id: Any, threshold: float, mode: str) -> None:
    st.markdown('<div class="scada-eyebrow">Pipeline Schematic</div>', unsafe_allow_html=True)
    center = segments.index(segment_id)
    window = segments[max(0, center - 2) : center + 3]

    cells = []
    for i, neighbor_id in enumerate(window):
        neighbor_records = service.segment_records(neighbor_id)
        neighbor_prediction = service.predict_record(neighbor_records.iloc[-1], threshold, mode)
        css_status = STATUS_CLASS[neighbor_prediction["pipeline_status"]]
        selected = neighbor_id == segment_id
        box_class = f"box {css_status}{' selected' if selected else ''}"
        tag_class = "tag show" if selected else "tag"
        cells.append(
            f'<div class="scada-seg"><div class="{box_class}">Seg {neighbor_id}</div>'
            f'<div class="{tag_class}">&#9650; SELECTED</div></div>'
        )
        if i < len(window) - 1:
            cells.append('<div class="scada-connector">&#9472;&#9472;</div>')

    st.markdown(
        f'<div class="scada-panel-card"><div class="scada-pipeline">{"".join(cells)}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Status reflects each segment's latest available observation. Not a geographic layout.")


def _format_delta(current: float, previous: float | None) -> str:
    if previous is None or previous == 0:
        return "N/A"
    change = (current - previous) / previous * 100
    arrow = "▲" if change > 0 else ("▼" if change < 0 else "▶")
    return f"{arrow} {change:+.1f}%"


def render_telemetry_cards(current_row: pd.Series, previous_row: pd.Series | None) -> None:
    """Live SCADA Telemetry cards, laid out 3-per-row so they fit a half-width column."""
    st.markdown('<div class="scada-eyebrow">Live SCADA Telemetry</div>', unsafe_allow_html=True)
    fields = TELEMETRY_NUMERIC_FIELDS
    for start in range(0, len(fields), 3):
        row_columns = st.columns(3)
        for column, (label, feature) in zip(row_columns, fields[start : start + 3]):
            previous_value = float(previous_row[feature]) if previous_row is not None else None
            column.metric(
                label, f"{current_row[feature]:,.2f}", _format_delta(float(current_row[feature]), previous_value)
            )

    state_columns = st.columns(3)
    for column, (label, feature) in zip(state_columns, TELEMETRY_STATE_FIELDS):
        column.metric(label, str(current_row[feature]))
    st.caption("Values are shown without units because units are not defined in the supplied dataset.")


def render_mini_trend(history: pd.DataFrame, theme: dict[str, str]) -> None:
    """A single compact preview chart for the landing page — not the full investigation view.

    Pressure and flow rate sit on very different scales (roughly 43-115 vs
    1-7.5), so they share a dual-axis chart rather than one misleading axis.
    """
    st.markdown('<div class="scada-eyebrow">Trend Preview</div>', unsafe_allow_html=True)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(x=history.index, y=history["pressure"], name="Pressure", line=dict(color=theme["accent"], width=2)),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(x=history.index, y=history["flow_rate"], name="Flow Rate", line=dict(color=theme["healthy"], width=2)),
        secondary_y=True,
    )
    apply_plotly_theme(figure, theme, height=220, show_legend=True)
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    figure.update_yaxes(title_text="Pressure", secondary_y=False)
    figure.update_yaxes(title_text="Flow Rate", secondary_y=True)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption("Open Telemetry & Investigation for detailed trends.")


def _line_trace(
    figure: go.Figure, x: pd.Index, y: pd.Series, name: str, color: str, row: int, col: int, theme: dict[str, str]
) -> None:
    figure.add_trace(
        go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=2), marker=dict(size=4)),
        row=row, col=col,
    )
    figure.add_trace(
        go.Scatter(
            x=[x[-1]], y=[y.iloc[-1]], mode="markers", showlegend=False,
            marker=dict(size=12, color=theme["accent"], line=dict(width=2, color=theme["card"])),
        ),
        row=row, col=col,
    )


def render_trend_tabs(history: pd.DataFrame, theme: dict[str, str]) -> None:
    st.markdown('<div class="scada-eyebrow">Telemetry Trends</div>', unsafe_allow_html=True)
    x = history.index

    overview_tab, pressure_flow_tab, equipment_tab, energy_tab = st.tabs(
        ["Overview", "Pressure & Flow", "Equipment", "Energy"]
    )

    with overview_tab:
        figure = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Pressure", "Flow Rate", "Temperature"))
        _line_trace(figure, x, history["pressure"], "Pressure", theme["accent"], 1, 1, theme)
        _line_trace(figure, x, history["flow_rate"], "Flow Rate", theme["healthy"], 2, 1, theme)
        _line_trace(figure, x, history["temperature"], "Temperature", theme["warning"], 3, 1, theme)
        st.plotly_chart(apply_plotly_theme(figure, theme, 560), width="stretch", config={"displayModeBar": False})

    with pressure_flow_tab:
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Pressure", "Flow Rate"))
        _line_trace(figure, x, history["pressure"], "Pressure", theme["accent"], 1, 1, theme)
        _line_trace(figure, x, history["flow_rate"], "Flow Rate", theme["healthy"], 1, 2, theme)
        st.plotly_chart(apply_plotly_theme(figure, theme, 340), width="stretch", config={"displayModeBar": False})

    with equipment_tab:
        figure = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Pump Speed", "Valve Status", "Compressor State"),
        )
        _line_trace(figure, x, history["pump_speed"], "Pump Speed", theme["accent"], 1, 1, theme)
        figure.add_trace(
            go.Scatter(x=x, y=history["valve_status"], mode="lines+markers", line_shape="hv", line=dict(color=theme["warning"])),
            row=2, col=1,
        )
        figure.add_trace(
            go.Scatter(x=x, y=history["compressor_state"], mode="lines+markers", line_shape="hv", line=dict(color=theme["healthy"])),
            row=3, col=1,
        )
        st.plotly_chart(apply_plotly_theme(figure, theme, 560), width="stretch", config={"displayModeBar": False})

    with energy_tab:
        figure = make_subplots(rows=1, cols=1, subplot_titles=("Energy Consumption",))
        _line_trace(figure, x, history["energy_consumption"], "Energy Consumption", theme["accent"], 1, 1, theme)
        st.plotly_chart(apply_plotly_theme(figure, theme, 340), width="stretch", config={"displayModeBar": False})


def render_risk_gauge(probability: float, threshold: float, theme: dict[str, str]) -> None:
    st.markdown('<div class="scada-eyebrow">Failure Risk Gauge</div>', unsafe_allow_html=True)
    threshold_pct = threshold * 100
    critical_pct = CRITICAL_THRESHOLD * 100
    if probability >= CRITICAL_THRESHOLD:
        bar_color = theme["critical"]
    elif probability >= threshold:
        bar_color = theme["warning"]
    else:
        bar_color = theme["healthy"]

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": theme["text_primary"], "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme["text_secondary"], "tickfont": {"color": theme["text_secondary"]}},
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": theme["card"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold_pct], "color": theme["healthy_soft"]},
                    {"range": [threshold_pct, critical_pct], "color": theme["warning_soft"]},
                    {"range": [critical_pct, 100], "color": theme["critical_soft"]},
                ],
                "threshold": {"line": {"color": theme["text_primary"], "width": 3}, "thickness": 0.85, "value": threshold_pct},
            },
        )
    )
    figure.update_layout(
        height=230, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=theme["card"], font=dict(color=theme["text_secondary"]),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(f"Normal below {threshold:.0%} · Warning {threshold:.0%}–80% · Critical above 80%")


def render_fault_intelligence(prediction: dict[str, Any]) -> None:
    st.markdown('<div class="scada-eyebrow">Fault Intelligence</div>', unsafe_allow_html=True)
    fault_type = prediction.get("fault_type")
    if not fault_type:
        st.caption("No fault classification required.")
        return
    confidence = prediction.get("fault_confidence")
    explanation = FAULT_EXPLANATIONS.get(fault_type.lower(), "Telemetry pattern is consistent with an abnormal condition.")
    st.markdown(
        f"""
        <div class="scada-fault-card">
            <div class="name">{fault_type.upper()}</div>
            <div class="conf">Confidence: {f"{confidence:.1%}" if confidence is not None else "N/A"}</div>
            <div class="desc">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_timeline(theme: dict[str, str]) -> None:
    st.markdown('<div class="scada-eyebrow">Alert &amp; Event Timeline</div>', unsafe_allow_html=True)
    header_col, filter_col, clear_col = st.columns([3, 2, 1])
    with filter_col:
        alert_filter = st.selectbox("Filter", ["All", "Warnings", "Critical"], label_visibility="collapsed")
    with clear_col:
        if st.button("Clear session history", width="stretch"):
            st.session_state[HISTORY_KEY] = []
            st.session_state[HISTORY_KEYS_KEY] = set()
            st.rerun()

    records = list(reversed(st.session_state.get(HISTORY_KEY, [])))
    if alert_filter == "Warnings":
        records = [row for row in records if row["alert_level"] == "WARNING"]
    elif alert_filter == "Critical":
        records = [row for row in records if row["alert_level"] == "CRITICAL"]

    if not records:
        st.caption("No predictions recorded in this session.")
        return

    frame = pd.DataFrame(records).rename(
        columns={
            "timestamp": "Time",
            "segment_id": "Segment",
            "status": "Status",
            "failure_probability": "Failure Risk",
            "fault_type": "Fault",
            "fault_confidence": "Confidence",
            "alert_level": "Alert Level",
        }
    )
    frame["Time"] = frame["Time"].map(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S"))
    frame["Failure Risk"] = frame["Failure Risk"].map(lambda value: f"{value:.1%}")
    frame["Fault"] = frame["Fault"].fillna("—").map(lambda value: str(value).title() if value != "—" else value)
    frame["Confidence"] = frame["Confidence"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")

    def _row_style(row: pd.Series) -> list[str]:
        color = {
            "INFO": "",
            "WARNING": f"background-color: {theme['warning_soft']};",
            "CRITICAL": f"background-color: {theme['critical_soft']};",
        }
        return [color.get(row["Alert Level"], "")] * len(row)

    styled = frame.style.apply(_row_style, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)


def render_evaluation_panel(prediction: dict[str, Any]) -> None:
    with st.expander("Ground Truth — Evaluation Only", expanded=True):
        truth = prediction["evaluation_only"]
        predicted_status = prediction["status"]
        actual_status = "abnormal" if truth["actual_target"] == 1 else "normal"
        binary_correct = predicted_status == actual_status

        cols = st.columns(3)
        cols[0].metric("Actual target", truth["actual_target"])
        cols[1].metric("Actual event type", truth["actual_event_type"].title())
        cols[2].metric("Original alarm state", truth["original_alarm_state"])

        st.markdown(
            f"**Binary prediction:** {'✅ Correct' if binary_correct else '❌ Incorrect'} "
            f"(predicted {predicted_status}, actual {actual_status})"
        )
        predicted_fault = prediction.get("fault_type")
        if predicted_fault is not None and truth["actual_event_type"] != "normal":
            fault_correct = predicted_fault == truth["actual_event_type"]
            if fault_correct:
                st.markdown("**Fault prediction:** ✅ Correct")
            else:
                st.markdown(
                    f"**Fault prediction:** ❌ Incorrect — Expected: {truth['actual_event_type'].title()}, "
                    f"Predicted: {predicted_fault.title()}"
                )
        st.caption("Ground truth is attached after prediction and never passed to either model.")


def _summary_card(heading: str, rows: list[tuple[str, str]]) -> str:
    row_html = "".join(f'<div class="row"><span class="rl">{label}</span><span class="rv">{value}</span></div>' for label, value in rows)
    return f'<div class="scada-panel-card scada-summary-card"><div class="heading">{heading}</div>{row_html}</div>'


def render_model_summary_cards() -> None:
    st.markdown('<div class="scada-eyebrow">Model Summary</div>', unsafe_allow_html=True)
    binary_col, fault_col = st.columns(2)
    with binary_col:
        st.markdown(
            _summary_card(
                "Binary Failure Detector",
                [
                    ("Algorithm", "Random Forest"),
                    ("Task", "Normal vs Abnormal"),
                    ("Failure Precision", "88.5%"),
                    ("Failure Recall", "93.1%"),
                    ("F1", "90.8%"),
                    ("PR-AUC", "97.2%"),
                    ("Held-out", "54 / 58 failures detected"),
                    ("False negatives", "4"),
                    ("False positives", "7"),
                ],
            ),
            unsafe_allow_html=True,
        )
        st.caption("Repeated group-aware CV: recall 88.1% ± 4.1%, F1 89.6% ± 3.2%.")
    with fault_col:
        st.markdown(
            _summary_card(
                "Fault Classifier",
                [
                    ("Algorithm", "Random Forest"),
                    ("Classes", "Blockage, Degradation, Leak, Surge"),
                    ("Macro F1", "97.3%"),
                    ("Balanced Accuracy", "96.9%"),
                    ("Group CV Macro F1", "97.2% ± 2.0%"),
                    ("Weakest class", "Blockage recall 87.5%"),
                ],
            ),
            unsafe_allow_html=True,
        )
        st.caption("Only held-out error: one blockage observation classified as leak.")
    st.caption("Metrics are from a small simulated-style dataset and are not production-performance claims.")


@st.cache_data
def _load_importance_table(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame.sort_values("permutation_importance_mean", ascending=False).head(5)


def _importance_chart(frame: pd.DataFrame, title: str, theme: dict[str, str]) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=frame["permutation_importance_mean"],
            y=frame["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker=dict(color=theme["accent"]),
        )
    )
    apply_plotly_theme(figure, theme, height=260)
    figure.update_layout(
        title=dict(text=title, font=dict(color=theme["text_primary"], size=13)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(title="Permutation importance")
    return figure


def render_feature_importance_panel(theme: dict[str, str]) -> None:
    st.markdown('<div class="scada-eyebrow">Feature Importance</div>', unsafe_allow_html=True)
    binary_col, fault_col = st.columns(2)
    binary_importance = _load_importance_table(str(FEATURE_IMPORTANCE_PATH))
    fault_importance = _load_importance_table(str(FAULT_IMPORTANCE_PATH))
    with binary_col:
        if binary_importance is not None:
            st.plotly_chart(
                _importance_chart(binary_importance, "Binary detector — top features", theme),
                width="stretch", config={"displayModeBar": False},
            )
        else:
            st.caption("Feature importance artifact not available.")
    with fault_col:
        if fault_importance is not None:
            st.plotly_chart(
                _importance_chart(fault_importance, "Fault classifier — top features", theme),
                width="stretch", config={"displayModeBar": False},
            )
        else:
            st.caption("Feature importance artifact not available.")
    st.caption("Permutation importance reflects predictive contribution and does not establish causality.")


def render_architecture_panel() -> None:
    st.markdown('<div class="scada-eyebrow">System Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="scada-arch">
SCADA Telemetry
      &darr;
Pydantic Validation
      &darr;
Prediction Service
      &darr;
<b>Binary Failure Detector</b>
      &darr;
Threshold Policy
   &swarr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&searr;
Healthy&nbsp;&nbsp;&nbsp;&nbsp;Abnormal
             &darr;
       <b>Fault Classifier</b>
             &darr;
     Structured Prediction
   &swarr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&searr;
FastAPI&nbsp;&nbsp;&nbsp;&nbsp;Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Both fitted artifacts contain preprocessing and classification in one scikit-learn pipeline. "
        "FastAPI and Streamlit call the same shared prediction service."
    )


def render_validation_strategy_panel() -> None:
    st.markdown('<div class="scada-eyebrow">Validation Strategy</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="scada-arch">
50 Segments
      &darr;
Group-Aware Train/Test Split
      &darr;
No Segment Overlap
      &darr;
Repeated StratifiedGroupKFold
      &darr;
Held-Out Evaluation
        </div>
        """,
        unsafe_allow_html=True,
    )
    highlight_col1, highlight_col2, highlight_col3 = st.columns(3)
    for column, text in zip(
        (highlight_col1, highlight_col2, highlight_col3),
        ("No random row split.", "No event_type leakage.", "No alarm_triggered leakage."),
    ):
        column.markdown(
            f'<div class="scada-card scada-card-normal" style="text-align:center;">'
            f'<div class="v" style="font-size:.92rem;">{text}</div></div>',
            unsafe_allow_html=True,
        )


def render_leakage_protection_panel() -> None:
    st.markdown('<div class="scada-eyebrow">Leakage-Safe Inputs</div>', unsafe_allow_html=True)
    approved_col, excluded_col = st.columns(2)
    with approved_col:
        items = "".join(f"<li><span class='mark'>&check;</span>{feature}</li>" for feature in MODEL_FEATURES)
        st.markdown(
            f'<div class="scada-panel-card"><div class="heading" style="font-size:.95rem; margin-bottom:8px;">'
            f'Model Inputs</div><ul class="scada-checklist scada-checklist-approved">{items}</ul></div>',
            unsafe_allow_html=True,
        )
    with excluded_col:
        excluded = ["event_type", "alarm_triggered", "target", "timestamp", "segment_id"]
        items = "".join(f"<li><span class='mark'>&cross;</span>{feature}</li>" for feature in excluded)
        st.markdown(
            f'<div class="scada-panel-card"><div class="heading" style="font-size:.95rem; margin-bottom:8px;">'
            f'Excluded</div><ul class="scada-checklist scada-checklist-excluded">{items}</ul></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "event_type and alarm_triggered encode the outcome being predicted; target is the label itself; "
        "timestamp and segment_id are grouping/ordering metadata, not sensor readings."
    )


def render_service_status_panel() -> None:
    st.markdown('<div class="scada-eyebrow">Service Status</div>', unsafe_allow_html=True)
    binary_loaded = Path(MODEL_PATH).is_file()
    fault_loaded = Path(FAULT_MODEL_PATH).is_file()
    rows = [
        ("Binary Model", "Loaded" if binary_loaded else "Missing"),
        ("Fault Model", "Loaded" if fault_loaded else "Missing"),
        ("Standard Threshold", f"{STANDARD_THRESHOLD:.2f}"),
        ("High Sensitivity Threshold", f"{HIGH_SENSITIVITY_THRESHOLD:.2f}"),
        ("Critical Threshold", f"{CRITICAL_THRESHOLD:.2f}"),
        ("API", "FastAPI inference service"),
        ("Dashboard", "Streamlit"),
    ]
    st.markdown(_summary_card("Platform", rows), unsafe_allow_html=True)


def render_limitations_panel() -> None:
    st.markdown('<div class="scada-eyebrow">Prototype Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        "- 1,000 observations\n"
        "- 306 abnormal examples\n"
        "- 50 segments\n"
        "- 17 timestamps\n"
        "- Roughly a 17-minute horizon\n"
        "- Simulated/synthetic-style data\n"
        "- Potentially strong dataset-generation patterns\n"
        "- No live industrial SCADA connection\n"
        "- No prospective field validation\n"
        "- **Not an operational safety system.**"
    )


def render_control_bar(service: Any, segments: list[Any]) -> dict[str, Any]:
    """Shared operational control bar for Operations Overview and Telemetry & Investigation.

    Reuses the same widget keys on both pages so segment, observation
    position, monitoring mode, and replay speed persist across navigation
    instead of resetting.
    """
    columns = st.columns([2, 2.3, 1.8, 0.9, 0.9, 0.8, 0.9])

    segment_kwargs: dict[str, Any] = {"key": SEGMENT_KEY}
    if SEGMENT_KEY not in st.session_state:
        remembered_segment = recall(SEGMENT_KEY, segments[0])
        segment_kwargs["index"] = segments.index(remembered_segment) if remembered_segment in segments else 0
    with columns[0]:
        segment_id = st.selectbox(
            "Segment", segments, format_func=lambda value: f"Segment {value}",
            on_change=reset_record_position, **segment_kwargs,
        )
    remember(SEGMENT_KEY, segment_id)

    records = service.segment_records(segment_id)
    record_count = len(records)

    position_kwargs: dict[str, Any] = {"key": POSITION_KEY}
    if POSITION_KEY not in st.session_state:
        remembered_position = recall(POSITION_KEY, 0)
        position_kwargs["index"] = min(remembered_position, record_count - 1)
    with columns[1]:
        current_index = st.selectbox(
            "Observation", list(range(record_count)),
            format_func=lambda position: format_observation(position, records),
            **position_kwargs,
        )
    current_index = min(current_index, record_count - 1)
    remember(POSITION_KEY, current_index)

    mode_kwargs: dict[str, Any] = {"key": MODE_KEY}
    if MODE_KEY not in st.session_state:
        remembered_mode = recall(MODE_KEY, "Standard")
        mode_options = ["Standard", "High Sensitivity"]
        mode_kwargs["index"] = mode_options.index(remembered_mode) if remembered_mode in mode_options else 0
    with columns[2]:
        mode = st.selectbox("Monitoring Mode", ["Standard", "High Sensitivity"], **mode_kwargs)
    remember(MODE_KEY, mode)
    threshold = get_monitoring_threshold(mode)

    replay_speed_kwargs: dict[str, Any] = {"key": REPLAY_SPEED_KEY}
    if REPLAY_SPEED_KEY not in st.session_state:
        remembered_speed = recall(REPLAY_SPEED_KEY, "1x")
        speed_options = ["1x", "2x", "5x"]
        replay_speed_kwargs["index"] = speed_options.index(remembered_speed) if remembered_speed in speed_options else 0
    with columns[3]:
        st.markdown('<div style="height:1.7rem;"></div>', unsafe_allow_html=True)
        st.button(
            "◀ Previous", width="stretch", disabled=current_index <= 0, key="control_bar_previous",
            on_click=go_previous, args=(current_index,),
        )
    with columns[4]:
        st.markdown('<div style="height:1.7rem;"></div>', unsafe_allow_html=True)
        st.button(
            "▶ Next", width="stretch", disabled=current_index >= record_count - 1, key="control_bar_next",
            on_click=go_next, args=(current_index, record_count),
        )
    with columns[5]:
        replay_speed = st.selectbox("Speed", ["1x", "2x", "5x"], **replay_speed_kwargs)
    remember(REPLAY_SPEED_KEY, replay_speed)
    with columns[6]:
        st.markdown('<div style="height:1.7rem;"></div>', unsafe_allow_html=True)
        st.button(
            "⏭ Skip", width="stretch", disabled=current_index >= record_count - 1, key="control_bar_skip",
            on_click=go_skip, args=(current_index, record_count, replay_speed),
        )

    return {
        "segment_id": segment_id,
        "records": records,
        "record_count": record_count,
        "current_index": current_index,
        "mode": mode,
        "threshold": threshold,
        "replay_speed": replay_speed,
    }


def render_observation_details(current_row: pd.Series, prediction: dict[str, Any], current_index: int, record_count: int) -> None:
    with st.expander("Observation Details"):
        info_col, telemetry_col = st.columns([1, 2])
        with info_col:
            st.markdown(
                f"**Timestamp:** {pd.Timestamp(prediction['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**Segment:** {prediction['segment_id']}\n\n"
                f"**Observation position:** {current_index + 1} of {record_count}"
            )
        with telemetry_col:
            telemetry = prediction["telemetry"]
            frame = pd.DataFrame({"Feature": list(telemetry.keys()), "Value": list(telemetry.values())})
            st.dataframe(frame, width="stretch", hide_index=True)
