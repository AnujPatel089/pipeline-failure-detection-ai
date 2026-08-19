"""Streamlit entry point for the SCADA monitoring portfolio prototype."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import GROUP_COLUMN, TIMESTAMP_COLUMN
from src.dashboard_service import DashboardService
from src.monitoring import advance_replay_index, get_monitoring_threshold

st.set_page_config(
    page_title="Pipeline Failure Detection AI",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .system-online {display:inline-block; padding:.25rem .65rem; border:1px solid #2e8b57;
      border-radius:999px; color:#1f7a4c; font-size:.78rem; font-weight:700; letter-spacing:.08em;}
    .subtle {color:#667085; margin-top:-.5rem;}
    [data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:.55rem;
      padding:.8rem 1rem; background:rgba(128,128,128,.035);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> DashboardService:
    """Load telemetry and both model pipelines once per Streamlit process."""
    return DashboardService()


def advance_record(timestamp_options: list[pd.Timestamp], current_index: int, replay_speed: str) -> None:
    """Advance replay state before the next Streamlit script rerun."""
    next_index = advance_replay_index(current_index, replay_speed, len(timestamp_options))
    st.session_state.record_index = next_index
    st.session_state.timestamp_selection = timestamp_options[next_index]


service = get_service()
segments = service.available_segments()

with st.sidebar:
    st.header("Monitoring controls")
    mode = st.radio("Monitoring mode", ["Standard", "High Sensitivity"])
    threshold = get_monitoring_threshold(mode)
    if mode == "High Sensitivity":
        st.warning("Uses a 0.30 threshold: higher failure recall, but substantially more false alerts.")
        st.caption("Validation: precision 76.4%, recall 95.2%.")
    else:
        st.caption("0.50 threshold. Validation: precision 92.5%, recall 89.5%.")

    segment_id = st.selectbox("Pipeline segment", segments, format_func=lambda value: f"Segment {value}")
    replay_speed = st.selectbox("Replay speed", ["Manual", "1x", "2x", "5x"])
    evaluation_mode = st.checkbox(
        "Evaluation mode",
        help="Shows dataset labels only after prediction. Labels never enter either model.",
    )

records = service.segment_records(segment_id)
state_segment = st.session_state.get("active_segment")
if state_segment != segment_id:
    st.session_state.active_segment = segment_id
    st.session_state.record_index = 0
    st.session_state.pop("timestamp_selection", None)

st.session_state.record_index = min(st.session_state.get("record_index", 0), len(records) - 1)
timestamp_options = records[TIMESTAMP_COLUMN].tolist()
timestamp_select_options = {"key": "timestamp_selection"}
if "timestamp_selection" not in st.session_state:
    timestamp_select_options["index"] = st.session_state.record_index
current_timestamp = st.sidebar.selectbox(
    "Timestamp",
    timestamp_options,
    format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S"),
    **timestamp_select_options,
)
selected_matches = records.index[records[TIMESTAMP_COLUMN] == current_timestamp].tolist()
current_index = selected_matches[0]
st.session_state.record_index = current_index

st.sidebar.button(
    "Next observation", width="stretch", disabled=current_index >= len(records) - 1,
    on_click=advance_record, args=(timestamp_options, current_index, replay_speed),
)
st.sidebar.caption("Replay advances through real observations; no synthetic telemetry is generated.")

current = records.iloc[current_index]
prediction = service.predict_record(current, threshold, mode, evaluation_mode=evaluation_mode)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "history_keys" not in st.session_state:
    st.session_state.history_keys = set()
history_key = (prediction["timestamp"], str(prediction["segment_id"]), mode)
if history_key not in st.session_state.history_keys:
    st.session_state.prediction_history.append({
        "timestamp": prediction["timestamp"],
        "segment_id": prediction["segment_id"],
        "status": prediction["pipeline_status"],
        "failure_probability": prediction["failure_probability"],
        "fault_type": prediction.get("fault_type"),
        "fault_confidence": prediction.get("fault_confidence"),
        "alert_level": prediction["alert_level"],
    })
    st.session_state.history_keys.add(history_key)

title_col, indicator_col = st.columns([5, 1])
with title_col:
    st.title("PIPELINE FAILURE DETECTION AI")
    st.markdown('<p class="subtle">Real-time SCADA monitoring & predictive fault intelligence</p>', unsafe_allow_html=True)
with indicator_col:
    st.markdown("<br><span class='system-online'>● SYSTEM ONLINE</span>", unsafe_allow_html=True)
    st.caption("Prototype replay")

kpis = st.columns(5)
kpis[0].metric("Pipeline Status", prediction["pipeline_status"])
kpis[1].metric("Failure Probability", f"{prediction['failure_probability']:.1%}")
kpis[2].metric("Fault Type", prediction.get("fault_type", "None detected").title())
kpis[3].metric("Fault Confidence", f"{prediction['fault_confidence']:.1%}" if prediction.get("fault_confidence") is not None else "N/A")
kpis[4].metric("Segment", f"Segment {prediction['segment_id']}")

if prediction["alert_level"] == "CRITICAL":
    st.error(f"CRITICAL ALERT — {prediction['operational_message']}")
elif prediction["alert_level"] == "WARNING":
    st.warning(f"WARNING — {prediction['operational_message']}")
else:
    st.success(f"INFO — {prediction['operational_message']}")
if prediction["alert_level"] in {"WARNING", "CRITICAL"}:
    confidence_text = (
        f"{prediction['fault_confidence']:.1%}"
        if prediction.get("fault_confidence") is not None else "N/A"
    )
    st.caption(
        f"Alert details: {pd.Timestamp(prediction['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Segment {prediction['segment_id']} · Failure {prediction['failure_probability']:.1%} · "
        f"Fault {prediction.get('fault_type', 'unclassified').title()} · Confidence {confidence_text}"
    )

st.subheader("SCADA Telemetry")
telemetry = prediction["telemetry"]
sensor_columns = st.columns(5)
for column, label, feature in zip(
    sensor_columns,
    ["Pressure", "Flow Rate", "Temperature", "Pump Speed", "Energy Consumption"],
    ["pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"],
):
    column.metric(label, f"{telemetry[feature]:,.2f}")
state_columns = st.columns(3)
state_columns[0].metric("Valve Status", str(telemetry["valve_status"]))
state_columns[1].metric("Pump State", str(telemetry["pump_state"]))
state_columns[2].metric("Compressor State", str(telemetry["compressor_state"]))
st.caption("Values are shown without units because units are not defined in the supplied dataset.")

st.subheader("Sensor Trends")
history = records.iloc[: current_index + 1].set_index(TIMESTAMP_COLUMN)
chart_groups = [
    ("Pressure", "pressure"), ("Flow Rate", "flow_rate"), ("Temperature", "temperature"),
    ("Pump Speed", "pump_speed"), ("Energy Consumption", "energy_consumption"),
]
for start in range(0, len(chart_groups), 2):
    chart_columns = st.columns(2)
    for chart_column, (label, feature) in zip(chart_columns, chart_groups[start : start + 2]):
        with chart_column:
            st.markdown(f"**{label}**")
            st.line_chart(history[[feature]], height=220, width="stretch")

st.subheader("AI Risk Assessment")
risk_left, risk_right = st.columns([2, 3])
with risk_left:
    st.metric("Failure probability", f"{prediction['failure_probability']:.1%}")
    st.write(f"**Active threshold:** {threshold:.0%}")
    st.write(f"**Monitoring mode:** {mode}")
with risk_right:
    st.write(f"**Predicted fault:** {prediction.get('fault_type', 'None detected').title()}")
    confidence = prediction.get("fault_confidence")
    st.write(f"**Fault confidence:** {confidence:.1%}" if confidence is not None else "**Fault confidence:** N/A")
    st.info(prediction["operational_message"])

if evaluation_mode:
    with st.expander("Ground Truth — Evaluation Only", expanded=True):
        truth = prediction["evaluation_only"]
        truth_columns = st.columns(3)
        truth_columns[0].metric("Actual target", truth["actual_target"])
        truth_columns[1].metric("Actual event type", truth["actual_event_type"].title())
        truth_columns[2].metric("Original alarm state", truth["original_alarm_state"])
        st.caption("Attached after prediction and never passed to either model.")

st.subheader("Recent Prediction History")
history_header, clear_column = st.columns([5, 1])
with clear_column:
    if st.button("Clear session history"):
        st.session_state.prediction_history = []
        st.session_state.history_keys = set()
        st.rerun()
if st.session_state.prediction_history:
    display_history = pd.DataFrame(reversed(st.session_state.prediction_history))
    display_history["failure_probability"] = display_history["failure_probability"].map(lambda value: f"{value:.1%}")
    display_history["fault_confidence"] = display_history["fault_confidence"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")
    st.dataframe(display_history, width="stretch", hide_index=True)
else:
    st.caption("No predictions recorded in this session.")

with st.expander("Model Information"):
    st.markdown("""
    - **Binary detector:** Random Forest — normal versus abnormal.
    - **Fault classifier:** Random Forest — blockage, degradation, leak, or surge.
    - **Main binary predictive features:** energy consumption, pressure, flow rate, pump speed, compressor state.
    - **Main fault-classifier predictive features:** pressure, flow rate, pump speed, energy consumption, compressor state.

    Importance rankings come from held-out permutation analysis. They describe predictive contribution, not causation.
    """)

with st.expander("Validation Performance"):
    st.markdown("""
    **Binary detector:** failure recall 93.1%, F1 90.8%, PR-AUC 97.2% on official held-out segments.

    **Repeated group validation:** recall 88.1% ± 4.1%.

    **Fault classifier:** macro F1 97.3%.

    **Fault group validation:** macro F1 97.2% ± 2.0%.

    These metrics are from a small synthetic/simulated-style SCADA dataset and should not be interpreted as production performance.
    """)

with st.expander("Prototype Limitations"):
    st.markdown("""
    - 1,000 rows, 306 abnormal examples, 50 segments, and only 17 timestamps (approximately 17 minutes).
    - Short time horizon and unusually separable telemetry patterns; synthetic or dataset-generation artifacts may exist.
    - No live SCADA integration, prospective validation, or real-world deployment evidence.

    **This application is an ML portfolio prototype and not an operational safety system.**
    """)
