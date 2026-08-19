"""Centralized session-state keys and pure state-resolution logic shared across pages.

Keeping these keys in one place is what lets Operations Overview and
Telemetry & Investigation share the same segment, replay position,
monitoring mode, and prediction history without duplicating state.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import TIMESTAMP_COLUMN
from src.monitoring import advance_replay_index

PAGE_NAMES = ["Operations Overview", "Telemetry & Investigation", "Model & System"]
DEFAULT_PAGE = PAGE_NAMES[0]

PAGE_KEY = "active_page"
THEME_KEY = "dashboard_theme"
SEGMENT_KEY = "active_segment"
POSITION_KEY = "record_position"
MODE_KEY = "monitoring_mode"
REPLAY_SPEED_KEY = "replay_speed"
EVALUATION_KEY = "evaluation_mode"
HISTORY_KEY = "prediction_history"
HISTORY_KEYS_KEY = "history_keys"


def _shadow_key(key: str) -> str:
    return f"_{key}_shadow"


def remember(key: str, value: Any) -> None:
    """Persist a widget's value outside Streamlit's per-run widget lifecycle.

    Streamlit clears a widget-backed session_state key whenever that widget
    is not instantiated during a run — which happens here on Model & System,
    since it never renders the control bar. Mirroring the value into a
    plain, non-widget key lets Operations/Investigation restore the right
    selection instead of silently falling back to a hardcoded default.
    """
    st.session_state[_shadow_key(key)] = value


def recall(key: str, default: Any) -> Any:
    """Read a widget's current value, falling back to its persisted shadow.

    Streamlit syncs a widget-backed key to its latest interaction before the
    script even starts, so that value (when present) is always the freshest
    — it must be preferred over the shadow, which only updates once the
    widget itself re-renders later in this same run.
    """
    if key in st.session_state:
        return st.session_state[key]
    return st.session_state.get(_shadow_key(key), default)


def format_observation(position: int, records: pd.DataFrame) -> str:
    """Label one row by position, disambiguating rows that share a timestamp.

    Row position (not timestamp value) is the source of truth for which
    observation is selected: several rows can share an identical timestamp,
    so resolving position from a timestamp value would silently snap back
    to the first duplicate. This preserves that fix exactly.
    """
    row = records.iloc[position]
    label = pd.Timestamp(row[TIMESTAMP_COLUMN]).strftime("%Y-%m-%d %H:%M:%S")
    same_timestamp = records.index[records[TIMESTAMP_COLUMN] == row[TIMESTAMP_COLUMN]].tolist()
    if len(same_timestamp) > 1:
        occurrence = same_timestamp.index(position) + 1
        label = f"{label} (obs {occurrence}/{len(same_timestamp)})"
    return label


def set_record_position(new_index: int) -> None:
    st.session_state[POSITION_KEY] = new_index
    remember(POSITION_KEY, new_index)


def go_previous(current_index: int) -> None:
    set_record_position(max(0, current_index - 1))


def go_next(current_index: int, record_count: int) -> None:
    set_record_position(min(record_count - 1, current_index + 1))


def go_skip(current_index: int, record_count: int, replay_speed: str) -> None:
    set_record_position(advance_replay_index(current_index, replay_speed, record_count))


def reset_record_position() -> None:
    """Callback for a segment change: the previous position no longer applies."""
    st.session_state.pop(POSITION_KEY, None)
    remember(POSITION_KEY, 0)


def record_prediction_history(prediction: dict[str, Any]) -> None:
    """Append one prediction to session history, de-duplicated by (time, segment, mode)."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if HISTORY_KEYS_KEY not in st.session_state:
        st.session_state[HISTORY_KEYS_KEY] = set()
    key = (prediction["timestamp"], str(prediction["segment_id"]), prediction["monitoring_mode"])
    if key not in st.session_state[HISTORY_KEYS_KEY]:
        st.session_state[HISTORY_KEY].append({
            "timestamp": prediction["timestamp"],
            "segment_id": prediction["segment_id"],
            "status": prediction["pipeline_status"],
            "failure_probability": prediction["failure_probability"],
            "fault_type": prediction.get("fault_type"),
            "fault_confidence": prediction.get("fault_confidence"),
            "alert_level": prediction["alert_level"],
        })
        st.session_state[HISTORY_KEYS_KEY].add(key)


def resolve_current_prediction(
    service: Any, records: pd.DataFrame, current_index: int, mode: str, threshold: float
) -> tuple[pd.Series, pd.Series | None, dict[str, Any]]:
    """Predict the selected observation and record it to shared session history.

    Both Operations Overview and Telemetry & Investigation call this so a
    prediction made on one page is consistent with the other; the history
    dedup guard makes revisiting the same (timestamp, segment, mode) safe.
    """
    current = records.iloc[current_index]
    previous = records.iloc[current_index - 1] if current_index > 0 else None
    # Read the persisted shadow, not the raw widget key: the evaluation
    # checkbox only renders on Telemetry & Investigation, so its widget-backed
    # session_state key is cleared by Streamlit whenever another page is
    # active for a run. The shadow survives that and stays correct.
    evaluation_mode = bool(recall(EVALUATION_KEY, False))
    prediction = service.predict_record(current, threshold, mode, evaluation_mode=evaluation_mode)
    record_prediction_history(prediction)
    return current, previous, prediction
