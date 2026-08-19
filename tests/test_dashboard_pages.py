from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import MODEL_FEATURES
from src.dashboard_service import sanitize_inference_features
from src.dashboard_state import DEFAULT_PAGE, PAGE_NAMES, format_observation
from src.dashboard_styles import DEFAULT_THEME, THEME_NAMES, get_theme

_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app() -> AppTest:
    app = AppTest.from_file(str(_APP_PATH), default_timeout=60)
    app.run()
    return app


def test_exactly_three_pages_with_operations_default() -> None:
    assert PAGE_NAMES == ["Operations Overview", "Telemetry & Investigation", "Model & System"]
    assert DEFAULT_PAGE == "Operations Overview"
    app = _run_app()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Operations Overview"


def test_navigation_preserves_segment_position_mode_theme() -> None:
    app = _run_app()
    app.main.selectbox[0].select(5)  # segment
    app.run()
    app.main.selectbox[2].select("High Sensitivity")  # monitoring mode
    app.run()
    app.sidebar.selectbox[0].select("Deep Navy")  # theme
    app.run()

    segment_before = app.session_state["active_segment"]
    mode_before = app.session_state["monitoring_mode"]
    theme_before = app.session_state["dashboard_theme"]
    position_before = app.session_state["record_position"]

    for page in ["Telemetry & Investigation", "Model & System", "Operations Overview"]:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception, f"Exception on {page}: {app.exception}"

    assert app.session_state["active_segment"] == segment_before
    assert app.session_state["record_position"] == position_before
    assert app.session_state["monitoring_mode"] == mode_before
    assert app.session_state["dashboard_theme"] == theme_before


def test_navigation_preserves_prediction_history() -> None:
    app = _run_app()
    app.main.selectbox[0].select(43)
    app.run()
    history_length = len(app.session_state["prediction_history"])
    assert history_length > 0

    app.sidebar.radio[0].set_value("Model & System")
    app.run()
    app.sidebar.radio[0].set_value("Operations Overview")
    app.run()
    assert len(app.session_state["prediction_history"]) >= history_length


def test_duplicate_timestamp_replay_is_row_position_based() -> None:
    records = pd.DataFrame({"timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:00", "2024-01-01 00:01:00"]})
    label_first = format_observation(0, records)
    label_second = format_observation(1, records)
    assert label_first != label_second
    assert "(obs 1/2)" in label_first
    assert "(obs 2/2)" in label_second


def test_all_themes_resolve_and_invalid_falls_back() -> None:
    for name in THEME_NAMES:
        assert get_theme(name)["background"]
    assert get_theme("Nonexistent Theme") == get_theme(DEFAULT_THEME)


def test_model_input_sanitization_unchanged() -> None:
    record = {feature: 1.0 for feature in MODEL_FEATURES}
    record.update({"timestamp": "2024-01-01", "segment_id": 1, "target": 0, "event_type": "normal", "alarm_triggered": 0})
    sanitized = sanitize_inference_features(record)
    assert list(sanitized) == MODEL_FEATURES
