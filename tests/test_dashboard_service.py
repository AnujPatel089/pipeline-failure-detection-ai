from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import FAULT_MODEL_PATH, GROUP_COLUMN, MODEL_FEATURES, MODEL_PATH, TIMESTAMP_COLUMN
from src.dashboard_service import DashboardService, attach_evaluation_metadata, sanitize_inference_features
from src.monitoring import CRITICAL_THRESHOLD, HIGH_SENSITIVITY_THRESHOLD, STANDARD_THRESHOLD, advance_replay_index, get_alert_level, get_pipeline_status


class BinarySpy:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.seen_columns: list[str] = []

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen_columns = frame.columns.tolist()
        return np.array([[1 - self.probability, self.probability]])


class FaultSpy:
    def __init__(self, fail_if_called: bool = False) -> None:
        self.fail_if_called = fail_if_called
        self.calls = 0
        self.seen_columns: list[str] = []
        classifier = SimpleNamespace(classes_=np.array(["blockage", "degradation", "leak", "surge"]))
        self.named_steps = {"classifier": classifier}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("Fault model must not run for a normal prediction")
        self.seen_columns = frame.columns.tolist()
        return np.array([[0.05, 0.10, 0.80, 0.05]])


def raw_records() -> pd.DataFrame:
    base = {feature: 1.0 for feature in MODEL_FEATURES}
    return pd.DataFrame([
        {**base, TIMESTAMP_COLUMN: "2024-01-01 00:02:00", GROUP_COLUMN: 1, "target": 1, "event_type": "leak", "alarm_triggered": 1},
        {**base, TIMESTAMP_COLUMN: "2024-01-01 00:00:00", GROUP_COLUMN: 1, "target": 0, "event_type": "normal", "alarm_triggered": 0},
        {**base, TIMESTAMP_COLUMN: "2024-01-01 00:01:00", GROUP_COLUMN: 1, "target": 1, "event_type": "surge", "alarm_triggered": 1},
    ])


@pytest.mark.parametrize("forbidden", ["event_type", "alarm_triggered", "target", GROUP_COLUMN, TIMESTAMP_COLUMN])
def test_leakage_fields_never_reach_inference_features(forbidden: str) -> None:
    sanitized = sanitize_inference_features(raw_records().iloc[0])
    assert forbidden not in sanitized
    assert list(sanitized) == MODEL_FEATURES


def test_segment_records_are_chronological() -> None:
    service = DashboardService(raw_records(), BinarySpy(0.2), FaultSpy())
    records = service.segment_records(1)
    assert records[TIMESTAMP_COLUMN].is_monotonic_increasing


@pytest.mark.parametrize(("speed", "expected"), [("Manual", 1), ("1x", 1), ("2x", 2), ("5x", 5)])
def test_replay_speed_advances_expected_observations(speed: str, expected: int) -> None:
    assert advance_replay_index(0, speed, 10) == expected
    assert advance_replay_index(8, speed, 10) <= 9


def test_monitoring_thresholds_and_critical_logic() -> None:
    assert STANDARD_THRESHOLD == 0.50
    assert HIGH_SENSITIVITY_THRESHOLD == 0.30
    assert CRITICAL_THRESHOLD == 0.80
    assert get_alert_level(0.79, STANDARD_THRESHOLD) == "WARNING"
    assert get_alert_level(0.80, STANDARD_THRESHOLD) == "CRITICAL"
    assert get_pipeline_status(0.29, HIGH_SENSITIVITY_THRESHOLD) == "NORMAL"


def test_normal_prediction_does_not_invoke_or_show_fault_type() -> None:
    binary, fault = BinarySpy(0.2), FaultSpy(fail_if_called=True)
    service = DashboardService(raw_records(), binary, fault)
    result = service.predict_record(service.segment_records(1).iloc[0], STANDARD_THRESHOLD, "Standard")
    assert result["status"] == "normal"
    assert "fault_type" not in result
    assert fault.calls == 0
    assert binary.seen_columns == MODEL_FEATURES


def test_evaluation_metadata_is_attached_only_after_prediction() -> None:
    record = raw_records().iloc[0]
    with pytest.raises(ValueError, match="Prediction must be completed"):
        attach_evaluation_metadata({}, record)
    binary, fault = BinarySpy(0.9), FaultSpy()
    service = DashboardService(raw_records(), binary, fault)
    result = service.predict_record(record, STANDARD_THRESHOLD, "Standard", evaluation_mode=True)
    assert result["fault_type"] == "leak"
    assert result["evaluation_only"]["actual_event_type"] == "leak"
    assert binary.seen_columns == MODEL_FEATURES
    assert fault.seen_columns == MODEL_FEATURES


def test_dashboard_service_loads_saved_pipelines(scada_frame) -> None:
    assert MODEL_PATH.is_file() and FAULT_MODEL_PATH.is_file()
    service = DashboardService(scada_frame)
    record = service.segment_records(service.available_segments()[0]).iloc[0]
    result = service.predict_record(record, STANDARD_THRESHOLD, "Standard")
    assert result["status"] in {"normal", "abnormal"}
    assert 0.0 <= result["failure_probability"] <= 1.0
