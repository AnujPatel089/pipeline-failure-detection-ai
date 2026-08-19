"""Central alert thresholds and deterministic operational messaging."""
from __future__ import annotations

STANDARD_THRESHOLD = 0.50
HIGH_SENSITIVITY_THRESHOLD = 0.30
CRITICAL_THRESHOLD = 0.80

MONITORING_THRESHOLDS = {
    "Standard": STANDARD_THRESHOLD,
    "High Sensitivity": HIGH_SENSITIVITY_THRESHOLD,
    "standard": STANDARD_THRESHOLD,
    "high_sensitivity": HIGH_SENSITIVITY_THRESHOLD,
}
REPLAY_STEPS = {"Manual": 1, "1x": 1, "2x": 2, "5x": 5}

FAULT_MESSAGES = {
    "leak": "Telemetry is consistent with a potential leak condition. Operator review is recommended.",
    "blockage": "Telemetry is consistent with a possible blockage. Inspect flow and pressure conditions.",
    "surge": "Telemetry indicates a possible surge condition. Operator review is recommended.",
    "degradation": "Telemetry suggests possible equipment or process degradation.",
}


def get_monitoring_threshold(mode: str) -> float:
    """Return the validated threshold for a named monitoring mode."""
    try:
        return MONITORING_THRESHOLDS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown monitoring mode: {mode!r}") from exc


def get_replay_step(replay_speed: str) -> int:
    """Return how many real observations one replay action advances."""
    try:
        return REPLAY_STEPS[replay_speed]
    except KeyError as exc:
        raise ValueError(f"Unknown replay speed: {replay_speed!r}") from exc


def advance_replay_index(current_index: int, replay_speed: str, record_count: int) -> int:
    """Advance by the selected speed without moving past available records."""
    if record_count <= 0:
        raise ValueError("Replay requires at least one record")
    return min(current_index + get_replay_step(replay_speed), record_count - 1)


def get_alert_level(failure_probability: float, active_threshold: float) -> str:
    """Map failure risk to INFO, WARNING, or CRITICAL."""
    if failure_probability >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if failure_probability >= active_threshold:
        return "WARNING"
    return "INFO"


def get_pipeline_status(failure_probability: float, active_threshold: float) -> str:
    """Map failure risk to a concise operator-facing status."""
    return {"INFO": "NORMAL", "WARNING": "WARNING", "CRITICAL": "CRITICAL"}[
        get_alert_level(failure_probability, active_threshold)
    ]


def get_operational_message(status: str, fault_type: str | None = None) -> str:
    """Return a cautious, deterministic message without claiming certainty."""
    if status == "NORMAL":
        return "No abnormal SCADA pattern detected at the current threshold."
    return FAULT_MESSAGES.get(
        (fault_type or "").lower(),
        "An abnormal SCADA pattern was detected. Operator review is recommended.",
    )
