"""Deterministic, redistribution-safe fixtures for software and model integration tests."""
from __future__ import annotations

from itertools import product

import pandas as pd
import pytest

from src.config import MODEL_FEATURES
from src.prediction_service import PredictionService


@pytest.fixture(scope="session")
def scada_frame() -> pd.DataFrame:
    """Construct valid SCADA-like rows; this is not copied from the Kaggle dataset."""
    events = ["normal", "blockage", "degradation", "leak", "surge"]
    rows = []
    for segment_id in range(1, 51):
        for minute, event_type in enumerate(events):
            abnormal = event_type != "normal"
            rows.append({
                "timestamp": f"2025-01-01 00:{minute:02d}:00",
                "segment_id": segment_id,
                "pressure": 58.0 + segment_id * 0.7 + minute * 3.0,
                "flow_rate": 2.0 + (segment_id % 8) * 0.4 + minute * 0.15,
                "temperature": 27.0 + (segment_id % 6) + minute * 0.2,
                "valve_status": (segment_id + minute) % 3,
                "pump_state": (segment_id + minute) % 2,
                "pump_speed": 0.0 if (segment_id + minute) % 4 == 0 else 900.0 + segment_id * 8.0,
                "compressor_state": (segment_id // 2 + minute) % 2,
                "energy_consumption": 12.0 + segment_id * 0.55 + minute * 2.0,
                "alarm_triggered": int(abnormal and minute % 2 == 0),
                "event_type": event_type,
                "target": int(abnormal),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def saved_prediction_service() -> PredictionService:
    """Load the real committed pipelines without loading the external dataset."""
    return PredictionService()


@pytest.fixture(scope="session")
def model_telemetry_examples(saved_prediction_service: PredictionService) -> dict[str, dict]:
    """Find deterministic behavioral examples from a generated telemetry grid."""
    candidates = [
        dict(zip(MODEL_FEATURES, values))
        for values in product(
            [50.0, 65.0, 80.0, 95.0, 110.0],
            [1.5, 3.0, 4.5, 6.0, 7.0],
            [27.0, 32.0, 37.0],
            [0.0, 800.0, 1500.0],
            [8.0, 25.0, 50.0],
            [0, 1, 2],
            [0, 1],
            [0, 1],
        )
    ]
    frame = pd.DataFrame(candidates, columns=MODEL_FEATURES)
    probabilities = saved_prediction_service.binary_model.predict_proba(frame)[:, 1]
    minimum = int(probabilities.argmin())
    maximum = int(probabilities.argmax())
    if probabilities[minimum] >= 0.50 or probabilities[maximum] < 0.50:
        raise AssertionError("Saved binary model must produce both standard-mode outcomes")
    return {"normal": candidates[minimum], "abnormal": candidates[maximum]}
