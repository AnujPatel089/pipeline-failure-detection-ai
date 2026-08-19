"""Reusable two-stage inference for binary condition and abnormal fault type."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FAULT_MODEL_PATH, MODEL_FEATURES, MODEL_PATH


def predict_with_models(
    record: dict[str, Any],
    binary_model: Any,
    fault_model: Any,
    failure_threshold: float = 0.5,
) -> dict[str, Any]:
    """Route approved telemetry through binary detection and, if needed, Model 2."""
    missing = sorted(set(MODEL_FEATURES) - set(record))
    if missing:
        raise ValueError(f"Missing telemetry features: {missing}")
    telemetry = pd.DataFrame([{feature: record[feature] for feature in MODEL_FEATURES}])
    failure_probability = float(binary_model.predict_proba(telemetry)[0, 1])
    if failure_probability < failure_threshold:
        return {"status": "normal", "failure_probability": failure_probability}

    fault_probabilities = fault_model.predict_proba(telemetry)[0]
    fault_index = int(fault_probabilities.argmax())
    fault_classes = fault_model.named_steps["classifier"].classes_
    return {
        "status": "abnormal",
        "failure_probability": failure_probability,
        "fault_type": str(fault_classes[fault_index]),
        "fault_confidence": float(fault_probabilities[fault_index]),
    }


def predict_pipeline_condition(
    record: dict[str, Any],
    binary_model_path: Path = MODEL_PATH,
    fault_model_path: Path = FAULT_MODEL_PATH,
    failure_threshold: float = 0.5,
) -> dict[str, Any]:
    """Load both fitted pipelines and predict one SCADA telemetry record."""
    binary_model = joblib.load(binary_model_path)
    fault_model = joblib.load(fault_model_path)
    return predict_with_models(record, binary_model, fault_model, failure_threshold)
