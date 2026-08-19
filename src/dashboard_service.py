"""Leakage-safe data and inference services for the Streamlit dashboard."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    FAULT_MODEL_PATH,
    FAULT_TARGET_COLUMN,
    GROUP_COLUMN,
    MODEL_FEATURES,
    MODEL_PATH,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.data_loader import load_dataset
from src.inference import predict_with_models
from src.monitoring import get_alert_level, get_operational_message, get_pipeline_status
from src.validation import validate_dataset

EVALUATION_FIELDS = [TARGET_COLUMN, FAULT_TARGET_COLUMN, "alarm_triggered"]


def sanitize_inference_features(record: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Return only the eight approved telemetry fields in canonical order."""
    source = record.to_dict() if isinstance(record, pd.Series) else record
    missing = sorted(set(MODEL_FEATURES) - set(source))
    if missing:
        raise ValueError(f"Missing dashboard telemetry features: {missing}")
    return {feature: source[feature] for feature in MODEL_FEATURES}


def attach_evaluation_metadata(
    prediction: dict[str, Any],
    raw_record: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    """Attach ground truth to an already-created prediction for evaluation display."""
    if "status" not in prediction or "failure_probability" not in prediction:
        raise ValueError("Prediction must be completed before ground truth is attached")
    source = raw_record.to_dict() if isinstance(raw_record, pd.Series) else raw_record
    result = deepcopy(prediction)
    result["evaluation_only"] = {
        "actual_target": int(source[TARGET_COLUMN]),
        "actual_event_type": str(source[FAULT_TARGET_COLUMN]),
        "original_alarm_state": int(source["alarm_triggered"]),
    }
    return result


class DashboardService:
    """Load real telemetry and route sanitized rows through both saved pipelines."""

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        binary_model: Any | None = None,
        fault_model: Any | None = None,
        binary_model_path: Path = MODEL_PATH,
        fault_model_path: Path = FAULT_MODEL_PATH,
    ) -> None:
        if data is None:
            data, _ = load_dataset()
            data, _ = validate_dataset(data)
        else:
            data = data.copy()
            data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="raise")
        self.data = data.sort_values([GROUP_COLUMN, TIMESTAMP_COLUMN], kind="stable").reset_index(drop=True)
        self.binary_model = binary_model if binary_model is not None else joblib.load(binary_model_path)
        self.fault_model = fault_model if fault_model is not None else joblib.load(fault_model_path)

    def available_segments(self) -> list[Any]:
        """Return segment IDs in natural numeric/string order."""
        return sorted(
            self.data[GROUP_COLUMN].unique().tolist(),
            key=lambda value: (0, int(str(value))) if str(value).isdigit() else (1, str(value)),
        )

    def segment_records(self, segment_id: Any) -> pd.DataFrame:
        """Return one segment's actual observations in chronological order."""
        records = self.data[self.data[GROUP_COLUMN] == segment_id].copy()
        if records.empty:
            raise ValueError(f"Unknown pipeline segment: {segment_id!r}")
        return records.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)

    def predict_record(
        self,
        raw_record: pd.Series | dict[str, Any],
        active_threshold: float,
        monitoring_mode: str,
        evaluation_mode: bool = False,
    ) -> dict[str, Any]:
        """Sanitize, predict, enrich for UI, then optionally attach ground truth."""
        source = raw_record.to_dict() if isinstance(raw_record, pd.Series) else raw_record
        telemetry = sanitize_inference_features(source)
        prediction = predict_with_models(
            telemetry, self.binary_model, self.fault_model, failure_threshold=active_threshold
        )
        probability = prediction["failure_probability"]
        status = get_pipeline_status(probability, active_threshold)
        result = {
            **prediction,
            "pipeline_status": status,
            "alert_level": get_alert_level(probability, active_threshold),
            "operational_message": get_operational_message(status, prediction.get("fault_type")),
            "active_threshold": active_threshold,
            "monitoring_mode": monitoring_mode,
            "timestamp": pd.Timestamp(source[TIMESTAMP_COLUMN]).isoformat(),
            "segment_id": source[GROUP_COLUMN],
            "telemetry": telemetry,
        }
        return attach_evaluation_metadata(result, source) if evaluation_mode else result
