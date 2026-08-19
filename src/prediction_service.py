"""Long-lived model service shared by external presentation layers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib

from src.config import FAULT_MODEL_PATH, MODEL_PATH
from src.inference import predict_with_models
from src.model_metadata import MODEL_METADATA_PATH, load_model_metadata
from src.monitoring import get_alert_level, get_monitoring_threshold


class PredictionService:
    """Own two loaded pipelines and provide monitoring-mode-aware inference."""

    def __init__(
        self,
        binary_model: Any | None = None,
        fault_model: Any | None = None,
        binary_model_path: Path = MODEL_PATH,
        fault_model_path: Path = FAULT_MODEL_PATH,
        metadata: dict[str, Any] | None = None,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        self.binary_model = binary_model if binary_model is not None else joblib.load(binary_model_path)
        self.fault_model = fault_model if fault_model is not None else joblib.load(fault_model_path)
        self.metadata = metadata if metadata is not None else load_model_metadata(metadata_path)

    def predict(self, record: dict[str, Any], monitoring_mode: str = "standard") -> dict[str, Any]:
        """Predict through the shared inference function at the selected threshold."""
        threshold = get_monitoring_threshold(monitoring_mode)
        prediction = predict_with_models(
            record, self.binary_model, self.fault_model, failure_threshold=threshold
        )
        return {
            **prediction,
            "alert_level": get_alert_level(prediction["failure_probability"], threshold).lower(),
            "active_threshold": threshold,
            "monitoring_mode": monitoring_mode,
        }

    def model_info(self) -> dict[str, Any]:
        """Return validated, non-sensitive centralized metadata."""
        return deepcopy(self.metadata)
