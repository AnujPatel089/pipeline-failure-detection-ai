"""Loading and strict validation for safe descriptive model metadata."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from src.config import FAULT_LABELS, MODEL_FEATURES, MODELS_DIR
from src.monitoring import HIGH_SENSITIVITY_THRESHOLD, STANDARD_THRESHOLD

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


class ModelMetadataError(ValueError):
    """Raised when centralized model metadata is absent or inconsistent."""


def validate_model_metadata(metadata: Any) -> dict[str, Any]:
    """Validate required structure and agreement with the deployed model contract."""
    if not isinstance(metadata, dict):
        raise ModelMetadataError("Model metadata must be a JSON object")
    required_sections = {"artifact_version", "binary_model", "fault_classifier", "dataset"}
    missing = sorted(required_sections - set(metadata))
    if missing:
        raise ModelMetadataError(f"Model metadata is missing sections: {missing}")
    binary = metadata["binary_model"]
    fault = metadata["fault_classifier"]
    dataset = metadata["dataset"]
    if not isinstance(binary, dict) or not isinstance(fault, dict) or not isinstance(dataset, dict):
        raise ModelMetadataError("Model metadata sections must be JSON objects")
    if binary.get("task") != "normal_vs_abnormal" or binary.get("algorithm") != "RandomForestClassifier":
        raise ModelMetadataError("Binary model metadata has an unsupported task or algorithm")
    if binary.get("features") != MODEL_FEATURES:
        raise ModelMetadataError("Binary metadata features do not match the deployed feature contract")
    if binary.get("standard_threshold") != STANDARD_THRESHOLD:
        raise ModelMetadataError("Binary metadata standard threshold is inconsistent")
    if binary.get("high_sensitivity_threshold") != HIGH_SENSITIVITY_THRESHOLD:
        raise ModelMetadataError("Binary metadata high-sensitivity threshold is inconsistent")
    if fault.get("classes") != FAULT_LABELS:
        raise ModelMetadataError("Fault metadata classes do not match the deployed class contract")
    if fault.get("task") != "fault_classification" or fault.get("algorithm") != "RandomForestClassifier":
        raise ModelMetadataError("Fault model metadata has an unsupported task or algorithm")
    for key in ("rows", "segments", "timestamps", "abnormal_rows"):
        value = dataset.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelMetadataError(f"Dataset metadata {key!r} must be a positive integer")
    if not isinstance(metadata["artifact_version"], str) or not metadata["artifact_version"].strip():
        raise ModelMetadataError("artifact_version must be a nonempty string")
    return deepcopy(metadata)


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Read and validate metadata, preserving a clear internal cause on failure."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelMetadataError(f"Unable to read valid model metadata from {path.name}") from exc
    return validate_model_metadata(document)
