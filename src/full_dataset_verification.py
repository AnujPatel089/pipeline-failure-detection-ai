"""Explicit local verification for the external Kaggle dataset and saved pipelines."""
from __future__ import annotations

import json

import joblib
import numpy as np

from src.config import (
    FAULT_LABELS,
    FAULT_MODEL_PATH,
    FAULT_TARGET_COLUMN,
    GROUP_COLUMN,
    MODEL_FEATURES,
    MODEL_PATH,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.data_loader import load_dataset
from src.validation import validate_dataset

EXPECTED_TARGET_COUNTS = {0: 694, 1: 306}
EXPECTED_EVENT_COUNTS = {
    "normal": 694,
    "degradation": 135,
    "leak": 65,
    "surge": 61,
    "blockage": 45,
}


class FullDatasetVerificationError(ValueError):
    """Raised when the local external dataset differs from the verified contract."""


def verify_full_dataset() -> dict[str, object]:
    """Validate the complete local CSV and confirm both pipelines accept its telemetry."""
    frame, path = load_dataset(RAW_DATA_DIR)
    frame, validation = validate_dataset(frame)
    checks = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "segments": frame[GROUP_COLUMN].nunique(),
        "timestamps": frame[TIMESTAMP_COLUMN].nunique(),
        "target_counts": frame[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "event_counts": frame[FAULT_TARGET_COLUMN].value_counts().to_dict(),
    }
    expected = {
        "rows": 1000, "columns": 13, "segments": 50, "timestamps": 17,
        "target_counts": EXPECTED_TARGET_COUNTS, "event_counts": EXPECTED_EVENT_COUNTS,
    }
    if checks != expected:
        raise FullDatasetVerificationError(
            f"Full dataset does not match the verified dimensions/distributions: {checks}"
        )

    binary_model = joblib.load(MODEL_PATH)
    fault_model = joblib.load(FAULT_MODEL_PATH)
    binary_probabilities = binary_model.predict_proba(frame[MODEL_FEATURES].head(10))[:, 1]
    abnormal = frame[frame[TARGET_COLUMN] == 1].head(10)
    fault_probabilities = fault_model.predict_proba(abnormal[MODEL_FEATURES])
    if not np.isfinite(binary_probabilities).all() or not np.isfinite(fault_probabilities).all():
        raise FullDatasetVerificationError("Saved model compatibility produced non-finite probabilities")
    if fault_model.named_steps["classifier"].classes_.tolist() != FAULT_LABELS:
        raise FullDatasetVerificationError("Saved fault-classifier classes are inconsistent")
    return {
        "status": "verified",
        "dataset_file": path.name,
        **checks,
        "missing_values": validation["missing_values"],
        "duplicate_rows": validation["duplicate_rows"],
        "binary_model_compatible": True,
        "fault_model_compatible": True,
    }


def main() -> None:
    print(json.dumps(verify_full_dataset(), indent=2))


if __name__ == "__main__":
    main()
