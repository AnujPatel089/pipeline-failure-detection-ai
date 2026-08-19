import inspect

import numpy as np
import pandas as pd
import pytest

from src.config import ABLATION_FEATURES, GROUP_COLUMN, LEAKAGE_COLUMNS, MODEL_FEATURES
from src.model_validation import analyze_thresholds, attach_error_metadata, iter_group_folds, validate_feature_sets
from src.validation import DataValidationError


def grouped_frame() -> pd.DataFrame:
    rows = []
    for segment in range(20):
        for row in range(5):
            rows.append({GROUP_COLUMN: segment, "target": int(row < 2), **{feature: 1.0 for feature in MODEL_FEATURES}})
    return pd.DataFrame(rows)


def test_cv_folds_never_overlap_groups() -> None:
    frame = grouped_frame()
    for _, _, train_index, validation_index in iter_group_folds(frame, seeds=[42], n_splits=5):
        assert set(frame.iloc[train_index][GROUP_COLUMN]).isdisjoint(frame.iloc[validation_index][GROUP_COLUMN])


def test_ablation_sets_are_approved_and_leakage_free() -> None:
    validate_feature_sets(ABLATION_FEATURES)
    for features in ABLATION_FEATURES.values():
        assert set(features) <= set(MODEL_FEATURES)
        assert set(features).isdisjoint(LEAKAGE_COLUMNS)


def test_invalid_ablation_rejects_leakage() -> None:
    with pytest.raises(DataValidationError, match="leakage"):
        validate_feature_sets({"bad": [*MODEL_FEATURES, "alarm_triggered"]})


def test_threshold_selector_accepts_validation_inputs_only() -> None:
    assert list(inspect.signature(analyze_thresholds).parameters) == ["y_validation", "probabilities"]
    results, threshold, _ = analyze_thresholds(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9]))
    assert threshold in results["threshold"].values


def test_error_metadata_requires_predictions_first() -> None:
    metadata = pd.DataFrame({"event_type": ["leak"], "alarm_triggered": [1]})
    with pytest.raises(ValueError, match="Predictions must be created"):
        attach_error_metadata(pd.DataFrame({"pressure": [70.0]}), metadata)
    predictions = pd.DataFrame({"actual_target": [1], "predicted_target": [0], "predicted_failure_probability": [0.4]})
    attached = attach_error_metadata(predictions, metadata)
    assert "event_type" not in predictions.columns
    assert attached.loc[0, "event_type"] == "leak"
