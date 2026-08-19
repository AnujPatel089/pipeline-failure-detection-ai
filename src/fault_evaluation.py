"""Multiclass metrics and error analysis for the abnormal-only classifier."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FAULT_LABELS, FAULT_TARGET_COLUMN, GROUP_COLUMN, MODEL_FEATURES, TIMESTAMP_COLUMN


def evaluate_fault_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    probability_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate aggregate and per-class metrics with fixed label ordering."""
    matrix = confusion_matrix(y_true, predictions, labels=FAULT_LABELS)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=FAULT_LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, label in enumerate(FAULT_LABELS)
    }
    roc_auc: float | None = None
    roc_note: str | None = None
    if probabilities is not None and probability_labels is not None:
        try:
            order = [probability_labels.index(label) for label in FAULT_LABELS]
            roc_auc = float(roc_auc_score(y_true, probabilities[:, order], labels=FAULT_LABELS, multi_class="ovr", average="macro"))
        except (ValueError, IndexError) as exc:
            roc_note = f"Multiclass ROC-AUC unavailable: {exc}"
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, labels=FAULT_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=FAULT_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=FAULT_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=FAULT_LABELS, average="weighted", zero_division=0)),
        "multiclass_roc_auc_ovr": roc_auc,
        "roc_auc_note": roc_note,
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "classification_report": classification_report(y_true, predictions, labels=FAULT_LABELS, output_dict=True, zero_division=0),
    }


def evaluate_fault_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted probability-capable multiclass pipeline."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    labels = model.named_steps["classifier"].classes_.tolist()
    return evaluate_fault_predictions(y, predictions, probabilities, labels)


def attach_fault_error_metadata(prediction_frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach timestamp/segment/alarm only after predictions have been generated."""
    required = {"actual_event_type", "predicted_event_type", "predicted_confidence"}
    if not required.issubset(prediction_frame.columns):
        raise ValueError("Fault predictions must exist before metadata is attached")
    allowed = {TIMESTAMP_COLUMN, GROUP_COLUMN, "alarm_triggered"}
    if set(metadata.columns) - allowed:
        raise ValueError("Unexpected fault error-analysis metadata")
    return prediction_frame.join(metadata)


def fault_error_analysis(model: Any, test_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return misclassified rows and ordered actual-to-predicted confusion patterns."""
    probabilities = model.predict_proba(test_df[MODEL_FEATURES])
    predictions = model.predict(test_df[MODEL_FEATURES])
    frame = test_df[MODEL_FEATURES].copy()
    frame["actual_event_type"] = test_df[FAULT_TARGET_COLUMN].to_numpy()
    frame["predicted_event_type"] = predictions
    frame["predicted_confidence"] = probabilities.max(axis=1)
    frame = attach_fault_error_metadata(frame, test_df[[TIMESTAMP_COLUMN, GROUP_COLUMN, "alarm_triggered"]])
    errors = frame[frame["actual_event_type"] != frame["predicted_event_type"]].copy()
    columns = [TIMESTAMP_COLUMN, GROUP_COLUMN, *MODEL_FEATURES, "actual_event_type", "predicted_event_type", "predicted_confidence", "alarm_triggered"]
    errors = errors[columns].sort_values(["actual_event_type", "predicted_event_type"])
    patterns = (
        errors.groupby(["actual_event_type", "predicted_event_type"]).size().rename("count")
        .reset_index().sort_values("count", ascending=False).to_dict(orient="records")
    )
    return errors, patterns
