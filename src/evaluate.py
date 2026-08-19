"""Model evaluation utilities."""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calculate failure-focused held-out metrics for a fitted classifier."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    return evaluate_predictions(y_test, predictions, probabilities)


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate binary metrics from already-created predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y_true, predictions)), "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)), "failure_precision": float(precision_score(y_true, predictions, zero_division=0)), "failure_recall": float(recall_score(y_true, predictions, zero_division=0)), "f1": float(f1_score(y_true, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(y_true, probabilities)), "pr_auc": float(average_precision_score(y_true, probabilities)), "false_negatives": int(fn), "false_positives": int(fp), "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def comparison_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [{"model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}} for name, metrics in results.items()]
    return pd.DataFrame(rows).sort_values(["failure_recall", "f1"], ascending=[False, False], ignore_index=True)


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    """Select by recall, then F1/AP, with false negatives as safeguard."""
    return max(results, key=lambda n: (results[n]["failure_recall"], results[n]["f1"], results[n]["pr_auc"], -results[n]["false_negatives"]))
