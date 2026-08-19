"""Generate reproducible evaluation plots without requiring a notebook."""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import (
    CROSS_VALIDATION_RESULTS_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    PLOTS_DIR,
    TARGET_COLUMN,
    THRESHOLD_ANALYSIS_PATH,
)
from src.data_loader import load_dataset
from src.train import group_aware_split
from src.validation import validate_dataset


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Create six plots from the fixed official model and validation artifacts."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_dataset()
    df, _ = validate_dataset(df)
    _, test_df = group_aware_split(df)
    model = joblib.load(MODEL_PATH)
    y_test = test_df[TARGET_COLUMN]
    probabilities = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    predictions = model.predict(test_df[MODEL_FEATURES])

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Normal", "Failure"], cmap="Blues")
    plt.title("Official Held-out Confusion Matrix")
    _save("confusion_matrix.png")

    RocCurveDisplay.from_predictions(y_test, probabilities, name="Random Forest")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Official Held-out ROC Curve")
    _save("roc_curve.png")

    PrecisionRecallDisplay.from_predictions(y_test, probabilities, name="Random Forest")
    plt.title("Official Held-out Precision-Recall Curve")
    _save("precision_recall_curve.png")

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(8).sort_values("permutation_importance_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(importance["feature"], importance["permutation_importance_mean"], xerr=importance["permutation_importance_std"], color="#4472C4")
    plt.xlabel("Decrease in held-out average precision")
    plt.title("Permutation Feature Importance")
    _save("feature_importance.png")

    thresholds = pd.read_csv(THRESHOLD_ANALYSIS_PATH)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds["threshold"], thresholds["failure_recall"], marker="o", label="Recall")
    plt.plot(thresholds["threshold"], thresholds["failure_precision"], marker="o", label="Precision")
    selected = thresholds.loc[thresholds["selected"], "threshold"]
    if not selected.empty:
        plt.axvline(selected.iloc[0], color="black", linestyle="--", label=f"Selected {selected.iloc[0]:.2f}")
    plt.xlabel("Probability threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Out-of-fold Training Threshold Tradeoff")
    _save("threshold_recall_precision.png")

    cv = pd.read_csv(CROSS_VALIDATION_RESULTS_PATH)
    metrics = ["failure_precision", "failure_recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
    plt.figure(figsize=(10, 5))
    plt.boxplot([cv[metric] for metric in metrics], tick_labels=[m.replace("failure_", "") for m in metrics])
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Repeated Segment-group CV Metric Distribution")
    plt.xticks(rotation=20)
    _save("cross_validation_distribution.png")

    metrics_document = json.loads(METRICS_PATH.read_text())
    print(f"Created six plots for {metrics_document['selected_model']} under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
