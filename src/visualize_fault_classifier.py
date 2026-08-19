"""Generate Model 2 confusion and feature-importance plots."""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import FAULT_IMPORTANCE_PATH, FAULT_LABELS, FAULT_MODEL_PATH, FAULT_TARGET_COLUMN, MODEL_FEATURES, PLOTS_DIR
from src.data_loader import load_dataset
from src.fault_classifier import official_fault_split
from src.validation import validate_dataset


def _save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Create absolute/normalized confusion matrices and importance plot."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, _ = load_dataset()
    frame, _ = validate_dataset(frame)
    _, test_df, _ = official_fault_split(frame)
    model = joblib.load(FAULT_MODEL_PATH)
    predictions = model.predict(test_df[MODEL_FEATURES])

    ConfusionMatrixDisplay.from_predictions(
        test_df[FAULT_TARGET_COLUMN], predictions, labels=FAULT_LABELS,
        display_labels=FAULT_LABELS, cmap="Blues", xticks_rotation=25,
    )
    plt.title("Fault Classifier Confusion Matrix")
    _save("fault_confusion_matrix.png")

    ConfusionMatrixDisplay.from_predictions(
        test_df[FAULT_TARGET_COLUMN], predictions, labels=FAULT_LABELS,
        display_labels=FAULT_LABELS, normalize="true", values_format=".2f",
        cmap="Blues", xticks_rotation=25,
    )
    plt.title("Fault Classifier Normalized Confusion Matrix")
    _save("fault_confusion_matrix_normalized.png")

    importance = pd.read_csv(FAULT_IMPORTANCE_PATH).head(8).sort_values("permutation_importance_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(
        importance["feature"], importance["permutation_importance_mean"],
        xerr=importance["permutation_importance_std"], color="#C55A11",
    )
    plt.xlabel("Decrease in held-out macro F1")
    plt.title("Fault Classifier Permutation Importance")
    _save("fault_feature_importance.png")
    print(f"Created three Model 2 plots under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
