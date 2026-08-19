"""Train and validate Model 2: the abnormal-only multiclass fault classifier."""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACTS_DIR,
    FAULT_COMPARISON_PATH,
    FAULT_CV_RESULTS_PATH,
    FAULT_CV_SUMMARY_PATH,
    FAULT_ERRORS_PATH,
    FAULT_IMPORTANCE_PATH,
    FAULT_LABELS,
    FAULT_LEAKAGE_COLUMNS,
    FAULT_METRICS_PATH,
    FAULT_MODEL_PATH,
    FAULT_TARGET_COLUMN,
    GROUP_COLUMN,
    MODEL_FEATURES,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loader import load_dataset
from src.fault_evaluation import evaluate_fault_model, fault_error_analysis
from src.preprocessing import build_preprocessor
from src.train import group_aware_split
from src.validation import DataValidationError, validate_dataset


def validate_fault_features(features: list[str]) -> None:
    """Enforce the exact approved telemetry inputs for Model 2."""
    leaked = sorted(set(features) & FAULT_LEAKAGE_COLUMNS)
    missing = sorted(set(MODEL_FEATURES) - set(features))
    extra = sorted(set(features) - set(MODEL_FEATURES))
    if leaked or missing or extra:
        raise DataValidationError(
            f"Invalid fault features; leakage={leaked}, missing={missing}, extra={extra}"
        )


def abnormal_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Select failure rows and reject normal/unknown event labels."""
    abnormal = frame.loc[frame[TARGET_COLUMN] == 1].copy()
    observed = set(abnormal[FAULT_TARGET_COLUMN].unique())
    expected = set(FAULT_LABELS)
    if observed != expected:
        raise DataValidationError(
            f"Abnormal fault labels must be exactly {FAULT_LABELS}; found {sorted(observed)}"
        )
    if "normal" in observed or not (abnormal[TARGET_COLUMN] == 1).all():
        raise DataValidationError("Only target==1 non-normal rows may enter Model 2")
    return abnormal


def official_fault_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reuse Model 1's official segment split, then filter both sides to failures."""
    binary_train, binary_test = group_aware_split(frame)
    train_df, test_df = abnormal_only(binary_train), abnormal_only(binary_test)
    overlap = set(train_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])
    if overlap:
        raise DataValidationError(f"Fault train/test segment overlap: {sorted(overlap)}")
    distributions = {}
    for name, partition in (("train", train_df), ("test", test_df)):
        counts = partition[FAULT_TARGET_COLUMN].value_counts().reindex(FAULT_LABELS, fill_value=0)
        if (counts < 2).any():
            raise DataValidationError(
                f"Official {name} partition has insufficient class coverage: {counts.to_dict()}. "
                "Construct a documented group-aware replacement for Model 2 only."
            )
        distributions[name] = {label: int(counts[label]) for label in FAULT_LABELS}
    summary = {
        "partition_source": "Filtered Model 1 official group-aware partition; no replacement required.",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "train_segments": train_df[GROUP_COLUMN].nunique(), "test_segments": test_df[GROUP_COLUMN].nunique(),
        "train_class_distribution": distributions["train"], "test_class_distribution": distributions["test"],
        "segments_disjoint": not overlap,
    }
    return train_df, test_df, summary


def build_fault_models() -> dict[str, Pipeline]:
    """Build fresh multiclass baseline pipelines."""
    return {
        "dummy": Pipeline([
            ("preprocessor", build_preprocessor(False, MODEL_FEATURES)),
            ("classifier", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(True, MODEL_FEATURES)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(False, MODEL_FEATURES)),
            ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


def iter_fault_cv_folds(frame: pd.DataFrame, n_splits: int = 5):
    """Yield class-aware segment folds with explicit overlap checks."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(frame[MODEL_FEATURES], frame[FAULT_TARGET_COLUMN], frame[GROUP_COLUMN]), start=1
    ):
        train_groups = set(frame.iloc[train_index][GROUP_COLUMN])
        validation_groups = set(frame.iloc[validation_index][GROUP_COLUMN])
        assert train_groups.isdisjoint(validation_groups), "Segment overlap in fault CV"
        yield fold, train_index, validation_index


def group_cross_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate Random Forest across safe folds of the official training partition."""
    rows = []
    for fold, train_index, validation_index in iter_fault_cv_folds(train_df):
        fit_df, validation_df = train_df.iloc[train_index], train_df.iloc[validation_index]
        model = build_fault_models()["random_forest"]
        model.fit(fit_df[MODEL_FEATURES], fit_df[FAULT_TARGET_COLUMN])
        metrics = evaluate_fault_model(model, validation_df[MODEL_FEATURES], validation_df[FAULT_TARGET_COLUMN])
        recalls = {label: metrics["per_class"][label]["recall"] for label in FAULT_LABELS}
        rows.append({
            "fold": fold, "train_rows": len(fit_df), "validation_rows": len(validation_df),
            "train_segments": fit_df[GROUP_COLUMN].nunique(), "validation_segments": validation_df[GROUP_COLUMN].nunique(),
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_recall": metrics["macro_recall"], "weakest_class_recall": min(recalls.values()),
            **{f"{label}_recall": recalls[label] for label in FAULT_LABELS},
        })
    results = pd.DataFrame(rows)
    summary = {
        "scope": "Official abnormal training partition only; final fault test segments untouched.",
        "fold_count": len(results),
        "metrics": {
            metric: {"mean": float(results[metric].mean()), "std": float(results[metric].std(ddof=1)), "min": float(results[metric].min()), "max": float(results[metric].max())}
            for metric in ["macro_f1", "balanced_accuracy", "macro_recall", "weakest_class_recall"]
        },
    }
    return results, summary


def fault_feature_importance(model: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-hot native importance and calculate raw-feature permutation importance."""
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    native = pd.Series(model.named_steps["classifier"].feature_importances_, index=transformed_names)
    aggregated = {feature: 0.0 for feature in MODEL_FEATURES}
    for transformed_name, value in native.items():
        plain = transformed_name.split("__", 1)[-1]
        parent = next((feature for feature in MODEL_FEATURES if plain == feature or plain.startswith(f"{feature}_")), None)
        if parent:
            aggregated[parent] += float(value)
    permutation = permutation_importance(
        model, test_df[MODEL_FEATURES], test_df[FAULT_TARGET_COLUMN], scoring="f1_macro",
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": MODEL_FEATURES,
        "model_importance": [aggregated[feature] for feature in MODEL_FEATURES],
        "permutation_importance_mean": permutation.importances_mean,
        "permutation_importance_std": permutation.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix", "normalized_confusion_matrix", "classification_report", "roc_auc_note"}}


def main() -> None:
    """Train, compare, validate, interpret, and save Model 2."""
    validate_fault_features(MODEL_FEATURES)
    frame, dataset_path = load_dataset()
    frame, _ = validate_dataset(frame)
    train_df, test_df, split_summary = official_fault_split(frame)
    fitted, evaluations = {}, {}
    for name, model in build_fault_models().items():
        model.fit(train_df[MODEL_FEATURES], train_df[FAULT_TARGET_COLUMN])
        fitted[name] = model
        evaluations[name] = evaluate_fault_model(model, test_df[MODEL_FEATURES], test_df[FAULT_TARGET_COLUMN])
    selected_name = max(evaluations, key=lambda name: (
        evaluations[name]["macro_f1"], evaluations[name]["balanced_accuracy"],
        evaluations[name]["multiclass_roc_auc_ovr"] or -1.0, evaluations[name]["weighted_f1"],
    ))
    selected = fitted[selected_name]

    comparison = pd.DataFrame([{"model": name, **_compact_metrics(metrics)} for name, metrics in evaluations.items()]).sort_values("macro_f1", ascending=False)
    cv_results, cv_summary = group_cross_validation(train_df)
    errors, confusion_patterns = fault_error_analysis(selected, test_df)
    importance_model_name = selected_name if selected_name == "random_forest" else "random_forest"
    importance = fault_feature_importance(fitted[importance_model_name], test_df)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(FAULT_COMPARISON_PATH, index=False)
    cv_results.to_csv(FAULT_CV_RESULTS_PATH, index=False)
    FAULT_CV_SUMMARY_PATH.write_text(json.dumps(cv_summary, indent=2) + "\n")
    errors.to_csv(FAULT_ERRORS_PATH, index=False)
    importance.to_csv(FAULT_IMPORTANCE_PATH, index=False)
    metrics_document = {
        "dataset_path": str(dataset_path.relative_to(dataset_path.parents[2])),
        "split": split_summary, "labels": FAULT_LABELS, "models": evaluations,
        "selected_model": selected_name,
        "selection_reason": "Highest held-out macro F1, then balanced accuracy, multiclass ROC-AUC, and weighted F1.",
        "feature_importance_model": importance_model_name,
        "misclassification_patterns": confusion_patterns,
    }
    FAULT_METRICS_PATH.write_text(json.dumps(metrics_document, indent=2) + "\n")
    joblib.dump(selected, FAULT_MODEL_PATH)

    print(f"Fault split: {split_summary}")
    print("\nModel comparison:\n" + comparison.to_string(index=False))
    print(f"\nSelected {selected_name}: {metrics_document['selection_reason']}")
    print(f"Confusion matrix {FAULT_LABELS}: {evaluations[selected_name]['confusion_matrix']}")
    print("Per-class metrics:", evaluations[selected_name]["per_class"])
    cv_macro = cv_summary["metrics"]["macro_f1"]
    print(f"CV macro F1: {cv_macro['mean']:.4f} ± {cv_macro['std']:.4f}; minimum {cv_macro['min']:.4f}")
    print("Top importance:\n" + importance.head().to_string(index=False))
    print("Misclassification patterns:", confusion_patterns)
    print(f"Saved pipeline: {FAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
