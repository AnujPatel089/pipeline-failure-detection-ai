"""Robustness diagnostics for the leakage-safe Random Forest baseline."""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    ABLATION_FEATURES,
    ARTIFACTS_DIR,
    CROSS_VALIDATION_RESULTS_PATH,
    CROSS_VALIDATION_SUMMARY_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SUMMARY_PATH,
    FEATURE_ABLATION_PATH,
    FEATURE_IMPORTANCE_PATH,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPORAL_VALIDATION_PATH,
    THRESHOLD_ANALYSIS_PATH,
    TIMESTAMP_COLUMN,
    VALIDATION_REPORT_PATH,
)
from src.data_loader import load_dataset
from src.evaluate import evaluate_model, evaluate_predictions
from src.preprocessing import build_random_forest_pipeline
from src.train import group_aware_split
from src.validation import DataValidationError, validate_dataset

CV_SEEDS = (42, 43, 44)
CV_FOLDS = 5
ERROR_METADATA = [TIMESTAMP_COLUMN, GROUP_COLUMN, "event_type", "alarm_triggered"]


def validate_feature_sets(feature_sets: dict[str, list[str]]) -> None:
    """Ensure every ablation is a nonempty subset of approved model inputs."""
    approved = set(MODEL_FEATURES)
    for name, features in feature_sets.items():
        leaked = sorted(set(features) & LEAKAGE_COLUMNS)
        invalid = sorted(set(features) - approved)
        if not features or leaked or invalid:
            raise DataValidationError(
                f"Invalid ablation {name!r}; leakage={leaked}, unapproved={invalid}, empty={not features}"
            )


def iter_group_folds(
    frame: pd.DataFrame,
    seeds: Iterable[int] = CV_SEEDS,
    n_splits: int = CV_FOLDS,
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield deterministic stratified folds and assert segment isolation."""
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(frame[MODEL_FEATURES], frame[TARGET_COLUMN], frame[GROUP_COLUMN]), start=1
        ):
            train_groups = set(frame.iloc[train_index][GROUP_COLUMN])
            validation_groups = set(frame.iloc[validation_index][GROUP_COLUMN])
            assert train_groups.isdisjoint(validation_groups), "Segment overlap in cross-validation"
            yield seed, fold, train_index, validation_index


def repeated_group_cross_validation(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate only within the official training partition across 15 group folds."""
    rows = []
    for seed, fold, train_index, validation_index in iter_group_folds(train_df):
        fold_train, fold_validation = train_df.iloc[train_index], train_df.iloc[validation_index]
        model = build_random_forest_pipeline(MODEL_FEATURES)
        model.fit(fold_train[MODEL_FEATURES], fold_train[TARGET_COLUMN])
        metrics = evaluate_model(model, fold_validation[MODEL_FEATURES], fold_validation[TARGET_COLUMN])
        rows.append({
            "seed": seed,
            "fold": fold,
            "train_rows": len(fold_train),
            "validation_rows": len(fold_validation),
            "train_segments": fold_train[GROUP_COLUMN].nunique(),
            "validation_segments": fold_validation[GROUP_COLUMN].nunique(),
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
        })
    results = pd.DataFrame(rows)
    metrics_to_summarize = [
        "failure_precision", "failure_recall", "f1", "balanced_accuracy",
        "roc_auc", "pr_auc", "false_negatives", "false_positive_rate",
    ]
    summary = {
        "scope": "Official training partition only; final held-out segments were untouched.",
        "fold_count": len(results),
        "seeds": list(CV_SEEDS),
        "folds_per_seed": CV_FOLDS,
        "metrics": {
            metric: {
                "mean": float(results[metric].mean()),
                "std": float(results[metric].std(ddof=1)),
                "min": float(results[metric].min()),
                "max": float(results[metric].max()),
            }
            for metric in metrics_to_summarize
        },
    }
    return results, summary


def temporal_diagnostic(train_df: pd.DataFrame) -> dict[str, Any]:
    """Train on early timestamps and diagnose performance on later timestamps."""
    timestamps = sorted(train_df[TIMESTAMP_COLUMN].unique())
    split_at = int(len(timestamps) * 0.8)
    train_times, test_times = timestamps[:split_at], timestamps[split_at:]
    early = train_df[train_df[TIMESTAMP_COLUMN].isin(train_times)].copy()
    late = train_df[train_df[TIMESTAMP_COLUMN].isin(test_times)].copy()
    if early.empty or late.empty or early[TARGET_COLUMN].nunique() < 2 or late[TARGET_COLUMN].nunique() < 2:
        raise DataValidationError("Temporal diagnostic requires nonempty early/late sets with both classes")
    model = build_random_forest_pipeline(MODEL_FEATURES)
    model.fit(early[MODEL_FEATURES], early[TARGET_COLUMN])
    overlap = set(early[GROUP_COLUMN]) & set(late[GROUP_COLUMN])
    return {
        "diagnostic_only": True,
        "scope": "Official training partition only; final held-out segments were untouched.",
        "limitation": (
            "Pipeline segments repeat across timestamps, so this isolates time but not segment identity. "
            "Only 17 one-minute timestamps exist, which cannot establish long-term generalization."
        ),
        "train_timestamp_range": [pd.Timestamp(min(train_times)).isoformat(), pd.Timestamp(max(train_times)).isoformat()],
        "test_timestamp_range": [pd.Timestamp(min(test_times)).isoformat(), pd.Timestamp(max(test_times)).isoformat()],
        "train_rows": int(len(early)), "test_rows": int(len(late)),
        "train_class_distribution": {str(k): int(v) for k, v in early[TARGET_COLUMN].value_counts().sort_index().items()},
        "test_class_distribution": {str(k): int(v) for k, v in late[TARGET_COLUMN].value_counts().sort_index().items()},
        "overlapping_segment_count": len(overlap),
        "metrics": evaluate_model(model, late[MODEL_FEATURES], late[TARGET_COLUMN]),
    }


def feature_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit controlled feature subsets on the exact official split."""
    validate_feature_sets(ABLATION_FEATURES)
    rows, fitted = [], {}
    for name, features in ABLATION_FEATURES.items():
        model = build_random_forest_pipeline(features)
        model.fit(train_df[features], train_df[TARGET_COLUMN])
        fitted[name] = model
        metrics = evaluate_model(model, test_df[features], test_df[TARGET_COLUMN])
        rows.append({"model": name, "features": "|".join(features), **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
    return pd.DataFrame(rows), fitted


def feature_importance(model: Any, test_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate native encoded importance and raw-feature permutation importance."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out()
    native = pd.Series(classifier.feature_importances_, index=transformed_names)
    aggregated = {feature: 0.0 for feature in MODEL_FEATURES}
    for transformed_name, value in native.items():
        plain = transformed_name.split("__", 1)[-1]
        parent = next((feature for feature in MODEL_FEATURES if plain == feature or plain.startswith(f"{feature}_")), None)
        if parent:
            aggregated[parent] += float(value)
    permutation = permutation_importance(
        model, test_df[MODEL_FEATURES], test_df[TARGET_COLUMN], scoring="average_precision",
        n_repeats=20, random_state=RANDOM_STATE, n_jobs=1,
    )
    result = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "model_importance": [aggregated[feature] for feature in MODEL_FEATURES],
        "permutation_importance_mean": permutation.importances_mean,
        "permutation_importance_std": permutation.importances_std,
    })
    return result.sort_values("permutation_importance_mean", ascending=False, ignore_index=True)


def attach_error_metadata(
    prediction_frame: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach analysis-only fields after prediction columns already exist."""
    required_predictions = {"actual_target", "predicted_target", "predicted_failure_probability"}
    if not required_predictions.issubset(prediction_frame.columns):
        raise ValueError("Predictions must be created before analysis metadata is attached")
    if set(metadata.columns) - set(ERROR_METADATA):
        raise ValueError("Unexpected error-analysis metadata columns")
    return prediction_frame.join(metadata)


def error_analysis(model: Any, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create row-level and grouped analysis of official held-out mistakes."""
    probabilities = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    prediction_frame = test_df[MODEL_FEATURES].copy()
    prediction_frame["actual_target"] = test_df[TARGET_COLUMN].to_numpy()
    prediction_frame["predicted_target"] = predictions
    prediction_frame["predicted_failure_probability"] = probabilities
    attached = attach_error_metadata(prediction_frame, test_df[ERROR_METADATA])
    errors = attached[attached["actual_target"] != attached["predicted_target"]].copy()
    errors.insert(0, "error_type", np.where(errors["actual_target"] == 1, "false_negative", "false_positive"))
    output_columns = ["error_type", TIMESTAMP_COLUMN, GROUP_COLUMN, *MODEL_FEATURES, "actual_target", "predicted_target", "predicted_failure_probability", "event_type", "alarm_triggered"]
    errors = errors[output_columns].sort_values(["error_type", "predicted_failure_probability"])
    summaries = {}
    for column in ["event_type", GROUP_COLUMN, "alarm_triggered"]:
        grouped = errors.groupby(["error_type", column], dropna=False).size().rename("count").reset_index()
        summaries[column] = grouped.to_dict(orient="records")
    return errors, {"total_errors": len(errors), "false_negatives": int((errors["error_type"] == "false_negative").sum()), "false_positives": int((errors["error_type"] == "false_positive").sum()), "grouped_errors": summaries}


def out_of_fold_training_probabilities(train_df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Generate validation probabilities using only the official training partition."""
    probabilities = np.full(len(train_df), np.nan)
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fit_index, validation_index in splitter.split(train_df[MODEL_FEATURES], train_df[TARGET_COLUMN], train_df[GROUP_COLUMN]):
        assert set(train_df.iloc[fit_index][GROUP_COLUMN]).isdisjoint(train_df.iloc[validation_index][GROUP_COLUMN])
        model = build_random_forest_pipeline(MODEL_FEATURES)
        model.fit(train_df.iloc[fit_index][MODEL_FEATURES], train_df.iloc[fit_index][TARGET_COLUMN])
        probabilities[validation_index] = model.predict_proba(train_df.iloc[validation_index][MODEL_FEATURES])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Every training row must receive exactly one out-of-fold probability")
    return train_df[TARGET_COLUMN].reset_index(drop=True), probabilities


def analyze_thresholds(y_validation: pd.Series, probabilities: np.ndarray) -> tuple[pd.DataFrame, float, str]:
    """Select a threshold from validation predictions, never final test labels."""
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        predictions = (probabilities >= threshold).astype(int)
        metrics = evaluate_predictions(y_validation, predictions, probabilities)
        rows.append({"threshold": float(threshold), **{k: v for k, v in metrics.items() if k not in {"confusion_matrix", "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "false_positive_rate"}}})
    results = pd.DataFrame(rows)
    safe = results[results["failure_recall"] >= 0.95]
    if not safe.empty:
        selected = safe.sort_values(["failure_precision", "threshold"], ascending=False).iloc[0]
        reason = "Highest validation precision among thresholds with at least 95% failure recall."
    else:
        selected = results.sort_values(["failure_recall", "failure_precision", "threshold"], ascending=False).iloc[0]
        reason = "No threshold achieved 95% validation recall; selected maximum recall, then precision."
    return results, float(selected["threshold"]), reason


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    """Run all hardening diagnostics without changing baseline model selection."""
    df, _ = load_dataset()
    df, _ = validate_dataset(df)
    official_train, official_test = group_aware_split(df)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cv_results, cv_summary = repeated_group_cross_validation(official_train)
    cv_results.to_csv(CROSS_VALIDATION_RESULTS_PATH, index=False)
    CROSS_VALIDATION_SUMMARY_PATH.write_text(json.dumps(cv_summary, indent=2) + "\n")

    temporal = temporal_diagnostic(official_train)
    TEMPORAL_VALIDATION_PATH.write_text(json.dumps(temporal, indent=2, default=_json_default) + "\n")

    ablation, fitted = feature_ablation(official_train, official_test)
    ablation.to_csv(FEATURE_ABLATION_PATH, index=False)
    importance = feature_importance(fitted["full"], official_test)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    errors, error_summary = error_analysis(fitted["full"], official_test)
    errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
    ERROR_SUMMARY_PATH.write_text(json.dumps(error_summary, indent=2, default=_json_default) + "\n")

    y_oof, oof_probabilities = out_of_fold_training_probabilities(official_train)
    thresholds, selected_threshold, threshold_reason = analyze_thresholds(y_oof, oof_probabilities)
    thresholds["selected"] = thresholds["threshold"].eq(selected_threshold)
    thresholds.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)

    report = {
        "official_test_untouched_during_cv_and_threshold_selection": True,
        "selected_validation_threshold": selected_threshold,
        "threshold_selection_reason": threshold_reason,
        "default_threshold_retained_in_saved_model": True,
    }
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Repeated group CV folds: {cv_summary['fold_count']}")
    for metric in ["failure_recall", "f1", "roc_auc", "pr_auc"]:
        values = cv_summary["metrics"][metric]
        print(f"CV {metric}: {values['mean']:.4f} ± {values['std']:.4f} (min {values['min']:.4f})")
    print(f"Temporal diagnostic: {temporal['metrics']}")
    print("\nFeature ablation:\n" + ablation.to_string(index=False))
    print("\nTop feature importance:\n" + importance.head().to_string(index=False))
    print(f"\nErrors: {error_summary['false_negatives']} FN, {error_summary['false_positives']} FP")
    print(f"Validation-selected threshold: {selected_threshold:.2f}. {threshold_reason}")


if __name__ == "__main__":
    main()
