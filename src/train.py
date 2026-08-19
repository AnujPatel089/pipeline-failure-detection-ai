"""Train and evaluate leakage-safe binary failure baselines."""
import json
from typing import Any
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.config import ARTIFACTS_DIR, COMPARISON_PATH, GROUP_COLUMN, METRICS_PATH, MODEL_FEATURES, MODEL_PATH, MODELS_DIR, RANDOM_STATE, SPLIT_SUMMARY_PATH, TARGET_COLUMN
from src.data_loader import load_dataset
from src.evaluate import comparison_frame, evaluate_model, select_best_model
from src.preprocessing import build_model_pipelines
from src.validation import assert_no_leakage, validate_dataset


def group_aware_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the 20% SGKF fold whose prevalence best matches the full dataset."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = list(splitter.split(df[MODEL_FEATURES], df[TARGET_COLUMN], groups=df[GROUP_COLUMN]))
    overall_rate = float(df[TARGET_COLUMN].mean())
    train_indices, test_indices = min(candidates, key=lambda s: abs(float(df.iloc[s[1]][TARGET_COLUMN].mean()) - overall_rate))
    train_df, test_df = df.iloc[train_indices].copy(), df.iloc[test_indices].copy()
    assert set(train_df[GROUP_COLUMN]).isdisjoint(test_df[GROUP_COLUMN]), "Segment leakage detected"
    return train_df, test_df


def _split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    def ordered(values: pd.Series) -> list[Any]:
        return sorted(values.unique().tolist(), key=lambda v: (0, int(str(v))) if str(v).isdigit() else (1, str(v)))
    train_ids, test_ids = ordered(train_df[GROUP_COLUMN]), ordered(test_df[GROUP_COLUMN])
    return {
        "train_row_count": int(len(train_df)), "test_row_count": int(len(test_df)),
        "train_segment_count": len(train_ids), "test_segment_count": len(test_ids),
        "train_segment_ids": [str(v) for v in train_ids], "test_segment_ids": [str(v) for v in test_ids],
        "train_target_distribution": {str(k): int(v) for k, v in train_df[TARGET_COLUMN].value_counts().sort_index().items()},
        "test_target_distribution": {str(k): int(v) for k, v in test_df[TARGET_COLUMN].value_counts().sort_index().items()},
        "segments_disjoint": set(train_ids).isdisjoint(test_ids),
    }


def main() -> None:
    df, dataset_path = load_dataset()
    df, validation_report = validate_dataset(df)
    assert_no_leakage(MODEL_FEATURES)
    train_df, test_df = group_aware_split(df)
    summary = _split_summary(train_df, test_df)
    X_train, y_train = train_df[MODEL_FEATURES], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET_COLUMN]
    fitted, results = {}, {}
    for name, model in build_model_pipelines().items():
        model.fit(X_train, y_train)
        fitted[name], results[name] = model, evaluate_model(model, X_test, y_test)
    comparison = comparison_frame(results)
    best_name = select_best_model(results)
    reason = (f"Selected {best_name} using failure recall first, followed by F1 and PR-AUC; "
              f"it produced {results[best_name]['false_negatives']} false negatives.")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    document = {"dataset_path": str(dataset_path.relative_to(dataset_path.parents[2])), "validation": validation_report, "split": summary, "models": results, "selected_model": best_name, "selection_reason": reason}
    METRICS_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(f"Dataset: {dataset_path}")
    print(f"Validated: {validation_report['row_count']} rows, {validation_report['unique_segments']} segments, {validation_report['duplicate_rows']} duplicate rows")
    print(f"Train segments ({summary['train_segment_count']}): {summary['train_segment_ids']}")
    print(f"Test segments ({summary['test_segment_count']}): {summary['test_segment_ids']}")
    print(f"Segment sets disjoint: {summary['segments_disjoint']}")
    print("\nHeld-out model comparison (failure recall, then F1):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for name, metrics in results.items():
        print(f"{name} confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"\n{reason}")
    print(f"Saved fitted pipeline: {MODEL_PATH}")


if __name__ == "__main__":
    main()
