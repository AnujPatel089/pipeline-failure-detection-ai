"""Reusable validation for the SCADA dataset and model features."""
from typing import Any, Sequence
import numpy as np
import pandas as pd
from src.config import CATEGORICAL_FEATURES, EXPECTED_CATEGORIES, GROUP_COLUMN, LEAKAGE_COLUMNS, MODEL_FEATURES, NUMERIC_FEATURES, REQUIRED_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN


class DataValidationError(ValueError):
    """Raised when input data violates the expected contract."""


def assert_no_leakage(feature_columns: Sequence[str]) -> None:
    """Reject metadata, target, and post-event fields from predictive features."""
    leaked = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise DataValidationError(f"Leakage columns cannot be model features: {leaked}")
    if set(feature_columns) != set(MODEL_FEATURES):
        missing = sorted(set(MODEL_FEATURES) - set(feature_columns))
        extra = sorted(set(feature_columns) - set(MODEL_FEATURES))
        raise DataValidationError(f"Model features differ from approved set; missing={missing}, extra={extra}")


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schema and values, returning a timestamp-parsed copy and report."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if unexpected_columns:
        raise DataValidationError(f"Unexpected columns: {unexpected_columns}")
    validated = df.copy()
    missing = {c: int(n) for c, n in validated.isna().sum().items() if n}
    for column in NUMERIC_FEATURES:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Column {column!r} must be numeric: {exc}") from exc
        if not np.isfinite(validated[column].dropna().to_numpy(dtype=float)).all():
            raise DataValidationError(f"Column {column!r} contains infinite values")
        if (validated[column].dropna() < 0).any():
            raise DataValidationError(f"Column {column!r} contains negative values")
    try:
        validated[TARGET_COLUMN] = pd.to_numeric(validated[TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Target must be numeric binary values: {exc}") from exc
    target_values = set(validated[TARGET_COLUMN].dropna().unique())
    if validated[TARGET_COLUMN].isna().any() or target_values != {0, 1}:
        raise DataValidationError(f"Target must contain both 0 and 1; found {sorted(target_values)}")
    validated[TARGET_COLUMN] = validated[TARGET_COLUMN].astype(int)
    for column in CATEGORICAL_FEATURES:
        observed = set(validated[column].dropna().unique())
        unexpected = observed - EXPECTED_CATEGORIES[column]
        if unexpected:
            raise DataValidationError(f"Unexpected values in {column!r}: {sorted(unexpected, key=str)}")
    if validated[GROUP_COLUMN].isna().any():
        raise DataValidationError("segment_id cannot be missing")
    timestamps = pd.to_datetime(validated[TIMESTAMP_COLUMN], errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError(f"Found {int(timestamps.isna().sum())} invalid or missing timestamps")
    validated[TIMESTAMP_COLUMN] = timestamps
    assert_no_leakage(MODEL_FEATURES)
    report = {
        "row_count": int(len(validated)), "column_count": int(len(validated.columns)),
        "missing_values": missing, "duplicate_rows": int(validated.duplicated().sum()),
        "target_distribution": {str(k): int(v) for k, v in validated[TARGET_COLUMN].value_counts().sort_index().items()},
        "unique_segments": int(validated[GROUP_COLUMN].nunique()), "unique_timestamps": int(timestamps.nunique()),
        "timestamp_min": timestamps.min().isoformat(), "timestamp_max": timestamps.max().isoformat(),
        "numeric_ranges": {c: {"min": float(validated[c].min()), "max": float(validated[c].max())} for c in NUMERIC_FEATURES},
        "categorical_values": {c: sorted(validated[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
    }
    return validated, report
