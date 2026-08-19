import joblib
import pandas as pd
import pytest

from src.config import FAULT_LABELS, FAULT_LEAKAGE_COLUMNS, FAULT_TARGET_COLUMN, GROUP_COLUMN, MODEL_FEATURES, TARGET_COLUMN
from src.fault_classifier import abnormal_only, build_fault_models, iter_fault_cv_folds, official_fault_split, validate_fault_features
from src.validation import DataValidationError, validate_dataset


@pytest.fixture(scope="module")
def validated_data(scada_frame) -> pd.DataFrame:
    return validate_dataset(scada_frame)[0]


def test_only_abnormal_rows_enter_fault_training(validated_data: pd.DataFrame) -> None:
    abnormal = abnormal_only(validated_data)
    assert (abnormal[TARGET_COLUMN] == 1).all()
    assert "normal" not in set(abnormal[FAULT_TARGET_COLUMN])


@pytest.mark.parametrize("leakage_column", sorted(FAULT_LEAKAGE_COLUMNS))
def test_fault_features_reject_every_leakage_column(leakage_column: str) -> None:
    with pytest.raises(DataValidationError, match="leakage"):
        validate_fault_features([*MODEL_FEATURES, leakage_column])


def test_fault_features_are_exactly_approved() -> None:
    validate_fault_features(MODEL_FEATURES)
    assert FAULT_TARGET_COLUMN not in MODEL_FEATURES


def test_official_fault_split_has_no_segment_overlap_and_all_labels(validated_data: pd.DataFrame) -> None:
    train_df, test_df, summary = official_fault_split(validated_data)
    assert set(train_df[GROUP_COLUMN]).isdisjoint(test_df[GROUP_COLUMN])
    assert summary["segments_disjoint"] is True
    assert set(train_df[FAULT_TARGET_COLUMN]) == set(FAULT_LABELS)
    assert set(test_df[FAULT_TARGET_COLUMN]) == set(FAULT_LABELS)


def test_fault_cv_folds_have_no_segment_overlap(validated_data: pd.DataFrame) -> None:
    train_df, _, _ = official_fault_split(validated_data)
    for _, train_index, validation_index in iter_fault_cv_folds(train_df):
        assert set(train_df.iloc[train_index][GROUP_COLUMN]).isdisjoint(
            train_df.iloc[validation_index][GROUP_COLUMN]
        )


def test_fault_pipeline_can_be_saved_and_reloaded(tmp_path) -> None:
    rows = []
    for index, label in enumerate(FAULT_LABELS * 2):
        rows.append({
            **{feature: float(index + 1) for feature in MODEL_FEATURES},
            FAULT_TARGET_COLUMN: label,
        })
    frame = pd.DataFrame(rows)
    model = build_fault_models()["dummy"]
    model.fit(frame[MODEL_FEATURES], frame[FAULT_TARGET_COLUMN])
    path = tmp_path / "fault_pipeline.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert list(loaded.named_steps) == ["preprocessor", "classifier"]
    assert loaded.predict(frame[MODEL_FEATURES]).shape == (len(frame),)
