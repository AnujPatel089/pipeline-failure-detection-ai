import pandas as pd
import pytest
from src.config import LEAKAGE_COLUMNS, MODEL_FEATURES
from src.data_loader import discover_csv, load_dataset
from src.validation import DataValidationError, assert_no_leakage, validate_dataset


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "segment_id": [1, 2], "pressure": [80.0, 81.0], "flow_rate": [4.0, 4.1], "temperature": [31.0, 32.0], "valve_status": [1, 2], "pump_state": [1, 0], "pump_speed": [1200.0, 0.0], "compressor_state": [1, 0], "energy_consumption": [30.0, 10.0], "alarm_triggered": [0, 1], "event_type": ["normal", "leak"], "target": [0, 1]})


def test_missing_required_column_has_clear_error() -> None:
    with pytest.raises(DataValidationError, match="Missing required columns.*pressure"):
        validate_dataset(valid_frame().drop(columns="pressure"))


@pytest.mark.parametrize("leakage_column", sorted(LEAKAGE_COLUMNS))
def test_leakage_columns_are_rejected(leakage_column: str) -> None:
    with pytest.raises(DataValidationError, match="Leakage columns"):
        assert_no_leakage([*MODEL_FEATURES, leakage_column])


def test_approved_features_pass_leakage_guard() -> None:
    assert_no_leakage(MODEL_FEATURES)


def test_data_loader_discovers_and_loads_one_valid_csv(tmp_path, scada_frame) -> None:
    path = tmp_path / "fixture.csv"
    scada_frame.to_csv(path, index=False)
    (tmp_path / "._ignored.csv").write_text("macOS metadata")
    assert discover_csv(tmp_path) == path
    loaded, loaded_path = load_dataset(tmp_path)
    assert loaded_path == path
    assert loaded.shape == scada_frame.shape
    validated, report = validate_dataset(loaded)
    assert validated["timestamp"].dtype.kind == "M"
    assert report["unique_segments"] == 50


def test_data_loader_missing_csv_has_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        discover_csv(tmp_path)


def test_data_loader_rejects_multiple_matching_csv_files(tmp_path, scada_frame) -> None:
    scada_frame.to_csv(tmp_path / "first.csv", index=False)
    scada_frame.to_csv(tmp_path / "second.csv", index=False)
    with pytest.raises(RuntimeError, match="Multiple CSV files match"):
        discover_csv(tmp_path)
