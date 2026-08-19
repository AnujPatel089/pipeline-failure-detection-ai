import json

import pytest

from src.model_metadata import ModelMetadataError, load_model_metadata
from src.settings import Settings


def test_settings_default_log_level() -> None:
    assert Settings.from_environment({}).log_level == "INFO"


def test_settings_environment_override_and_safe_invalid_fallback() -> None:
    assert Settings.from_environment({"LOG_LEVEL": "debug"}).log_level == "DEBUG"
    assert Settings.from_environment({"LOG_LEVEL": "not-a-level"}).log_level == "INFO"


def test_model_metadata_loads_and_matches_known_contract() -> None:
    metadata = load_model_metadata()
    assert metadata["artifact_version"] == "1.0.0"
    assert metadata["binary_model"]["standard_threshold"] == 0.50
    assert metadata["fault_classifier"]["classes"] == ["blockage", "degradation", "leak", "surge"]
    assert metadata["dataset"] == {"rows": 1000, "segments": 50, "timestamps": 17, "abnormal_rows": 306}


def test_malformed_model_metadata_fails_safely(tmp_path) -> None:
    malformed = tmp_path / "model_metadata.json"
    malformed.write_text(json.dumps({"artifact_version": "1.0.0"}))
    with pytest.raises(ModelMetadataError, match="missing sections"):
        load_model_metadata(malformed)


def test_invalid_json_metadata_fails_with_clear_internal_error(tmp_path) -> None:
    malformed = tmp_path / "model_metadata.json"
    malformed.write_text("not-json")
    with pytest.raises(ModelMetadataError, match="Unable to read valid model metadata"):
        load_model_metadata(malformed)
