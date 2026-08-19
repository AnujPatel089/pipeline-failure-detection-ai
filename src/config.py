"""Central configuration for the binary failure detector."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
NUMERIC_FEATURES = ["pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"]
CATEGORICAL_FEATURES = ["valve_status", "pump_state", "compressor_state"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ABLATION_FEATURES = {
    "full": MODEL_FEATURES,
    "sensor_only": NUMERIC_FEATURES,
    "core_process_sensors": ["pressure", "flow_rate", "temperature"],
}
TARGET_COLUMN = "target"
GROUP_COLUMN = "segment_id"
TIMESTAMP_COLUMN = "timestamp"
LEAKAGE_COLUMNS = {"event_type", "alarm_triggered", TARGET_COLUMN, TIMESTAMP_COLUMN, GROUP_COLUMN}
REQUIRED_COLUMNS = [TIMESTAMP_COLUMN, GROUP_COLUMN, "pressure", "flow_rate", "temperature", "valve_status", "pump_state", "pump_speed", "compressor_state", "energy_consumption", "alarm_triggered", "event_type", TARGET_COLUMN]
EXPECTED_CATEGORIES = {"valve_status": {0, 1, 2}, "pump_state": {0, 1}, "compressor_state": {0, 1}}
RANDOM_STATE = 42
TEST_SIZE = 0.20
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
SPLIT_SUMMARY_PATH = ARTIFACTS_DIR / "split_summary.json"
MODEL_PATH = MODELS_DIR / "best_binary_failure_model.joblib"
CROSS_VALIDATION_RESULTS_PATH = ARTIFACTS_DIR / "cross_validation_results.csv"
CROSS_VALIDATION_SUMMARY_PATH = ARTIFACTS_DIR / "cross_validation_summary.json"
TEMPORAL_VALIDATION_PATH = ARTIFACTS_DIR / "temporal_validation.json"
FEATURE_ABLATION_PATH = ARTIFACTS_DIR / "feature_ablation.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
ERROR_ANALYSIS_PATH = ARTIFACTS_DIR / "error_analysis.csv"
ERROR_SUMMARY_PATH = ARTIFACTS_DIR / "error_analysis_summary.json"
THRESHOLD_ANALYSIS_PATH = ARTIFACTS_DIR / "threshold_analysis.csv"
VALIDATION_REPORT_PATH = ARTIFACTS_DIR / "validation_report.json"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Model 2: abnormal-only fault classification.
FAULT_TARGET_COLUMN = "event_type"
FAULT_LABELS = ["blockage", "degradation", "leak", "surge"]
FAULT_LEAKAGE_COLUMNS = {
    FAULT_TARGET_COLUMN,
    TARGET_COLUMN,
    "alarm_triggered",
    TIMESTAMP_COLUMN,
    GROUP_COLUMN,
}
FAULT_METRICS_PATH = ARTIFACTS_DIR / "fault_classifier_metrics.json"
FAULT_COMPARISON_PATH = ARTIFACTS_DIR / "fault_model_comparison.csv"
FAULT_CV_RESULTS_PATH = ARTIFACTS_DIR / "fault_classifier_cv_results.csv"
FAULT_CV_SUMMARY_PATH = ARTIFACTS_DIR / "fault_classifier_cv_summary.json"
FAULT_ERRORS_PATH = ARTIFACTS_DIR / "fault_classifier_errors.csv"
FAULT_IMPORTANCE_PATH = ARTIFACTS_DIR / "fault_feature_importance.csv"
FAULT_MODEL_PATH = MODELS_DIR / "best_fault_classifier.joblib"
