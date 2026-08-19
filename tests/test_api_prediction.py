from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from api.dependencies import get_prediction_service
from api.main import app
from src.config import MODEL_FEATURES
from src.inference import predict_with_models
from src.prediction_service import PredictionService


def telemetry_payload() -> dict:
    return {
        "pressure": 80.0, "flow_rate": 4.0, "temperature": 32.0,
        "valve_status": 1, "pump_state": 1, "pump_speed": 1300.0,
        "compressor_state": 1, "energy_consumption": 30.0,
    }


def test_valid_normal_and_abnormal_predictions(model_telemetry_examples) -> None:
    with TestClient(app) as client:
        normal_payload = model_telemetry_examples["normal"]
        abnormal_payload = model_telemetry_examples["abnormal"]
        normal = client.post("/predict", json=normal_payload)
        assert normal.status_code == 200
        assert normal.json()["status"] == "normal"
        assert normal.json()["fault_type"] is None
        assert normal.json()["fault_confidence"] is None

        abnormal = client.post("/predict", json=abnormal_payload)
        assert abnormal.status_code == 200
        assert abnormal.json()["status"] == "abnormal"
        assert abnormal.json()["fault_type"] in {"blockage", "degradation", "leak", "surge"}
        assert 0 <= abnormal.json()["fault_confidence"] <= 1


def test_api_matches_shared_inference_and_preserves_metadata(model_telemetry_examples) -> None:
    with TestClient(app) as client:
        service = client.app.state.prediction_service
        payload = model_telemetry_examples["abnormal"].copy()
        payload.update({"segment_id": 43, "timestamp": "2024-01-01T00:12:00", "monitoring_mode": "standard"})
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        shared = predict_with_models(
            {feature: payload[feature] for feature in MODEL_FEATURES},
            service.binary_model, service.fault_model, failure_threshold=0.50,
        )
        assert body["status"] == shared["status"]
        assert body["failure_probability"] == shared["failure_probability"]
        assert body["fault_type"] == shared.get("fault_type")
        assert body["segment_id"] == 43
        assert body["timestamp"] == "2024-01-01T00:12:00"


def test_batch_predictions_use_one_loaded_service(model_telemetry_examples) -> None:
    with TestClient(app) as client:
        normal = model_telemetry_examples["normal"]
        abnormal = model_telemetry_examples["abnormal"]
        response = client.post("/predict/batch", json={
            "monitoring_mode": "standard", "records": [normal, abnormal, normal]
        })
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3


class BinarySpy:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.columns = []

    def predict_proba(self, frame):
        self.columns = frame.columns.tolist()
        return np.array([[1 - self.probability, self.probability]])


class FaultSpy:
    def __init__(self) -> None:
        self.calls = 0
        self.columns = []
        self.named_steps = {"classifier": SimpleNamespace(classes_=np.array(["blockage", "degradation", "leak", "surge"]))}

    def predict_proba(self, frame):
        self.calls += 1
        self.columns = frame.columns.tolist()
        return np.array([[0.1, 0.1, 0.7, 0.1]])


def test_active_threshold_controls_fault_invocation_and_metadata_is_not_predictive() -> None:
    binary, fault = BinarySpy(0.40), FaultSpy()
    service = PredictionService(binary, fault)
    app.dependency_overrides[get_prediction_service] = lambda: service
    try:
        with TestClient(app) as client:
            payload = {**telemetry_payload(), "segment_id": 99, "timestamp": "2024-01-01T00:00:00"}
            standard = client.post("/predict", json={**payload, "monitoring_mode": "standard"})
            assert standard.json()["status"] == "normal"
            assert standard.json()["active_threshold"] == 0.50
            assert fault.calls == 0

            sensitive = client.post("/predict", json={**payload, "monitoring_mode": "high_sensitivity"})
            assert sensitive.json()["status"] == "abnormal"
            assert sensitive.json()["active_threshold"] == 0.30
            assert sensitive.json()["fault_type"] == "leak"
            assert fault.calls == 1
            assert binary.columns == MODEL_FEATURES
            assert fault.columns == MODEL_FEATURES
    finally:
        app.dependency_overrides.clear()
