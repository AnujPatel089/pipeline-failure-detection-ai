from uuid import UUID

from fastapi.testclient import TestClient

from api.main import app
from src.model_metadata import load_model_metadata


def test_root_health_and_model_info() -> None:
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "healthy", "service": "pipeline-failure-detection-ai"}
        readiness = client.get("/ready")
        assert readiness.status_code == 200
        assert readiness.json() == {
            "status": "ready", "binary_model_loaded": True, "fault_model_loaded": True,
        }
        response = client.get("/model-info")
        assert response.status_code == 200
        info = response.json()
        assert info["binary_model"]["algorithm"] == "RandomForestClassifier"
        assert info["binary_model"]["standard_threshold"] == 0.50
        assert info["binary_model"]["high_sensitivity_threshold"] == 0.30
        assert info["fault_classifier"]["classes"] == ["blockage", "degradation", "leak", "surge"]
        assert info["artifact_version"] == "1.0.0"
        assert "model_path" not in str(info)


def test_models_are_not_reloaded_per_request(monkeypatch) -> None:
    with TestClient(app) as client:
        service = client.app.state.prediction_service
        binary_id, fault_id = id(service.binary_model), id(service.fault_model)

        def fail_load(*_args, **_kwargs):
            raise AssertionError("joblib.load must not run inside a request")

        monkeypatch.setattr("src.prediction_service.joblib.load", fail_load)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/model-info").status_code == 200
        assert id(service.binary_model) == binary_id
        assert id(service.fault_model) == fault_id


def test_request_id_is_returned_on_success_and_validation_error() -> None:
    with TestClient(app) as client:
        for response in (client.get("/health"), client.post("/predict", json={})):
            request_id = response.headers.get("X-Request-ID")
            assert request_id
            assert str(UUID(request_id)) == request_id


def test_health_is_alive_when_readiness_service_is_unavailable() -> None:
    with TestClient(app) as client:
        client.app.state.prediction_service = None
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_openapi_documentation_is_available() -> None:
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert schema.json()["info"]["title"] == "Pipeline Failure Detection AI API"


def test_model_info_uses_centralized_metadata() -> None:
    with TestClient(app) as client:
        assert client.get("/model-info").json() == load_model_metadata()
