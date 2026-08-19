import pytest
from fastapi.testclient import TestClient

from api.main import app


def valid_payload() -> dict:
    return {
        "pressure": 80.0, "flow_rate": 4.0, "temperature": 32.0,
        "valve_status": 1, "pump_state": 1, "pump_speed": 1300.0,
        "compressor_state": 1, "energy_consumption": 30.0,
    }


@pytest.mark.parametrize("field", ["target", "event_type", "alarm_triggered", "unknown_field"])
def test_forbidden_and_unknown_fields_are_rejected(field: str) -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={**valid_payload(), field: 1})
        assert response.status_code == 422
        assert any(error["type"] == "extra_forbidden" for error in response.json()["detail"])


def test_missing_required_feature_is_rejected() -> None:
    payload = valid_payload()
    payload.pop("pressure")
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


@pytest.mark.parametrize("invalid_json", [
    '{"pressure":NaN,"flow_rate":4,"temperature":32,"valve_status":1,"pump_state":1,"pump_speed":1300,"compressor_state":1,"energy_consumption":30}',
    '{"pressure":Infinity,"flow_rate":4,"temperature":32,"valve_status":1,"pump_state":1,"pump_speed":1300,"compressor_state":1,"energy_consumption":30}',
    '{"pressure":-Infinity,"flow_rate":4,"temperature":32,"valve_status":1,"pump_state":1,"pump_speed":1300,"compressor_state":1,"energy_consumption":30}',
])
def test_non_finite_values_are_rejected(invalid_json: str) -> None:
    with TestClient(app) as client:
        response = client.post("/predict", content=invalid_json, headers={"content-type": "application/json"})
        assert response.status_code == 422


def test_unsupported_monitoring_mode_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={**valid_payload(), "monitoring_mode": "maximum"})
        assert response.status_code == 422


def test_empty_and_oversized_batches_are_rejected() -> None:
    with TestClient(app) as client:
        assert client.post("/predict/batch", json={"records": []}).status_code == 422
        response = client.post("/predict/batch", json={"records": [valid_payload()] * 1001})
        assert response.status_code == 422
