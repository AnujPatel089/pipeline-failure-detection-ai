"""Production-style FastAPI application for two-stage SCADA inference."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from api.dependencies import PredictionServiceDependency, load_prediction_service
from api.errors import InferenceError, register_error_handlers
from api.middleware import add_request_id
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from src.config import MODEL_FEATURES
from src.logging_config import configure_logging
from src.settings import Settings

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once and release references at shutdown."""
    settings = Settings.from_environment()
    configure_logging(settings.log_level)
    app.state.settings = settings
    LOGGER.info("Starting pipeline-failure-detection-ai API")
    try:
        app.state.prediction_service = load_prediction_service()
    except Exception:
        LOGGER.exception("Model loading failed during startup")
        app.state.prediction_service = None
    yield
    app.state.prediction_service = None
    LOGGER.info("Pipeline inference API stopped")


app = FastAPI(
    title="Pipeline Failure Detection AI API",
    description="ML inference API for SCADA-based pipeline anomaly detection and fault classification.",
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)
app.middleware("http")(add_request_id)


def _prediction_payload(request: PredictionRequest | Any, service: Any, mode: str) -> dict[str, Any]:
    """Strip metadata, invoke the shared service, then restore response metadata."""
    telemetry = {feature: getattr(request, feature) for feature in MODEL_FEATURES}
    try:
        result = service.predict(telemetry, mode)
    except ValueError as exc:
        LOGGER.warning("Prediction service rejected validated input: %s", exc)
        raise InferenceError("Validated inference request was rejected") from exc
    except Exception as exc:
        LOGGER.exception("Unexpected inference failure")
        raise InferenceError("Unexpected inference failure") from exc
    return {
        **result,
        "fault_type": result.get("fault_type"),
        "fault_confidence": result.get("fault_confidence"),
        "segment_id": request.segment_id,
        "timestamp": request.timestamp,
    }


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    """Return a concise service index."""
    return {"service": "pipeline-failure-detection-ai", "docs": "/docs", "health": "/health", "ready": "/ready"}


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Confirm that the API process is alive; do not inspect or reload models."""
    return HealthResponse(
        status="healthy",
        service="pipeline-failure-detection-ai",
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["service"])
def ready(service: PredictionServiceDependency) -> ReadinessResponse:
    """Confirm that both in-memory pipelines are initialized and ready."""
    return ReadinessResponse(
        status="ready",
        binary_model_loaded=service.binary_model is not None,
        fault_model_loaded=service.fault_model is not None,
    )


@app.get("/model-info", tags=["service"])
def model_info(service: PredictionServiceDependency) -> dict[str, Any]:
    """Describe model tasks, thresholds, classes, and approved features."""
    return service.model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest, service: PredictionServiceDependency) -> PredictionResponse:
    """Predict one validated SCADA observation through both stages as needed."""
    return PredictionResponse.model_validate(
        _prediction_payload(request, service, request.monitoring_mode.value)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionServiceDependency,
) -> BatchPredictionResponse:
    """Predict up to 1,000 observations without reloading either model."""
    mode = request.monitoring_mode.value
    LOGGER.info("Processing batch prediction", extra={"batch_count": len(request.records)})
    predictions = [
        PredictionResponse.model_validate(_prediction_payload(record, service, mode))
        for record in request.records
    ]
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)
