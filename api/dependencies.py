"""Application-lifecycle model loading and dependency access."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Request

from api.errors import ModelUnavailableError
from src.prediction_service import PredictionService

LOGGER = logging.getLogger(__name__)


def load_prediction_service() -> PredictionService:
    """Load both pipelines once during application startup."""
    LOGGER.info("Loading binary and fault-classifier pipelines")
    service = PredictionService()
    LOGGER.info("Both model pipelines loaded successfully")
    return service


def get_prediction_service(request: Request) -> PredictionService:
    """Return the lifespan-owned service without loading model files."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise ModelUnavailableError("Prediction service was not initialized")
    return service


PredictionServiceDependency = Annotated[PredictionService, Depends(get_prediction_service)]
