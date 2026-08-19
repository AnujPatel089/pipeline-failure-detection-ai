"""Safe API-facing exceptions and handlers."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the application has no usable loaded model service."""


class InferenceError(RuntimeError):
    """Raised when model inference cannot complete safely."""


def register_error_handlers(app: FastAPI) -> None:
    """Install structured handlers that never expose internal tracebacks."""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        LOGGER.warning("Request validation failed at %s", [error.get("loc") for error in exc.errors()])
        safe_errors = [
            {
                "type": error.get("type", "validation_error"),
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", "Invalid request value"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.exception_handler(ModelUnavailableError)
    async def unavailable_handler(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        LOGGER.error("Model service unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Model service is unavailable."})

    @app.exception_handler(InferenceError)
    async def inference_handler(_: Request, exc: InferenceError) -> JSONResponse:
        LOGGER.error("Prediction failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Prediction could not be completed."})
