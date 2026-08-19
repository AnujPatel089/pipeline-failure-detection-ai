"""Request-correlation middleware for API responses and logs."""
from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.logging_config import reset_request_id, set_request_id

LOGGER = logging.getLogger(__name__)


async def add_request_id(request: Request, call_next) -> Response:
    """Generate one UUID per request and propagate it through logs and headers."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        LOGGER.exception("Unhandled request failure")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
            headers={"X-Request-ID": request_id},
        )
    finally:
        reset_request_id(token)
