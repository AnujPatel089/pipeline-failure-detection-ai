"""Strict request and response contracts for the inference API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MonitoringMode(str, Enum):
    """Supported binary decision modes."""

    standard = "standard"
    high_sensitivity = "high_sensitivity"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class TelemetryRecord(BaseModel):
    """One leakage-safe SCADA observation plus optional response metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pressure: FiniteFloat = Field(description="Observed pressure value; dataset units are unspecified.")
    flow_rate: FiniteFloat = Field(description="Observed flow-rate value; dataset units are unspecified.")
    temperature: FiniteFloat = Field(description="Observed temperature value; dataset units are unspecified.")
    valve_status: Literal[0, 1, 2] = Field(description="Observed valve status category.")
    pump_state: Literal[0, 1] = Field(description="Observed pump-state category.")
    pump_speed: FiniteFloat = Field(description="Observed pump-speed value; dataset units are unspecified.")
    compressor_state: Literal[0, 1] = Field(description="Observed compressor-state category.")
    energy_consumption: FiniteFloat = Field(description="Observed energy-consumption value; dataset units are unspecified.")
    segment_id: int | str | None = Field(default=None, description="Optional response metadata; never a model feature.")
    timestamp: datetime | None = Field(default=None, description="Optional response metadata; never a model feature.")


class PredictionRequest(TelemetryRecord):
    """Single prediction request."""

    monitoring_mode: MonitoringMode = Field(default=MonitoringMode.standard)


class BatchPredictionRequest(BaseModel):
    """Bounded collection of telemetry records using one monitoring mode."""

    model_config = ConfigDict(extra="forbid")
    monitoring_mode: MonitoringMode = Field(default=MonitoringMode.standard)
    records: list[TelemetryRecord] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Structured two-stage prediction."""

    status: Literal["normal", "abnormal"]
    alert_level: Literal["info", "warning", "critical"]
    failure_probability: float
    active_threshold: float
    monitoring_mode: MonitoringMode
    fault_type: Literal["blockage", "degradation", "leak", "surge"] | None = None
    fault_confidence: float | None = None
    segment_id: int | str | None = None
    timestamp: datetime | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    binary_model_loaded: bool
    fault_model_loaded: bool
