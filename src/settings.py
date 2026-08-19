"""Environment-backed operational settings with safe defaults."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from collections.abc import Mapping


@dataclass(frozen=True)
class Settings:
    """Runtime settings that do not alter validated model policy."""

    log_level: str = "INFO"

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if environment is None else environment
        requested = source.get("LOG_LEVEL", "INFO").strip().upper()
        valid_levels = set(logging.getLevelNamesMapping())
        return cls(log_level=requested if requested in valid_levels else "INFO")
