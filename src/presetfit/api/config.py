"""Web app configuration, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    """Operational limits for the web app.

    Configuration lives in the environment (the "twelve-factor" convention), so
    the same build runs unchanged on a laptop, in Docker or on a server.

    Attributes:
        max_upload_bytes: Largest accepted file. ``PRESETFIT_MAX_UPLOAD_MB``.
        max_pixels: Largest accepted image. ``PRESETFIT_MAX_MEGAPIXELS``.
        jpeg_quality: Quality of returned JPEGs. ``PRESETFIT_JPEG_QUALITY``.
        log_level: Logging threshold. ``PRESETFIT_LOG_LEVEL``.
    """

    max_upload_bytes: int = 25 * 1024 * 1024
    max_pixels: int = 40_000_000
    jpeg_quality: int = 92
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiSettings:
        """Build settings from ``environ`` (defaults to the process environment)."""
        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            max_upload_bytes=_positive_int(env, "PRESETFIT_MAX_UPLOAD_MB", 25) * 1024 * 1024,
            max_pixels=_positive_int(env, "PRESETFIT_MAX_MEGAPIXELS", 40) * 1_000_000,
            jpeg_quality=min(
                100, _positive_int(env, "PRESETFIT_JPEG_QUALITY", defaults.jpeg_quality)
            ),
            log_level=env.get("PRESETFIT_LOG_LEVEL", defaults.log_level).upper(),
        )


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got '{raw}'") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value
