"""HTTP interface. Requires the ``api`` extra: ``pip install "presetfit[api]"``."""

from presetfit.api.app import create_app
from presetfit.api.config import ApiSettings

__all__ = ["ApiSettings", "create_app"]
