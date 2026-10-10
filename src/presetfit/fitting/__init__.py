"""Fitting: finding the slider values that reproduce a reference look."""

from presetfit.fitting.base import FitResult, LookFitter
from presetfit.fitting.staged import FitConfig, FitStage, StagedLookFitter, default_stages

__all__ = [
    "FitConfig",
    "FitResult",
    "FitStage",
    "LookFitter",
    "StagedLookFitter",
    "default_stages",
]
