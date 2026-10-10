"""Shared fixtures."""

from __future__ import annotations

import pytest

from presetfit import EditPipeline, FitConfig, Pixels, PresetFitService, default_pipeline

from .helpers import make_scene


@pytest.fixture(scope="session")
def pipeline() -> EditPipeline:
    return default_pipeline()


@pytest.fixture(scope="session")
def scene() -> Pixels:
    return make_scene(seed=1)


@pytest.fixture(scope="session")
def other_scene() -> Pixels:
    return make_scene(seed=2)


@pytest.fixture(scope="session")
def service() -> PresetFitService:
    return PresetFitService.create_default(FitConfig(sample_size=4000))
