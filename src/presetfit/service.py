"""The application service: the one object callers need."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from presetfit.color.pixels import Pixels
from presetfit.edits import default_pipeline
from presetfit.edits.pipeline import EditPipeline
from presetfit.edits.settings import EditSettings
from presetfit.export import CubeLutExporter, PresetCodec
from presetfit.fitting.base import FitResult, LookFitter
from presetfit.fitting.staged import FitConfig, StagedLookFitter
from presetfit.rendering import Renderer
from presetfit.report import LookReport, ReportBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """What a match produces.

    Attributes:
        image: The source photo with the edits applied (sRGB).
        settings: Exactly the slider values that produced ``image``.
        report: Human-readable description of those values.
        fit: Diagnostics from the fitter.
    """

    image: Pixels
    settings: EditSettings
    report: LookReport
    fit: FitResult


class PresetFitService:
    """Facade over the fitting, rendering and export machinery.

    The command line and the web API both talk to this class and nothing below
    it, so they stay thin and cannot drift apart. Collaborators are passed in
    (dependency injection), which is what lets tests swap in fakes and lets a
    different fitter be plugged in later; :meth:`create_default` wires up the
    standard set.
    """

    def __init__(
        self,
        pipeline: EditPipeline,
        fitter: LookFitter,
        renderer: Renderer,
    ) -> None:
        self._pipeline = pipeline
        self._fitter = fitter
        self._renderer = renderer
        self._reports = ReportBuilder(pipeline)
        self._presets = PresetCodec(pipeline)

    @classmethod
    def create_default(cls, fit_config: FitConfig | None = None) -> PresetFitService:
        """Build a service with the standard pipeline and fitter."""
        pipeline = default_pipeline()
        return cls(
            pipeline=pipeline,
            fitter=StagedLookFitter(pipeline, config=fit_config),
            renderer=Renderer(pipeline),
        )

    @property
    def pipeline(self) -> EditPipeline:
        """The edit pipeline this service uses."""
        return self._pipeline

    @property
    def presets(self) -> PresetCodec:
        """Reader/writer for preset files compatible with this service."""
        return self._presets

    def match(self, source: Pixels, reference: Pixels, strength: float = 1.0) -> MatchResult:
        """Give ``source`` the look of ``reference``.

        Args:
            source: The photo to edit.
            reference: The photo whose look should be copied.
            strength: 1 applies the fitted edits in full; lower values tone
                every slider down proportionally.

        Raises:
            InvalidSettingsError: if ``strength`` is outside 0-1.
            FittingError: if no settings could be fitted.
        """
        started = time.perf_counter()
        fit = self._fitter.fit(source, reference)
        applied = fit.settings
        if strength != 1.0:
            applied = self._pipeline.quantize(self._pipeline.blend(fit.settings, strength))
        image = self._renderer.render(source, applied)
        logger.info(
            "matched %dx%d photo in %.2fs: look distance %.2f -> %.2f (%d evaluations)",
            source.data.shape[1],
            source.data.shape[0],
            time.perf_counter() - started,
            fit.distance_before,
            fit.distance_after,
            fit.evaluations,
        )
        return MatchResult(
            image=image,
            settings=applied,
            report=self._reports.build(fit, applied, strength),
            fit=fit,
        )

    def apply(self, source: Pixels, settings: EditSettings) -> Pixels:
        """Apply existing ``settings`` (for example a loaded preset) to ``source``."""
        return self._renderer.render(source, settings)

    def export_lut(self, settings: EditSettings, title: str = "PresetFit", size: int = 33) -> str:
        """Return ``settings`` baked into the text of a ``.cube`` LUT file."""
        return CubeLutExporter(self._pipeline, size).export(settings, title)
