"""A fitter that tunes the sliders in stages, one family at a time."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from presetfit._typing import FloatArray
from presetfit.analysis.sampling import PixelSampler
from presetfit.analysis.statistics import LookAnalyzer, LookStatistics
from presetfit.color.masks import BAND_NAMES
from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace
from presetfit.edits.base import ParamSpec
from presetfit.edits.pipeline import EditPipeline
from presetfit.edits.settings import EditSettings
from presetfit.exceptions import FittingError
from presetfit.fitting.base import FitResult, LookFitter
from presetfit.fitting.objectives import (
    CastObjective,
    CompositeObjective,
    HueBandObjective,
    Objective,
    SplitToneObjective,
    SpreadObjective,
    ToneObjective,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitStage:
    """One step of the fit: a group of edits tuned against one objective.

    Attributes:
        name: Label used in logs.
        edit_names: The edits whose sliders this stage may move. They must sit
            next to each other in the pipeline.
        objective: What the stage tries to match.
        ridge: Penalty for moving sliders away from neutral, in squared Lab
            units. Zero means "fit as hard as you can". A positive value makes
            the stage prefer no change unless the change clearly pays for
            itself, which stops flexible edits from chasing noise and picks the
            simplest answer when several slider combinations fit equally well.
        reach: Fraction (0-1] of each slider's range the fitter may use. Values
            below 1 keep a stage inside the range of plausible grading even when
            the two photos show very different subjects.
        detail: Coarse stages (``False``) run first and are repeated until they
            settle. Detail stages then run once, to explain only what the
            coarse sliders could not.
    """

    name: str
    edit_names: tuple[str, ...]
    objective: Objective
    ridge: float = 0.0
    reach: float = 1.0
    detail: bool = False

    def __post_init__(self) -> None:
        if not self.edit_names:
            raise ValueError(f"stage '{self.name}' must tune at least one edit")
        if self.ridge < 0:
            raise ValueError("ridge must not be negative")
        if not 0.0 < self.reach <= 1.0:
            raise ValueError("reach must be in (0, 1]")


@dataclass(frozen=True)
class FitConfig:
    """Tuning knobs for :class:`StagedLookFitter`.

    Attributes:
        sample_size: Number of pixels drawn from each image for fitting.
        seed: Seed for that draw, so results are reproducible.
        coarse_passes: How many times the coarse stages are repeated.
        max_iterations: Solver iteration limit per stage.
        tolerance: Solver convergence tolerance.
    """

    sample_size: int = 8_000
    seed: int = 0
    coarse_passes: int = 3
    max_iterations: int = 25
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.sample_size < 100:
            raise ValueError("sample_size must be at least 100")
        if self.coarse_passes < 1 or self.max_iterations < 1:
            raise ValueError("coarse_passes and max_iterations must be at least 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


def default_stages() -> tuple[FitStage, ...]:
    """Return the standard stages, ordered from the broadest edit to the finest.

    Broad sliders go first and soak up as much of the difference as they can;
    each later, more flexible stage only has to explain what is left. That is
    how the result stays readable ("exposure +0.4") instead of everything
    ending up in the tone curve.

    The order of the coarse stages is deliberate. Brightness changes how
    colourful a pixel is, so tone is settled before colour. Saturation changes
    the average colour but white balance barely changes colourfulness, so
    saturation is settled before white balance. Each stage therefore leaves the
    earlier ones' statistics almost untouched.
    """
    tone, cast, spread = ToneObjective(), CastObjective(), SpreadObjective()
    return (
        FitStage("exposure", ("exposure",), tone, reach=0.7),
        FitStage("contrast", ("contrast",), tone, ridge=0.1, reach=0.7),
        FitStage("highlights & shadows", ("highlights_shadows",), tone, ridge=0.25),
        FitStage("saturation", ("saturation",), spread, ridge=0.1),
        FitStage("white balance", ("white_balance",), cast, ridge=0.25, reach=0.5),
        FitStage("tone curve", ("tone_curve",), tone, ridge=2.0, detail=True),
        FitStage(
            "split toning",
            ("split_toning",),
            CompositeObjective((SplitToneObjective(), 1.0), (cast, 1.0), (spread, 1.0)),
            ridge=1.0,
            reach=0.5,
            detail=True,
        ),
        FitStage(
            "hsl",
            ("hsl",),
            CompositeObjective((HueBandObjective(), 1.0), (tone, 1.0), (cast, 2.0), (spread, 2.0)),
            ridge=10.0,
            reach=0.3,
            detail=True,
        ),
    )


class StagedLookFitter(LookFitter):
    """Fits slider values by non-linear least squares, stage by stage.

    For each stage the fitter repeatedly renders a small pixel sample with trial
    slider values, measures its look, and lets SciPy's trust-region solver
    adjust the sliders until the look matches the reference's.
    """

    def __init__(
        self,
        pipeline: EditPipeline,
        stages: Sequence[FitStage] | None = None,
        analyzer: LookAnalyzer | None = None,
        config: FitConfig | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._stages = tuple(stages) if stages is not None else default_stages()
        self._analyzer = analyzer or LookAnalyzer()
        self._config = config or FitConfig()
        self._sampler = PixelSampler(self._config.sample_size, self._config.seed)
        self._starts = {stage.name: self._first_edit_index(stage) for stage in self._stages}
        if len(self._starts) != len(self._stages):
            raise ValueError("stage names must be unique")

    def fit(self, source: Pixels, reference: Pixels) -> FitResult:
        sample = self._sampler.sample(source).to(ColorSpace.LINEAR_RGB)
        target = self._analyzer.analyze(self._sampler.sample(reference))

        settings = self._pipeline.identity_settings()
        distance_before = self._analyzer.analyze(sample).distance_to(target)
        evaluations = 0
        skipped_bands: set[str] = set()

        for stage in self._schedule():
            run = _StageRun(
                pipeline=self._pipeline,
                analyzer=self._analyzer,
                config=self._config,
                stage=stage,
                start=self._starts[stage.name],
                sample=sample,
                settings=settings,
                target=target,
            )
            settings = run.solve()
            evaluations += run.evaluations
            skipped_bands.update(stage.objective.skipped_bands(run.baseline, target))
            logger.debug(
                "stage '%s': cost %.3f -> %.3f", stage.name, run.initial_cost, run.final_cost
            )

        settings = self._pipeline.quantize(settings)
        final = self._analyzer.analyze(self._pipeline.run(sample, settings))
        return FitResult(
            settings=settings,
            distance_before=distance_before,
            distance_after=final.distance_to(target),
            skipped_bands=tuple(name for name in BAND_NAMES if name in skipped_bands),
            evaluations=evaluations,
        )

    def _schedule(self) -> list[FitStage]:
        """Return the stages in execution order: coarse ones repeated, then detail."""
        coarse = [stage for stage in self._stages if not stage.detail]
        detail = [stage for stage in self._stages if stage.detail]
        return coarse * self._config.coarse_passes + detail

    def _first_edit_index(self, stage: FitStage) -> int:
        """Return where ``stage``'s edits begin in the pipeline, checking adjacency."""
        positions = sorted(self._pipeline.index_of(name) for name in stage.edit_names)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"edits of stage '{stage.name}' must be adjacent in the pipeline")
        return positions[0]


class _StageRun:
    """Solves a single stage. Holds the state that is fixed while it runs."""

    _FINITE_DIFFERENCE_STEP = 1e-3

    def __init__(
        self,
        *,
        pipeline: EditPipeline,
        analyzer: LookAnalyzer,
        config: FitConfig,
        stage: FitStage,
        start: int,
        sample: Pixels,
        settings: EditSettings,
        target: LookStatistics,
    ) -> None:
        self._pipeline = pipeline
        self._analyzer = analyzer
        self._config = config
        self._stage = stage
        self._start = start
        self._settings = settings
        self._target = target

        # Edits before this stage do not change while it runs, so apply them once.
        self._prefix = self._pipeline.run(sample, settings, stop=start)
        self.baseline = self._measure(settings)

        frozen = stage.objective.frozen_parameters(self.baseline, target)
        self._specs: list[tuple[str, ParamSpec]] = [
            (name, spec)
            for name in stage.edit_names
            for spec in self._pipeline.edit(name).params
            if spec.name not in frozen
        ]
        # Sliders are mapped onto a shared 0-1 scale so the solver treats a
        # +/-3 EV slider and a +/-100 slider alike.
        self._low = np.array([spec.minimum for _, spec in self._specs])
        self._span = np.array([spec.maximum - spec.minimum for _, spec in self._specs])
        self._neutral = self._to_unit(np.array([spec.default for _, spec in self._specs]))
        self._lower_bound = self._neutral * (1.0 - stage.reach)
        self._upper_bound = self._neutral + (1.0 - self._neutral) * stage.reach

        self.evaluations = 0
        self.initial_cost = 0.0
        self.final_cost = 0.0

    def solve(self) -> EditSettings:
        """Optimise this stage's sliders and return the updated settings."""
        if not self._specs:
            return self._settings
        current = np.array([self._settings.for_edit(name)[spec.name] for name, spec in self._specs])
        start = np.clip(self._to_unit(current), self._lower_bound, self._upper_bound)
        self.initial_cost = _cost(self._residuals(start))
        try:
            solution = least_squares(
                self._residuals,
                start,
                bounds=(self._lower_bound, self._upper_bound),
                method="trf",
                diff_step=self._FINITE_DIFFERENCE_STEP,
                max_nfev=self._config.max_iterations,
                xtol=self._config.tolerance,
                ftol=self._config.tolerance,
                gtol=self._config.tolerance,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise FittingError(f"stage '{self._stage.name}' failed: {error}") from error

        self.final_cost = float(solution.cost)
        if not np.all(np.isfinite(solution.x)) or self.final_cost > self.initial_cost:
            # Never accept a step that made things worse.
            self.final_cost = self.initial_cost
            return self._settings
        return self._settings_with(solution.x)

    def _residuals(self, unit_values: FloatArray) -> FloatArray:
        self.evaluations += 1
        current = self._measure(self._settings_with(unit_values))
        gaps = self._stage.objective.residuals(current, self._target, self.baseline)
        if self._stage.ridge > 0:
            # The factor 2 makes a symmetric slider pushed from neutral to its
            # limit cost exactly sqrt(ridge), whatever its native range.
            penalty = np.sqrt(self._stage.ridge) * 2.0 * (unit_values - self._neutral)
            gaps = np.concatenate([gaps, penalty])
        return gaps

    def _measure(self, settings: EditSettings) -> LookStatistics:
        rendered = self._pipeline.run(self._prefix, settings, start=self._start)
        return self._analyzer.analyze(rendered)

    def _settings_with(self, unit_values: FloatArray) -> EditSettings:
        values = self._low + np.clip(unit_values, 0.0, 1.0) * self._span
        updates = {name: dict(self._settings.for_edit(name)) for name in self._stage.edit_names}
        for (name, spec), value in zip(self._specs, values, strict=True):
            updates[name][spec.name] = float(value)
        return self._settings.replace(updates)

    def _to_unit(self, values: FloatArray) -> FloatArray:
        result: FloatArray = (values - self._low) / self._span
        return result


def _cost(residuals: FloatArray) -> float:
    return 0.5 * float(residuals @ residuals)
