"""Objectives: what each fitting stage tries to make equal between two images."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from presetfit._typing import FloatArray
from presetfit.analysis.statistics import LookStatistics
from presetfit.color.masks import BAND_NAMES
from presetfit.edits.color import HueSaturationLuminance


class Objective(ABC):
    """Scores how far the edited photo's look is from the reference's.

    An objective returns a vector of *residuals*: signed gaps, in Lab units,
    that the optimiser drives towards zero. Each objective looks only at the
    statistics its sliders can influence, which is what keeps, say, the
    saturation slider from trying to fix an exposure problem.
    """

    @abstractmethod
    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        """Return the gaps between ``current`` and ``target``.

        Args:
            current: Statistics of the photo with the trial settings applied.
            target: Statistics of the reference.
            baseline: Statistics of the photo when this stage began. Objectives
                use it for decisions that must not change mid-optimisation,
                such as which colour bands to trust.
        """

    def skipped_bands(self, baseline: LookStatistics, target: LookStatistics) -> tuple[str, ...]:
        """Name colour bands this objective ignores for lack of shared colour.

        Most objectives do not work per colour band and report none.
        """
        return ()

    def frozen_parameters(self, baseline: LookStatistics, target: LookStatistics) -> frozenset[str]:
        """Name sliders the fitter must leave untouched for lack of evidence."""
        return frozenset()


class CompositeObjective(Objective):
    """Several objectives pursued at once, each with a weight.

    Used to give a flexible stage *guard rails*: the HSL stage, for example,
    matches the colour bands while also being held to the overall tone and
    colour statistics, so it cannot improve one colour at the expense of the
    whole picture.
    """

    def __init__(self, *parts: tuple[Objective, float]) -> None:
        if not parts:
            raise ValueError("a composite objective needs at least one part")
        self._parts = parts

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        return np.concatenate(
            [weight * part.residuals(current, target, baseline) for part, weight in self._parts]
        )

    def skipped_bands(self, baseline: LookStatistics, target: LookStatistics) -> tuple[str, ...]:
        names = {name for part, _ in self._parts for name in part.skipped_bands(baseline, target)}
        return tuple(name for name in BAND_NAMES if name in names)

    def frozen_parameters(self, baseline: LookStatistics, target: LookStatistics) -> frozenset[str]:
        return frozenset().union(
            *(part.frozen_parameters(baseline, target) for part, _ in self._parts)
        )


class ToneObjective(Objective):
    """Match the lightness distribution, quantile by quantile."""

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        return current.lightness_quantiles - target.lightness_quantiles


class CastObjective(Objective):
    """Match the overall colour cast (the average colour of the image)."""

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        return current.mean_cast - target.mean_cast


class SpreadObjective(Objective):
    """Match overall colourfulness (how far colours range from the average)."""

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        return current.color_spread - target.color_spread


class SplitToneObjective(Objective):
    """Match the colour cast separately in the shadows and in the highlights.

    A zone that is nearly absent from either image (a photo with no real
    shadows, say) gives an unreliable cast, so its residuals are faded out.
    """

    def __init__(self, full_confidence_coverage: float = 0.05) -> None:
        self._full_confidence_coverage = full_confidence_coverage

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        shadow_trust = _coverage_trust(
            baseline.shadow_coverage, target.shadow_coverage, self._full_confidence_coverage
        )
        highlight_trust = _coverage_trust(
            baseline.highlight_coverage, target.highlight_coverage, self._full_confidence_coverage
        )
        return np.concatenate(
            [
                shadow_trust * (current.shadow_cast - target.shadow_cast),
                highlight_trust * (current.highlight_cast - target.highlight_cast),
            ]
        )


class HueBandObjective(Objective):
    """Match the average colour and lightness inside each colour band.

    This is the objective for the :class:`HueSaturationLuminance` edit.

    This is the objective most exposed to *content* differences: if the
    reference has no blue in it, there is nothing to learn about blues. Bands
    that cover less than ``min_coverage`` of either image are ignored, and the
    rest are trusted in proportion to how much of both images they cover.
    """

    def __init__(
        self,
        min_coverage: float = 0.03,
        full_confidence_coverage: float = 0.15,
        lightness_weight: float = 0.5,
    ) -> None:
        self._min_coverage = min_coverage
        self._full_confidence_coverage = full_confidence_coverage
        self._lightness_weight = lightness_weight

    def residuals(
        self, current: LookStatistics, target: LookStatistics, baseline: LookStatistics
    ) -> FloatArray:
        trust = self._band_trust(baseline, target)
        return np.concatenate(
            [
                trust * (current.band_a - target.band_a),
                trust * (current.band_b - target.band_b),
                trust * self._lightness_weight * (current.band_lightness - target.band_lightness),
            ]
        )

    def skipped_bands(self, baseline: LookStatistics, target: LookStatistics) -> tuple[str, ...]:
        trust = self._band_trust(baseline, target)
        return tuple(name for name, value in zip(BAND_NAMES, trust, strict=True) if value == 0)

    def frozen_parameters(self, baseline: LookStatistics, target: LookStatistics) -> frozenset[str]:
        return frozenset(
            parameter
            for band in self.skipped_bands(baseline, target)
            for parameter in HueSaturationLuminance.parameter_names(band)
        )

    def _band_trust(self, baseline: LookStatistics, target: LookStatistics) -> FloatArray:
        shared = np.minimum(baseline.band_coverage, target.band_coverage)
        trust = np.sqrt(np.clip(shared / self._full_confidence_coverage, 0.0, 1.0))
        result: FloatArray = np.where(shared >= self._min_coverage, trust, 0.0)
        return result


def _coverage_trust(coverage_a: float, coverage_b: float, full_confidence: float) -> float:
    """Return 0-1 confidence in a statistic, given how much of each image backs it."""
    shared = min(coverage_a, coverage_b)
    return float(np.sqrt(min(1.0, shared / full_confidence)))
