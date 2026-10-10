"""The fitter interface and its result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from presetfit.color.pixels import Pixels
from presetfit.edits.settings import EditSettings


@dataclass(frozen=True)
class FitResult:
    """The outcome of fitting one photo to one reference.

    Attributes:
        settings: The fitted slider values, rounded to display precision.
        distance_before: Look distance between the untouched photo and the
            reference (see :meth:`LookStatistics.distance_to`).
        distance_after: The same distance once ``settings`` are applied.
        skipped_bands: Colour bands the fitter left alone because one of the
            images did not contain enough of that colour.
        evaluations: How many trial renders the optimiser needed.
    """

    settings: EditSettings
    distance_before: float
    distance_after: float
    skipped_bands: tuple[str, ...] = ()
    evaluations: int = 0

    @property
    def improvement(self) -> float:
        """Fraction of the original look gap that was closed, from 0 to 1."""
        if self.distance_before <= 0:
            return 0.0
        return max(0.0, 1.0 - self.distance_after / self.distance_before)


class LookFitter(ABC):
    """Finds edit settings that give one photo the look of another.

    This is the seam for swapping algorithms. The service depends only on this
    interface, so a neural-network fitter could later replace the optimiser
    without changing the CLI, the API or the renderer.
    """

    @abstractmethod
    def fit(self, source: Pixels, reference: Pixels) -> FitResult:
        """Return settings that move ``source`` towards the look of ``reference``."""
