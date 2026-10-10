"""Measuring the "look" of an image as a small set of numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presetfit._typing import FloatArray
from presetfit.color.masks import chroma_mask, hue_band_weights, hue_chroma, tonal_zone_weights
from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace

LIGHTNESS_QUANTILES: tuple[float, ...] = (0.01, *np.linspace(0.05, 0.95, 19).tolist(), 0.99)
"""Fractions at which the lightness distribution is sampled."""

SPREAD_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9)
"""Fractions at which the distance-from-average-colour distribution is sampled."""

_EPSILON = 1e-9


@dataclass(frozen=True, eq=False)
class LookStatistics:
    """A compact numerical description of how an image looks.

    Two images with the same statistics share the same exposure, contrast,
    colour cast and colourfulness, whatever they are pictures of. Each group of
    numbers is what one family of sliders is fitted against.

    Attributes:
        lightness_quantiles: Lab L at :data:`LIGHTNESS_QUANTILES`. Its level and
            spread capture exposure, contrast and the tone curve.
        mean_cast: Mean Lab ``(a, b)`` over the image: the overall colour cast.
            Drives white balance.
        color_spread: How far colours range from the average colour, as the
            :data:`SPREAD_QUANTILES` of that distance. Drives saturation.
            Quantiles are used rather than a mean so that one small, vivid
            object does not make a whole image count as colourful.
        shadow_cast / highlight_cast: The mean ``(a, b)`` measured separately in
            dark and bright regions. Their difference drives split toning.
        shadow_coverage / highlight_coverage: How much of the image those
            regions occupy (0-1).
        band_coverage: Share of the image in each of the eight colour bands.
        band_a / band_b / band_lightness: Mean Lab values inside each band.
            Drive the per-colour HSL sliders.
    """

    lightness_quantiles: FloatArray
    mean_cast: FloatArray
    color_spread: FloatArray
    shadow_cast: FloatArray
    highlight_cast: FloatArray
    shadow_coverage: float
    highlight_coverage: float
    band_coverage: FloatArray
    band_a: FloatArray
    band_b: FloatArray
    band_lightness: FloatArray

    def distance_to(self, other: LookStatistics) -> float:
        """Return a single "how different do these look" score, in Lab units.

        It is the root-mean-square gap across the global statistics. 0 means
        identical looks; roughly, values under 2 are hard to tell apart.
        """
        gaps = np.concatenate(
            [
                self.lightness_quantiles - other.lightness_quantiles,
                self.mean_cast - other.mean_cast,
                self.color_spread - other.color_spread,
                self.shadow_cast - other.shadow_cast,
                self.highlight_cast - other.highlight_cast,
            ]
        )
        return float(np.sqrt(np.mean(gaps**2)))


class LookAnalyzer:
    """Computes :class:`LookStatistics` for a set of pixels."""

    def analyze(self, pixels: Pixels) -> LookStatistics:
        """Measure the look of ``pixels`` (any colour space, any layout).

        Statistics describe the image as it would be *displayed*, so values
        outside the displayable range are clipped first.
        """
        displayable = pixels.to(ColorSpace.LINEAR_RGB).to(ColorSpace.LAB)
        lab = displayable.data.reshape(-1, 3).astype(np.float64, copy=False)
        lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        hue, chroma = hue_chroma(a, b)
        mean_cast = np.array([a.mean(), b.mean()])

        shadows, highlights = tonal_zone_weights(lightness / 100.0)
        bands = hue_band_weights(hue) * chroma_mask(chroma)[:, None]
        band_totals = bands.sum(axis=0) + _EPSILON

        return LookStatistics(
            lightness_quantiles=np.quantile(lightness, LIGHTNESS_QUANTILES),
            mean_cast=mean_cast,
            color_spread=np.quantile(
                np.hypot(a - mean_cast[0], b - mean_cast[1]), SPREAD_QUANTILES
            ),
            shadow_cast=_weighted_cast(a, b, shadows),
            highlight_cast=_weighted_cast(a, b, highlights),
            shadow_coverage=float(shadows.mean()),
            highlight_coverage=float(highlights.mean()),
            band_coverage=bands.mean(axis=0),
            band_a=(bands * a[:, None]).sum(axis=0) / band_totals,
            band_b=(bands * b[:, None]).sum(axis=0) / band_totals,
            band_lightness=(bands * lightness[:, None]).sum(axis=0) / band_totals,
        )


def _weighted_cast(a: FloatArray, b: FloatArray, weights: FloatArray) -> FloatArray:
    total = float(weights.sum()) + _EPSILON
    return np.array([float((weights * a).sum()) / total, float((weights * b).sum()) / total])
