"""Soft selection masks shared by the edits and the look analysis.

The same masks decide both *where an edit acts* (for example "the oranges") and
*where a statistic is measured*. Keeping them in one place guarantees the fitter
measures exactly the region each slider controls.
"""

from __future__ import annotations

import numpy as np

from presetfit._typing import FloatArray

BAND_NAMES: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "aqua",
    "blue",
    "purple",
    "magenta",
)
"""The eight colour bands, in hue order, matching a Lightroom-style HSL panel."""

# Centre of each band as a CIE Lab hue angle in degrees. Lab hue is not evenly
# spaced by colour name, so these are placed where sRGB primaries actually land.
BAND_CENTERS_DEG: tuple[float, ...] = (40.0, 65.0, 100.0, 140.0, 195.0, 270.0, 310.0, 340.0)

_CHROMA_KNEE = 12.0


def smoothstep(t: FloatArray) -> FloatArray:
    """Smoothly ramp from 0 at ``t <= 0`` to 1 at ``t >= 1``."""
    clipped = np.clip(t, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def hue_chroma(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert Lab ``a``/``b`` to hue in degrees (0-360) and chroma (colourfulness)."""
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    chroma = np.hypot(a, b)
    return hue, chroma


def hue_band_weights(hue_deg: FloatArray) -> FloatArray:
    """Return each pixel's membership of the eight colour bands.

    The result has shape ``(..., 8)`` and sums to 1 along the last axis. A hue
    exactly at a band centre belongs wholly to that band; hues in between are
    shared linearly by the two neighbouring bands, so there are no hard edges.
    """
    centers = np.asarray(BAND_CENTERS_DEG)
    count = len(centers)
    # Positions of the centres measured from the first one, closed into a loop.
    knots = np.append(centers - centers[0], 360.0)
    relative = ((np.asarray(hue_deg, dtype=np.float64) - centers[0]) % 360.0).ravel()
    lower = np.clip(np.searchsorted(knots, relative, side="right") - 1, 0, count - 1)
    fraction = (relative - knots[lower]) / (knots[lower + 1] - knots[lower])
    rows = np.arange(relative.size)
    weights = np.zeros((relative.size, count))
    weights[rows, lower] = 1.0 - fraction
    weights[rows, (lower + 1) % count] = fraction  # the last band wraps round to the first
    return weights.reshape(*np.shape(hue_deg), count)


def chroma_mask(chroma: FloatArray) -> FloatArray:
    """Fade per-colour edits out for near-grey pixels, whose hue is unreliable."""
    return smoothstep(chroma / _CHROMA_KNEE)


def tonal_zone_weights(lightness_fraction: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ``(shadows, highlights)`` weights for lightness given as 0-1.

    Both fade to zero at pure black and pure white, where a tint cannot be
    displayed anyway.
    """
    edge_fade = 1.0 - (2.0 * lightness_fraction - 1.0) ** 8
    shadows = smoothstep(1.0 - lightness_fraction) * edge_fade
    highlights = smoothstep(lightness_fraction) * edge_fade
    return shadows, highlights


def nearest_band_name(hue_deg: float) -> str:
    """Name the colour band whose centre is closest to ``hue_deg``."""
    centers = np.asarray(BAND_CENTERS_DEG)
    distance = np.abs((centers - hue_deg + 180.0) % 360.0 - 180.0)
    return BAND_NAMES[int(np.argmin(distance))]
