"""Colour-space conversions between sRGB, linear RGB and CIE Lab.

Three spaces are used, each for a reason:

* **sRGB** is what image files store. Values are gamma-encoded for displays.
* **Linear RGB** is proportional to light. Exposure and white balance are
  simple multiplications here, exactly as they are in a camera.
* **CIE Lab** separates lightness (``L``, 0-100) from colour (``a`` is
  green-magenta, ``b`` is blue-yellow) in a roughly perceptual way. Tonal and
  colour edits become independent of each other here.

All functions accept arrays of shape ``(..., 3)`` and preserve the float dtype.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from presetfit._typing import FloatArray


class ColorSpace(str, Enum):
    """The colour spaces a pixel array can be expressed in."""

    SRGB = "srgb"
    LINEAR_RGB = "linear_rgb"
    LAB = "lab"


# Linear sRGB -> CIE XYZ (D65), IEC 61966-2-1.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
# Dividing each row by the white point makes RGB white map to exactly (1, 1, 1),
# and therefore to Lab (100, 0, 0).
_RGB_TO_XYZ_NORMALIZED = _RGB_TO_XYZ / _RGB_TO_XYZ.sum(axis=1)[:, None]
_XYZ_NORMALIZED_TO_RGB = np.linalg.inv(_RGB_TO_XYZ_NORMALIZED)

LUMINANCE_WEIGHTS: tuple[float, float, float] = (0.2126729, 0.7151522, 0.0721750)
"""Contribution of linear R, G and B to perceived luminance."""

_SRGB_ENCODED_THRESHOLD = 0.04045
_SRGB_LINEAR_THRESHOLD = 0.0031308
_LAB_DELTA = 6.0 / 29.0


def srgb_to_linear(srgb: FloatArray) -> FloatArray:
    """Decode gamma-encoded sRGB values (0-1) to linear light."""
    low = srgb / 12.92
    high = ((np.maximum(srgb, _SRGB_ENCODED_THRESHOLD) + 0.055) / 1.055) ** 2.4
    return np.where(srgb <= _SRGB_ENCODED_THRESHOLD, low, high)


def linear_to_srgb(linear: FloatArray) -> FloatArray:
    """Encode linear light as sRGB, clipping to the displayable range 0-1."""
    clipped = np.clip(linear, 0.0, 1.0)
    low = clipped * 12.92
    high = 1.055 * np.maximum(clipped, _SRGB_LINEAR_THRESHOLD) ** (1.0 / 2.4) - 0.055
    return np.where(clipped <= _SRGB_LINEAR_THRESHOLD, low, high)


def linear_to_lab(linear: FloatArray) -> FloatArray:
    """Convert linear RGB to CIE Lab (D65 white point)."""
    matrix = _RGB_TO_XYZ_NORMALIZED.astype(linear.dtype)
    xyz = linear @ matrix.T
    f = np.where(
        xyz > _LAB_DELTA**3,
        np.cbrt(np.maximum(xyz, _LAB_DELTA**3)),
        xyz / (3.0 * _LAB_DELTA**2) + 4.0 / 29.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_linear(lab: FloatArray) -> FloatArray:
    """Convert CIE Lab to linear RGB. The result may fall outside 0-1 (out of gamut)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    xyz = np.where(f > _LAB_DELTA, f**3, 3.0 * _LAB_DELTA**2 * (f - 4.0 / 29.0))
    matrix = _XYZ_NORMALIZED_TO_RGB.astype(lab.dtype)
    result: FloatArray = xyz @ matrix.T
    return result
