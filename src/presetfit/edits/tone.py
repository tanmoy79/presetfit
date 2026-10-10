"""Tonal edits: they change how bright things are, not what colour they are."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from presetfit._typing import FloatArray
from presetfit.color.spaces import ColorSpace
from presetfit.edits.base import CurvePlot, Edit, ParamSpec, ReportLine


def _with_lightness(lab: FloatArray, lightness_fraction: FloatArray) -> FloatArray:
    """Return a copy of ``lab`` whose L channel is ``lightness_fraction`` (0-1) x 100."""
    result: FloatArray = lab.copy()
    result[..., 0] = lightness_fraction * 100.0
    return result


class Exposure(Edit):
    """Brighten or darken the whole image, measured in photographic stops.

    One stop doubles the light. Working in linear RGB makes this a plain
    multiplication, the same thing a longer shutter time does in a camera.
    """

    name = "exposure"
    label = "Exposure"
    space = ColorSpace.LINEAR_RGB
    params = (ParamSpec("exposure", "Exposure", -3.0, 3.0, step=0.01, unit="EV"),)

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        result: FloatArray = data * (2.0 ** values["exposure"])
        return result


class Contrast(Edit):
    """Push tones away from (or towards) middle grey with an S-shaped curve.

    The curve keeps black at black and white at white, and its steepness is
    capped so that it can never reverse the order of two tones.
    """

    name = "contrast"
    label = "Contrast"
    space = ColorSpace.LAB
    params = (ParamSpec("contrast", "Contrast", -100.0, 100.0),)

    _MAX_STRENGTH = 0.9  # must stay below 1 for the curve to remain monotonic

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        strength = values["contrast"] / 100.0 * self._MAX_STRENGTH
        x = data[..., 0] / 100.0
        return _with_lightness(data, x + strength * x * (1.0 - x) * (2.0 * x - 1.0))


class HighlightsShadows(Edit):
    """Lift or lower the bright end and the dark end independently.

    Each slider adds a smooth bump centred on the quarter-tones (L = 25 for
    shadows, L = 75 for highlights) that fades to nothing at black and white.
    """

    name = "highlights_shadows"
    label = "Highlights & shadows"
    space = ColorSpace.LAB
    params = (
        ParamSpec("highlights", "Highlights", -100.0, 100.0),
        ParamSpec("shadows", "Shadows", -100.0, 100.0),
    )

    _PEAK_NORMALIZER = 256.0 / 27.0  # makes each bump peak at exactly 1
    _MAX_SHIFT = 0.1  # largest lightness change, as a fraction of the full range

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        x = data[..., 0] / 100.0
        shadow_bump = self._PEAK_NORMALIZER * x * (1.0 - x) ** 3
        highlight_bump = self._PEAK_NORMALIZER * x**3 * (1.0 - x)
        shift = self._MAX_SHIFT * (
            values["shadows"] / 100.0 * shadow_bump + values["highlights"] / 100.0 * highlight_bump
        )
        return _with_lightness(data, x + shift)


class ToneCurve(Edit):
    """A five-point tone curve, like the point curve in a photo editor.

    Each parameter is how far one control point's output sits above or below
    its input, in 0-255 levels. Moving the end points is what creates faded
    blacks or dulled whites. The offset limits are chosen so that the curve can
    never fold back on itself.
    """

    name = "tone_curve"
    label = "Tone curve"
    space = ColorSpace.LAB

    _INPUT_LEVELS = (0, 64, 128, 191, 255)
    _MAX_OFFSET = 30.0
    params = (
        ParamSpec("point_0", "Input 0", 0.0, _MAX_OFFSET),
        ParamSpec("point_64", "Input 64", -_MAX_OFFSET, _MAX_OFFSET),
        ParamSpec("point_128", "Input 128", -_MAX_OFFSET, _MAX_OFFSET),
        ParamSpec("point_191", "Input 191", -_MAX_OFFSET, _MAX_OFFSET),
        ParamSpec("point_255", "Input 255", -_MAX_OFFSET, 0.0),
    )

    _KNOTS = np.linspace(0.0, 1.0, len(_INPUT_LEVELS))
    _PLOT_SAMPLES = 65

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        curve = self._interpolator(values)
        x = np.clip(data[..., 0] / 100.0, 0.0, 1.0)
        mapped = np.clip(curve(x), 0.0, 1.0).astype(data.dtype, copy=False)
        return _with_lightness(data, mapped)

    def plot(self, values: Mapping[str, float]) -> CurvePlot:
        """Return the curve in 0-255 levels, sampled finely enough to draw smoothly."""
        curve = self._interpolator(values)
        inputs = np.linspace(0.0, 1.0, self._PLOT_SAMPLES)
        outputs = np.clip(curve(inputs), 0.0, 1.0)
        return CurvePlot(
            points=tuple(
                (round(float(x) * 255.0, 2), round(float(y) * 255.0, 2))
                for x, y in zip(inputs, outputs, strict=True)
            ),
            handles=tuple(
                (float(level), float(level + values[spec.name]))
                for level, spec in zip(self._INPUT_LEVELS, self.params, strict=True)
            ),
            axis_max=255.0,
        )

    def _interpolator(self, values: Mapping[str, float]) -> PchipInterpolator:
        offsets = np.array([values[spec.name] for spec in self.params]) / 255.0
        # PCHIP is a smooth interpolation that preserves monotonicity.
        return PchipInterpolator(self._KNOTS, self._KNOTS + offsets)

    def describe(self, values: Mapping[str, float]) -> list[ReportLine]:
        if self.is_identity(values):
            return [ReportLine("Curve", "linear (no change)")]
        return [
            ReportLine(f"Point {level}", f"{level} → {round(level + values[spec.name])}")
            for level, spec in zip(self._INPUT_LEVELS, self.params, strict=True)
        ]
