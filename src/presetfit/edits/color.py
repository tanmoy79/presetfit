"""Colour edits: white balance, saturation, per-colour HSL and split toning."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from presetfit._typing import FloatArray
from presetfit.color.masks import (
    BAND_NAMES,
    chroma_mask,
    hue_band_weights,
    hue_chroma,
    nearest_band_name,
    tonal_zone_weights,
)
from presetfit.color.spaces import LUMINANCE_WEIGHTS, ColorSpace
from presetfit.edits.base import Edit, ParamSpec, ReportLine, format_signed

_HSL_KINDS: tuple[str, ...] = ("hue", "saturation", "luminance")
_BAND_LABELS: dict[str, str] = {
    "red": "Reds",
    "orange": "Oranges",
    "yellow": "Yellows",
    "green": "Greens",
    "aqua": "Aquas",
    "blue": "Blues",
    "purple": "Purples",
    "magenta": "Magentas",
}


class WhiteBalance(Edit):
    """Shift the overall colour cast along two axes.

    Temperature trades blue against red/yellow; tint trades green against
    magenta. The channel gains are normalised so that a neutral grey keeps its
    brightness, which keeps this edit independent of exposure.
    """

    name = "white_balance"
    label = "White balance"
    space = ColorSpace.LINEAR_RGB
    params = (
        ParamSpec("temperature", "Temperature", -100.0, 100.0),
        ParamSpec("tint", "Tint", -100.0, 100.0),
    )

    _TEMPERATURE_STRENGTH = 0.35
    _TINT_STRENGTH = 0.25

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        warmth = values["temperature"] / 100.0 * self._TEMPERATURE_STRENGTH
        magenta = values["tint"] / 100.0 * self._TINT_STRENGTH
        gains = np.array([math.exp(warmth), math.exp(-magenta), math.exp(-warmth)])
        gains /= float(gains @ np.asarray(LUMINANCE_WEIGHTS))
        result: FloatArray = data * gains.astype(data.dtype)
        return result

    def describe(self, values: Mapping[str, float]) -> list[ReportLine]:
        temperature, tint = values["temperature"], values["tint"]
        return [
            ReportLine(
                "Temperature", _with_hint(temperature, positive="warmer", negative="cooler")
            ),
            ReportLine("Tint", _with_hint(tint, positive="more magenta", negative="more green")),
        ]


class Saturation(Edit):
    """Scale colourfulness uniformly. ``-100`` gives black and white."""

    name = "saturation"
    label = "Saturation"
    space = ColorSpace.LAB
    params = (ParamSpec("saturation", "Saturation", -100.0, 100.0),)

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        result: FloatArray = data.copy()
        result[..., 1:] *= 1.0 + values["saturation"] / 100.0
        return result


class HueSaturationLuminance(Edit):
    """Adjust hue, saturation and luminance separately for eight colour bands.

    Each pixel is shared between its two nearest bands, and the effect fades out
    for near-grey pixels, so the adjustment has no hard edges.
    """

    name = "hsl"
    label = "HSL (per colour)"
    space = ColorSpace.LAB

    params = tuple(
        ParamSpec(f"{band}_{kind}", f"{_BAND_LABELS[band]} {kind}", -100.0, 100.0)
        for band in BAND_NAMES
        for kind in _HSL_KINDS
    )

    _MAX_HUE_SHIFT_DEG = 30.0
    _MAX_LUMINANCE_SHIFT = 20.0

    @staticmethod
    def parameter_names(band: str) -> tuple[str, ...]:
        """Return the names of the three sliders that control ``band``."""
        return tuple(f"{band}_{kind}" for kind in _HSL_KINDS)

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        lightness = data[..., 0]
        hue, chroma = hue_chroma(data[..., 1], data[..., 2])
        weights = hue_band_weights(hue) * chroma_mask(chroma)[..., None]

        def blended(kind: str) -> FloatArray:
            per_band = np.array([values[f"{band}_{kind}"] for band in BAND_NAMES]) / 100.0
            result: FloatArray = weights @ per_band
            return result

        new_hue = np.radians(hue + self._MAX_HUE_SHIFT_DEG * blended("hue"))
        new_chroma = chroma * (1.0 + blended("saturation"))
        new_lightness = np.clip(
            lightness + self._MAX_LUMINANCE_SHIFT * blended("luminance"), 0.0, 100.0
        )
        stacked = np.stack(
            [new_lightness, new_chroma * np.cos(new_hue), new_chroma * np.sin(new_hue)], axis=-1
        )
        return stacked.astype(data.dtype, copy=False)

    def describe(self, values: Mapping[str, float]) -> list[ReportLine]:
        lines = []
        for band in BAND_NAMES:
            parts = [
                f"{kind} {format_signed(values[f'{band}_{kind}'])}"
                for kind in _HSL_KINDS
                if values[f"{band}_{kind}"] != 0
            ]
            if parts:
                lines.append(ReportLine(_BAND_LABELS[band], ", ".join(parts)))
        return lines or [ReportLine("All colours", "no change")]


class SplitToning(Edit):
    """Tint the shadows one colour and the highlights another.

    The tints are stored as Lab ``a`` (green to magenta) and ``b`` (blue to
    yellow) offsets rather than as hue and strength, because offsets vary
    smoothly through zero, which the fitter needs. :meth:`describe` converts
    them back to a hue and strength for people.
    """

    name = "split_toning"
    label = "Split toning"
    space = ColorSpace.LAB

    _MAX_TINT = 30.0
    params = (
        ParamSpec("shadows_a", "Shadows green-magenta", -_MAX_TINT, _MAX_TINT, step=0.5),
        ParamSpec("shadows_b", "Shadows blue-yellow", -_MAX_TINT, _MAX_TINT, step=0.5),
        ParamSpec("highlights_a", "Highlights green-magenta", -_MAX_TINT, _MAX_TINT, step=0.5),
        ParamSpec("highlights_b", "Highlights blue-yellow", -_MAX_TINT, _MAX_TINT, step=0.5),
    )

    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        shadows, highlights = tonal_zone_weights(data[..., 0] / 100.0)
        result: FloatArray = data.copy()
        result[..., 1] += shadows * values["shadows_a"] + highlights * values["highlights_a"]
        result[..., 2] += shadows * values["shadows_b"] + highlights * values["highlights_b"]
        return result

    def describe(self, values: Mapping[str, float]) -> list[ReportLine]:
        return [
            ReportLine("Shadows", _describe_tint(values["shadows_a"], values["shadows_b"])),
            ReportLine(
                "Highlights", _describe_tint(values["highlights_a"], values["highlights_b"])
            ),
        ]


def _with_hint(value: float, *, positive: str, negative: str) -> str:
    text = format_signed(value)
    if value > 0:
        return f"{text} ({positive})"
    if value < 0:
        return f"{text} ({negative})"
    return text


def _describe_tint(a: float, b: float) -> str:
    strength = math.hypot(a, b)
    if strength < 0.25:
        return "none"
    hue = math.degrees(math.atan2(b, a)) % 360.0
    return f"{nearest_band_name(hue)} tint, hue {hue:.0f}°, strength {strength:.1f}"
