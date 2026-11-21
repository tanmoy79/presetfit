"""A differentiable PyTorch copy of the edit pipeline.

Training needs to ask "if the sliders moved a little, how much closer would the
picture get to the right answer?". PyTorch answers that automatically
(back-propagation) for any calculation written with its operations, and runs it
on a GPU for a whole batch of photos at once. The NumPy edits in
:mod:`presetfit.edits` cannot do either, so this module re-implements the same
eight edits with PyTorch.

The two versions must give the same pictures: a model trained on this renderer
is used with the NumPy renderer in the app. ``tests/test_learning.py`` renders
random settings with both and checks they agree. The tuning constants are read
from the NumPy edit classes, so they cannot drift apart.

Pixel tensors have shape ``(batch, pixels, 3)``; an image of shape
``(batch, height, width, 3)`` can be passed to :meth:`TorchRenderer.render`
directly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from presetfit.color.masks import _CHROMA_KNEE, BAND_CENTERS_DEG, BAND_NAMES
from presetfit.color.spaces import _RGB_TO_XYZ_NORMALIZED, LUMINANCE_WEIGHTS
from presetfit.edits import (
    Contrast,
    HighlightsShadows,
    HueSaturationLuminance,
    ToneCurve,
    WhiteBalance,
    default_pipeline,
)
from presetfit.edits.pipeline import EditPipeline
from presetfit.learning.sliders import SliderCodec

_LAB_DELTA = 6.0 / 29.0
_EXPECTED_EDITS = (
    "white_balance",
    "exposure",
    "contrast",
    "highlights_shadows",
    "tone_curve",
    "saturation",
    "hsl",
    "split_toning",
)


# --- colour spaces (mirrors presetfit.color.spaces) -------------------------


def srgb_to_linear(srgb: Tensor) -> Tensor:
    """Decode sRGB (0-1) to linear light."""
    high = ((srgb.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4
    return torch.where(srgb <= 0.04045, srgb / 12.92, high)


def linear_to_srgb(linear: Tensor) -> Tensor:
    """Encode linear light as sRGB, clipping to 0-1."""
    clipped = linear.clamp(0.0, 1.0)
    high = 1.055 * clipped.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055
    return torch.where(clipped <= 0.0031308, clipped * 12.92, high)


def linear_to_lab(linear: Tensor) -> Tensor:
    """Convert linear RGB (already clipped to 0-1) to CIE Lab."""
    matrix = torch.as_tensor(_RGB_TO_XYZ_NORMALIZED, dtype=linear.dtype, device=linear.device)
    xyz = linear @ matrix.T
    threshold = _LAB_DELTA**3
    f = torch.where(
        xyz > threshold,
        xyz.clamp_min(threshold) ** (1.0 / 3.0),
        xyz / (3.0 * _LAB_DELTA**2) + 4.0 / 29.0,
    )
    fx, fy, fz = f.unbind(-1)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-1)


def lab_to_linear(lab: Tensor) -> Tensor:
    """Convert CIE Lab to linear RGB (may leave 0-1)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    f = torch.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], dim=-1)
    xyz = torch.where(f > _LAB_DELTA, f**3, 3.0 * _LAB_DELTA**2 * (f - 4.0 / 29.0))
    inverse = torch.linalg.inv(
        torch.as_tensor(_RGB_TO_XYZ_NORMALIZED, dtype=torch.float64, device=lab.device)
    ).to(lab.dtype)
    result: Tensor = xyz @ inverse.T
    return result


def srgb_to_lab(srgb: Tensor) -> Tensor:
    """Convert sRGB (0-1) to CIE Lab."""
    return linear_to_lab(srgb_to_linear(srgb).clamp(0.0, 1.0))


def lab_to_srgb(lab: Tensor) -> Tensor:
    """Convert CIE Lab to displayable sRGB (0-1)."""
    return linear_to_srgb(lab_to_linear(lab))


# --- helpers (mirror presetfit.color.masks) ---------------------------------


def _smoothstep(t: Tensor) -> Tensor:
    clipped = t.clamp(0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _hue_band_weights(hue_deg: Tensor) -> Tensor:
    """Soft membership of the eight colour bands, shape ``(..., 8)``."""
    centers = torch.tensor(BAND_CENTERS_DEG, dtype=hue_deg.dtype, device=hue_deg.device)
    count = centers.numel()
    knots = torch.cat([centers - centers[0], centers.new_tensor([360.0])])
    relative = torch.remainder(hue_deg - centers[0], 360.0)
    lower = (torch.searchsorted(knots, relative.detach().contiguous(), right=True) - 1).clamp(
        0, count - 1
    )
    fraction = (relative - knots[lower]) / (knots[lower + 1] - knots[lower])
    below = torch.nn.functional.one_hot(lower, count).to(hue_deg.dtype)
    above = torch.nn.functional.one_hot((lower + 1) % count, count).to(hue_deg.dtype)
    return (1.0 - fraction)[..., None] * below + fraction[..., None] * above


def _pchip_slopes(y: Tensor, spacing: float) -> Tensor:
    """Knot derivatives of SciPy's PCHIP interpolation, for evenly spaced knots.

    ``y`` has shape ``(batch, knots)``. Follows the Fritsch-Carlson rule that
    :class:`scipy.interpolate.PchipInterpolator` uses, which keeps the curve
    monotonic between the knots.
    """
    m = (y[:, 1:] - y[:, :-1]) / spacing
    left, right = m[:, :-1], m[:, 1:]
    same_sign = (left * right) > 0
    safe_sum = torch.where(same_sign, left + right, torch.ones_like(left))
    interior = torch.where(same_sign, 2.0 * left * right / safe_sum, torch.zeros_like(left))

    def edge(m0: Tensor, m1: Tensor) -> Tensor:
        d = (3.0 * m0 - m1) / 2.0
        d = torch.where(torch.sign(d) != torch.sign(m0), torch.zeros_like(d), d)
        overshoot = (torch.sign(m0) != torch.sign(m1)) & (d.abs() > 3.0 * m0.abs())
        return torch.where(overshoot, 3.0 * m0, d)

    first = edge(m[:, 0], m[:, 1])
    last = edge(m[:, -1], m[:, -2])
    return torch.cat([first[:, None], interior, last[:, None]], dim=1)


def _pchip_eval(y: Tensor, x: Tensor) -> Tensor:
    """Evaluate the PCHIP curve through ``y`` (knots evenly spread over 0-1) at ``x``."""
    knots = y.shape[1]
    spacing = 1.0 / (knots - 1)
    slopes = _pchip_slopes(y, spacing)
    index = (x.detach() / spacing).floor().long().clamp(0, knots - 2)
    t = x / spacing - index.to(x.dtype)
    y0, y1 = y.gather(1, index), y.gather(1, index + 1)
    d0, d1 = slopes.gather(1, index), slopes.gather(1, index + 1)
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * y0
        + (t3 - 2 * t2 + t) * spacing * d0
        + (-2 * t3 + 3 * t2) * y1
        + (t3 - t2) * spacing * d1
    )


# --- the renderer ------------------------------------------------------------


class TorchRenderer:
    """Applies slider values to a batch of photos, differentiably.

    Args:
        pipeline: The NumPy pipeline this renderer copies. It must contain the
            standard eight edits in the standard order; anything else is
            refused rather than silently rendered differently.
    """

    def __init__(self, pipeline: EditPipeline | None = None) -> None:
        pipeline = pipeline or default_pipeline()
        names = tuple(edit.name for edit in pipeline.edits)
        if names != _EXPECTED_EDITS:
            raise ValueError(f"TorchRenderer supports the edits {_EXPECTED_EDITS}, got {names}")
        self.codec = SliderCodec(pipeline)
        self._index = {slot.key: position for position, slot in enumerate(self.codec.slots)}

    def to_values(self, normalized: Tensor) -> Tensor:
        """Turn normalised outputs (-1 ... 0 ... +1) into slider values."""
        clipped = normalized.clamp(-1.0, 1.0)
        kwargs = {"dtype": normalized.dtype, "device": normalized.device}
        default = torch.as_tensor(self.codec.default, **kwargs)  # type: ignore[arg-type]
        up = torch.as_tensor(self.codec.room_up, **kwargs)  # type: ignore[arg-type]
        down = torch.as_tensor(self.codec.room_down, **kwargs)  # type: ignore[arg-type]
        return default + torch.where(clipped >= 0, clipped * up, clipped * down)

    def render(self, srgb: Tensor, values: Tensor) -> Tensor:
        """Return ``srgb`` with the slider ``values`` applied.

        Args:
            srgb: Photos in sRGB 0-1, shape ``(batch, ..., 3)``.
            values: Slider values (not normalised), shape ``(batch, sliders)``.
        """
        shape = srgb.shape
        pixels = srgb.reshape(shape[0], -1, 3)
        lab = self._lab_stage(srgb_to_linear(pixels), values)
        return lab_to_srgb(lab).reshape(shape)

    def render_lab(self, srgb: Tensor, values: Tensor) -> Tensor:
        """Like :meth:`render`, but return CIE Lab, before the final clip to sRGB."""
        shape = srgb.shape
        pixels = srgb.reshape(shape[0], -1, 3)
        lab = self._lab_stage(srgb_to_linear(pixels), values)
        # Round-trip through sRGB so out-of-gamut colours are clipped as in the app.
        return srgb_to_lab(lab_to_srgb(lab)).reshape(shape)

    def _value(self, values: Tensor, key: str) -> Tensor:
        """One slider for every photo in the batch, shaped to broadcast over pixels."""
        return values[:, self._index[key], None]

    def _lab_stage(self, linear: Tensor, values: Tensor) -> Tensor:
        v = self._value
        # White balance (linear RGB).
        warmth = v(values, "white_balance.temperature") / 100.0 * WhiteBalance._TEMPERATURE_STRENGTH
        magenta = v(values, "white_balance.tint") / 100.0 * WhiteBalance._TINT_STRENGTH
        gains = torch.stack([warmth.exp(), (-magenta).exp(), (-warmth).exp()], dim=-1)
        weights = torch.tensor(LUMINANCE_WEIGHTS, dtype=linear.dtype, device=linear.device)
        gains = gains / (gains * weights).sum(-1, keepdim=True)
        linear = linear * gains
        # Exposure (linear RGB).
        linear = linear * (2.0 ** v(values, "exposure.exposure"))[..., None]

        lab = linear_to_lab(linear.clamp(0.0, 1.0))
        lightness, a, b = lab.unbind(-1)
        x = lightness / 100.0

        # Contrast.
        strength = v(values, "contrast.contrast") / 100.0 * Contrast._MAX_STRENGTH
        x = x + strength * x * (1.0 - x) * (2.0 * x - 1.0)
        # Highlights and shadows.
        shadow_bump = HighlightsShadows._PEAK_NORMALIZER * x * (1.0 - x) ** 3
        highlight_bump = HighlightsShadows._PEAK_NORMALIZER * x**3 * (1.0 - x)
        x = x + HighlightsShadows._MAX_SHIFT * (
            v(values, "highlights_shadows.shadows") / 100.0 * shadow_bump
            + v(values, "highlights_shadows.highlights") / 100.0 * highlight_bump
        )
        # Tone curve.
        offsets = torch.cat(
            [values[:, self._index[f"tone_curve.{spec.name}"], None] for spec in ToneCurve.params],
            dim=1,
        )
        knots = torch.linspace(0.0, 1.0, offsets.shape[1], dtype=x.dtype, device=x.device)
        x = _pchip_eval(knots + offsets / 255.0, x.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        lightness = x * 100.0

        # Saturation.
        scale = 1.0 + v(values, "saturation.saturation") / 100.0
        a, b = a * scale, b * scale

        # Per-colour hue, saturation and luminance.
        lightness, a, b = self._hsl(lightness, a, b, values)

        # Split toning.
        x = lightness / 100.0
        edge_fade = 1.0 - (2.0 * x - 1.0) ** 8
        shadows = _smoothstep(1.0 - x) * edge_fade
        highlights = _smoothstep(x) * edge_fade
        a = (
            a
            + shadows * v(values, "split_toning.shadows_a")
            + highlights * v(values, "split_toning.highlights_a")
        )
        b = (
            b
            + shadows * v(values, "split_toning.shadows_b")
            + highlights * v(values, "split_toning.highlights_b")
        )
        return torch.stack([lightness, a, b], dim=-1)

    def _hsl(
        self, lightness: Tensor, a: Tensor, b: Tensor, values: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Grey pixels have no defined hue; nudge them so the gradient stays finite.
        # Their weight is zero anyway (chroma mask), so the picture is unaffected.
        grey = (a * a + b * b) < 1e-12
        a_safe = torch.where(grey, torch.full_like(a, 1e-6), a)
        b_safe = torch.where(grey, torch.zeros_like(b), b)
        hue = torch.remainder(torch.rad2deg(torch.atan2(b_safe, a_safe)), 360.0)
        chroma = torch.sqrt(a_safe * a_safe + b_safe * b_safe)
        weights = _hue_band_weights(hue) * _smoothstep(chroma / _CHROMA_KNEE)[..., None]

        def blended(kind: str) -> Tensor:
            columns = [self._index[f"hsl.{band}_{kind}"] for band in BAND_NAMES]
            per_band = values[:, columns] / 100.0
            return (weights * per_band[:, None, :]).sum(-1)

        hsl = HueSaturationLuminance
        new_hue = torch.deg2rad(hue + hsl._MAX_HUE_SHIFT_DEG * blended("hue"))
        new_chroma = chroma * (1.0 + blended("saturation"))
        new_lightness = (lightness + hsl._MAX_LUMINANCE_SHIFT * blended("luminance")).clamp(
            0.0, 100.0
        )
        return new_lightness, new_chroma * torch.cos(new_hue), new_chroma * torch.sin(new_hue)


def delta_e76(lab_a: Tensor, lab_b: Tensor) -> Tensor:
    """Per-pixel CIE76 colour difference: straight-line distance in Lab."""
    return torch.sqrt(((lab_a - lab_b) ** 2).sum(-1) + 1e-8)


__all__ = [
    "TorchRenderer",
    "delta_e76",
    "lab_to_srgb",
    "linear_to_srgb",
    "srgb_to_lab",
    "srgb_to_linear",
]
