"""A pixel array that knows which colour space it is in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presetfit._typing import FloatArray
from presetfit.color.spaces import (
    ColorSpace,
    lab_to_linear,
    linear_to_lab,
    linear_to_srgb,
    srgb_to_linear,
)


@dataclass(frozen=True, eq=False)
class Pixels:
    """Pixel values tagged with their colour space.

    Carrying the space with the data removes a whole class of bugs: nothing can
    apply a Lab operation to sRGB values by accident, because every consumer
    asks for the space it needs with :meth:`to`.

    ``data`` has shape ``(..., 3)``: a full image ``(height, width, 3)`` or a
    flat sample of pixels ``(n, 3)``. The array is treated as immutable.
    """

    data: FloatArray
    space: ColorSpace

    def __post_init__(self) -> None:
        if self.data.ndim < 1 or self.data.shape[-1] != 3:
            raise ValueError(f"pixel data must have shape (..., 3), got {self.data.shape}")
        if not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError(f"pixel data must be floating point, got {self.data.dtype}")

    def to(self, space: ColorSpace) -> Pixels:
        """Return these pixels expressed in ``space``.

        Leaving linear light for sRGB or Lab clips values to the displayable
        range 0-1. That single rule is how overexposed or out-of-gamut values
        are handled everywhere in the library.
        """
        if space is self.space:
            return self
        linear = self._as_linear()
        if space is ColorSpace.LINEAR_RGB:
            return Pixels(linear, space)
        if space is ColorSpace.SRGB:
            return Pixels(linear_to_srgb(linear), space)
        return Pixels(linear_to_lab(np.clip(linear, 0.0, 1.0)), space)

    def flattened(self) -> Pixels:
        """Return the pixels as a flat ``(n, 3)`` list, dropping the image layout."""
        return Pixels(self.data.reshape(-1, 3), self.space)

    def _as_linear(self) -> FloatArray:
        if self.space is ColorSpace.LINEAR_RGB:
            return self.data
        if self.space is ColorSpace.SRGB:
            return srgb_to_linear(self.data)
        return lab_to_linear(self.data)
