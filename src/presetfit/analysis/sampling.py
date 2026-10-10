"""Drawing a representative sample of an image's pixels."""

from __future__ import annotations

import numpy as np

from presetfit._typing import FloatArray
from presetfit.color.pixels import Pixels


class PixelSampler:
    """Picks a fixed-size random subset of pixels.

    Fitting needs only the *distribution* of colours, not their positions, so a
    few thousand random pixels stand in for millions. A fixed seed makes every
    run on the same image give the same answer.
    """

    def __init__(self, size: int = 20_000, seed: int = 0) -> None:
        if size < 1:
            raise ValueError("sample size must be positive")
        self._size = size
        self._seed = seed

    def sample(self, pixels: Pixels) -> Pixels:
        """Return up to ``size`` pixels as a flat float64 ``(n, 3)`` array."""
        flat: FloatArray = pixels.data.reshape(-1, 3)
        if len(flat) > self._size:
            rng = np.random.default_rng(self._seed)
            flat = flat[rng.choice(len(flat), size=self._size, replace=False)]
        return Pixels(flat.astype(np.float64), pixels.space)
