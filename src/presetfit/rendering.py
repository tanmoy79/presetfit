"""Full-resolution rendering."""

from __future__ import annotations

import numpy as np

from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace
from presetfit.edits.pipeline import EditPipeline
from presetfit.edits.settings import EditSettings


class Renderer:
    """Applies a pipeline to a whole image in memory-bounded strips.

    Every edit acts on each pixel independently, so an image can be processed a
    few rows at a time with an identical result. That keeps peak memory flat
    whether the photo is 1 or 50 megapixels.
    """

    def __init__(self, pipeline: EditPipeline, chunk_pixels: int = 1_000_000) -> None:
        if chunk_pixels < 1:
            raise ValueError("chunk_pixels must be positive")
        self._pipeline = pipeline
        self._chunk_pixels = chunk_pixels

    def render(self, image: Pixels, settings: EditSettings) -> Pixels:
        """Return ``image`` with ``settings`` applied, as float32 sRGB in 0-1."""
        if image.data.ndim != 3:
            raise ValueError("render expects an image of shape (height, width, 3)")
        settings = self._pipeline.validate(settings)
        height, width, _ = image.data.shape
        rows_per_chunk = max(1, self._chunk_pixels // max(1, width))
        output = np.empty(image.data.shape, dtype=np.float32)
        for top in range(0, height, rows_per_chunk):
            strip = Pixels(image.data[top : top + rows_per_chunk], image.space)
            rendered = self._pipeline.run(strip, settings).to(ColorSpace.SRGB)
            # ``to`` only clips when it converts; clip here to cover the no-edit path.
            output[top : top + rows_per_chunk] = np.clip(rendered.data, 0.0, 1.0)
        return Pixels(output, ColorSpace.SRGB)
