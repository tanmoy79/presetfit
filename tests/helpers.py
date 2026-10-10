"""Helpers shared by the tests."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from presetfit import ColorSpace, EditPipeline, EditSettings, Pixels


def make_scene(seed: int = 0, height: int = 96, width: int = 128) -> Pixels:
    """Build a synthetic photo: a soft gradient, grey areas and coloured blobs.

    It has what the fitter relies on in a real photo: a full tonal range,
    neutral regions and several distinct colours.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    ramp = np.linspace(0.08, 0.85, width)[None, :, None]
    image = np.repeat(ramp, height, axis=0) * np.array([1.0, 0.97, 0.93])
    for _ in range(10):
        centre_x, centre_y = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(8, 28)
        colour = 0.5 + rng.uniform(-0.35, 0.35, 3)
        mask = np.exp(-((cols - centre_x) ** 2 + (rows - centre_y) ** 2) / (2 * radius**2))
        image = image * (1 - mask[..., None]) + colour * mask[..., None]
    image += rng.normal(0, 0.008, image.shape)
    return Pixels(np.clip(image, 0, 1).astype(np.float32), ColorSpace.SRGB)


def encode(image: Pixels, image_format: str = "PNG") -> bytes:
    """Encode pixels as image-file bytes."""
    eight_bit = np.rint(image.to(ColorSpace.SRGB).data * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(eight_bit).save(buffer, format=image_format)
    return buffer.getvalue()


def lab_difference(first: Pixels, second: Pixels) -> float:
    """Mean colour difference (Delta E 1976) between two images."""
    gap = first.to(ColorSpace.LAB).data - second.to(ColorSpace.LAB).data
    return float(np.sqrt((gap.astype(np.float64) ** 2).sum(axis=-1)).mean())


def settings_with(pipeline: EditPipeline, **edits: dict[str, float]) -> EditSettings:
    """Return validated settings where the named edits take the given values."""
    return pipeline.validate(EditSettings(edits))
