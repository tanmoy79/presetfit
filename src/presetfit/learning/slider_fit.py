"""Find the slider values that best turn one image into another, pixel by pixel.

After the network has redrawn a small copy of your photo, this module works out
which settings of PresetFit's eight edits reproduce that result. That gives
the edits list (and, later, Lightroom values) for an image the network made.

Unlike the original fitter, which compares overall statistics of two
*different* photos, here the two images show the same scene pixel for pixel,
so the comparison can be exact: every pixel's colour difference counts. The
differentiable renderer gives the direction in which to move all 40 sliders at
once, and a standard optimiser (Adam) follows it.
"""

from __future__ import annotations

import numpy as np
import torch

from presetfit._typing import FloatArray
from presetfit.edits.settings import EditSettings
from presetfit.learning.render import TorchRenderer, delta_e76, srgb_to_lab


def fit_sliders(
    source: FloatArray,
    target: FloatArray,
    renderer: TorchRenderer | None = None,
    *,
    steps: int = 300,
    sample: int = 16_384,
    ridge: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> EditSettings:
    """Return settings that make ``source`` look as much like ``target`` as possible.

    Args:
        source, target: Aligned sRGB images ``(..., 3)`` in 0-1.
        steps: Optimiser steps. 300 settles on typical photos.
        sample: Pixels used (a random subset keeps it fast).
        ridge: Gentle preference for neutral sliders when several settings fit
            equally well, so the edits list stays simple.
        device: ``"cuda"`` to run on a GPU.
    """
    renderer = renderer or TorchRenderer()
    src = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("source and target must have the same number of pixels")
    rng = np.random.default_rng(seed)
    if len(src) > sample:
        chosen = rng.choice(len(src), sample, replace=False)
        src, dst = src[chosen], dst[chosen]
    source_t = torch.from_numpy(src)[None].to(device)
    target_lab = srgb_to_lab(torch.from_numpy(dst)[None].to(device).clamp(0.0, 1.0))

    raw = torch.zeros(1, renderer.codec.size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(steps):
        normalized = torch.tanh(raw)
        rendered = renderer.render_lab(source_t, renderer.to_values(normalized))
        loss = delta_e76(rendered, target_lab).mean() + ridge * (normalized**2).mean()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        values = renderer.to_values(torch.tanh(raw))[0].double().cpu().numpy()
    return renderer.codec.from_vector(values)
