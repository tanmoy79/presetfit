"""Use a trained network on real, full-size photos.

1. Shrink your photo and the reference to 256 x 256.
2. The network redraws the small photo in the reference's look.
3. From the small photo and the small result, work out the colour change:
   a lookup table (:mod:`presetfit.learning.lut`) and the equivalent slider
   values (:mod:`presetfit.learning.slider_fit`).
4. Apply that to the full-size photo.

Two ways to make the final picture are offered so the benchmark can compare
them: ``"lut"`` (follows the network most closely) and ``"sliders"`` (the
picture is exactly what the edits list says).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from presetfit._typing import FloatArray
from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace
from presetfit.edits import default_pipeline
from presetfit.edits.settings import EditSettings
from presetfit.learning.lut import apply_lut, fit_lut
from presetfit.learning.model import LookTransferNet
from presetfit.learning.render import TorchRenderer
from presetfit.learning.slider_fit import fit_sliders
from presetfit.rendering import Renderer

WORKING_SIZE = 256


def _squash(image: FloatArray, size: int) -> FloatArray:
    data = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    small = Image.fromarray(data).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.float32) / 255.0


@dataclass(frozen=True)
class TransferResult:
    """The full-size result, plus the small images and fits it came from."""

    image: FloatArray
    settings: EditSettings | None
    small_content: FloatArray
    small_result: FloatArray


class LearnedLookTransfer:
    """Gives a photo the look of a reference with a trained :class:`LookTransferNet`."""

    def __init__(self, model: LookTransferNet, device: str = "cpu") -> None:
        self._model = model.to(device).eval()
        self._device = device
        self._pipeline = default_pipeline()
        self._renderer = Renderer(self._pipeline)
        self._torch_renderer = TorchRenderer(self._pipeline)

    @classmethod
    def load(cls, checkpoint: Path, device: str = "cpu") -> LearnedLookTransfer:
        from presetfit.learning.train import load_model

        return cls(load_model(checkpoint, device), device)

    @torch.no_grad()
    def small_result(
        self, content: FloatArray, reference: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the 256 x 256 photo and the network's 256 x 256 result."""
        small_content = _squash(content, WORKING_SIZE)
        small_reference = _squash(reference, WORKING_SIZE)
        batch = [
            torch.from_numpy(x)[None].to(self._device) for x in (small_content, small_reference)
        ]
        output = self._model(*batch).clamp(0.0, 1.0)[0].float().cpu().numpy()
        return small_content, output

    def transfer(
        self,
        content: FloatArray,
        reference: FloatArray,
        mode: str = "lut",
        with_settings: bool = True,
    ) -> TransferResult:
        """Give the full-size sRGB ``content`` the look of ``reference``.

        Args:
            mode: ``"lut"`` or ``"sliders"`` (see the module notes).
            with_settings: In ``"lut"`` mode, also fit the slider values for
                the edits list. Skipping it saves time when only the picture
                is needed, as in the benchmark.
        """
        if mode not in ("lut", "sliders"):
            raise ValueError("mode must be 'lut' or 'sliders'")
        small_content, small_output = self.small_result(content, reference)
        settings = None
        if mode == "sliders" or with_settings:
            settings = fit_sliders(
                small_content, small_output, self._torch_renderer, device=self._device
            )
        if mode == "lut":
            image = apply_lut(content, fit_lut(small_content, small_output))
        else:
            assert settings is not None
            image = self._renderer.render(
                Pixels(np.asarray(content, np.float32), ColorSpace.SRGB), settings
            ).data
        return TransferResult(np.clip(image, 0.0, 1.0), settings, small_content, small_output)
