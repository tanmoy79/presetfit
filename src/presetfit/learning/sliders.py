"""Turn slider settings into a fixed-length vector of numbers and back.

A neural network outputs a plain list of numbers, while PresetFit stores
settings as ``{edit: {slider: value}}``. :class:`SliderCodec` fixes one order
for all sliders, taken from the pipeline, and converts between the two forms.

It also defines the *normalised* scale the network works on: 0 is the neutral
value of every slider, +1 its maximum and -1 its minimum. One scale for every
slider keeps the network's outputs comparable whether a slider runs from -3 to
+3 EV or from -100 to +100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presetfit._typing import FloatArray
from presetfit.edits.pipeline import EditPipeline
from presetfit.edits.settings import EditSettings


@dataclass(frozen=True)
class SliderSlot:
    """One position in the vector: which edit and slider it belongs to."""

    edit: str
    param: str
    minimum: float
    maximum: float
    default: float

    @property
    def key(self) -> str:
        """``edit.param``, for example ``exposure.exposure``."""
        return f"{self.edit}.{self.param}"


class SliderCodec:
    """Converts between :class:`EditSettings` and vectors in pipeline order."""

    def __init__(self, pipeline: EditPipeline) -> None:
        self._pipeline = pipeline
        self._slots = tuple(
            SliderSlot(edit.name, spec.name, spec.minimum, spec.maximum, spec.default)
            for edit in pipeline.edits
            for spec in edit.params
        )
        self.minimum = np.array([slot.minimum for slot in self._slots])
        self.maximum = np.array([slot.maximum for slot in self._slots])
        self.default = np.array([slot.default for slot in self._slots])
        # How far each slider can move up and down from neutral. One-sided
        # sliders (a tone-curve end point) have zero room on one side.
        self.room_up = self.maximum - self.default
        self.room_down = self.default - self.minimum

    @property
    def slots(self) -> tuple[SliderSlot, ...]:
        """Every slider, in vector order."""
        return self._slots

    @property
    def size(self) -> int:
        """Length of the vector."""
        return len(self._slots)

    def to_vector(self, settings: EditSettings) -> FloatArray:
        """Return the slider values of ``settings`` in vector order."""
        complete = self._pipeline.validate(settings)
        return np.array([complete.for_edit(slot.edit)[slot.param] for slot in self._slots])

    def from_vector(self, values: FloatArray) -> EditSettings:
        """Build settings from a vector, clamping to each slider's range and rounding.

        The result is rounded to display precision, the same as the fitter's
        output, so the report shows exactly the values used for rendering.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError(f"expected {self.size} slider values, got shape {values.shape}")
        clamped = np.clip(values, self.minimum, self.maximum)
        grouped: dict[str, dict[str, float]] = {}
        for slot, value in zip(self._slots, clamped, strict=True):
            grouped.setdefault(slot.edit, {})[slot.param] = float(value)
        return self._pipeline.quantize(EditSettings(grouped))

    def normalize(self, values: FloatArray) -> FloatArray:
        """Map slider values to the -1 (minimum) ... 0 (neutral) ... +1 (maximum) scale."""
        offset = np.asarray(values, dtype=np.float64) - self.default
        up = np.divide(offset, self.room_up, out=np.zeros_like(offset), where=self.room_up > 0)
        down = np.divide(
            offset, self.room_down, out=np.zeros_like(offset), where=self.room_down > 0
        )
        result: FloatArray = np.where(offset >= 0, up, down)
        return result

    def denormalize(self, normalized: FloatArray) -> FloatArray:
        """Inverse of :meth:`normalize`; values beyond +-1 are clamped."""
        clipped = np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)
        result: FloatArray = self.default + np.where(
            clipped >= 0, clipped * self.room_up, clipped * self.room_down
        )
        return result
