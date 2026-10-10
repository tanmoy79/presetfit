"""The :class:`Edit` abstraction that every adjustment implements."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from presetfit._typing import FloatArray
from presetfit.color.spaces import ColorSpace
from presetfit.exceptions import InvalidSettingsError


@dataclass(frozen=True)
class ParamSpec:
    """Describes one slider: its name, allowed range, neutral value and precision."""

    name: str
    label: str
    minimum: float
    maximum: float
    default: float = 0.0
    step: float = 1.0
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"default of '{self.name}' lies outside its range")
        if self.step <= 0:
            raise ValueError(f"step of '{self.name}' must be positive")

    def quantize(self, value: float) -> float:
        """Round ``value`` to this slider's precision and keep it inside its range."""
        rounded = round(value / self.step) * self.step
        clamped = min(self.maximum, max(self.minimum, rounded))
        return clamped + 0.0  # normalise -0.0 to 0.0


@dataclass(frozen=True)
class ReportLine:
    """One human-readable line of the edit report, e.g. ``Exposure: +0.35 EV``."""

    label: str
    value: str


@dataclass(frozen=True)
class CurvePlot:
    """A curve an edit offers for graphical display, as an editor would draw it.

    Attributes:
        points: Closely spaced ``(input, output)`` samples tracing the curve.
        handles: The few control points that define the curve.
        axis_max: The value at the top of both axes (255 for a tone curve).
    """

    points: tuple[tuple[float, float], ...]
    handles: tuple[tuple[float, float], ...]
    axis_max: float


class Edit(ABC):
    """One photo adjustment, such as exposure or saturation.

    An edit is a *pure function with named parameters*: given pixel data in its
    working colour space and a value for each of its sliders, it returns new
    pixel data. It holds no image state, so one instance can be shared freely.

    Subclasses declare four class attributes and implement :meth:`apply`. Because
    every edit obeys this one contract, the pipeline, the fitter, the report and
    the LUT exporter work with any edit without knowing which one it is. Adding a
    new adjustment means adding one subclass and touching nothing else.
    """

    name: ClassVar[str]
    """Stable machine identifier, used as the key in presets."""

    label: ClassVar[str]
    """Human-readable name for reports."""

    space: ClassVar[ColorSpace]
    """Colour space in which :meth:`apply` expects and returns data."""

    params: ClassVar[tuple[ParamSpec, ...]]
    """The sliders this edit exposes."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        missing = [attr for attr in ("name", "label", "space", "params") if not hasattr(cls, attr)]
        if missing:
            raise TypeError(f"{cls.__name__} must define class attributes: {', '.join(missing)}")

    @abstractmethod
    def apply(self, data: FloatArray, values: Mapping[str, float]) -> FloatArray:
        """Apply the edit to ``data`` (shape ``(..., 3)``, in :attr:`space`).

        Must return a new array of the same shape and dtype, and must be the
        identity when every value equals its default.
        """

    def defaults(self) -> dict[str, float]:
        """Return the neutral value of every slider."""
        return {spec.name: spec.default for spec in self.params}

    def is_identity(self, values: Mapping[str, float]) -> bool:
        """Report whether ``values`` leave the image unchanged."""
        return all(values[spec.name] == spec.default for spec in self.params)

    def validate(self, values: Mapping[str, float]) -> dict[str, float]:
        """Check ``values`` and return a complete copy with defaults filled in.

        Raises:
            InvalidSettingsError: on an unknown slider, a non-numeric or
                non-finite value, or a value outside the slider's range.
        """
        known = {spec.name for spec in self.params}
        unknown = sorted(set(values) - known)
        if unknown:
            raise InvalidSettingsError(f"{self.name}: unknown parameter(s) {', '.join(unknown)}")
        complete: dict[str, float] = {}
        for spec in self.params:
            raw = values.get(spec.name, spec.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise InvalidSettingsError(f"{self.name}.{spec.name} must be a number")
            value = float(raw)
            if not math.isfinite(value):
                raise InvalidSettingsError(f"{self.name}.{spec.name} must be finite")
            if not spec.minimum <= value <= spec.maximum:
                raise InvalidSettingsError(
                    f"{self.name}.{spec.name}={value:g} is outside "
                    f"[{spec.minimum:g}, {spec.maximum:g}]"
                )
            complete[spec.name] = value
        return complete

    def plot(self, values: Mapping[str, float]) -> CurvePlot | None:
        """Return a curve picturing ``values``, for edits best shown as a graph.

        Most edits are plain sliders and return ``None``.
        """
        return None

    def describe(self, values: Mapping[str, float]) -> list[ReportLine]:
        """Return report lines for ``values``. Override for friendlier wording."""
        return [
            ReportLine(spec.label, format_signed(values[spec.name], spec.step, spec.unit))
            for spec in self.params
        ]


def format_signed(value: float, step: float = 1.0, unit: str = "") -> str:
    """Format a slider value with an explicit sign, e.g. ``+12`` or ``-0.35 EV``."""
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    text = f"{value:+.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return f"{text} {unit}".strip()
