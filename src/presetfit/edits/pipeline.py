"""An ordered chain of edits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from presetfit.color.pixels import Pixels
from presetfit.edits.base import CurvePlot, Edit, ReportLine
from presetfit.edits.settings import EditSettings
from presetfit.exceptions import InvalidSettingsError


@dataclass(frozen=True)
class EditDescription:
    """How one edit is reported: its text lines and, for some edits, a curve."""

    name: str
    label: str
    lines: tuple[ReportLine, ...]
    plot: CurvePlot | None = None


class EditPipeline:
    """Applies a fixed sequence of edits, converting colour spaces in between.

    The order matters, as it does in a real photo editor: light-level edits run
    first in linear RGB, then tonal and colour edits run in Lab. The pipeline is
    the only place that knows the order; everything else asks it.
    """

    def __init__(self, edits: Sequence[Edit]) -> None:
        names = [edit.name for edit in edits]
        if len(set(names)) != len(names):
            raise ValueError("edit names in a pipeline must be unique")
        self._edits: tuple[Edit, ...] = tuple(edits)
        self._index = {edit.name: position for position, edit in enumerate(self._edits)}

    @property
    def edits(self) -> tuple[Edit, ...]:
        """The edits, in the order they are applied."""
        return self._edits

    def edit(self, name: str) -> Edit:
        """Look an edit up by name."""
        return self._edits[self.index_of(name)]

    def index_of(self, name: str) -> int:
        """Return the position of the edit called ``name``."""
        try:
            return self._index[name]
        except KeyError:
            raise InvalidSettingsError(f"unknown edit '{name}'") from None

    def identity_settings(self) -> EditSettings:
        """Return settings that leave an image unchanged."""
        return EditSettings({edit.name: edit.defaults() for edit in self._edits})

    def validate(self, settings: EditSettings) -> EditSettings:
        """Return a complete, checked copy of ``settings``.

        Edits missing from ``settings`` take their defaults, so a preset saved
        by an older version with fewer edits still loads.

        Raises:
            InvalidSettingsError: if the settings name an unknown edit or
                parameter, or hold an out-of-range value.
        """
        unknown = sorted(set(settings) - set(self._index))
        if unknown:
            raise InvalidSettingsError(f"unknown edit(s): {', '.join(unknown)}")
        return EditSettings(
            {
                edit.name: edit.validate(
                    settings.for_edit(edit.name) if edit.name in settings else {}
                )
                for edit in self._edits
            }
        )

    def blend(self, settings: EditSettings, strength: float) -> EditSettings:
        """Scale every slider towards its neutral value.

        ``strength`` 1 returns the settings unchanged, 0 returns the identity,
        and 0.5 applies each edit at half power.
        """
        if not 0.0 <= strength <= 1.0:
            raise InvalidSettingsError("strength must be between 0 and 1")
        blended = {}
        for edit in self._edits:
            values = settings.for_edit(edit.name)
            blended[edit.name] = {
                spec.name: spec.default + strength * (values[spec.name] - spec.default)
                for spec in edit.params
            }
        return EditSettings(blended)

    def quantize(self, settings: EditSettings) -> EditSettings:
        """Round every slider to its display precision.

        Rendering with the rounded values guarantees that the numbers in the
        report are exactly the numbers that produced the picture.
        """
        return EditSettings(
            {
                edit.name: {
                    spec.name: spec.quantize(settings.for_edit(edit.name)[spec.name])
                    for spec in edit.params
                }
                for edit in self._edits
            }
        )

    def run(
        self,
        pixels: Pixels,
        settings: EditSettings,
        start: int = 0,
        stop: int | None = None,
    ) -> Pixels:
        """Apply edits ``start`` to ``stop`` (exclusive) and return the result.

        ``settings`` must already be validated; this method sits in the fitter's
        inner loop and does no checking of its own. The result is left in the
        colour space of the last edit that ran, so callers convert it to
        whatever they need. Edits at their neutral values are skipped.
        """
        for edit in self._edits[start:stop]:
            values = settings.for_edit(edit.name)
            if edit.is_identity(values):
                continue
            working = pixels.to(edit.space)
            pixels = Pixels(edit.apply(working.data, values), edit.space)
        return pixels

    def describe(self, settings: EditSettings) -> tuple[EditDescription, ...]:
        """Return human-readable report lines for every edit."""
        return tuple(
            EditDescription(
                name=edit.name,
                label=edit.label,
                lines=tuple(edit.describe(settings.for_edit(edit.name))),
                plot=edit.plot(settings.for_edit(edit.name)),
            )
            for edit in self._edits
        )
