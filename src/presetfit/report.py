"""Turning fitted settings into a report people can read."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from presetfit.edits.base import CurvePlot
from presetfit.edits.pipeline import EditDescription, EditPipeline
from presetfit.edits.settings import EditSettings
from presetfit.fitting.base import FitResult


@dataclass(frozen=True)
class LookReport:
    """Everything worth telling the user about one match.

    Attributes:
        sections: One entry per edit, each with its human-readable lines.
        notes: Plain-language caveats, such as colours that were skipped.
        distance_before / distance_after: Look distance to the reference
            before and after the edits (lower is closer).
        improvement: Fraction of the look gap that was closed, 0-1.
    """

    sections: tuple[EditDescription, ...]
    notes: tuple[str, ...]
    distance_before: float
    distance_after: float
    improvement: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "sections": [
                {
                    "name": section.name,
                    "label": section.label,
                    "lines": [{"label": line.label, "value": line.value} for line in section.lines],
                    "curve": _curve_to_dict(section.plot),
                }
                for section in self.sections
            ],
            "notes": list(self.notes),
            "distance_before": round(self.distance_before, 3),
            "distance_after": round(self.distance_after, 3),
            "improvement": round(self.improvement, 4),
        }


class ReportBuilder:
    """Assembles a :class:`LookReport` from a fit."""

    def __init__(self, pipeline: EditPipeline) -> None:
        self._pipeline = pipeline

    def build(self, fit: FitResult, applied: EditSettings, strength: float = 1.0) -> LookReport:
        """Describe ``applied`` (the settings actually rendered) in the context of ``fit``."""
        notes: list[str] = []
        if strength < 1.0:
            notes.append(
                f"Edits were applied at {strength:.0%} strength; "
                "the match score below is for full strength."
            )
        if fit.skipped_bands:
            notes.append(
                "Per-colour (HSL) edits were skipped for "
                f"{_join(fit.skipped_bands)}: the two photos do not share enough of "
                + ("that colour." if len(fit.skipped_bands) == 1 else "those colours.")
            )
        if fit.improvement < 0.5:
            notes.append(
                "The look could only be partly matched with global edits. This usually "
                "means the two photos differ in subject or lighting more than in editing."
            )
        return LookReport(
            sections=self._pipeline.describe(applied),
            notes=tuple(notes),
            distance_before=fit.distance_before,
            distance_after=fit.distance_after,
            improvement=fit.improvement,
        )


class ReportFormatter(ABC):
    """Renders a :class:`LookReport` in some output format."""

    @abstractmethod
    def format(self, report: LookReport) -> str:
        """Return ``report`` as text."""


class TextReportFormatter(ReportFormatter):
    """Formats a report as aligned plain text for a terminal or a ``.txt`` file."""

    _RULE = "-" * 56

    def format(self, report: LookReport) -> str:
        label_width = max(
            (len(line.label) for section in report.sections for line in section.lines), default=0
        )
        out = ["EDITS APPLIED", self._RULE]
        for section in report.sections:
            out.append(section.label)
            out.extend(f"  {line.label:<{label_width}}  {line.value}" for line in section.lines)
        out.append(self._RULE)
        out.append(
            f"Look distance to reference: {report.distance_before:.2f} -> "
            f"{report.distance_after:.2f} ({report.improvement:.0%} of the gap closed)"
        )
        if report.notes:
            out.append("")
            out.extend(f"Note: {note}" for note in report.notes)
        return "\n".join(out)


def _curve_to_dict(plot: CurvePlot | None) -> dict[str, Any] | None:
    if plot is None:
        return None
    return {
        "points": [list(point) for point in plot.points],
        "handles": [list(handle) for handle in plot.handles],
        "axis_max": plot.axis_max,
    }


def _join(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"
