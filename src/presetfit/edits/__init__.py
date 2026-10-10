"""The edits and the pipeline that chains them."""

from presetfit.edits.base import CurvePlot, Edit, ParamSpec, ReportLine
from presetfit.edits.color import HueSaturationLuminance, Saturation, SplitToning, WhiteBalance
from presetfit.edits.pipeline import EditDescription, EditPipeline
from presetfit.edits.settings import EditSettings
from presetfit.edits.tone import Contrast, Exposure, HighlightsShadows, ToneCurve


def default_pipeline() -> EditPipeline:
    """Build the standard eight-edit pipeline.

    Linear-light edits come first, then tonal edits, then colour edits. The
    fitter relies on this grouping: each of its stages tunes a run of
    neighbouring edits.
    """
    return EditPipeline(
        [
            WhiteBalance(),
            Exposure(),
            Contrast(),
            HighlightsShadows(),
            ToneCurve(),
            Saturation(),
            HueSaturationLuminance(),
            SplitToning(),
        ]
    )


__all__ = [
    "Contrast",
    "CurvePlot",
    "Edit",
    "EditDescription",
    "EditPipeline",
    "EditSettings",
    "Exposure",
    "HighlightsShadows",
    "HueSaturationLuminance",
    "ParamSpec",
    "ReportLine",
    "Saturation",
    "SplitToning",
    "ToneCurve",
    "WhiteBalance",
    "default_pipeline",
]
