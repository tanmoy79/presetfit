"""PresetFit: give one photo the look of another, and report the edits that did it.

Typical use::

    from presetfit import ImageReader, ImageWriter, PresetFitService, TextReportFormatter

    service = PresetFitService.create_default()
    reader = ImageReader()
    result = service.match(reader.read("my_photo.jpg"), reader.read("reference.jpg"))

    ImageWriter().write(result.image, "my_photo_matched.jpg")
    print(TextReportFormatter().format(result.report))
"""

from presetfit._version import __version__
from presetfit.color import ColorSpace, Pixels
from presetfit.edits import EditPipeline, EditSettings, default_pipeline
from presetfit.exceptions import PresetFitError
from presetfit.export import CubeLutExporter, PresetCodec
from presetfit.fitting import FitConfig, FitResult, LookFitter, StagedLookFitter
from presetfit.imaging import ImageReader, ImageWriter
from presetfit.report import LookReport, TextReportFormatter
from presetfit.service import MatchResult, PresetFitService

__all__ = [
    "ColorSpace",
    "CubeLutExporter",
    "EditPipeline",
    "EditSettings",
    "FitConfig",
    "FitResult",
    "ImageReader",
    "ImageWriter",
    "LookFitter",
    "LookReport",
    "MatchResult",
    "Pixels",
    "PresetCodec",
    "PresetFitError",
    "PresetFitService",
    "StagedLookFitter",
    "TextReportFormatter",
    "__version__",
    "default_pipeline",
]
