"""Exporting a look to formats other software understands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from presetfit._version import __version__
from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace
from presetfit.edits.pipeline import EditPipeline
from presetfit.edits.settings import EditSettings
from presetfit.exceptions import InvalidSettingsError

PRESET_FORMAT = "presetfit-preset"
PRESET_VERSION = 1


class PresetCodec:
    """Serialises settings to and from the JSON preset format.

    A preset is how a fitted look is saved and re-applied to other photos. The
    format carries a name and a version so future releases can recognise and
    migrate old files instead of misreading them.
    """

    def __init__(self, pipeline: EditPipeline) -> None:
        self._pipeline = pipeline

    def dumps(self, settings: EditSettings) -> str:
        """Return ``settings`` as a JSON preset document."""
        document = {
            "format": PRESET_FORMAT,
            "version": PRESET_VERSION,
            "generator": f"presetfit {__version__}",
            "settings": self._pipeline.validate(settings).to_dict(),
        }
        return json.dumps(document, indent=2) + "\n"

    def loads(self, text: str | bytes) -> EditSettings:
        """Parse and validate a JSON preset document.

        Raises:
            InvalidSettingsError: if the text is not a valid preset.
        """
        try:
            document: Any = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InvalidSettingsError(f"preset is not valid JSON: {error}") from error
        if not isinstance(document, dict) or document.get("format") != PRESET_FORMAT:
            raise InvalidSettingsError("not a PresetFit preset file")
        if document.get("version") != PRESET_VERSION:
            raise InvalidSettingsError(
                f"unsupported preset version {document.get('version')!r}; "
                f"this release reads version {PRESET_VERSION}"
            )
        return self._pipeline.validate(EditSettings.from_dict(document.get("settings")))

    def save(self, settings: EditSettings, path: str | Path) -> Path:
        """Write ``settings`` to ``path`` as a preset file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.dumps(settings), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> EditSettings:
        """Read a preset file."""
        try:
            return self.loads(Path(path).read_bytes())
        except OSError as error:
            raise InvalidSettingsError(f"cannot read preset '{path}': {error}") from error


class CubeLutExporter:
    """Bakes a look into a ``.cube`` 3D lookup table.

    A LUT lists, for a grid of input colours, the colour each should become.
    Lightroom, Photoshop, DaVinci Resolve and most video editors can load one,
    which is how a look fitted here is carried into other software. It works
    because every edit in the pipeline depends only on a pixel's own colour.
    """

    def __init__(self, pipeline: EditPipeline, size: int = 33) -> None:
        if not 2 <= size <= 65:
            raise ValueError("LUT size must be between 2 and 65")
        self._pipeline = pipeline
        self._size = size

    def export(self, settings: EditSettings, title: str = "PresetFit") -> str:
        """Return the contents of a ``.cube`` file for ``settings``."""
        settings = self._pipeline.validate(settings)
        axis = np.linspace(0.0, 1.0, self._size)
        # The .cube format lists entries with red changing fastest, then green, then blue.
        blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
        grid = Pixels(np.stack([red, green, blue], axis=-1).reshape(-1, 3), ColorSpace.SRGB)
        mapped = np.clip(self._pipeline.run(grid, settings).to(ColorSpace.SRGB).data, 0.0, 1.0)

        safe_title = title.replace('"', "'").replace("\n", " ")
        header = [
            f'TITLE "{safe_title}"',
            f"LUT_3D_SIZE {self._size}",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0",
        ]
        rows = [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in mapped]
        return "\n".join(header + rows) + "\n"

    def save(self, settings: EditSettings, path: str | Path, title: str | None = None) -> Path:
        """Write a ``.cube`` file for ``settings`` to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.export(settings, title or target.stem), encoding="ascii", errors="replace"
        )
        return target
