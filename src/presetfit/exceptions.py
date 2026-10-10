"""Exception hierarchy.

Every error the library raises on purpose derives from :class:`PresetFitError`,
so callers (the CLI, the web API) can catch one type and report it cleanly.
"""

from __future__ import annotations


class PresetFitError(Exception):
    """Base class for all expected PresetFit failures."""


class ImageDecodeError(PresetFitError):
    """The supplied bytes could not be decoded as an image."""


class UnsupportedImageError(PresetFitError):
    """The image decoded, but its format is not one we accept."""


class ImageTooLargeError(PresetFitError):
    """The image exceeds the configured pixel or byte limit."""


class InvalidSettingsError(PresetFitError):
    """Edit settings are malformed: unknown names, bad types or out-of-range values."""


class FittingError(PresetFitError):
    """The optimiser could not fit edit settings to the reference look."""
