"""Immutable container for the slider values of every edit."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from presetfit.exceptions import InvalidSettingsError


class EditSettings:
    """Slider values keyed by edit name, then by parameter name.

    Instances are immutable: every "change" returns a new object. That makes
    settings safe to share between threads and impossible to corrupt halfway
    through a fit.

    This class only stores values. Whether they are *valid* depends on which
    edits exist, so validation belongs to
    :meth:`presetfit.edits.pipeline.EditPipeline.validate`.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Mapping[str, float]]) -> None:
        self._values: Mapping[str, Mapping[str, float]] = MappingProxyType(
            {name: MappingProxyType(dict(params)) for name, params in values.items()}
        )

    def for_edit(self, edit_name: str) -> Mapping[str, float]:
        """Return the values of one edit."""
        try:
            return self._values[edit_name]
        except KeyError:
            raise InvalidSettingsError(f"settings have no entry for edit '{edit_name}'") from None

    def replace(self, updates: Mapping[str, Mapping[str, float]]) -> EditSettings:
        """Return a copy in which the edits named in ``updates`` take new values."""
        merged = {name: dict(params) for name, params in self._values.items()}
        for name, params in updates.items():
            merged[name] = dict(params)
        return EditSettings(merged)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return a plain, JSON-serialisable copy."""
        return {name: dict(params) for name, params in self._values.items()}

    @classmethod
    def from_dict(cls, data: Any) -> EditSettings:
        """Build settings from untrusted data, checking only its overall shape."""
        if not isinstance(data, Mapping):
            raise InvalidSettingsError("settings must be an object keyed by edit name")
        for name, params in data.items():
            if not isinstance(name, str) or not isinstance(params, Mapping):
                raise InvalidSettingsError("each edit entry must be an object of parameters")
            if not all(isinstance(key, str) for key in params):
                raise InvalidSettingsError(f"{name}: parameter names must be strings")
        return cls(data)

    def __contains__(self, edit_name: object) -> bool:
        return edit_name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EditSettings):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(tuple((name, tuple(params.items())) for name, params in self._values.items()))

    def __repr__(self) -> str:
        return f"EditSettings({self.to_dict()!r})"
