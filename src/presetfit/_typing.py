"""Shared type aliases."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
"""A floating-point NumPy array. Pixel arrays have a trailing axis of length 3."""
