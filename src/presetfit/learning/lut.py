"""Carry a look from a small image to a full-size photo with a colour lookup table.

The network works on 256 x 256 copies, but people want their full-size photo
back. The trick (also used by the ICTone paper): look at the small photo and
the network's small result side by side, and work out the *colour mapping*
between them: "this blue became that teal, this skin tone became that warmer
one". That mapping is stored as a 3D lookup table (LUT), a 17 x 17 x 17 grid
of colours, and applied to every pixel of the full-size photo. Detail and
sharpness come from the original; only colours change.

Fitting is a regularised least-squares problem: each small-image pixel says
"input colour X should become colour Y"; trilinear interpolation spreads that
over the eight surrounding grid points; and a smoothness term keeps
neighbouring grid points close, which fills in colours the small image never
showed and stops the table from following noise.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve

from presetfit._typing import FloatArray

DEFAULT_LUT_SIZE = 17


def _trilinear_weights(colors: FloatArray, size: int) -> sparse.csr_matrix:
    """Sparse matrix mapping LUT grid values to the colour at each pixel."""
    scaled = np.clip(colors, 0.0, 1.0) * (size - 1)
    lower = np.minimum(np.floor(scaled).astype(np.int64), size - 2)
    frac = scaled - lower
    rows, cols, vals = [], [], []
    pixel = np.arange(len(colors))
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                weight = (
                    (frac[:, 0] if dr else 1 - frac[:, 0])
                    * (frac[:, 1] if dg else 1 - frac[:, 1])
                    * (frac[:, 2] if db else 1 - frac[:, 2])
                )
                node = ((lower[:, 0] + dr) * size + lower[:, 1] + dg) * size + lower[:, 2] + db
                rows.append(pixel)
                cols.append(node)
                vals.append(weight)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(colors), size**3),
    )


def _smoothness(size: int) -> sparse.csr_matrix:
    """Differences between neighbouring grid points along each of the three axes."""
    one_axis = sparse.diags([-1.0, 1.0], [0, 1], shape=(size - 1, size))
    eye = sparse.identity(size)
    return sparse.vstack(
        [
            sparse.kron(sparse.kron(one_axis, eye), eye),
            sparse.kron(sparse.kron(eye, one_axis), eye),
            sparse.kron(sparse.kron(eye, eye), one_axis),
        ]
    ).tocsr()


def identity_lut(size: int = DEFAULT_LUT_SIZE) -> FloatArray:
    """A table that changes nothing: shape ``(size, size, size, 3)``, indexed [r, g, b]."""
    axis = np.linspace(0.0, 1.0, size)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def fit_lut(
    source: FloatArray,
    target: FloatArray,
    size: int = DEFAULT_LUT_SIZE,
    smoothness: float = 0.5,
) -> FloatArray:
    """Fit a LUT that turns the colours of ``source`` into those of ``target``.

    Args:
        source, target: Aligned sRGB images or pixel lists, ``(..., 3)`` in 0-1.
        size: Grid points per axis.
        smoothness: Strength of the smoothness term relative to the data,
            scaled for the number of pixels so results do not depend on it.

    Returns:
        The table, shape ``(size, size, size, 3)``.
    """
    colors = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    wanted = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if colors.shape != wanted.shape:
        raise ValueError("source and target must have the same number of pixels")
    weights = _trilinear_weights(colors, size)
    differences = _smoothness(size)
    # Fit the *change* from identity, so unseen colours default to "no change".
    identity = identity_lut(size).reshape(-1, 3)
    residual = wanted - weights @ identity
    strength = smoothness * len(colors) / size**3
    system = (weights.T @ weights + strength * (differences.T @ differences)).tocsc()
    system = system + 1e-6 * sparse.identity(size**3, format="csc")
    change = np.column_stack([spsolve(system, weights.T @ residual[:, c]) for c in range(3)])
    table: FloatArray = np.clip(identity + change, 0.0, 1.0).reshape(size, size, size, 3)
    return table


def apply_lut(image: FloatArray, lut: FloatArray, chunk_pixels: int = 2_000_000) -> FloatArray:
    """Apply ``lut`` to an sRGB image of any size with trilinear interpolation."""
    size = lut.shape[0]
    flat = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    output = np.empty_like(flat)
    for start in range(0, len(flat), chunk_pixels):
        coords = (np.clip(flat[start : start + chunk_pixels], 0.0, 1.0) * (size - 1)).T
        for channel in range(3):
            output[start : start + chunk_pixels, channel] = map_coordinates(
                lut[..., channel], coords, order=1, mode="nearest"
            )
    return output.reshape(np.shape(image))
