from __future__ import annotations

import numpy as np
import pytest

from presetfit import ColorSpace, Pixels
from presetfit.color.masks import (
    BAND_CENTERS_DEG,
    BAND_NAMES,
    chroma_mask,
    hue_band_weights,
    nearest_band_name,
    tonal_zone_weights,
)
from presetfit.color.spaces import lab_to_linear, linear_to_lab, linear_to_srgb, srgb_to_linear


@pytest.fixture
def random_srgb() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 1, (500, 3))


def test_srgb_linear_round_trip(random_srgb: np.ndarray) -> None:
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(random_srgb)), random_srgb, atol=1e-9)


def test_lab_round_trip(random_srgb: np.ndarray) -> None:
    np.testing.assert_allclose(lab_to_linear(linear_to_lab(random_srgb)), random_srgb, atol=1e-9)


@pytest.mark.parametrize(
    ("srgb", "expected_lab"),
    [
        ((1.0, 1.0, 1.0), (100.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (53.24, 80.09, 67.20)),
        ((0.0, 0.0, 1.0), (32.30, 79.19, -107.86)),
    ],
)
def test_known_lab_values(srgb: tuple[float, ...], expected_lab: tuple[float, ...]) -> None:
    lab = Pixels(np.array([srgb]), ColorSpace.SRGB).to(ColorSpace.LAB).data[0]
    np.testing.assert_allclose(lab, expected_lab, atol=0.05)


def test_middle_grey_is_neutral() -> None:
    lab = Pixels(np.full((1, 3), 0.5), ColorSpace.SRGB).to(ColorSpace.LAB).data[0]
    assert abs(lab[1]) < 1e-6
    assert abs(lab[2]) < 1e-6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("space", list(ColorSpace))
def test_conversion_preserves_dtype(dtype: type, space: ColorSpace) -> None:
    pixels = Pixels(np.full((4, 3), 0.4, dtype=dtype), ColorSpace.SRGB)
    assert pixels.to(space).data.dtype == dtype


def test_to_same_space_returns_self() -> None:
    pixels = Pixels(np.zeros((2, 3)), ColorSpace.LAB)
    assert pixels.to(ColorSpace.LAB) is pixels


def test_leaving_linear_light_clips_to_displayable_range() -> None:
    overexposed = Pixels(np.array([[4.0, 4.0, 4.0], [-1.0, -1.0, -1.0]]), ColorSpace.LINEAR_RGB)
    np.testing.assert_allclose(
        overexposed.to(ColorSpace.SRGB).data, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(overexposed.to(ColorSpace.LAB).data[:, 0], [100.0, 0.0], atol=1e-9)


def test_pixels_rejects_bad_shapes_and_types() -> None:
    with pytest.raises(ValueError, match="shape"):
        Pixels(np.zeros((4, 4)), ColorSpace.SRGB)
    with pytest.raises(TypeError, match="floating"):
        Pixels(np.zeros((4, 3), dtype=np.uint8), ColorSpace.SRGB)


def test_flattened_keeps_values() -> None:
    image = Pixels(np.random.default_rng(1).uniform(size=(5, 7, 3)), ColorSpace.SRGB)
    flat = image.flattened()
    assert flat.data.shape == (35, 3)
    assert flat.space is ColorSpace.SRGB


def test_band_weights_sum_to_one_everywhere() -> None:
    hues = np.linspace(0, 360, 721, endpoint=False)
    weights = hue_band_weights(hues)
    assert weights.shape == (721, len(BAND_NAMES))
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    assert weights.min() >= 0


def test_band_centre_belongs_wholly_to_its_band() -> None:
    weights = hue_band_weights(np.array(BAND_CENTERS_DEG))
    np.testing.assert_allclose(weights, np.eye(len(BAND_NAMES)), atol=1e-12)


def test_band_weights_keep_input_layout() -> None:
    assert hue_band_weights(np.zeros((4, 5))).shape == (4, 5, len(BAND_NAMES))


def test_band_weights_wrap_between_last_and_first_band() -> None:
    weights = hue_band_weights(np.array([10.0]))[0]  # between magenta (340) and red (40)
    assert weights[BAND_NAMES.index("magenta")] == pytest.approx(0.5)
    assert weights[BAND_NAMES.index("red")] == pytest.approx(0.5)


def test_nearest_band_name() -> None:
    assert nearest_band_name(42.0) == "red"
    assert nearest_band_name(268.0) == "blue"
    assert nearest_band_name(355.0) == "magenta"


def test_chroma_mask_ignores_greys_and_keeps_colours() -> None:
    mask = chroma_mask(np.array([0.0, 50.0]))
    assert mask.tolist() == [0.0, 1.0]


def test_tonal_zones_vanish_at_black_and_white() -> None:
    shadows, highlights = tonal_zone_weights(np.array([0.0, 0.2, 0.8, 1.0]))
    assert shadows[0] == 0
    assert highlights[-1] == 0
    assert shadows[1] > highlights[1]
    assert highlights[2] > shadows[2]
