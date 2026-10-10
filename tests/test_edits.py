from __future__ import annotations

import numpy as np
import pytest

from presetfit import ColorSpace, EditPipeline, EditSettings, Pixels
from presetfit.edits import (
    Contrast,
    Edit,
    Exposure,
    HighlightsShadows,
    HueSaturationLuminance,
    ParamSpec,
    Saturation,
    SplitToning,
    ToneCurve,
    WhiteBalance,
)
from presetfit.edits.base import format_signed
from presetfit.exceptions import InvalidSettingsError

from .helpers import settings_with

ALL_EDITS: list[Edit] = [
    WhiteBalance(),
    Exposure(),
    Contrast(),
    HighlightsShadows(),
    ToneCurve(),
    Saturation(),
    HueSaturationLuminance(),
    SplitToning(),
]


def sample(space: ColorSpace, dtype: type = np.float64) -> np.ndarray:
    srgb = Pixels(np.random.default_rng(3).uniform(0.02, 0.98, (400, 3)), ColorSpace.SRGB)
    return srgb.to(space).data.astype(dtype)


def extreme_values(edit: Edit, high: bool) -> dict[str, float]:
    return {spec.name: spec.maximum if high else spec.minimum for spec in edit.params}


def lab_ramp(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    lightness = np.linspace(0, 100, 101)
    return np.stack([lightness, np.full(101, a), np.full(101, b)], axis=-1)


@pytest.mark.parametrize("edit", ALL_EDITS, ids=lambda edit: edit.name)
class TestEveryEdit:
    def test_defaults_are_the_identity(self, edit: Edit) -> None:
        data = sample(edit.space)
        assert edit.is_identity(edit.defaults())
        np.testing.assert_allclose(edit.apply(data, edit.defaults()), data, atol=1e-9)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("high", [True, False])
    def test_keeps_shape_dtype_and_finiteness(self, edit: Edit, dtype: type, high: bool) -> None:
        data = sample(edit.space, dtype).reshape(20, 20, 3)
        result = edit.apply(data, extreme_values(edit, high))
        assert result.shape == data.shape
        assert result.dtype == dtype
        assert np.all(np.isfinite(result))

    def test_does_not_modify_its_input(self, edit: Edit) -> None:
        data = sample(edit.space)
        before = data.copy()
        edit.apply(data, extreme_values(edit, high=True))
        np.testing.assert_array_equal(data, before)

    def test_validate_fills_defaults(self, edit: Edit) -> None:
        assert edit.validate({}) == edit.defaults()

    def test_validate_rejects_unknown_parameters(self, edit: Edit) -> None:
        with pytest.raises(InvalidSettingsError, match="unknown parameter"):
            edit.validate({"no_such_slider": 1.0})

    def test_validate_rejects_out_of_range_values(self, edit: Edit) -> None:
        spec = edit.params[0]
        with pytest.raises(InvalidSettingsError, match="outside"):
            edit.validate({spec.name: spec.maximum + 1})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "12", None, True])
    def test_validate_rejects_non_numbers(self, edit: Edit, bad: object) -> None:
        with pytest.raises(InvalidSettingsError):
            edit.validate({edit.params[0].name: bad})  # type: ignore[dict-item]

    def test_describe_returns_text_for_any_values(self, edit: Edit) -> None:
        for values in (edit.defaults(), extreme_values(edit, True), extreme_values(edit, False)):
            lines = edit.describe(values)
            assert lines
            assert all(line.label and line.value for line in lines)


@pytest.mark.parametrize(
    "edit", [Contrast(), HighlightsShadows(), ToneCurve()], ids=lambda e: e.name
)
@pytest.mark.parametrize("high", [True, False])
def test_tonal_edits_never_reverse_tone_order(edit: Edit, high: bool) -> None:
    lightness = edit.apply(lab_ramp(), extreme_values(edit, high))[:, 0]
    assert np.all(np.diff(lightness) > 0)
    assert lightness.min() >= 0
    assert lightness.max() <= 100


@pytest.mark.parametrize("edit", [Contrast(), HighlightsShadows()], ids=lambda e: e.name)
def test_contrast_like_edits_keep_black_and_white(edit: Edit) -> None:
    lightness = edit.apply(lab_ramp(), extreme_values(edit, high=True))[:, 0]
    assert lightness[0] == pytest.approx(0)
    assert lightness[-1] == pytest.approx(100)


def test_exposure_one_stop_doubles_light() -> None:
    data = np.full((1, 3), 0.2)
    np.testing.assert_allclose(Exposure().apply(data, {"exposure": 1.0}), 0.4)
    np.testing.assert_allclose(Exposure().apply(data, {"exposure": -1.0}), 0.1)


def test_contrast_spreads_tones_around_middle_grey() -> None:
    result = Contrast().apply(lab_ramp(), {"contrast": 60.0})[:, 0]
    assert result[25] < 25
    assert result[50] == pytest.approx(50)
    assert result[75] > 75


def test_shadows_slider_leaves_highlights_almost_alone() -> None:
    result = HighlightsShadows().apply(lab_ramp(), {"highlights": 0.0, "shadows": 100.0})[:, 0]
    assert result[25] - 25 == pytest.approx(10, abs=0.01)
    assert abs(result[90] - 90) < 0.2


def test_tone_curve_passes_through_its_control_points() -> None:
    values = {
        "point_0": 20.0,
        "point_64": 10.0,
        "point_128": -5.0,
        "point_191": 0.0,
        "point_255": -15.0,
    }
    result = ToneCurve().apply(lab_ramp(), values)[:, 0]
    assert result[0] == pytest.approx(20 / 255 * 100)
    assert result[50] == pytest.approx(50 - 5 / 255 * 100)
    assert result[100] == pytest.approx(100 - 15 / 255 * 100)


def test_tone_curve_describes_points_as_input_to_output() -> None:
    lines = ToneCurve().describe({**ToneCurve().defaults(), "point_0": 12.0})
    assert (lines[0].label, lines[0].value) == ("Point 0", "0 → 12")
    assert ToneCurve().describe(ToneCurve().defaults())[0].value == "linear (no change)"


def test_tone_curve_plot_traces_the_curve_that_is_applied() -> None:
    edit = ToneCurve()
    values = {
        "point_0": 30.0,
        "point_64": -3.0,
        "point_128": 3.0,
        "point_191": 0.0,
        "point_255": 0.0,
    }
    plot = edit.plot(values)
    assert plot.axis_max == 255
    assert plot.handles == ((0, 30), (64, 61), (128, 131), (191, 191), (255, 255))
    assert plot.points[0] == (0, 30)
    assert plot.points[-1] == (255, 255)
    inputs = np.array([x for x, _ in plot.points])
    outputs = np.array([y for _, y in plot.points])
    assert np.all(np.diff(inputs) > 0)
    assert np.all(np.diff(outputs) > 0)
    lab = np.stack([inputs / 255 * 100, np.zeros_like(inputs), np.zeros_like(inputs)], axis=-1)
    np.testing.assert_allclose(edit.apply(lab, values)[:, 0] / 100 * 255, outputs, atol=0.01)


def test_tone_curve_plot_is_the_diagonal_when_unchanged() -> None:
    plot = ToneCurve().plot(ToneCurve().defaults())
    assert all(x == pytest.approx(y) for x, y in plot.points)


def test_plain_slider_edits_have_no_plot() -> None:
    assert Exposure().plot(Exposure().defaults()) is None


def test_white_balance_warms_and_keeps_grey_luminance() -> None:
    grey = np.full((1, 3), 0.3)
    warm = WhiteBalance().apply(grey, {"temperature": 50.0, "tint": 0.0})
    assert warm[0, 0] > warm[0, 1] > warm[0, 2]
    lab_before = Pixels(grey, ColorSpace.LINEAR_RGB).to(ColorSpace.LAB).data
    lab_after = Pixels(warm, ColorSpace.LINEAR_RGB).to(ColorSpace.LAB).data
    assert lab_after[0, 0] == pytest.approx(lab_before[0, 0], abs=1e-4)
    assert lab_after[0, 2] > 5  # b* positive: yellower


def test_white_balance_tint_moves_along_green_magenta() -> None:
    grey = Pixels(np.full((1, 3), 0.3), ColorSpace.LINEAR_RGB)
    magenta = WhiteBalance().apply(grey.data, {"temperature": 0.0, "tint": 60.0})
    assert Pixels(magenta, ColorSpace.LINEAR_RGB).to(ColorSpace.LAB).data[0, 1] > 5


def test_white_balance_description_names_the_direction() -> None:
    lines = WhiteBalance().describe({"temperature": 12.0, "tint": -4.0})
    assert lines[0].value == "+12 (warmer)"
    assert lines[1].value == "-4 (more green)"


def test_saturation_minimum_removes_all_colour() -> None:
    result = Saturation().apply(sample(ColorSpace.LAB), {"saturation": -100.0})
    np.testing.assert_allclose(result[:, 1:], 0.0, atol=1e-12)


def test_hsl_changes_only_the_targeted_colour() -> None:
    edit = HueSaturationLuminance()
    red = Pixels(np.array([[0.8, 0.1, 0.1]]), ColorSpace.SRGB).to(ColorSpace.LAB).data
    blue = Pixels(np.array([[0.1, 0.2, 0.9]]), ColorSpace.SRGB).to(ColorSpace.LAB).data
    grey = np.array([[50.0, 0.0, 0.0]])
    values = {**edit.defaults(), "red_saturation": -80.0, "red_luminance": 50.0}
    changed = edit.apply(red, values)
    assert np.hypot(changed[0, 1], changed[0, 2]) < 0.5 * np.hypot(red[0, 1], red[0, 2])
    assert changed[0, 0] > red[0, 0]
    np.testing.assert_allclose(edit.apply(blue, values), blue, atol=1e-9)
    np.testing.assert_allclose(edit.apply(grey, values), grey, atol=1e-9)


def test_hsl_hue_slider_rotates_hue() -> None:
    edit = HueSaturationLuminance()
    lab = np.array([[60.0, 40.0 * np.cos(np.radians(140)), 40.0 * np.sin(np.radians(140))]])
    rotated = edit.apply(lab, {**edit.defaults(), "green_hue": 100.0})
    hue = np.degrees(np.arctan2(rotated[0, 2], rotated[0, 1]))
    assert hue == pytest.approx(170, abs=0.01)
    assert np.hypot(rotated[0, 1], rotated[0, 2]) == pytest.approx(40)


def test_hsl_description_lists_only_changed_bands() -> None:
    edit = HueSaturationLuminance()
    lines = edit.describe({**edit.defaults(), "orange_saturation": -12.0, "orange_hue": 5.0})
    assert [(line.label, line.value) for line in lines] == [("Oranges", "hue +5, saturation -12")]
    assert edit.describe(edit.defaults())[0].value == "no change"


def test_split_toning_tints_shadows_and_highlights_separately() -> None:
    values = {"shadows_a": 0.0, "shadows_b": -10.0, "highlights_a": 0.0, "highlights_b": 10.0}
    result = SplitToning().apply(lab_ramp(), values)
    assert result[20, 2] < -5  # dark tones pushed towards blue
    assert result[80, 2] > 5  # bright tones pushed towards yellow
    np.testing.assert_allclose(result[[0, -1], 1:], 0.0, atol=1e-12)
    np.testing.assert_array_equal(result[:, 0], lab_ramp()[:, 0])


def test_split_toning_description_names_colour_and_strength() -> None:
    values = {"shadows_a": -6.0, "shadows_b": -9.0, "highlights_a": 0.0, "highlights_b": 0.0}
    lines = SplitToning().describe(values)
    assert lines[0].value == "blue tint, hue 236°, strength 10.8"
    assert lines[1].value == "none"


def test_param_spec_validates_and_quantizes() -> None:
    spec = ParamSpec("x", "X", -1.0, 1.0, step=0.25)
    assert spec.quantize(0.3) == 0.25
    assert spec.quantize(7.0) == 1.0
    assert str(spec.quantize(-0.01)) == "0.0"
    with pytest.raises(ValueError, match="default"):
        ParamSpec("x", "X", 0.0, 1.0, default=2.0)
    with pytest.raises(ValueError, match="step"):
        ParamSpec("x", "X", 0.0, 1.0, step=0.0)


@pytest.mark.parametrize(
    ("value", "step", "unit", "expected"),
    [
        (12.0, 1.0, "", "+12"),
        (-0.35, 0.01, "EV", "-0.35 EV"),
        (0.0, 1.0, "", "0"),
        (-0.2, 1.0, "", "0"),
    ],
)
def test_format_signed(value: float, step: float, unit: str, expected: str) -> None:
    assert format_signed(value, step, unit) == expected


def test_subclass_must_declare_its_metadata() -> None:
    with pytest.raises(TypeError, match="must define"):

        class Incomplete(Edit):
            name = "incomplete"

            def apply(self, data, values):  # type: ignore[no-untyped-def]
                return data


class TestPipeline:
    def test_identity_settings_leave_pixels_unchanged(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        assert pipeline.run(scene, pipeline.identity_settings()) is scene

    def test_edits_run_in_declared_order(self, pipeline: EditPipeline) -> None:
        names = [edit.name for edit in pipeline.edits]
        assert names.index("exposure") < names.index("contrast") < names.index("split_toning")
        assert pipeline.index_of("white_balance") == 0
        assert pipeline.edit("saturation").label == "Saturation"

    def test_start_and_stop_split_a_run_exactly(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        settings = settings_with(
            pipeline,
            exposure={"exposure": 0.4},
            contrast={"contrast": 30.0},
            saturation={"saturation": 20.0},
        )
        whole = pipeline.run(scene, settings).to(ColorSpace.LAB).data
        cut = pipeline.index_of("contrast")
        halves = pipeline.run(pipeline.run(scene, settings, stop=cut), settings, start=cut)
        np.testing.assert_allclose(halves.to(ColorSpace.LAB).data, whole, atol=1e-4)

    def test_rejects_duplicate_edits(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            EditPipeline([Exposure(), Exposure()])

    def test_validate_fills_missing_edits_and_rejects_unknown_ones(
        self, pipeline: EditPipeline
    ) -> None:
        partial = pipeline.validate(EditSettings({"exposure": {"exposure": 1.0}}))
        assert partial.for_edit("exposure") == {"exposure": 1.0}
        assert partial.for_edit("saturation") == {"saturation": 0.0}
        with pytest.raises(InvalidSettingsError, match="unknown edit"):
            pipeline.validate(EditSettings({"sharpen": {"amount": 1.0}}))
        with pytest.raises(InvalidSettingsError, match="unknown edit"):
            pipeline.index_of("sharpen")

    def test_blend_scales_towards_neutral(self, pipeline: EditPipeline) -> None:
        settings = settings_with(
            pipeline, exposure={"exposure": 1.0}, saturation={"saturation": -40.0}
        )
        half = pipeline.blend(settings, 0.5)
        assert half.for_edit("exposure")["exposure"] == pytest.approx(0.5)
        assert half.for_edit("saturation")["saturation"] == pytest.approx(-20.0)
        assert pipeline.blend(settings, 0.0) == pipeline.identity_settings()
        assert pipeline.blend(settings, 1.0) == settings
        with pytest.raises(InvalidSettingsError, match="strength"):
            pipeline.blend(settings, 1.5)

    def test_quantize_rounds_to_slider_precision(self, pipeline: EditPipeline) -> None:
        settings = settings_with(
            pipeline, exposure={"exposure": 0.33333}, contrast={"contrast": 12.6}
        )
        rounded = pipeline.quantize(settings)
        assert rounded.for_edit("exposure")["exposure"] == pytest.approx(0.33)
        assert rounded.for_edit("contrast")["contrast"] == 13.0

    def test_describe_covers_every_edit(self, pipeline: EditPipeline) -> None:
        described = pipeline.describe(pipeline.identity_settings())
        assert [item.name for item in described] == [edit.name for edit in pipeline.edits]
        assert [item.name for item in described if item.plot] == ["tone_curve"]


class TestEditSettings:
    def test_is_immutable(self) -> None:
        settings = EditSettings({"exposure": {"exposure": 1.0}})
        with pytest.raises(TypeError):
            settings.for_edit("exposure")["exposure"] = 2.0  # type: ignore[index]

    def test_replace_returns_a_new_object(self) -> None:
        original = EditSettings({"exposure": {"exposure": 1.0}})
        changed = original.replace({"exposure": {"exposure": 2.0}})
        assert original.for_edit("exposure")["exposure"] == 1.0
        assert changed.for_edit("exposure")["exposure"] == 2.0

    def test_round_trips_through_a_dict(self) -> None:
        settings = EditSettings({"exposure": {"exposure": 1.0}})
        again = EditSettings.from_dict(settings.to_dict())
        assert again == settings
        assert hash(again) == hash(settings)
        assert settings != "not settings"
        assert "exposure" in settings
        assert "exposure" in repr(settings)

    @pytest.mark.parametrize(
        "bad", [None, [], "text", {"exposure": 1.0}, {"exposure": {1: 2.0}}, {3: {}}]
    )
    def test_from_dict_rejects_malformed_data(self, bad: object) -> None:
        with pytest.raises(InvalidSettingsError):
            EditSettings.from_dict(bad)

    def test_missing_edit_is_reported(self) -> None:
        with pytest.raises(InvalidSettingsError, match="no entry"):
            EditSettings({}).for_edit("exposure")
