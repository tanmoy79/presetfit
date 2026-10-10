from __future__ import annotations

import numpy as np
import pytest

from presetfit import ColorSpace, EditPipeline, FitConfig, Pixels, StagedLookFitter
from presetfit.analysis import LookAnalyzer, PixelSampler
from presetfit.edits import Exposure, Saturation
from presetfit.fitting import FitResult, FitStage
from presetfit.fitting.objectives import (
    CastObjective,
    CompositeObjective,
    HueBandObjective,
    SplitToneObjective,
    ToneObjective,
)
from presetfit.rendering import Renderer

from .helpers import lab_difference, make_scene, settings_with

CONFIG = FitConfig(sample_size=4000)


@pytest.fixture(scope="module")
def fitter(pipeline: EditPipeline) -> StagedLookFitter:
    return StagedLookFitter(pipeline, config=CONFIG)


@pytest.fixture(scope="module")
def renderer(pipeline: EditPipeline) -> Renderer:
    return Renderer(pipeline)


class TestSampler:
    def test_small_images_are_used_whole(self) -> None:
        image = Pixels(np.zeros((4, 5, 3), dtype=np.float32), ColorSpace.SRGB)
        sample = PixelSampler(size=100).sample(image)
        assert sample.data.shape == (20, 3)
        assert sample.data.dtype == np.float64

    def test_large_images_are_subsampled_reproducibly(self, scene: Pixels) -> None:
        first = PixelSampler(size=500, seed=3).sample(scene).data
        second = PixelSampler(size=500, seed=3).sample(scene).data
        other = PixelSampler(size=500, seed=4).sample(scene).data
        assert first.shape == (500, 3)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other)

    def test_rejects_empty_sample_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PixelSampler(size=0)


class TestAnalyzer:
    def test_same_look_has_zero_distance(self, scene: Pixels) -> None:
        stats = LookAnalyzer().analyze(scene)
        assert stats.distance_to(stats) == 0

    def test_statistics_move_the_way_the_edit_does(
        self, scene: Pixels, pipeline: EditPipeline, renderer: Renderer
    ) -> None:
        analyzer = LookAnalyzer()
        before = analyzer.analyze(scene)
        brighter = analyzer.analyze(
            renderer.render(scene, settings_with(pipeline, exposure={"exposure": 0.7}))
        )
        warmer = analyzer.analyze(
            renderer.render(scene, settings_with(pipeline, white_balance={"temperature": 60.0}))
        )
        duller = analyzer.analyze(
            renderer.render(scene, settings_with(pipeline, saturation={"saturation": -60.0}))
        )
        assert np.all(brighter.lightness_quantiles > before.lightness_quantiles)
        assert warmer.mean_cast[1] > before.mean_cast[1] + 3
        assert np.all(duller.color_spread < 0.5 * before.color_spread)
        assert before.distance_to(brighter) > 5

    def test_band_coverage_reflects_image_content(self) -> None:
        red = Pixels(np.tile([0.8, 0.1, 0.1], (50, 1)), ColorSpace.SRGB)
        stats = LookAnalyzer().analyze(red)
        assert stats.band_coverage[0] > 0.9  # the red band
        assert stats.band_coverage[5] == 0  # the blue band


class TestObjectives:
    def test_band_objective_skips_colours_missing_from_either_image(self) -> None:
        analyzer = LookAnalyzer()
        red = analyzer.analyze(Pixels(np.tile([0.8, 0.1, 0.1], (50, 1)), ColorSpace.SRGB))
        blue = analyzer.analyze(Pixels(np.tile([0.1, 0.2, 0.9], (50, 1)), ColorSpace.SRGB))
        objective = HueBandObjective()
        assert len(objective.skipped_bands(red, blue)) == 8
        assert "red" not in objective.skipped_bands(red, red)
        assert "red_hue" not in objective.frozen_parameters(red, red)
        assert "blue_saturation" in objective.frozen_parameters(red, red)
        assert not np.any(objective.residuals(red, blue, red))

    def test_composite_combines_parts_with_weights(
        self, scene: Pixels, other_scene: Pixels
    ) -> None:
        analyzer = LookAnalyzer()
        current, target = analyzer.analyze(scene), analyzer.analyze(other_scene)
        tone, cast = ToneObjective(), CastObjective()
        combined = CompositeObjective((tone, 1.0), (cast, 3.0)).residuals(current, target, current)
        expected = np.concatenate(
            [
                tone.residuals(current, target, current),
                3.0 * cast.residuals(current, target, current),
            ]
        )
        np.testing.assert_allclose(combined, expected)
        with pytest.raises(ValueError, match="at least one"):
            CompositeObjective()

    def test_composite_passes_through_skipped_bands(self) -> None:
        analyzer = LookAnalyzer()
        red = analyzer.analyze(Pixels(np.tile([0.8, 0.1, 0.1], (50, 1)), ColorSpace.SRGB))
        composite = CompositeObjective((HueBandObjective(), 1.0), (ToneObjective(), 1.0))
        assert "blue" in composite.skipped_bands(red, red)
        assert "blue_hue" in composite.frozen_parameters(red, red)
        assert ToneObjective().skipped_bands(red, red) == ()

    def test_split_tone_objective_distrusts_missing_zones(self) -> None:
        analyzer = LookAnalyzer()
        black = analyzer.analyze(Pixels(np.zeros((50, 3)), ColorSpace.SRGB))
        tinted = analyzer.analyze(Pixels(np.tile([0.2, 0.3, 0.6], (50, 1)), ColorSpace.SRGB))
        assert not np.any(SplitToneObjective().residuals(tinted, black, tinted))


class TestStagedFitter:
    def test_matching_a_photo_to_itself_changes_nothing(
        self, fitter: StagedLookFitter, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        result = fitter.fit(scene, scene)
        assert result.settings == pipeline.identity_settings()
        assert result.distance_before == 0
        assert result.distance_after == 0
        assert result.improvement == 0

    @pytest.mark.parametrize(
        "edits",
        [
            {"exposure": {"exposure": 0.6}},
            {"exposure": {"exposure": -0.5}, "contrast": {"contrast": 30.0}},
            {
                "white_balance": {"temperature": 25.0, "tint": -10.0},
                "saturation": {"saturation": -30.0},
            },
            {
                "tone_curve": {"point_0": 20.0, "point_255": -10.0},
                "split_toning": {"shadows_b": -8.0, "highlights_b": 8.0},
            },
        ],
        ids=["exposure", "exposure+contrast", "white-balance+saturation", "faded+split-toned"],
    )
    def test_reproduces_a_known_edit_of_the_same_photo(
        self,
        fitter: StagedLookFitter,
        renderer: Renderer,
        pipeline: EditPipeline,
        scene: Pixels,
        edits: dict[str, dict[str, float]],
    ) -> None:
        full = {
            name: {**pipeline.edit(name).defaults(), **values} for name, values in edits.items()
        }
        reference = renderer.render(scene, settings_with(pipeline, **full))

        result = fitter.fit(scene, reference)
        matched = renderer.render(scene, result.settings)

        assert result.distance_after < max(0.25, 0.1 * result.distance_before)
        # Under 1 Delta E is below what the eye can see; subtle looks are held to
        # that, strong looks to closing 80% of the per-pixel gap.
        remaining = lab_difference(matched, reference)
        assert remaining < max(1.0, 0.2 * lab_difference(scene, reference))

    def test_recovers_a_single_slider_value(
        self, fitter: StagedLookFitter, renderer: Renderer, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        reference = renderer.render(scene, settings_with(pipeline, exposure={"exposure": 0.6}))
        result = fitter.fit(scene, reference)
        assert result.settings.for_edit("exposure")["exposure"] == pytest.approx(0.6, abs=0.1)

    def test_moves_towards_the_look_of_a_different_photo(
        self, fitter: StagedLookFitter, renderer: Renderer, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        graded = renderer.render(
            make_scene(seed=9),
            settings_with(pipeline, exposure={"exposure": -0.6}, saturation={"saturation": -50.0}),
        )
        result = fitter.fit(scene, graded)
        assert result.distance_after < 0.5 * result.distance_before
        assert result.settings.for_edit("exposure")["exposure"] < 0
        assert result.settings.for_edit("saturation")["saturation"] < 0

    def test_is_deterministic(
        self, fitter: StagedLookFitter, scene: Pixels, other_scene: Pixels
    ) -> None:
        assert fitter.fit(scene, other_scene).settings == fitter.fit(scene, other_scene).settings

    def test_reported_settings_are_rounded_and_valid(
        self, fitter: StagedLookFitter, pipeline: EditPipeline, scene: Pixels, other_scene: Pixels
    ) -> None:
        settings = fitter.fit(scene, other_scene).settings
        assert pipeline.validate(settings) == settings
        assert pipeline.quantize(settings) == settings

    def test_reach_limits_how_far_a_slider_may_go(self, scene: Pixels) -> None:
        pipeline = EditPipeline([Exposure()])
        stage = FitStage("exposure", ("exposure",), ToneObjective(), reach=0.1)
        fitter = StagedLookFitter(pipeline, stages=[stage], config=CONFIG)
        white = Pixels(np.ones((20, 20, 3), dtype=np.float32), ColorSpace.SRGB)
        assert fitter.fit(scene, white).settings.for_edit("exposure")["exposure"] == pytest.approx(
            0.3
        )

    def test_ridge_holds_sliders_near_neutral(self, scene: Pixels) -> None:
        pipeline = EditPipeline([Exposure()])
        target = Renderer(pipeline).render(
            scene, settings_with(pipeline, exposure={"exposure": 0.5})
        )

        def fitted(ridge: float) -> float:
            stage = FitStage("exposure", ("exposure",), ToneObjective(), ridge=ridge)
            fit = StagedLookFitter(pipeline, stages=[stage], config=CONFIG).fit(scene, target)
            return fit.settings.for_edit("exposure")["exposure"]

        assert fitted(0.0) == pytest.approx(0.5, abs=0.02)
        assert 0 <= fitted(1e6) < 0.1

    def test_stage_edits_must_be_adjacent(self, pipeline: EditPipeline) -> None:
        stage = FitStage("gap", ("white_balance", "saturation"), ToneObjective())
        with pytest.raises(ValueError, match="adjacent"):
            StagedLookFitter(pipeline, stages=[stage])

    def test_stage_names_must_be_unique(self) -> None:
        pipeline = EditPipeline([Exposure(), Saturation()])
        stages = [
            FitStage("same", ("exposure",), ToneObjective()),
            FitStage("same", ("saturation",), ToneObjective()),
        ]
        with pytest.raises(ValueError, match="unique"):
            StagedLookFitter(pipeline, stages=stages)

    @pytest.mark.parametrize(
        "kwargs", [{"edit_names": ()}, {"ridge": -1.0}, {"reach": 0.0}, {"reach": 1.5}]
    )
    def test_stage_rejects_bad_configuration(self, kwargs: dict[str, object]) -> None:
        arguments: dict[str, object] = {"edit_names": ("exposure",), **kwargs}
        with pytest.raises(ValueError, match=r"stage|ridge|reach"):
            FitStage("bad", objective=ToneObjective(), **arguments)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [{"sample_size": 10}, {"coarse_passes": 0}, {"max_iterations": 0}, {"tolerance": 0.0}],
    )
    def test_config_rejects_bad_values(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be"):
            FitConfig(**kwargs)  # type: ignore[arg-type]


def test_fit_result_improvement_is_bounded(pipeline: EditPipeline) -> None:
    settings = pipeline.identity_settings()
    assert FitResult(settings, 10.0, 2.5).improvement == pytest.approx(0.75)
    assert FitResult(settings, 10.0, 12.0).improvement == 0
    assert FitResult(settings, 0.0, 0.0).improvement == 0
