from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from presetfit import (
    EditSettings,
    FitResult,
    ImageReader,
    ImageWriter,
    LookFitter,
    Pixels,
    PresetFitService,
    TextReportFormatter,
    default_pipeline,
)
from presetfit import __version__ as package_version
from presetfit.cli import main
from presetfit.exceptions import InvalidSettingsError
from presetfit.rendering import Renderer
from presetfit.report import ReportBuilder

from .helpers import lab_difference, make_scene, settings_with


class FixedFitter(LookFitter):
    """A fitter that always returns the same settings, to test the service alone."""

    def __init__(self, settings: EditSettings, skipped: tuple[str, ...] = ()) -> None:
        self._settings = settings
        self._skipped = skipped

    def fit(self, source: Pixels, reference: Pixels) -> FitResult:
        return FitResult(self._settings, 10.0, 6.0, skipped_bands=self._skipped)


@pytest.fixture
def photos(tmp_path: Path) -> tuple[Path, Path]:
    writer = ImageWriter()
    pipeline = default_pipeline()
    graded = Renderer(pipeline).render(
        make_scene(2),
        settings_with(pipeline, exposure={"exposure": -0.5}, saturation={"saturation": -40.0}),
    )
    return writer.write(make_scene(1), tmp_path / "mine.png"), writer.write(
        graded, tmp_path / "ref.png"
    )


class TestService:
    def test_match_returns_image_settings_and_report(
        self, service: PresetFitService, scene: Pixels
    ) -> None:
        reference = service.apply(
            scene, settings_with(service.pipeline, exposure={"exposure": 0.5})
        )
        result = service.match(scene, reference)
        assert result.image.data.shape == scene.data.shape
        assert lab_difference(result.image, reference) < 1.0
        assert result.settings == result.fit.settings
        assert result.report.improvement > 0.9

    def test_image_is_rendered_from_the_reported_settings(
        self, service: PresetFitService, scene: Pixels, other_scene: Pixels
    ) -> None:
        result = service.match(scene, other_scene, strength=0.6)
        np.testing.assert_array_equal(service.apply(scene, result.settings).data, result.image.data)

    def test_strength_scales_the_fitted_settings(self, scene: Pixels) -> None:
        pipeline = default_pipeline()
        fitted = settings_with(pipeline, exposure={"exposure": 1.0}, contrast={"contrast": 40.0})
        service = PresetFitService(pipeline, FixedFitter(fitted), Renderer(pipeline))
        half = service.match(scene, scene, strength=0.5)
        assert half.settings.for_edit("exposure")["exposure"] == pytest.approx(0.5)
        assert half.settings.for_edit("contrast")["contrast"] == pytest.approx(20.0)
        assert any("50% strength" in note for note in half.report.notes)
        assert service.match(scene, scene, strength=0.0).settings == pipeline.identity_settings()
        with pytest.raises(InvalidSettingsError, match="strength"):
            service.match(scene, scene, strength=2.0)

    def test_report_explains_skipped_colours_and_weak_matches(self, scene: Pixels) -> None:
        pipeline = default_pipeline()
        service = PresetFitService(
            pipeline,
            FixedFitter(pipeline.identity_settings(), ("green", "blue")),
            Renderer(pipeline),
        )
        notes = " ".join(service.match(scene, scene).report.notes)
        assert "green and blue" in notes
        assert "those colours" in notes
        assert "partly matched" in notes  # FixedFitter reports only 40% of the gap closed

    def test_export_lut_and_presets_use_the_service_pipeline(
        self, service: PresetFitService
    ) -> None:
        settings = settings_with(service.pipeline, exposure={"exposure": 0.2})
        assert service.export_lut(settings, title="demo", size=2).startswith('TITLE "demo"')
        assert service.presets.loads(service.presets.dumps(settings)) == settings


class TestReport:
    def test_text_report_lists_every_edit_and_the_score(self) -> None:
        pipeline = default_pipeline()
        settings = settings_with(
            pipeline, exposure={"exposure": 0.35}, white_balance={"temperature": 8.0, "tint": 0.0}
        )
        fit = FitResult(settings, 12.0, 3.0, skipped_bands=("purple",))
        report = ReportBuilder(pipeline).build(fit, settings)
        text = TextReportFormatter().format(report)
        assert "+0.35 EV" in text
        assert "+8 (warmer)" in text
        assert "12.00 -> 3.00 (75% of the gap closed)" in text
        assert "skipped for purple" in text
        assert "that colour." in text
        for edit in pipeline.edits:
            assert edit.label in text

    def test_report_serialises_to_plain_data(self) -> None:
        pipeline = default_pipeline()
        fit = FitResult(pipeline.identity_settings(), 4.0, 1.0)
        data = ReportBuilder(pipeline).build(fit, fit.settings).to_dict()
        assert json.loads(json.dumps(data)) == data
        curves = {section["name"]: section["curve"] for section in data["sections"]}
        assert curves["exposure"] is None
        assert curves["tone_curve"]["axis_max"] == 255
        assert len(curves["tone_curve"]["handles"]) == 5
        assert curves["tone_curve"]["points"][-1] == [255, 255]
        assert data["improvement"] == 0.75
        assert [section["name"] for section in data["sections"]] == [e.name for e in pipeline.edits]


class TestCli:
    def test_match_writes_every_requested_output(
        self, photos: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mine, reference = photos
        out, preset, lut, report = (
            tmp_path / name for name in ("o/out.jpg", "o/look.json", "o/look.cube", "o/r.txt")
        )
        code = main(["match", str(mine), str(reference), "-o", str(out), "--preset", str(preset),
                     "--lut", str(lut), "--report", str(report)])  # fmt: skip
        printed = capsys.readouterr().out
        assert code == 0
        assert "EDITS APPLIED" in printed
        assert f"Saved {out}" in printed
        assert ImageReader().read(out).data.shape == ImageReader().read(mine).data.shape
        assert lut.read_text().startswith('TITLE "look"')
        assert report.read_text().startswith("EDITS APPLIED")
        exposure = json.loads(preset.read_text())["settings"]["exposure"]["exposure"]
        assert exposure < 0

    def test_match_defaults_the_output_name_and_prints_json(
        self, photos: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        mine, reference = photos
        assert main(["match", str(mine), str(reference), "--json", "--strength", "0.5"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["output"] == str(mine.with_name("mine_matched.jpg"))
        assert Path(payload["output"]).exists()
        assert {"settings", "report"} <= payload.keys()

    def test_apply_reuses_a_saved_preset(self, photos: tuple[Path, Path], tmp_path: Path) -> None:
        mine, reference = photos
        preset, first, second = tmp_path / "look.json", tmp_path / "a.png", tmp_path / "b.png"
        assert (
            main(["match", str(mine), str(reference), "-o", str(first), "--preset", str(preset)])
            == 0
        )
        assert main(["apply", str(mine), "--preset", str(preset), "-o", str(second)]) == 0
        np.testing.assert_array_equal(
            ImageReader().read(first).data, ImageReader().read(second).data
        )

    def test_errors_are_reported_without_a_traceback(
        self, photos: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mine, _ = photos
        junk = tmp_path / "junk.jpg"
        junk.write_bytes(b"not an image")
        assert main(["match", str(mine), str(junk)]) == 1
        assert capsys.readouterr().err.startswith("error: the file could not be read")
        assert main(["apply", str(mine), "--preset", str(tmp_path / "none.json")]) == 1
        assert "cannot read preset" in capsys.readouterr().err
        assert main(["match", str(mine), str(mine), "-o", str(tmp_path / "out.xyz")]) == 1
        assert "cannot infer an image format" in capsys.readouterr().err

    @pytest.mark.parametrize("bad", ["1.5", "-0.1", "lots"])
    def test_rejects_bad_strength(self, bad: str, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["match", "a.jpg", "b.jpg", "--strength", bad])
        assert exit_info.value.code == 2
        assert "strength must be" in capsys.readouterr().err

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert package_version in capsys.readouterr().out
