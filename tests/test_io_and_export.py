from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from presetfit import (
    ColorSpace,
    CubeLutExporter,
    EditPipeline,
    ImageReader,
    ImageWriter,
    Pixels,
    PresetCodec,
)
from presetfit.exceptions import (
    ImageDecodeError,
    ImageTooLargeError,
    InvalidSettingsError,
    UnsupportedImageError,
)
from presetfit.rendering import Renderer

from .helpers import encode, settings_with


class TestImageReader:
    def test_reads_bytes_paths_and_streams_alike(self, scene: Pixels, tmp_path: Path) -> None:
        data = encode(scene)
        path = tmp_path / "photo.png"
        path.write_bytes(data)
        reader = ImageReader()
        from_bytes = reader.read(data)
        np.testing.assert_array_equal(reader.read(path).data, from_bytes.data)
        np.testing.assert_array_equal(reader.read(str(path)).data, from_bytes.data)
        np.testing.assert_array_equal(reader.read(io.BytesIO(data)).data, from_bytes.data)

    def test_returns_float32_srgb_in_unit_range(self, scene: Pixels) -> None:
        image = ImageReader().read(encode(scene))
        assert image.space is ColorSpace.SRGB
        assert image.data.dtype == np.float32
        assert image.data.shape == scene.data.shape
        np.testing.assert_allclose(image.data, scene.data, atol=1 / 255)

    def test_applies_exif_orientation(self) -> None:
        picture = Image.new("RGB", (40, 20), (200, 10, 10))
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", exif=exif)
        assert ImageReader().read(buffer.getvalue()).data.shape == (40, 20, 3)

    def test_reads_camera_jpegs_with_an_embedded_preview(self, scene: Pixels) -> None:
        eight_bit = np.rint(scene.data * 255).astype(np.uint8)
        photo = Image.fromarray(eight_bit)
        buffer = io.BytesIO()
        photo.save(buffer, format="MPO", save_all=True, append_images=[photo.resize((16, 12))])
        with Image.open(io.BytesIO(buffer.getvalue())) as stored:
            assert stored.format == "MPO"
        image = ImageReader().read(buffer.getvalue())
        assert image.data.shape == scene.data.shape  # the full photo, not the preview

    def test_flattens_transparency_onto_white(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(buffer, format="PNG")
        np.testing.assert_array_equal(ImageReader().read(buffer.getvalue()).data, 1.0)

    def test_converts_greyscale_to_three_channels(self) -> None:
        buffer = io.BytesIO()
        Image.new("L", (4, 4), 128).save(buffer, format="PNG")
        image = ImageReader().read(buffer.getvalue())
        assert image.data.shape == (4, 4, 3)
        assert np.ptp(image.data) == 0

    @pytest.mark.parametrize("junk", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40])
    def test_rejects_data_that_is_not_an_image(self, junk: bytes) -> None:
        with pytest.raises(ImageDecodeError):
            ImageReader().read(junk)

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ImageDecodeError, match="no such file"):
            ImageReader().read(tmp_path / "missing.jpg")

    def test_rejects_unsupported_formats(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="GIF")
        with pytest.raises(UnsupportedImageError, match="GIF"):
            ImageReader().read(buffer.getvalue())

    def test_rejects_images_over_the_pixel_limit(self, scene: Pixels) -> None:
        with pytest.raises(ImageTooLargeError, match="limit"):
            ImageReader(max_pixels=100).read(encode(scene))

    def test_rejects_a_nonsensical_limit(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ImageReader(max_pixels=0)


class TestImageWriter:
    @pytest.mark.parametrize(
        "name", ["out.png", "out.jpg", "nested/dir/out.webp", "out.TIFF", "out.bmp"]
    )
    def test_writes_the_format_named_by_the_extension(
        self, scene: Pixels, tmp_path: Path, name: str
    ) -> None:
        path = ImageWriter().write(scene, tmp_path / name)
        assert ImageReader().read(path).data.shape == scene.data.shape

    def test_png_round_trip_is_exact_to_eight_bits(self, scene: Pixels, tmp_path: Path) -> None:
        path = ImageWriter().write(scene, tmp_path / "out.png")
        np.testing.assert_allclose(ImageReader().read(path).data, scene.data, atol=0.5 / 255 + 1e-6)

    def test_output_carries_no_metadata(self, scene: Pixels) -> None:
        with Image.open(io.BytesIO(ImageWriter().encode(scene, "JPEG"))) as written:
            assert not written.getexif()

    def test_accepts_pixels_in_any_colour_space(self, scene: Pixels) -> None:
        from_lab = ImageWriter().encode(scene.to(ColorSpace.LAB), "PNG")
        np.testing.assert_allclose(ImageReader().read(from_lab).data, scene.data, atol=1.5 / 255)

    def test_rejects_unknown_extensions_and_formats(self, scene: Pixels, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedImageError, match="infer"):
            ImageWriter().write(scene, tmp_path / "out.xyz")
        with pytest.raises(UnsupportedImageError, match="GIF"):
            ImageWriter().encode(scene, "GIF")

    def test_max_edge_shrinks_the_picture_and_keeps_its_shape(self, scene: Pixels) -> None:
        small = ImageReader().read(ImageWriter().encode(scene, "PNG", max_edge=32))
        assert small.data.shape == (24, 32, 3)  # the 96 x 128 scene, same proportions
        untouched = ImageReader().read(ImageWriter().encode(scene, "PNG", max_edge=5000))
        assert untouched.data.shape == scene.data.shape
        with pytest.raises(ValueError, match="max_edge"):
            ImageWriter().encode(scene, max_edge=0)

    def test_rejects_pixel_lists_and_bad_quality(self, scene: Pixels) -> None:
        with pytest.raises(ValueError, match="full images"):
            ImageWriter().encode(scene.flattened())
        with pytest.raises(ValueError, match="quality"):
            ImageWriter(jpeg_quality=0)


class TestRenderer:
    def test_strips_give_the_same_result_as_one_pass(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        settings = settings_with(
            pipeline,
            exposure={"exposure": 0.3},
            hsl={**pipeline.edit("hsl").defaults(), "blue_saturation": 40.0},
            split_toning={
                "shadows_a": 3.0,
                "shadows_b": -6.0,
                "highlights_a": 0.0,
                "highlights_b": 5.0,
            },
        )
        whole = Renderer(pipeline, chunk_pixels=10**9).render(scene, settings)
        strips = Renderer(pipeline, chunk_pixels=300).render(scene, settings)
        np.testing.assert_array_equal(strips.data, whole.data)

    def test_output_is_displayable_float32_srgb(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        result = Renderer(pipeline).render(
            scene, settings_with(pipeline, exposure={"exposure": 3.0})
        )
        assert result.space is ColorSpace.SRGB
        assert result.data.dtype == np.float32
        assert result.data.min() >= 0
        assert result.data.max() <= 1

    def test_identity_settings_reproduce_the_input(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        result = Renderer(pipeline).render(scene, pipeline.identity_settings())
        np.testing.assert_array_equal(result.data, scene.data)

    def test_rejects_invalid_settings_and_inputs(
        self, pipeline: EditPipeline, scene: Pixels
    ) -> None:
        with pytest.raises(InvalidSettingsError):
            Renderer(pipeline).render(
                scene, settings_with(pipeline).replace({"exposure": {"exposure": 99.0}})
            )
        with pytest.raises(ValueError, match="height, width"):
            Renderer(pipeline).render(scene.flattened(), pipeline.identity_settings())
        with pytest.raises(ValueError, match="positive"):
            Renderer(pipeline, chunk_pixels=0)


class TestPresetCodec:
    def test_round_trips_settings(self, pipeline: EditPipeline, tmp_path: Path) -> None:
        codec = PresetCodec(pipeline)
        settings = settings_with(
            pipeline, exposure={"exposure": 0.42}, saturation={"saturation": -12.0}
        )
        path = codec.save(settings, tmp_path / "deep" / "look.json")
        assert codec.load(path) == settings
        document = json.loads(path.read_text())
        assert document["format"] == "presetfit-preset"
        assert document["version"] == 1

    def test_missing_edits_take_defaults(self, pipeline: EditPipeline) -> None:
        text = json.dumps(
            {"format": "presetfit-preset", "version": 1, "settings": {"exposure": {"exposure": 1}}}
        )
        assert PresetCodec(pipeline).loads(text) == settings_with(
            pipeline, exposure={"exposure": 1.0}
        )

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            b"\xff\xfe",
            "[]",
            '{"format": "something-else", "version": 1, "settings": {}}',
            '{"format": "presetfit-preset", "version": 99, "settings": {}}',
            '{"format": "presetfit-preset", "version": 1}',
            '{"format": "presetfit-preset", "version": 1, "settings": {"exposure": {"exposure": 9}}}',  # noqa: E501
            '{"format": "presetfit-preset", "version": 1, "settings": {"sharpen": {}}}',
        ],
    )
    def test_rejects_invalid_documents(self, pipeline: EditPipeline, text: str | bytes) -> None:
        with pytest.raises(InvalidSettingsError):
            PresetCodec(pipeline).loads(text)

    def test_reports_unreadable_files(self, pipeline: EditPipeline, tmp_path: Path) -> None:
        with pytest.raises(InvalidSettingsError, match="cannot read"):
            PresetCodec(pipeline).load(tmp_path / "missing.json")


class TestCubeLutExporter:
    @staticmethod
    def parse(cube: str) -> tuple[list[str], np.ndarray]:
        lines = cube.strip().splitlines()
        header = [line for line in lines if line[0].isalpha()]
        table = np.array([[float(x) for x in line.split()] for line in lines[len(header) :]])
        return header, table

    def test_identity_lut_maps_every_colour_to_itself(self, pipeline: EditPipeline) -> None:
        header, table = self.parse(
            CubeLutExporter(pipeline, size=5).export(pipeline.identity_settings())
        )
        assert "LUT_3D_SIZE 5" in header
        assert table.shape == (125, 3)
        np.testing.assert_allclose(table[0], [0, 0, 0])
        np.testing.assert_allclose(table[1], [0.25, 0, 0])  # red changes fastest
        np.testing.assert_allclose(table[5], [0, 0.25, 0])
        np.testing.assert_allclose(table[25], [0, 0, 0.25])
        np.testing.assert_allclose(table[-1], [1, 1, 1])

    def test_lut_agrees_with_the_renderer(self, pipeline: EditPipeline) -> None:
        settings = settings_with(
            pipeline, exposure={"exposure": 0.5}, saturation={"saturation": -40.0}
        )
        _, table = self.parse(CubeLutExporter(pipeline, size=3).export(settings))
        grid = np.array([[r, g, b] for b in (0, 0.5, 1) for g in (0, 0.5, 1) for r in (0, 0.5, 1)])
        image = Pixels(grid.reshape(1, 27, 3).astype(np.float32), ColorSpace.SRGB)
        rendered = Renderer(pipeline).render(image, settings).data.reshape(27, 3)
        np.testing.assert_allclose(table, rendered, atol=1e-4)

    def test_title_is_sanitised_and_file_is_written(
        self, pipeline: EditPipeline, tmp_path: Path
    ) -> None:
        exporter = CubeLutExporter(pipeline, size=2)
        assert exporter.export(pipeline.identity_settings(), 'my "look"\n').startswith(
            "TITLE \"my 'look' \""
        )
        path = exporter.save(pipeline.identity_settings(), tmp_path / "luts" / "teal.cube")
        assert path.read_text().startswith('TITLE "teal"')

    @pytest.mark.parametrize("size", [1, 66])
    def test_rejects_unreasonable_sizes(self, pipeline: EditPipeline, size: int) -> None:
        with pytest.raises(ValueError, match="between"):
            CubeLutExporter(pipeline, size=size)
