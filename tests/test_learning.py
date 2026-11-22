"""Tests for the learning package. Skipped when PyTorch is not installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from scipy.interpolate import PchipInterpolator  # noqa: E402

from presetfit import ColorSpace, EditPipeline, EditSettings, Pixels  # noqa: E402
from presetfit.learning.data import (  # noqa: E402
    PackReader,
    Triplet,
    TripletDataset,
    find_overlap,
    photo_id,
    split_triplets,
    structure_signature,
    write_pack,
)
from presetfit.learning.lut import apply_lut, fit_lut, identity_lut  # noqa: E402
from presetfit.learning.model import LookTransferNet, ModelConfig, color_features  # noqa: E402
from presetfit.learning.render import TorchRenderer, _pchip_eval  # noqa: E402
from presetfit.learning.slider_fit import fit_sliders  # noqa: E402
from presetfit.learning.sliders import SliderCodec  # noqa: E402
from presetfit.learning.train import TrainConfig, Trainer, load_model  # noqa: E402
from presetfit.rendering import Renderer  # noqa: E402

from .helpers import make_scene  # noqa: E402


def random_settings(codec: SliderCodec, rng: np.random.Generator, spread: float) -> EditSettings:
    return codec.from_vector(codec.denormalize(rng.uniform(-spread, spread, codec.size)))


# --- sliders -----------------------------------------------------------------


def test_codec_round_trips_settings(pipeline: EditPipeline) -> None:
    codec = SliderCodec(pipeline)
    settings = random_settings(codec, np.random.default_rng(0), 1.0)
    assert codec.from_vector(codec.to_vector(settings)) == settings
    assert codec.size == 40


def test_codec_normalised_scale_is_minus_one_to_one(pipeline: EditPipeline) -> None:
    codec = SliderCodec(pipeline)
    assert np.allclose(codec.normalize(codec.default), 0.0)
    upper = codec.normalize(codec.maximum)
    assert np.all((upper == 1.0) | (codec.room_up == 0))
    # One-sided sliders (tone curve end points) have nothing below/above neutral.
    lower = codec.normalize(codec.minimum)
    assert np.all((lower == -1.0) | (codec.room_down == 0))
    values = codec.denormalize(np.random.default_rng(1).uniform(-1, 1, codec.size))
    assert np.allclose(codec.denormalize(codec.normalize(values)), values)


# --- differentiable renderer ------------------------------------------------


@pytest.mark.parametrize("spread", [0.3, 1.0])
def test_torch_renderer_matches_numpy_renderer(pipeline: EditPipeline, spread: float) -> None:
    renderer, torch_renderer = Renderer(pipeline), TorchRenderer(pipeline)
    image = make_scene(seed=3).data
    rng = np.random.default_rng(4)
    for _ in range(8):
        settings = random_settings(torch_renderer.codec, rng, spread)
        expected = renderer.render(Pixels(image, ColorSpace.SRGB), settings).data
        values = torch.tensor(torch_renderer.codec.to_vector(settings))[None]
        actual = torch_renderer.render(torch.tensor(image, dtype=torch.float64)[None], values)
        assert np.abs(actual[0].numpy() - expected).max() < 1e-3


def test_pchip_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    knots = np.linspace(0, 1, 5)
    for _ in range(20):
        y = knots + rng.uniform(-30, 30, 5) / 255.0
        x = np.linspace(0, 1, 101)
        ours = _pchip_eval(torch.tensor(y)[None], torch.tensor(x)[None])[0].numpy()
        assert np.allclose(ours, PchipInterpolator(knots, y)(x), atol=1e-9)


def test_renderer_gradients_are_finite_even_for_grey_pixels() -> None:
    renderer = TorchRenderer()
    image = torch.full((1, 8, 8, 3), 0.5)
    image[0, 0] = 0.0
    image[0, 1] = 1.0
    normalized = torch.zeros(1, renderer.codec.size, requires_grad=True)
    renderer.render(image, renderer.to_values(normalized)).sum().backward()
    assert normalized.grad is not None
    assert torch.isfinite(normalized.grad).all()


def test_renderer_refuses_an_unknown_pipeline() -> None:
    with pytest.raises(ValueError, match="supports the edits"):
        TorchRenderer(EditPipeline([]))


# --- LUT and slider fitting ---------------------------------------------------


def test_identity_lut_changes_nothing() -> None:
    image = make_scene(seed=6).data
    assert np.abs(apply_lut(image, identity_lut()) - image).max() < 1e-5


def test_lut_carries_a_look_from_a_small_copy_to_the_full_image(
    pipeline: EditPipeline,
) -> None:
    full = make_scene(seed=7, height=160, width=192).data
    settings = EditSettings(
        {"white_balance": {"temperature": 40.0}, "contrast": {"contrast": 30.0}}
    )
    renderer = Renderer(pipeline)
    edited = renderer.render(Pixels(full, ColorSpace.SRGB), settings).data
    small, small_edited = full[::4, ::4], edited[::4, ::4]
    result = apply_lut(full, fit_lut(small, small_edited))
    assert np.abs(result - edited).mean() < 0.01
    assert np.abs(full - edited).mean() > 0.02  # the edit itself was not tiny


def test_fit_sliders_recovers_a_known_edit(pipeline: EditPipeline) -> None:
    image = make_scene(seed=8).data
    truth = EditSettings({"exposure": {"exposure": 0.5}, "saturation": {"saturation": -30.0}})
    target = Renderer(pipeline).render(Pixels(image, ColorSpace.SRGB), truth).data
    found = fit_sliders(image, target, steps=200, ridge=0.0)
    assert found.for_edit("exposure")["exposure"] == pytest.approx(0.5, abs=0.1)
    # Several sliders can lower saturation, so compare pictures, not numbers.
    result = Renderer(pipeline).render(Pixels(image, ColorSpace.SRGB), found).data
    assert np.abs(result - target).mean() < 0.01
    assert np.abs(image - target).mean() > 0.05


# --- data ----------------------------------------------------------------------


def test_photo_id_ignores_folder_and_preset() -> None:
    assert photo_id("content_images/01/12615.tif") == "12615"
    assert photo_id("style_images/013/12615_0526.png") == "12615"


def test_same_preset_triplets_pair_photos_that_share_a_preset() -> None:
    from presetfit.learning.data import same_preset_triplets

    files = [f"content_images/00/{i:05d}.tif" for i in range(5)]
    files += [f"style_images/000/{i:05d}_0007.png" for i in range(5)]
    files += ["style_images/000/00001_0008.png", "style_images/000/00099_0007.png"]
    triplets = same_preset_triplets(files, per_preset=3)
    # Preset 0007: 3 of its 5 usable photos, paired every way = 6 examples.
    # Preset 0008 has one photo (nothing to pair); photo 99 has no original.
    assert len(triplets) == 6
    for triplet in triplets:
        assert photo_id(triplet.content) == photo_id(triplet.gt) != photo_id(triplet.reference)
        assert triplet.gt.endswith("_0007.png")
        assert triplet.reference.endswith("_0007.png")
    assert same_preset_triplets(files, per_preset=3) == triplets  # repeatable


def test_pack_round_trip(tmp_path: Path) -> None:
    image = (make_scene(seed=9).data * 255).astype(np.uint8)
    from presetfit.learning.data import shrink

    source = tmp_path / "a.png"
    from PIL import Image

    Image.fromarray(image).save(source)
    write_pack(tmp_path / "packs" / "one.pack", [("x/a.png", shrink(source, 64))])
    reader = PackReader(tmp_path / "packs")
    assert "x/a.png" in reader
    assert len(reader) == 1
    assert reader.read("x/a.png").shape == (64, 64, 3)


def test_signature_survives_tone_edits_but_tells_photos_apart(pipeline: EditPipeline) -> None:
    photo, other = make_scene(seed=10), make_scene(seed=11)
    edited = Renderer(pipeline).render(
        photo, EditSettings({"tone_curve": {"point_64": 25.0}, "exposure": {"exposure": 0.6}})
    )
    as_bytes = [(p.data * 255).astype(np.uint8) for p in (photo, edited, other)]
    a, b, c = (structure_signature(x) for x in as_bytes)
    assert float(a @ b) > 0.8
    assert float(a @ c) < 0.6
    assert find_overlap({"same": a, "different": c}, [b]) == {"same"}


def test_split_keeps_photos_apart_and_drops_excluded() -> None:
    triplets = [
        Triplet(
            f"content_images/00/{i:05d}.tif",
            f"style_images/000/{j:05d}_0001.png",
            f"style_images/000/{i:05d}_0002.png",
        )
        for i in range(40)
        for j in range(40, 44)
    ]
    train, validation = split_triplets(triplets, {"00003"}, validation_fraction=0.2)
    assert train
    assert validation
    train_photos = {p for t in train for p in t.photo_ids}
    validation_edited = {photo_id(t.content) for t in validation}
    assert not train_photos & validation_edited
    assert "00003" not in train_photos | {p for t in validation for p in t.photo_ids}


# --- model and training -------------------------------------------------------


def test_untrained_model_returns_the_photo_unchanged() -> None:
    model = LookTransferNet(ModelConfig(widths=(8, 16, 16, 16), style_dim=16), pretrained=False)
    content = torch.rand(2, 64, 64, 3)
    output = model(content, torch.rand(2, 64, 64, 3))
    assert output.shape == content.shape
    assert torch.allclose(output, content)
    assert color_features(content).shape[0] == 2


def test_model_config_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="backbone"):
        ModelConfig(backbone="alexnet")


def _tiny_dataset(folder: Path) -> tuple[Path, Path]:
    from PIL import Image

    from presetfit.learning.data import shrink

    images = folder / "images"
    images.mkdir(parents=True)
    keys = []
    for seed in range(6):
        for role in ("content", "style"):
            scene = make_scene(seed=seed, height=80, width=80).data
            if role == "style":
                scene = np.clip(scene * np.array([1.1, 1.0, 0.85]), 0, 1)
            key = f"{role}/{seed:05d}{'_0001' if role == 'style' else ''}.png"
            (images / key).parent.mkdir(exist_ok=True)
            Image.fromarray((scene * 255).astype(np.uint8)).save(images / key)
            keys.append(key)
    write_pack(folder / "packs" / "all.pack", [(k, shrink(images / k, 48)) for k in keys])
    triplets = [
        Triplet(
            f"content/{i:05d}.png", f"style/{(i + 1) % 6:05d}_0001.png", f"style/{i:05d}_0001.png"
        )
        for i in range(6)
    ]
    split = {
        "train": [vars(t) for t in triplets[:4]],
        "validation": [vars(t) for t in triplets[4:]],
    }
    (folder / "split.json").write_text(json.dumps(split))
    return folder / "packs", folder / "split.json"


def test_dataset_keeps_photo_and_answer_aligned(tmp_path: Path) -> None:
    packs, split = _tiny_dataset(tmp_path)
    rows = json.loads(split.read_text())["train"]
    dataset = TripletDataset(PackReader(packs), [Triplet(**r) for r in rows], crop=32)
    content, reference, gt = dataset[0]
    assert content.shape == reference.shape == gt.shape == (32, 32, 3)


def test_training_runs_saves_and_resumes(tmp_path: Path) -> None:
    packs, split = _tiny_dataset(tmp_path)
    config = TrainConfig(
        packs=str(packs),
        split=str(split),
        out=str(tmp_path / "run"),
        model=ModelConfig(widths=(8, 16, 16, 16), style_dim=16, encoder_size=32),
        epochs=1,
        batch_size=2,
        crop=32,
        workers=0,
        pretrained=False,
    )
    Trainer(config, device="cpu").train()
    assert (tmp_path / "run" / "best.pt").exists()
    assert len((tmp_path / "run" / "log.csv").read_text().splitlines()) == 2

    config.epochs = 2
    resumed = Trainer(config, device="cpu")
    assert resumed.epoch == 1
    resumed.train()
    assert load_model(tmp_path / "run" / "last.pt").config.widths == (8, 16, 16, 16)
