"""End-to-end tests: data preparation, using a trained network, and the benchmark.

Hugging Face is replaced by a local folder, so these run offline.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("skimage")

import huggingface_hub  # noqa: E402
from PIL import Image  # noqa: E402

from presetfit import ColorSpace, EditSettings, Pixels  # noqa: E402
from presetfit.evaluation import benchmark  # noqa: E402
from presetfit.learning import prepare as prepare_module  # noqa: E402
from presetfit.learning.data import PackReader  # noqa: E402
from presetfit.learning.model import LookTransferNet, ModelConfig  # noqa: E402
from presetfit.learning.train import main as train_main  # noqa: E402
from presetfit.learning.transfer import LearnedLookTransfer  # noqa: E402
from presetfit.rendering import Renderer  # noqa: E402

from .helpers import make_scene  # noqa: E402

TINY = ModelConfig(widths=(8, 16, 16, 16), style_dim=16, encoder_size=32)


def _save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def _render(image: np.ndarray, settings: EditSettings) -> np.ndarray:
    from presetfit.edits import default_pipeline

    return Renderer(default_pipeline()).render(Pixels(image, ColorSpace.SRGB), settings).data


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local stand-in for the two Hugging Face datasets."""
    hub = tmp_path / "hub"
    big = hub / "ToneStyle/TST100K"
    records = []
    for i in range(6):
        photo = make_scene(seed=20 + i, height=72, width=96).data
        _save(photo, big / f"content_images/00/{i:05d}.png")
        _save(
            _render(photo, EditSettings({"saturation": {"saturation": -40.0}})),
            big / f"style_images/000/{i:05d}_0001.png",
        )
        _save(
            _render(photo, EditSettings({"exposure": {"exposure": 0.4}})),
            big / f"style_images/000/{i:05d}_0002.png",
        )
        records.append(
            {
                "content": f"content_images/00/{i:05d}.png",
                "reference": f"style_images/000/{(i + 1) % 6:05d}_0001.png",
                "gt": f"style_images/000/{i:05d}_0002.png",
            }
        )
    (big / "triplet.json").write_text(json.dumps(records))
    # The test set reuses photo 00000 (edited), so it must be left out of training.
    test = hub / "ToneStyle/TST2K"
    reused = make_scene(seed=20, height=72, width=96).data
    _save(
        _render(reused, EditSettings({"contrast": {"contrast": 40.0}})), test / "0000/content.png"
    )
    _save(make_scene(seed=90).data, test / "0000/reference.png")
    _save(
        _render(
            reused, EditSettings({"contrast": {"contrast": 40.0}, "exposure": {"exposure": 0.3}})
        ),
        test / "0000/gt.png",
    )

    def hf_hub_download(repo: str, filename: str, repo_type: str, local_dir: Path) -> str:
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hub / repo / filename, target)
        return str(target)

    def snapshot_download(
        repo: str,
        repo_type: str,
        local_dir: Path,
        allow_patterns: list[str] | None = None,
        **_: object,
    ) -> str:
        root = hub / repo
        for file in root.rglob("*"):
            relative = file.relative_to(root).as_posix()
            if file.is_file() and (
                allow_patterns is None or any(fnmatch.fnmatch(relative, p) for p in allow_patterns)
            ):
                target = Path(local_dir) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, target)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return hub


def test_prepare_packs_data_and_leaves_out_test_photos(fake_hub: Path, tmp_path: Path) -> None:
    out = tmp_path / "data"
    counts = prepare_module.prepare(out, workers=1)
    assert counts["packed_triplets"] == 6
    assert counts["test_triplets"] == 1
    split = json.loads((out / "split.json").read_text())
    assert split["excluded"] == ["00000"]
    used = {p for row in split["train"] + split["validation"] for p in row.values()}
    assert not any("00000" in path for path in used)
    assert len(PackReader(out / "packs")) == 18
    assert (out / "tst2k_512" / "0000" / "gt.png").exists()
    # A second run finds everything done and changes nothing.
    assert prepare_module.prepare(out, workers=1) == counts


def test_prepare_command_line(
    fake_hub: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        prepare_module.main(["--out", str(tmp_path / "d"), "--limit", "4", "--workers", "1"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["triplets"] == 4


def _checkpoint(path: Path) -> Path:
    model = LookTransferNet(TINY, pretrained=False)
    torch.save({"model": model.state_dict(), "config": {"model": TINY.to_dict()}}, path)
    return path


def test_transfer_works_on_full_size_photos(tmp_path: Path) -> None:
    transfer = LearnedLookTransfer.load(_checkpoint(tmp_path / "net.pt"))
    photo, reference = make_scene(seed=30, height=90, width=140).data, make_scene(seed=31).data
    with_lut = transfer.transfer(photo, reference, "lut")
    assert with_lut.image.shape == photo.shape
    assert with_lut.settings is not None
    # An untrained network changes nothing, so neither does its colour table.
    assert np.abs(with_lut.image - photo).mean() < 0.01
    with_sliders = transfer.transfer(photo, reference, "sliders")
    assert with_sliders.image.shape == photo.shape
    assert transfer.transfer(photo, reference, "lut", with_settings=False).settings is None
    with pytest.raises(ValueError, match="mode"):
        transfer.transfer(photo, reference, "paint")


def _test_set(folder: Path) -> Path:
    for index in range(2):
        photo = make_scene(seed=40 + index).data
        _save(photo, folder / f"{index:04d}/content.png")
        _save(make_scene(seed=50 + index).data, folder / f"{index:04d}/reference.png")
        _save(
            _render(photo, EditSettings({"exposure": {"exposure": 0.3}})),
            folder / f"{index:04d}/gt.png",
        )
    return folder


def test_benchmark_scores_every_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _test_set(tmp_path / "tst2k")
    out = tmp_path / "results.csv"
    code = benchmark.main(
        [
            str(data),
            "--out",
            str(out),
            "--workers",
            "1",
            "--size",
            "64",
            "--methods",
            "identity,reinhard,learned,learned_sliders",
            "--checkpoint",
            str(_checkpoint(tmp_path / "net.pt")),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    for method in ("identity", "reinhard", "learned_sliders"):
        assert method in printed
    rows = out.read_text().splitlines()
    assert len(rows) == 1 + 2 * 4
    assert not any(row.endswith("Error") for row in rows)


def test_benchmark_reports_failures_and_refuses_bad_options(tmp_path: Path) -> None:
    data = _test_set(tmp_path / "tst2k")
    rows = benchmark.evaluate((benchmark.find_triplets(data)[0], 64, ("learned",), None, "cpu"))
    assert "checkpoint" in str(rows[0]["error"])
    assert "-" in benchmark.summarise(rows, ("learned",))
    with pytest.raises(SystemExit):
        benchmark.main([str(data), "--methods", "learned"])
    with pytest.raises(SystemExit):
        benchmark.main([str(data), "--methods", "magic"])
    with pytest.raises(SystemExit):
        benchmark.main([str(tmp_path / "empty")])


def test_train_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from presetfit.learning import train as train_module

    from .test_learning import _tiny_dataset

    packs, split = _tiny_dataset(tmp_path)
    real_config = train_module.TrainConfig

    def small_config(**kwargs: object) -> object:
        kwargs["model"] = TINY
        return real_config(**kwargs, crop=32, pretrained=False)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "TrainConfig", small_config)
    code = train_main(
        [
            "--packs",
            str(packs),
            "--split",
            str(split),
            "--out",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
        ]
    )
    assert code == 0
    assert "Best validation colour error" in capsys.readouterr().out
