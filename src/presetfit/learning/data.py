"""Training data: the TST100K triplets, packed small and split fairly.

**The data.** TST100K holds about 109,000 *triplets*: a photo, a reference
photo edited with some Lightroom preset, and the first photo edited with that
same preset (the right answer). Its manifest, ``triplet.json``, lists them as
paths such as ``content_images/01/12615.tif``.

**Packing.** The originals total about 75 GB. A global edit (exposure, white
balance, curves...) looks the same on a 256 x 256 copy, so each image is
shrunk once to that size, stored as a high-quality JPEG, and appended to a
*pack*: one big file per dataset folder plus a small index of where each image
starts. A few large files copy to and from Google Drive far faster than
170,000 small ones. Squashing to a square changes the shape of things in the
picture but not their colours, and the photo and its right answer are squashed
identically.

**Fair testing.** The test set, TST2K, was built from the same source photos
as TST100K. Its folders carry no original file names, so overlap is found by
comparing the pictures: :func:`structure_signature` captures the layout of a
photo while ignoring its colours and tones, so a photo still matches itself
after any preset. Every training triplet that involves a photo seen in the test
set is dropped (:func:`find_overlap`).
"""

from __future__ import annotations

import functools
import io
import json
import logging
import multiprocessing
import os
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

TST100K_REPO = "ToneStyle/TST100K"
PACK_SIZE = 256
_JPEG_QUALITY = 95
_SIGNATURE_SIZE = 64
_SIGNATURE_BLUR = 3

Image.MAX_IMAGE_PIXELS = None  # the datasets are trusted; some originals are large


@dataclass(frozen=True)
class Triplet:
    """One training example, as paths relative to the dataset root."""

    content: str
    reference: str
    gt: str

    @property
    def photo_ids(self) -> frozenset[str]:
        """The original photos this triplet is made from."""
        return frozenset({photo_id(self.content), photo_id(self.reference), photo_id(self.gt)})


def photo_id(path: str) -> str:
    """Return the source photo behind a TST100K path.

    ``content_images/01/12615.tif`` is photo ``12615``; an edited version such as
    ``style_images/013/12615_0526.png`` (photo 12615, preset 526) is the same photo.
    """
    return PurePosixPath(path).stem.split("_")[0]


def load_manifest(path: Path) -> list[Triplet]:
    """Read ``triplet.json``."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Triplet(r["content"], r["reference"], r["gt"]) for r in records]


def same_preset_triplets(files: Iterable[str], per_preset: int, seed: int = 0) -> list[Triplet]:
    """Build triplets from edited photos that share a preset.

    Two photos edited with the same Lightroom preset carry the same look, so
    photo A with preset X can serve as the reference for photo B, with photo B
    in preset X as the right answer. Most of TST100K's own triplets (74%) are
    exactly such pairs. Taking only ``per_preset`` edited photos per preset and
    pairing them every way gives many examples from a small download: with 12
    per preset, about 262,000 examples from 10.6 GB instead of 78 GB.

    Unlike TST100K's own triplets, these pairs were not checked by hand; a few
    presets adapt to each photo, so a small share may be imperfect examples.

    Args:
        files: Every file path in the dataset (content and style images).
        per_preset: How many edited photos to use per preset, chosen at random.
        seed: Makes the random choice repeatable.
    """
    originals = {photo_id(path): path for path in files if path.startswith("content_images/")}
    by_preset: dict[str, list[str]] = defaultdict(list)
    for path in files:
        if not path.startswith("style_images/"):
            continue
        photo, _, preset = PurePosixPath(path).stem.partition("_")
        if preset and photo in originals:
            by_preset[preset].append(path)
    rng = random.Random(seed)
    triplets = []
    for preset in sorted(by_preset):
        chosen = sorted(by_preset[preset])
        rng.shuffle(chosen)
        chosen = chosen[:per_preset]
        for answer in chosen:
            for reference in chosen:
                if photo_id(reference) != photo_id(answer):
                    triplets.append(Triplet(originals[photo_id(answer)], reference, answer))
    return triplets


# --- packs -------------------------------------------------------------------


def shrink(path: Path, size: int = PACK_SIZE) -> bytes:
    """Open an image, square it to ``size`` x ``size`` and return JPEG bytes."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=_JPEG_QUALITY, subsampling=0)
    return buffer.getvalue()


def write_pack(destination: Path, items: Iterable[tuple[str, bytes]]) -> int:
    """Write ``(key, jpeg)`` pairs to ``destination`` (.pack) and its .index.json.

    The index is written last, under a temporary name first, so a pack whose
    index exists is always complete; an interrupted run is simply redone.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    index: dict[str, tuple[int, int]] = {}
    offset = 0
    with destination.open("wb") as handle:
        for key, data in items:
            handle.write(data)
            index[key] = (offset, len(data))
            offset += len(data)
    index_path = destination.with_suffix(".index.json")
    partial = index_path.with_suffix(".partial")
    partial.write_text(json.dumps(index), encoding="utf-8")
    partial.replace(index_path)
    return len(index)


class PackReader:
    """Reads images from every pack in a folder by their dataset path.

    Safe to use from several data-loading processes: each process opens the
    pack files itself on first use.
    """

    def __init__(self, folder: Path) -> None:
        self._folder = Path(folder)
        self._index: dict[str, tuple[str, int, int]] = {}
        for index_path in sorted(self._folder.glob("*.index.json")):
            pack_name = index_path.name.removesuffix(".index.json") + ".pack"
            for key, (offset, length) in json.loads(index_path.read_text("utf-8")).items():
                self._index[key] = (pack_name, offset, length)
        self._maps: dict[str, np.memmap] = {}

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> list[str]:
        return list(self._index)

    def read_bytes(self, key: str) -> bytes:
        pack_name, offset, length = self._index[key]
        if pack_name not in self._maps:
            self._maps[pack_name] = np.memmap(self._folder / pack_name, mode="r", dtype=np.uint8)
        return self._maps[pack_name][offset : offset + length].tobytes()

    def read(self, key: str) -> np.ndarray:
        """Return the image as a ``(size, size, 3)`` uint8 array."""
        with Image.open(io.BytesIO(self.read_bytes(key))) as image:
            return np.asarray(image.convert("RGB"))

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_maps"] = {}  # memory maps are reopened in each worker process
        return state


def _shrink_job(job: tuple[str, Path, int]) -> tuple[str, bytes | None]:
    key, path, size = job
    try:
        return key, shrink(path, size)
    except Exception as error:  # one unreadable file must not stop hours of work
        logger.warning("skipping %s: %s", key, error)
        return key, None


def download_and_pack(
    triplets: Sequence[Triplet],
    out_dir: Path,
    *,
    size: int = PACK_SIZE,
    repo: str = TST100K_REPO,
    workers: int | None = None,
    scratch: Path | None = None,
) -> PackReader:
    """Download every image the ``triplets`` need and pack it at ``size`` px.

    Work is done one dataset folder at a time (about 10,000 files), which is
    downloaded, packed and deleted before the next starts, so the disk never
    holds more than one folder of originals. Folders already packed are
    skipped, so an interrupted run continues where it stopped.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    needed: dict[str, list[str]] = defaultdict(list)
    for triplet in triplets:
        for key in (triplet.content, triplet.reference, triplet.gt):
            needed[str(PurePosixPath(key).parent)].append(key)
    out_dir = Path(out_dir)
    for number, (folder, keys) in enumerate(sorted(needed.items()), start=1):
        keys = sorted(set(keys))
        pack_path = out_dir / (folder.replace("/", "_") + ".pack")
        index_path = pack_path.with_suffix(".index.json")
        if index_path.exists() and set(keys) <= set(json.loads(index_path.read_text("utf-8"))):
            logger.info("[%d/%d] %s already packed", number, len(needed), folder)
            continue
        logger.info("[%d/%d] %s: %d images", number, len(needed), folder, len(keys))
        download_dir = Path(tempfile.mkdtemp(prefix="presetfit-", dir=scratch))
        try:
            if len(keys) > 3000:  # most of a folder: one bulk download is faster
                snapshot_download(
                    repo,
                    repo_type="dataset",
                    allow_patterns=[f"{folder}/*"],
                    local_dir=download_dir,
                    max_workers=16,
                )
            else:  # a sample: fetch just those files, several at a time
                fetch = functools.partial(
                    hf_hub_download, repo, repo_type="dataset", local_dir=download_dir
                )
                with ThreadPoolExecutor(16) as downloads:
                    list(downloads.map(fetch, keys))
            jobs = [(key, download_dir / key, size) for key in keys]
            # "spawn" starts clean processes; forking one with download threads can hang.
            with ProcessPoolExecutor(
                workers or os.cpu_count() or 1, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                results = pool.map(_shrink_job, jobs, chunksize=32)
                write_pack(pack_path, ((k, d) for k, d in results if d is not None))
        finally:
            shutil.rmtree(download_dir, ignore_errors=True)
    return PackReader(out_dir)


# --- overlap with the test set ----------------------------------------------


def structure_signature(rgb: np.ndarray) -> np.ndarray:
    """Describe the fine detail of a photo in a way that ignores its tones and colours.

    The photo is shrunk to 64 x 64 grey pixels, and a blurred copy is
    subtracted so only edges and texture remain (the overall layout, such as
    "bright sky above dark ground", is shared by too many different photos to
    tell them apart). Each value is then replaced by its *rank*, which any tone
    edit that keeps dark things darker than bright things leaves almost
    unchanged. The result is scaled so the dot product of two signatures is
    their correlation: on TST100K the same photo under different presets scores
    above 0.8, different photos below 0.3.
    """
    grey = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("L")
    small = grey.resize((_SIGNATURE_SIZE, _SIGNATURE_SIZE), Image.Resampling.BILINEAR)
    detail = np.asarray(small, dtype=np.float64) - np.asarray(
        small.filter(ImageFilter.GaussianBlur(_SIGNATURE_BLUR)), dtype=np.float64
    )
    ranks = detail.ravel().argsort(kind="stable").argsort().astype(np.float64)
    ranks -= ranks.mean()
    return (ranks / (np.linalg.norm(ranks) + 1e-12)).astype(np.float32)


def find_overlap(
    candidates: dict[str, np.ndarray],
    test_signatures: Sequence[np.ndarray],
    threshold: float = 0.6,
) -> set[str]:
    """Return the ids in ``candidates`` whose photo appears in the test set.

    Args:
        candidates: Photo id to signature, for the training photos.
        test_signatures: Signatures of every test image (photos and references).
        threshold: Correlation above which two images count as the same photo.
            0.6 sits midway between edited copies of one photo (above 0.8)
            and different photos (below 0.3).
    """
    if not candidates or not test_signatures:
        return set()
    ids = list(candidates)
    train = np.stack([candidates[i] for i in ids])
    test = np.stack(list(test_signatures))
    best = (train @ test.T).max(axis=1)
    return {photo for photo, score in zip(ids, best, strict=True) if score >= threshold}


def split_triplets(
    triplets: Sequence[Triplet],
    excluded: set[str],
    validation_fraction: float = 0.03,
    seed: int = 0,
) -> tuple[list[Triplet], list[Triplet]]:
    """Split into training and validation sets.

    Triplets touching an ``excluded`` photo are dropped. A random share of the
    remaining photos is set aside. Validation triplets are those whose photo
    *to be edited* is set aside, so validation always shows the network photos
    it has never been trained on. Training triplets use none of the set-aside
    photos in any role.
    """
    photos = sorted({p for t in triplets for p in t.photo_ids} - excluded)
    rng = random.Random(seed)
    held_out = set(rng.sample(photos, max(1, round(len(photos) * validation_fraction))))
    train, validation = [], []
    for triplet in triplets:
        ids = triplet.photo_ids
        if ids & excluded:
            continue
        if photo_id(triplet.content) in held_out:
            validation.append(triplet)
        elif not ids & held_out:
            train.append(triplet)
    return train, validation


# --- PyTorch dataset ---------------------------------------------------------


def _torch_dataset_base() -> type:
    from torch.utils.data import Dataset

    return Dataset


class TripletDataset(_torch_dataset_base()):  # type: ignore[misc]
    """Serves ``(content, reference, gt)`` as float tensors ``(crop, crop, 3)`` in 0-1.

    In training mode the photo and its right answer get the *same* random crop
    and flip (they must stay aligned), and the reference gets its own. No
    colour changes are made: they would alter the very look being learned.
    """

    def __init__(
        self,
        pack: PackReader,
        triplets: Sequence[Triplet],
        crop: int = 224,
        train: bool = True,
    ) -> None:
        self._pack = pack
        self._triplets = list(triplets)
        self._crop = crop
        self._train = train

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, index: int) -> tuple[object, object, object]:
        import torch

        triplet = self._triplets[index]
        content = self._pack.read(triplet.content)
        gt = self._pack.read(triplet.gt)
        reference = self._pack.read(triplet.reference)
        if self._train:
            content, gt = _random_view([content, gt], self._crop)
            (reference,) = _random_view([reference], self._crop)
        else:
            content, gt, reference = (_center(x, self._crop) for x in (content, gt, reference))
        return tuple(  # type: ignore[return-value]
            torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0)
            for x in (content, reference, gt)
        )


def _random_view(images: list[np.ndarray], crop: int) -> list[np.ndarray]:
    height, width = images[0].shape[:2]
    top = random.randint(0, height - crop)
    left = random.randint(0, width - crop)
    flip = random.random() < 0.5
    views = []
    for image in images:
        view = image[top : top + crop, left : left + crop]
        views.append(view[:, ::-1] if flip else view)
    return views


def _center(image: np.ndarray, crop: int) -> np.ndarray:
    height, width = image.shape[:2]
    top, left = (height - crop) // 2, (width - crop) // 2
    return image[top : top + crop, left : left + crop]
