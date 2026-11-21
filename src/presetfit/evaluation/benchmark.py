"""Score look-transfer methods on the TST2K benchmark.

TST2K (https://huggingface.co/datasets/ToneStyle/TST2K) holds 2,000 triplets.
Each is a folder with three images:

    content    the unedited photo
    reference  another photo, edited in some style
    gt         the content photo edited in that same style (the right answer)

Every method gets ``content`` and ``reference`` and must produce something as
close to ``gt`` as possible. These methods can be compared:

    identity   return the content photo unchanged (the "do nothing" floor)
    reinhard   Reinhard et al. 2001: match mean and spread of each Lab channel
    presetfit  the current PresetFit fitter and renderer
    ceiling    PresetFit given ``gt`` itself as the reference: not a fair
               method, but an upper limit showing how well PresetFit's eight
               sliders can express these edits when the reference is no
               obstacle
    learned    the trained network (needs --checkpoint), full size via a LUT
    learned_sliders
               the trained network, full size via the fitted sliders

Metrics, each comparing the output with ``gt``:

    psnr      peak signal-to-noise ratio in dB; higher is better
    de2000    mean CIEDE2000 colour difference; lower is better
              (about 1 is barely visible, above 5 is clearly different)
    ssim      structural similarity, 0-1; higher is better

Usage:
    python -m presetfit.evaluation.benchmark DATA_DIR --out results.csv \
        [--limit N] [--size 512] [--methods identity,presetfit,learned --checkpoint best.pt]

Needs scikit-image (``pip install -e ".[bench]"``).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from presetfit import ColorSpace, FitConfig, Pixels, PresetFitService

ROLES = ("content", "reference", "gt")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
METRICS = ("psnr", "de2000", "ssim")

Image.MAX_IMAGE_PIXELS = None  # the dataset is trusted; some photos are large


@dataclass(frozen=True)
class Triplet:
    name: str
    content: Path
    reference: Path
    gt: Path


def find_triplets(root: Path) -> list[Triplet]:
    """Return every folder under ``root`` that holds a content, reference and gt image."""
    triplets = []
    for folder, _, files in os.walk(root):
        by_role: dict[str, Path] = {}
        for file in files:
            path = Path(folder, file)
            if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.lower() in ROLES:
                by_role[path.stem.lower()] = path
        if len(by_role) == len(ROLES):
            name = str(Path(folder).relative_to(root))
            triplets.append(Triplet(name, by_role["content"], by_role["reference"], by_role["gt"]))
    return sorted(triplets, key=lambda t: _natural_key(t.name))


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def load(path: Path, long_edge: int, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read an image as float32 sRGB in 0-1, shrunk to ``long_edge`` or to exactly ``size``."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if size is None:
            scale = long_edge / max(rgb.size)
            if scale < 1:
                size = (round(rgb.width * scale), round(rgb.height * scale))
        if size is not None and size != rgb.size:
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.float32) / 255.0


# --- methods -----------------------------------------------------------------

Method = Callable[[np.ndarray, np.ndarray], np.ndarray]


def identity(content: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return content


def reinhard(content: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_lab, reference_lab = rgb2lab(content), rgb2lab(reference)
    mean_s, std_s = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1)) + 1e-6
    mean_r, std_r = reference_lab.mean(axis=(0, 1)), reference_lab.std(axis=(0, 1))
    matched = (source_lab - mean_s) / std_s * std_r + mean_r
    result: np.ndarray = np.clip(lab2rgb(matched), 0.0, 1.0).astype(np.float32)
    return result


_SERVICE: PresetFitService | None = None


def presetfit(content: np.ndarray, reference: np.ndarray) -> np.ndarray:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PresetFitService.create_default(FitConfig())
    result = _SERVICE.match(Pixels(content, ColorSpace.SRGB), Pixels(reference, ColorSpace.SRGB))
    return np.clip(result.image.data, 0.0, 1.0)


def ceiling(content: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return presetfit(content, gt)


_CHECKPOINT: Path | None = None
_DEVICE = "cpu"
_TRANSFER: object = None


def _learned(content: np.ndarray, reference: np.ndarray, mode: str) -> np.ndarray:
    global _TRANSFER
    if _CHECKPOINT is None:
        raise ValueError("the learned methods need --checkpoint")
    if _TRANSFER is None:
        import torch

        from presetfit.learning.transfer import LearnedLookTransfer

        if _DEVICE == "cpu":
            torch.set_num_threads(1)  # one thread per worker; the pool runs them side by side
        _TRANSFER = LearnedLookTransfer.load(_CHECKPOINT, _DEVICE)
    result = _TRANSFER.transfer(  # type: ignore[attr-defined]
        content, reference, mode, with_settings=False
    )
    return result.image  # type: ignore[no-any-return]


def learned(content: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return _learned(content, reference, "lut")


def learned_sliders(content: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return _learned(content, reference, "sliders")


METHODS: dict[str, Method] = {
    "identity": identity,
    "reinhard": reinhard,
    "presetfit": presetfit,
    "ceiling": ceiling,
    "learned": learned,
    "learned_sliders": learned_sliders,
}
DEFAULT_METHODS = ("identity", "reinhard", "presetfit", "ceiling")
CHEATS = frozenset({"ceiling"})  # methods that are shown the answer instead of the reference


# --- scoring -----------------------------------------------------------------


def score(output: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    output = output.astype(np.float64)
    gt = gt.astype(np.float64)
    return {
        "psnr": float(peak_signal_noise_ratio(gt, output, data_range=1.0)),
        "de2000": float(deltaE_ciede2000(rgb2lab(gt), rgb2lab(output)).mean()),
        "ssim": float(structural_similarity(gt, output, data_range=1.0, channel_axis=-1)),
    }


def evaluate(
    job: tuple[Triplet, int, tuple[str, ...], Path | None, str],
) -> list[dict[str, object]]:
    global _CHECKPOINT, _DEVICE
    triplet, long_edge, method_names, checkpoint, device = job
    _CHECKPOINT, _DEVICE = checkpoint, device
    gt_full = load(triplet.gt, long_edge)
    size = (gt_full.shape[1], gt_full.shape[0])
    content = load(triplet.content, long_edge, size)  # content and gt are aligned
    reference = load(triplet.reference, long_edge)
    rows: list[dict[str, object]] = []
    for name in method_names:
        started = time.perf_counter()
        try:
            output = METHODS[name](content, gt_full if name in CHEATS else reference)
            row: dict[str, object] = {**score(output, gt_full), "error": ""}
        except Exception as error:  # report and carry on; one bad file must not stop a run
            row = {**dict.fromkeys(METRICS, ""), "error": f"{type(error).__name__}: {error}"}
        row |= {"triplet": triplet.name, "method": name, "seconds": time.perf_counter() - started}
        rows.append(row)
    return rows


def run(
    jobs: list[tuple[Triplet, int, tuple[str, ...], Path | None, str]], workers: int
) -> Iterator[list[dict[str, object]]]:
    if workers <= 1:
        yield from map(evaluate, jobs)
        return
    with ProcessPoolExecutor(workers) as pool:
        yield from pool.map(evaluate, jobs, chunksize=4)


def summarise(rows: list[dict[str, object]], method_names: tuple[str, ...]) -> str:
    lines = [
        f"{'method':<16} {'n':>5} {'PSNR dB':>8} {'dE2000':>7} "
        f"{'SSIM':>6} {'s/img':>6} {'failed':>6}"
    ]
    for name in method_names:
        ok = [r for r in rows if r["method"] == name and not r["error"]]
        failed = sum(1 for r in rows if r["method"] == name and r["error"])
        if not ok:
            lines.append(f"{name:<16} {0:>5} {'-':>8} {'-':>7} {'-':>6} {'-':>6} {failed:>6}")
            continue
        mean = {m: statistics.fmean(float(r[m]) for r in ok) for m in METRICS}  # type: ignore[arg-type]
        secs = statistics.fmean(float(r["seconds"]) for r in ok)  # type: ignore[arg-type]
        lines.append(
            f"{name:<16} {len(ok):>5} {mean['psnr']:>8.2f} {mean['de2000']:>7.2f} "
            f"{mean['ssim']:>6.3f} {secs:>6.2f} {failed:>6}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path, help="folder holding the TST2K triplet folders")
    parser.add_argument("--out", type=Path, default=Path("tst2k_results.csv"))
    parser.add_argument("--limit", type=int, help="score only the first N triplets")
    parser.add_argument("--start", type=int, default=0, help="skip the first N triplets")
    parser.add_argument("--size", type=int, default=512, help="long edge in pixels (default 512)")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated")
    parser.add_argument("--checkpoint", type=Path, help="trained network, for the learned methods")
    parser.add_argument(
        "--device", default="cpu", help="'cuda' runs the network on a GPU (uses one worker)"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    method_names = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(method_names) - set(METHODS)
    if unknown:
        parser.error(f"unknown method(s): {', '.join(sorted(unknown))}")
    triplets = find_triplets(args.data)[args.start :]
    triplets = triplets[: args.limit]
    if not triplets:
        parser.error(f"no content/reference/gt triplets found under {args.data}")

    print(
        f"Scoring {len(triplets)} triplets at {args.size} px with {args.workers} workers",
        flush=True,
    )
    if any(m.startswith("learned") for m in method_names) and args.checkpoint is None:
        parser.error("the learned methods need --checkpoint")
    if args.device != "cpu":
        args.workers = 1  # one process owns the GPU
    jobs = [(t, args.size, method_names, args.checkpoint, args.device) for t in triplets]
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for done, result in enumerate(run(jobs, args.workers), start=1):
        rows.extend(result)
        if done % 100 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} after {time.perf_counter() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["triplet", "method", *METRICS, "seconds", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(summarise(rows, method_names))
    print(f"\nPer-image results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
