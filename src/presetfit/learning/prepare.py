"""Get the training and test data ready, once.

Steps, each skipped if already done:

1. Choose the training examples: TST100K's own ``triplet.json``, or (with
   ``--per-preset N``) pairs of photos edited with the same preset, built from
   a small part of the dataset (see :func:`same_preset_triplets`).
2. Download the images those examples need and pack them at 256 x 256.
3. Download the TST2K test set and keep a copy at 512 px on the long edge, the
   size the benchmark uses (about 2 GB).
4. Find TST100K photos that also appear in TST2K and leave them out.
5. Split the rest into training and validation photos and write ``split.json``.

Run ``python -m presetfit.learning.prepare --help`` for the options.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from presetfit.learning.data import (
    TST100K_REPO,
    download_and_pack,
    find_overlap,
    load_manifest,
    photo_id,
    same_preset_triplets,
    split_triplets,
    structure_signature,
)

logger = logging.getLogger(__name__)

TST2K_REPO = "ToneStyle/TST2K"
TEST_SIZE = 512
ROLES = ("content", "reference", "gt")


def download_tst2k(out_dir: Path, limit: int | None = None) -> list[Path]:
    """Download TST2K and save each triplet folder at ``TEST_SIZE`` px as lossless PNG."""
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(p for p in out_dir.iterdir() if (p / "gt.png").exists())
    if done and (limit is None or len(done) >= limit) and (out_dir / "COMPLETE").exists():
        return done[:limit]
    patterns = [f"{i:04d}/*" for i in range(limit)] if limit else None
    scratch = Path(tempfile.mkdtemp(prefix="tst2k-"))
    try:
        snapshot_download(
            TST2K_REPO, repo_type="dataset", local_dir=scratch, allow_patterns=patterns
        )
        for folder in sorted(p for p in scratch.iterdir() if p.is_dir() and p.name.isdigit()):
            target = out_dir / folder.name
            target.mkdir(exist_ok=True)
            for role in ROLES:
                source = next(folder.glob(f"{role}.*"))
                with Image.open(source) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    scale = TEST_SIZE / max(rgb.size)
                    if scale < 1:
                        rgb = rgb.resize(
                            (round(rgb.width * scale), round(rgb.height * scale)),
                            Image.Resampling.LANCZOS,
                        )
                    rgb.save(target / f"{role}.png")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if limit is None:
        (out_dir / "COMPLETE").touch()
    return sorted(p for p in out_dir.iterdir() if (p / "gt.png").exists())[:limit]


def prepare(
    out: Path,
    limit: int | None = None,
    test_limit: int | None = None,
    threshold: float = 0.6,
    workers: int | None = None,
    per_preset: int | None = None,
) -> dict[str, int]:
    """Run every step and return a few counts for the record."""
    from huggingface_hub import HfApi, hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    if per_preset:
        files = HfApi().list_repo_files(TST100K_REPO, repo_type="dataset")
        triplets = same_preset_triplets(files, per_preset)
    else:
        manifest_path = Path(
            hf_hub_download(TST100K_REPO, "triplet.json", repo_type="dataset", local_dir=out)
        )
        triplets = load_manifest(manifest_path)
    if limit:
        triplets = random.Random(0).sample(triplets, limit)
    logger.info("step 2: packing images for %d triplets", len(triplets))
    pack = download_and_pack(triplets, out / "packs", workers=workers)

    logger.info("step 3: test set")
    test_folders = download_tst2k(out / "tst2k_512", test_limit)
    test_signatures = []
    for folder in test_folders:
        for role in ("content", "reference"):
            with Image.open(folder / f"{role}.png") as image:
                test_signatures.append(structure_signature(np.asarray(image.convert("RGB"))))

    logger.info("step 4: looking for test photos in the training data")
    signatures: dict[str, np.ndarray] = {}
    # Prefer each photo's unedited version; fall back to an edited one.
    for key in sorted(pack.keys(), key=lambda k: not k.startswith("content_images/")):
        if photo_id(key) not in signatures:
            signatures[photo_id(key)] = structure_signature(pack.read(key))
    excluded = find_overlap(signatures, test_signatures, threshold)

    available = [t for t in triplets if all(k in pack for k in (t.content, t.reference, t.gt))]
    train, validation = split_triplets(available, excluded)
    counts = {
        "triplets": len(triplets),
        "packed_triplets": len(available),
        "photos": len(signatures),
        "excluded_photos": len(excluded),
        "test_triplets": len(test_folders),
        "train": len(train),
        "validation": len(validation),
    }
    split = {
        "counts": counts,
        "threshold": threshold,
        "excluded": sorted(excluded),
        "train": [vars(t) for t in train],
        "validation": [vars(t) for t in validation],
    }
    (out / "split.json").write_text(json.dumps(split), encoding="utf-8")
    logger.info("done: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="data folder (packs, test set, split)"
    )
    parser.add_argument("--limit", type=int, help="use only N random triplets (for a quick test)")
    parser.add_argument("--test-limit", type=int, help="use only the first N test triplets")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--per-preset",
        type=int,
        help="build examples from N edited photos per preset instead of triplet.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per downloaded file is noise
    counts = prepare(
        args.out, args.limit, args.test_limit, args.threshold, args.workers, args.per_preset
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
