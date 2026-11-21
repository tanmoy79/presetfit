"""Train the look-transfer network.

Each training step:

1. takes a batch of triplets (your photo, the reference, the right answer),
2. lets the network redraw your photo in the reference's look,
3. measures how far that is from the right answer, as the average colour
   difference per pixel (CIE76 Delta E, the straight-line distance between
   colours in Lab),
4. nudges every weight in the network to make that number smaller.

Progress is saved regularly (``last.pt``), and the best version so far on the
validation photos is kept as ``best.pt``. Starting again with the same output
folder continues from ``last.pt``, which matters on Colab, where sessions can
end without warning.

Run ``python -m presetfit.learning.train --help`` for the options.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from presetfit.learning.data import PackReader, Triplet, TripletDataset
from presetfit.learning.model import LookTransferNet, ModelConfig
from presetfit.learning.render import delta_e76, srgb_to_lab

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything that decides how a training run goes. Saved with each checkpoint."""

    packs: str
    split: str
    out: str
    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    crop: int = 224
    train_fraction: float = 1.0
    workers: int = 2
    save_every_minutes: float = 10.0
    seed: int = 0
    pretrained: bool = True

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model"] = self.model.to_dict()
        return data


def look_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average colour difference (CIE76 Delta E) between two batches of sRGB images."""
    return delta_e76(srgb_to_lab(output.float()), srgb_to_lab(target.float())).mean()


def load_split(path: Path) -> tuple[list[Triplet], list[Triplet]]:
    """Read the training/validation split written by :mod:`presetfit.learning.prepare`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def triplets(rows: list[dict[str, str]]) -> list[Triplet]:
        return [Triplet(r["content"], r["reference"], r["gt"]) for r in rows]

    return triplets(data["train"]), triplets(data["validation"])


class Trainer:
    """Runs training and validation, and saves and restores progress."""

    def __init__(self, config: TrainConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out = Path(config.out)
        self.out.mkdir(parents=True, exist_ok=True)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        pack = PackReader(Path(config.packs))
        train, validation = load_split(Path(config.split))
        if config.train_fraction < 1.0:
            train = random.Random(config.seed).sample(
                train, max(1, round(len(train) * config.train_fraction))
            )
        self.train_loader: DataLoader[Any] = DataLoader(
            TripletDataset(pack, train, config.crop, train=True),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.workers > 0,
        )
        self.validation_loader: DataLoader[Any] = DataLoader(
            TripletDataset(pack, validation, config.crop, train=False),
            batch_size=config.batch_size,
            num_workers=config.workers,
        )
        if len(self.train_loader) == 0:
            raise ValueError("not enough training triplets for one batch")

        self.model = LookTransferNet(config.model, pretrained=config.pretrained).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(config.learning_rate), weight_decay=config.weight_decay
        )
        self.total_steps = config.epochs * len(self.train_loader)
        warmup = min(500, self.total_steps // 20 + 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: (
                min(1.0, (step + 1) / warmup)
                * 0.5
                * (1.0 + math.cos(math.pi * min(1.0, step / self.total_steps)))
            ),
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step = 0
        self.epoch = 0
        self.best = float("inf")
        self._resume()

    # --- saving and restoring ------------------------------------------------

    def _checkpoint(self) -> dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best": self.best,
            "config": self.config.to_dict(),
        }

    def save(self, name: str) -> None:
        temporary = self.out / f"{name}.tmp"
        torch.save(self._checkpoint(), temporary)
        temporary.replace(self.out / name)  # never leave a half-written checkpoint

    def _resume(self) -> None:
        last = self.out / "last.pt"
        if not last.exists():
            return
        state = torch.load(last, map_location=self.device, weights_only=False)
        saved_model = state["config"]["model"]
        if saved_model != self.config.model.to_dict():
            raise ValueError(
                f"{last} was trained with a different network ({saved_model}); "
                "use another output folder"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step, self.epoch, self.best = state["step"], state["epoch"], state["best"]
        logger.info("continuing from epoch %d, step %d", self.epoch, self.step)

    # --- training ------------------------------------------------------------

    def _move(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        return [x.to(self.device, non_blocking=True) for x in batch]

    def train(self) -> float:
        """Train to the configured number of epochs; return the best validation score."""
        log_path = self.out / "log.csv"
        new_log = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            log = csv.writer(handle)
            if new_log:
                log.writerow(
                    ["epoch", "step", "train_de76", "val_de76", "val_unchanged_de76", "minutes"]
                )
            started = time.time()
            last_save = time.time()
            while self.epoch < self.config.epochs:
                self.model.train()
                # Skip the batches of a partly finished epoch after resuming.
                skip = self.step - self.epoch * len(self.train_loader)
                running, count = 0.0, 0
                for index, batch in enumerate(self.train_loader):
                    if index < skip:
                        continue
                    content, reference, gt = self._move(batch)
                    with torch.autocast(
                        self.device.type, dtype=torch.float16, enabled=self.use_amp
                    ):
                        output = self.model(content, reference)
                    loss = look_loss(output, gt)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.step += 1
                    running += loss.item()
                    count += 1
                    if self.step % 50 == 0:
                        logger.info(
                            "epoch %d step %d/%d  colour error %.2f",
                            self.epoch + 1,
                            self.step,
                            self.total_steps,
                            running / count,
                        )
                    if time.time() - last_save > self.config.save_every_minutes * 60:
                        self.save("last.pt")
                        last_save = time.time()
                self.epoch += 1
                validation, unchanged = self.validate()
                logger.info(
                    "epoch %d done: validation colour error %.2f (unchanged photo: %.2f)",
                    self.epoch,
                    validation,
                    unchanged,
                )
                if validation < self.best:
                    self.best = validation
                    self.save("best.pt")
                self.save("last.pt")
                log.writerow(
                    [
                        self.epoch,
                        self.step,
                        round(running / max(count, 1), 4),
                        round(validation, 4),
                        round(unchanged, 4),
                        round((time.time() - started) / 60, 1),
                    ]
                )
                handle.flush()
        return self.best

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Mean colour error on the validation photos, and for leaving them unchanged."""
        self.model.eval()
        total = unchanged = 0.0
        batches = 0
        for batch in self.validation_loader:
            content, reference, gt = self._move(batch)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                output = self.model(content, reference)
            total += look_loss(output.clamp(0.0, 1.0), gt).item()
            unchanged += look_loss(content, gt).item()
            batches += 1
        return total / max(batches, 1), unchanged / max(batches, 1)


def load_model(checkpoint: Path, device: str = "cpu") -> LookTransferNet:
    """Rebuild a trained network from a checkpoint file, ready to use."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(state["config"]["model"])
    model = LookTransferNet(config, pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--packs", required=True, help="folder of packed images")
    parser.add_argument("--split", required=True, help="split.json from presetfit.learning.prepare")
    parser.add_argument("--out", required=True, help="folder for checkpoints and the log")
    parser.add_argument(
        "--backbone", default="resnet18", choices=["resnet18", "convnext_tiny", "dinov2_small"]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--train-fraction", type=float, default=1.0, help="use only this share of the training set"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per downloaded file is noise
    config = TrainConfig(
        packs=args.packs,
        split=args.split,
        out=args.out,
        model=ModelConfig(backbone=args.backbone),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_fraction=args.train_fraction,
        workers=args.workers,
    )
    best = Trainer(config).train()
    print(f"Best validation colour error: {best:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
