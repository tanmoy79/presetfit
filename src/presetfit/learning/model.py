"""The neural network that redraws a photo in the look of a reference.

The network has two parts:

**Reference reader** (:class:`StyleEncoder`). A pre-trained image network reads
the reference and sums up its look as one vector of numbers. Three are
supported so they can be compared (``backbone`` in :class:`ModelConfig`):

* ``resnet18``: ResNet-18, a classic, small and fast convolutional network.
* ``convnext_tiny``: ConvNeXt-Tiny, a modern convolutional network.
* ``dinov2_small``: DINOv2 ViT-S/14, a vision transformer trained without
  labels to understand images broadly.

Image networks are trained to recognise objects whatever their colour, so a
hand-made colour summary of both photos (:func:`color_features`) is added: a
lightness histogram, colour casts in shadows, mid-tones and highlights, and the
share of each colour band.

**Generator** (:class:`LookTransferNet`). A U-Net: it shrinks your photo step by
step to understand it, then builds it back up, with "skip connections" that
carry fine detail straight across so nothing gets blurry. At every step of the
way back up, the reference's look vector steers the result (FiLM: each feature
channel is scaled and shifted by amounts computed from the look). The network
predicts a *change* to your photo, and its last layer starts at zero, so an
untrained network returns your photo untouched rather than noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from presetfit.color.masks import _CHROMA_KNEE
from presetfit.learning.render import _hue_band_weights, _smoothstep, srgb_to_lab

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES: dict[str, tuple[str, dict[str, Any]]] = {
    "resnet18": ("resnet18.tv_in1k", {}),
    "convnext_tiny": ("convnext_tiny.fb_in1k", {}),
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", {"img_size": 224}),
}
"""Short name -> (timm model name, extra options). Weights come from Hugging Face."""

_HISTOGRAM_BINS = 16
COLOR_FEATURES = _HISTOGRAM_BINS + 5 + 6 + 8


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the network. Saved inside every checkpoint so it can be rebuilt."""

    backbone: str = "resnet18"
    widths: tuple[int, ...] = (32, 64, 128, 256)
    style_dim: int = 256
    encoder_size: int = 224

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {self.backbone!r}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModelConfig:
        values = dict(data)
        values["widths"] = tuple(values["widths"])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]


def color_features(srgb: Tensor) -> Tensor:
    """Summarise the tone and colour of each photo in ``srgb`` (``(B, H, W, 3)``, 0-1).

    Returns shape ``(B, COLOR_FEATURES)``, every value roughly in -1 ... 1.
    """
    lab = srgb_to_lab(srgb.reshape(srgb.shape[0], -1, 3).float())
    lightness, a, b = lab.unbind(-1)
    x = (lightness / 100.0).clamp(0.0, 1.0)

    centers = torch.linspace(0.0, 1.0, _HISTOGRAM_BINS, device=srgb.device)
    width = 1.0 / (_HISTOGRAM_BINS - 1)
    histogram = (1.0 - (x[..., None] - centers).abs() / width).clamp_min(0.0).mean(1)

    chroma = torch.sqrt(a * a + b * b + 1e-8)
    basics = torch.stack([a.mean(1), a.std(1), b.mean(1), b.std(1), chroma.mean(1)], 1) / 50.0

    zones = torch.stack(
        [_smoothstep(1.0 - 2.0 * x), 1.0 - (2.0 * x - 1.0).abs(), _smoothstep(2.0 * x - 1.0)], -1
    )
    zone_weight = zones.sum(1) + 1e-6
    casts = (
        torch.cat(
            [
                (zones * a[..., None]).sum(1) / zone_weight,
                (zones * b[..., None]).sum(1) / zone_weight,
            ],
            dim=1,
        )
        / 30.0
    )

    hue = torch.remainder(torch.rad2deg(torch.atan2(b, a)), 360.0)
    bands = (_hue_band_weights(hue) * _smoothstep(chroma / _CHROMA_KNEE)[..., None]).mean(1)
    return torch.cat([histogram, basics, casts, bands], dim=1)


class StyleEncoder(nn.Module):
    """Reads a photo with a pre-trained image network; returns one feature vector."""

    mean: Tensor
    std: Tensor

    def __init__(self, backbone: str, size: int, pretrained: bool) -> None:
        super().__init__()
        import timm

        name, options = BACKBONES[backbone]
        self.network = timm.create_model(name, pretrained=pretrained, num_classes=0, **options)
        self.size = size
        features: Any = self.network.num_features
        self.out_features = int(features)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, nchw: Tensor) -> Tensor:
        if nchw.shape[-2:] != (self.size, self.size):
            nchw = functional.interpolate(
                nchw,
                size=(self.size, self.size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
        features: Tensor = self.network((nchw - self.mean) / self.std)
        return features.float()


class _Block(nn.Module):
    """Two 3x3 convolutions, optionally steered by the look vector (FiLM)."""

    def __init__(self, channels_in: int, channels_out: int, style_dim: int | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels_out)
        self.film = nn.Linear(style_dim, 2 * channels_out) if style_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            if self.film.bias is not None:
                nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, style: Tensor | None = None) -> Tensor:
        x = functional.gelu(self.conv1(x))
        x = self.norm(self.conv2(x))
        if self.film is not None and style is not None:
            scale, shift = self.film(style).chunk(2, dim=1)
            x = x * (1.0 + scale[..., None, None]) + shift[..., None, None]
        return functional.gelu(x)


class LookTransferNet(nn.Module):
    """Redraws ``content`` in the look of ``reference``.

    Both inputs and the output are sRGB tensors of shape ``(B, H, W, 3)`` in
    0-1. ``H`` and ``W`` must be multiples of 16 (256 during training).
    """

    def __init__(self, config: ModelConfig | None = None, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        widths = self.config.widths
        style_dim = self.config.style_dim
        self.style_encoder = StyleEncoder(
            self.config.backbone, self.config.encoder_size, pretrained
        )
        self.style_head = nn.Sequential(
            nn.Linear(self.style_encoder.out_features + 2 * COLOR_FEATURES, 2 * style_dim),
            nn.GELU(),
            nn.Linear(2 * style_dim, style_dim),
            nn.GELU(),
        )
        self.down = nn.ModuleList()
        channels = 3
        for width in widths:
            self.down.append(_Block(channels, width, None))
            channels = width
        self.middle = _Block(widths[-1], widths[-1], style_dim)
        self.up = nn.ModuleList()
        for width in reversed(widths):
            self.up.append(_Block(channels + width, width, style_dim))
            channels = width
        self.out = nn.Conv2d(widths[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def look(self, content: Tensor, reference: Tensor) -> Tensor:
        """The look vector that steers the generator."""
        embedded = self.style_encoder(reference.permute(0, 3, 1, 2))
        summary = torch.cat([embedded, color_features(reference), color_features(content)], 1)
        out: Tensor = self.style_head(summary)
        return out

    def forward(self, content: Tensor, reference: Tensor) -> Tensor:
        style = self.look(content, reference)
        x = content.permute(0, 3, 1, 2)
        skips = []
        for index, block in enumerate(self.down):
            x = block(x)
            skips.append(x)
            if index < len(self.down) - 1:
                x = functional.avg_pool2d(x, 2)
        x = self.middle(functional.avg_pool2d(x, 2), style)
        for block in self.up:
            skip = skips.pop()
            x = functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            x = block(torch.cat([x, skip], dim=1), style)
        change: Tensor = self.out(x).permute(0, 2, 3, 1)
        return content + change

    def parameter_groups(self, learning_rate: float) -> list[dict[str, object]]:
        """The pre-trained reader learns ten times more gently than the new parts."""
        encoder = list(self.style_encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoder, "lr": learning_rate / 10.0},
            {"params": rest, "lr": learning_rate},
        ]
