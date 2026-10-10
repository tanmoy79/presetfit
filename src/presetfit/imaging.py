"""Reading and writing image files."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from presetfit.color.pixels import Pixels
from presetfit.color.spaces import ColorSpace
from presetfit.exceptions import ImageDecodeError, ImageTooLargeError, UnsupportedImageError

ImageSource = str | Path | bytes | BinaryIO

SUPPORTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "TIFF", "BMP"})
# Many cameras save their ".JPG" files as MPO: an ordinary JPEG with a small
# preview image appended. The first frame is the full photo, so it reads as a JPEG.
_READABLE_FORMATS: frozenset[str] = SUPPORTED_FORMATS | {"MPO"}
_TRANSPARENT_MODES = frozenset({"RGBA", "LA", "PA"})
_EXTENSION_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
}
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ImageReader:
    """Decodes image files into sRGB :class:`Pixels`.

    Input is untrusted, so the reader checks the declared size *before*
    decoding (a small file can claim to be billions of pixels) and accepts only
    a known list of formats.
    """

    def __init__(self, max_pixels: int = 50_000_000) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self._max_pixels = max_pixels

    def read(self, source: ImageSource) -> Pixels:
        """Decode ``source`` (a path, raw bytes or a binary stream).

        The photo is rotated according to its EXIF orientation, transparency is
        flattened onto white, and the result is float32 sRGB in 0-1. Colour
        profiles are not converted: the image is assumed to be sRGB.

        Raises:
            ImageDecodeError: if the data is not a readable image.
            UnsupportedImageError: if the format is not in :data:`SUPPORTED_FORMATS`.
            ImageTooLargeError: if the image has more than ``max_pixels`` pixels.
        """
        stream = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            with Image.open(stream) as image:
                if image.format not in _READABLE_FORMATS:
                    raise UnsupportedImageError(
                        f"unsupported image format '{image.format}'; "
                        f"use one of {', '.join(sorted(SUPPORTED_FORMATS))}"
                    )
                width, height = image.size
                if width * height > self._max_pixels:
                    raise ImageTooLargeError(
                        f"image is {width}x{height} ({width * height / 1e6:.1f} MP); "
                        f"the limit is {self._max_pixels / 1e6:.1f} MP"
                    )
                oriented = ImageOps.exif_transpose(image)
                rgb = self._flatten_to_rgb(oriented)
                data = np.asarray(rgb, dtype=np.float32) / 255.0
        except Image.DecompressionBombError as error:
            raise ImageTooLargeError("image is too large to decode safely") from error
        except FileNotFoundError as error:
            raise ImageDecodeError(f"no such file: {source!s}") from error
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
            raise ImageDecodeError("the file could not be read as an image") from error
        return Pixels(data, ColorSpace.SRGB)

    @staticmethod
    def _flatten_to_rgb(image: Image.Image) -> Image.Image:
        has_alpha = image.mode in _TRANSPARENT_MODES or "transparency" in image.info
        if not has_alpha:
            return image.convert("RGB")
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background


class ImageWriter:
    """Encodes :class:`Pixels` as image files.

    Output never carries the input's metadata (EXIF, GPS position, thumbnails),
    because it is built from pixel values alone.
    """

    def __init__(self, jpeg_quality: int = 95) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self._jpeg_quality = jpeg_quality

    def write(self, image: Pixels, path: str | Path) -> Path:
        """Save ``image`` to ``path``, choosing the format from the file extension."""
        target = Path(path)
        image_format = _EXTENSION_FORMATS.get(target.suffix.lower())
        if image_format is None:
            raise UnsupportedImageError(
                f"cannot infer an image format from '{target.name}'; "
                f"use one of {', '.join(sorted(_EXTENSION_FORMATS))}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.encode(image, image_format))
        return target

    def encode(
        self, image: Pixels, image_format: str = "JPEG", max_edge: int | None = None
    ) -> bytes:
        """Return ``image`` encoded in ``image_format`` (a name from :data:`SUPPORTED_FORMATS`).

        If ``max_edge`` is given, the picture is first shrunk so that neither
        side exceeds it. That is how small previews are made.
        """
        if image_format not in SUPPORTED_FORMATS:
            raise UnsupportedImageError(f"unsupported output format '{image_format}'")
        if max_edge is not None and max_edge < 1:
            raise ValueError("max_edge must be positive")
        srgb = image.to(ColorSpace.SRGB).data
        if srgb.ndim != 3:
            raise ValueError("only full images of shape (height, width, 3) can be encoded")
        eight_bit = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        options = {"quality": self._jpeg_quality} if image_format in {"JPEG", "WEBP"} else {}
        picture = Image.fromarray(eight_bit)
        if max_edge is not None:
            picture.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        picture.save(buffer, format=image_format, **options)
        return buffer.getvalue()
