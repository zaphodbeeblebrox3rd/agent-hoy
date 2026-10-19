# Prepare a screenshot for a vision request.
#
# The captured pixels are sent as-is. A spatial shrink happens only when the
# image would be rejected for exceeding the API patch cap after the model's
# own resizing rules, or when the encoded payload is still over the byte cap
# after lossless PNG and high-quality JPEG at the original dimensions.

import io
import math
from dataclasses import dataclass

from PIL import Image

PATCH_PX = 32
MAX_PATCHES = 30_000
# Far under the 512 MB request cap, including base64 expansion.
MAX_ENCODED_BYTES = 80 * 1024 * 1024
JPEG_QUALITIES = (95, 85, 75)


@dataclass(frozen=True)
class ImageLimits:
    """How the API treats an image for one model and detail level."""

    detail: str
    max_side: int | None
    patch_budget: int | None


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime: str
    detail: str
    width: int
    height: int
    resized: bool


# original detail keeps the server from shrinking a normal desktop screenshot.
# Models that do not support original use high. None/None means the server
# applies its own tile resize, so we do not pre-shrink.
_MODEL_LIMITS = {
    "gpt-5.5": ImageLimits("original", 6000, 10_000),
    "gpt-5.4": ImageLimits("original", 6000, 10_000),
    "gpt-5.4-mini": ImageLimits("original", 6000, 10_000),
    "gpt-5.4-nano": ImageLimits("original", 6000, 10_000),
    "gpt-4.1-mini": ImageLimits("high", 2048, 6144),
}

_DEFAULT_LIMITS = ImageLimits("high", None, None)


def limits_for_model(model: str) -> ImageLimits:
    return _MODEL_LIMITS.get(model, _DEFAULT_LIMITS)


def patch_count(width: int, height: int) -> int:
    if width < 1 or height < 1:
        return 0
    return math.ceil(width / PATCH_PX) * math.ceil(height / PATCH_PX)


def scale_to_patch_budget(width: int, height: int, budget: int) -> tuple[int, int]:
    """Smallest proportional shrink that lands on the patch budget."""
    if width < 1 or height < 1:
        return 1, 1
    if patch_count(width, height) <= budget:
        return width, height

    shrink = math.sqrt((PATCH_PX * PATCH_PX * budget) / (width * height))
    new_width = max(1, int(math.floor(width * shrink)))
    new_height = max(1, int(math.floor(height * shrink)))
    while patch_count(new_width, new_height) > budget and (new_width > 1 or new_height > 1):
        new_width = max(1, int(new_width * 0.99))
        new_height = max(1, int(new_height * 0.99))
    return new_width, new_height


def size_after_api_rules(width: int, height: int, limits: ImageLimits) -> tuple[int, int]:
    """Size the API will analyze after its own resize rules."""
    resized_width, resized_height = width, height
    if limits.max_side and max(resized_width, resized_height) > limits.max_side:
        scale = limits.max_side / max(resized_width, resized_height)
        resized_width = max(1, int(math.floor(resized_width * scale)))
        resized_height = max(1, int(math.floor(resized_height * scale)))
    if limits.patch_budget and patch_count(resized_width, resized_height) > limits.patch_budget:
        resized_width, resized_height = scale_to_patch_budget(
            resized_width, resized_height, limits.patch_budget
        )
    return resized_width, resized_height


def must_shrink_locally(width: int, height: int, limits: ImageLimits) -> tuple[int, int] | None:
    """Return a new size only when the API would reject the image.

    Models with no published pixel cap are left alone. Those models resize
    on the server (the tile path). Pre-shrinking them would throw away detail
    the request does not need to drop.
    """
    if limits.max_side is None and limits.patch_budget is None:
        return None

    api_width, api_height = size_after_api_rules(width, height, limits)
    if patch_count(api_width, api_height) <= MAX_PATCHES:
        return None
    return scale_to_patch_budget(width, height, MAX_PATCHES)


def prepare_rgb(
    rgb: bytes,
    width: int,
    height: int,
    model: str,
    max_encoded_bytes: int = MAX_ENCODED_BYTES,
) -> PreparedImage:
    image = Image.frombytes("RGB", (width, height), rgb)
    return _prepare_image(image, model, max_encoded_bytes, source_bytes=None)


def prepare_encoded(
    data: bytes,
    model: str,
    max_encoded_bytes: int = MAX_ENCODED_BYTES,
) -> PreparedImage:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        copied = image.copy()
    return _prepare_image(copied, model, max_encoded_bytes, source_bytes=data)


def shrink_to_long_side(data: bytes, long_side: int, model: str) -> PreparedImage:
    """Spatial shrink used only after the API rejects an image."""
    limits = limits_for_model(model)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        working = image.convert("RGB")
    width, height = working.size
    if max(width, height) > long_side:
        scale = long_side / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        working = working.resize(new_size, Image.Resampling.LANCZOS)
    encoded, mime = _encode_png(working)
    return PreparedImage(encoded, mime, limits.detail, working.size[0], working.size[1], True)


def _prepare_image(
    image: Image.Image,
    model: str,
    max_encoded_bytes: int,
    source_bytes: bytes | None,
) -> PreparedImage:
    limits = limits_for_model(model)
    working = image.convert("RGB")
    width, height = working.size
    target = must_shrink_locally(width, height, limits)
    resized = False
    if target is not None and target != (width, height):
        working = working.resize(target, Image.Resampling.LANCZOS)
        resized = True
        source_bytes = None

    if source_bytes is not None and len(source_bytes) <= max_encoded_bytes:
        return PreparedImage(
            source_bytes,
            _sniff_mime(source_bytes),
            limits.detail,
            width,
            height,
            False,
        )

    encoded, mime, shrunk_image, did_shrink = _encode_within_limit(working, max_encoded_bytes)
    final_width, final_height = shrunk_image.size
    return PreparedImage(
        encoded,
        mime,
        limits.detail,
        final_width,
        final_height,
        resized or did_shrink,
    )


def _encode_within_limit(
    image: Image.Image, max_encoded_bytes: int
) -> tuple[bytes, str, Image.Image, bool]:
    """PNG, then JPEG at the same size, then shrink only if both are too large."""
    png_bytes, _mime = _encode_png(image)
    if len(png_bytes) <= max_encoded_bytes:
        return png_bytes, "image/png", image, False

    for quality in JPEG_QUALITIES:
        jpeg_bytes = _encode_jpeg(image, quality)
        if len(jpeg_bytes) <= max_encoded_bytes:
            return jpeg_bytes, "image/jpeg", image, False

    working = image
    shrunk = False
    while max(working.size) > 32:
        shrunk = True
        new_width = max(1, int(working.size[0] * 0.9))
        new_height = max(1, int(working.size[1] * 0.9))
        if (new_width, new_height) == working.size:
            break
        working = working.resize((new_width, new_height), Image.Resampling.LANCZOS)
        jpeg_bytes = _encode_jpeg(working, 75)
        if len(jpeg_bytes) <= max_encoded_bytes:
            return jpeg_bytes, "image/jpeg", working, shrunk

    jpeg_bytes = _encode_jpeg(working, 75)
    return jpeg_bytes, "image/jpeg", working, shrunk


def _encode_png(image: Image.Image) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "image/png"
