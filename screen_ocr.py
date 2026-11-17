# Local OCR for captured screenshots.
#
# Windows reads text with the built-in Windows.Media.Ocr engine through the
# winsdk package. It is tens of milliseconds once the engine is warm and needs
# no download or service. Other platforms have no engine yet and return "".
#
# Rollback: set OCR_ENABLED to False to skip local OCR.

import asyncio
import os
import sys
import tempfile

OCR_ENABLED = True
# Bigger captures are shrunk before OCR. 4K reads at full size, so this is
# only a guard for pathological bitmap sizes, not a speed measure.
OCR_MAX_SIDE = 6000
# The prompt stays bounded even when the screen is a wall of text.
OCR_MAX_CHARS = 6000

_engine = None
_engine_checked = False


def ocr_png(png_bytes: bytes) -> str:
    """Text read from a screenshot, or "" when OCR is unavailable."""
    if not OCR_ENABLED or not png_bytes:
        return ""
    if sys.platform != "win32" or not _windows_engine():
        return ""
    try:
        data = _shrink(png_bytes)
        text = asyncio.run(_recognize(data))
        return (text or "").strip()[:OCR_MAX_CHARS]
    except Exception as exc:
        print(f"Local OCR failed: {exc}")
        return ""


def _windows_engine():
    """One shared OCR engine. None when winsdk or a language is missing."""
    global _engine, _engine_checked
    if not _engine_checked:
        _engine_checked = True
        try:
            from winsdk.windows.media.ocr import OcrEngine

            _engine = OcrEngine.try_create_from_user_profile_languages()
        except Exception:
            _engine = None
    return _engine


def _shrink(png_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(png_bytes))
    image.load()
    width, height = image.size
    long_side = max(width, height)
    if long_side <= OCR_MAX_SIDE:
        return png_bytes
    scale = OCR_MAX_SIDE / long_side
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    image = image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def _recognize(png_bytes: bytes) -> str:
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.storage import FileAccessMode, StorageFile

    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(png_bytes)
        storage = await StorageFile.get_file_from_path_async(path)
        stream = await storage.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await _engine.recognize_async(bitmap)
        return result.text or ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
