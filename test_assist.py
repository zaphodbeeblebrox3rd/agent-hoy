#!/usr/bin/env python3
"""Tests for assist mode, image payload sizing, and voice text cleanup."""

import os
import sys

from assist_mode import (
    ASSIST_MODES_ENABLED,
    VISION_OFF,
    VISION_SCREEN,
    VISION_WINDOW,
    AssistSettings,
    captures,
    guide_active,
    load_assist_settings,
    save_assist_settings,
    speaks,
)
from image_payload import (
    ImageLimits,
    limits_for_model,
    must_shrink_locally,
    patch_count,
    prepare_encoded,
    prepare_rgb,
)
from screen_capture import _clamp_region, capture_backend_name, session_is_wayland
from voice_output import text_for_speech


def test_rollback_defaults():
    assert ASSIST_MODES_ENABLED is True
    defaults = AssistSettings()
    assert defaults.transcription is True
    assert defaults.voice is False
    assert defaults.vision == VISION_OFF
    assert speaks(defaults) is False
    assert captures(defaults) is False
    assert guide_active(defaults) is False
    combined = AssistSettings(True, True, VISION_SCREEN)
    assert speaks(combined) is True
    assert captures(combined) is True
    assert guide_active(combined) is True
    assert captures(AssistSettings(True, True, VISION_WINDOW)) is True
    assert guide_active(AssistSettings(False, True, VISION_OFF)) is True
    print("OK assist mode defaults")
    return True


def test_persist_assist_mode(tmp_dir: str):
    save_assist_settings(AssistSettings(True, True, VISION_SCREEN), tmp_dir)
    loaded = load_assist_settings(tmp_dir)
    assert loaded == AssistSettings(True, True, VISION_SCREEN)
    with open(os.path.join(tmp_dir, "assist_mode.conf"), "w", encoding="utf-8") as handle:
        handle.write("mode = window\n")
    legacy = load_assist_settings(tmp_dir)
    assert legacy == AssistSettings(False, True, VISION_WINDOW)
    print("OK assist mode persistence")
    return True


def test_native_screenshots_are_not_shrunk():
    desktop = (1920, 1080)
    four_k = (3840, 2160)
    for model in (
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.6-sol",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6-luna",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
    ):
        assert must_shrink_locally(*desktop, limits_for_model(model)) is None
        assert must_shrink_locally(*four_k, limits_for_model(model)) is None

    huge_limits = ImageLimits("original", 65535, None)
    shrunk = must_shrink_locally(8000, 8000, huge_limits)
    assert shrunk is not None
    assert patch_count(*shrunk) <= 30000
    assert must_shrink_locally(10000, 10000, ImageLimits("high", None, None)) is None
    print("OK screenshot shrink policy")
    return True


def test_prepare_keeps_png_bytes():
    rgb = bytes([20, 40, 60]) * (64 * 48)
    prepared = prepare_rgb(rgb, 64, 48, "gpt-5.4")
    assert prepared.resized is False
    assert (prepared.width, prepared.height) == (64, 48)
    assert prepared.detail == "original"
    assert prepared.mime == "image/png"
    again = prepare_encoded(prepared.data, "gpt-5.4")
    assert again.data == prepared.data
    assert again.resized is False
    print("OK PNG passthrough")
    return True


def test_byte_cap_recompresses_before_resize():
    # Noise does not compress; a tiny cap forces JPEG, and a 1-byte cap forces a shrink.
    import os as _os

    rgb = _os.urandom(80 * 60 * 3)
    jpeg_fit = prepare_rgb(rgb, 80, 60, "gpt-4o", max_encoded_bytes=200_000)
    assert jpeg_fit.resized is False
    forced = prepare_rgb(rgb, 80, 60, "gpt-4o", max_encoded_bytes=1)
    assert forced.resized is True
    print("OK byte cap uses JPEG before a spatial shrink")
    return True


def test_voice_text_cleanup():
    spoken = text_for_speech("**Next:** click `Save` in the (dialog).")
    assert "*" not in spoken
    assert "`" not in spoken
    assert "Save" in spoken
    assert "click" in spoken
    print("OK speech text cleanup")
    return True


def test_guide_request_keeps_full_image():
    from openai_integration import openai_analyzer

    rgb = bytes([255, 255, 255]) * (32 * 32)
    prepared = prepare_rgb(rgb, 32, 32, "gpt-5.4")
    kwargs = openai_analyzer._build_responses_kwargs(
        "What is on screen?",
        image=(prepared.data, prepared.mime, prepared.detail),
        instructions="guide",
        max_output_tokens=400,
    )
    content = kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "original"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert kwargs["max_output_tokens"] == 400
    print("OK guide request attaches the full image")
    return True


def test_backend_names():
    if sys.platform == "win32":
        assert capture_backend_name() == "windows"
    elif sys.platform == "darwin":
        assert capture_backend_name() == "macos"
    else:
        previous = os.environ.get("WAYLAND_DISPLAY")
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        try:
            assert session_is_wayland() is True
            assert capture_backend_name() == "wayland"
        finally:
            if previous is None:
                os.environ.pop("WAYLAND_DISPLAY", None)
            else:
                os.environ["WAYLAND_DISPLAY"] = previous
        os.environ.pop("WAYLAND_DISPLAY", None)
        assert capture_backend_name() == "x11"
    print("OK capture backend name")
    return True


def test_clamp_region():
    bounds = {"left": 0, "top": 0, "width": 100, "height": 100}
    region = _clamp_region(-10, -10, 30, 40, bounds)
    assert region == {"left": 0, "top": 0, "width": 20, "height": 30}
    assert _clamp_region(200, 200, 10, 10, bounds) is None
    print("OK region clamp")
    return True


def test_live_screen_is_native():
    if sys.platform != "win32":
        print("SKIP live screen capture on this OS in the unit run")
        return True
    from screen_capture import capture_frame

    captured = capture_frame("screen")
    if captured.error or not captured.png_bytes:
        print(f"FAIL live screen capture: {captured.error}")
        return False
    prepared = prepare_encoded(captured.png_bytes, "gpt-5.4")
    assert prepared.resized is False
    assert (prepared.width, prepared.height) == (captured.width, captured.height)
    assert prepared.data == captured.png_bytes
    print(f"OK live screen {captured.width}x{captured.height} kept native")
    return True


def main():
    tmp_dir = os.path.join(os.path.dirname(__file__), "cache", "assist-mode-test")
    os.makedirs(tmp_dir, exist_ok=True)
    checks = [
        test_rollback_defaults(),
        test_persist_assist_mode(tmp_dir),
        test_native_screenshots_are_not_shrunk(),
        test_prepare_keeps_png_bytes(),
        test_byte_cap_recompresses_before_resize(),
        test_voice_text_cleanup(),
        test_guide_request_keeps_full_image(),
        test_backend_names(),
        test_clamp_region(),
        test_live_screen_is_native(),
    ]
    if not all(checks):
        print("FAIL assist tests")
        return 1
    print("OK all assist tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
