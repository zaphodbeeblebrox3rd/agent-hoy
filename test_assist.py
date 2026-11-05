#!/usr/bin/env python3
"""Tests for assist mode, image payload sizing, and voice text cleanup."""

import os
import sys

from assist_mode import (
    ASSIST_MODES_ENABLED,
    RAPID_FIRE_ENABLED,
    REUSE_UNCHANGED_PAGE,
    SCREEN_CAPTURE_DIRECT,
    VISION_OFF,
    VISION_SCREEN,
    VISION_WINDOW,
    AssistSettings,
    captures,
    capture_identity,
    followup_capture,
    guide_send_plan,
    general_mode,
    rapid_fire_mode,
    rapid_reply,
    skips_topic_path,
    page_key,
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
from voice_output import (
    BRIEF_PART_CHARS,
    BRIEF_SPOKEN_SENTENCES,
    MAX_SPOKEN_CHARS,
    PART_PAUSE_SECONDS,
    brief_for_speech,
    speech_parts,
    text_for_speech,
)


def test_rollback_defaults():
    assert ASSIST_MODES_ENABLED is True
    defaults = AssistSettings()
    assert defaults.transcription is True
    assert defaults.voice is False
    assert defaults.vision == VISION_OFF
    assert defaults.browser_control is False
    assert defaults.non_it is False
    assert defaults.rapid_fire is False
    assert RAPID_FIRE_ENABLED is True
    assert rapid_fire_mode(defaults) is False
    assert skips_topic_path(defaults) is False
    assert speaks(defaults) is False
    assert captures(defaults) is False
    assert general_mode(defaults) is False
    assert guide_active(defaults) is False
    general = AssistSettings(True, False, VISION_OFF, False, True)
    assert general_mode(general) is True
    assert guide_active(general) is True
    fired = AssistSettings(False, False, VISION_OFF, False, False, True)
    assert fired.rapid_fire is True
    assert rapid_fire_mode(fired) is True
    assert skips_topic_path(fired) is True
    assert guide_active(fired) is True
    assert captures(general) is False
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
    save_assist_settings(AssistSettings(True, False, VISION_OFF, True), tmp_dir)
    with_control = load_assist_settings(tmp_dir)
    assert with_control.browser_control is True
    assert with_control.non_it is False
    save_assist_settings(AssistSettings(True, False, VISION_OFF, False, True), tmp_dir)
    with_general = load_assist_settings(tmp_dir)
    assert with_general.non_it is True
    assert with_general.rapid_fire is False
    assert general_mode(with_general) is True
    save_assist_settings(AssistSettings(True, False, VISION_OFF, False, False, True), tmp_dir)
    with_rapid = load_assist_settings(tmp_dir)
    assert with_rapid.rapid_fire is True
    assert with_rapid.non_it is False
    with open(os.path.join(tmp_dir, "assist_mode.conf"), "w", encoding="utf-8") as handle:
        handle.write("mode = window\n")
    legacy = load_assist_settings(tmp_dir)
    assert legacy == AssistSettings(False, True, VISION_WINDOW)
    print("OK assist mode persistence")
    return True


def test_native_screenshots_are_not_shrunk():
    desktop = (1920, 1080)
    four_k = (3840, 2160)
    for model in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"):
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


def test_unchanged_page_skips_the_screenshot():
    assert REUSE_UNCHANGED_PAGE is True
    first = page_key("https://example.test/q", "question 30 motor unit", b"pixels")
    same = page_key("https://example.test/q", "question 30 motor unit", b"other-pixels")
    changed = page_key("https://example.test/q", "question 31 excitation", b"pixels")
    assert first == same
    assert first != changed
    assert guide_send_plan("", "", first, "Answer the question.", True) == "image"
    assert guide_send_plan(first, "Answer the question.", first, "Answer the question.", True) == "skip"
    assert guide_send_plan(first, "Answer the question.", first, "What about blank two?", True) == "text"
    assert guide_send_plan(first, "old", changed, "What about blank two?", True) == "image"
    assert guide_send_plan(first, "old", first, "new words", True, reuse=False) == "image"
    assert SCREEN_CAPTURE_DIRECT is True
    same_screen = capture_identity("https://example.test/q", "same text", b"pixels-a", screen=True)
    moved_screen = capture_identity("https://example.test/q", "same text", b"pixels-b", screen=True)
    same_window = capture_identity("https://example.test/q", "same text", b"pixels-a", screen=False)
    other_window = capture_identity("https://example.test/q", "same text", b"pixels-b", screen=False)
    assert same_screen != moved_screen
    assert same_window == other_window
    send, kept, note = followup_capture(
        "text",
        "the question on the page",
        "Screen, 1920x1080, original resolution.",
        repeated=True,
    )
    assert send is False
    assert kept == "the question on the page"
    assert "not sent again" in note
    assert "page text" in note
    send, kept, note = followup_capture(
        "text",
        "",
        "No screenshot (capture failed).",
        repeated=False,
    )
    assert send is True
    assert note == "No screenshot (capture failed)."
    long_reply = "The answer is motor unit. " * 40
    spoken = text_for_speech(long_reply)
    assert len(spoken) <= MAX_SPOKEN_CHARS
    assert spoken.endswith(".")
    brief = brief_for_speech(
        "The blanks are motor unit, powerful, finely controlled, thigh, and eye. "
        "That covers every row.\n\n"
        "Large units make stronger contractions. The thigh is the large example."
    )
    assert brief.startswith("The blanks are motor unit")
    assert "That covers every row" not in brief
    assert "thigh is the large example" not in brief
    assert brief.endswith(".")
    assert BRIEF_SPOKEN_SENTENCES == 1
    assert BRIEF_PART_CHARS == 80
    long_sentence = "A" * 500 + "."
    assert brief_for_speech(long_sentence + "\n\nExtra detail stays here.") == long_sentence
    assert PART_PAUSE_SECONDS == 2.0
    assert speech_parts("The plasma membrane is the sarcolemma.\n\nExtra detail.") == [
        "The plasma membrane is the sarcolemma."
    ]
    assert speech_parts("motor unit\npowerful\nfinely controlled\n\nThe thigh is the large example.") == [
        "motor unit",
        "powerful",
        "finely controlled",
    ]
    assert speech_parts("1. motor unit. 2. powerful. 3. eye.") == [
        "1. motor unit.",
        "2. powerful.",
        "3. eye.",
    ]
    assert speech_parts(
        "myofilaments\n"
        "They are made of actin and myosin and run the length of the myofibril inside the fiber.\n\n"
        "More detail stays on screen."
    ) == ["myofilaments"]
    long_lines = (
        "The subunits are called myofilaments and they fill each myofibril from one end to the other.\n"
        "Actin and myosin are the proteins that form those subunits and they overlap in the sarcomere."
    )
    assert speech_parts(long_lines) == [
        "The subunits are called myofilaments and they fill each myofibril from one end to the other."
    ]
    print("OK unchanged page skips the screenshot")
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


def test_analysis_history_continues_only_the_same_reply():
    from main import analysis_continues, is_direct_howto

    assert analysis_continues("Hello", "Hello there") is True
    assert is_direct_howto("How do I check whether a Linux disk is full?") is True
    assert is_direct_howto("How do I fix this error?") is False
    from main import is_direct_explanation

    assert is_direct_explanation("What does a non-zero exit code mean in a shell script?") is True
    assert is_direct_explanation("Why would a Kubernetes pod stay in Pending?") is True
    from main import keyboard_entry_allowed

    assert keyboard_entry_allowed(True, True, False, False) is False
    assert keyboard_entry_allowed(True, True, True, False) is True
    assert keyboard_entry_allowed(True, True, True, True) is False
    assert keyboard_entry_allowed(True, False, False, False) is True
    assert keyboard_entry_allowed(False, True, False, False) is True
    assert is_direct_explanation("How do I fix this error?") is False
    spoken = "Hello there, this is the spoken reply that should stay in the pane."
    assert analysis_continues(spoken + "\nMEMORY:\n- Uses Edge", spoken) is True
    assert analysis_continues(
        "The shell notes stay on screen for later reading.",
        "I need to see the latest line.",
    ) is False
    assert analysis_continues("Hello there", "He") is False
    print("OK analysis history keeps a new reply")
    return True


def test_file_context_and_output(tmp_dir: str):
    from file_exchange import load_context_file, safe_output_name, write_output_files
    from memory_store import split_guide_reply

    assert safe_output_name("../secret.txt") == "secret.txt"
    assert safe_output_name("..") == ""
    assert safe_output_name("notes.txt") == "notes.txt"

    source = os.path.join(tmp_dir, "notes.txt")
    with open(source, "w", encoding="utf-8") as handle:
        handle.write("alpha beta")
    loaded = load_context_file(source)
    assert "alpha beta" in loaded
    assert loaded.startswith("File: notes.txt")

    binary = os.path.join(tmp_dir, "blob.bin")
    with open(binary, "wb") as handle:
        handle.write(b"\x00\x01raw")
    assert "not included as text" in load_context_file(binary)

    reply = (
        "I saved the notes.\n"
        "FILE: scratch.txt\n"
        "line one\n"
        "line two\n"
        "ENDFILE\n"
    )
    spoken, _memory, _actions, produced = split_guide_reply(reply)
    assert spoken == "I saved the notes."
    assert "line one" not in spoken
    assert produced == [("scratch.txt", "line one\nline two")]

    output_dir = os.path.join(tmp_dir, "output")
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, name))
    written = write_output_files(produced, output_dir)
    assert len(written) == 1
    with open(written[0], encoding="utf-8") as handle:
        assert handle.read() == "line one\nline two\n"
    again = write_output_files(produced, output_dir)
    assert os.path.basename(again[0]) == "scratch-2.txt"
    print("OK file context and downloadable output")
    return True


def test_non_it_does_not_prime_computer_transcript():
    from transcription_config import transcript_is_prompt_echo, whisper_initial_prompt

    assert whisper_initial_prompt(False)
    assert whisper_initial_prompt(True) is None
    echo = "This is a technical discussion about computer systems, HPC clusters."
    assert transcript_is_prompt_echo(echo) is True
    assert transcript_is_prompt_echo("Where does the sarcolemma go?") is False
    print("OK non-IT transcript prompt")
    return True


def test_voice_is_conversational():
    from voice_output import TTS_INSTRUCTIONS, TTS_VOICE

    assert TTS_VOICE == "marin"
    folded = TTS_INSTRUCTIONS.casefold()
    assert "conversation" in folded
    assert "announcer" in folded
    print("OK marin voice")
    return True


def test_guide_is_a_conversation():
    from assist_mode import build_guide_prompt, format_recent_turns, guide_instructions

    quiet = guide_instructions(False, True)
    assert "you, the assistant" in quiet
    assert "Only the first paragraph is spoken" in quiet
    rapid_text = guide_instructions(True, False, False, False, True)
    assert "Do not explain" in rapid_text
    assert "further detail" not in rapid_text
    assert "choose" in rapid_text
    assert rapid_reply("contraction\n\nMuscle fibers develop tension.") == "contraction"
    assert rapid_reply("transverse\ntubule\n\nThe tubule opens the channels.") == "transverse\ntubule"
    assert rapid_reply("contraction. The fibers shorten after that.") == "contraction."
    assert "not the assumed topic" in quiet
    assert "Live scroll" in quiet
    assert "MEMORY:" in quiet
    assert "ACTIONS:" not in quiet
    assert "computer-support session" not in quiet
    general = guide_instructions(False, True, True, True)
    assert "debugging port is not the question" in general
    assert "empty boxes from top to bottom" in general
    assert "Give only the next step" not in quiet
    acting = guide_instructions(True, False)
    assert "ACTIONS:" in acting
    assert "choose" in acting
    assert "Do not submit" in acting
    assert "MEMORY:" not in acting
    history = format_recent_turns([("what can you do", "I can help with that.")])
    prompt = build_guide_prompt(
        "can you see this page",
        "Attached chrome tab Example.",
        history=history,
        memory="- Uses Edge",
        page_text="Hello from the page",
    )
    assert "Recent conversation:" in prompt
    assert "what can you do" in prompt
    assert "I can help with that." in prompt
    assert "What the user just said:" in prompt
    assert "can you see this page" in prompt
    assert "What you remember:" in prompt
    assert "Full page text:" in prompt
    print("OK conversational guide")
    return True


def test_memory_and_action_blocks(tmp_dir: str):
    import memory_store
    from memory_store import load_memory, remember, split_guide_reply, visible_reply

    reply = (
        "I can see the sign-in page.\n"
        "MEMORY:\n"
        "- Uses Edge\n"
        "- uses edge\n"
        "ACTIONS:\n"
        "click: Sign in\n"
        "type: hello\n"
        "choose: contraction\n"
        "choose: muscle fibers develop tension | contraction\n"
    )
    spoken, memories, actions, files = split_guide_reply(reply)
    assert files == []
    assert spoken == "I can see the sign-in page."
    assert "MEMORY" not in spoken
    assert "ACTIONS" not in spoken
    assert memories == ["Uses Edge", "uses edge"]
    assert actions[0] == ("click", "Sign in")
    assert actions[1] == ("type", "hello")
    assert actions[2] == ("choose", "contraction")
    assert actions[3] == ("choose", "muscle fibers develop tension | contraction")
    assert visible_reply(reply) == spoken

    memory_dir = os.path.join(tmp_dir, "memory")
    stale_memory = os.path.join(memory_dir, "agent_memory.md")
    if os.path.exists(stale_memory):
        os.remove(stale_memory)
    remember(memories, memory_dir)
    remember(["Uses Edge", "Lives in Chicago"], memory_dir)
    stored = load_memory(memory_dir)
    assert stored.count("Uses Edge") == 1
    assert "Lives in Chicago" in stored
    many = [f"fact {index}" for index in range(memory_store.MAX_MEMORY_LINES + 5)]
    remember(many, memory_dir)
    capped = load_memory(memory_dir)
    capped_facts = [line[2:].strip() for line in capped.splitlines() if line.startswith("- ")]
    assert "fact 0" not in capped_facts
    assert f"fact {memory_store.MAX_MEMORY_LINES + 4}" in capped_facts

    previous = memory_store.MEMORY_ENABLED
    memory_store.MEMORY_ENABLED = False
    try:
        disabled_dir = os.path.join(tmp_dir, "memory-off")
        remember(["should not write"], disabled_dir)
        assert load_memory(disabled_dir) == ""
        assert load_memory(memory_dir) == ""
        assert not os.path.exists(os.path.join(disabled_dir, "agent_memory.md"))
    finally:
        memory_store.MEMORY_ENABLED = previous
    print("OK memory file and hidden blocks")
    return True


def test_browser_matching_and_actions():
    import os as _os

    from browser_bridge import (
        CdpError,
        CdpSession,
        CHOOSE_ENABLED,
        action_is_blocked,
        browser_actions_allowed,
        debugging_status,
        parse_choose,
        look_at_browser,
        page_image_for_prompt,
        relaunch_command,
        titles_match,
    )
    from image_payload import prepare_rgb

    assert titles_match("Example Domain - Google Chrome", "Example Domain")
    assert titles_match("Example Domain - Personal - Microsoft Edge", "Example Domain")
    assert titles_match("Inbox and 2 more pages - Google Chrome", "Inbox")
    assert not titles_match("Inbox - Google Chrome", "Other page")
    command = relaunch_command(
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "chrome",
    )
    assert "--remote-debugging-port=9222" in command
    assert "chrome.exe" in command
    status = debugging_status("edge", command)
    assert "without a debugging port" in status
    assert command in status
    assert look_at_browser(None, "gpt-4.1-mini") is None
    assert browser_actions_allowed(False) is False
    assert browser_actions_allowed(True) is True
    assert CHOOSE_ENABLED is True
    assert action_is_blocked("Submit") is True
    assert action_is_blocked("Check my work") is True
    assert action_is_blocked("Save & Exit") is True
    assert action_is_blocked("Exit Assignment") is True
    assert action_is_blocked("contraction") is False
    assert parse_choose("Submit") is None
    assert parse_choose("contraction") == ("", "contraction")
    assert parse_choose("muscle fibers develop tension | contraction") == (
        "muscle fibers develop tension",
        "contraction",
    )

    rgb = _os.urandom(80 * 60 * 3)
    shrunk = prepare_rgb(rgb, 80, 60, "gpt-4o", max_encoded_bytes=1)
    assert shrunk.resized is True
    kept, too_large = page_image_for_prompt(shrunk)
    assert kept is None and too_large is True

    try:
        CdpSession("ws://example.com/devtools/page/1", timeout=1)
        raise AssertionError("non-local debugging socket was accepted")
    except CdpError as exc:
        assert "not local" in str(exc)
    print("OK browser title match and relaunch command")
    return True


def test_cdp_call_roundtrip():
    import base64
    import hashlib
    import json
    import socket
    import threading

    from browser_bridge import WEBSOCKET_GUID, CdpSession

    ready = threading.Event()
    holder = {}

    def serve():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        holder["server"] = server
        holder["port"] = server.getsockname()[1]
        ready.set()
        conn, _addr = server.accept()
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            key = ""
            for line in data.decode("iso-8859-1").split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
            accept = base64.b64encode(
                hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
            ).decode("ascii")
            conn.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "\r\n"
                ).encode("ascii")
            )
            frame = b""
            while len(frame) < 6:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                frame += chunk
            payload = json.dumps({"id": 1, "result": {"ok": True}}).encode("utf-8")
            conn.sendall(bytes([0x81, len(payload)]) + payload)
        finally:
            conn.close()
            server.close()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(2)
    with CdpSession(f"ws://127.0.0.1:{holder['port']}/devtools/page/1", timeout=3) as session:
        result = session.call("Runtime.evaluate", {"expression": "1"})
    assert result["ok"] is True
    print("OK local debugging call")
    return True


def test_turn_timing_log(tmp_dir):
    import json

    import metrics

    log_dir = os.path.join(tmp_dir, "timing")
    quiet_dir = os.path.join(tmp_dir, "timing-off")
    original_dir = metrics.METRICS_DIR
    original_enabled = metrics.METRICS_ENABLED
    metrics.METRICS_DIR = log_dir
    metrics.METRICS_ENABLED = True
    try:
        stale = os.path.join(log_dir, metrics.METRICS_FILE)
        if os.path.exists(stale):
            os.remove(stale)
        timer = metrics.TurnTimer("guide")
        timer.start("capture")
        timer.stop("attached 1288x903")
        timer.start("model")
        timer.stop("x" * 400)
        timer.finish()
        timer.finish()
        path = os.path.join(log_dir, metrics.METRICS_FILE)
        lines = open(path, encoding="utf-8").read().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "guide"
        assert "T" in record["at"]
        assert record["total_ms"] >= 0
        assert record["stages"][0]["stage"] == "capture"
        assert record["stages"][0]["note"] == "attached 1288x903"
        assert record["stages"][1]["stage"] == "model"
        assert len(record["stages"][1]["note"]) == 160
        metrics.METRICS_ENABLED = False
        metrics.METRICS_DIR = quiet_dir
        quiet = metrics.TurnTimer("speech")
        quiet.start("synthesize")
        quiet.stop("chars=12")
        quiet.finish()
        assert not os.path.exists(os.path.join(quiet_dir, metrics.METRICS_FILE))
    finally:
        metrics.METRICS_DIR = original_dir
        metrics.METRICS_ENABLED = original_enabled
    print("OK turn timing log")
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
        test_unchanged_page_skips_the_screenshot(),
        test_analysis_history_continues_only_the_same_reply(),
        test_non_it_does_not_prime_computer_transcript(),
        test_voice_is_conversational(),
        test_guide_is_a_conversation(),
        test_memory_and_action_blocks(tmp_dir),
        test_turn_timing_log(tmp_dir),
        test_file_context_and_output(tmp_dir),
        test_browser_matching_and_actions(),
        test_cdp_call_roundtrip(),
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
