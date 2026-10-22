# Attach to the Edge or Chrome window the user is sharing.
#
# A browser that is already open cannot be attached. It has to be started
# with a remote debugging port. Rollback: set BROWSER_CONTROL_ENABLED to
# False. The Control browser checkbox disappears and actions are not run.

import base64
import hashlib
import json
import os
import socket
import struct
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from image_payload import MAX_ENCODED_BYTES, PreparedImage, prepare_encoded

BROWSER_CONTROL_ENABLED = True
DEBUG_PORT = 9222
PAGE_TEXT_CHARS = 20_000
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_FRAME_BYTES = 96 * 1024 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_BROWSER_SUFFIXES = (
    " - google chrome",
    " - microsoft edge",
    " — google chrome",
    " — microsoft edge",
    " – google chrome",
    " – microsoft edge",
)


class CdpError(Exception):
    """The local debugging connection failed."""


@dataclass
class PageView:
    attached: bool
    browser: str
    title: str
    url: str
    text: str
    status: str
    websocket_url: str
    relaunch_command: str
    prepared: PreparedImage | None = None
    image_too_large: bool = False


def browser_actions_allowed(browser_control: bool) -> bool:
    """Actions run only when the feature is on and the checkbox is checked."""
    return BROWSER_CONTROL_ENABLED and bool(browser_control)


def relaunch_command(exe: str, kind: str) -> str:
    """One command that starts this browser with a debugging port."""
    if exe:
        return f'"{exe}" --remote-debugging-port={DEBUG_PORT}'
    binary = "msedge" if kind == "edge" else "chrome"
    return f"{binary} --remote-debugging-port={DEBUG_PORT}"


def debugging_status(kind: str, command: str) -> str:
    name = "Edge" if kind == "edge" else "Chrome"
    return (
        f"{name} is open without a debugging port, so the full page is not available. "
        f"Close it and start it with: {command}"
    )


def strip_window_title(title: str) -> str:
    """Drop the browser and 'more pages' suffix from a window title."""
    import re

    cleaned = (title or "").strip()
    changed = True
    while changed and cleaned:
        changed = False
        folded = cleaned.casefold()
        for suffix in _BROWSER_SUFFIXES:
            if folded.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
                break
    cleaned = re.sub(r"\s+and\s+\d+\s+more\s+pages?$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def titles_match(window_title: str, tab_title: str) -> bool:
    window = strip_window_title(window_title).casefold()
    tab = " ".join((tab_title or "").split()).casefold()
    if not window or not tab:
        return False
    if window == tab:
        return True
    if len(tab) >= 3 and (window.startswith(tab) or tab.startswith(window)):
        return True
    return False


def page_image_for_prompt(prepared: PreparedImage) -> tuple[PreparedImage | None, bool]:
    """Keep a full-page image only when it did not have to be shrunk."""
    if prepared.resized:
        return None, True
    return prepared, False


def look_at_browser(window_id: int | None, model: str) -> PageView | None:
    """Read the shared Edge or Chrome tab. None when the window is not one of those."""
    if not window_id:
        return None
    title, exe = _window_identity(int(window_id))
    kind = browser_kind(exe, title)
    if not kind:
        return None

    command = relaunch_command(exe, kind)
    port = find_debugging_port(kind)
    if not port:
        return _unattached(kind, title, debugging_status(kind, command), command)

    try:
        targets = list_pages(port)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        print(f"Browser debugging port {port} did not answer: {exc}")
        return _unattached(kind, title, debugging_status(kind, command), command)

    target = match_tab(title, targets)
    if not target:
        name = "Edge" if kind == "edge" else "Chrome"
        status = f"No open {name} tab title matched this window."
        return _unattached(kind, title, status, command)

    websocket_url = target.get("webSocketDebuggerUrl") or ""
    if not websocket_url:
        return _unattached(kind, title, debugging_status(kind, command), command)

    try:
        text, url, png, shot_too_large = read_page(websocket_url)
    except (CdpError, OSError, TimeoutError, ValueError) as exc:
        print(f"Could not read the browser page: {exc}")
        return _unattached(kind, title, f"Could not read the page ({exc}).", command)

    prepared = None
    too_large = shot_too_large
    if png and not too_large:
        try:
            if len(png) > MAX_ENCODED_BYTES:
                too_large = True
            else:
                prepared, too_large = page_image_for_prompt(prepare_encoded(png, model))
        except Exception as exc:
            print(f"Full-page image was not usable: {exc}")
            too_large = True
            prepared = None

    tab_title = target.get("title") or title
    note = [f"Attached {kind} tab {tab_title}."]
    if url:
        note.append(url)
    if text and len(text) >= PAGE_TEXT_CHARS:
        note.append("Page text was truncated.")
    if too_large:
        note.append("The full-page image was too large, so only the page text is included.")
    elif prepared is not None:
        note.append(f"Full page image {prepared.width}x{prepared.height}.")
    else:
        note.append("Full page text is included. No full-page image.")

    return PageView(
        attached=True,
        browser=kind,
        title=title,
        url=url,
        text=text,
        status=" ".join(note),
        websocket_url=websocket_url,
        relaunch_command=command,
        prepared=prepared,
        image_too_large=too_large,
    )


def run_actions(websocket_url: str, actions: list[tuple[str, str]]) -> str:
    """Run click, type, press, scroll, and navigate on the attached tab."""
    if not BROWSER_CONTROL_ENABLED or not actions or not websocket_url:
        return ""
    notes = []
    with CdpSession(websocket_url) as session:
        for verb, argument in actions:
            try:
                notes.append(_run_one(session, verb, argument))
            except (CdpError, OSError, TimeoutError, ValueError) as exc:
                notes.append(f"{verb} failed: {exc}")
    return "; ".join(note for note in notes if note)


def browser_kind(exe: str, title: str) -> str | None:
    name = os.path.basename(exe or "").casefold()
    if name in ("msedge.exe", "msedge", "microsoft-edge"):
        return "edge"
    if name in ("chrome.exe", "chrome", "google-chrome", "google-chrome-stable"):
        return "chrome"
    folded = (title or "").casefold()
    if "microsoft edge" in folded:
        return "edge"
    if "google chrome" in folded:
        return "chrome"
    return None


def find_debugging_port(kind: str) -> int | None:
    """Port from DevToolsActivePort, or 9222 when that browser is already listening."""
    candidates = []
    from_file = _port_from_file(kind)
    if from_file:
        candidates.append(from_file)
    if DEBUG_PORT not in candidates:
        candidates.append(DEBUG_PORT)
    for port in candidates:
        if _port_is_browser(port, kind):
            return port
    return None


def list_pages(port: int) -> list[dict]:
    payload = _json_get(f"http://127.0.0.1:{port}/json/list", timeout=1.5)
    if not isinstance(payload, list):
        return []
    pages = []
    for target in payload:
        if not isinstance(target, dict) or target.get("type") != "page":
            continue
        url = str(target.get("url") or "")
        if url.startswith("devtools://"):
            continue
        pages.append(target)
    return pages


def match_tab(window_title: str, targets: list[dict]) -> dict | None:
    matches = [target for target in targets if titles_match(window_title, str(target.get("title") or ""))]
    if not matches:
        return None
    exact = strip_window_title(window_title).casefold()

    def rank(target: dict) -> tuple[int, int]:
        title = " ".join(str(target.get("title") or "").split()).casefold()
        exact_score = 0 if title == exact else 1
        return exact_score, -len(title)

    matches.sort(key=rank)
    return matches[0]


def read_page(websocket_url: str) -> tuple[str, str, bytes | None, bool]:
    """Return page text, url, png bytes, and whether the screenshot was too large."""
    expression = (
        "(() => {"
        " const text = document.body ? (document.body.innerText || '') : '';"
        f" const limit = {PAGE_TEXT_CHARS};"
        " return {text: text.slice(0, limit), url: location.href};"
        "})()"
    )
    with CdpSession(websocket_url) as session:
        evaluated = session.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        value = (evaluated.get("result") or {}).get("value") or {}
        if not isinstance(value, dict):
            value = {}
        text = str(value.get("text") or "")
        url = str(value.get("url") or "")
        png = None
        too_large = False
        try:
            try:
                session.call("Page.enable")
            except CdpError:
                pass
            shot = session.call(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "captureBeyondViewport": True,
                    "fromSurface": True,
                },
                timeout=20,
            )
            encoded = shot.get("data") or ""
            if encoded:
                png = base64.b64decode(encoded)
        except CdpError as exc:
            if "too large" in str(exc).lower():
                too_large = True
            png = None
        return text, url, png, too_large


class CdpSession:
    """Minimal Chrome DevTools Protocol client for a local page socket."""

    def __init__(self, websocket_url: str, timeout: float = 10.0):
        from urllib.parse import urlparse

        parsed = urlparse(websocket_url)
        host = (parsed.hostname or "").strip("[]")
        if parsed.scheme not in ("ws", "wss") or host not in _LOCAL_HOSTS:
            raise CdpError("Refusing a debugging socket that is not local")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        self._timeout = timeout
        self._next_id = 1
        self._buffer = b""
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._sock.sendall(request.encode("ascii"))
        raw = self._read_until(b"\r\n\r\n")
        head, _, leftover = raw.partition(b"\r\n\r\n")
        self._buffer = leftover
        header = head.decode("iso-8859-1", errors="replace")
        status_line = header.split("\r\n", 1)[0]
        if " 101 " not in status_line:
            self.close()
            raise CdpError("Debugging socket did not accept the connection")
        accept = base64.b64encode(
            hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if accept.lower() not in header.lower():
            self.close()
            raise CdpError("Debugging socket handshake did not match")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        import time

        message_id = self._next_id
        self._next_id += 1
        payload = {"id": message_id, "method": method}
        if params:
            payload["params"] = params
        self._send_text(json.dumps(payload))
        deadline = time.time() + (timeout or self._timeout)
        while time.time() < deadline:
            message = self._recv_text(deadline)
            if message is None:
                break
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if data.get("id") != message_id:
                continue
            if "error" in data:
                raise CdpError(str(data["error"]))
            result = data.get("result")
            return result if isinstance(result, dict) else {}
        raise CdpError(f"{method} timed out")

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def _send_text(self, text: str) -> None:
        self._sock.sendall(encode_client_frame(text))

    def _read_until(self, marker: bytes) -> bytes:
        raw = self._buffer
        self._buffer = b""
        while marker not in raw:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise CdpError("Debugging socket closed during handshake")
            raw += chunk
            if len(raw) > 65536:
                raise CdpError("Debugging handshake was too large")
        return raw

    def _recv_exact(self, size: int, deadline: float) -> bytes:
        import time

        while len(self._buffer) < size:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise CdpError("Debugging socket timed out")
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(max(4096, size - len(self._buffer)))
            except socket.timeout as exc:
                raise CdpError("Debugging socket timed out") from exc
            if not chunk:
                raise CdpError("Debugging socket closed")
            self._buffer += chunk
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    def _recv_text(self, deadline: float) -> str | None:
        fragments = []
        while True:
            header = self._recv_exact(2, deadline)
            first, second = header[0], header[1]
            fin = first & 0x80
            opcode = first & 0x0F
            masked = second & 0x80
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2, deadline))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8, deadline))[0]
            if length > _MAX_FRAME_BYTES:
                raise CdpError("Debugging frame was too large")
            mask_key = self._recv_exact(4, deadline) if masked else b""
            payload = self._recv_exact(length, deadline) if length else b""
            if masked and mask_key:
                payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._sock.sendall(encode_client_frame(payload.decode("utf-8", errors="replace"), opcode=0xA))
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x0):
                fragments.append(payload)
                if fin:
                    return b"".join(fragments).decode("utf-8", errors="replace")
                continue
            if fin and not fragments:
                continue


def encode_client_frame(text: str, opcode: int = 0x1) -> bytes:
    """Masked client frame. Servers send unmasked frames back."""
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    header.extend(mask)
    header.extend(masked)
    return bytes(header)


def _unattached(kind: str, title: str, status: str, command: str) -> PageView:
    return PageView(
        attached=False,
        browser=kind,
        title=title,
        url="",
        text="",
        status=status,
        websocket_url="",
        relaunch_command=command,
    )


def _window_identity(window_id: int) -> tuple[str, str]:
    if sys.platform != "win32":
        return "", ""
    return _windows_title(window_id), _windows_exe(window_id)


def _windows_title(window_id: int) -> str:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    length = user32.GetWindowTextLengthW(int(window_id))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(int(window_id), buffer, length + 1)
    return buffer.value.strip()


def _windows_exe(window_id: int) -> str:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(int(window_id), ctypes.byref(pid))
    if not pid.value:
        return ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value
    finally:
        kernel32.CloseHandle(handle)


def _port_from_file(kind: str) -> int | None:
    for path in _port_file_candidates(kind):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                port = int(handle.readline().strip())
        except (OSError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return None


def _port_file_candidates(kind: str) -> list[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")
    if kind == "edge":
        names = (
            os.path.join(local, "Microsoft", "Edge", "User Data", "DevToolsActivePort"),
            os.path.join(home, "Library", "Application Support", "Microsoft Edge", "DevToolsActivePort"),
            os.path.join(home, ".config", "microsoft-edge", "DevToolsActivePort"),
        )
    else:
        names = (
            os.path.join(local, "Google", "Chrome", "User Data", "DevToolsActivePort"),
            os.path.join(home, "Library", "Application Support", "Google", "Chrome", "DevToolsActivePort"),
            os.path.join(home, ".config", "google-chrome", "DevToolsActivePort"),
        )
    return [path for path in names if path]


def _port_is_browser(port: int, kind: str) -> bool:
    try:
        payload = _json_get(f"http://127.0.0.1:{port}/json/version", timeout=0.8)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return False
    browser = str(payload.get("Browser") or "") if isinstance(payload, dict) else ""
    if kind == "edge":
        return "Edg" in browser
    return "Chrome" in browser and "Edg" not in browser


def _json_get(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_one(session: CdpSession, verb: str, argument: str) -> str:
    verb = verb.strip().lower()
    argument = argument.strip()
    if verb == "navigate":
        if not argument.startswith(("http://", "https://")) or len(argument) > 2000:
            return "navigate skipped"
        try:
            session.call("Page.navigate", {"url": argument})
        except CdpError:
            expression = f"location.assign({json.dumps(argument)})"
            session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return "navigated"
    if verb == "scroll":
        direction = "up" if argument.lower().startswith("up") else "down"
        delta = -700 if direction == "up" else 700
        session.call(
            "Runtime.evaluate",
            {"expression": f"window.scrollBy(0, {delta})", "returnByValue": True},
        )
        return f"scrolled {direction}"
    if verb == "press":
        key = argument[:40]
        expression = (
            "(() => {"
            " const el = window.__agentHoyField || document.activeElement || document.body;"
            f" const key = {json.dumps(key)};"
            " el.dispatchEvent(new KeyboardEvent('keydown', {key: key, bubbles: true}));"
            " el.dispatchEvent(new KeyboardEvent('keyup', {key: key, bubbles: true}));"
            " return 'pressed';"
            "})()"
        )
        session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return f"pressed {key}"
    if verb == "type":
        text = argument[:500]
        expression = (
            "(() => {"
            " const el = (document.activeElement && ('value' in document.activeElement) && document.activeElement)"
            "   || window.__agentHoyField;"
            f" const text = {json.dumps(text)};"
            " if (!el || !('value' in el)) return 'no field';"
            " el.focus();"
            " el.value = text;"
            " el.dispatchEvent(new Event('input', {bubbles: true}));"
            " el.dispatchEvent(new Event('change', {bubbles: true}));"
            " return 'typed';"
            "})()"
        )
        result = session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return str((result.get("result") or {}).get("value") or "typed")
    if verb == "click":
        label = argument[:200]
        expression = (
            "(() => {"
            f" const needle = {json.dumps(label.casefold())};"
            " if (!needle) return 'missing label';"
            " const nodes = document.querySelectorAll('a,button,input,textarea,select,label,[role=\"button\"],[role=\"link\"]');"
            " for (const node of nodes) {"
            "   const name = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('placeholder') || '').trim().toLowerCase();"
            "   if (name.includes(needle)) {"
            "     node.click();"
            "     if (node.focus) node.focus();"
            "     window.__agentHoyField = node;"
            "     return 'clicked';"
            "   }"
            " }"
            " return 'not found';"
            "})()"
        )
        result = session.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return str((result.get("result") or {}).get("value") or "clicked")
    return ""
