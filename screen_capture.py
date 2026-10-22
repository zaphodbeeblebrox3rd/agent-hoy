# Cross-platform screenshot capture at the display's native resolution.
#
# Windows, macOS, and X11 grab pixels directly. Wayland has to ask the
# desktop portal, because a normal screenshot of another app comes back black.

import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from PIL import Image


@dataclass
class CaptureResult:
    png_bytes: bytes | None
    width: int
    height: int
    label: str
    error: str | None
    window_id: int | None = None


def session_is_wayland() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("WAYLAND_DISPLAY"))


def capture_backend_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if session_is_wayland():
        return "wayland"
    return "x11"


def capture_frame(
    mode: str,
    *,
    exclude_window_id: int | None = None,
    preferred_window_id: int | None = None,
    logical_screen: tuple[int, int] | None = None,
) -> CaptureResult:
    """Capture one frame. mode is 'window' or 'screen'."""
    try:
        if mode == "screen":
            return _capture_screen()
        if mode == "window":
            return _capture_window(exclude_window_id, preferred_window_id, logical_screen)
        return CaptureResult(None, 0, 0, "", f"Unknown capture mode '{mode}'")
    except Exception as exc:
        return CaptureResult(None, 0, 0, "", str(exc))


def capture_window_by_id(window_id: int) -> CaptureResult:
    """Capture one window, including this app. Used to record a training outcome."""
    try:
        if sys.platform == "win32":
            hwnd = _windows_root(int(window_id))
            printed = _print_window_png(hwnd)
            if printed and printed.png_bytes:
                return printed
            return _capture_hwnd(hwnd)
        return _capture_screen()
    except Exception as exc:
        return CaptureResult(None, 0, 0, "", str(exc))


def foreign_window_id(exclude_window_id: int | None) -> int | None:
    """Foreground window that is not this app, when the platform can see it."""
    try:
        if sys.platform == "win32":
            return _windows_foreign_hwnd(exclude_window_id)
        if sys.platform.startswith("linux") and not session_is_wayland():
            return _x11_active_window(exclude_window_id)
    except Exception:
        return None
    return None


def _capture_screen() -> CaptureResult:
    if session_is_wayland():
        return _wayland_portal_capture(interactive=False, label="Wayland screen")
    return _mss_screen()


def _capture_window(
    exclude_window_id: int | None,
    preferred_window_id: int | None,
    logical_screen: tuple[int, int] | None,
) -> CaptureResult:
    if sys.platform == "win32":
        return _windows_window(exclude_window_id, preferred_window_id)
    if sys.platform == "darwin":
        return _macos_window(logical_screen)
    if session_is_wayland():
        return _wayland_portal_capture(interactive=True, label="Wayland window")
    return _x11_window(exclude_window_id, preferred_window_id)


def _mss_screen() -> CaptureResult:
    import mss

    with mss.MSS() as sct:
        monitor = _monitor_under_pointer(sct)
        shot = sct.grab(monitor)
        png_bytes = _png_from_rgb(shot.rgb, shot.size.width, shot.size.height)
        width, height = shot.size.width, shot.size.height
    return CaptureResult(png_bytes, width, height, "Screen", None, None)


def _monitor_under_pointer(sct):
    monitors = sct.monitors[1:] or sct.monitors
    point = _pointer_position(sct)
    if point is not None:
        px, py = point
        for monitor in monitors:
            if (
                monitor["left"] <= px < monitor["left"] + monitor["width"]
                and monitor["top"] <= py < monitor["top"] + monitor["height"]
            ):
                return monitor
    return monitors[0]


def _pointer_position(sct) -> tuple[float, float] | None:
    if sys.platform == "win32":
        return _windows_pointer(sct)
    if sys.platform == "darwin":
        return _macos_pointer()
    if sys.platform.startswith("linux") and not session_is_wayland():
        return _x11_pointer()
    return None


def _png_from_rgb(rgb: bytes, width: int, height: int) -> bytes:
    image = Image.frombytes("RGB", (width, height), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _region_png(left: int, top: int, width: int, height: int) -> CaptureResult:
    import mss

    if width < 2 or height < 2:
        return CaptureResult(None, 0, 0, "", "Window is too small to capture")

    with mss.MSS() as sct:
        bounds = sct.monitors[0]
        region = _clamp_region(left, top, width, height, bounds)
        if region is None:
            return CaptureResult(None, 0, 0, "", "Window is outside the visible displays")
        shot = sct.grab(region)
        png_bytes = _png_from_rgb(shot.rgb, shot.size.width, shot.size.height)
        captured_width, captured_height = shot.size.width, shot.size.height
    return CaptureResult(png_bytes, captured_width, captured_height, "Window", None, None)


def _clamp_region(left: int, top: int, width: int, height: int, bounds: dict) -> dict | None:
    right = left + width
    bottom = top + height
    bound_left = bounds["left"]
    bound_top = bounds["top"]
    bound_right = bound_left + bounds["width"]
    bound_bottom = bound_top + bounds["height"]

    crop_left = max(left, bound_left)
    crop_top = max(top, bound_top)
    crop_right = min(right, bound_right)
    crop_bottom = min(bottom, bound_bottom)
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width < 2 or crop_height < 2:
        return None
    return {
        "left": int(crop_left),
        "top": int(crop_top),
        "width": int(crop_width),
        "height": int(crop_height),
    }


def _windows_user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.GetAncestor.restype = wintypes.HWND
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    return user32, wintypes


def _windows_scale(sct) -> float:
    user32, _wintypes = _windows_user32()
    logical_width = user32.GetSystemMetrics(0)
    if logical_width <= 0 or len(sct.monitors) < 2:
        return 1.0
    physical_width = sct.monitors[1]["width"]
    if physical_width <= 0:
        return 1.0
    return physical_width / logical_width


def _windows_pointer(sct) -> tuple[float, float] | None:
    import ctypes

    user32, wintypes = _windows_user32()
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    scale = _windows_scale(sct)
    return point.x * scale, point.y * scale


def _windows_root(hwnd: int) -> int:
    user32, _wintypes = _windows_user32()
    root = user32.GetAncestor(hwnd, 2)
    return int(root or hwnd)


def _windows_foreign_hwnd(exclude_window_id: int | None) -> int | None:
    user32, _wintypes = _windows_user32()
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    hwnd_int = int(hwnd)
    if exclude_window_id and _windows_root(hwnd_int) == _windows_root(exclude_window_id):
        return None
    if user32.IsIconic(hwnd):
        return None
    return hwnd_int


def _windows_title(hwnd: int) -> str:
    import ctypes

    user32, _wintypes = _windows_user32()
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return "Window"
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    title = buffer.value.strip()
    return title or "Window"


def _windows_window(exclude_window_id: int | None, preferred_window_id: int | None) -> CaptureResult:
    user32, _wintypes = _windows_user32()
    hwnd = _windows_foreign_hwnd(exclude_window_id)
    if hwnd is None and preferred_window_id and user32.IsWindow(preferred_window_id):
        if not user32.IsIconic(preferred_window_id):
            hwnd = int(preferred_window_id)
    if hwnd is None:
        screen = _mss_screen()
        if screen.png_bytes:
            screen.label = "Screen (no other window was in front)"
        return screen

    return _capture_hwnd(hwnd)


def _print_window_png(hwnd: int) -> CaptureResult | None:
    """Copy the window's own pixels. A screen grab would show whatever covers it."""
    import ctypes

    user32, wintypes = _windows_user32()
    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    hwnd = int(hwnd)
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 20 or height < 20:
        return None

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    memory = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    if not memory or not bitmap:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory:
            gdi32.DeleteDC(memory)
        user32.ReleaseDC(hwnd, hdc)
        return None
    selected = gdi32.SelectObject(memory, bitmap)
    try:
        # 2 asks for the full window contents, including the client area.
        if not user32.PrintWindow(hwnd, memory, 2):
            if not user32.PrintWindow(hwnd, memory, 0):
                return None
        info = BITMAPINFOHEADER()
        info.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.biWidth = width
        info.biHeight = -height
        info.biPlanes = 1
        info.biBitCount = 32
        info.biCompression = 0
        pixels = ctypes.create_string_buffer(width * height * 4)
        lines = gdi32.GetDIBits(memory, bitmap, 0, height, pixels, ctypes.byref(info), 0)
        if not lines:
            return None
        image = Image.frombuffer("RGBA", (width, height), pixels, "raw", "BGRA", 0, 1)
        sample = image.resize((1, 1)).getpixel((0, 0))
        if isinstance(sample, tuple) and max(sample[:3]) < 8:
            return None
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return CaptureResult(buffer.getvalue(), width, height, _windows_title(hwnd), None, hwnd)
    finally:
        gdi32.SelectObject(memory, selected)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(hwnd, hdc)


def _capture_hwnd(hwnd: int) -> CaptureResult:
    import ctypes

    user32, wintypes = _windows_user32()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(int(hwnd), ctypes.byref(rect)):
        return CaptureResult(None, 0, 0, "", "Could not read the window bounds")

    import mss

    with mss.MSS() as sct:
        scale = _windows_scale(sct)
    left = int(rect.left * scale)
    top = int(rect.top * scale)
    width = int((rect.right - rect.left) * scale)
    height = int((rect.bottom - rect.top) * scale)
    captured = _region_png(left, top, width, height)
    if captured.png_bytes:
        captured.label = _windows_title(int(hwnd))
        captured.window_id = int(hwnd)
    return captured


def _macos_pointer() -> tuple[float, float] | None:
    import ctypes

    try:
        core = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
    except OSError:
        return None

    class CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    core.CGEventCreate.restype = ctypes.c_void_p
    core.CGEventCreate.argtypes = [ctypes.c_void_p]
    core.CGEventGetLocation.restype = CGPoint
    core.CGEventGetLocation.argtypes = [ctypes.c_void_p]
    core.CFRelease.argtypes = [ctypes.c_void_p]
    event = core.CGEventCreate(None)
    if not event:
        return None
    try:
        location = core.CGEventGetLocation(event)
    finally:
        core.CFRelease(event)
    return location.x, location.y


def _macos_window(logical_screen: tuple[int, int] | None) -> CaptureResult:
    if not shutil.which("osascript"):
        screen = _mss_screen()
        screen.label = "Screen (window bounds need osascript)"
        return screen

    script = r'''
tell application "System Events"
  set frontApp to first application process whose frontmost is true
  if unix id of frontApp is OUR_PID then
    set chosen to missing value
    repeat with proc in (application processes whose background only is false)
      if unix id of proc is not OUR_PID then
        try
          set chosen to proc
          exit repeat
        end try
      end if
    end repeat
  else
    set chosen to frontApp
  end if
  if chosen is missing value then
    return "NONE"
  end if
  tell window 1 of chosen
    set pos to position
    set sz to size
  end tell
  return (name of chosen) & "|" & (item 1 of pos as text) & "|" & (item 2 of pos as text) & "|" & (item 1 of sz as text) & "|" & (item 2 of sz as text)
end tell
'''.replace("OUR_PID", str(os.getpid()))
    completed = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip() or completed.stdout.strip() == "NONE":
        screen = _mss_screen()
        detail = completed.stderr.strip() or "no other window"
        screen.label = f"Screen ({detail})"
        return screen

    parts = completed.stdout.strip().split("|")
    if len(parts) != 5:
        return CaptureResult(None, 0, 0, "", "Could not read the macOS window bounds")

    title, x_text, y_text, w_text, h_text = parts
    scale = _macos_scale(logical_screen)
    left = int(float(x_text) * scale)
    top = int(float(y_text) * scale)
    width = int(float(w_text) * scale)
    height = int(float(h_text) * scale)
    captured = _region_png(left, top, width, height)
    if captured.png_bytes:
        captured.label = title or "Window"
    return captured


def _macos_scale(logical_screen: tuple[int, int] | None) -> float:
    if not logical_screen or logical_screen[0] <= 0:
        return 1.0
    import mss

    with mss.MSS() as sct:
        if len(sct.monitors) < 2:
            return 1.0
        physical_width = sct.monitors[1]["width"]
    if physical_width <= 0:
        return 1.0
    return physical_width / logical_screen[0]


def _x11_display():
    import ctypes

    lib = ctypes.CDLL("libX11.so.6")
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.XInternAtom.restype = ctypes.c_ulong
    lib.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    lib.XGetWindowProperty.restype = ctypes.c_int
    lib.XFree.argtypes = [ctypes.c_void_p]
    lib.XTranslateCoordinates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
    ]
    lib.XTranslateCoordinates.restype = ctypes.c_int
    lib.XGetGeometry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XGetGeometry.restype = ctypes.c_int
    lib.XQueryPointer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.XQueryPointer.restype = ctypes.c_int
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Could not open the X11 display")
    return lib, display


def _x11_active_window(exclude_window_id: int | None) -> int | None:
    import ctypes

    lib, display = _x11_display()
    try:
        root = lib.XDefaultRootWindow(display)
        atom = lib.XInternAtom(display, b"_NET_ACTIVE_WINDOW", 0)
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.POINTER(ctypes.c_ubyte)()
        status = lib.XGetWindowProperty(
            display,
            root,
            atom,
            0,
            1,
            0,
            0,
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(nitems),
            ctypes.byref(bytes_after),
            ctypes.byref(prop),
        )
        if status != 0 or not nitems.value:
            return None
        window = ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))[0]
        lib.XFree(prop)
        window_id = int(window)
        if exclude_window_id and window_id == int(exclude_window_id):
            return None
        return window_id
    finally:
        lib.XCloseDisplay(display)


def _x11_pointer() -> tuple[float, float] | None:
    import ctypes

    lib, display = _x11_display()
    try:
        root = lib.XDefaultRootWindow(display)
        root_return = ctypes.c_ulong()
        child_return = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()
        ok = lib.XQueryPointer(
            display,
            root,
            ctypes.byref(root_return),
            ctypes.byref(child_return),
            ctypes.byref(root_x),
            ctypes.byref(root_y),
            ctypes.byref(win_x),
            ctypes.byref(win_y),
            ctypes.byref(mask),
        )
        if not ok:
            return None
        return float(root_x.value), float(root_y.value)
    finally:
        lib.XCloseDisplay(display)


def _x11_window_geometry(window_id: int) -> tuple[int, int, int, int]:
    import ctypes

    lib, display = _x11_display()
    try:
        root = lib.XDefaultRootWindow(display)
        child = ctypes.c_ulong()
        x_pos = ctypes.c_int()
        y_pos = ctypes.c_int()
        translated = lib.XTranslateCoordinates(
            display,
            window_id,
            root,
            0,
            0,
            ctypes.byref(x_pos),
            ctypes.byref(y_pos),
            ctypes.byref(child),
        )
        if not translated:
            raise RuntimeError("Could not translate the window position")
        geo_root = ctypes.c_ulong()
        geo_x = ctypes.c_int()
        geo_y = ctypes.c_int()
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        border = ctypes.c_uint()
        depth = ctypes.c_uint()
        ok = lib.XGetGeometry(
            display,
            window_id,
            ctypes.byref(geo_root),
            ctypes.byref(geo_x),
            ctypes.byref(geo_y),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(border),
            ctypes.byref(depth),
        )
        if not ok:
            raise RuntimeError("Could not read the window geometry")
        return int(x_pos.value), int(y_pos.value), int(width.value), int(height.value)
    finally:
        lib.XCloseDisplay(display)


def _x11_window(exclude_window_id: int | None, preferred_window_id: int | None) -> CaptureResult:
    window_id = _x11_active_window(exclude_window_id)
    if window_id is None and preferred_window_id:
        window_id = int(preferred_window_id)
    if window_id is None:
        screen = _mss_screen()
        screen.label = "Screen (no other window was in front)"
        return screen
    left, top, width, height = _x11_window_geometry(window_id)
    captured = _region_png(left, top, width, height)
    if captured.png_bytes:
        captured.label = "Window"
        captured.window_id = window_id
    return captured


def _wayland_portal_capture(interactive: bool, label: str) -> CaptureResult:
    png_bytes = portal_screenshot(interactive)
    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
    return CaptureResult(png_bytes, width, height, label, None, None)


def portal_screenshot(interactive: bool, timeout: float | None = None) -> bytes:
    """One desktop-portal screenshot. Interactive window choice waits on the user."""
    if timeout is None:
        timeout = 120.0 if interactive else 30.0

    from urllib.parse import unquote, urlparse

    from jeepney import DBusAddress, MatchRule, MessageType, new_method_call
    from jeepney.io.blocking import open_dbus_connection

    connection = open_dbus_connection(bus="SESSION")
    try:
        unique = connection.unique_name
        if not unique:
            raise RuntimeError("Could not read the D-Bus unique name")
        token = f"agenthoy{os.getpid()}"
        sender = unique[1:].replace(".", "_") if unique.startswith(":") else unique.replace(".", "_")
        handle = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
        rule = _portal_response_rule(handle)
        connection.bus_proxy.AddMatch(rule)

        address = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.portal.Screenshot",
        )
        options = {
            "handle_token": ("s", token),
            "interactive": ("b", interactive),
            "modal": ("b", False),
        }
        message = new_method_call(address, "Screenshot", "sa{sv}", ("", options))
        with connection.filter(rule, bufsize=5) as queue:
            reply = connection.send_and_get_reply(message, timeout=timeout)
            if reply.header.message_type == MessageType.error:
                detail = reply.body[0] if reply.body else "portal call failed"
                raise RuntimeError(str(detail))
            returned_handle = reply.body[0] if reply.body else handle
            if returned_handle and returned_handle != handle:
                rule = _portal_response_rule(returned_handle)
                connection.bus_proxy.AddMatch(rule)
                with connection.filter(rule, bufsize=5) as returned_queue:
                    signal = connection.recv_until_filtered(returned_queue, timeout=timeout)
            else:
                signal = connection.recv_until_filtered(queue, timeout=timeout)

        response_code, results = signal.body
        if response_code != 0:
            raise RuntimeError("The desktop share dialog was cancelled")
        uri = _variant_value(results.get("uri"))
        if not uri:
            raise RuntimeError("The desktop portal did not return a screenshot")
        path = unquote(urlparse(str(uri)).path)
        with open(path, "rb") as handle_file:
            return handle_file.read()
    finally:
        connection.close()


def _portal_response_rule(path: str):
    from jeepney import MatchRule

    return MatchRule(
        type="signal",
        interface="org.freedesktop.portal.Request",
        member="Response",
        path=path,
    )


def _variant_value(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        return value[1]
    return value
