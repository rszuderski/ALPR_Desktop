# -*- coding: utf-8 -*-
"""Native desktop visibility assertions for Tk GUI TESTS, never application code.

A mapped widget with valid geometry need not be the actual desktop hit target.
On Win32 Tk delegates CoordsToWindow to WindowFromPoint. A foreign top-level
or unfinished native transition may therefore make winfo_containing return None.
The context below owns only temporary test foreground setup; it never changes
geometry, enabled states, grabs, application policy, or a visibility verdict.
No None/timeout is accepted, no fallback to geometry replaces the native hit.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import platform
import time
import tkinter as tk
from collections.abc import Iterator, Sequence


@contextmanager
def foreground_test_window(window: tk.Toplevel, *, focus: tk.Misc | None = None) -> Iterator[None]:
    """Temporarily isolate a visible TEST window, always restoring topmost.

Do not use this to implement production modality. A window which did not map
is a failure, not a reason to show it on behalf of the application. A local
modal grab is preserved verbatim. Nested/native dialogs are never dismissed.
"""
    if not window.winfo_exists() or not window.winfo_viewable():
        raise AssertionError("Test visibility setup requires an already visible modal")
    previous_topmost = window.attributes("-topmost")
    previous_focus = window.focus_get()
    try:
        window.attributes("-topmost", True)
        window.lift()
        (focus or window).focus_force()
        yield
    finally:
        try:
            if window.winfo_exists():
                window.attributes("-topmost", previous_topmost)
                if (previous_focus is not None and previous_focus.winfo_exists()
                        and previous_focus.winfo_viewable()
                        and previous_focus.winfo_toplevel() is window):
                    previous_focus.focus_set()
        except tk.TclError:
            # Cleanup only: destruction in the tested close route is legitimate.
            # A failed visibility assertion is never caught by this handler.
            pass


def _sample(root: tk.Misc, window: tk.Toplevel, widget: tk.Misc) -> dict:
    """Check client/ancestor clipping and the exact native hit at the centre."""
    item: dict = {"widget": str(widget)}
    try:
        x, y = widget.winfo_rootx(), widget.winfo_rooty()
        width, height = widget.winfo_width(), widget.winfo_height()
        item.update(rect=[x, y, width, height], mapped=bool(widget.winfo_ismapped()),
                    viewable=bool(widget.winfo_viewable()))
        if not item["mapped"] or not item["viewable"] or width <= 1 or height <= 1:
            item["problem"] = "not_viewable_or_empty"
            return item
        # A valid top-level bound alone would miss clipping by an inner frame.
        parent: tk.Misc | None = widget.master
        reached_window = False
        while parent is not None:
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw, ph = parent.winfo_width(), parent.winfo_height()
            if x < px or y < py or x + width > px + pw or y + height > py + ph:
                item.update(problem="clipped_by_ancestor", ancestor=str(parent),
                            ancestor_rect=[px, py, pw, ph])
                return item
            if parent is window:
                reached_window = True
                break
            parent = parent.master
        if not reached_window:
            item["problem"] = "wrong_dialog_parent"
            return item
        centre = (x + width // 2, y + height // 2)
        hit = root.winfo_containing(*centre)
        item.update(centre=list(centre), hit=str(hit) if hit is not None else None)
        item["problem"] = None if hit is widget else "native_hit_mismatch"
    except (tk.TclError, KeyError) as exc:
        item.update(problem="tk_query_failed", error=f"{type(exc).__name__}: {exc}")
    return item


def _win32_point_diagnostic(x: int, y: int) -> dict:
    """Read-only on failure. No titles, executable paths or foreground mutation."""
    if os.name != "nt":
        return {}
    try:
        import ctypes
        from ctypes import wintypes as wt
        u = ctypes.WinDLL("user32", use_last_error=True)
        u.WindowFromPoint.argtypes = [wt.POINT]
        u.WindowFromPoint.restype = wt.HWND
        u.GetForegroundWindow.argtypes = []
        u.GetForegroundWindow.restype = wt.HWND
        u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        u.GetWindowThreadProcessId.restype = wt.DWORD
        u.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u.GetClassNameW.restype = ctypes.c_int
        u.IsWindowEnabled.argtypes = [wt.HWND]
        u.IsWindowEnabled.restype = wt.BOOL
        u.IsWindowVisible.argtypes = [wt.HWND]
        u.IsWindowVisible.restype = wt.BOOL
        def info(handle):
            if not handle:
                return None
            pid = wt.DWORD()
            u.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            name = ctypes.create_unicode_buffer(128)
            u.GetClassNameW(handle, name, len(name))
            return {"hwnd": hex(int(handle)), "pid": int(pid.value), "class": name.value,
                    "enabled": bool(u.IsWindowEnabled(handle)),
                    "visible": bool(u.IsWindowVisible(handle))}
        return {"test_pid": os.getpid(), "point": [x, y],
                "window_at_point": info(u.WindowFromPoint(wt.POINT(x, y))),
                "foreground": info(u.GetForegroundWindow())}
    except (OSError, AttributeError, ValueError) as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _failure_details(root, window, samples, timeout):
    def read(call):
        try:
            return call()
        except (tk.TclError, KeyError):
            return "unavailable"
    result = {"platform": platform.system(), "tk": read(lambda: root.tk.call("package", "provide", "Tk")),
              "windowing_system": read(lambda: root.tk.call("tk", "windowingsystem")),
              "scaling": read(lambda: float(root.tk.call("tk", "scaling"))),
              "modal_geometry": read(window.winfo_geometry),
              "modal_root": read(lambda: [window.winfo_rootx(), window.winfo_rooty()]),
              "modal_state": read(window.state), "topmost": read(lambda: window.attributes("-topmost")),
              "focus": read(lambda: str(window.focus_get())),
              "grab": read(lambda: str(window.grab_current())),
              "screen": read(lambda: [window.winfo_screenwidth(), window.winfo_screenheight()]),
              "timeout_s": timeout, "controls": samples}
    centre = next((s.get("centre") for s in samples if s.get("problem") and s.get("centre")), None)
    if centre is not None:
        result["win32"] = _win32_point_diagnostic(*centre)
    return result


def assert_native_widgets_visible(root: tk.Misc, window: tk.Toplevel, widgets: Sequence[tk.Misc],
                                  *, timeout: float = 2.0, stable_for: float = .10) -> None:
    """Require stable, strict native hit equality for EVERY supplied control.

The window must first be prepared by foreground_test_window. This assertion
only pumps the existing event loop. It does not lift/resize/enable any widget.
A clipped widget, foreign hit, None, Tcl failure or expiry remains a failure.
"""
    controls = tuple(widgets)
    if not controls:
        raise ValueError("No controls supplied to the native visibility assertion")
    if not 0 <= stable_for < timeout <= 10:
        raise ValueError("Require 0 <= stable_for < timeout <= 10 seconds")
    deadline = time.monotonic() + timeout
    stable_since = None
    signature = None
    samples: list[dict] = []
    while True:
        root.update()
        samples = [_sample(root, window, widget) for widget in controls]
        now = time.monotonic()
        current = [(s.get("rect"), s.get("hit")) for s in samples]
        if all(s.get("problem") is None for s in samples):
            if stable_since is None or current != signature:
                stable_since = now
                signature = current
            if now - stable_since >= stable_for:
                return
        else:
            stable_since = None
            signature = None
        if now >= deadline:
            report = _failure_details(root, window, samples, timeout)
            raise AssertionError("NATIVE_VISIBILITY_NOT_CONFIRMED\n" + json.dumps(report, ensure_ascii=False, indent=2))
        time.sleep(min(.01, max(0., deadline - now)))
