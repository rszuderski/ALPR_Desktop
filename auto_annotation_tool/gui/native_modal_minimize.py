"""Win32 bridge for Tk 8.6's rejection of SC_MINIMIZE while a dialog owns a grab."""

import ctypes
from ctypes import wintypes

from ..config import logger

WM_SYSCOMMAND = 0x0112
SC_MINIMIZE = 0xF020
WM_NCDESTROY = 0x0082
GA_ROOT = 2


def release_modal_grab_before_system_command(window, message, command):
    if message != WM_SYSCOMMAND or (command & 0xFFF0) != SC_MINIMIZE:
        return False
    grabbed = window.grab_current()
    if grabbed is None or grabbed.winfo_toplevel() is not window:
        return False
    grabbed.grab_release()
    return True


class NativeModalMinimize:
    """Subclass only this dialog's wrapper; pass all messages to its native chain."""

    def __init__(self, window):
        self.window = window
        self.hwnd = None
        self.closed = False
        self._minimize_job = None
        self._poll_job = None
        self._requested_command = None
        self._forwarding = False
        self._id = id(self)
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._comctl32 = ctypes.WinDLL("comctl32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM, ctypes.c_size_t, ctypes.c_size_t,
        )
        self._user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self._user32.GetAncestor.restype = wintypes.HWND
        self._user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self._user32.PostMessageW.restype = wintypes.BOOL
        self._comctl32.SetWindowSubclass.argtypes = [wintypes.HWND, callback_type, ctypes.c_size_t, ctypes.c_size_t]
        self._comctl32.SetWindowSubclass.restype = wintypes.BOOL
        self._comctl32.RemoveWindowSubclass.argtypes = [wintypes.HWND, callback_type, ctypes.c_size_t]
        self._comctl32.RemoveWindowSubclass.restype = wintypes.BOOL
        self._comctl32.DefSubclassProc.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self._comctl32.DefSubclassProc.restype = ctypes.c_ssize_t
        # Keep the ctypes callback alive for as long as native code can call it.
        self._callback = callback_type(self._dispatch)

    def attach(self):
        if self.closed:
            return
        if not self.hwnd:
            hwnd = self._user32.GetAncestor(self.window.winfo_id(), GA_ROOT)
            if not hwnd:
                raise OSError("No native wrapper for resource modal")
            if not self._comctl32.SetWindowSubclass(hwnd, self._callback, self._id, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            self.hwnd = hwnd
        if self._poll_job is None:
            self._poll_job = self.window.after(24, self._poll)

    def _dispatch(self, hwnd, message, wparam, lparam, subclass_id, reference):
        try:
            if message == WM_NCDESTROY:
                self._comctl32.RemoveWindowSubclass(hwnd, self._callback, self._id)
                if hwnd == self.hwnd:
                    self.hwnd = None
                self._requested_command = None
                self._forwarding = False
            elif message == WM_SYSCOMMAND and (wparam & 0xFFF0) == SC_MINIMIZE:
                if self._forwarding:
                    self._forwarding = False
                else:
                    # No Tk calls here: reentering Tcl from a ctypes window proc
                    # corrupts _tkinter's saved thread state. The Tk timer drains
                    # this request safely, then forwards the native command.
                    self._requested_command = (hwnd, wparam, lparam)
                    return 0
        except Exception:
            logger.exception("[RESOURCE WINDOW] Native minimize callback failed")
        return self._comctl32.DefSubclassProc(hwnd, message, wparam, lparam)

    def _poll(self):
        self._poll_job = None
        if self.closed:
            return
        request = self._requested_command
        self._requested_command = None
        try:
            if request and self._minimize_job is None:
                grabbed = self.window.grab_current()
                if grabbed is None or grabbed.winfo_toplevel() is self.window:
                    release_modal_grab_before_system_command(self.window, WM_SYSCOMMAND, request[1])
                    # Tk's grab tree update must finish before its native handler.
                    self._minimize_job = self.window.after_idle(lambda: self._forward_minimize(*request))
        except Exception:
            logger.exception("[RESOURCE WINDOW] Cannot process modal minimize request")
        if not self.closed:
            self._poll_job = self.window.after(24, self._poll)

    def _forward_minimize(self, hwnd, wparam, lparam):
        self._minimize_job = None
        if not self.closed and self.hwnd == hwnd:
            grabbed = self.window.grab_current()
            if grabbed is not None and grabbed.winfo_toplevel() is not self.window:
                return
            self._forwarding = True
            if not self._user32.PostMessageW(hwnd, WM_SYSCOMMAND, wparam, lparam):
                self._forwarding = False
                logger.warning("[RESOURCE WINDOW] Could not forward native minimize command")
                if self.window.grab_current() is None and self.window.state() in {"normal", "zoomed"}:
                    self.window.grab_set()

    def pause(self):
        self._requested_command = None
        self._forwarding = False
        if self._poll_job is not None:
            self.window.after_cancel(self._poll_job)
            self._poll_job = None
        if self._minimize_job is not None:
            self.window.after_cancel(self._minimize_job)
            self._minimize_job = None

    def close(self):
        self.closed = True
        self.pause()
        if self.hwnd:
            self._comctl32.RemoveWindowSubclass(self.hwnd, self._callback, self._id)
            self.hwnd = None


def ensure_native_modal_minimize(window):
    """Called after mapping, never creates/maps a window during hidden building."""
    if bool(getattr(window, "_aat_native_minimize_unavailable", False)):
        return
    tk_app = getattr(window, "tk", None)
    if tk_app is None or tk_app.call("tk", "windowingsystem") != "win32":
        return
    hook = getattr(window, "_aat_native_minimize_hook", None)
    try:
        if hook is None:
            hook = NativeModalMinimize(window)
            window._aat_native_minimize_hook = hook
        hook.attach()
    except Exception:
        window._aat_native_minimize_unavailable = True
        logger.exception("[RESOURCE WINDOW] Cannot enable native modal minimization")
