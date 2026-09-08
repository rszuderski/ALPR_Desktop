#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window recovery helpers for the main Tk application."""

from ..config import logger


def _safe_widget_exists(widget) -> bool:
    try:
        return widget is not None and bool(widget.winfo_exists())
    except Exception:
        return False


def _safe_toplevel(widget):
    if not _safe_widget_exists(widget):
        return None
    try:
        return widget.winfo_toplevel()
    except Exception:
        return None


def _safe_window_state(window) -> str:
    try:
        return str(window.state() or "")
    except Exception:
        return ""


def _window_recovery_suspended(app) -> bool:
    try:
        return bool(
            getattr(app, "_native_file_dialog_active", False)
            or getattr(app, "_window_recovery_suspended", False)
        )
    except Exception:
        return False


def _window_recovery_opted_out(window) -> bool:
    try:
        return bool(getattr(window, "_aat_skip_window_recovery", False))
    except Exception:
        return False


def configure_minimizable_modal(window) -> None:
    """Let the native window manager minimize a modal without root recovery."""
    window._aat_skip_window_recovery = True
    if bool(getattr(window, "_aat_native_modal_state_bound", False)):
        return
    window._aat_native_modal_state_bound = True

    def _on_unmap(event):
        if event.widget is not window:
            return
        state = _safe_window_state(window)
        hook = getattr(window, "_aat_native_minimize_hook", None)
        if hook is not None and state in {"iconic", "withdrawn"}:
            hook.pause()
        if state != "iconic":
            return
        try:
            if window.grab_current() is window:
                window.grab_release()
        except Exception:
            pass

    def _on_map(event):
        if event.widget is not window or _safe_window_state(window) not in {"normal", "zoomed"}:
            return
        from .native_modal_minimize import ensure_native_modal_minimize
        ensure_native_modal_minimize(window)
        try:
            if window.grab_current() is None:
                window.grab_set()
        except Exception:
            pass

    def _on_configure(event):
        hook = getattr(window, "_aat_native_minimize_hook", None)
        if (event.widget is window and hook is not None and not hook.hwnd and not hook.closed
                and window.winfo_ismapped()):
            # Tk may recreate its wrapper after a wm attributes/transient change.
            from .native_modal_minimize import ensure_native_modal_minimize
            ensure_native_modal_minimize(window)

    def _on_destroy(event):
        if event.widget is window:
            hook = getattr(window, "_aat_native_minimize_hook", None)
            if hook is not None:
                hook.close()

    window.bind("<Unmap>", _on_unmap, add="+")
    window.bind("<Map>", _on_map, add="+")
    window.bind("<Configure>", _on_configure, add="+")
    window.bind("<Destroy>", _on_destroy, add="+")


def restore_visible_modal(window) -> bool:
    """Return from a child dialog, respecting the user's minimized state."""
    if not _safe_widget_exists(window) or _safe_window_state(window) not in {"normal", "zoomed"}:
        return False
    try:
        window.lift()
        window.focus_force()
        window.grab_set()
        return True
    except Exception:
        return False


def _append_recoverable_toplevel(app, window, result: list, seen: set[str]) -> None:
    top = _safe_toplevel(window)
    if top is None or top is getattr(app, "root", None):
        return
    if not _safe_widget_exists(top):
        return
    if _window_recovery_opted_out(top):
        return
    key = str(top)
    if key in seen:
        return
    seen.add(key)
    result.append(top)


def iter_recoverable_toplevels(app) -> list:
    """Return modeless app windows that should follow root activation."""

    result = []
    seen: set[str] = set()

    for attr_name in (
        "_global_terminal_window",
        "_mobile_export_center_dialog",
        "_mobile_report_browser_dialog",
        "_free_mode_assistant_context_override_owner",
    ):
        _append_recoverable_toplevel(app, getattr(app, attr_name, None), result, seen)

    host = getattr(app, "_mobile_export_menu_host", None)
    if host is not None:
        for attr_name in ("_mobile_export_center_dialog", "_mobile_report_browser_dialog"):
            _append_recoverable_toplevel(app, getattr(host, attr_name, None), result, seen)

    try:
        tabs = list(app._iter_loaded_tabs())
    except Exception:
        tabs = []
    for tab in tabs:
        for attr_name in ("_mobile_export_center_dialog", "_mobile_report_browser_dialog"):
            _append_recoverable_toplevel(app, getattr(tab, attr_name, None), result, seen)

    try:
        root_children = list(app.root.winfo_children())
    except Exception:
        root_children = []
    for child in root_children:
        top = _safe_toplevel(child)
        if top is child:
            _append_recoverable_toplevel(app, child, result, seen)

    return result


def _release_stale_grab(app) -> None:
    try:
        grabbed = app.root.grab_current()
    except Exception:
        grabbed = None
    if grabbed is None:
        return
    top = _safe_toplevel(grabbed)
    state = _safe_window_state(top)
    try:
        visible = bool(top.winfo_viewable()) if top is not None else False
    except Exception:
        visible = False
    if state in {"withdrawn", "iconic"} or not visible:
        try:
            grabbed.grab_release()
        except Exception:
            pass


def restore_app_window_stack(app, *, active_window=None, force_topmost: bool = False) -> None:
    """Raise root plus registered modeless windows without making them modal."""

    if _window_recovery_suspended(app):
        return

    root = getattr(app, "root", None)
    if not _safe_widget_exists(root):
        return

    _release_stale_grab(app)

    windows = []
    for window in iter_recoverable_toplevels(app):
        if not _safe_widget_exists(window):
            continue
        state = _safe_window_state(window)
        if state == "withdrawn":
            continue
        windows.append(window)

    try:
        root.deiconify()
    except Exception:
        pass
    try:
        root.lift()
    except Exception:
        pass

    for window in windows:
        try:
            if _safe_window_state(window) == "iconic":
                window.deiconify()
        except Exception:
            pass
        try:
            window.lift(root)
        except Exception:
            try:
                window.lift()
            except Exception:
                pass

    if force_topmost:
        to_toggle = [root] + windows
        for window in to_toggle:
            try:
                window.attributes("-topmost", True)
            except Exception:
                pass

        def _unset_topmost(targets=tuple(to_toggle)) -> None:
            for target in targets:
                try:
                    if _safe_widget_exists(target):
                        target.attributes("-topmost", False)
                except Exception:
                    pass

        try:
            if app._window_restore_topmost_after_id is not None:
                root.after_cancel(app._window_restore_topmost_after_id)
        except Exception:
            pass
        try:
            app._window_restore_topmost_after_id = root.after(180, _unset_topmost)
        except Exception:
            _unset_topmost()

    focus_target = active_window if _safe_widget_exists(active_window) else root
    try:
        focus_target.focus_force()
    except Exception:
        try:
            focus_target.focus_set()
        except Exception:
            pass


def register_recoverable_toplevel(app, window, *, attr_name: str = "") -> None:
    """Remember a detached tool window so it returns with the main app."""

    if not _safe_widget_exists(window):
        return
    if attr_name:
        try:
            setattr(app, attr_name, window)
        except Exception:
            pass
    try:
        setattr(window, "_aat_recover_with_app", True)
    except Exception:
        pass
    try:
        window.group(app.root)
    except Exception:
        pass

    try:
        already_bound = bool(getattr(window, "_aat_recovery_bindings_installed", False))
    except Exception:
        already_bound = False
    if already_bound:
        return
    try:
        setattr(window, "_aat_recovery_bindings_installed", True)
    except Exception:
        pass

    def _restore_from_tool_window(event=None) -> None:
        if _window_recovery_suspended(app):
            return
        try:
            if getattr(event, "widget", None) is not window:
                return
        except Exception:
            pass
        try:
            restore_app_window_stack(app, active_window=window)
        except Exception:
            pass

    try:
        window.bind("<FocusIn>", _restore_from_tool_window, add="+")
        window.bind("<Map>", _restore_from_tool_window, add="+")
    except Exception:
        pass


def close_floating_overlays_for_window_state(app):
    closer = getattr(app, "_close_menu_dropdown", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass

    hide_tooltip = getattr(app, "_hide_simple_tooltip", None)
    if callable(hide_tooltip):
        try:
            hide_tooltip()
        except Exception:
            pass


def release_window_grabs_for_recovery(app):
    seen = set()
    candidates = []
    try:
        current_grab = app.root.grab_current()
    except Exception:
        current_grab = None
    if current_grab is not None:
        candidates.append(current_grab)
    candidates.extend(
        [
            app.root,
            getattr(app, "startup_overlay_window", None),
            getattr(app, "startup_overlay_frame", None),
            getattr(app, "help_overlay_frame", None),
            getattr(app, "_global_terminal_window", None),
        ]
    )
    candidates.extend(iter_recoverable_toplevels(app))
    try:
        candidates.extend(list(app.root.winfo_children()))
    except Exception:
        pass

    for widget in candidates:
        if widget is None:
            continue
        key = str(widget)
        if key in seen:
            continue
        seen.add(key)
        try:
            widget.grab_release()
        except Exception:
            pass

    try:
        app.root.grab_release()
    except Exception:
        pass


def release_preview_fullscreens_for_recovery(app):
    for tab in list(app._iter_loaded_tabs()):
        try:
            if bool(getattr(tab, "_preview_fullscreen_active", False)):
                exit_fullscreen = getattr(tab, "_set_preview_fullscreen", None)
                if callable(exit_fullscreen):
                    exit_fullscreen(False)
        except Exception as e:
            logger.debug(f"Nie udało się zamknąć fullscreen preview podczas recovery okna: {e}")


def on_root_unmap(app, event=None):
    if _window_recovery_suspended(app):
        return None
    if getattr(event, "widget", None) is not app.root:
        return None
    try:
        root_state = str(app.root.state())
    except Exception:
        root_state = ""
    if root_state != "iconic":
        return None
    app._window_restore_pending = True
    close_floating_overlays_for_window_state(app)
    release_window_grabs_for_recovery(app)
    try:
        if bool(getattr(app, "_help_overlay_forced_visible", False)):
            app.hide_context_help_overlay()
    except Exception:
        pass
    release_preview_fullscreens_for_recovery(app)
    return None


def on_root_map(app, event=None):
    if _window_recovery_suspended(app):
        return None
    if getattr(event, "widget", None) is not app.root:
        return None
    app._window_restore_pending = True
    schedule_root_recovery(app, delay_ms=60, reset_attempts=True)
    return None


def on_root_visibility(app, event=None):
    if _window_recovery_suspended(app):
        return None
    if getattr(event, "widget", None) is not app.root:
        return None
    if not bool(getattr(app, "_window_restore_pending", False)) and int(getattr(app, "_window_restore_attempts", 0) or 0) <= 0:
        return None
    schedule_root_recovery(app, delay_ms=40)
    return None


def on_root_focus_in(app, event=None):
    if _window_recovery_suspended(app):
        return None
    if getattr(event, "widget", None) is not app.root:
        return None
    if not bool(getattr(app, "_window_restore_pending", False)) and int(getattr(app, "_window_restore_attempts", 0) or 0) <= 0:
        restore_app_window_stack(app)
        return None
    schedule_root_recovery(app, delay_ms=30)
    return None


def schedule_root_recovery(app, delay_ms: int = 60, *, reset_attempts: bool = False):
    if _window_recovery_suspended(app):
        return
    if reset_attempts:
        app._window_restore_attempts = 0
    try:
        if app._window_restore_after_id is not None:
            app.root.after_cancel(app._window_restore_after_id)
    except Exception:
        pass
    try:
        app._window_restore_after_id = app.root.after(max(0, int(delay_ms)), lambda: recover_root_after_map(app))
    except Exception:
        app._window_restore_after_id = None
        recover_root_after_map(app)


def recover_root_after_map(app):
    app._window_restore_after_id = None
    if _window_recovery_suspended(app):
        return
    try:
        root_state = str(app.root.state())
    except Exception:
        root_state = ""
    if root_state == "iconic":
        close_floating_overlays_for_window_state(app)
        app._window_restore_attempts = int(getattr(app, "_window_restore_attempts", 0) or 0) + 1
        if app._window_restore_attempts <= 24:
            delay_ms = min(900, 90 + (app._window_restore_attempts * 45))
            schedule_root_recovery(app, delay_ms=delay_ms)
        return
    app._window_restore_attempts = 0
    app._window_restore_pending = False

    release_window_grabs_for_recovery(app)
    try:
        if bool(getattr(app, "_help_overlay_forced_visible", False)):
            app.hide_context_help_overlay()
    except Exception:
        pass

    try:
        app.root.deiconify()
    except Exception:
        pass

    try:
        app.root.lift()
    except Exception:
        pass

    try:
        restore_app_window_stack(app, force_topmost=False)
    except Exception:
        pass

    try:
        if str(app.root.tk.call("tk", "windowingsystem")).lower() == "win32":
            restore_app_window_stack(app, force_topmost=True)
    except Exception:
        pass

    try:
        app.root.focus_force()
    except Exception:
        try:
            app.root.focus_set()
        except Exception:
            pass
