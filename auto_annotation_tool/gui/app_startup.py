#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Startup overlay and startup-readiness helpers for the main app shell."""

import faulthandler
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ..config import CONFIG, logger
from .lazy_notebook_tab import _LazyNotebookTab

try:
    from .tab_help import HelpTab
except ImportError:
    HelpTab = None


def enable_fatal_crash_logging(app):
    try:
        crash_log_path = Path(CONFIG.WORKSPACE_DIR) / "fatal_crash.log"
        crash_log_path.parent.mkdir(parents=True, exist_ok=True)
        app._faulthandler_stream = open(crash_log_path, "a", encoding="utf-8")
        faulthandler.enable(app._faulthandler_stream, all_threads=True)
    except Exception as e:
        logger.debug(f"Nie udało się włączyć fatal crash log dla GUI: {e}")


def raise_startup_overlay(app):
    overlay_window = getattr(app, "startup_overlay_window", None)
    overlay = getattr(app, "startup_overlay_frame", None)
    card = getattr(app, "startup_overlay_card", None)
    if overlay is None:
        return

    try:
        app.root.update_idletasks()
    except Exception:
        pass

    if overlay_window is not None:
        try:
            splash_w = max(380, min(560, int(card.winfo_reqwidth() or 460)))
            splash_h = max(112, min(180, int(card.winfo_reqheight() or 132)))
            screen_w = max(1, int(app.root.winfo_screenwidth() or 1))
            screen_h = max(1, int(app.root.winfo_screenheight() or 1))
            pos_x = max(0, int((screen_w - splash_w) / 2))
            pos_y = max(0, int((screen_h - splash_h) / 2))
            overlay_window.geometry(f"{splash_w}x{splash_h}+{pos_x}+{pos_y}")
            overlay_window.lift()
            overlay_window.attributes("-topmost", True)
        except Exception:
            pass
    else:
        try:
            overlay.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            pass

        try:
            overlay.lift()
        except Exception:
            pass

        for widget_name in ("menu_bar_frame", "notebook", "info_panel_frame", "help_overlay_frame"):
            widget = getattr(app, widget_name, None)
            if widget is None:
                continue
            try:
                overlay.lift(widget)
            except Exception:
                pass

    if card is not None:
        try:
            card.lift()
        except Exception:
            pass


def flush_startup_overlay(app):
    app._raise_startup_overlay()
    try:
        app.root.update_idletasks()
        app.root.update()
    except Exception:
        pass
    app._raise_startup_overlay()


def consume_startup_overlay_event(event=None):
    return "break"


def prepare_main_window_for_startup(app):
    try:
        app.root.withdraw()
        app._main_window_hidden_for_startup = True
        app._main_window_revealed = False
    except Exception:
        app._main_window_hidden_for_startup = False
        app._main_window_revealed = False


def reveal_main_window_after_startup(app):
    if bool(getattr(app, "_main_window_revealed", False)):
        return

    try:
        app.root.deiconify()
    except Exception:
        pass

    try:
        app.root.update_idletasks()
    except Exception:
        pass

    try:
        app.root.state("zoomed")
    except Exception:
        try:
            app.root.attributes("-fullscreen", True)
        except Exception:
            pass

    try:
        app.root.lift()
    except Exception:
        pass

    try:
        app.root.focus_force()
    except Exception:
        try:
            app.root.focus_set()
        except Exception:
            pass

    app._main_window_hidden_for_startup = False
    app._main_window_revealed = True

    try:
        app.root.after_idle(lambda: app._refresh_adaptive_wraps(app.root))
    except Exception:
        pass


def show_startup_overlay(app):
    if app.startup_overlay_frame is not None:
        return

    palette = app.palette
    app.startup_overlay_shown_at = time.monotonic()
    app._startup_progress_peak = 0.0
    overlay_parent = app.root

    try:
        overlay_window = tk.Toplevel(app.root)
        overlay_window.withdraw()
        overlay_window.overrideredirect(True)
        try:
            overlay_window.attributes("-topmost", True)
        except Exception:
            pass
        try:
            overlay_window.resizable(False, False)
        except Exception:
            pass
        overlay_window.configure(bg=palette.get("bg", "#1e1e1e"))
        app.startup_overlay_window = overlay_window
        overlay_parent = overlay_window
    except Exception:
        app.startup_overlay_window = None
        overlay_parent = app.root

    app.startup_overlay_frame = tk.Frame(
        overlay_parent,
        bg=palette.get("bg", "#1e1e1e"),
        bd=0,
        highlightthickness=0,
    )
    app.startup_overlay_frame.pack(fill=tk.BOTH, expand=True)

    app.startup_overlay_card = tk.Frame(
        app.startup_overlay_frame,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        padx=18,
        pady=16,
    )
    app.startup_overlay_card.pack(fill=tk.BOTH, expand=True)

    app.startup_overlay_title_lbl = tk.Label(
        app.startup_overlay_card,
        text="Ładowanie danych aplikacji",
        font=("Segoe UI", 11, "bold"),
        fg=palette.get("fg", "#f3f3f3"),
        bg=palette.get("panel", "#252526"),
        anchor="w",
    )
    app.startup_overlay_title_lbl.pack(fill=tk.X, pady=(0, 8))

    app.startup_overlay_progress = ttk.Progressbar(
        app.startup_overlay_card,
        orient=tk.HORIZONTAL,
        mode="determinate",
        maximum=100,
        variable=app.startup_progress_var,
        length=420,
        style="Horizontal.TProgressbar",
    )
    app.startup_overlay_progress.pack(fill=tk.X)

    app.startup_overlay_status_lbl = tk.Label(
        app.startup_overlay_card,
        textvariable=app.startup_status_var,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=palette.get("panel", "#252526"),
        anchor="w",
    )
    app.startup_overlay_status_lbl.pack(fill=tk.X, pady=(8, 0))

    for widget in (
        app.startup_overlay_card,
        app.startup_overlay_title_lbl,
        app.startup_overlay_status_lbl,
        app.startup_overlay_progress,
    ):
        if widget is None:
            continue
        for sequence in (
            "<ButtonPress-1>",
            "<ButtonRelease-1>",
            "<Double-Button-1>",
            "<MouseWheel>",
            "<Button-4>",
            "<Button-5>",
            "<KeyPress>",
            "<KeyRelease>",
            "<Tab>",
        ):
            try:
                widget.bind(sequence, app._consume_startup_overlay_event, add="+")
            except Exception:
                pass

    if app.startup_overlay_window is not None:
        try:
            app._raise_startup_overlay()
            app.startup_overlay_window.deiconify()
        except Exception:
            pass
    else:
        try:
            app.startup_overlay_frame.lift()
            app.startup_overlay_card.lift()
        except Exception:
            pass
    try:
        app.root.after_idle(app._raise_startup_overlay)
    except Exception:
        pass
    try:
        if app.startup_overlay_window is not None:
            app.startup_overlay_window.grab_set()
        else:
            app.startup_overlay_frame.grab_set()
    except Exception:
        pass
    try:
        if app.startup_overlay_window is not None:
            app.startup_overlay_window.focus_force()
        else:
            app.startup_overlay_frame.focus_force()
    except Exception:
        try:
            if app.startup_overlay_window is not None:
                app.startup_overlay_window.focus_set()
            else:
                app.startup_overlay_frame.focus_set()
        except Exception:
            pass
    app._flush_startup_overlay()


def set_startup_progress(app, value: int | float, message: str = None):
    if app.startup_overlay_frame is None:
        return
    try:
        requested = max(0.0, min(100.0, float(value or 0.0)))
        peak = max(float(getattr(app, "_startup_progress_peak", 0.0) or 0.0), requested)
        app._startup_progress_peak = peak
        app.startup_progress_var.set(peak)
    except Exception:
        pass
    if message is not None:
        try:
            app.startup_status_var.set(str(message))
        except Exception:
            pass
    app._flush_startup_overlay()


def hide_startup_overlay(app):
    overlay_window = getattr(app, "startup_overlay_window", None)
    overlay = getattr(app, "startup_overlay_frame", None)
    if overlay is None and overlay_window is None:
        return
    try:
        if overlay_window is not None:
            overlay_window.grab_release()
        elif overlay is not None:
            overlay.grab_release()
    except Exception:
        pass
    try:
        if overlay_window is not None:
            overlay_window.destroy()
        elif overlay is not None:
            overlay.destroy()
    except Exception:
        pass
    app.startup_overlay_window = None
    app.startup_overlay_frame = None
    app.startup_overlay_card = None
    app.startup_overlay_title_lbl = None
    app.startup_overlay_status_lbl = None
    app.startup_overlay_progress = None


def wait_for_startup_marker(app, *, delay_ms: int = 0, timeout_ms: int = 4000) -> bool:
    flag = {"done": False}

    def mark_done():
        flag["done"] = True

    try:
        if delay_ms > 0:
            app.root.after(int(delay_ms), mark_done)
        else:
            app.root.after_idle(mark_done)
    except Exception:
        return False

    deadline = time.monotonic() + max(0.1, float(timeout_ms) / 1000.0)
    while not flag["done"] and time.monotonic() < deadline:
        try:
            app.root.update_idletasks()
            app.root.update()
        except Exception:
            break

    return bool(flag["done"])


def drain_startup_pending_events(app):
    if app.startup_overlay_frame is None:
        return

    # Przy leniwym ładowaniu nie wymuszamy pełnego renderu wszystkich zakładek.
    # Wystarczy opróżnić najbliższe after_idle i krótki marker dla widocznego Z1.
    app._wait_for_startup_marker(delay_ms=0, timeout_ms=800)
    app._wait_for_startup_marker(delay_ms=80, timeout_ms=1200)


def get_expected_startup_tab_keys() -> list[str]:
    expected = ["campaign", "annotation", "characters", "training"]
    if HelpTab:
        expected.append("help")
    return expected


def startup_tabs_ready(app) -> tuple[bool, list[str]]:
    expected = app._get_expected_startup_tab_keys()
    missing = [key for key in expected if key not in app.tabs]
    if missing:
        return False, missing

    try:
        notebook_tabs = list(app.notebook.tabs()) if getattr(app, "notebook", None) is not None else []
    except Exception:
        notebook_tabs = []

    if len(notebook_tabs) < len(expected):
        return False, ["notebook_tabs"]

    # Gdy okno glowne jest ukryte przez splash startowy, Tk potrafi raportowac
    # niestabilna geometrie mimo poprawnie zarejestrowanych zakladek. Nie blokuj
    # startu na samej geometrii; realny rozmiar sprawdzamy dopiero po pokazaniu okna.
    if not bool(getattr(app, "_main_window_hidden_for_startup", False)):
        try:
            notebook_width = int(app.notebook.winfo_width() or 0)
            notebook_height = int(app.notebook.winfo_height() or 0)
            if notebook_width < 120 or notebook_height < 120:
                return False, ["notebook_layout"]
        except Exception:
            return False, ["notebook_layout"]

        try:
            root_width = int(app.root.winfo_width() or 0)
            root_height = int(app.root.winfo_height() or 0)
            if root_width < 240 or root_height < 180:
                return False, ["window_layout"]
        except Exception:
            return False, ["window_layout"]

    for key in expected:
        try:
            frame = getattr(app.tabs.get(key), "frame", None)
            if frame is None or str(frame) not in notebook_tabs:
                return False, [key]
            if not bool(frame.winfo_exists()):
                return False, [key]
            tab_obj = app.tabs.get(key)
            if tab_obj is not None:
                startup_ready_getter = getattr(tab_obj, "is_startup_ui_ready", None)
                if callable(startup_ready_getter):
                    try:
                        if not bool(startup_ready_getter()):
                            return False, [key]
                    except Exception:
                        return False, [key]
            if isinstance(tab_obj, _LazyNotebookTab):
                continue
            try:
                if frame.winfo_reqwidth() <= 1 or frame.winfo_reqheight() <= 1:
                    return False, [key]
            except Exception:
                return False, [key]
        except Exception:
            return False, [key]

    return True, []


def schedule_startup_finalize(app, delay_ms: int = 0):
    pending = getattr(app, "_startup_finalize_after_id", None)
    if pending:
        try:
            app.root.after_cancel(pending)
        except Exception:
            pass
    if int(delay_ms or 0) == 0 and app._startup_finalize_attempts == 0:
        app._startup_ready_streak = 0
        app._startup_tabs_present_since = None
    try:
        app._startup_finalize_after_id = app.root.after(
            max(0, int(delay_ms)),
            app._finalize_startup_after_tabs_ready
        )
    except Exception:
        app._startup_finalize_after_id = None
        app._finalize_startup_after_tabs_ready()


def finalize_startup_after_tabs_ready(app):
    app._startup_finalize_after_id = None
    if app.startup_overlay_frame is None:
        return

    app._startup_finalize_attempts += 1
    app._set_startup_progress(96, "Finalizacja inicjalizacji zakładek...")
    app._drain_startup_pending_events()

    ready, missing = app._startup_tabs_ready()
    if ready:
        now = time.monotonic()
        if app._startup_tabs_present_since is None:
            app._startup_tabs_present_since = now
        app._startup_ready_streak += 1
        settled_for = now - float(app._startup_tabs_present_since or now)
        shown_for = now - float(getattr(app, "startup_overlay_shown_at", now) or now)
        if app._startup_ready_streak >= 2 and settled_for >= 0.35 and shown_for >= 0.8:
            app._set_startup_progress(100, "Ładowanie danych zakończone")
            app._reveal_main_window_after_startup()
            app._hide_startup_overlay()
            try:
                app.root.after(
                    350,
                    lambda: (
                        app._refresh_global_yolo_devices_async(silent=True)
                        if not getattr(app, "_global_yolo_devices_cache_ready", False)
                        else None
                    ),
                )
            except Exception:
                pass
            logger.info("GUI zainicjalizowane pomyślnie")
            return
        app._set_startup_progress(97, "Domykam start widocznej zakładki...")
        app._schedule_startup_finalize(250)
        return

    app._startup_ready_streak = 0
    app._startup_tabs_present_since = None

    if app._startup_finalize_attempts < 120:
        if missing:
            app._set_startup_progress(
                96,
                "Czekam na pełne załadowanie zakładek: " + ", ".join(missing)
            )
        else:
            app._set_startup_progress(96, "Czekam na pełne załadowanie zakładek...")
        app._schedule_startup_finalize(200)
        return

    missing_text = ", ".join(missing) if missing else "uklad notebooka nie osiagnal stabilnego stanu"
    app._set_startup_progress(
        96,
        f"Błąd ładowania zakładek: {missing_text}. Overlay pozostaje aktywny."
    )
    logger.error(f"Startup GUI nie domknął wszystkich zakładek: {missing_text}")
