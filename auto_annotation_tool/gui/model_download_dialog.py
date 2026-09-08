"""A consented download splash; the worker never calls Tk."""
from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox

from ..model_download import DownloadCancelled, DownloadProgress, download_model, official_asset_url
from .app_theme_definitions import get_runtime_palette
from .canvas_progress_overlay import CanvasProgressOverlay


def transfer_caption(state: DownloadProgress) -> str:
    received = state.received / 1048576
    total = f"{state.total / 1048576:.1f} MB" if state.total else "rozmiar niepodany przez serwer"
    pct = f"{state.percent:.1f}%" if state.percent is not None else "Pobieranie"
    speed = state.bytes_per_second / 1048576
    return f"{pct} | {received:.1f} / {total} | {speed:.2f} MB/s"


def download_models_with_splash(owner, items) -> bool:
    parent = owner.frame.winfo_toplevel()
    palette = get_runtime_palette(owner)
    previous_grab = parent.grab_current()
    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    dialog._aat_skip_window_recovery = True
    dialog.overrideredirect(True)
    dialog.configure(bg=palette["panel"])
    cancel = threading.Event()
    done = threading.Event()
    lock = threading.Lock()
    latest = [None]
    outcome = [False, None]
    closing = [False]

    def request_cancel():
        cancel.set()
        overlay.configure_cancel(text="Anulowanie...", enabled=False)
        overlay.update_progress(meta_text="Anulowanie pobierania, oczekiwanie na zakończenie połączenia...")

    overlay = CanvasProgressOverlay(dialog, palette=palette, overlay_bg=palette["panel"], on_cancel=request_cancel)
    overlay.show(title="Pobieranie modelu pojazdów", details="Łączenie z oficjalnym repozytorium Ultralytics...")
    dialog.protocol("WM_DELETE_WINDOW", request_cancel)
    dialog.bind("<Escape>", lambda _event: request_cancel())
    width, height = min(620, parent.winfo_screenwidth()), min(300, parent.winfo_screenheight())
    x = max(0, min(parent.winfo_screenwidth() - width, parent.winfo_rootx() + (parent.winfo_width() - width) // 2))
    y = max(0, min(parent.winfo_screenheight() - height, parent.winfo_rooty() + (parent.winfo_height() - height) // 2))
    dialog.geometry(f"{width}x{height}+{x}+{y}")

    def publish(label, index, state):
        with lock:
            latest[0] = (label, index, state)

    def worker():
        try:
            for index, item in enumerate(items, 1):
                if cancel.is_set():
                    raise DownloadCancelled()
                label = str(item.get("label") or Path(item["target_path"]).name)
                url = official_asset_url(str(item["asset_name"]))
                download_model(url, Path(item["target_path"]), cancel,
                               lambda state: publish(label, index, state))
            outcome[0] = not cancel.is_set()
        except DownloadCancelled:
            pass
        except Exception as exc:
            outcome[1] = str(exc)
        finally:
            done.set()

    def finish():
        closing[0] = True
        dialog.destroy()

    def poll():
        with lock:
            update = latest[0]
            latest[0] = None
        if update and not cancel.is_set():
            label, index, state = update
            overlay._title_lbl.configure(text=f"{label} ({index}/{len(items)})")
            overlay._body_lbl.configure(text=f"Źródło pliku:\n{state.source}")
            overlay.update_progress(pct=state.percent or 0.0, filename=Path(items[index - 1]["target_path"]).name,
                                    meta_text=transfer_caption(state))
        if done.is_set():
            dialog.after(150 if outcome[0] else 0, finish)
        else:
            dialog.after(80, poll)

    def on_destroy(event):
        if event.widget is dialog and not closing[0]:
            cancel.set()

    dialog.bind("<Destroy>", on_destroy, add="+")
    dialog.deiconify()
    dialog.lift()
    dialog.grab_set()
    dialog.after(20, lambda: threading.Thread(target=worker, daemon=True, name="Z2-model-download").start())
    dialog.after(80, poll)
    parent.wait_window(dialog)
    if previous_grab is not None:
        try:
            if previous_grab.winfo_exists() and previous_grab.winfo_viewable() and parent.grab_current() is None:
                previous_grab.grab_set()
        except tk.TclError:
            pass
    if outcome[1] and not cancel.is_set():
        messagebox.showerror("Nie pobrano modelu", outcome[1], parent=parent)
    return bool(outcome[0] and not cancel.is_set())
