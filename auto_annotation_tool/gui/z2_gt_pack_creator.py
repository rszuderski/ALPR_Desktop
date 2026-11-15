"""Compact Z2 wizard for the canonical writable ALPR GT Pack."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import CONFIG
from ..gt_resource_companions import (
    DEFAULT_WORKING_PACK_NAME,
    ensure_working_gt_pack,
)
from ..utils import get_image_files


def _path_key(path: Path | str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path or "").replace("\\", "/").strip().lower()


def _current_input_dir(host) -> Path | None:
    candidates = [getattr(host, "current_input_dir", None)]
    try:
        candidates.append(host.input_dir_var.get())
    except Exception:
        pass
    for raw in candidates:
        if not str(raw or "").strip():
            continue
        try:
            path = Path(raw)
        except Exception:
            continue
        if path.exists() and path.is_dir():
            return path
    return None


def _show_error(host, title: str, message: str, *, parent=None) -> None:
    app = getattr(host, "app", None)
    themed = getattr(app, "themed_error", None)
    if callable(themed):
        try:
            themed(title, message, parent=parent)
            return
        except TypeError:
            try:
                themed(title, message)
                return
            except Exception:
                pass
        except Exception:
            pass
    messagebox.showerror(title, message, parent=parent)


def _show_info(host, title: str, message: str, *, parent=None) -> None:
    app = getattr(host, "app", None)
    themed = getattr(app, "themed_info", None)
    if callable(themed):
        try:
            themed(title, message, parent=parent, tone="success")
            return
        except TypeError:
            try:
                themed(title, message)
                return
            except Exception:
                pass
        except Exception:
            pass
    messagebox.showinfo(title, message, parent=parent)


def open_gt_pack_creator(host) -> None:
    """Create/open ``current_work.alprgt`` and activate it in Z2 free mode."""
    is_free_mode = getattr(host, "_is_free_mode_session_context", None)
    if callable(is_free_mode):
        try:
            if not bool(is_free_mode()):
                _show_info(
                    host,
                    "Roboczy pakiet GT",
                    "Ta funkcja jest dostępna w trybie swobodnym Z2.",
                    parent=getattr(host, "frame", None),
                )
                return
        except Exception:
            pass

    parent = getattr(host, "frame", None)
    window = tk.Toplevel(parent)
    window.title("Utwórz roboczy pakiet GT")
    try:
        window.transient(parent.winfo_toplevel())
    except Exception:
        pass
    window.resizable(False, False)
    window.minsize(620, 285)

    outer = ttk.Frame(window, padding=16)
    outer.pack(fill=tk.BOTH, expand=True)
    outer.columnconfigure(0, weight=1)

    ttk.Label(
        outer,
        text="Roboczy Ground Truth dla folderu obrazów",
        font=("Segoe UI", 11, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w")

    ttk.Label(
        outer,
        text=(
            f"Z2 używa jednego zapisywalnego pakietu: {DEFAULT_WORKING_PACK_NAME}. "
            "Pakiet nie kopiuje zdjęć. Przy normalnej pracy Z2 zapisze do niego "
            "tożsamość obrazu, polygon tablicy, tekst GT i układ 1R/2R."
        ),
        justify=tk.LEFT,
        wraplength=570,
    ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 14))

    current = _current_input_dir(host)
    initial_dir = current or Path(CONFIG.DIR_1_RAW)
    folder_var = tk.StringVar(value=str(current or ""))
    status_var = tk.StringVar(value="")

    ttk.Label(outer, text="Folder obrazów:").grid(row=2, column=0, sticky="w")
    ttk.Entry(
        outer,
        textvariable=folder_var,
        width=62,
    ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 8), padx=(0, 8))

    def refresh_status() -> None:
        raw = str(folder_var.get() or "").strip()
        if not raw:
            status_var.set("Wskaż folder z obrazami.")
            return
        try:
            path = Path(raw)
            count = len(get_image_files(path)) if path.is_dir() else 0
        except Exception:
            count = 0
        if count:
            status_var.set(
                f"Znaleziono obrazów: {count}   •   writable: {DEFAULT_WORKING_PACK_NAME}"
            )
        else:
            status_var.set("Folder nie zawiera obsługiwanych obrazów.")

    def choose_folder() -> None:
        selected = filedialog.askdirectory(
            parent=window,
            title="Wybierz folder obrazów dla roboczego GT",
            initialdir=str(initial_dir),
        )
        if selected:
            folder_var.set(selected)
            refresh_status()

    ttk.Button(
        outer,
        text="Wybierz…",
        command=choose_folder,
    ).grid(row=3, column=2, sticky="e", pady=(3, 8))

    ttk.Label(
        outer,
        textvariable=status_var,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 14))

    ttk.Label(
        outer,
        text=(
            "Jeżeli current_work.alprgt już istnieje i jest poprawny, zostanie "
            "otwarty bez czyszczenia ani nadpisywania danych."
        ),
        justify=tk.LEFT,
        wraplength=570,
    ).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 14))

    buttons = ttk.Frame(outer)
    buttons.grid(row=6, column=0, columnspan=3, sticky="ew")
    buttons.columnconfigure(0, weight=1)

    def create_and_use() -> None:
        raw_folder = str(folder_var.get() or "").strip()
        if not raw_folder:
            _show_error(
                host,
                "Roboczy pakiet GT",
                "Najpierw wskaż folder obrazów.",
                parent=window,
            )
            return

        image_dir = Path(raw_folder)
        try:
            images = get_image_files(image_dir)
        except Exception:
            images = []
        if not images:
            _show_error(
                host,
                "Roboczy pakiet GT",
                "Wybrany folder nie zawiera obsługiwanych obrazów.",
                parent=window,
            )
            return

        try:
            result = ensure_working_gt_pack(image_dir)
        except Exception as exc:
            _show_error(
                host,
                "Nie można przygotować roboczego GT",
                str(exc),
                parent=window,
            )
            return

        current_input = _current_input_dir(host)
        same_current = bool(
            current_input is not None
            and _path_key(current_input) == _path_key(image_dir)
        )

        if same_current:
            try:
                from . import z2_gt_companion_flow
                z2_gt_companion_flow.prepare_free_mode_gt_binding(
                    host,
                    image_dir,
                    parent=window,
                )
            except Exception as exc:
                _show_error(
                    host,
                    "Pakiet GT gotowy, ale nieaktywny",
                    f"Pakiet istnieje:\n{result['path']}\n\nNie udało się aktywować go w Z2:\n{exc}",
                    parent=window,
                )
                return
        else:
            switcher = getattr(host, "_switch_annotation_input_dir", None)
            if not callable(switcher):
                _show_error(
                    host,
                    "Pakiet GT gotowy, ale nieaktywny",
                    f"Pakiet istnieje:\n{result['path']}\n\nZ2 nie udostępnia zmiany folderu wejściowego.",
                    parent=window,
                )
                return
            try:
                switched = switcher(image_dir, show_hint=False)
            except Exception as exc:
                _show_error(
                    host,
                    "Pakiet GT gotowy, ale nieaktywny",
                    f"Pakiet istnieje:\n{result['path']}\n\nNie udało się przełączyć Z2:\n{exc}",
                    parent=window,
                )
                return
            if switched is False:
                _show_error(
                    host,
                    "Pakiet GT gotowy, ale nieaktywny",
                    (
                        f"Pakiet istnieje:\n{result['path']}\n\n"
                        "Z2 nie przełączył folderu wejściowego, np. z powodu niezapisanej edycji."
                    ),
                    parent=window,
                )
                return

        action = "Utworzono" if result.get("created") else "Otworzono"
        summary = dict(result.get("summary") or {})
        _show_info(
            host,
            "Roboczy pakiet GT gotowy",
            (
                f"{action}:\n{result['path']}\n\n"
                f"Obrazy zapisane już w packu: {int(summary.get('images', 0) or 0)}\n"
                f"Tablice zapisane już w packu: {int(summary.get('plates', 0) or 0)}\n\n"
                "Możesz teraz normalnie pracować w Z2."
            ),
            parent=window,
        )
        window.destroy()

    ttk.Button(
        buttons,
        text="Anuluj",
        command=window.destroy,
    ).pack(side=tk.RIGHT)
    ttk.Button(
        buttons,
        text="Utwórz i użyj",
        command=create_and_use,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    refresh_status()
    try:
        window.grab_set()
        window.focus_set()
    except Exception:
        pass
