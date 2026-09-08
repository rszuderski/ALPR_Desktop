"""Compact source fields for the mobile export candidate profile."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

from .web_slim_scrollbar import blend_hex_colors


def candidate_source_paths(candidate: dict | None) -> dict[str, Path | None]:
    candidate = candidate or {}
    run = candidate.get("run")
    run_dataset = run.get("dataset_path") if isinstance(run, dict) else getattr(run, "dataset_path", None)
    values = {
        "model": candidate.get("best_weights"),
        "dataset": candidate.get("dataset_path") or run_dataset,
    }
    paths = {}
    for key, value in values.items():
        raw = str(value or "").strip().strip('"')
        # Use the same working directory as the export request. Never substitute
        # the active project's dataset, a base model or another checkpoint copy.
        paths[key] = Path(os.path.abspath(raw)) if raw else None
    return paths


def reveal_source_path(path: Path) -> None:
    """Open a directory, or select the exact source file without executing it."""
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono na dysku: {path}")
    if sys.platform == "win32":
        if path.is_dir():
            os.startfile(str(path))
        else:
            subprocess.Popen(["explorer.exe", "/select,", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)] if path.is_dir() else ["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])


def relative_source_path(path: Path, candidate: dict | None) -> str:
    project = str((candidate or {}).get("project_root") or "").strip()
    for base in (Path(project) if project else None, Path.cwd()):
        if base is not None:
            try:
                return str(path.relative_to(base.absolute()))
            except ValueError:
                pass
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:  # A different Windows drive has no relative path.
        return str(path)


class MobileExportSourceLocations(tk.Frame):
    def __init__(self, master, *, bg: str, fg: str, muted: str, accent: str,
                 notify=None, column_widths=None):
        super().__init__(master, bg=bg)
        self.paths: dict[str, Path | None] = {}
        self.relative_paths: dict[str, str] = {}
        self.path_labels = {}
        self.open_buttons = {}
        self._rows = []
        self._notify = notify
        self._column_widths = column_widths
        self._font = tkfont.Font(root=self, family="Segoe UI", size=8)
        self._fg = fg
        self._muted = muted
        self.grid_columnconfigure(0, weight=1)
        for index, (key, title) in enumerate((
            ("model", "Ścieżka modelu"),
            ("dataset", "Ścieżka datasetu"),
        )):
            row_bg = blend_hex_colors(bg, fg, 0.025 if index % 2 else 0.012)
            row = tk.Frame(self, bg=row_bg, height=34)
            row.grid(row=index, column=0, sticky="ew", pady=(0, 3))
            row.grid_columnconfigure(0, minsize=112)
            row.grid_columnconfigure(1, weight=1)
            self._rows.append(row)
            tk.Label(row, text=title, bg=row_bg, fg=fg, anchor="w", font=self._font,
                     bd=0, padx=8, pady=8).grid(row=0, column=0, sticky="w")
            label = tk.Label(row, text="", bg=row_bg, fg=muted, anchor="w",
                             font=self._font, width=1, bd=0, padx=8, pady=8,
                             highlightthickness=0)
            label.grid(row=0, column=1, sticky="ew")
            label.bind("<Configure>", lambda event, source=key: self._fit_path(source, event.width))
            label.bind("<Button-3>", lambda _event, source=key: self.copy_path(source))
            open_button = tk.Button(row, text="Otwórz folder", command=lambda source=key: self.open_path(source),
                                    font=("Segoe UI", 7), padx=4, pady=1, bd=0, relief=tk.FLAT,
                                    highlightthickness=0, bg=blend_hex_colors(row_bg, muted, 0.07),
                                    fg=muted, activebackground=blend_hex_colors(row_bg, muted, 0.14),
                                    activeforeground=fg, disabledforeground=blend_hex_colors(muted, row_bg, 0.55),
                                    cursor="hand2")
            open_button.grid(row=0, column=2, padx=(3, 7))
            self.path_labels[key] = label
            self.open_buttons[key] = open_button
        self.bind("<Configure>", self._align_columns)
        self.set_candidate(None)

    def _align_columns(self, event) -> None:
        if callable(self._column_widths):
            width = self._column_widths(event.width)[0]
            for row in self._rows:
                row.grid_columnconfigure(0, minsize=width)

    def _fit_path(self, key: str, width: int) -> None:
        text = self.relative_paths.get(key, "—")
        available = max(12, width - 16)
        if self._font.measure(text) > available:
            parts = Path(text).parts
            for first in range(1, len(parts)):
                shortened = "…" + os.sep + os.path.join(*parts[first:])
                if self._font.measure(shortened) <= available:
                    text = shortened
                    break
            else:
                # A long filename still preserves its extension and tail.
                while text and self._font.measure("…" + text) > available:
                    text = text[1:]
                text = "…" + text
        self.path_labels[key].configure(text=text)

    def set_candidate(self, candidate: dict | None) -> None:
        self.paths = candidate_source_paths(candidate)
        for key, path in self.paths.items():
            self.relative_paths[key] = relative_source_path(path, candidate) if path else "—"
            self._fit_path(key, self.path_labels[key].winfo_width())
            self._update_open_button(key)

    def _update_open_button(self, key: str) -> bool:
        path = self.paths.get(key)
        exists = False
        is_dir = False
        if path:
            try:
                is_dir = path.is_dir()
                exists = path.is_file() or (key == "dataset" and is_dir)
            except (OSError, ValueError):
                pass
        self.open_buttons[key].configure(
            state="normal" if exists else "disabled",
            cursor="hand2" if exists else "arrow",
        )
        return exists

    def copy_path(self, key: str) -> str:
        path = self.paths.get(key)
        if path:
            try:
                self.clipboard_clear()
                self.clipboard_append(str(path))
            except tk.TclError:
                if callable(self._notify):
                    self._notify("Nie udało się skopiować ścieżki.", "error")
            else:
                if callable(self._notify):
                    self._notify("Skopiowano ścieżkę.", "success")
        return "break"

    def open_path(self, key: str) -> None:
        path = self.paths.get(key)
        if path is None:
            return
        try:
            reveal_source_path(path)
        except (OSError, ValueError) as exc:
            self._update_open_button(key)
            messagebox.showerror("Nie można otworzyć lokalizacji", str(exc), parent=self.winfo_toplevel())
