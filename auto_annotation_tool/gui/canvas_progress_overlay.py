from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk


class CanvasProgressOverlay:
    def __init__(
        self,
        host,
        *,
        palette: dict | None = None,
        overlay_bg: str = "#000000",
        on_cancel=None,
    ) -> None:
        palette = dict(palette or {})
        self._host = host
        self._palette = palette
        self._on_cancel = on_cancel

        panel_bg = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", "#2d2d30")
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        accent = palette.get("accent", "#0e639c")

        self._overlay = tk.Frame(
            host,
            bg=overlay_bg,
            bd=0,
            highlightthickness=0,
        )
        self._overlay.place_forget()
        for sequence in ("<Button-1>", "<ButtonRelease-1>", "<B1-Motion>"):
            try:
                self._overlay.bind(sequence, lambda _e: "break")
            except Exception:
                pass

        self._card = tk.Frame(
            self._overlay,
            bg=panel_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=16,
            pady=14,
        )
        self._card.place(relx=0.5, rely=0.5, anchor="center")
        try:
            self._card.bind("<Button-1>", lambda _e: "break")
        except Exception:
            pass

        self._title_lbl = tk.Label(
            self._card,
            text="Trwa autoanotacja",
            bg=panel_bg,
            fg=fg,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 10, "bold"),
        )
        self._title_lbl.pack(fill=tk.X)

        self._body_lbl = tk.Label(
            self._card,
            text="Lista i podgląd pozostają widoczne, ale są zablokowane do końca bieżącego runu.",
            bg=panel_bg,
            fg=muted,
            anchor="center",
            justify=tk.CENTER,
            wraplength=420,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self._body_lbl.pack(fill=tk.X, pady=(8, 0))

        self._meta_lbl = tk.Label(
            self._card,
            text="0% | 0/0 obrazów",
            bg=panel_bg,
            fg=muted,
            anchor="center",
            justify=tk.CENTER,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI Semibold", 9),
        )
        self._meta_lbl.pack(fill=tk.X, pady=(12, 4))

        self._progress_canvas = tk.Canvas(
            self._card,
            height=10,
            bg=panel_bg,
            bd=0,
            highlightthickness=0,
        )
        self._progress_canvas.pack(fill=tk.X)
        self._progress_trough_id = self._progress_canvas.create_rectangle(0, 0, 0, 0, width=0, fill=panel_alt)
        self._progress_fill_id = self._progress_canvas.create_rectangle(0, 0, 0, 0, width=0, fill=accent)
        try:
            self._progress_canvas.bind("<Configure>", lambda _e: self._render_progress_bar(), add="+")
        except Exception:
            pass

        self._file_lbl = tk.Label(
            self._card,
            text="",
            bg=panel_bg,
            fg=muted,
            anchor="center",
            justify=tk.CENTER,
            wraplength=420,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self._file_lbl.pack(fill=tk.X, pady=(8, 0))

        self._actions_row = tk.Frame(
            self._card,
            bg=panel_bg,
            bd=0,
            highlightthickness=0,
        )
        self._actions_row.pack(fill=tk.X, pady=(12, 0))

        self._cancel_btn = ttk.Button(
            self._actions_row,
            text="Anuluj",
            command=self._handle_cancel,
        )
        self._cancel_btn.pack(anchor=tk.CENTER)
        self._cancel_visible = True

        self._progress_value = 0.0
        self._visible = False

    @property
    def is_visible(self) -> bool:
        return bool(self._visible)

    def show(self, *, title: str = "", details: str = "") -> None:
        if title:
            try:
                self._title_lbl.configure(text=str(title))
            except Exception:
                pass
        if details:
            try:
                self._body_lbl.configure(text=str(details))
            except Exception:
                pass

        self.update_progress(pct=0.0, current=0, total=0, filename="", meta_text="0% | przygotowanie procesu")
        try:
            self._overlay.place(in_=self._host, relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
            self._overlay.lift()
        except Exception:
            pass
        self._visible = True

    def hide(self) -> None:
        self.update_progress(pct=0.0, current=0, total=0, filename="", meta_text="")
        try:
            self._overlay.place_forget()
        except Exception:
            pass
        self._visible = False

    def set_cancel_command(self, command) -> None:
        self._on_cancel = command

    def configure_cancel(
        self,
        *,
        command=None,
        text: str = "Anuluj",
        visible: bool = True,
        enabled: bool = True,
    ) -> None:
        self._on_cancel = command
        self._cancel_visible = bool(visible)
        try:
            self._cancel_btn.configure(
                text=str(text or "Anuluj"),
                state=(tk.NORMAL if enabled else tk.DISABLED),
            )
        except Exception:
            pass
        try:
            if self._cancel_visible:
                if not self._cancel_btn.winfo_manager():
                    self._cancel_btn.pack(anchor=tk.CENTER)
                if not self._actions_row.winfo_manager():
                    self._actions_row.pack(fill=tk.X, pady=(12, 0))
            else:
                self._cancel_btn.pack_forget()
                self._actions_row.pack_forget()
        except Exception:
            pass

    def _handle_cancel(self) -> None:
        callback = getattr(self, "_on_cancel", None)
        if not callable(callback):
            return
        try:
            callback()
        except Exception:
            pass

    def update_progress(
        self,
        *,
        pct: float | None = None,
        current: int | None = None,
        total: int | None = None,
        filename: str = "",
        meta_text: str = "",
    ) -> None:
        if pct is not None:
            try:
                self._progress_value = max(0.0, min(100.0, float(pct)))
            except Exception:
                pass
        self._render_progress_bar()

        resolved_meta = str(meta_text or "").strip()
        if not resolved_meta:
            parts: list[str] = []
            if pct is not None:
                parts.append(f"{int(round(self._progress_value))}%")
            if current is not None and total is not None:
                try:
                    parts.append(f"{int(current)}/{int(total)} obrazów")
                except Exception:
                    pass
            resolved_meta = " | ".join([part for part in parts if str(part).strip()])
        try:
            self._meta_lbl.configure(text=resolved_meta or "")
        except Exception:
            pass

        file_text = ""
        safe_name = str(filename or "").strip()
        if safe_name:
            try:
                safe_name = Path(safe_name).name or safe_name
            except Exception:
                pass
            file_text = f"Aktualnie: {safe_name}"
        try:
            self._file_lbl.configure(text=file_text)
        except Exception:
            pass

    def _render_progress_bar(self) -> None:
        try:
            width = int(self._progress_canvas.winfo_width() or self._progress_canvas.winfo_reqwidth() or 520)
        except Exception:
            width = 520
        width = max(1, int(width))
        height = 10
        fill_width = max(0, min(width, int((float(self._progress_value) / 100.0) * float(width))))
        try:
            self._progress_canvas.coords(self._progress_trough_id, 0, 0, width, height)
            self._progress_canvas.coords(self._progress_fill_id, 0, 0, fill_width, height)
        except Exception:
            pass
