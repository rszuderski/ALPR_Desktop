#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekki scrollbar canvasowy o nowoczesnym, webowym wygladzie.
"""

from __future__ import annotations

import tkinter as tk

from .app_theme_definitions import THEME_PALETTE_CONTRACT, get_theme_palette


def blend_hex_colors(color_a: str, color_b: str, ratio: float) -> str:
    def _to_rgb(raw: str) -> tuple[int, int, int]:
        value = str(raw or "").strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6:
            return (0, 0, 0)
        try:
            return tuple(int(value[idx:idx + 2], 16) for idx in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

    ratio = max(0.0, min(1.0, float(ratio)))
    rgb_a = _to_rgb(color_a)
    rgb_b = _to_rgb(color_b)
    blended = tuple(
        int(round((rgb_a[i] * (1.0 - ratio)) + (rgb_b[i] * ratio)))
        for i in range(3)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


def _default_scrollbar_palette() -> dict:
    try:
        return get_theme_palette()
    except Exception:
        return dict(THEME_PALETTE_CONTRACT)


class WebSlimScrollbar(tk.Canvas):
    def __init__(
        self,
        master,
        *,
        command=None,
        orient=tk.VERTICAL,
        thickness: int = 10,
        min_thumb_size: int = 13,
        auto_hide: bool = True,
        thumb_scale: float = 0.5,
        track_color: str | None = None,
        thumb_color: str | None = None,
        thumb_hover_color: str | None = None,
        **kwargs,
    ):
        default_palette = _default_scrollbar_palette()
        track_color = str(track_color or default_palette["scrollbar_track"])
        thumb_color = str(thumb_color or default_palette["scrollbar_thumb"])
        thumb_hover_color = str(thumb_hover_color or default_palette["scrollbar_thumb_hover"])
        self._orient = tk.VERTICAL if str(orient).lower().startswith("v") else tk.HORIZONTAL
        thickness = max(6, int(thickness))

        if self._orient == tk.VERTICAL:
            kwargs.setdefault("width", thickness)
        else:
            kwargs.setdefault("height", thickness)

        super().__init__(
            master,
            bg=track_color,
            bd=0,
            highlightthickness=0,
            takefocus=0,
            cursor="hand2",
            **kwargs,
        )
        self._command = command
        self._min_thumb_size = max(10, int(min_thumb_size))
        self._auto_hide = bool(auto_hide)
        self._thumb_scale = max(0.2, min(1.0, float(thumb_scale)))
        self._track_color = str(track_color)
        self._thumb_color = str(thumb_color)
        self._thumb_hover_color = str(thumb_hover_color)
        self._first = 0.0
        self._last = 1.0
        self._hover = False
        self._drag_offset = None
        self._thumb_id = self.create_rectangle(0, 0, 0, 0, outline="", width=0)
        self.bind("<Configure>", lambda _e: self._redraw(), add="+")
        self.bind("<Enter>", lambda _e: self._set_hover(True), add="+")
        self.bind("<Leave>", lambda _e: self._set_hover(False), add="+")
        self.bind("<Button-1>", self._on_press, add="+")
        self.bind("<B1-Motion>", self._on_drag, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self._redraw()

    def configure(self, cnf=None, **kwargs):
        if cnf is None and not kwargs:
            return super().configure()
        if cnf is not None and not isinstance(cnf, dict):
            return super().configure(cnf, **kwargs)

        merged = {}
        if isinstance(cnf, dict):
            merged.update(cnf)
        merged.update(kwargs)

        command = merged.pop("command", None)
        if command is not None:
            self._command = command
        auto_hide = merged.pop("auto_hide", None)
        if auto_hide is not None:
            self._auto_hide = bool(auto_hide)
        thumb_scale = merged.pop("thumb_scale", None)
        if thumb_scale is not None:
            self._thumb_scale = max(0.2, min(1.0, float(thumb_scale)))
        min_thumb_size = merged.pop("min_thumb_size", None)
        if min_thumb_size is not None:
            self._min_thumb_size = max(10, int(min_thumb_size))

        result = super().configure(**merged) if merged else None
        self._redraw()
        return result

    config = configure

    def configure_style(
        self,
        *,
        track_color: str | None = None,
        thumb_color: str | None = None,
        thumb_hover_color: str | None = None,
    ):
        if track_color is not None:
            self._track_color = str(track_color)
            try:
                super().configure(bg=self._track_color)
            except Exception:
                pass
        if thumb_color is not None:
            self._thumb_color = str(thumb_color)
        if thumb_hover_color is not None:
            self._thumb_hover_color = str(thumb_hover_color)
        self._redraw()

    def set(self, first, last):
        try:
            self._first = max(0.0, min(1.0, float(first)))
            self._last = max(self._first, min(1.0, float(last)))
        except Exception:
            self._first, self._last = 0.0, 1.0
        self._redraw()

    def _set_hover(self, enabled: bool):
        self._hover = bool(enabled)
        self._redraw()

    def _thumb_metrics(self):
        length = max(1.0, float(self.winfo_height() if self._orient == tk.VERTICAL else self.winfo_width()))
        thickness = max(1.0, float(self.winfo_width() if self._orient == tk.VERTICAL else self.winfo_height()))
        visible_fraction = max(0.0, min(1.0, self._last - self._first))
        if visible_fraction >= 0.999:
            if self._auto_hide:
                return None
            thumb_size = length
            max_start = 0.0
            start = 0.0
        else:
            natural_size = visible_fraction * length * self._thumb_scale
            thumb_size = min(length, max(float(self._min_thumb_size), natural_size))
            max_start = max(0.0, length - thumb_size)
            scrollable_fraction = max(1e-6, 1.0 - visible_fraction)
            start = 0.0 if max_start <= 0 else (self._first / scrollable_fraction) * max_start
            start = max(0.0, min(max_start, start))
        return {
            "length": length,
            "thickness": thickness,
            "thumb_size": thumb_size,
            "max_start": max_start,
            "start": start,
            "end": start + thumb_size,
            "visible_fraction": visible_fraction,
        }

    def _redraw(self):
        try:
            super().configure(bg=self._track_color)
        except Exception:
            pass

        metrics = self._thumb_metrics()
        if metrics is None:
            self.coords(self._thumb_id, 0, 0, 0, 0)
            self.itemconfigure(self._thumb_id, state="hidden")
            return

        pad = 1.5
        if self._orient == tk.VERTICAL:
            coords = (pad, metrics["start"], metrics["thickness"] - pad, metrics["end"])
        else:
            coords = (metrics["start"], pad, metrics["end"], metrics["thickness"] - pad)

        self.coords(self._thumb_id, *coords)
        self.itemconfigure(
            self._thumb_id,
            state="normal",
            fill=self._thumb_hover_color if self._hover else self._thumb_color,
            outline="",
        )

    def _moveto_from_start(self, start: float):
        metrics = self._thumb_metrics()
        if metrics is None or not callable(self._command):
            return

        max_start = max(0.0, float(metrics["max_start"]))
        if max_start <= 0:
            fraction = 0.0
        else:
            start_ratio = max(0.0, min(1.0, float(start) / max_start))
            fraction = start_ratio * max(0.0, 1.0 - float(metrics["visible_fraction"]))

        try:
            self._command("moveto", fraction)
        except Exception:
            pass

    def _event_axis(self, event) -> float:
        if self._orient == tk.VERTICAL:
            return float(getattr(event, "y", 0.0) or 0.0)
        return float(getattr(event, "x", 0.0) or 0.0)

    def _on_press(self, event):
        metrics = self._thumb_metrics()
        if metrics is None:
            return "break"

        axis_pos = self._event_axis(event)
        if metrics["start"] <= axis_pos <= metrics["end"]:
            self._drag_offset = axis_pos - metrics["start"]
        else:
            self._drag_offset = metrics["thumb_size"] / 2.0
            self._moveto_from_start(axis_pos - self._drag_offset)
        return "break"

    def _on_drag(self, event):
        if self._drag_offset is None:
            return "break"
        self._moveto_from_start(self._event_axis(event) - float(self._drag_offset))
        return "break"

    def _on_release(self, _event):
        self._drag_offset = None
        return "break"
