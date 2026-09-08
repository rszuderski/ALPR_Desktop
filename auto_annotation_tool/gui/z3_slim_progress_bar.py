#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slim canvas progress bar used by Z3 character annotation screens."""

from __future__ import annotations

import tkinter as tk


class SlimProgressBar(tk.Canvas):
    def __init__(
        self,
        master,
        *,
        maximum: float = 100.0,
        value: float = 0.0,
        thickness: int = 2,
        trough_color: str = "#3c3c3c",
        fill_color: str = "#0e639c",
        **kwargs,
    ):
        canvas_height = max(int(kwargs.pop("height", thickness + 4)), int(thickness) + 4)
        bg = kwargs.pop("bg", kwargs.pop("background", trough_color))
        super().__init__(
            master,
            height=canvas_height,
            bg=bg,
            bd=0,
            highlightthickness=0,
            **kwargs,
            )
        self._maximum = max(1.0, float(maximum))
        self._value = 0.0
        self._thickness = max(1, int(thickness))
        self._trough_color = str(trough_color)
        self._fill_color = str(fill_color)
        self._trough_id = self.create_line(0, 0, 0, 0, capstyle=tk.ROUND)
        self._fill_id = self.create_line(0, 0, 0, 0, capstyle=tk.ROUND)
        self.bind("<Configure>", lambda _event: self._redraw(), add="+")
        self.configure(value=value)

    def _redraw(self):
        width = max(1.0, float(self.winfo_width()))
        height = max(1.0, float(self.winfo_height()))
        center_y = height / 2.0
        half_thickness = max(0.5, float(self._thickness) / 2.0)
        left = half_thickness + 1.0
        right = max(left, width - half_thickness - 1.0)
        ratio = max(0.0, min(1.0, float(self._value) / max(1.0, float(self._maximum))))
        fill_right = left + ((right - left) * ratio)

        self.coords(self._trough_id, left, center_y, right, center_y)
        self.coords(self._fill_id, left, center_y, max(left, fill_right), center_y)
        self.itemconfigure(self._trough_id, fill=self._trough_color, width=self._thickness)
        self.itemconfigure(self._fill_id, fill=self._fill_color, width=self._thickness)

    def configure(self, cnf=None, **kwargs):
        if cnf is not None and not isinstance(cnf, dict):
            return super().configure(cnf, **kwargs)

        merged = {}
        if isinstance(cnf, dict):
            merged.update(cnf)
        merged.update(kwargs)

        if "maximum" in merged:
            self._maximum = max(1.0, float(merged.pop("maximum")))
        if "value" in merged:
            self._value = max(0.0, float(merged.pop("value")))
        if "thickness" in merged:
            self._thickness = max(1, int(merged.pop("thickness")))
            merged.setdefault("height", self._thickness + 4)
        if "trough_color" in merged:
            self._trough_color = str(merged.pop("trough_color"))
        if "fill_color" in merged:
            self._fill_color = str(merged.pop("fill_color"))

        background = merged.pop("background", None)
        bg = merged.pop("bg", None)
        resolved_bg = background if background is not None else bg
        if resolved_bg is not None:
            super().configure(bg=resolved_bg)

        result = super().configure(**merged) if merged else None
        self._redraw()
        return result

    config = configure
