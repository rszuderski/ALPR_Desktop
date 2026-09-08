#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight canvas progress bar used by Z4 training screens."""

from __future__ import annotations

import tkinter as tk


class TrainProgressBar(tk.Canvas):
    def __init__(
        self,
        master,
        *,
        variable=None,
        maximum: float = 100.0,
        value: float = 0.0,
        mode: str = "determinate",
        thickness: int = 6,
        trough_color: str = "#3c3c3c",
        fill_color: str = "#4ec9b0",
        segment_ratio: float = 0.28,
        **kwargs,
    ):
        canvas_height = max(int(kwargs.pop("height", thickness + 6)), int(thickness) + 6)
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
        self._mode = str(mode or "determinate").strip().lower() or "determinate"
        self._thickness = max(1, int(thickness))
        self._trough_color = str(trough_color)
        self._fill_color = str(fill_color)
        self._segment_ratio = max(0.12, min(0.50, float(segment_ratio)))
        self._variable = None
        self._variable_trace_id = None
        self._indeterminate_after_id = None
        self._indeterminate_running = False
        self._indeterminate_interval_ms = 40
        self._indeterminate_phase = 0.0

        self._trough_id = self.create_line(0, 0, 0, 0, capstyle=tk.ROUND)
        self._fill_id = self.create_line(0, 0, 0, 0, capstyle=tk.ROUND)
        self.bind("<Configure>", lambda _event: self._redraw(), add="+")

        self._set_variable(variable)
        self.configure(value=value)

    def _set_variable(self, variable):
        if self._variable is variable:
            return

        if self._variable is not None and self._variable_trace_id is not None:
            try:
                self._variable.trace_remove("write", self._variable_trace_id)
            except Exception:
                pass

        self._variable = variable
        self._variable_trace_id = None

        if variable is None:
            return

        try:
            self._variable_trace_id = variable.trace_add("write", lambda *_args: self._sync_variable_value())
        except Exception:
            self._variable_trace_id = None

        self._sync_variable_value()

    def _sync_variable_value(self):
        variable = getattr(self, "_variable", None)
        if variable is None:
            return

        try:
            self._value = max(0.0, float(variable.get()))
        except Exception:
            return

        self._redraw()

    def _redraw(self):
        width = max(1.0, float(self.winfo_width()))
        height = max(1.0, float(self.winfo_height()))
        center_y = height / 2.0
        half_thickness = max(0.5, float(self._thickness) / 2.0)
        left = half_thickness + 1.0
        right = max(left, width - half_thickness - 1.0)

        self.coords(self._trough_id, left, center_y, right, center_y)
        self.itemconfigure(self._trough_id, fill=self._trough_color, width=self._thickness)

        if self._mode == "indeterminate":
            if self._indeterminate_running:
                span = max(12.0, (right - left) * self._segment_ratio)
                travel = max(1.0, (right - left) + span)
                start = left - span + (travel * self._indeterminate_phase)
                end = start + span
                visible_start = max(left, start)
                visible_end = min(right, end)
                if visible_end <= visible_start:
                    visible_start = left
                    visible_end = left
            else:
                visible_start = left
                visible_end = left
        else:
            ratio = max(0.0, min(1.0, float(self._value) / max(1.0, float(self._maximum))))
            visible_start = left
            visible_end = left + ((right - left) * ratio)

        self.coords(self._fill_id, visible_start, center_y, max(visible_start, visible_end), center_y)
        self.itemconfigure(self._fill_id, fill=self._fill_color, width=self._thickness)

    def _tick_indeterminate(self):
        self._indeterminate_after_id = None
        if not self._indeterminate_running:
            return

        self._indeterminate_phase = (float(self._indeterminate_phase) + 0.04) % 1.0
        self._redraw()

        try:
            self._indeterminate_after_id = self.after(
                max(16, int(self._indeterminate_interval_ms)),
                self._tick_indeterminate,
            )
        except Exception:
            self._indeterminate_after_id = None

    def start(self, interval: int | None = None):
        if interval is not None:
            try:
                self._indeterminate_interval_ms = max(16, int(interval))
            except Exception:
                pass

        self._indeterminate_running = True
        if self._indeterminate_after_id is None:
            self._tick_indeterminate()

    def stop(self):
        self._indeterminate_running = False
        pending = self._indeterminate_after_id
        self._indeterminate_after_id = None
        if pending is not None:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        self._redraw()

    def configure(self, cnf=None, **kwargs):
        if cnf is not None and not isinstance(cnf, dict):
            return super().configure(cnf, **kwargs)

        merged = {}
        if isinstance(cnf, dict):
            merged.update(cnf)
        merged.update(kwargs)

        if "variable" in merged:
            self._set_variable(merged.pop("variable"))
        if "maximum" in merged:
            self._maximum = max(1.0, float(merged.pop("maximum")))
        if "value" in merged:
            self._value = max(0.0, float(merged.pop("value")))
        if "mode" in merged:
            self._mode = str(merged.pop("mode") or "determinate").strip().lower() or "determinate"
            if self._mode != "indeterminate":
                self.stop()
        if "thickness" in merged:
            self._thickness = max(1, int(merged.pop("thickness")))
            merged.setdefault("height", self._thickness + 6)
        if "trough_color" in merged:
            self._trough_color = str(merged.pop("trough_color"))
        if "fill_color" in merged:
            self._fill_color = str(merged.pop("fill_color"))

        background = merged.pop("background", None)
        bg = merged.pop("bg", None)
        resolved_bg = background if background is not None else bg
        if resolved_bg is not None:
            super().configure(bg=resolved_bg)

        result = super().configure(**merged)
        self._redraw()
        return result

    config = configure

    def destroy(self):
        self.stop()
        if self._variable is not None and self._variable_trace_id is not None:
            try:
                self._variable.trace_remove("write", self._variable_trace_id)
            except Exception:
                pass
        self._variable = None
        self._variable_trace_id = None
        super().destroy()
