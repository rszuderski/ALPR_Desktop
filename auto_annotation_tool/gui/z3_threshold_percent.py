"""Percentage presentation of PZ2 thresholds; runtime values keep their units."""

import math
import tkinter as tk
from tkinter import ttk


_PERCENT_REFERENCES = {
    "yolo_box_conf_var": 1.0,
    "yolo_symbol_conf_var": 1.0,
    "yolo_iou_var": 1.0,
    "yolo_overlap_var": 1.0,
    "yolo_seq_center_y_var": 1.0,
    "yolo_seq_min_h_ratio_var": 1.0,
    "yolo_seq_max_h_ratio_var": 1.0,
    "yolo_seq_max_w_ratio_var": 1.0,
    "yolo_seq_soft_overlap_var": 1.0,
    "yolo_seq_hard_overlap_var": 1.0,
    "ocr_conf_var": 1.0,
    "ocr_min_height_ratio_var": 1.0,
    "prep_clip_var": 255.0,
    "prep_c_var": 255.0,
}


def percent_reference(host, variable):
    for name, reference in _PERCENT_REFERENCES.items():
        if variable is getattr(host, name, None):
            return reference
    return None


def percent_label(text):
    if "[%]" in text:
        return text
    return text.rstrip(":") + " [%]:"


def format_percent(value, reference=1.0, *, suffix=False):
    text = f"{float(value) * 100.0 / reference:.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text + ("%" if suffix else "")


class _PercentBinding:
    def __init__(self, widget, variable, reference, lower, upper, command):
        self.widget = widget
        self.variable = variable
        self.reference = reference
        self.lower = lower
        self.upper = upper
        self.command = command
        self.display = tk.StringVar(master=widget)
        self._writing_display = False
        self._writing_source = False
        self._closed = False
        self._display_snapshot = None
        self._source_snapshot = None
        self.sync_from_source()
        self._source_trace = variable.trace_add("write", self.sync_from_source)
        self._display_trace = self.display.trace_add("write", self.on_display_write)
        widget.configure(textvariable=self.display, command=self.commit_and_save)
        widget.bind("<FocusOut>", self.commit_and_save, add="+")
        widget.bind("<Return>", self.commit_and_save, add="+")
        widget.bind("<Destroy>", self.close, add="+")

    def sync_from_source(self, *_args):
        if self._closed or self._writing_source:
            return
        value = float(self.variable.get())
        self._writing_display = True
        try:
            self._source_snapshot = value
            self._display_snapshot = format_percent(value, self.reference)
            self.display.set(self._display_snapshot)
        finally:
            self._writing_display = False

    def _input_value(self):
        value = float(self.display.get().strip().removesuffix("%").strip().replace(",", "."))
        if not math.isfinite(value):
            raise ValueError("Non-finite percentage")
        return value * self.reference / 100.0

    def _set_source(self, value):
        if isinstance(self.variable, tk.IntVar):
            value = round(value)
        if value != self.variable.get():
            self._writing_source = True
            try:
                self.variable.set(value)
            finally:
                self._writing_source = False

    def on_display_write(self, *_args):
        if self._closed or self._writing_display:
            return
        try:
            value = self._input_value()
        except (ValueError, tk.TclError):
            return
        if self.lower <= value <= self.upper:
            # Other visible editors and existing save traces use the source
            # variable immediately. Incomplete input never reaches detection.
            self._set_source(value)

    def commit_and_save(self, *_args):
        if self._closed:
            return
        try:
            # Merely focusing an existing rounded display must not round its
            # more precise stored threshold or change detector behavior.
            unchanged = (self.display.get() == self._display_snapshot
                         and float(self.variable.get()) == self._source_snapshot)
            if not unchanged:
                value = max(self.lower, min(self.upper, self._input_value()))
                self._set_source(value)
        except (ValueError, tk.TclError):
            pass
        self.sync_from_source()
        if self.command is not None:
            self.command()

    def close(self, event=None):
        if self._closed or (event is not None and event.widget is not self.widget):
            return
        self._closed = True
        for variable, trace in ((self.variable, self._source_trace), (self.display, self._display_trace)):
            try:
                variable.trace_remove("write", trace)
            except tk.TclError:
                pass


def create_percent_spinbox(parent, *, variable, from_, to, increment, reference=1.0,
                           width=9, command=None):
    factor = 100.0 / reference
    widget = ttk.Spinbox(parent, from_=from_ * factor, to=to * factor,
                         increment=increment * factor, width=width, format="%.3f")
    widget._percent_binding = _PercentBinding(widget, variable, reference, from_, to, command)
    return widget
