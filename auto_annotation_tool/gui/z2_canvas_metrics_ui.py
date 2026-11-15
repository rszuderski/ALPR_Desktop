#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small UI renderers for the Z2 canvas metrics overlay."""

import tkinter as tk


def render_preview_metrics_grab_handle(grab_widget, colors: dict) -> None:
    if grab_widget is None:
        return
    fill = str(colors.get("grab_fill", "#3a3f46"))
    outline = str(colors.get("outline", "#4b5563"))
    dot_fill = str(colors.get("muted", "#d7e1ec"))
    bg = str(colors.get("fill", "#101419"))
    try:
        grab_widget.configure(bg=bg)
    except Exception:
        pass
    if not hasattr(grab_widget, "create_rectangle"):
        try:
            grab_widget.configure(text="::::", fg=dot_fill, bg=bg)
        except Exception:
            pass
        return

    try:
        grab_widget.delete("all")
        handle_w = 18.0
        handle_h = 20.0
        x = 2.0
        y = 1.0
        grab_widget.create_rectangle(
            x,
            y,
            x + handle_w,
            y + handle_h,
            fill=fill,
            outline=outline,
            width=1,
        )
        dot_r = 1.2
        dot_x = x + (handle_w / 2.0)
        for row_y in (y + 6.0, y + 10.0, y + 14.0):
            for col_dx in (-3.0, 3.0):
                grab_widget.create_oval(
                    dot_x + col_dx - dot_r,
                    row_y - dot_r,
                    dot_x + col_dx + dot_r,
                    row_y + dot_r,
                    fill=dot_fill,
                    outline="",
                )
    except Exception:
        pass


def render_preview_metrics_table(
    body,
    rows: list[tuple[str, str, str]],
    colors: dict,
) -> None:
    if body is None:
        return
    widgets = getattr(body, "_z2_metric_rows", None)
    if widgets is None or not all(row.winfo_exists() for row, _key, _value in widgets):
        widgets = []
        for child in list(body.winfo_children()):
            child.destroy()
    while len(widgets) > len(rows):
        row, _key, _value = widgets.pop()
        row.destroy()

    fill = str(colors.get("fill", "#101419"))
    row_fill = str(colors.get("row_fill", fill))
    row_alt = str(colors.get("row_alt", row_fill))
    outline = str(colors.get("outline", "#4b5563"))
    text_fill = str(colors.get("text", "#f8fafc"))
    muted = str(colors.get("muted", "#c7c7c7"))
    success = str(colors.get("success", "#4ec9b0"))
    warning = str(colors.get("warning", "#f39c12"))
    error = str(colors.get("error", "#e74c3c"))

    for idx, (label_text, value_text, tone) in enumerate(rows):
        row_bg = row_fill if idx % 2 == 0 else row_alt
        tone_key = str(tone or "").strip().lower()
        if tone_key == "warning":
            value_fg = warning
        elif tone_key == "success":
            value_fg = success
        elif tone_key == "error":
            value_fg = error
        else:
            value_fg = text_fill
        if idx >= len(widgets):
            row = tk.Frame(body, bd=0, highlightthickness=1)
            row.pack(fill=tk.X, padx=6, pady=(0 if idx == 0 else 2, 2))
            key = tk.Label(row, anchor="w", justify=tk.LEFT, font=("Segoe UI", 8),
                           width=13, padx=6, pady=3)
            key.pack(side=tk.LEFT, fill=tk.Y)
            value = tk.Label(row, anchor="w", justify=tk.LEFT, font=("Segoe UI Semibold", 8),
                             padx=6, pady=3, wraplength=260)
            value.pack(side=tk.LEFT, fill=tk.X, expand=True)
            widgets.append((row, key, value))
        row, key, value = widgets[idx]
        signature = (str(label_text or ""), str(value_text or ""), row_bg, outline, muted, value_fg)
        if getattr(row, "_z2_metric_signature", None) != signature:
            row.configure(bg=row_bg, highlightbackground=outline)
            key.configure(text=signature[0], bg=row_bg, fg=muted)
            value.configure(text=signature[1], bg=row_bg, fg=value_fg)
            row._z2_metric_signature = signature
    body._z2_metric_rows = widgets
