#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small tooltip helpers used by the main application shell."""

import tkinter as tk


def bind_simple_tooltip(app, widget, text: str) -> None:
    if widget is None:
        return
    tooltip_text = str(text or "").strip()
    if not tooltip_text:
        return

    try:
        widget._simple_tooltip_text = tooltip_text
    except Exception:
        pass

    widget.bind(
        "<Enter>",
        lambda _event, target=widget, value=tooltip_text: schedule_simple_tooltip(app, target, value),
        add="+",
    )
    widget.bind("<Leave>", lambda _event: hide_simple_tooltip(app), add="+")
    widget.bind("<ButtonPress-1>", lambda _event: hide_simple_tooltip(app), add="+")
    widget.bind("<Destroy>", lambda _event: hide_simple_tooltip(app), add="+")


def schedule_simple_tooltip(app, widget, text: str) -> None:
    hide_simple_tooltip(app, cancel_pending=True)
    app._simple_tooltip_target = widget
    try:
        app._simple_tooltip_after_id = app.root.after(
            280,
            lambda target=widget, value=str(text or "").strip(): show_simple_tooltip(app, target, value),
        )
    except Exception:
        app._simple_tooltip_after_id = None


def show_simple_tooltip(app, widget, text: str) -> None:
    app._simple_tooltip_after_id = None
    text = str(text or "").strip()
    if not text or widget is None:
        return
    try:
        if not bool(widget.winfo_exists()):
            return
    except Exception:
        return

    hide_simple_tooltip(app, cancel_pending=False)

    palette = getattr(app, "palette", {})
    bg = palette.get("panel_alt", palette.get("panel", "#2d2d30"))
    border = palette.get("guide", palette.get("warning", "#f0b44c"))
    fg = palette.get("fg", "#f3f3f3")

    try:
        tooltip = tk.Toplevel(app.root)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        try:
            tooltip.attributes("-topmost", True)
        except Exception:
            pass
        tooltip.configure(bg=border)

        body = tk.Label(
            tooltip,
            text=text,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            bd=0,
            padx=9,
            pady=4,
            anchor="center",
            justify=tk.CENTER,
        )
        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        tooltip.update_idletasks()

        width = max(tooltip.winfo_reqwidth(), 1)
        height = max(tooltip.winfo_reqheight(), 1)
        widget_width = max(widget.winfo_width(), 1)
        screen_width = max(widget.winfo_screenwidth(), width)
        screen_height = max(widget.winfo_screenheight(), height)
        x = widget.winfo_rootx() + ((widget_width - width) // 2)
        y = widget.winfo_rooty() - height - 8
        x = max(4, min(x, screen_width - width - 4))
        if y < 4:
            y = min(widget.winfo_rooty() + max(widget.winfo_height(), 1) + 8, screen_height - height - 4)

        tooltip.geometry(f"+{int(x)}+{int(y)}")
        tooltip.deiconify()
        app._simple_tooltip_window = tooltip
    except Exception:
        app._simple_tooltip_window = None


def hide_simple_tooltip(app, cancel_pending: bool = True) -> None:
    if cancel_pending:
        pending = getattr(app, "_simple_tooltip_after_id", None)
        if pending is not None:
            try:
                app.root.after_cancel(pending)
            except Exception:
                pass
            app._simple_tooltip_after_id = None
    tooltip = getattr(app, "_simple_tooltip_window", None)
    if tooltip is not None:
        try:
            tooltip.destroy()
        except Exception:
            pass
    app._simple_tooltip_window = None
    app._simple_tooltip_target = None
