#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers preventing modal mouse-wheel events from scrolling background panes."""

from __future__ import annotations


def _event_pointer_root(event, owner_widget) -> tuple[int | None, int | None]:
    try:
        x_root = int(getattr(event, "x_root", 0) or owner_widget.winfo_pointerx())
        y_root = int(getattr(event, "y_root", 0) or owner_widget.winfo_pointery())
        return x_root, y_root
    except Exception:
        return None, None


def event_is_over_foreign_toplevel(owner_widget, event) -> bool:
    """Return True when a global wheel handler sees an event from another Toplevel.

    Tkinter ``bind_all`` handlers receive mouse-wheel events even when the pointer
    is above a modal dialog. Main panes should not react to those events.
    """

    if owner_widget is None:
        return False

    try:
        owner_top = owner_widget.winfo_toplevel()
    except Exception:
        owner_top = None
    if owner_top is None:
        return False

    pointer_widget = None
    x_root, y_root = _event_pointer_root(event, owner_widget)
    if x_root is not None and y_root is not None:
        try:
            pointer_widget = owner_top.winfo_containing(x_root, y_root)
        except Exception:
            pointer_widget = None

    candidates = [pointer_widget, getattr(event, "widget", None)]
    for widget in candidates:
        if widget is None:
            continue
        try:
            target_top = widget.winfo_toplevel()
        except Exception:
            target_top = None
        if target_top is None:
            continue
        try:
            if str(target_top) != str(owner_top):
                return True
        except Exception:
            try:
                if target_top is not owner_top:
                    return True
            except Exception:
                pass
    return False

