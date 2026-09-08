#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scroll and adaptive canvas helpers for Z3/PZ panels."""

from .modal_scroll_guard import event_is_over_foreign_toplevel


def sync_detect_right_scrollregion(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "detect_right_canvas") or host.detect_right_canvas is None:
        return

    canvas = host.detect_right_canvas
    pending = getattr(host, "_detect_right_scrollregion_after_id", None)
    if pending is not None:
        try:
            canvas.after_cancel(pending)
        except Exception:
            pass
        try:
            host._detect_right_scrollregion_after_id = None
        except Exception:
            pass

    def _apply_scrollregion() -> None:
        try:
            host._detect_right_scrollregion_after_id = None
        except Exception:
            pass
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
        except Exception:
            pass
        try:
            host._schedule_detect_right_adaptive_wrap_refresh()
        except Exception:
            pass
        try:
            schedule_detect_right_scroll_bind_refresh(host)
        except Exception:
            pass

    delay_ms = 85 if bool(getattr(host, "_pz2_panel_resize_active", False)) else 24
    try:
        host._detect_right_scrollregion_after_id = canvas.after(delay_ms, _apply_scrollregion)
    except Exception:
        _apply_scrollregion()


def sync_extract_left_scrollregion(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "extract_left_canvas") or host.extract_left_canvas is None:
        return

    try:
        host.extract_left_canvas.configure(scrollregion=host.extract_left_canvas.bbox("all"))
    except Exception:
        pass


def sync_extract_left_canvas_width(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "extract_left_canvas") or host.extract_left_canvas is None:
        return

    try:
        width = max(0, int(host.extract_left_canvas.winfo_width()))
        inset = max(0, int(getattr(host, "_extract_content_inset", 0)))
        max_width = max(320, int(getattr(host, "_extract_content_max_width", width)))
        # Przy leniwym ładowaniu Tk potrafi przez chwilę zwrócić szerokość 1 px.
        # Nie wolno na tej podstawie zwężać kart PZ1, bo użytkownik widzi
        # wtedy niedorysowane kafle do czasu kolejnego zdarzenia Configure.
        if width <= (2 * inset) + 120:
            target_width = max_width
        else:
            target_width = max(320, min(max_width, width - (2 * inset)))
        host.extract_left_canvas.coords(host.extract_left_content_window, inset, 0)
        host.extract_left_canvas.itemconfigure(host.extract_left_content_window, width=target_width)
    except Exception:
        pass


def sync_detect_right_canvas_width(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "detect_right_canvas") or host.detect_right_canvas is None:
        return

    canvas = host.detect_right_canvas
    try:
        width = max(50, int(canvas.winfo_width()))
    except Exception:
        width = 50

    def _apply_width() -> None:
        try:
            host._detect_right_width_after_id = None
        except Exception:
            pass
        try:
            canvas.itemconfigure(host.detect_right_content_window, width=width)
        except Exception:
            pass
        host._schedule_detect_right_adaptive_wrap_refresh()

    if bool(getattr(host, "_pz2_panel_resize_active", False)):
        pending = getattr(host, "_detect_right_width_after_id", None)
        if pending is not None:
            try:
                canvas.after_cancel(pending)
            except Exception:
                pass
        try:
            host._detect_right_width_after_id = canvas.after(85, _apply_width)
        except Exception:
            _apply_width()
        return

    try:
        pending = getattr(host, "_detect_right_width_after_id", None)
        if pending is not None:
            canvas.after_cancel(pending)
            host._detect_right_width_after_id = None
    except Exception:
        pass

    _apply_width()


def refresh_detect_right_adaptive_wraps(host: "CharacterAnnotationTab") -> None:
    try:
        host._detect_right_wrap_after_id = None
    except Exception:
        pass

    content = getattr(host, "detect_right_content", None)
    if content is None:
        return

    try:
        host.app._refresh_adaptive_wraps(content)
    except Exception:
        pass


def schedule_detect_right_adaptive_wrap_refresh(host: "CharacterAnnotationTab") -> None:
    content = getattr(host, "detect_right_content", None)
    if content is None:
        return

    pending = getattr(host, "_detect_right_wrap_after_id", None)
    if pending is not None:
        try:
            content.after_cancel(pending)
        except Exception:
            pass
        try:
            host._detect_right_wrap_after_id = None
        except Exception:
            pass

    try:
        host._detect_right_wrap_after_id = content.after(140, host._refresh_detect_right_adaptive_wraps)
    except Exception:
        try:
            host._detect_right_wrap_after_id = None
        except Exception:
            pass


def schedule_detect_right_scroll_bind_refresh(host: "CharacterAnnotationTab", delay_ms: int = 120) -> None:
    content = getattr(host, "detect_right_content", None)
    canvas = getattr(host, "detect_right_canvas", None)
    if content is None or canvas is None:
        return

    pending = getattr(host, "_detect_right_scroll_bind_after_id", None)
    if pending is not None:
        try:
            content.after_cancel(pending)
        except Exception:
            pass
        try:
            host._detect_right_scroll_bind_after_id = None
        except Exception:
            pass

    def _apply_bind_refresh() -> None:
        try:
            host._detect_right_scroll_bind_after_id = None
        except Exception:
            pass
        try:
            if not bool(content.winfo_exists()) or not bool(canvas.winfo_exists()):
                return
        except Exception:
            return
        try:
            host._bind_scroll_canvas_children(
                content,
                canvas,
                host._detect_right_canvas_overflows,
            )
        except Exception:
            pass
        pinned = getattr(host, "detect_right_pinned_status_host", None)
        if pinned is not None:
            try:
                host._bind_scroll_canvas_children(
                    pinned,
                    canvas,
                    host._detect_right_canvas_overflows,
                )
            except Exception:
                pass

    try:
        host._detect_right_scroll_bind_after_id = content.after(max(20, int(delay_ms)), _apply_bind_refresh)
    except Exception:
        _apply_bind_refresh()


def sync_cvat_export_scrollregion(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "cvat_export_canvas") or host.cvat_export_canvas is None:
        return

    try:
        host.cvat_export_canvas.configure(scrollregion=host.cvat_export_canvas.bbox("all"))
    except Exception:
        pass


def sync_cvat_export_canvas_width(host: "CharacterAnnotationTab", event=None) -> None:
    if not hasattr(host, "cvat_export_canvas") or host.cvat_export_canvas is None:
        return

    try:
        inset = max(0, int(getattr(host, "_pz3_content_inset", 0)))
        available_width = max(50, int(host.cvat_export_canvas.winfo_width()) - (2 * inset))
        width = max(320, available_width)
        host.cvat_export_canvas.coords(host.cvat_export_content_window, inset, 0)
        host.cvat_export_canvas.itemconfigure(host.cvat_export_content_window, width=width)
    except Exception:
        pass

    try:
        host.app._refresh_adaptive_wraps(host.cvat_export_content)
    except Exception:
        pass


def widget_contains_point(widget, x_root: int, y_root: int) -> bool:
    if widget is None:
        return False
    try:
        wx = int(widget.winfo_rootx())
        wy = int(widget.winfo_rooty())
        return wx <= x_root < (wx + int(widget.winfo_width())) and wy <= y_root < (wy + int(widget.winfo_height()))
    except Exception:
        return False


def mousewheel_units(host: "CharacterAnnotationTab", event) -> int:
    return host._inertial_scroll.mousewheel_units(event)


def on_plates_listbox_mousewheel(host: "CharacterAnnotationTab", event):
    listbox = getattr(host, "plates_listbox", None)
    if listbox is None:
        return None
    units = host._inertial_scroll.mousewheel_units(event)
    if units == 0 or not host._inertial_scroll.listbox_can_scroll(listbox, units):
        return None
    try:
        magnitude = host._inertial_scroll.mousewheel_magnitude(event)
    except Exception:
        magnitude = 1.0
    try:
        rows = max(1, min(6, int(round(3.0 * float(magnitude or 1.0)))))
    except Exception:
        rows = 3
    try:
        queued = int(getattr(host, "_plates_listbox_scroll_units", 0) or 0)
    except Exception:
        queued = 0
    queued += int(units) * int(rows)
    queued = max(-80, min(80, queued))
    host._plates_listbox_scroll_units = int(queued)

    def _drain_listbox_scroll() -> None:
        host._plates_listbox_scroll_after_id = None
        current_listbox = getattr(host, "plates_listbox", None)
        if current_listbox is None:
            host._plates_listbox_scroll_units = 0
            return
        try:
            remaining = int(getattr(host, "_plates_listbox_scroll_units", 0) or 0)
        except Exception:
            remaining = 0
        if remaining == 0:
            return
        step = 1 if remaining > 0 else -1
        try:
            if not host._inertial_scroll.listbox_can_scroll(current_listbox, step):
                host._plates_listbox_scroll_units = 0
                return
            current_listbox.yview_scroll(step, "units")
            host._plates_listbox_scroll_units = int(remaining - step)
        except Exception:
            host._plates_listbox_scroll_units = 0
            return
        try:
            if int(getattr(host, "_plates_listbox_scroll_units", 0) or 0) != 0:
                host._plates_listbox_scroll_after_id = current_listbox.after(7, _drain_listbox_scroll)
        except Exception:
            host._plates_listbox_scroll_after_id = None

    if getattr(host, "_plates_listbox_scroll_after_id", None) is None:
        _drain_listbox_scroll()
    return "break"


def canvas_overflows(canvas) -> bool:
    if canvas is None:
        return False

    try:
        bbox = canvas.bbox("all")
        if not bbox:
            return False
        content_height = int(bbox[3]) - int(bbox[1])
        viewport_height = int(canvas.winfo_height())
        return content_height > viewport_height + 1
    except Exception:
        return False


def on_detect_right_global_mousewheel(host: "CharacterAnnotationTab", event):
    if event_is_over_foreign_toplevel(getattr(host, "frame", None), event):
        return "break"

    try:
        if widget_contains_point(getattr(host, "preview_canvas", None), int(event.x_root), int(event.y_root)):
            return "break"
    except Exception:
        pass

    suppress_selection_hover_during_scroll(host)
    if host._inertial_scroll.scroll_canvas_if_targeted(
        getattr(host, "detect_right_canvas", None),
        event,
        pointer_widget=host.frame,
        overflow_checker=host._detect_right_canvas_overflows,
    ):
        return "break"
    return None


def on_cvat_export_global_mousewheel(host: "CharacterAnnotationTab", event):
    if event_is_over_foreign_toplevel(getattr(host, "frame", None), event):
        return "break"

    host._suppress_selection_hover_during_scroll()
    if host._inertial_scroll.scroll_canvas_if_targeted(
        getattr(host, "cvat_export_canvas", None),
        event,
        pointer_widget=host.frame,
        overflow_checker=host._cvat_export_canvas_overflows,
    ):
        return "break"
    return None


def suppress_selection_hover_during_scroll(host: "CharacterAnnotationTab", duration_ms: int = 320) -> None:
    host._selection_hover_suppressed = True
    after_id = getattr(host, "_selection_hover_suppress_after_id", None)
    if after_id:
        try:
            host.frame.after_cancel(after_id)
        except Exception:
            pass

    def _clear_hover_suppression():
        host._selection_hover_suppressed = False
        host._selection_hover_suppress_after_id = None

    try:
        host._selection_hover_suppress_after_id = host.frame.after(int(duration_ms), _clear_hover_suppression)
    except Exception:
        host._selection_hover_suppressed = False
        host._selection_hover_suppress_after_id = None


def restore_scroll_canvas_focus(canvas) -> None:
    if canvas is None:
        return
    try:
        if canvas.focus_get() is canvas:
            return
    except Exception:
        pass
    try:
        canvas.focus_set()
    except Exception:
        pass


def redirect_child_mousewheel_to_canvas(host: "CharacterAnnotationTab", event, canvas, overflow_checker=None):
    if canvas is getattr(host, "detect_right_canvas", None):
        suppress_selection_hover_during_scroll(host)

    if host._inertial_scroll.redirect_child_mousewheel_to_canvas(
        event,
        canvas,
        pointer_widget=host.frame,
        overflow_checker=overflow_checker,
    ):
        restore_scroll_canvas_focus(canvas)
        return "break"

    restore_scroll_canvas_focus(canvas)
    return "break"


def redirect_pz3_child_mousewheel_to_canvas(host: "CharacterAnnotationTab", event):
    canvas = getattr(host, "cvat_export_canvas", None)
    host._suppress_selection_hover_during_scroll()
    if host._inertial_scroll.redirect_child_mousewheel_to_canvas(
        event,
        canvas,
        pointer_widget=host.frame,
        overflow_checker=host._cvat_export_canvas_overflows,
    ):
        restore_scroll_canvas_focus(canvas)
        return "break"
    restore_scroll_canvas_focus(canvas)
    return "break"


def bind_scroll_canvas_children(host: "CharacterAnnotationTab", root, canvas, overflow_checker=None) -> None:
    if root is None or canvas is None:
        return

    release_focus_classes = {
        "TButton",
        "Button",
        "TCheckbutton",
        "Checkbutton",
        "TRadiobutton",
        "Radiobutton",
        "TScale",
        "Scale",
        "TCombobox",
        "Spinbox",
    }

    bind_marker = f"_z3_scroll_bound_{id(canvas)}"

    def _walk(widget):
        def _handle_mousewheel(event, c=canvas, oc=overflow_checker):
            if c is getattr(host, "cvat_export_canvas", None):
                return host._redirect_pz3_child_mousewheel_to_canvas(event)
            return host._redirect_child_mousewheel_to_canvas(event, c, oc)

        already_bound = False
        try:
            already_bound = bool(getattr(widget, bind_marker, False))
        except Exception:
            already_bound = False

        if not already_bound:
            try:
                widget.bind("<MouseWheel>", _handle_mousewheel, add="+")
                widget.bind("<Button-4>", _handle_mousewheel, add="+")
                widget.bind("<Button-5>", _handle_mousewheel, add="+")
                setattr(widget, bind_marker, True)
            except Exception:
                pass

        try:
            class_name = str(widget.winfo_class())
        except Exception:
            class_name = ""

        if class_name in release_focus_classes:
            try:
                widget.configure(takefocus=0)
            except Exception:
                pass
            try:
                widget.bind("<ButtonRelease-1>", lambda _e, c=canvas: restore_scroll_canvas_focus(c), add="+")
            except Exception:
                pass
            if class_name == "TCombobox":
                try:
                    widget.bind("<<ComboboxSelected>>", lambda _e, c=canvas: restore_scroll_canvas_focus(c), add="+")
                except Exception:
                    pass

        for child in widget.winfo_children():
            _walk(child)

    _walk(root)
