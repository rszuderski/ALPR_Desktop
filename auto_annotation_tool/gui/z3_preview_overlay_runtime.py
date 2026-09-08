#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 preview overlay, dock and action-clear helpers."""

from __future__ import annotations

from .z3_preview_dock_ui import refresh_preview_overlay_dock_tool_row
from ..config import logger as _COMPASS_LOG


def _get_preview_canvas_size(self):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return 0.0, 0.0
    try:
        return float(max(0, int(canvas.winfo_width() or 0))), float(max(0, int(canvas.winfo_height() or 0)))
    except Exception:
        return 0.0, 0.0

def _get_preview_mode_overlay_size(self):
    overlay = getattr(self, "preview_mode_overlay", None)
    if overlay is None:
        return 0.0, 0.0
    try:
        overlay.update_idletasks()
    except Exception:
        pass

    try:
        width = float(max(0, int(overlay.winfo_reqwidth() or overlay.winfo_width() or 0)))
        height = float(max(0, int(overlay.winfo_reqheight() or overlay.winfo_height() or 0)))
        return width, height
    except Exception:
        return 0.0, 0.0

def _clamp_preview_mode_overlay_position(self, x: float, y: float):
    canvas_w, canvas_h = self._get_preview_canvas_size()
    overlay_w, overlay_h = self._get_preview_mode_overlay_size()
    margin = 12.0

    if canvas_w <= 0 or canvas_h <= 0:
        return float(x), float(y)

    max_x = max(margin, canvas_w - overlay_w - margin)
    max_y = max(margin, canvas_h - overlay_h - margin)
    x = max(margin, min(max_x, float(x)))
    y = max(margin, min(max_y, float(y)))
    return float(x), float(y)

def _ensure_preview_mode_overlay_position(self):
    overlay = getattr(self, "preview_mode_overlay", None)
    if overlay is None:
        return
    try:
        overlay.place_forget()
    except Exception:
        pass

def _on_preview_mode_eye_press(self, event=None):
    if event is None:
        return "break"
    self._preview_mode_eye_drag_state = {
        "start_x": float(event.x_root),
        "start_y": float(event.y_root),
        "origin_x": float((getattr(self, "_preview_mode_overlay_position", {}) or {}).get("x", 12.0)),
        "origin_y": float((getattr(self, "_preview_mode_overlay_position", {}) or {}).get("y", 12.0)),
        "dragged": False,
    }
    return "break"

def _on_preview_mode_eye_drag(self, event=None):
    drag_state = getattr(self, "_preview_mode_eye_drag_state", None)
    if not isinstance(drag_state, dict) or event is None:
        return "break"

    delta_x = float(event.x_root) - float(drag_state.get("start_x", 0.0))
    delta_y = float(event.y_root) - float(drag_state.get("start_y", 0.0))
    if abs(delta_x) >= 3.0 or abs(delta_y) >= 3.0:
        drag_state["dragged"] = True

    new_x, new_y = self._clamp_preview_mode_overlay_position(
        float(drag_state.get("origin_x", 12.0)) + delta_x,
        float(drag_state.get("origin_y", 12.0)) + delta_y,
    )
    self._preview_mode_overlay_position = {"x": float(new_x), "y": float(new_y)}
    self._ensure_preview_mode_overlay_position()
    return "break"

def _on_preview_mode_eye_release(self, event=None):
    drag_state = getattr(self, "_preview_mode_eye_drag_state", None)
    self._preview_mode_eye_drag_state = None
    if isinstance(drag_state, dict) and bool(drag_state.get("dragged")):
        self._focus_preview_canvas()
        return "break"
    return self._toggle_preview_mode_overlay()

def _schedule_preview_overlay_toggle_job(self, attr_name: str, delay_ms: int, callback) -> None:
    scheduler = getattr(self, "preview_canvas_host", None) or getattr(self, "frame", None)
    previous_after_id = getattr(self, attr_name, None)
    if previous_after_id and scheduler is not None:
        try:
            scheduler.after_cancel(previous_after_id)
        except Exception:
            pass
    setattr(self, attr_name, None)

    def _run_toggle_job():
        setattr(self, attr_name, None)
        try:
            callback()
        except Exception:
            pass

    if scheduler is None:
        _run_toggle_job()
        return
    try:
        setattr(self, attr_name, scheduler.after(max(1, int(delay_ms)), _run_toggle_job))
    except Exception:
        setattr(self, attr_name, None)
        _run_toggle_job()

def _toggle_preview_overlay_dock(self, event=None):
    slide = getattr(self, "_preview_drawer_slide", None)
    if slide is not None:
        slide.toggle()
    self._place_preview_hint_overlay(refresh=True)
    self._focus_preview_canvas()
    return "break"

def _toggle_preview_overlay_dock_tool(self, tool_key: str):
    key = str(tool_key or "").strip().lower()
    if key == "legend":
        before = bool(getattr(self, "_preview_controls_legend_visible", True))
        self._preview_controls_legend_visible = not bool(getattr(self, "_preview_controls_legend_visible", True))
        try:
            _COMPASS_LOG.info(
                "[Z3 COMPASS] dock legend visible %s->%s",
                int(before),
                int(bool(getattr(self, "_preview_controls_legend_visible", True))),
            )
        except Exception:
            pass
        if not bool(getattr(self, "_preview_controls_legend_visible", True)):
            try:
                pending_after_id = getattr(self, "_preview_legend_toggle_after_id", None)
                if pending_after_id:
                    (getattr(self, "preview_canvas_host", None) or self.frame).after_cancel(pending_after_id)
                self._preview_legend_toggle_after_id = None
            except Exception:
                pass
            self._place_preview_hint_overlay()
        else:
            try:
                self._place_preview_hint_overlay(refresh=True)
            except Exception:
                pass
    elif key == "assistant":
        self._preview_operation_assistant_visible = not bool(
            getattr(self, "_preview_operation_assistant_visible", True)
        )
        if not bool(getattr(self, "_preview_operation_assistant_visible", True)):
            try:
                pending_after_id = getattr(self, "_preview_assistant_toggle_after_id", None)
                if pending_after_id:
                    (getattr(self, "preview_canvas_host", None) or self.frame).after_cancel(pending_after_id)
                self._preview_assistant_toggle_after_id = None
            except Exception:
                pass
            self._preview_typing_overlay_text = ""
            overlay = getattr(self, "preview_typing_overlay", None)
            if overlay is not None:
                try:
                    overlay.place_forget()
                except Exception:
                    pass
        else:
            try:
                self._refresh_preview_typing_overlay_visibility()
            except Exception:
                pass
    try:
        refresh_preview_overlay_dock_tool_row(self, key)
    except Exception:
        self._preview_overlay_dock_render_key = None
        try:
            self._place_preview_overlay_dock(force_render=False)
        except Exception:
            pass
    self._focus_preview_canvas()
    return "break"

def _extract_preview_badge_key_from_current_item(self):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None

    try:
        tags = canvas.gettags("current")
    except Exception:
        return None

    for tag in tags:
        if str(tag).startswith("preview_badge::"):
            return str(tag).split("preview_badge::", 1)[1]
    return None

def _extract_preview_action_from_current_item(self):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None

    try:
        tags = canvas.gettags("current")
    except Exception:
        return None

    for tag in tags:
        if str(tag).startswith("preview_action::"):
            return str(tag).split("preview_action::", 1)[1]
    return None

def _clear_preview_canvas_action(self, event=None):
    previous_selected_index = getattr(self, "_preview_char_selected_index", None)
    cleared_typing = self._cancel_preview_char_label_interaction(
        reset_mode=True,
        clear_hover=True,
        redraw_canvas=False,
    )
    if cleared_typing:
        self._preview_char_selected_index = None
        self._preview_pan_drag_state = None
        self._preview_badge_drag_state = None
        self._preview_char_drag_state = None
        self._preview_char_add_state = None
        self._refresh_preview_editor_toolbar()
        self._update_preview_edit_status("Zakończono tryb wpisywania znaków.", tone="muted")
        try:
            self._clear_preview_char_label_canvas_fields()
        except Exception:
            pass
        affected_indices = set()
        try:
            if previous_selected_index is not None:
                affected_indices.add(int(previous_selected_index))
        except Exception:
            affected_indices = set()
        if affected_indices and not self._refresh_preview_character_selection_visual(affected_indices):
            if not self._redraw_preview_character_overlays_light():
                self._on_preview_select(None)
        return "break"

    if (
        getattr(self, "_preview_char_add_state", None) is not None
        or getattr(self, "_preview_char_add_click_armed", False)
        or getattr(self, "_preview_char_add_modifier_down", False)
        or getattr(self, "_preview_char_add_mode", False)
    ):
        self._preview_char_add_state = None
        self._preview_char_add_click_armed = False
        self._preview_char_add_modifier_down = False
        self._preview_char_add_mode = False
        self._preview_pan_drag_state = None
        self._preview_badge_drag_state = None
        self._refresh_preview_editor_toolbar()
        self._apply_preview_canvas_cursor()
        self._update_preview_edit_status("Wyczyszczono rysowanie nowego boxa.", tone="muted")
        if not self._refresh_preview_character_selection_visual(None):
            if not self._redraw_preview_character_overlays_light():
                self._on_preview_select(None)
        return "break"

    self._update_preview_edit_status("Brak aktywnej akcji do wyczyszczenia.", tone="muted")
    return "break"
