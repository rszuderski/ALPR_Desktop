#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canvas event handlers for Z3/PZ2 preview."""

import tkinter as tk
import copy
import math
import time
from pathlib import Path

from ..config import logger
from .z3_preview_ui import (
    _get_cached_preview_photo,
    _get_cached_preview_source_image,
    _preview_char_record_trace_fields,
    _style_preview_character_edit_grips_fast,
)


def _mark_preview_char_edit_interaction(host) -> None:
    try:
        host._preview_last_char_edit_interaction_ts = time.monotonic()
    except Exception:
        pass


def _push_preview_char_drag_history_snapshot(host, drag_state: dict) -> None:
    if not isinstance(drag_state, dict) or bool(drag_state.get("history_pushed")):
        return

    chars = host._get_preview_active_character_records(create=False)
    try:
        char_idx = int(drag_state.get("index", -1))
    except Exception:
        char_idx = -1

    original_record = drag_state.get("original_record")
    if not (0 <= char_idx < len(chars)) or not isinstance(original_record, dict):
        host._push_preview_history_snapshot()
        drag_state["history_pushed"] = True
        return

    current_record = chars[char_idx]
    try:
        chars[char_idx] = copy.deepcopy(original_record)
        host._push_preview_history_snapshot()
    finally:
        chars[char_idx] = current_record
        drag_state["history_pushed"] = True


def _get_preview_char_row_for_record(host, rec, *, data=None) -> int | None:
    if not isinstance(rec, dict):
        return None
    try:
        row = int(rec.get("reading_row", 0) or 0)
        if row in (1, 2):
            return row
    except Exception:
        pass
    try:
        bbox = host._char_record_bbox(rec)
        row = int(host._get_preview_row_for_bbox(bbox, data=data) or 0)
        return row if row in (1, 2) else None
    except Exception:
        return None


def _get_preview_char_geometry_leader_index(host, *, row_hint: int | None = None):
    data = host._get_preview_active_data(create=False)
    chars = host._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not chars:
        return None, None

    target_row = row_hint if row_hint in (1, 2) else None
    if target_row is None:
        for idx_attr in ("_preview_char_hover_index", "_preview_char_selected_index"):
            try:
                hint_idx = int(getattr(host, idx_attr, None))
            except Exception:
                continue
            if 0 <= hint_idx < len(chars):
                target_row = _get_preview_char_row_for_record(host, chars[hint_idx], data=data)
                if target_row in (1, 2):
                    break

    candidates = []
    for idx, rec in enumerate(chars):
        if not isinstance(rec, dict):
            continue
        bbox = host._char_record_bbox(rec)
        if not bbox:
            continue
        row = _get_preview_char_row_for_record(host, rec, data=data) or 1
        if target_row in (1, 2) and row != target_row:
            continue
        try:
            reading_col = int(rec.get("reading_col", 0) or 0)
        except Exception:
            reading_col = 0
        try:
            x1 = float(bbox[0])
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        except Exception:
            continue
        candidates.append((row, reading_col if reading_col > 0 else 9999, x1, center_x, idx))

    if not candidates and target_row in (1, 2):
        return _get_preview_char_geometry_leader_index(host, row_hint=None)
    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    row, _col, _x1, _center_x, idx = candidates[0]
    return int(idx), int(row)


def _activate_preview_char_geometry_inheritance(host, event=None):
    self = host
    if self._get_preview_active_data(create=False) is None:
        self._update_preview_edit_status("G: najpierw wybierz tablicę z listy.", tone="warning")
        return "break"
    if getattr(self, "_preview_char_label_active_index", None) is not None:
        return None

    leader_idx, leader_row = _get_preview_char_geometry_leader_index(self)
    if leader_idx is None:
        self._update_preview_edit_status("G: brak boxów, z których można wybrać lidera rzędu.", tone="warning")
        return "break"

    already_active = bool(getattr(self, "_preview_char_geometry_inherit_down", False))
    selected_idx = getattr(self, "_preview_char_selected_index", None)
    try:
        selected_is_leader = selected_idx is not None and int(selected_idx) == int(leader_idx)
    except Exception:
        selected_is_leader = False

    self._preview_char_geometry_inherit_down = True
    self._preview_char_geometry_inherit_row = int(leader_row or 1)
    self._preview_char_edit_mode = True
    self._preview_char_add_mode = False
    self._preview_char_add_modifier_down = False
    self._preview_char_add_click_armed = False
    self._preview_char_add_state = None
    self._preview_char_label_mode = False
    self._preview_char_label_active_index = None
    self._preview_char_hover_label_index = None
    self._preview_char_hover_grip = None
    self._apply_preview_canvas_cursor()

    if already_active and selected_is_leader:
        return "break"

    return self._select_preview_character_box(
        int(leader_idx),
        activate_label=False,
        status_message=(
            f"G: liderem jest pierwszy box rzędu {int(leader_row or 1)}. "
            "Przeciągnij jego uchwyt, a po puszczeniu geometrię przejmą pozostałe boxy tego rzędu."
        ),
    )


def _deactivate_preview_char_geometry_inheritance(host, event=None):
    self = host
    was_active = bool(getattr(self, "_preview_char_geometry_inherit_down", False))
    self._preview_char_geometry_inherit_down = False
    self._preview_char_geometry_inherit_row = None
    if was_active and not isinstance(getattr(self, "_preview_char_drag_state", None), dict):
        self._update_preview_edit_status("G: wylaczono dziedziczenie geometrii boxow.", tone="muted")
        try:
            self._apply_preview_canvas_cursor()
        except Exception:
            pass
        selected_idx = getattr(self, "_preview_char_selected_index", None)
        if selected_idx is not None:
            styled_fast = False
            try:
                runtime = (getattr(self, "_preview_char_runtime", {}) or {}).get(f"FINAL:{int(selected_idx)}")
                canvas = getattr(self, "preview_canvas", None)
                if isinstance(runtime, dict) and canvas is not None:
                    styled_fast = bool(
                        _style_preview_character_edit_grips_fast(
                            self,
                            canvas,
                            runtime,
                            selection_color=getattr(getattr(self, "app", None), "palette", {}).get("accent", "#ffd166"),
                        )
                    )
            except Exception:
                styled_fast = False
            if styled_fast:
                return "break"
            try:
                self._refresh_preview_character_selection_visual({int(selected_idx)})
            except Exception:
                pass
    return "break" if was_active else None


def _toggle_preview_char_geometry_inheritance(host, event=None):
    self = host
    if bool(getattr(self, "_preview_char_geometry_inherit_key_down", False)):
        return "break"
    self._preview_char_geometry_inherit_key_down = True
    if bool(getattr(self, "_preview_char_geometry_inherit_down", False)):
        if isinstance(getattr(self, "_preview_char_drag_state", None), dict):
            self._update_preview_edit_status("G: zakoncz aktualny drag przed wylaczeniem trybu.", tone="warning")
            return "break"
        return _deactivate_preview_char_geometry_inheritance(self, event)
    return _activate_preview_char_geometry_inheritance(self, event)


def _deactivate_preview_char_geometry_inheritance_for_select(host) -> None:
    if not bool(getattr(host, "_preview_char_geometry_inherit_down", False)):
        return
    host._preview_char_geometry_inherit_down = False
    host._preview_char_geometry_inherit_row = None
    try:
        host._apply_preview_canvas_cursor()
    except Exception:
        pass


def _build_preview_char_geometry_inheritance_state(host, leader_idx: int, leader_row: int | None):
    chars = host._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not (0 <= int(leader_idx) < len(chars)):
        return None

    data = host._get_preview_active_data(create=False)
    row = leader_row if leader_row in (1, 2) else _get_preview_char_row_for_record(host, chars[int(leader_idx)], data=data)
    if row not in (1, 2):
        row = 1

    leader_bbox = host._char_record_bbox(chars[int(leader_idx)])
    if not leader_bbox:
        return None

    followers = []
    for idx, rec in enumerate(chars):
        if int(idx) == int(leader_idx) or not isinstance(rec, dict):
            continue
        bbox = host._char_record_bbox(rec)
        if not bbox:
            continue
        rec_row = _get_preview_char_row_for_record(host, rec, data=data) or 1
        if int(rec_row) != int(row):
            continue
        try:
            followers.append(
                {
                    "index": int(idx),
                    "record": rec,
                    "start_bbox": [float(v) for v in bbox[:4]],
                    "center_x": (float(bbox[0]) + float(bbox[2])) / 2.0,
                }
            )
        except Exception:
            continue

    if not followers:
        return None
    return {
        "leader_index": int(leader_idx),
        "row": int(row),
        "leader_start_bbox": [float(v) for v in leader_bbox[:4]],
        "followers": followers,
    }


def _apply_preview_char_geometry_inheritance(host, drag_state: dict) -> set[int]:
    group_state = drag_state.get("geometry_inherit")
    if not isinstance(group_state, dict):
        return set()

    chars = host._get_preview_active_character_records(create=False)
    if not isinstance(chars, list):
        return set()
    try:
        leader_idx = int(group_state.get("leader_index", drag_state.get("index", -1)))
    except Exception:
        leader_idx = -1
    if not (0 <= leader_idx < len(chars)):
        return set()

    leader_bbox = host._char_record_bbox(chars[leader_idx])
    if not leader_bbox:
        return set()
    try:
        target_w = max(4.0, float(leader_bbox[2]) - float(leader_bbox[0]))
        target_h = max(4.0, float(leader_bbox[3]) - float(leader_bbox[1]))
        target_y1 = float(leader_bbox[1])
        target_y2 = target_y1 + target_h
    except Exception:
        return set()

    state = getattr(host, "_preview_render_state", None) or {}
    try:
        image_w = max(1.0, float(state.get("orig_w", 1.0) or 1.0))
    except Exception:
        image_w = 1.0

    affected = {int(leader_idx)}
    row = int(group_state.get("row", drag_state.get("layout_row", 1)) or 1)
    for follower in list(group_state.get("followers", []) or []):
        if not isinstance(follower, dict):
            continue
        try:
            idx = int(follower.get("index", -1))
            rec = follower.get("record")
            center_x = float(follower.get("center_x", 0.0))
        except Exception:
            continue
        if not (0 <= idx < len(chars)) or rec is not chars[idx] or not isinstance(rec, dict):
            continue

        x1 = center_x - (target_w / 2.0)
        x1 = max(0.0, min(max(0.0, image_w - target_w), x1))
        new_bbox = [x1, target_y1, x1 + target_w, target_y2]
        normalized_bbox = host._normalize_preview_char_bbox(new_bbox)
        if normalized_bbox is None:
            continue
        normalized_bbox = host._constrain_preview_char_bbox_to_layout_separator(
            normalized_bbox,
            data=host._get_preview_active_data(create=False),
            row=row,
            min_size=4.0,
        )
        if normalized_bbox is None:
            continue
        old_bbox = host._char_record_bbox(rec)
        try:
            if old_bbox and all(abs(float(old_bbox[i]) - float(normalized_bbox[i])) < 0.25 for i in range(4)):
                continue
        except Exception:
            pass
        rec["bbox"] = normalized_bbox
        host._mark_preview_char_record_manual(rec, box=True, sign=False)
        affected.add(int(idx))

    return affected


def on_preview_list_mouse_primary(host, event):
    self = host
    try:
        self.plates_listbox.focus_set()
    except Exception:
        pass
    try:
        modifier_state = int(getattr(event, "state", 0) or 0)
    except Exception:
        modifier_state = 0

    try:
        target_index = int(self.plates_listbox.nearest(getattr(event, "y", 0)))
    except Exception:
        return None

    try:
        size = int(self.plates_listbox.size() or 0)
    except Exception:
        size = 0
    if size <= 0 or target_index < 0 or target_index >= size:
        return "break"

    # PZ2 is an editor for one plate at a time. Keeping range selection here
    # makes delayed Tk events feel like an accidental Shift-click under load.
    shift_pressed = False
    control_pressed = False
    preserve_index = self._get_current_preview_list_index()
    if preserve_index is not None and not (0 <= int(preserve_index) < size):
        preserve_index = None
    preserve_preview = False

    try:
        if shift_pressed:
            try:
                anchor_index = int(self.plates_listbox.index(tk.ANCHOR))
            except Exception:
                anchor_index = -1
            if anchor_index < 0 or anchor_index >= size:
                try:
                    anchor_index = int(self.plates_listbox.index(tk.ACTIVE))
                except Exception:
                    anchor_index = target_index
            anchor_index = max(0, min(anchor_index, size - 1))
            start_index = min(anchor_index, target_index)
            end_index = max(anchor_index, target_index)
            if not control_pressed:
                self._clear_listbox_selection_fast(self.plates_listbox)
            self.plates_listbox.selection_set(start_index, end_index)
            if preserve_index is not None:
                self.plates_listbox.activate(preserve_index)
            else:
                self.plates_listbox.activate(anchor_index)
            self.plates_listbox.see(target_index)
            preserve_preview = True
        elif control_pressed:
            if self.plates_listbox.selection_includes(target_index):
                self.plates_listbox.selection_clear(target_index)
            else:
                self.plates_listbox.selection_set(target_index)
            self.plates_listbox.selection_anchor(target_index)
            if preserve_index is not None:
                self.plates_listbox.activate(preserve_index)
            else:
                self.plates_listbox.activate(target_index)
            self.plates_listbox.see(target_index)
            preserve_preview = True
        else:
            self._suppress_preview_reload_on_list_select = True
            self._preview_fast_select_render = True
            self._clear_listbox_selection_fast(self.plates_listbox)
            self.plates_listbox.selection_set(target_index)
            self.plates_listbox.selection_anchor(target_index)
            self.plates_listbox.activate(target_index)
            self.plates_listbox.see(target_index)
    except Exception:
        return "break"

    if preserve_preview:
        self._suppress_preview_reload_on_list_select = True
        self._refresh_preview_editor_toolbar()
        return "break"

    try:
        scheduler = getattr(self, "_schedule_preview_select_render", None)
        if callable(scheduler):
            scheduler(delay_ms=1)
        else:
            self._suppress_preview_reload_on_list_select = False
            self._on_preview_select(None)
    except Exception:
        self._suppress_preview_reload_on_list_select = False
        self._on_preview_select(None)
    return "break"


def on_preview_canvas_motion(host, event=None):
    self = host
    if event is None:
        return
    motion_latency_start = time.perf_counter()
    click_add_state = getattr(self, "_preview_char_add_state", None)
    if isinstance(click_add_state, dict) and bool(click_add_state.get("click_draw")):
        _mark_preview_char_edit_interaction(self)
        if self._preview_point_inside_image(event.x, event.y):
            img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
            preview_bbox = [
                float(click_add_state.get("start_img_x", img_x)),
                float(click_add_state.get("start_img_y", img_y)),
                float(img_x),
                float(img_y),
            ]
            normalized_preview_bbox = self._normalize_preview_char_bbox(preview_bbox)
            click_add_state["bbox"] = normalized_preview_bbox if normalized_preview_bbox is not None else preview_bbox
            click_add_state["dirty"] = True
            if not self._redraw_preview_add_box_overlay_only():
                self._on_preview_select(None)
        self._apply_preview_canvas_cursor("crosshair")
        return
    if (
        self._preview_badge_drag_state is not None
        or getattr(self, "_preview_layout_separator_drag_state", None) is not None
        or self._preview_pan_drag_state is not None
        or self._preview_char_drag_state is not None
        or self._preview_char_add_state is not None
    ):
        return

    action_key = self._extract_preview_action_from_current_item()
    if action_key in {"reset_view", "edit_source_filename", "toggle_plate_layout", "toggle_plate_rows", "toggle_fullscreen"}:
        previous_hover_box = getattr(self, "_preview_char_hover_index", None)
        previous_hover_label = getattr(self, "_preview_char_hover_label_index", None)
        previous_hover_grip = getattr(self, "_preview_char_hover_grip", None)
        had_hover = bool(
            previous_hover_box is not None
            or previous_hover_label is not None
            or previous_hover_grip is not None
        )
        self._preview_char_hover_index = None
        self._preview_char_hover_grip = None
        if not bool(getattr(self, "_preview_char_label_mode", False)):
            self._preview_char_hover_label_index = None
        self._apply_preview_canvas_cursor("hand2")
        if had_hover and bool(
            getattr(self, "_preview_char_label_mode", False)
            or getattr(self, "_preview_char_label_active_index", None) is not None
        ):
            affected = {idx for idx in (previous_hover_box, previous_hover_label) if idx is not None}
            redrawn = False
            for idx in affected:
                redrawn = self._redraw_preview_character_overlay_only(int(idx)) or redrawn
        if previous_hover_grip is not None:
            selected_idx = getattr(self, "_preview_char_selected_index", None)
            if selected_idx is not None:
                try:
                    self._refresh_preview_character_selection_visual({int(selected_idx)})
                except Exception:
                    pass
        return

    label_mode_active = bool(getattr(self, "_preview_char_label_mode", False))
    edit_mode_active = bool(getattr(self, "_preview_char_edit_mode", False)) and not label_mode_active
    # Hover is not an edit. Updating the edit timestamp here postponed save
    # and counters indefinitely while the pointer moved over a selected box.
    grip_hit = self._find_preview_character_grip_hit(event.x, event.y) if edit_mode_active else None
    next_hover_grip = str((grip_hit or {}).get("key", "") or "") if isinstance(grip_hit, dict) else None
    if isinstance(grip_hit, dict):
        next_hover_box = int(grip_hit.get("index", -1))
        next_hover_label = None
    else:
        next_hover_box = self._find_preview_character_box_hit(event.x, event.y)
        next_hover_label = self._find_preview_char_label_hit(event.x, event.y)
    if label_mode_active:
        next_hover_label = next_hover_box if next_hover_box is not None else next_hover_label
    elif bool(getattr(self, "_preview_char_edit_mode", False)) and getattr(self, "_preview_char_label_active_index", None) is None:
        next_hover_label = None

    if next_hover_box is None and next_hover_label is None and self._find_preview_layout_separator_handle_hit(event.x, event.y):
        previous_hover_grip = getattr(self, "_preview_char_hover_grip", None)
        self._preview_char_hover_index = None
        self._preview_char_hover_label_index = None
        self._preview_char_hover_grip = None
        if previous_hover_grip is not None:
            selected_idx = getattr(self, "_preview_char_selected_index", None)
            if selected_idx is not None:
                try:
                    self._refresh_preview_character_selection_visual({int(selected_idx)})
                except Exception:
                    pass
        self._apply_preview_canvas_cursor("sb_v_double_arrow")
        return

    if isinstance(grip_hit, dict):
        self._apply_preview_canvas_cursor("crosshair" if str(grip_hit.get("kind", "")) == "corner" else "fleur")
    else:
        self._apply_preview_canvas_cursor()
    post_release_probe = getattr(self, "_preview_post_release_latency_probe", None)
    if isinstance(post_release_probe, dict) and next_hover_grip:
        self._preview_post_release_latency_probe = None
        try:
            release_start = float(post_release_probe.get("start", motion_latency_start))
            release_to_motion_ms = (motion_latency_start - release_start) * 1000.0
        except Exception:
            release_to_motion_ms = 0.0
        motion_to_ready_ms = (time.perf_counter() - motion_latency_start) * 1000.0
        self._log_preview_latency(
            post_release_probe,
            "post_release_next_grip_event_done",
            next_grip=next_hover_grip,
            release_to_motion_ms=round(release_to_motion_ms, 1),
            motion_to_ready_ms=round(motion_to_ready_ms, 1),
        )
        self._schedule_preview_latency_paint(
            post_release_probe,
            "post_release_next_grip_ready",
            next_grip=next_hover_grip,
            release_to_motion_ms=round(release_to_motion_ms, 1),
            motion_to_ready_ms=round(motion_to_ready_ms, 1),
        )
    if (
        next_hover_box == getattr(self, "_preview_char_hover_index", None)
        and next_hover_label == getattr(self, "_preview_char_hover_label_index", None)
        and next_hover_grip == getattr(self, "_preview_char_hover_grip", None)
    ):
        return

    previous_hover_box = getattr(self, "_preview_char_hover_index", None)
    previous_hover_label = getattr(self, "_preview_char_hover_label_index", None)
    previous_hover_grip = getattr(self, "_preview_char_hover_grip", None)
    self._preview_char_hover_index = next_hover_box
    self._preview_char_hover_label_index = next_hover_label
    self._preview_char_hover_grip = next_hover_grip
    if label_mode_active or getattr(self, "_preview_char_label_active_index", None) is not None:
        affected = {
            idx for idx in (previous_hover_box, previous_hover_label, next_hover_box, next_hover_label)
            if idx is not None
        }
        redrawn = False
        for idx in affected:
            redrawn = self._redraw_preview_character_overlay_only(int(idx)) or redrawn
    if previous_hover_grip != next_hover_grip:
        selected_idx = getattr(self, "_preview_char_selected_index", None)
        if selected_idx is not None:
            try:
                self._refresh_preview_character_selection_visual({int(selected_idx)})
            except Exception:
                pass


def on_preview_canvas_leave(host, event=None):
    self = host
    if (
        getattr(self, "_preview_char_drag_state", None) is not None
        or getattr(self, "_preview_char_add_state", None) is not None
        or getattr(self, "_preview_layout_separator_drag_state", None) is not None
    ):
        return
    if (
        getattr(self, "_preview_char_hover_index", None) is None
        and getattr(self, "_preview_char_hover_label_index", None) is None
        and getattr(self, "_preview_char_hover_grip", None) is None
    ):
        self._apply_preview_canvas_cursor()
        return
    previous_hover_box = getattr(self, "_preview_char_hover_index", None)
    previous_hover_label = getattr(self, "_preview_char_hover_label_index", None)
    previous_hover_grip = getattr(self, "_preview_char_hover_grip", None)
    self._preview_char_hover_index = None
    self._preview_char_hover_label_index = None
    self._preview_char_hover_grip = None
    self._apply_preview_canvas_cursor()
    if bool(
        getattr(self, "_preview_char_label_mode", False)
        or getattr(self, "_preview_char_label_active_index", None) is not None
    ):
        affected = {idx for idx in (previous_hover_box, previous_hover_label) if idx is not None}
        redrawn = False
        for idx in affected:
            redrawn = self._redraw_preview_character_overlay_only(int(idx)) or redrawn
    if previous_hover_grip is not None:
        selected_idx = getattr(self, "_preview_char_selected_index", None)
        if selected_idx is not None:
            try:
                self._refresh_preview_character_selection_visual({int(selected_idx)})
            except Exception:
                pass


def on_preview_canvas_keypress(host, event=None):
    self = host
    if event is None:
        return None

    keysym = str(getattr(event, "keysym", "") or "").lower()
    if keysym in {"alt_l", "alt_r", "option_l", "option_r"}:
        self._preview_alt_modifier_down = True
        return "break"

    label_mode_active = bool(getattr(self, "_preview_char_label_mode", False))
    if self._event_has_control_modifier(event):
        if keysym == "z":
            return self._undo_preview_edit(event)
        if keysym == "y":
            return self._redo_preview_edit(event)

    active_label_idx = getattr(self, "_preview_char_label_active_index", None)
    if keysym in {"left", "right"} and (label_mode_active or active_label_idx is not None):
        step = -1 if keysym == "left" else 1
        return self._cycle_preview_character_selection(step, activate_label=True)
    if keysym in {"up", "down"} and (label_mode_active or active_label_idx is not None):
        direction = -1 if keysym == "up" else 1
        return self._cycle_preview_character_row(direction, activate_label=True)
    typed_symbol = self._sanitize_preview_char_symbol(getattr(event, "char", ""))
    if active_label_idx is not None and typed_symbol:
        if self._assign_character_to_active_preview_label(typed_symbol):
            return "break"

    if keysym == "g":
        return _toggle_preview_char_geometry_inheritance(self, event)
    if keysym in {"return", "kp_enter"}:
        return self._on_preview_enter_fullscreen_shortcut(event)
    if keysym == "escape":
        return self._on_preview_escape_shortcut(event)
    if keysym == "q":
        return self._on_preview_prev_shortcut(event)
    if keysym == "e":
        return self._on_preview_next_shortcut(event)
    if keysym == "f":
        return self._on_preview_fit_shortcut(event)
    if keysym == "space":
        if bool(getattr(self, "_preview_char_geometry_inherit_down", False)):
            self._update_preview_edit_status(
                "G: tryb grupowy jest aktywny. Spacja nie zmienia teraz selekcji boxa.",
                tone="info",
            )
            return "break"
        step = -1 if self._event_has_shift_modifier(event) else 1
        return self._cycle_preview_character_selection(
            step,
            activate_label=bool(getattr(self, "_preview_char_label_mode", False)),
        )
    if keysym == "d":
        if active_label_idx is None:
            click_add_state = getattr(self, "_preview_char_add_state", None)
            if bool(getattr(self, "_preview_char_add_click_armed", False)) or (
                isinstance(click_add_state, dict) and bool(click_add_state.get("click_draw"))
            ):
                self._preview_char_add_click_armed = False
                self._preview_char_add_modifier_down = False
                self._preview_char_add_mode = False
                self._preview_char_add_state = None
                self._preview_char_hover_grip = None
                try:
                    self.preview_canvas.delete("preview_char_add_preview")
                except Exception:
                    pass
                self._refresh_preview_editor_toolbar()
                self._apply_preview_canvas_cursor()
                self._update_preview_edit_status("D: wylaczono uzbrojenie rysowania boxa.", tone="muted")
                return "break"
            if self._get_preview_active_data(create=False) is None:
                self._set_preview_box_info("Najpierw wybierz tablicę z listy.", "warning")
                return "break"
            self._ensure_preview_final_box_mode(render_preview=False)
            previous_selected_index = getattr(self, "_preview_char_selected_index", None)
            self._preview_char_add_click_armed = True
            self._preview_char_add_modifier_down = False
            self._preview_char_add_mode = False
            self._preview_char_add_state = None
            self._preview_char_edit_mode = False
            self._preview_char_label_mode = False
            self._preview_char_label_active_index = None
            self._preview_char_hover_label_index = None
            self._preview_char_hover_grip = None
            self._refresh_preview_editor_toolbar()
            self._apply_preview_canvas_cursor()
            self._update_preview_edit_status(
                "D uzbrojone. Kliknij pierwszy narożnik boxa, przesuń mysz i kliknij drugi narożnik.",
                tone="info",
            )
            if previous_selected_index is not None:
                try:
                    self._refresh_preview_character_selection_visual({int(previous_selected_index)})
                except Exception:
                    pass
            return "break"
        return None
    if keysym == "n":
        if active_label_idx is None:
            return self._toggle_preview_char_add_mode(event)
        return None
    if keysym == "s":
        if not bool(getattr(self, "_preview_char_label_mode", False)):
            if bool(getattr(self, "_preview_char_geometry_inherit_down", False)):
                _deactivate_preview_char_geometry_inheritance_for_select(self)
            try:
                self._preview_pending_select_latency_probe = self._start_preview_latency_probe(
                    "key_s_select",
                    hover=getattr(self, "_preview_char_hover_index", "-"),
                    selected=getattr(self, "_preview_char_selected_index", "-"),
                    grip=getattr(self, "_preview_char_hover_grip", "-"),
                )
            except Exception:
                self._preview_pending_select_latency_probe = None
            return self._select_hovered_preview_char_box(event)
        return None
    if keysym == "t":
        if active_label_idx is None:
            return self._edit_selected_preview_char_symbol(event)
        return None
    return None


def on_preview_canvas_keyrelease(host, event=None):
    self = host
    if event is None:
        return None
    keysym = str(getattr(event, "keysym", "") or "").lower()
    if keysym in {"alt_l", "alt_r", "option_l", "option_r"}:
        self._preview_alt_modifier_down = False
        return "break"
    if keysym == "g":
        self._preview_char_geometry_inherit_key_down = False
        return "break"
    if keysym == "d" and bool(getattr(self, "_preview_char_add_modifier_down", False)):
        self._preview_char_add_modifier_down = False
        self._apply_preview_canvas_cursor()
        if getattr(self, "_preview_char_add_state", None) is None:
            self._update_preview_edit_status()
        return "break"
    return None


def on_preview_canvas_press(host, event):
    self = host
    self._focus_preview_canvas()
    action_key = self._extract_preview_action_from_current_item()
    if action_key == "reset_view":
        self._reset_current_preview_view()
        return "break"
    if action_key == "toggle_fullscreen":
        self._toggle_preview_fullscreen()
        return "break"
    if action_key == "edit_source_filename":
        return self._edit_preview_source_filename(event)
    if action_key == "toggle_plate_layout":
        return self._cycle_preview_plate_layout_override(event)
    if action_key == "toggle_plate_rows":
        from .z3_preview_layout_control import toggle_plate_rows
        return toggle_plate_rows(self)

    badge_key = self._extract_preview_badge_key_from_current_item()
    if badge_key:
        self._set_preview_selected_badge(badge_key)
        self._preview_pan_drag_state = None
        start_dx, start_dy = self._get_preview_badge_offset(self._preview_active_pid, badge_key)
        self._preview_badge_drag_state = {
            "badge_key": str(badge_key),
            "start_x": float(event.x),
            "start_y": float(event.y),
            "start_dx": float(start_dx),
            "start_dy": float(start_dy),
        }
        try:
            self.preview_canvas.configure(cursor="hand2")
        except Exception:
            pass
        return "break"

    if not getattr(self, "_preview_render_state", None):
        return

    if _preview_has_clearable_canvas_action(self) and not _preview_point_inside_plate_status_frame(
        self,
        event.x,
        event.y,
    ):
        return self._clear_preview_canvas_action(event)

    if getattr(self, "_preview_selected_badge_key", None) is not None:
        self._set_preview_selected_badge(None)
    self._preview_badge_drag_state = None
    self._preview_pan_drag_state = None
    edit_mode = bool(getattr(self, "_preview_char_edit_mode", False))
    label_mode = bool(getattr(self, "_preview_char_label_mode", False))
    if edit_mode or bool(getattr(self, "_preview_char_add_mode", False)) or getattr(self, "_preview_char_add_state", None) is not None:
        _mark_preview_char_edit_interaction(self)

    click_add_state = getattr(self, "_preview_char_add_state", None)
    if isinstance(click_add_state, dict) and bool(click_add_state.get("click_draw")):
        if not self._preview_point_inside_image(event.x, event.y):
            self._update_preview_edit_status(
                "Drugi narożnik boxa musi być wewnątrz tablicy.",
                tone="warning",
            )
            return "break"
        img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
        click_add_state["bbox"] = [
            float(click_add_state.get("start_img_x", img_x)),
            float(click_add_state.get("start_img_y", img_y)),
            float(img_x),
            float(img_y),
        ]
        click_add_state["dirty"] = True
        click_add_state["click_finalize"] = True
        return self._finalize_preview_char_add_state()

    if bool(getattr(self, "_preview_char_add_click_armed", False)):
        if not self._preview_point_inside_image(event.x, event.y):
            self._update_preview_edit_status(
                "Kliknij pierwszy narożnik boxa wewnątrz tablicy.",
                tone="warning",
            )
            return "break"
        self._preview_char_edit_mode = False
        self._preview_char_add_mode = False
        self._preview_char_add_modifier_down = False
        img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
        self._preview_char_selected_index = None
        self._preview_char_hover_grip = None
        self._preview_char_add_state = {
            "start_img_x": float(img_x),
            "start_img_y": float(img_y),
            "bbox": [float(img_x), float(img_y), float(img_x), float(img_y)],
            "dirty": False,
            "click_draw": True,
        }
        self._refresh_preview_editor_toolbar()
        self._apply_preview_canvas_cursor("crosshair")
        self._update_preview_edit_status(
            "Pierwszy narożnik ustawiony. Przesuń mysz i kliknij drugi narożnik boxa.",
            tone="info",
        )
        if not self._redraw_preview_add_box_overlay_only():
            self._on_preview_select(None)
        return "break"

    if edit_mode and not label_mode:
        grip_hit = self._find_preview_character_grip_hit(event.x, event.y)
        if isinstance(grip_hit, dict) and str(grip_hit.get("kind", "")) == "corner":
            char_idx = int(grip_hit.get("index", -1))
            handle_name = str(grip_hit.get("handle", "se") or "se")
            probe = self._start_preview_latency_probe(
                "corner_press",
                idx=char_idx,
                handle=handle_name,
            )
            if self._start_preview_character_box_drag(
                char_idx,
                "resize",
                event,
                handle_name=handle_name,
                cursor="crosshair",
            ):
                drag_state = getattr(self, "_preview_char_drag_state", None)
                if isinstance(drag_state, dict):
                    drag_state["latency_probe"] = probe
                self._log_preview_latency(probe, "corner_press_event_done", idx=char_idx, handle=handle_name)
                self._schedule_preview_latency_paint(probe, "corner_grab_ready", idx=char_idx, handle=handle_name)
                return "break"
        if isinstance(grip_hit, dict) and str(grip_hit.get("kind", "")) == "move":
            char_idx = int(grip_hit.get("index", -1))
            probe = self._start_preview_latency_probe(
                "center_press",
                idx=char_idx,
                handle="center",
            )
            if self._start_preview_character_box_drag(char_idx, "move", event, cursor="fleur"):
                drag_state = getattr(self, "_preview_char_drag_state", None)
                if isinstance(drag_state, dict):
                    drag_state["latency_probe"] = probe
                self._log_preview_latency(probe, "center_press_event_done", idx=char_idx, handle="center")
                self._schedule_preview_latency_paint(probe, "center_grab_ready", idx=char_idx, handle="center")
                return "break"

    label_hit = self._find_preview_char_label_hit(event.x, event.y)
    box_hit = self._find_preview_character_box_hit(event.x, event.y)
    separator_handle = None
    if label_hit is None and box_hit is None:
        separator_handle = self._find_preview_layout_separator_handle_hit(event.x, event.y)
    if separator_handle:
        self._push_preview_history_snapshot()
        self._preview_pan_drag_state = None
        self._preview_badge_drag_state = None
        separator_runtime = getattr(self, "_preview_layout_separator_runtime", None)
        start_separator = {}
        if isinstance(separator_runtime, dict):
            start_separator = copy.deepcopy(separator_runtime.get("separator", {}) or {})
        self._preview_layout_separator_drag_state = {
            "handle": str(separator_handle),
            "start_separator": start_separator,
            "start_canvas_x": float(event.x),
            "start_canvas_y": float(event.y),
            "dirty": False,
        }
        try:
            self.preview_canvas.configure(cursor="sb_v_double_arrow")
        except Exception:
            pass
        return "break"

    if label_mode:
        target_idx = label_hit if label_hit is not None else box_hit
        if target_idx is not None:
            return self._select_preview_character_box(
                int(target_idx),
                activate_label=True,
                status_message="Wpisywanie znaków: wpisz 0-9 lub A-Z, aby nadpisać znak w aktywnym boxie.",
            )

    if label_hit is not None and getattr(self, "_preview_char_label_active_index", None) is not None:
        self._preview_char_edit_mode = True
        self._preview_char_selected_index = int(label_hit)
        self._preview_char_hover_label_index = int(label_hit)
        self._refresh_preview_editor_toolbar()
        self._activate_preview_char_label_input(
            int(label_hit),
            status_message="Pole znaku pozostaje aktywne. Wpisz 0-9 lub A-Z, aby nadpisać etykietę, albo Esc aby wyjść.",
        )
        return "break"

    if getattr(self, "_preview_char_label_active_index", None) is not None:
        self._preview_char_label_active_index = None

    if self._preview_char_add_requested() and self._preview_point_inside_image(event.x, event.y):
        self._preview_char_edit_mode = False
        img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
        self._preview_char_selected_index = None
        self._preview_char_hover_grip = None
        self._preview_char_add_state = {
            "start_img_x": float(img_x),
            "start_img_y": float(img_y),
            "bbox": [float(img_x), float(img_y), float(img_x), float(img_y)],
            "dirty": False,
        }
        self._refresh_preview_editor_toolbar()
        try:
            self.preview_canvas.configure(cursor="crosshair")
        except Exception:
            pass
        try:
            self.preview_canvas.delete("preview_char_add_preview")
        except Exception:
            pass
        return "break"

    if box_hit is not None:
        if (
            edit_mode
            and getattr(self, "_preview_char_selected_index", None) is not None
            and int(self._preview_char_selected_index) == int(box_hit)
        ):
            self._update_preview_edit_status(
                "Zaznaczony box: przeciągaj środkowy okrąg, aby przenieść ramkę, albo okrąg narożnika, aby zmienić rozmiar.",
                tone="info",
            )
            return "break"
        self._update_preview_edit_status(
            "Najedź kursorem na box i naciśnij S, aby go zaznaczyć. Zaznaczony box przesuwasz środkowym okręgiem, zmieniasz narożnikami i usuwasz PPM.",
            tone="info",
        )
        if edit_mode or label_mode:
            return "break"

    previous_selected_index = getattr(self, "_preview_char_selected_index", None)
    self._preview_char_selected_index = None
    self._preview_char_hover_grip = None
    self._refresh_preview_editor_toolbar()
    self._preview_pan_drag_state = {
        "start_x": float(event.x),
        "start_y": float(event.y),
        "start_pan_x": float(self._preview_pan_x),
        "start_pan_y": float(self._preview_pan_y),
    }
    try:
        self.preview_canvas.configure(cursor="fleur")
    except Exception:
        pass
    if previous_selected_index is not None:
        try:
            self._refresh_preview_character_selection_visual({int(previous_selected_index)})
        except Exception:
            pass
    return "break"


def _preview_has_clearable_canvas_action(host) -> bool:
    return bool(
        getattr(host, "_preview_char_label_mode", False)
        or getattr(host, "_preview_char_label_active_index", None) is not None
        or getattr(host, "_preview_char_add_state", None) is not None
        or getattr(host, "_preview_char_add_click_armed", False)
        or getattr(host, "_preview_char_add_modifier_down", False)
        or getattr(host, "_preview_char_add_mode", False)
    )


def _preview_point_inside_image_with_padding(host, canvas_x: float, canvas_y: float, *, padding: float = 18.0) -> bool:
    state = getattr(host, "_preview_render_state", None) or {}
    try:
        pad = max(0.0, float(padding))
        return bool(
            float(state.get("image_left", 0.0)) - pad <= float(canvas_x) <= float(state.get("image_right", -1.0)) + pad
            and float(state.get("image_top", 0.0)) - pad <= float(canvas_y) <= float(state.get("image_bottom", -1.0)) + pad
        )
    except Exception:
        return False


def _preview_point_inside_plate_status_frame(host, canvas_x: float, canvas_y: float) -> bool:
    try:
        handle_radius = float(host._get_preview_char_handle_radius())
    except Exception:
        handle_radius = 9.0
    frame_pad = max(20.0, min(30.0, handle_radius + 12.0))
    return _preview_point_inside_image_with_padding(host, canvas_x, canvas_y, padding=frame_pad)


def on_preview_canvas_drag(host, event):
    self = host
    separator_drag_state = getattr(self, "_preview_layout_separator_drag_state", None)
    if isinstance(separator_drag_state, dict):
        _mark_preview_char_edit_interaction(self)
        if not self._is_preview_layout_separator_interactive(ignore_active_char=True):
            self._preview_layout_separator_drag_state = None
            try:
                self.preview_canvas.configure(cursor="arrow")
            except Exception:
                pass
            return "break"
        separator = self._build_preview_layout_separator_drag_preview(separator_drag_state, event.x, event.y)
        if isinstance(separator, dict):
            separator_drag_state["dirty"] = True
            separator_drag_state["preview_separator"] = separator
            active_data = self._get_preview_active_data(create=False)
            if isinstance(active_data, dict):
                rows = tuple(self._get_preview_row_for_bbox(rec.get("bbox"), active_data)
                             for rec in active_data.get("characters", []) if isinstance(rec, dict))
                if rows != separator_drag_state.get("badge_rows"):
                    separator_drag_state["badge_rows"] = rows
                    self._redraw_preview_character_overlays_light()
            if not self._update_preview_layout_separator_visual(separator, active_data):
                if isinstance(active_data, dict):
                    preview_data = dict(active_data)
                    preview_data["layout_separator"] = dict(separator)
                    preview_data["plate_layout_override"] = "two_row"
                    preview_data["layout_override_source"] = "separator"
                    self._draw_preview_layout_separator(preview_data)
                    self._update_preview_layout_separator_visual(separator, preview_data)
        return "break"

    drag_state = self._preview_badge_drag_state
    if isinstance(drag_state, dict):
        badge_key = drag_state.get("badge_key")
        new_dx = float(drag_state.get("start_dx", 0.0)) + (float(event.x) - float(drag_state.get("start_x", 0.0)))
        new_dy = float(drag_state.get("start_dy", 0.0)) + (float(event.y) - float(drag_state.get("start_y", 0.0)))
        new_dx, new_dy = self._clamp_preview_badge_offset(badge_key, new_dx, new_dy)
        self._set_preview_badge_offset(self._preview_active_pid, badge_key, new_dx, new_dy)
        self._move_preview_badge_to_offset(badge_key, new_dx, new_dy)
        return "break"

    char_drag_state = getattr(self, "_preview_char_drag_state", None)
    if isinstance(char_drag_state, dict):
        drag_perf_start = time.perf_counter()
        visual_ms = 0.0
        _mark_preview_char_edit_interaction(self)
        chars = self._get_preview_active_character_records(create=False)
        char_idx = int(char_drag_state.get("index", -1))
        if 0 <= char_idx < len(chars):
            rec = chars[char_idx]
            bbox = list(char_drag_state.get("start_bbox", rec.get("bbox", [0, 0, 0, 0])))
            img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
            mode = str(char_drag_state.get("mode", "move") or "move").lower()

            if mode == "move":
                start_x = float(char_drag_state.get("start_img_x", img_x))
                start_y = float(char_drag_state.get("start_img_y", img_y))
                delta_x = float(img_x) - start_x
                delta_y = float(img_y) - start_y
                width = max(1.0, float(bbox[2]) - float(bbox[0]))
                height = max(1.0, float(bbox[3]) - float(bbox[1]))
                state = getattr(self, "_preview_render_state", None) or {}
                max_w = max(width, float(state.get("orig_w", width)))
                max_h = max(height, float(state.get("orig_h", height)))
                new_x1 = min(max(0.0, float(bbox[0]) + delta_x), max_w - width)
                new_y1 = min(max(0.0, float(bbox[1]) + delta_y), max_h - height)
                new_bbox = [new_x1, new_y1, new_x1 + width, new_y1 + height]
            else:
                handle_name = str(char_drag_state.get("handle", "se") or "se").lower()
                x1, y1, x2, y2 = bbox
                if "w" in handle_name:
                    x1 = float(img_x)
                else:
                    x2 = float(img_x)
                if "n" in handle_name:
                    y1 = float(img_y)
                else:
                    y2 = float(img_y)
                new_bbox = [x1, y1, x2, y2]

            normalized_bbox = self._normalize_preview_char_bbox(new_bbox)
            if normalized_bbox is not None:
                active_data = self._get_preview_active_data(create=False)
                normalized_bbox = self._constrain_preview_char_bbox_to_layout_separator(
                    normalized_bbox,
                    data=active_data,
                    row=char_drag_state.get("layout_row"),
                    min_size=4.0,
                )
            if normalized_bbox is not None:
                current_bbox = self._char_record_bbox(rec)
                try:
                    if current_bbox and all(
                        abs(float(current_bbox[i]) - float(normalized_bbox[i])) < 0.25
                        for i in range(4)
                    ):
                        return "break"
                except Exception:
                    pass
                rec["bbox"] = normalized_bbox
                if not bool(char_drag_state.get("manual_marked")):
                    self._mark_preview_char_record_manual(rec, box=True, sign=False)
                    char_drag_state["manual_marked"] = True
                char_drag_state["dirty"] = True
                if not bool(char_drag_state.get("motion_flow_logged")):
                    char_drag_state["motion_flow_logged"] = True
                    try:
                        self._log_preview_edit_flow(
                            "char_drag_motion",
                            trace=char_drag_state.get("trace_id", "-"),
                            idx=char_idx,
                            mode=str(char_drag_state.get("mode", "-") or "-"),
                            handle=str(char_drag_state.get("handle", "-") or "-"),
                            bbox=",".join(str(round(float(v), 1)) for v in normalized_bbox[:4]),
                        )
                    except Exception:
                        pass
                visual_ids = char_drag_state.get("visual_ids")
                now = time.monotonic()
                last_visual_at = float(char_drag_state.get("last_visual_at", 0.0) or 0.0)
                should_draw_visual = (
                    not isinstance(visual_ids, dict)
                    or (now - last_visual_at) >= 0.024
                )
                if should_draw_visual:
                    char_drag_state["last_visual_at"] = now
                    phase_start = time.perf_counter()
                    if not self._update_preview_character_drag_visual(char_idx, rec, normalized_bbox):
                        self._on_preview_select(None)
                    visual_ms = (time.perf_counter() - phase_start) * 1000.0
                    probe = char_drag_state.get("latency_probe")
                    if isinstance(probe, dict) and not bool(char_drag_state.get("first_visual_latency_logged")):
                        char_drag_state["first_visual_latency_logged"] = True
                        self._log_preview_latency(
                            probe,
                            "drag_first_visual_event_done",
                            idx=char_idx,
                            visual_ms=round(visual_ms, 1),
                        )
                        self._schedule_preview_latency_paint(
                            probe,
                            "drag_first_visual",
                            idx=char_idx,
                            visual_ms=round(visual_ms, 1),
                        )
        total_ms = (time.perf_counter() - drag_perf_start) * 1000.0
        if total_ms >= 70.0:
            logger.info(
                "[Z3/PZ2 PERF] char_drag_motion total=%.1fms visual=%.1fms dirty=%s",
                total_ms,
                visual_ms,
                bool(char_drag_state.get("dirty")),
            )
        return "break"

    char_add_state = getattr(self, "_preview_char_add_state", None)
    if isinstance(char_add_state, dict):
        img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
        preview_bbox = [
            float(char_add_state.get("start_img_x", img_x)),
            float(char_add_state.get("start_img_y", img_y)),
            float(img_x),
            float(img_y),
        ]
        normalized_preview_bbox = self._normalize_preview_char_bbox(preview_bbox)
        char_add_state["bbox"] = normalized_preview_bbox if normalized_preview_bbox is not None else preview_bbox
        char_add_state["dirty"] = True
        if not self._redraw_preview_add_box_overlay_only():
            self._on_preview_select(None)
        return "break"

    pan_state = self._preview_pan_drag_state
    if not isinstance(pan_state, dict):
        return

    state = getattr(self, "_preview_render_state", None) or {}
    canvas = getattr(self, "preview_canvas", None)
    if not state or canvas is None:
        return "break"

    old_left = float(state.get("image_left", 0.0) or 0.0)
    old_top = float(state.get("image_top", 0.0) or 0.0)
    image_w = max(1.0, float(state.get("image_right", old_left + 1.0) or old_left + 1.0) - old_left)
    image_h = max(1.0, float(state.get("image_bottom", old_top + 1.0) or old_top + 1.0) - old_top)
    base_x_off = float(state.get("fit_x_off", old_left) or old_left) - ((image_w - float(state.get("fit_new_w", image_w) or image_w)) / 2.0)
    base_y_off = float(state.get("fit_y_off", old_top) or old_top) - ((image_h - float(state.get("fit_new_h", image_h) or image_h)) / 2.0)
    target_x = base_x_off + float(pan_state.get("start_pan_x", 0.0)) + (float(event.x) - float(pan_state.get("start_x", 0.0)))
    target_y = base_y_off + float(pan_state.get("start_pan_y", 0.0)) + (float(event.y) - float(pan_state.get("start_y", 0.0)))
    try:
        canvas_w = float(canvas.winfo_width() or 0)
        canvas_h = float(canvas.winfo_height() or 0)
    except Exception:
        canvas_w = canvas_h = 0.0
    target_x, target_y = self._clamp_preview_image_position(
        target_x,
        target_y,
        image_w=image_w,
        image_h=image_h,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        top_reserved=float(state.get("top_reserved", 0.0) or 0.0),
        bottom_reserved=float(state.get("bottom_reserved", 0.0) or 0.0),
        preferred_x=base_x_off,
        preferred_y=base_y_off,
    )
    dx = float(target_x) - old_left
    dy = float(target_y) - old_top
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return "break"

    self._preview_pan_x = float(target_x - base_x_off)
    self._preview_pan_y = float(target_y - base_y_off)
    state["image_left"] = float(target_x)
    state["image_top"] = float(target_y)
    state["image_right"] = float(target_x + image_w)
    state["image_bottom"] = float(target_y + image_h)
    self._preview_render_state = state
    for tag in (
        "preview_plate_image",
        "preview_char",
        "preview_fast_detail",
        "preview_badge",
        "preview_plate_status_frame",
        "preview_layout_separator",
        "preview_char_drag_preview",
        "preview_char_add_preview",
    ):
        try:
            canvas.move(tag, dx, dy)
        except Exception:
            pass
    for runtime in (getattr(self, "_preview_badge_runtime", {}) or {}).values():
        if not isinstance(runtime, dict):
            continue
        for key in ("base_center_x", "base_line_y", "base_bottom_y", "box_anchor_y"):
            if key not in runtime:
                continue
            try:
                runtime[key] = float(runtime.get(key, 0.0) or 0.0) + (dy if key != "base_center_x" else dx)
            except Exception:
                pass
    for badge_key, runtime in list((getattr(self, "_preview_badge_runtime", {}) or {}).items()):
        if not isinstance(runtime, dict):
            continue
        try:
            saved_dx, saved_dy = self._get_preview_badge_offset(getattr(self, "_preview_active_pid", ""), str(badge_key))
            self._move_preview_badge_to_offset(
                str(badge_key),
                float(saved_dx),
                float(saved_dy),
            )
        except Exception:
            pass
    return "break"


def on_preview_canvas_release(host, event):
    self = host
    separator_drag_state = getattr(self, "_preview_layout_separator_drag_state", None)
    if isinstance(separator_drag_state, dict):
        self._preview_layout_separator_drag_state = None
        try:
            self.preview_canvas.configure(cursor="arrow")
        except Exception:
            pass
        if bool(separator_drag_state.get("dirty")):
            _mark_preview_char_edit_interaction(self)
            release_perf_start = time.perf_counter()
            data = self._get_preview_active_data(create=True)
            if isinstance(data, dict):
                separator = separator_drag_state.get("preview_separator")
                if isinstance(separator, dict):
                    state = getattr(self, "_preview_render_state", None) or {}
                    try:
                        image_w = max(1.0, float(state.get("orig_w", data.get("plate_image_width", 1.0)) or 1.0))
                        image_h = max(1.0, float(state.get("orig_h", data.get("plate_image_height", 1.0)) or 1.0))
                    except Exception:
                        image_w = max(1.0, float(data.get("plate_image_width", 1.0) or 1.0))
                        image_h = max(1.0, float(data.get("plate_image_height", 1.0) or 1.0))
                    data["layout_separator"] = dict(separator)
                    data["plate_image_width"] = float(image_w)
                    data["plate_image_height"] = float(image_h)
                    data["plate_layout_override"] = "two_row"
                    data["layout_override_source"] = "separator"
                    data["manual_layout"] = True
                    data["layout_manual"] = True
                    data["layout_override_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    data["plate_layout"] = "two_row"
                    data["layout_row_count"] = 2
                    data["layout_confidence"] = 1.0
                    data["layout_source"] = "manual_override"
                chars = list(data.get("characters", []) or []) if isinstance(data.get("characters"), list) else []
                ordered_chars = self._sort_character_records_by_x(chars, data=data)
                ordered_chars = self._annotate_preview_character_reading_positions(ordered_chars, data=data)
                data["characters"] = ordered_chars
                separator_conflict = False
                try:
                    separator_conflict = bool(self._preview_layout_separator_conflicts_with_chars(data, ordered_chars))
                except Exception:
                    separator_conflict = False
                status_now = self._derive_preview_status_from_characters(ordered_chars)
                if status_now == "perfect" and separator_conflict:
                    status_now = "needs_fix"
                elif status_now == "perfect":
                    try:
                        expected_resolution = self._resolve_preview_expected_text_for_crop(data, ordered_chars)
                        expected_texts = list(expected_resolution.get("expected_texts", []) or [])
                        if expected_texts:
                            status_now = "perfect" if bool(expected_resolution.get("text_resolved")) else "needs_fix"
                        elif self._preview_has_reference_text_source(data):
                            status_now = "needs_fix"
                    except Exception:
                        pass
                data["status"] = status_now
                try:
                    self._schedule_preview_metadata_save(delay_ms=650)
                except Exception:
                    pass
                try:
                    self._redraw_preview_character_overlays_light()
                    if not (isinstance(separator, dict) and self._update_preview_layout_separator_visual(separator, data)):
                        self._draw_preview_layout_separator(data)
                except Exception:
                    pass
                self._refresh_preview_live_metadata_ui(
                    status_message=(
                        "Belka rzędów ustawiona. Box nachodzący na belkę wymaga korekty."
                        if separator_conflict
                        else "Zaktualizowano belkę podziału rzędów tablicy."
                    ),
                    status_tone="warning" if separator_conflict else "info",
                    render_preview=False,
                )
                total_ms = (time.perf_counter() - release_perf_start) * 1000.0
                if total_ms >= 80.0:
                    logger.info(
                        "[Z3/PZ2 PERF] separator_release total=%.1fms chars=%s conflict=%s",
                        total_ms,
                        len(ordered_chars),
                        int(bool(separator_conflict)),
                    )
        else:
            self._on_preview_select(None)
        return "break"

    char_drag_state = getattr(self, "_preview_char_drag_state", None)
    if isinstance(char_drag_state, dict):
        _mark_preview_char_edit_interaction(self)
        release_probe = self._start_preview_latency_probe(
            "corner_release" if str(char_drag_state.get("mode", "")) == "resize" else "center_release",
            trace=char_drag_state.get("trace_id", "-"),
            idx=char_drag_state.get("index", "-"),
            handle=char_drag_state.get("handle", "center"),
        )
        release_perf_start = time.perf_counter()
        history_ms = persist_ms = cleanup_ms = 0.0
        self._unbind_preview_char_drag_session()
        try:
            self.preview_canvas.configure(cursor="arrow")
        except Exception:
            pass
        if bool(char_drag_state.get("dirty")):
            try:
                chars = self._get_preview_active_character_records(create=False)
                char_idx = int(char_drag_state.get("index", -1))
                selected_record = chars[char_idx] if 0 <= char_idx < len(chars) else None
                try:
                    self._log_preview_edit_flow(
                        "char_drag_release",
                        trace=char_drag_state.get("trace_id", "-"),
                        idx=char_idx,
                        dirty=int(bool(char_drag_state.get("dirty"))),
                        chars=len(chars),
                        **_preview_char_record_trace_fields(
                            self,
                            selected_record,
                            data=self._get_preview_active_data(create=False),
                            fallback_index=char_idx,
                        ),
                    )
                except Exception:
                    pass
                phase_start = time.perf_counter()
                _push_preview_char_drag_history_snapshot(self, char_drag_state)
                history_ms = (time.perf_counter() - phase_start) * 1000.0
                group_affected_indices = set()
                if isinstance(char_drag_state.get("geometry_inherit"), dict):
                    group_affected_indices = _apply_preview_char_geometry_inheritance(self, char_drag_state)
                    try:
                        if len(group_affected_indices) > 1:
                            group_state = char_drag_state.get("geometry_inherit") or {}
                            self._log_preview_edit_flow(
                                "char_geometry_inherit_applied",
                                trace=char_drag_state.get("trace_id", "-"),
                                row=group_state.get("row", "-"),
                                affected=len(group_affected_indices),
                            )
                    except Exception:
                        pass
                # Keep the drag preview visible until the canonical box is rebuilt.
                phase_start = time.perf_counter()
                self._persist_active_preview_characters(
                    selected_record=selected_record,
                    success_message="Zapisano ręczna korekte boxu znaku w metadata.json.",
                    render_preview=False,
                    save_immediately=False,
                    save_delay_ms=1400,
                    refresh_row=False,
                    light_redraw_indices=(group_affected_indices if len(group_affected_indices) > 1 else "selected"),
                )
                persist_ms = (time.perf_counter() - phase_start) * 1000.0
                try:
                    self._log_preview_edit_flow(
                        "char_drag_release_done",
                        trace=char_drag_state.get("trace_id", "-"),
                        idx=char_idx,
                        persist_ms=round(persist_ms, 1),
                    )
                except Exception:
                    pass
            finally:
                phase_start = time.perf_counter()
                self._clear_preview_character_drag_visual()
                self._preview_char_drag_state = None
                cleanup_ms = (time.perf_counter() - phase_start) * 1000.0
                total_ms = (time.perf_counter() - release_perf_start) * 1000.0
                if total_ms >= 180.0:
                    logger.info(
                        "[Z3/PZ2 PERF] char_drag_release total=%.1fms history=%.1fms persist=%.1fms cleanup=%.1fms",
                        total_ms,
                        history_ms,
                        persist_ms,
                        cleanup_ms,
                    )
                self._log_preview_latency(
                    release_probe,
                    "release_event_done",
                    history_ms=round(history_ms, 1),
                    persist_ms=round(persist_ms, 1),
                    cleanup_ms=round(cleanup_ms, 1),
                )
                self._schedule_preview_latency_paint(
                    release_probe,
                    "release_ready",
                    history_ms=round(history_ms, 1),
                    persist_ms=round(persist_ms, 1),
                    cleanup_ms=round(cleanup_ms, 1),
                )
                self._preview_post_release_latency_probe = release_probe
        else:
            self._clear_preview_character_drag_visual()
            self._preview_char_drag_state = None
            self._on_preview_select(None)
            self._log_preview_latency(release_probe, "release_event_done", dirty=0)
            self._schedule_preview_latency_paint(release_probe, "release_ready", dirty=0)
            self._preview_post_release_latency_probe = release_probe
        return "break"

    char_add_state = getattr(self, "_preview_char_add_state", None)
    if isinstance(char_add_state, dict):
        if bool(char_add_state.get("click_draw")) and not bool(char_add_state.get("click_finalize")):
            return "break"
        return self._finalize_preview_char_add_state()

    if self._preview_badge_drag_state is None and self._preview_pan_drag_state is None:
        return
    self._preview_badge_drag_state = None
    self._preview_pan_drag_state = None
    try:
        self.preview_canvas.configure(cursor=("crosshair" if self._preview_char_add_requested() else "arrow"))
    except Exception:
        pass
    return "break"


def _preview_zoom_interaction_blocked(host) -> bool:
    return bool(
        getattr(host, "_preview_badge_drag_state", None) is not None
        or getattr(host, "_preview_layout_separator_drag_state", None) is not None
        or getattr(host, "_preview_pan_drag_state", None) is not None
        or getattr(host, "_preview_char_drag_state", None) is not None
        or getattr(host, "_preview_char_add_state", None) is not None
    )


def _get_preview_safe_zoom_max(host, state: dict | None = None) -> float:
    source_state = state if isinstance(state, dict) else (getattr(host, "_preview_render_state", {}) or {})
    try:
        safe_zoom_max = float(
            source_state.get(
                "safe_zoom_max",
                getattr(host, "_preview_safe_zoom_max", getattr(host, "_preview_zoom_max", 2.4)),
            )
        )
    except Exception:
        safe_zoom_max = float(getattr(host, "_preview_zoom_max", 2.4) or 2.4)
    return max(1.0, min(float(getattr(host, "_preview_zoom_max", 2.4) or 2.4), safe_zoom_max))


def _capture_preview_zoom_anchor(host, event, state: dict) -> dict:
    image_left = float(state.get("image_left", 0.0))
    image_top = float(state.get("image_top", 0.0))
    image_right = float(state.get("image_right", image_left))
    image_bottom = float(state.get("image_bottom", image_top))
    current_display_w = max(1.0, image_right - image_left)
    current_display_h = max(1.0, image_bottom - image_top)

    # Zoom w edytorze ma stabilizowac aktualne polozenie tablicy, a nie
    # "podplywac" pod kursor. Kotwiczymy wiec srodek aktualnego widoku tablicy.
    rel_x = 0.5
    rel_y = 0.5
    anchor_x = image_left + (current_display_w / 2.0)
    anchor_y = image_top + (current_display_h / 2.0)

    return {
        "plate_id": str(state.get("plate_id", "") or ""),
        "rel_x": float(max(0.0, min(1.0, rel_x))),
        "rel_y": float(max(0.0, min(1.0, rel_y))),
        "anchor_x": float(anchor_x),
        "anchor_y": float(anchor_y),
    }


def _apply_preview_zoom_view(host, target_zoom: float, anchor: dict | None) -> bool:
    self = host
    state = dict(getattr(self, "_preview_render_state", None) or {})
    if not state:
        return False
    safe_zoom_max = _get_preview_safe_zoom_max(self, state)
    target_zoom = max(float(getattr(self, "_preview_zoom_min", 0.72) or 0.72), min(safe_zoom_max, float(target_zoom)))

    if not isinstance(anchor, dict):
        image_left = float(state.get("image_left", 0.0) or 0.0)
        image_top = float(state.get("image_top", 0.0) or 0.0)
        image_right = float(state.get("image_right", image_left) or image_left)
        image_bottom = float(state.get("image_bottom", image_top) or image_top)
        anchor = {
            "rel_x": 0.5,
            "rel_y": 0.5,
            "anchor_x": image_left + (max(1.0, image_right - image_left) / 2.0),
            "anchor_y": image_top + (max(1.0, image_bottom - image_top) / 2.0),
            "plate_id": str(state.get("plate_id", "") or ""),
        }

    try:
        orig_w = max(1.0, float(state.get("orig_w", 1.0)))
        orig_h = max(1.0, float(state.get("orig_h", 1.0)))
        fit_scale = max(0.001, float(state.get("fit_scale", state.get("scale", 1.0))))
        fit_x_off = float(state.get("fit_x_off", state.get("image_left", 0.0)))
        fit_y_off = float(state.get("fit_y_off", state.get("image_top", 0.0)))
        fit_new_w = max(1.0, float(state.get("fit_new_w", max(1, int(orig_w * fit_scale)))))
        fit_new_h = max(1.0, float(state.get("fit_new_h", max(1, int(orig_h * fit_scale)))))
        next_scale = fit_scale * float(target_zoom)
        next_w = max(1.0, float(max(1, int(orig_w * next_scale))))
        next_h = max(1.0, float(max(1, int(orig_h * next_scale))))
        base_x_off = fit_x_off - ((next_w - fit_new_w) / 2.0)
        base_y_off = fit_y_off - ((next_h - fit_new_h) / 2.0)
        canvas_w = float(getattr(self.preview_canvas, "winfo_width", lambda: 0)() or 0)
        canvas_h = float(getattr(self.preview_canvas, "winfo_height", lambda: 0)() or 0)
        rel_x = float(anchor.get("rel_x", 0.5))
        rel_y = float(anchor.get("rel_y", 0.5))
        anchor_x = float(anchor.get("anchor_x", canvas_w / 2.0))
        anchor_y = float(anchor.get("anchor_y", canvas_h / 2.0))
    except Exception:
        return False

    self._preview_zoom_level = float(target_zoom)
    self._preview_pan_x = anchor_x - base_x_off - (rel_x * next_w)
    self._preview_pan_y = anchor_y - base_y_off - (rel_y * next_h)
    self._preview_zoom_pending_state = {
        "plate_id": str(state.get("plate_id", "") or ""),
        "zoom": float(self._preview_zoom_level),
        "pan_x": float(self._preview_pan_x),
        "pan_y": float(self._preview_pan_y),
    }
    if _render_preview_zoom_frame(self):
        return True
    self._preview_fast_select_render = True
    self._on_preview_select(None)
    return False


def _clear_preview_zoom_animation(host) -> None:
    canvas = getattr(host, "preview_canvas", None)
    after_id = getattr(host, "_preview_zoom_anim_after_id", None)
    if after_id and canvas is not None:
        try:
            canvas.after_cancel(after_id)
        except Exception:
            pass
    host._preview_zoom_anim_after_id = None
    host._preview_zoom_velocity = 0.0
    host._preview_zoom_last_ts = None
    host._preview_zoom_last_input_ts = None


def _schedule_preview_zoom_animation(host, delay_ms: int = 0) -> None:
    canvas = getattr(host, "preview_canvas", None)
    if canvas is None:
        return
    if getattr(host, "_preview_zoom_anim_after_id", None):
        return
    try:
        host._preview_zoom_anim_after_id = canvas.after(
            max(1, int(delay_ms)),
            lambda: _animate_preview_zoom_step(host),
        )
    except Exception:
        host._preview_zoom_anim_after_id = None


def _settle_preview_zoom_without_full_redraw(host) -> None:
    canvas = getattr(host, "preview_canvas", None)
    if canvas is None:
        return
    try:
        pending_detail_after = getattr(host, "_preview_detail_render_after_id", None)
        if pending_detail_after:
            host.frame.after_cancel(pending_detail_after)
            host._preview_detail_render_after_id = None
    except Exception:
        pass
    try:
        canvas.tag_lower("preview_plate_image")
        canvas.tag_raise("preview_plate_status_frame")
        canvas.tag_raise("preview_char")
        canvas.tag_raise("preview_badge")
        canvas.tag_raise("preview_layout_separator")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    try:
        host._apply_preview_badge_selection_style()
    except Exception:
        pass
    try:
        host._apply_preview_canvas_cursor()
        host._refresh_preview_editor_toolbar()
    except Exception:
        pass


def _animate_preview_zoom_step(host) -> None:
    self = host
    self._preview_zoom_anim_after_id = None
    state = dict(getattr(self, "_preview_render_state", None) or {})
    if not state or _preview_zoom_interaction_blocked(self):
        _clear_preview_zoom_animation(self)
        return

    anchor = getattr(self, "_preview_zoom_anchor", None)
    if isinstance(anchor, dict):
        anchor_pid = str(anchor.get("plate_id", "") or "")
        current_pid = str(state.get("plate_id", "") or "")
        if anchor_pid and current_pid and anchor_pid != current_pid:
            _clear_preview_zoom_animation(self)
            return

    now = time.perf_counter()
    last_ts = getattr(self, "_preview_zoom_last_ts", None)
    if not isinstance(last_ts, (int, float)):
        dt = 0.024
    else:
        dt = max(0.010, min(0.045, float(now) - float(last_ts)))
    self._preview_zoom_last_ts = float(now)

    current_zoom = float(getattr(self, "_preview_zoom_level", 1.0) or 1.0)
    safe_zoom_max = _get_preview_safe_zoom_max(self, state)
    target_zoom = max(
        float(getattr(self, "_preview_zoom_min", 0.72) or 0.72),
        min(safe_zoom_max, float(getattr(self, "_preview_zoom_target", current_zoom) or current_zoom)),
    )
    velocity = float(getattr(self, "_preview_zoom_velocity", 0.0) or 0.0)
    stiffness = float(getattr(self, "_preview_zoom_stiffness", 34.0) or 34.0)
    damping = float(getattr(self, "_preview_zoom_damping", 10.5) or 10.5)
    last_input_ts = getattr(self, "_preview_zoom_last_input_ts", None)
    try:
        input_age = float(now) - float(last_input_ts) if isinstance(last_input_ts, (int, float)) else 0.0
    except Exception:
        input_age = 0.0
    max_tail_s = max(0.15, float(getattr(self, "_preview_zoom_max_tail_s", 1.0) or 1.0))

    velocity += (target_zoom - current_zoom) * stiffness * dt
    velocity *= math.exp(-damping * dt)
    next_zoom = current_zoom + (velocity * dt)
    min_zoom = float(getattr(self, "_preview_zoom_min", 0.72) or 0.72)
    if next_zoom <= min_zoom:
        next_zoom = min_zoom
        velocity = 0.0
    elif next_zoom >= safe_zoom_max:
        next_zoom = safe_zoom_max
        velocity = 0.0

    close_enough = abs(target_zoom - next_zoom) < 0.010 and abs(velocity) < 0.075
    if input_age >= max_tail_s:
        close_enough = True
    if close_enough:
        next_zoom = target_zoom
        velocity = 0.0

    self._preview_zoom_velocity = float(velocity)
    self._preview_zoom_target = float(target_zoom)
    if not _apply_preview_zoom_view(self, next_zoom, anchor):
        _clear_preview_zoom_animation(self)
        return

    if close_enough:
        self._preview_zoom_anim_after_id = None
        self._preview_zoom_last_ts = None
        self._preview_zoom_velocity = 0.0
        self._preview_zoom_pending_state = None
        self._preview_fast_select_render = False
        _settle_preview_zoom_without_full_redraw(self)
        return

    frame_ms = int(getattr(self, "_preview_zoom_frame_ms", 22) or 22)
    _schedule_preview_zoom_animation(self, delay_ms=frame_ms)


def on_preview_canvas_mousewheel(host, event):
    self = host
    if not getattr(self, "_preview_render_state", None):
        return
    if _preview_zoom_interaction_blocked(self):
        return "break"

    delta = 0
    if hasattr(event, "delta") and event.delta:
        delta = int(event.delta)
    elif hasattr(event, "num"):
        if int(event.num) == 4:
            delta = 120
        elif int(event.num) == 5:
            delta = -120

    if delta == 0:
        return

    pending_detail = getattr(self, "_preview_detail_render_after_id", None)
    if pending_detail:
        try:
            self.frame.after_cancel(pending_detail)
        except Exception:
            pass
        self._preview_detail_render_after_id = None

    state = dict(self._preview_render_state)
    safe_zoom_max = _get_preview_safe_zoom_max(self, state)
    current_zoom = float(getattr(self, "_preview_zoom_level", 1.0) or 1.0)
    target_base = current_zoom
    try:
        impulse = max(-4.0, min(4.0, float(delta) / 120.0))
    except Exception:
        impulse = 1.0 if delta > 0 else -1.0
    target_zoom = target_base * (float(getattr(self, "_preview_zoom_step", 1.16) or 1.16) ** impulse)
    target_zoom = max(float(getattr(self, "_preview_zoom_min", 0.72) or 0.72), min(safe_zoom_max, target_zoom))
    if abs(target_zoom - current_zoom) < 1e-6 and abs(float(getattr(self, "_preview_zoom_velocity", 0.0) or 0.0)) < 1e-6:
        return "break"

    previous_after_id = getattr(self, "_preview_zoom_render_after_id", None)
    if previous_after_id:
        try:
            self.preview_canvas.after_cancel(previous_after_id)
        except Exception:
            pass
        self._preview_zoom_render_after_id = None

    _clear_preview_zoom_animation(self)
    anchor = _capture_preview_zoom_anchor(self, event, state)
    self._preview_zoom_anchor = anchor
    self._preview_zoom_target = float(target_zoom)
    self._preview_zoom_last_input_ts = time.perf_counter()
    self._preview_zoom_velocity = 0.0
    self._preview_zoom_last_ts = None
    if _apply_preview_zoom_view(self, target_zoom, anchor):
        self._preview_zoom_pending_state = None
        self._preview_fast_select_render = False
        _settle_preview_zoom_without_full_redraw(self)
    else:
        _clear_preview_zoom_animation(self)
    return "break"


def _render_preview_zoom_frame(host) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    state = dict(getattr(self, "_preview_render_state", None) or {})
    if canvas is None or not state:
        return False

    pid = str(state.get("plate_id", "") or getattr(self, "_preview_active_pid", "") or "").strip()
    if not pid:
        return False
    try:
        img_path = Path(str(self.preview_dir_var.get() or "").strip()) / "images" / f"{pid}.jpg"
    except Exception:
        return False
    if not img_path.exists():
        return False

    try:
        pil_img, orig_w, orig_h, source_image_key = _get_cached_preview_source_image(self, img_path)
        fit_scale = max(0.001, float(state.get("fit_scale", state.get("scale", 1.0)) or 1.0))
        fit_new_w = max(1.0, float(state.get("fit_new_w", max(1, int(orig_w * fit_scale))) or 1.0))
        fit_new_h = max(1.0, float(state.get("fit_new_h", max(1, int(orig_h * fit_scale))) or 1.0))
        fit_x_off = float(state.get("fit_x_off", 0.0) or 0.0)
        fit_y_off = float(state.get("fit_y_off", 0.0) or 0.0)
        render_scale = fit_scale * float(getattr(self, "_preview_zoom_level", 1.0) or 1.0)
        new_w = max(1, int(float(orig_w) * render_scale))
        new_h = max(1, int(float(orig_h) * render_scale))
        photo = _get_cached_preview_photo(self, source_image_key, pil_img, new_w, new_h)
        canvas_w = max(50, int(canvas.winfo_width() or 50))
        canvas_h = max(50, int(canvas.winfo_height() or 50))
        base_x_off = fit_x_off - ((float(new_w) - fit_new_w) / 2.0)
        base_y_off = fit_y_off - ((float(new_h) - fit_new_h) / 2.0)
        x_off = base_x_off + float(getattr(self, "_preview_pan_x", 0.0) or 0.0)
        y_off = base_y_off + float(getattr(self, "_preview_pan_y", 0.0) or 0.0)
        x_off, y_off = self._clamp_preview_image_position(
            x_off,
            y_off,
            image_w=float(new_w),
            image_h=float(new_h),
            canvas_w=float(canvas_w),
            canvas_h=float(canvas_h),
            top_reserved=float(state.get("top_reserved", 0.0) or 0.0),
            bottom_reserved=float(state.get("bottom_reserved", 0.0) or 0.0),
            preferred_x=base_x_off,
            preferred_y=base_y_off,
        )
    except Exception:
        return False

    self._current_photo = photo
    self._preview_pan_x = float(x_off - base_x_off)
    self._preview_pan_y = float(y_off - base_y_off)
    state.update(
        {
            "plate_id": pid,
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "scale": float(render_scale),
            "image_left": float(x_off),
            "image_top": float(y_off),
            "image_right": float(x_off + float(new_w)),
            "image_bottom": float(y_off + float(new_h)),
        }
    )
    self._preview_render_state = state

    for tag in (
        "preview_plate_image",
        "preview_char",
        "preview_fast_detail",
        "preview_badge",
        "preview_plate_status_frame",
        "preview_layout_separator",
        "preview_canvas_caption",
    ):
        try:
            canvas.delete(tag)
        except Exception:
            pass

    try:
        canvas.create_image(
            float(x_off),
            float(y_off),
            anchor=tk.NW,
            image=self._current_photo,
            tags=("preview_plate_image", "preview_movable_plate"),
        )
    except Exception:
        return False

    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        data = {}

    try:
        self._redraw_preview_character_overlays_light()
    except Exception:
        pass

    try:
        canvas.create_text(
            10,
            max(14, canvas_h - 10),
            text="Podgląd tablicy",
            fill=getattr(self.app, "palette", {}).get("muted", "#b0b0b0"),
            font=("Segoe UI", 8, "bold"),
            anchor=tk.SW,
            tags=("preview_canvas_caption",),
        )
        canvas.tag_lower("preview_plate_image")
        canvas.tag_raise("preview_plate_status_frame")
        canvas.tag_raise("preview_char")
        canvas.tag_raise("preview_badge")
        canvas.tag_raise("preview_layout_separator")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    return True


def _schedule_preview_zoom_details(host) -> None:
    self = host
    try:
        pending_after = getattr(self, "_preview_detail_render_after_id", None)
        if pending_after:
            self.frame.after_cancel(pending_after)
            self._preview_detail_render_after_id = None
    except Exception:
        pass
    active_pid = str(getattr(self, "_preview_active_pid", "") or "")
    try:
        generation = int(getattr(self, "_preview_select_generation", 0) or 0)
    except Exception:
        generation = 0

    def _render_zoom_details_once():
        self._preview_detail_render_after_id = None
        if str(getattr(self, "_preview_active_pid", "") or "") != active_pid:
            return
        try:
            if int(getattr(self, "_preview_select_generation", 0) or 0) != int(generation):
                return
        except Exception:
            return
        if getattr(self, "_preview_zoom_render_after_id", None):
            return
        if (
            getattr(self, "_preview_badge_drag_state", None) is not None
            or getattr(self, "_preview_layout_separator_drag_state", None) is not None
            or getattr(self, "_preview_pan_drag_state", None) is not None
            or getattr(self, "_preview_char_drag_state", None) is not None
            or getattr(self, "_preview_char_add_state", None) is not None
        ):
            return
        try:
            self._draw_preview_fast_render_details(active_pid)
        except Exception:
            pass

    try:
        self._preview_detail_render_after_id = self.frame.after(420, _render_zoom_details_once)
    except Exception:
        self._preview_detail_render_after_id = None


def start_preview_character_box_drag(host, char_idx, mode, event, *, handle_name=None, cursor="fleur") -> bool:
    self = host
    perf_start = time.perf_counter()
    layout_ms = copy_ms = bind_ms = 0.0
    chars = self._get_preview_active_character_records(create=False)
    try:
        char_idx = int(char_idx)
    except Exception:
        return False
    if not (0 <= char_idx < len(chars)):
        return False

    rec = chars[char_idx]
    bbox = self._char_record_bbox(rec)
    if not bbox:
        return False
    phase_start = time.perf_counter()
    active_data = self._get_preview_active_data(create=False)
    layout_row = None
    if isinstance(rec, dict):
        try:
            raw_row = int(rec.get("reading_row", 0) or 0)
            layout_row = raw_row if raw_row in (1, 2) else None
        except Exception:
            layout_row = None
    if layout_row is None:
        layout_row = self._get_preview_row_for_bbox(bbox, data=active_data)
    layout_ms = (time.perf_counter() - phase_start) * 1000.0
    geometry_inherit_state = None
    if bool(getattr(self, "_preview_char_geometry_inherit_down", False)):
        leader_idx, leader_row = _get_preview_char_geometry_leader_index(self, row_hint=layout_row)
        if leader_idx is not None and int(leader_idx) != int(char_idx):
            self._select_preview_character_box(
                int(leader_idx),
                activate_label=False,
                status_message=(
                    f"G: liderem jest pierwszy box rzędu {int(leader_row or layout_row or 1)}. "
                    "Przeciągnij uchwyt lidera, aby rozesłać geometrię na resztę rzędu."
                ),
            )
            return False
        if leader_idx is not None:
            geometry_inherit_state = _build_preview_char_geometry_inheritance_state(
                self,
                int(leader_idx),
                leader_row if leader_row in (1, 2) else layout_row,
            )
    phase_start = time.perf_counter()
    try:
        original_record = dict(rec) if isinstance(rec, dict) else {}
        original_record["bbox"] = list(bbox)
    except Exception:
        original_record = dict(rec) if isinstance(rec, dict) else {}
    copy_ms = (time.perf_counter() - phase_start) * 1000.0

    img_x, img_y = self._preview_canvas_to_image_point(event.x, event.y)
    self._preview_char_label_active_index = None
    self._preview_char_selected_index = int(char_idx)
    try:
        trace_id = int(getattr(self, "_preview_char_edit_trace_seq", 0) or 0) + 1
    except Exception:
        trace_id = 1
    self._preview_char_edit_trace_seq = trace_id
    drag_state = {
        "trace_id": int(trace_id),
        "index": int(char_idx),
        "mode": str(mode),
        "start_img_x": float(img_x),
        "start_img_y": float(img_y),
        "start_bbox": list(bbox),
        "dirty": False,
        "history_pushed": False,
        "original_record": original_record,
        "last_visual_at": 0.0,
        "layout_row": layout_row,
    }
    if isinstance(geometry_inherit_state, dict):
        drag_state["geometry_inherit"] = geometry_inherit_state
    if handle_name:
        drag_state["handle"] = str(handle_name)
    self._preview_char_drag_state = drag_state
    trace_fields = _preview_char_record_trace_fields(self, rec, data=active_data, fallback_index=char_idx)
    try:
        self._log_preview_edit_flow(
            "char_drag_start",
            trace=trace_id,
            idx=char_idx,
            mode=str(mode),
            handle=str(handle_name or "-"),
            **trace_fields,
        )
    except Exception:
        pass
    phase_start = time.perf_counter()
    self._bind_preview_char_drag_session()
    bind_ms = (time.perf_counter() - phase_start) * 1000.0
    try:
        self.preview_canvas.configure(cursor=str(cursor))
    except Exception:
        pass
    total_ms = (time.perf_counter() - perf_start) * 1000.0
    if total_ms >= 55.0:
        logger.info(
            "[Z3/PZ2 PERF] char_drag_start total=%.1fms mode=%s handle=%s phases=[layout=%.1fms, copy=%.1fms, bind=%.1fms]",
            total_ms,
            str(mode),
            str(handle_name or ""),
            layout_ms,
            copy_ms,
            bind_ms,
        )
    return True


def on_preview_canvas_secondary_press(host, event=None):
    self = host
    self._focus_preview_canvas()
    if event is None or not getattr(self, "_preview_render_state", None):
        return None
    if self._extract_preview_action_from_current_item() == "toggle_plate_rows":
        from .z3_preview_layout_control import hide_layout_tip
        hide_layout_tip(self)
        return self._cycle_preview_plate_layout_override(event)

    box_hit = self._find_preview_character_box_hit(event.x, event.y)
    if box_hit is None:
        if getattr(self, "_preview_char_label_active_index", None) is not None:
            self._preview_char_label_active_index = None
            self._update_preview_edit_status("Zamknieto aktywne pole znaku.", tone="muted")
            self._on_preview_select(None)
            return "break"
        return None

    selected_idx = getattr(self, "_preview_char_selected_index", None)
    if selected_idx is None or int(selected_idx) != int(box_hit):
        self._update_preview_edit_status(
            "PPM usuwa tylko aktywny box. Najedź kursorem na wybrany box i naciśnij S, aby go zaznaczyć.",
            tone="warning",
        )
        self._on_preview_select(None)
        return "break"

    self._preview_char_label_active_index = None
    self._preview_char_hover_label_index = None
    self._refresh_preview_editor_toolbar()
    return self._delete_selected_preview_char_box(event)


def finalize_preview_char_add_state(host) -> str:
    self = host
    char_add_state = getattr(self, "_preview_char_add_state", None)
    if not isinstance(char_add_state, dict):
        return "break"

    click_draw = bool(char_add_state.get("click_draw"))
    self._preview_char_add_state = None
    if click_draw:
        self._preview_char_add_click_armed = False
        self._preview_char_add_modifier_down = False
        self._preview_char_add_mode = False

    try:
        self.preview_canvas.configure(cursor=("crosshair" if self._preview_char_add_requested() else "arrow"))
    except Exception:
        pass

    bbox = self._normalize_preview_char_bbox(char_add_state.get("bbox"))
    if bbox is None:
        if click_draw:
            self._preview_char_add_click_armed = True
        try:
            self.preview_canvas.delete("preview_char_add_preview")
        except Exception:
            pass
        return "break"

    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if width < 6.0 or height < 6.0:
        if click_draw:
            self._preview_char_add_click_armed = True
        self._set_preview_box_info("Nowy box jest zbyt mały. Spróbuj ponownie.", "warning")
        self._update_preview_edit_status("Nowy box jest zbyt mały. Wskaż pierwszy narożnik ponownie.", tone="warning")
        self._apply_preview_canvas_cursor("crosshair" if click_draw else None)
        try:
            self.preview_canvas.delete("preview_char_add_preview")
        except Exception:
            pass
        return "break"

    chars = self._get_preview_active_character_records(create=True)
    self._push_preview_history_snapshot()
    new_record = {
        "character": "",
        "bbox": bbox,
        "confidence": 1.0,
        "method": "manual",
        "source_tag": "manual",
        "source_kind": "local_manual",
        "box_source": "manual_box",
        "sign_source": "",
    }
    chars.append(new_record)
    self._mark_preview_char_record_manual(new_record, box=True, sign=False)
    self._preview_char_edit_mode = True
    self._preview_char_label_active_index = None
    self._preview_char_hover_label_index = None
    try:
        self.preview_canvas.delete("preview_char_add_preview")
    except Exception:
        pass
    self._persist_active_preview_characters(
        selected_record=new_record,
        success_message="Dodano nowy box znaku do metadata.json.",
        render_preview=False,
        save_immediately=False,
        save_delay_ms=650,
        refresh_row=True,
        light_redraw_indices=None,
    )
    if click_draw:
        status_message = "Dodano nowy box. Tryb rysowania z D został wyłączony."
    else:
        status_message = "Dodano nowy box. Pozostaje zaznaczony do dalszej korekty, ale nie włącza pola wpisywania znaku."
    self._update_preview_edit_status(status_message, tone="success")
    return "break"
