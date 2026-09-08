#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 preview editor selection, toolbar and shortcut helpers."""

from __future__ import annotations

import tkinter as tk
import time
from pathlib import Path
from tkinter import messagebox

from ..config import logger


def _get_preview_selected_char_record(self):
    chars = self._get_preview_active_character_records(create=False)
    idx = getattr(self, "_preview_char_selected_index", None)
    if idx is None:
        return None, None
    try:
        idx = int(idx)
    except Exception:
        self._preview_char_selected_index = None
        return None, None
    if not (0 <= idx < len(chars)):
        self._preview_char_selected_index = None
        return None, None
    return idx, chars[idx]

def _is_preview_char_record_selected(self, record, *, fallback_index: int | None = None) -> bool:
    selected_idx, selected_record = self._get_preview_selected_char_record()
    if selected_record is not None:
        return selected_record is record
    if selected_idx is None or fallback_index is None:
        return False
    try:
        return int(selected_idx) == int(fallback_index)
    except Exception:
        return False

def _preview_char_add_requested(self) -> bool:
    return bool(
        getattr(self, "_preview_char_add_state", None) is not None
        or getattr(self, "_preview_char_add_mode", False)
        or getattr(self, "_preview_char_add_modifier_down", False)
        or getattr(self, "_preview_char_add_click_armed", False)
    )

def _find_preview_char_record_index(chars, target_record) -> int | None:
    if not isinstance(chars, list) or target_record is None:
        return None
    for idx, rec in enumerate(chars):
        if rec is target_record:
            return idx
    return None

def _select_hovered_preview_char_box(self, event=None):
    hover_idx = getattr(self, "_preview_char_hover_index", None)
    if hover_idx is None:
        selected_idx, _selected_rec = self._get_preview_selected_char_record()
        if selected_idx is not None:
            affected_indices = {int(selected_idx)}
            self._preview_char_selected_index = None
            self._preview_char_label_active_index = None
            self._preview_char_hover_label_index = None
            self._refresh_preview_editor_toolbar()
            self._update_preview_edit_status("Odznaczono aktywny box znaku.", tone="muted")
            if not self._refresh_preview_character_selection_visual(affected_indices):
                self._on_preview_select(None)
            return "break"
        self._update_preview_edit_status("Najedź kursorem na box i naciśnij S, aby go zaznaczyć.", tone="warning")
        return "break"
    selected_idx, _selected_rec = self._get_preview_selected_char_record()
    if selected_idx is not None and int(selected_idx) == int(hover_idx):
        affected_indices = {int(selected_idx), int(hover_idx)}
        self._preview_char_selected_index = None
        self._preview_char_label_active_index = None
        self._preview_char_hover_label_index = None
        self._refresh_preview_editor_toolbar()
        self._update_preview_edit_status("Odznaczono hoverowany box znaku.", tone="muted")
        if not self._refresh_preview_character_selection_visual(affected_indices):
            self._on_preview_select(None)
        return "break"
    return self._select_preview_character_box(
        int(hover_idx),
        activate_label=False,
        status_message="Zaznaczono hoverowany box. PPM usuwa aktywny box, a S ponownie go odznacza.",
    )

def _cycle_preview_character_selection(self, step: int, *, activate_label: bool = False):
    chars = self._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not chars:
        self._update_preview_edit_status("Brak boxów znaków do przełączenia.", tone="warning")
        return "break"

    ordered_chars = self._sort_character_records_by_x(list(chars))
    selected_idx, selected_rec = self._get_preview_selected_char_record()
    ordered_pos = None
    if selected_rec is not None:
        for idx, rec in enumerate(ordered_chars):
            if rec is selected_rec:
                ordered_pos = idx
                break

    if ordered_pos is None:
        target_pos = 0 if int(step) >= 0 else (len(ordered_chars) - 1)
    else:
        target_pos = (int(ordered_pos) + int(step)) % len(ordered_chars)

    target_rec = ordered_chars[int(target_pos)]
    target_idx = self._find_preview_char_record_index(chars, target_rec)
    if target_idx is None:
        if selected_idx is not None:
            target_idx = max(0, min(int(selected_idx), len(chars) - 1))
        else:
            target_idx = 0

    return self._select_preview_character_box(
        int(target_idx),
        activate_label=(activate_label or bool(getattr(self, "_preview_char_label_mode", False))),
        status_message=(
            f"Wybrano box {int(target_pos) + 1}/{len(ordered_chars)} według rosnącego X. "
            + (
                "Wpisz znak z klawiatury, aby nadpisać etykietę."
                if bool(getattr(self, "_preview_char_label_mode", False))
                else "Strzałki lewo/prawo przełączają kolejne boxy."
            )
        ),
    )

def _cycle_preview_character_row(self, direction: int, *, activate_label: bool = False):
    chars = self._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not chars:
        self._update_preview_edit_status("Brak boxów znaków do przełączenia.", tone="warning")
        return "break"

    data = self._get_preview_active_data(create=False)
    selected_idx, selected_rec = self._get_preview_selected_char_record()
    active_idx = getattr(self, "_preview_char_label_active_index", None)
    try:
        active_idx = int(active_idx) if active_idx is not None else None
    except Exception:
        active_idx = None
    if active_idx is not None and 0 <= active_idx < len(chars):
        selected_idx = active_idx
        selected_rec = chars[active_idx]
    if selected_rec is None:
        selected_idx = self._resolve_preview_char_label_entry_index()
        if selected_idx is not None and 0 <= int(selected_idx) < len(chars):
            selected_idx = int(selected_idx)
            selected_rec = chars[selected_idx]
    if selected_rec is None:
        return self._cycle_preview_character_selection(1, activate_label=activate_label)

    def _record_row(rec) -> int:
        try:
            row = int(rec.get("reading_row", 0) if isinstance(rec, dict) else getattr(rec, "reading_row", 0) or 0)
        except Exception:
            row = 0
        if row in (1, 2):
            return row
        try:
            bbox = self._char_record_bbox(rec)
            row = int(self._get_preview_row_for_bbox(bbox, data=data) or 0)
        except Exception:
            row = 0
        return row if row in (1, 2) else 0

    current_row = _record_row(selected_rec)
    target_row = 1 if int(direction) < 0 else 2
    if current_row == target_row:
        self._update_preview_edit_status(
            "Jesteś już w górnym rzędzie." if target_row == 1 else "Jesteś już w dolnym rzędzie.",
            tone="muted",
        )
        return "break"

    current_bbox = self._char_record_bbox(selected_rec)
    if current_bbox:
        current_cx = (float(current_bbox[0]) + float(current_bbox[2])) / 2.0
    else:
        current_cx = 0.0

    candidates = []
    for idx, rec in enumerate(chars):
        if _record_row(rec) != target_row:
            continue
        bbox = self._char_record_bbox(rec)
        if not bbox:
            continue
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        candidates.append((abs(cx - current_cx), cx, idx, rec))

    if not candidates:
        self._update_preview_edit_status(
            "Nie znaleziono boxów w górnym rzędzie." if target_row == 1 else "Nie znaleziono boxów w dolnym rzędzie.",
            tone="warning",
        )
        return "break"

    _dist, _cx, target_idx, _rec = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return self._select_preview_character_box(
        int(target_idx),
        activate_label=(activate_label or bool(getattr(self, "_preview_char_label_mode", False))),
        status_message=(
            "Przełączono na najbliższy box w górnym rzędzie."
            if target_row == 1
            else "Przełączono na najbliższy box w dolnym rzędzie."
        ),
    )

def _get_preview_char_handle_radius(self) -> float:
    state = getattr(self, "_preview_render_state", None) or {}
    scale = float(state.get("scale", 1.0) or 1.0)
    return max(7.0, min(13.0, 5.5 + (scale * 0.16)))

def _get_preview_char_move_handle_radius(self) -> float:
    state = getattr(self, "_preview_render_state", None) or {}
    scale = float(state.get("scale", 1.0) or 1.0)
    return max(11.0, min(20.0, 9.0 + (scale * 0.24)))

def _find_preview_character_box_hit(self, canvas_x: float, canvas_y: float):
    chars = self._get_preview_active_character_records(create=False)
    hits = []
    for idx in range(len(chars) - 1, -1, -1):
        bbox = self._char_record_bbox(chars[idx])
        if not bbox:
            continue
        cx1, cy1 = self._preview_image_to_canvas_point(bbox[0], bbox[1])
        cx2, cy2 = self._preview_image_to_canvas_point(bbox[2], bbox[3])
        pad = 3.0
        if (cx1 - pad) <= float(canvas_x) <= (cx2 + pad) and (cy1 - pad) <= float(canvas_y) <= (cy2 + pad):
            area = max(1.0, abs(float(cx2) - float(cx1)) * abs(float(cy2) - float(cy1)))
            hits.append((area, -int(idx), int(idx)))
    if not hits:
        return None
    return sorted(hits, key=lambda item: (item[0], item[1]))[0][2]

def _find_preview_character_handle_hit(self, canvas_x: float, canvas_y: float):
    idx, rec = self._get_preview_selected_char_record()
    if rec is None:
        return None

    bbox = self._char_record_bbox(rec)
    if not bbox:
        return None

    cx1, cy1 = self._preview_image_to_canvas_point(bbox[0], bbox[1])
    cx2, cy2 = self._preview_image_to_canvas_point(bbox[2], bbox[3])
    radius = self._get_preview_char_handle_radius()
    handles = {
        "nw": (cx1, cy1),
        "ne": (cx2, cy1),
        "sw": (cx1, cy2),
        "se": (cx2, cy2),
    }
    best_hit = None
    for handle_name, (hx, hy) in handles.items():
        dx = float(canvas_x) - float(hx)
        dy = float(canvas_y) - float(hy)
        dist_sq = (dx * dx) + (dy * dy)
        if dist_sq <= (radius * radius):
            candidate = (dist_sq, idx, handle_name)
            if best_hit is None or candidate[0] < best_hit[0]:
                best_hit = candidate
    if best_hit is None:
        return None
    return best_hit[1], best_hit[2]

def _find_preview_character_move_handle_hit(self, canvas_x: float, canvas_y: float):
    idx, rec = self._get_preview_selected_char_record()
    if rec is None:
        return None

    bbox = self._char_record_bbox(rec)
    if not bbox:
        return None

    cx1, cy1 = self._preview_image_to_canvas_point(bbox[0], bbox[1])
    cx2, cy2 = self._preview_image_to_canvas_point(bbox[2], bbox[3])
    center_x = (float(cx1) + float(cx2)) / 2.0
    center_y = (float(cy1) + float(cy2)) / 2.0
    radius = self._get_preview_char_move_handle_radius()
    dx = float(canvas_x) - center_x
    dy = float(canvas_y) - center_y
    if ((dx * dx) + (dy * dy)) <= (radius * radius):
        return idx
    return None

def _find_preview_character_grip_hit(self, canvas_x: float, canvas_y: float):
    corner_hit = self._find_preview_character_handle_hit(canvas_x, canvas_y)
    if corner_hit is not None:
        char_idx, handle_name = corner_hit
        return {
            "index": int(char_idx),
            "kind": "corner",
            "handle": str(handle_name),
            "key": f"corner:{handle_name}",
        }

    move_hit = self._find_preview_character_move_handle_hit(canvas_x, canvas_y)
    if move_hit is not None:
        return {
            "index": int(move_hit),
            "kind": "move",
            "handle": "center",
            "key": "move:center",
        }
    return None

def _get_preview_character_canvas_tag(box_source: str, box_idx: int) -> str:
    return f"preview_char::{box_source}:{int(box_idx)}"

def _get_preview_character_record_canvas_tag(record) -> str:
    return f"preview_char_record::{id(record)}"

def _clear_preview_character_drag_visual(self):
    canvas = getattr(self, "preview_canvas", None)
    drag_state = getattr(self, "_preview_char_drag_state", None)
    if canvas is not None:
        record_tag = drag_state.get("visual_record_tag") if isinstance(drag_state, dict) else None
        if record_tag and not bool(drag_state.get("dirty")):
            try:
                canvas.itemconfigure(record_tag, state="normal")
            except Exception:
                pass
        try:
            canvas.delete("preview_char_drag_preview")
        except Exception:
            pass
    if isinstance(drag_state, dict):
        drag_state.pop("visual_ids", None)
        drag_state.pop("visual_record_tag", None)

def _refresh_preview_character_selection_visual(self, indices=None) -> bool:
    if not getattr(self, "_preview_active_pid", None):
        return False
    if not getattr(self, "_preview_render_state", None):
        return False
    chars = self._get_preview_active_character_records(create=False)
    if not isinstance(chars, list):
        return False

    normalized_indices = []
    if indices is None:
        normalized_indices = list(range(len(chars)))
    else:
        for raw_idx in set(indices or []):
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if 0 <= idx < len(chars):
                normalized_indices.append(idx)

    if not normalized_indices:
        return True

    runtime_map = getattr(self, "_preview_char_runtime", None)
    if isinstance(runtime_map, dict):
        runtime_mismatch = False
        for idx in normalized_indices:
            payload = runtime_map.get(f"FINAL:{int(idx)}")
            if not isinstance(payload, dict):
                continue
            record_id = payload.get("record_id")
            if record_id is not None and int(record_id) != id(chars[int(idx)]):
                runtime_mismatch = True
                break
        if runtime_mismatch:
            try:
                return bool(self._redraw_preview_character_overlays_light())
            except Exception:
                return False

    if not bool(getattr(self, "_preview_char_label_mode", False)) and getattr(self, "_preview_char_label_active_index", None) is None:
        fast_update = getattr(self, "_update_preview_character_selection_items_fast", None)
        if callable(fast_update):
            try:
                if fast_update(normalized_indices):
                    return True
            except Exception:
                pass

    redrawn = False
    for idx in sorted(set(normalized_indices)):
        redrawn = self._redraw_preview_character_overlay_only(int(idx)) or redrawn

    if redrawn:
        try:
            self._refresh_preview_editor_toolbar()
        except Exception:
            pass
        try:
            self._focus_preview_canvas()
        except Exception:
            pass
    return bool(redrawn)

def _refresh_preview_editor_toolbar(self):
    hidden_controls = getattr(self, "preview_hidden_controls", None)
    if hidden_controls is not None:
        try:
            if not str(hidden_controls.winfo_manager()):
                return
        except Exception:
            pass

    active_data = self._get_preview_active_data(create=False)
    has_plate = isinstance(active_data, dict)
    selected_idx, _selected_rec = self._get_preview_selected_char_record()
    has_selection = selected_idx is not None
    try:
        total = len(getattr(self, "_listbox_pid_by_index", []) or [])
        current_idx = self._get_current_preview_list_index()
    except Exception:
        total = 0
        current_idx = None
    toolbar_signature = (
        bool(has_plate),
        bool(has_selection),
        bool(getattr(self, "_preview_char_edit_mode", False)),
        bool(getattr(self, "_preview_char_add_mode", False)),
        bool(getattr(self, "_preview_fullscreen_active", False)),
        int(total),
        int(current_idx) if current_idx is not None else -1,
    )
    if getattr(self, "_preview_editor_toolbar_signature", None) == toolbar_signature:
        return
    self._preview_editor_toolbar_signature = toolbar_signature

    button_specs = (
        ("preview_edit_toggle_btn", has_plate, "Edytuj boxy", "Edytowanie boxów"),
        ("preview_add_box_btn", has_plate, "Nowy box", "Rysowanie boxu"),
        ("preview_edit_char_btn", has_selection, "Zmień znak", "Zmień znak"),
        ("preview_delete_char_btn", has_selection, "Usuń box", "Usuń box"),
    )

    for attr_name, enabled, idle_text, active_text in button_specs:
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        is_active = bool(
            (attr_name == "preview_edit_toggle_btn" and getattr(self, "_preview_char_edit_mode", False))
            or (attr_name == "preview_add_box_btn" and getattr(self, "_preview_char_add_mode", False))
        )
        try:
            widget.configure(
                state=(tk.NORMAL if enabled else tk.DISABLED),
                text=(active_text if is_active else idle_text),
            )
        except Exception:
            pass

    try:
        self._update_preview_toolbar_state()
    except Exception:
        pass

def _set_preview_char_editor_modes(self, *, edit: bool | None = None, add: bool | None = None, message: str | None = None):
    if edit is not None:
        self._preview_char_edit_mode = bool(edit)
        if not self._preview_char_edit_mode:
            self._unbind_preview_char_drag_session()
            self._preview_char_drag_state = None
        else:
            self._preview_char_add_click_armed = False
            self._preview_char_add_modifier_down = False
            self._preview_char_add_state = None
            self._preview_char_label_mode = False
            self._preview_char_label_active_index = None
            self._preview_char_hover_label_index = None
    if add is not None:
        self._preview_char_add_mode = bool(add)
        if not self._preview_char_add_mode:
            self._preview_char_add_state = None
            self._preview_char_add_click_armed = False
            self._preview_char_add_modifier_down = False
        else:
            self._preview_char_add_click_armed = False
            self._preview_char_add_modifier_down = False
            self._preview_char_label_mode = False
            self._preview_char_label_active_index = None
            self._preview_char_hover_label_index = None
    if bool(getattr(self, "_preview_char_add_mode", False)):
        self._preview_char_label_active_index = None
    self._refresh_preview_editor_toolbar()
    if message:
        self._set_preview_box_info(message, "info")
        self._update_preview_edit_status(message, tone="info")
    else:
        self._update_preview_edit_status()
    self._focus_preview_canvas()
    self._on_preview_select(None)

def _toggle_preview_char_edit_mode(self, event=None):
    if self._get_preview_active_data(create=False) is None:
        self._set_preview_box_info("Najpierw wybierz tablicę z listy.", "warning")
        return "break"
    self._ensure_preview_final_box_mode(render_preview=False)
    self._set_preview_char_editor_modes(
        edit=(not bool(getattr(self, "_preview_char_edit_mode", False))),
        add=False,
        message="Tryb korekty boxów znaków działa na wyniku końcowym.",
    )
    return "break"

def _toggle_preview_char_add_mode(self, event=None):
    if self._get_preview_active_data(create=False) is None:
        self._set_preview_box_info("Najpierw wybierz tablicę z listy.", "warning")
        return "break"
    self._ensure_preview_final_box_mode(render_preview=False)
    self._set_preview_char_editor_modes(
        edit=False,
        add=(not bool(getattr(self, "_preview_char_add_mode", False))),
        message="Narysuj nowy box na znaku w podglądzie tablicy.",
    )
    return "break"

def _toggle_preview_char_label_mode(self, event=None):
    if self._get_preview_active_data(create=False) is None:
        self._set_preview_box_info("Najpierw wybierz tablicę z listy.", "warning")
        return "break"

    self._ensure_preview_final_box_mode(render_preview=False)
    next_state = not bool(getattr(self, "_preview_char_label_mode", False))
    self._preview_char_label_mode = next_state
    self._preview_char_edit_mode = False
    self._preview_char_add_mode = False
    self._preview_char_add_modifier_down = False
    self._preview_char_add_click_armed = False
    self._preview_char_add_state = None
    self._preview_char_drag_state = None

    if next_state:
        target_idx = self._resolve_preview_char_label_entry_index()
        if target_idx is not None:
            self._preview_char_selected_index = int(target_idx)
            self._preview_char_hover_label_index = int(target_idx)
            self._preview_char_label_active_index = int(target_idx)
        self._update_preview_edit_status(
            "Tryb wpisywania znaków aktywny. Kliknij box LPM albo użyj strzałek lewo/prawo, a potem wpisz znak z klawiatury.",
            tone="info",
        )
    else:
        self._preview_char_hover_label_index = None
        self._preview_char_label_active_index = None
        self._update_preview_edit_status("Wylaczono tryb wpisywania znaków.", tone="muted")

    self._apply_preview_canvas_cursor()
    if not next_state:
        self._refresh_preview_editor_toolbar()
        self._focus_preview_canvas()
        try:
            self._clear_preview_char_label_canvas_fields()
        except Exception:
            pass
    elif not self._refresh_preview_character_selection_visual(None):
        self._refresh_preview_editor_toolbar()
        self._focus_preview_canvas()
        if not self._redraw_preview_character_overlays_light():
            self._on_preview_select(None)
    return "break"

def _on_preview_char_label_shortcut(self, event=None):
    self._preview_alt_modifier_down = False
    return self._toggle_preview_char_label_mode(event)

def _edit_selected_preview_char_symbol(self, event=None):
    idx, rec = self._get_preview_selected_char_record()
    if rec is None:
        self._set_preview_box_info("Najpierw zaznacz box znaku do korekty.", "warning")
        return "break"
    self._activate_preview_char_label_input(
        idx,
        status_message="Pole znaku jest aktywne tymczasowo. Wpisz 0-9 lub A-Z, aby nadpisać etykietę, albo Esc aby wyjść.",
    )
    return "break"

def _edit_preview_source_filename(self, event=None):
    try:
        from ..campaign_manager import CAMPAIGN

        in_campaign = bool(CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False

    guidance_lines = [
        "Zmianę nazwy pliku źródłowego wykonuj w Z2, a nie w Z3/PZ2.",
        "",
        "Dlaczego:",
        "• Z2 pracuje jeszcze na obrazie źródłowym i całym workflow tablic.",
        "• Z3 powinno służyć już do pracy na wyodrębnionych tablicach i znakach.",
        "",
        "W Z2 zmienisz nazwę bezpieczniej, zanim dane zostaną użyte w kolejnych etapach.",
    ]
    if in_campaign:
        guidance_lines.extend(
            [
                "",
                "Przejdź do grafu i wróć do Z2, aby wykonać zmianę nazwy na właściwej bramce.",
            ]
        )

    messagebox.showinfo(
        "Zmiana nazwy w Z2",
        "\n".join(guidance_lines),
        parent=self.frame.winfo_toplevel() if self.frame is not None else None,
    )
    self._update_preview_edit_status(
        "Zmianę nazwy pliku źródłowego przenieśliśmy do Z2. W Z3/PZ2 ta operacja jest już tylko informacyjna.",
        tone="info",
    )
    self._focus_preview_canvas()
    return "break"

def _delete_selected_preview_char_box(self, event=None):
    perf_start = time.perf_counter()
    idx, rec = self._get_preview_selected_char_record()
    chars = self._get_preview_active_character_records(create=False)
    if rec is None or not isinstance(chars, list):
        self._set_preview_box_info("Najpierw zaznacz box klawiszem S, aby go usunąć.", "warning")
        self._update_preview_edit_status("Najpierw zaznacz box klawiszem S, a potem użyj PPM albo przycisku Usuń box.", tone="warning")
        return "break"

    self._push_preview_history_snapshot()
    try:
        chars.pop(idx)
    except Exception:
        return "break"

    self._preview_char_selected_index = min(idx, len(chars) - 1) if chars else None
    self._preview_char_label_active_index = None
    self._preview_char_hover_label_index = None
    self._preview_char_hover_index = None
    self._preview_char_hover_grip = None
    self._persist_active_preview_characters(
        selected_record=(chars[self._preview_char_selected_index] if self._preview_char_selected_index is not None and chars else None),
        render_preview=False,
        save_immediately=False,
        save_delay_ms=350,
        refresh_row=True,
        success_message="Usunięto box znaku i zapisano zmianę do metadata.json.",
    )
    try:
        elapsed_ms = (time.perf_counter() - perf_start) * 1000.0
        if elapsed_ms >= 80.0:
            logger.info(
                "[Z3/PZ2 PERF] delete_selected_box idx=%s remaining=%s total=%.1fms",
                idx,
                len(chars),
                elapsed_ms,
            )
    except Exception:
        pass
    return "break"

def _on_preview_prev_shortcut(self, event=None):
    if (
        getattr(self, "_preview_char_label_active_index", None) is not None
        and not bool(getattr(self, "_preview_char_label_mode", False))
    ):
        return None
    self._preview_keyboard_crop_navigation_active = True
    if self._select_preview_relative(-1):
        return "break"
    self._preview_keyboard_crop_navigation_active = False
    return None

def _on_preview_next_shortcut(self, event=None):
    if (
        getattr(self, "_preview_char_label_active_index", None) is not None
        and not bool(getattr(self, "_preview_char_label_mode", False))
    ):
        return None
    self._preview_keyboard_crop_navigation_active = True
    if self._select_preview_relative(1):
        return "break"
    self._preview_keyboard_crop_navigation_active = False
    return None

def _on_preview_fit_shortcut(self, event=None):
    if not bool(getattr(self, "_preview_render_state", None)):
        return None
    self._reset_current_preview_view()
    self._update_preview_edit_status("Dopasowano widok tablicy.", tone="muted")
    return "break"

def _on_preview_enter_fullscreen_shortcut(self, event=None):
    self._toggle_preview_fullscreen()
    return "break"

def _on_preview_escape_shortcut(self, event=None):
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        self._set_preview_fullscreen(False)
        return "break"

    if bool(getattr(self, "_preview_char_label_mode", False)):
        previous_active_label_idx = getattr(self, "_preview_char_label_active_index", None)
        previous_hover_label_idx = getattr(self, "_preview_char_hover_label_index", None)
        self._preview_char_label_mode = False
        self._preview_char_hover_label_index = None
        self._preview_char_label_active_index = None
        self._apply_preview_canvas_cursor()
        self._update_preview_edit_status("Wylaczono tryb wpisywania znaków klawiszem Esc.", tone="muted")
        affected_indices = {
            idx for idx in (previous_active_label_idx, previous_hover_label_idx)
            if idx is not None
        }
        if not self._refresh_preview_character_selection_visual(None if not affected_indices else affected_indices):
            self._refresh_preview_editor_toolbar()
            if not self._redraw_preview_character_overlays_light():
                self._on_preview_select(None)
        return "break"
    if getattr(self, "_preview_char_label_active_index", None) is not None:
        previous_active_label_idx = getattr(self, "_preview_char_label_active_index", None)
        previous_hover_label_idx = getattr(self, "_preview_char_hover_label_index", None)
        self._preview_char_label_active_index = None
        self._preview_char_hover_label_index = None
        self._apply_preview_canvas_cursor()
        self._update_preview_edit_status("Zamknieto aktywne pole znaku klawiszem Esc.", tone="muted")
        affected_indices = {
            idx for idx in (previous_active_label_idx, previous_hover_label_idx)
            if idx is not None
        }
        if not self._refresh_preview_character_selection_visual(affected_indices):
            self._refresh_preview_editor_toolbar()
            if not self._redraw_preview_character_overlays_light():
                self._on_preview_select(None)
        return "break"
    return None

def _toggle_preview_fullscreen(self):
    self._set_preview_fullscreen(not bool(getattr(self, "_preview_fullscreen_active", False)))

def _truncate_preview_filename(filename: str, max_chars: int = 44) -> str:
    text = str(filename or "").strip()
    if len(text) <= max_chars:
        return text
    head = max(10, (max_chars // 2) - 2)
    tail = max(10, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"

def _get_preview_legend_context(self):
    current_idx = self._get_current_preview_list_index()
    total = len(getattr(self, "_listbox_pid_by_index", []))
    current_no = (int(current_idx) + 1) if current_idx is not None and total > 0 else 0
    pid = str(getattr(self, "_preview_active_pid", "") or "Brak tablicy")
    active_data = self._get_preview_active_data(create=False) or {}
    source_image = str(active_data.get("source_image", "") or "")
    filename = self._truncate_preview_filename(Path(source_image).name if source_image else pid)
    chars = self._get_preview_active_character_records(create=False)
    mode_text = (
        "Tryb: nowy box" if self._preview_char_add_requested() else
        ("Tryb: znakowanie" if bool(getattr(self, "_preview_char_label_mode", False)) else
         ("Tryb: edycja" if self._preview_char_edit_mode else "Tryb: podgląd"))
    )
    return {
        "filename": filename or "Brak obrazu",
        "image_text": f"Tablica: {current_no}/{total}",
        "vehicle_text": mode_text,
        "plate_text": f"Boxy: {len(chars)}",
    }
