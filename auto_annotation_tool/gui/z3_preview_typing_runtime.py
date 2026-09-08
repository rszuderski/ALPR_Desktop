#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 preview typing overlay and character-label editing helpers."""

from __future__ import annotations

import re
import textwrap
import tkinter as tk


def _append_preview_status_sentence(base: str, sentence: str) -> str:
    base_text = str(base or "").strip()
    addition = str(sentence or "").strip()
    if not addition:
        return base_text
    if not base_text:
        return addition
    if addition.lower() in base_text.lower():
        return base_text
    if base_text.endswith((".", "!", "?", ":")):
        return f"{base_text} {addition}"
    return f"{base_text}. {addition}"

def _clear_preview_char_label_canvas_fields(self) -> bool:
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return False
    try:
        items = canvas.find_withtag("preview_char_label_field")
    except Exception:
        items = ()
    if not items:
        return False
    try:
        canvas.delete("preview_char_label_field")
        return True
    except Exception:
        return False

def _cancel_preview_char_label_interaction(
    self,
    *,
    reset_mode: bool = True,
    clear_hover: bool = True,
    redraw_canvas: bool = True,
) -> bool:
    had_state = bool(getattr(self, "_preview_char_label_mode", False))
    had_state = had_state or getattr(self, "_preview_char_label_active_index", None) is not None
    had_state = had_state or getattr(self, "_preview_char_hover_label_index", None) is not None

    if reset_mode:
        self._preview_char_label_mode = False
    self._preview_char_label_active_index = None
    if clear_hover:
        self._preview_char_hover_label_index = None

    try:
        self._preview_typing_overlay_text = ""
    except Exception:
        pass

    if had_state:
        self._apply_preview_canvas_cursor()
        try:
            self._refresh_preview_editor_toolbar()
        except Exception:
            pass
        try:
            self._sync_preview_edit_status_visibility()
            self._refresh_preview_typing_overlay_visibility()
        except Exception:
            pass
        if redraw_canvas:
            try:
                self._clear_preview_char_label_canvas_fields()
            except Exception:
                pass

    return had_state

def _get_preview_typing_state(self) -> tuple[bool, bool]:
    label_mode_active = bool(getattr(self, "_preview_char_label_mode", False))
    label_input_active = getattr(self, "_preview_char_label_active_index", None) is not None
    return label_mode_active, label_input_active

def _format_preview_typing_status(self, message: str | None = None, *, persistent: bool = False) -> str:
    base_text = str(message or "").strip()
    if persistent:
        if not base_text:
            base_text = "TRYB WPISYWANIA ZNAKÓW AKTYWNY. Kliknij box LPM albo użyj strzałek lewo/prawo i góra/dół, potem wpisz 0-9 lub A-Z."
        elif "tryb wpisywania znaków aktywny" not in base_text.lower():
            base_text = f"TRYB WPISYWANIA ZNAKÓW AKTYWNY. {base_text}"
        return self._append_preview_status_sentence(base_text, "Esc kończy tryb wpisywania.")

    if not base_text:
        base_text = "AKTYWNE POLE ZNAKU. Wpisz 0-9 lub A-Z, aby nadpisać etykietę."
    elif "aktywne pole znaku" not in base_text.lower() and "pole znaku jest aktywne" not in base_text.lower():
        base_text = f"AKTYWNE POLE ZNAKU. {base_text}"
    return self._append_preview_status_sentence(base_text, "Strzałki góra/dół przełączają rzędy. Esc zamyka wpisywanie.")

def _get_preview_typing_overlay_text(self) -> str:
    if not bool(getattr(self, "_preview_operation_assistant_visible", True)):
        return ""
    cached = str(getattr(self, "_preview_typing_overlay_text", "") or "").strip()
    if cached:
        return cached

    label_mode_active, label_input_active = self._get_preview_typing_state()
    if label_mode_active:
        return self._format_preview_typing_status(persistent=True)
    if label_input_active:
        return self._format_preview_typing_status(persistent=False)
    return self._format_preview_operation_assistant_status()

def _format_preview_operation_assistant_status(self) -> str:
    if not self._preview_status_overlay_available():
        return ""
    add_state = getattr(self, "_preview_char_add_state", None)
    if isinstance(add_state, dict) and bool(add_state.get("click_draw")):
        return "AS: D aktywne - przesuń mysz i kliknij drugi narożnik boxa."
    if bool(getattr(self, "_preview_char_add_click_armed", False)):
        return "AS: D uzbraja box - kliknij pierwszy narożnik."
    if self._preview_char_add_requested():
        return "AS: rysuj box - LPM przeciąga narożniki."
    if bool(getattr(self, "_preview_char_edit_mode", False)):
        return "AS: edycja boxów - S zaznacza, LPM przesuwa, uchwyty skalują."
    return (
        "AS: Q/E tablice, D nowy box, S zaznacza, Alt+W wpisywanie. "
        f"{self._format_preview_layout_semantics(short=True)}"
    )

def _preview_status_overlay_available(self) -> bool:
    if getattr(self, "preview_canvas", None) is None:
        return False
    return bool(str(getattr(self, "_preview_active_pid", "") or "").strip())

def _preview_operation_overlay_available(self) -> bool:
    if not bool(getattr(self, "_preview_operation_assistant_visible", True)):
        return False
    return self._preview_status_overlay_available()

def _resolve_preview_typing_overlay_anchor(self, overlay_w: float, overlay_h: float) -> tuple[float, float] | None:
    canvas_w, canvas_h = self._get_preview_canvas_size()
    margin = 8.0
    bottom_offset = self._get_preview_typing_overlay_bottom_offset()
    max_y = max(margin, float(canvas_h) - float(overlay_h) - float(bottom_offset))

    base_x, base_y = self._get_preview_typing_overlay_anchor(overlay_h)
    base_x = max(margin, min(max(margin, float(canvas_w) - float(overlay_w) - margin), float(base_x)))
    base_y = max(margin, min(max_y, float(base_y)))
    bounds = getattr(self, "_preview_controls_legend_current_bounds", None)
    if bool(getattr(self, "_preview_controls_legend_visible", True)) and bounds:
        left, top, right, bottom = bounds
        if base_x < right and base_x + overlay_w > left and base_y < bottom and base_y + overlay_h > top:
            if bottom + margin + overlay_h <= canvas_h - margin:
                base_y = bottom + margin
            elif right + margin + overlay_w <= canvas_w - margin:
                base_x = right + margin
    # AS jest czystą nakładką canvasa: nie rezerwuje miejsca i nie omija tablicy.
    # Jeśli nachodzi na obraz, ma go jedynie przysłonić, nigdy przestawiać.
    return float(base_x), float(base_y)

def _get_preview_typing_overlay_bottom_offset(self) -> float:
    # The compass floats above the image; it is not a reserved bottom rail.
    return 8.0

def _get_preview_controls_legend_clearance_y(self) -> float:
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        return 0.0
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        return 0.0

    bounds = getattr(self, "_preview_controls_legend_current_bounds", None)
    if isinstance(bounds, tuple) and len(bounds) == 4:
        try:
            return max(0.0, float(bounds[3]) + 6.0)
        except Exception:
            pass

    legend_height = float(getattr(self, "_preview_controls_legend_current_height", 0.0) or 0.0)
    if legend_height <= 0.0:
        return 0.0
    top_bar_height = max(38.0, float(getattr(self, "_preview_overlay_top_bar_height", 38.0) or 38.0))
    return float(top_bar_height + legend_height + 12.0)

def _get_preview_typing_overlay_anchor(self, overlay_h: float) -> tuple[float, float]:
    canvas_w, canvas_h = self._get_preview_canvas_size()
    margin = 8.0
    x = margin
    top_bar_height = max(38.0, float(getattr(self, "_preview_overlay_top_bar_height", 38.0) or 38.0))
    y = max(margin, float(top_bar_height + 6.0))

    bottom_offset = self._get_preview_typing_overlay_bottom_offset()
    max_y = max(margin, float(canvas_h) - float(overlay_h) - float(bottom_offset))
    y = min(y, max_y)
    return float(x), float(y)

def _configure_preview_typing_overlay_text(self, text: str, wraplength: int) -> None:
    widget = getattr(self, "preview_typing_overlay_lbl", None)
    if widget is None:
        return

    content = str(text or "")
    config_key = (content, int(wraplength))
    if getattr(self, "_preview_typing_overlay_config_key", None) == config_key:
        return
    if not isinstance(widget, tk.Text):
        try:
            widget.configure(text=content, wraplength=wraplength, font=("Segoe UI", 8, "normal"))
            self._preview_typing_overlay_config_key = config_key
        except Exception:
            pass
        return

    width_chars = max(18, min(34, int(max(130, wraplength) / 7)))
    wrapped = textwrap.wrap(content, width=width_chars) if content else [""]
    height_lines = max(1, min(5, len(wrapped)))
    key_pattern = re.compile(
        r"(?<![\w+])(?:Ctrl\+Z|Ctrl\+Y|Alt\+W|D\+LPM|LPM\+drag|Q/E|0-9|A-Z|LPM|PPM|Esc|Enter|Spacja|Space|D|S|N|F|Q|E)(?![\w+])"
    )

    try:
        widget.configure(state=tk.NORMAL, width=width_chars, height=height_lines)
        widget.delete("1.0", tk.END)
        cursor = 0
        for match in key_pattern.finditer(content):
            if match.start() > cursor:
                widget.insert(tk.END, content[cursor:match.start()], ("body",))
            widget.insert(tk.END, match.group(0), ("key",))
            cursor = match.end()
        if cursor < len(content):
            widget.insert(tk.END, content[cursor:], ("body",))
        widget.configure(state=tk.DISABLED)
        self._preview_typing_overlay_config_key = config_key
    except Exception:
        try:
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, content)
            widget.configure(state=tk.DISABLED)
            self._preview_typing_overlay_config_key = config_key
        except Exception:
            pass

def _get_preview_char_label_canvas_rect(self, rec) -> tuple[float, float, float, float] | None:
    bbox = self._char_record_bbox(rec)
    if not bbox:
        return None
    cx1, cy1 = self._preview_image_to_canvas_point(bbox[0], bbox[1])
    cx2, _cy2 = self._preview_image_to_canvas_point(bbox[2], bbox[3])
    label_w = min(max(18.0, (cx2 - cx1) * 0.34), 28.0)
    label_h = 18.0
    inset = 2.0
    return (
        float(cx1 + inset),
        float(cy1 + inset),
        float(cx1 + inset + label_w),
        float(cy1 + inset + label_h),
    )

def _find_preview_char_label_hit(self, canvas_x: float, canvas_y: float):
    if (
        not bool(getattr(self, "_preview_char_label_mode", False))
        and getattr(self, "_preview_char_label_active_index", None) is None
    ):
        return None
    chars = self._get_preview_active_character_records(create=False)
    for idx, rec in enumerate(chars):
        rect = self._get_preview_char_label_canvas_rect(rec)
        if rect is None:
            continue
        rx1, ry1, rx2, ry2 = rect
        pad = 2.0
        if (rx1 - pad) <= float(canvas_x) <= (rx2 + pad) and (ry1 - pad) <= float(canvas_y) <= (ry2 + pad):
            return idx
    return None

def _get_leftmost_preview_char_index(self, chars=None) -> int | None:
    local_chars = chars if isinstance(chars, list) else self._get_preview_active_character_records(create=False)
    if not isinstance(local_chars, list) or not local_chars:
        return None

    ordered_chars = self._sort_character_records_by_x(list(local_chars))
    target_rec = ordered_chars[0] if ordered_chars else None
    if target_rec is None:
        return None
    return self._find_preview_char_record_index(local_chars, target_rec)

def _resolve_preview_char_label_entry_index(self) -> int | None:
    chars = self._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not chars:
        return None

    total = len(chars)

    def _safe_index(value) -> int | None:
        try:
            idx = int(value)
        except Exception:
            return None
        return idx if 0 <= idx < total else None

    explicit_hover_idx = _safe_index(getattr(self, "_preview_char_hover_label_index", None))
    if explicit_hover_idx is None:
        explicit_hover_idx = _safe_index(getattr(self, "_preview_char_hover_index", None))
    if explicit_hover_idx is not None:
        return explicit_hover_idx

    explicit_active_idx = _safe_index(getattr(self, "_preview_char_label_active_index", None))
    if explicit_active_idx is not None:
        return explicit_active_idx

    return self._get_leftmost_preview_char_index(chars)

def _activate_preview_char_label_input(self, char_idx: int | None, *, keep_selection: bool = True, status_message: str | None = None):
    previous_selected_idx = getattr(self, "_preview_char_selected_index", None)
    previous_hover_label_idx = getattr(self, "_preview_char_hover_label_index", None)
    previous_active_label_idx = getattr(self, "_preview_char_label_active_index", None)
    if char_idx is None:
        self._preview_char_label_active_index = None
    else:
        self._preview_char_label_active_index = int(char_idx)
        if keep_selection:
            self._preview_char_selected_index = int(char_idx)
        self._preview_char_hover_label_index = int(char_idx)
    if status_message:
        self._update_preview_edit_status(status_message, tone="info")
    else:
        self._update_preview_edit_status()
    self._focus_preview_canvas()
    affected_indices = {
        idx for idx in (
            previous_selected_idx,
            previous_hover_label_idx,
            previous_active_label_idx,
            char_idx,
        )
        if idx is not None
    }
    if not self._refresh_preview_character_selection_visual(affected_indices):
        self._on_preview_select(None)

def _assign_character_to_active_preview_label(self, symbol: str):
    idx = getattr(self, "_preview_char_label_active_index", None)
    chars = self._get_preview_active_character_records(create=False)
    if idx is None or not (0 <= int(idx) < len(chars)):
        return False

    symbol = self._sanitize_preview_char_symbol(symbol)
    if not symbol:
        return False

    rec = chars[int(idx)]
    if self._sanitize_preview_char_symbol(rec.get("character", "")) == symbol:
        self._preview_char_selected_index = int(idx)
        self._preview_char_label_active_index = int(idx)
        self._preview_char_hover_label_index = int(idx)
        self._update_preview_edit_status(
            "Znak już ma taką wartość. Użyj strzałek lewo/prawo albo kliknij inny box LPM.",
            tone="muted",
        )
        if not self._refresh_preview_character_selection_visual({int(idx)}):
            self._on_preview_select(None)
        return True
    self._push_preview_history_snapshot()
    rec["character"] = symbol
    self._mark_preview_char_record_manual(rec, box=False, sign=True)
    self._preview_char_selected_index = int(idx)
    self._preview_char_label_active_index = int(idx)
    self._preview_char_hover_label_index = int(idx)
    self._persist_active_preview_characters(
        selected_record=rec,
        success_message=f"Zapisano znak {symbol} w metadata.json.",
        render_preview=False,
        save_immediately=False,
        save_delay_ms=900,
        refresh_row=True,
        light_redraw_indices="selected",
    )
    if bool(getattr(self, "_preview_char_label_mode", False)):
        self._update_preview_edit_status(
            f"Zapisano znak {symbol}. Użyj strzałek lewo/prawo albo kliknij kolejny box LPM.",
            tone="success",
        )
    return True
