from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from ..config import logger
from .web_slim_scrollbar import blend_hex_colors

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def normalize_preview_sort_mode_key(mode_key: str | None, sort_labels: dict) -> str:
    normalized = str(mode_key or "DEFAULT").strip().upper() or "DEFAULT"
    legacy_aliases = {
        "PERFECT": "OK",
        "MANUAL": "M",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    return normalized if normalized in sort_labels else "DEFAULT"


def normalize_preview_layout_filter_key(filter_key: str | None, layout_filter_labels: dict) -> str:
    normalized = str(filter_key or "ALL").strip().upper() or "ALL"
    legacy_aliases = {
        "TWO_ROW": "2R",
        "TWO_ROW_CANDIDATE": "2R?",
        "SINGLE_ROW": "1R",
        "OVERRIDE": "MANUAL",
        "AUTO": "ALL",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    return normalized if normalized in layout_filter_labels else "ALL"


def get_preview_sort_mode_key(host: "CharacterAnnotationTab", sort_labels: dict) -> str:
    raw_value = (
        getattr(host, "preview_sort_mode_var", None).get()
        if hasattr(host, "preview_sort_mode_var")
        else "DEFAULT"
    )
    return normalize_preview_sort_mode_key(raw_value, sort_labels)


def get_preview_layout_filter_key(host: "CharacterAnnotationTab", layout_filter_labels: dict) -> str:
    raw_value = (
        getattr(host, "preview_layout_filter_var", None).get()
        if hasattr(host, "preview_layout_filter_var")
        else "ALL"
    )
    return normalize_preview_layout_filter_key(raw_value, layout_filter_labels)


def plate_matches_preview_layout_filter(
    host: "CharacterAnnotationTab",
    data: dict | None,
    layout_filter_labels: dict,
) -> bool:
    # Historycznie ten przełącznik ukrywał pozycje 1R/2R/2R?.
    # To sprawiało wrażenie utraty anotacji po zmianie układu tablicy.
    # Od teraz układ jest tylko kryterium grupowania/sortowania listy.
    return True


def preview_layout_group_matches(
    host: "CharacterAnnotationTab",
    data: dict | None,
    mode_key: str,
) -> bool:
    mode_key = str(mode_key or "ALL").strip().upper() or "ALL"
    if mode_key not in {"1R", "2R"}:
        return False
    if not isinstance(data, dict):
        return False

    layout_label, _tone = host._get_preview_plate_layout_label(data)
    layout_label = str(layout_label or "?").strip().upper()
    layout = str(data.get("plate_layout", "") or "").strip().lower()

    if mode_key == "1R":
        return layout == "single_row" or layout_label.startswith("1R")
    if mode_key == "2R":
        return layout == "two_row" or layout_label in {"2R", "2R*"}
    return False


def get_preview_layout_group_priority(
    host: "CharacterAnnotationTab",
    data: dict | None,
    original_index: int,
) -> tuple:
    mode_key = host._get_preview_layout_filter_key()
    if mode_key == "ALL":
        return (0, int(original_index))

    matches = preview_layout_group_matches(host, data, mode_key)
    return (0 if matches else 1, int(original_index))


def get_preview_sort_source_count(host: "CharacterAnnotationTab", mode_key: str, data: dict | None = None) -> int:
    normalized_key = str(mode_key or "").strip().upper()
    source_data = data if isinstance(data, dict) else {}
    chars = list(source_data.get("characters", []) or []) if isinstance(source_data, dict) else []
    if not chars:
        return 0

    counts = host._count_character_sources(chars, data=source_data)
    if normalized_key == "YOLO":
        return int(counts.get("yolo_box", 0)) + int(counts.get("yolo_symbol", 0))
    if normalized_key == "OCR":
        return int(counts.get("generated_box", 0)) + int(counts.get("ocr_symbol", 0))
    if normalized_key == "M":
        return int(counts.get("manual_box", 0)) + int(counts.get("manual_sign", 0))
    if normalized_key == "BOXES":
        return int(len(chars))
    return 0


def get_preview_sort_priority(host: "CharacterAnnotationTab", pid: str, data: dict, original_index: int):
    mode_key = host._get_preview_sort_mode_key()
    status = str((data or {}).get("status", "unknown")).strip().lower()
    strategy_bucket = host._get_perfect_strategy_bucket(data)
    total_boxes = len(list((data or {}).get("characters", []) or [])) if isinstance(data, dict) else 0

    if mode_key == "DEFAULT":
        return (0, int(original_index))

    if mode_key == "OK":
        if status == "perfect":
            return (0, -int(total_boxes), int(original_index))
        if status == "needs_fix":
            return (1, -int(total_boxes), int(original_index))
        return (2, -int(total_boxes), int(original_index))

    if mode_key in {"1R", "2R"}:
        matches_layout = preview_layout_group_matches(host, data, mode_key)
        return (
            0 if matches_layout else 1,
            0 if status == "perfect" else (1 if status == "needs_fix" else 2),
            -int(total_boxes),
            int(original_index),
        )

    if mode_key == "BOXES":
        return (
            -int(total_boxes),
            0 if status == "perfect" else (1 if status == "needs_fix" else 2),
            int(original_index),
        )

    preferred_bucket = {
        "YOLO": {"yolo_exact", "yolo_box_ocr"},
        "OCR": {"ocr_exact"},
        "M": {"other_perfect"},
    }.get(mode_key)

    if preferred_bucket:
        source_count = get_preview_sort_source_count(host, mode_key, data=data)
        return (
            -int(source_count),
            -int(total_boxes),
            0 if (status == "perfect" and strategy_bucket in preferred_bucket) else 1,
            0 if status == "perfect" else (1 if status == "needs_fix" else 2),
            int(original_index),
        )

    return (0, int(original_index))


def get_sorted_preview_plate_ids(host: "CharacterAnnotationTab", ordered_pids):
    base_order = list(ordered_pids or [])
    mode_key = host._get_preview_sort_mode_key()
    if mode_key == "DEFAULT":
        return base_order

    indexed = list(enumerate(base_order))
    indexed.sort(
        key=lambda item: get_preview_sort_priority(
            host,
            item[1],
            host.preview_metadata.get(item[1], {}),
            item[0],
        )
    )
    return [pid for _, pid in indexed]


def set_preview_sort_hover(host: "CharacterAnnotationTab", mode_key: str, hovered: bool) -> None:
    normalized_key = str(mode_key or "").strip().upper()
    if hovered:
        host._preview_sort_hover_key = normalized_key
    elif host._preview_sort_hover_key == normalized_key:
        host._preview_sort_hover_key = None
    host._apply_preview_sort_bar_style()


def set_preview_layout_filter_hover(host: "CharacterAnnotationTab", filter_key: str, hovered: bool) -> None:
    normalized_key = host._normalize_preview_layout_filter_key(filter_key)
    if hovered:
        host._preview_layout_filter_hover_key = normalized_key
    elif host._preview_layout_filter_hover_key == normalized_key:
        host._preview_layout_filter_hover_key = None
    host._apply_preview_sort_bar_style()


def get_preview_box_variants(host: "CharacterAnnotationTab", data: dict) -> dict:
    if not isinstance(data, dict):
        data = {}
    canonical_chars = host._sort_character_records_by_x(data.get("characters", []))
    yolo_filtered = host._sort_character_records_by_x(data.get("yolo_detections", []))
    yolo_nms = host._sort_character_records_by_x(data.get("yolo_nms_detections", []))
    yolo_raw = host._sort_character_records_by_x(data.get("yolo_raw_detections", []))

    if not yolo_filtered:
        yolo_from_canonical = [
            rec for rec in canonical_chars
            if isinstance(rec, dict) and str(rec.get("method", "")).strip().lower() == "yolo"
        ]
        yolo_filtered = host._sort_character_records_by_x(yolo_from_canonical)

    return {
        "FINAL": canonical_chars,
        "YOLO_FILTERED": yolo_filtered,
        "YOLO_NMS": yolo_nms,
        "YOLO_RAW": yolo_raw,
    }


def get_preview_box_records(host: "CharacterAnnotationTab", data: dict):
    variants = get_preview_box_variants(host, data)
    method_name = host._get_detection_method_key()
    mode_key = host._get_preview_box_mode_key()

    if mode_key == "AUTO":
        final_records = list(variants.get("FINAL", []) or [])
        if final_records:
            return final_records, "FINAL"
        if method_name == "YOLO":
            for candidate_key in ("YOLO_FILTERED", "YOLO_NMS", "YOLO_RAW"):
                candidate_records = variants.get(candidate_key, [])
                if candidate_records:
                    return candidate_records, candidate_key
        for candidate_key in ("YOLO_FILTERED", "YOLO_NMS", "YOLO_RAW"):
            candidate_records = variants.get(candidate_key, [])
            if candidate_records:
                return candidate_records, candidate_key
        return final_records, "FINAL"

    return variants.get(mode_key, []), mode_key


def characters_to_text(host: "CharacterAnnotationTab", chars, data=None) -> str:
    if chars is None:
        return ""

    if isinstance(chars, str):
        return chars

    if not isinstance(chars, list):
        return str(chars)

    prepared = []
    for i, rec in enumerate(host._sort_character_records_by_x(chars, data=data)):
        symbol, _x_key = host._char_record_to_symbol_and_x(rec, fallback_index=i)
        if symbol:
            prepared.append(symbol)

    return "".join(prepared)


def characters_to_display_rows(host: "CharacterAnnotationTab", chars, data=None) -> list[str]:
    """Return visual reading rows without changing the canonical text order."""
    if chars is None:
        return []
    if isinstance(chars, str):
        return [chars] if chars else []
    if not isinstance(chars, list):
        text = str(chars)
        return [text] if text else []

    try:
        ordered = host._sort_character_records_by_x(chars, data=data)
    except Exception:
        ordered = list(chars or [])

    try:
        records = host._annotate_preview_character_reading_positions(ordered, data=data)
    except Exception:
        records = list(ordered or [])

    rows: dict[int, list[tuple[int, int, str]]] = {}
    fallback: list[str] = []
    max_row = 1
    for i, rec in enumerate(records):
        symbol, _x_key = host._char_record_to_symbol_and_x(rec, fallback_index=i)
        if not symbol:
            continue
        fallback.append(symbol)
        row_no = 1
        col_no = i + 1
        if isinstance(rec, dict):
            try:
                row_no = int(rec.get("reading_row", 1) or 1)
            except Exception:
                row_no = 1
            try:
                col_no = int(rec.get("reading_col", i + 1) or (i + 1))
            except Exception:
                col_no = i + 1
        row_no = max(1, int(row_no))
        col_no = max(1, int(col_no))
        max_row = max(max_row, row_no)
        rows.setdefault(row_no, []).append((col_no, i, symbol))

    if not fallback:
        return []

    two_row_active = False
    try:
        two_row_active = bool(host._is_preview_two_row_layout_active(data))
    except Exception:
        two_row_active = False
    if max_row < 2 and not two_row_active:
        return ["".join(fallback)]

    display_rows = []
    for row_no in sorted(rows):
        parts = [symbol for _col, _idx, symbol in sorted(rows[row_no], key=lambda item: (item[0], item[1]))]
        display_rows.append("".join(parts))
    return [row for row in display_rows if row]


def characters_to_display_text(host: "CharacterAnnotationTab", chars, data=None, *, separator: str = " / ") -> str:
    rows = characters_to_display_rows(host, chars, data=data)
    if not rows:
        return ""
    return str(separator).join(rows)


def get_plate_listbox_ordinal(host: "CharacterAnnotationTab", plate_id: str | None = None) -> int | None:
    pid = str(plate_id or "").strip()
    if not pid:
        return None

    current_order = list(getattr(host, "_listbox_pid_by_index", []) or [])
    if pid in current_order:
        return int(current_order.index(pid)) + 1

    metadata = getattr(host, "preview_metadata", {})
    if not isinstance(metadata, dict) or pid not in metadata:
        return None

    candidate_ids = list(getattr(host, "preview_plate_ids", []) or [])
    candidate_ids = [item for item in candidate_ids if item in metadata]
    if not candidate_ids:
        candidate_ids = list(metadata.keys())

    try:
        sorted_ids = host._get_sorted_preview_plate_ids(candidate_ids)
    except Exception:
        sorted_ids = list(candidate_ids)

    return (int(sorted_ids.index(pid)) + 1) if pid in sorted_ids else None


def format_preview_record_source_label(host: "CharacterAnnotationTab", data: dict | None, gold_source_labels: dict) -> tuple[str, str]:
    if not isinstance(data, dict):
        return "Źródło rekordu: brak zaznaczonej tablicy", "muted"

    bucket = host._get_plate_source_bucket(data, meta_path=host._loaded_meta_path)
    source_label = gold_source_labels.get(bucket, "Nieznane źródło")
    tone = "muted"
    if bucket in {"local_manual", "cvat_manual"}:
        tone = "success"
    elif bucket == "auto_preview":
        tone = "info"
    return f"Źródło rekordu: {source_label}", tone


def format_plate_listbox_label(host: "CharacterAnnotationTab", plate_id: str, data: dict, *,
                              ordinal: int | None = None, evaluate_status: bool = True) -> str:
    try:
        status = str(
            (host._get_preview_live_status(
                data,
                chars=data.get("characters", []) if isinstance(data, dict) else None,
                plate_id=plate_id,
            ) if evaluate_status else data.get("status", "unknown"))
            or data.get("status", "unknown")
        ).strip().lower()
        if isinstance(data, dict) and status:
            data["status"] = status
    except Exception:
        status = str(data.get("status", "unknown")).strip().lower()
    if hasattr(host, "_characters_to_display_text"):
        chars_txt = host._characters_to_display_text(data.get("characters", []), data=data, separator=" / ")
    else:
        chars_txt = host._characters_to_text(data.get("characters", []), data=data)
    if ordinal is None:
        ordinal = host._get_plate_listbox_ordinal(plate_id)

    if status == "perfect":
        icon = "🟢"
    elif status == "needs_fix":
        icon = "🔴"
    else:
        icon = "⚪"

    prefix = f"{ordinal}. " if ordinal is not None else ""
    flags = []
    if status == "perfect":
        flags.append("OK")
    layout_flag = host._get_plate_listbox_layout_flag(data)
    if layout_flag:
        flags.append(layout_flag)
    flags.extend(host._get_plate_listbox_source_flags(data))
    flag_prefix = f"{'|'.join(flags)} " if flags else ""

    label = f"{prefix}{flag_prefix}{icon} {plate_id}"
    if chars_txt:
        label += f" [{chars_txt}]"
    return label


def get_plate_row_foreground(host: "CharacterAnnotationTab", status: str) -> str:
    status = str(status or "unknown").strip().lower()
    palette = getattr(host.app, "palette", {})

    if status == "perfect":
        return palette.get("success", "#27ae60")
    if status == "needs_fix":
        return palette.get("error", "#c0392b")
    return palette.get("muted_dim", "#444444")


def apply_plate_listbox_row_style(host: "CharacterAnnotationTab", row_index: int, status: str) -> None:
    try:
        fg = get_plate_row_foreground(host, status)
        _select_bg, select_fg = host.app.get_list_selection_colors()
        host.plates_listbox.itemconfig(
            row_index,
            foreground=fg,
            selectforeground=select_fg,
        )
    except Exception as e:
        logger.debug(f"Nie udało się ustawić stylu wiersza listy [{row_index}]: {e}")


def refresh_preview_import_focus_ui(host: "CharacterAnnotationTab") -> None:
    frame = getattr(host, "preview_import_focus_frame", None)
    button = getattr(host, "preview_import_focus_btn", None)
    hint = getattr(host, "preview_import_focus_hint_lbl", None)

    known_ids = [
        str(pid or "").strip()
        for pid in list(getattr(host, "_preview_import_focus_plate_ids", []) or [])
        if str(pid or "").strip()
    ]
    matching_ids = [pid for pid in known_ids if pid in getattr(host, "preview_metadata", {})]
    known_count = len(matching_ids)
    active = bool(getattr(host, "_preview_import_focus_active", False) and matching_ids)

    if frame is not None:
        try:
            if known_count:
                if not str(frame.winfo_manager()):
                    frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            elif str(frame.winfo_manager()):
                frame.grid_remove()
        except Exception:
            pass

    if button is not None:
        try:
            button.configure(
                state=(tk.NORMAL if known_count else tk.DISABLED),
                text=("Pokaż wszystkie tablice" if active else f"Pokaż tylko zaimportowane ({known_count})"),
            )
        except Exception:
            pass

    if hint is not None and known_count:
        if active:
            host._set_inline_status_label_state(
                hint,
                text=f"Widok zawężony do ostatniego importu CVAT: {known_count} tablic.",
                tone="success",
                emphasis=False,
            )
        else:
            host._set_inline_status_label_state(
                hint,
                text=f"Ostatni import CVAT zaktualizował {known_count} tablic. Użyj widoku zaimportowanych, aby szybko je sprawdzić.",
                tone="muted",
                emphasis=False,
            )


def set_preview_import_focus(
    host: "CharacterAnnotationTab",
    plate_ids,
    *,
    import_batch_id: str = "",
    activate: bool = False,
) -> None:
    unique_ids = []
    seen = set()
    for pid in list(plate_ids or []):
        normalized = str(pid or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_ids.append(normalized)

    host._preview_import_focus_plate_ids = unique_ids
    host._preview_import_focus_batch_id = str(import_batch_id or "").strip()
    host._preview_import_focus_active = bool(unique_ids) and bool(activate)
    host._refresh_preview_import_focus_ui()
    if hasattr(host, "plates_listbox"):
        host._rebuild_preview_listbox(preserve_selection=False)


def toggle_preview_import_focus(host: "CharacterAnnotationTab") -> None:
    if not list(getattr(host, "_preview_import_focus_plate_ids", []) or []):
        return
    host._preview_import_focus_active = not bool(getattr(host, "_preview_import_focus_active", False))
    host._refresh_preview_import_focus_ui()
    if hasattr(host, "plates_listbox"):
        host._rebuild_preview_listbox(preserve_selection=False)


def apply_preview_sort_bar_style(
    host: "CharacterAnnotationTab",
    *,
    sort_labels: dict,
    sort_color_keys: dict,
) -> None:
    palette = getattr(host.app, "palette", {})
    frame = getattr(host, "preview_sort_bar", None)
    label = getattr(host, "preview_sort_title_lbl", None)
    list_host = getattr(host, "preview_list_host", None)
    listbox = getattr(host, "plates_listbox", None)
    shell_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    list_bg = palette.get("input_bg", palette.get("panel", "#1f1f1f"))
    border = palette.get("border", "#3c3c3c")
    hover_bg = palette.get("button_hover", palette.get("surface_info", "#34373b"))
    active_bg = palette.get("surface_info", hover_bg)

    if list_host is not None:
        try:
            list_host.configure(bg=shell_bg)
        except Exception:
            pass
    if listbox is not None:
        try:
            select_bg, select_fg = host.app.get_list_selection_colors()
            listbox.configure(
                bg=list_bg,
                fg=palette.get("fg", "#f3f3f3"),
                selectbackground=select_bg,
                selectforeground=select_fg,
                disabledforeground=palette.get("muted", "#a0a0a0"),
                insertbackground=palette.get("fg", "#f3f3f3"),
                activestyle="none",
                highlightthickness=1,
                highlightbackground=border,
                highlightcolor=border,
                relief=tk.FLAT,
                bd=0,
            )
        except Exception:
            pass
    if frame is not None:
        try:
            frame.configure(
                bg=shell_bg,
                highlightbackground=shell_bg,
                highlightcolor=shell_bg,
                padx=0,
                pady=0,
            )
        except Exception:
            pass
    if label is not None:
        try:
            label.configure(
                text="Sortowanie listy:",
                bg=shell_bg,
                fg=palette.get("muted", "#a0a0a0"),
                font=("Segoe UI", 8, "bold"),
            )
        except Exception:
            pass
    buttons_frame = getattr(host, "preview_sort_buttons_frame", None)
    if buttons_frame is not None:
        try:
            buttons_frame.configure(bg=shell_bg)
        except Exception:
            pass
        for child in buttons_frame.winfo_children():
            if isinstance(child, tk.Frame):
                try:
                    child.configure(bg=shell_bg)
                except Exception:
                    pass

    active_key = host._get_preview_sort_mode_key()
    hover_key = str(getattr(host, "_preview_sort_hover_key", "") or "").strip().upper()
    for mode_key, button in getattr(host, "preview_sort_buttons", {}).items():
        if button is None:
            continue
        is_active = mode_key == active_key
        is_hovered = mode_key == hover_key
        tone_key = sort_color_keys.get(mode_key, "muted")
        accent = palette.get(tone_key, palette.get("accent", "#4aa3ff"))
        base_mix = 0.80 if mode_key == "DEFAULT" else 0.72
        bg = blend_hex_colors(accent, shell_bg, base_mix)
        fg = palette.get("fg", "#f3f3f3")
        relief = tk.FLAT
        border_width = 0

        if is_active:
            bg = blend_hex_colors(accent, shell_bg, 0.42)
        elif is_hovered:
            bg = blend_hex_colors(accent, shell_bg, 0.56)

        resolved_fg = host._get_readable_text_color(bg, preferred=fg)
        active_bg_value = blend_hex_colors(accent, shell_bg, 0.48)
        active_fg = host._get_readable_text_color(active_bg_value, preferred=resolved_fg)

        try:
            button.configure(
                text=str(sort_labels.get(mode_key, mode_key) or mode_key),
                bg=bg,
                fg=resolved_fg,
                activebackground=active_bg_value,
                activeforeground=active_fg,
                relief=relief,
                bd=border_width,
                highlightbackground=shell_bg,
                highlightcolor=shell_bg,
                highlightthickness=0,
                disabledforeground=resolved_fg,
                font=("Segoe UI", 8, "bold" if is_active else "normal"),
            )
        except Exception:
            pass


