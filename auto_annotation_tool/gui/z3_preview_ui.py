from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from .web_slim_scrollbar import blend_hex_colors
from .z3_preview_status_ui import (
    apply_preview_source_actions_style,
    apply_preview_info_stats_style,
    apply_preview_typing_overlay_style,
    apply_preview_surface_style,
    sync_preview_intro_wraplength,
    apply_preview_focus_prompt_style,
    set_preview_processing_overlay,
    update_preview_processing_overlay_progress,
    set_preview_info,
    set_preview_counts_info,
    format_preview_layout_summary,
    set_preview_layout_summary_info,
    build_plates_list_legend_counts,
    set_plates_legend_info,
    set_preview_repair_progress_status,
    sync_preview_edit_status_visibility,
    update_preview_edit_status,
    refresh_preview_typing_overlay_visibility,
    get_preview_repair_progress_snapshot,
    update_preview_repair_progress_ui,
    set_preview_fusion_info,
    set_preview_box_info,
    update_preview_box_info_label,
    update_preview_info_label,
    refresh_preview_live_metadata_ui,
    refresh_preview_layout_override_ui_light,
)
from .z3_preview_dock_ui import (
    get_preview_overlay_dock_theme,
    render_preview_overlay_dock,
    place_preview_overlay_dock,
)
from .z3_preview_compass_ui import (
    _hex_to_rgba,
    _measure_preview_legend_token,
    _normalize_preview_legend_interaction,
    _preview_legend_image_cache,
    get_preview_legend_theme,
    get_preview_legend_compass_photo,
    get_preview_legend_interaction_marker_photo,
    draw_preview_legend_interaction_marker,
    draw_preview_legend_keycap,
    draw_preview_legend_compass_toggle,
    draw_preview_legend_grab_handle,
    clamp_preview_controls_legend_offsets,
    toggle_preview_controls_legend,
    update_preview_controls_legend_cursor,
    on_preview_controls_legend_press,
    on_preview_controls_legend_drag,
    on_preview_controls_legend_motion,
    on_preview_controls_legend_leave,
    on_preview_controls_legend_release,
    on_preview_controls_legend_mousewheel,
    build_preview_legend_sections,
    refresh_preview_controls_legend,
    place_preview_hint_overlay,
)
from .z3_preview_list_ui import (
    normalize_preview_sort_mode_key,
    normalize_preview_layout_filter_key,
    get_preview_sort_mode_key,
    get_preview_layout_filter_key,
    plate_matches_preview_layout_filter,
    get_preview_sort_source_count,
    get_preview_sort_priority,
    get_sorted_preview_plate_ids,
    set_preview_sort_hover,
    set_preview_layout_filter_hover,
    get_preview_box_variants,
    get_preview_box_records,
    characters_to_text,
    characters_to_display_rows,
    characters_to_display_text,
    get_plate_listbox_ordinal,
    format_preview_record_source_label,
    format_plate_listbox_label,
    get_plate_row_foreground,
    apply_plate_listbox_row_style,
    refresh_preview_import_focus_ui,
    set_preview_import_focus,
    toggle_preview_import_focus,
    apply_preview_sort_bar_style,
)
from .z3_preview_badges import (
    draw_preview_badge_stack,
    draw_preview_fixed_text_badge,
    draw_preview_source_legend,
    draw_preview_text_badge,
    estimate_preview_badge_layout,
    estimate_preview_source_legend_height,
    estimate_preview_source_legend_width,
    fit_preview_text_to_width,
    get_preview_badge_component_style,
    get_preview_badge_layout_metrics,
    get_preview_source_badge_layers,
    get_preview_source_component_legend_items,
    get_preview_source_visual_style,
    measure_preview_badge_stack,
    measure_preview_overlay_font_height,
    measure_preview_overlay_text_width,
    measure_preview_text_badge,
)
from .z3_detection_runtime import clear_detection_review_snapshot_after_manual_edit
from .z3_metadata_cache import read_preview_metadata, mark_preview_metadata_changed
from .z3_preview_layout_control import draw_plate_layout_control, hide_layout_tip
from .z2_shared_ui import campaign_gate_id_for_edge

try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab

_PREVIEW_SOURCE_IMAGE_CACHE_LIMIT = 64
_PREVIEW_RESIZED_PHOTO_CACHE_LIMIT = 96


def _log_preview_perf(label: str, start_time: float, *, threshold_ms: float = 80.0, **phases) -> float:
    total_ms = (time.perf_counter() - float(start_time)) * 1000.0
    if total_ms < float(threshold_ms):
        return total_ms
    phase_parts = []
    for key, value in phases.items():
        try:
            phase_parts.append(f"{key}={float(value):.1f}ms")
        except Exception:
            phase_parts.append(f"{key}={value}")
    logger.info(
        "[Z3/PZ2 PERF] %s total=%.1fms phases=[%s]",
        str(label),
        total_ms,
        ", ".join(phase_parts) if phase_parts else "no_slow_phase",
    )
    return total_ms


def _preview_flow_value(value) -> str:
    text = str(value if value is not None else "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) > 96:
        text = text[:93] + "..."
    if not text:
        return "-"
    if any(ch.isspace() for ch in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def _preview_char_bbox_text(host: "CharacterAnnotationTab", rec) -> str:
    try:
        bbox = host._char_record_bbox(rec)
        if not bbox:
            return "-"
        return ",".join(str(round(float(v), 1)) for v in bbox[:4])
    except Exception:
        return "-"


def _preview_char_record_trace_fields(host: "CharacterAnnotationTab", rec, *, data=None, fallback_index=None) -> dict:
    if not isinstance(rec, dict):
        return {}
    try:
        source_tag = host._get_character_source_tag(rec, data=data, fallback_index=fallback_index)
    except Exception:
        source_tag = str(rec.get("source_tag", "") or rec.get("method", "") or "")
    try:
        row = int(rec.get("reading_row", 0) or 0)
    except Exception:
        row = 0
    try:
        col = int(rec.get("reading_col", 0) or 0)
    except Exception:
        col = 0
    return {
        "char": str(rec.get("character", "") or ""),
        "src": str(source_tag or "-"),
        "method": str(rec.get("method", "") or "-"),
        "kind": str(rec.get("source_kind", "") or "-"),
        "rowcol": f"{row}.{col}" if row or col else "-",
        "bbox": _preview_char_bbox_text(host, rec),
    }


def _preview_character_source_parts(host: "CharacterAnnotationTab", rec, *, data=None, fallback_index: int = 0) -> tuple[str, str]:
    try:
        box_source = host._get_character_box_source_tag(rec, data=data, fallback_index=fallback_index)
    except Exception:
        box_source = ""
    try:
        sign_source = host._get_character_sign_source_tag(rec, data=data, fallback_index=fallback_index)
    except Exception:
        sign_source = ""
    return (
        str(box_source or "").strip().lower().replace("-", "_"),
        str(sign_source or "").strip().lower().replace("-", "_"),
    )


def _preview_visual_tag_for_box_source(host: "CharacterAnnotationTab", box_source: str, sign_source: str = "") -> str:
    normalized_box = str(box_source or "").strip().lower().replace("-", "_")
    if normalized_box == "manual_box":
        return "manual"
    if normalized_box == "yolo_box":
        return "yolo_box"
    return "ocr"


def log_preview_edit_flow(host: "CharacterAnnotationTab", event: str, **fields) -> None:
    try:
        pid = str(getattr(host, "_preview_active_pid", "") or "")
        data = host._get_preview_active_data(create=False)
        status = str((data or {}).get("status", "") or "") if isinstance(data, dict) else ""
    except Exception:
        pid = ""
        status = ""
    payload = {
        "event": str(event),
        "pid": pid or "-",
        "status": status or "-",
    }
    payload.update(fields)
    parts = [f"{key}={_preview_flow_value(value)}" for key, value in payload.items()]
    logger.info("[Z3/PZ2 FLOW] %s", " ".join(parts))


def start_preview_latency_probe(host: "CharacterAnnotationTab", intent: str, **fields) -> dict:
    try:
        trace = int(getattr(host, "_preview_latency_trace_seq", 0) or 0) + 1
    except Exception:
        trace = 1
    try:
        host._preview_latency_trace_seq = trace
    except Exception:
        pass
    return {
        "trace": int(trace),
        "intent": str(intent),
        "start": time.perf_counter(),
        "fields": dict(fields or {}),
    }


def log_preview_latency(host: "CharacterAnnotationTab", probe, stage: str, **fields) -> None:
    if not isinstance(probe, dict):
        return
    try:
        elapsed_ms = (time.perf_counter() - float(probe.get("start", time.perf_counter()))) * 1000.0
    except Exception:
        elapsed_ms = 0.0
    payload = dict(probe.get("fields", {}) or {})
    payload.update(fields)
    payload.update(
        {
            "trace": probe.get("trace", "-"),
            "intent": probe.get("intent", "-"),
            "elapsed_ms": round(elapsed_ms, 1),
        }
    )
    log_preview_edit_flow(host, f"latency_{stage}", **payload)


def schedule_preview_latency_paint(host: "CharacterAnnotationTab", probe, stage: str, **fields) -> None:
    if not isinstance(probe, dict):
        return
    event_done_at = time.perf_counter()
    try:
        event_done_ms = (event_done_at - float(probe.get("start", event_done_at))) * 1000.0
    except Exception:
        event_done_ms = 0.0

    def _log_after_idle():
        try:
            paint_ms = (time.perf_counter() - float(probe.get("start", event_done_at))) * 1000.0
        except Exception:
            paint_ms = event_done_ms
        payload = dict(probe.get("fields", {}) or {})
        payload.update(fields)
        payload.update(
            {
                "trace": probe.get("trace", "-"),
                "intent": probe.get("intent", "-"),
                "event_done_ms": round(event_done_ms, 1),
                "paint_ms": round(paint_ms, 1),
            }
        )
        log_preview_edit_flow(host, f"latency_{stage}_paint", **payload)

    try:
        host.frame.after_idle(_log_after_idle)
    except Exception:
        _log_after_idle()


def _get_preview_source_cache_lock(host: "CharacterAnnotationTab"):
    lock = getattr(host, "_preview_source_image_cache_lock", None)
    if lock is None or not hasattr(lock, "acquire"):
        lock = threading.RLock()
        host._preview_source_image_cache_lock = lock
    return lock


def _trim_preview_lru_cache(cache, limit: int) -> None:
    if not isinstance(cache, OrderedDict):
        return
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 1
    while len(cache) > limit:
        try:
            cache.popitem(last=False)
        except Exception:
            break


def _get_preview_cache_key(path: Path):
    resolved = Path(path)
    try:
        stat = resolved.stat()
        return (str(resolved), float(stat.st_mtime), int(stat.st_size))
    except Exception:
        return (str(resolved), 0.0, 0)


def _get_cached_preview_source_image(host: "CharacterAnnotationTab", img_path: Path):
    cache = getattr(host, "_preview_source_image_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        host._preview_source_image_cache = cache
    lock = _get_preview_source_cache_lock(host)

    source_key = _get_preview_cache_key(Path(img_path))
    with lock:
        cached = cache.get(source_key)
        if cached is not None:
            try:
                cache.move_to_end(source_key)
            except Exception:
                pass
            return cached[0], int(cached[1]), int(cached[2]), source_key

    with Image.open(img_path) as opened:
        pil_img = opened.copy()
    orig_w, orig_h = pil_img.size
    with lock:
        cached = cache.get(source_key)
        if cached is not None:
            try:
                cache.move_to_end(source_key)
            except Exception:
                pass
            return cached[0], int(cached[1]), int(cached[2]), source_key
        cache[source_key] = (pil_img, int(orig_w), int(orig_h))
        _trim_preview_lru_cache(cache, _PREVIEW_SOURCE_IMAGE_CACHE_LIMIT)
    return pil_img, int(orig_w), int(orig_h), source_key


def _get_cached_preview_photo(
    host: "CharacterAnnotationTab",
    source_key,
    pil_img,
    width: int,
    height: int,
):
    width = max(1, int(width))
    height = max(1, int(height))
    cache = getattr(host, "_preview_resized_photo_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        host._preview_resized_photo_cache = cache

    photo_key = (source_key, int(width), int(height))
    cached = cache.get(photo_key)
    if cached is not None:
        try:
            cache.move_to_end(photo_key)
        except Exception:
            pass
        return cached

    resized = pil_img.resize((width, height), Image.Resampling.BILINEAR)
    photo = ImageTk.PhotoImage(resized)
    cache[photo_key] = photo
    _trim_preview_lru_cache(cache, _PREVIEW_RESIZED_PHOTO_CACHE_LIMIT)
    return photo


def _prefetch_preview_neighbor_sources(host: "CharacterAnnotationTab", image_paths: list[Path]) -> None:
    for img_path in image_paths:
        try:
            if not Path(img_path).exists():
                continue
            _get_cached_preview_source_image(host, Path(img_path))
        except Exception:
            continue


def _collect_preview_neighbor_image_paths(host: "CharacterAnnotationTab", center_pid: str | None = None) -> list[Path]:
    try:
        pid_map = list(getattr(host, "_listbox_pid_by_index", []) or [])
    except Exception:
        pid_map = []
    if not pid_map:
        return []
    active_pid = str(center_pid or getattr(host, "_preview_active_pid", "") or "").strip()
    if not active_pid:
        return []
    try:
        center_idx = pid_map.index(active_pid)
    except ValueError:
        return []

    try:
        preview_root = Path(str(host.preview_dir_var.get() or "").strip()) / "images"
    except Exception:
        return []
    if not preview_root.exists():
        return []

    image_paths: list[Path] = []
    for neighbor_idx in (center_idx + 1, center_idx - 1):
        if not (0 <= int(neighbor_idx) < len(pid_map)):
            continue
        neighbor_pid = str(pid_map[int(neighbor_idx)] or "").strip()
        if not neighbor_pid:
            continue
        img_path = preview_root / f"{neighbor_pid}.jpg"
        if not img_path.exists():
            continue
        image_paths.append(img_path)
    return image_paths


def _schedule_preview_neighbor_prefetch(host: "CharacterAnnotationTab", center_pid: str | None = None, *, delay_ms: int = 120) -> None:
    if not bool(getattr(host, "_preview_neighbor_prefetch_enabled", True)):
        try:
            pending_after = getattr(host, "_preview_neighbor_prefetch_after_id", None)
            if pending_after:
                host.frame.after_cancel(pending_after)
                host._preview_neighbor_prefetch_after_id = None
        except Exception:
            pass
        return

    try:
        pending_after = getattr(host, "_preview_neighbor_prefetch_after_id", None)
        if pending_after:
            host.frame.after_cancel(pending_after)
    except Exception:
        pass

    active_pid = str(center_pid or getattr(host, "_preview_active_pid", "") or "").strip()

    def _prefetch_once():
        try:
            host._preview_neighbor_prefetch_after_id = None
        except Exception:
            pass
        if active_pid and str(getattr(host, "_preview_active_pid", "") or "").strip() != active_pid:
            return
        if getattr(host, "_preview_list_select_after_id", None):
            return
        image_paths = _collect_preview_neighbor_image_paths(host, active_pid)
        if not image_paths:
            return
        try:
            threading.Thread(
                target=_prefetch_preview_neighbor_sources,
                args=(host, image_paths),
                daemon=True,
            ).start()
        except Exception:
            _prefetch_preview_neighbor_sources(host, image_paths)

    try:
        host._preview_neighbor_prefetch_after_id = host.frame.after(max(1, int(delay_ms)), _prefetch_once)
    except Exception:
        _prefetch_once()


def clamp_preview_image_position(
    x_off: float,
    y_off: float,
    *,
    image_w: float,
    image_h: float,
    canvas_w: float,
    canvas_h: float,
    top_reserved: float,
    bottom_reserved: float = 0.0,
    preferred_x: float | None = None,
    preferred_y: float | None = None,
):
    min_visible = 48.0
    canvas_w = max(min_visible, float(canvas_w))
    canvas_h = max(min_visible, float(canvas_h))
    image_w = max(1.0, float(image_w))
    image_h = max(1.0, float(image_h))
    top_reserved = max(0.0, float(top_reserved))
    bottom_reserved = max(0.0, float(bottom_reserved))
    available_h = max(min_visible, canvas_h - top_reserved - bottom_reserved)

    if image_w <= canvas_w:
        min_x = 0.0
        max_x = canvas_w - image_w
    else:
        min_x = float(min_visible) - image_w
        max_x = canvas_w - float(min_visible)
        if max_x < min_x:
            center_x = float(preferred_x) if preferred_x is not None else (canvas_w - image_w) / 2.0
            min_x = center_x
            max_x = center_x

    if image_h <= available_h:
        min_y = top_reserved
        max_y = canvas_h - bottom_reserved - image_h
    else:
        min_y = top_reserved + float(min_visible) - image_h
        max_y = canvas_h - bottom_reserved - float(min_visible)
        if max_y < min_y:
            fallback_y = float(preferred_y) if preferred_y is not None else min_y
            min_y = fallback_y
            max_y = fallback_y

    x_off = max(min_x, min(max_x, float(x_off)))
    y_off = max(min_y, min(max_y, float(y_off)))
    return float(x_off), float(y_off)


def get_preview_image_bottom_reserved(host: "CharacterAnnotationTab", data: dict | None = None) -> float:
    """Reserve canvas space for labels rendered below two-row plates."""
    try:
        source = data if isinstance(data, dict) else host._get_preview_active_data(create=False)
    except Exception:
        source = data if isinstance(data, dict) else {}
    try:
        two_row_active = bool(host._should_preview_use_two_row_layers(source if isinstance(source, dict) else {}))
    except Exception:
        try:
            two_row_active = bool(host._is_preview_two_row_layout_active(source if isinstance(source, dict) else {}))
        except Exception:
            two_row_active = False
    return 72.0 if two_row_active else 18.0


def select_preview_character_box(
    host,
    char_idx: int | None,
    *,
    activate_label: bool = False,
    status_message: str | None = None,
):
    perf_start = time.perf_counter()
    ensure_ms = toolbar_ms = status_ms = visual_ms = 0.0
    chars = host._get_preview_active_character_records(create=False)
    if not isinstance(chars, list) or not chars:
        host._preview_char_selected_index = None
        host._preview_char_label_active_index = None
        host._preview_char_hover_label_index = None
        host._refresh_preview_editor_toolbar()
        host._update_preview_edit_status("Brak boxów znaków do zaznaczenia.", tone="warning")
        host._on_preview_select(None)
        return "break"

    try:
        safe_idx = max(0, min(int(char_idx), len(chars) - 1))
    except Exception:
        host._update_preview_edit_status("Nie udało się zaznaczyć boxu znaku.", tone="warning")
        return "break"

    try:
        host._preview_last_char_edit_interaction_ts = time.monotonic()
    except Exception:
        pass

    phase_start = time.perf_counter()
    host._ensure_preview_final_box_mode()
    ensure_ms = (time.perf_counter() - phase_start) * 1000.0
    previous_selected_idx = getattr(host, "_preview_char_selected_index", None)
    previous_hover_idx = getattr(host, "_preview_char_hover_index", None)
    previous_hover_label_idx = getattr(host, "_preview_char_hover_label_index", None)
    previous_active_label_idx = getattr(host, "_preview_char_label_active_index", None)
    host._preview_char_hover_grip = None
    label_mode_active = bool(getattr(host, "_preview_char_label_mode", False))
    host._preview_char_edit_mode = False if label_mode_active else True
    host._preview_char_add_mode = False
    host._preview_char_add_modifier_down = False
    host._preview_char_add_click_armed = False
    host._preview_char_add_state = None
    host._preview_char_selected_index = safe_idx
    host._preview_char_hover_index = safe_idx
    label_target_active = bool(activate_label or label_mode_active)
    host._preview_char_hover_label_index = safe_idx if label_target_active else None
    host._preview_char_label_active_index = safe_idx if label_target_active else None
    phase_start = time.perf_counter()
    host._refresh_preview_editor_toolbar()
    toolbar_ms = (time.perf_counter() - phase_start) * 1000.0
    phase_start = time.perf_counter()
    if status_message or label_target_active:
        host._update_preview_edit_status(
            status_message or "Tryb wpisywania znaków jest aktywny. Kliknij kolejny box LPM albo użyj strzałek lewo/prawo, a potem wpisz znak.",
            tone="info",
        )
    else:
        # Hot path: selecting boxes must only update selection visuals, not reflow the instruction overlay.
        quick_message = "Box znaku jest aktywny. Chwyć środek albo narożnik."
        try:
            if hasattr(host, "preview_edit_status_var") and str(host.preview_edit_status_var.get() or "") != quick_message:
                host.preview_edit_status_var.set(quick_message)
        except Exception:
            pass
        try:
            overlay = getattr(host, "preview_typing_overlay", None)
            if overlay is not None:
                overlay.place_forget()
            host._preview_typing_overlay_text = ""
        except Exception:
            pass
    status_ms = (time.perf_counter() - phase_start) * 1000.0
    host._focus_preview_canvas()
    affected_indices = {
        idx for idx in (
            previous_selected_idx,
            previous_hover_idx,
            previous_hover_label_idx,
            previous_active_label_idx,
            safe_idx,
        )
        if idx is not None
    }
    phase_start = time.perf_counter()
    if not host._refresh_preview_character_selection_visual(affected_indices):
        host._on_preview_select(None)
    visual_ms = (time.perf_counter() - phase_start) * 1000.0
    _log_preview_perf(
        "char_select",
        perf_start,
        threshold_ms=55.0,
        ensure=ensure_ms,
        toolbar=toolbar_ms,
        status=status_ms,
        visual=visual_ms,
        affected=len(affected_indices),
    )
    selected_rec = chars[safe_idx] if 0 <= int(safe_idx) < len(chars) else None
    trace_fields = _preview_char_record_trace_fields(host, selected_rec, data=host._get_preview_active_data(create=False), fallback_index=safe_idx)
    log_preview_edit_flow(
        host,
        "char_select",
        idx=safe_idx,
        activate_label=int(bool(label_target_active)),
        chars=len(chars),
        **trace_fields,
    )
    probe = getattr(host, "_preview_pending_select_latency_probe", None)
    if isinstance(probe, dict):
        host._preview_pending_select_latency_probe = None
        log_preview_latency(
            host,
            probe,
            "select_event_done",
            idx=safe_idx,
            visual_ms=round(visual_ms, 1),
            status_ms=round(status_ms, 1),
        )
        schedule_preview_latency_paint(
            host,
            probe,
            "select_ready",
            idx=safe_idx,
        )
    return "break"


def serialize_character_records(host, chars, fusion_strategy="", fusion_details=None, data=None):
    ordered = host._annotate_preview_character_reading_positions(
        host._sort_character_records_by_x(list(chars or []), data=data),
        data=data,
    )
    source_tags = host._build_character_source_tags(
        ordered,
        fusion_strategy=fusion_strategy,
        fusion_details=fusion_details,
    )
    clean = []

    for idx, char in enumerate(ordered):
        legacy_source_tag = source_tags[idx] if idx < len(source_tags) else host._normalize_character_source_tag(
            method=(char.get("method", "") if isinstance(char, dict) else getattr(char, "method", ""))
        )
        box_source, sign_source = _preview_character_source_parts(host, char, data=data, fallback_index=idx)
        source_tag = host._compose_character_source_tag(box_source, sign_source)
        source_kind = host._normalize_character_source_kind(char)
        if source_tag != legacy_source_tag:
            if source_tag == "manual":
                source_kind = "local_manual"
            elif source_tag in {"yolo", "yolo_box", "yolo_symbol", "yolo_box_ocr", "yolo_rescue", "ocr"}:
                source_kind = source_tag
        record = {
            "character": str(char["character"] if isinstance(char, dict) else char.character),
            "bbox": [float(x) for x in (char["bbox"] if isinstance(char, dict) else char.bbox)],
            "confidence": float(char["confidence"] if isinstance(char, dict) else char.confidence),
            "method": str(char["method"] if isinstance(char, dict) else char.method),
            "source_tag": str(source_tag),
            "source_kind": str(source_kind),
            "box_source": str(box_source),
            "sign_source": str(sign_source),
            "source_batch_id": str((char.get("source_batch_id", "") if isinstance(char, dict) else getattr(char, "source_batch_id", "")) or ""),
        }
        for field_name in ("reading_row", "reading_col", "reading_index"):
            try:
                raw_value = char.get(field_name, None) if isinstance(char, dict) else getattr(char, field_name, None)
                if raw_value is not None:
                    record[field_name] = int(raw_value)
            except Exception:
                pass
        if (
            box_source == "yolo_box"
            or host._character_record_uses_yolo_box_backend(char)
            or (
                str(source_tag) == "yolo_box_ocr"
                and idx in host._fusion_details_yolo_box_backend_positions(fusion_details)
            )
        ):
            record["box_backend"] = "yolo"
            record["box_backend_source"] = str(
                (char.get("box_backend_source", "") if isinstance(char, dict) else getattr(char, "box_backend_source", ""))
                or "yolo_filtered"
            )
            record["geometry_source"] = "yolo"
            record["geometry_method"] = "yolo_box"
            backend_confidence = host._character_record_yolo_box_backend_confidence(char)
            if backend_confidence is not None:
                record["box_backend_confidence"] = float(backend_confidence)
        clean.append(record)

    return clean


def get_plate_listbox_source_flags(host, data: dict | None) -> list[str]:
    if not isinstance(data, dict):
        return []

    flags = set()
    source_bucket = host._get_plate_source_bucket(data)
    if source_bucket in {"local_manual", "cvat_manual"}:
        flags.add("M")

    for idx, rec in enumerate(list(data.get("characters", []) or [])):
        box_source, sign_source = _preview_character_source_parts(host, rec, data=data, fallback_index=idx)
        source_kind = host._get_character_source_kind(rec, data=data)
        source_tag = host._get_character_source_tag(rec, data=data, fallback_index=idx)
        method_name = str(
            rec.get("method", "") if isinstance(rec, dict) else getattr(rec, "method", "")
        ).strip().lower()
        uses_yolo_box_backend = host._character_record_uses_yolo_box_backend(rec)

        if box_source == "manual_box":
            flags.add("MB")
        elif box_source == "yolo_box":
            flags.add("YB")
        elif box_source == "generated_box":
            flags.add("GB")

        if sign_source == "manual_sign":
            flags.add("MS")
        elif sign_source == "yolo_symbol":
            flags.add("YS")
        elif sign_source == "ocr_symbol":
            flags.add("OS")

        if source_kind in {"local_manual", "cvat_manual"} or source_tag == "manual" or method_name in {"manual", "cvat_manual"}:
            flags.add("M")

        if (
            source_tag in {"ocr", "yolo_box_ocr", "yolo_rescue"}
            or source_kind in {"ocr", "yolo_box_ocr", "yolo_rescue"}
            or method_name in {"ocr", "yolo_ocr"}
        ):
            flags.add("O")

        if (
            uses_yolo_box_backend
            or source_tag in {"yolo", "yolo_box", "yolo_box_ocr"}
            or source_kind in {"yolo", "yolo_box", "yolo_box_ocr"}
            or method_name in {"yolo_ocr"}
        ):
            flags.add("YB")

        if (
            source_tag in {"yolo", "yolo_symbol", "yolo_rescue"}
            or source_kind in {"yolo", "yolo_symbol", "yolo_rescue"}
            or (
                method_name == "yolo"
                and source_tag not in {"yolo_box", "yolo_box_ocr"}
                and source_kind not in {"yolo_box", "yolo_box_ocr"}
            )
            or method_name == "yolo_symbol"
        ):
            flags.add("YS")

    if flags.intersection({"M", "MB", "MS"}):
        flags.add("MANUAL")
        flags.discard("M")
    if "OS" in flags:
        flags.discard("O")
    preferred_order = ("MANUAL", "MB", "MS", "GB", "OS", "YB", "YS", "O")
    return [flag for flag in preferred_order if flag in flags]


def rebuild_preview_listbox(
    host: "CharacterAnnotationTab",
    preserve_selection: bool = True,
    schedule_render: bool = True,
) -> None:
    selected_pid = None

    if preserve_selection:
        try:
            sel = host.plates_listbox.curselection()
            if sel:
                idx = sel[0]
                if 0 <= idx < len(host._listbox_pid_by_index):
                    selected_pid = host._listbox_pid_by_index[idx]
        except Exception:
            selected_pid = None

    current_order = [pid for pid in host._preview_base_plate_ids if pid in host.preview_metadata]
    current_ids = set(current_order)
    appended = [pid for pid in host.preview_metadata.keys() if pid not in current_ids]
    host._preview_base_plate_ids = current_order + appended
    visible_plate_ids = list(host._preview_base_plate_ids)
    if bool(getattr(host, "_preview_import_focus_active", False)):
        imported_set = {
            str(pid or "").strip()
            for pid in list(getattr(host, "_preview_import_focus_plate_ids", []) or [])
            if str(pid or "").strip()
        }
        visible_plate_ids = [pid for pid in visible_plate_ids if pid in imported_set]
    if not visible_plate_ids and host.preview_metadata:
        if bool(getattr(host, "_preview_import_focus_active", False)):
            host._preview_import_focus_active = False
        visible_plate_ids = list(host._preview_base_plate_ids)

    host.preview_plate_ids = list(visible_plate_ids)
    host._listbox_pid_by_index = host._get_sorted_preview_plate_ids(host.preview_plate_ids)
    host._refresh_preview_import_focus_ui()

    host._reloading_preview = True
    try:
        host.plates_listbox.delete(0, tk.END)
        row_statuses: list[str] = []
        labels = []
        defer_row_styles = len(host._listbox_pid_by_index) > 80
        for idx, pid in enumerate(host._listbox_pid_by_index):
            data = host.preview_metadata.get(pid, {})
            label = host._format_plate_listbox_label(pid, data, ordinal=idx + 1)
            status = str(data.get("status", "unknown")).strip().lower()

            labels.append(label)
            row_statuses.append(status)

        if labels:
            host.plates_listbox.insert(tk.END, *labels)
        if not defer_row_styles:
            for idx, status in enumerate(row_statuses):
                host._apply_plate_listbox_row_style(idx, status)

        if defer_row_styles:
            style_token = object()
            host._preview_listbox_style_token = style_token

            def _style_rows_chunk(start: int = 0) -> None:
                if getattr(host, "_preview_listbox_style_token", None) is not style_token:
                    return
                end = min(start + 80, len(row_statuses))
                for row_idx in range(start, end):
                    host._apply_plate_listbox_row_style(row_idx, row_statuses[row_idx])
                if end < len(row_statuses):
                    try:
                        host.frame.after(1, lambda next_start=end: _style_rows_chunk(next_start))
                    except Exception:
                        _style_rows_chunk(end)

            try:
                host.frame.after(40, _style_rows_chunk)
            except Exception:
                _style_rows_chunk()

        restore_idx = None
        if selected_pid and selected_pid in host._listbox_pid_by_index:
            restore_idx = host._listbox_pid_by_index.index(selected_pid)
        elif host._listbox_pid_by_index:
            restore_idx = 0

        if restore_idx is not None:
            host._suppress_preview_reload_on_list_select = True
            host._preview_fast_select_render = True
            host._clear_listbox_selection_fast(host.plates_listbox)
            host.plates_listbox.selection_set(restore_idx)
            host.plates_listbox.activate(restore_idx)
            host.plates_listbox.see(restore_idx)
            if schedule_render:
                try:
                    scheduler = getattr(host, "_schedule_preview_select_render", None)
                    if callable(scheduler):
                        scheduler(delay_ms=60)
                    else:
                        host.frame.after_idle(lambda: host._on_preview_select(None))
                except Exception:
                    pass
            else:
                def _clear_select_suppression() -> None:
                    try:
                        host._suppress_preview_reload_on_list_select = False
                        host._preview_fast_select_render = False
                    except Exception:
                        pass

                try:
                    host.frame.after_idle(_clear_select_suppression)
                except Exception:
                    _clear_select_suppression()
        else:
            host._update_preview_record_source_label(None)

        host._update_preview_info_label()

    finally:
        host._reloading_preview = False


def format_preview_source_counts_line(host: "CharacterAnnotationTab", source_counts: dict | None) -> str:
    counts = source_counts if isinstance(source_counts, dict) else {}
    return (
        f"Ramki MB={int(counts.get('manual_box', 0) or 0)} | "
        f"GB={int(counts.get('generated_box', 0) or 0)} | "
        f"YB={int(counts.get('yolo_box', 0) or 0)} | "
        f"Znaki MS={int(counts.get('manual_sign', 0) or 0)} | "
        f"OS={int(counts.get('ocr_symbol', 0) or 0)} | "
        f"YS={int(counts.get('yolo_symbol', 0) or 0)}"
    )


def preview_record_has_symbol(host: "CharacterAnnotationTab", rec) -> bool | None:
    if isinstance(rec, dict):
        symbol = rec.get("character", "")
    else:
        symbol = getattr(rec, "character", "")
    normalized = host._sanitize_preview_char_symbol(symbol)
    if normalized:
        return True
    return False


def get_preview_status_presentation(
    host: "CharacterAnnotationTab",
    data: dict | None = None,
    chars=None,
    plate_id: str | None = None,
) -> dict:
    source_data = data if isinstance(data, dict) else host._get_preview_active_data(create=False)
    source_chars = chars if isinstance(chars, list) else (
        source_data.get("characters", []) if isinstance(source_data, dict) else []
    )
    status = host._get_preview_live_status(data=source_data, chars=source_chars, plate_id=plate_id)
    expected_resolution = host._resolve_preview_expected_text_for_crop(
        source_data if isinstance(source_data, dict) else None,
        source_chars,
    )
    expected_texts = list(expected_resolution.get("expected_texts", []) or [])
    target_len = int(expected_resolution.get("target_length", 0) or 0)
    target_lengths = [
        int(value)
        for value in list(expected_resolution.get("target_lengths", []) or [])
        if int(value or 0) > 0
    ]
    count_resolved = bool(expected_resolution.get("count_resolved"))

    total_boxes = len(source_chars) if isinstance(source_chars, list) else 0
    filled_boxes = 0
    if isinstance(source_chars, list):
        for rec in source_chars:
            symbol = ""
            if isinstance(rec, dict):
                symbol = rec.get("character", "")
            else:
                symbol = getattr(rec, "character", "")
            if host._sanitize_preview_char_symbol(symbol):
                filled_boxes += 1

    candidate_text = host._characters_to_text(source_chars, data=source_data).strip().upper()
    severity = "muted"
    canvas_text = "Nieocenione"
    info_text = "status: nieoceniona"

    if status == "perfect":
        severity = "success"
        canvas_text = "Perfect"
        info_text = "status: OK"
    elif total_boxes <= 0:
        severity = "warning"
        canvas_text = "Brak ramek"
        info_text = "status: brak ramek znaków"
    elif filled_boxes < total_boxes:
        severity = "warning"
        canvas_text = f"Znaki {filled_boxes}/{total_boxes}"
        info_text = f"status: wpisane znaki {filled_boxes}/{total_boxes}"
    elif expected_texts:
        if target_len > 0 and int(total_boxes) != int(target_len):
            severity = "warning"
            canvas_text = f"Ramki {total_boxes}/{int(target_len)}"
            info_text = f"status: liczba ramek znaków {total_boxes}/{int(target_len)}"
        elif candidate_text:
            severity = "error"
            canvas_text = "Do korekty"
            info_text = "status: wymaga korekty"
        elif not count_resolved and target_lengths:
            target_label = " lub ".join(str(value) for value in target_lengths)
            severity = "warning"
            canvas_text = f"Ramki {total_boxes}/{target_label}"
            info_text = f"status: oczekiwane ramki {target_label}"
        else:
            severity = "warning"
            canvas_text = "Brak znaków"
            info_text = "status: brak znaków"
    elif candidate_text:
        severity = "warning"
        canvas_text = "Bez wzorca"
        info_text = "status: brak wzorca"

    return {
        "status": status,
        "severity": severity,
        "canvas_text": canvas_text,
        "info_text": info_text,
        "candidate_text": candidate_text,
        "expected_texts": list(expected_texts or []),
        "expected_resolution": dict(expected_resolution or {}),
        "target_length": int(target_len or 0),
        "target_lengths": list(target_lengths or []),
        "count_resolved": bool(count_resolved),
        "filled_boxes": int(filled_boxes),
        "total_boxes": int(total_boxes),
    }


def _build_preview_canvas_status_badge_specs(
    host: "CharacterAnnotationTab",
    canvas_width: int,
    *,
    data: dict | None = None,
    box_chars=None,
) -> dict:
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_border = palette.get("border", "#3c3c3c")
    muted_fg = palette.get("muted", "#8b949e")
    success_fg = palette.get("success", "#2ecc71")
    error_fg = palette.get("error", "#e74c3c")

    source_data = data if isinstance(data, dict) else {}
    source_chars = box_chars if isinstance(box_chars, list) else source_data.get("characters", [])
    if not isinstance(source_chars, list):
        source_chars = []

    status_meta = host._get_preview_status_presentation(
        data=source_data,
        chars=source_chars,
        plate_id=str(source_data.get("plate_id", "") or getattr(host, "_preview_active_pid", "") or ""),
    )
    live_status = str(status_meta.get("status", "") or "").strip().lower()
    expected_texts = [
        str(text or "").strip().upper()
        for text in status_meta.get("expected_texts", [])
        if str(text or "").strip()
    ]
    candidate_text = str(status_meta.get("candidate_text", "") or "").strip().upper()

    try:
        display_rows = host._characters_to_display_rows(source_chars, data=source_data)
    except Exception:
        display_rows = []
    display_rows = [str(row or "").strip().upper() for row in display_rows if str(row or "").strip()]
    if not display_rows:
        display_rows = [candidate_text] if candidate_text else []
    reading_text = " / ".join(display_rows) if display_rows else "brak"
    two_row_display = len(display_rows) >= 2

    def _valid_frame_count(records) -> int:
        count = 0
        for rec in records if isinstance(records, list) else []:
            bbox = rec.get("bbox") if isinstance(rec, dict) else getattr(rec, "bbox", None)
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                count += 1
        return int(count)

    frame_count = _valid_frame_count(source_chars)

    reading_ok = bool(live_status == "perfect" or (candidate_text and expected_texts and candidate_text in expected_texts))
    status_ok = bool(live_status == "perfect")

    try:
        layout_text, layout_tone = host._get_preview_plate_layout_dock_text(source_data)
    except Exception:
        layout_text, layout_tone = "AUTO ?", "muted"
    normalized_layout = str(layout_text or "").strip().upper()
    try:
        layout_conflict = bool(host._preview_layout_separator_conflicts_with_chars(source_data, source_chars))
    except Exception:
        layout_conflict = False
    layout_uncertain = (
        not normalized_layout
        or "?" in normalized_layout
        or str(layout_tone or "").strip().lower() in {"error", "muted"}
    )
    layout_ok = bool(status_ok or (not layout_uncertain and not layout_conflict))

    status_text = "kompletne" if status_ok else "do korekty"
    states = [
        {
            "text": f"Odczyt: [{reading_text}]",
            "ok": reading_ok,
            "width": max(156.0, min(232.0, float(canvas_width) * 0.26)),
            "tags": ("preview_overlay",),
        },
        {
            "text": f"Układ: {layout_text}",
            "ok": layout_ok,
            "width": max(124.0, min(174.0, float(canvas_width) * 0.17)),
            "tags": ("preview_overlay", "preview_overlay_action", "preview_action::toggle_plate_layout"),
        },
        {
            "text": f"Status tablicy: {status_text}",
            "ok": status_ok,
            "width": max(184.0, min(252.0, float(canvas_width) * 0.25)),
            "tags": ("preview_overlay",),
        },
        {
            "text": f"Ramki: {int(frame_count)}",
            "ok": None,
            "neutral": True,
            "width": max(124.0, min(168.0, float(canvas_width) * 0.16)),
            "tags": ("preview_overlay",),
        },
    ]
    evaluated_states = [item for item in states if not bool(item.get("neutral"))]
    pulse_red = bool(
        evaluated_states
        and any(bool(item["ok"]) for item in evaluated_states)
        and not all(bool(item["ok"]) for item in evaluated_states)
    )
    pulse_on = bool(int(time.time() * 2.0) % 2 == 0)

    badges = []
    neutral_badges = []
    for item in states:
        is_neutral = bool(item.get("neutral"))
        is_ok = bool(item["ok"])
        if is_neutral:
            base_color = muted_fg
            fill = blend_hex_colors(panel_border, panel_bg, 0.28)
        elif is_ok:
            base_color = success_fg
            fill = blend_hex_colors(base_color, panel_bg, 0.26)
        elif pulse_red:
            base_color = error_fg
            fill = blend_hex_colors(base_color, panel_bg, 0.10 if pulse_on else 0.34)
        else:
            base_color = error_fg
            fill = blend_hex_colors(base_color, panel_bg, 0.24)
        badge_spec = {
            "text": str(item["text"]),
            "fill": fill,
            "outline": base_color,
            "width": float(item["width"]),
            "tags": item.get("tags", ("preview_overlay",)),
            "pulse": bool((not is_neutral) and (not is_ok) and pulse_red),
        }
        if is_neutral:
            neutral_badges.append(badge_spec)
        else:
            badges.append(badge_spec)

    return {
        "badges": badges,
        "neutral_badges": neutral_badges,
        "pulse": any(bool(item.get("pulse")) for item in badges),
        "two_row_display": bool(two_row_display),
    }


def _sync_preview_canvas_status_pulse(
    host: "CharacterAnnotationTab",
    canvas,
    pulse_needed: bool,
) -> None:
    after_id = getattr(host, "_preview_status_pulse_after_id", None)
    if not pulse_needed:
        if after_id:
            try:
                canvas.after_cancel(after_id)
            except Exception:
                pass
            setattr(host, "_preview_status_pulse_after_id", None)
        return
    if after_id:
        return

    def _tick():
        setattr(host, "_preview_status_pulse_after_id", None)
        try:
            if canvas is not None and bool(canvas.winfo_exists()) and getattr(host, "_preview_render_state", None):
                refresh_preview_canvas_info_overlay_only(host)
        except tk.TclError:
            return
        except Exception:
            return

    try:
        setattr(host, "_preview_status_pulse_after_id", canvas.after(500, _tick))
    except Exception:
        setattr(host, "_preview_status_pulse_after_id", None)


def get_preview_step3_gate_overlay_state(host: "CharacterAnnotationTab") -> dict:
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False
    if not in_campaign:
        return {"visible": False}

    try:
        readiness = host._get_campaign_step3_annotation_readiness()
    except Exception:
        readiness = {}

    try:
        graph_context = dict(getattr(host, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    if not graph_gate_id:
        graph_gate_id = "T05"
    graph_gate_label = str(graph_context.get("graph_gate_label") or "").strip()

    annotation_ready = bool(readiness.get("ok"))
    dataset_gate = {}
    if graph_gate_id == "T05":
        try:
            campaign_tab = getattr(getattr(host, "app", None), "tabs", {}).get("campaign")
            getter = getattr(campaign_tab, "_detect_campaign_char_ready_dataset_state", None)
            if callable(getter):
                dataset_gate = dict(getter() or {})
        except Exception:
            dataset_gate = {}
    dataset_ready = bool(dataset_gate.get("ok")) if graph_gate_id == "T05" else False
    ready = dataset_ready if graph_gate_id == "T05" else annotation_ready
    try:
        perfect_count = int(readiness.get("perfect_count", 0) or 0)
    except Exception:
        perfect_count = 0
    try:
        exportable_plate_count = int(readiness.get("exportable_plate_count", 0) or 0)
    except Exception:
        exportable_plate_count = 0
    try:
        exportable_char_count = int(readiness.get("exportable_char_count", 0) or 0)
    except Exception:
        exportable_char_count = 0
    try:
        min_exportable_plate_count = int(readiness.get("min_exportable_plate_count", 10) or 10)
    except Exception:
        min_exportable_plate_count = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    try:
        missing_exportable_plate_count = int(
            readiness.get(
                "missing_exportable_plate_count",
                max(0, int(min_exportable_plate_count) - int(exportable_plate_count or 0)),
            )
            or 0
        )
    except Exception:
        missing_exportable_plate_count = max(0, int(min_exportable_plate_count) - int(exportable_plate_count or 0))

    quality_plate_count = max(0, int(perfect_count), int(exportable_plate_count))
    if graph_gate_id == "T05":
        quality_plate_count = max(0, int(exportable_plate_count))
    quality_info = CONFIG.describe_yolo_char_dataset_quality(
        perfect_plates=quality_plate_count,
        char_boxes=exportable_char_count,
    )
    quality_text = str(quality_info.get("label", "SŁABY") or "SŁABY")
    quality_tone = str(quality_info.get("tone", "error") or "error").strip().lower()
    quality_have_text = f"Tablice: {quality_plate_count}\nZnaki: {int(exportable_char_count)}"
    next_quality_label = str(quality_info.get("next_label", "KOLEJNY") or "KOLEJNY").strip().upper()
    quality_missing_plates = int(quality_info.get("missing_next_plates", 0) or 0)
    quality_missing_boxes = int(quality_info.get("missing_next_boxes", 0) or 0)
    quality_missing_text = f"Do {next_quality_label}\nTablice: {quality_missing_plates}\nZnaki: {quality_missing_boxes}"
    quality_missing_ready = bool(quality_missing_plates <= 0 and quality_missing_boxes <= 0)
    pz2_base_text = "GOTOWE" if annotation_ready else "UZUPEŁNIJ"
    pz2_base_tone = "success" if annotation_ready else "error"
    pz3_dataset_text = "GOTOWY" if dataset_ready else ("UTWÓRZ" if annotation_ready else "PO PZ2")
    pz3_dataset_tone = "success" if dataset_ready else ("warning" if annotation_ready else "muted")
    if graph_gate_id == "T05" and not annotation_ready:
        quality_text = "PONIŻEJ MINIMUM"
        quality_tone = "error"
        quality_missing_plates = max(0, int(missing_exportable_plate_count))
        quality_missing_boxes = 0 if int(exportable_char_count) > 0 else 1
        if quality_missing_plates > 0:
            quality_missing_text = f"Do MINIMUM\nTablice: {quality_missing_plates}\nZnaki: {quality_missing_boxes}"
            pz2_base_text = f"BRAKUJE {quality_missing_plates}"
        else:
            quality_missing_text = "Do MINIMUM\nRamki znaków"
            pz2_base_text = "UZUPEŁNIJ ZNAKI"
        quality_missing_ready = False
    gate_missing_text = "OK"
    gate_missing_tone = "success"
    gate_missing_ready = bool(ready)
    if not ready:
        gate_missing_ready = False
        if not annotation_ready:
            if missing_exportable_plate_count > 0:
                gate_missing_text = f"{missing_exportable_plate_count} tablic"
            elif exportable_char_count <= 0:
                gate_missing_text = "ramki znaków"
            else:
                gate_missing_text = "zakres PZ3"
            gate_missing_tone = "error"
        elif graph_gate_id == "T05":
            gate_missing_text = "dataset PZ3"
            gate_missing_tone = "warning"
        else:
            gate_missing_text = "zatwierdzenie"
            gate_missing_tone = "warning"

    return {
        "visible": True,
        "gate_id": graph_gate_id,
        "gate_label": graph_gate_label,
        "ready": ready,
        "gate_text": "OTWARTA" if ready else "ZAMKNIĘTA",
        "gate_tone": "success" if ready else "error",
        "export_condition_text": pz2_base_text if graph_gate_id == "T05" else ("OK" if annotation_ready else "BRAK"),
        "export_condition_tone": pz2_base_tone if graph_gate_id == "T05" else ("success" if annotation_ready else "error"),
        "quality_text": quality_text,
        "quality_tone": quality_tone,
        "quality_have": quality_have_text,
        "quality_missing": quality_missing_text,
        "quality_missing_ready": quality_missing_ready,
        "gate_missing": gate_missing_text,
        "gate_missing_tone": gate_missing_tone,
        "gate_missing_ready": gate_missing_ready,
        "pz3_dataset_text": pz3_dataset_text,
        "pz3_dataset_tone": pz3_dataset_tone,
        "pz3_dataset_ready": bool(dataset_ready),
        "quality_ranges": str(quality_info.get("range_text", "") or "").strip(),
        "perfect_count": int(perfect_count),
        "exportable_plate_count": int(exportable_plate_count),
        "exportable_char_count": int(exportable_char_count),
        "min_exportable_plate_count": int(min_exportable_plate_count),
        "missing_exportable_plate_count": int(missing_exportable_plate_count),
    }


def _estimate_preview_canvas_info_badges_bottom(
    host: "CharacterAnnotationTab",
    canvas_width: int,
    *,
    data: dict | None = None,
    box_chars=None,
) -> float:
    canvas_w = max(40.0, float(canvas_width or 0.0))
    row_start_x = 10.0
    row_gap_x = 8.0
    row_gap_y = 30.0
    row_right_limit = max(row_start_x + 80.0, canvas_w - 10.0)
    row_y = 40.0

    widths = [
        88.0,
        max(160.0, min(240.0, canvas_w * 0.25)),
    ]
    try:
        status_layout = _build_preview_canvas_status_badge_specs(
            host,
            canvas_width,
            data=data,
            box_chars=box_chars,
        )
        widths.extend(float(item.get("width", 100.0)) for item in status_layout.get("neutral_badges", []))
        widths.extend(float(item.get("width", 100.0)) for item in status_layout.get("badges", []))
    except Exception:
        widths.extend(
            [
                max(124.0, min(168.0, canvas_w * 0.16)),
                max(156.0, min(232.0, canvas_w * 0.26)),
                max(124.0, min(174.0, canvas_w * 0.17)),
                max(184.0, min(252.0, canvas_w * 0.25)),
            ]
        )

    badge_x = row_start_x
    badge_y = row_y
    for badge_width in widths:
        width = float(badge_width)
        if badge_x > row_start_x and (badge_x + width) > row_right_limit:
            badge_x = row_start_x
            badge_y += row_gap_y
        badge_x += width + row_gap_x

    badge_height = max(24.0, host._measure_preview_overlay_font_height(("Segoe UI", 9, "bold")) + 8.0)
    bottom = badge_y + badge_height

    try:
        legend_width = float(host._estimate_preview_source_legend_width())
        legend_height = float(host._estimate_preview_source_legend_height())
    except Exception:
        legend_width = legend_height = 0.0
    if legend_width > 0.0 and legend_height > 0.0:
        legend_x = badge_x + 2.0
        legend_y = badge_y
        if badge_x > row_start_x and (legend_x + legend_width) > row_right_limit:
            legend_y += row_gap_y
        bottom = max(bottom, legend_y + legend_height)

    return float(bottom + 6.0)


def plan_preview_canvas_info_overlay_layout(
    host: "CharacterAnnotationTab",
    canvas_width: int,
    source_line: str,
    status_text: str,
    *,
    data: dict | None = None,
    box_chars=None,
):
    canvas_w = max(40.0, float(canvas_width or 0.0))
    reset_width = host._measure_preview_overlay_text_width("Reset widoku", ("Segoe UI", 9, "bold")) + 14.0
    title_x = 10.0 + reset_width + 14.0
    result_slot_width = max(170.0, min(240.0, canvas_w * 0.21))
    file_slot_width = max(170.0, min(250.0, canvas_w * 0.22))
    badge_end_x = title_x + result_slot_width + 12.0 + file_slot_width + 6.0 + 78.0 + 6.0 + 84.0 + 6.0
    legend_x = badge_end_x + 8.0

    source_width = host._measure_preview_overlay_text_width(source_line, ("Segoe UI", 9))
    status_width = host._measure_preview_overlay_text_width(status_text, ("Segoe UI", 9, "bold"))
    source_right = canvas_w - 10.0
    source_left = source_right - source_width
    status_right = source_left - 10.0
    status_left = status_right - status_width

    legend_width = host._estimate_preview_source_legend_width()
    legend_height = host._estimate_preview_source_legend_height()
    info_text_height = host._measure_preview_overlay_font_height(("Segoe UI", 10, "bold"))
    stack_right_info = bool(status_left <= (badge_end_x + 14.0))
    available_legend_right = (canvas_w - 12.0) if stack_right_info else (status_left - 12.0)
    stack_legend = bool((legend_x + legend_width) > available_legend_right)

    legend_y = float(36.0 if stack_legend else 8.0)
    source_y = float(44.0 if stack_right_info else 16.0)
    status_y = float(44.0 if stack_right_info else 16.0)
    legend_bottom = legend_y + float(legend_height)
    right_info_bottom = max(source_y + info_text_height, status_y + info_text_height)
    min_bar_height = 78.0 if (stack_legend or stack_right_info) else 74.0
    badges_bottom = _estimate_preview_canvas_info_badges_bottom(
        host,
        canvas_width,
        data=data,
        box_chars=box_chars,
    )
    bar_height = max(min_bar_height, legend_bottom + 8.0, right_info_bottom + 8.0, badges_bottom)

    return {
        "title_x": float(title_x),
        "result_slot_width": float(result_slot_width),
        "file_slot_width": float(file_slot_width),
        "legend_x": float(title_x if stack_legend else legend_x),
        "legend_y": legend_y,
        "source_x": float(max(20.0, source_right)),
        "source_y": source_y,
        "status_x": float(max(20.0, status_right)),
        "status_y": status_y,
        "bar_height": float(bar_height),
    }


def focus_preview_canvas(host: "CharacterAnnotationTab"):
    canvas = getattr(host, "preview_canvas", None)
    if canvas is None:
        return
    try:
        canvas.focus_set()
    except Exception:
        pass


def resolve_preview_canvas_cursor(host: "CharacterAnnotationTab") -> str:
    if not bool(getattr(host, "_preview_render_state", None)):
        return "arrow"
    if getattr(host, "_preview_layout_separator_drag_state", None) is not None:
        return "sb_v_double_arrow"
    if getattr(host, "_preview_badge_drag_state", None) is not None:
        return "hand2"
    char_drag_state = getattr(host, "_preview_char_drag_state", None)
    if isinstance(char_drag_state, dict):
        mode = str(char_drag_state.get("mode", "move") or "move").lower()
        return "crosshair" if mode == "resize" else "fleur"
    if getattr(host, "_preview_pan_drag_state", None) is not None:
        return "fleur"
    if host._preview_char_add_requested():
        return "crosshair"
    return "arrow"


def apply_preview_canvas_cursor(host: "CharacterAnnotationTab", cursor: str | None = None):
    canvas = getattr(host, "preview_canvas", None)
    if canvas is None:
        return
    desired = str(cursor or resolve_preview_canvas_cursor(host) or "arrow")
    try:
        current = str(canvas.cget("cursor") or "")
    except Exception:
        current = ""
    if current == desired:
        return
    try:
        canvas.configure(cursor=desired)
    except Exception:
        pass


def on_preview_canvas_enter(host: "CharacterAnnotationTab", event=None):
    focus_preview_canvas(host)
    apply_preview_canvas_cursor(host)
    return None


def place_preview_record_overlay(
    host: "CharacterAnnotationTab",
    canvas_width: int,
    bar_height: float,
    *,
    left_x: float = 10.0,
    top_y: float = 6.0,
    right_limit: float | None = None,
):
    canvas = getattr(host, "preview_canvas", None)
    overlay = getattr(host, "preview_record_overlay", None)
    if canvas is None or overlay is None:
        return
    try:
        overlay_w = float(max(0, int(overlay.winfo_reqwidth() or overlay.winfo_width() or 0)))
        x = float(left_x)
        if right_limit is not None and overlay_w > 0.0:
            x = min(x, max(10.0, float(right_limit) - overlay_w))
        y = float(top_y)
        overlay.place(in_=canvas, x=x, y=y, anchor="nw")
        overlay.lift()
    except Exception:
        pass


def get_preview_mode_overlay_default_position(
    host: "CharacterAnnotationTab",
    *,
    fullscreen: bool = False,
):
    if bool(fullscreen):
        top_bar_height = max(38.0, float(getattr(host, "_preview_overlay_top_bar_height", 38.0) or 38.0))
        return {"x": 12.0, "y": float(top_bar_height + 8.0)}

    canvas_w, _canvas_h = host._get_preview_canvas_size()
    overlay_w, _overlay_h = host._get_preview_mode_overlay_size()
    desired_x = max(12.0, canvas_w - overlay_w - 12.0)
    return {"x": float(desired_x), "y": 12.0}


def _clear_preview_character_overlay_items(
    host: "CharacterAnnotationTab",
    canvas,
    *,
    box_source: str,
    display_idx: int,
    char_record: dict,
    clear_drag_preview: bool = False,
) -> None:
    tags_to_delete = {
        host._get_preview_character_canvas_tag(box_source, int(display_idx)),
        host._get_preview_character_record_canvas_tag(char_record),
    }

    runtime_key = f"{box_source}:{int(display_idx)}"
    runtime_map = getattr(host, "_preview_char_runtime", None)
    if isinstance(runtime_map, dict):
        payload = runtime_map.pop(runtime_key, None)
        if isinstance(payload, dict):
            old_tag = str(payload.get("tag", "") or "").strip()
            if old_tag:
                tags_to_delete.add(old_tag)

    badge_runtime_map = getattr(host, "_preview_badge_runtime", None)
    if isinstance(badge_runtime_map, dict):
        badge_payload = badge_runtime_map.pop(runtime_key, None)
        if isinstance(badge_payload, dict):
            old_badge_tag = str(badge_payload.get("tag", "") or "").strip()
            if old_badge_tag:
                tags_to_delete.add(old_badge_tag)
            line_id = badge_payload.get("line_id")
            if line_id is not None:
                try:
                    canvas.delete(line_id)
                except Exception:
                    pass
        tags_to_delete.add(f"preview_badge::{runtime_key}")
        tags_to_delete.add(f"preview_badge_line::{runtime_key}")

    record_tags = getattr(host, "_preview_char_record_render_tags", None)
    if isinstance(record_tags, dict):
        old_record_tag = str(record_tags.pop(id(char_record), "") or "").strip()
        if old_record_tag:
            tags_to_delete.add(old_record_tag)

    for item_tag in tags_to_delete:
        try:
            canvas.delete(item_tag)
        except Exception:
            pass

    if clear_drag_preview:
        try:
            canvas.delete("preview_char_drag_preview")
        except Exception:
            pass


def _build_preview_compact_char_meta_text(badge_layers, reading_label: str | None = None) -> str:
    parts: list[str] = []
    for layer in badge_layers or []:
        if not isinstance(layer, dict):
            continue
        token = str(layer.get("text", "") or "").strip().split(" ")[0].strip()
        if token and token not in parts:
            parts.append(token)
    return " | ".join(parts)


def _measure_preview_signature_text(host: "CharacterAnnotationTab", font_key: str, font_spec, text: str) -> float:
    normalized_text = str(text or " ")
    try:
        fonts = getattr(host, "_preview_signature_fonts", None)
        if not isinstance(fonts, dict):
            fonts = {}
            host._preview_signature_fonts = fonts
        font_obj = fonts.get(font_key)
        if font_obj is None:
            font_obj = tkfont.Font(font=font_spec)
            fonts[font_key] = font_obj

        cache = getattr(host, "_preview_signature_measure_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            host._preview_signature_measure_cache = cache
        cache_key = (font_key, normalized_text)
        cached = cache.get(cache_key)
        if cached is not None:
            return float(cached)
        width = float(font_obj.measure(normalized_text))
        if len(cache) > 512:
            cache.clear()
        cache[cache_key] = width
        return width
    except Exception:
        return float(max(8, len(normalized_text) * 8))


def _get_preview_signature_stagger_index(rec: dict | None, fallback_index: int = 0) -> int:
    try:
        col = int((rec or {}).get("reading_col", 0) or 0)
    except Exception:
        col = 0
    if col > 0:
        return max(0, col - 1)
    try:
        return max(0, int(fallback_index))
    except Exception:
        return 0


def _get_preview_signature_stagger_offset(stagger_index: int | None) -> float:
    try:
        idx = max(0, int(stagger_index or 0))
    except Exception:
        idx = 0
    return -10.0 if (idx % 2 == 0) else 10.0


def _draw_preview_compact_character_signature(
    host: "CharacterAnnotationTab",
    canvas,
    *,
    center_x: float,
    box_anchor_y: float,
    image_top: float,
    image_bottom: float,
    canvas_height: float,
    top_limit: float,
    side: str,
    char_text: str,
    char_color: str,
    guide_color: str,
    badge_layers,
    reading_label: str | None,
    tags,
    stagger_index: int = 0,
) -> None:
    normalized_side = str(side or "top").strip().lower()
    char_font = ("Segoe UI", 16, "bold")
    meta_font = ("Segoe UI", 7, "bold")
    meta_text = _build_preview_compact_char_meta_text(badge_layers, reading_label)
    char_width = _measure_preview_signature_text(host, "preview_signature_char", char_font, str(char_text or " "))
    meta_width = _measure_preview_signature_text(host, "preview_signature_meta", meta_font, str(meta_text or " ")) + 10.0
    line_width = max(22.0, char_width + 12.0, meta_width + 4.0)

    if normalized_side == "bottom":
        try:
            handle_radius = float(host._get_preview_char_handle_radius())
        except Exception:
            handle_radius = 9.0
        status_frame_pad = max(20.0, min(30.0, handle_radius + 12.0))
        min_char_y = float(image_bottom) + status_frame_pad + 22.0
        max_char_y = float(canvas_height) - 38.0
        if max_char_y >= min_char_y:
            char_y = max(min_char_y, min(max_char_y, min_char_y + 14.0))
        else:
            char_y = min_char_y
        rule_y = char_y + 15.0
        meta_y = rule_y + 9.0
        if meta_y + 8.0 > float(canvas_height) - 6.0:
            shift = (meta_y + 8.0) - (float(canvas_height) - 6.0)
            char_y = max(min_char_y, char_y - shift)
            rule_y = char_y + 15.0
            meta_y = rule_y + 9.0
        line_start_y = float(box_anchor_y)
        line_end_y = char_y - 12.0
    else:
        safe_char_y = float(top_limit) + 46.0
        char_y = min(float(image_top) - 18.0, max(safe_char_y, float(image_top) - 24.0))
        min_char_y = float(top_limit) + 42.0
        max_char_y = float(image_top) - 32.0
        if max_char_y >= min_char_y:
            char_y = max(min_char_y, min(max_char_y, char_y + _get_preview_signature_stagger_offset(stagger_index)))
        else:
            char_y = min(float(char_y), max_char_y)
        rule_y = char_y - 15.0
        meta_y = rule_y - 9.0
        if meta_y - 8.0 < float(top_limit) + 10.0:
            shift = (float(top_limit) + 10.0) - (meta_y - 8.0)
            char_y = min(float(image_top) - 32.0, char_y + shift)
            rule_y = char_y - 15.0
            meta_y = rule_y - 9.0
        if meta_y - 8.0 < float(top_limit) + 10.0:
            char_y = min(safe_char_y, float(image_top) - 32.0)
            rule_y = char_y - 15.0
            meta_y = rule_y - 9.0
        line_start_y = float(box_anchor_y)
        line_end_y = char_y + 12.0

    try:
        canvas.create_line(
            float(center_x),
            line_start_y,
            float(center_x),
            line_end_y,
            fill=guide_color,
            width=1,
            tags=tags,
        )
        canvas.create_line(
            float(center_x) - (line_width / 2.0),
            rule_y,
            float(center_x) + (line_width / 2.0),
            rule_y,
            fill=guide_color,
            width=1,
            tags=tags,
        )
        canvas.create_text(
            float(center_x) + 1,
            char_y + 1,
            text=str(char_text or ""),
            fill="#000000",
            font=char_font,
            anchor=tk.CENTER,
            tags=tags,
        )
        canvas.create_text(
            float(center_x),
            char_y,
            text=str(char_text or ""),
            fill=char_color,
            font=char_font,
            anchor=tk.CENTER,
            tags=tags,
        )
        if meta_text:
            palette = getattr(host.app, "palette", {})
            panel = str(palette.get("panel", "#111111"))
            meta_fill = blend_hex_colors(panel, guide_color, 0.18)
            meta_text_color = host._get_readable_text_color(meta_fill, preferred="#ffffff")
            x1 = float(center_x) - (meta_width / 2.0)
            y1 = meta_y - 7.0
            x2 = float(center_x) + (meta_width / 2.0)
            y2 = meta_y + 7.0
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=meta_fill,
                outline=guide_color,
                width=1,
                tags=tags,
            )
            canvas.create_text(
                float(center_x),
                meta_y,
                text=meta_text,
                fill=meta_text_color,
                font=meta_font,
                anchor=tk.CENTER,
                tags=tags,
            )
    except Exception:
        pass


def _get_preview_character_edit_grip_style(host: "CharacterAnnotationTab") -> dict:
    palette = getattr(host.app, "palette", {})
    canvas_bg = str(palette.get("panel", "#101010"))
    group_mode = bool(getattr(host, "_preview_char_geometry_inherit_down", False))
    if group_mode:
        active_outline = str(palette.get("success", palette.get("accent", "#22c55e")))
        grip_fill_base = str(palette.get("accent_alt", "#14b8a6"))
        idle_fill_alpha = 0.42
        active_fill_alpha = 0.72
    else:
        active_outline = str(palette.get("warning", "#ff9f1a"))
        grip_fill_base = str(palette.get("info", palette.get("accent_alt", "#38bdf8")))
        idle_fill_alpha = 0.36
        active_fill_alpha = 0.68
    return {
        "active_outline": active_outline,
        "idle_outline": blend_hex_colors(active_outline, canvas_bg, 0.38),
        "idle_fill": blend_hex_colors(canvas_bg, grip_fill_base, idle_fill_alpha),
        "active_fill": blend_hex_colors(canvas_bg, grip_fill_base, active_fill_alpha),
    }


def _get_preview_selected_character_box_color(host: "CharacterAnnotationTab") -> str:
    return "#d000a8"


def _draw_preview_new_character_box(canvas, x1, y1, x2, y2):
    # Cyan with a dark keyline stays visible over white plates, black glyphs
    # and the photo, independently of the application theme.
    tags = ("preview_char_add_preview",)
    canvas.create_rectangle(x1, y1, x2, y2, outline="#071923", width=6, tags=tags)
    canvas.create_rectangle(x1, y1, x2, y2, outline="#00e5ff", width=3, tags=tags)
    label = canvas.create_text(x1 + 4, max(8, y1 - 8), text="Nowy box", fill="#00e5ff",
                               font=("Segoe UI", 9, "bold"), anchor=tk.SW, tags=tags)
    bounds = canvas.bbox(label)
    if bounds:
        background = canvas.create_rectangle(bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2,
                                             fill="#071923", outline="", tags=tags)
        canvas.tag_lower(background, label)


def _draw_preview_character_edit_grips(
    host: "CharacterAnnotationTab",
    canvas,
    *,
    tags,
    cx1: float,
    cy1: float,
    cx2: float,
    cy2: float,
    selection_color: str,
) -> tuple[list[int], int | None]:
    grip_style = _get_preview_character_edit_grip_style(host)
    active_outline = str(grip_style["active_outline"])
    idle_outline = str(grip_style["idle_outline"])
    idle_fill = str(grip_style["idle_fill"])
    active_fill = str(grip_style["active_fill"])
    hover_key = str(getattr(host, "_preview_char_hover_grip", "") or "")

    handle_radius = float(host._get_preview_char_handle_radius())
    move_radius = float(host._get_preview_char_move_handle_radius())
    handle_ids: list[int] = []
    for handle_name, handle_x, handle_y in (
        ("nw", float(cx1), float(cy1)),
        ("ne", float(cx2), float(cy1)),
        ("sw", float(cx1), float(cy2)),
        ("se", float(cx2), float(cy2)),
    ):
        active = hover_key == f"corner:{handle_name}"
        handle_ids.append(
            canvas.create_oval(
                handle_x - handle_radius,
                handle_y - handle_radius,
                handle_x + handle_radius,
                handle_y + handle_radius,
                fill=active_fill if active else idle_fill,
                outline=active_outline if active else idle_outline,
                width=2 if active else 1,
                tags=tags,
            )
        )

    center_x = (float(cx1) + float(cx2)) / 2.0
    center_y = (float(cy1) + float(cy2)) / 2.0
    move_active = hover_key == "move:center"
    move_handle_id = canvas.create_oval(
        center_x - move_radius,
        center_y - move_radius,
        center_x + move_radius,
        center_y + move_radius,
        fill=active_fill if move_active else idle_fill,
        outline=active_outline if move_active else idle_outline,
        width=2 if move_active else 1,
        tags=tags,
    )
    return handle_ids, move_handle_id


def _preview_char_label_canvas_tags(tags) -> tuple:
    return tuple(tags or ()) + ("preview_char_label_field",)


def _style_preview_character_edit_grips_fast(
    host: "CharacterAnnotationTab",
    canvas,
    runtime: dict,
    *,
    selection_color: str,
) -> bool:
    handle_ids = list(runtime.get("handle_ids", []) or [])
    move_handle_id = runtime.get("move_handle_id")
    if len(handle_ids) < 4 or move_handle_id is None:
        return False

    grip_style = _get_preview_character_edit_grip_style(host)
    active_outline = str(grip_style["active_outline"])
    idle_outline = str(grip_style["idle_outline"])
    idle_fill = str(grip_style["idle_fill"])
    active_fill = str(grip_style["active_fill"])
    hover_key = str(getattr(host, "_preview_char_hover_grip", "") or "")

    try:
        for handle_name, item_id in zip(("nw", "ne", "sw", "se"), handle_ids):
            active = hover_key == f"corner:{handle_name}"
            canvas.itemconfigure(
                item_id,
                fill=active_fill if active else idle_fill,
                stipple="",
                outline=active_outline if active else idle_outline,
                width=2 if active else 1,
            )
        move_active = hover_key == "move:center"
        canvas.itemconfigure(
            move_handle_id,
            fill=active_fill if move_active else idle_fill,
            stipple="",
            outline=active_outline if move_active else idle_outline,
            width=2 if move_active else 1,
        )
        return True
    except Exception:
        return False


def update_preview_character_selection_items_fast(host: "CharacterAnnotationTab", indices=None) -> bool:
    perf_start = time.perf_counter()
    canvas = getattr(host, "preview_canvas", None)
    state = getattr(host, "_preview_render_state", None) or {}
    runtime_map = getattr(host, "_preview_char_runtime", None)
    if canvas is None or not state or not isinstance(runtime_map, dict):
        return False
    if bool(getattr(host, "_preview_char_label_mode", False)):
        return False
    if getattr(host, "_preview_char_label_active_index", None) is not None:
        return False

    data = host._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return False
    chars = host._get_preview_active_character_records(create=False)
    if not isinstance(chars, list):
        return False

    if indices is None:
        target_indices = range(len(chars))
    else:
        normalized = []
        for raw_idx in set(indices or []):
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if 0 <= idx < len(chars):
                normalized.append(idx)
        target_indices = sorted(normalized)

    try:
        render_scale = max(0.001, float(state.get("scale", 1.0) or 1.0))
        x_off = float(state.get("image_left", 0.0) or 0.0)
        y_off = float(state.get("image_top", 0.0) or 0.0)
    except Exception:
        return False

    try:
        selected_idx = getattr(host, "_preview_char_selected_index", None)
        selected_idx = int(selected_idx) if selected_idx is not None else None
    except Exception:
        selected_idx = None

    selection_color = getattr(host.app, "palette", {}).get("accent", "#ffd166")
    selected_box_color = _get_preview_selected_character_box_color(host)
    edit_mode = bool(getattr(host, "_preview_char_edit_mode", False))
    touched = False
    for idx in target_indices:
        rec = chars[idx] if 0 <= int(idx) < len(chars) else None
        if not isinstance(rec, dict):
            continue
        runtime_key = f"FINAL:{int(idx)}"
        runtime = runtime_map.get(runtime_key)
        if not isinstance(runtime, dict) or runtime.get("box_id") is None:
            return False

        is_selected = bool(selected_idx is not None and int(idx) == int(selected_idx))
        if (
            is_selected
            and edit_mode
            and runtime.get("selection_id") is not None
            and _style_preview_character_edit_grips_fast(
                host,
                canvas,
                runtime,
                selection_color=selection_color,
            )
        ):
            try:
                canvas.itemconfigure(runtime.get("box_id"), outline=selected_box_color, width=3)
            except Exception:
                pass
            touched = True
            continue

        for item_id in list(runtime.get("handle_ids", []) or []):
            try:
                canvas.delete(item_id)
            except Exception:
                pass
        for item_id in (runtime.get("selection_id"), runtime.get("move_handle_id")):
            if item_id is None:
                continue
            try:
                canvas.delete(item_id)
            except Exception:
                pass
        runtime["selection_id"] = None
        runtime["handle_ids"] = []
        runtime["move_handle_id"] = None

        bbox = host._char_record_bbox(rec)
        if not bbox:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            cx1 = (x1 * render_scale) + x_off
            cy1 = (y1 * render_scale) + y_off
            cx2 = (x2 * render_scale) + x_off
            cy2 = (y2 * render_scale) + y_off
        except Exception:
            continue

        try:
            box_source, sign_source = _preview_character_source_parts(host, rec, data=data, fallback_index=idx)
            visual_tag = _preview_visual_tag_for_box_source(host, box_source, sign_source)
            box_color = host._get_preview_source_visual_style(visual_tag)["outline"]
            canvas.itemconfigure(
                runtime.get("box_id"),
                outline=(selected_box_color if is_selected else box_color),
                width=(3 if is_selected else 2),
            )
        except Exception:
            pass

        if is_selected:
            tag = host._get_preview_character_canvas_tag("FINAL", int(idx))
            record_tag = host._get_preview_character_record_canvas_tag(rec)
            tags = ("preview_char", tag, record_tag)
            try:
                runtime["selection_id"] = canvas.create_rectangle(
                    cx1 - 2,
                    cy1 - 2,
                    cx2 + 2,
                    cy2 + 2,
                    outline=selection_color,
                    width=1,
                    dash=(4, 2),
                    tags=tags,
                )
                if edit_mode:
                    handle_ids, move_handle_id = _draw_preview_character_edit_grips(
                        host,
                        canvas,
                        tags=tags,
                        cx1=cx1,
                        cy1=cy1,
                        cx2=cx2,
                        cy2=cy2,
                        selection_color=selection_color,
                    )
                    runtime["handle_ids"] = handle_ids
                    runtime["move_handle_id"] = move_handle_id
                canvas.tag_raise(record_tag)
            except Exception:
                return False
        touched = True

    if touched:
        try:
            canvas.tag_raise("preview_char_add_preview")
            canvas.tag_raise("preview_overlay")
            canvas.tag_raise("preview_overlay_action")
        except Exception:
            pass
        _log_preview_perf(
            "char_selection_visual_fast",
            perf_start,
            threshold_ms=35.0,
            items=len(list(target_indices)) if not isinstance(target_indices, range) else len(target_indices),
        )
    return bool(touched)


def redraw_preview_character_overlay_only(
    host: "CharacterAnnotationTab",
    char_idx: int,
    *,
    drag_preview: bool = False,
) -> bool:
    canvas = getattr(host, "preview_canvas", None)
    state = getattr(host, "_preview_render_state", None) or {}
    if canvas is None or not state:
        return False

    data = host._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return False

    try:
        char_idx = int(char_idx)
    except Exception:
        return False

    canonical_chars = host._get_preview_active_character_records(create=False)
    if not (0 <= char_idx < len(canonical_chars)):
        return False

    box_source = "FINAL"
    display_idx = char_idx
    char_record = canonical_chars[char_idx]
    if not isinstance(char_record, dict):
        return False

    tag = host._get_preview_character_canvas_tag(box_source, int(display_idx))
    record_tag = host._get_preview_character_record_canvas_tag(char_record)
    tags = ("preview_char", tag, record_tag)
    _clear_preview_character_overlay_items(
        host,
        canvas,
        box_source=box_source,
        display_idx=display_idx,
        char_record=char_record,
        clear_drag_preview=not bool(drag_preview),
    )

    bbox = host._char_record_bbox(char_record)
    if not bbox:
        return True

    try:
        render_scale = max(0.001, float(state.get("scale", 1.0) or 1.0))
        x_off = float(state.get("image_left", 0.0) or 0.0)
        y_off = float(state.get("image_top", 0.0) or 0.0)
        image_bottom_y = float(state.get("image_bottom", 0.0) or 0.0)
        canvas_height = max(50, int(canvas.winfo_height() or 50))
    except Exception:
        return False

    x1, y1, x2, y2 = bbox[:4]
    cx1, cy1 = (float(x1) * render_scale) + x_off, (float(y1) * render_scale) + y_off
    cx2, cy2 = (float(x2) * render_scale) + x_off, (float(y2) * render_scale) + y_off
    center_x = cx1 + (cx2 - cx1) / 2.0

    source_tag = host._get_character_source_tag(char_record, data=data, fallback_index=char_idx)
    box_source, sign_source = _preview_character_source_parts(host, char_record, data=data, fallback_index=char_idx)
    visual_tag = _preview_visual_tag_for_box_source(host, box_source, sign_source)
    source_style = host._get_preview_source_visual_style(visual_tag)
    box_color = source_style["outline"]
    guide_color = source_style["guide"]
    char_fill = box_color
    selection_color = getattr(host.app, "palette", {}).get("accent", "#ffd166")
    selected_box_color = _get_preview_selected_character_box_color(host)
    label_focus_color = getattr(host.app, "palette", {}).get("warning", "#f59e0b")
    is_selected_box = host._is_preview_char_record_selected(char_record, fallback_index=char_idx)
    drag_preview = bool(drag_preview)
    try:
        two_row_layout_active = bool(host._should_preview_use_two_row_layers(data))
    except Exception:
        try:
            two_row_layout_active = bool(host._is_preview_two_row_layout_active(data))
        except Exception:
            two_row_layout_active = False
    if two_row_layout_active:
        try:
            host._ensure_preview_layout_separator(
                data,
                data.get("characters", []),
                image_w=float(state.get("orig_w", data.get("plate_image_width", 1.0)) or 1.0),
                image_h=float(state.get("orig_h", data.get("plate_image_height", 1.0)) or 1.0),
            )
        except Exception:
            pass
    row_no = host._get_preview_row_for_bbox([float(x1), float(y1), float(x2), float(y2)], data) or 1
    badge_side = "bottom" if two_row_layout_active and int(row_no) == 2 else "top"

    selection_id = None
    handle_ids = []
    move_handle_id = None
    try:
        box_id = canvas.create_rectangle(
            cx1,
            cy1,
            cx2,
            cy2,
            outline=(selected_box_color if is_selected_box else box_color),
            width=(3 if is_selected_box else 2),
            tags=tags,
        )
        if is_selected_box:
            selection_id = canvas.create_rectangle(
                cx1 - 2,
                cy1 - 2,
                cx2 + 2,
                cy2 + 2,
                outline=selection_color,
                width=1,
                dash=(4, 2),
                tags=tags,
            )

        char_text = str(char_record.get("character", ""))
        if not drag_preview:
            reading_label = host._get_preview_character_reading_position_label(char_record, data=data)
            _draw_preview_compact_character_signature(
                host,
                canvas,
                center_x=center_x,
                box_anchor_y=cy2 if badge_side == "bottom" else cy1,
                image_top=y_off,
                image_bottom=image_bottom_y,
                canvas_height=float(canvas_height),
                top_limit=float(state.get("info_bar_height", 0.0) or 0.0) + 18.0,
                side=badge_side,
                char_text=char_text,
                char_color=char_fill,
                guide_color=guide_color,
                badge_layers=host._get_preview_source_badge_layers(
                    source_tag,
                    confidence=None,
                    box_backend_confidence=host._character_record_yolo_box_backend_confidence(char_record),
                    include_confidence=False,
                    has_symbol=host._preview_record_has_symbol(char_record),
                    uses_yolo_box_backend=host._character_record_uses_yolo_box_backend(char_record),
                    box_source=box_source,
                    sign_source=sign_source,
                ),
                reading_label=reading_label,
                tags=tags,
                stagger_index=_get_preview_signature_stagger_index(char_record, char_idx),
            )

        if is_selected_box and bool(getattr(host, "_preview_char_edit_mode", False)):
            handle_ids, move_handle_id = _draw_preview_character_edit_grips(
                host,
                canvas,
                tags=tags,
                cx1=cx1,
                cy1=cy1,
                cx2=cx2,
                cy2=cy2,
                selection_color=selection_color,
            )

        label_mode_active = bool(getattr(host, "_preview_char_label_mode", False))
        active_label_idx = getattr(host, "_preview_char_label_active_index", None)
        hover_label_idx = getattr(host, "_preview_char_hover_label_index", None)
        show_label_box = bool(
            not drag_preview
            and (
                label_mode_active
                or (active_label_idx is not None and int(active_label_idx) == int(char_idx))
                or (
                    hover_label_idx is not None
                    and int(hover_label_idx) == int(char_idx)
                    and (label_mode_active or active_label_idx is not None)
                )
            )
        )
        if show_label_box:
            label_rect = host._get_preview_char_label_canvas_rect(char_record)
            if label_rect is not None:
                lx1, ly1, lx2, ly2 = label_rect
                label_active = bool(active_label_idx is not None and int(active_label_idx) == int(char_idx))
                label_hovered = bool(hover_label_idx is not None and int(hover_label_idx) == int(char_idx))
                label_highlight = bool(label_active or label_hovered)
                label_accent = label_focus_color if (label_mode_active and label_highlight) else selection_color
                if label_active:
                    label_fill = label_focus_color
                    label_outline = host._get_readable_text_color(label_fill, preferred="#111111")
                    label_text_fill = host._get_readable_text_color(label_fill, preferred="#111111")
                    label_width = 3
                else:
                    label_fill = blend_hex_colors("#ffffff", label_accent, 0.22 if label_highlight else 0.08)
                    label_outline = label_accent if label_highlight else box_color
                    label_text_fill = "#111111"
                    label_width = 1
                label_tags = _preview_char_label_canvas_tags(tags)
                if label_active:
                    canvas.create_rectangle(
                        lx1 - 2,
                        ly1 - 2,
                        lx2 + 2,
                        ly2 + 2,
                        outline=label_focus_color,
                        width=2,
                        tags=label_tags,
                    )
                canvas.create_rectangle(
                    lx1,
                    ly1,
                    lx2,
                    ly2,
                    fill=label_fill,
                    outline=label_outline,
                    width=label_width,
                    tags=label_tags,
                )
                valid_char = host._sanitize_preview_char_symbol(char_text)
                if valid_char:
                    canvas.create_text(
                        (lx1 + lx2) / 2.0,
                        (ly1 + ly2) / 2.0 + 0.5,
                        text=valid_char,
                        fill=label_text_fill,
                        font=("Segoe UI", 8, "bold"),
                        anchor=tk.CENTER,
                        tags=label_tags,
                    )
                elif label_active:
                    canvas.create_line(
                        lx1 + 7,
                        ly1 + 4,
                        lx1 + 7,
                        ly2 - 4,
                        fill=label_text_fill,
                        width=2,
                        tags=label_tags,
                    )
    except Exception:
        return False

    runtime_map = getattr(host, "_preview_char_runtime", None)
    if not isinstance(runtime_map, dict):
        runtime_map = {}
        host._preview_char_runtime = runtime_map
    runtime_map[f"{box_source}:{int(display_idx)}"] = {
        "tag": tag,
        "box_id": box_id,
        "selection_id": selection_id,
        "handle_ids": handle_ids,
        "move_handle_id": move_handle_id,
        "record_id": id(char_record),
    }
    record_tags = getattr(host, "_preview_char_record_render_tags", None)
    if not isinstance(record_tags, dict):
        record_tags = {}
        host._preview_char_record_render_tags = record_tags
    record_tags[id(char_record)] = record_tag

    badge_key = f"{box_source}:{int(display_idx)}"
    badge_runtime = getattr(host, "_preview_badge_runtime", {}).get(badge_key)
    if isinstance(badge_runtime, dict):
        badge_runtime["box_id"] = box_id
        line_id = badge_runtime.get("line_id")
        if line_id is not None:
            try:
                badge_center_x = float(badge_runtime.get("base_center_x", center_x)) + float(badge_runtime.get("offset_dx", 0.0))
                badge_line_y = float(badge_runtime.get("base_line_y", badge_runtime.get("base_bottom_y", cy1))) + float(badge_runtime.get("offset_dy", 0.0))
                anchor_y = cy2 if str(badge_runtime.get("badge_side", "top") or "top").lower() == "bottom" else cy1
                badge_runtime["box_anchor_y"] = float(anchor_y)
                canvas.coords(line_id, badge_center_x, badge_line_y, center_x, anchor_y)
            except Exception:
                pass

    try:
        canvas.tag_raise(record_tag)
        canvas.tag_raise("preview_badge")
        canvas.tag_raise("preview_char_add_preview")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    return True


def draw_preview_top_badges(
    host: "CharacterAnnotationTab",
    canvas,
    badge_specs,
    *,
    image_left: float,
    image_right: float,
    image_top: float,
    image_bottom: float | None = None,
    top_limit: float,
    bottom_limit: float | None = None,
) -> None:
    if canvas is None:
        return

    metrics = host._get_preview_badge_layout_metrics()
    badge_font = metrics["font"]
    pad_x = metrics["pad_x"]
    pad_y = metrics["pad_y"]
    col_gap = metrics["col_gap"]
    row_gap = metrics["row_gap"]
    prepared = []

    for spec in sorted(list(badge_specs or []), key=lambda item: float(item.get("center_x", 0.0))):
        badge_layers = list(spec.get("badge_layers", []) or [])
        text = str(spec.get("text", "") or "").strip()
        if badge_layers:
            width, height = host._measure_preview_badge_stack(
                canvas,
                badge_layers,
                font=badge_font,
                pad_x=pad_x,
                pad_y=pad_y,
                layer_gap=row_gap,
            )
        else:
            if not text:
                continue
            width, height = host._measure_preview_text_badge(
                canvas,
                text,
                font=badge_font,
                pad_x=pad_x,
                pad_y=pad_y,
            )
        if width <= 0 or height <= 0:
            continue
        prepared.append({
            **spec,
            "text": text,
            "badge_layers": badge_layers,
            "width": width,
            "height": height,
        })

    if not prepared:
        return

    def _place_group(group_specs, *, side: str):
        if not group_specs:
            return

        group_badge_height = max(item["height"] for item in group_specs)
        badge_to_plate_gap = max(12.0, float(row_gap) + 8.0)
        normalized_side = str(side or "top").strip().lower()
        if normalized_side == "bottom":
            try:
                handle_radius = float(host._get_preview_char_handle_radius())
            except Exception:
                handle_radius = 9.0
            status_frame_pad = max(20.0, min(30.0, handle_radius + 12.0))
            badge_to_plate_gap = max(badge_to_plate_gap, status_frame_pad + max(10.0, float(row_gap) + 6.0))
        row_tops = []
        if normalized_side == "bottom":
            lower_bound = float(bottom_limit) if bottom_limit is not None else float(canvas.winfo_height() or 0) - 8.0
            start_top = float(image_bottom if image_bottom is not None else image_top) + badge_to_plate_gap
            next_row_top = start_top
            while next_row_top + group_badge_height <= lower_bound:
                row_tops.append(next_row_top)
                next_row_top += group_badge_height + row_gap
            if not row_tops:
                row_tops.append(start_top)
        else:
            next_row_top = float(image_top) - badge_to_plate_gap - group_badge_height
            while next_row_top >= float(top_limit):
                row_tops.append(next_row_top)
                next_row_top -= group_badge_height + row_gap
            if not row_tops:
                row_tops.append(max(float(top_limit), float(image_top) - badge_to_plate_gap - group_badge_height))

        row_last_right = [float(image_left) - col_gap for _ in row_tops]

        for spec in group_specs:
            desired_left = float(spec["center_x"]) - (float(spec["width"]) / 2.0)
            desired_left = max(float(image_left), min(desired_left, float(image_right) - float(spec["width"])))

            placed_row_idx = None
            placed_left = None
            for row_idx, _row_top in enumerate(row_tops):
                candidate_left = max(desired_left, row_last_right[row_idx] + col_gap)
                if candidate_left + float(spec["width"]) <= float(image_right):
                    placed_row_idx = row_idx
                    placed_left = candidate_left
                    break

            if placed_row_idx is None:
                if normalized_side == "bottom":
                    extra_row_top = row_tops[-1] + (group_badge_height + row_gap)
                    can_add = extra_row_top + group_badge_height <= (
                        float(bottom_limit) if bottom_limit is not None else float(canvas.winfo_height() or 0) - 8.0
                    )
                else:
                    extra_row_top = row_tops[-1] - (group_badge_height + row_gap)
                    can_add = extra_row_top >= float(top_limit)
                if can_add:
                    row_tops.append(extra_row_top)
                    row_last_right.append(float(image_left) - col_gap)
                    placed_row_idx = len(row_tops) - 1
                    placed_left = max(
                        float(image_left),
                        min(desired_left, float(image_right) - float(spec["width"]))
                    )

            if placed_row_idx is None:
                best_row_idx = min(range(len(row_tops)), key=lambda idx: row_last_right[idx])
                fallback_left = max(float(image_left), min(desired_left, float(image_right) - float(spec["width"])))
                fallback_left = max(fallback_left, row_last_right[best_row_idx] + col_gap)
                fallback_left = min(fallback_left, float(image_right) - float(spec["width"]))
                placed_row_idx = best_row_idx
                placed_left = fallback_left

            row_last_right[placed_row_idx] = placed_left + float(spec["width"])
            base_top = float(row_tops[placed_row_idx])
            badge_key = str(spec.get("badge_key", spec.get("text", "")))
            offset_dx, offset_dy = host._get_preview_badge_offset(host._preview_active_pid, badge_key)
            badge_left = float(placed_left) + float(offset_dx)
            badge_top = float(base_top) + float(offset_dy)
            min_badge_top = float(top_limit)
            max_badge_top = (
                (float(bottom_limit) - float(spec["height"]))
                if bottom_limit is not None
                else (float(canvas.winfo_height() or 0) - float(spec["height"]) - 8.0)
            )
            if max_badge_top < min_badge_top:
                max_badge_top = min_badge_top
            clamped_badge_top = max(min_badge_top, min(max_badge_top, float(badge_top)))
            if abs(clamped_badge_top - float(badge_top)) > 0.001:
                offset_dy = float(offset_dy) + (clamped_badge_top - float(badge_top))
                badge_top = clamped_badge_top
            badge_center_x = badge_left + (float(spec["width"]) / 2.0)
            badge_line_y = badge_top if normalized_side == "bottom" else badge_top + float(spec["height"])
            box_anchor_y = float(spec.get("box_bottom_y", image_top)) if normalized_side == "bottom" else float(spec.get("box_top_y", image_top))

            line_id = None
            try:
                line_id = canvas.create_line(
                    badge_center_x,
                    badge_line_y,
                    float(spec["center_x"]),
                    box_anchor_y,
                    fill=str(spec.get("guide_color", spec.get("fill_color", "#cccccc"))),
                    dash=(2, 2),
                    width=1,
                    tags=("preview_badge", f"preview_badge_line::{badge_key}"),
                )
            except Exception:
                pass

            badge_tag = f"preview_badge::{badge_key}"
            if spec.get("badge_layers"):
                host._draw_preview_badge_stack(
                    canvas,
                    badge_left,
                    badge_top,
                    spec.get("badge_layers"),
                    font=badge_font,
                    pad_x=pad_x,
                    pad_y=pad_y,
                    layer_gap=row_gap,
                    tags=("preview_badge", badge_tag),
                )
            else:
                host._draw_preview_text_badge(
                    canvas,
                    badge_left,
                    badge_top,
                    spec["text"],
                    fill_color=str(spec.get("fill_color", "#3c3c3c")),
                    outline_color=str(spec.get("outline_color", spec.get("fill_color", "#3c3c3c"))),
                    text_color=str(spec.get("text_color", "#ffffff")),
                    font=badge_font,
                    anchor=tk.NW,
                    pad_x=pad_x,
                    pad_y=pad_y,
                    tags=("preview_badge", badge_tag),
                )

            host._preview_badge_runtime[badge_key] = {
                "tag": badge_tag,
                "line_id": line_id,
                "box_id": spec.get("box_id"),
                "box_color": str(spec.get("box_color", spec.get("fill_color", "#cccccc"))),
                "line_color": str(spec.get("guide_color", spec.get("fill_color", "#cccccc"))),
                "box_width": 2,
                "line_width": 1,
                "badge_width": float(spec["width"]),
                "badge_height": float(spec["height"]),
                "left_limit": float(image_left),
                "right_limit": float(image_right),
                "top_limit": float(top_limit),
                "bottom_limit": float(bottom_limit) if bottom_limit is not None else float(canvas.winfo_height() or 0) - 8.0,
                "badge_side": normalized_side,
                "base_left": float(placed_left),
                "base_top": float(base_top),
                "base_center_x": float(placed_left) + (float(spec["width"]) / 2.0),
                "base_bottom_y": float(base_top) + float(spec["height"]),
                "base_line_y": float(base_top) if normalized_side == "bottom" else float(base_top) + float(spec["height"]),
                "box_center_x": float(spec["center_x"]),
                "box_top_y": float(spec.get("box_top_y", image_top)),
                "box_bottom_y": float(spec.get("box_bottom_y", spec.get("box_top_y", image_top))),
                "box_anchor_y": float(box_anchor_y),
                "offset_dx": float(offset_dx),
                "offset_dy": float(offset_dy),
            }

    top_specs = [item for item in prepared if str(item.get("badge_side", "top") or "top").strip().lower() != "bottom"]
    bottom_specs = [item for item in prepared if str(item.get("badge_side", "top") or "top").strip().lower() == "bottom"]
    _place_group(top_specs, side="top")
    _place_group(bottom_specs, side="bottom")


def draw_preview_canvas_info_overlay(
    host: "CharacterAnnotationTab",
    canvas,
    canvas_width: int,
    data: dict,
    box_chars,
    has_boxes: bool,
) -> None:
    if canvas is None:
        return

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_border = palette.get("border", "#3c3c3c")
    title_fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted", "#b0b0b0")
    success_fg = palette.get("success", "#2ecc71")
    error_fg = palette.get("error", "#e74c3c")
    try:
        status_layout = _build_preview_canvas_status_badge_specs(
            host,
            canvas_width,
            data=data,
            box_chars=box_chars,
        )
    except Exception:
        status_layout = {"badges": [], "neutral_badges": [], "pulse": False, "two_row_display": False}
    neutral_status_badges = list(status_layout.get("neutral_badges", []) or [])
    status_badges = list(status_layout.get("badges", []) or [])
    two_row_display = bool(status_layout.get("two_row_display", False))
    source_image = str((data or {}).get("source_image", "") or "").strip()
    source_name = host._truncate_preview_filename(Path(source_image).name if source_image else "Brak pliku", max_chars=28)
    current_idx = host._get_current_preview_list_index()
    total = len(getattr(host, "_listbox_pid_by_index", []))
    current_no = (int(current_idx) + 1) if current_idx is not None and total > 0 else 0
    bar_height = 74.0 if bool(getattr(host, "_preview_fullscreen_active", False)) else 80.0
    if two_row_display:
        bar_height += 18.0
    bar_height = max(
        bar_height,
        _estimate_preview_canvas_info_badges_bottom(
            host,
            canvas_width,
            data=data,
            box_chars=box_chars,
        ),
    )
    host._preview_overlay_top_bar_height = float(bar_height)
    canvas.create_rectangle(
        0,
        0,
        max(40, int(canvas_width)),
        bar_height,
        fill=panel_bg,
        outline=panel_border,
        width=1,
        tags=("preview_overlay",),
    )

    reset_text_id, reset_bg_id = host._draw_preview_text_badge(
        canvas,
        10,
        8,
        "Reset widoku",
        fill_color=panel_border,
        outline_color=panel_border,
        text_color=host._get_readable_text_color(panel_border, preferred=title_fg),
        font=("Segoe UI", 9, "bold"),
        anchor=tk.NW,
        pad_x=7,
        pad_y=3,
        tags=("preview_overlay_action", "preview_action::reset_view"),
    )
    reset_bbox = None
    try:
        reset_bbox = canvas.bbox(reset_bg_id or reset_text_id)
    except Exception:
        reset_bbox = None
    row1_left_x = float((float(reset_bbox[2]) + 8.0) if reset_bbox else 98.0)
    toggle_size = 24.0
    toggle_pad = 12.0
    toggle_x2 = max(toggle_pad + toggle_size, float(canvas_width) - toggle_pad)
    toggle_x1 = toggle_x2 - toggle_size
    toggle_y1 = bar_height + 8.0
    toggle_y2 = toggle_y1 + toggle_size
    host._preview_fullscreen_toggle_rect = (
        float(toggle_x1),
        float(toggle_y1),
        float(toggle_x2),
        float(toggle_y2),
    )
    try:
        canvas.delete("preview_action::toggle_fullscreen")
    except Exception:
        pass
    legend_theme = host._get_preview_legend_theme()
    toggle_fill = str(legend_theme.get("panel_fill", "#1f2933"))
    toggle_outline = str(legend_theme.get("badge_plate_outline", "#2fbf71"))
    toggle_icon = str(legend_theme.get("entry_text", "#f8fafc"))
    canvas.create_rectangle(
        toggle_x1,
        toggle_y1,
        toggle_x2,
        toggle_y2,
        outline=toggle_outline,
        fill=toggle_fill,
        width=1,
        tags=("preview_overlay", "preview_action::toggle_fullscreen"),
    )
    inner_pad = 5.0
    inner_x1 = toggle_x1 + inner_pad
    inner_y1 = toggle_y1 + inner_pad
    inner_x2 = toggle_x2 - inner_pad
    inner_y2 = toggle_y2 - inner_pad
    corner_len = 5.0
    icon_tags = ("preview_overlay", "preview_action::toggle_fullscreen")
    if bool(getattr(host, "_preview_fullscreen_active", False)):
        canvas.create_line(inner_x1 + corner_len, inner_y1, inner_x1, inner_y1, inner_x1, inner_y1 + corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x2 - corner_len, inner_y1, inner_x2, inner_y1, inner_x2, inner_y1 + corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x1 + corner_len, inner_y2, inner_x1, inner_y2, inner_x1, inner_y2 - corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x2 - corner_len, inner_y2, inner_x2, inner_y2, inner_x2, inner_y2 - corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
    else:
        canvas.create_line(inner_x1, inner_y1 + corner_len, inner_x1, inner_y1, inner_x1 + corner_len, inner_y1, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x2, inner_y1 + corner_len, inner_x2, inner_y1, inner_x2 - corner_len, inner_y1, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x1, inner_y2 - corner_len, inner_x1, inner_y2, inner_x1 + corner_len, inner_y2, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)
        canvas.create_line(inner_x2, inner_y2 - corner_len, inner_x2, inner_y2, inner_x2 - corner_len, inner_y2, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=icon_tags)

    row2_y = 40.0
    row2_badges = [
        {
            "text": f"LP: {current_no}/{total}",
            "fill": blend_hex_colors(muted_fg, panel_bg, 0.84),
            "outline": muted_fg,
            "width": 88.0,
            "tags": ("preview_overlay",),
        },
        {
            "text": f"Plik: {source_name}",
            "fill": blend_hex_colors(panel_border, panel_bg, 0.84),
            "outline": panel_border,
            "width": max(160.0, min(240.0, float(canvas_width) * 0.25)),
            "tags": ("preview_overlay", "preview_overlay_action", "preview_action::edit_source_filename"),
        },
        *neutral_status_badges,
        *status_badges,
    ]
    row_start_x = 10.0
    row_gap_x = 8.0
    row_gap_y = 30.0
    row_right_limit = max(row_start_x + 80.0, float(canvas_width) - 10.0)
    badge_x = row_start_x
    badge_y = row2_y
    for badge_spec in row2_badges:
        badge_width = float(badge_spec["width"])
        if badge_x > row_start_x and (badge_x + badge_width) > row_right_limit:
            badge_x = row_start_x
            badge_y += row_gap_y
        host._draw_preview_fixed_text_badge(
            canvas,
            badge_x,
            badge_y,
            badge_width,
            badge_spec["text"],
            fill_color=badge_spec["fill"],
            outline_color=badge_spec["outline"],
            text_color=host._get_readable_text_color(badge_spec["fill"], preferred=title_fg),
            font=("Segoe UI", 9, "bold"),
            pad_x=7,
            pad_y=3,
            tags=badge_spec.get("tags"),
        )
        badge_x += badge_width + row_gap_x

    _sync_preview_canvas_status_pulse(host, canvas, bool(status_layout.get("pulse", False)))

    legend_width = float(host._estimate_preview_source_legend_width())
    legend_height = float(host._estimate_preview_source_legend_height())
    legend_x = badge_x + 2.0
    legend_y = badge_y
    if legend_width > 0:
        if badge_x > row_start_x and (legend_x + legend_width) > row_right_limit:
            legend_x = row_start_x
            legend_y += row_gap_y
        if legend_y + legend_height <= (bar_height - 6.0):
            host._draw_preview_source_legend(canvas, legend_x, legend_y)

    host._place_preview_record_overlay(
        canvas_width,
        bar_height,
        left_x=row1_left_x,
        top_y=6.0,
        right_limit=(toggle_x1 - 8.0),
    )


def refresh_preview_canvas_info_overlay_only(
    host: "CharacterAnnotationTab",
    data: dict | None = None,
    box_chars=None,
) -> bool:
    canvas = getattr(host, "preview_canvas", None)
    if canvas is None or not getattr(host, "_preview_render_state", None):
        return False

    source_data = data if isinstance(data, dict) else host._get_preview_active_data(create=False)
    if not isinstance(source_data, dict):
        return False

    if box_chars is None:
        box_chars = source_data.get("characters", [])
    if not isinstance(box_chars, list):
        box_chars = []

    try:
        canvas.delete("preview_overlay")
        canvas.delete("preview_overlay_action")
        canvas.delete("preview_plate_status_frame")
        host._draw_preview_plate_status_frame(source_data)
        host._draw_preview_canvas_info_overlay(
            canvas,
            max(50, int(canvas.winfo_width() or 50)),
            data=source_data,
            box_chars=box_chars,
            has_boxes=bool(box_chars),
        )
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
        return True
    except Exception as exc:
        logger.debug(f"Nie udało się odświeżyć paska liczników PZ2: {exc}")
        return False


def apply_preview_fullscreen_chrome(host: "CharacterAnnotationTab"):
    preview_tools = getattr(host, "preview_tools", None)
    tools_hidden_by_design = bool(getattr(host, "_preview_tools_hidden_by_design", False))

    if bool(getattr(host, "_preview_fullscreen_active", False)):
        try:
            if preview_tools is not None:
                preview_tools.grid_remove()
        except Exception:
            pass
        try:
            host._place_preview_overlay_dock(force_render=True)
        except Exception:
            pass
        try:
            place_preview_hint_overlay(host, refresh=True)
        except Exception:
            pass
        host._sync_preview_edit_status_visibility()
        host._refresh_preview_typing_overlay_visibility()
        return

    try:
        if preview_tools is not None:
            if tools_hidden_by_design:
                preview_tools.grid_remove()
            else:
                preview_tools.grid()
    except Exception:
        pass
    try:
        host._place_preview_overlay_dock(force_render=True)
    except Exception:
        pass
    try:
        place_preview_hint_overlay(host, refresh=True)
    except Exception:
        pass
    host._sync_preview_edit_status_visibility()
    host._refresh_preview_typing_overlay_visibility()


def update_preview_toolbar_state(host: "CharacterAnnotationTab"):
    total = len(getattr(host, "_listbox_pid_by_index", []))
    current_idx = host._get_current_preview_list_index()
    has_selection = current_idx is not None and total > 0
    has_image = bool(getattr(host, "_preview_render_state", None))
    can_go_prev = has_selection and int(current_idx) > 0
    can_go_next = has_selection and int(current_idx) < (total - 1)

    button_specs = (
        ("preview_prev_btn", can_go_prev),
        ("preview_next_btn", can_go_next),
        ("preview_fit_btn", has_image),
        ("preview_fullscreen_btn", has_image),
    )
    for attr_name, enabled in button_specs:
        widget = getattr(host, attr_name, None)
        if widget is None:
            continue
        try:
            widget.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    fullscreen_btn = getattr(host, "preview_fullscreen_btn", None)
    if fullscreen_btn is not None:
        try:
            fullscreen_btn.configure(
                text=("Wyjdź z pełnego ekranu (Enter)" if host._preview_fullscreen_active else "Pełny ekran (Enter)")
            )
        except Exception:
            pass


def set_preview_fullscreen(host, active: bool):
    self = host
    next_state = bool(active)
    if next_state == bool(getattr(self, "_preview_fullscreen_active", False)):
        if next_state:
            self._focus_preview_canvas()
        return

    has_image = bool(getattr(self, "_preview_render_state", None))
    if next_state and not has_image:
        self._update_preview_edit_status("Pełny ekran jest dostępny po załadowaniu obrazu podglądu.", tone="warning")
        return

    self._preview_fullscreen_transition_active = True
    self._preview_fullscreen_transition_started = time.perf_counter()
    self._preview_fullscreen_old_canvas_size = (
        self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height(),
    )
    root = getattr(self.app, "root", None)
    try:
        windowing_system = str(self.frame.tk.call("tk", "windowingsystem")).lower()
    except Exception:
        windowing_system = ""
    use_native_root_fullscreen = windowing_system not in {"win32"}

    split = getattr(self, "detect_split", None)
    right_panel = getattr(self, "detect_right_panel", None)
    preview_split = getattr(self, "preview_vertical_split", None)
    list_panel = getattr(self, "preview_list_lf", None)
    footer_nav = getattr(self, "detect_footer_nav", None)
    log_frame = getattr(self, "detection_log_frame", None)

    if next_state:
        self._preview_fullscreen_restore_panel_widths = {
            "right": right_panel.winfo_width() if right_panel is not None else 300,
            "list": list_panel.winfo_width() if list_panel is not None else 240,
        }
        self._preview_fullscreen_restore_log_visible = bool(getattr(self, "_detection_log_visible", False))
        try:
            self._preview_fullscreen_restore_root_state = bool(root.attributes("-fullscreen")) if root is not None else False
        except Exception:
            self._preview_fullscreen_restore_root_state = False
        try:
            self._preview_fullscreen_restore_window_state = str(root.state()) if root is not None else "normal"
        except Exception:
            self._preview_fullscreen_restore_window_state = "normal"
        try:
            self._preview_fullscreen_restore_geometry = str(root.geometry()) if root is not None else ""
        except Exception:
            self._preview_fullscreen_restore_geometry = ""

        try:
            if self._pane_has_child(split, right_panel):
                split.forget(right_panel)
        except Exception:
            pass
        try:
            if self._pane_has_child(preview_split, list_panel):
                preview_split.forget(list_panel)
        except Exception:
            pass
        try:
            if log_frame is not None:
                log_frame.grid_remove()
        except Exception:
            pass
        try:
            if footer_nav is not None:
                footer_nav.grid_remove()
        except Exception:
            pass

        self._set_detection_process_log_visibility(False)
        if root is not None:
            try:
                if use_native_root_fullscreen:
                    root.attributes("-fullscreen", True)
                else:
                    root.attributes("-fullscreen", False)
                    try:
                        root.state("zoomed")
                    except Exception:
                        screen_w = int(root.winfo_screenwidth())
                        screen_h = int(root.winfo_screenheight())
                        root.geometry(f"{screen_w}x{screen_h}+0+0")
            except Exception:
                pass
        self._preview_fullscreen_active = True
        self._preview_mode_overlay_position = self._get_preview_mode_overlay_default_position(fullscreen=True)
    else:
        if root is not None:
            try:
                if use_native_root_fullscreen:
                    root.attributes("-fullscreen", bool(getattr(self, "_preview_fullscreen_restore_root_state", False)))
                else:
                    root.attributes("-fullscreen", False)
                    restore_state = str(getattr(self, "_preview_fullscreen_restore_window_state", "normal") or "normal")
                    restore_geometry = str(getattr(self, "_preview_fullscreen_restore_geometry", "") or "")
                    try:
                        root.state(restore_state if restore_state in {"normal", "zoomed"} else "normal")
                    except Exception:
                        pass
                    if restore_state != "zoomed" and restore_geometry:
                        try:
                            root.geometry(restore_geometry)
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            if not self._pane_has_child(split, right_panel) and right_panel is not None:
                panel_widths = getattr(self, "_preview_fullscreen_restore_panel_widths", {})
                split.add(right_panel, minsize=300, stretch="never", width=panel_widths.get("right", 300))
                try:
                    split.paneconfigure(right_panel, minsize=300, stretch="never")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if not self._pane_has_child(preview_split, list_panel) and list_panel is not None:
                panel_widths = getattr(self, "_preview_fullscreen_restore_panel_widths", {})
                preview_split.add(list_panel, minsize=240, width=panel_widths.get("list", 240),
                                  before=getattr(self, "preview_lf", None))
        except Exception:
            pass
        try:
            if footer_nav is not None:
                footer_nav.grid()
        except Exception:
            pass
        self._set_detection_process_log_visibility(bool(getattr(self, "_preview_fullscreen_restore_log_visible", False)))
        try:
            self._sync_detect_right_canvas_width()
            self._sync_detect_right_scrollregion()
        except Exception:
            pass
        self._preview_fullscreen_active = False

    # Resize once after Tk applies the final pane and window geometry. The
    # old synchronous render was repeated by the pending Configure callback.
    preview_tools = getattr(self, "preview_tools", None)
    if preview_tools is not None:
        if next_state or getattr(self, "_preview_tools_hidden_by_design", False):
            preview_tools.grid_remove()
        else:
            preview_tools.grid()
    self._sync_preview_edit_status_visibility()
    self._update_preview_toolbar_state()
    self._schedule_preview_stabilized_rerender(delay_ms=90)
    self._focus_preview_canvas()


def reset_preview_cache(host):
    mark_preview_metadata_changed(host)
    if bool(getattr(host, "_preview_fullscreen_active", False)):
        try:
            host._set_preview_fullscreen(False)
        except Exception:
            pass
    host._flush_scheduled_preview_metadata_save()
    writer = getattr(host, "_preview_autosave_writer", None)
    if writer is not None:
        writer.close()
        host._preview_autosave_writer = None
    host.preview_metadata = {}
    host._preview_base_plate_ids = []
    host._loaded_meta_path = None
    host._loaded_meta_mtime = None
    host._preview_active_pid = None
    try:
        zoom_after_id = getattr(host, "_preview_zoom_anim_after_id", None)
        canvas = getattr(host, "preview_canvas", None)
        if zoom_after_id and canvas is not None:
            canvas.after_cancel(zoom_after_id)
    except Exception:
        pass
    host._preview_zoom_anim_after_id = None
    host._preview_zoom_level = 1.0
    host._preview_zoom_target = 1.0
    host._preview_zoom_velocity = 0.0
    host._preview_zoom_last_ts = None
    host._preview_zoom_last_input_ts = None
    host._preview_zoom_anchor = None
    host._preview_zoom_pending_state = None
    host._preview_pan_x = 0.0
    host._preview_pan_y = 0.0
    host._preview_render_state = {}
    host._preview_source_image_cache = OrderedDict()
    host._preview_resized_photo_cache = OrderedDict()
    host._preview_badge_offsets = {}
    host._preview_badge_runtime = {}
    host._preview_badge_drag_state = None
    host._preview_layout_separator_runtime = {}
    host._preview_layout_separator_drag_state = None
    host._preview_pan_drag_state = None
    host._preview_selected_badge_key = None
    host._preview_mode_overlay_position = None
    host._preview_mode_eye_drag_state = None
    host._preview_char_selected_index = None
    host._unbind_preview_char_drag_session()
    host._preview_char_drag_state = None
    host._preview_char_add_state = None
    host._preview_char_edit_mode = False
    host._preview_char_add_mode = False
    host._preview_char_add_modifier_down = False
    host._preview_char_add_click_armed = False
    host._preview_alt_modifier_down = False
    host._preview_char_label_mode = False
    host._preview_char_hover_index = None
    host._preview_char_hover_grip = None
    host._preview_char_hover_label_index = None
    host._preview_char_label_active_index = None
    host._preview_char_record_render_tags = {}
    host._preview_history_undo = {}
    host._preview_history_redo = {}
    host._preview_history_replaying = False
    host._preview_metadata_save_after_id = None
    host._preview_import_focus_plate_ids = []
    host._preview_import_focus_batch_id = ""
    host._preview_import_focus_active = False
    host._preview_status_counts_snapshot = {
        "perfect": 0,
        "needs_fix": 0,
        "unknown": 0,
        "total": 0,
        "strategy_counts": {},
        "strategy_char_counts": {},
    }
    host._apply_preview_canvas_cursor("arrow")
    try:
        host._update_preview_record_source_label(None)
        host._refresh_preview_import_focus_ui()
    except Exception:
        pass


def reset_preview_view_state(host):
    host._preview_zoom_level = 1.0
    host._preview_zoom_target = 1.0
    host._preview_zoom_velocity = 0.0
    host._preview_zoom_last_ts = None
    host._preview_zoom_last_input_ts = None
    host._preview_zoom_anchor = None
    host._preview_pan_x = 0.0
    host._preview_pan_y = 0.0
    host._preview_render_state = {}
    host._preview_badge_runtime = {}
    host._preview_badge_drag_state = None
    host._preview_layout_separator_runtime = {}
    host._preview_layout_separator_drag_state = None
    host._preview_pan_drag_state = None
    host._preview_selected_badge_key = None
    host._preview_char_selected_index = None
    host._unbind_preview_char_drag_session()
    host._preview_char_drag_state = None
    host._preview_char_add_state = None
    host._preview_char_add_modifier_down = False
    host._preview_char_add_click_armed = False
    host._preview_alt_modifier_down = False
    host._preview_char_hover_index = None
    host._preview_char_hover_grip = None
    host._preview_char_hover_label_index = None
    host._preview_char_label_active_index = None
    host._preview_char_record_render_tags = {}
    host._apply_preview_canvas_cursor("arrow")


def _preview_metadata_has_character_records(metadata) -> bool:
    if not isinstance(metadata, dict):
        return False
    for data in metadata.values():
        if not isinstance(data, dict):
            continue
        characters = data.get("characters")
        if isinstance(characters, list) and characters:
            return True
    return False


def _rescue_empty_preview_characters_from_history(host, preview_dir, metadata=None):
    if _preview_metadata_has_character_records(metadata):
        return None
    try:
        if not CAMPAIGN.get_active_project_name():
            return None
    except Exception:
        return None

    try:
        preview_path = Path(preview_dir)
    except Exception:
        return None
    meta_path = preview_path / "metadata.json"
    if not meta_path.exists():
        return None

    try:
        rescue_summary = host._merge_reextract_seed_metadata_into_preview(
            preview_path,
            allow_historical_fallback=True,
        )
    except Exception as rescue_exc:
        logger.debug(f"Nie udalo sie zaadoptowac historycznych znakow PZ2: {rescue_exc}")
        return None

    matched = int((rescue_summary or {}).get("matched", 0) or 0)
    if matched <= 0:
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as rescued_file:
            rescued_loaded = json.load(rescued_file)
    except Exception:
        return None
    if not isinstance(rescued_loaded, dict):
        return None

    try:
        logger.info(
            "[PZ2 load_preview_data] rescued_historical_characters matched=%s total=%s path=%s",
            matched,
            int((rescue_summary or {}).get("total", 0) or 0),
            meta_path,
        )
    except Exception:
        pass
    return rescued_loaded


def ensure_detection_preview_loaded(host, force_reload: bool = False) -> bool:
    if not hasattr(host, "plates_listbox"):
        return False

    preview_dir = ""
    try:
        in_campaign_context = bool(
            getattr(host, "_step3_linear_mode", False)
            or CAMPAIGN.get_active_project_name()
        )
        if in_campaign_context:
            saved_preview_dir = str(host._get_saved_step3_preview_dir(require_plates=True) or "").strip()
            if saved_preview_dir and host._is_usable_step3_preview_dir(
                saved_preview_dir,
                require_plates=True,
                check_campaign_inflated=False,
            ):
                preview_dir = saved_preview_dir
    except Exception:
        preview_dir = ""

    if not preview_dir:
        preview_dir = str(host.preview_dir_var.get() or "").strip()
        if preview_dir and not host._is_extract_preview_ready(preview_dir):
            preview_dir = ""

    if not host._preview_dir_has_plate_entries(preview_dir):
        preview_dir = host._get_preferred_step3_preview_dir(require_plates=True, allow_fallback=False)
        if not preview_dir:
            preview_dir = host._find_latest_extract_preview_run_dir(require_plates=True)
        if not preview_dir:
            preview_dir = host._get_preferred_step3_preview_dir(require_plates=True, allow_fallback=True)
        if preview_dir and not host._is_extract_preview_ready(preview_dir):
            preview_dir = ""
        if not preview_dir:
            return False

    meta_path = Path(preview_dir) / "metadata.json"
    current_meta_path = getattr(host, "_loaded_meta_path", None)
    preview_changed = preview_dir != str(host.preview_dir_var.get() or "").strip()
    needs_reload = (
        force_reload
        or preview_changed
        or not bool(host.preview_metadata)
        or current_meta_path != meta_path
    )

    if preview_changed:
        try:
            host.preview_dir_var.set(preview_dir)
        except Exception:
            return False

    if not needs_reload:
        rescued_loaded = _rescue_empty_preview_characters_from_history(
            host,
            preview_dir,
            getattr(host, "preview_metadata", None),
        )
        if isinstance(rescued_loaded, dict):
            try:
                host._loaded_meta_mtime = meta_path.stat().st_mtime
            except Exception:
                pass
            try:
                host._apply_preview_metadata_update(rescued_loaded, preserve_selection=True)
            except Exception:
                return False
        if not getattr(host, "_listbox_pid_by_index", None) and host.preview_metadata:
            try:
                host._apply_preview_metadata_update(host.preview_metadata, preserve_selection=True)
            except Exception:
                return False
        return bool(host.preview_plate_ids or host.preview_metadata)

    try:
        host._reset_preview_cache()
        host._load_preview_data(quiet=True)
    except Exception as exc:
        logger.debug(f"Nie udało się automatycznie wczytać zestawu PZ2: {exc}")
        return False

    return bool(host.preview_plate_ids)


def open_detection_subtab_with_preview(host, preview_dir=None, *, force_reload: bool = True) -> bool:
    """
    Otwiera PZ2 i ładuje konkretny preview run.

    Po świeżym wyodrębnieniu PZ1 znamy dokładny katalog runu, więc nie
    opieramy się wyłącznie na późniejszym autoloadzie i wyszukiwaniu
    fallbacków. To chroni tryb swobodny przed pustą listą PZ2 po skoku z PZ1.
    """
    preview_dir_text = str(preview_dir or "").strip()
    if preview_dir_text:
        try:
            host.preview_dir_var.set(str(Path(preview_dir_text)))
        except Exception:
            try:
                host.preview_dir_var.set(preview_dir_text)
            except Exception:
                pass

    try:
        host._set_subtab_state(host.tab_detect, "normal")
    except Exception:
        pass

    if not host._ensure_detect_tab_built():
        return False

    try:
        host.main_nb.select(str(host.tab_detect))
    except Exception as exc:
        logger.debug(f"Nie udało się przełączyć do PZ2 po wyodrębnianiu: {exc}")

    effective_preview_dir = preview_dir_text
    if not effective_preview_dir:
        try:
            effective_preview_dir = str(host.preview_dir_var.get() or "").strip()
        except Exception:
            effective_preview_dir = ""
    if not effective_preview_dir:
        return False

    if force_reload:
        try:
            host._reset_preview_cache()
        except Exception:
            pass

    loaded = False
    try:
        host._load_preview_data(quiet=True)
        loaded = bool(getattr(host, "_listbox_pid_by_index", None))
    except Exception as exc:
        logger.debug(f"Nie udało się wczytać preview PZ2 po wyodrębnianiu: {exc}")
        loaded = False

    if loaded:
        try:
            if getattr(host, "_listbox_pid_by_index", None) and not host.plates_listbox.curselection():
                host.plates_listbox.selection_set(0)
                host.plates_listbox.activate(0)
                host.plates_listbox.see(0)
        except Exception:
            pass
        try:
            host.frame.after_idle(lambda: host._on_preview_select(None))
        except Exception:
            try:
                host._on_preview_select(None)
            except Exception:
                pass

    return bool(loaded)


def persist_active_preview_characters(
    host,
    *,
    selected_record=None,
    success_message: str,
    render_preview: bool = True,
    save_immediately: bool = True,
    save_delay_ms: int = 450,
    refresh_row: bool = True,
    light_redraw_indices=None,
):
    mark_preview_metadata_changed(host)
    perf_start = time.perf_counter()
    phase_start = perf_start
    prepare_ms = live_ui_ms = redraw_ms = save_ms = 0.0
    data = host._get_preview_active_data(create=True)
    if isinstance(data, dict):
        data.pop("_layout_override_chars_backup", None)
    previous_chars = list(data.get("characters", [])) if isinstance(data.get("characters"), list) else []
    previous_status = str(data.get("status", "unknown") or "unknown").strip().lower()
    chars = host._get_preview_active_character_records(create=True)
    selection_marker_key = "__preview_selected_marker"
    selection_marker_value = f"selected:{id(selected_record)}"
    if isinstance(selected_record, dict):
        selected_record[selection_marker_key] = selection_marker_value
    previous_selected_index = None
    if isinstance(selected_record, dict):
        for idx, rec in enumerate(chars):
            if rec is selected_record:
                previous_selected_index = idx
                break
    normalized_chars = []
    for rec in list(chars):
        if not isinstance(rec, dict):
            continue
        bbox = host._normalize_preview_char_bbox(rec.get("bbox"))
        if bbox is None:
            continue
        rec["bbox"] = bbox
        normalized_chars.append(rec)

    sorted_chars = host._sort_character_records_by_x(normalized_chars, data=data)
    host._update_preview_plate_layout_metadata(data, sorted_chars)
    sorted_chars = host._annotate_preview_character_reading_positions(sorted_chars, data=data)
    data["characters"] = sorted_chars
    status_now = host._derive_preview_status_from_data(data, sorted_chars)
    if (
        status_now == "perfect"
        and previous_status == "perfect"
        and len(sorted_chars) != len(previous_chars)
        and not host._get_preview_expected_texts(data)
    ):
        status_now = "needs_fix"
    data["status"] = status_now
    data["fusion_strategy"] = "manual_correction"
    data["fusion_details"] = {"source": "preview_editor"}
    host._ensure_plate_source_metadata(
        data,
        plate_id=str(getattr(host, "_preview_active_pid", "") or data.get("plate_id", "") or ""),
        default_bucket="local_manual",
        default_origin="preview_editor",
        modified_by="human",
    )
    prepare_ms = (time.perf_counter() - phase_start) * 1000.0

    if selected_record is not None:
        host._preview_char_selected_index = None
        for idx, rec in enumerate(sorted_chars):
            if rec is selected_record or rec.get(selection_marker_key) == selection_marker_value:
                host._preview_char_selected_index = idx
                break
    elif host._preview_char_selected_index is not None:
        host._preview_char_selected_index = (
            min(int(host._preview_char_selected_index), len(sorted_chars) - 1)
            if sorted_chars
            else None
        )
    for rec in sorted_chars:
        if isinstance(rec, dict):
            rec.pop(selection_marker_key, None)

    resolved_selected_index = getattr(host, "_preview_char_selected_index", None)
    status_suffix = "Status tablicy: OK." if status_now == "perfect" else "Status tablicy: wymaga korekty."
    live_message = f"{success_message} {status_suffix}".strip()
    live_tone = "success" if status_now == "perfect" else "info"
    # Provenance and the text can change while quality stays needs_fix/perfect.
    # Refresh only this row, also for the lightweight drag/typing save path.
    row_refresh_needed = True

    status_frame_needed = status_now in {"perfect", "bad", "needs_fix"} or previous_status in {
        "perfect",
        "bad",
        "needs_fix",
    }
    if status_frame_needed:
        phase_start = time.perf_counter()
        try:
            host._draw_preview_plate_status_frame(data)
            canvas = getattr(host, "preview_canvas", None)
            if canvas is not None:
                canvas.update_idletasks()
        except Exception:
            pass
        status_frame_ms = (time.perf_counter() - phase_start) * 1000.0
        if status_frame_ms >= 40.0:
            try:
                logger.info(
                    "[Z3/PZ2 PERF] immediate_status_frame status=%s ms=%.1f",
                    status_now,
                    status_frame_ms,
                )
            except Exception:
                pass

    phase_start = time.perf_counter()
    host._refresh_preview_live_metadata_ui(
        status_message=live_message,
        status_tone=live_tone,
        render_preview=bool(render_preview),
        refresh_row=row_refresh_needed,
    )
    live_ui_ms = (time.perf_counter() - phase_start) * 1000.0
    if not bool(render_preview):
        phase_start = time.perf_counter()
        redraw_indices = None
        partial_redraw_allowed = True
        if light_redraw_indices == "selected":
            try:
                partial_redraw_allowed = (
                    previous_selected_index == resolved_selected_index
                    and len(previous_chars) == len(sorted_chars)
                )
            except Exception:
                partial_redraw_allowed = False
        if light_redraw_indices is not None and partial_redraw_allowed:
            raw_indices = (
                [resolved_selected_index, previous_selected_index]
                if light_redraw_indices == "selected"
                else light_redraw_indices
            )
            if not isinstance(raw_indices, (list, tuple, set, frozenset)):
                raw_indices = [raw_indices]
            redraw_indices = set()
            for raw_idx in raw_indices:
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if 0 <= idx < len(sorted_chars):
                    redraw_indices.add(idx)

        redrawn = False
        if redraw_indices:
            try:
                host._draw_preview_plate_status_frame(data)
            except Exception:
                pass
            for idx in sorted(redraw_indices):
                redrawn = host._redraw_preview_character_overlay_only(idx) or redrawn
            try:
                canvas = getattr(host, "preview_canvas", None)
                host._apply_preview_badge_selection_style()
                if canvas is not None:
                    canvas.tag_raise("preview_char_add_preview")
                    canvas.tag_raise("preview_overlay")
                    canvas.tag_raise("preview_overlay_action")
            except Exception:
                pass
        else:
            redrawn = host._redraw_preview_character_overlays_light()
        try:
            refresh_preview_canvas_info_overlay_only(host, data=data, box_chars=sorted_chars)
        except Exception:
            pass
        if not redrawn:
            host._on_preview_select(None)
        redraw_ms = (time.perf_counter() - phase_start) * 1000.0
    if bool(render_preview):
        phase_start = time.perf_counter()
        try:
            host.frame.update_idletasks()
        except Exception:
            pass
        redraw_ms += (time.perf_counter() - phase_start) * 1000.0
    phase_start = time.perf_counter()
    if bool(save_immediately):
        host._persist_preview_metadata(success_message=None, refresh_list=False, sync_access=False)
        try:
            host._schedule_preview_info_refresh(delay_ms=900)
        except Exception:
            pass
    else:
        host._schedule_preview_metadata_save(delay_ms=save_delay_ms)
    try:
        clear_detection_review_snapshot_after_manual_edit(host)
    except Exception:
        pass
    save_ms = (time.perf_counter() - phase_start) * 1000.0
    _log_preview_perf(
        "persist_active_preview_chars",
        perf_start,
        threshold_ms=90.0,
        prepare=prepare_ms,
        live_ui=live_ui_ms,
        redraw=redraw_ms,
        save=save_ms,
        chars=len(sorted_chars),
    )
    log_preview_edit_flow(
        host,
        "metadata_update",
        selected=resolved_selected_index if resolved_selected_index is not None else "-",
        chars=len(sorted_chars),
        status_from=previous_status,
        status_to=status_now,
        render=int(bool(render_preview)),
        save_now=int(bool(save_immediately)),
        save_delay_ms=save_delay_ms if not bool(save_immediately) else 0,
    )


def load_preview_data(host, quiet=False):
    self = host
    load_started = time.perf_counter()
    read_ms = normalize_ms = write_ms = apply_ms = source_panel_ms = 0.0
    out_dir = Path(self.preview_dir_var.get().strip())
    meta_path = out_dir / "metadata.json"
    try:
        if getattr(self, "_loaded_meta_path", None) == meta_path:
            self._flush_scheduled_preview_metadata_save()
    except Exception:
        pass

    try:
        self.frame.after(0, self._update_winner_label)
    except Exception:
        pass

    if not meta_path.exists():
        if not quiet:
            messagebox.showerror("Brak pliku", f"Nie znaleziono metadata.json w folderze:\n{out_dir}")
        try:
            self._set_preview_info("Brak wczytanych danych", "error")
            self._set_preview_counts_info(0, 0, 0)
            self._set_preview_box_info("Źródło ramek znaków: brak wczytanych danych", "muted")
            self._refresh_preview_bound_action_states()
        except Exception:
            pass
        return

    try:
        current_mtime = meta_path.stat().st_mtime
        need_reload = (
            self._loaded_meta_path != meta_path
            or self._loaded_meta_mtime != current_mtime
        )

        if (not self.preview_metadata) or (not quiet) or need_reload:
            phase_started = time.perf_counter()
            loaded = read_preview_metadata(self, meta_path, prefer_live=False)
            read_ms = (time.perf_counter() - phase_started) * 1000.0

            if not isinstance(loaded, dict):
                loaded = {}

            try:
                campaign_context = bool(
                    getattr(self, "_campaign_controlled", False)
                    or getattr(self, "_campaign_step3_context_active", False)
                    or getattr(self, "_campaign_pz2_sync_loading", False)
                )
                needs_reextract_rescue = bool(loaded) and not _preview_metadata_has_character_records(loaded)
                if campaign_context and needs_reextract_rescue:
                    rescued_loaded = _rescue_empty_preview_characters_from_history(
                        self,
                        out_dir,
                        loaded,
                    )
                    if isinstance(rescued_loaded, dict):
                        loaded = rescued_loaded
                        current_mtime = meta_path.stat().st_mtime
            except Exception as rescue_exc:
                logger.debug(f"Nie udało się uzupełnić metadata PZ2 po reekstrakcji: {rescue_exc}")

            def _preview_layout_metadata_ready(record: dict) -> bool:
                if not isinstance(record, dict):
                    return False
                return bool(
                    str(record.get("plate_layout", "") or "").strip()
                    and "layout_row_count" in record
                    and "layout_confidence" in record
                    and str(record.get("layout_source", "") or "").strip()
                    and str(record.get("plate_layout_inferred", "") or "").strip()
                    and "layout_inferred_row_count" in record
                    and "layout_inferred_confidence" in record
                    and str(record.get("layout_inferred_source", "") or "").strip()
                )

            def _preview_characters_order_ready(records) -> bool:
                if not isinstance(records, list):
                    return False
                for rec in records:
                    if not isinstance(rec, dict):
                        return False
                    if (
                        "reading_index" not in rec
                        or "reading_row" not in rec
                        or "reading_col" not in rec
                    ):
                        return False
                return True

            def _preview_source_metadata_ready(record: dict) -> bool:
                if not isinstance(record, dict):
                    return False
                source_info = record.get("source_info")
                gold_state = record.get("gold_state")
                if not isinstance(source_info, dict) or not isinstance(gold_state, dict):
                    return False
                if not str(source_info.get("bucket", "") or "").strip():
                    return False
                if not str(source_info.get("origin", "") or "").strip():
                    return False
                for list_key in ("characters", "yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
                    records = record.get(list_key)
                    if not isinstance(records, list):
                        continue
                    for rec in records:
                        if not isinstance(rec, dict):
                            return False
                        if "source_kind" not in rec or "source_batch_id" not in rec:
                            return False
                return True

            phase_started = time.perf_counter()
            changed = False
            if self._backfill_preview_expected_texts_from_sources(loaded):
                changed = True
            for pid, d in loaded.items():
                if not isinstance(d, dict):
                    continue
                if "characters" not in d or not isinstance(d.get("characters"), list):
                    d["characters"] = []
                    changed = True
                else:
                    if not (quiet and _preview_layout_metadata_ready(d)):
                        layout_before = (
                            d.get("plate_layout"),
                            d.get("layout_row_count"),
                            d.get("layout_confidence"),
                            d.get("layout_source"),
                            d.get("plate_layout_inferred"),
                            d.get("layout_inferred_row_count"),
                            d.get("layout_inferred_confidence"),
                            d.get("layout_inferred_source"),
                        )
                        self._update_preview_plate_layout_metadata(d, d.get("characters", []))
                        layout_after = (
                            d.get("plate_layout"),
                            d.get("layout_row_count"),
                            d.get("layout_confidence"),
                            d.get("layout_source"),
                            d.get("plate_layout_inferred"),
                            d.get("layout_inferred_row_count"),
                            d.get("layout_inferred_confidence"),
                            d.get("layout_inferred_source"),
                        )
                        if layout_after != layout_before:
                            changed = True
                    if not (quiet and _preview_characters_order_ready(d.get("characters", []))):
                        sorted_chars = self._sort_character_records_by_x(d.get("characters", []), data=d)
                        sorted_chars = self._annotate_preview_character_reading_positions(sorted_chars, data=d)
                        if sorted_chars != d.get("characters", []):
                            d["characters"] = sorted_chars
                            changed = True
                current_status = str(d.get("status", "") or "").strip().lower()
                should_recalculate_status = current_status in {"", "unknown"}
                if not should_recalculate_status and current_status == "needs_fix":
                    try:
                        should_recalculate_status = bool(self._get_preview_expected_texts(d)) and (
                            self._derive_preview_status_from_characters(d.get("characters", [])) == "perfect"
                        )
                    except Exception:
                        should_recalculate_status = False
                if should_recalculate_status:
                    next_status = self._derive_preview_status_from_data(d, d.get("characters", []))
                    if next_status != current_status:
                        d["status"] = next_status
                        changed = True
                for yolo_key in ("yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
                    if yolo_key not in d:
                        continue
                    if not isinstance(d.get(yolo_key), list):
                        d[yolo_key] = []
                        changed = True
                        continue
                    if quiet:
                        continue
                    sorted_yolo = self._sort_character_records_by_x(d.get(yolo_key, []))
                    if sorted_yolo != d.get(yolo_key, []):
                        d[yolo_key] = sorted_yolo
                        changed = True
                if not (quiet and _preview_source_metadata_ready(d)) and self._ensure_plate_source_metadata(d, plate_id=str(pid or ""), meta_path=meta_path):
                    changed = True
            normalize_ms = (time.perf_counter() - phase_started) * 1000.0

            if changed and not quiet:
                phase_started = time.perf_counter()
                self._atomic_write_json(meta_path, loaded)
                current_mtime = meta_path.stat().st_mtime
                write_ms = (time.perf_counter() - phase_started) * 1000.0

            self.preview_metadata = loaded
            self._loaded_meta_path = meta_path
            self._loaded_meta_mtime = current_mtime

        phase_started = time.perf_counter()
        self._apply_preview_metadata_update(
            self.preview_metadata,
            preserve_selection=True,
            render_selection=False,
            recalculate_statuses=not bool(quiet),
        )
        self._sync_step3_access_from_preview_state(self.preview_metadata)
        try:
            self._refresh_last_detection_status_label()
        except Exception:
            pass
        apply_ms = (time.perf_counter() - phase_started) * 1000.0

        try:
            self.frame.after(100, self._update_winner_label)
        except Exception:
            pass

    except Exception as e:
        if not quiet:
            messagebox.showerror("Błąd odświeżania listy", str(e))
        logger.error(f"Błąd _load_preview_data: {e}")
    finally:
        self._reloading_preview = False
        try:
            self._refresh_preview_bound_action_states()
        except Exception:
            pass

    try:
        pid_map = getattr(self, "_listbox_pid_by_index", []) or []
        if self.preview_plate_ids and hasattr(self, "plates_listbox"):
            if not self.plates_listbox.curselection() and pid_map:
                self.plates_listbox.selection_set(0)
                self.plates_listbox.activate(0)
                self.plates_listbox.see(0)
            if self.plates_listbox.curselection():
                scheduler = getattr(self, "_schedule_preview_select_render", None)
                if callable(scheduler):
                    scheduler(delay_ms=20)
                else:
                    self.frame.after_idle(lambda: self._on_preview_select(None))
    except Exception as e:
        try:
            self._preview_fast_select_render = False
        except Exception:
            pass
        try:
            self._preview_fast_select_render = False
        except Exception:
            pass
        logger.debug(f"Nie udało się odświeżyć preview po _load_preview_data: {e}")

    def _refresh_source_panel_later():
        nonlocal source_panel_ms
        try:
            phase_started = time.perf_counter()
            self._refresh_preview_source_panel()
            source_panel_ms = (time.perf_counter() - phase_started) * 1000.0
        except Exception:
            source_panel_ms = 0.0

    try:
        if quiet:
            self.frame.after_idle(_refresh_source_panel_later)
        else:
            _refresh_source_panel_later()
    except Exception:
        _refresh_source_panel_later()

    total_ms = (time.perf_counter() - load_started) * 1000.0
    if total_ms >= 250.0:
        try:
            logger.info(
                "[PZ2 load_preview_data] quiet=%s count=%s total=%.1fms read=%.1fms normalize=%.1fms write=%.1fms apply=%.1fms right_panel=%.1fms",
                bool(quiet),
                len(getattr(self, "preview_metadata", {}) or {}),
                total_ms,
                read_ms,
                normalize_ms,
                write_ms,
                apply_ms,
                source_panel_ms,
            )
        except Exception:
            pass


def on_preview_select(host, event=None):
    self = host
    """Podgląd tablicy + bboxy znaków."""
    if not PIL_AVAILABLE:
        return
    canvas = getattr(self, "preview_canvas", None)
    if canvas is not None and (not canvas.winfo_ismapped() or canvas.winfo_width() <= 1 or canvas.winfo_height() <= 1):
        # T05 builds this tab before it is mapped. Drawing at 1x1 computes the
        # wrong image/box transform and queues competing overlay layouts.
        self._preview_render_when_visible = True
        return
    self._preview_render_when_visible = False
    fast_select_render = bool(getattr(self, "_preview_fast_select_render", False))
    render_profile_start = time.perf_counter()
    render_profile_marks: dict[str, float] = {}

    def _mark_render_profile(name: str) -> None:
        try:
            render_profile_marks[name] = (time.perf_counter() - render_profile_start) * 1000.0
        except Exception:
            pass

    # jeśli akurat przebudowujemy listę - nie renderujemy
    if getattr(self, "_reloading_preview", False):
        return
    if getattr(self, "_suppress_preview_reload_on_list_select", False):
        self._suppress_preview_reload_on_list_select = False
        if not bool(getattr(self, "_preview_fast_select_render", False)):
            self._refresh_preview_editor_toolbar()
        return

    pid_map = getattr(self, "_listbox_pid_by_index", [])
    sel = self.plates_listbox.curselection()
    if not sel:
        try:
            active_idx = int(self.plates_listbox.index(tk.ACTIVE))
        except Exception:
            active_idx = -1

        if 0 <= active_idx < len(pid_map):
            try:
                self._clear_listbox_selection_fast(self.plates_listbox)
                self.plates_listbox.selection_set(active_idx)
                self.plates_listbox.activate(active_idx)
                self.plates_listbox.see(active_idx)
            except Exception:
                pass
            sel = (active_idx,)
        else:
            self._cancel_preview_char_label_interaction()
            self._preview_char_selected_index = None
            self._preview_char_hover_index = None
            self._preview_char_hover_grip = None
            self._apply_preview_canvas_cursor("arrow")
            self._refresh_preview_editor_toolbar()
            self._update_preview_box_info_label()
            self._update_preview_record_source_label(None)
            return

    try:
        idx = int(sel[0])
    except Exception:
        self._cancel_preview_char_label_interaction()
        self._preview_char_selected_index = None
        self._preview_char_hover_index = None
        self._preview_char_hover_grip = None
        self._apply_preview_canvas_cursor("arrow")
        self._refresh_preview_editor_toolbar()
        self._update_preview_record_source_label(None)
        return

    if not (0 <= idx < len(pid_map)):
        self._cancel_preview_char_label_interaction()
        self._preview_char_selected_index = None
        self._preview_char_hover_index = None
        self._preview_char_hover_grip = None
        self._apply_preview_canvas_cursor("arrow")
        self._refresh_preview_editor_toolbar()
        self._update_preview_record_source_label(None)
        return

    pid = pid_map[idx]
    data = self.preview_metadata.get(pid, {})
    if not fast_select_render:
        self._update_preview_record_source_label(data)
    current_render_state = getattr(self, "_preview_render_state", None) or {}
    if (
        event is not None
        and pid == getattr(self, "_preview_active_pid", None)
        and str(current_render_state.get("plate_id", "") or "") == str(pid)
        and getattr(self, "_preview_char_add_state", None) is None
        and getattr(self, "_preview_char_drag_state", None) is None
        and getattr(self, "_preview_layout_separator_drag_state", None) is None
    ):
        self._refresh_preview_editor_toolbar()
        return
    if pid != self._preview_active_pid:
        self._cancel_preview_char_label_interaction()
        self._preview_active_pid = pid
        self._reset_preview_view_state()
        self._preview_active_pid = pid
        self._preview_char_selected_index = None
        self._preview_char_hover_grip = None
        self._unbind_preview_char_drag_session()
        self._preview_char_drag_state = None
        self._preview_char_add_state = None
    mode_key = self._get_preview_box_mode_key()
    if fast_select_render and mode_key in {"AUTO", "FINAL"}:
        final_records = self._sort_character_records_by_x(data.get("characters", []))
        if final_records:
            yolo_variants = {"FINAL": final_records, "YOLO_FILTERED": [], "YOLO_NMS": [], "YOLO_RAW": []}
            box_chars, box_source = final_records, "FINAL"
            yolo_raw_count = yolo_nms_count = yolo_filtered_count = 0
        else:
            yolo_variants = self._get_preview_box_variants(data)
            box_chars, box_source = self._get_preview_box_records(data)
            yolo_raw_count = len(yolo_variants.get("YOLO_RAW", []))
            yolo_nms_count = len(yolo_variants.get("YOLO_NMS", []))
            yolo_filtered_count = len(yolo_variants.get("YOLO_FILTERED", []))
    else:
        yolo_variants = self._get_preview_box_variants(data)
        method_name = self._get_detection_method_key()
        if mode_key == "AUTO":
            final_records = list(yolo_variants.get("FINAL", []) or [])
            if final_records:
                box_chars, box_source = final_records, "FINAL"
            elif method_name == "YOLO":
                box_chars, box_source = [], "FINAL"
                for candidate_key in ("YOLO_FILTERED", "YOLO_NMS", "YOLO_RAW"):
                    candidate_records = yolo_variants.get(candidate_key, [])
                    if candidate_records:
                        box_chars, box_source = candidate_records, candidate_key
                        break
            else:
                box_chars, box_source = final_records, "FINAL"
                for candidate_key in ("YOLO_FILTERED", "YOLO_NMS", "YOLO_RAW"):
                    candidate_records = yolo_variants.get(candidate_key, [])
                    if candidate_records:
                        box_chars, box_source = candidate_records, candidate_key
                        break
        else:
            box_chars, box_source = yolo_variants.get(mode_key, []), mode_key
        yolo_raw_count = len(yolo_variants.get("YOLO_RAW", []))
        yolo_nms_count = len(yolo_variants.get("YOLO_NMS", []))
        yolo_filtered_count = len(yolo_variants.get("YOLO_FILTERED", []))
    canonical_chars = self._get_preview_active_character_records(create=False)
    selected_char_idx, _selected_char_rec = self._get_preview_selected_char_record()
    if selected_char_idx is not None and not (0 <= int(selected_char_idx) < len(canonical_chars)):
        self._preview_char_selected_index = None
        selected_char_idx = None
    if not fast_select_render:
        self._update_preview_box_info_label(
            plate_id=pid,
            mode_key=box_source,
            shown_count=len(box_chars),
            yolo_raw_count=yolo_raw_count,
            yolo_nms_count=yolo_nms_count,
            yolo_filtered_count=yolo_filtered_count,
        )
    _mark_render_profile("data")

    img_path = Path(self.preview_dir_var.get().strip()) / "images" / f"{pid}.jpg"

    self._apply_preview_canvas_cursor("arrow")
    if not fast_select_render:
        self._refresh_preview_editor_toolbar()
    if not img_path.exists():
        self._set_preview_box_info(f"{pid} | {self._get_preview_box_mode_label(box_source)} | brak obrazu podglądu", "error")
        self._apply_preview_canvas_cursor("arrow")
        return

    try:
        pil_img, orig_w, orig_h, source_image_key = _get_cached_preview_source_image(self, img_path)
        data["plate_image_width"] = float(orig_w)
        data["plate_image_height"] = float(orig_h)
        try:
            two_row_layout_active = bool(self._should_preview_use_two_row_layers(data))
        except Exception:
            try:
                two_row_layout_active = bool(self._is_preview_two_row_layout_active(data))
            except Exception:
                two_row_layout_active = False
        if two_row_layout_active:
            try:
                self._ensure_preview_layout_separator(
                    data,
                    data.get("characters", []),
                    image_w=float(orig_w),
                    image_h=float(orig_h),
                )
            except Exception:
                pass

        c_w = max(50, self.preview_canvas.winfo_width())
        c_h = max(50, self.preview_canvas.winfo_height())

        if fast_select_render:
            # First frame: reserve a stable top area, but defer expensive status/HUD work.
            info_bar_height = 74 if bool(getattr(self, "_preview_fullscreen_active", False)) else 80
        else:
            status_meta = self._get_preview_status_presentation(
                data=data,
                chars=(data or {}).get("characters", []),
                plate_id=str((data or {}).get("plate_id", "") or getattr(self, "_preview_active_pid", "") or ""),
            )
            status_text = str(status_meta.get("canvas_text", "Nieocenione") or "Nieocenione")
            source_counts = self._count_character_sources(box_chars, data=data)
            source_line = self._format_preview_source_counts_line(source_counts)
            info_layout = self._plan_preview_canvas_info_overlay_layout(
                c_w,
                source_line,
                status_text,
                data=data,
                box_chars=box_chars,
            )
            info_bar_height = int(max(32.0, float(info_layout.get("bar_height", 32.0))))
        self._preview_overlay_top_bar_height = float(info_bar_height)
        add_state = getattr(self, "_preview_char_add_state", None)
        mute_existing_boxes_during_add = isinstance(add_state, dict)
        preview_canvas_bg = str(getattr(self.app, "palette", {}).get("panel", "#101010"))
        margin_x = max(84, int(c_w * 0.18))
        badge_plan_specs = []
        estimated_image_width = max(80, min(c_w - margin_x, int(c_w * 0.78)))
        if not fast_select_render:
            for box_idx, c in enumerate(box_chars):
                if mute_existing_boxes_during_add:
                    continue
                if not isinstance(c, dict):
                    continue

                bbox = c.get("bbox", [0, 0, 0, 0])
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = bbox[:4]
                center_ratio = ((float(x1) + float(x2)) * 0.5) / max(1.0, float(orig_w))
                source_tag = self._get_character_source_tag(c, data=data, fallback_index=box_idx)
                char_box_source, char_sign_source = _preview_character_source_parts(self, c, data=data, fallback_index=box_idx)
                source_style = self._get_preview_source_visual_style(
                    _preview_visual_tag_for_box_source(self, char_box_source, char_sign_source)
                )
                uses_yolo_box_backend = self._character_record_uses_yolo_box_backend(c)
                is_yolo_box = (
                    uses_yolo_box_backend
                    or char_box_source == "yolo_box"
                    or char_sign_source == "yolo_symbol"
                    or source_tag in ("yolo", "yolo_box", "yolo_symbol", "yolo_box_ocr", "yolo_rescue")
                    or str(c.get("method", "")).strip().lower() in {"yolo", "yolo_box", "yolo_symbol"}
                )
                try:
                    badge_confidence = float(c.get("confidence", 0.0)) if is_yolo_box else None
                except Exception:
                    badge_confidence = None
                box_backend_confidence = self._character_record_yolo_box_backend_confidence(c)
                badge_layers = self._get_preview_source_badge_layers(
                    source_tag,
                    confidence=badge_confidence,
                    box_backend_confidence=box_backend_confidence,
                    include_confidence=bool(is_yolo_box),
                    has_symbol=self._preview_record_has_symbol(c),
                    uses_yolo_box_backend=uses_yolo_box_backend,
                    box_source=char_box_source,
                    sign_source=char_sign_source,
                )
                badge_text = str((badge_layers[0] if badge_layers else {}).get("text", source_style["label"]) or source_style["label"])

                badge_plan_specs.append({
                    "center_ref": center_ratio,
                    "text": badge_text,
                    "badge_layers": badge_layers,
                })

            estimated_badge_rows, estimated_badge_height = self._estimate_preview_badge_layout(
                self.preview_canvas,
                badge_plan_specs,
                estimated_image_width,
            )
        else:
            estimated_badge_rows, estimated_badge_height = 1, 24
        badge_metrics = self._get_preview_badge_layout_metrics()
        box_label_clearance = max(
            108,
            10 + (estimated_badge_rows * estimated_badge_height) + (max(0, estimated_badge_rows - 1) * badge_metrics["row_gap"])
        )
        margin_y_top = max(info_bar_height + box_label_clearance, int(c_h * 0.18))
        margin_y_bottom = max(128, int(c_h * 0.26))
        if two_row_layout_active:
            margin_y_bottom = max(margin_y_bottom, box_label_clearance + 48)

        for _ in range(2):
            usable_w = max(80, min(c_w - margin_x, int(c_w * 0.78)))
            usable_h = max(60, c_h - (margin_y_top + margin_y_bottom))

            scale_w = usable_w / float(orig_w)
            scale_h = usable_h / float(orig_h)
            default_fit_ratio = 0.90
            SCALE = max(0.001, min(min(scale_w, scale_h), 5.0) * default_fit_ratio)

            new_w, new_h = int(orig_w * SCALE), int(orig_h * SCALE)
            if fast_select_render:
                reevaluated_rows, reevaluated_badge_height = estimated_badge_rows, estimated_badge_height
            else:
                reevaluated_rows, reevaluated_badge_height = self._estimate_preview_badge_layout(
                    self.preview_canvas,
                    badge_plan_specs,
                    max(80, new_w),
                )
            required_clearance = max(
                108,
                10 + (reevaluated_rows * reevaluated_badge_height) + (max(0, reevaluated_rows - 1) * badge_metrics["row_gap"])
            )
            if required_clearance <= box_label_clearance:
                break
            box_label_clearance = required_clearance
            margin_y_top = max(info_bar_height + box_label_clearance, int(c_h * 0.18))
            if two_row_layout_active:
                margin_y_bottom = max(margin_y_bottom, box_label_clearance + 48)

        fit_scale = float(SCALE)
        fit_new_w = int(orig_w * fit_scale)
        fit_new_h = int(orig_h * fit_scale)
        fit_x_off = (c_w - fit_new_w) / 2.0
        if bool(getattr(self, "_preview_fullscreen_active", False)):
            # Keep the plate visually clear of the fullscreen drawer without changing
            # the image/box transform relationship.
            drawer_bias = 72.0
            try:
                dock = getattr(self, "preview_overlay_dock", None)
                if dock is not None and str(dock.winfo_manager()):
                    dock_width = float(dock.winfo_width() or dock.winfo_reqwidth() or 0.0)
                    if dock_width > 0.0:
                        drawer_bias = min(132.0, max(56.0, dock_width * 0.42))
            except Exception:
                drawer_bias = 72.0
            available_left_margin = max(0.0, float(fit_x_off) - 16.0)
            fit_x_off -= min(float(drawer_bias), available_left_margin)
        fit_y_off = (c_h - fit_new_h - margin_y_bottom + margin_y_top) / 2.0
        fit_y_off = max(float(info_bar_height + box_label_clearance), fit_y_off)

        safe_zoom_max = float(getattr(self, "_preview_zoom_max", 2.4) or 2.4)
        try:
            safe_w = max(80.0, float(c_w) - 40.0)
            safe_h = max(80.0, float(c_h) - float(info_bar_height + box_label_clearance) - 92.0)
            safe_zoom_max = max(
                1.0,
                min(
                    safe_zoom_max,
                    2.4,
                    safe_w / max(1.0, float(fit_new_w)),
                    safe_h / max(1.0, float(fit_new_h)),
                ),
            )
        except Exception:
            safe_zoom_max = min(safe_zoom_max, 2.4)
        self._preview_safe_zoom_max = float(safe_zoom_max)
        self._preview_zoom_level = max(self._preview_zoom_min, min(float(safe_zoom_max), float(self._preview_zoom_level)))
        render_scale = fit_scale * float(self._preview_zoom_level)
        new_w = max(1, int(orig_w * render_scale))
        new_h = max(1, int(orig_h * render_scale))
        next_photo = _get_cached_preview_photo(self, source_image_key, pil_img, new_w, new_h)
        _mark_render_profile("image")
        if fast_select_render:
            for tag in (
                "preview_plate_image",
                "preview_char",
                "preview_fast_detail",
                "preview_badge",
                "preview_plate_status_frame",
                "preview_layout_separator",
                "preview_char_drag_preview",
                "preview_char_add_preview",
                "preview_canvas_caption",
            ):
                try:
                    self.preview_canvas.delete(tag)
                except Exception:
                    pass
        else:
            self.preview_canvas.delete("all")
        self._preview_render_state = {}
        self._preview_badge_runtime = {}
        self._preview_char_runtime = {}
        self._current_photo = next_photo
        base_x_off = fit_x_off - ((new_w - fit_new_w) / 2.0)
        base_y_off = fit_y_off - ((new_h - fit_new_h) / 2.0)
        top_reserved = float(info_bar_height + box_label_clearance)
        bottom_reserved = float(self._get_preview_image_bottom_reserved(data))
        x_off = base_x_off + float(self._preview_pan_x)
        y_off = base_y_off + float(self._preview_pan_y)
        x_off, y_off = self._clamp_preview_image_position(
            x_off,
            y_off,
            image_w=new_w,
            image_h=new_h,
            canvas_w=c_w,
            canvas_h=c_h,
            top_reserved=top_reserved,
            bottom_reserved=bottom_reserved,
            preferred_x=base_x_off,
            preferred_y=base_y_off,
        )
        self._preview_pan_x = float(x_off - base_x_off)
        self._preview_pan_y = float(y_off - base_y_off)
        self._preview_render_state = {
            "plate_id": pid,
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "fit_scale": float(fit_scale),
            "scale": float(render_scale),
            "fit_new_w": float(fit_new_w),
            "fit_new_h": float(fit_new_h),
            "fit_x_off": float(fit_x_off),
            "fit_y_off": float(fit_y_off),
            "top_reserved": float(top_reserved),
            "bottom_reserved": float(bottom_reserved),
            "safe_zoom_max": float(safe_zoom_max),
            "info_bar_height": float(info_bar_height),
            "image_left": float(x_off),
            "image_top": float(y_off),
            "image_right": float(x_off + new_w),
            "image_bottom": float(y_off + new_h),
        }

        self.preview_canvas.create_image(
            x_off,
            y_off,
            anchor=tk.NW,
            image=self._current_photo,
            tags=("preview_plate_image", "preview_movable_plate"),
        )
        try:
            self.preview_canvas.tag_lower("preview_plate_image")
        except Exception:
            pass

        self._preview_badge_runtime = {}
        self._preview_layout_separator_runtime = {}
        self._preview_char_runtime = {}
        self._preview_char_record_render_tags = {}
        image_bottom_y = y_off + new_h
        if not fast_select_render:
            self._draw_preview_plate_status_frame(data)
        if not fast_select_render:
            self._draw_preview_canvas_info_overlay(
                self.preview_canvas,
                c_w,
                data=data,
                box_chars=box_chars,
                has_boxes=bool(box_chars),
            )
        selection_color = getattr(self.app, "palette", {}).get("accent", "#ffd166")
        selected_box_color = _get_preview_selected_character_box_color(self)
        label_focus_color = getattr(self.app, "palette", {}).get("warning", "#f59e0b")
        label_mode_active = bool(getattr(self, "_preview_char_label_mode", False))
        active_label_idx = getattr(self, "_preview_char_label_active_index", None)
        hover_label_idx = getattr(self, "_preview_char_hover_label_index", None)
        selected_char_record = None
        active_label_record = None
        hover_label_record = None

        if box_source == "FINAL" and isinstance(canonical_chars, list):
            try:
                if selected_char_idx is not None and 0 <= int(selected_char_idx) < len(canonical_chars):
                    selected_char_record = canonical_chars[int(selected_char_idx)]
            except Exception:
                selected_char_record = None
            try:
                if active_label_idx is not None and 0 <= int(active_label_idx) < len(canonical_chars):
                    active_label_record = canonical_chars[int(active_label_idx)]
            except Exception:
                active_label_record = None
            try:
                if hover_label_idx is not None and 0 <= int(hover_label_idx) < len(canonical_chars):
                    hover_label_record = canonical_chars[int(hover_label_idx)]
            except Exception:
                hover_label_record = None

        canonical_index_by_id = {}
        if box_source == "FINAL" and isinstance(canonical_chars, list):
            canonical_index_by_id = {
                id(rec): int(idx)
                for idx, rec in enumerate(canonical_chars)
                if isinstance(rec, dict)
            }

        # rysowanie bboxów + znaków
        for box_idx, c in enumerate(box_chars):
            if not isinstance(c, dict):
                continue

            bbox = c.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            runtime_idx = int(canonical_index_by_id.get(id(c), int(box_idx))) if box_source == "FINAL" else int(box_idx)
            char_canvas_tag = self._get_preview_character_canvas_tag(box_source, runtime_idx)
            char_record_tag = self._get_preview_character_record_canvas_tag(c)
            char_canvas_tags = ("preview_char", char_canvas_tag, char_record_tag)

            x1, y1, x2, y2 = bbox[:4]
            cx1, cy1 = (float(x1) * render_scale) + x_off, (float(y1) * render_scale) + y_off
            cx2, cy2 = (float(x2) * render_scale) + x_off, (float(y2) * render_scale) + y_off

            center_x = cx1 + (cx2 - cx1) / 2
            row_no = self._get_preview_row_for_bbox([float(x1), float(y1), float(x2), float(y2)], data) or 1
            badge_side = "bottom" if two_row_layout_active and int(row_no) == 2 else "top"
            source_tag = self._get_character_source_tag(c, data=data, fallback_index=runtime_idx)
            char_box_source, char_sign_source = _preview_character_source_parts(self, c, data=data, fallback_index=runtime_idx)
            source_style = self._get_preview_source_visual_style(
                _preview_visual_tag_for_box_source(self, char_box_source, char_sign_source)
            )
            uses_yolo_box_backend = self._character_record_uses_yolo_box_backend(c)
            is_yolo_box = (
                uses_yolo_box_backend
                or char_box_source == "yolo_box"
                or char_sign_source == "yolo_symbol"
                or source_tag in ("yolo", "yolo_box", "yolo_symbol", "yolo_box_ocr", "yolo_rescue")
                or str(c.get("method", "")).strip().lower() in {"yolo", "yolo_box", "yolo_symbol"}
            )
            box_color = source_style["outline"]
            guide_color = source_style["guide"]
            char_fill = box_color
            badge_fill = str(source_style.get("badge_fill", box_color))
            badge_outline = str(source_style.get("badge_outline", badge_fill))
            is_selected_box = bool(box_source == "FINAL" and selected_char_record is not None and c is selected_char_record)
            draw_badge = True
            draw_box_details = True
            if fast_select_render:
                draw_badge = False
                draw_box_details = False
            box_kwargs = {
                "outline": (selected_box_color if is_selected_box else box_color),
                "width": (3 if is_selected_box else 2),
            }
            if mute_existing_boxes_during_add:
                box_kwargs["outline"] = blend_hex_colors(box_color, preview_canvas_bg, 0.48)
                box_kwargs["fill"] = blend_hex_colors(box_color, preview_canvas_bg, 0.84)
                box_kwargs["stipple"] = "gray25"
                box_kwargs["width"] = 1
                draw_badge = False
                draw_box_details = False
                is_selected_box = False

            selection_id = None
            handle_ids = []
            move_handle_id = None
            box_id = self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, tags=char_canvas_tags, **box_kwargs)
            self._preview_char_runtime[f"{box_source}:{runtime_idx}"] = {
                "tag": char_canvas_tag,
                "box_id": box_id,
                "record_id": id(c),
            }
            self._preview_char_record_render_tags[id(c)] = char_record_tag

            if is_selected_box:
                selection_id = self.preview_canvas.create_rectangle(
                    cx1 - 2, cy1 - 2, cx2 + 2, cy2 + 2,
                    outline=selection_color,
                    width=1,
                    dash=(4, 2),
                    tags=char_canvas_tags,
                )

            badge_text = source_style["label"]
            try:
                badge_confidence = float(c.get("confidence", 0.0)) if is_yolo_box else None
            except Exception:
                badge_confidence = None
            box_backend_confidence = self._character_record_yolo_box_backend_confidence(c)
            badge_layers = self._get_preview_source_badge_layers(
                source_tag,
                confidence=badge_confidence,
                box_backend_confidence=box_backend_confidence,
                include_confidence=bool(is_yolo_box),
                has_symbol=self._preview_record_has_symbol(c),
                uses_yolo_box_backend=uses_yolo_box_backend,
                box_source=char_box_source,
                sign_source=char_sign_source,
            )
            badge_text = str((badge_layers[0] if badge_layers else {}).get("text", source_style["label"]) or source_style["label"])
            char_text = str(c.get("character", ""))
            if draw_box_details:
                reading_label = self._get_preview_character_reading_position_label(c, data=data)
                _draw_preview_compact_character_signature(
                    self,
                    self.preview_canvas,
                    center_x=center_x,
                    box_anchor_y=cy2 if badge_side == "bottom" else cy1,
                    image_top=y_off,
                    image_bottom=image_bottom_y,
                    canvas_height=float(c_h),
                    top_limit=float(info_bar_height) + 18.0,
                    side=badge_side,
                    char_text=char_text,
                    char_color=char_fill,
                    guide_color=guide_color,
                    badge_layers=badge_layers,
                    reading_label=reading_label,
                    tags=char_canvas_tags,
                    stagger_index=_get_preview_signature_stagger_index(c, runtime_idx),
                )

            if is_selected_box and bool(getattr(self, "_preview_char_edit_mode", False)):
                handle_ids, move_handle_id = _draw_preview_character_edit_grips(
                    self,
                    self.preview_canvas,
                    tags=char_canvas_tags,
                    cx1=cx1,
                    cy1=cy1,
                    cx2=cx2,
                    cy2=cy2,
                    selection_color=selection_color,
                )

            self._preview_char_runtime[f"{box_source}:{runtime_idx}"].update(
                {
                    "selection_id": selection_id,
                    "handle_ids": handle_ids,
                    "move_handle_id": move_handle_id,
                }
            )

            # Mini-pola etykiety mają być widoczne tylko wtedy, gdy użytkownik
            # rzeczywiście pracuje w trybie wpisywania albo ma aktywne pole znaku.
            # Sam brak znaku w boxie nie powinien już rysować "okienka wpisywania".
            show_label_box = bool(
                draw_box_details
                and box_source == "FINAL"
                and (
                    label_mode_active
                    or (active_label_record is not None and c is active_label_record)
                    or (
                        hover_label_record is not None
                        and c is hover_label_record
                        and (label_mode_active or active_label_record is not None)
                    )
                )
            )
            if show_label_box:
                label_rect = self._get_preview_char_label_canvas_rect(c)
                if label_rect is not None:
                    lx1, ly1, lx2, ly2 = label_rect
                    label_active = bool(active_label_record is not None and c is active_label_record)
                    label_hovered = bool(hover_label_record is not None and c is hover_label_record)
                    label_highlight = bool(label_active or label_hovered)
                    label_accent = label_focus_color if (label_mode_active and label_highlight) else selection_color
                    if label_active:
                        label_fill = label_focus_color
                        label_outline = self._get_readable_text_color(label_fill, preferred="#111111")
                        label_text_fill = self._get_readable_text_color(label_fill, preferred="#111111")
                        label_width = 3
                    else:
                        label_fill = blend_hex_colors("#ffffff", label_accent, 0.22 if label_highlight else 0.08)
                        label_outline = label_accent if label_highlight else box_color
                        label_text_fill = "#111111"
                        label_width = 1
                    label_tags = _preview_char_label_canvas_tags(char_canvas_tags)
                    if label_active:
                        self.preview_canvas.create_rectangle(
                            lx1 - 2,
                            ly1 - 2,
                            lx2 + 2,
                            ly2 + 2,
                            outline=label_focus_color,
                            width=2,
                            tags=label_tags,
                        )
                    self.preview_canvas.create_rectangle(
                        lx1,
                        ly1,
                        lx2,
                        ly2,
                        fill=label_fill,
                        outline=label_outline,
                        width=label_width,
                        tags=label_tags,
                    )
                    valid_char = self._sanitize_preview_char_symbol(char_text)
                    if valid_char:
                        self.preview_canvas.create_text(
                            (lx1 + lx2) / 2.0,
                            (ly1 + ly2) / 2.0 + 0.5,
                            text=valid_char,
                            fill=label_text_fill,
                            font=("Segoe UI", 8, "bold"),
                            anchor=tk.CENTER,
                            tags=label_tags,
                        )
                    elif label_active:
                        self.preview_canvas.create_line(
                            lx1 + 7,
                            ly1 + 4,
                            lx1 + 7,
                            ly2 - 4,
                            fill=label_text_fill,
                            width=2,
                            tags=label_tags,
                        )

        self._draw_preview_layout_separator(data)

        if isinstance(add_state, dict):
            preview_bbox = add_state.get("bbox")
            if isinstance(preview_bbox, (list, tuple)) and len(preview_bbox) >= 4:
                ax1, ay1, ax2, ay2 = (float(v) for v in preview_bbox[:4])
                if ax2 < ax1:
                    ax1, ax2 = ax2, ax1
                if ay2 < ay1:
                    ay1, ay2 = ay2, ay1
                add_cx1, add_cy1 = self._preview_image_to_canvas_point(ax1, ay1)
                add_cx2, add_cy2 = self._preview_image_to_canvas_point(ax2, ay2)
                _draw_preview_new_character_box(self.preview_canvas, add_cx1, add_cy1, add_cx2, add_cy2)

        _mark_render_profile("draw")
        try:
            self.preview_canvas.delete("preview_canvas_caption")
        except Exception:
            pass
        self.preview_canvas.create_text(
            10,
            max(14, c_h - 10),
            text="Podgląd tablicy",
            fill=getattr(self.app, "palette", {}).get("muted", "#b0b0b0"),
            font=("Segoe UI", 8, "bold"),
            anchor=tk.SW,
            tags=("preview_canvas_caption",),
        )
        if not fast_select_render:
            self._ensure_preview_mode_overlay_position()
        try:
            if not fast_select_render:
                self._place_preview_overlay_dock()
                self._place_preview_hint_overlay(refresh=False)
                self._refresh_preview_typing_overlay_visibility()
        except Exception:
            pass
        if not fast_select_render:
            self._apply_preview_canvas_cursor()
            self._refresh_preview_editor_toolbar()
        if fast_select_render:
            self._preview_fast_select_render = False
            try:
                pending_detail_after = getattr(self, "_preview_detail_render_after_id", None)
                if pending_detail_after:
                    self.frame.after_cancel(pending_detail_after)
            except Exception:
                pass
            active_pid_for_details = str(pid)
            try:
                detail_generation = int(getattr(self, "_preview_select_generation", 0) or 0)
            except Exception:
                detail_generation = 0

            def _render_preview_hud_once():
                self._preview_detail_render_after_id = None
                if str(getattr(self, "_preview_active_pid", "") or "") != active_pid_for_details:
                    return
                try:
                    if int(getattr(self, "_preview_select_generation", 0) or 0) != int(detail_generation):
                        return
                except Exception:
                    return
                if getattr(self, "_preview_list_select_after_id", None):
                    return
                if getattr(self, "_preview_char_drag_state", None) is not None:
                    return
                if getattr(self, "_preview_char_add_state", None) is not None:
                    return
                try:
                    self._draw_preview_fast_render_details(active_pid_for_details)
                except Exception:
                    pass

            try:
                keyboard_navigation = bool(getattr(self, "_preview_keyboard_crop_navigation_active", False))
                detail_delay = 320 if keyboard_navigation else 180
                self._preview_detail_render_after_id = self.frame.after(detail_delay, _render_preview_hud_once)
                if keyboard_navigation:
                    self._preview_keyboard_crop_navigation_active = False
            except Exception:
                self._preview_detail_render_after_id = None
                self._preview_keyboard_crop_navigation_active = False
        _schedule_preview_neighbor_prefetch(self, pid, delay_ms=180 if fast_select_render else 650)

        total_ms = (time.perf_counter() - render_profile_start) * 1000.0
        if total_ms >= 80.0:
            logger.info(
                "[PZ2 preview render] pid=%s fast=%s boxes=%s total=%.1fms data=%.1fms image=%.1fms draw=%.1fms",
                pid,
                fast_select_render,
                len(box_chars),
                total_ms,
                float(render_profile_marks.get("data", 0.0)),
                float(render_profile_marks.get("image", 0.0)),
                float(render_profile_marks.get("draw", 0.0)),
            )

    except Exception as e:
        logger.error(f"Błąd wyświetlania podglądu tablicy: {e}")


def draw_preview_fast_render_details(host, plate_id: str | None = None) -> bool:
    self = host
    detail_start = time.perf_counter()
    canvas = getattr(self, "preview_canvas", None)
    state = getattr(self, "_preview_render_state", None) or {}
    if canvas is None or not state:
        return False

    expected_pid = str(plate_id or "").strip()
    current_pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
    if expected_pid and current_pid != expected_pid:
        return False
    if expected_pid and str(state.get("plate_id", "") or "") != expected_pid:
        return False

    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return False
    chars = self._get_preview_active_character_records(create=False)
    if not isinstance(chars, list):
        chars = []

    for stale_tag in ("preview_fast_detail", "preview_badge"):
        try:
            canvas.delete(stale_tag)
        except Exception:
            pass
    self._preview_badge_runtime = {}
    self._preview_badge_drag_state = None

    try:
        render_scale = max(0.001, float(state.get("scale", 1.0) or 1.0))
        x_off = float(state.get("image_left", 0.0) or 0.0)
        y_off = float(state.get("image_top", 0.0) or 0.0)
        image_bottom_y = float(state.get("image_bottom", 0.0) or 0.0)
        canvas_height = max(50, int(canvas.winfo_height() or 50))
        info_bar_height = float(state.get("info_bar_height", 0.0) or 0.0)
    except Exception:
        return False

    try:
        detail_generation = int(getattr(self, "_preview_select_generation", 0) or 0)
    except Exception:
        detail_generation = 0

    def _schedule_fast_hud_refresh() -> None:
        try:
            pending_after = getattr(self, "_preview_fast_hud_refresh_after_id", None)
            if pending_after:
                self.frame.after_cancel(pending_after)
        except Exception:
            pass

        def _refresh_hud_once():
            self._preview_fast_hud_refresh_after_id = None
            if str(getattr(self, "_preview_active_pid", "") or "").strip() != current_pid:
                return
            try:
                if int(getattr(self, "_preview_select_generation", 0) or 0) != int(detail_generation):
                    return
            except Exception:
                return
            if getattr(self, "_preview_list_select_after_id", None):
                return
            try:
                variants = self._get_preview_box_variants(data)
                self._update_preview_record_source_label(data)
                self._update_preview_box_info_label(
                    plate_id=current_pid,
                    mode_key="FINAL",
                    shown_count=len(chars),
                    yolo_raw_count=len(variants.get("YOLO_RAW", []) or []),
                    yolo_nms_count=len(variants.get("YOLO_NMS", []) or []),
                    yolo_filtered_count=len(variants.get("YOLO_FILTERED", []) or []),
                )
            except Exception:
                pass
            try:
                canvas.delete("preview_overlay")
                canvas.delete("preview_overlay_action")
                self._draw_preview_plate_status_frame(data)
                self._draw_preview_canvas_info_overlay(
                    canvas,
                    max(50, int(canvas.winfo_width() or 50)),
                    data=data,
                    box_chars=chars,
                    has_boxes=bool(chars),
                )
            except Exception:
                pass
            try:
                if False:
                    canvas.create_text(
                    10,
                    max(14, canvas_height - 10),
                    text="PodglÄ…d tablicy",
                    fill=getattr(self.app, "palette", {}).get("muted", "#b0b0b0"),
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.SW,
                    tags=("preview_fast_detail",),
                )
            except Exception:
                pass
            try:
                self._ensure_preview_mode_overlay_position()
                self._place_preview_overlay_dock()
                self._place_preview_hint_overlay(refresh=False)
                self._refresh_preview_typing_overlay_visibility()
                self._apply_preview_canvas_cursor()
                self._refresh_preview_editor_toolbar()
            except Exception:
                pass

        try:
            self._preview_fast_hud_refresh_after_id = self.frame.after(120, _refresh_hud_once)
        except Exception:
            self._preview_fast_hud_refresh_after_id = None
            _refresh_hud_once()

    if not chars:
        _schedule_fast_hud_refresh()
        return True

    try:
        two_row_layout_active = bool(self._should_preview_use_two_row_layers(data))
    except Exception:
        try:
            two_row_layout_active = bool(self._is_preview_two_row_layout_active(data))
        except Exception:
            two_row_layout_active = False
    # The fast pass runs after keyboard navigation. Keep it cheap: coordinates are
    # no longer shown on badges, so reading-order recomputation can wait for a
    # normal/full render instead of blocking rapid Q/E navigation.

    selection_color = getattr(self.app, "palette", {}).get("accent", "#ffd166")
    label_focus_color = getattr(self.app, "palette", {}).get("warning", "#f59e0b")
    label_mode_active = bool(getattr(self, "_preview_char_label_mode", False))
    active_label_idx = getattr(self, "_preview_char_label_active_index", None)
    hover_label_idx = getattr(self, "_preview_char_hover_label_index", None)
    selected_idx, selected_record = self._get_preview_selected_char_record()

    for box_idx, c in enumerate(chars):
        if not isinstance(c, dict):
            continue
        bbox = self._char_record_bbox(c)
        if not bbox:
            continue
        try:
            x1, y1, x2, y2 = bbox[:4]
            cx1, cy1 = (float(x1) * render_scale) + x_off, (float(y1) * render_scale) + y_off
            cx2, cy2 = (float(x2) * render_scale) + x_off, (float(y2) * render_scale) + y_off
        except Exception:
            continue

        center_x = cx1 + (cx2 - cx1) / 2.0
        row_no = self._get_preview_row_for_bbox([float(x1), float(y1), float(x2), float(y2)], data) or 1
        badge_side = "bottom" if two_row_layout_active and int(row_no) == 2 else "top"
        source_tag = self._get_character_source_tag(c, data=data, fallback_index=box_idx)
        char_box_source, char_sign_source = _preview_character_source_parts(self, c, data=data, fallback_index=box_idx)
        source_style = self._get_preview_source_visual_style(
            _preview_visual_tag_for_box_source(self, char_box_source, char_sign_source)
        )
        # Fast detail signatures are part of the same visual object as the box.
        # Keep the record/index tags so drag/redraw cleanup does not leave stale
        # badges and leader lines behind after a manual correction.
        char_canvas_tag = self._get_preview_character_canvas_tag("FINAL", int(box_idx))
        char_record_tag = self._get_preview_character_record_canvas_tag(c)
        tags = ("preview_fast_detail", char_canvas_tag, char_record_tag)
        _draw_preview_compact_character_signature(
            self,
            canvas,
            center_x=center_x,
            box_anchor_y=cy2 if badge_side == "bottom" else cy1,
            image_top=y_off,
            image_bottom=image_bottom_y,
            canvas_height=float(canvas_height),
            top_limit=info_bar_height + 18.0,
            side=badge_side,
            char_text=str(c.get("character", "")),
            char_color=source_style["outline"],
            guide_color=source_style["guide"],
            badge_layers=self._get_preview_source_badge_layers(
                source_tag,
                confidence=None,
                box_backend_confidence=self._character_record_yolo_box_backend_confidence(c),
                include_confidence=False,
                has_symbol=self._preview_record_has_symbol(c),
                uses_yolo_box_backend=self._character_record_uses_yolo_box_backend(c),
                box_source=char_box_source,
                sign_source=char_sign_source,
            ),
            reading_label=None,
            tags=tags,
            stagger_index=_get_preview_signature_stagger_index(c, box_idx),
        )

        is_selected_box = bool(selected_record is c or (selected_idx is not None and int(selected_idx) == int(box_idx)))
        if is_selected_box and bool(getattr(self, "_preview_char_edit_mode", False)):
            _draw_preview_character_edit_grips(
                self,
                canvas,
                tags=tags,
                cx1=cx1,
                cy1=cy1,
                cx2=cx2,
                cy2=cy2,
                selection_color=selection_color,
            )

        show_label_box = bool(
            label_mode_active
            or (active_label_idx is not None and int(active_label_idx) == int(box_idx))
            or (
                hover_label_idx is not None
                and int(hover_label_idx) == int(box_idx)
                and (label_mode_active or active_label_idx is not None)
            )
        )
        if show_label_box:
            label_rect = self._get_preview_char_label_canvas_rect(c)
            if label_rect is not None:
                lx1, ly1, lx2, ly2 = label_rect
                label_active = bool(active_label_idx is not None and int(active_label_idx) == int(box_idx))
                label_hovered = bool(hover_label_idx is not None and int(hover_label_idx) == int(box_idx))
                label_accent = label_focus_color if (label_mode_active and (label_active or label_hovered)) else selection_color
                if label_active:
                    label_fill = label_focus_color
                    label_outline = self._get_readable_text_color(label_fill, preferred="#111111")
                    label_text_fill = self._get_readable_text_color(label_fill, preferred="#111111")
                    label_width = 3
                else:
                    palette = getattr(self.app, "palette", {})
                    label_fill = palette.get("panel", "#202020")
                    label_outline = label_accent
                    label_text_fill = label_accent
                    label_width = 2 if label_hovered else 1
                label_tags = _preview_char_label_canvas_tags(tags)
                canvas.create_rectangle(
                    lx1,
                    ly1,
                    lx2,
                    ly2,
                    fill=label_fill,
                    outline=label_outline,
                    width=label_width,
                    tags=label_tags,
                )
                canvas.create_text(
                    (lx1 + lx2) / 2.0,
                    (ly1 + ly2) / 2.0,
                    text=str(c.get("character", "") or "?"),
                    fill=label_text_fill,
                    font=("Segoe UI", 9, "bold"),
                    anchor=tk.CENTER,
                    tags=label_tags,
                )

    try:
        canvas.tag_raise("preview_fast_detail")
        canvas.tag_raise("preview_char_add_preview")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    detail_ms = (time.perf_counter() - detail_start) * 1000.0
    if detail_ms >= 100.0:
        logger.info(
            "[PZ2 preview badge] pid=%s chars=%s total=%.1fms",
            current_pid,
            len(chars),
            detail_ms,
        )
    _schedule_fast_hud_refresh()
    return True


def redraw_preview_add_box_overlay_only(host) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return False

    try:
        canvas.delete("preview_char_add_preview")
    except Exception:
        pass

    add_state = getattr(self, "_preview_char_add_state", None)
    if not isinstance(add_state, dict):
        return True

    preview_bbox = add_state.get("bbox")
    if not isinstance(preview_bbox, (list, tuple)) or len(preview_bbox) < 4:
        return True

    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in preview_bbox[:4])
    except Exception:
        return False
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1

    add_cx1, add_cy1 = self._preview_image_to_canvas_point(ax1, ay1)
    add_cx2, add_cy2 = self._preview_image_to_canvas_point(ax2, ay2)
    try:
        _draw_preview_new_character_box(canvas, add_cx1, add_cy1, add_cx2, add_cy2)
        return True
    except Exception:
        return False


def update_preview_character_drag_visual(host, char_idx: int, rec: dict, bbox) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    state = getattr(self, "_preview_render_state", None) or {}
    drag_state = getattr(self, "_preview_char_drag_state", None)
    if canvas is None or not state or not isinstance(drag_state, dict) or not isinstance(rec, dict):
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False

    try:
        render_scale = max(0.001, float(state.get("scale", 1.0) or 1.0))
        x_off = float(state.get("image_left", 0.0) or 0.0)
        y_off = float(state.get("image_top", 0.0) or 0.0)
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except Exception:
        return False

    cx1 = (x1 * render_scale) + x_off
    cy1 = (y1 * render_scale) + y_off
    cx2 = (x2 * render_scale) + x_off
    cy2 = (y2 * render_scale) + y_off
    center_x = cx1 + ((cx2 - cx1) / 2.0)
    try:
        handle_radius = float(drag_state.get("handle_radius", 0.0) or 0.0)
    except Exception:
        handle_radius = 0.0
    if handle_radius <= 0.0:
        handle_radius = float(self._get_preview_char_handle_radius())
        drag_state["handle_radius"] = handle_radius

    visual_ids = drag_state.get("visual_ids")
    using_live_visual = bool(drag_state.get("using_live_visual"))
    if not isinstance(visual_ids, dict):
        record_tag = self._get_preview_character_record_canvas_tag(rec)
        runtime_key = f"FINAL:{int(char_idx)}"
        runtime_payload = (getattr(self, "_preview_char_runtime", None) or {}).get(runtime_key)
        if isinstance(runtime_payload, dict) and runtime_payload.get("box_id") is not None:
            visual_ids = {
                "box": runtime_payload.get("box_id"),
                "selection": runtime_payload.get("selection_id"),
                "handles": list(runtime_payload.get("handle_ids", []) or []),
                "move_handle": runtime_payload.get("move_handle_id"),
            }
            drag_state["visual_ids"] = visual_ids
            drag_state["visual_record_tag"] = record_tag
            drag_state["using_live_visual"] = True
            using_live_visual = True
        else:
            try:
                canvas.delete("preview_char_drag_preview")
            except Exception:
                pass
            try:
                selection_color = getattr(self.app, "palette", {}).get("accent", "#ffd166")
                selected_box_color = _get_preview_selected_character_box_color(self)
                box_id = canvas.create_rectangle(
                    cx1,
                    cy1,
                    cx2,
                    cy2,
                    outline=selected_box_color,
                    width=3,
                    tags=("preview_char_drag_preview",),
                )
                selection_id = canvas.create_rectangle(
                    cx1 - 2,
                    cy1 - 2,
                    cx2 + 2,
                    cy2 + 2,
                    outline=selection_color,
                    width=1,
                    dash=(4, 2),
                    tags=("preview_char_drag_preview",),
                )
                handle_ids, move_handle_id = _draw_preview_character_edit_grips(
                    self,
                    canvas,
                    tags=("preview_char_drag_preview",),
                    cx1=cx1,
                    cy1=cy1,
                    cx2=cx2,
                    cy2=cy2,
                    selection_color=selection_color,
                )
                visual_ids = {
                    "box": box_id,
                    "selection": selection_id,
                    "handles": handle_ids,
                    "move_handle": move_handle_id,
                }
                drag_state["visual_ids"] = visual_ids
                drag_state["visual_record_tag"] = record_tag
            except Exception:
                return False

    try:
        box_item = visual_ids.get("box")
        if box_item is not None:
            canvas.coords(box_item, cx1, cy1, cx2, cy2)
        selection_item = visual_ids.get("selection")
        if selection_item is not None:
            canvas.coords(selection_item, cx1 - 2, cy1 - 2, cx2 + 2, cy2 + 2)
        handle_ids = list(visual_ids.get("handles", []) or [])
        handle_points = ((cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2))
        for item_id, (handle_x, handle_y) in zip(handle_ids, handle_points):
            if item_id is None:
                continue
            canvas.coords(
                item_id,
                handle_x - handle_radius,
                handle_y - handle_radius,
                handle_x + handle_radius,
                handle_y + handle_radius,
            )
        move_handle_item = visual_ids.get("move_handle")
        if move_handle_item is not None:
            move_radius = float(self._get_preview_char_move_handle_radius())
            center_y = cy1 + ((cy2 - cy1) / 2.0)
            canvas.coords(
                move_handle_item,
                center_x - move_radius,
                center_y - move_radius,
                center_x + move_radius,
                center_y + move_radius,
            )
    except Exception:
        return False

    badge_key = f"FINAL:{int(char_idx)}"
    badge_runtime = getattr(self, "_preview_badge_runtime", {}).get(badge_key)
    if isinstance(badge_runtime, dict):
        line_id = badge_runtime.get("line_id")
        if line_id is not None:
            try:
                badge_runtime["box_id"] = visual_ids.get("box")
                badge_center_x = float(badge_runtime.get("base_center_x", center_x)) + float(badge_runtime.get("offset_dx", 0.0))
                badge_line_y = float(badge_runtime.get("base_line_y", badge_runtime.get("base_bottom_y", cy1))) + float(badge_runtime.get("offset_dy", 0.0))
                anchor_y = cy2 if str(badge_runtime.get("badge_side", "top") or "top").lower() == "bottom" else cy1
                badge_runtime["box_anchor_y"] = float(anchor_y)
                canvas.coords(line_id, badge_center_x, badge_line_y, center_x, anchor_y)
            except Exception:
                pass

    try:
        if using_live_visual:
            canvas.tag_raise(self._get_preview_character_record_canvas_tag(rec))
        else:
            canvas.tag_raise("preview_char_drag_preview")
        canvas.tag_raise("preview_char_add_preview")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    return True


def draw_preview_plate_status_frame(host, data: dict | None = None) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    state = getattr(self, "_preview_render_state", None) or {}
    if canvas is None or not state:
        return False

    try:
        canvas.delete("preview_plate_status_frame")
    except Exception:
        pass

    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    if not isinstance(source_data, dict):
        canvas.delete("preview_plate_layout_control")
        hide_layout_tip(self)
        return False
    status = str(source_data.get("status", "") or "").strip().lower()
    severity = str(source_data.get("severity", "") or "").strip().lower()
    if status not in {"perfect", "bad", "needs_fix"}:
        try:
            status = str(
                self._derive_preview_status_from_data(source_data, source_data.get("characters", []))
                or status
                or ""
            ).strip().lower()
        except Exception:
            pass
    if status == "perfect":
        frame_tone = "success"
    elif status in {"bad", "needs_fix"} or severity == "error":
        frame_tone = "error"
    else:
        frame_tone = "muted"

    try:
        left = float(state.get("image_left", 0.0) or 0.0)
        top = float(state.get("image_top", 0.0) or 0.0)
        right = float(state.get("image_right", left) or left)
        bottom = float(state.get("image_bottom", top) or top)
    except Exception:
        return False
    if right <= left or bottom <= top:
        return False

    palette = getattr(self.app, "palette", {})
    frame_color = (
        palette.get("success", "#2ecc71")
        if frame_tone == "success"
        else palette.get("error", palette.get("danger", "#ef4444"))
    )
    if frame_tone == "muted":
        frame_color = palette.get("border", "#808890")
    tags = ("preview_plate_status_frame",)
    try:
        handle_radius = float(self._get_preview_char_handle_radius())
    except Exception:
        handle_radius = 9.0
    frame_pad = max(20.0, min(30.0, handle_radius + 12.0))
    try:
        canvas.create_rectangle(
            left - frame_pad,
            top - frame_pad,
            right + frame_pad,
            bottom + frame_pad,
            outline=frame_color,
            width=3,
            tags=tags,
        )
        try:
            canvas.tag_raise("preview_plate_status_frame", "preview_plate_image")
        except Exception:
            pass
        try:
            canvas.tag_raise("preview_char")
        except Exception:
            pass
        try:
            canvas.tag_raise("preview_badge")
        except Exception:
            pass
        try:
            canvas.tag_raise("preview_char_drag_preview")
            canvas.tag_raise("preview_char_add_preview")
        except Exception:
            pass
        draw_plate_layout_control(self, source_data, right=right + frame_pad, top=top - frame_pad)
        return True
    except Exception:
        return False


def get_preview_layout_separator_canvas_geometry(host, separator: dict, data: dict | None = None) -> dict | None:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    state = getattr(self, "_preview_render_state", None) or {}
    if canvas is None or not state or not isinstance(separator, dict):
        return None

    try:
        image_left = float(state.get("image_left", 0.0) or 0.0)
        image_right = float(state.get("image_right", image_left) or image_left)
        canvas_width = float(canvas.winfo_width() or 0.0)
        x1 = float(separator.get("x1", 0.0) or 0.0)
        y1 = float(separator.get("y1", 0.0) or 0.0)
        x2 = float(separator.get("x2", 0.0) or 0.0)
        y2 = float(separator.get("y2", y1) or y1)
    except Exception:
        return None

    cx1, cy1 = self._preview_image_to_canvas_point(x1, y1)
    cx2, cy2 = self._preview_image_to_canvas_point(x2, y2)
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    interactive = bool(self._is_preview_layout_separator_interactive(source_data, ignore_active_char=True))
    handle_radius = 11.0 if interactive else 6.0
    try:
        char_handle_radius = float(self._get_preview_char_handle_radius())
    except Exception:
        char_handle_radius = 9.0
    frame_pad = max(20.0, min(30.0, char_handle_radius + 12.0))
    handle_gap = frame_pad + handle_radius + 12.0
    left_handle_x = image_left - handle_gap
    right_handle_x = image_right + handle_gap

    if canvas_width > 1.0:
        left_handle_x = max(handle_radius + 6.0, left_handle_x)
        right_handle_x = min(canvas_width - handle_radius - 6.0, right_handle_x)

    line_points = (
        float(left_handle_x),
        float(cy1),
        float(cx1),
        float(cy1),
        float(cx2),
        float(cy2),
        float(right_handle_x),
        float(cy2),
    )
    return {
        "left_point": (float(left_handle_x), float(cy1)),
        "right_point": (float(right_handle_x), float(cy2)),
        "left_handle_bbox": (
            float(left_handle_x) - handle_radius,
            float(cy1) - handle_radius,
            float(left_handle_x) + handle_radius,
            float(cy1) + handle_radius,
        ),
        "right_handle_bbox": (
            float(right_handle_x) - handle_radius,
            float(cy2) - handle_radius,
            float(right_handle_x) + handle_radius,
            float(cy2) + handle_radius,
        ),
        "line_points": line_points,
        "handle_radius": float(handle_radius),
        "interactive": bool(interactive),
    }


def update_preview_layout_separator_visual(host, separator: dict, data: dict | None = None) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    runtime = getattr(self, "_preview_layout_separator_runtime", None)
    if canvas is None or not isinstance(runtime, dict) or not isinstance(separator, dict):
        return False

    geometry = get_preview_layout_separator_canvas_geometry(self, separator, data)
    if not isinstance(geometry, dict):
        return False

    try:
        line_points = geometry["line_points"]
        canvas.coords(runtime["shadow_id"], *line_points)
        canvas.coords(runtime["line_id"], *line_points)
        canvas.coords(runtime["left_handle_id"], *geometry["left_handle_bbox"])
        canvas.coords(runtime["right_handle_id"], *geometry["right_handle_bbox"])
        runtime["separator"] = dict(separator)
        runtime["left_point"] = geometry["left_point"]
        runtime["right_point"] = geometry["right_point"]
        runtime["handle_radius"] = float(geometry.get("handle_radius", runtime.get("handle_radius", 7.0)) or 7.0)
        runtime["line_points"] = tuple(line_points)
        runtime["interactive"] = bool(geometry.get("interactive", runtime.get("interactive", True)))
        canvas.tag_raise("preview_layout_separator")
        return True
    except Exception:
        return False


def draw_preview_layout_separator(host, data: dict | None = None) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    state = getattr(self, "_preview_render_state", None) or {}
    if canvas is None or not state:
        return False

    try:
        canvas.delete("preview_layout_separator")
    except Exception:
        pass
    self._preview_layout_separator_runtime = {}

    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    if not isinstance(source_data, dict):
        return False

    try:
        image_w = max(1.0, float(state.get("orig_w", source_data.get("plate_image_width", 1.0)) or 1.0))
        image_h = max(1.0, float(state.get("orig_h", source_data.get("plate_image_height", 1.0)) or 1.0))
    except Exception:
        return False

    source_data["plate_image_width"] = float(image_w)
    source_data["plate_image_height"] = float(image_h)
    if not self._should_preview_use_two_row_layers(source_data):
        return False
    separator = self._ensure_preview_layout_separator(
        source_data,
        source_data.get("characters", []),
        image_w=image_w,
        image_h=image_h,
    )
    if not isinstance(separator, dict):
        return False

    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#101010")
    accent = palette.get("accent", "#ffd166")
    muted = palette.get("muted", "#8a8f98")
    border = palette.get("border", "#4b5563")
    interactive = bool(self._is_preview_layout_separator_interactive(source_data, ignore_active_char=True))
    separator_color = accent if interactive and str(separator.get("source", "")).startswith("manual") else blend_hex_colors(muted, border, 0.42)
    shadow = blend_hex_colors(separator_color, panel, 0.16)
    handle_fill = blend_hex_colors(panel, separator_color, 0.12)
    geometry = get_preview_layout_separator_canvas_geometry(self, separator, source_data)
    if not isinstance(geometry, dict):
        return False
    handle_radius = float(geometry.get("handle_radius", 5.5 if interactive else 3.5) or 5.5)
    tags = ("preview_layout_separator",)
    line_tags = tags + ("preview_layout_separator_line",) if interactive else tags
    left_handle_tags = tags + ("preview_layout_separator_handle", "preview_layout_separator_handle::left") if interactive else tags
    right_handle_tags = tags + ("preview_layout_separator_handle", "preview_layout_separator_handle::right") if interactive else tags

    try:
        shadow_id = canvas.create_line(
            *geometry["line_points"],
            fill=shadow,
            width=3,
            capstyle="round",
            tags=tags,
        )
        line_id = canvas.create_line(
            *geometry["line_points"],
            fill=separator_color,
            width=1.25,
            dash=(7, 6),
            capstyle="round",
            tags=line_tags,
        )
        left_handle_id = canvas.create_oval(
            *geometry["left_handle_bbox"],
            fill=handle_fill,
            outline=separator_color,
            width=1,
            tags=left_handle_tags,
        )
        right_handle_id = canvas.create_oval(
            *geometry["right_handle_bbox"],
            fill=handle_fill,
            outline=separator_color,
            width=1,
            tags=right_handle_tags,
        )
        self._preview_layout_separator_runtime = {
            "separator": dict(separator),
            "shadow_id": shadow_id,
            "line_id": line_id,
            "left_handle_id": left_handle_id,
            "right_handle_id": right_handle_id,
            "left_point": geometry["left_point"],
            "right_point": geometry["right_point"],
            "handle_radius": float(handle_radius),
            "line_points": tuple(geometry["line_points"]),
            "line_hit_radius": 11.0,
            "interactive": bool(interactive),
        }
        canvas.tag_raise("preview_layout_separator")
        return True
    except Exception:
        self._preview_layout_separator_runtime = {}
        return False


def redraw_preview_character_overlays_light(host) -> bool:
    self = host
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None or not getattr(self, "_preview_render_state", None):
        return False

    try:
        canvas.delete("preview_char")
        canvas.delete("preview_fast_detail")
        canvas.delete("preview_badge")
    except Exception:
        pass
    self._preview_char_runtime = {}
    self._preview_char_record_render_tags = {}
    self._preview_badge_runtime = {}
    self._preview_badge_drag_state = None

    self._draw_preview_plate_status_frame()
    self._draw_preview_layout_separator()

    redrawn = False
    chars = self._get_preview_active_character_records(create=False)
    for idx in range(len(chars)):
        redrawn = self._redraw_preview_character_overlay_only(int(idx)) or redrawn

    try:
        self._apply_preview_badge_selection_style()
        canvas.tag_raise("preview_char_add_preview")
        canvas.tag_raise("preview_overlay")
        canvas.tag_raise("preview_overlay_action")
    except Exception:
        pass
    return bool(redrawn)
