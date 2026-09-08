#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 plate layout semantics and character-source helpers."""

from __future__ import annotations

import copy
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

from ..config import logger
from ..character_recognition.reading_order import (
    annotate_records_reading_order,
    group_records_into_reading_rows,
    infer_plate_layout_from_records,
    record_bbox,
    separator_y_at_x,
    sort_records_reading_order,
)
from .z3_goldpack_ui import ensure_plate_source_metadata
from .z3_detection_runtime import clear_detection_review_snapshot_after_manual_edit
from .z3_preview_records import (
    build_character_source_tags,
    build_plate_layout_export_metadata,
    compose_character_source_tag,
    fusion_details_yolo_box_backend_positions,
    normalize_character_box_source,
    get_character_source_tag,
    normalize_character_source_kind,
    normalize_character_sign_source,
    normalize_plate_source_bucket,
    normalize_plate_source_origin,
)
from .z3_preview_ui import serialize_character_records

_VALID_PREVIEW_LAYOUT_OVERRIDES = {"single_row", "two_row"}


def _preview_manual_layout_record_key(data) -> str:
    if not isinstance(data, dict):
        return ""
    source_image = str(data.get("source_image", "") or "").strip().lower().replace("\\", "/")
    bbox = data.get("source_bbox")
    if not source_image or not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        return ""
    try:
        bbox_key = ",".join(f"{float(value):.2f}" for value in bbox[:4])
    except Exception:
        return ""
    return f"{source_image}|{bbox_key}"


def _sort_character_records_by_x(self, chars, data=None):
    if not isinstance(chars, list):
        return []

    forced_layout = self._get_preview_forced_layout_key(data)
    separator = self._get_preview_layout_separator_for_reading(data)

    return sort_records_reading_order(chars, forced_layout=forced_layout, separator=separator)

def _get_preview_forced_layout_key(self, data=None) -> str | None:
    if not isinstance(data, dict):
        return None
    override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if override in {"single_row", "two_row"}:
        return override
    layout = str(data.get("plate_layout", "") or "").strip().lower()
    if layout == "two_row":
        return "two_row"
    return None

def _is_preview_two_row_layout_active(self, data=None) -> bool:
    if not isinstance(data, dict):
        return False
    override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if override == "two_row":
        return True
    if override == "single_row":
        return False
    return str(data.get("plate_layout", "") or "").strip().lower() == "two_row"


def _is_preview_layout_separator_interactive(self, data=None, *, ignore_active_char: bool = False) -> bool:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    if not self._should_preview_use_two_row_layers(source_data):
        return False
    if ignore_active_char:
        return True
    if (
        getattr(self, "_preview_char_drag_state", None) is not None
        or getattr(self, "_preview_char_add_state", None) is not None
        or bool(getattr(self, "_preview_char_add_click_armed", False))
        or bool(getattr(self, "_preview_char_label_mode", False))
        or getattr(self, "_preview_char_label_active_index", None) is not None
    ):
        return False
    if bool(getattr(self, "_preview_char_edit_mode", False)) and getattr(self, "_preview_char_selected_index", None) is not None:
        return False
    return True

def _should_preview_use_two_row_layers(self, data=None) -> bool:
    if not isinstance(data, dict):
        return False
    override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if override == "single_row":
        return False
    if self._is_preview_two_row_layout_active(data):
        return True
    layout = str(data.get("plate_layout", "") or "").strip().lower()
    if layout in {"two_row_candidate", "2_row", "2r"}:
        return True
    chars = data.get("characters", [])
    if isinstance(chars, list):
        for rec in chars:
            if not isinstance(rec, dict):
                continue
            try:
                if int(rec.get("reading_row", 0) or 0) == 2:
                    return True
            except Exception:
                continue
    return False

def _normalize_preview_layout_separator(self, separator, *, image_w: float | None = None, image_h: float | None = None):
    if not isinstance(separator, dict):
        return None
    try:
        x1 = float(separator.get("x1", 0.0) or 0.0)
        y1 = float(separator.get("y1", 0.0) or 0.0)
        x2 = float(separator.get("x2", image_w if image_w is not None else x1) or 0.0)
        y2 = float(separator.get("y2", y1) or y1)
    except Exception:
        return None
    max_w = max(1.0, float(image_w if image_w is not None else max(x1, x2, 1.0)))
    max_h = max(1.0, float(image_h if image_h is not None else max(y1, y2, 1.0)))
    return {
        "x1": 0.0 if image_w is not None else max(0.0, min(max_w, x1)),
        "y1": max(0.0, min(max_h, y1)),
        "x2": max_w if image_w is not None else max(0.0, min(max_w, x2)),
        "y2": max(0.0, min(max_h, y2)),
        "source": str(separator.get("source", "manual") or "manual"),
    }

def _build_auto_preview_layout_separator(self, data, chars=None, *, image_w: float | None = None, image_h: float | None = None):
    source_chars = chars if isinstance(chars, list) else ((data or {}).get("characters", []) if isinstance(data, dict) else [])
    inferred_w = 1.0
    inferred_h = 1.0
    for rec in source_chars if isinstance(source_chars, list) else []:
        bbox = record_bbox(rec)
        if not bbox:
            continue
        inferred_w = max(inferred_w, float(bbox[2]))
        inferred_h = max(inferred_h, float(bbox[3]))
    max_w = max(1.0, float(image_w if image_w is not None else (data or {}).get("plate_image_width", inferred_w) or inferred_w))
    max_h = max(1.0, float(image_h if image_h is not None else (data or {}).get("plate_image_height", inferred_h) or inferred_h))
    rows = [row for row in group_records_into_reading_rows(source_chars) if row]
    separator_y = max_h * 0.5
    source = "default_midline"
    if len(rows) >= 2:
        top_boxes = [record_bbox(rec) for rec in rows[0]]
        bottom_boxes = [record_bbox(rec) for rec in rows[1]]
        top_boxes = [bbox for bbox in top_boxes if bbox]
        bottom_boxes = [bbox for bbox in bottom_boxes if bbox]
        if top_boxes and bottom_boxes:
            top_bottom = max(float(bbox[3]) for bbox in top_boxes)
            bottom_top = min(float(bbox[1]) for bbox in bottom_boxes)
            if bottom_top > top_bottom:
                separator_y = (top_bottom + bottom_top) / 2.0
                source = "row_gap"
            else:
                # Slanted or overlapping boxes may have no empty horizontal gap.
                # Split between the row centres instead of the image midpoint.
                from statistics import median
                top_center = median((float(bbox[1]) + float(bbox[3])) / 2 for bbox in top_boxes)
                bottom_center = median((float(bbox[1]) + float(bbox[3])) / 2 for bbox in bottom_boxes)
                separator_y = (top_center + bottom_center) / 2.0
                source = "row_centers"
    separator_y = max(0.0, min(max_h, float(separator_y)))
    return {
        "x1": 0.0,
        "y1": separator_y,
        "x2": max_w,
        "y2": separator_y,
        "source": f"auto_{source}",
    }

def _ensure_preview_layout_separator(self, data, chars=None, *, image_w: float | None = None, image_h: float | None = None):
    if not isinstance(data, dict):
        return None
    source_chars = chars if isinstance(chars, list) else data.get("characters", [])
    boxes = [bbox for rec in source_chars if (bbox := record_bbox(rec)) is not None]
    inferred_w = max((float(bbox[2]) for bbox in boxes), default=1.0)
    inferred_h = max((float(bbox[3]) for bbox in boxes), default=1.0)
    max_w = float(image_w or data.get("plate_image_width") or 1)
    max_h = float(image_h or data.get("plate_image_height") or 1)
    max_w = max_w if max_w > 1 else max(1.0, inferred_w)
    max_h = max_h if max_h > 1 else max(1.0, inferred_h)
    raw = data.get("layout_separator") if isinstance(data.get("layout_separator"), dict) else {}
    try:
        legacy_unit_separator = abs(float(raw.get("x2", 0)) - float(raw.get("x1", 0))) <= 1 and inferred_w > 1
    except (ValueError, TypeError):
        legacy_unit_separator = True
    existing = self._normalize_preview_layout_separator(data.get("layout_separator"), image_w=max_w, image_h=max_h)
    if existing and not legacy_unit_separator and str(existing.get("source", "")).lower().startswith("manual"):
        data["layout_separator"] = existing
        return existing
    separator = self._build_auto_preview_layout_separator(data, chars, image_w=max_w, image_h=max_h)
    data["layout_separator"] = separator
    # Geometry is the source of an automatic split. Cached reading_row values
    # may come from the old 1x1 separator and must not pull it above both rows.
    return separator

def _get_preview_layout_separator_for_reading(self, data=None):
    if not isinstance(data, dict) or not self._should_preview_use_two_row_layers(data):
        return None
    drag = getattr(self, "_preview_layout_separator_drag_state", None)
    if isinstance(drag, dict) and isinstance(drag.get("preview_separator"), dict):
        return self._normalize_preview_layout_separator(drag["preview_separator"])
    separator = self._normalize_preview_layout_separator(data.get("layout_separator"))
    if separator is None or abs(separator["x2"] - separator["x1"]) <= 1:
        separator = self._ensure_preview_layout_separator(data, data.get("characters", []))
    return separator


def _normalize_preview_manual_layout_override(self, data) -> str:
    if not isinstance(data, dict):
        return ""

    override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if override in _VALID_PREVIEW_LAYOUT_OVERRIDES:
        return override

    source = str(data.get("layout_source", "") or "").strip().lower()
    if source != "manual_override":
        return ""

    layout = str(data.get("plate_layout", "") or "").strip().lower()
    try:
        row_count = int(data.get("layout_row_count", 0) or 0)
    except Exception:
        row_count = 0

    if layout == "two_row" or row_count == 2:
        data["plate_layout_override"] = "two_row"
        return "two_row"
    if layout == "single_row" or row_count == 1:
        data["plate_layout_override"] = "single_row"
        return "single_row"
    return ""


def _capture_preview_manual_layout_state(self, data) -> dict:
    if not isinstance(data, dict):
        return {}

    override = self._normalize_preview_manual_layout_override(data)
    if override not in _VALID_PREVIEW_LAYOUT_OVERRIDES:
        return {}

    state = {"plate_layout_override": override}
    separator = data.get("layout_separator")
    if override == "two_row" and isinstance(separator, dict):
        state["layout_separator"] = copy.deepcopy(separator)
    for key in ("layout_override_source", "layout_override_updated_at"):
        if key in data:
            state[key] = copy.deepcopy(data.get(key))
    return state


def _restore_preview_manual_layout_state(self, data, state) -> bool:
    if not isinstance(data, dict) or not isinstance(state, dict):
        return False

    override = str(state.get("plate_layout_override", "") or "").strip().lower()
    if override not in _VALID_PREVIEW_LAYOUT_OVERRIDES:
        return False

    data["plate_layout_override"] = override
    for key in ("layout_override_source", "layout_override_updated_at"):
        if key in state:
            data[key] = copy.deepcopy(state.get(key))

    if override == "single_row":
        data.pop("layout_separator", None)
    elif isinstance(state.get("layout_separator"), dict):
        data["layout_separator"] = copy.deepcopy(state.get("layout_separator"))

    chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
    self._update_preview_plate_layout_metadata(data, chars)
    ordered_chars = self._sort_character_records_by_x(chars, data=data)
    ordered_chars = self._annotate_preview_character_reading_positions(ordered_chars, data=data)
    data["characters"] = ordered_chars
    data["status"] = self._derive_preview_status_from_data(data, ordered_chars)
    return True


def _backfill_preview_manual_layout_from_related_runs(self, metadata):
    if not isinstance(metadata, dict):
        return metadata

    try:
        current_meta_path = self._get_preview_metadata_path()
    except Exception:
        current_meta_path = None
    if current_meta_path is None:
        return metadata

    try:
        current_meta_path = Path(current_meta_path)
        runs_root = current_meta_path.parent.parent
    except Exception:
        return metadata
    if not runs_root.exists():
        return metadata

    cache_key = str(current_meta_path.resolve() if current_meta_path.exists() else current_meta_path)
    if getattr(self, "_preview_manual_layout_backfill_done_for", "") == cache_key:
        return metadata

    target_keys = {}
    for plate_id, raw_data in metadata.items():
        if not isinstance(raw_data, dict):
            continue
        if self._normalize_preview_manual_layout_override(raw_data):
            continue
        key = _preview_manual_layout_record_key(raw_data)
        if key:
            target_keys[key] = str(plate_id or "")
    if not target_keys:
        self._preview_manual_layout_backfill_done_for = cache_key
        return metadata

    recovered = {}
    try:
        sibling_meta_paths = sorted(
            path for path in runs_root.glob("run_*/metadata.json")
            if path != current_meta_path
        )
    except Exception:
        sibling_meta_paths = []

    for meta_path in sibling_meta_paths:
        try:
            source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(source_meta, dict):
            continue
        for source_data in source_meta.values():
            if not isinstance(source_data, dict):
                continue
            key = _preview_manual_layout_record_key(source_data)
            if key not in target_keys:
                continue
            override = self._normalize_preview_manual_layout_override(source_data)
            if override not in _VALID_PREVIEW_LAYOUT_OVERRIDES:
                continue
            payload = {
                "plate_layout_override": override,
                "layout_backfilled_from": meta_path.parent.name,
            }
            for field_name in (
                "layout_separator",
                "layout_override_source",
                "layout_override_updated_at",
                "plate_layout",
                "layout_row_count",
                "layout_confidence",
                "layout_source",
            ):
                if field_name in source_data:
                    payload[field_name] = copy.deepcopy(source_data.get(field_name))
            recovered[key] = payload

    applied = 0
    for raw_data in metadata.values():
        if not isinstance(raw_data, dict):
            continue
        if self._normalize_preview_manual_layout_override(raw_data):
            continue
        payload = recovered.get(_preview_manual_layout_record_key(raw_data))
        if not isinstance(payload, dict):
            continue
        raw_data.update(copy.deepcopy(payload))
        applied += 1

    self._preview_manual_layout_backfill_done_for = cache_key
    if applied > 0:
        try:
            logger.info("[Z3/PZ2] Przywrócono ręczny układ tablic z poprzednich runów: %s", applied)
        except Exception:
            pass
    return metadata


def _capture_preview_two_row_layout_state(self, data, chars=None) -> None:
    if not isinstance(data, dict):
        return
    source_chars = chars if isinstance(chars, list) else data.get("characters", [])
    data["_two_row_layout_backup"] = {
        "characters": copy.deepcopy(source_chars if isinstance(source_chars, list) else []),
        "layout_separator": copy.deepcopy(data.get("layout_separator")),
        "plate_layout": copy.deepcopy(data.get("plate_layout")),
        "layout_row_count": copy.deepcopy(data.get("layout_row_count")),
        "layout_confidence": copy.deepcopy(data.get("layout_confidence")),
        "layout_source": copy.deepcopy(data.get("layout_source")),
    }

def _restore_preview_two_row_layout_state(self, data) -> list | None:
    if not isinstance(data, dict):
        return None
    backup = data.get("_two_row_layout_backup")
    if not isinstance(backup, dict):
        return None
    chars = backup.get("characters")
    if isinstance(chars, list):
        data["characters"] = copy.deepcopy(chars)
    separator = backup.get("layout_separator")
    if isinstance(separator, dict):
        restored_separator = copy.deepcopy(separator)
        restored_separator["source"] = "manual_restored"
        data["layout_separator"] = restored_separator
    for key in ("plate_layout", "layout_row_count", "layout_confidence", "layout_source"):
        if key in backup:
            data[key] = copy.deepcopy(backup.get(key))
    return data.get("characters") if isinstance(data.get("characters"), list) else None

def _preview_separator_y_at_x(self, data, x: float) -> float | None:
    separator = self._get_preview_layout_separator_for_reading(data)
    return separator_y_at_x(separator, x)

def _get_preview_row_for_bbox(self, bbox, data=None) -> int | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    except Exception:
        return None
    sep_y = self._preview_separator_y_at_x(data, cx)
    if sep_y is None:
        return None
    return 1 if cy <= sep_y else 2

def _constrain_preview_char_bbox_to_layout_separator(self, bbox, data=None, *, row: int | None = None, min_size: float = 4.0):
    if not isinstance(data, dict) or not self._should_preview_use_two_row_layers(data):
        return bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return bbox
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    center_x = (x1 + x2) / 2.0
    sep_y = self._preview_separator_y_at_x(data, center_x)
    if sep_y is None:
        return [x1, y1, x2, y2]
    try:
        max_h = max(1.0, float((data or {}).get("plate_image_height", 0.0) or 0.0))
    except Exception:
        max_h = max(1.0, y2)
    try:
        resolved_row = int(row) if row is not None else int(self._get_preview_row_for_bbox([x1, y1, x2, y2], data) or 0)
    except Exception:
        resolved_row = 0
    min_h = max(1.0, float(min_size))
    if resolved_row == 1:
        y2 = min(y2, sep_y - 1.0)
        if y2 - y1 < min_h:
            y1 = max(0.0, y2 - min_h)
    elif resolved_row == 2:
        y1 = max(y1, sep_y + 1.0)
        if y2 - y1 < min_h:
            y2 = min(max_h, y1 + min_h)
    return [float(x1), float(y1), float(x2), float(y2)]

def _find_preview_layout_separator_handle_hit(self, canvas_x: float, canvas_y: float):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None
    if not self._is_preview_layout_separator_interactive():
        return None

    try:
        current_items = canvas.find_withtag("current")
    except Exception:
        current_items = ()
    for item in current_items or ():
        try:
            tags = canvas.gettags(item)
        except Exception:
            tags = ()
        for tag in tags:
            tag_text = str(tag or "")
            if tag_text.startswith("preview_layout_separator_handle::"):
                handle = tag_text.split("::", 1)[1].strip().lower()
                if handle in {"left", "right"}:
                    return handle
            if tag_text == "preview_layout_separator_line":
                return "line"

    runtime = getattr(self, "_preview_layout_separator_runtime", None)
    if not isinstance(runtime, dict):
        return None
    radius = max(6.0, float(runtime.get("handle_radius", 9.0) or 9.0) + 4.0)
    for handle in ("left", "right"):
        point = runtime.get(f"{handle}_point")
        if not (isinstance(point, (list, tuple)) and len(point) >= 2):
            continue
        try:
            dx = float(canvas_x) - float(point[0])
            dy = float(canvas_y) - float(point[1])
        except Exception:
            continue
        if (dx * dx) + (dy * dy) <= radius * radius:
            return handle

    line_points = runtime.get("line_points")
    if isinstance(line_points, (list, tuple)) and len(line_points) >= 4:
        try:
            px, py = float(canvas_x), float(canvas_y)
            line_hit_radius = max(7.0, float(runtime.get("line_hit_radius", 9.0) or 9.0))
            values = [float(value) for value in line_points]
            for idx in range(0, len(values) - 3, 2):
                x1, y1 = values[idx], values[idx + 1]
                x2, y2 = values[idx + 2], values[idx + 3]
                vx, vy = x2 - x1, y2 - y1
                length_sq = (vx * vx) + (vy * vy)
                if length_sq <= 0.0001:
                    distance = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
                else:
                    t = max(0.0, min(1.0, (((px - x1) * vx) + ((py - y1) * vy)) / length_sq))
                    nearest_x = x1 + (t * vx)
                    nearest_y = y1 + (t * vy)
                    distance = ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5
                if distance <= line_hit_radius:
                    return "line"
        except Exception:
            pass
    return None

def _set_preview_layout_separator_handle_y_from_canvas(self, handle: str, canvas_x: float, canvas_y: float):
    data = self._get_preview_active_data(create=True)
    if not isinstance(data, dict):
        return None
    if str(data.get("plate_layout_override", "") or "").strip().lower() != "two_row":
        data["plate_layout_override"] = "two_row"
    data["layout_override_source"] = "separator"
    data["layout_override_updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    state = getattr(self, "_preview_render_state", None) or {}
    try:
        image_w = max(1.0, float(state.get("orig_w", data.get("plate_image_width", 1.0)) or 1.0))
        image_h = max(1.0, float(state.get("orig_h", data.get("plate_image_height", 1.0)) or 1.0))
    except Exception:
        image_w = 1.0
        image_h = 1.0

    _img_x, img_y = self._preview_canvas_to_image_point(canvas_x, canvas_y)
    separator = self._ensure_preview_layout_separator(
        data,
        data.get("characters", []),
        image_w=image_w,
        image_h=image_h,
    )
    if not isinstance(separator, dict):
        return None

    y_value = max(0.0, min(image_h, float(img_y)))
    handle_key = str(handle or "").strip().lower()
    separator["x1"] = 0.0
    separator["x2"] = float(image_w)
    if handle_key == "line":
        separator["y1"] = y_value
        separator["y2"] = y_value
    elif handle_key == "right":
        separator["y2"] = y_value
    else:
        separator["y1"] = y_value
    separator["source"] = "manual"
    data["layout_separator"] = separator
    data["plate_image_width"] = float(image_w)
    data["plate_image_height"] = float(image_h)
    return separator

def _move_preview_layout_separator_from_canvas_delta(
    self,
    start_separator,
    start_canvas_x: float,
    start_canvas_y: float,
    current_canvas_x: float,
    current_canvas_y: float,
):
    data = self._get_preview_active_data(create=True)
    if not isinstance(data, dict):
        return None
    if str(data.get("plate_layout_override", "") or "").strip().lower() != "two_row":
        data["plate_layout_override"] = "two_row"
    data["layout_override_source"] = "separator"
    data["layout_override_updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if not isinstance(start_separator, dict):
        return self._set_preview_layout_separator_handle_y_from_canvas("line", current_canvas_x, current_canvas_y)

    state = getattr(self, "_preview_render_state", None) or {}
    try:
        image_w = max(1.0, float(state.get("orig_w", data.get("plate_image_width", 1.0)) or 1.0))
        image_h = max(1.0, float(state.get("orig_h", data.get("plate_image_height", 1.0)) or 1.0))
    except Exception:
        image_w = 1.0
        image_h = 1.0

    try:
        _start_x, start_img_y = self._preview_canvas_to_image_point(start_canvas_x, start_canvas_y)
        _current_x, current_img_y = self._preview_canvas_to_image_point(current_canvas_x, current_canvas_y)
        delta_y = float(current_img_y) - float(start_img_y)
    except Exception:
        delta_y = 0.0

    prepared = dict(start_separator)
    prepared["x1"] = 0.0
    prepared["x2"] = float(image_w)
    for key in ("y1", "y2"):
        try:
            base_y = float(start_separator.get(key, 0.0) or 0.0)
        except Exception:
            base_y = image_h * 0.5
        prepared[key] = max(0.0, min(float(image_h), base_y + delta_y))
    prepared["source"] = "manual"
    data["layout_separator"] = prepared
    data["plate_image_width"] = float(image_w)
    data["plate_image_height"] = float(image_h)
    return prepared

def _build_preview_layout_separator_drag_preview(self, drag_state, canvas_x: float, canvas_y: float):
    if not isinstance(drag_state, dict):
        return None
    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return None
    state = getattr(self, "_preview_render_state", None) or {}
    try:
        image_w = max(1.0, float(state.get("orig_w", data.get("plate_image_width", 1.0)) or 1.0))
        image_h = max(1.0, float(state.get("orig_h", data.get("plate_image_height", 1.0)) or 1.0))
    except Exception:
        image_w = 1.0
        image_h = 1.0

    start_separator = self._normalize_preview_layout_separator(
        drag_state.get("start_separator"),
        image_w=image_w,
        image_h=image_h,
    )
    if not isinstance(start_separator, dict):
        start_separator = {
            "x1": 0.0,
            "y1": image_h * 0.5,
            "x2": float(image_w),
            "y2": image_h * 0.5,
            "source": "manual",
        }

    prepared = dict(start_separator)
    prepared["x1"] = 0.0
    prepared["x2"] = float(image_w)
    handle_key = str(drag_state.get("handle", "line") or "line").strip().lower()

    try:
        _current_x, current_img_y = self._preview_canvas_to_image_point(canvas_x, canvas_y)
    except Exception:
        current_img_y = float(prepared.get("y1", image_h * 0.5) or image_h * 0.5)
    current_img_y = max(0.0, min(float(image_h), float(current_img_y)))

    if handle_key == "line":
        try:
            _start_x, start_img_y = self._preview_canvas_to_image_point(
                float(drag_state.get("start_canvas_x", canvas_x) or canvas_x),
                float(drag_state.get("start_canvas_y", canvas_y) or canvas_y),
            )
            delta_y = float(current_img_y) - float(start_img_y)
        except Exception:
            delta_y = 0.0
        for key in ("y1", "y2"):
            try:
                base_y = float(start_separator.get(key, image_h * 0.5) or image_h * 0.5)
            except Exception:
                base_y = image_h * 0.5
            prepared[key] = max(0.0, min(float(image_h), base_y + delta_y))
    elif handle_key == "right":
        prepared["y2"] = current_img_y
    else:
        prepared["y1"] = current_img_y

    prepared["source"] = "manual"
    return prepared

def _clamp_preview_layout_separator_to_existing_rows(self, data, separator):
    if not isinstance(data, dict) or not isinstance(separator, dict):
        return separator
    chars = data.get("characters", [])
    if not isinstance(chars, list) or not chars:
        return separator

    upper_bottoms = []
    lower_tops = []
    for rec in chars:
        if not isinstance(rec, dict):
            continue
        bbox = rec.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except Exception:
            continue
        if y2 < y1:
            y1, y2 = y2, y1
        try:
            row = int(rec.get("reading_row", 0) or 0)
        except Exception:
            row = 0
        if row not in (1, 2):
            row = self._get_preview_row_for_bbox([x1, y1, x2, y2], data) or 1
        if row == 1:
            upper_bottoms.append(float(y2))
        elif row == 2:
            lower_tops.append(float(y1))

    margin = 2.0
    min_y = (max(upper_bottoms) + margin) if upper_bottoms else None
    max_y = (min(lower_tops) - margin) if lower_tops else None
    if min_y is not None and max_y is not None and min_y > max_y:
        # Rzedy juz na siebie nachodza. Wtedy separator nie powinien naprawiac
        # boxow na sile, tylko zostaje w najbezpieczniejszym srodku konfliktu.
        midpoint = (min_y + max_y) / 2.0
        min_y = midpoint
        max_y = midpoint

    prepared = dict(separator)
    for key in ("y1", "y2"):
        try:
            y_value = float(prepared.get(key, 0.0) or 0.0)
        except Exception:
            continue
        if min_y is not None:
            y_value = max(float(min_y), y_value)
        if max_y is not None:
            y_value = min(float(max_y), y_value)
        prepared[key] = float(y_value)
    return prepared

def _apply_preview_layout_separator_constraints_to_chars(self, data, chars=None):
    source_chars = chars if isinstance(chars, list) else (data.get("characters", []) if isinstance(data, dict) else [])
    if not isinstance(source_chars, list):
        return []
    # Separator rzędów jest narzędziem semantycznym: zmienia kolejność
    # czytania (1.x/2.x), ale nie może przesuwać ani przycinać wykrytych boxów.
    return list(source_chars)

def _preview_layout_separator_conflicts_with_chars(self, data, chars=None) -> bool:
    if not isinstance(data, dict) or not self._should_preview_use_two_row_layers(data):
        return False
    separator = self._get_preview_layout_separator_for_reading(data)
    if not isinstance(separator, dict):
        return False
    source_chars = chars if isinstance(chars, list) else data.get("characters", [])
    if not isinstance(source_chars, list):
        return False

    margin = 0.5
    row_counts = {1: 0, 2: 0}
    for rec in source_chars:
        if not isinstance(rec, dict):
            continue
        bbox = rec.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except Exception:
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        row = self._get_preview_row_for_bbox([x1, y1, x2, y2], data)
        if row in (1, 2):
            row_counts[int(row)] += 1
        probes = (x1, (x1 + x2) / 2.0, x2)
        for probe_x in probes:
            sep_y = self._preview_separator_y_at_x(data, float(probe_x))
            if sep_y is None:
                continue
            if (y1 + margin) < float(sep_y) < (y2 - margin):
                return True
    if row_counts[1] <= 0 or row_counts[2] <= 0:
        return True
    return False

def _annotate_preview_character_reading_positions(self, chars, data=None):
    if not isinstance(chars, list):
        return []
    return annotate_records_reading_order(
        chars,
        forced_layout=self._get_preview_forced_layout_key(data),
        separator=self._get_preview_layout_separator_for_reading(data),
    )

def _update_preview_plate_layout_metadata(self, data, chars=None):
    if not isinstance(data, dict):
        return {}

    source_chars = chars if isinstance(chars, list) else data.get("characters", [])
    square_hint = bool(data.get("is_square", False))
    layout_meta = infer_plate_layout_from_records(source_chars, square_hint=square_hint)
    data["plate_layout_inferred"] = str(layout_meta.get("plate_layout", "unknown") or "unknown")
    data["layout_inferred_row_count"] = int(layout_meta.get("layout_row_count", 0) or 0)
    data["layout_inferred_confidence"] = float(layout_meta.get("layout_confidence", 0.0) or 0.0)
    data["layout_inferred_source"] = str(layout_meta.get("layout_source", "none") or "none")

    override = self._normalize_preview_manual_layout_override(data)
    if override in _VALID_PREVIEW_LAYOUT_OVERRIDES:
        layout_meta = {
            "plate_layout": override,
            "layout_row_count": 2 if override == "two_row" else 1,
            "layout_confidence": 1.0,
            "layout_source": "manual_override",
        }

    data.update(layout_meta)
    return layout_meta

def _get_preview_plate_layout_label(self, data) -> tuple[str, str]:
    override = str((data or {}).get("plate_layout_override", "") or "").strip().lower()
    if override == "single_row":
        return "1R*", "success"
    if override == "two_row":
        return "2R*", "warning"

    layout = str((data or {}).get("plate_layout", "") or "").strip().lower()
    confidence = (data or {}).get("layout_confidence", None)
    confidence_suffix = ""
    try:
        confidence_value = float(confidence)
        if 0.0 < confidence_value < 0.75:
            confidence_suffix = "?"
    except Exception:
        confidence_suffix = "?"

    if layout == "two_row":
        return f"2R{confidence_suffix}", "warning"
    if layout == "two_row_candidate":
        return "2R?", "warning"
    if layout == "single_row":
        return f"1R{confidence_suffix}", "success"
    return "?", "muted"

def _get_preview_plate_layout_dock_text(self, data) -> tuple[str, str]:
    override = str((data or {}).get("plate_layout_override", "") or "").strip().lower()
    if override == "single_row":
        return "RĘCZ. 1R", "success"
    if override == "two_row":
        return "RĘCZ. 2R", "warning"

    layout_text, tone = self._get_preview_plate_layout_label(data)
    normalized = str(layout_text or "").strip().upper()
    if normalized.startswith("1R?"):
        return "AUTO 1R?", tone
    if normalized.startswith("1R"):
        return "AUTO 1R", tone
    if normalized.startswith("2R?"):
        return "AUTO 2R?", tone
    if normalized.startswith("2R"):
        return "AUTO 2R", tone
    return "AUTO ?", "muted"

def _get_preview_character_reading_position_label(self, rec, data=None) -> str:
    if not isinstance(rec, dict):
        return ""
    try:
        row = int(rec.get("reading_row", 0) or 0)
        col = int(rec.get("reading_col", 0) or 0)
    except Exception:
        return ""
    if row <= 0 or col <= 0:
        return ""
    return f"{row}.{col}"

def _format_preview_layout_semantics(self, *, short: bool = False) -> str:
    if short:
        return "AUTO pokazuje wykryty układ. Ręczna decyzja ma pierwszeństwo; 1.2 = rząd 1, znak 2."
    return (
        "Semantyka układu: AUTO 1R/2R/2R? pokazuje układ wykryty przez program, "
        "1R oznacza tablicę jednorzędową, 2R tablicę dwurzędową, "
        "znak ? oznacza rozpoznanie niepewne, a * decyzję ręcznie wymuszoną przez użytkownika. "
        "Ręcznie wymuszony układ ma pierwszeństwo przed automatyczną inferencją. "
        "Małe indeksy przy boxach, np. 1.2, oznaczają kolejność czytania: rząd 1, znak 2."
    )

def _apply_preview_plate_layout_override(self, override: str | None, *, source: str = "manual"):
    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return "break"

    next_override = str(override or "").strip().lower()
    if next_override not in {"single_row", "two_row"}:
        next_override = ""

    try:
        self._push_preview_history_snapshot()
    except Exception:
        pass

    chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
    selected_records = {}
    for attr in ("_preview_char_selected_index", "_preview_char_label_active_index", "_preview_char_hover_label_index"):
        index = getattr(self, attr, None)
        if isinstance(index, int) and 0 <= index < len(chars):
            selected_records[attr] = chars[index]
    if next_override:
        data["plate_layout_override"] = next_override
        data["layout_override_source"] = str(source or "manual")
        data["layout_override_updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if next_override == "single_row":
            data.pop("layout_separator", None)
        elif next_override == "two_row":
            try:
                self._ensure_preview_layout_separator(data, chars)
            except Exception:
                pass
    else:
        data.pop("plate_layout_override", None)
        data.pop("layout_override_source", None)
        data.pop("layout_override_updated_at", None)
        # Legacy metadata can otherwise recreate the just-cleared override.
        data.pop("layout_source", None)
        data.pop("manual_layout", None)
        data.pop("layout_manual", None)
        data.pop("layout_separator", None)

    self._update_preview_plate_layout_metadata(data, chars)
    ordered_chars = self._sort_character_records_by_x(chars, data=data)
    for attr, record in selected_records.items():
        setattr(self, attr, next((index for index, item in enumerate(ordered_chars) if item is record), None))
    ordered_chars = self._annotate_preview_character_reading_positions(ordered_chars, data=data)
    data["characters"] = ordered_chars
    data["status"] = self._derive_preview_status_from_data(data, ordered_chars)

    try:
        self._schedule_preview_metadata_save(delay_ms=450)
        try:
            clear_detection_review_snapshot_after_manual_edit(self)
        except Exception:
            pass
        try:
            self._schedule_preview_info_refresh(delay_ms=900)
        except Exception:
            pass
    except Exception:
        self._persist_preview_metadata(success_message=None, refresh_list=False, sync_access=False)

    layout_text, tone = self._get_preview_plate_layout_dock_text(data)
    if next_override == "single_row":
        message = f"Wymuszono układ 1R. Status: {layout_text}."
    elif next_override == "two_row":
        message = f"Wymuszono układ 2R. Status: {layout_text}."
    else:
        message = f"Wrócono do automatycznej oceny układu. Status: {layout_text}."
    self._refresh_preview_layout_override_ui_light(
        message=message,
        tone=tone if tone in {"success", "warning", "error", "info", "muted"} else "info",
    )
    return "break"


def _open_preview_plate_layout_override_menu(self, event=None):
    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return "break"

    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return "break"

    current = str(data.get("plate_layout_override", "") or "").strip().lower()

    def _label(text: str, key: str) -> str:
        return f"✓ {text}" if current == key else f"  {text}"

    menu = tk.Menu(canvas, tearoff=0)
    menu.add_command(
        label=_label("Wymuś układ 1R", "single_row"),
        command=lambda: _apply_preview_plate_layout_override(self, "single_row", source="menu"),
    )
    menu.add_command(
        label=_label("Wymuś układ 2R", "two_row"),
        command=lambda: _apply_preview_plate_layout_override(self, "two_row", source="menu"),
    )
    menu.add_separator()
    menu.add_command(
        label=_label("AUTO - zdejmij wymuszenie", ""),
        command=lambda: _apply_preview_plate_layout_override(self, "", source="menu"),
    )
    self._preview_plate_layout_menu = menu
    try:
        x_root = int(getattr(event, "x_root", 0) or 0)
        y_root = int(getattr(event, "y_root", 0) or 0)
        if x_root <= 0 or y_root <= 0:
            x_root = int(canvas.winfo_rootx() + 24)
            y_root = int(canvas.winfo_rooty() + 24)
        menu.tk_popup(x_root, y_root)
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass
    return "break"


def _cycle_preview_plate_layout_override(self, event=None):
    data = self._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return "break"

    if event is not None and getattr(event, "x_root", None) is not None:
        return _open_preview_plate_layout_override_menu(self, event)

    current_override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if current_override == "single_row":
        next_override = "two_row"
    elif current_override == "two_row":
        next_override = "single_row"
    else:
        next_override = "single_row"
    return _apply_preview_plate_layout_override(self, next_override, source="cycle")

def _normalize_character_source_tag(self, raw_tag=None, method=None) -> str:
    tag = str(raw_tag or "").strip().lower().replace("-", "_")
    method_name = str(method or "").strip().lower()

    if tag in ("manual", "manual_correction", "cvat_manual"):
        return "manual"
    if tag in ("yolo_box_ocr", "yolo_ocr", "yolo_box_plus_ocr"):
        return "yolo_box_ocr"
    if tag in ("yolo_box", "yb"):
        return "yolo_box"
    if tag in ("yolo_symbol", "ys"):
        return "yolo_symbol"
    if tag in ("yolo_rescue", "rescue", "yolorescue"):
        return "yolo_rescue"
    if tag == "yolo":
        return "yolo"
    if tag == "ocr":
        return "ocr"

    if method_name in {"manual", "cvat_manual"}:
        return "manual"
    if method_name in {"yolo_ocr"}:
        return "yolo_box_ocr"
    if method_name in {"yolo_box"}:
        return "yolo_box"
    if method_name in {"yolo_symbol"}:
        return "yolo_symbol"
    if method_name == "yolo":
        return "yolo"
    return "ocr"

def _character_record_uses_yolo_box_backend(rec) -> bool:
    if isinstance(rec, dict):
        values = [
            rec.get("box_source"),
            rec.get("box_backend"),
            rec.get("box_backend_source"),
            rec.get("geometry_source"),
            rec.get("geometry_method"),
            rec.get("bbox_source"),
        ]
    else:
        values = [
            getattr(rec, "box_source", None),
            getattr(rec, "box_backend", None),
            getattr(rec, "box_backend_source", None),
            getattr(rec, "geometry_source", None),
            getattr(rec, "geometry_method", None),
            getattr(rec, "bbox_source", None),
        ]

    normalized_values = {
        str(value or "").strip().lower().replace("-", "_")
        for value in values
        if str(value or "").strip()
    }
    return bool(normalized_values.intersection({"yolo", "yolo_box", "yolo_backend", "yolo_filtered"}))

def _character_record_yolo_box_backend_confidence(rec) -> float | None:
    keys = ("box_backend_confidence", "geometry_confidence", "bbox_confidence")
    for key in keys:
        try:
            raw_value = rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)
        except Exception:
            raw_value = None
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            return float(raw_value)
        except Exception:
            continue
    return None

def _fusion_details_yolo_box_backend_positions(fusion_details) -> set[int]:
    return fusion_details_yolo_box_backend_positions(fusion_details)

def _normalize_character_source_kind(self, rec, data=None, plate_source_bucket: str = "") -> str:
    return normalize_character_source_kind(self, rec, data=data, plate_source_bucket=plate_source_bucket)

def _normalize_plate_source_bucket(self, raw_bucket=None, data=None, meta_path: Path | None = None) -> str:
    return normalize_plate_source_bucket(self, raw_bucket, data=data, meta_path=meta_path)

def _normalize_plate_source_origin(self, raw_origin=None, bucket: str = "") -> str:
    return normalize_plate_source_origin(self, raw_origin, bucket=bucket)

def _get_plate_source_bucket(self, data: dict, meta_path: Path | None = None) -> str:
    if not isinstance(data, dict):
        return "auto_preview"
    source_info = data.get("source_info", {})
    bucket = source_info.get("bucket") if isinstance(source_info, dict) else ""
    return self._normalize_plate_source_bucket(bucket, data=data, meta_path=meta_path)

def _ensure_plate_source_metadata(
    self,
    data: dict,
    *,
    plate_id: str = "",
    meta_path: Path | None = None,
    default_bucket: str | None = None,
    default_origin: str | None = None,
    import_batch_id: str = "",
    review_manifest_id: str = "",
    modified_by: str = "",
) -> bool:
    return ensure_plate_source_metadata(
        self,
        data,
        plate_id=plate_id,
        meta_path=meta_path,
        default_bucket=default_bucket,
        default_origin=default_origin,
        import_batch_id=import_batch_id,
        review_manifest_id=review_manifest_id,
        modified_by=modified_by,
    )

def _build_character_source_tags(self, chars, fusion_strategy="", fusion_details=None):
    return build_character_source_tags(
        self,
        chars,
        fusion_strategy=fusion_strategy,
        fusion_details=fusion_details,
    )

def _serialize_character_records(self, chars, fusion_strategy="", fusion_details=None, data=None):
    return serialize_character_records(
        self,
        chars,
        fusion_strategy=fusion_strategy,
        fusion_details=fusion_details,
        data=data,
    )

def _get_character_source_tag(self, rec, data=None, fallback_index: int = 0) -> str:
    return get_character_source_tag(self, rec, data=data, fallback_index=fallback_index)

def _get_character_box_source_tag(self, rec, data=None, fallback_index: int = 0) -> str:
    return normalize_character_box_source(self, rec, data=data, fallback_index=fallback_index)

def _get_character_sign_source_tag(self, rec, data=None, fallback_index: int = 0) -> str:
    return normalize_character_sign_source(self, rec, data=data, fallback_index=fallback_index)

def _compose_character_source_tag(self, box_source: str = "", sign_source: str = "") -> str:
    return compose_character_source_tag(box_source, sign_source)

def _get_character_source_kind(self, rec, data=None) -> str:
    plate_bucket = self._get_plate_source_bucket(data) if isinstance(data, dict) else ""
    return self._normalize_character_source_kind(rec, data=data, plate_source_bucket=plate_bucket)

def _get_plate_listbox_layout_flag(self, data: dict | None) -> str:
    if not isinstance(data, dict):
        return ""
    layout_label, _tone = self._get_preview_plate_layout_label(data)
    layout_label = str(layout_label or "").strip()
    return layout_label if layout_label and layout_label != "?" else ""

def _get_plate_layout_count_label(self, data: dict | None) -> str:
    if not isinstance(data, dict):
        return "?"
    layout_label, _tone = self._get_preview_plate_layout_label(data)
    layout_label = str(layout_label or "").strip()
    return layout_label if layout_label else "?"

def _build_plate_layout_export_metadata(self, data: dict | None) -> dict:
    return build_plate_layout_export_metadata(self, data)

def _is_uncertain_plate_layout(self, data: dict | None) -> bool:
    if not isinstance(data, dict):
        return True
    layout_label, _tone = self._get_preview_plate_layout_label(data)
    layout_label = str(layout_label or "?").strip().upper()
    layout = str(data.get("plate_layout", "") or "").strip().lower()
    source = str(data.get("layout_source", "") or "").strip().lower()
    override = str(data.get("plate_layout_override", "") or "").strip().lower()
    if override in {"single_row", "two_row"}:
        return False
    return layout_label in {"?", "2R?"} or layout == "two_row_candidate" or source == "square_hint"

def _count_uncertain_layout_plate_entries(self, plate_entries) -> int:
    count = 0
    for plate_entry in list(plate_entries or []):
        data = plate_entry.get("data", {}) if isinstance(plate_entry, dict) else {}
        if self._is_uncertain_plate_layout(data):
            count += 1
    return int(count)

def _confirm_export_with_uncertain_layouts(self, plate_entries, *, export_label: str) -> bool:
    uncertain_count = self._count_uncertain_layout_plate_entries(plate_entries)
    if uncertain_count <= 0:
        return True

    total_count = len(list(plate_entries or []))
    message = (
        f"W wybranym eksporcie ({export_label}) wykryto {uncertain_count} z {total_count} tablic "
        "z niepewnym układem 2R? lub ?. \n\n"
        "2R? oznacza, że program podejrzewa tablicę dwurzędową, ale nie ma pewności. "
        "To wpływa na kolejność czytania znaków i manifest eksportu.\n\n"
        "Najbezpieczniej wrócić do PZ2, użyć grupowania „2R?” i sprawdzić belkę rzędów na canvasie. "
        "Manualna praca na belce oraz boxach znaków ma pierwszeństwo przed automatem. "
        "Możesz też kontynuować eksport, jeśli świadomie akceptujesz ten stan."
    )
    try:
        parent = self.frame.winfo_toplevel()
    except Exception:
        parent = self.frame
    return bool(messagebox.askyesno("Niepewny układ tablic", message, parent=parent))

def _count_character_sources(self, chars, data=None):
    counts = {
        "ocr": 0,
        "yolo": 0,
        "yolo_box": 0,
        "yolo_symbol": 0,
        "yolo_box_ocr": 0,
        "yolo_rescue": 0,
        "manual": 0,
        "manual_box": 0,
        "manual_sign": 0,
        "generated_box": 0,
        "ocr_symbol": 0,
    }
    for idx, rec in enumerate(list(chars or [])):
        box_source = self._get_character_box_source_tag(rec, data=data, fallback_index=idx)
        sign_source = self._get_character_sign_source_tag(rec, data=data, fallback_index=idx)
        if box_source in counts:
            counts[box_source] += 1
        if sign_source in counts:
            counts[sign_source] += 1
        tag = self._get_character_source_tag(rec, data=data, fallback_index=idx)
        if tag not in counts:
            continue
        counts[tag] += 1
    return counts
