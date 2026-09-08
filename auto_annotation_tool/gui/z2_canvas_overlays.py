from __future__ import annotations

"""Z2 canvas overlay helpers extracted from tab_annotation.py."""

import copy
import csv
import json
from collections import deque
import os
import re
import tkinter as tk
import tkinter.font as tkfont
import xml.etree.ElementTree as ET
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import datetime
import threading
import math
import shutil
import time
import queue
import numpy as np
from .z2_drawer_slide import place_drawer, suspend_drawer

import cv2
from PIL import Image
from .inertial_scroll import InertialScrollController
from .zoomable_canvas import ZoomableCanvas
from .section_header_label import SectionHeaderLabel
from . import z2_workflow_methods
from .z2_main_widgets import create_annotation_widgets
from .z2_auto_scope_modal import prompt_plate_auto_scope_choice
from .z2_canvas_metrics_ui import (
    render_preview_metrics_grab_handle as z2_render_preview_metrics_grab_handle,
    render_preview_metrics_table as z2_render_preview_metrics_table,
)
from .z2_overlay_dock_ui import (
    get_preview_overlay_dock_theme as z2_get_preview_overlay_dock_theme,
    render_preview_campaign_gate_overlay as z2_render_preview_campaign_gate_overlay,
    render_preview_image_status_overlay as z2_render_preview_image_status_overlay,
    render_preview_overlay_dock as z2_render_preview_overlay_dock,
)
from .z2_legend_assets import (
    clear_preview_legend_image_cache as z2_clear_preview_legend_image_cache,
    get_preview_legend_compass_photo as z2_get_preview_legend_compass_photo,
    get_preview_legend_group_shell_photo as z2_get_preview_legend_group_shell_photo,
    get_preview_legend_interaction_marker_photo as z2_get_preview_legend_interaction_marker_photo,
    get_preview_legend_keycap_photo as z2_get_preview_legend_keycap_photo,
    get_preview_legend_shell_photo as z2_get_preview_legend_shell_photo,
    get_preview_legend_text_color as z2_get_preview_legend_text_color,
    hex_to_rgba as z2_hex_to_rgba,
    legend_color_is_light as z2_legend_color_is_light,
    normalize_preview_legend_interaction as z2_normalize_preview_legend_interaction,
    trim_preview_legend_image_cache as z2_trim_preview_legend_image_cache,
)
from .z2_legend_ui import (
    build_preview_controls_context_rows as z2_build_preview_controls_context_rows,
    build_preview_legend_sections as z2_build_preview_legend_sections,
    get_preview_controls_legend_target_height as z2_get_preview_controls_legend_target_height,
    get_preview_controls_legend_target_width as z2_get_preview_controls_legend_target_width,
    get_preview_legend_theme as z2_get_preview_legend_theme,
    preview_controls_row_is_multi_plate_badge as z2_preview_controls_row_is_multi_plate_badge,
    refresh_preview_controls_legend as z2_refresh_preview_controls_legend,
)
from .z2_model_quality_ui import (
    auto_model_metric_tone as z2_auto_model_metric_tone,
    describe_auto_model_quality as z2_describe_auto_model_quality,
    format_model_quality_epoch as z2_format_model_quality_epoch,
    format_model_quality_number as z2_format_model_quality_number,
    get_model_identity_caption as z2_get_model_identity_caption,
    model_quality_float as z2_model_quality_float,
    model_quality_value as z2_model_quality_value,
    render_auto_annotation_model_quality_table as z2_render_auto_annotation_model_quality_table,
    shorten_model_quality_text as z2_shorten_model_quality_text,
)
from .z2_actions import Z2ActionContext, build_z2_primary_actions, build_z2_secondary_actions
from .z2_campaign_flow import (
    apply_campaign_step2_workflow_preset,
    build_z2_cta_state_campaign,
    build_z2_layout_state_campaign,
    build_z2_left_panel_copy_payload_campaign,
    open_campaign_step2_entry as dispatch_open_campaign_step2_entry,
    open_existing_run_for_campaign_review,
    prepare_campaign_workflow_runtime,
)
from .z2_free_mode_flow import (
    AUTO_REVIEW_FOLLOWUP_TEXT,
    AUTO_REVIEW_FOLLOWUP_TITLE,
    MANUAL_REVIEW_FOLLOWUP_TEXT,
    MANUAL_REVIEW_FOLLOWUP_TITLE,
    build_z2_cta_state_free_mode,
    build_z2_layout_state_free_mode,
    build_z2_left_panel_copy_payload_free_mode,
    get_manual_review_history_display_entries as dispatch_get_manual_review_history_display_entries,
    jump_to_export_section as dispatch_jump_to_export_section,
    open_existing_run_for_manual_review as dispatch_open_existing_run_for_manual_review,
    prepare_free_mode_workflow_runtime,
    refresh_manual_review_history_ui as dispatch_refresh_manual_review_history_ui,
    remember_manual_review_run as dispatch_remember_manual_review_run,
    select_free_mode_route,
    set_manual_entry_mode as dispatch_set_manual_entry_mode,
)
from .z2_flow_models import Z2CopyPayload, Z2LeftPanelCopyContext
from .z2_shared_ui import (
    apply_z2_workflow_left_layout as dispatch_apply_z2_workflow_left_layout,
    apply_z2_workflow_cta_ui as dispatch_apply_z2_workflow_cta_ui,
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel

from ..config import CONFIG, logger, YOLO_AVAILABLE, AVAILABLE_DETECT_MODELS, SESSION

_COMPASS_LOG = logger
from ..annotators.runtime_factory import validate_pt_model_path_for_runtime
from ..icons import IconManager
from ..exporters import CVATExporter, ReportGenerator
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..project_cache import PROJECT_CACHE
from ..training import DatasetCreator
from ..utils import cleanup_gpu_memory, count_images_in_directory, format_duration, get_image_files, get_image_size
from ..data_models import Detection, AnnotationStatus, ImageAnnotation, AnnotationReport
from ..rectification.polygon_validator import PolygonValidator
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .canvas_progress_overlay import CanvasProgressOverlay

NAV_BUTTON_WIDTH = 18
def _get_preview_legend_font(self, size: int, weight: str = "normal"):
    cache = getattr(self, "_preview_legend_font_cache", None)
    if cache is None:
        cache = {}
        self._preview_legend_font_cache = cache

    key = (int(size), str(weight))
    font_obj = cache.get(key)
    if font_obj is None:
        font_obj = tkfont.Font(self.frame, family="Segoe UI", size=int(size), weight=str(weight))
        cache[key] = font_obj
    return font_obj

def _truncate_preview_filename(filename: str, max_chars: int = 44) -> str:
    text = str(filename or "").strip()
    if len(text) <= max_chars:
        return text
    head = max(10, (max_chars // 2) - 2)
    tail = max(10, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"

def _get_preview_legend_context(self):
    ann = self._get_preview_annotation()
    filename = self._truncate_preview_filename(getattr(ann, "filename", "Brak obrazu"))
    plates = self._get_plate_detections(ann)
    vehicles = self._get_vehicle_detections(ann)
    selected_idx = self._get_selected_plate_index_for_ann(ann) if ann is not None else None
    selected_vehicle_idx = self._get_selected_vehicle_index_for_ann(ann) if ann is not None else None
    current_no = 0 if not plates else ((int(selected_idx) + 1) if selected_idx is not None else 1)
    current_vehicle_no = 0 if not vehicles else ((int(selected_vehicle_idx) + 1) if selected_vehicle_idx is not None else 1)
    total_images = len(self.current_annotations or [])
    display_indices = list(getattr(self, "_preview_list_display_indices", []) or [])
    display_total = len(display_indices) if display_indices else total_images
    current_image_no = 0
    if self.current_preview_index is not None:
        current_display_index = self._get_preview_display_index(self.current_preview_index)
        if current_display_index is not None and display_total > 0:
            current_image_no = max(0, min(int(current_display_index), display_total - 1)) + 1
        elif total_images > 0:
            current_image_no = max(0, min(int(self.current_preview_index), total_images - 1)) + 1
    return {
        "filename": filename or "Brak obrazu",
        "image_text": f"Zdjęcie: {current_image_no}/{display_total}",
        "vehicle_text": f"Pojazd: {current_vehicle_no}/{len(vehicles)}",
        "plate_text": f"Tablica: {current_no}/{len(plates)}",
        "plate_total": int(len(plates)),
        "pool_text": (
            f"Pula: {total_images}"
            if total_images > 0 and display_total > 0 and display_total != total_images
            else ""
        ),
    }

def _fit_preview_text_to_width(self, text: str, max_width: float, font_obj) -> str:
    content = str(text or "")
    if not content:
        return ""
    try:
        limit = max(0.0, float(max_width or 0.0))
        if limit <= 0.0 or float(font_obj.measure(content)) <= limit:
            return content
        ellipsis = "..."
        if float(font_obj.measure(ellipsis)) > limit:
            return ""
        trimmed = content
        while trimmed and float(font_obj.measure(f"{trimmed}{ellipsis}")) > limit:
            trimmed = trimmed[:-1]
        return f"{trimmed.rstrip()}{ellipsis}" if trimmed else ellipsis
    except Exception:
        return content

def _legend_color_is_light(color: str) -> bool:
    return z2_legend_color_is_light(color)

def _get_preview_legend_text_color(self, fill: str) -> str:
    return z2_get_preview_legend_text_color(self, fill)

def _get_preview_legend_compass_photo(self, theme: dict, *, size: int = 38):
    return z2_get_preview_legend_compass_photo(self, theme, size=size)

def _trim_preview_legend_image_cache(self, *, max_entries: int = 48):
    return z2_trim_preview_legend_image_cache(self, max_entries=max_entries)

def _clear_preview_legend_image_cache(self):
    return z2_clear_preview_legend_image_cache(self)

def _get_preview_legend_shell_photo(self, theme: dict, *, width: int, height: int, overlay_x: int = 10, overlay_y: int = 10):
    return z2_get_preview_legend_shell_photo(
        self,
        theme,
        width=width,
        height=height,
        overlay_x=overlay_x,
        overlay_y=overlay_y,
    )

def _get_preview_legend_theme(self) -> dict:
    return z2_get_preview_legend_theme(self)

def _get_preview_legend_keycap_photo(self, text: str, *, fill: str, outline: str, text_fill: str):
    return z2_get_preview_legend_keycap_photo(
        self,
        text,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )

def _get_preview_legend_group_shell_photo(self, theme: dict, *, width: int, height: int, outline: str):
    return z2_get_preview_legend_group_shell_photo(
        self,
        theme,
        width=width,
        height=height,
        outline=outline,
    )

def _normalize_preview_legend_interaction(interaction: str | None) -> str:
    return z2_normalize_preview_legend_interaction(interaction)

def _get_preview_legend_interaction_marker_photo(
    self,
    mode: str,
    *,
    width: int,
    height: int,
    fill: str,
    outline: str,
    text_fill: str,
):
    return z2_get_preview_legend_interaction_marker_photo(
        self,
        mode,
        width=width,
        height=height,
        fill=fill,
        outline=outline,
        text_fill=text_fill,
    )

def _is_preview_controls_legend_expanded(self) -> bool:
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        return bool(getattr(self, "_preview_controls_legend_fullscreen_expanded", False))
    return bool(getattr(self, "_preview_controls_legend_inline_expanded", False))

def _toggle_preview_controls_legend(self, event=None):
    before = self._is_preview_controls_legend_expanded()
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        self._preview_controls_legend_fullscreen_expanded = not bool(
            getattr(self, "_preview_controls_legend_fullscreen_expanded", False)
        )
        self._preview_controls_legend_render_key = None
        self._preview_controls_legend_scroll_offset = 0.0
        self._animate_preview_controls_legend_height()
    else:
        self._preview_controls_legend_inline_expanded = not bool(
            getattr(self, "_preview_controls_legend_inline_expanded", False)
        )
        # Inline legend can be clipped and scrolled. Avoid height animation here:
        # animated viewport and scrollregion used to fight each other and made
        # the compass look like it stopped expanding.
        self._cancel_preview_controls_legend_animation()
        self._preview_controls_legend_render_key = None
        self._preview_controls_legend_scroll_offset = 0.0
        self._preview_controls_legend_render_width_override = 0.0
        self._preview_controls_legend_render_height_override = 0.0
        self._place_preview_legend_overlay(refresh=True)
    try:
        _COMPASS_LOG.info(
            "[Z2 COMPASS] toggle expanded %s->%s fullscreen=%s visible=%s",
            int(bool(before)),
            int(bool(self._is_preview_controls_legend_expanded())),
            int(bool(getattr(self, "_preview_fullscreen_active", False))),
            int(bool(getattr(self, "_preview_controls_legend_visible", True))),
        )
    except Exception:
        pass
    return "break"

def _is_preview_controls_legend_grab_hit(self, local_x: float, local_y: float) -> bool:
    bbox = getattr(self, "_preview_controls_legend_grab_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x1 <= float(local_x) <= x2 and y1 <= float(local_y) <= y2

def _is_preview_controls_legend_toggle_hit(self, local_x: float, local_y: float) -> bool:
    bbox = getattr(self, "_preview_controls_legend_toggle_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x1 <= float(local_x) <= x2 and y1 <= float(local_y) <= y2

def _is_preview_controls_legend_compact_hit(self, local_x: float, local_y: float) -> bool:
    if self._is_preview_controls_legend_expanded():
        return False
    try:
        width = float(
            getattr(self, "_preview_controls_legend_current_width", 0.0)
            or self._get_preview_controls_legend_target_width()
        )
        height = float(
            getattr(self, "_preview_controls_legend_current_height", 0.0)
            or self._get_preview_controls_legend_target_height(width)
        )
    except Exception:
        width = height = 0.0
    if width <= 0.0 or height <= 0.0:
        return False
    return 0.0 <= float(local_x) <= width and 0.0 <= float(local_y) <= height

def _clamp_preview_controls_legend_offsets(self, offset_x: float, offset_y: float, *, width: float | None = None, height: float | None = None) -> tuple[float, float]:
    canvas_frame = getattr(self, "canvas_frame", None)
    try:
        frame_w = float(canvas_frame.winfo_width() or canvas_frame.winfo_reqwidth() or 0) if canvas_frame is not None else 0.0
        frame_h = float(canvas_frame.winfo_height() or canvas_frame.winfo_reqheight() or 0) if canvas_frame is not None else 0.0
    except Exception:
        frame_w = 0.0
        frame_h = 0.0
    if frame_w <= 0.0:
        frame_w = float((width or 0.0) + 20.0)
    if frame_h <= 0.0:
        frame_h = float((height or 0.0) + 20.0)
    safe_w = float(width or getattr(self, "_preview_controls_legend_current_width", 0.0) or self._get_preview_controls_legend_target_width())
    safe_h = float(height or getattr(self, "_preview_controls_legend_current_height", 0.0) or self._get_preview_controls_legend_target_height(safe_w))
    clamped_x = min(max(0.0, float(offset_x)), max(0.0, frame_w - safe_w))
    clamped_y = min(max(0.0, float(offset_y)), max(0.0, frame_h - safe_h))
    return float(clamped_x), float(clamped_y)

def _preview_controls_legend_event_xy(self, event) -> tuple[float, float]:
    local_x = float(getattr(event, "x", 0.0) or 0.0)
    local_y = float(getattr(event, "y", 0.0) or 0.0)
    canvas = getattr(self, "preview_controls_canvas", None)
    try:
        if canvas is not None and getattr(event, "widget", None) is canvas:
            if bool(getattr(self, "_preview_controls_legend_scroll_enabled", False)):
                local_x = float(canvas.canvasx(local_x))
                local_y = float(canvas.canvasy(local_y))
        elif bool(getattr(self, "_preview_controls_legend_scroll_enabled", False)):
            local_y += float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0)
    except Exception:
        pass
    return local_x, local_y

def _preview_controls_event_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

def _sync_preview_controls_scrollbar_thumb(self) -> None:
    canvas = getattr(self, "preview_controls_canvas", None)
    vbar = getattr(self, "preview_controls_vbar", None)
    if canvas is None or vbar is None:
        return
    try:
        content_h = float(getattr(self, "_preview_controls_legend_content_height", 0.0) or 0.0)
        viewport_h = float(
            getattr(self, "_preview_controls_legend_current_height", 0.0)
            or canvas.winfo_height()
            or 0.0
        )
        max_scroll = max(0.0, content_h - viewport_h)
        if max_scroll <= 0.0 or content_h <= 0.0:
            vbar.set(0.0, 1.0)
            return
        offset = max(0.0, min(float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0), max_scroll))
        first = max(0.0, min(1.0, offset / max(content_h, 1.0)))
        last = max(first, min(1.0, (offset + viewport_h) / max(content_h, 1.0)))
        vbar.set(first, last)
    except Exception:
        pass

def _apply_preview_controls_legend_scroll_offset(self, next_offset: float) -> bool:
    canvas = getattr(self, "preview_controls_canvas", None)
    if canvas is None:
        return False
    try:
        if not canvas.find_withtag("preview_legend_scroll_content"):
            return False
        content_h = float(getattr(self, "_preview_controls_legend_content_height", 0.0) or 0.0)
        viewport_h = float(
            getattr(self, "_preview_controls_legend_current_height", 0.0)
            or canvas.winfo_height()
            or 0.0
        )
        max_scroll = max(0.0, content_h - viewport_h)
        next_scroll = max(0.0, min(float(next_offset or 0.0), max_scroll))
        old_scroll = float(
            getattr(
                self,
                "_preview_controls_legend_rendered_scroll_offset",
                getattr(self, "_preview_controls_legend_scroll_offset", 0.0),
            )
            or 0.0
        )
        delta = old_scroll - next_scroll
        if abs(delta) > 0.01:
            canvas.move("preview_legend_scroll_content", 0.0, delta)
        self._preview_controls_legend_scroll_offset = float(next_scroll)
        self._preview_controls_legend_rendered_scroll_offset = float(next_scroll)
        _sync_preview_controls_scrollbar_thumb(self)
        return True
    except Exception:
        return False

def _preview_controls_pointer_inside(self, event=None) -> bool:
    """Return True only when the mouse is over the compass overlay."""
    event_x = None
    event_y = None
    if event is not None:
        try:
            event_x = int(getattr(event, "x_root"))
            event_y = int(getattr(event, "y_root"))
        except Exception:
            event_x = None
            event_y = None
    for widget_name in (
        "preview_hint_frame",
        "preview_controls_canvas",
        "preview_controls_vbar",
    ):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            px = event_x if event_x is not None else int(widget.winfo_pointerx())
            py = event_y if event_y is not None else int(widget.winfo_pointery())
            x1 = int(widget.winfo_rootx())
            y1 = int(widget.winfo_rooty())
            x2 = x1 + int(widget.winfo_width())
            y2 = y1 + int(widget.winfo_height())
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        except Exception:
            continue
    return False

def _on_preview_controls_scrollbar_command(self, *args):
    canvas = getattr(self, "preview_controls_canvas", None)
    if canvas is None:
        return "break"
    try:
        content_h = float(getattr(self, "_preview_controls_legend_content_height", 0.0) or 0.0)
        viewport_h = float(
            getattr(self, "_preview_controls_legend_current_height", 0.0)
            or canvas.winfo_height()
            or 0.0
        )
        max_scroll = max(0.0, content_h - viewport_h)
        next_offset = float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0)
        if max_scroll <= 0.0:
            next_offset = 0.0
        elif len(args) >= 2 and str(args[0]) == "moveto":
            fraction = max(0.0, min(float(args[1]), 1.0))
            # WebSlimScrollbar reports the standard Tk "first" fraction
            # (top/content), so convert it back to pixels using content height.
            next_offset = max(0.0, min(fraction * content_h, max_scroll))
        elif len(args) >= 3 and str(args[0]) == "scroll":
            units = float(args[1])
            mode = str(args[2])
            step = viewport_h if mode == "pages" else 54.0
            current = float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0)
            next_offset = max(0.0, min(current + (units * step), max_scroll))
        if not _apply_preview_controls_legend_scroll_offset(self, next_offset):
            self._preview_controls_legend_scroll_offset = float(next_offset)
            self._preview_controls_legend_render_key = None
            self._preview_controls_legend_static_key = None
            self._refresh_preview_controls_legend()
            _sync_preview_controls_scrollbar_thumb(self)
    except Exception:
        pass
    return "break"

def _on_preview_controls_legend_mousewheel(self, event=None):
    if event is None:
        return None
    direct_widgets = {
        widget
        for widget in (
            getattr(self, "preview_hint_frame", None),
            getattr(self, "preview_controls_canvas", None),
            getattr(self, "preview_controls_vbar", None),
        )
        if widget is not None
    }
    if getattr(event, "widget", None) not in direct_widgets and not _preview_controls_pointer_inside(self, event):
        return None
    if not bool(getattr(self, "_preview_controls_legend_scroll_enabled", False)):
        try:
            if self._is_preview_controls_legend_expanded():
                self._place_preview_legend_overlay(refresh=True)
        except Exception:
            pass
        if not bool(getattr(self, "_preview_controls_legend_scroll_enabled", False)):
            return None
    canvas = getattr(self, "preview_controls_canvas", None)
    if canvas is None:
        return None
    try:
        content_h = float(getattr(self, "_preview_controls_legend_content_height", 0.0) or 0.0)
        viewport_h = float(getattr(self, "_preview_controls_legend_current_height", 0.0) or canvas.winfo_height() or 0.0)
        max_scroll = max(0.0, content_h - viewport_h)
    except Exception:
        max_scroll = 0.0
    if max_scroll <= 0.0:
        self._preview_controls_legend_scroll_offset = 0.0
        return None
    delta = _preview_controls_event_int(getattr(event, "delta", 0) or 0)
    button = _preview_controls_event_int(getattr(event, "num", 0) or 0)
    if button == 4:
        steps = -1
    elif button == 5:
        steps = 1
    elif delta:
        steps = -1 if delta > 0 else 1
    else:
        return None
    current = float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0)
    next_offset = max(0.0, min(current + (steps * 54.0), max_scroll))
    try:
        if not _apply_preview_controls_legend_scroll_offset(self, next_offset):
            self._preview_controls_legend_scroll_offset = float(next_offset)
            self._preview_controls_legend_render_key = None
            self._preview_controls_legend_static_key = None
            self._refresh_preview_controls_legend()
            _sync_preview_controls_scrollbar_thumb(self)
    except Exception:
        pass
    return "break"

def _handle_preview_canvas_overlay_mousewheel(self, event=None):
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        return None
    if not bool(getattr(self, "_preview_controls_legend_hover", False)) and not _preview_controls_pointer_inside(self, event):
        return None
    result = _on_preview_controls_legend_mousewheel(self, event)
    return "break" if result == "break" or _preview_controls_pointer_inside(self, event) else result

def _on_preview_controls_legend_enter(self, event=None):
    """Let the compass capture mouse-wheel events as soon as the cursor enters it."""
    self._preview_controls_legend_hover = True
    canvas = getattr(self, "preview_controls_canvas", None)
    target = canvas or getattr(event, "widget", None)
    try:
        if target is not None:
            target.focus_set()
    except Exception:
        pass
    try:
        self._sync_preview_controls_legend_scrollbar()
    except Exception:
        pass
    return None

def _on_preview_controls_legend_press(self, event=None):
    if event is None:
        return None
    try:
        local_x, local_y = _preview_controls_legend_event_xy(self, event)
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        return "break"
    try:
        grab_widget = getattr(event, "widget", None)
        if grab_widget is not None:
            grab_widget.grab_set()
            self._preview_controls_legend_active_grab_widget = grab_widget
    except Exception:
        self._preview_controls_legend_active_grab_widget = None
    if self._is_preview_controls_legend_grab_hit(local_x, local_y):
        try:
            _COMPASS_LOG.info("[Z2 COMPASS] press grab x=%.1f y=%.1f", local_x, local_y)
        except Exception:
            pass
        self._preview_controls_legend_drag_state = {
            "press_root_x": root_x,
            "press_root_y": root_y,
            "start_x": float(getattr(self, "_preview_controls_legend_offset_x", 10.0) or 10.0),
            "start_y": float(getattr(self, "_preview_controls_legend_offset_y", 10.0) or 10.0),
        }
        self._preview_controls_legend_click_state = None
        self._update_preview_controls_legend_cursor(local_x, local_y)
        return "break"
    self._preview_controls_legend_drag_state = None
    compact_hit = bool(_is_preview_controls_legend_compact_hit(self, local_x, local_y))
    self._preview_controls_legend_click_state = {
        "press_root_x": root_x,
        "press_root_y": root_y,
        "press_local_x": local_x,
        "press_local_y": local_y,
        "toggle_hit": bool(self._is_preview_controls_legend_toggle_hit(local_x, local_y)),
        "compact_hit": compact_hit,
        "start_x": float(getattr(self, "_preview_controls_legend_offset_x", 10.0) or 10.0),
        "start_y": float(getattr(self, "_preview_controls_legend_offset_y", 10.0) or 10.0),
    }
    try:
        _COMPASS_LOG.info(
            "[Z2 COMPASS] press click x=%.1f y=%.1f toggle=%s compact=%s expanded=%s",
            local_x,
            local_y,
            int(bool(self._preview_controls_legend_click_state.get("toggle_hit"))),
            int(compact_hit),
            int(bool(self._is_preview_controls_legend_expanded())),
        )
    except Exception:
        pass
    return "break"

def _on_preview_controls_legend_drag(self, event=None):
    drag_state = getattr(self, "_preview_controls_legend_drag_state", None)
    if not isinstance(drag_state, dict):
        click_state = getattr(self, "_preview_controls_legend_click_state", None)
        if event is None or not isinstance(click_state, dict) or not bool(click_state.get("compact_hit")):
            return None
        try:
            root_x = float(getattr(event, "x_root", 0.0) or 0.0)
            root_y = float(getattr(event, "y_root", 0.0) or 0.0)
        except Exception:
            return "break"
        moved_x = root_x - float(click_state.get("press_root_x", root_x))
        moved_y = root_y - float(click_state.get("press_root_y", root_y))
        if abs(moved_x) <= 4.0 and abs(moved_y) <= 4.0:
            return "break"
        drag_state = {
            "press_root_x": float(click_state.get("press_root_x", root_x)),
            "press_root_y": float(click_state.get("press_root_y", root_y)),
            "start_x": float(click_state.get("start_x", getattr(self, "_preview_controls_legend_offset_x", 10.0))),
            "start_y": float(click_state.get("start_y", getattr(self, "_preview_controls_legend_offset_y", 10.0))),
        }
        self._preview_controls_legend_drag_state = drag_state
        self._preview_controls_legend_click_state = None
        self._update_preview_controls_legend_cursor()
    if event is None:
        return None
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        return "break"
    next_x = float(drag_state.get("start_x", 10.0)) + (root_x - float(drag_state.get("press_root_x", root_x)))
    next_y = float(drag_state.get("start_y", 10.0)) + (root_y - float(drag_state.get("press_root_y", root_y)))
    clamped_x, clamped_y = self._clamp_preview_controls_legend_offsets(next_x, next_y)
    self._preview_controls_legend_offset_x = clamped_x
    self._preview_controls_legend_offset_y = clamped_y
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        self._preview_controls_legend_inline_manual_position = True
    self._place_preview_legend_overlay(refresh=False)
    return "break"

def _update_preview_controls_legend_cursor(self, local_x: float | None = None, local_y: float | None = None):
    cursor = "arrow"
    if isinstance(getattr(self, "_preview_controls_legend_drag_state", None), dict):
        cursor = "fleur"
    elif local_x is not None and local_y is not None and self._is_preview_controls_legend_grab_hit(local_x, local_y):
        cursor = "fleur"
    elif local_x is not None and local_y is not None and self._is_preview_controls_legend_toggle_hit(local_x, local_y):
        cursor = "hand2"
    elif local_x is not None and local_y is not None and _is_preview_controls_legend_compact_hit(self, local_x, local_y):
        cursor = "hand2"
    for widget_name in ("preview_hint_frame", "preview_controls_canvas"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(cursor=cursor)
        except Exception:
            pass

def _on_preview_controls_legend_motion(self, event=None):
    if event is None:
        return None
    try:
        local_x, local_y = _preview_controls_legend_event_xy(self, event)
        self._update_preview_controls_legend_cursor(
            local_x,
            local_y,
        )
    except Exception:
        self._update_preview_controls_legend_cursor()
    return None

def _on_preview_controls_legend_leave(self, event=None):
    self._preview_controls_legend_hover = False
    for widget_name in ("preview_hint_frame", "preview_controls_canvas"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(cursor="arrow")
        except Exception:
            pass
    return None

def _on_preview_controls_legend_release(self, event=None):
    drag_state = getattr(self, "_preview_controls_legend_drag_state", None)
    click_state = getattr(self, "_preview_controls_legend_click_state", None)
    self._preview_controls_legend_drag_state = None
    self._preview_controls_legend_click_state = None
    grab_widget = getattr(self, "_preview_controls_legend_active_grab_widget", None)
    self._preview_controls_legend_active_grab_widget = None
    if grab_widget is not None:
        try:
            grab_widget.grab_release()
        except Exception:
            pass
    self._update_preview_controls_legend_cursor()
    if isinstance(drag_state, dict):
        return "break"
    if not isinstance(click_state, dict) or event is None:
        return None
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
        local_x, local_y = _preview_controls_legend_event_xy(self, event)
    except Exception:
        return "break"
    moved_x = abs(root_x - float(click_state.get("press_root_x", root_x)))
    moved_y = abs(root_y - float(click_state.get("press_root_y", root_y)))
    started_on_toggle = bool(click_state.get("toggle_hit"))
    ended_on_toggle = bool(self._is_preview_controls_legend_toggle_hit(local_x, local_y))
    started_on_compact = bool(click_state.get("compact_hit"))
    ended_on_compact = bool(_is_preview_controls_legend_compact_hit(self, local_x, local_y))
    try:
        _COMPASS_LOG.info(
            "[Z2 COMPASS] release moved=%.1f/%.1f started_toggle=%s ended_toggle=%s started_compact=%s ended_compact=%s drag=%s",
            moved_x,
            moved_y,
            int(started_on_toggle),
            int(ended_on_toggle),
            int(started_on_compact),
            int(ended_on_compact),
            int(isinstance(drag_state, dict)),
        )
    except Exception:
        pass
    tolerance = 16.0 if (started_on_toggle or ended_on_toggle) else 10.0
    if moved_x > tolerance or moved_y > tolerance:
        return "break"
    if not (started_on_toggle or ended_on_toggle or started_on_compact or ended_on_compact):
        return "break"
    return self._toggle_preview_controls_legend(event)

def _hide_preview_controls_legend_overlay(self) -> None:
    try:
        self._cancel_preview_controls_legend_animation()
    except Exception:
        pass
    self._preview_controls_legend_drag_state = None
    self._preview_controls_legend_click_state = None
    self._preview_controls_legend_grab_bbox = None
    self._preview_controls_legend_toggle_bbox = None
    grab_widget = getattr(self, "_preview_controls_legend_active_grab_widget", None)
    self._preview_controls_legend_active_grab_widget = None
    if grab_widget is not None:
        try:
            grab_widget.grab_release()
        except Exception:
            pass
    self._preview_controls_legend_render_key = None
    self._preview_controls_legend_render_width_override = 0.0
    self._preview_controls_legend_render_height_override = 0.0
    overlay = getattr(self, "preview_hint_frame", None)
    if overlay is not None:
        try:
            overlay.place_forget()
        except Exception:
            pass
    canvas = getattr(self, "preview_controls_canvas", None)
    if canvas is not None:
        try:
            canvas.delete("all")
        except Exception:
            pass

def _raise_preview_controls_legend_overlay(self) -> None:
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        return
    overlay = getattr(self, "preview_hint_frame", None)
    if overlay is None:
        return
    try:
        if not str(overlay.winfo_manager()):
            return
        overlay.lift()
        canvas = getattr(self, "preview_controls_canvas", None)
        if canvas is not None:
            canvas.lift()
    except Exception:
        pass

def _hide_preview_image_status_overlay(self) -> None:
    self._preview_image_status_overlay_render_key = None
    frame = getattr(self, "preview_image_status_frame", None)
    if frame is not None:
        try:
            frame.place_forget()
        except Exception:
            pass

def _hide_preview_overlay_dock_stack(self) -> None:
    """Hide the whole right-side overlay stack before canvas geometry changes."""
    suspend_drawer(self)
    self._preview_overlay_dock_render_key = None
    self._preview_overlay_dock_size = None
    self._preview_overlay_dock_size_key = None
    self._preview_overlay_dock_pre_gate_key = None
    self._preview_overlay_dock_gate_render_key = None
    self._preview_overlay_dock_inline_gate_state = None
    self._preview_overlay_dock_inline_gate_source_key = None
    dock = getattr(self, "preview_overlay_dock", None)
    if dock is not None:
        try:
            dock.place_forget()
        except Exception:
            pass
    try:
        self._hide_preview_image_status_overlay()
    except Exception:
        pass
    try:
        self._hide_preview_campaign_gate_overlay()
    except Exception:
        pass

def _preview_fullscreen_overlay_transition_blocked(self) -> bool:
    return bool(getattr(self, "_preview_fullscreen_transition_active", False)) and not bool(
        getattr(self, "_preview_fullscreen_overlay_ready", False)
    )

def _on_preview_canvas_frame_configure(self, event=None):
    if _preview_fullscreen_overlay_transition_blocked(self):
        return None
    try:
        self._place_preview_legend_overlay()
    except Exception:
        pass
    try:
        self._update_preview_canvas_metrics_overlay()
    except Exception:
        pass
    try:
        self._place_preview_overlay_dock()
    except Exception:
        pass
    try:
        self._place_preview_campaign_gate_overlay()
    except Exception:
        pass
    return None

def _render_preview_image_status_overlay(self, *, force_render: bool = False) -> tuple[int, int]:
    return z2_render_preview_image_status_overlay(self, force_render=force_render)

def _place_preview_image_status_overlay(self, *, force_render: bool = False) -> None:
    frame = getattr(self, "preview_image_status_frame", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    canvas = getattr(self, "preview_canvas", None)
    if frame is None or canvas_frame is None or canvas is None:
        return
    if _preview_fullscreen_overlay_transition_blocked(self):
        self._hide_preview_image_status_overlay()
        return
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        self._hide_preview_image_status_overlay()
        return
    if getattr(canvas, "original_image", None) is None or self._get_preview_annotation() is None:
        self._hide_preview_image_status_overlay()
        return

    try:
        frame_width = int(canvas_frame.winfo_width() or 0)
        frame_height = int(canvas_frame.winfo_height() or 0)
    except Exception:
        frame_width = frame_height = 0
    if frame_width <= 180 or frame_height <= 120:
        self._hide_preview_image_status_overlay()
        return

    dock = getattr(self, "preview_overlay_dock", None)
    dock_width = 126
    dock_height = 0
    dock_x = None
    dock_y = None
    if dock is not None and str(dock.winfo_manager()):
        try:
            dock_width = int(dock.winfo_width() or dock.winfo_reqwidth() or dock_width)
            dock_height = int(dock.winfo_height() or dock.winfo_reqheight() or 0)
            dock_x = int(dock.winfo_x())
            dock_y = int(dock.winfo_y())
        except Exception:
            dock_width = 126
            dock_height = 0
            dock_x = None
            dock_y = None

    width, height = self._render_preview_image_status_overlay(force_render=force_render)
    width = max(96, min(frame_width - 16, int(dock_width or width)))
    if dock_x is None:
        x = max(8, frame_width - width - 10)
    else:
        x = max(8, min(int(dock_x), max(8, frame_width - width - 8)))
    if dock_y is None:
        y = 46 + max(0, int(dock_height or 0)) - 1
    else:
        y = int(dock_y) + max(0, int(dock_height or 0)) - 1
    y = max(8, y)
    if y + height > frame_height - 8:
        available_height = max(0, frame_height - int(y) - 8)
        if available_height < 24:
            self._hide_preview_image_status_overlay()
            return
        height = min(height, available_height)
    try:
        frame.place(
            in_=canvas_frame,
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            anchor="nw",
        )
        frame.lift()
    except Exception:
        pass
    _raise_preview_controls_legend_overlay(self)

def _hide_preview_campaign_gate_overlay(self) -> None:
    self._preview_campaign_gate_overlay_render_key = None
    frame = getattr(self, "preview_campaign_gate_frame", None)
    if frame is not None:
        try:
            frame.place_forget()
        except Exception:
            pass

def _render_preview_campaign_gate_overlay(self, state: dict, *, force_render: bool = False) -> tuple[int, int]:
    return z2_render_preview_campaign_gate_overlay(self, state, force_render=force_render)

def _place_preview_campaign_gate_overlay(self, *, force_render: bool = False) -> None:
    frame = getattr(self, "preview_campaign_gate_frame", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    canvas = getattr(self, "preview_canvas", None)
    if frame is None or canvas_frame is None or canvas is None:
        return
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        # In fullscreen the gate is rendered inside the Z2 drawer, so it must
        # not appear as a second floating panel.
        self._hide_preview_campaign_gate_overlay()
        return
    if _preview_fullscreen_overlay_transition_blocked(self):
        self._hide_preview_campaign_gate_overlay()
        return
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        self._hide_preview_campaign_gate_overlay()
        return
    if getattr(canvas, "original_image", None) is None:
        self._hide_preview_campaign_gate_overlay()
        return

    state = self._build_campaign_z2_gate_overlay_state()
    self._campaign_step2_gate_overlay_state = dict(state)
    if not bool(state.get("visible")):
        self._hide_preview_campaign_gate_overlay()
        return

    try:
        frame_width = int(canvas_frame.winfo_width() or 0)
        frame_height = int(canvas_frame.winfo_height() or 0)
    except Exception:
        frame_width = frame_height = 0
    if frame_width <= 260 or frame_height <= 120:
        self._hide_preview_campaign_gate_overlay()
        return

    dock = getattr(self, "preview_overlay_dock", None)
    status_frame = getattr(self, "preview_image_status_frame", None)
    if status_frame is None or not str(status_frame.winfo_manager()):
        self._place_preview_image_status_overlay(force_render=force_render)
    anchor_widget = status_frame if status_frame is not None and str(status_frame.winfo_manager()) else dock
    dock_width = 126
    dock_height = 0
    dock_x = None
    dock_y = None
    if anchor_widget is not None and str(anchor_widget.winfo_manager()):
        try:
            dock_width = int(anchor_widget.winfo_width() or anchor_widget.winfo_reqwidth() or dock_width)
            dock_height = int(anchor_widget.winfo_height() or anchor_widget.winfo_reqheight() or 0)
            dock_x = int(anchor_widget.winfo_x())
            dock_y = int(anchor_widget.winfo_y())
        except Exception:
            dock_width = 126
            dock_height = 0
            dock_x = None
            dock_y = None
    width, height = self._render_preview_campaign_gate_overlay(state, force_render=force_render)
    width = max(96, min(frame_width - 16, int(dock_width or width)))
    if dock_x is None:
        x = max(8, frame_width - width - 10)
    else:
        x = max(8, min(int(dock_x), max(8, frame_width - width - 8)))
    if dock_y is None:
        y = 46 + max(0, int(dock_height or 0)) - 1
    else:
        y = int(dock_y) + max(0, int(dock_height or 0)) - 1
    y = max(8, y)
    if y + height > frame_height - 8:
        available_height = max(0, frame_height - int(y) - 8)
        if available_height < 72:
            self._hide_preview_campaign_gate_overlay()
            return
        height = min(height, available_height)
    try:
        frame.place(
            in_=canvas_frame,
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            anchor="nw",
        )
        frame.lift()
    except Exception:
        pass
    _raise_preview_controls_legend_overlay(self)

def _toggle_preview_overlay_dock(self, event=None):
    slide = getattr(self, "_preview_drawer_slide", None)
    if slide is not None:
        slide.toggle()
    return "break"

def _toggle_preview_overlay_dock_tool(self, tool_key: str):
    key = str(tool_key or "").strip().lower()
    if key == "legend":
        before = bool(getattr(self, "_preview_controls_legend_visible", True))
        self._preview_controls_legend_visible = not bool(
            getattr(self, "_preview_controls_legend_visible", True)
        )
        try:
            _COMPASS_LOG.info(
                "[Z2 COMPASS] dock legend visible %s->%s",
                int(before),
                int(bool(getattr(self, "_preview_controls_legend_visible", True))),
            )
        except Exception:
            pass
        if bool(getattr(self, "_preview_controls_legend_visible", True)):
            self._place_preview_legend_overlay(refresh=True)
        else:
            self._hide_preview_controls_legend_overlay()
    elif key == "metrics":
        next_visible = not bool(getattr(self, "_preview_metrics_overlay_visible", True))
        self._preview_metrics_overlay_visible = next_visible
        if next_visible:
            self._preview_metrics_overlay_user_moved = False
            self._preview_metrics_overlay_offset_x = 12.0
            self._preview_metrics_overlay_offset_y = 12.0
        self._preview_metrics_overlay_render_key = None
        self._update_preview_canvas_metrics_overlay(force_render=True)
    elif key == "super":
        self._preview_super_correction_badge_visible = False
        result = self._toggle_preview_super_correction()
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass
        return result
    self._preview_overlay_dock_render_key = None
    self._preview_overlay_dock_size = None
    self._preview_overlay_dock_size_key = None
    self._preview_overlay_dock_pre_gate_key = None
    self._preview_overlay_dock_gate_render_key = None
    self._preview_overlay_dock_inline_gate_state = None
    self._preview_overlay_dock_inline_gate_source_key = None
    self._place_preview_overlay_dock(force_render=True)
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    return "break"

def _get_preview_overlay_dock_theme(self) -> dict:
    return z2_get_preview_overlay_dock_theme(self)

def _get_preview_overlay_dock_tools_state(self) -> dict[str, tuple[bool, str]]:
    legend_visible = bool(getattr(self, "_preview_controls_legend_visible", True))
    metrics_visible = bool(getattr(self, "_preview_metrics_overlay_visible", True))
    super_active = bool(getattr(self, "_preview_super_correction_active", False))
    return {
        "legend": (legend_visible, "ON" if legend_visible else "OFF"),
        "metrics": (metrics_visible, "ON" if metrics_visible else "OFF"),
        "super": (super_active, "ON" if super_active else "OFF"),
    }

def _render_preview_overlay_dock(self, *, force_render: bool = False) -> tuple[int, int]:
    return z2_render_preview_overlay_dock(self, force_render=force_render)

def _place_preview_overlay_dock(self, *, force_render: bool = False) -> None:
    dock = getattr(self, "preview_overlay_dock", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    canvas = getattr(self, "preview_canvas", None)
    if dock is None or canvas_frame is None or canvas is None:
        return
    if _preview_fullscreen_overlay_transition_blocked(self):
        suspend_drawer(self)
        try:
            dock.place_forget()
        except Exception:
            pass
        self._hide_preview_image_status_overlay()
        self._hide_preview_campaign_gate_overlay()
        return
    if getattr(canvas, "original_image", None) is None:
        suspend_drawer(self)
        try:
            dock.place_forget()
        except Exception:
            pass
        self._hide_preview_image_status_overlay()
        return

    try:
        frame_width = int(canvas_frame.winfo_width() or 0)
        frame_height = int(canvas_frame.winfo_height() or 0)
    except Exception:
        frame_width = frame_height = 0
    if frame_width <= 180 or frame_height <= 120:
        suspend_drawer(self)
        try:
            dock.place_forget()
        except Exception:
            pass
        self._hide_preview_image_status_overlay()
        return

    dock_width, dock_height = self._render_preview_overlay_dock(force_render=force_render)
    x = max(8, frame_width - dock_width - 10)
    y = 46 if bool(getattr(self, "_preview_fullscreen_active", False)) else 52
    y = min(max(8, y), max(8, frame_height - dock_height - 10))
    try:
        place_drawer(self, frame_width, int(x), int(y), int(dock_width), int(dock_height))
    except Exception:
        pass
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        self._hide_preview_image_status_overlay()
        self._hide_preview_campaign_gate_overlay()
    else:
        self._place_preview_image_status_overlay(force_render=force_render)
    _raise_preview_controls_legend_overlay(self)

def _should_freeze_preview_legend_updates(self) -> bool:
    return bool(
        getattr(self, "_preview_draw_mode", False)
        or bool(getattr(self, "_preview_draw_points", []))
        or bool(getattr(self, "_preview_corner_drag_modifier_down", False))
        or isinstance(getattr(self, "_preview_drag_state", None), dict)
    )

def _on_preview_controls_legend_configure(self, event=None):
    if bool(getattr(self, "_preview_controls_legend_animating", False)):
        return None
    if self._should_freeze_preview_legend_updates():
        return None
    self._refresh_preview_controls_legend()
    return None

def _build_preview_legend_sections(self):
    return z2_build_preview_legend_sections(self)

def _refresh_preview_controls_legend(self):
    return z2_refresh_preview_controls_legend(self)

def _build_preview_canvas_metrics_rows(
    self,
    canvas: ZoomableCanvas,
    ann,
    selected_plate_idx: int | None,
    plate_detections: list[Detection],
) -> list[tuple[str, str, str]]:
    image_metadata = self._get_preview_image_file_metadata(ann, canvas=canvas)
    try:
        image_width = int(image_metadata.get("width", 0) or 0)
        image_height = int(image_metadata.get("height", 0) or 0)
    except Exception:
        image_width = 0
        image_height = 0
    try:
        if image_width <= 0:
            image_width = int(getattr(canvas.original_image, "width", 0) or 0)
        if image_height <= 0:
            image_height = int(getattr(canvas.original_image, "height", 0) or 0)
    except Exception:
        pass
    if image_width <= 0 or image_height <= 0:
        return []

    dpi_text = str(image_metadata.get("dpi_text", "") or self._get_preview_image_dpi_text(ann))
    approved_now = bool(self._preview_annotation_is_explicitly_approved(ann))
    rows: list[tuple[str, str, str]] = [
        (
            "Status",
            "Zatwierdzone [OK] | Spacja cofa"
            if approved_now
            else "Niezatwierdzone | Spacja zatwierdza",
            "success" if approved_now else "error",
        ),
        ("Obraz", f"{image_width}x{image_height} px", "default"),
        ("DPI", str(dpi_text).replace("DPI:", "").strip() or "brak w pliku", "muted"),
    ]

    plates = list(plate_detections or [])
    if not plates:
        if self._preview_draw_mode and self._preview_draw_points:
            rows.append(("Tablica", f"nowa, punkty {len(self._preview_draw_points)}/4", "warning"))
        else:
            rows.append(("Tablica", "brak ramki, D dodaje polygon", "warning"))
        return rows

    safe_idx = selected_plate_idx
    if safe_idx is None:
        safe_idx = 0 if len(plates) == 1 else None
    if safe_idx is None:
        rows.append(("Tablice", f"{len(plates)}; wybierz ramkę, aby zobaczyć udział pola", "warning"))
        return rows

    try:
        safe_idx = max(0, min(int(safe_idx), len(plates) - 1))
    except Exception:
        safe_idx = 0
    det = plates[safe_idx]
    polygon = self._detection_polygon(det)
    try:
        bbox = tuple(float(v) for v in (getattr(det, "bbox", None) or ())[:4])
        if len(bbox) < 4:
            bbox = self._bbox_from_polygon(polygon)
    except Exception:
        bbox = self._bbox_from_polygon(polygon)

    x1, y1, x2, y2 = bbox[:4]
    bbox_width = max(0.0, float(x2) - float(x1))
    bbox_height = max(0.0, float(y2) - float(y1))
    bbox_area = bbox_width * bbox_height
    image_area = max(1.0, float(image_width) * float(image_height))
    bbox_pct = (bbox_area / image_area) * 100.0
    polygon_area = self._polygon_area(polygon)
    polygon_pct = (polygon_area / image_area) * 100.0 if polygon_area > 0.0 else 0.0
    polygon_to_bbox_ratio = (polygon_area / bbox_area) if bbox_area > 0.0 and polygon_area > 0.0 else 0.0

    rows.append(("Tablica", f"{safe_idx + 1}/{len(plates)} | {bbox_width:.0f}x{bbox_height:.0f} px", "default"))
    rows.append(("Obrys detekcji", f"{bbox_area:.0f} px | {bbox_pct:.3f}% obrazu", "success"))
    if polygon_area > 0.0:
        rows.append(("Pow. tablicy", f"{polygon_area:.0f} px | {polygon_pct:.3f}% obrazu", "success"))
        ratio_tone = "success" if polygon_to_bbox_ratio >= 0.45 else "warning"
        rows.append(("Poligon / detekcja", f"{polygon_to_bbox_ratio * 100.0:.1f}% ramki detekcji", ratio_tone))

    if bbox_pct < 0.05:
        rows.append(("Skala", "poniżej 0.05% obrazu, zwykle za mała", "warning"))
    elif bbox_pct < 0.15:
        rows.append(("Skala", "poniżej 0.15% obrazu, sprawdź czytelność", "warning"))
    else:
        rows.append(("Skala", "powyżej progu 0.15% obrazu", "muted"))
    return rows

def _toggle_preview_metrics_overlay(self, event=None):
    self._preview_metrics_overlay_expanded = not bool(
        getattr(self, "_preview_metrics_overlay_expanded", True)
    )
    self._preview_metrics_overlay_render_key = None
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    self._update_preview_canvas_metrics_overlay(force_render=True)
    return "break"

def _on_preview_metrics_overlay_press(self, event=None):
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        root_x = root_y = 0.0
    self._preview_metrics_overlay_drag_state = {
        "root_x": root_x,
        "root_y": root_y,
        "start_x": float(getattr(self, "_preview_metrics_overlay_offset_x", 12.0) or 12.0),
        "start_y": float(getattr(self, "_preview_metrics_overlay_offset_y", 12.0) or 12.0),
    }
    return "break"

def _on_preview_metrics_overlay_drag(self, event=None):
    drag_state = getattr(self, "_preview_metrics_overlay_drag_state", None)
    if not isinstance(drag_state, dict):
        return "break"
    try:
        root_x = float(getattr(event, "x_root", 0.0) or 0.0)
        root_y = float(getattr(event, "y_root", 0.0) or 0.0)
    except Exception:
        root_x = float(drag_state.get("root_x", 0.0) or 0.0)
        root_y = float(drag_state.get("root_y", 0.0) or 0.0)
    next_x = float(drag_state.get("start_x", 12.0) or 12.0) + (root_x - float(drag_state.get("root_x", root_x) or root_x))
    next_y = float(drag_state.get("start_y", 12.0) or 12.0) + (root_y - float(drag_state.get("root_y", root_y) or root_y))
    frame = getattr(self, "preview_metrics_frame", None)
    width = float(getattr(self, "_preview_metrics_overlay_current_width", 0.0) or 0.0)
    height = float(getattr(self, "_preview_metrics_overlay_current_height", 0.0) or 0.0)
    if frame is not None and (width <= 0.0 or height <= 0.0):
        try:
            width = float(frame.winfo_width() or frame.winfo_reqwidth() or 84.0)
            height = float(frame.winfo_height() or frame.winfo_reqheight() or 34.0)
        except Exception:
            width = 84.0
            height = 34.0
    next_x, next_y = self._clamp_preview_metrics_overlay_offsets(next_x, next_y, width=width, height=height)
    self._preview_metrics_overlay_offset_x = float(next_x)
    self._preview_metrics_overlay_offset_y = float(next_y)
    self._preview_metrics_overlay_user_moved = True
    if frame is not None:
        try:
            frame.place_configure(x=int(round(next_x)), y=int(round(next_y)))
            frame.lift()
        except Exception:
            pass
    return "break"

def _on_preview_metrics_overlay_release(self, event=None):
    self._preview_metrics_overlay_drag_state = None
    return "break"

def _clamp_preview_metrics_overlay_offsets(self, x: float, y: float, *, width: float | None = None, height: float | None = None) -> tuple[float, float]:
    canvas_frame = getattr(self, "canvas_frame", None)
    frame = getattr(self, "preview_metrics_frame", None)
    try:
        frame_width = float(canvas_frame.winfo_width() or 0.0) if canvas_frame is not None else 0.0
        frame_height = float(canvas_frame.winfo_height() or 0.0) if canvas_frame is not None else 0.0
    except Exception:
        frame_width = frame_height = 0.0
    try:
        overlay_width = float(width or (frame.winfo_reqwidth() if frame is not None else 0.0) or 84.0)
        overlay_height = float(height or (frame.winfo_reqheight() if frame is not None else 0.0) or 34.0)
    except Exception:
        overlay_width = 84.0
        overlay_height = 34.0
    max_x = max(6.0, frame_width - overlay_width - 6.0)
    max_y = max(6.0, frame_height - overlay_height - 6.0)
    return (
        max(6.0, min(float(x), max_x)),
        max(6.0, min(float(y), max_y)),
    )

def _render_preview_metrics_grab_handle(self, grab_widget, colors: dict) -> None:
    return z2_render_preview_metrics_grab_handle(grab_widget, colors)

def _render_preview_metrics_table(self, rows: list[tuple[str, str, str]], colors: dict) -> None:
    body = getattr(self, "preview_metrics_body", None)
    return z2_render_preview_metrics_table(body, rows, colors)

def _update_preview_canvas_metrics_overlay(self, *, force_render: bool = False) -> None:
    frame = getattr(self, "preview_metrics_frame", None)
    header = getattr(self, "preview_metrics_header", None)
    icon = getattr(self, "preview_metrics_icon_lbl", None)
    title = getattr(self, "preview_metrics_title_lbl", None)
    toggle = getattr(self, "preview_metrics_toggle_lbl", None)
    grab = getattr(self, "preview_metrics_grab_lbl", None)
    body = getattr(self, "preview_metrics_body", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    canvas = getattr(self, "preview_canvas", None)
    if frame is None or canvas_frame is None or canvas is None:
        return
    if not bool(getattr(self, "_preview_metrics_overlay_visible", True)):
        try:
            frame.place_forget()
        except Exception:
            pass
        return

    ann = self._get_preview_annotation()
    if ann is None or getattr(canvas, "original_image", None) is None:
        try:
            frame.place_forget()
        except Exception:
            pass
        return

    try:
        frame_width = int(canvas_frame.winfo_width() or 0)
        frame_height = int(canvas_frame.winfo_height() or 0)
    except Exception:
        frame_width = 0
        frame_height = 0
    if frame_width <= 180 or frame_height <= 120:
        try:
            frame.place_forget()
        except Exception:
            pass
        return

    plate_detections = self._get_plate_detections(ann)
    selected_plate_idx = self._get_selected_plate_index_for_ann(ann)
    try:
        rows = self._build_preview_canvas_metrics_rows(
            canvas,
            ann,
            selected_plate_idx,
            plate_detections,
        )
    except Exception as exc:
        logger.debug(f"Nie udało się zbudować metryk canvasa Z2: {exc}")
        rows = []
    if not rows:
        try:
            frame.place_forget()
        except Exception:
            pass
        return

    try:
        legend_theme = self._get_preview_legend_theme()
    except Exception:
        legend_theme = {}
    palette = getattr(self.app, "palette", {}) if getattr(self, "app", None) is not None else {}
    fill = str(legend_theme.get("panel_fill", palette.get("panel", "#101419")))
    outline = str(legend_theme.get("panel_outline", palette.get("panel_border", "#4b5563")))
    text_fill = str(legend_theme.get("entry_text", palette.get("fg", "#f8fafc")))
    muted = str(legend_theme.get("section_muted", palette.get("muted", "#c7c7c7")))
    accent = str(legend_theme.get("badge_plate_outline", palette.get("accent", "#f1c40f")))
    warning = str(palette.get("warning", "#f39c12"))
    success = str(palette.get("success", "#4ec9b0"))
    error = str(palette.get("error", "#e74c3c"))
    row_fill = blend_hex_colors(fill, outline, 0.10)
    row_alt = blend_hex_colors(fill, outline, 0.05)
    icon_bg = blend_hex_colors(accent, fill, 0.34)
    grab_fill = str(legend_theme.get("entry_fill", palette.get("field", "#3a3f46")))

    expanded = bool(getattr(self, "_preview_metrics_overlay_expanded", True))
    image_key = (
        int(getattr(self, "current_preview_index", -1) if getattr(self, "current_preview_index", None) is not None else -1),
        str(getattr(ann, "filename", "") or "").strip().lower(),
    )
    render_key = (
        int(expanded),
        image_key,
        tuple((str(label), str(value), str(tone)) for label, value, tone in rows),
        fill,
        outline,
        text_fill,
        muted,
        accent,
        warning,
        success,
        error,
        grab_fill,
    )
    needs_render = bool(force_render) or render_key != getattr(self, "_preview_metrics_overlay_render_key", None)
    if needs_render:
        try:
            frame.configure(bg=fill, highlightbackground=outline, highlightcolor=outline)
            if header is not None:
                header.configure(bg=fill)
            if icon is not None:
                icon.configure(bg=icon_bg, fg=text_fill)
            if title is not None:
                title.configure(bg=fill, fg=text_fill)
            if toggle is not None:
                toggle.configure(text=("-" if expanded else "+"), bg=fill, fg=accent)
            if body is not None:
                body.configure(bg=fill)
            self._render_preview_metrics_grab_handle(
                grab,
                {
                    "fill": fill,
                    "grab_fill": grab_fill,
                    "outline": outline,
                    "muted": muted,
                },
            )
        except Exception:
            pass

        try:
            if title is not None:
                if expanded and not str(title.winfo_manager()):
                    title.pack(side=tk.LEFT, fill=tk.X, expand=True, before=toggle)
                elif (not expanded) and str(title.winfo_manager()):
                    title.pack_forget()
            if body is not None:
                if expanded:
                    if not str(body.winfo_manager()):
                        body.pack(fill=tk.BOTH, expand=True)
                    self._render_preview_metrics_table(
                        rows,
                        {
                            "fill": fill,
                            "row_fill": row_fill,
                            "row_alt": row_alt,
                            "outline": outline,
                            "text": text_fill,
                            "muted": muted,
                            "success": success,
                            "warning": warning,
                            "error": error,
                        },
                    )
                elif str(body.winfo_manager()):
                    body.pack_forget()
        except Exception:
            pass
        self._preview_metrics_overlay_render_key = render_key

    max_width = max(78, min(frame_width - 24, 420 if expanded else 96))
    if needs_render or float(getattr(self, "_preview_metrics_overlay_current_width", 0.0) or 0.0) <= 0.0:
        try:
            frame.update_idletasks()
            req_width = min(max_width, max(72 if not expanded else 300, int(frame.winfo_reqwidth() or 120)))
            req_height = min(frame_height - 24, max(30, int(frame.winfo_reqheight() or 34)))
        except Exception:
            req_width = 360 if expanded else 86
            req_height = 126 if expanded else 34
    else:
        try:
            req_width = min(max_width, max(72 if not expanded else 300, int(getattr(self, "_preview_metrics_overlay_current_width", 0.0) or frame.winfo_width() or frame.winfo_reqwidth() or 120)))
            req_height = min(frame_height - 24, max(30, int(getattr(self, "_preview_metrics_overlay_current_height", 0.0) or frame.winfo_height() or frame.winfo_reqheight() or 34)))
        except Exception:
            req_width = 360 if expanded else 86
            req_height = 126 if expanded else 34

    x = float(getattr(self, "_preview_metrics_overlay_offset_x", 12.0) or 12.0)
    y = float(getattr(self, "_preview_metrics_overlay_offset_y", 12.0) or 12.0)
    if not bool(getattr(self, "_preview_metrics_overlay_user_moved", False)):
        try:
            legend_frame = getattr(self, "preview_hint_frame", None)
            legend_mapped = bool(legend_frame is not None and str(legend_frame.winfo_manager()))
        except Exception:
            legend_mapped = False
        if legend_mapped:
            try:
                legend_x = float(getattr(self, "_preview_controls_legend_offset_x", 10.0) or 10.0)
                legend_y = float(getattr(self, "_preview_controls_legend_offset_y", 10.0) or 10.0)
                legend_w = float(legend_frame.winfo_width() or getattr(self, "_preview_controls_legend_current_width", 0.0) or 0.0)
                legend_h = float(legend_frame.winfo_height() or getattr(self, "_preview_controls_legend_current_height", 0.0) or 0.0)
                if legend_w <= 0.0:
                    legend_w = float(self._get_preview_controls_legend_target_width())
                if legend_h <= 0.0:
                    legend_h = float(self._get_preview_controls_legend_target_height(legend_w))
                overlaps_x = x < (legend_x + legend_w + 8.0) and (x + req_width) > (legend_x - 8.0)
                overlaps_y = y < (legend_y + legend_h + 8.0) and (y + req_height) > (legend_y - 8.0)
                if overlaps_x and overlaps_y:
                    candidate_y = legend_y + legend_h + 10.0
                    bottom_reserved = 72.0 if bool(getattr(self, "_preview_fullscreen_active", False)) else 14.0
                    if candidate_y + req_height <= frame_height - bottom_reserved:
                        y = candidate_y
                    elif frame_width >= (legend_w + req_width + 36.0):
                        x = max(12.0, frame_width - req_width - 12.0)
                    else:
                        y = max(12.0, frame_height - req_height - bottom_reserved)
            except Exception:
                pass

        try:
            dock = getattr(self, "preview_overlay_dock", None)
            dock_mapped = bool(dock is not None and str(dock.winfo_manager()))
        except Exception:
            dock_mapped = False
        if dock_mapped:
            try:
                dock_x = float(dock.winfo_x() or 0.0)
                dock_y = float(dock.winfo_y() or 0.0)
                dock_w = float(dock.winfo_width() or dock.winfo_reqwidth() or 0.0)
                dock_h = float(dock.winfo_height() or dock.winfo_reqheight() or 0.0)
                overlaps_x = x < (dock_x + dock_w + 8.0) and (x + req_width) > (dock_x - 8.0)
                overlaps_y = y < (dock_y + dock_h + 8.0) and (y + req_height) > (dock_y - 8.0)
                if overlaps_x and overlaps_y:
                    left_candidate = dock_x - req_width - 10.0
                    below_candidate = dock_y + dock_h + 10.0
                    if left_candidate >= 8.0:
                        x = left_candidate
                    elif below_candidate + req_height <= frame_height - 10.0:
                        y = below_candidate
                    else:
                        x = 12.0
                        y = max(12.0, min(y, frame_height - req_height - 10.0))
            except Exception:
                pass

    x, y = self._clamp_preview_metrics_overlay_offsets(x, y, width=req_width, height=req_height)
    self._preview_metrics_overlay_offset_x = float(x)
    self._preview_metrics_overlay_offset_y = float(y)
    try:
        frame.place(
            in_=canvas_frame,
            x=int(round(x)),
            y=int(round(y)),
            width=int(round(req_width)),
            height=int(round(req_height)),
            anchor="nw",
        )
        frame.lift()
        self._preview_metrics_overlay_current_width = float(req_width)
        self._preview_metrics_overlay_current_height = float(req_height)
    except Exception:
        pass
    _raise_preview_controls_legend_overlay(self)
