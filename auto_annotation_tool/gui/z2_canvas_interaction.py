from __future__ import annotations

"""Z2 canvas interaction helpers extracted from tab_annotation.py."""

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
import logging
import math
import shutil
import time
import queue
import numpy as np

import cv2
from PIL import Image
from .inertial_scroll import InertialScrollController
from .zoomable_canvas import ZoomableCanvas
from .section_header_label import SectionHeaderLabel
from . import z2_workflow_methods
from . import z2_canvas_overlays
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

def _preview_super_perf_active(self) -> bool:
    try:
        return time.perf_counter() <= float(getattr(self, "_preview_super_perf_until", 0.0) or 0.0)
    except Exception:
        return False


def _log_preview_super_perf(self, operation: str, elapsed_ms: float, *, threshold_ms: float = 80.0, **details) -> None:
    if not _preview_super_perf_active(self) and float(elapsed_ms) < float(threshold_ms):
        return
    try:
        suffix = " ".join(f"{key}={value}" for key, value in details.items())
        if suffix:
            suffix = " " + suffix
        logger.info(
            "[Z2 DRAG PERF] %s %.1fms label=%s%s",
            operation,
            float(elapsed_ms),
            str(getattr(self, "_preview_super_perf_label", "") or "-"),
            suffix,
        )
    except Exception:
        pass


def _get_preview_focus_image_key(self, ann=None) -> str:
    target_ann = self._get_preview_annotation() if ann is None else ann
    return str(getattr(target_ann, "filename", "") or "")

def _preview_focus_restore_matches_current_image(self) -> bool:
    state = getattr(self, "_preview_polygon_focus_restore_state", None)
    if not isinstance(state, dict):
        return False
    return str(state.get("filename", "") or "") == self._get_preview_focus_image_key()

def _capture_preview_view_state(self):
    if getattr(self.preview_canvas, "original_image", None) is None:
        return None
    state = dict(self.preview_canvas.get_view_state())
    state["filename"] = self._get_preview_focus_image_key()
    return state

def _compute_preview_bbox_focus_view_state(
    self,
    bbox: tuple[float, float, float, float],
    *,
    pad_x_ratio: float = 0.20,
    pad_y_ratio: float = 0.25,
    min_padding: float = 16.0,
):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None or canvas.original_image is None or not bbox:
        return None

    if not bool(getattr(self, "_preview_fullscreen_transition_active", False)):
        try:
            canvas.update_idletasks()
        except Exception:
            pass

    canvas_width = max(1.0, float(canvas.winfo_width()))
    canvas_height = max(1.0, float(canvas.winfo_height()))
    if canvas_width <= 1.0 or canvas_height <= 1.0:
        return None

    min_x, min_y, max_x, max_y = [float(v) for v in bbox]
    box_width = max(1.0, float(max_x) - float(min_x))
    box_height = max(1.0, float(max_y) - float(min_y))
    padding_x = max(float(min_padding), box_width * float(pad_x_ratio))
    padding_y = max(float(min_padding), box_height * float(pad_y_ratio))

    image_width = max(1.0, float(canvas.original_image.width))
    image_height = max(1.0, float(canvas.original_image.height))
    box_min_x = max(0.0, float(min_x) - padding_x)
    box_min_y = max(0.0, float(min_y) - padding_y)
    box_max_x = min(image_width - 1.0, float(max_x) + padding_x)
    box_max_y = min(image_height - 1.0, float(max_y) + padding_y)
    box_width = max(1.0, box_max_x - box_min_x)
    box_height = max(1.0, box_max_y - box_min_y)

    zoom_level = min(
        canvas_width / box_width,
        canvas_height / box_height,
        float(getattr(canvas, "max_zoom", 5.0) or 5.0),
    )
    zoom_level = max(float(getattr(canvas, "min_zoom", 0.1) or 0.1), float(zoom_level))

    center_x = (box_min_x + box_max_x) / 2.0
    center_y = (box_min_y + box_max_y) / 2.0
    origin_x = (canvas_width / 2.0) - (center_x * zoom_level)
    origin_y = (canvas_height / 2.0) - (center_y * zoom_level)

    return {
        "zoom_level": float(zoom_level),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
    }

def _compute_preview_plate_focus_view_state(self, polygon: list[tuple[float, float]]):
    if not polygon:
        return None
    return self._compute_preview_bbox_focus_view_state(
        self._bbox_from_polygon(polygon),
        pad_x_ratio=0.20,
        pad_y_ratio=0.25,
        min_padding=16.0,
    )

def _refresh_preview_plate_context_overlays(self) -> None:
    """Refresh lightweight fullscreen overlays after changing image/plate context."""
    try:
        self._refresh_preview_legend_backdrop()
    except Exception:
        pass
    try:
        self._preview_metrics_overlay_render_key = None
        self._update_preview_canvas_metrics_overlay(force_render=True)
    except Exception:
        pass
    try:
        self._place_preview_overlay_dock()
    except Exception:
        pass

def _focus_preview_plate(
    self,
    plate_idx: int | None = None,
    *,
    store_restore: bool = True,
    push_debug: bool = True,
    status_message: str | None = None,
) -> bool:
    focus_started = time.perf_counter()
    phase_at = focus_started
    phase_ms: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal phase_at
        now = time.perf_counter()
        phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
        phase_at = now

    ann = self._get_preview_annotation()
    canvas = getattr(self, "preview_canvas", None)
    if ann is None or canvas is None or canvas.original_image is None:
        return False
    try:
        self._cancel_preview_layout_restore_jobs()
    except Exception:
        pass

    plate_detections = self._get_plate_detections(ann)
    if not plate_detections:
        return False
    mark_phase("data")

    if not self._preview_focus_restore_matches_current_image():
        self._preview_polygon_focus_restore_state = None

    if plate_idx is None:
        plate_idx = self._get_selected_plate_index_for_ann(ann)
    if plate_idx is None:
        plate_idx = 0

    safe_idx = self._set_selected_plate_index_for_ann(ann, int(plate_idx))
    if safe_idx is None:
        return False

    if store_restore and self._preview_polygon_focus_restore_state is None:
        self._preview_polygon_focus_restore_state = self._capture_preview_view_state()

    polygon = self._detection_polygon(plate_detections[int(safe_idx)])
    view_state = self._compute_preview_plate_focus_view_state(polygon)
    if not isinstance(view_state, dict):
        return False
    mark_phase("view_state")

    super_mode = bool(getattr(self, "_preview_super_correction_active", False))
    if super_mode:
        try:
            canvas._cancel_zoom_animation()
            canvas._cancel_deferred_display()
            canvas._cancel_final_quality_display()
        except Exception:
            pass
        mark_phase("cancel")
        if not canvas.set_view_state(view_state, redraw=False):
            return False
        mark_phase("set_view")
        try:
            canvas._update_display(interaction_fast=True)
            canvas._schedule_final_quality_display(delay_ms=900)
        except Exception:
            canvas.refresh_overlay_only(skip_info=True)
        mark_phase("fast_render")
    else:
        try:
            canvas._cancel_zoom_animation()
            canvas._cancel_deferred_display()
        except Exception:
            pass
        mark_phase("cancel")
        if not canvas.set_view_state(view_state, redraw=False):
            return False
        mark_phase("set_view")
        try:
            canvas._update_display(interaction_fast=True)
            canvas._schedule_final_quality_display(delay_ms=500)
        except Exception:
            canvas.refresh_overlay_only(skip_info=True)
        mark_phase("fast_render")

    self._set_preview_focus_target("plate", int(safe_idx))
    self._refresh_preview_plate_context_overlays()
    mark_phase("context")

    if push_debug:
        self._push_preview_debug_event(
            "focus-plate",
            (
                f"p{int(safe_idx) + 1} zoom={float(view_state.get('zoom_level', 0.0)):.3f} "
                f"origin=({float(view_state.get('origin_x', 0.0)):.1f},{float(view_state.get('origin_y', 0.0)):.1f})"
            )
        )
    if status_message:
        self._update_preview_edit_status(
            status_message,
            refresh_toolbar=False,
            refresh_debug=False,
        )
    else:
        self._update_preview_edit_status(
            refresh_toolbar=False,
            refresh_debug=False,
        )
    mark_phase("status")
    try:
        canvas.focus_set()
    except Exception:
        pass
    mark_phase("focus_set")
    if super_mode:
        self._prime_preview_corner_drag_history()
        mark_phase("prime_history")
    elapsed_ms = max(0.0, (time.perf_counter() - focus_started) * 1000.0)
    _log_preview_super_perf(
        self,
        "focus_plate",
        elapsed_ms,
        threshold_ms=0.0 if super_mode and _preview_super_perf_active(self) else 80.0,
        super=int(bool(super_mode)),
        plate=safe_idx,
        data=f"{phase_ms.get('data', 0.0):.1f}",
        view=f"{phase_ms.get('view_state', 0.0):.1f}",
        cancel=f"{phase_ms.get('cancel', 0.0):.1f}",
        set_view=f"{phase_ms.get('set_view', 0.0):.1f}",
        fast_render=f"{phase_ms.get('fast_render', 0.0):.1f}",
        render=f"{phase_ms.get('render', 0.0):.1f}",
        context=f"{phase_ms.get('context', 0.0):.1f}",
        status=f"{phase_ms.get('status', 0.0):.1f}",
        focus_set=f"{phase_ms.get('focus_set', 0.0):.1f}",
        prime=f"{phase_ms.get('prime_history', 0.0):.1f}",
    )
    return True

def _focus_preview_vehicle(
    self,
    vehicle_idx: int | None = None,
    *,
    store_restore: bool = True,
    push_debug: bool = True,
    status_message: str | None = None,
) -> bool:
    ann = self._get_preview_annotation()
    canvas = getattr(self, "preview_canvas", None)
    if ann is None or canvas is None or canvas.original_image is None:
        return False

    vehicle_detections = self._get_vehicle_detections(ann)
    if not vehicle_detections:
        return False

    if not self._preview_focus_restore_matches_current_image():
        self._preview_polygon_focus_restore_state = None

    if vehicle_idx is None:
        vehicle_idx = self._get_selected_vehicle_index_for_ann(ann)
    if vehicle_idx is None:
        vehicle_idx = 0

    safe_idx = self._set_selected_vehicle_index_for_ann(ann, int(vehicle_idx))
    if safe_idx is None:
        return False

    if store_restore and self._preview_polygon_focus_restore_state is None:
        self._preview_polygon_focus_restore_state = self._capture_preview_view_state()

    view_state = self._compute_preview_bbox_focus_view_state(
        tuple(float(v) for v in vehicle_detections[int(safe_idx)].bbox[:4]),
        pad_x_ratio=0.015,
        pad_y_ratio=0.02,
        min_padding=6.0,
    )
    if not isinstance(view_state, dict):
        return False

    if not canvas.set_view_state(view_state, redraw=True):
        return False

    self._set_preview_focus_target("vehicle", int(safe_idx))

    if push_debug:
        self._push_preview_debug_event(
            "focus-vehicle",
            (
                f"v{int(safe_idx) + 1} zoom={float(view_state.get('zoom_level', 0.0)):.3f} "
                f"origin=({float(view_state.get('origin_x', 0.0)):.1f},{float(view_state.get('origin_y', 0.0)):.1f})"
            )
        )
    if status_message:
        self._update_preview_edit_status(status_message)
    else:
        self._update_preview_edit_status()
    try:
        canvas.focus_set()
    except Exception:
        pass
    return True

def _restore_preview_focus_view(self, status_message: str | None = None) -> bool:
    if not self._preview_focus_restore_matches_current_image():
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None
        return False

    restore_state = dict(self._preview_polygon_focus_restore_state or {})
    self._preview_polygon_focus_restore_state = None
    self._preview_focus_target = None
    if not self.preview_canvas.set_view_state(restore_state, redraw=True):
        return False

    self._push_preview_debug_event(
        "focus-restore",
        (
            f"zoom={float(restore_state.get('zoom_level', 0.0)):.3f} "
            f"origin=({float(restore_state.get('origin_x', 0.0)):.1f},{float(restore_state.get('origin_y', 0.0)):.1f})"
        )
    )
    if status_message:
        self._update_preview_edit_status(status_message)
    else:
        self._update_preview_edit_status()
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    return True

def _restore_preview_layout_view_after_resize(self):
    if bool(getattr(self, "_preview_force_fit_after_resize", False)):
        self._preview_force_fit_after_resize = False
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None
        self._fit_preview_image_to_view()
        return

    ann = self._get_preview_annotation()
    focus_target = self._get_preview_focus_target()
    if isinstance(focus_target, dict):
        focus_kind = str(focus_target.get("kind", "") or "")
        focus_index = focus_target.get("index")
        if focus_kind == "vehicle":
            if self._focus_preview_vehicle(focus_index, store_restore=False, push_debug=False, status_message=None):
                self._push_preview_debug_event("focus-layout", "odswiezono fokus pojazdu po zmianie ukladu")
                return
        elif focus_kind == "plate":
            selected_idx = self._get_selected_plate_index_for_ann(ann) if ann is not None else None
            if focus_index is not None:
                selected_idx = int(focus_index)
            if self._focus_preview_plate(selected_idx, store_restore=False, push_debug=False, status_message=None):
                self._push_preview_debug_event("focus-layout", "odswiezono fokus polygonu po zmianie ukladu")
                return
        self._preview_focus_target = None

    if self._preview_focus_restore_matches_current_image():
        focus_target = self._get_preview_focus_target()
        focus_kind = str((focus_target or {}).get("kind", "") or "")
        focus_index = (focus_target or {}).get("index")
        if focus_kind == "vehicle":
            if self._focus_preview_vehicle(focus_index, store_restore=False, push_debug=False, status_message=None):
                self._push_preview_debug_event("focus-layout", "odswiezono fokus pojazdu po zmianie ukladu")
                return
        else:
            selected_idx = self._get_selected_plate_index_for_ann(ann) if ann is not None else None
            if focus_index is not None:
                selected_idx = int(focus_index)
            if self._focus_preview_plate(selected_idx, store_restore=False, push_debug=False, status_message=None):
                self._push_preview_debug_event("focus-layout", "odswiezono fokus polygonu po zmianie ukladu")
                return
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None

    self._fit_preview_image_to_view()

def _cancel_preview_layout_restore_jobs(self):
    pending = list(getattr(self, "_preview_layout_restore_after_ids", []) or [])
    self._preview_layout_restore_after_ids = []
    for after_id in pending:
        try:
            self.frame.after_cancel(after_id)
        except Exception:
            pass

def _schedule_preview_layout_restore_after_resize(self):
    self._cancel_preview_layout_restore_jobs()

    def schedule(delay_ms: int, attempt: int, last_size: tuple[int, int] | None = None, stable_count: int = 0):
        try:
            after_id = self.frame.after(
                int(delay_ms),
                lambda a=attempt, ls=last_size, sc=stable_count: self._stabilize_preview_layout_after_resize(a, ls, sc),
            )
            self._preview_layout_restore_after_ids.append(after_id)
        except Exception:
            pass

    schedule(60, 0, None, 0)

def _stabilize_preview_layout_after_resize(
    self,
    attempt: int = 0,
    last_size: tuple[int, int] | None = None,
    stable_count: int = 0,
):
    self._preview_layout_restore_after_ids = [
        after_id
        for after_id in (getattr(self, "_preview_layout_restore_after_ids", []) or [])
        if after_id
    ]

    try:
        canvas = self.preview_canvas
    except Exception:
        return

    if canvas is None or getattr(canvas, "original_image", None) is None:
        self._cancel_preview_layout_restore_jobs()
        return

    try:
        canvas.update_idletasks()
    except Exception:
        pass

    try:
        current_size = (max(1, int(canvas.winfo_width() or 1)), max(1, int(canvas.winfo_height() or 1)))
    except Exception:
        current_size = (1, 1)

    next_stable = (stable_count + 1) if last_size == current_size else 0
    if attempt >= 8 or next_stable >= 2:
        self._restore_preview_layout_view_after_resize()
        try:
            if bool(getattr(self, "_preview_fullscreen_active", False)):
                setattr(self, "_preview_fullscreen_overlay_ready", True)
                setattr(self, "_preview_fullscreen_transition_active", False)
                setattr(self, "_preview_controls_legend_current_width", 0.0)
                setattr(self, "_preview_controls_legend_current_height", 0.0)
                setattr(self, "_preview_controls_legend_render_key", None)
                self._place_preview_legend_overlay(refresh=True)
                self._place_preview_overlay_dock(force_render=True)
                self._place_preview_campaign_gate_overlay(force_render=True)
        except Exception:
            pass
        self._cancel_preview_layout_restore_jobs()
        return

    try:
        after_id = self.frame.after(
            70,
            lambda a=attempt + 1, ls=current_size, sc=next_stable: self._stabilize_preview_layout_after_resize(a, ls, sc),
        )
        self._preview_layout_restore_after_ids.append(after_id)
    except Exception:
        pass

def _format_preview_debug_vertex_ref(self, plate_idx: int | None, vertex_idx: int | None) -> str:
    if plate_idx is None or int(plate_idx) < 0:
        return "-"
    vertex_part = "-" if vertex_idx is None or int(vertex_idx) < 0 else f"v{int(vertex_idx) + 1}"
    return f"p{int(plate_idx) + 1}:{vertex_part}"

def _format_preview_debug_pending(self) -> str:
    pending = getattr(self, "_preview_pending_vertex_hit", None)
    if not isinstance(pending, dict):
        return "-"
    return (
        f"{self._format_preview_debug_vertex_ref(pending.get('plate_idx'), pending.get('vertex_idx'))} "
        f"anchor=({float(pending.get('anchor_canvas_x', 0.0)):.1f},{float(pending.get('anchor_canvas_y', 0.0)):.1f})"
    )

def _format_preview_debug_drag(self) -> str:
    drag_state = getattr(self, "_preview_drag_state", None)
    if not isinstance(drag_state, dict):
        return "-"
    return (
        f"{self._format_preview_debug_vertex_ref(drag_state.get('plate_idx'), drag_state.get('vertex_idx'))} "
        f"started={int(bool(drag_state.get('drag_started', False)))} "
        f"moved={int(bool(drag_state.get('was_moved', False)))} "
        f"press=({float(drag_state.get('press_canvas_x', 0.0)):.1f},{float(drag_state.get('press_canvas_y', 0.0)):.1f})"
    )

def _build_preview_debug_lines(self, history_limit: int = 4):
    ann = self._get_preview_annotation()
    filename = str(getattr(ann, "filename", "-") or "-")
    selected_plate = self._get_selected_plate_index_for_ann(ann) if ann is not None else None
    if selected_plate is None:
        selected_plate_text = "-"
    else:
        selected_plate_text = str(int(selected_plate) + 1)

    zoom_value = float(getattr(self.preview_canvas, "zoom_level", 0.0) or 0.0)
    try:
        origin_x, origin_y = self.preview_canvas.get_image_origin()
    except Exception:
        origin_x, origin_y = 0.0, 0.0

    history = list(getattr(self, "_preview_debug_events", []) or [])
    history_text = " | ".join(history[:history_limit]) if history else "-"
    return [
        "DEBUG Z2",
        (
            f"img={filename} idx={self.current_preview_index if self.current_preview_index is not None else '-'} "
            f"sel_plate={selected_plate_text} zoom={zoom_value:.3f} origin=({origin_x:.1f},{origin_y:.1f})"
        ),
        (
            f"flags: W={int(bool(self._preview_corner_drag_modifier_down))} "
            f"draw={int(bool(self._preview_draw_mode))} delete={int(bool(self._preview_delete_mode))} "
            f"fullscreen={int(bool(self._preview_fullscreen_active))} "
            f"focus={int(bool(self._preview_focus_restore_matches_current_image()))} "
            f"dirty={len(self._preview_dirty_images)}"
        ),
        f"pending: {self._format_preview_debug_pending()}",
        f"drag: {self._format_preview_debug_drag()}",
        f"history: {history_text}",
    ]

def _refresh_preview_debug_status(self):
    if not bool(getattr(self, "_preview_debug_enabled", False)):
        return
    self.preview_debug_var.set("\n".join(self._build_preview_debug_lines(history_limit=4)))

def _print_preview_debug_console(self, event_name: str, detail: str = ""):
    if not bool(getattr(self, "_preview_debug_enabled", False)):
        return
    summary = f"[DEBUG Z2] event={event_name}"
    if detail:
        summary += f" {detail}"
    output_lines = [summary]
    for line in self._build_preview_debug_lines(history_limit=6)[1:]:
        output_lines.append(f"[DEBUG Z2] {line}")
    if event_name in {"select", "zoom", "press-target", "press-handle", "press-polygon", "press-miss"}:
        output_lines.append(f"[DEBUG Z2] {self._describe_selected_polygon_vertices_debug()}")
    for line in output_lines:
        print(line, flush=True)
    try:
        log_path = Path(getattr(self, "_preview_debug_log_path", Path(CONFIG.WORKSPACE_DIR) / "z2_debug.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(output_lines))
            fh.write("\n")
    except Exception:
        pass

def _append_z2_trace(self, tag: str, detail: str = ""):
    try:
        log_path = Path(getattr(self, "_preview_debug_log_path", Path(CONFIG.WORKSPACE_DIR) / "z2_debug.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[TRACE Z2] {stamp} {str(tag or '').strip()}"
        if detail:
            line += f" | {detail}"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
    except Exception:
        pass

def _push_preview_debug_event(self, event_name: str, detail: str = "", refresh_only: bool = False):
    if not bool(getattr(self, "_preview_debug_enabled", False)):
        return

    if not refresh_only:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        event_text = f"{stamp} {event_name}"
        if detail:
            event_text += f" {detail}"
        self._preview_debug_events.appendleft(event_text)
    self._refresh_preview_debug_status()
    if not refresh_only:
        self._print_preview_debug_console(event_name, detail)

def _widget_is_descendant_of(widget, ancestor) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "master", None)
    return False

def _event_has_control_modifier(event=None) -> bool:
    if event is None:
        return False
    try:
        return bool(int(getattr(event, "state", 0) or 0) & 0x0004)
    except Exception:
        return False

def _preview_shortcuts_enabled(self, event=None, allow_when_fullscreen: bool = False) -> bool:
    if allow_when_fullscreen and bool(getattr(self, "_preview_fullscreen_active", False)):
        return True

    if (
        self._preview_drag_state is not None
        or self._preview_corner_drag_modifier_down
        or self._preview_focus_zoom_modifier_down
        or self._preview_delete_mode
    ):
        return True

    target_widget = getattr(event, "widget", None)
    if target_widget is None:
        try:
            target_widget = self.frame.focus_get()
        except Exception:
            target_widget = None

    if self._widget_is_descendant_of(target_widget, getattr(self, "preview_host", None)):
        return True

    try:
        x_root = int(getattr(event, "x_root", 0) or self.frame.winfo_pointerx())
        y_root = int(getattr(event, "y_root", 0) or self.frame.winfo_pointery())
    except Exception:
        return False

    for widget in (
        getattr(self, "preview_canvas", None),
        getattr(self, "preview_listbox", None),
        getattr(self, "preview_host", None),
    ):
        if self._widget_contains_point(widget, x_root, y_root):
            return True

    return False

def _preview_shortcut_is_duplicate(self, event, action_key: str) -> bool:
    if event is None:
        return False
    try:
        serial = int(getattr(event, "serial", 0) or 0)
    except Exception:
        serial = 0
    try:
        event_time = int(getattr(event, "time", 0) or 0)
    except Exception:
        event_time = 0
    signature = (str(action_key), serial, event_time)
    guard = dict(getattr(self, "_preview_shortcut_event_guard", {}) or {})
    if signature == guard.get(str(action_key)):
        return True
    guard[str(action_key)] = signature
    self._preview_shortcut_event_guard = guard
    return False

def _pane_has_child(pane, child) -> bool:
    try:
        return str(child) in [str(item) for item in pane.panes()]
    except Exception:
        return False

def _sync_left_column_pane_layout(self, *, show_preview: bool, compact_layout: bool):
    left_frame = getattr(self, "main_left_frame", None)
    top_shell = getattr(self, "left_scroll_shell", None)
    bottom_shell = getattr(self, "preview_left_list_shell", None)
    if left_frame is None or top_shell is None:
        return

    free_mode_expand_top = bool(
        self._is_free_mode_session_context()
        and not show_preview
    )

    try:
        left_frame.grid_rowconfigure(0, weight=(1 if free_mode_expand_top else 0))
        left_frame.grid_rowconfigure(1, weight=(1 if show_preview else 0))
        left_frame.grid_rowconfigure(1, minsize=0)
        if not show_preview:
            left_frame.grid_rowconfigure(0, minsize=0)
    except Exception:
        pass

    try:
        top_shell.grid_configure(
            row=0,
            column=0,
            sticky=(
                "nsew"
                if free_mode_expand_top
                else ("ew" if compact_layout and show_preview else ("nsew" if show_preview else "ew"))
            ),
        )
    except Exception:
        pass

    try:
        if show_preview:
            if bottom_shell is not None:
                bottom_shell.grid()
                bottom_shell.grid_configure(
                    row=1,
                    column=0,
                    sticky="nsew",
                    pady=((4, 0) if compact_layout else (8, 0)),
                )
        elif bottom_shell is not None:
            bottom_shell.grid_remove()
    except Exception:
        pass

def _bind_preview_shortcuts(self):
    bindings = (
        ("<Button-2>", self._on_preview_middle_click_zoom_shortcut),
        ("<ButtonRelease-2>", self._on_preview_middle_click_zoom_shortcut),
        ("<Up>", self._on_preview_arrow_up_shortcut),
        ("<Down>", self._on_preview_arrow_down_shortcut),
        ("<KP_Up>", self._on_preview_arrow_up_shortcut),
        ("<KP_Down>", self._on_preview_arrow_down_shortcut),
        ("<KeyPress-w>", self._on_preview_edit_modifier_press),
        ("<KeyPress-W>", self._on_preview_edit_modifier_press),
        ("<KeyRelease-w>", self._on_preview_edit_modifier_release),
        ("<KeyRelease-W>", self._on_preview_edit_modifier_release),
        ("<KeyPress-d>", self._on_preview_draw_toggle_shortcut),
        ("<KeyPress-D>", self._on_preview_draw_toggle_shortcut),
        ("<KeyPress-s>", self._on_preview_delete_mode_shortcut),
        ("<KeyPress-S>", self._on_preview_delete_mode_shortcut),
        ("<KeyPress-r>", self._on_preview_focus_zoom_modifier_press),
        ("<KeyPress-R>", self._on_preview_focus_zoom_modifier_press),
        ("<KeyRelease-r>", self._on_preview_focus_zoom_modifier_release),
        ("<KeyRelease-R>", self._on_preview_focus_zoom_modifier_release),
        ("<KeyPress-f>", self._on_preview_fit_shortcut),
        ("<KeyPress-F>", self._on_preview_fit_shortcut),
        ("<KeyPress-a>", self._on_preview_cycle_plate_shortcut),
        ("<KeyPress-A>", self._on_preview_cycle_plate_shortcut),
        ("<KeyPress-y>", self._on_preview_toggle_super_correction_shortcut),
        ("<KeyPress-Y>", self._on_preview_toggle_super_correction_shortcut),
        ("<KeyPress-space>", self._on_preview_toggle_image_approval_shortcut),
        ("<KeyPress-q>", self._on_preview_prev_shortcut),
        ("<KeyPress-Q>", self._on_preview_prev_shortcut),
        ("<KeyPress-e>", self._on_preview_next_shortcut),
        ("<KeyPress-E>", self._on_preview_next_shortcut),
        ("<Delete>", self._on_preview_delete_image_shortcut),
        ("<Control-s>", self._on_preview_save_shortcut),
        ("<Control-S>", self._on_preview_save_shortcut),
        ("<Control-z>", self._on_preview_undo_shortcut),
        ("<Control-Z>", self._on_preview_undo_shortcut),
        ("<Control-y>", self._on_preview_redo_shortcut),
        ("<Control-Y>", self._on_preview_redo_shortcut),
        ("<Return>", self._on_preview_enter_fullscreen_shortcut),
        ("<Escape>", self._on_preview_escape_shortcut),
    )
    preview_widgets = (
        getattr(self, "preview_canvas", None),
        getattr(self, "preview_listbox", None),
    )
    for sequence, handler in bindings:
        for widget in preview_widgets:
            if widget is not None:
                widget.bind(sequence, handler, add="+")
        self.frame.bind_all(sequence, handler, add="+")

def _on_preview_prev_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-prev"):
        return "break"
    if bool(getattr(self, "_preview_super_correction_active", False)):
        return self._select_preview_global_plate_relative(-1)
    return self._select_preview_relative(-1)

def _on_preview_arrow_up_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-arrow-up"):
        return "break"
    if bool(getattr(self, "_preview_super_correction_active", False)):
        return self._select_preview_global_plate_relative(-1)
    return self._select_preview_relative(-1)

def _on_preview_next_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-next"):
        return "break"
    if bool(getattr(self, "_preview_super_correction_active", False)):
        return self._select_preview_global_plate_relative(1)
    return self._select_preview_relative(1)

def _on_preview_arrow_down_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-arrow-down"):
        return "break"
    if bool(getattr(self, "_preview_super_correction_active", False)):
        return self._select_preview_global_plate_relative(1)
    return self._select_preview_relative(1)

def _on_preview_middle_click_zoom_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None
    try:
        return canvas._on_middle_click_zoom(event)
    except Exception:
        return "break"

def _on_preview_edit_modifier_press(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._set_preview_corner_drag_modifier(True)
    return "break"

def _on_preview_edit_modifier_release(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._set_preview_corner_drag_modifier(False)
    return "break"

def _on_preview_draw_toggle_shortcut(self, event=None):
    if self._event_has_control_modifier(event):
        return None
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._toggle_preview_draw_mode()
    return "break"

def _on_preview_delete_mode_shortcut(self, event=None):
    if self._event_has_control_modifier(event):
        return None
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._toggle_preview_delete_mode()
    return "break"

def _on_preview_delete_image_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if not self._ensure_preview_is_editable_for_action():
        self._update_preview_edit_status(
            "Usuwanie obrazu jest dostepne dopiero po przygotowaniu annotations.xml."
        )
        return "break"
    return self._delete_current_preview_image_hard(event)

def _on_preview_save_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if not self._ensure_preview_is_editable_for_action():
        self._update_preview_edit_status(
            "Zapis poprawek bedzie dostepny po przygotowaniu annotations.xml."
        )
        return "break"
    self._save_preview_edits()
    return "break"

def _on_preview_undo_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if not self._ensure_preview_is_editable_for_action():
        return "break"
    return self._undo_preview_edit(event)

def _on_preview_redo_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if not self._ensure_preview_is_editable_for_action():
        return "break"
    return self._redo_preview_edit(event)

def _on_preview_fit_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._fit_preview_image_to_view()
    self._update_preview_edit_status()
    return "break"

def _on_preview_focus_zoom_modifier_press(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if bool(getattr(self, "_preview_focus_zoom_modifier_down", False)):
        return "break"
    self._preview_focus_zoom_modifier_down = True
    self._preview_focus_zoom_modifier_consumed = False
    self._sync_preview_canvas_cursor()
    return "break"

def _on_preview_focus_zoom_modifier_release(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    was_consumed = bool(getattr(self, "_preview_focus_zoom_modifier_consumed", False))
    self._preview_focus_zoom_modifier_down = False
    self._preview_focus_zoom_modifier_consumed = False
    self._sync_preview_canvas_cursor()
    if was_consumed:
        return "break"
    return self._on_preview_focus_toggle_shortcut(event)

def _on_preview_focus_toggle_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_draw_mode:
        self._update_preview_edit_status(
            "Dokończ albo anuluj rysowanie nowej ramki przed użyciem R."
        )
        return "break"

    if self._preview_focus_restore_matches_current_image():
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_history = []
        self._restore_preview_focus_view(
            "Przywrócono poprzedni kadr. R ponownie zbliża aktywną ramkę."
        )
        return "break"

    ann = self._get_preview_annotation()
    plates = self._get_plate_detections(ann)
    if not plates:
        self._update_preview_edit_status("Na tym obrazie nie ma ramki tablicy do zbliżenia klawiszem R.")
        return "break"

    selected_idx = self._get_selected_plate_index_for_ann(ann)
    self._focus_preview_plate(
        selected_idx,
        store_restore=True,
        push_debug=True,
        status_message=(
            "Widok został dopasowany do aktywnej ramki. "
            + (
                "Q/E przechodzą po tablicach globalnie, A przełącza lokalnie, Y wyłącza super korektę."
                if bool(getattr(self, "_preview_super_correction_active", False))
                else "R wraca do poprzedniego kadru, A przełącza tablice."
            )
        ),
    ) or self._update_preview_edit_status("Nie udało się dopasować widoku do aktywnej ramki.")
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_history = []
    return "break"

def _on_preview_cycle_plate_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-cycle-plate"):
        return "break"
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass
    if self._preview_draw_mode:
        self._update_preview_edit_status(
            "Dokończ albo anuluj rysowanie nowej ramki przed użyciem A."
        )
        return "break"

    ann = self._get_preview_annotation()
    plates = self._get_plate_detections(ann)
    if not plates:
        self._update_preview_edit_status("Na tym obrazie nie ma ramek tablic do przełączania klawiszem A.")
        return "break"

    super_mode = bool(getattr(self, "_preview_super_correction_active", False))
    current_idx = self._get_selected_plate_index_for_ann(ann)
    if current_idx is None:
        current_idx = -1
    elif super_mode and len(plates) <= 1:
        focus_target = self._get_preview_focus_target()
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_restore_state = None
        self._preview_focus_zoom_history = []
        if not (
            isinstance(focus_target, dict)
            and str(focus_target.get("kind", "") or "") == "plate"
            and int(focus_target.get("index", -1) or -1) == int(current_idx)
        ):
            self._focus_preview_plate(
                int(current_idx),
                store_restore=False,
                push_debug=False,
                status_message=(
                    f"Aktywna tablica 1/1. Q/E przechodzą po tablicach globalnie, "
                    "A nie zmienia kadru, bo na tym obrazie jest tylko jedna tablica."
                ),
            )
        else:
            self._update_preview_edit_status(
                "Aktywna tablica 1/1. Q/E przechodzą po tablicach globalnie.",
                refresh_toolbar=False,
                refresh_debug=False,
            )
        return "break"
    target_idx = (int(current_idx) + 1) % len(plates)

    if super_mode:
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_restore_state = None
        self._preview_focus_zoom_history = []
    elif not self._preview_focus_restore_matches_current_image():
        self._preview_polygon_focus_restore_state = self._capture_preview_view_state()

    if self._focus_preview_plate(
        target_idx,
        store_restore=False,
        push_debug=False,
        status_message=(
            f"Aktywna tablica {int(target_idx) + 1}/{len(plates)}. "
            + (
                "Q/E przechodzą po tablicach globalnie, A przełącza lokalnie, Y wyłącza super korektę."
                if super_mode
                else "A przełącza kolejne ramki, R wraca do poprzedniego kadru."
            )
        ),
    ):
        self._push_preview_debug_event("cycle-plate", f"p{int(target_idx) + 1}/{len(plates)}")
    else:
        self._update_preview_edit_status("Nie udalo sie dopasowac widoku do wybranej tablicy.")
    return "break"

def _toggle_preview_super_correction(self, event=None):
    self._preview_super_correction_badge_visible = False
    if self._preview_draw_mode:
        self._update_preview_edit_status(
            "Dokończ albo anuluj rysowanie nowej ramki przed użyciem Y."
        )
        return "break"
    if self._preview_delete_mode:
        self._update_preview_edit_status(
            "Wyłącz tryb usuwania przed użyciem Y."
        )
        return "break"

    ann = self._get_preview_annotation()
    plates = self._get_plate_detections(ann) if ann is not None else []
    next_state = not bool(getattr(self, "_preview_super_correction_active", False))
    self._preview_super_correction_active = next_state
    self._preview_overlay_dock_render_key = None
    if not next_state:
        self._refresh_preview_canvas()
        self._place_preview_overlay_dock(force_render=True)
        self._update_preview_edit_status(
            "Super korekta wyłączona. Q/E znowu przełączają zdjęcia.",
            refresh_legend=True,
        )
        return "break"

    if plates:
        selected_idx = self._get_selected_plate_index_for_ann(ann)
        if selected_idx is None:
            selected_idx = 0
        self._focus_preview_plate(
            selected_idx,
            store_restore=False,
            push_debug=True,
            status_message=(
                f"Super korekta włączona. Q/E przechodzą teraz po tablicach globalnie, "
                f"A nadal przełącza tablice lokalnie, Y wyłącza tryb."
            ),
        )
    else:
        self._refresh_preview_canvas()
        self._update_preview_edit_status(
            "Super korekta włączona. Q/E będą szukać kolejnych zdjęć z tablicami, Y wyłącza tryb.",
            refresh_legend=True,
        )
    self._place_preview_overlay_dock(force_render=True)
    return "break"

def _on_preview_toggle_super_correction_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-toggle-super-correction"):
        return "break"
    return self._toggle_preview_super_correction(event)

def _on_preview_toggle_image_approval_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if self._preview_shortcut_is_duplicate(event, "preview-toggle-image-approval"):
        return "break"
    if self._preview_draw_mode:
        self._update_preview_edit_status(
            "Dokończ albo anuluj rysowanie nowej ramki przed zmianą statusu OK spacją."
        )
        return "break"

    ann = self._get_preview_annotation()
    if ann is None or self.current_preview_index is None:
        self._update_preview_edit_status("Najpierw wybierz zdjęcie do zatwierdzenia.")
        return "break"

    try:
        actual_index = int(self.current_preview_index)
    except Exception:
        self._update_preview_edit_status("Nie udało się ustalić bieżącego zdjęcia.")
        return "break"

    filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
    approved_lookup = {
        str(name or "").strip().lower()
        for name in set(getattr(self, "_preview_approved_filenames", set()) or set())
        if str(name or "").strip()
    }
    if not self._is_free_mode_session_context():
        approved_lookup.update(
            str(name or "").strip().lower()
            for name in set(getattr(self, "_campaign_pending_approved_filenames", set()) or set())
            if str(name or "").strip()
        )
    currently_approved = bool(
        filename_key
        and (
            bool(getattr(ann, "_approved_for_training", False))
            or filename_key in approved_lookup
        )
        and self._preview_annotation_can_be_approved_for_export(ann)
    )
    next_approved = not currently_approved
    if next_approved and not self._preview_annotation_can_be_approved_for_export(ann):
        self._update_preview_edit_status(
            "Nie można zatwierdzić zdjęcia spacją: najpierw dodaj ramkę tablicy."
        )
        try:
            self._update_preview_canvas_metrics_overlay(force_render=True)
        except Exception:
            pass
        return "break"

    self._set_selected_preview_images_approved(
        next_approved,
        actual_indices=[actual_index],
        show_warning_modal=False,
        persist_immediately=False,
        refresh_export_sources=False,
        schedule_followup_refresh=True,
    )
    return "break"

def _on_preview_enter_fullscreen_shortcut(self, event=None):
    if not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    self._toggle_preview_fullscreen()
    return "break"

def _is_preview_super_correction_badge_hit(self, canvas: ZoomableCanvas, canvas_x: float, canvas_y: float) -> bool:
    try:
        overlapping = canvas.find_overlapping(
            float(canvas_x) - 1.0,
            float(canvas_y) - 1.0,
            float(canvas_x) + 1.0,
            float(canvas_y) + 1.0,
        )
    except Exception:
        overlapping = ()
    for item_id in overlapping or ():
        try:
            tags = set(canvas.gettags(item_id) or ())
        except Exception:
            tags = set()
        if "preview_super_correction_badge" in tags:
            return True

    bbox = getattr(self, "_preview_super_correction_badge_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x1 <= float(canvas_x) <= x2 and y1 <= float(canvas_y) <= y2

def _is_preview_super_correction_handle_hit(self, canvas_x: float, canvas_y: float) -> bool:
    bbox = getattr(self, "_preview_super_correction_handle_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x1 <= float(canvas_x) <= x2 and y1 <= float(canvas_y) <= y2

def _is_preview_fullscreen_toggle_hit(self, canvas_x: float, canvas_y: float) -> bool:
    bbox = getattr(self, "_preview_fullscreen_toggle_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x1 <= float(canvas_x) <= x2 and y1 <= float(canvas_y) <= y2

def _on_preview_escape_shortcut(self, event=None):
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        return None
    self._set_preview_fullscreen(False)
    return "break"

def _toggle_preview_fullscreen(self):
    self._set_preview_fullscreen(not bool(getattr(self, "_preview_fullscreen_active", False)))

def _cancel_preview_controls_legend_animation(self):
    pending = getattr(self, "_preview_controls_legend_anim_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_controls_legend_anim_after_id = None
    self._preview_controls_legend_animating = False
    self._preview_controls_legend_render_width_override = 0.0
    self._preview_controls_legend_render_height_override = 0.0

def _get_preview_controls_legend_target_width(self) -> int:
    return z2_get_preview_controls_legend_target_width(self)

def _build_preview_controls_context_rows(self) -> list[tuple[str, str, str, str]]:
    return z2_build_preview_controls_context_rows(self)

def _preview_controls_row_is_multi_plate_badge(row_label: str, row_value: str) -> bool:
    return z2_preview_controls_row_is_multi_plate_badge(row_label, row_value)

def _get_preview_controls_legend_target_height(self, width: float | None = None) -> int:
    return z2_get_preview_controls_legend_target_height(self, width)

def _animate_preview_controls_legend_height(self):
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        self._hide_preview_controls_legend_overlay()
        return
    self._cancel_preview_controls_legend_animation()

    target_width = float(self._get_preview_controls_legend_target_width())
    target_height = float(self._get_preview_controls_legend_target_height(target_width))
    current_width = float(getattr(self, "_preview_controls_legend_current_width", 0.0) or 0.0)
    if current_width <= 0.0:
        try:
            current_width = float(getattr(self, "preview_controls_canvas", None).winfo_width() or 0.0)
        except Exception:
            current_width = 0.0
    if current_width <= 0.0:
        current_width = target_width
    current_height = float(getattr(self, "_preview_controls_legend_current_height", 0.0) or 0.0)
    if current_height <= 0.0:
        try:
            current_height = float(getattr(self, "preview_controls_canvas", None).winfo_height() or 0.0)
        except Exception:
            current_height = 0.0
    if current_height <= 0.0:
        current_height = 126.0

    start_width = current_width
    start_height = current_height
    render_width = max(start_width, target_width)
    render_height = max(start_height, target_height)
    self._preview_controls_legend_render_width_override = float(render_width)
    self._preview_controls_legend_render_height_override = float(render_height)

    # Rysujemy stan na pełnym obszarze pośrednim, a potem animujemy szerokość i wysokość.
    try:
        self._preview_controls_legend_current_width = float(render_width)
        self._preview_controls_legend_current_height = float(render_height)
        self._preview_controls_legend_render_key = None
        self._refresh_preview_controls_legend()
    except Exception:
        pass

    steps = 9
    duration_ms = 160
    step_ms = max(12, duration_ms // steps)
    self._preview_controls_legend_animating = True

    def ease_out(t: float) -> float:
        inv = 1.0 - max(0.0, min(1.0, t))
        return 1.0 - (inv * inv)

    def tick(step_idx: int = 0):
        progress = ease_out(float(step_idx) / float(max(1, steps)))
        width = start_width + ((target_width - start_width) * progress)
        height = start_height + ((target_height - start_height) * progress)
        self._preview_controls_legend_current_width = float(width)
        self._preview_controls_legend_current_height = float(height)
        self._place_preview_legend_overlay(
            width_override=float(width),
            height_override=float(height),
            refresh=False,
        )
        if step_idx >= steps:
            self._preview_controls_legend_anim_after_id = None
            final_width = float(target_width)
            final_height = max(
                float(target_height),
                float(getattr(self, "_preview_controls_legend_current_height", 0.0) or 0.0),
            )
            self._place_preview_legend_overlay(
                width_override=final_width,
                height_override=final_height,
                refresh=False,
            )
            self._preview_controls_legend_current_width = final_width
            self._preview_controls_legend_current_height = float(
                getattr(self, "_preview_controls_legend_viewport_height", final_height) or final_height
            )
            self._preview_controls_legend_render_width_override = 0.0
            self._preview_controls_legend_render_height_override = 0.0
            self._preview_controls_legend_animating = False
            try:
                self._preview_controls_legend_render_key = None
                self._refresh_preview_controls_legend()
                self._update_preview_canvas_metrics_overlay()
            except Exception:
                pass
            return
        try:
            self._preview_controls_legend_anim_after_id = self.frame.after(
                step_ms,
                lambda: tick(step_idx + 1),
            )
        except Exception:
            self._preview_controls_legend_anim_after_id = None
            self._preview_controls_legend_animating = False
            self._preview_controls_legend_render_width_override = 0.0
            self._preview_controls_legend_render_height_override = 0.0
            self._preview_controls_legend_current_width = float(target_width)
            self._preview_controls_legend_current_height = float(target_height)
            self._place_preview_legend_overlay(
                width_override=float(target_width),
                height_override=float(target_height),
                refresh=True,
            )

    tick(0)

def _sync_preview_controls_legend_scrollbar(self) -> None:
    canvas = getattr(self, "preview_controls_canvas", None)
    vbar = getattr(self, "preview_controls_vbar", None)
    if canvas is None or vbar is None:
        return
    try:
        expanded_scrollable = bool(self._is_preview_controls_legend_expanded())
        content_h = float(getattr(self, "_preview_controls_legend_content_height", 0.0) or 0.0)
        viewport_h = float(
            getattr(self, "_preview_controls_legend_viewport_height", 0.0)
            or getattr(self, "_preview_controls_legend_current_height", 0.0)
            or canvas.winfo_height()
            or 0.0
        )
        scroll_enabled = bool(expanded_scrollable and content_h > viewport_h + 2.0)
        self._preview_controls_legend_scroll_enabled = scroll_enabled
        if scroll_enabled:
            if not str(vbar.winfo_manager()):
                vbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))
            try:
                try:
                    legend_theme = self._get_preview_legend_theme()
                except Exception:
                    legend_theme = {}
                vbar.configure_style(
                    track_color=str(legend_theme.get("panel_fill", "#10251d")),
                    thumb_color=str(legend_theme.get("badge_plate_outline", "#4ade80")),
                    thumb_hover_color=str(legend_theme.get("shell_outline", "#86efac")),
                )
                max_scroll = max(0.0, content_h - viewport_h)
                if max_scroll > 0.0 and content_h > 0.0:
                    offset = max(
                        0.0,
                        min(float(getattr(self, "_preview_controls_legend_scroll_offset", 0.0) or 0.0), max_scroll),
                    )
                    first = max(0.0, min(1.0, offset / max(content_h, 1.0)))
                    last = max(first, min(1.0, (offset + viewport_h) / max(content_h, 1.0)))
                    vbar.set(first, last)
                else:
                    vbar.set(0.0, 1.0)
                vbar.lift()
            except Exception:
                pass
            try:
                canvas.configure(yscrollcommand="")
            except Exception:
                pass
        else:
            if str(vbar.winfo_manager()):
                vbar.pack_forget()
            self._preview_controls_legend_scroll_offset = 0.0
            canvas.yview_moveto(0.0)
    except Exception:
        pass

def _place_preview_legend_overlay(
    self,
    *,
    width_override: float | None = None,
    height_override: float | None = None,
    refresh: bool = True,
):
    if not bool(getattr(self, "_startup_ui_ready", False)):
        return
    preview_overlay = getattr(self, "preview_hint_frame", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    if preview_overlay is None or canvas_frame is None:
        return
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None or getattr(canvas, "original_image", None) is None:
        self._hide_preview_controls_legend_overlay()
        return
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        self._hide_preview_controls_legend_overlay()
        return

    fullscreen = bool(getattr(self, "_preview_fullscreen_active", False))
    if width_override is not None:
        overlay_width = float(width_override or 0.0)
    elif fullscreen and self._is_preview_controls_legend_expanded():
        overlay_width = float(getattr(self, "_preview_controls_legend_current_width", 0.0) or 0.0)
    else:
        overlay_width = float(self._get_preview_controls_legend_target_width())
    if overlay_width <= 0.0:
        overlay_width = float(self._get_preview_controls_legend_target_width())
    self._preview_controls_legend_current_width = overlay_width

    expanded_scrollable = bool(self._is_preview_controls_legend_expanded())
    content_height = float(self._get_preview_controls_legend_target_height(overlay_width))
    if height_override is not None:
        overlay_height = float(height_override or 0.0)
    elif fullscreen and expanded_scrollable:
        overlay_height = float(getattr(self, "_preview_controls_legend_current_height", 0.0) or 0.0)
    else:
        overlay_height = float(content_height)
    if overlay_height <= 0.0:
        overlay_height = float(content_height)

    if fullscreen:
        offset_x = float(getattr(self, "_preview_controls_legend_offset_x", 10.0) or 10.0)
        offset_y = float(getattr(self, "_preview_controls_legend_offset_y", 10.0) or 10.0)
    elif bool(getattr(self, "_preview_controls_legend_inline_manual_position", False)):
        offset_x = float(getattr(self, "_preview_controls_legend_offset_x", 10.0) or 10.0)
        offset_y = float(getattr(self, "_preview_controls_legend_offset_y", 52.0) or 52.0)
    else:
        # Keep the inline compass away from the top-right canvas fullscreen icon.
        offset_x = 10.0
        offset_y = 52.0

    if expanded_scrollable:
        try:
            frame_h = float(canvas_frame.winfo_height() or canvas_frame.winfo_reqheight() or 0.0)
        except Exception:
            frame_h = 0.0
        if frame_h > 0.0:
            available_h = max(156.0, frame_h - float(offset_y) - 12.0)
            overlay_height = min(content_height, available_h)
    self._preview_controls_legend_content_height = float(content_height)
    self._preview_controls_legend_viewport_height = float(overlay_height)
    self._preview_controls_legend_scroll_enabled = bool(expanded_scrollable and content_height > overlay_height + 2.0)
    if not bool(getattr(self, "_preview_controls_legend_scroll_enabled", False)):
        self._preview_controls_legend_scroll_offset = 0.0
        controls_canvas = getattr(self, "preview_controls_canvas", None)
        if controls_canvas is not None:
            try:
                controls_canvas.yview_moveto(0.0)
            except Exception:
                pass
        _sync_preview_controls_legend_scrollbar(self)
    self._preview_controls_legend_current_height = overlay_height

    offset_x, offset_y = self._clamp_preview_controls_legend_offsets(
        offset_x,
        offset_y,
        width=overlay_width,
        height=overlay_height,
    )
    self._preview_controls_legend_offset_x = float(offset_x)
    self._preview_controls_legend_offset_y = float(offset_y)

    try:
        preview_overlay.place(
            in_=canvas_frame,
            x=int(round(offset_x)),
            y=int(round(offset_y)),
            width=int(overlay_width),
            height=int(math.ceil(overlay_height)),
            anchor="nw",
        )
        preview_overlay.lift()
        controls_canvas = getattr(self, "preview_controls_canvas", None)
        if controls_canvas is not None:
            controls_canvas.lift()
    except Exception:
        pass

    if refresh:
        try:
            controls_canvas = getattr(self, "preview_controls_canvas", None)
            has_rendered_legend = False
            if controls_canvas is not None:
                try:
                    has_rendered_legend = bool(controls_canvas.find_withtag("preview_legend"))
                except Exception:
                    has_rendered_legend = False
            if self._should_freeze_preview_legend_updates() and has_rendered_legend:
                _sync_preview_controls_legend_scrollbar(self)
                return
            self._refresh_preview_controls_legend()
        except Exception:
            pass
    _sync_preview_controls_legend_scrollbar(self)

def _apply_preview_fullscreen_chrome(self):
    preview_tools = getattr(self, "preview_tools", None)
    preview_hint_label = getattr(self, "preview_fullscreen_hint_lbl", None)
    preview_hint_frame = getattr(self, "preview_hint_frame", None)
    canvas_frame = getattr(self, "canvas_frame", None)
    status_label = getattr(self, "preview_edit_status_lbl", None)
    debug_label = getattr(self, "preview_debug_lbl", None)
    log_tools = getattr(self, "log_tools", None)

    if bool(getattr(self, "_preview_fullscreen_active", False)):
        try:
            if preview_tools is not None:
                preview_tools.pack_forget()
        except Exception:
            pass
        try:
            if preview_hint_label is not None:
                preview_hint_label.pack_forget()
        except Exception:
            pass
        try:
            if status_label is not None:
                status_label.pack_forget()
        except Exception:
            pass
        try:
            if debug_label is not None:
                debug_label.pack_forget()
        except Exception:
            pass
        try:
            if log_tools is not None:
                log_tools.pack_forget()
        except Exception:
            pass
        try:
            if preview_hint_frame is not None and not bool(getattr(self, "_preview_fullscreen_transition_active", False)):
                self._place_preview_legend_overlay()
        except Exception:
            pass
        return

    try:
        if preview_hint_frame is not None and not bool(getattr(self, "_preview_fullscreen_transition_active", False)):
            self._place_preview_legend_overlay()
    except Exception:
        pass
    try:
        if preview_hint_label is not None and str(preview_hint_label.winfo_manager()):
            preview_hint_label.pack_forget()
    except Exception:
        pass
    try:
        if status_label is not None and str(status_label.winfo_manager()):
            status_label.pack_forget()
    except Exception:
        pass
    try:
        if log_tools is not None and str(log_tools.winfo_manager()):
            log_tools.pack_forget()
    except Exception:
        pass
    if bool(getattr(self, "_preview_debug_enabled", False)):
        try:
            if debug_label is not None and not str(debug_label.winfo_manager()):
                debug_label.pack(fill=tk.X, pady=(6, 0), after=status_label)
        except Exception:
            pass

def _set_preview_fullscreen(self, active: bool):
    next_state = bool(active)
    if next_state == bool(getattr(self, "_preview_fullscreen_active", False)):
        if next_state:
            try:
                self.preview_canvas.focus_set()
            except Exception:
                pass
        return

    has_image = getattr(self.preview_canvas, "original_image", None) is not None
    if next_state and not has_image:
        self._update_preview_edit_status("Pełny ekran jest dostępny po załadowaniu obrazu podglądu.")
        return

    self._preview_fullscreen_transition_active = True
    self._preview_fullscreen_overlay_ready = False

    root = getattr(self.app, "root", None)
    try:
        windowing_system = str(self.frame.tk.call("tk", "windowingsystem")).lower()
    except Exception:
        windowing_system = ""
    use_native_root_fullscreen = windowing_system not in {"win32"}

    try:
        self._hide_preview_overlay_dock_stack()
    except Exception:
        pass
    try:
        self._hide_preview_controls_legend_overlay()
    except Exception:
        pass
    transition_seq = int(getattr(self, "_preview_fullscreen_transition_seq", 0) or 0) + 1
    self._preview_fullscreen_transition_seq = transition_seq

    if next_state:
        self._preview_fullscreen_restore_log_visible = bool(getattr(self, "_annotation_log_visible", False))
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

        if self._pane_has_child(self.main_pane, self.main_left_frame):
            self.main_pane.forget(self.main_left_frame)
        if self._pane_has_child(self.main_pane, self.main_right_frame):
            self.main_pane.forget(self.main_right_frame)
        self._set_annotation_process_log_visibility(False)
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
        self._preview_controls_legend_fullscreen_expanded = False
        self._preview_controls_legend_current_width = 0.0
        self._preview_controls_legend_current_height = 0.0
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

        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None
        self._preview_force_fit_after_resize = True
        # _should_show_right_panel() celowo zwraca False w fullscreen,
        # więc najpierw zdejmujemy flagę, dopiero potem odtwarzamy panele.
        self._preview_fullscreen_active = False
        if not self._pane_has_child(self.main_pane, self.main_left_frame):
            self.main_pane.insert(0, self.main_left_frame, weight=2)
        if self._should_show_right_panel() and not self._pane_has_child(self.main_pane, self.main_right_frame):
            self.main_pane.add(self.main_right_frame, weight=1)

        self._set_annotation_process_log_visibility(bool(getattr(self, "_preview_fullscreen_restore_log_visible", False)))
        self._preview_controls_legend_current_width = 0.0
        self._preview_controls_legend_current_height = 0.0

    self._apply_preview_fullscreen_chrome()
    self._cancel_preview_controls_legend_animation()
    self._preview_controls_legend_render_key = None
    self._push_preview_debug_event(
        "fullscreen",
        f"active={int(bool(self._preview_fullscreen_active))} mode={'native' if use_native_root_fullscreen else 'zoomed'} ws={windowing_system or '-'}"
    )
    self._schedule_preview_layout_restore_after_resize()
    if not self._preview_fullscreen_active:
        self._sync_main_pane_right_panel_visibility()
        self._schedule_right_panel_content_restore_after_fullscreen(delay_ms=180)
        self._schedule_main_pane_layout_refresh(force_defaults=True, delay_ms=120)
    refresh_delay_ms = 210 if bool(getattr(self, "_preview_fullscreen_active", False)) else 230

    def _refresh_after_fullscreen_transition():
        if transition_seq != int(getattr(self, "_preview_fullscreen_transition_seq", 0) or 0):
            return
        if not bool(getattr(self, "_preview_fullscreen_active", False)) and bool(
            getattr(self, "_preview_list_selection_sync_pending", False)
        ):
            try:
                current_idx = int(getattr(self, "current_preview_index", 0) or 0)
                display_idx = self._get_preview_display_index(current_idx)
                if display_idx is None:
                    display_idx = max(0, min(current_idx, max(0, self.preview_listbox.size() - 1)))
                self._clear_listbox_selection_fast(self.preview_listbox)
                self.preview_listbox.selection_set(display_idx)
                self.preview_listbox.activate(display_idx)
                self.preview_listbox.see(display_idx)
            except Exception:
                pass
            self._preview_list_selection_sync_pending = False
        try:
            self._update_preview_toolbar_state(refresh_summary=False)
        except Exception:
            pass
        try:
            self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
        except Exception:
            pass
        if not bool(getattr(self, "_preview_fullscreen_active", False)):
            try:
                self._refresh_preview_list_row_for_actual_index(
                    getattr(self, "current_preview_index", None),
                    refresh_summary=False,
                    lightweight=True,
                )
            except Exception:
                pass
            try:
                self._refresh_preview_list_summary(lightweight=True)
            except Exception:
                pass
            pending_counter_refresh = bool(
                getattr(self, "_preview_campaign_counter_refresh_pending_after_fullscreen", False)
            )
            pending_char_refresh = bool(
                getattr(self, "_campaign_char_effective_source_refresh_pending_after_fullscreen", False)
            )
            self._preview_campaign_counter_refresh_pending_after_fullscreen = False
            self._campaign_char_effective_source_refresh_pending_after_fullscreen = False
            if pending_counter_refresh:
                try:
                    self._schedule_preview_approval_followup_refresh(delay_ms=700)
                except Exception:
                    pass
            if pending_char_refresh:
                try:
                    self._schedule_campaign_char_effective_source_refresh(delay_ms=350)
                except Exception:
                    pass
        try:
            setattr(self, "_preview_controls_legend_current_width", 0.0)
            setattr(self, "_preview_controls_legend_current_height", 0.0)
            setattr(self, "_preview_controls_legend_render_key", None)
            setattr(self, "_preview_fullscreen_overlay_ready", True)
            setattr(self, "_preview_fullscreen_transition_active", False)
            self._place_preview_legend_overlay(refresh=True)
            self._place_preview_overlay_dock(force_render=True)
            self._place_preview_campaign_gate_overlay(force_render=True)
        except Exception:
            pass

    def _schedule_transition_refresh(delay_ms: int) -> None:
        try:
            self.frame.after(int(delay_ms), _refresh_after_fullscreen_transition)
        except Exception:
            _refresh_after_fullscreen_transition()

    try:
        _schedule_transition_refresh(refresh_delay_ms)
    except Exception:
        _refresh_after_fullscreen_transition()
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass

def _toggle_preview_draw_mode(self):
    ann = self._get_preview_annotation()
    if ann is None or self.preview_canvas.original_image is None:
        return
    if not self._ensure_preview_is_editable_for_action():
        self._update_preview_edit_status(
            f"Ten podgląd jest tylko informacyjny. Najpierw kliknij {self._get_step2_start_action_reference()}, aby przygotować annotations.xml."
        )
        return

    if self._preview_draw_mode:
        self._preview_draw_mode = False
        self._preview_draw_points = []
    else:
        self._preview_delete_mode = False
        self._preview_delete_candidate_idx = None
        self._preview_drag_state = None
        self._preview_draw_mode = True
        self._preview_draw_points = []
        try:
            if getattr(self, "_preview_autosave_after_id", None) and getattr(self, "_preview_dirty_images", None):
                self._defer_preview_autosave_for_navigation(delay_ms=12000)
        except Exception:
            pass

    self._refresh_preview_canvas(refresh_chrome=False)
    self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
    self._update_preview_toolbar_state(refresh_summary=False)
    if self._is_free_mode_session_context():
        try:
            self._sync_main_pane_right_panel_visibility()
            self._refresh_free_mode_manual_right_panel()
        except Exception:
            pass
    self._push_preview_debug_event("draw-mode", f"{'on' if self._preview_draw_mode else 'off'}")
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass

def _toggle_preview_delete_mode(self):
    ann = self._get_preview_annotation()
    if ann is None or self.preview_canvas.original_image is None:
        return
    if not self._ensure_preview_is_editable_for_action():
        self._update_preview_edit_status(
            f"Ten podgląd jest tylko informacyjny. Najpierw kliknij {self._get_step2_start_action_reference()}, aby przygotować annotations.xml."
        )
        return

    plates = self._get_plate_detections(ann)
    if not self._preview_delete_mode and not plates:
        self._update_preview_edit_status("Na tym obrazie nie ma ramki tablicy do usunięcia.")
        return

    if self._preview_delete_mode:
        self._preview_delete_mode = False
        self._preview_delete_candidate_idx = None
    else:
        if self._preview_draw_mode:
            self._preview_draw_mode = False
            self._preview_draw_points = []
        if self._preview_drag_state is not None:
            self._finish_preview_vertex_drag(
                mark_dirty=bool(self._preview_drag_state.get("was_moved", False))
            )
        self._set_preview_corner_drag_modifier(False)
        self._preview_pending_vertex_hit = None
        pointer_canvas = self._get_preview_pointer_canvas_position()
        hovered_idx = None
        if pointer_canvas is not None:
            hovered_idx = self._find_preview_polygon_hit(float(pointer_canvas[0]), float(pointer_canvas[1]))

        if hovered_idx is None:
            self._preview_delete_mode = False
            self._preview_delete_candidate_idx = None
            self._refresh_preview_canvas_light()
            self._update_preview_edit_status(
                "Najedź kursorem na wnętrze ramki i naciśnij S, aby uzbroić usuwanie tej anotacji tablicy.",
                refresh_toolbar=False,
                refresh_debug=False,
            )
            self._push_preview_debug_event("delete-mode", "off no-hover-target")
            try:
                self.preview_canvas.focus_set()
            except Exception:
                pass
            return

        self._preview_delete_mode = True
        self._preview_delete_candidate_idx = int(hovered_idx)
        self._set_selected_plate_index_for_ann(ann, int(hovered_idx))

    self._refresh_preview_canvas_light()
    self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
    self._update_preview_toolbar_state(refresh_summary=False)
    if self._preview_delete_mode and self._preview_delete_candidate_idx is not None:
        self._push_preview_debug_event("delete-mode", f"on p{int(self._preview_delete_candidate_idx) + 1}")
    else:
        self._push_preview_debug_event("delete-mode", "off")
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass

def _set_preview_corner_drag_modifier(self, active: bool):
    modifier_started = time.perf_counter()
    phase_at = modifier_started
    phase_ms: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal phase_at
        now = time.perf_counter()
        phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
        phase_at = now

    next_state = bool(active)
    if bool(getattr(self, "_preview_corner_drag_modifier_down", False)) == next_state:
        return

    needs_canvas_refresh = False
    self._preview_corner_drag_modifier_down = next_state
    if next_state:
        self._preview_last_modifier_press_at = time.monotonic()
        try:
            if getattr(self, "_preview_autosave_after_id", None) and getattr(self, "_preview_dirty_images", None):
                self._defer_preview_autosave_for_navigation(delay_ms=12000)
        except Exception:
            pass
        try:
            self.preview_canvas._cancel_deferred_display()
            self.preview_canvas._cancel_final_quality_display()
        except Exception:
            pass
        mark_phase("cancel_canvas")
        self._preview_corner_drag_grab_ready = False
        self._preview_drag_session_history_image_key = ""
        try:
            ann = self._get_preview_annotation()
            image_key = str(self._get_preview_history_image_key(ann) or "").strip()
            if image_key and self._get_plate_detections(ann):
                # The first real drag after switching plates must not build the
                # undo snapshot under the cursor. Prime a light snapshot when W
                # is armed and mark it as ready for this drag session.
                self._push_preview_history_snapshot(ann, lightweight_plate_edit=True)
                self._preview_drag_session_history_image_key = image_key
        except Exception:
            self._preview_drag_session_history_image_key = ""
        mark_phase("prime_history")
        try:
            self.preview_canvas.grab_set()
            self._preview_corner_drag_grab_ready = True
        except Exception:
            self._preview_corner_drag_grab_ready = False
        mark_phase("grab_set")
    else:
        self._preview_last_modifier_press_at = 0.0
        self._preview_drag_session_history_image_key = ""
        self._preview_corner_drag_grab_ready = False
    try:
        self.preview_canvas.pan_data["press_x"] = None
        self.preview_canvas.pan_data["press_y"] = None
    except Exception:
        pass
    self._sync_preview_canvas_cursor()
    if not next_state:
        needs_canvas_refresh = self._preview_pending_vertex_hit is not None
        self._preview_pending_vertex_hit = None
    if not next_state and self._preview_drag_state is not None:
        self._finish_preview_vertex_drag(
            mark_dirty=bool(self._preview_drag_state.get("was_moved", False))
        )
        return
    if not next_state and self._preview_drag_state is None and self._preview_dirty_images:
        ann = self._get_preview_annotation()
        if ann is not None:
            try:
                for det in self._get_plate_detections(ann):
                    fixed_polygon = PolygonValidator.fix_polygon(self._detection_polygon(det))
                    det.polygon = fixed_polygon
                    det.bbox = self._bbox_from_polygon(fixed_polygon)
                    det.keypoints = self._keypoints_from_polygon(fixed_polygon)
            except Exception:
                pass
        try:
            self._invalidate_preview_runtime_caches()
        except Exception:
            pass
        self._schedule_preview_autosave(delay_ms=6500)
        self._refresh_preview_canvas_interactive(delay_ms=140)
        self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
        needs_canvas_refresh = False

    if not next_state and self._preview_drag_state is None:
        try:
            self.preview_canvas.grab_release()
        except Exception:
            pass

    self._push_preview_debug_event("modifier", f"W={'on' if next_state else 'off'}")
    if needs_canvas_refresh:
        self._refresh_preview_canvas_interactive(delay_ms=160)
    elapsed_ms = max(0.0, (time.perf_counter() - modifier_started) * 1000.0)
    _log_preview_super_perf(
        self,
        "modifier.set",
        elapsed_ms,
        threshold_ms=0.0 if _preview_super_perf_active(self) else 50.0,
        active=int(next_state),
        cancel=f"{phase_ms.get('cancel_canvas', 0.0):.1f}",
        prime=f"{phase_ms.get('prime_history', 0.0):.1f}",
        grab=f"{phase_ms.get('grab_set', 0.0):.1f}",
    )

def _preview_modifier_active(self) -> bool:
    return bool(getattr(self, "_preview_corner_drag_modifier_down", False))

def _prime_preview_corner_drag_history(self) -> None:
    try:
        ann = self._get_preview_annotation()
        image_key = str(self._get_preview_history_image_key(ann) or "").strip()
        if not image_key or not self._get_plate_detections(ann):
            return
        if str(getattr(self, "_preview_drag_session_history_image_key", "") or "") == image_key:
            return
        self._push_preview_history_snapshot(ann, lightweight_plate_edit=True)
        self._preview_drag_session_history_image_key = image_key
    except Exception:
        pass

def _begin_preview_vertex_drag(
    self,
    plate_idx: int,
    vertex_idx: int,
    anchor_canvas_x: float,
    anchor_canvas_y: float,
):
    drag_started_at = time.perf_counter()
    phase_at = drag_started_at
    phase_ms: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal phase_at
        now = time.perf_counter()
        phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
        phase_at = now

    # Szybka seria korekt nie powinna byc przerywana opoznionym zapisem poprzedniego ruchu.
    self._cancel_preview_autosave()
    self._cancel_preview_post_interaction_refresh()
    self._cancel_preview_layout_restore_jobs()
    try:
        self.preview_canvas._cancel_zoom_animation()
        self.preview_canvas._cancel_deferred_display()
        self.preview_canvas._cancel_final_quality_display()
    except Exception:
        pass
    mark_phase("cancel")
    self._preview_last_modifier_press_at = time.monotonic()
    start_cursor_x = 0.0
    start_cursor_y = 0.0
    start_vertex_x = 0.0
    start_vertex_y = 0.0

    ann = self._get_preview_annotation()
    if ann is not None and self.preview_canvas.original_image is not None:
        plate_detections = self._get_plate_detections(ann)
        if 0 <= int(plate_idx) < len(plate_detections):
            polygon = self._detection_polygon(plate_detections[int(plate_idx)])
            if 0 <= int(vertex_idx) < len(polygon):
                start_vertex_x, start_vertex_y = polygon[int(vertex_idx)]
                start_cursor_x, start_cursor_y = self.preview_canvas.canvas_to_image_coords(
                    anchor_canvas_x,
                    anchor_canvas_y,
                    clamp=True,
                )
    mark_phase("coords")

    self._preview_drag_state = {
        "plate_idx": int(plate_idx),
        "vertex_idx": int(vertex_idx),
        "press_canvas_x": float(anchor_canvas_x),
        "press_canvas_y": float(anchor_canvas_y),
        "start_cursor_x": float(start_cursor_x),
        "start_cursor_y": float(start_cursor_y),
        "start_vertex_x": float(start_vertex_x),
        "start_vertex_y": float(start_vertex_y),
        "drag_started": False,
        "was_moved": False,
    }
    self._preview_pending_vertex_hit = None

    try:
        self.preview_canvas.pan_data["press_x"] = None
        self.preview_canvas.pan_data["press_y"] = None
        if not bool(getattr(self, "_preview_corner_drag_grab_ready", False)):
            self.preview_canvas.grab_set()
            self._preview_corner_drag_grab_ready = True
    except Exception:
        self._preview_corner_drag_grab_ready = False
        pass
    mark_phase("grab")
    elapsed_ms = max(0.0, (time.perf_counter() - drag_started_at) * 1000.0)
    _log_preview_super_perf(
        self,
        "drag.begin",
        elapsed_ms,
        threshold_ms=0.0 if _preview_super_perf_active(self) else 40.0,
        plate=plate_idx,
        vertex=vertex_idx,
        cancel=f"{phase_ms.get('cancel', 0.0):.1f}",
        coords=f"{phase_ms.get('coords', 0.0):.1f}",
        grab=f"{phase_ms.get('grab', 0.0):.1f}",
        grab_ready=int(bool(getattr(self, "_preview_corner_drag_grab_ready", False))),
    )

def _on_preview_canvas_focus_out(self, event=None):
    try:
        self.preview_canvas.pan_data["press_x"] = None
        self.preview_canvas.pan_data["press_y"] = None
    except Exception:
        pass

    try:
        self.preview_canvas.grab_release()
    except Exception:
        pass

    self._preview_pending_vertex_hit = None
    self._preview_focus_zoom_modifier_down = False
    self._preview_focus_zoom_modifier_consumed = False
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_restore_state = None
    self._preview_focus_zoom_history = []
    if self._preview_drag_state is not None:
        self._finish_preview_vertex_drag(
            mark_dirty=bool(self._preview_drag_state.get("was_moved", False))
        )
    self._push_preview_debug_event("focus-out", "canvas utracil fokus")

def on_zoomable_canvas_should_block_pan(self, canvas: ZoomableCanvas, event):
    if canvas is not self.preview_canvas:
        return False

    if self._preview_draw_mode:
        return True

    if self._preview_delete_mode:
        return True

    if self._preview_drag_state is not None:
        return True

    if isinstance(getattr(self, "_preview_bottom_hint_drag_state", None), dict):
        return True

    if self._preview_pending_vertex_hit is not None:
        return True

    if self._preview_focus_zoom_modifier_down:
        return True

    return bool(self._preview_corner_drag_modifier_down)

def on_zoomable_canvas_zoom(self, canvas: ZoomableCanvas, event):
    if canvas is not self.preview_canvas:
        return False
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass
    self._push_preview_debug_event(
        "zoom",
        f"level={float(getattr(canvas, 'zoom_level', 0.0) or 0.0):.3f} at=({float(getattr(event, 'x', 0.0)):.1f},{float(getattr(event, 'y', 0.0)):.1f})"
    )
    return False


def on_zoomable_canvas_final_quality_delay_ms(self, canvas: ZoomableCanvas) -> int:
    if canvas is not self.preview_canvas:
        return 0
    try:
        return self._preview_user_interaction_quiet_remaining_ms(padding_ms=260)
    except Exception:
        return 0


def _get_preview_canvas_cursor(self) -> str:
    if isinstance(getattr(self, "_preview_super_correction_drag_state", None), dict):
        return "fleur"
    if bool(getattr(self, "_preview_draw_mode", False)):
        return "crosshair"
    if bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        return "crosshair"
    if bool(getattr(self, "_preview_focus_zoom_modifier_down", False)):
        return "crosshair"
    pointer_pos = self._get_preview_pointer_canvas_position()
    if pointer_pos is not None:
        canvas_x, canvas_y = pointer_pos
        bbox = getattr(self, "_preview_bottom_hint_move_bbox", None)
        if isinstance(bbox, tuple) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                if x1 <= float(canvas_x) <= x2 and y1 <= float(canvas_y) <= y2:
                    return "fleur"
            except Exception:
                pass
        for bbox_name in (
            "_preview_bottom_hint_restore_bbox",
            "_preview_bottom_hint_collapse_bbox",
        ):
            bbox = getattr(self, bbox_name, None)
            if isinstance(bbox, tuple) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                    if x1 <= float(canvas_x) <= x2 and y1 <= float(canvas_y) <= y2:
                        return "hand2"
                except Exception:
                    pass
        if self._is_preview_fullscreen_toggle_hit(canvas_x, canvas_y):
            return "hand2"
        if self._is_preview_super_correction_handle_hit(canvas_x, canvas_y):
            return "fleur"
        try:
            canvas = getattr(self, "preview_canvas", None)
            if canvas is not None and self._is_preview_super_correction_badge_hit(canvas, canvas_x, canvas_y):
                return "hand2"
        except Exception:
            pass
    return "arrow"

def _undo_preview_edit(self, event=None):
    image_key = self._get_preview_history_image_key()
    undo_stack = self._get_preview_history_stack("undo", image_key, create=False)
    if not image_key or not isinstance(undo_stack, list) or not undo_stack:
        self._update_preview_edit_status("Brak zmian do cofniecia.")
        return "break"

    current_snapshot = self._clone_preview_annotation_history_snapshot()
    redo_stack = self._get_preview_history_stack("redo", image_key, create=True)
    if current_snapshot is not None:
        redo_stack.append(current_snapshot)

    target_snapshot = undo_stack.pop()
    self._restore_preview_annotation_history_snapshot(
        target_snapshot,
        action_label="Cofnięto ostatnią zmianę ramek tablic.",
    )
    return "break"

def _redo_preview_edit(self, event=None):
    image_key = self._get_preview_history_image_key()
    redo_stack = self._get_preview_history_stack("redo", image_key, create=False)
    if not image_key or not isinstance(redo_stack, list) or not redo_stack:
        self._update_preview_edit_status("Brak zmian do ponowienia.")
        return "break"

    current_snapshot = self._clone_preview_annotation_history_snapshot()
    undo_stack = self._get_preview_history_stack("undo", image_key, create=True)
    if current_snapshot is not None:
        undo_stack.append(current_snapshot)

    target_snapshot = redo_stack.pop()
    self._restore_preview_annotation_history_snapshot(
        target_snapshot,
        action_label="Przywrócono ostatnią cofniętą zmianę ramek tablic.",
    )
    return "break"

def _sync_preview_canvas_cursor(self):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return

    desired_cursor = self._get_preview_canvas_cursor()
    try:
        canvas.current_cursor = desired_cursor
    except Exception:
        pass

    try:
        if canvas.pan_data.get("press_x") is None and canvas.pan_data.get("press_y") is None:
            canvas.config(cursor=desired_cursor)
    except Exception:
        try:
            canvas.config(cursor=desired_cursor)
        except Exception:
            pass

def _preview_is_editable(self) -> bool:
    if self._get_preview_annotation() is None:
        return False
    if self._is_free_mode_manual_xml_waiting_for_review():
        return False
    return self._get_current_annotation_xml_path() is not None

def _ensure_preview_is_editable_for_action(self) -> bool:
    if self._preview_is_editable():
        return True
    if self._get_preview_annotation() is None:
        return False
    if self._is_free_mode_session_context():
        return False
    if bool(getattr(self, "_campaign_project_restore_in_progress", False)):
        return False
    try:
        if self._ensure_campaign_preview_edit_run():
            return self._preview_is_editable()
    except Exception:
        pass
    return False
