#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 panel layout, workflow-card and filter UI helpers extracted from tab_annotation.py."""

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
from . import z2_canvas_interaction
from . import z2_preview_editor
from . import z2_preview_state
from . import z2_miniflow_runtime
from . import z2_model_runtime
from . import z2_session_runtime
from . import z2_context_runtime
from . import z2_run_io_runtime
from . import z2_run_lifecycle
from . import z2_status_ui_runtime
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

def _sync_right_panel_scrollregion(self, event=None):
    canvas = getattr(self, "right_settings_canvas", None)
    if canvas is None:
        return
    try:
        canvas.configure(scrollregion=canvas.bbox("all"))
    except Exception:
        pass


def _sync_right_panel_canvas_width(self, event=None):
    canvas = getattr(self, "right_settings_canvas", None)
    window_id = getattr(self, "_right_settings_window_id", None)
    if canvas is None or window_id is None:
        return

    width = getattr(event, "width", 0) or canvas.winfo_width()
    if width <= 1:
        return

    try:
        if int(float(canvas.itemcget(window_id, "width") or 0)) == int(width):
            return
        canvas.itemconfigure(window_id, width=width)
    except Exception:
        pass
    self._schedule_main_pane_layout_refresh(force_defaults=False, delay_ms=40)


def _sync_left_panel_scrollregion(self, event=None):
    canvas = getattr(self, "left_settings_canvas", None)
    if canvas is None:
        return
    try:
        canvas.configure(scrollregion=canvas.bbox("all"))
    except Exception:
        pass


def _sync_left_panel_canvas_width(self, event=None):
    canvas = getattr(self, "left_settings_canvas", None)
    window_id = getattr(self, "_left_settings_window_id", None)
    if canvas is None or window_id is None:
        return

    width = getattr(event, "width", 0) or canvas.winfo_width()
    if width <= 1:
        return

    try:
        if int(float(canvas.itemcget(window_id, "width") or 0)) == int(width):
            return
        canvas.itemconfigure(window_id, width=width)
    except Exception:
        pass
    self._schedule_main_pane_layout_refresh(force_defaults=False, delay_ms=40)


def _should_show_free_mode_manual_right_panel(self) -> bool:
    if not self._is_free_mode_session_context():
        return False
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        return False
    try:
        route = str(self._get_workflow_route() or "").strip().lower()
        if route not in {"auto", "manual"}:
            return False
        screen = str(self._coerce_free_mode_screen() or "").strip().lower()
        step = str(self._coerce_workflow_step() or "").strip().lower()
    except Exception:
        return False
    try:
        has_active_run = bool(self._get_preferred_annotation_run_dir(require_xml=True) is not None)
    except Exception:
        has_active_run = False
    has_loaded_input_workspace = False
    try:
        has_loaded_input_workspace = bool(getattr(self, "current_annotations", None))
    except Exception:
        has_loaded_input_workspace = False
    if not has_loaded_input_workspace:
        try:
            has_loaded_input_workspace = bool(str(self.input_dir_var.get() or "").strip())
        except Exception:
            has_loaded_input_workspace = False
    if screen == "workflow" and route in {"auto", "manual"}:
        return bool(
            step in {"auto_start", "manual_start"}
            and (has_active_run or has_loaded_input_workspace)
        )
    if screen in {"manual_review", "auto_summary", "export"}:
        return has_active_run
    return bool(
        screen == "workflow"
        and step in {"auto_start", "manual_start"}
        and (has_active_run or has_loaded_input_workspace)
    )


def _should_show_right_panel(self) -> bool:
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        return False
    if self._is_free_mode_session_context():
        return self._should_show_free_mode_manual_right_panel()
    return bool(getattr(self, "_annotation_right_panel_visible", True))


def _get_preview_left_counter_total(self) -> int:
    totals = []
    try:
        totals.append(int(len(getattr(self, "current_annotations", []) or [])))
    except Exception:
        pass
    try:
        totals.append(int(len(getattr(self, "_preview_list_display_indices", []) or [])))
    except Exception:
        pass
    try:
        pending_summary = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
        totals.append(int(pending_summary.get("pending_count", 0) or 0))
    except Exception:
        pass
    return max([0] + [max(0, int(value or 0)) for value in totals])


def _get_preview_left_counter_width_bucket(self) -> int:
    total = self._get_preview_left_counter_total()
    if total <= 0:
        return 0
    return max(1, len(str(max(1, int(total)))))


def _get_preview_left_counter_min_width(self) -> int:
    digits = self._get_preview_left_counter_width_bucket()
    if digits <= 0:
        return 0
    # Stable digit buckets: enough room for legend counters without resizing on every refresh.
    return min(420, 318 + max(0, int(digits) - 3) * 28)


def _sync_preview_left_counter_layout_width(self) -> None:
    bucket = self._get_preview_left_counter_width_bucket()
    if bucket == int(getattr(self, "_preview_left_counter_width_bucket", 0) or 0):
        return
    self._preview_left_counter_width_bucket = int(bucket)
    try:
        self._schedule_main_pane_layout_refresh(force_defaults=False, delay_ms=80)
    except Exception:
        pass


def _sync_main_pane_right_panel_visibility(self):
    pane = getattr(self, "main_pane", None)
    right_frame = getattr(self, "main_right_frame", None)
    if pane is None or right_frame is None:
        return

    should_show = self._should_show_right_panel()
    has_right = self._pane_has_child(pane, right_frame)

    try:
        if should_show and not has_right:
            pane.add(right_frame, weight=1)
        elif not should_show and has_right:
            pane.forget(right_frame)
    except Exception:
        pass


def _get_z2_free_status_panel_run_dir(self, *, require_xml: bool = True) -> Path | None:
    if not self._is_free_mode_session_context():
        return self._get_preferred_annotation_run_dir(require_xml=require_xml)

    try:
        route = str(self._get_workflow_route() or "").strip().lower()
        screen = str(self._coerce_free_mode_screen() or "").strip().lower()
        step = str(self._coerce_workflow_step() or "").strip().lower()
    except Exception:
        route = ""
        screen = ""
        step = ""

    explicit_candidates = (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.plate_dataset_run_var.get() or "").strip(),
    )

    if route == "auto" and screen == "workflow" and step in {"auto_input", "auto_start"}:
        auto_completed = bool(
            self._normalize_workflow_route_value(getattr(self, "_last_completed_workflow_route", "")) == "auto"
        )
        for candidate in explicit_candidates:
            safe_run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=require_xml)
            if safe_run_dir is None:
                continue
            try:
                manifest = self._load_annotation_run_manifest(safe_run_dir)
            except Exception:
                manifest = {}
            run_type = str(manifest.get("annotation_run_type") or "").strip().lower()
            if bool(manifest.get("manual_xml_template", False)) or run_type == "manual_template":
                continue
            if auto_completed or run_type == "auto_annotation":
                return safe_run_dir
        return None

    return self._get_preferred_annotation_run_dir(require_xml=require_xml)


def _build_z2_free_export_status_state(self) -> dict:
    run_dir = self._get_z2_free_status_panel_run_dir(require_xml=True)
    images_dir = self._resolve_existing_dir(str(self.plate_dataset_images_var.get() or "").strip())
    if images_dir is None:
        images_dir = self._resolve_existing_dir(str(self.input_dir_var.get() or "").strip())

    images_with_plates, total_plates = self._get_run_plate_annotation_counts(run_dir)
    annotation_state = (
        self._get_plate_annotation_package_export_state(run_dir=run_dir)
        if run_dir is not None
        else {
            "ok": False,
            "run_dir": None,
            "images_with_plates": 0,
            "total_plates": 0,
            "message": "Brak gotowego runu anotacji tablic.",
        }
    )
    annotation_images_with_plates = int(annotation_state.get("images_with_plates", 0) or 0)
    annotation_total_plates = int(annotation_state.get("total_plates", 0) or 0)
    if annotation_total_plates > int(total_plates or 0):
        images_with_plates = annotation_images_with_plates
        total_plates = annotation_total_plates
    try:
        current_run_dir = getattr(self, "current_annotation_run_dir", None)
        current_matches_run = bool(
            run_dir is not None
            and current_run_dir is not None
            and self._paths_equivalent(run_dir, current_run_dir)
        )
    except Exception:
        current_matches_run = False
    if current_matches_run and getattr(self, "current_annotations", None):
        current_images_with_plates, current_total_plates = self._count_plate_annotations(self.current_annotations)
        if int(current_total_plates or 0) > int(total_plates or 0):
            images_with_plates = int(current_images_with_plates or 0)
            total_plates = int(current_total_plates or 0)
    approval_state = self._get_run_plate_strict_approved_state(run_dir)
    approved_images = int(approval_state.get("approved_images", 0) or 0)
    approved_plates = int(approval_state.get("approved_plates", 0) or 0)
    total_images = int(approval_state.get("total_images", 0) or 0)
    skipped_images = max(0, int(total_images or 0) - int(approved_images or 0))
    dataset_min_approved_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    dataset_missing_approved_plates = max(
        0,
        int(dataset_min_approved_plates) - int(approved_plates or 0),
    )
    dataset_gate_ready = bool(approved_plates >= dataset_min_approved_plates)

    quality_info = CONFIG.describe_yolo_pose_dataset_quality(approved_plates)
    dataset_ready = bool(
        run_dir is not None
        and (Path(run_dir) / "annotations.xml").exists()
        and images_dir is not None
        and dataset_gate_ready
        and not bool(getattr(self, "is_processing", False))
    )
    annotation_ready = bool(
        annotation_state.get("ok")
        and not bool(getattr(self, "is_processing", False))
    )

    return {
        "run_dir": run_dir,
        "images_dir": images_dir,
        "images_with_plates": int(images_with_plates or 0),
        "total_plates": int(total_plates or 0),
        "approved_images": int(approved_images or 0),
        "approved_plates": int(approved_plates or 0),
        "total_images": int(total_images or 0),
        "skipped_images": int(skipped_images or 0),
        "dataset_min_approved_plates": int(dataset_min_approved_plates),
        "dataset_missing_approved_plates": int(dataset_missing_approved_plates),
        "dataset_gate_ready": bool(dataset_gate_ready),
        "dataset_ready": bool(dataset_ready),
        "annotation_ready": bool(annotation_ready),
        "quality_info": dict(quality_info or {}),
        "annotation_state": dict(annotation_state or {}),
    }


def _is_z2_free_export_choice_available(self) -> bool:
    if bool(getattr(self, "is_processing", False)):
        return False
    state = self._build_z2_free_export_status_state()
    return bool(state.get("dataset_ready") or state.get("annotation_ready"))


def _refresh_free_mode_manual_right_panel(self, *args, **kwargs):
    return z2_workflow_methods._refresh_free_mode_manual_right_panel(self, *args, **kwargs)


def _restore_right_panel_content_after_fullscreen(self):
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        return
    if not self._should_show_right_panel():
        return

    pane = getattr(self, "main_pane", None)
    right_frame = getattr(self, "main_right_frame", None)
    if pane is None or right_frame is None:
        return

    try:
        if not self._pane_has_child(pane, right_frame):
            pane.add(right_frame, weight=1)
    except Exception:
        pass

    # W fullscreen odswiezanie E2 celowo ukrywa prawy panel. Po wyjsciu
    # trzeba przebudowac nie tylko rame, ale tez jej wewnetrzne sekcje.
    try:
        self._refresh_step2_action_states(lightweight=False)
    except Exception:
        pass
    try:
        self._sync_approve_hint_wraplength()
    except Exception:
        pass
    try:
        self._normalize_approve_panel_order()
    except Exception:
        pass
    try:
        self._sync_right_panel_canvas_width()
        self._sync_right_panel_scrollregion()
    except Exception:
        pass
    try:
        self._schedule_main_pane_layout_refresh(force_defaults=True, delay_ms=40)
    except Exception:
        pass


def _schedule_right_panel_content_restore_after_fullscreen(self, delay_ms: int = 0):
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        return
    if not self._should_show_right_panel():
        return

    def _restore():
        self._restore_right_panel_content_after_fullscreen()

    try:
        if int(delay_ms or 0) > 0:
            self.frame.after(int(delay_ms), _restore)
        else:
            self.frame.after_idle(_restore)
    except Exception:
        _restore()


def _get_main_pane_width_limits(self) -> tuple[int, int]:
    left_content_req = 0
    right_content_req = 0
    approve_req = 0
    left_scrollbar_req = 10
    right_scrollbar_req = 10

    try:
        left_content_req = int(getattr(self, "left_settings_content", None).winfo_reqwidth() or 0)
    except Exception:
        left_content_req = 0
    try:
        right_content_req = int(getattr(self, "right_settings_content", None).winfo_reqwidth() or 0)
    except Exception:
        right_content_req = 0
    try:
        approve_req = int(getattr(self, "approve_btn_row", None).winfo_reqwidth() or 0)
    except Exception:
        approve_req = 0
    try:
        left_scrollbar_req = int(getattr(self, "left_settings_scrollbar", None).winfo_reqwidth() or 10)
    except Exception:
        left_scrollbar_req = 10
    try:
        right_scrollbar_req = int(getattr(self, "right_settings_scrollbar", None).winfo_reqwidth() or 10)
    except Exception:
        right_scrollbar_req = 10

    right_content_req = max(int(right_content_req), int(approve_req))
    left_counter_req = 0
    try:
        left_counter_req = int(self._get_preview_left_counter_min_width() or 0)
    except Exception:
        left_counter_req = 0
    left_base_req = max(
        int(left_content_req) + int(left_scrollbar_req) + 12,
        int(left_counter_req),
    )
    left_min = max(280, min(420, left_base_req))
    right_min = max(320, min(460, right_content_req + right_scrollbar_req + 20))
    return int(left_min), int(right_min)


def _schedule_main_pane_layout_refresh(self, *, force_defaults: bool = False, delay_ms: int = 0):
    if bool(getattr(self, "_preview_list_population_active", False)):
        self._main_pane_layout_deferred_for_preview_population = True
        self._main_pane_layout_deferred_force_defaults = bool(
            getattr(self, "_main_pane_layout_deferred_force_defaults", False) or force_defaults
        )
        return

    if bool(getattr(self, "_main_pane_layout_in_progress", False)):
        self._main_pane_layout_pending_force_defaults = bool(
            getattr(self, "_main_pane_layout_pending_force_defaults", False) or force_defaults
        )
        return

    pending = getattr(self, "_main_pane_layout_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
        self._main_pane_layout_after_id = None

    def _run():
        self._main_pane_layout_after_id = None
        self._apply_main_pane_layout(force_defaults=force_defaults)

    try:
        self._main_pane_layout_after_id = self.frame.after(max(0, int(delay_ms)), _run)
    except Exception:
        _run()


def _reset_main_pane_left_width_for_z2(self):
    try:
        self._main_pane_layout_initialized = False
    except Exception:
        pass
    try:
        self._schedule_main_pane_layout_refresh(force_defaults=True)
    except Exception:
        pass
    try:
        self.frame.after_idle(lambda: self._schedule_main_pane_layout_refresh(force_defaults=True))
    except Exception:
        pass


def _apply_main_pane_layout(self, *args, **kwargs):
    return z2_workflow_methods._apply_main_pane_layout(self, *args, **kwargs)


def _on_main_pane_configure(self, event=None):
    self._schedule_main_pane_layout_refresh(force_defaults=not bool(getattr(self, "_main_pane_layout_initialized", False)))


def _on_main_pane_button_press(self, event=None):
    return None


def _on_main_pane_drag_motion(self, event=None):
    return None


def _on_main_pane_drag_release(self, event=None):
    self._schedule_main_pane_layout_refresh(force_defaults=False)
    return None


def _sync_approve_hint_wraplength(self, event=None):
    host = getattr(self, "approve_btn_row", None)
    if host is None:
        return

    width = getattr(event, "width", 0) or host.winfo_width()
    if width <= 1:
        return

    wraplength = max(220, int(width) - 34)
    for widget_name in ("approve_context_lbl", "approve_gate_hint_lbl", "approve_breakdown_lbl"):
        label = getattr(self, widget_name, None)
        if label is None:
            continue
        try:
            if int(float(label.cget("wraplength") or 0)) != wraplength:
                label.configure(wraplength=wraplength)
        except Exception:
            pass


def _sync_workflow_copy_wraplength(self, event=None):
    fallback_host = getattr(self, "workflow_entry_shell_inner", None)
    try:
        fallback_width = int(getattr(event, "width", 0) or (fallback_host.winfo_width() if fallback_host is not None else 0) or 0)
    except Exception:
        fallback_width = 0

    for widget_name in (
        "route_badge_lbl",
        "route_summary_lbl",
        "workflow_action_hint_lbl",
        "workflow_start_intro_lbl",
        "manual_entry_hint_lbl",
        "manual_xml_template_hint_lbl",
        "manual_vehicle_assist_hint_lbl",
        "workflow_manual_vehicle_assist_hint_lbl",
        "workflow_conf_hint_lbl",
        "workflow_vehicle_model_hint_lbl",
        "workflow_input_hint_lbl",
        "followup_intro_lbl",
        "post_annotation_hint_lbl",
        "export_intro_lbl",
        "plate_export_status_lbl",
        "manual_stage_help_lbl",
        "manual_stage_status_lbl",
        "manual_stage_export_help_lbl",
    ):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            host = widget.master
            host_width = int(host.winfo_width() or 0) if host is not None else 0
        except Exception:
            host_width = 0

        if getattr(widget, "_wrap_container", None) is not None:
            self._refresh_bound_label_wraplength(widget)
            continue
        effective_width = host_width if host_width > 1 else fallback_width
        try:
            padding_px = int(getattr(widget, "_wrap_padding_px", 40) or 40)
        except Exception:
            padding_px = 40
        try:
            min_px = int(getattr(widget, "_wrap_min_px", 220) or 220)
        except Exception:
            min_px = 220
        wraplength = max(min_px, int(effective_width or 0) - padding_px)
        try:
            if int(float(widget.cget("wraplength") or 0)) != wraplength:
                widget.configure(wraplength=wraplength)
        except Exception:
            pass


def _widget_contains_point(self, widget, x_root: int, y_root: int) -> bool:
    if widget is None:
        return False
    try:
        wx = int(widget.winfo_rootx())
        wy = int(widget.winfo_rooty())
        return wx <= x_root < (wx + int(widget.winfo_width())) and wy <= y_root < (wy + int(widget.winfo_height()))
    except Exception:
        return False


def _get_preview_pointer_canvas_position(self):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None

    try:
        x_root = int(canvas.winfo_pointerx())
        y_root = int(canvas.winfo_pointery())
    except Exception:
        return None

    if not self._widget_contains_point(canvas, x_root, y_root):
        return None

    try:
        local_x = float(x_root - int(canvas.winfo_rootx()))
        local_y = float(y_root - int(canvas.winfo_rooty()))
        return (
            float(canvas.canvasx(local_x)),
            float(canvas.canvasy(local_y)),
        )
    except Exception:
        return None


def _mousewheel_units(self, event) -> int:
    return self._inertial_scroll.mousewheel_units(event)


def _on_preview_listbox_mousewheel(self, event):
    listbox = getattr(self, "preview_listbox", None)
    if listbox is None:
        return None
    if bool(getattr(self, "_plate_auto_scope_selection_mode_active", False)):
        try:
            listbox.focus_set()
        except Exception:
            pass
    units = self._inertial_scroll.mousewheel_units(event)
    if units == 0 or not self._inertial_scroll.listbox_can_scroll(listbox, units):
        return None
    self._inertial_scroll.queue_listbox_by_units(
        listbox,
        units,
        magnitude=self._inertial_scroll.mousewheel_magnitude(event),
    )
    return "break"


def _panel_canvas_overflows(self, canvas) -> bool:
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


def _scroll_panel_canvas_if_targeted(self, canvas, event):
    if self._inertial_scroll.scroll_canvas_if_targeted(
        canvas,
        event,
        pointer_widget=self.frame,
        overflow_checker=lambda: self._panel_canvas_overflows(canvas),
    ):
        return "break"
    return None


def _on_left_panel_global_mousewheel(self, event):
    return self._scroll_panel_canvas_if_targeted(
        getattr(self, "left_settings_canvas", None),
        event
    )


def _on_right_panel_global_mousewheel(self, event):
    return self._scroll_panel_canvas_if_targeted(
        getattr(self, "right_settings_canvas", None),
        event
    )


def _restore_scroll_canvas_focus(self, canvas):
    if canvas is None:
        return
    try:
        canvas.focus_set()
    except Exception:
        pass


def _redirect_child_mousewheel_to_canvas(self, event, canvas):
    if self._inertial_scroll.redirect_child_mousewheel_to_canvas(
        event,
        canvas,
        pointer_widget=self.frame,
        overflow_checker=lambda: self._panel_canvas_overflows(canvas),
    ):
        self._restore_scroll_canvas_focus(canvas)
        return "break"

    self._restore_scroll_canvas_focus(canvas)
    return "break"


def _bind_scroll_canvas_children(self, root, canvas):
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

    def _walk(widget):
        try:
            widget.bind("<MouseWheel>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
            widget.bind("<Button-4>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
            widget.bind("<Button-5>", lambda e, c=canvas: self._redirect_child_mousewheel_to_canvas(e, c), add="+")
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
                widget.bind("<ButtonRelease-1>", lambda _e, c=canvas: self._restore_scroll_canvas_focus(c), add="+")
            except Exception:
                pass
            if class_name == "TCombobox":
                try:
                    widget.bind("<<ComboboxSelected>>", lambda _e, c=canvas: self._restore_scroll_canvas_focus(c), add="+")
                except Exception:
                    pass

        for child in widget.winfo_children():
            _walk(child)

    _walk(root)


def _build_left_section_separator(self, parent, pady=(0, 0)):
    if parent is None:
        return None

    palette = getattr(self.app, "palette", {})
    host = tk.Frame(
        parent,
        height=4,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", "#252526"),
    )
    host.pack(fill=tk.X, pady=pady)
    host.pack_propagate(False)

    accent_line = tk.Frame(
        host,
        height=1,
        bd=0,
        highlightthickness=0,
        bg=palette.get("surface_info", palette.get("accent", "#0e639c")),
    )
    accent_line.pack(fill=tk.X, side=tk.TOP)

    shadow_line = tk.Frame(
        host,
        height=1,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_border", palette.get("border", "#3c3c3c")),
    )
    shadow_line.pack(fill=tk.X, side=tk.TOP, pady=(1, 0))

    self._left_section_separators.append(
        {
            "host": host,
            "accent": accent_line,
            "shadow": shadow_line,
        }
    )
    return host


def _build_left_path_row(
    self,
    parent,
    textvariable,
    *,
    state: str = "normal",
    button_text: str | None = None,
    button_command=None,
):
    row = ttk.Frame(parent, style="Panel.TFrame")
    row.pack(fill=tk.X, pady=(2, 8))
    row.columnconfigure(0, weight=1)
    row.columnconfigure(1, minsize=int(getattr(self, "_left_path_action_minsize", 140)))

    entry = ttk.Entry(row, textvariable=textvariable, state=state)
    entry.grid(row=0, column=0, sticky="ew")

    button = None
    if button_text and button_command is not None:
        button = ttk.Button(
            row,
            text=button_text,
            width=int(getattr(self, "_left_path_button_width", 15)),
            command=button_command,
        )
        button.grid(row=0, column=1, sticky="e", padx=(8, 0))

    return row, entry, button


def _bind_workflow_card(self, card_widget, route: str):
    if card_widget is None:
        return

    widgets = [card_widget]
    try:
        widgets.extend(list(card_widget.winfo_children()))
    except Exception:
        pass

    for widget in widgets:
        try:
            widget.bind(
                "<Button-1>",
                lambda _event, selected_route=route: self._select_workflow_route(selected_route),
                add="+",
            )
            widget.bind(
                "<Enter>",
                lambda _event, selected_route=route: self._set_workflow_route_card_hover(selected_route, True),
                add="+",
            )
            widget.bind(
                "<Leave>",
                lambda _event, selected_route=route: self._set_workflow_route_card_hover(selected_route, False),
                add="+",
            )
        except Exception:
            pass


def _bind_preview_sort_tile(self, tile_widget, sort_mode: str):
    if tile_widget is None:
        return

    widgets = [tile_widget]
    try:
        widgets.extend(list(tile_widget.winfo_children()))
    except Exception:
        pass

    for widget in widgets:
        try:
            widget.configure(cursor="hand2")
        except Exception:
            pass
        try:
            widget.bind(
                "<Button-1>",
                lambda _event, selected_sort=sort_mode: self._set_preview_list_sort_mode(selected_sort),
                add="+",
            )
        except Exception:
            pass


def _set_preview_list_sort_mode(self, sort_mode: str):
    if bool(getattr(self, "_plate_auto_scope_modal_open", False)):
        return
    normalized = self._normalize_preview_list_sort_mode(sort_mode)
    if not normalized:
        return
    current_sort = self._normalize_preview_list_sort_mode()
    if normalized == current_sort:
        self._refresh_preview_list_legend_theme()
        return
    self.preview_list_sort_var.set(normalized)
    self._invalidate_preview_list_frozen_order()

    # Sortowanie zmienia tylko kolejność listy. Nie może przełączać podglądu.
    self._suppress_preview_reload_on_list_select = True
    try:
        self.frame.after(250, lambda: setattr(self, "_suppress_preview_reload_on_list_select", False))
    except Exception:
        pass

    try:
        entries_count = int(len(self.current_annotations or []))
    except Exception:
        entries_count = 0
    can_reuse_state_cache = isinstance(getattr(self, "_preview_list_render_state_cache", None), dict)
    if entries_count >= 800 or self._should_use_async_preview_list_population(entries_count):
        self._populate_preview_list_async(
            preserve_selection=True,
            render_current=False,
            batch_size=1200,
            invalidate_runtime=False,
            rebuild_state_cache=not can_reuse_state_cache,
            recolor_rows=True,
            refresh_summary=True,
            lightweight_summary=True,
            show_population_state=False,
        )
    else:
        self._refresh_preview_list(
            preserve_selection=True,
            render_current=False,
            invalidate_runtime=False,
            rebuild_state_cache=not can_reuse_state_cache,
            lightweight_summary=True,
            recolor_rows=True,
            refresh_summary=True,
        )
    self._refresh_preview_list_legend_theme()
    self._update_preview_toolbar_state(refresh_summary=False)


def _normalize_preview_list_sort_mode(self, sort_mode: str | None = None) -> str:
    normalized = str(
        self.preview_list_sort_var.get() if sort_mode is None else sort_mode
    ).strip()
    if not normalized:
        return ""
    allowed_modes = set(getattr(self, "_preview_list_sort_options", ()) or ())
    if normalized in allowed_modes:
        return normalized
    return str(next(iter(getattr(self, "_preview_list_sort_options", ()) or ("",)), "") or "")


def _get_preview_metric_filter_input_thresholds(self) -> tuple[float, float]:
    conf_raw = 0.0
    fit_raw = 0.0
    if hasattr(self, "preview_filter_conf_var"):
        try:
            conf_raw = self.preview_filter_conf_var._tk.globalgetvar(self.preview_filter_conf_var._name)
        except Exception:
            conf_raw = 0.0
    if hasattr(self, "preview_filter_fit_var"):
        try:
            fit_raw = self.preview_filter_fit_var._tk.globalgetvar(self.preview_filter_fit_var._name)
        except Exception:
            fit_raw = 0.0

    if str(conf_raw or "").strip() == "":
        conf_raw = 0.0
    if str(fit_raw or "").strip() == "":
        fit_raw = 0.0

    conf_threshold = self._normalize_preview_metric_threshold(conf_raw)
    fit_threshold = self._normalize_preview_metric_threshold(fit_raw)
    return conf_threshold, fit_threshold


def _get_preview_metric_filter_thresholds(self) -> tuple[float, float]:
    conf_threshold = self._normalize_preview_metric_threshold(
        getattr(self, "_preview_filter_conf_applied", 0.0)
    )
    fit_threshold = self._normalize_preview_metric_threshold(
        getattr(self, "_preview_filter_fit_applied", 0.0)
    )
    return conf_threshold, fit_threshold


def _preview_annotation_passes_metric_filters(self, ann) -> bool:
    conf_threshold, fit_threshold = self._get_preview_metric_filter_thresholds()
    if conf_threshold <= 0.0 and fit_threshold <= 0.0:
        return True

    quality = self._get_preview_annotation_quality_summary(ann)
    if int(quality.get("plate_count", 0) or 0) <= 0:
        return False

    if conf_threshold > 0.0 and float(quality.get("min_confidence", 0.0) or 0.0) < conf_threshold:
        return False
    if fit_threshold > 0.0:
        if int(quality.get("fit_count", 0) or 0) <= 0:
            return False
        if float(quality.get("min_fit_score", 0.0) or 0.0) < fit_threshold:
            return False
    return True


def _filter_preview_list_entries(
    self,
    entries: list[tuple[int, ImageAnnotation]],
) -> list[tuple[int, ImageAnnotation]]:
    filtered_entries = list(entries or [])
    if not self._is_free_mode_session_context():
        try:
            project_approved_lookup = set(self._get_campaign_hidden_project_approved_filenames_runtime() or set())
        except Exception:
            project_approved_lookup = set()
        if project_approved_lookup:
            filtered_entries = [
                (actual_idx, ann)
                for actual_idx, ann in filtered_entries
                if str(getattr(ann, "filename", "") or "").strip().lower() not in project_approved_lookup
            ]

    conf_threshold, fit_threshold = self._get_preview_metric_filter_thresholds()
    if conf_threshold <= 0.0 and fit_threshold <= 0.0:
        return filtered_entries
    return [
        (actual_idx, ann)
        for actual_idx, ann in filtered_entries
        if self._preview_annotation_passes_metric_filters(ann)
    ]


def _reset_preview_metric_filters(self):
    self.preview_filter_conf_var.set(0.0)
    self.preview_filter_fit_var.set(0.0)
    self._preview_filter_conf_applied = 0.0
    self._preview_filter_fit_applied = 0.0
    self._invalidate_preview_list_frozen_order()
    self._refresh_preview_list(preserve_selection=True, render_current=False)
    try:
        self._load_current_preview_selection(reset_view=False, selection_changed=False)
    except Exception:
        pass
    self._refresh_preview_list_summary()
    self._update_preview_toolbar_state()
    self._refresh_preview_filter_bar_state()


def _apply_preview_metric_filters(self):
    conf_threshold, fit_threshold = self._get_preview_metric_filter_input_thresholds()
    previous_conf = self._normalize_preview_metric_threshold(getattr(self, "_preview_filter_conf_applied", 0.0))
    previous_fit = self._normalize_preview_metric_threshold(getattr(self, "_preview_filter_fit_applied", 0.0))

    self._preview_filter_conf_applied = conf_threshold
    self._preview_filter_fit_applied = fit_threshold
    prospective_entries = self._filter_preview_list_entries(
        self._build_preview_list_sorted_entries(self._normalize_preview_list_sort_mode())
    )
    total_annotations = len(list(self.current_annotations or []))
    if total_annotations > 0 and not prospective_entries and (conf_threshold > 0.0 or fit_threshold > 0.0):
        self._preview_filter_conf_applied = previous_conf
        self._preview_filter_fit_applied = previous_fit
        try:
            self.preview_filter_conf_var.set(previous_conf)
            self.preview_filter_fit_var.set(previous_fit)
        except Exception:
            pass
        self._update_preview_edit_status(
            "Ten filtr ukryłby wszystkie obrazy. Lista została bez zmian, poluzuj progi Det/Fit."
        )
        self._refresh_preview_filter_bar_state()
        return

    try:
        self.preview_filter_conf_var.set(conf_threshold)
        self.preview_filter_fit_var.set(fit_threshold)
    except Exception:
        pass
    self._invalidate_preview_list_frozen_order()
    self._refresh_preview_list(preserve_selection=True, render_current=False)
    try:
        self._load_current_preview_selection(reset_view=False, selection_changed=False)
    except Exception:
        pass
    self._refresh_preview_list_summary()
    self._update_preview_toolbar_state()
    self._update_preview_edit_status(
        f"Zastosowano filtr: Det >= {conf_threshold:.2f} | Fit >= {fit_threshold:.2f}."
        if (conf_threshold > 0.0 or fit_threshold > 0.0)
        else "Filtr listy został wyłączony."
    )
    self._refresh_preview_filter_bar_state()


def _open_preview_metric_filter_modal(self, *args, **kwargs):
    return z2_workflow_methods._open_preview_metric_filter_modal(self, *args, **kwargs)


def _refresh_preview_filter_bar_state(self):
    palette = getattr(self.app, "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#0e639c")
    success = palette.get("success", "#27ae60")
    warning = palette.get("warning", "#f39c12")

    input_conf, input_fit = self._get_preview_metric_filter_input_thresholds()
    applied_conf, applied_fit = self._get_preview_metric_filter_thresholds()
    filter_active = bool(applied_conf > 0.0 or applied_fit > 0.0)

    items = (
        (
            "preview_filter_bar_setup_item",
            "preview_filter_bar_setup_badge",
            "preview_filter_bar_setup_lbl",
            "preview_filter_bar_setup_value_lbl",
            accent,
            f"D>={input_conf:.2f} F>={input_fit:.2f}",
            False,
        ),
        (
            "preview_filter_bar_apply_item",
            "preview_filter_bar_apply_badge",
            "preview_filter_bar_apply_lbl",
            "preview_filter_bar_apply_value_lbl",
            success,
            ("ON" if filter_active else "--"),
            filter_active,
        ),
        (
            "preview_filter_bar_reset_item",
            "preview_filter_bar_reset_badge",
            "preview_filter_bar_reset_lbl",
            "preview_filter_bar_reset_value_lbl",
            warning,
            "",
            False,
        ),
    )

    for item_name, badge_name, label_name, value_name, accent_color, value_text, active in items:
        item = getattr(self, item_name, None)
        badge = getattr(self, badge_name, None)
        label = getattr(self, label_name, None)
        value_lbl = getattr(self, value_name, None)
        item_bg = blend_hex_colors(accent_color, panel_bg, 0.18) if active else panel_bg
        item_border = accent_color if active else panel_border
        main_fg = "#1b1b1b" if active else fg
        value_fg = "#1b1b1b" if active else accent_color
        try:
            if item is not None:
                item.configure(bg=item_bg, highlightbackground=item_border, highlightcolor=item_border)
            if badge is not None:
                badge.configure(bg=accent_color, fg=("#1b1b1b" if accent_color != accent else panel_bg))
            if label is not None:
                label.configure(bg=item_bg, fg=main_fg)
            if value_lbl is not None:
                value_lbl.configure(bg=item_bg, fg=value_fg, text=str(value_text or ""))
        except Exception:
            pass


def _invalidate_preview_list_frozen_order(self) -> None:
    self._preview_list_frozen_sort_mode = ""
    self._preview_list_frozen_filename_order = []
    self._preview_list_frozen_context_key = ""
    self._preview_list_frozen_bucket_snapshot = {}


def _get_preview_list_context_key(self) -> str:
    conf_threshold, fit_threshold = self._get_preview_metric_filter_thresholds()
    filters_key = f"|conf>={conf_threshold:.2f}|fit>={fit_threshold:.2f}"
    safe_run_dir = self._resolve_safe_annotation_run_dir(
        getattr(self, "current_annotation_run_dir", None),
        require_xml=False,
    )
    if safe_run_dir is None:
        safe_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "last_staging_run_dir", None),
            require_xml=False,
    )
    if safe_run_dir is not None:
        return f"{str(safe_run_dir)}{filters_key}"

    current_xml_path = self._get_current_annotation_xml_path()
    if current_xml_path is not None:
        try:
            return f"{str(current_xml_path.parent)}{filters_key}"
        except Exception:
            return f"{str(current_xml_path)}{filters_key}"
    return filters_key


def _store_preview_list_frozen_order(
    self,
    sort_mode: str,
    entries: list[tuple[int, ImageAnnotation]],
) -> None:
    self._preview_list_frozen_sort_mode = self._normalize_preview_list_sort_mode(sort_mode)
    self._preview_list_frozen_context_key = self._get_preview_list_context_key()
    self._preview_list_frozen_filename_order = [
        str(getattr(ann, "filename", "") or "").strip().lower()
        for _actual_idx, ann in entries
    ]
    self._preview_list_frozen_bucket_snapshot = {
        str(getattr(ann, "filename", "") or "").strip().lower(): self._preview_annotation_sort_bucket_for_mode(
            ann,
            sort_mode,
        )
        for _actual_idx, ann in entries
    }


def _set_workflow_route_card_hover(self, route: str, enabled: bool):
    next_hover = route if enabled else None
    if getattr(self, "_workflow_route_hover_mode", None) == next_hover:
        return
    self._workflow_route_hover_mode = next_hover
    self._refresh_workflow_route_cards(refresh_content=False)


def _build_z2_action_context(self) -> Z2ActionContext:
    campaign_step = 0
    iteration_target = ""
    has_plate_model = False
    campaign_repair_mode = False

    try:
        from ..campaign_manager import CAMPAIGN

        if not self._is_free_mode_session_context() and CAMPAIGN.get_active_project_name():
            campaign_step = int(CAMPAIGN.get_current_step() or 0)
            iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
            project_plate_model = str(CAMPAIGN.get_global_model("plate") or "").strip()
            has_plate_model = bool(project_plate_model and Path(project_plate_model).exists())
            campaign_repair_mode = bool(
                (iteration_target == "char" and self._is_campaign_char_repair_return_mode())
                or (iteration_target == "plate" and self._is_campaign_plate_step4_repair_return_mode())
            )
    except Exception:
        campaign_step = 0
        iteration_target = ""
        campaign_repair_mode = False

    if not has_plate_model:
        plate_model_value = str(self.plate_custom_var.get() or "").strip()
        has_plate_model = bool(plate_model_value and Path(plate_model_value).exists())

    preferred_run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    has_existing_run = preferred_run_dir is not None
    manual_review_active = bool(self._manual_review_active and has_existing_run)
    export_ready = False
    if preferred_run_dir is not None:
        try:
            export_ready = bool(
                self._get_run_plate_strict_approved_state(preferred_run_dir).get("ok")
            )
        except Exception:
            export_ready = False

    return Z2ActionContext(
        mode=("free" if self._is_free_mode_session_context() else "campaign"),
        campaign_step=campaign_step,
        iteration_target=iteration_target,
        campaign_repair_mode=campaign_repair_mode,
        route=self._get_workflow_route(),
        input_dir=str(self.input_dir_var.get() or "").strip(),
        has_plate_model=bool(has_plate_model),
        has_existing_run=bool(has_existing_run),
        has_manual_review=manual_review_active,
        has_export_ready_run=export_ready,
        is_processing=bool(self.is_processing),
        auto_vehicle_choice=self._get_auto_vehicle_choice(),
    )


def _get_z2_primary_actions(self) -> dict[str, object]:
    return build_z2_primary_actions()


def _get_z2_secondary_actions(self) -> dict[str, object]:
    return build_z2_secondary_actions()


def _activate_z2_secondary_action(self, action_id: str) -> None:
    action = self._get_z2_secondary_actions().get(str(action_id or "").strip())
    if action is None:
        return

    ctx = self._build_z2_action_context()
    if not action.is_available(ctx) or not action.is_enabled(ctx):
        return

    action.activate(self, ctx)


def _select_workflow_route(self, route: str):
    if bool(getattr(self, "_plate_auto_scope_modal_open", False)):
        return
    normalized_route = self._normalize_workflow_route_value(route)
    if normalized_route not in {"auto", "manual"}:
        return

    action = self._get_z2_primary_actions().get(normalized_route)
    if action is None:
        return

    ctx = self._build_z2_action_context()
    if not action.is_available(ctx) or not action.is_enabled(ctx):
        return

    action.activate(self, ctx)


def _build_workflow_step_card(self, parent):
    palette = getattr(self.app, "palette", {})
    border = blend_hex_colors(
        palette.get("success", "#4ec9b0"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.18,
    )
    return tk.Frame(
        parent,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        padx=10,
        pady=8,
    )


def _register_workflow_step_card(
    self,
    key: str,
    card,
    *,
    title=None,
    labels=None,
    child_frames=None,
    step_keys=None,
    style_targets=None,
):
    if card is None:
        return
    self._workflow_step_cards.append(
        {
            "key": str(key or "").strip(),
            "card": card,
            "title": title,
            "labels": list(labels or []),
            "child_frames": list(child_frames or []),
            "step_keys": {str(item).strip() for item in (step_keys or []) if str(item).strip()},
            "style_targets": list(style_targets or []),
        }
    )


def _refresh_bound_label_wraplength(self, widget) -> None:
    if widget is None or not isinstance(widget, tk.Label):
        return

    container = getattr(widget, "_wrap_container", None)
    if container is None:
        return

    try:
        width = int(container.winfo_width() or 0)
    except Exception:
        width = 0
    if width <= 1:
        return

    try:
        padding_px = int(getattr(widget, "_wrap_padding_px", 44) or 44)
    except Exception:
        padding_px = 44
    try:
        min_px = int(getattr(widget, "_wrap_min_px", 220) or 220)
    except Exception:
        min_px = 220

    target_wrap = max(min_px, width - padding_px)
    try:
        current_wrap = int(float(widget.cget("wraplength") or 0))
    except Exception:
        current_wrap = 0

    if abs(current_wrap - target_wrap) > 2:
        try:
            widget.configure(wraplength=target_wrap)
        except Exception:
            pass


def _bind_label_wrap_to_container(
    self,
    widget,
    container,
    *,
    padding_px: int = 44,
    min_px: int = 220,
) -> None:
    if widget is None or container is None or not isinstance(widget, tk.Label):
        return

    widget._wrap_container = container
    widget._wrap_padding_px = int(padding_px)
    widget._wrap_min_px = int(min_px)

    if not bool(getattr(widget, "_wrap_container_bound", False)):
        try:
            container.bind(
                "<Configure>",
                lambda _event, _widget=widget: self._refresh_bound_label_wraplength(_widget),
                add="+",
            )
            widget._wrap_container_bound = True
        except Exception:
            pass

    try:
        self.frame.after_idle(lambda _widget=widget: self._refresh_bound_label_wraplength(_widget))
    except Exception:
        pass


def _apply_workflow_step_widget_style(self, *args, **kwargs):
    return z2_workflow_methods._apply_workflow_step_widget_style(self, *args, **kwargs)


def _refresh_workflow_button_styles(self, *args, **kwargs):
    return z2_workflow_methods._refresh_workflow_button_styles(self, *args, **kwargs)


def _refresh_workflow_progress_style(self, background: str | None = None):
    progress = getattr(self, "progress", None)
    if progress is None:
        return

    palette = getattr(self.app, "palette", {})
    card_bg = str(
        background
        or getattr(getattr(self, "workflow_start_section", None), "cget", lambda _key: None)("bg")
        or palette.get("panel_alt", palette.get("panel", "#252526"))
    )
    field = palette.get("field", palette.get("panel", "#252526"))
    success = palette.get("success", "#4ec9b0")
    trough = blend_hex_colors(card_bg, field, 0.6)
    fill = blend_hex_colors(success, card_bg, 0.1)

    try:
        progress.configure(
            trough_color=trough,
            fill_color=fill,
            bg=card_bg,
        )
    except Exception:
        pass


def _refresh_workflow_step_cards(self):
    cards = getattr(self, "_workflow_step_cards", [])
    if not cards:
        return

    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel_alt", palette.get("panel", "#252526"))
    hover_bg = palette.get("button_hover", panel)
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    active_bg = blend_hex_colors(panel, hover_bg, 0.30)
    current_step = self._coerce_workflow_step()
    compact_card_padding = bool(
        not self._is_free_mode_session_context()
        and getattr(self, "_compact_left_column_layout", False)
    )
    inline_manual_vehicle_block = bool(
        self._is_free_mode_session_context()
        and self._get_workflow_route() == "manual"
        and self._get_manual_entry_mode() == "new"
        and current_step == "manual_input"
        and self._manual_vehicle_assist_enabled()
    )

    for entry in cards:
        card = entry.get("card")
        if card is None:
            continue

        step_keys = entry.get("step_keys", set())
        card_key = str(entry.get("key") or "").strip()
        is_inline_manual_vehicle_card = bool(
            inline_manual_vehicle_block
            and card_key in {"workflow_vehicle_model", "workflow_conf"}
        )
        is_active = bool(current_step and current_step in step_keys and self._widget_is_packed(card))
        card_bg = active_bg if (is_active or is_inline_manual_vehicle_card) else panel
        card_border = card_bg if is_inline_manual_vehicle_card else panel_border
        title_fg = fg

        try:
            card.configure(
                bg=card_bg,
                highlightbackground=card_border,
                highlightcolor=card_border,
                highlightthickness=(0 if is_inline_manual_vehicle_card else 1),
                padx=(10 if is_inline_manual_vehicle_card else 14),
                pady=(2 if is_inline_manual_vehicle_card else (4 if compact_card_padding else 12)),
            )
        except Exception:
            pass

        title_widget = entry.get("title")
        if title_widget is not None:
            try:
                title_widget.configure(bg=card_bg, fg=title_fg, font=("Segoe UI Semibold", 11))
            except Exception:
                pass

        for child_frame in entry.get("child_frames", []):
            if child_frame is None:
                continue
            try:
                child_frame.configure(bg=card_bg)
            except Exception:
                pass

        for label in entry.get("labels", []):
            if label is None:
                continue
            try:
                label.configure(bg=card_bg)
            except Exception:
                pass
            try:
                tone = getattr(label, "_inline_tone", "muted")
                emphasis = getattr(label, "_inline_emphasis", False)
                text_value = None
                try:
                    if not label.cget("textvariable"):
                        text_value = label.cget("text")
                except Exception:
                    text_value = label.cget("text")
                self._set_inline_label_state(label, text=text_value, tone=tone, emphasis=emphasis)
            except Exception:
                pass

        for target in entry.get("style_targets", []):
            if not isinstance(target, dict):
                continue
            self._apply_workflow_step_widget_style(
                target.get("widget"),
                str(target.get("kind") or "").strip(),
                str(target.get("style") or "").strip(),
                card_bg,
            )

        if str(entry.get("key") or "").strip() == "workflow_manual_stage":
            self._refresh_manual_stage_export_box_style()

        if str(entry.get("key") or "").strip() == "workflow_start":
            self._refresh_workflow_progress_style(card_bg)


def _widget_is_packed(widget) -> bool:
    try:
        return str(widget.winfo_manager()) == "pack"
    except Exception:
        return False


def _set_widget_packed(self, widget, visible: bool, **pack_kwargs):
    if widget is None:
        return

    try:
        is_packed = str(widget.winfo_manager()) == "pack"
    except Exception:
        is_packed = False

    safe_pack_kwargs = dict(pack_kwargs)
    for ref_key in ("before", "after"):
        ref_widget = safe_pack_kwargs.get(ref_key)
        if ref_widget is None:
            continue
        try:
            if str(ref_widget.winfo_manager()) != "pack":
                safe_pack_kwargs.pop(ref_key, None)
        except Exception:
            safe_pack_kwargs.pop(ref_key, None)

    if visible:
        if not is_packed:
            try:
                widget.pack(**safe_pack_kwargs)
            except tk.TclError:
                safe_pack_kwargs.pop("before", None)
                safe_pack_kwargs.pop("after", None)
                widget.pack(**safe_pack_kwargs)
        elif safe_pack_kwargs:
            try:
                widget.pack_configure(**safe_pack_kwargs)
            except tk.TclError:
                safe_pack_kwargs.pop("before", None)
                safe_pack_kwargs.pop("after", None)
                try:
                    widget.pack_configure(**safe_pack_kwargs)
                except Exception:
                    pass
    elif is_packed:
        widget.pack_forget()


def _normalize_workflow_start_section_order(self) -> None:
    ordered_widgets = [
        (getattr(self, "workflow_start_title_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 6)}),
        (getattr(self, "workflow_start_intro_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (4, 0)}),
        (getattr(self, "workflow_start_manual_vehicle_assist_check", None), {"anchor": tk.W, "pady": (6, 2)}),
        (getattr(self, "workflow_start_manual_vehicle_assist_hint_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (0, 6)}),
        (getattr(self, "start_btn_row", None), {"fill": tk.X, "pady": (6, 0)}),
        (getattr(self, "progress_info_row", None), {"fill": tk.X, "pady": (10, 4)}),
        (getattr(self, "workflow_start_action_hint_lbl", None), {"anchor": tk.W, "fill": tk.X, "pady": (6, 0)}),
    ]

    visible_widgets: list[tuple[object, dict]] = []
    for widget, pack_kwargs in ordered_widgets:
        if widget is None:
            continue
        try:
            if str(widget.winfo_manager()) == "pack":
                visible_widgets.append((widget, pack_kwargs))
        except Exception:
            continue

    if not visible_widgets:
        return

    for widget, _pack_kwargs in visible_widgets:
        try:
            widget.pack_forget()
        except Exception:
            continue

    for widget, pack_kwargs in visible_widgets:
        try:
            widget.pack(**pack_kwargs)
        except Exception:
            continue


def _normalize_approve_panel_order(self):
    row = getattr(self, "approve_btn_row", None)
    if row is None:
        return

    ordered_widgets = [
        (getattr(self, "approve_context_box", None), {"fill": tk.X, "pady": (0, 10)}),
        (getattr(self, "approve_hint_box", None), {"fill": tk.X, "pady": (0, 10)}),
        (getattr(self, "approve_breakdown_box", None), {"fill": tk.X, "pady": (0, 10)}),
        (getattr(self, "approve_btn_frame", None), {"fill": tk.X}),
    ]

    visible_widgets = []
    for widget, pack_kwargs in ordered_widgets:
        if widget is None:
            continue
        try:
            if str(widget.winfo_manager()) == "pack":
                visible_widgets.append((widget, pack_kwargs))
        except Exception:
            continue

    if not visible_widgets:
        return

    for widget, _pack_kwargs in visible_widgets:
        try:
            widget.pack_forget()
        except Exception:
            continue

    for widget, pack_kwargs in visible_widgets:
        try:
            widget.pack(**pack_kwargs)
        except Exception:
            continue


def _measure_visible_pack_height(self, widget) -> int:
    if widget is None:
        return 0

    try:
        children = [
            child
            for child in widget.pack_slaves()
            if str(child.winfo_manager()) == "pack"
        ]
    except Exception:
        children = []

    if not children:
        try:
            return int(widget.winfo_reqheight() or 0)
        except Exception:
            return 0

    total = 0
    for child in children:
        try:
            grandchildren = [
                grandchild
                for grandchild in child.pack_slaves()
                if str(grandchild.winfo_manager()) == "pack"
            ]
        except Exception:
            grandchildren = []

        if grandchildren:
            child_height = self._measure_visible_pack_height(child)
        else:
            try:
                child_height = int(child.winfo_reqheight() or 0)
            except Exception:
                child_height = 0

        try:
            child_pady = int(float(str(child.cget("pady") or 0)))
        except Exception:
            child_pady = 0
        try:
            child_ipady = int(float(str(child.cget("ipady") or 0)))
        except Exception:
            child_ipady = 0
        try:
            pack_info = child.pack_info()
            pady = pack_info.get("pady", 0)
        except Exception:
            pady = 0

        if isinstance(pady, (tuple, list)) and len(pady) >= 2:
            padding_total = int(pady[0] or 0) + int(pady[1] or 0)
        else:
            try:
                padding_total = int(float(str(pady).split()[0])) * 2
            except Exception:
                padding_total = 0

        total += child_height + padding_total + child_pady * 2 + child_ipady * 2
    return int(total)


def _scroll_left_panel_to_widget(self, widget):
    canvas = getattr(self, "left_settings_canvas", None)
    content = getattr(self, "left_settings_content", None)
    if canvas is None or content is None or widget is None:
        return

    def scroll_after_layout():
        self._left_panel_scroll_after_id = None
        try:
            bbox = canvas.bbox("all")
            if not bbox:
                return
            top_y = 0
            current = widget
            while current is not None and current != content:
                top_y += int(current.winfo_y())
                parent_name = str(current.winfo_parent() or "").strip()
                if not parent_name:
                    break
                try:
                    current = current.nametowidget(parent_name)
                except Exception:
                    current = None
            total_height = max(1, int(bbox[3] - bbox[1]))
            fraction = max(0.0, min(1.0, float(top_y) / float(total_height)))
            canvas.yview_moveto(fraction)
        except tk.TclError:
            pass

    # Let pending geometry settle rather than recursively pumping the Tk queue.
    pending = getattr(self, "_left_panel_scroll_after_id", None)
    if pending:
        self.frame.after_cancel(pending)
    self._left_panel_scroll_after_id = self.frame.after_idle(scroll_after_layout)
