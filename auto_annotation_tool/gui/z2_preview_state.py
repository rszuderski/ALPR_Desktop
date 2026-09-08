#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 preview state, list and restore helpers extracted from tab_annotation.py."""

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

def _arm_preview_super_perf_probe(self, reason: str, *, step: int | None = None) -> str:
    trace_id = int(getattr(self, "_preview_super_perf_trace_id", 0) or 0) + 1
    now = time.perf_counter()
    label = f"super-qe-{trace_id}"
    self._preview_super_perf_trace_id = trace_id
    self._preview_super_perf_label = label
    self._preview_super_perf_until = now + 6.0
    self._preview_super_perf_switch_started_at = now
    self._preview_super_perf_waiting_first_press = True
    try:
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            canvas._perf_probe_label_value = label
            canvas._perf_probe_until = now + 6.0
    except Exception:
        pass
    try:
        logger.info(
            "[Z2 DRAG PERF] switch.begin label=%s reason=%s step=%s",
            label,
            str(reason or "-"),
            "" if step is None else int(step),
        )
    except Exception:
        pass
    return label


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


def _resolve_preview_image_path(self, ann) -> Path | None:
    filename = str(getattr(ann, "filename", "") or "").strip()
    if not filename:
        return None

    image_map = dict(getattr(self, "_preview_image_path_map", {}) or {})
    mapped_path = image_map.get(filename)
    if mapped_path is not None:
        try:
            mapped_candidate = Path(mapped_path)
            if mapped_candidate.exists():
                return mapped_candidate
        except Exception:
            pass

    current_input_dir = getattr(self, "current_input_dir", None)
    if current_input_dir is None:
        return None

    try:
        candidate = Path(current_input_dir) / filename
    except Exception:
        return None
    return candidate


def _format_preview_dpi_value(dpi_value) -> str:
    if not dpi_value:
        return ""
    try:
        if isinstance(dpi_value, (list, tuple)) and len(dpi_value) >= 2:
            x_dpi = float(dpi_value[0] or 0.0)
            y_dpi = float(dpi_value[1] or 0.0)
        else:
            x_dpi = y_dpi = float(dpi_value or 0.0)
    except Exception:
        return ""
    if x_dpi <= 0.0 or y_dpi <= 0.0:
        return ""

    def _fmt(value: float) -> str:
        rounded = int(round(float(value)))
        return str(rounded) if abs(float(value) - rounded) < 0.25 else f"{float(value):.1f}"

    if abs(x_dpi - y_dpi) < 0.25:
        return f"DPI: {_fmt(x_dpi)}"
    return f"DPI: {_fmt(x_dpi)}x{_fmt(y_dpi)}"


def _get_preview_image_file_metadata(self, *args, **kwargs):
    return z2_workflow_methods._get_preview_image_file_metadata(self, *args, **kwargs)


def _get_preview_image_dpi_text(self, ann) -> str:
    metadata = self._get_preview_image_file_metadata(ann)
    return str(metadata.get("dpi_text", "") or "DPI: brak pliku")


def _is_plate_detection_label(label) -> bool:
    return str(label or "").strip().lower() in CONFIG.PLATE_LABELS


def _preview_list_item_text(
    self,
    ann,
    *,
    display_index: int | None = None,
    total_count: int | None = None,
    lightweight: bool = False,
) -> str:
    cached_state = self._get_preview_list_render_state(ann)
    status_text = (
        str(cached_state.get("status_text", "") or "")
        if cached_state is not None
        else ""
    )
    if not status_text:
        status_text = self._preview_annotation_status_tag(ann)
    if display_index is None:
        order_text = "--."
    else:
        width = max(2, len(str(max(1, int(total_count or (display_index + 1))))))
        order_text = f"{int(display_index) + 1:0{width}d}."
    reused = (
        bool(cached_state.get("reused"))
        if cached_state is not None
        else self._preview_annotation_is_reused_from_previous_manual(ann)
    )
    reuse_prefix = f"{self._campaign_reuse_manual_badge()} " if reused else ""
    if lightweight:
        return f"{order_text} [{status_text}] {reuse_prefix}{ann.filename}"
    quality = self._get_preview_annotation_quality_summary(ann)
    metrics_prefix = ""
    if int(quality.get("plate_count", 0) or 0) > 0:
        metric_parts = [f"D{float(quality.get('min_confidence', 0.0) or 0.0):.2f}"]
        if int(quality.get("fit_count", 0) or 0) > 0:
            metric_parts.append(f"F{float(quality.get('min_fit_score', 0.0) or 0.0):.2f}")
        metrics_prefix = f"{' '.join(metric_parts)} | "
    return f"{order_text} [{status_text}] {reuse_prefix}{metrics_prefix}{ann.filename}"


def _preview_annotation_is_reused_from_previous_manual(self, ann) -> bool:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return bool(cached_state.get("reused"))

    filename = str(getattr(ann, "filename", "") or "").strip()
    if not filename:
        return False
    return filename in set(getattr(self, "_campaign_reuse_manual_filenames", set()) or set())


def _get_preview_list_render_state(self, ann) -> dict | None:
    cache = getattr(self, "_preview_list_render_state_cache", None)
    if not isinstance(cache, dict) or ann is None:
        return None
    return cache.get(id(ann))


def _build_preview_list_render_state_cache(
    self,
    entries: list[tuple[int, ImageAnnotation]] | None = None,
) -> None:
    source_entries = list(entries if entries is not None else enumerate(list(self.current_annotations or [])))
    if not source_entries:
        self._preview_list_render_state_cache = {}
        return

    try:
        approved_names = set(self._get_preview_approved_filenames_base() or set())
    except Exception:
        approved_names = set()
    hidden_project_approved = set()
    if not self._is_free_mode_session_context():
        try:
            hidden_project_approved = set(self._get_campaign_hidden_project_approved_filenames_runtime() or set())
        except Exception:
            hidden_project_approved = set()
    try:
        manual_touched_names = set(self._get_campaign_manual_touched_filenames() or set())
    except Exception:
        manual_touched_names = set()
    try:
        reused_names = set(getattr(self, "_campaign_reuse_manual_filenames", set()) or set())
    except Exception:
        reused_names = set()

    cache: dict[int, dict] = {}
    for _actual_idx, ann in source_entries:
        filename = str(getattr(ann, "filename", "") or "").strip()
        filename_key = filename.lower()
        detections = list(getattr(ann, "detections", []) or [])
        has_plate = False
        manual = False
        auto = False

        for det in detections:
            if not _is_plate_detection_label(getattr(det, "label", "")):
                continue
            has_plate = True
            attributes = dict(getattr(det, "attributes", {}) or {})
            manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
            manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
            cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
            origin_source = str(
                attributes.get("annotation_origin")
                or attributes.get("cvat_source")
                or attributes.get("source")
                or ""
            ).strip().lower()
            auto_source = str(attributes.get("auto_source", "") or "").strip().lower()
            is_manual = bool(
                manually_edited
                or manual_source
                or cvat_source == "manual"
                or origin_source == "manual"
            )
            if is_manual:
                manual = True
                continue
            if cvat_source == "auto" or origin_source == "auto" or auto_source or not manually_edited:
                auto = True

        if filename_key and filename_key in manual_touched_names:
            manual = True
        approved = bool(
            has_plate
            and filename_key not in hidden_project_approved
            and (
                bool(getattr(ann, "_approved_for_training", False))
                or (filename_key and filename_key in approved_names)
            )
        )
        origin_tag = "M" if manual else "A" if auto else "--"
        if approved:
            status_text = f"{origin_tag}|OK" if origin_tag in {"M", "A"} else "OK"
            bucket = "approved"
        elif manual:
            status_text = "M"
            bucket = "manual"
        elif auto:
            status_text = "A"
            bucket = "auto"
        else:
            status_text = "--"
            bucket = "problem"

        cache[id(ann)] = {
            "approved": approved,
            "auto": auto,
            "bucket": bucket,
            "manual": manual,
            "origin_tag": origin_tag,
            "reused": filename in reused_names,
            "status_text": status_text,
        }

    self._preview_list_render_state_cache = cache


def _preview_annotation_has_manual_touch_direct(ann) -> bool:
    if ann is None:
        return False

    for det in getattr(ann, "detections", []) or []:
        if not _is_plate_detection_label(getattr(det, "label", "")):
            continue
        attributes = dict(getattr(det, "attributes", {}) or {})
        manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
        manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
        cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
        origin_source = str(
            attributes.get("annotation_origin")
            or attributes.get("cvat_source")
            or attributes.get("source")
            or ""
        ).strip().lower()
        if manually_edited or manual_source or cvat_source == "manual" or origin_source == "manual":
            return True
    return False


def _preview_annotation_has_manual_touch(self, ann) -> bool:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return bool(cached_state.get("manual"))

    if self._preview_annotation_has_manual_touch_direct(ann):
        return True

    filename = str(getattr(ann, "filename", "") or "").strip().lower()
    if not filename:
        return False

    try:
        return filename in set(self._get_campaign_manual_touched_filenames() or set())
    except Exception:
        return False


def _preview_annotation_is_manually_corrected(self, ann) -> bool:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return bool(cached_state.get("manual"))

    if ann is None:
        return False
    for det in getattr(ann, "detections", []) or []:
        if not self._is_plate_detection_label(getattr(det, "label", "")):
            continue
        attributes = dict(getattr(det, "attributes", {}) or {})
        manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
        manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
        cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
        origin_source = str(
            attributes.get("annotation_origin")
            or attributes.get("cvat_source")
            or attributes.get("source")
            or ""
        ).strip().lower()
        if manually_edited or manual_source or cvat_source == "manual" or origin_source == "manual":
            return True
    return False


def _preview_annotation_origin_tag(self, ann) -> str:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return str(cached_state.get("origin_tag", "") or "--")

    if self._preview_annotation_is_manually_corrected(ann):
        return "M"
    if self._preview_annotation_has_auto_plate(ann):
        return "A"
    return "--"


def _preview_annotation_status_tag(self, ann) -> str:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return str(cached_state.get("status_text", "") or "--")

    origin_tag = self._preview_annotation_origin_tag(ann)
    if self._preview_annotation_is_explicitly_approved(ann):
        if origin_tag in {"M", "A"}:
            return f"{origin_tag}|OK"
        return "OK"
    return origin_tag


def _preview_annotation_has_auto_plate(self, ann) -> bool:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return bool(cached_state.get("auto"))

    if ann is None:
        return False

    for det in self._get_plate_detections(ann):
        attributes = dict(getattr(det, "attributes", {}) or {})
        manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
        manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
        cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
        origin_source = str(
            attributes.get("annotation_origin")
            or attributes.get("cvat_source")
            or attributes.get("source")
            or ""
        ).strip().lower()
        auto_source = str(attributes.get("auto_source", "") or "").strip().lower()
        if cvat_source == "manual" or origin_source == "manual":
            continue
        if (
            cvat_source == "auto"
            or origin_source == "auto"
            or auto_source
            or (not manually_edited and not manual_source)
        ):
            return True
    return False


def _plate_detection_is_manual(det: Detection | None) -> bool:
    if det is None:
        return False
    attributes = dict(getattr(det, "attributes", {}) or {})
    manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
    manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
    cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
    origin_source = str(
        attributes.get("annotation_origin")
        or attributes.get("cvat_source")
        or attributes.get("source")
        or ""
    ).strip().lower()
    return bool(manually_edited or manual_source or cvat_source == "manual" or origin_source == "manual")


def _get_preview_any_auto_in_run(self) -> bool:
    cached = getattr(self, "_preview_any_auto_in_run_cache", None)
    if cached is not None:
        return bool(cached)

    any_auto = any(
        self._preview_annotation_has_auto_plate(ann)
        for ann in list(self.current_annotations or [])
    )
    self._preview_any_auto_in_run_cache = bool(any_auto)
    return bool(any_auto)


def _clear_selected_preview_auto_plates(self, *args, **kwargs):
    return z2_workflow_methods._clear_selected_preview_auto_plates(self, *args, **kwargs)


def _collect_preview_manually_touched_filenames(
    self,
    annotations: list | None = None,
    *,
    include_dirty: bool = True,
) -> set[str]:
    touched: set[str] = set()
    source_annotations = list(annotations if annotations is not None else (self.current_annotations or []))
    for ann in source_annotations:
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        if self._preview_annotation_has_manual_touch(ann):
            touched.add(filename)

    return touched


def _collect_preview_current_iteration_manual_filenames(
    self,
    annotations: list | None = None,
) -> set[str]:
    touched: set[str] = set()
    for ann in list(annotations if annotations is not None else (self.current_annotations or [])):
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        if self._preview_annotation_is_reused_from_previous_manual(ann):
            continue
        if self._preview_annotation_is_manually_corrected(ann):
            touched.add(filename)
    return touched


def _collect_campaign_current_iteration_manual_filenames(
    self,
    annotations: list | None = None,
) -> set[str]:
    manual_names = {
        str(name or "").strip().lower()
        for name in self._collect_preview_current_iteration_manual_filenames(annotations)
        if str(name or "").strip()
    }
    if self._is_free_mode_session_context():
        return manual_names

    tracked_names = {
        str(name or "").strip().lower()
        for name in set(getattr(self, "_campaign_iteration_manual_filenames", set()) or set())
        if str(name or "").strip()
    }
    return manual_names | tracked_names


def _build_manual_override_annotation(self, ann):
    normalized_ann = copy.deepcopy(ann)
    kept_detections = []
    fallback_plate_detections = []
    for det in list(getattr(normalized_ann, "detections", []) or []):
        if not self._is_plate_detection_label(getattr(det, "label", "")):
            kept_detections.append(det)
            continue

        attributes = dict(getattr(det, "attributes", {}) or {})
        manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
        manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
        if manually_edited or manual_source:
            kept_detections.append(det)
            continue
        fallback_plate_detections.append(copy.deepcopy(det))

    if not any(self._is_plate_detection_label(getattr(det, "label", "")) for det in kept_detections):
        if self._preview_annotation_has_manual_touch(ann) and fallback_plate_detections:
            for det in fallback_plate_detections:
                attributes = dict(getattr(det, "attributes", {}) or {})
                attributes.setdefault("manually_edited", "true")
                attributes.setdefault("manual_source", "preview")
                det.attributes = attributes
                kept_detections.append(det)

    normalized_ann.detections = kept_detections
    try:
        has_plate = any(self._is_plate_detection_label(getattr(det, "label", "")) for det in kept_detections)
        has_vehicle = any(
            str(getattr(det, "label", "") or "").strip().lower() in CONFIG.VEHICLE_LABELS
            for det in kept_detections
        )
        normalized_ann.status = (
            AnnotationStatus.SUCCESS
            if (has_plate or has_vehicle)
            else AnnotationStatus.NO_PLATE
        )
    except Exception:
        pass
    return normalized_ann


def _get_current_campaign_manual_preview_bundle(self) -> dict[str, tuple[ImageAnnotation, Path]]:
    bundle: dict[str, tuple[ImageAnnotation, Path]] = {}
    approved_lookup = set(self._get_preview_approved_filenames())
    project_approved_filenames: set[str] = set()
    project_approved_source_keys: set[str] = set()
    if not self._is_free_mode_session_context():
        try:
            project_approved_filenames = {
                str(name or "").strip().lower()
                for name in set(self._get_campaign_plate_approved_filenames() or set())
                if str(name or "").strip()
            }
        except Exception:
            project_approved_filenames = set()
        try:
            project_approved_source_keys = {
                str(key or "").strip()
                for key in set(self._get_campaign_plate_approved_source_keys() or set())
                if str(key or "").strip()
            }
        except Exception:
            project_approved_source_keys = set()

    def _register_annotation(ann, image_path: Path | None, approved_names: set[str] | None = None):
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename or image_path is None:
            return
        try:
            if not Path(image_path).exists():
                return
        except Exception:
            return
        safe_name = filename.lower()
        source_key = ""
        if project_approved_source_keys:
            try:
                source_key = self._build_campaign_source_image_key(Path(image_path))
            except Exception:
                source_key = ""
        if safe_name in project_approved_filenames:
            return
        if source_key and source_key in project_approved_source_keys:
            return
        is_manual = self._preview_annotation_has_manual_touch(ann)
        is_approved = bool(self._preview_annotation_is_explicitly_approved(ann)) or safe_name in set(approved_names or set())
        if not is_manual and not is_approved:
            return
        if is_manual:
            normalized_ann = self._build_manual_override_annotation(ann)
        else:
            normalized_ann = copy.deepcopy(ann)
        try:
            setattr(normalized_ann, "_approved_for_training", bool(is_approved))
        except Exception:
            pass
        bundle[filename] = (normalized_ann, Path(image_path))

    for ann in list(self.current_annotations or []):
        try:
            _register_annotation(ann, self._resolve_preview_image_path(ann), approved_lookup)
        except Exception:
            continue

    if bundle:
        return bundle

    xml_path = self._get_current_annotation_xml_path()
    run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None), require_xml=True)
    if xml_path is None or run_dir is None:
        return bundle

    try:
        annotations = self._parse_cvat_preview_annotations(xml_path)
    except Exception:
        return bundle

    manifest = self._load_annotation_run_manifest(run_dir)
    approved_from_run = self._load_annotation_run_approved_filenames(run_dir)
    image_dirs: list[Path] = []
    for raw_dir in (
        str(manifest.get("source_input_dir") or "").strip(),
        str(manifest.get("input_dir") or "").strip(),
        str(run_dir / "images"),
    ):
        if not raw_dir:
            continue
        try:
            candidate_dir = Path(raw_dir)
        except Exception:
            continue
        if candidate_dir not in image_dirs:
            image_dirs.append(candidate_dir)

    for ann in annotations:
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        resolved_path = None
        for image_dir in image_dirs:
            try:
                candidate = Path(image_dir) / filename
            except Exception:
                continue
            if candidate.exists():
                resolved_path = candidate
                break
        _register_annotation(ann, resolved_path, approved_from_run)

    return bundle


def _get_current_campaign_preview_snapshot(self) -> dict[str, tuple[ImageAnnotation, Path]]:
    snapshot: dict[str, tuple[ImageAnnotation, Path]] = {}
    approved_filenames: set[str] = set()
    approved_source_keys: set[str] = set()
    if not self._is_free_mode_session_context():
        try:
            approved_filenames = {
                str(name or "").strip().lower()
                for name in set(self._get_campaign_plate_approved_filenames() or set())
                if str(name or "").strip()
            }
        except Exception:
            approved_filenames = set()
        try:
            approved_source_keys = {
                str(key or "").strip()
                for key in set(self._get_campaign_plate_approved_source_keys() or set())
                if str(key or "").strip()
            }
        except Exception:
            approved_source_keys = set()
    for ann in list(self.current_annotations or []):
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        try:
            image_path = self._resolve_preview_image_path(ann)
        except Exception:
            image_path = None
        if image_path is None:
            continue
        try:
            if not Path(image_path).exists():
                continue
        except Exception:
            continue
        safe_name_key = filename.lower()
        source_key = ""
        if approved_source_keys:
            try:
                source_key = self._build_campaign_source_image_key(Path(image_path))
            except Exception:
                source_key = ""
        if safe_name_key in approved_filenames:
            continue
        if source_key and source_key in approved_source_keys:
            continue
        normalized_ann = copy.deepcopy(ann)
        try:
            setattr(
                normalized_ann,
                "_approved_for_training",
                bool(getattr(ann, "_approved_for_training", False) or self._preview_annotation_is_explicitly_approved(ann)),
            )
        except Exception:
            pass
        snapshot[filename] = (normalized_ann, Path(image_path))
    return snapshot


def _restore_pre_run_preview_snapshot(self) -> bool:
    visible_state = dict(getattr(self, "_campaign_auto_pre_run_visible_state", {}) or {})
    visible_annotations = visible_state.get("annotations")
    if isinstance(visible_annotations, list):
        try:
            self.current_annotations = copy.deepcopy(list(visible_annotations))
            self._preview_image_path_map = dict(visible_state.get("image_map") or {})

            raw_run_dir = str(visible_state.get("run_dir") or "").strip()
            raw_xml_path = str(visible_state.get("xml_path") or "").strip()
            raw_input_dir = str(visible_state.get("input_dir") or "").strip()
            raw_staging_dir = str(visible_state.get("last_staging_run_dir") or "").strip()
            self.current_annotation_run_dir = Path(raw_run_dir) if raw_run_dir else None
            self.current_annotation_xml_path = Path(raw_xml_path) if raw_xml_path else None
            self.current_input_dir = Path(raw_input_dir) if raw_input_dir else None
            self.last_staging_run_dir = Path(raw_staging_dir) if raw_staging_dir else self.current_annotation_run_dir

            approved = {
                str(name or "").strip().lower()
                for name in set(visible_state.get("approved_filenames") or set())
                if str(name or "").strip()
            }
            pending_approved = {
                str(name or "").strip().lower()
                for name in set(visible_state.get("campaign_pending_approved") or set())
                if str(name or "").strip()
            }
            self._preview_approved_filenames = set(approved)
            if not self._is_free_mode_session_context():
                self._campaign_pending_approved_filenames = set(pending_approved or approved)
                self._campaign_hidden_project_approved_filenames = set(
                    str(name or "").strip().lower()
                    for name in set(visible_state.get("hidden_project_approved") or set())
                    if str(name or "").strip()
                )
                self._campaign_hidden_char_effective_filenames = set(
                    str(name or "").strip().lower()
                    for name in set(visible_state.get("hidden_char_effective") or set())
                    if str(name or "").strip()
                )

            restore_index = visible_state.get("preview_index")
            try:
                restore_index = int(restore_index)
            except Exception:
                restore_index = 0 if self.current_annotations else None
            if restore_index is not None and self.current_annotations:
                self.current_preview_index = max(0, min(int(restore_index), len(self.current_annotations) - 1))
            else:
                self.current_preview_index = None

            try:
                self._invalidate_preview_runtime_caches()
            except Exception:
                pass
            try:
                self._populate_preview_list()
            except Exception:
                try:
                    self._refresh_preview_list(preserve_selection=False, render_current=True)
                except Exception:
                    pass
            try:
                self._sync_campaign_pending_batch_summary_from_preview()
            except Exception:
                pass
            try:
                logger.info(
                    "[Z2 ROLLBACK] Przywrocono widoczny stan sprzed autoanotacji: annotations=%s approved=%s run=%s",
                    len(self.current_annotations or []),
                    len(set(getattr(self, "_preview_approved_filenames", set()) or set())),
                    self.current_annotation_run_dir or "",
                )
            except Exception:
                pass
            return True
        except Exception as e:
            logger.debug(f"Nie udało się przywrócić widocznego stanu Z2 sprzed autoanotacji: {e}")

    snapshot = dict(getattr(self, "_campaign_auto_pre_run_snapshot", {}) or {})
    context = dict(getattr(self, "_campaign_auto_pre_run_context", {}) or {})
    previous_run_dir = None
    previous_xml_path = None
    previous_input_dir = None
    try:
        raw_run_dir = str(context.get("run_dir") or "").strip()
        if raw_run_dir:
            previous_run_dir = self._resolve_safe_annotation_run_dir(raw_run_dir, require_xml=True)
    except Exception:
        previous_run_dir = None
    try:
        raw_xml_path = str(context.get("xml_path") or "").strip()
        if raw_xml_path:
            previous_xml_path = Path(raw_xml_path)
    except Exception:
        previous_xml_path = None
    try:
        raw_input_dir = str(context.get("input_dir") or "").strip()
        if raw_input_dir:
            previous_input_dir = Path(raw_input_dir)
    except Exception:
        previous_input_dir = None

    if not snapshot:
        if previous_run_dir is None:
            return False
        try:
            restored_from_run = bool(self._restore_preview_from_annotation_run(previous_run_dir))
            if restored_from_run:
                approved_from_context = {
                    str(name or "").strip().lower()
                    for name in set(context.get("approved_filenames") or set())
                    if str(name or "").strip()
                }
                if approved_from_context:
                    self._preview_approved_filenames = set(approved_from_context)
                    if not self._is_free_mode_session_context():
                        self._campaign_pending_approved_filenames = set(approved_from_context)
                if previous_input_dir is not None:
                    self.current_input_dir = previous_input_dir
                return True
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ przywrĂłciÄ‡ poprzedniego runu Z2 po zatrzymaniu autoanotacji: {e}")
        return False

    try:
        if not self._is_free_mode_session_context():
            project_approved_filenames = {
                str(name or "").strip().lower()
                for name in set(self._get_campaign_plate_approved_filenames() or set())
                if str(name or "").strip()
            }
            project_approved_source_keys = {
                str(key or "").strip()
                for key in set(self._get_campaign_plate_approved_source_keys() or set())
                if str(key or "").strip()
            }
            if project_approved_filenames or project_approved_source_keys:
                filtered_snapshot = {}
                for filename, payload in snapshot.items():
                    safe_name = str(filename or "").strip()
                    if not safe_name or not isinstance(payload, tuple) or len(payload) < 2:
                        continue
                    safe_name_key = safe_name.lower()
                    source_key = ""
                    if project_approved_source_keys:
                        try:
                            source_key = self._build_campaign_source_image_key(Path(payload[1]))
                        except Exception:
                            source_key = ""
                    if safe_name_key in project_approved_filenames:
                        continue
                    if source_key and source_key in project_approved_source_keys:
                        continue
                    filtered_snapshot[safe_name] = payload
                snapshot = filtered_snapshot
        if not snapshot:
            if previous_run_dir is not None:
                return bool(self._restore_preview_from_annotation_run(previous_run_dir))
            return False

        self.current_annotations = []
        self._preview_image_path_map = {}
        approved_snapshot = {
            str(name or "").strip().lower()
            for name in set(getattr(self, "_pending_preview_approved_filenames", set()) or set())
            if str(name or "").strip()
        }
        restored_count = self._merge_preview_annotation_bundle(snapshot)
        if restored_count <= 0:
            if previous_run_dir is not None:
                return bool(self._restore_preview_from_annotation_run(previous_run_dir))
            return False
        self._preview_approved_filenames = set(approved_snapshot)
        context_approved = {
            str(name or "").strip().lower()
            for name in set(context.get("approved_filenames") or set())
            if str(name or "").strip()
        }
        if context_approved:
            self._preview_approved_filenames = set(context_approved)
        self._campaign_hidden_project_approved_filenames = set()
        self._campaign_iteration_manual_filenames = set()
        if not self._is_free_mode_session_context():
            self._campaign_pending_approved_filenames = set(approved_snapshot)
            context_campaign_approved = {
                str(name or "").strip().lower()
                for name in set(context.get("campaign_pending_approved") or set())
                if str(name or "").strip()
            }
            if context_campaign_approved:
                self._campaign_pending_approved_filenames = set(context_campaign_approved)
        if previous_run_dir is not None:
            self.current_annotation_run_dir = previous_run_dir
            self.current_annotation_xml_path = (
                previous_xml_path
                if previous_xml_path is not None
                else previous_run_dir / "annotations.xml"
            )
            try:
                self.last_staging_run_dir = previous_run_dir
            except Exception:
                pass
        if previous_input_dir is not None:
            self.current_input_dir = previous_input_dir
        try:
            preview_index = context.get("preview_index")
            if preview_index is not None:
                preview_index = int(preview_index)
                if 0 <= preview_index < len(self.current_annotations or []):
                    self.current_preview_index = preview_index
        except Exception:
            pass
        self._refresh_preview_list(preserve_selection=False, render_current=True)
        return True
    except Exception as e:
        logger.debug(f"Nie udało się przywrócić listy Z2 po zatrzymaniu autoanotacji: {e}")
        return False


def _merge_preview_annotation_bundle(
    self,
    bundle: dict[str, tuple[ImageAnnotation, Path]] | None,
) -> int:
    if not isinstance(bundle, dict) or not bundle:
        return 0

    annotations = list(self.current_annotations or [])
    annotation_indices = {
        str(getattr(ann, "filename", "") or "").strip(): idx
        for idx, ann in enumerate(annotations)
        if str(getattr(ann, "filename", "") or "").strip()
    }
    image_map = dict(getattr(self, "_preview_image_path_map", {}) or {})
    merged_count = 0

    for filename, payload in bundle.items():
        safe_name = str(filename or "").strip()
        if not safe_name or not isinstance(payload, tuple) or len(payload) < 2:
            continue
        ann, image_path = payload
        normalized_ann = copy.deepcopy(ann)
        normalized_ann.filename = safe_name
        # Empty placeholders can stay at 1x1 until the image is actually
        # opened on canvas. Reading dimensions for thousands of files made
        # T06 -> Z2 painfully slow on large project folders.
        if list(getattr(normalized_ann, "detections", []) or []):
            try:
                width, height = get_image_size(image_path)
                normalized_ann.width = max(1, int(width))
                normalized_ann.height = max(1, int(height))
            except Exception:
                pass
        else:
            normalized_ann.width = max(1, int(getattr(normalized_ann, "width", 1) or 1))
            normalized_ann.height = max(1, int(getattr(normalized_ann, "height", 1) or 1))

        if safe_name in annotation_indices:
            annotations[int(annotation_indices[safe_name])] = normalized_ann
        else:
            annotation_indices[safe_name] = len(annotations)
            annotations.append(normalized_ann)
        try:
            image_map[safe_name] = Path(image_path)
        except Exception:
            pass
        merged_count += 1

    self.current_annotations = annotations
    self._preview_image_path_map = image_map
    return merged_count


def _build_missing_preview_annotations_bundle(
    self,
    image_dir: Path | None,
    *,
    existing_annotations: list[ImageAnnotation] | None = None,
    extra_hidden_filenames: set[str] | None = None,
    scope_filenames: set[str] | None = None,
) -> dict[str, tuple[ImageAnnotation, Path]]:
    try:
        source_dir = Path(image_dir) if image_dir is not None else None
    except Exception:
        source_dir = None
    if source_dir is None:
        return {}
    try:
        if not source_dir.exists() or not source_dir.is_dir():
            return {}
    except Exception:
        return {}

    existing_names = {
        str(getattr(ann, "filename", "") or "").strip().lower()
        for ann in list(existing_annotations or self.current_annotations or [])
        if str(getattr(ann, "filename", "") or "").strip()
    }
    filtered_extra_hidden = {
        str(name or "").strip().lower()
        for name in set(extra_hidden_filenames or set())
        if str(name or "").strip()
    }
    project_approved_filenames: set[str] = set()
    project_approved_source_keys: set[str] = set()
    if not self._is_free_mode_session_context():
        try:
            project_approved_filenames = set(self._get_campaign_plate_approved_filenames() or set())
        except Exception:
            project_approved_filenames = set()
        try:
            project_approved_source_keys = set(self._get_campaign_plate_approved_source_keys() or set())
        except Exception:
            project_approved_source_keys = set()

    bundle: dict[str, tuple[ImageAnnotation, Path]] = {}
    scope_names = {
        str(name or "").strip().lower()
        for name in set(scope_filenames or set())
        if str(name or "").strip()
    }
    if scope_names:
        image_paths = []
        missing_scope_names: set[str] = set()
        for safe_name in sorted(scope_names):
            candidate = source_dir / safe_name
            try:
                if candidate.exists() and candidate.is_file():
                    image_paths.append(candidate)
                    continue
            except Exception:
                pass
            missing_scope_names.add(safe_name)
        if missing_scope_names:
            try:
                shallow_map = {
                    str(path.name or "").strip().lower(): Path(path)
                    for path in source_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
                }
            except Exception:
                shallow_map = {}
            for safe_name in sorted(missing_scope_names):
                candidate = shallow_map.get(safe_name)
                if candidate is not None:
                    image_paths.append(candidate)
    else:
        try:
            image_paths = get_image_files(source_dir)
        except Exception:
            image_paths = []

    for image_path in image_paths:
        safe_name = str(getattr(image_path, "name", "") or "").strip()
        if not safe_name or safe_name.lower() in existing_names:
            continue
        safe_name_key = safe_name.lower()
        if safe_name_key in filtered_extra_hidden:
            continue
        source_key = ""
        if project_approved_source_keys:
            try:
                source_key = self._build_campaign_source_image_key(Path(image_path))
            except Exception:
                source_key = ""
        if (
            not self._is_free_mode_session_context()
            and (
                safe_name_key in project_approved_filenames
                or (source_key and source_key in project_approved_source_keys)
            )
        ):
            continue
        ann = ImageAnnotation(
            filename=safe_name,
            width=1,
            height=1,
            detections=[],
            status=AnnotationStatus.NO_PLATE,
            status_message="Ten obraz nie był częścią zapisanego runu autoanotacji i wraca jako pozycja robocza [--].",
        )
        bundle[safe_name] = (ann, Path(image_path))
    return bundle


def _merge_annotation_bundle_into_payload(
    annotations: list[ImageAnnotation] | None,
    image_map: dict[str, Path] | None,
    bundle: dict[str, tuple[ImageAnnotation, Path]] | None,
) -> tuple[list[ImageAnnotation], dict[str, Path], int]:
    merged_annotations = list(annotations or [])
    merged_image_map = dict(image_map or {})
    if not isinstance(bundle, dict) or not bundle:
        return merged_annotations, merged_image_map, 0

    replacement_names = {
        str(filename or "").strip().lower()
        for filename in bundle.keys()
        if str(filename or "").strip()
    }
    if replacement_names:
        merged_annotations = [
            ann
            for ann in merged_annotations
            if str(getattr(ann, "filename", "") or "").strip().lower() not in replacement_names
        ]

    annotation_indices = {
        str(getattr(ann, "filename", "") or "").strip().lower(): idx
        for idx, ann in enumerate(merged_annotations)
        if str(getattr(ann, "filename", "") or "").strip()
    }
    merged_count = 0

    for filename, payload in bundle.items():
        safe_name = str(filename or "").strip()
        if not safe_name or not isinstance(payload, tuple) or len(payload) < 2:
            continue
        ann, image_path = payload
        normalized_ann = copy.deepcopy(ann)
        normalized_ann.filename = safe_name
        safe_key = safe_name.lower()
        try:
            setattr(
                normalized_ann,
                "_approved_for_training",
                bool(getattr(ann, "_approved_for_training", False)),
            )
        except Exception:
            pass
        try:
            current_width = int(getattr(normalized_ann, "width", 0) or 0)
            current_height = int(getattr(normalized_ann, "height", 0) or 0)
        except Exception:
            current_width, current_height = 0, 0
        if (
            (current_width <= 1 or current_height <= 1)
            and list(getattr(normalized_ann, "detections", []) or [])
        ):
            try:
                width, height = get_image_size(image_path)
                normalized_ann.width = max(1, int(width))
                normalized_ann.height = max(1, int(height))
            except Exception:
                pass
        elif current_width <= 1 or current_height <= 1:
            normalized_ann.width = 1
            normalized_ann.height = 1
        else:
            normalized_ann.width = max(1, current_width)
            normalized_ann.height = max(1, current_height)

        if safe_key in annotation_indices:
            merged_annotations[int(annotation_indices[safe_key])] = normalized_ann
        else:
            annotation_indices[safe_key] = len(merged_annotations)
            merged_annotations.append(normalized_ann)
        try:
            merged_image_map[safe_name] = Path(image_path)
        except Exception:
            pass
        merged_count += 1

    return merged_annotations, merged_image_map, merged_count


def _preview_list_item_color(self, ann) -> str:
    return _preview_list_color_for_bucket(self, _preview_list_effective_color_bucket(self, ann))


def _preview_list_color_for_bucket(self, bucket: str) -> str:
    palette = getattr(self.app, "palette", {})
    normalized = str(bucket or "").strip().lower()
    if normalized == "reused":
        return palette.get("warning", "#f39c12")
    if normalized == "approved":
        return palette.get("accent", "#0e639c")
    if normalized == "manual":
        return palette.get("warning", "#f39c12")
    if normalized == "auto":
        return palette.get("success", "#27ae60")
    return palette.get("error", "#c0392b")


def _preview_list_effective_color_bucket(self, ann) -> str:
    cached_state = self._get_preview_list_render_state(ann)
    if isinstance(cached_state, dict):
        if bool(cached_state.get("reused")):
            return "reused"
        bucket = str(cached_state.get("bucket", "") or "").strip().lower()
        return bucket if bucket in {"approved", "manual", "auto", "problem"} else "problem"

    if self._preview_annotation_is_reused_from_previous_manual(ann):
        return "reused"
    bucket = self._preview_annotation_sort_bucket(ann)
    return bucket if bucket in {"approved", "manual", "auto", "problem"} else "problem"


def _preview_list_color_plan(self, entries: list[tuple[int, ImageAnnotation]]) -> tuple[str, str]:
    counts = {"approved": 0, "manual": 0, "auto": 0, "problem": 0, "reused": 0}
    for _actual_idx, ann in entries or []:
        bucket = _preview_list_effective_color_bucket(self, ann)
        counts[bucket if bucket in counts else "problem"] += 1
    dominant_bucket = max(
        counts,
        key=lambda key: (counts.get(key, 0), {"problem": 4, "auto": 3, "approved": 2, "manual": 1, "reused": 0}.get(key, 0)),
    )
    return dominant_bucket, _preview_list_color_for_bucket(self, dominant_bucket)


def _preview_annotation_sort_bucket(self, ann) -> str:
    cached_state = self._get_preview_list_render_state(ann)
    if cached_state is not None:
        return str(cached_state.get("bucket", "") or "problem")

    if self._preview_annotation_is_explicitly_approved(ann):
        return "approved"
    if self._preview_annotation_is_manually_corrected(ann):
        return "manual"
    if self._preview_annotation_has_auto_plate(ann):
        return "auto"
    return "problem"


def _preview_annotation_sort_bucket_for_mode(self, ann, sort_mode: str | None = None) -> str:
    normalized_sort = self._normalize_preview_list_sort_mode(sort_mode)
    if normalized_sort == "Status: A, M, OK, problem":
        cached_state = self._get_preview_list_render_state(ann)
        if cached_state is not None:
            origin_tag = str(cached_state.get("origin_tag", "") or "").strip().upper()
            if origin_tag == "A":
                return "auto"
            if origin_tag == "M":
                return "manual"
            if bool(cached_state.get("approved")):
                return "approved"
            return "problem"
        origin_tag = self._preview_annotation_origin_tag(ann)
        if origin_tag == "A":
            return "auto"
        if origin_tag == "M":
            return "manual"
        if self._preview_annotation_is_explicitly_approved(ann):
            return "approved"
        return "problem"
    return self._preview_annotation_sort_bucket(ann)


def _preview_annotation_auto_scope_bucket(self, ann) -> str:
    try:
        origin_tag = str(self._preview_annotation_origin_tag(ann) or "").strip().upper()
    except Exception:
        origin_tag = ""
    if origin_tag == "M":
        return "manual"
    if origin_tag == "A":
        return "auto"
    return "problem"


def _preview_list_status_priority(self, sort_mode: str | None = None) -> dict[str, int]:
    sort_mode = self._normalize_preview_list_sort_mode(sort_mode)
    if sort_mode == "Status: A, M, OK, problem":
        return {"auto": 0, "manual": 1, "approved": 2, "problem": 3}
    if sort_mode == "Status: OK, ED, problem":
        return {"approved": 0, "manual": 1, "auto": 2, "problem": 3}
    if sort_mode == "Status: problem, ED, OK":
        return {"problem": 0, "auto": 1, "manual": 2, "approved": 3}
    return {"manual": 0, "approved": 1, "auto": 2, "problem": 3}


def _build_preview_list_sorted_entries(
    self,
    sort_mode: str,
    entries: list[tuple[int, ImageAnnotation]] | None = None,
) -> list[tuple[int, ImageAnnotation]]:
    entries = list(entries if entries is not None else enumerate(list(self.current_annotations or [])))
    if sort_mode == "Nazwa pliku A-Z":
        entries.sort(
            key=lambda item: str(getattr(item[1], "filename", "") or "").lower()
        )
        return entries

    priority = self._preview_list_status_priority(sort_mode)
    entries.sort(
        key=lambda item: (
            priority.get(self._preview_annotation_sort_bucket_for_mode(item[1], sort_mode), 99),
            str(getattr(item[1], "filename", "") or "").lower(),
        )
    )
    return entries


def _apply_preview_list_frozen_order(
    self,
    entries: list[tuple[int, ImageAnnotation]],
) -> tuple[list[tuple[int, ImageAnnotation]], int]:
    frozen_order = list(getattr(self, "_preview_list_frozen_filename_order", []) or [])
    if not frozen_order:
        return list(entries), 0

    order_slots: dict[str, list[int]] = {}
    for order_index, filename in enumerate(frozen_order):
        order_slots.setdefault(str(filename or "").strip().lower(), []).append(order_index)

    sortable_rows: list[tuple[int, int, int, int, ImageAnnotation]] = []
    matched_count = 0
    fallback_base = len(frozen_order)
    for fallback_index, (actual_idx, ann) in enumerate(entries):
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        slot_list = order_slots.get(filename_key)
        if slot_list:
            matched_count += 1
            sort_group = 0
            sort_index = int(slot_list.pop(0))
        else:
            sort_group = 1
            sort_index = fallback_base + fallback_index
        sortable_rows.append((sort_group, sort_index, fallback_index, actual_idx, ann))

    sortable_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ordered_entries = [(actual_idx, ann) for _group, _index, _fallback, actual_idx, ann in sortable_rows]
    return ordered_entries, matched_count


def _get_preview_list_entries(self) -> list[tuple[int, ImageAnnotation]]:
    sort_mode = self._normalize_preview_list_sort_mode()
    base_entries = self._filter_preview_list_entries(
        self._build_preview_list_sorted_entries(sort_mode)
    )
    if not base_entries:
        self._invalidate_preview_list_frozen_order()
        return []

    current_context_key = self._get_preview_list_context_key()
    frozen_sort_mode = self._normalize_preview_list_sort_mode(
        getattr(self, "_preview_list_frozen_sort_mode", "")
    )
    frozen_context_key = str(getattr(self, "_preview_list_frozen_context_key", "") or "").strip()
    frozen_order = list(getattr(self, "_preview_list_frozen_filename_order", []) or [])

    if (
        sort_mode != frozen_sort_mode
        or current_context_key != frozen_context_key
        or not frozen_order
    ):
        self._store_preview_list_frozen_order(sort_mode, base_entries)
        return base_entries

    ordered_entries, matched_count = self._apply_preview_list_frozen_order(base_entries)
    if matched_count <= 0:
        self._store_preview_list_frozen_order(sort_mode, base_entries)
        return base_entries
    return ordered_entries


def _get_preview_display_index(self, actual_index: int | None) -> int | None:
    if actual_index is None:
        return None
    index_map = dict(getattr(self, "_preview_list_display_index_map", {}) or {})
    try:
        normalized_index = int(actual_index)
    except Exception:
        normalized_index = None
    if normalized_index is not None and normalized_index in index_map:
        return int(index_map[normalized_index])
    indices = list(getattr(self, "_preview_list_display_indices", []) or [])
    try:
        return indices.index(int(actual_index))
    except Exception:
        return None


def _get_preview_actual_index_from_display(self, display_index: int | None) -> int | None:
    indices = getattr(self, "_preview_list_display_indices", []) or []
    if display_index is None:
        return None
    try:
        safe_index = int(display_index)
    except Exception:
        return None
    if safe_index < 0 or safe_index >= len(indices):
        return None
    return int(indices[safe_index])


def _get_selected_preview_actual_indices(self) -> list[int]:
    selected_actual_indices: list[int] = []
    seen_actual_indices: set[int] = set()
    for display_index in list(self.preview_listbox.curselection() or ()):
        actual_index = self._get_preview_actual_index_from_display(display_index)
        if actual_index is None:
            continue
        safe_actual_index = int(actual_index)
        if safe_actual_index in seen_actual_indices:
            continue
        seen_actual_indices.add(safe_actual_index)
        selected_actual_indices.append(safe_actual_index)
    return selected_actual_indices


def _select_all_visible_preview_images(self) -> None:
    try:
        total_visible = int(self.preview_listbox.size() or 0)
    except Exception:
        total_visible = 0

    if total_visible <= 0:
        self._update_preview_edit_status("Brak obrazów spełniających bieżący filtr.")
        return

    try:
        self._clear_listbox_selection_fast(self.preview_listbox)
        self.preview_listbox.selection_set(0, tk.END)
        self.preview_listbox.selection_anchor(0)
        self.preview_listbox.activate(0)
        self.preview_listbox.see(0)
    except Exception:
        return

    self._update_preview_toolbar_state()
    self._update_preview_edit_status(f"Zaznaczono obrazy spełniające filtr: {total_visible}.")
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        self._refresh_plate_auto_scope_modal_selection_state()


def _set_selected_preview_images_approved(self, *args, **kwargs):
    return z2_workflow_methods._set_selected_preview_images_approved(self, *args, **kwargs)


def _refresh_preview_list_summary(self, *args, **kwargs):
    return z2_workflow_methods._refresh_preview_list_summary(self, *args, **kwargs)


def _refresh_preview_list_legend_theme(self, *args, **kwargs):
    return z2_workflow_methods._refresh_preview_list_legend_theme(self, *args, **kwargs)


def _get_preview_annotation(self, idx: int | None = None):
    if not self.current_annotations:
        return None
    index = self.current_preview_index if idx is None else idx
    if index is None or index < 0 or index >= len(self.current_annotations):
        return None
    return self.current_annotations[index]


def _replace_preview_filename_in_set(values, old_name: str, new_name: str) -> set[str]:
    safe_old = str(old_name or "").strip()
    safe_new = str(new_name or "").strip()
    if not safe_old:
        return {
            str(item or "").strip()
            for item in set(values or set())
            if str(item or "").strip()
        }
    normalized = set()
    for item in set(values or set()):
        text = str(item or "").strip()
        if not text:
            continue
        if text == safe_old:
            text = safe_new
        normalized.add(text)
    return normalized


def _replace_preview_filename_in_dict_keys(payload: dict | None, old_name: str, new_name: str) -> dict:
    result: dict = {}
    safe_old = str(old_name or "").strip()
    safe_new = str(new_name or "").strip()
    for key, value in dict(payload or {}).items():
        text = str(key or "").strip()
        if not text:
            continue
        if text == safe_old:
            text = safe_new
        result[text] = value
    return result


def _rename_selected_preview_image_file(self, *args, **kwargs):
    return z2_workflow_methods._rename_selected_preview_image_file(self, *args, **kwargs)


def _serialize_quality_metric_value(value: float) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.000"


def _compute_plate_detection_fit_metrics(self, det: Detection, ann=None) -> dict:
    polygon = self._detection_polygon(det)
    bbox = getattr(det, "bbox", None)
    keypoints = getattr(det, "keypoints", None)
    image_size = None
    if ann is not None:
        try:
            image_size = (int(getattr(ann, "width", 0) or 0), int(getattr(ann, "height", 0) or 0))
        except Exception:
            image_size = None
    return compute_plate_polygon_fit_metrics(
        float(getattr(det, "confidence", 0.0) or 0.0),
        polygon,
        bbox,
        keypoints=keypoints,
        image_size=image_size,
    )


def _refresh_plate_detection_quality_metrics(self, det: Detection, ann=None, *, force: bool = False) -> dict:
    attributes = dict(getattr(det, "attributes", {}) or {})
    if self._plate_detection_is_manual(det):
        for key in (
            "fit_score",
            "fit_label",
            "fit_keypoint_score",
            "fit_shape_score",
            "fit_bbox_score",
            "fit_size_score",
        ):
            attributes.pop(key, None)
        det.attributes = attributes
        return attributes

    if not force and str(attributes.get("fit_score", "") or "").strip():
        return attributes

    metrics = self._compute_plate_detection_fit_metrics(det, ann)
    attributes.update({
        "fit_score": self._serialize_quality_metric_value(metrics.get("fit_score", 0.0)),
        "fit_label": str(metrics.get("fit_label") or "").strip(),
        "fit_keypoint_score": self._serialize_quality_metric_value(metrics.get("keypoint_score", 0.0)),
        "fit_shape_score": self._serialize_quality_metric_value(metrics.get("shape_score", 0.0)),
        "fit_bbox_score": self._serialize_quality_metric_value(metrics.get("bbox_alignment_score", 0.0)),
        "fit_size_score": self._serialize_quality_metric_value(metrics.get("size_score", 0.0)),
    })
    det.attributes = attributes
    return attributes


def _get_plate_detection_fit_score(self, det: Detection, ann=None) -> float | None:
    attributes = self._refresh_plate_detection_quality_metrics(det, ann)
    raw_value = str(attributes.get("fit_score", "") or "").strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except Exception:
        return None


def _get_preview_annotation_quality_summary(self, ann) -> dict:
    plates = self._get_plate_detections(ann)
    if not plates:
        return {
            "plate_count": 0,
            "min_confidence": 0.0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "fit_count": 0,
            "min_fit_score": 0.0,
            "avg_fit_score": 0.0,
            "max_fit_score": 0.0,
        }

    def _cache_key() -> tuple:
        key_parts = []
        for det in plates:
            attributes = dict(getattr(det, "attributes", {}) or {})
            key_parts.append(
                (
                    id(det),
                    tuple(round(float(value or 0.0), 3) for value in (getattr(det, "bbox", None) or ())),
                    tuple(
                        (round(float(x or 0.0), 3), round(float(y or 0.0), 3))
                        for x, y in (getattr(det, "polygon", None) or ())
                    ),
                    str(attributes.get("fit_score", "") or ""),
                    str(attributes.get("fit_score_source", "") or ""),
                    round(float(getattr(det, "confidence", 0.0) or 0.0), 4),
                )
            )
        return (id(ann), len(plates), tuple(key_parts))

    initial_key = _cache_key()
    cache = getattr(self, "_preview_annotation_quality_summary_cache", None)
    if isinstance(cache, dict):
        cached = cache.get(initial_key)
        if isinstance(cached, dict):
            return dict(cached)

    for det in plates:
        try:
            self._refresh_plate_detection_quality_metrics(det, ann)
        except Exception:
            continue
    confidences = [max(0.0, min(1.0, float(getattr(det, "confidence", 0.0) or 0.0))) for det in plates]
    fit_scores = [
        score
        for score in (self._get_plate_detection_fit_score(det, ann) for det in plates)
        if score is not None
    ]
    summary = {
        "plate_count": len(plates),
        "min_confidence": min(confidences) if confidences else 0.0,
        "avg_confidence": (sum(confidences) / float(len(confidences))) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "fit_count": len(fit_scores),
        "min_fit_score": min(fit_scores) if fit_scores else 0.0,
        "avg_fit_score": (sum(fit_scores) / float(len(fit_scores))) if fit_scores else 0.0,
        "max_fit_score": max(fit_scores) if fit_scores else 0.0,
    }
    try:
        cache = getattr(self, "_preview_annotation_quality_summary_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._preview_annotation_quality_summary_cache = cache
        final_key = _cache_key()
        cache[initial_key] = dict(summary)
        cache[final_key] = dict(summary)
        if len(cache) > 12000:
            self._preview_annotation_quality_summary_cache = dict(list(cache.items())[-6000:])
    except Exception:
        pass
    return summary


def _get_plate_detections(self, ann) -> list[Detection]:
    if ann is None:
        return []
    return [
        det
        for det in getattr(ann, "detections", []) or []
        if self._is_plate_detection_label(getattr(det, "label", ""))
    ]


def _detection_polygon(det: Detection) -> list[tuple[float, float]]:
    if det.polygon and len(det.polygon) >= 4:
        return [(float(x), float(y)) for x, y in det.polygon[:4]]

    x1, y1, x2, y2 = det.bbox
    return [
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    ]


def _bbox_from_polygon(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [float(x) for x, _y in points]
    ys = [float(y) for _x, y in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    prepared = [(float(x), float(y)) for x, y in points]
    for idx, (x1, y1) in enumerate(prepared):
        x2, y2 = prepared[(idx + 1) % len(prepared)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _keypoints_from_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    return [(float(x), float(y), 1.0) for x, y in points[:4]]


def _get_selected_plate_index_for_ann(self, ann) -> int | None:
    plates = self._get_plate_detections(ann)
    if not plates:
        self._preview_selected_plate_by_image.pop(getattr(ann, "filename", ""), None)
        return None

    filename = str(getattr(ann, "filename", "") or "")
    idx = self._preview_selected_plate_by_image.get(filename, 0)
    if idx is None:
        idx = 0
    idx = max(0, min(int(idx), len(plates) - 1))
    self._preview_selected_plate_by_image[filename] = idx
    return idx


def _set_selected_plate_index_for_ann(self, ann, plate_idx: int | None):
    plates = self._get_plate_detections(ann)
    filename = str(getattr(ann, "filename", "") or "")
    if not plates or plate_idx is None:
        self._preview_selected_plate_by_image.pop(filename, None)
        return None

    safe_idx = max(0, min(int(plate_idx), len(plates) - 1))
    self._preview_selected_plate_by_image[filename] = safe_idx
    return safe_idx


def _get_vehicle_detections(ann) -> list[Detection]:
    if ann is None:
        return []
    return [det for det in ann.detections if str(det.label or "").lower() == "vehicle"]


def _get_selected_vehicle_index_for_ann(self, ann) -> int | None:
    vehicles = self._get_vehicle_detections(ann)
    if not vehicles:
        self._preview_selected_vehicle_by_image.pop(getattr(ann, "filename", ""), None)
        return None

    filename = str(getattr(ann, "filename", "") or "")
    idx = self._preview_selected_vehicle_by_image.get(filename, 0)
    if idx is None:
        idx = 0
    idx = max(0, min(int(idx), len(vehicles) - 1))
    self._preview_selected_vehicle_by_image[filename] = idx
    return idx


def _set_selected_vehicle_index_for_ann(self, ann, vehicle_idx: int | None):
    vehicles = self._get_vehicle_detections(ann)
    filename = str(getattr(ann, "filename", "") or "")
    if not vehicles or vehicle_idx is None:
        self._preview_selected_vehicle_by_image.pop(filename, None)
        return None

    safe_idx = max(0, min(int(vehicle_idx), len(vehicles) - 1))
    self._preview_selected_vehicle_by_image[filename] = safe_idx
    return safe_idx


def _set_preview_focus_target(self, kind: str | None, index: int | None = None):
    filename = self._get_preview_focus_image_key()
    if not kind or index is None or not filename:
        self._preview_focus_target = None
        return
    self._preview_focus_target = {
        "filename": filename,
        "kind": str(kind),
        "index": int(index),
    }


def _get_preview_focus_target(self) -> dict | None:
    target = getattr(self, "_preview_focus_target", None)
    if not isinstance(target, dict):
        return None
    if str(target.get("filename", "") or "") != self._get_preview_focus_image_key():
        return None
    return target


def _mark_preview_image_dirty(
    self,
    ann,
    refresh_list: bool = True,
    refresh_row: bool = True,
    invalidate_runtime: bool = True,
):
    if ann is None:
        return
    if invalidate_runtime:
        self._invalidate_preview_runtime_caches()
    was_dirty = ann.filename in self._preview_dirty_images
    self._preview_dirty_images.add(ann.filename)
    if refresh_list and not was_dirty:
        self._refresh_preview_list(preserve_selection=True, render_current=False)
    elif not was_dirty and refresh_row:
        self._refresh_preview_list_row_for_actual_index(self.current_preview_index, refresh_summary=False)
        self._update_preview_toolbar_state(refresh_summary=False)
    elif not was_dirty:
        self._update_preview_toolbar_state(refresh_summary=False)


def _get_preview_history_image_key(self, ann=None) -> str:
    target_ann = self._get_preview_annotation() if ann is None else ann
    return str(getattr(target_ann, "filename", "") or "").strip()


def _clone_preview_history_detection(det: Detection) -> Detection:
    keypoints = None
    if isinstance(getattr(det, "keypoints", None), list):
        keypoints = [
            (float(point[0]), float(point[1]), float(point[2]))
            for point in det.keypoints
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ]

    polygon = None
    if isinstance(getattr(det, "polygon", None), list):
        polygon = [
            (float(point[0]), float(point[1]))
            for point in det.polygon
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]

    bbox = tuple(getattr(det, "bbox", (0.0, 0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0, 0.0))
    if len(bbox) >= 4:
        bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    return Detection(
        label=str(getattr(det, "label", "") or ""),
        confidence=float(getattr(det, "confidence", 0.0) or 0.0),
        bbox=bbox,
        keypoints=keypoints,
        polygon=polygon,
        text=getattr(det, "text", None),
        text_confidence=float(getattr(det, "text_confidence", 0.0) or 0.0),
        attributes=dict(getattr(det, "attributes", {}) or {}),
    )


def _clone_preview_history_annotation(self, ann=None, *, lightweight_plate_edit: bool = False):
    target_ann = self._get_preview_annotation() if ann is None else ann
    image_key = self._get_preview_history_image_key(target_ann)
    if target_ann is None or not image_key:
        return None

    detections = []
    for det in list(getattr(target_ann, "detections", []) or []):
        if isinstance(det, Detection):
            if lightweight_plate_edit and not self._is_plate_detection_label(getattr(det, "label", "")):
                detections.append(det)
            else:
                detections.append(self._clone_preview_history_detection(det))

    return ImageAnnotation(
        filename=image_key,
        width=int(getattr(target_ann, "width", 0) or 0),
        height=int(getattr(target_ann, "height", 0) or 0),
        detections=detections,
        status=getattr(target_ann, "status", AnnotationStatus.SUCCESS),
        status_message=str(getattr(target_ann, "status_message", "") or ""),
    )


def _clone_preview_annotation_history_snapshot(self, ann=None, *, lightweight_plate_edit: bool = False):
    target_ann = self._get_preview_annotation() if ann is None else ann
    cloned_ann = self._clone_preview_history_annotation(
        target_ann,
        lightweight_plate_edit=bool(lightweight_plate_edit),
    )
    if cloned_ann is None:
        return None
    return {
        "annotation": cloned_ann,
        "selected_plate_idx": self._get_selected_plate_index_for_ann(target_ann),
        "selected_vehicle_idx": self._get_selected_vehicle_index_for_ann(target_ann),
    }


def _get_preview_history_stack(self, kind: str, image_key: str | None = None, create: bool = False):
    key = str(image_key or self._get_preview_history_image_key() or "").strip()
    if not key:
        return None
    store_attr = "_preview_history_undo" if str(kind).lower() == "undo" else "_preview_history_redo"
    store = getattr(self, store_attr, None)
    if not isinstance(store, dict):
        store = {}
        setattr(self, store_attr, store)
    if create:
        return store.setdefault(key, [])
    return store.get(key)


def _push_preview_history_snapshot(self, ann=None, *, lightweight_plate_edit: bool = False):
    if bool(getattr(self, "_preview_history_replaying", False)):
        return
    image_key = self._get_preview_history_image_key(ann)
    if not image_key:
        return

    snapshot = self._clone_preview_annotation_history_snapshot(
        ann,
        lightweight_plate_edit=bool(lightweight_plate_edit),
    )
    if snapshot is None:
        return

    undo_stack = self._get_preview_history_stack("undo", image_key, create=True)
    if isinstance(undo_stack, list) and undo_stack and undo_stack[-1] == snapshot:
        return

    undo_stack.append(snapshot)
    limit = max(10, int(getattr(self, "_preview_history_limit", 80) or 80))
    if len(undo_stack) > limit:
        del undo_stack[:-limit]

    redo_stack = self._get_preview_history_stack("redo", image_key, create=True)
    if isinstance(redo_stack, list):
        redo_stack.clear()


def _restore_preview_annotation_history_snapshot(self, snapshot, *, action_label: str):
    ann = self._get_preview_annotation()
    image_key = self._get_preview_history_image_key(ann)
    if ann is None or not image_key or self.current_preview_index is None:
        return False

    restored_ann = None
    if isinstance(snapshot, dict):
        candidate = snapshot.get("annotation")
        if isinstance(candidate, ImageAnnotation):
            restored_ann = self._clone_preview_history_annotation(candidate)
    if restored_ann is None:
        return False

    self._preview_history_replaying = True
    try:
        safe_index = int(self.current_preview_index)
        if safe_index < 0 or safe_index >= len(self.current_annotations):
            return False

        self.current_annotations[safe_index] = restored_ann
        selected_plate_idx = snapshot.get("selected_plate_idx") if isinstance(snapshot, dict) else None
        selected_vehicle_idx = snapshot.get("selected_vehicle_idx") if isinstance(snapshot, dict) else None
        self._set_selected_plate_index_for_ann(restored_ann, selected_plate_idx)
        self._set_selected_vehicle_index_for_ann(restored_ann, selected_vehicle_idx)
        self._preview_drag_state = None
        self._preview_pending_vertex_hit = None
        self._preview_focus_zoom_modifier_down = False
        self._preview_focus_zoom_modifier_consumed = False
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_restore_state = None
        self._preview_focus_zoom_history = []
        self._preview_draw_mode = False
        self._preview_draw_points = []
        self._preview_delete_mode = False
        self._preview_delete_candidate_idx = None
        self._mark_preview_image_dirty(restored_ann, refresh_list=True)
        self._refresh_preview_canvas()
        self._update_preview_toolbar_state()
        if self._save_preview_edits(interactive=False, status_message=action_label):
            self._update_preview_edit_status(action_label)
        else:
            self._update_preview_edit_status(
                f"{action_label} Nie udalo sie od razu zapisac annotations.xml. Uzyj Ctrl+S."
            )
        self._remember_annotation_run_resume_state()
        self._queue_free_mode_session_save()
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass
        return True
    finally:
        self._preview_history_replaying = False


def _clear_preview_editor_state(self, clear_dirty: bool = True):
    self._cancel_deferred_campaign_restore_ui()
    self._cancel_deferred_campaign_run_restore()
    self._cancel_preview_list_population()
    self._cancel_preview_resume_persist()
    self._cancel_preview_selection_render()
    self._cancel_preview_post_interaction_refresh()
    has_pending_preview_save = bool(getattr(self, "_preview_dirty_images", None))
    has_unfinished_draw = bool(self._preview_draw_mode and self._preview_draw_points)
    if has_pending_preview_save and not has_unfinished_draw:
        try:
            self._save_preview_edits(
                interactive=False,
                status_message="Zapisano poprawki polygonow przed zamknieciem podgladu.",
            )
        except Exception:
            self._cancel_preview_autosave()
    else:
        self._cancel_preview_autosave()
    self._cancel_preview_drag_refresh()
    self._cancel_preview_layout_restore_jobs()
    if getattr(self, "_preview_fullscreen_active", False):
        self._set_preview_fullscreen(False)
    if not bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        try:
            self.preview_canvas.grab_release()
        except Exception:
            pass
        self._preview_corner_drag_grab_ready = False
    try:
        self.preview_canvas.clear_image()
    except Exception:
        try:
            self.preview_canvas.original_image = None
            self.preview_canvas.photo_image = None
            self.preview_canvas.image_id = None
            self.preview_canvas.delete("all")
        except Exception:
            pass
    self.current_preview_index = None
    self._preview_drag_state = None
    self._preview_pending_vertex_hit = None
    self._preview_corner_drag_modifier_down = False
    self._preview_last_modifier_press_at = 0.0
    self._preview_focus_zoom_modifier_down = False
    self._preview_focus_zoom_modifier_consumed = False
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_restore_state = None
    self._preview_focus_zoom_history = []
    self._preview_super_correction_active = False
    self._preview_draw_mode = False
    self._preview_draw_points = []
    self._preview_delete_mode = False
    self._preview_delete_candidate_idx = None
    self._preview_polygon_focus_restore_state = None
    self._preview_selected_plate_by_image = {}
    self._preview_selected_vehicle_by_image = {}
    self._preview_focus_target = None
    self._preview_history_undo = {}
    self._preview_history_redo = {}
    self._preview_history_replaying = False
    self._preview_list_display_indices = []
    self._preview_list_display_index_map = {}
    self._invalidate_preview_runtime_caches()
    if clear_dirty:
        self._preview_dirty_images.clear()
    self.current_annotation_run_dir = None
    self.current_annotation_xml_path = None
    self._refresh_preview_list_summary()
    self._update_preview_edit_status()


def _cancel_preview_list_population(self):
    pending = getattr(self, "_preview_list_populate_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_list_populate_after_id = None
    self._preview_list_populate_token = int(getattr(self, "_preview_list_populate_token", 0) or 0) + 1
    self._set_preview_list_population_active(False)


def _should_use_async_preview_list_population(self, count: int | None = None) -> bool:
    try:
        total = int(len(self.current_annotations or []) if count is None else count)
    except Exception:
        total = 0
    return total > 5000


def _cancel_deferred_campaign_source_preview_load(self) -> None:
    pending = getattr(self, "_campaign_deferred_preview_load_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._campaign_deferred_preview_load_after_id = None
    self._campaign_deferred_preview_load_token = int(
        getattr(self, "_campaign_deferred_preview_load_token", 0) or 0
    ) + 1


def _cancel_deferred_campaign_restore_ui(self) -> None:
    pending = getattr(self, "_campaign_deferred_restore_ui_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._campaign_deferred_restore_ui_after_id = None
    self._campaign_deferred_restore_ui_token = int(
        getattr(self, "_campaign_deferred_restore_ui_token", 0) or 0
    ) + 1


def _cancel_deferred_campaign_run_restore(self) -> None:
    self._campaign_deferred_run_restore_in_progress = False
    self._campaign_deferred_run_restore_payload_applied = False
    self._campaign_deferred_run_restore_target_dir = ""
    self._campaign_deferred_run_restore_token = int(
        getattr(self, "_campaign_deferred_run_restore_token", 0) or 0
    ) + 1


def _resolve_run_image_dir_for_annotations(
    self,
    annotations: list[ImageAnnotation],
    manifest: dict | None,
    run_dir: Path,
    *,
    candidate_dir_values: list[str] | None = None,
) -> Path | None:
    image_dir_candidates: list[Path] = []
    seen_candidates: set[str] = set()
    raw_candidates = [
        str((manifest or {}).get("imported_source_input_dir") or "").strip(),
        str((manifest or {}).get("source_input_dir") or "").strip(),
        str((manifest or {}).get("input_dir") or "").strip(),
        *(list(candidate_dir_values or [])),
        str(run_dir / "images"),
        str(run_dir),
    ]
    for raw_value in raw_candidates:
        if not raw_value:
            continue
        try:
            candidate = Path(raw_value)
            candidate_key = str(candidate.resolve())
        except Exception:
            candidate = Path(raw_value)
            candidate_key = str(candidate)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        image_dir_candidates.append(candidate)

    image_dir = None
    fallback_dir = None
    best_match_count = -1
    sample_filenames = [
        str(getattr(ann, "filename", "") or "").strip()
        for ann in annotations
        if str(getattr(ann, "filename", "") or "").strip()
    ][:25]

    for candidate in image_dir_candidates:
        try:
            if not candidate.exists() or not candidate.is_dir():
                continue
        except Exception:
            continue

        if fallback_dir is None:
            fallback_dir = candidate

        match_count = 0
        for filename in sample_filenames:
            try:
                if (candidate / Path(filename)).exists():
                    match_count += 1
            except Exception:
                continue

        if match_count > best_match_count:
            best_match_count = match_count
            image_dir = candidate

    if image_dir is None:
        image_dir = fallback_dir
    return image_dir


def _compute_annotation_run_restore_index(
    self,
    annotations: list[ImageAnnotation],
    manifest: dict | None,
    *,
    restore_filename: str = "",
    restore_index: int | None = None,
) -> int | None:
    def _filename_key(value) -> str:
        return str(value or "").strip().lower()

    def _annotation_has_plate(ann: ImageAnnotation | None) -> bool:
        try:
            return bool(self._get_plate_detections(ann))
        except Exception:
            return False

    resolved_index = None
    safe_restore_filename = str(restore_filename or "").strip()
    if safe_restore_filename:
        safe_restore_filename_key = _filename_key(safe_restore_filename)
        for idx, ann in enumerate(annotations):
            if _filename_key(getattr(ann, "filename", "")) == safe_restore_filename_key:
                resolved_index = idx
                break
    if resolved_index is None and isinstance(restore_index, int) and 0 <= int(restore_index) < len(annotations):
        resolved_index = int(restore_index)
    if resolved_index is None:
        manifest_restore_filename = str((manifest or {}).get("resume_preview_filename") or "").strip()
        if manifest_restore_filename:
            manifest_restore_filename_key = _filename_key(manifest_restore_filename)
            for idx, ann in enumerate(annotations):
                if _filename_key(getattr(ann, "filename", "")) == manifest_restore_filename_key:
                    resolved_index = idx
                    break
    if resolved_index is None:
        try:
            manifest_restore_idx = int((manifest or {}).get("resume_preview_index", -1))
        except (TypeError, ValueError):
            manifest_restore_idx = -1
        if 0 <= manifest_restore_idx < len(annotations):
            resolved_index = manifest_restore_idx
    if resolved_index is None and annotations:
        resolved_index = 0

    approved_names = {
        _filename_key(name)
        for name in list((manifest or {}).get("approved_filenames") or [])
        if _filename_key(name)
    }
    if (
        resolved_index is not None
        and 0 <= int(resolved_index) < len(annotations)
        and not _annotation_has_plate(annotations[int(resolved_index)])
    ):
        fallback_candidates: list[tuple[int, int, int]] = []
        anchor_index = int(resolved_index)
        for idx, ann in enumerate(annotations):
            if not _annotation_has_plate(ann):
                continue
            filename_key = _filename_key(getattr(ann, "filename", ""))
            is_approved = filename_key in approved_names
            try:
                is_manual = bool(self._preview_annotation_has_manual_touch_direct(ann))
            except Exception:
                is_manual = False
            priority = 0 if is_approved else (1 if is_manual else 2)
            fallback_candidates.append((priority, abs(idx - anchor_index), idx))
        if fallback_candidates:
            fallback_candidates.sort()
            resolved_index = int(fallback_candidates[0][2])
    return resolved_index


def _prepare_annotation_run_restore_payload(self, *args, **kwargs):
    return z2_workflow_methods._prepare_annotation_run_restore_payload(self, *args, **kwargs)


def _apply_annotation_run_restore_payload(self, *args, **kwargs):
    return z2_workflow_methods._apply_annotation_run_restore_payload(self, *args, **kwargs)


def _schedule_deferred_campaign_run_restore(
    self,
    run_dir: Path | None,
    *,
    status_message: str | None = None,
    splash_token: int | None = None,
) -> bool:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return False

    self._cancel_deferred_campaign_run_restore()
    self._campaign_deferred_run_restore_in_progress = True
    self._campaign_deferred_run_restore_payload_applied = False
    self._campaign_deferred_run_restore_target_dir = str(safe_run_dir)
    token = int(getattr(self, "_campaign_deferred_run_restore_token", 0) or 0)
    try:
        candidate_dir_values = [
            str(getattr(self, "current_input_dir", "") or "").strip(),
            str(self.input_dir_var.get() or "").strip(),
            str(self.plate_dataset_images_var.get() or "").strip(),
        ]
    except Exception:
        candidate_dir_values = []
    restore_filename = str(getattr(self, "_preview_session_restore_filename", "") or "").strip()
    restore_index = getattr(self, "_preview_session_restore_index", None)

    if status_message:
        try:
            self._set_status_label_state(str(status_message).strip(), "neutral")
        except Exception:
            pass
    scheduled_at = time.perf_counter()

    def worker() -> None:
        started_at = time.perf_counter()
        payload = None
        # Do not bind this background restore to the transient notebook selection.
        # During graph-controlled navigation Tk may report the old tab for a short
        # moment, which used to cancel the worker silently and leave the right
        # panel stuck in the "loading Z2 work" state.
        is_cancelled = lambda: token != int(getattr(self, "_campaign_deferred_run_restore_token", 0) or 0)
        try:
            payload = self._prepare_annotation_run_restore_payload(
                safe_run_dir,
                candidate_dir_values=candidate_dir_values,
                restore_filename=restore_filename,
                restore_index=restore_index,
                is_cancelled=is_cancelled,
            )
        except Exception as e:
            logger.debug(f"Nie udało się przygotować odroczonego restore runu Z2: {e}")
            payload = None

        if is_cancelled():
            return

        prepared_at = time.perf_counter()
        posted_at = {"value": 0.0}

        def apply_payload() -> None:
            apply_started_at = time.perf_counter()
            try:
                queue_wait_ms = (
                    apply_started_at - float(posted_at.get("value") or apply_started_at)
                ) * 1000.0
                prepare_ms = (prepared_at - started_at) * 1000.0
                start_delay_ms = (started_at - scheduled_at) * 1000.0
                if queue_wait_ms >= 250.0 or prepare_ms >= 250.0 or start_delay_ms >= 250.0:
                    logger.info(
                        "[Z2 PERF] deferred_run_restore phases "
                        "start_delay=%.0fms prepare=%.0fms ui_queue=%.0fms payload=%s run=%s",
                        start_delay_ms,
                        prepare_ms,
                        queue_wait_ms,
                        int(bool(payload)),
                        safe_run_dir,
                    )
            except Exception:
                pass
            if token != int(getattr(self, "_campaign_deferred_run_restore_token", 0) or 0):
                return
            if not payload:
                self._campaign_deferred_run_restore_in_progress = False
                self._campaign_deferred_run_restore_payload_applied = False
                self._campaign_deferred_run_restore_target_dir = ""
                try:
                    logger.warning("Odroczone odtworzenie runu Z2 nie zwróciło danych: %s", safe_run_dir)
                except Exception:
                    pass
                try:
                    self._hide_campaign_step2_splash(token=splash_token)
                except Exception:
                    pass
                try:
                    self._refresh_step2_action_states()
                except Exception:
                    pass
                return
            applied = self._apply_annotation_run_restore_payload(
                payload,
                clear_existing_state=False,
                use_async_list=True,
                status_message=(
                    "Przywrócono aktywny run anotacji Z2 dla tego etapu."
                ),
            )
            if not applied:
                self._campaign_deferred_run_restore_payload_applied = False
                self._campaign_deferred_run_restore_in_progress = False
                try:
                    self._refresh_step2_action_states(lightweight=False)
                except Exception:
                    pass
            try:
                self._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass

        posted_at["value"] = time.perf_counter()
        self._post_to_ui(apply_payload)
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 20.0:
            logger.debug(
                "[AnnotationTab][PERF] deferred_campaign_run_restore: "
                f"{elapsed_ms:.1f} ms | run_dir={safe_run_dir}"
            )

    try:
        threading.Thread(target=worker, daemon=True).start()
    except Exception:
        self._campaign_deferred_run_restore_in_progress = False
        self._campaign_deferred_run_restore_payload_applied = False
        self._campaign_deferred_run_restore_target_dir = ""
        return False

    def watchdog() -> None:
        if token != int(getattr(self, "_campaign_deferred_run_restore_token", 0) or 0):
            return
        if not bool(getattr(self, "_campaign_deferred_run_restore_in_progress", False)):
            return
        try:
            logger.warning("Odroczone odtworzenie runu Z2 nadal trwa po timeout; zdejmuję blokadę prawego panelu: %s", safe_run_dir)
        except Exception:
            pass
    try:
        self.frame.after(60000, watchdog)
    except Exception:
        pass
    return True


def _schedule_deferred_campaign_source_preview_load(
    self,
    input_dir: Path | None,
    *,
    status_message: str | None = None,
    splash_token: int | None = None,
) -> bool:
    if input_dir is None:
        return False

    try:
        source_dir = Path(input_dir)
    except Exception:
        return False

    self._cancel_deferred_campaign_source_preview_load()
    token = int(getattr(self, "_campaign_deferred_preview_load_token", 0) or 0)
    try:
        include_previous = bool(self.campaign_reuse_manual_var.get())
    except Exception:
        include_previous = False
    try:
        current_manual_bundle = self._get_current_campaign_manual_preview_bundle()
    except Exception:
        current_manual_bundle = {}
    try:
        approved_filenames = self._get_campaign_plate_approved_filenames()
    except Exception:
        approved_filenames = set()

    if status_message:
        try:
            self._set_status_label_state(str(status_message).strip(), "neutral")
        except Exception:
            pass

    def worker() -> None:
        is_cancelled = lambda: (
            token != int(getattr(self, "_campaign_deferred_preview_load_token", 0) or 0)
            or not self._is_annotation_tab_selected()
        )
        if is_cancelled():
            return
        started_at = time.perf_counter()
        payload = None
        try:
            payload = self._prepare_campaign_source_preview_payload(
                source_dir,
                include_previous=include_previous,
                current_manual_bundle=current_manual_bundle,
                approved_filenames=approved_filenames,
                is_cancelled=is_cancelled,
            )
        except Exception as e:
            logger.debug(f"Nie udało się przygotować odroczonego zestawu Z2: {e}")

        if is_cancelled():
            return

        def apply_payload() -> None:
            if token != int(getattr(self, "_campaign_deferred_preview_load_token", 0) or 0):
                return
            self._campaign_deferred_preview_load_after_id = None
            try:
                if payload:
                    self._apply_campaign_source_preview_payload(payload)
            except Exception as e:
                logger.debug(f"Nie udało się zastosować odroczonego zestawu Z2: {e}")
            try:
                self._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass
            try:
                self._refresh_step2_action_states()
            except Exception:
                pass
            try:
                self._refresh_free_mode_workflow_ui()
            except Exception:
                pass

        self._post_to_ui(apply_payload)
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 20.0:
            logger.debug(
                "[AnnotationTab][PERF] deferred_campaign_source_preview_load: "
                f"{elapsed_ms:.1f} ms | images_dir={source_dir}"
            )

    try:
        threading.Thread(target=worker, daemon=True).start()
    except Exception:
        self._campaign_deferred_preview_load_after_id = None
        worker()
    return True


def _clear_listbox_selection_fast(listbox) -> None:
    if listbox is None:
        return
    try:
        selected_indices = list(listbox.curselection() or ())
    except Exception:
        selected_indices = []
    for selected_index in selected_indices:
        try:
            listbox.selection_clear(selected_index)
        except Exception:
            pass


def _set_preview_list_population_active(self, active: bool) -> None:
    normalized_active = bool(active)
    if normalized_active == bool(getattr(self, "_preview_list_population_active", False)):
        return

    self._preview_list_population_active = normalized_active
    left_frame = getattr(self, "main_left_frame", None)

    if normalized_active:
        if left_frame is None:
            return
        try:
            current_row_cfg = left_frame.grid_rowconfigure(0)
            previous_minsize = int(current_row_cfg.get("minsize", 0) or 0)
        except Exception:
            previous_minsize = 0
        self._left_panel_top_row_minsize_before_preview_population = previous_minsize

        try:
            top_shell = getattr(self, "left_scroll_shell", None)
            if top_shell is not None:
                top_shell.update_idletasks()
                freeze_height = int(top_shell.winfo_height() or top_shell.winfo_reqheight() or 0)
            else:
                freeze_height = 0
        except Exception:
            freeze_height = 0

        if freeze_height > 0:
            try:
                left_frame.grid_rowconfigure(0, minsize=max(previous_minsize, freeze_height))
            except Exception:
                pass
        return

    previous_minsize = getattr(
        self,
        "_left_panel_top_row_minsize_before_preview_population",
        None,
    )
    if left_frame is not None and previous_minsize is not None:
        try:
            left_frame.grid_rowconfigure(0, minsize=int(previous_minsize))
        except Exception:
            pass
    self._left_panel_top_row_minsize_before_preview_population = None

    deferred_layout = bool(
        getattr(self, "_main_pane_layout_deferred_for_preview_population", False)
    )
    deferred_force_defaults = bool(
        getattr(self, "_main_pane_layout_deferred_force_defaults", False)
    )
    self._main_pane_layout_deferred_for_preview_population = False
    self._main_pane_layout_deferred_force_defaults = False
    if deferred_layout:
        try:
            self.frame.after_idle(
                lambda: self._schedule_main_pane_layout_refresh(
                    force_defaults=deferred_force_defaults
                )
            )
        except Exception:
            self._schedule_main_pane_layout_refresh(
                force_defaults=deferred_force_defaults
            )


def _populate_preview_list_async(self, *args, **kwargs):
    return z2_workflow_methods._populate_preview_list_async(self, *args, **kwargs)


def _refresh_preview_list(
    self,
    preserve_selection: bool = True,
    render_current: bool = False,
    on_progress=None,
    *,
    invalidate_runtime: bool = True,
    rebuild_state_cache: bool = True,
    lightweight_summary: bool = False,
    recolor_rows: bool = True,
    refresh_summary: bool = True,
):
    self._cancel_preview_list_population()
    self._set_preview_list_population_active(False)
    if invalidate_runtime:
        self._invalidate_preview_runtime_caches()
    if rebuild_state_cache:
        try:
            self._build_preview_list_render_state_cache(
                list(enumerate(list(getattr(self, "current_annotations", []) or [])))
            )
        except Exception:
            self._preview_list_render_state_cache = None
    entries = self._get_preview_list_entries()
    self._preview_list_display_indices = [actual_idx for actual_idx, _ann in entries]
    self._preview_list_display_index_map = {
        int(actual_idx): int(display_idx)
        for display_idx, actual_idx in enumerate(self._preview_list_display_indices)
    }
    selected_actual_index = self.current_preview_index if preserve_selection else None
    if selected_actual_index is None and entries:
        selected_actual_index = entries[0][0]
    selected_display_index = self._get_preview_display_index(selected_actual_index)
    if selected_display_index is None and entries:
        selected_actual_index = entries[0][0]
        selected_display_index = 0

    default_color_bucket = "auto"
    try:
        if recolor_rows:
            default_color_bucket, default_fg = _preview_list_color_plan(self, entries)
        else:
            palette = getattr(self.app, "palette", {}) or {}
            default_fg = palette.get("success", "#27ae60")
        self.preview_listbox.configure(fg=default_fg)
    except Exception:
        pass
    self.preview_listbox.delete(0, tk.END)
    if refresh_summary:
        self._refresh_preview_list_summary(lightweight=bool(lightweight_summary or len(entries) >= 1200))

    total_count = len(entries)
    lightweight_labels = bool(total_count >= 1200)
    insert_failures: list[tuple[int, str, str]] = []
    for idx, (_actual_idx, ann) in enumerate(entries):
        filename = str(getattr(ann, "filename", "") or "").strip()
        label_text = self._preview_list_item_text(
            ann,
            display_index=idx,
            total_count=total_count,
            lightweight=lightweight_labels,
        )
        primary_insert_error = ""
        try:
            self.preview_listbox.insert(
                tk.END,
                label_text,
            )
        except Exception as exc:
            primary_insert_error = str(exc)
        if recolor_rows:
            bucket = _preview_list_effective_color_bucket(self, ann)
            if bucket != default_color_bucket:
                try:
                    item_color = _preview_list_color_for_bucket(self, bucket)
                    self.preview_listbox.itemconfig(idx, foreground=item_color)
                except Exception:
                    pass
        try:
            current_size = int(self.preview_listbox.size() or 0)
        except Exception:
            current_size = idx + 1
        if primary_insert_error or current_size < (idx + 1):
            fallback_status = self._preview_annotation_status_tag(ann)
            fallback_label = f"  {idx + 1}. [{fallback_status}] {filename or '<brak_nazwy>'}"
            try:
                self.preview_listbox.insert(tk.END, fallback_label)
            except Exception as exc:
                insert_failures.append((idx, filename, primary_insert_error or str(exc)))
                continue
            insert_failures.append((idx, filename, primary_insert_error or "primary_insert_missing"))
        loaded_count = idx + 1
        if callable(on_progress) and (loaded_count % 50 == 0 or loaded_count == total_count):
            try:
                on_progress(loaded_count, total_count)
            except Exception:
                pass

    if selected_display_index is not None and selected_actual_index is not None and entries and not render_current:
        self._suppress_preview_reload_on_list_select = True
    self._clear_listbox_selection_fast(self.preview_listbox)
    if selected_display_index is not None and selected_actual_index is not None and entries:
        self.preview_listbox.selection_set(selected_display_index)
        self.preview_listbox.activate(selected_display_index)
        self.preview_listbox.see(selected_display_index)
        self.current_preview_index = int(selected_actual_index)
        if render_current:
            self._load_current_preview_selection(reset_view=not preserve_selection, selection_changed=True)
        else:
            try:
                self.frame.after(
                    250,
                    lambda: setattr(self, "_suppress_preview_reload_on_list_select", False),
                )
            except Exception:
                self._suppress_preview_reload_on_list_select = False
    else:
        self.current_preview_index = None
        try:
            self.preview_canvas.clear_image()
        except Exception:
            self.preview_canvas.delete("all")
        self._update_preview_toolbar_state()
        self._update_preview_edit_status()

    try:
        final_size = int(self.preview_listbox.size() or 0)
    except Exception:
        final_size = total_count
    try:
        last_label = str(self.preview_listbox.get(tk.END) or "").strip() if final_size > 0 else ""
    except Exception:
        last_label = ""
    failure_preview = "; ".join(
        f"idx={idx} file={name} err={err}"
        for idx, name, err in insert_failures[:8]
    )
    self._append_z2_trace(
        "preview-list-sync",
        (
            f"entries={total_count} listbox={final_size} failures={len(insert_failures)} last={last_label[:120]}"
            + (f" sample={failure_preview}" if failure_preview else "")
        ),
    )


def _populate_preview_list(self, *, on_complete=None, on_progress=None, batch_size: int | None = None) -> bool:
    if self._should_use_async_preview_list_population():
        self._populate_preview_list_async(
            preserve_selection=False,
            render_current=True,
            batch_size=max(1, int(batch_size or 500)),
            on_progress=on_progress,
            on_complete=on_complete,
        )
        return True
    self._refresh_preview_list(
        preserve_selection=False,
        render_current=True,
        on_progress=on_progress,
    )
    if callable(on_complete):
        try:
            on_complete()
        except Exception:
            pass
    return False


def _clear_preview_metric_filters_for_new_run(self) -> bool:
    try:
        conf_threshold, fit_threshold = self._get_preview_metric_filter_thresholds()
    except Exception:
        conf_threshold, fit_threshold = 0.0, 0.0
    if conf_threshold <= 0.0 and fit_threshold <= 0.0:
        return False

    try:
        self.preview_filter_conf_var.set(0.0)
        self.preview_filter_fit_var.set(0.0)
    except Exception:
        pass
    self._preview_filter_conf_applied = 0.0
    self._preview_filter_fit_applied = 0.0
    try:
        self._invalidate_preview_list_frozen_order()
    except Exception:
        pass
    try:
        self._refresh_preview_filter_bar_state()
    except Exception:
        pass
    return True


def _ensure_preview_list_population_consistency(
    self,
    *,
    reason: str = "",
    render_current: bool = False,
) -> None:
    try:
        annotations_count = int(len(self.current_annotations or []))
    except Exception:
        annotations_count = 0
    if annotations_count <= 0 or not hasattr(self, "preview_listbox"):
        return

    if bool(getattr(self, "_preview_list_population_active", False)):
        self._schedule_preview_list_population_consistency_check(
            reason=reason or "population-active",
            delay_ms=180,
            render_current=render_current,
        )
        return

    try:
        entries = self._get_preview_list_entries()
    except Exception:
        entries = []

    filters_reset = False
    if not entries and annotations_count > 0:
        filters_reset = self._clear_preview_metric_filters_for_new_run()
        if filters_reset:
            try:
                entries = self._get_preview_list_entries()
            except Exception:
                entries = []

    expected_count = int(len(entries))
    try:
        listbox_count = int(self.preview_listbox.size() or 0)
    except Exception:
        listbox_count = 0

    if expected_count <= 0:
        return
    if listbox_count == expected_count and not filters_reset:
        return

    try:
        self._append_z2_trace(
            "preview-list-consistency",
            (
                f"reason={str(reason or '').strip()} annotations={annotations_count} "
                f"entries={expected_count} listbox={listbox_count} filters_reset={int(filters_reset)}"
            ),
        )
    except Exception:
        pass

    if self._should_use_async_preview_list_population(expected_count):
        self._populate_preview_list_async(
            preserve_selection=True,
            render_current=False,
            batch_size=500,
            on_complete=(
                (lambda: self._load_current_preview_selection(reset_view=False, selection_changed=False))
                if render_current
                else None
            ),
        )
    else:
        self._refresh_preview_list(
            preserve_selection=True,
            render_current=render_current,
        )


def _schedule_preview_list_population_consistency_check(
    self,
    *,
    reason: str = "",
    delay_ms: int = 120,
    render_current: bool = False,
) -> None:
    pending = getattr(self, "_preview_list_consistency_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_list_consistency_after_id = None

    def _run() -> None:
        self._preview_list_consistency_after_id = None
        self._ensure_preview_list_population_consistency(
            reason=reason,
            render_current=render_current,
        )

    try:
        self._preview_list_consistency_after_id = self.frame.after(
            max(0, int(delay_ms)),
            _run,
        )
    except Exception:
        self._preview_list_consistency_after_id = None
        _run()


def _select_preview_index(self, idx: int, *, reset_view: bool = True):
    if not self.current_annotations:
        return
    select_started = time.perf_counter()
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass
    self._cancel_preview_selection_render()
    self._defer_preview_autosave_for_navigation(delay_ms=4500)
    safe_idx = max(0, min(int(idx), len(self.current_annotations) - 1))
    previous_idx = self.current_preview_index
    if bool(getattr(self, "_preview_fullscreen_active", False)):
        self._preview_list_selection_sync_pending = True
    else:
        display_idx = self._get_preview_display_index(safe_idx)
        if display_idx is None:
            display_idx = max(0, min(safe_idx, max(0, self.preview_listbox.size() - 1)))
        self._clear_listbox_selection_fast(self.preview_listbox)
        self.preview_listbox.selection_set(display_idx)
        self.preview_listbox.activate(display_idx)
        self.preview_listbox.see(display_idx)
    self.current_preview_index = safe_idx
    self._preview_session_restore_index = safe_idx
    self._preview_session_restore_filename = str(getattr(self.current_annotations[safe_idx], "filename", "") or "")
    self._preview_metrics_overlay_render_key = None
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_restore_state = None
    self._preview_focus_zoom_history = []
    self._load_current_preview_selection(
        reset_view=bool(reset_view),
        selection_changed=(safe_idx != previous_idx),
        refresh_summary=False,
    )
    self._schedule_preview_resume_persist(
        include_preview_approved=False,
        delay_ms=900 if bool(getattr(self, "_preview_fullscreen_active", False)) else 120,
    )
    elapsed_ms = max(0.0, (time.perf_counter() - select_started) * 1000.0)
    if elapsed_ms >= 20.0:
        logger.debug(
            "[AnnotationTab][PERF] select_preview_index: "
            f"{elapsed_ms:.1f} ms | idx={safe_idx} total={len(self.current_annotations or [])}"
        )
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    self._refresh_plate_auto_scope_modal_selection_state()


def _get_preview_image_cache_lock(self):
    lock = getattr(self, "_preview_render_image_cache_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._preview_render_image_cache_lock = lock
    return lock


def _get_preview_image_cache_key(img_path: Path):
    try:
        stat = img_path.stat()
        return (
            str(img_path.resolve()),
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )
    except Exception:
        return (str(img_path), 0, 0)


def _load_preview_image_cached(self, img_path: Path, ann=None, *, update_annotation: bool = True):
    cache_key = _get_preview_image_cache_key(img_path)
    image_cache = getattr(self, "_preview_render_image_cache", None)
    if not isinstance(image_cache, dict):
        image_cache = {}
        self._preview_render_image_cache = image_cache

    lock = _get_preview_image_cache_lock(self)
    with lock:
        preview_image = image_cache.get(cache_key)
    if preview_image is None:
        load_started_at = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Nie można załadować obrazu do podglądu.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preview_image = Image.fromarray(img_rgb)
        with lock:
            image_cache[cache_key] = preview_image
            load_elapsed_ms = max(0.0, (time.perf_counter() - load_started_at) * 1000.0)
            if load_elapsed_ms >= 120.0:
                try:
                    logger.info(
                        "[Z2 PERF] preview_image_load_cached total=%.0fms cache=%s file=%s",
                        load_elapsed_ms,
                        len(image_cache),
                        Path(img_path).name,
                    )
                except Exception:
                    pass
            while len(image_cache) > z2_preview_editor.PREVIEW_IMAGE_CACHE_LIMIT:
                try:
                    image_cache.pop(next(iter(image_cache)))
                except Exception:
                    break
    if update_annotation and ann is not None:
        try:
            image_width = max(1, int(getattr(preview_image, "width", 1) or 1))
            image_height = max(1, int(getattr(preview_image, "height", 1) or 1))
            if (
                int(getattr(ann, "width", 0) or 0) != image_width
                or int(getattr(ann, "height", 0) or 0) != image_height
            ):
                ann.width = image_width
                ann.height = image_height
        except Exception:
            pass
    return preview_image


def _schedule_preview_neighbor_prefetch(
    self,
    current_actual: int,
    direction: int,
    nav_indices: list[int],
    *,
    require_plates: bool = True,
) -> None:
    if not self.current_annotations or not nav_indices:
        return
    try:
        current_pos = list(nav_indices).index(int(current_actual))
    except Exception:
        return

    candidates = []
    step = -1 if int(direction) < 0 else 1
    for offset in range(1, 4):
        pos = current_pos + (step * offset)
        if pos < 0 or pos >= len(nav_indices):
            break
        actual_idx = int(nav_indices[pos])
        if actual_idx < 0 or actual_idx >= len(self.current_annotations):
            continue
        ann = self.current_annotations[actual_idx]
        if require_plates and not self._get_plate_detections(ann):
            continue
        img_path = self._resolve_preview_image_path(ann)
        if img_path is not None and img_path.exists():
            candidates.append(Path(img_path))
        if len(candidates) >= 2:
            break
    if not candidates:
        return

    token = int(getattr(self, "_preview_prefetch_token", 0) or 0) + 1
    self._preview_prefetch_token = token

    def worker(paths: list[Path], expected_token: int) -> None:
        for path in paths:
            if expected_token != int(getattr(self, "_preview_prefetch_token", 0) or 0):
                return
            try:
                _load_preview_image_cached(self, path, update_annotation=False)
            except Exception:
                continue

    try:
        threading.Thread(target=worker, args=(candidates, token), daemon=True).start()
    except Exception:
        pass


def _select_preview_index_for_super_correction(self, idx: int):
    if not self.current_annotations:
        return
    select_started = time.perf_counter()
    try:
        self._mark_preview_user_interaction(quiet_ms=1600)
    except Exception:
        pass
    phase_at = select_started
    phase_ms: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal phase_at
        now = time.perf_counter()
        phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
        phase_at = now

    self._defer_preview_autosave_for_navigation(delay_ms=4500)
    safe_idx = max(0, min(int(idx), len(self.current_annotations) - 1))
    previous_idx = self.current_preview_index
    display_idx = self._get_preview_display_index(safe_idx)
    if display_idx is None:
        display_idx = max(0, min(safe_idx, max(0, self.preview_listbox.size() - 1)))
    self._clear_listbox_selection_fast(self.preview_listbox)
    self.preview_listbox.selection_set(display_idx)
    self.preview_listbox.activate(display_idx)
    self.preview_listbox.see(display_idx)
    self.current_preview_index = safe_idx
    self._preview_session_restore_index = safe_idx
    self._preview_session_restore_filename = str(getattr(self.current_annotations[safe_idx], "filename", "") or "")
    self._preview_metrics_overlay_render_key = None
    mark_phase("list_select")
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_restore_state = None
    self._preview_focus_zoom_history = []
    ann = self._get_preview_annotation()
    if ann is None:
        return
    if not bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        try:
            self.preview_canvas.grab_release()
            self._preview_corner_drag_grab_ready = False
        except Exception:
            pass
    self._preview_drag_state = None
    self._preview_pending_vertex_hit = None
    self._preview_draw_mode = False
    self._preview_draw_points = []
    self._preview_delete_mode = False
    self._preview_delete_candidate_idx = None
    self._preview_polygon_focus_restore_state = None
    self._preview_focus_target = None
    self._push_preview_debug_event(
        "select",
        f"idx={self.current_preview_index if self.current_preview_index is not None else '-'} file={getattr(ann, 'filename', '-')}"
    )
    img_path = self._resolve_preview_image_path(ann)
    mark_phase("resolve_path")
    if img_path is None or not img_path.exists():
        self.preview_canvas.clear_image()
        self._clear_preview_legend_image_cache()
        self.preview_canvas.create_text(20, 20, text="Plik nie istnieje na dysku!", fill="red", anchor="nw")
        self._update_preview_toolbar_state(refresh_summary=False)
        return
    try:
        preview_image = _load_preview_image_cached(self, img_path, ann)
        mark_phase("image")
        self._cancel_preview_layout_restore_jobs()
        self._preview_force_fit_after_resize = False
        self._clear_preview_legend_image_cache()
        self.preview_canvas.set_image_preserve_view(preview_image, redraw=False)
        mark_phase("set_image")
        try:
            self._update_preview_canvas_metrics_overlay(force_render=True)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Błąd rysowania podglądu YOLO: {e}")
        self.preview_canvas.clear_image()
        self._clear_preview_legend_image_cache()
        self.preview_canvas.create_text(20, 20, text=f"Błąd podglądu: {e}", fill="red", anchor="nw")
        self._refresh_preview_legend_backdrop()
    elapsed_ms = max(0.0, (time.perf_counter() - select_started) * 1000.0)
    _log_preview_super_perf(
        self,
        "select_image.super",
        elapsed_ms,
        threshold_ms=0.0 if _preview_super_perf_active(self) else 80.0,
        idx=safe_idx,
        changed=int(safe_idx != previous_idx),
        list=f"{phase_ms.get('list_select', 0.0):.1f}",
        resolve=f"{phase_ms.get('resolve_path', 0.0):.1f}",
        image=f"{phase_ms.get('image', 0.0):.1f}",
        set_image=f"{phase_ms.get('set_image', 0.0):.1f}",
    )
    if elapsed_ms >= 20.0:
        logger.debug(
            "[AnnotationTab][PERF] select_preview_index_for_super_correction: "
            f"{elapsed_ms:.1f} ms | idx={safe_idx} total={len(self.current_annotations or [])}"
        )
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass


def _select_preview_relative(self, step: int):
    if not self.current_annotations:
        return "break"
    display_indices = list(getattr(self, "_preview_list_display_indices", []) or [])
    if not display_indices:
        current = 0 if self.current_preview_index is None else int(self.current_preview_index)
        target_actual = max(0, min(current + int(step), len(self.current_annotations) - 1))
        self._select_preview_index(
            target_actual,
            reset_view=True,
        )
        _schedule_preview_neighbor_prefetch(
            self,
            int(target_actual),
            int(step),
            list(range(len(self.current_annotations or []))),
            require_plates=False,
        )
        return "break"

    current_display = self._get_preview_display_index(self.current_preview_index)
    if current_display is None:
        current_display = 0
    target_display = max(0, min(int(current_display) + int(step), len(display_indices) - 1))
    target_actual = self._get_preview_actual_index_from_display(target_display)
    if target_actual is None:
        return "break"
    self._select_preview_index(
        target_actual,
        reset_view=True,
    )
    _schedule_preview_neighbor_prefetch(
        self,
        int(target_actual),
        int(step),
        self._get_preview_navigation_actual_indices(),
        require_plates=False,
    )
    return "break"


def _get_preview_navigation_actual_indices(self) -> list[int]:
    display_indices = list(getattr(self, "_preview_list_display_indices", []) or [])
    if display_indices:
        # _preview_list_display_indices already stores actual annotation indices
        # in the exact order currently visible in the listbox.  Do not map those
        # values again as display positions, otherwise sorted lists make Q/E jump
        # through a different order than the user sees.
        max_index = len(self.current_annotations or []) - 1
        resolved = []
        for actual_idx in display_indices:
            try:
                safe_idx = int(actual_idx)
            except Exception:
                continue
            if 0 <= safe_idx <= max_index:
                resolved.append(safe_idx)
        if resolved:
            return resolved
    return list(range(len(self.current_annotations or [])))


def _select_preview_global_plate_relative(self, step: int):
    if not self.current_annotations:
        return "break"

    switch_started = time.perf_counter()
    _arm_preview_super_perf_probe(self, "global_plate_relative", step=int(step))
    try:
        self._defer_preview_autosave_for_navigation(delay_ms=12000)
    except Exception:
        pass
    direction = -1 if int(step) < 0 else 1
    nav_indices = self._get_preview_navigation_actual_indices()
    if not nav_indices:
        self._update_preview_edit_status(
            "Super korekta jest włączona, ale w tej puli nie ma żadnych tablic do przechodzenia.",
            refresh_toolbar=False,
            refresh_debug=False,
        )
        return "break"

    selected_actual = None
    try:
        selected_rows = tuple(int(row) for row in (self.preview_listbox.curselection() or ()))
        if selected_rows:
            try:
                active_row = int(self.preview_listbox.index(tk.ACTIVE))
            except Exception:
                active_row = int(selected_rows[-1])
            if active_row not in selected_rows:
                active_row = int(selected_rows[-1])
            selected_actual = self._get_preview_actual_index_from_display(active_row)
    except Exception:
        selected_actual = None

    current_actual = selected_actual if selected_actual is not None else self.current_preview_index
    if current_actual is None or int(current_actual) not in nav_indices:
        current_nav_pos = -1 if direction > 0 else len(nav_indices)
    else:
        current_nav_pos = nav_indices.index(int(current_actual))

    try:
        if current_actual is not None and (
            self.current_preview_index is None
            or int(current_actual) != int(self.current_preview_index)
        ):
            self._select_preview_index_for_super_correction(int(current_actual))
    except Exception:
        pass

    current_ann = None
    current_plates = []
    if current_actual is not None and 0 <= int(current_actual) < len(self.current_annotations):
        current_ann = self.current_annotations[int(current_actual)]
        current_plates = self._get_plate_detections(current_ann)

    if current_plates:
        current_plate_idx = self._get_selected_plate_index_for_ann(current_ann)
        if current_plate_idx is None:
            current_plate_idx = 0
        next_plate_idx = int(current_plate_idx) + direction
        if 0 <= next_plate_idx < len(current_plates):
            image_ord = current_nav_pos + 1 if current_nav_pos >= 0 else 1
            image_total = len(nav_indices)
            focus_started = time.perf_counter()
            self._focus_preview_plate(
                int(next_plate_idx),
                store_restore=False,
                push_debug=True,
                status_message=(
                    f"Super korekta: tablica {int(next_plate_idx) + 1}/{len(current_plates)} na obrazie {image_ord}/{image_total}. "
                    "Q/E przechodzą po kolejnych tablicach, A przełącza lokalnie, Y wyłącza tryb."
                ),
            )
            _log_preview_super_perf(
                self,
                "switch.same_image",
                max(0.0, (time.perf_counter() - switch_started) * 1000.0),
                threshold_ms=0.0,
                focus=f"{(time.perf_counter() - focus_started) * 1000.0:.1f}",
                image=current_actual,
                plate=next_plate_idx,
            )
            _schedule_preview_neighbor_prefetch(self, int(current_actual), direction, nav_indices)
            return "break"

    if direction > 0:
        candidate_positions = range(max(0, current_nav_pos + 1), len(nav_indices))
    else:
        candidate_positions = range(min(len(nav_indices) - 1, current_nav_pos - 1), -1, -1)

    for nav_pos in candidate_positions:
        target_actual = int(nav_indices[nav_pos])
        if target_actual < 0 or target_actual >= len(self.current_annotations):
            continue
        target_ann = self.current_annotations[target_actual]
        target_plates = self._get_plate_detections(target_ann)
        if not target_plates:
            continue
        select_started = time.perf_counter()
        self._select_preview_index_for_super_correction(target_actual)
        start_plate_idx = 0 if direction > 0 else (len(target_plates) - 1)
        focus_started = time.perf_counter()
        self._focus_preview_plate(
            int(start_plate_idx),
            store_restore=False,
            push_debug=True,
            status_message=(
                f"Super korekta: tablica {int(start_plate_idx) + 1}/{len(target_plates)} na obrazie {nav_pos + 1}/{len(nav_indices)}. "
                "Q/E przechodzą po kolejnych tablicach, A przełącza lokalnie, Y wyłącza tryb."
            ),
        )
        _log_preview_super_perf(
            self,
            "switch.next_image",
            max(0.0, (time.perf_counter() - switch_started) * 1000.0),
            threshold_ms=0.0,
            select=f"{(focus_started - select_started) * 1000.0:.1f}",
            focus=f"{(time.perf_counter() - focus_started) * 1000.0:.1f}",
            image=target_actual,
            plate=start_plate_idx,
        )
        _schedule_preview_neighbor_prefetch(self, int(target_actual), direction, nav_indices)
        return "break"

    boundary_label = "Pierwsza tablica" if direction < 0 else "Ostatnia tablica"
    self._update_preview_edit_status(
        f"{boundary_label} w aktualnej puli. Y wyłącza super korektę.",
        refresh_toolbar=False,
        refresh_debug=False,
    )
    return "break"


def _fit_preview_image_to_view(self):
    self._preview_polygon_focus_restore_state = None
    self._preview_focus_target = None
    self._preview_focus_zoom_click_stage = 0
    self._preview_focus_zoom_restore_state = None
    self._preview_focus_zoom_history = []
    try:
        if self.preview_canvas.original_image is not None:
            self.preview_canvas.fit_to_view()
            self.preview_canvas.focus_set()
    except Exception:
        pass
    self._push_preview_debug_event("fit", "dopasowano obraz do widoku")


def _get_preview_focus_zoom_target(self, canvas: ZoomableCanvas) -> float:
    base_target = float(canvas._get_middle_click_target_zoom())
    max_zoom = float(getattr(canvas, "max_zoom", base_target) or base_target)
    stage = int(getattr(self, "_preview_focus_zoom_click_stage", 0) or 0)
    if stage <= 0:
        return min(max_zoom, max(base_target + 0.35, base_target * 1.15))
    return max_zoom


def _restore_preview_focus_zoom_view(self) -> bool:
    canvas = getattr(self, "preview_canvas", None)
    history = list(getattr(self, "_preview_focus_zoom_history", []) or [])
    restore_state = history.pop() if history else getattr(self, "_preview_focus_zoom_restore_state", None)
    if canvas is None or not isinstance(restore_state, dict):
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_restore_state = None
        self._preview_focus_zoom_history = []
        return False
    try:
        canvas._animate_to_view_state(restore_state, duration_ms=180)
        return True
    except Exception:
        return False
    finally:
        self._preview_focus_zoom_history = history
        self._preview_focus_zoom_click_stage = max(0, len(history))
        self._preview_focus_zoom_restore_state = history[0] if history else None
