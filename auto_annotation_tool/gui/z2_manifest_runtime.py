#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 annotation-run manifest and dataset-source helpers extracted from tab_annotation.py."""

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
from . import z2_layout_ui_runtime
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

def _format_workspace_relative_path(self, path_like) -> str:
    raw_text = self._path_value_to_text(path_like)
    if not raw_text:
        return ""

    try:
        path = Path(raw_text).resolve()
        workspace = Path(CONFIG.WORKSPACE_DIR).resolve()
        rel = path.relative_to(workspace)
        return str(Path("Workspace") / rel)
    except Exception:
        try:
            path = Path(raw_text).resolve()
        except Exception:
            try:
                path = Path(raw_text)
            except Exception:
                return raw_text

        try:
            parts = [
                str(part or "").strip().replace(":", "")
                for part in path.parts
                if str(part or "").strip() not in {"", ".", "..", "/", "\\", str(path.anchor or "").strip()}
            ]
            if not parts:
                return str(path.name or raw_text)
            if len(parts) <= 3:
                return str(Path(*parts))
            return str(Path("...") / Path(*parts[-3:]))
        except Exception:
            return raw_text


def _get_annotation_run_storage_display_path(self) -> str:
    try:
        base_dir = self._get_annotation_output_base_dir()
    except Exception:
        return "Workspace"
    return self._format_workspace_relative_path(base_dir) or str(base_dir)


def _get_annotation_run_definition_text(self) -> str:
    return (
        "Run autoanotacji Z2 to katalog z plikiem annotations.xml i zgodnymi obrazami "
        "(najczesciej w folderze images/ albo obok XML). Taki run moze tez zawierac "
        "report.txt i run_manifest.json. To nie jest dataset treningowy ani katalog eksportu."
    )


def _show_full_path_dialog(self, path_like, *, title: str = "Pełna ścieżka"):
    full_path = str(path_like or "").strip()
    if not full_path:
        return

    dialog = tk.Toplevel(self.frame)
    dialog.title(title)
    try:
        dialog.transient(getattr(self.app, "root", None) or self.frame.winfo_toplevel())
    except Exception:
        pass
    try:
        dialog.grab_set()
    except Exception:
        pass
    dialog.resizable(True, False)

    body = ttk.Frame(dialog, padding=12)
    body.pack(fill=tk.BOTH, expand=True)

    ttk.Label(body, text="Pełna ścieżka", style="Panel.TLabel").pack(anchor=tk.W, fill=tk.X)

    text = tk.Text(body, height=4, wrap=tk.NONE, bd=1, highlightthickness=0)
    text.pack(fill=tk.BOTH, expand=True, pady=(8, 10))
    text.insert("1.0", full_path)
    text.configure(state="disabled")

    buttons = ttk.Frame(body)
    buttons.pack(fill=tk.X)
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    def copy_path():
        try:
            dialog.clipboard_clear()
            dialog.clipboard_append(full_path)
            dialog.update_idletasks()
        except Exception:
            pass

    ttk.Button(buttons, text="Kopiuj", command=copy_path).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ttk.Button(buttons, text="Zamknij", command=dialog.destroy).grid(row=0, column=1, sticky="ew", padx=(6, 0))


def _bind_full_path_dialog_on_click(self, widget, path_provider, *, title: str):
    if widget is None or not callable(path_provider):
        return

    def open_dialog(_event=None):
        try:
            path_value = path_provider()
        except Exception:
            path_value = ""
        if str(path_value or "").strip():
            self._show_full_path_dialog(path_value, title=title)
        return "break"

    try:
        widget.configure(cursor="hand2")
    except Exception:
        pass

    for sequence in ("<Button-1>", "<Return>", "<space>"):
        try:
            widget.bind(sequence, open_dialog, add="+")
        except Exception:
            pass


def _enable_compact_path_entry(self, entry, path_var, *, title: str):
    if entry is None or path_var is None:
        return

    display_var = tk.StringVar(value=self._format_workspace_relative_path(path_var.get()))
    self._compact_path_display_vars.append(display_var)

    def sync_display(*_args):
        try:
            display_var.set(self._format_workspace_relative_path(path_var.get()))
        except Exception:
            pass

    try:
        path_var.trace_add("write", sync_display)
    except Exception:
        pass

    try:
        entry.configure(textvariable=display_var, state="readonly", cursor="hand2")
    except Exception:
        return

    self._bind_full_path_dialog_on_click(
        entry,
        lambda: str(path_var.get() or "").strip(),
        title=title,
    )


def _prompt_z2_text_input(self, *args, **kwargs):
    return z2_workflow_methods._prompt_z2_text_input(self, *args, **kwargs)


def _annotation_run_manifest_path(self, run_dir: Path) -> Path:
    return Path(run_dir) / "run_manifest.json"


def _write_annotation_run_manifest(self, run_dir: Path, input_dir: Path):
    run_dir = self._resolve_safe_annotation_run_dir(run_dir)
    if run_dir is None:
        return

    manifest_path = self._annotation_run_manifest_path(run_dir)
    source_input_dir = getattr(self, "_annotation_source_input_dir", None)
    if not source_input_dir:
        source_input_dir = input_dir
    payload = {
        "input_dir": str(Path(input_dir).resolve()),
        "source_input_dir": str(Path(source_input_dir).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
        "mode": str(self.mode_var.get() or "").strip(),
        "device": str(self.device_var.get() or "").strip(),
        "annotation_run_type": (
            "manual_template" if getattr(self, "_current_run_manual_template", False) else "auto_annotation"
        ),
        "manual_xml_template": bool(getattr(self, "_current_run_manual_template", False)),
        "manual_vehicle_assist": bool(getattr(self, "_current_run_manual_vehicle_assist", False)),
        "has_manual_edits": False,
        "last_manual_edit_at": "",
        "last_manual_edit_kind": "",
        "run_status": "created",
        "completed_at": "",
        "last_error": "",
        "result_total_images": 0,
        "result_successful_images": 0,
        "result_total_plates": 0,
        "selected_count": 0,
        "source_plan_total_count": 0,
        "resume_preview_index": -1,
        "resume_preview_filename": "",
        "resume_preview_saved_at": "",
        "manual_touched_filenames": [],
        "approved_filenames": [],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(self._collect_plate_model_manifest_fields())
    payload.update(self._collect_annotation_run_scope_manifest_fields())
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    PROJECT_CACHE.invalidate_json(manifest_path)


def _load_annotation_run_manifest(self, run_dir: Path) -> dict:
    manifest_path = self._annotation_run_manifest_path(run_dir)
    payload = PROJECT_CACHE.load_json(manifest_path, default={})
    return payload if isinstance(payload, dict) else {}


def _update_annotation_run_manifest(self, run_dir: Path, **fields) -> bool:
    run_dir = self._resolve_safe_annotation_run_dir(run_dir)
    if run_dir is None:
        return False

    manifest = self._load_annotation_run_manifest(run_dir)
    if not isinstance(manifest, dict):
        manifest = {}

    for key, value in fields.items():
        manifest[key] = value

    try:
        manifest_path = self._annotation_run_manifest_path(run_dir)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        PROJECT_CACHE.invalidate_json(manifest_path)
        try:
            self._campaign_manual_touched_cache = None
        except Exception:
            pass
        try:
            from ..campaign_manager import CAMPAIGN

            CAMPAIGN.invalidate_step3_char_source_state_cache()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _collect_preview_resume_manifest_fields(self) -> dict:
    selected_ann = self._get_preview_annotation()
    return {
        "resume_preview_index": (
            int(self.current_preview_index)
            if self.current_preview_index is not None
            else -1
        ),
        "resume_preview_filename": str(getattr(selected_ann, "filename", "") or "").strip(),
        "resume_preview_saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _load_annotation_run_manual_touched_filenames(self, run_dir: Path | None) -> set[str]:
    if run_dir is None:
        return set()

    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir)
    if safe_run_dir is None:
        return set()

    try:
        manifest = self._load_annotation_run_manifest(safe_run_dir)
    except Exception:
        manifest = {}

    filenames: set[str] = set()
    for entry in list(manifest.get("manual_touched_filenames") or []):
        safe_name = str(entry or "").strip()
        if safe_name:
            filenames.add(safe_name)
    return filenames


def _load_annotation_run_approved_filenames(self, run_dir: Path | None) -> set[str]:
    if run_dir is None:
        return set()

    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir)
    if safe_run_dir is None:
        return set()

    try:
        manifest = self._load_annotation_run_manifest(safe_run_dir)
    except Exception:
        manifest = {}

    filenames: set[str] = set()
    for entry in list(manifest.get("approved_filenames") or []):
        safe_name = str(entry or "").strip().lower()
        if safe_name:
            filenames.add(safe_name)
    return filenames


def _get_campaign_hidden_project_approved_filenames_runtime(self) -> set[str]:
    if self._is_free_mode_session_context():
        return set()
    from .z2_shared_ui import is_campaign_t02_at_review_context

    if is_campaign_t02_at_review_context(self):
        # T02 explicitly reviews existing AT, including the approved project pool.
        return set()
    hidden_source = {
        str(entry or "").strip().lower()
        for entry in set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
        if str(entry or "").strip()
    }
    project_name = ""
    approved_manifest_token: tuple[str, int, int] | None = None
    try:
        project_name = str(getattr(self, "_campaign_context_project_name", "") or "").strip()
    except Exception:
        project_name = ""
    try:
        from ..campaign_manager import CAMPAIGN

        if not project_name:
            project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        pass

    fast_key = (
        project_name,
        tuple(sorted(hidden_source)),
    )
    cache = getattr(self, "_campaign_hidden_project_approved_runtime_cache", None)
    try:
        now = time.monotonic()
    except Exception:
        now = 0.0
    if (
        isinstance(cache, dict)
        and cache.get("fast_key") == fast_key
        and float(cache.get("expires_at", 0.0) or 0.0) > now
    ):
        return set(cache.get("names") or set())

    try:
        from ..campaign_manager import CAMPAIGN

        approved_path = CAMPAIGN.get_plate_approved_set_path(project_name or None)
        if approved_path is not None:
            try:
                stat = Path(approved_path).stat()
                approved_manifest_token = (
                    str(Path(approved_path).resolve()),
                    int(getattr(stat, "st_mtime_ns", 0) or 0),
                    int(getattr(stat, "st_size", 0) or 0),
                )
            except Exception:
                approved_manifest_token = (str(approved_path), 0, 0)
    except Exception:
        approved_manifest_token = None

    cache_key = (
        project_name,
        approved_manifest_token,
        tuple(sorted(hidden_source)),
    )
    cache = getattr(self, "_campaign_hidden_project_approved_runtime_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return set(cache.get("names") or set())

    hidden = set(hidden_source)
    try:
        project_approved = set(self._get_campaign_plate_approved_filenames() or set())
    except Exception:
        project_approved = set()
    for entry in project_approved:
        raw_text = str(entry or "").strip()
        if not raw_text:
            continue
        hidden.add(raw_text.lower())
        try:
            hidden.add(str(Path(raw_text.replace("\\", "/")).name or "").strip().lower())
        except Exception:
            hidden.add(raw_text.replace("\\", "/").rsplit("/", 1)[-1].strip().lower())
    result = {name for name in hidden if name}
    try:
        self._campaign_hidden_project_approved_runtime_cache = {
            "fast_key": fast_key,
            "key": cache_key,
            "names": set(result),
            "expires_at": now + 60.0,
        }
    except Exception:
        pass
    return result


def _get_preview_approved_filenames_base(self) -> set[str]:
    approved: set[str] = set()
    for entry in set(getattr(self, "_preview_approved_filenames", set()) or set()):
        safe_name = str(entry or "").strip().lower()
        if safe_name:
            approved.add(safe_name)
    if not self._is_free_mode_session_context():
        for entry in set(getattr(self, "_campaign_pending_approved_filenames", set()) or set()):
            safe_name = str(entry or "").strip().lower()
            if safe_name:
                approved.add(safe_name)
        hidden_project_approved = _get_campaign_hidden_project_approved_filenames_runtime(self)
        if hidden_project_approved:
            approved = {name for name in approved if name not in hidden_project_approved}
    return approved


def _preview_annotation_can_be_approved_for_export(self, ann) -> bool:
    try:
        return len(self._get_plate_detections(ann)) > 0
    except Exception:
        return False


def _get_preview_approved_filenames(self) -> set[str]:
    approved = self._get_preview_approved_filenames_base()
    hidden_project_approved = _get_campaign_hidden_project_approved_filenames_runtime(self)
    annotations = list(getattr(self, "current_annotations", []) or [])
    if annotations:
        exportable_names = {
            str(getattr(ann, "filename", "") or "").strip().lower()
            for ann in annotations
            if self._preview_annotation_can_be_approved_for_export(ann)
            and str(getattr(ann, "filename", "") or "").strip()
        }
        approved = {name for name in approved if name in exportable_names}

    for ann in annotations:
        try:
            if (
                bool(getattr(ann, "_approved_for_training", False))
                and self._preview_annotation_can_be_approved_for_export(ann)
            ):
                safe_name = str(getattr(ann, "filename", "") or "").strip().lower()
                if safe_name and safe_name not in hidden_project_approved:
                    approved.add(safe_name)
        except Exception:
            continue
    return approved


def _sync_preview_approval_flags_from_current_sets(
    self,
    *,
    refresh_list: bool = False,
    render_current: bool = False,
) -> bool:
    annotations = list(getattr(self, "current_annotations", []) or [])
    if not annotations:
        return False

    approved_names = set(self._get_preview_approved_filenames_base() or set())
    changed = False
    for ann in annotations:
        filename = str(getattr(ann, "filename", "") or "").strip().lower()
        approved = bool(
            filename
            and filename in approved_names
            and self._preview_annotation_can_be_approved_for_export(ann)
        )
        try:
            if bool(getattr(ann, "_approved_for_training", False)) != approved:
                setattr(ann, "_approved_for_training", approved)
                changed = True
        except Exception:
            continue

    if refresh_list:
        try:
            self._refresh_preview_list(preserve_selection=True, render_current=render_current)
        except Exception:
            pass
        try:
            self._refresh_preview_list_summary()
        except Exception:
            pass
        try:
            self._refresh_step2_action_states()
        except Exception:
            pass

    return changed


def _filter_campaign_project_approved_annotations(
    self,
    annotations: list[ImageAnnotation] | None,
    *,
    image_dir: Path | None,
    run_dir: Path | None = None,
    run_approved_filenames: set[str] | None = None,
    extra_hidden_filenames: set[str] | None = None,
    preserve_manual_hidden_filenames: set[str] | None = None,
) -> tuple[list[ImageAnnotation], set[str], int]:
    def _normalize_image_name(value) -> str:
        try:
            return str(Path(str(value or "").replace("\\", "/")).name or "").strip().lower()
        except Exception:
            return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip().lower()

    filtered_annotations = list(annotations or [])
    filtered_run_approved = {
        _normalize_image_name(name)
        for name in set(run_approved_filenames or set())
        if _normalize_image_name(name)
    }
    filtered_extra_hidden = {
        _normalize_image_name(name)
        for name in set(extra_hidden_filenames or set())
        if _normalize_image_name(name)
    }
    preserve_manual_hidden = {
        _normalize_image_name(name)
        for name in set(preserve_manual_hidden_filenames or set())
        if _normalize_image_name(name)
    }
    if self._is_free_mode_session_context() or not filtered_annotations:
        try:
            self._last_campaign_project_approved_filter_hidden_filenames = set()
        except Exception:
            pass
        return filtered_annotations, filtered_run_approved, 0

    project_approved_filenames = {
        _normalize_image_name(name)
        for name in set(self._get_campaign_plate_approved_filenames() or set())
        if _normalize_image_name(name)
    }
    project_approved_source_keys = self._get_campaign_plate_approved_source_keys()
    if not project_approved_filenames and not project_approved_source_keys:
        try:
            from ..campaign_manager import CAMPAIGN

            project_name = ""
            resolver = getattr(self, "_resolve_campaign_context_project_name", None)
            if callable(resolver):
                try:
                    project_name = str(resolver(run_dir, image_dir) or "").strip()
                except Exception:
                    project_name = ""
            if not project_name:
                project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
            approved_entries = list(CAMPAIGN.list_plate_approved_entries(project_name or None) or [])
        except Exception:
            approved_entries = []

        for entry in approved_entries:
            if not isinstance(entry, dict):
                continue
            for raw_name in (entry.get("image_name", ""), entry.get("entry_key", "")):
                normalized_name = _normalize_image_name(raw_name)
                if normalized_name:
                    project_approved_filenames.add(normalized_name)
            source_image_path = str(entry.get("source_image_path", "") or "").strip()
            if source_image_path:
                try:
                    source_key = self._build_campaign_source_image_key(source_image_path)
                except Exception:
                    source_key = ""
                if source_key:
                    project_approved_source_keys.add(source_key)
    if not project_approved_filenames and not project_approved_source_keys and not filtered_extra_hidden:
        try:
            self._last_campaign_project_approved_filter_hidden_filenames = set()
        except Exception:
            pass
        return filtered_annotations, filtered_run_approved, 0

    visible_annotations: list[ImageAnnotation] = []
    visible_names: set[str] = set()
    removed_project_approved_names: set[str] = set()
    removed_count = 0
    safe_image_dir = image_dir if isinstance(image_dir, Path) else None

    for ann in filtered_annotations:
        filename = str(getattr(ann, "filename", "") or "").strip()
        filename_key = _normalize_image_name(filename)
        if not filename_key:
            visible_annotations.append(ann)
            continue

        source_key = ""
        if safe_image_dir is not None:
            try:
                candidate_path = Path(filename)
                if not candidate_path.is_absolute():
                    candidate_path = safe_image_dir / filename
                source_key = self._build_campaign_source_image_key(candidate_path)
            except Exception:
                source_key = ""

        is_project_approved = bool(
            (source_key and source_key in project_approved_source_keys)
            or filename_key in project_approved_filenames
        )
        manual_hidden_override = bool(
            filename_key in preserve_manual_hidden
            and (
                self._preview_annotation_has_manual_touch_direct(ann)
                or filename_key in filtered_run_approved
            )
        )
        if (is_project_approved or filename_key in filtered_extra_hidden) and not manual_hidden_override:
            removed_count += 1
            if is_project_approved:
                removed_project_approved_names.add(filename_key)
            continue

        visible_annotations.append(ann)
        visible_names.add(filename_key)

    filtered_run_approved = {
        safe_name for safe_name in filtered_run_approved
        if safe_name in visible_names
    }
    try:
        self._last_campaign_project_approved_filter_hidden_filenames = set(removed_project_approved_names)
    except Exception:
        pass
    return visible_annotations, filtered_run_approved, removed_count


def _persist_preview_approved_filenames(self) -> bool:
    run_dir = self._get_active_annotation_run_dir(require_xml=False)
    if run_dir is None:
        self._append_z2_trace(
            "approved-persist-skip",
            f"run=None count={len(self._get_preview_approved_filenames())}",
        )
        return False
    try:
        approved_payload = set(self._get_preview_approved_filenames())
        if not self._is_free_mode_session_context():
            approved_payload |= {
                str(name or "").strip().lower()
                for name in set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
                if str(name or "").strip()
            }
        approved_payload = sorted(approved_payload)
        self._append_z2_trace(
            "approved-persist",
            f"run={run_dir} count={len(approved_payload)} sample={approved_payload[:8]}",
        )
        return bool(
            self._update_annotation_run_manifest(
                run_dir,
                approved_filenames=approved_payload,
                **self._collect_preview_resume_manifest_fields(),
            )
        )
    except Exception:
        return False


def _preview_annotation_is_explicitly_approved(self, ann, approved_names: set[str] | None = None) -> bool:
    if not self._preview_annotation_can_be_approved_for_export(ann):
        return False
    filename = str(getattr(ann, "filename", "") or "").strip().lower()
    if not filename:
        return False
    if filename in _get_campaign_hidden_project_approved_filenames_runtime(self):
        return False
    try:
        if bool(getattr(ann, "_approved_for_training", False)):
            return True
    except Exception:
        pass
    if approved_names is None:
        approved_names = self._get_preview_approved_filenames_base()
    return filename in approved_names


def _get_campaign_manual_touched_filenames(self) -> set[str]:
    if self._is_free_mode_session_context():
        return set()

    try:
        overlay_keys = tuple(
            sorted(
                str(name or "").strip().lower()
                for name in dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {}).keys()
                if str(name or "").strip()
            )
        )
    except Exception:
        overlay_keys = tuple()

    candidate_run_dirs = [
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
    ]
    cache_key = (
        tuple(str(run_dir or "") for run_dir in candidate_run_dirs),
        overlay_keys,
    )
    cached = getattr(self, "_campaign_manual_touched_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
        try:
            return set(cached[1] or set())
        except Exception:
            return set()

    touched: set[str] = set()
    for run_dir in candidate_run_dirs:
        try:
            for filename in self._load_annotation_run_manual_touched_filenames(run_dir):
                safe_name = str(filename or "").strip().lower()
                if safe_name:
                    touched.add(safe_name)
        except Exception:
            continue

    try:
        for filename, payload in dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {}).items():
            safe_name = str(filename or "").strip().lower()
            if not safe_name:
                continue
            ann = payload[0] if isinstance(payload, tuple) and len(payload) >= 1 else None
            if ann is None or not self._preview_annotation_has_manual_touch_direct(ann):
                continue
            touched.add(safe_name)
    except Exception:
        pass

    try:
        self._campaign_manual_touched_cache = (cache_key, set(touched))
    except Exception:
        pass
    return touched


def _remember_annotation_run_resume_state(self, run_dir: Path | None = None) -> bool:
    started_at = time.perf_counter()
    candidate = run_dir
    if candidate is None:
        candidate = getattr(self, "current_annotation_run_dir", None)
    if candidate is None:
        candidate = getattr(self, "last_staging_run_dir", None)
    if candidate is None:
        run_dir_value = str(self.plate_dataset_run_var.get() or "").strip()
        candidate = Path(run_dir_value) if run_dir_value else None

    if candidate is None:
        return False

    candidate = self._resolve_safe_annotation_run_dir(candidate)
    if candidate is None:
        return False

    result = self._update_annotation_run_manifest(
        candidate,
        **self._collect_preview_resume_manifest_fields(),
    )
    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    if elapsed_ms >= 120.0:
        try:
            logger.info("[Z2 PERF] resume_manifest_save total=%.0fms", elapsed_ms)
        except Exception:
            pass
    return result


def _mark_annotation_run_completed(
    self,
    run_dir: Path | None,
    annotations: list[ImageAnnotation] | None = None,
    *,
    manual_template: bool = False,
    report: AnnotationReport | None = None,
) -> bool:
    if run_dir is None:
        return False

    records = list(annotations or [])
    total_images = len(records)
    successful_images = (
        total_images
        if manual_template
        else sum(1 for ann in records if getattr(ann, "is_successful", False))
    )
    total_plates = sum(len(self._get_plate_detections(ann)) for ann in records)

    return self._update_annotation_run_manifest(
        run_dir,
        run_status="completed",
        completed_at=datetime.datetime.now().isoformat(timespec="seconds"),
        last_error="",
        result_total_images=int(total_images),
        result_successful_images=int(successful_images),
        result_total_plates=int(total_plates),
        selected_count=int(total_images),
        source_plan_total_count=int(total_images),
        result_report_errors=int(getattr(report, "errors", 0) or 0),
        result_report_skipped=int(getattr(report, "skipped", 0) or 0),
    )


def _restore_campaign_step2_generated_from_run(
    self,
    run_dir: Path | None = None,
    *,
    only_when_pending: bool = False,
) -> bool:
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return False

    if not CAMPAIGN.get_active_project_name():
        return False

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
    except Exception:
        current_step = 0

    if current_step != 2:
        return False

    step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
    if only_when_pending and step2_status not in {"", "pending"}:
        return False

    candidate = self._resolve_existing_run_dir(run_dir)
    if candidate is None:
        candidate = self._resolve_existing_run_dir(getattr(self, "current_annotation_run_dir", None))
    if candidate is None:
        candidate = self._resolve_existing_run_dir(getattr(self, "last_staging_run_dir", None))
    if candidate is None:
        return False

    xml_path = candidate / "annotations.xml"
    if not xml_path.exists():
        return False

    try:
        staging_root = CAMPAIGN.get_staging_dir("auto_ann")
    except Exception:
        staging_root = None

    if staging_root is not None and not self._path_is_within(candidate, staging_root):
        return False

    current_saved_run = str(CAMPAIGN.get_step2_staging_run() or "").strip()
    if current_saved_run and self._paths_equivalent(current_saved_run, candidate) and step2_status == "generated":
        return False

    CAMPAIGN.set_step2_generated(str(candidate))
    return True


def _annotation_run_manifest_has_manual_value(manifest: dict | None) -> bool:
    if not isinstance(manifest, dict):
        return False

    if bool(manifest.get("has_manual_edits", False)):
        return True

    if bool(manifest.get("manual_xml_template", False)):
        return True

    annotation_run_type = str(manifest.get("annotation_run_type") or "").strip().lower()
    return annotation_run_type == "manual_template"


def _annotation_run_manifest_is_manual_template(manifest: dict | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    if bool(manifest.get("manual_xml_template", False)):
        return True
    annotation_run_type = str(manifest.get("annotation_run_type") or "").strip().lower()
    return annotation_run_type == "manual_template"


def _get_active_annotation_run_dir(self, *, require_xml: bool = False) -> Path | None:
    for candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
    ):
        safe_run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=require_xml)
        if safe_run_dir is not None:
            return safe_run_dir
    return None


def _has_active_manual_template_run(self) -> bool:
    run_dir = self._get_active_annotation_run_dir(require_xml=True)
    if run_dir is None:
        return False

    current_input_dir = str(self.input_dir_var.get() or "").strip()
    if not current_input_dir:
        return False

    try:
        manifest = self._load_annotation_run_manifest(run_dir)
    except Exception:
        manifest = {}

    manifest_input_dir = str(manifest.get("input_dir") or "").strip() if isinstance(manifest, dict) else ""
    if manifest_input_dir and not self._paths_equivalent(manifest_input_dir, current_input_dir):
        return False

    if bool(getattr(self, "_current_run_manual_template", False)):
        return True

    try:
        manifest = self._load_annotation_run_manifest(run_dir)
    except Exception:
        manifest = {}
    return bool(self._annotation_run_manifest_is_manual_template(manifest))


def _remember_campaign_manual_plate_source(
    self,
    run_dir: Path | None = None,
    xml_path: Path | None = None,
    input_dir: Path | None = None,
) -> None:
    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return
    except Exception:
        return

    source_run = Path(run_dir) if run_dir is not None else None
    source_xml = Path(xml_path) if xml_path is not None else None
    source_input = Path(input_dir) if input_dir is not None else None

    if source_xml is None and source_run is not None:
        source_xml = source_run / "annotations.xml"
    if source_run is None and source_xml is not None:
        source_run = source_xml.parent
    if source_input is None and getattr(self, "current_input_dir", None) is not None:
        try:
            source_input = Path(self.current_input_dir)
        except Exception:
            source_input = None

    try:
        CAMPAIGN.set_last_plate_manual_source(
            source_run_path=(str(source_run.resolve()) if source_run is not None and source_run.exists() else str(source_run or "")),
            source_xml_path=(str(source_xml.resolve()) if source_xml is not None and source_xml.exists() else str(source_xml or "")),
            source_input_path=(str(source_input.resolve()) if source_input is not None and source_input.exists() else str(source_input or "")),
        )
    except Exception:
        pass

    try:
        self._sync_campaign_iteration_artifact_registry(
            run_dir=source_run,
            xml_path=source_xml,
            input_dir=source_input,
        )
    except Exception:
        pass


def _sync_campaign_iteration_artifact_registry(self, *args, **kwargs):
    return z2_workflow_methods._sync_campaign_iteration_artifact_registry(self, *args, **kwargs)


def _plate_dataset_source_manifest_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "dataset_source_manifest.json"


def _load_plate_dataset_source_manifest(self, dataset_dir: Path) -> dict:
    manifest_path = self._plate_dataset_source_manifest_path(dataset_dir)
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_plate_dataset_source_manifest(
    self,
    dataset_dir: Path,
    *,
    source_kind: str,
    source_run_dir: Path | None = None,
    source_xml_path: Path | None = None,
    source_images_dir: Path | None = None,
) -> None:
    try:
        dataset_dir = Path(dataset_dir)
    except Exception:
        return

    if not self._path_is_within(dataset_dir, self._get_plate_dataset_base_dir()):
        return

    payload = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": "",
        "iteration": 0,
        "dataset_dir": str(dataset_dir.resolve()),
        "source_kind": str(source_kind or "").strip(),
        "source_run_dir": "",
        "source_run_name": "",
        "source_xml_path": "",
        "source_images_dir": "",
    }

    try:
        from ..campaign_manager import CAMPAIGN

        payload["project"] = str(CAMPAIGN.get_active_project_name() or "").strip()
        payload["iteration"] = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        pass

    if source_run_dir is not None:
        try:
            source_run_dir = Path(source_run_dir)
            payload["source_run_dir"] = str(source_run_dir.resolve())
            payload["source_run_name"] = source_run_dir.name
        except Exception:
            payload["source_run_dir"] = str(source_run_dir)
            payload["source_run_name"] = str(Path(source_run_dir).name)

    if source_xml_path is not None:
        try:
            payload["source_xml_path"] = str(Path(source_xml_path).resolve())
        except Exception:
            payload["source_xml_path"] = str(source_xml_path)

    if source_images_dir is not None:
        try:
            payload["source_images_dir"] = str(Path(source_images_dir).resolve())
        except Exception:
            payload["source_images_dir"] = str(source_images_dir)

    manifest_path = self._plate_dataset_source_manifest_path(dataset_dir)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_campaign_plate_approved_entries_from_run(self, *args, **kwargs):
    return z2_workflow_methods._build_campaign_plate_approved_entries_from_run(self, *args, **kwargs)


def _build_campaign_plate_entry_key(*, image_name: str = "", source_image_path=None) -> str:
    source_text = str(source_image_path or "").strip()
    if source_text:
        try:
            return str(Path(source_text).resolve()).strip().lower()
        except Exception:
            return source_text.replace("\\", "/").strip().lower()
    return str(image_name or "").strip().lower()


def _get_campaign_plate_entry_merge_key(self, entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return ""
    source_image_path = str(entry.get("source_image_path", "") or "").strip()
    image_name = str(entry.get("image_name", "") or "").strip()
    merged_key = self._build_campaign_plate_entry_key(
        image_name=image_name,
        source_image_path=source_image_path,
    )
    if merged_key:
        return merged_key
    return str(entry.get("entry_key", "") or image_name).strip().lower()


def _resolve_campaign_project_name_from_run_dir(run_dir: Path | str | None) -> str:
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return ""

    try:
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        active_project = ""
    if active_project:
        return active_project

    try:
        safe_run_dir = Path(run_dir).resolve() if run_dir is not None else None
    except Exception:
        try:
            safe_run_dir = Path(run_dir) if run_dir is not None else None
        except Exception:
            safe_run_dir = None
    if safe_run_dir is None:
        return ""

    try:
        projects = dict((CAMPAIGN.state or {}).get("projects", {}) or {})
    except Exception:
        projects = {}
    for project_name, project_data in projects.items():
        if not isinstance(project_data, dict):
            continue
        folder_name = str(project_data.get("folder_name", "") or "").strip()
        if not folder_name:
            continue
        try:
            project_root = (Path(CONFIG.DIR_9_PROJECTS) / folder_name).resolve()
        except Exception:
            project_root = Path(CONFIG.DIR_9_PROJECTS) / folder_name
        try:
            if safe_run_dir == project_root or project_root in safe_run_dir.parents:
                return str(project_name or "").strip()
        except Exception:
            run_text = str(safe_run_dir).lower()
            root_text = str(project_root).lower().rstrip("\\/")
            if run_text == root_text or run_text.startswith(root_text + os.sep):
                return str(project_name or "").strip()
    return ""


def _resolve_campaign_annotation_run_dir(run_dir: Path | str | None, project_name: str = "") -> Path | None:
    try:
        requested = Path(run_dir) if run_dir is not None else None
    except Exception:
        requested = None
    if requested is None:
        return None

    try:
        if requested.exists() and requested.is_dir() and (requested / "annotations.xml").exists():
            return requested.resolve()
    except Exception:
        pass

    run_name = str(requested.name or "").strip()
    if not run_name:
        return None

    search_roots: list[Path] = []
    try:
        from ..campaign_manager import CAMPAIGN

        resolved_project = str(project_name or "").strip() or _resolve_campaign_project_name_from_run_dir(requested)
        if resolved_project:
            project_root = CAMPAIGN.get_project_root_dir(resolved_project)
            search_roots.extend([
                project_root / "2_auto_annotations",
                project_root / "_staging" / "auto_annotations",
            ])
    except Exception:
        pass

    try:
        parent = requested.parent
        if parent:
            search_roots.append(parent)
    except Exception:
        pass

    seen: set[str] = set()
    for root in search_roots:
        try:
            root_key = str(root.resolve()).strip().lower()
        except Exception:
            root_key = str(root).strip().lower()
        if not root_key or root_key in seen:
            continue
        seen.add(root_key)
        candidate = root / run_name
        try:
            if candidate.exists() and candidate.is_dir() and (candidate / "annotations.xml").exists():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _get_campaign_plate_approved_run_pool_state(
    self,
    run_dir: Path | str | None,
    *,
    project_name: str | None = None,
) -> dict:
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {"ok": False, "reason": "campaign_unavailable"}

    resolved_project_name = str(project_name or "").strip() or _resolve_campaign_project_name_from_run_dir(run_dir)
    if not resolved_project_name:
        return {"ok": False, "reason": "campaign_inactive"}

    resolved_run_dir = _resolve_campaign_annotation_run_dir(run_dir, resolved_project_name)
    if resolved_run_dir is None:
        return {"ok": False, "reason": "missing_run_dir", "run_dir": str(run_dir or "")}

    try:
        manifest = self._load_annotation_run_manifest(resolved_run_dir)
    except Exception:
        manifest = {}

    try:
        approved_names = set(self._load_annotation_run_approved_filenames(resolved_run_dir) or set())
    except Exception:
        approved_names = set()

    try:
        current_run_dir = getattr(self, "current_annotation_run_dir", None)
        if current_run_dir is not None and self._paths_equivalent(current_run_dir, resolved_run_dir):
            approved_names.update(set(self._get_preview_approved_filenames() or set()))
    except Exception:
        pass

    normalized_approved_names = {
        CAMPAIGN._normalize_image_set_name(name)
        for name in set(approved_names or set())
        if CAMPAIGN._normalize_image_set_name(name)
    }
    explicit_approval = bool(normalized_approved_names)
    if not normalized_approved_names:
        # Older campaign runs did not persist approval flags; for those runs,
        # every image with a plate is the effective approved contribution.
        if isinstance(manifest, dict) and "approved_filenames" not in manifest:
            try:
                normalized_approved_names = set(
                    dict(CAMPAIGN._load_run_plate_counts_by_image(resolved_run_dir) or {}).keys()
                )
            except Exception:
                normalized_approved_names = set()

    if not normalized_approved_names:
        return {
            "ok": False,
            "reason": "missing_approved_names",
            "run_dir": str(resolved_run_dir),
        }

    try:
        run_counts = dict(
            CAMPAIGN._load_run_plate_counts_by_image(
                resolved_run_dir,
                image_names=set(normalized_approved_names) if explicit_approval else None,
            )
            or {}
        )
    except Exception:
        run_counts = {}

    approved_plate_names = {
        CAMPAIGN._normalize_image_set_name(name)
        for name, plate_count in run_counts.items()
        if CAMPAIGN._normalize_image_set_name(name) and int(plate_count or 0) > 0
    }
    if not approved_plate_names:
        return {
            "ok": False,
            "reason": "missing_approved_plates",
            "run_dir": str(resolved_run_dir),
            "approved_names": len(normalized_approved_names),
        }

    project_counts_by_name: dict[str, int] = {}
    project_entry_keys: set[str] = set()
    try:
        approved_entries = list(CAMPAIGN.list_plate_approved_entries(resolved_project_name) or [])
    except Exception:
        approved_entries = []
    for entry in approved_entries:
        if not isinstance(entry, dict):
            continue
        entry_key = str(entry.get("entry_key", "") or "").strip().lower()
        if entry_key:
            project_entry_keys.add(entry_key)
        safe_name = CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
        if not safe_name:
            continue
        plate_count = 0
        for plate_entry in list(entry.get("plates") or []):
            if not isinstance(plate_entry, dict):
                continue
            if len(list(plate_entry.get("polygon") or [])) >= 4:
                plate_count += 1
        if plate_count <= 0:
            try:
                plate_count = int(entry.get("plate_count", 0) or 0)
            except Exception:
                plate_count = 0
        if plate_count > 0:
            project_counts_by_name[safe_name] = max(
                int(project_counts_by_name.get(safe_name, 0) or 0),
                int(plate_count),
            )

    input_dir = None
    for raw_dir in (
        manifest.get("source_input_dir") if isinstance(manifest, dict) else "",
        manifest.get("input_dir") if isinstance(manifest, dict) else "",
    ):
        try:
            candidate_dir = Path(str(raw_dir or "").strip())
        except Exception:
            candidate_dir = None
        if candidate_dir is not None and candidate_dir.exists():
            input_dir = candidate_dir
            break
    run_images_dir = resolved_run_dir / "images"
    expected_entry_keys_by_name: dict[str, str] = {}
    for safe_name in approved_plate_names:
        source_image_path = None
        for base_dir in (input_dir, run_images_dir if run_images_dir.exists() else None):
            if base_dir is None:
                continue
            try:
                candidate_path = Path(base_dir) / safe_name
                if candidate_path.exists():
                    source_image_path = candidate_path
                    break
            except Exception:
                continue
        if source_image_path is None:
            continue
        expected_key = _build_campaign_plate_entry_key(
            image_name=safe_name,
            source_image_path=source_image_path,
        )
        if expected_key:
            expected_entry_keys_by_name[safe_name] = expected_key

    def has_project_pool_entry(safe_name: str) -> bool:
        expected_key = str(expected_entry_keys_by_name.get(safe_name, "") or "").strip().lower()
        if expected_key:
            return expected_key in project_entry_keys
        return int(project_counts_by_name.get(safe_name, 0) or 0) > 0

    missing_names = sorted(
        safe_name
        for safe_name in approved_plate_names
        if not has_project_pool_entry(safe_name)
    )
    matched_names = sorted(set(approved_plate_names) - set(missing_names))
    matched_plates = sum(int(run_counts.get(name, 0) or 0) for name in matched_names)
    missing_plates = sum(int(run_counts.get(name, 0) or 0) for name in missing_names)

    return {
        "ok": not bool(missing_names),
        "reason": "already_promoted" if not missing_names else "missing_from_project_pool",
        "run_dir": str(resolved_run_dir),
        "project_name": resolved_project_name,
        "matched_images": len(matched_names),
        "matched_plates": int(matched_plates or 0),
        "missing_images": len(missing_names),
        "missing_plates": int(missing_plates or 0),
        "missing_names": missing_names,
    }


def _promote_run_to_campaign_plate_approved_set(
    self,
    run_dir: Path,
    *,
    force_parse_xml: bool = False,
    project_name: str | None = None,
) -> dict:
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {"ok": False, "reason": "campaign_unavailable"}

    try:
        active_project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        active_project_name = ""
    if self._is_free_mode_session_context() and not active_project_name:
        return {"ok": False, "reason": "free_mode"}

    resolved_project_name = str(project_name or "").strip() or _resolve_campaign_project_name_from_run_dir(run_dir)
    if not resolved_project_name:
        return {"ok": False, "reason": "campaign_inactive"}

    resolved_run_dir = _resolve_campaign_annotation_run_dir(run_dir, resolved_project_name)
    if resolved_run_dir is None:
        return {"ok": False, "reason": "missing_run_dir", "run_dir": str(run_dir or "")}

    entries = self._build_campaign_plate_approved_entries_from_run(
        resolved_run_dir,
        force_parse_xml=force_parse_xml,
    )
    if not entries:
        pool_state = _get_campaign_plate_approved_run_pool_state(
            self,
            resolved_run_dir,
            project_name=resolved_project_name,
        )
        if bool(dict(pool_state or {}).get("ok")):
            return {
                "ok": True,
                "reason": "already_promoted",
                "run_dir": str(resolved_run_dir),
                "matched_images": int(dict(pool_state or {}).get("matched_images", 0) or 0),
                "matched_plates": int(dict(pool_state or {}).get("matched_plates", 0) or 0),
            }
        return {
            "ok": False,
            "reason": "missing_entries",
            "run_dir": str(resolved_run_dir),
            "pool_state": dict(pool_state or {}),
        }

    result = CAMPAIGN.upsert_plate_approved_entries(entries, project_name=resolved_project_name)
    try:
        result = dict(result or {})
        result.setdefault("run_dir", str(resolved_run_dir))
        if str(Path(run_dir).resolve()) != str(resolved_run_dir):
            result["requested_run_dir"] = str(run_dir)
            result["resolved_run_dir"] = str(resolved_run_dir)
    except Exception:
        pass
    if not bool(dict(result or {}).get("ok")):
        pool_state = _get_campaign_plate_approved_run_pool_state(
            self,
            resolved_run_dir,
            project_name=resolved_project_name,
        )
        if bool(dict(pool_state or {}).get("ok")):
            result = {
                "ok": True,
                "reason": "already_promoted",
                "run_dir": str(resolved_run_dir),
                "previous_result": dict(result or {}),
                "matched_images": int(dict(pool_state or {}).get("matched_images", 0) or 0),
                "matched_plates": int(dict(pool_state or {}).get("matched_plates", 0) or 0),
            }
    try:
        self._campaign_hidden_project_approved_runtime_cache = None
    except Exception:
        pass
    return result


def _build_campaign_plate_approved_export_source(self, *args, **kwargs):
    return z2_workflow_methods._build_campaign_plate_approved_export_source(self, *args, **kwargs)


def _build_campaign_plate_approved_preview_bundle(self, *args, **kwargs):
    return z2_workflow_methods._build_campaign_plate_approved_preview_bundle(self, *args, **kwargs)


def _build_path_change_token(path_like) -> str:
    if not path_like:
        return ""

    try:
        path = Path(path_like)
    except Exception:
        return ""

    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)

    try:
        stat = path.stat()
        return (
            f"{resolved}|"
            f"{int(getattr(stat, 'st_mtime_ns', 0) or 0)}|"
            f"{int(getattr(stat, 'st_size', 0) or 0)}"
        )
    except Exception:
        return resolved


def _build_campaign_char_effective_source(self, *args, **kwargs):
    return z2_workflow_methods._build_campaign_char_effective_source(self, *args, **kwargs)


def _resolve_plate_source_run_from_dataset(self, dataset_path: Path | None) -> Path | None:
    if dataset_path is None:
        return None

    try:
        dataset_path = Path(dataset_path)
    except Exception:
        return None

    try:
        if not dataset_path.exists() or not dataset_path.is_dir():
            return None
    except Exception:
        return None

    manifest = self._load_plate_dataset_source_manifest(dataset_path)
    for raw_path in (
        str(manifest.get("source_run_dir") or "").strip(),
        str(manifest.get("source_xml_path") or "").strip(),
    ):
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.suffix.lower() == ".xml":
            candidate = candidate.parent
        if candidate.exists() and candidate.is_dir() and (candidate / "annotations.xml").exists():
            return candidate

    run_name = str(manifest.get("source_run_name") or "").strip()
    if not run_name:
        match = re.match(r"^Plates_Z2_(.+)_\d{8}_\d{6}$", dataset_path.name)
        if match:
            run_name = str(match.group(1) or "").strip()

    if not run_name:
        return None

    search_roots = []
    try:
        from ..campaign_manager import CAMPAIGN

        for root_candidate in (CAMPAIGN.get_dir("auto_ann"), CAMPAIGN.get_staging_dir("auto_ann")):
            if root_candidate is not None:
                search_roots.append(Path(root_candidate))
    except Exception:
        return None

    candidates = []
    for root in search_roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
            for candidate in root.rglob(run_name):
                if (
                    candidate.is_dir()
                    and candidate.name == run_name
                    and (candidate / "annotations.xml").exists()
                ):
                    candidates.append(candidate)
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]
