#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted Z2 workflow/state methods for AnnotationTab.

This module intentionally keeps methods as plain functions receiving ``self``.
The owning class delegates to them, which physically reduces tab_annotation.py
without changing the state model or the public method names used by callbacks.
"""

import copy
import csv
import datetime
import json
import logging
import math
import os
import queue
import re
import shutil
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from ..annotators.runtime_factory import (
    create_combined_plate_annotator,
    create_plate_annotator,
    create_vehicle_annotator,
)
from ..campaign_manager import CAMPAIGN
from ..config import AVAILABLE_DETECT_MODELS, CONFIG, SESSION, YOLO_AVAILABLE, logger
from ..data_models import AnnotationReport, AnnotationStatus, Detection, ImageAnnotation
from ..exporters import CVATExporter, ReportGenerator
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..rectification.polygon_validator import PolygonValidator
from ..training import DatasetCreator
from ..utils import cleanup_gpu_memory, count_images_in_directory, format_duration, get_image_files, get_image_size
from ..validators import format_yolo_model_identity, validate_model_file
from .canvas_progress_overlay import CanvasProgressOverlay
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
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
from .z2_flow_models import Z2CopyPayload, Z2LeftPanelCopyContext
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
from .z2_shared_ui import (
    apply_z2_workflow_cta_ui as dispatch_apply_z2_workflow_cta_ui,
    apply_z2_workflow_left_layout as dispatch_apply_z2_workflow_left_layout,
    build_campaign_gate_focus_state,
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    campaign_gate_id_for_edge,
    campaign_visible_gate_id,
    get_campaign_return_to_graph_copy as dispatch_get_campaign_return_to_graph_copy,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
from .z2_panel_workflow import (
    _refresh_free_mode_workflow_ui,
    get_campaign_step2_view_model,
    _refresh_manual_review_followup_ui,
    apply_theme,
    _finalize_successful_annotation_run_ui,
    _apply_workflow_step_widget_style,
    _render_compact_info_table,
    _refresh_free_mode_manual_right_panel,
    _refresh_manual_plate_stage_ui,
    _show_auto_annotation_success_dialog,
    _apply_z2_left_panel_copy_payload,
    _refresh_workflow_button_styles,
    _refresh_z2_miniflow_progress,
    _apply_main_pane_layout,
)

from .z2_preview_workflow import (
    _build_t06_counter_rows,
    _set_selected_preview_images_approved,
    _rename_selected_preview_image_file,
    _refresh_preview_workspace_visibility,
    _populate_preview_list_async,
    _refresh_preview_list_legend_theme,
    _refresh_preview_list_summary,
    _parse_cvat_preview_annotations,
    _open_preview_metric_filter_modal,
    _get_preview_image_file_metadata,
    _update_preview_edit_status,
    _save_preview_edits,
    _clear_selected_preview_auto_plates,
)
from .z2_campaign_runtime import (
    _build_campaign_char_effective_source,
    _get_campaign_auto_annotation_bootstrap,
    _collect_campaign_auto_annotation_sources,
    _schedule_deferred_campaign_route_cleanup,
    get_campaign_step2_source_state,
    _get_campaign_step3_preview_source_context,
    _build_campaign_plate_approved_entries_from_run,
    reset_campaign_iteration_route_state,
    _sync_campaign_iteration_artifact_registry,
    _build_campaign_plate_approved_preview_bundle,
    _build_campaign_plate_approved_export_source,
    _prepare_approved_step3_source_from_z2_run,
    _promote_campaign_char_repair_ok_to_approved_pool_before_return,
    _reset_campaign_runtime_state,
    _build_campaign_z2_gate_overlay_state,
    _apply_campaign_plate_auto_model_choice,
)
from .z2_restore_workflow import (
    _restore_preview_from_annotation_run,
    _apply_campaign_project_snapshot,
    _apply_annotation_run_restore_payload,
    _ensure_free_mode_input_workspace_preview,
    _restore_preview_from_session_run,
    _apply_free_mode_session_snapshot,
    _prepare_campaign_source_preview_payload,
    _apply_campaign_source_preview_payload,
    _prepare_annotation_run_restore_payload,
)
from .z2_export_workflow import (
    _start_plate_dataset_export,
    _prompt_z2_export_choice,
    _start_plate_annotation_package_export,
    _prompt_plate_annotation_package_export_options,
    _refresh_plate_dataset_export_sources,
)

CHAR_WORK_GATE_DISPLAY_ID = "T05"
NAV_BUTTON_WIDTH = 18
YOLO = None


def _ensure_campaign_graph_context_for_z2(self) -> dict:
    try:
        if self._is_free_mode_session_context():
            return {}
    except Exception:
        return {}

    try:
        context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        context = {}
    if str(context.get("graph_gate_id") or "").strip():
        return context

    try:
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    except Exception:
        active_project = ""
        current_step = 0
        iteration_target = ""

    if not active_project or iteration_target not in {"plate", "char"}:
        return {}

    if iteration_target == "char" and current_step >= 3:
        context = {
            "source": "campaign_graph_inferred_refresh",
            "graph_edge_key": "e3_to_e4",
            "graph_gate_id": "T05",
            "graph_gate_label": "Dataset znaków",
            "graph_transition_title": "Bramka datasetu znaków do treningu",
            "graph_transition_source": "E3",
            "graph_transition_target": "E4Z",
            "graph_path_key": "",
        }
    elif iteration_target == "char":
        context = {
            "source": "campaign_graph_inferred_refresh",
            "graph_edge_key": "e2_to_e3",
            "graph_gate_id": "T03",
            "graph_gate_label": "Przekazanie tablic do pracy nad znakami",
            "graph_transition_title": "Przekazanie tablic do pracy nad znakami",
            "graph_transition_source": "E2",
            "graph_transition_target": "E3",
            "graph_path_key": "char_from_images",
        }
    else:
        context = {
            "source": "campaign_graph_inferred_refresh",
            "graph_edge_key": "e2_to_e4",
            "graph_gate_id": "T04",
            "graph_gate_label": "Trening modelu tablic",
            "graph_transition_title": "Trenuj model tablic",
            "graph_transition_source": "E2",
            "graph_transition_target": "E4T",
            "graph_path_key": "plate_training",
        }

    self._campaign_graph_entry_context = context
    try:
        logger.info(
            "[Z2 GRAPH] refresh inferred gate=%s edge=%s target=%s step=%s",
            str(context.get("graph_gate_id") or "-"),
            str(context.get("graph_edge_key") or "-"),
            iteration_target,
            current_step,
        )
    except Exception:
        pass
    return context


def _build_t06_right_panel_rows(
    *,
    previous_images: int,
    previous_plates: int,
    session_delta_images: int,
    session_delta_plates: int,
    required_plates: int,
) -> list[tuple[str, str, str]]:
    previous_images = max(0, int(previous_images or 0))
    previous_plates = max(0, int(previous_plates or 0))
    session_delta_images = int(session_delta_images or 0)
    session_delta_plates = int(session_delta_plates or 0)
    required_plates = max(0, int(required_plates or 0))
    total_images = max(0, previous_images + session_delta_images)
    total_plates = max(0, previous_plates + session_delta_plates)
    return _build_t06_counter_rows(
        required_plates,
        previous_images,
        previous_plates,
        session_delta_images,
        session_delta_plates,
        total_images,
        total_plates,
    )


def _resolve_t06_project_name_from_run_dir(run_dir: Path | str | None) -> str:
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


def _build_t06_right_panel_rows_from_campaign_state(
    self,
    *,
    required_plates: int,
    run_dir: Path | str | None = None,
    invalidate_cache: bool = False,
    return_snapshot: bool = False,
) -> list[tuple[str, str, str]] | dict | None:
    """Build T06 counters from the persistent campaign source state.

    T06 cannot use only the currently visible Z2 list: after a reload part of
    the approved material may already be represented by the project source or
    by the run manifest, not by the filtered list widget.
    """
    try:
        if invalidate_cache:
            CAMPAIGN.invalidate_step3_char_source_state_cache()
    except Exception:
        pass
    project_counts_by_name: dict[str, int] = {}
    safe_run_dir = None
    try:
        safe_run_dir = Path(run_dir) if run_dir is not None else None
    except Exception:
        safe_run_dir = None
    if safe_run_dir is None:
        try:
            current_run_dir = getattr(self, "current_annotation_run_dir", None)
            safe_run_dir = Path(current_run_dir) if current_run_dir is not None else None
        except Exception:
            safe_run_dir = None

    project_name = _resolve_t06_project_name_from_run_dir(safe_run_dir)
    try:
        approved_entries = list(CAMPAIGN.list_plate_approved_entries(project_name or None) or [])
    except Exception:
        approved_entries = []
    for entry in approved_entries:
        if not isinstance(entry, dict):
            continue
        try:
            safe_name = CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
        except Exception:
            safe_name = str(entry.get("image_name", "") or "").strip().lower()
        if not safe_name:
            continue
        plate_count = 0
        for plate_entry in list(entry.get("plates") or []):
            if not isinstance(plate_entry, dict):
                continue
            if len(list(plate_entry.get("polygon") or [])) >= 4:
                plate_count += 1
        if plate_count <= 0:
            plate_count = int(entry.get("plate_count", 0) or 0)
        if plate_count > 0:
            project_counts_by_name[safe_name] = max(
                int(project_counts_by_name.get(safe_name, 0) or 0),
                int(plate_count),
            )

    run_delta_counts_by_name: dict[str, int] = {}
    run_delta_image_names: set[str] = set()
    if safe_run_dir is not None:
        try:
            approved_names = set(self._load_annotation_run_approved_filenames(safe_run_dir) or set())
        except Exception:
            approved_names = set()
        try:
            current_run_dir = getattr(self, "current_annotation_run_dir", None)
            if current_run_dir is not None and self._paths_equivalent(safe_run_dir, current_run_dir):
                approved_names.update(set(self._get_preview_approved_filenames() or set()))
        except Exception:
            pass
        if approved_names:
            approved_name_lookup = {
                str(name or "").strip().lower()
                for name in set(approved_names or set())
                if str(name or "").strip()
            }
            try:
                for ann in list(getattr(self, "current_annotations", []) or []):
                    safe_name = str(getattr(ann, "filename", "") or "").strip().lower()
                    if not safe_name or safe_name not in approved_name_lookup:
                        continue
                    if safe_name in project_counts_by_name:
                        continue
                    plate_count = len(self._get_plate_detections(ann))
                    if plate_count > 0:
                        run_delta_image_names.add(safe_name)
                        run_delta_counts_by_name[safe_name] = max(
                            int(run_delta_counts_by_name.get(safe_name, 0) or 0),
                            int(plate_count),
                        )
            except Exception:
                pass
            missing_count_names = {
                str(name or "").strip().lower()
                for name in set(approved_names or set())
                if str(name or "").strip()
            }
            missing_count_names = {
                safe_name
                for safe_name in missing_count_names
                if safe_name not in project_counts_by_name
                and safe_name not in run_delta_counts_by_name
            }
            try:
                run_counts = (
                    dict(
                        CAMPAIGN._load_run_plate_counts_by_image(
                            safe_run_dir,
                            image_names=missing_count_names,
                        )
                        or {}
                    )
                    if missing_count_names
                    else {}
                )
            except Exception:
                run_counts = {}
            for image_name, plate_count in run_counts.items():
                safe_name = str(image_name or "").strip().lower()
                if not safe_name or int(plate_count or 0) <= 0:
                    continue
                if safe_name in project_counts_by_name:
                    continue
                run_delta_image_names.add(safe_name)
                run_delta_counts_by_name[safe_name] = max(
                    int(run_delta_counts_by_name.get(safe_name, 0) or 0),
                    int(plate_count),
                )

    project_images = len(project_counts_by_name)
    project_plates = sum(int(count or 0) for count in project_counts_by_name.values())
    run_delta_images = len(run_delta_image_names)
    run_delta_plates = sum(int(count or 0) for count in run_delta_counts_by_name.values())
    if project_images <= 0 and project_plates <= 0 and run_delta_images <= 0 and run_delta_plates <= 0:
        return None
    rows = _build_t06_right_panel_rows(
        previous_images=project_images,
        previous_plates=project_plates,
        session_delta_images=run_delta_images,
        session_delta_plates=run_delta_plates,
        required_plates=required_plates,
    )
    if return_snapshot:
        total_images = int(project_images or 0) + int(run_delta_images or 0)
        total_plates = int(project_plates or 0) + int(run_delta_plates or 0)
        return {
            "rows": rows,
            "project_images": int(project_images or 0),
            "project_plates": int(project_plates or 0),
            "session_delta_images": int(run_delta_images or 0),
            "session_delta_plates": int(run_delta_plates or 0),
            "total_images": int(total_images or 0),
            "total_plates": int(total_plates or 0),
            "ready": bool(int(total_plates or 0) >= int(required_plates or 0)),
        }
    return rows


def _resolve_t06_right_panel_run_dir(self, current_run_dir: str = "") -> Path | None:
    try:
        plate_dataset_var = getattr(self, "plate_dataset_run_var", None)
        plate_dataset_value = str(plate_dataset_var.get() or "").strip() if hasattr(plate_dataset_var, "get") else ""
    except Exception:
        plate_dataset_value = ""
    for raw_candidate in (
        current_run_dir,
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        plate_dataset_value,
    ):
        if not raw_candidate:
            continue
        try:
            candidate = self._resolve_safe_annotation_run_dir(raw_candidate, require_xml=False)
        except TypeError:
            try:
                candidate = self._resolve_safe_annotation_run_dir(raw_candidate)
            except Exception:
                candidate = None
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate
        try:
            path_candidate = Path(raw_candidate)
            if path_candidate.exists():
                return path_candidate
        except Exception:
            continue
    return None


def _build_t06_right_panel_fast_snapshot(
    self,
    *,
    required_plates: int,
    run_dir: Path | None,
) -> dict:
    token = ""
    try:
        token = f"{Path(run_dir).resolve() if run_dir else ''}|T06"
    except Exception:
        token = f"{run_dir or ''}|T06"

    snapshot = _build_t06_right_panel_rows_from_campaign_state(
        self,
        required_plates=required_plates,
        run_dir=run_dir,
        return_snapshot=True,
    )
    if isinstance(snapshot, dict) and snapshot.get("rows"):
        return snapshot

    source_baseline = getattr(self, "_campaign_t06_entry_source_baseline", None)
    approval_baseline = getattr(self, "_campaign_t06_entry_approval_baseline", None)
    previous_images = 0
    previous_plates = 0
    baseline_images = 0
    baseline_plates = 0
    if isinstance(source_baseline, dict) and source_baseline.get("token") == token:
        previous_images = int(source_baseline.get("images", 0) or 0)
        previous_plates = int(source_baseline.get("plates", 0) or 0)
    if isinstance(approval_baseline, dict) and approval_baseline.get("token") == token:
        baseline_images = int(approval_baseline.get("images", 0) or 0)
        baseline_plates = int(approval_baseline.get("plates", 0) or 0)

    current_images = baseline_images
    current_plates = baseline_plates
    try:
        if getattr(self, "current_annotations", None):
            current_images, current_plates = self._get_current_preview_plate_approved_counts()
    except Exception:
        current_images = baseline_images
        current_plates = baseline_plates

    session_delta_images = int(current_images or 0)
    session_delta_plates = int(current_plates or 0)
    total_images = max(0, int(previous_images or 0) + int(session_delta_images or 0))
    total_plates = max(0, int(previous_plates or 0) + int(session_delta_plates or 0))
    rows = _build_t06_right_panel_rows(
        previous_images=previous_images,
        previous_plates=previous_plates,
        session_delta_images=session_delta_images,
        session_delta_plates=session_delta_plates,
        required_plates=required_plates,
    )
    return {
        "rows": rows,
        "project_images": int(previous_images or 0),
        "project_plates": int(previous_plates or 0),
        "session_delta_images": int(session_delta_images or 0),
        "session_delta_plates": int(session_delta_plates or 0),
        "total_images": int(total_images or 0),
        "total_plates": int(total_plates or 0),
        "ready": bool(int(total_plates or 0) >= int(required_plates or 0)),
    }


def _force_render_campaign_graph_t06_right_panel(
    self,
    *,
    graph_context: dict,
    render_signature: tuple,
    current_run_dir: str,
) -> bool:
    started = time.perf_counter()
    phase_started = started
    phases: list[str] = []

    def _mark_phase(name: str) -> None:
        nonlocal phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - phase_started) * 1000.0
            if elapsed_ms >= 80.0:
                phases.append(f"{name}={elapsed_ms:.0f}ms")
            phase_started = now
        except Exception:
            pass

    required_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    run_dir = _resolve_t06_right_panel_run_dir(self, current_run_dir=current_run_dir)
    xml_exists = bool(run_dir and (Path(run_dir) / "annotations.xml").exists())
    _mark_phase("resolve_run")

    snapshot = _build_t06_right_panel_fast_snapshot(
        self,
        required_plates=required_plates,
        run_dir=run_dir,
    )
    rows = list(snapshot.get("rows") or [])
    ready = bool(snapshot.get("ready"))
    total_images = int(snapshot.get("total_images", 0) or 0)
    total_plates = int(snapshot.get("total_plates", 0) or 0)
    missing_plates = max(0, int(required_plates or 0) - int(total_plates or 0))
    try:
        quality_info = CONFIG.describe_yolo_pose_dataset_quality(max(0, total_plates))
    except Exception:
        quality_info = {}
    next_quality_label = str(quality_info.get("next_label", "") or "").strip()
    try:
        missing_next_quality = max(0, int(quality_info.get("missing_next", 0) or 0))
    except Exception:
        missing_next_quality = 0

    focus_state = build_campaign_gate_focus_state(missing_plates, quality_info)
    missing_focus_label = str(focus_state.get("label") or "DO MIN.")
    missing_focus_row_label = str(focus_state.get("row_label") or "Do otwarcia bramki brakuje")
    missing_focus_value = int(focus_state.get("value") or 0)
    missing_focus_text = str(focus_state.get("text") or "")
    missing_focus_tone = str(focus_state.get("tone") or "warning")
    _mark_phase("build_rows")

    gate_label = str(graph_context.get("graph_gate_label") or CHAR_WORK_GATE_DISPLAY_ID).strip()
    intro = (
        f"Pracujesz w Z2 dla bramki {CHAR_WORK_GATE_DISPLAY_ID}: {gate_label}. "
        "Dodajesz albo korygujesz tablice, które mają zasilić dalszą pracę nad znakami. "
        "Pozycje oznaczone jako [OK] czekają na przekazanie; po formalnym powrocie do grafu zasilą źródło Z3 do wyodrębniania tablic."
    )
    tone = "success" if ready else "warning"

    try:
        self._campaign_step2_gate_overlay_state = {
            "visible": True,
            "ready": bool(ready),
            "tone": tone,
            "title": f"BRAMKA {CHAR_WORK_GATE_DISPLAY_ID}",
            "gate_id": CHAR_WORK_GATE_DISPLAY_ID,
            "gate_label": gate_label,
            "status": "OTWARTA" if ready else "W TRAKCIE",
            "approved_images": int(total_images or 0),
            "approved_plates": int(total_plates or 0),
            "required_plates": int(required_plates or 0),
            "missing_plates": int(missing_plates or 0),
            "missing_to_open": int(missing_plates or 0),
            "gate_metric": "plates",
            "missing_focus_label": missing_focus_label,
            "missing_focus_row_label": missing_focus_row_label,
            "missing_focus_value": int(missing_focus_value or 0),
            "missing_focus_text": missing_focus_text,
            "missing_focus_tone": missing_focus_tone,
            "next_quality_label": next_quality_label,
            "missing_next_quality": int(missing_next_quality or 0),
            "xml": "XML: OK" if xml_exists else "XML: brak",
        }
        self._campaign_step2_approval_ready = bool(ready)
        self._campaign_step2_approval_action = "campaign_graph_t06"
        self._campaign_step2_approval_iteration_target = "char"
        self._campaign_step2_approval_repair_mode = True
        self._campaign_step2_approval_hint_text = intro
        self._campaign_step2_approval_hint_tone = tone
    except Exception:
        pass
    _mark_phase("state")

    try:
        self.approve_btn_row.configure(text=f" Status bramki {CHAR_WORK_GATE_DISPLAY_ID} ")
        self.approve_context_var.set(intro)
        self.approve_hint_title_var.set(f"Warunek bramki {CHAR_WORK_GATE_DISPLAY_ID}")
        self.approve_gate_hint_var.set("")
        self.approve_breakdown_title_var.set("")
        self.approve_breakdown_var.set("")
        self._set_approve_context_box_state("info")
        self._set_approve_hint_box_state(tone)
    except Exception:
        pass
    _mark_phase("vars")

    try:
        self._render_compact_info_table(
            getattr(self, "approve_hint_table_frame", None),
            rows,
            default_value_tone=tone,
            reuse_existing=True,
            show_header=False,
        )
    except Exception:
        pass
    _mark_phase("table")

    processing = bool(getattr(self, "is_processing", False)) or bool(getattr(self, "_annotation_stop_requested", False))
    try:
        self._set_widget_packed(getattr(self, "approve_btn_row", None), True, fill=tk.X, pady=(8, 0))
        self._set_widget_packed(getattr(self, "approve_context_box", None), True, fill=tk.X, pady=(0, 10))
        self._set_widget_packed(getattr(self, "approve_hint_box", None), True, fill=tk.X, pady=(0, 10))
        self._set_widget_packed(getattr(self, "approve_hint_title_lbl", None), True, fill=tk.X)
        self._set_widget_packed(getattr(self, "approve_hint_table_frame", None), True, fill=tk.X, pady=(6, 8))
        self._set_widget_packed(getattr(self, "approve_gate_hint_lbl", None), False)
        self._set_widget_packed(getattr(self, "approve_breakdown_box", None), False)
        self._set_widget_packed(getattr(self, "approve_btn_frame", None), True, fill=tk.X)
        self._set_widget_packed(getattr(self, "approve_btn", None), False)
        try:
            getattr(self, "return_to_campaign_right_btn", None).configure(
                state=(tk.DISABLED if processing else tk.NORMAL)
            )
        except Exception:
            pass
        self._set_widget_packed(
            getattr(self, "return_to_campaign_right_btn", None),
            not processing,
            fill=tk.X,
            pady=(0, 0),
        )
        self._normalize_approve_panel_order()
    except Exception:
        pass
    _mark_phase("layout")

    try:
        self._annotation_right_panel_visible = True
        self._sync_main_pane_right_panel_visibility()
    except Exception:
        pass
    try:
        self.frame.after_idle(self._sync_right_panel_scrollregion)
    except Exception:
        try:
            self._sync_right_panel_scrollregion()
        except Exception:
            pass
    _mark_phase("pane")

    try:
        self._z2_graph_right_panel_render_signature = render_signature
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "[Z2 RIGHT] forced gate=T06 rows=%s ready=%s xml=%s total=%s/%s fast=1 elapsed=%.0fms phases=[%s]",
            len(rows),
            int(bool(ready)),
            int(bool(xml_exists)),
            total_images,
            total_plates,
            elapsed_ms,
            ", ".join(phases) if phases else "ok",
        )
    except Exception:
        pass
    return True


def _force_render_campaign_graph_right_panel(self) -> bool:
    graph_context = _ensure_campaign_graph_context_for_z2(self)
    gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    if not gate_id:
        return False
    repair_origin_edge_key = str(
        graph_context.get("repair_origin_edge_key")
        or graph_context.get("source_graph_edge_key")
        or ""
    ).strip()
    repair_origin_gate_id = str(
        graph_context.get("repair_origin_gate_id")
        or graph_context.get("source_graph_gate_id")
        or ""
    ).strip().upper()
    repair_origin_gate_id = campaign_gate_id_for_edge(repair_origin_edge_key, repair_origin_gate_id)
    repair_from_t07 = bool(gate_id == "T04" and repair_origin_gate_id == "T06")

    current_run_dir = str(getattr(self, "current_annotation_run_dir", "") or "").strip()
    render_signature = (
        gate_id,
        repair_origin_gate_id,
        current_run_dir,
        int(len(getattr(self, "current_annotations", []) or [])),
        int(len(getattr(self, "_preview_approved_filenames", set()) or set())),
        int(len(getattr(self, "_campaign_pending_approved_filenames", set()) or set())),
        int(len(getattr(self, "_preview_dirty_images", set()) or set())),
        int(bool(getattr(self, "is_processing", False))),
        int(bool(getattr(self, "_annotation_stop_requested", False))),
        int(bool(getattr(self, "_campaign_deferred_run_restore_payload_applied", False))),
    )
    try:
        already_visible = bool(
            str(getattr(self, "approve_context_box", None).winfo_manager())
            and str(getattr(self, "approve_hint_table_frame", None).winfo_manager())
        )
    except Exception:
        already_visible = False
    if (
        already_visible
        and getattr(self, "_z2_graph_right_panel_render_signature", None) == render_signature
    ):
        return True

    if gate_id == "T05":
        return _force_render_campaign_graph_t06_right_panel(
            self,
            graph_context=graph_context,
            render_signature=render_signature,
            current_run_dir=current_run_dir,
        )

    try:
        gate_state = dict(self._build_campaign_z2_gate_overlay_state() or {})
    except Exception:
        gate_state = {}

    try:
        approval_context = dict(self._get_campaign_step2_approval_context() or {})
    except Exception:
        approval_context = {}
    run_dir = approval_context.get("run_dir")
    xml_exists = bool(run_dir and (Path(run_dir) / "annotations.xml").exists())

    total_images = 0
    total_plates = 0
    approved_images = 0
    approved_plates = 0
    if getattr(self, "current_annotations", None):
        try:
            count_state = dict(self._get_current_preview_plate_count_state() or {})
            total_images = int(count_state.get("images_with_plates", 0) or 0)
            total_plates = int(count_state.get("total_plates", 0) or 0)
        except Exception:
            total_images, total_plates = 0, 0
        try:
            approved_images = int(count_state.get("approved_images", 0) or 0)
            approved_plates = int(count_state.get("approved_plates", 0) or 0)
        except Exception:
            approved_images, approved_plates = 0, 0
    else:
        try:
            total_images, total_plates = self._get_run_plate_annotation_counts(run_dir)
        except Exception:
            total_images, total_plates = 0, 0
        try:
            approved_images, approved_plates = self._get_run_plate_approved_counts(run_dir)
        except Exception:
            approved_images, approved_plates = 0, 0

    try:
        required_plates = int(gate_state.get("required_plates", 0) or 0)
    except Exception:
        required_plates = 0
    if required_plates <= 0:
        required_plates = (
            int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
            if gate_id in {"T02", "T03", "T05"}
            else int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        )
    try:
        effective_approved_plates = int(gate_state.get("approved_plates", approved_plates) or 0)
    except Exception:
        effective_approved_plates = int(approved_plates or 0)
    try:
        effective_approved_images = int(gate_state.get("approved_images", approved_images) or 0)
    except Exception:
        effective_approved_images = int(approved_images or 0)
    t02_project_images = 0
    t02_project_plates = 0
    t02_session_images = int(approved_images or 0)
    t02_session_plates = int(approved_plates or 0)
    if gate_id == "T02":
        try:
            project_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            t02_project_images = int(project_stats.get("images", 0) or 0)
            t02_project_plates = int(project_stats.get("plates", 0) or 0)
        except Exception:
            t02_project_images = 0
            t02_project_plates = 0
        effective_approved_images = max(0, int(t02_project_images or 0) + int(t02_session_images or 0))
        effective_approved_plates = max(0, int(t02_project_plates or 0) + int(t02_session_plates or 0))
        gate_state = {
            **dict(gate_state or {}),
            "approved_images": int(effective_approved_images or 0),
            "approved_plates": int(effective_approved_plates or 0),
        }
    missing_plates = max(0, int(required_plates) - int(effective_approved_plates))
    ready = bool(gate_state.get("ready", missing_plates <= 0 and xml_exists))
    if gate_id == "T02":
        ready = bool(missing_plates <= 0)
    tone = "success" if ready else "warning"
    status_text = "OTWARTA" if ready else ("DO KONTROLI" if gate_id == "T02" else "W TRAKCIE")
    missing_focus_row_label = str(gate_state.get("missing_focus_row_label") or "").strip()
    missing_focus_text = str(gate_state.get("missing_focus_text") or "").strip()
    missing_focus_tone = str(gate_state.get("missing_focus_tone") or "").strip().lower()
    if not missing_focus_row_label or not missing_focus_text:
        try:
            fallback_quality_info = CONFIG.describe_yolo_pose_dataset_quality(int(effective_approved_plates or 0))
        except Exception:
            fallback_quality_info = {}
        focus_state = build_campaign_gate_focus_state(missing_plates, fallback_quality_info)
        missing_focus_row_label = str(focus_state.get("row_label") or "Do otwarcia bramki brakuje")
        missing_focus_text = str(focus_state.get("text") or "")
        missing_focus_tone = str(focus_state.get("tone") or "warning")
    elif missing_plates <= 0 and "otwarcia bramki" in missing_focus_row_label.lower():
        try:
            fallback_quality_info = CONFIG.describe_yolo_pose_dataset_quality(int(effective_approved_plates or 0))
        except Exception:
            fallback_quality_info = {}
        focus_state = build_campaign_gate_focus_state(0, fallback_quality_info)
        missing_focus_row_label = str(focus_state.get("row_label") or "Minimum bramki")
        missing_focus_text = str(focus_state.get("text") or "Spełnione")
        missing_focus_tone = str(focus_state.get("tone") or "success")

    gate_label = str(
        graph_context.get("graph_gate_label")
        or gate_state.get("gate_label")
        or gate_id
    ).strip()
    display_gate_id = campaign_visible_gate_id(gate_id) or gate_id
    if repair_from_t07:
        display_gate_title = "Naprawa T06"
        display_row_label = "Status naprawy"
        intro = (
            "Pracujesz w Z2 w trybie naprawczym bramki T06. "
            "Dodajesz albo poprawiasz ramki tablic; po powrocie decyzję podejmujesz na T07."
        )
    else:
        display_gate_title = f"Bramka {display_gate_id}"
        display_row_label = "Status bramki"
        intro = (
            f"Pracujesz w Z2 dla bramki {display_gate_id}: {gate_label}. "
            "Prawy panel pokazuje stan aktywnego pliku anotacji i pozycji oznaczonych jako [OK]."
        )
        if gate_id == "T02":
            intro = (
                "Kontrolujesz AT dla bramki T02. Panel pokazuje, ile materiału jest już w puli projektu "
                "oraz ile nowych pozycji [OK] dojdzie po zapisaniu tej kontroli."
            )
    if repair_from_t07:
        intro = intro.replace("T07", "T06")

    if gate_id == "T05":
        intro += " Ta ścieżka pozwala dodać lub poprawić tablice przed dalszą pracą nad znakami."

    if gate_id == "T05":
        rows = _build_t06_right_panel_rows_from_campaign_state(
            self,
            required_plates=required_plates,
            run_dir=run_dir,
        )
        if rows is None:
            rows = _build_t06_right_panel_rows(
                previous_images=effective_approved_images,
                previous_plates=effective_approved_plates,
                session_delta_images=0,
                session_delta_plates=0,
                required_plates=required_plates,
            )
    elif gate_id == "T02":
        if ready and t02_session_plates <= 0 and t02_project_plates >= required_plates:
            t02_focus_label = "Spełnienie T02"
            t02_focus_text = "Spełnione przez wcześniejszą pulę projektu"
            t02_focus_tone = "success"
        elif ready:
            t02_focus_label = "Po zapisie kontroli"
            t02_focus_text = "T02 będzie gotowa do zatwierdzenia"
            t02_focus_tone = "success"
        else:
            t02_focus_label = "Do wejścia do E3 brakuje"
            t02_focus_text = f"{missing_plates} tablic zatwierdzonych [OK]"
            t02_focus_tone = "warning"
        rows = [
            (display_row_label, status_text, tone),
            (
                "AT do kontroli",
                f"{total_images} obrazów / {total_plates} tablic",
                "success" if total_plates > 0 else "warning",
            ),
            (
                "Pula projektu [OK]",
                f"{t02_project_images} obrazów / {t02_project_plates} tablic",
                "success" if t02_project_plates > 0 else "muted",
            ),
            (
                "Nowe [OK] w kontroli",
                f"+{t02_session_images} obrazów / +{t02_session_plates} tablic",
                "success" if t02_session_plates > 0 else "muted",
            ),
            (
                "Razem po zapisie",
                f"{effective_approved_images} obrazów / {effective_approved_plates} tablic",
                "success" if effective_approved_plates > 0 else "warning",
            ),
            (
                t02_focus_label,
                t02_focus_text,
                t02_focus_tone,
            ),
        ]
    else:
        rows = [
            (display_row_label, status_text, tone),
            (
                "Zatwierdzone [OK]",
                f"{effective_approved_images} obrazów / {effective_approved_plates} tablic",
                "success" if effective_approved_plates > 0 else "warning",
            ),
            (
                missing_focus_row_label,
                missing_focus_text,
                missing_focus_tone or ("success" if missing_plates <= 0 else "warning"),
            ),
            (
                "XML anotacji",
                "XML: OK" if xml_exists else f"XML: aktywny run ({total_images} obrazów / {total_plates} tablic)",
                "success" if xml_exists else "warning",
            ),
        ]

    try:
        self.approve_btn_row.configure(text=f" {display_gate_title} ")
    except Exception:
        pass
    try:
        self.approve_context_var.set(intro)
        self.approve_hint_title_var.set(display_gate_title)
        self.approve_gate_hint_var.set("")
        self.approve_breakdown_title_var.set("")
        self.approve_breakdown_var.set("")
    except Exception:
        pass

    try:
        self._set_approve_context_box_state("info")
        self._set_approve_hint_box_state(tone)
    except Exception:
        pass
    try:
        self._render_compact_info_table(
            getattr(self, "approve_hint_table_frame", None),
            rows,
            default_value_tone=tone,
            reuse_existing=True,
            show_header=False,
        )
    except Exception:
        pass

    processing = bool(getattr(self, "is_processing", False)) or bool(getattr(self, "_annotation_stop_requested", False))
    try:
        self._set_widget_packed(getattr(self, "approve_btn_row", None), True, fill=tk.X, pady=(8, 0))
        self._set_widget_packed(getattr(self, "approve_context_box", None), True, fill=tk.X, pady=(0, 10))
        self._set_widget_packed(getattr(self, "approve_hint_box", None), True, fill=tk.X, pady=(0, 10))
        self._set_widget_packed(getattr(self, "approve_hint_title_lbl", None), True, fill=tk.X)
        self._set_widget_packed(getattr(self, "approve_hint_table_frame", None), True, fill=tk.X, pady=(6, 8))
        self._set_widget_packed(getattr(self, "approve_gate_hint_lbl", None), False)
        self._set_widget_packed(getattr(self, "approve_breakdown_box", None), False)
        self._set_widget_packed(getattr(self, "approve_btn_frame", None), True, fill=tk.X)
        self._set_widget_packed(getattr(self, "approve_btn", None), False)
        try:
            return_copy = dispatch_get_campaign_return_to_graph_copy(self)
            getattr(self, "return_to_campaign_right_btn", None).configure(
                text=str(return_copy.get("button") or "Wróć do grafu"),
                state=(tk.DISABLED if processing else tk.NORMAL),
                width=int(return_copy.get("width", 18) or 18),
            )
        except Exception:
            pass
        self._set_widget_packed(
            getattr(self, "return_to_campaign_right_btn", None),
            not processing,
            fill=tk.X,
            pady=(0, 0),
        )
        self._normalize_approve_panel_order()
        self._sync_right_panel_scrollregion()
    except Exception:
        pass
    try:
        self._annotation_right_panel_visible = True
        self._sync_main_pane_right_panel_visibility()
    except Exception:
        pass

    try:
        self._z2_graph_right_panel_render_signature = render_signature
        logger.info(
            "[Z2 RIGHT] forced gate=%s rows=%s ready=%s xml=%s ok=%s/%s total=%s/%s context=%s",
            gate_id,
            len(rows),
            int(bool(ready)),
            int(bool(xml_exists)),
            effective_approved_images,
            effective_approved_plates,
            total_images,
            total_plates,
            str(graph_context.get("source") or "-"),
        )
    except Exception:
        pass
    return True


def _refresh_step2_action_states(self, *, lightweight: bool = False):
    if getattr(self, "_free_mode_session_restore_in_progress", False) and self._is_free_mode_session_context():
        return
    _ensure_campaign_graph_context_for_z2(self)
    self._campaign_step2_approval_ready = False
    self._campaign_step2_approval_action = ""
    self._campaign_step2_approval_iteration_target = ""
    self._campaign_step2_approval_repair_mode = False
    self._campaign_step2_approval_hint_text = ""
    self._campaign_step2_approval_hint_tone = "muted"
    self._campaign_step2_gate_overlay_state = {}
    if bool(getattr(self, "_campaign_step2_transition_in_progress", False)):
        self._campaign_step2_transition_refresh_pending = True
        try:
            self._hide_preview_campaign_gate_overlay()
        except Exception:
            pass
        try:
            current_hint = str(self.approve_gate_hint_var.get() or "")
            if current_hint.startswith("Wczyt"):
                self.approve_gate_hint_var.set("")
                self._set_widget_packed(getattr(self, "approve_gate_hint_lbl", None), False)
        except Exception:
            pass
        return
    deferred_run_restore = bool(getattr(self, "_campaign_deferred_run_restore_in_progress", False))
    if deferred_run_restore and bool(getattr(self, "_campaign_deferred_run_restore_payload_applied", False)):
        self._campaign_deferred_run_restore_in_progress = False
        deferred_run_restore = False
    if deferred_run_restore:
        try:
            target_dir = str(getattr(self, "_campaign_deferred_run_restore_target_dir", "") or "").strip()
            current_run_dir = str(getattr(self, "current_annotation_run_dir", "") or "").strip()
            restored_count = len(getattr(self, "current_annotations", []) or [])
            if (
                target_dir
                and current_run_dir
                and restored_count > 0
                and Path(current_run_dir).resolve() == Path(target_dir).resolve()
            ):
                self._campaign_deferred_run_restore_payload_applied = True
                self._campaign_deferred_run_restore_in_progress = False
                deferred_run_restore = False
        except Exception:
            pass
    if False and deferred_run_restore:
        return_copy = dispatch_get_campaign_return_to_graph_copy(self)
        try:
            self._hide_preview_campaign_gate_overlay()
        except Exception:
            pass
        try:
            self.approve_btn_row.configure(text=str(return_copy.get("section") or " Powrót do grafu "))
        except Exception:
            pass
        try:
            self.approve_btn.config(state=tk.DISABLED)
        except Exception:
            pass
        try:
            self.return_to_campaign_right_btn.configure(
                text=str(return_copy.get("button") or "Wróć do grafu"),
                state=(tk.DISABLED if self.is_processing else tk.NORMAL),
                width=int(return_copy.get("width", 18) or 18),
            )
            if not lightweight:
                if str(self.approve_btn.winfo_manager()) == "pack":
                    self.approve_btn.pack_forget()
                if str(self.return_to_campaign_right_btn.winfo_manager()) != "pack":
                    self.return_to_campaign_right_btn.pack(fill=tk.X, pady=(0, 0))
        except Exception:
            pass
        try:
            self.approve_gate_hint_var.set("")
        except Exception:
            pass
        try:
            self._set_approve_hint_box_state("info")
        except Exception:
            pass
        try:
            self._set_widget_packed(getattr(self, "approve_hint_title_lbl", None), False)
            self._set_widget_packed(getattr(self, "approve_hint_table_frame", None), False)
            self._set_widget_packed(
                getattr(self, "approve_gate_hint_lbl", None),
                bool(self._should_show_right_panel()),
                fill=tk.X,
                pady=(4, 8),
            )
            self._set_widget_packed(
                getattr(self, "approve_hint_box", None),
                bool(self._should_show_right_panel()),
                fill=tk.X,
                pady=(0, 10),
                before=getattr(self, "approve_breakdown_box", None),
            )
        except Exception:
            pass
        return

    if deferred_run_restore and not lightweight:
        try:
            current_hint = str(self.approve_gate_hint_var.get() or "")
            if current_hint.startswith("Wczyt"):
                self.approve_gate_hint_var.set("")
                self._set_widget_packed(getattr(self, "approve_gate_hint_lbl", None), False)
        except Exception:
            pass

    if not self._is_free_mode_session_context():
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
            graph_gate_id = campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
        except Exception:
            graph_gate_id = ""
        if not lightweight and graph_gate_id in {"T02", "T03", "T04", "T05"}:
            try:
                if _force_render_campaign_graph_right_panel(self):
                    return
            except Exception as exc:
                logger.debug(f"Nie udało się wyrenderować prawego panelu bramki grafu Z2: {exc}")

    if self._is_free_mode_session_context():
        dataset_run_dir = self._get_z2_free_status_panel_run_dir(require_xml=False)
    else:
        dataset_run_dir = self._resolve_safe_annotation_run_dir(self.plate_dataset_run_var.get())
        if dataset_run_dir is None:
            dataset_run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None))
        if dataset_run_dir is None:
            dataset_run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "last_staging_run_dir", None))

    dataset_images_dir = self._resolve_existing_dir(self.plate_dataset_images_var.get())
    if dataset_images_dir is None:
        dataset_images_dir = self._resolve_existing_dir(self.input_dir_var.get())

    dataset_xml_exists = bool(dataset_run_dir and (dataset_run_dir / "annotations.xml").exists())
    _dataset_images_with_plates, dataset_total_plates = self._get_run_plate_annotation_counts(dataset_run_dir)
    dataset_approval_state = self._get_run_plate_strict_approved_state(dataset_run_dir)
    dataset_approved_images = int(dataset_approval_state.get("approved_images", 0) or 0)
    dataset_approved_plates = int(dataset_approval_state.get("approved_plates", 0) or 0)
    manual_route = bool(self._manual_xml_template_enabled())
    dataset_ready = bool(
        dataset_xml_exists
        and dataset_images_dir is not None
        and dataset_total_plates > 0
        and dataset_approved_images > 0
        and dataset_approved_plates > 0
    )
    if self.is_processing:
        dataset_ready = False

    try:
        if self._is_free_mode_session_context():
            annotation_package_state = (
                self._get_plate_annotation_package_export_state(run_dir=dataset_run_dir)
                if dataset_run_dir is not None
                else {"ok": False}
            )
            export_choice_ready = bool(
                (dataset_ready or annotation_package_state.get("ok"))
                and not self.is_processing
            )
            self.export_plate_dataset_btn.configure(
                text="EKSPORT",
                command=self._start_z2_export_choice_flow,
                state=(tk.NORMAL if export_choice_ready else tk.DISABLED),
            )
        else:
            self.export_plate_dataset_btn.configure(
                text="EKSPORTUJ DATASET",
                command=self._start_z2_export_choice_flow,
                state=(tk.NORMAL if dataset_ready else tk.DISABLED),
            )
    except Exception:
        pass
    self._set_plate_annotation_export_button_state()

    if dataset_xml_exists and dataset_images_dir is not None and dataset_total_plates <= 0:
        if self._is_free_mode_session_context():
            no_plate_message = (
                "Eksport Z2 jest jeszcze niedostępny: XML nie zawiera zapisanej tablicy.\n\n"
                "Eksport anotacji XML wymaga co najmniej jednej zapisanej tablicy. "
                "Dataset YOLO dodatkowo wymaga bramki pozycji [OK]."
            )
        else:
            no_plate_message = (
                "Eksport zablokowany: w runie nie ma jeszcze zapisanej tablicy.\n\n"
                "Jak odblokować: narysuj i zapisz co najmniej jedną ramkę tablicy, potem zaznacz obraz na liście, "
                "kliknij PPM i wybierz „Oznacz zaznaczone jako OK”. Bez statusu OK eksport nie będzie możliwy."
                if manual_route
                else "Eksport zablokowany: w runie nie ma jeszcze zapisanej tablicy.\n\n"
                "Jak odblokować: popraw wynik albo dodaj co najmniej jedną tablicę, zapisz zmiany, potem zaznacz obraz na liście, "
                "kliknij PPM i wybierz „Oznacz zaznaczone jako OK”. Bez statusu OK eksport nie będzie możliwy."
            )
        self._set_plate_export_status(
            no_plate_message,
            "warning",
        )
    elif dataset_xml_exists and dataset_images_dir is not None and dataset_total_plates > 0 and dataset_approved_plates <= 0:
        if self._is_free_mode_session_context():
            no_ok_message = (
                "Eksport anotacji XML jest dostępny, bo XML zawiera zapisane tablice.\n\n"
                "Dataset YOLO Pose ze splitem jest zablokowany: wymaga zatwierdzonych pozycji [OK]. "
                "Zaznacz poprawne obrazy na liście, kliknij PPM i wybierz „Oznacz zaznaczone jako OK”.\n\n"
                f"OK: {dataset_approved_images} obrazów / {dataset_approved_plates} tablic; "
                f"w runie: {dataset_total_plates} tablic."
            )
        else:
            no_ok_message = (
                "Eksport zablokowany: żadna anotacja nie ma statusu OK.\n\n"
                "Jak odblokować: zaznacz poprawne obrazy na liście, kliknij PPM i wybierz "
                "„Oznacz zaznaczone jako OK”. Eksport obejmie wyłącznie pozycje zatwierdzone OK.\n\n"
                f"OK: {dataset_approved_images} obrazów / {dataset_approved_plates} tablic; "
                f"w runie: {dataset_total_plates} tablic."
            )
        self._set_plate_export_status(
            no_ok_message,
            "warning",
        )

    if self._is_free_mode_session_context():
        try:
            self._sync_main_pane_right_panel_visibility()
            self._refresh_free_mode_manual_right_panel()
        except Exception:
            pass
        return

    approval_context = self._get_campaign_step2_approval_context()
    approval_run_dir = approval_context.get("run_dir")
    project_active = bool(approval_context.get("project_active"))
    current_step = int(approval_context.get("current_step") or 0)
    approval_iteration_target = str(approval_context.get("iteration_target") or "").strip().lower()
    repair_mode = bool(approval_context.get("repair_mode"))
    campaign_pending_summary = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
    current_manual_images = len(self._collect_preview_current_iteration_manual_filenames())
    if campaign_pending_summary or current_manual_images > 0:
        campaign_pending_summary["current_manual_count"] = int(current_manual_images)
        self._campaign_pending_batch_summary = campaign_pending_summary

    graph_context_for_sync = {}
    try:
        graph_context_for_sync = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context_for_sync = {}
    graph_gate_for_sync = campaign_gate_id_for_edge(
        graph_context_for_sync.get("graph_edge_key"),
        graph_context_for_sync.get("graph_gate_id"),
    )

    approval_xml_exists = bool(approval_run_dir and (approval_run_dir / "annotations.xml").exists())
    approval_images_with_plates, approval_total_plates = self._get_run_plate_annotation_counts(approval_run_dir)
    approval_marked_images, approval_marked_plates = self._get_run_plate_approved_counts(approval_run_dir)
    min_plate_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    min_char_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    project_approved_images = 0
    project_approved_plates = 0
    char_effective_images = 0
    char_effective_plates = 0
    char_effective_ready = False
    char_source_state: dict = {}
    campaign_manager_obj = None
    if project_active and approval_iteration_target in {"plate", "char"}:
        try:
            from ..campaign_manager import CAMPAIGN
            campaign_manager_obj = CAMPAIGN
            approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            project_approved_images = int(approved_stats.get("images", 0) or 0)
            project_approved_plates = int(approved_stats.get("plates", 0) or 0)
        except Exception:
            project_approved_images = 0
            project_approved_plates = 0
    if project_active and approval_iteration_target == "char":
        refreshed_char_effective_source = {}
        try:
            if campaign_manager_obj is not None:
                char_source_state = dict(campaign_manager_obj.get_step3_char_source_state() or {})
        except Exception:
            char_source_state = {}
        try:
            if not char_source_state:
                char_source_state = dict(self.get_campaign_step2_source_state(iteration_target="char") or {})
        except Exception:
            char_source_state = {}
        if not char_source_state and not lightweight:
            try:
                refreshed_char_effective_source = dict(self._build_campaign_char_effective_source() or {})
            except Exception:
                refreshed_char_effective_source = {}
        if refreshed_char_effective_source:
            char_source_state.update(
                {
                    "source_scope": "campaign_char_effective_source",
                    "run_dir": str(refreshed_char_effective_source.get("run_dir") or ""),
                    "xml_path": str(refreshed_char_effective_source.get("xml_path") or ""),
                    "images_dir": str(refreshed_char_effective_source.get("images_dir") or ""),
                    "images_with_plates": int(refreshed_char_effective_source.get("images_with_plates", 0) or 0),
                    "total_plates": int(refreshed_char_effective_source.get("total_plates", 0) or 0),
                    "ready": bool(
                        int(refreshed_char_effective_source.get("total_plates", 0) or 0) >= int(min_char_approval_plates)
                    ),
                    "has_source": bool(int(refreshed_char_effective_source.get("total_plates", 0) or 0) > 0),
                }
            )
        char_effective_images = int(char_source_state.get("images_with_plates", 0) or 0)
        char_effective_plates = int(char_source_state.get("total_plates", 0) or 0)
        char_effective_ready = bool(char_source_state.get("ready"))
        project_approved_images = int(char_source_state.get("project_images_with_plates", project_approved_images) or 0)
        project_approved_plates = int(char_source_state.get("project_total_plates", project_approved_plates) or 0)
    char_effective_pending_images = int(
        char_source_state.get("pending_images_with_plates", 0) or 0
    )
    char_effective_pending_plates = int(
        char_source_state.get("pending_total_plates", 0) or 0
    )
    char_effective_current_images = int(
        char_source_state.get(
            "current_images_with_plates",
            max(0, int(char_effective_images or 0) - int(project_approved_images or 0)),
        )
        or 0
    )
    char_effective_current_plates = int(
        char_source_state.get(
            "current_total_plates",
            max(0, int(char_effective_plates or 0) - int(project_approved_plates or 0)),
        )
        or 0
    )
    hidden_t06_approved = bool(
        int(getattr(self, "_campaign_hidden_project_approved_count", 0) or 0) > 0
        or int(getattr(self, "_campaign_hidden_char_effective_count", 0) or 0) > 0
    )
    project_approved_base_images = int(project_approved_images or 0)
    project_approved_base_plates = int(project_approved_plates or 0)
    if (
        approval_iteration_target == "char"
        and (current_step >= 3 or repair_mode or graph_gate_for_sync == "T05")
        and int(char_effective_plates or 0) > int(project_approved_plates or 0)
        and int(char_effective_images or 0) >= int(project_approved_images or 0)
        and hidden_t06_approved
    ):
        project_approved_images = int(char_effective_images or 0)
        project_approved_plates = int(char_effective_plates or 0)
        char_effective_current_images = 0
        char_effective_current_plates = 0
    char_display_images = int(char_effective_images or 0)
    char_display_plates = int(char_effective_plates or 0)
    char_display_current_images = int(char_effective_pending_images or 0)
    char_display_current_plates = int(char_effective_pending_plates or 0)
    effective_project_plate_images = int(project_approved_images or 0) + int(approval_marked_images or 0)
    effective_project_plate_plates = int(project_approved_plates or 0) + int(approval_marked_plates or 0)
    if project_active and approval_iteration_target == "plate" and current_step == 2 and not repair_mode:
        if self.current_annotations:
            approval_images_with_plates, approval_total_plates = self._count_plate_annotations(self.current_annotations)
            approval_marked_images, approval_marked_plates = self._get_current_preview_plate_approved_counts()
            effective_project_plate_images = int(project_approved_images or 0) + int(approval_marked_images or 0)
            effective_project_plate_plates = int(project_approved_plates or 0) + int(approval_marked_plates or 0)
        elif approval_run_dir is None:
            approval_images_with_plates = 0
            approval_total_plates = 0
            approval_marked_images = 0
            approval_marked_plates = 0
            effective_project_plate_images = int(project_approved_images or 0)
            effective_project_plate_plates = int(project_approved_plates or 0)
    if project_active and approval_iteration_target == "char" and current_step == 2 and self.current_annotations:
        try:
            approval_images_with_plates, approval_total_plates = self._count_plate_annotations(self.current_annotations)
            approval_marked_images, approval_marked_plates = self._get_current_preview_plate_approved_counts()
        except Exception:
            pass
        char_display_current_images = int(approval_marked_images or 0)
        char_display_current_plates = int(approval_marked_plates or 0)
        char_display_images = int(project_approved_images or 0) + int(char_display_current_images or 0)
        char_display_plates = int(project_approved_plates or 0) + int(char_display_current_plates or 0)
    char_effective_summary_line = ""
    if project_active and approval_iteration_target == "char":
        char_effective_summary_line = (
            f"Aktywne źródło tablic dla Z3 obejmuje teraz {int(char_display_images or 0)} zatwierdzonych obrazów "
            f"i {int(char_display_plates or 0)} tablic."
        )
    current_iteration_plate_ready = bool(
        int(approval_marked_plates or 0) >= int(min_plate_approval_plates)
    )
    cumulative_project_plate_ready = bool(
        int(effective_project_plate_plates or 0) >= int(min_plate_approval_plates)
    )
    char_current_ready = bool(
        approval_xml_exists
        and int(approval_marked_plates or 0) >= int(min_char_approval_plates)
    )
    approve_ready = bool(
        (
            project_active
            and (current_step == 2 or repair_mode)
            and approval_iteration_target == "plate"
            and cumulative_project_plate_ready
            and not self.is_processing
        )
        or (
            project_active
            and (current_step == 2 or repair_mode)
            and approval_iteration_target == "char"
            and (
                char_current_ready
                or (
                    char_effective_ready
                    and int(char_effective_plates or 0) >= int(min_char_approval_plates)
                )
            )
            and not self.is_processing
        )
    )
    approval_action = ""
    if approve_ready:
        if (
            approval_iteration_target == "char"
            and char_effective_ready
            and not char_current_ready
        ):
            approval_action = "continue_characters"
        else:
            approval_action = "approve_stage"

    try:
        return_copy = dispatch_get_campaign_return_to_graph_copy(self)
        self.approve_btn_row.configure(text=str(return_copy.get("section") or " Powrót do grafu "))
        self.approve_btn.configure(
            text=str(return_copy.get("button") or "Wróć do grafu"),
            command=self._return_to_campaign_wizard,
            state=(tk.DISABLED if self.is_processing else tk.NORMAL),
            width=int(return_copy.get("width", 18) or 18),
        )
        self.return_to_campaign_right_btn.configure(
            text=str(return_copy.get("button") or "Wróć do grafu"),
            state=(tk.DISABLED if self.is_processing else tk.NORMAL),
            width=int(return_copy.get("width", 18) or 18),
        )
        if not lightweight:
            if str(self.approve_btn.winfo_manager()) == "pack":
                self.approve_btn.pack_forget()
            if str(self.return_to_campaign_right_btn.winfo_manager()) != "pack":
                self.return_to_campaign_right_btn.pack(fill=tk.X, pady=(0, 0))
    except Exception:
        pass

    approve_hint_text = ""
    approve_hint_tone = "muted"
    campaign_context = not self._is_free_mode_session_context()
    try:
        graph_entry_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_entry_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_entry_context.get("graph_edge_key"),
        graph_entry_context.get("graph_gate_id"),
    )
    graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) or graph_gate_id
    graph_display_gate_id = graph_display_gate_id or graph_gate_id
    graph_repair_origin_edge_key = str(
        graph_entry_context.get("repair_origin_edge_key")
        or graph_entry_context.get("source_graph_edge_key")
        or ""
    ).strip()
    graph_repair_origin_gate_id = str(
        graph_entry_context.get("repair_origin_gate_id")
        or graph_entry_context.get("source_graph_gate_id")
        or ""
    ).strip().upper()
    graph_repair_origin_gate_id = campaign_gate_id_for_edge(
        graph_repair_origin_edge_key,
        graph_repair_origin_gate_id,
    )
    graph_gate_is_t04 = bool(graph_gate_id == "T03")
    graph_gate_is_t05 = bool(graph_gate_id == "T04")
    graph_gate_is_t06 = bool(graph_gate_id == "T05")
    graph_gate_is_t05_repair_from_t07 = bool(graph_gate_is_t05 and graph_repair_origin_gate_id == "T06")
    graph_gate_known = bool(graph_gate_id)
    if graph_gate_known and not lightweight:
        try:
            if graph_gate_is_t05_repair_from_t07:
                return_copy = dispatch_get_campaign_return_to_graph_copy(self)
                self.approve_btn_row.configure(
                    text=str(return_copy.get("section") or " Przekazanie do puli YOLO ")
                )
            else:
                self.approve_btn_row.configure(text=f" Status bramki {graph_display_gate_id or graph_gate_id} ")
        except Exception:
            pass
    current_iteration_num = 0
    total_project_images = 0
    if project_active:
        try:
            from ..campaign_manager import CAMPAIGN
            current_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            current_iteration_num = 0
        try:
            total_project_images = int(self._get_campaign_project_total_image_count() or 0)
        except Exception:
            total_project_images = 0
    if project_active and (current_step == 2 or repair_mode) and approval_iteration_target == "plate":
        pending_count = (
            int(len(self.current_annotations or []))
            if self.current_annotations
            else int(campaign_pending_summary.get("pending_count", len(self.current_annotations or [])) or 0)
        )
        future_project_images = int(project_approved_images or 0) + int(approval_marked_images or 0)
        future_project_plates = int(project_approved_plates or 0) + int(approval_marked_plates or 0)
        hint_lines = [
            f"Projekt: {total_project_images} zdjęć łącznie.",
            (
                f"W stage tej iteracji: {pending_count} obrazów do sprawdzenia."
                if pending_count > 0
                else "W stage tej iteracji: 0 obrazów do sprawdzenia."
            ),
            (
                f"Wcześniej zatwierdzone w projekcie: {project_approved_images} obrazów / {project_approved_plates} tablic."
                if (project_approved_images > 0 or project_approved_plates > 0)
                else "Wcześniej zatwierdzone w projekcie: 0 obrazów / 0 tablic."
            ),
        ]
        if cumulative_project_plate_ready:
            if approval_marked_images > 0 or approval_marked_plates > 0:
                hint_lines.append(
                    (
                        f"Bramkę {graph_display_gate_id} możesz już zamknąć. Po {graph_display_gate_id} projekt będzie miał {future_project_images} zatwierdzonych obrazów i {future_project_plates} tablic do treningu modelu tablic."
                        if graph_gate_is_t05
                        else f"Po zamknięciu tej bramki projekt będzie miał {future_project_images} zatwierdzonych obrazów i {future_project_plates} tablic."
                    )
                )
                hint_lines.append(
                    "Jeśli masz jeszcze czas na kilka kolejnych [OK], zwykle warto dopisać je teraz w Z2, bo większy zbiór projektu lepiej pracuje na korzyść następnych iteracji modelu."
                )
            else:
                hint_lines.append(
                    (
                        f"Bramkę {graph_display_gate_id} możesz już zamknąć. Projekt ma już {project_approved_images} zatwierdzonych obrazów i {project_approved_plates} tablic, więc {graph_display_gate_id} nie wymaga nowych [OK] w tej iteracji."
                        if graph_gate_is_t05
                        else f"Bramkę można zamknąć. Projekt ma już {project_approved_images} zatwierdzonych obrazów i {project_approved_plates} tablic z poprzednich iteracji."
                    )
                )
                hint_lines.append(
                    "To znaczy, że szybkie zatwierdzenie jest legalne, ale niekoniecznie najlepsze. Jeśli zależy Ci na mocniejszym materiale do kolejnych iteracji, warto wejść jeszcze do Z2 i dopisać kilka nowych [OK]."
                )
            approve_hint_tone = "success"
        elif approval_total_plates > 0:
            missing_plates = max(0, int(min_plate_approval_plates) - int(effective_project_plate_plates or 0))
            hint_lines.append(
                (
                    f"Do otwarcia bramki {graph_display_gate_id} potrzebujesz jeszcze {missing_plates} zatwierdzonych tablic."
                    if graph_gate_is_t05 and missing_plates > 0
                    else f"Do otwarcia tej bramki potrzebujesz jeszcze {missing_plates} zatwierdzonych tablic."
                    if missing_plates > 0
                    else (
                        f"Dodaj jeszcze poprawne oznaczenia i zatwierdź obrazy jako [OK], aby otworzyć bramkę {graph_display_gate_id}."
                        if graph_gate_is_t05
                        else "Dodaj jeszcze poprawne oznaczenia i zatwierdź obrazy jako [OK], aby otworzyć bramkę."
                    )
                )
            )
            approve_hint_tone = "warning"
        else:
            hint_lines.append(
                (
                    f"Najpierw przygotuj poprawne tablice w Z2 i oznacz je jako [OK]. Dopiero po zamknięciu bramki {graph_display_gate_id} zasilą materiał do treningu modelu tablic."
                    if graph_gate_is_t05
                    else "Najpierw przygotuj pierwsze poprawne tablice w Z2 i oznacz je jako [OK]. Po zamknięciu bramki dołączą do zbioru projektu."
                )
            )
            approve_hint_tone = "warning"
        approve_hint_text = "\n".join(line for line in hint_lines if str(line or "").strip())
        if graph_gate_is_t05_repair_from_t07:
            approve_hint_text = (
                "Tryb naprawczy T07 pozwala dopisać albo poprawić ramki tablic przed decyzją końcową. "
                "Zatwierdzone wcześniej zdjęcia są już w puli YOLO i nie wracają na listę. "
                "Po wyjściu wrócisz do bramki T07."
            )
    elif project_active and current_step == 2 and approval_iteration_target == "char" and not approval_xml_exists:
        pending_count = (
            int(len(self.current_annotations or []))
            if self.current_annotations
            else int(campaign_pending_summary.get("pending_count", len(self.current_annotations or [])) or 0)
        )
        if char_effective_ready:
            hint_lines = [char_effective_summary_line]
            hint_lines.append("Jeśli chcesz, możesz nadal dopisywać kolejne tablice. Większy zatwierdzony zbiór poprawi skuteczność następnych iteracji modelu tablic.")
            approve_hint_text = "\n".join(line for line in hint_lines if str(line or "").strip())
            approve_hint_tone = "success"
        elif char_effective_plates > 0:
            missing_plates = max(0, int(min_char_approval_plates) - int(char_display_plates or 0))
            hint_lines = [char_effective_summary_line]
            hint_lines.append(
                (
                    f"Do wejścia do Z3 brakuje jeszcze {missing_plates} tablic oznaczonych na obrazach ze statusem [OK]. "
                    f"Na liście Z2 masz obecnie {pending_count} obrazów tej iteracji do sprawdzenia."
                    if missing_plates > 0
                    else f"Na liście Z2 masz obecnie {pending_count} obrazów tej iteracji do sprawdzenia."
                )
            )
            hint_lines.append("Warto dalej dopisywać kolejne tablice, bo większy zatwierdzony zbiór poprawi skuteczność następnych iteracji modelu tablic.")
            approve_hint_text = "\n".join(line for line in hint_lines if str(line or "").strip())
            approve_hint_tone = "warning"
        else:
            if graph_gate_is_t04:
                approve_hint_text = (
                    "Pracujesz nad otwarciem bramki T04. "
                    f"Na liście Z2 masz obecnie {pending_count} obrazów tej iteracji do sprawdzenia. "
                    f"Utwórz XML, oznacz tablice i nadaj status [OK] obrazom z poprawnymi anotacjami. Minimum bramki T04 to {min_char_approval_plates} tablic. "
                    "Dopiero wtedy odblokuje się zamknięcie T04 i dalsza praca nad znakami."
                )
            else:
                approve_hint_text = (
                    "Ta bramka przygotowuje źródło tablic dla pracy nad znakami. "
                    f"Na liście Z2 masz obecnie {pending_count} obrazów tej iteracji do sprawdzenia. "
                    f"Utwórz XML, oznacz tablice i nadaj status [OK] obrazom z poprawnymi anotacjami. Minimum otwarcia bramki to {min_char_approval_plates} tablic. "
                    "Dopiero wtedy odblokuje się powrót do grafu i dalsza praca nad znakami."
                )
            approve_hint_tone = "info"
    elif project_active and (current_step == 2 or repair_mode) and approval_xml_exists:
        if approval_iteration_target == "char":
            if char_current_ready:
                hint_lines = [char_effective_summary_line]
                hint_lines.append("Minimalny próg bramki jest już spełniony, ale jeśli masz jeszcze chwilę, zwykle warto dopisać kolejne poprawne tablice w Z2. Większy zatwierdzony zbiór poprawi skuteczność następnych iteracji modelu tablic.")
                approve_hint_text = "\n".join(line for line in hint_lines if str(line or "").strip())
                approve_hint_tone = "success"
            elif char_effective_ready and int(char_display_plates or 0) >= int(min_char_approval_plates):
                hint_lines = [char_effective_summary_line]
                hint_lines.append("Możesz wrócić do grafu bez dokładania nowych [OK] w tej iteracji.")
                hint_lines.append("To jest wystarczające minimum, ale zwykle lepszą decyzją jest dopisanie jeszcze kilku poprawnych tablic w Z2, zanim zamkniesz etap.")
                approve_hint_text = "\n".join(line for line in hint_lines if str(line or "").strip())
                approve_hint_tone = "success"
            elif approval_total_plates > 0:
                missing_plates = max(0, int(min_char_approval_plates) - int(char_display_plates or 0))
                approve_hint_text = (
                    (
                        f"Aby wrócić do grafu i przebudować źródło znaków, potrzebujesz co najmniej {min_char_approval_plates} tablic na obrazach oznaczonych jako [OK]. "
                        if repair_mode
                        else (
                            f"Aby otworzyć bramkę T04, potrzebujesz co najmniej {min_char_approval_plates} tablic na obrazach oznaczonych jako [OK]. "
                            if graph_gate_is_t04
                            else f"Aby przejść z tablic do znaków, potrzebujesz co najmniej {min_char_approval_plates} tablic na obrazach oznaczonych jako [OK]. "
                        )
                    )
                    + (
                        f"Obecnie źródło dla pracy nad znakami ma {char_display_images} obrazów [OK] i {char_display_plates} tablic. "
                        f"Minimum: {int(min_char_approval_plates)} tablic. "
                    )
                    + (
                        f"Brakuje jeszcze {missing_plates} tablic. "
                        if missing_plates > 0
                        else ""
                    )
                    + (
                        "Przy jednej tablicy nie przygotujesz potem poprawnego train i val dla treningu znaków."
                        if repair_mode
                        else "Przy jednej tablicy nie przygotujesz potem poprawnego train i val dla treningu znakow."
                    )
                )
                approve_hint_tone = "warning"
            else:
                approve_hint_text = (
                    (
                        f"Aby wrócić do grafu i przebudować źródło znaków, przygotuj co najmniej {min_char_approval_plates} tablic i oznacz ich obrazy jako [OK]. "
                        if repair_mode
                        else (
                            f"Aby otworzyć bramkę T04, przygotuj co najmniej {min_char_approval_plates} tablic i oznacz ich obrazy jako [OK]. "
                            if graph_gate_is_t04
                            else f"Aby przejść do znaków w Z3, przygotuj co najmniej {min_char_approval_plates} tablic i oznacz ich obrazy jako [OK]. "
                        )
                    )
                    + (
                        "Dodaj lub popraw polygony tablic, a potem nadaj obrazom status [OK]. "
                        if manual_route
                        else "Popraw wynik albo przejdź od razu do ręcznej korekty i oznacz poprawne obrazy jako [OK]. "
                    )
                )
                approve_hint_tone = "warning"
        else:
            if int(effective_project_plate_plates or 0) >= int(min_plate_approval_plates):
                approve_hint_text = (
                    (
                        f"Bramka {graph_display_gate_id} jest gotowa do zamknięcia. Zatwierdzonych tablic: {effective_project_plate_plates}. "
                        if graph_gate_is_t05
                        else f"Gotowe do zamknięcia E2. Zatwierdzonych tablic: {effective_project_plate_plates}. "
                    )
                    + (
                        f"Możesz teraz wrócić do mapy kampanii i zamknąć {graph_display_gate_id}."
                        if graph_gate_is_t05
                        else "Możesz teraz od razu domknąć ten etap."
                        if campaign_context
                        else "Możesz teraz wyeksportować dataset YOLO Pose albo od razu domknąć ten etap."
                    )
                )
                approve_hint_tone = "success"
            elif int(effective_project_plate_plates or 0) > 0:
                missing_plates = max(0, int(min_plate_approval_plates) - int(effective_project_plate_plates or 0))
                approve_hint_text = (
                    (
                        f"Aby otworzyć bramkę {graph_display_gate_id}, potrzebujesz co najmniej {min_plate_approval_plates} zatwierdzonych tablic. "
                        if graph_gate_is_t05
                        else f"Aby odblokować domknięcie E2 w torze tablic, potrzebujesz co najmniej {min_plate_approval_plates} zatwierdzonych tablic. "
                    )
                    + f"Zatwierdzonych tablic: {effective_project_plate_plates}. "
                    + (
                        f"Brakuje jeszcze {missing_plates} tablic. "
                        if missing_plates > 0
                        else ""
                    )
                    + "Dodaj brakujące oznaczenia i zapisz zmiany."
                )
                approve_hint_tone = "warning"
            else:
                approve_hint_text = (
                    (
                        f"Aby otworzyć bramkę {graph_display_gate_id}, potrzebujesz co najmniej {min_plate_approval_plates} zatwierdzonych tablic. "
                        if graph_gate_is_t05
                        else f"Aby odblokować domknięcie E2, potrzebujesz co najmniej {min_plate_approval_plates} zatwierdzonych tablic 'plate'. "
                    )
                    + (
                        "Dodaj i zapisz polygony tablic, a potem oznacz poprawne obrazy jako [OK]. "
                        if manual_route
                        else "Popraw wynik albo dodaj tablice ręcznie, a potem oznacz poprawne obrazy jako [OK]. "
                    )
                    + (
                        "Ten sam warunek odblokowuje też eksport datasetu YOLO Pose."
                        if not campaign_context
                        else (
                            f"To warunek konieczny do otwarcia bramki {graph_display_gate_id}."
                            if graph_gate_is_t05
                            else "To warunek konieczny do domknięcia E2."
                        )
                    )
                )
                approve_hint_tone = "warning"

    approve_context_text = ""
    approve_hint_title_text = ""
    approve_breakdown_title_text = ""
    approve_hint_table_rows: list[tuple[str, str, str]] = []
    if project_active and (current_step == 2 or repair_mode) and approval_iteration_target == "plate":
        gate_current_images = int(effective_project_plate_images or 0)
        gate_current_plates = int(effective_project_plate_plates or 0)
        gate_missing_plates = max(0, int(min_plate_approval_plates) - gate_current_plates)
        gate_xml_required = bool(int(current_iteration_num or 0) <= 1)
        gate_xml_missing = bool(gate_xml_required and not approval_xml_exists)
        plate_gate_ready = bool(gate_current_plates >= int(min_plate_approval_plates) and not gate_xml_missing)
        quality_info = CONFIG.describe_yolo_pose_dataset_quality(gate_current_plates)
        quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
        quality_tone = str(quality_info.get("tone", "error") or "error").strip().lower()
        next_quality_label = str(quality_info.get("next_label", "") or "").strip()
        missing_next_quality = max(0, int(quality_info.get("missing_next", 0) or 0))
        quality_goal_state = build_campaign_gate_focus_state(0, quality_info)
        quality_next_label = str(quality_goal_state.get("row_label") or "Cel jakości")
        quality_next_text = str(quality_goal_state.get("text") or "MAX")
        quality_next_tone = str(quality_goal_state.get("tone") or "success")
        approve_hint_title_text = f"Bramka {graph_display_gate_id}" if graph_gate_known else "Bramka grafu"
        approve_hint_text = ""
        approve_hint_table_rows = [
            (
                "Status bramki",
                ("OTWARTA" if plate_gate_ready else "ZAMKNIĘTA"),
                ("success" if plate_gate_ready else "warning"),
            ),
            (
                "Plik XML anotacji",
                ("UTWORZONY" if approval_xml_exists else ("WYMAGANY - BRAK" if gate_xml_required else "BRAK")),
                ("success" if approval_xml_exists else "warning"),
            ),
            (
                f"Po zamknięciu bramki {graph_display_gate_id}" if graph_gate_known else "Po zamknięciu bramki",
                f"{gate_current_images} obrazów [OK] / {gate_current_plates} tablic",
                ("success" if plate_gate_ready else "warning"),
            ),
            (
                "Brakuje do minimum",
                (
                    f"{gate_missing_plates} tablic"
                    + (" / XML" if gate_xml_missing else "")
                ),
                ("success" if gate_missing_plates == 0 and not gate_xml_missing else "warning"),
            ),
            (
                "Jakość zbioru",
                quality_label,
                quality_tone,
            ),
            (
                quality_next_label,
                quality_next_text,
                quality_next_tone,
            ),
        ]
        if graph_gate_is_t05_repair_from_t07:
            approve_hint_title_text = "Naprawa T07"
            approve_hint_table_rows[0] = (
                "Status naprawy",
                ("GOTOWE DO POWROTU T07" if plate_gate_ready else "W TRAKCIE"),
                ("success" if plate_gate_ready else "warning"),
            )
            approve_hint_table_rows[2] = (
                "Pula tablic po uzupełnieniu",
                f"{gate_current_images} obrazĂłw [OK] / {gate_current_plates} tablic",
                ("success" if plate_gate_ready else "warning"),
            )
    elif project_active and current_step == 2 and approval_iteration_target == "char":
        gate_current_images = int(char_display_images or 0)
        gate_current_plates = int(char_display_plates or 0)
        gate_missing_plates = max(0, int(min_char_approval_plates) - gate_current_plates)
        gate_xml_required = bool(int(current_iteration_num or 0) <= 1)
        gate_xml_missing = bool(gate_xml_required and not approval_xml_exists)
        char_gate_ready = bool(gate_current_plates >= int(min_char_approval_plates) and not gate_xml_missing)
        quality_info = CONFIG.describe_yolo_pose_dataset_quality(gate_current_plates)
        quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
        quality_tone = str(quality_info.get("tone", "error") or "error").strip().lower()
        next_quality_label = str(quality_info.get("next_label", "") or "").strip()
        missing_next_quality = max(0, int(quality_info.get("missing_next", 0) or 0))
        quality_goal_state = build_campaign_gate_focus_state(0, quality_info)
        quality_next_label = str(quality_goal_state.get("row_label") or "Cel jakości")
        quality_next_text = str(quality_goal_state.get("text") or "MAX")
        quality_next_tone = str(quality_goal_state.get("tone") or "success")
        approve_hint_title_text = f"Bramka {graph_gate_id}" if graph_gate_known else "Bramka grafu"
        approve_hint_text = ""
        approve_hint_table_rows = [
            (
                "Status bramki",
                ("OTWARTA" if char_gate_ready else "ZAMKNIĘTA"),
                ("success" if char_gate_ready else "warning"),
            ),
            (
                "Plik XML anotacji",
                ("UTWORZONY" if approval_xml_exists else ("WYMAGANY - BRAK" if gate_xml_required else "BRAK")),
                ("success" if approval_xml_exists else "warning"),
            ),
            (
                "Już zatwierdzone w projekcie",
                f"{int(project_approved_images or 0)} obrazów [OK] / {int(project_approved_plates or 0)} tablic",
                ("success" if int(project_approved_plates or 0) > 0 else "muted"),
            ),
            (
                "Nowe [OK] z bieżącej listy",
                f"{int(char_display_current_images or 0)} obrazów [OK] / {int(char_display_current_plates or 0)} tablic",
                ("success" if int(char_display_current_plates or 0) > 0 else "warning"),
            ),
            (
                f"Źródło tablic po {graph_gate_id}" if graph_gate_known else "Źródło tablic",
                f"{gate_current_images} obrazów [OK] / {gate_current_plates} tablic",
                ("success" if gate_current_plates > 0 else "warning"),
            ),
            (
                "Brakuje do minimum",
                (
                    f"{gate_missing_plates} tablic"
                    + (" / XML" if gate_xml_missing else "")
                ),
                ("success" if gate_missing_plates == 0 and not gate_xml_missing else "warning"),
            ),
            (
                "Jakość źródła tablic",
                quality_label,
                quality_tone,
            ),
            (
                quality_next_label,
                quality_next_text,
                quality_next_tone,
            ),
        ]
    elif project_active and approval_iteration_target == "char" and (current_step >= 3 or repair_mode):
        if graph_gate_is_t06:
            approve_context_text = (
                f"To jest pomocnicze wejście do Z2 z bramki {CHAR_WORK_GATE_DISPLAY_ID}.\n\n"
                "Uzupełniasz albo korygujesz tablice, które mają zasilić pracę nad znakami. "
                "Obrazy oznaczone statusem [OK] zostaną po powrocie do grafu dołączone do projektowego "
                f"źródła tablic dla {CHAR_WORK_GATE_DISPLAY_ID}. Samo wyjście z Z2 nie zamyka bramki; decyzję podejmujesz na mapie grafu."
            )
            approve_hint_title_text = f"Warunek bramki {CHAR_WORK_GATE_DISPLAY_ID}"
        else:
            repair_gate_label = f"bramką {graph_gate_id}" if graph_gate_known else "źródłem tablic"
            approve_context_text = (
                f"Pracujesz w Z2 nad {repair_gate_label}.\n\n"
                "Po powrocie do grafu zdjęcia ze statusem [OK] trafią do katalogu zatwierdzonych. "
                "Tylko zdjęcia z tego katalogu będą użyte do wyodrębniania tablic dla pracy nad znakami. "
                "Po wyodrębnieniu tablic nie wracamy już do ich ponownej anotacji w tej samej ścieżce."
            )
            approve_hint_title_text = f"Źródło tablic dla bramki {graph_gate_id}" if graph_gate_known else "Źródło tablic"
        approve_hint_text = ""
        if graph_gate_is_t06:
            t06_rows = _build_t06_right_panel_rows_from_campaign_state(
                self,
                required_plates=int(min_char_approval_plates),
                run_dir=approval_run_dir,
            )
            if t06_rows is None:
                t06_current_ok_images = int(approval_marked_images or 0)
                t06_current_ok_plates = int(approval_marked_plates or 0)
                if self.current_annotations:
                    try:
                        t06_current_ok_images, t06_current_ok_plates = self._get_current_preview_plate_approved_counts()
                    except Exception:
                        t06_current_ok_images = int(approval_marked_images or 0)
                        t06_current_ok_plates = int(approval_marked_plates or 0)
                t06_rows = (
                _build_t06_right_panel_rows(
                    previous_images=int(project_approved_base_images or 0),
                    previous_plates=int(project_approved_base_plates or 0),
                    session_delta_images=int(t06_current_ok_images or 0),
                    session_delta_plates=int(t06_current_ok_plates or 0),
                    required_plates=int(min_char_approval_plates),
                )
                )
            approve_hint_table_rows.extend(t06_rows)
        else:
            approve_hint_table_rows.append(
                (
                    "Pula projektowa",
                    f"{int(total_project_images or 0)} zdjęć łącznie",
                    "info" if int(total_project_images or 0) > 0 else "muted",
                )
            )
            approve_hint_table_rows.append(
                (
                    "Liczba oznaczonych tablic",
                    f"{int(approval_total_plates or 0)} w bieżącym runie Z2",
                    "success" if int(approval_total_plates or 0) > 0 else "warning",
                )
            )
            approve_hint_table_rows.append(
                (
                    "Z tego do katalogu już zatwierdzonych trafiło",
                    f"{int(project_approved_images or 0)} obrazów / {int(project_approved_plates or 0)} tablic",
                    "success" if int(project_approved_images or 0) > 0 and int(project_approved_plates or 0) > 0 else "warning",
                )
            )
    elif campaign_context and str(approve_hint_text or "").strip():
        approve_hint_title_text = "Status bieżącego etapu"

    if graph_gate_id == "T03":
        approve_context_text = approve_context_text.replace("T04", "T03")
        approve_hint_text = approve_hint_text.replace("T04", "T03")
        approve_hint_title_text = approve_hint_title_text.replace("T04", "T03")
        approve_breakdown_title_text = approve_breakdown_title_text.replace("T04", "T03")
        approve_hint_table_rows = [
            (str(label).replace("T04", "T03"), str(value).replace("T04", "T03"), tone)
            for label, value, tone in approve_hint_table_rows
        ]
    if graph_gate_id == "T04":
        approve_context_text = approve_context_text.replace("T05", "T04")
        approve_hint_text = approve_hint_text.replace("T05", "T04")
        approve_hint_title_text = approve_hint_title_text.replace("T05", "T04")
        approve_breakdown_title_text = approve_breakdown_title_text.replace("T05", "T04")
        approve_hint_table_rows = [
            (str(label).replace("T05", "T04"), str(value).replace("T05", "T04"), tone)
            for label, value, tone in approve_hint_table_rows
        ]
    if graph_gate_id == "T05":
        approve_context_text = approve_context_text.replace("T06", "T05")
        approve_hint_text = approve_hint_text.replace("T06", "T05")
        approve_hint_title_text = approve_hint_title_text.replace("T06", "T05")
        approve_breakdown_title_text = approve_breakdown_title_text.replace("T06", "T05")
        approve_hint_table_rows = [
            (str(label).replace("T06", "T05"), str(value).replace("T06", "T05"), tone)
            for label, value, tone in approve_hint_table_rows
        ]
    if graph_gate_is_t05_repair_from_t07:
        approve_context_text = approve_context_text.replace("T07", "T06")
        approve_hint_text = approve_hint_text.replace("T07", "T06")
        approve_hint_title_text = approve_hint_title_text.replace("T07", "T06")
        approve_breakdown_title_text = approve_breakdown_title_text.replace("T07", "T06")
        approve_hint_table_rows = [
            (str(label).replace("T07", "T06"), str(value).replace("T07", "T06"), tone)
            for label, value, tone in approve_hint_table_rows
        ]

    if graph_gate_id and graph_display_gate_id and graph_display_gate_id != graph_gate_id:
        approve_context_text = approve_context_text.replace(graph_gate_id, graph_display_gate_id)
        approve_hint_text = approve_hint_text.replace(graph_gate_id, graph_display_gate_id)
        approve_hint_title_text = approve_hint_title_text.replace(graph_gate_id, graph_display_gate_id)
        approve_breakdown_title_text = approve_breakdown_title_text.replace(graph_gate_id, graph_display_gate_id)
        approve_hint_table_rows = [
            (
                str(label).replace(graph_gate_id, graph_display_gate_id),
                str(value).replace(graph_gate_id, graph_display_gate_id),
                tone,
            )
            for label, value, tone in approve_hint_table_rows
        ]
    elif approval_iteration_target == "plate" and graph_gate_known and graph_display_gate_id and graph_display_gate_id != "T05":
        approve_context_text = approve_context_text.replace("T05", graph_display_gate_id)
        approve_hint_text = approve_hint_text.replace("T05", graph_display_gate_id)
        approve_hint_title_text = approve_hint_title_text.replace("T05", graph_display_gate_id)
        approve_breakdown_title_text = approve_breakdown_title_text.replace("T05", graph_display_gate_id)
        approve_hint_table_rows = [
            (
                str(label).replace("T05", graph_display_gate_id),
                str(value).replace("T05", graph_display_gate_id),
                tone,
            )
            for label, value, tone in approve_hint_table_rows
        ]

    if not lightweight:
        try:
            self.approve_context_var.set(approve_context_text)
        except Exception:
            pass
        try:
            self.approve_hint_title_var.set(approve_hint_title_text)
        except Exception:
            pass
        try:
            self.approve_breakdown_title_var.set(approve_breakdown_title_text)
        except Exception:
            pass
        try:
            self.approve_gate_hint_var.set(approve_hint_text)
        except Exception:
            pass
    self._campaign_step2_approval_ready = bool(approve_ready)
    self._campaign_step2_approval_action = str(approval_action or "").strip()
    self._campaign_step2_approval_iteration_target = str(approval_iteration_target or "").strip()
    self._campaign_step2_approval_repair_mode = bool(repair_mode)
    self._campaign_step2_approval_hint_text = str(approve_hint_text or "").strip()
    self._campaign_step2_approval_hint_tone = str(approve_hint_tone or "muted").strip()
    try:
        self._campaign_step2_gate_overlay_state = self._build_campaign_z2_gate_overlay_state()
        self._place_preview_campaign_gate_overlay(force_render=True)
    except Exception:
        self._campaign_step2_gate_overlay_state = {}

    if (
        campaign_context
        and not approve_hint_table_rows
        and not str(approve_hint_text or "").strip()
        and (
            graph_gate_known
            or not str(approve_context_text or "").strip()
        )
    ):
        gate_state = {}
        try:
            gate_state = dict(getattr(self, "_campaign_step2_gate_overlay_state", {}) or {})
        except Exception:
            gate_state = {}
        if not gate_state:
            fallback_total_images = int(approval_images_with_plates or 0)
            fallback_total_plates = int(approval_total_plates or 0)
            fallback_approved_images = int(approval_marked_images or 0)
            fallback_approved_plates = int(approval_marked_plates or 0)
            if (fallback_total_plates <= 0 or fallback_approved_plates <= 0) and self.current_annotations:
                try:
                    fallback_total_images, fallback_total_plates = self._count_plate_annotations(self.current_annotations)
                except Exception:
                    pass
                try:
                    fallback_approved_images, fallback_approved_plates = self._get_current_preview_plate_approved_counts()
                except Exception:
                    pass
            fallback_required = (
                int(min_char_approval_plates)
                if approval_iteration_target == "char" or graph_gate_id in {"T03", "T05"}
                else int(min_plate_approval_plates)
            )
            fallback_missing = max(0, int(fallback_required) - int(fallback_approved_plates))
            gate_state = {
                "gate_id": graph_gate_id,
                "title": f"Bramka {graph_gate_id}" if graph_gate_known else "Status pracy Z2",
                "tone": "info",
                "instruction": "Pracujesz w Z2 w kontekście kampanii. Prawy panel pokazuje stan aktywnego pliku anotacji i pozycji oznaczonych jako [OK].",
                "status": "GOTOWE" if fallback_missing <= 0 and approval_xml_exists else "W TRAKCIE",
                "ready": bool(fallback_missing <= 0 and approval_xml_exists),
                "approved_images": fallback_approved_images,
                "approved_plates": fallback_approved_plates,
                "required_plates": fallback_required,
                "missing_plates": fallback_missing,
                "xml": (
                    "XML: OK"
                    if approval_xml_exists
                    else f"XML: aktywny run ({fallback_total_images} obrazów / {fallback_total_plates} tablic)"
                    if fallback_total_plates > 0
                    else "XML: brak",
                ),
            }
            try:
                logger.info(
                    "[Z2 RIGHT] fallback status panel: gate=%s target=%s step=%s xml=%s ok=%s/%s total=%s/%s",
                    graph_gate_id or "-",
                    approval_iteration_target or "-",
                    current_step,
                    int(bool(approval_xml_exists)),
                    fallback_approved_images,
                    fallback_approved_plates,
                    fallback_total_images,
                    fallback_total_plates,
                )
            except Exception:
                pass
        if gate_state or graph_gate_known:
            fallback_gate_id = str(gate_state.get("gate_id") or graph_gate_id or "").strip().upper()
            fallback_display_gate_id = campaign_visible_gate_id(fallback_gate_id)
            if fallback_gate_id and fallback_display_gate_id and fallback_display_gate_id != fallback_gate_id:
                for key in ("title", "message", "detail", "instruction"):
                    try:
                        gate_state[key] = str(gate_state.get(key) or "").replace(fallback_gate_id, fallback_display_gate_id)
                    except Exception:
                        pass
            fallback_title = str(gate_state.get("title") or "").strip()
            if not fallback_title:
                fallback_title = f"Bramka {fallback_display_gate_id or fallback_gate_id}" if fallback_gate_id else "Bramka grafu"
            fallback_tone = str(gate_state.get("tone") or "info").strip().lower()
            fallback_message = str(gate_state.get("message") or "").strip()
            fallback_detail = str(gate_state.get("detail") or "").strip()
            fallback_instruction = str(gate_state.get("instruction") or "").strip()
            approve_context_text = "\n".join(
                part
                for part in (fallback_instruction, fallback_message, fallback_detail)
                if str(part or "").strip()
            ) or "Pracujesz w Z2 nad materiałem wymaganym przez wybraną bramkę grafu."
            approve_hint_title_text = fallback_title
            approve_hint_tone = fallback_tone or "info"
            approve_hint_text = ""
            if fallback_gate_id == "T05":
                fallback_approved_images = int(gate_state.get("approved_images", 0) or 0)
                fallback_approved_plates = int(gate_state.get("approved_plates", 0) or 0)
                fallback_required_plates = int(gate_state.get("required_plates", 0) or 0)
                approve_hint_table_rows = _build_t06_right_panel_rows(
                    previous_images=fallback_approved_images,
                    previous_plates=fallback_approved_plates,
                    session_delta_images=0,
                    session_delta_plates=0,
                    required_plates=fallback_required_plates,
                )
            else:
                fallback_missing_plates = int(gate_state.get("missing_plates", 0) or 0)
                fallback_missing_row_label = str(
                    gate_state.get("missing_focus_row_label") or "Do otwarcia bramki brakuje"
                ).strip()
                fallback_missing_text = str(
                    gate_state.get("missing_focus_text")
                    or f"{fallback_missing_plates} tablic zatwierdzonych [OK]"
                ).strip()
                fallback_missing_tone = str(
                    gate_state.get("missing_focus_tone")
                    or ("success" if fallback_missing_plates <= 0 else "warning")
                ).strip().lower()
                if fallback_missing_plates <= 0 and "otwarcia bramki" in fallback_missing_row_label.lower():
                    try:
                        fallback_quality_info = CONFIG.describe_yolo_pose_dataset_quality(
                            int(gate_state.get("approved_plates", 0) or 0)
                        )
                    except Exception:
                        fallback_quality_info = {}
                    focus_state = build_campaign_gate_focus_state(0, fallback_quality_info)
                    fallback_missing_row_label = str(focus_state.get("row_label") or "Minimum bramki")
                    fallback_missing_text = str(focus_state.get("text") or "Spełnione")
                    fallback_missing_tone = str(focus_state.get("tone") or "success")
                approve_hint_table_rows = [
                    (
                        "Status bramki",
                        str(gate_state.get("status") or ("OTWARTA" if gate_state.get("ready") else "ZAMKNIĘTA")),
                        "success" if bool(gate_state.get("ready")) else "warning",
                    ),
                    (
                        "Zatwierdzone [OK]",
                        f"{int(gate_state.get('approved_images', 0) or 0)} obrazów / {int(gate_state.get('approved_plates', 0) or 0)} tablic",
                        "success" if int(gate_state.get("approved_plates", 0) or 0) > 0 else "warning",
                    ),
                    (
                        fallback_missing_row_label,
                        fallback_missing_text,
                        fallback_missing_tone,
                    ),
                    (
                        "XML anotacji",
                        str(gate_state.get("xml") or "XML: brak danych"),
                        "success" if "OK" in str(gate_state.get("xml") or "") else "warning",
                    ),
                ]
            self._campaign_step2_approval_hint_text = ""
            self._campaign_step2_approval_hint_tone = str(approve_hint_tone or "info").strip()
            if not lightweight:
                try:
                    self.approve_context_var.set(approve_context_text)
                    self.approve_hint_title_var.set(approve_hint_title_text)
                    self.approve_gate_hint_var.set("")
                except Exception:
                    pass

    if campaign_context and graph_gate_known and not bool(getattr(self, "_preview_fullscreen_active", False)):
        try:
            self._annotation_right_panel_visible = True
            self._sync_main_pane_right_panel_visibility()
        except Exception:
            pass
        if not lightweight:
            try:
                self._set_widget_packed(
                    getattr(self, "approve_btn_row", None),
                    True,
                    fill=tk.X,
                    pady=(8, 0),
                )
            except Exception:
                pass
            try:
                logger.info(
                    "[Z2 RIGHT] render gate=%s step=%s target=%s show=%s rows=%s context=%s hint=%s current=%s packed_hint=%s packed_context=%s",
                    graph_gate_id or "-",
                    current_step,
                    approval_iteration_target or "-",
                    int(bool(self._should_show_right_panel())),
                    int(len(approve_hint_table_rows or [])),
                    int(bool(str(approve_context_text or "").strip())),
                    int(bool(str(approve_hint_text or "").strip())),
                    int(len(getattr(self, "current_annotations", []) or [])),
                    str(getattr(self, "approve_hint_box", None).winfo_manager() if getattr(self, "approve_hint_box", None) is not None else ""),
                    str(getattr(self, "approve_context_box", None).winfo_manager() if getattr(self, "approve_context_box", None) is not None else ""),
                )
            except Exception:
                pass

    approve_hint_lbl = getattr(self, "approve_gate_hint_lbl", None)
    if not lightweight:
        if approve_hint_lbl is not None:
            self._set_inline_label_state(approve_hint_lbl, tone=approve_hint_tone, emphasis=False)
        self._set_approve_hint_box_state(approve_hint_tone if (approve_hint_text or approve_hint_table_rows) else "muted")
        if not str(approve_hint_text or "").strip():
            try:
                self.approve_gate_hint_var.set("")
            except Exception:
                pass
            try:
                self._set_widget_packed(approve_hint_lbl, False)
            except Exception:
                pass
    if not lightweight:
        self._set_widget_packed(
            getattr(self, "approve_context_box", None),
            bool(self._should_show_right_panel() and str(approve_context_text or "").strip()),
            fill=tk.X,
            pady=(0, 10),
            before=getattr(self, "approve_hint_box", None),
        )
    approve_context_lbl = getattr(self, "approve_context_lbl", None)
    if approve_context_lbl is not None and not lightweight:
        self._set_inline_label_state(approve_context_lbl, tone="info", emphasis=True)
    if not lightweight:
        self._set_approve_context_box_state("info")
    self._render_compact_info_table(
        getattr(self, "approve_hint_table_frame", None),
        approve_hint_table_rows,
        default_value_tone=("success" if approve_hint_tone == "success" else "warning"),
        reuse_existing=bool(lightweight or (campaign_context and graph_gate_known)),
        show_header=False,
    )
    if not lightweight:
        self._set_widget_packed(
            getattr(self, "approve_hint_table_frame", None),
            bool(approve_hint_table_rows),
            fill=tk.X,
            pady=(6, 8),
            before=getattr(self, "approve_gate_hint_lbl", None),
        )
        self._set_widget_packed(
            getattr(self, "approve_hint_box", None),
            bool(
                self._should_show_right_panel()
                and (str(approve_hint_text or "").strip() or approve_hint_table_rows)
            ),
            fill=tk.X,
            pady=(0, 10),
            before=getattr(self, "approve_breakdown_box", None),
        )
        self._set_widget_packed(
            getattr(self, "approve_hint_title_lbl", None),
            bool(str(approve_hint_title_text or "").strip()),
            fill=tk.X,
            before=(
                getattr(self, "approve_hint_table_frame", None)
                if approve_hint_table_rows
                else getattr(self, "approve_gate_hint_lbl", None)
            ),
        )
        if campaign_context and graph_gate_known and approve_hint_table_rows:
            try:
                self._set_widget_packed(
                    getattr(self, "approve_btn_row", None),
                    True,
                    fill=tk.X,
                    pady=(8, 0),
                )
                self._set_widget_packed(
                    getattr(self, "approve_context_box", None),
                    bool(str(approve_context_text or "").strip()),
                    fill=tk.X,
                    pady=(0, 10),
                )
                self._set_widget_packed(
                    getattr(self, "approve_hint_box", None),
                    True,
                    fill=tk.X,
                    pady=(0, 10),
                )
                self._set_widget_packed(
                    getattr(self, "approve_hint_title_lbl", None),
                    bool(str(approve_hint_title_text or "").strip()),
                    fill=tk.X,
                )
                self._set_widget_packed(
                    getattr(self, "approve_hint_table_frame", None),
                    True,
                    fill=tk.X,
                    pady=(6, 8),
                )
            except Exception:
                pass
    if not lightweight:
        self._set_inline_label_state(getattr(self, "approve_hint_title_lbl", None), tone="muted", emphasis=True)
    if not lightweight:
        self._set_widget_packed(
            getattr(self, "approve_breakdown_title_lbl", None),
            bool(str(approve_breakdown_title_text or "").strip()),
            fill=tk.X,
            before=(
                getattr(self, "approve_breakdown_table_frame", None)
                if campaign_context
                else getattr(self, "approve_breakdown_lbl", None)
            ),
        )
    if not lightweight:
        self._set_inline_label_state(getattr(self, "approve_breakdown_title_lbl", None), tone="muted", emphasis=True)
    if not lightweight:
        self._set_widget_packed(
            getattr(self, "approve_gate_hint_lbl", None),
            bool(str(approve_hint_text or "").strip()),
            fill=tk.X,
            pady=(4, 8),
        )
        self._normalize_approve_panel_order()
        if campaign_context and graph_gate_known:
            try:
                logger.info(
                    "[Z2 RIGHT] packed gate=%s row=%s context=%s hint=%s table=%s breakdown=%s rows=%s",
                    graph_gate_id or "-",
                    str(getattr(self, "approve_btn_row", None).winfo_manager() if getattr(self, "approve_btn_row", None) is not None else ""),
                    str(getattr(self, "approve_context_box", None).winfo_manager() if getattr(self, "approve_context_box", None) is not None else ""),
                    str(getattr(self, "approve_hint_box", None).winfo_manager() if getattr(self, "approve_hint_box", None) is not None else ""),
                    str(getattr(self, "approve_hint_table_frame", None).winfo_manager() if getattr(self, "approve_hint_table_frame", None) is not None else ""),
                    str(getattr(self, "approve_breakdown_box", None).winfo_manager() if getattr(self, "approve_breakdown_box", None) is not None else ""),
                    int(len(approve_hint_table_rows or [])),
                )
            except Exception:
                pass
    if not lightweight:
        self._refresh_approval_breakdown_canvas()
    if not lightweight and not (campaign_context and graph_gate_known and approve_hint_table_rows):
        try:
            _force_render_campaign_graph_right_panel(self)
        except Exception as exc:
            logger.debug(f"Nie udało się wymusić prawego panelu bramki grafu Z2: {exc}")

def _start_annotation(self):
    route = self._get_workflow_route()
    manual_entry_mode = self._get_manual_entry_mode()
    try:
        campaign_manual_xml_start = bool(
            not self._is_free_mode_session_context()
            and (
                (route == "manual" and manual_entry_mode == "new")
                or str(self._coerce_workflow_step() or "").strip().lower() == "manual_start"
            )
        )
    except Exception:
        campaign_manual_xml_start = False
    if campaign_manual_xml_start:
        route = "manual"
        manual_entry_mode = "new"
        try:
            self._set_workflow_route_state("manual", campaign_context=True)
            self._set_manual_entry_mode_state("new", campaign_context=True)
            self.manual_xml_template_var.set(True)
            self.manual_vehicle_assist_var.set(False)
            self._set_workflow_step_state("manual_start", campaign_context=True)
        except Exception:
            pass
    if self._is_free_mode_session_context() and not route:
        return messagebox.showwarning("Wybierz tor", "Najpierw wybierz autoanotację albo anotację ręczną.")

    if route == "manual" and manual_entry_mode == "continue":
        selected_run = self._get_selected_manual_review_history_run_dir()
        if selected_run is None:
            return messagebox.showwarning(
                "Wybierz run anotacji",
                "W trybie ręcznej kontynuacji wybierz run anotacji z historii korekt.",
            )
        return self._open_existing_run_for_manual_review(
            run_dir=selected_run,
            allow_fallback=False,
            show_dialog=False,
        )

    in_d = self.input_dir_var.get().strip()
    if not in_d or not Path(in_d).exists():
        return messagebox.showerror("Błąd", "Wybierz folder z obrazami wejściowymi.")

    resolved_input_dir = self._resolve_annotation_input_images_dir(Path(in_d))
    if resolved_input_dir is not None and str(resolved_input_dir) != str(in_d):
        try:
            self.input_dir_var.set(str(resolved_input_dir))
            self.plate_dataset_images_var.set(str(resolved_input_dir))
            self.current_input_dir = resolved_input_dir
        except Exception:
            pass
        in_d = str(resolved_input_dir)

    self._clear_annotation_run_scope_meta()

    try:
        mode_text = self._normalize_mode_value()
        if route == "auto":
            if self._get_auto_vehicle_choice() == "skip":
                mode_text = "B: Tylko tablice"
            elif self._get_auto_vehicle_choice() == "use":
                mode_text = "C: Pojazdy + tablice"
        self.mode_var.set(mode_text)
        manual_template = self._manual_xml_template_enabled()
        manual_vehicle_assist = self._manual_vehicle_assist_enabled()
        if route == "manual" and manual_entry_mode == "new" and manual_template and self._has_active_manual_template_run():
            self._refresh_free_mode_workflow_ui()
            return messagebox.showinfo(
                "Run juz istnieje",
                "Dla tego stanu Z2 run recznej anotacji jest juz utworzony.\n\n"
                "Przejdz do edycji istniejacego runu zamiast tworzyc kolejny XML."
            )
        effective_input_dir = Path(in_d)
        self._annotation_source_input_dir = Path(in_d)
        self._pending_source_image_map = {}
        scope_selection = None
        if not self._is_free_mode_session_context():
            try:
                source_plan = self._collect_campaign_auto_annotation_sources(
                    effective_input_dir,
                    include_previous=(bool(self.campaign_reuse_manual_var.get()) if not manual_template else False),
                    exclude_manual_touched=True,
                )
                pending_images = list(source_plan.get("image_paths") or [])
                self._pending_source_image_map = dict(source_plan.get("image_map") or {})
                manual_skip_count = int(source_plan.get("manual_skip_count", 0) or 0)
                char_effective_skip_count = int(source_plan.get("char_effective_skip_count", 0) or 0)
                try:
                    live_manual_skip_names = {
                        str(name or "").strip().lower()
                        for name in self._collect_preview_manually_touched_filenames()
                        if str(name or "").strip()
                    }
                except Exception:
                    live_manual_skip_names = set()
                try:
                    overlay_skip_names = {
                        str(name or "").strip().lower()
                        for name in dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {}).keys()
                        if str(name or "").strip()
                    }
                except Exception:
                    overlay_skip_names = set()
                effective_skip_names = set(overlay_skip_names) | set(live_manual_skip_names)
                defer_scope_protection_to_modal = bool(route == "auto" and not manual_template)
                if effective_skip_names and not defer_scope_protection_to_modal:
                    filtered_pending_images = []
                    filtered_source_map = {}
                    skipped_overlay_count = 0
                    for image_path in pending_images:
                        safe_name = str(getattr(image_path, "name", "") or "").strip().lower()
                        if safe_name in effective_skip_names:
                            skipped_overlay_count += 1
                            continue
                        filtered_pending_images.append(image_path)
                        filtered_source_map[str(image_path.name)] = image_path
                    if skipped_overlay_count > 0:
                        pending_images = filtered_pending_images
                        self._pending_source_image_map = filtered_source_map
                        manual_skip_count += skipped_overlay_count
                if not pending_images:
                    return messagebox.showinfo(
                        "Brak nowych obrazów",
                        (
                            "Ten zestaw nie zawiera już obrazów oczekujących na pracę w Z2.\n\n"
                            f"{char_effective_skip_count} obrazów jest już użytych w aktywnym E3, "
                            "a pozostałe są już w zatwierdzonym zbiorze projektu."
                            if char_effective_skip_count > 0
                            else "Ten zestaw nie zawiera już obrazów oczekujących na pracę w Z2.\n\n"
                            "Wszystkie obrazy z tego wejścia są już w zatwierdzonym zbiorze projektu."
                        )
                    )
                if route == "auto" and not manual_template:
                    try:
                        logger.info(
                            "[Z2 AUTO MODAL] opening campaign scope modal images=%s input=%s",
                            len(pending_images),
                            str(effective_input_dir),
                        )
                    except Exception:
                        pass
                    scope_selection = self._prompt_plate_auto_scope_choice(
                        candidate_image_paths=pending_images,
                    )
                    if scope_selection is None:
                        try:
                            logger.info("[Z2 AUTO MODAL] scope modal cancelled")
                        except Exception:
                            pass
                        return
                    try:
                        logger.info(
                            "[Z2 AUTO MODAL] scope selected mode=%s count=%s",
                            str(scope_selection.get("mode") or ""),
                            len(list(scope_selection.get("image_paths") or [])),
                        )
                    except Exception:
                        pass
                    selected_scope_paths = self._dedupe_image_paths_by_name(
                        scope_selection.get("image_paths") or pending_images
                    )
                else:
                    selected_scope_paths = self._dedupe_image_paths_by_name(pending_images)
                    scope_selection = {
                        "mode": "all",
                        "label": "Cały zestaw zdjęć",
                        "image_paths": selected_scope_paths,
                    }
                if not selected_scope_paths:
                    return messagebox.showinfo(
                        "Brak obrazów w wybranym zakresie",
                        "Wybrany zakres autoanotacji nie zawiera obrazów gotowych do przetworzenia.",
                    )
                try:
                    self._apply_campaign_manual_reuse_context(source_plan)
                except Exception:
                    pass
                try:
                    manifest_expected_count = self._get_campaign_iteration_manifest_image_count()
                    manifest_base_images = self._get_campaign_iteration_manifest_image_paths(Path(in_d))
                    raw_base_images = (
                        manifest_base_images
                        if manifest_expected_count > 0
                        else get_image_files(Path(in_d))
                    )
                except Exception:
                    raw_base_images = []
                needs_scope_dir = bool(
                    bool(source_plan.get("manifest_scoped", False))
                    or
                    int(source_plan.get("reused_count", 0) or 0) > 0
                    or int(source_plan.get("approved_skip_count", 0) or 0) > 0
                    or len(selected_scope_paths) != len(raw_base_images)
                )
                if needs_scope_dir:
                    scope_dir, scope_map = self._build_annotation_input_subset_dir(
                        selected_scope_paths,
                        scope_suffix=str(scope_selection.get("mode") or "all"),
                    )
                    if scope_dir is None:
                        return messagebox.showerror(
                            "Błąd zakresu autoanotacji",
                            "Nie udało się przygotować wybranego zakresu obrazów do autoanotacji.",
                        )
                    effective_input_dir = scope_dir
                    self._pending_source_image_map = dict(scope_map or {})
                else:
                    effective_input_dir = Path(in_d)
                    self._pending_source_image_map = {}
                self._remember_annotation_run_scope_meta(
                    mode=str(scope_selection.get("mode") or "all"),
                    label=str(scope_selection.get("label") or "Cały zestaw zdjęć"),
                    count=len(selected_scope_paths),
                    filenames=[Path(path).name for path in selected_scope_paths],
                )
                if manual_skip_count > 0:
                    logger.info(
                        f"Z2 pominie {manual_skip_count} obrazów poprawionych ręcznie przy budowie zestawu autoanotacji."
                    )
            except Exception as e:
                logger.exception("Nie udało się przygotować kampanijnego modala autoanotacji Z2")
                if route == "auto" and not manual_template:
                    return messagebox.showerror(
                        "Błąd autoanotacji",
                        (
                            "Nie udało się przygotować okna autoanotacji Z2, więc proces nie został uruchomiony.\n\n"
                            f"Szczegóły: {e}"
                        ),
                    )
                logger.debug(f"Nie udało się przygotować zestawu pending Z2: {e}")
                effective_input_dir = Path(in_d)
        elif route == "auto" and not manual_template:
            try:
                raw_scope_images = get_image_files(effective_input_dir)
            except Exception:
                raw_scope_images = []
            if not raw_scope_images:
                if self._is_free_mode_session_context():
                    try:
                        self._set_workflow_step("auto_input")
                    except Exception:
                        pass
                return messagebox.showerror(
                    "Brak obrazów",
                    "Wybrany folder nie zawiera obrazów gotowych do autoanotacji Z2.",
                )
            if self._can_offer_preview_scope_selection_for_auto_run():
                scope_selection = self._prompt_plate_auto_scope_choice(
                    candidate_image_paths=raw_scope_images,
                )
                if scope_selection is None:
                    return
                selected_scope_paths = self._dedupe_image_paths_by_name(
                    scope_selection.get("image_paths") or raw_scope_images
                )
            else:
                selected_scope_paths = self._dedupe_image_paths_by_name(raw_scope_images)
                scope_selection = {
                    "mode": "all",
                    "label": "Cały zestaw zdjęć",
                    "image_paths": selected_scope_paths,
                }
            if not selected_scope_paths:
                return messagebox.showinfo(
                    "Brak obrazów w wybranym zakresie",
                    "Wybrany zakres autoanotacji nie zawiera obrazów do przetworzenia.",
                )
            if len(selected_scope_paths) != len(raw_scope_images):
                scope_dir, scope_map = self._build_annotation_input_subset_dir(
                    selected_scope_paths,
                    scope_suffix=str(scope_selection.get("mode") or "selected"),
                )
                if scope_dir is None:
                    return messagebox.showerror(
                        "Błąd zakresu autoanotacji",
                        "Nie udało się przygotować wybranego zakresu obrazów do autoanotacji.",
                    )
                effective_input_dir = scope_dir
                self._pending_source_image_map = dict(scope_map or {})
            self._remember_annotation_run_scope_meta(
                mode=str(scope_selection.get("mode") or "all"),
                label=str(scope_selection.get("label") or "Cały zestaw zdjęć"),
                count=len(selected_scope_paths),
                filenames=[Path(path).name for path in selected_scope_paths],
            )
        if route == "auto":
            mode_text = (
                "C: Pojazdy + tablice"
                if self._get_auto_vehicle_choice() == "use"
                else "B: Tylko tablice"
            )
            self.mode_var.set(mode_text)
        selected_device = self._get_effective_yolo_device_choice()
        dev = self._device_to_ultralytics(selected_device)
        success, msg = True, ""
        conf = self.conf_var.get()
        v_p = None
        p_p = None
        safe_output_dir = None

        if not manual_template:
            self._validate_models()

            pending_downloads = self._collect_pending_model_downloads(mode_text)
            if not self._confirm_and_download_missing_models(pending_downloads):
                return

            v_p = self._get_model_path("vehicle") if self._mode_uses_vehicle(mode_text) else None
            p_p = self._get_model_path("plate") if self._mode_uses_plate(mode_text) else None
        elif manual_vehicle_assist:
            self._validate_vehicle_model_selection()
            pending_downloads = self._collect_pending_model_downloads("C: Pojazdy + tablice")
            if not self._confirm_and_download_missing_models(pending_downloads):
                return
            v_p = self._get_model_path("vehicle")

        self._manual_review_active = False
        self._manual_review_from_auto = False
        self._manual_review_origin_route = ""
        self._manual_review_export_ready = False
        self._current_run_manual_template = manual_template
        self._current_run_manual_vehicle_assist = manual_vehicle_assist
        safe_output_dir = self._coerce_annotation_output_dir(self.output_dir_var.get())
        self.output_dir_var.set(str(safe_output_dir))
        app = getattr(self, "app", None)
        if app is not None and hasattr(app, "try_begin_exclusive_operation"):
            ok, busy_message = app.try_begin_exclusive_operation("z2.annotation.run", "Z2: przygotowanie anotacji")
            if not ok:
                return messagebox.showinfo("Proces w toku", busy_message)
        else:
            self.app.set_processing(True)
        self._annotation_stop_requested = False
        self.is_processing = True
        self._last_run_progress_visible = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.approve_btn.config(state=tk.DISABLED)
        self.export_plate_dataset_btn.config(state=tk.DISABLED)
        try:
            self._refresh_step2_action_states(lightweight=False)
        except Exception:
            pass
        self.progress.configure(value=0)
        self._set_progress_counters(0, 0, 0)
        self._set_status_label_state(
            (
                "Ładowanie modelu pojazdów..."
                if manual_template and manual_vehicle_assist
                else "Ładowanie modeli YOLO..."
                if not manual_template
                else "Przygotowywanie szablonu anotacji..."
            ),
            "info",
        )
        self._set_post_annotation_hint("")
        with self._progress_update_lock:
            self._pending_progress_update = None
            self._progress_update_flush_queued = False
        self._progress_update_seen = False

        self._set_preview_processing_overlay(
            not (
                manual_template
                and self._is_free_mode_session_context()
                and route == "manual"
                and manual_entry_mode == "new"
            ),
            title=(
                "Trwa przygotowanie autoanotacji"
                if not manual_template
                else "Trwa przygotowanie XML"
            ),
            details="Ładuję modele i przygotowuję run. Za chwilę pojawi się właściwy postęp.",
            cancel_visible=False,
        )
        self._update_preview_processing_overlay_progress(
            pct=0.0,
            current=0,
            total=0,
            filename="",
            meta_text="0% | przygotowanie modeli",
        )
        if not bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
            try:
                self.frame.update_idletasks()
                self.frame.update()
            except Exception:
                pass

        if self.annotator is not None:
            try:
                self.annotator.unload_models()
            except Exception:
                pass
            self.annotator = None

        if not manual_template:
            if self._mode_uses_vehicle(mode_text):
                self.annotator = create_combined_plate_annotator(v_p, p_p, conf, dev)
            else:
                self.annotator = create_plate_annotator(p_p, conf, dev)
        elif manual_vehicle_assist:
            self.annotator = create_vehicle_annotator(v_p, conf, dev)

        # The currently visible run stays active until the new autoannotation
        # run is committed. Resetting campaign Step 2 here clears the list in
        # the UI before the replacement annotations exist.

        if not manual_template:
            live_manual_bundle = self._get_current_campaign_manual_preview_bundle()
            self._campaign_auto_pre_run_snapshot = self._get_current_campaign_preview_snapshot()
            self._campaign_auto_pre_run_visible_state = {
                "annotations": copy.deepcopy(list(getattr(self, "current_annotations", []) or [])),
                "image_map": dict(getattr(self, "_preview_image_path_map", {}) or {}),
                "run_dir": str(getattr(self, "current_annotation_run_dir", "") or ""),
                "xml_path": str(getattr(self, "current_annotation_xml_path", "") or ""),
                "input_dir": str(getattr(self, "current_input_dir", "") or ""),
                "last_staging_run_dir": str(getattr(self, "last_staging_run_dir", "") or ""),
                "preview_index": getattr(self, "current_preview_index", None),
                "approved_filenames": set(self._get_preview_approved_filenames()),
                "campaign_pending_approved": set(getattr(self, "_campaign_pending_approved_filenames", set()) or set()),
                "hidden_project_approved": set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set()),
                "hidden_char_effective": set(getattr(self, "_campaign_hidden_char_effective_filenames", set()) or set()),
            }
            self._campaign_auto_pre_run_context = {
                "run_dir": str(getattr(self, "current_annotation_run_dir", "") or ""),
                "xml_path": str(getattr(self, "current_annotation_xml_path", "") or ""),
                "input_dir": str(getattr(self, "current_input_dir", "") or ""),
                "image_map": dict(getattr(self, "_preview_image_path_map", {}) or {}),
                "preview_index": getattr(self, "current_preview_index", None),
                "approved_filenames": set(self._get_preview_approved_filenames()),
                "campaign_pending_approved": set(getattr(self, "_campaign_pending_approved_filenames", set()) or set()),
            }
            persisted_overlay_bundle = dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {})
            self._campaign_auto_manual_overlay_bundle = (
                live_manual_bundle
                if live_manual_bundle
                else persisted_overlay_bundle
            )
            self._pending_preview_approved_filenames = set(
                self._get_preview_approved_filenames()
            ) | set(
                str(name or "").strip().lower()
                for name in set(getattr(self, "_campaign_pending_approved_filenames", set()) or set())
                if str(name or "").strip()
            )
            self._append_z2_trace(
                "start-snapshot",
                (
                    f"protected={len(dict(getattr(self, '_campaign_auto_manual_overlay_bundle', {}) or {}))} "
                    f"full={len(dict(getattr(self, '_campaign_auto_pre_run_snapshot', {}) or {}))} "
                    f"approved={len(set(getattr(self, '_pending_preview_approved_filenames', set()) or set()))} "
                    f"route={route} manual_template={int(bool(manual_template))}"
                ),
            )
        else:
            self._campaign_auto_manual_overlay_bundle = {}
            self._campaign_auto_pre_run_snapshot = {}
            self._campaign_auto_pre_run_visible_state = {}
            self._campaign_auto_pre_run_context = {}
            self._pending_preview_approved_filenames = set()

        self._set_preview_processing_overlay(
            not (
                manual_template
                and self._is_free_mode_session_context()
                and route == "manual"
                and manual_entry_mode == "new"
            ),
            title=(
                "Trwa przygotowanie autoanotacji"
                if not manual_template
                else "Trwa przygotowanie XML"
            ),
            details=(
                "Lista i podgląd pozostają widoczne, ale są zablokowane do końca bieżącego runu."
            ),
        )
        if not bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
            try:
                self._plate_auto_scope_selection_mode_active = False
                self._plate_auto_scope_modal_open = False
                self._plate_auto_scope_active_dialog = None
                self._plate_auto_scope_locked_widget_states = {}
            except Exception:
                pass
        preserve_existing_preview_during_auto = False
        try:
            visible_state = dict(getattr(self, "_campaign_auto_pre_run_visible_state", {}) or {})
            preserve_existing_preview_during_auto = bool(
                not manual_template
                and not self._is_free_mode_session_context()
                and list(visible_state.get("annotations") or [])
            )
        except Exception:
            preserve_existing_preview_during_auto = False
        if preserve_existing_preview_during_auto or (
            not manual_template
            and not self._is_free_mode_session_context()
        ):
            try:
                self._preview_draw_mode = False
                self._preview_draw_points = []
                self._preview_drag_state = None
                self._preview_pending_vertex_hit = None
                self._preview_delete_mode = False
                self._preview_delete_candidate_idx = None
                self._update_preview_edit_status(
                    "Autoanotacja działa w tle. Poprzednia lista pozostaje widoczna do czasu zakończenia nowego przebiegu."
                )
            except Exception:
                pass
        else:
            self._clear_preview_editor_state(clear_dirty=True)
        self._refresh_free_mode_workflow_ui()
        try:
            self._schedule_left_panel_scroll_to_widget(
                getattr(self, "workflow_start_section", None),
                delay_ms=0,
            )
        except Exception:
            pass

        logger.info("=" * 50)
        logger.info(
            "ROZPOCZĘTO PRZYGOTOWANIE XML DO RĘCZNEJ ANOTACJI TABLIC"
            if manual_template
            else "ROZPOCZĘTO AUTOANOTACJĘ OBRAZÓW (YOLO)"
        )
        logger.info("=" * 50)
        logger.info(
            (
                f"Tryb Z2: ręczna anotacja + auto-boxy pojazdów | obrazy: {effective_input_dir}"
                if manual_vehicle_assist
                else f"Tryb Z2: ręczna anotacja | obrazy: {effective_input_dir}"
            )
            if manual_template
            else f"Urządzenie Z2: {selected_device} -> runtime={dev}"
        )

        self._start_pre_progress_activity(
            "Inicjalizacja modelu i analiza pierwszego obrazu"
            if (manual_vehicle_assist or not manual_template)
            else "Przygotowywanie szablonu dla pierwszego obrazu"
        )

        worker_thread = threading.Thread(
            target=self._process_thread,
            args=(Path(effective_input_dir), safe_output_dir, manual_template, manual_vehicle_assist),
            daemon=True,
        )
        self._annotation_worker_thread = worker_thread

        def launch_worker():
            try:
                print("Z2 TRACE | before worker.start()", flush=True)
            except Exception:
                pass
            try:
                worker_thread.start()
            except Exception as e:
                logger.exception("Z2 worker: start() nie powiodlo sie")
                self._post_to_ui(lambda err=str(e): messagebox.showerror("Błąd Startu", err))
                return
            try:
                print("Z2 TRACE | after worker.start()", flush=True)
            except Exception:
                pass

        try:
            self.frame.after_idle(launch_worker)
        except Exception:
            launch_worker()

    except Exception as e:
        if getattr(self, "is_processing", False):
            self.is_processing = False
            if hasattr(self.app, "end_exclusive_operation"):
                self.app.end_exclusive_operation("z2.annotation.run")
            else:
                self.app.set_processing(False)
            try:
                self._set_preview_processing_overlay(False)
            except Exception:
                pass
            try:
                close_scope_progress = getattr(self, "_close_plate_auto_scope_progress_modal", None)
                if callable(close_scope_progress):
                    close_scope_progress()
            except Exception:
                pass
            try:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
            except Exception:
                pass
            try:
                self._refresh_free_mode_workflow_ui()
            except Exception:
                pass
        logger.error(f"Nie można wystartować: {e}")
        messagebox.showerror("Błąd Startu", str(e))

def _show_t02_approval_project_pool_summary(
    self,
    *,
    gate_label: str,
    target_dir: Path,
    iteration_num: int,
    previous_images: int,
    previous_plates: int,
    run_ok_images: int,
    run_ok_plates: int,
    added_images: int,
    added_plates: int,
    total_images: int,
    total_plates: int,
) -> None:
    """Show T02 as previous project pool + current control contribution."""
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    parent = getattr(self, "frame", None)
    panel_bg = palette.get("panel", "#252526")
    card_bg = palette.get("card", "#2d2d30")
    border = palette.get("border", "#3f3f46")
    fg = palette.get("fg", "#f3f4f6")
    muted = palette.get("muted", "#a1a1aa")
    success = palette.get("success", "#27ae60")
    warning = palette.get("warning", "#f39c12")
    accent = palette.get("accent", "#0e639c")
    try:
        added_images = max(0, int(added_images or 0))
        added_plates = max(0, int(added_plates or 0))
        total_images = max(0, int(total_images or 0))
        total_plates = max(0, int(total_plates or 0))
        previous_images = max(0, int(previous_images or 0))
        previous_plates = max(0, int(previous_plates or 0))
        run_ok_images = max(0, int(run_ok_images or 0))
        run_ok_plates = max(0, int(run_ok_plates or 0))
        iteration_num = max(1, int(iteration_num or 1))
    except Exception:
        pass
    skipped_images = max(0, int(run_ok_images or 0) - int(added_images or 0))
    skipped_plates = max(0, int(run_ok_plates or 0) - int(added_plates or 0))

    dialog = tk.Toplevel(parent)
    try:
        self.app.style_dialog_window(
            dialog,
            title=f"{gate_label} zatwierdzona",
            geometry="760x500",
            parent=parent,
        )
    except Exception:
        dialog.title(f"{gate_label} zatwierdzona")
        dialog.configure(bg=panel_bg)
    dialog.resizable(False, False)

    try:
        body = self.app._build_themed_dialog_surface(dialog, tone="success")
    except Exception:
        body = tk.Frame(dialog, bg=panel_bg, padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

    shell = tk.Frame(body, bg=panel_bg)
    shell.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

    tk.Label(
        shell,
        text=f"{gate_label}: kontrola AT zapisana do puli projektu",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 13, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))
    tk.Label(
        shell,
        text=(
            "Do projektu trafiaja tylko obrazy zatwierdzone w kontroli Z2 jako [OK]. "
            "Duplikaty nie zwiekszaja puli."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=700,
    ).pack(fill=tk.X, pady=(0, 14))

    table = tk.Frame(shell, bg=border, bd=0, highlightthickness=1, highlightbackground=border)
    table.pack(fill=tk.X, pady=(0, 12))
    columns = [
        ("Zakres", 24),
        ("Iteracja", 11),
        ("Obrazy [OK]", 13),
        ("Tablice", 10),
        ("Znaczenie", 25),
    ]
    for col, (_label, _width) in enumerate(columns):
        table.columnconfigure(col, weight=1 if col in {0, 4} else 0)

    def _cell(row: int, col: int, text: str, *, bg: str, color: str = fg, bold: bool = False, width: int = 12):
        label = tk.Label(
            table,
            text=str(text),
            bg=bg,
            fg=color,
            font=("Segoe UI", 9, "bold" if bold else "normal"),
            padx=10,
            pady=8,
            anchor="w" if col in {0, 4} else "center",
            width=width,
        )
        label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
        return label

    header_bg = blend_hex_colors(accent, panel_bg, 0.30)
    row_bg = card_bg
    alt_bg = blend_hex_colors(card_bg, panel_bg, 0.20)
    for col, (label, width) in enumerate(columns):
        _cell(0, col, label, bg=header_bg, color=fg, bold=True, width=width)

    rows = [
        (
            "W projekcie przed T02",
            f"do it{iteration_num - 1}" if iteration_num > 1 else "start",
            str(previous_images),
            str(previous_plates),
            "wczesniejsza pula projektu",
            muted,
        ),
        (
            "Dodane po kontroli T02",
            f"it{iteration_num}",
            f"+{added_images}",
            f"+{added_plates}",
            "nowy wklad po kontroli",
            success if added_images > 0 or added_plates > 0 else warning,
        ),
        (
            "Razem po zatwierdzeniu",
            "projekt",
            str(total_images),
            str(total_plates),
            "aktualna pula do dalszej pracy",
            success,
        ),
    ]
    for idx, row in enumerate(rows, start=1):
        bg = row_bg if idx % 2 else alt_bg
        tone = row[5]
        for col, value in enumerate(row[:5]):
            color = tone if col in {2, 3} else fg
            _cell(idx, col, value, bg=bg, color=color, bold=col in {0, 2, 3}, width=columns[col][1])

    note_text = (
        f"W kontroli Z2 zaznaczono {run_ok_images} obrazow [OK] / {run_ok_plates} tablic. "
        f"Do puli projektu dopisano {added_images} nowych obrazow / {added_plates} tablic."
    )
    if skipped_images > 0 or skipped_plates > 0:
        note_text += (
            f" Odsiano lub zaktualizowano istniejace pozycje: {skipped_images} obrazow / {skipped_plates} tablic."
        )
    tk.Label(
        shell,
        text=note_text,
        bg=blend_hex_colors(card_bg, panel_bg, 0.12),
        fg=fg,
        font=("Segoe UI", 9),
        padx=12,
        pady=10,
        anchor="w",
        justify=tk.LEFT,
        wraplength=700,
        highlightthickness=1,
        highlightbackground=border,
    ).pack(fill=tk.X, pady=(0, 10))

    tk.Label(
        shell,
        text=f"Run zrodlowy: {target_dir}",
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        wraplength=700,
        justify=tk.LEFT,
    ).pack(fill=tk.X, pady=(0, 12))

    footer = tk.Frame(shell, bg=panel_bg)
    footer.pack(fill=tk.X)
    tk.Button(
        footer,
        text="Zamknij podsumowanie",
        command=dialog.destroy,
        bg=blend_hex_colors(success, panel_bg, 0.20),
        fg=fg,
        activebackground=blend_hex_colors(success, panel_bg, 0.35),
        activeforeground=fg,
        relief=tk.FLAT,
        padx=18,
        pady=8,
        cursor="hand2",
    ).pack(side=tk.RIGHT)

    try:
        dialog.transient(parent)
        dialog.grab_set()
        dialog.update_idletasks()
        center = getattr(self.app, "_center_dialog_window", None)
        if callable(center):
            center(dialog, parent=parent, width=760, height=500)
        dialog.lift()
        dialog.focus_force()
        dialog.wait_window()
    except Exception:
        try:
            dialog.wait_window()
        except Exception:
            pass


def _approve_annotation_stage(self, *, _run_deferred: bool = False):
    """
    Domyka etap E2 i przenosi staging runu anotacji do katalogu docelowego projektu.
    """
    if not _run_deferred:
        if bool(getattr(self, "_z2_approval_in_progress", False)):
            try:
                self.app.update_status("Zatwierdzanie bramki jest juz w toku.", "warning")
            except Exception:
                pass
            return
        self._z2_approval_in_progress = True
        approval_owner = "z2.approve.stage"
        try:
            begin_exclusive = getattr(self.app, "try_begin_exclusive_operation", None)
            if callable(begin_exclusive):
                ok, busy_message = begin_exclusive(approval_owner, "Z2: zatwierdzanie bramki grafu")
                if not ok:
                    self._z2_approval_in_progress = False
                    return messagebox.showwarning("Aplikacja jest zajeta", busy_message)
                self._z2_approval_operation_owner = approval_owner
        except Exception:
            pass
        try:
            self.app.update_status("Zatwierdzam bramke grafu i zapisuje pule YOLO...", "info")
        except Exception:
            pass
        try:
            self.approve_btn.config(state=tk.DISABLED)
        except Exception:
            pass
        try:
            self.frame.winfo_toplevel().config(cursor="watch")
        except Exception:
            pass
        try:
            self.frame.update_idletasks()
        except Exception:
            pass

        def _run_approval():
            return _approve_annotation_stage(self, _run_deferred=True)

        try:
            self.frame.after(40, _run_approval)
        except Exception:
            _run_approval()
        return

    approval_perf_started = time.perf_counter()
    approval_perf_last = approval_perf_started
    approval_perf_phases: list[str] = []

    def _approval_mark(label: str) -> None:
        nonlocal approval_perf_last
        try:
            now = time.perf_counter()
            elapsed_ms = int((now - approval_perf_last) * 1000)
            total_ms = int((now - approval_perf_started) * 1000)
            approval_perf_phases.append(f"{label}={elapsed_ms}ms/{total_ms}ms")
            approval_perf_last = now
        except Exception:
            pass

    approval_perf_logged = False

    def _approval_log(result: str = "done") -> None:
        nonlocal approval_perf_logged
        if approval_perf_logged:
            return
        approval_perf_logged = True
        try:
            total_ms = int((time.perf_counter() - approval_perf_started) * 1000)
            logger.info(
                "[Z2 PERF] approve_annotation_stage total=%sms result=%s phases=[%s]",
                total_ms,
                str(result or "done"),
                ", ".join(approval_perf_phases) if approval_perf_phases else "no_slow_phase",
            )
        except Exception:
            pass

    try:
        from ..campaign_manager import CAMPAIGN
        import shutil

        if not self._ensure_preview_edits_saved("domkniecie etapu E2"):
            return
        _approval_mark("save_edits")

        if not CAMPAIGN.get_active_project_name():
            return messagebox.showwarning("Brak projektu", "Nie ma aktywnego projektu.")

        approval_context = self._get_campaign_step2_approval_context()
        _approval_mark("approval_context")
        staging_run = approval_context.get("run_dir")
        approval_source_kind = str(approval_context.get("source_kind") or "").strip().lower()
        approval_iteration_target = str(approval_context.get("iteration_target") or "").strip().lower()
        repair_mode = bool(approval_context.get("repair_mode"))
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        except Exception:
            graph_context = {}
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
        graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) or graph_gate_id
        graph_display_gate_id = graph_display_gate_id or graph_gate_id
        graph_close_gate_display_id = "T06"
        graph_gate_is_t05 = bool(graph_gate_id == "T04")

        if staging_run is None:
            try:
                approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            except Exception:
                approved_stats = {}
            approved_images = int(approved_stats.get("images", 0) or 0)
            approved_plates = int(approved_stats.get("plates", 0) or 0)
            return messagebox.showwarning(
                "Brak danych do domknięcia E2",
                (
                    "Nie znaleziono aktywnego runu anotacji Z2 z plikiem annotations.xml, który można domknąć.\n\n"
                    f"Zatwierdzone w projekcie: {approved_images} obrazów / {approved_plates} tablic.\n\n"
                    "Otwórz istniejący run tej iteracji albo wróć do właściwego kroku Z2 i zapisz anotacje przed domknięciem etapu."
                ),
            )

        staging_run = Path(staging_run)
        if not staging_run.exists():
            return messagebox.showerror("Brak folderu", f"Folder stagingu nie istnieje:\n{staging_run}")
        try:
            current_run_dir = self._resolve_safe_annotation_run_dir(
                getattr(self, "current_annotation_run_dir", None),
                require_xml=True,
            )
        except Exception:
            current_run_dir = None
        if current_run_dir is not None and self._paths_equivalent(current_run_dir, staging_run):
            try:
                approved_payload = set(self._get_preview_approved_filenames() or set())
                if not self._is_free_mode_session_context():
                    approved_payload |= {
                        str(name or "").strip().lower()
                        for name in set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
                        if str(name or "").strip()
                    }
                self._update_annotation_run_manifest(
                    staging_run,
                    approved_filenames=sorted(approved_payload),
                    **self._collect_preview_resume_manifest_fields(),
                )
            except Exception as exc:
                logger.debug(f"Nie udalo sie jawnie zapisac statusow OK przed zatwierdzeniem bramki Z2: {exc}")
        if approval_source_kind == "staging":
            staging_root = CAMPAIGN.get_staging_dir("auto_ann")
            auto_root = CAMPAIGN.get_dir("auto_ann")
            in_project_staging = bool(staging_root and self._path_is_within(staging_run, staging_root))
            in_project_auto = bool(auto_root and self._path_is_within(staging_run, auto_root))
            if in_project_auto and not in_project_staging:
                approval_source_kind = "existing"
            elif not in_project_staging:
                return messagebox.showerror(
                "Błędny staging",
                "Run anotacji do zatwierdzenia leży poza projektowym workspace stagingu.",
                )

        _approval_images_with_plates, approval_total_plates = self._get_run_plate_annotation_counts(staging_run)
        _approval_mark("run_annotation_counts")
        project_approved_images = 0
        project_approved_plates = 0
        run_approved_images = 0
        run_approved_plates = 0
        cumulative_plate_approved_images = 0
        cumulative_plate_approved_plates = 0
        cumulative_plate_gate_ready = False
        min_plate_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        min_char_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
        cumulative_char_gate_ready = False
        cumulative_char_approved_plates = 0
        if approval_iteration_target in {"plate", "char"}:
            try:
                approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
                project_approved_images = int(approved_stats.get("images", 0) or 0)
                project_approved_plates = int(approved_stats.get("plates", 0) or 0)
            except Exception:
                project_approved_images = 0
                project_approved_plates = 0
            try:
                run_approved_images, run_approved_plates = self._get_run_plate_approved_counts(staging_run)
            except Exception:
                run_approved_images, run_approved_plates = 0, 0
            cumulative_plate_approved_images = int(project_approved_images or 0) + int(run_approved_images or 0)
            cumulative_plate_approved_plates = int(project_approved_plates or 0) + int(run_approved_plates or 0)
            cumulative_char_approved_plates = int(cumulative_plate_approved_plates or 0)
            cumulative_plate_gate_ready = bool(
                approval_iteration_target == "plate"
                and int(cumulative_plate_approved_plates or 0) >= int(min_plate_approval_plates)
            )
            cumulative_char_gate_ready = bool(
                approval_iteration_target == "char"
                and int(cumulative_char_approved_plates or 0) >= int(min_char_approval_plates)
            )
        _approval_mark("approved_counts")

        if approval_total_plates <= 0 and not (
            (approval_iteration_target == "plate" and cumulative_plate_gate_ready)
            or (approval_iteration_target == "char" and cumulative_char_gate_ready)
        ):
            self._refresh_step2_action_states()
            return messagebox.showwarning(
                "Brak tablic do zatwierdzenia",
                (
                    "Ten run anotacji Z2 nie zawiera jeszcze ani jednej zapisanej tablicy 'plate'.\n\n"
                    f"Dodaj i zatwierdź co najmniej {min_plate_approval_plates if approval_iteration_target == 'plate' else min_char_approval_plates} tablic, a dopiero potem zatwierdź E2."
                    if bool(self._manual_xml_template_enabled())
                    else "Ten run anotacji Z2 nie zawiera jeszcze ani jednej zapisanej tablicy 'plate'.\n\n"
                    f"Popraw wynik albo dodaj tablice ręcznie. Do zatwierdzenia E2 potrzebujesz co najmniej {min_plate_approval_plates if approval_iteration_target == 'plate' else min_char_approval_plates} zatwierdzonych tablic."
                ),
            )

        if (
            approval_iteration_target == "char"
            and not cumulative_char_gate_ready
        ) or (
            approval_iteration_target == "plate"
            and not cumulative_plate_gate_ready
            and int(cumulative_plate_approved_plates or 0) < int(min_plate_approval_plates)
        ):
            self._refresh_step2_action_states()
            missing_char_plates = max(0, int(min_char_approval_plates) - int(cumulative_char_approved_plates or 0))
            missing_plate_plates = max(0, int(min_plate_approval_plates) - int(cumulative_plate_approved_plates or 0))
            return messagebox.showwarning(
                "Za mało zatwierdzonych tablic",
                (
                    f"Aby domknąć E2 w torze tablic i przejść do E4T, potrzebujesz co najmniej {min_plate_approval_plates} zatwierdzonych tablic.\n\n"
                    f"Projekt i bieżący run mają teraz {cumulative_plate_approved_plates} takich tablic.\n"
                    f"Brakuje jeszcze: {missing_plate_plates}.\n"
                    "Wróć do Z2, dodaj brakujące oznaczenia i dopiero wtedy zatwierdź etap."
                )
                if approval_iteration_target == "plate"
                else (
                    f"Aby przejść z tablic do znaków, potrzebujesz co najmniej {min_char_approval_plates} tablic na obrazach oznaczonych jako [OK].\n\n"
                    f"Projekt i bieżący run mają teraz {cumulative_char_approved_plates} takich tablic.\n"
                    f"Brakuje jeszcze: {missing_char_plates}.\n"
                    "Wróć do Z2, dodaj brakujące oznaczenia i dopiero wtedy przejdź dalej. "
                    "Zbyt mała liczba tablic nie pozwoli przygotować sensownego zbioru train/val dla znaków."
                ),
            )

        if approval_iteration_target != "plate":
            try:
                run_approved_images, run_approved_plates = self._get_run_plate_approved_counts(staging_run)
            except Exception:
                run_approved_images, run_approved_plates = 0, 0
        if approval_iteration_target == "plate" and (run_approved_images <= 0 or run_approved_plates <= 0):
            if project_approved_images <= 0 or project_approved_plates <= 0:
                self._refresh_step2_action_states()
                return messagebox.showwarning(
                    "Brak zatwierdzonych obrazów",
                    "Aby domknąć E2 w torze tablic, zaznacz na liście co najmniej jeden poprawny obraz i oznacz go jako OK.",
                )
        try:
            campaign_tab = self.app.tabs.get("campaign") if getattr(self.app, "tabs", None) else None
            confirm_no_progress = getattr(campaign_tab, "_confirm_step2_without_current_iteration_contribution", None)
            if callable(confirm_no_progress) and not confirm_no_progress(
                approval_context=approval_context,
                run_approved_images=int(run_approved_images or 0),
                run_approved_plates=int(run_approved_plates or 0),
            ):
                self._refresh_step2_action_states()
                try:
                    self.app.update_status(
                        "Zatwierdzenie E2 przerwane. Dodaj nowe zatwierdzone tablice albo świadomie zatwierdź E2 bez nowego wkładu.",
                        "warning",
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        _approval_mark("confirm_current_iteration")

        final_auto_dir = CAMPAIGN.get_dir("auto_ann")
        if final_auto_dir is None:
            return messagebox.showerror("Błąd", "Nie udało się ustalić katalogu docelowego autoanotacji dla projektu.")

        final_auto_dir = Path(final_auto_dir)
        final_auto_dir.mkdir(parents=True, exist_ok=True)
        if not self._path_is_within(final_auto_dir, CAMPAIGN.get_active_project_root_dir()):
            return messagebox.showerror(
                "Błędny katalog docelowy",
                "Docelowy katalog autoanotacji leży poza workspace aktywnego projektu.",
            )

        target_dir = staging_run
        staging_root = CAMPAIGN.get_staging_dir("auto_ann")
        in_project_staging = bool(staging_root and self._path_is_within(staging_run, staging_root))
        in_project_auto = bool(self._path_is_within(staging_run, final_auto_dir))

        if not in_project_staging and not in_project_auto:
            return messagebox.showerror(
                "Błędny run anotacji",
                "Run wybrany do dalszej pracy nie leży w projektowym katalogu autoanotacji ani w stagingu projektu.",
            )

        # Jeżeli run nadal jest w stagingu, przenosimy go do docelowego katalogu projektu.
        if in_project_staging:
            target_dir = final_auto_dir / staging_run.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(staging_run), str(target_dir))
        _approval_mark("resolve_target_run")
        self.current_annotation_run_dir = target_dir
        self.current_annotation_xml_path = target_dir / "annotations.xml"
        self.last_staging_run_dir = target_dir
        approved_set_result = {}
        run_approved_images, run_approved_plates = self._get_run_plate_approved_counts(target_dir)
        try:
            manifest = self._load_annotation_run_manifest(target_dir)
            if self._annotation_run_manifest_has_manual_value(manifest):
                self._remember_campaign_manual_plate_source(
                    run_dir=target_dir,
                    xml_path=target_dir / "annotations.xml",
                    input_dir=self.current_input_dir,
                )
        except Exception:
            pass
        try:
            if Path(str(self.plate_dataset_run_var.get() or "").strip()) == staging_run:
                self.plate_dataset_run_var.set(str(target_dir))
                self._refresh_plate_dataset_export_sources()
        except Exception:
            pass
        try:
            approved_set_result = self._promote_run_to_campaign_plate_approved_set(target_dir)
        except Exception as e:
            logger.debug(f"Nie udało się zaktualizować ApprovedSet tablic po domknięciu E2: {e}")
            approved_set_result = {"ok": False, "reason": "promotion_failed"}

        _approval_mark("promote_approved_set")

        approved_pool_total_images = int(project_approved_images or 0)
        approved_pool_total_plates = int(project_approved_plates or 0)
        approved_pool_added_images = 0
        approved_pool_added_plates = 0
        try:
            approved_after_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            approved_pool_total_images = int(approved_after_stats.get("images", approved_pool_total_images) or 0)
            approved_pool_total_plates = int(approved_after_stats.get("plates", approved_pool_total_plates) or 0)
            approved_pool_added_images = max(0, int(approved_pool_total_images) - int(project_approved_images or 0))
            approved_pool_added_plates = max(0, int(approved_pool_total_plates) - int(project_approved_plates or 0))
        except Exception:
            approved_pool_total_images = int(project_approved_images or 0) + int(run_approved_images or 0)
            approved_pool_total_plates = int(project_approved_plates or 0) + int(run_approved_plates or 0)
            approved_pool_added_images = int(run_approved_images or 0)
            approved_pool_added_plates = int(run_approved_plates or 0)

        if (
            approval_iteration_target == "plate"
            and graph_gate_is_t05
            and int(run_approved_images or 0) > 0
            and int(run_approved_plates or 0) > 0
            and not bool(dict(approved_set_result or {}).get("ok"))
        ):
            self._refresh_step2_action_states()
            try:
                self.app.update_status(
                    f"Zatwierdzenie {graph_display_gate_id} zatrzymane: nie udało się dopisać zatwierdzonych [OK] do puli YOLO projektu.",
                    "error",
                )
            except Exception:
                pass
            return messagebox.showerror(
                "Nie zapisano puli YOLO",
                (
                    f"Bramka {graph_display_gate_id} nie została zamknięta.\n\n"
                    "W aktywnej pracy Z2 są zatwierdzone obrazy [OK], ale nie udało się przenieść ich do puli YOLO projektu. "
                    "Zamknięcie bramki bez tego kroku groziłoby utratą wkładu tej iteracji.\n\n"
                    f"Pozostań w Z2, zapisz anotacje i spróbuj ponownie wrócić do grafu albo zatwierdzić {graph_display_gate_id}."
                ),
            )

        if approval_iteration_target == "plate" and graph_gate_is_t05 and bool(dict(approved_set_result or {}).get("ok")):
            try:
                now = datetime.datetime.now().isoformat(timespec="seconds")
                resolved_run_dir = Path(str(dict(approved_set_result or {}).get("run_dir") or target_dir))
                CAMPAIGN.upsert_iteration_state(
                    updates={
                        "t05_work_session": {
                            "active": False,
                            "state": "resolved",
                            "resolved_at": now,
                            "updated_at": now,
                            "run_dir": str(resolved_run_dir.resolve()),
                            "last_return_result": dict(approved_set_result or {}),
                        }
                    }
                )
            except Exception as exc:
                logger.debug(f"Nie udało się domknąć znacznika przerwanej pracy T05: {exc}")

        try:
            self._sync_campaign_iteration_artifact_registry(
                run_dir=target_dir,
                xml_path=target_dir / "annotations.xml",
                input_dir=self.current_input_dir,
                extra_updates={
                    "route_hints": {
                        "plate_entry_mode": "ready_run",
                        "char_entry_mode": (
                            "ready"
                            if int(cumulative_char_approved_plates or 0) >= int(min_char_approval_plates)
                            else "needs_more_tables"
                        ),
                        "char_ready": bool(
                            int(cumulative_char_approved_plates or 0) >= int(min_char_approval_plates)
                        ),
                        "char_has_source": bool(int(cumulative_char_approved_plates or 0) > 0),
                        "needs_more_tables": bool(
                            int(cumulative_char_approved_plates or 0) > 0
                            and int(cumulative_char_approved_plates or 0) < int(min_char_approval_plates)
                        ),
                        "images_with_plates": int(cumulative_plate_approved_images or 0),
                        "total_plates": int(cumulative_char_approved_plates or 0),
                    }
                },
            )
        except Exception:
            pass
        _approval_mark("artifact_registry")

        self._queue_free_mode_session_save()

        CAMPAIGN.approve_step2()
        next_step = 4 if CAMPAIGN.get_iteration_target() == "plate" else 3
        CAMPAIGN.set_current_step(next_step)
        _approval_mark("campaign_state")

        self.approve_btn.config(state=tk.DISABLED)
        self._refresh_step2_action_states()
        _approval_mark("z2_action_states")

        if 'campaign' in self.app.tabs:
            campaign_tab = self.app.tabs.get("campaign")
            try:
                campaign_tab.request_wizard_stage_focus(step_num=next_step)
            except Exception:
                pass
            previous_lightweight_refresh = bool(getattr(campaign_tab, "_project_open_lightweight_refresh", False))
            try:
                campaign_tab._project_open_lightweight_refresh = True
                campaign_tab._refresh_dashboard()
            finally:
                try:
                    campaign_tab._project_open_lightweight_refresh = previous_lightweight_refresh
                except Exception:
                    pass
            _approval_mark("campaign_refresh_initial")
            if next_step == 3 and (graph_display_gate_id == "T02" or graph_gate_id == "T02"):
                if repair_mode:
                    try:
                        CAMPAIGN.reset_step3_progress()
                        CAMPAIGN.set_step3_needs_rework()
                        CAMPAIGN.set_current_step(3)
                    except Exception as e:
                        logger.debug(f"Nie udalo sie zresetowac E3 po zatwierdzeniu T02: {e}")
                try:
                    self.app.update_status(
                        f"Bramka {graph_display_gate_id or 'T02'} zostala zatwierdzona. "
                        "Skontrolowane pozycje [OK] dopisano do puli projektu.",
                        "success",
                    )
                except Exception:
                    pass
                _show_t02_approval_project_pool_summary(
                    self,
                    gate_label=graph_display_gate_id or "T02",
                    target_dir=target_dir,
                    iteration_num=int(CAMPAIGN.get_current_iteration_num() or 1),
                    previous_images=project_approved_images,
                    previous_plates=project_approved_plates,
                    run_ok_images=run_approved_images,
                    run_ok_plates=run_approved_plates,
                    added_images=approved_pool_added_images,
                    added_plates=approved_pool_added_plates,
                    total_images=approved_pool_total_images,
                    total_plates=approved_pool_total_plates,
                )
                try:
                    campaign_tab = self.app.tabs.get("campaign")
                    if campaign_tab is not None:
                        try:
                            campaign_tab.request_wizard_stage_focus(step_num=3)
                        except Exception:
                            pass
                        campaign_tab._refresh_dashboard()
                    self.app.open_controlled_tab("campaign")
                    self.app.update_campaign_tab_access()
                except Exception as e:
                    logger.debug(f"Nie udalo sie wrocic do grafu po zatwierdzeniu T02: {e}")
                return
            if next_step == 3:
                if repair_mode:
                    try:
                        CAMPAIGN.reset_step3_progress()
                        CAMPAIGN.set_step3_needs_rework()
                        CAMPAIGN.set_current_step(3)
                    except Exception as e:
                        logger.debug(f"Nie udalo sie zresetowac E3 po naprawie źródła tablic: {e}")
                    self.app.update_status(
                        "Tablice zostały zatwierdzone. Wracam do etapu E3, aby przebudować dataset znaków.",
                        "info"
                    )
                    messagebox.showinfo(
                        "Źródło tablic zatwierdzone",
                        f"Źródło tablic zostało zatwierdzone.\n\nWracam do etapu E3. Tam przebudujesz dataset znaków; dopiero po poprawnym domknięciu E3 odblokuje się E4Z.\n\nRun źródłowy:\n{target_dir}"
                    )
                    try:
                        campaign_tab = self.app.tabs.get("campaign")
                        if campaign_tab is not None:
                            try:
                                campaign_tab.request_wizard_stage_focus(step_num=3)
                            except Exception:
                                pass
                            campaign_tab._refresh_dashboard()
                        self.app.open_controlled_tab("campaign")
                        self.app.update_campaign_tab_access()
                    except Exception as e:
                        logger.debug(f"Nie udalo sie wrocic do E3 po zatwierdzeniu źródła tablic: {e}")
                    return
                else:
                    self.app.update_status(
                        "Tablice zostały zatwierdzone jako źródło dla toru znaków. Wracam do etapu E3.",
                        "info"
                    )
                    messagebox.showinfo(
                        "Źródło tablic zatwierdzone",
                        f"Źródło tablic zostało zatwierdzone.\n\nWracam do etapu E3. Stamtąd przejdziesz dalej do Z3.\n\nRun źródłowy:\n{target_dir}"
                    )
                    try:
                        campaign_tab = self.app.tabs.get("campaign")
                        if campaign_tab is not None:
                            try:
                                campaign_tab.request_wizard_stage_focus(step_num=3)
                            except Exception:
                                pass
                            campaign_tab._refresh_dashboard()
                        self.app.open_controlled_tab("campaign")
                        self.app.update_campaign_tab_access()
                    except Exception:
                        pass
                    return

        next_step_label = "E4T / trening modelu tablic" if next_step == 4 else "E3 / tor znaków"
        if graph_gate_is_t05 and next_step == 4:
            self.app.update_status(
                f"Bramka {graph_display_gate_id} została zatwierdzona. Zatwierdzone anotacje tablic trafiły do puli projektu; bramkę {graph_close_gate_display_id} obsłużysz osobną decyzją w grafie.",
                "info",
            )
        else:
            self.app.update_status(
                "Domknięto etap E2 anotacji. Wyniki przeniesiono z katalogu stagingu do "
                f"2_auto_annotations projektu. Kolejny krok: {next_step_label}.",
                "info"
            )
        if approved_set_result.get("ok"):
            try:
                self.app.update_status(
                    f"ApprovedSet tablic zaktualizowany: {approved_set_result.get('total', 0)} obrazów w zbiorze projektu.",
                    "success",
                )
            except Exception:
                pass
        # po zatwierdzeniu wracamy do Wizarda
        try:
            campaign_tab = self.app.tabs.get("campaign")
            if campaign_tab is not None:
                try:
                    campaign_tab.request_wizard_stage_focus(step_num=next_step)
                except Exception:
                    pass
            self.app.open_controlled_tab("campaign")
            self.app.update_campaign_tab_access()
            _approval_mark("open_campaign")
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ wrĂłciÄ‡ do zakĹ‚adki Wizarda: {e}")

        _approval_mark("before_success_dialog")
        _approval_log("success")
        if graph_gate_is_t05 and next_step == 4:
            messagebox.showinfo(
                f"Bramka {graph_display_gate_id} zatwierdzona",
                (
                    f"Bramka {graph_display_gate_id} została zatwierdzona.\n\n"
                    "Zatwierdzone anotacje tablic zostały przyjęte do puli projektu. "
                    "Odblokowano E4T, czyli węzeł treningu modelu tablic.\n\n"
                    f"Bramka {graph_close_gate_display_id} zamknięcia iteracji pozostaje osobną decyzją: przygotuj trening w Z4 "
                    "albo świadomie zakończ iterację bez treningu."
                ),
            )
            try:
                self._campaign_graph_entry_context = {}
            except Exception:
                pass
        else:
            messagebox.showinfo(
                "Bramka zatwierdzona",
                (
                    "Zatwierdzone anotacje tablic zostały zapisane jako źródło dla dalszej pracy.\n\n"
                    f"Wyniki przeniesiono do:\n{target_dir}"
                ),
            )

    except Exception as e:
        _approval_log("error")
        messagebox.showerror("Błąd domknięcia E2", str(e))
    finally:
        try:
            _approval_log("finished")
        except Exception:
            pass
        try:
            self._z2_approval_in_progress = False
        except Exception:
            pass
        try:
            approval_owner = str(getattr(self, "_z2_approval_operation_owner", "") or "").strip()
            if approval_owner and hasattr(self.app, "end_exclusive_operation"):
                self.app.end_exclusive_operation(approval_owner)
            self._z2_approval_operation_owner = ""
        except Exception:
            pass
        try:
            self.frame.winfo_toplevel().config(cursor="")
        except Exception:
            pass

def _process_thread(
    self,
    in_dir: Path,
    base_out_dir: Path,
    manual_template: bool = False,
    manual_vehicle_assist: bool = False,
):
    success = False
    message = ""
    run_dir = None
    
    try:
        try:
            print("Z2 TRACE | entered _process_thread", flush=True)
        except Exception:
            pass
        try:
            print(f"Z2 TRACE | output_dir={base_out_dir}", flush=True)
        except Exception:
            pass
        self._post_to_ui(lambda: self._set_status_label_state("Skanowanie folderu obrazow...", "neutral"))
        try:
            print("Z2 TRACE | before count_images", flush=True)
        except Exception:
            pass
        total_images = count_images_in_directory(in_dir)
        try:
            print(f"Z2 TRACE | after count_images total={total_images}", flush=True)
        except Exception:
            pass
        if total_images == 0:
            self._post_to_ui(lambda: self._finish(False, "Brak obrazów we wskazanym folderze wejściowym."))
            return
        self._post_to_ui(lambda total=total_images: self._set_progress_counters(0, 0, total))

        annotator = getattr(self, "annotator", None)
        try:
            print(
                f"Z2 TRACE | annotator={type(annotator).__name__ if annotator is not None else 'None'}",
                flush=True,
            )
        except Exception:
            pass
        if annotator is not None:
            load_started_at = datetime.datetime.now()
            try:
                print("Z2 TRACE | before annotator.load_models()", flush=True)
            except Exception:
                pass
            load_ok, load_msg = annotator.load_models()
            try:
                print(f"Z2 TRACE | after annotator.load_models() ok={load_ok}", flush=True)
            except Exception:
                pass
            if not load_ok:
                message = f"Błąd silnika YOLO: {load_msg}"
                success = False
                return
            load_elapsed_s = max(
                0.0,
                (datetime.datetime.now() - load_started_at).total_seconds(),
            )
            uses_accelerator = str(getattr(annotator, "device", "") or "").strip().lower() != "cpu"
            first_image_hint = (
                " Pierwszy obraz może potrwać dłużej przez inicjalizację CUDA."
                if uses_accelerator
                else ""
            )
            self._post_to_ui(
                lambda total=total_images, load_elapsed_s=load_elapsed_s, first_image_hint=first_image_hint:
                    self._set_status_label_state(
                        f"Model załadowany ({load_elapsed_s:.1f}s). Rozpoczynam analizę 1/{total}.{first_image_hint}",
                        "neutral",
                    )
            )
            if annotator.is_stopped() or not self.is_processing:
                message = "Anulowano przez użytkownika."
                success = False
                return
            
        self.start_time = datetime.datetime.now()
        
        def prog_cb(current, total, filename, successful=0):
            if not self.is_processing: raise KeyboardInterrupt("Anulowano")
            pct = (current / total) * 100 if total > 0 else 0
            self._queue_progress_update(pct, current, total, filename, successful)
        
        if manual_template and not manual_vehicle_assist:
            annotations, report = self._build_manual_annotations_template(in_dir, prog_cb)
        elif manual_template:
            vehicle_annotations, _vehicle_report = self.annotator.process_directory(in_dir, prog_cb)
            annotations, report = self._build_manual_annotations_template_from_vehicle_seed(
                in_dir,
                vehicle_annotations,
            )
        else:
            annotations, report = self.annotator.process_directory(in_dir, prog_cb)

        if not manual_template:
            self._mark_auto_plate_origin_for_annotations(annotations)

        if ((not manual_template) or manual_vehicle_assist) and (self.annotator.is_stopped() or not self.is_processing):
            message = "Przetwarzanie przerwane przez użytkownika."
            success = False
            return

        if not manual_template and not self._is_free_mode_session_context():
            try:
                annotations, self._pending_source_image_map, merged_manual_count = self._merge_annotation_bundle_into_payload(
                    annotations,
                    dict(getattr(self, "_pending_source_image_map", {}) or {}),
                    getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {},
                )
                if merged_manual_count > 0:
                    logger.info(
                        f"Do wyniku runu auto Z2 dołączono {merged_manual_count} obrazów poprawionych ręcznie w tej iteracji."
                    )
            except Exception as e:
                logger.debug(f"Nie udało się dołączyć ręcznych poprawek do wyniku runu auto Z2: {e}")
        
        pending_image_path_map = dict(getattr(self, "_pending_source_image_map", {}) or {})

        run_dir = self._allocate_annotation_run_dir(base_out_dir)
        cvat_xml_path = run_dir / "annotations.xml"
        use_transactional_cache = bool(not manual_template)
        cache_xml_path = run_dir / "auto_annotation_cache.xml" if use_transactional_cache else None
        try:
            if self._is_free_mode_session_context():
                self._register_free_mode_branch_artifact(run_dir, artifact_type="owned_run_dirs")
        except Exception:
            pass

        logger.info(
            (
                "Zapisywanie annotations.xml z boxami pojazdów do ręcznej anotacji tablic..."
                if manual_vehicle_assist
                else "Zapisywanie pustego annotations.xml do ręcznej anotacji..."
            )
            if manual_template
            else "Zapisywanie cache autoanotacji (auto_annotation_cache.xml)..."
        )
        export_only_successful = bool(not manual_template and self._is_free_mode_session_context())
        export_xml_path = cache_xml_path if cache_xml_path is not None else cvat_xml_path
        CVATExporter().export(
            annotations,
            export_xml_path,
            only_successful=export_only_successful,
        )

        logger.info("Generowanie raportu statystycznego...")
        ReportGenerator.generate_text_report(report, run_dir / "report.txt")

        try:
            self._write_annotation_run_manifest(run_dir, in_dir)
        except Exception as e:
            logger.debug(f"Nie udało się zapisać manifestu runu Z2: {e}")
        try:
            manual_overlay_filenames = sorted(
                str(name or "").strip()
                for name in dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {}).keys()
                if str(name or "").strip()
            )
            if manual_overlay_filenames:
                self._update_annotation_run_manifest(
                    run_dir,
                    has_manual_edits=True,
                    last_manual_edit_at=datetime.datetime.now().isoformat(timespec="seconds"),
                    last_manual_edit_kind="auto_overlay_merge",
                    manual_touched_filenames=manual_overlay_filenames,
                )
        except Exception as e:
            logger.debug(f"Nie udało się dopisać ręcznych poprawek do manifestu runu auto Z2: {e}")
        try:
            approved_filenames_snapshot = sorted(
                str(name or "").strip().lower()
                for name in set(getattr(self, "_pending_preview_approved_filenames", set()) or set())
                if str(name or "").strip()
            )
            self._append_z2_trace(
                "run-manifest-approved",
                f"run={run_dir} count={len(approved_filenames_snapshot)} sample={approved_filenames_snapshot[:8]}",
            )
            self._update_annotation_run_manifest(
                run_dir,
                approved_filenames=approved_filenames_snapshot,
            )
        except Exception as e:
            logger.debug(f"Nie udało się dopisać zatwierdzonych pozycji do manifestu runu auto Z2: {e}")

        if cache_xml_path is not None:
            try:
                self._update_annotation_run_manifest(
                    run_dir,
                    auto_annotation_cache=str(cache_xml_path.name),
                    transaction_state="cache_ready",
                )
            except Exception as e:
                logger.debug(f"Nie udało się dopisać cache autoanotacji do manifestu runu Z2: {e}")
            logger.info("Cache autoanotacji gotowy. Utrwalam wynik jako annotations.xml...")
            shutil.copy2(cache_xml_path, cvat_xml_path)
            try:
                self._update_annotation_run_manifest(
                    run_dir,
                    transaction_state="committed",
                    committed_annotation_xml=str(cvat_xml_path.name),
                )
            except Exception as e:
                logger.debug(f"Nie udało się oznaczyć commitu autoanotacji w manifeście Z2: {e}")

        self.current_annotations = annotations
        self.current_input_dir = in_dir
        self._preview_image_path_map = pending_image_path_map
        self.current_annotation_run_dir = run_dir
        self.current_annotation_xml_path = cvat_xml_path

        elapsed = format_duration((datetime.datetime.now() - self.start_time).total_seconds())

        # Zachowaj ścieżkę do ostatniego runu w stagingu.
        self.last_staging_run_dir = run_dir

        message = (
            f"Przygotowano XML do ręcznej anotacji: {run_dir.name} (w czasie {elapsed})"
            if manual_template
            else f"Zakończono! Zapisano do: {run_dir.name} (w czasie {elapsed})"
        )
        success = True

        try:
            self._mark_annotation_run_completed(
                run_dir,
                annotations,
                manual_template=manual_template,
                report=report,
            )
        except Exception as e:
            logger.debug(f"Nie udało się oznaczyć runu Z2 jako zakończonego: {e}")

        try:
            self._restore_campaign_step2_generated_from_run(run_dir)
        except Exception as e:
            logger.debug(f"Nie udało się przywrócić stanu Kroku 2 z gotowego runu Z2: {e}")

        try:
            self._post_to_ui(
                lambda run_dir=run_dir, manual_template=manual_template: self._finalize_successful_annotation_run_ui(
                    run_dir,
                    manual_template=manual_template,
                )
            )
        except Exception as e:
            logger.debug(f"Nie udało się zaplanować odświeżenia UI po zakończeniu Z2: {e}")

        logger.info(f"[OK] {message}")
        
    except KeyboardInterrupt:
        message = "Anulowano przez użytkownika."
        try:
            if run_dir is not None:
                self._update_annotation_run_manifest(
                    run_dir,
                    run_status="cancelled",
                    transaction_state="cancelled",
                    last_error=message,
                )
        except Exception:
            pass
        success = False
    except Exception as e:
        message = f"Krytyczny błąd: {e}"
        try:
            if run_dir is not None:
                self._update_annotation_run_manifest(
                    run_dir,
                    run_status="failed",
                    transaction_state="aborted",
                    last_error=str(e),
                )
        except Exception:
            pass
        success = False
    finally:
        try:
            self._post_to_ui(lambda: self._finish(success, message))
        except Exception:
            pass

def _finish(self, success, msg):
    stop_requested = bool(getattr(self, "_annotation_stop_requested", False))
    self.is_processing = False
    self._annotation_stop_requested = False
    self._progress_update_seen = False
    self._cancel_pre_progress_activity()
    with self._progress_update_lock:
        self._pending_progress_update = None
        self._progress_update_flush_queued = False
    try:
        self._set_preview_processing_overlay(False)
    except Exception:
        pass
    try:
        close_scope_progress = getattr(self, "_close_plate_auto_scope_progress_modal", None)
        if callable(close_scope_progress):
            close_scope_progress()
    except Exception:
        pass
    try:
        self._plate_auto_scope_selection_mode_active = False
        self._plate_auto_scope_modal_open = False
        self._plate_auto_scope_active_dialog = None
        self._plate_auto_scope_locked_widget_states = {}
    except Exception:
        pass
    if hasattr(self.app, "end_exclusive_operation"):
        self.app.end_exclusive_operation("z2.annotation.run")
    else:
        self.app.set_processing(False)
    self.start_btn.config(state=tk.NORMAL)
    self.stop_btn.config(state=tk.DISABLED)
    try:
        self._refresh_step2_action_states(lightweight=False)
    except Exception:
        pass
    try:
        if self._is_free_mode_session_context():
            export_state = tk.NORMAL if self._is_z2_free_export_choice_available() else tk.DISABLED
            self.export_plate_dataset_btn.configure(
                text="EKSPORT",
                command=self._start_z2_export_choice_flow,
                state=export_state,
            )
        else:
            export_state = (
                tk.NORMAL
                if self._is_plate_dataset_export_allowed_for_current_selection()
                else tk.DISABLED
            )
            self.export_plate_dataset_btn.configure(
                text="EKSPORTUJ DATASET",
                command=self._start_z2_export_choice_flow,
                state=export_state,
            )
        self._set_plate_annotation_export_button_state()
    except Exception:
        self.export_plate_dataset_btn.config(state=tk.DISABLED)
    self.progress.configure(value=(100 if success else 0))
    if success:
        self._last_run_progress_visible = True
        total = len(getattr(self, "current_annotations", []) or [])
        if getattr(self, "_current_run_manual_template", False):
            successful = total
        else:
            successful = sum(1 for ann in (self.current_annotations or []) if getattr(ann, "is_successful", False))
        self._set_progress_counters(successful, total, total)
    else:
        self._last_run_progress_visible = False
        self._set_progress_counters(0, 0, 0)

    if success:
        manual_template_success = bool(getattr(self, "_current_run_manual_template", False))
        manual_template_free_mode_new_run = bool(
            manual_template_success
            and self._is_free_mode_session_context()
            and self._get_workflow_route() == "manual"
            and self._get_manual_entry_mode() == "new"
        )
        hold_manual_template_on_start = bool(manual_template_free_mode_new_run)
        self._last_completed_workflow_route = self._get_workflow_route() or (
            "manual" if manual_template_success else "auto"
        )
        self._manual_review_active = bool(manual_template_success and not hold_manual_template_on_start)
        self._manual_review_from_auto = False
        self._manual_review_origin_route = (
            "manual" if manual_template_success else "auto"
        )
        self._manual_review_export_ready = False
        self._manual_template_ready_for_review = bool(hold_manual_template_on_start)
        if self._is_free_mode_session_context():
            if hold_manual_template_on_start:
                self.free_mode_screen_var.set("workflow")
                self.workflow_step_var.set("manual_start")
            elif manual_template_success:
                self.free_mode_screen_var.set("manual_review")
                self.workflow_step_var.set("manual_start")
            else:
                self.free_mode_screen_var.set(
                    "manual_review" if self._manual_review_active else "workflow"
                )
                if not self._manual_review_active:
                    self.workflow_step_var.set("auto_start")
        self._set_status_label_state("Zakonczono pomyslnie!", "success")
        next_steps = self._build_annotation_success_next_steps()
        self._set_post_annotation_hint(next_steps, "success")

        try:
            from ..campaign_manager import CAMPAIGN
            if CAMPAIGN.get_active_project_name() and CAMPAIGN.get_current_step() == 2:
                staging_run = getattr(self, "last_staging_run_dir", None)
                _staging_images_with_plates, staging_total_plates = self._get_run_plate_annotation_counts(staging_run)
                if staging_run is not None:
                    CAMPAIGN.set_step2_generated(str(staging_run))
                if staging_total_plates > 0:
                    self.approve_btn.config(state=tk.NORMAL)
                if "campaign" in self.app.tabs:
                    self.app.tabs["campaign"]._refresh_dashboard()
        except Exception as e:
            logger.debug(f"Nie udalo sie zaktualizowac stanu kroku 2: {e}")

        self._refresh_step2_action_states()
        self._refresh_left_panel_route_copy()
        self._refresh_free_mode_workflow_ui()
        if manual_template_free_mode_new_run:
            try:
                self._schedule_left_panel_scroll_to_widget(
                    getattr(self, "workflow_start_section", None)
                    or getattr(self, "actions_section", None),
                    delay_ms=80,
                )
            except Exception:
                pass
        if not self._is_free_mode_session_context():
            self._refresh_step2_action_states()
        if self._is_free_mode_session_context():
            try:
                self.app.update_status(f"{msg} {next_steps}", "success")
            except Exception:
                pass
            if manual_template_free_mode_new_run:
                try:
                    parent_window = self.frame.winfo_toplevel()
                except Exception:
                    parent_window = getattr(self.app, "root", None)

                def _show_manual_xml_success_dialog():
                    messagebox.showinfo(
                        "XML anotacji utworzony",
                        (
                            "Utworzono nowy run ręcznej anotacji Z2 oraz plik annotations.xml.\n\n"
                            "Etap wejścia jest gotowy. Kliknij Dalej w karcie, aby przejść do kroku "
                            "„Korekta i decyzja po anotacji ręcznej”. Tam wykonasz właściwą pracę na tablicach.\n\n"
                            "Warunek dalszych akcji: narysuj lub popraw ramkę/poligon tablicy na co najmniej "
                            "jednym obrazie, zapisz anotację i nadaj temu obrazowi status [OK]. Dopiero wtedy "
                            "aktywują się akcje „Wyodrębnij tablice” oraz „Eksport”."
                        ),
                        parent=parent_window,
                    )

                try:
                    self.frame.after_idle(_show_manual_xml_success_dialog)
                except Exception:
                    _show_manual_xml_success_dialog()
            elif not manual_template_success:
                try:
                    self.frame.after_idle(
                        lambda next_steps=next_steps: self._show_auto_annotation_success_dialog(next_steps)
                    )
                except Exception:
                    self._show_auto_annotation_success_dialog(next_steps)
        else:
            try:
                self.app.update_status(f"{msg} {next_steps}", "success")
            except Exception:
                pass
        try:
            self._schedule_preview_list_population_consistency_check(
                reason="annotation-finish-success",
                delay_ms=180,
                render_current=False,
            )
        except Exception:
            pass
    else:
        msg_text = str(msg or "").strip()
        lowered_msg = msg_text.lower()
        cancellation_tokens = (
            "anulowano",
            "przerwano przez uzytkownika",
            "przerwano przez użytkownika",
            "przerwane przez uzytkownika",
            "przerwane przez użytkownika",
            "zatrzymano",
            "zatrzymana",
            "zatrzymane",
        )
        cancelled_by_user = bool(
            stop_requested
            or any(token in lowered_msg for token in cancellation_tokens)
        )
        has_pre_run_state = bool(
            dict(getattr(self, "_campaign_auto_pre_run_visible_state", {}) or {})
            or dict(getattr(self, "_campaign_auto_pre_run_snapshot", {}) or {})
            or dict(getattr(self, "_campaign_auto_pre_run_context", {}) or {})
        )
        restored_previous_preview = bool(has_pre_run_state and self._restore_pre_run_preview_snapshot())
        self._set_status_label_state(
            "Autoanotacja zatrzymana" if cancelled_by_user else "Przerwano / Blad",
            "warning" if cancelled_by_user else "error",
        )
        self._manual_review_active = False
        self._manual_review_from_auto = False
        self._manual_review_export_ready = False
        if self._is_free_mode_session_context() and not restored_previous_preview:
            self.free_mode_screen_var.set("workflow" if self._get_workflow_route() else "route_choice")
        if restored_previous_preview:
            self._set_post_annotation_hint(
                (
                    "Przerwano bieżący run autoanotacji. Przywrócono poprzednią listę wyników Z2, "
                    "więc możesz wybrać inny model albo uruchomić próbę ponownie."
                )
                if cancelled_by_user
                else (
                    "Bieżący run autoanotacji nie zakończył się sukcesem. Przywrócono poprzednią "
                    "listę wyników Z2, żeby nie utracić istniejących znaczników."
                ),
                "warning",
            )
        else:
            self._set_post_annotation_hint(str(msg or "").strip(), "error")
        self._refresh_step2_action_states()
        self._refresh_left_panel_route_copy()
        self._refresh_free_mode_workflow_ui()
        if not self._is_free_mode_session_context():
            self._refresh_step2_action_states()
        try:
            parent_window = self.frame.winfo_toplevel()
        except Exception:
            parent_window = getattr(self.app, "root", None)

        def _show_error_dialog():
            if cancelled_by_user:
                messagebox.showwarning("Zatrzymano", msg_text or "Autoanotacja została zatrzymana.", parent=parent_window)
            else:
                messagebox.showerror("Zatrzymano", msg, parent=parent_window)

        try:
            self.frame.after_idle(_show_error_dialog)
        except Exception:
            _show_error_dialog()
        finally:
            self._campaign_auto_pre_run_snapshot = {}
            self._campaign_auto_pre_run_visible_state = {}
            self._campaign_auto_pre_run_context = {}
            self._pending_preview_approved_filenames = set()

def _switch_annotation_input_dir(self, input_dir: Path, *, show_hint: bool = True) -> bool:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        messagebox.showerror("Brak obrazów", "Wybrany folder obrazów nie istnieje.")
        return False

    free_mode_context = bool(self._is_free_mode_session_context())
    current_route = self._get_workflow_route()
    current_manual_entry_mode = self._get_manual_entry_mode()
    current_auto_vehicle_choice = self._get_auto_vehicle_choice()
    current_step_before_switch = self._coerce_workflow_step()
    lightweight_path_only = bool(
        free_mode_context
        and (
            (
                current_route == "auto"
                and current_step_before_switch in {"auto_input", "auto_start"}
                and bool(getattr(self, "_auto_route_settings_pending", False))
            )
            or (
                current_route == "manual"
                and current_manual_entry_mode == "new"
                and current_step_before_switch in {"manual_entry", "manual_input", "manual_start"}
                and not bool(getattr(self, "_manual_template_ready_for_review", False))
            )
        )
    )

    if lightweight_path_only:
        resolved_input_dir = input_dir
    else:
        resolved_input_dir = self._resolve_annotation_input_images_dir(input_dir)
        if resolved_input_dir is None:
            messagebox.showerror(
                "Brak obrazów",
                "Wybrany folder nie zawiera obsługiwanych obrazów wejściowych do Z2.",
            )
            return False

    if not self._ensure_preview_edits_saved("zmiana puli obrazow Z2"):
        return False

    defer_heavy_source_refresh = bool(free_mode_context and current_route == "auto")

    if free_mode_context:
        self._clear_active_annotation_run_context(preserve_input_dir=False)
        self.free_mode_screen_var.set("workflow")
    else:
        self._reset_campaign_runtime_state(input_dir)

    self._set_input_dir_path_only(input_dir, resolved_input_dir=resolved_input_dir)

    if not free_mode_context:
        self._set_campaign_paths_lock_state(True)

    if not defer_heavy_source_refresh:
        self._refresh_plate_dataset_export_sources()
        self._refresh_manual_plate_stage_ui()

    if show_hint:
        if self._is_manual_plate_stage_input(resolved_input_dir):
            self._set_post_annotation_hint(
                f"Stage został ustawiony jako nowa pula obrazów Z2. Kliknij {self._get_step2_start_action_reference()}, aby utworzyć kolejny run anotacji Z2 dla tych zdjęć.",
                "success",
            )
        else:
            self._set_post_annotation_hint("")

    if current_route == "auto":
        try:
            self._set_workflow_route_state("auto", campaign_context=(not free_mode_context))
            self.manual_xml_template_var.set(False)
            self._set_auto_vehicle_choice_state(current_auto_vehicle_choice, campaign_context=(not free_mode_context))
            self._auto_route_settings_pending = True
        except Exception:
            pass

        if free_mode_context:
            self._set_workflow_step("auto_input", refresh_detection_ui=True)
            return True

        if current_step_before_switch == "auto_plate_model":
            next_step = "auto_plate_model"
        elif current_step_before_switch == "auto_conf":
            next_step = "auto_conf"
        elif current_step_before_switch == "auto_vehicle_choice":
            next_step = "auto_vehicle_choice"
        elif current_step_before_switch == "auto_vehicle_model" and current_auto_vehicle_choice == "use":
            next_step = "auto_vehicle_model"
        else:
            next_step = "auto_start"
        self._set_workflow_step(next_step)
        return True

    if current_route == "manual":
        try:
            self._set_workflow_route_state("manual", campaign_context=(not free_mode_context))
            self._set_manual_entry_mode_state(current_manual_entry_mode, campaign_context=(not free_mode_context))
            self.manual_xml_template_var.set(current_manual_entry_mode == "new")
            if current_manual_entry_mode != "new":
                self.manual_vehicle_assist_var.set(False)
        except Exception:
            pass

        if current_manual_entry_mode == "new":
            if free_mode_context:
                self._set_workflow_step("manual_input", refresh_detection_ui=True)
                return True
            self._set_workflow_step("manual_start")
        else:
            self._set_workflow_step("manual_history")
        return True

    if free_mode_context:
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()

    self._queue_free_mode_session_save()
    return True

def _go_to_next_workflow_step(self):
    if self._is_plate_auto_scope_modal_blocking_actions():
        return
    if self._is_free_mode_session_context():
        screen = self._coerce_free_mode_screen()
        if screen == "auto_summary":
            self._start_z2_export_choice_flow()
            return
        if screen == "manual_review":
            if self._get_workflow_route() == "manual":
                self._start_z2_export_choice_flow()
                return
            self._open_step3_from_z2_annotation_source()
            return
        if screen == "export":
            if self._dataset_export_completed:
                route = self._get_workflow_route()
                if route == "auto":
                    dataset_path = self._resolve_latest_exported_plate_dataset_dir()
                    choice = self._prompt_post_z2_export_completion_action(dataset_path)
                    if choice == "training":
                        if self._open_step4_training_from_z2_export(dataset_path):
                            return
                    elif choice == "finish":
                        self._clear_free_mode_route_selection()
                        return
                    return
                self._clear_free_mode_route_selection()
            else:
                self._start_z2_export_choice_flow()
            return

    if self._dataset_export_completed and self._is_free_mode_session_context():
        self._clear_free_mode_route_selection()
        return

    current = self._coerce_workflow_step()
    route = self._get_workflow_route()
    if (
        self._is_free_mode_session_context()
        and route == "auto"
        and current == "auto_start"
        and self._get_preferred_annotation_run_dir(require_xml=True) is not None
        and self._is_z2_auto_flow_completed(route=route)
    ):
        self.free_mode_screen_var.set("auto_summary")
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()
        self._scroll_left_panel_to_widget(
            getattr(self, "followup_section", None)
            or getattr(self, "actions_section", None)
        )
        self._queue_free_mode_session_save()
        return
    manual_entry_mode = self._get_manual_entry_mode()
    if route == "manual" and current == "manual_entry" and manual_entry_mode == "new":
        self._clear_active_annotation_run_context(preserve_input_dir=True)
    if self._manual_review_active and route == "manual":
        if current == "manual_history":
            self._jump_to_export_section()
            return
        if current == "manual_entry" and manual_entry_mode == "import":
            self._jump_to_export_section()
            return
    if route == "manual" and current == "manual_entry" and manual_entry_mode == "import":
        self._import_or_open_manual_review_run_from_dialog()
        return
    if route == "manual" and current == "manual_history":
        self._open_selected_manual_review_history_run()
        return
    if (
        self._is_free_mode_session_context()
        and route == "manual"
        and current == "manual_start"
        and manual_entry_mode == "new"
        and bool(getattr(self, "_manual_template_ready_for_review", False))
    ):
        target_run_dir = self._get_active_annotation_run_dir(require_xml=True)
        self._manual_template_ready_for_review = False
        self._manual_review_active = True
        self._manual_review_from_auto = False
        self._manual_review_origin_route = "manual"
        self.free_mode_screen_var.set("manual_review")
        if target_run_dir is not None:
            restored_preview = self._restore_preview_from_annotation_run(target_run_dir)
            if not restored_preview:
                self._load_plate_dataset_context_from_run(target_run_dir, force_images_update=True)
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()
        self._scroll_left_panel_to_widget(
            getattr(self, "manual_stage_section", None)
            or getattr(self, "actions_section", None)
        )
        self._queue_free_mode_session_save()
        return
    steps = self._get_current_workflow_steps()
    if current not in steps or not self._is_workflow_step_complete(current):
        return
    idx = steps.index(current)
    if idx >= len(steps) - 1:
        return
    self._set_workflow_step(steps[idx + 1])
