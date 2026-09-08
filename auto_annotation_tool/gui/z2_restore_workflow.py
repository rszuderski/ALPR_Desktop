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
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    campaign_gate_id_for_edge,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
from .z2_export_workflow import (
    _start_plate_dataset_export,
    _prompt_z2_export_choice,
    _start_plate_annotation_package_export,
    _prompt_plate_annotation_package_export_options,
    _refresh_plate_dataset_export_sources,
)

NAV_BUTTON_WIDTH = 18
YOLO = None


def _annotation_run_uses_scoped_input(manifest: dict | None) -> bool:
    """Return True when the run processed a temporary subset, not the full source."""
    if not isinstance(manifest, dict):
        return False
    try:
        scope_count = int(manifest.get("input_scope_count", 0) or 0)
    except Exception:
        scope_count = 0
    scope_mode = str(manifest.get("input_scope_mode") or "").strip().lower()
    if scope_count > 0 and scope_mode not in {"", "all", "full"}:
        return True

    input_dir = str(manifest.get("input_dir") or "").strip()
    source_dir = str(
        manifest.get("source_input_dir")
        or manifest.get("imported_source_input_dir")
        or ""
    ).strip()
    if not input_dir or not source_dir:
        return False
    try:
        return Path(input_dir).resolve() != Path(source_dir).resolve()
    except Exception:
        return input_dir.rstrip("\\/").lower() != source_dir.rstrip("\\/").lower()


def _annotation_run_input_scope_filenames(manifest: dict | None) -> set[str]:
    """Return the intended input-scope filenames when the campaign knows them."""
    names: set[str] = set()
    expected_count = 0
    if isinstance(manifest, dict):
        for count_key in (
            "input_scope_count",
            "source_plan_total_count",
            "selected_count",
            "image_count",
        ):
            try:
                expected_count = int(manifest.get(count_key, 0) or 0)
            except Exception:
                expected_count = 0
            if expected_count > 0:
                break
        for key in (
            "input_scope_filenames",
            "source_manifest_filenames",
            "selected_filenames",
        ):
            raw_items = manifest.get(key)
            if not isinstance(raw_items, (list, tuple, set)):
                continue
            for item in raw_items:
                safe_name = str(item or "").strip()
                if safe_name:
                    names.add(Path(safe_name).name.strip().lower())
            if names:
                return {name for name in names if name}

    try:
        ingest_manifest = CAMPAIGN.load_ingest_manifest() or {}
    except Exception:
        ingest_manifest = {}
    try:
        ingest_count = int((ingest_manifest or {}).get("selected_count", 0) or 0)
    except Exception:
        ingest_count = 0
    if expected_count > 0 and ingest_count > 0 and ingest_count != expected_count:
        return set()
    for item in list((ingest_manifest or {}).get("selected_images") or []):
        if not isinstance(item, dict):
            continue
        safe_name = str(item.get("name", "") or "").strip()
        if not safe_name:
            for key in ("target_path", "source_path", "iteration_target_path"):
                candidate = str(item.get(key, "") or "").strip()
                if candidate:
                    safe_name = Path(candidate).name.strip()
                    break
        if safe_name:
            names.add(Path(safe_name).name.strip().lower())
    return {name for name in names if name}


def _is_t02_at_review_restore_context(self) -> bool:
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    try:
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    return bool(
        str(graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or str(graph_gate_id or "").strip().upper() == "T02"
    )


def _sync_t06_approved_run_before_restore_filter(self, run_dir: Path | None, approved_filenames: set[str]) -> None:
    if self._is_free_mode_session_context() or not approved_filenames:
        return
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    if graph_gate_id != "T05":
        return
    try:
        normalized_approved = {
            CAMPAIGN._normalize_image_set_name(name)
            for name in set(approved_filenames or set())
            if CAMPAIGN._normalize_image_set_name(name)
        }
    except Exception:
        normalized_approved = {
            str(name or "").strip().lower()
            for name in set(approved_filenames or set())
            if str(name or "").strip()
        }
    if not normalized_approved:
        return
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        project_name = ""
    try:
        existing_project_names = {
            CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
            for entry in list(CAMPAIGN.list_plate_approved_entries(project_name or None) or [])
            if isinstance(entry, dict) and CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
        }
    except Exception:
        existing_project_names = set()
    if existing_project_names and normalized_approved.issubset(existing_project_names):
        try:
            logger.info(
                "[Z2 PERF] t06_restore_sync skipped already_project_source approved=%s run=%s",
                len(normalized_approved),
                run_dir,
            )
        except Exception:
            pass
        return
    try:
        self._sync_campaign_char_repair_approved_run_to_project_source(
            run_dir,
            refresh_effective_source=False,
            reason="pre_restore_filter",
        )
    except Exception as exc:
        logger.debug(f"Nie udalo sie zsynchronizowac T06 przed filtrowaniem listy Z2: {exc}")


def _is_t06_interrupted_restore_context(self, run_dir: Path | None = None) -> bool:
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    if graph_gate_id != "T05":
        return False

    resume_flag = str(graph_context.get("resume_interrupted_t06") or "").strip().lower()
    if resume_flag in {"1", "true", "yes", "tak"}:
        return True

    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
        session = dict(iteration_state.get("t06_work_session") or {})
    except Exception:
        session = {}
    session_state = str(session.get("state") or "").strip().lower()
    if session_state in {"resolved", "abandoned", "closed", "complete", "completed"}:
        return False
    session_active = bool(session.get("active")) or session_state in {"active", "started", "interrupted", "dirty"}
    if not session_active:
        return False

    session_run = str(session.get("run_dir") or "").strip()
    if run_dir is not None and session_run:
        try:
            return bool(self._paths_equivalent(Path(session_run), Path(run_dir)))
        except Exception:
            try:
                return str(Path(session_run).resolve()).lower() == str(Path(run_dir).resolve()).lower()
            except Exception:
                return session_run.rstrip("\\/").lower() == str(run_dir).rstrip("\\/").lower()
    return True


def _keep_pending_t06_approvals_visible_for_restore(
    self,
    run_dir: Path | None,
    annotations: list,
    run_approved_filenames: set[str],
    hidden_char_effective_filenames: set[str],
    hidden_char_effective_count: int,
) -> tuple[set[str], int]:
    if not _is_t06_interrupted_restore_context(self, run_dir):
        return hidden_char_effective_filenames, hidden_char_effective_count
    try:
        project_approved_filenames = set(self._get_campaign_plate_approved_filenames() or set())
    except Exception:
        project_approved_filenames = set()
    run_only_approved = set(run_approved_filenames or set()) - project_approved_filenames
    if not run_only_approved:
        return hidden_char_effective_filenames, hidden_char_effective_count
    updated_hidden = set(hidden_char_effective_filenames or set()) - set(run_only_approved)
    updated_count = sum(
        1
        for ann in annotations
        if str(getattr(ann, "filename", "") or "").strip().lower() in updated_hidden
    )
    return updated_hidden, int(updated_count or 0)


def _restore_preview_from_annotation_run(
    self,
    run_dir: Path | None,
    *,
    defer_ui_restore: bool = False,
) -> bool:
    show_context_box = True
    if run_dir is None:
        return False

    run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if run_dir is None:
        return False

    restore_started = time.perf_counter()
    try:
        xml_path = run_dir / "annotations.xml"
        manifest = self._load_annotation_run_manifest(run_dir)
        try:
            run_is_manual_template = bool(self._annotation_run_manifest_is_manual_template(manifest))
        except Exception:
            run_is_manual_template = False
        run_is_auto_annotation = bool(
            str(manifest.get("annotation_run_type") or "").strip().lower() == "auto_annotation"
        )
        annotations = self._parse_cvat_preview_annotations(xml_path)
        if not annotations:
            return False
        if run_is_auto_annotation:
            try:
                self._mark_auto_plate_origin_for_annotations(annotations)
            except Exception:
                pass

        image_dir_candidates = []
        seen_candidates = set()
        for raw_value in (
            str(manifest.get("imported_source_input_dir") or "").strip(),
            str(manifest.get("source_input_dir") or "").strip(),
            str(manifest.get("input_dir") or "").strip(),
            str(self.input_dir_var.get() or "").strip(),
            str(self.plate_dataset_images_var.get() or "").strip(),
            str(run_dir / "images"),
            str(run_dir),
        ):
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
        if image_dir is None:
            return False

        t02_at_review_context = _is_t02_at_review_restore_context(self)
        hidden_char_effective_filenames = set()
        hidden_char_effective_count = 0
        if not self._is_free_mode_session_context() and not t02_at_review_context:
            active_extract_run = None
            try:
                from ..campaign_manager import CAMPAIGN

                extract_state = dict(CAMPAIGN.get_step3_extract_state() or {})
                raw_extract_run = str(extract_state.get("annotation_run_dir") or "").strip()
                if raw_extract_run:
                    active_extract_run = self._resolve_safe_annotation_run_dir(raw_extract_run, require_xml=True)
            except Exception:
                active_extract_run = None
            same_as_extract_run = False
            if active_extract_run is not None:
                try:
                    same_as_extract_run = self._paths_equivalent(active_extract_run, run_dir)
                except Exception:
                    same_as_extract_run = False
            if not same_as_extract_run:
                hidden_char_effective_filenames = self._get_campaign_char_effective_source_hidden_filenames()
                if hidden_char_effective_filenames:
                    hidden_char_effective_count = sum(
                        1
                        for ann in annotations
                        if str(getattr(ann, "filename", "") or "").strip().lower() in hidden_char_effective_filenames
                    )

        run_approved_filenames = self._load_annotation_run_approved_filenames(run_dir)
        _sync_t06_approved_run_before_restore_filter(self, run_dir, run_approved_filenames)
        extra_hidden_for_restore = set(hidden_char_effective_filenames)
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        except Exception:
            graph_context = {}
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
        repair_origin_gate_id = campaign_gate_id_for_edge(
            graph_context.get("repair_origin_edge_key"),
            graph_context.get("repair_origin_gate_id")
            or graph_context.get("source_graph_gate_id"),
        )
        if graph_gate_id == "T04" and repair_origin_gate_id == "T06":
            project_approved_filenames = set(self._get_campaign_plate_approved_filenames() or set())
            run_only_approved = set(run_approved_filenames or set()) - project_approved_filenames
            hidden_char_effective_filenames -= run_only_approved
            if run_only_approved:
                hidden_char_effective_count = sum(
                    1
                    for ann in annotations
                    if str(getattr(ann, "filename", "") or "").strip().lower() in hidden_char_effective_filenames
                )
            extra_hidden_for_restore = set(hidden_char_effective_filenames)
        hidden_char_effective_filenames, hidden_char_effective_count = (
            _keep_pending_t06_approvals_visible_for_restore(
                self,
                run_dir,
                annotations,
                set(run_approved_filenames or set()),
                set(hidden_char_effective_filenames),
                int(hidden_char_effective_count or 0),
            )
        )
        extra_hidden_for_restore = set(hidden_char_effective_filenames)
        preserve_manual_hidden_for_restore = (
            set(run_approved_filenames or set())
            if _is_t06_interrupted_restore_context(self, run_dir)
            else set()
        )

        if t02_at_review_context:
            filtered_run_approved_filenames = set(run_approved_filenames or set())
            hidden_project_approved_count = 0
        else:
            annotations, filtered_run_approved_filenames, hidden_project_approved_count = (
                self._filter_campaign_project_approved_annotations(
                    annotations,
                    image_dir=image_dir,
                    run_dir=run_dir,
                    run_approved_filenames=run_approved_filenames,
                    extra_hidden_filenames=extra_hidden_for_restore,
                    preserve_manual_hidden_filenames=preserve_manual_hidden_for_restore,
                )
            )

        self._clear_preview_editor_state(clear_dirty=True)
        self.current_annotations = annotations
        self.current_input_dir = image_dir
        self._preview_image_path_map = {}
        self._clear_campaign_manual_reuse_context()
        self._campaign_iteration_manual_filenames = set()
        self.input_dir_var.set(str(image_dir))
        self.plate_dataset_images_var.set(str(image_dir))
        self.current_annotation_run_dir = run_dir
        self.current_annotation_xml_path = xml_path
        self.last_staging_run_dir = run_dir
        self._preview_approved_filenames = set(filtered_run_approved_filenames)
        if not self._is_free_mode_session_context():
            hidden_project_filter_filenames = {
                str(name or "").strip().lower()
                for name in set(getattr(self, "_last_campaign_project_approved_filter_hidden_filenames", set()) or set())
                if str(name or "").strip()
            }
            self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
            self._campaign_hidden_project_approved_filenames = {
                str(name or "").strip().lower()
                for name in set(run_approved_filenames or set())
                if str(name or "").strip()
            } - set(self._preview_approved_filenames)
            self._campaign_hidden_project_approved_filenames |= hidden_project_filter_filenames
            self._campaign_hidden_char_effective_filenames = set(hidden_char_effective_filenames)
            self._campaign_hidden_project_approved_count = int(hidden_project_approved_count or 0)
            self._campaign_hidden_char_effective_count = int(hidden_char_effective_count or 0)
        else:
            self._campaign_hidden_project_approved_filenames = set()
            self._campaign_hidden_char_effective_filenames = set()
            self._campaign_hidden_char_effective_filenames = set()
            self._campaign_hidden_project_approved_count = 0
            self._campaign_hidden_char_effective_count = 0
        approved_runtime_lookup = set(self._get_preview_approved_filenames())
        for ann in annotations:
            try:
                setattr(
                    ann,
                    "_approved_for_training",
                    str(getattr(ann, "filename", "") or "").strip().lower() in approved_runtime_lookup,
                )
            except Exception:
                continue

        if not self._is_free_mode_session_context():
            approved_missing_lookup: dict[str, Path] = {}
            for ann in annotations:
                safe_name = str(getattr(ann, "filename", "") or "").strip()
                if not safe_name or safe_name.lower() not in approved_runtime_lookup:
                    continue
                if len(self._get_plate_detections(ann)) > 0:
                    continue
                resolved_path = None
                try:
                    resolved_path = self._resolve_preview_image_path(ann)
                except Exception:
                    resolved_path = None
                if resolved_path is None:
                    try:
                        candidate = Path(image_dir) / safe_name
                        if candidate.exists():
                            resolved_path = candidate
                    except Exception:
                        resolved_path = None
                if resolved_path is not None:
                    approved_missing_lookup[safe_name] = Path(resolved_path)
            if approved_missing_lookup:
                try:
                    approved_bundle = self._build_campaign_plate_approved_preview_bundle(
                        image_names=set(approved_missing_lookup.keys()),
                        preferred_image_dir=image_dir,
                        image_candidates=approved_missing_lookup,
                    )
                except Exception:
                    approved_bundle = {}
                if approved_bundle:
                    try:
                        self._merge_preview_annotation_bundle(approved_bundle)
                        annotations = list(self.current_annotations or [])
                    except Exception:
                        pass
            if hidden_project_approved_count > 0:
                try:
                    self._append_z2_trace(
                        "restore-preview-hidden-approved",
                        f"ukryto={hidden_project_approved_count} run={run_dir}",
                    )
                except Exception:
                    pass
            if hidden_char_effective_count > 0:
                try:
                    self._append_z2_trace(
                        "restore-preview-hidden-char-effective",
                        f"ukryto={hidden_char_effective_count} run={run_dir}",
                    )
                except Exception:
                    pass

        expected_run_image_count = 0
        for count_key in (
            "input_scope_count",
            "source_plan_total_count",
            "selected_count",
            "image_count",
            "result_total_images",
        ):
            try:
                expected_run_image_count = int(manifest.get(count_key, 0) or 0)
            except Exception:
                expected_run_image_count = 0
            if expected_run_image_count > 0:
                break
        scoped_input_run = _annotation_run_uses_scoped_input(manifest)
        skip_missing_scan = bool(
            not scoped_input_run
            and expected_run_image_count > 0
            and len(annotations) >= expected_run_image_count
        )
        if not run_is_manual_template and not skip_missing_scan:
            missing_source_dir = self._resolve_existing_dir(
                manifest.get("source_input_dir")
                or manifest.get("imported_source_input_dir")
                or manifest.get("input_dir")
            )
            if missing_source_dir is None:
                missing_source_dir = image_dir
            scope_filenames = _annotation_run_input_scope_filenames(manifest)
            try:
                missing_bundle = self._build_missing_preview_annotations_bundle(
                    missing_source_dir,
                    existing_annotations=annotations,
                    extra_hidden_filenames=hidden_char_effective_filenames,
                    scope_filenames=scope_filenames,
                )
            except Exception:
                missing_bundle = {}
            try:
                self._append_z2_trace(
                    "restore-preview-missing-candidate",
                    (
                        f"source={missing_source_dir} "
                        f"existing={len(annotations)} "
                        f"missing={len(missing_bundle)} "
                        f"run={run_dir}"
                    ),
                )
            except Exception:
                pass
            if missing_bundle:
                try:
                    self._merge_preview_annotation_bundle(missing_bundle)
                    annotations = list(self.current_annotations or [])
                    self._append_z2_trace(
                        "restore-preview-missing",
                        f"restored={len(missing_bundle)} total={len(annotations)} run={run_dir}",
                    )
                except Exception:
                    pass
        self._load_plate_dataset_context_from_run(run_dir, force_images_update=False)

        try:
            self._restore_campaign_step2_generated_from_run(run_dir, only_when_pending=True)
        except Exception:
            pass

        self._sync_preview_approval_flags_from_current_sets()

        restore_idx = self._compute_annotation_run_restore_index(
            annotations,
            manifest,
            restore_filename=str(getattr(self, "_preview_session_restore_filename", "") or "").strip(),
            restore_index=getattr(self, "_preview_session_restore_index", None),
        )

        self.current_preview_index = restore_idx
        self._preview_session_restore_index = restore_idx
        self._preview_session_restore_filename = (
            str(getattr(annotations[restore_idx], "filename", "") or "")
            if restore_idx is not None and 0 <= int(restore_idx) < len(annotations)
            else ""
        )

        self._append_z2_trace(
            "restore-preview-run",
            f"run={run_dir} annotations={len(annotations)} deferred={int(bool(defer_ui_restore))}",
        )
        try:
            self._sync_campaign_pending_batch_summary_from_preview(
                hidden_project_approved_count=hidden_project_approved_count,
            )
        except Exception:
            pass
        try:
            pending_summary = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
            pending_summary["char_effective_skip_count"] = int(hidden_char_effective_count or 0)
            self._campaign_pending_batch_summary = pending_summary
        except Exception:
            pass
        try:
            if hidden_char_effective_count > 0:
                self._set_status_label_state(
                    (
                        f"Wczytano run anotacji: {run_dir.name}. "
                        f"{hidden_char_effective_count} obrazów jest już użytych w aktywnym E3 i nie wraca do listy Z2."
                    ),
                    "neutral",
                )
        except Exception:
            pass

        if not defer_ui_restore:
            self._refresh_preview_list(preserve_selection=True, render_current=False)
            if self.current_preview_index is not None:
                self._select_preview_index(int(self.current_preview_index), reset_view=True)
            else:
                self._load_current_preview_selection(reset_view=True, selection_changed=True)
                self._schedule_preview_selection_render_after_restore()
            self._refresh_preview_list_summary()
            self._refresh_plate_dataset_export_sources()
            self._refresh_step2_action_states()
        return True
    except Exception as e:
        logger.debug(f"Nie udalo sie przywrocic podgladu Z2 z runu {run_dir}: {e}")
        return False
    finally:
        elapsed_ms = max(0.0, (time.perf_counter() - restore_started) * 1000.0)
        if elapsed_ms >= 20.0:
            logger.debug(
                "[AnnotationTab][PERF] restore_preview_from_annotation_run: "
                f"{elapsed_ms:.1f} ms | deferred={int(bool(defer_ui_restore))}"
            )

def _apply_campaign_project_snapshot(self, session_state: dict | None = None, restore_preview: bool = True) -> bool:
    state = dict(session_state or self._load_campaign_project_snapshot())
    if not state:
        return False

    deferred_preview_load = False

    try:
        from ..campaign_manager import CAMPAIGN

        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        return False

    if active_project and str(state.get("project", "") or "").strip() not in {"", active_project}:
        return False

    try:
        saved_iteration = int(state.get("iteration", current_iteration))
    except (TypeError, ValueError):
        saved_iteration = current_iteration
    if saved_iteration != current_iteration:
        return False

    current_input = str(self.input_dir_var.get() or "").strip()
    snapshot_input = str(state.get("input_dir") or "").strip()
    snapshot_uses_stage = bool(snapshot_input and self._is_manual_plate_stage_input(snapshot_input))
    if current_input and snapshot_input and not self._paths_equivalent(snapshot_input, current_input):
        if not snapshot_uses_stage:
            return False
        try:
            if not Path(snapshot_input).exists():
                return False
        except Exception:
            return False

    expected_preview_input = self._resolve_existing_dir(snapshot_input or current_input)
    preview_source_mismatch = False

    self._append_z2_trace(
        "apply-project-snapshot-start",
        f"restore_preview={int(bool(restore_preview))} state_project={str(state.get('project') or '').strip()}",
    )
    self._campaign_project_restore_in_progress = True
    try:
        registry_active_entry = self._get_campaign_step2_active_run_entry(
            images_dir=expected_preview_input,
            iteration_num=current_iteration,
        )
        registry_active_images_dir = self._resolve_existing_dir(
            registry_active_entry.get("images_dir")
        )
        if expected_preview_input is None and registry_active_images_dir is not None:
            expected_preview_input = registry_active_images_dir

        registry_active_run_dir = self._resolve_safe_annotation_run_dir(
            registry_active_entry.get("run_dir"),
            require_xml=True,
        )

        safe_run_dir = (
            registry_active_run_dir
            or self._resolve_safe_annotation_run_dir(state.get("plate_dataset_run"), require_xml=True)
        )
        safe_last_preview_run_dir = (
            registry_active_run_dir
            or self._resolve_safe_annotation_run_dir(state.get("last_preview_run_dir"), require_xml=True)
        )
        if expected_preview_input is not None:
            if safe_run_dir is not None and not self._annotation_run_matches_expected_input_dir(
                safe_run_dir,
                expected_preview_input,
            ):
                preview_source_mismatch = True
                self._append_z2_trace(
                    "apply-project-snapshot-run-mismatch",
                    f"plate_dataset_run={safe_run_dir} expected={expected_preview_input}",
                )
                safe_run_dir = None
            if safe_last_preview_run_dir is not None and not self._annotation_run_matches_expected_input_dir(
                safe_last_preview_run_dir,
                expected_preview_input,
            ):
                preview_source_mismatch = True
                self._append_z2_trace(
                    "apply-project-snapshot-preview-mismatch",
                    f"last_preview_run_dir={safe_last_preview_run_dir} expected={expected_preview_input}",
                )
                safe_last_preview_run_dir = None
        if snapshot_uses_stage:
            self.input_dir_var.set(snapshot_input)
        self.mode_var.set(self._normalize_mode_value(state.get("mode")))
        self.vehicle_model_var.set(str(state.get("vehicle_model") or "").strip())
        self.vehicle_custom_var.set(str(state.get("vehicle_custom") or "").strip())
        self.plate_custom_var.set(str(state.get("plate_custom") or "").strip())
        self.character_model_var.set(str(state.get("character_model") or "Brak / OCR").strip() or "Brak / OCR")
        self.character_custom_var.set(str(state.get("character_custom") or "").strip())
        self.device_var.set(str(state.get("device") or "auto").strip() or "auto")
        self.conf_var.set(float(state.get("conf", CONFIG.DEFAULT_CONFIDENCE)))
        self.plate_dataset_run_var.set(str(safe_run_dir or ""))
        snapshot_plate_images = str(state.get("plate_dataset_images") or "").strip()
        if (
            expected_preview_input is not None
            and snapshot_plate_images
            and not self._paths_equivalent(snapshot_plate_images, expected_preview_input)
            and not self._is_manual_plate_stage_input(snapshot_plate_images)
        ):
            snapshot_plate_images = str(expected_preview_input)
        self.plate_dataset_images_var.set(snapshot_plate_images)
        self.plate_train_pct.set(float(state.get("plate_train_pct", 80.0)))
        self.plate_val_pct.set(float(state.get("plate_val_pct", 10.0)))
        self.manual_xml_template_var.set(bool(state.get("manual_xml_template", False)))
        self.manual_vehicle_assist_var.set(bool(state.get("manual_vehicle_assist", False)))
        self._set_auto_vehicle_choice_state(state.get("auto_vehicle_choice"), campaign_context=True)
        self._set_workflow_route_state("", campaign_context=True)
        self._set_manual_entry_mode_state("continue", campaign_context=True)
        self._set_workflow_step_state("", campaign_context=True)
        self._set_screen_state(
            self._normalize_free_mode_screen_value(state.get("free_mode_screen")),
            campaign_context=True,
        )
        self._manual_review_active = False
        self._manual_review_from_auto = False
        self._manual_review_origin_route = ""
        self._last_completed_workflow_route = ""
        self._manual_review_export_ready = False
        self._manual_review_history_entries = self._normalize_manual_review_history_entries(
            state.get("manual_review_history", [])
        )

        self.last_staging_run_dir = safe_last_preview_run_dir
        try:
            self._preview_session_restore_index = int(state.get("last_preview_index", -1))
        except (TypeError, ValueError):
            self._preview_session_restore_index = -1
        self._preview_session_restore_filename = str(state.get("last_preview_filename") or "").strip()
        restored_approved = {
            str(name or "").strip().lower()
            for name in list(state.get("preview_approved_filenames") or [])
            if str(name or "").strip()
        }
        hidden_restored_approved = set()
        if not self._is_free_mode_session_context() and restored_approved:
            try:
                project_approved_filenames = {
                    str(Path(str(name or "").replace("\\", "/")).name or "").strip().lower()
                    for name in set(self._get_campaign_plate_approved_filenames() or set())
                    if str(name or "").strip()
                }
            except Exception:
                project_approved_filenames = set()
            if project_approved_filenames:
                hidden_restored_approved = set(restored_approved) & project_approved_filenames
                restored_approved = set(restored_approved) - hidden_restored_approved
        self._preview_approved_filenames = set(restored_approved)
        self._campaign_pending_approved_filenames = set(restored_approved)
        if hidden_restored_approved:
            self._campaign_hidden_project_approved_filenames = (
                set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
                | set(hidden_restored_approved)
            )

        self.apply_global_yolo_device_choice(self.device_var.get())
        self._refresh_device_options()
        self._update_model_lists()

        if hasattr(self, "vehicle_combo"):
            try:
                vehicle_values = list(self.vehicle_combo["values"])
            except Exception:
                vehicle_values = []
            current_vehicle = str(self.vehicle_model_var.get() or "").strip()
            if vehicle_values and current_vehicle not in vehicle_values:
                preferred_vehicle = "yolo11s" if "yolo11s" in vehicle_values else vehicle_values[0]
                self.vehicle_model_var.set(preferred_vehicle)

        if hasattr(self, "character_combo"):
            self._refresh_character_model_choices()

        self._repair_restored_z2_workflow_completion_state(
            ""
        )

        self._refresh_manual_review_history_ui()

        self._on_mode_change()
        self._on_vehicle_model_change()
        self._update_manual_xml_template_ui()
        self._update_plate_dataset_ratio_labels()
        self._refresh_plate_dataset_export_sources()
        self._set_campaign_paths_lock_state(True)

        suppress_preview_restore = False
        try:
            campaign_step = int(CAMPAIGN.get_current_step() or 0)
            campaign_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
            step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
            current_step2_run = self._resolve_safe_annotation_run_dir(
                CAMPAIGN.get_step2_staging_run(),
                require_xml=True,
            )
            suppress_preview_restore = bool(
                campaign_step == 2
                and campaign_target == "plate"
                and step2_status in {"", "pending"}
                and current_step2_run is None
            )
        except Exception:
            suppress_preview_restore = False

        if (
            not suppress_preview_restore
            and not self._should_restore_campaign_generated_step2_run_preview(
                safe_run_dir or safe_last_preview_run_dir,
                session_state=state,
            )
        ):
            suppress_preview_restore = True
        if preview_source_mismatch:
            suppress_preview_restore = True

        if suppress_preview_restore:
            safe_run_dir = None
            safe_last_preview_run_dir = None
            self.plate_dataset_run_var.set("")
            self.last_staging_run_dir = None
            self.current_annotation_run_dir = None
            self.current_annotation_xml_path = None
            self._campaign_t06_entry_approval_baseline = None
            self.current_annotations = []
            self._preview_image_path_map = {}
            if expected_preview_input is not None:
                self.plate_dataset_images_var.set(str(expected_preview_input))
            self._set_workflow_route_state("", campaign_context=True)
            self._set_workflow_step_state("", campaign_context=True)
            self.manual_xml_template_var.set(False)
            self._manual_review_active = False
            self._manual_review_from_auto = False
            self._manual_review_export_ready = False
            self._last_completed_workflow_route = ""
            self._append_z2_trace(
                "apply-project-snapshot-suppressed",
                "wyczyszczono stary workflow_route, aby kampania mogla na nowo nalozyc preset E2/Z2",
            )

        input_dir_value = str(self.input_dir_var.get() or "").strip()
        self.current_input_dir = Path(input_dir_value) if input_dir_value else None

        if restore_preview and not suppress_preview_restore:
            restored_preview = bool(self._restore_preview_from_session_run())
        else:
            restored_preview = False
        if not restored_preview and self.current_input_dir is not None and not self.current_annotations:
            try:
                self._prime_campaign_source_preview(self.current_input_dir)
            except Exception as e:
                logger.debug(f"Nie udało się przygotować podglądu wejściowego Z2 po restarcie: {e}")
        approved_runtime_lookup = set(self._get_preview_approved_filenames_base())
        if self.current_annotations:
            sync_changed = self._sync_preview_approval_flags_from_current_sets(
                refresh_list=bool(approved_runtime_lookup),
                render_current=False,
            )
            if approved_runtime_lookup or sync_changed:
                deferred_preview_load = True
        self._manual_review_active = bool(self._manual_review_active and restored_preview)
        self._manual_review_from_auto = bool(self._manual_review_from_auto and self._manual_review_active)
        self._manual_review_export_ready = bool(self._manual_review_export_ready and self._manual_review_active)
        self._refresh_step2_action_states()
    finally:
        self._campaign_project_restore_in_progress = False

    if bool(getattr(self, "_campaign_restore_pending_preview_load", False)):
        deferred_preview_load = True
        self._campaign_restore_pending_preview_load = False

    if deferred_preview_load and self.current_annotations and self.current_preview_index is not None:
        try:
            self._append_z2_trace(
                "apply-project-snapshot-final-preview",
                f"idx={int(self.current_preview_index)} annotations={len(self.current_annotations or [])}",
            )
            self._load_current_preview_selection(reset_view=True, selection_changed=True)
            self._update_preview_edit_status("Przywrócono ostatni run anotacji Z2 z poprzedniej sesji.")
            self._refresh_step2_action_states()
        except Exception as e:
            logger.debug(f"Nie udało się odtworzyć końcowego podglądu Z2 po restarcie projektu: {e}")

    self._append_z2_trace(
        "apply-project-snapshot-end",
        f"restored={int(bool(self.current_annotations))} annotations={len(self.current_annotations or [])}",
    )

    return True

def _apply_annotation_run_restore_payload(
    self,
    payload: dict | None,
    *,
    clear_existing_state: bool = True,
    use_async_list: bool = False,
    status_message: str | None = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    apply_started = time.perf_counter()
    apply_phase_started = apply_started
    slow_apply_phases: list[str] = []

    def _mark_apply_phase(name: str) -> None:
        nonlocal apply_phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - apply_phase_started) * 1000.0
            if elapsed_ms >= 250.0:
                slow_apply_phases.append(f"{name}={elapsed_ms:.0f}ms")
            apply_phase_started = now
        except Exception:
            pass

    run_dir = self._resolve_safe_annotation_run_dir(payload.get("run_dir"), require_xml=True)
    image_dir = self._resolve_existing_dir(payload.get("image_dir"))
    xml_path = self._path_value_to_path(payload.get("xml_path"))
    annotations = list(payload.get("annotations") or [])
    image_map = dict(payload.get("image_map") or {})
    if run_dir is None or image_dir is None or xml_path is None or not annotations:
        return False
    run_approved_filenames = {
        str(name or "").strip().lower()
        for name in set(payload.get("approved_filenames") or set())
        if str(name or "").strip()
    }
    manifest = dict(payload.get("manifest") or {})
    payload_is_auto_annotation = bool(
        str(manifest.get("annotation_run_type") or "").strip().lower() == "auto_annotation"
    )
    if payload_is_auto_annotation:
        try:
            self._mark_auto_plate_origin_for_annotations(annotations)
            self._invalidate_preview_runtime_caches()
        except Exception:
            pass
    _sync_t06_approved_run_before_restore_filter(self, run_dir, run_approved_filenames)
    _mark_apply_phase("pre_sync")
    t02_at_review_context = _is_t02_at_review_restore_context(self)
    hidden_char_effective_filenames = set()
    hidden_char_effective_count = 0
    if not self._is_free_mode_session_context() and not t02_at_review_context:
        active_extract_run = None
        try:
            from ..campaign_manager import CAMPAIGN

            extract_state = dict(CAMPAIGN.get_step3_extract_state() or {})
            raw_extract_run = str(extract_state.get("annotation_run_dir") or "").strip()
            if raw_extract_run:
                active_extract_run = self._resolve_safe_annotation_run_dir(raw_extract_run, require_xml=True)
        except Exception:
            active_extract_run = None
        same_as_extract_run = False
        if active_extract_run is not None:
            try:
                same_as_extract_run = self._paths_equivalent(active_extract_run, run_dir)
            except Exception:
                same_as_extract_run = False
        if not same_as_extract_run:
            hidden_char_effective_filenames = self._get_campaign_char_effective_source_hidden_filenames()
            if hidden_char_effective_filenames:
                hidden_char_effective_count = sum(
                    1
                    for ann in annotations
                    if str(getattr(ann, "filename", "") or "").strip().lower() in hidden_char_effective_filenames
                )
    _mark_apply_phase("hidden_char")
    extra_hidden_for_restore = set(hidden_char_effective_filenames)
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    repair_origin_gate_id = campaign_gate_id_for_edge(
        graph_context.get("repair_origin_edge_key"),
        graph_context.get("repair_origin_gate_id")
        or graph_context.get("source_graph_gate_id"),
    )
    if graph_gate_id == "T04" and repair_origin_gate_id == "T06":
        project_approved_filenames = set(self._get_campaign_plate_approved_filenames() or set())
        run_only_approved = set(run_approved_filenames or set()) - project_approved_filenames
        hidden_char_effective_filenames -= run_only_approved
        if run_only_approved:
            hidden_char_effective_count = sum(
                1
                for ann in annotations
                if str(getattr(ann, "filename", "") or "").strip().lower() in hidden_char_effective_filenames
            )
        extra_hidden_for_restore = set(hidden_char_effective_filenames)
    hidden_char_effective_filenames, hidden_char_effective_count = (
        _keep_pending_t06_approvals_visible_for_restore(
            self,
            run_dir,
            annotations,
            set(run_approved_filenames or set()),
            set(hidden_char_effective_filenames),
            int(hidden_char_effective_count or 0),
        )
    )
    extra_hidden_for_restore = set(hidden_char_effective_filenames)
    preserve_manual_hidden_for_restore = (
        set(run_approved_filenames or set())
        if _is_t06_interrupted_restore_context(self, run_dir)
        else set()
    )

    if t02_at_review_context:
        filtered_run_approved_filenames = set(run_approved_filenames or set())
        hidden_project_approved_count = 0
    else:
        annotations, filtered_run_approved_filenames, hidden_project_approved_count = (
            self._filter_campaign_project_approved_annotations(
                annotations,
                image_dir=image_dir,
                run_dir=run_dir,
                run_approved_filenames=run_approved_filenames,
                extra_hidden_filenames=extra_hidden_for_restore,
                preserve_manual_hidden_filenames=preserve_manual_hidden_for_restore,
            )
        )
    _mark_apply_phase("filter_project_approved")

    if clear_existing_state:
        self._clear_preview_editor_state(clear_dirty=True)
    self.current_annotations = annotations
    self.current_input_dir = image_dir
    self._preview_image_path_map = image_map
    self._clear_campaign_manual_reuse_context()
    self.input_dir_var.set(str(image_dir))
    self.plate_dataset_images_var.set(str(image_dir))
    self.current_annotation_run_dir = run_dir
    self.current_annotation_xml_path = xml_path
    self.last_staging_run_dir = run_dir
    try:
        self._campaign_t06_previous_source_counts = None
    except Exception:
        pass
    self.plate_dataset_run_var.set(str(run_dir))
    self._preview_approved_filenames = set(filtered_run_approved_filenames)
    if not self._is_free_mode_session_context() and not t02_at_review_context:
        try:
            self._sync_campaign_iteration_artifact_registry(
                run_dir=run_dir,
                xml_path=xml_path,
                input_dir=image_dir,
            )
        except Exception:
            pass
    if not self._is_free_mode_session_context():
        hidden_project_filter_filenames = {
            str(name or "").strip().lower()
            for name in set(getattr(self, "_last_campaign_project_approved_filter_hidden_filenames", set()) or set())
            if str(name or "").strip()
        }
        self._campaign_hidden_project_approved_filenames = (
            set(run_approved_filenames) - set(self._preview_approved_filenames)
        )
        self._campaign_hidden_project_approved_filenames |= hidden_project_filter_filenames
        self._campaign_hidden_char_effective_filenames = set(hidden_char_effective_filenames)
        self._campaign_hidden_project_approved_count = int(hidden_project_approved_count or 0)
        self._campaign_hidden_char_effective_count = int(hidden_char_effective_count or 0)
    else:
        self._campaign_hidden_project_approved_filenames = set()
        self._campaign_hidden_char_effective_filenames = set()
        self._campaign_hidden_project_approved_count = 0
        self._campaign_hidden_char_effective_count = 0
    self._campaign_iteration_manual_filenames = set()
    if not self._is_free_mode_session_context():
        self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
    approved_runtime_lookup = {
        str(name or "").strip().lower()
        for name in set(filtered_run_approved_filenames or set())
        if str(name or "").strip()
    }
    approved_runtime_images = 0
    approved_runtime_plates = 0
    for ann in annotations:
        image_name = str(getattr(ann, "filename", "") or "").strip().lower()
        try:
            plate_count = len(self._get_plate_detections(ann))
        except Exception:
            plate_count = 0
        is_approved_for_training = bool(
            image_name
            and image_name in approved_runtime_lookup
            and int(plate_count or 0) > 0
        )
        if is_approved_for_training:
            approved_runtime_images += 1
            approved_runtime_plates += int(plate_count or 0)
        try:
            setattr(ann, "_approved_for_training", is_approved_for_training)
        except Exception:
            continue
    _mark_apply_phase("state_and_approval_flags")

    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    if graph_gate_id == "T05":
        try:
            baseline_token = f"{Path(run_dir).resolve()}|T06"
        except Exception:
            baseline_token = f"{run_dir}|T06"
        baseline_images, baseline_plates = approved_runtime_images, approved_runtime_plates
        try:
            self._campaign_t06_entry_approval_baseline = {
                "token": baseline_token,
                "images": int(baseline_images or 0),
                "plates": int(baseline_plates or 0),
            }
        except Exception:
            pass
        try:
            project_name = ""
            try:
                active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
            except Exception:
                active_project = ""
            if active_project:
                project_name = active_project
            else:
                try:
                    safe_run_dir = Path(run_dir).resolve()
                    projects = dict((CAMPAIGN.state or {}).get("projects", {}) or {})
                    for candidate_name, project_data in projects.items():
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
                                project_name = str(candidate_name or "").strip()
                                break
                        except Exception:
                            continue
                except Exception:
                    project_name = ""
            stats = dict(CAMPAIGN.get_plate_approved_set_stats(project_name or None) or {})
            self._campaign_t06_entry_source_baseline = {
                "token": baseline_token,
                "images": int(stats.get("images", 0) or 0),
                "plates": int(stats.get("plates", 0) or 0),
            }
        except Exception:
            self._campaign_t06_entry_source_baseline = {
                "token": baseline_token,
                "images": 0,
                "plates": 0,
            }
    else:
        try:
            self._campaign_t06_entry_approval_baseline = None
        except Exception:
            pass
        try:
            self._campaign_t06_entry_source_baseline = None
        except Exception:
            pass
    _mark_apply_phase("t06_baseline")

    if not self._is_free_mode_session_context():
        approved_missing_lookup: dict[str, Path] = {}
        for ann in annotations:
            safe_name = str(getattr(ann, "filename", "") or "").strip()
            if not safe_name or safe_name.lower() not in approved_runtime_lookup:
                continue
            if len(self._get_plate_detections(ann)) > 0:
                continue
            try:
                candidate = Path(image_dir) / safe_name
                if candidate.exists():
                    approved_missing_lookup[safe_name] = candidate
            except Exception:
                continue
        if approved_missing_lookup:
            try:
                approved_bundle = self._build_campaign_plate_approved_preview_bundle(
                    image_names=set(approved_missing_lookup.keys()),
                    preferred_image_dir=image_dir,
                    image_candidates=approved_missing_lookup,
                )
            except Exception:
                approved_bundle = {}
            if approved_bundle:
                try:
                    self._merge_preview_annotation_bundle(approved_bundle)
                    annotations = list(self.current_annotations or [])
                except Exception:
                    pass
    _mark_apply_phase("approved_missing_bundle")

    self._load_plate_dataset_context_from_run(run_dir, force_images_update=False)
    if not t02_at_review_context:
        try:
            self._restore_campaign_step2_generated_from_run(run_dir, only_when_pending=True)
        except Exception:
            pass
    self._sync_preview_approval_flags_from_current_sets()
    _mark_apply_phase("run_context")

    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        active_graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        active_graph_gate_id = ""
    restore_index = payload.get("restore_index")
    if active_graph_gate_id == "T05":
        # T06 is a "continue adding plates" gate. Showing the last preview
        # first and then jumping to the first row looks like a visual glitch,
        # so make the first visible row the initial source of truth.
        try:
            first_entries = self._get_preview_list_entries()
            restore_index = int(first_entries[0][0]) if first_entries else (0 if annotations else None)
        except Exception:
            restore_index = 0 if annotations else None
    self.current_preview_index = restore_index
    self._preview_session_restore_index = restore_index
    self._preview_session_restore_filename = str(payload.get("restore_filename") or "").strip()
    large_async_restore = bool(use_async_list and (len(annotations or []) >= 1200 or t02_at_review_context))
    # T02 może wczytywać kilka tysięcy pozycji naraz. Duże porcje listboxa
    # wyglądają jak "async", ale w Tk potrafią zamrozić UI na długie sekundy.
    async_list_batch_size = 220 if t02_at_review_context else 350
    initial_preview_scheduled = False

    def _schedule_t02_post_restore_layout_fit() -> None:
        if not t02_at_review_context:
            return
        try:
            self._schedule_main_pane_layout_refresh(
                force_defaults=not bool(getattr(self, "_main_pane_layout_initialized", False)),
                delay_ms=40,
            )
        except Exception:
            pass

        def _fit_after_layout() -> None:
            try:
                if getattr(getattr(self, "preview_canvas", None), "original_image", None) is None:
                    return
                self._preview_force_fit_after_resize = True
                self._schedule_preview_layout_restore_after_resize()
            except Exception:
                pass

        for delay_ms in (140, 360):
            try:
                self.frame.after(int(delay_ms), _fit_after_layout)
            except Exception:
                _fit_after_layout()
                break

    def _schedule_initial_preview_render() -> None:
        nonlocal initial_preview_scheduled
        if initial_preview_scheduled:
            return
        initial_preview_scheduled = True

        def _render_initial_preview() -> None:
            if not self.current_annotations or self.current_preview_index is None:
                return
            try:
                self._select_preview_index(int(self.current_preview_index), reset_view=True)
            except Exception:
                pass

        try:
            self.frame.after(80, _render_initial_preview)
        except Exception:
            _render_initial_preview()

    def _finish_restore() -> None:
        finish_started = time.perf_counter()
        finish_phase_started = finish_started
        finish_phases: list[str] = []

        def _mark_finish_phase(name: str) -> None:
            nonlocal finish_phase_started
            try:
                now = time.perf_counter()
                elapsed_ms = (now - finish_phase_started) * 1000.0
                if elapsed_ms >= 100.0:
                    finish_phases.append(f"{name}={elapsed_ms:.0f}ms")
                finish_phase_started = now
            except Exception:
                pass

        try:
            self._refresh_preview_list_summary(lightweight=large_async_restore)
        except Exception:
            pass
        _mark_finish_phase("list_summary")
        graph_gate_id = ""
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
            graph_gate_id = campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
        except Exception:
            graph_gate_id = ""
        campaign_context = bool(not self._is_free_mode_session_context())
        if (
            graph_gate_id == "T05"
            and campaign_context
            and run_approved_filenames
        ):
            try:
                sync_result = self._sync_campaign_char_repair_approved_run_to_project_source(
                    run_dir,
                    refresh_effective_source=False,
                    reason="restore",
                )
                if bool(dict(sync_result or {}).get("ok")):
                    self._campaign_t06_previous_source_counts = None
                    self._append_z2_trace(
                        "t06-restore-approved-sync",
                        (
                            f"run={run_dir} approved={len(run_approved_filenames)} "
                            f"total={int(dict(sync_result or {}).get('total', 0) or 0)}"
                        ),
                    )
            except Exception as e:
                logger.debug(f"Nie udało się pogodzić zatwierdzeń T06 po odtworzeniu runu: {e}")
        _mark_finish_phase("t06_sync")
        skip_export_source_refresh = bool(
            campaign_context
            and graph_gate_id in {"T03", "T04", "T05"}
        )
        if skip_export_source_refresh:
            try:
                self._set_plate_export_status(
                    "Status eksportu zostanie przeliczony dopiero po wejściu w przepływ eksportu.",
                    "muted",
                )
            except Exception:
                pass
        else:
            try:
                self._refresh_plate_dataset_export_sources()
            except Exception:
                pass
        self._campaign_deferred_run_restore_payload_applied = True
        self._campaign_deferred_run_restore_in_progress = False
        try:
            self._refresh_step2_action_states()
        except Exception:
            pass
        _mark_finish_phase("action_states")
        try:
            self._refresh_free_mode_workflow_ui()
        except Exception:
            pass
        _mark_finish_phase("workflow_ui")
        try:
            self._sync_right_panel_scrollregion()
        except Exception:
            pass
        _mark_finish_phase("scrollregion")
        if status_message:
            try:
                self._update_preview_edit_status(status_message)
            except Exception:
                pass
        if hidden_project_approved_count > 0:
            try:
                self._append_z2_trace(
                    "apply-restore-hidden-approved",
                    f"ukryto={hidden_project_approved_count} run={run_dir}",
                )
            except Exception:
                pass
        if hidden_char_effective_count > 0:
            try:
                self._append_z2_trace(
                    "apply-restore-hidden-char-effective",
                    f"ukryto={hidden_char_effective_count} run={run_dir}",
                )
            except Exception:
                pass
        try:
            self._sync_campaign_pending_batch_summary_from_preview(
                hidden_project_approved_count=hidden_project_approved_count,
            )
        except Exception:
            pass
        try:
            pending_summary = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
            pending_summary["char_effective_skip_count"] = int(hidden_char_effective_count or 0)
            self._campaign_pending_batch_summary = pending_summary
        except Exception:
            pass
        if hidden_char_effective_count > 0:
            try:
                self._set_status_label_state(
                    (
                        f"Wczytano run anotacji: {run_dir.name}. "
                        f"{hidden_char_effective_count} obrazów jest już użytych w aktywnym E3 i nie wraca do listy Z2."
                    ),
                    "neutral",
                )
            except Exception:
                pass
        elapsed_ms = (time.perf_counter() - apply_started) * 1000.0
        if elapsed_ms >= 250.0:
            try:
                logger.info(
                    "[AnnotationTab][PERF] apply_annotation_run_restore_payload: "
                    "total=%.1fms annotations=%s async=%s hidden_project=%s hidden_char=%s phases=[%s] finish=[%s] run=%s",
                    elapsed_ms,
                    len(self.current_annotations or []),
                    bool(use_async_list and len(self.current_annotations or []) >= 1200),
                    int(hidden_project_approved_count or 0),
                    int(hidden_char_effective_count or 0),
                    ", ".join(slow_apply_phases) if slow_apply_phases else "no_slow_phase",
                    ", ".join(finish_phases) if finish_phases else "no_slow_finish_phase",
                    run_dir,
                )
            except Exception:
                pass

    if large_async_restore:
        _schedule_initial_preview_render()
        _schedule_t02_post_restore_layout_fit()
    else:
        try:
            if self.current_annotations and self.current_preview_index is not None:
                self._select_preview_index(int(self.current_preview_index), reset_view=True)
        except Exception:
            pass
        _schedule_t02_post_restore_layout_fit()

    if large_async_restore:
        # The run data is already restored at this point. Painting thousands of
        # rows in the listbox is only a background UI task, so the right panel
        # must not stay in the "loading" gate state until list rendering ends.
        _finish_restore()
        self._populate_preview_list_async(
            preserve_selection=True,
            render_current=False,
            batch_size=async_list_batch_size,
            on_complete=_schedule_t02_post_restore_layout_fit if t02_at_review_context else None,
            lightweight_summary=bool(t02_at_review_context),
        )
        _mark_apply_phase("schedule_async_list")
    else:
        self._refresh_preview_list(
            preserve_selection=True,
            render_current=False,
        )
        _finish_restore()
        _mark_apply_phase("sync_list")
    return True

def _ensure_free_mode_input_workspace_preview(
    self,
    *,
    expected_route: str = "auto",
    expected_step: str | None = None,
) -> bool:
    if not self._is_free_mode_session_context():
        return False
    current_route = str(self._get_workflow_route() or "").strip().lower()
    expected_route = str(expected_route or "").strip().lower()
    if current_route != expected_route:
        return False
    if str(self._coerce_free_mode_screen() or "").strip().lower() != "workflow":
        return False
    current_step = str(self._coerce_workflow_step() or "").strip().lower()
    expected_step = str(expected_step or f"{expected_route}_start").strip().lower()
    if current_step != expected_step:
        return False
    manual_preview = bool(current_route == "manual")

    images_value = str(self.input_dir_var.get() or "").strip()
    if not images_value:
        return False

    resolved_input_dir = self._resolve_annotation_input_images_dir(Path(images_value))
    if resolved_input_dir is None:
        return False

    current_input_matches = False
    try:
        if getattr(self, "current_input_dir", None) is not None:
            current_input_matches = self._paths_equivalent(Path(self.current_input_dir), resolved_input_dir)
    except Exception:
        current_input_matches = False

    if current_input_matches and bool(getattr(self, "current_annotations", None)):
        return True

    try:
        image_paths = get_image_files(resolved_input_dir)
    except Exception:
        image_paths = []
    if not image_paths:
        return False

    total_images = len(image_paths)
    splash_token = 0
    try:
        splash_token = self._show_campaign_step2_splash(
            title="Ładowanie katalogu obrazów",
            body=(
                f"Przygotowuję zdjęcia wejściowe do Z2. "
                f"Wczytano 0 z {total_images} obrazów."
            ),
            tone="info",
            progress=0.0,
        )
        self.frame.update_idletasks()
        self.frame.update()
    except Exception:
        splash_token = 0

    preview_annotations: list[ImageAnnotation] = []
    image_map: dict[str, Path] = {}
    for idx, image_path in enumerate(image_paths, start=1):
        preview_annotations.append(
            ImageAnnotation(
                filename=image_path.name,
                width=1,
                height=1,
                detections=[],
                status=AnnotationStatus.NO_PLATE,
                status_message=(
                    "Obraz źródłowy gotowy do utworzenia XML ręcznej anotacji."
                    if manual_preview
                    else "Obraz źródłowy gotowy do autoanotacji. Użyj Start, aby uruchomić proces dla tego zestawu."
                ),
            )
        )
        image_map[str(image_path.name)] = Path(image_path)
        if idx % 50 == 0 or idx == total_images:
            try:
                prep_progress = min(55.0, (float(idx) / float(max(1, total_images))) * 55.0)
            except Exception:
                prep_progress = 0.0
            try:
                self._show_campaign_step2_splash(
                    title="Ładowanie katalogu obrazów",
                    body=(
                        f"Przygotowuję zdjęcia wejściowe do Z2. "
                        f"Wczytano {idx} z {total_images} obrazów."
                    ),
                    tone="info",
                    progress=prep_progress,
                )
                self.frame.update_idletasks()
            except Exception:
                pass

    self._clear_preview_editor_state(clear_dirty=True)
    self.current_annotations = preview_annotations
    self.current_input_dir = resolved_input_dir
    self._preview_image_path_map = image_map
    self._clear_campaign_manual_reuse_context()
    self._campaign_hidden_project_approved_filenames = set()
    self._campaign_hidden_char_effective_filenames = set()
    self._campaign_hidden_project_approved_count = 0
    self._campaign_hidden_char_effective_count = 0
    self._campaign_t06_entry_approval_baseline = None
    self._campaign_iteration_manual_filenames = set()
    self._preview_approved_filenames = set()
    self.current_preview_index = None
    self.current_annotation_run_dir = None
    self.current_annotation_xml_path = None
    self.last_staging_run_dir = None
    self.input_dir_var.set(str(resolved_input_dir))
    self.plate_dataset_images_var.set(str(resolved_input_dir))
    self.plate_dataset_run_var.set("")
    try:
        self._refresh_preview_workspace_visibility(manual_review_active=False)
    except Exception:
        pass

    def _update_workspace_load_progress(done_count: int, total_count: int) -> None:
        safe_total = max(1, int(total_count or 0))
        safe_done = max(0, min(int(done_count or 0), safe_total))
        try:
            list_progress = 55.0 + ((float(safe_done) / float(safe_total)) * 45.0)
        except Exception:
            list_progress = 55.0
        try:
            self._show_campaign_step2_splash(
                title="Ładowanie katalogu obrazów",
                body=(
                    f"Buduję listę i podgląd Z2. "
                    f"Wczytano {safe_done} z {safe_total} obrazów."
                ),
                tone="info",
                progress=list_progress,
            )
            self.frame.update_idletasks()
        except Exception:
            pass

    def _finish_workspace_preview_load() -> None:
        try:
            self._set_status_label_state(
                (
                    f"Wczytano {len(preview_annotations)} obrazów do pracy w Z2. Możesz teraz utworzyć XML anotacji ręcznej."
                    if manual_preview
                    else f"Wczytano {len(preview_annotations)} obrazów do pracy w Z2. Możesz zaznaczyć zakres i uruchomić autoanotację przyciskiem Start."
                ),
                "neutral",
            )
        except Exception:
            pass
        try:
            self._hide_campaign_step2_splash(token=splash_token if splash_token else None)
        except Exception:
            pass

    try:
        self._populate_preview_list(
            on_complete=_finish_workspace_preview_load,
            on_progress=_update_workspace_load_progress,
            batch_size=50,
        )
    except Exception:
        self._refresh_preview_list(
            preserve_selection=False,
            render_current=True,
            on_progress=_update_workspace_load_progress,
        )
        _finish_workspace_preview_load()
    try:
        self._set_progress_counters(0, 0, len(preview_annotations))
    except Exception:
        pass
    return True

def _restore_preview_from_session_run(self):
    run_dir = self._resolve_best_free_mode_restore_run()

    if run_dir is None:
        return False

    xml_path = run_dir / "annotations.xml"
    if not xml_path.exists():
        return False

    manifest = self._load_annotation_run_manifest(run_dir)
    try:
        annotations = self._parse_cvat_preview_annotations(xml_path)
    except Exception as e:
        logger.debug(f"Nie udalo sie przywrocic ostatniego runu Z2: {e}")
        return False

    if not annotations:
        return False

    image_dir_candidates = []
    seen_candidates = set()
    for raw_value in (
        str(self.input_dir_var.get() or "").strip(),
        str(self.plate_dataset_images_var.get() or "").strip(),
        str(manifest.get("input_dir") or "").strip(),
        str(manifest.get("source_input_dir") or "").strip(),
        str(manifest.get("imported_source_input_dir") or "").strip(),
        str(run_dir / "images"),
    ):
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
    if image_dir is None:
        return False

    self._clear_preview_editor_state(clear_dirty=True)
    self.current_annotations = annotations
    self.current_input_dir = image_dir
    self._preview_image_path_map = {}
    self._clear_campaign_manual_reuse_context()
    self.plate_dataset_images_var.set(str(image_dir))
    self.current_annotation_run_dir = run_dir
    self.current_annotation_xml_path = xml_path
    self.last_staging_run_dir = run_dir
    run_approved_filenames = set(self._load_annotation_run_approved_filenames(run_dir))
    if run_approved_filenames:
        self._preview_approved_filenames = set(self._get_preview_approved_filenames_base()) | set(run_approved_filenames)
        if not self._is_free_mode_session_context():
            self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
    hidden_char_effective_filenames = set()
    if not self._is_free_mode_session_context():
        try:
            hidden_char_effective_filenames = self._get_campaign_char_effective_source_hidden_filenames()
        except Exception:
            hidden_char_effective_filenames = set()
    try:
        run_is_manual_template = bool(self._annotation_run_manifest_is_manual_template(manifest))
    except Exception:
        run_is_manual_template = False
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    successful_run_image_count = 0
    try:
        successful_run_image_count = int(manifest.get("result_successful_images", 0) or 0)
    except Exception:
        successful_run_image_count = 0
    expected_run_image_count = 0
    for count_key in (
        "input_scope_count",
        "source_plan_total_count",
        "selected_count",
        "image_count",
        "result_total_images",
    ):
        try:
            expected_run_image_count = int(manifest.get(count_key, 0) or 0)
        except Exception:
            expected_run_image_count = 0
        if expected_run_image_count > 0:
            break
    scoped_input_run = _annotation_run_uses_scoped_input(manifest)
    skip_missing_scan = bool(
        not scoped_input_run
        and expected_run_image_count > 0
        and len(annotations) >= expected_run_image_count
    )
    missing_count_hint = (
        max(0, int(expected_run_image_count or 0) - int(len(annotations)))
        if expected_run_image_count > 0
        else 0
    )
    skip_t06_successful_scan = bool(
        graph_gate_id == "T05"
        and successful_run_image_count > 0
        and len(annotations) >= successful_run_image_count
    )
    if skip_t06_successful_scan and not scoped_input_run:
        skip_missing_scan = True
        _mark_prepare_phase("skip_missing_t06_successful_run")
    skip_large_nearly_complete_scan = bool(
        expected_run_image_count >= 3000
        and len(annotations) >= 3000
        and 0 < missing_count_hint <= 25
    )
    if skip_large_nearly_complete_scan and not scoped_input_run:
        skip_missing_scan = True
        _mark_prepare_phase("skip_missing_nearly_complete_large_run")
    if skip_missing_scan:
        _mark_prepare_phase("skip_missing_complete_run")
    if not run_is_manual_template and not skip_missing_scan:
        missing_source_dir = self._resolve_existing_dir(
            manifest.get("source_input_dir")
            or manifest.get("imported_source_input_dir")
            or manifest.get("input_dir")
        )
        if missing_source_dir is None:
            missing_source_dir = image_dir
        scope_filenames = _annotation_run_input_scope_filenames(manifest)
        try:
            missing_bundle = self._build_missing_preview_annotations_bundle(
                missing_source_dir,
                existing_annotations=annotations,
                extra_hidden_filenames=hidden_char_effective_filenames,
                scope_filenames=scope_filenames,
            )
        except Exception:
            missing_bundle = {}
        try:
            self._append_z2_trace(
                "restore-session-missing-candidate",
                (
                    f"source={missing_source_dir} "
                    f"existing={len(annotations)} "
                    f"missing={len(missing_bundle)} "
                    f"run={run_dir}"
                ),
            )
        except Exception:
            pass
        if missing_bundle:
            try:
                self._merge_preview_annotation_bundle(missing_bundle)
                annotations = list(self.current_annotations or [])
                self._append_z2_trace(
                    "restore-session-missing",
                    f"restored={len(missing_bundle)} total={len(annotations)} run={run_dir}",
                )
            except Exception:
                pass
    try:
        self._restore_campaign_step2_generated_from_run(run_dir, only_when_pending=True)
    except Exception:
        pass
    self._sync_preview_approval_flags_from_current_sets()
    restore_idx = self._compute_annotation_run_restore_index(
        annotations,
        manifest,
        restore_filename=str(getattr(self, "_preview_session_restore_filename", "") or "").strip(),
        restore_index=getattr(self, "_preview_session_restore_index", None),
    )

    self.current_preview_index = restore_idx
    self._preview_session_restore_index = restore_idx
    self._preview_session_restore_filename = (
        str(getattr(annotations[restore_idx], "filename", "") or "")
        if restore_idx is not None and 0 <= int(restore_idx) < len(annotations)
        else ""
    )
    defer_heavy_refresh = bool(getattr(self, "_campaign_project_restore_in_progress", False))
    self._append_z2_trace(
        "restore-preview-run",
        f"run={run_dir} annotations={len(annotations)} deferred={int(defer_heavy_refresh)}",
    )
    self._refresh_preview_list(preserve_selection=True, render_current=not defer_heavy_refresh)
    if defer_heavy_refresh:
        try:
            self._campaign_restore_pending_preview_load = True
        except Exception:
            pass
        return True

    self._load_current_preview_selection(reset_view=True, selection_changed=True)
    self._refresh_step2_action_states()
    self._update_preview_edit_status("Przywrocono ostatni run anotacji Z2 z poprzedniej sesji.")
    return True

def _apply_free_mode_session_snapshot(self, session_state: dict | None = None, restore_preview: bool = True):
    state = self._sanitize_free_mode_session_snapshot(session_state or self._load_free_mode_session_snapshot())
    self._free_mode_session_restore_in_progress = True
    try:
        restored_route = self._normalize_workflow_route_value(state.get("workflow_route"))
        restored_manual_entry_mode = self._normalize_manual_entry_mode(state.get("manual_entry_mode"))
        safe_output_dir = self._coerce_annotation_output_dir(state.get("output_dir"))
        safe_run_dir = self._resolve_safe_annotation_run_dir(state.get("plate_dataset_run"), require_xml=True)
        safe_last_preview_run_dir = self._resolve_safe_annotation_run_dir(state.get("last_preview_run_dir"), require_xml=True)
        self.input_dir_var.set(str(state.get("input_dir") or self._annotation_session_defaults()["input_dir"]))
        self.output_dir_var.set(str(safe_output_dir))
        self.mode_var.set(self._normalize_mode_value(state.get("mode")))
        self.vehicle_model_var.set(str(state.get("vehicle_model") or "").strip())
        self.vehicle_custom_var.set(str(state.get("vehicle_custom") or "").strip())
        self.plate_custom_var.set(str(state.get("plate_custom") or "").strip())
        self.character_model_var.set(str(state.get("character_model") or "Brak / OCR").strip() or "Brak / OCR")
        self.character_custom_var.set(str(state.get("character_custom") or "").strip())
        self.device_var.set(str(state.get("device") or "auto").strip() or "auto")
        self.conf_var.set(float(state.get("conf", CONFIG.DEFAULT_CONFIDENCE)))
        self.plate_dataset_run_var.set(str(safe_run_dir or ""))
        self.plate_dataset_images_var.set(str(state.get("plate_dataset_images") or "").strip())
        self.plate_train_pct.set(float(state.get("plate_train_pct", 80.0)))
        self.plate_val_pct.set(float(state.get("plate_val_pct", 10.0)))
        self.manual_xml_template_var.set(bool(state.get("manual_xml_template", False)))
        self.manual_vehicle_assist_var.set(bool(state.get("manual_vehicle_assist", False)))
        self.workflow_route_var.set(restored_route)
        self.manual_entry_mode_var.set(restored_manual_entry_mode)
        self.auto_vehicle_choice_var.set(self._normalize_auto_vehicle_choice(state.get("auto_vehicle_choice")))
        self.workflow_step_var.set(self._normalize_workflow_step_value(state.get("workflow_step")))
        self.free_mode_screen_var.set(
            self._normalize_free_mode_screen_value(state.get("free_mode_screen"))
        )
        self._manual_review_active = bool(state.get("manual_review_active", False))
        self._manual_review_from_auto = bool(state.get("manual_review_from_auto", False))
        self._manual_review_origin_route = self._normalize_workflow_route_value(
            state.get("manual_review_origin_route")
        )
        self._last_completed_workflow_route = self._normalize_workflow_route_value(
            state.get("last_completed_workflow_route")
        )
        self._manual_review_export_ready = bool(state.get("manual_review_export_ready", False))
        self._manual_review_history_entries = self._normalize_manual_review_history_entries(
            state.get("manual_review_history", [])
        )

        preview_run_manifest = {}
        try:
            preview_run_manifest = self._load_annotation_run_manifest(safe_last_preview_run_dir or safe_run_dir)
        except Exception:
            preview_run_manifest = {}
        preview_run_is_manual_template = bool(
            self._annotation_run_manifest_is_manual_template(preview_run_manifest)
        )

        if restored_route == "manual" and restored_manual_entry_mode == "continue" and preview_run_is_manual_template:
            self._manual_review_active = False
            self._manual_review_from_auto = False
            self._manual_review_origin_route = ""
            self._manual_review_export_ready = False
            self._clear_preview_editor_state(clear_dirty=True)
            self.current_annotations = []
            self.current_input_dir = None
            self.current_annotation_run_dir = None
            self.current_annotation_xml_path = None
            self._campaign_t06_entry_approval_baseline = None

        self.last_staging_run_dir = safe_last_preview_run_dir
        try:
            preview_restore_index = int(state.get("last_preview_index", -1))
        except (TypeError, ValueError):
            preview_restore_index = -1
        self._preview_session_restore_index = preview_restore_index
        self._preview_session_restore_filename = str(state.get("last_preview_filename") or "").strip()

        self.apply_global_yolo_device_choice(self.device_var.get())
        self._refresh_device_options()
        self._update_model_lists()

        if hasattr(self, "vehicle_combo"):
            try:
                vehicle_values = list(self.vehicle_combo["values"])
            except Exception:
                vehicle_values = []
            current_vehicle = str(self.vehicle_model_var.get() or "").strip()
            if vehicle_values and current_vehicle not in vehicle_values:
                preferred_vehicle = "yolo11s" if "yolo11s" in vehicle_values else vehicle_values[0]
                self.vehicle_model_var.set(preferred_vehicle)

        if hasattr(self, "character_combo"):
            self._refresh_character_model_choices()

        self._refresh_manual_review_history_ui()

        self._on_mode_change()
        self._on_vehicle_model_change()
        self._update_manual_xml_template_ui()
        self._update_plate_dataset_ratio_labels()
        self._refresh_plate_dataset_export_sources()
        self._set_campaign_paths_lock_state(False)

        input_dir_value = str(self.input_dir_var.get() or "").strip()
        self.current_input_dir = Path(input_dir_value) if input_dir_value else None

        if restore_preview and not (
            restored_route == "manual"
            and restored_manual_entry_mode == "continue"
            and preview_run_is_manual_template
        ):
            restored_preview = bool(self._restore_preview_from_session_run())
        else:
            restored_preview = False
        if restored_preview:
            try:
                restored_manifest = self._load_annotation_run_manifest(getattr(self, "current_annotation_run_dir", None))
            except Exception:
                restored_manifest = {}
            restored_manual_template = bool(
                self._annotation_run_manifest_is_manual_template(restored_manifest)
            )
            current_route = self._normalize_workflow_route_value(self.workflow_route_var.get())
            current_screen = self._normalize_free_mode_screen_value(self.free_mode_screen_var.get())
            if not current_route:
                if restored_manual_template:
                    self.workflow_route_var.set("manual")
                    self.free_mode_screen_var.set("manual_review")
                    self._manual_review_active = True
                    self._manual_review_from_auto = False
                    self._manual_review_origin_route = "manual"
                    self._last_completed_workflow_route = "manual"
                else:
                    self.workflow_route_var.set("auto")
                    if current_screen in {"", "route_choice", "workflow"}:
                        self.free_mode_screen_var.set("auto_summary")
                    self._manual_review_active = False
                    self._manual_review_from_auto = False
                    self._manual_review_origin_route = ""
                    self._last_completed_workflow_route = "auto"
        self._manual_review_active = bool(self._manual_review_active and restored_preview)
        self._manual_review_from_auto = bool(self._manual_review_from_auto and self._manual_review_active)
        self._manual_review_origin_route = (
            self._manual_review_origin_route
            if self._manual_review_active
            else ""
        )
        self._manual_review_export_ready = bool(self._manual_review_export_ready and self._manual_review_active)
        self._repair_restored_z2_workflow_completion_state(
            state.get("last_completed_workflow_route")
        )
        if restored_preview and self._get_z2_thematic_route() == "auto":
            try:
                self._restore_plate_model_selection_from_active_run()
            except Exception:
                pass
    finally:
        self._free_mode_session_restore_in_progress = False
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()

def _prepare_campaign_source_preview_payload(
    self,
    input_dir: Path | None,
    *,
    include_previous: bool,
    current_manual_bundle: dict[str, tuple[ImageAnnotation, Path]] | None = None,
    approved_filenames: set[str] | None = None,
    is_cancelled=None,
) -> dict | None:
    if input_dir is None:
        return None

    try:
        source_dir = Path(input_dir)
    except Exception:
        return None

    source_plan = self._collect_campaign_auto_annotation_sources(
        source_dir,
        include_previous=bool(include_previous),
        exclude_manual_touched=False,
        is_cancelled=is_cancelled,
    )
    if callable(is_cancelled) and is_cancelled():
        return None
    safe_manual_bundle = dict(current_manual_bundle or {})
    approved_names = {
        str(name or "").strip().lower()
        for name in set(approved_filenames or set())
        if str(name or "").strip()
    }
    original_total_count = int(source_plan.get("total_count", 0) or 0)
    original_base_count = int(source_plan.get("base_count", 0) or 0)
    if safe_manual_bundle:
        image_paths = list(source_plan.get("image_paths") or [])
        image_map = dict(source_plan.get("image_map") or {})
        appended_manual = 0
        for filename, (_ann, image_path) in safe_manual_bundle.items():
            if callable(is_cancelled) and is_cancelled():
                return None
            safe_name = str(filename or "").strip()
            if not safe_name:
                continue
            if safe_name.lower() in approved_names:
                continue
            if safe_name in image_map:
                continue
            image_paths.append(image_path)
            image_map[safe_name] = image_path
            appended_manual += 1
        if appended_manual > 0:
            source_plan["image_paths"] = image_paths
            source_plan["image_map"] = image_map
            source_plan["total_count"] = original_total_count
            source_plan["base_count"] = original_base_count
            source_plan["current_manual_count"] = appended_manual

    image_paths = list(source_plan.get("image_paths") or [])
    if not image_paths:
        return {
            "source_dir": source_dir,
            "source_plan": source_plan,
            "preview_annotations": [],
            "image_path_map": {},
        }

    previous_annotations_by_name = {}
    if bool(include_previous):
        try:
            previous_bundle = self._get_campaign_previous_manual_source_bundle()
            previous_xml_path = previous_bundle.get("xml_path")
            if isinstance(previous_xml_path, Path) and previous_xml_path.exists():
                previous_annotations = self._parse_cvat_preview_annotations(previous_xml_path)
                if callable(is_cancelled) and is_cancelled():
                    return None
                previous_annotations_by_name = {
                    str(getattr(ann, "filename", "") or "").strip(): ann
                    for ann in previous_annotations
                    if str(getattr(ann, "filename", "") or "").strip()
                }
        except Exception as e:
            logger.debug(f"Nie udało się wczytać poprzednich ręcznych anotacji do podglądu kampanijnego Z2: {e}")

    reused_filenames = set(source_plan.get("reused_filenames") or set())
    preview_annotations: list[ImageAnnotation] = []
    for image_path in image_paths:
        if callable(is_cancelled) and is_cancelled():
            return None
        filename = image_path.name
        current_manual_entry = safe_manual_bundle.get(filename)
        if current_manual_entry is not None:
            normalized_current_ann = copy.deepcopy(current_manual_entry[0])
            normalized_current_ann.filename = filename
            try:
                width, height = get_image_size(image_path)
                normalized_current_ann.width = max(1, int(width))
                normalized_current_ann.height = max(1, int(height))
            except Exception:
                pass
            preview_annotations.append(normalized_current_ann)
            continue
        previous_ann = previous_annotations_by_name.get(filename)
        if previous_ann is not None and filename in reused_filenames:
            normalized_previous_ann = copy.deepcopy(previous_ann)
            normalized_previous_ann.filename = filename
            try:
                width, height = get_image_size(image_path)
                normalized_previous_ann.width = max(1, int(width))
                normalized_previous_ann.height = max(1, int(height))
            except Exception:
                pass
            preview_annotations.append(normalized_previous_ann)
        else:
            # Do not probe image dimensions for every empty work item during
            # campaign entry. Large folders can contain thousands of files; the
            # real size is filled lazily when the image is opened on canvas.
            preview_annotations.append(
                ImageAnnotation(
                    filename=filename,
                    width=1,
                    height=1,
                    detections=[],
                    status=AnnotationStatus.NO_PLATE,
                    status_message="Obraz źródłowy gotowy do przygotowania XML lub autoanotacji.",
                )
            )

    return {
        "source_dir": source_dir,
        "source_plan": source_plan,
        "preview_annotations": preview_annotations,
        "image_path_map": {
            str(name): Path(path)
            for name, path in dict(source_plan.get("image_map") or {}).items()
        },
    }

def _apply_campaign_source_preview_payload(self, payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False

    source_dir = payload.get("source_dir")
    source_plan = dict(payload.get("source_plan") or {})
    preview_annotations = list(payload.get("preview_annotations") or [])
    image_path_map = {
        str(name): Path(path)
        for name, path in dict(payload.get("image_path_map") or {}).items()
    }

    self.current_input_dir = source_dir if isinstance(source_dir, Path) else None
    self._preview_image_path_map = image_path_map
    self._apply_campaign_manual_reuse_context(source_plan)
    self.current_annotations = preview_annotations
    self._campaign_hidden_project_approved_filenames = set()
    self._campaign_hidden_char_effective_filenames = set()
    self._campaign_hidden_project_approved_count = int(source_plan.get("approved_skip_count", 0) or 0)
    self._campaign_hidden_char_effective_count = int(source_plan.get("char_effective_skip_count", 0) or 0)
    self._campaign_iteration_manual_filenames = set()
    approved_lookup = set(self._get_preview_approved_filenames())
    for ann in preview_annotations:
        try:
            setattr(
                ann,
                "_approved_for_training",
                str(getattr(ann, "filename", "") or "").strip().lower() in approved_lookup,
            )
        except Exception:
            continue
    self.current_preview_index = None

    try:
        approved_skip_count = int(source_plan.get("approved_skip_count", 0) or 0)
        char_effective_skip_count = int(source_plan.get("char_effective_skip_count", 0) or 0)
        reused_count = int(source_plan.get("reused_count", 0) or 0)
        current_manual_count = int(source_plan.get("current_manual_count", 0) or 0)
    except Exception:
        approved_skip_count = 0
        char_effective_skip_count = 0
        reused_count = 0
        current_manual_count = 0

    if not preview_annotations:
        try:
            self._refresh_preview_list(preserve_selection=False, render_current=True)
        except Exception:
            pass
        try:
            self._set_progress_counters(0, 0, 0)
        except Exception:
            pass
        try:
            self._set_status_label_state(
                (
                    "Ten zestaw nie zawiera już obrazów oczekujących na pracę w Z2. "
                    f"Wszystkie obrazy z tego wejścia są już w zatwierdzonym zbiorze projektu ({approved_skip_count})."
                    if approved_skip_count > 0
                    else (
                        f"Ten zestaw nie zawiera już obrazów oczekujących na pracę w Z2. "
                        f"{char_effective_skip_count} obrazów zostało już użytych w aktywnym E3."
                        if char_effective_skip_count > 0
                        else "Ten zestaw nie zawiera już obrazów oczekujących na pracę w Z2. Wszystkie zdjęcia z tego wejścia są już w zatwierdzonym zbiorze projektu."
                    )
                ),
                "info",
            )
        except Exception:
            pass
        try:
            self._refresh_preview_workspace_visibility(manual_review_active=False)
        except Exception:
            pass
        return True

    def _set_loaded_preview_status() -> None:
        status_text = (
            f"Wczytano {len(preview_annotations)} obrazów do pracy w Z2. Kliknij {self._get_step2_start_action_reference()}, aby przygotować XML albo uruchomić autoanotację."
        )
        if approved_skip_count > 0:
            status_text += (
                f" {approved_skip_count} obrazów jest już zatwierdzonych w projekcie i nie wraca do tego zestawu."
            )
        if char_effective_skip_count > 0:
            status_text += (
                f" {char_effective_skip_count} obrazów jest już użytych w aktywnym E3 i nie wraca do tego zestawu."
            )
        if reused_count > 0:
            status_text += (
                f" Dołączono też {reused_count} obrazów z wcześniejszej ręcznej anotacji, bo nie są jeszcze zatwierdzone w projekcie."
            )
        if current_manual_count > 0:
            status_text += (
                f" Zachowano też {current_manual_count} obrazów już poprawionych ręcznie w tej iteracji; pozostają na liście, ale autoanotacja ich nie ruszy."
            )
        self._set_status_label_state(status_text, "neutral")

    used_async_population = False
    try:
        used_async_population = bool(
            self._populate_preview_list(on_complete=_set_loaded_preview_status)
        )
    except Exception:
        self._refresh_preview_list(preserve_selection=False, render_current=True)
        _set_loaded_preview_status()
    if used_async_population:
        try:
            self._set_status_label_state(
                (
                    f"Przygotowuję listę {len(preview_annotations)} obrazów do pracy w Z2. "
                    "Pozycje doładują się jeszcze chwilę w tle."
                ),
                "neutral",
            )
        except Exception:
            pass
    try:
        self._set_progress_counters(0, 0, len(preview_annotations))
    except Exception:
        pass
    try:
        self._refresh_preview_workspace_visibility(manual_review_active=False)
    except Exception:
        pass
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    return True

def _prepare_annotation_run_restore_payload(
    self,
    run_dir: Path | None,
    *,
    candidate_dir_values: list[str] | None = None,
    restore_filename: str = "",
    restore_index: int | None = None,
    is_cancelled=None,
) -> dict | None:
    prepare_started = time.perf_counter()
    phase_started = prepare_started
    slow_phases: list[str] = []

    def _mark_prepare_phase(name: str) -> None:
        nonlocal phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - phase_started) * 1000.0
            if elapsed_ms >= 250.0:
                slow_phases.append(f"{name}={elapsed_ms:.0f}ms")
            phase_started = now
        except Exception:
            pass

    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return None
    if callable(is_cancelled) and is_cancelled():
        return None

    xml_path = safe_run_dir / "annotations.xml"
    manifest = self._load_annotation_run_manifest(safe_run_dir)
    _mark_prepare_phase("manifest")
    run_is_auto_annotation_for_origin = bool(
        str(manifest.get("annotation_run_type") or "").strip().lower() == "auto_annotation"
    )
    if callable(is_cancelled) and is_cancelled():
        return None
    annotations = self._parse_cvat_preview_annotations(xml_path)
    if run_is_auto_annotation_for_origin:
        try:
            self._mark_auto_plate_origin_for_annotations(annotations)
        except Exception:
            pass
    _mark_prepare_phase("parse_xml")
    if not annotations:
        return None
    if callable(is_cancelled) and is_cancelled():
        return None

    image_dir = self._resolve_run_image_dir_for_annotations(
        annotations,
        manifest,
        safe_run_dir,
        candidate_dir_values=candidate_dir_values,
    )
    _mark_prepare_phase("resolve_image_dir")
    if image_dir is None:
        return None
    if callable(is_cancelled) and is_cancelled():
        return None

    image_map: dict[str, Path] = {}
    missing_restored_count = 0
    t02_at_review_context = _is_t02_at_review_restore_context(self)
    hidden_char_effective_filenames = set()
    if not self._is_free_mode_session_context() and not t02_at_review_context:
        try:
            hidden_char_effective_filenames = self._get_campaign_char_effective_source_hidden_filenames()
        except Exception:
            hidden_char_effective_filenames = set()
    _mark_prepare_phase("hidden_char_effective")
    try:
        run_is_manual_template = bool(self._annotation_run_manifest_is_manual_template(manifest))
    except Exception:
        run_is_manual_template = False
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
    except Exception:
        graph_gate_id = ""
    successful_run_image_count = 0
    try:
        successful_run_image_count = int(manifest.get("result_successful_images", 0) or 0)
    except Exception:
        successful_run_image_count = 0
    expected_run_image_count = 0
    for count_key in (
        "input_scope_count",
        "source_plan_total_count",
        "selected_count",
        "image_count",
        "result_total_images",
    ):
        try:
            expected_run_image_count = int(manifest.get(count_key, 0) or 0)
        except Exception:
            expected_run_image_count = 0
        if expected_run_image_count > 0:
            break
    scoped_input_run = _annotation_run_uses_scoped_input(manifest)
    skip_missing_scan = bool(
        not scoped_input_run
        and expected_run_image_count > 0
        and len(annotations) >= expected_run_image_count
    )
    missing_count_hint = (
        max(0, int(expected_run_image_count or 0) - int(len(annotations)))
        if expected_run_image_count > 0
        else 0
    )
    skip_t06_successful_scan = bool(
        graph_gate_id == "T05"
        and successful_run_image_count > 0
        and len(annotations) >= successful_run_image_count
    )
    if skip_t06_successful_scan and not scoped_input_run:
        skip_missing_scan = True
        _mark_prepare_phase("skip_missing_t06_successful_run")
    skip_large_nearly_complete_scan = bool(
        expected_run_image_count >= 3000
        and len(annotations) >= 3000
        and 0 < missing_count_hint <= 25
    )
    if skip_large_nearly_complete_scan and not scoped_input_run:
        skip_missing_scan = True
        _mark_prepare_phase("skip_missing_nearly_complete_large_run")
    if t02_at_review_context:
        skip_missing_scan = True
        _mark_prepare_phase("skip_missing_t02_at_review")
    if skip_missing_scan:
        _mark_prepare_phase("skip_missing_complete_run")
    if not run_is_manual_template and not skip_missing_scan:
        missing_source_dir = self._resolve_existing_dir(
            manifest.get("source_input_dir")
            or manifest.get("imported_source_input_dir")
            or manifest.get("input_dir")
        )
        if missing_source_dir is None:
            missing_source_dir = image_dir
        if callable(is_cancelled) and is_cancelled():
            return None
        scope_filenames = _annotation_run_input_scope_filenames(manifest)
        try:
            missing_bundle = self._build_missing_preview_annotations_bundle(
                missing_source_dir,
                existing_annotations=annotations,
                extra_hidden_filenames=hidden_char_effective_filenames,
                scope_filenames=scope_filenames,
            )
        except Exception:
            missing_bundle = {}
        _mark_prepare_phase("missing_bundle")
        if callable(is_cancelled) and is_cancelled():
            return None
        if missing_bundle:
            annotations, image_map, missing_restored_count = self._merge_annotation_bundle_into_payload(
                annotations,
                image_map,
                missing_bundle,
            )
            _mark_prepare_phase("merge_missing")
        try:
            self._append_z2_trace(
                "prepare-restore-missing",
                (
                    f"source={missing_source_dir} "
                    f"existing={len(annotations) - int(missing_restored_count or 0)} "
                    f"missing={int(missing_restored_count or 0)} "
                    f"run={safe_run_dir}"
                ),
            )
        except Exception:
            pass

    resolved_restore_index = self._compute_annotation_run_restore_index(
        annotations,
        manifest,
        restore_filename=restore_filename,
        restore_index=restore_index,
    )
    _mark_prepare_phase("restore_index")
    if callable(is_cancelled) and is_cancelled():
        return None
    approved_filenames = set(self._load_annotation_run_approved_filenames(safe_run_dir))
    _mark_prepare_phase("approved_filenames")
    total_ms = (time.perf_counter() - prepare_started) * 1000.0
    if total_ms >= 500.0:
        try:
            logger.info(
                "[Z2 PERF] prepare_restore_payload total=%.0fms annotations=%s missing=%s phases=[%s] run=%s",
                total_ms,
                len(annotations),
                int(missing_restored_count or 0),
                ", ".join(slow_phases) if slow_phases else "no_slow_phase",
                safe_run_dir,
            )
        except Exception:
            pass

    return {
        "run_dir": safe_run_dir,
        "xml_path": xml_path,
        "manifest": manifest,
        "annotations": annotations,
        "image_dir": image_dir,
        "image_map": image_map,
        "missing_restored_count": int(missing_restored_count or 0),
        "restore_index": resolved_restore_index,
        "restore_filename": (
            str(getattr(annotations[resolved_restore_index], "filename", "") or "")
            if resolved_restore_index is not None and 0 <= int(resolved_restore_index) < len(annotations)
            else ""
        ),
        "approved_filenames": approved_filenames,
    }
