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
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
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

NAV_BUTTON_WIDTH = 18
YOLO = None


def _build_campaign_char_effective_source(self) -> dict:
    if self._is_free_mode_session_context():
        return {}

    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {}

    if not CAMPAIGN.get_active_project_name():
        return {}

    state_dir = CAMPAIGN.get_project_state_dir()
    if state_dir is None:
        return {}
    state_dir = Path(state_dir)
    export_root = state_dir / "char_effective_source"
    images_dir = export_root / "images"
    xml_path = export_root / "annotations.xml"
    state_path = export_root / "source_state.json"
    default_display_name = "Zbiór projektu tablic + bieżące [OK]"
    preview_source_context = self._get_campaign_step3_preview_source_context()
    preview_source_filenames = {
        str(name or "").strip().lower()
        for name in set(preview_source_context.get("filenames") or set())
        if str(name or "").strip()
    }
    preview_source_dir = preview_source_context.get("preview_dir")
    preview_meta_path = preview_source_context.get("meta_path")

    contributor_run_dir = None
    for raw_candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
    ):
        candidate = self._resolve_safe_annotation_run_dir(raw_candidate, require_xml=True)
        if candidate is None:
            continue
        try:
            if self._path_is_within(candidate, export_root):
                continue
        except Exception:
            pass
        contributor_run_dir = candidate
        break

    try:
        approved_manifest_path = CAMPAIGN.get_plate_approved_set_path()
    except Exception:
        approved_manifest_path = None

    contributor_manifest_path = (
        self._annotation_run_manifest_path(contributor_run_dir)
        if contributor_run_dir is not None
        else None
    )
    contributor_annotations_path = (
        contributor_run_dir / "annotations.xml"
        if contributor_run_dir is not None
        else None
    )
    contributor_approved_token = ""
    if contributor_run_dir is not None:
        try:
            if (
                getattr(self, "current_annotation_run_dir", None) is not None
                and self.current_annotations
                and self._paths_equivalent(self.current_annotation_run_dir, contributor_run_dir)
            ):
                contributor_approved_values = sorted(
                    str(name or "").strip().lower()
                    for name in self._get_preview_approved_filenames()
                    if str(name or "").strip()
                )
            else:
                contributor_approved_values = sorted(
                    str(name or "").strip().lower()
                    for name in self._load_annotation_run_approved_filenames(contributor_run_dir)
                    if str(name or "").strip()
                )
            contributor_approved_token = json.dumps(contributor_approved_values, ensure_ascii=False)
        except Exception:
            contributor_approved_token = ""

    source_state = {
        "builder_version": 4,
        "approved_manifest_token": self._build_path_change_token(approved_manifest_path),
        "contributor_run_dir": str(contributor_run_dir or ""),
        "contributor_manifest_token": self._build_path_change_token(contributor_manifest_path),
        "contributor_annotations_token": self._build_path_change_token(contributor_annotations_path),
        "contributor_approved_token": contributor_approved_token,
    }

    existing_state = {}
    try:
        if state_path.exists():
            existing_state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        existing_state = {}

    if (
        isinstance(existing_state, dict)
        and all(existing_state.get(key) == value for key, value in source_state.items())
        and export_root.exists()
        and images_dir.exists()
        and xml_path.exists()
    ):
        try:
            self._sync_campaign_iteration_artifact_registry(
                extra_updates={
                    "char_effective_source": {
                        "run_dir": str(export_root.resolve()),
                        "images_dir": str(images_dir.resolve()),
                        "xml_path": str(xml_path.resolve()),
                        "state_path": str(state_path.resolve()),
                        "entries_count": int(existing_state.get("entries_count", 0) or 0),
                        "images_with_plates": int(existing_state.get("images_with_plates", 0) or 0),
                        "total_plates": int(existing_state.get("total_plates", 0) or 0),
                        "display_name": str(existing_state.get("display_name") or default_display_name),
                        "contributor_run_dir": str(contributor_run_dir or ""),
                        "artifact_iteration": int(existing_state.get("artifact_iteration", 0) or 0),
                        "source_iteration": int(existing_state.get("source_iteration", 0) or 0),
                    }
                },
            )
        except Exception:
            pass
        return {
            "run_dir": export_root,
            "images_dir": images_dir,
            "xml_path": xml_path,
            "entries_count": int(existing_state.get("entries_count", 0) or 0),
            "images_with_plates": int(existing_state.get("images_with_plates", 0) or 0),
            "total_plates": int(existing_state.get("total_plates", 0) or 0),
            "display_name": str(existing_state.get("display_name") or default_display_name),
            "contributor_run_dir": str(contributor_run_dir or ""),
            "artifact_iteration": int(existing_state.get("artifact_iteration", 0) or 0),
            "source_iteration": int(existing_state.get("source_iteration", 0) or 0),
        }

    approved_entries = list(CAMPAIGN.list_plate_approved_entries() or [])
    # Preview PZ2 jest wynikiem wycinania z ApprovedSet, więc nie może wracać
    # jako kolejne źródło wejściowe. Inaczej każde odtworzenie E3 mnoży cropy
    # tablic: ApprovedSet -> preview -> nowe źródło -> kolejny preview.
    preview_entries: list[dict] = []

    contributor_entries: list[dict] = []
    contributor_approved_lookup: set[str] = set()
    if contributor_run_dir is not None:
        try:
            contributor_entries = list(
                self._build_campaign_plate_approved_entries_from_run(
                    contributor_run_dir,
                    extra_included_filenames=preview_source_filenames,
                ) or []
            )
        except Exception:
            contributor_entries = []
        try:
            if (
                getattr(self, "current_annotation_run_dir", None) is not None
                and self.current_annotations
                and self._paths_equivalent(self.current_annotation_run_dir, contributor_run_dir)
            ):
                contributor_approved_lookup = {
                    str(name or "").strip().lower()
                    for name in sorted(self._get_preview_approved_filenames())
                    if str(name or "").strip()
                }
            else:
                contributor_approved_lookup = {
                    str(name or "").strip().lower()
                    for name in sorted(self._load_annotation_run_approved_filenames(contributor_run_dir))
                    if str(name or "").strip()
                }
        except Exception:
            contributor_approved_lookup = set()
        if preview_source_filenames:
            contributor_approved_lookup |= set(preview_source_filenames)
        if contributor_approved_lookup:
            filtered_entries: list[dict] = []
            for entry in contributor_entries:
                if not isinstance(entry, dict):
                    continue
                image_name = str(entry.get("image_name", "") or "").strip().lower()
                entry_key = self._get_campaign_plate_entry_merge_key(entry)
                if image_name in contributor_approved_lookup or entry_key in contributor_approved_lookup:
                    filtered_entries.append(entry)
            contributor_entries = filtered_entries
        else:
            contributor_entries = []

    merged_entries: dict[str, dict] = {}
    for entry in approved_entries:
        if not isinstance(entry, dict):
            continue
        entry_key = self._get_campaign_plate_entry_merge_key(entry)
        if not entry_key:
            continue
        merged_entries[entry_key] = dict(entry)

    for entry in preview_entries:
        if not isinstance(entry, dict):
            continue
        entry_key = self._get_campaign_plate_entry_merge_key(entry)
        if not entry_key:
            continue
        merged_entries[entry_key] = dict(entry)

    for entry in contributor_entries:
        if not isinstance(entry, dict):
            continue
        entry_key = self._get_campaign_plate_entry_merge_key(entry)
        if not entry_key:
            continue
        merged_entries[entry_key] = dict(entry)

    merged_list = list(merged_entries.values())
    if not merged_list:
        try:
            if export_root.exists() and self._path_is_within(export_root, state_dir):
                shutil.rmtree(export_root)
        except Exception:
            pass
        return {}

    try:
        if export_root.exists() and self._path_is_within(export_root, state_dir):
            shutil.rmtree(export_root)
    except Exception as e:
        logger.debug(f"Nie udało się wyczyścić tymczasowego źródła znaków z ApprovedSet: {e}")

    images_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[ImageAnnotation] = []
    used_names: set[str] = set()
    copied = 0
    images_with_plates = 0
    total_plates = 0
    source_iterations: list[int] = []
    for entry in merged_list:
        if not isinstance(entry, dict):
            continue
        for field in ("first_approved_iteration", "approved_iteration", "source_iteration", "iteration"):
            try:
                iteration_num = int(entry.get(field, 0) or 0)
            except Exception:
                iteration_num = 0
            if iteration_num > 0:
                source_iterations.append(iteration_num)
                break
    try:
        artifact_iteration = max(source_iterations) if source_iterations else int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        artifact_iteration = 0

    for entry in merged_list:
        image_name = str(entry.get("image_name", "") or "").strip()
        source_image_path = str(entry.get("source_image_path", "") or "").strip()
        if not image_name or not source_image_path:
            continue

        try:
            source_path = Path(source_image_path)
        except Exception:
            continue
        if not source_path.exists():
            continue

        export_name = image_name
        if export_name in used_names:
            stem = Path(image_name).stem or "image"
            suffix = Path(image_name).suffix or ".jpg"
            counter = 1
            while export_name in used_names:
                export_name = f"{stem}__{counter:03d}{suffix}"
                counter += 1
        used_names.add(export_name)

        try:
            shutil.copy2(source_path, images_dir / export_name)
            copied += 1
        except Exception as e:
            logger.debug(f"Nie udało się skopiować obrazu do źródła znaków ({source_path}): {e}")
            continue

        detections: list[Detection] = []
        for plate_entry in list(entry.get("plates") or []):
            polygon = [
                (float(point[0]), float(point[1]))
                for point in list(plate_entry.get("polygon") or [])[:4]
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(polygon) < 4:
                continue
            attributes = dict(plate_entry.get("attributes") or {})
            if str(entry.get("annotation_origin", "") or "").strip().lower() == "manual":
                attributes.setdefault("manually_edited", "true")
                attributes.setdefault("manual_source", "campaign_approved")
            detections.append(
                Detection(
                    label="plate",
                    confidence=float(plate_entry.get("confidence", 1.0) or 1.0),
                    bbox=self._bbox_from_polygon(polygon),
                    keypoints=self._keypoints_from_polygon(polygon),
                    polygon=polygon,
                    attributes=attributes,
                )
            )

        if not detections:
            continue

        images_with_plates += 1
        total_plates += len(detections)
        annotations.append(
            ImageAnnotation(
                filename=export_name,
                width=max(1, int(entry.get("width", 0) or 0)),
                height=max(1, int(entry.get("height", 0) or 0)),
                detections=detections,
                status=AnnotationStatus.SUCCESS,
            )
        )

    if not annotations or copied <= 0:
        return {}

    if not CVATExporter(task_name="Campaign Char Effective Source").export(
        annotations,
        xml_path,
        include_confidence=True,
        only_successful=False,
    ):
        return {}

    try:
        source_state.update(
            {
                "entries_count": int(len(annotations)),
                "images_with_plates": int(images_with_plates),
                "total_plates": int(total_plates),
                "display_name": default_display_name,
                "artifact_iteration": int(artifact_iteration or 0),
                "source_iteration": int(artifact_iteration or 0),
            }
        )
        state_path.write_text(json.dumps(source_state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        self._sync_campaign_iteration_artifact_registry(
            extra_updates={
                "char_effective_source": {
                    "run_dir": str(export_root.resolve()),
                    "images_dir": str(images_dir.resolve()),
                    "xml_path": str(xml_path.resolve()),
                    "state_path": str(state_path.resolve()),
                    "entries_count": int(len(annotations) or 0),
                    "images_with_plates": int(images_with_plates or 0),
                    "total_plates": int(total_plates or 0),
                    "display_name": default_display_name,
                    "contributor_run_dir": str(contributor_run_dir or ""),
                    "artifact_iteration": int(artifact_iteration or 0),
                    "source_iteration": int(artifact_iteration or 0),
                }
            },
        )
    except Exception:
        pass

    if graph_gate_id == "T05" and graph_display_gate_id != graph_gate_id:
        message = message.replace(graph_gate_id, graph_display_gate_id)
        detail = detail.replace(graph_gate_id, graph_display_gate_id)
        instruction = instruction.replace(graph_gate_id, graph_display_gate_id)
        gate_title = gate_title.replace(graph_gate_id, graph_display_gate_id)

    return {
        "run_dir": export_root,
        "images_dir": images_dir,
        "xml_path": xml_path,
        "entries_count": len(annotations),
        "images_with_plates": int(images_with_plates),
        "total_plates": int(total_plates),
        "display_name": default_display_name,
        "contributor_run_dir": str(contributor_run_dir or ""),
        "artifact_iteration": int(artifact_iteration or 0),
        "source_iteration": int(artifact_iteration or 0),
    }

def _get_campaign_auto_annotation_bootstrap(self, iteration_target: str | None = None) -> dict:
    bootstrap = {
        "input_dir": None,
        "input_source": "raw",
        "manual_template": False,
        "plate_model_path": "",
        "restore_run_dir": None,
    }

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return bootstrap

        target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
        raw_dir = CAMPAIGN.get_dir("raw")
        if raw_dir is None:
            return bootstrap

        iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        raw_root = Path(raw_dir)
        iter_dir = raw_root / f"Iteracja_{iter_num:03d}"
        try:
            default_input = CAMPAIGN.get_iteration_image_source_dir(iter_num) or iter_dir
        except Exception:
            default_input = iter_dir if iter_dir.exists() else raw_root
        auto_dir = CAMPAIGN.get_dir("auto_ann")
        staging_auto_dir = CAMPAIGN.get_staging_dir("auto_ann")
        project_start_mode = str(getattr(CAMPAIGN, "get_project_start_mode", lambda *_a, **_k: "fresh")() or "fresh").strip().lower()

        registry_bundle = self._get_campaign_iteration_artifact_bundle(
            images_dir=default_input or CAMPAIGN.get_iteration_image_source_dir(iter_num) or CAMPAIGN.get_master_pool_dir(),
            iteration_num=iter_num,
        )
        registry_plate_source = dict(registry_bundle.get("plate_source") or {})
        registry_step2_active_run = dict(registry_bundle.get("step2_active_run") or {})
        registry_plate_model = dict(registry_bundle.get("plate_model") or {})
        registry_image_source = dict(registry_bundle.get("image_source") or {})

        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
        registry_plate_model_path = str(registry_plate_model.get("path") or "").strip()
        if not plate_model_path or not Path(plate_model_path).exists():
            plate_model_path = registry_plate_model_path
        plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())

        bootstrap["input_dir"] = default_input
        bootstrap["plate_model_path"] = plate_model_path if plate_model_ready else ""
        bootstrap["manual_template"] = bool(target == "plate" and not plate_model_ready)

        registry_expected_image_token = str(registry_image_source.get("image_set_token") or "").strip()
        registry_plate_expected_token = str(registry_plate_source.get("expected_image_set_token") or "").strip()
        registry_plate_xml_token = str(registry_plate_source.get("xml_image_set_token") or "").strip()
        registry_plate_image_set_match = registry_plate_source.get("image_set_match")
        if registry_plate_image_set_match is None:
            registry_plate_source_usable = bool(
                not registry_expected_image_token
                or not registry_plate_xml_token
                or registry_expected_image_token == registry_plate_xml_token
                or (registry_plate_expected_token and registry_plate_expected_token == registry_plate_xml_token)
            )
        else:
            registry_plate_source_usable = bool(registry_plate_image_set_match)

        registry_run_path = str(registry_plate_source.get("run_dir") or "").strip()
        registry_images_path = str(registry_plate_source.get("images_dir") or "").strip()
        active_run_path = str(registry_step2_active_run.get("run_dir") or "").strip()
        active_images_path = str(registry_step2_active_run.get("images_dir") or "").strip()

        def _matches_current_iteration_input(path_value) -> bool:
            if path_value is None:
                return False
            try:
                return self._paths_equivalent(path_value, default_input)
            except Exception:
                return False

        def _run_matches_current_iteration_input(run_dir) -> bool:
            try:
                return self._annotation_run_matches_expected_input_dir(run_dir, default_input)
            except Exception:
                return True

        def _registry_bundle_tracks_current_iteration(bundle: dict) -> bool:
            try:
                iterations = {
                    int(value)
                    for value in list((bundle or {}).get("iterations") or [])
                    if str(value).strip().isdigit()
                }
                if int(iter_num) in iterations:
                    return True
            except Exception:
                pass
            try:
                first_seen = int((bundle or {}).get("iteration_first_seen", 0) or 0)
                last_seen = int((bundle or {}).get("iteration_last_seen", 0) or 0)
                if first_seen and last_seen and first_seen <= int(iter_num) <= last_seen:
                    return True
            except Exception:
                pass
            return False

        def _run_has_unapproved_work(run_dir: Path | None) -> bool:
            safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
            if safe_run_dir is None:
                return False
            try:
                manifest = self._load_annotation_run_manifest(safe_run_dir)
            except Exception:
                manifest = {}
            approved_names = {
                str(name or "").strip().lower()
                for name in list(manifest.get("approved_filenames") or [])
                if str(name or "").strip()
            }
            total_images = 0
            for key in ("result_successful_images", "result_total_images", "input_scope_count"):
                try:
                    total_images = max(total_images, int(manifest.get(key, 0) or 0))
                except Exception:
                    pass
            if total_images > 0:
                return len(approved_names) < total_images
            try:
                images_with_plates, _total_plates = self._get_run_plate_annotation_counts(safe_run_dir)
            except Exception:
                images_with_plates = 0
            return int(images_with_plates or 0) > len(approved_names)

        registry_tracks_current_iteration = _registry_bundle_tracks_current_iteration(registry_bundle)
        registry_has_unapproved_work = False
        for raw_registry_run in (active_run_path, registry_run_path):
            if registry_has_unapproved_work:
                break
            try:
                registry_run_dir_for_work = Path(raw_registry_run) if str(raw_registry_run or "").strip() else None
            except Exception:
                registry_run_dir_for_work = None
            if registry_run_dir_for_work is not None:
                registry_has_unapproved_work = _run_has_unapproved_work(registry_run_dir_for_work)

        if (
            target == "plate"
            and int(iter_num or 1) > 1
            and (not registry_tracks_current_iteration or not registry_has_unapproved_work)
            and bootstrap.get("restore_run_dir") is None
        ):
            try:
                previous_bundle = self._get_campaign_previous_manual_source_bundle()
            except Exception:
                previous_bundle = {}
            previous_run_dir = self._resolve_safe_annotation_run_dir(
                (previous_bundle or {}).get("run_dir"),
                require_xml=True,
            )
            if (
                previous_run_dir is not None
                and _run_matches_current_iteration_input(previous_run_dir)
                and _run_has_unapproved_work(previous_run_dir)
            ):
                bootstrap["restore_run_dir"] = previous_run_dir
                bootstrap["input_source"] = "previous_iteration_working_run"
                bootstrap["manual_template"] = False
                previous_input_dir = (previous_bundle or {}).get("input_dir")
                try:
                    if previous_input_dir is not None and self._dir_has_images(previous_input_dir):
                        bootstrap["input_dir"] = previous_input_dir
                except Exception:
                    pass

        if active_images_path:
            try:
                active_images_dir = Path(active_images_path)
            except Exception:
                active_images_dir = None
            if (
                active_images_dir is not None
                and _matches_current_iteration_input(active_images_dir)
                and self._dir_has_images(active_images_dir)
            ):
                bootstrap["input_dir"] = active_images_dir
        if active_run_path and bootstrap.get("restore_run_dir") is None:
            try:
                active_run_dir = Path(active_run_path)
            except Exception:
                active_run_dir = None
            if (
                active_run_dir is not None
                and active_run_dir.exists()
                and active_run_dir.is_dir()
                and (active_run_dir / "annotations.xml").exists()
                and _run_matches_current_iteration_input(active_run_dir)
            ):
                bootstrap["restore_run_dir"] = active_run_dir
                bootstrap["input_source"] = "registry_step2_active_run"
                if target == "plate":
                    bootstrap["manual_template"] = False
        if bootstrap.get("restore_run_dir") is None:
            try:
                current_staging_run = self._resolve_safe_annotation_run_dir(
                    CAMPAIGN.get_step2_staging_run(),
                    require_xml=True,
                )
            except Exception:
                current_staging_run = None
            if (
                current_staging_run is not None
                and _run_matches_current_iteration_input(current_staging_run)
            ):
                bootstrap["restore_run_dir"] = current_staging_run
                bootstrap["input_source"] = "campaign_step2_staging_run"
                if target == "plate":
                    bootstrap["manual_template"] = False
        if registry_images_path:
            try:
                registry_images_dir = Path(registry_images_path)
            except Exception:
                registry_images_dir = None
            if (
                registry_images_dir is not None
                and _matches_current_iteration_input(registry_images_dir)
                and self._dir_has_images(registry_images_dir)
            ):
                bootstrap["input_dir"] = registry_images_dir
        if registry_plate_source_usable and registry_run_path and bootstrap.get("restore_run_dir") is None:
            try:
                registry_run_dir = Path(registry_run_path)
            except Exception:
                registry_run_dir = None
            if (
                registry_run_dir is not None
                and registry_run_dir.exists()
                and registry_run_dir.is_dir()
                and (registry_run_dir / "annotations.xml").exists()
                and _run_matches_current_iteration_input(registry_run_dir)
            ):
                bootstrap["restore_run_dir"] = registry_run_dir
                bootstrap["input_source"] = "registry_plate_source"
                if target == "plate":
                    bootstrap["manual_template"] = False

        manual_source_run = self._find_reused_manual_source_run(default_input)
        if (
            bootstrap.get("restore_run_dir") is None
            and manual_source_run is not None
            and (manual_source_run / "annotations.xml").exists()
        ):
            bootstrap["restore_run_dir"] = manual_source_run
            bootstrap["input_source"] = "manual_source_run"
            if target == "plate":
                bootstrap["manual_template"] = False
        elif bootstrap.get("restore_run_dir") is None and iter_num == 1 and project_start_mode == "assets":
            stored_manual_source = CAMPAIGN.get_last_plate_manual_source()
            for raw_candidate in (
                str(stored_manual_source.get("source_run_path") or "").strip(),
                str(stored_manual_source.get("source_xml_path") or "").strip(),
            ):
                if not raw_candidate:
                    continue
                try:
                    run_candidate = Path(raw_candidate)
                    if run_candidate.suffix.lower() == ".xml":
                        run_candidate = run_candidate.parent
                except Exception:
                    continue
                if run_candidate.exists() and run_candidate.is_dir() and (run_candidate / "annotations.xml").exists():
                    bootstrap["restore_run_dir"] = run_candidate
                    bootstrap["input_source"] = "project_imported_manual_source"
                    if target == "plate":
                        bootstrap["manual_template"] = False
                    break

            stored_input_path = str(stored_manual_source.get("source_input_path") or "").strip()
            if stored_input_path and not self._dir_has_images(default_input):
                try:
                    stored_input_dir = Path(stored_input_path)
                except Exception:
                    stored_input_dir = None
                if stored_input_dir is not None and self._dir_has_images(stored_input_dir):
                    bootstrap["input_dir"] = stored_input_dir
                    if bootstrap["input_source"] == "raw":
                        bootstrap["input_source"] = "project_imported_images"

        reuse_source_input = None
        try:
            ingest_manifest = CAMPAIGN.load_ingest_manifest(iter_num)
        except Exception:
            ingest_manifest = {}

        if isinstance(ingest_manifest, dict):
            selection_mode = str(ingest_manifest.get("selection_mode") or "").strip().lower()
            try:
                reused_from_iteration = int(ingest_manifest.get("reused_from_iteration", 0) or 0)
            except (TypeError, ValueError):
                reused_from_iteration = 0

            if selection_mode == "iteration_reuse" and reused_from_iteration > 0:
                reuse_source_input = CAMPAIGN.get_iteration_image_source_dir(reused_from_iteration)

            if (
                bootstrap.get("restore_run_dir") is None
                and selection_mode in {"pool_reuse", "stage_reuse", "iteration_reuse"}
                and reused_from_iteration > 0
            ):
                try:
                    previous_bundle = self._get_campaign_previous_manual_source_bundle()
                except Exception:
                    previous_bundle = {}

                previous_run_dir = self._resolve_safe_annotation_run_dir(
                    (previous_bundle or {}).get("run_dir"),
                    require_xml=True,
                )
                if previous_run_dir is not None:
                    bootstrap["restore_run_dir"] = previous_run_dir
                    bootstrap["input_source"] = "reused_previous_manual_source"
                    if target == "plate":
                        bootstrap["manual_template"] = False

        if bootstrap.get("restore_run_dir") is None and reuse_source_input is not None and auto_dir is not None:
            reuse_source_input = Path(reuse_source_input)
            manual_reuse_run = self._find_reused_manual_source_run(reuse_source_input)
            if manual_reuse_run is not None and (manual_reuse_run / "annotations.xml").exists():
                bootstrap["restore_run_dir"] = manual_reuse_run
                bootstrap["input_source"] = "reused_manual_source_run"
                if target == "plate":
                    bootstrap["manual_template"] = False
            else:
                training_source_run = self._find_reused_training_source_run(reuse_source_input)
                if training_source_run is not None and (training_source_run / "annotations.xml").exists():
                    bootstrap["restore_run_dir"] = training_source_run
                    bootstrap["input_source"] = "reused_training_source_run"
                    if target == "plate":
                        bootstrap["manual_template"] = False
                else:
                    reuse_run = self._find_latest_annotation_run_for_input(
                        reuse_source_input,
                        [Path(auto_dir)],
                    )
                    if reuse_run is not None and (reuse_run / "annotations.xml").exists():
                        bootstrap["restore_run_dir"] = reuse_run
                        bootstrap["input_source"] = "reused_iteration_run"
                        if target == "plate":
                            bootstrap["manual_template"] = False

        if bootstrap.get("restore_run_dir") is None and (auto_dir is not None or staging_auto_dir is not None):
            try:
                search_roots = [
                    Path(root)
                    for root in (staging_auto_dir, auto_dir)
                    if root is not None
                ]
                latest_for_input = self._find_latest_annotation_run_for_input(
                    Path(bootstrap.get("input_dir")) if bootstrap.get("input_dir") else None,
                    search_roots,
                )
            except Exception:
                latest_for_input = None
            if latest_for_input is not None and (latest_for_input / "annotations.xml").exists():
                bootstrap["restore_run_dir"] = latest_for_input
                bootstrap["input_source"] = "latest_approved_run"
                if target == "plate":
                    bootstrap["manual_template"] = False

        if target != "plate":
            return bootstrap

        if self._dir_has_images(default_input):
            return bootstrap

        current_stage = self._get_manual_plate_stage_images_dir()
        if self._dir_has_images(current_stage):
            bootstrap["input_dir"] = current_stage
            bootstrap["input_source"] = "stage_current_iteration"
            return bootstrap

        if iter_num > 1:
            prev_stage = self._get_manual_plate_stage_dir().parent / f"Iteracja_{iter_num - 1:03d}" / "images"
            if self._dir_has_images(prev_stage):
                bootstrap["input_dir"] = prev_stage
                bootstrap["input_source"] = "stage_previous_iteration"
                return bootstrap

        latest_run = None
        if auto_dir is not None:
            latest_run = self._find_latest_annotation_run_dir(Path(auto_dir))

        if latest_run is None:
            return bootstrap

        manifest = self._load_annotation_run_manifest(latest_run)
        manifest_input = str(manifest.get("input_dir") or "").strip()
        if manifest_input and self._dir_has_images(manifest_input):
            bootstrap["input_dir"] = Path(manifest_input)
            bootstrap["input_source"] = "latest_approved_run"
            bootstrap["restore_run_dir"] = latest_run
            if target == "plate":
                bootstrap["manual_template"] = False
            return bootstrap

        if (latest_run / "annotations.xml").exists():
            bootstrap["input_source"] = "latest_approved_run"
            bootstrap["restore_run_dir"] = latest_run
            if target == "plate":
                bootstrap["manual_template"] = False
    except Exception as e:
        logger.debug(f"Nie udalo sie zbudowac bootstrapu Z2 dla kampanii: {e}")

    return self._normalize_campaign_step2_bootstrap_manual_template(bootstrap)

def _collect_campaign_auto_annotation_sources(
    self,
    base_input_dir: Path | None,
    *,
    include_previous: bool | None = None,
    exclude_manual_touched: bool = False,
    is_cancelled=None,
) -> dict:
    plan = {
        "base_input_dir": None,
        "previous_manual_dir": None,
        "previous_xml_path": None,
        "previous_image_dirs": [],
        "source_label": "wcześniejszej iteracji",
        "image_paths": [],
        "image_map": {},
        "reused_filenames": set(),
        "reused_count": 0,
        "base_count": 0,
        "total_count": 0,
        "has_previous_manual": False,
        "approved_skip_count": 0,
        "char_effective_skip_count": 0,
        "manual_skip_count": 0,
        "manual_skip_filenames": set(),
        "manifest_scoped": False,
        "manifest_scope_count": 0,
    }

    try:
        base_dir = Path(base_input_dir) if base_input_dir is not None else None
    except Exception:
        base_dir = None

    previous_bundle = self._get_campaign_previous_manual_source_bundle()
    previous_manual_dir = previous_bundle.get("input_dir")
    if base_dir is not None:
        plan["base_input_dir"] = base_dir
    previous_xml_path = previous_bundle.get("xml_path")
    previous_image_dirs = [
        path
        for path in list(previous_bundle.get("image_dirs") or [])
        if isinstance(path, Path)
    ]

    if isinstance(previous_manual_dir, Path):
        plan["previous_manual_dir"] = previous_manual_dir
        plan["has_previous_manual"] = True
        plan["source_label"] = str(previous_bundle.get("source_label") or self._extract_iteration_label_from_path(previous_manual_dir))
    if isinstance(previous_xml_path, Path) and previous_xml_path.exists():
        plan["previous_xml_path"] = previous_xml_path
    if previous_image_dirs:
        plan["previous_image_dirs"] = previous_image_dirs

    if include_previous is None:
        include_previous = bool(self.campaign_reuse_manual_var.get())

    if callable(is_cancelled) and is_cancelled():
        return plan

    approved_filenames = self._get_campaign_plate_approved_filenames()
    approved_source_keys = self._get_campaign_plate_approved_source_keys()
    char_effective_hidden_filenames = self._get_campaign_char_effective_source_hidden_filenames()
    manual_touched_filenames = (
        self._get_campaign_manual_touched_filenames()
        if exclude_manual_touched
        else set()
    )
    manually_skipped_names: set[str] = set()
    manifest_expected_count = self._get_campaign_iteration_manifest_image_count()
    manifest_images = self._get_campaign_iteration_manifest_image_paths(base_dir)
    if manifest_expected_count > 0:
        base_images = manifest_images
    else:
        base_images = get_image_files(base_dir) if base_dir is not None and self._dir_has_images(base_dir) else []
    plan["manifest_scoped"] = bool(manifest_expected_count > 0 or manifest_images)
    plan["manifest_scope_count"] = int(len(manifest_images))
    plan["manifest_expected_count"] = int(manifest_expected_count)
    plan["manifest_missing_count"] = max(0, int(manifest_expected_count) - int(len(manifest_images)))
    if approved_filenames or approved_source_keys:
        filtered_base_images = []
        for image_path in base_images:
            if callable(is_cancelled) and is_cancelled():
                plan["image_paths"] = []
                plan["image_map"] = {}
                plan["reused_filenames"] = set()
                plan["reused_count"] = 0
                plan["base_count"] = 0
                plan["total_count"] = 0
                return plan
            image_name = str(image_path.name or "").strip().lower()
            source_key = self._build_campaign_source_image_key(image_path) if approved_source_keys else ""
            is_approved_match = bool(
                (source_key and source_key in approved_source_keys)
                or image_name in approved_filenames
            )
            if is_approved_match:
                plan["approved_skip_count"] += 1
                continue
            if image_name in char_effective_hidden_filenames:
                plan["char_effective_skip_count"] += 1
                continue
            if image_name in manual_touched_filenames:
                manually_skipped_names.add(image_name)
                continue
            filtered_base_images.append(image_path)
        base_images = filtered_base_images
    elif manual_touched_filenames or char_effective_hidden_filenames:
        filtered_base_images = []
        for image_path in base_images:
            if callable(is_cancelled) and is_cancelled():
                plan["image_paths"] = []
                plan["image_map"] = {}
                plan["reused_filenames"] = set()
                plan["reused_count"] = 0
                plan["base_count"] = 0
                plan["total_count"] = 0
                return plan
            image_name = str(image_path.name or "").strip().lower()
            if image_name in char_effective_hidden_filenames:
                plan["char_effective_skip_count"] += 1
                continue
            if image_name in manual_touched_filenames:
                manually_skipped_names.add(image_name)
                continue
            filtered_base_images.append(image_path)
        base_images = filtered_base_images
    previous_images = []
    if previous_manual_dir is not None:
        same_dir = False
        if base_dir is not None:
            try:
                same_dir = previous_manual_dir.resolve() == base_dir.resolve()
            except Exception:
                same_dir = str(previous_manual_dir) == str(base_dir)
        if not same_dir:
            annotated_filenames = self._load_cvat_plate_annotated_filenames(previous_xml_path)
            if annotated_filenames:
                resolved_previous_images: list[Path] = []
                for filename in annotated_filenames:
                    resolved_path = None
                    for image_dir in previous_image_dirs or ([previous_manual_dir] if previous_manual_dir is not None else []):
                        try:
                            candidate = Path(image_dir) / filename
                        except Exception:
                            continue
                        if candidate.exists():
                            resolved_path = candidate
                            break
                    if resolved_path is not None:
                        resolved_previous_images.append(resolved_path)
                if approved_filenames or approved_source_keys:
                    filtered_previous_images = []
                    for image_path in resolved_previous_images:
                        if callable(is_cancelled) and is_cancelled():
                            plan["image_paths"] = []
                            plan["image_map"] = {}
                            plan["reused_filenames"] = set()
                            plan["reused_count"] = 0
                            plan["base_count"] = 0
                            plan["total_count"] = 0
                            return plan
                        image_name = str(image_path.name or "").strip().lower()
                        source_key = self._build_campaign_source_image_key(image_path) if approved_source_keys else ""
                        is_approved_match = bool(
                            (source_key and source_key in approved_source_keys)
                            or image_name in approved_filenames
                        )
                        if is_approved_match:
                            plan["approved_skip_count"] += 1
                            continue
                        if image_name in char_effective_hidden_filenames:
                            plan["char_effective_skip_count"] += 1
                            continue
                        if image_name in manual_touched_filenames:
                            manually_skipped_names.add(image_name)
                            continue
                        filtered_previous_images.append(image_path)
                    previous_images = filtered_previous_images
                else:
                    if manual_touched_filenames or char_effective_hidden_filenames:
                        filtered_previous_images = []
                        for image_path in resolved_previous_images:
                            if callable(is_cancelled) and is_cancelled():
                                plan["image_paths"] = []
                                plan["image_map"] = {}
                                plan["reused_filenames"] = set()
                                plan["reused_count"] = 0
                                plan["base_count"] = 0
                                plan["total_count"] = 0
                                return plan
                            image_name = str(image_path.name or "").strip().lower()
                            if image_name in char_effective_hidden_filenames:
                                plan["char_effective_skip_count"] += 1
                                continue
                            if image_name in manual_touched_filenames:
                                manually_skipped_names.add(image_name)
                                continue
                            filtered_previous_images.append(image_path)
                        previous_images = filtered_previous_images
                    else:
                        previous_images = resolved_previous_images

    image_paths: list[Path] = []
    image_map: dict[str, Path] = {}
    reused_filenames: set[str] = set()
    seen_names: set[str] = set()

    if include_previous:
        for image_path in previous_images:
            if callable(is_cancelled) and is_cancelled():
                plan["image_paths"] = image_paths
                plan["image_map"] = image_map
                plan["reused_filenames"] = reused_filenames
                plan["reused_count"] = len(reused_filenames)
                plan["base_count"] = max(0, len(image_paths) - len(reused_filenames))
                plan["total_count"] = len(image_paths)
                return plan
            filename = image_path.name
            if filename in seen_names:
                continue
            seen_names.add(filename)
            image_paths.append(image_path)
            image_map[filename] = image_path
            reused_filenames.add(filename)

    for image_path in base_images:
        if callable(is_cancelled) and is_cancelled():
            plan["image_paths"] = image_paths
            plan["image_map"] = image_map
            plan["reused_filenames"] = reused_filenames
            plan["reused_count"] = len(reused_filenames)
            plan["base_count"] = max(0, len(image_paths) - len(reused_filenames))
            plan["total_count"] = len(image_paths)
            return plan
        filename = image_path.name
        if filename in seen_names:
            continue
        seen_names.add(filename)
        image_paths.append(image_path)
        image_map[filename] = image_path

    plan["image_paths"] = image_paths
    plan["image_map"] = image_map
    plan["reused_filenames"] = reused_filenames
    plan["reused_count"] = len(reused_filenames)
    plan["base_count"] = max(0, len(image_paths) - len(reused_filenames))
    plan["total_count"] = len(image_paths)
    plan["manual_skip_count"] = len(manually_skipped_names)
    plan["manual_skip_filenames"] = manually_skipped_names
    return plan

def _schedule_deferred_campaign_route_cleanup(self, payload: dict | None) -> bool:
    if not isinstance(payload, dict) or not bool(payload.get("remove_persisted_runs")):
        return False

    token = int(getattr(self, "_campaign_route_cleanup_token", 0) or 0)
    active_project = str(payload.get("active_project") or "").strip()
    candidate_inputs = list(payload.get("candidate_inputs") or [])
    explicit_run_dirs = set(payload.get("explicit_run_dirs") or set())
    manual_run_keep = set(payload.get("manual_run_keep") or set())
    search_roots = list(payload.get("search_roots") or [])
    stage_root = payload.get("stage_root")
    stage_dir = payload.get("stage_dir")

    def _is_cancelled() -> bool:
        if token != int(getattr(self, "_campaign_route_cleanup_token", 0) or 0):
            return True
        try:
            from ..campaign_manager import CAMPAIGN

            current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        except Exception:
            current_project = ""
        return bool(active_project and current_project != active_project)

    def _start_worker() -> None:
        if _is_cancelled():
            return
        self._campaign_route_cleanup_after_id = None

        def worker() -> None:
            cleanup_started = time.perf_counter()
            result = {
                "removed_runs": 0,
                "removed_stage": False,
            }
            invalidate_roots: list[Path] = []
            visited: set[str] = set()

            if not _is_cancelled():
                for root in search_roots:
                    if _is_cancelled():
                        return
                    try:
                        if not root.exists() or not root.is_dir():
                            continue
                    except Exception:
                        continue

                    try:
                        run_paths = PROJECT_CACHE.list_annotation_run_dirs(root, require_xml=False)
                    except Exception:
                        continue

                    deleted_any = False
                    for run_dir in run_paths:
                        if _is_cancelled():
                            return
                        try:
                            if not run_dir.is_dir():
                                continue
                            run_key = str(run_dir.resolve())
                        except Exception:
                            try:
                                run_key = str(Path(run_dir))
                            except Exception:
                                continue

                        if run_key in visited:
                            continue
                        visited.add(run_key)

                        manifest = {}
                        remove_run = run_key in explicit_run_dirs
                        if not remove_run:
                            manifest = self._load_annotation_run_manifest(run_dir)
                            manifest_input = str(manifest.get("input_dir") or "").strip()
                            if manifest_input:
                                remove_run = any(
                                    self._paths_equivalent(manifest_input, candidate_input)
                                    for candidate_input in candidate_inputs
                                )

                        if not remove_run or not self._path_is_within(run_dir, root):
                            continue

                        if run_key in manual_run_keep or self._annotation_run_manifest_has_manual_value(manifest):
                            continue

                        try:
                            shutil.rmtree(run_dir)
                            result["removed_runs"] += 1
                            deleted_any = True
                        except Exception as e:
                            logger.debug(
                                f"Nie udalo sie usunac runu po zmianie toru E2 ({run_dir}): {e}"
                            )

                    if deleted_any:
                        invalidate_roots.append(root)

            def _stage_dir_is_manifest_source(candidate_stage_dir: Path) -> bool:
                try:
                    from ..campaign_manager import CAMPAIGN

                    current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
                    if not current_project:
                        return False
                    current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
                except Exception:
                    return False

                try:
                    stage_resolved = Path(candidate_stage_dir).resolve()
                except Exception:
                    stage_resolved = Path(candidate_stage_dir)

                def _is_stage_path(path_like) -> bool:
                    raw_text = str(path_like or "").strip()
                    if not raw_text:
                        return False
                    try:
                        candidate = Path(raw_text).resolve()
                    except Exception:
                        candidate = Path(raw_text)
                    try:
                        return candidate == stage_resolved or candidate.is_relative_to(stage_resolved)
                    except Exception:
                        try:
                            return str(candidate).startswith(str(stage_resolved))
                        except Exception:
                            return False

                for iter_value in range(1, max(1, current_iter) + 1):
                    try:
                        manifest = CAMPAIGN.load_ingest_manifest(iter_value, current_project) or {}
                    except Exception:
                        manifest = {}
                    if not isinstance(manifest, dict):
                        continue
                    for key in ("source_dir", "master_pool_dir", "target_dir"):
                        if _is_stage_path(manifest.get(key, "")):
                            return True
                    for item in list(manifest.get("selected_images") or []):
                        if not isinstance(item, dict):
                            continue
                        for key in ("target_path", "source_path", "iteration_target_path"):
                            if _is_stage_path(item.get(key, "")):
                                return True
                return False

            if (
                not _is_cancelled()
                and stage_root is not None
                and isinstance(stage_dir, Path)
            ):
                try:
                    if (
                        stage_dir.exists()
                        and self._path_is_within(stage_dir, stage_root)
                    ):
                        if _stage_dir_is_manifest_source(stage_dir):
                            logger.debug(
                                f"Pomijam cleanup stage, bo manifest iteracji nadal wskazuje na: {stage_dir}"
                            )
                        else:
                            shutil.rmtree(stage_dir)
                            result["removed_stage"] = True
                except Exception as e:
                    logger.debug(f"Nie udalo sie wyczyscic stage po zmianie toru E2: {e}")

            def _finish() -> None:
                if _is_cancelled():
                    return
                for root in invalidate_roots:
                    try:
                        PROJECT_CACHE.invalidate_annotation_run_dirs(root)
                    except Exception:
                        continue

                elapsed_ms = max(0.0, (time.perf_counter() - cleanup_started) * 1000.0)
                if elapsed_ms >= 20.0 or result["removed_runs"] or result["removed_stage"]:
                    logger.debug(
                        "[AnnotationTab][PERF] deferred_campaign_route_cleanup: "
                        f"{elapsed_ms:.1f} ms | removed_runs={int(result['removed_runs'])} "
                        f"removed_stage={bool(result['removed_stage'])}"
                    )

            self._post_to_ui(_finish)

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            logger.debug(f"Nie udalo sie uruchomic odroczonego cleanupu toru E2: {e}")

    try:
        self._campaign_route_cleanup_after_id = self.frame.after(180, _start_worker)
    except Exception:
        self._campaign_route_cleanup_after_id = None
        _start_worker()
    return True

def get_campaign_step2_source_state(self, *, iteration_target: str | None = None) -> dict:
    result = {
        "iteration_target": "",
        "ready": False,
        "has_source": False,
        "needs_more_tables": False,
        "images_with_plates": 0,
        "total_plates": 0,
        "restore_run_dir": None,
        "run_name": "",
        "bootstrap": {},
        "input_source": "",
        "manual_template": False,
        "plate_model_ready": False,
    }

    if self._is_free_mode_session_context():
        return result

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return result

        target = self._normalize_campaign_iteration_target_value(
            iteration_target or CAMPAIGN.get_iteration_target()
        )
        if target not in {"plate", "char"}:
            return result

        current_step = int(CAMPAIGN.get_current_step() or 0)
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
        last_iteration_target = self._normalize_campaign_iteration_target_value(
            CAMPAIGN.get_last_iteration_target()
        )
        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    except Exception:
        return result

    bootstrap = self.get_campaign_step2_bootstrap(iteration_target=target)
    restore_run_dir = bootstrap.get("restore_run_dir")
    effective_plate_model_path = str(bootstrap.get("plate_model_path") or plate_model_path or "").strip()

    result.update(
        iteration_target=target,
        bootstrap=dict(bootstrap),
        input_source=str(bootstrap.get("input_source") or "").strip(),
        manual_template=bool(bootstrap.get("manual_template")),
        plate_model_ready=bool(effective_plate_model_path and Path(effective_plate_model_path).exists()),
    )

    continuation_char_effective_allowed = False
    if target == "char" and current_iteration > 1 and current_step == 2 and step2_status == "pending":
        try:
            ingest_manifest = CAMPAIGN.load_ingest_manifest(current_iteration)
        except Exception:
            ingest_manifest = {}
        manifest_mode = str((ingest_manifest or {}).get("selection_mode", "") or "").strip().lower()
        continuation_char_effective_allowed = bool(
            last_iteration_target == "char"
            and manifest_mode in {"pool_reuse", "iteration_reuse", "stage_reuse"}
        )

    if target == "char" and (
        current_step >= 3
        or step2_status == "approved"
        or continuation_char_effective_allowed
    ):
        try:
            project_char_state = dict(CAMPAIGN.get_step3_char_source_state() or {})
        except Exception:
            project_char_state = {}
        project_char_images = int(project_char_state.get("images_with_plates", 0) or 0)
        project_char_plates = int(project_char_state.get("total_plates", 0) or 0)
        if project_char_plates > 0:
            result["has_source"] = True
            result["images_with_plates"] = project_char_images
            result["total_plates"] = project_char_plates
            result["project_images_with_plates"] = int(
                project_char_state.get("project_images_with_plates", project_char_images) or 0
            )
            result["project_total_plates"] = int(
                project_char_state.get("project_total_plates", project_char_plates) or 0
            )
            result["source_scope"] = str(project_char_state.get("source_scope") or "campaign_approved_set").strip()
            result["run_name"] = str(
                project_char_state.get("run_name")
                or project_char_state.get("display_name")
                or "Zatwierdzony zbiór projektu tablic"
            ).strip()
            result["input_source"] = "campaign_plate_approved_set"
            result["manual_template"] = False
            min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
            result["ready"] = bool(project_char_plates >= int(min_char_plates))
            result["needs_more_tables"] = bool(not result["ready"])
            result["bootstrap"] = {
                **dict(result.get("bootstrap") or {}),
                "input_source": "campaign_plate_approved_set",
                "manual_template": False,
                "plate_model_path": str(plate_model_path or "").strip(),
                "display_name": result["run_name"],
                "run_name": result["run_name"],
                "source_scope": result["source_scope"],
            }
            return result

        try:
            effective_source = dict(self._build_campaign_char_effective_source() or {})
        except Exception:
            effective_source = {}

        effective_run_dir = self._resolve_existing_run_dir(effective_source.get("run_dir"))
        effective_images_dir = self._resolve_existing_dir(effective_source.get("images_dir"))
        effective_xml_path = self._path_value_to_path(effective_source.get("xml_path"))
        if (
            effective_run_dir is not None
            and effective_images_dir is not None
            and effective_xml_path is not None
            and effective_xml_path.exists()
        ):
            result["has_source"] = True
            result["restore_run_dir"] = effective_run_dir
            result["run_name"] = str(
                effective_source.get("display_name")
                or getattr(effective_run_dir, "name", "")
                or ""
            ).strip()
            result["bootstrap"] = {
                "restore_run_dir": effective_run_dir,
                "input_dir": effective_images_dir,
                "xml_path": effective_xml_path,
                "input_source": "campaign_char_effective_source",
                "manual_template": False,
                "plate_model_path": str(plate_model_path or "").strip(),
                "display_name": result["run_name"],
                "run_name": result["run_name"],
                "contributor_run_dir": str(effective_source.get("contributor_run_dir") or "").strip(),
            }
            result["input_source"] = "campaign_char_effective_source"
            result["manual_template"] = False
            result["images_with_plates"] = int(effective_source.get("images_with_plates", 0) or 0)
            result["total_plates"] = int(effective_source.get("total_plates", 0) or 0)
            min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
            if int(result["total_plates"] or 0) >= int(min_char_plates):
                result["ready"] = True
            elif int(result["total_plates"] or 0) > 0:
                result["needs_more_tables"] = True
            return result

    if target == "char":
        try:
            project_char_state = dict(CAMPAIGN.get_step3_char_source_state() or {})
        except Exception:
            project_char_state = {}
        project_char_images = int(project_char_state.get("images_with_plates", 0) or 0)
        project_char_plates = int(project_char_state.get("total_plates", 0) or 0)
        if project_char_plates > 0:
            result["has_source"] = True
            result["images_with_plates"] = project_char_images
            result["total_plates"] = project_char_plates
            result["project_images_with_plates"] = int(
                project_char_state.get("project_images_with_plates", project_char_images) or 0
            )
            result["project_total_plates"] = int(
                project_char_state.get("project_total_plates", project_char_plates) or 0
            )
            result["source_scope"] = str(project_char_state.get("source_scope") or "campaign_approved_set").strip()
            result["run_name"] = str(
                project_char_state.get("run_name")
                or project_char_state.get("display_name")
                or "Zatwierdzony zbiór projektu tablic"
            ).strip()
            result["input_source"] = "campaign_plate_approved_set"
            result["manual_template"] = False
            min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
            result["ready"] = bool(project_char_plates >= int(min_char_plates))
            result["needs_more_tables"] = bool(not result["ready"])
            result["bootstrap"] = {
                **dict(result.get("bootstrap") or {}),
                "input_source": "campaign_plate_approved_set",
                "manual_template": False,
                "plate_model_path": str(plate_model_path or "").strip(),
                "display_name": result["run_name"],
                "run_name": result["run_name"],
                "source_scope": result["source_scope"],
            }
            return result

    if restore_run_dir is None:
        return result

    result["has_source"] = True
    result["restore_run_dir"] = restore_run_dir
    try:
        result["run_name"] = str(Path(restore_run_dir).name or "").strip()
    except Exception:
        result["run_name"] = ""

    try:
        manifest = self._load_annotation_run_manifest(restore_run_dir)
        manual_ready = bool(self._annotation_run_manifest_has_manual_value(manifest))
    except Exception:
        manifest = {}
        manual_ready = False

    try:
        images_with_plates, total_plates = self._get_run_plate_annotation_counts(restore_run_dir)
    except Exception:
        images_with_plates, total_plates = 0, 0
    try:
        approved_images_with_plates, approved_total_plates = self._get_run_plate_approved_counts(restore_run_dir)
    except Exception:
        approved_images_with_plates, approved_total_plates = 0, 0

    result["images_with_plates"] = int(images_with_plates or 0)
    result["total_plates"] = int(total_plates or 0)
    result["approved_images_with_plates"] = int(approved_images_with_plates or 0)
    result["approved_total_plates"] = int(approved_total_plates or 0)

    if target == "plate":
        result["ready"] = True
        return result

    plate_annotations_ready = int(total_plates or 0) > 0
    if not manual_ready and not plate_annotations_ready:
        return result

    result["images_with_plates"] = int(approved_images_with_plates or 0)
    min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    if int(approved_total_plates or 0) >= int(min_char_plates):
        result["ready"] = True
    else:
        result["needs_more_tables"] = True

    return result

def _get_campaign_step3_preview_source_context(self) -> dict:
    if self._is_free_mode_session_context():
        return {"filenames": set(), "preview_dir": None, "meta_path": None}

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return {"filenames": set(), "preview_dir": None, "meta_path": None}
        if str(CAMPAIGN.get_iteration_target() or "").strip().lower() != "char":
            return {"filenames": set(), "preview_dir": None, "meta_path": None}
    except Exception:
        return {"filenames": set(), "preview_dir": None, "meta_path": None}

    preview_dir = None
    try:
        tab_char = getattr(self.app, "tabs", {}).get("characters")
    except Exception:
        tab_char = None
    if tab_char is not None and hasattr(tab_char, "preview_dir_var"):
        try:
            preview_dir_raw = str(tab_char.preview_dir_var.get() or "").strip()
        except Exception:
            preview_dir_raw = ""
        if preview_dir_raw:
            try:
                candidate = Path(preview_dir_raw)
                if (
                    candidate.exists()
                    and candidate.is_dir()
                    and (candidate / "metadata.json").exists()
                    and (candidate / "images").exists()
                ):
                    preview_dir = candidate
            except Exception:
                preview_dir = None

    if preview_dir is None:
        try:
            bundle = dict(
                CAMPAIGN.get_iteration_artifact_bundle(
                    images_dir=CAMPAIGN.get_iteration_image_source_dir()
                    or CAMPAIGN.get_master_pool_dir()
                    or CAMPAIGN.get_iteration_raw_dir(),
                    iteration_num=int(CAMPAIGN.get_current_iteration_num() or 1),
                ) or {}
            )
        except Exception:
            bundle = {}
        preview_entry = dict(bundle.get("step3_preview_source") or {})
        preview_dir_raw = str(preview_entry.get("preview_dir") or "").strip()
        if preview_dir_raw:
            try:
                candidate = Path(preview_dir_raw)
                if (
                    candidate.exists()
                    and candidate.is_dir()
                    and (candidate / "metadata.json").exists()
                    and (candidate / "images").exists()
                ):
                    preview_dir = candidate
            except Exception:
                preview_dir = None

    if preview_dir is None:
        try:
            saved_preview_dir = str(CAMPAIGN.get_step3_preview_dir() or "").strip()
        except Exception:
            saved_preview_dir = ""
        if saved_preview_dir:
            try:
                candidate = Path(saved_preview_dir)
                if (
                    candidate.exists()
                    and candidate.is_dir()
                    and (candidate / "metadata.json").exists()
                    and (candidate / "images").exists()
                ):
                    preview_dir = candidate
            except Exception:
                preview_dir = None

    if preview_dir is None:
        # W nowej architekturze źródło PZ3 powinno być wskazane przez rejestr
        # artefaktów. Rekurencyjne zgadywanie po całym katalogu znaków blokowało
        # wejście T06/Z2 na dużych projektach, więc nie wykonujemy go w gorącej
        # ścieżce ładowania karty.
        return {"filenames": set(), "preview_dir": None, "meta_path": None}

    if preview_dir is None:
        return {"filenames": set(), "preview_dir": None, "meta_path": None}

    meta_path = preview_dir / "metadata.json"
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict) or not loaded:
        return {"filenames": set(), "preview_dir": preview_dir, "meta_path": meta_path}

    preview_filenames: set[str] = set()
    for pid, payload in loaded.items():
        source_name = ""
        if isinstance(payload, dict):
            source_info = payload.get("source_info") or {}
            if not isinstance(source_info, dict):
                source_info = {}
            source_name = str(
                payload.get("source_name")
                or source_info.get("image_name")
                or payload.get("source_image")
                or pid
                or ""
            ).strip()
        else:
            source_name = str(pid or "").strip()
        if not source_name:
            continue
        try:
            normalized = Path(source_name).name.strip().lower()
        except Exception:
            normalized = str(source_name or "").strip().lower()
        if normalized:
            preview_filenames.add(normalized)

    return {
        "filenames": preview_filenames,
        "preview_dir": preview_dir,
        "meta_path": meta_path,
    }


def _resolve_campaign_project_name_from_run_dir(run_dir: Path | str | None) -> str:
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


def _build_campaign_plate_approved_entries_from_run(
    self,
    run_dir: Path,
    *,
    extra_included_filenames: set[str] | None = None,
    force_parse_xml: bool = False,
) -> list[dict]:
    if self._is_free_mode_session_context():
        return []

    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return []

    project_name = _resolve_campaign_project_name_from_run_dir(safe_run_dir)

    annotations: list[ImageAnnotation] = []
    if (
        not force_parse_xml
        and
        getattr(self, "current_annotation_run_dir", None) is not None
        and self.current_annotations
        and self._paths_equivalent(self.current_annotation_run_dir, safe_run_dir)
    ):
        annotations = list(self.current_annotations or [])
    else:
        try:
            annotations = self._parse_cvat_preview_annotations(safe_run_dir / "annotations.xml")
        except Exception:
            annotations = []

    if not annotations:
        return []

    manifest = self._load_annotation_run_manifest(safe_run_dir)
    explicit_approval_enabled = isinstance(manifest, dict) and "approved_filenames" in manifest
    approved_filenames = (
        self._get_preview_approved_filenames()
        if (
            not force_parse_xml
            and
            getattr(self, "current_annotation_run_dir", None) is not None
            and self.current_annotations
            and self._paths_equivalent(self.current_annotation_run_dir, safe_run_dir)
        )
        else self._load_annotation_run_approved_filenames(safe_run_dir)
    )
    if explicit_approval_enabled and not approved_filenames:
        try:
            approved_state = dict(self._get_run_plate_strict_approved_state(safe_run_dir) or {})
            approved_filenames = set(approved_state.get("approved_filenames") or set())
        except Exception:
            approved_filenames = set()
    extra_included_lookup = {
        str(name or "").strip().lower()
        for name in set(extra_included_filenames or set())
        if str(name or "").strip()
    }
    input_dir = (
        self._resolve_existing_dir(manifest.get("source_input_dir"))
        or self._resolve_existing_dir(manifest.get("input_dir"))
        or self._resolve_existing_dir(self.current_input_dir)
    )
    run_images_dir = safe_run_dir / "images"
    source_xml_path = safe_run_dir / "annotations.xml"
    manual_origin = bool(self._annotation_run_manifest_has_manual_value(manifest))
    approved_at = datetime.datetime.now().isoformat(timespec="seconds")
    approved_iteration = 0
    try:
        approved_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        approved_iteration = 0
    if approved_iteration <= 0 and project_name:
        try:
            project_data = dict((CAMPAIGN.state or {}).get("projects", {}).get(project_name, {}) or {})
            approved_iteration = int(project_data.get("current_iteration", 1) or 1)
        except Exception:
            approved_iteration = 0
    if approved_iteration <= 0:
        approved_iteration = 1

    entries: list[dict] = []
    for ann in annotations:
        image_name = str(getattr(ann, "filename", "") or "").strip()
        if not image_name:
            continue
        image_name_key = image_name.lower()
        if (
            explicit_approval_enabled
            and image_name_key not in approved_filenames
            and image_name_key not in extra_included_lookup
        ):
            continue

        plate_detections = self._get_plate_detections(ann)
        if not plate_detections:
            continue

        source_image_path = None
        preview_map = dict(getattr(self, "_preview_image_path_map", {}) or {})
        try:
            preview_source = preview_map.get(image_name)
            if preview_source:
                preview_source = Path(preview_source)
                if preview_source.exists():
                    source_image_path = preview_source
        except Exception:
            source_image_path = None

        if source_image_path is None and input_dir is not None:
            candidate = Path(input_dir) / image_name
            if candidate.exists():
                source_image_path = candidate

        if source_image_path is None and run_images_dir.exists():
            candidate = run_images_dir / image_name
            if candidate.exists():
                source_image_path = candidate

        if source_image_path is None or not Path(source_image_path).exists():
            continue

        entry_key = self._build_campaign_plate_entry_key(
            image_name=image_name,
            source_image_path=source_image_path,
        )

        plate_entries = []
        has_manual_polygon = False
        for det in plate_detections:
            polygon = self._detection_polygon(det)
            if len(polygon) < 4:
                continue
            attributes = dict(getattr(det, "attributes", {}) or {})
            if str(attributes.get("manually_edited", "") or "").strip().lower() == "true":
                has_manual_polygon = True
            if str(attributes.get("manual_source", "") or "").strip():
                has_manual_polygon = True
            plate_entries.append(
                {
                    "confidence": float(getattr(det, "confidence", 1.0) or 1.0),
                    "polygon": [[float(x), float(y)] for x, y in polygon[:4]],
                    "attributes": attributes,
                }
            )

        if not plate_entries:
            continue

        entries.append(
            {
                "entry_key": entry_key,
                "image_name": image_name,
                "source_image_path": str(Path(source_image_path).resolve()),
                "approved_from_run": str(safe_run_dir.resolve()),
                "approved_from_xml": str(source_xml_path.resolve()),
                "width": int(getattr(ann, "width", 0) or 0),
                "height": int(getattr(ann, "height", 0) or 0),
                "plate_count": int(len(plate_entries)),
                "plates": plate_entries,
                "annotation_origin": ("manual" if (manual_origin or has_manual_polygon) else "auto_accepted"),
                "approved_at": approved_at,
                "approved_iteration": int(approved_iteration),
                "first_approved_iteration": int(approved_iteration),
                "first_approved_at": approved_at,
            }
        )

    return entries

def reset_campaign_iteration_route_state(
    self,
    new_target: str | None = None,
    *,
    remove_persisted_runs: bool = True,
) -> dict:
    reset_started = time.perf_counter()
    result = {
        "removed_runs": 0,
        "removed_snapshot": False,
        "removed_stage": False,
    }

    try:
        from ..campaign_manager import CAMPAIGN

        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        if not active_project:
            return result

        target = str(new_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
        if target not in {"plate", "char"}:
            target = "plate"

        pending = getattr(self, "_free_mode_session_save_after_id", None)
        if pending:
            try:
                self.frame.after_cancel(pending)
            except Exception:
                pass
        self._free_mode_session_save_after_id = None

        candidate_inputs: list[Path] = []
        try:
            iteration_raw_dir = CAMPAIGN.get_iteration_image_source_dir()
        except Exception:
            iteration_raw_dir = CAMPAIGN.get_iteration_raw_dir()
        if iteration_raw_dir is not None:
            candidate_inputs.append(Path(iteration_raw_dir))

        raw_root = CAMPAIGN.get_dir("raw")
        if raw_root is not None:
            iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
            raw_iter_dir = Path(raw_root) / f"Iteracja_{iter_num:03d}"
            candidate_inputs.append(raw_iter_dir if raw_iter_dir.exists() else Path(raw_root))

        try:
            candidate_inputs.append(self._get_manual_plate_stage_images_dir())
        except Exception:
            pass

        stored_manual_source = {}
        try:
            stored_manual_source = CAMPAIGN.get_last_plate_manual_source()
        except Exception:
            stored_manual_source = {}

        manual_run_keep = set()
        for candidate in (
            str(stored_manual_source.get("source_run_path") or "").strip(),
            str(stored_manual_source.get("source_xml_path") or "").strip(),
        ):
            if not candidate:
                continue
            try:
                manual_path = Path(candidate)
                if manual_path.suffix.lower() == ".xml":
                    manual_path = manual_path.parent
                manual_run_keep.add(str(manual_path.resolve()))
            except Exception:
                try:
                    manual_run_keep.add(str(Path(candidate)))
                except Exception:
                    pass

        explicit_run_dirs = set()
        for candidate in (
            str(CAMPAIGN.get_step2_staging_run() or "").strip(),
            str(getattr(self, "current_annotation_run_dir", "") or "").strip(),
            str(getattr(self, "last_staging_run_dir", "") or "").strip(),
            str(self.plate_dataset_run_var.get() or "").strip(),
        ):
            if not candidate:
                continue
            try:
                explicit_run_dirs.add(str(Path(candidate).resolve()))
            except Exception:
                explicit_run_dirs.add(str(Path(candidate)))

        self._cancel_pending_campaign_route_cleanup()
        cleanup_payload = self._build_campaign_route_cleanup_payload(
            active_project=active_project,
            candidate_inputs=candidate_inputs,
            explicit_run_dirs=explicit_run_dirs,
            manual_run_keep=manual_run_keep,
            remove_persisted_runs=bool(remove_persisted_runs),
        )

        snapshot_path = self._get_campaign_annotation_state_path(active_project)
        try:
            if snapshot_path is not None and snapshot_path.exists():
                snapshot_path.unlink()
                result["removed_snapshot"] = True
        except Exception as e:
            logger.debug(f"Nie udalo sie usunac snapshotu Z2 po zmianie toru E2: {e}")

        self._preview_session_restore_index = -1
        self._preview_session_restore_filename = ""
        if raw_root is not None:
            iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
            try:
                input_dir = CAMPAIGN.get_iteration_image_source_dir(iter_num)
            except Exception:
                input_dir = None
            if input_dir is None:
                input_dir = Path(raw_root) / f"Iteracja_{iter_num:03d}"
                if not input_dir.exists():
                    input_dir = Path(raw_root)
        else:
            input_dir = None

        self._reset_campaign_runtime_state(input_dir=input_dir)
        self._campaign_context_project_name = str(active_project or "").strip()
        try:
            self._refresh_step2_action_states()
        except Exception:
            pass
        if bool(remove_persisted_runs):
            self._schedule_deferred_campaign_route_cleanup(cleanup_payload)
            result["cleanup_deferred"] = True
    except Exception as e:
        logger.debug(f"Nie udalo sie zresetowac stanu iteracji po zmianie toru E2: {e}")

    elapsed_ms = max(0.0, (time.perf_counter() - reset_started) * 1000.0)
    if elapsed_ms >= 20.0:
        logger.debug(
            "[AnnotationTab][PERF] reset_campaign_iteration_route_state: "
            f"{elapsed_ms:.1f} ms | removed_runs={int(result.get('removed_runs', 0) or 0)}, "
            f"removed_stage={bool(result.get('removed_stage'))}, "
            f"removed_snapshot={bool(result.get('removed_snapshot'))} "
            f"cleanup_deferred={1 if bool(result.get('cleanup_deferred')) else 0}"
        )
    return result

def _sync_campaign_iteration_artifact_registry(
    self,
    *,
    run_dir: Path | None = None,
    xml_path: Path | None = None,
    input_dir: Path | None = None,
    extra_updates: dict | None = None,
) -> None:
    if self._is_free_mode_session_context():
        return

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        return

    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=False)
    safe_xml_path = None
    if xml_path is not None:
        try:
            candidate_xml = Path(xml_path)
            if candidate_xml.exists():
                safe_xml_path = candidate_xml
        except Exception:
            safe_xml_path = None
    if safe_xml_path is None and safe_run_dir is not None:
        candidate_xml = safe_run_dir / "annotations.xml"
        if candidate_xml.exists():
            safe_xml_path = candidate_xml

    manifest = self._load_annotation_run_manifest(safe_run_dir) if safe_run_dir is not None else {}

    images_dir = None
    manifest_input_candidates = []
    if isinstance(manifest, dict):
        for key in ("input_dir", "source_input_dir", "imported_source_input_dir"):
            raw_value = str(manifest.get(key) or "").strip()
            if raw_value:
                manifest_input_candidates.append(raw_value)
    try:
        iteration_source_dir = CAMPAIGN.get_iteration_image_source_dir(iteration_num)
    except Exception:
        iteration_source_dir = None

    if safe_run_dir is not None or safe_xml_path is not None:
        candidate_dirs = (
            *manifest_input_candidates,
            input_dir,
            getattr(self, "current_input_dir", None),
            iteration_source_dir,
            CAMPAIGN.get_master_pool_dir(),
            CAMPAIGN.get_iteration_raw_dir(iteration_num),
        )
    else:
        candidate_dirs = (
            iteration_source_dir,
            CAMPAIGN.get_master_pool_dir(),
            CAMPAIGN.get_iteration_raw_dir(iteration_num),
            input_dir,
            getattr(self, "current_input_dir", None),
        )

    for candidate in candidate_dirs:
        try:
            resolved = self._resolve_existing_dir(candidate)
        except Exception:
            resolved = None
        if resolved is not None:
            images_dir = resolved
            break
    if images_dir is None:
        return

    updates: dict[str, object] = {}

    try:
        plate_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
    except Exception:
        plate_meta = {}
    plate_model_path = str(plate_meta.get("path") or "").strip()
    if plate_model_path:
        updates["plate_model"] = {
            "path": plate_model_path,
            "token": CAMPAIGN._build_registry_path_token(plate_model_path),
            "identity": str(plate_meta.get("identity") or "").strip(),
            "source": str(plate_meta.get("source") or "").strip(),
            "scope": str(plate_meta.get("scope") or "").strip(),
        }

    if safe_run_dir is not None or safe_xml_path is not None:
        try:
            images_with_plates, total_plates = (
                self._get_run_plate_annotation_counts(safe_run_dir)
                if safe_run_dir is not None and safe_xml_path is not None
                else (0, 0)
            )
        except Exception:
            images_with_plates, total_plates = 0, 0
        try:
            approved_images, approved_plates = (
                self._get_run_plate_approved_counts(safe_run_dir)
                if safe_run_dir is not None
                else (0, 0)
            )
        except Exception:
            approved_images, approved_plates = 0, 0

        updates["plate_source"] = {
            "run_dir": str(safe_run_dir.resolve()) if safe_run_dir is not None else "",
            "run_token": CAMPAIGN._build_registry_path_token(safe_run_dir),
            "xml_path": str(safe_xml_path.resolve()) if safe_xml_path is not None else "",
            "xml_token": CAMPAIGN._build_registry_path_token(safe_xml_path),
            "images_dir": str(images_dir.resolve()),
            "images_token": CAMPAIGN._build_registry_path_token(images_dir),
            "annotation_run_type": str(manifest.get("annotation_run_type") or "").strip(),
            "manual_template": bool(manifest.get("manual_xml_template", False)),
            "manual_ready": bool(self._annotation_run_manifest_has_manual_value(manifest)),
            "run_status": str(manifest.get("run_status") or "").strip(),
            "input_source": str(manifest.get("input_source_dir") or manifest.get("source_input_dir") or "").strip(),
            "images_with_plates": int(images_with_plates or 0),
            "total_plates": int(total_plates or 0),
            "approved_images": int(approved_images or 0),
            "approved_plates": int(approved_plates or 0),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        updates["step2_active_run"] = {
            "run_dir": str(safe_run_dir.resolve()) if safe_run_dir is not None else "",
            "run_token": CAMPAIGN._build_registry_path_token(safe_run_dir),
            "run_name": str(getattr(safe_run_dir, "name", "") or "").strip(),
            "xml_path": str(safe_xml_path.resolve()) if safe_xml_path is not None else "",
            "xml_token": CAMPAIGN._build_registry_path_token(safe_xml_path),
            "images_dir": str(images_dir.resolve()),
            "images_token": CAMPAIGN._build_registry_path_token(images_dir),
            "input_source": str(manifest.get("input_source_dir") or manifest.get("source_input_dir") or "").strip(),
            "annotation_run_type": str(manifest.get("annotation_run_type") or "").strip(),
            "manual_template": bool(manifest.get("manual_xml_template", False)),
            "run_status": str(manifest.get("run_status") or "").strip(),
            "approved_images": int(approved_images or 0),
            "approved_plates": int(approved_plates or 0),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

    if isinstance(extra_updates, dict) and extra_updates:
        updates.update(dict(extra_updates))

    if not updates:
        return

    try:
        CAMPAIGN.upsert_iteration_artifact_bundle(
            images_dir=images_dir,
            iteration_num=iteration_num,
            updates=updates,
        )
    except Exception as e:
        logger.debug(f"Nie udało się zsynchronizować rejestru artefaktów kampanii z Z2: {e}")

def _build_campaign_plate_approved_preview_bundle(
    self,
    *,
    image_names: set[str] | None = None,
    preferred_image_dir: Path | None = None,
    image_candidates: dict[str, Path] | None = None,
) -> dict[str, tuple[ImageAnnotation, Path]]:
    if self._is_free_mode_session_context():
        return {}

    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {}

    approved_entries = list(CAMPAIGN.list_plate_approved_entries() or [])
    if not approved_entries:
        return {}

    target_names = {
        str(name or "").strip().lower()
        for name in set(image_names or set())
        if str(name or "").strip()
    }

    candidate_name_by_resolved_path: dict[str, str] = {}
    candidate_name_by_basename: dict[str, str] = {}
    candidate_path_by_name: dict[str, Path] = {}
    for raw_name, raw_path in dict(image_candidates or {}).items():
        safe_name = str(raw_name or "").strip()
        if not safe_name:
            continue
        try:
            candidate_path = Path(raw_path)
        except Exception:
            continue
        candidate_path_by_name[safe_name] = candidate_path
        target_names.add(safe_name.lower())
        try:
            candidate_name_by_resolved_path[str(candidate_path.resolve()).strip().lower()] = safe_name
        except Exception:
            pass
        basename = Path(safe_name).name.strip().lower()
        if basename and basename not in candidate_name_by_basename:
            candidate_name_by_basename[basename] = safe_name

    bundle: dict[str, tuple[ImageAnnotation, Path]] = {}
    for entry in approved_entries:
        image_name = str(entry.get("image_name", "") or "").strip()
        if not image_name:
            continue

        source_image_path = str(entry.get("source_image_path", "") or "").strip()
        safe_name = image_name.lower()
        matched_name = image_name if (not target_names or safe_name in target_names) else ""
        if not matched_name and source_image_path:
            try:
                matched_name = candidate_name_by_resolved_path.get(
                    str(Path(source_image_path).resolve()).strip().lower(),
                    "",
                )
            except Exception:
                matched_name = ""
        if not matched_name:
            matched_name = candidate_name_by_basename.get(Path(image_name).name.strip().lower(), "")
        if target_names and not matched_name:
            continue
        if not matched_name:
            matched_name = image_name

        image_path = None
        if matched_name in candidate_path_by_name:
            try:
                candidate = Path(candidate_path_by_name[matched_name])
                if candidate.exists():
                    image_path = candidate
            except Exception:
                image_path = None
        if preferred_image_dir is not None:
            try:
                candidate = Path(preferred_image_dir) / matched_name
                if candidate.exists():
                    image_path = candidate
            except Exception:
                image_path = None
        if image_path is None and source_image_path:
            try:
                candidate = Path(source_image_path)
                if candidate.exists():
                    image_path = candidate
            except Exception:
                image_path = None
        if image_path is None:
            continue

        detections: list[Detection] = []
        for plate_entry in list(entry.get("plates") or []):
            polygon = [
                (float(point[0]), float(point[1]))
                for point in list(plate_entry.get("polygon") or [])[:4]
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(polygon) < 4:
                continue
            attributes = dict(plate_entry.get("attributes") or {})
            if str(entry.get("annotation_origin", "") or "").strip().lower() == "manual":
                attributes.setdefault("manually_edited", "true")
                attributes.setdefault("manual_source", "campaign_approved")
            detections.append(
                Detection(
                    label="plate",
                    confidence=float(plate_entry.get("confidence", 1.0) or 1.0),
                    bbox=self._bbox_from_polygon(polygon),
                    keypoints=self._keypoints_from_polygon(polygon),
                    polygon=polygon,
                    attributes=attributes,
                )
            )

        if not detections:
            continue

        annotation = ImageAnnotation(
            filename=matched_name,
            width=max(1, int(entry.get("width", 0) or 0)),
            height=max(1, int(entry.get("height", 0) or 0)),
            detections=detections,
            status=AnnotationStatus.SUCCESS,
        )
        try:
            setattr(annotation, "_approved_for_training", True)
        except Exception:
            pass
        bundle[matched_name] = (annotation, Path(image_path))

    return bundle

def _build_campaign_plate_approved_export_source(self, *, export_root=None) -> dict:
    if self._is_free_mode_session_context():
        return {}

    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {}

    approved_entries = list(CAMPAIGN.list_plate_approved_entries() or [])
    if not approved_entries:
        return {}

    state_dir = CAMPAIGN.get_project_state_dir()
    if state_dir is None:
        return {}

    export_root = Path(export_root) if export_root is not None else Path(state_dir) / "plate_approved_export"
    try:
        if export_root.exists() and self._path_is_within(export_root, state_dir):
            shutil.rmtree(export_root)
    except Exception as e:
        logger.debug(f"Nie udało się wyczyścić tymczasowego eksportu ApprovedSet: {e}")

    images_dir = export_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    xml_path = export_root / "annotations.xml"

    annotations: list[ImageAnnotation] = []
    used_names: set[str] = set()
    copied = 0

    for entry in approved_entries:
        image_name = str(entry.get("image_name", "") or "").strip()
        source_image_path = str(entry.get("source_image_path", "") or "").strip()
        if not image_name or not source_image_path:
            continue

        try:
            source_path = Path(source_image_path)
        except Exception:
            continue
        if not source_path.exists():
            continue

        export_name = image_name
        if export_name in used_names:
            stem = Path(image_name).stem or "image"
            suffix = Path(image_name).suffix or ".jpg"
            counter = 1
            while export_name in used_names:
                export_name = f"{stem}__{counter:03d}{suffix}"
                counter += 1
        used_names.add(export_name)

        try:
            shutil.copy2(source_path, images_dir / export_name)
            copied += 1
        except Exception as e:
            logger.debug(f"Nie udało się skopiować obrazu ApprovedSet do eksportu ({source_path}): {e}")
            continue

        detections: list[Detection] = []
        for plate_entry in list(entry.get("plates") or []):
            polygon = [
                (float(point[0]), float(point[1]))
                for point in list(plate_entry.get("polygon") or [])[:4]
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(polygon) < 4:
                continue
            attributes = dict(plate_entry.get("attributes") or {})
            if str(entry.get("annotation_origin", "") or "").strip().lower() == "manual":
                attributes.setdefault("manually_edited", "true")
                attributes.setdefault("manual_source", "campaign_approved")
            detections.append(
                Detection(
                    label="plate",
                    confidence=float(plate_entry.get("confidence", 1.0) or 1.0),
                    bbox=self._bbox_from_polygon(polygon),
                    keypoints=self._keypoints_from_polygon(polygon),
                    polygon=polygon,
                    attributes=attributes,
                )
            )

        if not detections:
            continue

        annotations.append(
            ImageAnnotation(
                filename=export_name,
                width=max(1, int(entry.get("width", 0) or 0)),
                height=max(1, int(entry.get("height", 0) or 0)),
                detections=detections,
                status=AnnotationStatus.SUCCESS,
            )
        )

    if not annotations or copied <= 0:
        return {}

    if not CVATExporter(task_name="Campaign Approved Plates").export(
        annotations,
        xml_path,
        include_confidence=True,
        only_successful=False,
    ):
        return {}

    return {
        "run_dir": export_root,
        "images_dir": images_dir,
        "xml_path": xml_path,
        "entries_count": len(annotations),
    }

def _prepare_approved_step3_source_from_z2_run(self, run_dir: Path, images_dir: Path | None) -> Path | None:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return None

    approval_state = self._get_run_plate_strict_approved_state(safe_run_dir)
    if not approval_state.get("ok"):
        self._warn_plate_cut_requires_ok(approval_state)
        return None

    approved_lookup = {
        str(name or "").strip().lower()
        for name in set(approval_state.get("approved_filenames") or set())
        if str(name or "").strip()
    }
    if not approved_lookup:
        self._warn_plate_cut_requires_ok(approval_state)
        return None

    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(safe_run_dir, current_run_dir)
    ):
        source_annotations = list(self.current_annotations or [])
    else:
        try:
            source_annotations = self._parse_cvat_preview_annotations(safe_run_dir / "annotations.xml")
        except Exception as e:
            messagebox.showerror(
                "Błąd źródła Z2",
                f"Nie mogę odczytać annotations.xml przed wycinaniem tablic:\n{e}",
                parent=self.frame.winfo_toplevel(),
            )
            return None

    approved_annotations = []
    for ann in list(source_annotations or []):
        image_name = str(getattr(ann, "filename", "") or "").strip().lower()
        if not image_name or image_name not in approved_lookup:
            continue
        try:
            if len(self._get_plate_detections(ann)) <= 0:
                continue
        except Exception:
            continue
        approved_annotations.append(copy.deepcopy(ann))

    if not approved_annotations:
        self._warn_plate_cut_requires_ok(approval_state)
        return None

    derived_root = safe_run_dir.parent / "_z3_approved_sources"
    derived_run_dir = derived_root / f"{safe_run_dir.name}_ok"
    try:
        derived_run_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        messagebox.showerror(
            "Błąd źródła Z3",
            f"Nie mogę przygotować filtrowanego źródła dla PZ1:\n{e}",
            parent=self.frame.winfo_toplevel(),
        )
        return None

    derived_xml_path = derived_run_dir / "annotations.xml"
    if not CVATExporter(task_name="Z2 Approved Plates for Z3").export(
        approved_annotations,
        derived_xml_path,
        include_confidence=True,
        only_successful=False,
    ):
        messagebox.showerror(
            "Błąd źródła Z3",
            f"Nie udało się zapisać filtrowanego XML z pozycjami [OK]:\n{derived_xml_path}",
            parent=self.frame.winfo_toplevel(),
        )
        return None

    try:
        source_manifest = self._load_annotation_run_manifest(safe_run_dir)
    except Exception:
        source_manifest = {}

    manifest_payload = dict(source_manifest or {})
    manifest_payload.update(
        {
            "annotation_run_type": "z2_approved_for_z3",
            "source_run_dir": str(safe_run_dir),
            "source_xml_path": str(safe_run_dir / "annotations.xml"),
            "source_input_dir": str(images_dir or ""),
            "input_dir": str(images_dir or ""),
            "approved_filenames": sorted(
                str(getattr(ann, "filename", "") or "").strip().lower()
                for ann in approved_annotations
                if str(getattr(ann, "filename", "") or "").strip()
            ),
            "derived_for": "z3_plate_cutting",
            "derived_from_approved_only": True,
            "derived_source_run_name": str(safe_run_dir.name or ""),
            "derived_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "approved_images": int(approval_state.get("approved_images", 0) or 0),
            "approved_plates": int(approval_state.get("approved_plates", 0) or 0),
        }
    )
    try:
        manifest_path = self._annotation_run_manifest_path(derived_run_dir)
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        PROJECT_CACHE.invalidate_json(manifest_path)
    except Exception as e:
        logger.debug(f"Nie udało się zapisać manifestu źródła Z3 z pozycji OK: {e}")

    return derived_run_dir

def _promote_campaign_char_repair_ok_to_approved_pool_before_return(self) -> dict:
    if self._is_free_mode_session_context():
        return {"ok": False, "reason": "free_mode"}

    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {"ok": False, "reason": "campaign_unavailable"}

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    except Exception:
        current_step = 0
        iteration_target = ""

    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) if graph_gate_id else ""
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
    is_t07_plate_repair = bool(
        graph_gate_id == "T04"
        and repair_origin_gate_id == "T06"
        and current_step == 4
        and iteration_target == "plate"
    )
    is_t06_char_work = bool(
        graph_gate_id == "T05"
        and current_step == 3
        and iteration_target == "char"
    )
    is_char_repair = bool(current_step == 3 and iteration_target == "char")

    if not CAMPAIGN.get_active_project_name() or not (is_char_repair or is_t07_plate_repair):
        return {"ok": False, "reason": "not_repair_return"}

    approval_context = self._get_campaign_step2_approval_context()
    run_dir = self._resolve_safe_annotation_run_dir(approval_context.get("run_dir"), require_xml=True)
    if run_dir is None:
        run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None), require_xml=True)
    if run_dir is None:
        run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "last_staging_run_dir", None), require_xml=True)
    if run_dir is None:
        return {"ok": False, "reason": "missing_run"}

    if is_t07_plate_repair:
        save_reason = "wyjście do grafu z trybu naprawczego T07"
    elif is_t06_char_work:
        save_reason = "przekazanie zdjęć [OK] do źródła Z3 z bramki T06"
    else:
        save_reason = "wyjście do grafu z trybu naprawczego E3"
    if not self._ensure_preview_edits_saved(save_reason):
        return {"ok": False, "reason": "save_failed", "abort": True}

    try:
        current_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "current_annotation_run_dir", None),
            require_xml=True,
        )
    except Exception:
        current_run_dir = None
    should_persist_preview_approval = False
    if current_run_dir is not None and self._paths_equivalent(current_run_dir, run_dir):
        try:
            preview_approved = set(self._get_preview_approved_filenames_base() or set())
        except Exception:
            preview_approved = set()
        try:
            approval_version = int(getattr(self, "_preview_approval_version", 0) or 0)
        except Exception:
            approval_version = 0
        should_persist_preview_approval = bool(preview_approved or approval_version > 0)
    if should_persist_preview_approval:
        try:
            self._persist_preview_approved_filenames()
        except Exception:
            pass

    current_ok_images, current_ok_plates = 0, 0
    try:
        current_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "current_annotation_run_dir", None),
            require_xml=True,
        )
        if current_run_dir is not None and self._paths_equivalent(current_run_dir, run_dir):
            current_ok_images, current_ok_plates = self._get_current_preview_plate_approved_counts()
    except Exception:
        current_ok_images, current_ok_plates = 0, 0
    try:
        run_ok_images, run_ok_plates = self._get_run_plate_approved_counts(run_dir)
    except Exception:
        run_ok_images, run_ok_plates = 0, 0

    approved_images = max(int(current_ok_images or 0), int(run_ok_images or 0))
    approved_plates = max(int(current_ok_plates or 0), int(run_ok_plates or 0))
    if approved_images <= 0 or approved_plates <= 0:
        return {"ok": False, "reason": "no_ok_images"}

    try:
        result = self._promote_run_to_campaign_plate_approved_set(
            run_dir,
            force_parse_xml=True,
            project_name=str(CAMPAIGN.get_active_project_name() or "").strip() or None,
        )
    except Exception as e:
        logger.debug(f"Nie udało się przenieść [OK] z trybu naprawczego E3 do puli zatwierdzonych: {e}")
        result = {"ok": False, "reason": "promotion_exception"}

    if not bool(result.get("ok")):
        try:
            warning_title = (
                "Nie zapisano puli YOLO"
                if is_t07_plate_repair
                else "Nie zapisano puli do wycinania"
            )
            warning_body = (
                "Nie udało się przenieść zdjęć [OK] do projektowej puli YOLO, więc po ponownym wejściu "
                "mogą nadal wisieć na liście roboczej.\n\n"
                "Pozostań w Z2, zapisz anotacje i spróbuj ponownie wrócić do grafu."
                if is_t07_plate_repair
                else (
                    "Nie udało się przenieść zdjęć [OK] do katalogu zatwierdzonych, więc nie będą jeszcze "
                    "bezpiecznie dostępne dla procesu wycinania tablic w E3.\n\n"
                    "Pozostań w Z2, zapisz anotacje i spróbuj ponownie wrócić do grafu."
                )
            )
            messagebox.showwarning(
                warning_title,
                warning_body,
                parent=self.frame.winfo_toplevel(),
            )
        except Exception:
            pass
        return {**dict(result or {}), "abort": True}

    try:
        promoted_names = self._load_annotation_run_approved_filenames(run_dir)
        self._campaign_hidden_project_approved_filenames = (
            set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
            | {
                str(name or "").strip().lower()
                for name in set(promoted_names or set())
                if str(name or "").strip()
            }
        )
        self._campaign_hidden_project_approved_count = len(self._campaign_hidden_project_approved_filenames)
    except Exception:
        pass

    if not is_t07_plate_repair:
        try:
            self._refresh_campaign_char_effective_source_after_approval()
        except Exception as e:
            logger.debug(f"Nie udało się odświeżyć źródła E3 po wyjściu z naprawczego Z2: {e}")

    try:
        self._append_z2_trace(
            (
                "t07-repair-return-promote-ok"
                if is_t07_plate_repair
                else "t06-return-promote-ok"
                if is_t06_char_work
                else "char-repair-return-promote-ok"
            ),
            (
                f"run={run_dir} images={approved_images} plates={approved_plates} "
                f"approved_total={int(result.get('total', 0) or 0)}"
            ),
        )
    except Exception:
        pass

    try:
        if is_t07_plate_repair:
            status_message = f"Przeniesiono {approved_images} zdjęć [OK] do projektowej puli YOLO dla T07."
        elif is_t06_char_work:
            status_message = (
                f"Przekazano {approved_images} zdjęć [OK] do źródła Z3 dla bramki "
                "T05."
            )
        else:
            status_message = f"Przeniesiono {approved_images} zdjęć [OK] do katalogu zatwierdzonych dla E3."
        self.app.update_status(
            status_message,
            "success",
        )
    except Exception:
        pass

    if is_t06_char_work:
        try:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            CAMPAIGN.upsert_iteration_state(
                updates={
                    "t06_work_session": {
                        "active": False,
                        "state": "resolved",
                        "resolved_at": now,
                        "updated_at": now,
                        "run_dir": str(run_dir.resolve()),
                        "approved_images": int(approved_images),
                        "approved_plates": int(approved_plates),
                        "last_return_result": dict(result or {}),
                    }
                }
            )
        except Exception as exc:
            logger.debug(f"Nie udało się domknąć znacznika pracy T06 po powrocie do grafu: {exc}")

    return result


def _promote_campaign_t05_ok_to_approved_pool_before_return(self) -> dict:
    """Promote T05 [OK] images when the user formally returns from Z2 to the graph.

    This is intentionally narrower than approving T05. Returning from Z2 should only
    consume approved images into the project YOLO pool; the gate decision remains on
    the campaign graph.
    """
    if self._is_free_mode_session_context():
        return {"ok": False, "reason": "free_mode"}

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        current_step = 0
        iteration_target = ""
        active_project = ""

    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) if graph_gate_id else ""
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

    if not (
        active_project
        and graph_gate_id == "T04"
        and repair_origin_gate_id != "T06"
        and current_step == 2
        and iteration_target == "plate"
    ):
        return {"ok": False, "reason": "not_t04_return"}

    try:
        approval_context = dict(self._get_campaign_step2_approval_context() or {})
    except Exception:
        approval_context = {}

    candidates = [
        approval_context.get("run_dir"),
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
    ]
    try:
        candidates.append(self._get_preferred_annotation_run_dir(require_xml=True))
    except Exception:
        pass

    run_dir = None
    for candidate in candidates:
        run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=True)
        if run_dir is not None:
            break
    if run_dir is None:
        return {"ok": False, "reason": "missing_run"}

    if not self._ensure_preview_edits_saved("powrót do grafu z pracy T05"):
        return {"ok": False, "reason": "save_failed", "abort": True}

    try:
        current_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "current_annotation_run_dir", None),
            require_xml=True,
        )
    except Exception:
        current_run_dir = None
    should_persist_preview_approval = False
    if current_run_dir is not None and self._paths_equivalent(current_run_dir, run_dir):
        try:
            preview_approved = set(self._get_preview_approved_filenames_base() or set())
        except Exception:
            preview_approved = set()
        try:
            approval_version = int(getattr(self, "_preview_approval_version", 0) or 0)
        except Exception:
            approval_version = 0
        should_persist_preview_approval = bool(preview_approved or approval_version > 0)
    if should_persist_preview_approval:
        try:
            self._persist_preview_approved_filenames()
        except Exception:
            pass

    approved_filenames = set()
    try:
        approved_filenames = {
            str(name or "").strip().lower()
            for name in set(self._load_annotation_run_approved_filenames(run_dir) or set())
            if str(name or "").strip()
        }
    except Exception:
        approved_filenames = set()

    try:
        annotations = self._parse_cvat_preview_annotations(run_dir / "annotations.xml")
    except Exception:
        annotations = []

    approved_images = 0
    approved_plates = 0
    if approved_filenames and annotations:
        for ann in list(annotations or []):
            image_name = str(getattr(ann, "filename", "") or "").strip().lower()
            if not image_name or image_name not in approved_filenames:
                continue
            try:
                plate_count = len(self._get_plate_detections(ann))
            except Exception:
                plate_count = 0
            if plate_count <= 0:
                continue
            approved_images += 1
            approved_plates += int(plate_count)

    if approved_images <= 0 or approved_plates <= 0:
        return {"ok": False, "reason": "no_ok_images", "run_dir": str(run_dir)}

    try:
        result = dict(
            self._promote_run_to_campaign_plate_approved_set(
                run_dir,
                force_parse_xml=True,
                project_name=active_project,
            )
            or {}
        )
    except Exception as exc:
        logger.debug(f"Nie udało się przenieść [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do puli YOLO przed powrotem do grafu: {exc}")
        result = {"ok": False, "reason": "promotion_exception"}

    if not bool(result.get("ok")):
        try:
            messagebox.showwarning(
                "Nie zapisano puli YOLO",
                (
                    f"Nie udało się dopisać zatwierdzonych obrazów [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do projektowej puli YOLO.\n\n"
                    "Pozostań w Z2 i spróbuj ponownie wrócić do grafu, żeby nie zgubić wkładu tej pracy."
                ),
                parent=self.frame.winfo_toplevel(),
            )
        except Exception:
            pass
        return {**dict(result or {}), "abort": True, "run_dir": str(run_dir)}

    try:
        self._campaign_hidden_project_approved_filenames = (
            set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
            | set(approved_filenames or set())
        )
        self._campaign_hidden_project_approved_count = len(self._campaign_hidden_project_approved_filenames)
    except Exception:
        pass

    try:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        resolved_run_dir = Path(str(result.get("run_dir") or run_dir))
        CAMPAIGN.upsert_iteration_state(
            updates={
                "t05_work_session": {
                    "active": False,
                    "state": "resolved",
                    "resolved_at": now,
                    "updated_at": now,
                    "run_dir": str(resolved_run_dir.resolve()),
                    "approved_images": int(approved_images),
                    "approved_plates": int(approved_plates),
                    "last_return_result": dict(result or {}),
                }
            }
        )
    except Exception as exc:
        logger.debug(f"Nie udało się domknąć znacznika pracy T05 po powrocie do grafu: {exc}")

    try:
        CAMPAIGN.invalidate_step3_char_source_state_cache()
    except Exception:
        pass

    try:
        self._append_z2_trace(
            "t05-return-promote-ok",
            (
                f"run={run_dir} images={approved_images} plates={approved_plates} "
                f"added={int(result.get('added', 0) or 0)} updated={int(result.get('updated', 0) or 0)}"
            ),
        )
    except Exception:
        pass

    try:
        self.app.update_status(
            f"Przeniesiono {approved_images} zdjęć [OK] z {graph_display_gate_id or graph_gate_id or 'bramki'} do projektowej puli YOLO.",
            "success",
        )
    except Exception:
        pass

    return {**dict(result or {}), "run_dir": str(run_dir), "approved_images": approved_images, "approved_plates": approved_plates}


def _sync_campaign_char_repair_approved_run_to_project_source(
    self,
    run_dir: Path | str | None = None,
    *,
    refresh_effective_source: bool = True,
    reason: str = "",
) -> dict:
    """Silently reconcile approved T06/Z2 run entries with the campaign source.

    The visible Z2 list may already be filtered by project/character-effective
    sources during restore, so this path intentionally promotes from the full
    run XML instead of the currently visible annotations.
    """
    if self._is_free_mode_session_context():
        return {"ok": False, "reason": "free_mode"}

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        current_step = 0
        iteration_target = ""
        active_project = ""

    graph_gate_id = ""
    repair_origin_gate_id = ""
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
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
    except Exception:
        graph_gate_id = ""
        repair_origin_gate_id = ""

    approval_context = {}
    try:
        approval_context = dict(self._get_campaign_step2_approval_context() or {})
    except Exception:
        approval_context = {}
    repair_mode = bool(approval_context.get("repair_mode"))

    candidates = [run_dir]
    try:
        candidates.append(approval_context.get("run_dir"))
    except Exception:
        pass
    candidates.extend(
        [
            getattr(self, "current_annotation_run_dir", None),
            getattr(self, "last_staging_run_dir", None),
        ]
    )

    safe_run_dir = None
    for candidate in candidates:
        safe_run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=True)
        if safe_run_dir is not None:
            break
    if safe_run_dir is None:
        return {"ok": False, "reason": "missing_run"}

    resolved_project = active_project or _resolve_campaign_project_name_from_run_dir(safe_run_dir)
    if not active_project and resolved_project:
        try:
            project_data = dict((CAMPAIGN.state or {}).get("projects", {}).get(resolved_project, {}) or {})
        except Exception:
            project_data = {}
        try:
            current_step = int(project_data.get("current_step", current_step) or current_step or 0)
        except Exception:
            pass
        iteration_target = str(project_data.get("iteration_target", iteration_target) or iteration_target).strip().lower()

    is_t07_plate_repair = bool(
        graph_gate_id == "T04"
        and repair_origin_gate_id == "T06"
        and current_step == 4
        and iteration_target == "plate"
    )
    is_t05_plate_work = bool(
        graph_gate_id == "T04"
        and repair_origin_gate_id != "T06"
        and current_step == 2
        and iteration_target == "plate"
    )
    if is_t05_plate_work:
        try:
            approved_images, approved_plates = self._get_run_plate_approved_counts(safe_run_dir)
        except Exception:
            approved_images, approved_plates = 0, 0
        try:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
            session = dict(iteration_state.get("t05_work_session") or {})
            last_result = dict(session.get("last_return_result") or {})
            session_run = str(session.get("run_dir", "") or "").strip()
            already_resolved = bool(last_result.get("ok"))
            if already_resolved and session_run:
                try:
                    already_resolved = self._paths_equivalent(session_run, safe_run_dir)
                except Exception:
                    already_resolved = str(session_run).strip().lower() == str(safe_run_dir).strip().lower()
            if already_resolved:
                session.update(
                    {
                        "active": False,
                        "state": "resolved",
                        "resolved_at": str(session.get("resolved_at") or now),
                        "updated_at": now,
                        "run_dir": str(safe_run_dir.resolve()),
                        "approved_images": int(session.get("approved_images", approved_images) or approved_images or 0),
                        "approved_plates": int(session.get("approved_plates", approved_plates) or approved_plates or 0),
                        "last_return_result": last_result,
                    }
                )
                CAMPAIGN.upsert_iteration_state(updates={"t05_work_session": session})
                return {**last_result, "ok": True, "reason": "already_resolved", "run_dir": str(safe_run_dir.resolve())}
            for stale_key in ("resolved_at", "closed_at", "completed_at", "last_return_result"):
                session.pop(stale_key, None)
            session.update(
                {
                    "active": True,
                    "state": "interrupted",
                    "source_gate_id": "T04",
                    "working_gate_id": "T04",
                    "edge_key": "e2_to_e4",
                    "run_dir": str(safe_run_dir.resolve()),
                    "approved_images": int(approved_images or 0),
                    "approved_plates": int(approved_plates or 0),
                    "interrupted_at": str(session.get("interrupted_at") or now),
                    "updated_at": now,
                }
            )
            if not str(session.get("started_at") or "").strip():
                session["started_at"] = now
            CAMPAIGN.upsert_iteration_state(updates={"t05_work_session": session})
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać przerwanej sesji T05: {exc}")
        try:
            self._append_z2_trace(
                "t05-work-sync-skip",
                f"reason={reason or 'runtime'} formal_return_required=1 run={safe_run_dir}",
            )
        except Exception:
            pass
        return {"ok": False, "reason": "t05_work_requires_formal_return"}
    if is_t07_plate_repair:
        try:
            approved_images, approved_plates = self._get_run_plate_approved_counts(safe_run_dir)
        except Exception:
            approved_images, approved_plates = 0, 0
        try:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
            session = dict(iteration_state.get("t07_repair_session") or {})
            for stale_key in ("resolved_at", "closed_at", "completed_at", "last_return_result"):
                session.pop(stale_key, None)
            session.update(
                {
                    "active": True,
                    "state": "interrupted",
                    "source_gate_id": "T06",
                    "working_gate_id": "T04",
                    "edge_key": "e4_to_e1",
                    "run_dir": str(safe_run_dir.resolve()),
                    "approved_images": int(approved_images or 0),
                    "approved_plates": int(approved_plates or 0),
                    "interrupted_at": str(session.get("interrupted_at") or now),
                    "updated_at": now,
                }
            )
            if not str(session.get("started_at") or "").strip():
                session["started_at"] = now
            CAMPAIGN.upsert_iteration_state(updates={"t07_repair_session": session})
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać przerwanej sesji naprawczej T07: {exc}")
        try:
            self._append_z2_trace(
                "t07-repair-sync-skip",
                f"reason={reason or 'runtime'} formal_return_required=1 run={safe_run_dir}",
            )
        except Exception:
            pass
        return {"ok": False, "reason": "t07_repair_requires_formal_return"}
    is_char_repair_context = bool(
        iteration_target == "char"
        and (current_step == 3 or graph_gate_id == "T05" or repair_mode)
    )
    is_t06_char_work = bool(
        graph_gate_id == "T05"
        and iteration_target == "char"
        and current_step == 3
    )
    if not resolved_project or not (is_char_repair_context or is_t07_plate_repair):
        return {"ok": False, "reason": "not_repair_context"}

    approved_filenames = set()
    try:
        approved_filenames = set(self._load_annotation_run_approved_filenames(safe_run_dir) or set())
    except Exception:
        approved_filenames = set()

    try:
        approved_filenames = {
            str(name or "").strip().lower()
            for name in approved_filenames
            if str(name or "").strip()
        }
    except Exception:
        approved_filenames = set()

    try:
        current_payload = {
            str(name or "").strip().lower()
            for name in set(self._get_preview_approved_filenames() or set())
            if str(name or "").strip()
        }
        current_payload |= {
            str(name or "").strip().lower()
            for name in set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
            if str(name or "").strip()
        }
        current_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "current_annotation_run_dir", None),
            require_xml=True,
        )
        if (
            str(reason or "").strip().lower() != "pre_restore_filter"
            and
            current_run_dir is not None
            and self._paths_equivalent(current_run_dir, safe_run_dir)
            and len(current_payload) >= len(approved_filenames)
        ):
            self._persist_preview_approved_filenames()
            approved_filenames = set(self._load_annotation_run_approved_filenames(safe_run_dir) or set())
    except Exception:
        pass

    approved_filenames = {
        str(name or "").strip().lower()
        for name in approved_filenames
        if str(name or "").strip()
    }
    if not approved_filenames:
        return {"ok": False, "reason": "no_approved_filenames"}

    if str(reason or "").strip().lower() in {"pre_restore_filter", "restore"} or is_t06_char_work:
        try:
            project_names = {
                CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
                for entry in list(CAMPAIGN.list_plate_approved_entries(resolved_project) or [])
                if isinstance(entry, dict) and CAMPAIGN._normalize_image_set_name(entry.get("image_name", ""))
            }
        except Exception:
            project_names = set()
        try:
            approved_names = {
                CAMPAIGN._normalize_image_set_name(name)
                for name in set(approved_filenames or set())
                if CAMPAIGN._normalize_image_set_name(name)
            }
        except Exception:
            approved_names = set(approved_filenames or set())
        if approved_names and project_names and approved_names.issubset(project_names):
            try:
                self._append_z2_trace(
                    "char-repair-sync-skip",
                    f"reason={reason or 'runtime'} approved={len(approved_names)} already_in_project=1 run={safe_run_dir}",
                )
            except Exception:
                pass
            return {
                "ok": True,
                "reason": "already_project_source",
                "total": len(project_names),
            }

    if is_t06_char_work:
        try:
            approved_images, approved_plates = self._get_run_plate_approved_counts(safe_run_dir)
        except Exception:
            approved_images, approved_plates = 0, 0
        if approved_images <= 0 and approved_filenames:
            approved_images = len(approved_filenames)
        try:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
            session = dict(iteration_state.get("t06_work_session") or {})
            for stale_key in ("resolved_at", "closed_at", "completed_at", "last_return_result"):
                session.pop(stale_key, None)
            session.update(
                {
                    "active": True,
                    "state": "interrupted",
                    "source_gate_id": "T05",
                    "working_gate_id": "T05",
                    "edge_key": "e3_to_e4",
                    "run_dir": str(safe_run_dir.resolve()),
                    "approved_images": int(approved_images or 0),
                    "approved_plates": int(approved_plates or 0),
                    "interrupted_at": str(session.get("interrupted_at") or now),
                    "updated_at": now,
                }
            )
            if not str(session.get("started_at") or "").strip():
                session["started_at"] = now
            CAMPAIGN.upsert_iteration_state(updates={"t06_work_session": session})
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać przerwanej sesji T06: {exc}")
        try:
            self._append_z2_trace(
                "t06-work-sync-skip",
                f"reason={reason or 'runtime'} formal_return_required=1 run={safe_run_dir}",
            )
        except Exception:
            pass
        return {
            "ok": False,
            "reason": "t06_work_requires_formal_return",
            "run_dir": str(safe_run_dir),
            "approved_images": int(approved_images or 0),
            "approved_plates": int(approved_plates or 0),
        }

    try:
        result = self._promote_run_to_campaign_plate_approved_set(
            safe_run_dir,
            force_parse_xml=True,
            project_name=resolved_project,
        )
    except Exception as e:
        logger.debug(
            f"Nie udało się zsynchronizować [OK] z runu Z2 do źródła projektu ({reason}): {e}"
        )
        return {"ok": False, "reason": "promotion_exception"}

    if not bool(dict(result or {}).get("ok")):
        return dict(result or {"ok": False, "reason": "promotion_failed"})

    try:
        self._campaign_hidden_project_approved_filenames = (
            set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
            | approved_filenames
        )
        self._campaign_hidden_project_approved_count = len(self._campaign_hidden_project_approved_filenames)
    except Exception:
        pass

    if refresh_effective_source and not is_t07_plate_repair:
        try:
            self._refresh_campaign_char_effective_source_after_approval()
        except Exception as e:
            logger.debug(
                f"Nie udało się odświeżyć źródła znaków po synchronizacji T06 ({reason}): {e}"
            )

    try:
        self._append_z2_trace(
            "t07-repair-sync-approved-run" if is_t07_plate_repair else "char-repair-sync-approved-run",
            (
                f"reason={reason or 'runtime'} run={safe_run_dir} "
                f"approved={len(approved_filenames)} total={int(dict(result or {}).get('total', 0) or 0)}"
            ),
        )
    except Exception:
        pass

    return result

def _reset_campaign_runtime_state(self, input_dir: Path | None = None):
    self._cancel_deferred_campaign_source_preview_load()
    self._cancel_deferred_campaign_restore_ui()
    self._cancel_deferred_campaign_run_restore()
    self._campaign_context_project_name = ""
    self.current_annotations = []
    self._preview_image_path_map = {}
    self._clear_campaign_manual_reuse_context()
    self._clear_preview_editor_state(clear_dirty=True)
    self.is_processing = False
    self._current_run_manual_template = False
    self.current_annotation_run_dir = None
    self.current_annotation_xml_path = None
    self.last_staging_run_dir = None
    self._manual_review_active = False
    self._manual_review_from_auto = False
    self._manual_review_origin_route = ""
    self._manual_review_export_ready = False
    self._dataset_export_completed = False
    self._last_completed_workflow_route = ""

    try:
        self.current_input_dir = Path(input_dir) if input_dir is not None else None
    except Exception:
        self.current_input_dir = None
    self._pending_source_image_map = {}
    self._annotation_source_input_dir = self.current_input_dir
    self._campaign_auto_manual_overlay_bundle = {}
    self._preview_approved_filenames = set()
    self._campaign_pending_approved_filenames = set()
    self._campaign_hidden_project_approved_filenames = set()
    self._campaign_hidden_char_effective_filenames = set()
    self._campaign_hidden_project_approved_count = 0
    self._campaign_hidden_char_effective_count = 0
    self._campaign_t06_entry_approval_baseline = None
    self._campaign_iteration_manual_filenames = set()
    self._clear_plate_model_runtime_meta()

    try:
        self.workflow_route_var.set("")
    except Exception:
        pass
    try:
        self.workflow_step_var.set("")
    except Exception:
        pass
    try:
        self.free_mode_screen_var.set("route_choice")
    except Exception:
        pass
    try:
        self.manual_history_run_var.set("")
    except Exception:
        pass

    try:
        self.preview_listbox.delete(0, tk.END)
    except Exception:
        pass

    try:
        self.preview_canvas.delete("all")
    except Exception:
        pass

    try:
        self.progress.configure(value=0)
    except Exception:
        pass

    try:
        self._set_progress_counters(0, 0, 0)
    except Exception:
        pass

    try:
        self._set_status_label_state("Gotowy do uruchomienia", "neutral")
    except Exception:
        pass

    try:
        self._set_post_annotation_hint("")
    except Exception:
        pass

    try:
        self.start_btn.config(state=tk.NORMAL)
    except Exception:
        pass

    try:
        self.stop_btn.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self.approve_btn.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self.export_plate_dataset_btn.config(state=tk.DISABLED)
    except Exception:
        pass

    graph_gate_active = False
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_active = bool(str(graph_context.get("graph_gate_id") or "").strip())
    except Exception:
        graph_gate_active = False
    if not graph_gate_active:
        try:
            self.approve_gate_hint_var.set("")
        except Exception:
            pass

        approve_hint_lbl = getattr(self, "approve_gate_hint_lbl", None)
        if approve_hint_lbl is not None:
            self._set_inline_label_state(approve_hint_lbl, tone="muted", emphasis=False)
        self._set_approve_hint_box_state("muted")
        self._set_widget_packed(getattr(self, "approve_hint_box", None), False)

def _build_campaign_z2_gate_overlay_state(self) -> dict:
    if self._is_free_mode_session_context():
        return {}
    try:
        approval_context = self._get_campaign_step2_approval_context()
    except Exception:
        approval_context = {}
    if not bool(approval_context.get("project_active")):
        return {}

    current_step = int(approval_context.get("current_step") or 0)
    repair_mode = bool(approval_context.get("repair_mode"))
    iteration_target = str(approval_context.get("iteration_target") or "").strip().lower()
    graph_context = {}
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
    graph_gate_label = str(graph_context.get("graph_gate_label") or "").strip()
    graph_repair_origin_edge_key = str(
        graph_context.get("repair_origin_edge_key")
        or graph_context.get("source_graph_edge_key")
        or ""
    ).strip()
    graph_repair_origin_gate_id = str(
        graph_context.get("repair_origin_gate_id")
        or graph_context.get("source_graph_gate_id")
        or ""
    ).strip().upper()
    graph_repair_origin_gate_id = campaign_gate_id_for_edge(
        graph_repair_origin_edge_key,
        graph_repair_origin_gate_id,
    )
    graph_gate_is_t04 = graph_gate_id == "T03"
    graph_gate_is_t05 = graph_gate_id == "T04"
    graph_gate_is_t06 = graph_gate_id == "T05"
    graph_gate_is_t05_repair_from_t07 = bool(graph_gate_is_t05 and graph_repair_origin_gate_id == "T06")
    graph_gate_known = bool(graph_gate_is_t04 or graph_gate_is_t05 or graph_gate_is_t06)
    graph_gate_copy_id = graph_display_gate_id or graph_gate_id
    graph_context_active = bool(graph_gate_id or graph_context.get("graph_edge_key"))

    if iteration_target not in {"plate", "char"}:
        if graph_gate_is_t05 or graph_gate_is_t06:
            iteration_target = "char"
        elif graph_gate_is_t04 or graph_context_active:
            iteration_target = "plate"
        else:
            return {}
    if not (current_step == 2 or repair_mode or graph_context_active):
        return {}

    min_images = 0
    min_plate_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    gate_metric = "plates"
    approval_run_dir = approval_context.get("run_dir")
    approval_xml_exists = bool(approval_run_dir and (Path(approval_run_dir) / "annotations.xml").exists())
    project_approved_images = 0
    project_approved_plates = 0
    current_iteration_num = 0
    try:
        from ..campaign_manager import CAMPAIGN
        current_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
        project_approved_images = int(approved_stats.get("images", 0) or 0)
        project_approved_plates = int(approved_stats.get("plates", 0) or 0)
    except Exception:
        current_iteration_num = 0
        project_approved_images = 0
        project_approved_plates = 0

    run_ok_images = 0
    run_ok_plates = 0
    if current_step == 2 and not repair_mode and self.current_annotations:
        try:
            run_ok_images, run_ok_plates = self._get_current_preview_plate_approved_counts()
        except Exception:
            run_ok_images, run_ok_plates = 0, 0
    else:
        try:
            run_ok_images, run_ok_plates = self._get_run_plate_approved_counts(approval_run_dir)
        except Exception:
            run_ok_images, run_ok_plates = 0, 0

    if graph_gate_is_t06:
        try:
            t06_token = f"{Path(approval_run_dir).resolve() if approval_run_dir else ''}|T06"
        except Exception:
            t06_token = f"{approval_run_dir or ''}|T06"
        approval_baseline = getattr(self, "_campaign_t06_entry_approval_baseline", None)
        source_baseline = getattr(self, "_campaign_t06_entry_source_baseline", None)
        if (
            isinstance(approval_baseline, dict)
            and approval_baseline.get("token") == t06_token
            and isinstance(source_baseline, dict)
            and source_baseline.get("token") == t06_token
        ):
            project_approved_images = max(0, int(source_baseline.get("images", 0) or 0))
            project_approved_plates = max(0, int(source_baseline.get("plates", 0) or 0))
            run_ok_images = int(run_ok_images or 0) - int(approval_baseline.get("images", 0) or 0)
            run_ok_plates = int(run_ok_plates or 0) - int(approval_baseline.get("plates", 0) or 0)

    effective_images = max(0, int(project_approved_images or 0) + int(run_ok_images or 0))
    effective_plates = max(0, int(project_approved_plates or 0) + int(run_ok_plates or 0))
    xml_required = bool(int(current_iteration_num or 0) <= 1)
    xml_missing = bool(xml_required and not approval_xml_exists)
    if iteration_target == "char":
        missing_images = 0
        missing_plates = max(0, int(min_char_plates) - int(effective_plates or 0))
        ready = bool(
            int(effective_plates or 0) >= int(min_char_plates)
            and not xml_missing
            and not bool(getattr(self, "is_processing", False))
        )
    else:
        missing_images = 0
        missing_plates = max(0, int(min_plate_plates) - int(effective_plates or 0))
        ready = bool(
            int(effective_plates or 0) >= int(min_plate_plates)
            and not xml_missing
            and not bool(getattr(self, "is_processing", False))
        )

    missing_to_open = int(missing_plates)
    quality_score = max(0, int(effective_plates or 0))
    try:
        quality_info = CONFIG.describe_yolo_pose_dataset_quality(quality_score)
    except Exception:
        quality_info = {}
    next_quality_label = str(quality_info.get("next_label", "") or "").strip()
    try:
        missing_next_quality = max(0, int(quality_info.get("missing_next", 0) or 0))
    except Exception:
        missing_next_quality = 0

    focus_state = build_campaign_gate_focus_state(missing_to_open, quality_info)
    missing_focus_label = str(focus_state.get("label") or "DO MIN.")
    missing_focus_row_label = str(focus_state.get("row_label") or "Do otwarcia bramki brakuje")
    missing_focus_value = int(focus_state.get("value") or 0)
    missing_focus_text = str(focus_state.get("text") or "")
    missing_focus_tone = str(focus_state.get("tone") or "warning")
    if ready:
        if graph_gate_is_t05_repair_from_t07:
            message = "Materiał tablic jest gotowy; wróć do bramki T07."
            detail = "To tryb naprawczy. Decyzję końcową podejmujesz na bramce T07."
        elif graph_gate_known:
            message = f"Bramka {graph_gate_copy_id} jest otwarta."
            detail = f"Wróć do mapy kampanii i użyj pola Zatwierdź na bramce {graph_gate_id}."
        else:
            message = (
                "Bramka jest gotowa do zamknięcia."
                if iteration_target == "char"
                else "Bramka jest gotowa do zamknięcia."
            )
            detail = ""
        tone = "success"
    elif xml_missing and missing_to_open <= 0:
        message = (
            f"Do otwarcia bramki {graph_gate_copy_id} brakuje pliku anotacji XML."
            if graph_gate_known
            else "Do otwarcia bramki brakuje pliku anotacji XML."
        )
        detail = "Utwórz XML anotacji tablic w Z2."
        tone = "warning"
    elif iteration_target == "char":
        noun = "tablicy" if missing_to_open == 1 else "tablic"
        message = (
            f"Do otwarcia bramki {graph_gate_copy_id} brakuje {missing_to_open} {noun}."
            if graph_gate_known
            else f"Do otwarcia bramki brakuje {missing_to_open} {noun}."
        )
        detail = "Zatwierdzaj obrazy z poprawnymi ramkami tablic jako OK."
        tone = "warning"
    else:
        noun = "tablicy" if missing_to_open == 1 else "tablic"
        message = (
            f"Do otwarcia bramki {graph_gate_copy_id} brakuje {missing_to_open} {noun}."
            if graph_gate_known
            else f"Do otwarcia bramki brakuje {missing_to_open} {noun}."
        )
        detail = "Zatwierdź zdjęcia z ramkami tablic jako OK."
        tone = "warning"

    gate_title = "NAPRAWA T07" if graph_gate_is_t05_repair_from_t07 else (f"BRAMKA {graph_gate_copy_id}" if graph_gate_known else "BRAMKA GRAFU")
    instruction = ""
    if graph_gate_is_t05_repair_from_t07:
        instruction = "Wyjście z Z2 prowadzi do bramki T07."
    elif graph_gate_known:
        instruction = (
            f"Praca nad otwarciem bramki {graph_gate_copy_id}."
            if not ready
            else f"Bramka {graph_gate_id} gotowa do zamknięcia."
        )

    if graph_gate_id == "T03":
        message = str(message or "").replace("T04", "T03")
        detail = str(detail or "").replace("T04", "T03")
        instruction = str(instruction or "").replace("T04", "T03")
        gate_title = str(gate_title or "").replace("T04", "T03")
    if graph_gate_id == "T04":
        message = str(message or "").replace("T05", "T04")
        detail = str(detail or "").replace("T05", "T04")
        instruction = str(instruction or "").replace("T05", "T04")
        gate_title = str(gate_title or "").replace("T05", "T04")
    if graph_gate_id == "T05":
        message = str(message or "").replace("T06", "T05")
        detail = str(detail or "").replace("T06", "T05")
        instruction = str(instruction or "").replace("T06", "T05")
        gate_title = str(gate_title or "").replace("T06", "T05")
    if graph_gate_is_t05_repair_from_t07:
        message = str(message or "").replace("T07", "T06")
        detail = str(detail or "").replace("T07", "T06")
        instruction = str(instruction or "").replace("T07", "T06")
        gate_title = str(gate_title or "").replace("T07", "T06")

    if graph_gate_id and graph_gate_copy_id and graph_gate_copy_id != graph_gate_id:
        message = str(message or "").replace(graph_gate_id, graph_gate_copy_id)
        detail = str(detail or "").replace(graph_gate_id, graph_gate_copy_id)
        instruction = str(instruction or "").replace(graph_gate_id, graph_gate_copy_id)
        gate_title = str(gate_title or "").replace(graph_gate_id, graph_gate_copy_id)

    return {
        "visible": True,
        "ready": bool(ready),
        "tone": tone,
        "title": gate_title,
        "gate_id": ("T06" if graph_gate_is_t05_repair_from_t07 else graph_display_gate_id),
        "source_gate_id": graph_gate_id,
        "gate_label": graph_gate_label,
        "status": "OTWARTA" if ready else "ZAMKNIĘTA",
        "message": message,
        "detail": detail,
        "instruction": instruction,
        "approved_images": int(effective_images or 0),
        "approved_plates": int(effective_plates or 0),
        "required_images": int(min_images or 0),
        "required_plates": int(min_char_plates if iteration_target == "char" else min_plate_plates),
        "missing_images": int(missing_images or 0),
        "missing_plates": int(missing_plates or 0),
        "missing_to_open": int(missing_to_open or 0),
        "missing_focus_label": missing_focus_label,
        "missing_focus_row_label": missing_focus_row_label,
        "missing_focus_value": int(missing_focus_value or 0),
        "missing_focus_text": missing_focus_text,
        "missing_focus_tone": missing_focus_tone,
        "next_quality_label": next_quality_label,
        "missing_next_quality": int(missing_next_quality or 0),
        "gate_metric": gate_metric,
        "xml": "XML: OK" if approval_xml_exists else ("XML: wymagany" if xml_required else "XML: brak"),
        "iteration": int(current_iteration_num or 0),
    }


def _open_campaign_plate_model_check_dialog(owner, model_path: Path) -> tk.Toplevel | None:
    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    parent = getattr(owner, "frame", None)
    dialog = None
    try:
        dialog = tk.Toplevel(parent)
        dialog.withdraw()
        dialog.title("Sprawdzam model tablic")
        dialog.resizable(False, False)
        panel_bg = palette.get("panel", "#252526")
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        dialog.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
        try:
            dialog.transient(getattr(owner.app, "root", None) or parent.winfo_toplevel())
        except Exception:
            pass

        body = tk.Frame(
            dialog,
            bg=panel_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
        )
        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        content = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
        content.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)
        tk.Label(
            content,
            text="Sprawdzam model tablic",
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

        try:
            model_name = Path(model_path).name
        except Exception:
            model_name = str(model_path or "").strip()
        tk.Label(
            content,
            text=(
                f"Wybrany plik: {model_name}\n"
                "Czytam metadane modelu. Jeśli nie ma lekkiego pliku metadanych, "
                "program może chwilę ładować model .pt."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=460,
        ).pack(anchor=tk.W, fill=tk.X, pady=(8, 12))
        progress = ttk.Progressbar(content, mode="indeterminate", length=360)
        progress.pack(fill=tk.X)
        try:
            progress.start(12)
        except Exception:
            pass
        try:
            owner._fit_borderless_dialog(dialog, parent=parent, min_width=520, min_height=190)
        except Exception:
            try:
                dialog.geometry("520x190")
            except Exception:
                pass
        dialog.deiconify()
        dialog.lift()
        dialog.update_idletasks()
        dialog.update()
        try:
            owner.app.update_status(f"Sprawdzam model tablic: {model_name}", "info")
        except Exception:
            pass
        return dialog
    except Exception:
        try:
            if dialog is not None:
                dialog.destroy()
        except Exception:
            pass
        return None


def _close_campaign_plate_model_check_dialog(dialog) -> None:
    if dialog is None:
        return
    try:
        dialog.destroy()
    except Exception:
        pass


def _apply_campaign_plate_auto_model_choice(self, model_path: Path, *, adopt_to_project: bool) -> bool | str:
    safe_path = Path(model_path)
    if not safe_path.exists():
        messagebox.showerror("Brak modelu", "Wybrany model tablic nie istnieje.")
        return False

    busy_dialog = _open_campaign_plate_model_check_dialog(self, safe_path)
    try:
        ok, message, info = validate_model_file(safe_path)
    finally:
        _close_campaign_plate_model_check_dialog(busy_dialog)
    if not ok:
        messagebox.showerror("Nieprawidłowy model", f"Nie udało się użyć wybranego modelu:\n{message}")
        return False

    task = str(info.get("task") or info.get("type") or "").strip().lower()
    if task and task not in {"pose", "unknown"}:
        proceed = messagebox.askyesno(
            "Nietypowy typ modelu",
            "Wybrany plik nie wygląda na model POSE tablic.\n\n"
            "Może to działać nieprawidłowo w autoanotacji Z2.\n\n"
            "Czy mimo to użyć tego modelu?",
            parent=self.frame,
        )
        if not proceed:
            return "retry"

    if not self._confirm_campaign_plate_model_identity_choice(safe_path, info, adopt_to_project=adopt_to_project):
        return "retry"

    if adopt_to_project:
        try:
            from ..campaign_manager import CAMPAIGN
            if CAMPAIGN.get_active_project_name():
                safe_path = self._adopt_campaign_plate_model_into_project(safe_path)
                CAMPAIGN.set_global_model("plate", str(safe_path))
        except Exception as e:
            messagebox.showerror(
                "Błąd adopcji modelu",
                f"Nie udało się skopiować modelu do katalogu projektu:\n{e}",
            )
            return False

    identity_label = format_yolo_model_identity(info)
    runtime_source = "external"
    runtime_scope = "project" if adopt_to_project else "run"
    try:
        project_model_path = self._get_campaign_project_plate_model_path()
        if (
            not adopt_to_project
            and project_model_path is not None
            and safe_path.resolve() == project_model_path.resolve()
        ):
            runtime_source = "project"
            runtime_scope = "project"
    except Exception:
        pass
    self._remember_plate_model_runtime_meta(
        model_path=safe_path,
        identity=identity_label,
        source=runtime_source,
        scope=runtime_scope,
    )
    self.plate_custom_var.set(str(safe_path))
    try:
        current_run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None))
        if current_run_dir is not None:
            self._update_annotation_run_manifest(
                current_run_dir,
                **self._collect_plate_model_manifest_fields(),
            )
    except Exception:
        pass

    try:
        self._sync_campaign_iteration_artifact_registry(
            run_dir=getattr(self, "current_annotation_run_dir", None),
            input_dir=getattr(self, "current_input_dir", None),
        )
    except Exception:
        pass

    try:
        if "campaign" in getattr(self.app, "tabs", {}):
            self.app.tabs["campaign"]._refresh_dashboard()
    except Exception:
        pass

    try:
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()
    except Exception:
        pass
    try:
        self._advance_campaign_auto_step_after_plate_model_selection()
    except Exception:
        pass

    try:
        source_label = "Aktywny model projektu" if runtime_scope == "project" else "Model tylko dla tego runu Z2"
        if identity_label:
            source_label = f"{source_label}: {identity_label}"
        self.app.update_status(
            f"{source_label} | {safe_path.name}",
            "info",
        )
    except Exception:
        pass

    return True
