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
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
from .z2_model_dialogs import (
    _prompt_campaign_plate_auto_model_choice,
    _prompt_campaign_return_to_wizard_ok_modal,
    _confirm_campaign_plate_model_identity_choice,
    _prompt_z2_text_input,
    _extract_plate_model_metrics_from_rows,
    _build_auto_annotation_model_quality_rows,
)
from .z2_annotation_process import (
    _refresh_step2_action_states,
    _start_annotation,
    _approve_annotation_stage,
    _process_thread,
    _finish,
    _switch_annotation_input_dir,
    _go_to_next_workflow_step,
)
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

NAV_BUTTON_WIDTH = 18
YOLO = None


def _import_external_annotation_run_to_workspace(
    self,
    source_run_dir: Path | str | None,
    *,
    compatible_images_dir: Path | str | None = None,
    allowed_normalized_names: set[str] | None = None,
    copy_images: bool = True,
    progress_callback=None,
) -> tuple[Path | None, str, bool]:
    def _notify_progress(percent: float, message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            value = max(0.0, min(100.0, float(percent or 0.0)))
        except Exception:
            value = 0.0
        try:
            progress_callback(value, str(message or ""))
        except Exception:
            pass

    _notify_progress(2, "Sprawdzam katalog runu AT...")
    source_run_dir = self._resolve_existing_run_dir(source_run_dir)
    if source_run_dir is None:
        return None, "Nie znaleziono wskazanego katalogu runu.", False

    source_xml_path = source_run_dir / "annotations.xml"
    if not source_xml_path.exists():
        return None, "Wybrany katalog nie zawiera pliku annotations.xml.", False

    _notify_progress(8, "Odczytuję annotations.xml...")
    try:
        annotations = self._parse_cvat_preview_annotations(source_xml_path)
    except Exception as e:
        return None, f"Nie udalo sie odczytac annotations.xml:\n{e}", False

    allowed_names = {
        str(name or "").strip().lower()
        for name in set(allowed_normalized_names or set())
        if str(name or "").strip()
    }
    if allowed_names:
        try:
            from ..campaign_manager import CAMPAIGN
            normalize_name = CAMPAIGN._normalize_image_set_name
        except Exception:
            normalize_name = lambda value: Path(str(value or "")).name.strip().lower()

        annotations = [
            ann for ann in annotations
            if normalize_name(str(getattr(ann, "filename", "") or "")) in allowed_names
        ]
    _notify_progress(18, "Filtruję AT pasujące do aktualnego zbioru obrazów...")

    if not annotations:
        return None, "Wybrany run nie zawiera obrazow zgodnych z wybranym katalogiem zdjęć.", False

    source_manifest = self._load_annotation_run_manifest(source_run_dir)
    image_roots = self._get_external_run_image_roots(
        source_run_dir,
        source_manifest,
        compatible_images_dir=compatible_images_dir,
    )

    resolved_images, missing_images = self._resolve_external_run_source_images(
        annotations,
        image_roots,
    )
    _notify_progress(32, "Sprawdzam powiązanie AT z obrazami...")
    should_copy_images = bool(copy_images)
    relative_name_map: dict[str, str] = {}
    used_import_paths: set[str] = set()
    if should_copy_images:
        total_images = max(1, len(resolved_images))
        for idx, (filename, _raw_path, source_image_path) in enumerate(resolved_images, start=1):
            relative_path = self._build_safe_imported_image_relative_path(
                filename,
                index=idx - 1,
                used_paths=used_import_paths,
            )
            relative_name_map[filename] = str(relative_path).replace("\\", "/")
            resolved_images[idx - 1] = (filename, relative_path, source_image_path)
            if idx == 1 or idx == total_images or idx % 25 == 0:
                _notify_progress(34 + (idx / total_images) * 12, "Przygotowuję mapowanie obrazów importu...")
    else:
        for filename, _raw_path, _source_image_path in resolved_images:
            relative_name_map[filename] = str(filename or "").replace("\\", "/")
        _notify_progress(46, "Przygotowuję import AT bez kopiowania obrazów...")

    if missing_images:
        preview_missing = "\n".join(missing_images[:5])
        extra_missing = len(missing_images) - min(len(missing_images), 5)
        suffix = f"\n... i jeszcze {extra_missing} plikow." if extra_missing > 0 else ""
        return (
            None,
            "Nie mozna bezpiecznie zaimportowac tego runu, bo brakuje obrazow "
            f"wzgledem annotations.xml.\n\nBrakujace pliki:\n{preview_missing}{suffix}",
            True,
        )

    imported_run_dir = None
    try:
        imported_run_dir = self._allocate_annotation_run_dir(
            self._get_annotation_output_base_dir(),
            suffix="import",
        )
        imported_images_dir = imported_run_dir / "images"

        if should_copy_images:
            imported_images_dir.mkdir(parents=True, exist_ok=True)
            total_images = max(1, len(resolved_images))
            for idx, (_original_name, relative_path, source_image_path) in enumerate(resolved_images, start=1):
                target_image_path = imported_images_dir / relative_path
                target_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image_path, target_image_path)
                if idx == 1 or idx == total_images or idx % 10 == 0:
                    _notify_progress(46 + (idx / total_images) * 20, "Kopiuję obrazy do roboczego runu importu...")
        else:
            _notify_progress(66, "Tworzę roboczy run importu AT...")

        imported_xml_path = imported_run_dir / "annotations.xml"
        _notify_progress(70, "Przepisuję annotations.xml tylko dla pasujących obrazów...")
        xml_tree = ET.parse(source_xml_path)
        xml_root = xml_tree.getroot()
        allowed_original_names = set(relative_name_map.keys())
        parent_by_child = {child: parent for parent in xml_root.iter() for child in parent}
        for image_el in list(xml_root.findall(".//image")):
            original_name = str(image_el.get("name", "") or "").strip()
            if original_name not in allowed_original_names:
                parent = parent_by_child.get(image_el)
                if parent is not None:
                    parent.remove(image_el)
                continue
            if should_copy_images and original_name in relative_name_map:
                image_el.set("name", relative_name_map[original_name])
        xml_tree.write(imported_xml_path, encoding="utf-8", xml_declaration=True)
        _notify_progress(82, "Zapisano annotations.xml roboczego importu...")

        source_report_path = source_run_dir / "report.txt"
        if source_report_path.exists():
            shutil.copy2(source_report_path, imported_run_dir / "report.txt")

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        successful_images = sum(1 for ann in annotations if bool(getattr(ann, "is_successful", False)))
        _images_with_plates, total_plates = self._count_plate_annotations(annotations)
        manifest_input_dir = (
            str(imported_images_dir.resolve())
            if should_copy_images
            else (
                str(Path(compatible_images_dir).resolve())
                if compatible_images_dir
                else str(source_manifest.get("input_dir") or "").strip()
            )
        )
        input_scope_count = len(allowed_names) if allowed_names else len(annotations)
        input_scope_filenames = sorted(
            str(name or "").strip().lower()
            for name in set(allowed_names or [])
            if str(name or "").strip()
        )
        imported_manifest = {
            "input_dir": manifest_input_dir,
            "run_dir": str(imported_run_dir.resolve()),
            "mode": str(source_manifest.get("mode") or self.mode_var.get() or "").strip(),
            "device": str(source_manifest.get("device") or self._get_effective_yolo_device_choice() or "").strip(),
            "annotation_run_type": str(source_manifest.get("annotation_run_type") or "imported_run").strip() or "imported_run",
            "manual_xml_template": bool(source_manifest.get("manual_xml_template", False)),
            "manual_vehicle_assist": bool(source_manifest.get("manual_vehicle_assist", False)),
            "has_manual_edits": bool(source_manifest.get("has_manual_edits", False)),
            "last_manual_edit_at": str(source_manifest.get("last_manual_edit_at") or "").strip(),
            "last_manual_edit_kind": str(source_manifest.get("last_manual_edit_kind") or "import").strip(),
            "run_status": "completed",
            "completed_at": str(source_manifest.get("completed_at") or source_manifest.get("generated_at") or now_iso).strip(),
            "last_error": "",
            "result_total_images": len(annotations),
            "result_successful_images": successful_images,
            "result_total_plates": total_plates,
            "input_scope_count": int(input_scope_count),
            "selected_count": int(input_scope_count),
            "source_plan_total_count": int(input_scope_count),
            "source_manifest_count": int(input_scope_count),
            "input_scope_filenames": input_scope_filenames,
            "imported_annotation_images": len(annotations),
            "resume_preview_index": -1,
            "resume_preview_filename": "",
            "resume_preview_saved_at": "",
            "manual_touched_filenames": sorted(
                set(source_manifest.get("manual_touched_filenames") or [])
                or self._collect_preview_manually_touched_filenames(annotations, include_dirty=False)
            ),
            "approved_filenames": [],
            "imported_source_approved_filenames": sorted(
                str(relative_name_map.get(str(name or "").strip(), str(name or "").strip()) or "").lower()
                for name in list(source_manifest.get("approved_filenames") or [])
                if str(name or "").strip() in allowed_original_names
            ),
            "generated_at": str(source_manifest.get("generated_at") or now_iso).strip(),
            "imported_at": now_iso,
            "imported_from_run_dir": str(source_run_dir.resolve()),
            "images_copied_to_import_run": bool(should_copy_images),
            "imported_source_input_dir": (
                str(Path(compatible_images_dir).resolve())
                if compatible_images_dir
                else str(source_manifest.get("input_dir") or "").strip()
            ),
        }
        self._annotation_run_manifest_path(imported_run_dir).write_text(
            json.dumps(imported_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _notify_progress(96, "Zapisano manifest importu AT...")
        try:
            if self._is_free_mode_session_context():
                self._register_free_mode_branch_artifact(imported_run_dir, artifact_type="owned_run_dirs")
        except Exception:
            pass
        _notify_progress(100, "Import AT przygotowany do kontroli w Z2.")
        return imported_run_dir, "", False
    except Exception as e:
        if imported_run_dir is not None:
            try:
                shutil.rmtree(imported_run_dir)
            except Exception:
                pass
        return None, f"Nie udalo sie zaimportowac runu do workspace:\n{e}", False
