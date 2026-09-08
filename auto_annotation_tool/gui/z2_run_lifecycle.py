#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 annotation run lifecycle helpers extracted from tab_annotation.py."""

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

def _find_reused_manual_source_run(self, expected_input_dir: Path | None) -> Path | None:
    def matches_expected_input(run_dir: Path | None) -> bool:
        if run_dir is None:
            return False
        if expected_input_dir is None:
            return True
        try:
            return self._annotation_run_matches_input(run_dir, expected_input_dir)
        except Exception:
            return False

    try:
        from ..campaign_manager import CAMPAIGN

        stored_source = CAMPAIGN.get_last_plate_manual_source()
    except Exception:
        stored_source = {}

    for candidate in (
        str(stored_source.get("source_run_path") or "").strip(),
        str(stored_source.get("source_xml_path") or "").strip(),
    ):
        if not candidate:
            continue
        try:
            run_dir = Path(candidate)
            if run_dir.suffix.lower() == ".xml":
                run_dir = run_dir.parent
        except Exception:
            continue
        if run_dir.exists() and run_dir.is_dir() and (run_dir / "annotations.xml").exists() and matches_expected_input(run_dir):
            return run_dir

    try:
        from ..campaign_manager import CAMPAIGN

        search_roots = []
        for root_candidate in (CAMPAIGN.get_dir("auto_ann"), CAMPAIGN.get_staging_dir("auto_ann")):
            if root_candidate is not None:
                search_roots.append(Path(root_candidate))
    except Exception:
        search_roots = []

    candidates = []
    for root in search_roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
            run_paths = PROJECT_CACHE.list_annotation_run_dirs(root, require_xml=True)
        except Exception:
            continue

        for run_dir in run_paths:
            try:
                if (
                    not run_dir.is_dir()
                    or not (run_dir / "annotations.xml").exists()
                    or not matches_expected_input(run_dir)
                ):
                    continue
            except Exception:
                continue

            manifest = self._load_annotation_run_manifest(run_dir)
            if not self._annotation_run_manifest_has_manual_value(manifest):
                continue

            stamp = str(manifest.get("last_manual_edit_at") or "").strip()
            try:
                score = datetime.datetime.fromisoformat(stamp).timestamp() if stamp else float(run_dir.stat().st_mtime)
            except Exception:
                try:
                    score = float(run_dir.stat().st_mtime)
                except Exception:
                    score = 0.0
            candidates.append((score, run_dir.name, run_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _find_reused_training_source_run(self, expected_input_dir: Path | None) -> Path | None:
    def matches_expected_input(run_dir: Path | None) -> bool:
        if run_dir is None:
            return False
        if expected_input_dir is None:
            return True
        try:
            return self._annotation_run_matches_input(run_dir, expected_input_dir)
        except Exception:
            return False

    try:
        from ..campaign_manager import CAMPAIGN

        stored_source = CAMPAIGN.get_last_plate_training_source()
    except Exception:
        stored_source = {}

    stored_run = self._resolve_existing_run_dir(stored_source.get("source_run_path"))
    if matches_expected_input(stored_run):
        return stored_run

    stored_dataset = str(stored_source.get("dataset_path") or "").strip()
    if stored_dataset:
        dataset_run = self._resolve_plate_source_run_from_dataset(Path(stored_dataset))
        if matches_expected_input(dataset_run):
            return dataset_run

    try:
        from ..campaign_manager import CAMPAIGN
        from ..training import TrainingHistory

        runs_root = CAMPAIGN.get_dir("runs")
        if runs_root is None:
            return None

        history = TrainingHistory(history_dir=Path(runs_root) / "plates")
        for run in history.get_all_runs():
            dataset_value = str(getattr(run, "dataset_path", "") or "").strip()
            if not dataset_value:
                continue
            dataset_run = self._resolve_plate_source_run_from_dataset(Path(dataset_value))
            if matches_expected_input(dataset_run):
                return dataset_run
    except Exception as e:
        logger.debug(f"Nie udalo sie odczytac historii treningow tablic: {e}")

    return None


def _find_latest_annotation_run_dir(self, base_dir: Path | None = None) -> Path | None:
    roots = []
    if base_dir is not None:
        try:
            candidate_root = Path(base_dir)
        except Exception:
            candidate_root = None
        if candidate_root is not None and self._path_is_within_any(candidate_root, self._get_annotation_run_roots()):
            roots.append(candidate_root)
    else:
        roots.extend(self._get_annotation_run_roots())

    roots = self._dedupe_paths(roots)
    if not roots:
        return None

    candidates = []
    for root in roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
        except Exception:
            continue

        try:
            run_paths = PROJECT_CACHE.list_annotation_run_dirs(root, require_xml=True)
        except Exception:
            continue

        for path in run_paths:
            try:
                stamp = path.stat().st_mtime
            except Exception:
                stamp = 0
            candidates.append((stamp, path.name, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _annotation_run_matches_input(self, run_dir: Path, input_dir: Path | None) -> bool:
    if input_dir is None:
        return True
    return self._annotation_run_matches_expected_input_dir(run_dir, input_dir)


def _find_latest_annotation_run_for_input(
    self,
    input_dir: Path | None,
    search_roots: list[Path] | None = None,
) -> Path | None:
    if input_dir is None:
        return None

    candidates = []
    roots = list(search_roots or [])
    allowed_roots = self._get_annotation_run_roots()

    for root in roots:
        try:
            root = Path(root)
        except Exception:
            continue

        if not self._path_is_within_any(root, allowed_roots):
            continue

        try:
            if not root.exists() or not root.is_dir():
                continue
        except Exception:
            continue

        try:
            run_paths = PROJECT_CACHE.list_annotation_run_dirs(root, require_xml=True)
        except Exception:
            continue

        for path in run_paths:
            try:
                if not path.is_dir() or not (path / "annotations.xml").exists():
                    continue
                if not self._annotation_run_matches_input(path, input_dir):
                    continue
                stamp = path.stat().st_mtime
            except Exception:
                continue
            candidates.append((stamp, path.name, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _score_annotation_run_restore_candidate(self, run_dir: Path | None) -> float:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return float("-inf")

    manifest = {}
    try:
        manifest = self._load_annotation_run_manifest(safe_run_dir)
    except Exception:
        manifest = {}

    for key in (
        "last_manual_edit_at",
        "resume_preview_saved_at",
        "completed_at",
        "generated_at",
        "created_at",
    ):
        raw_value = str(manifest.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            return float(datetime.datetime.fromisoformat(raw_value).timestamp())
        except Exception:
            continue

    try:
        return float(safe_run_dir.stat().st_mtime)
    except Exception:
        return float("-inf")


def _resolve_best_free_mode_restore_run(self) -> Path | None:
    if self._is_free_mode_session_context():
        # Reopening a tab must not replace the selected run with a newer one.
        return self._get_preferred_annotation_run_dir(require_xml=True)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(raw_candidate) -> None:
        safe_candidate = self._resolve_safe_annotation_run_dir(raw_candidate, require_xml=True)
        if safe_candidate is None:
            return
        try:
            key = str(safe_candidate.resolve())
        except Exception:
            key = str(safe_candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(safe_candidate)

    add_candidate(getattr(self, "current_annotation_run_dir", None))
    add_candidate(getattr(self, "last_staging_run_dir", None))
    add_candidate(str(self.plate_dataset_run_var.get() or "").strip())

    for entry in list(getattr(self, "_manual_review_history_entries", []) or []):
        add_candidate(entry.get("run_dir"))

    current_input = getattr(self, "current_input_dir", None)
    if current_input is not None:
        try:
            latest_for_input = self._find_latest_annotation_run_for_input(
                Path(current_input),
                [self._get_annotation_output_base_dir()],
            )
        except Exception:
            latest_for_input = None
        add_candidate(latest_for_input)

    try:
        latest_overall = self._find_latest_annotation_run_dir(self._get_annotation_output_base_dir())
    except Exception:
        latest_overall = None
    add_candidate(latest_overall)

    if not candidates:
        return None

    scored_candidates = []
    for candidate in candidates:
        scored_candidates.append(
            (
                self._score_annotation_run_restore_candidate(candidate),
                str(candidate.name or ""),
                candidate,
            )
        )

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_candidate = scored_candidates[0][2]
    try:
        self._append_z2_trace(
            "free-restore-candidate",
            f"chosen={best_candidate} score={scored_candidates[0][0]:.0f} total={len(scored_candidates)}",
        )
    except Exception:
        pass
    return best_candidate


def _get_plate_dataset_base_dir(self) -> Path:
    if not self._is_free_mode_session_context():
        try:
            from ..campaign_manager import CAMPAIGN

            if CAMPAIGN.get_active_project_name():
                datasets_dir = CAMPAIGN.get_dir("datasets")
                if datasets_dir is not None:
                    return Path(datasets_dir)
        except Exception:
            pass

    return Path(CONFIG.get_datasets_dir("plate"))


def _get_manual_plate_stage_dir(self) -> Path:
    if not self._is_free_mode_session_context():
        try:
            from ..campaign_manager import CAMPAIGN

            if CAMPAIGN.get_active_project_name():
                base_dir = CAMPAIGN.get_staging_dir("plate_stage")
                if base_dir is not None:
                    iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
                    return Path(base_dir) / f"Iteracja_{iter_num:03d}"
        except Exception:
            pass

    return Path(CONFIG.get_auto_annotations_dir("plate")) / "_manual_stage"


def _get_manual_plate_stage_images_dir(self) -> Path:
    return self._get_manual_plate_stage_dir() / "images"


def _is_manual_plate_stage_input(self, path_like) -> bool:
    if not path_like:
        return False
    try:
        return Path(path_like).resolve() == self._get_manual_plate_stage_images_dir().resolve()
    except Exception:
        return str(Path(path_like)) == str(self._get_manual_plate_stage_images_dir())


def _load_manual_plate_stage_manifest(self) -> dict:
    manifest_path = self._get_manual_plate_stage_dir() / "stage_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _refresh_manual_plate_stage_ui(self, *args, **kwargs):
    return z2_workflow_methods._refresh_manual_plate_stage_ui(self, *args, **kwargs)


def _refresh_manual_review_followup_ui(self, *args, **kwargs):
    return z2_workflow_methods._refresh_manual_review_followup_ui(self, *args, **kwargs)


def _refresh_preview_workspace_visibility(self, *args, **kwargs):
    return z2_workflow_methods._refresh_preview_workspace_visibility(self, *args, **kwargs)


def _set_preview_processing_overlay(
    self,
    active: bool,
    *,
    title: str = "",
    details: str = "",
    cancel_command=None,
    cancel_visible: bool = True,
    cancel_text: str = "Anuluj",
) -> None:
    controller = getattr(self, "preview_processing_overlay_controller", None)
    listbox = getattr(self, "preview_listbox", None)
    self._preview_processing_overlay_active = bool(active)
    try:
        if listbox is not None:
            listbox.configure(state=(tk.DISABLED if active else tk.NORMAL))
    except Exception:
        pass

    scope_progress_active = bool(getattr(self, "_plate_auto_scope_progress_modal_active", False))
    if active and not scope_progress_active:
        try:
            campaign_auto_without_modal = bool(
                not self._is_free_mode_session_context()
                and str(self._get_workflow_route() or "").strip().lower() == "auto"
                and not bool(self._manual_xml_template_enabled())
            )
        except Exception:
            campaign_auto_without_modal = False
        if campaign_auto_without_modal:
            return
    if controller is None:
        return

    if active and scope_progress_active:
        try:
            controller.hide()
        except Exception:
            pass
        updater = getattr(self, "_update_plate_auto_scope_progress_modal", None)
        if callable(updater):
            try:
                updater(pct=0.0, current=0, total=0, filename="", meta_text=str(details or title or "Przygotowuję autoanotację..."))
            except Exception:
                pass
        return

    if active:
        try:
            if hasattr(controller, "configure_cancel"):
                resolved_cancel = cancel_command
                if resolved_cancel is None and cancel_visible:
                    resolved_cancel = self._stop_annotation
                controller.configure_cancel(
                    command=resolved_cancel,
                    text=cancel_text,
                    visible=cancel_visible,
                    enabled=callable(resolved_cancel),
                )
            controller.show(title=str(title or ""), details=str(details or ""))
        except Exception:
            pass
    else:
        try:
            controller.hide()
            if hasattr(controller, "configure_cancel"):
                controller.configure_cancel(
                    command=self._stop_annotation,
                    text="Anuluj",
                    visible=True,
                    enabled=True,
                )
        except Exception:
            pass


def _update_preview_processing_overlay_progress(
    self,
    *,
    pct: float | None = None,
    current: int | None = None,
    total: int | None = None,
    filename: str = "",
    meta_text: str = "",
) -> None:
    updater = getattr(self, "_update_plate_auto_scope_progress_modal", None)
    if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)) and callable(updater):
        try:
            updater(
                pct=pct,
                current=current,
                total=total,
                filename=filename,
                meta_text=meta_text,
            )
        except Exception:
            pass
        return

    controller = getattr(self, "preview_processing_overlay_controller", None)
    if controller is None:
        return
    try:
        controller.update_progress(
            pct=pct,
            current=current,
            total=total,
            filename=filename,
            meta_text=meta_text,
        )
    except Exception:
        pass


def _refresh_export_followup_ui(self, *, compact_active_run: bool):
    title_label = getattr(self, "export_title_lbl", None)
    intro_label = getattr(self, "export_intro_lbl", None)
    run_row = getattr(getattr(self, "plate_dataset_run_title_lbl", None), "master", None)
    images_row = getattr(getattr(self, "plate_dataset_images_title_lbl", None), "master", None)
    output_row = getattr(getattr(self, "plate_dataset_out_title_lbl", None), "master", None)
    back_btn = getattr(self, "export_back_btn", None)
    campaign_context = not self._is_free_mode_session_context()
    focused_export = bool(campaign_context and self._manual_review_export_ready)
    manual_review_active = bool(
        getattr(self, "_manual_review_active", False)
        and self._get_preferred_annotation_run_dir(require_xml=True) is not None
    )
    thematic_prefixes = self._get_z2_thematic_title_prefixes()

    try:
        if back_btn is not None:
            back_btn.configure(
                text=(
                    "Wroc do korekty"
                    if (not campaign_context and manual_review_active)
                    else "Wroc do podsumowania wynikow autoanotacji"
                    if not campaign_context
                    else "Wroc do stage"
                )
            )
    except Exception:
        pass

    if compact_active_run:
        try:
            if campaign_context:
                title_text = self._format_z2_thematic_title(
                    "Ustaw split i wyeksportuj dataset",
                    thematic_prefixes.get("export"),
                )
            else:
                title_text = self._format_z2_thematic_title(
                    "Eksport: dataset YOLO albo anotacje XML",
                    thematic_prefixes.get("export"),
                )
            title_label.configure(text=title_text)
        except Exception:
            pass
        if not campaign_context:
            self.export_intro_var.set(
                "Masz dwa niezależne sposoby wyjścia z gotowego runu Z2:\n\n"
                "Dataset YOLO Pose - tworzy zbiór treningowy tablic, wymaga zatwierdzonych pozycji [OK] i korzysta ze splitu train/val/test.\n"
                "Anotacje XML - eksportuje zapisane anotacje do współpracy lub późniejszego importu w E1; nie używa splitu i nie wymaga bramki datasetu."
            )
            self._set_widget_packed(intro_label, True, anchor=tk.W, fill=tk.X, pady=(0, 10))
        else:
            self._set_widget_packed(intro_label, False)
        self._set_widget_packed(run_row, False)
        self._set_widget_packed(images_row, False)
        self._set_widget_packed(output_row, False)
        return

    try:
        if focused_export:
            base_title = self._format_z2_thematic_title(
                "Dataset do Z4",
                thematic_prefixes.get("export"),
            )
        elif campaign_context:
            base_title = self._format_z2_thematic_title(
                "Opcjonalny eksport datasetu do Z4",
                thematic_prefixes.get("export"),
            )
        else:
            base_title = self._format_z2_thematic_title(
                "Eksport: dataset YOLO albo anotacje XML",
                thematic_prefixes.get("export"),
            )
        title_label.configure(text=base_title)
    except Exception:
        pass
    if focused_export:
        self.export_intro_var.set(
            "Opcjonalnie przygotuj dataset tablic z gotowego runu Z2, aby przekazac go do treningu w Z4."
        )
    elif not campaign_context:
        self.export_intro_var.set(
            "Masz dwa niezależne sposoby wyjścia z gotowego runu Z2:\n\n"
            "Dataset YOLO Pose - tworzy zbiór treningowy tablic, wymaga zatwierdzonych pozycji [OK] i korzysta ze splitu train/val/test.\n"
            "Anotacje XML - eksportuje zapisane anotacje do współpracy lub późniejszego importu w E1; nie używa splitu i nie wymaga bramki datasetu."
        )
    self._set_widget_packed(intro_label, True, anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_widget_packed(run_row, True, fill=tk.X, pady=(0, 4))
    self._set_widget_packed(images_row, True, fill=tk.X, pady=(0, 4))
    self._set_widget_packed(output_row, True, fill=tk.X, pady=(0, 8))


def _switch_annotation_input_dir(self, *args, **kwargs):
    return z2_workflow_methods._switch_annotation_input_dir(self, *args, **kwargs)


def _use_manual_plate_stage_as_input(self):
    stage_images_dir = self._get_manual_plate_stage_images_dir()
    stage_images = get_image_files(stage_images_dir)
    if not stage_images:
        messagebox.showinfo(
            "Stage jest puste",
            "Stage nie zawiera jeszcze zadnych zdjec. Najpierw wyeksportuj dataset lub dodaj nowy zestaw zdjec do stage."
        )
        return

    if self._switch_annotation_input_dir(stage_images_dir):
        self._set_plate_export_status(
            f"Dla aktualnego stage nie ma jeszcze runu anotacji Z2. Kliknij {self._get_step2_start_action_reference()}, aby utworzyć nowy run anotacji dla tej puli obrazów.",
            "success",
        )


def _add_images_to_manual_plate_stage(self):
    stage_dir = self._get_manual_plate_stage_dir()
    stage_images_dir = stage_dir / "images"
    initialdir = self.input_dir_var.get().strip() or str(CONFIG.DIR_1_RAW)
    path = filedialog.askdirectory(initialdir=initialdir)
    if not path:
        return

    source_dir = Path(path)
    ok, msg, stats = self.dataset_creator.add_images_to_stage(source_dir, stage_dir)
    self._refresh_manual_plate_stage_ui()

    if not ok:
        messagebox.showerror("Błąd stage", msg)
        return

    stage_total = int(stats.get("stage_images_total", 0) or 0)
    added = int(stats.get("added", 0) or 0)
    updated = int(stats.get("updated", 0) or 0)
    messagebox.showinfo(
        "Stage zaktualizowane",
        (
            f"Stage zawiera teraz {stage_total} obrazow.\n"
            f"{stage_images_dir}\n\n"
            f"Dodane nowe pliki: {added}\n"
            f"Odswiezone istniejace wpisy: {updated}"
        )
    )

    if self._is_manual_plate_stage_input(self.input_dir_var.get()):
        self._set_post_annotation_hint(
            f"Dołożono nowe zdjęcia do aktywnego stage. Kliknij {self._get_step2_start_action_reference()}, aby utworzyć kolejny run anotacji Z2 dla tej rozszerzonej puli.",
            "success",
        )


def _plate_dataset_output_preview(self, run_dir: Path | None = None) -> str:
    run_name = run_dir.name if isinstance(run_dir, Path) else "run_xxx"
    return str(self._get_plate_dataset_base_dir() / f"Plates_Z2_{run_name}_[DATA_I_CZAS]")


def _reset_free_mode_branch_artifacts(self) -> None:
    self._free_mode_branch_artifacts = {
        "owned_run_dirs": [],
        "owned_temp_dirs": [],
    }


def _register_free_mode_branch_artifact(self, path_value, *, artifact_type: str) -> None:
    if not self._is_free_mode_session_context():
        return
    if artifact_type not in {"owned_run_dirs", "owned_temp_dirs"}:
        return

    candidate = self._path_value_to_path(path_value)
    if candidate is None:
        return
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate

    payload = dict(getattr(self, "_free_mode_branch_artifacts", {}) or {})
    values = list(payload.get(artifact_type) or [])
    key = str(resolved)
    if key not in values:
        values.append(key)
    payload[artifact_type] = values
    self._free_mode_branch_artifacts = payload


def _forget_manual_review_run(self, run_dir: Path | str | None) -> None:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=False)
    if safe_run_dir is None:
        return

    safe_run_dir_text = str(safe_run_dir)
    updated_entries = [
        entry
        for entry in list(self._manual_review_history_entries or [])
        if str(entry.get("run_dir") or "").strip() != safe_run_dir_text
    ]
    self._manual_review_history_entries = self._normalize_manual_review_history_entries(updated_entries)
    if str(self.manual_history_run_var.get() or "").strip() and (
        self._manual_review_history_label_map.get(str(self.manual_history_run_var.get() or "").strip()) == safe_run_dir_text
    ):
        self.manual_history_run_var.set("")
    self._refresh_manual_review_history_ui()


def _collect_free_mode_branch_cleanup_targets(self) -> dict[str, list[Path]]:
    payload = dict(getattr(self, "_free_mode_branch_artifacts", {}) or {})
    result: dict[str, list[Path]] = {
        "owned_run_dirs": [],
        "owned_temp_dirs": [],
    }

    allowed_roots = self._dedupe_paths(
        [
            self._get_annotation_output_base_dir(),
            *list(self._get_annotation_run_roots() or []),
        ]
    )

    for artifact_type in ("owned_run_dirs", "owned_temp_dirs"):
        seen: set[str] = set()
        for raw_value in list(payload.get(artifact_type) or []):
            candidate = self._path_value_to_path(raw_value)
            if candidate is None:
                continue
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            try:
                if not resolved.exists():
                    continue
            except Exception:
                continue
            if not self._path_is_within_any(resolved, allowed_roots):
                continue
            result[artifact_type].append(resolved)

    return result


def _cleanup_free_mode_branch_artifacts(self) -> dict[str, int]:
    cleanup_targets = self._collect_free_mode_branch_cleanup_targets()
    removed_runs = 0
    removed_temp_dirs = 0
    deleted_roots: set[str] = set()

    current_run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None), require_xml=False)
    active_owned_run = False
    for run_dir in cleanup_targets.get("owned_run_dirs", []):
        try:
            if current_run_dir is not None and self._paths_equivalent(run_dir, current_run_dir):
                active_owned_run = True
                break
        except Exception:
            continue

    if active_owned_run:
        self._clear_active_annotation_run_context(preserve_input_dir=True)

    for run_dir in cleanup_targets.get("owned_run_dirs", []):
        self._forget_manual_review_run(run_dir)
        try:
            shutil.rmtree(run_dir)
            removed_runs += 1
            deleted_roots.add(str(run_dir.parent))
        except Exception as e:
            logger.debug(f"Nie udało się usunąć runu gałęzi free mode ({run_dir}): {e}")

    for temp_dir in cleanup_targets.get("owned_temp_dirs", []):
        try:
            shutil.rmtree(temp_dir)
            removed_temp_dirs += 1
            deleted_roots.add(str(temp_dir.parent))
        except Exception as e:
            logger.debug(f"Nie udało się usunąć katalogu tymczasowego gałęzi free mode ({temp_dir}): {e}")

    for root_value in deleted_roots:
        try:
            PROJECT_CACHE.invalidate_annotation_run_dirs(Path(root_value))
        except Exception:
            pass

    self._reset_free_mode_branch_artifacts()
    return {
        "removed_runs": int(removed_runs),
        "removed_temp_dirs": int(removed_temp_dirs),
    }


def _confirm_and_return_to_free_mode_route_choice(self) -> bool:
    cleanup_targets = self._collect_free_mode_branch_cleanup_targets()
    run_dirs = list(cleanup_targets.get("owned_run_dirs") or [])
    temp_dirs = list(cleanup_targets.get("owned_temp_dirs") or [])
    details: list[str] = []
    if run_dirs:
        details.append(f"- runy Z2 do usunięcia: {len(run_dirs)}")
        for path in run_dirs[:3]:
            details.append(f"  • {self._format_workspace_relative_path(path) or str(path)}")
        if len(run_dirs) > 3:
            details.append(f"  • ... i jeszcze {len(run_dirs) - 3}")
    if temp_dirs:
        details.append(f"- katalogi tymczasowe do usunięcia: {len(temp_dirs)}")
        for path in temp_dirs[:3]:
            details.append(f"  • {self._format_workspace_relative_path(path) or str(path)}")
        if len(temp_dirs) > 3:
            details.append(f"  • ... i jeszcze {len(temp_dirs) - 3}")
    if not details:
        details.append("- bieżące ustawienia mini-flow zostaną wyczyszczone")

    proceed = messagebox.askyesno(
        "Porzucić bieżący mini-flow?",
        (
            "Powrót do wyboru toru porzuci bieżącą gałąź pracy Z2.\n\n"
            "Skutki tej operacji:\n"
            f"{chr(10).join(details)}\n\n"
            "Istniejące wcześniej runy nie znikną. Wrócisz do nich w Z2 przez tor ręczny: "
            "„Historia runów autoanotacji Z2” albo „Wskaż run autoanotacji Z2”.\n\n"
            "Czy na pewno wrócić do wyboru toru?"
        ),
        parent=self.frame.winfo_toplevel(),
    )
    if not proceed:
        return False

    if run_dirs or temp_dirs:
        cleanup_result = self._cleanup_free_mode_branch_artifacts()
        try:
            removed_runs = int(cleanup_result.get("removed_runs", 0) or 0)
            removed_temp_dirs = int(cleanup_result.get("removed_temp_dirs", 0) or 0)
            summary = (
                f"Porzucono gałąź mini-flow Z2. Usunięto runy: {removed_runs}, katalogi tymczasowe: {removed_temp_dirs}."
            )
            self._set_status_label_state(summary, "warning")
            self._set_post_annotation_hint(summary, "warning")
        except Exception:
            pass

    self._reset_free_mode_branch_artifacts()
    self._clear_free_mode_route_selection()
    return True


def _clear_active_annotation_run_context(self, *, preserve_input_dir: bool = True):
    input_dir_value = str(self.input_dir_var.get() or "").strip() if preserve_input_dir else ""
    try:
        preserved_input_dir = Path(input_dir_value) if input_dir_value else None
    except Exception:
        preserved_input_dir = None

    self.current_annotations = []
    self._preview_image_path_map = {}
    self._preview_approved_filenames = set()
    self._pending_preview_approved_filenames = set()
    self._campaign_pending_approved_filenames = set()
    self._clear_preview_editor_state(clear_dirty=True)
    try:
        self._invalidate_preview_list_frozen_order()
    except Exception:
        pass
    try:
        self._refresh_preview_list(preserve_selection=False, render_current=False)
    except Exception:
        pass
    self.last_staging_run_dir = None
    self._current_run_manual_template = False
    self._current_run_manual_vehicle_assist = False
    self._manual_review_active = False
    self._manual_review_from_auto = False
    self._manual_review_export_ready = False
    self._dataset_export_completed = False
    self._preview_session_restore_index = -1
    self._preview_session_restore_filename = ""
    self.current_input_dir = preserved_input_dir
    self.current_annotation_run_dir = None
    self.current_annotation_xml_path = None
    self._pending_source_image_map = {}
    self._annotation_source_input_dir = preserved_input_dir
    self._clear_annotation_run_scope_meta()

    try:
        self.plate_dataset_run_var.set("")
    except Exception:
        pass

    try:
        self.plate_dataset_out_var.set(self._plate_dataset_output_preview())
    except Exception:
        pass


def _reset_annotation_miniflow_state(
    self,
    *,
    clear_input_dir: bool = True,
    clear_plate_model: bool = True,
    clear_auto_modal_settings: bool = True,
) -> None:
    preserve_input_dir = not bool(clear_input_dir)
    self._clear_active_annotation_run_context(preserve_input_dir=preserve_input_dir)

    if clear_input_dir:
        try:
            self.input_dir_var.set("")
        except Exception:
            pass
        try:
            self.plate_dataset_images_var.set("")
        except Exception:
            pass

    if clear_plate_model:
        try:
            self.plate_custom_var.set("")
        except Exception:
            pass
        try:
            self._clear_plate_model_runtime_meta()
        except Exception:
            pass

    if clear_auto_modal_settings:
        self._reset_annotation_auto_modal_state()

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

    self._auto_route_settings_pending = False
    try:
        self.manual_history_run_var.set("")
    except Exception:
        pass
    try:
        self.manual_entry_mode_var.set("new")
    except Exception:
        pass
    try:
        self.manual_xml_template_var.set(True)
    except Exception:
        pass
    try:
        self.manual_vehicle_assist_var.set(False)
    except Exception:
        pass
    self._manual_review_active = False
    self._manual_review_from_auto = False
    self._manual_review_origin_route = ""
    self._manual_review_export_ready = False
    self._manual_template_ready_for_review = False
    self._dataset_export_completed = False
    self._last_completed_workflow_route = ""
