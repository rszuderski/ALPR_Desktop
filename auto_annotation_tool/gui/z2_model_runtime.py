#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 model selection and auto-annotation scope helpers extracted from tab_annotation.py."""

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

def _normalize_mode_value(self, mode: str | None = None) -> str:
    raw = str(mode if mode is not None else self.mode_var.get() or "").strip()
    if raw.startswith("B:"):
        return "B: Tylko tablice"
    return "C: Pojazdy + tablice"


def _mode_uses_vehicle(self, mode: str | None = None) -> bool:
    return self._normalize_mode_value(mode).startswith("C:")


def _mode_uses_plate(self, mode: str | None = None) -> bool:
    normalized = self._normalize_mode_value(mode)
    return normalized.startswith("B:") or normalized.startswith("C:")


def _vehicle_model_controls_enabled(self, mode: str | None = None) -> bool:
    if self._manual_xml_template_enabled():
        return True
    return self._mode_uses_vehicle(mode)


def _plate_model_controls_enabled(self, mode: str | None = None) -> bool:
    if self._manual_xml_template_enabled():
        return False
    return self._mode_uses_plate(mode)


def _refresh_detection_configuration_ui(self):
    try:
        self._legacy_detection_panel_visible = False
    except Exception:
        pass
    right_scroll_host = getattr(self, "right_scroll_host", None)
    detection_settings_lf = getattr(self, "detection_settings_lf", None)
    try:
        if detection_settings_lf is not None:
            detection_settings_lf.pack_forget()
    except Exception:
        pass
    try:
        if right_scroll_host is not None:
            right_scroll_host.pack_forget()
    except Exception:
        pass

    mode = self._normalize_mode_value()
    manual_enabled = self._manual_xml_template_enabled()
    vehicle_assist_enabled = self._manual_vehicle_assist_enabled()

    mode_combo = getattr(self, "mode_combo", None)
    if mode_combo is not None:
        try:
            mode_combo.configure(state="disabled" if manual_enabled else "readonly")
        except Exception:
            pass

    mode_hint_lbl = getattr(self, "mode_hint_lbl", None)
    if mode_hint_lbl is not None:
        if manual_enabled:
            if vehicle_assist_enabled:
                self.detection_mode_hint_var.set(
                    "Przy ręcznej anotacji model pojazdów służy wyłącznie jako pomoc przy utworzeniu XML z boxami pojazdów."
                )
            else:
                self.detection_mode_hint_var.set(
                    "Przy ręcznej anotacji model pojazdów pozostaje zasobem pomocniczym i nie trafia do eksportu tablic."
                )
            self._set_inline_label_state(mode_hint_lbl, tone="info", emphasis=False)
        else:
            self.detection_mode_hint_var.set(
                "Z2 korzysta z modeli ustawionych w aktualnym przepływie lub w modalu autoanotacji."
            )
            self._set_inline_label_state(mode_hint_lbl, tone="muted", emphasis=False)

    vehicle_combo_state = "readonly" if self._vehicle_model_controls_enabled(mode) else "disabled"
    for combo in (
        getattr(self, "vehicle_combo", None),
        getattr(self, "workflow_vehicle_combo", None),
    ):
        if combo is None:
            continue
        try:
            combo.config(state=vehicle_combo_state)
        except Exception:
            pass

    self._on_vehicle_model_change(refresh_workflow=False)
    self._set_plate_model_controls_state(self._plate_model_controls_enabled(mode))
    self._refresh_plate_model_runtime_info_ui()


def _set_progress_counters(self, successful: int, current: int, total: int):
    if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
        updater = getattr(self, "_update_plate_auto_scope_progress_modal", None)
        if callable(updater):
            try:
                pct = (float(current) / float(total) * 100.0) if int(total or 0) > 0 else None
                updater(
                    pct=pct,
                    current=int(current or 0),
                    total=int(total or 0),
                    successful=int(successful or 0),
                )
            except Exception:
                pass
        return

    try:
        self.progress_counts_var.set(
            f"OK / teraz / razem: {int(successful)}/{int(current)}/{int(total)}"
        )
    except Exception:
        pass


def _on_mode_change(self, event=None):
    mode = self._normalize_mode_value()
    if mode != self.mode_var.get():
        self.mode_var.set(mode)
    self._refresh_detection_configuration_ui()


def _update_model_lists(self):
    if YOLO_AVAILABLE:
        v_keys = sorted(list(AVAILABLE_DETECT_MODELS.keys()))
        vehicle_values = v_keys + ["Custom"]
        for combo in (
            getattr(self, "vehicle_combo", None),
            getattr(self, "workflow_vehicle_combo", None),
        ):
            if combo is None:
                continue
            try:
                combo["values"] = vehicle_values
            except Exception:
                pass
        if not self.vehicle_model_var.get() and v_keys:
            preferred_vehicle = "yolo11s" if "yolo11s" in v_keys else v_keys[0]
            self.vehicle_model_var.set(preferred_vehicle)

    if hasattr(self, "character_combo"):
        self._refresh_character_model_choices()


def _on_vehicle_model_change(self, event=None, *, refresh_workflow: bool = True):
    for combo, custom_row in (
        (getattr(self, "vehicle_combo", None), getattr(self, "veh_custom_row", None)),
        (
            getattr(self, "workflow_vehicle_combo", None),
            getattr(self, "workflow_vehicle_custom_row", None),
        ),
    ):
        if combo is None or custom_row is None:
            continue

        try:
            combo_state = str(combo.cget("state"))
        except Exception:
            combo_state = "readonly"

        should_show = self.vehicle_model_var.get() == "Custom" and combo_state != "disabled"
        if should_show:
            custom_row.pack(fill=tk.X, pady=(5, 0))
        else:
            custom_row.pack_forget()

    if refresh_workflow and self._is_free_mode_session_context():
        current_step = self._coerce_workflow_step()
        if current_step in {"manual_entry", "manual_input", "auto_vehicle_model", "manual_vehicle_model"}:
            self._refresh_left_panel_route_copy()
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()


def _set_plate_model_controls_state(self, enabled: bool):
    state = "normal" if enabled else "disabled"

    for widget in (
        getattr(self, "plate_path_entry", None),
        getattr(self, "plate_browse_btn", None),
        getattr(self, "workflow_plate_path_entry", None),
        getattr(self, "workflow_plate_browse_btn", None),
    ):
        if widget is None:
            continue
        try:
            widget.configure(state=state)
        except Exception:
            pass


def _get_bound_character_model_path(self) -> str:
    try:
        from ..campaign_manager import CAMPAIGN
        model_path = CAMPAIGN.get_global_model("char")
        if model_path and Path(model_path).exists():
            return str(Path(model_path))
    except Exception:
        pass

    try:
        tab_char = getattr(self.app, "tabs", {}).get("characters")
        if tab_char is not None:
            model_path = (tab_char.yolo_model_path_var.get() or "").strip()
            if model_path and model_path != "Brak modelu znakow w projekcie" and Path(model_path).exists():
                return str(Path(model_path))
    except Exception:
        pass

    return ""


def _collect_character_model_candidates(self, active_path: str = ""):
    candidates = []
    for chars_dir in CONFIG.get_model_search_dirs("char"):
        if not chars_dir.exists():
            continue
        candidates.extend(
            sorted(
                chars_dir.rglob("*.pt"),
                key=lambda p: (str(p.parent).lower(), p.name.lower())
            )
        )

    if active_path:
        active_model = Path(active_path)
        if active_model.exists() and active_model not in candidates:
            candidates.append(active_model)

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique


def _refresh_character_model_choices(self):
    if not hasattr(self, "character_combo"):
        return

    current_path = (self._get_selected_character_model_path() or "").strip()
    if not current_path:
        current_path = self._get_bound_character_model_path()

    options = {"Brak / OCR": ""}
    for model_path in self._collect_character_model_candidates(current_path):
        label = model_path.name
        if label in options:
            label = f"{model_path.parent.name}/{model_path.name}"
        if label in options:
            label = str(model_path)
        options[label] = str(model_path)

    options["Custom"] = "__custom__"
    self._character_model_options = options
    self.character_combo["values"] = list(options.keys())

    if current_path:
        for label, model_path in options.items():
            if model_path == current_path:
                self.character_model_var.set(label)
                break
        else:
            self.character_model_var.set("Custom")
            self.character_custom_var.set(current_path)
    elif not self.character_model_var.get() or self.character_model_var.get() not in options:
        self.character_model_var.set("Brak / OCR")

    self._on_character_model_change(propagate=False)


def _get_selected_character_model_path(self) -> str:
    selected = (self.character_model_var.get() or "").strip()
    if selected == "Custom":
        return (self.character_custom_var.get() or "").strip()
    return self._character_model_options.get(selected, "")


def _apply_character_model_selection(self):
    if not hasattr(self, "character_combo"):
        return

    selected_path = (self._get_selected_character_model_path() or "").strip()
    effective_path = selected_path if selected_path and Path(selected_path).exists() else ""

    try:
        from ..campaign_manager import CAMPAIGN
        if CAMPAIGN.get_active_project_name():
            CAMPAIGN.set_global_model("char", effective_path)
            try:
                campaign_tab = getattr(self.app, "tabs", {}).get("campaign")
                if campaign_tab is not None:
                    campaign_tab._refresh_dashboard()
            except Exception:
                pass
    except Exception:
        pass

    try:
        tab_char = getattr(self.app, "tabs", {}).get("characters")
        if tab_char is not None:
            tab_char.yolo_model_path_var.set(effective_path)

            if effective_path:
                try:
                    version, size = tab_char._infer_yolo_arch_from_model_path(effective_path)
                    if version in {"8", "11", "26"}:
                        tab_char.yolo_model_version_var.set(version)
                    if size in {"n", "s", "m", "l", "x"}:
                        tab_char.yolo_model_size_var.set(size)
                except Exception:
                    pass

            try:
                tab_char._sync_yolo_model_binding()
            except Exception:
                pass

            try:
                tab_char._update_yolo_visibility()
            except Exception:
                pass

            try:
                tab_char._force_save_all()
            except Exception:
                pass
    except Exception:
        pass


def _on_character_model_change(self, event=None, propagate=True):
    if not hasattr(self, "char_custom_row"):
        return

    if self.character_model_var.get() == "Custom" and str(self.character_combo.cget("state")) != "disabled":
        self.char_custom_row.pack(fill=tk.X, pady=(5,0))
    else:
        self.char_custom_row.pack_forget()
        if self.character_model_var.get() != "Custom":
            self.character_custom_var.set("")

    if propagate:
        self._apply_character_model_selection()


def _select_vehicle_custom(self):
    initial_dir = CONFIG.get_trained_models_dir("vehicle")
    if not initial_dir.exists():
        initial_dir = CONFIG.DIR_6_MODELS
    p = filedialog.askopenfilename(initialdir=str(Path(initial_dir).absolute()), filetypes=[("YOLO Model", "*.pt")])
    if p:
        self.vehicle_custom_var.set(p)
        if self._is_free_mode_session_context():
            current_step = self._coerce_workflow_step()
            if current_step == "auto_vehicle_model":
                self._go_to_next_workflow_step()
            elif current_step in {"manual_entry", "manual_input", "manual_vehicle_model"}:
                self._refresh_left_panel_route_copy()
                self._refresh_detection_configuration_ui()
                self._refresh_step2_action_states()
                self._refresh_free_mode_workflow_ui()


def _select_plate_custom(self):
    initial_dir = CONFIG.get_trained_models_dir("plate")
    if not initial_dir.exists():
        initial_dir = CONFIG.DIR_6_MODELS
    p = filedialog.askopenfilename(initialdir=str(Path(initial_dir).absolute()), filetypes=[("YOLO Model", "*.pt")])
    if p:
        self.plate_custom_var.set(p)
        try:
            source = ""
            scope = ""
            if not self._is_free_mode_session_context():
                source = "external"
                scope = "run"
            self._remember_plate_model_runtime_meta(model_path=p, source=source, scope=scope)
        except Exception:
            pass
        if self._is_free_mode_session_context() and self._get_workflow_step() == "auto_plate_model":
            self._go_to_next_workflow_step()
        else:
            self._advance_campaign_auto_step_after_plate_model_selection()


def _advance_campaign_auto_step_after_plate_model_selection(self) -> None:
    if self._is_free_mode_session_context():
        return
    try:
        if self._get_workflow_route() != "auto":
            return
    except Exception:
        return

    current_step = self._coerce_workflow_step()
    if current_step != "auto_plate_model":
        return

    try:
        input_dir_text = str(self.input_dir_var.get() or "").strip()
        input_ready = bool(input_dir_text and Path(input_dir_text).exists())
    except Exception:
        input_ready = False

    try:
        self._set_workflow_step("auto_start" if input_ready else "auto_input")
    except Exception:
        pass


def _get_campaign_project_plate_model_path(self) -> Path | None:
    if self._is_free_mode_session_context():
        return None
    try:
        from ..campaign_manager import CAMPAIGN
        if not CAMPAIGN.get_active_project_name():
            return None
        raw_value = str(CAMPAIGN.get_global_model("plate") or "").strip()
    except Exception:
        raw_value = ""
    if not raw_value:
        return None
    try:
        candidate = Path(raw_value)
    except Exception:
        return None
    return candidate if candidate.exists() else None


def _get_model_identity_caption(model_path: Path | None) -> str:
    return z2_get_model_identity_caption(model_path)


def _model_quality_float(value) -> float | None:
    return z2_model_quality_float(value)


def _model_quality_value(cls, mapping: dict | None, keys: list[str] | tuple[str, ...]) -> float | None:
    return z2_model_quality_value(mapping, keys)


def _paths_refer_to_same_model(left: str | Path | None, right: str | Path | None) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    try:
        left_key = os.path.normcase(str(Path(left_text).resolve()))
        right_key = os.path.normcase(str(Path(right_text).resolve()))
        return left_key == right_key
    except Exception:
        return os.path.normcase(left_text) == os.path.normcase(right_text)


def _path_is_inside_dir(path_value: str | Path | None, dir_value: str | Path | None) -> bool:
    path_text = str(path_value or "").strip()
    dir_text = str(dir_value or "").strip()
    if not path_text or not dir_text:
        return False
    try:
        path_key = os.path.normcase(str(Path(path_text).resolve()))
        dir_key = os.path.normcase(str(Path(dir_text).resolve()))
        return path_key == dir_key or path_key.startswith(dir_key + os.sep)
    except Exception:
        return False


def _iter_plate_training_history_files(self) -> list[Path]:
    candidates: list[Path] = []
    try:
        candidates.append(Path(CONFIG.get_training_runs_dir("plate")) / "training_history.json")
    except Exception:
        pass
    try:
        candidates.append(Path(CONFIG.DIR_5_RUNS_PLATES) / "training_history.json")
    except Exception:
        pass
    try:
        candidates.append(Path(CONFIG.DEFAULT_TRAINING_DIR) / "training_history.json")
    except Exception:
        pass
    try:
        root = Path(CONFIG.DEFAULT_TRAINING_DIR)
        if root.exists():
            candidates.extend(root.rglob("training_history.json"))
    except Exception:
        pass
    try:
        from ..campaign_manager import CAMPAIGN

        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
        if active_project:
            project_root = CAMPAIGN.get_project_root(active_project)
            if project_root is not None:
                candidates.append(Path(project_root) / "5_training_runs" / "training_history.json")
    except Exception:
        pass
    try:
        projects_root = Path(CONFIG.DIR_9_PROJECTS)
        if projects_root.exists():
            candidates.extend(projects_root.glob("*/5_training_runs/training_history.json"))
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(str(candidate.resolve()))
        except Exception:
            key = os.path.normcase(str(candidate))
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _training_history_run_looks_like_plate(run: dict | None) -> bool:
    if not isinstance(run, dict):
        return False

    text = " ".join(
        str(run.get(key) or "")
        for key in ("name", "dataset_path", "base_model", "output_dir")
    ).lower()
    if any(token in text for token in ("plate", "plates", "tablic", "pose", "-pose", "_pose")):
        return True
    if any(token in text for token in ("char", "chars", "character", "ocr", "znak", "znaki", "vehicle", "pojazd")):
        return False
    return True


def _load_training_runs_from_history_file(history_file: Path) -> list[dict]:
    try:
        with history_file.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception:
        return []

    raw_runs = data.get("runs", {}) if isinstance(data, dict) else {}
    if isinstance(raw_runs, dict):
        runs: list[dict] = []
        for run_id, run_data in raw_runs.items():
            if not isinstance(run_data, dict):
                continue
            cloned = dict(run_data)
            cloned.setdefault("id", str(run_id or "").strip())
            runs.append(cloned)
        return runs
    if isinstance(raw_runs, list):
        return [dict(item) for item in raw_runs if isinstance(item, dict)]
    return []


def _find_plate_training_run_for_model(self, model_path: Path) -> dict | None:
    safe_path = Path(model_path)
    model_name = str(safe_path.name or "").strip().lower()
    try:
        model_key = os.path.normcase(str(safe_path.resolve()))
    except Exception:
        model_key = os.path.normcase(str(safe_path))

    for history_file in self._iter_plate_training_history_files():
        for run in self._load_training_runs_from_history_file(history_file):
            if not self._training_history_run_looks_like_plate(run):
                continue
            candidate_paths = [
                str(run.get("best_weights") or "").strip(),
                str(run.get("last_weights") or "").strip(),
            ]
            output_dir = str(run.get("output_dir") or "").strip()
            if output_dir:
                candidate_paths.extend(
                    [
                        str(Path(output_dir) / "train" / "weights" / "best.pt"),
                        str(Path(output_dir) / "train" / "weights" / "last.pt"),
                    ]
                )

            for candidate in candidate_paths:
                if candidate and self._paths_refer_to_same_model(candidate, safe_path):
                    run["_history_file"] = str(history_file)
                    return run

            if output_dir and self._path_is_inside_dir(model_key, output_dir):
                run["_history_file"] = str(history_file)
                return run

            run_id = str(run.get("id") or "").strip().lower()
            if run_id and run_id in model_name:
                run["_history_file"] = str(history_file)
                return run

    return None


def _extract_plate_model_metrics_from_rows(self, *args, **kwargs):
    return z2_workflow_methods._extract_plate_model_metrics_from_rows(self, *args, **kwargs)


def _read_results_csv_metrics(self, csv_path: Path, *, source: str = "results.csv") -> dict:
    if not csv_path.exists():
        return {}
    rows: list[dict] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                rows.append({str(k or "").strip(): str(v or "").strip() for k, v in row.items()})
    except Exception as e:
        logger.debug(f"Nie udało się odczytać metryk modelu z results.csv: {e}")
        return {}
    metrics = self._extract_plate_model_metrics_from_rows(rows, source=source)
    if metrics:
        metrics["results_csv"] = str(csv_path)
    return metrics


def _read_results_csv_metrics_for_model(self, model_path: Path) -> dict:
    safe_path = Path(model_path)
    candidates: list[Path] = []
    try:
        if safe_path.parent.name.lower() == "weights":
            candidates.append(safe_path.parent.parent / "results.csv")
        candidates.extend(
            [
                safe_path.parent / "results.csv",
                safe_path.parent.parent / "results.csv",
            ]
        )
    except Exception:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(str(candidate.resolve()))
        except Exception:
            key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        metrics = self._read_results_csv_metrics(candidate, source="results.csv obok wag")
        if metrics:
            return metrics
    return {}


def _read_exported_model_metadata_metrics(self, model_path: Path) -> dict:
    safe_path = Path(model_path)
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        try:
            key = os.path.normcase(str(candidate.resolve()))
        except Exception:
            key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    seen: set[str] = set()
    add_candidate(safe_path.with_suffix(".json"))
    add_candidate(safe_path.with_name(f"{safe_path.stem}_metadata.json"))
    try:
        stem_key = str(safe_path.stem or "").strip().lower()
        for candidate in safe_path.parent.glob("*.json"):
            candidate_name = str(candidate.stem or "").strip().lower()
            if stem_key and stem_key in candidate_name:
                add_candidate(candidate)
    except Exception:
        pass

    for metadata_path in candidates:
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except Exception as e:
            logger.debug(f"Nie udało się odczytać metadanych modelu {metadata_path}: {e}")
            continue
        if not isinstance(data, dict):
            continue

        metrics_root = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        best_row = metrics_root.get("best_row") if isinstance(metrics_root.get("best_row"), dict) else {}
        latest = metrics_root.get("latest") if isinstance(metrics_root.get("latest"), dict) else {}
        training = data.get("training") if isinstance(data.get("training"), dict) else {}
        run_snapshot = data.get("run_snapshot") if isinstance(data.get("run_snapshot"), dict) else {}
        source_root = data.get("source") if isinstance(data.get("source"), dict) else {}
        source = "metadata eksportu modelu"

        def first_metric(*values):
            for value in values:
                if value is not None:
                    return value
            return None

        metrics = {
            "map50": first_metric(
                self._model_quality_float(metrics_root.get("best_map50")),
                self._model_quality_float(run_snapshot.get("best_map50")),
                self._model_quality_value(best_row, ("map50", "box_map50", "metrics/mAP50(B)", "metrics/mAP50", "pose_map50", "metrics/mAP50(P)")),
                self._model_quality_value(latest, ("map50", "box_map50", "metrics/mAP50(B)", "metrics/mAP50", "pose_map50", "metrics/mAP50(P)")),
            ),
            "map50_95": first_metric(
                self._model_quality_float(metrics_root.get("best_map50_95")),
                self._model_quality_float(run_snapshot.get("best_map50_95")),
                self._model_quality_value(best_row, ("map50_95", "box_map50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95", "pose_map50_95", "metrics/mAP50-95(P)")),
                self._model_quality_value(latest, ("map50_95", "box_map50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95", "pose_map50_95", "metrics/mAP50-95(P)")),
            ),
            "precision": first_metric(
                self._model_quality_value(best_row, ("precision", "box_precision", "metrics/precision(B)", "pose_precision", "metrics/precision(P)")),
                self._model_quality_value(latest, ("precision", "box_precision", "metrics/precision(B)", "pose_precision", "metrics/precision(P)")),
            ),
            "recall": first_metric(
                self._model_quality_value(best_row, ("recall", "box_recall", "metrics/recall(B)", "pose_recall", "metrics/recall(P)")),
                self._model_quality_value(latest, ("recall", "box_recall", "metrics/recall(B)", "pose_recall", "metrics/recall(P)")),
            ),
            "loss": first_metric(
                self._model_quality_value(best_row, ("loss", "val/loss", "train/loss", "val/box_loss", "train/box_loss", "val/pose_loss", "train/pose_loss")),
                self._model_quality_value(latest, ("loss", "val/loss", "train/loss", "val/box_loss", "train/box_loss", "val/pose_loss", "train/pose_loss")),
            ),
            "epoch": first_metric(
                self._model_quality_float(metrics_root.get("best_epoch")),
                self._model_quality_value(best_row, ("epoch", "Epoch")),
                self._model_quality_float(training.get("current_epoch")),
                self._model_quality_float(run_snapshot.get("current_epoch")),
            ),
            "source": source,
            "metadata_json": str(metadata_path),
            "run_id": str(training.get("run_id") or run_snapshot.get("id") or "").strip(),
            "run_name": str(training.get("run_name") or run_snapshot.get("name") or "").strip(),
            "status": str(training.get("status") or run_snapshot.get("status") or "").strip(),
            "dataset_path": str(training.get("dataset_path") or run_snapshot.get("dataset_path") or "").strip(),
            "output_dir": str(source_root.get("run_output_dir") or "").strip(),
        }
        return {key: value for key, value in metrics.items() if value not in (None, "")}

    return {}


def _build_plate_model_quality_metrics(self, model_path: Path) -> dict:
    safe_path = Path(model_path)
    metrics: dict = self._read_exported_model_metadata_metrics(safe_path)
    run = self._find_plate_training_run_for_model(safe_path)

    if isinstance(run, dict):
        history_rows = list(run.get("metrics_history") or [])
        history_metrics = self._extract_plate_model_metrics_from_rows(history_rows, source="historia treningu")

        if not history_metrics:
            history_metrics = {
                "map50": self._model_quality_float(run.get("best_map50")),
                "map50_95": self._model_quality_float(run.get("best_map50_95")),
                "source": "historia treningu",
            }
            history_metrics = {key: value for key, value in history_metrics.items() if value not in (None, "")}

        output_dir = str(run.get("output_dir") or "").strip()
        if output_dir:
            csv_metrics = self._read_results_csv_metrics(
                Path(output_dir) / "train" / "results.csv",
                source="results.csv runu",
            )
            if csv_metrics:
                history_metrics.update({key: value for key, value in csv_metrics.items() if value not in (None, "")})

        history_metrics.update(
            {
                "run_id": str(run.get("id") or "").strip(),
                "run_name": str(run.get("name") or "").strip(),
                "status": str(run.get("status") or "").strip(),
                "dataset_path": str(run.get("dataset_path") or "").strip(),
                "output_dir": str(run.get("output_dir") or "").strip(),
            }
        )
        for key, value in history_metrics.items():
            if key not in metrics or metrics.get(key) in (None, ""):
                metrics[key] = value

    if not metrics:
        metrics = self._read_results_csv_metrics_for_model(safe_path)

    return {key: value for key, value in metrics.items() if value not in (None, "")}


def _get_plate_model_quality_metrics(self, model_path: Path | None) -> dict:
    if model_path is None or not Path(model_path).exists():
        return {}
    safe_path = Path(model_path)
    try:
        stat = safe_path.stat()
        cache_key = (
            os.path.normcase(str(safe_path.resolve())),
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )
    except Exception:
        cache_key = (os.path.normcase(str(safe_path)), 0, 0)

    cache = getattr(self, "_plate_model_quality_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._plate_model_quality_cache = cache
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)

    metrics = self._build_plate_model_quality_metrics(safe_path)
    cache[cache_key] = dict(metrics)
    return metrics


def _format_model_quality_number(value) -> str:
    return z2_format_model_quality_number(value)


def _format_model_quality_epoch(value) -> str:
    return z2_format_model_quality_epoch(value)


def _describe_auto_model_quality(value) -> str:
    return z2_describe_auto_model_quality(value)


def _auto_model_metric_tone(value, profile: str = "strict") -> str:
    return z2_auto_model_metric_tone(value, profile)


def _shorten_model_quality_text(value, limit: int = 72) -> str:
    return z2_shorten_model_quality_text(value, limit)


def _build_auto_annotation_model_quality_rows(self, *args, **kwargs):
    return z2_workflow_methods._build_auto_annotation_model_quality_rows(self, *args, **kwargs)


def _format_auto_annotation_model_quality_summary(self, model_path: Path | str | None) -> tuple[str, str]:
    rows, tone = self._build_auto_annotation_model_quality_rows(model_path)
    lines = [f"{label}: {value}" for label, value, _row_tone in rows if str(value or "").strip()]
    return "\n".join(lines), tone


def _render_auto_annotation_model_quality_table(
    self,
    host,
    rows: list[tuple[str, str, str]],
    tone: str,
    *,
    bg: str | None = None,
    fg: str | None = None,
    muted: str | None = None,
    border: str | None = None,
    success: str | None = None,
    warning: str | None = None,
    wraplength: int = 390,
    label_width: int = 15,
    model_path: Path | str | None = None,
) -> None:
    return z2_render_auto_annotation_model_quality_table(
        self,
        host,
        rows,
        tone,
        bg=bg,
        fg=fg,
        muted=muted,
        border=border,
        success=success,
        warning=warning,
        wraplength=wraplength,
        label_width=label_width,
        model_path=model_path,
    )


def _confirm_campaign_plate_model_identity_choice(self, *args, **kwargs):
    return z2_workflow_methods._confirm_campaign_plate_model_identity_choice(self, *args, **kwargs)


def _normalize_plate_model_source(value: str | None) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"project", "external"} else ""


def _normalize_plate_model_scope(value: str | None) -> str:
    scope = str(value or "").strip().lower()
    return scope if scope in {"project", "run"} else ""


def _remember_plate_model_runtime_meta(
    self,
    *,
    model_path: Path | str | None = None,
    identity: str = "",
    source: str = "",
    scope: str = "",
) -> None:
    path_text = str(model_path or self.plate_custom_var.get() or "").strip()
    identity_text = str(identity or "").strip()
    if not identity_text and path_text:
        try:
            identity_text = self._get_model_identity_caption(Path(path_text))
        except Exception:
            identity_text = ""

    self._plate_model_runtime_meta = {
        "path": path_text,
        "identity": identity_text,
        "source": self._normalize_plate_model_source(source),
        "scope": self._normalize_plate_model_scope(scope),
    }


def _clear_plate_model_runtime_meta(self) -> None:
    self._plate_model_runtime_meta = {
        "path": "",
        "identity": "",
        "source": "",
        "scope": "",
    }


def _collect_plate_model_manifest_fields(self) -> dict:
    meta = self._get_effective_plate_model_runtime_meta()
    return {
        "plate_model_path": str(meta.get("path") or "").strip(),
        "plate_model_identity": str(meta.get("identity") or "").strip(),
        "plate_model_source": str(meta.get("source") or "").strip(),
        "plate_model_scope": str(meta.get("scope") or "").strip(),
    }


def _get_effective_plate_model_runtime_meta(self) -> dict:
    current_path = str(self.plate_custom_var.get() or "").strip()
    meta = dict(getattr(self, "_plate_model_runtime_meta", {}) or {})
    manifest_meta = {}

    safe_run_dir = self._resolve_safe_annotation_run_dir(getattr(self, "current_annotation_run_dir", None))
    if safe_run_dir is not None:
        manifest = self._load_annotation_run_manifest(safe_run_dir)
        if isinstance(manifest, dict):
            manifest_meta = {
                "path": str(manifest.get("plate_model_path") or "").strip(),
                "identity": str(manifest.get("plate_model_identity") or "").strip(),
                "source": self._normalize_plate_model_source(manifest.get("plate_model_source")),
                "scope": self._normalize_plate_model_scope(manifest.get("plate_model_scope")),
            }

    if current_path:
        meta_path = str(meta.get("path") or "").strip()
        manifest_path = str(manifest_meta.get("path") or "").strip()
        if manifest_path and manifest_path == current_path:
            meta.update({k: v for k, v in manifest_meta.items() if v})
        elif meta_path != current_path:
            meta["path"] = current_path
            meta["identity"] = ""
            meta["source"] = ""
            meta["scope"] = ""
    elif manifest_meta.get("path"):
        current_path = str(manifest_meta.get("path") or "").strip()
        meta.update({k: v for k, v in manifest_meta.items() if v})

    if not current_path:
        return {"path": "", "identity": "", "source": "", "scope": ""}

    if not self._normalize_plate_model_source(meta.get("source")):
        project_model = self._get_campaign_project_plate_model_path()
        try:
            if project_model is not None and Path(current_path).resolve() == project_model.resolve():
                meta["source"] = "project"
        except Exception:
            pass
    if not self._normalize_plate_model_scope(meta.get("scope")) and meta.get("source") == "project":
        meta["scope"] = "project"

    meta["path"] = current_path
    meta["source"] = self._normalize_plate_model_source(meta.get("source"))
    meta["scope"] = self._normalize_plate_model_scope(meta.get("scope"))
    self._plate_model_runtime_meta = dict(meta)
    return meta


def _refresh_plate_model_runtime_info_ui(self) -> None:
    box = getattr(self, "workflow_plate_model_info_box", None)
    label = getattr(self, "workflow_plate_model_info_lbl", None)
    if box is None or label is None:
        return

    route = self._get_workflow_route()
    campaign_context = not self._is_free_mode_session_context()
    meta = self._get_effective_plate_model_runtime_meta()
    path_text = str(meta.get("path") or "").strip()
    identity = str(meta.get("identity") or "").strip()
    source = self._normalize_plate_model_source(meta.get("source"))
    scope = self._normalize_plate_model_scope(meta.get("scope"))

    visible = bool(route == "auto" and (campaign_context or path_text))
    if not visible:
        self.workflow_plate_model_info_var.set("")
        self._set_widget_packed(box, False)
        return

    file_label = ""
    try:
        file_label = Path(path_text).name
    except Exception:
        file_label = path_text

    lines: list[str] = []
    tone = "info"
    if campaign_context:
        project_model_path = self._get_campaign_project_plate_model_path()
        project_identity = ""
        try:
            if (
                project_model_path is not None
                and path_text
                and Path(path_text).resolve() == project_model_path.resolve()
            ):
                project_identity = identity
        except Exception:
            project_identity = identity if source == "project" else ""
        if project_model_path is not None:
            project_line = f"Model projektu tablic: {project_model_path.name}"
            if project_identity:
                project_line += f" | {project_identity}"
        else:
            project_line = "Model projektu tablic: brak"

        if path_text:
            runtime_line = f"Model autoanotacji dla tego runu: {file_label}"
            if identity and file_label.lower() not in identity.lower():
                runtime_line += f" | {identity}"
            elif identity:
                runtime_line = f"Model autoanotacji dla tego runu: {identity}"

            if scope == "project" or source == "project":
                relation_line = (
                    "Ten model jest zapisany w projekcie i pozostaje domyślnym modelem tablic także poza tym runem."
                )
                tone = "success"
            else:
                relation_line = (
                    "To lokalna podmiana tylko dla bieżącego runu Z2. Sama zmiana tutaj nie podmienia modelu projektu."
                )
                tone = "warning"
        else:
            runtime_line = "Model autoanotacji dla tego runu: brak"
            relation_line = (
                "Aby uruchomić autoanotację, użyj modelu projektu albo wskaż model tylko dla tego runu Z2."
            )
            tone = "warning"

        lines.extend([project_line, runtime_line, relation_line])
    else:
        source_label = {
            "project": "projekt",
            "external": "zewnętrzny",
        }.get(source, "")
        scope_label = {
            "project": "zapisano w projekcie",
            "run": "tylko ten run",
        }.get(scope, "")

        lines = [identity or file_label]
        if source_label:
            lines.append(f"Źródło: {source_label}")
        if scope_label:
            lines.append(f"Zakres: {scope_label}")
        if file_label and (not identity or file_label.lower() not in identity.lower()):
            lines.append(f"Plik: {file_label}")
        tone = "success" if scope == "project" or source == "project" else "info"

    self.workflow_plate_model_info_var.set("\n".join(lines))
    self._set_widget_packed(
        box,
        True,
        anchor=tk.W,
        fill=tk.X,
        pady=(6, 0),
    )
    self._set_plate_model_info_box_state(tone=tone)


def _restore_plate_model_selection_from_active_run(self) -> bool:
    current_value = str(self.plate_custom_var.get() or "").strip()
    if current_value and Path(current_value).exists():
        return True

    meta = dict(self._get_effective_plate_model_runtime_meta() or {})
    candidate_path = str(meta.get("path") or "").strip()
    if not candidate_path:
        return False

    try:
        candidate = Path(candidate_path)
    except Exception:
        return False
    if not candidate.exists():
        return False

    self.plate_custom_var.set(str(candidate))
    try:
        self._remember_plate_model_runtime_meta(
            model_path=candidate,
            identity=str(meta.get("identity") or "").strip(),
            source=str(meta.get("source") or "").strip(),
            scope=str(meta.get("scope") or "").strip(),
        )
    except Exception:
        pass
    return True


def _get_auto_model_picker_initial_dir(
    self,
    target: str,
    *,
    picker_scope: str = "free",
    preferred_path: Path | None = None,
) -> Path:
    scope = str(picker_scope or "").strip().lower()
    if scope == "project" and not self._is_free_mode_session_context():
        try:
            from ..campaign_manager import CAMPAIGN
            project_models_dir = CAMPAIGN.get_dir("models")
            if project_models_dir is not None and Path(project_models_dir).exists():
                return Path(project_models_dir)
        except Exception:
            pass

    if scope == "free":
        try:
            initial_dir = CONFIG.get_trained_models_dir(target)
            if initial_dir.exists():
                return initial_dir
        except Exception:
            pass
        try:
            if CONFIG.DIR_6_MODELS.exists():
                return Path(CONFIG.DIR_6_MODELS)
        except Exception:
            pass

    if preferred_path is not None:
        try:
            candidate = preferred_path.parent if preferred_path.is_file() else preferred_path
            if candidate.exists():
                return candidate
        except Exception:
            pass

    try:
        initial_dir = CONFIG.get_trained_models_dir(target)
        if initial_dir.exists():
            return initial_dir
    except Exception:
        pass
    return Path(CONFIG.DIR_6_MODELS)


def _get_campaign_plate_model_picker_initial_dir(
    self,
    preferred_path: Path | None = None,
    *,
    picker_scope: str = "project",
) -> Path:
    scoped_dir = self._get_auto_model_picker_initial_dir(
        "plate",
        picker_scope=picker_scope,
        preferred_path=preferred_path,
    )
    if scoped_dir.exists():
        return scoped_dir
    if preferred_path is not None:
        try:
            candidate = preferred_path.parent if preferred_path.is_file() else preferred_path
            if candidate.exists():
                return candidate
        except Exception:
            pass
    try:
        from ..campaign_manager import CAMPAIGN
        project_models_dir = CAMPAIGN.get_dir("models")
        if project_models_dir is not None and Path(project_models_dir).exists():
            return Path(project_models_dir)
    except Exception:
        pass
    return Path(CONFIG.DIR_6_MODELS)


def _pick_campaign_plate_auto_model_file(
    self,
    preferred_path: Path | None = None,
    *,
    picker_scope: str = "project",
) -> Path | None:
    initial_dir = self._get_campaign_plate_model_picker_initial_dir(
        preferred_path=preferred_path,
        picker_scope=picker_scope,
    )
    chosen = filedialog.askopenfilename(
        initialdir=str(initial_dir.absolute()),
        filetypes=[("YOLO Model", "*.pt")],
        title="Wskaż model tablic dla autoanotacji Z2",
    )
    if not chosen:
        return None
    try:
        candidate = Path(chosen)
    except Exception:
        return None
    return candidate if candidate.exists() else None


def _adopt_campaign_plate_model_into_project(self, model_path: Path) -> Path:
    safe_path = Path(model_path)
    from ..campaign_manager import CAMPAIGN

    project_models_dir = CAMPAIGN.get_dir("models")
    if project_models_dir is None:
        raise RuntimeError("Nie udało się ustalić katalogu modeli aktywnego projektu.")

    project_models_dir = Path(project_models_dir)
    project_models_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_resolved = safe_path.resolve()
    except Exception:
        source_resolved = safe_path

    try:
        models_resolved = project_models_dir.resolve()
    except Exception:
        models_resolved = project_models_dir

    try:
        if source_resolved.parent == models_resolved:
            return source_resolved
    except Exception:
        pass

    target_path = project_models_dir / safe_path.name
    try:
        target_resolved = target_path.resolve()
    except Exception:
        target_resolved = target_path

    same_target = False
    try:
        same_target = source_resolved == target_resolved
    except Exception:
        same_target = False
    if same_target:
        return target_path

    if target_path.exists():
        stem = str(safe_path.stem or "model").strip() or "model"
        suffix = str(safe_path.suffix or ".pt")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = project_models_dir / f"{stem}_adopted_{timestamp}{suffix}"
        counter = 2
        while candidate.exists():
            candidate = project_models_dir / f"{stem}_adopted_{timestamp}_{counter}{suffix}"
            counter += 1
        target_path = candidate

    shutil.copy2(str(safe_path), str(target_path))
    return target_path


def _apply_campaign_plate_auto_model_choice(self, *args, **kwargs):
    return z2_workflow_methods._apply_campaign_plate_auto_model_choice(self, *args, **kwargs)


def _clear_annotation_run_scope_meta(self) -> None:
    self._annotation_run_scope_meta = {
        "mode": "",
        "label": "",
        "count": 0,
        "filenames": [],
    }


def _remember_annotation_run_scope_meta(
    self,
    *,
    mode: str,
    label: str,
    count: int,
    filenames: list[str] | None = None,
) -> None:
    normalized_filenames = sorted(
        {
            str(name or "").strip().lower()
            for name in list(filenames or [])
            if str(name or "").strip()
        }
    )
    self._annotation_run_scope_meta = {
        "mode": str(mode or "").strip().lower(),
        "label": str(label or "").strip(),
        "count": max(0, int(count or 0)),
        "filenames": normalized_filenames,
    }


def _collect_annotation_run_scope_manifest_fields(self) -> dict:
    meta = dict(getattr(self, "_annotation_run_scope_meta", {}) or {})
    return {
        "input_scope_mode": str(meta.get("mode") or "").strip(),
        "input_scope_label": str(meta.get("label") or "").strip(),
        "input_scope_count": max(0, int(meta.get("count") or 0)),
        "input_scope_filenames": list(meta.get("filenames") or []),
    }


def _dedupe_image_paths_by_name(image_paths) -> list[Path]:
    result: list[Path] = []
    seen_names: set[str] = set()
    for raw_path in list(image_paths or []):
        try:
            safe_path = Path(raw_path)
        except Exception:
            continue
        name_key = str(getattr(safe_path, "name", "") or "").strip().lower()
        if not name_key or name_key in seen_names:
            continue
        seen_names.add(name_key)
        result.append(safe_path)
    return result


def _get_preview_group_actual_indices(self) -> list[int]:
    if not self.current_annotations:
        return []
    sort_mode = self._normalize_preview_list_sort_mode()
    if sort_mode == "Nazwa pliku A-Z":
        return []

    current_actual = self.current_preview_index
    current_display = self._get_preview_display_index(current_actual)
    display_indices = list(getattr(self, "_preview_list_display_indices", []) or [])
    if current_actual is None or current_display is None or not display_indices:
        return []

    try:
        current_ann = self.current_annotations[int(current_actual)]
    except Exception:
        return []

    frozen_buckets = dict(getattr(self, "_preview_list_frozen_bucket_snapshot", {}) or {})
    current_key = str(getattr(current_ann, "filename", "") or "").strip().lower()
    current_bucket = frozen_buckets.get(current_key) or self._preview_annotation_sort_bucket_for_mode(
        current_ann,
        sort_mode,
    )
    if not current_bucket:
        return []

    def bucket_for_actual_index(actual_index: int) -> str:
        try:
            ann = self.current_annotations[int(actual_index)]
        except Exception:
            return ""
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        return str(
            frozen_buckets.get(filename_key)
            or self._preview_annotation_sort_bucket_for_mode(ann, sort_mode)
            or ""
        ).strip().lower()

    group_indices = [int(current_actual)]
    left = int(current_display) - 1
    while left >= 0:
        candidate_actual = self._get_preview_actual_index_from_display(left)
        if candidate_actual is None or bucket_for_actual_index(candidate_actual) != current_bucket:
            break
        group_indices.insert(0, int(candidate_actual))
        left -= 1

    right = int(current_display) + 1
    while right < len(display_indices):
        candidate_actual = self._get_preview_actual_index_from_display(right)
        if candidate_actual is None or bucket_for_actual_index(candidate_actual) != current_bucket:
            break
        group_indices.append(int(candidate_actual))
        right += 1

    return group_indices


def _resolve_preview_scope_paths(
    self,
    actual_indices: list[int],
    *,
    candidate_path_map: dict[str, Path] | None = None,
    protected_filenames: set[str] | None = None,
) -> list[Path]:
    resolved_paths: list[Path] = []
    seen_names: set[str] = set()
    path_map = {
        str(name or "").strip().lower(): Path(path)
        for name, path in dict(candidate_path_map or {}).items()
        if str(name or "").strip()
    }
    protected_names = {
        str(name or "").strip().lower()
        for name in set(protected_filenames or set())
        if str(name or "").strip()
    }
    for actual_index in list(actual_indices or []):
        try:
            ann = self.current_annotations[int(actual_index)]
        except Exception:
            continue
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key or filename_key in seen_names:
            continue
        if filename_key in protected_names:
            continue
        resolved_path = path_map.get(filename_key)
        if resolved_path is None:
            resolved_path = self._resolve_preview_image_path(ann)
        if resolved_path is None:
            continue
        try:
            if not Path(resolved_path).exists():
                continue
        except Exception:
            continue
        seen_names.add(filename_key)
        resolved_paths.append(Path(resolved_path))
    return resolved_paths


def _count_unique_preview_scope_indices(
    annotations: list,
    actual_indices: list[int],
) -> int:
    seen_names: set[str] = set()
    count = 0
    for actual_index in list(actual_indices or []):
        try:
            ann = annotations[int(actual_index)]
        except Exception:
            continue
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key or filename_key in seen_names:
            continue
        seen_names.add(filename_key)
        count += 1
    return count


def _count_protected_preview_scope_indices(
    annotations: list,
    actual_indices: list[int],
    protected_filenames: set[str],
) -> int:
    protected_names = {
        str(name or "").strip().lower()
        for name in set(protected_filenames or set())
        if str(name or "").strip()
    }
    if not protected_names:
        return 0

    seen_names: set[str] = set()
    skipped_count = 0
    for actual_index in list(actual_indices or []):
        try:
            ann = annotations[int(actual_index)]
        except Exception:
            continue
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key or filename_key in seen_names:
            continue
        seen_names.add(filename_key)
        if filename_key in protected_names:
            skipped_count += 1
    return skipped_count


def _get_preview_auto_scope_protected_filenames(self) -> set[str]:
    protected: set[str] = set()
    annotations = list(getattr(self, "current_annotations", []) or [])
    if not annotations:
        return protected

    try:
        approved_names = set(self._get_preview_approved_filenames_base() or set())
    except Exception:
        approved_names = set()
    try:
        campaign_manual_names = set(self._get_campaign_manual_touched_filenames() or set())
    except Exception:
        campaign_manual_names = set()

    for ann in annotations:
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key:
            continue
        is_approved = False
        try:
            is_approved = bool(
                self._preview_annotation_is_explicitly_approved(
                    ann,
                    approved_names=approved_names,
                )
            )
        except Exception:
            is_approved = filename_key in approved_names
        is_manual = bool(filename_key in campaign_manual_names)
        try:
            is_manual = bool(is_manual or self._preview_annotation_has_manual_touch(ann))
        except Exception:
            pass
        if is_approved or is_manual:
            protected.add(filename_key)

    return protected


def _collect_plate_auto_scope_candidates(
    self,
    *,
    candidate_image_paths: list[Path] | None = None,
    protect_existing: bool = True,
) -> dict:
    protected_filenames = (
        self._get_preview_auto_scope_protected_filenames()
        if bool(protect_existing)
        else set()
    )
    annotations = list(getattr(self, "current_annotations", []) or [])
    raw_candidate_paths = self._dedupe_image_paths_by_name(candidate_image_paths or [])
    candidate_paths = [
        path
        for path in raw_candidate_paths
        if str(getattr(path, "name", "") or "").strip().lower() not in protected_filenames
    ]
    candidate_path_map = {
        str(path.name or "").strip().lower(): path
        for path in candidate_paths
        if str(path.name or "").strip()
    }
    selected_indices = self._get_selected_preview_actual_indices()
    selected_paths = self._resolve_preview_scope_paths(
        selected_indices,
        candidate_path_map=candidate_path_map,
        protected_filenames=protected_filenames,
    )
    selected_row_count = self._get_preview_listbox_selected_row_count()
    selected_total_count = self._count_unique_preview_scope_indices(
        annotations,
        selected_indices,
    )
    selected_protected_skip_count = self._count_protected_preview_scope_indices(
        annotations,
        selected_indices,
        protected_filenames,
    )
    group_indices = self._get_preview_group_actual_indices()
    group_paths = self._resolve_preview_scope_paths(
        group_indices,
        candidate_path_map=candidate_path_map,
        protected_filenames=protected_filenames,
    )
    group_bucket = ""
    if group_indices:
        try:
            first_ann = self.current_annotations[int(group_indices[0])]
            group_bucket = self._preview_annotation_auto_scope_bucket(first_ann)
        except Exception:
            group_bucket = ""
    return {
        "raw_all_paths": raw_candidate_paths,
        "all_paths": candidate_paths,
        "protected_skip_count": max(0, len(raw_candidate_paths) - len(candidate_paths)),
        "selected_paths": selected_paths,
        "selected_count": len(selected_paths),
        "selected_row_count": selected_row_count,
        "selected_total_count": selected_total_count,
        "selected_duplicate_skip_count": max(0, int(selected_row_count) - int(selected_total_count)),
        "selected_protected_skip_count": selected_protected_skip_count,
        "selected_unresolved_skip_count": max(
            0,
            int(selected_total_count) - int(selected_protected_skip_count) - int(len(selected_paths)),
        ),
        "group_paths": group_paths,
        "group_count": len(group_paths),
        "group_protected_skip_count": self._count_protected_preview_scope_indices(
            annotations,
            group_indices,
            protected_filenames,
        ),
        "group_bucket": group_bucket,
    }


def _collect_plate_auto_scope_bucket_protected_counts(
    self,
    *,
    candidate_image_paths: list[Path] | None = None,
    protect_existing: bool = True,
) -> dict[str, int]:
    if not bool(protect_existing):
        return {"manual": 0, "auto": 0, "problem": 0}
    protected_filenames = self._get_preview_auto_scope_protected_filenames()
    candidate_paths = self._dedupe_image_paths_by_name(candidate_image_paths or [])
    candidate_names = {
        str(getattr(path, "name", "") or "").strip().lower()
        for path in candidate_paths
        if str(getattr(path, "name", "") or "").strip()
    }
    if not protected_filenames or not candidate_names:
        return {"manual": 0, "auto": 0, "problem": 0}

    counts: dict[str, int] = {"manual": 0, "auto": 0, "problem": 0}
    seen_names: set[str] = set()
    for actual_index in list(getattr(self, "_preview_list_display_indices", []) or []):
        try:
            ann = self.current_annotations[int(actual_index)]
        except Exception:
            continue
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key or filename_key in seen_names:
            continue
        if filename_key not in candidate_names or filename_key not in protected_filenames:
            continue
        bucket = self._preview_annotation_auto_scope_bucket(ann)
        if bucket not in counts:
            continue
        seen_names.add(filename_key)
        counts[bucket] += 1
    return counts


def _collect_plate_auto_scope_bucket_paths(
    self,
    *,
    candidate_image_paths: list[Path] | None = None,
    protect_existing: bool = True,
) -> dict[str, list[Path]]:
    protected_filenames = (
        self._get_preview_auto_scope_protected_filenames()
        if bool(protect_existing)
        else set()
    )
    candidate_paths = self._dedupe_image_paths_by_name(candidate_image_paths or [])
    candidate_paths = [
        path
        for path in candidate_paths
        if str(getattr(path, "name", "") or "").strip().lower() not in protected_filenames
    ]
    if not candidate_paths:
        return {"manual": [], "auto": [], "problem": []}

    candidate_path_map = {
        str(path.name or "").strip().lower(): path
        for path in candidate_paths
        if str(path.name or "").strip()
    }
    ordered_paths: dict[str, list[Path]] = {"manual": [], "auto": [], "problem": []}
    seen_names: set[str] = set()

    for actual_index in list(getattr(self, "_preview_list_display_indices", []) or []):
        try:
            ann = self.current_annotations[int(actual_index)]
        except Exception:
            continue
        filename_key = str(getattr(ann, "filename", "") or "").strip().lower()
        if not filename_key or filename_key in seen_names:
            continue
        if filename_key in protected_filenames:
            continue
        resolved_path = candidate_path_map.get(filename_key)
        if resolved_path is None:
            continue
        bucket = self._preview_annotation_auto_scope_bucket(ann)
        if bucket not in ordered_paths:
            continue
        ordered_paths[bucket].append(resolved_path)
        seen_names.add(filename_key)

    return ordered_paths


def _can_offer_preview_scope_selection_for_auto_run(self) -> bool:
    if not self._is_free_mode_session_context():
        return True

    free_mode_screen = self._coerce_free_mode_screen()
    if free_mode_screen == "workflow":
        return True
    if free_mode_screen not in {"auto_summary", "manual_review", "export"}:
        return False

    return bool(getattr(self, "current_annotations", None))


def _set_plate_auto_scope_modal_ui_lock(self, active: bool) -> None:
    normalized_active = bool(active)
    previous_state = bool(getattr(self, "_plate_auto_scope_modal_open", False))
    self._plate_auto_scope_modal_open = normalized_active
    if previous_state == normalized_active:
        return

    if normalized_active:
        locked_states: dict = {}
        for widget_name in (
            "workflow_back_btn",
            "workflow_next_btn",
            "start_btn",
            "stop_btn",
            "approve_btn",
            "export_plate_dataset_btn",
            "export_plate_annotations_btn",
            "export_back_btn",
            "workflow_input_browse_btn",
            "workflow_plate_browse_btn",
            "workflow_vehicle_custom_browse_btn",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                if hasattr(widget, "state") and callable(widget.state):
                    was_disabled = bool(widget.instate(["disabled"]))
                    locked_states[widget] = ("ttk", was_disabled)
                    widget.state(["disabled"])
                else:
                    current_state = str(widget.cget("state") or "").strip()
                    if current_state:
                        locked_states[widget] = ("tk", current_state)
                        widget.configure(state=tk.DISABLED)
            except Exception:
                pass
        self._plate_auto_scope_locked_widget_states = locked_states
    else:
        for widget, payload in dict(getattr(self, "_plate_auto_scope_locked_widget_states", {}) or {}).items():
            kind, previous_value = payload
            try:
                if kind == "ttk":
                    if bool(previous_value):
                        widget.state(["disabled"])
                    else:
                        widget.state(["!disabled"])
                elif kind == "tk":
                    widget.configure(state=previous_value)
            except Exception:
                pass
        self._plate_auto_scope_locked_widget_states = {}

    try:
        self._update_preview_toolbar_state(refresh_summary=False)
    except Exception:
        pass


def _focus_plate_auto_scope_modal(self) -> None:
    if bool(getattr(self, "_plate_auto_scope_selection_mode_active", False)):
        return
    dialog = getattr(self, "_plate_auto_scope_active_dialog", None)
    if dialog is None:
        return
    try:
        if dialog.winfo_exists():
            dialog.lift()
            dialog.focus_force()
    except Exception:
        pass


def _is_plate_auto_scope_modal_blocking_actions(self) -> bool:
    active = bool(getattr(self, "_plate_auto_scope_modal_open", False))
    if active:
        self._focus_plate_auto_scope_modal()
    return active


def _fit_borderless_dialog(self, dialog, *, parent=None, min_width: int = 700, min_height: int = 420) -> None:
    fitter = getattr(self.app, "_fit_dialog_to_content", None)
    if callable(fitter):
        try:
            fitter(dialog, parent=parent or self.frame, min_width=min_width, min_height=min_height)
            return
        except Exception:
            pass
    try:
        dialog.update_idletasks()
    except Exception:
        pass
        try:
            widget.bind("<ButtonPress-1>", _start_drag, add="+")
            widget.bind("<B1-Motion>", _drag, add="+")
        except Exception:
            pass


def _set_plate_auto_scope_selection_mode(self, active: bool, *, dialog=None) -> None:
    normalized_active = bool(active)
    previous_state = bool(getattr(self, "_plate_auto_scope_selection_mode_active", False))
    self._plate_auto_scope_selection_mode_active = normalized_active
    if previous_state == normalized_active and dialog is None:
        return
    if active:
        try:
            if dialog is not None:
                dialog.grab_release()
        except Exception:
            pass
        try:
            self.preview_listbox.focus_set()
        except Exception:
            pass
    else:
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.grab_set()
        except Exception:
            pass
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.focus_set()
        except Exception:
            pass
    if previous_state != normalized_active:
        try:
            self._update_preview_toolbar_state(refresh_summary=False)
        except Exception:
            pass


def _get_preview_listbox_selected_row_count(self) -> int:
    return len(self._get_preview_listbox_selection_signature())


def _get_preview_listbox_selection_signature(self) -> tuple[int, ...]:
    listbox = getattr(self, "preview_listbox", None)
    if listbox is None:
        return ()
    try:
        return tuple(int(index) for index in list(listbox.curselection() or ()))
    except Exception:
        return ()


def _refresh_plate_auto_scope_modal_selection_state(self) -> None:
    selection_refresh_callback = getattr(self, "_plate_auto_scope_modal_selection_refresh_callback", None)
    if not callable(selection_refresh_callback):
        return

    try:
        self._plate_auto_scope_modal_last_selection_signature = self._get_preview_listbox_selection_signature()
    except Exception:
        pass

    frame = getattr(self, "frame", None)
    if frame is None:
        try:
            selection_refresh_callback()
        except Exception:
            pass
        return

    def _refresh_selection_again() -> None:
        self._plate_auto_scope_modal_selection_refresh_after_id = None
        callback = getattr(self, "_plate_auto_scope_modal_selection_refresh_callback", None)
        if callable(callback):
            try:
                callback()
            except Exception:
                pass

    try:
        previous_after_id = getattr(self, "_plate_auto_scope_modal_selection_refresh_after_id", None)
        if previous_after_id:
            frame.after_cancel(previous_after_id)
    except Exception:
        pass
    try:
        self._plate_auto_scope_modal_selection_refresh_after_id = frame.after(35, _refresh_selection_again)
    except Exception:
        pass


def _prompt_plate_auto_scope_choice(
    self,
    *,
    candidate_image_paths: list[Path] | None = None,
) -> dict | None:
    return prompt_plate_auto_scope_choice(
        self,
        candidate_image_paths=candidate_image_paths,
    )


def _build_annotation_input_subset_dir(
    self,
    image_paths: list[Path],
    *,
    scope_suffix: str,
) -> tuple[Path | None, dict[str, Path]]:
    scoped_paths = self._dedupe_image_paths_by_name(image_paths)
    if not scoped_paths:
        return None, {}

    subset_root = None
    if not self._is_free_mode_session_context():
        try:
            from ..campaign_manager import CAMPAIGN
            project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
            staging_root = CAMPAIGN.get_staging_dir("auto_ann")
            if project_name and staging_root is not None:
                iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
                subset_root = Path(staging_root) / "_campaign_input_scope" / f"Iteracja_{iter_num:03d}"
        except Exception:
            subset_root = None
    if subset_root is None:
        subset_root = self._get_annotation_output_base_dir() / "_input_scope"

    try:
        subset_dir = self._allocate_annotation_run_dir(subset_root, suffix=f"scope_{scope_suffix}")
    except Exception as e:
        logger.debug(f"Nie udało się przygotować katalogu zakresu autoanotacji: {e}")
        return None, {}

    try:
        if self._is_free_mode_session_context():
            self._register_free_mode_branch_artifact(subset_dir, artifact_type="owned_temp_dirs")
    except Exception:
        pass

    source_map: dict[str, Path] = {}
    prepared_count = 0
    linked_count = 0
    copied_count = 0
    for image_path in scoped_paths:
        try:
            target_path = subset_dir / image_path.name
            try:
                os.link(image_path, target_path)
                linked_count += 1
            except Exception:
                shutil.copy2(image_path, target_path)
                copied_count += 1
            source_map[str(image_path.name)] = image_path
            prepared_count += 1
        except Exception as e:
            logger.debug(f"Nie udało się przygotować obrazu do zakresu autoanotacji ({image_path}): {e}")

    if prepared_count <= 0:
        try:
            if subset_dir.exists():
                shutil.rmtree(subset_dir)
        except Exception:
            pass
        return None, {}
    if copied_count > 0:
        logger.debug(
            "[AnnotationTab] Zakres autoanotacji przygotowany z fallbackiem copy: "
            f"linked={int(linked_count)} copied={int(copied_count)}"
        )
    return subset_dir, source_map


def _prompt_campaign_plate_auto_model_choice(self, *args, **kwargs):
    return z2_workflow_methods._prompt_campaign_plate_auto_model_choice(self, *args, **kwargs)
