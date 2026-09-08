#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 session, snapshot and campaign source helpers extracted from tab_annotation.py."""

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
    campaign_gate_id_for_edge,
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

def _annotation_session_defaults(self) -> dict:
    return {
        "input_dir": str(Path(CONFIG.DIR_1_RAW).absolute()),
        "output_dir": str(Path(CONFIG.get_auto_annotations_dir("plate")).absolute()),
        "mode": "C: Pojazdy + tablice",
        "vehicle_model": "",
        "vehicle_custom": "",
        "plate_custom": "",
        "character_model": "Brak / OCR",
        "character_custom": "",
        "device": "auto",
        "conf": float(CONFIG.DEFAULT_CONFIDENCE),
        "plate_dataset_run": "",
        "plate_dataset_images": "",
        "plate_train_pct": 80.0,
        "plate_val_pct": 10.0,
        "manual_xml_template": True,
        "manual_vehicle_assist": False,
        "workflow_route": "",
        "manual_entry_mode": "new",
        "auto_vehicle_choice": "skip",
        "workflow_step": "",
        "free_mode_screen": "route_choice",
        "manual_review_active": False,
        "manual_review_from_auto": False,
        "manual_review_origin_route": "",
        "last_completed_workflow_route": "",
        "manual_review_export_ready": False,
        "manual_review_history": [],
        "last_preview_run_dir": "",
        "last_preview_index": -1,
        "last_preview_filename": "",
        "plate_model_identity": "",
        "plate_model_source": "",
        "plate_model_scope": "",
    }


def _annotation_auto_modal_defaults(self) -> dict:
    defaults = self._annotation_session_defaults()
    return {
        "mode": self._normalize_mode_value(defaults.get("mode")),
        "vehicle_model": str(defaults.get("vehicle_model") or "").strip(),
        "vehicle_custom": str(defaults.get("vehicle_custom") or "").strip(),
        "plate_custom": str(defaults.get("plate_custom") or "").strip(),
        "conf": float(defaults.get("conf", CONFIG.DEFAULT_CONFIDENCE)),
        "auto_vehicle_choice": self._normalize_auto_vehicle_choice(
            defaults.get("auto_vehicle_choice")
        ),
        "plate_model_identity": str(defaults.get("plate_model_identity") or "").strip(),
        "plate_model_source": str(defaults.get("plate_model_source") or "").strip(),
        "plate_model_scope": str(defaults.get("plate_model_scope") or "").strip(),
    }


def _apply_annotation_auto_modal_defaults_to_snapshot(
    self,
    session_state: dict | None = None,
) -> dict:
    state = dict(session_state or {})
    for key, value in self._annotation_auto_modal_defaults().items():
        state[key] = value
    return state


def _strip_persisted_annotation_miniflow_snapshot(
    self,
    session_state: dict | None = None,
) -> dict:
    state = dict(session_state or {})
    defaults = self._annotation_session_defaults()
    state.update(
        {
            "input_dir": defaults["input_dir"],
            "plate_dataset_run": "",
            "plate_dataset_images": "",
            "workflow_route": "",
            "manual_entry_mode": defaults["manual_entry_mode"],
            "workflow_step": "",
            "free_mode_screen": "route_choice",
            "manual_xml_template": True,
            "manual_vehicle_assist": False,
            "manual_review_active": False,
            "manual_review_from_auto": False,
            "manual_review_origin_route": "",
            "last_completed_workflow_route": "",
            "manual_review_export_ready": False,
            "last_preview_run_dir": "",
            "last_preview_index": -1,
            "last_preview_filename": "",
        }
    )
    return state


def _reset_annotation_auto_modal_state(self) -> None:
    defaults = self._annotation_auto_modal_defaults()

    try:
        self.mode_var.set(str(defaults["mode"]))
    except Exception:
        pass
    try:
        self.vehicle_model_var.set(str(defaults["vehicle_model"]))
    except Exception:
        pass
    try:
        self.vehicle_custom_var.set(str(defaults["vehicle_custom"]))
    except Exception:
        pass
    try:
        self.plate_custom_var.set(str(defaults["plate_custom"]))
    except Exception:
        pass
    try:
        self.conf_var.set(float(defaults["conf"]))
    except Exception:
        pass

    try:
        self._set_auto_vehicle_choice_state(defaults["auto_vehicle_choice"], campaign_context=False)
    except Exception:
        pass
    try:
        self._clear_plate_model_runtime_meta()
    except Exception:
        pass
    try:
        self._refresh_confidence_value_labels()
    except Exception:
        pass
    try:
        self._on_vehicle_model_change(refresh_workflow=False)
    except Exception:
        pass


def _normalize_workflow_route_value(route: str | None = None) -> str:
    value = str(route or "").strip().lower()
    if value in {"auto", "manual"}:
        return value
    return ""


def _normalize_manual_entry_mode(mode: str | None = None) -> str:
    value = str(mode or "").strip().lower()
    if value in {"new", "continue", "import"}:
        return value
    return "continue"


def _normalize_auto_vehicle_choice(choice: str | None = None) -> str:
    value = str(choice or "").strip().lower()
    if value in {"use", "skip"}:
        return value
    return "skip"


def _normalize_workflow_step_value(step: str | None = None) -> str:
    value = str(step or "").strip().lower()
    allowed = {
        "auto_plate_model",
        "auto_conf",
        "auto_vehicle_choice",
        "auto_vehicle_model",
        "auto_input",
        "auto_start",
        "manual_entry",
        "manual_history",
        "manual_conf",
        "manual_vehicle_model",
        "manual_input",
        "manual_start",
    }
    return value if value in allowed else ""


def _normalize_free_mode_screen_value(screen: str | None = None) -> str:
    value = str(screen or "").strip().lower()
    allowed = {
        "route_choice",
        "workflow",
        "auto_summary",
        "manual_review",
        "export",
    }
    return value if value in allowed else ""


def _is_free_mode_session_context(self) -> bool:
    try:
        from ..campaign_manager import CAMPAIGN
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        active_project = ""

    # Aktywny projekt ma pierwszeństwo przed flagą trybu swobodnego.
    # W przeciwnym razie stary stan app.campaign_free_mode potrafi wpuścić
    # buildery (F) do ekranów kampanii (C).
    campaign_context_project = str(getattr(self, "_campaign_context_project_name", "") or "").strip()
    try:
        app_free_mode = bool(getattr(self.app, "campaign_free_mode", False))
    except Exception:
        app_free_mode = False
    if active_project or (campaign_context_project and not app_free_mode):
        try:
            if bool(getattr(self.app, "campaign_free_mode", False)):
                self.app.campaign_free_mode = False
        except Exception:
            pass
        return False

    return True


def _is_campaign_step2_context(self) -> bool:
    return not self._is_free_mode_session_context()


def _is_annotation_tab_selected(self) -> bool:
    try:
        notebook = getattr(self.app, "notebook", None)
        if notebook is None:
            return False
        current_tab = str(notebook.select() or "").strip()
        return bool(current_tab and current_tab == str(self.frame))
    except Exception:
        return False


def reset_preview_selection_to_first_visible_on_tab_entry(
    self,
    *,
    reason: str = "tab-entry",
    attempts_left: int = 8,
) -> None:
    pending = getattr(self, "_preview_tab_entry_reset_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_tab_entry_reset_after_id = None
    self._preview_tab_entry_reset_token = int(
        getattr(self, "_preview_tab_entry_reset_token", 0) or 0
    ) + 1
    token = int(self._preview_tab_entry_reset_token)

    def _run(remaining: int) -> None:
        self._preview_tab_entry_reset_after_id = None
        if token != int(getattr(self, "_preview_tab_entry_reset_token", 0) or 0):
            return
        if not self._is_annotation_tab_selected():
            return
        if getattr(self, "is_processing", False):
            if remaining > 0:
                self._preview_tab_entry_reset_after_id = self.frame.after(
                    180,
                    lambda: _run(remaining - 1),
                )
            return

        annotations = list(getattr(self, "current_annotations", []) or [])
        if not annotations:
            if remaining > 0:
                self._preview_tab_entry_reset_after_id = self.frame.after(
                    180,
                    lambda: _run(remaining - 1),
                )
            return

        if bool(getattr(self, "_preview_list_population_active", False)):
            if remaining > 0:
                self._preview_tab_entry_reset_after_id = self.frame.after(
                    180,
                    lambda: _run(remaining - 1),
                )
            return

        try:
            display_indices = list(getattr(self, "_preview_list_display_indices", []) or [])
            list_size = int(getattr(self, "preview_listbox", None).size() or 0)
        except Exception:
            display_indices = []
            list_size = 0

        if not display_indices or list_size <= 0:
            try:
                if self._should_use_async_preview_list_population(len(annotations)):
                    self._populate_preview_list_async(
                        preserve_selection=False,
                        render_current=True,
                        batch_size=500,
                    )
                else:
                    self._refresh_preview_list(
                        preserve_selection=False,
                        render_current=True,
                    )
            except Exception as e:
                logger.debug(f"Nie udało się ustawić pierwszej pozycji Z2 po wejściu ({reason}): {e}")
            return

        try:
            first_actual_index = int(display_indices[0])
        except Exception:
            first_actual_index = 0
        if first_actual_index < 0 or first_actual_index >= len(annotations):
            first_actual_index = 0

        try:
            self._select_preview_index(first_actual_index, reset_view=True)
            self._append_z2_trace(
                "tab-entry-first-selection",
                f"reason={str(reason or '').strip()} actual={first_actual_index}",
            )
        except Exception as e:
            logger.debug(f"Nie udało się przeskoczyć na pierwszą pozycję Z2 po wejściu ({reason}): {e}")

    try:
        self._preview_tab_entry_reset_after_id = self.frame.after(
            120,
            lambda: _run(max(0, int(attempts_left or 0))),
        )
    except Exception:
        _run(max(0, int(attempts_left or 0)))


def _get_campaign_annotation_state_path(self, project_name: str | None = None) -> Path | None:
    try:
        from ..campaign_manager import CAMPAIGN

        state_dir = CAMPAIGN.get_project_state_dir(project_name)
    except Exception:
        state_dir = None

    if state_dir is None:
        return None

    return Path(state_dir) / "annotation_ui_state.json"


def _path_value_to_text(path_value) -> str:
    if path_value is None:
        return ""

    try:
        return str(path_value).strip()
    except RecursionError:
        pass
    except Exception:
        pass

    if isinstance(path_value, Path):
        visited: set[int] = set()

        def _flatten(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, bytes):
                try:
                    return os.fsdecode(value).strip()
                except Exception:
                    return ""
            if isinstance(value, Path):
                obj_id = id(value)
                if obj_id in visited:
                    return ""
                visited.add(obj_id)
                raw_parts = getattr(value, "_raw_paths", None) or ()
                if raw_parts:
                    chunks = [_flatten(item) for item in raw_parts]
                    chunks = [chunk for chunk in chunks if chunk]
                    if not chunks:
                        return ""
                    try:
                        return os.path.join(*chunks).strip()
                    except Exception:
                        return " ".join(chunks).strip()
            try:
                return os.fsdecode(os.fspath(value)).strip()
            except Exception:
                try:
                    return str(value).strip()
                except Exception:
                    return ""

        return _flatten(path_value)

    try:
        return os.fsdecode(os.fspath(path_value)).strip()
    except Exception:
        return ""


def _path_value_to_path(cls, path_value) -> Path | None:
    raw_value = cls._path_value_to_text(path_value)
    if not raw_value:
        return None
    try:
        return Path(raw_value)
    except Exception:
        return None


def _paths_equivalent(cls, left, right) -> bool:
    left_path = cls._path_value_to_path(left)
    right_path = cls._path_value_to_path(right)
    if left_path is None or right_path is None:
        return False

    try:
        return left_path.resolve() == right_path.resolve()
    except Exception:
        return str(left_path) == str(right_path)


def _path_is_within(cls, candidate, root) -> bool:
    candidate_path = cls._path_value_to_path(candidate)
    root_path = cls._path_value_to_path(root)
    if candidate_path is None or root_path is None:
        return False

    try:
        candidate_path = candidate_path.resolve()
        root_path = root_path.resolve()
    except Exception:
        return False

    try:
        candidate_path.relative_to(root_path)
        return True
    except Exception:
        return False


def _dedupe_paths(cls, candidates) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates or []:
        if not candidate:
            continue

        path = cls._path_value_to_path(candidate)
        if path is None:
            continue

        try:
            path = path.resolve()
        except Exception:
            pass

        key = str(path)
        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return unique


def _path_is_within_any(self, candidate, roots) -> bool:
    for root in roots or []:
        if self._path_is_within(candidate, root):
            return True
    return False


def _get_annotation_output_base_dir(self) -> Path:
    if not self._is_free_mode_session_context():
        try:
            from ..campaign_manager import CAMPAIGN

            if CAMPAIGN.get_active_project_name():
                stage_dir = CAMPAIGN.get_staging_dir("auto_ann")
                if stage_dir is not None:
                    return Path(stage_dir)
        except Exception:
            pass

    return Path(CONFIG.get_auto_annotations_dir("plate"))


def _get_annotation_run_roots(self) -> list[Path]:
    if self._is_free_mode_session_context():
        return self._dedupe_paths([CONFIG.get_auto_annotations_dir("plate")])

    roots = []
    try:
        from ..campaign_manager import CAMPAIGN

        if CAMPAIGN.get_active_project_name():
            roots.extend([
                CAMPAIGN.get_staging_dir("auto_ann"),
                CAMPAIGN.get_dir("auto_ann"),
            ])
    except Exception:
        pass

    if not roots:
        roots.append(CONFIG.get_auto_annotations_dir("plate"))

    return self._dedupe_paths(roots)


def _coerce_annotation_output_dir(self, path_value) -> Path:
    fallback = self._get_annotation_output_base_dir()
    raw_value = self._path_value_to_text(path_value)

    candidate = self._path_value_to_path(raw_value) if raw_value else fallback
    if candidate is None:
        candidate = fallback

    if not self._path_is_within(candidate, fallback):
        if raw_value:
            workspace_root = Path(CONFIG.WORKSPACE_DIR)
            if self._path_is_within(candidate, workspace_root):
                logger.debug(
                    "Z2 dostosowalo katalog wyjsciowy do aktualnego workspace: %s -> %s",
                    candidate,
                    fallback,
                )
            else:
                logger.warning(
                    "Z2 skorygowalo katalog wyjsciowy do bezpiecznego workspace: %s",
                    fallback,
                )
        return fallback

    try:
        return candidate.resolve()
    except Exception:
        return candidate


def _resolve_safe_annotation_run_dir(self, path_value, *, require_xml: bool = False) -> Path | None:
    candidate = self._path_value_to_path(path_value)
    if candidate is None:
        return None

    try:
        candidate = candidate.resolve()
    except Exception:
        pass

    if not self._path_is_within_any(candidate, self._get_annotation_run_roots()):
        return None

    try:
        if not candidate.exists() or not candidate.is_dir():
            return None
    except Exception:
        return None

    if require_xml and not (candidate / "annotations.xml").exists():
        return None

    return candidate


def _get_annotation_session_text(
    self,
    key: str,
    default: str = "",
    *,
    allow_empty: bool = False,
    legacy_key: str | None = None,
) -> str:
    value = default
    if SESSION:
        try:
            value = SESSION.get("annotation", key, default)
        except Exception:
            value = default
        if legacy_key and (value is None or not str(value).strip()):
            try:
                value = SESSION.get("annotation", legacy_key, default)
            except Exception:
                value = default

    text = "" if value is None else str(value).strip()
    if text or allow_empty:
        return text
    return str(default)


def _get_annotation_session_float(self, key: str, default: float) -> float:
    value = default
    if SESSION:
        try:
            value = SESSION.get("annotation", key, default)
        except Exception:
            value = default

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get_annotation_session_int(self, key: str, default: int) -> int:
    value = default
    if SESSION:
        try:
            value = SESSION.get("annotation", key, default)
        except Exception:
            value = default

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_annotation_session_bool(self, key: str, default: bool) -> bool:
    value = default
    if SESSION:
        try:
            value = SESSION.get("annotation", key, default)
        except Exception:
            value = default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _get_annotation_session_list(self, key: str, default: list | None = None) -> list:
    fallback = list(default or [])
    value = fallback
    if SESSION:
        try:
            value = SESSION.get("annotation", key, fallback)
        except Exception:
            value = fallback

    if isinstance(value, list):
        return list(value)

    if isinstance(value, str):
        text = str(value or "").strip()
        if not text:
            return list(fallback)
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            return list(fallback)

    return list(fallback)


def _load_free_mode_session_snapshot(self) -> dict:
    defaults = self._annotation_session_defaults()
    snapshot = {
        "input_dir": self._get_annotation_session_text("input_dir", defaults["input_dir"], legacy_key="images_dir"),
        "output_dir": self._get_annotation_session_text("output_dir", defaults["output_dir"]),
        "mode": self._normalize_mode_value(self._get_annotation_session_text("mode", defaults["mode"])),
        "vehicle_model": self._get_annotation_session_text("vehicle_model", defaults["vehicle_model"], allow_empty=True),
        "vehicle_custom": self._get_annotation_session_text("vehicle_custom", defaults["vehicle_custom"], allow_empty=True),
        "plate_custom": self._get_annotation_session_text("plate_custom", defaults["plate_custom"], allow_empty=True),
        "character_model": self._get_annotation_session_text("character_model", defaults["character_model"]),
        "character_custom": self._get_annotation_session_text("character_custom", defaults["character_custom"], allow_empty=True),
        "device": self._get_annotation_session_text("device", defaults["device"]),
        "conf": self._get_annotation_session_float("conf", defaults["conf"]),
        "plate_dataset_run": self._get_annotation_session_text("plate_dataset_run", defaults["plate_dataset_run"], allow_empty=True),
        "plate_dataset_images": self._get_annotation_session_text("plate_dataset_images", defaults["plate_dataset_images"], allow_empty=True),
        "plate_train_pct": self._get_annotation_session_float("plate_train_pct", defaults["plate_train_pct"]),
        "plate_val_pct": self._get_annotation_session_float("plate_val_pct", defaults["plate_val_pct"]),
        "manual_xml_template": self._get_annotation_session_bool("manual_xml_template", defaults["manual_xml_template"]),
        "manual_vehicle_assist": self._get_annotation_session_bool("manual_vehicle_assist", defaults["manual_vehicle_assist"]),
        "workflow_route": self._get_annotation_session_text("workflow_route", defaults["workflow_route"], allow_empty=True),
        "manual_entry_mode": self._get_annotation_session_text("manual_entry_mode", defaults["manual_entry_mode"]),
        "auto_vehicle_choice": self._get_annotation_session_text("auto_vehicle_choice", defaults["auto_vehicle_choice"]),
        "workflow_step": self._get_annotation_session_text("workflow_step", defaults["workflow_step"], allow_empty=True),
        "free_mode_screen": self._get_annotation_session_text("free_mode_screen", defaults["free_mode_screen"]),
        "manual_review_active": self._get_annotation_session_bool("manual_review_active", defaults["manual_review_active"]),
        "manual_review_from_auto": self._get_annotation_session_bool("manual_review_from_auto", defaults["manual_review_from_auto"]),
        "manual_review_origin_route": self._get_annotation_session_text("manual_review_origin_route", defaults["manual_review_origin_route"], allow_empty=True),
        "last_completed_workflow_route": self._get_annotation_session_text("last_completed_workflow_route", defaults["last_completed_workflow_route"], allow_empty=True),
        "manual_review_export_ready": self._get_annotation_session_bool("manual_review_export_ready", defaults["manual_review_export_ready"]),
        "manual_review_history": self._get_annotation_session_list("manual_review_history", defaults["manual_review_history"]),
        "last_preview_run_dir": self._get_annotation_session_text("last_preview_run_dir", defaults["last_preview_run_dir"], allow_empty=True),
        "last_preview_index": self._get_annotation_session_int("last_preview_index", defaults["last_preview_index"]),
        "last_preview_filename": self._get_annotation_session_text("last_preview_filename", defaults["last_preview_filename"], allow_empty=True),
        "plate_model_identity": self._get_annotation_session_text("plate_model_identity", defaults["plate_model_identity"], allow_empty=True),
        "plate_model_source": self._get_annotation_session_text("plate_model_source", defaults["plate_model_source"], allow_empty=True),
        "plate_model_scope": self._get_annotation_session_text("plate_model_scope", defaults["plate_model_scope"], allow_empty=True),
    }
    snapshot = self._apply_annotation_auto_modal_defaults_to_snapshot(snapshot)
    return self._strip_persisted_annotation_miniflow_snapshot(snapshot)


def _collect_free_mode_session_snapshot(self) -> dict:
    run_dir_value = ""
    selected_ann = self._get_preview_annotation()
    input_dir_value = str(self.input_dir_var.get() or "").strip()
    if (
        self._normalize_workflow_route_value() == "manual"
        and self._normalize_manual_entry_mode() == "new"
        and not self._manual_review_active
        and not self._manual_review_from_auto
        and not self.is_processing
    ):
        input_dir_value = ""

    for candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.plate_dataset_run_var.get() or "").strip(),
    ):
        safe_run_dir = self._resolve_safe_annotation_run_dir(candidate)
        if safe_run_dir is not None:
            run_dir_value = str(safe_run_dir)
            break

    return {
        "input_dir": input_dir_value,
        "output_dir": str(self._coerce_annotation_output_dir(self.output_dir_var.get() or "")),
        "mode": self._normalize_mode_value(),
        "vehicle_model": str(self.vehicle_model_var.get() or "").strip(),
        "vehicle_custom": str(self.vehicle_custom_var.get() or "").strip(),
        "plate_custom": str(self.plate_custom_var.get() or "").strip(),
        "character_model": str(self.character_model_var.get() or "").strip(),
        "character_custom": str(self.character_custom_var.get() or "").strip(),
        "device": self._get_effective_yolo_device_choice(),
        "conf": float(self.conf_var.get()),
        "plate_dataset_run": str(self._resolve_safe_annotation_run_dir(self.plate_dataset_run_var.get()) or ""),
        "plate_dataset_images": str(self.plate_dataset_images_var.get() or "").strip(),
        "plate_train_pct": float(self.plate_train_pct.get()),
        "plate_val_pct": float(self.plate_val_pct.get()),
        "manual_xml_template": bool(self.manual_xml_template_var.get()),
        "manual_vehicle_assist": bool(self.manual_vehicle_assist_var.get()),
        "workflow_route": self._normalize_workflow_route_value(),
        "manual_entry_mode": self._normalize_manual_entry_mode(),
        "auto_vehicle_choice": self._normalize_auto_vehicle_choice(),
        "workflow_step": self._get_workflow_step(),
        "free_mode_screen": self._coerce_free_mode_screen(),
        "manual_review_active": bool(self._manual_review_active),
        "manual_review_from_auto": bool(self._manual_review_from_auto),
        "manual_review_origin_route": self._normalize_workflow_route_value(self._manual_review_origin_route),
        "last_completed_workflow_route": self._normalize_workflow_route_value(self._last_completed_workflow_route),
        "manual_review_export_ready": bool(self._manual_review_export_ready),
        "manual_review_history": list(self._manual_review_history_entries or []),
        "last_preview_run_dir": run_dir_value,
        "last_preview_index": (
            int(self.current_preview_index)
            if self.current_preview_index is not None
            else -1
        ),
        "last_preview_filename": str(getattr(selected_ann, "filename", "") or "").strip(),
        "plate_model_identity": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("identity") or "").strip(),
        "plate_model_source": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("source") or "").strip(),
        "plate_model_scope": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("scope") or "").strip(),
    }


def _collect_campaign_project_snapshot(self, *, include_preview_approved: bool = True) -> dict:
    selected_ann = self._get_preview_annotation()
    run_dir_value = ""
    for candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.plate_dataset_run_var.get() or "").strip(),
    ):
        safe_run_dir = self._resolve_safe_annotation_run_dir(candidate)
        if safe_run_dir is not None:
            run_dir_value = str(safe_run_dir)
            break

    try:
        from ..campaign_manager import CAMPAIGN
        active_project = CAMPAIGN.get_active_project_name()
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        if not run_dir_value:
            fallback_run_dir = self._resolve_safe_annotation_run_dir(
                CAMPAIGN.get_step2_staging_run(),
                require_xml=True,
            )
            if fallback_run_dir is not None:
                run_dir_value = str(fallback_run_dir)
    except Exception:
        active_project = ""
        iteration_num = 1

    payload = {
        "project": str(active_project or "").strip(),
        "iteration": int(iteration_num),
        "input_dir": str(self.input_dir_var.get() or "").strip(),
        "output_dir": str(self._coerce_annotation_output_dir(self.output_dir_var.get() or "")),
        "mode": self._normalize_mode_value(),
        "vehicle_model": str(self.vehicle_model_var.get() or "").strip(),
        "vehicle_custom": str(self.vehicle_custom_var.get() or "").strip(),
        "plate_custom": str(self.plate_custom_var.get() or "").strip(),
        "character_model": str(self.character_model_var.get() or "").strip(),
        "character_custom": str(self.character_custom_var.get() or "").strip(),
        "device": self._get_effective_yolo_device_choice(),
        "conf": float(self.conf_var.get()),
        "plate_dataset_run": run_dir_value,
        "plate_dataset_images": str(self.plate_dataset_images_var.get() or "").strip(),
        "plate_train_pct": float(self.plate_train_pct.get()),
        "plate_val_pct": float(self.plate_val_pct.get()),
        "manual_xml_template": bool(self.manual_xml_template_var.get()),
        "manual_vehicle_assist": bool(self.manual_vehicle_assist_var.get()),
        "auto_vehicle_choice": self._normalize_auto_vehicle_choice(),
        "manual_review_history": list(self._manual_review_history_entries or []),
        "last_preview_run_dir": run_dir_value,
        "last_preview_index": (
            int(self.current_preview_index)
            if self.current_preview_index is not None
            else -1
        ),
        "last_preview_filename": str(getattr(selected_ann, "filename", "") or "").strip(),
        "plate_model_identity": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("identity") or "").strip(),
        "plate_model_source": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("source") or "").strip(),
        "plate_model_scope": str(dict(getattr(self, "_plate_model_runtime_meta", {}) or {}).get("scope") or "").strip(),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if include_preview_approved:
        payload["preview_approved_filenames"] = sorted(self._get_preview_approved_filenames())
    return payload


def _sanitize_free_mode_session_snapshot(self, session_state: dict | None = None) -> dict:
    defaults = self._annotation_session_defaults()
    state = dict(session_state or {})
    project_root = CONFIG.DIR_9_PROJECTS

    def _sanitize_path_value(key: str, fallback: str = ""):
        raw_value = str(state.get(key) or "").strip()
        if raw_value and self._path_is_within(raw_value, project_root):
            state[key] = fallback
        elif raw_value:
            state[key] = raw_value
        else:
            state[key] = fallback

    _sanitize_path_value("input_dir", defaults["input_dir"])
    _sanitize_path_value("output_dir", defaults["output_dir"])
    _sanitize_path_value("vehicle_custom", "")
    _sanitize_path_value("plate_custom", "")
    _sanitize_path_value("character_custom", "")
    _sanitize_path_value("plate_dataset_run", "")
    _sanitize_path_value("plate_dataset_images", "")
    _sanitize_path_value("last_preview_run_dir", "")

    if not str(state.get("vehicle_custom") or "").strip() and str(state.get("vehicle_model") or "").strip() == "Custom":
        state["vehicle_model"] = defaults["vehicle_model"]

    state["free_mode_screen"] = (
        self._normalize_free_mode_screen_value(state.get("free_mode_screen"))
        or defaults["free_mode_screen"]
    )
    raw_manual_entry_mode = str(state.get("manual_entry_mode") or "").strip().lower()
    state["manual_entry_mode"] = (
        raw_manual_entry_mode
        if raw_manual_entry_mode in {"new", "continue", "import"}
        else defaults["manual_entry_mode"]
    )
    state["manual_review_origin_route"] = self._normalize_workflow_route_value(
        state.get("manual_review_origin_route")
    )

    return state


def capture_free_mode_snapshot_for_project_return(self):
    try:
        snapshot = self._collect_free_mode_session_snapshot()
    except Exception as e:
        logger.debug(f"Nie udalo sie zapisac migawki free mode przed wejsciem w projekt: {e}")
        return

    self._pre_campaign_free_mode_snapshot = self._sanitize_free_mode_session_snapshot(snapshot)


def _load_campaign_project_snapshot(self) -> dict:
    snapshot_path = self._get_campaign_annotation_state_path()
    if snapshot_path is None or not snapshot_path.exists():
        return {}

    try:
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return loaded if isinstance(loaded, dict) else {}


def _save_campaign_project_snapshot(self) -> bool:
    return self._save_campaign_project_snapshot_with_options(include_preview_approved=True)


def _save_campaign_project_snapshot_with_options(self, *, include_preview_approved: bool = True) -> bool:
    started_at = time.perf_counter()
    snapshot_path = self._get_campaign_annotation_state_path()
    if snapshot_path is None:
        return False

    try:
        payload = self._collect_campaign_project_snapshot(
            include_preview_approved=include_preview_approved,
        )
        if not include_preview_approved:
            existing = self._load_campaign_project_snapshot()
            if "preview_approved_filenames" in existing and "preview_approved_filenames" not in payload:
                payload["preview_approved_filenames"] = list(existing.get("preview_approved_filenames") or [])
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 180.0:
            try:
                logger.info(
                    "[Z2 PERF] campaign_project_snapshot_save total=%.0fms include_ok=%s",
                    elapsed_ms,
                    int(bool(include_preview_approved)),
                )
            except Exception:
                pass
        return True
    except Exception as e:
        logger.debug(f"Nie udalo sie zapisac projektowego stanu Z2: {e}")
        return False


def _cancel_pending_campaign_route_cleanup(self) -> None:
    pending = getattr(self, "_campaign_route_cleanup_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._campaign_route_cleanup_after_id = None
    self._campaign_route_cleanup_token = int(
        getattr(self, "_campaign_route_cleanup_token", 0) or 0
    ) + 1


def _build_campaign_route_cleanup_payload(
    self,
    *,
    active_project: str,
    candidate_inputs: list[Path],
    explicit_run_dirs: set[str],
    manual_run_keep: set[str],
    remove_persisted_runs: bool,
) -> dict:
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return {}

    search_roots: list[Path] = []
    if bool(remove_persisted_runs):
        for root_candidate in (
            CAMPAIGN.get_staging_dir("auto_ann"),
            CAMPAIGN.get_dir("auto_ann"),
        ):
            if root_candidate is None:
                continue
            try:
                search_roots.append(Path(root_candidate))
            except Exception:
                continue

    stage_root = CAMPAIGN.get_staging_dir("plate_stage")
    try:
        stage_root_path = Path(stage_root) if stage_root is not None else None
    except Exception:
        stage_root_path = None

    try:
        stage_dir = self._get_manual_plate_stage_dir()
    except Exception:
        stage_dir = None

    return {
        "active_project": str(active_project or "").strip(),
        "candidate_inputs": list(candidate_inputs or []),
        "explicit_run_dirs": set(explicit_run_dirs or set()),
        "manual_run_keep": set(manual_run_keep or set()),
        "search_roots": search_roots,
        "stage_root": stage_root_path,
        "stage_dir": stage_dir,
        "remove_persisted_runs": bool(remove_persisted_runs),
    }


def _schedule_deferred_campaign_route_cleanup(self, *args, **kwargs):
    return z2_workflow_methods._schedule_deferred_campaign_route_cleanup(self, *args, **kwargs)


def reset_campaign_iteration_route_state(self, *args, **kwargs):
    return z2_workflow_methods.reset_campaign_iteration_route_state(self, *args, **kwargs)


def _dir_has_images(path_like) -> bool:
    if not path_like:
        return False

    try:
        candidate_dir = Path(path_like)
    except Exception:
        return False

    try:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            return False
    except Exception:
        return False

    try:
        for item in candidate_dir.iterdir():
            try:
                if item.is_file() and item.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _resolve_annotation_input_images_dir(path_like) -> Path | None:
    if not path_like:
        return None

    try:
        base_dir = Path(path_like)
    except Exception:
        return None

    try:
        nested_images_dir = base_dir / "images"
        if (
            nested_images_dir.exists()
            and nested_images_dir.is_dir()
            and _dir_has_images(nested_images_dir)
        ):
            return nested_images_dir
    except Exception:
        pass

    try:
        if base_dir.exists() and base_dir.is_dir() and _dir_has_images(base_dir):
            return base_dir
    except Exception:
        pass

    return None


def _annotation_input_cache_key(path_like) -> str:
    if not path_like:
        return ""
    try:
        return str(Path(path_like).absolute()).strip().lower()
    except Exception:
        return str(path_like or "").strip().lower()


def _remember_annotation_input_dir_ready(self, path_like, resolved_dir: Path | None) -> None:
    if resolved_dir is None:
        return
    cache = dict(getattr(self, "_annotation_input_ready_cache", {}) or {})
    resolved_text = str(Path(resolved_dir))
    for candidate in (path_like, resolved_dir):
        key = self._annotation_input_cache_key(candidate)
        if key:
            cache[key] = resolved_text
    self._annotation_input_ready_cache = cache


def _annotation_input_dir_ready(self, path_like=None) -> bool:
    candidate = path_like
    if candidate is None:
        candidate = str(self.input_dir_var.get() or "").strip() if hasattr(self, "input_dir_var") else ""
    key = self._annotation_input_cache_key(candidate)
    cache = getattr(self, "_annotation_input_ready_cache", {}) or {}
    if key and key in cache:
        return True
    try:
        current_input = getattr(self, "current_input_dir", None)
        if current_input is not None and key == self._annotation_input_cache_key(current_input):
            self._remember_annotation_input_dir_ready(candidate, Path(current_input))
            return True
    except Exception:
        pass
    try:
        resolved = self._resolve_annotation_input_images_dir(candidate)
    except Exception:
        resolved = None
    if resolved is not None:
        self._remember_annotation_input_dir_ready(candidate, resolved)
        return True
    return False


def prepare_campaign_iteration_transition(self, *, input_dir: Path | None = None, refresh_ui: bool = True) -> dict:
    result = {"removed_snapshot": False}

    try:
        from ..campaign_manager import CAMPAIGN

        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        active_project = ""

    pending = getattr(self, "_free_mode_session_save_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._free_mode_session_save_after_id = None

    try:
        if active_project:
            snapshot_path = self._get_campaign_annotation_state_path(active_project)
        else:
            snapshot_path = None
        if snapshot_path is not None and snapshot_path.exists():
            snapshot_path.unlink()
            result["removed_snapshot"] = True
    except Exception as e:
        logger.debug(f"Nie udalo sie usunac snapshotu Z2 przed nowa iteracja: {e}")

    try:
        self._preview_session_restore_index = -1
        self._preview_session_restore_filename = ""
        self._reset_campaign_runtime_state(input_dir=input_dir)
    except Exception as e:
        logger.debug(f"Nie udalo sie wyczyscic runtime Z2 przed nowa iteracja: {e}")

    self._campaign_context_project_name = ""
    if refresh_ui:
        try:
            self._refresh_step2_action_states()
        except Exception:
            pass
    return result


def _get_campaign_iteration_artifact_bundle(
    self,
    *,
    images_dir: Path | None = None,
    iteration_num: int | None = None,
) -> dict:
    if self._is_free_mode_session_context():
        return {}
    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return {}
        iter_value = int(iteration_num or CAMPAIGN.get_current_iteration_num() or 1)
        preferred_images_dir = (
            images_dir
            or CAMPAIGN.get_iteration_image_source_dir(iter_value)
            or CAMPAIGN.get_master_pool_dir()
            or CAMPAIGN.get_iteration_raw_dir(iter_value)
        )
        return dict(
            CAMPAIGN.get_iteration_artifact_bundle(
                images_dir=preferred_images_dir,
                iteration_num=iter_value,
            ) or {}
        )
    except Exception:
        return {}


def _get_campaign_step2_active_run_entry(
    self,
    *,
    images_dir: Path | None = None,
    iteration_num: int | None = None,
) -> dict:
    bundle = self._get_campaign_iteration_artifact_bundle(
        images_dir=images_dir,
        iteration_num=iteration_num,
    )
    return dict(bundle.get("step2_active_run") or {})


def _normalize_campaign_step2_bootstrap_manual_template(self, bootstrap: dict | None) -> dict:
    result = dict(bootstrap or {})
    restore_run_dir = self._resolve_safe_annotation_run_dir(
        result.get("restore_run_dir"),
        require_xml=True,
    )
    if restore_run_dir is None:
        return result

    result["restore_run_dir"] = restore_run_dir
    try:
        manifest = self._load_annotation_run_manifest(restore_run_dir)
    except Exception:
        manifest = {}
    if self._annotation_run_manifest_is_manual_template(manifest):
        result["manual_template"] = True
    return result


def _get_campaign_auto_annotation_bootstrap(self, *args, **kwargs):
    return z2_workflow_methods._get_campaign_auto_annotation_bootstrap(self, *args, **kwargs)


def _normalize_campaign_iteration_target_value(target: str | None) -> str:
    value = str(target or "").strip().lower()
    if value in {"plate", "plates", "tablica", "tablice", "pose"}:
        return "plate"
    if value in {"char", "chars", "character", "characters", "znak", "znaki"}:
        return "char"
    return ""


def _is_campaign_char_repair_return_mode(self) -> bool:
    if self._is_free_mode_session_context():
        return False
    try:
        from ..campaign_manager import CAMPAIGN

        return bool(
            CAMPAIGN.get_active_project_name()
            and int(CAMPAIGN.get_current_step() or 0) == 3
            and str(CAMPAIGN.get_iteration_target() or "").strip().lower() == "char"
        )
    except Exception:
        return False


def _is_campaign_plate_step4_repair_return_mode(self) -> bool:
    if self._is_free_mode_session_context():
        return False
    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return False
        if int(CAMPAIGN.get_current_step() or 0) != 4:
            return False
        if str(CAMPAIGN.get_iteration_target() or "").strip().lower() != "plate":
            return False

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
            return True

        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
        approved_plates = int(approved_stats.get("plates", 0) or 0)
        min_plate_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        return approved_plates < min_plate_approval_plates
    except Exception:
        return False


def get_campaign_step2_bootstrap(self, *, iteration_target: str | None = None) -> dict:
    target = self._normalize_campaign_iteration_target_value(iteration_target)
    bootstrap = self._normalize_campaign_step2_bootstrap_manual_template(
        self._get_campaign_auto_annotation_bootstrap(target)
    )
    restore_run_dir = self._resolve_safe_annotation_run_dir(
        bootstrap.get("restore_run_dir"),
        require_xml=True,
    )
    bootstrap["restore_run_dir"] = restore_run_dir
    bootstrap["input_source"] = str(bootstrap.get("input_source") or "").strip()
    bootstrap["manual_template"] = bool(bootstrap.get("manual_template"))
    bootstrap["plate_model_path"] = str(bootstrap.get("plate_model_path") or "").strip()
    return bootstrap


def get_campaign_step2_source_state(self, *args, **kwargs):
    return z2_workflow_methods.get_campaign_step2_source_state(self, *args, **kwargs)


def _get_campaign_latest_approved_plate_run_state(self, *, input_dir: Path | None = None) -> dict:
    result = {
        "run_dir": None,
        "run_name": "",
        "images_with_plates": 0,
        "total_plates": 0,
    }

    if self._is_free_mode_session_context():
        return result

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return result

        auto_dir = CAMPAIGN.get_dir("auto_ann")
        if auto_dir is None:
            return result
    except Exception:
        return result

    try:
        auto_root = Path(auto_dir)
    except Exception:
        return result

    approved_run = None
    if input_dir is not None:
        try:
            approved_run = self._find_latest_annotation_run_for_input(input_dir, [auto_root])
        except Exception:
            approved_run = None

    if approved_run is None:
        approved_run = self._find_latest_annotation_run_dir(auto_root)

    approved_run = self._resolve_safe_annotation_run_dir(approved_run, require_xml=True)
    if approved_run is None:
        return result

    try:
        images_with_plates, total_plates = self._get_run_plate_annotation_counts(approved_run)
    except Exception:
        images_with_plates, total_plates = 0, 0

    result.update(
        run_dir=approved_run,
        run_name=str(approved_run.name or "").strip(),
        images_with_plates=int(images_with_plates or 0),
        total_plates=int(total_plates or 0),
    )
    return result


def get_campaign_step2_view_model(self, *args, **kwargs):
    return z2_workflow_methods.get_campaign_step2_view_model(self, *args, **kwargs)


def get_campaign_step2_wizard_status(self) -> dict:
    try:
        self._refresh_step2_action_states()
    except Exception:
        pass
    vm = self.get_campaign_step2_view_model()
    primary_cta = vm.primary_cta
    secondary_cta = vm.secondary_cta
    return {
        "stage_key": vm.stage_key,
        "iteration_target": vm.iteration_target,
        "current_step": vm.current_step,
        "step2_status": vm.step2_status,
        "state": vm.state,
        "summary": vm.summary,
        "details": vm.details,
        "primary_action_id": (primary_cta.command_id if primary_cta else ""),
        "primary_action_label": (primary_cta.label if primary_cta else ""),
        "primary_action_context": (dict(primary_cta.command_context) if primary_cta else {}),
        "secondary_action_id": (secondary_cta.command_id if secondary_cta else ""),
        "secondary_action_label": (secondary_cta.label if secondary_cta else ""),
        "secondary_action_context": (dict(secondary_cta.command_context) if secondary_cta else {}),
        "ready_for_approval": bool(getattr(self, "_campaign_step2_approval_ready", False)),
        "approval_action": str(getattr(self, "_campaign_step2_approval_action", "") or "").strip(),
        "approval_iteration_target": str(getattr(self, "_campaign_step2_approval_iteration_target", "") or "").strip(),
        "approval_repair_mode": bool(getattr(self, "_campaign_step2_approval_repair_mode", False)),
        "approval_hint_text": str(getattr(self, "_campaign_step2_approval_hint_text", "") or "").strip(),
        "approval_hint_tone": str(getattr(self, "_campaign_step2_approval_hint_tone", "") or "").strip(),
    }


def _get_campaign_previous_manual_source_bundle(self) -> dict:
    if self._is_free_mode_session_context():
        return {"input_dir": None, "run_dir": None, "xml_path": None, "image_dirs": [], "source_label": "wcześniejszej iteracji"}

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return {"input_dir": None, "run_dir": None, "xml_path": None, "image_dirs": [], "source_label": "wcześniejszej iteracji"}

        stored_manual_source = CAMPAIGN.get_last_plate_manual_source()
        run_dir = None
        xml_path = None

        for raw_candidate in (
            str(stored_manual_source.get("source_run_path") or "").strip(),
            str(stored_manual_source.get("source_xml_path") or "").strip(),
        ):
            if not raw_candidate:
                continue
            try:
                candidate = Path(raw_candidate)
                if candidate.suffix.lower() == ".xml":
                    xml_path = candidate if candidate.exists() else xml_path
                    candidate = candidate.parent
                if candidate.exists() and candidate.is_dir() and (candidate / "annotations.xml").exists():
                    run_dir = candidate
                    xml_path = candidate / "annotations.xml"
                    break
            except Exception:
                continue

        if run_dir is None:
            run_dir = self._find_reused_manual_source_run(None)
            if run_dir is not None and (run_dir / "annotations.xml").exists():
                xml_path = run_dir / "annotations.xml"

        input_dir_candidates = []
        source_input_path = str(stored_manual_source.get("source_input_path") or "").strip()
        if source_input_path:
            input_dir_candidates.append(source_input_path)

        if run_dir is not None:
            manifest = self._load_annotation_run_manifest(run_dir)
            manifest_input_dir = str(manifest.get("input_dir") or "").strip()
            if manifest_input_dir:
                input_dir_candidates.append(manifest_input_dir)

        resolved_input_dir = None
        resolved_image_dirs: list[Path] = []
        for raw_dir in input_dir_candidates:
            try:
                candidate_dir = Path(raw_dir)
            except Exception:
                continue
            if self._dir_has_images(candidate_dir):
                if all(str(existing) != str(candidate_dir) for existing in resolved_image_dirs):
                    resolved_image_dirs.append(candidate_dir)
                if resolved_input_dir is None:
                    resolved_input_dir = candidate_dir
        if run_dir is not None:
            run_images_dir = run_dir / "images"
            if self._dir_has_images(run_images_dir):
                if all(str(existing) != str(run_images_dir) for existing in resolved_image_dirs):
                    resolved_image_dirs.append(run_images_dir)

        source_label = self._extract_iteration_label_from_path(
            resolved_input_dir or run_dir or source_input_path
        )
        return {
            "input_dir": resolved_input_dir,
            "run_dir": run_dir,
            "xml_path": xml_path if xml_path is not None and xml_path.exists() else None,
            "image_dirs": resolved_image_dirs,
            "source_label": source_label,
        }
    except Exception:
        return {"input_dir": None, "run_dir": None, "xml_path": None, "image_dirs": [], "source_label": "wcześniejszej iteracji"}


def _extract_iteration_label_from_path(path_like) -> str:
    if not path_like:
        return "wcześniejszej iteracji"

    try:
        path = Path(path_like)
    except Exception:
        path = None

    for part in reversed(list(path.parts) if path is not None else [str(path_like)]):
        match = re.search(r"iteracja[_\-\s]*0*(\d+)", str(part), flags=re.IGNORECASE)
        if match:
            try:
                return f"Iteracji {int(match.group(1))}"
            except Exception:
                return f"Iteracji {match.group(1)}"

    return "wcześniejszej iteracji"


def _load_cvat_plate_annotated_filenames(xml_path: Path | None) -> list[str]:
    if xml_path is None:
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    filenames: list[str] = []
    seen: set[str] = set()
    for image_el in root.findall(".//image"):
        filename = str(image_el.get("name", "") or "").strip()
        if not filename or filename in seen:
            continue
        has_plate = False
        for poly_el in image_el.findall("polygon"):
            label = str(poly_el.get("label", "") or "").strip().lower()
            if label in CONFIG.PLATE_LABELS:
                has_plate = True
                break
        if not has_plate:
            for box_el in image_el.findall("box"):
                label = str(box_el.get("label", "") or "").strip().lower()
                if label in CONFIG.PLATE_LABELS:
                    has_plate = True
                    break
        if has_plate:
            seen.add(filename)
            filenames.append(filename)
    return filenames


def _get_campaign_reused_manual_annotation_count(self) -> int:
    if self._is_free_mode_session_context():
        return 0

    try:
        previous_bundle = self._get_campaign_previous_manual_source_bundle()
        previous_xml_path = previous_bundle.get("xml_path")
        return len(self._load_cvat_plate_annotated_filenames(previous_xml_path))
    except Exception:
        return 0


def _ensure_campaign_manual_package_ready(self, *, iteration_target: str | None = None) -> bool:
    if self._is_free_mode_session_context():
        return False
    if bool(getattr(self, "is_processing", False)):
        return False
    if self._get_workflow_route() != "manual":
        return False
    if self._get_manual_entry_mode() != "new":
        return False

    input_dir_value = str(self.input_dir_var.get() or "").strip()
    if not input_dir_value or not Path(input_dir_value).exists():
        return False

    manual_run_dir = self._get_active_annotation_run_dir(require_xml=True)
    if manual_run_dir is not None and self._has_active_manual_template_run():
        return self._open_existing_run_for_campaign_review(
            run_dir=manual_run_dir,
            iteration_target=iteration_target,
            manual_template=True,
        )

    self._start_annotation()
    return True


def _ensure_campaign_preview_edit_run(self) -> bool:
    if self._is_free_mode_session_context():
        return False
    if not self.current_annotations:
        return False
    if self._get_current_annotation_xml_path() is not None:
        return True

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return False
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
        char_repair_return = bool(
            current_step == 3
            and iteration_target == "char"
            and step3_status == "needs_rework"
        )
        plate_step4_repair_return = bool(
            current_step == 4
            and iteration_target == "plate"
            and self._is_campaign_plate_step4_repair_return_mode()
        )
        if not (current_step == 2 or char_repair_return or plate_step4_repair_return):
            return False
        if current_step == 2 and int(CAMPAIGN.get_current_iteration_num() or 1) <= 1:
            return False
    except Exception:
        return False

    if iteration_target not in {"plate", "char"}:
        return False

    input_dir = getattr(self, "current_input_dir", None)
    if input_dir is None:
        input_dir_value = str(self.input_dir_var.get() or "").strip()
        try:
            input_dir = Path(input_dir_value) if input_dir_value else None
        except Exception:
            input_dir = None
    if input_dir is None:
        return False

    try:
        base_out_dir = self._coerce_annotation_output_dir(self.output_dir_var.get())
        run_dir = self._allocate_annotation_run_dir(base_out_dir, suffix="campaign_preview")
        xml_path = run_dir / "annotations.xml"
        success = CVATExporter().export(
            self.current_annotations,
            xml_path,
            include_confidence=True,
            only_successful=False,
        )
        if not success:
            return False

        prev_manual_template = bool(getattr(self, "_current_run_manual_template", False))
        prev_vehicle_assist = bool(getattr(self, "_current_run_manual_vehicle_assist", False))
        try:
            self._current_run_manual_template = True
            self._current_run_manual_vehicle_assist = False
            self._write_annotation_run_manifest(run_dir, Path(input_dir))
        finally:
            self._current_run_manual_template = prev_manual_template
            self._current_run_manual_vehicle_assist = prev_vehicle_assist

        self.current_annotation_run_dir = run_dir
        self.current_annotation_xml_path = xml_path
        self.last_staging_run_dir = run_dir
        self._update_annotation_run_manifest(
            run_dir,
            has_manual_edits=bool(self._collect_preview_manually_touched_filenames()),
            last_manual_edit_at="",
            last_manual_edit_kind="campaign_preview_seed",
            manual_touched_filenames=sorted(self._collect_preview_manually_touched_filenames()),
            approved_filenames=sorted(self._get_preview_approved_filenames()),
            **self._collect_preview_resume_manifest_fields(),
        )
        try:
            self._load_plate_dataset_context_from_run(run_dir, force_images_update=False)
        except Exception:
            pass
        try:
            self._save_campaign_project_snapshot()
        except Exception:
            pass
        return True
    except Exception as e:
        logger.debug(f"Nie udało się przygotować draft XML dla kampanijnego Z2: {e}")
        return False


def _get_campaign_plate_approved_filenames(self) -> set[str]:
    if self._is_free_mode_session_context():
        return set()

    try:
        from ..campaign_manager import CAMPAIGN

        project_name = _resolve_campaign_context_project_name(self)
        if not project_name:
            return set()
        entries = list(CAMPAIGN.list_plate_approved_entries(project_name) or [])
    except Exception:
        return set()

    approved_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for raw_name in (entry.get("image_name", ""), entry.get("entry_key", "")):
            raw_text = str(raw_name or "").strip()
            if not raw_text:
                continue
            approved_names.add(raw_text.lower())
            try:
                normalized_name = CAMPAIGN._normalize_image_set_name(raw_text)
            except Exception:
                try:
                    normalized_name = str(Path(raw_text.replace("\\", "/")).name or "").strip().lower()
                except Exception:
                    normalized_name = raw_text.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
            if normalized_name:
                approved_names.add(normalized_name)
    return approved_names


def _build_campaign_source_image_key(image_path) -> str:
    try:
        # This key is used for same-file matching across project manifests.
        # `Path.resolve()` is needlessly expensive on large Windows folders,
        # because it touches the filesystem for every image during Z2 restore.
        return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(image_path)))).strip().lower()
    except Exception:
        try:
            return os.path.normcase(os.path.abspath(str(Path(image_path)))).strip().lower()
        except Exception:
            return str(image_path or "").strip().lower()


def _resolve_campaign_context_project_name(self, *path_candidates) -> str:
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

    context_project = str(getattr(self, "_campaign_context_project_name", "") or "").strip()
    if context_project:
        return context_project

    try:
        snapshot_project = str(dict(self._load_campaign_project_snapshot() or {}).get("project") or "").strip()
    except Exception:
        snapshot_project = ""
    if snapshot_project:
        return snapshot_project

    candidates = list(path_candidates or [])
    candidates.extend(
        [
            getattr(self, "current_annotation_run_dir", None),
            getattr(self, "last_staging_run_dir", None),
            getattr(self, "current_input_dir", None),
        ]
    )
    for var_name in ("input_dir_var", "plate_dataset_images_var", "plate_dataset_run_var"):
        try:
            value = getattr(self, var_name).get()
        except Exception:
            value = ""
        if value:
            candidates.append(value)

    normalized_candidates: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            normalized_candidates.append(Path(candidate).resolve())
        except Exception:
            try:
                normalized_candidates.append(Path(candidate))
            except Exception:
                continue
    if not normalized_candidates:
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
        for candidate_path in normalized_candidates:
            try:
                if candidate_path == project_root or project_root in candidate_path.parents:
                    return str(project_name or "").strip()
            except Exception:
                candidate_text = str(candidate_path).lower().rstrip("\\/")
                root_text = str(project_root).lower().rstrip("\\/")
                if candidate_text == root_text or candidate_text.startswith(root_text + os.sep):
                    return str(project_name or "").strip()
    return ""


def _get_campaign_plate_approved_source_keys(self) -> set[str]:
    if self._is_free_mode_session_context():
        return set()

    try:
        from ..campaign_manager import CAMPAIGN

        project_name = _resolve_campaign_context_project_name(self)
        if not project_name:
            return set()
        entries = list(CAMPAIGN.list_plate_approved_entries(project_name) or [])
    except Exception:
        return set()

    approved_source_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_image_path = str(entry.get("source_image_path", "") or "").strip()
        if not source_image_path:
            continue
        source_key = self._build_campaign_source_image_key(source_image_path)
        if source_key:
            approved_source_keys.add(source_key)
    return approved_source_keys


def _get_campaign_step3_preview_source_context(self, *args, **kwargs):
    return z2_workflow_methods._get_campaign_step3_preview_source_context(self, *args, **kwargs)


def _get_campaign_char_effective_source_hidden_filenames(self) -> set[str]:
    if self._is_free_mode_session_context():
        return set()

    try:
        from ..campaign_manager import CAMPAIGN

        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        if not project_name:
            return set()
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        if iteration_target != "char":
            return set()
        try:
            iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            iteration_num = 0
        extract_state = dict(CAMPAIGN.get_step3_extract_state() or {})
        try:
            project_state_dir = CAMPAIGN.get_project_state_dir()
        except Exception:
            project_state_dir = None
    except Exception:
        return set()

    def _path_signature(raw_path) -> tuple[str, int, int]:
        if not raw_path:
            return ("", 0, 0)
        try:
            path = Path(raw_path)
            key = str(path.resolve()).strip().lower()
            stat = path.stat() if path.exists() else None
            return (
                key,
                int(getattr(stat, "st_mtime_ns", 0) or 0) if stat is not None else 0,
                int(getattr(stat, "st_size", 0) or 0) if stat is not None else 0,
            )
        except Exception:
            return (str(raw_path).strip().lower(), 0, 0)

    raw_run_dir = str(extract_state.get("annotation_run_dir") or "").strip()
    candidate_signatures = [
        _path_signature(str(extract_state.get("xml_path") or "").strip()),
        _path_signature(Path(raw_run_dir) / "annotations.xml" if raw_run_dir else ""),
        _path_signature(Path(project_state_dir) / "char_effective_source" / "annotations.xml" if project_state_dir else ""),
    ]
    cache_key = (
        project_name,
        iteration_num,
        iteration_target,
        tuple(sorted((str(key), str(value)) for key, value in extract_state.items())),
        tuple(candidate_signatures),
    )
    cache = getattr(self, "_campaign_char_effective_hidden_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return set(cache.get("filenames") or set())

    preview_context = self._get_campaign_step3_preview_source_context()
    hidden_filenames = {
        str(name or "").strip().lower()
        for name in set(preview_context.get("filenames") or set())
        if str(name or "").strip()
    }

    candidate_xml_paths: list[Path] = []
    seen_candidates: set[str] = set()

    def add_candidate(raw_path) -> None:
        if not raw_path:
            return
        try:
            candidate = Path(raw_path)
        except Exception:
            return
        try:
            key = str(candidate.resolve()).strip().lower()
        except Exception:
            key = str(candidate).strip().lower()
        if not key or key in seen_candidates:
            return
        seen_candidates.add(key)
        candidate_xml_paths.append(candidate)

    add_candidate(str(extract_state.get("xml_path") or "").strip())
    run_dir_raw = str(extract_state.get("annotation_run_dir") or "").strip()
    if run_dir_raw:
        add_candidate(Path(run_dir_raw) / "annotations.xml")

    if project_state_dir:
        add_candidate(Path(project_state_dir) / "char_effective_source" / "annotations.xml")

    for xml_path in candidate_xml_paths:
        try:
            if not xml_path.exists() or not xml_path.is_file():
                continue
        except Exception:
            continue
        filenames = self._load_cvat_plate_annotated_filenames(xml_path)
        if filenames:
            hidden_filenames |= {
                str(name or "").strip().lower()
                for name in filenames
                if str(name or "").strip()
            }

    self._campaign_char_effective_hidden_cache = {
        "key": cache_key,
        "filenames": set(hidden_filenames),
    }
    return hidden_filenames


def _get_campaign_iteration_manifest_image_paths(self, base_dir: Path | None = None) -> list[Path]:
    if self._is_free_mode_session_context():
        return []

    try:
        from ..campaign_manager import CAMPAIGN

        return list(CAMPAIGN.get_iteration_manifest_image_paths(base_dir=base_dir) or [])
    except Exception:
        return []


def _get_campaign_iteration_manifest_image_count(self) -> int:
    if self._is_free_mode_session_context():
        return 0

    try:
        from ..campaign_manager import CAMPAIGN

        return int(CAMPAIGN.get_iteration_manifest_image_count() or 0)
    except Exception:
        return 0


def _collect_campaign_auto_annotation_sources(self, *args, **kwargs):
    return z2_workflow_methods._collect_campaign_auto_annotation_sources(self, *args, **kwargs)


def _clear_campaign_manual_reuse_context(self):
    self._campaign_reuse_manual_filenames = set()
    self._campaign_reuse_manual_summary = {}
    self._campaign_pending_batch_summary = {}


def _campaign_reuse_manual_badge() -> str:
    return "[RĘ↺]"


def _apply_campaign_manual_reuse_context(self, source_plan: dict | None = None):
    if not isinstance(source_plan, dict):
        self._clear_campaign_manual_reuse_context()
        return

    self._campaign_pending_batch_summary = {
        "pending_count": int(source_plan.get("total_count", 0) or 0),
        "base_count": int(source_plan.get("base_count", 0) or 0),
        "reused_count": int(source_plan.get("reused_count", 0) or 0),
        "approved_skip_count": int(source_plan.get("approved_skip_count", 0) or 0),
        "char_effective_skip_count": int(source_plan.get("char_effective_skip_count", 0) or 0),
        "manual_skip_count": int(source_plan.get("manual_skip_count", 0) or 0),
        "current_manual_count": int(source_plan.get("current_manual_count", 0) or 0),
        "source_label": str(source_plan.get("source_label") or "wcześniejszej iteracji"),
    }

    reused_filenames = set(source_plan.get("reused_filenames") or set())
    if not reused_filenames:
        self._campaign_reuse_manual_filenames = set()
        self._campaign_reuse_manual_summary = {}
        return

    self._campaign_reuse_manual_filenames = reused_filenames
    self._campaign_reuse_manual_summary = {
        "source_label": str(source_plan.get("source_label") or "wcześniejszej iteracji"),
        "reused_count": int(source_plan.get("reused_count", 0) or 0),
        "base_count": int(source_plan.get("base_count", 0) or 0),
        "total_count": int(source_plan.get("total_count", 0) or 0),
    }


def _sync_campaign_pending_batch_summary_from_preview(
    self,
    *,
    hidden_project_approved_count: int | None = None,
) -> None:
    summary = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
    summary["pending_count"] = int(len(self.current_annotations or []))
    if hidden_project_approved_count is not None:
        summary["approved_skip_count"] = max(0, int(hidden_project_approved_count or 0))
    else:
        summary["approved_skip_count"] = int(summary.get("approved_skip_count", 0) or 0)
    summary["current_manual_count"] = int(
        len(self._collect_campaign_current_iteration_manual_filenames(self.current_annotations))
    )
    summary["char_effective_skip_count"] = max(
        0,
        int(
            getattr(self, "_campaign_hidden_char_effective_count", summary.get("char_effective_skip_count", 0))
            or 0
        ),
    )
    self._campaign_pending_batch_summary = summary


def _on_campaign_reuse_manual_toggle(self):
    self._refresh_campaign_manual_reuse_option_ui()

    if self._is_free_mode_session_context() or getattr(self, "is_processing", False):
        return

    if getattr(self, "current_annotation_xml_path", None):
        return

    input_dir_raw = str(self.input_dir_var.get() or "").strip()
    if not input_dir_raw:
        return

    try:
        self._prime_campaign_source_preview(Path(input_dir_raw))
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć podglądu Z2 po zmianie checkboxu dołączania ręcznych zdjęć: {e}")


def _get_campaign_manual_reuse_ui_state(self) -> dict:
    state = {
        "show_option": False,
        "source_label": "wcześniejszej iteracji",
        "reused_count": 0,
        "base_count": 0,
        "total_count": 0,
    }

    if self._is_free_mode_session_context():
        return state
    return state


def _refresh_campaign_manual_reuse_option_ui(self):
    check = getattr(self, "campaign_reuse_manual_check", None)
    hint = getattr(self, "campaign_reuse_manual_hint_lbl", None)
    if check is None or hint is None:
        return

    ui_state = self._get_campaign_manual_reuse_ui_state()
    show_option = bool(ui_state.get("show_option"))
    source_label = str(ui_state.get("source_label") or "wcześniejszej iteracji")
    reused_count = int(ui_state.get("reused_count", 0) or 0)
    base_count = int(ui_state.get("base_count", 0) or 0)
    total_count = int(ui_state.get("total_count", 0) or 0)

    if not show_option:
        self.campaign_reuse_manual_var.set(False)
        self.campaign_reuse_manual_hint_var.set("")
        try:
            check.configure(text="Dołącz ręcznie anotowane zdjęcia z wcześniejszych iteracji")
        except Exception:
            pass

    self._set_widget_packed(
        check,
        show_option,
        anchor=tk.W,
        pady=(6, 2),
    )
    self._set_widget_packed(
        hint,
        show_option,
        fill=tk.X,
        pady=(0, 6),
    )

    if show_option:
        try:
            check.configure(
                text=f"Dołącz {reused_count} ręcznie anotowanych zdjęć z {source_label} ({self._campaign_reuse_manual_badge()})"
            )
        except Exception:
            pass

        if bool(self.campaign_reuse_manual_var.get()):
            self.campaign_reuse_manual_hint_var.set(
                f"Dołączysz {reused_count} ręcznie anotowanych obrazów z {source_label}. Razem do autoanotacji "
                f"trafi {total_count} obrazów: {base_count} z bieżącego zestawu oraz {reused_count} z wcześniejszej iteracji. "
                f"Na liście po prawej pozycje z wcześniejszej iteracji są oznaczone jako {self._campaign_reuse_manual_badge()}. "
                "Uwaga: poprzednie ręczne anotacje tych zdjęć "
                "nie zostaną zachowane w nowym runie."
            )
        else:
            self.campaign_reuse_manual_hint_var.set(
                f"Możesz dołączyć {reused_count} ręcznie anotowanych obrazów z {source_label}. Po zaznaczeniu "
                f"checkboxu pojawią się one na liście po prawej z oznaczeniem {self._campaign_reuse_manual_badge()}. Uwaga: poprzednie ręczne "
                "anotacje tych zdjęć nie zostaną zachowane w nowym runie."
            )


def _restore_preview_from_annotation_run(self, *args, **kwargs):
    return z2_workflow_methods._restore_preview_from_annotation_run(self, *args, **kwargs)


def _finalize_successful_annotation_run_ui(self, *args, **kwargs):
    return z2_workflow_methods._finalize_successful_annotation_run_ui(self, *args, **kwargs)


def _apply_campaign_project_snapshot(self, *args, **kwargs):
    return z2_workflow_methods._apply_campaign_project_snapshot(self, *args, **kwargs)


def _bind_free_mode_session_observers(self):
    observed_vars = (
        self.input_dir_var,
        self.output_dir_var,
        self.mode_var,
        self.vehicle_model_var,
        self.vehicle_custom_var,
        self.plate_custom_var,
        self.character_model_var,
        self.character_custom_var,
        self.device_var,
        self.conf_var,
        self.plate_dataset_run_var,
        self.plate_dataset_images_var,
        self.plate_train_pct,
        self.plate_val_pct,
        self.manual_xml_template_var,
        self.manual_vehicle_assist_var,
        self.workflow_route_var,
        self.manual_entry_mode_var,
        self.auto_vehicle_choice_var,
        self.workflow_step_var,
        self.free_mode_screen_var,
    )
    for var in observed_vars:
        try:
            var.trace_add("write", self._on_free_mode_session_var_changed)
        except Exception:
            pass


def _on_free_mode_session_var_changed(self, *_args):
    # Zwykłe pisanie w polach Z2 nie powinno przepisywać ciężkich
    # elementów stanu ani odpalać I/O po każdym znaku.
    self._queue_free_mode_session_save(
        include_preview_approved=False,
        delay_ms=1200,
    )


def _queue_free_mode_session_save(self, *, include_preview_approved: bool = True, delay_ms: int = 350):
    if self._free_mode_session_restore_in_progress or self._campaign_project_restore_in_progress:
        return

    if self._is_free_mode_session_context():
        if not SESSION:
            return
    else:
        try:
            from ..campaign_manager import CAMPAIGN
            if not CAMPAIGN.get_active_project_name():
                return
        except Exception:
            return

    pending = getattr(self, "_free_mode_session_save_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    include_preview_approved = bool(include_preview_approved)
    previous_mode = bool(getattr(self, "_pending_session_save_include_preview_approved", False))
    self._pending_session_save_include_preview_approved = bool(previous_mode or include_preview_approved)

    try:
        self._free_mode_session_save_after_id = self.frame.after(
            max(200, int(delay_ms or 350)),
            self.flush_free_mode_session_state,
        )
    except Exception:
        self.flush_free_mode_session_state()


def _mark_preview_user_interaction(self, *, quiet_ms: int = 1200) -> None:
    try:
        quiet_until = time.perf_counter() + (max(120, int(quiet_ms or 1200)) / 1000.0)
    except Exception:
        quiet_until = time.perf_counter() + 1.2
    try:
        self._preview_user_interaction_quiet_until = max(
            float(getattr(self, "_preview_user_interaction_quiet_until", 0.0) or 0.0),
            float(quiet_until),
        )
    except Exception:
        self._preview_user_interaction_quiet_until = float(quiet_until)


def _preview_user_interaction_quiet_remaining_ms(self, *, padding_ms: int = 180) -> int:
    try:
        quiet_until = float(getattr(self, "_preview_user_interaction_quiet_until", 0.0) or 0.0)
    except Exception:
        quiet_until = 0.0
    if quiet_until <= 0.0:
        return 0
    try:
        remaining = int(max(0.0, quiet_until - time.perf_counter()) * 1000.0)
    except Exception:
        remaining = 0
    if remaining <= 0:
        return 0
    return max(80, remaining + max(0, int(padding_ms or 0)))


def _cancel_campaign_char_effective_source_refresh(self) -> None:
    pending = getattr(self, "_campaign_char_effective_refresh_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._campaign_char_effective_refresh_after_id = None


def _campaign_char_effective_source_refresh_needed(self) -> bool:
    if self._is_free_mode_session_context():
        return False
    try:
        from ..campaign_manager import CAMPAIGN

        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    except Exception:
        return False
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_edge_key = str(graph_context.get("graph_edge_key") or "").strip()
        graph_gate_id = campaign_gate_id_for_edge(
            graph_edge_key,
            graph_context.get("graph_gate_id"),
        )
        repair_origin_gate_id = campaign_gate_id_for_edge(
            graph_context.get("repair_origin_edge_key"),
            graph_context.get("repair_origin_gate_id")
            or graph_context.get("source_graph_gate_id"),
        )
        if graph_gate_id == "T03" or graph_edge_key == "e2_to_e3":
            return False
        if graph_gate_id == "T04" and repair_origin_gate_id == "T06":
            return False
    except Exception:
        pass
    return bool((iteration_target == "char" and current_step >= 3) or current_step >= 3)


def _schedule_campaign_char_effective_source_refresh(self, *, delay_ms: int = 1800) -> None:
    if not self._campaign_char_effective_source_refresh_needed():
        return

    self._cancel_campaign_char_effective_source_refresh()

    def _run() -> None:
        self._campaign_char_effective_refresh_after_id = None
        quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=350)
        if quiet_remaining > 0:
            self._schedule_campaign_char_effective_source_refresh(delay_ms=quiet_remaining)
            return
        self._refresh_campaign_char_effective_source_after_approval()

    try:
        self._campaign_char_effective_refresh_after_id = self.frame.after(max(0, int(delay_ms)), _run)
    except Exception:
        self._campaign_char_effective_refresh_after_id = None
        _run()


def _cancel_preview_approved_persist(self) -> None:
    pending = getattr(self, "_preview_approved_persist_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_approved_persist_after_id = None


def _schedule_preview_approved_persist(self, *, delay_ms: int = 6500) -> None:
    self._cancel_preview_approved_persist()

    def _run() -> None:
        self._preview_approved_persist_after_id = None
        quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=250)
        if quiet_remaining > 0:
            self._schedule_preview_approved_persist(delay_ms=quiet_remaining)
            return
        self._persist_preview_approved_filenames()

    try:
        self._preview_approved_persist_after_id = self.frame.after(max(0, int(delay_ms)), _run)
    except Exception:
        self._preview_approved_persist_after_id = None
        _run()


def _flush_preview_approved_persist(self) -> None:
    pending = getattr(self, "_preview_approved_persist_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
        self._preview_approved_persist_after_id = None
        self._persist_preview_approved_filenames()


def _cancel_preview_approval_followup_refresh(self) -> None:
    pending = getattr(self, "_preview_approval_followup_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_approval_followup_after_id = None


def _schedule_preview_approval_followup_refresh(self, *, delay_ms: int = 650) -> None:
    self._cancel_preview_approval_followup_refresh()

    def _run() -> None:
        self._preview_approval_followup_after_id = None
        quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=350)
        if quiet_remaining > 0:
            self._schedule_preview_approval_followup_refresh(delay_ms=quiet_remaining)
            return
        try:
            active_run_dir = self._get_active_annotation_run_dir(require_xml=True)
            if active_run_dir is not None:
                self._load_plate_dataset_context_from_run(active_run_dir, force_images_update=True)
            self._refresh_plate_dataset_export_sources()
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
        except Exception:
            try:
                self._refresh_step2_action_states()
            except Exception:
                pass

    try:
        self._preview_approval_followup_after_id = self.frame.after(max(0, int(delay_ms)), _run)
    except Exception:
        self._preview_approval_followup_after_id = None
        _run()


def _refresh_campaign_char_effective_source_after_approval(self) -> None:
    if not self._campaign_char_effective_source_refresh_needed():
        return

    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return

    try:
        refreshed_char_source = dict(self._build_campaign_char_effective_source() or {})
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć char_effective_source po zmianie OK w Z2: {e}")
        refreshed_char_source = {}

    try:
        self._append_z2_trace(
            "approved-mark-char-effective-refresh",
            (
                f"entries={int(refreshed_char_source.get('entries_count', 0) or 0)} "
                f"images={int(refreshed_char_source.get('images_with_plates', 0) or 0)} "
                f"plates={int(refreshed_char_source.get('total_plates', 0) or 0)}"
            ),
        )
    except Exception:
        pass

    try:
        refreshed_run_dir = self._resolve_existing_run_dir(refreshed_char_source.get("run_dir"))
    except Exception:
        refreshed_run_dir = None
    try:
        refreshed_xml_path = self._path_value_to_path(refreshed_char_source.get("xml_path"))
    except Exception:
        refreshed_xml_path = None
    try:
        refreshed_images_dir = self._resolve_existing_dir(refreshed_char_source.get("images_dir"))
    except Exception:
        refreshed_images_dir = None

    try:
        CAMPAIGN.set_step3_needs_rework()
        CAMPAIGN.set_step3_stage1_done(False)
        CAMPAIGN.set_step3_stage2_done(False)
        CAMPAIGN.set_step3_substep(1)
        CAMPAIGN.set_step3_extract_state(
            entry_mode="continue",
            workflow_step="start",
            annotation_run_dir=(str(refreshed_run_dir) if refreshed_run_dir is not None else ""),
            xml_path=(str(refreshed_xml_path) if refreshed_xml_path is not None else ""),
            images_dir=(str(refreshed_images_dir) if refreshed_images_dir is not None else ""),
        )
    except Exception as e:
        logger.debug(f"Nie udało się zresetować kroku E3 po zmianie OK w Z2: {e}")

    try:
        campaign_tab = self.app.tabs.get("campaign") if getattr(self.app, "tabs", None) else None
        if campaign_tab is not None:
            campaign_tab._clear_dashboard_perf_cache()
            campaign_tab._refresh_dashboard()
        self.app.update_campaign_tab_access()
    except Exception:
        pass


def _cancel_preview_resume_persist(self) -> None:
    pending = getattr(self, "_preview_resume_persist_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_resume_persist_after_id = None


def _schedule_preview_resume_persist(
    self,
    *,
    include_preview_approved: bool = False,
    delay_ms: int = 120,
) -> None:
    self._cancel_preview_resume_persist()

    def _flush():
        self._preview_resume_persist_after_id = None
        quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=220)
        if quiet_remaining > 0:
            self._schedule_preview_resume_persist(
                include_preview_approved=include_preview_approved,
                delay_ms=quiet_remaining,
            )
            return
        try:
            self._remember_annotation_run_resume_state()
        except Exception:
            pass
        if self._is_free_mode_session_context() or bool(include_preview_approved):
            self._queue_free_mode_session_save(include_preview_approved=include_preview_approved)

    try:
        self._preview_resume_persist_after_id = self.frame.after(max(0, int(delay_ms)), _flush)
    except Exception:
        self._preview_resume_persist_after_id = None
        _flush()


def _invalidate_preview_runtime_caches(self) -> None:
    self._preview_any_auto_in_run_cache = None
    self._preview_list_summary_cache = None
    self._preview_list_render_state_cache = None
    self._preview_annotation_quality_summary_cache = None
    self._current_preview_plate_count_cache = None
    self._preview_render_image_cache = {}


def flush_free_mode_session_state(self):
    flush_started_at = time.perf_counter()
    pending = getattr(self, "_free_mode_session_save_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._free_mode_session_save_after_id = None
    quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=350)
    if quiet_remaining > 0:
        try:
            self._free_mode_session_save_after_id = self.frame.after(
                int(quiet_remaining),
                self.flush_free_mode_session_state,
            )
        except Exception:
            self._free_mode_session_save_after_id = None
        return
    include_preview_approved = bool(
        getattr(self, "_pending_session_save_include_preview_approved", True)
    )
    self._pending_session_save_include_preview_approved = False

    if self._free_mode_session_restore_in_progress or self._campaign_project_restore_in_progress:
        return

    if self._is_free_mode_session_context():
        if not SESSION:
            return

        try:
            snapshot = self._collect_free_mode_session_snapshot()
            snapshot = self._apply_annotation_auto_modal_defaults_to_snapshot(snapshot)
            snapshot = self._strip_persisted_annotation_miniflow_snapshot(snapshot)
            try:
                signature = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                signature = ""
            if signature and signature == str(getattr(self, "_last_free_mode_session_snapshot_signature", "") or ""):
                return
            SESSION.set("annotation", "images_dir", snapshot["input_dir"])
            for key, value in snapshot.items():
                SESSION.set("annotation", key, value)
            SESSION.save_session()
            self._last_free_mode_session_snapshot_signature = signature
        except Exception as e:
            logger.debug(f"Nie udalo sie zapisac stanu Z2: {e}")
        return

    self._save_campaign_project_snapshot_with_options(
        include_preview_approved=include_preview_approved,
    )
    elapsed_ms = max(0.0, (time.perf_counter() - flush_started_at) * 1000.0)
    if elapsed_ms >= 180.0:
        try:
            logger.info(
                "[Z2 PERF] session_flush total=%.0fms include_ok=%s campaign=%s",
                elapsed_ms,
                int(bool(include_preview_approved)),
                int(not self._is_free_mode_session_context()),
            )
        except Exception:
            pass


def _apply_free_mode_session_snapshot(self, *args, **kwargs):
    return z2_workflow_methods._apply_free_mode_session_snapshot(self, *args, **kwargs)
