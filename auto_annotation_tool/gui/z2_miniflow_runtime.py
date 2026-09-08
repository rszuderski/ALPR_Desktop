#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 miniflow and route runtime extracted from tab_annotation.py."""

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

def _get_workflow_route(self) -> str:
    if self._is_campaign_step2_context():
        return self._normalize_workflow_route_value(
            getattr(self, "_campaign_step2_workflow_route", "")
        )
    return self._normalize_workflow_route_value(self.workflow_route_var.get())


def _get_manual_entry_mode(self) -> str:
    if self._is_campaign_step2_context():
        return self._normalize_manual_entry_mode(
            getattr(self, "_campaign_step2_manual_entry_mode", "continue")
        )
    return self._normalize_manual_entry_mode(self.manual_entry_mode_var.get())


def _get_auto_vehicle_choice(self) -> str:
    if self._is_campaign_step2_context():
        return self._normalize_auto_vehicle_choice(
            getattr(self, "_campaign_step2_auto_vehicle_choice", "skip")
        )
    return self._normalize_auto_vehicle_choice(self.auto_vehicle_choice_var.get())


def _get_workflow_step(self) -> str:
    if self._is_campaign_step2_context():
        return self._normalize_workflow_step_value(
            getattr(self, "_campaign_step2_workflow_step", "")
        )
    return self._normalize_workflow_step_value(self.workflow_step_var.get())


def _get_free_mode_screen(self) -> str:
    if self._is_campaign_step2_context():
        return self._normalize_free_mode_screen_value(
            getattr(self, "_campaign_step2_screen", "")
        )
    return self._normalize_free_mode_screen_value(self.free_mode_screen_var.get())


def _set_workflow_route_state(self, value: str, *, campaign_context: bool | None = None) -> None:
    normalized = self._normalize_workflow_route_value(value)
    use_campaign = self._is_campaign_step2_context() if campaign_context is None else bool(campaign_context)
    if use_campaign:
        self._campaign_step2_workflow_route = normalized
    else:
        self.workflow_route_var.set(normalized)


def _set_workflow_step_state(self, value: str, *, campaign_context: bool | None = None) -> None:
    normalized = self._normalize_workflow_step_value(value)
    use_campaign = self._is_campaign_step2_context() if campaign_context is None else bool(campaign_context)
    if use_campaign:
        self._campaign_step2_workflow_step = normalized
    else:
        self.workflow_step_var.set(normalized)


def _set_screen_state(self, value: str, *, campaign_context: bool | None = None) -> None:
    normalized = self._normalize_free_mode_screen_value(value)
    use_campaign = self._is_campaign_step2_context() if campaign_context is None else bool(campaign_context)
    if use_campaign:
        self._campaign_step2_screen = normalized
    else:
        self.free_mode_screen_var.set(normalized)


def _set_manual_entry_mode_state(self, value: str, *, campaign_context: bool | None = None) -> None:
    normalized = self._normalize_manual_entry_mode(value)
    use_campaign = self._is_campaign_step2_context() if campaign_context is None else bool(campaign_context)
    if use_campaign:
        self._campaign_step2_manual_entry_mode = normalized
    else:
        self.manual_entry_mode_var.set(normalized)


def _set_auto_vehicle_choice_state(self, value: str, *, campaign_context: bool | None = None) -> None:
    normalized = self._normalize_auto_vehicle_choice(value)
    use_campaign = self._is_campaign_step2_context() if campaign_context is None else bool(campaign_context)
    if use_campaign:
        self._campaign_step2_auto_vehicle_choice = normalized
    else:
        self.auto_vehicle_choice_var.set(normalized)


def _reset_campaign_step2_runtime_state(self) -> None:
    self._campaign_step2_workflow_route = ""
    self._campaign_step2_workflow_step = ""
    self._campaign_step2_screen = ""
    self._campaign_step2_manual_entry_mode = "continue"
    self._campaign_step2_auto_vehicle_choice = "skip"


def _is_z2_auto_flow_completed(
    self,
    *,
    route: str | None = None,
    has_existing_run: bool | None = None,
    free_mode_screen: str | None = None,
) -> bool:
    normalized_route = self._normalize_workflow_route_value(route or self._get_z2_thematic_route())
    if normalized_route != "auto":
        return False

    try:
        if self._is_campaign_char_repair_return_mode() or self._is_campaign_plate_step4_repair_return_mode():
            return False
    except Exception:
        pass

    run_ready = (
        bool(has_existing_run)
        if has_existing_run is not None
        else (self._get_preferred_annotation_run_dir(require_xml=True) is not None)
    )
    if not run_ready or self.is_processing:
        return False

    if not self._is_free_mode_session_context():
        return bool(
            self._normalize_workflow_route_value(
                getattr(self, "_last_completed_workflow_route", "")
            ) == "auto"
        )

    screen_value = self._normalize_free_mode_screen_value(
        self._get_free_mode_screen() if free_mode_screen is None else free_mode_screen
    )
    manual_origin_auto = (
        self._normalize_workflow_route_value(getattr(self, "_manual_review_origin_route", "")) == "auto"
    )

    return bool(
        self._normalize_workflow_route_value(getattr(self, "_last_completed_workflow_route", "")) == "auto"
        or manual_origin_auto
        or screen_value in {"auto_summary", "manual_review", "export"}
    )


def _repair_restored_z2_workflow_completion_state(self, state_value: str | None = None) -> None:
    normalized_state = self._normalize_workflow_route_value(
        state_value if state_value is not None else getattr(self, "_last_completed_workflow_route", "")
    )
    if normalized_state in {"auto", "manual"}:
        self._last_completed_workflow_route = normalized_state
        return

    thematic_route = self._get_z2_thematic_route()
    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
    screen_value = self._normalize_free_mode_screen_value(self.free_mode_screen_var.get())

    if self._is_z2_auto_flow_completed(
        route=thematic_route,
        has_existing_run=has_existing_run,
        free_mode_screen=screen_value,
    ):
        self._last_completed_workflow_route = "auto"
        return

    if thematic_route == "manual" and has_existing_run and (
        bool(getattr(self, "_manual_review_active", False))
        or screen_value == "export"
    ):
        self._last_completed_workflow_route = "manual"
        return

    self._last_completed_workflow_route = ""


def _coerce_free_mode_screen(self, screen: str | None = None) -> str:
    if not self._is_free_mode_session_context():
        return ""

    route = self._get_workflow_route()
    requested = self._normalize_free_mode_screen_value(
        self._get_free_mode_screen() if screen is None else screen
    )
    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
    manual_review_active = bool(self._manual_review_active and has_existing_run)
    export_active = bool(self._manual_review_export_ready or self._dataset_export_completed)
    auto_completed = self._is_z2_auto_flow_completed(
        route=route,
        has_existing_run=has_existing_run,
        free_mode_screen=requested,
    )

    if not route:
        return "route_choice"

    if requested == "route_choice":
        return "workflow"

    if requested == "workflow":
        if export_active:
            return "export"
        if manual_review_active:
            return "manual_review"
        return "workflow"

    if requested == "auto_summary":
        if export_active:
            return "export"
        if manual_review_active:
            return "manual_review"
        return "auto_summary" if (route == "auto" and has_existing_run) else "workflow"

    if requested == "manual_review":
        if export_active:
            return "export"
        if manual_review_active:
            return "manual_review"
        return "auto_summary" if auto_completed else "workflow"

    if requested == "export":
        if export_active:
            return "export"
        if manual_review_active:
            return "manual_review"
        return "auto_summary" if auto_completed else "workflow"

    if export_active:
        return "export"
    if manual_review_active:
        return "manual_review"
    if auto_completed:
        return "auto_summary"
    return "workflow"


def _get_workflow_progress_display(self) -> tuple[int, int]:
    route = self._get_workflow_route()
    current_step = self._coerce_workflow_step()
    steps = self._get_current_workflow_steps()
    current_index = steps.index(current_step) + 1 if current_step in steps else 0
    total_steps = len(steps)

    if not self._is_free_mode_session_context():
        if bool(getattr(self, "_manual_review_active", False)):
            return 0, 0
        return current_index, total_steps

    screen = self._coerce_free_mode_screen()
    manual_entry_mode = self._get_manual_entry_mode()
    auto_setup_pending = bool(
        self._is_free_mode_session_context()
        and route == "auto"
        and getattr(self, "_auto_route_settings_pending", False)
    )

    if route == "auto":
        total_steps = 5
        if screen == "export":
            return 5, total_steps
        if screen in {"auto_summary", "manual_review"}:
            return 4, total_steps
        if current_step in {"auto_input", "auto_start"}:
            return (2 if auto_setup_pending else 3), total_steps
        return (2 if current_step in {"auto_plate_model", "auto_conf", "auto_vehicle_choice", "auto_vehicle_model"} else 1), total_steps

    if route == "manual":
        if manual_entry_mode == "continue":
            total_steps = 4
            if screen == "manual_review":
                return 3, total_steps
            if screen == "export":
                return 4, total_steps
            if current_step == "manual_entry":
                return 1, total_steps
            if current_step == "manual_history":
                return 2, total_steps
        elif manual_entry_mode == "import":
            total_steps = 3
            if screen == "manual_review":
                return 2, total_steps
            if screen == "export":
                return 3, total_steps
            if current_step == "manual_entry":
                return 1, total_steps
        elif manual_entry_mode == "new":
            total_steps = 5
            if screen == "manual_review":
                return 4, total_steps
            if screen == "export":
                return 5, total_steps
            if current_step == "manual_entry":
                return 1, total_steps
            if current_step == "manual_input":
                return 2, total_steps
            if current_step == "manual_start":
                return 3, total_steps
    return current_index, total_steps


def _get_z2_thematic_route(self) -> str:
    route = self._get_workflow_route()
    manual_origin = self._normalize_workflow_route_value(
        getattr(self, "_manual_review_origin_route", "")
    )
    if route == "manual" and manual_origin == "auto":
        return "auto"
    return route


def _get_z2_miniflow_current_key(
    self,
    *,
    route: str,
    current_step: str,
    free_mode_context: bool,
    free_mode_screen: str,
    manual_review_active: bool,
    export_active: bool,
    auto_completed: bool,
) -> str:
    route_value = self._normalize_workflow_route_value(route)
    step_value = self._normalize_workflow_step_value(current_step)
    screen_value = self._normalize_free_mode_screen_value(free_mode_screen)

    if route_value == "auto":
        if export_active or screen_value == "export":
            return "export"
        if manual_review_active or screen_value in {"auto_summary", "manual_review"}:
            return "review"
        if free_mode_context and screen_value == "workflow":
            if step_value == "auto_input":
                return "input"
            if step_value == "auto_start":
                return "annotate"
            if step_value in {
                "auto_plate_model",
                "auto_conf",
                "auto_vehicle_choice",
                "auto_vehicle_model",
            }:
                return "annotate"
            return "route"
        if auto_completed:
            return "review"
        if step_value == "auto_input":
            return "input"
        if step_value == "auto_start":
            return "annotate"
        if step_value in {
            "auto_plate_model",
            "auto_conf",
            "auto_vehicle_choice",
            "auto_vehicle_model",
        }:
            return "annotate"
        return "route"

    if route_value == "manual":
        if export_active or screen_value == "export":
            return "export"
        if manual_review_active or screen_value == "manual_review":
            return "review"
        if step_value in {"manual_entry", "manual_history", "manual_input", "manual_start"}:
            return "input"
        return "route"

    return "route"


def _get_z2_miniflow_progress_items(self) -> tuple[list[dict], bool]:
    if not self._is_free_mode_session_context():
        return [], False

    route = self._get_z2_thematic_route()
    if route not in {"auto", "manual"}:
        return [], False

    free_mode_context = bool(self._is_free_mode_session_context())
    free_mode_screen = self._coerce_free_mode_screen() if free_mode_context else ""
    route_label = "Tor"
    if not self._is_free_mode_session_context():
        target = ""
        try:
            from ..campaign_manager import CAMPAIGN
            target = self._normalize_campaign_iteration_target_value(
                CAMPAIGN.get_iteration_target()
            )
        except Exception:
            target = ""
        if target == "plate":
            route_label = "Tor: Tablice"
        elif target == "char":
            route_label = "Tor: Znaki"

    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
    manual_review_active = bool(self._manual_review_active and has_existing_run)
    export_active = bool(self._manual_review_export_ready or self._dataset_export_completed)
    auto_completed = self._is_z2_auto_flow_completed(
        route=route,
        has_existing_run=has_existing_run,
    )
    current_step = self._coerce_workflow_step()

    if route == "auto":
        items = [
            {"key": "route", "label": route_label},
            {"key": "input", "label": "Wejście"},
            {"key": "annotate", "label": "Autoanotacja"},
            {"key": "review", "label": "Korekta"},
            {"key": "export", "label": "Eksport"},
        ]
    else:
        items = [
            {"key": "route", "label": route_label},
            {"key": "input", "label": "Wejście"},
            {"key": "review", "label": "Korekta"},
            {"key": "export", "label": "Eksport"},
        ]
    current_key = self._get_z2_miniflow_current_key(
        route=route,
        current_step=current_step,
        free_mode_context=free_mode_context,
        free_mode_screen=free_mode_screen,
        manual_review_active=manual_review_active,
        export_active=export_active,
        auto_completed=auto_completed,
    )

    current_idx = 0
    for idx, item in enumerate(items):
        if str(item.get("key") or "") == current_key:
            current_idx = idx
            break

    all_done = bool(self._dataset_export_completed)
    result = []
    for idx, item in enumerate(items):
        state = "upcoming"
        if all_done or idx < current_idx:
            state = "done"
        elif idx == current_idx:
            state = "current"
        result.append(
            {
                "key": str(item.get("key") or ""),
                "label": str(item.get("label") or ""),
                "state": state,
            }
        )

    return result, all_done


def _refresh_z2_miniflow_progress(self, *args, **kwargs):
    return z2_workflow_methods._refresh_z2_miniflow_progress(self, *args, **kwargs)


def _strip_workflow_step_suffix(text: str | None) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    return re.sub(r"\s+\|\s*Krok\s+\d+\s+z\s+\d+\s*$", "", raw_text).strip()


def _strip_workflow_heading_prefix(text: str | None) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    return re.sub(r"^\d+(?:\.[a-z])?\.?\s+", "", raw_text, flags=re.IGNORECASE).strip()


def _format_z2_thematic_title(self, text: str | None, prefix: str | None = None) -> str:
    base_text = self._strip_workflow_heading_prefix(self._strip_workflow_step_suffix(text))
    clean_prefix = str(prefix or "").strip()
    if not clean_prefix:
        return base_text
    return f"{clean_prefix} {base_text}".strip()


def _get_z2_thematic_title_prefixes(self) -> dict[str, str]:
    if not self._is_free_mode_session_context():
        return {}

    route = self._get_z2_thematic_route()
    if route not in {"auto", "manual"}:
        return {}

    current_step = self._coerce_workflow_step()
    free_mode_context = bool(self._is_free_mode_session_context())
    free_mode_screen = self._coerce_free_mode_screen() if free_mode_context else ""
    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
    manual_review_active = bool(self._manual_review_active and has_existing_run)
    export_active = bool(self._manual_review_export_ready or self._dataset_export_completed)
    auto_completed = self._is_z2_auto_flow_completed(
        route=route,
        has_existing_run=has_existing_run,
        free_mode_screen=free_mode_screen,
    )

    if route == "auto":
        run_prefix = "1."
        if free_mode_context and free_mode_screen == "workflow":
            if current_step == "auto_input":
                run_prefix = "2."
            elif current_step == "auto_start":
                run_prefix = "3."
            elif current_step in {"auto_plate_model", "auto_conf", "auto_vehicle_choice", "auto_vehicle_model"}:
                run_prefix = "3."
        else:
            if export_active or free_mode_screen == "export":
                run_prefix = "5."
            elif manual_review_active or auto_completed or free_mode_screen in {"auto_summary", "manual_review"}:
                run_prefix = "4."
            elif current_step == "auto_input":
                run_prefix = "2."
            elif current_step == "auto_start":
                run_prefix = "3."
            elif current_step in {"auto_plate_model", "auto_conf", "auto_vehicle_choice", "auto_vehicle_model"}:
                run_prefix = "3."

        return {
            "workflow_entry": "1.",
            "run": run_prefix,
            "auto_plate_model": "3.a",
            "workflow_conf": "3.b",
            "auto_vehicle_choice": "3.c",
            "workflow_vehicle_model": "3.d",
            "workflow_input": "2.",
            "workflow_start": "3.",
            "followup": "4.",
            "manual_stage": "4.",
            "export": "5.",
            "split": "5.a",
        }

    manual_entry_mode = self._get_manual_entry_mode()
    run_prefix = "1."
    if export_active or free_mode_screen == "export":
        run_prefix = "4."
    elif manual_review_active or free_mode_screen == "manual_review":
        run_prefix = "3."
    elif current_step in {"manual_entry", "manual_history", "manual_input", "manual_conf", "manual_vehicle_model", "manual_start"}:
        run_prefix = "2."

    input_prefix = "2.c" if manual_entry_mode == "continue" else "2.b"
    conf_prefix = "2.d" if manual_entry_mode == "continue" else "2.c"
    vehicle_prefix = "2.e" if manual_entry_mode == "continue" else "2.d"
    start_prefix = "2.f" if manual_entry_mode == "continue" else "2.e"

    return {
        "workflow_entry": "1.",
        "run": run_prefix,
        "manual_entry": "2.a",
        "manual_history": ("2.b" if manual_entry_mode == "continue" else "2.a"),
        "workflow_input": input_prefix,
        "workflow_conf": conf_prefix,
        "workflow_vehicle_model": vehicle_prefix,
        "workflow_start": start_prefix,
        "followup": "3.",
        "manual_stage": "3.",
        "export": "4.",
        "split": "4.a",
    }


def _resolve_manual_review_history_created_at(self, entry: dict, safe_run_dir: Path) -> str:
    created_at = str(entry.get("created_at") or "").strip()
    if created_at:
        return created_at
    manifest = self._load_annotation_run_manifest(safe_run_dir)
    for candidate in (
        str(manifest.get("generated_at") or "").strip(),
        str(manifest.get("completed_at") or "").strip(),
        str(manifest.get("created_at") or "").strip(),
    ):
        if candidate:
            return candidate

    try:
        return datetime.datetime.fromtimestamp(safe_run_dir.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return ""


def _normalize_manual_review_history_entries(self, entries) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()

    for entry in entries or []:
        if isinstance(entry, str):
            entry = {"run_dir": entry}
        if not isinstance(entry, dict):
            continue

        safe_run_dir = self._resolve_safe_annotation_run_dir(entry.get("run_dir"), require_xml=True)
        if safe_run_dir is None:
            continue

        run_dir_text = str(safe_run_dir)
        if run_dir_text in seen:
            continue

        seen.add(run_dir_text)
        normalized.append(
            {
                "run_dir": run_dir_text,
                "created_at": self._resolve_manual_review_history_created_at(entry, safe_run_dir),
                "source": str(entry.get("source") or "").strip(),
            }
        )

    normalized.sort(
        key=lambda item: (
            str(item.get("created_at") or "").strip(),
            str(item.get("run_dir") or "").strip().lower(),
        ),
        reverse=True,
    )
    return normalized[:20]


def _get_manual_review_history_display_entries(self) -> list[dict]:
    return dispatch_get_manual_review_history_display_entries(self)


def _format_manual_review_history_label(self, entry: dict) -> str:
    run_dir = str(entry.get("run_dir") or "").strip()
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return ""

    created_at = str(entry.get("created_at") or "").strip()
    label = safe_run_dir.name
    if created_at:
        try:
            created_dt = datetime.datetime.fromisoformat(created_at)
            created_text = created_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            created_text = created_at.replace("T", " ").strip()
        if created_text:
            label = f"{label} | {created_text}"
    return label


def _refresh_manual_review_history_ui(self):
    dispatch_refresh_manual_review_history_ui(self)


def _remember_manual_review_run(self, run_dir: Path | str | None, *, source: str = "manual", created_at: str | None = None):
    dispatch_remember_manual_review_run(self, run_dir, source=source, created_at=created_at)


def _get_selected_manual_review_history_run_dir(self) -> Path | None:
    selected_label = str(self.manual_history_run_var.get() or "").strip()
    selected_run = self._manual_review_history_label_map.get(selected_label)
    if selected_run:
        return self._resolve_safe_annotation_run_dir(selected_run, require_xml=True)
    return None


def _open_selected_manual_review_history_run(self):
    selected_run = self._get_selected_manual_review_history_run_dir()
    if selected_run is None:
        return
    self._open_existing_run_for_manual_review(
        run_dir=selected_run,
        allow_fallback=False,
        show_dialog=False,
        entry_mode="continue",
    )


def _on_manual_history_selection_changed(self, event=None):
    selected = bool(self._get_selected_manual_review_history_run_dir() is not None)
    try:
        self.manual_history_open_btn.configure(
            state=(tk.NORMAL if selected else tk.DISABLED)
        )
    except Exception:
        pass
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()
    self._queue_free_mode_session_save()


def _get_auto_workflow_steps(self) -> list[str]:
    return [
        "auto_input",
        "auto_start",
    ]


def _get_manual_workflow_steps(self) -> list[str]:
    steps = ["manual_entry"]
    manual_entry_mode = self._get_manual_entry_mode()
    if manual_entry_mode == "new":
        steps.extend(["manual_input", "manual_start"])
    elif manual_entry_mode == "continue":
        steps.append("manual_history")
    return steps


def _get_current_workflow_steps(self) -> list[str]:
    route = self._get_workflow_route()
    if route == "auto":
        return self._get_auto_workflow_steps()
    if route == "manual":
        return self._get_manual_workflow_steps()
    return []


def _get_default_workflow_step(self) -> str:
    route = self._get_workflow_route()
    if route == "auto":
        if self._is_free_mode_session_context():
            return "auto_input"
        input_ready = self._annotation_input_dir_ready()
        return "auto_start" if input_ready else "auto_input"
    if route == "manual":
        return "manual_entry"
    return ""


def _coerce_workflow_step(self, step: str | None = None) -> str:
    route = self._get_workflow_route()
    current = self._normalize_workflow_step_value(
        self._get_workflow_step() if step is None else step
    )
    steps = self._get_current_workflow_steps()
    if not route or not steps:
        return ""

    if not current:
        return self._get_default_workflow_step()

    if current in {"auto_plate_model", "auto_conf", "auto_vehicle_choice", "auto_vehicle_model"}:
        if route != "auto":
            return self._get_default_workflow_step()
        return "auto_start" if "auto_start" in steps else self._get_default_workflow_step()
    if current in {"auto_input", "auto_start"}:
        if route != "auto":
            return self._get_default_workflow_step()
        if (
            self._is_free_mode_session_context()
            and bool(getattr(self, "_auto_route_settings_pending", False))
        ):
            input_ready = self._annotation_input_dir_ready()
            if current == "auto_start" and not input_ready:
                return "auto_input"
        return current
    if current == "manual_entry":
        return current if route == "manual" else self._get_default_workflow_step()
    if current == "manual_history":
        if route != "manual":
            return self._get_default_workflow_step()
        return current if self._get_manual_entry_mode() == "continue" else "manual_entry"
    if current in {"manual_conf", "manual_vehicle_model"}:
        if route != "manual":
            return self._get_default_workflow_step()
        return "manual_input" if self._get_manual_entry_mode() == "new" else "manual_entry"
    if current == "manual_input":
        if route != "manual":
            return self._get_default_workflow_step()
        return current if self._get_manual_entry_mode() == "new" else "manual_entry"
    if current == "manual_start":
        if route != "manual":
            return self._get_default_workflow_step()
        return current if self._get_manual_entry_mode() == "new" else "manual_entry"

    return self._get_default_workflow_step()


def _set_workflow_step(
    self,
    step: str,
    *,
    refresh: bool = True,
    save: bool = True,
    refresh_detection_ui: bool = False,
):
    normalized = self._coerce_workflow_step(step)
    self._set_workflow_step_state(normalized)
    if refresh:
        self._refresh_left_panel_route_copy()
        if refresh_detection_ui:
            self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()
        anchor = {
            "auto_plate_model": getattr(self, "auto_plate_model_section", None),
            "auto_conf": getattr(self, "workflow_conf_section", None),
            "auto_vehicle_choice": getattr(self, "auto_vehicle_choice_section", None),
            "auto_vehicle_model": getattr(self, "workflow_vehicle_model_section", None),
            "auto_input": getattr(self, "workflow_input_section", None),
            "auto_start": getattr(self, "workflow_start_section", None),
            "manual_entry": getattr(self, "manual_entry_section", None),
            "manual_history": getattr(self, "manual_history_section", None),
            "manual_conf": getattr(self, "workflow_conf_section", None),
            "manual_vehicle_model": getattr(self, "workflow_vehicle_model_section", None),
            "manual_input": getattr(self, "workflow_input_section", None),
            "manual_start": getattr(self, "workflow_start_section", None),
        }.get(normalized)
        self._schedule_left_panel_scroll_to_widget(anchor or getattr(self, "actions_section", None))
    if save:
        self._queue_free_mode_session_save()


def _clear_free_mode_route_selection(self):
    self._reset_annotation_miniflow_state(clear_input_dir=True, clear_plate_model=True)
    self._refresh_left_panel_route_copy()
    self._refresh_detection_configuration_ui()
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()
    self._scroll_left_panel_to_widget(
        getattr(self, "workflow_entry_shell", None)
        or getattr(self, "workflow_entry_section", None)
    )
    self._queue_free_mode_session_save()


def _exit_manual_review_stage(self):
    self._manual_review_export_ready = False
    self._dataset_export_completed = False

    if self._normalize_workflow_route_value(getattr(self, "_manual_review_origin_route", "")) == "auto":
        self._manual_review_active = False
        self._manual_review_from_auto = False
        self._manual_review_origin_route = ""
        self.workflow_route_var.set("auto")
        self.workflow_step_var.set("auto_start")
        self.free_mode_screen_var.set("auto_summary")
        self._last_completed_workflow_route = "auto"
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()
        self._scroll_left_panel_to_widget(getattr(self, "followup_section", None) or getattr(self, "actions_section", None))
        self._queue_free_mode_session_save()
        return

    self._manual_review_active = False
    self._manual_review_from_auto = False
    self._manual_review_origin_route = ""
    self.free_mode_screen_var.set("workflow")
    manual_entry_mode = self._get_manual_entry_mode()
    if manual_entry_mode == "new" and self._has_active_manual_template_run():
        self._manual_template_ready_for_review = True
        self._set_workflow_step("manual_start")
    else:
        self._manual_template_ready_for_review = False
        self._set_workflow_step(
            "manual_history" if manual_entry_mode == "continue" else "manual_entry"
        )


def _is_workflow_step_complete(self, step: str | None = None) -> bool:
    current = self._coerce_workflow_step(step)
    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
    input_dir_value = str(self.input_dir_var.get() or "").strip()
    input_ready = bool(input_dir_value and Path(input_dir_value).exists())
    vehicle_model_value = str(self.vehicle_model_var.get() or "").strip()
    vehicle_custom_value = str(self.vehicle_custom_var.get() or "").strip()
    vehicle_model_ready = bool(vehicle_model_value)
    if vehicle_model_ready and vehicle_model_value == "Custom":
        vehicle_model_ready = bool(vehicle_custom_value and Path(vehicle_custom_value).exists())

    if current == "auto_plate_model":
        return bool(str(self.plate_custom_var.get() or "").strip())
    if current == "auto_conf":
        return True
    if current == "auto_vehicle_choice":
        return True
    if current in {"auto_vehicle_model", "manual_vehicle_model"}:
        return vehicle_model_ready
    if current == "auto_input":
        return input_ready
    if current == "manual_entry":
        return True
    if current == "manual_history":
        return self._get_selected_manual_review_history_run_dir() is not None
    if current == "manual_input":
        return bool(input_ready)
    if current == "manual_conf":
        return True
    if current in {"auto_start", "manual_start"}:
        if current == "auto_start":
            if bool(
                self._is_free_mode_session_context()
                and self._get_workflow_route() == "auto"
                and has_existing_run
                and self._is_z2_auto_flow_completed(route="auto", has_existing_run=has_existing_run)
            ):
                return True
            return input_ready
        return bool(
            self._is_free_mode_session_context()
            and self._get_workflow_route() == "manual"
            and self._get_manual_entry_mode() == "new"
            and bool(getattr(self, "_manual_template_ready_for_review", False))
        )
    return False


def _go_to_previous_workflow_step(self):
    if self._is_plate_auto_scope_modal_blocking_actions():
        return
    if self.is_processing:
        return

    if self._is_free_mode_session_context():
        screen = self._coerce_free_mode_screen()
        if screen == "export":
            self._close_export_followup()
            return
        if screen == "manual_review":
            if self._get_workflow_route() == "manual":
                self._exit_manual_review_stage()
                return
            self._exit_manual_review_stage()
            return
        if screen == "auto_summary":
            self.free_mode_screen_var.set("workflow")
            if self._get_workflow_route() == "auto":
                self._restore_plate_model_selection_from_active_run()
                self._set_workflow_step("auto_start")
                return
            self._refresh_left_panel_route_copy()
            self._refresh_detection_configuration_ui()
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            self._scroll_left_panel_to_widget(
                getattr(self, "workflow_start_section", None)
                or getattr(self, "actions_section", None)
            )
            self._queue_free_mode_session_save()
            return

    if self._manual_review_export_ready:
        self._close_export_followup()
        return

    if self._dataset_export_completed:
        return

    if self._manual_review_active:
        self._exit_manual_review_stage()
        return

    current = self._coerce_workflow_step()
    steps = self._get_current_workflow_steps()
    if current not in steps:
        if self._is_free_mode_session_context() and self._get_workflow_route():
            self._confirm_and_return_to_free_mode_route_choice()
        return
    idx = steps.index(current)
    if idx <= 0:
        if self._is_free_mode_session_context() and self._get_workflow_route():
            self._confirm_and_return_to_free_mode_route_choice()
        return
    self._set_workflow_step(steps[idx - 1])


def _close_export_followup(self):
    if self.is_processing:
        return

    self._manual_review_export_ready = False
    self._dataset_export_completed = False
    if self._is_free_mode_session_context():
        has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None
        if self._manual_review_active and has_existing_run:
            self.free_mode_screen_var.set("manual_review")
        elif self._get_workflow_route() == "auto" and has_existing_run:
            self.free_mode_screen_var.set("auto_summary")
        else:
            self.free_mode_screen_var.set("workflow")
    self._refresh_left_panel_route_copy()
    self._refresh_detection_configuration_ui()
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()
    scroll_target = None
    if self._is_free_mode_session_context():
        screen = self._coerce_free_mode_screen()
        if screen == "auto_summary":
            scroll_target = getattr(self, "followup_section", None)
        elif screen == "manual_review":
            scroll_target = (
                getattr(self, "manual_stage_export_box", None)
                or getattr(self, "manual_stage_section", None)
            )
        elif self._get_manual_entry_mode() == "continue":
            scroll_target = getattr(self, "manual_history_section", None)
        else:
            scroll_target = getattr(self, "actions_section", None)
    self._scroll_left_panel_to_widget(
        scroll_target
        or getattr(self, "manual_stage_section", None)
        or getattr(self, "followup_section", None)
    )
    self._queue_free_mode_session_save()


def _resolve_latest_exported_plate_dataset_dir(self) -> Path | None:
    candidates = []
    try:
        training_tab = getattr(getattr(self, "app", None), "tabs", {}).get("training")
    except Exception:
        training_tab = None
    if training_tab is not None:
        try:
            dataset_value = str(getattr(training_tab, "dataset_var", tk.StringVar()).get() or "").strip()
        except Exception:
            dataset_value = ""
        if dataset_value:
            candidates.append(dataset_value)
    try:
        out_value = str(getattr(self, "plate_dataset_out_var", tk.StringVar()).get() or "").strip()
    except Exception:
        out_value = ""
    if out_value:
        candidates.append(out_value)

    for candidate in candidates:
        try:
            path = Path(candidate)
        except Exception:
            continue
        try:
            if path.exists() and path.is_dir():
                return path
        except Exception:
            continue
    return None


def _prompt_post_z2_export_completion_action(self, dataset_path: Path | None = None) -> str:
    parent = self.frame.winfo_toplevel()
    result = {"choice": "stay"}

    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    dialog.transient(parent)
    dialog.title("Co dalej po eksporcie?")
    dialog.resizable(False, False)
    try:
        dialog.configure(bg=self.app.palette.get("panel", "#252526"))
    except Exception:
        pass

    panel_bg = self.app.palette.get("panel", "#252526")
    fg = self.app.palette.get("fg", "#f3f3f3")
    muted = self.app.palette.get("muted", "#c7c7c7")

    shell = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0, padx=18, pady=16)
    shell.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        shell,
        text="Dataset jest gotowy",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI Semibold", 12),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    dataset_line = ""
    if dataset_path is not None:
        try:
            dataset_line = str(dataset_path)
        except Exception:
            dataset_line = ""

    body_lines = [
        "Eksport datasetu YOLO Pose zostal zakonczony pomyslnie.",
        "",
        "Mozesz teraz:",
        "- zamknac mini-flow eksportu i wrocic do Z2,",
        "- albo przejsc od razu do treningu w Z4, gdzie dataset tablic bedzie juz wczytany.",
    ]
    if dataset_line:
        body_lines.extend(["", f"Dataset: {dataset_line}"])

    tk.Label(
        shell,
        text="\n".join(body_lines),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=620,
    ).pack(anchor=tk.W, fill=tk.X, pady=(10, 0))

    buttons = tk.Frame(shell, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def choose(mode: str) -> None:
        result["choice"] = str(mode or "").strip().lower()
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(
        buttons,
        text="Przejdz od razu do treningu",
        style="Accent.TButton",
        command=lambda: choose("training"),
    ).pack(side=tk.RIGHT)

    ttk.Button(
        buttons,
        text="Zamknij mini-flow",
        command=lambda: choose("finish"),
    ).pack(side=tk.RIGHT, padx=(0, 8))

    ttk.Button(
        buttons,
        text="Wroc",
        command=lambda: choose("stay"),
    ).pack(side=tk.LEFT)

    try:
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
    except Exception:
        pass

    dialog.bind("<Escape>", lambda _e: choose("stay"))
    dialog.wait_window()
    return str(result.get("choice") or "stay").strip().lower()


def _open_step4_training_from_z2_export(self, dataset_path: Path | None = None) -> bool:
    try:
        training_tab = getattr(getattr(self, "app", None), "tabs", {}).get("training")
    except Exception:
        training_tab = None

    if training_tab is None:
        try:
            messagebox.showwarning(
                "Brak Z4",
                "Zakladka Z4 nie jest teraz dostepna, wiec nie moge przejsc od razu do treningu.",
                parent=self.frame.winfo_toplevel(),
            )
        except Exception:
            pass
        return False

    target_dataset = dataset_path or self._resolve_latest_exported_plate_dataset_dir()
    if target_dataset is None:
        try:
            messagebox.showwarning(
                "Brak datasetu",
                "Nie znaleziono gotowego datasetu tablic do przekazania do Z4.",
                parent=self.frame.winfo_toplevel(),
            )
        except Exception:
            pass
        return False

    try:
        self.release_gpu_resources_for_training()
    except Exception:
        pass

    try:
        self.app.open_controlled_tab("training")
    except Exception:
        return False

    try:
        self._clear_free_mode_route_selection()
    except Exception as e:
        logger.debug(f"Nie udało się wyczyścić kontekstu Z2 po przejściu do Z4: {e}")

    def _bootstrap_training_entry() -> None:
        try:
            ensure_layout = getattr(training_tab, "ensure_visible_layout_ready", None)
            if callable(ensure_layout):
                ensure_layout(force=True)
        except Exception:
            pass

        accepted_context = False
        try:
            accept_context = getattr(training_tab, "_accept_training_input_context", None)
            if callable(accept_context):
                accepted_context = bool(
                    accept_context(
                        source="z2_export",
                        target="plate",
                        dataset_path=target_dataset,
                        select_training=True,
                    )
                )
        except Exception:
            accepted_context = False

        if not accepted_context:
            try:
                training_tab._set_step4_dataset_mode("plate")
            except Exception:
                pass
            try:
                training_tab._mark_step4_dataset_ready(dataset_path=target_dataset)
            except Exception:
                try:
                    training_tab.dataset_var.set(str(target_dataset))
                except Exception:
                    pass
            try:
                training_tab._update_step4_notebook_mode()
            except Exception:
                pass
            try:
                training_tab._step4_dataset_go_next()
            except Exception:
                try:
                    training_tab.main_nb.select(training_tab.tab_train)
                except Exception:
                    pass
        try:
            training_tab._refresh_training_start_state()
        except Exception:
            pass

    try:
        training_tab.frame.after_idle(_bootstrap_training_entry)
        training_tab.frame.after(120, _bootstrap_training_entry)
    except Exception:
        _bootstrap_training_entry()
    return True


def _go_to_next_workflow_step(self, *args, **kwargs):
    return z2_workflow_methods._go_to_next_workflow_step(self, *args, **kwargs)


def _get_preferred_annotation_run_dir(self, *, require_xml: bool = False) -> Path | None:
    for candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.plate_dataset_run_var.get() or "").strip(),
    ):
        safe_run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=require_xml)
        if safe_run_dir is not None:
            return safe_run_dir

    # Free-mode presentation reads an explicit selection, not a discovered run.
    # History/import actions resolve their selection before entering the editor.
    if self._is_free_mode_session_context():
        return None

    if self._should_skip_annotation_run_lookup_for_current_input():
        return None

    current_input = getattr(self, "current_input_dir", None)
    if current_input is not None:
        run_dir = self._find_latest_annotation_run_for_input(Path(current_input), self._get_annotation_run_roots())
        run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=require_xml)
        if run_dir is not None:
            return run_dir

    return self._find_latest_annotation_run_dir() if not require_xml else self._resolve_safe_annotation_run_dir(
        self._find_latest_annotation_run_dir(),
        require_xml=True,
    )


def _should_skip_annotation_run_lookup_for_current_input(self) -> bool:
    if not self._is_free_mode_session_context():
        return False
    try:
        screen_value = str(self._get_free_mode_screen() or "").strip().lower()
        if screen_value and screen_value != "workflow":
            return False
    except Exception:
        return False

    route = self._get_workflow_route()
    step = self._normalize_workflow_step_value(self._get_workflow_step())
    if route == "auto":
        return bool(
            step in {"auto_input", "auto_start"}
            and getattr(self, "_auto_route_settings_pending", False)
        )
    if route == "manual":
        return bool(
            self._get_manual_entry_mode() == "new"
            and step in {"manual_input", "manual_start"}
            and not getattr(self, "_manual_template_ready_for_review", False)
        )
    return False


def _refresh_workflow_route_cards(self, *, refresh_content: bool = True):
    dispatch_refresh_workflow_route_cards(self, refresh_content=refresh_content)


def _select_free_mode_route(self, route: str):
    select_free_mode_route(self, route)


def _set_manual_entry_mode(self, mode: str):
    dispatch_set_manual_entry_mode(self, mode)


def _on_manual_entry_mode_change(self):
    self._set_manual_entry_mode(self.manual_entry_mode_var.get())


def _on_auto_vehicle_skip_toggle(self):
    normalized_choice = self._normalize_auto_vehicle_choice(self.auto_vehicle_choice_var.get())
    campaign_context = not self._is_free_mode_session_context()
    self._set_workflow_route_state("auto", campaign_context=campaign_context)
    if not campaign_context:
        self._set_screen_state("workflow", campaign_context=False)
    self.auto_vehicle_choice_var.set(normalized_choice)
    self.manual_xml_template_var.set(False)
    self._manual_review_from_auto = False
    self._manual_review_origin_route = ""
    self._manual_review_export_ready = False
    self._dataset_export_completed = False
    if (
        normalized_choice == "skip"
        and self._normalize_workflow_step_value(self._get_workflow_step()) == "auto_vehicle_model"
    ):
        self._set_workflow_step_state("auto_vehicle_choice", campaign_context=campaign_context)
    if normalized_choice == "skip":
        self.mode_var.set("B: Tylko tablice")
    else:
        self.mode_var.set("C: Pojazdy + tablice")
    self._refresh_auto_vehicle_choice_ui()
    self._refresh_left_panel_route_copy()
    self._refresh_detection_configuration_ui()
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()


def _refresh_auto_vehicle_choice_ui(self):
    skip_check = getattr(self, "auto_vehicle_choice_skip_check", None)
    if skip_check is None:
        return

    try:
        skip_check.grid_configure(row=0, column=0, columnspan=1, sticky="w")
    except Exception:
        pass


def _refresh_run_output_info(self, run_dir: Path | None = None):
    if run_dir is None:
        run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    if run_dir is None:
        self.run_output_info_var.set("")
        return

    pretty_path = self._format_workspace_relative_path(run_dir)
    self.run_output_info_var.set(
        f"Ostatni run anotacji Z2 jest zapisany w: {pretty_path}"
    )


def _open_current_run_dir(self):
    run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    if run_dir is None:
        messagebox.showinfo("Brak runu anotacji", "Najpierw utworz albo otworz run anotacji Z2.")
        return

    try:
        if os.name == "nt":
            os.startfile(str(run_dir))
        else:
            messagebox.showinfo("Info", f"Folder runu anotacji:\n{run_dir}")
    except Exception as e:
        messagebox.showerror("Nie mozna otworzyc folderu", f"{run_dir}\n\n{e}")


def _jump_to_export_section(self):
    dispatch_jump_to_export_section(self)


def _resolve_step3_images_dir_from_z2_run(self, run_dir: Path | None) -> Path | None:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    candidates = []

    if safe_run_dir is not None:
        try:
            manifest = self._load_annotation_run_manifest(safe_run_dir)
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            candidates.extend(
                [
                    manifest.get("source_input_dir"),
                    manifest.get("input_dir"),
                    manifest.get("imported_source_input_dir"),
                ]
            )

    candidates.extend(
        [
            getattr(self, "current_input_dir", None),
            str(self.input_dir_var.get() or "").strip() if hasattr(self, "input_dir_var") else "",
            str(self.plate_dataset_images_var.get() or "").strip()
            if hasattr(self, "plate_dataset_images_var")
            else "",
        ]
    )

    for candidate in candidates:
        resolved = self._resolve_existing_dir(candidate)
        if resolved is not None:
            return resolved
    return None


def _prepare_approved_step3_source_from_z2_run(self, *args, **kwargs):
    return z2_workflow_methods._prepare_approved_step3_source_from_z2_run(self, *args, **kwargs)


def _open_step3_from_z2_annotation_source(self) -> bool:
    if self.is_processing:
        return False

    run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    if run_dir is None:
        messagebox.showwarning(
            "Brak źródła dla Z3",
            "Nie znalazłem gotowego runu Z2 z plikiem annotations.xml. "
            "Najpierw utwórz albo otwórz run anotacji tablic.",
            parent=self.frame.winfo_toplevel(),
        )
        return False

    if not self._ensure_preview_edits_saved("przejście do Z3/PZ1"):
        return False

    xml_path = run_dir / "annotations.xml"
    images_dir = self._resolve_step3_images_dir_from_z2_run(run_dir)
    step3_run_dir = self._prepare_approved_step3_source_from_z2_run(run_dir, images_dir)
    if step3_run_dir is None:
        return False
    xml_path = step3_run_dir / "annotations.xml"

    try:
        character_tab = getattr(getattr(self, "app", None), "tabs", {}).get("characters")
    except Exception:
        character_tab = None

    if character_tab is None:
        messagebox.showwarning(
            "Brak Z3",
            "Zakładka Z3 nie jest teraz dostępna, więc nie mogę przejść bezpośrednio do PZ1.",
            parent=self.frame.winfo_toplevel(),
        )
        return False

    try:
        character_tab.set_pending_z2_annotation_source(
            run_dir=str(step3_run_dir),
            xml_path=str(xml_path),
            images_dir=(str(images_dir) if images_dir is not None else ""),
        )
    except Exception as e:
        logger.debug(f"Nie udało się przekazać źródła Z2 do Z3: {e}")

    try:
        self.app.open_controlled_tab("characters")
    except Exception:
        return False

    bootstrapped = {"done": False}

    def _bootstrap_step3_pz1() -> None:
        if bootstrapped["done"]:
            return
        bootstrapped["done"] = True

        try:
            character_tab.reset_subtab_flow()
        except Exception:
            pass
        try:
            character_tab.back_to_substep_1()
        except Exception:
            pass
        try:
            character_tab.preview_dir_var.set("")
            character_tab._reset_preview_cache()
        except Exception:
            pass

        source_ready = False
        try:
            source_ready = bool(
                character_tab._use_z2_source_on_demand(
                    notify_on_failure=True,
                    advance_to_start=True,
                )
            )
        except Exception as e:
            logger.debug(f"Nie udało się automatycznie przygotować PZ1 z runu Z2: {e}")

        try:
            character_tab._refresh_extract_workflow_ui()
        except Exception:
            pass

        try:
            status_text = (
                "Z2 przekazało annotations.xml do Z3/PZ1. Możesz wyciąć tablice do pracy nad znakami."
                if source_ready
                else "Otworzyłem Z3/PZ1. Sprawdź zgodność annotations.xml i katalogu obrazów."
            )
            self.app.update_status(status_text, "success" if source_ready else "warning")
        except Exception:
            pass

    try:
        character_tab.frame.after_idle(_bootstrap_step3_pz1)
    except Exception:
        _bootstrap_step3_pz1()

    return True


def _open_existing_run_for_manual_review(
    self,
    run_dir: Path | None = None,
    *,
    allow_fallback: bool = True,
    from_auto: bool = False,
    show_dialog: bool = True,
    entry_mode: str | None = None,
) -> bool:
    return dispatch_open_existing_run_for_manual_review(
        self,
        run_dir=run_dir,
        allow_fallback=allow_fallback,
        from_auto=from_auto,
        show_dialog=show_dialog,
        entry_mode=entry_mode,
    )


def _open_existing_run_for_campaign_review(
    self,
    run_dir: Path | None = None,
    *,
    iteration_target: str | None = None,
    manual_template: bool = False,
    defer_ui_restore: bool = False,
) -> bool:
    return open_existing_run_for_campaign_review(
        self,
        run_dir=run_dir,
        iteration_target=iteration_target,
        manual_template=manual_template,
        defer_ui_restore=defer_ui_restore,
    )


def _build_z2_layout_state_campaign(
    self,
    *,
    route: str,
    campaign_stage: int,
    campaign_iteration_target: str,
    available_primary_action_ids: list[str],
    manual_review_active: bool,
    auto_completed: bool,
    has_existing_run: bool,
) -> dict:
    return build_z2_layout_state_campaign(
        self,
        route=route,
        campaign_stage=campaign_stage,
        campaign_iteration_target=campaign_iteration_target,
        available_primary_action_ids=available_primary_action_ids,
        manual_review_active=manual_review_active,
        auto_completed=auto_completed,
        has_existing_run=has_existing_run,
    )


def _build_z2_layout_state_free_mode(
    self,
    *,
    free_mode_screen: str,
    route: str,
    manual_review_active: bool,
) -> dict:
    return build_z2_layout_state_free_mode(
        self,
        free_mode_screen=free_mode_screen,
        route=route,
        manual_review_active=manual_review_active,
    )


def _build_z2_cta_state_campaign(
    self,
    *,
    route: str,
    current_step: str,
    show_workflow_steps: bool,
    show_export_followup: bool,
    auto_completed: bool,
    manual_run_already_created: bool,
    input_dir_ready: bool,
    auto_setup_pending: bool,
    auto_vehicle_choice: str,
    manual_setup: bool,
    campaign_reused_manual_count: int,
) -> dict:
    return build_z2_cta_state_campaign(
        self,
        route=route,
        current_step=current_step,
        show_workflow_steps=show_workflow_steps,
        show_export_followup=show_export_followup,
        auto_completed=auto_completed,
        manual_run_already_created=manual_run_already_created,
        input_dir_ready=input_dir_ready,
        auto_setup_pending=auto_setup_pending,
        auto_vehicle_choice=auto_vehicle_choice,
        manual_setup=manual_setup,
        campaign_reused_manual_count=campaign_reused_manual_count,
    )


def _build_z2_cta_state_free_mode(
    self,
    *,
    route: str,
    current_step: str,
    free_mode_screen: str,
    show_workflow_steps: bool,
    has_existing_run: bool,
    input_dir_ready: bool,
    auto_setup_pending: bool,
    auto_vehicle_choice: str,
    manual_setup: bool,
    manual_run_already_created: bool,
) -> dict:
    return build_z2_cta_state_free_mode(
        self,
        route=route,
        current_step=current_step,
        free_mode_screen=free_mode_screen,
        show_workflow_steps=show_workflow_steps,
        has_existing_run=has_existing_run,
        input_dir_ready=input_dir_ready,
        auto_setup_pending=auto_setup_pending,
        auto_vehicle_choice=auto_vehicle_choice,
        manual_setup=manual_setup,
        manual_run_already_created=manual_run_already_created,
    )


def _refresh_free_mode_workflow_ui(self, *args, **kwargs):
    return z2_workflow_methods._refresh_free_mode_workflow_ui(self, *args, **kwargs)


def _manual_xml_template_enabled(self) -> bool:
    if self._is_campaign_step2_context():
        try:
            return bool(
                self._get_workflow_route() == "manual"
                and self._get_manual_entry_mode() == "new"
            )
        except Exception:
            pass
    return bool(self.manual_xml_template_var.get())


def _manual_vehicle_assist_enabled(self) -> bool:
    return self._manual_xml_template_enabled() and bool(self.manual_vehicle_assist_var.get())


def _return_to_campaign_wizard(self):
    if self._is_plate_auto_scope_modal_blocking_actions():
        return
    if bool(getattr(self, "is_processing", False)) or bool(getattr(self, "_annotation_stop_requested", False)):
        try:
            self._refresh_step2_action_states(lightweight=False)
        except Exception:
            pass
        message = (
            "Autoanotacja trwa. Zatrzymaj ją przyciskiem "
            "\"Zatrzymaj autoanotację\", zanim wrócisz do grafu."
        )
        try:
            self._set_status_label_state(message, "warning")
        except Exception:
            pass
        try:
            app = getattr(self, "app", None)
            if app is not None:
                app.update_status(
                    "Autoanotacja trwa. Najpierw zatrzymaj proces w Z2, potem wróć do grafu.",
                    "warning",
                )
        except Exception:
            pass
        return
    try:
        from ..campaign_manager import CAMPAIGN
    except Exception:
        return

    if not CAMPAIGN.get_active_project_name():
        return

    try:
        early_graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        early_graph_context = {}
    early_graph_edge_key = str(early_graph_context.get("graph_edge_key") or "").strip()
    try:
        early_graph_gate_id = campaign_gate_id_for_edge(
            early_graph_edge_key,
            early_graph_context.get("graph_gate_id"),
        )
    except Exception:
        early_graph_gate_id = ""
    if (
        str(early_graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or early_graph_gate_id == "T02"
        or early_graph_edge_key == "e1_to_e3"
    ):
        if not _return_to_campaign_t02_at_review(self, CAMPAIGN, early_graph_context):
            return
        return

    try:
        warning_ctx = self._get_campaign_return_to_wizard_ok_warning_context()
    except Exception:
        warning_ctx = {}
    if bool(warning_ctx.get("show")):
        try:
            proceed = self._prompt_campaign_return_to_wizard_ok_modal(
                approved_images=int(warning_ctx.get("approved_images", 0) or 0),
                approved_plates=int(warning_ctx.get("approved_plates", 0) or 0),
                images_with_plates=int(warning_ctx.get("images_with_plates", 0) or 0),
                required_images=int(warning_ctx.get("required_images", 0) or 0),
                required_plates=int(warning_ctx.get("required_plates", 0) or 0),
                total_plates=int(warning_ctx.get("total_plates", 0) or 0),
                xml_required=bool(warning_ctx.get("xml_required", False)),
                xml_exists=bool(warning_ctx.get("xml_exists", False)),
            )
        except Exception:
            proceed = True
        if not proceed:
            return

    stage_num = int(CAMPAIGN.get_current_step() or 0)
    stage_label = f"E{stage_num}" if stage_num > 0 else "aktualnego etapu"
    return_edge_key = ""
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_edge_key = str(graph_context.get("graph_edge_key") or "").strip()
    graph_gate_id = campaign_gate_id_for_edge(
        graph_edge_key,
        graph_context.get("graph_gate_id"),
    )
    graph_visible_gate_id = str(
        graph_context.get("graph_visible_gate_id")
        or graph_context.get("graph_display_gate_id")
        or ""
    ).strip().upper()
    graph_visible_gate_id = campaign_gate_id_for_edge(graph_edge_key, graph_visible_gate_id)
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
    if graph_gate_id == "T04" and repair_origin_gate_id == "T06":
        stage_num = 4
        stage_label = "E4T / T06"
        return_edge_key = "e4_to_e1"
    is_t02_at_review_return = bool(
        str(graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or graph_gate_id == "T02"
        or graph_edge_key == "e1_to_e3"
    )
    is_step2_char_gate_return = bool(
        (
            graph_gate_id == "T03"
            or graph_edge_key == "e2_to_e3"
        )
        and repair_origin_gate_id != "T06"
    )

    try:
        current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    except Exception:
        current_target = ""
    try:
        step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
    except Exception:
        step3_status = ""

    if not return_edge_key and stage_num >= 3 and step3_status == "needs_rework" and current_target != "char":
        try:
            CAMPAIGN.set_iteration_target("char")
            CAMPAIGN.set_current_step(3)
            stage_num = 3
            stage_label = "E3"
        except Exception:
            pass

    if is_t02_at_review_return:
        if not _return_to_campaign_t02_at_review(self, CAMPAIGN, graph_context):
            return
        return

    if is_step2_char_gate_return:
        if current_target != "char":
            old_target = current_target
            try:
                CAMPAIGN.set_iteration_target("char")
                current_target = "char"
                logger.info(
                    "[Z2 GRAPH] corrected target for T03 return old_target=%s edge=%s gate=%s display=%s",
                    old_target or "-",
                    graph_edge_key or "-",
                    graph_gate_id or "-",
                    graph_visible_gate_id or "-",
                )
            except Exception:
                pass
        if not _return_to_campaign_step2_char_gate(self, CAMPAIGN, graph_context):
            return
        return
    promotion_result = self._promote_campaign_char_repair_ok_to_approved_pool_before_return()
    if bool(promotion_result.get("abort")):
        return
    t05_promotion_result = z2_workflow_methods._promote_campaign_t05_ok_to_approved_pool_before_return(self)
    if bool(t05_promotion_result.get("abort")):
        return
    if graph_gate_id == "T04" and repair_origin_gate_id == "T06":
        try:
            from datetime import datetime as _datetime
            now = _datetime.now().isoformat(timespec="seconds")
            CAMPAIGN.upsert_iteration_state(
                updates={
                    "t07_repair_session": {
                        "active": False,
                        "state": "resolved",
                        "resolved_at": now,
                        "updated_at": now,
                        "last_return_result": dict(promotion_result or {}),
                    }
                }
            )
        except Exception as exc:
            logger.debug(f"Nie udało się domknąć znacznika naprawy T06: {exc}")

    campaign_tab = None
    try:
        campaign_tab = self.app.tabs.get("campaign")
        if campaign_tab is not None:
            if return_edge_key:
                try:
                    campaign_tab._campaign_graph_selected_edge_key = return_edge_key
                except Exception:
                    pass
            try:
                campaign_tab.request_wizard_stage_focus(step_num=stage_num)
            except Exception:
                pass
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        self.app.open_controlled_tab("campaign")
        self.app.update_campaign_tab_access()
        self.app.update_status(
            f"Wracam do grafu kampanii na {stage_label}.",
            "info",
        )
    except Exception as e:
        logger.debug(f"Nie udalo sie wrocic do grafu kampanii: {e}")


def _prompt_t02_at_review_commit_choice(self, *, approved_images: int, approved_plates: int) -> bool:
    parent = self.frame.winfo_toplevel()
    result = {"ok": False}
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    card_bg = palette.get("card", "#2d2d30")
    fg = palette.get("fg", "#f3f4f6")
    muted = palette.get("muted", "#c7c7c7")
    warning = palette.get("warning", "#f39c12")
    border = palette.get("border", "#3f3f46")

    dialog = tk.Toplevel(parent)
    dialog.withdraw()
    try:
        self.app.style_dialog_window(
            dialog,
            title="Kontrola AT wybierze T02",
            geometry="640x360",
            parent=parent,
        )
    except Exception:
        dialog.title("Kontrola AT wybierze T02")
        dialog.configure(bg=panel_bg)
    dialog.resizable(False, False)
    try:
        dialog.transient(parent)
    except Exception:
        pass

    try:
        body = self.app._build_themed_dialog_surface(dialog, tone="warning")
    except Exception:
        body = tk.Frame(dialog, bg=panel_bg, padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

    shell = tk.Frame(body, bg=panel_bg)
    shell.pack(fill=tk.BOTH, expand=True)
    tk.Label(
        shell,
        text="Zapis kontroli AT zamknie wybór toru tej iteracji",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI Semibold", 13),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 8))
    tk.Label(
        shell,
        text=(
            "Masz oznaczone pozycje [OK] w kontroli importu AT. "
            "Jeśli je zapiszesz, bieżąca iteracja zostanie przypisana do bramki T02 "
            "i nie przełączymy jej później na T01 bez osobnego cofnięcia tej kontroli."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        justify=tk.LEFT,
        anchor="w",
        wraplength=580,
    ).pack(fill=tk.X, pady=(0, 12))

    summary = tk.Frame(shell, bg=card_bg, highlightthickness=1, highlightbackground=border, padx=12, pady=10)
    summary.pack(fill=tk.X, pady=(0, 12))
    tk.Label(
        summary,
        text=f"Do zapisania: {int(approved_images or 0)} obrazów [OK] / {int(approved_plates or 0)} tablic",
        bg=card_bg,
        fg=warning,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
    ).pack(fill=tk.X)
    tk.Label(
        summary,
        text="To jest pierwszy realny wkład T02 w tej iteracji.",
        bg=card_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
    ).pack(fill=tk.X, pady=(4, 0))

    buttons = tk.Frame(shell, bg=panel_bg)
    buttons.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

    def choose(ok: bool) -> None:
        result["ok"] = bool(ok)
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(
        buttons,
        text="Zapisz i wybierz T02",
        style="Accent.TButton",
        command=lambda: choose(True),
    ).pack(side=tk.RIGHT)
    ttk.Button(
        buttons,
        text="Wróć do kontroli",
        command=lambda: choose(False),
    ).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
    except Exception:
        pass
    dialog.bind("<Escape>", lambda _event: choose(False))
    dialog.wait_window()
    return bool(result.get("ok"))


def _return_to_campaign_t02_at_review(self, CAMPAIGN, graph_context: dict) -> bool:
    """Return from Z2 opened as the T02 AT review without approving the gate."""
    return_started = time.perf_counter()
    phase_started = return_started
    slow_phases: list[str] = []

    def _mark_return_phase(name: str, *, threshold_ms: float = 150.0) -> None:
        nonlocal phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - phase_started) * 1000.0
            if elapsed_ms >= float(threshold_ms):
                slow_phases.append(f"{name}={elapsed_ms:.0f}ms")
            phase_started = now
        except Exception:
            pass

    if not self._ensure_preview_edits_saved("powrót z kontroli AT do bramki T02"):
        return False

    _mark_return_phase("save_preview")
    run_dir = None
    for raw_candidate in (
        getattr(self, "current_annotation_run_dir", None),
        graph_context.get("restore_run_dir") if isinstance(graph_context, dict) else None,
        getattr(self, "last_staging_run_dir", None),
    ):
        try:
            run_dir = self._resolve_safe_annotation_run_dir(raw_candidate, require_xml=True)
        except Exception:
            run_dir = None
        if run_dir is not None:
            break

    _mark_return_phase("resolve_initial_run")
    if run_dir is None:
        try:
            source_info = dict(CAMPAIGN.get_project_start_plate_source() or {})
        except Exception:
            source_info = {}
        for raw_candidate in (
            source_info.get("source_run_path"),
            source_info.get("source_xml_path"),
        ):
            if not str(raw_candidate or "").strip():
                continue
            try:
                candidate = Path(str(raw_candidate))
                if candidate.suffix.lower() == ".xml":
                    candidate = candidate.parent
                run_dir = self._resolve_safe_annotation_run_dir(candidate, require_xml=True)
            except Exception:
                run_dir = None
            if run_dir is not None:
                break

    _mark_return_phase("resolve_fallback_run")
    if run_dir is not None:
        try:
            self._persist_preview_approved_filenames()
        except Exception:
            pass
        try:
            approved_payload = set(self._get_preview_approved_filenames() or set())
            if not self._is_free_mode_session_context():
                approved_payload |= {
                    str(name or "").strip().lower()
                    for name in set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
                    if str(name or "").strip()
                }
            self._update_annotation_run_manifest(
                run_dir,
                approved_filenames=sorted(approved_payload),
                **self._collect_preview_resume_manifest_fields(),
            )
        except Exception as exc:
            logger.debug(f"Nie udalo sie jawnie zapisac statusow OK kontroli T02: {exc}")

    _mark_return_phase("persist_approved")
    approved_images = 0
    approved_plates = 0
    if run_dir is not None:
        try:
            approved_images, approved_plates = self._get_run_plate_approved_counts(run_dir)
        except Exception:
            approved_images, approved_plates = 0, 0

    _mark_return_phase("count_approved")
    promotion_result = {"ok": True, "reason": "no_ok_images"}
    if run_dir is not None and int(approved_images or 0) > 0 and int(approved_plates or 0) > 0:
        try:
            t02_already_committed = bool(CAMPAIGN.is_t02_at_review_committed_current_iteration())
        except Exception:
            t02_already_committed = False
        if not t02_already_committed:
            try:
                proceed_with_t02 = _prompt_t02_at_review_commit_choice(
                    self,
                    approved_images=int(approved_images or 0),
                    approved_plates=int(approved_plates or 0),
                )
            except Exception:
                proceed_with_t02 = True
            if not proceed_with_t02:
                return False
        try:
            promotion_result = dict(
                self._promote_run_to_campaign_plate_approved_set(
                    run_dir,
                    force_parse_xml=False,
                    project_name=str(CAMPAIGN.get_active_project_name() or "").strip() or None,
                )
                or {}
            )
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać [OK] z kontroli T02 do puli projektu: {exc}")
            promotion_result = {"ok": False, "reason": "promotion_exception"}
        if not bool(promotion_result.get("ok")):
            try:
                messagebox.showwarning(
                    "Nie zapisano kontroli AT",
                    (
                        "Z2 ma zaznaczone pozycje [OK], ale nie udało się dopisać ich do projektowej "
                        "puli zatwierdzonych tablic.\n\n"
                        "Pozostań w Z2 i spróbuj ponownie wrócić do T02, żeby nie zgubić wyniku kontroli."
                    ),
                    parent=self.frame.winfo_toplevel(),
                )
            except Exception:
                pass
            return False
        try:
            CAMPAIGN.mark_t02_at_review_committed(
                run_dir=str(run_dir or ""),
                approved_images=int(approved_images or 0),
                approved_plates=int(approved_plates or 0),
            )
        except Exception as exc:
            logger.debug(f"Nie udalo sie zapisac znacznika wyboru T02 po kontroli AT: {exc}")

    _mark_return_phase("promote_approved")
    try:
        CAMPAIGN.set_iteration_path("char_from_ready_plates")
        CAMPAIGN.set_current_step(1)
        if hasattr(CAMPAIGN, "set_graph_selected_edge_key"):
            CAMPAIGN.set_graph_selected_edge_key("e1_to_e3")
        if hasattr(CAMPAIGN, "invalidate_step3_char_source_state_cache"):
            CAMPAIGN.invalidate_step3_char_source_state_cache()
    except Exception as exc:
        logger.debug(f"Nie udało się przywrócić fokusu T02 po kontroli AT: {exc}")

    try:
        campaign_tab = self.app.tabs.get("campaign")
        if campaign_tab is not None:
            try:
                campaign_tab._campaign_graph_selected_edge_key = "e1_to_e3"
            except Exception:
                pass
            try:
                campaign_tab.request_wizard_stage_focus(step_num=1)
            except Exception:
                pass
    except Exception:
        campaign_tab = None

    _mark_return_phase("prepare_campaign_tab")

    try:
        promotion_ok = bool((promotion_result or {}).get("ok"))
        promotion_reason = str(
            (promotion_result or {}).get("reason")
            or ("ok" if promotion_ok else "failed")
        )
        logger.info(
            "[Z2 GRAPH] return_to_t02_at_review run=%s ok_images=%s ok_plates=%s promotion=%s",
            str(run_dir or "-"),
            int(approved_images or 0),
            int(approved_plates or 0),
            promotion_reason,
        )
    except Exception:
        pass

    try:
        if int(approved_plates or 0) > 0:
            status_message = (
                f"Zapisano kontrolę AT: {approved_images} obrazów [OK] / "
                f"{approved_plates} tablic. Wrócono do bramki T02."
            )
            tone = "success"
        else:
            status_message = "Wrócono do bramki T02 bez nowych pozycji [OK]. Zasób AT pozostaje do kontroli."
            tone = "warning"
        self.app.open_controlled_tab("campaign")
        self.app.update_status(status_message, tone)
        if campaign_tab is not None:
            def _refresh_t02_dashboard_after_return(tab=campaign_tab) -> None:
                previous_lightweight_refresh = bool(getattr(tab, "_project_open_lightweight_refresh", False))
                try:
                    tab._project_open_lightweight_refresh = True
                    refresh_active_only = getattr(tab, "_refresh_active_project_wizard_only", None)
                    if callable(refresh_active_only):
                        refresh_active_only()
                    else:
                        tab._refresh_dashboard()
                except Exception as exc:
                    logger.debug(f"Nie udało się lekko odświeżyć grafu po kontroli T02: {exc}")
                finally:
                    try:
                        tab._project_open_lightweight_refresh = previous_lightweight_refresh
                    except Exception:
                        pass

            try:
                self.frame.after(180, _refresh_t02_dashboard_after_return)
            except Exception:
                _refresh_t02_dashboard_after_return()
    except Exception as e:
        logger.debug(f"Nie udalo sie wrocic do bramki T02 po kontroli AT: {e}")
    _mark_return_phase("open_campaign")

    try:
        elapsed_ms = (time.perf_counter() - return_started) * 1000.0
        if elapsed_ms >= 250.0 or slow_phases:
            logger.info(
                "[Z2 PERF] return_to_t02_at_review total=%.0fms phases=[%s] run=%s ok=%s/%s",
                elapsed_ms,
                ", ".join(slow_phases) if slow_phases else "no_slow_phase",
                str(run_dir or "-"),
                int(approved_images or 0),
                int(approved_plates or 0),
            )
    except Exception:
        pass
    return True


def _return_to_campaign_step2_char_gate(self, CAMPAIGN, graph_context: dict) -> bool:
    """Return from Z2 opened by the E2->E3 gate without closing that gate."""
    if not self._ensure_preview_edits_saved("powrót z Z2 do bramki pracy nad znakami"):
        return False

    try:
        current_run_dir = self._resolve_safe_annotation_run_dir(
            getattr(self, "current_annotation_run_dir", None),
            require_xml=True,
        )
    except Exception:
        current_run_dir = None
    if current_run_dir is not None:
        try:
            self._persist_preview_approved_filenames()
        except Exception:
            pass

    edge_key = str(graph_context.get("graph_edge_key") or "e2_to_e3").strip() or "e2_to_e3"
    graph_gate_id = campaign_gate_id_for_edge(edge_key, graph_context.get("graph_gate_id"))
    graph_visible_gate_id = str(
        graph_context.get("graph_visible_gate_id")
        or graph_context.get("graph_display_gate_id")
        or ""
    ).strip().upper()
    graph_visible_gate_id = campaign_gate_id_for_edge(edge_key, graph_visible_gate_id)
    if graph_gate_id == "T03" or graph_visible_gate_id == "T03":
        edge_key = "e2_to_e3"
    try:
        if current_run_dir is not None:
            CAMPAIGN.set_step2_generated(str(current_run_dir))
        else:
            existing_step2_run = str(CAMPAIGN.get_step2_staging_run() or "").strip()
            if existing_step2_run:
                CAMPAIGN.set_step2_generated(existing_step2_run)
    except Exception as exc:
        logger.debug(f"Nie udało się przywrócić T03 jako pracy oczekującej na zatwierdzenie: {exc}")
    try:
        CAMPAIGN.reset_step3()
    except Exception as exc:
        logger.debug(f"Nie udało się wyczyścić roboczego stanu E3 przy powrocie do T03: {exc}")
    try:
        CAMPAIGN.set_current_step(2)
    except Exception:
        pass
    try:
        if hasattr(CAMPAIGN, "set_graph_selected_edge_key"):
            CAMPAIGN.set_graph_selected_edge_key(edge_key)
    except Exception:
        pass

    try:
        campaign_tab = self.app.tabs.get("campaign")
        if campaign_tab is not None:
            try:
                campaign_tab._campaign_graph_selected_edge_key = edge_key
            except Exception:
                pass
            try:
                campaign_tab.request_wizard_stage_focus(step_num=2)
            except Exception:
                pass
            try:
                campaign_tab._clear_dashboard_perf_cache()
            except Exception:
                pass
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    try:
        logger.info("[Z2 GRAPH] return_to_step2_char_gate edge=%s gate=T03", edge_key)
    except Exception:
        pass
    try:
        self.app.open_controlled_tab("campaign")
        self.app.update_campaign_tab_access()
        self.app.update_status(
            "Wrócono do bramki T03. Zapisane [OK] są dostępne dla pracy nad znakami; przejście dalej wymaga jawnego zatwierdzenia bramki na grafie.",
            "info",
        )
    except Exception as e:
        logger.debug(f"Nie udalo sie wrocic do bramki T03 po Z2: {e}")
    return True


def _promote_campaign_char_repair_ok_to_approved_pool_before_return(self, *args, **kwargs):
    return z2_workflow_methods._promote_campaign_char_repair_ok_to_approved_pool_before_return(self, *args, **kwargs)


def _get_campaign_return_to_wizard_ok_warning_context(self) -> dict:
    if self._is_free_mode_session_context():
        return {"show": False, "images_with_plates": 0, "approved_images": 0, "approved_plates": 0, "required_images": 0, "required_plates": 0, "total_plates": 0, "xml_required": False, "xml_exists": False}

    if bool(getattr(self, "is_processing", False)):
        return {"show": False, "images_with_plates": 0, "approved_images": 0, "approved_plates": 0, "required_images": 0, "required_plates": 0, "total_plates": 0, "xml_required": False, "xml_exists": False}

    annotations = list(getattr(self, "current_annotations", []) or [])
    campaign_manager_obj = None
    try:
        from ..campaign_manager import CAMPAIGN
        campaign_manager_obj = CAMPAIGN
        current_step = int(CAMPAIGN.get_current_step() or 0)
        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_step = 0
        iteration_target = ""
        iteration_num = 1
    required_images = 0
    if current_step == 2 and iteration_target == "char":
        required_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
    elif current_step == 2 and iteration_target == "plate":
        required_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    else:
        required_plates = 0
    xml_required = bool(current_step == 2 and iteration_target == "plate" and int(iteration_num or 1) <= 1)
    approval_context = self._get_campaign_step2_approval_context()
    approval_run_dir = approval_context.get("run_dir")
    xml_exists = bool(approval_run_dir and Path(approval_run_dir) and (Path(approval_run_dir) / "annotations.xml").exists())
    project_approved_images = 0
    project_approved_plates = 0
    if current_step == 2 and iteration_target in {"plate", "char"} and campaign_manager_obj is not None:
        try:
            approved_stats = dict(campaign_manager_obj.get_plate_approved_set_stats() or {})
            project_approved_images = int(approved_stats.get("images", 0) or 0)
            project_approved_plates = int(approved_stats.get("plates", 0) or 0)
        except Exception:
            project_approved_images = 0
            project_approved_plates = 0
    if not annotations:
        effective_approved_images = int(project_approved_images or 0)
        effective_approved_plates = int(project_approved_plates or 0)
        return {
            "show": bool(
                (required_plates > 0 and effective_approved_plates < int(required_plates or 0))
                or (xml_required and not xml_exists)
            ),
            "images_with_plates": 0,
            "approved_images": effective_approved_images,
            "approved_plates": effective_approved_plates,
            "required_images": required_images,
            "required_plates": required_plates,
            "total_plates": 0,
            "xml_required": xml_required,
            "xml_exists": xml_exists,
        }

    approved_names = set(self._get_preview_approved_filenames_base())
    images_with_plates = 0
    approved_images = 0
    approved_plates = 0
    total_plates = 0

    for ann in annotations:
        try:
            plate_count = len(self._get_plate_detections(ann))
            if plate_count <= 0:
                continue
        except Exception:
            continue
        images_with_plates += 1
        total_plates += int(plate_count or 0)
        try:
            if self._preview_annotation_is_explicitly_approved(ann, approved_names=approved_names):
                approved_images += 1
                approved_plates += int(plate_count or 0)
        except Exception:
            pass

    effective_approved_images = int(project_approved_images or 0) + int(approved_images or 0)
    effective_approved_plates = int(project_approved_plates or 0) + int(approved_plates or 0)

    show = bool(images_with_plates > 0 and approved_images < images_with_plates)
    if required_plates > 0:
        show = bool(show or effective_approved_plates < required_plates)
    if xml_required and not xml_exists:
        show = True
    return {
        "show": show,
        "images_with_plates": int(images_with_plates),
        "approved_images": int(effective_approved_images),
        "approved_plates": int(effective_approved_plates),
        "required_images": int(required_images),
        "required_plates": int(required_plates),
        "total_plates": int(total_plates),
        "xml_required": bool(xml_required),
        "xml_exists": bool(xml_exists),
    }


def _prompt_campaign_return_to_wizard_ok_modal(self, *args, **kwargs):
    return z2_workflow_methods._prompt_campaign_return_to_wizard_ok_modal(self, *args, **kwargs)


def _build_z2_left_panel_copy_context(self) -> Z2LeftPanelCopyContext:
    actual_route = self._get_workflow_route()
    route = self._get_z2_thematic_route()
    campaign_context = not self._is_free_mode_session_context()
    manual_entry_mode = self._get_manual_entry_mode()
    current_step = self._coerce_workflow_step()
    has_existing_run = self._get_preferred_annotation_run_dir(require_xml=True) is not None

    campaign_iteration_target = ""
    campaign_stage = 0
    campaign_iteration_num = 0
    if campaign_context:
        try:
            from ..campaign_manager import CAMPAIGN
            campaign_iteration_target = self._normalize_campaign_iteration_target_value(
                CAMPAIGN.get_iteration_target()
            )
            campaign_stage = int(CAMPAIGN.get_current_step() or 0)
            campaign_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            campaign_iteration_target = ""
            campaign_stage = 0
            campaign_iteration_num = 0

    return Z2LeftPanelCopyContext(
        actual_route=actual_route,
        route=route,
        campaign_context=campaign_context,
        auto_setup_pending=bool(
            self._is_free_mode_session_context()
            and route == "auto"
            and getattr(self, "_auto_route_settings_pending", False)
        ),
        manual_entry_mode=manual_entry_mode,
        manual_setup=(actual_route == "manual" and manual_entry_mode == "new"),
        manual_import=(actual_route == "manual" and manual_entry_mode == "import"),
        vehicle_assist_enabled=self._manual_vehicle_assist_enabled(),
        auto_vehicle_choice=self._get_auto_vehicle_choice(),
        has_existing_run=has_existing_run,
        manual_run_already_created=bool(
            actual_route == "manual"
            and manual_entry_mode == "new"
            and self._has_active_manual_template_run()
        ),
        plate_model_selected=bool(str(self.plate_custom_var.get() or "").strip()),
        campaign_iteration_target=campaign_iteration_target,
        campaign_stage=campaign_stage,
        campaign_iteration_num=campaign_iteration_num,
        current_step=current_step,
        current_index=self._get_workflow_progress_display()[0],
        total_steps=self._get_workflow_progress_display()[1],
        has_manual_history=bool(
            (campaign_context or (actual_route == "manual" and manual_entry_mode == "continue"))
            and self._get_manual_review_history_display_entries()
        ),
        auto_completed=self._is_z2_auto_flow_completed(
            route=route,
            has_existing_run=has_existing_run,
        ),
        manual_review_active=bool(self._manual_review_active and has_existing_run),
        campaign_reused_manual_count=(
            self._get_campaign_reused_manual_annotation_count()
            if campaign_context
            else 0
        ),
        campaign_char_repair_mode=(
            self._is_campaign_char_repair_return_mode()
            if campaign_context
            else False
        ),
        campaign_manual_skip_count=int(
            dict(getattr(self, "_campaign_pending_batch_summary", {}) or {}).get("manual_skip_count", 0) or 0
        ),
    )


def _build_z2_left_panel_copy_defaults() -> Z2CopyPayload:
    return Z2CopyPayload()


def _apply_z2_left_panel_copy_payload(self, *args, **kwargs):
    return z2_workflow_methods._apply_z2_left_panel_copy_payload(self, *args, **kwargs)


def _build_z2_left_panel_copy_payload_campaign(self, ctx: Z2LeftPanelCopyContext, payload: Z2CopyPayload) -> Z2CopyPayload:
    return build_z2_left_panel_copy_payload_campaign(self, ctx, payload)


def _build_z2_left_panel_copy_payload_free_mode(self, ctx: Z2LeftPanelCopyContext, payload: Z2CopyPayload) -> Z2CopyPayload:
    return build_z2_left_panel_copy_payload_free_mode(self, ctx, payload)


def _refresh_left_panel_route_copy(self):
    ctx = self._build_z2_left_panel_copy_context()
    self.workflow_intro_var.set("")
    palette = getattr(self.app, "palette", {})
    payload = self._build_z2_left_panel_copy_defaults()
    if ctx.campaign_context:
        payload = self._build_z2_left_panel_copy_payload_campaign(ctx, payload)
    else:
        payload = self._build_z2_left_panel_copy_payload_free_mode(ctx, payload)

    prefix_lookup = self._get_z2_thematic_title_prefixes()

    payload.run_title = self._format_z2_thematic_title(str(payload.run_title or ""), prefix_lookup.get("run"))
    payload.auto_plate_model_title = self._strip_workflow_heading_prefix(str(payload.auto_plate_model_title or ""))
    payload.workflow_conf_title = self._strip_workflow_heading_prefix(str(payload.workflow_conf_title or ""))
    payload.auto_vehicle_choice_title = self._strip_workflow_heading_prefix(str(payload.auto_vehicle_choice_title or ""))
    payload.workflow_vehicle_title = self._strip_workflow_heading_prefix(str(payload.workflow_vehicle_title or ""))
    payload.workflow_input_title = self._strip_workflow_heading_prefix(str(payload.workflow_input_title or ""))
    payload.workflow_start_title = self._strip_workflow_heading_prefix(str(payload.workflow_start_title or ""))
    payload.manual_entry_title = self._strip_workflow_heading_prefix(str(payload.manual_entry_title or ""))
    payload.manual_history_title = self._strip_workflow_heading_prefix(str(payload.manual_history_title or ""))
    payload.followup_title = self._format_z2_thematic_title(str(payload.followup_title or ""), prefix_lookup.get("followup"))
    payload.export_title = self._strip_workflow_heading_prefix(str(payload.export_title or ""))
    self._apply_z2_left_panel_copy_payload(payload, prefix_lookup, palette)
