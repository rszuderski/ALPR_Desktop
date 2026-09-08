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
    campaign_visible_gate_id,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
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


def _refresh_free_mode_workflow_ui(self):
    if getattr(self, "_free_mode_session_restore_in_progress", False) and self._is_free_mode_session_context():
        return
    if bool(getattr(self, "_campaign_step2_transition_in_progress", False)):
        self._campaign_step2_transition_refresh_pending = True
        return

    if bool(getattr(self, "_workflow_ui_refresh_in_progress", False)):
        self._workflow_ui_refresh_pending = True
        return

    self._workflow_ui_refresh_in_progress = True
    self._refresh_workflow_route_cards()
    preferred_run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    if (
        preferred_run_dir is None
        and self._is_free_mode_session_context()
        and bool(getattr(self, "_manual_review_active", False))
        and self._coerce_free_mode_screen() == "manual_review"
    ):
        preferred_run_dir = self._get_active_annotation_run_dir(require_xml=True)
    self._refresh_run_output_info(preferred_run_dir)
    secondary_actions = self._get_z2_secondary_actions()
    secondary_ctx = self._build_z2_action_context()

    base_ctx = dispatch_build_z2_workflow_base_context(self, preferred_run_dir)
    actual_route = str(base_ctx.actual_route or "")
    route = str(base_ctx.route or "")
    campaign_context = bool(base_ctx.campaign_context)
    auto_setup_pending = bool(base_ctx.auto_setup_pending)
    manual_entry_mode = str(base_ctx.manual_entry_mode or "")
    auto_vehicle_choice = str(base_ctx.auto_vehicle_choice or "")
    plate_model_selected = bool(base_ctx.plate_model_selected)
    input_dir_ready = bool(base_ctx.input_dir_ready)
    vehicle_assist_enabled = bool(base_ctx.vehicle_assist_enabled)
    has_existing_run = bool(base_ctx.has_existing_run)
    manual_setup = bool(base_ctx.manual_setup)
    manual_continue = bool(base_ctx.manual_continue)
    manual_import = bool(base_ctx.manual_import)
    manual_run_already_created = bool(base_ctx.manual_run_already_created)
    current_step = str(base_ctx.current_step or "")
    manual_review_active = bool(base_ctx.manual_review_active)
    manual_review_from_auto = bool(base_ctx.manual_review_from_auto)
    auto_completed = bool(base_ctx.auto_completed)
    campaign_reused_manual_count = int(base_ctx.campaign_reused_manual_count or 0)
    if campaign_context and not route and manual_review_active:
        route = "manual"
    campaign_stage = 0
    campaign_iteration_target = ""
    campaign_iteration_num = 1
    campaign_char_repair_mode = False
    campaign_plate_step4_repair_mode = False
    free_mode_screen = ""
    if campaign_context:
        runtime_state = prepare_campaign_workflow_runtime(
            self,
            route=route,
            manual_review_active=manual_review_active,
            input_dir_ready=input_dir_ready,
        )
        route = str(runtime_state.route or route)
        campaign_stage = int(runtime_state.campaign_stage or 0)
        campaign_iteration_target = str(runtime_state.campaign_iteration_target or "").strip().lower()
        campaign_iteration_num = int(runtime_state.campaign_iteration_num or 1)
        campaign_char_repair_mode = bool(runtime_state.campaign_char_repair_mode)
        campaign_plate_step4_repair_mode = bool(runtime_state.campaign_plate_step4_repair_mode)
        available_primary_action_ids = list(runtime_state.available_primary_action_ids or [])
        manual_entry_mode = self._get_manual_entry_mode()
        manual_setup = route == "manual" and manual_entry_mode == "new"
        manual_continue = route == "manual" and manual_entry_mode == "continue"
        manual_import = route == "manual" and manual_entry_mode == "import"
        manual_run_already_created = bool(manual_setup and self._has_active_manual_template_run())
        current_step = self._coerce_workflow_step()
        if self._get_workflow_step() != current_step:
            self._set_workflow_step_state(current_step, campaign_context=True)
    else:
        runtime_state = prepare_free_mode_workflow_runtime(self)
        available_primary_action_ids = list(runtime_state.available_primary_action_ids or [])
        free_mode_screen = str(runtime_state.free_mode_screen or "")
        if route == "auto" and free_mode_screen == "workflow" and current_step == "auto_start":
            try:
                self._schedule_free_mode_auto_workspace_preview_load()
            except Exception as e:
                logger.debug(f"Nie udało się przygotować preview workspace Z2 dla freemode auto: {e}")
        elif route == "manual" and free_mode_screen == "workflow" and current_step == "manual_start":
            try:
                self._schedule_free_mode_manual_workspace_preview_load()
            except Exception as e:
                logger.debug(f"Nie udało się przygotować preview workspace Z2 dla freemode manual: {e}")

    if campaign_context:
        layout_state = self._build_z2_layout_state_campaign(
            route=route,
            campaign_stage=campaign_stage,
            campaign_iteration_target=campaign_iteration_target,
            available_primary_action_ids=available_primary_action_ids,
            manual_review_active=manual_review_active,
            auto_completed=auto_completed,
            has_existing_run=has_existing_run,
        )
    else:
        layout_state = self._build_z2_layout_state_free_mode(
            free_mode_screen=free_mode_screen,
            route=route,
            manual_review_active=manual_review_active,
        )

    show_route_choice = bool(layout_state.show_route_choice)
    show_export_followup = bool(layout_state.show_export_followup)
    show_auto_followup = bool(layout_state.show_auto_followup)
    show_manual_review_followup = bool(layout_state.show_manual_review_followup)
    show_stage_export_cta = bool(layout_state.show_stage_export_cta)
    compact_export_followup = bool(layout_state.compact_export_followup)
    show_workflow_steps = bool(layout_state.show_workflow_steps)
    compact_single_route_layout = bool(layout_state.compact_single_route_layout)
    compact_left_column_layout = bool(layout_state.compact_left_column_layout)
    show_campaign_context_header = bool(layout_state.show_campaign_context_header)
    show_nav_panel = bool(layout_state.show_nav_panel)
    show_right_panel = bool(layout_state.show_right_panel)

    self._compact_left_column_layout = compact_left_column_layout
    auto_settings_in_scope_modal = bool(show_workflow_steps and route == "auto")
    show_workflow_entry_shell = bool(
        show_route_choice
        or show_workflow_steps
        or show_nav_panel
        or show_campaign_context_header
    )
    show_miniflow_panel = bool(
        (not campaign_context)
        and route
        and (
            show_workflow_steps
            or show_nav_panel
            or show_auto_followup
            or show_manual_review_followup
            or show_export_followup
            or show_campaign_context_header
        )
    )
    self._annotation_right_panel_visible = show_right_panel
    self._sync_main_pane_right_panel_visibility()
    if show_workflow_steps:
        nav_anchor = self.actions_section
    elif show_export_followup:
        nav_anchor = self.export_section
    elif free_mode_screen == "manual_review" and show_stage_export_cta:
        nav_anchor = self.manual_stage_export_box
    elif show_manual_review_followup:
        nav_anchor = self.manual_stage_section
    elif show_auto_followup:
        nav_anchor = self.followup_section
    else:
        nav_anchor = self.workflow_entry_section
    self._set_widget_packed(
        self.workflow_entry_shell,
        show_workflow_entry_shell,
        fill=tk.X,
        pady=((0, 10) if compact_left_column_layout else (0, 16)),
    )
    self._set_widget_packed(
        self.workflow_nav_panel,
        False,
    )
    self._set_widget_packed(
        self.workflow_entry_section,
        show_route_choice,
        fill=tk.X,
        before=(self.actions_section if show_workflow_steps else self.workflow_nav_panel),
    )
    self._set_widget_packed(self.workflow_entry_separator, False)
    self._set_widget_packed(self.workflow_intro_lbl, False)
    self._set_widget_packed(self.source_section, False)
    self._set_widget_packed(self.source_section_separator, False)
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
        graph_gate_id = campaign_gate_id_for_edge(
            graph_context.get("graph_edge_key"),
            graph_context.get("graph_gate_id"),
        )
        repair_origin_gate_id = campaign_gate_id_for_edge(
            graph_context.get("repair_origin_edge_key"),
            graph_context.get("repair_origin_gate_id")
            or graph_context.get("source_graph_gate_id"),
        )
        graph_t05_repair_from_t07 = bool(
            graph_gate_id == "T04"
            and repair_origin_gate_id == "T06"
        )
    except Exception:
        graph_t05_repair_from_t07 = False
    dispatch_apply_z2_workflow_left_layout(
        self,
        campaign_context=campaign_context,
        route=route,
        current_step=current_step,
        manual_setup=manual_setup,
        manual_continue=manual_continue,
        manual_review_active=manual_review_active,
        manual_review_from_auto=manual_review_from_auto,
        vehicle_assist_enabled=vehicle_assist_enabled,
        campaign_stage=campaign_stage,
        campaign_char_repair_mode=campaign_char_repair_mode,
        campaign_plate_step4_repair_mode=bool(campaign_plate_step4_repair_mode or graph_t05_repair_from_t07),
        show_route_choice=show_route_choice,
        show_workflow_steps=show_workflow_steps,
        show_auto_followup=show_auto_followup,
        show_manual_review_followup=show_manual_review_followup,
        show_export_followup=show_export_followup,
        show_stage_export_cta=show_stage_export_cta,
        show_nav_panel=show_nav_panel,
        show_miniflow_panel=show_miniflow_panel,
        show_campaign_context_header=show_campaign_context_header,
        show_right_panel=show_right_panel,
        compact_single_route_layout=compact_single_route_layout,
        compact_export_followup=compact_export_followup,
        auto_settings_in_scope_modal=auto_settings_in_scope_modal,
        nav_anchor=nav_anchor,
        secondary_actions=secondary_actions,
        secondary_ctx=secondary_ctx,
    )
    if campaign_context:
        cta_state = self._build_z2_cta_state_campaign(
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
    else:
        cta_state = self._build_z2_cta_state_free_mode(
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

    dispatch_apply_z2_workflow_cta_ui(
        self,
        campaign_context=campaign_context,
        route=route,
        current_step=current_step,
        show_workflow_steps=show_workflow_steps,
        show_auto_followup=show_auto_followup,
        show_export_followup=show_export_followup,
        manual_review_active=manual_review_active,
        auto_completed=auto_completed,
        cta_state=cta_state,
    )

    if (
        not campaign_context
        and str(route or "").strip().lower() == "auto"
        and str(current_step or "").strip().lower() == "auto_input"
    ):
        try:
            self.workflow_input_title_lbl.configure(text="Wska\u017c katalog obraz\u00f3w")
        except Exception:
            pass
    elif (
        not campaign_context
        and str(route or "").strip().lower() == "manual"
        and str(current_step or "").strip().lower() == "manual_input"
    ):
        try:
            self.workflow_input_title_lbl.configure(text="Wska\u017c katalog obraz\u00f3w")
        except Exception:
            pass
    elif (
        not campaign_context
        and str(route or "").strip().lower() == "manual"
        and str(current_step or "").strip().lower() == "manual_start"
    ):
        try:
            self.workflow_start_title_lbl.configure(text="Utw\u00f3rz XML anotacji")
        except Exception:
            pass
        try:
            self.start_btn.configure(
                text=(
                    "Utwórz XML + boxy pojazdów"
                    if self._manual_vehicle_assist_enabled()
                    else "Utwórz XML anotacji"
                )
            )
        except Exception:
            pass

    self._refresh_workflow_button_styles()
    self._refresh_workflow_step_cards()

    self._workflow_ui_refresh_in_progress = False
    if bool(getattr(self, "_workflow_ui_refresh_pending", False)):
        self._workflow_ui_refresh_pending = False
        try:
            self.frame.after_idle(self._refresh_free_mode_workflow_ui)
        except Exception:
            self._refresh_free_mode_workflow_ui()

def get_campaign_step2_view_model(self) -> Step2ViewModel:
    result = Step2ViewModel(
        stage_key="step2",
        iteration_target="",
        current_step=0,
        step2_status="pending",
        state="locked",
        title="E2. Tablice",
        summary="E2 odblokuje się po zatwierdzeniu katalogu zdjęć wejściowych w E1.",
        details="Najpierw domknij E1.",
        primary_cta=None,
        secondary_cta=None,
    )

    if self._is_free_mode_session_context():
        return result

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return result

        current_step = int(CAMPAIGN.get_current_step() or 0)
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower() or "pending"
        step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower() or "pending"
        iteration_target = self._normalize_campaign_iteration_target_value(CAMPAIGN.get_iteration_target())
    except Exception:
        return result

    summary = result.summary
    details = result.details
    state = result.state
    primary_cta = None
    secondary_cta = None
    title = result.title

    try:
        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
        plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())
    except Exception:
        plate_model_ready = False

    if not iteration_target:
        state = "in_progress" if current_step >= 2 else "locked"
        summary = "Brak wybranego toru iteracji."
        details = (
            "Wybór toru należy do E1. Wróć do E1, wybierz tor tablic albo tor znaków "
            "i dopiero wtedy uruchom etap tablic w E2."
        )
        return Step2ViewModel(
            stage_key="step2",
            iteration_target=iteration_target,
            current_step=current_step,
            step2_status=step2_status,
            state=state,
            title=title,
            summary=summary,
            details=details,
            primary_cta=primary_cta,
            secondary_cta=secondary_cta,
        )

    source_state = self.get_campaign_step2_source_state(iteration_target=iteration_target)
    plate_model_ready = bool(source_state.get("plate_model_ready"))
    title = (
        "E2. Tablice Z2"
        if iteration_target == "plate"
        else "E2. Tablice jako źródło dla znaków"
    )

    if iteration_target == "plate":
        bootstrap = dict(source_state.get("bootstrap") or {})
        source_run_name = str(source_state.get("run_name") or "").strip()
        input_dir = bootstrap.get("input_dir")
        approved_state = self._get_campaign_latest_approved_plate_run_state(input_dir=input_dir)
        approved_images = int(approved_state.get("images_with_plates", 0) or 0)
        approved_total = int(approved_state.get("total_plates", 0) or 0)
        project_approved_images = 0
        project_approved_plates = 0
        try:
            from ..campaign_manager import CAMPAIGN
            approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            project_approved_images = int(approved_stats.get("images", 0) or 0)
            project_approved_plates = int(approved_stats.get("plates", 0) or 0)
        except Exception:
            project_approved_images = 0
            project_approved_plates = 0
        min_plate_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        project_has_ready_plate_set = bool(
            project_approved_plates >= int(min_plate_approval_plates)
        )

        if (
            (current_step >= 4 or step2_status == "approved")
            and not project_has_ready_plate_set
            and approved_state.get("run_dir") is not None
            and int(approved_total or 0) < int(min_plate_approval_plates)
        ):
            state = "needs_attention"
            if approved_total <= 0:
                summary = "Zatwierdzony run Z2 nie zawiera jeszcze poprawnych tablic do dalszej pracy."
                details = (
                    f"Wróć do Z2 i zapisz co najmniej {min_plate_approval_plates} zatwierdzonych tablic, a następnie domknij etap ponownie."
                )
            else:
                summary = f"E2 wymaga jeszcze co najmniej {min_plate_approval_plates} zatwierdzonych tablic. Aktualnie: {approved_total}."
                details = (
                    "Wróć do Z2 i dodaj brakujące oznaczenia tablic, zanim projekt przejdzie do E4T."
                )
            primary_cta = Step2CtaViewModel(label="Wróć do Z2", command_id="return_to_z2")
        elif step2_status == "generated":
            state = "needs_attention"
            summary = "Z2 utworzyło już run anotacji tablic, ale etap E2 nie został jeszcze zatwierdzony."
            details = (
                "Wejdź do Z2, sprawdź lub popraw polygony tablic. "
                "Gdy spełnisz warunki etapu, wrócisz tutaj i zamkniesz E2 badge'em po prawej."
            )
            primary_cta = Step2CtaViewModel(
                label="Sprawdź tablice w Z2",
                command_id="open_z2",
            )
        elif current_step > 2 or step2_status == "approved":
            state = "done"
            if project_has_ready_plate_set and int(approved_total or 0) < int(min_plate_approval_plates):
                summary = "Tor tablic został domknięty na podstawie zatwierdzonego zbioru projektu."
                details = (
                    f"Projekt ma już {project_approved_images} zatwierdzonych obrazów i {project_approved_plates} tablic "
                    "z poprzednich iteracji, więc E2 nie wymaga nowych poprawnych tablic w tej iteracji. "
                    "W torze tablic etap Z3 jest pomijany."
                )
            else:
                summary = "Tor tablic został domknięty i projekt może przejść bezpośrednio do Z4."
                details = "W torze tablic etap Z3 jest pomijany."
        elif bool(source_state.get("has_source")):
            state = "in_progress" if current_step == 2 else "ready"
            summary = "Dla tego zestawu zdjęć wykryto już gotowe anotacje tablic."
            details = (
                f"Z2 może wystartować od runu {source_run_name} zamiast od pustego XML."
                if source_run_name
                else "Z2 może wystartować od istniejącego runu zamiast od pustego XML."
            )
            primary_cta = Step2CtaViewModel(
                label=(
                    "Przygotuj tablice na modelu projektu (Z2)"
                    if (plate_model_ready and not bool(source_state.get("manual_template")))
                    else "Przejdź do anotacji tablic"
                ),
                command_id="open_z2",
            )
        elif current_step == 2:
            state = "in_progress"
            summary = "Ta iteracja pracuje w torze tablic i kieruje do Z2."
            details = (
                "W Z2 przygotujesz ręczną albo automatyczną anotację tablic przed treningiem Pose. "
                "Po domknięciu E2 projekt przejdzie dalej do Z4."
            )
            primary_cta = Step2CtaViewModel(
                label=(
                    "Przygotuj tablice na modelu projektu (Z2)"
                    if plate_model_ready
                    else "Przejdź do anotacji tablic"
                ),
                command_id="open_z2",
            )
    else:
        if bool(source_state.get("ready")):
            source_context = dict(source_state.get("bootstrap") or {})
            source_run_name = str(source_state.get("run_name") or "").strip()
            source_images = int(source_state.get("images_with_plates", 0) or 0)
            source_plates = int(source_state.get("total_plates", 0) or 0)
            if current_step > 2 or step2_status == "approved":
                state = "done"
                summary = "E2 zostało zatwierdzone. Źródła tablic do pracy nad znakami są gotowe."
                source_prefix = (
                    f"Zatwierdzone źródło tablic: {source_run_name}."
                    if source_run_name
                    else "Źródła tablic zostały zatwierdzone."
                )
                details = (
                    f"{source_prefix} Zatwierdzonych obrazów: {source_images}. Zapisanych tablic: {source_plates}. "
                    "Dalsza praca odbywa się już w E3 / Z3. "
                    "Jeśli chcesz, możesz nadal powiększać zbiór tablic w Z2, bo większy zatwierdzony zbiór poprawi kolejne iteracje modelu."
                )
            else:
                state = "ready"
                summary = "Źródło tablic dla tego zestawu zdjęć jest już gotowe."
                source_prefix = (
                    f"Źródło tablic: {source_run_name}."
                    if source_run_name
                    else "Źródło tablic dla tego zestawu zdjęć jest już gotowe."
                )
                details = (
                    f"{source_prefix} Zatwierdzonych obrazów: {source_images}. Zapisanych tablic: {source_plates}. "
                    "Jeśli wszystko jest w porządku, zamknij E2 badge’em po prawej i przejdź dalej do E3. "
                    "Z2 możesz jeszcze otworzyć, jeśli chcesz sprawdzić albo powiększyć zbiór tablic."
                )
                primary_cta = Step2CtaViewModel(
                    label="Sprawdź źródło tablic w Z2",
                    command_id="open_z2_step2_review",
                )
                secondary_cta = None
        elif step2_status == "generated":
            state = "needs_attention"
            summary = "Tablice dla tego zestawu zdjęć są już przygotowane, ale E2 czeka na zatwierdzenie."
            details = (
                "Wejdź do Z2, jeśli chcesz jeszcze sprawdzić albo poprawić tablice. "
                "Gdy źródło będzie gotowe, wrócisz tutaj i przejdziesz dalej badge'em po prawej."
            )
            primary_cta = Step2CtaViewModel(
                label="Sprawdź tablice w Z2",
                command_id="open_z2",
            )
        elif bool(source_state.get("needs_more_tables")):
            source_images = int(source_state.get("images_with_plates", 0) or 0)
            source_plates = int(source_state.get("total_plates", 0) or 0)
            min_char_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
            missing_plates = max(0, min_char_plates - source_plates)
            source_run_name = str(source_state.get("run_name", "") or "").strip()
            state = "needs_attention"
            summary = "Źródło tablic dla toru znaków wymaga jeszcze uzupełnienia."
            source_prefix = f"Źródło tablic: {source_run_name}." if source_run_name else ""
            details = (
                f"{source_prefix} Zatwierdzonych obrazów: {source_images}. Zapisanych tablic: {source_plates}. "
                f"Minimum do wejścia do E3: {min_char_plates} tablic; brakuje {missing_plates}. "
                "Wróć do Z2 i przygotuj więcej tablic, a potem przejdź dalej do E3."
            ).strip()
            primary_cta = Step2CtaViewModel(
                label="Przygotuj więcej tablic w Z2",
                command_id="open_z2_step2_review",
            )
        elif current_step > 2 or step2_status == "approved":
            state = "done"
            summary = "Źródła tablic do pracy nad znakami zostały zatwierdzone."
            details = "E2 jest domkniete. Dalsza praca odbywa się już w E3 / Z3."
        elif current_step == 2:
            state = "in_progress"
            summary = "Ta iteracja pracuje w torze znaków i najpierw kieruje do Z2."
            details = (
                "W Z2 przygotujesz źródło tablic potrzebne do pracy nad znakami. "
                "Po domknięciu E2 projekt przejdzie dalej do E3."
            )
            primary_cta = Step2CtaViewModel(
                label=(
                    "Przygotuj tablice na modelu projektu (Z2)"
                    if plate_model_ready
                    else "Przygotuj tablice w Z2"
                ),
                command_id="open_z2",
            )

        if (
            current_step == 2
            and state == "in_progress"
            and not bool(source_state.get("ready"))
            and not plate_model_ready
        ):
            state = "needs_attention"
            summary = "Tor znaków potrzebuje źródła tablic."
            details = (
                "Aby przejść dalej do E3, przygotuj źródło tablic w Z2. "
                "Możesz zrobić to ręcznie, a jeśli projekt ma model tablic, Z2 wykorzysta go jako przyspieszenie."
            )
            primary_cta = Step2CtaViewModel(
                label="Przygotuj tablice w Z2",
                command_id="open_z2",
            )

    return Step2ViewModel(
        stage_key="step2",
        iteration_target=iteration_target,
        current_step=current_step,
        step2_status=step2_status,
        state=state,
        title=title,
        summary=summary,
        details=details,
        primary_cta=primary_cta,
        secondary_cta=secondary_cta,
    )

def _refresh_manual_review_followup_ui(self, *, from_auto: bool, active_run: bool):
    title_label = getattr(self, "manual_stage_title_lbl", None)
    help_label = getattr(self, "manual_stage_help_lbl", None)
    path_title = getattr(self, "manual_stage_path_title_lbl", None)
    path_row = getattr(self, "manual_stage_path_row", None)
    status_label = getattr(self, "manual_stage_status_lbl", None)
    buttons_row = getattr(self, "manual_stage_buttons_row", None)
    campaign_context = not self._is_free_mode_session_context()
    manual_review_screen = bool(
        (not campaign_context)
        and getattr(self, "_manual_review_active", False)
        and self._coerce_free_mode_screen() == "manual_review"
    )
    compact_followup = bool(active_run or from_auto or campaign_context or manual_review_screen)
    repair_followup = False
    if campaign_context:
        try:
            repair_followup = bool(
                self._is_campaign_char_repair_return_mode()
                or self._is_campaign_plate_step4_repair_return_mode()
            )
        except Exception:
            repair_followup = False
    thematic_prefixes = self._get_z2_thematic_title_prefixes()

    def _open_campaign_auto_annotation_modal() -> None:
        try:
            if getattr(self, "is_processing", False):
                return
            self._set_workflow_route_state("auto", campaign_context=True)
            self.manual_xml_template_var.set(False)
            self._set_workflow_step_state("auto_start", campaign_context=True)
            self._set_auto_vehicle_choice_state(self._get_auto_vehicle_choice(), campaign_context=True)
            self._start_annotation()
        except Exception:
            try:
                self._start_annotation()
            except Exception:
                pass

    if compact_followup:
        current_run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
        current_run_name = (
            str(getattr(current_run_dir, "name", "") or "").strip()
            if current_run_dir is not None
            else ""
        )
        current_input_dir = str(self.input_dir_var.get() or "").strip()
        route = self._get_workflow_route()
        payload_followup_title = ""
        payload_followup_text = ""
        if (not campaign_context) and from_auto:
            try:
                payload_followup_title = str(self.followup_title_lbl.cget("text") or "").strip()
            except Exception:
                payload_followup_title = ""
            try:
                payload_followup_text = str(self.followup_intro_var.get() or "").strip()
            except Exception:
                payload_followup_text = ""
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
            graph_gate_id_for_followup = campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
            graph_origin_gate_id = campaign_gate_id_for_edge(
                graph_context.get("repair_origin_edge_key"),
                graph_context.get("repair_origin_gate_id")
                or graph_context.get("source_graph_gate_id"),
            )
            repair_t07_followup = bool(
                repair_followup
                and graph_gate_id_for_followup == "T04"
                and graph_origin_gate_id == "T06"
            )
        except Exception:
            repair_t07_followup = False
        self._set_widget_packed(title_label, True, anchor=tk.W, fill=tk.X)
        try:
            base_title = (
                "Naprawa źródła tablic dla E3"
                if repair_followup and not repair_t07_followup
                else "Naprawa tablic przed decyzją T06"
                if repair_t07_followup
                else "Korekta wyniku autoanotacji"
                if (campaign_context and from_auto)
                else (payload_followup_title or AUTO_REVIEW_FOLLOWUP_TITLE)
                if from_auto
                else MANUAL_REVIEW_FOLLOWUP_TITLE
                if (not campaign_context and route == "manual")
                else "Anotacja i korekta tablic"
                if (campaign_context and route == "manual")
                else "Aktywny run ręcznej korekty"
                if active_run
                else "Korekta ręczna zakończona?"
            )
            title_text = self._format_z2_thematic_title(
                base_title,
                thematic_prefixes.get("manual_stage"),
            )
            title_label.configure(
                text=title_text
            )
        except Exception:
            pass
        try:
            graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
            graph_gate_id = campaign_gate_id_for_edge(
                graph_context.get("graph_edge_key"),
                graph_context.get("graph_gate_id"),
            )
        except Exception:
            graph_gate_id = ""
        graph_display_gate_id = campaign_visible_gate_id(graph_gate_id) or graph_gate_id
        if graph_gate_id == "T02":
            campaign_help_text = (
                "Kontrolujesz import AT dla bramki T02. Sprawdź ramki tablic na obrazach pasujących "
                "do aktualnego zbioru O i nadaj [OK] tylko poprawnym pozycjom. Po powrocie wrócisz "
                "do T02; dopiero przycisk Zatwierdź na bramce zdecyduje o przejściu dalej do pracy nad znakami."
            )
        elif repair_t07_followup:
            campaign_help_text = (
                "Uzupełnij albo popraw ramki tablic w Z2. Poprawne obrazy oznacz [OK]. "
                "Po wyjściu wrócisz do bramki T06."
            )
        elif repair_followup:
            campaign_help_text = (
                "Uzupełnij brakujące ramki tablic w Z2. Po zapisaniu oznacz poprawne obrazy jako OK "
                "z menu listy i zatwierdź powrót do E3 po prawej."
            )
        elif graph_gate_id == "T05":
            campaign_help_text = (
                f"Pracujesz nad otwarciem bramki {graph_display_gate_id or graph_gate_id}: uzupełnij lub popraw ramki tablic potrzebne "
                "do dalszej pracy nad znakami. Program zapisuje zmiany automatycznie. Obrazy poprawne "
                "oznacz statusem [OK]; ta zatwierdzona pula zasili Z3/PZ2 po powrocie do grafu."
            )
        elif graph_gate_id == "T03":
            campaign_help_text = (
                "Masz otwartą kartę Z2 z załadowanym katalogiem zawierającym plik anotacji XML. "
                "Pracujesz nad otwarciem bramki T04: rysuj nowe ramki tablic albo koryguj istniejące "
                "na podglądzie; program zapisuje zmiany automatycznie. Obrazy uznane za poprawnie "
                "opisane nadaj statusem [OK] na liście wyników. Status [OK] zasila licznik bramki T04 "
                "i po osiągnięciu minimum pozwala zamknąć T04 na mapie kampanii oraz przejść do pracy nad znakami."
            )
        elif graph_gate_id == "T04":
            campaign_help_text = (
                "Masz otwartą kartę Z2 z załadowanym katalogiem zawierającym plik anotacji XML. "
                "Pracujesz nad otwarciem bramki T05: rysuj nowe ramki tablic albo koryguj istniejące "
                "na podglądzie; program zapisuje zmiany automatycznie. Obrazy uznane za poprawnie "
                "opisane nadaj statusem [OK] na liście wyników. Status [OK] zasila licznik bramki T05 "
                "i po osiągnięciu minimum pozwala zamknąć T05 na mapie kampanii."
            )
        else:
            campaign_help_text = (
                "Masz otwartą kartę Z2 z załadowanym katalogiem zawierającym plik anotacji XML. "
                "Rysuj nowe ramki tablic albo koryguj istniejące na podglądzie; program zapisuje "
                "zmiany automatycznie. Obrazy uznane za poprawnie opisane nadaj statusem [OK] na "
                "liście wyników. Status [OK] jest warunkiem otwarcia bramki E2 i powrotu do grafu. "
                "Autoanotację możesz uruchomić jako wsparcie pracy w Z2, jeśli chcesz użyć modelu "
                "tablic do przygotowania lub uzupełnienia ramek; wynik nadal wymaga kontroli."
            )
        if graph_gate_id == "T03":
            campaign_help_text = campaign_help_text.replace("T04", "T03")
        elif graph_gate_id == "T04":
            campaign_help_text = campaign_help_text.replace("T05", "T04")
        elif graph_gate_id == "T05":
            campaign_help_text = campaign_help_text.replace("T06", "T05")
        if repair_t07_followup:
            campaign_help_text = campaign_help_text.replace("T07", "T06")
        self._set_inline_label_state(
            help_label,
            text=(
                campaign_help_text
                if campaign_context
                else (
                    (payload_followup_text or AUTO_REVIEW_FOLLOWUP_TEXT)
                    if from_auto
                    else MANUAL_REVIEW_FOLLOWUP_TEXT
                )
            ),
            tone="muted",
            emphasis=False,
        )
        try:
            self._refresh_bound_label_wraplength(help_label)
        except Exception:
            pass
        self._set_widget_packed(
            help_label,
            True,
            anchor=tk.W,
            fill=tk.X,
            pady=((2, 4) if repair_followup else (0, 6)),
        )
        self._set_widget_packed(path_title, False)
        self._set_widget_packed(path_row, False)
        run_chunks = []
        if current_run_name:
            run_chunks.append(f"Run: {current_run_name}")
        if current_input_dir and not repair_followup:
            run_chunks.append(
                f"Obrazy: {self._format_workspace_relative_path(current_input_dir)}"
            )
        if campaign_context or run_chunks:
            status_tone = "success" if campaign_context else "muted"
            if not campaign_context and current_run_dir is not None:
                try:
                    approval_state = self._get_run_plate_strict_approved_state(current_run_dir)
                except Exception:
                    approval_state = {}
                approved_ready = bool(approval_state.get("ok"))
                approved_images = int(approval_state.get("approved_images", 0) or 0)
                approved_plates = int(approval_state.get("approved_plates", 0) or 0)
                total_images = int(approval_state.get("total_images", 0) or 0)
                total_plates = int(approval_state.get("total_plates", 0) or 0)
                if approved_ready:
                    run_chunks.append(
                        f"Gotowe do datasetu YOLO i wyodrębniania: {approved_images} obrazów [OK] / {approved_plates} tablic."
                    )
                    status_tone = "success"
                else:
                    run_chunks.append(
                        "Dataset YOLO i wyodrębnianie tablic wymagają statusu [OK]. Eksport samych "
                        "anotacji XML jest wyjątkiem: wystarczy, że w runie zapisano co najmniej jedną tablicę."
                    )
                    if total_images > 0 or total_plates > 0:
                        run_chunks.append(
                            f"Teraz: {approved_images} obrazów [OK] / {approved_plates} tablic [OK]; "
                            f"w runie: {total_images} obrazów / {total_plates} tablic."
                        )
                    status_tone = "warning"
            self._set_inline_label_state(
                status_label,
                text=("\n".join(run_chunks) if run_chunks else "Trwa korekta bieżącego runu Z2."),
                tone=status_tone,
                emphasis=False,
            )
            try:
                self._refresh_bound_label_wraplength(status_label)
            except Exception:
                pass
            self._set_widget_packed(
                status_label,
                True,
                anchor=tk.W,
                fill=tk.X,
                pady=((0, 4) if repair_followup else (0, 8)),
            )
        else:
            self._set_widget_packed(status_label, False)
        if campaign_context:
            self._set_widget_packed(
                buttons_row,
                bool(active_run),
                fill=tk.X,
                pady=(0, 2),
            )
            try:
                self.manual_stage_use_btn.configure(
                    text="Uruchom autoanotację",
                    command=_open_campaign_auto_annotation_modal,
                    state=(tk.NORMAL if not self.is_processing else tk.DISABLED),
                )
                if bool(active_run):
                    if str(self.manual_stage_use_btn.winfo_manager()) != "grid":
                        self.manual_stage_use_btn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0))
                    else:
                        self.manual_stage_use_btn.grid_configure(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0))
                elif str(self.manual_stage_use_btn.winfo_manager()) == "grid":
                    self.manual_stage_use_btn.grid_remove()
            except Exception:
                pass
            try:
                if str(self.manual_stage_add_btn.winfo_manager()) == "grid":
                    self.manual_stage_add_btn.grid_remove()
            except Exception:
                pass
        else:
            if manual_review_screen:
                approved_ready = False
                try:
                    approved_ready = bool(
                        current_run_dir is not None
                        and self._get_run_plate_strict_approved_state(current_run_dir).get("ok")
                    )
                except Exception:
                    approved_ready = False
                export_ready = False
                try:
                    export_state = self._build_z2_free_export_status_state()
                    export_ready = bool(export_state.get("dataset_ready") or export_state.get("annotation_ready"))
                except Exception:
                    try:
                        export_ready = bool(self._is_z2_free_export_choice_available())
                    except Exception:
                        export_ready = False
                self._set_widget_packed(buttons_row, True, fill=tk.X, pady=(0, 2))
                try:
                    self.manual_stage_use_btn.configure(
                        text="Wyodrębnij tablice",
                        command=self._open_step3_from_z2_annotation_source,
                        state=(tk.NORMAL if approved_ready and not self.is_processing else tk.DISABLED),
                    )
                    if str(self.manual_stage_use_btn.winfo_manager()) != "grid":
                        self.manual_stage_use_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
                    else:
                        self.manual_stage_use_btn.grid_configure(row=0, column=0, columnspan=1, sticky="ew", padx=(0, 6))
                except Exception:
                    pass
                try:
                    self.manual_stage_add_btn.configure(
                        text="Otwórz eksport",
                        command=self._start_z2_export_choice_flow,
                        state=(tk.NORMAL if export_ready and not self.is_processing else tk.DISABLED),
                    )
                    if str(self.manual_stage_add_btn.winfo_manager()) != "grid":
                        self.manual_stage_add_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
                    else:
                        self.manual_stage_add_btn.grid_configure(row=0, column=1, columnspan=1, sticky="ew", padx=(6, 0))
                except Exception:
                    pass
            else:
                self._set_widget_packed(buttons_row, False)
                try:
                    if str(self.manual_stage_use_btn.winfo_manager()) == "grid":
                        self.manual_stage_use_btn.grid_remove()
                except Exception:
                    pass
                try:
                    if str(self.manual_stage_add_btn.winfo_manager()) == "grid":
                        self.manual_stage_add_btn.grid_remove()
                except Exception:
                    pass
        return

    self._set_widget_packed(title_label, True, anchor=tk.W, fill=tk.X)
    try:
        title_label.configure(
            text=self._format_z2_thematic_title(
                "Stage kolejnej iteracji",
                thematic_prefixes.get("manual_stage"),
            )
        )
    except Exception:
        pass
    self._set_inline_label_state(
        help_label,
        text=(
            "Stage to pomocnicza pula zdjec do kolejnej iteracji recznej. "
            "Po eksporcie moga trafiac tu nieoznaczone obrazy, a recznie mozesz tez "
            "dolozyc nowy zestaw zdjec bez mieszania z gotowym runem."
        ),
        tone="muted",
        emphasis=False,
    )
    self._set_widget_packed(help_label, True, anchor=tk.W, fill=tk.X, pady=(0, 6))
    self._set_widget_packed(path_title, True, anchor=tk.W, fill=tk.X)
    self._set_widget_packed(path_row, True, fill=tk.X, pady=(2, 8))
    self._set_widget_packed(status_label, True, anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_widget_packed(buttons_row, True, fill=tk.X, pady=(0, 2))
    try:
        self.manual_stage_use_btn.configure(
            text="Użyj stage jako wejścia Z2",
            command=self._use_manual_plate_stage_as_input,
        )
        self.manual_stage_add_btn.configure(
            text="Dodaj zdjęcia do stage",
            command=self._add_images_to_manual_plate_stage,
        )
    except Exception:
        pass
    try:
        if str(self.manual_stage_use_btn.winfo_manager()) != "grid":
            self.manual_stage_use_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        else:
            self.manual_stage_use_btn.grid_configure(row=0, column=0, columnspan=1, sticky="ew", padx=(0, 6))
    except Exception:
        pass
    try:
        if str(self.manual_stage_add_btn.winfo_manager()) != "grid":
            self.manual_stage_add_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
    except Exception:
        pass

def apply_theme(self):
    palette = getattr(self.app, "palette", {})
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    for label_name in (
        "workflow_entry_title_lbl",
        "sources_title_lbl",
        "run_title_lbl",
        "followup_title_lbl",
        "manual_stage_title_lbl",
        "export_title_lbl",
        "split_title_lbl",
    ):
        label = getattr(self, label_name, None)
        if label is None or not hasattr(label, "apply_theme"):
            continue
        try:
            label.apply_theme()
        except Exception:
            pass

    try:
        self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
    except Exception:
        pass

    try:
        self.app.style_text_widget(self.log_text, role="console")
    except Exception:
        pass

    try:
        self.app.style_listbox_widget(self.preview_listbox, bordercolor=panel_border)
        self._refresh_preview_list(preserve_selection=True, render_current=False)
    except Exception:
        pass

    try:
        self._refresh_preview_list_legend_theme()
        self._refresh_preview_list_summary()
        self._refresh_approval_breakdown_canvas()
    except Exception:
        pass

    try:
        self.app.style_canvas_widget(
            self.preview_canvas,
            background=palette.get("panel", "#1e1e1e"),
            bordercolor=panel_border
        )
    except Exception:
        pass

    try:
        legend_theme = self._get_preview_legend_theme()
        if getattr(self, "preview_hint_frame", None) is not None:
            self.preview_hint_frame.configure(bg=legend_theme["canvas_bg"])
        self.preview_controls_canvas.configure(bg=legend_theme["canvas_bg"])
        self._refresh_preview_controls_legend()
        self._preview_overlay_dock_render_key = None
        self._place_preview_overlay_dock(force_render=True)
        self._place_preview_campaign_gate_overlay(force_render=True)
    except Exception:
        pass

    try:
        self.left_settings_canvas.configure(
            bg=palette.get("panel", "#252526"),
            highlightthickness=0,
            bd=0,
        )
    except Exception:
        pass

    try:
        self.right_settings_canvas.configure(
            bg=palette.get("panel", "#252526"),
            highlightthickness=0,
            bd=0,
        )
    except Exception:
        pass

    for shell_name in ("left_scroll_shell", "preview_left_list_shell", "right_scroll_shell"):
        shell = getattr(self, shell_name, None)
        if shell is None:
            continue
        try:
            shell.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=panel_border,
                highlightcolor=panel_border,
            )
        except Exception:
            pass

    try:
        workflow_shell_border = blend_hex_colors(
            palette.get("accent", "#4f8de3"),
            panel_border,
            0.62,
        )
        workflow_shell_fill = blend_hex_colors(
            palette.get("surface_info", palette.get("panel", "#252526")),
            palette.get("panel", "#252526"),
            0.80,
        )
        if hasattr(self, "workflow_entry_shell"):
            self.workflow_entry_shell.configure(bg=workflow_shell_border)
        if hasattr(self, "workflow_entry_shell_inner"):
            self.workflow_entry_shell_inner.configure(bg=workflow_shell_fill)
            self.app.style_panel_surface(
                self.workflow_entry_shell_inner,
                background=workflow_shell_fill,
            )
    except Exception:
        pass

    try:
        self.app.style_web_scrollbar(
            self.left_settings_scrollbar,
            track_color=palette.get("panel", "#252526"),
        )
    except Exception:
        pass

    try:
        self.app.style_web_scrollbar(
            self.right_settings_scrollbar,
            track_color=palette.get("panel", "#252526"),
        )
    except Exception:
        pass

    try:
        self._update_device_hint()
    except Exception:
        pass

    try:
        self._refresh_detection_configuration_ui()
    except Exception:
        pass

    try:
        self._refresh_confidence_value_labels()
    except Exception:
        pass

    try:
        self._refresh_workflow_route_cards(refresh_content=False)
    except Exception:
        pass

    try:
        self._refresh_workflow_button_styles()
    except Exception:
        pass

    try:
        self._refresh_workflow_progress_style()
    except Exception:
        pass

    for line in getattr(self, "_left_section_separators", []):
        if not isinstance(line, dict):
            continue
        try:
            host = line.get("host")
            accent = line.get("accent")
            shadow = line.get("shadow")
            if host is not None:
                host.configure(bg=palette.get("panel", "#252526"))
            if accent is not None:
                accent.configure(bg=palette.get("surface_info", palette.get("accent", "#0e639c")))
            if shadow is not None:
                shadow.configure(bg=panel_border)
        except Exception:
            pass

    frame_backgrounds = {
        "start_btn_frame": palette.get("panel", "#252526"),
        "approve_btn_frame": palette.get("panel", "#252526"),
    }
    for frame_name, background in frame_backgrounds.items():
        frame = getattr(self, frame_name, None)
        if frame is None:
            continue
        try:
            frame.configure(bg=background)
        except Exception:
            pass

    inline_label_defaults = {
        "project_paths_info_lbl": ("muted", False),
        "project_paths_rel_lbl": ("muted", False),
        "status_label": ("neutral", True),
        "progress_counts_lbl": ("muted", False),
        "post_annotation_hint_lbl": ("muted", False),
        "plate_export_status_lbl": ("muted", False),
    }
    for label_name, (default_tone, default_emphasis) in inline_label_defaults.items():
        label = getattr(self, label_name, None)
        if label is None or not isinstance(label, tk.Label):
            continue
        try:
            text_value = label.cget("text")
            try:
                if label.cget("textvariable"):
                    text_value = None
            except Exception:
                pass
            self._set_inline_label_state(
                label,
                text=text_value,
                tone=getattr(label, "_inline_tone", default_tone),
                emphasis=getattr(label, "_inline_emphasis", default_emphasis),
            )
        except Exception:
            pass

    try:
        self._refresh_workflow_step_cards()
    except Exception:
        pass

def _merge_pre_run_visible_state_after_auto(self, run_dir: Path) -> int:
    visible_state = dict(getattr(self, "_campaign_auto_pre_run_visible_state", {}) or {})
    visible_annotations = list(visible_state.get("annotations") or [])
    if not visible_annotations:
        return 0

    current_annotations = list(getattr(self, "current_annotations", []) or [])
    current_by_name = {
        str(getattr(ann, "filename", "") or "").strip().lower(): ann
        for ann in current_annotations
        if str(getattr(ann, "filename", "") or "").strip()
    }
    if not current_by_name:
        return 0

    run_result_names: set[str] = set()
    try:
        manifest = self._load_annotation_run_manifest(Path(run_dir))
        run_result_names = {
            str(name or "").strip().lower()
            for name in list((manifest or {}).get("input_scope_filenames") or [])
            if str(name or "").strip()
        }
        if not run_result_names:
            input_dir_text = str((manifest or {}).get("input_dir") or "").strip()
            source_dir_text = str(
                (manifest or {}).get("source_input_dir")
                or (manifest or {}).get("imported_source_input_dir")
                or ""
            ).strip()
            scoped_input = False
            if input_dir_text and source_dir_text:
                try:
                    scoped_input = Path(input_dir_text).resolve() != Path(source_dir_text).resolve()
                except Exception:
                    scoped_input = input_dir_text.rstrip("\\/").lower() != source_dir_text.rstrip("\\/").lower()
            if scoped_input:
                try:
                    run_result_names = {
                        Path(path).name.strip().lower()
                        for path in get_image_files(Path(input_dir_text))
                        if Path(path).name.strip()
                    }
                except Exception:
                    run_result_names = set()
    except Exception:
        run_result_names = set()
    try:
        xml_path = Path(getattr(self, "current_annotation_xml_path", None) or (Path(run_dir) / "annotations.xml"))
        if not run_result_names:
            run_result_names = {
                str(getattr(ann, "filename", "") or "").strip().lower()
                for ann in self._parse_cvat_preview_annotations(xml_path)
                if str(getattr(ann, "filename", "") or "").strip()
            }
    except Exception:
        if not run_result_names:
            run_result_names = set()
    if not run_result_names:
        run_result_names = set(current_by_name.keys())

    def _lookup_path(image_map: dict, filename: str):
        if not isinstance(image_map, dict) or not filename:
            return None
        if filename in image_map:
            return image_map.get(filename)
        lower_name = filename.lower()
        for key, value in image_map.items():
            if str(key or "").strip().lower() == lower_name:
                return value
        return None

    old_image_map = dict(visible_state.get("image_map") or {})
    current_image_map = dict(getattr(self, "_preview_image_path_map", {}) or {})
    merged_annotations: list[ImageAnnotation] = []
    merged_image_map: dict[str, Path] = {}
    seen_names: set[str] = set()
    restored_count = 0

    for old_ann in visible_annotations:
        filename = str(getattr(old_ann, "filename", "") or "").strip()
        if not filename:
            continue
        name_key = filename.lower()
        replacement = current_by_name.get(name_key) if name_key in run_result_names else None
        if replacement is not None:
            merged_ann = copy.deepcopy(replacement)
        else:
            merged_ann = copy.deepcopy(old_ann)
            restored_count += 1
        merged_ann.filename = filename
        merged_annotations.append(merged_ann)
        seen_names.add(name_key)

        image_path = _lookup_path(old_image_map, filename) or _lookup_path(current_image_map, filename)
        if image_path:
            try:
                merged_image_map[filename] = Path(image_path)
            except Exception:
                pass

    for ann in current_annotations:
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename or filename.lower() in seen_names:
            continue
        merged_annotations.append(copy.deepcopy(ann))
        image_path = _lookup_path(current_image_map, filename)
        if image_path:
            try:
                merged_image_map[filename] = Path(image_path)
            except Exception:
                pass

    if restored_count <= 0:
        return 0

    self.current_annotations = merged_annotations
    if merged_image_map:
        self._preview_image_path_map = merged_image_map
    raw_input_dir = str(visible_state.get("input_dir") or "").strip()
    if raw_input_dir:
        try:
            input_dir = Path(raw_input_dir)
            self.current_input_dir = input_dir
            self.input_dir_var.set(str(input_dir))
            self.plate_dataset_images_var.set(str(input_dir))
        except Exception:
            pass
    try:
        self._invalidate_preview_runtime_caches()
    except Exception:
        pass

    try:
        xml_path = Path(getattr(self, "current_annotation_xml_path", None) or (Path(run_dir) / "annotations.xml"))
        if CVATExporter().export(
            self.current_annotations,
            xml_path,
            include_confidence=True,
            only_successful=False,
        ):
            _images_with_plates, total_plates = self._count_plate_annotations(self.current_annotations)
            self.current_annotation_xml_path = xml_path
            self.current_annotation_run_dir = Path(run_dir)
            self.last_staging_run_dir = Path(run_dir)
            self._update_annotation_run_manifest(
                Path(run_dir),
                result_total_images=len(self.current_annotations),
                result_total_plates=total_plates,
                full_view_restored_after_scope=True,
                full_view_restored_count=restored_count,
                source_input_dir=str(getattr(self, "current_input_dir", "") or ""),
                resume_preview_saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )
            logger.info(
                "[Z2 AUTO SCOPE] Przywrócono pełną listę po autoanotacji zakresu: restored=%s total=%s plates=%s",
                restored_count,
                len(self.current_annotations),
                total_plates,
            )
    except Exception as exc:
        logger.debug(f"Nie udało się utrwalić pełnej listy po autoanotacji zakresu Z2: {exc}")

    return restored_count


def _finalize_successful_annotation_run_ui(self, run_dir: Path, *, manual_template: bool = False) -> None:
    try:
        pending_summary_snapshot = dict(getattr(self, "_campaign_pending_batch_summary", {}) or {})
        reuse_filenames_snapshot = set(getattr(self, "_campaign_reuse_manual_filenames", set()) or set())
        reuse_summary_snapshot = dict(getattr(self, "_campaign_reuse_manual_summary", {}) or {})
        auto_manual_overlay_snapshot = dict(getattr(self, "_campaign_auto_manual_overlay_bundle", {}) or {})
        pre_run_snapshot = dict(getattr(self, "_campaign_auto_pre_run_snapshot", {}) or {})
        approved_filenames_snapshot = set(getattr(self, "_pending_preview_approved_filenames", set()) or set())
        if manual_template and not self._is_free_mode_session_context():
            try:
                from ..campaign_manager import CAMPAIGN

                iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower() or "plate"
            except Exception:
                iteration_target = "plate"
            if self._open_existing_run_for_campaign_review(
                run_dir=run_dir,
                iteration_target=iteration_target,
                manual_template=True,
            ):
                self._queue_free_mode_session_save()
                return

        if manual_template and self._is_free_mode_session_context():
            self.current_annotation_run_dir = run_dir
            self.current_annotation_xml_path = run_dir / "annotations.xml"
            self.last_staging_run_dir = run_dir
            self._load_plate_dataset_context_from_run(run_dir, force_images_update=True)
            self._refresh_preview_workspace_visibility(manual_review_active=False)
            try:
                self._populate_preview_list(batch_size=150)
            except Exception:
                try:
                    self._refresh_preview_list(preserve_selection=False, render_current=True)
                except Exception:
                    pass
            try:
                if self.current_annotations and self.current_preview_index is None:
                    self._select_preview_index(0, reset_view=True)
            except Exception:
                pass
            self._refresh_preview_workspace_visibility(manual_review_active=False)
            self._refresh_step2_action_states()
            self._queue_free_mode_session_save()
            return

        restored_preview = False
        if not manual_template:
            # Publish the list only after restoring and merging the entire result.
            restored_preview = self._restore_preview_from_annotation_run(run_dir, defer_ui_restore=True)
            if restored_preview:
                self._campaign_pending_batch_summary = pending_summary_snapshot
                self._campaign_reuse_manual_filenames = reuse_filenames_snapshot
                self._campaign_reuse_manual_summary = reuse_summary_snapshot
                if approved_filenames_snapshot:
                    self._preview_approved_filenames = set(
                        self._get_preview_approved_filenames()
                    ) | set(
                        str(name or "").strip().lower()
                        for name in approved_filenames_snapshot
                        if str(name or "").strip()
                    )
                    if not self._is_free_mode_session_context():
                        self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
                if auto_manual_overlay_snapshot:
                    try:
                        merged_overlay_count = self._merge_preview_annotation_bundle(auto_manual_overlay_snapshot)
                    except Exception as e:
                        logger.debug(f"Nie udało się przywrócić ręcznych polygonów po runie auto Z2: {e}")
                        merged_overlay_count = 0
                    if merged_overlay_count > 0:
                        try:
                            self._campaign_pending_batch_summary["current_manual_count"] = len(
                                self._collect_preview_current_iteration_manual_filenames()
                            )
                        except Exception:
                            pass
                try:
                    self._sync_campaign_pending_batch_summary_from_preview(
                        hidden_project_approved_count=len(
                            set(getattr(self, "_campaign_hidden_project_approved_filenames", set()) or set())
                        ),
                    )
                except Exception:
                    pass
                if pre_run_snapshot:
                    try:
                        existing_names = {
                            str(getattr(ann, "filename", "") or "").strip()
                            for ann in list(self.current_annotations or [])
                            if str(getattr(ann, "filename", "") or "").strip()
                        }
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
                        missing_snapshot = {
                            name: payload
                            for name, payload in pre_run_snapshot.items()
                            if (
                                str(name or "").strip()
                                and str(name or "").strip() not in existing_names
                                and str(name or "").strip().lower() not in project_approved_filenames
                                and not (
                                    isinstance(payload, tuple)
                                    and len(payload) >= 2
                                    and str(
                                        self._build_campaign_source_image_key(Path(payload[1]))
                                    ).strip() in project_approved_source_keys
                                )
                            )
                        }
                        if missing_snapshot:
                            self._merge_preview_annotation_bundle(missing_snapshot)
                            self._append_z2_trace(
                                "post-auto-restore-missing",
                                f"restored={len(missing_snapshot)} total={len(self.current_annotations or [])}",
                            )
                    except Exception as e:
                        logger.debug(f"Nie udało się przywrócić brakujących obrazów po auto Z2: {e}")
                if approved_filenames_snapshot:
                    try:
                        self._persist_preview_approved_filenames()
                    except Exception:
                        pass

        if not restored_preview and manual_template:
            if (
                manual_template
                and bool(getattr(self, "_current_run_manual_vehicle_assist", False))
                and self._should_use_async_preview_list_population()
            ):
                self._populate_preview_list_async(
                    preserve_selection=False,
                    render_current=False,
                    batch_size=150,
                )
                self._set_post_annotation_hint(
                    "Run Z2 z preboxingiem pojazdow jest gotowy. Wybierz obraz na liscie po prawej, aby zaladowac podglad i rozpoczac korekte.",
                    "success",
                )
            else:
                self._populate_preview_list()
            self._load_plate_dataset_context_from_run(run_dir, force_images_update=True)
            if approved_filenames_snapshot:
                try:
                    self._preview_approved_filenames = set(
                        str(name or "").strip().lower()
                        for name in approved_filenames_snapshot
                        if str(name or "").strip()
                    )
                    if not self._is_free_mode_session_context():
                        self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
                    self._persist_preview_approved_filenames()
                except Exception:
                    pass

        if not manual_template:
            try:
                _merge_pre_run_visible_state_after_auto(self, run_dir)
            except Exception as exc:
                logger.debug(f"Nie udało się scalić listy po autoanotacji zakresu Z2: {exc}")
            self._mark_auto_plate_origin_for_annotations(self.current_annotations)
            self._invalidate_preview_runtime_caches()
            if approved_filenames_snapshot:
                self._preview_approved_filenames = set(self._get_preview_approved_filenames()) | {
                    str(name).strip().lower() for name in approved_filenames_snapshot if str(name).strip()
                }
                if not self._is_free_mode_session_context():
                    self._campaign_pending_approved_filenames = set(self._preview_approved_filenames)
                self._persist_preview_approved_filenames()
            # Same filenames/count do not mean unchanged rows. Also cancel any
            # queued population that still holds the pre-detection annotations.
            self._populate_preview_list_async(
                preserve_selection=True,
                render_current=True,
                invalidate_runtime=False,
                rebuild_state_cache=True,
                recolor_rows=True,
                lightweight_summary=True,
            )
            if not restored_preview:
                self._load_plate_dataset_context_from_run(run_dir, force_images_update=True)
            logger.info("[Z2 AUTO] Odświeżanie znaczników i kolorów po detekcji: %s obrazów", len(self.current_annotations or []))

        if not self._is_free_mode_session_context():
            try:
                sync_input_dir = self._resolve_existing_dir(getattr(self, "current_input_dir", None))
                if sync_input_dir is None:
                    manifest = self._load_annotation_run_manifest(run_dir)
                    sync_input_dir = self._resolve_existing_dir(
                        manifest.get("source_input_dir")
                        or manifest.get("input_dir")
                        or manifest.get("imported_source_input_dir")
                    )
                self.current_annotation_run_dir = run_dir
                self.current_annotation_xml_path = run_dir / "annotations.xml"
                self.last_staging_run_dir = run_dir
                try:
                    self.plate_dataset_run_var.set(str(run_dir))
                except Exception:
                    pass
                self._sync_campaign_iteration_artifact_registry(
                    run_dir=run_dir,
                    xml_path=run_dir / "annotations.xml",
                    input_dir=sync_input_dir,
                )
                self._save_campaign_project_snapshot()
            except Exception as e:
                logger.debug(f"Nie udało się utrwalić kampanijnego runu Z2 po zakończeniu anotacji: {e}")

        self._refresh_plate_dataset_export_sources()
        self._refresh_preview_list_summary()
        self._refresh_step2_action_states()
        self._queue_free_mode_session_save()
    except Exception:
        logger.exception("Blad finalizacji UI po zakonczonym runie Z2")
    finally:
        self._campaign_auto_manual_overlay_bundle = {}
        self._campaign_auto_pre_run_snapshot = {}
        self._campaign_auto_pre_run_visible_state = {}
        self._campaign_auto_pre_run_context = {}
        self._pending_preview_approved_filenames = set()

def _apply_workflow_step_widget_style(self, widget, kind: str, style_name: str, background: str):
    if widget is None or not style_name:
        return

    style = getattr(getattr(self, "app", None), "style", None)
    if style is None:
        return

    palette = getattr(self.app, "palette", {})
    fg = palette.get("fg", "#f3f3f3")
    field = palette.get("field", "#3c3c3c")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    panel = palette.get("panel", "#252526")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    success = palette.get("success", "#4ec9b0")
    success_surface = palette.get("surface_success", blend_hex_colors(background, success, 0.24))
    muted_dim = palette.get("muted_dim", palette.get("muted", "#8c8c8c"))
    field_bg = blend_hex_colors(background, field, 0.58)
    disabled_field_bg = blend_hex_colors(background, panel, 0.35)
    control_arrow = success
    control_arrow_disabled = muted_dim

    try:
        if kind == "frame":
            style.configure(style_name, background=background)
        elif kind == "checkbutton":
            style.configure(
                style_name,
                background=background,
                foreground=fg,
                focuscolor=background,
                indicatorcolor=field,
            )
            style.map(
                style_name,
                background=[("active", background), ("disabled", background)],
                foreground=[("disabled", muted_dim)],
                indicatorcolor=[
                    ("selected", success),
                    ("active", field),
                    ("!selected", field),
                    ("disabled", panel_alt),
                ],
            )
        elif kind == "radiobutton":
            style.configure(
                style_name,
                background=background,
                foreground=fg,
                focuscolor=background,
                indicatorcolor=field,
            )
            style.map(
                style_name,
                background=[("active", background), ("disabled", background)],
                foreground=[("disabled", muted_dim)],
                indicatorcolor=[
                    ("selected", success),
                    ("active", field),
                    ("!selected", field),
                    ("disabled", panel_alt),
                ],
            )
        elif kind == "scale":
            self.app.style_ttk_scale_widget(widget, background=background, base_style=style_name)
        elif kind == "entry":
            style.configure(
                style_name,
                fieldbackground=field_bg,
                foreground=fg,
                bordercolor=border,
                lightcolor=border,
                darkcolor=border,
            )
            style.map(
                style_name,
                fieldbackground=[
                    ("readonly", field_bg),
                    ("disabled", disabled_field_bg),
                ],
                foreground=[
                    ("readonly", fg),
                    ("disabled", muted_dim),
                ],
                selectbackground=[
                    ("readonly", success_surface),
                    ("disabled", disabled_field_bg),
                ],
                selectforeground=[
                    ("readonly", fg),
                    ("disabled", muted_dim),
                ],
            )
        elif kind == "combobox":
            style.configure(
                style_name,
                fieldbackground=field_bg,
                background=field_bg,
                foreground=fg,
                bordercolor=border,
                lightcolor=border,
                darkcolor=border,
                arrowsize=14,
                arrowcolor=control_arrow,
            )
            style.map(
                style_name,
                fieldbackground=[
                    ("readonly", field_bg),
                    ("disabled", disabled_field_bg),
                ],
                background=[
                    ("readonly", field_bg),
                    ("disabled", disabled_field_bg),
                ],
                selectbackground=[
                    ("readonly", success_surface),
                    ("disabled", disabled_field_bg),
                ],
                selectforeground=[
                    ("readonly", fg),
                    ("disabled", muted_dim),
                ],
                foreground=[
                    ("readonly", fg),
                    ("disabled", muted_dim),
                ],
                arrowcolor=[
                    ("readonly", control_arrow),
                    ("active", control_arrow),
                    ("disabled", control_arrow_disabled),
                ],
            )
        elif kind == "title_label":
            style.configure(
                style_name,
                background=background,
                foreground=fg,
                font=("Segoe UI Semibold", 10),
            )
            style.map(
                style_name,
                background=[("disabled", background)],
                foreground=[("disabled", muted_dim)],
            )
        elif kind == "label":
            style.configure(
                style_name,
                background=background,
                foreground=fg,
                font=("Segoe UI", 10),
            )
            style.map(
                style_name,
                background=[("disabled", background)],
                foreground=[("disabled", muted_dim)],
            )
        elif kind == "muted_label":
            style.configure(
                style_name,
                background=background,
                foreground=palette.get("muted", "#9a9a9a"),
                font=("Segoe UI", 10),
            )
            style.map(
                style_name,
                background=[("disabled", background)],
                foreground=[("disabled", muted_dim)],
            )
        elif kind == "progressbar":
            trough = blend_hex_colors(background, field, 0.6)
            fill = blend_hex_colors(success, background, 0.12)
            style.configure(
                style_name,
                troughcolor=trough,
                background=fill,
                lightcolor=fill,
                darkcolor=fill,
                bordercolor=border,
            )
            style.map(
                style_name,
                troughcolor=[("disabled", trough)],
                background=[("disabled", disabled_field_bg)],
                lightcolor=[("disabled", disabled_field_bg)],
                darkcolor=[("disabled", disabled_field_bg)],
            )
        else:
            return

        widget.configure(style=style_name)
    except Exception:
        pass

def _render_compact_info_table(
    self,
    host,
    rows,
    *,
    default_value_tone: str = "default",
    reuse_existing: bool = False,
    show_header: bool = True,
):
    if host is None:
        return

    normalized_rows = [
        (
            str(label or "").strip(),
            str(value or "").strip(),
            str(tone or default_value_tone).strip().lower() or default_value_tone,
        )
        for label, value, tone in (rows or [])
        if str(label or "").strip() or str(value or "").strip()
    ]
    if not normalized_rows:
        return

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("border", "#3a3a3a")
    value_col_width = 118
    row_label_widgets = []
    label_signature = tuple(label for label, _value, _tone in normalized_rows)

    def _sync_table_wrap(_event=None):
        try:
            host_width = int(host.winfo_width() or 0)
        except Exception:
            host_width = 0
        if host_width <= 1:
            try:
                host_width = int(host.winfo_reqwidth() or 0)
            except Exception:
                host_width = 0
        target_wrap = max(140, int(host_width) - int(value_col_width) - 28)
        for widget in list(row_label_widgets):
            try:
                widget.configure(wraplength=target_wrap)
            except Exception:
                continue

    cache = getattr(host, "_compact_table_cache", None)
    if (
        reuse_existing
        and isinstance(cache, dict)
        and tuple(cache.get("labels") or ()) == label_signature
        and len(list(cache.get("rows") or [])) == len(normalized_rows)
    ):
        try:
            for row_widgets, (_label_text, value_text, tone) in zip(list(cache.get("rows") or []), normalized_rows):
                value_widget = row_widgets.get("value")
                if value_widget is not None:
                    try:
                        current_text = str(value_widget.cget("text") or "")
                    except Exception:
                        current_text = ""
                    if current_text != value_text:
                        value_widget.configure(text=value_text)
                    if str(getattr(value_widget, "_compact_table_tone", "") or "") != str(tone):
                        self._set_inline_label_state(value_widget, tone=tone, emphasis=True)
                        try:
                            setattr(value_widget, "_compact_table_tone", str(tone))
                        except Exception:
                            pass
            return
        except Exception:
            pass

    try:
        children = list(host.winfo_children())
    except Exception:
        children = []
    for child in children:
        try:
            child.destroy()
        except Exception:
            pass

    try:
        host.configure(style="Panel.TFrame")
    except Exception:
        pass

    if show_header:
        header = tk.Frame(
            host,
            bg=panel_alt,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
        )
        header.pack(fill=tk.X, pady=(0, 2))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, minsize=value_col_width)

        header_left = tk.Label(
            header,
            text="Pozycja",
            anchor="w",
            justify=tk.LEFT,
            bg=panel_alt,
            fg=palette.get("muted", "#b0b0b0"),
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=3,
            font=("Segoe UI", 9, "bold"),
        )
        header_left.grid(row=0, column=0, sticky="ew")
        row_label_widgets.append(header_left)

        header_right = tk.Label(
            header,
            text="Licznik",
            anchor="e",
            justify=tk.RIGHT,
            bg=panel_alt,
            fg=palette.get("muted", "#b0b0b0"),
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=3,
            font=("Segoe UI", 9, "bold"),
        )
        header_right.grid(row=0, column=1, sticky="e")

    cache_rows = []
    for label_text, value_text, tone in normalized_rows:
        row = tk.Frame(
            host,
            bg=panel_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
        )
        row.pack(fill=tk.X, pady=(0, 2))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, minsize=value_col_width)

        label_widget = tk.Label(
            row,
            text=label_text,
            anchor="nw",
            justify=tk.LEFT,
            bg=panel_bg,
            fg=palette.get("fg", "#f3f3f3"),
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=2,
            wraplength=max(140, int(host.winfo_width() or 0) - value_col_width - 28),
        )
        label_widget.grid(row=0, column=0, sticky="nsew")
        row_label_widgets.append(label_widget)

        value_widget = tk.Label(
            row,
            text=value_text,
            anchor="ne",
            justify=tk.RIGHT,
            bg=panel_bg,
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=2,
            font=("Segoe UI", 9, "bold"),
        )
        value_widget.grid(row=0, column=1, sticky="ne")
        self._set_inline_label_state(value_widget, tone=tone, emphasis=True)
        try:
            setattr(value_widget, "_compact_table_tone", str(tone))
        except Exception:
            pass
        cache_rows.append({"label": label_widget, "value": value_widget})

    try:
        if not bool(getattr(host, "_compact_table_wrap_bound", False)):
            host.bind("<Configure>", _sync_table_wrap, add="+")
            setattr(host, "_compact_table_wrap_bound", True)
    except Exception:
        pass
    try:
        setattr(host, "_compact_table_cache", {"labels": label_signature, "rows": cache_rows})
    except Exception:
        pass
    try:
        self.frame.after_idle(_sync_table_wrap)
    except Exception:
        pass

def _refresh_free_mode_manual_right_panel(self, *, reuse_existing_tables: bool = False) -> bool:
    if not self._should_show_free_mode_manual_right_panel():
        try:
            self.approve_context_var.set("")
        except Exception:
            pass
        try:
            self.approve_gate_hint_var.set("")
        except Exception:
            pass
        try:
            self.approve_hint_title_var.set("")
        except Exception:
            pass
        try:
            self.approve_breakdown_title_var.set("")
            self.approve_breakdown_var.set("")
        except Exception:
            pass
        for widget_name in (
            "approve_context_box",
            "approve_hint_box",
            "approve_breakdown_box",
            "approve_hint_table_frame",
            "approve_breakdown_table_frame",
        ):
            try:
                self._set_widget_packed(getattr(self, widget_name, None), False)
            except Exception:
                pass
        return False

    state = self._build_z2_free_export_status_state()
    run_dir = state.get("run_dir")
    approved_images = int(state.get("approved_images", 0) or 0)
    approved_plates = int(state.get("approved_plates", 0) or 0)
    total_plates = int(state.get("total_plates", 0) or 0)
    images_with_plates = int(state.get("images_with_plates", 0) or 0)
    skipped_images = int(state.get("skipped_images", 0) or 0)
    dataset_min_approved_plates = int(state.get("dataset_min_approved_plates", 0) or 0)
    dataset_missing_approved_plates = int(state.get("dataset_missing_approved_plates", 0) or 0)
    dataset_gate_ready = bool(state.get("dataset_gate_ready"))
    dataset_ready = bool(state.get("dataset_ready"))
    annotation_ready = bool(state.get("annotation_ready"))
    quality_info = dict(state.get("quality_info") or {})
    quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
    quality_tone = str(quality_info.get("tone", "warning") or "warning").strip().lower()
    quality_ranges = (
        f"średni {int(quality_info.get('average_min', 0) or 0)}+ / "
        f"dobry {int(quality_info.get('good_min', 0) or 0)}+"
    )
    ready = bool(dataset_ready or annotation_ready)
    has_saved_annotations = bool(total_plates > 0)
    has_any_ok = bool(approved_images > 0 or approved_plates > 0)
    route = ""
    try:
        route = str(self._get_workflow_route() or "").strip().lower()
    except Exception:
        route = ""

    show_context_box = True
    if run_dir is None:
        if route == "auto":
            context_text = (
                "Najpierw wskaż katalog obrazów i uruchom autoanotację. "
                "Po utworzeniu runu Z2 pokaże tutaj liczniki, status pozycji [OK] "
                "oraz możliwości eksportu lub wyodrębniania tablic."
            )
        else:
            context_text = (
                "Najpierw utwórz XML anotacji. Dopiero wtedy Z2 będzie miało run, "
                "który można później wyodrębniać albo eksportować jako dataset tablic."
            )
    else:
        run_label = "autoanotacji" if route == "auto" else "anotacji"

    if run_dir is None:
        pass
    elif annotation_ready and not has_any_ok:
        context_text = (
            f"Run {run_label} ma zapisane anotacje tablic.\n"
            "Eksport anotacji XML jest dostępny bez statusu [OK]. Dataset YOLO Pose i wyodrębnianie tablic "
            f"wymagają minimum {dataset_min_approved_plates} tablic [OK]; brakuje jeszcze {dataset_missing_approved_plates}."
        )
    elif annotation_ready and not dataset_ready:
        context_text = (
            f"Run {run_label} ma zapisane anotacje tablic i co najmniej jedną pozycję [OK].\n"
            f"Do eksportu datasetu YOLO Pose brakuje jeszcze {dataset_missing_approved_plates} "
            f"tablic [OK] z wymaganego minimum {dataset_min_approved_plates}."
        )
    elif ready:
        context_text = (
            f"Run {run_label} jest gotowy.\n"
            f"Zatwierdzone [OK]: {approved_images} obrazów / {approved_plates} tablic.\n"
            "Możesz eksportować dataset YOLO Pose ze splitem albo pakiet anotacji XML bez splitu."
        )
    else:
        context_text = (
            f"Run {run_label} jest aktywny.\n"
            f"Zapisane anotacje: {images_with_plates} obrazów / {int(total_plates or 0)} tablic. "
            "Poniżej rozdzielono eksport anotacji XML i dataset YOLO Pose."
        )

    status_tone = "success" if run_dir is not None else "info"
    try:
        self.approve_btn_row.configure(text=(" Status pracy Z2 " if run_dir is None else " Status runu anotacji "))
    except Exception:
        pass
    try:
        self.approve_context_var.set(context_text)
        self._set_inline_label_state(self.approve_context_lbl, tone=status_tone, emphasis=True)
        self._set_approve_context_box_state(status_tone)
        self._set_widget_packed(self.approve_context_box, bool(show_context_box), fill=tk.X, pady=(0, 10))
    except Exception:
        pass
    dataset_rows = []
    annotation_rows = []
    if run_dir is not None:
        dataset_rows = [
            ("Pozycje [OK]", f"{approved_images} obrazów / {approved_plates} tablic", "success" if dataset_gate_ready else "warning"),
            (
                "Status bramki",
                "OTWARTA" if dataset_gate_ready else "ZAMKNIĘTA",
                "success" if dataset_ready else "warning",
            ),
            ("Jakość zbioru", quality_label, quality_tone),
            ("Progi jakości", quality_ranges, "info"),
        ]
        if not dataset_gate_ready:
            dataset_rows.insert(2, ("Brakuje do minimum", f"{dataset_missing_approved_plates} tablic [OK]", "warning"))
        dataset_rows.append(("Pominięte w YOLO", f"{skipped_images} obrazów bez [OK]", "muted" if skipped_images == 0 else "warning"))
    if run_dir is not None:
        annotation_rows = [
            ("Status eksportu XML", "DOSTĘPNY" if annotation_ready else "BRAK ZAPISANYCH TABLIC", "success" if annotation_ready else "warning"),
            ("Zapisane anotacje", f"{images_with_plates} obrazów / {total_plates} tablic", "info"),
            ("Próg [OK]", "nie wymagany", "muted"),
            ("Split train/val/test", "nie dotyczy", "muted"),
        ]
    try:
        self.approve_hint_title_var.set("Dataset YOLO Pose")
        self.approve_gate_hint_var.set("")
        self._render_compact_info_table(
            getattr(self, "approve_hint_table_frame", None),
            dataset_rows,
            default_value_tone="muted",
            reuse_existing=bool(reuse_existing_tables),
            show_header=False,
        )
        self._set_widget_packed(
            getattr(self, "approve_hint_title_lbl", None),
            bool(dataset_rows),
            fill=tk.X,
            before=getattr(self, "approve_hint_table_frame", None),
        )
        self._set_widget_packed(
            getattr(self, "approve_hint_table_frame", None),
            bool(dataset_rows),
            fill=tk.X,
            pady=(6, 8),
            before=getattr(self, "approve_gate_hint_lbl", None),
        )
        self._set_widget_packed(getattr(self, "approve_gate_hint_lbl", None), False)
        self._set_approve_hint_box_state("success" if dataset_ready else "warning")
        self._set_widget_packed(self.approve_hint_box, bool(dataset_rows), fill=tk.X, pady=(0, 10))
    except Exception:
        pass
    try:
        self.approve_breakdown_title_var.set("Anotacje XML bez datasetu")
        self.approve_breakdown_var.set("")
        for widget_name in ("approve_breakdown_lbl", "approve_breakdown_canvas"):
            widget = getattr(self, widget_name, None)
            if widget is not None and str(widget.winfo_manager()):
                try:
                    widget.pack_forget()
                except Exception:
                    pass
        self._render_compact_info_table(
            getattr(self, "approve_breakdown_table_frame", None),
            annotation_rows,
            default_value_tone="muted",
            reuse_existing=bool(reuse_existing_tables),
            show_header=False,
        )
        self._set_widget_packed(
            getattr(self, "approve_breakdown_title_lbl", None),
            bool(annotation_rows),
            fill=tk.X,
            before=getattr(self, "approve_breakdown_table_frame", None),
        )
        self._set_widget_packed(
            getattr(self, "approve_breakdown_table_frame", None),
            bool(annotation_rows),
            fill=tk.X,
            pady=(6, 8),
        )
        palette = getattr(self.app, "palette", {}) or {}
        panel_bg = palette.get("panel", "#252526")
        border = palette.get("success", "#2ecc71") if annotation_ready else palette.get("warning", "#f39c12")
        breakdown_box = getattr(self, "approve_breakdown_box", None)
        if breakdown_box is not None:
            breakdown_box.configure(bg=panel_bg, highlightbackground=border, highlightcolor=border)
        self._set_inline_label_state(
            getattr(self, "approve_breakdown_title_lbl", None),
            tone=("success" if annotation_ready else "warning"),
            emphasis=True,
        )
        self._set_widget_packed(self.approve_breakdown_box, bool(annotation_rows), fill=tk.X, pady=(0, 10))
    except Exception:
        try:
            self._set_widget_packed(getattr(self, "approve_breakdown_box", None), False)
        except Exception:
            pass
    for button_name in ("approve_btn", "return_to_campaign_right_btn"):
        try:
            button = getattr(self, button_name, None)
            if button is not None and str(button.winfo_manager()) == "pack":
                button.pack_forget()
        except Exception:
            pass
    return True

def _refresh_manual_plate_stage_ui(self):
    section = getattr(self, "manual_stage_section", None)
    if section is None:
        return
    campaign_context = not self._is_free_mode_session_context()
    if (
        not campaign_context
        and bool(getattr(self, "_manual_review_active", False))
        and self._coerce_free_mode_screen() == "manual_review"
    ):
        self._refresh_manual_review_followup_ui(
            from_auto=bool(getattr(self, "_manual_review_from_auto", False)),
            active_run=True,
        )
        return

    manual_enabled = self._manual_xml_template_enabled()
    separator = getattr(self, "manual_stage_separator", None)
    if campaign_context:
        if manual_enabled:
            if separator is not None and not str(separator.winfo_manager()):
                separator.pack(fill=tk.X, pady=(18, 20))
            if not str(section.winfo_manager()):
                pack_kwargs = {"fill": tk.X}
                if separator is not None:
                    pack_kwargs["before"] = separator
                section.pack(**pack_kwargs)
        else:
            if separator is not None and str(separator.winfo_manager()):
                separator.pack_forget()
            if str(section.winfo_manager()):
                section.pack_forget()
            return

    stage_dir = self._get_manual_plate_stage_dir()
    stage_images_dir = stage_dir / "images"
    manifest = self._load_manual_plate_stage_manifest()
    stage_images = get_image_files(stage_images_dir)
    stage_count = len(stage_images)
    current_is_stage = self._is_manual_plate_stage_input(self.input_dir_var.get())

    self.manual_stage_dir_var.set(str(stage_images_dir))
    try:
        self.manual_stage_title_lbl.configure(
            text=("Pula następnej iteracji" if campaign_context else "Stage kolejnej iteracji")
        )
    except Exception:
        pass
    try:
        self.manual_stage_path_title_lbl.configure(
            text=(
                "Folder puli następnej iteracji:"
                if campaign_context
                else "Folder stage (kolejna pula do anotacji):"
            )
        )
    except Exception:
        pass
    self._set_inline_label_state(
        self.manual_stage_help_lbl,
        text=(
            "Tutaj trafiają obrazy odłożone na później. Program zasila tę pulę automatycznie po eksporcie, "
            "a pojedyncze zdjęcia możesz odkładać przyciskiem pod podglądem."
            if campaign_context
            else "Stage to pomocnicza pula zdjec do kolejnej iteracji recznej. "
            "Po eksporcie moga trafiac tu nieoznaczone obrazy, a recznie mozesz tez "
            "dolozyc nowy zestaw zdjec bez mieszania z gotowym runem."
        ),
        tone="muted",
        emphasis=False,
    )

    updated_at = str(manifest.get("updated_at") or "").strip()
    updated_suffix = f" Ostatnia synchronizacja: {updated_at}." if updated_at else ""

    if current_is_stage and stage_count == 0:
        status_text = (
            (
                "Pula następnej iteracji jest teraz aktywnym wejściem Z2, ale nie ma w niej jeszcze żadnych zdjęć. "
                "Wróć do głównego wejścia albo odłóż nowe obrazy na później."
                if campaign_context
                else "Stage jest aktualnym wejsciem Z2, ale nie ma w nim jeszcze zadnych zdjec. "
                "Dodaj nowe obrazy do stage albo wroc do glownego wejscia."
            ) + updated_suffix
        )
        tone = "warning"
    elif stage_count > 0:
        stage_rel = self._format_workspace_relative_path(stage_images_dir)
        status_text = (
            (
                f"Pula następnej iteracji zawiera {stage_count} zdjęć oczekujących na kolejną rundę pracy. "
                f"Możesz przełączyć Z2 na {stage_rel}."
                if campaign_context
                else f"Stage zawiera {stage_count} zdjec oczekujacych na kolejna runde recznej anotacji. "
                f"Możesz przełączyć Z2 na {stage_rel} albo dołożyć nowy zestaw zdjęć."
            ) + updated_suffix
        )
        tone = "success"
    else:
        status_text = (
            (
                "Pula następnej iteracji jest pusta. Po eksporcie trafiają tu tylko obrazy jeszcze niezatwierdzone."
                if campaign_context
                else "Stage jest pusty. Po eksporcie datasetu trafia tutaj tylko nieoznaczona czesc obrazow. "
                "Mozesz tez dolozyc nowe zdjecia do kolejnej iteracji."
            ) + updated_suffix
        )
        tone = "muted"

    self._set_inline_label_state(
        self.manual_stage_status_lbl,
        text=status_text,
        tone=tone,
        emphasis=False,
    )

    if campaign_context:
        if current_is_stage:
            self.manual_stage_use_btn.configure(
                text="Pula następnej iteracji jest aktywnym wejściem Z2",
                state=tk.DISABLED,
            )
        elif stage_count > 0:
            self.manual_stage_use_btn.configure(
                text="Użyj puli następnej iteracji w Z2",
                state=tk.NORMAL,
            )
        else:
            self.manual_stage_use_btn.configure(
                text="Użyj puli następnej iteracji w Z2",
                state=tk.DISABLED,
            )
        try:
            if str(self.manual_stage_add_btn.winfo_manager()) == "grid":
                self.manual_stage_add_btn.grid_remove()
        except Exception:
            pass
        if stage_count > 0 or current_is_stage:
            self._set_widget_packed(self.manual_stage_buttons_row, True, fill=tk.X, pady=(0, 2))
            try:
                if str(self.manual_stage_use_btn.winfo_manager()) != "grid":
                    self.manual_stage_use_btn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0))
                else:
                    self.manual_stage_use_btn.grid_configure(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0))
            except Exception:
                pass
        else:
            self._set_widget_packed(self.manual_stage_buttons_row, False)
    else:
        if current_is_stage:
            self.manual_stage_use_btn.configure(text="Stage jest aktywnym wejsciem Z2", state=tk.DISABLED)
        elif stage_count > 0:
            self.manual_stage_use_btn.configure(text="Uzyj stage jako wejscia Z2", state=tk.NORMAL)
        else:
            self.manual_stage_use_btn.configure(text="Uzyj stage jako wejscia Z2", state=tk.DISABLED)

        self.manual_stage_add_btn.configure(state=tk.NORMAL)

def _show_auto_annotation_success_dialog(self, next_steps: str = "") -> None:
    try:
        parent = self.frame.winfo_toplevel()
    except Exception:
        parent = getattr(self.app, "root", None)

    dialog = tk.Toplevel(parent if parent is not None else self.frame)
    dialog.withdraw()
    try:
        dialog.transient(parent)
    except Exception:
        pass
    dialog.title("Autoanotacja zakończona")
    dialog.resizable(False, False)

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    error = palette.get("error", "#e74c3c")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    try:
        dialog.configure(bg=panel_bg)
    except Exception:
        pass

    shell = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0, padx=20, pady=18)
    shell.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        shell,
        text="Autoanotacja została zakończona",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI Semibold", 12),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    summary_text = str(next_steps or "").strip() or (
        "Run anotacji Z2 jest gotowy. Sprawdź wynik i popraw ramki tablic tam, gdzie to potrzebne."
    )
    tk.Label(
        shell,
        text=summary_text,
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=620,
    ).pack(anchor=tk.W, fill=tk.X, pady=(10, 0))

    warning_box = tk.Frame(
        shell,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=error,
        highlightcolor=error,
        padx=12,
        pady=10,
    )
    warning_box.pack(anchor=tk.W, fill=tk.X, pady=(14, 0))

    tk.Label(
        warning_box,
        text=(
            "WARUNEK KONIECZNY: eksport datasetu albo wycinanie tablic będzie możliwe dopiero po "
            "zatwierdzeniu odpowiedniej liczby zdjęć prawym przyciskiem myszy na liście."
        ),
        bg=panel_alt,
        fg=error,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
        wraplength=600,
    ).pack(anchor=tk.W, fill=tk.X)

    tk.Label(
        warning_box,
        text=(
            "Na liście wyników kliknij PPM na wybranym zdjęciu lub zaznaczonej grupie i nadaj status [OK]. "
            "Dopiero po tym program odblokuje dalsze CTA dla zatwierdzonych pozycji."
        ),
        bg=panel_alt,
        fg=fg,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=600,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 0))

    buttons = tk.Frame(shell, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def close_dialog() -> None:
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(
        buttons,
        text="Rozumiem",
        style="Accent.TButton",
        command=close_dialog,
    ).pack(side=tk.RIGHT)

    try:
        dialog.update_idletasks()
        width = max(680, int(dialog.winfo_reqwidth() or 680))
        height = max(260, int(dialog.winfo_reqheight() or 260))
        if parent is not None:
            px = int(parent.winfo_rootx() or 0)
            py = int(parent.winfo_rooty() or 0)
            pw = int(parent.winfo_width() or 0)
            ph = int(parent.winfo_height() or 0)
            x = px + max(0, (pw - width) // 2)
            y = py + max(0, (ph - height) // 2)
        else:
            x = y = 120
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
    except Exception:
        try:
            dialog.deiconify()
        except Exception:
            pass

    dialog.bind("<Escape>", lambda _event: close_dialog())
    try:
        dialog.wait_window()
    except Exception:
        pass

def _apply_z2_left_panel_copy_payload(self, payload: Z2CopyPayload, prefix_lookup: dict, palette: dict) -> None:
    self.auto_vehicle_choice_hint_var.set(str(payload.auto_choice_hint or ""))
    self.manual_entry_title_var.set(str(payload.manual_entry_title or ""))
    self.manual_entry_hint_var.set(str(payload.manual_hint or ""))
    self.manual_xml_template_hint_var.set(str(payload.manual_template_hint or ""))
    self.manual_vehicle_assist_hint_var.set(str(payload.manual_vehicle_hint or ""))
    self.workflow_conf_title_var.set(str(payload.workflow_conf_title or ""))
    self.workflow_conf_hint_var.set(str(payload.workflow_conf_hint or ""))
    self.workflow_vehicle_model_title_var.set(str(payload.workflow_vehicle_title or ""))
    self.workflow_vehicle_model_hint_var.set(str(payload.workflow_vehicle_hint or ""))
    self.workflow_start_intro_var.set(str(payload.workflow_start_intro or "").strip())
    followup_text = str(payload.followup_text or "").strip()
    export_text = str(payload.export_text or "").strip()
    self.followup_intro_var.set(followup_text)
    self.export_intro_var.set(export_text)
    self.run_intro_var.set(str(payload.run_intro_text or "").strip())
    self._set_inline_label_state(self.route_badge_lbl, text=str(payload.badge_text or ""), tone=str(payload.badge_tone or "muted"), emphasis=True)
    self._set_inline_label_state(self.run_intro_lbl, tone="muted", emphasis=False)
    self._set_inline_label_state(self.route_summary_lbl, text=str(payload.route_text or ""), tone=str(payload.route_tone or "muted"), emphasis=False)
    self._set_inline_label_state(self.workflow_start_intro_lbl, tone="muted", emphasis=False)
    self._set_inline_label_state(self.workflow_action_hint_lbl, text=str(payload.action_text or ""), tone="muted", emphasis=False)
    try:
        free_mode_screen = str(self._coerce_free_mode_screen() or "").strip().lower()
        free_mode_route = str(self._get_workflow_route() or "").strip().lower()
    except Exception:
        free_mode_screen = ""
        free_mode_route = ""
    followup_tone = (
        "warning"
        if (
            self._is_free_mode_session_context()
            and free_mode_route in {"auto", "manual"}
            and free_mode_screen in {"auto_summary", "manual_review"}
            and followup_text
        )
        else "muted"
    )
    self._set_inline_label_state(self.followup_intro_lbl, text=followup_text, tone=followup_tone, emphasis=False)
    self._set_inline_label_state(self.export_intro_lbl, text=export_text, tone="muted", emphasis=False)
    self._set_inline_label_state(self.auto_vehicle_choice_hint_lbl, tone="muted", emphasis=False)
    self._set_inline_label_state(self.manual_entry_hint_lbl, tone=str(payload.manual_hint_tone or "muted"), emphasis=False)
    self._set_inline_label_state(self.manual_xml_template_hint_lbl, tone=str(payload.manual_template_tone or "muted"), emphasis=False)
    self._set_inline_label_state(self.manual_vehicle_assist_hint_lbl, tone=str(payload.manual_vehicle_tone or "muted"), emphasis=False)
    self._set_inline_label_state(
        self.workflow_manual_vehicle_assist_hint_lbl,
        tone=str(payload.manual_vehicle_tone or "muted"),
        emphasis=False,
    )
    self._set_inline_label_state(
        self.auto_plate_model_hint_lbl,
        text=str(payload.auto_plate_model_hint_text or ""),
        tone=str(payload.auto_plate_model_hint_tone or "muted"),
        emphasis=False,
    )
    self._set_inline_label_state(self.workflow_conf_hint_lbl, tone="muted", emphasis=False)
    self._set_inline_label_state(self.workflow_vehicle_model_hint_lbl, tone="muted", emphasis=False)
    try:
        self.workflow_entry_title_lbl.configure(
            text=self._format_z2_thematic_title("Co chcesz zrobic?", prefix_lookup.get("workflow_entry"))
        )
    except Exception:
        pass
    try:
        self.run_title_lbl.configure(text=str(payload.run_title or ""))
    except Exception:
        pass
    try:
        workflow_input_title = self._strip_workflow_heading_prefix(str(payload.workflow_input_title or ""))
        if (
            self._is_free_mode_session_context()
            and str(self._get_workflow_route() or "").strip().lower() == "auto"
            and str(self._coerce_workflow_step() or "").strip().lower() == "auto_input"
        ):
            workflow_input_title = "Wska\u017c katalog obraz\u00f3w"
        self.workflow_input_title_lbl.configure(
            text=workflow_input_title
        )
    except Exception:
        pass
    try:
        self.auto_plate_model_title_lbl.configure(
            text=self._strip_workflow_heading_prefix(str(payload.auto_plate_model_title or ""))
        )
    except Exception:
        pass
    try:
        self.auto_vehicle_choice_title_lbl.configure(
            text=self._strip_workflow_heading_prefix(str(payload.auto_vehicle_choice_title or ""))
        )
    except Exception:
        pass
    try:
        self.manual_history_title_lbl.configure(
            text=self._strip_workflow_heading_prefix(str(payload.manual_history_title or ""))
        )
    except Exception:
        pass
    self._set_inline_label_state(
        self.workflow_input_hint_lbl,
        text=str(payload.workflow_input_hint or ""),
        tone="muted",
        emphasis=False,
    )
    try:
        self.workflow_start_title_lbl.configure(
            text=self._strip_workflow_heading_prefix(str(payload.workflow_start_title or ""))
        )
        self.workflow_start_title_lbl.configure(
            bg=palette.get("panel_alt", palette.get("panel", "#252526")),
            fg=palette.get("fg", "#f3f3f3"),
        )
    except Exception:
        pass
    try:
        self.followup_title_lbl.configure(text=str(payload.followup_title or ""))
    except Exception:
        pass
    try:
        self.export_title_lbl.configure(
            text=self._strip_workflow_heading_prefix(str(payload.export_title or "")),
            bg=palette.get("panel_alt", palette.get("panel", "#252526")),
            fg=palette.get("fg", "#f3f3f3"),
        )
    except Exception:
        pass
    try:
        self.split_title_lbl.configure(
            text=self._strip_workflow_heading_prefix("Podzial train / val / test")
        )
        self.split_title_lbl.configure(
            bg=palette.get("panel_alt", palette.get("panel", "#252526")),
            fg=palette.get("fg", "#f3f3f3"),
        )
    except Exception:
        pass
    try:
        self._refresh_plate_model_runtime_info_ui()
    except Exception:
        pass

def _refresh_workflow_button_styles(self):
    style = getattr(getattr(self, "app", None), "style", None)
    if style is None:
        return

    palette = getattr(self.app, "palette", {})
    panel_alt = palette.get("panel_alt", palette.get("panel", "#252526"))
    hover_bg = palette.get("button_hover", panel_alt)
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    primary_bg = blend_hex_colors(panel_alt, hover_bg, 0.35)
    disabled_bg = palette.get("panel", "#252526")
    muted_dim = palette.get("muted_dim", palette.get("muted", "#8c8c8c"))
    cta_outline = blend_hex_colors(
        palette.get("success", "#4ec9b0"),
        panel_border,
        0.18,
    )
    cta_outline_hover = blend_hex_colors(
        palette.get("success", "#4ec9b0"),
        palette.get("accent_hover", palette.get("accent", "#63c7ff")),
        0.24,
    )

    style.configure(
        "WorkflowCard.TButton",
        background=panel_alt,
        foreground=fg,
        bordercolor=cta_outline,
        lightcolor=cta_outline,
        darkcolor=cta_outline,
        padding=8,
        borderwidth=1,
        relief=tk.SOLID,
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "WorkflowCard.TButton",
        background=[
            ("active", hover_bg),
            ("pressed", hover_bg),
            ("disabled", disabled_bg),
        ],
        foreground=[("disabled", muted_dim)],
        bordercolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
        lightcolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
        darkcolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
    )

    style.configure(
        "WorkflowCardPrimary.TButton",
        background=primary_bg,
        foreground=fg,
        bordercolor=cta_outline,
        lightcolor=cta_outline,
        darkcolor=cta_outline,
        padding=8,
        borderwidth=1,
        relief=tk.SOLID,
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "WorkflowCardPrimary.TButton",
        background=[
            ("active", hover_bg),
            ("pressed", hover_bg),
            ("disabled", disabled_bg),
        ],
        foreground=[("disabled", muted_dim)],
        bordercolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
        lightcolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
        darkcolor=[
            ("active", cta_outline_hover),
            ("pressed", cta_outline_hover),
            ("disabled", panel_border),
        ],
    )

    for attr_name, style_name in {
        "workflow_plate_browse_btn": "WorkflowCard.TButton",
        "workflow_vehicle_custom_browse_btn": "WorkflowCard.TButton",
        "manual_history_open_btn": "WorkflowCardPrimary.TButton",
        "manual_history_import_btn": "WorkflowCard.TButton",
        "workflow_input_browse_btn": "WorkflowCard.TButton",
        "workflow_back_btn": "WorkflowCard.TButton",
        "workflow_next_btn": "Accent.TButton",
        "start_btn": "WorkflowCardPrimary.TButton",
        "stop_btn": "WorkflowCard.TButton",
        "enter_manual_review_btn": "WorkflowCardPrimary.TButton",
        "jump_to_export_btn": "WorkflowCard.TButton",
        "open_run_dir_btn": "WorkflowCard.TButton",
        "manual_stage_use_btn": "WorkflowCard.TButton",
        "manual_stage_add_btn": "WorkflowCard.TButton",
        "manual_stage_export_btn": "WorkflowCard.TButton",
        "plate_dataset_run_btn": "WorkflowCard.TButton",
        "plate_dataset_images_btn": "WorkflowCard.TButton",
        "export_back_btn": "WorkflowCard.TButton",
        "export_plate_annotations_btn": "WorkflowCard.TButton",
        "export_plate_dataset_btn": "WorkflowCardPrimary.TButton",
    }.items():
        button = getattr(self, attr_name, None)
        if button is None:
            continue
        try:
            button.configure(style=style_name)
        except Exception:
            pass

    try:
        self._refresh_auto_vehicle_choice_ui()
    except Exception:
        pass

def _refresh_z2_miniflow_progress(self):
    canvas = getattr(self, "workflow_progress_canvas", None)
    nav_panel = getattr(self, "workflow_nav_panel", None)
    if canvas is None or nav_panel is None:
        return

    items, all_done = self._get_z2_miniflow_progress_items()
    visible = bool(items) and self._widget_is_packed(nav_panel)
    self._set_widget_packed(
        canvas,
        visible,
        fill=tk.X,
        pady=(0, 6),
        before=getattr(self, "workflow_nav_row", None),
    )
    if not visible:
        try:
            canvas.delete("all")
            canvas.configure(height=1)
        except Exception:
            pass
        return

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    muted = palette.get("muted", "#c7c7c7")
    fg = palette.get("fg", "#f3f3f3")
    accent = palette.get("accent", "#4f8de3")
    success = palette.get("success", "#4ec9b0")
    current_fill = blend_hex_colors(accent, panel_bg, 0.18)
    done_fill = blend_hex_colors(success, panel_bg, 0.14)
    upcoming_fill = blend_hex_colors(border, panel_bg, 0.30)
    connector_bg = blend_hex_colors(border, panel_bg, 0.35)

    try:
        width = max(320, int(canvas.winfo_width() or canvas.winfo_reqwidth() or 0))
    except Exception:
        width = 320

    try:
        canvas.delete("all")
        canvas.configure(bg=panel_bg, height=74)
    except Exception:
        return

    count = max(1, len(items))
    base_inner_width = max(120.0, float(width) - 16.0)
    rough_gap = base_inner_width / float(max(1, count - 1)) if count > 1 else base_inner_width
    label_width = max(56.0, min(92.0, rough_gap + 10.0))
    edge_pad = max(18.0, (label_width / 2.0) + 2.0)
    usable_width = max(120.0, float(width) - edge_pad - edge_pad)
    step_gap = usable_width / float(max(1, count - 1)) if count > 1 else 0.0
    node_y = 14.0
    label_y = 26.0
    radius = 7.0

    positions = []
    for idx in range(count):
        x = edge_pad + (idx * step_gap if count > 1 else usable_width / 2.0)
        positions.append(x)

    for idx in range(count - 1):
        next_item = items[idx + 1]
        line_color = success if (all_done or next_item.get("state") in {"current", "done"}) else connector_bg
        canvas.create_line(
            positions[idx] + radius,
            node_y,
            positions[idx + 1] - radius,
            node_y,
            fill=line_color,
            width=2,
        )

    for idx, item in enumerate(items):
        x = positions[idx]
        state = str(item.get("state") or "upcoming")
        if state == "done":
            outline = success
            fill = done_fill
            text_fill = success
        elif state == "current":
            outline = accent
            fill = current_fill
            text_fill = fg
        else:
            outline = border
            fill = upcoming_fill
            text_fill = muted

        canvas.create_oval(
            x - radius,
            node_y - radius,
            x + radius,
            node_y + radius,
            fill=fill,
            outline=outline,
            width=1,
        )
        canvas.create_text(
            x,
            node_y,
            text=str(idx + 1),
            fill=text_fill,
            font=("Segoe UI", 8, "bold"),
        )
        canvas.create_text(
            x,
            label_y,
            text=str(item.get("label") or ""),
            fill=(fg if state == "current" else muted),
            anchor="n",
            justify=tk.CENTER,
            width=label_width,
            font=("Segoe UI", 7, "bold"),
        )

def _apply_main_pane_layout(self, *, force_defaults: bool = False):
    if bool(getattr(self, "_main_pane_layout_in_progress", False)):
        self._main_pane_layout_pending_force_defaults = bool(
            getattr(self, "_main_pane_layout_pending_force_defaults", False) or force_defaults
        )
        return

    pane = getattr(self, "main_pane", None)
    if pane is None:
        return

    self._main_pane_layout_in_progress = True

    try:
        try:
            self._sync_main_pane_right_panel_visibility()
        except Exception:
            pass

        try:
            pane.update_idletasks()
        except Exception:
            pass

        try:
            if not self._pane_has_child(pane, self.main_left_frame):
                return
            if not self._pane_has_child(pane, self.main_center_frame):
                return
        except Exception:
            return

        try:
            total_width = int(pane.winfo_width() or pane.winfo_reqwidth() or 0)
        except Exception:
            total_width = 0
        if total_width <= 0:
            return

        left_min, right_min = self._get_main_pane_width_limits()
        has_right = self._pane_has_child(pane, self.main_right_frame)
        if not has_right:
            center_min = min(820, max(420, total_width - left_min))
            max_left = max(left_min, total_width - center_min)

            try:
                current_left = int(pane.sashpos(0) or 0)
            except Exception:
                current_left = left_min

            default_left = min(max_left, max(left_min, min(int(total_width * 0.26), 336)))
            desired_left = (
                default_left
                if force_defaults or not self._main_pane_layout_initialized
                else current_left
            )
            desired_left = max(left_min, min(int(desired_left), max_left))

            try:
                if abs(int(current_left) - int(desired_left)) > 1:
                    pane.sashpos(0, int(desired_left))
                self._main_pane_layout_initialized = True
            except Exception:
                pass
            return

        available_center = max(220, total_width - left_min - right_min)
        center_min = min(640, available_center)

        max_left = max(left_min, total_width - right_min - center_min)
        max_second = max(left_min + center_min, total_width - right_min)

        try:
            current_left = int(pane.sashpos(0) or 0)
        except Exception:
            current_left = left_min
        try:
            current_second = int(pane.sashpos(1) or 0)
        except Exception:
            current_second = max(left_min + center_min, total_width - right_min)

        default_left = min(max_left, max(left_min, min(int(total_width * 0.20), 304)))
        default_right_width = max(right_min, min(int(total_width * 0.22), 360))
        default_second = max(default_left + center_min, total_width - default_right_width)
        default_second = min(default_second, max_second)

        desired_left = default_left if force_defaults or not self._main_pane_layout_initialized else current_left
        desired_second = default_second if force_defaults or not self._main_pane_layout_initialized else current_second

        desired_left = max(left_min, min(int(desired_left), max_left))
        desired_second = max(desired_left + center_min, int(desired_second))
        desired_second = min(desired_second, max_second)

        try:
            if abs(int(current_left) - int(desired_left)) > 1:
                pane.sashpos(0, int(desired_left))
            if abs(int(current_second) - int(desired_second)) > 1:
                pane.sashpos(1, int(desired_second))
            self._main_pane_layout_initialized = True
        except Exception:
            pass
    finally:
        self._main_pane_layout_in_progress = False
        pending_force_defaults = bool(getattr(self, "_main_pane_layout_pending_force_defaults", False))
        self._main_pane_layout_pending_force_defaults = False
        if pending_force_defaults:
            try:
                self.frame.after_idle(lambda: self._schedule_main_pane_layout_refresh(force_defaults=True))
            except Exception:
                self._schedule_main_pane_layout_refresh(force_defaults=True)
