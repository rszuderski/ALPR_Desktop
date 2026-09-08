#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka: Autoanotacja - główne przetwarzanie YOLO (pojazdy + tablice)
Układ 3-kolumnowy z interaktywną przeglądarką na Canvasie.
"""

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
from . import z2_context_runtime
from . import z2_run_io_runtime
from . import z2_run_lifecycle
from . import z2_manifest_runtime
from . import z2_annotation_startup
from . import z2_status_ui_runtime
from .z2_main_widgets import create_annotation_widgets
from .z2_annotation_delegates import bind_annotation_tab_delegates
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
from .z3_slim_progress_bar import SlimProgressBar as _SharedSlimProgressBar

NAV_BUTTON_WIDTH = 18


class SlimProgressBar(_SharedSlimProgressBar):
    def __init__(
        self,
        master,
        *,
        maximum: float = 100.0,
        value: float = 0.0,
        thickness: int = 2,
        trough_color: str = "#3c3c3c",
        fill_color: str = "#4ec9b0",
        **kwargs,
    ):
        super().__init__(
            master,
            maximum=maximum,
            value=value,
            thickness=thickness,
            trough_color=trough_color,
            fill_color=fill_color,
            **kwargs,
        )


class AnnotationTab:
    def release_gpu_resources_for_training(self) -> None:
        """Zwalnia modele YOLO trzymane przez Z2 przed startem treningu w Z4."""
        annotator = getattr(self, "annotator", None)
        if annotator is not None:
            try:
                annotator.unload_models()
            except Exception as e:
                logger.debug(f"Nie udało się zwolnić modeli Z2 przed treningiem: {e}")
            self.annotator = None
        try:
            cleanup_gpu_memory()
        except Exception as e:
            logger.debug(f"Nie udało się wyczyścić pamięci GPU po Z2: {e}")

    def get_free_mode_assistant_context(self, *args, **kwargs):
        return z2_workflow_methods.get_free_mode_assistant_context(self, *args, **kwargs)

    def __init__(self, parent, app):
        _init_started_at = time.perf_counter()
        _phase_started_at = _init_started_at

        def _log_init_phase(name: str) -> None:
            nonlocal _phase_started_at
            try:
                now = time.perf_counter()
                elapsed_ms = (now - _phase_started_at) * 1000.0
                total_ms = (now - _init_started_at) * 1000.0
                if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                    logger.info(
                        "[Z2 PERF] AnnotationTab.__init__ "
                        f"{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                    )
                _phase_started_at = now
            except Exception:
                pass

        self.parent = parent
        self.app = app
        self.icon = IconManager
        self.frame = ttk.Frame(parent)
        self._inertial_scroll = InertialScrollController(self.frame)
        self._startup_ui_ready = False
        session_state = self._load_free_mode_session_snapshot()
        _log_init_phase("load_session_snapshot")

        self.annotator = None
        self.dataset_creator = DatasetCreator()
        self.is_processing = False
        self.start_time = None
        self._free_mode_session_restore_in_progress = False
        self._campaign_project_restore_in_progress = False
        self._campaign_step2_transition_in_progress = False
        self._campaign_step2_transition_refresh_pending = False
        self._campaign_step2_splash_visible = False
        self._campaign_step2_splash_token = 0
        self._campaign_step2_splash_overlay = None
        self._campaign_step2_splash_card = None
        self._campaign_step2_splash_title_lbl = None
        self._campaign_step2_splash_body_lbl = None
        self._campaign_step2_splash_progress = None
        self._preview_processing_overlay_active = False
        self._campaign_context_project_name = ""
        self._free_mode_session_save_after_id = None
        self._pending_session_save_include_preview_approved = False
        self._last_free_mode_session_snapshot_signature = ""
        self._available_devices_cache = None
        self._available_devices_cache_time = 0.0
        self._pre_campaign_free_mode_snapshot = None
        self._ui_dispatch_queue = queue.Queue()
        self._ui_dispatch_after_id = None
        self._progress_update_lock = threading.Lock()
        self._pending_progress_update = None
        self._progress_update_flush_queued = False
        self._pre_progress_after_id = None
        self._pre_progress_tick = 0
        self._progress_update_seen = False
        self._pre_progress_message = ""
        self._last_run_progress_visible = False
        self._annotation_input_ready_cache = {}

        # Zmienne do przeglÄ…darki
        self.current_annotations = []
        self.current_input_dir = None
        self._preview_image_path_map = {}
        self._campaign_reuse_manual_filenames = set()
        self._campaign_reuse_manual_summary = {}
        self.current_annotation_run_dir = None
        self.current_annotation_xml_path = None
        self.current_preview_index = None
        self._preview_session_restore_index = None
        self._preview_session_restore_filename = ""
        self._preview_selected_plate_by_image = {}
        self._preview_selected_vehicle_by_image = {}
        self._preview_drag_state = None
        self._preview_pending_vertex_hit = None
        self._preview_corner_drag_modifier_down = False
        self._preview_last_modifier_press_at = 0.0
        self._preview_focus_zoom_modifier_down = False
        self._preview_focus_zoom_modifier_consumed = False
        self._preview_focus_zoom_click_stage = 0
        self._preview_focus_zoom_restore_state = None
        self._preview_focus_zoom_history = []
        self._preview_super_correction_active = False
        self._preview_super_correction_badge_bbox = None
        self._preview_shortcut_event_guard = {}
        self._preview_fullscreen_toggle_bbox = None
        self._auto_route_settings_pending = False
        self._preview_draw_mode = False
        self._preview_draw_points = []
        self._preview_delete_mode = False
        self._preview_delete_candidate_idx = None
        self._preview_dirty_images = set()
        self._preview_autosave_after_id = None
        self._preview_drag_refresh_after_id = None
        self._preview_select_render_after_id = None
        self._preview_pending_select_render = None
        self._preview_post_interaction_refresh_after_id = None
        self._preview_light_overlay_refresh = False
        self._preview_fullscreen_active = False
        self._preview_controls_legend_render_key = None
        self._preview_controls_legend_inline_expanded = False
        self._preview_controls_legend_fullscreen_expanded = False
        self._preview_controls_legend_anim_after_id = None
        self._preview_controls_legend_animating = False
        self._preview_controls_legend_current_height = 0.0
        self._preview_controls_legend_current_width = 0.0
        self._preview_controls_legend_render_width_override = 0.0
        self._preview_controls_legend_render_height_override = 0.0
        self._preview_controls_legend_offset_x = 10.0
        self._preview_controls_legend_offset_y = 10.0
        self._preview_controls_legend_inline_manual_position = False
        self._preview_controls_legend_grab_bbox = None
        self._preview_controls_legend_toggle_bbox = None
        self._preview_controls_legend_drag_state = None
        self._preview_controls_legend_click_state = None
        self._preview_controls_legend_active_grab_widget = None
        self._preview_controls_legend_visible = True
        self._preview_metrics_overlay_expanded = True
        self._preview_metrics_overlay_visible = False
        self._preview_metrics_overlay_offset_x = 12.0
        self._preview_metrics_overlay_offset_y = 12.0
        self._preview_metrics_overlay_drag_state = None
        self._preview_metrics_overlay_user_moved = False
        self._preview_metrics_overlay_render_key = None
        self._preview_metrics_overlay_current_width = 0.0
        self._preview_metrics_overlay_current_height = 0.0
        self._preview_bottom_hint_drag_state = None
        self._preview_bottom_hint_manual_position = None
        self._preview_super_correction_badge_visible = False
        self._preview_overlay_dock_expanded = True
        self._preview_overlay_dock_render_key = None
        self._preview_overlay_dock_size = None
        self._preview_overlay_dock_size_key = None
        self._preview_overlay_dock_pre_gate_key = None
        self._preview_overlay_dock_gate_render_key = None
        self._preview_overlay_dock_inline_gate_state = None
        self._preview_overlay_dock_tool_rows = {}
        self._preview_overlay_dock_status_rows = {}
        self._preview_image_status_overlay_render_key = None
        self._preview_campaign_gate_overlay_render_key = None
        self._campaign_step2_gate_overlay_state = {}
        self._preview_fullscreen_transition_active = False
        self._preview_fullscreen_overlay_ready = True
        self._preview_fullscreen_restore_log_visible = False
        self._preview_fullscreen_restore_root_state = False
        self._preview_fullscreen_restore_window_state = "normal"
        self._preview_fullscreen_restore_geometry = ""
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None
        self._preview_force_fit_after_resize = False
        self._preview_debug_enabled = False
        self._preview_debug_events = deque(maxlen=8)
        self._preview_debug_last_drag_update_at = 0.0
        self._preview_debug_log_path = Path(CONFIG.WORKSPACE_DIR) / "z2_debug.log"
        self._preview_layout_restore_after_ids = []
        self._preview_history_undo = {}
        self._preview_history_redo = {}
        self._preview_history_replaying = False
        self._preview_history_limit = 80
        self._preview_list_populate_after_id = None
        self._preview_list_populate_token = 0
        self._preview_list_population_active = False
        self._preview_tab_entry_reset_after_id = None
        self._preview_tab_entry_reset_token = 0
        self._preview_list_display_indices = []
        self._preview_list_display_index_map = {}
        self._preview_super_correction_badge_offset_x = None
        self._preview_super_correction_badge_offset_y = 14.0
        self._preview_super_correction_handle_bbox = None
        self._preview_super_correction_drag_state = None
        self._campaign_deferred_preview_load_after_id = None
        self._campaign_deferred_preview_load_token = 0
        self._campaign_deferred_restore_ui_after_id = None
        self._campaign_deferred_restore_ui_token = 0
        self._campaign_deferred_run_restore_in_progress = False
        self._campaign_deferred_run_restore_payload_applied = False
        self._campaign_deferred_run_restore_target_dir = ""
        self._campaign_deferred_run_restore_token = 0
        self._campaign_step2_transition_skip_heavy_finalize = False
        self._campaign_route_cleanup_after_id = None
        self._campaign_route_cleanup_token = 0
        self._campaign_char_effective_refresh_after_id = None
        self._preview_approved_persist_after_id = None
        self._preview_approval_followup_after_id = None
        self._preview_campaign_post_approval_after_id = None
        self._preview_user_interaction_quiet_until = 0.0
        self._preview_resume_persist_after_id = None
        self._preview_any_auto_in_run_cache = None
        self._preview_list_summary_cache = None
        self._current_preview_plate_count_cache = None
        self._preview_render_image_cache = {}
        self._preview_list_sort_tiles = {}
        self._preview_list_frozen_sort_mode = ""
        self._preview_list_frozen_filename_order = []
        self._preview_list_frozen_context_key = ""
        self._preview_list_frozen_bucket_snapshot = {}
        self._preview_approved_filenames = set()
        self._preview_image_meta_cache = {}
        self._pending_preview_approved_filenames = set()
        self._campaign_pending_approved_filenames = set()
        self._campaign_hidden_project_approved_filenames = set()
        self._campaign_hidden_char_effective_filenames = set()
        self._campaign_hidden_project_approved_count = 0
        self._campaign_hidden_char_effective_count = 0
        self._campaign_t06_entry_approval_baseline = None
        self._campaign_iteration_manual_filenames = set()
        self._campaign_auto_pre_run_snapshot = {}
        self._campaign_auto_pre_run_visible_state = {}
        self._run_plate_count_cache = {}
        self._annotation_run_scope_meta = {
            "mode": "",
            "label": "",
            "count": 0,
        }
        self._free_mode_branch_artifacts = {
            "owned_run_dirs": [],
            "owned_temp_dirs": [],
        }
        self._plate_auto_scope_modal_refresh_callback = None
        self._plate_auto_scope_modal_refresh_after_id = None
        self._plate_auto_scope_modal_selection_refresh_callback = None
        self._plate_auto_scope_modal_selection_refresh_after_id = None
        self._plate_auto_scope_modal_selection_watch_after_id = None
        self._plate_auto_scope_modal_last_selection_signature = ()
        self._plate_auto_scope_selection_mode_active = False
        self._plate_auto_scope_modal_open = False
        self._plate_auto_scope_locked_widget_states = {}
        self._main_pane_layout_after_id = None
        self._main_pane_layout_initialized = False
        self._main_pane_layout_in_progress = False
        self._main_pane_layout_pending_force_defaults = False
        self._main_pane_layout_deferred_for_preview_population = False
        self._main_pane_layout_deferred_force_defaults = False
        self._left_panel_scroll_after_id = None
        self._left_panel_top_row_minsize_before_preview_population = None
        self._preview_left_counter_width_bucket = 0
        self._current_run_manual_template = False
        self._current_run_manual_vehicle_assist = False
        self._plate_model_runtime_meta = {
            "path": str(session_state.get("plate_custom") or "").strip(),
            "identity": "",
            "source": "",
            "scope": "",
        }
        self._manual_review_active = False
        self._manual_review_from_auto = False
        self._manual_review_origin_route = ""
        self._manual_review_export_ready = False
        self._manual_template_ready_for_review = False
        self._dataset_export_completed = False
        self._last_completed_workflow_route = ""
        self._last_run_progress_visible = False
        self.preview_edit_status_var = tk.StringVar(value="Po zakończeniu anotacji tutaj poprawisz rogi tablic.")
        self.preview_list_summary_var = tk.StringVar(
            value="Tablice po korekcie: 0/0\nObrazy z poprawkami: 0/0 | Niezapisane: 0"
        )
        self._preview_list_sort_options = (
            "Status: ED, OK, problem",
            "Status: A, M, OK, problem",
            "Status: OK, ED, problem",
            "Status: problem, ED, OK",
            "Nazwa pliku A-Z",
        )
        self.preview_list_sort_var = tk.StringVar(value=self._preview_list_sort_options[0])
        self.preview_filter_conf_var = tk.DoubleVar(value=0.0)
        self.preview_filter_fit_var = tk.DoubleVar(value=0.0)
        self._preview_filter_conf_applied = 0.0
        self._preview_filter_fit_applied = 0.0
        self.preview_debug_var = tk.StringVar(value="DEBUG Z2 | oczekiwanie na zdarzenia")
        self.preview_fullscreen_hint_var = tk.StringVar(value="")

        self.mode_var = tk.StringVar(value=session_state["mode"])
        self.vehicle_model_var = tk.StringVar(value=session_state["vehicle_model"])
        self.character_model_var = tk.StringVar(value=session_state["character_model"])
        self.vehicle_custom_var = tk.StringVar(value=session_state["vehicle_custom"])
        self.plate_custom_var = tk.StringVar(value=session_state["plate_custom"])
        self.character_custom_var = tk.StringVar(value=session_state["character_custom"])
        initial_device = str(session_state.get("device") or "auto").strip() or "auto"
        try:
            app_device_getter = getattr(self.app, "get_global_yolo_device_choice", None)
            if callable(app_device_getter):
                initial_device = str(app_device_getter() or initial_device).strip() or initial_device
        except Exception:
            pass
        self.device_var = tk.StringVar(value=initial_device)
        self.conf_var = tk.DoubleVar(value=session_state["conf"])
        self._campaign_paths_locked = False
        self._annotation_log_visible = False
        self._character_model_options = {}
        self._left_section_separators = []
        self._compact_path_display_vars = []
        self._workflow_step_cards = []
        self._left_path_button_width = 15
        self._left_path_action_minsize = 140
        self.project_paths_info_var = tk.StringVar(value="")
        self.project_paths_rel_var = tk.StringVar(value="")
        self.plate_dataset_run_var = tk.StringVar(value=session_state["plate_dataset_run"])
        self.plate_dataset_images_var = tk.StringVar(value=session_state["plate_dataset_images"])
        self.plate_dataset_out_var = tk.StringVar(value="")
        self.plate_train_pct = tk.DoubleVar(value=session_state["plate_train_pct"])
        self.plate_val_pct = tk.DoubleVar(value=session_state["plate_val_pct"])
        self.plate_export_progress_var = tk.DoubleVar(value=0.0)
        self.progress_counts_var = tk.StringVar(value="udane/przer./całość: 0/0/0")
        # DomyĹ›lnie podpowiadaj katalog wejĹ›ciowy z workspace.
        self.input_dir_var = tk.StringVar(value=session_state["input_dir"])
        self.output_dir_var = tk.StringVar(value=session_state["output_dir"])
        self.manual_xml_template_var = tk.BooleanVar(value=bool(session_state["manual_xml_template"]))
        self.manual_vehicle_assist_var = tk.BooleanVar(value=bool(session_state["manual_vehicle_assist"]))
        self.workflow_route_var = tk.StringVar(value=str(session_state.get("workflow_route") or "").strip())
        self.manual_entry_mode_var = tk.StringVar(
            value=str(session_state.get("manual_entry_mode") or "new").strip() or "new"
        )
        self.auto_vehicle_choice_var = tk.StringVar(
            value=str(session_state.get("auto_vehicle_choice") or "skip").strip() or "skip"
        )
        self._workflow_route_hover_mode = None
        self.workflow_step_var = tk.StringVar(
            value=str(session_state.get("workflow_step") or "").strip()
        )
        self.free_mode_screen_var = tk.StringVar(
            value=str(session_state.get("free_mode_screen") or "").strip()
        )
        self._campaign_step2_workflow_route = ""
        self._campaign_step2_workflow_step = ""
        self._campaign_step2_screen = ""
        self._campaign_step2_manual_entry_mode = "continue"
        self._campaign_step2_auto_vehicle_choice = "skip"
        self.manual_entry_title_var = tk.StringVar(value="")
        self.workflow_intro_var = tk.StringVar(value="")
        self.workflow_start_intro_var = tk.StringVar(value="")
        self.workflow_action_hint_var = tk.StringVar(value="")
        self.run_intro_var = tk.StringVar(value="")
        self.workflow_conf_title_var = tk.StringVar(value="")
        self.workflow_conf_hint_var = tk.StringVar(value="")
        self.workflow_vehicle_model_title_var = tk.StringVar(value="")
        self.workflow_vehicle_model_hint_var = tk.StringVar(value="")
        self.auto_vehicle_choice_hint_var = tk.StringVar(value="")
        self.manual_entry_hint_var = tk.StringVar(value="")
        self.manual_history_run_var = tk.StringVar(value="")
        self.manual_history_hint_var = tk.StringVar(value="")
        self.run_output_info_var = tk.StringVar(value="")
        self.manual_xml_template_hint_var = tk.StringVar(value="")
        self.manual_vehicle_assist_hint_var = tk.StringVar(value="")
        self.campaign_reuse_manual_var = tk.BooleanVar(value=False)
        self._plate_model_runtime_meta.update(
            {
                "identity": str(session_state.get("plate_model_identity") or "").strip(),
                "source": str(session_state.get("plate_model_source") or "").strip(),
                "scope": str(session_state.get("plate_model_scope") or "").strip(),
            }
        )
        self.campaign_reuse_manual_hint_var = tk.StringVar(value="")
        self.manual_stage_dir_var = tk.StringVar(value="")
        self.manual_stage_status_var = tk.StringVar(value="")
        self.approve_context_var = tk.StringVar(value="")
        self.approve_hint_title_var = tk.StringVar(value="")
        self.approve_breakdown_title_var = tk.StringVar(value="")
        self.approve_gate_hint_var = tk.StringVar(value="")
        self.detection_mode_hint_var = tk.StringVar(value="")
        self.route_badge_var = tk.StringVar(value="")
        self.route_summary_var = tk.StringVar(value="")
        self.followup_intro_var = tk.StringVar(value="")
        self.export_intro_var = tk.StringVar(value="")
        self._manual_review_history_entries = []
        self._manual_review_history_label_map = {}
        last_preview_run_dir = str(session_state["last_preview_run_dir"] or "").strip()
        if not last_preview_run_dir:
            try:
                from ..campaign_manager import CAMPAIGN

                candidate_run = None
                if not self._is_free_mode_session_context():
                    candidate_run = self._resolve_safe_annotation_run_dir(
                        CAMPAIGN.get_step2_staging_run(),
                        require_xml=True,
                    )
                if candidate_run is not None:
                    last_preview_run_dir = str(candidate_run)
            except Exception:
                last_preview_run_dir = ""
        self.last_staging_run_dir = Path(last_preview_run_dir) if last_preview_run_dir else None

        self._create_widgets()
        _log_init_phase("create_widgets")
        self._ensure_ui_dispatch_pump()
        _log_init_phase("ensure_ui_dispatch")
        skip_free_mode_snapshot = False
        try:
            from ..campaign_manager import CAMPAIGN

            skip_free_mode_snapshot = bool(CAMPAIGN.get_active_project_name()) and not bool(
                getattr(self.app, "campaign_free_mode", False)
            )
        except Exception:
            skip_free_mode_snapshot = False
        if skip_free_mode_snapshot:
            # The state graph applies campaign context explicitly. Loading the
            # free-mode snapshot here can scan old runs and export sources
            # before the T04/T05/T06 gate context is even known.
            _log_init_phase("skip_free_mode_session_snapshot_for_campaign")
        else:
            self._apply_free_mode_session_snapshot(session_state, restore_preview=False)
            _log_init_phase("apply_session_snapshot")
        self._bind_free_mode_session_observers()
        _log_init_phase("bind_session_observers")
        self.frame.after_idle(self._mark_startup_ui_ready)
        _log_init_phase("schedule_startup_ready")

    def _mark_startup_ui_ready(self):
        self._startup_ui_ready = True
        try:
            self._preview_controls_legend_render_key = None
            self.frame.after(80, lambda: self._place_preview_legend_overlay(refresh=True))
        except Exception:
            pass

    def _post_to_ui(self, fn):
        if not callable(fn):
            return

        if threading.current_thread() is threading.main_thread():
            try:
                fn()
            except Exception:
                logger.exception("Blad zadania UI w zakladce Z2")
            return

        try:
            self._ui_dispatch_queue.put_nowait(fn)
        except Exception:
            pass
        try:
            self._ensure_ui_dispatch_pump()
        except Exception:
            pass

    def _queue_progress_update(self, pct, current, total, filename, successful=0):
        payload = (
            float(pct),
            int(current),
            int(total),
            str(filename or ""),
            int(successful or 0),
        )

        should_schedule = False
        with self._progress_update_lock:
            self._pending_progress_update = payload
            if not self._progress_update_flush_queued:
                self._progress_update_flush_queued = True
                should_schedule = True

        if should_schedule:
            self._post_to_ui(self._flush_queued_progress_update)

    def _cancel_pre_progress_activity(self):
        pending = getattr(self, "_pre_progress_after_id", None)
        if pending:
            try:
                self.frame.after_cancel(pending)
            except Exception:
                pass
        self._pre_progress_after_id = None

    def _start_pre_progress_activity(self, message: str):
        self._cancel_pre_progress_activity()
        self._progress_update_seen = False
        self._pre_progress_tick = 0
        self._pre_progress_message = str(message or "Inicjalizacja analizy")

        def pulse():
            self._pre_progress_after_id = None
            if not self.is_processing or bool(getattr(self, "_progress_update_seen", False)):
                return

            tick = int(getattr(self, "_pre_progress_tick", 0) or 0)
            dots = "." * ((tick % 3) + 1)
            pulse_value = 1.5 + (tick % 4) * 1.5
            try:
                self._update_preview_processing_overlay_progress(
                    pct=min(12.0, float(pulse_value) * 2.0),
                    meta_text=f"{self._pre_progress_message}{dots}",
                    filename="",
                )
            except Exception:
                pass
            if not bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
                self._set_status_label_state(f"{self._pre_progress_message}{dots}", "neutral")
            self._pre_progress_tick = tick + 1

            try:
                self._pre_progress_after_id = self.frame.after(350, pulse)
            except Exception:
                self._pre_progress_after_id = None

        try:
            self._pre_progress_after_id = self.frame.after(0, pulse)
        except Exception:
            self._pre_progress_after_id = None

    def _flush_queued_progress_update(self):
        payload = None
        with self._progress_update_lock:
            payload = self._pending_progress_update
            self._pending_progress_update = None
            self._progress_update_flush_queued = False

        if payload is None:
            return

        self._update_progress(*payload)

    def _ensure_ui_dispatch_pump(self):
        if getattr(self, "_ui_dispatch_after_id", None):
            return
        try:
            self._ui_dispatch_after_id = self.frame.after(20, self._drain_ui_dispatch_queue)
        except Exception:
            self._ui_dispatch_after_id = None

    def _drain_ui_dispatch_queue(self):
        self._ui_dispatch_after_id = None

        for _ in range(200):
            try:
                fn = self._ui_dispatch_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            try:
                fn()
            except Exception:
                logger.exception("Blad podczas obslugi kolejki UI w Z2")

        try:
            self._ui_dispatch_after_id = self.frame.after(20, self._drain_ui_dispatch_queue)
        except Exception:
            self._ui_dispatch_after_id = None

    def is_startup_ui_ready(self) -> bool:
        return bool(getattr(self, "_startup_ui_ready", False))

    def _auto_device_label(self) -> str:
        return "Auto"

    def _cpu_device_label(self) -> str:
        return "CPU"

    def _get_available_devices(self, *, allow_probe: bool = False):
        devices = [self._auto_device_label(), self._cpu_device_label()]
        if not bool(getattr(self, "_startup_ui_ready", False)):
            return devices
        cached = getattr(self, "_available_devices_cache", None)
        if isinstance(cached, list) and cached:
            return list(cached)
        if not allow_probe:
            return devices
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    devices.append(f"cuda:{i} ({name})")
        except: pass
        self._available_devices_cache = list(devices)
        self._available_devices_cache_time = time.monotonic()
        return devices

    def _normalize_selected_device(self, raw_value: str | None = None, devices=None) -> str:
        available = list(devices or [])
        current = str(raw_value if raw_value is not None else self.device_var.get() or "").strip()
        current_lower = current.lower()

        if not current or current_lower.startswith("auto"):
            return available[0] if available else self._auto_device_label()
        if current_lower.startswith("cpu"):
            return self._cpu_device_label()
        if current_lower.startswith("cuda:"):
            prefix = current.split()[0]
            for option in available:
                if str(option or "").lower().startswith(prefix):
                    return option
            return self._auto_device_label()

        return current if (not available or current in available) else (available[0] if available else self._auto_device_label())

    def _refresh_device_options(self, *, allow_probe: bool = False):
        devices = self._get_available_devices(allow_probe=allow_probe)
        if hasattr(self, "device_combo"):
            try:
                self.device_combo.configure(values=devices)
            except Exception:
                pass

        normalized = self._normalize_selected_device(devices=devices)
        if normalized:
            self.device_var.set(normalized)

        self._update_device_hint()

    def _device_to_ultralytics(self, device_str: str):
        raw = str(device_str or "").strip().lower()
        if not raw or raw.startswith("auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    return 0
            except Exception:
                pass
            return "cpu"

        if raw.startswith("cpu"):
            return "cpu"

        if raw.startswith("cuda:"):
            try:
                import torch
                if not torch.cuda.is_available():
                    return "cpu"
                return int(str(device_str).split(":")[1].split()[0])
            except Exception:
                return "cpu"

        return "cpu"

    def _update_device_hint(self, event=None):
        label = getattr(self, "device_hint_lbl", None)
        if label is None:
            return

        devices = self._get_available_devices(allow_probe=False)
        normalized = self._normalize_selected_device(devices=devices)
        current = str(self.device_var.get() or "").strip()
        if normalized != current:
            self.device_var.set(normalized)
            current = normalized

        palette = getattr(self.app, "palette", {})
        gpu_devices = [item for item in devices if item.startswith("cuda:")]
        current_lower = current.lower()

        if current_lower.startswith("auto"):
            if gpu_devices:
                text = "Auto preferuje GPU/CUDA, a przy braku akceleracji przejdzie na CPU."
                fg = palette.get("info", "#3498db")
            else:
                text = "Auto: brak CUDA, wiec autoanotacja uruchomi sie na CPU."
                fg = palette.get("warning", "#f39c12")
        elif current_lower.startswith("cpu"):
            text = "CPU wymusza prace bez akceleracji GPU."
            fg = palette.get("muted", "#9a9a9a")
        else:
            text = "Wybrana karta GPU zostanie uzyta do inferencji YOLO."
            fg = palette.get("success", "#27ae60")

        try:
            label.configure(text=text, foreground=fg)
        except Exception:
            pass

    def apply_global_yolo_device_choice(self, value: str):
        normalized = self._normalize_selected_device(raw_value=value)
        try:
            self.device_var.set(normalized)
        except Exception:
            pass
        try:
            self._update_device_hint()
        except Exception:
            pass
        self._queue_free_mode_session_save()

    def _get_effective_yolo_device_choice(self) -> str:
        try:
            app_device_getter = getattr(self.app, "get_global_yolo_device_choice", None)
            if callable(app_device_getter):
                normalized = self._normalize_selected_device(raw_value=app_device_getter())
                if str(self.device_var.get() or "").strip() != normalized:
                    self.device_var.set(normalized)
                return normalized
        except Exception:
            pass

        normalized = self._normalize_selected_device()
        if str(self.device_var.get() or "").strip() != normalized:
            self.device_var.set(normalized)
        return normalized

    def _refresh_confidence_value_labels(self):
        for label in (
            getattr(self, "workflow_conf_value_lbl", None),
            getattr(self, "conf_value_lbl", None),
        ):
            if label is None:
                continue
            try:
                label.configure(text=f"{self.conf_var.get():.2f}")
            except Exception:
                pass

    # Delegates from z2_session_runtime are bound after class creation.

    def _parse_cvat_preview_annotations(self, *args, **kwargs):
        return z2_workflow_methods._parse_cvat_preview_annotations(self, *args, **kwargs)

    @staticmethod
    def _mark_auto_plate_origin_for_annotations(annotations: list[ImageAnnotation] | None) -> None:
        for ann in list(annotations or []):
            for det in list(getattr(ann, "detections", []) or []):
                if not AnnotationTab._is_plate_detection_label(getattr(det, "label", "")):
                    continue
                attributes = dict(getattr(det, "attributes", {}) or {})
                manually_edited = str(attributes.get("manually_edited", "") or "").strip().lower() == "true"
                manual_source = str(attributes.get("manual_source", "") or "").strip().lower()
                cvat_source = str(getattr(det, "_cvat_source", "") or "").strip().lower()
                if manually_edited or manual_source or cvat_source == "manual":
                    continue
                try:
                    attributes.setdefault("annotation_origin", "auto")
                    attributes.setdefault("auto_source", "auto_annotation")
                    det.attributes = attributes
                    setattr(det, "_cvat_source", "auto")
                except Exception:
                    pass

    def _restore_preview_from_session_run(self, *args, **kwargs):
        return z2_workflow_methods._restore_preview_from_session_run(self, *args, **kwargs)

    # Delegates from z2_layout_ui_runtime are bound after class creation.

    def _schedule_left_panel_scroll_to_widget(self, widget, *, delay_ms: int = 0):
        pending = getattr(self, "_left_panel_scroll_after_id", None)
        if pending:
            try:
                self.frame.after_cancel(pending)
            except Exception:
                pass
            self._left_panel_scroll_after_id = None

        if widget is None:
            return

        def _run():
            self._left_panel_scroll_after_id = None
            self._scroll_left_panel_to_widget(widget)

        try:
            if int(delay_ms) <= 0:
                self.frame.after_idle(_run)
            else:
                self._left_panel_scroll_after_id = self.frame.after(int(delay_ms), _run)
        except Exception:
            _run()

    def _create_widgets(self):
        return create_annotation_widgets(self, SlimProgressBar, NAV_BUTTON_WIDTH)

    # Delegates from z2_model_runtime are bound after class creation.

    def _select_input_dir(self):
        if self._is_plate_auto_scope_modal_blocking_actions():
            return
        p = filedialog.askdirectory(initialdir=str(Path(CONFIG.DIR_1_RAW).absolute()))
        if p:
            self._switch_annotation_input_dir(Path(p), show_hint=False)

    def _set_input_dir_path_only(self, input_dir: Path, *, resolved_input_dir: Path | None = None) -> None:
        safe_input_dir = Path(resolved_input_dir or input_dir)
        self.input_dir_var.set(str(safe_input_dir))
        self.plate_dataset_images_var.set(str(safe_input_dir))
        self.plate_dataset_run_var.set("")
        self.plate_dataset_out_var.set(self._plate_dataset_output_preview())
        self.current_input_dir = safe_input_dir
        self._annotation_source_input_dir = safe_input_dir
        self._remember_annotation_input_dir_ready(input_dir, safe_input_dir)

    def _format_workspace_relative_path(self, *args, **kwargs):
        return z2_manifest_runtime._format_workspace_relative_path(self, *args, **kwargs)


    def _get_annotation_run_storage_display_path(self, *args, **kwargs):
        return z2_manifest_runtime._get_annotation_run_storage_display_path(self, *args, **kwargs)


    def _get_annotation_run_definition_text(self, *args, **kwargs):
        return z2_manifest_runtime._get_annotation_run_definition_text(self, *args, **kwargs)


    def _show_full_path_dialog(self, *args, **kwargs):
        return z2_manifest_runtime._show_full_path_dialog(self, *args, **kwargs)


    def _bind_full_path_dialog_on_click(self, *args, **kwargs):
        return z2_manifest_runtime._bind_full_path_dialog_on_click(self, *args, **kwargs)


    def _enable_compact_path_entry(self, *args, **kwargs):
        return z2_manifest_runtime._enable_compact_path_entry(self, *args, **kwargs)


    def _prompt_z2_text_input(self, *args, **kwargs):
        return z2_manifest_runtime._prompt_z2_text_input(self, *args, **kwargs)


    def _annotation_run_manifest_path(self, *args, **kwargs):
        return z2_manifest_runtime._annotation_run_manifest_path(self, *args, **kwargs)


    def _write_annotation_run_manifest(self, *args, **kwargs):
        return z2_manifest_runtime._write_annotation_run_manifest(self, *args, **kwargs)


    def _load_annotation_run_manifest(self, *args, **kwargs):
        return z2_manifest_runtime._load_annotation_run_manifest(self, *args, **kwargs)


    def _update_annotation_run_manifest(self, *args, **kwargs):
        return z2_manifest_runtime._update_annotation_run_manifest(self, *args, **kwargs)


    def _collect_preview_resume_manifest_fields(self, *args, **kwargs):
        return z2_manifest_runtime._collect_preview_resume_manifest_fields(self, *args, **kwargs)


    def _load_annotation_run_manual_touched_filenames(self, *args, **kwargs):
        return z2_manifest_runtime._load_annotation_run_manual_touched_filenames(self, *args, **kwargs)


    def _load_annotation_run_approved_filenames(self, *args, **kwargs):
        return z2_manifest_runtime._load_annotation_run_approved_filenames(self, *args, **kwargs)


    def _get_preview_approved_filenames_base(self, *args, **kwargs):
        return z2_manifest_runtime._get_preview_approved_filenames_base(self, *args, **kwargs)


    def _get_campaign_hidden_project_approved_filenames_runtime(self, *args, **kwargs):
        return z2_manifest_runtime._get_campaign_hidden_project_approved_filenames_runtime(self, *args, **kwargs)


    def _preview_annotation_can_be_approved_for_export(self, *args, **kwargs):
        return z2_manifest_runtime._preview_annotation_can_be_approved_for_export(self, *args, **kwargs)


    def _get_preview_approved_filenames(self, *args, **kwargs):
        return z2_manifest_runtime._get_preview_approved_filenames(self, *args, **kwargs)


    def _sync_preview_approval_flags_from_current_sets(self, *args, **kwargs):
        return z2_manifest_runtime._sync_preview_approval_flags_from_current_sets(self, *args, **kwargs)


    def _filter_campaign_project_approved_annotations(self, *args, **kwargs):
        return z2_manifest_runtime._filter_campaign_project_approved_annotations(self, *args, **kwargs)


    def _persist_preview_approved_filenames(self, *args, **kwargs):
        return z2_manifest_runtime._persist_preview_approved_filenames(self, *args, **kwargs)


    def _preview_annotation_is_explicitly_approved(self, *args, **kwargs):
        return z2_manifest_runtime._preview_annotation_is_explicitly_approved(self, *args, **kwargs)


    def _get_campaign_manual_touched_filenames(self, *args, **kwargs):
        return z2_manifest_runtime._get_campaign_manual_touched_filenames(self, *args, **kwargs)


    def _remember_annotation_run_resume_state(self, *args, **kwargs):
        return z2_manifest_runtime._remember_annotation_run_resume_state(self, *args, **kwargs)


    def _mark_annotation_run_completed(self, *args, **kwargs):
        return z2_manifest_runtime._mark_annotation_run_completed(self, *args, **kwargs)


    def _restore_campaign_step2_generated_from_run(self, *args, **kwargs):
        return z2_manifest_runtime._restore_campaign_step2_generated_from_run(self, *args, **kwargs)


    @staticmethod
    def _annotation_run_manifest_has_manual_value(*args, **kwargs):
        return z2_manifest_runtime._annotation_run_manifest_has_manual_value(*args, **kwargs)


    @staticmethod
    def _annotation_run_manifest_is_manual_template(*args, **kwargs):
        return z2_manifest_runtime._annotation_run_manifest_is_manual_template(*args, **kwargs)


    def _get_active_annotation_run_dir(self, *args, **kwargs):
        return z2_manifest_runtime._get_active_annotation_run_dir(self, *args, **kwargs)


    def _has_active_manual_template_run(self, *args, **kwargs):
        return z2_manifest_runtime._has_active_manual_template_run(self, *args, **kwargs)


    def _remember_campaign_manual_plate_source(self, *args, **kwargs):
        return z2_manifest_runtime._remember_campaign_manual_plate_source(self, *args, **kwargs)


    def _sync_campaign_iteration_artifact_registry(self, *args, **kwargs):
        return z2_manifest_runtime._sync_campaign_iteration_artifact_registry(self, *args, **kwargs)


    @staticmethod
    def _plate_dataset_source_manifest_path(*args, **kwargs):
        return z2_manifest_runtime._plate_dataset_source_manifest_path(*args, **kwargs)


    def _load_plate_dataset_source_manifest(self, *args, **kwargs):
        return z2_manifest_runtime._load_plate_dataset_source_manifest(self, *args, **kwargs)


    def _write_plate_dataset_source_manifest(self, *args, **kwargs):
        return z2_manifest_runtime._write_plate_dataset_source_manifest(self, *args, **kwargs)


    def _build_campaign_plate_approved_entries_from_run(self, *args, **kwargs):
        return z2_manifest_runtime._build_campaign_plate_approved_entries_from_run(self, *args, **kwargs)


    @staticmethod
    def _build_campaign_plate_entry_key(*args, **kwargs):
        return z2_manifest_runtime._build_campaign_plate_entry_key(*args, **kwargs)


    def _get_campaign_plate_entry_merge_key(self, *args, **kwargs):
        return z2_manifest_runtime._get_campaign_plate_entry_merge_key(self, *args, **kwargs)


    def _promote_run_to_campaign_plate_approved_set(self, *args, **kwargs):
        return z2_manifest_runtime._promote_run_to_campaign_plate_approved_set(self, *args, **kwargs)


    def _build_campaign_plate_approved_export_source(self, *args, **kwargs):
        return z2_manifest_runtime._build_campaign_plate_approved_export_source(self, *args, **kwargs)


    def _build_campaign_plate_approved_preview_bundle(self, *args, **kwargs):
        return z2_manifest_runtime._build_campaign_plate_approved_preview_bundle(self, *args, **kwargs)


    @staticmethod
    def _build_path_change_token(*args, **kwargs):
        return z2_manifest_runtime._build_path_change_token(*args, **kwargs)


    def _build_campaign_char_effective_source(self, *args, **kwargs):
        return z2_manifest_runtime._build_campaign_char_effective_source(self, *args, **kwargs)


    def _resolve_plate_source_run_from_dataset(self, *args, **kwargs):
        return z2_manifest_runtime._resolve_plate_source_run_from_dataset(self, *args, **kwargs)


    def _find_reused_manual_source_run(self, *args, **kwargs):
        return z2_run_lifecycle._find_reused_manual_source_run(self, *args, **kwargs)


    def _find_reused_training_source_run(self, *args, **kwargs):
        return z2_run_lifecycle._find_reused_training_source_run(self, *args, **kwargs)


    def _find_latest_annotation_run_dir(self, *args, **kwargs):
        return z2_run_lifecycle._find_latest_annotation_run_dir(self, *args, **kwargs)


    def _annotation_run_matches_input(self, *args, **kwargs):
        return z2_run_lifecycle._annotation_run_matches_input(self, *args, **kwargs)


    def _find_latest_annotation_run_for_input(self, *args, **kwargs):
        return z2_run_lifecycle._find_latest_annotation_run_for_input(self, *args, **kwargs)


    def _score_annotation_run_restore_candidate(self, *args, **kwargs):
        return z2_run_lifecycle._score_annotation_run_restore_candidate(self, *args, **kwargs)


    def _resolve_best_free_mode_restore_run(self, *args, **kwargs):
        return z2_run_lifecycle._resolve_best_free_mode_restore_run(self, *args, **kwargs)


    def _get_plate_dataset_base_dir(self, *args, **kwargs):
        return z2_run_lifecycle._get_plate_dataset_base_dir(self, *args, **kwargs)


    def _get_manual_plate_stage_dir(self, *args, **kwargs):
        return z2_run_lifecycle._get_manual_plate_stage_dir(self, *args, **kwargs)


    def _get_manual_plate_stage_images_dir(self, *args, **kwargs):
        return z2_run_lifecycle._get_manual_plate_stage_images_dir(self, *args, **kwargs)


    def _is_manual_plate_stage_input(self, *args, **kwargs):
        return z2_run_lifecycle._is_manual_plate_stage_input(self, *args, **kwargs)


    def _load_manual_plate_stage_manifest(self, *args, **kwargs):
        return z2_run_lifecycle._load_manual_plate_stage_manifest(self, *args, **kwargs)


    def _refresh_manual_plate_stage_ui(self, *args, **kwargs):
        return z2_run_lifecycle._refresh_manual_plate_stage_ui(self, *args, **kwargs)


    def _refresh_manual_review_followup_ui(self, *args, **kwargs):
        return z2_run_lifecycle._refresh_manual_review_followup_ui(self, *args, **kwargs)


    def _refresh_preview_workspace_visibility(self, *args, **kwargs):
        return z2_run_lifecycle._refresh_preview_workspace_visibility(self, *args, **kwargs)


    def _set_preview_processing_overlay(self, *args, **kwargs):
        return z2_run_lifecycle._set_preview_processing_overlay(self, *args, **kwargs)


    def _update_preview_processing_overlay_progress(self, *args, **kwargs):
        return z2_run_lifecycle._update_preview_processing_overlay_progress(self, *args, **kwargs)


    def _refresh_export_followup_ui(self, *args, **kwargs):
        return z2_run_lifecycle._refresh_export_followup_ui(self, *args, **kwargs)


    def _switch_annotation_input_dir(self, *args, **kwargs):
        return z2_run_lifecycle._switch_annotation_input_dir(self, *args, **kwargs)


    def _use_manual_plate_stage_as_input(self, *args, **kwargs):
        return z2_run_lifecycle._use_manual_plate_stage_as_input(self, *args, **kwargs)


    def _add_images_to_manual_plate_stage(self, *args, **kwargs):
        return z2_run_lifecycle._add_images_to_manual_plate_stage(self, *args, **kwargs)


    def _plate_dataset_output_preview(self, *args, **kwargs):
        return z2_run_lifecycle._plate_dataset_output_preview(self, *args, **kwargs)


    def _reset_free_mode_branch_artifacts(self, *args, **kwargs):
        return z2_run_lifecycle._reset_free_mode_branch_artifacts(self, *args, **kwargs)


    def _register_free_mode_branch_artifact(self, *args, **kwargs):
        return z2_run_lifecycle._register_free_mode_branch_artifact(self, *args, **kwargs)


    def _forget_manual_review_run(self, *args, **kwargs):
        return z2_run_lifecycle._forget_manual_review_run(self, *args, **kwargs)


    def _collect_free_mode_branch_cleanup_targets(self, *args, **kwargs):
        return z2_run_lifecycle._collect_free_mode_branch_cleanup_targets(self, *args, **kwargs)


    def _cleanup_free_mode_branch_artifacts(self, *args, **kwargs):
        return z2_run_lifecycle._cleanup_free_mode_branch_artifacts(self, *args, **kwargs)


    def _confirm_and_return_to_free_mode_route_choice(self, *args, **kwargs):
        return z2_run_lifecycle._confirm_and_return_to_free_mode_route_choice(self, *args, **kwargs)


    def _clear_active_annotation_run_context(self, *args, **kwargs):
        return z2_run_lifecycle._clear_active_annotation_run_context(self, *args, **kwargs)


    def _reset_annotation_miniflow_state(self, *args, **kwargs):
        return z2_run_lifecycle._reset_annotation_miniflow_state(self, *args, **kwargs)


    def _set_plate_export_status(self, *args, **kwargs):
        return z2_run_io_runtime._set_plate_export_status(self, *args, **kwargs)


    def _set_post_annotation_hint(self, *args, **kwargs):
        return z2_run_io_runtime._set_post_annotation_hint(self, *args, **kwargs)


    @staticmethod
    def _resolve_existing_run_dir(*args, **kwargs):
        return z2_run_io_runtime._resolve_existing_run_dir(*args, **kwargs)


    @staticmethod
    def _resolve_existing_dir(*args, **kwargs):
        return z2_run_io_runtime._resolve_existing_dir(*args, **kwargs)


    @staticmethod
    def _normalize_preview_metric_threshold(*args, **kwargs):
        return z2_run_io_runtime._normalize_preview_metric_threshold(*args, **kwargs)


    def _annotation_run_matches_expected_input_dir(self, *args, **kwargs):
        return z2_run_io_runtime._annotation_run_matches_expected_input_dir(self, *args, **kwargs)


    def _repair_campaign_step2_generated_state_for_input(self, *args, **kwargs):
        return z2_run_io_runtime._repair_campaign_step2_generated_state_for_input(self, *args, **kwargs)


    def _allocate_annotation_run_dir(self, *args, **kwargs):
        return z2_run_io_runtime._allocate_annotation_run_dir(self, *args, **kwargs)


    def _get_manual_review_import_initial_dir(self, *args, **kwargs):
        return z2_run_io_runtime._get_manual_review_import_initial_dir(self, *args, **kwargs)


    @staticmethod
    def _build_safe_imported_image_relative_path(*args, **kwargs):
        return z2_run_io_runtime._build_safe_imported_image_relative_path(*args, **kwargs)


    def _get_external_run_image_roots(self, *args, **kwargs):
        return z2_run_io_runtime._get_external_run_image_roots(self, *args, **kwargs)


    def _get_external_run_images_initial_dir(self, *args, **kwargs):
        return z2_run_io_runtime._get_external_run_images_initial_dir(self, *args, **kwargs)


    def _get_external_run_expected_image_count(self, *args, **kwargs):
        return z2_run_io_runtime._get_external_run_expected_image_count(self, *args, **kwargs)


    def _resolve_external_run_source_images(self, *args, **kwargs):
        return z2_run_io_runtime._resolve_external_run_source_images(self, *args, **kwargs)


    def _import_external_annotation_run_to_workspace(self, *args, **kwargs):
        return z2_run_io_runtime._import_external_annotation_run_to_workspace(self, *args, **kwargs)


    def _import_or_open_manual_review_run_from_dialog(self, *args, **kwargs):
        return z2_run_io_runtime._import_or_open_manual_review_run_from_dialog(self, *args, **kwargs)


    def _count_plate_annotations(self, *args, **kwargs):
        return z2_run_io_runtime._count_plate_annotations(self, *args, **kwargs)


    def _get_plate_annotation_exports_base_dir(self, *args, **kwargs):
        return z2_run_io_runtime._get_plate_annotation_exports_base_dir(self, *args, **kwargs)


    def _get_plate_annotation_export_annotations(self, *args, **kwargs):
        return z2_run_io_runtime._get_plate_annotation_export_annotations(self, *args, **kwargs)


    def _get_plate_annotation_export_approved_filenames(self, *args, **kwargs):
        return z2_run_io_runtime._get_plate_annotation_export_approved_filenames(self, *args, **kwargs)


    def _plate_annotation_filename_matches_lookup(self, *args, **kwargs):
        return z2_run_io_runtime._plate_annotation_filename_matches_lookup(self, *args, **kwargs)


    def _filter_plate_annotation_export_annotations_by_approved(self, *args, **kwargs):
        return z2_run_io_runtime._filter_plate_annotation_export_annotations_by_approved(self, *args, **kwargs)


    def _get_plate_annotation_package_export_state(self, *args, **kwargs):
        return z2_run_io_runtime._get_plate_annotation_package_export_state(self, *args, **kwargs)


    def _set_plate_annotation_export_button_state(self, *args, **kwargs):
        return z2_run_io_runtime._set_plate_annotation_export_button_state(self, *args, **kwargs)


    def _prompt_z2_export_choice(self, *args, **kwargs):
        return z2_run_io_runtime._prompt_z2_export_choice(self, *args, **kwargs)


    def _start_z2_export_choice_flow(self, *args, **kwargs):
        return z2_run_io_runtime._start_z2_export_choice_flow(self, *args, **kwargs)


    def _prompt_plate_annotation_package_export_options(self, *args, **kwargs):
        return z2_run_io_runtime._prompt_plate_annotation_package_export_options(self, *args, **kwargs)


    def _start_plate_annotation_package_export(self, *args, **kwargs):
        return z2_run_io_runtime._start_plate_annotation_package_export(self, *args, **kwargs)


    def _get_run_plate_annotation_counts(self, *args, **kwargs):
        return z2_run_io_runtime._get_run_plate_annotation_counts(self, *args, **kwargs)


    def _get_current_preview_plate_count_state(self, *args, **kwargs):
        return z2_run_io_runtime._get_current_preview_plate_count_state(self, *args, **kwargs)


    def _get_campaign_project_total_image_count(self, *args, **kwargs):
        return z2_run_io_runtime._get_campaign_project_total_image_count(self, *args, **kwargs)


    def _get_run_plate_approved_counts(self, *args, **kwargs):
        return z2_run_io_runtime._get_run_plate_approved_counts(self, *args, **kwargs)


    def _get_current_preview_plate_approved_counts(self, *args, **kwargs):
        return z2_run_io_runtime._get_current_preview_plate_approved_counts(self, *args, **kwargs)


    def _get_run_plate_strict_approved_state(self, *args, **kwargs):
        return z2_run_io_runtime._get_run_plate_strict_approved_state(self, *args, **kwargs)


    def _get_plate_dataset_export_approval_state(self, *args, **kwargs):
        return z2_run_io_runtime._get_plate_dataset_export_approval_state(self, *args, **kwargs)


    def _is_plate_dataset_export_allowed_for_current_selection(self, *args, **kwargs):
        return z2_run_io_runtime._is_plate_dataset_export_allowed_for_current_selection(self, *args, **kwargs)


    def _warn_plate_dataset_export_requires_ok(self, *args, **kwargs):
        return z2_run_io_runtime._warn_plate_dataset_export_requires_ok(self, *args, **kwargs)


    def _warn_plate_cut_requires_ok(self, *args, **kwargs):
        return z2_run_io_runtime._warn_plate_cut_requires_ok(self, *args, **kwargs)


    def _get_campaign_step2_approval_context(self, *args, **kwargs):
        return z2_run_io_runtime._get_campaign_step2_approval_context(self, *args, **kwargs)


    def _refresh_step2_action_states(self, *args, **kwargs):
        return z2_run_io_runtime._refresh_step2_action_states(self, *args, **kwargs)


    def _build_annotation_success_next_steps(self, *args, **kwargs):
        return z2_run_io_runtime._build_annotation_success_next_steps(self, *args, **kwargs)


    def _show_auto_annotation_success_dialog(self, *args, **kwargs):
        return z2_run_io_runtime._show_auto_annotation_success_dialog(self, *args, **kwargs)


    def _load_plate_dataset_context_from_run(self, *args, **kwargs):
        return z2_run_io_runtime._load_plate_dataset_context_from_run(self, *args, **kwargs)


    def _refresh_plate_dataset_export_sources(self, *args, **kwargs):
        return z2_run_io_runtime._refresh_plate_dataset_export_sources(self, *args, **kwargs)


    def _select_plate_dataset_run_dir(self, *args, **kwargs):
        return z2_run_io_runtime._select_plate_dataset_run_dir(self, *args, **kwargs)


    def _select_plate_dataset_images_dir(self, *args, **kwargs):
        return z2_run_io_runtime._select_plate_dataset_images_dir(self, *args, **kwargs)


    def _update_plate_dataset_ratio_labels(self, *args, **kwargs):
        return z2_run_io_runtime._update_plate_dataset_ratio_labels(self, *args, **kwargs)


    def _start_plate_dataset_export(self, *args, **kwargs):
        return z2_run_io_runtime._start_plate_dataset_export(self, *args, **kwargs)


    def _redirect_logs(self, *args, **kwargs):
        return z2_status_ui_runtime._redirect_logs(self, *args, **kwargs)


    def apply_theme(self, *args, **kwargs):
        return z2_status_ui_runtime.apply_theme(self, *args, **kwargs)


    def _set_inline_label_state(self, *args, **kwargs):
        return z2_status_ui_runtime._set_inline_label_state(self, *args, **kwargs)


    def _set_approve_hint_box_state(self, *args, **kwargs):
        return z2_status_ui_runtime._set_approve_hint_box_state(self, *args, **kwargs)


    def _set_approve_context_box_state(self, *args, **kwargs):
        return z2_status_ui_runtime._set_approve_context_box_state(self, *args, **kwargs)


    def _render_compact_info_table(self, *args, **kwargs):
        return z2_status_ui_runtime._render_compact_info_table(self, *args, **kwargs)


    def _set_plate_model_info_box_state(self, *args, **kwargs):
        return z2_status_ui_runtime._set_plate_model_info_box_state(self, *args, **kwargs)


    def _collect_preview_category_breakdown(self, *args, **kwargs):
        return z2_status_ui_runtime._collect_preview_category_breakdown(self, *args, **kwargs)


    def _refresh_approval_breakdown_canvas(self, *args, **kwargs):
        return z2_status_ui_runtime._refresh_approval_breakdown_canvas(self, *args, **kwargs)


    def _refresh_manual_stage_export_box_style(self, *args, **kwargs):
        return z2_status_ui_runtime._refresh_manual_stage_export_box_style(self, *args, **kwargs)


    def _set_status_label_state(self, *args, **kwargs):
        return z2_status_ui_runtime._set_status_label_state(self, *args, **kwargs)


    def _set_annotation_process_log_visibility(self, *args, **kwargs):
        return z2_status_ui_runtime._set_annotation_process_log_visibility(self, *args, **kwargs)


    def _toggle_annotation_process_log(self):
        try:
            if hasattr(self.app, "toggle_global_terminal"):
                self.app.toggle_global_terminal()
                return
        except Exception:
            pass
        self._set_annotation_process_log_visibility(
            not getattr(self, "_annotation_log_visible", False)
        )

    def _validate_models(self):
        mode = self._normalize_mode_value()
        if self._mode_uses_vehicle(mode):
            if self.vehicle_model_var.get() == "Custom":
                validate_pt_model_path_for_runtime(
                    self.vehicle_custom_var.get(),
                    "pojazdów",
                )

        if self._mode_uses_plate(mode):
            plate_path = str(self.plate_custom_var.get() or "").strip()
            if not plate_path:
                try:
                    plate_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
                    plate_path = str(plate_meta.get("path") or "").strip()
                    if plate_path:
                        self.plate_custom_var.set(plate_path)
                except Exception:
                    plate_path = ""
            if not plate_path:
                try:
                    campaign_auto_context = bool(
                        not self._is_free_mode_session_context()
                        and str(self._get_workflow_route() or "").strip().lower() == "auto"
                    )
                except Exception:
                    campaign_auto_context = False
                if campaign_auto_context:
                    try:
                        plate_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
                        plate_path = str(plate_meta.get("path") or "").strip()
                    except Exception:
                        plate_path = ""
                    if not plate_path:
                        try:
                            plate_path = str(self.plate_custom_var.get() or "").strip()
                        except Exception:
                            plate_path = ""
                    if plate_path:
                        try:
                            self.plate_custom_var.set(plate_path)
                        except Exception:
                            pass
            validate_pt_model_path_for_runtime(
                plate_path,
                "tablic",
            )

    def _validate_vehicle_model_selection(self):
        if self.vehicle_model_var.get() == "Custom":
            validate_pt_model_path_for_runtime(
                self.vehicle_custom_var.get(),
                "pojazdów",
            )

    def ensure_campaign_context_ready_for_active_project(self, *args, **kwargs):
        return z2_context_runtime.ensure_campaign_context_ready_for_active_project(self, *args, **kwargs)


    def ensure_free_mode_session_preview_ready(self, *args, **kwargs):
        return z2_context_runtime.ensure_free_mode_session_preview_ready(self, *args, **kwargs)


    def clear_campaign_context(self, *args, **kwargs):
        return z2_context_runtime.clear_campaign_context(self, *args, **kwargs)


    def _format_project_relative_path(self, *args, **kwargs):
        return z2_context_runtime._format_project_relative_path(self, *args, **kwargs)


    def _set_campaign_paths_lock_state(self, *args, **kwargs):
        return z2_context_runtime._set_campaign_paths_lock_state(self, *args, **kwargs)


    def _show_campaign_step2_splash(self, *args, **kwargs):
        return z2_context_runtime._show_campaign_step2_splash(self, *args, **kwargs)


    def _hide_campaign_step2_splash(self, *args, **kwargs):
        return z2_context_runtime._hide_campaign_step2_splash(self, *args, **kwargs)


    def _begin_campaign_step2_transition(self, *args, **kwargs):
        return z2_context_runtime._begin_campaign_step2_transition(self, *args, **kwargs)


    def _end_campaign_step2_transition(self, *args, **kwargs):
        return z2_context_runtime._end_campaign_step2_transition(self, *args, **kwargs)


    def _reset_campaign_runtime_state(self, *args, **kwargs):
        return z2_context_runtime._reset_campaign_runtime_state(self, *args, **kwargs)


    def _apply_campaign_step2_workflow_preset(self, *args, **kwargs):
        return z2_context_runtime._apply_campaign_step2_workflow_preset(self, *args, **kwargs)


    def restore_campaign_context_from_project(self, *args, **kwargs):
        return z2_context_runtime.restore_campaign_context_from_project(self, *args, **kwargs)


    def open_campaign_step2_entry(self, *args, **kwargs):
        return z2_context_runtime.open_campaign_step2_entry(self, *args, **kwargs)


    def _apply_campaign_iteration_model_defaults(self, *args, **kwargs):
        return z2_context_runtime._apply_campaign_iteration_model_defaults(self, *args, **kwargs)


    def _enforce_campaign_plate_only_auto_default(self, *args, **kwargs):
        return z2_context_runtime._enforce_campaign_plate_only_auto_default(self, *args, **kwargs)


    def _should_restore_existing_campaign_step2_run(self, *args, **kwargs):
        return z2_context_runtime._should_restore_existing_campaign_step2_run(self, *args, **kwargs)


    def _should_restore_campaign_generated_step2_run_preview(self, *args, **kwargs):
        return z2_context_runtime._should_restore_campaign_generated_step2_run_preview(self, *args, **kwargs)


    def _refresh_campaign_workflow_input_lock_state(self, *args, **kwargs):
        return z2_context_runtime._refresh_campaign_workflow_input_lock_state(self, *args, **kwargs)


    def apply_campaign_context(self, *args, **kwargs):
        return z2_context_runtime.apply_campaign_context(self, *args, **kwargs)


    def _prepare_campaign_source_preview_payload(self, *args, **kwargs):
        return z2_context_runtime._prepare_campaign_source_preview_payload(self, *args, **kwargs)


    def _apply_campaign_source_preview_payload(self, *args, **kwargs):
        return z2_context_runtime._apply_campaign_source_preview_payload(self, *args, **kwargs)


    def _ensure_free_mode_input_workspace_preview(self, *args, **kwargs):
        return z2_context_runtime._ensure_free_mode_input_workspace_preview(self, *args, **kwargs)


    def _schedule_free_mode_input_workspace_preview_load(self, *args, **kwargs):
        return z2_context_runtime._schedule_free_mode_input_workspace_preview_load(self, *args, **kwargs)


    def _schedule_free_mode_auto_workspace_preview_load(self) -> bool:
        return self._schedule_free_mode_input_workspace_preview_load(
            expected_route="auto",
            expected_step="auto_start",
            require_auto_pending=True,
        )

    def _schedule_free_mode_manual_workspace_preview_load(self) -> bool:
        return self._schedule_free_mode_input_workspace_preview_load(
            expected_route="manual",
            expected_step="manual_start",
        )

    def _prime_campaign_source_preview(self, input_dir: Path | None) -> bool:
        started_at = time.perf_counter()
        try:
            payload = self._prepare_campaign_source_preview_payload(
                input_dir,
                include_previous=bool(self.campaign_reuse_manual_var.get()),
                current_manual_bundle=self._get_current_campaign_manual_preview_bundle(),
                approved_filenames=self._get_campaign_plate_approved_filenames(),
            )
            if not payload:
                return False
            result = self._apply_campaign_source_preview_payload(payload)
            return bool(result)
        finally:
            elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
            if elapsed_ms >= 20.0:
                payload_images = len(list((payload or {}).get("preview_annotations") or [])) if "payload" in locals() else 0
                source_dir = (payload or {}).get("source_dir") if "payload" in locals() else input_dir
                source_plan = dict((payload or {}).get("source_plan") or {}) if "payload" in locals() else {}
                logger.debug(
                    "[AnnotationTab][PERF] prime_campaign_source_preview: "
                    f"{elapsed_ms:.1f} ms | images={payload_images} "
                    f"approved_skipped={int(source_plan.get('approved_skip_count', 0) or 0)} "
                    f"reused={int(source_plan.get('reused_count', 0) or 0)} "
                    f"manual_current={int(source_plan.get('current_manual_count', 0) or 0)} "
                    f"source_dir={source_dir}"
                )

    def _get_workflow_route(self, *args, **kwargs):
        return z2_miniflow_runtime._get_workflow_route(self, *args, **kwargs)


    def _get_manual_entry_mode(self, *args, **kwargs):
        return z2_miniflow_runtime._get_manual_entry_mode(self, *args, **kwargs)


    def _get_auto_vehicle_choice(self, *args, **kwargs):
        return z2_miniflow_runtime._get_auto_vehicle_choice(self, *args, **kwargs)


    def _get_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._get_workflow_step(self, *args, **kwargs)


    def _get_free_mode_screen(self, *args, **kwargs):
        return z2_miniflow_runtime._get_free_mode_screen(self, *args, **kwargs)


    def _set_workflow_route_state(self, *args, **kwargs):
        return z2_miniflow_runtime._set_workflow_route_state(self, *args, **kwargs)


    def _set_workflow_step_state(self, *args, **kwargs):
        return z2_miniflow_runtime._set_workflow_step_state(self, *args, **kwargs)


    def _set_screen_state(self, *args, **kwargs):
        return z2_miniflow_runtime._set_screen_state(self, *args, **kwargs)


    def _set_manual_entry_mode_state(self, *args, **kwargs):
        return z2_miniflow_runtime._set_manual_entry_mode_state(self, *args, **kwargs)


    def _set_auto_vehicle_choice_state(self, *args, **kwargs):
        return z2_miniflow_runtime._set_auto_vehicle_choice_state(self, *args, **kwargs)


    def _reset_campaign_step2_runtime_state(self, *args, **kwargs):
        return z2_miniflow_runtime._reset_campaign_step2_runtime_state(self, *args, **kwargs)


    def _is_z2_auto_flow_completed(self, *args, **kwargs):
        return z2_miniflow_runtime._is_z2_auto_flow_completed(self, *args, **kwargs)


    def _repair_restored_z2_workflow_completion_state(self, *args, **kwargs):
        return z2_miniflow_runtime._repair_restored_z2_workflow_completion_state(self, *args, **kwargs)


    def _coerce_free_mode_screen(self, *args, **kwargs):
        return z2_miniflow_runtime._coerce_free_mode_screen(self, *args, **kwargs)


    def _get_workflow_progress_display(self, *args, **kwargs):
        return z2_miniflow_runtime._get_workflow_progress_display(self, *args, **kwargs)


    def _get_z2_thematic_route(self, *args, **kwargs):
        return z2_miniflow_runtime._get_z2_thematic_route(self, *args, **kwargs)


    def _get_z2_miniflow_current_key(self, *args, **kwargs):
        return z2_miniflow_runtime._get_z2_miniflow_current_key(self, *args, **kwargs)


    def _get_z2_miniflow_progress_items(self, *args, **kwargs):
        return z2_miniflow_runtime._get_z2_miniflow_progress_items(self, *args, **kwargs)


    def _refresh_z2_miniflow_progress(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_z2_miniflow_progress(self, *args, **kwargs)


    @staticmethod
    def _strip_workflow_step_suffix(*args, **kwargs):
        return z2_miniflow_runtime._strip_workflow_step_suffix(*args, **kwargs)


    @staticmethod
    def _strip_workflow_heading_prefix(*args, **kwargs):
        return z2_miniflow_runtime._strip_workflow_heading_prefix(*args, **kwargs)


    def _format_z2_thematic_title(self, *args, **kwargs):
        return z2_miniflow_runtime._format_z2_thematic_title(self, *args, **kwargs)


    def _get_z2_thematic_title_prefixes(self, *args, **kwargs):
        return z2_miniflow_runtime._get_z2_thematic_title_prefixes(self, *args, **kwargs)


    def _resolve_manual_review_history_created_at(self, *args, **kwargs):
        return z2_miniflow_runtime._resolve_manual_review_history_created_at(self, *args, **kwargs)


    def _normalize_manual_review_history_entries(self, *args, **kwargs):
        return z2_miniflow_runtime._normalize_manual_review_history_entries(self, *args, **kwargs)


    def _get_manual_review_history_display_entries(self, *args, **kwargs):
        return z2_miniflow_runtime._get_manual_review_history_display_entries(self, *args, **kwargs)


    def _format_manual_review_history_label(self, *args, **kwargs):
        return z2_miniflow_runtime._format_manual_review_history_label(self, *args, **kwargs)


    def _refresh_manual_review_history_ui(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_manual_review_history_ui(self, *args, **kwargs)


    def _remember_manual_review_run(self, *args, **kwargs):
        return z2_miniflow_runtime._remember_manual_review_run(self, *args, **kwargs)


    def _get_selected_manual_review_history_run_dir(self, *args, **kwargs):
        return z2_miniflow_runtime._get_selected_manual_review_history_run_dir(self, *args, **kwargs)


    def _open_selected_manual_review_history_run(self, *args, **kwargs):
        return z2_miniflow_runtime._open_selected_manual_review_history_run(self, *args, **kwargs)


    def _on_manual_history_selection_changed(self, *args, **kwargs):
        return z2_miniflow_runtime._on_manual_history_selection_changed(self, *args, **kwargs)


    def _get_auto_workflow_steps(self, *args, **kwargs):
        return z2_miniflow_runtime._get_auto_workflow_steps(self, *args, **kwargs)


    def _get_manual_workflow_steps(self, *args, **kwargs):
        return z2_miniflow_runtime._get_manual_workflow_steps(self, *args, **kwargs)


    def _get_current_workflow_steps(self, *args, **kwargs):
        return z2_miniflow_runtime._get_current_workflow_steps(self, *args, **kwargs)


    def _get_default_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._get_default_workflow_step(self, *args, **kwargs)


    def _coerce_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._coerce_workflow_step(self, *args, **kwargs)


    def _set_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._set_workflow_step(self, *args, **kwargs)


    def _clear_free_mode_route_selection(self, *args, **kwargs):
        return z2_miniflow_runtime._clear_free_mode_route_selection(self, *args, **kwargs)


    def _exit_manual_review_stage(self, *args, **kwargs):
        return z2_miniflow_runtime._exit_manual_review_stage(self, *args, **kwargs)


    def _is_workflow_step_complete(self, *args, **kwargs):
        return z2_miniflow_runtime._is_workflow_step_complete(self, *args, **kwargs)


    def _go_to_previous_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._go_to_previous_workflow_step(self, *args, **kwargs)


    def _close_export_followup(self, *args, **kwargs):
        return z2_miniflow_runtime._close_export_followup(self, *args, **kwargs)


    def _resolve_latest_exported_plate_dataset_dir(self, *args, **kwargs):
        return z2_miniflow_runtime._resolve_latest_exported_plate_dataset_dir(self, *args, **kwargs)


    def _prompt_post_z2_export_completion_action(self, *args, **kwargs):
        return z2_miniflow_runtime._prompt_post_z2_export_completion_action(self, *args, **kwargs)


    def _open_step4_training_from_z2_export(self, *args, **kwargs):
        return z2_miniflow_runtime._open_step4_training_from_z2_export(self, *args, **kwargs)


    def _go_to_next_workflow_step(self, *args, **kwargs):
        return z2_miniflow_runtime._go_to_next_workflow_step(self, *args, **kwargs)


    def _get_preferred_annotation_run_dir(self, *args, **kwargs):
        return z2_miniflow_runtime._get_preferred_annotation_run_dir(self, *args, **kwargs)


    def _should_skip_annotation_run_lookup_for_current_input(self, *args, **kwargs):
        return z2_miniflow_runtime._should_skip_annotation_run_lookup_for_current_input(self, *args, **kwargs)


    def _refresh_workflow_route_cards(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_workflow_route_cards(self, *args, **kwargs)


    def _select_free_mode_route(self, *args, **kwargs):
        return z2_miniflow_runtime._select_free_mode_route(self, *args, **kwargs)


    def _set_manual_entry_mode(self, *args, **kwargs):
        return z2_miniflow_runtime._set_manual_entry_mode(self, *args, **kwargs)


    def _on_manual_entry_mode_change(self, *args, **kwargs):
        return z2_miniflow_runtime._on_manual_entry_mode_change(self, *args, **kwargs)


    def _on_auto_vehicle_skip_toggle(self, *args, **kwargs):
        return z2_miniflow_runtime._on_auto_vehicle_skip_toggle(self, *args, **kwargs)


    def _refresh_auto_vehicle_choice_ui(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_auto_vehicle_choice_ui(self, *args, **kwargs)


    def _refresh_run_output_info(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_run_output_info(self, *args, **kwargs)


    def _open_current_run_dir(self, *args, **kwargs):
        return z2_miniflow_runtime._open_current_run_dir(self, *args, **kwargs)


    def _jump_to_export_section(self, *args, **kwargs):
        return z2_miniflow_runtime._jump_to_export_section(self, *args, **kwargs)


    def _resolve_step3_images_dir_from_z2_run(self, *args, **kwargs):
        return z2_miniflow_runtime._resolve_step3_images_dir_from_z2_run(self, *args, **kwargs)


    def _prepare_approved_step3_source_from_z2_run(self, *args, **kwargs):
        return z2_miniflow_runtime._prepare_approved_step3_source_from_z2_run(self, *args, **kwargs)


    def _open_step3_from_z2_annotation_source(self, *args, **kwargs):
        return z2_miniflow_runtime._open_step3_from_z2_annotation_source(self, *args, **kwargs)


    def _open_existing_run_for_manual_review(self, *args, **kwargs):
        return z2_miniflow_runtime._open_existing_run_for_manual_review(self, *args, **kwargs)


    def _open_existing_run_for_campaign_review(self, *args, **kwargs):
        return z2_miniflow_runtime._open_existing_run_for_campaign_review(self, *args, **kwargs)


    def _build_z2_layout_state_campaign(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_layout_state_campaign(self, *args, **kwargs)


    def _build_z2_layout_state_free_mode(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_layout_state_free_mode(self, *args, **kwargs)


    def _build_z2_cta_state_campaign(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_cta_state_campaign(self, *args, **kwargs)


    def _build_z2_cta_state_free_mode(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_cta_state_free_mode(self, *args, **kwargs)


    def _refresh_free_mode_workflow_ui(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_free_mode_workflow_ui(self, *args, **kwargs)


    def _manual_xml_template_enabled(self, *args, **kwargs):
        return z2_miniflow_runtime._manual_xml_template_enabled(self, *args, **kwargs)


    def _manual_vehicle_assist_enabled(self, *args, **kwargs):
        return z2_miniflow_runtime._manual_vehicle_assist_enabled(self, *args, **kwargs)


    def _return_to_campaign_wizard(self, *args, **kwargs):
        return z2_miniflow_runtime._return_to_campaign_wizard(self, *args, **kwargs)


    def _promote_campaign_char_repair_ok_to_approved_pool_before_return(self, *args, **kwargs):
        return z2_miniflow_runtime._promote_campaign_char_repair_ok_to_approved_pool_before_return(self, *args, **kwargs)


    def _sync_campaign_char_repair_approved_run_to_project_source(self, *args, **kwargs):
        return z2_workflow_methods._sync_campaign_char_repair_approved_run_to_project_source(self, *args, **kwargs)


    def _get_campaign_return_to_wizard_ok_warning_context(self, *args, **kwargs):
        return z2_miniflow_runtime._get_campaign_return_to_wizard_ok_warning_context(self, *args, **kwargs)


    def _prompt_campaign_return_to_wizard_ok_modal(self, *args, **kwargs):
        return z2_miniflow_runtime._prompt_campaign_return_to_wizard_ok_modal(self, *args, **kwargs)


    def _build_z2_left_panel_copy_context(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_left_panel_copy_context(self, *args, **kwargs)


    @staticmethod
    def _build_z2_left_panel_copy_defaults(*args, **kwargs):
        return z2_miniflow_runtime._build_z2_left_panel_copy_defaults(*args, **kwargs)


    def _apply_z2_left_panel_copy_payload(self, *args, **kwargs):
        return z2_miniflow_runtime._apply_z2_left_panel_copy_payload(self, *args, **kwargs)


    def _build_z2_left_panel_copy_payload_campaign(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_left_panel_copy_payload_campaign(self, *args, **kwargs)


    def _build_z2_left_panel_copy_payload_free_mode(self, *args, **kwargs):
        return z2_miniflow_runtime._build_z2_left_panel_copy_payload_free_mode(self, *args, **kwargs)


    def _refresh_left_panel_route_copy(self, *args, **kwargs):
        return z2_miniflow_runtime._refresh_left_panel_route_copy(self, *args, **kwargs)


    def _update_manual_xml_template_ui(self):
        if self._get_workflow_route() == "manual" and self._get_manual_entry_mode() != "new":
            self.manual_xml_template_var.set(False)

        if not self._manual_xml_template_enabled():
            self.manual_vehicle_assist_var.set(False)

        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_manual_plate_stage_ui()
        self._refresh_free_mode_workflow_ui()

    def _get_model_path(self, *args, **kwargs):
        return z2_annotation_startup._get_model_path(self, *args, **kwargs)


    def _collect_pending_model_downloads(self, *args, **kwargs):
        return z2_annotation_startup._collect_pending_model_downloads(self, *args, **kwargs)


    def _confirm_and_download_missing_models(self, *args, **kwargs):
        return z2_annotation_startup._confirm_and_download_missing_models(self, *args, **kwargs)


    def _start_annotation(self, *args, **kwargs):
        return z2_annotation_startup._start_annotation(self, *args, **kwargs)


    def _build_manual_annotations_template(self, *args, **kwargs):
        return z2_annotation_startup._build_manual_annotations_template(self, *args, **kwargs)


    def _build_manual_annotations_template_from_vehicle_seed(self, *args, **kwargs):
        return z2_annotation_startup._build_manual_annotations_template_from_vehicle_seed(self, *args, **kwargs)


    def _process_thread(self, *args, **kwargs):
        return z2_workflow_methods._process_thread(self, *args, **kwargs)

    # ==========================================================
    # LOGIKA PRZEGLÄ„DARKI (CANVAS)
    # ==========================================================
    def _resolve_preview_image_path(self, *args, **kwargs):
        return z2_preview_state._resolve_preview_image_path(self, *args, **kwargs)


    @staticmethod
    def _format_preview_dpi_value(*args, **kwargs):
        return z2_preview_state._format_preview_dpi_value(*args, **kwargs)


    def _get_preview_image_file_metadata(self, *args, **kwargs):
        return z2_preview_state._get_preview_image_file_metadata(self, *args, **kwargs)


    def _get_preview_image_dpi_text(self, *args, **kwargs):
        return z2_preview_state._get_preview_image_dpi_text(self, *args, **kwargs)


    @staticmethod
    def _is_plate_detection_label(*args, **kwargs):
        return z2_preview_state._is_plate_detection_label(*args, **kwargs)


    def _preview_list_item_text(self, *args, **kwargs):
        return z2_preview_state._preview_list_item_text(self, *args, **kwargs)


    def _preview_annotation_is_reused_from_previous_manual(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_is_reused_from_previous_manual(self, *args, **kwargs)


    @staticmethod
    def _preview_annotation_has_manual_touch_direct(*args, **kwargs):
        return z2_preview_state._preview_annotation_has_manual_touch_direct(*args, **kwargs)


    def _preview_annotation_has_manual_touch(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_has_manual_touch(self, *args, **kwargs)


    def _preview_annotation_is_manually_corrected(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_is_manually_corrected(self, *args, **kwargs)


    def _preview_annotation_origin_tag(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_origin_tag(self, *args, **kwargs)


    def _preview_annotation_status_tag(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_status_tag(self, *args, **kwargs)


    def _preview_annotation_has_auto_plate(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_has_auto_plate(self, *args, **kwargs)


    def _get_preview_list_render_state(self, *args, **kwargs):
        return z2_preview_state._get_preview_list_render_state(self, *args, **kwargs)


    def _build_preview_list_render_state_cache(self, *args, **kwargs):
        return z2_preview_state._build_preview_list_render_state_cache(self, *args, **kwargs)


    @staticmethod
    def _plate_detection_is_manual(*args, **kwargs):
        return z2_preview_state._plate_detection_is_manual(*args, **kwargs)


    def _get_preview_any_auto_in_run(self, *args, **kwargs):
        return z2_preview_state._get_preview_any_auto_in_run(self, *args, **kwargs)


    def _clear_selected_preview_auto_plates(self, *args, **kwargs):
        return z2_preview_state._clear_selected_preview_auto_plates(self, *args, **kwargs)


    def _collect_preview_manually_touched_filenames(self, *args, **kwargs):
        return z2_preview_state._collect_preview_manually_touched_filenames(self, *args, **kwargs)


    def _collect_preview_current_iteration_manual_filenames(self, *args, **kwargs):
        return z2_preview_state._collect_preview_current_iteration_manual_filenames(self, *args, **kwargs)


    def _collect_campaign_current_iteration_manual_filenames(self, *args, **kwargs):
        return z2_preview_state._collect_campaign_current_iteration_manual_filenames(self, *args, **kwargs)


    def _build_manual_override_annotation(self, *args, **kwargs):
        return z2_preview_state._build_manual_override_annotation(self, *args, **kwargs)


    def _get_current_campaign_manual_preview_bundle(self, *args, **kwargs):
        return z2_preview_state._get_current_campaign_manual_preview_bundle(self, *args, **kwargs)


    def _get_current_campaign_preview_snapshot(self, *args, **kwargs):
        return z2_preview_state._get_current_campaign_preview_snapshot(self, *args, **kwargs)


    def _restore_pre_run_preview_snapshot(self, *args, **kwargs):
        return z2_preview_state._restore_pre_run_preview_snapshot(self, *args, **kwargs)


    def _merge_preview_annotation_bundle(self, *args, **kwargs):
        return z2_preview_state._merge_preview_annotation_bundle(self, *args, **kwargs)


    def _build_missing_preview_annotations_bundle(self, *args, **kwargs):
        return z2_preview_state._build_missing_preview_annotations_bundle(self, *args, **kwargs)


    @staticmethod
    def _merge_annotation_bundle_into_payload(*args, **kwargs):
        return z2_preview_state._merge_annotation_bundle_into_payload(*args, **kwargs)


    def _preview_list_item_color(self, *args, **kwargs):
        return z2_preview_state._preview_list_item_color(self, *args, **kwargs)


    def _preview_list_color_for_bucket(self, *args, **kwargs):
        return z2_preview_state._preview_list_color_for_bucket(self, *args, **kwargs)


    def _preview_list_effective_color_bucket(self, *args, **kwargs):
        return z2_preview_state._preview_list_effective_color_bucket(self, *args, **kwargs)


    def _preview_list_color_plan(self, *args, **kwargs):
        return z2_preview_state._preview_list_color_plan(self, *args, **kwargs)


    def _preview_annotation_sort_bucket(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_sort_bucket(self, *args, **kwargs)


    def _preview_annotation_sort_bucket_for_mode(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_sort_bucket_for_mode(self, *args, **kwargs)


    def _preview_annotation_auto_scope_bucket(self, *args, **kwargs):
        return z2_preview_state._preview_annotation_auto_scope_bucket(self, *args, **kwargs)


    def _preview_list_status_priority(self, *args, **kwargs):
        return z2_preview_state._preview_list_status_priority(self, *args, **kwargs)


    def _build_preview_list_sorted_entries(self, *args, **kwargs):
        return z2_preview_state._build_preview_list_sorted_entries(self, *args, **kwargs)


    def _apply_preview_list_frozen_order(self, *args, **kwargs):
        return z2_preview_state._apply_preview_list_frozen_order(self, *args, **kwargs)


    def _get_preview_list_entries(self, *args, **kwargs):
        return z2_preview_state._get_preview_list_entries(self, *args, **kwargs)


    def _get_preview_display_index(self, *args, **kwargs):
        return z2_preview_state._get_preview_display_index(self, *args, **kwargs)


    def _get_preview_actual_index_from_display(self, *args, **kwargs):
        return z2_preview_state._get_preview_actual_index_from_display(self, *args, **kwargs)


    def _get_selected_preview_actual_indices(self, *args, **kwargs):
        return z2_preview_state._get_selected_preview_actual_indices(self, *args, **kwargs)


    def _select_all_visible_preview_images(self, *args, **kwargs):
        return z2_preview_state._select_all_visible_preview_images(self, *args, **kwargs)


    def _set_selected_preview_images_approved(self, *args, **kwargs):
        return z2_preview_state._set_selected_preview_images_approved(self, *args, **kwargs)


    def _refresh_preview_list_summary(self, *args, **kwargs):
        return z2_preview_state._refresh_preview_list_summary(self, *args, **kwargs)


    def _refresh_preview_list_legend_theme(self, *args, **kwargs):
        return z2_preview_state._refresh_preview_list_legend_theme(self, *args, **kwargs)


    def _get_preview_annotation(self, *args, **kwargs):
        return z2_preview_state._get_preview_annotation(self, *args, **kwargs)


    @staticmethod
    def _replace_preview_filename_in_set(*args, **kwargs):
        return z2_preview_state._replace_preview_filename_in_set(*args, **kwargs)


    @staticmethod
    def _replace_preview_filename_in_dict_keys(*args, **kwargs):
        return z2_preview_state._replace_preview_filename_in_dict_keys(*args, **kwargs)


    def _rename_selected_preview_image_file(self, *args, **kwargs):
        return z2_preview_state._rename_selected_preview_image_file(self, *args, **kwargs)


    @staticmethod
    def _serialize_quality_metric_value(*args, **kwargs):
        return z2_preview_state._serialize_quality_metric_value(*args, **kwargs)


    def _compute_plate_detection_fit_metrics(self, *args, **kwargs):
        return z2_preview_state._compute_plate_detection_fit_metrics(self, *args, **kwargs)


    def _refresh_plate_detection_quality_metrics(self, *args, **kwargs):
        return z2_preview_state._refresh_plate_detection_quality_metrics(self, *args, **kwargs)


    def _get_plate_detection_fit_score(self, *args, **kwargs):
        return z2_preview_state._get_plate_detection_fit_score(self, *args, **kwargs)


    def _get_preview_annotation_quality_summary(self, *args, **kwargs):
        return z2_preview_state._get_preview_annotation_quality_summary(self, *args, **kwargs)


    def _get_plate_detections(self, *args, **kwargs):
        return z2_preview_state._get_plate_detections(self, *args, **kwargs)


    @staticmethod
    def _detection_polygon(*args, **kwargs):
        return z2_preview_state._detection_polygon(*args, **kwargs)


    @staticmethod
    def _bbox_from_polygon(*args, **kwargs):
        return z2_preview_state._bbox_from_polygon(*args, **kwargs)


    @staticmethod
    def _polygon_area(*args, **kwargs):
        return z2_preview_state._polygon_area(*args, **kwargs)


    @staticmethod
    def _keypoints_from_polygon(*args, **kwargs):
        return z2_preview_state._keypoints_from_polygon(*args, **kwargs)


    def _get_selected_plate_index_for_ann(self, *args, **kwargs):
        return z2_preview_state._get_selected_plate_index_for_ann(self, *args, **kwargs)


    def _set_selected_plate_index_for_ann(self, *args, **kwargs):
        return z2_preview_state._set_selected_plate_index_for_ann(self, *args, **kwargs)


    @staticmethod
    def _get_vehicle_detections(*args, **kwargs):
        return z2_preview_state._get_vehicle_detections(*args, **kwargs)


    def _get_selected_vehicle_index_for_ann(self, *args, **kwargs):
        return z2_preview_state._get_selected_vehicle_index_for_ann(self, *args, **kwargs)


    def _set_selected_vehicle_index_for_ann(self, *args, **kwargs):
        return z2_preview_state._set_selected_vehicle_index_for_ann(self, *args, **kwargs)


    def _set_preview_focus_target(self, *args, **kwargs):
        return z2_preview_state._set_preview_focus_target(self, *args, **kwargs)


    def _get_preview_focus_target(self, *args, **kwargs):
        return z2_preview_state._get_preview_focus_target(self, *args, **kwargs)


    def _mark_preview_image_dirty(self, *args, **kwargs):
        return z2_preview_state._mark_preview_image_dirty(self, *args, **kwargs)


    def _get_preview_history_image_key(self, *args, **kwargs):
        return z2_preview_state._get_preview_history_image_key(self, *args, **kwargs)


    @staticmethod
    def _clone_preview_history_detection(*args, **kwargs):
        return z2_preview_state._clone_preview_history_detection(*args, **kwargs)


    def _clone_preview_history_annotation(self, *args, **kwargs):
        return z2_preview_state._clone_preview_history_annotation(self, *args, **kwargs)


    def _clone_preview_annotation_history_snapshot(self, *args, **kwargs):
        return z2_preview_state._clone_preview_annotation_history_snapshot(self, *args, **kwargs)


    def _get_preview_history_stack(self, *args, **kwargs):
        return z2_preview_state._get_preview_history_stack(self, *args, **kwargs)


    def _push_preview_history_snapshot(self, *args, **kwargs):
        return z2_preview_state._push_preview_history_snapshot(self, *args, **kwargs)


    def _restore_preview_annotation_history_snapshot(self, *args, **kwargs):
        return z2_preview_state._restore_preview_annotation_history_snapshot(self, *args, **kwargs)


    def _clear_preview_editor_state(self, *args, **kwargs):
        return z2_preview_state._clear_preview_editor_state(self, *args, **kwargs)


    def _cancel_preview_list_population(self, *args, **kwargs):
        return z2_preview_state._cancel_preview_list_population(self, *args, **kwargs)


    def _should_use_async_preview_list_population(self, *args, **kwargs):
        return z2_preview_state._should_use_async_preview_list_population(self, *args, **kwargs)


    def _cancel_deferred_campaign_source_preview_load(self, *args, **kwargs):
        return z2_preview_state._cancel_deferred_campaign_source_preview_load(self, *args, **kwargs)


    def _cancel_deferred_campaign_restore_ui(self, *args, **kwargs):
        return z2_preview_state._cancel_deferred_campaign_restore_ui(self, *args, **kwargs)


    def _cancel_deferred_campaign_run_restore(self, *args, **kwargs):
        return z2_preview_state._cancel_deferred_campaign_run_restore(self, *args, **kwargs)


    def _resolve_run_image_dir_for_annotations(self, *args, **kwargs):
        return z2_preview_state._resolve_run_image_dir_for_annotations(self, *args, **kwargs)


    def _compute_annotation_run_restore_index(self, *args, **kwargs):
        return z2_preview_state._compute_annotation_run_restore_index(self, *args, **kwargs)


    def _prepare_annotation_run_restore_payload(self, *args, **kwargs):
        return z2_preview_state._prepare_annotation_run_restore_payload(self, *args, **kwargs)


    def _apply_annotation_run_restore_payload(self, *args, **kwargs):
        return z2_preview_state._apply_annotation_run_restore_payload(self, *args, **kwargs)


    def _schedule_deferred_campaign_run_restore(self, *args, **kwargs):
        return z2_preview_state._schedule_deferred_campaign_run_restore(self, *args, **kwargs)


    def _schedule_deferred_campaign_source_preview_load(self, *args, **kwargs):
        return z2_preview_state._schedule_deferred_campaign_source_preview_load(self, *args, **kwargs)


    @staticmethod
    def _clear_listbox_selection_fast(*args, **kwargs):
        return z2_preview_state._clear_listbox_selection_fast(*args, **kwargs)


    def _set_preview_list_population_active(self, *args, **kwargs):
        return z2_preview_state._set_preview_list_population_active(self, *args, **kwargs)


    def _populate_preview_list_async(self, *args, **kwargs):
        return z2_preview_state._populate_preview_list_async(self, *args, **kwargs)


    def _refresh_preview_list(self, *args, **kwargs):
        return z2_preview_state._refresh_preview_list(self, *args, **kwargs)


    def _populate_preview_list(self, *args, **kwargs):
        return z2_preview_state._populate_preview_list(self, *args, **kwargs)


    def _clear_preview_metric_filters_for_new_run(self, *args, **kwargs):
        return z2_preview_state._clear_preview_metric_filters_for_new_run(self, *args, **kwargs)


    def _ensure_preview_list_population_consistency(self, *args, **kwargs):
        return z2_preview_state._ensure_preview_list_population_consistency(self, *args, **kwargs)


    def _schedule_preview_list_population_consistency_check(self, *args, **kwargs):
        return z2_preview_state._schedule_preview_list_population_consistency_check(self, *args, **kwargs)


    def _select_preview_index(self, *args, **kwargs):
        return z2_preview_state._select_preview_index(self, *args, **kwargs)


    def _select_preview_index_for_super_correction(self, *args, **kwargs):
        return z2_preview_state._select_preview_index_for_super_correction(self, *args, **kwargs)


    def _select_preview_relative(self, *args, **kwargs):
        return z2_preview_state._select_preview_relative(self, *args, **kwargs)


    def _get_preview_navigation_actual_indices(self, *args, **kwargs):
        return z2_preview_state._get_preview_navigation_actual_indices(self, *args, **kwargs)


    def _select_preview_global_plate_relative(self, *args, **kwargs):
        return z2_preview_state._select_preview_global_plate_relative(self, *args, **kwargs)


    def _fit_preview_image_to_view(self, *args, **kwargs):
        return z2_preview_state._fit_preview_image_to_view(self, *args, **kwargs)


    def _get_preview_focus_zoom_target(self, *args, **kwargs):
        return z2_preview_state._get_preview_focus_zoom_target(self, *args, **kwargs)


    def _restore_preview_focus_zoom_view(self, *args, **kwargs):
        return z2_preview_state._restore_preview_focus_zoom_view(self, *args, **kwargs)


    def _get_preview_legend_font(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_font(self, *args, **kwargs)
    @staticmethod
    def _truncate_preview_filename(*args, **kwargs):
        return z2_canvas_overlays._truncate_preview_filename(*args, **kwargs)
    def _get_preview_legend_context(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_context(self, *args, **kwargs)
    def _fit_preview_text_to_width(self, *args, **kwargs):
        return z2_canvas_overlays._fit_preview_text_to_width(self, *args, **kwargs)
    @staticmethod
    def _legend_color_is_light(*args, **kwargs):
        return z2_canvas_overlays._legend_color_is_light(*args, **kwargs)
    def _get_preview_legend_text_color(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_text_color(self, *args, **kwargs)
    @staticmethod
    def _hex_to_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
        return z2_hex_to_rgba(color, alpha)

    def _get_preview_legend_compass_photo(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_compass_photo(self, *args, **kwargs)
    def _trim_preview_legend_image_cache(self, *args, **kwargs):
        return z2_canvas_overlays._trim_preview_legend_image_cache(self, *args, **kwargs)
    def _clear_preview_legend_image_cache(self, *args, **kwargs):
        return z2_canvas_overlays._clear_preview_legend_image_cache(self, *args, **kwargs)
    def _get_preview_legend_shell_photo(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_shell_photo(self, *args, **kwargs)
    def _get_preview_legend_theme(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_theme(self, *args, **kwargs)
    def _get_preview_legend_keycap_photo(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_keycap_photo(self, *args, **kwargs)
    def _get_preview_legend_group_shell_photo(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_group_shell_photo(self, *args, **kwargs)
    @staticmethod
    def _normalize_preview_legend_interaction(*args, **kwargs):
        return z2_canvas_overlays._normalize_preview_legend_interaction(*args, **kwargs)
    def _get_preview_legend_interaction_marker_photo(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_legend_interaction_marker_photo(self, *args, **kwargs)
    def _is_preview_controls_legend_expanded(self, *args, **kwargs):
        return z2_canvas_overlays._is_preview_controls_legend_expanded(self, *args, **kwargs)
    def _toggle_preview_controls_legend(self, *args, **kwargs):
        return z2_canvas_overlays._toggle_preview_controls_legend(self, *args, **kwargs)
    def _is_preview_controls_legend_grab_hit(self, *args, **kwargs):
        return z2_canvas_overlays._is_preview_controls_legend_grab_hit(self, *args, **kwargs)
    def _is_preview_controls_legend_toggle_hit(self, *args, **kwargs):
        return z2_canvas_overlays._is_preview_controls_legend_toggle_hit(self, *args, **kwargs)
    def _clamp_preview_controls_legend_offsets(self, *args, **kwargs):
        return z2_canvas_overlays._clamp_preview_controls_legend_offsets(self, *args, **kwargs)
    def _on_preview_controls_legend_press(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_press(self, *args, **kwargs)
    def _on_preview_controls_legend_drag(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_drag(self, *args, **kwargs)
    def _update_preview_controls_legend_cursor(self, *args, **kwargs):
        return z2_canvas_overlays._update_preview_controls_legend_cursor(self, *args, **kwargs)
    def _on_preview_controls_legend_motion(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_motion(self, *args, **kwargs)
    def _on_preview_controls_legend_leave(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_leave(self, *args, **kwargs)
    def _on_preview_controls_legend_release(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_release(self, *args, **kwargs)
    def _hide_preview_controls_legend_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._hide_preview_controls_legend_overlay(self, *args, **kwargs)
    def _hide_preview_image_status_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._hide_preview_image_status_overlay(self, *args, **kwargs)
    def _hide_preview_overlay_dock_stack(self, *args, **kwargs):
        return z2_canvas_overlays._hide_preview_overlay_dock_stack(self, *args, **kwargs)
    def _render_preview_image_status_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._render_preview_image_status_overlay(self, *args, **kwargs)
    def _place_preview_image_status_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._place_preview_image_status_overlay(self, *args, **kwargs)
    def _build_campaign_z2_gate_overlay_state(self, *args, **kwargs):
        return z2_workflow_methods._build_campaign_z2_gate_overlay_state(self, *args, **kwargs)

    def _hide_preview_campaign_gate_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._hide_preview_campaign_gate_overlay(self, *args, **kwargs)
    def _render_preview_campaign_gate_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._render_preview_campaign_gate_overlay(self, *args, **kwargs)
    def _place_preview_campaign_gate_overlay(self, *args, **kwargs):
        return z2_canvas_overlays._place_preview_campaign_gate_overlay(self, *args, **kwargs)
    def _toggle_preview_overlay_dock(self, *args, **kwargs):
        return z2_canvas_overlays._toggle_preview_overlay_dock(self, *args, **kwargs)
    def _toggle_preview_overlay_dock_tool(self, *args, **kwargs):
        return z2_canvas_overlays._toggle_preview_overlay_dock_tool(self, *args, **kwargs)
    def _get_preview_overlay_dock_theme(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_overlay_dock_theme(self, *args, **kwargs)
    def _get_preview_overlay_dock_tools_state(self, *args, **kwargs):
        return z2_canvas_overlays._get_preview_overlay_dock_tools_state(self, *args, **kwargs)
    def _render_preview_overlay_dock(self, *args, **kwargs):
        return z2_canvas_overlays._render_preview_overlay_dock(self, *args, **kwargs)
    def _place_preview_overlay_dock(self, *args, **kwargs):
        return z2_canvas_overlays._place_preview_overlay_dock(self, *args, **kwargs)
    def _on_preview_canvas_frame_configure(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_canvas_frame_configure(self, *args, **kwargs)
    def _should_freeze_preview_legend_updates(self, *args, **kwargs):
        return z2_canvas_overlays._should_freeze_preview_legend_updates(self, *args, **kwargs)
    def _on_preview_controls_legend_configure(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_configure(self, *args, **kwargs)
    def _on_preview_controls_legend_mousewheel(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_mousewheel(self, *args, **kwargs)
    def _handle_preview_canvas_overlay_mousewheel(self, *args, **kwargs):
        return z2_canvas_overlays._handle_preview_canvas_overlay_mousewheel(self, *args, **kwargs)
    def _on_preview_controls_legend_enter(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_legend_enter(self, *args, **kwargs)
    def _on_preview_controls_scrollbar_command(self, *args, **kwargs):
        return z2_canvas_overlays._on_preview_controls_scrollbar_command(self, *args, **kwargs)
    def _build_preview_legend_sections(self, *args, **kwargs):
        return z2_canvas_overlays._build_preview_legend_sections(self, *args, **kwargs)
    def _refresh_preview_controls_legend(self, *args, **kwargs):
        return z2_canvas_overlays._refresh_preview_controls_legend(self, *args, **kwargs)
    def _get_preview_focus_image_key(self, *args, **kwargs):
        return z2_canvas_interaction._get_preview_focus_image_key(self, *args, **kwargs)
    def _preview_focus_restore_matches_current_image(self, *args, **kwargs):
        return z2_canvas_interaction._preview_focus_restore_matches_current_image(self, *args, **kwargs)
    def _capture_preview_view_state(self, *args, **kwargs):
        return z2_canvas_interaction._capture_preview_view_state(self, *args, **kwargs)
    def _compute_preview_bbox_focus_view_state(self, *args, **kwargs):
        return z2_canvas_interaction._compute_preview_bbox_focus_view_state(self, *args, **kwargs)
    def _compute_preview_plate_focus_view_state(self, *args, **kwargs):
        return z2_canvas_interaction._compute_preview_plate_focus_view_state(self, *args, **kwargs)
    def _refresh_preview_plate_context_overlays(self, *args, **kwargs):
        return z2_canvas_interaction._refresh_preview_plate_context_overlays(self, *args, **kwargs)
    def _focus_preview_plate(self, *args, **kwargs):
        return z2_canvas_interaction._focus_preview_plate(self, *args, **kwargs)
    def _focus_preview_vehicle(self, *args, **kwargs):
        return z2_canvas_interaction._focus_preview_vehicle(self, *args, **kwargs)
    def _restore_preview_focus_view(self, *args, **kwargs):
        return z2_canvas_interaction._restore_preview_focus_view(self, *args, **kwargs)
    def _restore_preview_layout_view_after_resize(self, *args, **kwargs):
        return z2_canvas_interaction._restore_preview_layout_view_after_resize(self, *args, **kwargs)
    def _cancel_preview_layout_restore_jobs(self, *args, **kwargs):
        return z2_canvas_interaction._cancel_preview_layout_restore_jobs(self, *args, **kwargs)
    def _schedule_preview_layout_restore_after_resize(self, *args, **kwargs):
        return z2_canvas_interaction._schedule_preview_layout_restore_after_resize(self, *args, **kwargs)
    def _stabilize_preview_layout_after_resize(self, *args, **kwargs):
        return z2_canvas_interaction._stabilize_preview_layout_after_resize(self, *args, **kwargs)
    def _format_preview_debug_vertex_ref(self, *args, **kwargs):
        return z2_canvas_interaction._format_preview_debug_vertex_ref(self, *args, **kwargs)
    def _format_preview_debug_pending(self, *args, **kwargs):
        return z2_canvas_interaction._format_preview_debug_pending(self, *args, **kwargs)
    def _format_preview_debug_drag(self, *args, **kwargs):
        return z2_canvas_interaction._format_preview_debug_drag(self, *args, **kwargs)
    def _build_preview_debug_lines(self, *args, **kwargs):
        return z2_canvas_interaction._build_preview_debug_lines(self, *args, **kwargs)
    def _refresh_preview_debug_status(self, *args, **kwargs):
        return z2_canvas_interaction._refresh_preview_debug_status(self, *args, **kwargs)
    def _print_preview_debug_console(self, *args, **kwargs):
        return z2_canvas_interaction._print_preview_debug_console(self, *args, **kwargs)
    def _append_z2_trace(self, *args, **kwargs):
        return z2_canvas_interaction._append_z2_trace(self, *args, **kwargs)
    def _push_preview_debug_event(self, *args, **kwargs):
        return z2_canvas_interaction._push_preview_debug_event(self, *args, **kwargs)
    @staticmethod
    def _widget_is_descendant_of(*args, **kwargs):
        return z2_canvas_interaction._widget_is_descendant_of(*args, **kwargs)
    @staticmethod
    def _event_has_control_modifier(*args, **kwargs):
        return z2_canvas_interaction._event_has_control_modifier(*args, **kwargs)
    def _preview_shortcuts_enabled(self, *args, **kwargs):
        return z2_canvas_interaction._preview_shortcuts_enabled(self, *args, **kwargs)
    def _preview_shortcut_is_duplicate(self, *args, **kwargs):
        return z2_canvas_interaction._preview_shortcut_is_duplicate(self, *args, **kwargs)
    @staticmethod
    def _pane_has_child(*args, **kwargs):
        return z2_canvas_interaction._pane_has_child(*args, **kwargs)
    def _sync_left_column_pane_layout(self, *args, **kwargs):
        return z2_canvas_interaction._sync_left_column_pane_layout(self, *args, **kwargs)
    def _bind_preview_shortcuts(self, *args, **kwargs):
        return z2_canvas_interaction._bind_preview_shortcuts(self, *args, **kwargs)
    def _on_preview_prev_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_prev_shortcut(self, *args, **kwargs)
    def _on_preview_arrow_up_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_arrow_up_shortcut(self, *args, **kwargs)
    def _on_preview_next_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_next_shortcut(self, *args, **kwargs)
    def _on_preview_arrow_down_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_arrow_down_shortcut(self, *args, **kwargs)
    def _on_preview_middle_click_zoom_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_middle_click_zoom_shortcut(self, *args, **kwargs)
    def _on_preview_edit_modifier_press(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_edit_modifier_press(self, *args, **kwargs)
    def _on_preview_edit_modifier_release(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_edit_modifier_release(self, *args, **kwargs)
    def _on_preview_draw_toggle_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_draw_toggle_shortcut(self, *args, **kwargs)
    def _on_preview_delete_mode_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_delete_mode_shortcut(self, *args, **kwargs)
    def _on_preview_delete_image_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_delete_image_shortcut(self, *args, **kwargs)
    def _on_preview_save_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_save_shortcut(self, *args, **kwargs)
    def _on_preview_undo_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_undo_shortcut(self, *args, **kwargs)
    def _on_preview_redo_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_redo_shortcut(self, *args, **kwargs)
    def _on_preview_fit_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_fit_shortcut(self, *args, **kwargs)
    def _on_preview_focus_zoom_modifier_press(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_focus_zoom_modifier_press(self, *args, **kwargs)
    def _on_preview_focus_zoom_modifier_release(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_focus_zoom_modifier_release(self, *args, **kwargs)
    def _on_preview_focus_toggle_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_focus_toggle_shortcut(self, *args, **kwargs)
    def _on_preview_cycle_plate_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_cycle_plate_shortcut(self, *args, **kwargs)
    def _toggle_preview_super_correction(self, *args, **kwargs):
        return z2_canvas_interaction._toggle_preview_super_correction(self, *args, **kwargs)
    def _on_preview_toggle_super_correction_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_toggle_super_correction_shortcut(self, *args, **kwargs)
    def _on_preview_toggle_image_approval_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_toggle_image_approval_shortcut(self, *args, **kwargs)
    def _on_preview_enter_fullscreen_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_enter_fullscreen_shortcut(self, *args, **kwargs)
    def _is_preview_super_correction_badge_hit(self, *args, **kwargs):
        return z2_canvas_interaction._is_preview_super_correction_badge_hit(self, *args, **kwargs)
    def _is_preview_super_correction_handle_hit(self, *args, **kwargs):
        return z2_canvas_interaction._is_preview_super_correction_handle_hit(self, *args, **kwargs)
    def _is_preview_fullscreen_toggle_hit(self, *args, **kwargs):
        return z2_canvas_interaction._is_preview_fullscreen_toggle_hit(self, *args, **kwargs)
    def _on_preview_escape_shortcut(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_escape_shortcut(self, *args, **kwargs)
    def _toggle_preview_fullscreen(self, *args, **kwargs):
        return z2_canvas_interaction._toggle_preview_fullscreen(self, *args, **kwargs)
    def _cancel_preview_controls_legend_animation(self, *args, **kwargs):
        return z2_canvas_interaction._cancel_preview_controls_legend_animation(self, *args, **kwargs)
    def _get_preview_controls_legend_target_width(self, *args, **kwargs):
        return z2_canvas_interaction._get_preview_controls_legend_target_width(self, *args, **kwargs)
    def _build_preview_controls_context_rows(self, *args, **kwargs):
        return z2_canvas_interaction._build_preview_controls_context_rows(self, *args, **kwargs)
    @staticmethod
    def _preview_controls_row_is_multi_plate_badge(*args, **kwargs):
        return z2_canvas_interaction._preview_controls_row_is_multi_plate_badge(*args, **kwargs)
    def _get_preview_controls_legend_target_height(self, *args, **kwargs):
        return z2_canvas_interaction._get_preview_controls_legend_target_height(self, *args, **kwargs)
    def _animate_preview_controls_legend_height(self, *args, **kwargs):
        return z2_canvas_interaction._animate_preview_controls_legend_height(self, *args, **kwargs)
    def _place_preview_legend_overlay(self, *args, **kwargs):
        return z2_canvas_interaction._place_preview_legend_overlay(self, *args, **kwargs)
    def _apply_preview_fullscreen_chrome(self, *args, **kwargs):
        return z2_canvas_interaction._apply_preview_fullscreen_chrome(self, *args, **kwargs)
    def _set_preview_fullscreen(self, *args, **kwargs):
        return z2_canvas_interaction._set_preview_fullscreen(self, *args, **kwargs)
    def _toggle_preview_draw_mode(self, *args, **kwargs):
        return z2_canvas_interaction._toggle_preview_draw_mode(self, *args, **kwargs)
    def _toggle_preview_delete_mode(self, *args, **kwargs):
        return z2_canvas_interaction._toggle_preview_delete_mode(self, *args, **kwargs)
    def _set_preview_corner_drag_modifier(self, *args, **kwargs):
        return z2_canvas_interaction._set_preview_corner_drag_modifier(self, *args, **kwargs)
    def _preview_modifier_active(self, *args, **kwargs):
        return z2_canvas_interaction._preview_modifier_active(self, *args, **kwargs)
    def _prime_preview_corner_drag_history(self, *args, **kwargs):
        return z2_canvas_interaction._prime_preview_corner_drag_history(self, *args, **kwargs)
    def _begin_preview_vertex_drag(self, *args, **kwargs):
        return z2_canvas_interaction._begin_preview_vertex_drag(self, *args, **kwargs)
    def _on_preview_canvas_focus_out(self, *args, **kwargs):
        return z2_canvas_interaction._on_preview_canvas_focus_out(self, *args, **kwargs)
    def on_zoomable_canvas_should_block_pan(self, *args, **kwargs):
        return z2_canvas_interaction.on_zoomable_canvas_should_block_pan(self, *args, **kwargs)
    def on_zoomable_canvas_zoom(self, *args, **kwargs):
        return z2_canvas_interaction.on_zoomable_canvas_zoom(self, *args, **kwargs)
    def on_zoomable_canvas_final_quality_delay_ms(self, *args, **kwargs):
        return z2_canvas_interaction.on_zoomable_canvas_final_quality_delay_ms(self, *args, **kwargs)
    def _get_preview_canvas_cursor(self, *args, **kwargs):
        return z2_canvas_interaction._get_preview_canvas_cursor(self, *args, **kwargs)
    def _undo_preview_edit(self, *args, **kwargs):
        return z2_canvas_interaction._undo_preview_edit(self, *args, **kwargs)
    def _redo_preview_edit(self, *args, **kwargs):
        return z2_canvas_interaction._redo_preview_edit(self, *args, **kwargs)
    def _sync_preview_canvas_cursor(self, *args, **kwargs):
        return z2_canvas_interaction._sync_preview_canvas_cursor(self, *args, **kwargs)
    def _preview_is_editable(self, *args, **kwargs):
        return z2_canvas_interaction._preview_is_editable(self, *args, **kwargs)
    def _ensure_preview_is_editable_for_action(self, *args, **kwargs):
        return z2_canvas_interaction._ensure_preview_is_editable_for_action(self, *args, **kwargs)
    def _is_free_mode_manual_xml_waiting_for_review(self, *args, **kwargs):
        return z2_preview_editor._is_free_mode_manual_xml_waiting_for_review(self, *args, **kwargs)


    def _get_step2_start_action_label(self, *args, **kwargs):
        return z2_preview_editor._get_step2_start_action_label(self, *args, **kwargs)


    def _get_step2_start_action_reference(self, *args, **kwargs):
        return z2_preview_editor._get_step2_start_action_reference(self, *args, **kwargs)


    def _on_preview_canvas_motion(self, *args, **kwargs):
        return z2_preview_editor._on_preview_canvas_motion(self, *args, **kwargs)


    def _on_preview_canvas_leave(self, *args, **kwargs):
        return z2_preview_editor._on_preview_canvas_leave(self, *args, **kwargs)


    def _update_preview_toolbar_state(self, *args, **kwargs):
        return z2_preview_editor._update_preview_toolbar_state(self, *args, **kwargs)


    @staticmethod
    def _preview_draw_corner_label(*args, **kwargs):
        return z2_preview_editor._preview_draw_corner_label(*args, **kwargs)


    def _preview_campaign_reuse_manual_note(self, *args, **kwargs):
        return z2_preview_editor._preview_campaign_reuse_manual_note(self, *args, **kwargs)


    def _update_preview_edit_status(self, *args, **kwargs):
        return z2_preview_editor._update_preview_edit_status(self, *args, **kwargs)


    def _load_current_preview_selection(self, *args, **kwargs):
        return z2_preview_editor._load_current_preview_selection(self, *args, **kwargs)


    def _schedule_preview_selection_render_after_restore(self, *args, **kwargs):
        return z2_preview_editor._schedule_preview_selection_render_after_restore(self, *args, **kwargs)


    def _render_preview_image(self, *args, **kwargs):
        return z2_preview_editor._render_preview_image(self, *args, **kwargs)


    def _refresh_preview_canvas(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_canvas(self, *args, **kwargs)


    def _cancel_preview_post_interaction_refresh(self, *args, **kwargs):
        return z2_preview_editor._cancel_preview_post_interaction_refresh(self, *args, **kwargs)


    def _refresh_preview_canvas_light(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_canvas_light(self, *args, **kwargs)


    def _schedule_preview_post_interaction_refresh(self, *args, **kwargs):
        return z2_preview_editor._schedule_preview_post_interaction_refresh(self, *args, **kwargs)


    def _refresh_preview_canvas_interactive(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_canvas_interactive(self, *args, **kwargs)


    def _refresh_preview_legend_backdrop(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_legend_backdrop(self, *args, **kwargs)


    def _cancel_preview_drag_refresh(self, *args, **kwargs):
        return z2_preview_editor._cancel_preview_drag_refresh(self, *args, **kwargs)


    def _schedule_preview_drag_refresh(self, *args, **kwargs):
        return z2_preview_editor._schedule_preview_drag_refresh(self, *args, **kwargs)


    def _cancel_preview_selection_render(self, *args, **kwargs):
        return z2_preview_editor._cancel_preview_selection_render(self, *args, **kwargs)


    def _schedule_preview_selection_render(self, *args, **kwargs):
        return z2_preview_editor._schedule_preview_selection_render(self, *args, **kwargs)


    def _draw_annotation_preview_overlay(self, *args, **kwargs):
        return z2_preview_editor._draw_annotation_preview_overlay(self, *args, **kwargs)


    def _build_preview_canvas_metrics_rows(self, *args, **kwargs):
        return z2_preview_editor._build_preview_canvas_metrics_rows(self, *args, **kwargs)

    def _toggle_preview_metrics_overlay(self, *args, **kwargs):
        return z2_preview_editor._toggle_preview_metrics_overlay(self, *args, **kwargs)

    def _on_preview_metrics_overlay_press(self, *args, **kwargs):
        return z2_preview_editor._on_preview_metrics_overlay_press(self, *args, **kwargs)

    def _on_preview_metrics_overlay_drag(self, *args, **kwargs):
        return z2_preview_editor._on_preview_metrics_overlay_drag(self, *args, **kwargs)

    def _on_preview_metrics_overlay_release(self, *args, **kwargs):
        return z2_preview_editor._on_preview_metrics_overlay_release(self, *args, **kwargs)

    def _clamp_preview_metrics_overlay_offsets(self, *args, **kwargs):
        return z2_preview_editor._clamp_preview_metrics_overlay_offsets(self, *args, **kwargs)

    def _render_preview_metrics_grab_handle(self, *args, **kwargs):
        return z2_preview_editor._render_preview_metrics_grab_handle(self, *args, **kwargs)

    def _render_preview_metrics_table(self, *args, **kwargs):
        return z2_preview_editor._render_preview_metrics_table(self, *args, **kwargs)

    def _update_preview_canvas_metrics_overlay(self, *args, **kwargs):
        return z2_preview_editor._update_preview_canvas_metrics_overlay(self, *args, **kwargs)

    def _get_preview_bottom_hint_text(self, *args, **kwargs):
        return z2_preview_editor._get_preview_bottom_hint_text(self, *args, **kwargs)


    def _draw_preview_bottom_hint(self, *args, **kwargs):
        return z2_preview_editor._draw_preview_bottom_hint(self, *args, **kwargs)


    def _clamp_preview_point(self, *args, **kwargs):
        return z2_preview_editor._clamp_preview_point(self, *args, **kwargs)


    def _find_preview_vertex_hit(self, *args, **kwargs):
        return z2_preview_editor._find_preview_vertex_hit(self, *args, **kwargs)


    def _get_nearest_preview_vertex(self, *args, **kwargs):
        return z2_preview_editor._get_nearest_preview_vertex(self, *args, **kwargs)


    def _find_preview_polygon_hit(self, *args, **kwargs):
        return z2_preview_editor._find_preview_polygon_hit(self, *args, **kwargs)


    def _find_preview_vehicle_hit(self, *args, **kwargs):
        return z2_preview_editor._find_preview_vehicle_hit(self, *args, **kwargs)


    def _get_global_nearest_preview_vertex(self, *args, **kwargs):
        return z2_preview_editor._get_global_nearest_preview_vertex(self, *args, **kwargs)


    def _get_preview_vertex_canvas_position(self, *args, **kwargs):
        return z2_preview_editor._get_preview_vertex_canvas_position(self, *args, **kwargs)


    def _describe_preview_hit_debug(self, *args, **kwargs):
        return z2_preview_editor._describe_preview_hit_debug(self, *args, **kwargs)


    def _describe_selected_polygon_vertices_debug(self, *args, **kwargs):
        return z2_preview_editor._describe_selected_polygon_vertices_debug(self, *args, **kwargs)


    def _get_preview_vertex_hit_radius(self, *args, **kwargs):
        return z2_preview_editor._get_preview_vertex_hit_radius(self, *args, **kwargs)


    def _resolve_preview_drag_target(self, *args, **kwargs):
        return z2_preview_editor._resolve_preview_drag_target(self, *args, **kwargs)


    def _finish_preview_vertex_drag(self, *args, **kwargs):
        return z2_preview_editor._finish_preview_vertex_drag(self, *args, **kwargs)


    def _persist_preview_structural_change(self, *args, **kwargs):
        return z2_preview_editor._persist_preview_structural_change(self, *args, **kwargs)


    def _delete_preview_polygon(self, *args, **kwargs):
        return z2_preview_editor._delete_preview_polygon(self, *args, **kwargs)


    def _commit_new_preview_polygon(self, *args, **kwargs):
        return z2_preview_editor._commit_new_preview_polygon(self, *args, **kwargs)


    def _refresh_free_mode_manual_review_export_controls(self, *args, **kwargs):
        return z2_preview_editor._refresh_free_mode_manual_review_export_controls(self, *args, **kwargs)


    def _get_current_annotation_xml_path(self, *args, **kwargs):
        return z2_preview_editor._get_current_annotation_xml_path(self, *args, **kwargs)


    def _ensure_preview_edits_saved(self, *args, **kwargs):
        return z2_preview_editor._ensure_preview_edits_saved(self, *args, **kwargs)


    def _cancel_preview_autosave(self, *args, **kwargs):
        return z2_preview_editor._cancel_preview_autosave(self, *args, **kwargs)


    def _schedule_preview_autosave(self, *args, **kwargs):
        return z2_preview_editor._schedule_preview_autosave(self, *args, **kwargs)


    def _defer_preview_autosave_for_navigation(self, *args, **kwargs):
        return z2_preview_editor._defer_preview_autosave_for_navigation(self, *args, **kwargs)


    def _refresh_preview_list_row_for_actual_index(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_list_row_for_actual_index(self, *args, **kwargs)


    def _remove_image_from_stage_manifest(self, *args, **kwargs):
        return z2_preview_editor._remove_image_from_stage_manifest(self, *args, **kwargs)


    def _save_preview_edits(self, *args, **kwargs):
        return z2_preview_editor._save_preview_edits(self, *args, **kwargs)


    def _delete_current_preview_image_hard(self, *args, **kwargs):
        return z2_preview_editor._delete_current_preview_image_hard(self, *args, **kwargs)


    def _move_current_preview_image_to_stage(self, *args, **kwargs):
        return z2_preview_editor._move_current_preview_image_to_stage(self, *args, **kwargs)


    def _on_preview_canvas_right_click(self, *args, **kwargs):
        return z2_preview_editor._on_preview_canvas_right_click(self, *args, **kwargs)


    def on_zoomable_canvas_press(self, *args, **kwargs):
        return z2_preview_editor.on_zoomable_canvas_press(self, *args, **kwargs)


    def on_zoomable_canvas_drag(self, *args, **kwargs):
        return z2_preview_editor.on_zoomable_canvas_drag(self, *args, **kwargs)


    def on_zoomable_canvas_release(self, *args, **kwargs):
        return z2_preview_editor.on_zoomable_canvas_release(self, *args, **kwargs)


    def _on_preview_list_mouse_primary(self, *args, **kwargs):
        return z2_preview_editor._on_preview_list_mouse_primary(self, *args, **kwargs)


    def _refresh_preview_list_context_menu_state(self, *args, **kwargs):
        return z2_preview_editor._refresh_preview_list_context_menu_state(self, *args, **kwargs)


    def _show_preview_list_context_menu(self, *args, **kwargs):
        return z2_preview_editor._show_preview_list_context_menu(self, *args, **kwargs)


    def _open_preview_list_context_menu_from_keyboard(self, *args, **kwargs):
        return z2_preview_editor._open_preview_list_context_menu_from_keyboard(self, *args, **kwargs)


    def _on_preview_list_mouse_secondary(self, *args, **kwargs):
        return z2_preview_editor._on_preview_list_mouse_secondary(self, *args, **kwargs)


    def _on_preview_select(self, *args, **kwargs):
        return z2_preview_editor._on_preview_select(self, *args, **kwargs)


    def _update_progress(self, pct, current, total, filename, successful=0):
        self._progress_update_seen = True
        self._last_run_progress_visible = True
        self._cancel_pre_progress_activity()
        try:
            self._update_preview_processing_overlay_progress(
                pct=pct,
                current=current,
                total=total,
                filename=filename,
            )
        except Exception:
            pass
        if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
            self._set_progress_counters(successful, current, total)
            return
        self._set_progress_counters(successful, current, total)
        self._set_status_label_state(
            f"Przetwarzanie {current}/{total} ({int(pct)}%)",
            "success"
        )

    def _finish(self, *args, **kwargs):
        return z2_workflow_methods._finish(self, *args, **kwargs)

    def _approve_annotation_stage(self, *args, **kwargs):
        return z2_workflow_methods._approve_annotation_stage(self, *args, **kwargs)

    def _stop_annotation(self):
        self._annotation_stop_requested = True
        self.is_processing = False
        if self.annotator and hasattr(self.annotator, 'stop'):
            self.annotator.stop()
        self._set_status_label_state("Zatrzymywanie...", "warning")
        try:
            self._refresh_step2_action_states(lightweight=False)
        except Exception:
            pass


bind_annotation_tab_delegates(AnnotationTab)
