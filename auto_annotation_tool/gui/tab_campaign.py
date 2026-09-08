#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZakĹ‚adka: Panel kampanii i etapow projektu.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from datetime import datetime
from time import perf_counter
import xml.etree.ElementTree as ET
import shutil
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_transition_graph import TRAINING_STAGE_CHARS, TRAINING_STAGE_PLATES
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from . import campaign_model_status
from . import campaign_step1_assets
from . import campaign_step1_ingest
from . import campaign_stage_ui
from . import campaign_stage_logic
from . import campaign_navigation
from . import campaign_iteration_flow
from . import campaign_dashboard_ui
from . import campaign_graph_actions
from . import campaign_shell_ui
from . import campaign_assistant
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
from .campaign_models import WizardStageStatus


class CampaignTab:
    _PROJECT_VIEW_CACHE_SCHEMA_VERSION = 3
    STEP1_CHAR_MIN_IMAGES = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_IMAGES", 10) or 10)
    STEP2_PLATE_MIN_PLATES = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    STEP3_CHAR_MIN_PLATES = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.icon_manager = IconManager
        self.frame = ttk.Frame(parent)
        self._startup_ui_ready = False

        # UI state
        self.right_panel = None
        self.right_graph_host = None
        self.right_graph_content = None
        self.right_panel_canvas = None
        self.right_content = None
        self.right_content_window = None
        self.left_panel_host = None
        self.right_sidebar_tab_host = None
        self.right_sidebar_tab_btn = None
        self.right_sidebar_tab_canvas = None
        self.right_sidebar_collapse_btn = None
        self._project_side_panel_collapsed = False
        self._campaign_graph_fullscreen = True
        self._campaign_graph_prev_side_panel_collapsed = False
        self._model_status_title_labels = []
        self._model_status_meta_labels = []
        self.project_listbox = None
        self.project_list_scrollbar = None
        self.project_list_host = None
        self.project_list_status_lbl = None
        self.project_list_status_labels = []
        self._project_name_by_index = []
        self._project_list_refreshing = False
        self.project_start_mode_var = tk.StringVar(master=self.frame, value="")
        self.ingest_master_pool_var = tk.StringVar(master=self.frame, value="")
        self.ingest_status_labels = []
        self.ingest_status_shell = None
        self.ingest_status_summary_lbl = None
        self.ingest_status_table = None
        self.ingest_status_meta_lbl = None
        self.ingest_status_table_value_labels = []
        self.ingest_plan_listbox = None
        self.ingest_plan_host = None
        self.ingest_plan_items = []
        self.current_ingest_plan = {}
        self.ingest_panel_frame = None
        self.step1_panel_expanded = False
        self.step1_ingest_host_item = None
        self.last_ingest_snapshot = {}
        self._existing_iteration_ingest_plan_signature = None
        self.ingest_start_shell = None
        self.ingest_start_panel = None
        self.ingest_start_title_lbl = None
        self.ingest_start_summary_lbl = None
        self.ingest_start_assets_table_shell = None
        self.ingest_start_assets_table = None
        self.ingest_start_assets_title_lbl = None
        self.ingest_start_asset_row_widgets = {}
        self.project_start_asset_scope_vars = {}
        self.btn_ingest_asset_more = {}
        self.btn_ingest_clear_asset = {}
        self.btn_ingest_start_fresh = None
        self.btn_ingest_start_assets = None
        self.btn_ingest_import_plate_run = None
        self.btn_ingest_pick_plate_model = None
        self.btn_ingest_pick_char_model = None
        self.btn_ingest_master_analysis = None
        self.btn_apply_ingest_plan = None
        self.ingest_list_title_lbl = None
        self.ingest_logic_lbl = None
        self.ingest_selection_lbl = None
        self.ingest_balance_canvas = None
        self.ingest_balance_summary_lbl = None
        self.ingest_insights_shell = None
        self.ingest_insights_toggle_shell = None
        self.ingest_insights_toggle_btn = None
        self.ingest_insights_hint_lbl = None
        self.ingest_insights_expanded = False
        self.ingest_chart_panel = None
        self.ingest_info_panel = None
        self.campaign_banner_shell = None
        self.banner_progress_row = None
        self.wizard_header_shell = None
        self.wizard_header_title_lbl = None
        self.wizard_header_summary_lbl = None
        self.wizard_header_metro_canvas = None
        self.wizard_exit_button_canvas = None
        self._wizard_header_metro_photo = None
        self.project_add_button_canvas = None
        self.project_browser_footer = None
        self.project_status_top_row = None
        self._icon_button_images = {}
        self._pil_font_cache = {}
        self._icon_button_state = {
            "project_add": {"hover": False, "pressed": False, "enabled": True},
            "exit_project": {"hover": False, "pressed": False, "enabled": False},
        }
        self._scroll_inertia_jobs = {}
        self._iteration_target_change_after_id = None
        self._dashboard_perf_cache = {
            "image_counts": {},
            "json_payloads": {},
            "model_created": {},
            "approved_stats": {},
            "step2_source_states": {},
            "step2_view_models": {},
        }
        self._wizard_header_metro_statuses = []
        self._wizard_step2_target_var = tk.StringVar(master=self.frame, value="")
        self.wizard_empty_state_card = None
        self.wizard_stage_cards = {}
        self.wizard_stage_cards_host = None
        self.wizard_transition_graph_shell = None
        self.wizard_transition_graph_body = None
        self.campaign_transition_graph_canvas = None
        self._campaign_graph_gate_offsets = {}
        self.project_loading_overlay = None
        self.project_loading_card = None
        self.project_loading_title_lbl = None
        self.project_loading_body_lbl = None
        self.project_loading_progress = None
        self._project_loading_overlay_visible = False
        self._project_open_refresh_after_id = None
        self._project_open_context_after_id = None
        self._project_open_post_refresh_after_id = None
        self._wizard_stage_badge_stabilize_after_id = None
        self._project_switch_in_progress = False
        self._wizard_focus_stage_request = ""
        self._wizard_focus_after_id = None
        self._wizard_assistant_stage_key = ""
        self._iteration_advance_thread = None
        self._iteration_advance_result = None
        self._iteration_advance_mode = ""
        self._iteration_advance_poll_after_id = None

        self._build_ui()
        self._refresh_dashboard()
        self.frame.after_idle(self._mark_startup_ui_ready)

    def _mark_startup_ui_ready(self):
        self._startup_ui_ready = True

    def _get_project_list_selection_bg(self) -> str:
        try:
            return self.app.get_list_selection_colors()[0]
        except Exception:
            return "#f8fafc"

    def _get_project_list_selection_fg(self) -> str:
        try:
            return self.app.get_list_selection_colors()[1]
        except Exception:
            return "#111827"

    def _get_campaign_green_accent(self) -> str:
        palette = getattr(self.app, "palette", {})
        success = palette.get("success", "#27ae60")
        panel = palette.get("panel", "#252526")
        try:
            return blend_hex_colors(success, panel, 0.10)
        except Exception:
            return "#2b8f57"

    def is_startup_ui_ready(self) -> bool:
        return bool(getattr(self, "_startup_ui_ready", False))

    def get_free_mode_assistant_context(self) -> dict:
        return campaign_assistant.get_free_mode_assistant_context(self)

    def _ensure_wizard_stage_ui_ready(self) -> None:
        def _exists(widget) -> bool:
            try:
                return bool(widget is not None and widget.winfo_exists())
            except Exception:
                return False

        if (
            not _exists(getattr(self, "wizard_transition_graph_shell", None))
            or not _exists(getattr(self, "wizard_transition_graph_body", None))
        ):
            self._rebuild_wizard_stage_ui()

    def _clear_dashboard_perf_cache(self, *args, **kwargs):
        return campaign_dashboard_cache._clear_dashboard_perf_cache(self, *args, **kwargs)
    @staticmethod
    def _build_cache_token_for_path(*args, **kwargs):
        return campaign_dashboard_cache._build_cache_token_for_path(*args, **kwargs)
    def _get_dashboard_cache_bucket(self, *args, **kwargs):
        return campaign_dashboard_cache._get_dashboard_cache_bucket(self, *args, **kwargs)
    def _get_project_view_cache_path(self, *args, **kwargs):
        return campaign_dashboard_cache._get_project_view_cache_path(self, *args, **kwargs)
    @classmethod
    def _normalize_project_view_cache_value(cls, *args, **kwargs):
        return campaign_dashboard_cache._normalize_project_view_cache_value(cls, *args, **kwargs)
    @classmethod
    def _serialize_project_view_cache_signature(cls, *args, **kwargs):
        return campaign_dashboard_cache._serialize_project_view_cache_signature(cls, *args, **kwargs)
    def _build_directory_children_signature(self, *args, **kwargs):
        return campaign_dashboard_cache._build_directory_children_signature(self, *args, **kwargs)
    def _load_project_view_cache_store(self, *args, **kwargs):
        return campaign_dashboard_cache._load_project_view_cache_store(self, *args, **kwargs)
    def _save_project_view_cache_store(self, *args, **kwargs):
        return campaign_dashboard_cache._save_project_view_cache_store(self, *args, **kwargs)
    def _get_project_view_cache_entry(self, *args, **kwargs):
        return campaign_dashboard_cache._get_project_view_cache_entry(self, *args, **kwargs)
    def _set_project_view_cache_entry(self, *args, **kwargs):
        return campaign_dashboard_cache._set_project_view_cache_entry(self, *args, **kwargs)
    def _build_step1_manifest_context_signature(self, *args, **kwargs):
        return campaign_dashboard_cache._build_step1_manifest_context_signature(self, *args, **kwargs)
    def _build_step2_source_state_signature(self, *args, **kwargs):
        return campaign_dashboard_cache._build_step2_source_state_signature(self, *args, **kwargs)
    @staticmethod
    def _elapsed_ms(*args, **kwargs):
        return campaign_dashboard_cache._elapsed_ms(*args, **kwargs)
    def _log_perf(self, *args, **kwargs):
        return campaign_dashboard_cache._log_perf(self, *args, **kwargs)
    def _build_ui(self):
        return campaign_shell_ui._build_ui(self)

    def _sync_left_panel_scrollregion(self, event=None):
        return campaign_shell_ui._sync_left_panel_scrollregion(self, event)

    def _sync_left_panel_canvas_width(self, event=None):
        return campaign_shell_ui._sync_left_panel_canvas_width(self, event)

    def _sync_right_panel_scrollregion(self, event=None):
        return campaign_shell_ui._sync_right_panel_scrollregion(self, event)

    def _sync_right_panel_canvas_width(self, event=None):
        return campaign_shell_ui._sync_right_panel_canvas_width(self, event)
    def _set_project_side_panel_collapsed(self, *args, **kwargs):
        return campaign_shell_ui._set_project_side_panel_collapsed(self, *args, **kwargs)
    def _toggle_project_side_panel(self, *args, **kwargs):
        return campaign_shell_ui._toggle_project_side_panel(self, *args, **kwargs)
    def _toggle_campaign_graph_fullscreen(self, *args, **kwargs):
        return campaign_shell_ui._toggle_campaign_graph_fullscreen(self, *args, **kwargs)
    def _apply_campaign_graph_fullscreen_layout(self, *args, **kwargs):
        return campaign_shell_ui._apply_campaign_graph_fullscreen_layout(self, *args, **kwargs)

    def _wizard_stage_key_from_step_num(step_num: int | None) -> str:
        return campaign_assistant._wizard_stage_key_from_step_num(step_num)

    def request_wizard_stage_focus(self, step_num: int | None = None, *, stage_key: str | None = None) -> None:
        return campaign_assistant.request_wizard_stage_focus(self, step_num, stage_key=stage_key)

    def _set_wizard_assistant_stage_context(self, stage_key: str) -> None:
        return campaign_assistant._set_wizard_assistant_stage_context(self, stage_key)

    def _get_wizard_assistant_stage_status(self) -> WizardStageStatus | None:
        return campaign_assistant._get_wizard_assistant_stage_status(self)

    @staticmethod
    def _wizard_state_assistant_label(state: str) -> str:
        return campaign_assistant._wizard_state_assistant_label(state)

    @staticmethod
    def _is_step3_z2_repair_status(status: WizardStageStatus) -> bool:
        return campaign_assistant._is_step3_z2_repair_status(status)

    def _build_wizard_stage_assistant_context(self, status: WizardStageStatus) -> dict:
        return campaign_assistant._build_wizard_stage_assistant_context(self, status)

    def _cancel_pending_wizard_stage_focus(self) -> None:
        return campaign_assistant._cancel_pending_wizard_stage_focus(self)

    def _schedule_pending_wizard_stage_focus(self, *, attempts_left: int = 4) -> None:
        return campaign_assistant._schedule_pending_wizard_stage_focus(self, attempts_left=attempts_left)

    def _apply_pending_wizard_stage_focus(self) -> bool:
        return campaign_assistant._apply_pending_wizard_stage_focus(self)

    def _widget_contains_point(self, *args, **kwargs):
        return campaign_ui_helpers._widget_contains_point(self, *args, **kwargs)
    def _mousewheel_units(self, *args, **kwargs):
        return campaign_ui_helpers._mousewheel_units(self, *args, **kwargs)
    def _mousewheel_magnitude(self, *args, **kwargs):
        return campaign_ui_helpers._mousewheel_magnitude(self, *args, **kwargs)
    def _compute_canvas_scroll_delta(self, *args, **kwargs):
        return campaign_ui_helpers._compute_canvas_scroll_delta(self, *args, **kwargs)
    def _compute_listbox_scroll_delta(self, *args, **kwargs):
        return campaign_ui_helpers._compute_listbox_scroll_delta(self, *args, **kwargs)
    def _apply_canvas_scroll_delta(self, *args, **kwargs):
        return campaign_ui_helpers._apply_canvas_scroll_delta(self, *args, **kwargs)
    def _apply_listbox_scroll_delta(self, *args, **kwargs):
        return campaign_ui_helpers._apply_listbox_scroll_delta(self, *args, **kwargs)
    def _queue_scroll_inertia(self, *args, **kwargs):
        return campaign_ui_helpers._queue_scroll_inertia(self, *args, **kwargs)
    def _advance_scroll_inertia(self, *args, **kwargs):
        return campaign_ui_helpers._advance_scroll_inertia(self, *args, **kwargs)
    def _canvas_can_scroll(self, *args, **kwargs):
        return campaign_ui_helpers._canvas_can_scroll(self, *args, **kwargs)
    def _listbox_can_scroll(self, *args, **kwargs):
        return campaign_ui_helpers._listbox_can_scroll(self, *args, **kwargs)
    def _on_listbox_mousewheel(self, *args, **kwargs):
        return campaign_ui_helpers._on_listbox_mousewheel(self, *args, **kwargs)
    def _on_global_mousewheel(self, *args, **kwargs):
        return campaign_ui_helpers._on_global_mousewheel(self, *args, **kwargs)
    def _bind_icon_button(self, *args, **kwargs):
        return campaign_ui_helpers._bind_icon_button(self, *args, **kwargs)
    def _set_icon_button_enabled(self, *args, **kwargs):
        return campaign_ui_helpers._set_icon_button_enabled(self, *args, **kwargs)
    def _set_icon_button_visual(self, *args, **kwargs):
        return campaign_ui_helpers._set_icon_button_visual(self, *args, **kwargs)
    def _on_icon_button_release(self, *args, **kwargs):
        return campaign_ui_helpers._on_icon_button_release(self, *args, **kwargs)
    def _get_pil_font(self, *args, **kwargs):
        return campaign_ui_helpers._get_pil_font(self, *args, **kwargs)
    @staticmethod
    def _hex_to_rgba(*args, **kwargs):
        return campaign_ui_helpers._hex_to_rgba(*args, **kwargs)
    @classmethod
    def _pick_readable_text_color(cls, *args, **kwargs):
        return campaign_ui_helpers._pick_readable_text_color(cls, *args, **kwargs)
    @staticmethod
    def _format_tail_text(*args, **kwargs):
        return campaign_ui_helpers._format_tail_text(*args, **kwargs)
    def _style_project_start_badge_button(self, *args, **kwargs):
        return campaign_ui_helpers._style_project_start_badge_button(self, *args, **kwargs)
    def _create_project_start_badge_button(self, *args, **kwargs):
        return campaign_ui_helpers._create_project_start_badge_button(self, *args, **kwargs)
    def _set_project_start_badge_button_state(self, *args, **kwargs):
        return campaign_ui_helpers._set_project_start_badge_button_state(self, *args, **kwargs)
    def _draw_icon_button(self, *args, **kwargs):
        return campaign_ui_helpers._draw_icon_button(self, *args, **kwargs)
    def _build_projects_browser(self, parent):
        return campaign_project_browser._build_projects_browser(self, parent)

    def _build_model_status(self, *args, **kwargs):
        return campaign_model_status._build_model_status(self, *args, **kwargs)
    def _format_model_created_label(self, *args, **kwargs):
        return campaign_model_status._format_model_created_label(self, *args, **kwargs)
    def _get_model_identity_label(self, *args, **kwargs):
        return campaign_model_status._get_model_identity_label(self, *args, **kwargs)
    def _format_model_meta_label(self, *args, **kwargs):
        return campaign_model_status._format_model_meta_label(self, *args, **kwargs)
    def _count_images_in_dir(self, *args, **kwargs):
        return campaign_model_status._count_images_in_dir(self, *args, **kwargs)
    def _collect_image_names_in_dir(self, *args, **kwargs):
        return campaign_model_status._collect_image_names_in_dir(self, *args, **kwargs)
    def _collect_previous_project_image_names_for_step1(self, *args, **kwargs):
        return campaign_model_status._collect_previous_project_image_names_for_step1(self, *args, **kwargs)
    @staticmethod
    def _normalize_project_start_mode(*args, **kwargs):
        return campaign_step1_assets._normalize_project_start_mode(*args, **kwargs)
    def _is_first_iteration_start_context(self, *args, **kwargs):
        return campaign_step1_assets._is_first_iteration_start_context(self, *args, **kwargs)
    def _is_step1_operational_context(self, *args, **kwargs):
        return campaign_step1_assets._is_step1_operational_context(self, *args, **kwargs)
    def _get_step1_presentation_mode(self, *args, **kwargs):
        return campaign_step1_assets._get_step1_presentation_mode(self, *args, **kwargs)
    def _get_project_start_mode(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_mode(self, *args, **kwargs)
    def _get_active_step1_draft_plan(self, *args, **kwargs):
        return campaign_step1_assets._get_active_step1_draft_plan(self, *args, **kwargs)
    def _set_project_start_mode(self, *args, **kwargs):
        return campaign_step1_assets._set_project_start_mode(self, *args, **kwargs)
    def _choose_project_start_model(self, *args, **kwargs):
        return campaign_step1_assets._choose_project_start_model(self, *args, **kwargs)
    def _project_start_model_candidate_roots(self, *args, **kwargs):
        return campaign_step1_assets._project_start_model_candidate_roots(self, *args, **kwargs)
    def _summarize_project_start_model_candidate(self, *args, **kwargs):
        return campaign_step1_assets._summarize_project_start_model_candidate(self, *args, **kwargs)
    def _find_project_start_model_candidates(self, *args, **kwargs):
        return campaign_step1_assets._find_project_start_model_candidates(self, *args, **kwargs)
    def _open_project_start_model_candidate_browser(self, *args, **kwargs):
        return campaign_step1_assets._open_project_start_model_candidate_browser(self, *args, **kwargs)
    @staticmethod
    def _looks_like_character_model_classes(class_names: list[str]) -> bool:
        normalized = [str(name).strip().upper() for name in class_names if str(name).strip()]
        if len(normalized) < 8:
            return False

        allowed = set(CHAR_ALPHABET)
        for token in normalized:
            if len(token) != 1 or token not in allowed:
                return False
        return True

    @staticmethod
    def _is_project_model_pose_info(info: dict) -> bool:
        task = str(info.get("task") or "").strip().lower()
        inferred_type = str(info.get("type") or "").strip().lower()
        kpt_shape = info.get("kpt_shape")
        has_keypoints = bool(info.get("keypoints")) or bool(kpt_shape)
        if has_keypoints or task == "pose" or inferred_type == "pose":
            return True
        if task in {"detect", "detection"} or inferred_type in {"detect", "detection"}:
            return False
        architecture_text = " ".join(
            str(info.get(key) or "").strip().lower()
            for key in ("architecture_label", "source_architecture_label")
        )
        return "pose" in architecture_text

    def _validate_project_model_selection(self, model_type: str, model_path: Path) -> tuple[bool, str]:
        ok, message, info = validate_model_file(model_path)
        if not ok:
            return False, message or "Nie udaĹ‚o siÄ™ odczytaÄ‡ modelu."

        is_pose = self._is_project_model_pose_info(info)
        class_names = [str(name).strip() for name in (info.get("classes") or []) if str(name).strip()]
        class_names_lower = [name.lower() for name in class_names]
        joined_names = " ".join(class_names_lower)
        looks_like_char_model = self._looks_like_character_model_classes(class_names)

        if model_type == "plate":
            if looks_like_char_model:
                return False, "To wygląda na model znaków YOLO Detect, nie model tablic YOLO Pose."
            if not is_pose:
                return False, "Model tablic musi byÄ‡ modelem YOLO Pose z punktami kluczowymi."
            return True, ""

        if model_type == "char":
            if is_pose:
                return False, "Model znakĂłw nie moĹĽe byÄ‡ modelem Pose."
            if not class_names:
                return False, "Nie udaĹ‚o siÄ™ odczytaÄ‡ klas modelu znakĂłw z pliku .pt."
            if not looks_like_char_model:
                return False, "Model znakĂłw powinien mieÄ‡ klasy znakĂłw 0-9 i A-Z."
            return True, ""

        if model_type == "vehicle":
            if is_pose:
                return False, "Model pojazdĂłw powinien byÄ‡ modelem detekcyjnym, nie Pose."
            if class_names and looks_like_char_model:
                return False, "To wyglÄ…da na model znakĂłw, nie pojazdĂłw."
            vehicle_markers = (
                "vehicle", "car", "truck", "bus", "motorcycle", "motorbike", "van",
                "pickup", "suv", "pojazd", "samochod", "auto"
            )
            if class_names and not any(marker in joined_names for marker in vehicle_markers):
                return False, "To nie wyglÄ…da na model pojazdĂłw: w klasach nie widaÄ‡ typowych znacznikow pojazdĂłw."
            return True, ""

        return True, ""

    def _resolve_project_start_run_images_dir(self, *args, **kwargs):
        return campaign_step1_assets._resolve_project_start_run_images_dir(self, *args, **kwargs)
    @staticmethod
    def _normalize_project_start_asset_full_path(*args, **kwargs):
        return campaign_step1_assets._normalize_project_start_asset_full_path(*args, **kwargs)
    def _format_workspace_relative_path(self, path_like) -> str:
        raw_text = self._normalize_project_start_asset_full_path(path_like)
        if not raw_text:
            return ""
        try:
            workspace_root = Path(CONFIG.WORKSPACE_DIR).resolve()
            return os.path.relpath(raw_text, str(workspace_root))
        except Exception:
            try:
                candidate = Path(raw_text)
                parent_name = str(candidate.parent.name or "").strip()
                if parent_name:
                    return f"{parent_name}/{candidate.name}"
                return candidate.name
            except Exception:
                return str(raw_text)

    def _format_project_relative_path(self, path_value: str) -> str:
        raw_text = self._normalize_project_start_asset_full_path(path_value)
        if not raw_text:
            return ""
        try:
            project_root = CAMPAIGN.get_active_project_root_dir()
            if project_root is not None:
                project_root = Path(project_root).resolve()
                candidate = Path(raw_text)
                return os.path.relpath(str(candidate), str(project_root))
        except Exception:
            pass
        return self._format_workspace_relative_path(raw_text)

    def _format_project_start_asset_source(self, *args, **kwargs):
        return campaign_step1_assets._format_project_start_asset_source(self, *args, **kwargs)
    def _get_project_start_effective_model_state(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_effective_model_state(self, *args, **kwargs)
    def _collect_project_start_image_names(self, *args, **kwargs):
        return campaign_step1_assets._collect_project_start_image_names(self, *args, **kwargs)
    def _build_project_start_image_set_token(self, *args, **kwargs):
        return campaign_step1_assets._build_project_start_image_set_token(self, *args, **kwargs)
    def _build_project_start_plate_xml_image_set_token(self, *args, **kwargs):
        return campaign_step1_assets._build_project_start_plate_xml_image_set_token(self, *args, **kwargs)
    def _show_project_start_asset_source_context_menu(self, *args, **kwargs):
        return campaign_step1_assets._show_project_start_asset_source_context_menu(self, *args, **kwargs)
    def _copy_project_start_asset_source_path(self, *args, **kwargs):
        return campaign_step1_assets._copy_project_start_asset_source_path(self, *args, **kwargs)
    def _is_project_start_asset_clearable(self, *args, **kwargs):
        return campaign_step1_assets._is_project_start_asset_clearable(self, *args, **kwargs)
    def _refresh_project_start_clear_buttons(self, *args, **kwargs):
        return campaign_step1_assets._refresh_project_start_clear_buttons(self, *args, **kwargs)
    def _clear_project_start_asset(self, *args, **kwargs):
        return campaign_step1_assets._clear_project_start_asset(self, *args, **kwargs)
    @staticmethod
    def _short_project_start_asset_validation(*args, **kwargs):
        return campaign_step1_assets._short_project_start_asset_validation(*args, **kwargs)
    def _get_step1_ingest_frame_bg(self, *args, **kwargs):
        return campaign_step1_assets._get_step1_ingest_frame_bg(self, *args, **kwargs)
    def _get_project_start_asset_scope(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_asset_scope(self, *args, **kwargs)
    def _set_project_start_asset_scope(self, *args, **kwargs):
        return campaign_step1_assets._set_project_start_asset_scope(self, *args, **kwargs)
    def _restore_project_start_asset_scopes_from_state(self, *args, **kwargs):
        return campaign_step1_assets._restore_project_start_asset_scopes_from_state(self, *args, **kwargs)
    def _select_project_start_asset_scope(self, *args, **kwargs):
        return campaign_step1_assets._select_project_start_asset_scope(self, *args, **kwargs)
    @staticmethod
    def _path_is_inside_root(*args, **kwargs):
        return campaign_step1_assets._path_is_inside_root(*args, **kwargs)
    def _infer_project_start_asset_scope_from_path(self, *args, **kwargs):
        return campaign_step1_assets._infer_project_start_asset_scope_from_path(self, *args, **kwargs)
    def _get_project_start_plate_source_info(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_plate_source_info(self, *args, **kwargs)
    def _get_project_start_asset_initial_dir(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_asset_initial_dir(self, *args, **kwargs)
    def _sync_iteration_artifact_registry_from_project_start(self, *args, **kwargs):
        return campaign_step1_assets._sync_iteration_artifact_registry_from_project_start(self, *args, **kwargs)
    def _set_project_start_asset_row_state(self, *args, **kwargs):
        return campaign_step1_assets._set_project_start_asset_row_state(self, *args, **kwargs)
    def _refresh_project_start_assets_table_theme(self, *args, **kwargs):
        if getattr(self, "ingest_start_assets_table", None) is None:
            return None
        return campaign_step1_assets._refresh_project_start_assets_table_theme(self, *args, **kwargs)
    def _render_ingest_balance_chart(self, *args, **kwargs):
        return campaign_step1_assets._render_ingest_balance_chart(self, *args, **kwargs)
    def _show_project_start_images_analysis_dialog(self, *args, **kwargs):
        return campaign_step1_assets._show_project_start_images_analysis_dialog(self, *args, **kwargs)
    def _get_project_start_effective_images_source(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_effective_images_source(self, *args, **kwargs)
    def _scan_project_start_normalized_image_names(self, *args, **kwargs):
        return campaign_step1_assets._scan_project_start_normalized_image_names(self, *args, **kwargs)
    def _collect_project_start_image_paths_by_normalized_name(self, *args, **kwargs):
        return campaign_step1_assets._collect_project_start_image_paths_by_normalized_name(self, *args, **kwargs)
    def _maybe_extend_project_start_images_from_annotation_package(self, *args, **kwargs):
        return campaign_step1_assets._maybe_extend_project_start_images_from_annotation_package(self, *args, **kwargs)
    def _get_project_start_approved_normalized_image_names(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_approved_normalized_image_names(self, *args, **kwargs)
    def _get_project_start_adoptable_normalized_image_names(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_adoptable_normalized_image_names(self, *args, **kwargs)
    def _show_project_start_asset_details(self, *args, **kwargs):
        return campaign_step1_assets._show_project_start_asset_details(self, *args, **kwargs)
    @staticmethod
    def _count_project_start_plate_detections(*args, **kwargs):
        return campaign_step1_assets._count_project_start_plate_detections(*args, **kwargs)
    @staticmethod
    def _get_project_start_filename_plate_texts(*args, **kwargs):
        return campaign_step1_assets._get_project_start_filename_plate_texts(*args, **kwargs)
    def _project_start_annotation_covers_filename_plates(self, *args, **kwargs):
        return campaign_step1_assets._project_start_annotation_covers_filename_plates(self, *args, **kwargs)
    @staticmethod
    def _format_project_start_annotation_adoption_summary(*args, **kwargs):
        return campaign_step1_assets._format_project_start_annotation_adoption_summary(*args, **kwargs)
    @staticmethod
    def _is_project_start_manual_plate_detection(*args, **kwargs):
        return campaign_step1_assets._is_project_start_manual_plate_detection(*args, **kwargs)
    def _load_project_start_annotation_run_manifest(self, *args, **kwargs):
        return campaign_step1_assets._load_project_start_annotation_run_manifest(self, *args, **kwargs)
    def _summarize_project_start_annotation_import_origin(self, *args, **kwargs):
        return campaign_step1_assets._summarize_project_start_annotation_import_origin(self, *args, **kwargs)
    def _format_project_start_annotation_import_origin_table(self, *args, **kwargs):
        return campaign_step1_assets._format_project_start_annotation_import_origin_table(self, *args, **kwargs)
    def _choose_project_start_annotation_import_mode(self, *args, **kwargs):
        return campaign_step1_assets._choose_project_start_annotation_import_mode(self, *args, **kwargs)
    def _promote_project_start_annotation_import_to_approved_set(self, *args, **kwargs):
        return campaign_step1_assets._promote_project_start_annotation_import_to_approved_set(self, *args, **kwargs)
    def _check_project_start_run_compatibility(self, *args, **kwargs):
        return campaign_step1_assets._check_project_start_run_compatibility(self, *args, **kwargs)
    def _get_project_start_normalized_image_names(self, *args, **kwargs):
        return campaign_step1_assets._get_project_start_normalized_image_names(self, *args, **kwargs)
    def _summarize_project_start_xml_match(self, *args, **kwargs):
        return campaign_step1_assets._summarize_project_start_xml_match(self, *args, **kwargs)
    def _import_project_start_plate_run(self, *args, **kwargs):
        return campaign_step1_assets._import_project_start_plate_run(self, *args, **kwargs)
    @staticmethod
    def _get_step1_assets_intro_text(*args, **kwargs):
        return campaign_step1_assets._get_step1_assets_intro_text(*args, **kwargs)
    def _refresh_project_start_panel(self, *args, **kwargs):
        return None
    def _refresh_ingest_insights_visibility(self, *args, **kwargs):
        return campaign_step1_ingest._refresh_ingest_insights_visibility(self, *args, **kwargs)
    def _get_ingest_summary_style(self, *args, **kwargs):
        return campaign_step1_ingest._get_ingest_summary_style(self, *args, **kwargs)
    def _sync_ingest_wraps(self, *args, **kwargs):
        return campaign_step1_ingest._sync_ingest_wraps(self, *args, **kwargs)
    def _set_ingest_status_lines(self, *args, **kwargs):
        return campaign_step1_ingest._set_ingest_status_lines(self, *args, **kwargs)
    def _get_iteration_image_count(self) -> int:
        if bool(getattr(self, "_project_open_lightweight_refresh", False)):
            try:
                summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
            except Exception:
                summary = {}
            if summary:
                try:
                    summary_iter = int(summary.get("iteration", 0) or 0)
                    current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
                except Exception:
                    summary_iter = 0
                    current_iter = 1
                summary_project = str(summary.get("project", "") or "").strip()
                current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
                if summary_iter == current_iter and (not summary_project or summary_project == current_project):
                    for key in ("selected_total", "raw_total", "source_new_to_project_total"):
                        try:
                            count = int(summary.get(key, 0) or 0)
                        except Exception:
                            count = 0
                        if count > 0:
                            return int(count)
        try:
            return int(CAMPAIGN.get_iteration_image_count() or 0)
        except Exception:
            iter_dir = CAMPAIGN.get_iteration_raw_dir()
            return self._count_images_in_dir(iter_dir, recursive=True)

    def _load_ingest_manifest_cached(self, *args, **kwargs):
        return campaign_step1_ingest._load_ingest_manifest_cached(self, *args, **kwargs)
    def _load_previous_iteration_ingest_manifest(self, *args, **kwargs):
        return campaign_step1_ingest._load_previous_iteration_ingest_manifest(self, *args, **kwargs)
    def _load_latest_ingest_plan_for_current_iteration(self, *args, **kwargs):
        return campaign_step1_ingest._load_latest_ingest_plan_for_current_iteration(self, *args, **kwargs)
    def _normalize_ingest_plan_for_display(self, *args, **kwargs):
        return campaign_step1_ingest._normalize_ingest_plan_for_display(self, *args, **kwargs)
    def _build_ingest_plan_from_manifest_for_display(self, *args, **kwargs):
        return campaign_step1_ingest._build_ingest_plan_from_manifest_for_display(self, *args, **kwargs)
    def _build_ingest_plan_from_iteration_dir_for_display(self, *args, **kwargs):
        return campaign_step1_ingest._build_ingest_plan_from_iteration_dir_for_display(self, *args, **kwargs)
    def _restore_ingest_plan_from_existing_iteration(self, *args, **kwargs):
        return campaign_step1_ingest._restore_ingest_plan_from_existing_iteration(self, *args, **kwargs)
    def _recalculate_current_ingest_plan(self, *args, **kwargs):
        return campaign_step1_ingest._recalculate_current_ingest_plan(self, *args, **kwargs)
    def _format_histogram_compact(self, char_hist: dict, limit: int = 5) -> str:
        if not isinstance(char_hist, dict):
            return "brak"

        items = [
            (str(ch), int(value))
            for ch, value in char_hist.items()
            if int(value or 0) > 0
        ]
        if not items:
            return "brak"

        items.sort(key=lambda entry: (-entry[1], entry[0]))
        return ", ".join(f"{ch}:{value}" for ch, value in items[:limit])

    def _get_selected_ingest_indices(self, *args, **kwargs):
        return campaign_step1_ingest._get_selected_ingest_indices(self, *args, **kwargs)
    def _refresh_ingest_logic_text(self, *args, **kwargs):
        return campaign_step1_ingest._refresh_ingest_logic_text(self, *args, **kwargs)
    def _refresh_ingest_selection_info(self, *args, **kwargs):
        return campaign_step1_ingest._refresh_ingest_selection_info(self, *args, **kwargs)
    def _refresh_ingest_balance_chart(self, *args, **kwargs):
        return campaign_step1_ingest._refresh_ingest_balance_chart(self, *args, **kwargs)
    def _theme_step1_ingest_panel(self, *args, **kwargs):
        if getattr(self, "ingest_panel_frame", None) is None:
            return None
        return campaign_step1_ingest._theme_step1_ingest_panel(self, *args, **kwargs)
    def _populate_ingest_plan_list(self, *args, **kwargs):
        return campaign_step1_ingest._populate_ingest_plan_list(self, *args, **kwargs)
    def _refresh_ingest_panel(self, *args, **kwargs):
        if getattr(self, "ingest_panel_frame", None) is None:
            try:
                master_pool = CAMPAIGN.get_master_pool_dir()
                self.ingest_master_pool_var.set(str(master_pool) if master_pool else "")
            except Exception:
                pass
            if bool(getattr(self, "_campaign_resource_modal_suppress_graph_refresh", False)):
                return None
            try:
                self._refresh_wizard_transition_graph()
            except Exception:
                pass
            return None
        return campaign_step1_ingest._refresh_ingest_panel(self, *args, **kwargs)
    def _choose_master_pool_dir(self, parent=None) -> bool:
        if not CAMPAIGN.get_active_project_name():
            return False

        current = CAMPAIGN.get_master_pool_dir()
        initial = Path(CONFIG.DIR_1_RAW)
        if current:
            try:
                current_path = Path(current)
                if current_path.exists() and current_path.is_dir():
                    initial_parent = current_path.parent
                    initial = initial_parent if initial_parent.exists() and initial_parent.is_dir() else current_path
                elif current_path.parent.exists() and current_path.parent.is_dir():
                    initial = current_path.parent
            except Exception:
                initial = Path(CONFIG.DIR_1_RAW)
        dialog_title = "Wybierz obrazy tej iteracji"
        if not self._is_first_iteration_start_context():
            dialog_title = "Wybierz katalog głównej puli zdjęć dla aktywnego projektu"
        selected = filedialog.askdirectory(
            initialdir=str(initial),
            title=dialog_title,
            parent=parent or self.frame,
        )
        if not selected:
            return False

        campaign_step1_ingest._show_ingest_plan_progress_dialog(self, selected, parent=parent)
        campaign_step1_ingest._update_ingest_plan_progress_dialog(
            self,
            6,
            "Weryfikuję wybrany katalog obrazów.",
            "Sprawdzam, czy wskazany folder zawiera obrazy i czy nie jest zbyt szeroki.",
        )
        if not CAMPAIGN.set_master_pool_dir(selected):
            campaign_step1_ingest._hide_ingest_plan_progress_dialog(self)
            self.app.themed_info(
                "Nieprawidłowy katalog zdjęć",
                (
                    "Wybrany katalog jest zbyt szeroki albo nie zawiera obrazów. "
                    "Wskaż konkretny katalog ze zdjęciami, a nie katalog aplikacji, Workspace ani katalog projektu."
                ),
                parent=self.frame,
                tone="warning",
            )
            return False
        try:
            campaign_step1_ingest._update_ingest_plan_progress_dialog(
                self,
                12,
                "Aktualizuję kontrakt zasobów E1.",
                "Synchronizuję wybrane obrazy z aktualną iteracją projektu.",
            )
            self._sync_iteration_artifact_registry_from_project_start(
                progress_callback=lambda _count, detail: campaign_step1_ingest._update_ingest_plan_progress_dialog(
                    self,
                    14,
                    "Buduję opis wybranego zbioru obrazów.",
                    str(detail or "Zbieram nazwy obrazów do kontraktu O."),
                )
            )
        except Exception:
            pass
        self.current_ingest_plan = {}
        iter_image_count = self._get_iteration_image_count()

        if iter_image_count == 0:
            try:
                self.app.update_status("Wybrano katalog zdjęć. Przygotowuję plan wejścia E1...", "info")
            except Exception:
                pass
            self._generate_ingest_plan(parent=parent)
        else:
            campaign_step1_ingest._update_ingest_plan_progress_dialog(
                self,
                86,
                "Odświeżam informacje o wybranym zbiorze obrazów.",
                "W tej iteracji istnieją już obrazy, więc nie tworzę nowego planu E1.",
            )
            self._refresh_ingest_panel()
            campaign_step1_ingest._update_ingest_plan_progress_dialog(
                self,
                100,
                "Zapisano wybór katalogu obrazów.",
                "Gotowe.",
            )
            campaign_step1_ingest._hide_ingest_plan_progress_dialog(self, delay_ms=500)
            try:
                self.app.update_status(
                    "Zapisano katalog zdjęć. W folderze iteracji są już zdjęcia, więc aktywne pozostaje zatwierdzenie E1.",
                    "info",
                )
            except Exception:
                pass
        return True

    def _build_main_pack_plan(
        self,
        master_pool_dir: Path,
        current_balance: dict | None = None,
        progress_callback=None,
    ) -> dict:
        master_pool_dir = Path(master_pool_dir)
        if not master_pool_dir.exists() or not master_pool_dir.is_dir():
            raise FileNotFoundError(f"Główna pula zdjęć nie istnieje: {master_pool_dir}")

        def _progress(progress: float, message: str = "", detail: str = "", *, force: bool = False) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(progress, message, detail=detail, force=force)
            except TypeError:
                try:
                    progress_callback(progress, message)
                except Exception:
                    pass
            except Exception:
                pass

        planner = CampaignIngestPlanner()
        selected_items = []
        selected_hist = Counter()
        skipped_invalid_gt = 0
        skipped_duplicate_filenames = 0
        skipped_duplicate_approved = 0
        project_overlap_filenames = 0
        pending_iteration_overlap_filenames = 0
        approved_registry = CAMPAIGN.get_used_image_registry()
        project_packet_registry = CAMPAIGN.get_project_packet_filename_registry(
            exclude_iteration_num=int(CAMPAIGN.get_current_iteration_num() or 1)
        )
        approved_names = {
            str(name or "").strip().lower()
            for name in (approved_registry.get("filenames", []) or [])
            if str(name or "").strip()
        }
        project_packet_names = {
            str(name or "").strip().lower()
            for name in (project_packet_registry.get("filenames", []) or [])
            if str(name or "").strip()
        }
        project_packet_total = int(project_packet_registry.get("total_filenames", 0) or 0)
        pending_iteration_names: set[str] = set()
        try:
            current_step = int(CAMPAIGN.get_current_step() or 1)
        except Exception:
            current_step = 1
        step1_status = str(CAMPAIGN.get_step1_status() or "").strip().lower()
        if current_step == 1 and step1_status != "approved":
            draft_plan = self._get_active_step1_draft_plan()
            if isinstance(draft_plan, dict) and draft_plan:
                latest_selected = list(draft_plan.get("selected") or [])
                latest_master_text = str(draft_plan.get("master_pool_dir") or "").strip()
                try:
                    latest_master_dir = Path(latest_master_text).resolve() if latest_master_text else None
                except Exception:
                    latest_master_dir = None
                try:
                    current_master_dir = master_pool_dir.resolve()
                except Exception:
                    current_master_dir = master_pool_dir
                if latest_selected and latest_master_dir is not None and latest_master_dir != current_master_dir:
                    pending_iteration_names = {
                        str(item.get("name", "") or "").strip().lower()
                        for item in latest_selected
                        if isinstance(item, dict) and str(item.get("name", "") or "").strip()
                    }

        current_counter = Counter()
        for ch in CHAR_ALPHABET:
            current_counter[ch] = int((current_balance or {}).get(ch, 0))

        _progress(
            28,
            "Skanuję wybrany zbiór obrazów.",
            "Szukam plików graficznych w katalogu i podkatalogach.",
            force=True,
        )
        image_paths = []
        last_scan_progress = perf_counter()
        for image_path in master_pool_dir.rglob("*"):
            try:
                if not image_path.is_file() or image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
            except Exception:
                continue
            image_paths.append(image_path)
            now = perf_counter()
            if len(image_paths) == 1 or len(image_paths) % 250 == 0 or now - last_scan_progress >= 0.45:
                last_scan_progress = now
                soft_progress = min(42.0, 28.0 + min(14.0, float(len(image_paths)) / 3500.0 * 14.0))
                _progress(
                    soft_progress,
                    "Skanuję wybrany zbiór obrazów.",
                    f"Znaleziono {len(image_paths)} obrazów...",
                )
        image_paths = sorted(image_paths, key=lambda p: p.as_posix().lower())
        raw_total = int(len(image_paths))
        _progress(
            44,
            "Analizuję obrazy z wybranego zbioru.",
            f"Znaleziono {raw_total} obrazów. Sprawdzam duble i nazwy plików.",
            force=True,
        )

        last_process_progress = perf_counter()

        def _report_processed_progress(processed_count: int, *, force: bool = False) -> None:
            nonlocal last_process_progress
            now = perf_counter()
            if not force and not (
                processed_count == 1
                or processed_count == raw_total
                or processed_count % 100 == 0
                or now - last_process_progress >= 0.35
            ):
                return
            last_process_progress = now
            ratio = (float(processed_count) / float(raw_total)) if raw_total > 0 else 1.0
            _progress(
                44.0 + 46.0 * ratio,
                "Analizuję obrazy z wybranego zbioru.",
                (
                    f"Przetworzono {processed_count}/{raw_total}. "
                    f"Do użycia: {len(selected_items)}, duble: {skipped_duplicate_filenames}, "
                    f"bez poprawnego GT: {skipped_invalid_gt}."
                ),
            )

        for processed_count, image_path in enumerate(image_paths, start=1):
            image_name_key = str(image_path.name or "").strip().lower()
            if image_name_key and image_name_key in project_packet_names:
                skipped_duplicate_filenames += 1
                project_overlap_filenames += 1
                if image_name_key in approved_names:
                    skipped_duplicate_approved += 1
                _report_processed_progress(processed_count)
                continue
            if image_name_key and image_name_key in pending_iteration_names:
                skipped_duplicate_filenames += 1
                pending_iteration_overlap_filenames += 1
                _report_processed_progress(processed_count)
                continue

            gt_texts = planner.extract_true_texts_from_filename(image_path.name)
            if not gt_texts:
                skipped_invalid_gt += 1
                _report_processed_progress(processed_count)
                continue

            char_hist = planner.build_char_histogram(gt_texts)
            if not char_hist:
                skipped_invalid_gt += 1
                _report_processed_progress(processed_count)
                continue

            try:
                source_path = str(image_path.resolve())
            except Exception:
                source_path = str(image_path.absolute())

            selected_hist.update(char_hist)
            selected_items.append(
                {
                    "name": image_path.name,
                    "source_path": source_path,
                    "source_key": planner.make_source_key(image_path, master_pool_dir=master_pool_dir),
                    "ground_truth_texts": list(gt_texts),
                    "char_histogram": dict(char_hist),
                    "score": 0.0,
                    "score_details": {},
                }
            )
            _report_processed_progress(processed_count)

        predicted_counter = Counter(current_counter)
        predicted_counter.update(selected_hist)
        source_new_to_project_total = max(0, int(raw_total) - int(project_overlap_filenames))
        new_to_project_total = int(len(selected_items))
        _progress(
            92,
            "Podsumowuję wynik analizy obrazów.",
            f"Do planu E1 trafi {new_to_project_total} z {raw_total} obrazów.",
            force=True,
        )

        return {
            "ok": True,
            "planner_version": "main_pack_v1",
            "generated_at": datetime.now().isoformat(),
            "project": CAMPAIGN.get_active_project_name() or "",
            "iteration": int(CAMPAIGN.get_current_iteration_num() or 1),
            "master_pool_dir": str(master_pool_dir.resolve()),
            "batch_size": 0,
            "raw_total": raw_total,
            "candidates_total": raw_total,
            "selected_total": len(selected_items),
            "new_to_project_total": new_to_project_total,
            "source_new_to_project_total": source_new_to_project_total,
            "skipped_used": skipped_duplicate_filenames,
            "skipped_duplicate_filenames": skipped_duplicate_filenames,
            "skipped_duplicate_approved_filenames": skipped_duplicate_approved,
            "project_overlap_filenames": project_overlap_filenames,
            "pending_iteration_overlap_filenames": pending_iteration_overlap_filenames,
            "project_pool_total_before_iteration": project_packet_total,
            "project_pool_total_after_iteration": project_packet_total + new_to_project_total,
            "skipped_invalid_ground_truth": skipped_invalid_gt,
            "current_balance": {ch: int(current_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
            "selected_balance": {ch: int(selected_hist.get(ch, 0)) for ch in CHAR_ALPHABET},
            "predicted_balance_after": {ch: int(predicted_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
            "selected": selected_items,
        }

    def _generate_ingest_plan(self, *args, **kwargs):
        return campaign_step1_ingest._generate_ingest_plan(self, *args, **kwargs)
    def _should_reuse_step1_source_despite_duplicate_plan(self, *args, **kwargs):
        return campaign_step1_ingest._should_reuse_step1_source_despite_duplicate_plan(self, *args, **kwargs)
    def _build_step1_source_reuse_plan(self, *args, **kwargs):
        return campaign_step1_ingest._build_step1_source_reuse_plan(self, *args, **kwargs)
    def _ensure_current_ingest_plan_from_master_pool(self, *args, **kwargs):
        return campaign_step1_ingest._ensure_current_ingest_plan_from_master_pool(self, *args, **kwargs)
    def _ensure_step1_iteration_target_selected(self, *args, **kwargs):
        return campaign_step1_ingest._ensure_step1_iteration_target_selected(self, *args, **kwargs)
    def _continue_to_step3_after_step1_if_char_ready(self, *args, **kwargs):
        return campaign_step1_ingest._continue_to_step3_after_step1_if_char_ready(self, *args, **kwargs)
    def _format_step1_adopted_annotations_z2_scope_notice(self, *args, **kwargs):
        return campaign_step1_ingest._format_step1_adopted_annotations_z2_scope_notice(self, *args, **kwargs)
    def _approve_step1_with_existing_char_material(self, *args, **kwargs):
        return campaign_step1_ingest._approve_step1_with_existing_char_material(self, *args, **kwargs)
    def _approve_current_iteration_package(
        self,
        target_iter_dir: Path,
        source_dir: Path = None,
        selected_source_files=None,
        selection_mode: str = "manual",
        proposal_summary: dict = None,
        progress_callback=None,
        selected_source_metadata=None,
    ) -> int:
        started_at = perf_counter()
        manifest_ms = 0.0
        approve_ms = 0.0
        refresh_ms = 0.0

        def _progress(value: float, message: str = "", detail: str = "", *, force: bool = False) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(value, message, detail=detail, force=force)
            except TypeError:
                try:
                    progress_callback(value, message)
                except Exception:
                    pass
            except Exception:
                pass

        _progress(4, "Przygotowuję zatwierdzenie E1.", "Sprawdzam folder iteracji.", force=True)
        target_iter_dir = Path(target_iter_dir)
        target_iter_dir.mkdir(parents=True, exist_ok=True)

        selected_files = list(selected_source_files or [])
        if not selected_files:
            selected_files = [
                image_path for image_path in target_iter_dir.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
            ]
        _progress(
            12,
            "Przygotowuję zatwierdzenie E1.",
            f"Do manifestu trafi {len(selected_files)} obrazów.",
            force=True,
        )

        summary_payload = dict(proposal_summary or {})
        package_count = int(len(selected_files) or 0)
        new_to_project_count = int(
            summary_payload.get(
                "new_to_project_count",
                package_count,
            ) or 0
        )

        approved_stats = self._get_plate_approved_set_stats() or {}
        approved_images_before = int(summary_payload.get("approved_images_before_iteration", approved_stats.get("images", 0)) or 0)
        approved_plates_before = int(summary_payload.get("approved_plates_before_iteration", approved_stats.get("plates", 0)) or 0)
        project_pool_total_before = int(
            summary_payload.get(
                "project_pool_total_before_iteration",
                approved_images_before,
            ) or 0
        )
        project_pool_total_after = int(
            summary_payload.get(
                "project_pool_total_after_iteration",
                max(project_pool_total_before, approved_images_before) + new_to_project_count,
            ) or 0
        )

        if package_count > 0 and "source_total" not in summary_payload:
            summary_payload["source_total"] = package_count
        summary_payload["current_iteration_package_count"] = package_count
        summary_payload["new_to_project_count"] = new_to_project_count
        summary_payload["project_pool_total_before_iteration"] = project_pool_total_before
        summary_payload["project_pool_total_after_iteration"] = project_pool_total_after
        summary_payload["approved_images_before_iteration"] = approved_images_before
        summary_payload["approved_plates_before_iteration"] = approved_plates_before

        source_root = Path(source_dir) if source_dir else target_iter_dir
        _progress(
            20,
            "Zapisuję manifest obrazów E1.",
            "Tworzę listę obrazów, źródeł i histogram znaków.",
            force=True,
        )
        manifest_started = perf_counter()
        try:
            CAMPAIGN.record_iteration_ingest(
                source_dir=source_root,
                selected_source_files=selected_files,
                selection_mode=selection_mode,
                proposal_summary=summary_payload,
                selected_source_metadata=selected_source_metadata,
                progress_callback=lambda value, message="", **kwargs: _progress(
                    20.0 + (float(value or 0.0) * 0.60),
                    message,
                    detail=str(kwargs.get("detail", "") or ""),
                    force=bool(kwargs.get("force", False)),
                ),
            )
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ zapisaÄ‡ manifestu E1 dla {target_iter_dir}: {e}")
        finally:
            manifest_ms = max(0.0, (perf_counter() - manifest_started) * 1000.0)

        _progress(84, "Zatwierdzam E1.", "Zmieniam status wejścia i odblokowuję kolejny etap.", force=True)
        approve_started = perf_counter()
        CAMPAIGN.approve_step1()
        if CAMPAIGN.get_current_step() < 2:
            CAMPAIGN.set_current_step(2)
        approve_ms = max(0.0, (perf_counter() - approve_started) * 1000.0)
        self.step1_panel_expanded = False
        if self._continue_to_step3_after_step1_if_char_ready():
            self.current_ingest_plan = {}
            _progress(100, "E1 zatwierdzone.", "Gotowe.", force=True)
            logger.info(
                "[E1 PERF] approve_current_iteration_package total=%sms files=%s manifest=%sms approve=%sms refresh=%sms mode=%s",
                int(max(0.0, (perf_counter() - started_at) * 1000.0)),
                len(selected_files),
                int(manifest_ms),
                int(approve_ms),
                int(refresh_ms),
                selection_mode,
            )
            return len(selected_files)
        self.current_ingest_plan = {}
        _progress(94, "Odświeżam graf kampanii.", "Aktualizuję statusy bramek i etapów.", force=True)
        refresh_started = perf_counter()
        self._refresh_dashboard()
        refresh_ms = max(0.0, (perf_counter() - refresh_started) * 1000.0)
        _progress(100, "E1 zatwierdzone.", "Gotowe.", force=True)
        logger.info(
            "[E1 PERF] approve_current_iteration_package total=%sms files=%s manifest=%sms approve=%sms refresh=%sms mode=%s",
            int(max(0.0, (perf_counter() - started_at) * 1000.0)),
            len(selected_files),
            int(manifest_ms),
            int(approve_ms),
            int(refresh_ms),
            selection_mode,
        )
        return len(selected_files)

    def _apply_current_ingest_plan(self, *args, **kwargs):
        return campaign_step1_ingest._apply_current_ingest_plan(self, *args, **kwargs)
    def apply_theme(self):
        palette = getattr(self.app, "palette", {})
        bg = palette.get("bg", "#1e1e1e")
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("surface_info", palette.get("panel_alt", panel))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")

        try:
            self.app.style_panel_surface(self.frame, background=panel)
        except Exception:
            pass

        try:
            if getattr(self, "header_frame", None) is not None:
                self.header_frame.config(bg=panel)
            if getattr(self, "proj_frame", None) is not None:
                self.proj_frame.config(bg=panel)
        except Exception:
            pass

        try:
            self.lbl_title.config(bg=panel, fg=fg)
        except Exception:
            pass

        try:
            self._configure_campaign_banner(bg=panel)
        except Exception:
            pass

        try:
            if self.wizard_header_title_lbl is not None:
                self.wizard_header_title_lbl.config(bg=panel, fg=fg)
        except Exception:
            pass

        try:
            if self.wizard_header_summary_lbl is not None:
                self.wizard_header_summary_lbl.config(bg=panel, fg=muted)
        except Exception:
            pass

        try:
            self.left_panel_hint_lbl.config(bg=panel, fg=muted)
        except Exception:
            pass

        try:
            if getattr(self, "left_panel_canvas", None) is not None:
                self.left_panel_canvas.config(bg=panel)
        except Exception:
            pass

        try:
            if getattr(self, "right_panel_canvas", None) is not None:
                self.right_panel_canvas.config(bg=panel)
        except Exception:
            pass

        try:
            if getattr(self, "right_sidebar_tab_host", None) is not None:
                self.right_sidebar_tab_host.set_palette(palette)
            if getattr(self, "project_sidebar_title_lbl", None) is not None:
                self.project_sidebar_title_lbl.configure(bg=panel, fg=fg)
        except Exception:
            pass

        try:
            if self.project_list_status_lbl is not None:
                self.project_list_status_lbl.config(bg=panel)
            if self.project_status_top_row is not None:
                self.project_status_top_row.config(bg=panel)
            for lbl in getattr(self, "project_list_status_labels", []):
                lbl.config(bg=panel, fg=muted)
        except Exception:
            pass

        try:
            if self.project_list_host is not None:
                green = self._get_campaign_green_accent()
                self.project_list_host.config(
                    bg=palette.get("field", "#1a1a1a"),
                    highlightbackground=green,
                    highlightcolor=green
                )
        except Exception:
            pass

        try:
            if self.project_listbox is not None:
                green = self._get_campaign_green_accent()
                self.project_listbox.config(
                    bg=palette.get("field", "#1a1a1a"),
                    fg=fg,
                    selectbackground=self._get_project_list_selection_bg(),
                    selectforeground=self._get_project_list_selection_fg(),
                    highlightbackground=green,
                    highlightcolor=green,
                )
        except Exception:
            pass

        try:
            green = self._get_campaign_green_accent()
            if getattr(self, "project_list_scrollbar", None) is not None:
                self.project_list_scrollbar.configure_style(
                    track_color=palette.get("field", "#1a1a1a"),
                    thumb_color=green,
                    thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
                )
            if getattr(self, "left_panel_scrollbar", None) is not None:
                self.left_panel_scrollbar.configure_style(
                    track_color=palette.get("panel", "#252526"),
                    thumb_color=green,
                    thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
                )
            if getattr(self, "right_panel_scrollbar", None) is not None:
                self.right_panel_scrollbar.configure_style(
                    track_color=palette.get("panel", "#252526"),
                    thumb_color=green,
                    thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
                )
        except Exception:
            pass

        try:
            if self.project_browser_footer is not None:
                self.project_browser_footer.config(bg=panel)
            if self.project_add_button_canvas is not None:
                self.project_add_button_canvas.config(bg=panel)
            if self.wizard_exit_button_canvas is not None:
                self.wizard_exit_button_canvas.config(bg=panel)
        except Exception:
            pass

        for widget_name in (
            "ingest_header_lbl",
            "ingest_intro_lbl",
            "lbl_ingest_master_title",
            "lbl_ingest_master_value",
            "lbl_ingest_batch_title",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.config(bg=panel)
            except Exception:
                pass
        self._theme_step1_ingest_panel()
        self._draw_wizard_stage_metro()
        self._draw_icon_button("project_add")
        self._draw_icon_button("exit_project")

        try:
            if self.ingest_plan_host is not None:
                self.ingest_plan_host.config(
                    bg=palette.get("field", "#1a1a1a"),
                    highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                    highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c"))
                )
        except Exception:
            pass

        try:
            if self.ingest_plan_listbox is not None:
                select_bg, select_fg = self.app.get_list_selection_colors()
                self.ingest_plan_listbox.config(
                    bg=palette.get("field", "#1a1a1a"),
                    fg=fg,
                    selectbackground=select_bg,
                    selectforeground=select_fg,
                    highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                    highlightcolor=palette.get("accent", "#2980b9"),
                )
        except Exception:
            pass

        for lbl in getattr(self, "_model_status_title_labels", []):
            try:
                lbl.config(bg=panel, fg=fg)
            except Exception:
                pass

        for model_type in ("vehicle", "plate", "char"):
            lbl_val = getattr(self, f"lbl_model_{model_type}", None)
            if lbl_val is not None:
                try:
                    lbl_val.config(bg=panel)
                except Exception:
                    pass
            lbl_meta = getattr(self, f"lbl_model_{model_type}_meta", None)
            if lbl_meta is not None:
                try:
                    lbl_meta.config(bg=panel, fg=palette.get("muted", "#b0b0b0"))
                except Exception:
                    pass

        try:
            # Repaint the existing graph without re-entering campaign navigation,
            # changing tab access, or dispatching a pending return to gate work.
            graph = getattr(self, "wizard_transition_graph_shell", None)
            if graph is not None and graph.winfo_exists() and graph.winfo_manager():
                self._refresh_wizard_transition_graph(allow_pending_actions=False)
        except Exception:
            logger.exception("Nie udało się odświeżyć motywu grafu kampanii")

    def _ask_project_from_list(self, title="Wybierz projekt", action_label="OK"):
        """WyĹ›wietla modalny wybĂłr projektu i zwraca nazwÄ™ albo None."""
        projects = CAMPAIGN.get_all_projects()
        if not projects:
            self.app.themed_info("Brak projektĂłw", "Nie ma ĹĽadnych zapisanych projektĂłw.", parent=self.frame)
            return None

        dialog = tk.Toplevel(self.frame)
        self.app.style_dialog_window(dialog, title=title, geometry="460x300", parent=self.frame)
        palette = self.app.palette

        shell = tk.Frame(
            dialog,
            bg=palette["bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.get("panel_border", palette["border"]),
            highlightcolor=palette.get("panel_border", palette["border"])
        )
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        body = tk.Frame(shell, bg=palette["panel"])
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            body,
            text=title,
            bg=palette["panel"],
            fg=palette["fg"],
            font=("Segoe UI", 11, "bold")
        ).pack(anchor=tk.W, padx=16, pady=(16, 8))

        tk.Label(
            body,
            text="Wybierz projekt z listy:",
            bg=palette["panel"],
            fg=palette["fg"],
            font=("Segoe UI", 10)
        ).pack(anchor=tk.W, padx=16, pady=(0, 6))

        list_frame = tk.Frame(body, bg=palette["panel"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))

        scroll = WebSlimScrollbar(list_frame, orient=tk.VERTICAL)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        select_bg, select_fg = self.app.get_list_selection_colors()

        project_list = tk.Listbox(
            list_frame,
            exportselection=False,
            font=("Segoe UI", 10),
            bg=palette["field"],
            fg=palette["fg"],
            selectbackground=select_bg,
            selectforeground=select_fg,
            activestyle="none",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.get("panel_border", palette["border"]),
            highlightcolor=palette.get("panel_border", palette["border"]),
            yscrollcommand=scroll.set,
        )
        project_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=project_list.yview)

        for project in projects:
            project_list.insert(tk.END, project)
        project_list.selection_set(0)
        project_list.activate(0)
        project_list.focus_set()

        result = {"value": None}

        btn_row = tk.Frame(body, bg=palette["panel"])
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 16))

        def accept():
            selection = project_list.curselection()
            if selection:
                result["value"] = project_list.get(selection[0]).strip()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(btn_row, text=action_label, command=accept, style="Accent.TButton").pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Anuluj", command=cancel).pack(side=tk.RIGHT, padx=(0, 8))

        project_list.bind("<Double-Button-1>", lambda _e: accept())
        fit_dialog = getattr(self.app, "_fit_dialog_to_content", None)
        if callable(fit_dialog):
            fit_dialog(dialog, parent=self.frame, min_width=460, min_height=300)
        dialog.bind("<Return>", lambda _e: accept())
        dialog.bind("<Escape>", lambda _e: cancel())
        dialog.wait_window()
        return result["value"]

    @staticmethod
    def _training_stage_label_for_target(target: str | None) -> str:
        value = str(target or "").strip().lower()
        if value == "plate":
            return TRAINING_STAGE_PLATES
        if value == "char":
            return TRAINING_STAGE_CHARS
        return f"{TRAINING_STAGE_PLATES}/{TRAINING_STAGE_CHARS}"

    def _ask_iteration_advance_mode(self, *, completed_with_training: bool = True):
        dialog = tk.Toplevel(self.frame)
        try:
            current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        except Exception:
            current_target = ""
        training_stage_label = self._training_stage_label_for_target(current_target)
        self.app.style_dialog_window(dialog, title=f"Co dalej po {training_stage_label}", geometry="620x300", parent=self.frame)
        palette = self.app.palette
        try:
            parent_window = getattr(self.app, "root", None) or self.frame.winfo_toplevel()
        except Exception:
            parent_window = None
        try:
            if parent_window is not None:
                dialog.transient(parent_window)
        except Exception:
            pass

        shell = tk.Frame(
            dialog,
            bg=palette["bg"],
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.get("panel_border", palette["border"]),
            highlightcolor=palette.get("panel_border", palette["border"])
        )
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        body = tk.Frame(shell, bg=palette["panel"])
        body.pack(fill=tk.BOTH, expand=True)

        header_text = (
            f"{training_stage_label} jest gotowe do zamknięcia. Możesz przejść do E1 kolejnej iteracji."
            if completed_with_training
            else f"{training_stage_label} może zostać zamknięte bez treningu. Możesz przejść do E1."
        )
        intro_text = (
            "Jeśli wybierzesz przejście dalej, bieżąca iteracja zostanie domknięta, a nowa rozpocznie się w E1. "
            "Tam wybierzesz katalog zdjęć wejściowych, tor pracy oraz zatwierdzisz zdjęcia startowe. "
            "Aktywne modele projektu pozostają dostępne."
            if completed_with_training
            else "Jeśli wybierzesz przejście dalej, iteracja zostanie formalnie zamknięta bez treningu. "
            "Kolejny cykl rozpocznie się w E1, gdzie wybierzesz katalog zdjęć wejściowych, tor pracy "
            "oraz zatwierdzisz zdjęcia startowe. Modele projektu pozostają dostępne."
        )

        tk.Label(
            body,
            text=header_text,
            bg=palette["panel"],
            fg=palette["fg"],
            font=("Segoe UI", 11, "bold")
        ).pack(anchor=tk.W, padx=16, pady=(16, 6))

        tk.Label(
            body,
            text=intro_text,
            bg=palette["panel"],
            fg=palette.get("muted", palette["fg"]),
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=600
        ).pack(anchor=tk.W, padx=16, pady=(0, 12))

        result = {"value": None}

        def choose(mode: str):
            result["value"] = mode
            dialog.destroy()

        def choose_e1_transition():
            mode = "new_input"
            try:
                stage_state = CAMPAIGN.get_manual_plate_stage_images_state()
                if int(stage_state.get("image_count", 0) or 0) > 0:
                    mode = "reuse_input"
            except Exception:
                mode = "new_input"
            choose(mode)

        def cancel():
            dialog.destroy()

        info_card = tk.Frame(
            body,
            bg=palette["field"],
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.get("panel_border", palette["border"]),
            highlightcolor=palette.get("panel_border", palette["border"]),
        )
        info_card.pack(fill=tk.X, padx=16, pady=(0, 14))
        tk.Label(
            info_card,
            text="Co stanie się po przejściu?",
            bg=palette["field"],
            fg=palette["fg"],
            font=("Segoe UI", 10, "bold"),
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(anchor=tk.W, fill=tk.X, padx=14, pady=(12, 4))
        tk.Label(
            info_card,
            text=(
                "Program utworzy kolejną iterację, wyczyści roboczy stan E2/E3/E4T/E4Z i otworzy E1. "
                "W E1 ustawisz wejścia, wybierzesz tor tablic albo znaków i zatwierdzisz etap dopiero wtedy, "
                "gdy zdjęcia startowe będą gotowe."
            ),
            bg=palette["field"],
            fg=palette.get("muted", palette["fg"]),
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            anchor=tk.W,
            wraplength=540,
        ).pack(anchor=tk.W, fill=tk.X, padx=14, pady=(0, 12))

        btn_row = tk.Frame(body, bg=palette["panel"])
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 16))
        ttk.Button(
            btn_row,
            text="Przejdź do E1",
            command=choose_e1_transition,
            style="Accent.TButton",
        ).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text=f"Zostań w {training_stage_label}", command=cancel).pack(side=tk.RIGHT, padx=(0, 8))

        fit_dialog = getattr(self.app, "_fit_dialog_to_content", None)
        if callable(fit_dialog):
            fit_dialog(dialog, parent=self.frame, min_width=620, min_height=300)
        try:
            dialog.lift()
            dialog.grab_set()
        except Exception:
            pass
        try:
            dialog.focus_force()
        except Exception:
            try:
                dialog.focus_set()
            except Exception:
                pass
        dialog.bind("<Escape>", lambda _e: cancel())
        dialog.wait_window()
        return result["value"]

    def _set_model(self, model_type, initial_dir: Path | None = None):
        # OtwĂłrz wybĂłr modelu od katalogu modeli aktywnego projektu.
        if initial_dir is None:
            initial_dir = CAMPAIGN.get_dir("models")
            if initial_dir is None:
                initial_dir = Path(CONFIG.DIR_6_MODELS)
            else:
                initial_dir = Path(initial_dir)
        else:
            initial_dir = Path(initial_dir)

        p = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[("YOLO Model", "*.pt")]
        )
        if p:
            model_path = Path(p)
            is_valid, error_message = self._validate_project_model_selection(model_type, model_path)
            if not is_valid:
                self.app.themed_error(
                    "Nieprawidlowy model",
                    error_message,
                    parent=self.frame,
                )
                return
            CAMPAIGN.set_global_model(model_type, str(model_path))
            row_key = "plate_model" if str(model_type or "").strip().lower() == "plate" else "char_model"
            self._set_project_start_asset_scope(
                row_key,
                self._infer_project_start_asset_scope_from_path(row_key, model_path),
            )
            try:
                self._sync_iteration_artifact_registry_from_project_start()
            except Exception:
                pass
            self._refresh_dashboard()

    # ======================================================
    # WIZARD UI
    # ======================================================

    def _rebuild_wizard_stage_ui(self, *args, **kwargs):
        return campaign_stage_ui._rebuild_wizard_stage_ui(self, *args, **kwargs)


    def _refresh_wizard_transition_graph(self, *args, **kwargs):
        return campaign_stage_ui._refresh_wizard_transition_graph(self, *args, **kwargs)


    def _show_wizard_transition_graph(self, *args, **kwargs):
        return campaign_stage_ui._show_wizard_transition_graph(self, *args, **kwargs)


    def _set_pack_visibility(self, *args, **kwargs):
        return campaign_stage_ui._set_pack_visibility(self, *args, **kwargs)


    def _set_grid_visibility(self, *args, **kwargs):
        return campaign_stage_ui._set_grid_visibility(self, *args, **kwargs)


    def _cancel_wizard_stage_curtain_animation(self, *args, **kwargs):
        return campaign_stage_ui._cancel_wizard_stage_curtain_animation(self, *args, **kwargs)


    def _set_wizard_stage_curtain_height(self, *args, **kwargs):
        return campaign_stage_ui._set_wizard_stage_curtain_height(self, *args, **kwargs)


    def _animate_wizard_stage_curtain(self, *args, **kwargs):
        return campaign_stage_ui._animate_wizard_stage_curtain(self, *args, **kwargs)


    def _toggle_wizard_stage_curtain(self, *args, **kwargs):
        return campaign_stage_ui._toggle_wizard_stage_curtain(self, *args, **kwargs)


    def _collapse_wizard_stage_curtain(self, *args, **kwargs):
        return campaign_stage_ui._collapse_wizard_stage_curtain(self, *args, **kwargs)


    def _collapse_all_wizard_stage_curtains(self, *args, **kwargs):
        return campaign_stage_ui._collapse_all_wizard_stage_curtains(self, *args, **kwargs)


    def _refresh_wizard_stage_curtain_style(self, *args, **kwargs):
        return campaign_stage_ui._refresh_wizard_stage_curtain_style(self, *args, **kwargs)


    @staticmethod
    def _format_step1_selection_mode_label(*args, **kwargs):
        return campaign_stage_ui._format_step1_selection_mode_label(*args, **kwargs)

    def _get_step1_manifest_context(self, *args, **kwargs):
        return campaign_stage_ui._get_step1_manifest_context(self, *args, **kwargs)

    def _get_step1_manifest_context_lightweight(self, *args, **kwargs):
        return campaign_stage_ui._get_step1_manifest_context_lightweight(self, *args, **kwargs)

    def _build_step1_summary_payload(self, *args, **kwargs):
        return campaign_stage_ui._build_step1_summary_payload(self, *args, **kwargs)

    def _render_step1_stage_curtain(self, *args, **kwargs):
        return campaign_stage_ui._render_step1_stage_curtain(self, *args, **kwargs)


    def _refresh_wizard_stage_metro(self, *args, **kwargs):
        return campaign_stage_ui._refresh_wizard_stage_metro(self, *args, **kwargs)


    def _draw_wizard_stage_metro(self, *args, **kwargs):
        return campaign_stage_ui._draw_wizard_stage_metro(self, *args, **kwargs)


    def _build_wizard_stage_card(self, *args, **kwargs):
        return campaign_stage_ui._build_wizard_stage_card(self, *args, **kwargs)


    def _is_wizard_stage_emphasized(self, *args, **kwargs):
        return campaign_stage_ui._is_wizard_stage_emphasized(self, *args, **kwargs)


    def _get_wizard_stage_state_style(self, *args, **kwargs):
        return campaign_stage_ui._get_wizard_stage_state_style(self, *args, **kwargs)


    def _configure_wizard_stage_button(self, *args, **kwargs):
        return campaign_stage_ui._configure_wizard_stage_button(self, *args, **kwargs)


    def _configure_wizard_stage_badge_cta(self, *args, **kwargs):
        return campaign_stage_ui._configure_wizard_stage_badge_cta(self, *args, **kwargs)


    def _get_step2_render_disk_approval_fallback(self, *args, **kwargs):
        return campaign_stage_ui._get_step2_render_disk_approval_fallback(self, *args, **kwargs)


    def _force_current_step2_badge_from_disk(self, *args, **kwargs):
        return campaign_stage_ui._force_current_step2_badge_from_disk(self, *args, **kwargs)


    def _get_persistent_wizard_stage_badge_override(self, *args, **kwargs):
        return campaign_stage_ui._get_persistent_wizard_stage_badge_override(self, *args, **kwargs)


    def _stabilize_wizard_stage_badges(self, *args, **kwargs):
        return campaign_stage_ui._stabilize_wizard_stage_badges(self, *args, **kwargs)


    def _schedule_wizard_stage_badge_stabilization(self, *args, **kwargs):
        return campaign_stage_ui._schedule_wizard_stage_badge_stabilization(self, *args, **kwargs)


    def _apply_wizard_stage_status(self, *args, **kwargs):
        return campaign_stage_ui._apply_wizard_stage_status(self, *args, **kwargs)


    def _configure_campaign_banner(self, *, bg: str):
        palette = getattr(self.app, "palette", {})
        panel_bg = str(palette.get("panel", "#252526"))

        try:
            if self.campaign_banner_shell is not None:
                self.campaign_banner_shell.config(bg=panel_bg)
        except Exception:
            pass

        try:
            if self.banner_progress_row is not None:
                self.banner_progress_row.config(bg=panel_bg)
        except Exception:
            pass

        try:
            if self.wizard_header_metro_canvas is not None:
                self.wizard_header_metro_canvas.config(bg=panel_bg)
            if self.wizard_exit_button_canvas is not None:
                self.wizard_exit_button_canvas.config(bg=panel_bg)
        except Exception:
            pass

        self.frame.after_idle(self._draw_wizard_stage_metro)
        self.frame.after_idle(lambda: self._draw_icon_button("exit_project"))

    def _show_project_loading_overlay(self, *args, **kwargs):
        return campaign_dashboard_ui._show_project_loading_overlay(self, *args, **kwargs)


    def _hide_project_loading_overlay(self, *args, **kwargs):
        return campaign_dashboard_ui._hide_project_loading_overlay(self, *args, **kwargs)


    def _refresh_wizard_empty_state(self, *args, **kwargs):
        return campaign_dashboard_ui._refresh_wizard_empty_state(self, *args, **kwargs)


    def _build_active_project_dashboard_state(self, *args, **kwargs):
        return campaign_dashboard_ui._build_active_project_dashboard_state(self, *args, **kwargs)


    def _refresh_active_project_wizard_only(self, *args, **kwargs):
        return campaign_dashboard_ui._refresh_active_project_wizard_only(self, *args, **kwargs)


    def _cancel_deferred_project_open_tasks(self, *args, **kwargs):
        return campaign_dashboard_ui._cancel_deferred_project_open_tasks(self, *args, **kwargs)


    def _refresh_wizard_active_dashboard(self, *args, **kwargs):
        return campaign_dashboard_ui._refresh_wizard_active_dashboard(self, *args, **kwargs)


    @staticmethod
    def _format_project_iteration_title(project_name: str, iteration_num: int | None = None) -> str:
        name = str(project_name or "").strip()
        if not name:
            return "Brak aktywnego projektu"
        try:
            iter_value = max(1, int(iteration_num or 1))
        except Exception:
            iter_value = 1
        return f"Projekt: {name} | Iteracja {iter_value}"

    def _align_step3_details_with_cta_labels(
        self,
        *,
        state: str,
        details: str,
        primary_label: str = "",
        secondary_label: str = "",
        badge_label: str = "",
    ) -> str:
        state_key = str(state or "").strip().lower()
        primary = str(primary_label or "").strip()
        secondary = str(secondary_label or "").strip()
        badge = str(badge_label or "").strip()

        if state_key == "ready":
            actions = []
            if badge:
                actions.append(f"zamknÄ…Ä‡ E3 przyciskiem â€ž{badge}â€ť")
            if primary:
                actions.append(f"wrĂłciÄ‡ do Z2 przez â€ž{primary}â€ť")
            if secondary:
                actions.append(f"pracowaÄ‡ dalej w Z3 przez â€ž{secondary}â€ť")
            if actions:
                if len(actions) == 1:
                    return f"MoĹĽesz {actions[0]}."
                return f"MoĹĽesz {', '.join(actions[:-1])} albo {actions[-1]}."

        return str(details or "").strip()

    def _get_wizard_stage_statuses(
        self,
        *,
        active_project: str,
        current_step: int,
        iteration_target: str,
        project_status: str,
        project_paused_at: str,
        project_completed_at: str,
        step1_status: str,
        step2_status: str,
        step3_status: str,
    ) -> list[WizardStageStatus]:
        statuses: list[WizardStageStatus] = []
        iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        project_status_norm = str(project_status or "active").strip().lower()
        project_completed = project_status_norm == "completed"
        project_paused = project_status_norm == "paused"
        project_suspended = bool(project_completed or project_paused)
        lightweight_open = bool(getattr(self, "_project_open_lightweight_refresh", False))
        if lightweight_open:
            return campaign_dashboard_ui._get_lightweight_wizard_stage_statuses(
                self,
                active_project=active_project,
                current_step=current_step,
                iteration_target=iteration_target,
                project_status=project_status_norm,
                project_paused_at=project_paused_at,
                project_completed_at=project_completed_at,
                step1_status=step1_status,
                step2_status=step2_status,
                step3_status=step3_status,
            )
        target_label = self._iteration_target_label(iteration_target)
        iter_image_count = self._get_iteration_image_count()
        step1_approved = str(step1_status or "").strip().lower() == "approved"
        if not step1_approved and int(current_step or 1) != 1:
            current_step = 1
            try:
                CAMPAIGN.set_current_step(1)
            except Exception:
                pass
        master_pool = CAMPAIGN.get_master_pool_dir()
        master_pool_ready = bool(master_pool and master_pool.exists() and master_pool.is_dir())
        draft_plan = self._get_active_step1_draft_plan()
        plan_count = int(draft_plan.get("selected_total", 0) or 0)
        plate_ready_source = self._get_plate_route_ready_source() if iteration_target == "plate" else {}
        char_ready_source = self._get_char_route_ready_source() if iteration_target == "char" else {}
        char_route_source_state = self._get_char_route_source_state() if iteration_target == "char" else {}
        plate_approved_stats = self._get_plate_approved_set_stats()
        try:
            project_approved_images_current = int(plate_approved_stats.get("images", 0) or 0)
        except Exception:
            project_approved_images_current = 0
        try:
            project_approved_plates_current = int(plate_approved_stats.get("plates", 0) or 0)
        except Exception:
            project_approved_plates_current = 0
        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
        plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())
        training_tab = self.app.tabs.get("training") if getattr(self.app, "tabs", None) else None
        plate_step4_gate = {
            "ok": True,
            "reason": "",
            "message": "",
            "annotated_images": 0,
            "required_images": 0,
            "required_plates": int(getattr(self, "STEP2_PLATE_MIN_PLATES", 10) or 10),
            "source_run": "",
        }
        char_step4_gate = {
            "ok": True,
            "reason": "",
            "message": "",
            "ready_dataset": "",
            "dataset_hint": "",
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "validation_message": "",
        }
        if (
            not lightweight_open
            and iteration_target == "plate"
            and (current_step >= 4 or str(step2_status or "").strip().lower() == "approved")
        ):
            try:
                if training_tab is not None and hasattr(training_tab, "get_campaign_step4_readiness"):
                    plate_step4_gate = training_tab.get_campaign_step4_readiness(iteration_target="plate")
            except Exception as e:
                logger.debug(f"Nie udało się sprawdzić gotowości E4T dla toru tablic: {e}")
            try:
                gate_images = int(plate_step4_gate.get("project_approved_images", 0) or 0)
            except Exception:
                gate_images = 0
            try:
                gate_plates = int(plate_step4_gate.get("project_approved_plates", 0) or 0)
            except Exception:
                gate_plates = 0
            if project_approved_images_current > gate_images:
                try:
                    gate_annotated_images = int(plate_step4_gate.get("annotated_images", 0) or 0)
                except Exception:
                    gate_annotated_images = 0
                plate_step4_gate["project_approved_images"] = project_approved_images_current
                plate_step4_gate["annotated_images"] = max(
                    gate_annotated_images,
                    project_approved_images_current,
                )
            if project_approved_plates_current > gate_plates:
                plate_step4_gate["project_approved_plates"] = project_approved_plates_current
            if project_approved_images_current > 0 and not str(plate_step4_gate.get("source_run") or "").strip():
                plate_step4_gate["source_run"] = "ApprovedSet projektu"

        helper_char_step4_gate = (
            self._detect_campaign_char_ready_dataset_state()
            if (
                not lightweight_open
                and iteration_target == "char"
                and (current_step >= 3 or str(step3_status or "").strip().lower() in {"approved", "needs_rework", "ready"})
            )
            else {}
        )

        if (
            not lightweight_open
            and iteration_target == "char"
            and (current_step >= 3 or str(step3_status or "").strip().lower() in {"approved", "needs_rework", "ready"})
        ):
            try:
                if training_tab is not None and hasattr(training_tab, "get_campaign_step4_readiness"):
                    char_step4_gate = training_tab.get_campaign_step4_readiness(iteration_target="char")
            except Exception as e:
                logger.debug(f"Nie udało się sprawdzić gotowości E4Z dla toru znaków: {e}")
        if helper_char_step4_gate:
            if not bool(helper_char_step4_gate.get("ok")):
                char_step4_gate = dict(helper_char_step4_gate)
            elif not bool(char_step4_gate.get("ok")) or not str(char_step4_gate.get("ready_dataset") or "").strip():
                char_step4_gate = dict(helper_char_step4_gate)
            if (
                bool(helper_char_step4_gate.get("ok"))
                and int(helper_char_step4_gate.get("perfect_count", 0) or 0) > 0
                and iteration_target == "char"
                and current_step == 3
                and str(step3_status or "").strip().lower() in {"pending", "needs_rework"}
            ):
                step3_status = "ready"

        char_repair_guidance = (
            self._get_char_repair_guidance(char_route_source_state)
            if iteration_target == "char"
            else {}
        )

        plate_step4_blocked = bool(
            iteration_target == "plate"
            and (current_step >= 4 or str(step2_status or "").strip().lower() == "approved")
            and not bool(plate_step4_gate.get("ok", True))
        )
        plate_step4_reason = str(plate_step4_gate.get("reason") or "").strip().lower()
        char_step4_blocked = bool(
            iteration_target == "char"
            and (current_step >= 4 or str(step3_status or "").strip().lower() == "approved")
            and not bool(char_step4_gate.get("ok", True))
        )
        step4_gate_blocked = bool(plate_step4_blocked or char_step4_blocked)
        step4_finish_state = {}
        try:
            if training_tab is not None and hasattr(training_tab, "get_campaign_step4_finish_state"):
                step4_finish_state = training_tab.get_campaign_step4_finish_state(iteration_target=iteration_target) or {}
            else:
                step4_finish_state = CAMPAIGN.get_step4_finish_state() or {}
        except Exception:
            try:
                step4_finish_state = CAMPAIGN.get_step4_finish_state() or {}
            except Exception:
                step4_finish_state = {}
        step4_finish_ready = bool(step4_finish_state.get("ready", False))
        step4_finish_target = self._normalize_iteration_target(step4_finish_state.get("target", ""))
        step4_finish_iteration = int(step4_finish_state.get("iteration", 0) or 0)
        if not step4_finish_target:
            step4_finish_target = iteration_target
        step4_can_advance = bool(
            current_step == 4
            and not step4_gate_blocked
            and step4_finish_ready
            and step4_finish_target == iteration_target
            and step4_finish_iteration == int(iter_num or 0)
        )
        training_stage_label = self._training_stage_label_for_target(iteration_target)

        manifest = self._load_ingest_manifest_cached()
        manifest_mode = str((manifest or {}).get("selection_mode", "") or "").strip().lower() if isinstance(manifest, dict) else ""
        is_iteration_reuse = bool(iter_num > 1 and manifest_mode in {"iteration_reuse", "pool_reuse", "stage_reuse"})
        step1_context = self._get_step1_manifest_context(manifest if isinstance(manifest, dict) else None)
        project_pool_total = int(step1_context.get("project_pool_total_after", step1_context.get("project_pool_total", 0)) or 0)
        approved_images = max(
            int(step1_context.get("approved_images", 0) or 0),
            int(project_approved_images_current or 0),
        )
        approved_plates = max(
            int(step1_context.get("approved_plates", 0) or 0),
            int(project_approved_plates_current or 0),
        )
        source_total = int(step1_context.get("source_total", 0) or 0)
        new_to_project_count = int(step1_context.get("new_to_project_count", 0) or 0)
        project_overlap_count = int(step1_context.get("project_overlap_filenames", 0) or 0)
        approved_overlap_count = int(step1_context.get("skipped_duplicate_approved", 0) or 0)
        current_iteration_package = int(
            step1_context.get("current_iteration_package_count", 0)
            or iter_image_count
            or 0
        )
        try:
            current_master_pool_dir = CAMPAIGN.get_master_pool_dir()
            master_pool_ready = bool(current_master_pool_dir and current_master_pool_dir.exists() and current_master_pool_dir.is_dir())
        except Exception:
            current_master_pool_dir = None
            master_pool_ready = False
        current_source_dir_count = int(source_total or step1_context.get("source_total", 0) or current_iteration_package or 0)
        source_diverged = bool(
            current_source_dir_count > 0
            and current_iteration_package > 0
            and current_source_dir_count != current_iteration_package
        )
        source_label = str(step1_context.get("source_label", "") or "").strip()
        target_selected = bool(iteration_target in {"plate", "char"})

        project_scope_parts: list[str] = []
        if approved_images > 0 or approved_plates > 0:
            project_scope_parts.append(
                f"PudeĹ‚ko tablic zatwierdzonych: {approved_images} zdjÄ™Ä‡ / {approved_plates} tablic."
            )
        if project_pool_total > 0:
            project_scope_parts.append(f"Pula projektu informacyjnie: {project_pool_total} zdjÄ™Ä‡.")
        project_scope_text = " ".join(project_scope_parts).strip()
        step1_has_selected_image_folder = bool(current_source_dir_count > 0 or master_pool_ready or plan_count > 0)
        step1_ready_for_approval = bool(
            not project_suspended
            and current_step == 1
            and target_selected
            and not step1_approved
            and step1_has_selected_image_folder
        )
        step1_body_visible = bool(
            not project_suspended
            and (
                (current_step == 1 and not step1_approved)
                or (current_step >= 2 and not target_selected)
            )
        )

        if step1_approved and not target_selected:
            step1_state = "needs_attention"
            step1_summary = "E1 ma zdjÄ™cia, ale brakuje zapisanego toru iteracji."
            step1_details = "To najpewniej starszy zapis projektu. Wybierz tor w E1, aby E2 mogĹ‚o przejĹ›Ä‡ do pracy na tablicach."
        elif step1_approved and iter_image_count > 0:
            step1_state = "done"
            step1_summary = ""
            step1_details = ""
        elif iter_image_count > 0 and not target_selected:
            step1_state = "needs_attention"
            step1_summary = f"W folderze iteracji sÄ… juĹĽ {iter_image_count} zdjÄ™cia. Wybierz jeszcze tor iteracji."
            step1_details = "Tor wybierasz w panelu E1. Dopiero wtedy moĹĽna zatwierdziÄ‡ E1 i przejĹ›Ä‡ do E2."
        elif iter_image_count > 0:
            step1_state = "needs_attention"
            step1_summary = f"W folderze iteracji sÄ… juĹĽ {iter_image_count} zdjÄ™cia, ale E1 czeka na zatwierdzenie."
            step1_details = "ZejdĹş do panelu E1 i zatwierdĹş zestaw zdjÄ™Ä‡, aby odblokowaÄ‡ E2."
        elif master_pool_ready or plan_count > 0 or current_step == 1:
            step1_state = "in_progress" if current_step == 1 else "ready"
            step1_summary = ""
            step1_details = ""
        else:
            step1_state = "ready"
            step1_summary = ""
            step1_details = ""

        if step1_body_visible and current_step == 1 and not step1_approved:
            step1_intro = self._get_step1_assets_intro_text()
            step1_details = (
                f"{step1_intro}\n{step1_details}"
                if str(step1_details or "").strip()
                else step1_intro
            )

        statuses.append(
            WizardStageStatus(
                key="step1",
                title="E1. WejĹ›cie i tor iteracji",
                state=step1_state,
                summary=step1_summary,
                details=step1_details,
                primary_label="",
                primary_command=None,
                badge_action_label=("ZatwierdĹş etap" if step1_ready_for_approval else ""),
                badge_action_command=(
                    (lambda: campaign_graph_actions.execute_campaign_graph_action(self, "approve_step1"))
                    if step1_ready_for_approval
                    else None
                ),
                body_mode="step1_ingest",
                body_visible=step1_body_visible,
                is_current=bool(not project_suspended and current_step == 1),
            )
        )

        step2_vm = self._get_annotation_step2_view_model()
        annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
        try:
            step2_wizard_status = (
                dict(annotation_tab.get_campaign_step2_wizard_status() or {})
                if annotation_tab is not None and hasattr(annotation_tab, "get_campaign_step2_wizard_status")
                else {}
            )
        except Exception:
            step2_wizard_status = {}
        step2_title = "E2. Tablice"
        if step2_vm is not None:
            primary_cta = getattr(step2_vm, "primary_cta", None)
            secondary_cta = getattr(step2_vm, "secondary_cta", None)
            step2_title = str(getattr(step2_vm, "title", "") or "").strip() or step2_title
            step2_state = str(getattr(step2_vm, "state", "") or "").strip() or "locked"
            step2_summary = str(getattr(step2_vm, "summary", "") or "").strip()
            step2_details = str(getattr(step2_vm, "details", "") or "").strip()
            step2_primary_label = str(getattr(primary_cta, "label", "") or "").strip()
            step2_primary_command = self._resolve_step2_wizard_action_command(
                str(getattr(primary_cta, "command_id", "") or "").strip(),
                context=dict(getattr(primary_cta, "command_context", {}) or {}),
            )
            step2_secondary_label = str(getattr(secondary_cta, "label", "") or "").strip()
            step2_secondary_command = self._resolve_step2_wizard_action_command(
                str(getattr(secondary_cta, "command_id", "") or "").strip(),
                context=dict(getattr(secondary_cta, "command_context", {}) or {}),
            )
            vm_current_step = int(getattr(step2_vm, "current_step", current_step) or current_step)
            step2_body_mode = ""
            step2_body_visible = False
        elif not iteration_target:
            step2_state = "needs_attention" if current_step >= 2 else "locked"
            step2_summary = "Brak wybranego toru iteracji."
            step2_details = (
                "WybĂłr toru naleĹĽy teraz do E1. WrĂłÄ‡ do E1, wybierz tor tablic albo tor znakĂłw, "
                "a dopiero potem zatwierdĹş katalog zdjÄ™Ä‡ wejĹ›ciowych."
            )
            step2_primary_label = ""
            step2_primary_command = None
            step2_secondary_label = ""
            step2_secondary_command = None
            step2_body_mode = ""
            step2_body_visible = False
        else:
            step2_primary_label = self._get_step2_jump_button_text(iteration_target)
            step2_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "open_z2_campaign_context",
            )
            step2_secondary_label = ""
            step2_secondary_command = None
            step2_body_mode = ""
            step2_body_visible = False

            if current_step == 2 and step1_approved:
                step2_state = "in_progress"
                step2_summary = "E2 jest w toku: przygotuj anotacje tablic w Z2."
                step2_details = (
                    "PrzejdĹş do Z2, aby utworzyÄ‡ albo poprawiÄ‡ anotacje tablic "
                    "dla wybranego katalogu zdjÄ™Ä‡ z E1."
                )
            elif current_step > 2 or str(step2_status or "").strip().lower() == "approved":
                step2_state = "done"
                step2_summary = "E2 zostaĹ‚o juĹĽ domkniÄ™te."
                step2_details = "Dalsza praca odbywa siÄ™ w kolejnych etapach wizarda."
                step2_primary_label = ""
                step2_primary_command = None
            else:
                step2_state = "locked"
                step2_summary = "E2 odblokuje siÄ™ po zatwierdzeniu katalogu zdjÄ™Ä‡ wejĹ›ciowych z E1."
                step2_details = "Najpierw domknij E1."

        if current_step == 2 and not iteration_target:
            step2_primary_label = ""
            step2_primary_command = None
            step2_secondary_label = ""
            step2_secondary_command = None
            step2_summary = "Brak wybranego toru iteracji."
            step2_details = (
                "Ten stan oznacza stary albo niepeĹ‚ny zapis projektu. WrĂłÄ‡ do E1, wybierz tor iteracji "
                "i zatwierdĹş E1 ponownie, ĹĽeby E2 mogĹ‚o pracowaÄ‡ wyĹ‚Ä…cznie jako etap tablic."
            )

        step2_already_approved = str(step2_status or "").strip().lower() == "approved"
        step2_ready_for_approval = bool(
            step1_approved
            and step2_wizard_status.get("ready_for_approval")
            and not step2_already_approved
        )
        step2_approval_action = str(step2_wizard_status.get("approval_action", "") or "").strip().lower()
        step2_approval_iteration_target = str(
            step2_wizard_status.get("approval_iteration_target", "") or ""
        ).strip().lower()
        if step2_ready_for_approval and step2_approval_iteration_target not in {"plate", "char"}:
            step2_approval_iteration_target = self._normalize_iteration_target(iteration_target)
        if (
            not step2_already_approved
            and step1_approved
            and int(current_step or 0) <= 2
            and not step2_ready_for_approval
        ):
            fallback_approval_context = {}
            try:
                if annotation_tab is not None and hasattr(annotation_tab, "_get_campaign_step2_approval_context"):
                    fallback_approval_context = dict(annotation_tab._get_campaign_step2_approval_context() or {})
            except Exception:
                fallback_approval_context = {}
            fallback_target = str(
                iteration_target or fallback_approval_context.get("iteration_target") or ""
            ).strip().lower()
            try:
                fallback_step = int(fallback_approval_context.get("current_step") or current_step or 0)
            except Exception:
                fallback_step = int(current_step or 0)
            try:
                campaign_fallback_step = int(current_step or 0)
            except Exception:
                campaign_fallback_step = fallback_step
            if (
                (campaign_fallback_step == 2 or fallback_step == 2)
                and fallback_target in {"plate", "char"}
            ):
                disk_fallback = self._get_step2_render_disk_approval_fallback(fallback_target)
                if bool(disk_fallback.get("ready")):
                    step2_ready_for_approval = True
                    if not step2_approval_action:
                        step2_approval_action = str(disk_fallback.get("action") or "approve_stage").strip()
                    if not step2_approval_iteration_target:
                        step2_approval_iteration_target = fallback_target
        if step2_ready_for_approval:
            approval_iteration_target = step2_approval_iteration_target
            next_stage_label = "E4T" if approval_iteration_target == "plate" else "E3"
            last_target = self._get_last_iteration_target()
            same_route_as_previous = bool(
                approval_iteration_target in {"plate", "char"}
                and last_target == approval_iteration_target
            )
            if approval_iteration_target == "char" and step2_approval_action == "continue_characters":
                step2_state = "ready"
                step2_summary = "ĹąrĂłdĹ‚o tablic dla wybranego katalogu zdjÄ™Ä‡ jest juĹĽ gotowe."
                step2_details = (
                    "Minimalny prĂłg wejĹ›cia do E3 jest juĹĽ speĹ‚niony, wiÄ™c badge po prawej moĹĽe od razu zamknÄ…Ä‡ E2. "
                    "JeĹ›li jednak masz jeszcze chwilÄ™, zwykle wiÄ™cej daje dopisanie kolejnych poprawnych tablic w Z2 niĹĽ samo szybkie przejĹ›cie dalej."
                )
                step2_primary_label = "Dodaj jeszcze tablice w Z2"
                step2_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "return_to_z2_review",
                    payload={"mark_step3_rework": False},
                )
            else:
                step2_state = "ready"
                step2_summary = "Etap 2 moĹĽesz juĹĽ zamknÄ…Ä‡, ale warto jeszcze rozwaĹĽyÄ‡ dopisanie tablic w Z2."
                step2_details = (
                    f"Minimalny prĂłg projektu jest juĹĽ speĹ‚niony, wiÄ™c moĹĽesz od razu zamknÄ…Ä‡ E2 i odblokowaÄ‡ {next_stage_label}. "
                    "JeĹ›li jednak zaleĹĽy Ci na lepszym materiale do kolejnych iteracji modelu, zwykle bardziej opĹ‚aca siÄ™ dopisaÄ‡ jeszcze kilka poprawnych anotacji niĹĽ koĹ„czyÄ‡ etap od razu."
                )
                if same_route_as_previous:
                    step2_details += " Dotyczy to szczegĂłlnie sytuacji, gdy zostajesz w tym samym torze co poprzednio."
                else:
                    step2_details += " To dobry moment, ĹĽeby zdecydowaÄ‡, czy chcesz tylko przejĹ›Ä‡ dalej, czy jeszcze trochÄ™ wzmocniÄ‡ zbiĂłr projektu."
                step2_primary_label = "Dodaj jeszcze tablice w Z2"
                step2_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "open_z2_campaign_context",
                )
            step2_secondary_label = ""
            step2_secondary_command = None
            step2_body_mode = ""
            step2_body_visible = False

        if step2_state == "done":
            step2_primary_label = ""
            step2_primary_command = None
            step2_secondary_label = ""
            step2_secondary_command = None
            step2_body_mode = ""
            step2_body_visible = False

        statuses.append(
            WizardStageStatus(
                key="step2",
                title=step2_title,
                state=step2_state,
                summary=step2_summary,
                details=step2_details,
                primary_label=step2_primary_label,
                primary_command=step2_primary_command,
                secondary_label=step2_secondary_label,
                secondary_command=step2_secondary_command,
                badge_action_label=("ZatwierdĹş etap" if step2_ready_for_approval else ""),
                badge_action_command=(
                    (lambda: campaign_graph_actions.execute_campaign_graph_action(self, "approve_step2"))
                    if step2_ready_for_approval
                    else None
                ),
                body_mode=step2_body_mode,
                body_visible=step2_body_visible,
                is_current=bool(
                    not project_suspended
                    and (
                        current_step == 2
                        or plate_step4_reason in {"missing_plate_annotations", "insufficient_plate_annotations"}
                    )
                ),
            )
        )

        step3_vm = self._get_campaign_step3_view_model(
            current_step=current_step,
            iteration_target=iteration_target,
            project_completed=project_suspended,
            step2_status=step2_status,
            step3_status=step3_status,
            char_ready_source=char_ready_source,
            char_repair_guidance=char_repair_guidance,
            char_step4_gate=char_step4_gate,
        )
        step3_primary_cta = getattr(step3_vm, "primary_cta", None)
        step3_secondary_cta = getattr(step3_vm, "secondary_cta", None)
        step3_state = str(getattr(step3_vm, "state", "") or "").strip() or "locked"
        step3_summary = str(getattr(step3_vm, "summary", "") or "").strip()
        step3_details = str(getattr(step3_vm, "details", "") or "").strip()
        step3_primary_label = str(getattr(step3_primary_cta, "label", "") or "").strip()
        step3_primary_command = self._resolve_step3_wizard_action_command(
            str(getattr(step3_primary_cta, "command_id", "") or "").strip(),
            context=dict(getattr(step3_primary_cta, "command_context", {}) or {}),
        )
        step3_secondary_label = str(getattr(step3_secondary_cta, "label", "") or "").strip()
        step3_secondary_command = self._resolve_step3_wizard_action_command(
            str(getattr(step3_secondary_cta, "command_id", "") or "").strip(),
            context=dict(getattr(step3_secondary_cta, "command_context", {}) or {}),
        )

        if step3_state.lower() == "ready":
            step3_summary = "Etap 3 jest gotowy do zamkniÄ™cia albo dalszej pracy."
            step3_details = (
                "MoĹĽesz zamknÄ…Ä‡ E3, oznaczyÄ‡ wiÄ™cej tablic w Z2 albo pracowaÄ‡ dalej na znakach tablic w Z3."
            )
            step3_primary_label = "Oznacz wiÄ™cej tablic"
            step3_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "open_z2_step3_repair",
            )
            step3_secondary_label = "Pracuj na znakach tablic"
            step3_secondary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "open_z3",
            )

        step3_badge_label = "ZatwierdĹş etap" if step3_state.lower() == "ready" else ""
        step3_details = self._align_step3_details_with_cta_labels(
            state=step3_state,
            details=step3_details,
            primary_label=step3_primary_label,
            secondary_label=step3_secondary_label,
            badge_label=step3_badge_label,
        )

        statuses.append(
            WizardStageStatus(
                key="step3",
                title=str(getattr(step3_vm, "title", "") or "").strip() or "E3. Znaki i gold pack",
                state=step3_state,
                summary=step3_summary,
                details=step3_details,
                primary_label=step3_primary_label,
                primary_command=step3_primary_command,
                secondary_label=step3_secondary_label,
                secondary_command=step3_secondary_command,
                badge_action_label=step3_badge_label,
                badge_action_command=(
                    lambda: campaign_graph_actions.execute_campaign_graph_action(self, "approve_step3")
                    if step3_state.lower() == "ready"
                    else None
                ),
                body_mode=str(getattr(step3_vm, "body_mode", "") or "").strip(),
                body_visible=bool(getattr(step3_vm, "body_visible", False)),
                is_current=bool(not project_suspended and current_step == 3 and iteration_target == "char"),
            )
        )

        step4_badge_label = ""
        step4_badge_command = None

        if project_completed:
            step4_state = "done"
            step4_summary = "Projekt został zakończony. Z4 pozostaje dostępne do przeglądu wyników i analiz."
            step4_details = "Jeśli chcesz uruchomić kolejną iterację w tym projekcie, wznów go z karty projektu."
            step4_secondary_label = "Wznów projekt"
            step4_secondary_command = self._toggle_project_completion
        elif project_paused:
            step4_state = "done"
            step4_summary = "Projekt został odłożony. Możesz wrócić do niego później bez utraty stanu iteracji."
            step4_details = "Po wznowieniu możesz wrócić do Z4 albo rozpocząć nową iterację w tym samym projekcie."
            step4_secondary_label = "Wznów projekt"
            step4_secondary_command = self._toggle_project_completion
        elif current_step >= 5:
            step4_state = "done"
            step4_summary = "Iteracja została domknięta. Możesz przejść do E1 kolejnej iteracji."
            step4_details = (
                "Z4 pozostaje dostępne do przeglądu datasetu, przebiegów treningu i analiz. "
                "Jeśli poprzednio zamknąłeś modal decyzji bez przejścia dalej, użyj badge'a „Przejdź do E1”."
            )
            step4_secondary_label = ""
            step4_secondary_command = None
            step4_badge_label = "Przejdź do E1"
            step4_badge_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "start_next_iteration",
            )
        elif step4_can_advance:
            step4_state = "ready"
            step4_summary = f"Trening tej iteracji jest zakończony. {training_stage_label} jest gotowe do zatwierdzenia."
            step4_details = (
                "Model i przebieg treningu są już gotowe. Użyj badge'a „Zatwierdź etap”, "
                f"aby formalnie zamknąć {training_stage_label} i dopiero potem otwierać kolejną iterację projektu."
            )
            step4_secondary_label = ""
            step4_secondary_command = None
            step4_badge_label = "Zatwierdź etap"
            step4_badge_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "approve_step4",
            )
        elif plate_step4_blocked:
            step4_state = "needs_attention"
            if plate_step4_reason == "stale_plate_dataset":
                step4_summary = "E4T wymaga przebudowy datasetu tablic."
                step4_details = str(plate_step4_gate.get("message") or "").strip() or (
                    "ApprovedSet projektu jest już większy niż ostatnio przygotowany dataset treningowy. "
                    "Przejdź do Z4 i przebuduj dataset w PZ1."
                )
            else:
                step4_summary = "E4T czeka na uzupełnienie oznaczeń tablic w Z2."
                step4_details = str(plate_step4_gate.get("message") or "").strip() or (
                    f"W torze tablic potrzebujesz co najmniej {int(getattr(self, 'STEP2_PLATE_MIN_PLATES', 10) or 10)} zatwierdzonych tablic, zanim wejdziesz do Z4."
                )
            step4_secondary_label = ""
            step4_secondary_command = None
            if not project_suspended and current_step == 4:
                step4_badge_label = "Zakończ etap bez treningu"
                step4_badge_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "approve_step4_without_training",
                )
        elif current_step == 4:
            step4_state = "in_progress"
            step4_summary = f"Budowa datasetu i trening odbywają się w Z4 dla {training_stage_label}."
            if iteration_target == "plate":
                approved_images = max(
                    int(plate_step4_gate.get("project_approved_images", 0) or 0),
                    int(project_approved_images_current or 0),
                )
                approved_plates = max(
                    int(plate_step4_gate.get("project_approved_plates", 0) or 0),
                    int(project_approved_plates_current or 0),
                )
                step4_details = (
                    "W węźle E4T Z4 przygotuje dataset YOLO Pose i uruchomi trening modelu tablic. "
                    f"Źródłem jest zatwierdzony zbiór projektu: {approved_images} obraz(y), {approved_plates} tablic(e)."
                )
            else:
                step4_details = "W węźle E4Z Z4 zbuduje dataset znaków i uruchomi trening modelu YOLO Detect."
            step4_secondary_label = ""
            step4_secondary_command = None
            step4_badge_label = "Zakończ etap bez treningu"
            step4_badge_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                self,
                "approve_step4_without_training",
            )
        else:
            step4_state = "locked"
            if iteration_target == "plate":
                step4_summary = "E4T odblokuje się po zakończeniu Z2 w torze tablic."
            elif iteration_target == "char":
                step4_summary = "E4Z odblokuje się po zakończeniu Z3 w torze znaków."
            else:
                step4_summary = "Najpierw wybierz tor iteracji i przejdź przez wcześniejsze etapy."
            step4_details = "Wizard pokaże Z4 jako ostatni etap, ale cała praca będzie się odbywać już w zakładce treningu."
            step4_secondary_label = ""
            step4_secondary_command = None

        if char_step4_blocked and not project_suspended and current_step < 5:
            step4_state = "needs_attention"
            step4_summary = "E4Z czeka na poprawny dataset znaków z Z3."
            gate_msg = str(char_step4_gate.get("message") or "").strip()
            repair_msg = str(char_repair_guidance.get("details") or "").strip()
            if gate_msg and repair_msg:
                step4_details = gate_msg + "\n\n" + repair_msg
            else:
                step4_details = gate_msg or repair_msg or (
                    "Wróć do Z3 i przygotuj dataset znaków gotowy do treningu."
                )
            step4_secondary_label = ""
            step4_secondary_command = None
            if not project_suspended and current_step == 4:
                step4_badge_label = "Zakończ etap bez treningu"
                step4_badge_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "approve_step4_without_training",
                )

        step4_primary_label = ""
        step4_primary_command = None
        if project_suspended or current_step >= 5 or (current_step == 4 and not step4_gate_blocked):
            step4_primary_label = "Otwórz Z4"
            step4_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z4")
        elif plate_step4_blocked:
            plate_step4_reason = str(plate_step4_gate.get("reason") or "").strip().lower()
            if plate_step4_reason == "stale_plate_dataset":
                step4_primary_label = "Przebuduj dataset w Z4"
                step4_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "open_z4_dataset",
                )
            else:
                step4_primary_label = "Wróć do Z2"
                step4_primary_command = lambda: campaign_graph_actions.execute_campaign_graph_action(
                    self,
                    "open_z2_step3_repair",
                )
        elif char_step4_blocked:
            step4_primary_label = str(char_repair_guidance.get("primary_label") or "Wróć do Z3")
            step4_primary_command = char_repair_guidance.get("primary_command") or (
                lambda: campaign_graph_actions.execute_campaign_graph_action(self, "open_z3")
            )

        statuses.append(
            WizardStageStatus(
                key="step4",
                title=f"{training_stage_label}. Dataset i trening",
                state=step4_state,
                summary=step4_summary,
                details=step4_details,
                primary_label=step4_primary_label,
                primary_command=step4_primary_command,
                secondary_label=step4_secondary_label,
                secondary_command=step4_secondary_command,
                badge_action_label=step4_badge_label,
                badge_action_command=step4_badge_command,
                is_current=bool(not project_suspended and current_step >= 4),
            )
        )

        return [status for status in statuses if getattr(status, "key", "") != "project"]

    @staticmethod
    def _normalize_iteration_target(target: str | None) -> str:
        value = str(target or "").strip().lower()
        if value == "plate":
            return "plate"
        if value == "char":
            return "char"
        return ""

    def _get_iteration_target(self) -> str:
        return self._normalize_iteration_target(CAMPAIGN.get_iteration_target())

    def _get_last_iteration_target(self) -> str:
        try:
            return self._normalize_iteration_target(CAMPAIGN.get_last_iteration_target())
        except Exception:
            return ""

    def _get_default_step2_iteration_target(
        self,
        *,
        current_step: int,
        step1_status: str,
        step2_status: str,
        step3_status: str,
        iteration_target: str,
    ) -> str:
        normalized_target = self._normalize_iteration_target(iteration_target)
        if normalized_target in {"plate", "char"}:
            return normalized_target

        # Od teraz tor jest jawnie wybierany w E1. Nie zgadujemy go na E2,
        # bo to ukrywa stare stany projektu i miesza logikÄ™ tablic/znakĂłw.
        return ""

    def _get_annotation_bootstrap_for_target(self, target: str) -> dict:
        target = self._normalize_iteration_target(target)
        if target not in {"plate", "char"}:
            return {}

        registry_state = self._get_annotation_step2_source_state_from_registry(target)
        if isinstance(registry_state, dict) and registry_state:
            bootstrap = registry_state.get("bootstrap")
            if isinstance(bootstrap, dict) and bootstrap:
                return dict(bootstrap)

        annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
        if annotation_tab is None:
            return {}

        try:
            bootstrap = annotation_tab._get_campaign_auto_annotation_bootstrap(target)
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ pobrac bootstrapu Z2 dla toru {target}: {e}")
            return {}

        return dict(bootstrap) if isinstance(bootstrap, dict) else {}

    def _get_annotation_step2_source_state_from_registry(self, target: str) -> dict:
        target = self._normalize_iteration_target(target)
        if target not in {"plate", "char"}:
            return {}

        try:
            iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        except Exception:
            iteration_num = 1
        images_dir = CAMPAIGN.get_iteration_image_source_dir(iteration_num) or CAMPAIGN.get_master_pool_dir() or CAMPAIGN.get_iteration_raw_dir(iteration_num)
        if images_dir is None:
            return {}

        bundle = dict(
            CAMPAIGN.get_iteration_artifact_bundle(
                images_dir=images_dir,
                iteration_num=iteration_num,
            ) or {}
        )
        if not bundle:
            return {}

        plate_source = dict(bundle.get("plate_source") or {})
        step2_active_run = dict(bundle.get("step2_active_run") or {})
        plate_model = dict(bundle.get("plate_model") or {})
        char_effective = dict(bundle.get("char_effective_source") or {})
        route_hints = dict(bundle.get("route_hints") or {})
        image_source = dict(bundle.get("image_source") or {})
        expected_image_set_token = str(image_source.get("image_set_token") or "").strip()
        plate_expected_token = str(plate_source.get("expected_image_set_token") or "").strip()
        plate_xml_token = str(plate_source.get("xml_image_set_token") or "").strip()
        plate_image_set_match = plate_source.get("image_set_match")
        if plate_image_set_match is None:
            plate_source_usable = bool(
                not expected_image_set_token
                or not plate_xml_token
                or expected_image_set_token == plate_xml_token
                or (plate_expected_token and plate_expected_token == plate_xml_token)
            )
        else:
            plate_source_usable = bool(plate_image_set_match)

        try:
            effective_input_dir = Path(
                str(
                    step2_active_run.get("images_dir")
                    or plate_source.get("images_dir")
                    or image_source.get("master_pool_dir")
                    or CAMPAIGN.get_iteration_image_source_dir(iteration_num)
                    or CAMPAIGN.get_iteration_raw_dir(iteration_num)
                    or ""
                ).strip()
            )
        except Exception:
            effective_input_dir = None
        if effective_input_dir is not None and (not effective_input_dir.exists() or not effective_input_dir.is_dir()):
            effective_input_dir = None

        plate_model_path = str(plate_model.get("path") or "").strip()
        plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())

        result = {
            "iteration_target": target,
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
            "plate_model_ready": plate_model_ready,
        }

        if target == "plate":
            active_run_dir_raw = str(step2_active_run.get("run_dir") or "").strip()
            active_xml_path_raw = str(step2_active_run.get("xml_path") or "").strip()
            try:
                active_run_dir = Path(active_run_dir_raw) if active_run_dir_raw else None
            except Exception:
                active_run_dir = None
            if active_run_dir is not None and (not active_run_dir.exists() or not active_run_dir.is_dir()):
                active_run_dir = None
            active_xml_ok = False
            if active_xml_path_raw:
                try:
                    active_xml_ok = Path(active_xml_path_raw).exists()
                except Exception:
                    active_xml_ok = False
            run_dir_raw = str(plate_source.get("run_dir") or "").strip()
            xml_path_raw = str(plate_source.get("xml_path") or "").strip()
            try:
                run_dir = Path(run_dir_raw) if run_dir_raw else None
            except Exception:
                run_dir = None
            if run_dir is not None and (not run_dir.exists() or not run_dir.is_dir()):
                run_dir = None
            xml_ok = False
            if xml_path_raw:
                try:
                    xml_ok = Path(xml_path_raw).exists()
                except Exception:
                    xml_ok = False
            restore_run_dir = run_dir if (run_dir is not None and xml_ok and plate_source_usable) else None
            if active_run_dir is not None and active_xml_ok:
                restore_run_dir = active_run_dir
            result["restore_run_dir"] = restore_run_dir
            result["run_name"] = str(getattr(restore_run_dir, "name", "") or "").strip()
            result["has_source"] = bool(restore_run_dir is not None)
            result["ready"] = bool(restore_run_dir is not None)
            result["manual_template"] = bool(not restore_run_dir and not plate_model_ready)
            result["input_source"] = (
                str(
                    (
                        step2_active_run.get("input_source")
                        if restore_run_dir is not None and active_run_dir is not None
                        else plate_source.get("input_source")
                    ) or ""
                ).strip()
                or (
                    "registry_step2_active_run"
                    if restore_run_dir is not None and active_run_dir is not None
                    else ("registry_plate_source" if restore_run_dir is not None else "registry")
                )
            )
            result["bootstrap"] = {
                "restore_run_dir": restore_run_dir,
                "input_dir": effective_input_dir,
                "input_source": result["input_source"],
                "manual_template": bool(result["manual_template"]),
                "plate_model_path": plate_model_path if plate_model_ready else "",
            }
            if restore_run_dir is not None or plate_model_ready or effective_input_dir is not None:
                return result
            return {}

        try:
            approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            project_approved_images = int(approved_stats.get("images", 0) or 0)
            project_approved_plates = int(approved_stats.get("plates", 0) or 0)
        except Exception:
            project_approved_images = 0
            project_approved_plates = 0

        raw_char_images = max(
            int(char_effective.get("images_with_plates", 0) or 0),
            int(route_hints.get("images_with_plates", 0) or 0),
            int(project_approved_images or 0),
        )
        raw_char_plates = max(
            int(char_effective.get("total_plates", 0) or 0),
            int(route_hints.get("total_plates", 0) or 0),
            int(project_approved_plates or 0),
        )
        approved_char_images = int(
            max(
                int(char_effective.get("images_with_plates", 0) or 0),
                int(plate_source.get("approved_images", 0) or 0),
                int(project_approved_images or 0),
            )
        )
        approved_char_plates = int(
            max(
                int(char_effective.get("total_plates", 0) or 0),
                int(plate_source.get("approved_plates", 0) or 0),
                int(project_approved_plates or 0),
            )
        )
        run_dir_raw = str(char_effective.get("run_dir") or plate_source.get("run_dir") or "").strip()
        run_name = str(char_effective.get("display_name") or "").strip()
        char_effective_images_dir = None
        char_images_raw = str(char_effective.get("images_dir") or "").strip()
        if char_images_raw:
            try:
                candidate_char_images = Path(char_images_raw)
                if candidate_char_images.exists() and candidate_char_images.is_dir():
                    char_effective_images_dir = candidate_char_images
            except Exception:
                char_effective_images_dir = None
        try:
            restore_run_dir = Path(run_dir_raw) if run_dir_raw else None
        except Exception:
            restore_run_dir = None
        if restore_run_dir is not None and (not restore_run_dir.exists() or not restore_run_dir.is_dir()):
            restore_run_dir = None
        if not run_name and restore_run_dir is not None:
            run_name = str(restore_run_dir.name or "").strip()
        if not run_name and project_approved_plates > 0:
            run_name = "Zatwierdzony zbiĂłr projektu tablic"

        has_source = bool(
            (restore_run_dir is not None and plate_source_usable)
            or raw_char_plates > 0
            or project_approved_plates > 0
            or str(route_hints.get("char_entry_mode") or "").strip().lower() in {"ready", "needs_more_tables"}
        )
        min_char_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
        ready = bool(approved_char_plates >= min_char_plates)
        needs_more_tables = bool(has_source and not ready)
        char_source_scope = (
            "campaign_approved_set"
            if project_approved_plates > 0 and project_approved_images >= approved_char_images
            else "campaign_char_effective_source"
        )
        result.update(
            has_source=has_source,
            ready=ready,
            needs_more_tables=needs_more_tables,
            images_with_plates=approved_char_images,
            total_plates=approved_char_plates,
            project_images_with_plates=project_approved_images,
            project_total_plates=project_approved_plates,
            source_scope=char_source_scope,
            restore_run_dir=restore_run_dir,
            run_name=run_name,
            input_source=(
                "campaign_char_effective_source"
                if char_effective
                else ("registry_plate_source" if has_source else "")
            ),
            manual_template=False,
            bootstrap={
                "restore_run_dir": restore_run_dir,
                "input_dir": char_effective_images_dir or effective_input_dir,
                "xml_path": (
                    Path(str(char_effective.get("xml_path") or "").strip())
                    if str(char_effective.get("xml_path") or "").strip()
                    else None
                ),
                "input_source": (
                    "campaign_char_effective_source"
                    if char_effective
                    else ("registry_plate_source" if has_source else "")
                ),
                "manual_template": False,
                "plate_model_path": plate_model_path if plate_model_ready else "",
                "display_name": run_name,
                "run_name": run_name,
                "contributor_run_dir": str(char_effective.get("contributor_run_dir") or "").strip(),
                "source_scope": char_source_scope,
            },
        )
        return result if has_source or plate_model_ready else {}

    def _get_annotation_step2_source_state(self, target: str) -> dict:
        target = self._normalize_iteration_target(target)
        if target not in {"plate", "char"}:
            return {}

        cache_key = (
            str(CAMPAIGN.get_active_project_name() or "").strip(),
            int(CAMPAIGN.get_current_iteration_num() or 1),
            int(CAMPAIGN.get_current_step() or 1),
            str(CAMPAIGN.get_step2_status() or "").strip().lower(),
            str(CAMPAIGN.get_step3_status() or "").strip().lower(),
            target,
        )
        state_cache = self._get_dashboard_cache_bucket("step2_source_states")
        cached = state_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)

        persistent_signature = self._build_step2_source_state_signature(target)
        persistent_scope = f"step2_source_state:{target}"
        persistent_cached = self._get_project_view_cache_entry(persistent_scope, persistent_signature)
        if isinstance(persistent_cached, dict) and persistent_cached:
            state_cache[cache_key] = dict(persistent_cached)
            return dict(persistent_cached)

        registry_state = self._get_annotation_step2_source_state_from_registry(target)
        if isinstance(registry_state, dict) and registry_state:
            state_cache[cache_key] = dict(registry_state)
            self._set_project_view_cache_entry(
                persistent_scope,
                persistent_signature,
                registry_state,
            )
            return dict(registry_state)

        annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
        if annotation_tab is None:
            return {}

        getter = getattr(annotation_tab, "get_campaign_step2_source_state", None)
        if not callable(getter):
            return {}

        source_started = perf_counter()
        try:
            source_state = getter(iteration_target=target)
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ pobrac stanu ĹşrĂłdĹ‚a E2 z Z2 dla toru {target}: {e}")
            return {}

        normalized_state = dict(source_state) if isinstance(source_state, dict) else {}
        state_cache[cache_key] = dict(normalized_state)
        if normalized_state:
            self._set_project_view_cache_entry(
                persistent_scope,
                persistent_signature,
                normalized_state,
            )
        self._log_perf(
            f"step2_source_state[{target}]",
            source_started,
            threshold_ms=20.0,
            extra=f"ready={bool(normalized_state.get('ready'))}, has_source={bool(normalized_state.get('has_source'))}",
        )
        return normalized_state

    def _get_plate_approved_set_stats(self) -> dict:
        try:
            manifest_path = CAMPAIGN.get_plate_approved_set_path()
        except Exception:
            manifest_path = None

        cache = getattr(self, "_dashboard_perf_cache", {})
        stats_cache = cache.get("approved_stats", {}) if isinstance(cache, dict) else {}
        cache_key = ("plate_approved_stats", self._build_cache_token_for_path(manifest_path))
        cached = stats_cache.get(cache_key) if isinstance(stats_cache, dict) else None
        if isinstance(cached, dict):
            return dict(cached)

        try:
            stats = CAMPAIGN.get_plate_approved_set_stats()
        except Exception as e:
            logger.debug(f"Nie udaĹ‚o siÄ™ pobrac statystyk ApprovedSet tablic: {e}")
            return {}
        result = dict(stats) if isinstance(stats, dict) else {}
        if isinstance(stats_cache, dict):
            if len(stats_cache) > 64:
                stats_cache.clear()
            stats_cache[cache_key] = dict(result)
        return result

    def _get_step2_disk_approval_fallback(self, *args, **kwargs):
        return campaign_stage_logic._get_step2_disk_approval_fallback(self, *args, **kwargs)


    def _get_step2_current_iteration_contribution_state(self, *args, **kwargs):
        return campaign_stage_logic._get_step2_current_iteration_contribution_state(self, *args, **kwargs)


    def _confirm_step2_without_current_iteration_contribution(self, *args, **kwargs):
        return campaign_stage_logic._confirm_step2_without_current_iteration_contribution(self, *args, **kwargs)


    def _resolve_step2_wizard_action_command(self, *args, **kwargs):
        return campaign_stage_logic._resolve_step2_wizard_action_command(self, *args, **kwargs)


    def _approve_step2_from_wizard(self, *args, **kwargs):
        return campaign_stage_logic._approve_step2_from_wizard(self, *args, **kwargs)


    def _approve_step1_from_wizard(self, *args, **kwargs):
        return campaign_stage_logic._approve_step1_from_wizard(self, *args, **kwargs)


    def _get_annotation_step2_view_model(self, *args, **kwargs):
        return campaign_stage_logic._get_annotation_step2_view_model(self, *args, **kwargs)


    def _resolve_step3_wizard_action_command(self, *args, **kwargs):
        return campaign_stage_logic._resolve_step3_wizard_action_command(self, *args, **kwargs)


    def _get_step3_current_iteration_contribution_state(self, *args, **kwargs):
        return campaign_stage_logic._get_step3_current_iteration_contribution_state(self, *args, **kwargs)


    def _confirm_step3_without_current_iteration_contribution(self, *args, **kwargs):
        return campaign_stage_logic._confirm_step3_without_current_iteration_contribution(self, *args, **kwargs)


    def _approve_step3_from_wizard(self, *args, **kwargs):
        return campaign_stage_logic._approve_step3_from_wizard(self, *args, **kwargs)


    def _get_campaign_step3_view_model(self, *args, **kwargs):
        return campaign_stage_logic._get_campaign_step3_view_model(self, *args, **kwargs)


    def _get_char_route_ready_source(self, *args, **kwargs):
        return campaign_stage_logic._get_char_route_ready_source(self, *args, **kwargs)


    def _get_char_training_split_preview(self, *args, **kwargs):
        return campaign_stage_logic._get_char_training_split_preview(self, *args, **kwargs)


    def _looks_like_campaign_char_dataset_dir(self, *args, **kwargs):
        return campaign_stage_logic._looks_like_campaign_char_dataset_dir(self, *args, **kwargs)


    def _detect_campaign_char_ready_dataset_state(self, *args, **kwargs):
        return campaign_stage_logic._detect_campaign_char_ready_dataset_state(self, *args, **kwargs)


    def _get_char_repair_guidance(self, *args, **kwargs):
        return campaign_stage_logic._get_char_repair_guidance(self, *args, **kwargs)


    def _get_pending_step3_char_union_state(self, *args, **kwargs):
        return campaign_stage_logic._get_pending_step3_char_union_state(self, *args, **kwargs)


    def _get_char_route_source_state(self, *args, **kwargs):
        return campaign_stage_logic._get_char_route_source_state(self, *args, **kwargs)


    def _get_plate_route_ready_source(self, *args, **kwargs):
        return campaign_stage_logic._get_plate_route_ready_source(self, *args, **kwargs)


    def _get_step2_jump_button_text(self, *args, **kwargs):
        return campaign_stage_logic._get_step2_jump_button_text(self, *args, **kwargs)


    @staticmethod
    def _iteration_target_label(*args, **kwargs):
        return campaign_stage_logic._iteration_target_label(*args, **kwargs)


    def _iteration_target_button_label(self, *args, **kwargs):
        return campaign_stage_logic._iteration_target_button_label(self, *args, **kwargs)


    def _get_iteration_target_lock_reason(self, *args, **kwargs):
        return campaign_stage_logic._get_iteration_target_lock_reason(self, *args, **kwargs)


    def _get_step1_char_preflight_image_count(self, *args, **kwargs):
        return campaign_stage_logic._get_step1_char_preflight_image_count(self, *args, **kwargs)


    @staticmethod
    def _count_plate_annotations_in_xml(*args, **kwargs):
        return campaign_stage_logic._count_plate_annotations_in_xml(*args, **kwargs)


    def _count_step1_imported_plate_annotations(self, *args, **kwargs):
        return campaign_stage_logic._count_step1_imported_plate_annotations(self, *args, **kwargs)


    def _get_step1_char_route_preflight_state(self, *args, **kwargs):
        return campaign_stage_logic._get_step1_char_route_preflight_state(self, *args, **kwargs)


    def _get_step1_char_preflight_signature(self, *args, **kwargs):
        return campaign_stage_logic._get_step1_char_preflight_signature(self, *args, **kwargs)


    def _confirm_step1_char_preflight_warning(self, *args, **kwargs):
        return campaign_stage_logic._confirm_step1_char_preflight_warning(self, *args, **kwargs)


    def _validate_step1_char_preflight_for_approval(self, *args, **kwargs):
        return campaign_stage_logic._validate_step1_char_preflight_for_approval(self, *args, **kwargs)


    def _format_step1_char_route_card_body(self, *args, **kwargs):
        return campaign_stage_logic._format_step1_char_route_card_body(self, *args, **kwargs)


    def _get_step1_char_route_block_reason(self, *args, **kwargs):
        return campaign_stage_logic._get_step1_char_route_block_reason(self, *args, **kwargs)


    def _has_saved_step3_progress(self) -> bool:
        try:
            saved_substep = int(CAMPAIGN.get_step3_substep() or 1)
        except Exception:
            saved_substep = 1

        try:
            stage1_done = bool(CAMPAIGN.is_step3_stage1_done())
        except Exception:
            stage1_done = False

        try:
            stage2_done = bool(CAMPAIGN.is_step3_stage2_done())
        except Exception:
            stage2_done = False

        try:
            extract_state = CAMPAIGN.get_step3_extract_state() or {}
        except Exception:
            extract_state = {}

        entry_mode = str(extract_state.get("entry_mode", "") or "").strip()
        workflow_step = str(extract_state.get("workflow_step", "entry") or "entry").strip().lower()
        annotation_run_dir = str(extract_state.get("annotation_run_dir", "") or "").strip()
        xml_path = str(extract_state.get("xml_path", "") or "").strip()
        images_dir = str(extract_state.get("images_dir", "") or "").strip()

        return bool(
            saved_substep > 1
            or stage1_done
            or stage2_done
            or entry_mode
            or workflow_step != "entry"
            or annotation_run_dir
            or xml_path
            or images_dir
        )

    @staticmethod
    def _campaign_paths_equivalent(path_a, path_b) -> bool:
        text_a = str(path_a or "").strip()
        text_b = str(path_b or "").strip()
        if not text_a or not text_b:
            return False
        try:
            return Path(text_a).resolve() == Path(text_b).resolve()
        except Exception:
            return text_a == text_b

    def _should_reset_stale_char_iteration_state(
        self,
        *,
        current_step: int,
        step2_status: str,
        step3_status: str,
        iteration_target: str,
    ) -> bool:
        if bool(getattr(self, "_project_open_lightweight_refresh", False)):
            return False

        try:
            suppress_until = float(getattr(self, "_suppress_stale_char_iteration_reset_until", 0.0) or 0.0)
        except Exception:
            suppress_until = 0.0
        if suppress_until > 0.0 and perf_counter() < suppress_until:
            return False

        if self._normalize_iteration_target(iteration_target) != "char":
            return False

        try:
            current_char_source_state = dict(
                self._get_annotation_step2_source_state_from_registry("char") or {}
            )
        except Exception:
            current_char_source_state = {}
        if bool(current_char_source_state.get("has_source")):
            return False

        normalized_step2_status = str(step2_status or "").strip().lower()
        normalized_step3_status = str(step3_status or "").strip().lower()
        if (
            int(current_step or 0) < 3
            and normalized_step2_status not in {"approved", "generated"}
            and normalized_step3_status == "pending"
        ):
            return False

        try:
            if str(CAMPAIGN.get_step2_staging_run() or "").strip():
                return False
        except Exception:
            pass

        try:
            current_iter_dir = CAMPAIGN.get_iteration_image_source_dir() or CAMPAIGN.get_iteration_raw_dir()
        except Exception:
            current_iter_dir = None
        if current_iter_dir is None:
            return False

        try:
            stored_manual_source = CAMPAIGN.get_last_plate_manual_source() or {}
        except Exception:
            stored_manual_source = {}

        source_input = str((stored_manual_source or {}).get("source_input_path") or "").strip()
        if not source_input:
            return False

        return not self._campaign_paths_equivalent(current_iter_dir, source_input)

    def _choose_step1_iteration_target(self, *args, **kwargs):
        return campaign_dashboard_ui._choose_step1_iteration_target(self, *args, **kwargs)


    def _render_step1_route_actions(self, *args, **kwargs):
        return campaign_dashboard_ui._render_step1_route_actions(self, *args, **kwargs)


    # ======================================================
    # DASHBOARD REFRESH
    # ======================================================

    def _refresh_dashboard(self, *args, **kwargs):
        return campaign_dashboard_ui._refresh_dashboard(self, *args, **kwargs)


    def get_active_campaign_graph_gate_badge_label(self, *args, **kwargs):
        return campaign_dashboard_ui.get_active_campaign_graph_gate_badge_label(self, *args, **kwargs)


    def _update_main_tabs_highlight(self, *args, **kwargs):
        return campaign_dashboard_ui._update_main_tabs_highlight(self, *args, **kwargs)


    # ======================================================
    # PROJECT CRUD
    # ======================================================

    def _ensure_project_context_is_switchable(self, *args, **kwargs):
        return campaign_project_browser._ensure_project_context_is_switchable(self, *args, **kwargs)
    def _reset_campaign_graph_runtime_state(self, *args, **kwargs):
        return campaign_project_browser._reset_campaign_graph_runtime_state(self, *args, **kwargs)
    def _add_new_project(self, *args, **kwargs):
        return campaign_project_browser._add_new_project(self, *args, **kwargs)
    def _get_selected_projects_from_list(self, *args, **kwargs):
        return campaign_project_browser._get_selected_projects_from_list(self, *args, **kwargs)
    def _get_selected_project_from_list(self, *args, **kwargs):
        return campaign_project_browser._get_selected_project_from_list(self, *args, **kwargs)
    def _select_project_in_list(self, *args, **kwargs):
        return campaign_project_browser._select_project_in_list(self, *args, **kwargs)
    def _show_project_context_menu(self, *args, **kwargs):
        return campaign_project_browser._show_project_context_menu(self, *args, **kwargs)
    def _set_project_list_status(self, *args, **kwargs):
        return campaign_project_browser._set_project_list_status(self, *args, **kwargs)
    def _refresh_projects_list(self, *args, **kwargs):
        return campaign_project_browser._refresh_projects_list(self, *args, **kwargs)
    def _on_project_changed(self, *args, **kwargs):
        return campaign_project_browser._on_project_changed(self, *args, **kwargs)
    def _open_selected_project(self, *args, **kwargs):
        return campaign_project_browser._open_selected_project(self, *args, **kwargs)
    def _schedule_project_open_post_refresh(self, *args, **kwargs):
        return campaign_project_browser._schedule_project_open_post_refresh(self, *args, **kwargs)
    def _get_project_switch_blocker_message(self, *args, **kwargs):
        return campaign_project_browser._get_project_switch_blocker_message(self, *args, **kwargs)
    def _release_active_project_resources_before_switch(self, *args, **kwargs):
        return campaign_project_browser._release_active_project_resources_before_switch(self, *args, **kwargs)
    def _delete_project(self, *args, **kwargs):
        return campaign_project_browser._delete_project(self, *args, **kwargs)
    def _clear_project_contexts(self, *args, **kwargs):
        return campaign_project_browser._clear_project_contexts(self, *args, **kwargs)
    def _exit_project_mode(self, *args, **kwargs):
        return campaign_project_browser._exit_project_mode(self, *args, **kwargs)
    def _step_open_z2_from_step2_review(self, *args, **kwargs):
        return campaign_navigation._step_open_z2_from_step2_review(self, *args, **kwargs)


    def _step_open_z2_repair_from_later_stage(self, *args, **kwargs):
        return campaign_navigation._step_open_z2_repair_from_later_stage(self, *args, **kwargs)


    def _step_return_to_annotation_review(self, *args, **kwargs):
        return campaign_navigation._step_return_to_annotation_review(self, *args, **kwargs)


    def _step_goto_auto_annotation(self, *args, **kwargs):
        return campaign_navigation._step_goto_auto_annotation(self, *args, **kwargs)


    def _return_to_step1_for_char_source_rework(self, *args, **kwargs):
        return campaign_navigation._return_to_step1_for_char_source_rework(self, *args, **kwargs)


    def _show_step2_char_minimum_not_met_dialog(self, *args, **kwargs):
        return campaign_navigation._show_step2_char_minimum_not_met_dialog(self, *args, **kwargs)


    def _finish_step2_char_and_focus_step3(self, *args, **kwargs):
        return campaign_navigation._finish_step2_char_and_focus_step3(self, *args, **kwargs)


    def _step_continue_characters_from_ready_source(self, *args, **kwargs):
        return campaign_navigation._step_continue_characters_from_ready_source(self, *args, **kwargs)


    def _step_goto_characters_detect(self, *args, **kwargs):
        return campaign_navigation._step_goto_characters_detect(self, *args, **kwargs)


    def _step_goto_characters(self, *args, **kwargs):
        return campaign_navigation._step_goto_characters(self, *args, **kwargs)


    def _step_goto_training(self, *args, **kwargs):
        return campaign_navigation._step_goto_training(self, *args, **kwargs)


    def _step_goto_training_dataset(self, *args, **kwargs):
        return campaign_iteration_flow._step_goto_training_dataset(self, *args, **kwargs)


    def _show_step4_completion_next_action_modal(self, *args, **kwargs):
        return campaign_iteration_flow._show_step4_completion_next_action_modal(self, *args, **kwargs)


    def _start_iteration_advance(self, *args, **kwargs):
        return campaign_iteration_flow._start_iteration_advance(self, *args, **kwargs)


    def _finish_step4_iteration(self, *args, **kwargs):
        return campaign_iteration_flow._finish_step4_iteration(self, *args, **kwargs)


    def _finish_step4_without_training(self, *args, **kwargs):
        return campaign_iteration_flow._finish_step4_without_training(self, *args, **kwargs)


    def _set_iteration_advance_busy(self, *args, **kwargs):
        return campaign_iteration_flow._set_iteration_advance_busy(self, *args, **kwargs)


    def _run_iteration_advance_worker(self, *args, **kwargs):
        return campaign_iteration_flow._run_iteration_advance_worker(self, *args, **kwargs)


    def _schedule_iteration_advance_poll(self, *args, **kwargs):
        return campaign_iteration_flow._schedule_iteration_advance_poll(self, *args, **kwargs)


    def _poll_iteration_advance_worker(self, *args, **kwargs):
        return campaign_iteration_flow._poll_iteration_advance_worker(self, *args, **kwargs)


    def _finish_iteration_advance(self, *args, **kwargs):
        return campaign_iteration_flow._finish_iteration_advance(self, *args, **kwargs)


    def _toggle_project_completion(self, *args, **kwargs):
        return campaign_iteration_flow._toggle_project_completion(self, *args, **kwargs)
