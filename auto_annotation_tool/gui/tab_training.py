#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka Treningu: trening YOLO + analiza modeli.

W trybie swobodnym Z4 konsumuje gotowy dataset z Z2 lub Z3.
Pomost datasetowy pozostaje tylko na potrzeby kampanii.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import datetime
import time
import webbrowser
import csv
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import (
    CONFIG,
    YOLO_AVAILABLE,
    AVAILABLE_POSE_MODELS,
    AVAILABLE_DETECT_MODELS,
    PIL_AVAILABLE,
    get_torch_module,
    get_yolo_class,
    is_cuda_available,
    logger,
)
from ..icons import IconManager
from ..validators import validate_model_file, format_yolo_model_identity
from ..training import YOLOPoseTrainer, TrainingHistory, TrainingStatus, DatasetCreator, DatasetSplitter
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .dataset_display import build_dataset_display_ref
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_train_progress_bar import TrainProgressBar
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    finish_campaign_step4,
    get_campaign_training_target,
    open_campaign_step4_entry,
    poll_training_completion,
    restore_step4_campaign_project_state,
    set_campaign_context,
    set_campaign_training_target,
)
from .z4_flow_models import (
    CharYoloDatasetSourceAdapter,
    PlateXmlImagesSourceAdapter,
    TrainingSource,
    TrainingSourceStats,
)
from .z4_free_mode_flow import (
    refresh_free_training_route_cards,
    refresh_free_training_route_ui,
    update_step4_notebook_mode,
)
from .z4_shared_ui import (
    accept_training_input_context,
    clear_step4_guidance,
    guide_step4_builder_action,
    guide_step4_finish_action,
    guide_step4_next_action,
    guide_step4_route_selection,
    mark_step4_dataset_ready,
    open_step4_dataset_stage,
    refresh_step4_campaign_builder_inputs_ui,
    refresh_step4_analysis_tab_visibility,
    refresh_step4_campaign_navigation_ui,
    refresh_step4_dataset_mode_ui,
    refresh_step4_training_inputs_mode_ui,
    set_step4_dataset_mode,
    sync_step4_analysis_nav_buttons,
    step4_dataset_go_back,
    step4_dataset_go_next,
    step4_train_go_back,
)
from . import z4_dataset_sources
from . import z4_training_metrics
from . import z4_dataset_builder
from . import z4_analysis_ranking
from . import z4_training_runtime
from . import z4_training_compare
from . import z4_campaign_state
from . import z4_dataset_validation
from . import z4_layout_runtime
from . import z4_device_runtime
from . import z4_model_export
from . import z4_character_balance
from . import z4_training_progress
from . import z4_ui_runtime
from . import z4_theme_runtime
from . import z4_tab_shell
from . import z4_history_runtime
from . import z4_dataset_panels
from . import z4_validation_panel
from . import z4_dataset_tab_builder
from . import z4_train_tab_builder
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None




class TrainingTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.icon_manager = IconManager

        self.frame = ttk.Frame(parent)
        self._inertial_scroll = InertialScrollController(self.frame)
        self._startup_ui_ready = False

        self.history = TrainingHistory(history_dir=Path(CONFIG.get_training_runs_dir("char")))
        self.trainer = YOLOPoseTrainer(history=self.history)
        self.creator = DatasetCreator()
        self.splitter = DatasetSplitter()
        self.ranking_engine = ModelRanking(ranking_dir=Path(CONFIG.get_ranking_dir("plate")))

        self.current_run_id = None
        self._pending_campaign_model_type = None
        self._campaign_training_target = "plate"
        self._step4_dataset_mode = "plate"
        self._step4_builder_log_visible = False
        self._step4_train_log_visible = False
        self._current_training_dataset_is_pose = None
        self._step4_campaign_finish_ready = False
        self._step4_route_selected = False
        self._step4_train_unlocked = False
        self._training_completion_poll_job = None
        # Kontekst projektu jest ustawiany przez wizard kampanii.
        self._campaign_runs_dir = None
        self._campaign_datasets_dir = None
        self._plots_paths = []
        self._plot_photo = None
        self._plot_img_id = None
        self._plot_original_path = None
        self._free_route_hover_mode = None
        self._free_training_route_cards = {}
        self._training_device_profiles = []
        self._training_device_label_map = {}
        self._training_auto_device_label = "Auto"
        self._training_cpu_device_label = "CPU"
        self._step4_ranking_tab_visible = False
        self._train_left_wrap_targets = []
        self._train_left_section_separators = []
        self._latest_training_metrics = {}
        self._current_training_metric_history = []
        self._last_training_completion_summary_run_id = None
        self._dynamic_metric_tables = []
        self._analysis_dialog = None
        self._analysis_dialog_shell = None
        self._analysis_plot_paths = []
        self._analysis_plot_canvas = None
        self._analysis_plots_list = None
        self._training_compare_dialog = None
        self._training_compare_runs_data = []
        self._training_compare_hover_points = []
        self._ui_dispatch_queue = queue.Queue()
        self._ui_dispatch_after_id = None
        self._ui_dispatch_lock = threading.Lock()
        self._ui_dispatch_closed = False
        self._train_pane_layout_initialized = False
        self._training_visible_layout_after_id = None
        self._training_started_monotonic = None
        self._training_started_wall_clock = None
        self._training_eta_seconds = None
        self._training_last_epoch = 0
        self._training_last_total_epochs = 0
        self._training_last_batch = 0
        self._training_last_total_batches = 0
        self._step4_train_tab_built = False

        self.is_processing = False
        self.dataset_build_is_running = False
        self.dataset_split_is_running = False
        self.val_is_running = False
        self.rank_is_running = False
        self.rank_cancel_requested = False
        self._rank_watchdog_job = None
        self._rank_watchdog_last_touch = 0.0
        self._rank_watchdog_last_notice = 0.0
        self._rank_watchdog_stage = ""
        self._rank_watchdog_warn_after_s = 15.0
        self._rank_watchdog_repeat_s = 15.0

        self._ensure_step4_training_state_vars()
        self._build_ui()
        self._ensure_ui_dispatch_pump()
        self._attach_training_log_handlers()
        self._bind_trainer_callbacks()
        self._schedule_initial_step4_refresh()

    def _mark_startup_ui_ready(self):
        try:
            self._configure_train_progress_styles()
        except Exception:
            pass
        self._startup_ui_ready = True

    def is_startup_ui_ready(self) -> bool:
        return bool(getattr(self, "_startup_ui_ready", False))

    def _on_app_close(self, event=None):
        """Persist an interrupted campaign Z4 session when the app is closed from this card."""
        if not CAMPAIGN.get_active_project_name():
            return
        try:
            selected_tab = str(self.parent.select() or "")
            if selected_tab and selected_tab != str(self.frame):
                return
        except Exception:
            return
        try:
            current_step = int(CAMPAIGN.get_current_step() or 0)
        except Exception:
            current_step = 0
        if current_step != 4:
            return
        try:
            target = str(CAMPAIGN.get_iteration_target() or self.get_campaign_training_target() or "").strip().lower()
        except Exception:
            target = str(getattr(self, "_campaign_training_target", "") or "").strip().lower()
        if target not in {"char", "plate"}:
            target = "char"

        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            current_iteration = 0
        if current_iteration <= 0:
            return

        try:
            self._reload_history_snapshot_from_disk()
        except Exception:
            pass
        try:
            run_id = str(getattr(self, "current_run_id", "") or "").strip()
            if run_id:
                CAMPAIGN.sync_step4_training_record_from_history(iteration_num=current_iteration)
                run = self.history.get_run(run_id) if getattr(self, "history", None) is not None else None
                run_status = str(getattr(run, "status", "") or "").strip().lower() if run is not None else ""
                if run_status in {
                    TrainingStatus.COMPLETED.value,
                    TrainingStatus.FAILED.value,
                    TrainingStatus.PAUSED.value,
                    TrainingStatus.CANCELLED.value,
                }:
                    return
        except Exception:
            pass

        finish_state = {}
        try:
            finish_state = dict(self.get_campaign_step4_finish_state(iteration_target=target) or {})
        except Exception:
            finish_state = {}
        if bool(finish_state.get("ready")):
            return
        try:
            no_training = dict(CAMPAIGN.get_step4_without_training_decision() or {})
            no_training_iteration = int(no_training.get("iteration", 0) or 0)
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
            no_training_target = str(no_training.get("target", "") or "").strip().lower()
            if (
                bool(no_training.get("ready"))
                and no_training_iteration == current_iteration
                and (not no_training_target or no_training_target == target)
            ):
                return
        except Exception:
            try:
                current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
            except Exception:
                current_iteration = 0

        try:
            has_visible_work = bool(
                getattr(self, "_step4_route_selected", False)
                or getattr(self, "_step4_train_unlocked", False)
                or str(getattr(self, "current_run_id", "") or "").strip()
            )
        except Exception:
            has_visible_work = False
        try:
            if str(self.dataset_var.get() or "").strip():
                has_visible_work = True
        except Exception:
            pass
        try:
            if str(self.split_src_var.get() or "").strip():
                has_visible_work = True
        except Exception:
            pass
        if not has_visible_work:
            return

        substep = "pz1"
        try:
            selected_subtab = str(self.main_nb.select() or "")
            if selected_subtab == str(getattr(self, "tab_train", "")):
                substep = "pz2"
            elif selected_subtab == str(getattr(self, "tab_dataset", "")):
                substep = "pz1"
        except Exception:
            pass

        now = datetime.datetime.now().isoformat(timespec="seconds")
        session = {
            "active": True,
            "state": "interrupted",
            "work_area": "z4",
            "substep": substep,
            "working_gate_id": "T06",
            "target": target,
            "iteration": current_iteration,
            "reason": "app_close_without_graph_return",
            "interrupted_at": now,
            "updated_at": now,
        }
        try:
            run_id = str(getattr(self, "current_run_id", "") or "").strip()
            if run_id:
                session["run_id"] = run_id
        except Exception:
            pass
        try:
            dataset_path = str(self.dataset_var.get() or "").strip()
            if dataset_path:
                session["dataset_path"] = dataset_path
        except Exception:
            pass

        try:
            CAMPAIGN.upsert_iteration_state(
                iteration_num=current_iteration,
                updates={"step4_work_session": session},
            )
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać przerwanej sesji Z4: {exc}")

    def _ensure_step4_training_state_vars(self) -> None:
        """Keep Z4 campaign state available before the lazy PZ2 tab is built."""
        if not hasattr(self, "dataset_var"):
            self.dataset_var = tk.StringVar()
        if not hasattr(self, "dataset_variant_var"):
            self.dataset_variant_var = tk.StringVar()
        if not hasattr(self, "base_model_var"):
            self.base_model_var = tk.StringVar(value=self._get_default_base_model_for_mode("char"))
        if not hasattr(self, "base_custom_var"):
            self.base_custom_var = tk.StringVar()
        if not hasattr(self, "name_var"):
            self.name_var = tk.StringVar()
        if not hasattr(self, "epochs_var"):
            self.epochs_var = tk.IntVar(value=100)
        if not hasattr(self, "batch_var"):
            self.batch_var = tk.IntVar(value=16)
        if not hasattr(self, "imgsz_var"):
            self.imgsz_var = tk.IntVar(value=640)
        if not hasattr(self, "lr0_var"):
            self.lr0_var = tk.DoubleVar(value=0.01)
        if not hasattr(self, "device_var"):
            try:
                device_value = self._get_global_training_device_choice()
            except Exception:
                device_value = "Auto"
            self.device_var = tk.StringVar(value=device_value)

    def _schedule_initial_step4_refresh(self):
        frame = getattr(self, "frame", None)
        if frame is None:
            return

        def run_lightweight_refresh():
            try:
                self._refresh_step4_dataset_mode_ui(lightweight=True)
            except Exception:
                pass
            try:
                self._refresh_free_training_route_ui()
            except Exception:
                pass
            try:
                self._mark_startup_ui_ready()
            except Exception:
                pass

        def run_training_refresh():
            if not bool(getattr(self, "_step4_train_tab_built", False)):
                return
            try:
                self._refresh_base_model_choices()
            except Exception:
                pass
            try:
                self._refresh_dataset_variant_choices()
            except Exception:
                pass
            try:
                self._refresh_step4_training_inputs_mode_ui()
            except Exception:
                pass

        def run_history_refresh():
            if not bool(getattr(self, "_step4_train_tab_built", False)):
                return
            try:
                self._load_history()
            except Exception:
                pass

        def run_ranking_refresh():
            if not bool(getattr(self, "_step4_train_tab_built", False)):
                return
            try:
                self._load_ranking()
            except Exception:
                pass

        try:
            frame.after_idle(run_lightweight_refresh)
            frame.after(80, run_training_refresh)
            frame.after(180, run_history_refresh)
            frame.after(320, run_ranking_refresh)
        except Exception:
            run_lightweight_refresh()

    def _ui(self, *args, **kwargs):
        return z4_ui_runtime._ui(self, *args, **kwargs)

    def _ensure_ui_dispatch_pump(self, *args, **kwargs):
        return z4_ui_runtime._ensure_ui_dispatch_pump(self, *args, **kwargs)

    def _drain_ui_dispatch_queue(self, *args, **kwargs):
        return z4_ui_runtime._drain_ui_dispatch_queue(self, *args, **kwargs)

    @staticmethod
    def _capture_text_widget_view_state(text_widget) -> dict:
        state = {
            "insert": None,
            "x_first": 0.0,
            "y_first": 0.0,
            "view_at_end": True,
            "insert_at_end": True,
            "sel_first": None,
            "sel_last": None,
        }
        if text_widget is None:
            return state

        try:
            state["insert"] = text_widget.index(tk.INSERT)
        except Exception:
            state["insert"] = None

        try:
            x_first, _x_last = text_widget.xview()
            state["x_first"] = float(x_first)
        except Exception:
            state["x_first"] = 0.0

        try:
            y_first, y_last = text_widget.yview()
            state["y_first"] = float(y_first)
            state["view_at_end"] = float(y_last) >= 0.999
        except Exception:
            state["y_first"] = 0.0
            state["view_at_end"] = True

        try:
            state["insert_at_end"] = bool(text_widget.compare(tk.INSERT, ">=", "end-2c"))
        except Exception:
            state["insert_at_end"] = True

        try:
            state["sel_first"] = text_widget.index("sel.first")
            state["sel_last"] = text_widget.index("sel.last")
        except Exception:
            state["sel_first"] = None
            state["sel_last"] = None

        return state

    @staticmethod
    def _restore_text_widget_view_state(text_widget, state: dict) -> None:
        if text_widget is None or not isinstance(state, dict):
            return

        try:
            insert_index = state.get("insert")
            if insert_index:
                text_widget.mark_set(tk.INSERT, str(insert_index))
        except Exception:
            pass

        try:
            text_widget.xview_moveto(float(state.get("x_first", 0.0) or 0.0))
        except Exception:
            pass

        try:
            text_widget.yview_moveto(float(state.get("y_first", 0.0) or 0.0))
        except Exception:
            pass

        try:
            text_widget.tag_remove(tk.SEL, "1.0", tk.END)
            sel_first = state.get("sel_first")
            sel_last = state.get("sel_last")
            if sel_first and sel_last:
                text_widget.tag_add(tk.SEL, str(sel_first), str(sel_last))
        except Exception:
            pass

    def _append_to_step4_process_console(self, *args, **kwargs):
        return z4_ui_runtime._append_to_step4_process_console(self, *args, **kwargs)

    def _append_train_log(self, *args, **kwargs):
        return z4_ui_runtime._append_train_log(self, *args, **kwargs)

    def _append_ranking_log(self, *args, **kwargs):
        return z4_ui_runtime._append_ranking_log(self, *args, **kwargs)

    @staticmethod
    def _sanitize_training_text(message) -> str:
        text = "" if message is None else str(message)
        if not text:
            return ""
        suspicious_tokens = ("Ä", "Å", "Ĺ", "Ă", "â", "Ђ", "€", "™", "�")
        if any(token in text for token in suspicious_tokens):
            candidates = [text]
            for source_encoding in ("latin1", "cp1252"):
                try:
                    repaired = text.encode(source_encoding, errors="ignore").decode("utf-8", errors="ignore")
                    if repaired:
                        candidates.append(repaired)
                except Exception:
                    pass

            def score(candidate: str) -> tuple[int, int]:
                suspicious = sum(candidate.count(token) for token in suspicious_tokens)
                replacement = candidate.count("�") + candidate.count("?")
                return (suspicious + replacement, -len(candidate))

            try:
                text = min(candidates, key=score)
            except Exception:
                pass

        replacements = {
            "â": "–",
            "â": "—",
            "â¦": "…",
            "â": "„",
            "â": "\"",
            "â": "\"",
            "â": "'",
            "â": "'",
            "Â ": " ",
        }
        for broken, fixed in replacements.items():
            if broken in text:
                text = text.replace(broken, fixed)
        return text

    def _set_training_widget_text(self, *args, **kwargs):
        return z4_ui_runtime._set_training_widget_text(self, *args, **kwargs)

    def _set_training_stringvar_text(self, *args, **kwargs):
        return z4_ui_runtime._set_training_stringvar_text(self, *args, **kwargs)

    def _start_ranking_watchdog(self, *args, **kwargs):
        return z4_ui_runtime._start_ranking_watchdog(self, *args, **kwargs)

    def _touch_ranking_watchdog(self, *args, **kwargs):
        return z4_ui_runtime._touch_ranking_watchdog(self, *args, **kwargs)

    def _stop_ranking_watchdog(self, *args, **kwargs):
        return z4_ui_runtime._stop_ranking_watchdog(self, *args, **kwargs)

    def _poll_ranking_watchdog(self, *args, **kwargs):
        return z4_ui_runtime._poll_ranking_watchdog(self, *args, **kwargs)

    def _get_active_step4_operation_label(self, *args, **kwargs):
        return z4_ui_runtime._get_active_step4_operation_label(self, *args, **kwargs)

    def _step4_has_active_operation(self, *args, **kwargs):
        return z4_ui_runtime._step4_has_active_operation(self, *args, **kwargs)

    def _begin_step4_operation(self, *args, **kwargs):
        return z4_ui_runtime._begin_step4_operation(self, *args, **kwargs)

    def _end_step4_operation(self, *args, **kwargs):
        return z4_ui_runtime._end_step4_operation(self, *args, **kwargs)

    def _set_ranking_ui_state(self, *args, **kwargs):
        return z4_ui_runtime._set_ranking_ui_state(self, *args, **kwargs)

    def _attach_training_log_handlers(self, *args, **kwargs):
        return z4_ui_runtime._attach_training_log_handlers(self, *args, **kwargs)

    @staticmethod
    def _metric_float(*args, **kwargs):
        return z4_training_metrics._metric_float(*args, **kwargs)
    @staticmethod
    def _describe_metric_band(*args, **kwargs):
        return z4_training_metrics._describe_metric_band(*args, **kwargs)
    def _build_training_metric_reference_text(self, *args, **kwargs):
        return z4_training_metrics._build_training_metric_reference_text(self, *args, **kwargs)
    def _refresh_training_metric_reference(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_metric_reference(self, *args, **kwargs)
    def _set_training_metric_interpretation(self, *args, **kwargs):
        return z4_training_metrics._set_training_metric_interpretation(self, *args, **kwargs)
    def _metric_value_from_epoch_row(self, *args, **kwargs):
        return z4_training_metrics._metric_value_from_epoch_row(self, *args, **kwargs)
    def _format_best_epoch_metric_piece(self, *args, **kwargs):
        return z4_training_metrics._format_best_epoch_metric_piece(self, *args, **kwargs)
    def _build_training_best_epoch_summary(self, *args, **kwargs):
        return z4_training_metrics._build_training_best_epoch_summary(self, *args, **kwargs)
    def _build_training_metric_interpretation(self, *args, **kwargs):
        return z4_training_metrics._build_training_metric_interpretation(self, *args, **kwargs)
    @staticmethod
    def _shorten_training_text(*args, **kwargs):
        return z4_training_metrics._shorten_training_text(*args, **kwargs)
    @staticmethod
    def _format_training_metric_name(*args, **kwargs):
        return z4_training_metrics._format_training_metric_name(*args, **kwargs)
    def _resolve_training_metric_profile(self, *args, **kwargs):
        return z4_training_metrics._resolve_training_metric_profile(self, *args, **kwargs)
    def _format_training_metric_value(self, *args, **kwargs):
        return z4_training_metrics._format_training_metric_value(self, *args, **kwargs)
    def _format_training_metric_band(self, *args, **kwargs):
        return z4_training_metrics._format_training_metric_band(self, *args, **kwargs)
    def _iter_training_metric_keys(self, *args, **kwargs):
        return z4_training_metrics._iter_training_metric_keys(self, *args, **kwargs)
    def _build_training_metric_rows(self, *args, **kwargs):
        return z4_training_metrics._build_training_metric_rows(self, *args, **kwargs)
    def _build_training_run_detail_rows(self, *args, **kwargs):
        return z4_training_metrics._build_training_run_detail_rows(self, *args, **kwargs)
    def _build_training_run_metric_rows(self, *args, **kwargs):
        return z4_training_metrics._build_training_run_metric_rows(self, *args, **kwargs)
    def _set_train_live_metrics(self, *args, **kwargs):
        return z4_training_metrics._set_train_live_metrics(self, *args, **kwargs)
    def _configure_training_status_table_tags(self, *args, **kwargs):
        return z4_training_metrics._configure_training_status_table_tags(self, *args, **kwargs)
    def _set_training_resource_sample(self, *args, **kwargs):
        return z4_training_metrics._set_training_resource_sample(self, *args, **kwargs)
    def _set_training_resource_report(self, *args, **kwargs):
        return z4_training_metrics._set_training_resource_report(self, *args, **kwargs)
    def _build_training_cockpit_summary(self, *args, **kwargs):
        return z4_training_metrics._build_training_cockpit_summary(self, *args, **kwargs)
    def _refresh_training_cockpit(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_cockpit(self, *args, **kwargs)
    def _build_training_start_gate_message(self, *args, **kwargs):
        return z4_training_metrics._build_training_start_gate_message(self, *args, **kwargs)
    def _refresh_training_start_gate(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_start_gate(self, *args, **kwargs)
    def _build_training_recommendation_rows(self, *args, **kwargs):
        return z4_training_metrics._build_training_recommendation_rows(self, *args, **kwargs)
    def _refresh_training_recommendation_table(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_recommendation_table(self, *args, **kwargs)
    def _apply_training_recommendation_table_theme(self, *args, **kwargs):
        return z4_training_metrics._apply_training_recommendation_table_theme(self, *args, **kwargs)
    def _apply_training_device_recommendation(self, *args, **kwargs):
        return z4_training_metrics._apply_training_device_recommendation(self, *args, **kwargs)
    def _on_training_base_model_value_write(self, *args, **kwargs):
        return z4_training_metrics._on_training_base_model_value_write(self, *args, **kwargs)
    def _schedule_step4_deferred_model_refresh(self, *args, **kwargs):
        return z4_training_metrics._schedule_step4_deferred_model_refresh(self, *args, **kwargs)
    def _apply_training_recommended_start_params(self, *args, **kwargs):
        return z4_training_metrics._apply_training_recommended_start_params(self, *args, **kwargs)
    def _safe_training_numeric_var_value(self, *args, **kwargs):
        return z4_training_metrics._safe_training_numeric_var_value(self, *args, **kwargs)
    def _safe_training_int_value(self, *args, **kwargs):
        return z4_training_metrics._safe_training_int_value(self, *args, **kwargs)
    def _safe_training_float_value(self, *args, **kwargs):
        return z4_training_metrics._safe_training_float_value(self, *args, **kwargs)
    def _resolve_training_dataset_yaml_path(self, *args, **kwargs):
        return z4_training_metrics._resolve_training_dataset_yaml_path(self, *args, **kwargs)
    def _validate_active_training_source_for_pz2(self, *args, **kwargs):
        return z4_training_metrics._validate_active_training_source_for_pz2(self, *args, **kwargs)
    def _looks_like_char_classification_dataset(self, *args, **kwargs):
        return z4_training_metrics._looks_like_char_classification_dataset(self, *args, **kwargs)
    def _char_classification_dataset_message(self, *args, **kwargs):
        return z4_training_metrics._char_classification_dataset_message(self, *args, **kwargs)
    def _is_training_configuration_ready(self, *args, **kwargs):
        return z4_training_metrics._is_training_configuration_ready(self, *args, **kwargs)
    def _validate_training_base_model_target_compatibility(self, *args, **kwargs):
        return z4_training_metrics._validate_training_base_model_target_compatibility(self, *args, **kwargs)
    def _refresh_training_start_state(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_start_state(self, *args, **kwargs)
    def _get_pinned_step4_result_state(self, *args, **kwargs):
        return z4_training_metrics._get_pinned_step4_result_state(self, *args, **kwargs)
    def _refresh_step4_pinned_result_ui(self, *args, **kwargs):
        return z4_training_metrics._refresh_step4_pinned_result_ui(self, *args, **kwargs)
    def _clear_pinned_step4_result(self, *args, **kwargs):
        return z4_training_metrics._clear_pinned_step4_result(self, *args, **kwargs)
    def _set_step4_fine_tune_parent_state(self, *args, **kwargs):
        return z4_training_metrics._set_step4_fine_tune_parent_state(self, *args, **kwargs)
    def _clear_step4_fine_tune_parent_state(self, *args, **kwargs):
        return z4_training_metrics._clear_step4_fine_tune_parent_state(self, *args, **kwargs)
    def _resolve_step4_fine_tune_parent_run(self, *args, **kwargs):
        return z4_training_metrics._resolve_step4_fine_tune_parent_run(self, *args, **kwargs)
    def _sync_step4_fine_tune_parent_selection(self, *args, **kwargs):
        return z4_training_metrics._sync_step4_fine_tune_parent_selection(self, *args, **kwargs)
    def _refresh_training_base_model_identity_ui(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_base_model_identity_ui(self, *args, **kwargs)
    def _resolve_selected_training_base_model_display(self, *args, **kwargs):
        return z4_training_metrics._resolve_selected_training_base_model_display(self, *args, **kwargs)
    def _resolve_selected_training_base_model_path(self, *args, **kwargs):
        return z4_training_metrics._resolve_selected_training_base_model_path(self, *args, **kwargs)
    def _resolve_selected_training_base_model_source_run(self, *args, **kwargs):
        return z4_training_metrics._resolve_selected_training_base_model_source_run(self, *args, **kwargs)
    def _selected_training_base_model_nonfinal_run(self, *args, **kwargs):
        return z4_training_metrics._selected_training_base_model_nonfinal_run(self, *args, **kwargs)
    def _resolve_selected_training_base_model_inspection_path(self, *args, **kwargs):
        return z4_training_metrics._resolve_selected_training_base_model_inspection_path(self, *args, **kwargs)
    def _resolve_selected_training_base_model_info(self, *args, **kwargs):
        return z4_training_metrics._resolve_selected_training_base_model_info(self, *args, **kwargs)
    def _build_selected_training_base_model_identity_lines(self, *args, **kwargs):
        return z4_training_metrics._build_selected_training_base_model_identity_lines(self, *args, **kwargs)
    def _build_training_recommendation_model_text(self, *args, **kwargs):
        return z4_training_metrics._build_training_recommendation_model_text(self, *args, **kwargs)
    @staticmethod
    def _format_training_model_run_label(*args, **kwargs):
        return z4_training_metrics._format_training_model_run_label(*args, **kwargs)
    @staticmethod
    def _format_training_model_size_label(*args, **kwargs):
        return z4_training_metrics._format_training_model_size_label(*args, **kwargs)
    def _build_training_model_summary_value(self, *args, **kwargs):
        return z4_training_metrics._build_training_model_summary_value(self, *args, **kwargs)
    @staticmethod
    def _resolve_training_model_version_from_info(*args, **kwargs):
        return z4_training_metrics._resolve_training_model_version_from_info(*args, **kwargs)
    def _resolve_training_model_size_from_info(self, *args, **kwargs):
        return z4_training_metrics._resolve_training_model_size_from_info(self, *args, **kwargs)
    @staticmethod
    def _get_pose_model_memory_bucket(*args, **kwargs):
        return z4_training_metrics._get_pose_model_memory_bucket(*args, **kwargs)
    def _get_training_gpu_capacity_block_reason(self, *args, **kwargs):
        return z4_training_metrics._get_training_gpu_capacity_block_reason(self, *args, **kwargs)
    def _resolve_training_run_from_model_path(self, *args, **kwargs):
        return z4_training_metrics._resolve_training_run_from_model_path(self, *args, **kwargs)
    def _build_selected_training_base_model_origin_rows(self, *args, **kwargs):
        return z4_training_metrics._build_selected_training_base_model_origin_rows(self, *args, **kwargs)
    def _append_training_execution_summary_row_widget(self, *args, **kwargs):
        return z4_training_metrics._append_training_execution_summary_row_widget(self, *args, **kwargs)
    def _ensure_training_execution_summary_row_capacity(self, *args, **kwargs):
        return z4_training_metrics._ensure_training_execution_summary_row_capacity(self, *args, **kwargs)
    def _build_training_execution_summary_rows(self, *args, **kwargs):
        return z4_training_metrics._build_training_execution_summary_rows(self, *args, **kwargs)
    def _refresh_training_execution_summary(self, *args, **kwargs):
        return z4_training_metrics._refresh_training_execution_summary(self, *args, **kwargs)
    @staticmethod
    def _terminal_metric_tag(*args, **kwargs):
        return z4_training_metrics._terminal_metric_tag(*args, **kwargs)
    @staticmethod
    def _format_terminal_table_line(*args, **kwargs):
        return z4_training_metrics._format_terminal_table_line(*args, **kwargs)
    def _build_training_metric_terminal_entries(self, *args, **kwargs):
        return z4_training_metrics._build_training_metric_terminal_entries(self, *args, **kwargs)
    def _append_training_metric_table_to_global(self, *args, **kwargs):
        return z4_training_metrics._append_training_metric_table_to_global(self, *args, **kwargs)
    def _set_history_run_tables(self, *args, **kwargs):
        return z4_training_metrics._set_history_run_tables(self, *args, **kwargs)
    def _get_datasets_base_dir(self, *args, **kwargs):
        return z4_dataset_sources._get_datasets_base_dir(self, *args, **kwargs)
    def _get_manual_plate_stage_dir(self, *args, **kwargs):
        return z4_dataset_sources._get_manual_plate_stage_dir(self, *args, **kwargs)
    def _iter_dataset_search_roots(self, *args, **kwargs):
        return z4_dataset_sources._iter_dataset_search_roots(self, *args, **kwargs)
    def _find_dataset_source_candidates(self, *args, **kwargs):
        return z4_dataset_sources._find_dataset_source_candidates(self, *args, **kwargs)
    def _find_ready_dataset_candidates(self, *args, **kwargs):
        return z4_dataset_sources._find_ready_dataset_candidates(self, *args, **kwargs)
    def _get_free_dataset_variant_choices(self, *args, **kwargs):
        return z4_dataset_sources._get_free_dataset_variant_choices(self, *args, **kwargs)
    def _refresh_dataset_variant_choices(self, *args, **kwargs):
        return z4_dataset_sources._refresh_dataset_variant_choices(self, *args, **kwargs)
    def _normalize_dataset_variant_root(self, *args, **kwargs):
        return z4_dataset_sources._normalize_dataset_variant_root(self, *args, **kwargs)
    def _is_free_training_dataset_variant_selected(self, *args, **kwargs):
        return z4_dataset_sources._is_free_training_dataset_variant_selected(self, *args, **kwargs)
    def _sync_dataset_variant_selection(self, *args, **kwargs):
        return z4_dataset_sources._sync_dataset_variant_selection(self, *args, **kwargs)
    def _on_dataset_variant_selected(self, *args, **kwargs):
        return z4_dataset_sources._on_dataset_variant_selected(self, *args, **kwargs)
    def _get_dataset_split_image_counts(self, *args, **kwargs):
        return z4_dataset_sources._get_dataset_split_image_counts(self, *args, **kwargs)
    @staticmethod
    def _format_training_source_provenance(*args, **kwargs):
        return z4_dataset_sources._format_training_source_provenance(*args, **kwargs)
    def _build_step4_dataset_training_source(self, *args, **kwargs):
        return z4_dataset_sources._build_step4_dataset_training_source(self, *args, **kwargs)
    def _resolve_step4_dataset_summary_source(self, *args, **kwargs):
        return z4_dataset_sources._resolve_step4_dataset_summary_source(self, *args, **kwargs)
    def _refresh_step4_dataset_summary_table(self, *args, **kwargs):
        return z4_dataset_sources._refresh_step4_dataset_summary_table(self, *args, **kwargs)
    def _schedule_step4_dataset_summary_refresh(self, *args, **kwargs):
        return z4_dataset_sources._schedule_step4_dataset_summary_refresh(self, *args, **kwargs)
    def _draw_step4_dataset_split_bar(self, *args, **kwargs):
        return z4_dataset_sources._draw_step4_dataset_split_bar(self, *args, **kwargs)
    @staticmethod
    def _median_int(*args, **kwargs):
        return z4_dataset_sources._median_int(*args, **kwargs)
    @staticmethod
    def _nearest_training_imgsz(*args, **kwargs):
        return z4_dataset_sources._nearest_training_imgsz(*args, **kwargs)
    @staticmethod
    def _parse_training_model_params_millions(*args, **kwargs):
        return z4_dataset_sources._parse_training_model_params_millions(*args, **kwargs)
    def _normalize_training_model_catalog_key(self, *args, **kwargs):
        return z4_dataset_sources._normalize_training_model_catalog_key(self, *args, **kwargs)
    @staticmethod
    def _infer_training_model_bucket(*args, **kwargs):
        return z4_dataset_sources._infer_training_model_bucket(*args, **kwargs)
    @staticmethod
    def _infer_training_model_version_size_from_text(*args, **kwargs):
        return z4_dataset_sources._infer_training_model_version_size_from_text(*args, **kwargs)
    @staticmethod
    def _format_training_model_version_label(*args, **kwargs):
        return z4_dataset_sources._format_training_model_version_label(*args, **kwargs)
    def _resolve_selected_training_base_model_profile(self, *args, **kwargs):
        return z4_dataset_sources._resolve_selected_training_base_model_profile(self, *args, **kwargs)
    def _get_training_dataset_profile(self, *args, **kwargs):
        return z4_dataset_sources._get_training_dataset_profile(self, *args, **kwargs)
    def _open_character_class_distribution_dialog(self, *args, **kwargs):
        return z4_character_balance.open_character_class_distribution_dialog(self, *args, **kwargs)
    def _get_campaign_plate_builder_source(self, *args, **kwargs):
        return z4_dataset_sources._get_campaign_plate_builder_source(self, *args, **kwargs)
    def _load_step4_training_ui_state(self, *args, **kwargs):
        return z4_dataset_sources._load_step4_training_ui_state(self, *args, **kwargs)
    def _save_step4_training_ui_state(self, *args, **kwargs):
        return z4_dataset_sources._save_step4_training_ui_state(self, *args, **kwargs)
    def _get_preferred_plate_pose_base_model(self, *args, **kwargs):
        return z4_dataset_sources._get_preferred_plate_pose_base_model(self, *args, **kwargs)
    def _resolve_saved_step4_training_model_selection(self, *args, **kwargs):
        return z4_dataset_sources._resolve_saved_step4_training_model_selection(self, *args, **kwargs)
    def _apply_saved_step4_training_model_selection(self, *args, **kwargs):
        return z4_dataset_sources._apply_saved_step4_training_model_selection(self, *args, **kwargs)
    def _remember_current_step4_training_model_selection(self, *args, **kwargs):
        return z4_dataset_sources._remember_current_step4_training_model_selection(self, *args, **kwargs)
    def _resolve_campaign_plate_ready_dataset(self, *args, **kwargs):
        return z4_dataset_sources._resolve_campaign_plate_ready_dataset(self, *args, **kwargs)
    def _get_runs_base_dir(self) -> Path:
        """Zwraca bazowy katalog runów treningowych dla aktywnego projektu albo globalny fallback."""
        if self._campaign_runs_dir:
            return Path(self._campaign_runs_dir)
        target = getattr(self, "_step4_dataset_mode", getattr(self, "_campaign_training_target", "char"))
        return Path(CONFIG.get_training_runs_dir(target))

    def _rebind_free_mode_training_storage(self, target: str | None = None, reload_history: bool = True):
        if CAMPAIGN.get_active_project_name():
            return

        normalized_target = CONFIG.normalize_task_target(target or getattr(self, "_step4_dataset_mode", "char"))

        runs_dir = Path(CONFIG.get_training_runs_dir(normalized_target))
        history_dir = getattr(getattr(self, "history", None), "history_dir", None)
        same_storage = bool(history_dir and Path(history_dir).resolve() == runs_dir.resolve())
        if not same_storage:
            self.history = TrainingHistory(history_dir=runs_dir)
            self.trainer = YOLOPoseTrainer(history=self.history)
            self._bind_trainer_callbacks()

        if reload_history and hasattr(self, "tree"):
            try:
                self._load_history()
            except Exception:
                pass
        if reload_history and hasattr(self, "rank_tree"):
            try:
                if normalized_target in {"plate", "char"} and self._is_ranking_tab_active():
                    self._load_ranking()
            except Exception:
                pass

    def _format_workspace_relative_path(self, path_like) -> str:
        try:
            path = Path(path_like).resolve()
            workspace = Path(CONFIG.WORKSPACE_DIR).resolve()
            rel = path.relative_to(workspace)
            rel_posix = PurePosixPath(rel.as_posix())
            return str(PurePosixPath("Workspace") / rel_posix)
        except Exception:
            try:
                return Path(path_like).as_posix()
            except Exception:
                return str(path_like)

    def _get_training_dataset_hint_text(self) -> str:
        campaign_active = bool(CAMPAIGN.get_active_project_name())
        selected_target = self._get_selected_training_target()
        workspace_dir = self._format_workspace_relative_path(CONFIG.get_datasets_dir(selected_target))

        if campaign_active:
            return f"Katalog splitów: {workspace_dir}"

        if selected_target == "plate":
            source_hint = (
                "Splity YOLO Pose do treningu tablic."
            )
        else:
            source_hint = (
                "Splity YOLO Detect do treningu znaków. Dataset OCR/klasyfikacyjny nie jest wejściem treningu YOLO."
            )

        return (
            f"{source_hint}\n"
            f"Katalog splitów: {workspace_dir}"
        )

    def _get_training_dataset_quality_thresholds(self, target: str | None = None) -> dict:
        normalized = CONFIG.normalize_task_target(target or self._get_selected_training_target())
        if normalized == "plate":
            return {
                "object_label": "anotacje tablic",
                "min_images": int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10),
                "min_train": 8,
                "min_val": 1,
                "min_objects": int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10),
                "recommended_images": int(getattr(CONFIG, "YOLO_POSE_AVERAGE_PLATES", 50) or 50),
                "recommended_train": 40,
                "recommended_val": 5,
                "recommended_objects": int(getattr(CONFIG, "YOLO_POSE_GOOD_PLATES", 200) or 200),
            }
        return {
            "object_label": "boxy znaków",
            "min_images": int(getattr(CONFIG, "YOLO_CHAR_AVERAGE_PLATES", 10) or 10),
            "min_train": 8,
            "min_val": 1,
            "min_objects": int(getattr(CONFIG, "YOLO_CHAR_AVERAGE_BOXES", 100) or 100),
            "recommended_images": int(getattr(CONFIG, "YOLO_CHAR_GOOD_PLATES", 50) or 50),
            "recommended_train": 40,
            "recommended_val": 5,
            "recommended_objects": int(getattr(CONFIG, "YOLO_CHAR_GOOD_BOXES", 1000) or 1000),
        }

    @staticmethod
    def _format_dataset_quality_need(current: int, expected: int) -> str:
        current = max(0, int(current or 0))
        expected = max(0, int(expected or 0))
        if expected <= 0:
            return str(current)
        return f"{current}/{expected}"

    def _build_training_dataset_quality_summary(self) -> dict:
        target = self._get_selected_training_target()
        normalized_target = CONFIG.normalize_task_target(target)
        thresholds = self._get_training_dataset_quality_thresholds(target)
        dataset_yaml = self._resolve_training_dataset_yaml_path()
        if dataset_yaml is None:
            return {
                "visible": True,
                "title": "Podsumowanie materiału",
                "tone": "muted",
                "status": "Wybierz wariant",
                "rows": [
                    ("Status", "Nie wybrano aktywnego wariantu splitu."),
                    ("Co zrobić", "Wybierz wariant z listy albo wróć do PZ1 i utwórz nowy split."),
                    ("Po wyborze", "Pokażemy liczebność train / val / test oraz ocenę próbki."),
                ],
            }

        profile = self._get_training_dataset_profile(dataset_yaml)
        if profile.get("pending") or profile.get("error"):
            message = (
                "Liczę obrazy i anotacje wybranego wariantu w tle. Możesz dalej korzystać z karty."
                if profile.get("pending") else "Nie udało się odczytać podsumowania wybranego datasetu."
            )
            return {
                "visible": True, "title": "Podsumowanie materiału", "tone": "muted",
                "status": "Analiza w toku" if profile.get("pending") else "Brak podsumowania",
                "rows": [("Status", message)],
            }
        train_images = int(profile.get("train_images", 0) or 0)
        val_images = int(profile.get("val_images", 0) or 0)
        test_images = int(profile.get("test_images", 0) or 0)
        total_images = int(profile.get("total_images", 0) or 0)
        total_objects = int(profile.get("total_objects", 0) or 0)
        created_at = str(profile.get("created_at") or "").strip()

        min_ready = (
            total_images >= int(thresholds["min_images"])
            and train_images >= int(thresholds["min_train"])
            and val_images >= int(thresholds["min_val"])
            and total_objects >= int(thresholds["min_objects"])
        )
        recommended_ready = (
            total_images >= int(thresholds["recommended_images"])
            and train_images >= int(thresholds["recommended_train"])
            and val_images >= int(thresholds["recommended_val"])
            and total_objects >= int(thresholds["recommended_objects"])
        )

        if normalized_target == "plate":
            quality_info = CONFIG.describe_yolo_pose_dataset_quality(total_objects)
        else:
            quality_info = CONFIG.describe_yolo_char_dataset_quality(
                perfect_plates=total_images,
                char_boxes=total_objects,
            )
        quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY").strip()
        if recommended_ready:
            tone = "success"
            status = f"{quality_label}. Materiał wygląda sensownie do treningu."
            next_step = "Możesz rozpocząć trening albo wrócić do PZ1, jeśli chcesz przygotować inny wariant."
        elif min_ready:
            tone = "warning"
            status = f"{quality_label}. Wystarczy do testowego startu, ale warto powiększyć zbiór."
            next_step = "Trening jest możliwy, jednak lepszy efekt da większa próbka lub augmentacja."
        else:
            tone = "danger"
            status = f"{quality_label}. Dataset jest za mały na sensowny trening."
            next_step = "Wróć do PZ1 i przygotuj większy wariant albo zwiększ pulę danych."

        if normalized_target == "plate":
            sample_text = f"{total_images} obrazów wejściowych"
            object_row = ("Tablice w anotacjach", f"{total_objects} tablic")
        else:
            sample_text = f"{total_images} wyodrębnionych tablic"
            object_row = ("Boxy znaków", f"{total_objects} boxów znaków")

        rows = [
            ("Próbka", f"{sample_text} | {object_row[1]}"),
            ("Split", f"train {train_images} | val {val_images} | test {test_images}"),
            ("Utworzono", created_at or "-"),
            ("Ocena", status),
            ("Sugestia", next_step),
        ]
        return {
            "visible": True,
            "title": "Podsumowanie materiału",
            "tone": tone,
            "status": status,
            "rows": rows,
        }

    def _get_preferred_models_dir(self, target: str | None = None) -> Path:
        normalized_target = CONFIG.normalize_task_target(target or self._get_selected_training_target())

        preferred = CONFIG.get_trained_models_dir(normalized_target)
        return preferred if preferred.exists() else Path(CONFIG.DEFAULT_MODELS_DIR)

    def _pick_base_custom_model(self):
        previous_value = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        self._pick_file(
            self.base_custom_var,
            "*.pt",
            self._get_preferred_models_dir(self._get_selected_training_target())
        )
        selected_value = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        if selected_value and selected_value != previous_value:
            try:
                self._clear_step4_fine_tune_parent_state()
            except Exception:
                pass
            selection_ok, _selection_message = self._validate_training_base_model_target_compatibility(
                target=self._get_selected_training_target(),
                show_dialog=True,
            )
            if not selection_ok:
                self.base_custom_var.set(previous_value)
        try:
            self._refresh_training_start_state()
        except Exception:
            pass

    def _get_selected_training_target(self) -> str:
        if CAMPAIGN.get_active_project_name():
            target = self.get_campaign_training_target()
            return target if target in ("char", "plate") else "char"

        target = CONFIG.normalize_task_target(getattr(self, "_step4_dataset_mode", "char"))
        return target if target in ("char", "plate") else "char"

    def _get_locked_campaign_training_target(self) -> str | None:
        if not CAMPAIGN.get_active_project_name():
            return None

        target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        return target if target in ("char", "plate") else None

    @staticmethod
    def _is_finish_eligible_training_status(status: str | None) -> bool:
        value = str(status or "").strip().lower()
        return value == TrainingStatus.COMPLETED.value

    def _get_campaign_history_for_finish_recovery(self) -> TrainingHistory | None:
        if not CAMPAIGN.get_active_project_name():
            return None

        runs_dir = CAMPAIGN.get_dir("runs")
        if runs_dir is None:
            return None

        try:
            runs_path = Path(runs_dir).resolve()
        except Exception:
            runs_path = Path(runs_dir)

        try:
            if self.history is not None and str(Path(getattr(self.history, "history_dir", "")).resolve()) == str(runs_path):
                return self.history
        except Exception:
            pass

        try:
            return TrainingHistory(history_dir=runs_path)
        except Exception:
            return None

    def _validate_campaign_finish_run(self, *args, **kwargs):
        return z4_campaign_state._validate_campaign_finish_run(self, *args, **kwargs)


    def _recover_campaign_finish_state_from_history(self, *args, **kwargs):
        return z4_campaign_state._recover_campaign_finish_state_from_history(self, *args, **kwargs)


    def get_campaign_step4_finish_state(self, *args, **kwargs):
        return z4_campaign_state.get_campaign_step4_finish_state(self, *args, **kwargs)


    def _build_training_dataset_validation_message(self, *args, **kwargs):
        return z4_campaign_state._build_training_dataset_validation_message(self, *args, **kwargs)


    def _get_pose_dataset_size_warning(self, *args, **kwargs):
        return z4_campaign_state._get_pose_dataset_size_warning(self, *args, **kwargs)


    @staticmethod
    def _extract_dataset_class_names(cfg: dict | None) -> list[str]:
        if not isinstance(cfg, dict):
            return []

        names = cfg.get("names", [])
        if isinstance(names, dict):
            try:
                ordered_keys = sorted(
                    names.keys(),
                    key=lambda item: int(item) if str(item).isdigit() else str(item)
                )
                names = [names[key] for key in ordered_keys]
            except Exception:
                names = list(names.values())
        elif not isinstance(names, (list, tuple)):
            names = []

        return [str(name).strip() for name in names if str(name).strip()]

    @staticmethod
    def _looks_like_character_alphabet(class_names: list[str]) -> bool:
        if len(class_names) < 8:
            return False

        import string

        allowed = set(string.ascii_uppercase + string.digits)
        for name in class_names:
            token = str(name).strip().upper()
            if len(token) != 1 or token not in allowed:
                return False

        return True

    def _infer_detect_dataset_target(self, cfg: dict | None, path_like) -> str:
        path_str = str(path_like or "").replace("\\", "/").lower()
        vehicle_keywords = (
            "vehicle", "vehicles", "pojazd", "pojazdy", "car", "cars", "truck", "trucks",
            "bus", "buses", "motorcycle", "motorbike", "bike", "van", "pickup", "suv",
            "samochod", "samochody"
        )
        char_keywords = (
            "char", "chars", "character", "characters", "znak", "znaki", "litera", "litery"
        )

        class_names = self._extract_dataset_class_names(cfg)
        joined_names = " ".join(name.lower() for name in class_names)

        if any(keyword in path_str for keyword in vehicle_keywords):
            return "vehicle"
        if any(keyword in joined_names for keyword in vehicle_keywords):
            return "vehicle"
        plate_class_names = {"plate", "plates", "license_plate", "license-plate", "tablica", "tablice"}
        if class_names and all(str(name).strip().lower() in plate_class_names for name in class_names):
            return "plate"
        if self._looks_like_character_alphabet(class_names):
            return "char"
        if any(keyword in path_str for keyword in char_keywords):
            return "char"

        return "char"

    def _infer_dataset_target(self, dataset_value: str | None = None) -> str | None:
        if dataset_value is None:
            var = getattr(self, "dataset_var", None)
            dataset_value = var.get() if var is not None else ""
        dataset_value = str(dataset_value).strip()
        if not dataset_value:
            return None

        dataset_path = Path(dataset_value)
        yaml_path = dataset_path / "data.yaml" if dataset_path.is_dir() else dataset_path
        cfg = None

        if yaml_path.exists():
            try:
                cfg = safe_load_yaml(yaml_path)
            except Exception:
                cfg = None

        if isinstance(cfg, dict) and "kpt_shape" in cfg:
            return "plate"

        return self._infer_detect_dataset_target(cfg, yaml_path if yaml_path.exists() else dataset_path)

    @staticmethod
    def _format_training_target_label(target: str) -> str:
        normalized = CONFIG.normalize_task_target(target)
        labels = {
            "plate": "tablice (YOLO Pose)",
            "char": "znaki tablic (YOLO Detect)",
            "vehicle": "pojazdy (YOLO Detect)",
        }
        return labels.get(normalized, "znaki tablic (YOLO Detect)")

    @staticmethod
    def _format_history_run_target_label(target: str) -> str:
        normalized = CONFIG.normalize_task_target(target)
        labels = {
            "plate": "Tablice",
            "char": "Znaki",
            "vehicle": "Pojazdy",
        }
        return labels.get(normalized, "Inny")

    def _get_training_scope_hint_text(self) -> str:
        campaign_active = bool(CAMPAIGN.get_active_project_name())
        selected_target = self._get_selected_training_target()
        selected_label = self._format_training_target_label(selected_target)
        try:
            active_source = self._resolve_step4_dataset_summary_source()
        except Exception:
            active_source = None

        if campaign_active:
            dataset_value = str(
                getattr(active_source, "dataset_dir", "")
                or getattr(active_source, "yaml_path", "")
                or getattr(self, "dataset_var", tk.StringVar()).get()
                or ""
            ).strip()
            dataset_counts = self._get_dataset_split_image_counts(dataset_value)
            total_images = int(dataset_counts.get("total", 0) or 0)
            if total_images > 0:
                return (
                    f"Tor kampanii: {selected_label}. "
                    f"train={dataset_counts.get('train', 0)}, "
                    f"val={dataset_counts.get('val', 0)}, "
                    f"test={dataset_counts.get('test', 0)}, "
                    f"razem={total_images}."
                )
            return f"Tor kampanii: {selected_label}."

        dataset_value = (
            str(getattr(active_source, "dataset_dir", "") or getattr(active_source, "yaml_path", "") or "").strip()
            if active_source is not None
            else ""
        )
        if not dataset_value:
            dataset_var = getattr(self, "dataset_var", None)
            dataset_value = dataset_var.get() if dataset_var is not None else ""
        inferred_target = self._infer_dataset_target(dataset_value)
        if inferred_target and inferred_target != selected_target:
            inferred_label = self._format_training_target_label(inferred_target)
            return (
                f"Tor: {selected_label}. Dataset wygląda na: {inferred_label}."
            )

        return f"Tor: {selected_label}."

    def _get_base_model_choices_for_mode(self, mode: str | None = None) -> list[str]:
        normalized = CONFIG.normalize_task_target(mode or self._get_selected_training_target())
        custom_label = self._get_custom_base_model_label()
        if normalized == "plate":
            return list(AVAILABLE_POSE_MODELS.keys()) + [custom_label]
        return list(AVAILABLE_DETECT_MODELS.keys()) + [custom_label]

    def _get_custom_base_model_label(self) -> str:
        return "Własny plik .pt"

    def _is_custom_base_model_key(self, value: str | None) -> bool:
        normalized = str(value or "").strip()
        return normalized in {"Custom", self._get_custom_base_model_label()}

    def _normalize_base_model_choice_for_ui(self, value: str | None) -> str:
        normalized = str(value or "").strip()
        return self._get_custom_base_model_label() if self._is_custom_base_model_key(normalized) else normalized

    def _get_default_base_model_for_mode(self, mode: str | None = None) -> str:
        choices = self._get_base_model_choices_for_mode(mode)
        for choice in choices:
            if not self._is_custom_base_model_key(choice):
                return choice
        return self._get_custom_base_model_label()

    def _is_pose_base_model(self, base_key: str, base_model: str) -> bool:
        base_key = str(base_key or "").strip()
        base_model = str(base_model or "").strip()

        if base_key in AVAILABLE_POSE_MODELS:
            return True
        if base_key in AVAILABLE_DETECT_MODELS:
            return False

        model_text = base_model.lower()
        if "pose" in model_text:
            return True

        try:
            model_path = Path(base_model)
        except Exception:
            model_path = None

        if model_path is not None and model_path.suffix.lower() == ".pt" and model_path.exists():
            ok, _message, info = validate_model_file(model_path)
            if ok:
                task = str(info.get("task") or "").strip().lower()
                inferred_type = str(info.get("type") or "").strip().lower()
                if bool(info.get("keypoints")) or task == "pose" or inferred_type == "pose":
                    return True

        return False

    def _refresh_base_model_choices(self):
        combo = getattr(self, "base_combo", None)
        var = getattr(self, "base_model_var", None)
        if combo is None or var is None:
            return

        choices = self._get_base_model_choices_for_mode()
        current = self._normalize_base_model_choice_for_ui(str(var.get() or "").strip())

        try:
            combo.configure(values=choices)
        except Exception:
            pass

        if current not in choices:
            var.set(self._get_default_base_model_for_mode())
        elif current != str(var.get() or "").strip():
            var.set(current)

        self._on_base_model_change()
        try:
            self._refresh_training_start_state()
        except Exception:
            pass

    def _refresh_training_base_model_selection_ui(self):
        base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
        normalized_base_key = self._normalize_base_model_choice_for_ui(base_key)
        if normalized_base_key and normalized_base_key != base_key:
            try:
                self.base_model_var.set(normalized_base_key)
            except Exception:
                pass
            base_key = normalized_base_key
        base_combo = getattr(self, "base_combo", None)
        custom_row = getattr(self, "custom_row", None)
        custom_entry = getattr(self, "base_custom_entry", None)
        custom_btn = getattr(self, "base_custom_btn", None)

        show_custom_row = bool(self._is_custom_base_model_key(base_key))

        if custom_row is not None:
            try:
                if show_custom_row:
                    if not str(custom_row.winfo_manager()):
                        custom_row.pack(fill=tk.X, padx=10, pady=(0, 9), after=base_combo)
                else:
                    custom_row.pack_forget()
            except Exception:
                pass

        if custom_entry is not None:
            try:
                if self._is_custom_base_model_key(base_key):
                    custom_entry.configure(state="readonly")
                else:
                    custom_entry.configure(state=tk.DISABLED)
            except Exception:
                pass

        if custom_btn is not None:
            try:
                if not str(custom_btn.winfo_manager()):
                    custom_btn.pack(side=tk.LEFT, padx=(8, 0))
                custom_btn.configure(state=(tk.NORMAL if self._is_custom_base_model_key(base_key) else tk.DISABLED))
            except Exception:
                pass

        try:
            self._refresh_training_base_model_identity_ui()
        except Exception:
            pass
        try:
            self._refresh_step4_pinned_result_ui()
        except Exception:
            pass

    def _prepare_picker_initial_dir(self, path: Path | str | None) -> str:
        if path is None:
            return ""
        try:
            candidate = Path(path)
            if candidate.suffix and not candidate.is_dir():
                candidate = candidate.parent
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate.resolve())
        except Exception:
            try:
                return str(path)
            except Exception:
                return ""

    def _get_current_picker_parent(self, *args, **kwargs):
        return z4_dataset_validation._get_current_picker_parent(self, *args, **kwargs)


    def _get_first_picker_dir(self, *args, **kwargs):
        return z4_dataset_validation._get_first_picker_dir(self, *args, **kwargs)


    def _get_plate_xml_picker_dir(self, *args, **kwargs):
        return z4_dataset_validation._get_plate_xml_picker_dir(self, *args, **kwargs)


    def _get_plate_images_picker_dir(self, *args, **kwargs):
        return z4_dataset_validation._get_plate_images_picker_dir(self, *args, **kwargs)


    def _get_char_dataset_source_picker_dir(self, *args, **kwargs):
        return z4_dataset_validation._get_char_dataset_source_picker_dir(self, *args, **kwargs)


    def _count_yolo_image_label_pairs(self, *args, **kwargs):
        return z4_dataset_validation._count_yolo_image_label_pairs(self, *args, **kwargs)


    def _has_supported_yolo_label_layout(self, *args, **kwargs):
        return z4_dataset_validation._has_supported_yolo_label_layout(self, *args, **kwargs)


    def _find_char_yolo_dataset_root_candidates(self, *args, **kwargs):
        return z4_dataset_validation._find_char_yolo_dataset_root_candidates(self, *args, **kwargs)


    def _write_char_yolo_data_yaml_if_missing(self, *args, **kwargs):
        return z4_dataset_validation._write_char_yolo_data_yaml_if_missing(self, *args, **kwargs)


    def _validate_char_yolo_split_source(self, *args, **kwargs):
        return z4_dataset_validation._validate_char_yolo_split_source(self, *args, **kwargs)


    def _show_char_split_source_validation_modal(self, *args, **kwargs):
        return z4_dataset_validation._show_char_split_source_validation_modal(self, *args, **kwargs)


    def _pick_char_split_source_dir(self, *args, **kwargs):
        return z4_dataset_validation._pick_char_split_source_dir(self, *args, **kwargs)


    def _pick_char_split_source_yaml(self, *args, **kwargs):
        return z4_dataset_validation._pick_char_split_source_yaml(self, *args, **kwargs)


    def _get_validation_model_picker_dir(self) -> str:
        current_value = str(getattr(self, "val_model_var", tk.StringVar()).get() or "").strip()
        if current_value:
            current_path = Path(current_value)
            if current_path.is_file():
                current_path = current_path.parent
            if current_path.exists():
                return str(current_path)
        return str(self._get_preferred_models_dir(self._get_selected_training_target()))

    def _get_validation_dataset_picker_dir(self) -> str:
        current_value = str(getattr(self, "val_data_var", tk.StringVar()).get() or "").strip()
        if current_value:
            current_path = Path(current_value)
            if current_path.is_file() and current_path.name.lower() == "data.yaml":
                current_path = current_path.parent
            if current_path.exists():
                return str(current_path)
        return str(CONFIG.get_datasets_dir(self._get_selected_training_target()))

    def _get_ranking_task_target(self) -> str:
        try:
            target = CONFIG.normalize_task_target(self._get_selected_training_target())
        except Exception:
            target = "plate"
        return target if target in {"plate", "char"} else "plate"

    def _get_ranking_task_label(self, target: str | None = None) -> str:
        normalized = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
        if normalized == "char":
            return "Znaki (Detect)"
        return "Tablice (Pose)"

    def _get_ranking_split_name(self) -> str:
        raw = str(getattr(self, "rank_split_var", tk.StringVar(value="test")).get() or "test").strip().lower()
        return raw if raw in {"val", "test"} else "test"

    def _get_ranking_models_default_dir(self) -> Path:
        return self._get_preferred_models_dir(self._get_ranking_task_target())

    def _get_ranking_models_picker_dir(self) -> str:
        current_value = str(getattr(self, "rank_models_dir", tk.StringVar()).get() or "").strip()
        if current_value:
            current_path = Path(current_value)
            if current_path.is_file():
                current_path = current_path.parent
            if current_path.exists():
                return str(current_path)
        return str(self._get_ranking_models_default_dir())

    def _ensure_plate_ranking_engine(self):
        ranking_dir = Path(CONFIG.get_ranking_dir(self._get_ranking_task_target()))
        current_dir = Path(getattr(self.ranking_engine, "ranking_dir", ranking_dir))
        try:
            same_dir = current_dir.resolve() == ranking_dir.resolve()
        except Exception:
            same_dir = current_dir == ranking_dir

        if not same_dir:
            self.ranking_engine = ModelRanking(ranking_dir=ranking_dir)

    def _get_default_ranking_reference_dir(self) -> str:
        if self._get_ranking_task_target() == "char":
            try:
                dataset_yaml = self._resolve_training_dataset_yaml_path()
                if dataset_yaml is not None and Path(dataset_yaml).exists():
                    return str(Path(dataset_yaml).parent.resolve())
            except Exception:
                pass

        candidates: list[Path] = []

        try:
            dataset_yaml = self._resolve_training_dataset_yaml_path()
            if dataset_yaml is not None:
                source_info = self._resolve_plate_training_source_from_dataset(dataset_yaml)
                source_run = str(source_info.get("source_run_path") or "").strip()
                if source_run:
                    candidates.append(Path(source_run))
        except Exception:
            pass

        try:
            stored = CAMPAIGN.get_last_plate_training_source()
            source_run = str((stored or {}).get("source_run_path") or "").strip()
            if source_run:
                candidates.append(Path(source_run))
        except Exception:
            pass

        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate.resolve())
            except Exception:
                continue

        return ""

    def _get_ranking_reference_picker_dir(self) -> str:
        current_value = str(getattr(self, "rank_data_dir", tk.StringVar()).get() or "").strip()
        if current_value:
            try:
                current_path = Path(current_value)
                if current_path.is_file():
                    current_path = current_path.parent
                if current_path.exists():
                    parent_dir = current_path.parent
                    if parent_dir.exists():
                        return str(parent_dir.resolve())
                    return str(current_path.resolve())
            except Exception:
                pass

        if self._get_ranking_task_target() == "char":
            try:
                datasets_dir = Path(CONFIG.get_datasets_dir("char"))
                if datasets_dir.exists():
                    return str(datasets_dir.resolve())
            except Exception:
                pass
        else:
            try:
                campaign_auto_dir = CAMPAIGN.get_dir("auto_ann")
                if campaign_auto_dir is not None and Path(campaign_auto_dir).exists():
                    return str(Path(campaign_auto_dir).resolve())
            except Exception:
                pass

        try:
            default_dir = Path(CONFIG.get_auto_annotations_dir("plate"))
            if default_dir.exists():
                return str(default_dir.resolve())
        except Exception:
            pass

        return ""

    def _prefill_ranking_reference_if_empty(self):
        var = getattr(self, "rank_data_dir", None)
        if var is None:
            return

        current_value = str(var.get() or "").strip()
        if current_value:
            try:
                if Path(current_value).exists():
                    return
            except Exception:
                pass

        default_value = self._get_default_ranking_reference_dir()
        if not default_value:
            return

        try:
            var.set(default_value)
        except Exception:
            pass

    def _resolve_ranking_reference_source(self, raw_value: str | None = None) -> dict:
        selected_raw = str(
            raw_value if raw_value is not None else getattr(self, "rank_data_dir", tk.StringVar()).get()
        ).strip()
        if self._get_ranking_task_target() == "char":
            split_name = self._get_ranking_split_name()
            result = {
                "ok": False,
                "selected_path": selected_raw,
                "reference_dir": "",
                "reference_name": "",
                "yaml_path": "",
                "data_yaml_path": "",
                "split_name": split_name,
                "image_count": 0,
                "message": (
                    "Wybierz tor testowy znaków: folder z data.yaml albo sam plik data.yaml. "
                    "Wszystkie modele znaków pobiegną po tym samym splicie."
                ),
            }
            if not selected_raw:
                return result

            try:
                selected_path = Path(selected_raw)
            except Exception:
                result["message"] = "Nie udało się odczytać wskazanego datasetu odniesienia."
                return result

            if not selected_path.exists():
                result["message"] = f"Nie znaleziono wskazanego datasetu odniesienia: {selected_path}"
                return result

            yaml_path = selected_path / "data.yaml" if selected_path.is_dir() else selected_path
            if not yaml_path.exists() or yaml_path.name.lower() != "data.yaml":
                result["message"] = "Wskaż folder datasetu znaków z plikiem data.yaml albo sam plik data.yaml."
                return result

            try:
                inferred_target = self._infer_dataset_target(str(yaml_path))
            except Exception:
                inferred_target = "char"
            if inferred_target and inferred_target != "char":
                result["message"] = (
                    "Wskazany dataset nie wygląda na dataset znaków YOLO Detect. "
                    f"Rozpoznany tor: {self._format_training_target_label(inferred_target)}."
                )
                return result

            try:
                cfg = safe_load_yaml(yaml_path) or {}
            except Exception:
                cfg = {}

            split_value = cfg.get(split_name)
            if not split_value:
                result["message"] = (
                    f"Tor testowy nie ma splitu `{split_name}` w data.yaml. "
                    "Zmień split w Zaawansowanych albo wskaż inny dataset."
                )
                return result

            def _dataset_root() -> Path:
                raw_root = str(cfg.get("path") or "").strip()
                if not raw_root:
                    return yaml_path.parent
                root_path = Path(raw_root)
                return root_path if root_path.is_absolute() else yaml_path.parent / root_path

            def _count_split_images(value) -> int:
                root = _dataset_root()
                values = value if isinstance(value, list) else [value]
                total = 0
                for item in values:
                    raw_item = str(item or "").strip()
                    if not raw_item:
                        continue
                    split_path = Path(raw_item)
                    if not split_path.is_absolute():
                        split_path = root / split_path
                    try:
                        if split_path.is_file() and split_path.suffix.lower() == ".txt":
                            total += sum(
                                1
                                for line in split_path.read_text(encoding="utf-8-sig").splitlines()
                                if line.strip()
                            )
                        elif split_path.is_dir():
                            total += len(get_image_files(split_path))
                        elif split_path.is_file() and split_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                            total += 1
                    except Exception:
                        continue
                return total

            image_count = _count_split_images(split_value)
            if image_count <= 0:
                result["message"] = (
                    f"Split `{split_name}` w datasecie odniesienia nie zawiera obrazów możliwych do policzenia."
                )
                return result

            try:
                reference_dir = str(yaml_path.parent.resolve())
                yaml_value = str(yaml_path.resolve())
            except Exception:
                reference_dir = str(yaml_path.parent)
                yaml_value = str(yaml_path)

            dataset_ref = build_dataset_display_ref(
                yaml_path.parent,
                target_hint="char",
                counts={split_name: image_count, "total": image_count},
            )

            result.update(
                {
                    "ok": True,
                    "reference_dir": reference_dir,
                    "reference_name": f"{dataset_ref.id} / {split_name}",
                    "yaml_path": yaml_value,
                    "data_yaml_path": yaml_value,
                    "image_count": image_count,
                "message": (
                    f"Gotowy tor znaków: {dataset_ref.id} | split `{split_name}` | "
                    f"{image_count} obrazów. Każdy model dostanie ten sam egzamin."
                ),
                }
            )
            return result

        result = {
            "ok": False,
            "selected_path": selected_raw,
            "reference_dir": "",
            "reference_name": "",
            "xml_path": "",
            "images_dir": "",
            "image_paths": [],
            "image_count": 0,
            "message": (
                "Wybierz tor testowy tablic: run Z2/PZ2 z zapisanym annotations.xml oraz zgodnymi obrazami. "
                "To będzie wspólny egzamin dla modeli tablic."
            ),
        }
        if not selected_raw:
            return result

        try:
            selected_path = Path(selected_raw)
        except Exception:
            result["message"] = "Nie udało się odczytać wskazanego folderu runu."
            return result

        if not selected_path.exists():
            result["message"] = f"Nie znaleziono wskazanego folderu runu: {selected_path}"
            return result

        xml_path: Path | None = None
        reference_dir: Path | None = None

        if selected_path.is_file() and selected_path.name.lower() == "annotations.xml":
            xml_path = selected_path
            reference_dir = selected_path.parent
        elif selected_path.is_dir():
            direct_xml = selected_path / "annotations.xml"
            if direct_xml.exists():
                xml_path = direct_xml
                reference_dir = selected_path
            else:
                try:
                    xml_candidates = sorted(
                        [path for path in selected_path.rglob("annotations.xml") if path.is_file()],
                        key=lambda path: (0 if path.parent == selected_path else 1, len(str(path))),
                    )
                except Exception:
                    xml_candidates = []
                if xml_candidates:
                    xml_path = xml_candidates[0]
                    reference_dir = xml_path.parent

        if xml_path is None or reference_dir is None:
            result["message"] = (
                "W tym folderze nie ma zapisanych zmian tablic. "
                "Wskaż run Z2/PZ2, który został przejrzany i zapisany po poprawkach."
            )
            return result

        manifest = {}
        manifest_path = reference_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    manifest = payload
            except Exception:
                manifest = {}

        candidate_dirs: list[Path] = []
        seen_dirs: set[str] = set()

        def add_candidate_dir(path_like):
            if not path_like:
                return
            try:
                candidate = Path(path_like)
            except Exception:
                return
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen_dirs:
                return
            seen_dirs.add(key)
            candidate_dirs.append(candidate)

        add_candidate_dir(manifest.get("input_dir"))
        add_candidate_dir(reference_dir / "images")
        add_candidate_dir(reference_dir)
        if selected_path.is_dir():
            add_candidate_dir(selected_path)

        image_paths: list[Path] = []
        resolved_images_dir = ""
        for candidate_dir in candidate_dirs:
            try:
                files = get_image_files(candidate_dir)
            except Exception:
                files = []
            if files:
                image_paths = files
                try:
                    resolved_images_dir = str(candidate_dir.resolve())
                except Exception:
                    resolved_images_dir = str(candidate_dir)
                break

        if not image_paths and selected_path.is_dir():
            try:
                recursive_images = sorted(
                    [
                        path for path in selected_path.rglob("*")
                        if path.is_file() and path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
                    ]
                )
            except Exception:
                recursive_images = []
            if recursive_images:
                image_paths = recursive_images
                try:
                    resolved_images_dir = str(selected_path.resolve())
                except Exception:
                    resolved_images_dir = str(selected_path)

        if not image_paths:
            result["message"] = "W wybranym zestawie testowym nie znaleziono obrazów do porównania modeli."
            return result

        try:
            reference_dir_value = str(reference_dir.resolve())
        except Exception:
            reference_dir_value = str(reference_dir)

        result.update(
            {
                "ok": True,
                "reference_dir": reference_dir_value,
                "reference_name": reference_dir.name,
                "xml_path": str(xml_path.resolve()) if xml_path.exists() else str(xml_path),
                "images_dir": resolved_images_dir,
                "image_paths": image_paths,
                "image_count": len(image_paths),
                "message": (
                    f"Gotowy tor tablic: {reference_dir.name} | {len(image_paths)} obrazów. "
                    "System użyje zapisanych zmian z annotations.xml jako punktu odniesienia."
                ),
            }
        )
        return result

    def _refresh_ranking_start_state(self):
        button = getattr(self, "btn_run_rank", None)
        cancel_button = getattr(self, "btn_cancel_rank", None)
        if button is None:
            return

        if getattr(self, "rank_is_running", False):
            if cancel_button is not None:
                try:
                    cancel_button.configure(state=tk.NORMAL)
                except Exception:
                    pass
            return

        ready = False
        models_found = False
        reference_info = {"ok": False}
        if self._is_ranking_available_for_selected_target():
            models_dir_raw = str(getattr(self, "rank_models_dir", tk.StringVar()).get() or "").strip()
            reference_info = self._resolve_ranking_reference_source()
            target = self._get_ranking_task_target()
            scope = self._get_ranking_scope()
            try:
                models_dir = Path(models_dir_raw) if models_dir_raw else Path(".")
                models_source_ready = scope != "Globalne" or (models_dir.exists() and models_dir.is_dir())
                participants = (
                    list(self._collect_ranking_participant_candidates(models_dir, target, scope))
                    if models_source_ready
                    else []
                )
                enabled_participants = self._filter_enabled_ranking_participants(participants)
                models_found = bool(enabled_participants)
                ready = (
                    models_source_ready
                    and models_found
                    and reference_info["ok"]
                )
            except Exception:
                ready = False

        try:
            if not reference_info.get("ok"):
                button_text = "[ TOR ] Wybierz tor testowy"
            elif not models_found:
                button_text = "[ KONIE ] Brak startujących"
            else:
                button_text = "[ START ] Uruchom wyścig"
            button.configure(
                state=(tk.NORMAL if ready else tk.DISABLED),
                text=button_text,
            )
        except Exception:
            pass
        if cancel_button is not None:
            try:
                cancel_button.configure(state=(tk.NORMAL if getattr(self, "rank_is_running", False) else tk.DISABLED))
            except Exception:
                pass

    def _is_ranking_tab_active(self) -> bool:
        if not hasattr(self, "right_nb") or not hasattr(self, "ranking_tab"):
            return False
        if not getattr(self, "_step4_ranking_tab_visible", False):
            return False

        try:
            return str(self.right_nb.select()) == str(self.ranking_tab)
        except Exception:
            return False

    def _refresh_ranking_reference_ui(self, *_args):
        if not self._is_ranking_available_for_selected_target():
            return
        if not self._is_ranking_tab_active() and getattr(self, "rank_target_lbl", None) is None:
            return

        target = self._get_ranking_task_target()
        if not getattr(self, "_rank_target_syncing", False):
            previous_target = str(getattr(self, "_ranking_ui_target", "") or "")
            if previous_target != target:
                self._ranking_ui_target = target
                self._rank_target_syncing = True
                try:
                    try:
                        self._ensure_plate_ranking_engine()
                    except Exception:
                        pass
                    models_var = getattr(self, "rank_models_dir", None)
                    if models_var is not None:
                        models_var.set(str(self._get_ranking_models_default_dir()))
                    data_var = getattr(self, "rank_data_dir", None)
                    if data_var is not None:
                        data_var.set("")
                    try:
                        self._prefill_ranking_reference_if_empty()
                    except Exception:
                        pass
                finally:
                    self._rank_target_syncing = False

        target_label = getattr(self, "rank_target_lbl", None)
        info = self._resolve_ranking_reference_source()
        model_count = 0
        enabled_model_count = 0
        scope = self._get_ranking_scope()
        try:
            models_dir_raw = str(getattr(self, "rank_models_dir", tk.StringVar()).get() or "").strip()
            models_dir = Path(models_dir_raw) if models_dir_raw else Path(".")
            if scope != "Globalne" or (models_dir.exists() and models_dir.is_dir()):
                participants = list(self._collect_ranking_participant_candidates(models_dir, target, scope))
                model_count = len(participants)
                enabled_model_count = len(self._filter_enabled_ranking_participants(participants))
        except Exception:
            model_count = 0
            enabled_model_count = 0
        if target_label is not None:
            try:
                target_label.configure(
                    text=(
                        f"Konie: {self._format_ranking_scope_label(scope, target)} | "
                        f"startuje {enabled_model_count}/{model_count}."
                    )
                )
            except Exception:
                pass
        scope_label = getattr(self, "rank_scope_lbl", None)
        if scope_label is not None:
            try:
                scope_label.configure(text=f"Aktywny zakres: {self._format_ranking_scope_label(scope, target)}")
            except Exception:
                pass
        palette = getattr(getattr(self, "app", None), "palette", {}) or {}
        scope_value_label = getattr(self, "rank_scope_value_lbl", None)
        if scope_value_label is not None:
            try:
                scope_value_label.configure(
                    text=self._format_ranking_scope_label(scope, target),
                    fg=palette.get("accent", "#0e639c"),
                )
            except Exception:
                pass
        count_value_label = getattr(self, "rank_count_value_lbl", None)
        if count_value_label is not None:
            try:
                count_value_label.configure(
                    text=f"{enabled_model_count}/{model_count}" if model_count else "0",
                    fg=(
                        palette.get("success", "#2ecc71")
                        if enabled_model_count > 0
                        else palette.get("warning", "#f0b44c") if model_count > 0 else palette.get("error", "#e05d5d")
                    ),
                )
            except Exception:
                pass

        track_label = getattr(self, "rank_track_lbl", None)
        if track_label is not None:
            try:
                if info.get("ok"):
                    track_label.configure(
                        text=(
                            f"Tor testowy: {info.get('reference_name') or '-'} | "
                            f"{int(info.get('image_count', 0) or 0)} próbek | wspólny egzamin dla wszystkich modeli."
                        )
                    )
                else:
                    track_label.configure(text=f"Tor testowy: nie wybrano | {info.get('message') or ''}")
            except Exception:
                pass
        track_count_value_label = getattr(self, "rank_track_count_value_lbl", None)
        if track_count_value_label is not None:
            try:
                if info.get("ok"):
                    track_count_value_label.configure(
                        text=f"{int(info.get('image_count', 0) or 0)} próbek",
                        fg=palette.get("success", "#2ecc71"),
                    )
                else:
                    track_count_value_label.configure(
                        text="brak",
                        fg=palette.get("warning", "#f0b44c"),
                    )
            except Exception:
                pass

        hint_label = getattr(self, "rank_reference_hint_lbl", None)
        if hint_label is not None:
            try:
                hint_label.configure(text=info["message"])
            except Exception:
                pass

        self._refresh_ranking_start_state()

        try:
            self._load_ranking()
        except Exception:
            pass

    def _refresh_free_training_route_cards(self):
        refresh_free_training_route_cards(self)

    def _refresh_free_training_route_ui(self):
        refresh_free_training_route_ui(self)

    def _update_step4_notebook_mode(self):
        update_step4_notebook_mode(self)

    def _ensure_training_dataset_quality_rows(self, required_rows: int) -> None:
        grid = getattr(self, "train_dataset_quality_grid", None)
        if grid is None:
            return
        rows = getattr(self, "_train_dataset_quality_row_widgets", None)
        if not isinstance(rows, list):
            self._train_dataset_quality_row_widgets = []
            rows = self._train_dataset_quality_row_widgets

        palette = getattr(self.app, "palette", {})
        while len(rows) < max(0, int(required_rows)):
            row_index = len(rows) + 1
            key_label = tk.Label(
                grid,
                text="",
                font=("Segoe UI", 8, "bold"),
                anchor=tk.W,
                justify=tk.LEFT,
                bd=0,
                padx=6,
                pady=3,
                bg=palette.get("panel", "#252526"),
                fg=palette.get("fg", "#f3f3f3"),
            )
            key_label.grid(row=row_index, column=0, sticky="nsew", padx=(0, 1), pady=(0, 1))
            value_label = tk.Label(
                grid,
                text="",
                font=("Segoe UI", 8),
                anchor=tk.W,
                justify=tk.LEFT,
                bd=0,
                padx=6,
                pady=3,
                wraplength=260,
                bg=palette.get("panel_alt", palette.get("panel", "#252526")),
                fg=palette.get("muted", "#c7c7c7"),
            )
            value_label.grid(row=row_index, column=1, sticky="nsew", padx=(0, 1), pady=(0, 1))
            self._register_train_left_wrap_target(value_label, container=grid, padding=150, min_wrap=180)
            rows.append({"key": key_label, "value": value_label, "row_index": row_index})

    def _refresh_training_dataset_quality_summary(self) -> None:
        shell = getattr(self, "train_dataset_quality_shell", None)
        title = getattr(self, "train_dataset_quality_title_lbl", None)
        if shell is None:
            return

        summary = self._build_training_dataset_quality_summary()
        visible = bool(summary.get("visible", True))
        try:
            is_visible = bool(str(shell.winfo_manager()))
        except Exception:
            is_visible = False
        if visible and not is_visible:
            try:
                shell.pack(anchor=tk.W, fill=tk.X, pady=(2, 8), after=getattr(self, "train_dataset_hint_lbl", None))
            except Exception:
                try:
                    shell.pack(anchor=tk.W, fill=tk.X, pady=(2, 8))
                except Exception:
                    pass
        elif not visible and is_visible:
            try:
                shell.pack_forget()
            except Exception:
                pass

        rows = list(summary.get("rows", []) or [])
        self._ensure_training_dataset_quality_rows(len(rows))
        row_widgets = list(getattr(self, "_train_dataset_quality_row_widgets", []) or [])
        if title is not None:
            self._set_training_widget_text(title, str(summary.get("title") or "Podsumowanie materiału"))

        for row_index, widgets in enumerate(row_widgets):
            row_visible = row_index < len(rows)
            key_label = widgets.get("key")
            value_label = widgets.get("value")
            if row_visible:
                key_text, value_text = rows[row_index]
            else:
                key_text, value_text = "", ""
            for widget in (key_label, value_label):
                if widget is None:
                    continue
                try:
                    if row_visible:
                        widget.grid()
                    else:
                        widget.grid_remove()
                except Exception:
                    pass
            self._set_training_widget_text(key_label, key_text)
            self._set_training_widget_text(value_label, value_text)

        try:
            self._apply_training_dataset_quality_theme(str(summary.get("tone") or "muted"))
        except Exception:
            pass

    def _refresh_character_class_distribution_button(self) -> None:
        button = getattr(self, "btn_character_class_distribution", None)
        if button is None:
            return
        try:
            selected_target = CONFIG.normalize_task_target(self._get_selected_training_target())
        except Exception:
            selected_target = "char"
        should_show = selected_target == "char"
        is_visible = bool(getattr(self, "_character_class_distribution_button_visible", False))
        if should_show == is_visible:
            return
        try:
            if should_show:
                button.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
            else:
                button.pack_forget()
            self._character_class_distribution_button_visible = should_show
        except Exception:
            pass

    def _apply_training_dataset_quality_theme(self, tone: str = "muted") -> None:
        palette = getattr(self.app, "palette", {})
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        shell_bg = palette.get("panel", "#252526")
        title_bg = palette.get("panel_alt", "#2d2d30")
        title_fg = palette.get("fg", "#f3f3f3")
        key_bg = palette.get("panel", "#252526")
        key_fg = palette.get("fg", "#f3f3f3")
        value_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
        value_fg = palette.get("muted", "#c7c7c7")
        success_fg = palette.get("success", "#4ec9b0")
        warning_fg = palette.get("warning", "#d7ba7d")
        danger_fg = palette.get("danger", palette.get("error", "#f48771"))
        tone_fg = {
            "success": success_fg,
            "warning": warning_fg,
            "danger": danger_fg,
        }.get(str(tone or "").strip().lower(), value_fg)

        for attr_name, bg, fg in (
            ("train_dataset_quality_shell", border, title_fg),
            ("train_dataset_quality_grid", border, title_fg),
            ("train_dataset_quality_title_row", title_bg, title_fg),
            ("train_dataset_quality_title_lbl", title_bg, title_fg),
        ):
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            try:
                if isinstance(widget, tk.Label):
                    widget.configure(bg=bg, fg=fg)
                else:
                    widget.configure(bg=bg, highlightbackground=border, highlightcolor=border)
            except Exception:
                pass

        for row_index, widgets in enumerate(list(getattr(self, "_train_dataset_quality_row_widgets", []) or [])):
            row_bg = value_bg if row_index % 2 else shell_bg
            key_label = widgets.get("key")
            value_label = widgets.get("value")
            try:
                key_text = str(key_label.cget("text") if key_label is not None else "")
            except Exception:
                key_text = ""
            is_status_row = key_text.strip().lower() == "ocena"
            if key_label is not None:
                try:
                    key_label.configure(bg=key_bg, fg=key_fg)
                except Exception:
                    pass
            if value_label is not None:
                try:
                    value_label.configure(
                        bg=row_bg,
                        fg=(tone_fg if is_status_row else value_fg),
                    )
                except Exception:
                    pass

    def _update_training_dataset_hint(self):
        label = getattr(self, "train_dataset_hint_lbl", None)
        if label is not None:
            try:
                label.configure(text=self._get_training_dataset_hint_text())
            except Exception:
                pass

        intro_label = getattr(self, "train_dataset_intro_lbl", None)
        if intro_label is not None:
            try:
                if bool(CAMPAIGN.get_active_project_name()):
                    text = "Wariant splitu wynika z pracy w bramce. Możesz go zmienić, jeśli chcesz trenować na innym przygotowanym zbiorze."
                else:
                    text = "Wybierz wariant datasetu przygotowany w PZ1. Trening użyje części train, a walidacja części val."
                intro_label.configure(text=text)
            except Exception:
                pass

        selected_path_label = getattr(self, "train_dataset_selected_path_lbl", None)
        if selected_path_label is not None:
            try:
                dataset_yaml = self._resolve_training_dataset_yaml_path()
                if dataset_yaml is None:
                    selected_text = "Brak wybranego wariantu."
                else:
                    try:
                        counts = self._get_dataset_split_image_counts(dataset_yaml.parent)
                    except Exception:
                        counts = {}
                    dataset_ref = build_dataset_display_ref(
                        dataset_yaml.parent,
                        target_hint=self._get_selected_training_target(),
                        counts=counts,
                    )
                    selected_text = f"{dataset_ref.id} | {self._format_workspace_relative_path(dataset_yaml.parent)}"
                selected_path_label.configure(text=selected_text)
            except Exception:
                pass

        scope_label = getattr(self, "train_scope_hint_lbl", None)
        if scope_label is not None:
            try:
                scope_label.configure(text=self._get_training_scope_hint_text())
            except Exception:
                pass

        warning_label = getattr(self, "train_pose_warning_lbl", None)
        if warning_label is not None:
            warning_text = self._get_pose_dataset_size_warning()
            try:
                warning_label.configure(text=warning_text)
            except Exception:
                pass

            try:
                is_visible = bool(str(warning_label.winfo_manager()))
            except Exception:
                is_visible = False

            if warning_text:
                if not is_visible:
                    try:
                        warning_label.pack(anchor=tk.W, fill=tk.X, pady=(0, self._train_left_section_gap))
                    except Exception:
                        pass
            else:
                if is_visible:
                    try:
                        warning_label.pack_forget()
                    except Exception:
                        pass

        try:
            self._refresh_training_dataset_quality_summary()
        except Exception:
            pass
        try:
            self._refresh_character_class_distribution_button()
        except Exception:
            pass

        self._update_training_dataset_hint_wraplength()
        try:
            self._refresh_training_device_hint()
        except Exception:
            pass

    def _register_train_left_wrap_target(
        self,
        widget,
        *,
        container=None,
        padding: int = 20,
        min_wrap: int = 140,
    ):
        if widget is None:
            return

        try:
            self._train_left_wrap_targets.append(
                {
                    "widget": widget,
                    "container": container,
                    "padding": int(padding),
                    "min_wrap": int(min_wrap),
                }
            )
        except Exception:
            return

        for bind_target in (widget, container):
            if bind_target is None:
                continue
            try:
                bind_target.bind("<Configure>", self._update_training_dataset_hint_wraplength, add="+")
            except Exception:
                pass

    def _build_train_left_separator(self, parent, pady=(0, 0)):
        if parent is None:
            return None

        palette = getattr(self.app, "palette", {})
        spacer = tk.Frame(
            parent,
            height=1,
            bd=0,
            highlightthickness=0,
            bg=palette.get("panel", "#252526"),
        )
        spacer.pack(fill=tk.X, pady=pady)
        spacer.pack_propagate(False)
        try:
            self._train_left_section_separators.append({"host": spacer, "neutral": True})
        except Exception:
            pass
        return spacer

    def _init_train_pane_layout(self):
        pane = getattr(self, "train_pane", None)
        if pane is None or bool(getattr(self, "_train_pane_layout_initialized", False)):
            return

        try:
            total_width = int(pane.winfo_width() or 0)
        except Exception:
            total_width = 0

        if total_width < 900:
            try:
                self.frame.after(120, self._init_train_pane_layout)
            except Exception:
                pass
            return

        preferred_right = max(720, int(total_width * 0.64))
        preferred_right = min(preferred_right, max(520, total_width - 430))
        preferred_left = max(420, total_width - preferred_right)

        try:
            pane.sashpos(0, int(preferred_left))
            self._train_pane_layout_initialized = True
        except Exception:
            pass

    def _create_metric_table(
        self,
        parent,
        columns: list[tuple[str, int, str]],
        *,
        height: int | None = None,
    ) -> ttk.Treeview:
        host = ttk.Frame(parent, style="Panel.TFrame")
        host.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(
            host,
            columns=tuple(name for name, _width, _anchor in columns),
            show="headings",
            height=height,
        )

        for name, width, anchor in columns:
            tree.heading(name, text=name)
            tree.column(name, width=width, anchor=anchor, stretch=(anchor == tk.W))

        scrollbar = WebSlimScrollbar(host, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        try:
            self._dynamic_metric_tables.append({"tree": tree, "host": host, "scrollbar": scrollbar})
        except Exception:
            pass
        return tree

    def _set_metric_table_rows(self, tree, rows: list[tuple]):
        if tree is None:
            return

        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass

        for row in rows or []:
            try:
                tree.insert("", tk.END, values=tuple(row))
            except Exception:
                continue

    def _update_training_dataset_hint_wraplength(self, event=None):
        label = getattr(self, "train_dataset_hint_lbl", None)
        scope_label = getattr(self, "train_scope_hint_lbl", None)
        warning_label = getattr(self, "train_pose_warning_lbl", None)
        device_label = getattr(self, "train_device_hint_lbl", None)
        extra_targets = getattr(self, "_train_left_wrap_targets", [])
        if label is None and scope_label is None and warning_label is None and device_label is None and not extra_targets:
            return

        base_width = 0
        try:
            inset = max(0, int(getattr(self, "_train_left_content_inset", 0)))
            base_width = int(self.train_left_canvas.winfo_width()) - (2 * inset)
        except Exception:
            base_width = 0

        specs = []
        for widget in (label, scope_label, warning_label, device_label):
            if widget is not None:
                specs.append((widget, None, 2, 120))

        for spec in extra_targets:
            specs.append(
                (
                    spec.get("widget"),
                    spec.get("container"),
                    int(spec.get("padding", 20)),
                    int(spec.get("min_wrap", 140)),
                )
            )

        seen: set[int] = set()
        for widget, container, padding, min_wrap in specs:
            if widget is None:
                continue

            widget_id = id(widget)
            if widget_id in seen:
                continue
            seen.add(widget_id)

            width = 0
            try:
                if container is not None:
                    width = int(container.winfo_width())
            except Exception:
                width = 0

            if width <= 1:
                try:
                    width = int(widget.winfo_width())
                except Exception:
                    width = 0

            if width <= 1:
                width = base_width

            target = max(int(min_wrap), int(width) - int(padding))
            try:
                current = int(float(widget.cget("wraplength")))
            except Exception:
                current = 0

            if abs(current - target) <= 2:
                continue

            try:
                widget.configure(wraplength=target)
            except Exception:
                pass

    def _configure_train_progress_styles(self):
        style = getattr(self.app, "style", None) or ttk.Style()
        palette = getattr(self.app, "palette", {})
        trough = palette.get("surface_info", palette.get("panel_alt", palette.get("panel", "#252526")))
        shell_bg = palette.get("panel", palette.get("bg", "#1f1f1f"))
        border = shell_bg
        dataset_fill = palette.get("success", "#2ecc71")
        split_fill = palette.get("success", "#2ecc71")
        rank_fill = palette.get("accent_hover", dataset_fill)
        epoch_fill = palette.get("guide", palette.get("warning", "#f0b44c"))
        overall_fill = palette.get("success", "#2ecc71")
        success_fg = self._get_training_success_fg()

        try:
            style.configure(
                "TrainSplitSuccess.TLabel",
                background=palette.get("panel", "#252526"),
                foreground=success_fg,
                padding=2,
            )
            style.map(
                "TrainSplitSuccess.TLabel",
                foreground=[("disabled", success_fg), ("active", success_fg)],
                background=[
                    ("disabled", palette.get("panel", "#252526")),
                    ("active", palette.get("panel", "#252526")),
                ],
            )
        except Exception:
            pass

        try:
            style.configure(
                "TrainEpoch.Horizontal.TProgressbar",
                thickness=4,
                background=epoch_fill,
                troughcolor=trough,
                bordercolor=border,
                lightcolor=epoch_fill,
                darkcolor=epoch_fill,
            )
            style.configure(
                "TrainOverall.Horizontal.TProgressbar",
                thickness=4,
                background=overall_fill,
                troughcolor=trough,
                bordercolor=border,
                lightcolor=overall_fill,
                darkcolor=overall_fill,
            )
            style.configure(
                "TrainDataset.Horizontal.TProgressbar",
                thickness=8,
                background=dataset_fill,
                troughcolor=trough,
                bordercolor=border,
                lightcolor=dataset_fill,
                darkcolor=dataset_fill,
            )
            style.configure(
                "TrainSplit.Horizontal.TProgressbar",
                thickness=8,
                background=split_fill,
                troughcolor=trough,
                bordercolor=border,
                lightcolor=split_fill,
                darkcolor=split_fill,
            )
            style.configure(
                "TrainRank.Horizontal.TProgressbar",
                thickness=8,
                background=rank_fill,
                troughcolor=trough,
                bordercolor=border,
                lightcolor=rank_fill,
                darkcolor=rank_fill,
            )
        except Exception:
            pass

        progress_configs = {
            "train_epoch_progress": {
                "bg": shell_bg,
                "trough_color": trough,
                "fill_color": epoch_fill,
                "thickness": 4,
            },
            "train_progress": {
                "bg": shell_bg,
                "trough_color": trough,
                "fill_color": overall_fill,
                "thickness": 4,
            },
            "ds_progress": {
                "bg": palette.get("panel", "#252526"),
                "trough_color": palette.get("panel_alt", palette.get("panel", "#252526")),
                "fill_color": dataset_fill,
                "thickness": 6,
            },
            "split_progress": {
                "bg": palette.get("panel", "#252526"),
                "trough_color": palette.get("panel_alt", palette.get("panel", "#252526")),
                "fill_color": split_fill,
                "thickness": 6,
            },
            "rank_progress": {
                "bg": palette.get("panel", "#252526"),
                "trough_color": palette.get("panel_alt", palette.get("panel", "#252526")),
                "fill_color": rank_fill,
                "thickness": 6,
            },
        }

        style_targets = {
            "train_epoch_progress": "TrainEpoch.Horizontal.TProgressbar",
            "train_progress": "TrainOverall.Horizontal.TProgressbar",
            "ds_progress": "TrainDataset.Horizontal.TProgressbar",
            "split_progress": "TrainSplit.Horizontal.TProgressbar",
            "rank_progress": "TrainRank.Horizontal.TProgressbar",
        }

        for widget_name, config in progress_configs.items():
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            if isinstance(widget, TrainProgressBar):
                try:
                    widget.configure(**config)
                except Exception:
                    pass
                continue
            try:
                widget.configure(style=style_targets.get(widget_name, "Horizontal.TProgressbar"))
            except Exception:
                pass

        shell = getattr(self, "train_progress_shell", None)
        if shell is not None:
            try:
                shell.configure(bg=shell_bg, highlightbackground=shell_bg, highlightcolor=shell_bg)
            except Exception:
                pass

        for widget_name in ("train_epoch_progress_row", "train_overall_progress_row"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(bg=shell_bg, highlightbackground=shell_bg, highlightcolor=shell_bg)
            except Exception:
                pass

        for widget_name in ("train_epoch_progress_measure_lbl", "train_progress_measure_lbl"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(bg=shell_bg, fg=palette.get("muted", "#c7c7c7"))
            except Exception:
                pass

        for widget_name in ("train_epoch_progress_hint_lbl", "train_progress_hint_lbl"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(bg=shell_bg, fg=palette.get("muted_dim", palette.get("muted", "#9a9a9a")))
            except Exception:
                pass

        split_feedback_frame = getattr(self, "split_feedback_frame", None)
        if split_feedback_frame is not None:
            try:
                self.app.style_ttk_frame_widget(split_feedback_frame, background=shell_bg)
            except Exception:
                pass
        for widget_name in (
            "train_lbl",
            "val_lbl",
            "test_lbl",
            "creator_train_lbl",
            "creator_val_lbl",
            "creator_test_lbl",
            "ds_status",
            "split_status",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.configure(style="TrainSplitSuccess.TLabel")
            except Exception:
                pass
            try:
                widget.configure(foreground=success_fg)
            except Exception:
                pass

    def _get_training_success_fg(self) -> str:
        try:
            palette = getattr(self.app, "palette", {})
            success = str(palette.get("success", "#2ecc71"))
            is_dark = bool(getattr(self.app, "is_dark_theme", lambda: False)())
            return blend_hex_colors(success, "#ffffff", 0.18) if is_dark else success
        except Exception:
            return "#2ecc71"

    def _style_training_success_label(self, widget) -> None:
        if widget is None:
            return
        success_fg = self._get_training_success_fg()
        try:
            widget.configure(style="TrainSplitSuccess.TLabel")
        except Exception:
            pass
        try:
            widget.configure(foreground=success_fg)
        except Exception:
            pass

    def _style_training_error_label(self, widget) -> None:
        if widget is None:
            return
        palette = getattr(self.app, "palette", {})
        error_fg = palette.get("error", palette.get("danger", "#e05d5d"))
        try:
            widget.configure(style="PanelError.TLabel")
        except Exception:
            pass
        try:
            widget.configure(foreground=error_fg)
        except Exception:
            pass

    def _set_train_progress_values(self, *, overall: float | None = None, epoch: float | None = None):
        if overall is not None:
            try:
                self.train_progress_var.set(max(0.0, min(100.0, float(overall))))
            except Exception:
                pass

        if epoch is not None:
            try:
                self.train_epoch_progress_var.set(max(0.0, min(100.0, float(epoch))))
            except Exception:
                pass

    @staticmethod
    def _format_training_eta(self, *args, **kwargs):
        return z4_training_progress._format_training_eta(self, *args, **kwargs)

    def _reset_training_runtime_progress(self, *args, **kwargs):
        return z4_training_progress._reset_training_runtime_progress(self, *args, **kwargs)

    def _update_training_progress_meta(self, *args, **kwargs):
        return z4_training_progress._update_training_progress_meta(self, *args, **kwargs)

    def _is_memory_failure_text(self, *args, **kwargs):
        return z4_training_progress._is_memory_failure_text(self, *args, **kwargs)

    def _is_cuda_runtime_broken_text(self, *args, **kwargs):
        return z4_training_progress._is_cuda_runtime_broken_text(self, *args, **kwargs)

    def _get_training_gpu_memory_snapshot(self, *args, **kwargs):
        return z4_training_progress._get_training_gpu_memory_snapshot(self, *args, **kwargs)

    def _build_training_gpu_memory_lines(self, *args, **kwargs):
        return z4_training_progress._build_training_gpu_memory_lines(self, *args, **kwargs)

    def _build_training_failure_message(self, *args, **kwargs):
        return z4_training_progress._build_training_failure_message(self, *args, **kwargs)

    #=====================================
    def set_campaign_context(self, runs_dir=None, datasets_dir=None):
        set_campaign_context(self, runs_dir=runs_dir, datasets_dir=datasets_dir)

    def get_campaign_step4_readiness(self, *args, **kwargs):
        return z4_campaign_state.get_campaign_step4_readiness(self, *args, **kwargs)


    def open_campaign_step4_entry(self, *args, **kwargs):
        return z4_campaign_state.open_campaign_step4_entry(self, *args, **kwargs)



    #=====================================
    def _restore_step4_campaign_project_state(self, *args, **kwargs):
        return z4_campaign_state._restore_step4_campaign_project_state(self, *args, **kwargs)


    @staticmethod
    def _plate_dataset_source_manifest_path(*args, **kwargs):
        return z4_campaign_state._plate_dataset_source_manifest_path(*args, **kwargs)


    def _load_plate_dataset_source_manifest(self, *args, **kwargs):
        return z4_campaign_state._load_plate_dataset_source_manifest(self, *args, **kwargs)


    def _write_plate_dataset_source_manifest(self, *args, **kwargs):
        return z4_campaign_state._write_plate_dataset_source_manifest(self, *args, **kwargs)


    def _resolve_plate_training_source_from_dataset(self, *args, **kwargs):
        return z4_campaign_state._resolve_plate_training_source_from_dataset(self, *args, **kwargs)


    def _remember_campaign_plate_training_source(self, *args, **kwargs):
        return z4_campaign_state._remember_campaign_plate_training_source(self, *args, **kwargs)


    def _remember_campaign_training_run_in_registry(self, *args, **kwargs):
        return z4_campaign_state._remember_campaign_training_run_in_registry(self, *args, **kwargs)


    def clear_campaign_context(self):
        clear_campaign_context(self)

    def _reset_step4_transient_ui(self, *args, **kwargs):
        return z4_layout_runtime._reset_step4_transient_ui(self, *args, **kwargs)


    def set_campaign_training_target(self, *args, **kwargs):
        return z4_layout_runtime.set_campaign_training_target(self, *args, **kwargs)


    def get_campaign_training_target(self, *args, **kwargs):
        return z4_layout_runtime.get_campaign_training_target(self, *args, **kwargs)


    def _append_step4_builder_log(self, *args, **kwargs):
        return z4_layout_runtime._append_step4_builder_log(self, *args, **kwargs)


    def _set_step4_builder_log_visibility(self, *args, **kwargs):
        return z4_layout_runtime._set_step4_builder_log_visibility(self, *args, **kwargs)


    def _toggle_step4_builder_log(self, *args, **kwargs):
        return z4_layout_runtime._toggle_step4_builder_log(self, *args, **kwargs)


    def _set_split_feedback_visibility(self, *args, **kwargs):
        return z4_layout_runtime._set_split_feedback_visibility(self, *args, **kwargs)


    def _set_step4_process_console_text(self, *args, **kwargs):
        return z4_layout_runtime._set_step4_process_console_text(self, *args, **kwargs)


    def _sync_train_left_scrollregion(self, *args, **kwargs):
        return z4_layout_runtime._sync_train_left_scrollregion(self, *args, **kwargs)


    def _sync_dataset_mode_scrollregion(self, *args, **kwargs):
        return z4_layout_runtime._sync_dataset_mode_scrollregion(self, *args, **kwargs)


    def _sync_train_left_canvas_width(self, *args, **kwargs):
        return z4_layout_runtime._sync_train_left_canvas_width(self, *args, **kwargs)


    def _sync_dataset_mode_canvas_width(self, *args, **kwargs):
        return z4_layout_runtime._sync_dataset_mode_canvas_width(self, *args, **kwargs)


    def ensure_visible_layout_ready(self, *args, **kwargs):
        return z4_layout_runtime.ensure_visible_layout_ready(self, *args, **kwargs)


    def _widget_contains_point(self, *args, **kwargs):
        return z4_layout_runtime._widget_contains_point(self, *args, **kwargs)


    def _mousewheel_units(self, *args, **kwargs):
        return z4_layout_runtime._mousewheel_units(self, *args, **kwargs)


    def _train_left_canvas_overflows(self, *args, **kwargs):
        return z4_layout_runtime._train_left_canvas_overflows(self, *args, **kwargs)


    def _dataset_mode_canvas_overflows(self, *args, **kwargs):
        return z4_layout_runtime._dataset_mode_canvas_overflows(self, *args, **kwargs)


    def _scroll_canvas_overflows(self, *args, **kwargs):
        return z4_layout_runtime._scroll_canvas_overflows(self, *args, **kwargs)


    def _on_train_left_global_mousewheel(self, *args, **kwargs):
        return z4_layout_runtime._on_train_left_global_mousewheel(self, *args, **kwargs)


    def _on_dataset_mode_global_mousewheel(self, *args, **kwargs):
        return z4_layout_runtime._on_dataset_mode_global_mousewheel(self, *args, **kwargs)


    def _restore_scroll_canvas_focus(self, *args, **kwargs):
        return z4_layout_runtime._restore_scroll_canvas_focus(self, *args, **kwargs)


    def _redirect_child_mousewheel_to_canvas(self, *args, **kwargs):
        return z4_layout_runtime._redirect_child_mousewheel_to_canvas(self, *args, **kwargs)


    def _redirect_combobox_mousewheel_to_canvas(self, *args, **kwargs):
        return z4_layout_runtime._redirect_combobox_mousewheel_to_canvas(self, *args, **kwargs)


    def _register_training_scroll_guard_combobox(self, *args, **kwargs):
        return z4_layout_runtime._register_training_scroll_guard_combobox(self, *args, **kwargs)


    def _mark_training_combobox_scroll_guard(self, *args, **kwargs):
        return z4_layout_runtime._mark_training_combobox_scroll_guard(self, *args, **kwargs)


    def _schedule_training_combobox_guard_poll(self, *args, **kwargs):
        return z4_layout_runtime._schedule_training_combobox_guard_poll(self, *args, **kwargs)


    def _poll_training_combobox_scroll_guard(self, *args, **kwargs):
        return z4_layout_runtime._poll_training_combobox_scroll_guard(self, *args, **kwargs)


    def _combobox_popdown_visible(self, *args, **kwargs):
        return z4_layout_runtime._combobox_popdown_visible(self, *args, **kwargs)


    def _any_training_combobox_popdown_visible(self, *args, **kwargs):
        return z4_layout_runtime._any_training_combobox_popdown_visible(self, *args, **kwargs)


    def _training_combobox_scroll_guard_active(self, *args, **kwargs):
        return z4_layout_runtime._training_combobox_scroll_guard_active(self, *args, **kwargs)


    def _widget_is_combobox_or_popdown(self, *args, **kwargs):
        return z4_layout_runtime._widget_is_combobox_or_popdown(self, *args, **kwargs)


    def _mousewheel_event_from_combobox(self, *args, **kwargs):
        return z4_layout_runtime._mousewheel_event_from_combobox(self, *args, **kwargs)


    def _bind_scroll_canvas_children(self, *args, **kwargs):
        return z4_layout_runtime._bind_scroll_canvas_children(self, *args, **kwargs)


    def apply_theme(self, *args, **kwargs):
        return z4_theme_runtime.apply_theme(self, *args, **kwargs)

    def _set_step4_train_log_visibility(self, visible: bool):
        if not hasattr(self, "step4_train_log_frame"):
            return

        self._step4_train_log_visible = bool(visible)
        try:
            self.step4_train_log_frame.pack_forget()
        except Exception:
            pass
        if hasattr(self, "step4_train_log_host"):
            try:
                self.step4_train_log_host.grid_remove()
            except Exception:
                pass

        if self._step4_train_log_visible:
            try:
                if hasattr(self.app, "show_global_terminal"):
                    self.app.show_global_terminal()
            except Exception:
                pass

        if hasattr(self, "btn_toggle_step4_train_log"):
            try:
                self.btn_toggle_step4_train_log.configure(text="Terminal")
            except Exception:
                pass
        return

        if self._step4_train_log_visible:
            if hasattr(self, "step4_train_log_host"):
                try:
                    self.step4_train_log_host.grid()
                except Exception:
                    pass
            self.step4_train_log_frame.pack(fill=tk.BOTH, expand=False)
            if hasattr(self, "btn_toggle_step4_train_log"):
                self.btn_toggle_step4_train_log.configure(text="Ukryj terminal")
        else:
            self.step4_train_log_frame.pack_forget()
            if hasattr(self, "step4_train_log_host"):
                try:
                    self.step4_train_log_host.grid_remove()
                except Exception:
                    pass
            if hasattr(self, "btn_toggle_step4_train_log"):
                self.btn_toggle_step4_train_log.configure(text="Pokaż terminal")

    def _toggle_step4_train_log(self):
        try:
            if hasattr(self.app, "toggle_global_terminal"):
                self.app.toggle_global_terminal()
                return
        except Exception:
            pass
        self._set_step4_train_log_visibility(
            not getattr(self, "_step4_train_log_visible", False)
        )

    def _select_step4_analysis_tab(self, tab_widget):
        try:
            self._ensure_step4_train_tab_built()
        except Exception:
            pass
        if not hasattr(self, "right_nb"):
            return

        try:
            self.right_nb.select(tab_widget)
            self._sync_step4_analysis_nav_buttons()
        except Exception:
            pass

    def _is_ranking_available_for_selected_target(self) -> bool:
        return CONFIG.normalize_task_target(self._get_selected_training_target()) in {"plate", "char"}

    def _refresh_step4_analysis_tab_visibility(self):
        refresh_step4_analysis_tab_visibility(self)

    def _sync_step4_analysis_nav_buttons(self, event=None):
        sync_step4_analysis_nav_buttons(self, event=event)

    def _clear_step4_guidance(self):
        clear_step4_guidance(self)

    def _guide_step4_route_selection(self):
        guide_step4_route_selection(self)

    def _guide_step4_builder_action(self):
        guide_step4_builder_action(self)

    def _guide_step4_next_action(self):
        guide_step4_next_action(self)

    def _guide_step4_training_action(self):
        guide_step4_training_action(self)

    def _guide_step4_finish_action(self):
        guide_step4_finish_action(self)

    def _open_step4_dataset_stage(self):
        open_step4_dataset_stage(self)

    def _mark_step4_dataset_ready(
        self,
        dataset_path: str | Path | None = None,
        *,
        target: str | None = None,
    ):
        mark_step4_dataset_ready(self, dataset_path=dataset_path, target=target)

    def _accept_training_input_context(
        self,
        *,
        source: str = "",
        target: str = "",
        dataset_path: str | Path | None = None,
        select_training: bool = False,
    ) -> bool:
        return accept_training_input_context(
            self,
            source=source,
            target=target,
            dataset_path=dataset_path,
            select_training=select_training,
        )

    def _set_step4_dataset_mode(self, mode: str, *, show_locked_message: bool = True):
        set_step4_dataset_mode(self, mode, show_locked_message=show_locked_message)

    def _get_step4_dataset_workflow_view_model(self) -> Step4DatasetWorkflowViewModel:
        return build_step4_dataset_workflow_view_model(self)

    def _get_step4_training_inputs_view_model(self) -> Step4TrainingInputsViewModel:
        return build_step4_training_inputs_view_model(self)

    def _get_step4_campaign_navigation_view_model(self) -> Step4CampaignNavigationViewModel:
        return build_step4_campaign_navigation_view_model(self)

    def _refresh_step4_campaign_builder_inputs_ui(self):
        refresh_step4_campaign_builder_inputs_ui(self)

    def _refresh_step4_training_inputs_mode_ui(self):
        if not bool(getattr(self, "_step4_train_tab_built", False)):
            return
        refresh_step4_training_inputs_mode_ui(self)

    def _refresh_step4_dataset_mode_ui(self, *, lightweight: bool = False):
        refresh_step4_dataset_mode_ui(self, lightweight=lightweight)

    def _step4_dataset_go_next(self):
        step4_dataset_go_next(self)

    def _step4_dataset_go_back(self):
        step4_dataset_go_back(self)

    def _toggle_step4_char_split_details(self):
        self._step4_char_split_details_visible = not bool(getattr(self, "_step4_char_split_details_visible", False))
        try:
            self._refresh_step4_campaign_builder_inputs_ui()
        except Exception:
            pass

    def _refresh_step4_campaign_navigation_ui(self):
        refresh_step4_campaign_navigation_ui(self)


    def _step4_train_go_back(self):
        step4_train_go_back(self)

    def _finish_campaign_step4(self):
        return finish_campaign_step4(self)

    def _complete_campaign_project(self):
        complete_campaign_project(self)

    def _get_run_dir_for_run_id(self, run_id: str) -> Path | None:
        if not run_id:
            return None

        base = self._get_runs_base_dir()
        candidate = base / run_id
        if candidate.exists() and candidate.is_dir():
            return candidate

        # fallback: szukaj po stemie / nazwie zawierającej run_id
        try:
            matches = [p for p in base.iterdir() if p.is_dir() and run_id in p.name]
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return matches[0]
        except Exception:
            pass

        return None


    def _find_best_weights_for_run(self, run_id: str) -> Path | None:
        run_dir = self._get_run_dir_for_run_id(run_id)
        if run_dir is None:
            return None

        direct = run_dir / "weights" / "best.pt"
        if direct.exists():
            return direct

        try:
            candidates = list(run_dir.rglob("best.pt"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0]
        except Exception:
            pass

        return None

    @staticmethod
    def _json_safe_training_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): TrainingTab._json_safe_training_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [TrainingTab._json_safe_training_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _training_metric_float(value) -> float | None:
        if value in (None, "", "-"):
            return None
        try:
            numeric = float(str(value).strip().replace(",", "."))
        except Exception:
            return None
        return numeric if numeric == numeric else None

    @classmethod
    def _training_metric_value(cls, mapping: dict | None, keys: tuple[str, ...]) -> float | None:
        if not isinstance(mapping, dict):
            return None

        for key in keys:
            if key in mapping:
                value = cls._training_metric_float(mapping.get(key))
                if value is not None:
                    return value

        lowered = {str(k or "").strip().lower(): v for k, v in mapping.items()}
        for key in keys:
            value = cls._training_metric_float(lowered.get(str(key or "").strip().lower()))
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_model_export_slug(value: str, fallback: str = "model") -> str:
        text = str(value or "").strip()
        if not text:
            text = fallback
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
        return (slug or fallback)[:48]

    def _infer_history_run_target(self, *args, **kwargs):
        return z4_model_export._infer_history_run_target(self, *args, **kwargs)

    def _resolve_history_run_best_weights(self, *args, **kwargs):
        return z4_model_export._resolve_history_run_best_weights(self, *args, **kwargs)

    def _build_history_run_metric_summary(self, *args, **kwargs):
        return z4_model_export._build_history_run_metric_summary(self, *args, **kwargs)

    def _build_free_mode_model_export_path(self, *args, **kwargs):
        return z4_model_export._build_free_mode_model_export_path(self, *args, **kwargs)

    def _build_mobile_model_export_path(self, *args, **kwargs):
        return z4_model_export._build_mobile_model_export_path(self, *args, **kwargs)

    def _build_mobile_alpr_package_export_path(self, *args, **kwargs):
        return z4_model_export._build_mobile_alpr_package_export_path(self, *args, **kwargs)

    def _build_mobile_export_metadata(self, *args, **kwargs):
        return z4_model_export._build_mobile_export_metadata(self, *args, **kwargs)

    def _build_exported_model_metadata(self, *args, **kwargs):
        return z4_model_export._build_exported_model_metadata(self, *args, **kwargs)

    def _export_selected_run_model_to_free_mode(self, *args, **kwargs):
        return z4_model_export._export_selected_run_model_to_free_mode(self, *args, **kwargs)

    def _export_selected_run_model_to_mobile_package(self, *args, **kwargs):
        return z4_model_export._export_selected_run_model_to_mobile_package(self, *args, **kwargs)

    def _open_mobile_model_export_center(self, *args, **kwargs):
        return z4_model_export._open_mobile_model_export_center(self, *args, **kwargs)


    def _poll_training_completion(self):
        poll_training_completion(self)

    def _set_training_ui_idle_state(self, status_text="Czekam na start...", color="gray"):
        try:
            self._refresh_training_start_state()
        except Exception:
            pass

        try:
            self.btn_pause_train.configure(state=tk.DISABLED)
        except Exception:
            pass

        try:
            self.btn_stop_train.configure(state=tk.DISABLED)
        except Exception:
            pass

        try:
            self.train_progress_label.configure(
                text=status_text,
                foreground=color
            )
        except Exception:
            pass
        try:
            self._reset_training_runtime_progress()
        except Exception:
            pass


    def _scan_training_cuda_devices(self, *args, **kwargs):
        return z4_device_runtime._scan_training_cuda_devices(self, *args, **kwargs)

    def _get_available_devices(self, *args, **kwargs):
        return z4_device_runtime._get_available_devices(self, *args, **kwargs)

    def _get_global_training_device_choice(self, *args, **kwargs):
        return z4_device_runtime._get_global_training_device_choice(self, *args, **kwargs)

    def apply_global_yolo_device_choice(self, *args, **kwargs):
        return z4_device_runtime.apply_global_yolo_device_choice(self, *args, **kwargs)

    def _normalize_training_device_choice(self, *args, **kwargs):
        return z4_device_runtime._normalize_training_device_choice(self, *args, **kwargs)

    def _get_selected_training_device_raw(self, *args, **kwargs):
        return z4_device_runtime._get_selected_training_device_raw(self, *args, **kwargs)

    def _get_effective_training_device_profile(self, *args, **kwargs):
        return z4_device_runtime._get_effective_training_device_profile(self, *args, **kwargs)

    def _get_training_device_recommendation(self, *args, **kwargs):
        return z4_device_runtime._get_training_device_recommendation(self, *args, **kwargs)

    def _refresh_training_device_hint(self, *args, **kwargs):
        return z4_device_runtime._refresh_training_device_hint(self, *args, **kwargs)

    @staticmethod
    def _is_history_run_resumable(*args, **kwargs):
        return z4_history_runtime._is_history_run_resumable(*args, **kwargs)

    def _get_latest_campaign_resumable_run_id(self, *args, **kwargs):
        return z4_history_runtime._get_latest_campaign_resumable_run_id(self, *args, **kwargs)

    def _is_history_run_resume_allowed(self, *args, **kwargs):
        return z4_history_runtime._is_history_run_resume_allowed(self, *args, **kwargs)

    def _format_history_run_status_label(self, *args, **kwargs):
        return z4_history_runtime._format_history_run_status_label(self, *args, **kwargs)

    def _does_history_run_match_active_campaign_target(self, *args, **kwargs):
        return z4_history_runtime._does_history_run_match_active_campaign_target(self, *args, **kwargs)

    def _device_to_ultralytics(self, *args, **kwargs):
        return z4_history_runtime._device_to_ultralytics(self, *args, **kwargs)

    def _open_path(self, *args, **kwargs):
        return z4_history_runtime._open_path(self, *args, **kwargs)

    def _selected_run(self, *args, **kwargs):
        return z4_history_runtime._selected_run(self, *args, **kwargs)

    def _build_ui(self, *args, **kwargs):
        return z4_tab_shell._build_ui(self, *args, **kwargs)

    def _build_train_tab_placeholder(self, *args, **kwargs):
        return z4_tab_shell._build_train_tab_placeholder(self, *args, **kwargs)

    def _ensure_step4_train_tab_built(self, *args, **kwargs):
        return z4_tab_shell._ensure_step4_train_tab_built(self, *args, **kwargs)

    def _on_main_nb_tab_changed(self, *args, **kwargs):
        return z4_tab_shell._on_main_nb_tab_changed(self, *args, **kwargs)

    def get_free_mode_assistant_context(self, *args, **kwargs):
        return z4_tab_shell.get_free_mode_assistant_context(self, *args, **kwargs)

    def _build_dataset_tab(self, *args, **kwargs):
        return z4_dataset_tab_builder._build_dataset_tab(self, *args, **kwargs)

    def _build_creator_ui(self, *args, **kwargs):
        return z4_dataset_panels._build_creator_ui(self, *args, **kwargs)

    def _build_step4_augmentation_controls(self, *args, **kwargs):
        return z4_dataset_panels._build_step4_augmentation_controls(self, *args, **kwargs)

    def _build_splitter_ui(self, *args, **kwargs):
        return z4_dataset_panels._build_splitter_ui(self, *args, **kwargs)

    def _build_train_tab(self, *args, **kwargs):
        return z4_train_tab_builder._build_train_tab(self, *args, **kwargs)

    @staticmethod
    def _format_validation_metric_name(*args, **kwargs):
        return z4_validation_panel._format_validation_metric_name(*args, **kwargs)

    @staticmethod
    def _format_validation_metric_value(*args, **kwargs):
        return z4_validation_panel._format_validation_metric_value(*args, **kwargs)

    def _build_validation_panel_v2(self, *args, **kwargs):
        return z4_validation_panel._build_validation_panel_v2(self, *args, **kwargs)

    def _open_model_validation_modal(self, *args, **kwargs):
        return z4_validation_panel._open_model_validation_modal(self, *args, **kwargs)

    def _open_selected_run_validation_modal(self, *args, **kwargs):
        return z4_validation_panel._open_selected_run_validation_modal(self, *args, **kwargs)

    def _open_current_run_validation_modal(self, *args, **kwargs):
        return z4_validation_panel._open_current_run_validation_modal(self, *args, **kwargs)

    def _open_selected_ranking_validation_modal(self, *args, **kwargs):
        return z4_validation_panel._open_selected_ranking_validation_modal(self, *args, **kwargs)

    def _format_validation_metric_band(self, *args, **kwargs):
        return z4_validation_panel._format_validation_metric_band(self, *args, **kwargs)

    def _extract_validation_metric_rows(self, *args, **kwargs):
        return z4_validation_panel._extract_validation_metric_rows(self, *args, **kwargs)

    def _open_validation_results_modal(self, *args, **kwargs):
        return z4_validation_panel._open_validation_results_modal(self, *args, **kwargs)

    def _set_validation_summary(self, *args, **kwargs):
        return z4_validation_panel._set_validation_summary(self, *args, **kwargs)

    def _collect_run_analysis_paths(self, *args, **kwargs):
        return z4_analysis_ranking._collect_run_analysis_paths(self, *args, **kwargs)
    def _load_run_results_csv_rows(self, *args, **kwargs):
        return z4_analysis_ranking._load_run_results_csv_rows(self, *args, **kwargs)
    @staticmethod
    def _run_analysis_font(*args, **kwargs):
        return z4_analysis_ranking._run_analysis_font(*args, **kwargs)
    def _wrap_run_analysis_line(self, *args, **kwargs):
        return z4_analysis_ranking._wrap_run_analysis_line(self, *args, **kwargs)
    def _render_run_analysis_sheet(self, *args, **kwargs):
        return z4_analysis_ranking._render_run_analysis_sheet(self, *args, **kwargs)
    def _analysis_plot_info(self, *args, **kwargs):
        return z4_analysis_ranking._analysis_plot_info(self, *args, **kwargs)
    def _analysis_plot_list_label(self, *args, **kwargs):
        return z4_analysis_ranking._analysis_plot_list_label(self, *args, **kwargs)
    def _build_fallback_run_analysis_paths(self, *args, **kwargs):
        return z4_analysis_ranking._build_fallback_run_analysis_paths(self, *args, **kwargs)
    def _close_analysis_dialog(self, *args, **kwargs):
        return z4_analysis_ranking._close_analysis_dialog(self, *args, **kwargs)
    def _style_analysis_dialog(self, *args, **kwargs):
        return z4_analysis_ranking._style_analysis_dialog(self, *args, **kwargs)
    def _populate_analysis_dialog(self, *args, **kwargs):
        return z4_analysis_ranking._populate_analysis_dialog(self, *args, **kwargs)
    def _open_run_analysis_window(self, *args, **kwargs):
        return z4_analysis_ranking._open_run_analysis_window(self, *args, **kwargs)
    def _open_selected_run_analysis(self, *args, **kwargs):
        return z4_analysis_ranking._open_selected_run_analysis(self, *args, **kwargs)
    def _open_selected_runs_compare(self, *args, **kwargs):
        return z4_training_compare._open_selected_runs_compare(self, *args, **kwargs)
    def _close_training_compare_dialog(self, *args, **kwargs):
        return z4_training_compare._close_training_compare_dialog(self, *args, **kwargs)
    def _draw_training_compare_chart(self, *args, **kwargs):
        return z4_training_compare._draw_training_compare_chart(self, *args, **kwargs)
    def _on_training_compare_canvas_motion(self, *args, **kwargs):
        return z4_training_compare._on_training_compare_canvas_motion(self, *args, **kwargs)
    def _open_training_compare_selected_details(self, *args, **kwargs):
        return z4_training_compare._open_training_compare_selected_details(self, *args, **kwargs)
    def _promote_training_compare_selected_run(self, *args, **kwargs):
        return z4_training_compare._promote_training_compare_selected_run(self, *args, **kwargs)
    def _on_analysis_plot_selected(self, *args, **kwargs):
        return z4_analysis_ranking._on_analysis_plot_selected(self, *args, **kwargs)
    def _show_analysis_plot(self, *args, **kwargs):
        return z4_analysis_ranking._show_analysis_plot(self, *args, **kwargs)
    def _build_ranking_panel_v2(self, *args, **kwargs):
        return z4_analysis_ranking._build_ranking_panel_v2(self, *args, **kwargs)
    def _open_ranking_track_modal(self, *args, **kwargs):
        return z4_analysis_ranking._open_ranking_track_modal(self, *args, **kwargs)
    def _open_ranking_participants_modal(self, *args, **kwargs):
        return z4_analysis_ranking._open_ranking_participants_modal(self, *args, **kwargs)
    def _open_ranking_advanced_modal(self, *args, **kwargs):
        return z4_analysis_ranking._open_ranking_advanced_modal(self, *args, **kwargs)
    def _run_ranking_v2(self, *args, **kwargs):
        return z4_analysis_ranking._run_ranking_v2(self, *args, **kwargs)
    def _cancel_ranking_v2(self, *args, **kwargs):
        return z4_analysis_ranking._cancel_ranking_v2(self, *args, **kwargs)
    def _open_ranking_report_viewer(self, *args, **kwargs):
        return z4_analysis_ranking._open_ranking_report_viewer(self, *args, **kwargs)
    def _export_ranking_analysis_report(self, *args, **kwargs):
        return z4_analysis_ranking._export_ranking_analysis_report(self, *args, **kwargs)
    def _collect_project_plate_ranking_model_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_project_plate_ranking_model_candidates(self, *args, **kwargs)
    def _collect_project_ranking_model_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_project_ranking_model_candidates(self, *args, **kwargs)
    def _collect_plate_ranking_model_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_plate_ranking_model_candidates(self, *args, **kwargs)
    def _collect_ranking_model_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_ranking_model_candidates(self, *args, **kwargs)
    def _get_ranking_scope(self, *args, **kwargs):
        return z4_analysis_ranking._get_ranking_scope(self, *args, **kwargs)
    def _format_ranking_scope_label(self, *args, **kwargs):
        return z4_analysis_ranking._format_ranking_scope_label(self, *args, **kwargs)
    def _ranking_model_candidate_scope(self, *args, **kwargs):
        return z4_analysis_ranking._ranking_model_candidate_scope(self, *args, **kwargs)
    def _collect_ranking_participant_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_ranking_participant_candidates(self, *args, **kwargs)
    def _filter_enabled_ranking_participants(self, *args, **kwargs):
        return z4_analysis_ranking._filter_enabled_ranking_participants(self, *args, **kwargs)
    def _collect_ranking_track_candidates(self, *args, **kwargs):
        return z4_analysis_ranking._collect_ranking_track_candidates(self, *args, **kwargs)
    def _pick_file(self, *args, **kwargs):
        return z4_dataset_builder._pick_file(self, *args, **kwargs)
    def _pick_dir(self, *args, **kwargs):
        return z4_dataset_builder._pick_dir(self, *args, **kwargs)
    def _update_ratio_labels(self, *args, **kwargs):
        return z4_dataset_builder._update_ratio_labels(self, *args, **kwargs)
    def _invalidate_pending_character_balance_plan(self, *args, **kwargs):
        return z4_dataset_builder._invalidate_pending_character_balance_plan(self, *args, **kwargs)
    def _on_base_model_change(self, *args, **kwargs):
        return z4_dataset_builder._on_base_model_change(self, *args, **kwargs)
    def _on_cvat_xml_source_changed(self, *args, **kwargs):
        return z4_dataset_builder._on_cvat_xml_source_changed(self, *args, **kwargs)
    def _on_cvat_images_source_changed(self, *args, **kwargs):
        return z4_dataset_builder._on_cvat_images_source_changed(self, *args, **kwargs)
    def _schedule_cvat_source_binding_refresh(self, *args, **kwargs):
        return z4_dataset_builder._schedule_cvat_source_binding_refresh(self, *args, **kwargs)
    def _normalize_cvat_xml_image_relpath(self, *args, **kwargs):
        return z4_dataset_builder._normalize_cvat_xml_image_relpath(self, *args, **kwargs)
    def _cvat_xml_relpath_to_path(self, *args, **kwargs):
        return z4_dataset_builder._cvat_xml_relpath_to_path(self, *args, **kwargs)
    def _read_cvat_xml_image_names(self, *args, **kwargs):
        return z4_dataset_builder._read_cvat_xml_image_names(self, *args, **kwargs)
    def _evaluate_images_dir_for_cvat_xml(self, *args, **kwargs):
        return z4_dataset_builder._evaluate_images_dir_for_cvat_xml(self, *args, **kwargs)
    def _get_cvat_image_source_roots(self, *args, **kwargs):
        return z4_dataset_builder._get_cvat_image_source_roots(self, *args, **kwargs)
    def _derive_cvat_candidate_root_for_match(self, *args, **kwargs):
        return z4_dataset_builder._derive_cvat_candidate_root_for_match(self, *args, **kwargs)
    def _find_matching_images_dir_for_cvat_xml(self, *args, **kwargs):
        return z4_dataset_builder._find_matching_images_dir_for_cvat_xml(self, *args, **kwargs)
    def _auto_bind_cvat_images_dir_from_xml(self, *args, **kwargs):
        return z4_dataset_builder._auto_bind_cvat_images_dir_from_xml(self, *args, **kwargs)
    def _get_creator_source_mode(self, *args, **kwargs):
        return z4_dataset_builder._get_creator_source_mode(self, *args, **kwargs)
    def _set_creator_source_mode(self, *args, **kwargs):
        return z4_dataset_builder._set_creator_source_mode(self, *args, **kwargs)
    def _on_creator_source_mode_change(self, *args, **kwargs):
        return z4_dataset_builder._on_creator_source_mode_change(self, *args, **kwargs)
    def _resolve_dataset_creator_inputs(self, *args, **kwargs):
        return z4_dataset_builder._resolve_dataset_creator_inputs(self, *args, **kwargs)
    @staticmethod
    def _normalize_xml_image_name(*args, **kwargs):
        return z4_dataset_builder._normalize_xml_image_name(*args, **kwargs)
    def _validate_dataset_creator_xml_images_alignment(self, *args, **kwargs):
        return z4_dataset_builder._validate_dataset_creator_xml_images_alignment(self, *args, **kwargs)
    def _refresh_dataset_creator_cta_state(self, *args, **kwargs):
        return z4_dataset_builder._refresh_dataset_creator_cta_state(self, *args, **kwargs)
    def _resolve_dataset_split_inputs(self, *args, **kwargs):
        return z4_dataset_builder._resolve_dataset_split_inputs(self, *args, **kwargs)
    def _refresh_dataset_split_cta_state(self, *args, **kwargs):
        return z4_dataset_builder._refresh_dataset_split_cta_state(self, *args, **kwargs)
    def _can_open_pz2_after_dataset_result(self, *args, **kwargs):
        return z4_dataset_builder._can_open_pz2_after_dataset_result(self, *args, **kwargs)
    def _show_step4_dataset_result_modal(self, *args, **kwargs):
        return z4_dataset_builder._show_step4_dataset_result_modal(self, *args, **kwargs)
    def _format_step4_dataset_result_path(self, *args, **kwargs):
        return z4_dataset_builder._format_step4_dataset_result_path(self, *args, **kwargs)
    def _open_pz2_from_dataset_result(self, *args, **kwargs):
        return z4_dataset_builder._open_pz2_from_dataset_result(self, *args, **kwargs)
    def _refresh_step4_augmentation_dependency_state(self, *args, **kwargs):
        return z4_dataset_builder._refresh_step4_augmentation_dependency_state(self, *args, **kwargs)
    def _install_step4_albumentations_dependency(self, *args, **kwargs):
        return z4_dataset_builder._install_step4_albumentations_dependency(self, *args, **kwargs)
    def _ensure_step4_augmentation_profile(self, *args, **kwargs):
        return z4_dataset_builder._ensure_step4_augmentation_profile(self, *args, **kwargs)
    def _set_step4_augmentation_profile(self, *args, **kwargs):
        return z4_dataset_builder._set_step4_augmentation_profile(self, *args, **kwargs)
    def _refresh_step4_augmentation_summary(self, *args, **kwargs):
        return z4_dataset_builder._refresh_step4_augmentation_summary(self, *args, **kwargs)
    def _refresh_step4_creator_decision_summary(self, *args, **kwargs):
        return z4_dataset_builder._refresh_step4_creator_decision_summary(self, *args, **kwargs)
    def _clear_step4_augmentation_profile(self, *args, **kwargs):
        return z4_dataset_builder._clear_step4_augmentation_profile(self, *args, **kwargs)
    def _get_step4_augmentation_preview_images(self, *args, **kwargs):
        return z4_dataset_builder._get_step4_augmentation_preview_images(self, *args, **kwargs)
    def _estimate_step4_augmentation_train_pool_limit(self, *args, **kwargs):
        return z4_dataset_builder._estimate_step4_augmentation_train_pool_limit(self, *args, **kwargs)
    def _clamp_step4_augmentation_profile_to_pool(self, *args, **kwargs):
        return z4_dataset_builder._clamp_step4_augmentation_profile_to_pool(self, *args, **kwargs)
    def _open_step4_augmentation_modal(self, *args, **kwargs):
        return z4_dataset_builder._open_step4_augmentation_modal(self, *args, **kwargs)
    def _get_step4_augmentation_profile(self, *args, **kwargs):
        return z4_dataset_builder._get_step4_augmentation_profile(self, *args, **kwargs)
    def _build_step4_augmented_dataset_dir(self, *args, **kwargs):
        return z4_dataset_builder._build_step4_augmented_dataset_dir(self, *args, **kwargs)
    def _create_step4_augmented_dataset_variant(self, *args, **kwargs):
        return z4_dataset_builder._create_step4_augmented_dataset_variant(self, *args, **kwargs)
    def _apply_step4_dataset_postprocessing(self, *args, **kwargs):
        return z4_dataset_builder._apply_step4_dataset_postprocessing(self, *args, **kwargs)
    def _handle_step4_dataset_success_result(self, *args, **kwargs):
        return z4_dataset_builder._handle_step4_dataset_success_result(self, *args, **kwargs)
    def _handle_step4_dataset_failure_result(self, *args, **kwargs):
        return z4_dataset_builder._handle_step4_dataset_failure_result(self, *args, **kwargs)
    def _create_dataset_thread(self, *args, **kwargs):
        return z4_dataset_builder._create_dataset_thread(self, *args, **kwargs)
    def _split_dataset_thread(self, *args, **kwargs):
        return z4_dataset_builder._split_dataset_thread(self, *args, **kwargs)
    def _release_gpu_resources_before_training(self, *args, **kwargs):
        return z4_training_runtime._release_gpu_resources_before_training(self, *args, **kwargs)
    def _start_training(self, *args, **kwargs):
        return z4_training_runtime._start_training(self, *args, **kwargs)
    def _pause_training(self, *args, **kwargs):
        return z4_training_runtime._pause_training(self, *args, **kwargs)
    def _stop_training(self, *args, **kwargs):
        return z4_training_runtime._stop_training(self, *args, **kwargs)
    def _resolve_training_end_feedback(self, *args, **kwargs):
        return z4_training_runtime._resolve_training_end_feedback(self, *args, **kwargs)
    def _bind_trainer_callbacks(self, *args, **kwargs):
        return z4_training_runtime._bind_trainer_callbacks(self, *args, **kwargs)
    def _reload_history_snapshot_from_disk(self, *args, **kwargs):
        return z4_training_runtime._reload_history_snapshot_from_disk(self, *args, **kwargs)
    def _set_training_running_ui_state(self, *args, **kwargs):
        return z4_training_runtime._set_training_running_ui_state(self, *args, **kwargs)
    def _load_history(self, *args, **kwargs):
        return z4_training_runtime._load_history(self, *args, **kwargs)
    def _delete_selected(self, *args, **kwargs):
        return z4_training_runtime._delete_selected(self, *args, **kwargs)
    def _open_run_folder(self, *args, **kwargs):
        return z4_training_runtime._open_run_folder(self, *args, **kwargs)
    def _resume_selected_run(self, *args, **kwargs):
        return z4_training_runtime._resume_selected_run(self, *args, **kwargs)
    def _is_history_run_fine_tune_candidate(self, *args, **kwargs):
        return z4_training_runtime._is_history_run_fine_tune_candidate(self, *args, **kwargs)
    def _select_selected_run_as_fine_tune_base(self, *args, **kwargs):
        return z4_training_runtime._select_selected_run_as_fine_tune_base(self, *args, **kwargs)
    def _select_selected_ranking_run_as_fine_tune_base(self, *args, **kwargs):
        return z4_training_runtime._select_selected_ranking_run_as_fine_tune_base(self, *args, **kwargs)
    def _promote_selected_run_model_to_campaign(self, *args, **kwargs):
        return z4_training_runtime._promote_selected_run_model_to_campaign(self, *args, **kwargs)
    def _refresh_campaign_training_result_selector(self, *args, **kwargs):
        return z4_training_runtime._refresh_campaign_training_result_selector(self, *args, **kwargs)
    def _on_campaign_training_result_choice(self, *args, **kwargs):
        return z4_training_runtime._on_campaign_training_result_choice(self, *args, **kwargs)
    def _use_campaign_training_result_choice(self, *args, **kwargs):
        return z4_training_runtime._use_campaign_training_result_choice(self, *args, **kwargs)
    def _show_history_context_menu(self, *args, **kwargs):
        return z4_training_runtime._show_history_context_menu(self, *args, **kwargs)
    def _show_ranking_context_menu(self, *args, **kwargs):
        return z4_training_runtime._show_ranking_context_menu(self, *args, **kwargs)
    def _use_selected_ranking_model_as_campaign_result(self, *args, **kwargs):
        return z4_training_runtime._use_selected_ranking_model_as_campaign_result(self, *args, **kwargs)
    def _open_selected_ranking_run_details(self, *args, **kwargs):
        return z4_training_runtime._open_selected_ranking_run_details(self, *args, **kwargs)
    def _open_selected_ranking_model_folder(self, *args, **kwargs):
        return z4_training_runtime._open_selected_ranking_model_folder(self, *args, **kwargs)
    def _copy_selected_ranking_choice_label(self, *args, **kwargs):
        return z4_training_runtime._copy_selected_ranking_choice_label(self, *args, **kwargs)
    def _autofill_validation_inputs_from_run(self, *args, **kwargs):
        return z4_training_runtime._autofill_validation_inputs_from_run(self, *args, **kwargs)
    def _close_run_details_dialog(self, *args, **kwargs):
        return z4_training_runtime._close_run_details_dialog(self, *args, **kwargs)
    def _open_current_run_details_analysis(self, *args, **kwargs):
        return z4_training_runtime._open_current_run_details_analysis(self, *args, **kwargs)
    def _open_current_run_details_folder(self, *args, **kwargs):
        return z4_training_runtime._open_current_run_details_folder(self, *args, **kwargs)
    def _open_run_details_modal(self, *args, **kwargs):
        return z4_training_runtime._open_run_details_modal(self, *args, **kwargs)
    def _open_selected_run_details(self, *args, **kwargs):
        return z4_training_runtime._open_selected_run_details(self, *args, **kwargs)
    def _on_run_selected(self, *args, **kwargs):
        return z4_training_runtime._on_run_selected(self, *args, **kwargs)
    def _run_validation(self, *args, **kwargs):
        return z4_training_runtime._run_validation(self, *args, **kwargs)
    def _load_ranking(self, *args, **kwargs):
        return z4_training_runtime._load_ranking(self, *args, **kwargs)
