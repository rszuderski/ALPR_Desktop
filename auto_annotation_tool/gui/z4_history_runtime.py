#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 training history helpers extracted from tab_training.py."""

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
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
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
from . import z4_campaign_state
from . import z4_dataset_validation
from . import z4_layout_runtime
from . import z4_device_runtime
from . import z4_model_export
from . import z4_training_progress
from . import z4_ui_runtime
from . import z4_theme_runtime
from . import z4_tab_shell
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _is_history_run_resumable(run) -> bool:
    if run is None:
        return False
    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value not in {TrainingStatus.PAUSED.value, TrainingStatus.FAILED.value}:
        return False
    last_weights = str(getattr(run, "last_weights", "") or "").strip()
    if last_weights and Path(last_weights).exists():
        return True
    try:
        fallback_last = Path(str(getattr(run, "output_dir", "") or "")) / "train" / "weights" / "last.pt"
        return bool(fallback_last.exists())
    except Exception:
        return False

def _get_latest_campaign_resumable_run_id(self) -> str:
    if not CAMPAIGN.get_active_project_name():
        return ""

    try:
        runs = list(self.history.get_all_runs() or [])
    except Exception:
        runs = []

    for run in runs:
        if not self._is_history_run_resumable(run):
            continue
        if not self._does_history_run_match_active_campaign_target(run):
            continue
        return str(getattr(run, "id", "") or "").strip()
    return ""

def _is_history_run_resume_allowed(self, run) -> bool:
    if not self._is_history_run_resumable(run):
        return False
    if not self._does_history_run_match_active_campaign_target(run):
        return False
    if not CAMPAIGN.get_active_project_name():
        return True
    return bool(str(getattr(run, "id", "") or "").strip() == self._get_latest_campaign_resumable_run_id())

def _format_history_run_status_label(self, run) -> str:
    if run is None:
        return "-"
    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value == TrainingStatus.RUNNING.value:
        return "trwa trening"
    if status_value == TrainingStatus.PENDING.value:
        return "oczekuje"
    if status_value == TrainingStatus.PAUSED.value:
        if self._is_history_run_resume_allowed(run):
            return "paused (resume)"
        if self._is_history_run_resumable(run):
            return "paused (archiwalny)"
        return "paused (brak last.pt)"
    if status_value == TrainingStatus.FAILED.value:
        if self._is_history_run_resume_allowed(run):
            return "failed (resume)"
        if self._is_history_run_resumable(run):
            return "failed (archiwalny)"
        return "failed (brak last.pt)"
    if status_value == TrainingStatus.COMPLETED.value:
        lineage_mode = str(getattr(run, "lineage_mode", "") or "").strip().lower()
        if lineage_mode == "fine_tune":
            return "completed (dotren.)"
    return str(getattr(run, "status", "-") or "-")

def _does_history_run_match_active_campaign_target(self, run) -> bool:
    if run is None:
        return False
    if not CAMPAIGN.get_active_project_name():
        return True

    active_target = str(CAMPAIGN.get_iteration_target() or self.get_campaign_training_target() or "").strip().lower()
    if active_target not in {"char", "plate"}:
        return True

    run_target = ""
    try:
        infer_target = getattr(self, "_infer_history_run_target", None)
        if callable(infer_target):
            run_target = str(infer_target(run) or "").strip().lower()
    except Exception:
        run_target = ""
    if run_target not in {"char", "plate"}:
        try:
            run_target = str(
                self._infer_dataset_target(getattr(run, "dataset_path", "")) or ""
            ).strip().lower()
        except Exception:
            run_target = ""
    return bool(run_target == active_target)

def _device_to_ultralytics(self, device_str: str):
    effective_raw, _ = self._get_effective_training_device_profile(device_str)
    if effective_raw == "cpu":
        return "cpu"
    if effective_raw.startswith("cuda:"):
        try:
            return int(effective_raw.split(":")[1].split()[0])
        except Exception:
            return 0
    return "cpu"

def _open_path(self, path: Path):
    try:
        if os.name == "nt": os.startfile(str(path))
        else: webbrowser.open(path.as_uri())
    except Exception as e:
        messagebox.showinfo("Info", f"Nie mogę otworzyć: {path}\n\n{e}")

def _selected_run(self):
    sel = self.tree.selection()
    if not sel:
        return None

    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass

    item_id = str(sel[0] or "").strip()
    if not item_id:
        return None

    run = self.history.get_run(item_id)
    if run is not None:
        return run

    # Fallback dla starszych wpisów / ewentualnych niespójności.
    for db_key, run_obj in self.history.runs.items():
        if str(db_key or "").strip() == item_id:
            return run_obj
    return None
