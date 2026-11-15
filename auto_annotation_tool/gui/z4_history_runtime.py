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
    if not _history_run_matches_active_storage(self, run):
        return False
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

def _get_visible_training_history_sources(self):
    """UI catalog = global histories + archived/project histories."""
    active = self.history
    if CAMPAIGN.get_active_project_name():
        return [(self.get_campaign_training_target(), active)]

    sources = []
    seen_dirs = set()

    def _add(target, directory):
        try:
            directory = Path(directory)
            key = str(directory.resolve()).lower()
        except Exception:
            return
        if key in seen_dirs or not directory.exists():
            return
        seen_dirs.add(key)
        try:
            source = active if Path(active.history_dir).resolve() == directory.resolve() else TrainingHistory(
                history_dir=directory, reconcile_on_load=False
            )
        except Exception:
            return
        sources.append((target, source))

    for target in ("plate", "char", "vehicle"):
        try:
            _add(target, Path(CONFIG.get_training_runs_dir(target)))
        except Exception:
            pass

    try:
        projects_root = Path(CONFIG.DIR_9_PROJECTS)
        if projects_root.exists():
            for project_root in sorted(
                (path for path in projects_root.iterdir() if path.is_dir()),
                key=lambda path: path.name.lower(),
            ):
                _add("", project_root / "5_training_runs")
    except Exception:
        pass

    return sources


def _history_row_ref(self, item_id):
    refs = getattr(self, "_history_row_refs", None)
    if isinstance(refs, dict):
        return refs.get(str(item_id))
    # Compatibility with pre-catalog views; never resolve a namespaced ID here.
    if str(item_id).startswith(("char:", "plate:", "vehicle:")):
        return None
    run = self.history.get_run(str(item_id))
    return {"run": run, "history": self.history, "target": ""} if run is not None else None


def _history_row_id_for_run(self, run, *, history=None):
    refs = getattr(self, "_history_row_refs", None)
    if not isinstance(refs, dict):
        return str(getattr(run, "id", "") or "")
    candidates = []
    for iid, ref in refs.items():
        candidate = ref["run"]
        if candidate is run:
            return iid
        if str(candidate.id) != str(run.id):
            continue
        if history is not None and Path(ref["history"].history_dir).resolve() != Path(history.history_dir).resolve():
            continue
        if history is None and str(getattr(candidate, "output_dir", "")) != str(getattr(run, "output_dir", "")):
            continue
        candidates.append(iid)
    return candidates[0] if len(candidates) == 1 else ""


def _history_run_matches_active_storage(self, run):
    if CAMPAIGN.get_active_project_name():
        return True
    if run is None:
        return False
    refs = getattr(self, "_history_row_refs", None)
    if isinstance(refs, dict):
        iid = _history_row_id_for_run(self, run)
        if not iid:
            return False
        source = refs[iid]["history"]
        return Path(source.history_dir).resolve() == Path(self.history.history_dir).resolve()
    # Before a history table exists, keep the existing target contract.
    try:
        target = self._infer_history_run_target(run)
        active_target = self._get_selected_training_target()
        if target in {"plate", "char", "vehicle"} and active_target in {"plate", "char", "vehicle"}:
            return target == active_target
    except Exception:
        pass
    return True


def _ensure_history_run_storage_for_action(self, run):
    if _history_run_matches_active_storage(self, run):
        return True
    iid = _history_row_id_for_run(self, run)
    refs = getattr(self, "_history_row_refs", {})
    target = refs.get(iid, {}).get("target", "") if isinstance(refs, dict) else ""
    label = self._format_history_run_target_label(target)
    messagebox.showwarning("Przełącz tor treningu", f"Ten run należy do toru {label}. Przełącz Z4 na ten tor przed wznowieniem lub dotrenowaniem.")
    return False


def _selected_run(self):
    selected = self.tree.selection()
    ref = _history_row_ref(self, selected[0]) if selected else None
    return ref["run"] if ref else None
