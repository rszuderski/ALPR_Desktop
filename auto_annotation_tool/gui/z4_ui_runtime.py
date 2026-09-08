#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 UI dispatch, logs and operation-state helpers extracted from tab_training.py."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _close_ui_dispatch(self):
    with self._ui_dispatch_lock:
        self._ui_dispatch_closed = True
        while True:
            try:
                self._ui_dispatch_queue.get_nowait()
            except queue.Empty:
                break
    self._training_start_in_progress = False
    self._training_preflight_thread = None
    pending = getattr(self, "_ui_dispatch_after_id", None)
    self._ui_dispatch_after_id = None
    if pending:
        try:
            self.frame.after_cancel(pending)
        except tk.TclError:
            pass


def _ui_host_exists(self) -> bool:
    # This helper must only be called on the Tk thread.
    if getattr(self, "_ui_dispatch_closed", False):
        return False
    try:
        return bool(self.frame.winfo_exists())
    except tk.TclError:
        return False


def _ui(self, fn):
    if not callable(fn):
        return False

    if threading.current_thread() is threading.main_thread():
        if not _ui_host_exists(self):
            _close_ui_dispatch(self)
            return False

        def guarded_callback():
            if _ui_host_exists(self):
                fn()

        try:
            self.frame.after(0, guarded_callback)
            return True
        except tk.TclError:
            _close_ui_dispatch(self)
            return False

    with self._ui_dispatch_lock:
        if self._ui_dispatch_closed:
            return False
        self._ui_dispatch_queue.put_nowait(fn)
    return True

def _ensure_ui_dispatch_pump(self):
    if not hasattr(self, "_ui_dispatch_lock"):
        self._ui_dispatch_lock = threading.Lock()
        self._ui_dispatch_closed = False
    if not getattr(self, "_ui_dispatch_destroy_bound", False):
        def on_destroy(event):
            if event.widget is self.frame:
                _close_ui_dispatch(self)
        self.frame.bind("<Destroy>", on_destroy, add="+")
        self._ui_dispatch_destroy_bound = True
    if not _ui_host_exists(self):
        _close_ui_dispatch(self)
        return
    if getattr(self, "_ui_dispatch_after_id", None):
        return
    try:
        self._ui_dispatch_after_id = self.frame.after(20, self._drain_ui_dispatch_queue)
    except Exception:
        self._ui_dispatch_after_id = None

def _drain_ui_dispatch_queue(self):
    self._ui_dispatch_after_id = None
    if not _ui_host_exists(self):
        _close_ui_dispatch(self)
        return

    processed = 0
    started = time.perf_counter()
    for _ in range(200):
        try:
            fn = self._ui_dispatch_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break

        try:
            fn()
            processed += 1
        except Exception:
            pass
        if not _ui_host_exists(self):
            _close_ui_dispatch(self)
            return
        if time.perf_counter() - started >= 0.008:
            break

    has_pending = False
    try:
        has_pending = not self._ui_dispatch_queue.empty()
    except Exception:
        has_pending = False
    active = bool(
        has_pending
        or processed > 0
        or getattr(getattr(self, "trainer", None), "is_training", False)
        or getattr(self, "is_processing", False)
        or getattr(self, "_training_start_in_progress", False)
        or getattr(self, "dataset_build_is_running", False)
        or getattr(self, "dataset_split_is_running", False)
        or getattr(self, "val_is_running", False)
        or getattr(self, "rank_is_running", False)
    )
    delay_ms = 20 if active else 250
    try:
        self._ui_dispatch_after_id = self.frame.after(delay_ms, self._drain_ui_dispatch_queue)
    except Exception:
        self._ui_dispatch_after_id = None

def _append_to_step4_process_console(self, text: str, *, autoscroll_if_at_end: bool = True) -> None:
    widget = getattr(self, "train_log_console", None)
    if widget is None:
        return

    payload = self._sanitize_training_text(text)
    if not payload:
        return

    try:
        state_before = self._capture_text_widget_view_state(widget)
        follow_end = bool(autoscroll_if_at_end and state_before.get("view_at_end") and state_before.get("insert_at_end"))
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, payload)
        if follow_end:
            widget.see(tk.END)
        else:
            self._restore_text_widget_view_state(widget, state_before)
    except Exception:
        pass
    finally:
        try:
            widget.config(state=tk.DISABLED)
        except Exception:
            pass

def _append_train_log(self, message: str, mirror_global: bool = True):
    """Bezpieczne dopisywanie linii do konsoli treningu z dowolnego wątku."""
    text = self._sanitize_training_text(message)
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"

    if mirror_global:
        self._ui(
            lambda message=text.rstrip("\n"): (
                self.app.append_global_terminal(message, source="Z4")
                if hasattr(self.app, "append_global_terminal")
                else None
            )
        )

    def update():
        try:
            self._append_to_step4_process_console(text)
        except Exception:
            pass
    self._ui(update)

def _append_ranking_log(self, message: str):
    text = str(message or "").strip()
    if not text:
        return
    self._append_train_log(f"[RANKING] {text}")

def _set_training_widget_text(self, widget, text) -> None:
    if widget is None:
        return
    try:
        widget.configure(text=self._sanitize_training_text(text))
    except Exception:
        pass

def _set_training_stringvar_text(self, variable, text) -> None:
    if variable is None:
        return
    try:
        variable.set(self._sanitize_training_text(text))
    except Exception:
        pass

def _start_ranking_watchdog(self, stage: str):
    self._stop_ranking_watchdog()
    self._touch_ranking_watchdog(stage)
    try:
        self._rank_watchdog_job = self.frame.after(3000, self._poll_ranking_watchdog)
    except Exception:
        self._rank_watchdog_job = None

def _touch_ranking_watchdog(self, stage: str | None = None):
    self._rank_watchdog_last_touch = time.perf_counter()
    if stage is not None:
        self._rank_watchdog_stage = str(stage or "").strip()

def _stop_ranking_watchdog(self):
    job = getattr(self, "_rank_watchdog_job", None)
    self._rank_watchdog_job = None
    if job is not None:
        try:
            self.frame.after_cancel(job)
        except Exception:
            pass

def _poll_ranking_watchdog(self):
    self._rank_watchdog_job = None
    if not (getattr(self, "rank_is_running", False) or getattr(self, "rank_cancel_requested", False)):
        return

    now = time.perf_counter()
    last_touch = float(getattr(self, "_rank_watchdog_last_touch", 0.0) or 0.0)
    stalled_for = max(0.0, now - last_touch)
    warn_after = float(getattr(self, "_rank_watchdog_warn_after_s", 15.0) or 15.0)
    repeat_after = float(getattr(self, "_rank_watchdog_repeat_s", 15.0) or 15.0)

    if stalled_for >= warn_after:
        last_notice = float(getattr(self, "_rank_watchdog_last_notice", 0.0) or 0.0)
        if (last_notice <= 0.0) or ((now - last_notice) >= repeat_after):
            stage = str(getattr(self, "_rank_watchdog_stage", "") or "nieznany etap")
            seconds = int(round(stalled_for))
            self._rank_watchdog_last_notice = now
            if getattr(self, "rank_cancel_requested", False):
                self._append_ranking_log(
                    f"Watchdog: nadal czekam na zatrzymanie od {seconds}s | ostatni etap: {stage}."
                )
            else:
                self._append_ranking_log(
                    f"Watchdog: brak nowego postepu od {seconds}s | etap: {stage}. "
                    "Możesz poczekac albo kliknac 'Anuluj ranking'."
                )
            self._set_ranking_ui_state(
                status=(
                    f"Czekam na zatrzymanie od {seconds}s | {stage}"
                    if getattr(self, "rank_cancel_requested", False)
                    else f"Bez nowego postepu od {seconds}s | {stage}"
                ),
                status_color="#d35400",
                cancel_enabled=not getattr(self, "rank_cancel_requested", False),
            )

    if getattr(self, "rank_is_running", False) or getattr(self, "rank_cancel_requested", False):
        try:
            self._rank_watchdog_job = self.frame.after(3000, self._poll_ranking_watchdog)
        except Exception:
            self._rank_watchdog_job = None

def _get_active_step4_operation_label(self) -> str:
    if bool(getattr(self.trainer, "is_training", False)):
        return "trening modelu"
    if bool(getattr(self, "dataset_build_is_running", False)):
        return "przygotowanie datasetu tablic"
    if bool(getattr(self, "dataset_split_is_running", False)):
        return "przygotowanie datasetu znaków"
    if bool(getattr(self, "val_is_running", False)):
        return "walidacja modelu"
    if bool(getattr(self, "rank_is_running", False)):
        return "ranking modeli"
    if bool(getattr(self, "is_processing", False)):
        return "operacja Z4"
    return ""

def _step4_has_active_operation(self) -> bool:
    return bool(self._get_active_step4_operation_label())

def _begin_step4_operation(self, owner: str, label: str) -> bool:
    app = getattr(self, "app", None)
    if app is not None and hasattr(app, "try_begin_exclusive_operation"):
        ok, busy_message = app.try_begin_exclusive_operation(owner, label)
        if not ok:
            messagebox.showinfo("Proces w toku", busy_message)
            return False
    else:
        try:
            self.app.set_processing(True)
        except Exception:
            pass
    self.is_processing = True
    if owner != "z4.training.run":
        try:
            self._refresh_training_start_state()
        except Exception:
            pass
    return True

def _end_step4_operation(self, owner: str):
    self.is_processing = False
    app = getattr(self, "app", None)
    if app is not None and hasattr(app, "end_exclusive_operation"):
        app.end_exclusive_operation(owner)
    else:
        try:
            self.app.set_processing(False)
        except Exception:
            pass
    try:
        self._refresh_training_start_state()
    except Exception:
        pass

def _set_ranking_ui_state(
    self,
    *,
    status: str | None = None,
    status_color: str | None = None,
    button_text: str | None = None,
    cancel_enabled: bool | None = None,
    preparing: bool | None = None,
    progress_value: float | None = None,
):
    def update():
        if button_text is not None and hasattr(self, "btn_run_rank"):
            try:
                self.btn_run_rank.config(text=button_text)
            except Exception:
                pass

        if cancel_enabled is not None and hasattr(self, "btn_cancel_rank"):
            try:
                self.btn_cancel_rank.config(state=(tk.NORMAL if cancel_enabled else tk.DISABLED))
            except Exception:
                pass

        if status is not None:
            for attr_name in ("rank_status", "rank_advanced_status"):
                label = getattr(self, attr_name, None)
                if label is None:
                    continue
                try:
                    kwargs = {"text": status}
                    if status_color is not None:
                        kwargs["foreground"] = status_color
                    label.configure(**kwargs)
                except Exception:
                    pass

        progress = getattr(self, "rank_progress", None)
        if progress is not None and preparing is not None:
            try:
                progress.stop()
            except Exception:
                pass
            try:
                progress.configure(mode="indeterminate" if preparing else "determinate")
            except Exception:
                pass
            if preparing:
                try:
                    progress.start(12)
                except Exception:
                    pass

        if progress_value is not None:
            try:
                self.rank_progress_var.set(float(progress_value))
            except Exception:
                pass

    self._ui(update)

def _attach_training_log_handlers(self):
    """Przekierowuje logi aplikacji i Ultralytics do konsoli treningu w GUI."""
    if getattr(self, "_training_log_handlers_attached", False):
        return

    import logging

    class GuiLogHandler(logging.Handler):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def emit(self, record):
            try:
                msg = self.format(record)
                if msg:
                    self.owner._append_train_log(msg, mirror_global=False)
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    raw_fmt = logging.Formatter("%(message)s")

    # Handler dla loggera aplikacji.
    self._gui_app_log_handler = GuiLogHandler(self)
    self._gui_app_log_handler.setFormatter(fmt)
    logger.addHandler(self._gui_app_log_handler)

    # Handler dla loggera Ultralytics.
    self._gui_yolo_log_handler = GuiLogHandler(self)
    self._gui_yolo_log_handler.setFormatter(raw_fmt)

    self._ultralytics_logger = logging.getLogger("ultralytics")
    self._ultralytics_logger.addHandler(self._gui_yolo_log_handler)

    self._training_log_handlers_attached = True
