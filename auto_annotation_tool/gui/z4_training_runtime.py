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
from ..training import (
    YOLOPoseTrainer,
    TrainingHistory,
    TrainingStatus,
    DatasetCreator,
    DatasetSplitter,
    format_resource_sample_line,
)
from ..ranking import ModelRanking, format_ranking_model_label
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .dataset_display import build_dataset_display_ref
from .model_display import build_model_display_ref
from .run_display import build_run_display_ref
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    complete_campaign_step4_if_needed,
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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)


def _step4_finish_gate_display_id() -> str:
    return "T06"

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None

def _detach_gpu_resources_before_training(self) -> list:
    """Detach GUI-owned models while the exclusive training lock is held."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Models must be detached on the GUI thread")
    detached = []
    app_tabs = getattr(getattr(self, "app", None), "tabs", {}) or {}
    tab = app_tabs.get("annotation")
    annotator = getattr(tab, "annotator", None)
    if annotator is not None:
        detached.append(annotator)
        tab.annotator = None
    return detached


def _release_detached_gpu_resources(models) -> None:
    """Only detached model objects and CUDA/Python cleanup may run in this worker."""
    while models:
        model = models.pop()
        try:
            model.unload_models()
        finally:
            del model
    cleanup_gpu_memory()


def _release_gpu_resources_before_training(self):
    _release_detached_gpu_resources(_detach_gpu_resources_before_training(self))

def _set_training_preparing_ui_state(self, *, text: str = "Przygotowanie treningu...", progress: float = 5.0):
    entering = not bool(getattr(self, "_training_start_in_progress", False))
    if entering:
        self._last_training_preflight_stage = None
    self._training_start_in_progress = True
    for attr, state in (
        ("btn_start_train", tk.DISABLED),
        ("btn_pause_train", tk.DISABLED),
        ("btn_stop_train", tk.DISABLED),
    ):
        try:
            getattr(self, attr).configure(state=state)
        except Exception:
            pass
    try:
        self._set_train_progress_values(overall=max(0.0, min(100.0, float(progress))), epoch=0.0)
    except Exception:
        pass
    if entering:
        try:
            self._refresh_training_cockpit(ready=False)
        except Exception:
            pass
    try:
        self._set_training_widget_text(self.train_progress_label, text)
        self.train_progress_label.configure(foreground="#d35400")
    except Exception:
        pass

def _update_training_preflight_progress(self, stage: str, progress: float, detail: str = ""):
    stage_text = str(stage or "Przygotowanie treningu").strip()
    detail_text = str(detail or "").strip()
    label = f"Przygotowanie treningu: {stage_text}"
    if detail_text:
        label = f"{label} | {detail_text}"
    _set_training_preparing_ui_state(self, text=label, progress=progress)
    if getattr(self, "_last_training_preflight_stage", None) != stage_text:
        self._last_training_preflight_stage = stage_text
        try:
            self._append_train_log(f"[PREFLIGHT] {stage_text}" + (f" | {detail_text}" if detail_text else ""))
        except Exception:
            pass

def _finish_training_preflight_failure(self, *, title: str, message: str):
    self._training_start_in_progress = False
    self._training_preflight_thread = None
    try:
        self._end_step4_operation("z4.training.run")
    except Exception:
        pass
    self._pending_campaign_model_type = None
    try:
        self._set_train_progress_values(overall=0.0, epoch=0.0)
    except Exception:
        pass
    try:
        self.train_progress_label.configure(
            text="Nie udało się uruchomić treningu.",
            foreground="#c0392b",
        )
    except Exception:
        pass
    try:
        self._append_train_log("[START] Trening nie wystartował. Sprawdź dataset, model startowy i log powyżej.")
    except Exception:
        pass
    messagebox.showerror(title, message)
    try:
        self._refresh_training_start_state()
    except Exception:
        pass

def _start_training(self):
    if bool(getattr(self, "_training_start_in_progress", False)):
        return
    if not YOLO_AVAILABLE:
        return messagebox.showerror("Błąd", "Brak ultralytics.")

    try:
        pinned_state = dict(self._get_pinned_step4_result_state() or {})
    except Exception:
        pinned_state = {}
    if pinned_state:
        return messagebox.showinfo(
            "Wynik bramki przypięty",
            (
                "Ta bramka ma już przypięty model wynikowy.\n\n"
                "Aby uruchomić nowy trening albo zmienić konfigurację, najpierw użyj `Odepnij wynik`."
            ),
        )

    try:
        self._clear_step4_guidance()
    except Exception:
        pass

    self._set_step4_process_console_text("Uruchamianie treningu...\n")
    self._latest_training_metrics = {}
    self._current_training_metric_history = []
    self._set_training_metric_interpretation("Najlepsza epoka tego runu pojawi się po pierwszej zakończonej epoce.")
    self._last_training_resource_log_at = 0.0
    try:
        self._set_training_widget_text(
            getattr(self, "train_resource_label", None),
            "Zasoby w czasie treningu",
        )
        self._set_training_resource_sample(None)
    except Exception:
        pass

    light_started = time.perf_counter()
    source_state = z4_training_metrics._validate_training_source_lightweight(self)
    logger.info("[PREFLIGHT] gui_light_validation %.3f s", time.perf_counter() - light_started)
    if not bool(source_state.get("ok")):
        validation_msg = str(source_state.get("message") or "Dataset niegotowy do treningu.")
        self._append_train_log(f"[WALIDACJA] {validation_msg}")
        self.train_progress_label.configure(
            text="Dataset wymaga poprawy przed treningiem.",
            foreground="#c0392b"
        )
        return messagebox.showerror("Dataset niegotowy do treningu", validation_msg)

    yaml_path = Path(source_state["yaml_path"])
    dataset_root = Path(source_state["dataset_root"])

    # Rozpoznaj typ datasetu na podstawie zawartości data.yaml.
    try:
        cfg = safe_load_yaml(yaml_path)
        is_pose_dataset = "kpt_shape" in cfg

        self._current_training_dataset_is_pose = bool(is_pose_dataset)
        inferred_target = self._infer_dataset_target(str(dataset_root)) or ("plate" if is_pose_dataset else "char")
        selected_target = self._get_selected_training_target()

        if not CAMPAIGN.get_active_project_name():
            if inferred_target != selected_target:
                selected_label = self._format_training_target_label(selected_target)
                inferred_label = self._format_training_target_label(inferred_target)
                return messagebox.showerror(
                    "Niezgodny tor treningu",
                    "Wybrany tor treningu nie pasuje do wskazanego datasetu.\n\n"
                    f"Wybrany tor: {selected_label}\n"
                    f"Rozpoznany dataset: {inferred_label}\n\n"
                    "Zmień tor treningu albo wskaż dataset zgodny z tym wyborem."
                )
            # Storage was bound when the target/variant was selected. Rebinding here
            # would reload history and replace the trainer on every Start.

        if CAMPAIGN.get_active_project_name() and not is_pose_dataset:
            self._pending_campaign_model_type = "char"
        else:
            self._pending_campaign_model_type = None
    except Exception as e:
        return messagebox.showerror("Błąd", f"Nie udało się odczytać data.yaml:\n{e}")

    dataset_path = str(dataset_root)
    base_key = self.base_model_var.get().strip()
    base_model = self.base_custom_var.get().strip() if self._is_custom_base_model_key(base_key) else base_key
    base_model_display = self._resolve_selected_training_base_model_display()
    _base_model_info_path, base_model_info = self._resolve_selected_training_base_model_info(lightweight=True)
    device = self._device_to_ultralytics(self.device_var.get())

    selection_ok, _selection_message = self._validate_training_base_model_target_compatibility(
        target=selected_target,
        show_dialog=True,
        lightweight=True,
    )
    if not selection_ok:
        return

    # Rozpoznaj, czy wybrany model jest modelem pose.
    model_task = z4_training_metrics._training_base_model_task_lightweight(self, base_key, base_model)

    # Zablokuj niezgodne pary dataset-model przed startem treningu.
    if model_task is not None and is_pose_dataset and model_task != "pose":
        return messagebox.showerror(
            "Niezgodność typu treningu",
            "Wybrany dataset jest typu POSE (z keypointami), ale model startowy NIE jest modelem pose.\n\n"
            "Wybierz model z dopiskiem '-pose'."
        )

    if model_task is not None and not is_pose_dataset and model_task != "detect":
        return messagebox.showerror(
            "Niezgodność typu treningu",
            "Wybrany dataset jest typu DETECT, ale model startowy jest typu POSE.\n\n"
            "Dla znaków tablic wybierz zwykły model detect, np. 'yolo11n' lub 'yolo11s'."
        )

    fine_tune_parent_run = None
    fine_tune_run_label = ""
    fine_tune_metadata = {}
    try:
        fine_tune_parent_run = self._resolve_step4_fine_tune_parent_run()
    except Exception:
        fine_tune_parent_run = None
    if fine_tune_parent_run is not None:
        try:
            parent_best = self._resolve_history_run_best_weights(fine_tune_parent_run)
        except Exception:
            parent_best = None
        if parent_best is not None and Path(parent_best).exists():
            fine_tune_run_label = self._format_training_model_run_label(fine_tune_parent_run)
            fine_tune_metadata = {
                "lineage_mode": "fine_tune",
                "parent_run_id": str(getattr(fine_tune_parent_run, "id", "") or "").strip(),
                "parent_model_path": str(parent_best),
                "parent_model_name": Path(parent_best).name,
                "parent_model_target": selected_target,
                "parent_dataset_path": str(getattr(fine_tune_parent_run, "dataset_path", "") or "").strip(),
                "parent_best_map50": float(getattr(fine_tune_parent_run, "best_map50", 0.0) or 0.0),
                "parent_best_map50_95": float(getattr(fine_tune_parent_run, "best_map50_95", 0.0) or 0.0),
            }

    try:
        requested_imgsz = self._safe_training_int_value("imgsz_var", default=640, minimum=0)
    except Exception:
        requested_imgsz = 0
    if is_pose_dataset and requested_imgsz < 256:
        return messagebox.showerror(
            "Zbyt mała rozdzielczość wejściowa",
            "Dla treningu POSE rozdzielczość wejściowa musi mieć co najmniej 256 px.\n\n"
            "Praktyczny bezpieczny start dla tego projektu to zwykle 512 albo 640."
        )

    # Zapisz czytelny nagłówek sesji w terminalu procesu.
    selected_device_display = self._normalize_training_device_choice(self.device_var.get())
    effective_device_raw, effective_device_profile = self._get_effective_training_device_profile(selected_device_display)
    if effective_device_profile is not None:
        effective_device_desc = (
            f"{effective_device_profile.get('name', effective_device_raw)} "
            f"({float(effective_device_profile.get('memory_gb', 0.0) or 0.0):.1f} GB VRAM)"
        )
    else:
        effective_device_desc = "CPU"

    gpu_capacity_block_reason = self._get_training_gpu_capacity_block_reason(
        is_pose_dataset=bool(is_pose_dataset),
        base_model_info=base_model_info,
        effective_device_profile=effective_device_profile,
    )
    if gpu_capacity_block_reason and device != "cpu":
        self._append_train_log("[BLOKADA STARTU] " + gpu_capacity_block_reason.replace("\n", " "))
        self.train_progress_label.configure(
            text="Wybrany model jest zbyt ciężki dla aktywnego GPU.",
            foreground="#c0392b"
        )
        return messagebox.showerror(
            "Model zbyt ciężki dla GPU",
            gpu_capacity_block_reason
        )

    self._append_train_log("=" * 70)
    self._append_train_log(f"START TRENINGU | Nazwa: {self.name_var.get()}")
    self._append_train_log(f"Dataset: {dataset_path}")
    self._append_train_log(f"Wybór w polu 'Model startowy treningu (.pt)': {base_model_display}")
    self._append_train_log(f"Model przekazany do treningu: {base_model}")
    if fine_tune_metadata:
        self._append_train_log(
            f"Tryb runu: dotrenowanie | rodzic: {fine_tune_run_label or fine_tune_metadata.get('parent_run_id', '-')}"
        )
    self._append_train_log(
        f"Urządzenie: {selected_device_display} -> {effective_device_desc} | backend Ultralytics: {device}"
    )
    self._append_train_log(
        f"Epoki: {self._safe_training_int_value('epochs_var', default=100, minimum=1)} | "
        f"Rozmiar partii: {self._safe_training_int_value('batch_var', default=16, minimum=1)} | "
        f"Rozdzielczość wejściowa: {self._safe_training_int_value('imgsz_var', default=640, minimum=32)} | "
        f"Współczynnik uczenia: {self._safe_training_float_value('lr0_var', default=0.01, minimum=0.0001)}"
    )
    self._append_train_log("Liczebność zbiorów train/val/test zostanie sprawdzona w tle.")
    self._append_train_log("=" * 70)

    if not self._begin_step4_operation("z4.training.run", "Z4: przygotowanie treningu"):
        return

    request = {
        "name": str(self.name_var.get() or ""),
        "dataset_path": dataset_path,
        "base_model": base_model,
        "epochs": self._safe_training_int_value("epochs_var", default=100, minimum=1),
        "batch_size": self._safe_training_int_value("batch_var", default=16, minimum=1),
        "img_size": self._safe_training_int_value("imgsz_var", default=640, minimum=32),
        "device": device,
        "lr0": self._safe_training_float_value("lr0_var", default=0.01, minimum=0.0001),
        "training_target": selected_target,
        "validate_custom_model": model_task is None,
        **dict(fine_tune_metadata or {}),
    }
    _set_training_preparing_ui_state(self, text="Przygotowanie treningu: waliduję i zamrażam dane...", progress=5.0)
    detached_models = _detach_gpu_resources_before_training(self)
    trainer = self.trainer

    def progress_callback(stage, progress, detail=""):
        safe_stage = str(stage or "").strip()
        safe_detail = str(detail or "").strip()
        try:
            safe_progress = float(progress)
        except Exception:
            safe_progress = 0.0
        self._ui(
            lambda s=safe_stage, p=safe_progress, d=safe_detail: _update_training_preflight_progress(
                self,
                s,
                p,
                d,
            )
        )

    def finish_success(run_id):
        self._training_start_in_progress = False
        self._training_preflight_thread = None
        self.current_run_id = run_id
        z4_training_metrics._apply_preflight_dataset_validation(
            self, getattr(trainer, "_last_preflight_dataset_validation", None),
        )
        self._last_training_completion_summary_run_id = None
        try:
            self._remember_campaign_plate_training_source(dataset_root)
        except Exception as e:
            logger.debug(f"Nie udało się zapamiętać źródła treningu tablic: {e}")
        self._step4_campaign_finish_ready = False
        try:
            if CAMPAIGN.get_active_project_name():
                CAMPAIGN.set_step4_finish_state(False)
        except Exception:
            pass
        self._set_train_progress_values(overall=0.0, epoch=0.0)
        self._reset_training_runtime_progress()
        self._set_train_live_metrics(None)
        try:
            self._set_training_widget_text(getattr(self, "train_resource_label", None), "Zasoby w czasie treningu")
            self._set_training_resource_sample(None)
        except Exception:
            pass
        try:
            run_display = build_run_display_ref({"run_id": run_id}, kind_hint="training").id
        except Exception:
            run_display = str(run_id or "").strip()
        self._set_training_running_ui_state(
            run_id,
            status_text=f"Uruchomiono run treningowy: {run_display}",
        )
        self._training_started_monotonic = time.perf_counter()
        self._training_started_wall_clock = datetime.datetime.now()

        try:
            self._remember_campaign_training_run_in_registry(
                run_id=str(run_id or "").strip(),
                status=TrainingStatus.RUNNING.value,
                target=self.get_campaign_training_target(),
            )
        except Exception:
            pass

        try:
            self._refresh_step4_campaign_navigation_ui()
        except Exception:
            pass

        if CAMPAIGN.get_active_project_name():
            self._pending_campaign_model_type = self.get_campaign_training_target()
            try:
                label = "znaków" if self._pending_campaign_model_type == "char" else "tablic"
                self._append_train_log(
                    f"[TARGET] Ten trening utworzy kandydata na model {label}. "
                    "Wynikiem bramki stanie się dopiero po jawnym wyborze w sekcji wyniku."
                )
            except Exception:
                pass
        else:
            self._pending_campaign_model_type = None

        if self._training_completion_poll_job is not None:
            try:
                self.frame.after_cancel(self._training_completion_poll_job)
            except Exception:
                pass
            self._training_completion_poll_job = None

        self._training_completion_poll_job = self.frame.after(3000, self._poll_training_completion)

    def finish_failure(error_text=""):
        z4_training_metrics._apply_preflight_dataset_validation(
            self, getattr(trainer, "_last_preflight_dataset_validation", None),
        )
        message = (
            "Trening nie wystartował.\n\n"
            "Sprawdź poprawność datasetu, modelu startowego i log w terminalu procesu."
        )
        if str(error_text or "").strip():
            message = f"{message}\n\nSzczegóły:\n{str(error_text).strip()}"
        _finish_training_preflight_failure(
            self,
            title="Nie udało się uruchomić treningu",
            message=message,
        )

    def worker():
        run_id = None
        error_text = ""
        try:
            _release_detached_gpu_resources(detached_models)
            run_id = trainer.start_training(**request, progress_callback=progress_callback)
            validation = getattr(trainer, "_last_preflight_dataset_validation", None)
            if not run_id and isinstance(validation, dict) and not validation.get("ok"):
                error_text = str(validation.get("message") or "")
        except Exception as e:
            logger.exception("Nie udało się wystartować treningu")
            error_text = str(e)

        delivered = self._ui(
            lambda rid=run_id, err=error_text: (
                finish_success(rid) if rid else finish_failure(err)
            )
        )
        if delivered is False and run_id:
            trainer.shutdown()

    thread = threading.Thread(target=worker, name="Z4TrainingPreflight", daemon=True)
    self._training_preflight_thread = thread
    try:
        thread.start()
    except Exception as exc:
        finish_failure(str(exc))

def _pause_training(self):
    self.trainer.pause_training()
    _update_training_history_progress(self, run_id=getattr(self, "current_run_id", None))
    self.btn_pause_train.configure(state=tk.DISABLED)
    self.btn_stop_train.configure(state=tk.DISABLED)
    self.train_progress_label.configure(foreground="#d35400")
    self._set_training_widget_text(self.train_progress_label, "Wstrzymywanie treningu...")

def _stop_training(self):
    self.trainer.stop_training()
    _update_training_history_progress(self, run_id=getattr(self, "current_run_id", None))
    self.btn_pause_train.configure(state=tk.DISABLED)
    self.btn_stop_train.configure(state=tk.DISABLED)
    self.train_progress_label.configure(foreground="#c0392b")
    self._set_training_widget_text(self.train_progress_label, "Zatrzymywanie treningu...")

def _resolve_training_end_feedback(self, success: bool, msg: str) -> tuple[str, str]:
    normalized = str(msg or "").strip().lower()
    if success:
        return "Trening zakończony.", "#2c3e50"
    if normalized == "wstrzymano":
        return "Trening wstrzymany.", "#d35400"
    if normalized == "zatrzymano":
        return "Trening zatrzymany.", "#c0392b"
    if self._is_memory_failure_text(msg):
        return "Trening przerwany przez błąd pamięci.", "#c0392b"
    return "Trening zakończony błędem.", "#c0392b"

def _bind_trainer_callbacks(self):
    def on_batch_progress(epoch, batch_idx, total_batches, batch_pct):
        run = self.trainer.current_run
        if not run:
            return
        now = time.perf_counter()
        try:
            batch_idx_int = int(batch_idx)
            total_batches_int = int(total_batches)
        except Exception:
            batch_idx_int = 0
            total_batches_int = 0
        terminal_batch = bool(total_batches_int > 0 and batch_idx_int >= total_batches_int)
        last_emit = float(getattr(self, "_last_training_batch_ui_emit_at", 0.0) or 0.0)
        if not terminal_batch and (now - last_emit) < 0.18:
            return
        self._last_training_batch_ui_emit_at = now

        overall_pct = (((max(1, int(epoch)) - 1) + (float(batch_pct) / 100.0)) / max(1, int(run.epochs))) * 100.0
        if batch_idx_int > 0 and total_batches_int > 0:
            status_text = f"Trwa trening: Epoka {epoch}/{run.epochs} | partia {batch_idx_int}/{total_batches_int}"
        else:
            status_text = f"Trwa trening: Epoka {epoch}/{run.epochs} | przygotowanie partii"

        started = getattr(self, "_training_started_monotonic", None)
        if started is None:
            self._training_started_monotonic = time.perf_counter()
            self._training_started_wall_clock = datetime.datetime.now()
            started = self._training_started_monotonic
        eta_seconds = None
        try:
            overall_fraction = max(0.0, min(1.0, float(overall_pct) / 100.0))
            if started is not None and overall_fraction >= 0.01:
                elapsed = max(0.001, time.perf_counter() - float(started))
                eta_seconds = max(0.0, (elapsed / overall_fraction) - elapsed)
        except Exception:
            eta_seconds = None

        def update_ui():
            _update_training_history_progress(self, run_id=run.id, epoch=epoch, total_epochs=run.epochs)
            self._set_train_progress_values(overall=overall_pct, epoch=batch_pct)
            self._update_training_progress_meta(
                epoch=int(epoch),
                total_epochs=int(run.epochs),
                batch_idx=batch_idx_int,
                total_batches=total_batches_int,
                overall_pct=float(overall_pct),
                epoch_pct=float(batch_pct),
                eta_seconds=eta_seconds,
            )
            self._set_training_widget_text(self.train_progress_label, status_text)

        self._ui(update_ui)

    def on_epoch(epoch, metrics):
        run = self.trainer.current_run
        if not run: return
        pct = (epoch / max(1, run.epochs)) * 100.0
        epoch_metrics = dict(metrics or {})
        try:
            epoch_metrics["epoch"] = int(epoch)
        except Exception:
            epoch_metrics["epoch"] = epoch
        self._latest_training_metrics = dict(epoch_metrics)
        try:
            metric_history = list(getattr(self, "_current_training_metric_history", []) or [])
            epoch_key = str(epoch_metrics.get("epoch", "") or "").strip()
            if epoch_key:
                metric_history = [
                    row for row in metric_history
                    if str((row or {}).get("epoch", "") or "").strip() != epoch_key
                ]
            metric_history.append(dict(epoch_metrics))
            self._current_training_metric_history = metric_history
        except Exception:
            self._current_training_metric_history = [dict(epoch_metrics)]

        # Pobieranie wyników mAP
        map50 = metrics.get('map50', 0)
        map50_95 = metrics.get('map50_95', 0)
        loss = metrics.get('loss', 0)
        interpretation = self._build_training_best_epoch_summary(epoch_metrics, epoch=epoch)

        # Formatowanie logu na żywo
        if self.get_campaign_training_target() == "plate":
            pose_map50 = self._metric_float(metrics.get("pose_map50", map50))
            pose_map50_95 = self._metric_float(metrics.get("pose_map50_95", map50_95))
            box_map50 = self._metric_float(metrics.get("box_map50", map50))
            box_map50_95 = self._metric_float(metrics.get("box_map50_95", map50_95))
            log_line = (
                f"Epoka {epoch}/{run.epochs} | Strata(Loss): {loss:.3f} | "
                f"Box mAP50: {box_map50:.3f} | Box mAP50-95: {box_map50_95:.3f} | "
                f"Pose mAP50: {pose_map50:.3f} | Pose mAP50-95: {pose_map50_95:.3f}\n"
            )
        else:
            log_line = f"Epoka {epoch}/{run.epochs} | Strata(Loss): {loss:.3f} | mAP50: {map50:.3f} | mAP50-95: {map50_95:.3f}\n"

        # Aktualizacja UI w głównym wątku
        def update_ui():
            _update_training_history_progress(self, run_id=run.id, epoch=epoch, total_epochs=run.epochs)
            self._set_train_progress_values(overall=pct, epoch=100.0)
            self._update_training_progress_meta(
                epoch=int(epoch),
                total_epochs=int(run.epochs),
                batch_idx=int(getattr(self, "_training_last_total_batches", 0) or 0),
                total_batches=int(getattr(self, "_training_last_total_batches", 0) or 0),
                overall_pct=float(pct),
                epoch_pct=100.0,
                eta_seconds=0.0,
            )
            self._set_training_widget_text(self.train_progress_label, f"Trening trwa: zakończono epokę {epoch}/{run.epochs}")
            self._set_training_metric_interpretation(interpretation)
            self._set_train_live_metrics(metrics)
            self._append_training_metric_table_to_global(epoch, run.epochs, metrics)
            self._append_to_step4_process_console(log_line)
            self._append_to_step4_process_console(f"{interpretation}\n")

        self._ui(update_ui)

    def on_end(success, msg):
        safe_msg = self._sanitize_training_text(msg)
        try:
            self._end_step4_operation("z4.training.run")
        except Exception as e:
            logger.debug(f"Nie udalo sie zwolnic blokady Z4 po zakonczeniu treningu: {e}")
        end_line = f"[KONIEC] {'SUKCES' if success else 'BŁĄD/STOP'} | {safe_msg}"
        self._append_train_log(end_line)
        if not success:
            for gpu_line in self._build_training_gpu_memory_lines(
                getattr(getattr(self, "trainer", None), "current_run", None).device
                if getattr(getattr(self, "trainer", None), "current_run", None) is not None
                else None
            ):
                self._append_train_log(f"[GPU] {gpu_line}")
        final_interpretation = self._build_training_metric_interpretation(getattr(self, "_latest_training_metrics", {}))
        if getattr(self, "_latest_training_metrics", {}):
            self._append_train_log(f"[OCENA] {final_interpretation}")
        self._end_step4_operation("z4.training.run")
        if not success and CAMPAIGN.get_active_project_name():
            self._step4_campaign_finish_ready = False
            try:
                CAMPAIGN.set_step4_finish_state(False)
            except Exception:
                pass
        try:
            self._reload_history_snapshot_from_disk()
            current_run_id = str(getattr(self, "current_run_id", "") or "").strip()
            current_run = self.history.get_run(current_run_id) if current_run_id else None
            registry_status = str(getattr(current_run, "status", "") or "").strip().lower()
            if current_run_id and CAMPAIGN.get_active_project_name():
                self._remember_campaign_training_run_in_registry(
                    run_id=current_run_id,
                    status=registry_status or (TrainingStatus.COMPLETED.value if success else TrainingStatus.FAILED.value),
                    target=getattr(self, "_pending_campaign_model_type", None) or self.get_campaign_training_target(),
                )
        except Exception:
            pass

        status_text, status_color = self._resolve_training_end_feedback(success, safe_msg)
        if success and CAMPAIGN.get_active_project_name():
            try:
                self._preferred_campaign_training_result_run_id = str(getattr(self, "current_run_id", "") or "").strip()
            except Exception:
                pass

        self._ui(lambda: self.btn_start_train.configure(state=tk.NORMAL))
        self._ui(lambda: self.btn_pause_train.configure(state=tk.DISABLED))
        self._ui(lambda: self.btn_stop_train.configure(state=tk.DISABLED))
        if success:
            self._ui(lambda: self._set_train_progress_values(overall=100.0, epoch=100.0))
        self._ui(lambda: self._set_training_metric_interpretation(final_interpretation))
        self._ui(lambda: self._set_train_live_metrics(getattr(self, "_latest_training_metrics", {})))
        self._ui(lambda: self.train_progress_label.configure(
            text=status_text,
            foreground=status_color,
        ))
        if not success:
            self._ui(lambda: self._update_training_progress_meta(eta_seconds=0.0))
        self._ui(lambda: self._load_history())
        self._ui(self._refresh_training_start_state)

    def on_resource_sample(sample):
        summary = str((sample or {}).get("summary_text") or format_resource_sample_line(sample or {})).strip()
        if not summary:
            return
        now = time.perf_counter()

        def update_ui():
            self._set_training_widget_text(
                getattr(self, "train_resource_label", None),
                "Zasoby w czasie treningu",
            )
            self._set_training_resource_sample(sample or {})

        self._ui(update_ui)
        if now - float(getattr(self, "_last_training_resource_log_at", 0.0) or 0.0) >= 15.0:
            self._last_training_resource_log_at = now
            self._append_train_log(f"[ZASOBY] {summary}")

    def on_resource_report(report):
        summary = str((report or {}).get("summary_text") or "").strip()
        report_path = str((report or {}).get("report_path") or "").strip()

        def update_ui():
            self._set_training_widget_text(
                getattr(self, "train_resource_label", None),
                "Raport zasobów po treningu",
            )
            self._set_training_resource_report(report or {})

        self._ui(
            update_ui
        )
        self._append_train_log(f"[RAPORT ZASOBÓW] {summary or 'zapisany'}")
        if report_path:
            self._append_train_log(f"[RAPORT ZASOBÓW] Plik: {report_path}")

    self.trainer.on_batch_progress = on_batch_progress
    self.trainer.on_epoch_end = on_epoch
    self.trainer.on_training_end = on_end
    self.trainer.on_resource_sample = on_resource_sample
    self.trainer.on_resource_report = on_resource_report

def _reload_history_snapshot_from_disk(self) -> bool:
    history_obj = getattr(self, "history", None)
    history_dir = getattr(history_obj, "history_dir", None)
    if not history_dir:
        return False

    try:
        refreshed = TrainingHistory(history_dir=Path(history_dir))
    except Exception as e:
        logger.debug(f"Nie udało się przeładować historii treningu z dysku: {e}")
        return False

    self.history = refreshed

    try:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            trainer.history = refreshed
            current_run_id = str(getattr(self, "current_run_id", "") or "").strip()
            if current_run_id:
                refreshed_run = refreshed.get_run(current_run_id)
                if refreshed_run is not None:
                    trainer.current_run = refreshed_run
    except Exception:
        pass

    return True

def _configure_history_tree_tags(self):
    tree = getattr(self, "tree", None)
    if tree is None:
        return
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    success = palette.get("success", "#2ecc71")
    panel = palette.get("panel", "#252526")
    pinned_bg = blend_hex_colors(success, panel, 0.76)
    try:
        tree.tag_configure("pinned_result", foreground=success, background=pinned_bg, font=("Segoe UI Semibold", 9, "bold"))
    except Exception:
        try:
            tree.tag_configure("pinned_result", foreground=success, background=pinned_bg)
        except Exception:
            pass

def _history_model_path_key(path_like) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve()).casefold()
    except Exception:
        return raw.casefold()

def _get_pinned_history_result_keys(self) -> tuple[set[str], set[str]]:
    run_ids: set[str] = set()
    model_keys: set[str] = set()

    try:
        active_target = str(_campaign_training_result_target(self) or "").strip().lower()
    except Exception:
        active_target = ""

    def _target_matches(value: str) -> bool:
        raw = str(value or "").strip().lower()
        return bool(not active_target or not raw or raw == active_target)

    try:
        stored = dict(CAMPAIGN.get_step4_finish_state() or {})
    except Exception:
        stored = {}
    if bool(stored.get("selection_confirmed")) and _target_matches(str(stored.get("target", "") or "")):
        run_id = str(stored.get("run_id", "") or "").strip()
        if run_id:
            run_ids.add(run_id)
        model_key = _history_model_path_key(stored.get("model_path", ""))
        if model_key:
            model_keys.add(model_key)

    try:
        project_name = CAMPAIGN.get_active_project_name()
        project_data = (CAMPAIGN.state.get("projects", {}) or {}).get(project_name, {}) if project_name else {}
    except Exception:
        project_data = {}
    if bool(project_data.get("step4_model_choice_confirmed", False)) and _target_matches(str(project_data.get("step4_last_target", "") or "")):
        run_id = str(project_data.get("step4_last_run_id", "") or "").strip()
        if run_id:
            run_ids.add(run_id)
        model_key = _history_model_path_key(project_data.get("step4_selected_model_path", ""))
        if model_key:
            model_keys.add(model_key)

    candidate_targets = [active_target] if active_target in {"char", "plate"} else ["char", "plate"]
    for target in candidate_targets:
        try:
            global_model = str(CAMPAIGN.get_global_model(target) or "").strip()
        except Exception:
            global_model = ""
        model_key = _history_model_path_key(global_model)
        if model_key:
            model_keys.add(model_key)
        if global_model:
            try:
                source_run = self._resolve_training_run_from_model_path(Path(global_model))
            except Exception:
                source_run = None
            source_run_id = str(getattr(source_run, "id", "") or "").strip() if source_run is not None else ""
            if source_run_id:
                run_ids.add(source_run_id)

    return run_ids, model_keys

def _set_training_running_ui_state(self, run_id: str | None = None, *, status_text: str | None = None):
    resolved_run_id = str(run_id or getattr(self, "current_run_id", "") or "").strip()

    try:
        self.btn_start_train.configure(state=tk.DISABLED)
    except Exception:
        pass
    try:
        self.btn_pause_train.configure(state=tk.NORMAL)
    except Exception:
        pass
    try:
        self.btn_stop_train.configure(state=tk.NORMAL)
    except Exception:
        pass

    if status_text:
        try:
            self._set_training_widget_text(self.train_progress_label, status_text)
            self.train_progress_label.configure(foreground="#2c3e50")
        except Exception:
            pass

    if not resolved_run_id:
        return

    try:
        self._load_history()
    except Exception as e:
        logger.debug(f"Nie udalo sie odswiezyc historii po starcie treningu: {e}")

    tree = getattr(self, "tree", None)
    if tree is None:
        return
    try:
        if tree.exists(resolved_run_id):
            tree.selection_set(resolved_run_id)
            tree.focus(resolved_run_id)
            tree.see(resolved_run_id)
            self._on_run_selected()
    except Exception:
        pass

def _update_training_history_progress(self, *, run_id, epoch=None, total_epochs=None) -> bool:
    """Update only live cells, without disk reads, rebuilding rows or changing selection."""
    tree = getattr(self, "tree", None)
    trainer = getattr(self, "trainer", None)
    current_run = getattr(trainer, "current_run", None)
    row_id = str(run_id or "")
    if (tree is None or not row_id or not bool(getattr(trainer, "is_training", False))
            or str(getattr(current_run, "id", "")) != row_id
            or str(getattr(self, "current_run_id", "") or row_id) != row_id):
        return False
    try:
        if not tree.exists(row_id):
            return False
        if bool(getattr(trainer, "should_stop", False)):
            status = "zatrzymywanie"
        elif bool(getattr(trainer, "should_pause", False)):
            status = "wstrzymywanie"
        else:
            status = "trwa trening"
        if "pinned_result" in tree.item(row_id, "tags"):
            status = f"★ PODPIĘTY | {status}"
        values = {"Status": status}
        if epoch is not None and total_epochs is not None:
            total = max(1, int(total_epochs))
            values["Epoki"] = f"{max(0, min(int(epoch), total))}/{total}"
        changed = False
        for column, value in values.items():
            if tree.set(row_id, column) != value:
                tree.set(row_id, column, value)
                changed = True
        return changed
    except tk.TclError:
        # The view may have been closed while a progress callback was queued.
        return False


def _load_history(self):
    if not hasattr(self, "tree"):
        return
    selected_run_id = ""
    try:
        selection = self.tree.selection()
        if selection:
            selected_run_id = str(selection[0] or "").strip()
    except Exception:
        selected_run_id = ""

    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass

    def _format_history_datetime(value: str | None, *, fallback: str = "-") -> str:
        raw = str(value or "").strip()
        if not raw:
            return fallback
        try:
            return datetime.datetime.fromisoformat(raw).strftime("%d.%m %H:%M")
        except Exception:
            return raw.replace("T", " ")[:16] or fallback

    _configure_history_tree_tags(self)
    try:
        pinned_run_ids, pinned_model_keys = _get_pinned_history_result_keys(self)
    except Exception:
        pinned_run_ids, pinned_model_keys = set(), set()

    self.tree.delete(*self.tree.get_children())
    for run in self.history.get_all_runs():
        if CAMPAIGN.get_active_project_name():
            try:
                if not self._does_history_run_match_active_campaign_target(run):
                    continue
            except Exception:
                continue
        # Zachowaj pełne run.id, aby wybór historii i folderów był jednoznaczny.
        best_map = getattr(run, 'best_map50_95', 0.0) or 0.0
        run_target = ""
        try:
            infer_target = getattr(self, "_infer_history_run_target", None)
            if callable(infer_target):
                run_target = str(infer_target(run) or "").strip().lower()
        except Exception:
            run_target = ""
        if not run_target:
            try:
                run_target = str(
                    self._infer_dataset_target(getattr(run, "dataset_path", "")) or ""
                ).strip().lower()
            except Exception:
                run_target = ""
        target_label = self._format_history_run_target_label(run_target)
        run_label = build_run_display_ref(run, kind_hint="training").id
        dataset_path = str(getattr(run, "dataset_path", "") or "").strip()
        if dataset_path:
            try:
                dataset_label = build_dataset_display_ref(dataset_path, target_hint=run_target).id
            except Exception:
                dataset_label = Path(dataset_path).name or "-"
        else:
            dataset_label = "-"
        try:
            best_weights = self._resolve_history_run_best_weights(run)
        except Exception:
            best_weights = None
        if best_weights is not None:
            try:
                model_label = build_model_display_ref(
                    best_weights,
                    run=run,
                    target_hint=run_target,
                    source_run_label=run_label,
                ).id
            except Exception:
                model_label = Path(best_weights).name or "-"
        else:
            model_label = "-"
        started_short = _format_history_datetime(
            getattr(run, "started_at", None),
            fallback=_format_history_datetime(getattr(run, "created_at", None)),
        )
        run_id = str(getattr(run, "id", "") or "").strip()
        best_model_key = _history_model_path_key(best_weights)
        row_is_pinned = bool(
            (run_id and run_id in pinned_run_ids)
            or (best_model_key and best_model_key in pinned_model_keys)
        )
        row_tags = ("pinned_result",) if row_is_pinned else ()
        status_label = self._format_history_run_status_label(run)
        if row_is_pinned:
            gate_label = _step4_finish_gate_display_id()
            model_label = f"★ WYNIK {gate_label} | {model_label}"
            status_label = f"★ PODPIĘTY | {status_label}"

        self.tree.insert("", tk.END, iid=str(run.id), values=(
            target_label,
            model_label,
            dataset_label,
            run_label,
            started_short,
            status_label,
            f"{run.current_epoch}/{run.epochs}",
            f"{float(best_map):.3f}",
        ), tags=row_tags)

    if selected_run_id:
        for item_id in self.tree.get_children():
            try:
                values = self.tree.item(item_id, "values")
            except Exception:
                values = ()
            if str(item_id) == str(selected_run_id):
                try:
                    self.tree.selection_set(item_id)
                    self.tree.focus(item_id)
                except Exception:
                    pass
                break

    self._on_run_selected()
    try:
        _refresh_campaign_training_result_selector(self)
    except Exception:
        pass

def _delete_selected(self):
    run = self._selected_run()
    if run and messagebox.askyesno("Potwierdź", "Usunąć run treningu?"):
        self.history.delete_run(run.id, delete_files=True)
        self._load_history()

def _open_run_folder(self):
    run = self._selected_run()
    if run and Path(run.output_dir).exists():
        self._open_path(Path(run.output_dir))

def _campaign_training_result_target(self) -> str:
    target = str(self.get_campaign_training_target() or CAMPAIGN.get_iteration_target() or "").strip().lower()
    return target if target in {"plate", "char"} else ""

def _campaign_training_result_candidates(self) -> list:
    if not CAMPAIGN.get_active_project_name():
        return []
    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass
    target = _campaign_training_result_target(self)
    result = []
    try:
        runs = list(self.history.get_all_runs() or [])
    except Exception:
        runs = []
    for run in runs:
        try:
            if str(getattr(run, "status", "") or "").strip().lower() != TrainingStatus.COMPLETED.value:
                continue
            if target and not self._does_history_run_match_active_campaign_target(run):
                continue
            if self._resolve_history_run_best_weights(run) is None:
                continue
            result.append(run)
        except Exception:
            continue

    def _run_sort_key(run) -> tuple[str, str]:
        finished = str(getattr(run, "finished_at", "") or "").strip()
        created = str(getattr(run, "created_at", "") or "").strip()
        run_id = str(getattr(run, "id", "") or "").strip()
        return (finished or created or "", run_id)

    return sorted(result, key=_run_sort_key, reverse=True)

def _campaign_training_result_choice_label(self, run) -> str:
    run_id = str(getattr(run, "id", "") or "").strip() or "-"
    run_ref = build_run_display_ref(run, kind_hint="training")
    try:
        target_hint = self._infer_history_run_target(run)
    except Exception:
        target_hint = ""
    try:
        best_weights = self._resolve_history_run_best_weights(run)
    except Exception:
        best_weights = None
    if best_weights is not None:
        try:
            model_ref = build_model_display_ref(
                best_weights,
                run=run,
                target_hint=target_hint,
                source_run_label=run_ref.id or run_id,
            )
            model_name = model_ref.id
        except Exception:
            model_name = Path(best_weights).name or "model"
    else:
        model_name = "model"
    try:
        score = float(getattr(run, "best_map50_95", 0.0) or 0.0)
    except Exception:
        score = 0.0
    return f"{model_name} | mAP50-95 {score:.3f} | run {run_ref.id or run_id}"

def _campaign_training_result_detail_text(self, run) -> str:
    if run is None:
        return ""
    try:
        target = str(self._infer_history_run_target(run) or "").strip().lower()
    except Exception:
        target = ""
    target_label = "model tablic" if target == "plate" else "model znaków" if target == "char" else "model"
    try:
        best_weights = self._resolve_history_run_best_weights(run)
    except Exception:
        best_weights = None
    run_ref = build_run_display_ref(run, kind_hint="training")
    try:
        model_ref = build_model_display_ref(
            best_weights,
            run=run,
            target_hint=target,
            source_run_label=run_ref.id or str(getattr(run, "id", "") or "-"),
        )
        model_text = model_ref.id
    except Exception:
        model_text = Path(best_weights).name if best_weights else "-"
    dataset_path = str(getattr(run, "dataset_path", "") or "").strip()
    if dataset_path:
        try:
            dataset_text = build_dataset_display_ref(dataset_path, target_hint=target).id
        except Exception:
            dataset_text = Path(dataset_path).name or "-"
    else:
        dataset_text = "-"
    try:
        score = float(getattr(run, "best_map50_95", 0.0) or 0.0)
    except Exception:
        score = 0.0
    started = str(getattr(run, "started_at", "") or getattr(run, "created_at", "") or "").strip()
    if started:
        try:
            started_text = datetime.datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            started_text = started.replace("T", " ")
    else:
        started_text = "-"
    return (
        f"{target_label} | model: {model_text} | "
        f"dataset: {dataset_text} | run: {run_ref.id or str(getattr(run, 'id', '') or '-')} | "
        f"start: {started_text} | "
        f"epoki: {getattr(run, 'current_epoch', '-')}/{getattr(run, 'epochs', '-')} | "
        f"mAP50-95: {score:.3f} | best.pt: {Path(best_weights).name if best_weights else '-'}"
    )

def _selected_campaign_training_result_run(self):
    choice_var = getattr(self, "campaign_training_result_var", None)
    try:
        choice = str(choice_var.get() or "").strip()
    except Exception:
        choice = ""
    choices = getattr(self, "_campaign_training_result_choices", {}) or {}
    run_id = str(choices.get(choice, "") or "").strip()
    if not run_id:
        return None
    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass
    try:
        return self.history.get_run(run_id)
    except Exception:
        return None

def _refresh_campaign_training_result_selector(self):
    selector = getattr(self, "campaign_training_result_entry", None)
    if selector is None:
        selector = getattr(self, "campaign_training_result_combo", None)
    dropdown_btn = getattr(self, "campaign_training_result_dropdown_btn", None)
    dropdown_menu = getattr(self, "campaign_training_result_menu", None)
    title_lbl = getattr(self, "campaign_training_result_title_lbl", None)
    status_lbl = getattr(self, "campaign_training_result_status_lbl", None)
    detail_lbl = getattr(self, "campaign_training_result_detail_lbl", None)
    action_btn = getattr(self, "btn_use_campaign_training_result", None)
    if selector is None:
        return
    palette = getattr(self.app, "palette", {}) if getattr(self, "app", None) is not None else {}
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f39c12")
    muted = palette.get("muted", "#888888")
    panel = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")

    def _set_status(text: str, tone: str) -> None:
        if status_lbl is None:
            return
        color = success if tone == "success" else warning if tone == "warning" else muted
        try:
            status_lbl.configure(
                text=text,
                bg=blend_hex_colors(color, panel, 0.78),
                fg=fg,
                highlightbackground=blend_hex_colors(color, panel, 0.35),
                highlightcolor=blend_hex_colors(color, panel, 0.35),
            )
        except Exception:
            pass

    def _set_title(pinned: bool) -> None:
        if title_lbl is None:
            return
        gate_label = _step4_finish_gate_display_id()
        try:
            title_lbl.configure(
                text=(f"★ Wynik bramki {gate_label}: model podpięty" if pinned else f"Wynik bramki {gate_label}"),
                fg=success,
            )
        except Exception:
            pass

    def _set_selector_enabled(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        try:
            selector.configure(state=state)
        except Exception:
            pass
        if dropdown_btn is not None:
            try:
                dropdown_btn.configure(state=state)
            except Exception:
                pass

    def _select_campaign_result_label(label: str) -> None:
        try:
            self._preferred_campaign_training_result_run_id = ""
        except Exception:
            pass
        try:
            self.campaign_training_result_var.set(str(label or ""))
        except Exception:
            pass
        _refresh_campaign_training_result_selector(self)

    def _rebuild_campaign_result_menu(labels: list[str]) -> None:
        if dropdown_menu is None:
            try:
                selector.configure(values=tuple(labels))
            except Exception:
                pass
            return
        try:
            dropdown_menu.delete(0, tk.END)
            for label in labels:
                dropdown_menu.add_command(
                    label=str(label or ""),
                    command=lambda value=label: _select_campaign_result_label(value),
                )
        except Exception:
            pass

    def _scroll_result_selector_start() -> None:
        try:
            if selector is None or not selector.winfo_exists():
                return
            selector.icursor(0)
            selector.xview_moveto(0.0)
        except Exception:
            pass

    if not CAMPAIGN.get_active_project_name():
        _set_title(False)
        _rebuild_campaign_result_menu([])
        _set_selector_enabled(False)
        if detail_lbl is not None:
            detail_lbl.configure(text=f"Wybór wyniku bramki {_step4_finish_gate_display_id()} jest dostępny tylko w kampanii.")
        if action_btn is not None:
            action_btn.configure(
                state=tk.DISABLED,
                text=f"Podepnij jako wynik bramki {_step4_finish_gate_display_id()}",
                command=self._use_campaign_training_result_choice,
            )
        _set_status("POZA KAMPANIĄ", "muted")
        return

    candidates = _campaign_training_result_candidates(self)
    choices: dict[str, str] = {}
    labels: list[str] = []
    for run in candidates:
        label = _campaign_training_result_choice_label(self, run)
        base_label = label
        suffix = 2
        while label in choices:
            label = f"{base_label} ({suffix})"
            suffix += 1
        choices[label] = str(getattr(run, "id", "") or "").strip()
        labels.append(label)
    self._campaign_training_result_choices = choices
    _rebuild_campaign_result_menu(labels)
    _set_selector_enabled(bool(labels))

    try:
        finish_state = dict(self.get_campaign_step4_finish_state(iteration_target=_campaign_training_result_target(self)) or {})
    except Exception:
        try:
            finish_state = dict(CAMPAIGN.get_step4_finish_state() or {})
        except Exception:
            finish_state = {}
    finish_run_id = str(finish_state.get("run_id", "") or "").strip() if bool(finish_state.get("ready")) else ""

    try:
        selected_label = str(self.campaign_training_result_var.get() or "").strip()
    except Exception:
        selected_label = ""
    preferred_run_id = str(getattr(self, "_preferred_campaign_training_result_run_id", "") or "").strip()
    if selected_label not in choices and finish_run_id and finish_run_id in set(choices.values()):
        for label, run_id in choices.items():
            if run_id == finish_run_id:
                selected_label = label
                break
    elif not finish_run_id and preferred_run_id and preferred_run_id in set(choices.values()):
        for label, run_id in choices.items():
            if run_id == preferred_run_id:
                selected_label = label
                break
    elif selected_label not in choices and labels:
        selected_label = labels[0]
    if labels:
        try:
            self.campaign_training_result_var.set(selected_label)
        except Exception:
            pass
        try:
            selector.after_idle(_scroll_result_selector_start)
        except Exception:
            pass

    selected_run_id = str(choices.get(selected_label, "") or "").strip()
    selected_run = None
    if selected_run_id:
        try:
            selected_run = self.history.get_run(selected_run_id)
        except Exception:
            selected_run = None

    if not labels:
        _set_title(False)
        if detail_lbl is not None:
            detail_lbl.configure(text=f"Brak ukończonych runów pasujących do aktywnego toru bramki {_step4_finish_gate_display_id()}.")
        if action_btn is not None:
            action_btn.configure(
                state=tk.DISABLED,
                text=f"Podepnij jako wynik bramki {_step4_finish_gate_display_id()}",
                command=self._use_campaign_training_result_choice,
            )
        _set_status("BRAK KANDYDATA", "warning")
        return

    detail_text = _campaign_training_result_detail_text(self, selected_run)
    if selected_run_id and selected_run_id == finish_run_id:
        _set_title(True)
        if action_btn is not None:
            action_btn.configure(state=tk.NORMAL, text="Odepnij wynik", command=self._clear_pinned_step4_result)
        _set_status(f"★ WYNIK {_step4_finish_gate_display_id()} WYBRANY", "success")
        if detail_text:
            detail_text = (
                f"★ Ten model jest podpięty jako wynik bramki {_step4_finish_gate_display_id()}.\n"
                f"{detail_text}\n"
                "Jeśli chcesz zmienić decyzję, odepnij wynik tutaj."
            )
    elif finish_run_id:
        _set_title(True)
        if action_btn is not None:
            action_btn.configure(state=tk.NORMAL, text="Odepnij wynik", command=self._clear_pinned_step4_result)
        _set_status(f"★ WYNIK {_step4_finish_gate_display_id()} PODPIĘTY", "success")
        if detail_text:
            detail_text = (
                f"★ Inny model jest już podpięty jako wynik bramki {_step4_finish_gate_display_id()}.\n"
                f"{detail_text}\n"
                "Nowy model możesz wskazać dopiero po odpięciu aktualnego wyniku w tej sekcji."
            )
    else:
        _set_title(False)
        if action_btn is not None:
            action_btn.configure(
                state=tk.NORMAL,
                text=f"Podepnij jako wynik bramki {_step4_finish_gate_display_id()}",
                command=self._use_campaign_training_result_choice,
            )
        _set_status("DO WYBORU", "warning")
        if detail_text:
            detail_text = (
                f"{detail_text}\n"
                f"Jeśli to właściwy model, wybierz go jako wynik {_step4_finish_gate_display_id()}. To jeszcze nie zamyka bramki."
            )
    if detail_lbl is not None:
        detail_lbl.configure(text=detail_text)

def _on_campaign_training_result_choice(self, event=None):
    try:
        self._preferred_campaign_training_result_run_id = ""
    except Exception:
        pass
    _refresh_campaign_training_result_selector(self)

def _use_campaign_training_result_choice(self):
    run = _selected_campaign_training_result_run(self)
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            f"Najpierw wybierz kandydata na wynik treningu dla bramki {_step4_finish_gate_display_id()}.",
        )
    run_id = str(getattr(run, "id", "") or "").strip()
    selected_ok = False
    try:
        if hasattr(self, "tree") and run_id:
            self.tree.selection_set(run_id)
            self.tree.focus(run_id)
            self.tree.see(run_id)
            selected_ok = run_id in {str(item) for item in self.tree.selection()}
    except Exception:
        selected_ok = False
    if not selected_ok:
        return messagebox.showwarning(
            "Nie wybrano runu",
            "Nie udało się zaznaczyć wybranego runu w historii treningu. Odśwież historię i spróbuj ponownie.",
        )
    try:
        result = self._promote_selected_run_model_to_campaign()
    finally:
        try:
            _refresh_campaign_training_result_selector(self)
        except Exception:
            pass
    return result

def _promote_selected_run_model_to_campaign(self):
    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz zakończony run treningu z historii."
        )
    if not CAMPAIGN.get_active_project_name():
        return messagebox.showwarning(
            "Brak projektu",
            f"Wynik bramki {_step4_finish_gate_display_id()} można wskazać tylko w aktywnej kampanii."
        )

    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value != TrainingStatus.COMPLETED.value:
        return messagebox.showwarning(
            "Run nie jest gotowy",
            f"Jako wynik bramki {_step4_finish_gate_display_id()} można wskazać tylko trening zakończony sukcesem."
        )
    if not self._does_history_run_match_active_campaign_target(run):
        active_target = str(self.get_campaign_training_target() or CAMPAIGN.get_iteration_target() or "").strip().lower()
        active_label = "tablic" if active_target == "plate" else "znaków" if active_target == "char" else "bieżącego toru"
        return messagebox.showerror(
            "Niezgodny tor modelu",
            f"Wybrany run nie pasuje do aktywnego toru {active_label}."
        )

    target = self._infer_history_run_target(run)
    if target not in {"plate", "char"}:
        target = str(self.get_campaign_training_target() or CAMPAIGN.get_iteration_target() or "").strip().lower()
    if target not in {"plate", "char"}:
        return messagebox.showerror(
            "Nie rozpoznano toru",
            "Nie mogę ustalić, czy wybrany model dotyczy tablic czy znaków."
        )

    best_weights = self._resolve_history_run_best_weights(run)
    if best_weights is None or not Path(best_weights).exists():
        return messagebox.showerror(
            "Brak best.pt",
            "Wybrany run nie ma dostępnego pliku best.pt."
        )

    run_id = str(getattr(run, "id", "") or "").strip()
    if not run_id:
        return messagebox.showerror(
            "Brak identyfikatora runu",
            "Wybrany wpis historii nie ma identyfikatora runu."
        )

    try:
        pinned_state = dict(self._get_pinned_step4_result_state() or {})
    except Exception:
        pinned_state = {}
    pinned_run_id = str(pinned_state.get("run_id", "") or "").strip()
    if pinned_run_id and pinned_run_id != run_id:
        return messagebox.showwarning(
            "Wynik bramki jest przypięty",
            (
                "Ta bramka ma już przypięty model wynikowy.\n\n"
                "Najpierw użyj `Odepnij wynik` w sekcji wyboru wyniku bramki, "
                "a dopiero potem wybierz inny run albo uruchom nowy trening."
            ),
        )

    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        iteration_num = 0

    CAMPAIGN.set_global_model(target, str(best_weights))
    CAMPAIGN.set_step4_finish_state(
        True,
        run_id=run_id,
        target=target,
        iteration_num=iteration_num,
        selection_confirmed=True,
        model_path=str(best_weights),
    )
    self.current_run_id = run_id
    self._pending_campaign_model_type = None
    self._step4_campaign_finish_ready = True

    try:
        complete_campaign_step4_if_needed(self, target)
    except Exception:
        pass
    try:
        self._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass
    try:
        self._refresh_step4_pinned_result_ui()
    except Exception:
        pass
    try:
        self._load_history()
    except Exception:
        pass
    try:
        campaign_tab = self.app.tabs.get("campaign")
        if campaign_tab:
            campaign_tab._refresh_dashboard()
    except Exception:
        pass

    target_label = "znaków" if target == "char" else "tablic"
    try:
        self._append_train_log(
            f"[MODEL] Jawnie podpięto model {target_label} jako wynik bramki {_step4_finish_gate_display_id()}: {best_weights}"
        )
    except Exception:
        pass
    try:
        _refresh_campaign_training_result_selector(self)
    except Exception:
        pass
    return messagebox.showinfo(
        f"Wynik {_step4_finish_gate_display_id()} wybrany",
        (
            f"Wybrano model {target_label} jako wynik bramki {_step4_finish_gate_display_id()}:\n{Path(best_weights).name}\n\n"
            f"To wybór artefaktu. Wróć do grafu i użyj pola Zatwierdź na bramce {_step4_finish_gate_display_id()}, "
            "aby formalnie zamknąć przejście."
        )
    )

def _resume_selected_run(self):
    if bool(getattr(self, "_training_start_in_progress", False)):
        return
    run = self._selected_run()
    if run is None:
        return

    if not self._is_history_run_resume_allowed(run):
        if self._is_history_run_resumable(run) and CAMPAIGN.get_active_project_name():
            return messagebox.showerror(
                "Wznowienie niedostępne",
                "W kampanii możesz wznowić tylko ostatni wznowialny run aktywnego toru.\n\n"
                "Starsze wstrzymane runy pozostają w historii jako archiwum, ale nie są już ścieżką roboczą tej iteracji."
            )

    if not self._does_history_run_match_active_campaign_target(run):
        active_target = str(self.get_campaign_training_target() or CAMPAIGN.get_iteration_target() or "").strip().lower()
        active_label = "tablic" if active_target == "plate" else "znaków" if active_target == "char" else "bieżącego toru"
        return messagebox.showerror(
            "Niezgodny tor wznowienia",
            "Wybrany run należy do innego toru treningu niż aktualnie otwarty węzeł treningowy.\n\n"
            f"W tej chwili możesz wznowić tylko runy dla toru {active_label}."
        )

    last_weights = str(getattr(run, "last_weights", "") or "").strip()
    if not last_weights or not Path(last_weights).exists():
        try:
            fallback_last = Path(str(getattr(run, "output_dir", "") or "")) / "train" / "weights" / "last.pt"
            if fallback_last.exists():
                last_weights = str(fallback_last)
                try:
                    run.last_weights = last_weights
                    self.history.update_run(str(run.id), last_weights=last_weights)
                except Exception:
                    pass
        except Exception:
            pass
    if not last_weights or not Path(last_weights).exists():
        return messagebox.showerror(
            "Brak checkpointu do wznowienia",
            "Wybrany run nie ma poprawnego pliku last.pt.\n\n"
            "Tego treningu nie da się wznowić od miejsca pauzy."
        )

    if not self._begin_step4_operation("z4.training.run", "Z4: przygotowanie wznowienia treningu"):
        return

    run_id_to_resume = str(run.id)
    _set_training_preparing_ui_state(self, text="Przygotowanie wznowienia treningu...", progress=5.0)
    detached_models = _detach_gpu_resources_before_training(self)
    trainer = self.trainer

    def progress_callback(stage, progress, detail=""):
        safe_stage = str(stage or "").strip()
        safe_detail = str(detail or "").strip()
        try:
            safe_progress = float(progress)
        except Exception:
            safe_progress = 0.0
        self._ui(
            lambda s=safe_stage, p=safe_progress, d=safe_detail: _update_training_preflight_progress(
                self,
                s,
                p,
                d,
            )
        )

    def finish_success(resumed_run_id):
        self._training_start_in_progress = False
        self._training_preflight_thread = None
        self.current_run_id = resumed_run_id
        z4_training_metrics._apply_preflight_dataset_validation(
            self, getattr(trainer, "_last_preflight_dataset_validation", None),
        )
        refreshed_run = run
        try:
            self._reload_history_snapshot_from_disk()
            history_run = self.history.get_run(str(resumed_run_id))
            if history_run is not None:
                refreshed_run = history_run
        except Exception:
            pass
        try:
            resumed_run_display = build_run_display_ref(refreshed_run, kind_hint="training").id
        except Exception:
            resumed_run_display = str(resumed_run_id or "").strip()
        self._last_training_completion_summary_run_id = None
        self._step4_campaign_finish_ready = False
        try:
            if CAMPAIGN.get_active_project_name():
                CAMPAIGN.set_step4_finish_state(False)
        except Exception:
            pass
        self._set_training_running_ui_state(
            resumed_run_id,
            status_text=f"Wznowiono run treningu: {resumed_run_display}",
        )

        try:
            previous_metrics = [
                dict(row)
                for row in list(getattr(refreshed_run, "metrics_history", []) or [])
                if isinstance(row, dict)
            ]
        except Exception:
            previous_metrics = []
        self._current_training_metric_history = previous_metrics
        self._latest_training_metrics = dict(previous_metrics[-1]) if previous_metrics else {}
        try:
            self._set_train_progress_values(overall=0.0, epoch=0.0)
            self._reset_training_runtime_progress()
            self._set_train_live_metrics(self._latest_training_metrics or None)
            self._set_training_metric_interpretation(
                self._build_training_best_epoch_summary(self._latest_training_metrics or None)
            )
        except Exception as e:
            logger.debug(f"Nie udalo sie odtworzyc metryk UI po wznowieniu treningu: {e}")
        self._set_training_running_ui_state(
            resumed_run_id,
            status_text=f"Wznowiono run treningu: {resumed_run_display}",
        )
        self._training_started_monotonic = time.perf_counter()
        self._training_started_wall_clock = datetime.datetime.now()
        self._append_train_log(f"[RESUME] Wznowiono trening z checkpointu: {last_weights}")
        if CAMPAIGN.get_active_project_name():
            self._pending_campaign_model_type = self.get_campaign_training_target()
            try:
                self._remember_campaign_training_run_in_registry(
                    run_id=str(resumed_run_id or "").strip(),
                    status=TrainingStatus.RUNNING.value,
                    target=self._pending_campaign_model_type,
                )
            except Exception:
                pass
        else:
            self._pending_campaign_model_type = None

        try:
            self._refresh_step4_campaign_navigation_ui()
        except Exception:
            pass

        if self._training_completion_poll_job is not None:
            try:
                self.frame.after_cancel(self._training_completion_poll_job)
            except Exception:
                pass
            self._training_completion_poll_job = None
        self._training_completion_poll_job = self.frame.after(3000, self._poll_training_completion)

    def finish_failure(error_text=""):
        z4_training_metrics._apply_preflight_dataset_validation(
            self, getattr(trainer, "_last_preflight_dataset_validation", None),
        )
        message = (
            "Wznowienie treningu nie wystartowało.\n\n"
            "Sprawdź, czy run nadal ma poprawny checkpoint `last.pt`."
        )
        if str(error_text or "").strip():
            message = f"{message}\n\nSzczegóły:\n{str(error_text).strip()}"
        _finish_training_preflight_failure(
            self,
            title="Nie udało się wznowić treningu",
            message=message,
        )

    def worker():
        resumed_run_id = None
        error_text = ""
        try:
            _release_detached_gpu_resources(detached_models)
            resumed_run_id = trainer.resume_training(run_id_to_resume, progress_callback=progress_callback)
        except Exception as e:
            logger.exception("Nie udało się wznowić treningu")
            error_text = str(e)

        delivered = self._ui(
            lambda rid=resumed_run_id, err=error_text: (
                finish_success(rid) if rid else finish_failure(err)
            )
        )
        if delivered is False and resumed_run_id:
            trainer.shutdown()

    thread = threading.Thread(target=worker, name="Z4TrainingResumePreflight", daemon=True)
    self._training_preflight_thread = thread
    try:
        thread.start()
    except Exception as exc:
        finish_failure(str(exc))

def _show_history_context_menu(self, event=None):
    if event is None or not hasattr(self, "tree"):
        return

    try:
        row_id = self.tree.identify_row(event.y)
    except Exception:
        row_id = ""
    if not row_id:
        return

    try:
        self.tree.selection_set(row_id)
        self.tree.focus(row_id)
    except Exception:
        pass

    try:
        self._on_run_selected()
    except Exception:
        pass

    menu = getattr(self, "history_context_menu", None)
    if menu is None:
        return

    selected_run = self._selected_run()
    resumable = bool(selected_run is not None and self._is_history_run_resume_allowed(selected_run))
    fine_tunable = bool(selected_run is not None and self._is_history_run_fine_tune_candidate(selected_run))
    exportable = False
    promotable = False
    validation_ready = False
    if selected_run is not None:
        try:
            exportable = bool(
                self._infer_history_run_target(selected_run) in {"plate", "char", "vehicle"}
                and self._resolve_history_run_best_weights(selected_run) is not None
            )
            validation_ready = exportable
        except Exception:
            exportable = False
            validation_ready = False
        try:
            promotable = bool(
                CAMPAIGN.get_active_project_name()
                and str(getattr(selected_run, "status", "") or "").strip().lower() == TrainingStatus.COMPLETED.value
                and self._infer_history_run_target(selected_run) in {"plate", "char"}
                and self._resolve_history_run_best_weights(selected_run) is not None
                and self._does_history_run_match_active_campaign_target(selected_run)
            )
        except Exception:
            promotable = False
    try:
        menu.entryconfigure("Wznów trening", state=(tk.NORMAL if resumable else tk.DISABLED))
    except Exception:
        pass
    try:
        menu.entryconfigure("Dotrenuj od best.pt", state=(tk.NORMAL if fine_tunable else tk.DISABLED))
    except Exception:
        pass
    try:
        menu.entryconfigure("[ TEST ] Waliduj best.pt", state=(tk.NORMAL if validation_ready else tk.DISABLED))
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Eksportuj best.pt do modeli trybu swobodnego",
            state=(tk.NORMAL if exportable else tk.DISABLED),
        )
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Eksportuj model mobilny (.alprmodel)",
            state=(tk.NORMAL if exportable else tk.DISABLED),
        )
    except Exception:
        pass
    try:
        menu.entryconfigure(
            f"Podepnij jako wynik bramki {_step4_finish_gate_display_id()}",
            state=(tk.NORMAL if promotable else tk.DISABLED),
        )
    except Exception:
        pass

    try:
        menu.tk_popup(event.x_root, event.y_root)
    except Exception:
        pass
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass

def _selected_ranking_entry_ref(self) -> dict:
    tree = getattr(self, "rank_tree", None)
    if tree is None:
        return {}
    try:
        item_id = str(tree.focus() or "")
        if not item_id:
            selection = list(tree.selection() or [])
            item_id = str(selection[0]) if selection else ""
    except Exception:
        item_id = ""
    refs = getattr(self, "_ranking_tree_entry_refs", {}) or {}
    ref = refs.get(item_id, {})
    return dict(ref) if isinstance(ref, dict) else {}

def _selected_ranking_run(self):
    ref = _selected_ranking_entry_ref(self)
    run_id = str(ref.get("run_id", "") or "").strip()
    if not run_id:
        return None
    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass
    try:
        return self.history.get_run(run_id)
    except Exception:
        return None

def _focus_history_run_from_ranking(self, run) -> bool:
    run_id = str(getattr(run, "id", "") or "").strip()
    if not run_id or not hasattr(self, "tree"):
        return False
    try:
        self.tree.selection_set(run_id)
        self.tree.focus(run_id)
        self.tree.see(run_id)
        return True
    except Exception:
        return False

def _use_selected_ranking_model_as_campaign_result(self):
    run = _selected_ranking_run(self)
    if run is None:
        return messagebox.showwarning(
            "Brak runu projektu",
            "Ten wpis rankingu nie jest powiązany z runem historii projektu, więc nie można go wskazać jako wynik bramki.",
        )
    if not _focus_history_run_from_ranking(self, run):
        return messagebox.showwarning(
            "Nie udało się wybrać runu",
            "Nie udało się zaznaczyć runu z rankingu w historii treningów. Odśwież historię i spróbuj ponownie.",
        )
    return self._promote_selected_run_model_to_campaign()


def _is_history_run_fine_tune_candidate(self, run) -> bool:
    if run is None:
        return False
    if str(getattr(run, "status", "") or "").strip().lower() != TrainingStatus.COMPLETED.value:
        return False
    try:
        get_pinned = getattr(self, "_get_pinned_step4_result_state", None)
        if callable(get_pinned):
            if get_pinned():
                return False
    except Exception:
        pass
    try:
        if not self._does_history_run_match_active_campaign_target(run):
            return False
    except Exception:
        return False
    try:
        best_weights = self._resolve_history_run_best_weights(run)
    except Exception:
        best_weights = None
    return bool(best_weights is not None and Path(best_weights).exists())


def _select_selected_run_as_fine_tune_base(self):
    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz ukończony run z historii treningów."
        )
    if not self._is_history_run_fine_tune_candidate(run):
        return messagebox.showwarning(
            "Nie można dotrenować",
            (
                "Dotrenowanie może wystartować tylko od ukończonego runu zgodnego z aktywnym torem "
                "i posiadającego dostępny plik best.pt.\n\n"
                "Jeśli run jest przerwany, użyj `Wznów trening` zamiast dotrenowania."
            ),
        )

    best_weights = self._resolve_history_run_best_weights(run)
    if best_weights is None:
        return messagebox.showerror("Brak best.pt", "Wybrany run nie ma dostępnego pliku best.pt.")

    self._step4_setting_fine_tune_base = True
    try:
        self.base_model_var.set(self._get_custom_base_model_label())
        self.base_custom_var.set(str(best_weights))
        self._set_step4_fine_tune_parent_state(run, best_weights)
    finally:
        self._step4_setting_fine_tune_base = False

    try:
        self._refresh_training_base_model_selection_ui()
        self._refresh_training_base_model_identity_ui()
        self._refresh_training_execution_summary()
        self._refresh_training_start_state()
    except Exception:
        pass

    run_label = self._format_training_model_run_label(run)
    try:
        self._append_train_log(
            f"[DOTRENOWANIE] Ustawiono model startowy z runu {run_label}: {best_weights}"
        )
    except Exception:
        pass
    return messagebox.showinfo(
        "Dotrenowanie ustawione",
        (
            f"Model z runu {run_label} został ustawiony jako punkt startowy dotrenowania.\n\n"
            "Teraz wybierz wariant datasetu i parametry, a potem użyj `Rozpocznij trening`. "
            "Powstanie nowy run-kandydat; poprzedni model nie zostanie nadpisany."
        ),
    )


def _select_selected_ranking_run_as_fine_tune_base(self):
    run = _selected_ranking_run(self)
    if run is None:
        return messagebox.showwarning(
            "Brak runu projektu",
            "Ten wpis rankingu nie jest powiązany z runem historii projektu."
        )
    if not _focus_history_run_from_ranking(self, run):
        return messagebox.showwarning(
            "Nie udało się wybrać runu",
            "Nie udało się zaznaczyć runu z rankingu w historii treningów."
        )
    return self._select_selected_run_as_fine_tune_base()


def _open_selected_ranking_run_details(self):
    run = _selected_ranking_run(self)
    if run is None:
        return messagebox.showinfo(
            "Brak szczegółów runu",
            "Ten wpis rankingu nie jest powiązany z runem historii projektu.",
        )
    return self._open_run_details_modal(run)

def _open_selected_ranking_model_folder(self):
    ref = _selected_ranking_entry_ref(self)
    model_path = str(ref.get("model_path", "") or "").strip()
    target = None
    if model_path:
        try:
            path = Path(model_path)
            if path.exists():
                target = path.parent if path.is_file() else path
        except Exception:
            target = None
    if target is None:
        run = _selected_ranking_run(self)
        output_dir = str(getattr(run, "output_dir", "") or "").strip() if run is not None else ""
        if output_dir:
            try:
                path = Path(output_dir)
                if path.exists():
                    target = path
            except Exception:
                target = None
    if target is None:
        return messagebox.showinfo("Brak folderu", "Nie udało się odnaleźć folderu modelu z tego wpisu rankingu.")
    return self._open_path(target)

def _copy_selected_ranking_choice_label(self):
    ref = _selected_ranking_entry_ref(self)
    label = str(ref.get("choice_label", "") or "").strip()
    if not label:
        return
    try:
        root = self.frame.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(label)
    except Exception:
        pass

def _show_ranking_context_menu(self, event=None):
    tree = getattr(self, "rank_tree", None)
    if event is None or tree is None:
        return
    try:
        row_id = tree.identify_row(event.y)
    except Exception:
        row_id = ""
    if not row_id:
        return
    try:
        tree.selection_set(row_id)
        tree.focus(row_id)
    except Exception:
        pass

    ref = _selected_ranking_entry_ref(self)
    run = _selected_ranking_run(self)
    has_run = run is not None
    has_path = bool(str(ref.get("model_path", "") or "").strip())
    promotable = False
    fine_tunable = False
    if has_run:
        try:
            promotable = bool(
                CAMPAIGN.get_active_project_name()
                and str(getattr(run, "status", "") or "").strip().lower() == TrainingStatus.COMPLETED.value
                and self._infer_history_run_target(run) in {"plate", "char"}
                and self._resolve_history_run_best_weights(run) is not None
                and self._does_history_run_match_active_campaign_target(run)
            )
        except Exception:
            promotable = False
        try:
            fine_tunable = bool(self._is_history_run_fine_tune_candidate(run))
        except Exception:
            fine_tunable = False

    try:
        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(
            label=f"Podepnij jako wynik bramki {_step4_finish_gate_display_id()}",
            command=self._use_selected_ranking_model_as_campaign_result,
            state=(tk.NORMAL if promotable else tk.DISABLED),
        )
        menu.add_command(
            label="Dotrenuj od tego modelu",
            command=self._select_selected_ranking_run_as_fine_tune_base,
            state=(tk.NORMAL if fine_tunable else tk.DISABLED),
        )
        menu.add_command(
            label="Szczegóły runu",
            command=self._open_selected_ranking_run_details,
            state=(tk.NORMAL if has_run else tk.DISABLED),
        )
        menu.add_command(
            label="Otwórz folder wag",
            command=self._open_selected_ranking_model_folder,
            state=(tk.NORMAL if has_path or has_run else tk.DISABLED),
        )
        menu.add_command(
            label="[ TEST ] Waliduj ten model",
            command=self._open_selected_ranking_validation_modal,
            state=(tk.NORMAL if has_path else tk.DISABLED),
        )
        menu.add_separator()
        menu.add_command(
            label="Kopiuj podpis wyboru",
            command=self._copy_selected_ranking_choice_label,
            state=(tk.NORMAL if str(ref.get("choice_label", "") or "").strip() else tk.DISABLED),
        )
        menu.tk_popup(event.x_root, event.y_root)
    except Exception:
        pass
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass

def _autofill_validation_inputs_from_run(self, run):
    if run is None:
        return

    model_candidates = []
    for candidate in (
        getattr(run, "best_weights", ""),
        getattr(run, "last_weights", ""),
    ):
        candidate_str = str(candidate or "").strip()
        if candidate_str:
            model_candidates.append(Path(candidate_str))

    selected_model = next((path for path in model_candidates if path.exists()), None)
    if selected_model is not None and hasattr(self, "val_model_var"):
        try:
            self.val_model_var.set(str(selected_model))
        except Exception:
            pass

    dataset_value = str(getattr(run, "dataset_path", "") or "").strip()
    if dataset_value and hasattr(self, "val_data_var"):
        dataset_path = Path(dataset_value)
        if dataset_path.exists():
            try:
                self.val_data_var.set(str(dataset_path))
            except Exception:
                pass

def _close_run_details_dialog(self):
    dialog = getattr(self, "_run_details_dialog", None)
    if dialog is None:
        return
    try:
        dialog.withdraw()
    except Exception:
        pass

def _open_current_run_details_analysis(self):
    run = None
    run_id = str(getattr(self, "_run_details_current_run_id", "") or "").strip()
    if run_id:
        try:
            run = self.history.get_run(run_id)
        except Exception:
            run = None
    if run is None:
        run = self._selected_run()
    if run is None:
        return
    return self._open_run_analysis_window(run)

def _open_current_run_details_folder(self):
    run = None
    run_id = str(getattr(self, "_run_details_current_run_id", "") or "").strip()
    if run_id:
        try:
            run = self.history.get_run(run_id)
        except Exception:
            run = None
    if run is None:
        run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run z historii treningu.",
        )
    output_dir = str(getattr(run, "output_dir", "") or "").strip()
    if not output_dir:
        return messagebox.showwarning(
            "Brak folderu",
            "Wybrany run nie ma zapisanego folderu wynikowego.",
        )
    return self._open_path(Path(output_dir))

def _open_run_details_modal(self, run):
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run z historii treningu.",
        )

    run_id = str(getattr(run, "id", "") or "").strip()
    run_name = str(getattr(run, "name", "") or run_id or "run").strip()
    run_ref = build_run_display_ref(run, kind_hint="training")
    palette = getattr(self.app, "palette", {}) if getattr(self, "app", None) is not None else {}

    dialog = getattr(self, "_run_details_dialog", None)
    dialog_exists = False
    if dialog is not None:
        try:
            dialog_exists = bool(dialog.winfo_exists())
        except Exception:
            dialog_exists = False

    if not dialog_exists:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Szczegóły runu treningowego")
        dialog.geometry("1180x760")
        dialog.minsize(980, 640)
        dialog.transient(self.frame.winfo_toplevel())
        dialog.resizable(True, True)
        dialog.protocol("WM_DELETE_WINDOW", self._close_run_details_dialog)
        self._run_details_dialog = dialog

        shell = ttk.Frame(dialog, padding=12, style="Panel.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell, style="Panel.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        header.columnconfigure(0, weight=1)
        self.run_details_title_lbl = ttk.Label(
            header,
            text="Szczegóły runu",
            style="PanelTitle.TLabel",
            anchor=tk.W,
        )
        self.run_details_title_lbl.grid(row=0, column=0, sticky="ew")
        self.run_details_subtitle_lbl = ttk.Label(
            header,
            text="Konfiguracja i wyniki modelu są dostępne tutaj, a historia zostaje czysta do szybkiego wyboru.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
        )
        self.run_details_subtitle_lbl.grid(row=1, column=0, sticky="ew", pady=(3, 0))

        body = ttk.Frame(shell, style="Panel.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        config_box = ttk.LabelFrame(body, text=" Konfiguracja runu ", padding=8)
        config_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        metric_box = ttk.LabelFrame(body, text=" Wyniki modelu ", padding=8)
        metric_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.run_details_config_tree = self._create_metric_table(
            config_box,
            [
                ("Pole", 230, tk.W),
                ("Wartość", 820, tk.W),
            ],
            height=8,
        )
        self.run_details_metric_tree = self._create_metric_table(
            metric_box,
            [
                ("Metryka", 260, tk.W),
                ("Ostatnia", 150, tk.CENTER),
                ("Najlepsza", 150, tk.CENTER),
                ("Ocena", 220, tk.CENTER),
            ],
            height=9,
        )

        actions = ttk.Frame(shell, style="Panel.TFrame")
        actions.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(
            actions,
            text="Pokaż wykresy treningu",
            command=self._open_current_run_details_analysis,
            style="WorkflowCard.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            actions,
            text="Otwórz folder runu",
            command=self._open_current_run_details_folder,
            style="WorkflowCard.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            actions,
            text="[ TEST ] Waliduj model",
            command=self._open_current_run_validation_modal,
            style="WorkflowCard.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            actions,
            text="Zamknij",
            command=self._close_run_details_dialog,
        ).pack(side=tk.RIGHT)

    self._run_details_current_run_id = run_id
    try:
        self.run_details_title_lbl.configure(text=f"Szczegóły runu: {run_ref.id or self._shorten_training_text(run_name, 72)}")
    except Exception:
        pass
    try:
        target_label = self._format_history_run_target_label(self._infer_history_run_target(run))
        status_label = self._format_history_run_status_label(run)
        self.run_details_subtitle_lbl.configure(text=f"{target_label} | {status_label}")
    except Exception:
        pass

    self._set_metric_table_rows(
        getattr(self, "run_details_config_tree", None),
        self._build_training_run_detail_rows(run),
    )
    self._set_metric_table_rows(
        getattr(self, "run_details_metric_tree", None),
        self._build_training_run_metric_rows(run),
    )

    try:
        dialog.title(f"Szczegóły runu | {run_ref.id or self._shorten_training_text(run_name, 48)}")
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

def _open_selected_run_details(self, event=None):
    if event is not None and hasattr(self, "tree") and getattr(event, "y", None) is not None:
        try:
            row_id = self.tree.identify_row(event.y)
        except Exception:
            row_id = ""
        if not row_id:
            return
        try:
            self.tree.selection_set(row_id)
            self.tree.focus(row_id)
        except Exception:
            pass
        self._on_run_selected()

    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run z historii treningu.",
        )
    return self._open_run_details_modal(run)

def _on_run_selected(self, event=None):
    run = self._selected_run()
    self._set_history_run_tables(run)
    if run is None:
        return
    self._autofill_validation_inputs_from_run(run)

def _run_validation(self):
    if not YOLO_AVAILABLE:
        return messagebox.showerror("Błąd", "Brak modułu YOLO!")
    YoloClass = get_yolo_class()
    if YoloClass is None:
        return messagebox.showerror("Błąd", "Nie udało się załadować modułu YOLO.")
    if self.val_is_running: return

    model_path = self.val_model_var.get().strip()
    data_path = self.val_data_var.get().strip()

    if not Path(model_path).exists(): return messagebox.showerror("Błąd", "Wskazany plik modelu nie istnieje.")
    if Path(data_path).is_dir() and (Path(data_path)/"data.yaml").exists():
        data_path = str(Path(data_path)/"data.yaml")
    if not Path(data_path).exists() or not data_path.endswith(".yaml"):
        return messagebox.showerror("Błąd", "Wskaż plik data.yaml lub folder zawierający ten plik.")

    if not self._begin_step4_operation("z4.validation.run", "Z4: walidacja modelu"):
        return

    self.val_is_running = True
    self.btn_run_val.config(state=tk.DISABLED, text="[ START ] Walidacja w toku...")
    self.val_status.config(text="Walidacja w toku...", foreground="#d35400")
    self._set_validation_summary(
        f"Trwa walidacja: {Path(model_path).name} | split: {self.val_split_var.get()}",
        [],
        "Model jest sprawdzany na wskazanym splicie. Po zakończeniu odblokuje się tabela wyników.",
    )
    self._set_step4_process_console_text(
        f"Inicjalizowanie silnika YOLO do ewaluacji...\n"
        f"Model: {Path(model_path).name}\n"
        f"Dataset: {Path(data_path).parent.name}\n\n"
    )

    def worker():
        try:
            model = YoloClass(model_path)
            metrics = model.val(data=data_path, split=self.val_split_var.get())
            metric_rows = self._extract_validation_metric_rows(metrics)

            res = "\n=== OFICJALNE WYNIKI WALIDACJI YOLO ===\n"

            if hasattr(metrics, 'results_dict'):
                for k, v in metrics.results_dict.items():
                    res += f"• {k}: {v:.4f}\n"
            else:
                if hasattr(metrics, 'box'):
                    res += f"• mAP50:     {metrics.box.map50:.4f}\n"
                    res += f"• mAP50-95:  {metrics.box.map:.4f}\n"
                    res += f"• Precision: {metrics.box.mp:.4f} (Mean Precision)\n"
                    res += f"• Recall:    {metrics.box.mr:.4f} (Mean Recall)\n"
                elif hasattr(metrics, 'pose'):
                    res += f"• Pose mAP50: {metrics.pose.map50:.4f}\n"
                    res += f"• Pose mAP:   {metrics.pose.map:.4f}\n"
                    if hasattr(metrics, 'box'):
                        res += f"• Box mAP50:  {metrics.box.map50:.4f}\n"
                else:
                    res += str(metrics)

            self._append_train_log(res.rstrip())
            self._ui(lambda: self.val_status.config(text="Walidacja zakończona.", foreground="green"))
            self._ui(
                lambda rows=metric_rows, model_name=Path(model_path).name, split_name=self.val_split_var.get():
                self._set_validation_summary(
                    f"Walidacja zakończona: {model_name} | split: {split_name}",
                    rows,
                    "Otwórz tabelę metryk, aby przejrzeć wartości zwrócone przez YOLO.",
                )
            )
            self._ui(lambda: messagebox.showinfo("Sukces", "Walidacja zakończona pomyślnie!"))

        except Exception as e:
            self._append_train_log(f"\nBŁĄD WALIDACJI:\n{e}")
            self._ui(lambda: self.val_status.config(text="Błąd walidacji", foreground="red"))
            self._ui(
                lambda err=str(e), model_name=Path(model_path).name:
                self._set_validation_summary(
                    f"Walidacja nie powiodła się: {model_name}",
                    [("Błąd", self._shorten_training_text(err, 72), "-", "-")],
                    err,
                )
            )
            logger.error(f"Validation error: {e}")

        finally:
            self.val_is_running = False
            self._end_step4_operation("z4.validation.run")
            self._ui(lambda: self.btn_run_val.config(state=tk.NORMAL, text="[ START ] Uruchom walidację"))
            self._ui(self._refresh_training_start_state)

    threading.Thread(target=worker, daemon=True).start()

def _load_ranking(self):
    if not hasattr(self, "rank_tree"):
        return
    self._ensure_plate_ranking_engine()
    target = self._get_ranking_task_target()
    target_task = self._get_ranking_task_label(target)
    get_unique_entries = getattr(self.ranking_engine, "get_unique_entries", None)
    if callable(get_unique_entries):
        entries = get_unique_entries()
    else:
        entries = getattr(self.ranking_engine, 'entries', [])
    self.rank_tree.delete(*self.rank_tree.get_children())
    self._ranking_tree_entry_refs = {}

    selected_reference = self._resolve_ranking_reference_source()
    selected_reference_path = str(selected_reference.get("reference_dir") or "").strip()
    selected_reference_raw = str(selected_reference.get("selected_path") or "").strip()
    selected_split = str(selected_reference.get("split_name") or "").strip()
    selected_scope = self._get_ranking_scope()
    models_dir_raw = str(getattr(getattr(self, "rank_models_dir", None), "get", lambda: "")() or "").strip()
    try:
        models_dir = Path(models_dir_raw) if models_dir_raw else None
    except Exception:
        models_dir = None

    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        project_root = Path(project_root).resolve() if project_root is not None else None
    except Exception:
        project_root = None

    def normalize_path(path_like: str) -> str:
        raw = str(path_like or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).resolve())
        except Exception:
            return str(Path(raw))

    project_model_paths: set[str] = set()
    if CAMPAIGN.get_active_project_name():
        try:
            for candidate in self._collect_project_ranking_model_candidates(target):
                try:
                    project_model_paths.add(str(Path(candidate).resolve()).lower())
                except Exception:
                    project_model_paths.add(str(Path(candidate)).lower())
        except Exception:
            project_model_paths = set()

    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass
    ranking_run_cache: dict[str, object | None] = {}

    def entry_training_run(entry):
        raw = str(getattr(entry, "model_path", "") or "").strip()
        if not raw:
            return None
        try:
            key = str(Path(raw).resolve()).lower()
        except Exception:
            key = str(Path(raw)).lower()
        if key in ranking_run_cache:
            return ranking_run_cache[key]
        run = None
        resolver = getattr(self, "_resolve_training_run_from_model_path", None)
        if callable(resolver):
            try:
                run = resolver(Path(raw))
            except Exception:
                run = None
        ranking_run_cache[key] = run
        return run

    def entry_labels(entry, scope: str) -> tuple[str, str, object | None]:
        run = entry_training_run(entry)
        model_path_raw = str(getattr(entry, "model_path", "") or "").strip()
        model_file = Path(model_path_raw).name if model_path_raw else str(getattr(entry, "model_name", "") or "-")
        if run is not None:
            try:
                choice_label = _campaign_training_result_choice_label(self, run)
            except Exception:
                choice_label = self._format_training_model_run_label(run)
            return choice_label, model_file or "best.pt", run
        fallback = format_ranking_model_label(
            getattr(entry, "model_name", ""),
            getattr(entry, "model_path", ""),
            getattr(entry, "task_type", ""),
        )
        if scope == "Projekt":
            return fallback, model_file or fallback, None
        return fallback, model_file or fallback, None

    def entry_scope(entry) -> str:
        model_path_raw = str(getattr(entry, "model_path", "") or "").strip()
        if not model_path_raw:
            return "Globalne"
        return model_path_scope(model_path_raw)

    def model_path_key(path_like) -> str:
        raw = str(path_like or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).resolve()).lower()
        except Exception:
            return str(Path(raw)).lower()

    def model_path_scope(path_like) -> str:
        model_path_raw = str(path_like or "").strip()
        if not model_path_raw:
            return "Globalne"
        try:
            model_path = Path(model_path_raw).resolve()
            model_key = str(model_path).lower()
            if model_key in project_model_paths:
                return "Projekt"
            if project_root is not None and (model_path == project_root or project_root in model_path.parents):
                return "Projekt"
        except Exception:
            pass
        return "Globalne"

    def candidate_labels(model_path: Path, scope: str) -> tuple[str, str, object | None]:
        run = None
        resolver = getattr(self, "_resolve_training_run_from_model_path", None)
        if callable(resolver):
            try:
                run = resolver(Path(model_path))
            except Exception:
                run = None
        model_file = Path(model_path).name
        if run is not None:
            try:
                return _campaign_training_result_choice_label(self, run), model_file or "best.pt", run
            except Exception:
                try:
                    return self._format_training_model_run_label(run), model_file or "best.pt", run
                except Exception:
                    pass
        fallback = format_ranking_model_label(model_file, str(model_path), target_task)
        return fallback, model_file or fallback, run

    candidate_paths: list[Path] = []
    try:
        if selected_scope in {"Projekt", "Wszystkie"}:
            candidate_paths = list(self._collect_ranking_participant_candidates(models_dir, target, selected_scope) or [])
        elif models_dir is not None and models_dir.exists() and models_dir.is_dir():
            candidate_paths = list(self._collect_ranking_participant_candidates(models_dir, target, selected_scope) or [])
    except Exception:
        candidate_paths = []
    try:
        candidate_paths = list(self._filter_enabled_ranking_participants(candidate_paths) or [])
    except Exception:
        pass

    candidate_by_key: dict[str, Path] = {}
    for candidate_path in candidate_paths:
        key = model_path_key(candidate_path)
        if not key:
            continue
        scope = model_path_scope(candidate_path)
        if selected_scope in {"Projekt", "Globalne"} and scope != selected_scope:
            continue
        candidate_by_key.setdefault(key, Path(candidate_path))

    def entry_decision(entry, scope: str, index: int) -> str:
        if index == 0:
            if scope == "Projekt":
                return "WYGRANY - wybierz jawnie"
            return "WYGRANY referencyjny"
        if scope == "Projekt":
            return "Kandydat projektu"
        if scope == "Globalne":
            return "Model referencyjny"
        return "Kandydat"

    def entry_reference_label(entry) -> str:
        raw_path = str(getattr(entry, "reference_path", "") or "").strip()
        raw_name = str(getattr(entry, "reference_name", "") or "").strip()
        if target == "char" and raw_path:
            try:
                return build_dataset_display_ref(raw_path, target_hint="char").id
            except Exception:
                pass
        return raw_name or (Path(raw_path).name if raw_path else "-") or "-"

    def pending_reference_label() -> str:
        raw_path = selected_reference_path
        raw_name = str(selected_reference.get("reference_name") or "").strip()
        if target == "char" and raw_path:
            try:
                return build_dataset_display_ref(raw_path, target_hint="char").id
            except Exception:
                pass
        return raw_name or (Path(raw_path).name if raw_path else "-") or "-"

    def ranking_date_label(entry) -> str:
        raw = str(getattr(entry, "date_evaluated", "") or "").strip()
        if not raw:
            return "-"
        try:
            return datetime.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return raw.replace("T", " ")[:16] or "-"

    def set_leader(title: str, hint: str, *, tone: str = "muted"):
        title_label = getattr(self, "rank_leader_title", None)
        hint_label = getattr(self, "rank_leader_hint", None)
        palette = getattr(getattr(self, "app", None), "palette", {}) or {}
        tone_fg = {
            "success": palette.get("success", "#2ecc71"),
            "warning": palette.get("warning", "#f0b44c"),
            "danger": palette.get("error", "#e05d5d"),
            "muted": palette.get("fg", "#f3f3f3"),
        }.get(tone, palette.get("fg", "#f3f3f3"))
        if title_label is not None:
            try:
                title_label.configure(text=title, fg=tone_fg)
            except Exception:
                pass
        if hint_label is not None:
            try:
                hint_label.configure(text=hint)
            except Exception:
                pass

    def insert_pending_candidate_rows(ranked_keys: set[str] | None = None, *, start_index: int = 1) -> None:
        ranked_keys = ranked_keys or set()
        row_index = start_index
        for key, candidate_path in sorted(candidate_by_key.items(), key=lambda item: str(item[1]).lower()):
            if key in ranked_keys:
                continue
            scope = model_path_scope(candidate_path)
            result_label, _model_label, run = candidate_labels(candidate_path, scope)
            tag = "pending_project" if scope == "Projekt" else "pending_global"
            item_id = self.rank_tree.insert("", tk.END, values=(
                f"#{row_index}",
                result_label,
                scope,
                "-",
                "-",
                "-",
                "czeka na test",
            ), tags=(tag,))
            row_index += 1
            try:
                self._ranking_tree_entry_refs[item_id] = {
                    "run_id": str(getattr(run, "id", "") or "").strip() if run is not None else "",
                    "model_path": str(candidate_path),
                    "choice_label": result_label,
                    "scope": scope,
                }
            except Exception:
                pass

    filtered_entries = [e for e in entries if getattr(e, 'task_type', '') == target_task]
    if selected_reference_raw and not selected_reference.get("ok"):
        filtered_entries = []
    elif selected_reference_path:
        filtered_entries = [
            e for e in filtered_entries
            if normalize_path(getattr(e, "reference_path", "")) == normalize_path(selected_reference_path)
        ]
        if target == "char" and selected_split:
            filtered_entries = [
                e for e in filtered_entries
                if str(getattr(e, "split_name", "") or "").strip() == selected_split
            ]
    if selected_scope in {"Projekt", "Globalne"}:
        filtered_entries = [e for e in filtered_entries if entry_scope(e) == selected_scope]
    filtered_entries.sort(key=lambda x: getattr(x, 'ranking_score', getattr(x, 'f1_score', 0)), reverse=True)

    if not filtered_entries and not candidate_by_key:
        scope_hint = selected_scope.lower()
        if selected_reference_raw and not selected_reference.get("ok"):
            set_leader(
                "Brak wyników: zestaw odniesienia nie jest gotowy",
                str(selected_reference.get("message") or "Wskaż poprawny folder runu odniesienia."),
                tone="warning",
            )
        else:
            set_leader(
                f"Brak wyników dla zakresu: {selected_scope}",
                f"Uruchom ranking albo przełącz zakres. Zakres {scope_hint} nie zawiera jeszcze porównanych modeli dla trybu {target_task}.",
                tone="muted",
            )
        return

    if not filtered_entries:
        if selected_reference_raw and not selected_reference.get("ok"):
            set_leader(
                "Kandydaci są, ale zestaw odniesienia nie jest gotowy",
                str(selected_reference.get("message") or "Wskaż poprawny folder runu odniesienia przed testem rankingowym."),
                tone="warning",
            )
        else:
            set_leader(
                "Kandydaci czekają na test rankingowy",
                (
                    f"Zakres: {selected_scope}. Tabela pokazuje modele, które wezmą udział w porównaniu. "
                    "Po uruchomieniu rankingu te wiersze dostaną metryki i kolejność."
                ),
                tone="warning",
            )
        insert_pending_candidate_rows()
        return

    best = filtered_entries[0]
    best_scope = entry_scope(best)
    best_reference = entry_reference_label(best)
    best_result_label, _best_model_label, best_run = entry_labels(best, best_scope)
    best_score = float(getattr(best, 'ranking_score', getattr(best, 'f1_score', 0)) or 0)
    if best_run is not None:
        best_title = f"Wygrywa: {best_result_label} | ocena {best_score:.1f}%"
        best_hint = (
            f"Zakres: {best_scope}. Zestaw odniesienia: {best_reference}. "
            f"Tego samego podpisu szukaj w sekcji wyboru wyniku bramki {_step4_finish_gate_display_id()}. "
            "Ranking nie wybiera modelu automatycznie."
        )
    else:
        best_title = f"Wygrywa: {best_result_label} | ocena {best_score:.1f}%"
        best_hint = (
            f"Zakres: {best_scope}. Zestaw odniesienia: {best_reference}. "
            "To kandydat spoza historii projektu, więc nie ma podpisu runu z wyboru wyniku."
        )
    set_leader(
        best_title,
        best_hint,
        tone="success" if best_scope == "Projekt" else "warning",
    )

    ranked_model_keys: set[str] = set()
    for i, rep in enumerate(filtered_entries):
        scope = entry_scope(rep)
        row_tags = ("leader",) if i == 0 else ("project" if scope == "Projekt" else "global",)
        result_label, _model_label, run = entry_labels(rep, scope)
        score = float(getattr(rep, 'ranking_score', getattr(rep, 'f1_score', 0)) or 0)
        precision = float(getattr(rep, 'precision', 0) or 0)
        recall = float(getattr(rep, 'recall', 0) or 0)
        item_id = self.rank_tree.insert("", tk.END, values=(
            "WYGRANY" if i == 0 else f"#{i + 1}",
            result_label,
            scope,
            ranking_date_label(rep),
            f"{score:.1f}%",
            f"{precision:.1f}/{recall:.1f}" if (precision > 0 or recall > 0) else "-",
            entry_decision(rep, scope, i),
        ), tags=row_tags)
        try:
            self._ranking_tree_entry_refs[item_id] = {
                "run_id": str(getattr(run, "id", "") or "").strip() if run is not None else "",
                "model_path": str(getattr(rep, "model_path", "") or "").strip(),
                "choice_label": result_label,
                "scope": scope,
            }
        except Exception:
            pass
        ranked_model_keys.add(model_path_key(getattr(rep, "model_path", "")))

    insert_pending_candidate_rows(ranked_model_keys, start_index=len(filtered_entries) + 1)
