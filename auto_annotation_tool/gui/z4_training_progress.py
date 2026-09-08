#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 training progress and failure diagnostics extracted from tab_training.py."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _format_training_eta(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except Exception:
        return "-"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def _reset_training_runtime_progress(self) -> None:
    self._training_started_monotonic = None
    self._training_started_wall_clock = None
    self._training_eta_seconds = None
    self._training_last_epoch = 0
    self._training_last_total_epochs = 0
    self._training_last_batch = 0
    self._training_last_total_batches = 0
    self._last_training_batch_ui_emit_at = 0.0
    try:
        self.train_epoch_progress_measure_lbl.configure(text="Bieżąca epoka")
    except Exception:
        pass
    try:
        self.train_epoch_progress_hint_lbl.configure(
            text="Po starcie zobaczysz numer epoki i liczbę batchy w bieżącej epoce."
        )
    except Exception:
        pass
    try:
        self.train_progress_measure_lbl.configure(text="Cały run")
    except Exception:
        pass
    try:
        self.train_progress_hint_lbl.configure(
            text="Po starcie pojawi się szacowany czas do końca treningu."
        )
    except Exception:
        pass

def _update_training_progress_meta(
    self,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    batch_idx: int | None = None,
    total_batches: int | None = None,
    overall_pct: float | None = None,
    epoch_pct: float | None = None,
    eta_seconds: float | None = None,
) -> None:
    if epoch is not None:
        self._training_last_epoch = max(0, int(epoch))
    if total_epochs is not None:
        self._training_last_total_epochs = max(0, int(total_epochs))
    if batch_idx is not None:
        self._training_last_batch = max(0, int(batch_idx))
    if total_batches is not None:
        self._training_last_total_batches = max(0, int(total_batches))
    if eta_seconds is not None:
        self._training_eta_seconds = max(0.0, float(eta_seconds))

    epoch_value = int(self._training_last_epoch or 0)
    total_epoch_value = int(self._training_last_total_epochs or 0)
    batch_value = int(self._training_last_batch or 0)
    total_batch_value = int(self._training_last_total_batches or 0)

    if total_epoch_value > 0 and total_batch_value > 0:
        epoch_text = f"Epoka {epoch_value}/{total_epoch_value} | batch {batch_value}/{total_batch_value}"
    elif total_epoch_value > 0:
        epoch_text = f"Epoka {epoch_value}/{total_epoch_value}"
    else:
        epoch_text = "Bieżąca epoka"
    try:
        self.train_epoch_progress_measure_lbl.configure(text=epoch_text)
    except Exception:
        pass

    if total_batch_value > 0:
        safe_epoch_pct = max(0.0, min(100.0, float(epoch_pct or 0.0)))
        epoch_hint = f"Bieżąca epoka: {safe_epoch_pct:.1f}% | batch {batch_value}/{total_batch_value}"
    else:
        epoch_hint = "Przygotowanie batchy dla bieżącej epoki..."
    try:
        self.train_epoch_progress_hint_lbl.configure(text=epoch_hint)
    except Exception:
        pass

    if total_epoch_value > 0:
        overall_text = f"Cały run {max(0.0, min(100.0, float(overall_pct or 0.0))):.1f}%"
    else:
        overall_text = "Cały run"
    try:
        self.train_progress_measure_lbl.configure(text=overall_text)
    except Exception:
        pass

    eta_text = self._format_training_eta(self._training_eta_seconds)
    started_text = (
        self._training_started_wall_clock.strftime("%H:%M:%S")
        if isinstance(self._training_started_wall_clock, datetime.datetime)
        else "--:--:--"
    )
    try:
        self.train_progress_hint_lbl.configure(text=f"Start: {started_text} | ETA: {eta_text}")
    except Exception:
        pass

def _is_memory_failure_text(self, message: str | None) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    needles = (
        "out of memory",
        "outofmemory",
        "memory allocation failure",
        "unable to allocate",
        "defaultcpuallocator: not enough memory",
        "not enough memory",
        "taskalignedassigner",
        "cuda error: unknown error",
    )
    return any(needle in normalized for needle in needles)

def _is_cuda_runtime_broken_text(self, message: str | None) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    needles = (
        "cuda error: unknown error",
        "unable to find an engine to execute this computation",
        "get was unable to find an engine to execute this computation",
        "utracił sprawny stan cuda",
    )
    return any(needle in normalized for needle in needles)

def _get_training_gpu_memory_snapshot(self, device_value: str | None = None) -> dict:
    snapshot = {
        "available": False,
        "device_name": "",
        "free_mib": 0.0,
        "used_mib": 0.0,
        "total_mib": 0.0,
        "source": "",
        "error": "",
    }

    effective_raw, profile = self._get_effective_training_device_profile(device_value)
    if effective_raw == "cpu" or profile is None:
        snapshot["error"] = "Trening nie używa GPU CUDA."
        return snapshot

    device_name = str(profile.get("name", effective_raw) or effective_raw)
    snapshot["device_name"] = device_name

    try:
        device_index = int(profile.get("index", 0) or 0)
    except Exception:
        device_index = 0

    torch = get_torch_module()
    if is_cuda_available() and torch is not None:
        try:
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                free_mib = float(free_bytes) / (1024.0 ** 2)
                total_mib = float(total_bytes) / (1024.0 ** 2)
                used_mib = max(0.0, total_mib - free_mib)
                snapshot.update(
                    available=True,
                    free_mib=free_mib,
                    used_mib=used_mib,
                    total_mib=total_mib,
                    source="torch.cuda.mem_get_info",
                )
                return snapshot
        except Exception as e:
            snapshot["error"] = str(e)

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        output = str(completed.stdout or "").strip()
        if completed.returncode == 0 and output:
            rows = [row.strip() for row in output.splitlines() if row.strip()]
            if rows:
                chosen_row = rows[min(device_index, len(rows) - 1)]
                parts = [part.strip() for part in chosen_row.split(",")]
                if len(parts) >= 4:
                    snapshot.update(
                        available=True,
                        device_name=str(parts[0] or device_name),
                        total_mib=float(parts[1] or 0.0),
                        used_mib=float(parts[2] or 0.0),
                        free_mib=float(parts[3] or 0.0),
                        source="nvidia-smi",
                        error="",
                    )
                    return snapshot
        if not snapshot["error"]:
            snapshot["error"] = str(completed.stderr or "Nie udało się odczytać danych z nvidia-smi.").strip()
    except Exception as e:
        if not snapshot["error"]:
            snapshot["error"] = str(e)

    return snapshot

def _build_training_gpu_memory_lines(self, device_value: str | None = None) -> list[str]:
    snapshot = self._get_training_gpu_memory_snapshot(device_value)
    if not snapshot.get("available"):
        if snapshot.get("error"):
            return [f"Stan GPU: {snapshot['error']}"]
        return []

    device_name = str(snapshot.get("device_name", "") or "GPU CUDA")
    free_mib = float(snapshot.get("free_mib", 0.0) or 0.0)
    used_mib = float(snapshot.get("used_mib", 0.0) or 0.0)
    total_mib = float(snapshot.get("total_mib", 0.0) or 0.0)
    source = str(snapshot.get("source", "") or "").strip()

    line = (
        f"{device_name}: wolne {free_mib:.0f} MiB / {total_mib:.0f} MiB, "
        f"zajęte {used_mib:.0f} MiB."
    )
    if source:
        line += f" (źródło: {source})"
    return [line]

def _build_training_failure_message(self, run) -> str:
    if run is None:
        return (
            "Trening zakończył się błędem.\n\n"
            "Sprawdź terminal procesu i spróbuj ponownie na lżejszych ustawieniach."
        )

    error_text = self._sanitize_training_text(getattr(run, "error_message", "") or "")
    lines = [
        "Co się stało:",
        "Trening nie został ukończony i run ma status FAILED.",
        "",
        "Dlaczego:",
    ]

    if self._is_memory_failure_text(error_text):
        lines.append(
            "Podczas treningu zabrakło pamięci GPU albo pamięci roboczej alokowanej przez PyTorch."
        )
        lines.extend(
            [
                "",
                "Co możesz zrobić:",
                "1. Ustaw batch = 1.",
                "2. Zmniejsz rozdzielczość wejściową do 384 albo nawet 256 dla YOLO Pose tablic na 4 GB VRAM.",
                "3. Jeśli to nadal za dużo, użyj lżejszego modelu startowego n/s zamiast m/l/x.",
                "4. Zamknij inne procesy używające GPU i spróbuj ponownie.",
            ]
        )
        if "defaultcpuallocator" in error_text.lower() or "alloc_cpu" in error_text.lower():
            lines.extend(
                [
                    "5. Ten konkretny błąd pochodzi z alokatora CPU PyTorch i bywa skutkiem ubocznym wcześniejszych OOM na GPU albo wyczerpanej pamięci wirtualnej systemu.",
                    "6. Po takim błędzie najlepiej zamknąć i uruchomić ponownie aplikację przed kolejną próbą treningu.",
                ]
            )
        if self._is_cuda_runtime_broken_text(error_text):
            lines.extend(
                [
                    "7. Zamknij i uruchom ponownie aplikację przed kolejną próbą na GPU.",
                ]
            )
    else:
        lines.append(
            "Run zakończył się wyjątkiem po stronie Ultralytics, CUDA albo konfiguracji treningu."
        )
        lines.extend(
            [
                "",
                "Co możesz zrobić:",
                "1. Sprawdź terminal procesu i ostatni traceback.",
                "2. Spróbuj ponownie na mniejszym batchu lub niższym imgsz.",
                "3. Jeśli wznawiasz z checkpointu, upewnij się, że dataset i model nadal istnieją.",
            ]
        )
        if self._is_cuda_runtime_broken_text(error_text):
            lines.append("4. Zamknij i uruchom ponownie aplikację przed następną próbą GPU.")

    lines.extend(
        [
            "",
            f"Run: {getattr(run, 'name', '-')}",
            f"Preset / plik startowy YOLO: {getattr(run, 'base_model', '-')}",
            f"Dataset: {getattr(run, 'dataset_path', '-')}",
            f"Parametry: batch={int(getattr(run, 'batch_size', 0) or 0)}, imgsz={int(getattr(run, 'img_size', 0) or 0)}, lr0={float(getattr(run, 'lr0', 0.0) or 0.0):.4f}",
        ]
    )
    gpu_lines = self._build_training_gpu_memory_lines(getattr(run, "device", None))
    if gpu_lines:
        lines.extend(["", "Stan GPU przy awarii:"])
        lines.extend(gpu_lines)
    if error_text:
        lines.extend(["", "Szczegóły błędu:", error_text.strip()])
    return "\n".join(lines).strip()
