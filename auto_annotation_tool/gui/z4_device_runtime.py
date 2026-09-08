#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 training device selection and recommendation helpers extracted from tab_training.py."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _scan_training_cuda_devices(self) -> list[dict]:
    # This is a UI query. The shared asynchronous hardware scan owns CUDA access.
    try:
        getter = getattr(self.app, "get_available_yolo_device_profiles", None)
        if callable(getter):
            return getter()
    except Exception:
        pass
    return []

def _get_available_devices(self):
    profiles = self._scan_training_cuda_devices()
    self._training_device_profiles = profiles
    self._training_device_label_map = {}

    auto_label = "Auto"
    cpu_label = "CPU"

    self._training_auto_device_label = auto_label
    self._training_cpu_device_label = cpu_label

    labels = [auto_label, cpu_label]
    self._training_device_label_map[auto_label] = "auto"
    self._training_device_label_map[cpu_label] = "cpu"

    for profile in profiles:
        mem = float(profile.get("memory_gb", 0.0) or 0.0)
        mem_text = f"{mem:.1f} GB VRAM" if mem > 0 else "VRAM ?"
        label = f"GPU/CUDA {profile['index']} - {profile['name']} ({mem_text})"
        labels.append(label)
        self._training_device_label_map[label] = str(profile["raw"])

    return labels

def _get_global_training_device_choice(self) -> str:
    getter = getattr(self.app, "get_global_yolo_device_choice", None)
    if callable(getter):
        try:
            value = str(getter() or "").strip()
            if value:
                return value
        except Exception:
            pass

    try:
        value = str(self.device_var.get() or "").strip()
        if value:
            return value
    except Exception:
        pass

    return "auto"

def apply_global_yolo_device_choice(self, value: str):
    normalized = str(value or "").strip() or "auto"
    normalizer = getattr(self.app, "normalize_global_yolo_device_choice", None)
    if callable(normalizer):
        try:
            normalized = str(normalizer(normalized) or "auto").strip() or "auto"
        except Exception:
            normalized = str(value or "").strip() or "auto"

    if not hasattr(self, "device_var"):
        self.device_var = tk.StringVar(value=normalized)
    else:
        try:
            self.device_var.set(normalized)
        except Exception:
            pass

    try:
        self._refresh_training_device_hint()
    except Exception:
        pass

def _normalize_training_device_choice(self, device_value: str | None = None) -> str:
    value = str(device_value or "").strip()
    if not value:
        return self._training_auto_device_label
    if value in self._training_device_label_map:
        return value

    raw = value.lower()
    if raw == "auto":
        return self._training_auto_device_label
    if raw == "cpu":
        return self._training_cpu_device_label

    for label, mapped in self._training_device_label_map.items():
        if str(mapped).lower() == raw:
            return label

    if raw.startswith("cuda:"):
        raw_token = raw.split()[0]
        try:
            wanted_index = int(raw_token.split(":", 1)[1])
        except Exception:
            wanted_index = 0
        for profile in self._training_device_profiles:
            if int(profile.get("index", -1)) == wanted_index:
                for label, mapped in self._training_device_label_map.items():
                    if mapped == profile.get("raw"):
                        return label

    return self._training_auto_device_label

def _get_selected_training_device_raw(self, device_value: str | None = None) -> str:
    if device_value is None:
        try:
            current_value = self.device_var.get()
        except Exception:
            current_value = self._get_global_training_device_choice()
    else:
        current_value = device_value

    value = str(current_value).strip()
    if not value:
        return "auto"
    if value in self._training_device_label_map:
        return str(self._training_device_label_map.get(value, "auto"))

    raw = value.lower()
    if raw.startswith("auto"):
        return "auto"
    if raw.startswith("cpu"):
        return "cpu"
    if raw.startswith("cuda:"):
        raw_token = raw.split()[0]
        for profile in list(getattr(self, "_training_device_profiles", []) or []):
            if str(profile.get("raw", "")).lower() == raw_token:
                return raw_token
        return "auto"
    if raw in {"auto", "cpu"}:
        return raw
    return "auto"

def _get_effective_training_device_profile(self, device_value: str | None = None) -> tuple[str, dict | None]:
    profiles = self._scan_training_cuda_devices()
    self._training_device_profiles = profiles
    selected_raw = self._get_selected_training_device_raw(device_value)

    if selected_raw == "auto":
        if profiles:
            return str(profiles[0].get("raw", "cuda:0")), profiles[0]
        return "cpu", None

    if selected_raw == "cpu":
        return "cpu", None

    for profile in profiles:
        if str(profile.get("raw")) == selected_raw:
            return selected_raw, profile

    return "cpu", None

def _get_training_device_recommendation(self, device_value: str | None = None) -> dict:
    target = self._get_selected_training_target()
    effective_raw, profile = self._get_effective_training_device_profile(device_value)
    selected_raw = self._get_selected_training_device_raw(device_value)
    hardware_known = bool(getattr(self.app, "_global_yolo_devices_cache_ready", False))
    hardware_error = str(getattr(self.app, "_global_yolo_devices_last_error", "") or "")
    if selected_raw != "cpu" and (
        not hardware_known or hardware_error or (profile and float(profile.get("memory_gb", 0) or 0) <= 0)
    ):
        return {
            "ready": False, "effective_raw": selected_raw,
            "device_name": str((profile or {}).get("name") or "Auto"),
            "note": "Sprawdź sprzęt w menu Konfiguracja, aby uzyskać rekomendację parametrów. Twoje ustawienia pozostają bez zmian.",
        }
    dataset_profile = self._get_training_dataset_profile()
    if dataset_profile.get("pending") or dataset_profile.get("error"):
        return {
            "ready": False, "effective_raw": effective_raw,
            "device_name": str((profile or {}).get("name") or "CPU"),
            "status": "Dataset: analiza w toku" if dataset_profile.get("pending") else "Dataset: brak podsumowania",
            "note": "Rekomendacja wymaga podsumowania datasetu. Trwa analiza w tle."
                    if dataset_profile.get("pending") else "Nie udało się odczytać podsumowania datasetu.",
        }
    model_profile = self._resolve_selected_training_base_model_profile()
    model_bucket = str(model_profile.get("bucket", "s") or "s").strip().lower()
    model_detected_label = str(model_profile.get("detected_label", "") or "").strip()
    model_label = str(model_profile.get("label", "") or "").strip()
    model_identity_label = model_detected_label or model_label
    model_version_label = str(model_profile.get("model_version_label", "") or "").strip()
    model_size_label = str(model_profile.get("model_size_label", "") or "").strip()
    model_params_text = str(model_profile.get("params_text", "") or "").strip()
    train_images = int(dataset_profile.get("train_images", 0) or 0)
    val_images = int(dataset_profile.get("val_images", 0) or 0)
    total_images = int(dataset_profile.get("total_images", 0) or 0)
    median_long_edge = int(dataset_profile.get("median_long_edge", 0) or 0)
    sampled_images = int(dataset_profile.get("sampled_images", 0) or 0)

    if effective_raw == "cpu" or profile is None:
        if target == "plate":
            cpu_imgsz = {
                "n": 512,
                "s": 448,
                "m": 384,
                "l": 320,
                "x": 256,
            }.get(model_bucket, 384)
            cpu_batch = {
                "n": 2,
                "s": 1,
                "m": 1,
                "l": 1,
                "x": 1,
            }.get(model_bucket, 1)
            base_lr0 = 0.003
        else:
            cpu_imgsz = {
                "n": 640,
                "s": 512,
                "m": 448,
                "l": 384,
                "x": 320,
            }.get(model_bucket, 512)
            cpu_batch = {
                "n": 4,
                "s": 3,
                "m": 2,
                "l": 1,
                "x": 1,
            }.get(model_bucket, 2)
            base_lr0 = 0.004

        if model_bucket in {"l", "x"}:
            base_lr0 *= 0.85
        elif model_bucket == "m":
            base_lr0 *= 0.92
        cpu_lr0 = round(max(0.0015, min(0.0060, base_lr0)), 4)

        return {
            "effective_raw": "cpu",
            "device_name": "CPU",
            "memory_gb": 0.0,
            "epochs": 100,
            "batch": cpu_batch,
            "imgsz": cpu_imgsz,
            "lr0": cpu_lr0,
            "model_detected_label": model_detected_label,
            "model_label": model_label,
            "model_identity_label": model_identity_label,
            "model_version_label": model_version_label,
            "model_size_label": model_size_label,
            "model_params_text": model_params_text,
            "model_bucket": model_bucket,
            "note": (
                "Zalecenie awaryjne dla CPU. Uwzględnia tor oraz rozmiar wybranego modelu, "
                "żeby cięższe warianty nie startowały z takimi samymi parametrami jak modele lekkie. "
                "Trening będzie wyraźnie wolniejszy niż na GPU CUDA."
            ),
        }

    memory_gb = float(profile.get("memory_gb", 0.0) or 0.0)
    if target == "char":
        if median_long_edge <= 0:
            desired_imgsz = 640
        elif median_long_edge <= 192:
            desired_imgsz = 416
        elif median_long_edge <= 320:
            desired_imgsz = 512
        elif median_long_edge <= 512:
            desired_imgsz = 576
        elif median_long_edge <= 768:
            desired_imgsz = 640
        else:
            desired_imgsz = 768
        min_imgsz = 416
    else:
        if median_long_edge <= 0:
            desired_imgsz = 768
        elif median_long_edge <= 768:
            desired_imgsz = 640
        elif median_long_edge <= 1024:
            desired_imgsz = 768
        elif median_long_edge <= 1400:
            desired_imgsz = 960
        elif median_long_edge <= 1800:
            desired_imgsz = 1024
        else:
            desired_imgsz = 1280
        min_imgsz = 640

    low_vram_pose_imgsz_cap = None
    if target == "plate" and memory_gb <= 4.5:
        low_vram_pose_imgsz_cap = {
            "n": 384,
            "s": 320,
            "m": 256,
            "l": 256,
            "x": 256,
        }.get(model_bucket, 256)
        desired_imgsz = min(int(desired_imgsz), int(low_vram_pose_imgsz_cap))
        min_imgsz = 256

    if memory_gb <= 4.5:
        base_cap_imgsz = 640
    elif memory_gb <= 6.5:
        base_cap_imgsz = 768
    elif memory_gb <= 8.5:
        base_cap_imgsz = 896
    elif memory_gb <= 12.5:
        base_cap_imgsz = 960
    elif memory_gb <= 16.5:
        base_cap_imgsz = 1024
    else:
        base_cap_imgsz = 1280

    cap_steps = [256, 320, 384, 416, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280]
    cap_minimum = 256 if (target == "plate" and memory_gb <= 4.5) else 384
    cap_anchor = self._nearest_training_imgsz(base_cap_imgsz, minimum=cap_minimum, maximum=1280)
    try:
        cap_index = cap_steps.index(cap_anchor)
    except ValueError:
        cap_index = len(cap_steps) - 1

    model_penalty = {"n": 0, "s": 0, "m": 1, "l": 2, "x": 3}.get(model_bucket, 1)
    task_penalty = 1 if target == "plate" else 0
    if train_images > 0 and train_images < 25:
        dataset_penalty = 2
    elif train_images > 0 and train_images < 80:
        dataset_penalty = 1
    else:
        dataset_penalty = 0
    cap_index = max(0, cap_index - model_penalty - task_penalty - dataset_penalty)
    cap_imgsz = max(min_imgsz, cap_steps[cap_index])
    imgsz = self._nearest_training_imgsz(
        min(desired_imgsz, cap_imgsz),
        minimum=min_imgsz,
        maximum=1280,
    )

    if target == "char":
        if memory_gb <= 4.5:
            base_batch = 8
        elif memory_gb <= 6.5:
            base_batch = 10
        elif memory_gb <= 8.5:
            base_batch = 14
        elif memory_gb <= 12.5:
            base_batch = 18
        elif memory_gb <= 16.5:
            base_batch = 24
        else:
            base_batch = 28
        reference_imgsz = 640.0
        batch_ceiling = 32
    else:
        if memory_gb <= 4.5:
            base_batch = 2
        elif memory_gb <= 6.5:
            base_batch = 4
        elif memory_gb <= 8.5:
            base_batch = 6
        elif memory_gb <= 12.5:
            base_batch = 8
        elif memory_gb <= 16.5:
            base_batch = 10
        else:
            base_batch = 12
        reference_imgsz = 768.0
        batch_ceiling = 16

    if target == "plate" and memory_gb <= 4.5:
        base_batch = 1
        reference_imgsz = 512.0
        batch_ceiling = 1

    model_factor = {"n": 1.15, "s": 1.0, "m": 0.80, "l": 0.65, "x": 0.50}.get(model_bucket, 0.85)
    resolution_factor = (reference_imgsz / float(max(imgsz, 1))) ** 2
    resolution_factor = max(0.25, min(1.25, resolution_factor))
    batch = int(base_batch * model_factor * resolution_factor)

    if target == "plate":
        if total_images < 24:
            dataset_batch_cap = {
                "n": 2,
                "s": 2,
                "m": 1,
                "l": 1,
                "x": 1,
            }.get(model_bucket, 1)
            batch = min(batch, dataset_batch_cap)
        elif total_images < 64:
            dataset_batch_cap = {
                "n": 4,
                "s": 3,
                "m": 2,
                "l": 1,
                "x": 1,
            }.get(model_bucket, 2)
            batch = min(batch, dataset_batch_cap)
        elif total_images < 120:
            dataset_batch_cap = {
                "n": 6,
                "s": 5,
                "m": 4,
                "l": 2,
                "x": 1,
            }.get(model_bucket, 4)
            batch = min(batch, dataset_batch_cap)
    else:
        if total_images < 24:
            batch = min(batch, 4)
        elif total_images < 64:
            batch = min(batch, 8)
        elif total_images < 120:
            batch = min(batch, 12)

    if train_images > 0:
        batch = min(batch, train_images)

    batch = max(1, min(batch_ceiling, int(batch)))

    # Dodatkowy bezpiecznik dla cięższych checkpointów pose na mniejszym VRAM.
    # W praktyce właśnie takie konfiguracje najczęściej wywracają się na plate/pose
    # przy mosaic=1 jeszcze przed końcem pierwszej epoki.
    if (
        target == "plate"
        and model_bucket in {"m", "l", "x"}
        and memory_gb <= 6.5
        and total_images >= 1000
    ):
        batch = 1
        imgsz = min(int(imgsz), 512)

    if target == "plate" and memory_gb <= 4.5:
        batch = 1
        if low_vram_pose_imgsz_cap is None:
            low_vram_pose_imgsz_cap = 256
        imgsz = min(int(imgsz), int(low_vram_pose_imgsz_cap))

    lr0 = 0.0100 if target == "char" else 0.0080
    if batch <= 2:
        lr0 *= 0.55
    elif batch <= 4:
        lr0 *= 0.70
    elif batch <= 8:
        lr0 *= 0.82
    elif batch <= 16:
        lr0 *= 0.92

    if model_bucket in {"l", "x"}:
        lr0 *= 0.85
    elif model_bucket == "m":
        lr0 *= 0.92

    if train_images > 0 and train_images < 50:
        lr0 *= 0.80
    elif train_images > 0 and train_images < 100:
        lr0 *= 0.90

    if target == "plate" and imgsz >= 960:
        lr0 *= 0.90

    lr0 = round(max(0.0025, min(0.0100, lr0)), 4)

    factor_bits = [f"aktywny tor to {self._format_training_target_label(target)}"]
    if model_detected_label:
        factor_bits.append(f"wybrany model to {model_detected_label}")
    elif model_label:
        factor_bits.append(f"wybrany model to {model_label}")
    else:
        factor_bits.append(f"klasa modelu to {model_bucket.upper()}")
    variant_bits = [bit for bit in (model_version_label, model_size_label) if bit]
    if variant_bits:
        factor_bits.append(f"wariant modelu to {' / '.join(variant_bits)}")
    if model_params_text:
        factor_bits.append(f"katalogowo model ma ok. {model_params_text} parametrów")
    factor_bits.append(f"dostępne VRAM to {memory_gb:.1f} GB")
    if sampled_images > 0 and median_long_edge > 0:
        factor_bits.append(f"mediana dłuższego boku obrazu wynosi {median_long_edge}px")
    if train_images > 0 or val_images > 0:
        factor_bits.append(f"split train/val ma układ {train_images}/{val_images}")

    return {
        "effective_raw": effective_raw,
        "device_name": str(profile.get("name", effective_raw)),
        "memory_gb": memory_gb,
        "epochs": 100,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "model_bucket": model_bucket,
        "model_detected_label": model_detected_label,
        "model_label": model_label,
        "model_identity_label": model_identity_label,
        "model_version_label": model_version_label,
        "model_size_label": model_size_label,
        "model_params_text": model_params_text,
        "train_images": train_images,
        "val_images": val_images,
        "total_images": total_images,
        "median_long_edge": median_long_edge,
        "note": (
            "To bezpieczny punkt startowy. Zalecenie uwzględnia, że "
            + ", ".join(factor_bits)
            + ". Jeśli mimo to zabraknie pamięci VRAM, najpierw zmniejsz rozmiar partii, "
              "a dopiero potem rozdzielczość wejściową."
        ),
    }

def _refresh_training_device_hint(self):
    if not bool(getattr(self, "_step4_train_tab_built", False)):
        return
    combo = getattr(self, "device_combo", None)
    if combo is not None:
        try:
            combo.configure(values=self._get_available_devices())
        except Exception:
            pass

    if hasattr(self, "device_var"):
        try:
            normalized = self._get_global_training_device_choice()
            if self.device_var.get() != normalized:
                self.device_var.set(normalized)
        except Exception:
            pass

    label = getattr(self, "train_device_hint_lbl", None)
    if label is not None:
        try:
            label.configure(text="")
            if str(label.winfo_manager()):
                label.pack_forget()
        except Exception:
            pass
    try:
        self._refresh_training_recommendation_table()
    except Exception:
        pass
