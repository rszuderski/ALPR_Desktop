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
    DatasetCreator,
    DatasetSplitter,
    TrainingHistory,
    TrainingStatus,
    YOLOPoseTrainer,
    ensure_yolo_dataset_yaml_points_to_root,
)
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .dataset_display import build_dataset_display_ref
from .model_display import build_model_display_ref
from .run_display import build_run_display_ref
from .z4_dataset_readiness import get_training_dataset_readiness
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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None
def _metric_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def _describe_metric_band(value: float, profile: str) -> tuple[str, str]:
    val = max(0.0, min(1.0, float(value)))
    if profile == "strict_map":
        if val < 0.40:
            return "slaby", "< 0.40"
        if val < 0.60:
            return "używalny", "0.40 - 0.60"
        if val < 0.80:
            return "dobry", "0.60 - 0.80"
        return "bardzo dobry", "> 0.80"

    if profile == "loose_map":
        if val < 0.70:
            return "slaby", "< 0.70"
        if val < 0.85:
            return "używalny", "0.70 - 0.85"
        if val < 0.93:
            return "dobry", "0.85 - 0.93"
        return "bardzo dobry", "> 0.93"

    if val < 0.60:
        return "slaby", "< 0.60"
    if val < 0.75:
        return "używalny", "0.60 - 0.75"
    if val < 0.90:
        return "dobry", "0.75 - 0.90"
    return "bardzo dobry", "> 0.90"

def _build_training_metric_reference_text(self) -> str:
    target = self.get_campaign_training_target()
    if target == "plate":
        return (
            "Progi w tabeli są orientacyjne. Dla tablic najważniejsze jest pose mAP50-95, "
            "a loss porównuj tylko między epokami tego samego treningu."
        )

    return (
        "Progi w tabeli są orientacyjne. Najważniejsze jest mAP50-95, "
        "a loss porównuj tylko między epokami tego samego treningu."
    )

def _refresh_training_metric_reference(self):
    label = getattr(self, "train_metric_reference_lbl", None)
    if label is None:
        return
    self._set_training_widget_text(label, self._build_training_metric_reference_text())

def _set_training_metric_interpretation(self, text: str):
    label = getattr(self, "train_metric_hint_lbl", None)
    if label is None:
        return
    self._set_training_widget_text(label, str(text or "").strip())

def _metric_value_from_epoch_row(self, row: dict, key: str):
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    if value is None and key == "pose_map50_95":
        value = row.get("map50_95")
    elif value is None and key == "pose_map50":
        value = row.get("map50")
    elif value is None and key == "box_map50_95":
        value = row.get("map50_95")
    elif value is None and key == "box_map50":
        value = row.get("map50")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None

def _format_best_epoch_metric_piece(self, row: dict, key: str, label: str) -> str:
    value = self._metric_value_from_epoch_row(row, key)
    if value is None:
        return ""
    return f"{label} {value:.3f}"

def _build_training_best_epoch_summary(self, metrics: dict | None = None, *, epoch: int | None = None) -> str:
    history_rows = [
        dict(row)
        for row in list(getattr(self, "_current_training_metric_history", []) or [])
        if isinstance(row, dict)
    ]

    if isinstance(metrics, dict) and metrics:
        current_row = dict(metrics)
        if epoch is not None and "epoch" not in current_row:
            try:
                current_row["epoch"] = int(epoch)
            except Exception:
                pass
        if current_row:
            current_epoch = str(current_row.get("epoch", "") or "").strip()
            has_same_epoch = any(str(row.get("epoch", "") or "").strip() == current_epoch for row in history_rows)
            if not current_epoch or not has_same_epoch:
                history_rows.append(current_row)

    if not history_rows:
        return "Najlepsza epoka tego runu pojawi się po pierwszej zakończonej epoce."

    target = self.get_campaign_training_target()
    total_epochs = 0
    try:
        run = getattr(getattr(self, "trainer", None), "current_run", None)
        total_epochs = int(getattr(run, "epochs", 0) or 0)
    except Exception:
        total_epochs = 0

    if target == "plate":
        primary_key = "pose_map50_95"
        primary_label = "pose mAP50-95"
        secondary_key = "pose_map50"
        secondary_label = "pose mAP50"
        profile = "strict_map"
    else:
        primary_key = "map50_95"
        primary_label = "mAP50-95"
        secondary_key = "map50"
        secondary_label = "mAP50"
        profile = "strict_map"

    scored_rows: list[tuple[float, dict]] = []
    for row in history_rows:
        value = self._metric_value_from_epoch_row(row, primary_key)
        if value is not None:
            scored_rows.append((value, row))

    if not scored_rows:
        return "Najlepsza epoka tego runu: czekam na główną metrykę po zakończeniu epoki."

    best_value, best_row = max(scored_rows, key=lambda item: item[0])
    best_epoch_raw = best_row.get("epoch")
    try:
        best_epoch = int(float(best_epoch_raw))
    except Exception:
        best_epoch = None

    latest_row = history_rows[-1]
    try:
        latest_epoch = int(float(latest_row.get("epoch", 0) or 0))
    except Exception:
        latest_epoch = None
    is_latest_best = bool(best_epoch is not None and latest_epoch is not None and best_epoch == latest_epoch)

    band, _range_text = self._describe_metric_band(best_value, profile)
    epoch_label = str(best_epoch) if best_epoch is not None else "-"
    if total_epochs > 0:
        epoch_label = f"{epoch_label}/{total_epochs}"

    metric_pieces = [
        f"{primary_label} {best_value:.3f}",
        self._format_best_epoch_metric_piece(best_row, secondary_key, secondary_label),
        self._format_best_epoch_metric_piece(best_row, "loss", "loss"),
    ]
    metric_text = " | ".join(piece for piece in metric_pieces if piece)
    lead = "Nowa najlepsza epoka" if is_latest_best else "Najlepsza epoka dotąd"
    return f"{lead}: {epoch_label} | {metric_text} | ocena: {band}. Porównanie dotyczy tylko tego runu."

def _build_training_metric_interpretation(self, metrics: dict | None) -> str:
    return self._build_training_best_epoch_summary(metrics)

def _shorten_training_text(value, limit: int = 58) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    if len(text) <= int(limit):
        return text
    if int(limit) <= 3:
        return text[:limit]
    return text[: max(1, int(limit) - 3)] + "..."

def _format_training_metric_name(metric_key: str) -> str:
    mapping = {
        "loss": "Loss",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
        "precision": "Precision",
        "recall": "Recall",
        "box_map50": "Box mAP50",
        "box_map50_95": "Box mAP50-95",
        "box_precision": "Box precision",
        "box_recall": "Box recall",
        "pose_map50": "Pose mAP50",
        "pose_map50_95": "Pose mAP50-95",
        "pose_precision": "Pose precision",
        "pose_recall": "Pose recall",
    }
    return mapping.get(str(metric_key or "").strip(), str(metric_key or "").strip() or "Metryka")

def _resolve_training_metric_profile(self, metric_key: str) -> str | None:
    raw = str(metric_key or "").strip().lower()
    if not raw:
        return None
    if raw == "loss":
        return None
    if "map50_95" in raw or raw.endswith("map"):
        return "strict_map"
    if "map50" in raw:
        return "loose_map"
    if "precision" in raw or "recall" in raw:
        return "standard"
    return None

def _format_training_metric_value(self, metric_key: str, value) -> str:
    if value in (None, "", "-"):
        return "-"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if str(metric_key or "").strip().lower() == "loss":
        return f"{numeric:.3f}"
    return f"{numeric:.3f}"

def _format_training_metric_band(self, metric_key: str, value) -> tuple[str, str]:
    raw = str(metric_key or "").strip().lower()
    if value in (None, "", "-"):
        return "-", "-"
    if raw == "loss":
        return "monitoruj", "powinien spadac"
    try:
        numeric = float(value)
    except Exception:
        return "-", "-"

    profile = self._resolve_training_metric_profile(raw)
    if not profile:
        return "-", "-"

    label, range_text = self._describe_metric_band(numeric, profile)
    return label, range_text

def _iter_training_metric_keys(self, target: str | None = None) -> list[str]:
    normalized_target = CONFIG.normalize_task_target(target or self.get_campaign_training_target())
    if normalized_target == "plate":
        return [
            "pose_map50_95",
            "pose_map50",
            "box_map50_95",
            "box_map50",
            "loss",
        ]
    return [
        "map50_95",
        "map50",
        "precision",
        "recall",
        "loss",
    ]

def _build_training_metric_rows(self, metrics: dict | None, target: str | None = None) -> list[tuple[str, str, str, str]]:
    if not isinstance(metrics, dict) or not metrics:
        return []

    rows: list[tuple[str, str, str, str]] = []
    for key in self._iter_training_metric_keys(target):
        value = metrics.get(key)
        if value is None and key == "pose_map50_95":
            value = metrics.get("map50_95")
        elif value is None and key == "pose_map50":
            value = metrics.get("map50")
        elif value is None and key == "box_map50_95":
            value = metrics.get("map50_95")
        elif value is None and key == "box_map50":
            value = metrics.get("map50")

        if value is None:
            continue

        band, range_text = self._format_training_metric_band(key, value)
        rows.append(
            (
                self._format_training_metric_name(key),
                self._format_training_metric_value(key, value),
                band,
                range_text,
            )
        )
    return rows

def _training_status_palette(self) -> dict:
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    panel_alt = palette.get("panel_alt", palette.get("panel", "#252526"))
    return {
        "ok": (
            palette.get("success", "#2ecc71"),
            palette.get("surface_success", blend_hex_colors(palette.get("success", "#2ecc71"), panel_alt, 0.86)),
        ),
        "warn": (
            palette.get("warning", "#f0b44c"),
            palette.get("surface_warning", blend_hex_colors(palette.get("warning", "#f0b44c"), panel_alt, 0.86)),
        ),
        "danger": (
            palette.get("error", palette.get("danger", "#e05d5d")),
            palette.get("surface_error", blend_hex_colors(palette.get("error", "#e05d5d"), panel_alt, 0.86)),
        ),
        "info": (
            palette.get("accent", "#569cd6"),
            palette.get("surface_info", blend_hex_colors(palette.get("accent", "#569cd6"), panel_alt, 0.88)),
        ),
        "muted": (
            palette.get("muted", "#c7c7c7"),
            panel_alt,
        ),
    }

def _configure_training_status_table_tags(self, tree):
    if tree is None:
        return
    for tag, (fg, bg) in _training_status_palette(self).items():
        try:
            tree.tag_configure(f"training_{tag}", foreground=fg, background=bg)
        except Exception:
            pass

def _training_status_tag_from_text(self, text) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "training_muted"
    if "kryt" in raw or "blad" in raw or "błąd" in raw or "slab" in raw or "słab" in raw:
        return "training_danger"
    if "wys" in raw or "uwag" in raw or "uzy" in raw or "uży" in raw or "ywal" in raw:
        return "training_warn"
    if "dobry" in raw or raw == "ok" or "gotow" in raw:
        return "training_ok"
    if "monitor" in raw or "czek" in raw:
        return "training_info"
    return "training_muted"

def _set_training_status_tree_rows(self, tree, rows):
    if tree is None:
        return
    _configure_training_status_table_tags(self, tree)
    try:
        tree.delete(*tree.get_children())
    except Exception:
        pass
    for row in rows or []:
        try:
            values, tag = row
        except Exception:
            values, tag = row, "training_muted"
        try:
            tree.insert("", tk.END, values=tuple(values), tags=(tag,))
        except Exception:
            continue

def _format_training_resource_mib(value) -> str:
    try:
        mib = float(value or 0.0)
    except Exception:
        mib = 0.0
    if mib <= 0:
        return "-"
    if mib >= 1024.0:
        return f"{mib / 1024.0:.2f} GB"
    return f"{mib:.0f} MB"

def _format_training_resource_table_value(value, *, precision: int = 0) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if numeric <= 0:
        return "-"
    if int(precision) <= 0:
        return f"{numeric:.0f}"
    return f"{numeric:.{int(precision)}f}"

def _format_training_resource_percent(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"

def _format_training_resource_percent_value(value) -> str:
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"

def _training_resource_extrema(self, key: str, current) -> tuple[str, str]:
    try:
        numeric = float(current)
    except Exception:
        return "-", "-"
    if numeric <= 0:
        return "-", "-"
    extrema = getattr(self, "_training_resource_live_extrema", None)
    if not isinstance(extrema, dict):
        extrema = {}
        try:
            self._training_resource_live_extrema = extrema
        except Exception:
            pass
    entry = dict(extrema.get(key) or {})
    min_value = float(entry.get("min", numeric) or numeric)
    max_value = float(entry.get("max", numeric) or numeric)
    min_value = min(min_value, numeric)
    max_value = max(max_value, numeric)
    entry["min"] = min_value
    entry["max"] = max_value
    extrema[key] = entry
    return _format_training_resource_table_value(min_value, precision=1 if key.endswith("_pct") else 0), _format_training_resource_table_value(max_value, precision=1 if key.endswith("_pct") else 0)

def _get_training_resource_extrema(self, key: str) -> tuple[str, str]:
    extrema = getattr(self, "_training_resource_live_extrema", None)
    if not isinstance(extrema, dict):
        return "-", "-"
    entry = dict(extrema.get(key) or {})
    return (
        _format_training_resource_table_value(entry.get("min"), precision=1 if key.endswith("_pct") else 0),
        _format_training_resource_table_value(entry.get("max"), precision=1 if key.endswith("_pct") else 0),
    )

def _training_resource_tone(percent, *, warning: float, danger: float) -> tuple[str, str]:
    try:
        value = float(percent)
    except Exception:
        return "Monitoruj", "training_info"
    if value >= float(danger):
        return "Krytycznie", "training_danger"
    if value >= float(warning):
        return "Wysoko", "training_warn"
    return "OK", "training_ok"

def _build_training_resource_sample_rows(self, sample: dict | None) -> list[tuple[tuple[str, str, str, str], str]]:
    if not isinstance(sample, dict) or not sample:
        try:
            self._training_resource_live_extrema = {}
        except Exception:
            pass
        return [
            (("RAM [MiB]", "-", "-", "-", "-", "czekam"), "training_info"),
            (("VRAM [MiB]", "-", "-", "-", "-", "czekam"), "training_info"),
            (("CPU [%]", "-", "-", "-", "-", "czekam"), "training_info"),
            (("Proces [MiB]", "-", "-", "-", "-", "czekam"), "training_info"),
        ]

    system = dict(sample.get("system") or {})
    process = dict(sample.get("process") or {})
    gpu = dict(sample.get("gpu") or {})
    rows: list[tuple[tuple[str, str, str, str], str]] = []

    ram_pct = system.get("ram_percent")
    ram_state, ram_tag = _training_resource_tone(ram_pct, warning=82.0, danger=92.0)
    ram_used = system.get("ram_used_mib")
    ram_min, ram_max = _training_resource_extrema(self, "ram_used_mib", ram_used)
    rows.append(
        (
            (
                "RAM [MiB]",
                _format_training_resource_table_value(ram_used),
                ram_min,
                ram_max,
                _format_training_resource_table_value(system.get("ram_available_mib")),
                ram_state,
            ),
            ram_tag,
        )
    )

    if bool(gpu.get("available")):
        total = float(gpu.get("total_mib") or 0.0)
        reserved = float(gpu.get("reserved_mib") or 0.0)
        allocated = float(gpu.get("allocated_mib") or 0.0)
        used = max(allocated, reserved)
        used_pct = (used / total * 100.0) if total > 0 else None
        gpu_state, gpu_tag = _training_resource_tone(used_pct, warning=82.0, danger=92.0)
        gpu_min, gpu_max = _training_resource_extrema(self, "gpu_used_mib", used)
        rows.append(
            (
                (
                    "VRAM [MiB]",
                    _format_training_resource_table_value(used),
                    gpu_min,
                    gpu_max,
                    _format_training_resource_table_value(gpu.get("free_mib")),
                    gpu_state,
                ),
                gpu_tag,
            )
        )
    else:
        rows.append((("VRAM [MiB]", "-", "-", "-", "-", "niedostępne"), "training_info"))

    cpu_pct = system.get("cpu_percent")
    cpu_state, cpu_tag = _training_resource_tone(cpu_pct, warning=85.0, danger=96.0)
    cpu_min, cpu_max = _training_resource_extrema(self, "cpu_pct", cpu_pct)
    rows.append(
        (
            (
                "CPU [%]",
                _format_training_resource_percent_value(cpu_pct),
                cpu_min,
                cpu_max,
                "-",
                cpu_state,
            ),
            cpu_tag,
        )
    )

    rss = process.get("rss_mib")
    process_min, process_max = _training_resource_extrema(self, "process_rss_mib", rss)
    rows.append(
        (
            (
                "Proces [MiB]",
                _format_training_resource_table_value(rss),
                process_min,
                process_max,
                "-",
                "Monitoruj",
            ),
            "training_info",
        )
    )
    return rows

def _build_training_resource_report_rows(self, report: dict | None) -> list[tuple[tuple[str, str, str, str], str]]:
    if not isinstance(report, dict) or not report:
        return _build_training_resource_sample_rows(self, None)

    last_sample = dict(report.get("last_sample") or {})
    system = dict(last_sample.get("system") or {})
    process = dict(last_sample.get("process") or {})
    gpu = dict(last_sample.get("gpu") or {})
    rows: list[tuple[tuple[str, str, str, str], str]] = []

    ram_pct = report.get("max_ram_percent")
    ram_state, ram_tag = _training_resource_tone(ram_pct, warning=82.0, danger=92.0)
    ram_min, _ram_live_max = _get_training_resource_extrema(self, "ram_used_mib")
    rows.append((
        (
            "RAM [MiB]",
            _format_training_resource_table_value(system.get("ram_used_mib")),
            ram_min,
            _format_training_resource_table_value(report.get("max_ram_used_mib")),
            _format_training_resource_table_value(system.get("ram_available_mib")),
            ram_state,
        ),
        ram_tag,
    ))

    gpu_total = float(report.get("gpu_total_mib") or 0.0)
    gpu_reserved = float(report.get("max_gpu_reserved_mib") or 0.0)
    if gpu_total > 0:
        gpu_pct = gpu_reserved / gpu_total * 100.0
        gpu_state, gpu_tag = _training_resource_tone(gpu_pct, warning=82.0, danger=92.0)
        gpu_min, _gpu_live_max = _get_training_resource_extrema(self, "gpu_used_mib")
        gpu_current = max(float(gpu.get("allocated_mib") or 0.0), float(gpu.get("reserved_mib") or 0.0))
        rows.append(
            (
                (
                    "VRAM [MiB]",
                    _format_training_resource_table_value(gpu_current),
                    gpu_min,
                    _format_training_resource_table_value(gpu_reserved),
                    _format_training_resource_table_value(gpu.get("free_mib") or report.get("min_gpu_free_mib")),
                    gpu_state,
                ),
                gpu_tag,
            )
        )
    else:
        rows.append((("VRAM [MiB]", "-", "-", "-", "-", "brak GPU"), "training_info"))

    cpu_pct = report.get("max_cpu_percent")
    cpu_state, cpu_tag = _training_resource_tone(cpu_pct, warning=85.0, danger=96.0)
    cpu_min, _cpu_live_max = _get_training_resource_extrema(self, "cpu_pct")
    rows.append((("CPU [%]", _format_training_resource_percent_value(system.get("cpu_percent")), cpu_min, _format_training_resource_percent_value(cpu_pct), "-", cpu_state), cpu_tag))

    process_min, _process_live_max = _get_training_resource_extrema(self, "process_rss_mib")
    rows.append((
        (
            "Proces [MiB]",
            _format_training_resource_table_value(process.get("rss_mib")),
            process_min,
            _format_training_resource_table_value(report.get("max_process_rss_mib")),
            "-",
            "Monitoruj",
        ),
        "training_info",
    ))
    return rows

def _set_training_resource_sample(self, sample: dict | None):
    tree = getattr(self, "train_resource_tree", None)
    _set_training_status_tree_rows(self, tree, _build_training_resource_sample_rows(self, sample))

def _set_training_resource_report(self, report: dict | None):
    tree = getattr(self, "train_resource_tree", None)
    _set_training_status_tree_rows(self, tree, _build_training_resource_report_rows(self, report))

def _build_training_run_detail_rows(self, run) -> list[tuple[str, str]]:
    if run is None:
        return []

    try:
        run_ref = build_run_display_ref(run, kind_hint="training")
    except Exception:
        run_ref = None
    run_id = str(getattr(run, "id", "") or "").strip()
    run_name = str(getattr(run, "name", "") or "").strip()
    dataset_path_raw = str(getattr(run, "dataset_path", "") or "").strip()
    if dataset_path_raw:
        try:
            dataset_target = self._infer_dataset_target(dataset_path_raw)
        except Exception:
            dataset_target = ""
        dataset_display = build_dataset_display_ref(
            dataset_path_raw,
            target_hint=dataset_target,
            count_loader=getattr(self, "_get_dataset_split_image_counts", None),
        ).detail_label
    else:
        dataset_display = "-"
    best_weights = self._shorten_training_text(Path(getattr(run, "best_weights", "") or "").name or "-", 36)
    last_weights_raw = str(getattr(run, "last_weights", "") or "").strip()
    last_weights_name = self._shorten_training_text(Path(last_weights_raw).name or "-", 36) if last_weights_raw else "-"
    base_model = self._shorten_training_text(Path(getattr(run, "base_model", "") or "").name or "-", 36)
    def _format_run_datetime(value: str | None) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "-"
        try:
            return datetime.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return raw.replace("T", " ") or "-"

    created_at = _format_run_datetime(getattr(run, "created_at", None))
    started_at = _format_run_datetime(getattr(run, "started_at", None))

    def _metric_float(row: dict, keys: tuple[str, ...]) -> float | None:
        if not isinstance(row, dict):
            return None
        for key in keys:
            try:
                value = row.get(key)
            except Exception:
                value = None
            if value is None or value == "":
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _best_weights_epoch_text() -> str:
        rows = [row for row in list(getattr(run, "metrics_history", []) or []) if isinstance(row, dict)]
        if not rows:
            return ""

        metric_keys = (
            "map50_95",
            "pose_map50_95",
            "box_map50_95",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(P)",
            "metrics/mAP50-95",
        )
        scored_rows: list[tuple[float, dict]] = []
        for row in rows:
            score = _metric_float(row, metric_keys)
            if score is not None:
                scored_rows.append((score, row))

        if not scored_rows:
            fallback_keys = (
                "map50",
                "pose_map50",
                "box_map50",
                "metrics/mAP50(B)",
                "metrics/mAP50(P)",
                "metrics/mAP50",
            )
            for row in rows:
                score = _metric_float(row, fallback_keys)
                if score is not None:
                    scored_rows.append((score, row))

        if not scored_rows:
            return ""

        _score, best_row = max(scored_rows, key=lambda item: item[0])
        epoch_value = _metric_float(best_row, ("epoch", "Epoch"))
        if epoch_value is None:
            return ""
        try:
            return f" | epoka {int(epoch_value)}"
        except Exception:
            return f" | epoka {epoch_value:g}"

    best_weights_detail = f"{best_weights}{_best_weights_epoch_text()}" if best_weights != "-" else best_weights
    resumable = self._is_history_run_resume_allowed(run)
    technically_resumable = self._is_history_run_resumable(run)
    if resumable:
        resume_hint = "TAK - zaznacz ten run w historii, kliknij PPM i wybierz `Wznów trening`."
    elif technically_resumable:
        resume_hint = "NIE - w kampanii można wznowić tylko najnowszy run aktywnego toru."
    else:
        resume_hint = "NIE - brakuje poprawnego checkpointu last.pt."

    lineage_rows: list[tuple[str, str]] = []
    lineage_mode = str(getattr(run, "lineage_mode", "") or "").strip().lower()
    if lineage_mode == "fine_tune":
        parent_run_id = str(getattr(run, "parent_run_id", "") or "").strip() or "-"
        parent_run_display = parent_run_id
        if parent_run_id != "-":
            try:
                parent_run_display = build_run_display_ref({"run_id": parent_run_id}, kind_hint="training").id
            except Exception:
                parent_run_display = parent_run_id
        parent_model_name = str(getattr(run, "parent_model_name", "") or "").strip()
        parent_model_path = str(getattr(run, "parent_model_path", "") or "").strip()
        if not parent_model_name and parent_model_path:
            parent_model_name = Path(parent_model_path).name
        parent_map = float(getattr(run, "parent_best_map50_95", 0.0) or 0.0)
        parent_suffix = f" | poprzedni mAP50-95 {parent_map:.3f}" if parent_map > 0 else ""
        lineage_rows.append((
            "Rodowód",
            f"Dotrenowanie od runu {parent_run_display} ({parent_model_name or 'best.pt'}){parent_suffix}",
        ))

    rows = [
        ("ID runu", getattr(run_ref, "id", "") or run_id or "-"),
        ("Nazwa techniczna", run_name or run_id or "-"),
        ("Status", self._format_history_run_status_label(run)),
        ("Dataset", dataset_display),
        ("Preset / plik startowy YOLO", base_model),
        *lineage_rows,
        ("Start treningu", started_at),
        ("Postęp", f"{int(getattr(run, 'current_epoch', 0) or 0)}/{int(getattr(run, 'epochs', 0) or 0)} epok"),
        (
            "Ustawienia",
            (
                f"rozdzielczość wejściowa {int(getattr(run, 'img_size', 0) or 0)} px | "
                f"rozmiar partii {int(getattr(run, 'batch_size', 0) or 0)} | "
                f"współczynnik uczenia {float(getattr(run, 'lr0', 0.0) or 0.0):.4f}"
            ),
        ),
        ("Urządzenie", self._shorten_training_text(str(getattr(run, "device", "") or "-"), 28)),
        ("Najlepsze wagi", best_weights_detail),
        ("Checkpoint last.pt", last_weights_name),
        ("Wznowienie", resume_hint),
        ("Utworzono", created_at),
    ]
    return rows

def _build_training_run_metric_rows(self, run) -> list[tuple[str, str, str, str]]:
    if run is None:
        return []

    history = list(getattr(run, "metrics_history", []) or [])
    if not history:
        latest_map = getattr(run, "best_map50", 0.0) or 0.0
        latest_map95 = getattr(run, "best_map50_95", 0.0) or 0.0
        fallback_metrics = {
            "map50": latest_map,
            "map50_95": latest_map95,
        }
        live_rows = self._build_training_metric_rows(fallback_metrics, self._infer_dataset_target(getattr(run, "dataset_path", "")))
        return [(label, value, value, band) for label, value, band, _range in live_rows]

    latest = history[-1]
    target = self._infer_dataset_target(getattr(run, "dataset_path", "")) or self.get_campaign_training_target()
    rows: list[tuple[str, str, str, str]] = []

    for key in self._iter_training_metric_keys(target):
        numeric_values: list[float] = []
        for item in history:
            try:
                if key in item:
                    numeric_values.append(float(item[key]))
                elif key == "pose_map50_95" and "map50_95" in item:
                    numeric_values.append(float(item["map50_95"]))
                elif key == "pose_map50" and "map50" in item:
                    numeric_values.append(float(item["map50"]))
                elif key == "box_map50_95" and "map50_95" in item:
                    numeric_values.append(float(item["map50_95"]))
                elif key == "box_map50" and "map50" in item:
                    numeric_values.append(float(item["map50"]))
            except Exception:
                continue

        latest_value = latest.get(key)
        if latest_value is None and key == "pose_map50_95":
            latest_value = latest.get("map50_95")
        elif latest_value is None and key == "pose_map50":
            latest_value = latest.get("map50")
        elif latest_value is None and key == "box_map50_95":
            latest_value = latest.get("map50_95")
        elif latest_value is None and key == "box_map50":
            latest_value = latest.get("map50")

        if latest_value is None and not numeric_values:
            continue

        if str(key).lower() == "loss":
            best_value = min(numeric_values) if numeric_values else latest_value
        else:
            best_value = max(numeric_values) if numeric_values else latest_value

        band, _range_text = self._format_training_metric_band(key, best_value)
        rows.append(
            (
                self._format_training_metric_name(key),
                self._format_training_metric_value(key, latest_value),
                self._format_training_metric_value(key, best_value),
                band,
            )
        )

    return rows

def _set_train_live_metrics(self, metrics: dict | None):
    tree = getattr(self, "train_live_metrics_tree", None)
    rows = self._build_training_metric_rows(metrics)
    if not rows:
        try:
            if tree is not None:
                tree.configure(height=3)
        except Exception:
            pass
        _set_training_status_tree_rows(
            self,
            tree,
            [
                (("mAP50-95", "-", "czekam", "po 1. epoce"), "training_info"),
                (("mAP50", "-", "czekam", "po 1. epoce"), "training_info"),
                (("Loss", "-", "monitoruj", "trend epok"), "training_info"),
            ],
        )
        return

    styled_rows = []
    for metric_name, value, band, range_text in rows:
        styled_rows.append(
            (
                (metric_name, value, band, range_text),
                _training_status_tag_from_text(self, band),
            )
        )
    try:
        if tree is not None:
            tree.configure(height=max(3, min(5, len(styled_rows))))
    except Exception:
        pass
    _set_training_status_tree_rows(self, tree, styled_rows)

def _build_training_cockpit_summary(self, *, ready: bool | None = None) -> dict:
    if getattr(self, "_training_start_in_progress", False):
        previous = dict(getattr(self, "_last_training_cockpit_summary", {}) or {})
        return {
            "status": "Przygotowanie treningu",
            "subtitle": "Sprawdzam dane i przygotowuję model. Możesz nadal korzystać z interfejsu.",
            "tone": "warning",
            "cards": list(previous.get("cards") or []),
        }
    target = self._get_selected_training_target()
    target_label = self._format_training_target_label(target)
    dataset_yaml = self._resolve_training_dataset_yaml_path()

    cards: list[tuple[str, str, str]] = []
    dataset_ready = dataset_yaml is not None
    if dataset_yaml is None:
        cards.extend(
            [
                ("Dataset", "Brak wariantu", "Najpierw wybierz albo utwórz split w PZ1."),
                ("Próbka", "-", "Po wyborze datasetu pokażemy train / val / test."),
            ]
        )
    else:
        dataset_root = dataset_yaml.parent
        counts = {"train": 0, "val": 0, "test": 0, "total": 0}
        source = getattr(self, "_last_training_source", None)
        source_stats = getattr(source, "stats", None)
        try:
            source_root = Path(str(getattr(source, "dataset_dir", "") or getattr(source, "yaml_path", "") or ""))
            if source_root.is_file():
                source_root = source_root.parent
            same_source = source_root.exists() and source_root.resolve() == dataset_root.resolve()
        except Exception:
            same_source = False
        if same_source and hasattr(source_stats, "split_counts"):
            try:
                counts = dict(source_stats.split_counts())
            except Exception:
                counts = {"train": 0, "val": 0, "test": 0, "total": 0}
        if int(counts.get("total", 0) or 0) <= 0:
            try:
                counts = self._get_dataset_split_image_counts(dataset_root)
            except Exception:
                counts = {"train": 0, "val": 0, "test": 0, "total": 0}

        try:
            cfg = safe_load_yaml(dataset_yaml) or {}
        except Exception:
            cfg = {}
        try:
            class_count = int(cfg.get("nc", 0) or 0)
        except Exception:
            class_count = 0
        if class_count <= 0:
            try:
                class_count = len(self._extract_dataset_class_names(cfg))
            except Exception:
                class_count = 0
        dataset_task = "Pose" if bool(cfg.get("kpt_shape")) else "Detect"

        total = int(counts.get("total", 0) or 0)
        train = int(counts.get("train", 0) or 0)
        val = int(counts.get("val", 0) or 0)
        test = int(counts.get("test", 0) or 0)
        dataset_ref = build_dataset_display_ref(dataset_root, target_hint=target, counts=counts)
        dataset_hint = f"YOLO {dataset_task}"
        if dataset_ref.created_label:
            dataset_hint = f"{dataset_hint} | utworzono {dataset_ref.created_label}"
        cards.extend(
            [
                ("Dataset", dataset_ref.compact_label, dataset_hint),
                ("Próbka", f"{total} obrazów", f"train {train} | val {val} | test {test} | klasy {class_count}"),
            ]
        )

    base_display = self._resolve_selected_training_base_model_display()
    try:
        model_state = _resolve_selected_training_base_model_training_state(self)
    except Exception:
        model_state = {}
    model_hint = str(model_state.get("detail") or "Model startowy tego treningu.").strip()
    try:
        base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
        custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        model_value = _format_training_base_model_summary_value(
            self,
            base_display,
            base_key=base_key,
            base_model=custom_model or base_key,
            context=_training_base_model_context(self),
        )
    except Exception:
        model_value = base_display
    if not str(model_value or "").strip():
        model_value = "Nie wybrano modelu"
    cards.append(("Model", self._shorten_training_text(model_value, 54), model_hint))

    try:
        recommendation = self._get_training_device_recommendation(self._get_global_training_device_choice())
    except Exception:
        recommendation = {}
    try:
        batch = self._safe_training_int_value("batch_var", default=int(recommendation.get("batch", 16) or 16), minimum=1)
    except Exception:
        batch = int(recommendation.get("batch", 16) or 16)
    try:
        imgsz = self._safe_training_int_value("imgsz_var", default=int(recommendation.get("imgsz", 640) or 640), minimum=32)
    except Exception:
        imgsz = int(recommendation.get("imgsz", 640) or 640)
    try:
        epochs = self._safe_training_int_value("epochs_var", default=100, minimum=1)
    except Exception:
        epochs = 100
    try:
        memory_gb = float(recommendation.get("memory_gb", 0.0) or 0.0)
    except Exception:
        memory_gb = 0.0
    if str(recommendation.get("effective_raw") or "").lower() == "cpu":
        device_value = "CPU"
    else:
        device_value = str(recommendation.get("device_name") or "GPU").strip() or "GPU"
        if memory_gb > 0:
            device_value = f"{device_value} | {memory_gb:.1f} GB VRAM"
    cards.append(("Sprzęt", self._shorten_training_text(device_value, 54), f"batch {batch} | imgsz {imgsz} | epoki {epochs}"))

    if ready is None:
        try:
            ready = bool(self._is_training_configuration_ready())
        except Exception:
            ready = False

    if ready:
        status = "Gotowe do startu"
        subtitle = "Masz wariant datasetu, zgodny model i dobrane parametry treningu."
        tone = "success"
    elif dataset_ready:
        status = "Uzupełnij model"
        subtitle = "Dataset jest wybrany. Sprawdź model startowy i zgodność typu treningu."
        tone = "warning"
    else:
        status = "Czekam na wariant"
        subtitle = "Utwórz lub wybierz wariant splitu w PZ1, a potem wróć do treningu."
        tone = "muted"

    return {
        "status": status,
        "subtitle": subtitle,
        "tone": tone,
        "cards": cards,
    }

def _refresh_training_cockpit(self, *, ready: bool | None = None):
    shell = getattr(self, "train_cockpit_shell", None)
    cards = list(getattr(self, "_train_cockpit_cards", []) or [])
    if shell is None or not cards:
        return

    palette = getattr(self.app, "palette", {})
    summary = self._build_training_cockpit_summary(ready=ready)
    self._last_training_cockpit_summary = dict(summary)
    tone = str(summary.get("tone") or "muted").strip().lower()
    tone_colors = {
        "success": (
            palette.get("success", "#2ecc71"),
            palette.get("surface_success", palette.get("panel_alt", "#2d2d30")),
            palette.get("success", "#2ecc71"),
        ),
        "warning": (
            palette.get("warning", "#f0b44c"),
            palette.get("surface_warning", palette.get("panel_alt", "#2d2d30")),
            palette.get("warning", "#f0b44c"),
        ),
        "muted": (
            palette.get("muted", "#c7c7c7"),
            palette.get("panel_alt", "#2d2d30"),
            palette.get("muted", "#c7c7c7"),
        ),
    }
    accent, status_bg, status_fg = tone_colors.get(tone, tone_colors["muted"])
    shell_bg = blend_hex_colors(accent, palette.get("panel", "#252526"), 0.88)
    shell_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.72,
    )
    card_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
    muted_fg = palette.get("muted", "#c7c7c7")
    fg = palette.get("fg", "#f3f3f3")

    for widget_name in ("train_cockpit_shell", "train_cockpit_header", "train_cockpit_grid"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(bg=shell_bg)
        except Exception:
            pass
    try:
        shell.configure(highlightbackground=shell_border, highlightcolor=shell_border)
    except Exception:
        pass
    for widget_name in ("train_cockpit_title_lbl", "train_cockpit_subtitle_lbl"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(bg=shell_bg)
        except Exception:
            pass
    try:
        self.train_cockpit_title_lbl.configure(fg=fg)
    except Exception:
        pass
    self._set_training_widget_text(getattr(self, "train_cockpit_status_lbl", None), summary.get("status", ""))
    self._set_training_widget_text(getattr(self, "train_cockpit_subtitle_lbl", None), summary.get("subtitle", ""))
    try:
        self.train_cockpit_status_lbl.configure(bg=status_bg, fg=status_fg)
    except Exception:
        pass
    try:
        self.train_cockpit_subtitle_lbl.configure(fg=muted_fg)
    except Exception:
        pass

    summary_cards = list(summary.get("cards") or [])
    for index, card_widgets in enumerate(cards):
        title, value, hint = summary_cards[index] if index < len(summary_cards) else ("", "", "")
        frame = card_widgets.get("frame")
        title_lbl = card_widgets.get("title")
        value_lbl = card_widgets.get("value")
        hint_lbl = card_widgets.get("hint")
        for widget in (frame, title_lbl, value_lbl, hint_lbl):
            if widget is None:
                continue
            try:
                widget.configure(bg=card_bg)
            except Exception:
                pass
        if frame is not None:
            try:
                frame.configure(
                    highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                    highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                )
            except Exception:
                pass
        self._set_training_widget_text(title_lbl, str(title or ""))
        self._set_training_widget_text(value_lbl, str(value or "-"))
        self._set_training_widget_text(hint_lbl, str(hint or ""))
        try:
            title_lbl.configure(fg=muted_fg)
            value_lbl.configure(fg=fg)
            hint_lbl.configure(fg=muted_fg)
        except Exception:
            pass

def _build_training_recommendation_rows(self) -> tuple[str, str, list[tuple[str, ...]], str]:
    recommendation = self._get_training_device_recommendation(self._get_global_training_device_choice())
    if recommendation.get("ready") is False:
        return (
            str(recommendation.get("status") or "Sprzęt: brak pełnego wyniku sprawdzenia"), "",
            [("Epoki", "ręcznie"), ("Batch", "-"), ("Rozdzielczość", "-"), ("LR", "-")],
            str(recommendation.get("note") or ""),
        )
    memory_gb = float(recommendation.get("memory_gb", 0.0) or 0.0)
    if recommendation.get("effective_raw") == "cpu":
        device_label = "Wykryty sprzęt: CPU"
    else:
        device_label = (
            f"Wykryty sprzęt: {recommendation.get('device_name', 'GPU')} | "
            f"{memory_gb:.1f} GB VRAM"
        )
    model_label = str(recommendation.get("model_detected_label") or "").strip()
    if not model_label:
        model_label = self._build_training_recommendation_model_text()

    rows = [
        ("Epoki", "ręcznie"),
        ("Batch", str(int(recommendation.get("batch", 0) or 0))),
        ("Rozdzielczość", str(int(recommendation.get("imgsz", 0) or 0))),
        ("LR", f"{float(recommendation.get('lr0', 0.0) or 0.0):.4f}"),
    ]
    note = str(
        recommendation.get("note")
        or "Jeśli zabraknie pamięci, najpierw zmniejsz rozmiar partii, a potem rozdzielczość wejściową."
    )
    return device_label, model_label, rows, note

def _refresh_training_recommendation_table(self):
    if not getattr(self, "_train_recommendation_cells", None):
        return
    hardware_text, model_text, rows, note_text = self._build_training_recommendation_rows()
    hardware_label = getattr(self, "train_recommendation_hardware_label", None)
    self._set_training_widget_text(hardware_label, hardware_text)
    model_label = getattr(self, "train_recommendation_model_label", None)
    self._set_training_widget_text(model_label, model_text)

    cells = list(getattr(self, "_train_recommendation_cells", []) or [])
    for row_index, row_values in enumerate(rows):
        if len(row_values) >= 3:
            label_text, current_text, recommended_text = row_values[:3]
        else:
            label_text, recommended_text = row_values[:2]
            current_text = None
        if row_index >= len(cells):
            continue
        row = cells[row_index]
        self._set_training_widget_text(row.get("key"), str(label_text))
        if current_text is not None:
            self._set_training_widget_text(row.get("current"), str(current_text))
        if row.get("card") is not None:
            self._set_training_widget_text(row.get("recommended"), f"Rekomendacja: {recommended_text}")
        else:
            self._set_training_widget_text(row.get("recommended"), str(recommended_text))

    note_var = getattr(self, "train_recommendation_note_var", None)
    self._set_training_stringvar_text(note_var, note_text)

def _apply_training_recommendation_table_theme(self):
    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    section_border = blend_hex_colors(palette.get("success", "#2ecc71"), border, 0.72)
    card_border = blend_hex_colors(border, panel, 0.30)
    container_bg = panel
    card_bg = blend_hex_colors(panel_alt, panel, 0.72)
    header_bg = blend_hex_colors(panel_alt, border, 0.70)
    header_fg = palette.get("fg", "#f3f3f3")
    cell_bg = panel
    alt_cell_bg = blend_hex_colors(panel_alt, panel, 0.62)
    cell_fg = palette.get("fg", "#f3f3f3")
    title_bg = container_bg
    title_fg = palette.get("fg", "#f3f3f3")
    hardware_bg = container_bg
    hardware_fg = palette.get("muted", "#c7c7c7")
    subtle_bg = blend_hex_colors(panel_alt, panel, 0.64)
    subtle_fg = palette.get("muted", "#c7c7c7")

    shell = getattr(self, "train_recommendation_table_shell", None)
    grid = getattr(self, "train_recommendation_grid", None)
    cards_grid = getattr(self, "train_recommendation_cards_grid", None)
    title_row = getattr(self, "train_recommendation_title_row", None)
    if shell is not None:
        try:
            shell.configure(bg=container_bg, highlightbackground=section_border, highlightcolor=section_border)
        except Exception:
            pass
    if grid is not None:
        try:
            grid.configure(bg=container_bg)
        except Exception:
            pass
    if cards_grid is not None:
        try:
            cards_grid.configure(bg=container_bg)
        except Exception:
            pass
    if title_row is not None:
        try:
            title_row.configure(bg=title_bg)
        except Exception:
            pass

    for label_name, bg, fg in (
        ("train_recommendation_title_label", title_bg, title_fg),
        ("train_recommendation_hardware_label", hardware_bg, hardware_fg),
        ("train_recommendation_model_label", hardware_bg, hardware_fg),
    ):
        label = getattr(self, label_name, None)
        if label is None:
            continue
        try:
            label.configure(bg=bg, fg=fg)
        except Exception:
            pass

    for label in list(getattr(self, "_train_recommendation_header_labels", []) or []):
        if label is None:
            continue
        try:
            label.configure(bg=header_bg, fg=header_fg)
        except Exception:
            pass

    for row_index, row in enumerate(list(getattr(self, "_train_recommendation_cells", []) or [])):
        card_widget = row.get("card")
        key_widget = row.get("key")
        current_widget = row.get("current")
        editor_host = row.get("editor_host")
        recommended_widget = row.get("recommended")
        if card_widget is not None:
            key_bg = card_bg
            editor_bg = card_bg
            try:
                card_widget.configure(
                    bg=card_bg,
                    highlightbackground=card_border,
                    highlightcolor=card_border,
                )
            except Exception:
                pass
        else:
            key_bg = alt_cell_bg if row_index % 2 == 1 else cell_bg
            editor_bg = alt_cell_bg if row_index % 2 == 1 else cell_bg
        if key_widget is not None:
            try:
                key_widget.configure(bg=key_bg, fg=cell_fg)
            except Exception:
                pass
        if editor_host is not None:
            try:
                editor_host.configure(bg=editor_bg, highlightbackground=editor_bg, highlightcolor=editor_bg)
            except Exception:
                pass
        if current_widget is not None:
            try:
                current_widget.configure(bg=editor_bg, fg=cell_fg)
            except Exception:
                pass
        if recommended_widget is not None:
            try:
                recommended_widget.configure(
                    bg=card_bg if card_widget is not None else subtle_bg,
                    fg=subtle_fg,
                )
            except Exception:
                pass

def _apply_training_device_recommendation(self):
    self._apply_training_recommended_start_params()

def _on_training_base_model_value_write(self, *_args):
    try:
        self._sync_step4_fine_tune_parent_selection()
    except Exception:
        pass
    try:
        self._refresh_training_base_model_selection_ui()
    except Exception:
        pass
    try:
        self._refresh_training_execution_summary()
    except Exception:
        pass
    try:
        self._refresh_training_start_state()
    except Exception:
        pass

def _schedule_step4_deferred_model_refresh(self):
    frame = getattr(self, "frame", None)
    if frame is None:
        return

    pending_job = getattr(self, "_step4_deferred_model_refresh_job", None)
    if pending_job is not None:
        try:
            frame.after_cancel(pending_job)
        except Exception:
            pass

    def run_refresh():
        self._step4_deferred_model_refresh_job = None
        if not bool(getattr(self, "_step4_train_tab_built", False)):
            return
        try:
            self._refresh_base_model_choices()
        except Exception:
            pass
        try:
            self._refresh_training_recommendation_table()
        except Exception:
            pass
        try:
            self._refresh_training_execution_summary()
        except Exception:
            pass

    try:
        self._step4_deferred_model_refresh_job = frame.after(25, run_refresh)
    except Exception:
        self._step4_deferred_model_refresh_job = None

def _apply_training_recommended_start_params(self):
    recommendation = self._get_training_device_recommendation(self._get_global_training_device_choice())
    if recommendation.get("ready") is False:
        self._refresh_training_recommendation_table()
        return
    try:
        current_batch = self._safe_training_int_value("batch_var", default=16, minimum=1)
        self.batch_var.set(int(recommendation.get("batch", current_batch) or current_batch))
    except Exception:
        pass
    try:
        current_imgsz = self._safe_training_int_value("imgsz_var", default=640, minimum=32)
        self.imgsz_var.set(int(recommendation.get("imgsz", current_imgsz) or current_imgsz))
    except Exception:
        pass
    try:
        current_lr0 = self._safe_training_float_value("lr0_var", default=0.01, minimum=0.0001)
        self.lr0_var.set(float(recommendation.get("lr0", current_lr0) or current_lr0))
    except Exception:
        pass
    self._refresh_training_recommendation_table()
    self._refresh_training_start_state()

def _safe_training_numeric_var_value(
    self,
    attr_name: str,
    *,
    default,
    cast,
    minimum=None,
):
    var_obj = getattr(self, attr_name, None)
    raw_value = None
    if var_obj is not None:
        try:
            raw_value = var_obj.get()
        except Exception:
            raw_value = None

    try:
        value = cast(raw_value)
    except Exception:
        value = cast(default)

    if minimum is not None:
        try:
            value = max(cast(minimum), value)
        except Exception:
            pass
    return value

def _safe_training_int_value(self, attr_name: str, *, default: int, minimum: int | None = None) -> int:
    return int(
        self._safe_training_numeric_var_value(
            attr_name,
            default=int(default),
            cast=lambda value: int(float(value)),
            minimum=minimum,
        )
    )

def _safe_training_float_value(self, attr_name: str, *, default: float, minimum: float | None = None) -> float:
    return float(
        self._safe_training_numeric_var_value(
            attr_name,
            default=float(default),
            cast=float,
            minimum=minimum,
        )
    )

def _resolve_training_dataset_yaml_path(self) -> Path | None:
    dataset_value = str(getattr(self, "dataset_var", tk.StringVar()).get() or "").strip()

    def _normalize_dataset_root(value: str | Path | None) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            path = Path(raw)
            if path.is_file() and path.name.lower() == "data.yaml":
                path = path.parent
            return str(path.resolve())
        except Exception:
            return raw

    def _source_matches_current_selection(source: TrainingSource) -> bool:
        selected_root = _normalize_dataset_root(dataset_value)
        if not selected_root:
            return True
        for raw in (source.dataset_dir, source.yaml_path):
            source_root = _normalize_dataset_root(raw)
            if source_root and source_root == selected_root:
                return True
        return False

    try:
        source = getattr(self, "_last_training_source", None)
        if (
            isinstance(source, TrainingSource)
            and source.has_dataset()
            and not _source_matches_current_selection(source)
        ):
            source = None
        if not isinstance(source, TrainingSource) or not source.has_dataset():
            resolver = getattr(self, "_resolve_step4_dataset_summary_source", None)
            if callable(resolver):
                source = resolver()
        if (
            isinstance(source, TrainingSource)
            and source.has_dataset()
            and not _source_matches_current_selection(source)
        ):
            source = None
        if isinstance(source, TrainingSource) and source.has_dataset():
            candidates: list[Path] = []
            if str(source.yaml_path or "").strip():
                candidates.append(Path(source.yaml_path))
            if str(source.dataset_dir or "").strip():
                dataset_dir = Path(source.dataset_dir)
                candidates.append(dataset_dir / "data.yaml" if dataset_dir.is_dir() else dataset_dir)
            for candidate in candidates:
                if candidate.exists() and candidate.is_file() and candidate.name.lower() == "data.yaml":
                    return candidate
    except Exception:
        pass

    if not dataset_value:
        return None

    candidate = Path(dataset_value)
    if candidate.is_dir():
        yaml_path = candidate / "data.yaml"
        return yaml_path if yaml_path.exists() else None
    if candidate.is_file() and candidate.name.lower() == "data.yaml":
        return candidate
    return None

def _validate_training_source_lightweight(self) -> dict:
    """Check the selected variant without scanning images or resolving UI summaries."""
    target = self._get_selected_training_target()
    result = dict(ok=False, yaml_path=None, dataset_root=None, target=target,
                  message="Najpierw wskaż dataset treningowy z plikiem data.yaml.", stats={})
    value = str(self.dataset_var.get() or "").strip()
    if not value:
        return result
    path = Path(value)
    yaml_path = path / "data.yaml" if path.is_dir() else path
    if yaml_path.name.lower() != "data.yaml" or not yaml_path.is_file():
        return result
    dataset_root = yaml_path.parent
    result.update(yaml_path=yaml_path, dataset_root=dataset_root)
    if not CAMPAIGN.get_active_project_name():
        selected_root = dataset_root.resolve()
        choices = getattr(self, "_dataset_variant_choices", ()) or ()
        known_roots = []
        for item in choices:
            raw = str(item.get("path") or "").strip()
            if raw:
                candidate = Path(raw)
                known_roots.append((candidate.parent if candidate.name.lower() == "data.yaml" else candidate).resolve())
        if selected_root not in known_roots:
            result["message"] = "Wybierz wariant splitu z listy PZ2 albo utwórz go w PZ1."
            return result
    try:
        config = safe_load_yaml(yaml_path)
        if not isinstance(config, dict) or not config.get("train") or not config.get("val"):
            result["message"] = "Plik data.yaml musi wskazywać zbiory train i val."
            return result
        inferred_target = self._infer_dataset_target(str(dataset_root))
    except Exception as exc:
        result["message"] = f"Nie udało się odczytać data.yaml: {exc}"
        return result
    if inferred_target and inferred_target != target:
        result["message"] = (
            "Wybrany dataset nie pasuje do aktywnego toru treningu.\n"
            f"Tor: {self._format_training_target_label(target)}\n"
            f"Dataset: {self._format_training_target_label(inferred_target)}"
        )
        return result
    readiness = get_training_dataset_readiness(dataset_root, target=target)
    if not readiness.get("ok", True):
        result["message"] = str(readiness.get("message") or "Wariant nie jest gotowy do treningu.")
        return result
    result.update(ok=True, message="Wybrano wariant. Pełna kontrola danych odbędzie się podczas przygotowania treningu.")
    return result


def _apply_preflight_dataset_validation(self, result: dict | None) -> None:
    if not result or not result.get("ok"):
        return
    dataset_root = Path(result["dataset_root"])
    selected = Path(str(self.dataset_var.get() or ""))
    if selected.name.lower() == "data.yaml":
        selected = selected.parent
    if selected.resolve() != dataset_root.resolve():
        return
    source = TrainingSource(
        target=result.get("target") or self._get_selected_training_target(),
        kind="yolo_dataset", dataset_dir=str(dataset_root), yaml_path=str(dataset_root / "data.yaml"),
        validated=True, stats=TrainingSourceStats.from_mapping(result.get("stats") or {}),
        provenance="Aktywny split", source_stage="Z4/PZ2", message=result.get("message", ""),
    )
    self._last_training_source = source
    def mtime(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
    self._active_training_source_validation_cache = {
        "key": (str(dataset_root.resolve()), source.target, mtime(dataset_root / "data.yaml"),
                *(mtime(dataset_root / "images" / split) for split in ("train", "val", "test"))),
        "result": dict(result, source=source, dataset_root=dataset_root, yaml_path=dataset_root / "data.yaml"),
    }


def _validate_active_training_source_for_pz2(self) -> dict:
    result = {
        "ok": False,
        "source": None,
        "yaml_path": None,
        "dataset_root": None,
        "target": self._get_selected_training_target(),
        "message": "Najpierw wskaż dataset treningowy.",
        "stats": {},
    }

    yaml_path = self._resolve_training_dataset_yaml_path()
    if yaml_path is None:
        result["message"] = "Najpierw wskaż poprawny dataset treningowy z plikiem data.yaml."
        return result

    dataset_root = yaml_path.parent
    ok_yaml_root, yaml_root_msg, yaml_root_changed = ensure_yolo_dataset_yaml_points_to_root(dataset_root)
    if not ok_yaml_root:
        result["message"] = str(yaml_root_msg or "Nie udało się zweryfikować pliku data.yaml.")
        return result
    if yaml_root_changed:
        try:
            logger.info(f"Poprawiono path w data.yaml przed treningiem: {dataset_root}")
        except Exception:
            pass
    selected_target = self._get_selected_training_target()
    if not CAMPAIGN.get_active_project_name():
        if not self._is_free_training_dataset_variant_selected(dataset_root):
            result["message"] = (
                "Wybierz aktywny wariant splitu z listy PZ2 albo utwórz go w PZ1. "
                "Trening nie powinien startować na przypadkowej ścieżce."
            )
            return result

    inferred_target = self._infer_dataset_target(str(dataset_root))
    if inferred_target and inferred_target != selected_target:
        target_context = "tor" if CAMPAIGN.get_active_project_name() else "typ datasetu"
        result["message"] = (
            f"Wybrany dataset należy do innego {target_context} niż aktywny trening.\n\n"
            f"Aktywny {target_context}: {self._format_training_target_label(selected_target)}\n"
            f"Dataset: {self._format_training_target_label(inferred_target)}"
        )
        return result

    readiness = get_training_dataset_readiness(dataset_root, target=(inferred_target or selected_target))
    if not bool(readiness.get("ok", True)):
        result["message"] = str(readiness.get("message") or "Wybrany wariant datasetu nie jest gotowy do treningu.")
        result["dataset_root"] = dataset_root
        result["yaml_path"] = yaml_path
        return result

    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    try:
        root_key = str(dataset_root.resolve())
    except Exception:
        root_key = str(dataset_root)
    cache_key = (
        root_key,
        str(selected_target),
        _safe_mtime(yaml_path),
        _safe_mtime(dataset_root / "images" / "train"),
        _safe_mtime(dataset_root / "images" / "val"),
        _safe_mtime(dataset_root / "images" / "test"),
    )
    cache = getattr(self, "_active_training_source_validation_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        cached_result = dict(cache.get("result") or {})
        cached_source = cached_result.get("source")
        if isinstance(cached_source, TrainingSource):
            try:
                self._last_training_source = cached_source
            except Exception:
                pass
        return cached_result

    try:
        is_valid, validation_msg, validation_stats = self.trainer.validate_dataset(dataset_root)
    except Exception as exc:
        is_valid = False
        validation_msg = f"Błąd walidacji datasetu: {exc}"
        validation_stats = {}

    validation_stats = dict(validation_stats or {})
    source = TrainingSource(
        target=CONFIG.normalize_task_target(inferred_target or selected_target),
        kind="yolo_dataset",
        dataset_dir=str(dataset_root),
        yaml_path=str(yaml_path),
        validated=bool(is_valid),
        stats=TrainingSourceStats.from_mapping(validation_stats),
        provenance=str(getattr(self._resolve_step4_dataset_summary_source(), "provenance", "") or "Aktywny split"),
        source_stage="Z4/PZ2",
        message=str(validation_msg or "").strip(),
    )
    try:
        self._last_training_source = source
    except Exception:
        pass

    result.update(
        {
            "ok": bool(is_valid),
            "source": source,
            "yaml_path": yaml_path,
            "dataset_root": dataset_root,
            "target": source.target,
            "message": str(validation_msg or "").strip(),
            "stats": validation_stats,
        }
    )
    try:
        self._active_training_source_validation_cache = {
            "key": cache_key,
            "result": dict(result),
        }
    except Exception:
        pass
    return result

def _looks_like_char_classification_dataset(self, path_like) -> bool:
    try:
        path = Path(path_like)
    except Exception:
        return False

    if path.is_file():
        if path.name.lower() == "manifest.json":
            manifest_path = path
            root = path.parent
        else:
            root = path.parent
            manifest_path = root / "manifest.json"
    else:
        root = path
        manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        return False

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_type = str(data.get("dataset_type") or "").strip().lower()
        if dataset_type == "char_classification":
            return True
    except Exception:
        pass

    try:
        has_class_splits = all((root / split).exists() for split in ("train", "val", "test"))
        has_yolo_shape = (root / "images").exists() or (root / "labels").exists() or (root / "data.yaml").exists()
        return bool(has_class_splits and not has_yolo_shape)
    except Exception:
        return False

def _char_classification_dataset_message(self) -> str:
    return (
        "Wybrane źródło wygląda na dataset OCR/klasyfikacyjny znaków (manifest.json). "
        "Z4/PZ1/PZ2 trenuje tutaj modele YOLO, więc wymaga datasetu YOLO Detect: "
        "folderu z images/labels oraz data.yaml. Wybierz np. katalog YOLO_MegaDataset_Chars_* "
        "albo najpierw wygeneruj/wyeksportuj dataset YOLO dla znaków. "
        "Datasety klasyfikacyjne są odkładane osobno w 4_training_datasets/4_char_classification."
    )

def _get_pinned_step4_result_state(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}

    try:
        context = _training_base_model_context(self)
    except Exception:
        context = {}
    target = str(context.get("target") or "").strip().lower()
    if target not in {"char", "plate"}:
        try:
            target = CONFIG.normalize_task_target(self.get_campaign_training_target())
        except Exception:
            target = "char"

    try:
        current_iteration = int(context.get("iteration") or CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0

    try:
        stored = dict(CAMPAIGN.get_step4_finish_state() or {})
    except Exception:
        stored = {}
    if not bool(stored.get("ready")) or not bool(stored.get("selection_confirmed")):
        return {}

    stored_target = str(stored.get("target", "") or "").strip().lower()
    if stored_target and target and stored_target != target:
        return {}

    try:
        stored_iteration = int(stored.get("iteration", 0) or 0)
    except Exception:
        stored_iteration = 0
    if stored_iteration > 0 and current_iteration > 0 and stored_iteration != current_iteration:
        return {}

    try:
        validated = dict(self.get_campaign_step4_finish_state(iteration_target=target) or {})
    except Exception:
        validated = {}
    if not bool(validated.get("ready")):
        return {}

    stored_run_id = str(stored.get("run_id", "") or "").strip()
    validated_run_id = str(validated.get("run_id", "") or "").strip()
    if stored_run_id and validated_run_id and stored_run_id != validated_run_id:
        return {}

    state = dict(stored)
    state.update({k: v for k, v in validated.items() if v not in ("", None)})
    state["target"] = target
    state["iteration"] = current_iteration or stored_iteration
    state["target_label"] = context.get("target_label") or ("model tablic" if target == "plate" else "model znaków")
    state["target_detail"] = context.get("target_detail") or state["target_label"]
    state["iteration_label"] = context.get("iteration_label") or (f"IT{state['iteration']}" if state.get("iteration") else "IT")
    state["model_path"] = str(stored.get("model_path", "") or state.get("model_path", "") or "").strip()
    state["selection_confirmed"] = True
    return state

def _format_pinned_step4_result_detail(self, state: dict) -> str:
    run_id = str(state.get("run_id", "") or "").strip()
    model_path = str(state.get("model_path", "") or "").strip()
    model_name = Path(model_path).name if model_path else "wybrany model"
    iteration_label = str(state.get("iteration_label", "") or "").strip()
    target_label = str(state.get("target_label", "") or "model").strip()
    if run_id:
        try:
            run_display = build_run_display_ref({"run_id": run_id}, kind_hint="training").id
        except Exception:
            run_display = run_id
    else:
        run_display = ""
    run_part = f" Run: {run_display}." if run_display else ""
    return (
        f"★ WYNIK T06 PODPIĘTY | {iteration_label} • {target_label}: {model_name}.{run_part} "
        "Ten model jest wynikiem bramki, więc split, model startowy, parametry i nowy trening są zablokowane. "
        "Aby zmienić decyzję, odepnij wynik w sekcji wyboru wyniku bramki."
    )

def _set_step4_training_config_widgets_locked(self, locked: bool):
    state_readonly = tk.DISABLED if locked else "readonly"
    state_normal = tk.DISABLED if locked else tk.NORMAL

    for attr in ("dataset_variant_combo", "base_combo"):
        widget = getattr(self, attr, None)
        if widget is None:
            continue
        try:
            widget.configure(state=state_readonly)
        except Exception:
            pass

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    try:
        is_custom = bool(self._is_custom_base_model_key(base_key))
    except Exception:
        is_custom = False

    custom_entry = getattr(self, "base_custom_entry", None)
    if custom_entry is not None:
        try:
            custom_entry.configure(state=(tk.DISABLED if locked or not is_custom else "readonly"))
        except Exception:
            pass

    custom_btn = getattr(self, "base_custom_btn", None)
    if custom_btn is not None:
        try:
            custom_btn.configure(state=(tk.DISABLED if locked or not is_custom else tk.NORMAL))
        except Exception:
            pass

    apply_btn = getattr(self, "btn_apply_training_recommendation", None)
    if apply_btn is not None:
        try:
            apply_btn.configure(state=state_normal)
        except Exception:
            pass

    for cell in list(getattr(self, "_train_recommendation_cells", []) or []):
        editor = cell.get("editor") if isinstance(cell, dict) else None
        if editor is None:
            continue
        try:
            editor.configure(state=(tk.DISABLED if locked else tk.NORMAL))
        except Exception:
            pass

def _refresh_step4_pinned_result_ui(self, pinned_state: dict | None = None) -> dict:
    state = dict(pinned_state or _get_pinned_step4_result_state(self) or {})
    locked = bool(state)
    shell = getattr(self, "step4_pinned_result_shell", None)
    detail_label = getattr(self, "step4_pinned_result_detail_lbl", None)
    title_label = getattr(self, "step4_pinned_result_title_lbl", None)

    try:
        _set_step4_training_config_widgets_locked(self, locked)
    except Exception:
        pass

    if shell is not None:
        try:
            if locked:
                if title_label is not None:
                    title_label.configure(text="★ Wynik bramki T06 jest podpięty")
                if detail_label is not None:
                    detail_label.configure(text=_format_pinned_step4_result_detail(self, state))
                if not str(shell.winfo_manager()):
                    shell.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 10))
            else:
                shell.pack_forget()
        except Exception:
            pass
    return state

def _clear_pinned_step4_result(self):
    state = _get_pinned_step4_result_state(self)
    if not state:
        return
    if not messagebox.askyesno(
        "Odepnij wynik bramki",
        (
            "Odpiąć wybrany model jako wynik tej bramki?\n\n"
            "Model i historia runu zostaną w projekcie, ale wróci możliwość zmiany splitu, "
            "modelu startowego i uruchomienia nowego treningu."
        ),
    ):
        return
    try:
        CAMPAIGN.set_step4_finish_state(False)
    except Exception:
        pass
    try:
        self._step4_campaign_finish_ready = False
    except Exception:
        pass
    try:
        self._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass
    try:
        self._refresh_campaign_training_result_selector()
    except Exception:
        pass
    try:
        self._load_history()
    except Exception:
        pass
    try:
        self._refresh_training_base_model_selection_ui()
    except Exception:
        pass
    try:
        self._refresh_training_start_state()
    except Exception:
        pass
    try:
        campaign_tab = self.app.tabs.get("campaign")
        if campaign_tab:
            campaign_tab._refresh_dashboard()
    except Exception:
        pass
    try:
        self._append_train_log("[MODEL] Odpięto model jako wynik bramki. Można ponownie trenować lub wybrać inny wynik.")
    except Exception:
        pass

def _is_training_configuration_ready(self) -> bool:
    if not YOLO_AVAILABLE:
        return False
    if _get_pinned_step4_result_state(self):
        return False
    if self._step4_has_active_operation():
        return False
    if bool(getattr(self.trainer, "is_training", False)):
        return False

    source_state = _validate_training_source_lightweight(self)
    if not bool(source_state.get("ok")):
        return False
    selected_target = CONFIG.normalize_task_target(str(source_state.get("target") or self._get_selected_training_target()))

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return False
    if self._is_custom_base_model_key(base_key):
        custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        if not custom_model or not Path(custom_model).exists():
            return False
        if _selected_training_base_model_nonfinal_run(self) is not None:
            return False
        valid_selection, _selection_message = self._validate_training_base_model_target_compatibility(
            target=selected_target,
            show_dialog=False,
            lightweight=True,
        )
        if not valid_selection:
            return False

    return True

def _validate_training_base_model_target_compatibility(
    self,
    *,
    target: str | None = None,
    show_dialog: bool = False,
    lightweight: bool = False,
) -> tuple[bool, str]:
    normalized_target = CONFIG.normalize_task_target(target or self._get_selected_training_target())
    if normalized_target not in {"char", "plate"}:
        return True, ""

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return False, "Nie wybrano modelu startowego treningu."

    base_model = (
        str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        if self._is_custom_base_model_key(base_key)
        else base_key
    )
    if not base_model:
        return False, "Nie wybrano modelu startowego treningu."

    if self._is_custom_base_model_key(base_key):
        try:
            model_path = Path(base_model)
        except Exception:
            model_path = None
        if model_path is None or model_path.suffix.lower() != ".pt" or not model_path.is_file():
            return False, "Wskaż poprawny plik modelu .pt."

        ok, message, _info = validate_model_file(model_path, allow_heavy_load=not lightweight)
        if not ok and not lightweight:
            if show_dialog:
                messagebox.showerror(
                    "Nieprawidłowy model",
                    f"Nie udało się użyć wybranego modelu:\n{message}",
                )
            return False, message

        nonfinal_run = _selected_training_base_model_nonfinal_run(self)
        if nonfinal_run is not None:
            run_id = str(getattr(nonfinal_run, "id", "") or "").strip()
            try:
                run_display = build_run_display_ref(nonfinal_run, kind_hint="training").id
            except Exception:
                run_display = run_id
            status_value = str(getattr(nonfinal_run, "status", "") or "").strip().lower()
            if status_value in {TrainingStatus.FAILED.value, TrainingStatus.PAUSED.value}:
                message = (
                    "Wybrany plik pochodzi z niedokończonego treningu.\n\n"
                    f"Run: {run_display or '-'} | status: {status_value or '-'}.\n"
                    "Jeśli chcesz kontynuować ten trening, użyj akcji `Wznów trening` w historii. "
                    "`Rozpocznij trening` tworzy nowy run i nie powinien po cichu wznawiać checkpointu."
                )
            else:
                message = (
                    "Wybrany plik pochodzi z treningu, który nie został ukończony.\n\n"
                    f"Run: {run_display or '-'} | status: {status_value or '-'}.\n"
                    "Wybierz ukończony model startowy albo uruchom właściwą akcję wznowienia."
                )
            if show_dialog:
                messagebox.showerror("To nie jest model startowy", message)
            return False, message

    if lightweight:
        task = _training_base_model_task_lightweight(self, base_key, base_model)
        if task is None:
            # An unrecognised checkpoint is inspected by the background preflight.
            return True, ""
        if task not in {"pose", "detect"}:
            message = f"Ten trening wymaga modelu POSE lub DETECT. Wybrano model {task.upper()}."
            if show_dialog:
                messagebox.showerror("Niezgodny model startowy", message)
            return False, message
        is_pose_model = task == "pose"
    else:
        is_pose_model = self._is_pose_base_model(base_key, base_model)
    if normalized_target == "plate" and not is_pose_model:
        message = (
            "Tor tablic wymaga modelu POSE.\n\n"
            "Wybierz model z dopiskiem '-pose' albo ukończony model .pt wytrenowany wcześniej dla tablic."
        )
        if show_dialog:
            messagebox.showerror("Niezgodny model startowy", message)
        return False, message

    if normalized_target == "char" and is_pose_model:
        message = (
            "Tor znaków wymaga zwykłego modelu DETECT.\n\n"
            "Wybierz model typu detect, np. `yolo11n` albo `yolo11s`, "
            "zamiast modelu pose."
        )
        if show_dialog:
            messagebox.showerror("Niezgodny model startowy", message)
        return False, message

    return True, ""

def _ensure_new_training_uses_final_base_model(self) -> bool:
    if not CAMPAIGN.get_active_project_name():
        return False
    if bool(getattr(self, "_step4_sanitizing_base_model_selection", False)):
        return False

    nonfinal_run = _selected_training_base_model_nonfinal_run(self)
    if nonfinal_run is None:
        return False

    target = CONFIG.normalize_task_target(self._get_selected_training_target())
    fallback = (
        self._get_preferred_plate_pose_base_model()
        if target == "plate"
        else self._get_default_base_model_for_mode(target)
    )
    if not fallback or self._is_custom_base_model_key(fallback):
        return False

    self._step4_sanitizing_base_model_selection = True
    try:
        self.base_model_var.set(fallback)
        self.base_custom_var.set("")
        try:
            run_id = str(getattr(nonfinal_run, "id", "") or "").strip()
            logger.info(
                "Z4/PZ2: checkpoint niedokończonego runu nie jest modelem startowym nowego treningu; "
                f"zdjęto z wyboru modelu startowego run={run_id or '-'} i ustawiono {fallback}."
            )
        except Exception:
            pass
    finally:
        self._step4_sanitizing_base_model_selection = False
    return True

def _refresh_training_start_state(self):
    button = getattr(self, "btn_start_train", None)
    if button is None:
        return
    ready = False
    pinned_state = {}
    try:
        pinned_state = _get_pinned_step4_result_state(self)
        _ensure_new_training_uses_final_base_model(self)
        preparing = bool(getattr(self, "_training_start_in_progress", False))
        ready = bool(not preparing and not pinned_state and self._is_training_configuration_ready())
        button.configure(state=(tk.NORMAL if ready else tk.DISABLED))
    except Exception:
        pass
    try:
        _refresh_step4_pinned_result_ui(self, pinned_state=pinned_state)
    except Exception:
        pass
    try:
        self._refresh_training_cockpit(ready=ready)
    except Exception:
        pass
    try:
        self._refresh_training_start_gate(ready=ready)
    except Exception:
        pass
    try:
        self._refresh_training_execution_summary()
    except Exception:
        pass
    try:
        self._refresh_training_base_model_identity_ui()
    except Exception:
        pass

def _build_training_start_gate_message(self, *, ready: bool) -> tuple[str, str, str]:
    if getattr(self, "_training_start_in_progress", False):
        return "Przygotowanie treningu", "Trwa sprawdzanie danych i przygotowanie modelu.", "warning"
    pinned_state = _get_pinned_step4_result_state(self)
    if pinned_state:
        return (
            "Model przypięty",
            _format_pinned_step4_result_detail(self, pinned_state),
            "success",
        )
    if ready:
        try:
            fine_tune_parent = self._resolve_step4_fine_tune_parent_run()
        except Exception:
            fine_tune_parent = None
        if fine_tune_parent is not None:
            run_label = self._format_training_model_run_label(fine_tune_parent)
            return (
                "Gotowe do dotrenowania",
                (
                    f"Kliknięcie rozpocznie nowy run od modelu z runu {run_label}. "
                    "Poprzedni model zostaje bez zmian; wynik nowego treningu wybierzesz jawnie po zakończeniu."
                ),
                "success",
            )
        try:
            resumable_run_id = str(self._get_latest_campaign_resumable_run_id() or "").strip()
        except Exception:
            resumable_run_id = ""
        if resumable_run_id:
            return (
                "Gotowe do nowego treningu",
                (
                    "Kliknięcie rozpocznie nowy run na aktualnym wariancie datasetu i modelu startowym. "
                    f"Aby wznowić niedokończony run {resumable_run_id}: w historii treningów zaznacz jego wiersz, "
                    "kliknij prawym przyciskiem myszy i wybierz `Wznów trening`."
                ),
                "success",
            )
        return (
            "Gotowe",
            "Po kliknięciu rozpoczniemy trening na wybranym wariancie datasetu i modelu startowym.",
            "success",
        )
    if not YOLO_AVAILABLE:
        return (
            "Brak YOLO",
            "Trening wymaga dostępnej biblioteki Ultralytics YOLO.",
            "error",
        )
    try:
        if self._step4_has_active_operation():
            return (
                "Zajęte",
                "Poczekaj na zakończenie bieżącej operacji w Z4.",
                "warning",
            )
    except Exception:
        pass
    try:
        if bool(getattr(self.trainer, "is_training", False)):
            return (
                "Trwa trening",
                "Aktualny trening jest już uruchomiony.",
                "success",
            )
    except Exception:
        pass

    dataset_yaml = None
    try:
        dataset_yaml = self._resolve_training_dataset_yaml_path()
    except Exception:
        dataset_yaml = None
    if dataset_yaml is None:
        return (
            "Brakuje datasetu",
            "Wybierz wariant splitu przygotowany w PZ1.",
            "warning",
        )

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return (
            "Brakuje modelu",
            "Wybierz model startowy zgodny z torem treningu.",
            "warning",
        )
    if self._is_custom_base_model_key(base_key):
        custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
        if not custom_model or not Path(custom_model).exists():
            return (
                "Brakuje modelu",
                "Wskaż poprawny plik .pt dla modelu startowego.",
                "warning",
            )
        nonfinal_run = _selected_training_base_model_nonfinal_run(self)
        if nonfinal_run is not None:
            status_value = str(getattr(nonfinal_run, "status", "") or "").strip().lower()
            return (
                "Checkpoint wznowienia",
                (
                    f"Wybrany plik pochodzi z runu {status_value or 'nieukończonego'} i nie jest modelem startowym. "
                    "Aby kontynuować: zaznacz ten run w historii treningów, kliknij PPM i wybierz `Wznów trening`. "
                    "Aby zacząć od nowa: wybierz preset albo ukończony model startowy."
                ),
                "warning",
            )
    return (
        "Do sprawdzenia",
        "Sprawdź zgodność datasetu, modelu i toru treningu.",
        "warning",
    )

def _refresh_training_start_gate(self, *, ready: bool):
    status_label = getattr(self, "train_start_gate_status_lbl", None)
    hint_label = getattr(self, "train_start_gate_hint_lbl", None)
    if status_label is None and hint_label is None:
        return
    status_text, hint_text, tone = self._build_training_start_gate_message(ready=ready)
    palette = getattr(self.app, "palette", {}) or {}
    tone_colors = {
        "success": (
            palette.get("surface_success", palette.get("panel_alt", "#2d2d30")),
            palette.get("success", "#2fa36b"),
        ),
        "warning": (
            palette.get("surface_warning", palette.get("panel_alt", "#2d2d30")),
            palette.get("warning", "#f0b44c"),
        ),
        "error": (
            palette.get("surface_error", palette.get("panel_alt", "#2d2d30")),
            palette.get("error", "#e05d5d"),
        ),
    }
    status_bg, status_fg = tone_colors.get(tone, tone_colors["warning"])
    try:
        if status_label is not None:
            status_label.configure(text=status_text, bg=status_bg, fg=status_fg)
    except Exception:
        pass
    try:
        if hint_label is not None:
            hint_label.configure(text=hint_text)
    except Exception:
        pass

def _training_base_model_context(self) -> dict:
    try:
        target = str(self._get_selected_training_target() or "").strip().lower()
    except Exception:
        try:
            target = str(self.get_campaign_training_target() or "").strip().lower()
        except Exception:
            target = ""
    target = CONFIG.normalize_task_target(target or "char")
    if target not in {"plate", "char"}:
        target = "char"

    try:
        iteration = int(CAMPAIGN.get_current_iteration_num() or 0) if CAMPAIGN.get_active_project_name() else 0
    except Exception:
        iteration = 0
    iteration_label = f"IT{iteration}" if iteration > 0 else "tryb swobodny"

    if target == "plate":
        target_label = "model tablic"
        target_detail = "model tablic (YOLO Pose)"
    else:
        target_label = "model znaków"
        target_detail = "model znaków (YOLO Detect)"
    return {
        "target": target,
        "target_label": target_label,
        "target_detail": target_detail,
        "iteration": iteration,
        "iteration_label": iteration_label,
        "prefix": f"{iteration_label} • {target_label}",
    }


def _set_step4_fine_tune_parent_state(self, run, model_path: str | Path) -> None:
    self._step4_fine_tune_parent_run_id = str(getattr(run, "id", "") or "").strip()
    self._step4_fine_tune_parent_model_path = str(model_path or "").strip()


def _clear_step4_fine_tune_parent_state(self) -> None:
    self._step4_fine_tune_parent_run_id = ""
    self._step4_fine_tune_parent_model_path = ""


def _resolve_step4_fine_tune_parent_run(self):
    parent_run_id = str(getattr(self, "_step4_fine_tune_parent_run_id", "") or "").strip()
    parent_model_path = str(getattr(self, "_step4_fine_tune_parent_model_path", "") or "").strip()
    if not parent_run_id or not parent_model_path:
        return None

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not self._is_custom_base_model_key(base_key):
        return None

    selected_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
    if not selected_model:
        return None

    try:
        if Path(selected_model).resolve() != Path(parent_model_path).resolve():
            return None
    except Exception:
        if selected_model != parent_model_path:
            return None

    history = getattr(self, "history", None)
    run = None
    try:
        run = history.get_run(parent_run_id) if history is not None else None
    except Exception:
        run = None
    if run is None:
        return None

    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value != TrainingStatus.COMPLETED.value:
        return None
    try:
        if not self._does_history_run_match_active_campaign_target(run):
            return None
    except Exception:
        pass
    return run


def _sync_step4_fine_tune_parent_selection(self) -> None:
    if bool(getattr(self, "_step4_setting_fine_tune_base", False)):
        return
    if not str(getattr(self, "_step4_fine_tune_parent_run_id", "") or "").strip():
        return
    if _resolve_step4_fine_tune_parent_run(self) is None:
        _clear_step4_fine_tune_parent_state(self)


def _resolve_selected_training_base_model_training_state(self) -> dict:
    context = _training_base_model_context(self)
    prefix = str(context.get("prefix") or "").strip()
    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return {
            "label": "WYBIERZ MODEL",
            "detail": f"{prefix}. Wskaż preset albo plik .pt jako punkt startu treningu.",
            "tone": "warning",
        }

    if not self._is_custom_base_model_key(base_key):
        return {
            "label": "PRESET STARTOWY YOLO",
            "detail": f"{prefix}. Świeże wagi startowe, nie wynik wcześniejszego treningu projektu.",
            "tone": "muted",
        }

    custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
    if not custom_model:
        return {
            "label": "BRAK PLIKU",
            "detail": f"{prefix}. Wskaż plik .pt albo wybierz preset YOLO.",
            "tone": "warning",
        }

    try:
        model_path = Path(custom_model)
    except Exception:
        model_path = None

    if model_path is None or not model_path.exists():
        return {
            "label": "BRAK PLIKU",
            "detail": f"{prefix}. Wskazany plik .pt nie istnieje.",
            "tone": "danger",
        }

    fine_tune_parent = _resolve_step4_fine_tune_parent_run(self)
    if fine_tune_parent is not None:
        run_label = self._format_training_model_run_label(fine_tune_parent)
        return {
            "label": "DOTRENOWANIE",
            "detail": (
                f"{prefix}. Nowy run wystartuje od modelu z runu {run_label}. "
                "Wynik pozostanie kandydatem, dopóki jawnie nie wybierzesz go jako wynik bramki."
            ),
            "tone": "info",
        }

    source_run = None
    try:
        source_run = self._resolve_training_run_from_model_path(model_path)
    except Exception:
        source_run = None

    if source_run is not None:
        status_value = str(getattr(source_run, "status", "") or "").strip().lower()
        run_label = self._format_training_model_run_label(source_run)
        if status_value == TrainingStatus.COMPLETED.value:
            try:
                model_id = build_model_display_ref(
                    model_path,
                    run=source_run,
                    target_hint=context.get("target"),
                    source_run_label=run_label,
                ).id
            except Exception:
                try:
                    model_id = Path(model_path).stem
                except Exception:
                    model_id = "model"
            return {
                "label": "GOTOWY MODEL",
                "detail": f"{prefix}. Ukończony model {model_id} z runu {run_label}.",
                "tone": "success",
            }
        return {
            "label": "CHECKPOINT RUNU",
            "detail": f"{prefix}. Run {run_label} ma status: {status_value or 'nieznany'}; do kontynuacji użyj `Wznów trening`.",
            "tone": "warning",
        }

    return {
        "label": "PLIK STARTOWY .PT",
        "detail": f"{prefix}. Zewnętrzne wagi startowe; aplikacja nie zna historii treningu tego pliku.",
        "tone": "info",
    }


def _format_training_base_model_summary_value(
    self,
    model_label: str,
    *,
    base_key: str = "",
    base_model: str = "",
    context: dict | None = None,
) -> str:
    label = str(model_label or "").strip() or "-"
    target_hint = str((context or {}).get("target") or self._get_selected_training_target() or "").strip().lower()
    try:
        if self._is_custom_base_model_key(str(base_key or "")):
            model_path = self._resolve_selected_training_base_model_path()
            if model_path is not None:
                source_run = None
                try:
                    source_run = _resolve_step4_fine_tune_parent_run(self)
                except Exception:
                    source_run = None
                if source_run is None:
                    try:
                        source_run = self._resolve_training_run_from_model_path(model_path)
                    except Exception:
                        source_run = None
                source_run_label = ""
                if source_run is not None:
                    try:
                        source_run_label = self._format_training_model_run_label(source_run)
                    except Exception:
                        source_run_label = ""
                label = build_model_display_ref(
                    model_path,
                    run=source_run,
                    target_hint=target_hint,
                    source_run_label=source_run_label,
                ).id
    except Exception:
        pass
    if label.lower().endswith(".pt"):
        try:
            label = Path(label).stem
        except Exception:
            label = label[:-3].rstrip(".") or label

    task_label = ""
    try:
        _model_path, info = self._resolve_selected_training_base_model_info()
    except Exception:
        info = {}
    if isinstance(info, dict):
        task = str(info.get("task") or info.get("type") or "").strip().lower()
        architecture = str(info.get("architecture_label") or "").strip().lower()
        if task == "pose" or "pose" in architecture:
            task_label = "Pose"
        elif task or "detect" in architecture:
            task_label = "Detect"

    if not task_label:
        try:
            is_pose = bool(self._is_pose_base_model(str(base_key or ""), str(base_model or "")))
        except Exception:
            is_pose = False
        if is_pose:
            task_label = "Pose"
        else:
            task_label = "Pose" if target_hint == "plate" else "Detect"

    return f"{label} | {task_label}" if task_label and label != "-" else label


def _refresh_training_base_model_identity_ui(self):
    label = getattr(self, "train_base_identity_lbl", None)
    status_label = getattr(self, "train_base_status_lbl", None)
    title_label = getattr(self, "train_base_title_lbl", None)
    caption_label = getattr(self, "train_base_caption_lbl", None)
    context = _training_base_model_context(self)
    state = _resolve_selected_training_base_model_training_state(self)
    palette = getattr(self.app, "palette", {}) or {}
    tone = str(state.get("tone") or "muted")
    panel = palette.get("panel", "#252526")
    tone_colors = {
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f0b44c"),
        "danger": palette.get("danger", "#e74c3c"),
        "info": palette.get("accent", "#0e639c"),
        "muted": palette.get("muted", "#c7c7c7"),
    }
    tone_fg = tone_colors.get(tone, palette.get("muted", "#c7c7c7"))
    tone_bg = blend_hex_colors(tone_fg, panel, 0.82)

    try:
        if title_label is not None:
            title_label.configure(text="Model startowy treningu")
    except Exception:
        pass
    try:
        if caption_label is not None:
            caption_label.configure(
                text=(
                    f"{context.get('iteration_label', 'IT')} • trenujesz {context.get('target_detail', 'model')}. "
                    "To wagi startowe nowego runu. Ukończone modele projektu znajdziesz obok "
                    "w Historii treningów; tam wybierzesz wynik bramki albo punkt dotrenowania."
                )
            )
    except Exception:
        pass
    try:
        chips = getattr(self, "train_base_context_chips", {}) or {}
        chip_values = {
            "iteration": str(context.get("iteration_label") or "IT"),
            "target": str(context.get("target_label") or "model"),
            "backend": "YOLO Pose" if context.get("target") == "plate" else "YOLO Detect",
        }
        for chip_key, chip_text in chip_values.items():
            chip = chips.get(chip_key)
            if chip is not None:
                chip.configure(text=chip_text)
    except Exception:
        pass

    if status_label is not None:
        try:
            status_label.configure(
                text=str(state.get("label") or "STATUS MODELU"),
                bg=tone_bg,
                fg=tone_fg,
                highlightbackground=blend_hex_colors(tone_fg, palette.get("panel_border", "#3c3c3c"), 0.55),
                highlightcolor=blend_hex_colors(tone_fg, palette.get("panel_border", "#3c3c3c"), 0.55),
            )
        except Exception:
            pass

    try:
        summary_values = getattr(self, "train_base_summary_values", {}) or {}
        base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
        selected_text = base_key or "-"
        origin_text = "Preset YOLO"
        state_text = str(state.get("label") or "-")
        custom_model = ""
        if self._is_custom_base_model_key(base_key):
            custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
            selected_text = Path(custom_model).name if custom_model else "-"
            origin_text = "Plik .pt"
            model_path = Path(custom_model) if custom_model else None
            fine_tune_parent = _resolve_step4_fine_tune_parent_run(self)
            source_run = None
            if model_path is not None:
                try:
                    source_run = self._resolve_training_run_from_model_path(model_path)
                except Exception:
                    source_run = None
            if fine_tune_parent is not None and model_path is not None:
                run_label = self._format_training_model_run_label(fine_tune_parent)
                selected_text = build_model_display_ref(
                    model_path,
                    run=fine_tune_parent,
                    target_hint=context.get("target"),
                    source_run_label=run_label,
                ).id
                origin_text = f"Dotrenowanie z {run_label}"
            elif source_run is not None and model_path is not None:
                run_label = self._format_training_model_run_label(source_run)
                selected_text = build_model_display_ref(
                    model_path,
                    run=source_run,
                    target_hint=context.get("target"),
                    source_run_label=run_label,
                ).id
                status_value = str(getattr(source_run, "status", "") or "").strip().lower()
                if status_value == TrainingStatus.COMPLETED.value:
                    origin_text = f"Historia treningów: {run_label}"
                    state_text = "Gotowy do użycia"
                else:
                    origin_text = f"Checkpoint {run_label}"
            elif custom_model:
                origin_text = "Zewnętrzny plik .pt"

        selected_text = _format_training_base_model_summary_value(
            self,
            selected_text,
            base_key=base_key,
            base_model=custom_model or base_key,
            context=context,
        )

        summary_payload = {
            "selected": selected_text,
            "origin": origin_text,
            "state": state_text,
        }
        for key, value in summary_payload.items():
            value_label = summary_values.get(key)
            if value_label is not None:
                value_label.configure(text=str(value or "-"))
        state_label = summary_values.get("state")
        if state_label is not None:
            state_label.configure(fg=tone_fg)
    except Exception:
        pass

    if label is None:
        return

    lines = self._build_selected_training_base_model_identity_lines()
    detail = str(state.get("detail") or "").strip()
    if detail and not getattr(self, "train_base_summary_values", None):
        lines.insert(0, detail)
    text = "\n".join(lines).strip()
    try:
        label.configure(text=text)
    except Exception:
        pass
    try:
        if text:
            if not str(label.winfo_manager()):
                label.pack(anchor=tk.W, fill=tk.X, padx=8, pady=(0, 8))
        else:
            label.pack_forget()
    except Exception:
        pass

def _resolve_selected_training_base_model_display(self) -> str:
    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return "Nie wybrano modelu"
    if not self._is_custom_base_model_key(base_key):
        return base_key

    custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
    if not custom_model:
        return "Własny plik .pt (brak pliku)"
    try:
        return f"Własny plik .pt -> {Path(custom_model).name}"
    except Exception:
        return f"Własny plik .pt -> {custom_model}"

def _resolve_selected_training_base_model_path(self) -> Path | None:
    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not self._is_custom_base_model_key(base_key):
        return None

    custom_model = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
    if not custom_model:
        return None

    try:
        model_path = Path(custom_model)
    except Exception:
        return None
    return model_path if model_path.exists() else None

def _resolve_selected_training_base_model_source_run(self):
    model_path = self._resolve_selected_training_base_model_path()
    if model_path is None:
        return None
    try:
        return self._resolve_training_run_from_model_path(model_path)
    except Exception:
        return None

def _selected_training_base_model_nonfinal_run(self):
    run = _resolve_selected_training_base_model_source_run(self)
    if run is None:
        return None
    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value and status_value != TrainingStatus.COMPLETED.value:
        return run
    return None

def _resolve_selected_training_base_model_inspection_path(self) -> Path | None:
    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    if not base_key:
        return None
    if self._is_custom_base_model_key(base_key):
        return self._resolve_selected_training_base_model_path()

    target = self._get_selected_training_target()
    catalog = AVAILABLE_POSE_MODELS if target == "plate" else AVAILABLE_DETECT_MODELS
    entry = catalog.get(base_key, {}) if isinstance(catalog, dict) else {}
    candidate_names = [str(entry.get("file") or "").strip(), base_key]
    search_roots = [Path.cwd(), CONFIG.get_base_models_dir(target)]

    for candidate_name in candidate_names:
        if not candidate_name:
            continue
        try:
            direct_path = Path(candidate_name)
        except Exception:
            continue
        if direct_path.exists():
            return direct_path
        for root in search_roots:
            try:
                candidate_path = root / candidate_name
            except Exception:
                continue
            if candidate_path.exists():
                return candidate_path
    return None

def _training_base_model_task_lightweight(self, base_key: str, base_model: str) -> str | None:
    if base_key in AVAILABLE_POSE_MODELS:
        return "pose"
    if base_key in AVAILABLE_DETECT_MODELS:
        return "detect"
    path = Path(base_model)
    if not path.is_file():
        return None
    ok, _message, info = validate_model_file(path, allow_heavy_load=False)
    if ok:
        task = str(info.get("task") or info.get("type") or "").lower()
        if task in {"pose", "detect", "classify", "segment", "obb"}:
            return task
    return None


def _resolve_selected_training_base_model_info(self, *, lightweight: bool = False) -> tuple[Path | None, dict]:
    model_path = self._resolve_selected_training_base_model_inspection_path()
    if model_path is None or not model_path.exists():
        return None, {}
    ok, _message, info = validate_model_file(model_path, allow_heavy_load=not lightweight)
    return model_path, info if ok and isinstance(info, dict) else {}

def _build_selected_training_base_model_identity_lines(self) -> list[str]:
    model_path, info = self._resolve_selected_training_base_model_info()
    if model_path is None or not info:
        return []

    lines: list[str] = []
    nonfinal_run = _selected_training_base_model_nonfinal_run(self)
    if nonfinal_run is not None:
        run_id = str(getattr(nonfinal_run, "id", "") or "").strip()
        status_value = str(getattr(nonfinal_run, "status", "") or "").strip().lower()
        lines.append(f"Checkpoint niedokończonego treningu: {run_id or '-'} ({status_value or '-'})")
        lines.append("Do kontynuacji użyj `Wznów trening`; to nie jest finalny model startowy.")

    return lines

def _build_training_recommendation_model_text(self) -> str:
    model_path, info = self._resolve_selected_training_base_model_info()
    base_display = self._resolve_selected_training_base_model_display()
    identity_label = format_yolo_model_identity(info)
    if not identity_label:
        return f"Model: {base_display}"

    source_architecture = str(info.get("source_architecture_label") or "").strip()
    source_model_name = str(info.get("source_model_name") or "").strip()
    source_display = source_architecture or source_model_name
    is_checkpoint_like = bool(
        model_path is not None
        and (
            model_path.name.lower() == "best.pt"
            or model_path.stem.lower().startswith("epoch")
        )
    )

    text = f"Model: {identity_label}"
    if is_checkpoint_like and source_display:
        text += f" | po wcześniejszym treningu: {source_display}"
    return text

def _format_training_model_run_label(run) -> str:
    if run is None:
        return "-"
    try:
        return build_run_display_ref(run, kind_hint="training").id
    except Exception:
        run_name = str(getattr(run, "name", "") or "").strip()
        run_id = str(getattr(run, "id", "") or "").strip()
        return run_name or run_id or "-"

def _format_training_model_size_label(raw_size: str | None) -> str:
    size = str(raw_size or "").strip().lower()
    labels = {
        "n": "n (najmniejszy)",
        "s": "s (mały)",
        "m": "m (średni)",
        "l": "l (duży)",
        "x": "x (bardzo duży)",
    }
    return labels.get(size, size)

def _build_training_model_summary_value(self, base_display: str, info: dict | None) -> str:
    if not isinstance(info, dict) or not info:
        return base_display

    identity_label = format_yolo_model_identity(info)
    version = str(info.get("yolo_version") or "").strip()
    size = self._format_training_model_size_label(
        str(info.get("yolo_size") or info.get("model_scale") or "").strip().lower()
    )

    extras: list[str] = []
    if version:
        extras.append(f"wersja {version}")
    if size:
        extras.append(f"rozmiar {size}")

    if identity_label and extras:
        return f"{identity_label} | {', '.join(extras)}"
    if identity_label:
        return identity_label
    return base_display

def _resolve_training_model_version_from_info(info: dict | None) -> str:
    if not isinstance(info, dict):
        return ""
    version = str(info.get("yolo_version") or "").strip()
    return f"YOLO {version}" if version else ""

def _resolve_training_model_size_from_info(self, info: dict | None) -> str:
    if not isinstance(info, dict):
        return ""
    return self._format_training_model_size_label(
        str(info.get("yolo_size") or info.get("model_scale") or "").strip().lower()
    )

def _get_pose_model_memory_bucket(info: dict | None) -> str:
    if not isinstance(info, dict):
        return ""
    size = str(info.get("yolo_size") or info.get("model_scale") or "").strip().lower()
    if size in {"n", "s", "m", "l", "x"}:
        return size

    architecture = str(info.get("architecture_label") or info.get("source_architecture_label") or "").strip().lower()
    for candidate in ("x", "l", "m", "s", "n"):
        if f"yolo26{candidate}" in architecture or f"yolo11{candidate}" in architecture or f"yolov8{candidate}" in architecture:
            return candidate
    return ""

def _get_training_gpu_capacity_block_reason(
    self,
    *,
    is_pose_dataset: bool,
    base_model_info: dict | None,
    effective_device_profile: dict | None,
) -> str:
    if not is_pose_dataset:
        return ""
    if not isinstance(effective_device_profile, dict):
        return ""

    try:
        memory_gb = float(effective_device_profile.get("memory_gb", 0.0) or 0.0)
    except Exception:
        memory_gb = 0.0
    if memory_gb <= 0.0:
        return ""

    bucket = self._get_pose_model_memory_bucket(base_model_info)
    if bucket not in {"m", "l", "x"}:
        return ""
    if memory_gb > 4.5:
        return ""

    architecture_label = format_yolo_model_identity(base_model_info) or "duży model YOLO Pose"
    gpu_name = str(effective_device_profile.get("name") or "GPU").strip()
    return (
        f"Wybrany model startowy to {architecture_label}, a aktywne urządzenie to {gpu_name} "
        f"z około {memory_gb:.1f} GB VRAM.\n\n"
        "Ten rozmiar modelu pose na 4 GB VRAM w obecnym środowisku kończy się błędami pamięci CUDA "
        "jeszcze przed stabilnym startem treningu albo w pierwszych batchach.\n\n"
        "Aby trening miał realną szansę powodzenia:\n"
        "1. wybierz mniejszy model pose, najlepiej `yolo11s-pose` albo `yolo26s-pose`,\n"
        "2. albo uruchom trening na CPU,\n"
        "3. albo użyj GPU z większym VRAM, jeśli chcesz kontynuować właśnie ten model."
    )

def _resolve_training_run_from_model_path(self, model_path: Path | None):
    if model_path is None:
        return None

    try:
        resolved_model_path = model_path.resolve()
    except Exception:
        resolved_model_path = model_path

    history = getattr(self, "history", None)
    if history is None:
        return None

    try:
        runs = list(history.get_all_runs() or [])
    except Exception:
        runs = []

    for run in runs:
        best_weights = str(getattr(run, "best_weights", "") or "").strip()
        if not best_weights:
            continue
        try:
            if Path(best_weights).resolve() == resolved_model_path:
                return run
        except Exception:
            continue

    for run in runs:
        output_dir = str(getattr(run, "output_dir", "") or "").strip()
        if not output_dir:
            continue
        try:
            resolved_output_dir = Path(output_dir).resolve()
            if resolved_output_dir == resolved_model_path or resolved_output_dir in resolved_model_path.parents:
                return run
        except Exception:
            continue

    try:
        model_parts = {str(part) for part in resolved_model_path.parts}
    except Exception:
        model_parts = set()

    for run in runs:
        run_id = str(getattr(run, "id", "") or "").strip()
        if run_id and run_id in model_parts:
            return run

    return None

def _build_selected_training_base_model_origin_rows(self, base_model_path: Path | None) -> list[tuple[str, str]]:
    if base_model_path is None or base_model_path.name.lower() != "best.pt":
        return []

    source_run = self._resolve_training_run_from_model_path(base_model_path)
    if source_run is None:
        return []

    run_label = self._format_training_model_run_label(source_run)
    return [("Pochodzenie modelu", f"Model z wcześniejszego treningu: {run_label}")]

def _append_training_execution_summary_row_widget(self, row_index: int) -> None:
    summary_grid = getattr(self, "train_run_summary_rows_frame", None)
    if summary_grid is None:
        summary_grid = getattr(self, "train_run_summary_grid", None)
        if summary_grid is None:
            return

    palette = getattr(self.app, "palette", {}) or {}
    row_frame = tk.Frame(
        summary_grid,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_border", palette.get("border", "#3c3c3c")),
    )
    row_frame.pack(fill=tk.X, padx=0, pady=(0, 1))
    key_label = tk.Label(
        row_frame,
        text="",
        font=("Segoe UI", 8, "bold"),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=7,
        pady=3,
        width=12,
        bg=palette.get("panel", "#252526"),
        fg=palette.get("fg", "#f3f3f3"),
    )
    key_label.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1), pady=0)
    value_label = tk.Label(
        row_frame,
        text="",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=7,
        pady=3,
        wraplength=250,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        fg=palette.get("muted", "#c7c7c7"),
    )
    value_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=0, pady=0)
    self._register_train_left_wrap_target(value_label, container=self.train_run_summary_shell, padding=140, min_wrap=180)
    row_frame.pack_forget()
    self._train_run_summary_row_widgets.append(
        {"frame": row_frame, "key": key_label, "value": value_label, "row_index": row_index}
    )

def _ensure_training_execution_summary_row_capacity(self, required_rows: int) -> None:
    summary_grid = getattr(self, "train_run_summary_grid", None)
    if summary_grid is None:
        return
    row_widgets = getattr(self, "_train_run_summary_row_widgets", None)
    if not isinstance(row_widgets, list):
        self._train_run_summary_row_widgets = []
        row_widgets = self._train_run_summary_row_widgets
    while len(row_widgets) < max(0, int(required_rows)):
        self._append_training_execution_summary_row_widget(len(row_widgets))

def _build_training_execution_summary_rows(self) -> list[tuple[str, str]]:
    target = self._get_selected_training_target()
    target_label = self._format_training_target_label(target)
    base_model_display = self._resolve_selected_training_base_model_display()
    _, base_model_info = self._resolve_selected_training_base_model_info()
    dataset_yaml = self._resolve_training_dataset_yaml_path()
    epochs_value = self._safe_training_int_value("epochs_var", default=100, minimum=1)
    batch_value = self._safe_training_int_value("batch_var", default=16, minimum=1)
    imgsz_value = self._safe_training_int_value("imgsz_var", default=640, minimum=32)
    lr0_value = self._safe_training_float_value("lr0_var", default=0.01, minimum=0.0001)
    model_summary = self._build_training_model_summary_value(base_model_display, base_model_info)

    if dataset_yaml is None:
        return [
            ("Tor", target_label),
            ("Dataset", "Wybierz wariant splitu z PZ1."),
            ("Model", model_summary),
            ("Parametry", f"{epochs_value} epok | batch {batch_value} | {imgsz_value}px | LR {lr0_value:.4f}"),
            ("Status", "Trening będzie dostępny po wskazaniu poprawnego datasetu YOLO."),
        ]

    dataset_root = dataset_yaml.parent
    dataset_rel = self._format_workspace_relative_path(dataset_root)
    counts = self._get_dataset_split_image_counts(dataset_root)
    total_images = int(counts.get("total", 0) or 0)
    train_images = int(counts.get("train", 0) or 0)
    val_images = int(counts.get("val", 0) or 0)
    test_images = int(counts.get("test", 0) or 0)

    try:
        cfg = safe_load_yaml(dataset_yaml) or {}
    except Exception:
        cfg = {}
    class_names = self._extract_dataset_class_names(cfg)
    class_count = len(class_names)
    if class_count <= 0:
        try:
            class_count = max(0, int(cfg.get("nc", 0) or 0))
        except Exception:
            class_count = 0
    dataset_task = "YOLO Pose" if bool(cfg.get("kpt_shape")) else "YOLO Detect"
    class_suffix = f" | klasy: {class_count}" if class_count > 0 else ""

    return [
        ("Tor", target_label),
        ("Dataset", dataset_rel),
        ("Próbka", f"{total_images} obrazów | train {train_images}, val {val_images}, test {test_images}"),
        ("Typ", f"{dataset_task}{class_suffix}"),
        ("Model", model_summary),
        ("Parametry", f"{epochs_value} epok | batch {batch_value} | {imgsz_value}px | LR {lr0_value:.4f}"),
    ]

def _refresh_training_execution_summary(self):
    if getattr(self, "train_run_summary_shell", None) is None:
        return
    row_widgets = list(getattr(self, "_train_run_summary_row_widgets", []) or [])
    if not row_widgets:
        return
    rows = self._build_training_execution_summary_rows()
    self._ensure_training_execution_summary_row_capacity(len(rows))
    for widgets in row_widgets:
        row_frame = widgets.get("frame")
        if row_frame is None:
            continue
        try:
            row_frame.pack_forget()
        except Exception:
            pass
    for row_index, widgets in enumerate(row_widgets):
        row_frame = widgets.get("frame")
        key_label = widgets.get("key")
        value_label = widgets.get("value")
        row_visible = row_index < len(rows)
        if row_visible:
            key_text, value_text = rows[row_index]
        else:
            key_text, value_text = "", ""
        if row_visible and row_frame is not None:
            try:
                row_frame.pack(fill=tk.X, padx=0, pady=(0, 1))
            except Exception:
                pass
        try:
            if key_label is not None:
                key_label.configure(text=str(key_text or ""))
        except Exception:
            pass
        try:
            if value_label is not None:
                value_label.configure(text=str(value_text or ""))
        except Exception:
            pass

def _terminal_metric_tag(band: str) -> str:
    normalized = str(band or "").strip().lower()
    if normalized in {"bardzo dobry", "dobry"}:
        return "terminal_success"
    if normalized in {"używalny", "monitoruj"}:
        return "terminal_warning"
    if normalized == "slaby":
        return "terminal_error"
    return "terminal_default"

def _format_terminal_table_line(values: list[str], widths: list[int], alignments: list[str]) -> str:
    cells = []
    for value, width, alignment in zip(values, widths, alignments):
        text = str(value or "")
        if alignment == "center":
            cells.append(text.center(width))
        elif alignment == "right":
            cells.append(text.rjust(width))
        else:
            cells.append(text.ljust(width))
    return "| " + " | ".join(cells) + " |"

def _build_training_metric_terminal_entries(self, epoch: int, total_epochs: int, metrics: dict | None) -> list[tuple[str, str]]:
    rows = self._build_training_metric_rows(metrics)
    if not rows:
        return []

    headers = ["Metryka", "Wynik", "Ocena", "Zakres"]
    alignments = ["left", "center", "center", "center"]
    widths = []
    for idx, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(str(row[idx] if idx < len(row) else "")))
        widths.append(max_width)

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = self._format_terminal_table_line(headers, widths, alignments)
    entries: list[tuple[str, str]] = []

    target_label = "tablice" if self.get_campaign_training_target() == "plate" else "znaki"
    entries.append((f"[Z4] BIEZACE WYNIKI | epoka {int(epoch)}/{int(total_epochs)} | tor: {target_label}", "terminal_info"))
    entries.append((separator, "terminal_border"))
    entries.append((header_row, "terminal_header"))
    entries.append((separator, "terminal_border"))

    for row in rows:
        band = str(row[2] or "")
        entries.append(
            (
                self._format_terminal_table_line([str(cell) for cell in row], widths, alignments),
                self._terminal_metric_tag(band),
            )
        )

    entries.append((separator, "terminal_border"))
    return entries

def _append_training_metric_table_to_global(self, epoch: int, total_epochs: int, metrics: dict | None):
    try:
        entries = self._build_training_metric_terminal_entries(epoch, total_epochs, metrics)
        if not entries:
            return
        if hasattr(self.app, "append_global_terminal_entries"):
            self.app.append_global_terminal_entries(entries + [("", "terminal_default")])
    except Exception:
        pass

def _set_history_run_tables(self, run):
    if run is None:
        self._set_metric_table_rows(getattr(self, "hist_detail_tree", None), [])
        self._set_metric_table_rows(getattr(self, "hist_metrics_tree", None), [])
        return

    self._set_metric_table_rows(
        getattr(self, "hist_detail_tree", None),
        self._build_training_run_detail_rows(run),
    )
    self._set_metric_table_rows(
        getattr(self, "hist_metrics_tree", None),
        self._build_training_run_metric_rows(run),
    )
