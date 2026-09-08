#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 trained model export helpers extracted from tab_training.py."""

from __future__ import annotations

import json
import importlib.util
import importlib.metadata as importlib_metadata
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import textwrap
import traceback
import datetime
import time
import urllib.parse
import urllib.request
import webbrowser
import csv
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
import tkinter.font as tkfont
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
from ..exporters.mobile_model_exporter import (
    MobileAlprPackageExporter,
    MobileAlprPackageRequest,
    MobileExportError,
    MobileExportRequest,
    MobileModelExporter,
    check_mobile_yolo_export_runtime,
    mobile_export_required_specs,
    mobile_export_requirement_status,
)
from ..validators import (
    validate_model_file,
    format_yolo_model_identity,
    read_model_metadata_sidecar,
    write_model_metadata_sidecar,
)
from ..training import (
    YOLOPoseTrainer,
    TrainingHistory,
    TrainingStatus,
    DatasetCreator,
    DatasetSplitter,
    build_model_training_provenance,
)
from ..ranking import ModelRanking
from ..training.model_provenance import build_checkpoint_metric_summary, _file_sha256
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .free_mode_assistant import get_mobile_export_assistant_context
from .app_theme_definitions import normalize_theme_palette
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_mobile_report_browser import open_mobile_report_browser
from .z4_mobile_export_locations import MobileExportSourceLocations
from .z4_mobile_export_contract import describe_mobile_export_selection
from .dataset_display import build_dataset_display_ref
from .model_display import build_model_display_ref
from .run_display import build_run_display_ref
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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _infer_history_run_target(self, run) -> str:
    target = ""
    try:
        infer_target = getattr(self.history, "_infer_run_target", None)
        if callable(infer_target):
            target = str(infer_target(run) or "").strip().lower()
    except Exception:
        target = ""

    if target not in {"plate", "char", "vehicle"}:
        try:
            target = str(self._infer_dataset_target(getattr(run, "dataset_path", "")) or "").strip().lower()
        except Exception:
            target = ""

    if target not in {"plate", "char", "vehicle"}:
        try:
            merged = " ".join(
                str(value or "")
                for value in (
                    getattr(run, "dataset_path", ""),
                    getattr(run, "base_model", ""),
                    getattr(run, "name", ""),
                    getattr(run, "output_dir", ""),
                )
            )
            infer_from_text = getattr(TrainingHistory, "_infer_target_from_text", None)
            if callable(infer_from_text):
                target = str(infer_from_text(merged) or "").strip().lower()
        except Exception:
            target = ""

    return target if target in {"plate", "char", "vehicle"} else ""

def _resolve_history_run_best_weights(self, run) -> Path | None:
    if run is None:
        return None
    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if status_value != TrainingStatus.COMPLETED.value:
        return None

    for raw_path in (
        getattr(run, "best_weights", ""),
        Path(str(getattr(run, "output_dir", "") or "")) / "train" / "weights" / "best.pt",
    ):
        text = str(raw_path or "").strip()
        if not text:
            continue
        try:
            candidate = Path(text)
        except Exception:
            continue
        if candidate.exists() and candidate.is_file():
            return candidate

    try:
        return self._find_best_weights_for_run(str(getattr(run, "id", "") or "").strip())
    except Exception:
        return None

def _build_history_run_metric_summary(self, run) -> dict:
    checkpoint = self._resolve_history_run_best_weights(run) if run is not None else None
    return build_checkpoint_metric_summary(run, checkpoint=checkpoint, verify_checkpoint=False)



def _build_free_mode_model_export_path(self, run, target: str, target_dir: Path, source_path: Path) -> Path:
    prefix = {
        "plate": "pose",
        "char": "char",
        "vehicle": "vehicle",
    }.get(target, "model")
    run_name = self._safe_model_export_slug(getattr(run, "name", "") or getattr(run, "id", ""), fallback=prefix)
    run_id = self._safe_model_export_slug(getattr(run, "id", ""), fallback=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    metric_summary = self._build_history_run_metric_summary(run)
    metric_value = metric_summary.get("best_map50_95")
    if metric_value is None:
        metric_value = metric_summary.get("best_map50")
    metric_tag = ""
    if metric_value is not None:
        try:
            clamped = max(0.0, min(1.0, float(metric_value)))
            metric_tag = f"_map{int(round(clamped * 100)):03d}"
        except Exception:
            metric_tag = ""
    suffix = source_path.suffix if source_path.suffix else ".pt"
    return target_dir / f"{prefix}_{run_name}_{run_id}{metric_tag}{suffix}"

def _mobile_role_from_training_target(target: str) -> str:
    normalized = str(target or "").strip().lower()
    if normalized == "plate":
        return "plate"
    if normalized == "vehicle":
        return "vehicle"
    return "character"

_MOBILE_EXPORT_CHARACTER_CLASS_TOKENS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MOBILE_EXPORT_PLATE_CLASS_NAMES = {
    "plate",
    "plates",
    "license_plate",
    "license-plate",
    "licence_plate",
    "licence-plate",
    "tablica",
    "tablice",
}
_MOBILE_EXPORT_VEHICLE_CLASS_NAMES = {
    "vehicle",
    "vehicles",
    "car",
    "cars",
    "truck",
    "trucks",
    "bus",
    "buses",
    "motorcycle",
    "motorbike",
    "bike",
    "van",
    "pickup",
    "suv",
    "pojazd",
    "pojazdy",
    "samochod",
    "samochody",
}


def _mobile_export_normalized_target_or_empty(value) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"plate", "plates", "pose", "tablica", "tablice", "mt"}:
        return "plate"
    if raw in {"char", "chars", "character", "characters", "ocr", "znak", "znaki", "mz"}:
        return "char"
    if raw in {"vehicle", "vehicles", "pojazd", "pojazdy", "car", "cars", "mp"}:
        return "vehicle"
    return ""


def _mobile_export_class_names_from_payload(payload: dict | None) -> list[str]:
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))
            values = [names[key] for key in ordered_keys]
        except Exception:
            values = list(names.values())
    elif isinstance(names, (list, tuple, set)):
        values = list(names)
    else:
        values = []
    return [str(name or "").strip() for name in values if str(name or "").strip()]


def _mobile_export_classes_look_like_character_alphabet(class_names: list[str]) -> bool:
    if len(class_names) < 8:
        return False
    normalized = [str(name or "").strip().upper() for name in class_names]
    return all(len(name) == 1 and name in _MOBILE_EXPORT_CHARACTER_CLASS_TOKENS for name in normalized)


def _mobile_export_infer_target_from_text(text) -> tuple[str, str]:
    raw = str(text or "").strip().casefold()
    if not raw:
        return "", ""
    scores = {"plate": 0, "char": 0, "vehicle": 0}
    if any(token in raw for token in ("char", "chars", "character", "characters", "ocr", "znak", "znaki", "mz-")):
        scores["char"] += 45
    if any(token in raw for token in ("vehicle", "vehicles", "pojazd", "pojazdy", "samochod", "samochody", "car", "cars", "coco", "mp-")):
        scores["vehicle"] += 45
    if any(token in raw for token in ("plate", "plates", "tablic", "license", "licence", "pose", "mt-")):
        scores["plate"] += 35
    winner, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "", ""
    tied = [target for target, value in scores.items() if value == score]
    if len(tied) > 1:
        return "", ""
    return winner, "tekst ścieżki lub nazwy"


def _mobile_export_infer_target_from_yaml_payload(payload: dict | None, path_hint="") -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", ""
    if payload.get("kpt_shape") is not None:
        return "plate", "data.yaml: kpt_shape / YOLO Pose"

    class_names = _mobile_export_class_names_from_payload(payload)
    lowered_names = {str(name or "").strip().casefold() for name in class_names}
    if class_names and lowered_names and lowered_names.issubset(_MOBILE_EXPORT_PLATE_CLASS_NAMES):
        return "plate", "data.yaml: klasy tablic"
    if _mobile_export_classes_look_like_character_alphabet(class_names):
        return "char", "data.yaml: alfabet znaków"
    if lowered_names and lowered_names.intersection(_MOBILE_EXPORT_VEHICLE_CLASS_NAMES):
        return "vehicle", "data.yaml: klasy pojazdów"

    target, reason = _mobile_export_infer_target_from_text(path_hint)
    return target, reason


def _mobile_export_infer_target_from_dataset_path(dataset_path) -> tuple[str, str]:
    raw = str(dataset_path or "").strip()
    if not raw:
        return "", ""
    try:
        path = Path(raw)
        yaml_path = path if path.is_file() and path.suffix.lower() in {".yaml", ".yml"} else path / "data.yaml"
        if yaml_path.exists() and yaml_path.is_file():
            payload = safe_load_yaml(yaml_path) or {}
            target, reason = _mobile_export_infer_target_from_yaml_payload(payload, yaml_path)
            if target:
                return target, reason
    except Exception:
        pass
    return _mobile_export_infer_target_from_text(raw)


def _mobile_export_infer_target_from_model_info(info: dict | None) -> tuple[str, str]:
    if not isinstance(info, dict) or not info:
        return "", ""
    task = str(info.get("task") or info.get("type") or info.get("model_task") or "").strip().lower()
    if task == "pose" or info.get("keypoints") is True or info.get("kpt_shape"):
        return "plate", "metadane modelu: YOLO Pose"
    classes = info.get("classes")
    if isinstance(classes, dict):
        class_names = [str(value or "").strip() for value in classes.values()]
    elif isinstance(classes, (list, tuple, set)):
        class_names = [str(value or "").strip() for value in classes]
    else:
        class_names = []
    lowered_names = {str(name or "").strip().casefold() for name in class_names if str(name or "").strip()}
    if _mobile_export_classes_look_like_character_alphabet(class_names):
        return "char", "metadane modelu: klasy znaków"
    if lowered_names and lowered_names.issubset(_MOBILE_EXPORT_PLATE_CLASS_NAMES):
        return "plate", "metadane modelu: klasy tablic"
    if lowered_names and lowered_names.intersection(_MOBILE_EXPORT_VEHICLE_CLASS_NAMES):
        return "vehicle", "metadane modelu: klasy pojazdów"
    return "", ""


def _mobile_export_infer_target_from_run_args(run_like) -> tuple[str, str]:
    output_dir = str(_mobile_export_run_like_value(run_like, "output_dir", "") or "").strip()
    if not output_dir:
        return "", ""
    try:
        args_path = Path(output_dir) / "train" / "args.yaml"
        if not args_path.exists():
            return "", ""
        args_payload = safe_load_yaml(args_path) or {}
    except Exception:
        return "", ""
    if not isinstance(args_payload, dict):
        return "", ""
    task = str(args_payload.get("task") or "").strip().lower()
    if task == "pose":
        return "plate", "args.yaml: task=pose"
    data_target, data_reason = _mobile_export_infer_target_from_dataset_path(args_payload.get("data"))
    if data_target:
        return data_target, data_reason
    target, reason = _mobile_export_infer_target_from_text(
        " ".join(str(args_payload.get(key) or "") for key in ("model", "data", "project", "name"))
    )
    if target:
        return target, f"args.yaml: {reason}"
    if task == "detect":
        return "char", "args.yaml: task=detect bez klas pojazdów/tablic"
    return "", ""


def _mobile_export_resolve_candidate_target(
    *,
    fallback_target: str = "",
    run_like=None,
    dataset_path="",
    model_info: dict | None = None,
) -> tuple[str, str, str]:
    observations: list[tuple[str, str, str]] = []

    info_target, info_reason = _mobile_export_infer_target_from_model_info(model_info)
    if info_target:
        observations.append((info_target, info_reason, "model"))

    dataset_target, dataset_reason = _mobile_export_infer_target_from_dataset_path(dataset_path)
    if dataset_target:
        observations.append((dataset_target, dataset_reason, "dataset"))

    args_target, args_reason = _mobile_export_infer_target_from_run_args(run_like)
    if args_target:
        observations.append((args_target, args_reason, "args"))

    explicit_target = _mobile_export_normalized_target_or_empty(
        _mobile_export_run_like_value(run_like, "training_target", "")
        or _mobile_export_run_like_value(run_like, "parent_model_target", "")
    )
    if explicit_target:
        observations.append((explicit_target, "historia treningu: jawny tor", "history"))

    fallback = _mobile_export_normalized_target_or_empty(fallback_target)
    if fallback:
        observations.append((fallback, "katalog lub źródło listy", "fallback"))

    if not observations:
        return "", "", ""

    priority = {"model": 0, "dataset": 1, "args": 2, "history": 3, "fallback": 4}
    observations.sort(key=lambda item: priority.get(item[2], 99))
    resolved_target, resolved_reason, _kind = observations[0]
    conflicts = sorted({target for target, _reason, _kind in observations if target and target != resolved_target})
    conflict_text = ""
    if conflicts:
        labels = {"plate": "MT", "char": "MZ", "vehicle": "MP"}
        conflict_text = "Sprzeczne slady: " + ", ".join(labels.get(target, target) for target in conflicts)
    return resolved_target, resolved_reason, conflict_text


def _mobile_export_task_from_target_and_info(target: str, info: dict | None = None) -> str:
    info_target, _reason = _mobile_export_infer_target_from_model_info(info)
    raw_task = str((info or {}).get("task") or (info or {}).get("type") or "").strip().lower() if isinstance(info, dict) else ""
    if raw_task == "pose" or info_target == "plate" or _mobile_export_normalized_target_or_empty(target) == "plate":
        return "pose"
    return "detect"

_MOBILE_EXPORT_COCO_VEHICLE_CLASS_IDS = (2, 3, 5, 7)
_MOBILE_EXPORT_DEFAULT_VEHICLE_CLASS_LABELS = (
    "car",
    "motorcycle",
    "bus",
    "truck",
    "van",
    "vehicle",
)

def _mobile_export_normalize_class_token(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

def _mobile_export_split_class_labels(value) -> list[str]:
    if value is None:
        return []
    raw_items = value
    if isinstance(value, str):
        raw_items = re.split(r"[,;\n]+", value)
    elif not isinstance(value, (list, tuple, set)):
        raw_items = [value]
    labels: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        key = _mobile_export_normalize_class_token(text)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(text)
    return labels

def _mobile_export_labels_from_model_value(value) -> list[str]:
    if isinstance(value, dict):
        indexed: list[tuple[int, str]] = []
        for raw_key, raw_label in value.items():
            try:
                index = int(raw_key)
            except Exception:
                continue
            label = str(raw_label or "").strip()
            if label:
                indexed.append((index, label))
        if indexed:
            return [label for _index, label in sorted(indexed, key=lambda item: item[0])]
        return _mobile_export_split_class_labels(list(value.values()))
    return _mobile_export_split_class_labels(value)

def _mobile_export_candidate_class_labels(candidate: dict | None) -> list[str]:
    if not isinstance(candidate, dict):
        return []
    metadata = candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {}
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), dict) else {}
    model_payload = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    payloads = [
        candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {},
        model_payload.get("info") if isinstance(model_payload.get("info"), dict) else {},
        raw.get("info") if isinstance(raw.get("info"), dict) else {},
        candidate,
    ]
    labels: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ("labels", "classes", "class_names", "names"):
            extracted = _mobile_export_labels_from_model_value(payload.get(key))
            for label in extracted:
                normalized = _mobile_export_normalize_class_token(label)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    labels.append(label)
        if labels:
            break
    return labels

def _mobile_export_default_vehicle_class_labels(candidate: dict | None) -> list[str]:
    known_labels = _mobile_export_candidate_class_labels(candidate)
    allowed = {
        _mobile_export_normalize_class_token(label)
        for label in _MOBILE_EXPORT_DEFAULT_VEHICLE_CLASS_LABELS
    }
    try:
        allowed.update(
            _mobile_export_normalize_class_token(label)
            for label in getattr(CONFIG, "VEHICLE_LABELS", set())
        )
    except Exception:
        pass
    matched = [
        label
        for label in known_labels
        if _mobile_export_normalize_class_token(label) in allowed
    ]
    if matched:
        return matched
    if known_labels and len(known_labels) <= 3:
        return known_labels
    defaults = list(_MOBILE_EXPORT_DEFAULT_VEHICLE_CLASS_LABELS)
    try:
        for label in sorted(getattr(CONFIG, "VEHICLE_LABELS", set())):
            normalized = _mobile_export_normalize_class_token(label)
            if normalized and all(_mobile_export_normalize_class_token(item) != normalized for item in defaults):
                defaults.append(str(label))
    except Exception:
        pass
    return defaults

def _mobile_export_vehicle_class_payload(candidate: dict | None, raw_value=None) -> dict:
    include_labels = _mobile_export_split_class_labels(raw_value)
    if not include_labels:
        include_labels = _mobile_export_default_vehicle_class_labels(candidate)
    known_labels = _mobile_export_candidate_class_labels(candidate)
    allowed = {_mobile_export_normalize_class_token(label) for label in include_labels}
    explicit_indices = {
        int(label)
        for label in include_labels
        if str(label).strip().isdigit()
    }
    matched_indices = {
        index
        for index, label in enumerate(known_labels)
        if _mobile_export_normalize_class_token(label) in allowed
    }
    valid_indices = sorted(
        index
        for index in (matched_indices | explicit_indices)
        if index >= 0 and (not known_labels or index < len(known_labels))
    )
    return {
        "filter_mode": "include",
        "include_labels": include_labels,
        "include_class_indices": valid_indices,
        "fallback_coco_class_indices": list(_MOBILE_EXPORT_COCO_VEHICLE_CLASS_IDS),
        "label_matching": "case_insensitive_normalized",
        "next_stage": "plate_detection",
        "available_labels": known_labels[:256],
        "available_label_count": len(known_labels),
    }

def _mobile_export_raw_metadata_candidates(model_path: Path) -> list[Path]:
    safe_path = Path(model_path)
    suffix = str(safe_path.suffix or "").strip()
    candidates = []
    if suffix:
        candidates.append(safe_path.with_suffix(f"{suffix}.metadata.json"))
    candidates.extend(
        (
            safe_path.with_suffix(".metadata.json"),
            safe_path.with_suffix(".json"),
            safe_path.with_name(f"{safe_path.stem}_metadata.json"),
            safe_path.with_name("model_metadata.json"),
            safe_path.with_name("metadata.json"),
        )
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique

def _mobile_export_read_raw_metadata(model_path: Path) -> dict:
    safe_path = Path(model_path)
    expected_name = str(safe_path.name or "").strip()
    for metadata_path in _mobile_export_raw_metadata_candidates(safe_path):
        if not metadata_path.exists() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        file_name = str(model_payload.get("file_name") or model_payload.get("name") or "").strip()
        model_path_text = str(model_payload.get("path") or "").strip()
        if file_name and expected_name and file_name != expected_name:
            continue
        if model_path_text:
            try:
                if Path(model_path_text).name != expected_name:
                    continue
            except Exception:
                pass
        payload["_metadata_json"] = str(metadata_path)
        return payload
    return {}

def _mobile_export_read_model_metadata(model_path: Path) -> dict:
    raw_payload = _mobile_export_read_raw_metadata(model_path)
    normalized_info: dict = {}
    validation_ok = None
    validation_message = ""
    try:
        sidecar_result = read_model_metadata_sidecar(Path(model_path))
    except Exception:
        sidecar_result = None
    if isinstance(sidecar_result, tuple) and len(sidecar_result) >= 3:
        validation_ok = bool(sidecar_result[0])
        validation_message = str(sidecar_result[1] or "")
        if isinstance(sidecar_result[2], dict):
            normalized_info = dict(sidecar_result[2] or {})

    if not normalized_info and isinstance(raw_payload.get("model"), dict):
        model_payload = raw_payload.get("model") or {}
        if isinstance(model_payload.get("info"), dict):
            normalized_info = dict(model_payload.get("info") or {})
        validation_ok = bool(model_payload.get("validation_ok", True)) if validation_ok is None else validation_ok
        validation_message = str(model_payload.get("validation_message") or validation_message)

    return {
        "raw": raw_payload,
        "info": normalized_info,
        "validation_ok": validation_ok,
        "validation_message": validation_message,
        "metadata_json": str(raw_payload.get("_metadata_json") or normalized_info.get("metadata_json") or ""),
    }

def _mobile_export_nested_value(payload: dict, *paths: str):
    for path in paths:
        cursor = payload
        ok = True
        for key in str(path or "").split("."):
            if not isinstance(cursor, dict) or key not in cursor:
                ok = False
                break
            cursor = cursor.get(key)
        if ok and cursor not in (None, ""):
            return cursor
    return None

def _mobile_export_metric_from_filename(path_like) -> float | None:
    try:
        stem = Path(path_like).stem.lower()
    except Exception:
        stem = str(path_like or "").lower()
    match = re.search(r"(?:^|[_-])map(\d{2,3})(?:$|[_-])", stem)
    if not match:
        return None
    try:
        value = int(match.group(1)) / 100.0
        return max(0.0, min(1.0, value))
    except Exception:
        return None

def _mobile_export_float_or_none(value) -> float | None:
    try:
        text = str(value if value is not None else "").strip().replace(",", ".")
        if not text:
            return None
        parsed = float(text)
        if parsed > 1.0 and parsed <= 100.0:
            parsed = parsed / 100.0
        return parsed
    except Exception:
        return None

def _mobile_export_file_datetime(path_like) -> str:
    try:
        return datetime.datetime.fromtimestamp(Path(path_like).stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return ""

def _mobile_export_file_size_mb(path_like) -> float | None:
    try:
        return round(Path(path_like).stat().st_size / (1024 * 1024), 2)
    except Exception:
        return None

def _mobile_export_filename_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def _mobile_export_marker_from_target(target: str) -> str:
    normalized = CONFIG.normalize_task_target(target)
    if normalized == "vehicle":
        return "MP"
    if normalized == "plate":
        return "MT"
    return "MZ"

def _mobile_export_size_filename_part(size_mb: float | None) -> str:
    if size_mb is None:
        return "razem-nieznanyMB"
    try:
        value = max(0.0, float(size_mb))
    except Exception:
        return "razem-nieznanyMB"
    return f"razem-{value:.1f}MB".replace(".", "p")

def _mobile_export_readable_package_filename(markers: list[str] | tuple[str, ...], total_size_mb: float | None) -> str:
    clean_markers: list[str] = []
    for marker in markers:
        text = re.sub(r"[^A-Za-z0-9]+", "", str(marker or "").strip().upper())
        if text and text not in clean_markers:
            clean_markers.append(text)
    marker_part = "-".join(clean_markers) if clean_markers else "MODEL"
    size_part = _mobile_export_size_filename_part(total_size_mb)
    return f"ALPR_{_mobile_export_filename_timestamp()}_{marker_part}_{size_part}.alprmodel"

def _mobile_export_candidate_size_mb(candidate: dict | None) -> float | None:
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("file_size_mb")
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    return _mobile_export_file_size_mb(candidate.get("best_weights"))

def _mobile_export_candidates_total_size_mb(candidates: list[dict] | tuple[dict, ...]) -> float | None:
    total = 0.0
    count = 0
    for candidate in candidates or ():
        size_mb = _mobile_export_candidate_size_mb(candidate)
        if size_mb is None:
            continue
        total += float(size_mb)
        count += 1
    return round(total, 1) if count else None

def _mobile_export_task_label(target: str, info: dict | None = None, task_hint: str = "") -> str:
    raw_task = str(task_hint or "").strip().lower()
    if isinstance(info, dict):
        raw_task = raw_task or str(info.get("task") or info.get("type") or info.get("model_task") or "").strip().lower()
    if raw_task == "pose" or CONFIG.normalize_task_target(target) == "plate":
        return "Pose"
    if raw_task == "detect" or CONFIG.normalize_task_target(target) in {"char", "vehicle"}:
        return "Detect"
    return ""

def _mobile_export_yolo_size_label(size: str) -> str:
    return {
        "n": "nano",
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "xlarge",
    }.get(str(size or "").strip().lower(), "")

_MOBILE_EXPORT_RUN_SNAPSHOT_INDEX: dict[str, dict] | None = None
_MOBILE_EXPORT_RUN_SNAPSHOT_SIGNATURE: tuple | None = None

def _mobile_export_run_id_from_text(text) -> str:
    match = re.search(r"(20\d{6}_\d{6})", str(text or ""))
    return str(match.group(1) or "").strip() if match else ""

def _mobile_export_run_snapshot_index() -> dict[str, dict]:
    global _MOBILE_EXPORT_RUN_SNAPSHOT_INDEX, _MOBILE_EXPORT_RUN_SNAPSHOT_SIGNATURE
    files = [Path(CONFIG.DIR_5_RUNS) / "training_history.json"]
    files.extend(Path(CONFIG.get_training_runs_dir(target)) / "training_history.json" for target in ("plate", "char", "vehicle"))
    files.extend(Path(CONFIG.DIR_9_PROJECTS).glob("*/5_training_runs/training_history.json"))
    signature = tuple(_mobile_export_file_signature(path) for path in sorted(set(files)))
    if _MOBILE_EXPORT_RUN_SNAPSHOT_INDEX is not None and _MOBILE_EXPORT_RUN_SNAPSHOT_SIGNATURE == signature:
        return _MOBILE_EXPORT_RUN_SNAPSHOT_INDEX

    index: dict[str, dict] = {}

    def read_history(path: Path) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except Exception:
            return
        runs = payload.get("runs") if isinstance(payload, dict) else {}
        if not isinstance(runs, dict):
            return
        for run_id, run_payload in runs.items():
            if not isinstance(run_payload, dict):
                continue
            key = str(run_payload.get("id") or run_id or "").strip()
            if not key:
                continue
            snapshot = dict(run_payload)
            snapshot.setdefault("id", key)
            snapshot["_history_dir"] = str(path.parent)
            if key in index and index[key] != snapshot:
                index[f"{path.parent}|{key}"] = snapshot
            else:
                index[key] = snapshot

    try:
        runs_root = Path(CONFIG.DIR_5_RUNS)
        direct_history_files = [runs_root / "training_history.json"]
        for target in ("plate", "char", "vehicle"):
            try:
                direct_history_files.append(Path(CONFIG.get_training_runs_dir(target)) / "training_history.json")
            except Exception:
                pass
        seen_history_files: set[str] = set()
        for path in direct_history_files:
            key = _mobile_export_safe_path_key(path)
            if not key or key in seen_history_files:
                continue
            seen_history_files.add(key)
            read_history(path)
    except Exception:
        pass
    try:
        for project_dir in Path(CONFIG.DIR_9_PROJECTS).iterdir():
            read_history(project_dir / "5_training_runs" / "training_history.json")
    except Exception:
        pass

    _MOBILE_EXPORT_RUN_SNAPSHOT_INDEX = index
    _MOBILE_EXPORT_RUN_SNAPSHOT_SIGNATURE = signature
    return index

def _mobile_export_run_snapshot_for_reference(reference) -> dict:
    path = Path(str(reference or ""))
    digest = _file_sha256(path)
    if not digest:
        return {}
    matches = []
    for snapshot in _mobile_export_run_snapshot_index().values():
        output = snapshot.get("output_checkpoint_snapshot") or {}
        expected = str(output.get("best_checkpoint_sha256") or (output.get("best") or {}).get("sha256") or "")
        if not expected:
            expected = _file_sha256(Path(str(snapshot.get("best_weights") or "")))
        if expected and expected == digest:
            matches.append(snapshot)
    identities = {(item.get("_history_dir"), item.get("id")) for item in matches}
    return dict(matches[0]) if len(identities) == 1 else {}

def _mobile_export_architecture_from_run_reference(reference, target: str = "", task_hint: str = "", visited: set[str] | None = None) -> str:
    run_id = _mobile_export_run_id_from_text(reference)
    if not run_id:
        return ""
    seen = set(visited or set())
    if run_id in seen:
        return ""
    seen.add(run_id)
    snapshot = _mobile_export_run_snapshot_index().get(run_id)
    if not isinstance(snapshot, dict):
        return ""
    for key in ("base_model", "model_file", "name"):
        parsed = _mobile_export_architecture_from_text(snapshot.get(key), target, task_hint)
        if parsed:
            return parsed
    for key in ("base_model", "model_file", "best_weights", "last_weights"):
        parsed = _mobile_export_architecture_from_run_reference(snapshot.get(key), target, task_hint, seen)
        if parsed:
            return parsed
    return ""

def _mobile_export_architecture_from_text(text, target: str = "", task_hint: str = "") -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    try:
        raw_text = Path(raw_text).name
    except Exception:
        pass
    match = re.search(r"yolo(?:v)?(8|11|26)([nsmxl])(?:[-_. ]?(pose|detect))?", raw_text.lower())
    if not match:
        return ""
    version = str(match.group(1) or "").strip()
    size = str(match.group(2) or "").strip().lower()
    task_token = str(match.group(3) or "").strip().lower()
    family = f"YOLOv{version}" if version == "8" else f"YOLO{version}"
    variant = f"{family}{size}"
    size_label = _mobile_export_yolo_size_label(size)
    task_label = _mobile_export_task_label(target, task_hint=task_token or task_hint)
    if size_label:
        variant = f"{variant} ({size_label})"
    return " ".join(part for part in (variant, task_label) if part).strip()

def _mobile_export_model_version_label(info: dict, target: str, fallback: str = "", *sources) -> str:
    task_label = _mobile_export_task_label(target, info)
    if isinstance(info, dict):
        for key in ("architecture_label", "source_architecture_label", "yolo_variant", "yaml_file"):
            parsed = _mobile_export_architecture_from_text(info.get(key), target, task_label)
            if parsed:
                return parsed

        family = str(info.get("yolo_family") or "").strip()
        version = str(info.get("yolo_version") or "").strip()
        size = str(info.get("yolo_size") or info.get("model_scale") or "").strip().lower()
        if not family and version:
            family = f"YOLOv{version}" if version == "8" else f"YOLO{version}"
        if family and size:
            size_label = _mobile_export_yolo_size_label(size)
            variant = f"{family}{size}"
            if size_label:
                variant = f"{variant} ({size_label})"
            return " ".join(part for part in (variant, task_label) if part).strip()

        for key in ("source_model_name", "source_model", "file_name"):
            parsed = _mobile_export_architecture_from_text(info.get(key), target, task_label)
            if parsed:
                return parsed

    for source in (*sources, fallback):
        parsed = _mobile_export_architecture_from_text(source, target, task_label)
        if parsed:
            return parsed
        parsed = _mobile_export_architecture_from_run_reference(source, target, task_label)
        if parsed:
            return parsed

    return " ".join(part for part in ("Nieznana architektura", task_label) if part).strip()

def _mobile_export_compact_yolo_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    match = re.search(r"yolo(?:v)?(8|11|26)\s*([nsmxl])", text.lower())
    if match:
        version = str(match.group(1) or "").strip()
        size = str(match.group(2) or "").strip().lower()
        family = f"YOLOv{version}" if version == "8" else f"YOLO{version}"
        return f"{family}{size}"
    compact = re.sub(r"\s*\([^)]*\)", "", text)
    compact = re.sub(r"\s+(pose|detect)\s*$", "", compact, flags=re.IGNORECASE).strip()
    return compact or text

def _mobile_export_parse_params_millions(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number <= 0:
            return None
        return number / 1_000_000.0 if number > 100_000 else number
    text = str(value or "").strip().lower().replace(",", ".")
    if not text:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        number = float(match.group(1))
    except Exception:
        return None
    if number <= 0:
        return None
    if "b" in text and "mb" not in text:
        return number * 1000.0
    if "k" in text:
        return number / 1000.0
    if "m" in text or "mln" in text or "million" in text:
        return number
    return number / 1_000_000.0 if number > 100_000 else number

def _mobile_export_scan_param_payload(payload, *, depth: int = 0) -> float | None:
    if depth > 4:
        return None
    if isinstance(payload, dict):
        preferred = {
            "parameter_count",
            "param_count",
            "num_parameters",
            "n_parameters",
            "n_params",
            "parameters",
            "total_parameters",
            "model_parameters",
            "params_m",
            "parameters_millions",
            "params_text",
            "params",
        }
        for key, value in payload.items():
            key_text = str(key or "").strip().lower()
            if key_text not in preferred:
                continue
            if isinstance(value, (dict, list, tuple, set)):
                continue
            parsed = _mobile_export_parse_params_millions(value)
            if parsed is not None:
                return parsed
        for value in payload.values():
            if isinstance(value, (dict, list, tuple)):
                parsed = _mobile_export_scan_param_payload(value, depth=depth + 1)
                if parsed is not None:
                    return parsed
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            parsed = _mobile_export_scan_param_payload(value, depth=depth + 1)
            if parsed is not None:
                return parsed
    return None

def _mobile_export_catalog_params_millions(candidate: dict | None) -> float | None:
    if not isinstance(candidate, dict):
        return None
    target = CONFIG.normalize_task_target(str(candidate.get("target") or ""))
    catalog = AVAILABLE_POSE_MODELS if target == "plate" else AVAILABLE_DETECT_MODELS
    if not catalog:
        return None
    info = candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {}
    texts = [
        candidate.get("model_version"),
        candidate.get("base_model"),
        candidate.get("best_weights"),
        info.get("source_model_name"),
        info.get("source_model"),
        info.get("architecture_label"),
        info.get("source_architecture_label"),
        info.get("yolo_variant"),
        info.get("yaml_file"),
    ]
    joined = " ".join(str(item or "").lower() for item in texts if item)
    try:
        joined = f"{joined} {' '.join(Path(str(item)).stem.lower() for item in texts if item)}"
    except Exception:
        pass
    yolo_match = re.search(r"yolo(?:v)?(8|11|26)\s*([nsmxl])", joined)
    if yolo_match:
        version = str(yolo_match.group(1) or "").strip()
        size = str(yolo_match.group(2) or "").strip().lower()
        prefix = f"yolov{version}{size}" if version == "8" else f"yolo{version}{size}"
        normalized_key = f"{prefix}-pose" if target == "plate" else prefix
        meta = catalog.get(normalized_key)
        if isinstance(meta, dict):
            parsed = _mobile_export_parse_params_millions(meta.get("params"))
            if parsed is not None:
                return parsed
    for key, meta in catalog.items():
        key_text = str(key or "").strip().lower()
        file_text = str((meta or {}).get("file") or "").strip().lower()
        try:
            file_stem = Path(file_text).stem.lower()
        except Exception:
            file_stem = file_text
        if key_text and key_text in joined:
            return _mobile_export_parse_params_millions((meta or {}).get("params"))
        if file_stem and file_stem in joined:
            return _mobile_export_parse_params_millions((meta or {}).get("params"))
    return None

def _mobile_export_candidate_params_millions(candidate: dict | None) -> tuple[float | None, str]:
    if not isinstance(candidate, dict):
        return None, ""
    metadata = candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {}
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), dict) else {}
    info = candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {}
    payloads = [
        candidate,
        info,
        metadata,
        raw.get("model") if isinstance(raw.get("model"), dict) else {},
        raw.get("training") if isinstance(raw.get("training"), dict) else {},
        raw.get("run_snapshot") if isinstance(raw.get("run_snapshot"), dict) else {},
        candidate.get("history_snapshot") if isinstance(candidate.get("history_snapshot"), dict) else {},
    ]
    run = candidate.get("run")
    if run is not None:
        try:
            if hasattr(run, "to_dict"):
                payloads.append(run.to_dict())
        except Exception:
            pass
    for payload in payloads:
        parsed = _mobile_export_scan_param_payload(payload)
        if parsed is not None:
            return parsed, "metadata"
    catalog_value = _mobile_export_catalog_params_millions(candidate)
    if catalog_value is not None:
        return catalog_value, "katalog"
    return None, ""

def _format_mobile_export_params(value: float | None, *, source: str = "", include_source: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return "-"
    if number <= 0:
        return "-"
    text = f"{number:.2f}" if number < 1.0 else f"{number:.1f}"
    text = text.rstrip("0").rstrip(".")
    suffix = f" ({source})" if include_source and source else ""
    return f"{text} mln{suffix}"

def _mobile_export_external_pose_filename(raw_url: str, display_name: str = "") -> str:
    parsed_name = ""
    try:
        parsed = urllib.parse.urlparse(str(raw_url or "").strip())
        parsed_name = Path(urllib.parse.unquote(parsed.path or "")).name
    except Exception:
        parsed_name = ""
    candidate_name = str(display_name or parsed_name or "external_yolo_pose").strip()
    try:
        candidate_name = Path(candidate_name).name
    except Exception:
        pass
    candidate_name = re.sub(r"\.pt$", "", candidate_name, flags=re.IGNORECASE)
    candidate_name = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_name).strip("._-")
    if not candidate_name:
        candidate_name = "external_yolo_pose"
    if len(candidate_name) > 72:
        candidate_name = candidate_name[:72].strip("._-") or "external_yolo_pose"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{candidate_name}_{timestamp}.pt"

def _mobile_export_pose_validation_message(info: dict | None) -> tuple[bool, str]:
    safe_info = info if isinstance(info, dict) else {}
    task = str(safe_info.get("task") or safe_info.get("type") or "").strip().lower()
    has_keypoints = bool(safe_info.get("keypoints"))
    kpt_shape = safe_info.get("kpt_shape")
    kpt_count = None
    try:
        if isinstance(kpt_shape, (list, tuple)) and kpt_shape:
            kpt_count = int(kpt_shape[0])
    except Exception:
        kpt_count = None
    if task != "pose" and not has_keypoints:
        return False, "Pobrany plik jest modelem YOLO, ale nie wygląda na model pose."
    if kpt_count is not None and kpt_count < 4:
        return False, f"Model pose ma za mało punktów kluczowych ({kpt_count}). Do MT potrzebujemy co najmniej 4 narożników tablicy."
    kpt_label = str(kpt_count) if kpt_count is not None else "nie ustalono"
    return (
        True,
        "Model technicznie pasuje do roli MT (YOLO pose). "
        f"Liczba punktów kluczowych: {kpt_label}. Upewnij się, że punkty opisują narożniki tablicy.",
    )

def _mobile_export_ultralytics_detect_catalog(*, include_runtime_assets: bool = True) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for key, raw_info in dict(AVAILABLE_DETECT_MODELS or {}).items():
        model_key = str(key or "").strip()
        if not model_key:
            continue
        info = dict(raw_info or {})
        info.setdefault("file", f"{model_key}.pt")
        info.setdefault("name", model_key)
        info.setdefault("source", "config")
        catalog[model_key] = info
    if not include_runtime_assets:
        return catalog
    try:
        from ultralytics.utils import downloads as ultralytics_downloads  # type: ignore

        assets = getattr(ultralytics_downloads, "GITHUB_ASSETS_NAMES", None) or []
        task_suffixes = ("-pose", "-seg", "-cls", "-obb", "-world")
        for asset_name in sorted(str(item or "").strip() for item in assets):
            lower_asset = asset_name.lower()
            if not lower_asset.endswith(".pt"):
                continue
            model_key = Path(asset_name).stem
            lower_key = model_key.lower()
            if any(lower_key.endswith(suffix) for suffix in task_suffixes):
                continue
            if not re.match(r"^yolo(?:v?\d+)?[nslmx]$", lower_key):
                continue
            info = catalog.setdefault(
                model_key,
                {
                    "name": model_key,
                    "file": asset_name,
                    "params": "",
                    "speed": "",
                    "version": "",
                    "description": "Model detect z listy assetow Ultralytics",
                },
            )
            info.setdefault("file", asset_name)
            info["source"] = "ultralytics"
    except Exception:
        pass
    return catalog

def _mobile_export_ultralytics_catalog_for_target(target: str) -> dict[str, dict]:
    # Katalogowy importer w centrum eksportu obsluguje tylko MP. MT powstaje
    # w torze projektu/treningu i nie jest tu traktowany jako asset Ultralytics.
    return _mobile_export_ultralytics_detect_catalog(include_runtime_assets=True)

def _mobile_export_catalog_validation_message(info: dict | None, target: str) -> tuple[bool, str]:
    normalized_target = CONFIG.normalize_task_target(target)
    if normalized_target == "vehicle":
        safe_info = info if isinstance(info, dict) else {}
        task = str(safe_info.get("task") or safe_info.get("type") or "").strip().lower()
        if task and task != "detect":
            return False, f"Model MP musi byc modelem YOLO detect, a wykryto zadanie: {task}."
        labels = _mobile_export_labels_from_model_value(
            safe_info.get("labels")
            or safe_info.get("classes")
            or safe_info.get("class_names")
            or safe_info.get("names")
        )
        vehicle_labels = _mobile_export_default_vehicle_class_labels({"model_info": safe_info})
        matched = []
        allowed = {_mobile_export_normalize_class_token(label) for label in vehicle_labels}
        for label in labels:
            if _mobile_export_normalize_class_token(label) in allowed:
                matched.append(label)
        if matched:
            return (
                True,
                "Model technicznie pasuje do roli MP (YOLO detect). "
                f"Rozpoznane klasy pojazdow: {', '.join(matched[:6])}.",
            )
        return (
            True,
            "Model technicznie pasuje do roli MP (YOLO detect). "
            "Nie znaleziono jawnej listy klas pojazdow, dlatego filtr klas zostanie zapisany w ustawieniach eksportu.",
        )
    return False, "Katalogowy importer Ultralytics w centrum eksportu obsluguje tylko MP (YOLO detect)."

def _mobile_export_catalog_label(
    model_key: str,
    info: dict | None,
    local_path: Path | None = None,
    *,
    include_local_state: bool = True,
) -> str:
    safe_info = info if isinstance(info, dict) else {}
    name = str(safe_info.get("name") or model_key or "-").strip()
    params = str(safe_info.get("params") or "").strip()
    pieces = [str(model_key or name)]
    if name and name != model_key:
        pieces.append(name)
    if params:
        pieces.append(params)
    label = " | ".join(pieces)
    if include_local_state:
        local_suffix = " | lokalny" if local_path and Path(local_path).exists() else " | do pobrania"
        label += local_suffix
    return label

def _mobile_export_find_local_catalog_model(file_name: str, *, target: str = "plate") -> Path | None:
    safe_file = Path(str(file_name or "").strip()).name
    if not safe_file:
        return None
    roots: list[Path] = []
    try:
        roots.append(Path(CONFIG.get_base_models_dir(target)))
    except Exception:
        pass
    for value in (
        getattr(CONFIG, "DIR_6_MODELS_BASE_POSE", None) if CONFIG.normalize_task_target(target) == "plate" else None,
        getattr(CONFIG, "DIR_6_MODELS_BASE_DETECT", None) if CONFIG.normalize_task_target(target) == "vehicle" else None,
        getattr(CONFIG, "DIR_6_MODELS", None),
    ):
        if value:
            try:
                roots.append(Path(value))
            except Exception:
                pass
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    seen: set[str] = set()
    for root in roots:
        try:
            root_key = str(root.resolve()).lower()
        except Exception:
            root_key = str(root).lower()
        if not root_key or root_key in seen:
            continue
        seen.add(root_key)
        direct = root / safe_file
        if direct.exists() and direct.is_file():
            return direct
        if root.name == safe_file and root.exists() and root.is_file():
            return root
        if root.exists() and root.is_dir() and str(root).lower().endswith(("6_models", "pose", "detect")):
            try:
                for candidate in root.rglob(safe_file):
                    if candidate.exists() and candidate.is_file():
                        return candidate
            except Exception:
                pass
    return None

def _mobile_export_local_catalog_model_index(*, target: str = "vehicle") -> dict[str, Path]:
    normalized_target = CONFIG.normalize_task_target(target)
    roots: list[Path] = []
    try:
        roots.append(Path(CONFIG.get_base_models_dir(normalized_target)))
    except Exception:
        pass
    for value in (
        getattr(CONFIG, "DIR_6_MODELS_BASE_POSE", None) if normalized_target == "plate" else None,
        getattr(CONFIG, "DIR_6_MODELS_BASE_DETECT", None) if normalized_target == "vehicle" else None,
    ):
        if value:
            try:
                roots.append(Path(value))
            except Exception:
                pass

    index: dict[str, Path] = {}
    seen_roots: set[str] = set()
    for root in roots:
        root_key = _mobile_export_safe_path_key(root)
        if not root_key or root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        try:
            if root.exists() and root.is_file() and root.suffix.lower() == ".pt":
                index.setdefault(root.name.lower(), root)
                continue
            if not root.exists() or not root.is_dir():
                continue
            for model_path in root.rglob("*.pt"):
                index.setdefault(model_path.name.lower(), model_path)
        except Exception:
            continue
    return index

def _mobile_export_catalog_import_destination(file_name: str, *, target: str = "plate") -> Path:
    safe_file = Path(str(file_name or "").strip()).name or "yolo_pose.pt"
    if not safe_file.lower().endswith(".pt"):
        safe_file = f"{safe_file}.pt"
    return Path(CONFIG.get_base_models_dir(target)) / "ultralytics" / safe_file

def _mobile_export_artifact_sources(self) -> list[tuple[Path, str, str, str, str]]:
    sources: list[tuple[Path, str, str, str, str]] = []
    seen: set[str] = set()

    def add_source(
        path_value,
        target: str,
        label: str,
        *,
        project_name: str = "",
        project_root: str = "",
    ) -> None:
        if not path_value:
            return
        try:
            path = Path(path_value)
        except Exception:
            return
        inferred_project_name, inferred_project_root = _mobile_export_project_identity_from_path(path)
        project_name = str(project_name or inferred_project_name or "").strip()
        project_root = str(project_root or inferred_project_root or "").strip()
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if not key or key in seen:
            return
        seen.add(key)
        sources.append((path, CONFIG.normalize_task_target(target), label, project_name, project_root))

    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        project_root = CAMPAIGN.get_active_project_root_dir() if project_name else None
        if project_root:
            project_models = Path(project_root) / "6_models" / "trained"
            add_source(project_models / "plates", "plate", f"Projekt {project_name}: modele tablic", project_name=project_name, project_root=str(project_root))
            add_source(project_models / "chars", "char", f"Projekt {project_name}: modele znaków", project_name=project_name, project_root=str(project_root))
            add_source(project_models / "vehicles", "vehicle", f"Projekt {project_name}: modele pojazdów", project_name=project_name, project_root=str(project_root))
    except Exception:
        pass

    try:
        projects_root = Path(CONFIG.DIR_9_PROJECTS)
        if projects_root.exists() and projects_root.is_dir():
            for project_root in sorted((path for path in projects_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
                project_name = project_root.name
                project_models = project_root / "6_models" / "trained"
                add_source(project_models / "plates", "plate", f"Projekt {project_name}: modele tablic", project_name=project_name, project_root=str(project_root))
                add_source(project_models / "chars", "char", f"Projekt {project_name}: modele znaków", project_name=project_name, project_root=str(project_root))
                add_source(project_models / "vehicles", "vehicle", f"Projekt {project_name}: modele pojazdów", project_name=project_name, project_root=str(project_root))
    except Exception:
        pass

    add_source(CONFIG.get_trained_models_dir("plate"), "plate", "Globalne modele tablic")
    add_source(CONFIG.get_trained_models_dir("char"), "char", "Globalne modele znaków")
    add_source(CONFIG.get_trained_models_dir("vehicle"), "vehicle", "Globalne modele pojazdów")
    add_source(getattr(CONFIG, "DIR_6_MODELS_BASE_POSE", None), "plate", "Importowane modele YOLO pose (MT)")
    add_source(getattr(CONFIG, "DIR_6_MODELS_BASE_DETECT", None), "vehicle", "Bazowe modele pojazdów COCO")
    add_source(getattr(CONFIG, "DIR_6_MODELS_PLATES", None), "plate", "Globalne modele tablic")
    add_source(getattr(CONFIG, "DIR_6_MODELS_CHARS", None), "char", "Globalne modele znaków")
    return sources

def _mobile_export_candidate_from_artifact(
    self,
    model_path: Path,
    target: str,
    scope_label: str,
    *,
    project_name: str = "",
    project_root: str = "",
) -> dict:
    safe_path = Path(model_path)
    fallback_target = CONFIG.normalize_task_target(target)
    inferred_project_name, inferred_project_root = _mobile_export_project_identity_from_path(safe_path)
    project_name = str(project_name or inferred_project_name or "").strip()
    project_root = str(project_root or inferred_project_root or "").strip()
    metadata = _mobile_export_read_model_metadata(safe_path)
    raw_payload = metadata.get("raw") if isinstance(metadata.get("raw"), dict) else {}
    if not project_name and isinstance(raw_payload, dict):
        project_payload = raw_payload.get("project") if isinstance(raw_payload.get("project"), dict) else {}
        source_payload = raw_payload.get("source") if isinstance(raw_payload.get("source"), dict) else {}
        project_name = str(project_payload.get("name") or source_payload.get("project_name") or "").strip()
        project_root = str(project_payload.get("root") or source_payload.get("project_root") or project_root or "").strip()
    info = metadata.get("info") if isinstance(metadata.get("info"), dict) else {}
    history_snapshot = _mobile_export_run_snapshot_for_reference(safe_path)
    if history_snapshot.get("_history_dir"):
        history_project, history_root = _mobile_export_project_identity_from_path(history_snapshot["_history_dir"])
        project_name = history_project or project_name
        project_root = history_root or project_root

    metric_root = raw_payload.get("metrics") if isinstance(raw_payload.get("metrics"), dict) else {}
    extra_root = raw_payload.get("extra") if isinstance(raw_payload.get("extra"), dict) else {}
    extra_metrics = extra_root.get("metrics") if isinstance(extra_root.get("metrics"), dict) else {}
    training_root = raw_payload.get("training") if isinstance(raw_payload.get("training"), dict) else {}
    run_snapshot = history_snapshot or (raw_payload.get("run_snapshot") if isinstance(raw_payload.get("run_snapshot"), dict) else {})
    best_row = metric_root.get("best_row") if isinstance(metric_root.get("best_row"), dict) else {}
    latest = metric_root.get("latest") if isinstance(metric_root.get("latest"), dict) else {}

    filename_map = _mobile_export_metric_from_filename(safe_path)
    best_map50 = _mobile_export_float_or_none(
        _mobile_export_nested_value(
            {
                "metrics": metric_root,
                "extra_metrics": extra_metrics,
                "best_row": best_row,
                "latest": latest,
                "run": run_snapshot,
                "history": history_snapshot,
            },
            "metrics.best_map50",
            "extra_metrics.best_map50",
            "run.best_map50",
            "history.best_map50",
            "best_row.map50",
            "best_row.box_map50",
            "best_row.pose_map50",
            "best_row.metrics/mAP50",
            "best_row.metrics/mAP50(B)",
            "best_row.metrics/mAP50(P)",
            "latest.map50",
            "latest.metrics/mAP50",
        )
    )
    best_map50_95 = _mobile_export_float_or_none(
        _mobile_export_nested_value(
            {
                "metrics": metric_root,
                "extra_metrics": extra_metrics,
                "best_row": best_row,
                "latest": latest,
                "run": run_snapshot,
                "history": history_snapshot,
            },
            "metrics.best_map50_95",
            "extra_metrics.best_map50_95",
            "run.best_map50_95",
            "history.best_map50_95",
            "best_row.map50_95",
            "best_row.box_map50_95",
            "best_row.pose_map50_95",
            "best_row.metrics/mAP50-95",
            "best_row.metrics/mAP50-95(B)",
            "best_row.metrics/mAP50-95(P)",
            "latest.map50_95",
            "latest.metrics/mAP50-95",
        )
    )
    if best_map50 is None:
        best_map50 = filename_map

    dataset_path = str(
        _mobile_export_nested_value(
            {"training": training_root, "run": run_snapshot, "extra": extra_root},
            "training.dataset_path",
            "run.dataset_path",
            "extra.training.dataset_path",
        )
        or ""
    ).strip()
    if not dataset_path and isinstance(history_snapshot, dict):
        dataset_path = str(history_snapshot.get("dataset_path") or "").strip()
    target, target_source, target_conflict = _mobile_export_resolve_candidate_target(
        fallback_target=fallback_target,
        run_like=run_snapshot or history_snapshot or training_root,
        dataset_path=dataset_path,
        model_info=info,
    )
    if not target:
        target = fallback_target
        target_source = "katalog lub źródło listy"
    role = _mobile_role_from_training_target(target)
    task = _mobile_export_task_from_target_and_info(target, info)
    try:
        model_ref = build_model_display_ref(safe_path, target_hint=target)
        model_label = model_ref.id
    except Exception:
        model_label = safe_path.name or "model"
    created_at = str(
        _mobile_export_nested_value(
            {"raw": raw_payload, "training": training_root, "run": run_snapshot},
            "raw.created_at",
            "training.finished_at",
            "training.started_at",
            "run.finished_at",
            "run.started_at",
            "run.created_at",
        )
        or ""
    ).strip()
    if not created_at and isinstance(history_snapshot, dict):
        created_at = str(
            history_snapshot.get("finished_at")
            or history_snapshot.get("started_at")
            or history_snapshot.get("created_at")
            or ""
        ).strip()
    if not created_at:
        created_at = _mobile_export_file_datetime(safe_path)
    base_model = str(
        training_root.get("base_model")
        or run_snapshot.get("base_model")
        or history_snapshot.get("base_model")
        or info.get("source_model_name")
        or info.get("source_model")
        or ""
    )
    model_version = _mobile_export_model_version_label(
        info,
        target,
        safe_path.name,
        base_model,
        training_root.get("base_model"),
        run_snapshot.get("base_model"),
        history_snapshot.get("base_model"),
        training_root.get("name"),
        run_snapshot.get("name"),
        history_snapshot.get("name"),
    )
    total_epochs = _mobile_export_int_or_none(
        _mobile_export_nested_value(
            {"training": training_root, "run": run_snapshot, "history": history_snapshot},
            "training.total_epochs",
            "run.total_epochs",
            "history.total_epochs",
        )
    )
    if total_epochs is None or total_epochs <= 0:
        total_epochs = _mobile_export_int_or_none(
            _mobile_export_nested_value(
                {"training": training_root, "run": run_snapshot, "history": history_snapshot},
                "training.current_epoch",
                "run.current_epoch",
                "history.current_epoch",
            )
        )
    training_provenance = dict(training_root) if isinstance(training_root, dict) and training_root.get("provenance_version") else {}
    known_total = _mobile_export_provenance_known_total(training_provenance)
    if known_total is not None:
        total_epochs = known_total
    elif total_epochs is None or total_epochs <= 0:
        total_epochs = _mobile_export_provenance_display_total(training_provenance)
    dataset_provenance = training_provenance.get("dataset") if isinstance(training_provenance.get("dataset"), dict) else {}
    dataset_label = (
        str(dataset_provenance.get("dataset_id") or "").strip()
        or _mobile_export_dataset_label(dataset_path, target)
    )

    return {
        "history_dir": "",
        "scope": scope_label,
        "project_name": project_name,
        "project_root": project_root,
        "run": None,
        "target": target,
        "target_label": self._format_training_target_label(target),
        "target_source": target_source,
        "target_conflict": target_conflict,
        "role": role,
        "task": task,
        "best_weights": safe_path,
        "run_label": "gotowy model",
        "model_label": model_label,
        "model_version": model_version,
        "dataset_label": dataset_label,
        "dataset_path": dataset_path,
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "best_epoch": build_checkpoint_metric_summary(run_snapshot, checkpoint=safe_path, verify_checkpoint=False)["best_epoch"],
        "created_at": created_at,
        "started_at": str(training_root.get("started_at") or run_snapshot.get("started_at") or history_snapshot.get("started_at") or ""),
        "finished_at": str(training_root.get("finished_at") or run_snapshot.get("finished_at") or history_snapshot.get("finished_at") or created_at),
        "img_size": _mobile_export_nested_value({"training": training_root, "run": run_snapshot, "history": history_snapshot}, "training.img_size", "run.img_size", "history.img_size"),
        "epochs": _mobile_export_nested_value({"training": training_root, "run": run_snapshot, "history": history_snapshot}, "training.epochs", "run.epochs", "history.epochs"),
        "current_epoch": _mobile_export_nested_value({"training": training_root, "run": run_snapshot, "history": history_snapshot}, "training.current_epoch", "run.current_epoch", "history.current_epoch"),
        "total_epochs": total_epochs,
        "total_epochs_known": _mobile_export_bool_or_none(training_provenance.get("total_epochs_known")) is True if training_provenance else False,
        "known_epochs_minimum": _mobile_export_int_or_none(training_provenance.get("known_epochs_minimum")) if training_provenance else None,
        "provenance_status": str(training_provenance.get("provenance_status") or ("complete" if known_total else "legacy_unknown")),
        "training_provenance": training_provenance,
        "batch_size": _mobile_export_nested_value({"training": training_root, "run": run_snapshot, "history": history_snapshot}, "training.batch_size", "run.batch_size", "history.batch_size"),
        "base_model": base_model,
        "history_snapshot": history_snapshot,
        "model_info": info,
        "model_metadata": metadata,
        "file_size_mb": _mobile_export_file_size_mb(safe_path),
    }

def _build_mobile_model_export_path(self, run, target: str, source_path: Path) -> Path:
    marker = _mobile_export_marker_from_target(target)
    target_dir = Path(CONFIG.get_mobile_model_packages_dir(target))
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / _mobile_export_readable_package_filename(
        [marker],
        _mobile_export_file_size_mb(source_path),
    )

def _build_mobile_alpr_package_export_path(
    self,
    plate_candidate: dict,
    char_candidate: dict,
    *,
    vehicle_candidate: dict | None = None,
) -> Path:
    target_dir = Path(getattr(CONFIG, "DIR_6_MODELS_MOBILE_PACKAGES", CONFIG.DIR_6_MODELS / "mobile_packages")) / "alpr"
    target_dir.mkdir(parents=True, exist_ok=True)
    selected_candidates = []
    markers = []
    if vehicle_candidate:
        selected_candidates.append(vehicle_candidate)
        markers.append("MP")
    selected_candidates.extend([plate_candidate, char_candidate])
    markers.extend(["MT", "MZ"])
    return target_dir / _mobile_export_readable_package_filename(
        markers,
        _mobile_export_candidates_total_size_mb(selected_candidates),
    )

def _build_mobile_export_metadata(self, run, target: str, best_weights: Path) -> dict:
    metric_summary = self._build_history_run_metric_summary(run)
    model_metadata = _mobile_export_read_model_metadata(Path(best_weights))
    model_info = model_metadata.get("info") if isinstance(model_metadata.get("info"), dict) else {}
    raw_model_metadata = model_metadata.get("raw") if isinstance(model_metadata.get("raw"), dict) else {}
    project_name, project_root = _mobile_export_project_identity_from_path(best_weights)
    if run is None:
        filename_metric = _mobile_export_metric_from_filename(best_weights)
        if metric_summary.get("best_map50") is None:
            metric_summary["best_map50"] = filename_metric
        if metric_summary.get("best_map50_95") is None:
            metric_summary["best_map50_95"] = filename_metric
    try:
        run_snapshot = run.to_dict() if hasattr(run, "to_dict") else {}
    except Exception:
        run_snapshot = {}
    raw_training = raw_model_metadata.get("training") if isinstance(raw_model_metadata.get("training"), dict) else {}
    raw_run_snapshot = raw_model_metadata.get("run_snapshot") if isinstance(raw_model_metadata.get("run_snapshot"), dict) else {}
    history_snapshot = _mobile_export_run_snapshot_for_reference(best_weights)
    if not isinstance(run_snapshot, dict) or not run_snapshot:
        run_snapshot = dict(history_snapshot or raw_run_snapshot or {})
    provenance_source = run if run is not None else (run_snapshot or raw_training)
    dataset_hint = (
        str(getattr(run, "dataset_path", "") or "").strip()
        or str(raw_training.get("dataset_path") or "").strip()
        or str(run_snapshot.get("dataset_path") or "").strip()
        or str(history_snapshot.get("dataset_path") or "").strip()
    )
    if run is None and not run_snapshot and isinstance(raw_training, dict) and raw_training.get("provenance_version"):
        training_payload = dict(raw_training)
    else:
        training_payload = _mobile_export_build_training_provenance(
            provenance_source,
            target=target,
            checkpoint=best_weights,
            dataset_path=dataset_hint,
            model_metadata=model_metadata,
            include_dataset_fingerprint=True,
        )
        if not training_payload and isinstance(raw_training, dict):
            training_payload = dict(raw_training)
    metric_summary = build_checkpoint_metric_summary(provenance_source, checkpoint=best_weights)
    training_payload["best_epoch"] = metric_summary["best_epoch"]
    training_payload["best_epoch_source"] = metric_summary["best_epoch_source"]
    if not project_name and history_snapshot.get("_history_dir"):
        project_name, project_root = _mobile_export_project_identity_from_path(history_snapshot["_history_dir"])
    compatibility_defaults = {
        "run_id": str(getattr(run, "id", "") or raw_training.get("run_id") or run_snapshot.get("id") or ""),
        "run_name": str(getattr(run, "name", "") or raw_training.get("run_name") or raw_training.get("name") or run_snapshot.get("name") or ""),
        "target": target,
        "target_label": self._format_training_target_label(target),
        "dataset_path": dataset_hint,
        "base_model": str(getattr(run, "base_model", "") or raw_training.get("base_model") or run_snapshot.get("base_model") or ""),
        "img_size": getattr(run, "img_size", None) if run is not None else (raw_training.get("img_size") or run_snapshot.get("img_size")),
        "epochs": getattr(run, "epochs", None) if run is not None else (raw_training.get("epochs") or run_snapshot.get("epochs")),
        "current_epoch": getattr(run, "current_epoch", None) if run is not None else (raw_training.get("current_epoch") or run_snapshot.get("current_epoch")),
        "batch_size": getattr(run, "batch_size", None) if run is not None else (raw_training.get("batch_size") or run_snapshot.get("batch_size")),
        "started_at": str(getattr(run, "started_at", "") or raw_training.get("started_at") or run_snapshot.get("started_at") or ""),
        "finished_at": str(getattr(run, "finished_at", "") or raw_training.get("finished_at") or run_snapshot.get("finished_at") or ""),
    }
    for key, value in compatibility_defaults.items():
        if value not in (None, ""):
            training_payload.setdefault(key, value)
    if _mobile_export_bool_or_none(training_payload.get("total_epochs_known")) is False:
        training_payload["total_epochs"] = None
    known_total = _mobile_export_provenance_known_total(training_payload)
    if isinstance(run_snapshot, dict):
        if known_total is not None:
            run_snapshot["total_epochs"] = known_total
        else:
            minimum = _mobile_export_int_or_none(training_payload.get("known_epochs_minimum"))
            if minimum is not None and minimum > 0:
                run_snapshot["known_epochs_minimum"] = minimum
                run_snapshot["total_epochs_known"] = False
    return self._json_safe_training_value(
        {
            "training": training_payload,
            "metrics": {
                "best_map50": metric_summary.get("best_map50"),
                "best_map50_95": metric_summary.get("best_map50_95"),
                "best_epoch": metric_summary.get("best_epoch"),
                "latest": metric_summary.get("latest") or {},
                "best_row": metric_summary.get("best_row") or {},
            },
            "source": {
                "checkpoint": str(best_weights),
                "run_output_dir": str(getattr(run, "output_dir", "") or ""),
                "training_history_dir": str(getattr(getattr(self, "history", None), "history_dir", "") or ""),
                "model_metadata_json": str(model_metadata.get("metadata_json") or ""),
                "project_name": project_name,
                "project_root": project_root,
            },
            "project": {
                "name": project_name,
                "root": project_root,
            },
            "model": {
                "file_name": Path(best_weights).name,
                "path": str(best_weights),
                "info": model_info,
                "metadata": raw_model_metadata,
            },
            "run_snapshot": run_snapshot,
        }
    )

def _format_mobile_export_datetime(value) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        return datetime.datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return text.replace("T", " ")

def _format_mobile_export_metric(value) -> str:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return "-"
        return f"{float(text):.3f}"
    except Exception:
        return "-"

def _format_mobile_export_int(value) -> str:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return "-"
        return str(int(float(text)))
    except Exception:
        return "-"

def _format_mobile_export_bytes(value: int | float | None) -> str:
    try:
        size = float(value if value is not None else 0)
    except Exception:
        size = 0.0
    if size <= 0:
        return "-"
    units = ("B", "KB", "MB", "GB")
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{size:.1f} {units[index]}"

def _format_mobile_export_duration(seconds: float | int | None) -> str:
    try:
        total = max(0, int(round(float(seconds or 0))))
    except Exception:
        total = 0
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def _mobile_export_requirement_name(spec: str) -> str:
    text = str(spec or "").strip()
    text = text.split(";", 1)[0].strip()
    text = re.split(r"\s*(?:===|==|~=|!=|<=|>=|<|>)\s*", text, maxsplit=1)[0].strip()
    text = text.split("[", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", text)
    return str(match.group(1) or text).strip() if match else text

def _mobile_export_parse_requirements_file(path: Path) -> list[dict]:
    items: list[dict] = []
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return items
    for line in lines:
        spec = str(line or "").strip()
        if not spec or spec.startswith("#"):
            continue
        if spec.startswith(("-r ", "--requirement", "-c ", "--constraint", "--")):
            continue
        inline_comment = spec.find(" #")
        if inline_comment >= 0:
            spec = spec[:inline_comment].strip()
        name = _mobile_export_requirement_name(spec)
        if not name:
            continue
        items.append({"spec": spec, "name": name})
    return items

def _mobile_export_module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(str(name or "").strip()) is not None
    except Exception:
        return False

def _mobile_export_progress_value(percent: float | int | None) -> float:
    try:
        return max(0.0, min(100.0, float(percent if percent is not None else 0.0)))
    except Exception:
        return 0.0


def _mobile_export_progress_label(percent: float | int | None) -> str:
    if percent is None:
        return "..."
    return f"{_mobile_export_progress_value(percent):.0f}%"


def _mobile_export_verbose_progress_message(percent: float | int | None, message: str | None) -> str:
    raw = re.sub(r"\s+", " ", str(message or "").strip())
    lower = raw.lower()
    prefix = f"{_mobile_export_progress_label(percent)} | "
    model_label = ""
    detail = raw

    if lower.startswith("mt:"):
        model_label = "MT, model tablic: "
        detail = raw[3:].strip()
        lower = detail.lower()
    elif lower.startswith("mz:"):
        model_label = "MZ, model znaków: "
        detail = raw[3:].strip()
        lower = detail.lower()

    if not detail:
        return prefix + "Przygotowuję następny krok eksportu mobilnego."

    stage = detail
    if "przygot" in lower and "mt" in lower and "mz" in lower:
        stage = (
            "Przygotowuję kompletny pakiet ALPR. Eksporter zbuduje model mobilny MT, "
            "model mobilny MZ i opcjonalny MP, a potem połączy je wspólnym manifestem dla aplikacji mobilnej."
        )
    elif "wczyt" in lower and "checkpoint" in lower:
        stage = (
            f"{model_label}Wczytuję checkpoint best.pt i sprawdzam, czy rola modelu, liczba klas "
            "oraz kształt wyjścia są zgodne z kontraktem klienta mobilnego."
        )
    elif "wariant" in lower and ("eksport" in lower or "konwert" in lower):
        variant = re.sub(r"(?i)^eksportuj\S*\s+(wariant\s+)?", "", detail).strip(" .")
        stage = (
            f"{model_label}Konwertuję checkpoint do wariantu mobilnego"
            f"{f' ({variant})' if variant else ''}. Ten krok może potrwać dłużej przy TFLite/LiteRT, "
            "bo Ultralytics buduje graf pośredni i zapisuje pliki runtime."
        )
    elif "manifest" in lower or "sha" in lower:
        stage = (
            f"{model_label}Tworzę manifest i liczę sumy SHA-256. Dzięki temu klient mobilny "
            "może sprawdzić kompletność oraz integralność eksportowanych plików."
        )
    elif "pakuj" in lower or "archiw" in lower or "zestaw" in lower:
        stage = f"{model_label}Pakuję pliki modelu, manifest i metadane do końcowego archiwum .alprmodel."
    elif "sprawdz" in lower or "sprawdzam" in lower or "zip" in lower:
        stage = f"{model_label}Otwieram zbudowaną paczkę ponownie i waliduję jej zawartość przed zapisaniem wyniku."
    elif "zweryfikowany" in lower or "dolacz" in lower:
        stage = f"{model_label}Model mobilny został sprawdzony i dołączony do kompletnego zestawu ALPR."
    elif "gotow" in lower:
        stage = f"{model_label}Eksport zakończony. Gotowy plik można przekazać do klienta mobilnego."
    elif model_label:
        stage = f"{model_label}{detail}"

    if raw and stage != raw:
        return f"{prefix}{stage}\nSzczegół techniczny: {raw}"
    return prefix + stage

_MOBILE_EXPORT_DIST_SIZE_CACHE: dict[str, int | None] = {}

def _mobile_export_installed_distribution_size(name: str) -> int | None:
    normalized = re.sub(r"[-_.]+", "-", str(name or "").strip().lower())
    if not normalized:
        return None
    if normalized in _MOBILE_EXPORT_DIST_SIZE_CACHE:
        return _MOBILE_EXPORT_DIST_SIZE_CACHE[normalized]
    try:
        distribution = importlib_metadata.distribution(normalized)
    except Exception:
        try:
            distribution = importlib_metadata.distribution(str(name or "").strip())
        except Exception:
            _MOBILE_EXPORT_DIST_SIZE_CACHE[normalized] = None
            return None
    total = 0
    try:
        files = list(distribution.files or [])
    except Exception:
        files = []
    for relative in files:
        try:
            file_path = Path(distribution.locate_file(relative))
            if file_path.is_file():
                total += int(file_path.stat().st_size or 0)
        except Exception:
            continue
    result = total if total > 0 else None
    _MOBILE_EXPORT_DIST_SIZE_CACHE[normalized] = result
    return result

def _mobile_export_size_to_bytes(value, unit: str) -> int | None:
    try:
        number = float(value)
    except Exception:
        return None
    normalized = str(unit or "").strip().lower()
    if normalized in {"gb", "gib"}:
        number *= 1024.0 * 1024.0 * 1024.0
    elif normalized in {"mb", "mib"}:
        number *= 1024.0 * 1024.0
    elif normalized in {"kb", "kib"}:
        number *= 1024.0
    return int(max(0.0, number))

def _mobile_export_clean_pip_artifact(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split("?", 1)[0].strip().rstrip("/")
    text = text.replace("\\", "/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.strip()

def _mobile_export_parse_pip_progress(line: str) -> dict:
    text = str(line or "").strip()
    info: dict[str, object] = {}
    if not text:
        return info

    artifact_match = re.search(
        r"\b(Downloading|Using cached)\s+(.+?)(?:\s+\([0-9]+(?:\.[0-9]+)?\s*(?:GB|GiB|MB|MiB|KB|KiB|B)\)|$)",
        text,
        flags=re.IGNORECASE,
    )
    if artifact_match:
        info["source"] = "cache" if artifact_match.group(1).lower().startswith("using") else "download"
        artifact = _mobile_export_clean_pip_artifact(artifact_match.group(2))
        if artifact:
            info["artifact"] = artifact

    collect_match = re.search(r"\bCollecting\s+(.+)$", text, flags=re.IGNORECASE)
    if collect_match and "artifact" not in info:
        artifact = _mobile_export_clean_pip_artifact(collect_match.group(1))
        if artifact:
            info["artifact"] = artifact
            info["source"] = "resolve"

    raw_match = re.search(r"\bProgress\s+([0-9]+)\s+of\s+([0-9]+)", text, flags=re.IGNORECASE)
    if raw_match:
        current = int(raw_match.group(1))
        total = max(1, int(raw_match.group(2)))
        info["current_bytes"] = current
        info["total_bytes"] = total
        info["fraction"] = max(0.0, min(0.98, current / total))

    progress_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*"
        r"(GB|GiB|MB|MiB|KB|KiB|B)"
        r"(?:\s+([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB|KB|KiB|B)/s)?"
        r"(?:.*?\beta\s+([0-9:.-]+))?",
        text,
        flags=re.IGNORECASE,
    )
    if progress_match:
        current = _mobile_export_size_to_bytes(progress_match.group(1), progress_match.group(3))
        total = _mobile_export_size_to_bytes(progress_match.group(2), progress_match.group(3))
        if current is not None:
            info["current_bytes"] = current
        if total is not None:
            info["total_bytes"] = total
        if current is not None and total:
            info["fraction"] = max(0.0, min(0.98, current / max(1, total)))
        if progress_match.group(4):
            speed = _mobile_export_size_to_bytes(progress_match.group(4), progress_match.group(5))
            if speed is not None:
                info["speed_bytes"] = speed
        if progress_match.group(6):
            info["eta"] = progress_match.group(6)

    size_matches = re.findall(
        r"\(([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB|KB|KiB|B)\)",
        text,
        flags=re.IGNORECASE,
    )
    if size_matches and "total_bytes" not in info:
        number, unit = size_matches[-1]
        size = _mobile_export_size_to_bytes(number, unit)
        if size is not None:
            info["total_bytes"] = size
    return info

def _mobile_export_install_detail(
    *,
    spec: str,
    artifact: str = "",
    current_bytes: int | None = None,
    total_bytes: int | None = None,
    speed_bytes: int | None = None,
    eta: str = "",
    source: str = "",
    installed_bytes: int | None = None,
) -> str:
    target = str(artifact or spec or "").strip()
    if target and len(target) > 72:
        target = f"...{target[-69:]}"
    source_key = str(source or "").lower()
    if source_key == "cache":
        prefix = "cache"
    elif source_key == "download":
        prefix = "plik"
    else:
        prefix = "pakiet"
    parts = [f"{prefix}: {target}" if target else "plik: -"]
    if current_bytes is not None and total_bytes:
        parts.append(f"{_format_mobile_export_bytes(current_bytes)}/{_format_mobile_export_bytes(total_bytes)}")
    elif total_bytes:
        parts.append(_format_mobile_export_bytes(total_bytes))
    elif installed_bytes:
        parts.append(f"lokalnie {_format_mobile_export_bytes(installed_bytes)}")
    if speed_bytes:
        parts.append(f"{_format_mobile_export_bytes(speed_bytes)}/s")
    if eta:
        parts.append(f"ETA {eta}")
    return " | ".join(part for part in parts if part)

def _mobile_export_parse_pip_size(line: str) -> tuple[int | None, float | None]:
    info = _mobile_export_parse_pip_progress(line)
    total = info.get("total_bytes")
    fraction = info.get("fraction")
    return (
        int(total) if isinstance(total, int) else None,
        float(fraction) if isinstance(fraction, (float, int)) else None,
    )

    progress_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB|KB|KiB|B)",
        text,
        flags=re.IGNORECASE,
    )
    if progress_match:
        current = float(progress_match.group(1)) * multiplier(progress_match.group(3))
        total = float(progress_match.group(2)) * multiplier(progress_match.group(3))
        progress = max(0.0, min(0.98, current / max(1.0, total)))
        return int(total), progress

    matches = re.findall(r"\(([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB|KB|KiB|B)\)", text, flags=re.IGNORECASE)
    if matches:
        number, unit = matches[-1]
        return int(float(number) * multiplier(unit)), None
    return None, None

def _mobile_export_int_or_none(value) -> int | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None

def _mobile_export_bool_or_none(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "tak", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "nie", "no", "n", "off"}:
        return False
    return None

def _mobile_export_run_like_value(run_like, key: str, default=None):
    if isinstance(run_like, dict):
        return run_like.get(key, default)
    try:
        return getattr(run_like, key, default)
    except Exception:
        return default

def _mobile_export_run_like_dict(run_like) -> dict:
    if not run_like:
        return {}
    if isinstance(run_like, dict):
        return dict(run_like)
    try:
        if hasattr(run_like, "to_dict"):
            payload = run_like.to_dict()
            if isinstance(payload, dict):
                return dict(payload)
    except Exception:
        pass
    payload: dict = {}
    for key in (
        "id",
        "name",
        "status",
        "dataset_path",
        "base_model",
        "epochs",
        "current_epoch",
        "metrics_history",
        "output_dir",
        "best_weights",
        "last_weights",
        "lineage_mode",
        "parent_run_id",
        "parent_model_path",
        "parent_model_name",
    ):
        value = _mobile_export_run_like_value(run_like, key, None)
        if value is not None:
            payload[key] = value
    return payload

def _mobile_export_completed_epoch_count(run_like) -> int:
    if not run_like:
        return 0
    values: list[int] = []
    for key in ("current_epoch", "completed_epochs", "trained_epochs"):
        parsed = _mobile_export_int_or_none(_mobile_export_run_like_value(run_like, key))
        if parsed is not None and parsed > 0:
            values.append(max(0, parsed))

    metrics = _mobile_export_run_like_value(run_like, "metrics_history", [])
    if isinstance(metrics, list):
        metric_epochs = [
            max(0, parsed)
            for row in metrics
            if isinstance(row, dict)
            for parsed in (_mobile_export_int_or_none(row.get("epoch") or row.get("Epoch")),)
            if parsed is not None
        ]
        if metric_epochs:
            values.append(max(metric_epochs))

    return max(values, default=0)

def _mobile_export_parent_run_id(run_like) -> str:
    parent_id = str(_mobile_export_run_like_value(run_like, "parent_run_id", "") or "").strip()
    if parent_id:
        return parent_id
    for key in ("parent_model_path", "parent_model_name", "base_model"):
        run_id = _mobile_export_run_id_from_text(_mobile_export_run_like_value(run_like, key, ""))
        if run_id:
            return run_id
    return ""

def _mobile_export_build_training_provenance(
    run_like,
    *,
    target: str = "",
    checkpoint: Path | str | None = None,
    dataset_path: str | Path | None = None,
    model_metadata: dict | None = None,
    include_dataset_fingerprint: bool = False,
) -> dict:
    """Build canonical provenance, keeping legacy callers non-fatal."""

    try:
        return build_model_training_provenance(
            run_like,
            history_index=_mobile_export_run_snapshot_index(),
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            target=target,
            model_sidecar=model_metadata if isinstance(model_metadata, dict) else None,
            include_dataset_fingerprint=include_dataset_fingerprint,
        )
    except Exception as exc:
        try:
            logger.warning("Nie udało się zbudować provenance modelu mobilnego: %s", exc)
        except Exception:
            pass
        return {}

def _mobile_export_provenance_known_total(provenance: dict | None) -> int | None:
    if not isinstance(provenance, dict):
        return None
    total = _mobile_export_int_or_none(provenance.get("total_epochs"))
    known_flag = _mobile_export_bool_or_none(provenance.get("total_epochs_known"))
    has_contract = bool(provenance.get("provenance_version"))
    if total is not None and total > 0 and known_flag is not False and (known_flag is True or has_contract):
        return total
    return None

def _mobile_export_provenance_display_total(provenance: dict | None) -> int:
    total = _mobile_export_provenance_known_total(provenance)
    if total is not None:
        return total
    minimum = _mobile_export_int_or_none((provenance or {}).get("known_epochs_minimum") if isinstance(provenance, dict) else None)
    return int(minimum or 0)

def _mobile_export_total_epochs_for_run_like(run_like, fallback=None) -> int:
    provenance = _mobile_export_build_training_provenance(
        run_like,
        include_dataset_fingerprint=False,
    )
    provenance_total = _mobile_export_provenance_display_total(provenance)
    if provenance_total > 0:
        return provenance_total

    total = 0
    seen: set[str] = set()
    cursor = _mobile_export_run_like_dict(run_like)
    snapshots = _mobile_export_run_snapshot_index()

    while cursor:
        run_id = str(cursor.get("id") or "").strip()
        if not run_id:
            run_id = _mobile_export_run_id_from_text(cursor.get("output_dir") or cursor.get("best_weights") or "")
        if run_id:
            if run_id in seen:
                break
            seen.add(run_id)

        total += _mobile_export_completed_epoch_count(cursor)
        parent_id = _mobile_export_parent_run_id(cursor)
        if not parent_id or parent_id in seen:
            break
        parent_snapshot = snapshots.get(parent_id)
        if not isinstance(parent_snapshot, dict):
            break
        cursor = dict(parent_snapshot)

    if total <= 0:
        for candidate in (
            fallback,
            _mobile_export_run_like_value(run_like, "total_epochs", None),
            _mobile_export_run_like_value(run_like, "current_epoch", None),
        ):
            parsed = _mobile_export_int_or_none(candidate)
            if parsed is not None and parsed > 0:
                return parsed
    return total

def _mobile_export_candidate_training_provenance(
    candidate: dict | None,
    *,
    include_dataset_fingerprint: bool = False,
) -> dict:
    """One checkpoint-bound profile shared by the table, details and manifest."""
    if not isinstance(candidate, dict):
        return {}
    metadata = candidate.get("model_metadata") or {}
    raw = metadata.get("raw") or {}
    run_like = (candidate.get("run") or candidate.get("history_snapshot")
                or raw.get("run_snapshot") or raw.get("training"))
    cached = candidate.get("training_provenance")
    # Small caller-supplied snapshots can have no source run (legacy packages).
    if not run_like and isinstance(cached, dict) and cached.get("provenance_version"):
        provenance = cached
    else:
        run_snapshot = _mobile_export_run_like_dict(run_like)
        signature = (
            _mobile_export_file_signature(candidate.get("best_weights")),
            json.dumps(run_snapshot, sort_keys=True, default=str),
            bool(include_dataset_fingerprint),
        )
        if candidate.get("_training_provenance_signature") == signature and isinstance(cached, dict):
            provenance = cached
        else:
            provenance = _mobile_export_build_training_provenance(
                run_like, target=str(candidate.get("target") or ""),
                checkpoint=candidate.get("best_weights"), dataset_path=candidate.get("dataset_path"),
                model_metadata=metadata, include_dataset_fingerprint=include_dataset_fingerprint)
            candidate["_training_provenance_signature"] = signature
        summary = build_checkpoint_metric_summary(run_like, checkpoint=candidate.get("best_weights"))
        provenance["best_epoch"] = summary["best_epoch"]
        provenance["best_epoch_source"] = summary["best_epoch_source"]
        for key in ("best_map50", "best_map50_95"):
            if summary["best_row"] or summary["checkpoint_mismatch"]:
                candidate[key] = summary[key]
        candidate["checkpoint_sha256"] = summary["checkpoint_sha256"]
    candidate["_training_provenance_loaded"] = True
    candidate["training_provenance"] = provenance
    candidate["best_epoch"] = provenance.get("best_epoch")
    candidate["best_epoch_source"] = provenance.get("best_epoch_source", "unknown")
    candidate["total_epochs"] = _mobile_export_provenance_display_total(provenance)
    candidate["total_epochs_known"] = _mobile_export_provenance_known_total(provenance) is not None
    for key in ("known_epochs_minimum", "lineage_stage_count_known", "known_stage_count_minimum", "provenance_status"):
        if key in provenance:
            candidate[key] = provenance[key]
    dataset = provenance.get("dataset") or {}
    if dataset.get("dataset_id"):
        candidate["dataset_label"] = dataset["dataset_id"]
    return provenance



def _mobile_export_candidate_training_int(candidate: dict | None, key: str) -> int | None:
    provenance = _mobile_export_candidate_training_provenance(candidate)
    value = _mobile_export_int_or_none(provenance.get(key))
    if value is not None:
        return value
    if isinstance(candidate, dict):
        return _mobile_export_int_or_none(candidate.get(key))
    return None


def _mobile_export_candidate_lineage_stage_count(candidate: dict | None) -> int:
    value = _mobile_export_candidate_training_int(candidate, "lineage_stage_count")
    if value is not None and value > 0:
        return value
    provenance = _mobile_export_candidate_training_provenance(candidate)
    lineage = provenance.get("lineage") if isinstance(provenance, dict) else None
    if isinstance(lineage, list) and lineage:
        return len(lineage)
    return 0


def _mobile_export_candidate_lineage_stage_count_known(candidate: dict | None) -> bool | None:
    provenance = _mobile_export_candidate_training_provenance(candidate)
    if isinstance(provenance, dict) and provenance.get("lineage_stage_count_known") is not None:
        return _mobile_export_bool_or_none(provenance.get("lineage_stage_count_known"))
    if isinstance(candidate, dict) and candidate.get("lineage_stage_count_known") is not None:
        return _mobile_export_bool_or_none(candidate.get("lineage_stage_count_known"))
    return None


def _mobile_export_known_or_minimum_label(value, *, known: bool | None, minimum=None) -> str:
    parsed = _mobile_export_int_or_none(value)
    minimum_value = _mobile_export_int_or_none(minimum)
    if known is False:
        if minimum_value is not None and minimum_value > 0:
            return f"co najmniej {_format_mobile_export_int(minimum_value)}"
        return "-"
    if parsed is not None and parsed > 0:
        return _format_mobile_export_int(parsed)
    if minimum_value is not None and minimum_value > 0:
        return f"co najmniej {_format_mobile_export_int(minimum_value)}"
    return "-"


def _mobile_export_provenance_capture_label(value) -> str:
    capture = str(value or "").strip()
    labels = {
        "frozen_at_training_start": "zamrożony przed treningiem",
        "reconstructed_from_training_artifacts": "odtworzony z artefaktów treningu",
        "reconstructed_at_export": "odtworzony podczas eksportu",
        "legacy_unknown": "historyczny, niepełny",
    }
    return labels.get(capture, capture or "-")


def _mobile_export_provenance_status_label(provenance: dict | None) -> str:
    if not isinstance(provenance, dict) or not provenance:
        return "dane historyczne niepełne"
    status = str(provenance.get("provenance_status") or "").strip().lower()
    capture = str(provenance.get("provenance_capture") or "").strip()
    if status == "complete" and capture == "frozen_at_training_start":
        return "pełny"
    if status == "complete":
        return "pełny, ale odtworzony"
    if status == "legacy_unknown":
        return "historyczny, niepełny"
    return "częściowy"


def _mobile_export_target_source_label(candidate: dict | None) -> str:
    if not isinstance(candidate, dict):
        return "-"
    marker = _mobile_export_target_marker(candidate)
    source = str(candidate.get("target_source") or "").strip() or "brak jawnego źródła"
    conflict = str(candidate.get("target_conflict") or "").strip()
    text = f"{marker} | {source}"
    if conflict:
        text = f"{text} | uwaga: {conflict}"
    return text


def _mobile_export_candidate_total_epochs(candidate: dict | None) -> int:
    if not isinstance(candidate, dict):
        return 0
    provenance = _mobile_export_candidate_training_provenance(candidate)
    provenance_total = _mobile_export_provenance_display_total(provenance)
    if provenance_total > 0:
        return provenance_total
    explicit = _mobile_export_int_or_none(candidate.get("total_epochs"))
    if explicit is not None and explicit > 0:
        return explicit
    run = candidate.get("run")
    if run is not None:
        return _mobile_export_total_epochs_for_run_like(run)

    metadata = candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {}
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
    run_snapshot = raw.get("run_snapshot") if isinstance(raw.get("run_snapshot"), dict) else {}
    history_snapshot = candidate.get("history_snapshot") if isinstance(candidate.get("history_snapshot"), dict) else {}
    total = _mobile_export_int_or_none(
        _mobile_export_nested_value(
            {"training": training, "run": run_snapshot, "history": history_snapshot, "candidate": candidate},
            "training.total_epochs",
            "run.total_epochs",
            "history.total_epochs",
            "candidate.current_epoch",
        )
    )
    if total is not None and total > 0:
        return total
    return _mobile_export_total_epochs_for_run_like(run_snapshot or history_snapshot, fallback=candidate.get("current_epoch"))


def _mobile_export_candidate_epochs_label(candidate: dict | None, *, compact: bool = False) -> str:
    provenance = _mobile_export_candidate_training_provenance(candidate)
    known = _mobile_export_bool_or_none(provenance.get("total_epochs_known"))
    total = provenance.get("total_epochs")
    minimum = provenance.get("known_epochs_minimum")
    if not provenance:
        known = _mobile_export_bool_or_none((candidate or {}).get("total_epochs_known"))
        total = _mobile_export_candidate_total_epochs(candidate)
        minimum = (candidate or {}).get("known_epochs_minimum")
    label = _mobile_export_known_or_minimum_label(total, known=known, minimum=minimum)
    return label.replace("co najmniej ", "≥ ") if compact else label


def _mobile_export_epoch_profile(candidate: dict | None) -> tuple[float | None, str] | None:
    if not isinstance(candidate, dict):
        return None
    provenance = _mobile_export_candidate_training_provenance(candidate)
    current_epoch = provenance.get("run_epochs_completed") if provenance else candidate.get("current_epoch")
    epochs = provenance.get("run_epochs_planned") if provenance else candidate.get("epochs")
    total_epochs = _mobile_export_candidate_total_epochs(candidate)
    run = candidate.get("run")
    if run is not None:
        current_epoch = current_epoch if current_epoch not in (None, "") else getattr(run, "current_epoch", None)
        epochs = epochs if epochs not in (None, "") else getattr(run, "epochs", None)

    current_value = _mobile_export_int_or_none(current_epoch)
    planned_value = _mobile_export_int_or_none(epochs)
    if planned_value is None or planned_value <= 0:
        if total_epochs > 0:
            return None, f"Łącznie w historii treningu: {_mobile_export_candidate_epochs_label(candidate)} epok"
        return None

    done_value = max(0, current_value or 0)
    ratio = max(0.0, min(1.0, float(done_value) / max(1.0, float(planned_value))))
    text = f"{_mobile_export_candidate_epochs_label(candidate, compact=True)} łącznie | {done_value} ostatnio"
    return ratio, text


def _mobile_export_epochs_tooltip(candidate: dict) -> str:
    provenance = _mobile_export_candidate_training_provenance(candidate)
    total = _mobile_export_candidate_epochs_label(candidate)
    completed = provenance.get("run_epochs_completed")
    text = f"Łączna liczba epok w historii treningu: {total}.\nLiczba epok ostatniego treningu: {completed if completed is not None else '-'}."
    if provenance.get("total_epochs_known") is False:
        text += "\n≥ oznacza znane minimum. Nie udało się odtworzyć całej historii modelu."
    text += "\nSuma obejmuje treningi projektu, bez wstępnego treningu bazowego YOLO."
    return text

def _mobile_export_target_marker(candidate: dict | None) -> str:
    if not isinstance(candidate, dict):
        return "M?"
    target = _mobile_export_normalized_target_or_empty(candidate.get("target"))
    role = str(candidate.get("role") or "").strip().lower()
    if target == "plate" or role == "plate":
        return "MT"
    if target in {"char", "character"} or role == "character":
        return "MZ"
    if target == "vehicle" or role == "vehicle":
        return "MP"
    return "M?"

def _mobile_export_model_cell_label(candidate: dict | None) -> str:
    if not isinstance(candidate, dict):
        return "-"
    model_label = str(candidate.get("model_label") or "-").strip() or "-"
    return f"{_mobile_export_target_marker(candidate)}  {model_label}"

def _mobile_export_target_tooltip(candidate: dict | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    marker = _mobile_export_target_marker(candidate)
    target_label = str(candidate.get("target_label") or "-").strip() or "-"
    role = str(candidate.get("role") or "-").strip() or "-"
    task = str(candidate.get("task") or "-").strip() or "-"
    return (
        f"{marker}: {target_label}\n"
        f"Rola Android: {role} / {task}\n"
        f"Projekt: {_mobile_export_project_label(candidate, empty='-')}\n"
        f"Model: {candidate.get('model_label') or '-'}\n"
        f"Run: {candidate.get('run_label') or '-'}"
    )

def _mobile_export_target_base_color(palette: dict, candidate: dict | None) -> str:
    marker = _mobile_export_target_marker(candidate)
    if marker == "MT":
        return palette.get("model_role_plate", palette.get("warning", "#d9822b"))
    if marker == "MZ":
        return palette.get("model_role_character", palette.get("accent", "#3f8cff"))
    if marker == "MP":
        return palette.get("model_role_vehicle", palette.get("success", "#00c2a8"))
    return palette.get("muted", "#9aa0a6")

def _mobile_export_target_color(palette: dict, candidate: dict | None, *, selected: bool = False) -> str:
    base = _mobile_export_target_base_color(palette, candidate)
    bg = palette.get("panel", "#252526")
    return blend_hex_colors(base, "#ffffff" if selected else bg, 0.12 if selected else 0.02)


def _mobile_export_scrollbar_kwargs(palette: dict, track_bg: str | None = None) -> dict[str, str]:
    resolved = normalize_theme_palette(palette)
    return {
        "track_color": str(resolved.get("scrollbar_track") or track_bg or resolved["field"]),
        "thumb_color": str(resolved["scrollbar_thumb"]),
        "thumb_hover_color": str(resolved["scrollbar_thumb_hover"]),
    }


def _mobile_export_contrast_text_color(hex_color: str, *, dark: str = "#101418", light: str = "#f7fbff") -> str:
    try:
        value = str(hex_color or "").strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return dark if luminance >= 0.56 else light
    except Exception:
        return light

_MOBILE_EXPORT_CALIBRATION_SCAN_CACHE: dict[tuple[str, str, int], list[Path]] = {}
_MOBILE_EXPORT_CALIBRATION_YAML_CACHE: dict[str, tuple[int, dict]] = {}


def _mobile_export_resolve_data_yaml(path_value) -> Path | None:
    raw = str(path_value or "").strip().strip('"')
    if not raw:
        return None
    try:
        path = Path(raw)
    except Exception:
        return None
    try:
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            return path
        candidate = path / "data.yaml"
        if candidate.exists() and candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _mobile_export_yaml_payload(yaml_path: Path | None) -> dict:
    if yaml_path is None:
        return {}
    try:
        safe_path = Path(yaml_path)
        stamp = int(safe_path.stat().st_mtime_ns)
        key = str(safe_path.resolve()).lower()
    except Exception:
        return {}
    cached = _MOBILE_EXPORT_CALIBRATION_YAML_CACHE.get(key)
    if cached and cached[0] == stamp:
        return dict(cached[1])
    try:
        payload = safe_load_yaml(safe_path) or {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    _MOBILE_EXPORT_CALIBRATION_YAML_CACHE[key] = (stamp, dict(payload))
    return payload


def _mobile_export_yaml_names_text(payload: dict) -> str:
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, dict):
        return " ".join(str(value) for value in names.values())
    if isinstance(names, (list, tuple, set)):
        return " ".join(str(value) for value in names)
    return str(names or "")


def _mobile_export_yaml_target_score(yaml_path: Path | None, target: str) -> tuple[int, str]:
    yaml_path = _mobile_export_resolve_data_yaml(yaml_path) or yaml_path
    if yaml_path is None:
        return 0, "brak pliku data.yaml"
    try:
        safe_path = Path(yaml_path)
        if not safe_path.exists() or not safe_path.is_file():
            return 0, "plik data.yaml nie istnieje"
    except Exception:
        return 0, "nie można odczytać pliku data.yaml"

    normalized = CONFIG.normalize_task_target(target)
    payload = _mobile_export_yaml_payload(safe_path)
    names_text = _mobile_export_yaml_names_text(payload)
    text = " ".join(
        str(part or "")
        for part in (
            safe_path,
            safe_path.parent,
            payload.get("path") if isinstance(payload, dict) else "",
            payload.get("train") if isinstance(payload, dict) else "",
            payload.get("val") if isinstance(payload, dict) else "",
            payload.get("test") if isinstance(payload, dict) else "",
            names_text,
        )
    ).casefold()
    has_kpt = isinstance(payload, dict) and "kpt_shape" in payload
    names_count = 0
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, dict):
        names_count = len(names)
    elif isinstance(names, (list, tuple, set)):
        names_count = len(names)

    plate_tokens = ("plate", "plates", "tablic", "license", "licence", "pose")
    char_tokens = ("char", "chars", "znak", "ocr", "character", "characters")
    vehicle_tokens = ("vehicle", "vehicles", "pojazd", "coco", "car", "truck", "bus", "motorcycle")
    score = 0
    reasons: list[str] = []

    if normalized == "plate":
        if has_kpt:
            score += 80
            reasons.append("YOLO Pose/kpt_shape")
        if any(token in text for token in plate_tokens):
            score += 30
            reasons.append("tor tablic")
        if any(token in text for token in char_tokens):
            score -= 35
            reasons.append("ślady toru znaków")
        if any(token in text for token in vehicle_tokens):
            score -= 20
            reasons.append("ślady toru pojazdów")
    elif normalized == "char":
        if not has_kpt:
            score += 20
        else:
            score -= 55
            reasons.append("dataset pose tablic")
        if any(token in text for token in char_tokens):
            score += 55
            reasons.append("tor znaków")
        if names_count >= 10:
            score += 20
            reasons.append("wiele klas znaków")
        if any(token in text for token in vehicle_tokens):
            score -= 25
            reasons.append("ślady toru pojazdów")
    elif normalized == "vehicle":
        if not has_kpt:
            score += 20
        else:
            score -= 55
            reasons.append("dataset pose tablic")
        if any(token in text for token in vehicle_tokens):
            score += 65
            reasons.append("tor pojazdów")
        if any(token in text for token in char_tokens):
            score -= 25
            reasons.append("ślady toru znaków")
    else:
        score += 10

    if not reasons:
        reasons.append("zgodność niepewna")
    return score, ", ".join(reasons)


def _mobile_export_dataset_root_label(yaml_path: Path, target: str) -> str:
    try:
        ref = build_dataset_display_ref(yaml_path.parent, target_hint=target)
        return ref.detail_label
    except Exception:
        return str(yaml_path)


def _mobile_export_add_calibration_candidate(
    result: list[dict],
    seen: set[str],
    path_value,
    *,
    target: str,
    reason: str,
    force: bool = False,
    preferred: bool = False,
) -> None:
    yaml_path = _mobile_export_resolve_data_yaml(path_value)
    if yaml_path is None:
        return
    try:
        key = str(yaml_path.resolve()).lower()
    except Exception:
        key = str(yaml_path).lower()
    if not key or key in seen:
        return
    score, score_reason = _mobile_export_yaml_target_score(yaml_path, target)
    if not force and score < 35:
        return
    seen.add(key)
    label = f"{_mobile_export_dataset_root_label(yaml_path, target)} | {reason}"
    result.append(
        {
            "path": str(yaml_path),
            "label": label,
            "reason": reason,
            "score": score,
            "score_reason": score_reason,
            "preferred": bool(preferred),
        }
    )


def _mobile_export_scan_calibration_yamls(root_value, target: str, *, limit: int = 80) -> list[Path]:
    if not root_value:
        return []
    try:
        root = Path(root_value)
        if not root.exists():
            return []
        root_key = str(root.resolve()).lower()
        stamp = int(root.stat().st_mtime_ns)
    except Exception:
        return []
    cache_key = (CONFIG.normalize_task_target(target), root_key, stamp)
    cached = _MOBILE_EXPORT_CALIBRATION_SCAN_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    found: list[tuple[int, str, Path]] = []
    try:
        candidates = [root] if root.is_file() else root.rglob("data.yaml")
        for yaml_path in candidates:
            yaml_path = _mobile_export_resolve_data_yaml(yaml_path)
            if yaml_path is None:
                continue
            score, _reason = _mobile_export_yaml_target_score(yaml_path, target)
            if score >= 35:
                found.append((score, str(yaml_path).lower(), yaml_path))
            if len(found) >= max(limit * 3, limit):
                break
    except Exception:
        found = []
    found.sort(key=lambda item: (-item[0], item[1]))
    result = [path for _score, _key, path in found[:limit]]
    _MOBILE_EXPORT_CALIBRATION_SCAN_CACHE[cache_key] = list(result)
    return result


def _mobile_export_calibration_yaml_candidates(candidate: dict | None, *, limit: int = 12) -> list[dict]:
    if not isinstance(candidate, dict):
        return []
    target = CONFIG.normalize_task_target(str(candidate.get("target") or ""))
    result: list[dict] = []
    seen: set[str] = set()

    _mobile_export_add_calibration_candidate(
        result,
        seen,
        candidate.get("dataset_path"),
        target=target,
        reason="dataset przypisany do kandydata",
        force=True,
        preferred=True,
    )
    run = candidate.get("run")
    if run is not None:
        _mobile_export_add_calibration_candidate(
            result,
            seen,
            getattr(run, "dataset_path", ""),
            target=target,
            reason="dataset zapisany w historii runu",
            force=True,
            preferred=not bool(result),
        )

    roots: list[tuple[Path, str]] = []
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        project_root = CAMPAIGN.get_active_project_root_dir() if project_name else None
        if project_root:
            roots.append((Path(project_root) / "4_training_datasets", f"projekt {project_name}"))
    except Exception:
        pass
    try:
        roots.append((Path(CONFIG.get_datasets_dir(target)), "globalny katalog toru"))
    except Exception:
        pass
    try:
        roots.append((Path(getattr(CONFIG, "DIR_4_DATASETS", "")), "globalne datasety"))
    except Exception:
        pass

    for root, source_label in roots:
        if len(result) >= limit:
            break
        for yaml_path in _mobile_export_scan_calibration_yamls(root, target, limit=limit):
            if len(result) >= limit:
                break
            _mobile_export_add_calibration_candidate(
                result,
                seen,
                yaml_path,
                target=target,
                reason=source_label,
                force=False,
                preferred=False,
            )
    result.sort(key=lambda row: (not bool(row.get("preferred")), -int(row.get("score") or 0), str(row.get("label") or "")))
    return result[:limit]


def _mobile_export_default_calibration_path(candidate: dict | None) -> str:
    suggestions = _mobile_export_calibration_yaml_candidates(candidate, limit=1)
    if suggestions:
        return str(suggestions[0].get("path") or "")
    return ""


def _mobile_export_calibration_problem(candidate: dict | None, calibration_path: Path | None) -> str:
    if calibration_path is None:
        return "Wariant INT8 wymaga reprezentatywnego pliku data.yaml dla tego modelu."
    try:
        safe_path = Path(calibration_path)
        if not safe_path.exists() or not safe_path.is_file():
            return "Wybrany plik data.yaml kalibracji INT8 nie istnieje."
    except Exception:
        return "Nie można odczytać pliku data.yaml kalibracji INT8."
    if not isinstance(candidate, dict):
        return ""
    target = CONFIG.normalize_task_target(str(candidate.get("target") or ""))
    score, reason = _mobile_export_yaml_target_score(safe_path, target)
    if score < 35:
        marker = _mobile_export_target_marker(candidate)
        return (
            f"{marker}: data.yaml wygląda na niezgodny z torem modelu "
            f"({reason}). Wybierz YAML z tego samego toru albo wskaż właściwy plik ręcznie."
        )
    return ""

def _mobile_export_percent_value(value) -> float | None:
    try:
        text = str(value if value is not None else "").strip().replace(",", ".")
        if not text:
            return None
        parsed = float(text)
        if parsed > 1.0 and parsed <= 100.0:
            parsed = parsed / 100.0
        return max(0.0, min(1.0, parsed))
    except Exception:
        return None

def _mobile_export_chart_color(palette: dict, tone: str) -> str:
    if tone == "success":
        return palette.get("success", "#2ecc71")
    if tone == "warning":
        return palette.get("warning", "#f1c40f")
    if tone == "error":
        return palette.get("error", "#e74c3c")
    return palette.get("accent", "#4f8de3")

def _mobile_export_metric_tone(value) -> str:
    parsed = _mobile_export_percent_value(value)
    if parsed is None:
        return "muted"
    if parsed >= 0.75:
        return "success"
    if parsed >= 0.45:
        return "warning"
    return "error"

def _mobile_export_text_for_width(text: str, pixel_width: int, *, min_chars: int = 10) -> str:
    raw = str(text or "-")
    width_chars = max(min_chars, min(180, int(max(40, pixel_width) / 7.0)))
    wrapped_lines: list[str] = []
    for paragraph in raw.splitlines() or [""]:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                paragraph,
                width=width_chars,
                break_long_words=True,
                break_on_hyphens=True,
                replace_whitespace=False,
                drop_whitespace=True,
            )
            or [paragraph]
        )
    return "\n".join(wrapped_lines)

def _draw_mobile_export_metric_chart(canvas: tk.Canvas, candidate: dict, palette: dict, max_size_mb: float = 0.0) -> None:
    try:
        canvas.delete("all")
        width = max(280, int(canvas.winfo_width() or canvas.cget("width") or 320))
        height = max(150, int(canvas.winfo_height() or canvas.cget("height") or 170))
    except Exception:
        return

    bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4f8de3")
    soft = blend_hex_colors(bg, accent, 0.10)
    border = blend_hex_colors(accent, bg, 0.45)
    canvas.configure(bg=bg, highlightthickness=0)
    canvas.create_rectangle(1, 1, width - 2, height - 2, fill=soft, outline=border, width=1)
    canvas.create_text(16, 16, text="Profil kandydata", fill=fg, anchor=tk.W, font=("Segoe UI", 9, "bold"))

    rows: list[tuple[str, float | None, str, str]] = [
        ("mAP50-95", _mobile_export_percent_value(candidate.get("best_map50_95")), _format_mobile_export_metric(candidate.get("best_map50_95")), _mobile_export_metric_tone(candidate.get("best_map50_95"))),
        ("mAP50", _mobile_export_percent_value(candidate.get("best_map50")), _format_mobile_export_metric(candidate.get("best_map50")), _mobile_export_metric_tone(candidate.get("best_map50"))),
    ]
    epoch_profile = _mobile_export_epoch_profile(candidate)
    if epoch_profile is not None:
        epoch_ratio, epoch_text = epoch_profile
        rows.append(("Epoki", epoch_ratio, epoch_text, "info"))
    size_mb = candidate.get("file_size_mb")
    try:
        if size_mb is not None:
            size_ratio = max(0.0, min(1.0, float(size_mb or 0.0) / max(1.0, float(max_size_mb or size_mb or 1.0))))
            rows.append(("Rozmiar", size_ratio, f"{float(size_mb):.1f} MB", "info"))
    except Exception:
        pass

    if not rows:
        canvas.create_text(width / 2, height / 2, text="Brak metryk do pokazania", fill=muted, anchor=tk.CENTER, font=("Segoe UI", 9))
        return

    left = 16
    right = width - 16
    top = 42
    row_h = max(28, int((height - top - 12) / max(1, len(rows))))
    for idx, (label, ratio, value_text, tone) in enumerate(rows):
        y = top + idx * row_h
        label_y = y + 7
        bar_y0 = y + 17
        bar_y1 = min(y + 25, height - 10)
        value_width = max(80, width - 115)
        display_value = _mobile_export_text_for_width(str(value_text or "-"), value_width, min_chars=10).splitlines()[0]
        canvas.create_text(left, label_y, text=label, fill=muted, anchor=tk.W, font=("Segoe UI", 8, "bold"))
        canvas.create_text(right, label_y, text=display_value, fill=fg, anchor=tk.E, font=("Segoe UI", 8))
        canvas.create_rectangle(left, bar_y0, right, bar_y1, fill=blend_hex_colors(bg, "#ffffff", 0.08), outline="")
        if ratio is not None:
            color = _mobile_export_chart_color(palette, tone)
            canvas.create_rectangle(left, bar_y0, left + max(2, int((right - left) * ratio)), bar_y1, fill=color, outline="")

def _draw_mobile_export_overview_chart(
    canvas: tk.Canvas,
    candidates: list[dict],
    selected_iid: str,
    palette: dict,
    export_iids: set[str] | None = None,
) -> None:
    try:
        canvas.delete("all")
        width = max(300, int(canvas.winfo_width() or canvas.cget("width") or 330))
        height = max(150, int(canvas.winfo_height() or canvas.cget("height") or 170))
    except Exception:
        return

    bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4f8de3")
    success = palette.get("success", "#2ecc71")
    chart_bg = blend_hex_colors(bg, accent, 0.08)
    selected_for_export = {str(item or "") for item in (export_iids or set()) if str(item or "")}
    canvas.configure(bg=bg, highlightthickness=0)
    canvas.create_rectangle(1, 1, width - 2, height - 2, fill=chart_bg, outline=blend_hex_colors(accent, bg, 0.45), width=1)
    canvas.create_text(14, 16, text="mAP50-95 kandydatów + wybór", fill=fg, anchor=tk.W, font=("Segoe UI", 9, "bold"))

    legend_items = (("MZ", {"role": "character"}), ("MT", {"role": "plate"}), ("MP", {"role": "vehicle"}))
    legend_x = max(158, width - 132)
    for idx, (marker, marker_candidate) in enumerate(legend_items):
        x = legend_x + idx * 42
        color = _mobile_export_target_color(palette, marker_candidate)
        canvas.create_rectangle(x, 10, x + 14, 20, fill=color, outline="")
        canvas.create_text(x + 18, 15, text=marker, fill=muted, anchor=tk.W, font=("Segoe UI", 7, "bold"))

    candidates_by_iid = {str(item.get("iid") or ""): item for item in candidates if isinstance(item, dict)}
    scored = [
        item
        for item in candidates
        if _mobile_export_percent_value(item.get("best_map50_95")) is not None
    ]
    scored.sort(key=lambda item: _mobile_export_percent_value(item.get("best_map50_95")) or 0.0, reverse=True)
    pinned_iids = set(selected_for_export)
    if selected_iid:
        pinned_iids.add(str(selected_iid))
    visible_scored = scored[:18]
    visible_iids = {str(item.get("iid") or "") for item in visible_scored}
    for pinned_iid in pinned_iids:
        item = candidates_by_iid.get(str(pinned_iid or ""))
        item_iid = str((item or {}).get("iid") or "")
        if not item or not item_iid or item_iid in visible_iids:
            continue
        if len(visible_scored) >= 18:
            replace_index = next(
                (
                    index
                    for index in range(len(visible_scored) - 1, -1, -1)
                    if str(visible_scored[index].get("iid") or "") not in pinned_iids
                ),
                len(visible_scored) - 1,
            )
            visible_iids.discard(str(visible_scored[replace_index].get("iid") or ""))
            visible_scored[replace_index] = item
        else:
            visible_scored.append(item)
        visible_iids.add(item_iid)
    scored = visible_scored
    if not scored:
        canvas.create_text(
            width / 2,
            height / 2,
            text="Brak metryk w kandydaturach",
            fill=muted,
            anchor=tk.CENTER,
            font=("Segoe UI", 9),
        )
        return

    left = 18
    right = width - 16
    bottom = height - 34
    top = 56
    gap = 4
    bar_w = max(6, int((right - left - gap * (len(scored) - 1)) / max(1, len(scored))))
    canvas.create_line(left, bottom, right, bottom, fill=blend_hex_colors(fg, bg, 0.55))
    for idx, item in enumerate(scored):
        value = _mobile_export_percent_value(item.get("best_map50_95"))
        missing_primary_metric = value is None
        if value is None:
            value = 0.0
        x0 = left + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = bottom - max(2, int((bottom - top) * value))
        item_iid = str(item.get("iid") or "")
        selected = item_iid == str(selected_iid or "")
        checked_for_export = item_iid in selected_for_export
        fill = _mobile_export_target_color(palette, item, selected=selected)
        outline = fg if selected else blend_hex_colors(fill, bg, 0.30) if missing_primary_metric else ""
        canvas.create_rectangle(x0, y0, x1, bottom, fill=fill, outline=outline, width=1 if selected or missing_primary_metric else 0)
        marker = _mobile_export_target_marker(item)
        marker_x = (x0 + x1) / 2
        badge_height = 11
        badge_width = max(10, int(bar_w))
        badge_x0 = max(left, marker_x - badge_width / 2)
        badge_x1 = min(right, marker_x + badge_width / 2)
        if badge_x1 - badge_x0 < badge_width:
            if badge_x0 <= left:
                badge_x1 = min(right, badge_x0 + badge_width)
            elif badge_x1 >= right:
                badge_x0 = max(left, badge_x1 - badge_width)
        badge_y1 = max(35, y0 - 4)
        badge_y0 = max(28, badge_y1 - badge_height)
        if badge_y1 - badge_y0 < badge_height:
            badge_y1 = badge_y0 + badge_height
        badge_fill = fill
        badge_outline = blend_hex_colors(fill, fg, 0.28)
        badge_text = marker
        badge_font = ("Segoe UI", 6, "bold")
        if checked_for_export:
            badge_fill = success
            badge_outline = blend_hex_colors(success, fg, 0.50)
            badge_text = marker if badge_width >= 16 else marker[-1:]
            badge_font = ("Segoe UI", 6, "bold")
        elif selected:
            badge_outline = blend_hex_colors(accent, fg, 0.42)
        canvas.create_rectangle(
            badge_x0,
            badge_y0,
            badge_x1,
            badge_y1,
            fill=badge_fill,
            outline=badge_outline,
            width=2 if checked_for_export or selected else 1,
        )
        canvas.create_text(
            (badge_x0 + badge_x1) / 2,
            (badge_y0 + badge_y1) / 2 - 0.5,
            text=badge_text,
            fill=_mobile_export_contrast_text_color(badge_fill),
            anchor=tk.CENTER,
            font=badge_font,
        )
        if checked_for_export or selected:
            circle_radius = 5 if checked_for_export else 4
            circle_y = max(20, badge_y0 - circle_radius - 3)
            circle_fill = success if checked_for_export else accent
            canvas.create_oval(
                marker_x - circle_radius,
                circle_y - circle_radius,
                marker_x + circle_radius,
                circle_y + circle_radius,
                fill=circle_fill,
                outline=blend_hex_colors(circle_fill, fg, 0.50),
                width=1,
            )
            if checked_for_export:
                canvas.create_text(
                    marker_x,
                    circle_y - 1,
                    text="✓",
                    fill=_mobile_export_contrast_text_color(circle_fill),
                    anchor=tk.CENTER,
                    font=("Segoe UI", 7, "bold"),
                )
    missing_count = sum(_mobile_export_percent_value(item.get("best_map50_95")) is None for item in candidates)
    canvas.create_text(left, height - 12, text=f"Bez metryk: {missing_count}", fill=muted, anchor=tk.W, font=("Segoe UI", 8))
    canvas.create_text(right, height - 12, text=f"{len(scored)}/{len(candidates)} modeli", fill=muted, anchor=tk.E, font=("Segoe UI", 8))

def _mobile_export_safe_path_key(path_value) -> str:
    try:
        return str(Path(str(path_value or "")).resolve()).lower()
    except Exception:
        return str(path_value or "").strip().lower()


def _mobile_export_file_signature(path_value) -> tuple:
    path = Path(str(path_value or ""))
    try:
        stat = path.stat()
        return (_mobile_export_safe_path_key(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return (_mobile_export_safe_path_key(path), None, None, None)

def _mobile_export_project_identity_from_path(path_value) -> tuple[str, str]:
    try:
        raw_path = Path(path_value)
    except Exception:
        return "", ""
    try:
        resolved = raw_path.resolve()
    except Exception:
        resolved = raw_path
    try:
        projects_root = Path(CONFIG.DIR_9_PROJECTS).resolve()
    except Exception:
        return "", ""
    try:
        relative = resolved.relative_to(projects_root)
    except Exception:
        return "", ""
    if not relative.parts:
        return "", ""
    project_name = str(relative.parts[0] or "").strip()
    if not project_name:
        return "", ""
    project_root = projects_root / project_name
    try:
        if _mobile_export_safe_path_key(CAMPAIGN.get_active_project_root_dir()) == _mobile_export_safe_path_key(project_root):
            return str(CAMPAIGN.get_active_project_name() or project_name), str(project_root)
        state = getattr(CAMPAIGN, "state", {})
        for label, item in (state.get("projects", {}) if isinstance(state, dict) else {}).items():
            if item.get("folder_name") == project_name:
                return str(label), str(project_root)
    except Exception:
        pass
    return project_name, str(project_root)

def _mobile_export_project_label(candidate: dict | None, *, empty: str = "") -> str:
    if not isinstance(candidate, dict):
        return empty
    project_name = str(candidate.get("project_name") or "").strip()
    return project_name or empty

def _mobile_export_candidate_dedupe_key(candidate: dict | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    path_key = _mobile_export_safe_path_key(candidate.get("best_weights"))
    digest = _file_sha256(Path(str(candidate.get("best_weights") or "")))
    if digest:
        candidate["checkpoint_sha256"] = digest
        return f"sha256|{digest}"
    return f"path|{path_key}" if path_key else ""



def _mobile_export_candidate_preference(candidate: dict | None) -> tuple:
    if not isinstance(candidate, dict):
        return (0, 0, 0.0, 0)
    scope = str(candidate.get("scope") or "").strip().lower()
    priority = 0
    if scope.startswith("projekt"):
        priority += 80
    if "lokalne modele pojazdów" in scope:
        priority += 70
    if "bazowe modele pojazdów" in scope:
        priority += 60
    if "globalne modele pojazdów" in scope:
        priority += 50
    metadata = candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {}
    if str(metadata.get("metadata_json") or "").strip():
        priority += 8
    try:
        path = Path(candidate.get("best_weights"))
        parts = [str(part).lower() for part in path.parts]
        if "ultralytics" not in parts:
            priority += 4
        depth_score = -len(parts)
        mtime = float(path.stat().st_mtime)
    except Exception:
        depth_score = 0
        mtime = 0.0
    return (int(candidate.get("run") is not None), int(bool(candidate.get("history_snapshot"))),
            priority, depth_score, mtime, len(str(candidate.get("model_label") or "")))

def _dedupe_mobile_export_candidates(candidates: list[dict]) -> list[dict]:
    result: list[dict] = []
    index_by_key: dict[str, int] = {}
    for candidate in candidates or []:
        key = _mobile_export_candidate_dedupe_key(candidate)
        if not key:
            result.append(candidate)
            continue
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(result)
            result.append(candidate)
            continue
        existing = result[existing_index]
        existing_run = existing.get("run")
        candidate_run = candidate.get("run")
        if (existing_run is not None and candidate_run is not None
                and _mobile_export_run_like_value(existing_run, "id") != _mobile_export_run_like_value(candidate_run, "id")):
            # Equal weights can belong to separate recorded training stages.
            # Do not silently replace one lineage with another.
            result.append(candidate)
            continue
        if _mobile_export_candidate_preference(candidate) > _mobile_export_candidate_preference(existing):
            preferred, other = dict(candidate), existing
        else:
            preferred, other = dict(existing), candidate
        preferred["alternate_paths"] = sorted({str(value) for item in (preferred, other)
                                              for value in [item.get("best_weights"), *item.get("alternate_paths", [])] if value})
        preferred["model_info"] = {**(other.get("model_info") or {}), **(preferred.get("model_info") or {})}
        for field in ("project_name", "project_root"):
            if not preferred.get(field):
                preferred[field] = other.get(field, "")
        if preferred.get("run") is not None:
            preferred.pop("_training_provenance_loaded", None)
            preferred["training_provenance"] = {}
        result[existing_index] = preferred
    return result

def _mobile_export_dataset_label(dataset_path: str, target: str) -> str:
    text = str(dataset_path or "").strip()
    if not text:
        return "-"
    try:
        return build_dataset_display_ref(text, target_hint=target).id
    except Exception:
        try:
            path = Path(text)
            if path.name.lower() == "data.yaml":
                return path.parent.name or path.name
            return path.name or text
        except Exception:
            return text

def _mobile_export_history_sources(self) -> list[tuple[Path, str, str, str]]:
    sources: list[tuple[Path, str, str, str]] = []
    seen: set[str] = set()

    def add_source(path_value, label: str, *, project_name: str = "", project_root: str = "") -> None:
        if not path_value:
            return
        try:
            path = Path(path_value)
        except Exception:
            return
        inferred_project_name, inferred_project_root = _mobile_export_project_identity_from_path(path)
        project_name = str(project_name or inferred_project_name or "").strip()
        project_root = str(project_root or inferred_project_root or "").strip()
        key = _mobile_export_safe_path_key(path)
        if not key or key in seen:
            return
        seen.add(key)
        sources.append((path, label, project_name, project_root))

    current_history = getattr(self, "history", None)
    add_source(getattr(current_history, "history_dir", None), "Aktywna historia")

    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        project_root = CAMPAIGN.get_active_project_root_dir() if project_name else None
        if project_root:
            add_source(Path(project_root) / "5_training_runs", f"Projekt: {project_name}", project_name=project_name, project_root=str(project_root))
    except Exception:
        pass

    try:
        projects_root = Path(CONFIG.DIR_9_PROJECTS)
        if projects_root.exists() and projects_root.is_dir():
            for project_root in sorted((path for path in projects_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
                add_source(
                    project_root / "5_training_runs",
                    f"Projekt: {project_root.name}",
                    project_name=project_root.name,
                    project_root=str(project_root),
                )
    except Exception:
        pass

    for target in ("plate", "char", "vehicle"):
        try:
            label = self._format_training_target_label(target)
        except Exception:
            label = target
        add_source(CONFIG.get_training_runs_dir(target), f"Globalne: {label}")

    return sources

def _mobile_export_candidates_source_signature(self) -> tuple:
    items: list[tuple] = []

    def add_stat(kind: str, path_value, *extra) -> None:
        try:
            path = Path(path_value)
        except Exception:
            return
        key = _mobile_export_safe_path_key(path)
        try:
            stat = path.stat()
            mtime_ns = int(getattr(stat, "st_mtime_ns", 0) or 0)
            size = int(getattr(stat, "st_size", 0) or 0)
            exists = 1
        except Exception:
            mtime_ns = 0
            size = 0
            exists = 0
        items.append((kind, key, exists, mtime_ns, size, *extra))

    try:
        for history_dir, _label, project_name, _project_root in _mobile_export_history_sources(self):
            history_file = Path(history_dir) / getattr(TrainingHistory, "HISTORY_FILE", "training_history.json")
            add_stat("history", history_file, project_name)
            try:
                runs = json.loads(history_file.read_text(encoding="utf-8-sig")).get("runs", {})
                for run_id, run in runs.items():
                    for path in (run.get("best_weights"), Path(history_dir) / run_id / "train/weights/best.pt"):
                        if path:
                            add_stat("checkpoint", path)
                            for sidecar in _mobile_export_raw_metadata_candidates(Path(path)):
                                add_stat("sidecar", sidecar)
            except (OSError, ValueError):
                pass
    except Exception:
        pass

    try:
        for models_dir, target, _label, project_name, _project_root in _mobile_export_artifact_sources(self):
            add_stat("models", models_dir, target, project_name)
            for path in sorted(Path(models_dir).rglob("*.pt")):
                add_stat("checkpoint", path, target)
                for sidecar in _mobile_export_raw_metadata_candidates(path):
                    add_stat("sidecar", sidecar)
    except Exception:
        pass

    return tuple(items)

def _collect_mobile_export_candidates(self) -> list[dict]:
    cache_signature = _mobile_export_candidates_source_signature(self)
    try:
        cache = getattr(self, "_mobile_export_candidates_cache", None)
        if isinstance(cache, dict) and cache.get("signature") == cache_signature:
            cached_candidates = cache.get("candidates")
            if isinstance(cached_candidates, list):
                return [dict(candidate) for candidate in cached_candidates if isinstance(candidate, dict)]
    except Exception:
        pass

    candidates: list[dict] = []
    seen: set[str] = set()

    current_history = getattr(self, "history", None)
    current_history_key = _mobile_export_safe_path_key(getattr(current_history, "history_dir", ""))

    for history_dir, scope_label, project_name, project_root in _mobile_export_history_sources(self):
        history_key = _mobile_export_safe_path_key(history_dir)
        try:
            history = current_history if history_key and history_key == current_history_key else TrainingHistory(history_dir=history_dir)
        except Exception:
            continue

        try:
            runs = list(history.get_all_runs() or [])
        except Exception:
            runs = []
        try:
            snapshot_index = _mobile_export_run_snapshot_index()
            for local_run in runs:
                run_id = str(getattr(local_run, "id", "") or "").strip()
                if not run_id:
                    continue
                snapshot = _mobile_export_run_like_dict(local_run)
                snapshot.setdefault("id", run_id)
                snapshot["_history_dir"] = str(history_dir)
                existing = snapshot_index.get(run_id)
                if existing and existing.get("_history_dir") != str(history_dir):
                    snapshot_index[f"{history_dir}|{run_id}"] = snapshot
                else:
                    snapshot_index[run_id] = snapshot
        except Exception:
            pass

        for run in runs:
            try:
                if str(getattr(run, "status", "") or "").strip().lower() != TrainingStatus.COMPLETED.value:
                    continue

                target = ""
                try:
                    infer_target = getattr(history, "_infer_run_target", None)
                    if callable(infer_target):
                        target = str(infer_target(run) or "").strip().lower()
                except Exception:
                    target = ""
                if target not in {"plate", "char", "vehicle"}:
                    target = str(self._infer_history_run_target(run) or "").strip().lower()
                if target not in {"plate", "char", "vehicle"}:
                    continue

                local_weights = Path(history_dir) / str(run.id) / "train/weights/best.pt"
                best_weights = local_weights if local_weights.is_file() else self._resolve_history_run_best_weights(run)
                if best_weights is None:
                    continue
                best_key = _mobile_export_safe_path_key(best_weights)
                run_key = f"{history_key}|{str(getattr(run, 'id', '') or '').strip()}|{best_key}"
                if best_key in seen or run_key in seen:
                    continue
                seen.add(best_key)
                seen.add(run_key)

                metric_summary = self._build_history_run_metric_summary(run)
                model_metadata = _mobile_export_read_model_metadata(best_weights)
                model_info = model_metadata.get("info") if isinstance(model_metadata.get("info"), dict) else {}
                dataset_path = str(getattr(run, "dataset_path", "") or "").strip()
                target, target_source, target_conflict = _mobile_export_resolve_candidate_target(
                    fallback_target=target,
                    run_like=run,
                    dataset_path=dataset_path,
                    model_info=model_info,
                )
                if target not in {"plate", "char", "vehicle"}:
                    continue
                role = _mobile_role_from_training_target(target)
                task = _mobile_export_task_from_target_and_info(target, model_info)
                candidate_project_name = str(project_name or "").strip()
                candidate_project_root = str(project_root or "").strip()
                if not candidate_project_name:
                    inferred_project_name, inferred_project_root = _mobile_export_project_identity_from_path(best_weights)
                    candidate_project_name = inferred_project_name
                    candidate_project_root = inferred_project_root
                run_ref = build_run_display_ref(run, kind_hint="training")
                try:
                    model_ref = build_model_display_ref(
                        best_weights,
                        run=run,
                        target_hint=target,
                        source_run_label=run_ref.id,
                    )
                    model_label = model_ref.id
                except Exception:
                    model_label = Path(best_weights).name or "model"

                training_provenance = {}
                total_epochs = _mobile_export_int_or_none(getattr(run, "total_epochs", None))
                if total_epochs is None or total_epochs <= 0:
                    total_epochs = _mobile_export_completed_epoch_count(run)
                dataset_provenance = {}
                candidates.append(
                    {
                        "history_dir": str(history_dir),
                        "scope": scope_label,
                        "project_name": candidate_project_name,
                        "project_root": candidate_project_root,
                        "run": run,
                        "target": target,
                        "target_label": self._format_training_target_label(target),
                        "target_source": target_source,
                        "target_conflict": target_conflict,
                        "role": role,
                        "task": task,
                        "best_weights": Path(best_weights),
                        "run_label": run_ref.id or str(getattr(run, "id", "") or "-"),
                        "model_label": model_label,
                        "model_version": _mobile_export_model_version_label(
                            model_info,
                            target,
                            getattr(run, "base_model", ""),
                            getattr(run, "base_model", ""),
                            getattr(run, "name", ""),
                            best_weights,
                        ),
                        "dataset_label": str(dataset_provenance.get("dataset_id") or "").strip() or _mobile_export_dataset_label(dataset_path, target),
                        "dataset_path": dataset_path,
                        "best_map50": metric_summary.get("best_map50"),
                        "best_map50_95": metric_summary.get("best_map50_95"),
                        "best_epoch": metric_summary.get("best_epoch"),
                        "created_at": str(getattr(run, "created_at", "") or ""),
                        "started_at": str(getattr(run, "started_at", "") or ""),
                        "finished_at": str(getattr(run, "finished_at", "") or ""),
                        "epochs": getattr(run, "epochs", None),
                        "current_epoch": getattr(run, "current_epoch", None),
                        "total_epochs": total_epochs,
                        "total_epochs_known": _mobile_export_bool_or_none(getattr(run, "total_epochs_known", None)) is True,
                        "known_epochs_minimum": _mobile_export_int_or_none(getattr(run, "known_epochs_minimum", None)),
                        "provenance_status": "deferred_until_export",
                        "training_provenance": training_provenance,
                        "batch_size": getattr(run, "batch_size", None),
                        "img_size": getattr(run, "img_size", None),
                        "base_model": getattr(run, "base_model", ""),
                        "file_size_mb": _mobile_export_file_size_mb(best_weights),
                        "model_info": model_info,
                        "model_metadata": model_metadata,
                    }
                )
            except Exception:
                continue

    for models_dir, target, scope_label, project_name, project_root in _mobile_export_artifact_sources(self):
        try:
            root = Path(models_dir)
        except Exception:
            continue
        if not root.exists() or not root.is_dir():
            continue
        try:
            artifact_paths = sorted(root.rglob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        except Exception:
            try:
                artifact_paths = sorted(root.rglob("*.pt"))
            except Exception:
                artifact_paths = []
        for model_path in artifact_paths:
            try:
                best_key = _mobile_export_safe_path_key(model_path)
                if not best_key or best_key in seen:
                    continue
                candidate = _mobile_export_candidate_from_artifact(
                    self,
                    Path(model_path),
                    target,
                    scope_label,
                    project_name=project_name,
                    project_root=project_root,
                )
                seen.add(best_key)
                candidates.append(candidate)
            except Exception:
                continue

    # MP from the Ultralytics/base detector catalog is not produced by our training history,
    # so we explicitly surface already downloaded detector checkpoints as export candidates.
    try:
        local_vehicle_catalog = _mobile_export_local_catalog_model_index(target="vehicle")
        for model_key, info in _mobile_export_ultralytics_detect_catalog(include_runtime_assets=False).items():
            file_name = str((info or {}).get("file") or f"{model_key}.pt").strip()
            local_model = local_vehicle_catalog.get(Path(file_name).name.lower())
            if not local_model or not Path(local_model).exists():
                continue
            best_key = _mobile_export_safe_path_key(local_model)
            if not best_key or best_key in seen:
                continue
            candidate = _mobile_export_candidate_from_artifact(
                self,
                Path(local_model),
                "vehicle",
                "Lokalne modele pojazdów MP",
            )
            seen.add(best_key)
            candidates.append(candidate)
    except Exception:
        pass

    def sort_key(candidate: dict) -> tuple[str, str, str]:
        run = candidate.get("run")
        timestamp = (
            str(candidate.get("finished_at") or "").strip()
            or str(candidate.get("started_at") or "").strip()
            or str(candidate.get("created_at") or "").strip()
        )
        return (timestamp, str(getattr(run, "id", "") or ""), str(candidate.get("model_label") or ""))

    result = sorted(_dedupe_mobile_export_candidates(candidates), key=sort_key, reverse=True)
    try:
        setattr(
            self,
            "_mobile_export_candidates_cache",
            {
                "signature": cache_signature,
                "candidates": [dict(candidate) for candidate in result if isinstance(candidate, dict)],
                "created_at": time.time(),
            },
        )
    except Exception:
        pass
    return result

def _mobile_export_candidate_matches_run(candidate: dict, run) -> bool:
    if not candidate or run is None:
        return False
    candidate_run = candidate.get("run")
    try:
        if candidate_run is run:
            return True
    except Exception:
        pass
    for attr in ("id", "output_dir", "best_weights"):
        left = str(getattr(candidate_run, attr, "") or "").strip()
        right = str(getattr(run, attr, "") or "").strip()
        if left and right and left == right:
            return True
    return False


def _mobile_export_candidate_training_detail_rows(candidate: dict | None) -> list[tuple[str, str]]:
    provenance = _mobile_export_candidate_training_provenance(candidate)
    if not provenance:
        return [
            ("Pochodzenie treningowe", "dane historyczne niepełne"),
            ("Rodowód treningu", "-"),
            ("Prezentacje próbek", "-"),
        ]

    total_known = _mobile_export_bool_or_none(provenance.get("lineage_total_epochs_known"))
    if total_known is None:
        total_known = _mobile_export_bool_or_none(provenance.get("total_epochs_known"))
    stage_count_known = _mobile_export_candidate_lineage_stage_count_known(candidate)
    sample_known = _mobile_export_bool_or_none(provenance.get("sample_presentations_known"))
    stages = _mobile_export_candidate_lineage_stage_count(candidate)
    stage_minimum = _mobile_export_int_or_none(provenance.get("known_stage_count_minimum"))
    if stage_count_known is False:
        stage_label = f"co najmniej {_format_mobile_export_int(stage_minimum or stages)} etapów" if (stage_minimum or stages) > 0 else "-"
    else:
        stage_label = f"{_format_mobile_export_int(stages)} etap" if stages == 1 else (f"{_format_mobile_export_int(stages)} etapów" if stages > 1 else "-")

    run_done = _mobile_export_int_or_none(provenance.get("run_epochs_completed"))
    run_plan = _mobile_export_int_or_none(provenance.get("run_epochs_planned"))
    if run_done is not None and run_plan is not None and run_plan > 0:
        last_stage_label = f"wykonano {_format_mobile_export_int(run_done)} z {_format_mobile_export_int(run_plan)} epok"
    elif run_done is not None:
        last_stage_label = f"wykonano {_format_mobile_export_int(run_done)} epok"
    else:
        last_stage_label = "-"

    total_epochs_label = _mobile_export_known_or_minimum_label(
        provenance.get("lineage_total_epochs") if provenance.get("lineage_total_epochs") is not None else provenance.get("total_epochs"),
        known=total_known,
        minimum=provenance.get("known_epochs_minimum"),
    )
    run_train_images = _format_mobile_export_int(provenance.get("run_train_images"))
    run_presentations = _format_mobile_export_int(provenance.get("run_nominal_sample_presentations"))
    lineage_presentations = _mobile_export_known_or_minimum_label(
        provenance.get("lineage_nominal_sample_presentations"),
        known=sample_known,
        minimum=provenance.get("known_sample_presentations_minimum"),
    )
    best_epoch = _mobile_export_int_or_none(provenance.get("best_epoch"))
    if best_epoch is not None and run_done is not None and run_done > 0:
        best_label = f"epoka {_format_mobile_export_int(best_epoch)} z {_format_mobile_export_int(run_done)} wykonanych"
    elif best_epoch is not None:
        best_label = f"epoka {_format_mobile_export_int(best_epoch)}"
    else:
        best_label = "-"
    best_source = str(provenance.get("best_epoch_source") or "").strip()
    if best_label != "-" and best_source:
        source_labels = {
            "checkpoint": "checkpoint",
            "metrics_history": "metryki",
            "run_state": "stan runu",
            "unknown": "nieznane źródło",
        }
        best_label = f"{best_label} | źródło: {source_labels.get(best_source, best_source)}"

    return [
        ("Pochodzenie treningowe", _mobile_export_provenance_status_label(provenance)),
        ("Rodowód treningu", stage_label),
        ("Ostatni etap", last_stage_label),
        ("Epoki łącznie", total_epochs_label),
        ("Obrazy train ostatniego", run_train_images),
        ("Prezentacje ostatniego", run_presentations),
        ("Prezentacje łącznie", lineage_presentations),
        ("Snapshot danych", _mobile_export_provenance_capture_label(provenance.get("provenance_capture"))),
        ("Najlepszy checkpoint", best_label),
    ]


def _mobile_export_candidate_detail_rows(candidate: dict) -> list[tuple[str, str]]:
    if not candidate:
        return []
    run = candidate.get("run")
    info = candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {}
    marker = _mobile_export_target_marker(candidate)
    role_labels = {
        "MP": "MP: pojazdy",
        "MT": "MT: tablice",
        "MZ": "MZ: znaki",
    }
    role_label = role_labels.get(marker, marker or "-")
    params_m, params_source = _mobile_export_candidate_params_millions(candidate)
    params_label = _format_mobile_export_params(params_m, source=params_source, include_source=True)
    training_rows = _mobile_export_candidate_training_detail_rows(candidate)
    if run is None:
        return [
            ("Model eksportowany", str(candidate.get("model_label") or "-")),
            ("Rola w paczce", role_label),
            ("Źródło roli", _mobile_export_target_source_label(candidate)),
            ("Projekt", _mobile_export_project_label(candidate, empty="-")),
            ("Źródło", "Gotowy plik .pt z listy modeli"),
            ("Wersja YOLO", str(candidate.get("model_version") or "-")),
            ("Parametry modelu", params_label),
            ("Rodzina modelu", str(info.get("architecture_label") or info.get("yolo_variant") or candidate.get("model_version") or "-")),
            ("Dataset treningowy", str(candidate.get("dataset_label") or "-")),
            *training_rows,
            ("Obraz wejściowy", f"{_format_mobile_export_int(candidate.get('img_size'))} px"),
            ("Najlepsza epoka", _format_mobile_export_int(candidate.get("best_epoch"))),
            ("mAP50", _format_mobile_export_metric(candidate.get("best_map50"))),
            ("mAP50-95", _format_mobile_export_metric(candidate.get("best_map50_95"))),
            ("Rozmiar pliku", f"{candidate.get('file_size_mb'):.2f} MB" if candidate.get("file_size_mb") is not None else "-"),
            ("Utworzono / zmodyfikowano", _format_mobile_export_datetime(candidate.get("created_at"))),
        ]
    return [
        ("Model eksportowany", str(candidate.get("model_label") or "-")),
        ("Run treningu", f"{candidate.get('run_label') or '-'} | {getattr(run, 'name', '') or '-'}"),
        ("Rola w paczce", role_label),
        ("Źródło roli", _mobile_export_target_source_label(candidate)),
        ("Projekt", _mobile_export_project_label(candidate, empty="-")),
        ("Wersja YOLO", str(candidate.get("model_version") or "-")),
        ("Parametry modelu", params_label),
        ("Rodzina modelu", str(info.get("architecture_label") or info.get("yolo_variant") or candidate.get("model_version") or "-")),
        ("Dataset treningowy", str(candidate.get("dataset_label") or "-")),
        *training_rows,
        ("Obraz wejściowy", f"{_format_mobile_export_int(getattr(run, 'img_size', None))} px"),
        ("Najlepsza epoka", _format_mobile_export_int(candidate.get("best_epoch"))),
        ("mAP50", _format_mobile_export_metric(candidate.get("best_map50"))),
        ("mAP50-95", _format_mobile_export_metric(candidate.get("best_map50_95"))),
        ("Rozmiar pliku", f"{candidate.get('file_size_mb'):.2f} MB" if candidate.get("file_size_mb") is not None else "-"),
        ("Utworzono", _format_mobile_export_datetime(candidate.get("created_at"))),
        ("Koniec treningu", _format_mobile_export_datetime(candidate.get("finished_at"))),
    ]

def _mobile_export_manifest_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _mobile_export_manifest_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_mobile_export_manifest_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _mobile_export_candidate_manifest_snapshot(candidate: dict | None) -> dict:
    if not isinstance(candidate, dict) or not candidate:
        return {}
    run = candidate.get("run")
    run_snapshot = {}
    try:
        if hasattr(run, "to_dict"):
            run_snapshot = run.to_dict()
    except Exception:
        run_snapshot = {}
    checkpoint = ""
    checkpoint_name = ""
    try:
        checkpoint_path = Path(candidate.get("best_weights"))
        checkpoint = str(checkpoint_path)
        checkpoint_name = checkpoint_path.name
    except Exception:
        checkpoint = str(candidate.get("best_weights") or "")
    marker = _mobile_export_target_marker(candidate)
    params_m, params_source = _mobile_export_candidate_params_millions(candidate)
    training_provenance = _mobile_export_candidate_training_provenance(candidate)
    snapshot = {
        "schema": "alpr.export.candidate_snapshot.v1",
        "marker": marker,
        "role": str(candidate.get("role") or ""),
        "task": str(candidate.get("task") or ""),
        "target": str(candidate.get("target") or ""),
        "target_label": str(candidate.get("target_label") or ""),
        "target_source": str(candidate.get("target_source") or ""),
        "target_conflict": str(candidate.get("target_conflict") or ""),
        "scope": str(candidate.get("scope") or ""),
        "project_name": str(candidate.get("project_name") or ""),
        "project_root": str(candidate.get("project_root") or ""),
        "project": {
            "name": str(candidate.get("project_name") or ""),
            "root": str(candidate.get("project_root") or ""),
        },
        "candidate_iid": str(candidate.get("iid") or ""),
        "model_label": str(candidate.get("model_label") or ""),
        "model_version": str(candidate.get("model_version") or ""),
        "checkpoint": checkpoint,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": _file_sha256(Path(checkpoint)) if checkpoint else "",
        "alternate_paths": list(candidate.get("alternate_paths") or []),
        "run_label": str(candidate.get("run_label") or ""),
        "history_dir": str(candidate.get("history_dir") or ""),
        "dataset_label": str(candidate.get("dataset_label") or ""),
        "dataset_path": str(candidate.get("dataset_path") or ""),
        "base_model": str(candidate.get("base_model") or ""),
        "img_size": candidate.get("img_size"),
        "batch_size": candidate.get("batch_size"),
        "epochs_last_run": candidate.get("epochs"),
        "current_epoch_last_run": training_provenance.get("run_epochs_completed", candidate.get("current_epoch")),
        "total_epochs": _mobile_export_candidate_total_epochs(candidate),
        "total_epochs_known": _mobile_export_bool_or_none(candidate.get("total_epochs_known")) if "total_epochs_known" in candidate else None,
        "known_epochs_minimum": candidate.get("known_epochs_minimum"),
        "provenance_status": str(candidate.get("provenance_status") or ""),
        "lineage_stage_count": training_provenance.get("lineage_stage_count") if isinstance(training_provenance, dict) else None,
        "lineage_stage_count_known": training_provenance.get("lineage_stage_count_known") if isinstance(training_provenance, dict) else None,
        "known_stage_count_minimum": training_provenance.get("known_stage_count_minimum") if isinstance(training_provenance, dict) else None,
        "run_train_images": training_provenance.get("run_train_images") if isinstance(training_provenance, dict) else None,
        "run_nominal_sample_presentations": training_provenance.get("run_nominal_sample_presentations") if isinstance(training_provenance, dict) else None,
        "lineage_nominal_sample_presentations": training_provenance.get("lineage_nominal_sample_presentations") if isinstance(training_provenance, dict) else None,
        "sample_presentations_known": training_provenance.get("sample_presentations_known") if isinstance(training_provenance, dict) else None,
        "known_sample_presentations_minimum": training_provenance.get("known_sample_presentations_minimum") if isinstance(training_provenance, dict) else None,
        "best_epoch_source": str(training_provenance.get("best_epoch_source") or "") if isinstance(training_provenance, dict) else "",
        "provenance_capture": str(training_provenance.get("provenance_capture") or "") if isinstance(training_provenance, dict) else "",
        "training_provenance": training_provenance if isinstance(training_provenance, dict) else {},
        "best_epoch": training_provenance.get("best_epoch"),
        "best_map50": candidate.get("best_map50"),
        "best_map50_95": candidate.get("best_map50_95"),
        "created_at": str(candidate.get("created_at") or ""),
        "started_at": str(candidate.get("started_at") or ""),
        "finished_at": str(candidate.get("finished_at") or ""),
        "file_size_mb": candidate.get("file_size_mb"),
        "parameters_millions": params_m,
        "parameters_source": params_source,
        "model_info": candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {},
        "model_metadata": candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {},
        "history_snapshot": candidate.get("history_snapshot") if isinstance(candidate.get("history_snapshot"), dict) else {},
        "run_snapshot": run_snapshot,
    }
    return _mobile_export_manifest_safe_value(snapshot)

def _build_exported_model_metadata(self, run, target: str, source_path: Path, destination_path: Path) -> dict:
    metric_summary = self._build_history_run_metric_summary(run)
    source_metadata = _mobile_export_read_model_metadata(Path(source_path))
    project_name, project_root = _mobile_export_project_identity_from_path(source_path)
    training_payload = _mobile_export_build_training_provenance(
        run,
        target=target,
        checkpoint=source_path,
        dataset_path=str(getattr(run, "dataset_path", "") or ""),
        model_metadata=source_metadata,
        include_dataset_fingerprint=True,
    )
    total_epochs = _mobile_export_provenance_display_total(training_payload)
    if total_epochs <= 0:
        total_epochs = _mobile_export_total_epochs_for_run_like(run)
    run_dict = {}
    try:
        if hasattr(run, "to_dict"):
            run_dict = run.to_dict()
    except Exception:
        run_dict = {}
    if not run_dict:
        run_dict = {
            key: getattr(run, key, None)
            for key in (
                "id",
                "name",
                "created_at",
                "status",
                "dataset_path",
                "base_model",
                "epochs",
                "batch_size",
                "img_size",
                "device",
                "lr0",
                "current_epoch",
                "best_map50",
                "best_map50_95",
                "output_dir",
                "best_weights",
                "last_weights",
                "started_at",
                "finished_at",
                "paused_at",
                "metrics_history",
                "error_message",
                "report_html",
                "plots_dir",
            )
        }
    if total_epochs > 0 and _mobile_export_bool_or_none(training_payload.get("total_epochs_known")) is not False:
        run_dict["total_epochs"] = total_epochs
    elif total_epochs > 0:
        run_dict["known_epochs_minimum"] = total_epochs
        run_dict["total_epochs_known"] = False
    for key, value in {
        "target": target,
        "target_label": self._format_training_target_label(target),
        "status": str(getattr(run, "status", "") or ""),
        "device": str(getattr(run, "device", "") or ""),
        "lr0": getattr(run, "lr0", None),
        "report_html": str(getattr(run, "report_html", "") or ""),
        "plots_dir": str(getattr(run, "plots_dir", "") or ""),
    }.items():
        if value not in (None, ""):
            training_payload.setdefault(key, value)
    if _mobile_export_bool_or_none(training_payload.get("total_epochs_known")) is False:
        training_payload["total_epochs"] = None

    validation_ok = False
    validation_message = ""
    model_info = {}
    identity = ""
    try:
        validation_ok, validation_message, model_info = validate_model_file(destination_path)
        identity = format_yolo_model_identity(model_info, include_ultralytics_version=True)
    except Exception as e:
        validation_message = str(e)
        model_info = {}

    return self._json_safe_training_value(
        {
            "schema": "auto_annotation_tool.exported_model.v1",
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "exported_for": "free_mode",
            "target": target,
            "target_label": self._format_training_target_label(target),
            "model": {
                "file_name": destination_path.name,
                "path": str(destination_path),
                "directory": str(destination_path.parent),
                "identity": identity,
                "validation_ok": validation_ok,
                "validation_message": validation_message,
                "info": model_info,
            },
            "source": {
                "best_weights": str(source_path),
                "run_output_dir": str(getattr(run, "output_dir", "") or ""),
                "training_history_dir": str(getattr(getattr(self, "history", None), "history_dir", "") or ""),
                "project_name": project_name,
                "project_root": project_root,
            },
            "project": {
                "name": project_name,
                "root": project_root,
            },
            "training": training_payload,
            "metrics": {
                "best_map50": metric_summary.get("best_map50"),
                "best_map50_95": metric_summary.get("best_map50_95"),
                "best_epoch": metric_summary.get("best_epoch"),
                "latest": metric_summary.get("latest") or {},
                "best_row": metric_summary.get("best_row") or {},
            },
            "run_snapshot": run_dict,
        }
    )

def _export_selected_run_model_to_free_mode(self):
    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run treningu z historii."
        )

    target = self._infer_history_run_target(run)
    if target not in {"plate", "char", "vehicle"}:
        return messagebox.showerror(
            "Nie rozpoznano toru",
            "Nie mogę jednoznacznie ustalić, czy wybrany run dotyczy tablic, znaków czy pojazdów.\n\n"
            "Eksport do modeli trybu swobodnego jest dostępny tylko dla rozpoznanych runów YOLO."
        )

    best_weights = self._resolve_history_run_best_weights(run)
    if best_weights is None:
        return messagebox.showerror(
            "Brak best.pt",
            "Wybrany run nie ma dostępnego pliku best.pt.\n\n"
            "Eksport jest możliwy dopiero po treningu, który zapisał najlepsze wagi modelu."
        )

    try:
        target_dir = Path(CONFIG.get_trained_models_dir(target))
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception("Nie udało się przygotować katalogu eksportu modelu")
        return messagebox.showerror(
            "Błąd katalogu eksportu",
            f"Nie udało się przygotować katalogu modeli trybu swobodnego:\n{e}"
        )

    destination = self._build_free_mode_model_export_path(run, target, target_dir, best_weights)
    metadata_path = destination.with_suffix(".json")

    if destination.exists() or metadata_path.exists():
        overwrite = messagebox.askyesno(
            "Model już istnieje",
            "W katalogu modeli trybu swobodnego istnieje już eksport dla tego runu.\n\n"
            f"Model: {destination.name}\n"
            f"Parametry: {metadata_path.name}\n\n"
            "Nadpisać te pliki?"
        )
        if not overwrite:
            return

    try:
        shutil.copy2(best_weights, destination)
        metadata = self._build_exported_model_metadata(run, target, best_weights, destination)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        model_meta = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
        write_model_metadata_sidecar(
            destination,
            model_meta.get("info") if isinstance(model_meta.get("info"), dict) else {},
            validation_ok=bool(model_meta.get("validation_ok", True)),
            validation_message=str(model_meta.get("validation_message") or ""),
            extra={
                "exported_metadata_path": str(metadata_path),
                "target": target,
                "metrics": metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {},
                "training": metadata.get("training") if isinstance(metadata.get("training"), dict) else {},
            },
        )
    except Exception as e:
        logger.exception("Nie udało się wyeksportować modelu do trybu swobodnego")
        return messagebox.showerror(
            "Błąd eksportu modelu",
            f"Nie udało się wyeksportować modelu:\n{e}"
        )

    try:
        self._append_train_log(
            f"[EXPORT] best.pt wyeksportowany do modeli trybu swobodnego: {destination}"
        )
    except Exception:
        pass

    return messagebox.showinfo(
        "Model wyeksportowany",
        "Model jest teraz dostępny w trybie swobodnym.\n\n"
        f"Tor: {self._format_training_target_label(target)}\n"
        f"Model: {destination}\n"
        f"Parametry: {metadata_path}"
    )

def _export_selected_run_model_to_mobile_package_legacy(self):
    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run treningu z historii."
        )

    target = self._infer_history_run_target(run)
    if target not in {"plate", "char", "vehicle"}:
        return messagebox.showerror(
            "Nie rozpoznano toru",
            "Nie mogę jednoznacznie ustalić, czy wybrany run dotyczy tablic, znaków czy pojazdów."
        )

    best_weights = self._resolve_history_run_best_weights(run)
    if best_weights is None:
        return messagebox.showerror(
            "Brak best.pt",
            "Model mobilny można wyeksportować dopiero z ukończonego runu, który ma plik best.pt."
        )

    role = _mobile_role_from_training_target(target)
    task = "pose" if role == "plate" else "detect"
    img_size = int(getattr(run, "img_size", CONFIG.DEFAULT_IMG_SIZE) or CONFIG.DEFAULT_IMG_SIZE)
    default_destination = self._build_mobile_model_export_path(run, target, best_weights)
    calibration_candidate = {
        "run": run,
        "target": target,
        "role": role,
        "task": task,
        "dataset_path": str(getattr(run, "dataset_path", "") or ""),
        "model_label": str(getattr(run, "name", "") or Path(best_weights).stem),
    }
    default_calibration = _mobile_export_default_calibration_path(calibration_candidate)

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(self.frame)
    try:
        self.app.style_dialog_window(
            dialog,
            title="Eksport modelu mobilnego",
            geometry="860x640",
            parent=self.frame,
        )
    except Exception:
        dialog.title("Eksport modelu mobilnego")
        dialog.geometry("860x640")
    dialog.transient(self.frame)
    try:
        dialog.grab_release()
    except Exception:
        pass
    dialog.minsize(760, 560)

    bg = palette.get("panel", "#252526")
    card_bg = blend_hex_colors(bg, palette.get("accent", "#4f8de3"), 0.045)
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4f8de3")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f1c40f")
    error = palette.get("error", "#e74c3c")
    border = blend_hex_colors(accent, bg, 0.38)

    root = tk.Frame(
        dialog,
        bg=bg,
        padx=18,
        pady=16,
        highlightthickness=1,
        highlightbackground=border,
    )
    root.pack(fill=tk.BOTH, expand=True)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(2, weight=1)

    def _mobile_export_text_for_width(text: str, pixel_width: int, *, min_chars: int = 10) -> str:
        raw = str(text or "-")
        width_chars = max(min_chars, min(180, int(max(40, pixel_width) / 7.0)))
        wrapped_lines: list[str] = []
        for paragraph in raw.splitlines() or [""]:
            if not paragraph.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=width_chars,
                    break_long_words=True,
                    break_on_hyphens=True,
                    replace_whitespace=False,
                    drop_whitespace=True,
                )
                or [paragraph]
            )
        return "\n".join(wrapped_lines)

    def _bind_mobile_export_dynamic_wrap(
        label: tk.Label,
        container=None,
        *,
        min_wrap: int = 90,
        margin_ratio: float = 0.045,
        margin_min: int = 14,
        margin_max: int = 64,
        textvariable: tk.Variable | None = None,
    ) -> tk.Label:
        target = container or label.master
        state = {"after": None, "width": 0, "raw": None, "wrap": 0}
        if textvariable is None and not hasattr(label, "_mobile_export_raw_text"):
            try:
                label._mobile_export_raw_text = str(label.cget("text") or "-")
            except Exception:
                pass

        def _raw_text() -> str:
            if textvariable is not None:
                try:
                    return str(textvariable.get() or "-")
                except Exception:
                    return "-"
            return str(getattr(label, "_mobile_export_raw_text", label.cget("text")) or "-")

        def _apply(width_hint: int = 0) -> None:
            state["after"] = None
            try:
                width = int(width_hint or target.winfo_width() or label.winfo_width() or 1)
                if width <= 2:
                    return
                raw = _raw_text()
                if abs(width - int(state.get("width") or 0)) < 3 and raw == state.get("raw"):
                    return
                margin = max(margin_min, min(margin_max, int(width * margin_ratio)))
                wrap_px = max(min_wrap, width - margin)
                display_text = _mobile_export_text_for_width(raw, wrap_px)
                updates = {}
                if int(state.get("wrap") or 0) != int(wrap_px):
                    updates["wraplength"] = wrap_px
                if label.cget("text") != display_text:
                    updates["text"] = display_text
                if updates:
                    label.configure(**updates)
                state["width"] = width
                state["raw"] = raw
                state["wrap"] = int(wrap_px)
            except Exception:
                pass

        def _refresh(event=None) -> None:
            try:
                width = int(getattr(event, "width", 0) or target.winfo_width() or label.winfo_width() or 1)
                raw = _raw_text()
                if abs(width - int(state.get("width") or 0)) < 3 and raw == state.get("raw"):
                    return
                pending = state.get("after")
                if pending:
                    try:
                        label.after_cancel(pending)
                    except Exception:
                        pass
                state["after"] = label.after(70, lambda w=width: _apply(w))
            except Exception:
                pass

        if textvariable is not None:
            try:
                textvariable.trace_add("write", lambda *_args: _refresh())
            except Exception:
                pass
        try:
            target.bind("<Configure>", _refresh, add="+")
        except Exception:
            pass
        try:
            label.after_idle(lambda: _apply())
        except Exception:
            pass
        return label

    tk.Label(
        root,
        text="Eksport modelu mobilnego",
        bg=bg,
        fg=fg,
        font=("Segoe UI", 14, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    tk.Label(
        root,
        text=(
            "Tworzysz zwykły plik ZIP z rozszerzeniem .alprmodel: manifest.json, warianty modelu "
            "i sumy SHA-256. Android nie importuje surowego best.pt."
        ),
        bg=bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=800,
    ).grid(row=1, column=0, sticky="ew", pady=(4, 12))

    form = tk.Frame(root, bg=card_bg, padx=14, pady=12, highlightthickness=1, highlightbackground=blend_hex_colors(accent, bg, 0.38))
    form.grid(row=2, column=0, sticky="nsew")
    form.grid_columnconfigure(1, weight=1)

    litert_var = tk.BooleanVar(value=True)
    onnx_var = tk.BooleanVar(value=False)
    int8_var = tk.BooleanVar(value=False)
    onnx_int8_var = tk.BooleanVar(value=False)
    ncnn_var = tk.BooleanVar(value=False)
    imgsz_var = tk.IntVar(value=img_size)
    conf_var = tk.DoubleVar(value=CONFIG.DEFAULT_CONFIDENCE)
    iou_var = tk.DoubleVar(value=CONFIG.DEFAULT_IOU)
    destination_var = tk.StringVar(value=str(default_destination))
    calibration_var = tk.StringVar(value=default_calibration)
    status_var = tk.StringVar(value="Gotowe do sprawdzenia zależności eksportu.")

    def row_label(row: int, text: str) -> None:
        tk.Label(
            form,
            text=text,
            bg=card_bg,
            fg=muted,
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
        ).grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=5)

    def readonly_value(row: int, text: str) -> None:
        tk.Label(
            form,
            text=text,
            bg=card_bg,
            fg=fg,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=620,
        ).grid(row=row, column=1, sticky="ew", pady=5)

    row_label(0, "Run")
    readonly_value(0, f"{getattr(run, 'id', '') or '-'} | {getattr(run, 'name', '') or '-'}")
    row_label(1, "Rola")
    readonly_value(1, f"{role} / {task} | tor: {self._format_training_target_label(target)}")
    row_label(2, "Checkpoint")
    readonly_value(2, str(best_weights))

    row_label(3, "Formaty i kwantyzacja")
    formats_frame = tk.Frame(form, bg=card_bg)
    formats_frame.grid(row=3, column=1, sticky="ew", pady=4)
    for idx, (text, var) in enumerate((
        ("LiteRT/TFLite FP32", litert_var),
        ("ONNX FP32", onnx_var),
        ("LiteRT/TFLite INT8", int8_var),
        ("ONNX INT8", onnx_int8_var),
        ("NCNN", ncnn_var),
    )):
        cb = ttk.Checkbutton(formats_frame, text=text, variable=var)
        cb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=(0, 18), pady=3)

    row_label(4, "Parametry")
    params = tk.Frame(form, bg=card_bg)
    params.grid(row=4, column=1, sticky="ew", pady=4)
    for col in range(6):
        params.grid_columnconfigure(col, weight=1 if col in {1, 3, 5} else 0)
    tk.Label(params, text="imgsz", bg=card_bg, fg=muted).grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=128, to=2048, increment=32, textvariable=imgsz_var, width=8).grid(row=0, column=1, sticky="w", padx=(0, 18))
    tk.Label(params, text="confidence", bg=card_bg, fg=muted).grid(row=0, column=2, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=conf_var, width=8, format="%.2f").grid(row=0, column=3, sticky="w", padx=(0, 18))
    tk.Label(params, text="IoU", bg=card_bg, fg=muted).grid(row=0, column=4, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=iou_var, width=8, format="%.2f").grid(row=0, column=5, sticky="w")

    row_label(5, "Kalibracja INT8")
    calibration_row = tk.Frame(form, bg=card_bg)
    calibration_row.grid(row=5, column=1, sticky="ew", pady=4)
    calibration_row.grid_columnconfigure(0, weight=1)
    calibration_suggestions = _mobile_export_calibration_yaml_candidates(calibration_candidate)
    calibration_labels = [str(item.get("label") or item.get("path") or "") for item in calibration_suggestions]
    calibration_by_label = {
        str(item.get("label") or item.get("path") or ""): item
        for item in calibration_suggestions
        if str(item.get("label") or item.get("path") or "")
    }
    calibration_choice_var = tk.StringVar(value=calibration_labels[0] if calibration_labels else "")
    if calibration_labels:
        calibration_combo = ttk.Combobox(
            calibration_row,
            textvariable=calibration_choice_var,
            values=calibration_labels,
            state="readonly",
            height=min(8, max(3, len(calibration_labels))),
        )
        calibration_combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        def apply_calibration_suggestion(_event=None) -> None:
            item = calibration_by_label.get(str(calibration_choice_var.get() or ""))
            if item:
                calibration_var.set(str(item.get("path") or ""))

        calibration_combo.bind("<<ComboboxSelected>>", apply_calibration_suggestion, add="+")
    else:
        tk.Label(
            calibration_row,
            text="Brak zgodnych propozycji. Przy INT8 wskaż data.yaml ręcznie.",
            bg=card_bg,
            fg=warning,
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
    ttk.Entry(calibration_row, textvariable=calibration_var).grid(row=1, column=0, sticky="ew", padx=(0, 8))

    def choose_calibration() -> None:
        selected = filedialog.askopenfilename(
            title="Wybierz data.yaml do kalibracji INT8",
            parent=dialog,
            filetypes=(("YOLO data.yaml", "data.yaml"), ("YAML", "*.yaml *.yml"), ("Wszystkie pliki", "*.*")),
        )
        if selected:
            calibration_var.set(selected)

    ttk.Button(calibration_row, text="Wybierz data.yaml", command=choose_calibration).grid(row=1, column=1, sticky="e")

    status_box = tk.Frame(root, bg=blend_hex_colors(bg, accent, 0.03), padx=12, pady=10, highlightthickness=1, highlightbackground=blend_hex_colors(accent, bg, 0.45))
    status_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    status_box.grid_columnconfigure(0, weight=1)
    status_label = tk.Label(
        status_box,
        textvariable=status_var,
        bg=status_box["bg"],
        fg=fg,
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    status_label.grid(row=0, column=0, sticky="ew")
    progress_var = tk.DoubleVar(value=0.0)
    progress_percent_var = tk.StringVar(value="0%")
    progress_row = tk.Frame(status_box, bg=status_box["bg"])
    progress_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    progress_row.grid_columnconfigure(0, weight=1)
    progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100.0, variable=progress_var)
    progress.grid(row=0, column=0, sticky="ew")
    tk.Label(
        progress_row,
        textvariable=progress_percent_var,
        bg=status_box["bg"],
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        width=5,
        anchor=tk.E,
    ).grid(row=0, column=1, sticky="e", padx=(10, 0))

    actions = tk.Frame(root, bg=bg)
    actions.grid(row=4, column=0, sticky="ew", pady=(14, 0))
    actions.grid_columnconfigure(0, weight=1)

    def build_request() -> MobileExportRequest:
        formats = []
        if litert_var.get() or int8_var.get():
            formats.append("litert")
        if onnx_var.get() or onnx_int8_var.get():
            formats.append("onnx")
        if ncnn_var.get():
            formats.append("ncnn")
        format_quantizations = {}
        if "litert" in formats:
            litert_quantizations = []
            if litert_var.get():
                litert_quantizations.append("fp32")
            if int8_var.get():
                litert_quantizations.append("int8")
            format_quantizations["litert"] = tuple(dict.fromkeys(litert_quantizations or ["fp32"]))
        if "onnx" in formats:
            onnx_quantizations = []
            if onnx_var.get():
                onnx_quantizations.append("fp32")
            if onnx_int8_var.get():
                onnx_quantizations.append("int8")
            format_quantizations["onnx"] = tuple(dict.fromkeys(onnx_quantizations or ["fp32"]))
        if "ncnn" in formats:
            format_quantizations["ncnn"] = ("fp32",)
        quantizations = []
        if litert_var.get():
            quantizations.append("fp32")
        if int8_var.get():
            quantizations.append("int8")
        if onnx_var.get():
            quantizations.append("fp32")
        if onnx_int8_var.get():
            quantizations.append("int8")
        if ncnn_var.get():
            quantizations.append("fp32")
        destination = Path(str(destination_var.get() or default_destination).strip())
        if destination.suffix.lower() != ".alprmodel":
            destination = destination.with_suffix(".alprmodel")
        destination.parent.mkdir(parents=True, exist_ok=True)
        calibration = Path(str(calibration_var.get()).strip()) if str(calibration_var.get()).strip() else None
        model_id_seed = f"{role}-{getattr(run, 'id', '') or best_weights.stem}"
        export_metadata = self._build_mobile_export_metadata(run, target, best_weights)
        model_payload = dict(export_metadata.get("model") or {}) if isinstance(export_metadata, dict) else {}
        model_payload["display_id"] = str(candidate.get("model_label") or best_weights.stem)
        model_payload["architecture_label"] = str(candidate.get("model_version") or "")
        params_m, params_source = _mobile_export_candidate_params_millions(candidate)
        if params_m is not None:
            model_payload["parameters_millions"] = round(float(params_m), 3)
            model_payload["parameters_source"] = params_source or "unknown"
            if params_source == "metadata":
                model_payload["parameter_count"] = int(round(float(params_m) * 1_000_000))
        export_metadata["model"] = model_payload
        training_payload = dict(export_metadata.get("training") or {}) if isinstance(export_metadata, dict) else {}
        for key in ("dataset_path", "base_model", "img_size", "epochs", "current_epoch", "batch_size", "started_at", "finished_at"):
            value = candidate.get(key)
            if value not in (None, ""):
                training_payload.setdefault(key, value)
        candidate_provenance = _mobile_export_candidate_training_provenance(candidate)
        for key in (
            "lineage_total_epochs",
            "lineage_total_epochs_known",
            "lineage_stage_count",
            "lineage_stage_count_known",
            "known_stage_count_minimum",
            "run_train_images",
            "run_nominal_sample_presentations",
            "lineage_nominal_sample_presentations",
            "sample_presentations_known",
            "known_sample_presentations_minimum",
            "best_epoch_source",
            "provenance_capture",
        ):
            if isinstance(candidate_provenance, dict) and candidate_provenance.get(key) not in (None, ""):
                training_payload.setdefault(key, candidate_provenance.get(key))
        known_total = _mobile_export_provenance_known_total(training_payload) or _mobile_export_provenance_known_total(candidate_provenance)
        if known_total is not None:
            training_payload["total_epochs"] = known_total
            training_payload["total_epochs_known"] = True
        else:
            minimum = _mobile_export_candidate_total_epochs(candidate)
            if minimum > 0:
                training_payload.setdefault("known_epochs_minimum", minimum)
            training_payload["total_epochs"] = None
            training_payload["total_epochs_known"] = False
            training_payload.setdefault("provenance_status", candidate.get("provenance_status") or "partial")
        export_metadata["training"] = training_payload
        export_metadata["candidate"] = _mobile_export_candidate_manifest_snapshot(candidate)
        if role == "vehicle":
            vehicle_payload = _mobile_export_vehicle_class_payload(candidate, "")
            export_metadata["vehicle_detection"] = vehicle_payload
            runtime_payload = dict(export_metadata.get("runtime") or {}) if isinstance(export_metadata, dict) else {}
            runtime_payload["vehicle_filter_mode"] = "include"
            runtime_payload["vehicle_filter_labels"] = list(vehicle_payload.get("include_labels") or [])
            runtime_payload["vehicle_filter_indices"] = list(vehicle_payload.get("include_class_indices") or [])
            export_metadata["runtime"] = runtime_payload
        export_metadata["export_settings"] = self._json_safe_training_value(
            {
                "schema": "alpr.export.settings_snapshot.v1",
                "formats": list(formats),
                "quantizations": list(dict.fromkeys(quantizations)),
                "format_quantizations": {key: list(value) for key, value in format_quantizations.items()},
                "image_size": int(imgsz_var.get() or img_size),
                "confidence_threshold": max(0.0, min(1.0, float(conf_var.get() or CONFIG.DEFAULT_CONFIDENCE))),
                "iou_threshold": max(0.0, min(1.0, float(iou_var.get() or CONFIG.DEFAULT_IOU))),
                "calibration_data": str(calibration or ""),
                "calibration_choice": str(calibration_choice_var.get() or ""),
            }
        )
        return MobileExportRequest(
            checkpoint=best_weights,
            destination=destination,
            role=role,
            formats=tuple(formats),
            image_size=int(imgsz_var.get() or img_size),
            quantizations=tuple(dict.fromkeys(quantizations)),
            format_quantizations=format_quantizations,
            calibration_data=calibration,
            confidence_threshold=max(0.0, min(1.0, float(conf_var.get() or CONFIG.DEFAULT_CONFIDENCE))),
            iou_threshold=max(0.0, min(1.0, float(iou_var.get() or CONFIG.DEFAULT_IOU))),
            model_id=self._safe_model_export_slug(model_id_seed, fallback=f"{role}-model")[:80],
            name=f"{self._format_training_target_label(target)} | {getattr(run, 'id', '') or best_weights.stem}",
            version="1",
            metadata=export_metadata,
        )

    exporter = MobileModelExporter()
    package_exporter = MobileAlprPackageExporter(exporter)

    def set_status(text: str, tone: str = "info") -> None:
        colors = {
            "success": success,
            "warning": warning,
            "error": error,
            "info": fg,
        }
        try:
            status_label.configure(fg=colors.get(tone, fg))
        except Exception:
            pass
        status_var.set(text)

    def run_preflight(show_dialog: bool = True) -> bool:
        try:
            request = build_request()
            problems = exporter.preflight(request)
        except Exception as exc:
            problems = [str(exc)]
        if problems:
            preflight_ready_state["ready"] = False
            text = (
                "Eksport nie może jeszcze ruszyć.\n"
                + "\n".join(f"- {item}" for item in problems)
                + "\n\nZainstaluj zależności: pip install -r requirements-mobile-export.txt"
            )
            set_status(text, "warning")
            if show_dialog:
                messagebox.showwarning("Sprawdzenie gotowości eksportu", text, parent=dialog)
            sync_primary_button()
            return False
        preflight_ready_state["ready"] = True
        set_status("Model gotowy do eksportu mobilnego.", "success")
        sync_primary_button()
        return True

    def ask_destination() -> Path | None:
        current = self._build_mobile_model_export_path(run, target, best_weights)
        initial_dir = current.parent
        if str(initial_dir) == ".":
            initial_dir = Path.cwd()
        selected = filedialog.asksaveasfilename(
            title="Eksportuj model mobilny (.alprmodel)",
            parent=dialog,
            initialdir=str(initial_dir),
            initialfile=current.name,
            defaultextension=".alprmodel",
            filetypes=(("Model mobilny lub pakiet ALPR", "*.alprmodel"), ("ZIP", "*.zip"), ("Wszystkie pliki", "*.*")),
        )
        if not selected:
            return None
        path = Path(selected)
        if path.suffix.lower() != ".alprmodel":
            path = path.with_suffix(".alprmodel")
        return path

    worker_state = {"running": False}
    preflight_ready_state = {"ready": False}
    primary_button = None

    def sync_primary_button() -> None:
        button = primary_button
        if button is None:
            return
        try:
            if worker_state.get("running"):
                button.configure(text="Eksport trwa...", state=tk.DISABLED)
            elif preflight_ready_state.get("ready"):
                button.configure(text="Eksportuj model mobilny (.alprmodel)", state=tk.NORMAL)
            else:
                button.configure(text="Sprawdź gotowość eksportu", state=tk.NORMAL)
        except Exception:
            pass

    def mark_preflight_dirty(*_args) -> None:
        if worker_state.get("running"):
            return
        preflight_ready_state["ready"] = False
        set_status("Ustawienia zmienione. Najpierw sprawdź gotowość eksportu.", "info")
        sync_primary_button()

    def set_running(running: bool) -> None:
        worker_state["running"] = running
        for widget in (primary_button, close_button):
            try:
                widget.configure(state=(tk.DISABLED if running else tk.NORMAL))
            except Exception:
                pass
        sync_primary_button()

    def export_package() -> None:
        if worker_state.get("running"):
            return
        destination = ask_destination()
        if destination is None:
            set_status("Eksport anulowany. Plik nie został zapisany.", "info")
            return
        destination_var.set(str(destination))
        if not run_preflight(show_dialog=True):
            return
        request = build_request()
        progress_var.set(0.0)
        progress_percent_var.set("0%")
        set_running(True)
        set_status(
            _mobile_export_verbose_progress_message(
                0.0,
                "Eksportuję model mobilny. To może potrwać, szczególnie dla LiteRT/TFLite.",
            ),
            "info",
        )

        def progress_cb(percent, message: str) -> None:
            def apply_update() -> None:
                if percent is None:
                    progress.configure(mode="indeterminate")
                    progress.start(18)
                    progress_percent_var.set("...")
                else:
                    progress.stop()
                    progress.configure(mode="determinate")
                    progress_value = _mobile_export_progress_value(percent)
                    progress_var.set(progress_value)
                    progress_percent_var.set(_mobile_export_progress_label(progress_value))
                set_status(_mobile_export_verbose_progress_message(percent, message), "info")
            try:
                dialog.after(0, apply_update)
            except Exception:
                pass

        def worker() -> None:
            try:
                package_path = exporter.export(request, progress=progress_cb)

                def done() -> None:
                    set_running(False)
                    progress.stop()
                    progress.configure(mode="determinate")
                    progress_var.set(100.0)
                    progress_percent_var.set("100%")
                    set_status(f"Model mobilny gotowy: {package_path}", "success")
                    try:
                        self._append_train_log(f"[MOBILE EXPORT] Utworzono model mobilny .alprmodel: {package_path}")
                    except Exception:
                        pass
                    messagebox.showinfo(
                        "Model mobilny gotowy",
                        f"Utworzono model mobilny dla klienta Android:\n{package_path}",
                        parent=dialog,
                    )

                dialog.after(0, done)
            except Exception as exc:
                logger.exception("Nie udało się wyeksportować modelu mobilnego")
                error_text = str(exc)

                def failed() -> None:
                    set_running(False)
                    progress.stop()
                    progress.configure(mode="determinate")
                    set_status(f"Nie udało się wyeksportować modelu mobilnego: {error_text}", "error")
                    messagebox.showerror("Błąd eksportu mobilnego", error_text, parent=dialog)

                try:
                    dialog.after(0, failed)
                except Exception:
                    pass

        threading.Thread(target=worker, name="mobile-model-export", daemon=True).start()

    def run_primary_action() -> None:
        if worker_state.get("running"):
            return
        if preflight_ready_state.get("ready"):
            export_package()
            return
        run_preflight(show_dialog=True)

    primary_button = ttk.Button(actions, text="Sprawdź gotowość eksportu", command=run_primary_action)
    primary_button.grid(row=0, column=1, sticky="e", padx=(0, 8), ipadx=10, ipady=4)
    close_button = ttk.Button(actions, text="Zamknij", command=dialog.destroy)
    close_button.grid(row=0, column=2, sticky="e", ipadx=8, ipady=4)

    for export_option_var in (
        litert_var,
        onnx_var,
        int8_var,
        onnx_int8_var,
        ncnn_var,
        imgsz_var,
        conf_var,
        iou_var,
        calibration_var,
    ):
        try:
            export_option_var.trace_add("write", mark_preflight_dirty)
        except Exception:
            pass
    sync_primary_button()
    try:
        dialog.wait_window()
    except Exception:
        pass


def _open_mobile_model_export_center(self, initial_run=None):
    palette = getattr(self.app, "palette", {})
    bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4f8de3")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f1c40f")
    error = palette.get("error", "#e74c3c")
    card_bg = blend_hex_colors(bg, accent, 0.045)
    border = blend_hex_colors(accent, bg, 0.38)

    try:
        existing_dialog = getattr(self, "_mobile_export_center_dialog", None)
        if existing_dialog is not None and existing_dialog.winfo_exists():
            try:
                setattr(existing_dialog, "_aat_skip_window_recovery", True)
            except Exception:
                pass
            try:
                setter = getattr(getattr(self, "app", None), "set_free_mode_assistant_context_override", None)
                if callable(setter):
                    setter("mobile_export_center", get_mobile_export_assistant_context(), owner=existing_dialog)
            except Exception:
                pass
            try:
                existing_dialog.deiconify()
            except Exception:
                pass
            try:
                existing_dialog.lift()
            except Exception:
                pass
            try:
                existing_dialog.focus_force()
            except Exception:
                pass
            return existing_dialog
    except Exception:
        try:
            self._mobile_export_center_dialog = None
        except Exception:
            pass

    dialog_parent = getattr(getattr(self, "app", None), "root", None) or self.frame
    loader_window = None
    loader_progress_canvas = None
    loader_status_var = tk.StringVar(value="Przygotowuję eksport mobilny...")
    loader_progress_var = tk.DoubleVar(value=4.0)
    dialog_build_state = {"dialog_created": False}

    def _paint_mobile_export_loader_progress() -> None:
        try:
            if loader_progress_canvas is None or not loader_progress_canvas.winfo_exists():
                return
            width = max(300, int(loader_progress_canvas.winfo_width() or 420))
            height = max(12, int(loader_progress_canvas.winfo_height() or 14))
            value = _mobile_export_progress_value(loader_progress_var.get())
            fill_width = max(0, int(width * (value / 100.0)))
            loader_progress_canvas.delete("all")
            loader_progress_canvas.create_rectangle(
                0,
                0,
                width,
                height,
                fill=blend_hex_colors(bg, fg, 0.08),
                outline=blend_hex_colors(border, bg, 0.30),
                width=1,
            )
            if fill_width > 1:
                loader_progress_canvas.create_rectangle(
                    1,
                    1,
                    max(1, fill_width - 1),
                    max(1, height - 1),
                    fill=blend_hex_colors(accent, success, 0.35),
                    outline="",
                )
        except Exception:
            pass

    def _show_mobile_export_loader() -> None:
        nonlocal loader_window, loader_progress_canvas
        try:
            loader = tk.Toplevel(dialog_parent)
            loader.withdraw()
            loader.overrideredirect(True)
            try:
                loader.attributes("-topmost", True)
            except Exception:
                pass
            loader.configure(bg=bg)
            card = tk.Frame(
                loader,
                bg=card_bg,
                padx=18,
                pady=15,
                highlightthickness=1,
                highlightbackground=border,
            )
            card.pack(fill=tk.BOTH, expand=True)
            tk.Label(
                card,
                text="Ładowanie eksportu mobilnego",
                bg=card_bg,
                fg=fg,
                font=("Segoe UI", 11, "bold"),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(0, 8))
            loader_progress_canvas = tk.Canvas(
                card,
                width=420,
                height=14,
                bg=card_bg,
                highlightthickness=0,
                bd=0,
            )
            loader_progress_canvas.pack(fill=tk.X)
            tk.Label(
                card,
                textvariable=loader_status_var,
                bg=card_bg,
                fg=muted,
                font=("Segoe UI", 9),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(8, 0))
            loader.update_idletasks()
            width = max(420, int(card.winfo_reqwidth() or 460))
            height = max(116, int(card.winfo_reqheight() or 128))
            screen_w = int(loader.winfo_screenwidth() or 1360)
            screen_h = int(loader.winfo_screenheight() or 840)
            x = max(0, int((screen_w - width) / 2))
            y = max(0, int((screen_h - height) / 2))
            loader.geometry(f"{width}x{height}+{x}+{y}")
            loader.deiconify()
            loader.lift()
            loader_window = loader
            _paint_mobile_export_loader_progress()
            # Przed utworzeniem glownego Toplevel wymuszamy realne namalowanie
            # splasha. Pozniej nie robimy juz globalnych update'ow podczas
            # skladania modala, bo Tk potrafi wtedy pokazac biale, puste okno.
            loader.update()
        except Exception:
            loader_window = None

    def _set_mobile_export_loader(message: str, percent: float, *, force_paint: bool = False) -> None:
        try:
            loader_status_var.set(str(message or "Ładowanie eksportu mobilnego..."))
            loader_progress_var.set(_mobile_export_progress_value(percent))
            _paint_mobile_export_loader_progress()
            if loader_window is not None and loader_window.winfo_exists():
                if force_paint or not bool(dialog_build_state.get("dialog_created")):
                    loader_window.update()
        except Exception:
            pass

    def _close_mobile_export_loader() -> None:
        nonlocal loader_window
        loader = loader_window
        loader_window = None
        if loader is None:
            return
        try:
            if loader.winfo_exists():
                loader.destroy()
        except Exception:
            pass

    _show_mobile_export_loader()
    _set_mobile_export_loader("Skanuję modele i historię treningów...", 8.0)

    candidates = _collect_mobile_export_candidates(self)
    if False and not candidates:
        _close_mobile_export_loader()
        return messagebox.showwarning(
            "Brak kandydatów",
            "Nie znalazłem ukończonego treningu z plikiem best.pt.\n\n"
            "Eksport mobilny wymaga gotowego modelu.",
        )

    for idx, candidate in enumerate(candidates):
        candidate["iid"] = f"mobile_export_candidate_{idx}"

    initial_iid = ""
    for candidate in candidates:
        if _mobile_export_candidate_matches_run(candidate, initial_run):
            initial_iid = str(candidate.get("iid") or "")
            break

    _set_mobile_export_loader("Buduję okno eksportu...", 18.0)

    dialog_build_state["dialog_created"] = True
    dialog = tk.Toplevel(dialog_parent, bg=bg)
    try:
        setattr(dialog, "_aat_skip_window_recovery", True)
    except Exception:
        pass
    try:
        self._mobile_export_center_dialog = dialog
    except Exception:
        pass
    try:
        app_obj = getattr(self, "app", None)
        if app_obj is not None:
            app_obj._mobile_export_center_dialog = dialog
    except Exception:
        pass
    try:
        dialog.withdraw()
    except Exception:
        pass
    try:
        dialog.configure(bg=bg)
    except Exception:
        pass
    dialog.title("Centrum eksportu mobilnego")
    try:
        screen_width = int(dialog.winfo_screenwidth() or 1360)
        screen_height = int(dialog.winfo_screenheight() or 840)
    except Exception:
        screen_width = 1360
        screen_height = 840
    dialog_width = min(1340, max(1140, screen_width - 90))
    dialog_height = min(800, max(680, screen_height - 110))
    dialog_x = max(24, int((screen_width - dialog_width) / 2))
    dialog_y = max(24, int((screen_height - dialog_height) / 2))
    try:
        if dialog_parent is not None and bool(dialog_parent.winfo_ismapped()):
            parent_x = int(dialog_parent.winfo_rootx())
            parent_y = int(dialog_parent.winfo_rooty())
            parent_w = int(dialog_parent.winfo_width())
            parent_h = int(dialog_parent.winfo_height())
            if parent_w > 120 and parent_h > 120:
                dialog_x = parent_x + int((parent_w - dialog_width) / 2)
                dialog_y = parent_y + int((parent_h - dialog_height) / 2)
    except Exception:
        pass
    dialog_x = max(24, min(dialog_x, screen_width - dialog_width - 24))
    dialog_y = max(24, min(dialog_y, screen_height - dialog_height - 24))
    dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}")
    try:
        dialog.resizable(True, True)
    except Exception:
        pass
    try:
        dialog.minsize(1100, 660)
    except Exception:
        pass

    windowing_system = ""
    try:
        windowing_system = str(dialog.tk.call("tk", "windowingsystem") or "")
    except Exception:
        windowing_system = ""
    if windowing_system == "win32":
        try:
            dialog.attributes("-toolwindow", False)
        except Exception:
            pass
    # Nie grupujemy tego okna z oknem glownym: na Windows/Tk potrafi to
    # wywolywac minimizacje dialogu po zwyklych akcjach w tabelach.

    # This export center behaves like a modeless tool window, not like a small modal prompt.
    try:
        dialog.grab_release()
    except Exception:
        pass
    if windowing_system == "win32":
        try:
            dialog.wm_transient("")
        except Exception:
            pass

    def _raise_mobile_export_center(event=None) -> None:
        try:
            if getattr(event, "widget", None) is not None and getattr(event, "widget", None).winfo_toplevel() is not dialog:
                return
        except Exception:
            pass
        try:
            dialog.lift(dialog_parent)
        except Exception:
            try:
                dialog.lift()
            except Exception:
                pass

    # Podnosimy okno przy pokazaniu, ale nie przy kazdym kliknieciu w srodku:
    # klik w naglowki Treeview ma sortowac, nie zmieniac stanu okna.
    def _clear_mobile_export_center_dialog(event=None) -> None:
        try:
            if getattr(event, "widget", None) is not dialog:
                return
            if getattr(self, "_mobile_export_center_dialog", None) is dialog:
                self._mobile_export_center_dialog = None
            app_obj = getattr(self, "app", None)
            if app_obj is not None and getattr(app_obj, "_mobile_export_center_dialog", None) is dialog:
                app_obj._mobile_export_center_dialog = None
            clearer = getattr(getattr(self, "app", None), "clear_free_mode_assistant_context_override", None)
            if callable(clearer):
                clearer("mobile_export_center")
        except Exception:
            pass

    try:
        dialog.bind("<Destroy>", _clear_mobile_export_center_dialog, add="+")
    except Exception:
        pass
    try:
        dialog.bind(
            "<Destroy>",
            lambda event: _close_mobile_export_loader() if getattr(event, "widget", None) is dialog else None,
            add="+",
        )
    except Exception:
        pass

    perf_enabled = str(os.environ.get("AAT_MOBILE_EXPORT_PERF", "")).strip().lower() in {"1", "true", "yes", "on"}
    try:
        perf_enabled = bool(perf_enabled or getattr(self, "_mobile_export_perf_enabled", False))
    except Exception:
        pass
    perf_state = {"last_flush": time.perf_counter(), "events": {}}
    window_move_tracking_flag = str(os.environ.get("AAT_MOBILE_EXPORT_TRACK_WINDOW_MOVE", "1")).strip().lower()
    window_move_tracking_enabled = window_move_tracking_flag in {"1", "true", "yes", "on"}
    if str(os.environ.get("AAT_DISABLE_MOBILE_EXPORT_WINDOW_MOVE_TRACKING", "")).strip().lower() in {"1", "true", "yes", "on"}:
        window_move_tracking_enabled = False

    def _mobile_export_perf_flush(*, force: bool = False) -> None:
        if not perf_enabled:
            return
        try:
            events = perf_state.get("events") or {}
            if not events:
                return
            now = time.perf_counter()
            last_flush = float(perf_state.get("last_flush") or now)
            if not force and (now - last_flush) < 2.5:
                return
            chunks = []
            for name, entry in sorted(events.items()):
                count = int(entry.get("count") or 0)
                if count <= 0:
                    continue
                total_ms = float(entry.get("total_ms") or 0.0)
                avg_ms = total_ms / max(1, count)
                max_ms = float(entry.get("max_ms") or 0.0)
                last = entry.get("last") if isinstance(entry.get("last"), dict) else {}
                meta = ""
                if last:
                    meta = " " + ",".join(f"{key}={value}" for key, value in last.items())[:160]
                chunks.append(f"{name}: n={count} avg={avg_ms:.1f}ms max={max_ms:.1f}ms{meta}")
            if chunks:
                logger.info(f"[MOBILE EXPORT PERF] {' | '.join(chunks)}")
            events.clear()
            perf_state["last_flush"] = now
        except Exception:
            pass

    def _mobile_export_perf_record(name: str, elapsed_ms: float = 0.0, **meta) -> None:
        if not perf_enabled:
            return
        try:
            events = perf_state.setdefault("events", {})
            key = str(name or "event")
            entry = events.setdefault(key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last": {}})
            elapsed = max(0.0, float(elapsed_ms or 0.0))
            entry["count"] = int(entry.get("count") or 0) + 1
            entry["total_ms"] = float(entry.get("total_ms") or 0.0) + elapsed
            entry["max_ms"] = max(float(entry.get("max_ms") or 0.0), elapsed)
            if meta:
                entry["last"] = {key: value for key, value in meta.items()}
            _mobile_export_perf_flush()
        except Exception:
            pass

    def _mobile_export_perf_timer(name: str):
        if not perf_enabled:
            return lambda **_meta: None
        started = time.perf_counter()

        def _finish(**meta) -> None:
            _mobile_export_perf_record(name, (time.perf_counter() - started) * 1000.0, **meta)

        return _finish

    def _mobile_export_perf_destroy(event=None) -> None:
        try:
            if getattr(event, "widget", None) is dialog:
                try:
                    _mobile_export_perf_record(
                        "mobile_export_window_destroy",
                        0.0,
                        mapped=int(bool(dialog.winfo_ismapped())),
                    )
                except Exception:
                    pass
                _mobile_export_perf_flush(force=True)
        except Exception:
            pass

    if perf_enabled:
        try:
            dialog.bind("<Destroy>", _mobile_export_perf_destroy, add="+")
        except Exception:
            pass

    event_loop_state = {"after": None, "last": time.perf_counter(), "closed": False}
    stall_probe_state = {"running": False, "last_report": 0.0}

    def _mobile_export_count_widgets(widget) -> tuple[int, dict[str, int]]:
        counts: dict[str, int] = {}

        def walk(current) -> int:
            try:
                klass = str(current.winfo_class() or current.__class__.__name__)
            except Exception:
                klass = current.__class__.__name__
            counts[klass] = int(counts.get(klass) or 0) + 1
            total = 1
            try:
                children = list(current.winfo_children())
            except Exception:
                children = []
            for child in children:
                total += walk(child)
            return total

        return walk(widget), counts

    def _mobile_export_log_widget_tree() -> None:
        try:
            total, counts = _mobile_export_count_widgets(dialog)
            top = ", ".join(f"{name}:{count}" for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8])
            logger.info(f"[MOBILE EXPORT PERF] widget_tree total={total} top={top}")
        except Exception:
            pass

    def _mobile_export_event_loop_tick() -> None:
        if event_loop_state.get("closed"):
            return
        now = time.perf_counter()
        previous = float(event_loop_state.get("last") or now)
        event_loop_state["last"] = now
        gap_ms = (now - previous) * 1000.0
        if gap_ms > 180.0:
            try:
                width = int(dialog.winfo_width() or 0)
                height = int(dialog.winfo_height() or 0)
            except Exception:
                width = 0
                height = 0
            _mobile_export_perf_record(
                "event_loop_gap",
                gap_ms,
                size=f"{width}x{height}",
            )
        try:
            event_loop_state["after"] = dialog.after(100, _mobile_export_event_loop_tick)
        except Exception:
            pass

    def _mobile_export_stall_probe() -> None:
        main_ident = threading.main_thread().ident
        while stall_probe_state.get("running") and not event_loop_state.get("closed"):
            time.sleep(0.12)
            try:
                last_tick = float(event_loop_state.get("last") or 0.0)
                if last_tick <= 0.0:
                    continue
                gap_ms = (time.perf_counter() - last_tick) * 1000.0
                if gap_ms < 320.0:
                    continue
                now = time.perf_counter()
                if (now - float(stall_probe_state.get("last_report") or 0.0)) < 1.6:
                    continue
                stall_probe_state["last_report"] = now
                frame = sys._current_frames().get(main_ident) if main_ident is not None else None
                stack = "".join(traceback.format_stack(frame, limit=18)).rstrip() if frame is not None else "-"
                logger.info(f"[MOBILE EXPORT STALL] gap={gap_ms:.1f}ms stack:\n{stack}")
            except Exception:
                pass

    def _start_mobile_export_event_loop_monitor() -> None:
        if event_loop_state.get("closed"):
            return
        event_loop_state["last"] = time.perf_counter()
        try:
            event_loop_state["after"] = dialog.after(100, _mobile_export_event_loop_tick)
            _mobile_export_perf_record("event_loop_monitor", 0.0, state="started")
        except Exception:
            pass
        try:
            stall_probe_enabled = str(os.environ.get("AAT_MOBILE_EXPORT_STALL_PROBE", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if stall_probe_enabled and not stall_probe_state.get("running"):
                stall_probe_state["running"] = True
                threading.Thread(
                    target=_mobile_export_stall_probe,
                    name="mobile-export-stall-probe",
                    daemon=True,
                ).start()
        except Exception:
            pass

    def _mobile_export_stop_event_loop_monitor(event=None) -> None:
        try:
            if getattr(event, "widget", None) is not dialog:
                return
            event_loop_state["closed"] = True
            stall_probe_state["running"] = False
            pending = event_loop_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
            _mobile_export_perf_flush(force=True)
        except Exception:
            pass

    if perf_enabled:
        try:
            dialog.bind("<Destroy>", _mobile_export_stop_event_loop_monitor, add="+")
        except Exception:
            pass

    root = tk.Frame(dialog, bg=bg, padx=12, pady=8)
    root.pack(fill=tk.BOTH, expand=True)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(2, weight=1)
    _set_mobile_export_loader("Układam szkielet okna...", 22.0)
    wrap_refreshers: list = []
    wrap_refresh_state = {
        "after": None,
        "size": (0, 0),
        "position": (None, None),
    }
    move_stream_state = {
        "after": None,
        "last_time": None,
        "last_pos": None,
        "events": 0,
        "total_dt": 0.0,
        "max_dt": 0.0,
        "distance": 0.0,
        "max_jump": 0.0,
    }
    window_config_state = {
        "size": (0, 0),
        "position": (None, None),
        "moving": False,
        "move_after": None,
    }
    child_window_motion_state = {"moving": False, "owner": None}
    native_move_state = {"moving": False, "owner": None, "after": None}
    native_move_hooks: dict[str, dict] = {}

    def _mobile_export_window_is_moving() -> bool:
        if not window_move_tracking_enabled:
            return False
        return bool(
            native_move_state.get("moving")
            or window_config_state.get("moving")
            or child_window_motion_state.get("moving")
        )

    try:
        setattr(dialog, "_aat_window_is_moving_callback", _mobile_export_window_is_moving)
    except Exception:
        pass
    try:
        setter = getattr(getattr(self, "app", None), "set_free_mode_assistant_context_override", None)
        if callable(setter):
            setter("mobile_export_center", get_mobile_export_assistant_context(), owner=dialog)
    except Exception:
        pass

    def _finish_mobile_export_window_move() -> None:
        window_config_state["moving"] = False
        window_config_state["move_after"] = None

    def _set_mobile_export_native_move_state(target_window, moving: bool) -> None:
        if not window_move_tracking_enabled:
            return
        try:
            pending = native_move_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
                native_move_state["after"] = None
        except Exception:
            pass
        if moving:
            native_move_state["moving"] = True
            native_move_state["owner"] = target_window
            if target_window is dialog:
                window_config_state["moving"] = True
            else:
                child_window_motion_state["moving"] = True
                child_window_motion_state["owner"] = target_window
            return

        def _finish_native_move() -> None:
            native_move_state["after"] = None
            if native_move_state.get("owner") is target_window:
                native_move_state["moving"] = False
                native_move_state["owner"] = None
            if target_window is dialog:
                _finish_mobile_export_window_move()
            elif child_window_motion_state.get("owner") is target_window:
                child_window_motion_state["moving"] = False
                child_window_motion_state["owner"] = None

        try:
            native_move_state["after"] = dialog.after(80, _finish_native_move)
        except Exception:
            _finish_native_move()

    def _flush_window_move_stream() -> None:
        move_stream_state["after"] = None
        events = int(move_stream_state.get("events") or 0)
        if events > 0:
            total_dt = float(move_stream_state.get("total_dt") or 0.0)
            _mobile_export_perf_record(
                "native_move_stream",
                0.0,
                events=events,
                avg_dt=f"{(total_dt / max(1, events)):.1f}",
                max_dt=f"{float(move_stream_state.get('max_dt') or 0.0):.1f}",
                distance=f"{float(move_stream_state.get('distance') or 0.0):.1f}",
                max_jump=f"{float(move_stream_state.get('max_jump') or 0.0):.1f}",
            )
        move_stream_state["events"] = 0
        move_stream_state["total_dt"] = 0.0
        move_stream_state["max_dt"] = 0.0
        move_stream_state["distance"] = 0.0
        move_stream_state["max_jump"] = 0.0
        move_stream_state["last_time"] = None
        move_stream_state["last_pos"] = None

    def _flush_window_move_stream_on_destroy(event=None) -> None:
        try:
            if getattr(event, "widget", None) is dialog:
                _flush_window_move_stream()
        except Exception:
            pass

    if perf_enabled:
        try:
            dialog.bind("<Destroy>", _flush_window_move_stream_on_destroy, add="+")
        except Exception:
            pass

    def _record_window_move_stream(position: tuple) -> None:
        try:
            x, y = position
            if x is None or y is None:
                return
            now = time.perf_counter()
            previous_time = move_stream_state.get("last_time")
            previous_pos = move_stream_state.get("last_pos")
            if previous_time is not None and previous_pos is not None:
                dt_ms = max(0.0, (now - float(previous_time)) * 1000.0)
                if dt_ms > 1000.0:
                    if int(move_stream_state.get("events") or 0) > 0:
                        _flush_window_move_stream()
                    _mobile_export_perf_record(
                        "native_move_stream_idle_break",
                        0.0,
                        idle=f"{dt_ms:.1f}",
                    )
                    move_stream_state["last_time"] = now
                    move_stream_state["last_pos"] = (x, y)
                    pending = move_stream_state.get("after")
                    if pending:
                        try:
                            dialog.after_cancel(pending)
                        except Exception:
                            pass
                    move_stream_state["after"] = dialog.after(650, _flush_window_move_stream)
                    return
                dx = float(x) - float(previous_pos[0])
                dy = float(y) - float(previous_pos[1])
                jump = (dx * dx + dy * dy) ** 0.5
                move_stream_state["events"] = int(move_stream_state.get("events") or 0) + 1
                move_stream_state["total_dt"] = float(move_stream_state.get("total_dt") or 0.0) + dt_ms
                move_stream_state["max_dt"] = max(float(move_stream_state.get("max_dt") or 0.0), dt_ms)
                move_stream_state["distance"] = float(move_stream_state.get("distance") or 0.0) + jump
                move_stream_state["max_jump"] = max(float(move_stream_state.get("max_jump") or 0.0), jump)
            move_stream_state["last_time"] = now
            move_stream_state["last_pos"] = (x, y)
            pending = move_stream_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
            move_stream_state["after"] = dialog.after(650, _flush_window_move_stream)
        except Exception:
            pass

    def _record_toplevel_window_configure(event=None) -> None:
        started = time.perf_counter()
        try:
            if event is not None and getattr(event, "widget", None) is not dialog:
                return
            size = (
                int(getattr(event, "width", 0) or dialog.winfo_width() or 0),
                int(getattr(event, "height", 0) or dialog.winfo_height() or 0),
            )
            position = (
                getattr(event, "x", None),
                getattr(event, "y", None),
            )
            previous_size = window_config_state.get("size")
            previous_position = window_config_state.get("position")
            size_changed = size != previous_size
            position_changed = position != previous_position
            window_config_state["size"] = size
            window_config_state["position"] = position
            if position_changed and not size_changed:
                window_config_state["moving"] = True
                pending = window_config_state.get("move_after")
                if pending:
                    try:
                        dialog.after_cancel(pending)
                    except Exception:
                        pass
                window_config_state["move_after"] = dialog.after(180, _finish_mobile_export_window_move)
            if perf_enabled and position_changed and not size_changed:
                _record_window_move_stream(position)
            _mobile_export_perf_record(
                "window_configure",
                (time.perf_counter() - started) * 1000.0,
                size=f"{size[0]}x{size[1]}",
                pos=f"{position[0]},{position[1]}",
                size_changed=int(bool(size_changed)),
                position_changed=int(bool(position_changed)),
            )
        except Exception:
            pass

    def _apply_mobile_export_win32_window_styles(target_window=None, *, window_key: str = "main") -> None:
        """Reduce Win32 repaint churn during native window moves without hiding content."""
        target_window = target_window or dialog
        if windowing_system != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            try:
                if not target_window.winfo_exists():
                    return
            except Exception:
                return

            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(int(target_window.winfo_id()))
            GWL_STYLE = -16
            GWL_EXSTYLE = -20
            WS_CLIPCHILDREN = 0x02000000
            WS_CLIPSIBLINGS = 0x04000000
            WS_EX_COMPOSITED = 0x02000000
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            LONG_PTR = ctypes.c_ssize_t

            get_window_long_ptr = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
            set_window_long_ptr = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
            try:
                get_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int]
                get_window_long_ptr.restype = LONG_PTR
                set_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
                set_window_long_ptr.restype = LONG_PTR
                user32.SetWindowPos.argtypes = [
                    wintypes.HWND,
                    wintypes.HWND,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    wintypes.UINT,
                ]
                user32.SetWindowPos.restype = wintypes.BOOL
            except Exception:
                pass

            changed = False
            try:
                style = int(get_window_long_ptr(hwnd, GWL_STYLE) or 0)
                styled = style | WS_CLIPCHILDREN | WS_CLIPSIBLINGS
                if styled != style:
                    set_window_long_ptr(hwnd, GWL_STYLE, LONG_PTR(styled))
                    changed = True
            except Exception:
                pass

            composited_flag = str(
                os.environ.get(
                    "AAT_MOBILE_EXPORT_WIN32_COMPOSITED_STYLE",
                    os.environ.get("AAT_ENABLE_WIN32_COMPOSITED", ""),
                )
            ).strip().lower()
            if composited_flag in {"1", "true", "yes", "on"}:
                try:
                    ex_style = int(get_window_long_ptr(hwnd, GWL_EXSTYLE) or 0)
                    styled_ex = ex_style | WS_EX_COMPOSITED
                    if styled_ex != ex_style:
                        set_window_long_ptr(hwnd, GWL_EXSTYLE, LONG_PTR(styled_ex))
                        changed = True
                except Exception:
                    pass

            if changed:
                try:
                    flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
                    user32.SetWindowPos(hwnd, wintypes.HWND(0), 0, 0, 0, 0, flags)
                except Exception:
                    pass
            _mobile_export_perf_record(
                "win32_window_styles",
                0.0,
                window=str(window_key or "main"),
                changed=int(bool(changed)),
            )
        except Exception as exc:
            logger.debug(f"Nie udalo sie ustawic stylow Win32 okna eksportu: {exc}")

    def _apply_mobile_export_native_drag_surface(target_window=None, *, window_key: str = "main") -> None:
        """Make Windows/DWM move the export dialog as a composed surface, not as many repainting Tk children."""
        target_window = target_window or dialog
        if windowing_system != "win32":
            return
        surface_flag = str(os.environ.get("AAT_MOBILE_EXPORT_ALPHA_SURFACE", "")).strip().lower()
        if surface_flag not in {"1", "true", "yes", "on"}:
            return
        try:
            alpha_raw = str(os.environ.get("AAT_MOBILE_EXPORT_ALPHA_SURFACE_VALUE", "0.995")).strip().replace(",", ".")
            alpha_value = float(alpha_raw)
        except Exception:
            alpha_value = 0.995
        alpha_value = max(0.90, min(0.999, alpha_value))
        try:
            if not target_window.winfo_exists():
                return
            target_window.attributes("-alpha", alpha_value)
            _mobile_export_perf_record(
                "native_drag_surface",
                0.0,
                window=str(window_key or "main"),
                alpha=f"{alpha_value:.3f}",
            )
        except Exception as exc:
            logger.debug(f"Nie udalo sie wlaczyc kompozytowanej powierzchni okna eksportu: {exc}")

    def _install_mobile_export_win32_move_state_hook(target_window=None, *, hook_key: str = "main") -> None:
        target_window = target_window or dialog
        if windowing_system != "win32":
            return
        if not window_move_tracking_enabled:
            return
        if str(os.environ.get("AAT_DISABLE_MOBILE_EXPORT_NATIVE_MOVE_STATE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        hook = native_move_hooks.setdefault(str(hook_key or "main"), {"installed": False})
        if hook.get("installed"):
            return
        try:
            import ctypes
            from ctypes import wintypes

            try:
                if not target_window.winfo_exists():
                    return
            except Exception:
                return

            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(int(target_window.winfo_id()))
            GWLP_WNDPROC = -4
            WM_ENTERSIZEMOVE = 0x0231
            WM_EXITSIZEMOVE = 0x0232
            WM_NCDESTROY = 0x0082
            LONG_PTR = ctypes.c_ssize_t
            WNDPROC = ctypes.WINFUNCTYPE(
                LONG_PTR,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            get_window_long_ptr = user32.GetWindowLongPtrW
            set_window_long_ptr = user32.SetWindowLongPtrW
            get_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long_ptr.restype = LONG_PTR
            set_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
            set_window_long_ptr.restype = LONG_PTR
            user32.CallWindowProcW.argtypes = [
                LONG_PTR,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.CallWindowProcW.restype = LONG_PTR
            old_proc = get_window_long_ptr(hwnd, GWLP_WNDPROC)
            if not old_proc:
                return

            @WNDPROC
            def _window_proc(hwnd_arg, msg, wparam, lparam):
                try:
                    if msg == WM_ENTERSIZEMOVE:
                        _set_mobile_export_native_move_state(target_window, True)
                    elif msg == WM_EXITSIZEMOVE:
                        _set_mobile_export_native_move_state(target_window, False)
                    elif msg == WM_NCDESTROY:
                        _set_mobile_export_native_move_state(target_window, False)
                        try:
                            set_window_long_ptr(hwnd_arg, GWLP_WNDPROC, old_proc)
                        except Exception:
                            pass
                        return user32.CallWindowProcW(old_proc, hwnd_arg, msg, wparam, lparam)
                except Exception:
                    pass
                return user32.CallWindowProcW(old_proc, hwnd_arg, msg, wparam, lparam)

            previous = set_window_long_ptr(hwnd, GWLP_WNDPROC, ctypes.cast(_window_proc, ctypes.c_void_p).value)
            if previous:
                hook.update(
                    {
                        "installed": True,
                        "old_proc": old_proc,
                        "proc": _window_proc,
                    }
                )
                _mobile_export_perf_record("native_move_state_hook", 0.0, window=str(hook_key or "main"))
        except Exception as exc:
            logger.debug(f"Nie udalo sie wlaczyc lekkiego hooka przesuwania okna eksportu: {exc}")

    try:
        dialog.after_idle(lambda: _apply_mobile_export_win32_window_styles(dialog, window_key="main"))
    except Exception:
        pass
    try:
        dialog.after_idle(lambda: _apply_mobile_export_native_drag_surface(dialog, window_key="main"))
    except Exception:
        pass
    try:
        dialog.after_idle(lambda: _install_mobile_export_win32_move_state_hook(dialog, hook_key="main"))
    except Exception:
        pass

    def _bind_mobile_export_child_window_motion(child_window, *, name: str = "child_window_configure") -> None:
        if windowing_system == "win32":
            try:
                child_window.grab_release()
            except Exception:
                pass
            try:
                child_window.wm_transient("")
            except Exception:
                pass
            try:
                child_window.attributes("-toolwindow", False)
            except Exception:
                pass
        try:
            child_window.after_idle(
                lambda w=child_window, key=name: _apply_mobile_export_win32_window_styles(w, window_key=key)
            )
        except Exception:
            pass
        try:
            child_window.after_idle(
                lambda w=child_window, key=name: _apply_mobile_export_native_drag_surface(w, window_key=key)
            )
        except Exception:
            pass
        try:
            child_window.after_idle(
                lambda w=child_window, key=name: _install_mobile_export_win32_move_state_hook(w, hook_key=key)
            )
        except Exception:
            pass
        try:
            child_window.after_idle(
                lambda w=child_window, key=name: _install_mobile_export_win32_move_guard(w, guard_key=key)
            )
        except Exception:
            pass
        if not window_move_tracking_enabled:
            return
        state = {"size": (0, 0), "position": (None, None), "after": None}

        def _finish_child_window_move() -> None:
            if child_window_motion_state.get("owner") is child_window:
                child_window_motion_state["moving"] = False
                child_window_motion_state["owner"] = None
            state["after"] = None

        def _record_child_window_configure(event=None) -> None:
            started = time.perf_counter()
            try:
                if event is not None and getattr(event, "widget", None) is not child_window:
                    return
                size = (
                    int(getattr(event, "width", 0) or child_window.winfo_width() or 0),
                    int(getattr(event, "height", 0) or child_window.winfo_height() or 0),
                )
                position = (
                    getattr(event, "x", None),
                    getattr(event, "y", None),
                )
                previous_size = state.get("size")
                previous_position = state.get("position")
                size_changed = size != previous_size
                position_changed = position != previous_position
                state["size"] = size
                state["position"] = position
                if position_changed and not size_changed:
                    child_window_motion_state["moving"] = True
                    child_window_motion_state["owner"] = child_window
                    pending = state.get("after")
                    if pending:
                        try:
                            child_window.after_cancel(pending)
                        except Exception:
                            pass
                    state["after"] = child_window.after(220, _finish_child_window_move)
                _mobile_export_perf_record(
                    name,
                    (time.perf_counter() - started) * 1000.0,
                    size=f"{size[0]}x{size[1]}",
                    pos=f"{position[0]},{position[1]}",
                    size_changed=int(bool(size_changed)),
                    position_changed=int(bool(position_changed)),
                )
            except Exception:
                pass

        def _clear_child_window_motion(event=None) -> None:
            try:
                if getattr(event, "widget", None) is child_window:
                    _finish_child_window_move()
            except Exception:
                pass

        try:
            child_window.bind("<Configure>", _record_child_window_configure, add="+")
            child_window.bind("<Destroy>", _clear_child_window_motion, add="+")
        except Exception:
            pass

    if window_move_tracking_enabled:
        try:
            dialog.bind("<Configure>", _record_toplevel_window_configure, add="+")
        except Exception:
            pass

    def _refresh_mobile_export_wraps() -> None:
        done = _mobile_export_perf_timer("wrap_refresh")
        wrap_refresh_state["after"] = None
        count = 0
        try:
            if _mobile_export_window_is_moving():
                try:
                    wrap_refresh_state["after"] = dialog.after(240, _refresh_mobile_export_wraps)
                except Exception:
                    pass
                return
            for refresher in tuple(wrap_refreshers):
                count += 1
                try:
                    refresher()
                except Exception:
                    pass
        finally:
            done(items=count)

    def _schedule_mobile_export_wraps(event=None) -> None:
        started = time.perf_counter()
        try:
            try:
                if any(bool(guard.get("frozen")) for guard in win32_move_guards.values()):
                    return
            except Exception:
                pass
            if not wrap_refreshers:
                return
            event_widget = getattr(event, "widget", None)
            if event is not None and event_widget is not root:
                try:
                    widget_class = str(event_widget.winfo_class() or "")
                except Exception:
                    widget_class = str(type(event_widget).__name__)
                _mobile_export_perf_record(
                    "content_configure_child_ignored",
                    (time.perf_counter() - started) * 1000.0,
                    widget=widget_class[:48],
                )
                return
            size = (
                int(getattr(event, "width", 0) or root.winfo_width() or 0),
                int(getattr(event, "height", 0) or root.winfo_height() or 0),
            )
            previous_size = wrap_refresh_state.get("size")
            size_changed = size != previous_size
            _mobile_export_perf_record(
                "content_configure",
                (time.perf_counter() - started) * 1000.0,
                size=f"{size[0]}x{size[1]}",
                size_changed=int(bool(size_changed)),
            )
            if not size_changed:
                return
            wrap_refresh_state["size"] = size
            pending = wrap_refresh_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
            wrap_refresh_state["after"] = dialog.after(120, _refresh_mobile_export_wraps)
        except Exception:
            pass

    win32_move_guards: dict[str, dict] = {}

    def _install_mobile_export_win32_move_guard(target_window=None, *, guard_key: str = "main") -> None:
        target_window = target_window or dialog
        guard = win32_move_guards.setdefault(str(guard_key or "main"), {"installed": False, "frozen": False, "hwnds": []})
        if guard.get("installed"):
            return
        if windowing_system != "win32":
            return
        guard_key_text = str(guard_key or "main").strip().lower()
        # The Win32 redraw guard is useful for diagnostics, but on this heavy Tk dialog
        # it can make native dragging feel elastic. Keep it opt-in instead of default.
        default_guard = "0"
        guard_flag = str(
            os.environ.get(
                "AAT_MOBILE_EXPORT_WIN32_MOVE_GUARD",
                os.environ.get("AAT_MOBILE_EXPORT_EXPERIMENTAL_WIN32_MOVE_GUARD", default_guard),
            )
        ).strip().lower()
        if guard_flag in {"0", "false", "no", "off"}:
            return
        if guard_flag not in {"1", "true", "yes", "on"}:
            return
        if str(os.environ.get("AAT_DISABLE_WIN32_MOVE_GUARD", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(int(target_window.winfo_id()))
            GWLP_WNDPROC = -4
            GWL_STYLE = -16
            GWL_EXSTYLE = -20
            WM_ENTERSIZEMOVE = 0x0231
            WM_EXITSIZEMOVE = 0x0232
            WM_NCDESTROY = 0x0082
            WM_SETREDRAW = 0x000B
            WS_CLIPCHILDREN = 0x02000000
            WS_CLIPSIBLINGS = 0x04000000
            WS_EX_COMPOSITED = 0x02000000
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            RDW_INVALIDATE = 0x0001
            RDW_ERASE = 0x0004
            RDW_ALLCHILDREN = 0x0080
            RDW_UPDATENOW = 0x0100
            RDW_FRAME = 0x0400

            LONG_PTR = ctypes.c_ssize_t
            WNDPROC = ctypes.WINFUNCTYPE(
                LONG_PTR,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            ENUM_CHILD_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            get_window_long_ptr = user32.GetWindowLongPtrW
            set_window_long_ptr = user32.SetWindowLongPtrW
            get_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long_ptr.restype = LONG_PTR
            set_window_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
            set_window_long_ptr.restype = LONG_PTR
            user32.CallWindowProcW.argtypes = [
                LONG_PTR,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.CallWindowProcW.restype = LONG_PTR
            user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.SendMessageW.restype = LONG_PTR
            user32.RedrawWindow.argtypes = [wintypes.HWND, wintypes.LPCRECT, wintypes.HRGN, wintypes.UINT]
            user32.RedrawWindow.restype = wintypes.BOOL
            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.restype = wintypes.BOOL
            user32.EnumChildWindows.argtypes = [wintypes.HWND, ENUM_CHILD_PROC, wintypes.LPARAM]
            user32.EnumChildWindows.restype = wintypes.BOOL
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            user32.SetWindowPos.restype = wintypes.BOOL

            old_proc = get_window_long_ptr(hwnd, GWLP_WNDPROC)
            if not old_proc:
                return

            try:
                style = int(get_window_long_ptr(hwnd, GWL_STYLE) or 0)
                styled = style | WS_CLIPCHILDREN | WS_CLIPSIBLINGS
                if styled != style:
                    set_window_long_ptr(hwnd, GWL_STYLE, styled)
            except Exception:
                pass
            try:
                if str(os.environ.get("AAT_ENABLE_WIN32_COMPOSITED", "")).strip().lower() in {"1", "true", "yes", "on"}:
                    ex_style = int(get_window_long_ptr(hwnd, GWL_EXSTYLE) or 0)
                    styled_ex = ex_style | WS_EX_COMPOSITED
                    if styled_ex != ex_style:
                        set_window_long_ptr(hwnd, GWL_EXSTYLE, styled_ex)
            except Exception:
                pass
            try:
                flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
                user32.SetWindowPos(hwnd, wintypes.HWND(0), 0, 0, 0, 0, flags)
            except Exception:
                pass

            def _current_child_hwnds() -> list[int]:
                hwnds: list[int] = []

                @ENUM_CHILD_PROC
                def _enum_child(child_hwnd, _lparam):
                    try:
                        hwnds.append(int(child_hwnd))
                    except Exception:
                        pass
                    return True

                try:
                    user32.EnumChildWindows(hwnd, _enum_child, 0)
                except Exception:
                    pass
                return hwnds

            def _set_window_redraw(enabled: bool) -> None:
                try:
                    if user32.IsWindow(hwnd):
                        user32.SendMessageW(hwnd, WM_SETREDRAW, int(bool(enabled)), 0)
                except Exception:
                    pass

            def _set_children_redraw(enabled: bool) -> None:
                for child_hwnd in list(guard.get("hwnds") or []):
                    try:
                        child = wintypes.HWND(int(child_hwnd))
                        if user32.IsWindow(child):
                            user32.SendMessageW(child, WM_SETREDRAW, int(bool(enabled)), 0)
                    except Exception:
                        pass

            def _freeze_children() -> None:
                if guard.get("frozen"):
                    return
                guard["hwnds"] = _current_child_hwnds()
                freeze_toplevel_flag = str(os.environ.get("AAT_MOBILE_EXPORT_FREEZE_TOPLEVEL_REDRAW", "1")).strip().lower()
                freeze_toplevel = freeze_toplevel_flag not in {"0", "false", "no", "off"}
                guard["freeze_toplevel"] = freeze_toplevel
                if freeze_toplevel:
                    _set_window_redraw(False)
                _set_children_redraw(False)
                guard["frozen"] = True
                _mobile_export_perf_record(
                    "win32_move_guard_freeze",
                    0.0,
                    window=str(guard_key or "main"),
                    children=len(guard.get("hwnds") or []),
                )

            def _thaw_children() -> None:
                if not guard.get("frozen"):
                    return
                _set_children_redraw(True)
                if bool(guard.get("freeze_toplevel")):
                    _set_window_redraw(True)
                flags = RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW | RDW_FRAME
                try:
                    user32.RedrawWindow(hwnd, None, None, flags)
                except Exception:
                    pass
                guard["frozen"] = False
                guard["hwnds"] = []
                _mobile_export_perf_record("win32_move_guard_thaw", 0.0, window=str(guard_key or "main"))

            @WNDPROC
            def _window_proc(hwnd_arg, msg, wparam, lparam):
                try:
                    if msg == WM_ENTERSIZEMOVE:
                        _freeze_children()
                    elif msg == WM_EXITSIZEMOVE:
                        _thaw_children()
                    elif msg == WM_NCDESTROY:
                        _thaw_children()
                        try:
                            set_window_long_ptr(hwnd_arg, GWLP_WNDPROC, old_proc)
                        except Exception:
                            pass
                        return user32.CallWindowProcW(old_proc, hwnd_arg, msg, wparam, lparam)
                except Exception:
                    pass
                return user32.CallWindowProcW(old_proc, hwnd_arg, msg, wparam, lparam)

            previous = set_window_long_ptr(hwnd, GWLP_WNDPROC, ctypes.cast(_window_proc, ctypes.c_void_p).value)
            if previous:
                guard.update(
                    {
                        "installed": True,
                        "old_proc": old_proc,
                        "proc": _window_proc,
                        "enum_proc_type": ENUM_CHILD_PROC,
                        "freeze": _freeze_children,
                        "thaw": _thaw_children,
                    }
                )

                configure_freeze_flag = str(os.environ.get("AAT_MOBILE_EXPORT_CONFIGURE_FREEZE", "1")).strip().lower()
                if configure_freeze_flag not in {"0", "false", "no", "off"}:
                    configure_state = {"size": (0, 0), "position": (None, None), "after": None}

                    def _finish_configure_freeze() -> None:
                        configure_state["after"] = None
                        _thaw_children()

                    def _configure_freeze(event=None) -> None:
                        try:
                            if event is not None and getattr(event, "widget", None) is not target_window:
                                return
                            size = (
                                int(getattr(event, "width", 0) or target_window.winfo_width() or 0),
                                int(getattr(event, "height", 0) or target_window.winfo_height() or 0),
                            )
                            position = (
                                getattr(event, "x", None),
                                getattr(event, "y", None),
                            )
                            previous_size = configure_state.get("size")
                            previous_position = configure_state.get("position")
                            configure_state["size"] = size
                            configure_state["position"] = position
                            if position != previous_position and size == previous_size:
                                _freeze_children()
                                pending = configure_state.get("after")
                                if pending:
                                    try:
                                        target_window.after_cancel(pending)
                                    except Exception:
                                        pass
                                configure_state["after"] = target_window.after(140, _finish_configure_freeze)
                        except Exception:
                            pass

                    def _destroy_configure_freeze(event=None) -> None:
                        try:
                            if getattr(event, "widget", None) is target_window:
                                pending = configure_state.get("after")
                                if pending:
                                    try:
                                        target_window.after_cancel(pending)
                                    except Exception:
                                        pass
                                _thaw_children()
                        except Exception:
                            pass

                    try:
                        target_window.bind("<Configure>", _configure_freeze, add="+")
                        target_window.bind("<Destroy>", _destroy_configure_freeze, add="+")
                    except Exception:
                        pass
                logger.info(f"[MOBILE EXPORT] Wlaczono natywna optymalizacje przesuwania okna Win32: {guard_key}.")
        except Exception as exc:
            logger.debug(f"Nie udało się włączyć win32 move guard dla eksportu mobilnego: {exc}")

    title_label = tk.Label(
        root,
        text="Centrum eksportu mobilnego",
        bg=bg,
        fg=fg,
        font=("Segoe UI", 13, "bold"),
        anchor=tk.W,
    )
    title_label.grid(row=0, column=0, sticky="ew", pady=(0, 3))

    def _bind_mobile_export_dynamic_wrap(
        label: tk.Label,
        container=None,
        *,
        min_wrap: int = 90,
        margin_ratio: float = 0.045,
        margin_min: int = 14,
        margin_max: int = 64,
        textvariable: tk.Variable | None = None,
        on_updated=None,
    ) -> tk.Label:
        target = container or label.master
        state = {"after": None, "width": 0, "raw": None, "wrap": 0}
        if textvariable is None and not hasattr(label, "_mobile_export_raw_text"):
            try:
                label._mobile_export_raw_text = str(label.cget("text") or "-")
            except Exception:
                pass

        def _raw_text() -> str:
            if textvariable is not None:
                try:
                    return str(textvariable.get() or "-")
                except Exception:
                    return "-"
            return str(getattr(label, "_mobile_export_raw_text", label.cget("text")) or "-")

        def _apply(width_hint: int = 0) -> None:
            done = _mobile_export_perf_timer("label_wrap_apply")
            state["after"] = None
            changed = 0
            try:
                width = int(width_hint or target.winfo_width() or label.winfo_width() or 1)
                if width <= 2:
                    return
                raw = _raw_text()
                if abs(width - int(state.get("width") or 0)) < 3 and raw == state.get("raw"):
                    return
                margin = max(margin_min, min(margin_max, int(width * margin_ratio)))
                wrap_px = max(min_wrap, width - margin)
                display_text = _mobile_export_text_for_width(raw, wrap_px)
                updates = {}
                if int(state.get("wrap") or 0) != int(wrap_px):
                    updates["wraplength"] = wrap_px
                if label.cget("text") != display_text:
                    updates["text"] = display_text
                if updates:
                    changed = 1
                    label.configure(**updates)
                    if on_updated is not None:
                        try:
                            label.after_idle(on_updated)
                        except Exception:
                            pass
                state["width"] = width
                state["raw"] = raw
                state["wrap"] = int(wrap_px)
            except Exception:
                pass
            finally:
                done(changed=changed)

        def _refresh(event=None) -> None:
            started = time.perf_counter()
            try:
                width = int(getattr(event, "width", 0) or target.winfo_width() or label.winfo_width() or 1)
                raw = _raw_text()
                if abs(width - int(state.get("width") or 0)) < 3 and raw == state.get("raw"):
                    return
                pending = state.get("after")
                if pending:
                    try:
                        label.after_cancel(pending)
                    except Exception:
                        pass
                state["after"] = label.after(70, lambda w=width: _apply(w))
                _mobile_export_perf_record(
                    "label_wrap_queue",
                    (time.perf_counter() - started) * 1000.0,
                    width=width,
                )
            except Exception:
                pass

        if textvariable is not None:
            try:
                textvariable.trace_add("write", lambda *_args: _refresh())
            except Exception:
                pass
        try:
            wrap_refreshers.append(_apply)
        except Exception:
            pass
        try:
            label.after_idle(lambda: _apply())
        except Exception:
            pass
        return label

    intro_label = tk.Label(
        root,
        text="Zbuduj pakiet eksportu mobilnego (maksymalnie 3 modele).",
        bg=bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=980,
    )
    intro_label.grid(row=1, column=0, sticky="ew", pady=(0, 5))
    try:
        intro_label.grid_remove()
    except Exception:
        pass

    workspace = tk.Frame(root, bg=bg)
    workspace.grid(row=2, column=0, sticky="nsew")
    workspace.grid_columnconfigure(0, weight=3, minsize=500)
    workspace.grid_columnconfigure(1, weight=2, minsize=540)
    workspace.grid_rowconfigure(0, weight=1)

    candidates_shell = tk.Frame(
        workspace,
        bg=card_bg,
        padx=9,
        pady=7,
        highlightthickness=1,
        highlightbackground=border,
    )
    candidates_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    candidates_shell.grid_columnconfigure(0, weight=1)
    candidates_shell.grid_rowconfigure(2, weight=1)
    candidates_shell.grid_rowconfigure(3, weight=0)
    candidate_count_var = tk.StringVar(value=f"Kandydaci do eksportu: {len(candidates)}")
    tk.Label(
        candidates_shell,
        textvariable=candidate_count_var,
        bg=card_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

    candidate_help_shell = tk.Frame(
        candidates_shell,
        bg=blend_hex_colors(card_bg, accent, 0.045),
        padx=8,
        pady=6,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, card_bg, 0.55),
    )
    candidate_help_shell.grid(row=1, column=0, sticky="ew", pady=(0, 7))
    candidate_help_shell.grid_columnconfigure(0, weight=1)
    tk.Label(
        candidate_help_shell,
        text="Zbuduj pakiet eksportu mobilnego (maksymalnie 3 modele).",
        bg=candidate_help_shell["bg"],
        fg=muted,
        font=("Segoe UI", 9, "bold"),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=390,
    ).grid(row=0, column=0, sticky="ew")

    candidates_table_shell = tk.Frame(candidates_shell, bg=card_bg)
    candidates_table_shell.grid(row=2, column=0, sticky="nsew")
    candidates_table_shell.grid_columnconfigure(0, weight=1)
    candidates_table_shell.grid_rowconfigure(0, weight=1)
    candidate_columns = ("Eksport", "Model", "Projekt", "YOLO", "Epoki", "Metryka", "Data")
    candidate_tree_style = "MobileExportCandidates.Treeview"
    try:
        tree_style = ttk.Style(dialog)
        tree_style.configure(
            candidate_tree_style,
            background=card_bg,
            fieldbackground=card_bg,
            foreground=fg,
            borderwidth=0,
            rowheight=24,
        )
        tree_style.map(
            candidate_tree_style,
            background=[("selected", blend_hex_colors(accent, bg, 0.30))],
            foreground=[("selected", "#ffffff")],
        )
    except Exception:
        pass
    candidate_tree = ttk.Treeview(
        candidates_table_shell,
        columns=candidate_columns,
        show="headings",
        height=min(15, max(7, len(candidates))),
        selectmode="browse",
        style=candidate_tree_style,
    )
    for column in candidate_columns:
        candidate_tree.heading(column, text=column, command=lambda col=column: _set_candidate_sort(col))
    candidate_tree.column("Eksport", width=58, minwidth=48, stretch=False, anchor=tk.CENTER)
    candidate_tree.column("Model", width=46, minwidth=38, stretch=False, anchor=tk.CENTER)
    candidate_tree.column("Projekt", width=82, minwidth=54, stretch=False, anchor=tk.W)
    candidate_tree.column("YOLO", width=64, minwidth=44, stretch=False, anchor=tk.W)
    candidate_tree.column("Epoki", width=54, minwidth=38, stretch=False, anchor=tk.CENTER)
    candidate_tree.column("Metryka", width=78, minwidth=58, stretch=False, anchor=tk.CENTER)
    candidate_tree.column("Data", width=82, minwidth=56, stretch=True, anchor=tk.CENTER)
    candidate_scroll = WebSlimScrollbar(
        candidates_table_shell,
        orient=tk.VERTICAL,
        command=candidate_tree.yview,
        **_mobile_export_scrollbar_kwargs(palette, card_bg),
    )
    candidate_tree.configure(yscrollcommand=candidate_scroll.set)
    candidate_tree.grid(row=0, column=0, sticky="nsew")
    candidate_scroll.grid(row=0, column=1, sticky="ns")
    candidate_width_state = {"after": None, "width": 0}

    def _apply_candidate_column_widths(width_hint: int = 0) -> None:
        candidate_width_state["after"] = None
        try:
            if _mobile_export_window_is_moving():
                candidate_width_state["after"] = candidate_tree.after(
                    240,
                    lambda w=width_hint: _apply_candidate_column_widths(w),
                )
                return
            width = max(300, int(width_hint or candidate_tree.winfo_width() or 420) - 16)
            if width >= 430:
                fixed = {"Eksport": 54, "Model": 44, "Projekt": 92, "YOLO": 62, "Epoki": 52, "Metryka": 76, "Data": 64}
            elif width >= 360:
                fixed = {"Eksport": 50, "Model": 40, "Projekt": 66, "YOLO": 52, "Epoki": 46, "Metryka": 66, "Data": 48}
            else:
                remaining = width
                fixed = {
                    "Eksport": 46,
                    "Model": 36,
                    "Projekt": max(52, int(remaining * 0.20)),
                    "YOLO": max(42, int(remaining * 0.15)),
                    "Epoki": max(38, int(remaining * 0.13)),
                    "Metryka": max(54, int(remaining * 0.20)),
                    "Data": max(50, remaining),
                }
                fixed["Data"] = max(46, width - fixed["Eksport"] - fixed["Model"] - fixed["Projekt"] - fixed["YOLO"] - fixed["Epoki"] - fixed["Metryka"])
            fixed_sum = sum(fixed.values())
            if fixed_sum != width and fixed:
                fixed["Data"] = max(46, fixed["Data"] + width - fixed_sum)
            candidate_tree.column("Eksport", width=max(48, fixed["Eksport"]), minwidth=44, stretch=False, anchor=tk.CENTER)
            candidate_tree.column("Model", width=max(38, fixed["Model"]), minwidth=34, stretch=False, anchor=tk.CENTER)
            candidate_tree.column("Projekt", width=max(54, fixed["Projekt"]), minwidth=48, stretch=False, anchor=tk.W)
            candidate_tree.column("YOLO", width=max(44, fixed["YOLO"]), minwidth=38, stretch=False, anchor=tk.W)
            candidate_tree.column("Epoki", width=max(38, fixed["Epoki"]), minwidth=32, stretch=False, anchor=tk.CENTER)
            candidate_tree.column("Metryka", width=max(58, fixed["Metryka"]), minwidth=50, stretch=False, anchor=tk.CENTER)
            candidate_tree.column("Data", width=max(56, fixed["Data"]), minwidth=48, stretch=True, anchor=tk.CENTER)
        except Exception:
            pass

    def _queue_candidate_column_widths(event=None, delay: int = 50) -> None:
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 220)
            width = int(getattr(event, "width", 0) or candidate_tree.winfo_width() or 0)
            if abs(width - int(candidate_width_state.get("width") or 0)) < 3 and event is not None:
                return
            candidate_width_state["width"] = width
            pending = candidate_width_state.get("after")
            if pending:
                try:
                    candidate_tree.after_cancel(pending)
                except Exception:
                    pass
            candidate_width_state["after"] = candidate_tree.after(delay, lambda w=width: _apply_candidate_column_widths(w))
        except Exception:
            pass

    try:
        candidate_tree.bind("<Configure>", _queue_candidate_column_widths, add="+")
        candidate_tree.after_idle(lambda: _apply_candidate_column_widths())
    except Exception:
        pass
    _set_mobile_export_loader("Buduję listę kandydatów...", 42.0)

    overview_shell = tk.Frame(
        candidates_shell,
        bg=card_bg,
        padx=0,
        pady=0,
        highlightthickness=0,
    )
    overview_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    overview_shell.grid_columnconfigure(0, weight=1)
    overview_canvas = tk.Canvas(
        overview_shell,
        width=360,
        height=150,
        bg=card_bg,
        highlightthickness=0,
        bd=0,
    )
    overview_canvas.grid(row=0, column=0, sticky="nsew")

    candidate_by_iid = {str(candidate.get("iid")): candidate for candidate in candidates}
    candidate_iid_by_dedupe_key: dict[str, str] = {}
    for _candidate_iid, _candidate in list(candidate_by_iid.items()):
        _dedupe_key = _mobile_export_candidate_dedupe_key(_candidate)
        if _dedupe_key and _dedupe_key not in candidate_iid_by_dedupe_key:
            candidate_iid_by_dedupe_key[_dedupe_key] = _candidate_iid
    next_candidate_iid = {"value": len(candidates)}
    export_marker_order = ("MP", "MT", "MZ")
    export_selection_state = {"MP": "", "MT": "", "MZ": ""}
    max_candidate_size_mb = max(
        (float(candidate.get("file_size_mb") or 0.0) for candidate in candidates),
        default=0.0,
    )
    candidate_sort_state = {"column": "Data", "descending": True}

    def _register_mobile_export_candidate(candidate: dict) -> str:
        nonlocal max_candidate_size_mb
        if not isinstance(candidate, dict):
            return ""
        dedupe_key = _mobile_export_candidate_dedupe_key(candidate)
        existing_dedupe_iid = candidate_iid_by_dedupe_key.get(dedupe_key) if dedupe_key else ""
        if existing_dedupe_iid and existing_dedupe_iid in candidate_by_iid:
            existing = candidate_by_iid.get(existing_dedupe_iid) or {}
            if _mobile_export_candidate_preference(candidate) > _mobile_export_candidate_preference(existing):
                candidate["iid"] = existing_dedupe_iid
                candidate_by_iid[existing_dedupe_iid] = candidate
                for index, current_candidate in enumerate(candidates):
                    if str(current_candidate.get("iid") or "") == existing_dedupe_iid:
                        candidates[index] = candidate
                        break
            return existing_dedupe_iid
        candidate_iid = str(candidate.get("iid") or "").strip()
        if not candidate_iid:
            while True:
                candidate_iid = f"mobile_export_candidate_{int(next_candidate_iid.get('value') or 0)}"
                next_candidate_iid["value"] = int(next_candidate_iid.get("value") or 0) + 1
                if candidate_iid not in candidate_by_iid:
                    break
            candidate["iid"] = candidate_iid
        if candidate_iid not in candidate_by_iid:
            candidates.append(candidate)
        candidate_by_iid[candidate_iid] = candidate
        if dedupe_key:
            candidate_iid_by_dedupe_key[dedupe_key] = candidate_iid
        try:
            max_candidate_size_mb = max(max_candidate_size_mb, float(candidate.get("file_size_mb") or 0.0))
        except Exception:
            pass
        try:
            candidate_count_var.set(f"Kandydaci do eksportu: {len(candidates)}")
        except Exception:
            pass
        return candidate_iid

    def _candidate_role_tag(candidate: dict) -> str:
        marker = _mobile_export_target_marker(candidate)
        if marker == "MT":
            return "mobile_export_role_mt"
        if marker == "MZ":
            return "mobile_export_role_mz"
        if marker == "MP":
            return "mobile_export_role_mp"
        return "mobile_export_role_unknown"

    def _configure_candidate_role_tags() -> None:
        role_specs = (
            ("mobile_export_role_mt", {"role": "plate"}),
            ("mobile_export_role_mz", {"role": "character"}),
            ("mobile_export_role_mp", {"role": "vehicle"}),
            ("mobile_export_role_unknown", {}),
        )
        for tag_name, marker_candidate in role_specs:
            fg_color = _mobile_export_target_base_color(palette, marker_candidate) if marker_candidate else muted
            bg_color = blend_hex_colors(card_bg, fg_color, 0.24 if marker_candidate else 0.08)
            try:
                candidate_tree.tag_configure(
                    tag_name,
                    foreground=fg_color,
                    background=bg_color,
                )
            except Exception:
                pass

    _configure_candidate_role_tags()
    _set_mobile_export_loader("Koloruję role modeli i wykresy...", 52.0)

    def _candidate_datetime_sort_value(candidate: dict) -> str:
        return str(
            candidate.get("finished_at")
            or candidate.get("started_at")
            or candidate.get("created_at")
            or ""
        ).strip()

    def _candidate_sort_value(candidate: dict, column: str):
        if column == "Eksport":
            marker = _mobile_export_target_marker(candidate)
            return 1 if str(export_selection_state.get(marker) or "") == str(candidate.get("iid") or "") else 0
        if column in {"Typ", "Model"}:
            return f"{_mobile_export_target_marker(candidate)} {_mobile_export_model_cell_label(candidate)}".casefold()
        if column == "Projekt":
            return _mobile_export_project_label(candidate, empty="").casefold()
        if column == "YOLO":
            return _mobile_export_compact_yolo_label(candidate.get("model_version")).casefold()
        if column == "Epoki":
            return _mobile_export_candidate_total_epochs(candidate)
        if column == "Metryka":
            return _candidate_metric_sort_value(candidate)
        if column == "Data":
            return _candidate_datetime_sort_value(candidate)
        return str(candidate.get("model_label") or "").casefold()

    def _selected_export_iids() -> set[str]:
        return {str(item or "") for item in export_selection_state.values() if str(item or "")}

    def _candidate_export_checkbox(candidate: dict) -> str:
        marker = _mobile_export_target_marker(candidate)
        if marker not in set(export_marker_order):
            return "-"
        return "☑" if str(export_selection_state.get(marker) or "") == str(candidate.get("iid") or "") else "☐"

    def _candidate_metric_value(candidate: dict | None):
        if not isinstance(candidate, dict):
            return None, ""
        if _mobile_export_percent_value(candidate.get("best_map50_95")) is not None:
            return candidate.get("best_map50_95"), "m95"
        if _mobile_export_percent_value(candidate.get("best_map50")) is not None:
            return candidate.get("best_map50"), "m50"
        return None, ""

    def _candidate_metric_cell(candidate: dict | None) -> str:
        value, label = _candidate_metric_value(candidate)
        if value is None:
            return "-"
        return f"{label} {_format_mobile_export_metric(value)}"

    def _candidate_metric_sort_value(candidate: dict | None) -> float:
        value, _label = _candidate_metric_value(candidate)
        parsed = _mobile_export_percent_value(value)
        return float(parsed) if parsed is not None else -1.0

    def _candidate_metric_label(candidate: dict | None) -> str:
        value, label = _candidate_metric_value(candidate)
        if value is None:
            return "metryka -"
        title = "mAP50-95" if label == "m95" else "mAP50"
        return f"{title} {_format_mobile_export_metric(value)}"

    def _candidate_values(candidate: dict) -> tuple[str, str, str, str, str, str, str]:
        return (
            _candidate_export_checkbox(candidate),
            _mobile_export_target_marker(candidate),
            _mobile_export_project_label(candidate, empty="-"),
            _mobile_export_compact_yolo_label(candidate.get("model_version")),
            _mobile_export_candidate_epochs_label(candidate, compact=True),
            _candidate_metric_cell(candidate),
            _format_mobile_export_datetime(
                candidate.get("finished_at") or candidate.get("started_at") or candidate.get("created_at")
            ),
        )

    def _refresh_candidate_headings() -> None:
        active_column = str(candidate_sort_state.get("column") or "")
        descending = bool(candidate_sort_state.get("descending"))
        for column in candidate_columns:
            arrow = " ↓" if descending else " ↑"
            label = f"{column}{arrow}" if column == active_column else column
            candidate_tree.heading(column, text=label, command=lambda col=column: _set_candidate_sort(col))

    def _populate_candidate_tree(*, refresh_selection: bool = False) -> None:
        current_selection = candidate_tree.selection()
        selected_iid = str(current_selection[0]) if current_selection else ""
        column = str(candidate_sort_state.get("column") or "Data")
        descending = bool(candidate_sort_state.get("descending"))
        ordered = sorted(
            candidates,
            key=lambda candidate: _candidate_sort_value(candidate, column),
            reverse=descending,
        )
        for row_id in candidate_tree.get_children():
            candidate_tree.delete(row_id)
        for candidate in ordered:
            candidate_tree.insert(
                "",
                tk.END,
                iid=str(candidate.get("iid")),
                values=_candidate_values(candidate),
                tags=(_candidate_role_tag(candidate),),
            )
        if selected_iid and selected_iid in candidate_by_iid:
            try:
                candidate_tree.selection_set(selected_iid)
                candidate_tree.focus(selected_iid)
                candidate_tree.see(selected_iid)
            except Exception:
                pass
        _refresh_candidate_headings()
        if refresh_selection and selected_iid:
            try:
                dialog.after_idle(on_candidate_select)
            except Exception:
                pass

    def _set_candidate_sort(column: str) -> None:
        current_column = str(candidate_sort_state.get("column") or "")
        if current_column == column:
            candidate_sort_state["descending"] = not bool(candidate_sort_state.get("descending"))
        else:
            candidate_sort_state["column"] = column
            candidate_sort_state["descending"] = column in {"Eksport", "Epoki", "Metryka", "Data"}
        _populate_candidate_tree(refresh_selection=True)

    def _selected_export_candidates() -> list[dict]:
        selected_candidates: list[dict] = []
        for marker in export_marker_order:
            candidate = candidate_by_iid.get(str(export_selection_state.get(marker) or ""))
            if candidate:
                selected_candidates.append(candidate)
        return selected_candidates

    def _export_selection_validation() -> tuple[bool, str, str]:
        selection = describe_mobile_export_selection(
            _mobile_export_target_marker(candidate) for candidate in _selected_export_candidates()
        )
        return selection.valid, selection.label, selection.message

    def _export_selection_candidate_label(candidate: dict) -> str:
        marker = _mobile_export_target_marker(candidate)
        model = str(candidate.get("model_label") or candidate.get("run_label") or "-").strip() or "-"
        yolo = _mobile_export_compact_yolo_label(candidate.get("model_version"))
        metric = _candidate_metric_label(candidate)
        size_mb = candidate.get("file_size_mb")
        size_text = f"{float(size_mb):.1f} MB" if size_mb is not None else "waga -"
        return f"{marker}: {model} | {yolo} | {metric} | {size_text}"

    def _export_selection_total_size_mb(selected_candidates: list[dict] | None = None) -> float | None:
        total = 0.0
        count = 0
        for candidate in selected_candidates if selected_candidates is not None else _selected_export_candidates():
            try:
                size_mb = candidate.get("file_size_mb")
                if size_mb is None:
                    continue
                total += float(size_mb)
                count += 1
            except Exception:
                continue
        return total if count else None

    def _export_selection_package_title(selected_candidates: list[dict] | None = None) -> str:
        selected = selected_candidates if selected_candidates is not None else _selected_export_candidates()
        if not selected:
            return "Nie wybrano modeli"
        markers = [_mobile_export_target_marker(candidate) for candidate in selected]
        return " + ".join(marker for marker in markers if marker) or "Wybrane modele"

    def _export_selection_package_weight_text(selected_candidates: list[dict] | None = None) -> str:
        total_size_mb = _export_selection_total_size_mb(selected_candidates)
        if total_size_mb is None:
            return "Łączna waga modeli: -"
        return f"Łączna waga modeli .pt: {total_size_mb:.1f} MB"

    def _refresh_export_selection_summary_vars(
        selected_candidates: list[dict] | None = None,
        *,
        is_valid: bool | None = None,
        selection_label: str | None = None,
        selection_message: str | None = None,
    ) -> None:
        selected = selected_candidates if selected_candidates is not None else _selected_export_candidates()
        if is_valid is None or selection_label is None or selection_message is None:
            is_valid, selection_label, selection_message = _export_selection_validation()
        package_title = _export_selection_package_title(selected)
        total_size_mb = _export_selection_total_size_mb(selected)
        weight_text = _export_selection_package_weight_text(selected)
        state_text = str(selection_message or "-")
        try:
            selected_params_var.set(str(selection_label or package_title or "-") if selected
                                    else "Wybierz kandydata z listy")
        except Exception:
            pass
        try:
            selected_size_var.set(f"{total_size_mb:.1f} MB" if total_size_mb is not None else "-")
        except Exception:
            pass
        try:
            run_value_var.set(package_title)
            role_value_var.set(weight_text)
            checkpoint_value_var.set(state_text)
        except Exception:
            pass

    def _update_export_selection_view(*, announce: bool = False) -> None:
        selected_candidates = _selected_export_candidates()
        is_valid, _selection_label, _selection_message = _export_selection_validation()
        _refresh_export_selection_summary_vars(
            selected_candidates,
            is_valid=is_valid,
            selection_label=_selection_label,
            selection_message=_selection_message,
        )
        if selected_candidates:
            export_selection_var.set(_selection_message + "\n" + "\n".join(
                _export_selection_candidate_label(candidate) for candidate in selected_candidates
            ))
            try:
                export_selection_label.configure(fg=success if is_valid else warning)
            except Exception:
                pass
        else:
            export_selection_var.set(
                "Do eksportu nie wybrano jeszcze modelu. Zaznacz pojedynczy MP, MT lub MZ albo komplet MT+MZ / MP+MT+MZ."
            )
            try:
                export_selection_label.configure(fg=muted)
            except Exception:
                pass
        try:
            if export_button is not None:
                export_button.configure(state=(tk.NORMAL if is_valid and not worker_state.get("running") else tk.DISABLED))
        except Exception:
            pass
        try:
            current_iid = str((candidate_tree.selection() or ("",))[0] or "")
        except Exception:
            current_iid = ""
        if str(candidate_sort_state.get("column") or "") == "Eksport":
            _populate_candidate_tree(refresh_selection=True)
        else:
            for row_id in candidate_tree.get_children():
                candidate = candidate_by_iid.get(str(row_id))
                if candidate:
                    candidate_tree.item(row_id, values=_candidate_values(candidate))
        try:
            if current_iid:
                candidate_tree.selection_set(current_iid)
                candidate_tree.focus(current_iid)
        except Exception:
            pass
        try:
            current_candidate = selected_state.get("candidate")
            _draw_mobile_export_overview_chart(
                overview_canvas,
                candidates,
                str((current_candidate or {}).get("iid") or current_iid),
                palette,
                _selected_export_iids(),
            )
        except Exception:
            pass
        if announce:
            message = _selection_message
            tone = "success" if is_valid and selected_candidates else "warning" if selected_candidates else "info"
            suffix = " Ustawienia wybierzesz w następnym kroku." if is_valid and selected_candidates else ""
            set_status(f"{message}{suffix}", tone)

    def _toggle_export_candidate(row_id: str) -> None:
        candidate = candidate_by_iid.get(str(row_id or ""))
        if not candidate:
            return
        marker = _mobile_export_target_marker(candidate)
        if marker not in set(export_marker_order):
            set_status("Ten typ modelu nie jest obsługiwany przez eksport mobilny ALPR.", "warning")
            return
        current = str(export_selection_state.get(marker) or "")
        candidate_iid = str(candidate.get("iid") or "")
        export_selection_state[marker] = "" if current == candidate_iid else candidate_iid
        _update_export_selection_view(announce=True)

    def on_candidate_export_click(event) -> str | None:
        try:
            if str(candidate_tree.identify_region(event.x, event.y)) == "heading":
                return None
            row_id = str(candidate_tree.identify_row(event.y) or "")
            column_id = str(candidate_tree.identify_column(event.x) or "")
        except Exception:
            return None
        if not row_id or column_id != "#1":
            return None
        try:
            candidate_tree.selection_set(row_id)
            candidate_tree.focus(row_id)
            candidate = candidate_by_iid.get(row_id)
            if candidate:
                apply_candidate(candidate)
        except Exception:
            pass
        _toggle_export_candidate(row_id)
        return "break"

    _populate_candidate_tree()

    candidate_model_tip = {"window": None, "iid": ""}

    def _hide_candidate_model_tip(_event=None) -> None:
        tip = candidate_model_tip.get("window")
        candidate_model_tip["window"] = None
        candidate_model_tip["iid"] = ""
        try:
            if tip is not None:
                tip.destroy()
        except Exception:
            pass

    def _show_candidate_model_tip(event) -> None:
        try:
            row_id = str(candidate_tree.identify_row(event.y) or "")
            column_id = str(candidate_tree.identify_column(event.x) or "")
        except Exception:
            _hide_candidate_model_tip()
            return
        if not row_id or column_id not in {"#2", "#5"}:
            _hide_candidate_model_tip()
            return
        candidate = candidate_by_iid.get(row_id)
        if not candidate:
            _hide_candidate_model_tip()
            return
        text = _mobile_export_epochs_tooltip(candidate) if column_id == "#5" else _mobile_export_target_tooltip(candidate)
        if not text:
            _hide_candidate_model_tip()
            return
        tip = candidate_model_tip.get("window")
        tip_key = f"{row_id}:{column_id}"
        if tip is None or str(candidate_model_tip.get("iid") or "") != tip_key:
            _hide_candidate_model_tip()
            tip = tk.Toplevel(dialog)
            try:
                tip.overrideredirect(True)
                tip.attributes("-topmost", True)
            except Exception:
                pass
            label = tk.Label(
                tip,
                text=text,
                bg=blend_hex_colors(bg, accent, 0.16),
                fg=fg,
                padx=10,
                pady=7,
                justify=tk.LEFT,
                anchor=tk.W,
                wraplength=320,
                font=("Segoe UI", 8),
                relief=tk.SOLID,
                bd=1,
            )
            label.pack(fill=tk.BOTH, expand=True)
            candidate_model_tip["window"] = tip
            candidate_model_tip["iid"] = tip_key
        try:
            tip.geometry(f"+{int(event.x_root) + 14}+{int(event.y_root) + 16}")
        except Exception:
            pass

    try:
        candidate_tree.bind("<Motion>", _show_candidate_model_tip, add="+")
        candidate_tree.bind("<Leave>", _hide_candidate_model_tip, add="+")
        candidate_tree.bind("<ButtonPress>", _hide_candidate_model_tip, add="+")
        dialog.bind("<Destroy>", _hide_candidate_model_tip, add="+")
    except Exception:
        pass

    main = tk.Frame(workspace, bg=bg)
    main.grid(row=0, column=1, sticky="nsew")
    main.grid_columnconfigure(0, weight=3, minsize=320)
    main.grid_columnconfigure(1, weight=2, minsize=220)
    main.grid_rowconfigure(0, weight=0)
    main.grid_rowconfigure(1, weight=1, uniform="mobile_export_info")
    main.grid_rowconfigure(2, weight=1, uniform="mobile_export_info")

    selected_model_var = tk.StringVar(value="-")
    selected_target_var = tk.StringVar(value="-")
    selected_metric_var = tk.StringVar(value="-")
    selected_params_var = tk.StringVar(value="Wybierz kandydata z listy")
    selected_size_var = tk.StringVar(value="-")

    selected_canvas = tk.Canvas(
        main,
        bg=blend_hex_colors(card_bg, success, 0.035),
        highlightthickness=1,
        highlightbackground=blend_hex_colors(success, bg, 0.40),
        bd=0,
        height=60,
    )
    selected_canvas.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
    selected_card_specs = (
        ("Wskazany model", selected_model_var, "info"),
        ("Rola / YOLO", selected_target_var, "info"),
        ("Jakość", selected_metric_var, "success"),
        ("Eksport", selected_params_var, "info"),
        ("Waga modeli", selected_size_var, "warning"),
    )
    selected_canvas_state = {"after": None, "width": 0}

    def _draw_selected_cards(width_hint: int = 0) -> None:
        selected_canvas_state["after"] = None
        try:
            if _mobile_export_window_is_moving():
                selected_canvas_state["after"] = selected_canvas.after(220, lambda w=width_hint: _draw_selected_cards(w))
                return
            width = max(320, int(width_hint or selected_canvas.winfo_width() or 860))
            empty_export = selected_params_var.get() == "Wybierz kandydata z listy"
            height_value = 76 if empty_export else 60
            if int(selected_canvas.cget("height")) != height_value:
                selected_canvas.configure(height=height_value)
            selected_canvas.delete("all")
            selected_canvas.create_rectangle(
                0,
                0,
                width,
                height_value,
                fill=selected_canvas["bg"],
                outline=blend_hex_colors(success, bg, 0.40),
                width=1,
            )
            colors = {
                "success": success,
                "warning": warning,
                "error": error,
                "info": fg,
            }
            padding = 10
            gap = 8
            card_count = max(1, len(selected_card_specs))
            card_width = max(94, int((width - padding * 2 - gap * (card_count - 1)) / card_count))
            x = padding
            for title, variable, tone in selected_card_specs:
                tone_color = colors.get(tone, fg)
                card_fill = blend_hex_colors(card_bg, tone_color, 0.055)
                card_outline = blend_hex_colors(tone_color, bg, 0.50)
                selected_canvas.create_rectangle(
                    x,
                    6,
                    min(width - padding, x + card_width),
                    height_value - 6,
                    fill=card_fill,
                    outline=card_outline,
                    width=1,
                )
                selected_canvas.create_text(
                    x + 8,
                    14,
                    text=title,
                    fill=muted,
                    font=("Segoe UI", 7, "bold"),
                    anchor=tk.W,
                )
                value = str(variable.get() or "-")
                display = _mobile_export_text_for_width(value, max(82, card_width - 18), min_chars=9)
                lines = display.splitlines()
                max_lines = 3 if title == "Eksport" and empty_export else 2
                if len(lines) > max_lines:
                    display = "\n".join(lines[:max_lines])
                selected_canvas.create_text(
                    x + 8,
                    28,
                    text=display,
                    fill=tone_color,
                    font=("Segoe UI", 9, "bold"),
                    anchor=tk.NW,
                    justify=tk.LEFT,
                    width=max(82, card_width - 18),
                )
                x += card_width + gap
        except Exception:
            pass

    def _queue_selected_cards_redraw(event=None, delay: int = 40) -> None:
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 180)
            width = int(getattr(event, "width", 0) or selected_canvas.winfo_width() or 0)
            if abs(width - int(selected_canvas_state.get("width") or 0)) < 3 and event is not None:
                return
            selected_canvas_state["width"] = width
            pending = selected_canvas_state.get("after")
            if pending:
                try:
                    selected_canvas.after_cancel(pending)
                except Exception:
                    pass
            selected_canvas_state["after"] = selected_canvas.after(delay, lambda w=width: _draw_selected_cards(w))
        except Exception:
            pass

    try:
        selected_canvas.bind("<Configure>", _queue_selected_cards_redraw, add="+")
        for _var in (selected_model_var, selected_target_var, selected_metric_var, selected_params_var, selected_size_var):
            _var.trace_add("write", lambda *_args: _queue_selected_cards_redraw(delay=0))
        selected_canvas.after_idle(lambda: _draw_selected_cards())
    except Exception:
        pass

    def _wrapped_table_mousewheel(canvas: tk.Canvas, event) -> str:
        try:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                units = -max(-6, min(6, int(getattr(event, "delta", 0) / 120))) or 0
            if units:
                canvas.yview_scroll(units, "units")
        except Exception:
            pass
        return "break"

    def _build_wrapped_info_table(
        parent,
        columns: list[dict],
        *,
        bg_color: str,
        header_bg: str,
        height: int,
    ) -> dict:
        shell = tk.Frame(master=parent, bg=bg_color)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_columnconfigure(1, weight=0)
        shell.grid_rowconfigure(1, weight=1)

        header_height = 30
        header_canvas = tk.Canvas(shell, bg=header_bg, highlightthickness=0, bd=0, height=header_height)
        header_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        canvas = tk.Canvas(shell, bg=bg_color, highlightthickness=0, bd=0, height=height)
        scrollbar = WebSlimScrollbar(
            shell,
            orient=tk.VERTICAL,
            command=canvas.yview,
            **_mobile_export_scrollbar_kwargs(palette, bg_color),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        table_state = {
            "after": None,
            "fingerprint": None,
            "rows": [],
            "width": 0,
            "height": 0,
        }
        font_cache: dict[str, tkfont.Font] = {}

        def _table_font(weight: str = "normal") -> tkfont.Font:
            key = str(weight or "normal")
            cached = font_cache.get(key)
            if cached is None:
                cached = tkfont.Font(root=dialog, family="Segoe UI", size=8, weight=key)
                font_cache[key] = cached
            return cached

        def _rows_fingerprint(rows: list[dict]) -> tuple:
            fingerprint = []
            for row in rows:
                values = row.get("values") if isinstance(row.get("values"), dict) else {}
                fingerprint.append(
                    (
                        tuple((str(key), str(value)) for key, value in sorted(values.items())),
                        str(row.get("tone") or ""),
                    )
                )
            return tuple(fingerprint)

        def _column_widths(total_width: int) -> list[int]:
            safe_width = max(160, int(total_width or 0))
            widths: list[int] = []
            hard_mins: list[int] = []
            weights: list[int] = []
            for column in columns:
                configured = int(column.get("width", column.get("minsize", 90)) or 90)
                minimum = int(column.get("minsize", configured) or configured)
                key = str(column.get("key") or "")
                if key == "status":
                    hard_min = 42
                elif key == "progress":
                    hard_min = 86
                elif key in {"label", "requirement"}:
                    hard_min = 84
                else:
                    hard_min = 120
                widths.append(max(hard_min, configured, minimum))
                hard_mins.append(hard_min)
                weights.append(max(0, int(column.get("weight", 0) or 0)))

            current = sum(widths)
            if current < safe_width and widths:
                expandable = [idx for idx, weight in enumerate(weights) if weight > 0] or list(range(len(widths)))
                total_weight = sum(weights[idx] or 1 for idx in expandable)
                remaining = safe_width - current
                for idx in expandable:
                    add = int(remaining * ((weights[idx] or 1) / max(1, total_weight)))
                    widths[idx] += add
                widths[-1] += safe_width - sum(widths)
            elif current > safe_width and widths:
                overflow = current - safe_width
                capacities = [max(0, width - hard_min) for width, hard_min in zip(widths, hard_mins)]
                capacity_total = sum(capacities)
                if capacity_total > 0:
                    for idx, capacity in enumerate(capacities):
                        shrink = min(capacity, int(overflow * (capacity / capacity_total)))
                        widths[idx] -= shrink
                    while sum(widths) > safe_width:
                        idx = max(range(len(widths)), key=lambda item: widths[item] - hard_mins[item])
                        if widths[idx] <= hard_mins[idx]:
                            break
                        widths[idx] -= 1
                if sum(widths) > safe_width:
                    ratio = safe_width / max(1, sum(widths))
                    widths = [max(30, int(width * ratio)) for width in widths]
                    widths[-1] += safe_width - sum(widths)
            return [max(24, int(width)) for width in widths]

        def _cell_text_height(text: str, pixel_width: int, font_obj: tkfont.Font) -> tuple[str, int]:
            wrap_px = max(24, int(pixel_width or 24))
            display = _mobile_export_text_for_width(text or "-", wrap_px, min_chars=max(4, int(wrap_px / 8)))
            line_count = max(1, display.count("\n") + 1)
            line_height = max(12, int(font_obj.metrics("linespace") or 12))
            return display, line_count * line_height

        def _draw_canvas_table(width_hint: int = 0) -> None:
            done = _mobile_export_perf_timer("canvas_table_redraw")
            table_state["after"] = None
            item_count = 0
            try:
                if _mobile_export_window_is_moving():
                    table_state["after"] = canvas.after(240, lambda w=width_hint: _draw_canvas_table(w))
                    return
                width = int(width_hint or canvas.winfo_width() or canvas.cget("width") or 420)
                width = max(160, width)
                if abs(width - int(table_state.get("width") or 0)) < 3 and not canvas.find_all():
                    width = int(table_state.get("width") or width)
                table_state["width"] = width
                widths = _column_widths(width)
                table_width = max(1, sum(widths))
                header_canvas.delete("all")
                header_canvas.create_rectangle(
                    0,
                    0,
                    table_width,
                    header_height,
                    fill=header_bg,
                    outline=blend_hex_colors(border, bg_color, 0.72),
                    width=1,
                )
                x = 0
                header_font = _table_font("bold")
                for index, column in enumerate(columns):
                    column_width = widths[index] if index < len(widths) else 80
                    if index:
                        header_canvas.create_line(x, 0, x, header_height, fill=blend_hex_colors(border, bg_color, 0.68))
                    header_canvas.create_text(
                        x + 8,
                        max(8, header_height // 2),
                        text=str(column.get("title") or ""),
                        fill=muted,
                        font=header_font,
                        anchor=tk.W,
                        width=max(1, column_width - 16),
                    )
                    x += column_width
                header_canvas.configure(scrollregion=(0, 0, table_width, header_height))

                try:
                    first, _last = canvas.yview()
                except Exception:
                    first = 0.0
                canvas.delete("all")
                y = 0
                regular_font = _table_font("normal")
                bold_font = _table_font("bold")
                rows = list(table_state.get("rows") or [])
                for row_index, row in enumerate(rows):
                    row_bg = blend_hex_colors(bg_color, fg, 0.025 if row_index % 2 else 0.012)
                    values = row.get("values") if isinstance(row.get("values"), dict) else {}
                    tone = str(row.get("tone") or "").strip().lower()
                    prepared: list[tuple[str, str, tkfont.Font, int]] = []
                    row_height = 34
                    for index, column in enumerate(columns):
                        key = str(column.get("key") or "")
                        raw = str(values.get(key, "") if isinstance(values, dict) else "") or "-"
                        cell_font = bold_font if key == "status" or tone == "metric" else regular_font
                        wrap_px = max(24, (widths[index] if index < len(widths) else 80) - 16)
                        display, text_height = _cell_text_height(raw, wrap_px, cell_font)
                        prepared.append((key, display, cell_font, wrap_px))
                        row_height = max(row_height, text_height + 12)

                    canvas.create_rectangle(
                        0,
                        y,
                        table_width,
                        y + row_height,
                        fill=row_bg,
                        outline=blend_hex_colors(border, bg_color, 0.72),
                        width=1,
                    )
                    item_count += 1
                    x = 0
                    for index, column in enumerate(columns):
                        key, display, cell_font, wrap_px = prepared[index]
                        cell_width = widths[index] if index < len(widths) else 80
                        if index:
                            canvas.create_line(x, y, x, y + row_height, fill=blend_hex_colors(border, bg_color, 0.76))
                            item_count += 1
                        cell_fg = fg
                        value_lower = display.strip().lower()
                        if key == "status":
                            if value_lower in {"jest", "gotowe", "już było", "już zainstalowane"}:
                                cell_fg = success
                            elif value_lower in {"brak", "błąd"}:
                                cell_fg = error
                            elif value_lower in {
                                "start",
                                "sprawdzam",
                                "pobieram",
                                "z cache",
                                "metadane",
                                "buduję wheel",
                                "instaluję",
                                "kończę",
                                "pauza",
                            }:
                                cell_fg = warning
                            else:
                                cell_fg = muted
                        elif tone == "metric":
                            cell_fg = success
                        elif tone == "path":
                            cell_fg = muted
                        if key == "status":
                            pill_fill = blend_hex_colors(bg_color, muted, 0.10)
                            pill_outline = blend_hex_colors(muted, bg_color, 0.50)
                            if value_lower in {"jest", "gotowe", "już było", "już zainstalowane"}:
                                pill_fill = blend_hex_colors(bg_color, success, 0.16)
                                pill_outline = blend_hex_colors(success, bg_color, 0.42)
                            elif value_lower in {"brak", "błąd"}:
                                pill_fill = blend_hex_colors(bg_color, error, 0.16)
                                pill_outline = blend_hex_colors(error, bg_color, 0.42)
                            elif value_lower in {
                                "start",
                                "sprawdzam",
                                "pobieram",
                                "z cache",
                                "metadane",
                                "buduję wheel",
                                "instaluję",
                                "kończę",
                                "pauza",
                            }:
                                pill_fill = blend_hex_colors(bg_color, warning, 0.18)
                                pill_outline = blend_hex_colors(warning, bg_color, 0.42)
                            canvas.create_rectangle(
                                x + 6,
                                y + 7,
                                x + max(30, cell_width - 6),
                                y + min(row_height - 7, 30),
                                fill=pill_fill,
                                outline=pill_outline,
                                width=1,
                            )
                            canvas.create_text(
                                x + cell_width / 2,
                                y + 18,
                                text=display.splitlines()[0],
                                fill=cell_fg,
                                font=cell_font,
                                anchor=tk.CENTER,
                            )
                            item_count += 2
                            x += cell_width
                            continue
                        if key == "progress":
                            raw_progress = values.get("_progress_value", display) if isinstance(values, dict) else display
                            progress_value = _mobile_export_progress_value(str(raw_progress).replace("%", ""))
                            bar_x0 = x + 8
                            bar_x1 = x + max(32, cell_width - 8)
                            bar_y0 = y + 10
                            bar_y1 = y + min(row_height - 10, 26)
                            bar_fill = blend_hex_colors(bg_color, "#ffffff", 0.08)
                            progress_color = success if progress_value >= 99.5 else accent
                            if value_lower in {"0%", "0.0%"} or progress_value <= 0.0:
                                progress_color = blend_hex_colors(muted, bg_color, 0.35)
                            canvas.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, fill=bar_fill, outline="")
                            fill_width = int((bar_x1 - bar_x0) * (progress_value / 100.0))
                            if fill_width > 0:
                                canvas.create_rectangle(
                                    bar_x0,
                                    bar_y0,
                                    bar_x0 + max(2, fill_width),
                                    bar_y1,
                                    fill=progress_color,
                                    outline="",
                                )
                            canvas.create_text(
                                (bar_x0 + bar_x1) / 2,
                                (bar_y0 + bar_y1) / 2,
                                text=_mobile_export_progress_label(progress_value),
                                fill=fg,
                                font=bold_font,
                                anchor=tk.CENTER,
                            )
                            item_count += 3
                            x += cell_width
                            continue
                        canvas.create_text(
                            x + 8,
                            y + 6,
                            text=display,
                            fill=cell_fg,
                            font=cell_font,
                            anchor=tk.NW,
                            justify=tk.LEFT,
                            width=wrap_px,
                        )
                        item_count += 1
                        x += cell_width
                    y += row_height + 3
                viewport_height = max(height, int(canvas.winfo_height() or height))
                canvas.configure(scrollregion=(0, 0, table_width, max(y, viewport_height)))
                table_state["height"] = y
                try:
                    canvas.yview_moveto(first)
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                done(width=table_state.get("width"), height=table_state.get("height"), items=item_count)

        def _queue_canvas_table_redraw(event=None, delay: int = 60, force: bool = False) -> None:
            started = time.perf_counter()
            try:
                if _mobile_export_window_is_moving():
                    delay = max(delay, 180)
                width = int(getattr(event, "width", 0) or canvas.winfo_width() or 1)
                if not force and abs(width - int(table_state.get("width") or 0)) < 3:
                    return
                pending = table_state.get("after")
                if pending:
                    try:
                        canvas.after_cancel(pending)
                    except Exception:
                        pass
                if delay <= 0:
                    table_state["after"] = canvas.after_idle(lambda w=width: _draw_canvas_table(w))
                else:
                    table_state["after"] = canvas.after(delay, lambda w=width: _draw_canvas_table(w))
                _mobile_export_perf_record(
                    "table_width_queue",
                    (time.perf_counter() - started) * 1000.0,
                    width=width,
                )
            except Exception:
                pass

        def _bind_wheel(widget) -> None:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    widget.bind(sequence, lambda event, target=canvas: _wrapped_table_mousewheel(target, event), add="+")
                except Exception:
                    pass

        _bind_wheel(shell)
        _bind_wheel(header_canvas)
        _bind_wheel(canvas)
        try:
            canvas.bind("<Configure>", _queue_canvas_table_redraw, add="+")
        except Exception:
            pass

        def set_rows(rows: list[dict]) -> None:
            done = _mobile_export_perf_timer("table_set_rows")
            try:
                fingerprint = _rows_fingerprint(rows)
                if fingerprint == table_state.get("fingerprint"):
                    _mobile_export_perf_record("table_set_rows_skip", 0.0, rows=len(rows))
                    return
                table_state["fingerprint"] = fingerprint
                table_state["rows"] = list(rows or [])
                _queue_canvas_table_redraw(delay=0, force=True)
            finally:
                done(rows=len(rows), widgets=0)

        return {"shell": shell, "set_rows": set_rows, "canvas": canvas,
                "scrollbar": scrollbar, "column_widths": _column_widths}

    requirements_shell = tk.Frame(
        main,
        bg=blend_hex_colors(card_bg, accent, 0.035),
        padx=9,
        pady=7,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, bg, 0.40),
    )
    requirements_shell.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 8))
    requirements_shell.grid_columnconfigure(0, weight=1)
    requirements_shell.grid_rowconfigure(1, weight=1)
    tk.Label(
        requirements_shell,
        text="Wymagania eksportu dla wybranego modelu",
        bg=requirements_shell["bg"],
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew", pady=(0, 5))
    requirements_table = _build_wrapped_info_table(
        requirements_shell,
        [
            {"key": "status", "title": "Stan", "minsize": 42, "width": 42, "wrap": 36},
            {"key": "requirement", "title": "Wymaganie", "minsize": 96, "width": 104, "wrap": 88},
            {"key": "description", "title": "Opis", "minsize": 190, "width": 230, "weight": 1, "wrap": 170},
        ],
        bg_color=requirements_shell["bg"],
        header_bg=blend_hex_colors(requirements_shell["bg"], accent, 0.10),
        height=172,
    )
    requirements_table["shell"].grid(row=1, column=0, sticky="nsew")

    details_shell = tk.Frame(
        main,
        bg=card_bg,
        padx=8,
        pady=6,
        highlightthickness=1,
        highlightbackground=border,
    )
    details_shell.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
    details_shell.grid_columnconfigure(0, weight=1)
    details_shell.grid_rowconfigure(1, weight=1)
    tk.Label(
        details_shell,
        text="Krótki profil kandydata",
        bg=card_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
    details_table = _build_wrapped_info_table(
        details_shell,
        [
            {"key": "label", "title": "Parametr", "minsize": 102, "width": 112, "wrap": 92},
            {"key": "value", "title": "Wartość", "minsize": 205, "width": 250, "weight": 1, "wrap": 170},
        ],
        bg_color=card_bg,
        header_bg=blend_hex_colors(card_bg, accent, 0.08),
        height=172,
    )
    details_table["shell"].grid(row=1, column=0, sticky="nsew")
    details_table["shell"].grid_rowconfigure(1, weight=0)
    details_table["shell"].grid_rowconfigure(2, weight=1)
    details_table["canvas"].grid_configure(row=2)
    details_table["scrollbar"].grid_configure(row=2)
    source_locations = MobileExportSourceLocations(
        details_table["shell"], bg=card_bg, fg=fg, muted=muted, accent=border,
        notify=getattr(self.app, "show_assistant_message", getattr(self.app, "update_status", None)),
        column_widths=details_table["column_widths"],
    )
    source_locations.grid(row=1, column=0, sticky="ew")
    try:
        requirements_shell.grid_remove()
        details_shell.grid_configure(row=1, rowspan=2, sticky="nsew", padx=(0, 10))
    except Exception:
        pass

    form_shell = tk.Frame(
        main,
        bg=card_bg,
        highlightthickness=1,
        highlightbackground=border,
    )
    form_shell.grid(row=1, column=1, rowspan=2, sticky="nsew")
    form_shell.grid_columnconfigure(0, weight=1)
    form_shell.grid_rowconfigure(0, weight=1)
    form_canvas = tk.Canvas(form_shell, bg=card_bg, highlightthickness=0, bd=0)
    form_scroll = WebSlimScrollbar(
        form_shell,
        orient=tk.VERTICAL,
        command=form_canvas.yview,
        **_mobile_export_scrollbar_kwargs(palette, card_bg),
    )
    form_canvas.configure(yscrollcommand=form_scroll.set)
    form_canvas.grid(row=0, column=0, sticky="nsew")
    form_scroll.grid(row=0, column=1, sticky="ns")
    form = tk.Frame(form_canvas, bg=card_bg, padx=11, pady=10)
    form_window = form_canvas.create_window((0, 0), window=form, anchor=tk.NW)
    form.grid_columnconfigure(0, weight=1)
    form_scroll_state = {"after": None, "width": 0, "height": 0}
    form_width_state = {"width": 0, "after": None}

    def _sync_export_form_scrollregion(_event=None) -> None:
        done = _mobile_export_perf_timer("form_scrollregion")
        try:
            form_canvas.configure(scrollregion=form_canvas.bbox("all"))
            form_scroll_state["width"] = int(form.winfo_width() or 0)
            form_scroll_state["height"] = int(form.winfo_height() or 0)
        except Exception:
            pass
        finally:
            done(width=form_scroll_state.get("width"), height=form_scroll_state.get("height"))

    def _queue_export_form_scrollregion(event=None, delay: int = 80) -> None:
        started = time.perf_counter()
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 180)
            width = int(getattr(event, "width", 0) or form.winfo_width() or 0)
            height = int(getattr(event, "height", 0) or form.winfo_height() or 0)
            if (
                abs(width - int(form_scroll_state.get("width") or 0)) < 3
                and abs(height - int(form_scroll_state.get("height") or 0)) < 3
            ):
                return
            pending = form_scroll_state.get("after")
            if pending:
                try:
                    form_canvas.after_cancel(pending)
                except Exception:
                    pass
            form_scroll_state["after"] = form_canvas.after(delay, _sync_export_form_scrollregion)
            _mobile_export_perf_record(
                "form_scrollregion_queue",
                (time.perf_counter() - started) * 1000.0,
                width=width,
                height=height,
            )
        except Exception:
            pass

    def _sync_export_form_width(event=None) -> None:
        done = _mobile_export_perf_timer("form_width_sync")
        try:
            if _mobile_export_window_is_moving():
                try:
                    pending = form_width_state.get("after")
                    if pending:
                        try:
                            form_canvas.after_cancel(pending)
                        except Exception:
                            pass
                    form_width_state["after"] = form_canvas.after(
                        180,
                        lambda current_event=event: _sync_export_form_width(current_event),
                    )
                except Exception:
                    pass
                return
            form_width_state["after"] = None
            width = int(getattr(event, "width", 0) or form_canvas.winfo_width() or 1)
            if abs(width - int(form_width_state.get("width") or 0)) < 3:
                return
            form_width_state["width"] = width
            form_canvas.itemconfigure(form_window, width=max(1, width - 4))
            try:
                form_canvas.after_idle(_sync_export_form_scrollregion)
            except Exception:
                    pass
        except Exception:
            pass
        finally:
            done(width=form_width_state.get("width"))

    try:
        form.bind("<Configure>", _queue_export_form_scrollregion, add="+")
        form_canvas.bind("<Configure>", _sync_export_form_width, add="+")
    except Exception:
        pass

    litert_var = tk.BooleanVar(value=True)
    onnx_var = tk.BooleanVar(value=False)
    int8_var = tk.BooleanVar(value=False)
    onnx_int8_var = tk.BooleanVar(value=False)
    ncnn_var = tk.BooleanVar(value=False)
    imgsz_var = tk.IntVar(value=CONFIG.DEFAULT_IMG_SIZE)
    conf_var = tk.DoubleVar(value=CONFIG.DEFAULT_CONFIDENCE)
    iou_var = tk.DoubleVar(value=CONFIG.DEFAULT_IOU)
    destination_var = tk.StringVar(value="")
    calibration_var = tk.StringVar(value="")
    run_value_var = tk.StringVar(value="Nie wybrano modeli")
    role_value_var = tk.StringVar(value="Łączna waga modeli: -")
    checkpoint_value_var = tk.StringVar(value="Zaznacz modele po lewej stronie.")
    status_var = tk.StringVar(value="Wybierz kandydata do podglądu i zaznacz modele, które mają trafić do eksportu.")
    dependency_tool_status_var = tk.StringVar(
        value="Sprawdzenie gotowości obejmuje model, formaty i biblioteki potrzebne do eksportu."
    )
    dependency_tool_eta_var = tk.StringVar(value="ETA: -")
    export_selection_var = tk.StringVar(
        value="Do eksportu nie wybrano jeszcze modelu. Zaznacz pojedynczy MP, MT lub MZ albo komplet MT+MZ / MP+MT+MZ."
    )
    selected_state = {"candidate": None}

    summary_canvas = tk.Canvas(
        form,
        bg=blend_hex_colors(card_bg, accent, 0.030),
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, bg, 0.48),
        bd=0,
        height=108,
    )
    summary_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    summary_canvas_state = {"after": None, "width": 0}
    summary_rows = (
        ("Eksport", run_value_var),
        ("Waga", role_value_var),
        ("Status", checkpoint_value_var),
    )

    def _draw_mobile_export_summary(width_hint: int = 0) -> None:
        summary_canvas_state["after"] = None
        try:
            if _mobile_export_window_is_moving():
                summary_canvas_state["after"] = summary_canvas.after(
                    220,
                    lambda w=width_hint: _draw_mobile_export_summary(w),
                )
                return
            width = max(260, int(width_hint or summary_canvas.winfo_width() or 340))
            value_x = 96
            value_width = max(138, width - value_x - 12)
            prepared_rows = []
            row_heights: list[int] = []
            for label, variable in summary_rows:
                display = _mobile_export_text_for_width(
                    str(variable.get() or "-"),
                    value_width,
                    min_chars=15,
                )
                lines = max(1, len(display.splitlines()))
                min_height = 32
                max_height = 66 if label == "Status" else 48
                row_height = min(max_height, max(min_height, lines * 14 + 12))
                prepared_rows.append((label, display))
                row_heights.append(row_height)
            height_value = max(sum(row_heights) + 10, 106)
            try:
                if int(summary_canvas.cget("height") or 0) != height_value:
                    summary_canvas.configure(height=height_value)
            except Exception:
                pass
            top = 5
            summary_canvas.delete("all")
            summary_canvas.create_rectangle(
                0,
                0,
                width,
                height_value,
                fill=summary_canvas["bg"],
                outline=blend_hex_colors(accent, bg, 0.48),
                width=1,
            )
            y = top
            for index, (label, display) in enumerate(prepared_rows):
                if index:
                    summary_canvas.create_line(8, y, width - 8, y, fill=blend_hex_colors(border, card_bg, 0.72))
                summary_canvas.create_text(
                    10,
                    y + 7,
                    text=label,
                    fill=muted,
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.W,
                )
                summary_canvas.create_text(
                    value_x,
                    y + 5,
                    text=display,
                    fill=fg,
                    font=("Segoe UI", 8),
                    anchor=tk.NW,
                    justify=tk.LEFT,
                    width=value_width,
                )
                y += row_heights[index] if index < len(row_heights) else 34
            summary_canvas.configure(scrollregion=(0, 0, width, height_value))
        except Exception:
            pass

    def _queue_mobile_export_summary_redraw(event=None, delay: int = 40) -> None:
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 180)
            width = int(getattr(event, "width", 0) or summary_canvas.winfo_width() or 0)
            if abs(width - int(summary_canvas_state.get("width") or 0)) < 3 and event is not None:
                return
            summary_canvas_state["width"] = width
            pending = summary_canvas_state.get("after")
            if pending:
                try:
                    summary_canvas.after_cancel(pending)
                except Exception:
                    pass
            summary_canvas_state["after"] = summary_canvas.after(delay, lambda w=width: _draw_mobile_export_summary(w))
        except Exception:
            pass

    try:
        summary_canvas.bind("<Configure>", _queue_mobile_export_summary_redraw, add="+")
        for _summary_var in (run_value_var, role_value_var, checkpoint_value_var):
            _summary_var.trace_add("write", lambda *_args: _queue_mobile_export_summary_redraw(delay=0))
        summary_canvas.after_idle(lambda: _draw_mobile_export_summary())
    except Exception:
        pass

    export_selection_shell = tk.Frame(
        form,
        bg=blend_hex_colors(card_bg, success, 0.035),
        padx=10,
        pady=8,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(success, bg, 0.45),
    )
    export_selection_shell.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    export_selection_shell.grid_columnconfigure(0, weight=1)
    tk.Label(
        export_selection_shell,
        text="Modele w paczce",
        bg=export_selection_shell["bg"],
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    export_selection_label = tk.Label(
        export_selection_shell,
        textvariable=export_selection_var,
        bg=export_selection_shell["bg"],
        fg=muted,
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=330,
    )
    export_selection_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))

    form_section_labels: dict[int, tk.Label] = {}

    def row_label(row: int, text: str) -> None:
        label = tk.Label(
            form,
            text=text,
            bg=card_bg,
            fg=muted,
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
        )
        label.grid(row=row * 2, column=0, sticky="ew", pady=(6 if row else 0, 2))
        form_section_labels[row] = label

    def content_row(row: int) -> int:
        return row * 2 + 1

    row_label(3, "Formaty")
    formats_frame = tk.Frame(
        form,
        bg=blend_hex_colors(card_bg, accent, 0.035),
        padx=10,
        pady=8,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, bg, 0.50),
    )
    formats_frame.grid(row=content_row(3), column=0, sticky="ew", pady=(0, 6))
    formats_frame.grid_columnconfigure(0, weight=1)
    for idx, (text, precision_text, var) in enumerate(
        (
            ("LiteRT/TFLite", "FP32, bez kwantyzacji", litert_var),
            ("LiteRT/TFLite", "INT8, kwantyzacja po kalibracji", int8_var),
            ("ONNX", "FP32", onnx_var),
            ("ONNX", "INT8, kwantyzacja po kalibracji", onnx_int8_var),
            ("NCNN", "FP32", ncnn_var),
        )
    ):
        ttk.Checkbutton(formats_frame, text=f"{text} | {precision_text}", variable=var).grid(
            row=idx,
            column=0,
            sticky="ew",
            pady=2,
        )

    row_label(4, "Parametry eksportu")
    params = tk.Frame(form, bg=card_bg)
    params.grid(row=content_row(4), column=0, sticky="ew", pady=(0, 6))
    for col in range(6):
        params.grid_columnconfigure(col, weight=1 if col in {1, 3, 5} else 0)
    tk.Label(params, text="imgsz", bg=card_bg, fg=muted).grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=128, to=2048, increment=32, textvariable=imgsz_var, width=8).grid(
        row=0,
        column=1,
        sticky="w",
        padx=(0, 18),
    )
    tk.Label(params, text="conf", bg=card_bg, fg=muted).grid(row=0, column=2, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=conf_var, width=8, format="%.2f").grid(
        row=0,
        column=3,
        sticky="w",
        padx=(0, 18),
    )
    tk.Label(params, text="IoU", bg=card_bg, fg=muted).grid(row=0, column=4, sticky="w", padx=(0, 6))
    ttk.Spinbox(params, from_=0.0, to=1.0, increment=0.01, textvariable=iou_var, width=8, format="%.2f").grid(
        row=0,
        column=5,
        sticky="w",
    )

    row_label(5, "Kalibracja INT8")
    calibration_row = tk.Frame(form, bg=card_bg)
    calibration_row.grid(row=content_row(5), column=0, sticky="ew", pady=(0, 6))
    calibration_row.grid_columnconfigure(0, weight=1)
    ttk.Entry(calibration_row, textvariable=calibration_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))

    def choose_calibration() -> None:
        selected = filedialog.askopenfilename(
            title="Wybierz data.yaml do kalibracji INT8",
            parent=dialog,
            filetypes=(("YOLO data.yaml", "data.yaml"), ("YAML", "*.yaml *.yml"), ("Wszystkie pliki", "*.*")),
        )
        if selected:
            calibration_var.set(selected)

    ttk.Button(calibration_row, text="Wybierz data.yaml", command=choose_calibration).grid(row=0, column=1, sticky="e")

    row_label(6, "Wykres")
    metric_chart_canvas = tk.Canvas(
        form,
        width=390,
        height=150,
        bg=card_bg,
        highlightthickness=0,
        bd=0,
    )
    metric_chart_canvas.grid(row=content_row(6), column=0, sticky="ew", pady=(0, 2))
    try:
        for section_row in (3, 4, 5):
            label = form_section_labels.get(section_row)
            if label is not None:
                label.grid_remove()
        formats_frame.grid_remove()
        params.grid_remove()
        calibration_row.grid_remove()
        chart_label = form_section_labels.get(6)
        if chart_label is not None:
            chart_label.configure(text="Profil metryk")
    except Exception:
        pass

    status_box = tk.Frame(
        root,
        bg=blend_hex_colors(bg, accent, 0.03),
        padx=10,
        pady=8,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, bg, 0.45),
    )
    status_box.grid(row=3, column=0, sticky="ew", pady=(6, 0))
    status_box.grid_columnconfigure(0, weight=1)
    status_canvas = tk.Canvas(
        status_box,
        bg=status_box["bg"],
        highlightthickness=0,
        bd=0,
        height=30,
    )
    status_canvas.grid(row=0, column=0, sticky="ew")
    status_canvas_state = {"after": None, "width": 0, "tone": "info"}

    def _draw_mobile_export_status(width_hint: int = 0) -> None:
        status_canvas_state["after"] = None
        try:
            if _mobile_export_window_is_moving():
                status_canvas_state["after"] = status_canvas.after(
                    220,
                    lambda w=width_hint: _draw_mobile_export_status(w),
                )
                return
            width = max(320, int(width_hint or status_canvas.winfo_width() or 900))
            tone = str(status_canvas_state.get("tone") or "info")
            tone_color = {
                "success": success,
                "warning": warning,
                "error": error,
                "info": fg,
            }.get(tone, fg)
            status_canvas.delete("all")
            text = _mobile_export_text_for_width(str(status_var.get() or "-"), max(320, width - 20), min_chars=44)
            status_canvas.create_text(
                2,
                3,
                text=text,
                fill=tone_color,
                font=("Segoe UI", 8),
                anchor=tk.NW,
                justify=tk.LEFT,
                width=max(320, width - 12),
            )
        except Exception:
            pass

    def _queue_mobile_export_status_redraw(event=None, delay: int = 35) -> None:
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 180)
            width = int(getattr(event, "width", 0) or status_canvas.winfo_width() or 0)
            if abs(width - int(status_canvas_state.get("width") or 0)) < 3 and event is not None:
                return
            status_canvas_state["width"] = width
            pending = status_canvas_state.get("after")
            if pending:
                try:
                    status_canvas.after_cancel(pending)
                except Exception:
                    pass
            status_canvas_state["after"] = status_canvas.after(delay, lambda w=width: _draw_mobile_export_status(w))
        except Exception:
            pass

    try:
        status_canvas.bind("<Configure>", _queue_mobile_export_status_redraw, add="+")
        status_var.trace_add("write", lambda *_args: _queue_mobile_export_status_redraw(delay=0))
        status_canvas.after_idle(lambda: _draw_mobile_export_status())
    except Exception:
        pass
    progress_var = tk.DoubleVar(value=0.0)
    progress_percent_var = tk.StringVar(value="0%")
    progress_row = tk.Frame(status_box, bg=status_box["bg"])
    progress_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    progress_row.grid_columnconfigure(0, weight=1)
    progress = ttk.Progressbar(progress_row, mode="determinate", maximum=100.0, variable=progress_var)
    progress.grid(row=0, column=0, sticky="ew")
    tk.Label(
        progress_row,
        textvariable=progress_percent_var,
        bg=status_box["bg"],
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        width=5,
        anchor=tk.E,
    ).grid(row=0, column=1, sticky="e", padx=(10, 0))
    dependency_tool_shell = tk.Frame(
        status_box,
        bg=blend_hex_colors(status_box["bg"], accent, 0.030),
        padx=8,
        pady=6,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(accent, bg, 0.52),
    )
    dependency_tool_shell.grid(row=2, column=0, sticky="ew", pady=(7, 0))
    dependency_tool_shell.grid_columnconfigure(0, weight=1)
    tk.Label(
        dependency_tool_shell,
        text="Gotowość eksportu i zależności",
        bg=dependency_tool_shell["bg"],
        fg=fg,
        font=("Segoe UI", 8, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
    dependency_tool_status_label = tk.Label(
        dependency_tool_shell,
        textvariable=dependency_tool_status_var,
        bg=dependency_tool_shell["bg"],
        fg=muted,
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=560,
    )
    dependency_tool_status_label.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 0))
    dependency_tool_eta_label = tk.Label(
        dependency_tool_shell,
        textvariable=dependency_tool_eta_var,
        bg=dependency_tool_shell["bg"],
        fg=muted,
        font=("Segoe UI", 8, "bold"),
        anchor=tk.E,
        width=16,
    )
    dependency_tool_eta_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 10))
    dependency_tool_buttons = tk.Frame(dependency_tool_shell, bg=dependency_tool_shell["bg"])
    dependency_tool_buttons.grid(row=0, column=2, rowspan=2, sticky="e")

    def _tree_mousewheel(tree: ttk.Treeview, event) -> str:
        try:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                units = -max(-6, min(6, int(getattr(event, "delta", 0) / 120))) or 0
            if units:
                tree.yview_scroll(units, "units")
        except Exception:
            pass
        return "break"

    for tree in (candidate_tree,):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                tree.bind(sequence, lambda event, current_tree=tree: _tree_mousewheel(current_tree, event), add="+")
            except Exception:
                pass

    def _form_mousewheel(event) -> str:
        try:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                units = -max(-6, min(6, int(getattr(event, "delta", 0) / 120))) or 0
            if units:
                form_canvas.yview_scroll(units, "units")
        except Exception:
            pass
        return "break"

    def _bind_export_form_mousewheel(widget) -> None:
        try:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, _form_mousewheel, add="+")
        except Exception:
            pass
        try:
            children = list(widget.winfo_children())
        except Exception:
            children = []
        for child in children:
            _bind_export_form_mousewheel(child)

    _bind_export_form_mousewheel(form_shell)

    chart_redraw_state = {"after": None, "overview": (0, 0), "metric": (0, 0)}

    def redraw_charts(_event=None) -> None:
        done = _mobile_export_perf_timer("chart_redraw")
        chart_redraw_state["after"] = None
        if _mobile_export_window_is_moving():
            try:
                chart_redraw_state["after"] = dialog.after(280, redraw_charts)
            except Exception:
                pass
            done(candidate="deferred_move")
            return
        candidate = selected_state.get("candidate")
        if not candidate:
            done(candidate=0)
            return
        try:
            _draw_mobile_export_overview_chart(
                overview_canvas,
                candidates,
                str(candidate.get("iid") or ""),
                palette,
                _selected_export_iids(),
            )
            _draw_mobile_export_metric_chart(metric_chart_canvas, candidate, palette, max_candidate_size_mb)
        except Exception:
            pass
        finally:
            done(candidate=1)

    def _queue_chart_redraw(_event=None) -> None:
        started = time.perf_counter()
        try:
            if _mobile_export_window_is_moving():
                pending = chart_redraw_state.get("after")
                if pending:
                    try:
                        dialog.after_cancel(pending)
                    except Exception:
                        pass
                chart_redraw_state["after"] = dialog.after(320, redraw_charts)
                _mobile_export_perf_record(
                    "chart_redraw_deferred_move",
                    (time.perf_counter() - started) * 1000.0,
                    reason="window_move",
                )
                return
            delay = 120
            overview_size = (
                int(overview_canvas.winfo_width() or 0),
                int(overview_canvas.winfo_height() or 0),
            )
            metric_size = (
                int(metric_chart_canvas.winfo_width() or 0),
                int(metric_chart_canvas.winfo_height() or 0),
            )
            if (
                overview_size == chart_redraw_state.get("overview")
                and metric_size == chart_redraw_state.get("metric")
            ):
                return
            chart_redraw_state["overview"] = overview_size
            chart_redraw_state["metric"] = metric_size
            pending = chart_redraw_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
            chart_redraw_state["after"] = dialog.after(delay, redraw_charts)
            _mobile_export_perf_record(
                "chart_redraw_queue",
                (time.perf_counter() - started) * 1000.0,
                overview=f"{overview_size[0]}x{overview_size[1]}",
                metric=f"{metric_size[0]}x{metric_size[1]}",
            )
        except Exception:
            pass

    try:
        overview_canvas.bind("<Configure>", _queue_chart_redraw, add="+")
        metric_chart_canvas.bind("<Configure>", _queue_chart_redraw, add="+")
    except Exception:
        pass

    actions = tk.Frame(root, bg=bg)
    actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
    actions.grid_columnconfigure(0, weight=1)

    exporter = MobileModelExporter()
    package_exporter = MobileAlprPackageExporter(exporter)
    preflight_button = None
    dependency_install_button = None
    dependency_pause_button = None
    complete_package_button = None
    export_button = None
    close_button = None
    preflight_state = {
        "request": None,
        "problems": [],
        "ready": False,
        "installable": False,
    }
    dependency_install_control_state = {
        "running": False,
        "pause_requested": False,
        "paused": False,
    }

    def set_status(text: str, tone: str = "info") -> None:
        status_canvas_state["tone"] = str(tone or "info")
        status_var.set(text)
        _queue_mobile_export_status_redraw(delay=0)

    def _open_legacy_url_pose_importer() -> None:
        import_dialog = tk.Toplevel(dialog)
        title = "Importuj model YOLO pose z sieci"
        try:
            self.app.style_dialog_window(
                import_dialog,
                title=title,
                geometry="760x420",
                parent=dialog,
            )
        except Exception:
            import_dialog.title(title)
            import_dialog.geometry("760x420")
        try:
            import_dialog.transient(dialog)
            import_dialog.resizable(True, True)
            import_dialog.minsize(680, 360)
        except Exception:
            pass
        _bind_mobile_export_child_window_motion(import_dialog, name="network_pose_import_configure")
        try:
            import_dialog.after_idle(
                lambda w=import_dialog: _install_mobile_export_win32_move_guard(w, guard_key=f"network-pose-{id(w)}")
            )
        except Exception:
            pass

        import_bg = bg
        import_card_bg = blend_hex_colors(card_bg, accent, 0.035)
        root_import = tk.Frame(import_dialog, bg=import_bg, padx=18, pady=16)
        root_import.pack(fill=tk.BOTH, expand=True)
        root_import.grid_columnconfigure(0, weight=1)
        root_import.grid_rowconfigure(3, weight=1)

        tk.Label(
            root_import,
            text="Import zewnętrznego modelu MT",
            bg=import_bg,
            fg=fg,
            font=("Segoe UI", 14, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            root_import,
            text=(
                "Wklej bezpośredni link do pliku .pt. Model zostanie pobrany do katalogu base/pose, "
                "zwalidowany jako YOLO pose i dopiero wtedy pojawi się na liście kandydatów eksportu jako MT. "
                "To powinien być model tablic pose, a nie ogólny model human-pose."
            ),
            bg=import_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=700,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        fields = tk.Frame(
            root_import,
            bg=import_card_bg,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(accent, import_bg, 0.45),
        )
        fields.grid(row=2, column=0, sticky="ew")
        fields.grid_columnconfigure(1, weight=1)

        url_var = tk.StringVar()
        name_var = tk.StringVar()
        import_status_var = tk.StringVar(value="Gotowe do pobrania modelu.")
        import_progress_var = tk.DoubleVar(value=0.0)
        import_percent_var = tk.StringVar(value="0%")
        import_state = {"running": False, "cancel": False}

        tk.Label(fields, text="URL modelu .pt", bg=import_card_bg, fg=muted, font=("Segoe UI", 9, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=(0, 7),
        )
        url_entry = ttk.Entry(fields, textvariable=url_var)
        url_entry.grid(row=0, column=1, sticky="ew", pady=(0, 7))
        tk.Label(fields, text="Nazwa w programie", bg=import_card_bg, fg=muted, font=("Segoe UI", 9, "bold")).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=(0, 7),
        )
        ttk.Entry(fields, textvariable=name_var).grid(row=1, column=1, sticky="ew", pady=(0, 7))
        tk.Label(
            fields,
            text="Opcjonalna nazwa pomaga odróżnić model w tabeli. Rozszerzenie .pt zostanie dopisane automatycznie.",
            bg=import_card_bg,
            fg=muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=610,
        ).grid(row=2, column=0, columnspan=2, sticky="ew")

        progress_box = tk.Frame(root_import, bg=import_bg)
        progress_box.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        progress_box.grid_columnconfigure(0, weight=1)
        tk.Label(
            progress_box,
            textvariable=import_status_var,
            bg=import_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=700,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        import_progress = ttk.Progressbar(progress_box, mode="determinate", maximum=100.0, variable=import_progress_var)
        import_progress.grid(row=1, column=0, sticky="ew")
        tk.Label(
            progress_box,
            textvariable=import_percent_var,
            bg=import_bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            width=5,
            anchor=tk.E,
        ).grid(row=1, column=1, sticky="e", padx=(10, 0))

        actions_import = tk.Frame(root_import, bg=import_bg)
        actions_import.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        actions_import.grid_columnconfigure(0, weight=1)

        def _set_import_running(running: bool) -> None:
            import_state["running"] = bool(running)
            try:
                import_button.configure(state=tk.DISABLED if running else tk.NORMAL)
            except Exception:
                pass
            try:
                close_button.configure(text="Przerwij" if running else "Zamknij")
            except Exception:
                pass

        def _update_import_status(text: str, tone: str = "info", percent: float | None = None) -> None:
            colors = {"success": success, "warning": warning, "error": error, "info": muted}
            try:
                import_status_var.set(str(text or "-"))
            except Exception:
                pass
            try:
                for child in progress_box.winfo_children():
                    if isinstance(child, tk.Label) and str(child.cget("textvariable")) == str(import_status_var):
                        child.configure(fg=colors.get(str(tone or "info"), muted))
            except Exception:
                pass
            if percent is not None:
                try:
                    value = _mobile_export_progress_value(percent)
                    import_progress_var.set(value)
                    import_percent_var.set(_mobile_export_progress_label(value))
                except Exception:
                    pass

        def _finish_network_pose_import(destination: Path, info: dict, validation_message: str) -> None:
            candidate = _mobile_export_candidate_from_artifact(
                self,
                destination,
                "plate",
                "Import sieciowy YOLO pose",
            )
            candidate["model_info"] = info if isinstance(info, dict) else candidate.get("model_info", {})
            candidate["model_metadata"] = _mobile_export_read_model_metadata(destination)
            candidate_iid = _register_mobile_export_candidate(candidate)
            if candidate_iid:
                export_selection_state["MT"] = candidate_iid
            _populate_candidate_tree(refresh_selection=True)
            try:
                candidate_tree.selection_set(candidate_iid)
                candidate_tree.focus(candidate_iid)
                candidate_tree.see(candidate_iid)
            except Exception:
                pass
            apply_candidate(candidate, preflight_after=True)
            _update_export_selection_view(announce=True)
            set_status(f"Zaimportowano model MT z sieci: {destination.name}", "success")
            _update_import_status(validation_message, "success", 100.0)

        def start_import() -> None:
            if import_state.get("running"):
                return
            raw_url = str(url_var.get() or "").strip()
            parsed = urllib.parse.urlparse(raw_url)
            if str(parsed.scheme or "").lower() not in {"http", "https"} or not str(parsed.netloc or "").strip():
                return messagebox.showwarning(
                    "Niepoprawny URL",
                    "Wklej bezpośredni link http/https do pliku .pt modelu YOLO pose.",
                    parent=import_dialog,
                )
            import_state["cancel"] = False
            _set_import_running(True)
            _update_import_status("Przygotowuję pobieranie modelu...", "info", 0.0)

            def worker() -> None:
                temp_path = None
                try:
                    target_dir = Path(CONFIG.get_base_models_dir("plate")) / "network"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    filename = _mobile_export_external_pose_filename(raw_url, name_var.get())
                    destination = target_dir / filename
                    temp_path = target_dir / f".{Path(filename).stem}.download.pt"
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                    request = urllib.request.Request(
                        raw_url,
                        headers={"User-Agent": "AutoAnnotationTool-MobileExporter/1.0"},
                    )

                    def ui(text: str, tone: str = "info", percent: float | None = None) -> None:
                        try:
                            import_dialog.after(0, lambda: _update_import_status(text, tone, percent))
                        except Exception:
                            pass

                    ui("Łączę się z adresem modelu...", "info", 3.0)
                    with urllib.request.urlopen(request, timeout=60) as response:
                        total_raw = response.headers.get("Content-Length")
                        try:
                            total_bytes = max(0, int(total_raw or 0))
                        except Exception:
                            total_bytes = 0
                        downloaded = 0
                        chunks = 0
                        with temp_path.open("wb") as handle:
                            while True:
                                if import_state.get("cancel"):
                                    raise MobileExportError("Import modelu został przerwany przez użytkownika.")
                                chunk = response.read(1024 * 512)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                downloaded += len(chunk)
                                chunks += 1
                                if total_bytes > 0:
                                    percent = 5.0 + min(1.0, downloaded / total_bytes) * 65.0
                                    size_text = f"{downloaded / (1024 * 1024):.1f}/{total_bytes / (1024 * 1024):.1f} MB"
                                else:
                                    percent = min(70.0, 5.0 + chunks * 1.5)
                                    size_text = f"{downloaded / (1024 * 1024):.1f} MB"
                                ui(f"Pobieram model YOLO pose: {size_text}", "info", percent)
                    if not temp_path.exists() or temp_path.stat().st_size <= 0:
                        raise MobileExportError("Pobrany plik jest pusty.")

                    ui("Waliduję plik jako model YOLO pose...", "info", 76.0)
                    ok, validation_message, model_info = validate_model_file(
                        temp_path,
                        prefer_sidecar=False,
                        allow_heavy_load=True,
                        write_sidecar=False,
                    )
                    if not ok:
                        raise MobileExportError(f"Walidacja modelu nie powiodła się: {validation_message}")
                    pose_ok, pose_message = _mobile_export_pose_validation_message(model_info)
                    if not pose_ok:
                        raise MobileExportError(pose_message)

                    if destination.exists():
                        stem = destination.stem
                        suffix = destination.suffix or ".pt"
                        counter = 2
                        while destination.exists():
                            destination = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.move(str(temp_path), str(destination))
                    temp_path = None
                    write_model_metadata_sidecar(
                        destination,
                        model_info,
                        validation_ok=True,
                        validation_message=pose_message,
                        extra={
                            "source": {
                                "kind": "network_url",
                                "url": raw_url,
                                "imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                            },
                            "target": "plate",
                            "role": "plate",
                            "task": "pose",
                        },
                    )

                    def done() -> None:
                        _set_import_running(False)
                        _finish_network_pose_import(destination, model_info, pose_message)

                    import_dialog.after(0, done)
                except Exception as exc:
                    error_text = str(exc)
                    logger.exception("Nie udało się zaimportować modelu YOLO pose z sieci")
                    if temp_path is not None:
                        try:
                            Path(temp_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                    def failed() -> None:
                        _set_import_running(False)
                        _update_import_status(f"Import nieudany: {error_text}", "error", 0.0)
                        set_status(f"Nie udało się zaimportować modelu MT z sieci: {error_text}", "error")
                        messagebox.showerror("Błąd importu modelu", error_text, parent=import_dialog)

                    try:
                        import_dialog.after(0, failed)
                    except Exception:
                        pass

            threading.Thread(target=worker, name="mobile-export-network-pose-import", daemon=True).start()

        def close_or_cancel() -> None:
            if import_state.get("running"):
                import_state["cancel"] = True
                _update_import_status("Przerywam import po zakończeniu bieżącej porcji danych...", "warning")
                return
            try:
                import_dialog.destroy()
            except Exception:
                pass

        import_button = ttk.Button(actions_import, text="Importuj model MT", command=start_import)
        import_button.grid(row=0, column=1, sticky="e", padx=(0, 8), ipadx=10, ipady=3)
        close_button = ttk.Button(actions_import, text="Zamknij", command=close_or_cancel)
        close_button.grid(row=0, column=2, sticky="e", ipadx=8, ipady=3)
        try:
            import_dialog.protocol("WM_DELETE_WINDOW", close_or_cancel)
            url_entry.focus_set()
        except Exception:
            pass

    def open_vehicle_model_importer() -> None:
        import_dialog = tk.Toplevel(dialog)
        title = "Import online modelu pojazdów"
        try:
            self.app.style_dialog_window(
                import_dialog,
                title=title,
                geometry="760x430",
                parent=dialog,
            )
        except Exception:
            import_dialog.title(title)
            import_dialog.geometry("760x430")
        try:
            import_dialog.transient(dialog)
            import_dialog.resizable(True, True)
            import_dialog.minsize(680, 370)
        except Exception:
            pass
        _bind_mobile_export_child_window_motion(import_dialog, name="vehicle_model_import_configure")

        import_bg = bg
        import_card_bg = blend_hex_colors(card_bg, accent, 0.035)
        root_import = tk.Frame(import_dialog, bg=import_bg, padx=18, pady=16)
        root_import.pack(fill=tk.BOTH, expand=True)
        root_import.grid_columnconfigure(0, weight=1)
        root_import.grid_rowconfigure(3, weight=1)

        def _selected_import_target() -> str:
            return "vehicle"

        def _selected_import_marker() -> str:
            return "MP"

        catalog = _mobile_export_ultralytics_catalog_for_target(_selected_import_target())
        model_keys = sorted(
            catalog,
            key=lambda item: (
                0 if item.startswith("yolo26") else 1 if item.startswith("yolo11") else 2,
                item,
            ),
        )
        label_to_key: dict[str, str] = {}
        labels: list[str] = []
        for key in model_keys:
            info = catalog.get(key) or {}
            label = _mobile_export_catalog_label(key, info, include_local_state=False)
            label_to_key[label] = key
            labels.append(label)
        catalog_state = {"catalog": catalog, "label_to_key": label_to_key}

        default_key = next((key for key in model_keys if key == "yolo26n"), model_keys[0] if model_keys else "")
        default_label = next((label for label, key in label_to_key.items() if key == default_key), labels[0] if labels else "")
        model_var = tk.StringVar(value=default_label)
        import_status_var = tk.StringVar(value="Wybierz model z katalogu online. Lokalne MP są już na liście kandydatów.")
        import_progress_var = tk.DoubleVar(value=0.0)
        import_percent_var = tk.StringVar(value="0%")
        import_state = {"running": False, "cancel": False}

        tk.Label(
            root_import,
            text="Import online modelu pojazdów",
            bg=import_bg,
            fg=fg,
            font=("Segoe UI", 14, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            root_import,
            text=(
                "Wybierz detektor pojazdów z katalogu Ultralytics. Ten modal służy tylko do importu online "
                "brakujących modeli MP. Modele, które są już w katalogach programu, pojawiają się automatycznie "
                "na liście kandydatów w głównym oknie eksportu."
            ),
            bg=import_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=700,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        fields = tk.Frame(
            root_import,
            bg=import_card_bg,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(accent, import_bg, 0.45),
        )
        fields.grid(row=2, column=0, sticky="ew")
        fields.grid_columnconfigure(1, weight=1)

        tk.Label(fields, text="Model Ultralytics", bg=import_card_bg, fg=muted, font=("Segoe UI", 9, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=(0, 7),
        )
        model_combo = ttk.Combobox(fields, textvariable=model_var, values=labels, state="readonly")
        model_combo.grid(row=0, column=1, sticky="ew", pady=(0, 7))
        model_hint_var = tk.StringVar(value="")
        tk.Label(
            fields,
            textvariable=model_hint_var,
            bg=import_card_bg,
            fg=muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=610,
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

        progress_box = tk.Frame(root_import, bg=import_bg)
        progress_box.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        progress_box.grid_columnconfigure(0, weight=1)
        import_status_label = tk.Label(
            progress_box,
            textvariable=import_status_var,
            bg=import_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=700,
        )
        import_status_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        import_progress = ttk.Progressbar(progress_box, mode="determinate", maximum=100.0, variable=import_progress_var)
        import_progress.grid(row=1, column=0, sticky="ew")
        tk.Label(
            progress_box,
            textvariable=import_percent_var,
            bg=import_bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            width=5,
            anchor=tk.E,
        ).grid(row=1, column=1, sticky="e", padx=(10, 0))

        actions_import = tk.Frame(root_import, bg=import_bg)
        actions_import.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        actions_import.grid_columnconfigure(0, weight=1)

        def _catalog_sort_key(item: str) -> tuple[int, str]:
            return (
                0 if item.startswith("yolo26") else 1 if item.startswith("yolo11") else 2,
                item,
            )

        def _refresh_catalog_options(*, preferred_key: str = "") -> None:
            target = _selected_import_target()
            refreshed_catalog = _mobile_export_ultralytics_catalog_for_target(target)
            refreshed_keys = sorted(refreshed_catalog, key=_catalog_sort_key)
            refreshed_labels: list[str] = []
            refreshed_label_to_key: dict[str, str] = {}
            for key in refreshed_keys:
                info = refreshed_catalog.get(key) or {}
                label = _mobile_export_catalog_label(key, info, include_local_state=False)
                refreshed_label_to_key[label] = key
                refreshed_labels.append(label)
            catalog_state["catalog"] = refreshed_catalog
            catalog_state["label_to_key"] = refreshed_label_to_key
            default_key = preferred_key or ("yolo26n" if target == "vehicle" else "yolo26n-pose")
            if default_key not in refreshed_catalog:
                default_key = refreshed_keys[0] if refreshed_keys else ""
            default_label = next(
                (label for label, key in refreshed_label_to_key.items() if key == default_key),
                refreshed_labels[0] if refreshed_labels else "",
            )
            try:
                model_combo.configure(values=refreshed_labels)
            except Exception:
                pass
            try:
                model_var.set(default_label)
            except Exception:
                pass

        def _selected_catalog_entry() -> tuple[str, dict]:
            selected_label = str(model_var.get() or "").strip()
            label_map = catalog_state.get("label_to_key") if isinstance(catalog_state.get("label_to_key"), dict) else {}
            catalog_map = catalog_state.get("catalog") if isinstance(catalog_state.get("catalog"), dict) else {}
            key = label_map.get(selected_label, "")
            if not key and selected_label:
                key = selected_label.split("|", 1)[0].strip()
            info = dict(catalog_map.get(key) or {})
            if key and not info:
                info = {"name": key, "file": f"{key}.pt"}
            return key, info

        def _refresh_model_hint(_event=None) -> None:
            key, info = _selected_catalog_entry()
            file_name = str(info.get("file") or f"{key}.pt").strip()
            target = _selected_import_target()
            marker = _selected_import_marker()
            local = _mobile_export_find_local_catalog_model(file_name, target=target)
            target_dir_name = "base/detect" if target == "vehicle" else "base/pose"
            if local:
                model_hint_var.set(
                    f"{file_name} jest już lokalnie, więc model powinien być widoczny na głównej liście kandydatów jako {marker}."
                )
            else:
                model_hint_var.set(
                    f"Po wyborze program pobierze {file_name} przez Ultralytics do katalogu {target_dir_name} i doda go jako {marker}."
                )

        def _set_import_running(running: bool) -> None:
            import_state["running"] = bool(running)
            try:
                import_button.configure(state=tk.DISABLED if running else tk.NORMAL)
            except Exception:
                pass
            try:
                model_combo.configure(state=tk.DISABLED if running else "readonly")
            except Exception:
                pass
            try:
                close_button.configure(text="Przerwij" if running else "Zamknij")
            except Exception:
                pass

        def _update_import_status(text: str, tone: str = "info", percent: float | None = None) -> None:
            colors = {"success": success, "warning": warning, "error": error, "info": muted}
            try:
                import_status_var.set(str(text or "-"))
                import_status_label.configure(fg=colors.get(str(tone or "info"), muted))
            except Exception:
                pass
            if percent is not None:
                try:
                    value = _mobile_export_progress_value(percent)
                    import_progress_var.set(value)
                    import_percent_var.set(_mobile_export_progress_label(value))
                except Exception:
                    pass

        def _finish_catalog_model_import(destination: Path, info: dict, validation_message: str, target: str) -> None:
            marker = "MP" if CONFIG.normalize_task_target(target) == "vehicle" else "MT"
            source_label = "Model Ultralytics YOLO detect" if marker == "MP" else "Model Ultralytics YOLO pose"
            destination_key = _mobile_export_safe_path_key(destination)
            existing_iid = next(
                (
                    iid
                    for iid, existing in candidate_by_iid.items()
                    if _mobile_export_safe_path_key(existing.get("best_weights")) == destination_key
                ),
                "",
            )
            if existing_iid:
                candidate = candidate_by_iid.get(existing_iid) or {}
                candidate_iid = existing_iid
            else:
                candidate = _mobile_export_candidate_from_artifact(
                    self,
                    destination,
                    target,
                    source_label,
                )
                candidate_iid = _register_mobile_export_candidate(candidate)
            candidate["model_info"] = info if isinstance(info, dict) else candidate.get("model_info", {})
            candidate["model_metadata"] = _mobile_export_read_model_metadata(destination)
            if candidate_iid:
                export_selection_state[marker] = candidate_iid
            _populate_candidate_tree(refresh_selection=True)
            try:
                candidate_tree.selection_set(candidate_iid)
                candidate_tree.focus(candidate_iid)
                candidate_tree.see(candidate_iid)
            except Exception:
                pass
            apply_candidate(candidate, preflight_after=True)
            _update_export_selection_view(announce=True)
            set_status(f"Dodano model {marker}: {destination.name}", "success")
            _update_import_status(validation_message, "success", 100.0)

        def start_import() -> None:
            if import_state.get("running"):
                return
            target = _selected_import_target()
            marker = _selected_import_marker()
            key, info = _selected_catalog_entry()
            if not key:
                return messagebox.showwarning("Brak modelu", "Wybierz model YOLO z listy.", parent=import_dialog)
            file_name = str(info.get("file") or f"{key}.pt").strip()
            local_path = _mobile_export_find_local_catalog_model(file_name, target=target)
            destination = _mobile_export_catalog_import_destination(file_name, target=target)
            import_state["cancel"] = False
            _set_import_running(True)
            _update_import_status("Sprawdzam lokalne katalogi modeli...", "info", 5.0)

            def worker() -> None:
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    def ui(text: str, tone: str = "info", percent: float | None = None) -> None:
                        try:
                            import_dialog.after(0, lambda: _update_import_status(text, tone, percent))
                        except Exception:
                            pass

                    source_path = local_path
                    if source_path and source_path.exists():
                        ui(f"Model jest już lokalnie: {source_path}", "info", 25.0)
                        try:
                            if _mobile_export_safe_path_key(source_path) != _mobile_export_safe_path_key(destination):
                                shutil.copy2(source_path, destination)
                                source_path = destination
                        except Exception as exc:
                            raise MobileExportError(f"Nie udało się skopiować lokalnego modelu do katalogu importu: {exc}") from exc
                    else:
                        if not YOLO_AVAILABLE:
                            raise MobileExportError("Ultralytics jest niedostępny, więc nie można pobrać modelu z listy.")
                        ui(f"Pobieram {file_name} przez Ultralytics...", "info", 18.0)
                        from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

                        downloaded = Path(attempt_download_asset(str(destination)))
                        if not downloaded.exists():
                            raise MobileExportError(f"Pobieranie modelu {file_name} nie utworzyło pliku.")
                        if _mobile_export_safe_path_key(downloaded) != _mobile_export_safe_path_key(destination):
                            shutil.copy2(downloaded, destination)
                        source_path = destination

                    if import_state.get("cancel"):
                        raise MobileExportError("Import modelu został przerwany przez użytkownika.")
                    if not destination.exists() or destination.stat().st_size <= 0:
                        raise MobileExportError("Model nie istnieje albo pobrany plik jest pusty.")
                    ui(f"Waliduję model jako {marker}...", "info", 72.0)
                    ok, validation_message, model_info = validate_model_file(
                        destination,
                        prefer_sidecar=False,
                        allow_heavy_load=True,
                        write_sidecar=False,
                    )
                    if not ok:
                        raise MobileExportError(f"Walidacja modelu nie powiodła się: {validation_message}")
                    role_ok, role_message = _mobile_export_catalog_validation_message(model_info, target)
                    if not role_ok:
                        raise MobileExportError(role_message)
                    write_model_metadata_sidecar(
                        destination,
                        model_info,
                        validation_ok=True,
                        validation_message=role_message,
                        extra={
                            "source": {
                                "kind": "ultralytics_catalog",
                                "model_key": key,
                                "asset_name": file_name,
                                "local_source": str(source_path or ""),
                                "imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                            },
                            "target": target,
                            "role": "vehicle" if marker == "MP" else "plate",
                            "task": "detect" if marker == "MP" else "pose",
                        },
                    )

                    def done() -> None:
                        _set_import_running(False)
                        _finish_catalog_model_import(destination, model_info, role_message, target)
                        _refresh_model_hint()

                    import_dialog.after(0, done)
                except Exception as exc:
                    error_text = str(exc)
                    logger.exception("Nie udało się dodać modelu YOLO z katalogu Ultralytics")

                    def failed() -> None:
                        _set_import_running(False)
                        _update_import_status(f"Import nieudany: {error_text}", "error", 0.0)
                        set_status(f"Nie udało się dodać modelu {marker}: {error_text}", "error")
                        messagebox.showerror("Błąd importu modelu", error_text, parent=import_dialog)

                    try:
                        import_dialog.after(0, failed)
                    except Exception:
                        pass

            threading.Thread(target=worker, name="mobile-export-ultralytics-vehicle-import", daemon=True).start()

        def close_or_cancel() -> None:
            if import_state.get("running"):
                import_state["cancel"] = True
                _update_import_status("Przerywam import po zakończeniu bieżącego kroku...", "warning")
                return
            try:
                import_dialog.destroy()
            except Exception:
                pass

        def _on_catalog_model_selected(_event=None) -> None:
            _refresh_model_hint()
            if import_state.get("running"):
                return
            try:
                import_dialog.after(80, start_import)
            except Exception:
                start_import()

        import_button = ttk.Button(actions_import, text="Dodaj wybrany model", command=start_import)
        import_button.grid(row=0, column=1, sticky="e", padx=(0, 8), ipadx=10, ipady=3)
        close_button = ttk.Button(actions_import, text="Zamknij", command=close_or_cancel)
        close_button.grid(row=0, column=2, sticky="e", ipadx=8, ipady=3)
        try:
            model_combo.bind("<<ComboboxSelected>>", _on_catalog_model_selected, add="+")
            import_dialog.protocol("WM_DELETE_WINDOW", close_or_cancel)
            _refresh_model_hint()
            model_combo.focus_set()
        except Exception:
            pass

    try:
        candidate_help_shell.grid_columnconfigure(0, weight=1)
        import_pose_button = ttk.Button(
            candidate_help_shell,
            text="Import online modelu pojazdów do listy kandydatów",
            command=open_vehicle_model_importer,
        )
        import_pose_button.grid(row=1, column=0, sticky="ew", pady=(7, 0))
    except Exception:
        pass

    def _set_dependency_tool(
        text: str,
        tone: str = "info",
        *,
        eta: str | None = None,
        progress_value: float | None = None,
    ) -> None:
        colors = {
            "success": success,
            "warning": warning,
            "error": error,
            "info": muted,
        }
        try:
            dependency_tool_status_label.configure(fg=colors.get(str(tone or "info"), muted))
        except Exception:
            pass
        try:
            dependency_tool_eta_label.configure(fg=colors.get(str(tone or "info"), muted))
        except Exception:
            pass
        try:
            dependency_tool_status_var.set(str(text or "-"))
        except Exception:
            pass
        if eta is not None:
            eta_text = str(eta or "-")
            if not eta_text.lower().startswith("eta"):
                eta_text = f"ETA: {eta_text}"
            try:
                dependency_tool_eta_var.set(eta_text)
            except Exception:
                pass
        if progress_value is not None:
            try:
                progress.stop()
                progress.configure(mode="determinate")
                percent_value = _mobile_export_progress_value(progress_value)
                progress_var.set(percent_value)
                progress_percent_var.set(_mobile_export_progress_label(percent_value))
            except Exception:
                pass

    def _update_dependency_buttons() -> None:
        try:
            running = bool(worker_state.get("running"))
        except Exception:
            running = False
        install_running = bool(dependency_install_control_state.get("running"))
        installable = bool(preflight_state.get("installable"))
        for widget, state in (
            (preflight_button, tk.DISABLED if running else tk.NORMAL),
            (dependency_install_button, tk.NORMAL if installable and not running else tk.DISABLED),
            (dependency_pause_button, tk.NORMAL if install_running else tk.DISABLED),
        ):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except Exception:
                pass
        try:
            if dependency_pause_button is not None:
                dependency_pause_button.configure(
                    text="Wznów" if dependency_install_control_state.get("pause_requested") else "Pauza"
                )
        except Exception:
            pass

    def _set_dependency_pause_requested(paused: bool) -> None:
        if not dependency_install_control_state.get("running"):
            return
        dependency_install_control_state["pause_requested"] = bool(paused)
        dependency_install_control_state["paused"] = bool(paused)
        if paused:
            _set_dependency_tool(
                "Pauza jest uzbrojona. Instalator zatrzyma się przed następną zależnością.",
                "warning",
                eta="pauza",
            )
        else:
            _set_dependency_tool("Wznawiam instalację zależności eksportu.", "info", eta="liczę")
        _update_dependency_buttons()

    def toggle_dependency_install_pause() -> None:
        _set_dependency_pause_requested(not bool(dependency_install_control_state.get("pause_requested")))

    def _preflight_has_installable_dependency_problem(problems: list[str] | tuple[str, ...] | None) -> bool:
        blob = " ".join(str(item or "").lower() for item in (problems or ()))
        return any(
            token in blob
            for token in (
                "brak pakietu",
                "brak zależności",
                "ultralytics",
                "torch",
                "torchvision",
                "runtime yolo",
                "niespójne zależności yolo",
                "nie można zaimportować",
                "onnx",
                "onnxruntime",
                "onnxslim",
                "tensorflow",
                "tf_keras",
                "sng4onnx",
                "onnx_graphsurgeon",
                "ai-edge-litert",
                "onnx2tf",
                "protobuf",
                "ncnn",
                "pnnx",
                "tflite",
                "nie spełnia",
            )
        )

    def _missing_install_requirements_for_request(request: MobileExportRequest | None) -> list[dict]:
        if request is None:
            return []
        try:
            specs = mobile_export_required_specs(
                tuple(request.formats or ()),
                quantizations=tuple(getattr(request, "quantizations", ()) or ()),
                format_quantizations=getattr(request, "format_quantizations", {}) or {},
            )
        except Exception:
            specs = []
        rows: list[dict] = []
        for item in specs:
            spec = str(item.get("spec") or "").strip()
            module_name = str(item.get("module") or "").strip()
            if not spec:
                continue
            try:
                ok, detail = mobile_export_requirement_status(spec, module_name)
            except Exception as exc:
                ok = False
                detail = str(exc)
            if ok:
                continue
            rows.append(
                {
                    "spec": spec,
                    "name": _mobile_export_requirement_name(spec),
                    "models": str(item.get("scope") or "Eksport"),
                    "description": detail,
                }
            )
        return rows

    def show_preflight_problems_dialog(
        problems: list[str] | tuple[str, ...],
        *,
        parent=None,
        title: str = "Sprawdzenie gotowości eksportu",
        context: str = "Eksport nie może jeszcze ruszyć.",
        after_install=None,
    ) -> None:
        normalized = [str(item or "").strip() for item in (problems or ()) if str(item or "").strip()]
        parent_widget = parent or dialog
        modal = tk.Toplevel(parent_widget)
        try:
            self.app.style_dialog_window(
                modal,
                title=title,
                geometry="760x560",
                parent=parent_widget,
            )
        except Exception:
            modal.title(title)
            modal.geometry("760x560")
        try:
            modal.transient(parent_widget)
            modal.resizable(True, True)
            modal.minsize(680, 460)
        except Exception:
            pass
        _bind_mobile_export_child_window_motion(modal, name="readiness_dialog_configure")
        try:
            modal.after_idle(lambda w=modal: _install_mobile_export_win32_move_guard(w, guard_key=f"readiness-{id(w)}"))
        except Exception:
            pass

        shell = tk.Frame(modal, bg=bg, padx=18, pady=16, highlightthickness=1, highlightbackground=border)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(2, weight=1)

        tk.Label(
            shell,
            text="Eksport wymaga dopięcia kilku rzeczy",
            bg=bg,
            fg=fg,
            font=("Segoe UI", 13, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            shell,
            text=context,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=600,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 10))

        problem_text = tk.Text(
            shell,
            height=9,
            bg=card_bg,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )
        problem_text.grid(row=2, column=0, sticky="nsew")
        problem_text.insert("1.0", "\n".join(f"- {item}" for item in normalized) or "-")
        problem_text.configure(state=tk.DISABLED)

        local_status_var = tk.StringVar(
            value=(
                "Jeśli brakuje bibliotek eksportu, zainstaluj je tutaj. "
                "Jeśli brakuje danych lub ścieżki, popraw ustawienia w modalu eksportu."
            )
        )
        tk.Label(
            shell,
            textvariable=local_status_var,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=600,
        ).grid(row=3, column=0, sticky="ew", pady=(10, 6))
        local_progress_var = tk.DoubleVar(value=0.0)
        local_progress = ttk.Progressbar(shell, mode="determinate", maximum=100.0, variable=local_progress_var)
        local_progress.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        install_shell = tk.Frame(
            shell,
            bg=blend_hex_colors(card_bg, accent, 0.025),
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(accent, bg, 0.45),
        )
        install_shell.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        install_shell.grid_remove()
        install_shell.grid_columnconfigure(0, weight=1)
        install_shell.grid_columnconfigure(1, weight=2)
        install_shell.grid_columnconfigure(3, weight=1)
        install_rows: dict[int, dict] = {}

        def build_install_rows(items: list[dict]) -> None:
            try:
                for child in install_shell.winfo_children():
                    child.destroy()
                install_rows.clear()
                install_shell.grid()
                headers = ("Pakiet", "Aktualny krok", "Stan", "Postęp")
                for col, header in enumerate(headers):
                    tk.Label(
                        install_shell,
                        text=header,
                        bg=install_shell["bg"],
                        fg=fg,
                        font=("Segoe UI", 8, "bold"),
                        anchor=tk.W,
                    ).grid(row=0, column=col, sticky="ew", padx=(0, 8), pady=(0, 5))
                for idx, item in enumerate(items):
                    row = idx + 1
                    size_var = tk.StringVar(value="-")
                    status_row_var = tk.StringVar(value="Czeka")
                    progress_row_var = tk.DoubleVar(value=0.0)
                    progress_text_var = tk.StringVar(value="0%")
                    row_bg = blend_hex_colors(install_shell["bg"], fg, 0.018 if idx % 2 else 0.032)
                    tk.Label(
                        install_shell,
                        text=str(item.get("spec") or item.get("name") or "-"),
                        bg=row_bg,
                        fg=fg,
                        font=("Segoe UI", 8, "bold"),
                        anchor=tk.W,
                        padx=7,
                        pady=5,
                    ).grid(row=row, column=0, sticky="nsew", padx=(0, 4), pady=2)
                    tk.Label(
                        install_shell,
                        textvariable=size_var,
                        bg=row_bg,
                        fg=muted,
                        font=("Segoe UI", 8),
                        anchor=tk.W,
                        justify=tk.LEFT,
                        wraplength=260,
                        padx=7,
                        pady=5,
                    ).grid(row=row, column=1, sticky="nsew", padx=(0, 4), pady=2)
                    status_label = tk.Label(
                        install_shell,
                        textvariable=status_row_var,
                        bg=row_bg,
                        fg=muted,
                        font=("Segoe UI", 8, "bold"),
                        width=16,
                        anchor=tk.CENTER,
                        padx=7,
                        pady=5,
                    )
                    status_label.grid(row=row, column=2, sticky="nsew", padx=(0, 4), pady=2)
                    progress_cell = tk.Frame(install_shell, bg=row_bg, padx=7, pady=5)
                    progress_cell.grid(row=row, column=3, sticky="nsew", pady=2)
                    progress_cell.grid_columnconfigure(0, weight=1)
                    ttk.Progressbar(
                        progress_cell,
                        mode="determinate",
                        maximum=100.0,
                        variable=progress_row_var,
                    ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
                    tk.Label(
                        progress_cell,
                        textvariable=progress_text_var,
                        bg=row_bg,
                        fg=fg,
                        font=("Segoe UI", 8, "bold"),
                        width=5,
                        anchor=tk.E,
                    ).grid(row=0, column=1, sticky="e")
                    install_rows[idx] = {
                        "size": size_var,
                        "status": status_row_var,
                        "status_label": status_label,
                        "progress": progress_row_var,
                        "progress_text": progress_text_var,
                    }
            except Exception:
                pass

        def update_install_row(index: int, *, size: str | None = None, status: str | None = None, progress_value: float | None = None) -> None:
            try:
                row = install_rows.get(int(index))
                if not row:
                    return
                if size is not None:
                    row["size"].set(str(size))
                if status is not None:
                    row["status"].set(str(status))
                    status_label = row.get("status_label")
                    if status_label is not None:
                        status_lower = str(status or "").strip().lower()
                        color = muted
                        if status_lower in {"gotowe", "już było", "już zainstalowane"}:
                            color = success
                        elif status_lower == "błąd":
                            color = error
                        elif status_lower:
                            color = warning
                        try:
                            status_label.configure(fg=color)
                        except Exception:
                            pass
                if progress_value is not None:
                    value = max(0.0, min(100.0, float(progress_value)))
                    row["progress"].set(value)
                    progress_text = row.get("progress_text")
                    if progress_text is not None:
                        progress_text.set(_mobile_export_progress_label(value))
            except Exception:
                pass

        actions_row = tk.Frame(shell, bg=bg)
        actions_row.grid(row=6, column=0, sticky="ew")
        actions_row.grid_columnconfigure(0, weight=1)

        install_button_context = None
        pause_button_context = None
        context_install_control = {
            "running": False,
            "pause_requested": False,
            "paused": False,
        }

        def _sync_context_install_buttons() -> None:
            install_running = bool(context_install_control.get("running"))
            try:
                if install_button_context is not None:
                    install_button_context.configure(state=(tk.DISABLED if install_running else tk.NORMAL))
            except Exception:
                pass
            try:
                if pause_button_context is not None:
                    pause_button_context.configure(
                        state=(tk.NORMAL if install_running else tk.DISABLED),
                        text="Wznów" if context_install_control.get("pause_requested") else "Pauza",
                    )
            except Exception:
                pass

        def toggle_context_install_pause() -> None:
            if not context_install_control.get("running"):
                return
            context_install_control["pause_requested"] = not bool(context_install_control.get("pause_requested"))
            context_install_control["paused"] = bool(context_install_control.get("pause_requested"))
            if context_install_control.get("pause_requested"):
                local_status_var.set("Pauza jest uzbrojona. Instalator zatrzyma się przed następną zależnością.")
            else:
                local_status_var.set("Wznawiam instalację zależności eksportu.")
            _sync_context_install_buttons()

        def install_from_context() -> None:
            context_install_control["pause_requested"] = False
            context_install_control["paused"] = False
            _sync_context_install_buttons()
            install_requirements = _missing_install_requirements_for_request(preflight_state.get("request"))
            if not install_requirements:
                local_status_var.set("Nie ma brakujących zależności bibliotek dla aktualnego wyboru.")
                try:
                    local_progress.configure(mode="determinate")
                    local_progress_var.set(100.0)
                except Exception:
                    pass
                return
            if install_button_context is not None:
                try:
                    install_button_context.configure(state=tk.DISABLED)
                except Exception:
                    pass
            local_status_var.set("Instaluję wymagania eksportu w bieżącym środowisku Pythona...")

            def _after_install(success: bool) -> None:
                context_install_control["running"] = False
                context_install_control["pause_requested"] = False
                context_install_control["paused"] = False
                _sync_context_install_buttons()
                if install_button_context is not None:
                    try:
                        install_button_context.configure(state=tk.NORMAL)
                    except Exception:
                        pass
                if after_install is not None:
                    try:
                        after_install(success)
                    except Exception:
                        pass

            install_export_requirements(
                parent=modal,
                status_var_override=local_status_var,
                progress_widget=local_progress,
                progress_var_override=local_progress_var,
                after_install=_after_install,
                install_control=context_install_control,
                requirements_override=install_requirements,
                install_ui={
                    "build": build_install_rows,
                    "update": update_install_row,
                },
            )
            _sync_context_install_buttons()

        if _preflight_has_installable_dependency_problem(normalized):
            install_button_context = ttk.Button(
                actions_row,
                text="Zainstaluj wymagania eksportu",
                command=install_from_context,
            )
            install_button_context.grid(row=0, column=1, sticky="e", padx=(0, 8), ipadx=10, ipady=3)
        pause_button_context = ttk.Button(actions_row, text="Pauza", command=toggle_context_install_pause, state=tk.DISABLED)
        pause_button_context.grid(row=0, column=2, sticky="e", padx=(0, 8), ipadx=8, ipady=3)
        ttk.Button(actions_row, text="Zamknij", command=modal.destroy).grid(row=0, column=3, sticky="e", ipadx=8, ipady=3)

    requirement_rows_defer_state = {"after": None}

    def _set_requirement_rows(rows: list[tuple[str, str, str]]) -> None:
        done = _mobile_export_perf_timer("requirements_set_rows")
        try:
            if _mobile_export_window_is_moving():
                pending = requirement_rows_defer_state.get("after")
                if pending:
                    try:
                        dialog.after_cancel(pending)
                    except Exception:
                        pass
                deferred_rows = list(rows or [])
                requirement_rows_defer_state["after"] = dialog.after(
                    260,
                    lambda current_rows=deferred_rows: _set_requirement_rows(current_rows),
                )
                return
            requirement_rows_defer_state["after"] = None
            requirements_table["set_rows"](
                [
                    {
                        "values": {
                            "status": str(status or "-"),
                            "requirement": str(requirement or "-"),
                            "description": str(description or "-"),
                        },
                    }
                    for status, requirement, description in rows
                ]
            )
        except Exception:
            pass
        finally:
            done(rows=len(rows))

    def _formats_description(request: MobileExportRequest | None) -> str:
        if request is None:
            return "-"
        formats = set(str(item or "").strip().lower() for item in (request.formats or ()))
        quantizations = set(str(item or "").strip().lower() for item in (request.quantizations or ()))
        format_quantizations = getattr(request, "format_quantizations", {}) if request is not None else {}
        if not isinstance(format_quantizations, dict) or not format_quantizations:
            format_quantizations = {item: tuple(quantizations or {"fp32"}) for item in formats}
        items: list[str] = []
        if "litert" in formats:
            litert_quantizations = set(str(item or "").strip().lower() for item in (format_quantizations.get("litert") or ()))
            precision = []
            if "fp32" in litert_quantizations:
                precision.append("FP32")
            if "int8" in litert_quantizations:
                precision.append("INT8")
            items.append("LiteRT/TFLite " + ("/".join(precision) if precision else "FP32"))
        if "onnx" in formats:
            onnx_quantizations = set(str(item or "").strip().lower() for item in (format_quantizations.get("onnx") or ()))
            precision = []
            if "fp32" in onnx_quantizations:
                precision.append("FP32")
            if "int8" in onnx_quantizations:
                precision.append("INT8")
            items.append("ONNX " + ("/".join(precision) if precision else "FP32"))
        if "ncnn" in formats:
            items.append("NCNN FP32")
        return ", ".join(items) if items else "Nie wybrano formatu eksportu"

    def _refresh_requirements_view(
        request: MobileExportRequest | None,
        problems: list[str] | tuple[str, ...] | None,
    ) -> None:
        done = _mobile_export_perf_timer("requirements_refresh")
        normalized_problems = [str(item or "").strip() for item in (problems or ()) if str(item or "").strip()]
        problem_blob = " ".join(normalized_problems).lower()
        installable_problem = _preflight_has_installable_dependency_problem(normalized_problems)
        preflight_state.update(
            {
                "request": request,
                "problems": list(normalized_problems),
                "ready": bool(request is not None and not normalized_problems),
                "installable": bool(installable_problem),
            }
        )

        def has_problem(*needles: str) -> bool:
            return any(str(needle or "").lower() in problem_blob for needle in needles)

        if request is None:
            _set_requirement_rows(
                [
                    ("Brak", "Model", "Najpierw zaznacz model na liście kandydatów."),
                    ("-", "Formaty", "Po wyborze modelu pokażę pełną checklistę eksportu."),
                ]
            )
            if not dependency_install_control_state.get("running"):
                _set_dependency_tool(
                    "Wybierz model. Sprawdzenie gotowości i instalacja zależności będą dostępne po wyborze kandydata.",
                    "warning",
                    eta="-",
                    progress_value=0.0,
                )
                _update_dependency_buttons()
            done(problems=len(normalized_problems), rows=2)
            return

        formats = set(str(item or "").strip().lower() for item in (request.formats or ()))
        quantizations = set(str(item or "").strip().lower() for item in (request.quantizations or ()))
        checkpoint_problem = has_problem("checkpoint")
        destination_problem = has_problem("docelowy", "nadrzędnego", ".alprmodel")
        format_problem = has_problem("co najmniej jeden format", "nieobsługiwany format")
        core_problem = has_problem("ultralytics", "torch", "torchvision", "runtime yolo", "niespójne zależności yolo")
        onnx_problem = has_problem("onnx")
        litert_problem = has_problem("litert", "tflite")
        calibration_problem = has_problem("kalibracyjnego", "kalibracji")

        rows: list[tuple[str, str, str]] = [
            (
                "Brak" if checkpoint_problem else "Jest",
                "Checkpoint .pt",
                Path(request.checkpoint).name if request.checkpoint else "Plik wag modelu.",
            ),
            (
                "Brak" if format_problem else "Jest",
                "Formaty eksportu",
                _formats_description(request),
            ),
            (
                "Brak" if core_problem else "Jest",
                "Biblioteki bazowe",
                "ultralytics, torch i torchvision wymagane do odczytu oraz eksportu YOLO.",
            ),
            (
                "Brak" if destination_problem else "Jest",
                "Miejsce zapisu",
                str(request.destination),
            ),
        ]
        rows.append(
            (
                "Brak" if onnx_problem else "Jest",
                "ONNX",
                "onnx, onnxruntime, onnxslim." if "onnx" in formats else "Nie wybrano tego wariantu.",
            )
            if "onnx" in formats
            else ("-", "ONNX", "Nie wybrano tego wariantu.")
        )
        rows.append(
            (
                "Brak" if litert_problem else "Jest",
                "LiteRT/TFLite",
                (
                    "tensorflow, tf_keras, onnx2tf, onnx_graphsurgeon, ai-edge-litert, onnxslim, protobuf."
                    if "litert" in formats
                    else "Nie wybrano tego wariantu."
                ),
            )
            if "litert" in formats
            else ("-", "LiteRT/TFLite", "Nie wybrano tego wariantu.")
        )
        rows.append(
            (
                "Brak" if calibration_problem or not request.calibration_data else "Jest",
                "Kalibracja INT8",
                str(request.calibration_data or "Wymagana tylko dla kwantyzacji INT8."),
            )
            if "int8" in quantizations
            else ("-", "Kalibracja INT8", "Kwantyzacja INT8 nie jest wybrana.")
        )

        known_checks = (
            checkpoint_problem
            or destination_problem
            or format_problem
            or core_problem
            or onnx_problem
            or litert_problem
            or calibration_problem
        )
        if normalized_problems and not known_checks:
            rows.append(("Brak", "Inne wymaganie", "; ".join(normalized_problems[:2])))
        _set_requirement_rows(rows)
        if not dependency_install_control_state.get("running"):
            if normalized_problems:
                if installable_problem:
                    _set_dependency_tool(
                        f"Sprawdzenie gotowości wykryło {len(normalized_problems)} problemów. Brakujące biblioteki możesz zainstalować tutaj przed eksportem.",
                        "warning",
                        eta="po starcie instalacji",
                        progress_value=0.0,
                    )
                else:
                    _set_dependency_tool(
                        "Sprawdzenie gotowości wymaga korekty ustawień lub danych. Instalacja bibliotek nie rozwiąże tego problemu.",
                        "warning",
                        eta="-",
                        progress_value=0.0,
                    )
            else:
                _set_dependency_tool(
                    "Eksport gotowy. Zależności i ustawienia są poprawne.",
                    "success",
                    eta="-",
                    progress_value=100.0,
                )
            _update_dependency_buttons()
        done(problems=len(normalized_problems), rows=len(rows))

    preflight_refresh_state = {"after": None, "suspend": False, "serial": 0}

    def _schedule_preflight_refresh(*, update_status: bool = False, delay: int = 180) -> None:
        started = time.perf_counter()
        try:
            if _mobile_export_window_is_moving():
                delay = max(delay, 300)
            preflight_refresh_state["serial"] = int(preflight_refresh_state.get("serial") or 0) + 1
            serial = int(preflight_refresh_state.get("serial") or 0)
            pending = preflight_refresh_state.get("after")
            if pending:
                try:
                    dialog.after_cancel(pending)
                except Exception:
                    pass
            preflight_refresh_state["after"] = dialog.after(
                delay,
                lambda current_serial=serial: _run_preflight_async(
                    current_serial,
                    update_status=update_status,
                ),
            )
            _mobile_export_perf_record(
                "preflight_queue",
                (time.perf_counter() - started) * 1000.0,
                serial=serial,
                delay=delay,
                update=int(bool(update_status)),
            )
        except Exception:
            pass

    def apply_candidate(candidate: dict, *, preflight_after: bool = False) -> None:
        started = time.perf_counter()
        if not candidate:
            _mobile_export_perf_record("apply_candidate", 0.0, candidate=0)
            return
        selected_state["candidate"] = candidate
        source_locations.set_candidate(candidate)
        run = candidate.get("run")
        target = str(candidate.get("target") or "").strip().lower()
        role = str(candidate.get("role") or _mobile_role_from_training_target(target)).strip()
        task = str(candidate.get("task") or ("pose" if role == "plate" else "detect")).strip()
        best_weights = Path(candidate.get("best_weights"))
        img_size = int(candidate.get("img_size") or getattr(run, "img_size", CONFIG.DEFAULT_IMG_SIZE) or CONFIG.DEFAULT_IMG_SIZE)
        default_destination = self._build_mobile_model_export_path(run, target, best_weights)
        default_calibration_path = _mobile_export_resolve_data_yaml(
            candidate.get("dataset_path") or getattr(run, "dataset_path", "")
        )
        default_calibration = str(default_calibration_path) if default_calibration_path else ""
        metric_text = _candidate_metric_label(candidate)

        preflight_refresh_state["suspend"] = True
        try:
            selected_model_var.set(str(candidate.get("model_label") or "-"))
            target_marker = _mobile_export_target_marker(candidate)
            yolo_marker = _mobile_export_compact_yolo_label(candidate.get("model_version"))
            selected_target_var.set(
                f"{target_marker} | {yolo_marker}" if yolo_marker and yolo_marker != "-" else target_marker
            )
            selected_metric_var.set(metric_text)
            imgsz_var.set(img_size)
            destination_var.set(str(default_destination))
            calibration_var.set(default_calibration)
            _refresh_export_selection_summary_vars()
        finally:
            preflight_refresh_state["suspend"] = False

        detail_rows: list[dict] = []
        for label, value in _mobile_export_candidate_detail_rows(candidate):
            label_lower = label.lower()
            tone = ""
            if label in {"mAP50", "mAP50-95", "Najlepsza epoka"}:
                tone = "metric"
            elif "ścieżka" in label_lower or "checkpoint" in label_lower or "katalog" in label_lower:
                tone = "path"
            detail_rows.append(
                {
                    "values": {
                        "label": str(label or "-"),
                        "value": str(value or "-"),
                    },
                    "tone": tone,
                }
            )
        try:
            details_table["set_rows"](detail_rows)
        except Exception:
            pass

        try:
            selected_iid = str(candidate.get("iid") or "")
            _draw_mobile_export_overview_chart(overview_canvas, candidates, selected_iid, palette, _selected_export_iids())
            _draw_mobile_export_metric_chart(metric_chart_canvas, candidate, palette, max_candidate_size_mb)
        except Exception:
            pass

        if run is None:
            set_status(f"Wybrano {candidate.get('model_label') or 'model'} z katalogu gotowych modeli.", "info")
        else:
            set_status(f"Wybrano {candidate.get('model_label') or 'model'} z runu {candidate.get('run_label') or '-'}.", "info")
        if preflight_after:
            _schedule_preflight_refresh(update_status=True, delay=220)
        _mobile_export_perf_record(
            "apply_candidate",
            (time.perf_counter() - started) * 1000.0,
            candidate=str(candidate.get("model_label") or "-")[:64],
            preflight=int(bool(preflight_after)),
        )

    def _selected_export_format_quantizations(option_vars: dict | None = None) -> dict[str, tuple[str, ...]]:
        source = option_vars if isinstance(option_vars, dict) else {}
        litert_source = source.get("litert") or litert_var
        onnx_source = source.get("onnx") or onnx_var
        int8_source = source.get("int8") or int8_var
        onnx_int8_source = source.get("onnx_int8") or onnx_int8_var
        ncnn_source = source.get("ncnn") or ncnn_var

        def _bool_value(variable) -> bool:
            try:
                return bool(variable.get())
            except Exception:
                return bool(variable)

        result: dict[str, tuple[str, ...]] = {}
        if _bool_value(litert_source) or _bool_value(int8_source):
            values = []
            if _bool_value(litert_source):
                values.append("fp32")
            if _bool_value(int8_source):
                values.append("int8")
            result["litert"] = tuple(dict.fromkeys(values or ["fp32"]))
        if _bool_value(onnx_source) or _bool_value(onnx_int8_source):
            values = []
            if _bool_value(onnx_source):
                values.append("fp32")
            if _bool_value(onnx_int8_source):
                values.append("int8")
            result["onnx"] = tuple(dict.fromkeys(values or ["fp32"]))
        if _bool_value(ncnn_source):
            result["ncnn"] = ("fp32",)
        return result

    def _selected_export_formats_and_quantizations(option_vars: dict | None = None) -> tuple[list[str], list[str]]:
        format_quantizations = _selected_export_format_quantizations(option_vars)
        formats = list(format_quantizations.keys())
        quantizations: list[str] = []
        for values in format_quantizations.values():
            quantizations.extend(values)
        return formats, quantizations

    def _build_mobile_export_request_for_candidate(
        candidate: dict,
        *,
        destination: Path | None = None,
        include_metadata: bool = False,
        option_vars: dict | None = None,
    ) -> MobileExportRequest:
        if not candidate:
            raise MobileExportError("Najpierw wybierz model do eksportu.")
        run = candidate.get("run")
        target = str(candidate.get("target") or "").strip().lower()
        role = str(candidate.get("role") or _mobile_role_from_training_target(target)).strip()
        best_weights = Path(candidate.get("best_weights"))
        img_size = int(candidate.get("img_size") or getattr(run, "img_size", CONFIG.DEFAULT_IMG_SIZE) or CONFIG.DEFAULT_IMG_SIZE)
        options = option_vars if isinstance(option_vars, dict) else {}
        formats, quantizations = _selected_export_formats_and_quantizations(options)
        format_quantizations = _selected_export_format_quantizations(options)

        def _option_value(name: str, fallback):
            variable = options.get(name)
            if variable is None:
                return fallback
            try:
                return variable.get()
            except Exception:
                return variable

        target_destination = destination
        if target_destination is None:
            raw_destination = str(destination_var.get()).strip()
            target_destination = Path(raw_destination) if raw_destination else self._build_mobile_model_export_path(run, target, best_weights)
        if target_destination.suffix.lower() != ".alprmodel":
            target_destination = target_destination.with_suffix(".alprmodel")
        calibration_text = str(_option_value("calibration", calibration_var.get()) or "").strip()
        calibration = Path(calibration_text) if calibration_text else None
        model_id_seed = f"{role}-{getattr(run, 'id', '') or best_weights.stem}"
        export_metadata = {}
        if include_metadata:
            target_destination.parent.mkdir(parents=True, exist_ok=True)
            export_metadata = self._build_mobile_export_metadata(run, target, best_weights)
            model_payload = dict(export_metadata.get("model") or {}) if isinstance(export_metadata, dict) else {}
            model_payload["display_id"] = str(candidate.get("model_label") or best_weights.stem)
            model_payload["architecture_label"] = str(candidate.get("model_version") or "")
            params_m, params_source = _mobile_export_candidate_params_millions(candidate)
            if params_m is not None:
                model_payload["parameters_millions"] = round(float(params_m), 3)
                model_payload["parameters_source"] = params_source or "unknown"
                if params_source == "metadata":
                    model_payload["parameter_count"] = int(round(float(params_m) * 1_000_000))
            export_metadata["model"] = model_payload
            project_name = str(candidate.get("project_name") or "").strip()
            project_root = str(candidate.get("project_root") or "").strip()
            export_metadata["project"] = {"name": project_name, "root": project_root}
            source_payload = dict(export_metadata.get("source") or {})
            source_payload["project_name"] = project_name
            source_payload["project_root"] = project_root
            export_metadata["source"] = source_payload
            training_payload = dict(export_metadata.get("training") or {}) if isinstance(export_metadata, dict) else {}
            for key in ("dataset_path", "base_model", "img_size", "epochs", "current_epoch", "batch_size", "started_at", "finished_at"):
                value = candidate.get(key)
                if value not in (None, ""):
                    training_payload.setdefault(key, value)
            candidate_provenance = _mobile_export_candidate_training_provenance(candidate)
            for key in (
                "lineage_total_epochs",
                "lineage_total_epochs_known",
                "lineage_stage_count",
                "lineage_stage_count_known",
                "known_stage_count_minimum",
                "run_train_images",
                "run_nominal_sample_presentations",
                "lineage_nominal_sample_presentations",
                "sample_presentations_known",
                "known_sample_presentations_minimum",
                "best_epoch_source",
                "provenance_capture",
            ):
                if isinstance(candidate_provenance, dict) and candidate_provenance.get(key) not in (None, ""):
                    training_payload.setdefault(key, candidate_provenance.get(key))
            known_total = _mobile_export_provenance_known_total(training_payload) or _mobile_export_provenance_known_total(candidate_provenance)
            if known_total is not None:
                training_payload["total_epochs"] = known_total
                training_payload["total_epochs_known"] = True
            else:
                minimum = _mobile_export_candidate_total_epochs(candidate)
                if minimum > 0:
                    training_payload.setdefault("known_epochs_minimum", minimum)
                training_payload["total_epochs"] = None
                training_payload["total_epochs_known"] = False
                training_payload.setdefault("provenance_status", candidate.get("provenance_status") or "partial")
            export_metadata["training"] = training_payload
            candidate_snapshot_source = dict(candidate)
            candidate_snapshot_source["training_provenance"] = training_payload
            known_snapshot_total = _mobile_export_provenance_known_total(training_payload)
            if known_snapshot_total is not None:
                candidate_snapshot_source["total_epochs"] = known_snapshot_total
                candidate_snapshot_source["total_epochs_known"] = True
            else:
                candidate_snapshot_source["known_epochs_minimum"] = training_payload.get("known_epochs_minimum")
                candidate_snapshot_source["total_epochs_known"] = False
            candidate_snapshot_source["provenance_status"] = training_payload.get("provenance_status") or candidate.get("provenance_status") or ""
            export_metadata["candidate"] = _mobile_export_candidate_manifest_snapshot(candidate_snapshot_source)
        if role == "vehicle":
            vehicle_payload = _mobile_export_vehicle_class_payload(
                candidate,
                _option_value("vehicle_classes", ""),
            )
            export_metadata["vehicle_detection"] = vehicle_payload
            runtime_payload = dict(export_metadata.get("runtime") or {}) if isinstance(export_metadata, dict) else {}
            runtime_payload["vehicle_filter_mode"] = "include"
            runtime_payload["vehicle_filter_labels"] = list(vehicle_payload.get("include_labels") or [])
            runtime_payload["vehicle_filter_indices"] = list(vehicle_payload.get("include_class_indices") or [])
            export_metadata["runtime"] = runtime_payload
        if include_metadata:
            export_metadata["export_settings"] = self._json_safe_training_value(
                {
                    "schema": "alpr.export.settings_snapshot.v1",
                    "formats": list(formats),
                    "quantizations": list(dict.fromkeys(quantizations)),
                    "format_quantizations": {key: list(value) for key, value in format_quantizations.items()},
                    "image_size": int(_option_value("imgsz", imgsz_var.get()) or img_size),
                    "confidence_threshold": max(
                        0.0,
                        min(1.0, float(_option_value("conf", conf_var.get()) or CONFIG.DEFAULT_CONFIDENCE)),
                    ),
                    "iou_threshold": max(
                        0.0,
                        min(1.0, float(_option_value("iou", iou_var.get()) or CONFIG.DEFAULT_IOU)),
                    ),
                    "calibration_data": str(calibration or ""),
                    "calibration_choice": str(_option_value("calibration_choice", "") or ""),
                    "vehicle_classes": str(_option_value("vehicle_classes", "") or "") if role == "vehicle" else "",
                }
            )
        return MobileExportRequest(
            checkpoint=best_weights,
            destination=target_destination,
            role=role,
            formats=tuple(formats),
            image_size=int(_option_value("imgsz", imgsz_var.get()) or img_size),
            quantizations=tuple(dict.fromkeys(quantizations)),
            format_quantizations=format_quantizations,
            calibration_data=calibration,
            confidence_threshold=max(0.0, min(1.0, float(_option_value("conf", conf_var.get()) or CONFIG.DEFAULT_CONFIDENCE))),
            iou_threshold=max(0.0, min(1.0, float(_option_value("iou", iou_var.get()) or CONFIG.DEFAULT_IOU))),
            model_id=self._safe_model_export_slug(model_id_seed, fallback=f"{role}-model")[:80],
            name=f"{candidate.get('target_label') or self._format_training_target_label(target)} | {candidate.get('model_label') or best_weights.stem}",
            version="1",
            metadata=export_metadata,
        )

    def build_preflight_request() -> MobileExportRequest:
        return _build_mobile_export_request_for_candidate(
            selected_state.get("candidate"),
            include_metadata=False,
        )

    def build_request() -> MobileExportRequest:
        return _build_mobile_export_request_for_candidate(
            selected_state.get("candidate"),
            include_metadata=True,
        )

    def run_preflight(show_dialog: bool = True, update_status: bool = True) -> bool:
        request = None
        try:
            request = build_preflight_request()
            problems = exporter.preflight(request)
        except Exception as exc:
            problems = [str(exc)]
        _refresh_requirements_view(request, problems)
        if problems:
            text = "Eksport nie może jeszcze ruszyć.\n" + "\n".join(f"- {item}" for item in problems)
            if update_status:
                set_status(text, "warning")
            if show_dialog:
                show_preflight_problems_dialog(
                    problems,
                    parent=dialog,
                    title="Sprawdzenie gotowości modelu",
                    context=(
                        "Najpierw sprawdzamy konkretny model, wybrane formaty i wymagane biblioteki. "
                        "Dopiero jeśli brakuje zależności eksportu, instalacja jest dostępna w tym oknie."
                    ),
                )
            return False
        if update_status:
            set_status("Model gotowy do eksportu mobilnego.", "success")
        return True

    def _ask_mobile_package_destination(
        default_destination: Path,
        *,
        parent,
        title: str = "Zapisz eksport mobilny (.alprmodel)",
    ) -> Path | None:
        current = Path(default_destination or "model.alprmodel")
        initial_dir = current.parent
        if str(initial_dir) == ".":
            initial_dir = Path.cwd()
        selected = filedialog.asksaveasfilename(
            title=title,
            parent=parent,
            initialdir=str(initial_dir),
            initialfile=current.name,
            defaultextension=".alprmodel",
            filetypes=(("Model mobilny lub pakiet ALPR", "*.alprmodel"), ("ZIP", "*.zip"), ("Wszystkie pliki", "*.*")),
        )
        if not selected:
            return None
        path = Path(selected)
        if path.suffix.lower() != ".alprmodel":
            path = path.with_suffix(".alprmodel")
        return path

    worker_state = {"running": False}

    def _run_preflight_async(serial: int, *, update_status: bool = False) -> None:
        scheduled_at = time.perf_counter()
        preflight_refresh_state["after"] = None
        if _mobile_export_window_is_moving():
            try:
                preflight_refresh_state["after"] = dialog.after(
                    220,
                    lambda current_serial=serial: _run_preflight_async(
                        current_serial,
                        update_status=update_status,
                    ),
                )
            except Exception:
                pass
            return
        if worker_state.get("running") or not selected_state.get("candidate"):
            _mobile_export_perf_record(
                "preflight_async_skip",
                0.0,
                running=int(bool(worker_state.get("running"))),
                candidate=int(bool(selected_state.get("candidate"))),
            )
            return
        request = None
        try:
            request = build_preflight_request()
        except Exception as exc:
            _refresh_requirements_view(None, [str(exc)])
            if update_status:
                set_status(f"Eksport nie może jeszcze ruszyć: {exc}", "warning")
            _mobile_export_perf_record("preflight_async_build_error", 0.0, error=str(exc)[:96])
            return

        def worker() -> None:
            worker_started = time.perf_counter()
            try:
                problems = exporter.preflight(request)
            except Exception as exc:
                problems = [str(exc)]
            worker_ms = (time.perf_counter() - worker_started) * 1000.0

            def done() -> None:
                if _mobile_export_window_is_moving():
                    try:
                        dialog.after(260, done)
                    except Exception:
                        pass
                    return
                if int(preflight_refresh_state.get("serial") or 0) != int(serial):
                    _mobile_export_perf_record(
                        "preflight_async_stale",
                        (time.perf_counter() - scheduled_at) * 1000.0,
                        serial=serial,
                    )
                    return
                _refresh_requirements_view(request, problems)
                if update_status:
                    if problems:
                        set_status(
                            "Eksport nie może jeszcze ruszyć.\n"
                            + "\n".join(f"- {item}" for item in problems),
                            "warning",
                        )
                    else:
                        set_status("Sprawdzenie wymagań modelu OK. Można rozpocząć eksport mobilny.", "success")
                _mobile_export_perf_record(
                    "preflight_async",
                    (time.perf_counter() - scheduled_at) * 1000.0,
                    worker=f"{worker_ms:.1f}",
                    problems=len(problems),
                    update=int(bool(update_status)),
                )

            try:
                dialog.after(0, done)
            except Exception:
                pass

        threading.Thread(target=worker, name="mobile-export-preflight", daemon=True).start()

    def _refresh_preflight_after_option_change(*_args) -> None:
        if (
            worker_state.get("running")
            or not selected_state.get("candidate")
            or preflight_refresh_state.get("suspend")
        ):
            return
        _schedule_preflight_refresh(update_status=False, delay=220)

    for export_option_var in (
        litert_var,
        onnx_var,
        int8_var,
        ncnn_var,
        imgsz_var,
        conf_var,
        iou_var,
        calibration_var,
    ):
        try:
            export_option_var.trace_add("write", _refresh_preflight_after_option_change)
        except Exception:
            pass

    def set_running(running: bool) -> None:
        worker_state["running"] = running
        for widget in (complete_package_button, preflight_button, dependency_install_button, close_button, candidate_tree):
            try:
                widget.configure(state=(tk.DISABLED if running else tk.NORMAL))
            except Exception:
                pass
        try:
            if export_button is not None:
                is_valid, _label, _message = _export_selection_validation()
                export_button.configure(
                    state=(tk.NORMAL if is_valid and not running else tk.DISABLED)
                )
        except Exception:
            pass
        _update_dependency_buttons()

    def install_export_requirements(
        *,
        parent=None,
        status_var_override: tk.StringVar | None = None,
        progress_widget=None,
        progress_var_override: tk.DoubleVar | None = None,
        after_install=None,
        install_control: dict | None = None,
        install_ui: dict | None = None,
        requirements_override: list[dict] | None = None,
    ) -> None:
        if worker_state.get("running"):
            return
        parent_widget = parent or dialog
        target_progress = progress_widget or progress
        target_progress_var = progress_var_override or progress_var
        control = install_control if isinstance(install_control, dict) else dependency_install_control_state

        def message_parent():
            try:
                if parent_widget is not None and parent_widget.winfo_exists():
                    return parent_widget
            except Exception:
                pass
            return dialog

        def update_install_status(text: str, tone: str = "info") -> None:
            set_status(text, tone)
            if status_var_override is not None:
                try:
                    status_var_override.set(text)
                except Exception:
                    pass

        if requirements_override is not None:
            requirements_path = Path(getattr(CONFIG, "WORKSPACE_DIR", Path.cwd())).resolve().parent / "requirements-mobile-export.txt"
            requirements = [
                {
                    "spec": str(item.get("spec") or item.get("name") or "").strip(),
                    "name": str(item.get("name") or _mobile_export_requirement_name(str(item.get("spec") or ""))).strip(),
                    "models": item.get("models") or item.get("model") or "",
                    "description": item.get("description") or "",
                }
                for item in requirements_override
                if str(item.get("spec") or item.get("name") or "").strip()
            ]
            if not requirements:
                return messagebox.showinfo(
                    "Zależności eksportu",
                    "Nie ma brakujących zależności do instalacji dla aktualnego wyboru.",
                    parent=message_parent(),
                )
        else:
            repo_root = Path(getattr(CONFIG, "WORKSPACE_DIR", Path.cwd())).resolve().parent
            requirement_candidates = [
                repo_root / "requirements-mobile-export.txt",
                Path.cwd() / "requirements-mobile-export.txt",
            ]
            requirements_path = next((path for path in requirement_candidates if path.exists() and path.is_file()), None)
            if requirements_path is None:
                return messagebox.showerror(
                    "Brak pliku wymagań",
                    "Nie znalazłem pliku requirements-mobile-export.txt w katalogu projektu.",
                    parent=message_parent(),
                )
            requirements = _mobile_export_parse_requirements_file(requirements_path)
        if not requirements:
            return messagebox.showerror(
                "Brak wymagań",
                f"Plik {requirements_path} nie zawiera zależności do instalacji.",
                parent=message_parent(),
            )

        try:
            if install_ui and callable(install_ui.get("build")):
                install_ui["build"](requirements)
        except Exception:
            pass

        try:
            target_progress.configure(mode="determinate")
            target_progress.stop()
            target_progress_var.set(0.0)
        except Exception:
            pass
        control["running"] = True
        control["pause_requested"] = False
        control["paused"] = False
        set_running(True)
        _update_dependency_buttons()
        update_install_status(
            f"Instaluję zależności eksportu sekwencyjnie: 0/{len(requirements)}. ETA: liczę po pierwszym kroku.",
            "info",
        )

        def worker() -> None:
            started_all = time.perf_counter()
            count = max(1, len(requirements))
            tail_lines: list[str] = []
            ok = True

            def post(callable_):
                try:
                    dialog.after(0, callable_)
                except Exception:
                    pass

            def update_row(index: int, *, size: str | None = None, status: str | None = None, progress_value: float | None = None) -> None:
                if install_ui and callable(install_ui.get("update")):
                    post(lambda: install_ui["update"](index, size=size, status=status, progress_value=progress_value))

            def update_total(index: int, local_progress: float, message: str) -> None:
                local = max(0.0, min(100.0, float(local_progress)))
                global_percent = max(0.0, min(99.0, ((float(index) + local / 100.0) / float(count)) * 100.0))
                elapsed = time.perf_counter() - started_all
                eta_text = "liczę"
                fraction = global_percent / 100.0
                if fraction > 0.02:
                    eta_text = _format_mobile_export_duration((elapsed / fraction) - elapsed)

                if control.get("pause_requested"):
                    eta_text = "pauza"

                def apply_total() -> None:
                    try:
                        target_progress_var.set(global_percent)
                    except Exception:
                        pass
                    update_install_status(
                        f"{message} | postęp całkowity {global_percent:.0f}% | ETA: {eta_text}",
                        "info",
                    )

                    if control is dependency_install_control_state:
                        _set_dependency_tool(message, "info", eta=eta_text, progress_value=global_percent)
                        _update_dependency_buttons()

                post(apply_total)

            row_publish_state: dict[int, dict] = {}
            total_publish_state: dict[str, float | str] = {"time": 0.0, "progress": -1.0, "message": ""}

            def publish_row(
                index: int,
                *,
                size: str | None = None,
                status: str | None = None,
                progress_value: float | None = None,
                force: bool = False,
            ) -> None:
                now = time.perf_counter()
                state = row_publish_state.setdefault(
                    int(index),
                    {"time": 0.0, "size": "", "status": "", "progress": -999.0},
                )
                next_progress = float(progress_value) if progress_value is not None else float(state.get("progress") or 0.0)
                progress_delta = abs(next_progress - float(state.get("progress") or 0.0))
                changed = (
                    (size is not None and str(size) != str(state.get("size") or ""))
                    or (status is not None and str(status) != str(state.get("status") or ""))
                    or progress_delta >= 1.0
                )
                if not force and not changed and (now - float(state.get("time") or 0.0)) < 0.15:
                    return
                state["time"] = now
                if size is not None:
                    state["size"] = str(size)
                if status is not None:
                    state["status"] = str(status)
                if progress_value is not None:
                    state["progress"] = next_progress
                update_row(index, size=size, status=status, progress_value=progress_value)

            def publish_total(index: int, local_progress: float, message: str, *, force: bool = False) -> None:
                now = time.perf_counter()
                progress_value = max(0.0, min(100.0, float(local_progress)))
                last_progress = float(total_publish_state.get("progress") or -1.0)
                last_message = str(total_publish_state.get("message") or "")
                if (
                    not force
                    and abs(progress_value - last_progress) < 1.0
                    and str(message) == last_message
                    and (now - float(total_publish_state.get("time") or 0.0)) < 0.2
                ):
                    return
                total_publish_state["time"] = now
                total_publish_state["progress"] = progress_value
                total_publish_state["message"] = str(message)
                update_total(index, local_progress, message)

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            for index, requirement in enumerate(requirements):
                spec = str(requirement.get("spec") or "").strip()
                name = str(requirement.get("name") or spec).strip()
                if not spec:
                    continue
                while control.get("pause_requested"):
                    control["paused"] = True
                    publish_row(index, status="Pauza", progress_value=0.0, force=True)
                    publish_total(index, 0.0, f"Pauza przed {index + 1}/{count}: {spec}", force=True)
                    time.sleep(0.25)
                control["paused"] = False
                post(_update_dependency_buttons)
                local_size = _mobile_export_installed_distribution_size(name)
                size_label = _mobile_export_install_detail(spec=spec, installed_bytes=local_size)
                publish_row(index, size=size_label, status="Start", progress_value=3.0, force=True)
                publish_total(index, 3.0, f"{index + 1}/{count}: przygotowuję {spec}", force=True)
                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--progress-bar",
                    "raw",
                    "--no-color",
                    "--extra-index-url",
                    "https://pypi.ngc.nvidia.com",
                    spec,
                ]
                process = None
                row_progress = 3.0
                downloaded_size = None
                current_artifact = name or spec
                current_bytes = None
                total_bytes = None
                speed_bytes = None
                eta = ""
                source = ""
                last_transfer_bytes = None
                last_transfer_time = None
                already_satisfied = False
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(requirements_path.parent),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        creationflags=creationflags,
                    )
                    while True:
                        line = process.stdout.readline() if process.stdout is not None else ""
                        if line == "" and process.poll() is not None:
                            break
                        if not line:
                            time.sleep(0.03)
                            continue
                        clean_line = str(line).strip()
                        if clean_line:
                            tail_lines.append(clean_line)
                            tail_lines[:] = tail_lines[-20:]
                        lower = clean_line.lower()
                        pip_info = _mobile_export_parse_pip_progress(clean_line)
                        if pip_info.get("artifact"):
                            current_artifact = str(pip_info.get("artifact") or current_artifact)
                            current_bytes = None
                            last_transfer_bytes = None
                            last_transfer_time = None
                        if pip_info.get("source"):
                            source = str(pip_info.get("source") or "")
                        if isinstance(pip_info.get("total_bytes"), int):
                            total_bytes = int(pip_info["total_bytes"])
                            downloaded_size = total_bytes
                        if isinstance(pip_info.get("current_bytes"), int):
                            now_transfer = time.perf_counter()
                            current_bytes = int(pip_info["current_bytes"])
                            if (
                                last_transfer_bytes is not None
                                and last_transfer_time is not None
                                and now_transfer > last_transfer_time
                                and current_bytes > last_transfer_bytes
                            ):
                                speed_bytes = int((current_bytes - last_transfer_bytes) / max(0.001, now_transfer - last_transfer_time))
                            last_transfer_bytes = current_bytes
                            last_transfer_time = now_transfer
                        if isinstance(pip_info.get("speed_bytes"), int):
                            speed_bytes = int(pip_info["speed_bytes"])
                        if pip_info.get("eta"):
                            eta = str(pip_info.get("eta") or "")
                        elif current_bytes is not None and total_bytes and speed_bytes:
                            eta = _format_mobile_export_duration((total_bytes - current_bytes) / max(1, speed_bytes))
                        if isinstance(pip_info.get("fraction"), (float, int)):
                            row_progress = max(row_progress, 25.0 + (float(pip_info["fraction"]) * 45.0))
                        if pip_info:
                            size_label = _mobile_export_install_detail(
                                spec=spec,
                                artifact=current_artifact,
                                current_bytes=current_bytes,
                                total_bytes=total_bytes,
                                speed_bytes=speed_bytes,
                                eta=eta,
                                source=source,
                                installed_bytes=local_size,
                            )
                        if "requirement already satisfied" in lower:
                            already_satisfied = True
                            row_progress = max(row_progress, 76.0)
                            publish_row(index, size=size_label, status="Już zainstalowane", progress_value=row_progress, force=True)
                        elif lower.startswith("collecting ") or "collecting " in lower:
                            row_progress = max(row_progress, 14.0)
                            publish_row(index, size=size_label, status="Sprawdzam", progress_value=row_progress)
                        elif "downloading " in lower:
                            publish_row(index, size=size_label, status="Pobieram", progress_value=row_progress)
                        elif "progress " in lower and total_bytes:
                            publish_row(index, size=size_label, status="Pobieram", progress_value=row_progress)
                        elif "using cached " in lower:
                            row_progress = max(row_progress, 48.0)
                            publish_row(index, size=size_label, status="Z cache", progress_value=row_progress, force=True)
                        elif "preparing metadata" in lower or "getting requirements to build wheel" in lower:
                            row_progress = max(row_progress, 58.0)
                            publish_row(index, size=size_label, status="Metadane", progress_value=row_progress)
                        elif "building wheel" in lower:
                            row_progress = max(row_progress, 72.0)
                            publish_row(index, size=size_label, status="Buduję wheel", progress_value=row_progress)
                        elif "installing collected packages" in lower:
                            row_progress = max(row_progress, 84.0)
                            publish_row(index, size=size_label, status="Instaluję", progress_value=row_progress, force=True)
                        elif "successfully installed" in lower:
                            row_progress = max(row_progress, 96.0)
                            publish_row(index, size=size_label, status="Kończę", progress_value=row_progress, force=True)
                        publish_total(index, row_progress, f"{index + 1}/{count}: {spec}")
                    return_code = process.wait()
                except Exception as exc:
                    ok = False
                    tail_lines.append(str(exc))
                    publish_row(index, status="Błąd", progress_value=100.0, force=True)
                    publish_total(index, 100.0, f"{index + 1}/{count}: błąd instalacji {spec}", force=True)
                    break
                if return_code != 0:
                    ok = False
                    publish_row(index, status="Błąd", progress_value=100.0, force=True)
                    publish_total(index, 100.0, f"{index + 1}/{count}: błąd instalacji {spec}", force=True)
                    break
                cache_key = re.sub(r"[-_.]+", "-", name.strip().lower())
                _MOBILE_EXPORT_DIST_SIZE_CACHE.pop(cache_key, None)
                installed_size = _mobile_export_installed_distribution_size(name)
                if installed_size:
                    if downloaded_size and downloaded_size != installed_size:
                        size_label = (
                            f"ostatni plik {_format_mobile_export_bytes(downloaded_size)} | "
                            f"lokalnie {_format_mobile_export_bytes(installed_size)}"
                        )
                    else:
                        size_label = f"lokalnie {_format_mobile_export_bytes(installed_size)}"
                elif downloaded_size:
                    size_label = f"ostatni plik {_format_mobile_export_bytes(downloaded_size)}"
                publish_row(
                    index,
                    size=size_label,
                    status="Już było" if already_satisfied else "Gotowe",
                    progress_value=100.0,
                    force=True,
                )
                publish_total(index + 1, 0.0, f"Zainstalowano {index + 1}/{count}: {spec}", force=True)

            tail = "\n".join(tail_lines[-12:])

            def done() -> None:
                control["running"] = False
                control["pause_requested"] = False
                control["paused"] = False
                set_running(False)
                _update_dependency_buttons()
                try:
                    target_progress.stop()
                    target_progress.configure(mode="determinate")
                    target_progress_var.set(100.0 if ok else target_progress_var.get())
                except Exception:
                    pass
                if ok:
                    update_install_status(
                        f"Wymagania eksportu są gotowe. Czas instalacji: {_format_mobile_export_duration(time.perf_counter() - started_all)}.",
                        "success",
                    )
                    run_preflight(show_dialog=False)
                    if after_install is not None:
                        try:
                            after_install(True)
                        except Exception:
                            pass
                    messagebox.showinfo(
                        "Wymagania zainstalowane",
                        "Biblioteki eksportu mobilnego są gotowe w tym środowisku Pythona.",
                        parent=message_parent(),
                    )
                    return
                update_install_status(f"Instalacja wymagań nie powiodła się.\n{tail}", "error")
                if after_install is not None:
                    try:
                        after_install(False)
                    except Exception:
                        pass
                messagebox.showerror(
                    "Nie udało się zainstalować wymagań",
                    tail or "Pip zakończył pracę błędem.",
                    parent=message_parent(),
                )

            try:
                dialog.after(0, done)
            except Exception:
                pass

        threading.Thread(target=worker, name="mobile-export-requirements-install", daemon=True).start()

    def install_from_dependency_tool() -> None:
        if worker_state.get("running"):
            return
        if run_preflight(show_dialog=False, update_status=True):
            _set_dependency_tool(
                "Eksport gotowy. Nie ma brakujących zależności do instalacji.",
                "success",
                eta="-",
                progress_value=100.0,
            )
            return
        problems = list(preflight_state.get("problems") or [])
        if not _preflight_has_installable_dependency_problem(problems):
            _set_dependency_tool(
                "Najpierw popraw ustawienia wskazane przez sprawdzenie gotowości. Instalacja bibliotek nie wystarczy.",
                "warning",
                eta="-",
                progress_value=0.0,
            )
            show_preflight_problems_dialog(
                problems,
                parent=dialog,
                title="Sprawdzenie gotowości modelu",
                context="Te problemy wymagają korekty ustawień lub danych, a nie instalacji bibliotek.",
            )
            return

        install_rows_state: dict[str, list[dict]] = {"rows": []}

        def render_install_rows() -> None:
            rows: list[tuple[str, str, str]] = []
            for row in install_rows_state.get("rows", []):
                progress_value = float(row.get("progress") or 0.0)
                rows.append(
                    (
                        str(row.get("status") or "Czeka"),
                        str(row.get("spec") or "-"),
                        f"{_mobile_export_progress_label(progress_value)} | {row.get('size') or '-'}",
                    )
                )
            _set_requirement_rows(rows)

        def build_install_rows(items: list[dict]) -> None:
            install_rows_state["rows"] = [
                {
                    "spec": str(item.get("spec") or item.get("name") or "-"),
                    "size": "-",
                    "status": "Czeka",
                    "progress": 0.0,
                }
                for item in (items or [])
            ]
            render_install_rows()

        def update_install_row(
            index: int,
            *,
            size: str | None = None,
            status: str | None = None,
            progress_value: float | None = None,
        ) -> None:
            try:
                rows = install_rows_state.get("rows", [])
                row = rows[int(index)]
                if size is not None:
                    row["size"] = str(size)
                if status is not None:
                    row["status"] = str(status)
                if progress_value is not None:
                    row["progress"] = max(0.0, min(100.0, float(progress_value)))
                render_install_rows()
            except Exception:
                pass

        def after_install(success_flag: bool) -> None:
            if success_flag:
                _set_dependency_tool(
                    "Instalacja zakończona. Ponownie sprawdzam gotowość wybranego eksportu.",
                    "success",
                    eta="-",
                    progress_value=100.0,
                )
                run_preflight(show_dialog=False, update_status=True)
            else:
                _set_dependency_tool(
                    "Instalacja nie została zakończona poprawnie. Szczegóły są w komunikacie i logu.",
                    "error",
                    eta="-",
                )

        _set_dependency_tool(
            "Instalator przygotowuje listę zależności eksportu. Pauza zatrzyma kolejny pakiet, nie przerwie aktywnego pip.",
            "info",
            eta="liczę",
            progress_value=0.0,
        )
        install_requirements = _missing_install_requirements_for_request(preflight_state.get("request"))
        if not install_requirements:
            _set_dependency_tool(
                "Nie ma brakujących zależności bibliotek dla aktualnego wyboru.",
                "success",
                eta="-",
                progress_value=100.0,
            )
            return
        install_export_requirements(
            parent=dialog,
            status_var_override=dependency_tool_status_var,
            progress_widget=progress,
            progress_var_override=progress_var,
            after_install=after_install,
            install_control=dependency_install_control_state,
            requirements_override=install_requirements,
            install_ui={
                "build": build_install_rows,
                "update": update_install_row,
            },
        )

    def open_selected_mobile_export_executor() -> None:
        if worker_state.get("running"):
            return
        selected_candidates = _selected_export_candidates()
        selection_valid, package_kind, selection_message = _export_selection_validation()
        if not selection_valid:
            set_status(selection_message, "warning")
            return messagebox.showwarning(
                "Brak wyboru eksportu",
                selection_message,
                parent=dialog,
            )

        selected_by_marker = {
            _mobile_export_target_marker(candidate): candidate
            for candidate in selected_candidates
            if _mobile_export_target_marker(candidate) in set(export_marker_order)
        }
        if len(selected_by_marker) != len(selected_candidates):
            return messagebox.showwarning(
                "Nieobsługiwany wybór",
                "Eksport obsługuje pojedynczy model mobilny MP, MT lub MZ oraz kompletny pakiet ALPR MT+MZ lub MP+MT+MZ.",
                parent=dialog,
            )

        artifact = describe_mobile_export_selection(selected_by_marker)
        executor_modal = tk.Toplevel(dialog)
        title = artifact.window_title
        try:
            self.app.style_dialog_window(
                executor_modal,
                title=title,
                geometry="1260x840",
                parent=dialog,
            )
        except Exception:
            executor_modal.title(title)
            executor_modal.geometry("1260x840")
        try:
            executor_modal.transient(dialog)
            executor_modal.resizable(True, True)
            executor_modal.minsize(1040, 740)
        except Exception:
            pass
        _bind_mobile_export_child_window_motion(executor_modal, name="executor_window_configure")
        try:
            executor_modal.after_idle(
                lambda w=executor_modal: _install_mobile_export_win32_move_guard(w, guard_key=f"executor-{id(w)}")
            )
        except Exception:
            pass

        exec_root = tk.Frame(
            executor_modal,
            bg=bg,
            padx=18,
            pady=15,
            highlightthickness=1,
            highlightbackground=border,
        )
        exec_root.pack(fill=tk.BOTH, expand=True)
        exec_root.grid_columnconfigure(0, weight=1)
        exec_root.grid_rowconfigure(2, weight=1)

        tk.Label(
            exec_root,
            text=f"Eksport mobilny: {package_kind}",
            bg=bg,
            fg=fg,
            font=("Segoe UI", 14, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            exec_root,
            text=(
                artifact.message + "\n"
                "Najpierw ustaw parametry dla każdego modelu. Potem możesz sprawdzić gotowość, uzupełnić braki "
                "i zapisać gotowy plik .alprmodel dla klienta mobilnego. Przy eksporcie sprawdzenie uruchomi się automatycznie."
            ),
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=980,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 10))

        model_grid = tk.Frame(exec_root, bg=bg)
        model_grid.grid(row=2, column=0, sticky="nsew")
        selected_count = max(1, len(selected_candidates))
        single_model_layout = selected_count == 1
        model_grid.grid_columnconfigure(0, weight=2, minsize=620)
        model_grid.grid_columnconfigure(1, weight=1, minsize=430)
        model_grid.grid_rowconfigure(0, weight=1)

        model_cards_shell = tk.Frame(model_grid, bg=bg)
        model_cards_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        model_cards_shell.grid_columnconfigure(0, weight=1)
        model_cards_shell.grid_rowconfigure(0, weight=1)
        model_cards_canvas = tk.Canvas(model_cards_shell, bg=bg, highlightthickness=0, bd=0)
        model_cards_scroll = WebSlimScrollbar(
            model_cards_shell,
            orient=tk.VERTICAL,
            command=model_cards_canvas.yview,
            **_mobile_export_scrollbar_kwargs(palette, bg),
        )
        model_cards_canvas.configure(yscrollcommand=model_cards_scroll.set)
        model_cards_canvas.grid(row=0, column=0, sticky="nsew")
        model_cards_scroll.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        model_cards_content = tk.Frame(model_cards_canvas, bg=bg)
        model_cards_window_id = model_cards_canvas.create_window((0, 0), window=model_cards_content, anchor=tk.NW)
        model_card_columns = 1 if single_model_layout else min(2, selected_count)
        for grid_index in range(model_card_columns):
            model_cards_content.grid_columnconfigure(grid_index, weight=1, uniform="mobile_export_model_cards")

        def _sync_model_cards_scrollregion(_event=None) -> None:
            try:
                model_cards_canvas.configure(scrollregion=model_cards_canvas.bbox("all"))
            except Exception:
                pass

        def _sync_model_cards_width(event=None) -> None:
            try:
                width = int(getattr(event, "width", 0) or model_cards_canvas.winfo_width() or 1)
                model_cards_canvas.itemconfigure(model_cards_window_id, width=max(1, width))
                _sync_model_cards_scrollregion()
            except Exception:
                pass

        model_cards_content.bind("<Configure>", _sync_model_cards_scrollregion, add="+")
        model_cards_canvas.bind("<Configure>", _sync_model_cards_width, add="+")

        def _on_model_cards_mousewheel(event) -> str:
            try:
                delta = int(getattr(event, "delta", 0) or 0)
                if delta:
                    model_cards_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
                    return "break"
            except Exception:
                pass
            return ""

        model_cards_canvas.bind("<MouseWheel>", _on_model_cards_mousewheel, add="+")
        model_cards_content.bind("<MouseWheel>", _on_model_cards_mousewheel, add="+")

        def _bind_model_cards_mousewheel(widget) -> None:
            try:
                widget.bind("<MouseWheel>", _on_model_cards_mousewheel, add="+")
            except Exception:
                pass
            try:
                children = list(widget.winfo_children())
            except Exception:
                children = []
            for child in children:
                _bind_model_cards_mousewheel(child)

        model_states: list[dict] = []
        exec_status_var = tk.StringVar(value="Gotowe do sprawdzenia. Eksport sam poprosi o nazwę i miejsce zapisu pliku.")
        exec_progress_var = tk.DoubleVar(value=0.0)
        exec_progress_percent_var = tk.StringVar(value="0%")
        exec_install_control = {
            "running": False,
            "pause_requested": False,
            "paused": False,
        }
        exec_preflight_state = {
            "problems": [],
            "installable": False,
            "ready": False,
        }
        exec_buttons: dict[str, ttk.Button] = {}

        def _default_calibration_for_candidate(candidate: dict) -> str:
            return _mobile_export_default_calibration_path(candidate)

        def _make_model_state(candidate: dict) -> dict:
            run = candidate.get("run")
            img_size = int(candidate.get("img_size") or getattr(run, "img_size", CONFIG.DEFAULT_IMG_SIZE) or CONFIG.DEFAULT_IMG_SIZE)
            calibration_suggestions = _mobile_export_calibration_yaml_candidates(candidate)
            return {
                "candidate": candidate,
                "marker": _mobile_export_target_marker(candidate),
                "litert": tk.BooleanVar(value=True),
                "int8": tk.BooleanVar(value=False),
                "onnx": tk.BooleanVar(value=False),
                "onnx_int8": tk.BooleanVar(value=False),
                "ncnn": tk.BooleanVar(value=False),
                "imgsz": tk.IntVar(value=img_size),
                "conf": tk.DoubleVar(value=CONFIG.DEFAULT_CONFIDENCE),
                "iou": tk.DoubleVar(value=CONFIG.DEFAULT_IOU),
                "calibration": tk.StringVar(value=_default_calibration_for_candidate(candidate)),
                "calibration_choice": tk.StringVar(
                    value=str(calibration_suggestions[0].get("label") or "") if calibration_suggestions else ""
                ),
                "calibration_suggestions": calibration_suggestions,
                "vehicle_classes": tk.StringVar(value=", ".join(_mobile_export_default_vehicle_class_labels(candidate))),
            }

        def _mark_executor_dirty(*_args) -> None:
            exec_preflight_state.update({"ready": False, "problems": [], "installable": False})
            exec_status_var.set("Ustawienia zmienione. Sprawdzenie gotowości zostanie wykonane ponownie przy eksporcie.")
            _sync_executor_buttons()

        def _executor_panel_wraplength() -> int:
            if single_model_layout:
                return 510
            if selected_count > 2:
                return 330
            return 470

        def _format_card_wraplength() -> int:
            if single_model_layout:
                return 230
            if selected_count > 2:
                return 135
            return 190

        def _create_format_hint_card(
            parent,
            *,
            row: int,
            column: int,
            title: str,
            body: str,
            color: str,
            bg_color: str,
            variable,
        ) -> None:
            card = tk.Frame(
                parent,
                bg=blend_hex_colors(bg_color, color, 0.075),
                padx=7,
                pady=5,
                highlightthickness=1,
                highlightbackground=blend_hex_colors(color, bg_color, 0.48),
            )
            card.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0), pady=(0, 5))
            card.grid_columnconfigure(0, weight=1)
            ttk.Checkbutton(
                card,
                text=title,
                variable=variable,
            ).grid(row=0, column=0, sticky="w")
            tk.Label(
                card,
                text=body,
                bg=card["bg"],
                fg=fg,
                font=("Segoe UI", 7),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=_format_card_wraplength(),
            ).grid(row=1, column=0, sticky="ew", pady=(1, 0))

        def _create_model_panel(parent, state: dict, column: int) -> None:
            candidate = state["candidate"]
            marker = str(state.get("marker") or "-")
            grid_column = column % max(1, model_card_columns)
            grid_row = column // max(1, model_card_columns)
            panel_padx = (0, 8 if grid_column < model_card_columns - 1 else 0)
            panel_bg = blend_hex_colors(card_bg, _mobile_export_target_color(palette, candidate), 0.08)
            panel = tk.Frame(
                parent,
                bg=panel_bg,
                padx=12,
                pady=10,
                highlightthickness=1,
                highlightbackground=blend_hex_colors(_mobile_export_target_color(palette, candidate), bg, 0.45),
            )
            panel.grid(row=grid_row, column=grid_column, sticky="nsew", padx=panel_padx, pady=(0, 10))
            panel.grid_columnconfigure(0, weight=1)
            tk.Label(
                panel,
                text=f"{marker} | {candidate.get('model_label') or '-'}",
                bg=panel_bg,
                fg=fg,
                font=("Segoe UI", 11, "bold"),
                anchor=tk.W,
            ).grid(row=0, column=0, sticky="ew")
            params_m, params_source = _mobile_export_candidate_params_millions(candidate)
            tk.Label(
                panel,
                text=(
                    f"YOLO: {_mobile_export_compact_yolo_label(candidate.get('model_version'))} | "
                    f"{_candidate_metric_label(candidate)} | "
                    f"łącznie epok: {_mobile_export_candidate_epochs_label(candidate)} | "
                    f"parametry: {_format_mobile_export_params(params_m, source=params_source, include_source=False)}"
                ),
                bg=panel_bg,
                fg=muted,
                font=("Segoe UI", 8),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=_executor_panel_wraplength(),
            ).grid(row=1, column=0, sticky="ew", pady=(2, 8))

            formats = tk.Frame(panel, bg=blend_hex_colors(panel_bg, accent, 0.045), padx=9, pady=7)
            formats.grid(row=2, column=0, sticky="ew", pady=(0, 8))
            formats.grid_columnconfigure(0, weight=1, uniform="mobile_export_format_cards")
            formats.grid_columnconfigure(1, weight=1, uniform="mobile_export_format_cards")
            tk.Label(
                formats,
                text="Formaty wykonawcze",
                bg=formats["bg"],
                fg=fg,
                font=("Segoe UI", 8, "bold"),
                anchor=tk.W,
            ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
            tk.Label(
                formats,
                text="Kilka formatów ma sens w badaniach; finalnie wybieramy wariant zmierzony na telefonie.",
                bg=formats["bg"],
                fg=muted,
                font=("Segoe UI", 7),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=_executor_panel_wraplength(),
            ).grid(row=0, column=1, sticky="ew", pady=(0, 4), padx=(8, 0))
            _create_format_hint_card(
                formats,
                row=1,
                column=0,
                title="Android stabilny",
                body="LiteRT/TFLite FP32. Najbezpieczniejszy wariant startowy i punkt odniesienia.",
                color=success,
                bg_color=formats["bg"],
                variable=state["litert"],
            )
            _create_format_hint_card(
                formats,
                row=1,
                column=1,
                title="Android lekki",
                body="LiteRT/TFLite INT8. Mniejszy i potencjalnie szybszy, ale wymaga kalibracji i testu jakości.",
                color=warning,
                bg_color=formats["bg"],
                variable=state["int8"],
            )
            _create_format_hint_card(
                formats,
                row=2,
                column=0,
                title="Kontrola",
                body="ONNX FP32. Dobry do diagnostyki, fallbacku i porównania z TFLite.",
                color=accent,
                bg_color=formats["bg"],
                variable=state["onnx"],
            )
            _create_format_hint_card(
                formats,
                row=2,
                column=1,
                title="ONNX INT8",
                body="Kwantyzowany ONNX do badan porownawczych. Wymaga data.yaml kalibracji.",
                color=warning,
                bg_color=formats["bg"],
                variable=state["onnx_int8"],
            )
            _create_format_hint_card(
                formats,
                row=3,
                column=0,
                title="Eksperyment",
                body="NCNN FP32. Tylko gdy klient Android ma obsługę runtime NCNN.",
                color=muted,
                bg_color=formats["bg"],
                variable=state["ncnn"],
            )

            params = tk.Frame(panel, bg=panel_bg)
            params.grid(row=4, column=0, sticky="ew", pady=(0, 8))
            for grid_col in range(6):
                params.grid_columnconfigure(grid_col, weight=1 if grid_col in {1, 3, 5} else 0)
            tk.Label(params, text="imgsz", bg=panel_bg, fg=muted, font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
            ttk.Spinbox(params, from_=128, to=2048, increment=32, textvariable=state["imgsz"], width=7).grid(row=0, column=1, sticky="w", padx=(0, 12))
            tk.Label(params, text="conf", bg=panel_bg, fg=muted, font=("Segoe UI", 8, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 5))
            ttk.Spinbox(params, from_=0.00001, to=1.0, increment=0.01, textvariable=state["conf"], width=7, format="%.5f").grid(row=0, column=3, sticky="w", padx=(0, 12))
            tk.Label(params, text="IoU", bg=panel_bg, fg=muted, font=("Segoe UI", 8, "bold")).grid(row=0, column=4, sticky="w", padx=(0, 5))
            ttk.Spinbox(params, from_=0.00001, to=1.0, increment=0.01, textvariable=state["iou"], width=7, format="%.5f").grid(row=0, column=5, sticky="w")

            next_row = 5
            if marker == "MP":
                vehicle_box = tk.Frame(
                    panel,
                    bg=blend_hex_colors(panel_bg, success, 0.055),
                    padx=9,
                    pady=7,
                    highlightthickness=1,
                    highlightbackground=blend_hex_colors(success, panel_bg, 0.42),
                )
                vehicle_box.grid(row=next_row, column=0, sticky="ew", pady=(0, 8))
                vehicle_box.grid_columnconfigure(0, weight=1)
                known_labels = _mobile_export_candidate_class_labels(candidate)
                known_count = len(known_labels)
                tk.Label(
                    vehicle_box,
                    text="MP: klasy ROI dla MT",
                    bg=vehicle_box["bg"],
                    fg=fg,
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.W,
                ).grid(row=0, column=0, sticky="ew", pady=(0, 3))
                ttk.Entry(vehicle_box, textvariable=state["vehicle_classes"]).grid(row=1, column=0, sticky="ew")
                tk.Label(
                    vehicle_box,
                    text=(
                        "Filtr pojazdów przed MT. Oddziel nazwy przecinkami. "
                        f"Etykiet w modelu: {known_count or 'odczyt przy eksporcie'}."
                    ),
                    bg=vehicle_box["bg"],
                    fg=muted,
                    font=("Segoe UI", 8),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=_executor_panel_wraplength(),
                ).grid(row=2, column=0, sticky="ew", pady=(4, 0))
                next_row += 1

            calibration = tk.Frame(panel, bg=panel_bg)
            calibration.grid(row=next_row, column=0, sticky="ew")
            calibration.grid_columnconfigure(0, weight=1)
            calibration_suggestions = list(state.get("calibration_suggestions") or [])
            calibration_labels = [str(item.get("label") or item.get("path") or "") for item in calibration_suggestions]
            calibration_by_label = {
                str(item.get("label") or item.get("path") or ""): item
                for item in calibration_suggestions
                if str(item.get("label") or item.get("path") or "")
            }
            if marker == "MP":
                calibration_intro = (
                    "INT8 dla MP: użyj data.yaml z reprezentatywnych kadrów O. To kalibracja, nie trening."
                )
                calibration_suggestion_text = (
                    f"Propozycje: {len(calibration_labels)}. Preferuj kadry projektu; COCO tylko awaryjnie."
                )
                calibration_missing_text = (
                    "Brak propozycji. FP32 działa bez data.yaml; INT8 wymaga kadrów O."
                )
            else:
                calibration_intro = (
                    "Kalibracja INT8: użyj data.yaml z tego samego toru co model. "
                    "To tylko reprezentatywna próbka do pomiaru zakresów, nie ponowny trening."
                )
                calibration_suggestion_text = (
                    f"Propozycji: {len(calibration_labels)}. "
                    "Pierwsza pozycja jest ustawiana automatycznie, gdy pasuje do kandydata."
                )
                calibration_missing_text = "Brak zgodnych propozycji. Przy INT8 wskaż data.yaml ręcznie."
            tk.Label(
                calibration,
                text=calibration_intro,
                bg=panel_bg,
                fg=muted,
                font=("Segoe UI", 8),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=_executor_panel_wraplength(),
            ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
            if calibration_labels:
                ttk.Combobox(
                    calibration,
                    textvariable=state["calibration_choice"],
                    values=calibration_labels,
                    state="readonly",
                    height=min(8, max(3, len(calibration_labels))),
                ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

                def apply_suggested_calibration(_event=None, current_state=state, by_label=calibration_by_label) -> None:
                    item = by_label.get(str(current_state["calibration_choice"].get() or ""))
                    if item:
                        current_state["calibration"].set(str(item.get("path") or ""))

                try:
                    calibration.winfo_children()[-1].bind("<<ComboboxSelected>>", apply_suggested_calibration, add="+")
                except Exception:
                    pass
                tk.Label(
                    calibration,
                    text=calibration_suggestion_text,
                    bg=panel_bg,
                    fg=muted,
                    font=("Segoe UI", 8),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=_executor_panel_wraplength(),
                ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
            else:
                tk.Label(
                    calibration,
                    text=calibration_missing_text,
                    bg=panel_bg,
                    fg=warning,
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=_executor_panel_wraplength(),
                ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
            ttk.Entry(calibration, textvariable=state["calibration"]).grid(row=3, column=0, sticky="ew", padx=(0, 7))

            def choose_model_calibration(current_state=state) -> None:
                selected = filedialog.askopenfilename(
                    title=f"Wybierz data.yaml kalibracji INT8 dla {current_state.get('marker')}",
                    parent=executor_modal,
                    filetypes=(("YOLO data.yaml", "data.yaml"), ("YAML", "*.yaml *.yml"), ("Wszystkie pliki", "*.*")),
                )
                if selected:
                    current_state["calibration"].set(selected)

            ttk.Button(calibration, text="Wybierz data.yaml", command=choose_model_calibration).grid(row=3, column=1, sticky="e")

            def _sync_int8_calibration_hint(*_args, current_state=state) -> None:
                try:
                    if (
                        (bool(current_state["int8"].get()) or bool(current_state["onnx_int8"].get()))
                        and not str(current_state["calibration"].get() or "").strip()
                    ):
                        suggestions = list(current_state.get("calibration_suggestions") or [])
                        if suggestions:
                            current_state["calibration_choice"].set(str(suggestions[0].get("label") or ""))
                            current_state["calibration"].set(str(suggestions[0].get("path") or ""))
                except Exception:
                    pass

            try:
                state["int8"].trace_add("write", _sync_int8_calibration_hint)
                state["onnx_int8"].trace_add("write", _sync_int8_calibration_hint)
            except Exception:
                pass

            for key in ("litert", "int8", "onnx", "onnx_int8", "ncnn", "imgsz", "conf", "iou", "calibration", "vehicle_classes"):
                try:
                    state[key].trace_add("write", _mark_executor_dirty)
                except Exception:
                    pass

        for idx, candidate in enumerate(selected_candidates):
            state = _make_model_state(candidate)
            model_states.append(state)
            _create_model_panel(model_cards_content, state, idx)
        _bind_model_cards_mousewheel(model_cards_content)

        checks_shell = tk.Frame(
            model_grid,
            bg=blend_hex_colors(card_bg, accent, 0.035),
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(accent, bg, 0.45),
        )
        checks_shell.grid(row=0, column=1, sticky="nsew")
        checks_shell.grid_columnconfigure(0, weight=1)
        checks_shell.grid_rowconfigure(1, weight=1, minsize=520)
        tk.Label(
            checks_shell,
            text="Gotowość eksportu i zależności",
            bg=checks_shell["bg"],
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        executor_requirements_table = _build_wrapped_info_table(
            checks_shell,
            [
                {"key": "status", "title": "Stan", "minsize": 44, "width": 48, "wrap": 38},
                {"key": "model", "title": "Model", "minsize": 38, "width": 40, "wrap": 34},
                {"key": "requirement", "title": "Wymaganie", "minsize": 112, "width": 128, "wrap": 92},
                {"key": "description", "title": "Opis", "minsize": 360, "width": 560, "weight": 1, "wrap": 320},
            ],
            bg_color=checks_shell["bg"],
            header_bg=blend_hex_colors(checks_shell["bg"], accent, 0.10),
            height=520,
        )
        executor_requirements_table["shell"].grid(row=1, column=0, sticky="nsew")
        executor_install_table = _build_wrapped_info_table(
            checks_shell,
            [
                {"key": "status", "title": "Stan", "minsize": 76, "width": 92, "wrap": 58},
                {"key": "model", "title": "Model", "minsize": 42, "width": 46, "wrap": 36},
                {"key": "requirement", "title": "Pakiet", "minsize": 118, "width": 136, "wrap": 102},
                {"key": "detail", "title": "Aktualny krok", "minsize": 300, "width": 430, "weight": 1, "wrap": 250},
                {"key": "progress", "title": "Postęp", "minsize": 92, "width": 104, "wrap": 64},
            ],
            bg_color=checks_shell["bg"],
            header_bg=blend_hex_colors(checks_shell["bg"], accent, 0.10),
            height=520,
        )
        executor_install_table["shell"].grid(row=1, column=0, sticky="nsew")
        executor_install_table["shell"].grid_remove()
        executor_table_state = {"mode": "preflight"}

        def _show_executor_table(mode: str) -> None:
            normalized = "install" if str(mode or "") == "install" else "preflight"
            if executor_table_state.get("mode") == normalized:
                return
            executor_table_state["mode"] = normalized
            try:
                if normalized == "install":
                    executor_requirements_table["shell"].grid_remove()
                    executor_install_table["shell"].grid()
                else:
                    executor_install_table["shell"].grid_remove()
                    executor_requirements_table["shell"].grid()
            except Exception:
                pass

        status_shell = tk.Frame(exec_root, bg=bg)
        status_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        status_shell.grid_columnconfigure(0, weight=1)
        exec_status_label = tk.Label(
            status_shell,
            textvariable=exec_status_var,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=900,
        )
        exec_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        exec_progress_row = tk.Frame(status_shell, bg=bg)
        exec_progress_row.grid(row=1, column=0, sticky="ew")
        exec_progress_row.grid_columnconfigure(0, weight=1)
        exec_progress = ttk.Progressbar(exec_progress_row, mode="determinate", maximum=100.0, variable=exec_progress_var)
        exec_progress.grid(row=0, column=0, sticky="ew")
        tk.Label(
            exec_progress_row,
            textvariable=exec_progress_percent_var,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            width=5,
            anchor=tk.E,
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        actions_exec = tk.Frame(exec_root, bg=bg)
        actions_exec.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        actions_exec.grid_columnconfigure(0, weight=1)

        def _set_executor_status(text: str, tone: str = "info") -> None:
            colors = {
                "success": success,
                "warning": warning,
                "error": error,
                "info": muted,
            }
            try:
                exec_status_label.configure(fg=colors.get(str(tone or "info"), muted))
            except Exception:
                pass
            exec_status_var.set(str(text or "-"))

        executor_rows_defer_state = {"after": None}

        def _set_executor_rows(rows: list[tuple[str, str, str, str]]) -> None:
            try:
                if _mobile_export_window_is_moving():
                    pending = executor_rows_defer_state.get("after")
                    if pending:
                        try:
                            executor_modal.after_cancel(pending)
                        except Exception:
                            pass
                    deferred_rows = list(rows or [])
                    executor_rows_defer_state["after"] = executor_modal.after(
                        260,
                        lambda current_rows=deferred_rows: _set_executor_rows(current_rows),
                    )
                    return
                executor_rows_defer_state["after"] = None
                _show_executor_table("preflight")
                executor_requirements_table["set_rows"](
                    [
                        {
                            "values": {
                                "status": str(status or "-"),
                                "model": str(model or "-"),
                                "requirement": str(requirement or "-"),
                                "description": str(description or "-"),
                            },
                        }
                        for status, model, requirement, description in rows
                    ]
                )
            except Exception:
                pass

        def _sync_executor_buttons() -> None:
            running = bool(worker_state.get("running"))
            install_running = bool(exec_install_control.get("running"))
            installable = bool(exec_preflight_state.get("installable"))
            ready = bool(exec_preflight_state.get("ready")) and not bool(exec_preflight_state.get("problems"))
            if running:
                primary_text = "Eksport trwa..."
                primary_state = tk.DISABLED
            elif install_running:
                primary_text = "Instaluję zależności..."
                primary_state = tk.DISABLED
            elif ready:
                primary_text = artifact.export_label
                primary_state = tk.NORMAL
            elif installable:
                primary_text = "Uzupełnij zależności"
                primary_state = tk.NORMAL
            elif exec_preflight_state.get("problems"):
                primary_text = "Sprawdź ponownie"
                primary_state = tk.NORMAL
            else:
                primary_text = "Sprawdź gotowość eksportu"
                primary_state = tk.NORMAL
            for key, state_value in (
                ("primary", primary_state),
                ("pause", tk.NORMAL if install_running else tk.DISABLED),
                ("close", tk.DISABLED if running else tk.NORMAL),
            ):
                button = exec_buttons.get(key)
                if button is None:
                    continue
                try:
                    button.configure(state=state_value)
                except Exception:
                    pass
            try:
                primary_button = exec_buttons.get("primary")
                if primary_button is not None:
                    primary_button.configure(text=primary_text)
            except Exception:
                pass
            try:
                pause_button = exec_buttons.get("pause")
                if pause_button is not None:
                    pause_button.configure(text="Wznów" if exec_install_control.get("pause_requested") else "Pauza")
            except Exception:
                pass

        def _default_executor_destination() -> Path:
            if "MT" in selected_by_marker and "MZ" in selected_by_marker:
                return self._build_mobile_alpr_package_export_path(
                    selected_by_marker["MT"],
                    selected_by_marker["MZ"],
                    vehicle_candidate=selected_by_marker.get("MP"),
                )
            candidate = model_states[0]["candidate"]
            return self._build_mobile_model_export_path(candidate.get("run"), str(candidate.get("target") or ""), Path(candidate.get("best_weights")))

        def _build_state_request(state: dict, *, destination: Path, include_metadata: bool) -> MobileExportRequest:
            return _build_mobile_export_request_for_candidate(
                state["candidate"],
                destination=destination,
                include_metadata=include_metadata,
                option_vars=state,
            )

        def _state_export_manifest_snapshot(state: dict) -> dict:
            candidate = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
            formats, quantizations = _selected_export_formats_and_quantizations(state)
            format_quantizations = _selected_export_format_quantizations(state)

            def _state_value(key: str, fallback=""):
                value = state.get(key)
                try:
                    return value.get()
                except Exception:
                    return fallback if value is None else value

            return self._json_safe_training_value(
                {
                    "schema": "alpr.export.model_slot_settings.v1",
                    "marker": str(state.get("marker") or _mobile_export_target_marker(candidate)),
                    "candidate": _mobile_export_candidate_manifest_snapshot(candidate),
                    "formats": list(formats),
                    "quantizations": list(dict.fromkeys(quantizations)),
                    "format_quantizations": {key: list(value) for key, value in format_quantizations.items()},
                    "image_size": int(_state_value("imgsz", CONFIG.DEFAULT_IMG_SIZE) or CONFIG.DEFAULT_IMG_SIZE),
                    "confidence_threshold": max(0.0, min(1.0, float(_state_value("conf", CONFIG.DEFAULT_CONFIDENCE) or CONFIG.DEFAULT_CONFIDENCE))),
                    "iou_threshold": max(0.0, min(1.0, float(_state_value("iou", CONFIG.DEFAULT_IOU) or CONFIG.DEFAULT_IOU))),
                    "calibration_data": str(_state_value("calibration", "") or ""),
                    "calibration_choice": str(_state_value("calibration_choice", "") or ""),
                    "vehicle_classes": str(_state_value("vehicle_classes", "") or ""),
                }
            )

        def _state_for_marker(marker: str) -> dict | None:
            for state in model_states:
                if str(state.get("marker") or "") == marker:
                    return state
            return None

        def _build_executor_package_request(destination: Path, *, include_metadata: bool) -> MobileAlprPackageRequest | MobileExportRequest:
            if len(model_states) == 1:
                return _build_state_request(model_states[0], destination=destination, include_metadata=include_metadata)
            plate_state = _state_for_marker("MT")
            char_state = _state_for_marker("MZ")
            vehicle_state = _state_for_marker("MP")
            if not plate_state or not char_state:
                raise MobileExportError("Komplet ALPR wymaga dokladnie jednego modelu MT i jednego modelu MZ.")
            package_parts = []
            if vehicle_state:
                package_parts.append(str(vehicle_state["candidate"].get("model_label") or "MP"))
            package_parts.extend(
                [
                    str(plate_state["candidate"].get("model_label") or "MT"),
                    str(char_state["candidate"].get("model_label") or "MZ"),
                ]
            )
            package_id_seed = "ALPR-" + "-".join(package_parts)
            package_variant = "MP+MT+MZ" if vehicle_state else "MT+MZ"
            vehicle_request = None
            if vehicle_state:
                vehicle_request = _build_state_request(
                    vehicle_state,
                    destination=destination.parent / "_mp_single_for_bundle.alprmodel",
                    include_metadata=include_metadata,
                )
            package_metadata = {
                "source": "mobile_export_center",
                "purpose": "mobile_detection_package",
                "pipeline_variant": package_variant,
                "vehicle_candidate_iid": str(vehicle_state["candidate"].get("iid") or "") if vehicle_state else "",
                "plate_candidate_iid": str(plate_state["candidate"].get("iid") or ""),
                "character_candidate_iid": str(char_state["candidate"].get("iid") or ""),
            }
            if include_metadata:
                package_metadata["package_configuration"] = self._json_safe_training_value(
                    {
                        "schema": "alpr.export.package_configuration.v1",
                        "pipeline_variant": package_variant,
                        "selection_rules": {
                            "max_models": 3,
                            "allowed_sets": ["MT+MZ", "MP+MT+MZ", "MP", "MT", "MZ"],
                            "complete_alpr_sets": ["MT+MZ", "MP+MT+MZ"],
                        },
                        "selected_models": {
                            marker: _state_export_manifest_snapshot(state)
                            for marker, state in (
                                ("MP", vehicle_state),
                                ("MT", plate_state),
                                ("MZ", char_state),
                            )
                            if state is not None
                        },
                        "projects": {
                            marker: {
                                "name": str(state["candidate"].get("project_name") or ""),
                                "root": str(state["candidate"].get("project_root") or ""),
                            }
                            for marker, state in (
                                ("MP", vehicle_state),
                                ("MT", plate_state),
                                ("MZ", char_state),
                            )
                            if state is not None
                        },
                    }
                )
            if vehicle_state:
                vehicle_classes_value = vehicle_state.get("vehicle_classes")
                try:
                    vehicle_classes_value = vehicle_classes_value.get()
                except Exception:
                    pass
                package_metadata["vehicle_detection"] = _mobile_export_vehicle_class_payload(
                    vehicle_state["candidate"],
                    vehicle_classes_value,
                )
            return MobileAlprPackageRequest(
                destination=destination,
                vehicle_request=vehicle_request,
                plate_request=_build_state_request(
                    plate_state,
                    destination=destination.parent / "_mt_single_for_bundle.alprmodel",
                    include_metadata=include_metadata,
                ),
                character_request=_build_state_request(
                    char_state,
                    destination=destination.parent / "_mz_single_for_bundle.alprmodel",
                    include_metadata=include_metadata,
                ),
                package_id=self._safe_model_export_slug(package_id_seed, fallback="ALPR-package")[:80],
                name=(
                    f"ALPR {package_variant} | "
                    + " + ".join(package_parts)
                ),
                version="1",
                metadata=package_metadata,
            )

        def _executor_formats_description(state: dict) -> str:
            formats, quantizations = _selected_export_formats_and_quantizations(state)
            format_quantizations = _selected_export_format_quantizations(state)
            request = MobileExportRequest(
                checkpoint=Path(state["candidate"].get("best_weights")),
                destination=Path("preview.alprmodel"),
                role=str(state["candidate"].get("role") or "character"),
                formats=tuple(formats),
                image_size=int(state["imgsz"].get() or CONFIG.DEFAULT_IMG_SIZE),
                quantizations=tuple(quantizations),
                format_quantizations=format_quantizations,
            )
            return _formats_description(request)

        def _executor_dependency_specs_for_state(state: dict) -> list[dict]:
            formats, quantizations = _selected_export_formats_and_quantizations(state)
            descriptions = {
                "ultralytics": "Eksport checkpointu YOLO do formatu mobilnego.",
                "torch": "Odczyt checkpointu i uruchomienie eksportera YOLO.",
                "torchvision": "Zgodna biblioteka wizji używana przez Ultralytics.",
                "onnx": "Budowa i inspekcja wariantu ONNX oraz ścieżki TFLite.",
                "onnxruntime": "Walidacja ONNX po eksporcie; używamy wariantu CPU.",
                "onnxslim": "Uproszczenie grafu ONNX.",
                "PIL": "Odczyt obrazów kalibracyjnych dla wariantu ONNX INT8.",
                "yaml": "Odczyt data.yaml używanego do kalibracji ONNX INT8.",
                "tensorflow": "Konwersja do wariantu TFLite/LiteRT.",
                "tf_keras": "Warstwa Keras wymagana przez konwerter TFLite.",
                "sng4onnx": "Narzędzie pomocnicze konwersji ONNX do TensorFlow.",
                "onnx_graphsurgeon": "Narzędzie NVIDIA używane w ścieżce ONNX do TensorFlow.",
                "ai_edge_litert": "Biblioteka Google LiteRT wymagana przy eksporcie TFLite.",
                "onnx2tf": "Konwerter pośredni ONNX do TensorFlow; wersja musi pasować do Ultralytics.",
                "google.protobuf": "Serializacja grafów TensorFlow/ONNX.",
                "ncnn": "Eksport wariantu NCNN.",
                "pnnx": "Konwerter wymagany przez eksport NCNN.",
            }
            result: list[dict] = []
            for row in mobile_export_required_specs(
                tuple(formats),
                quantizations=tuple(quantizations),
                format_quantizations=_selected_export_format_quantizations(state),
            ):
                module = str(row.get("module") or "").strip()
                result.append(
                    {
                        "module": module,
                        "spec": str(row.get("spec") or module).strip(),
                        "scope": [part.strip() for part in str(row.get("scope") or "Eksport").split(",") if part.strip()],
                        "description": [descriptions.get(module, "Biblioteka wymagana przez wybrany wariant eksportu.")],
                    }
                )
            return result

        def _executor_missing_install_requirements() -> list[dict]:
            by_spec: dict[str, dict] = {}
            for state in model_states:
                marker = str(state.get("marker") or "-")
                for item in _executor_dependency_specs_for_state(state):
                    module = str(item.get("module") or "").strip()
                    spec = str(item.get("spec") or module).strip()
                    requirement_ok, _detail = mobile_export_requirement_status(spec, module)
                    if not module or requirement_ok:
                        continue
                    entry = by_spec.setdefault(
                        spec,
                        {
                            "spec": spec,
                            "name": _mobile_export_requirement_name(spec),
                            "models": [],
                            "description": [],
                        },
                    )
                    if marker not in entry["models"]:
                        entry["models"].append(marker)
                    for part in item.get("description") or []:
                        if part not in entry["description"]:
                            entry["description"].append(part)
            result = list(by_spec.values())
            result.sort(key=lambda row: (str(row.get("name") or ""), str(row.get("spec") or "")))
            return result

        def _executor_model_check_rows(state: dict) -> list[tuple[str, str, str, str]]:
            rows: list[tuple[str, str, str, str]] = []
            marker = str(state.get("marker") or "-")
            candidate = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
            try:
                checkpoint = Path(candidate.get("best_weights"))
                checkpoint_ok = checkpoint.exists() and checkpoint.is_file()
            except Exception:
                checkpoint = Path("")
                checkpoint_ok = False
            rows.append(
                (
                    "Jest" if checkpoint_ok else "Brak",
                    marker,
                    "Checkpoint best.pt",
                    f"{'Gotowe' if checkpoint_ok else 'Brak'}: {checkpoint if str(checkpoint) else 'brak ścieżki checkpointu.'}",
                )
            )

            formats, quantizations = _selected_export_formats_and_quantizations(state)
            format_text = _executor_formats_description(state)
            rows.append(
                (
                    "Jest" if formats else "Brak",
                    marker,
                    "Formaty eksportu",
                    f"{'Wybrane formaty' if formats else 'Brak wybranych formatów'}: {format_text}",
                )
            )

            for item in _executor_dependency_specs_for_state(state):
                module = str(item.get("module") or "").strip()
                spec = str(item.get("spec") or module).strip()
                installed, install_detail = mobile_export_requirement_status(spec, module)
                scope = ", ".join(item.get("scope") or []) or "Eksport"
                description = " ".join(str(part) for part in (item.get("description") or []) if str(part).strip())
                rows.append(
                    (
                        "Jest" if installed else "Brak",
                        marker,
                        f"{scope}: {spec}",
                        (
                            f"{'Zainstalowane' if installed else 'Do instalacji'}: {install_detail}. "
                            f"{description or 'Biblioteka wymagana przez wybrany wariant eksportu.'}"
                        ),
                    )
                )

            runtime_problem = check_mobile_yolo_export_runtime()
            rows.append(
                (
                    "Błąd" if runtime_problem else "Jest",
                    marker,
                    "Runtime YOLO",
                    (
                        f"Błąd runtime: {runtime_problem}"
                        if runtime_problem
                        else "Runtime gotowy: ultralytics, torch i torchvision importują się poprawnie."
                    ),
                )
            )

            if "int8" in quantizations:
                calibration_text = ""
                try:
                    calibration_text = str(state["calibration"].get() or "").strip()
                except Exception:
                    calibration_text = ""
                calibration_path = Path(calibration_text) if calibration_text else None
                calibration_problem = _mobile_export_calibration_problem(candidate, calibration_path)
                calibration_ok = not bool(calibration_problem)
                rows.append(
                    (
                        "Jest" if calibration_ok else "Brak",
                        marker,
                        "Kalibracja INT8",
                        (
                            f"Dataset kalibracyjny zgodny z torem {marker}: {calibration_path}"
                            if calibration_ok
                            else calibration_problem
                        ),
                    )
                )
            return rows

        def _executor_local_validation_problems() -> list[str]:
            problems: list[str] = []
            for state in model_states:
                candidate = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
                marker = str(state.get("marker") or _mobile_export_target_marker(candidate) or "-")
                _formats, quantizations = _selected_export_formats_and_quantizations(state)
                if "int8" not in quantizations:
                    continue
                try:
                    calibration_text = str(state["calibration"].get() or "").strip()
                except Exception:
                    calibration_text = ""
                calibration_path = Path(calibration_text) if calibration_text else None
                problem = _mobile_export_calibration_problem(candidate, calibration_path)
                if problem:
                    problems.append(f"{marker}: {problem}")
            return problems

        def _executor_problem_is_reflected(problem: str) -> bool:
            lower = str(problem or "").lower()
            reflected_tokens = (
                "brak pakietu",
                "brak zależności",
                "brak zaleznosci",
                "ultralytics",
                "torch",
                "torchvision",
                "onnx",
                "onnxruntime",
                "onnxslim",
                "tensorflow",
                "onnx2tf",
                "ncnn",
                "int8 wymaga datasetu",
                "checkpointu",
                "format",
                "data.yaml",
                "data.yaml wygląda",
            )
            return any(token in lower for token in reflected_tokens)

        def run_executor_preflight(*, show_dialog: bool = False) -> bool:
            rows: list[tuple[str, str, str, str]] = []
            all_problems: list[str] = []
            for state in model_states:
                rows.extend(_executor_model_check_rows(state))
            local_problems = _executor_local_validation_problems()
            try:
                destination = _default_executor_destination()
                request = _build_executor_package_request(destination, include_metadata=False)
                if isinstance(request, MobileAlprPackageRequest):
                    problems = package_exporter.preflight(request)
                else:
                    problems = exporter.preflight(request)
            except Exception as exc:
                problems = [str(exc)]
            problems = list(local_problems) + list(problems or [])

            problem_seen: set[str] = set()
            for problem in problems:
                clean = str(problem or "").strip()
                if not clean or clean in problem_seen:
                    continue
                problem_seen.add(clean)
                all_problems.append(clean)
                marker = "-"
                if clean.startswith("MP:"):
                    marker = "MP"
                    clean = clean[3:].strip()
                elif clean.startswith("MT:"):
                    marker = "MT"
                    clean = clean[3:].strip()
                elif clean.startswith("MZ:"):
                    marker = "MZ"
                    clean = clean[3:].strip()
                if not _executor_problem_is_reflected(clean):
                    rows.append(("Brak", marker, "Sprawdzenie", clean))

            try:
                rows.append(("Jest", artifact.label, "Miejsce zapisu", str(_default_executor_destination())))
            except Exception as exc:
                rows.append(("Brak", artifact.label, "Miejsce zapisu", str(exc)))

            installable = bool(_executor_missing_install_requirements())
            exec_preflight_state.update(
                {
                    "problems": list(all_problems),
                    "installable": bool(installable),
                    "ready": not bool(all_problems),
                }
            )
            _set_executor_rows(rows)
            if all_problems:
                if installable:
                    _set_executor_status(
                        "Sprawdzenie gotowości wykryło braki bibliotek. Uzupełnij zależności w tym oknie, a potem ponów eksport.",
                        "warning",
                    )
                else:
                    _set_executor_status(
                        "Sprawdzenie gotowości wymaga korekty ustawień modelu lub danych. Szczegóły są w tabeli.",
                        "warning",
                    )
                if show_dialog:
                    messagebox.showwarning(
                        "Sprawdzenie gotowości eksportu",
                        "Eksport nie jest jeszcze gotowy. Szczegóły są w tabeli sprawdzenia.",
                        parent=executor_modal,
                    )
                _sync_executor_buttons()
                return False
            _set_executor_status("Eksport gotowy. Możesz zapisać gotowy plik .alprmodel.", "success")
            exec_progress.stop()
            exec_progress.configure(mode="determinate")
            exec_progress_var.set(100.0)
            exec_progress_percent_var.set("100%")
            _sync_executor_buttons()
            return True

        def install_executor_dependencies() -> None:
            if worker_state.get("running"):
                return
            if not exec_preflight_state.get("problems"):
                run_executor_preflight(show_dialog=False)
            install_requirements = _executor_missing_install_requirements()
            if not exec_preflight_state.get("installable") or not install_requirements:
                _set_executor_status(
                    "Nie ma braków bibliotek do instalacji albo problem wymaga zmiany ustawień.",
                    "warning",
                )
                return

            install_rows_state: dict[str, list[dict]] = {"rows": []}

            def render_install_rows() -> None:
                _show_executor_table("install")
                rows: list[dict] = []
                for item in install_rows_state.get("rows", []):
                    progress_value = float(item.get("progress") or 0.0)
                    model_text = item.get("models") or item.get("model") or "-"
                    if isinstance(model_text, (list, tuple, set)):
                        model_text = ", ".join(str(value) for value in model_text if str(value).strip())
                    rows.append(
                        {
                            "values": {
                                "status": str(item.get("status") or "Czeka"),
                                "model": str(model_text or "-"),
                                "requirement": str(item.get("spec") or "-"),
                                "detail": str(item.get("size") or "Czeka na rozpoczęcie instalacji."),
                                "progress": _mobile_export_progress_label(progress_value),
                                "_progress_value": progress_value,
                            },
                        }
                    )
                executor_install_table["set_rows"](rows)

            def build_install_rows(items: list[dict]) -> None:
                install_rows_state["rows"] = [
                    {
                        "spec": str(item.get("spec") or item.get("name") or "-"),
                        "models": item.get("models") or item.get("model") or "-",
                        "size": "-",
                        "status": "Czeka",
                        "progress": 0.0,
                    }
                    for item in (items or [])
                ]
                render_install_rows()

            def update_install_row(
                index: int,
                *,
                size: str | None = None,
                status: str | None = None,
                progress_value: float | None = None,
            ) -> None:
                try:
                    row = install_rows_state["rows"][int(index)]
                    if size is not None:
                        row["size"] = str(size)
                    if status is not None:
                        row["status"] = str(status)
                    if progress_value is not None:
                        row["progress"] = max(0.0, min(100.0, float(progress_value)))
                    render_install_rows()
                except Exception:
                    pass

            def after_install(success_flag: bool) -> None:
                exec_install_control["running"] = False
                exec_install_control["pause_requested"] = False
                exec_install_control["paused"] = False
                if success_flag:
                    _set_executor_status("Zależności gotowe. Ponownie sprawdzam gotowość eksportu.", "success")
                    run_executor_preflight(show_dialog=False)
                else:
                    _set_executor_status("Instalacja zależności nie została zakończona poprawnie.", "error")
                _sync_executor_buttons()

            exec_install_control["running"] = True
            exec_install_control["pause_requested"] = False
            exec_install_control["paused"] = False
            _sync_executor_buttons()
            install_export_requirements(
                parent=executor_modal,
                status_var_override=exec_status_var,
                progress_widget=exec_progress,
                progress_var_override=exec_progress_var,
                after_install=after_install,
                install_control=exec_install_control,
                requirements_override=install_requirements,
                install_ui={
                    "build": build_install_rows,
                    "update": update_install_row,
                },
            )

        def toggle_executor_pause() -> None:
            if not exec_install_control.get("running"):
                return
            exec_install_control["pause_requested"] = not bool(exec_install_control.get("pause_requested"))
            exec_install_control["paused"] = bool(exec_install_control.get("pause_requested"))
            _set_executor_status(
                "Pauza jest uzbrojona. Instalator zatrzyma się przed następną zależnością."
                if exec_install_control.get("pause_requested")
                else "Wznawiam instalację zależności eksportu.",
                "warning" if exec_install_control.get("pause_requested") else "info",
            )
            _sync_executor_buttons()

        def export_executor_package() -> None:
            if worker_state.get("running"):
                return
            exec_progress.stop()
            exec_progress.configure(mode="determinate")
            exec_progress_var.set(0.0)
            exec_progress_percent_var.set("0%")
            if not run_executor_preflight(show_dialog=False):
                if exec_preflight_state.get("installable"):
                    _set_executor_status(
                        "Najpierw uzupełnij zależności w tym oknie. Dopiero potem eksport będzie przewidywalny.",
                        "warning",
                    )
                return
            default_destination = _default_executor_destination()
            destination = _ask_mobile_package_destination(
                default_destination,
                parent=executor_modal,
                title=artifact.save_title,
            )
            if destination is None:
                _set_executor_status("Eksport anulowany. Plik nie został zapisany.", "info")
                return
            try:
                request = _build_executor_package_request(destination, include_metadata=True)
            except Exception as exc:
                _set_executor_status(str(exc), "error")
                return messagebox.showerror("Błąd eksportu mobilnego", str(exc), parent=executor_modal)

            set_running(True)
            _sync_executor_buttons()
            _set_executor_status(
                _mobile_export_verbose_progress_message(
                    0.0,
                    artifact.progress_message + " Postęp zapisuję poniżej.",
                ),
                "info",
            )
            exec_progress_var.set(0.0)
            exec_progress_percent_var.set("0%")

            def progress_cb(percent, message: str) -> None:
                def apply_update() -> None:
                    try:
                        if percent is None:
                            exec_progress.configure(mode="indeterminate")
                            exec_progress.start(18)
                            exec_progress_percent_var.set("...")
                        else:
                            exec_progress.stop()
                            exec_progress.configure(mode="determinate")
                            progress_value = _mobile_export_progress_value(percent)
                            exec_progress_var.set(progress_value)
                            exec_progress_percent_var.set(_mobile_export_progress_label(progress_value))
                    except Exception:
                        pass
                    _set_executor_status(_mobile_export_verbose_progress_message(percent, message), "info")

                try:
                    executor_modal.after(0, apply_update)
                except Exception:
                    pass

            def worker() -> None:
                try:
                    if isinstance(request, MobileAlprPackageRequest):
                        package_path = package_exporter.export(request, progress=progress_cb)
                        log_text = f"[MOBILE EXPORT] Utworzono pakiet ALPR {package_kind}: {package_path}"
                    else:
                        package_path = exporter.export(request, progress=progress_cb)
                        log_text = f"[MOBILE EXPORT] Utworzono model mobilny: {package_path}"

                    def done() -> None:
                        set_running(False)
                        _sync_executor_buttons()
                        exec_progress.stop()
                        exec_progress.configure(mode="determinate")
                        exec_progress_var.set(100.0)
                        exec_progress_percent_var.set("100%")
                        _set_executor_status(f"{artifact.success_title}: {package_path}", "success")
                        set_status(f"{artifact.success_title}: {package_path}", "success")
                        try:
                            self._append_train_log(log_text)
                        except Exception:
                            pass
                        messagebox.showinfo(
                            artifact.success_title,
                            f"{artifact.message}\n{package_path}",
                            parent=executor_modal,
                        )

                    executor_modal.after(0, done)
                except Exception as exc:
                    logger.exception("Nie udało się wykonać eksportu mobilnego")
                    error_text = str(exc)

                    def failed() -> None:
                        set_running(False)
                        _sync_executor_buttons()
                        exec_progress.stop()
                        exec_progress.configure(mode="determinate")
                        _set_executor_status(f"Błąd eksportu: {error_text}", "error")
                        set_status(f"Nie udało się wykonać eksportu mobilnego: {error_text}", "error")
                        messagebox.showerror("Błąd eksportu mobilnego", error_text, parent=executor_modal)

                    try:
                        executor_modal.after(0, failed)
                    except Exception:
                        pass

            threading.Thread(target=worker, name="mobile-selected-export", daemon=True).start()

        def run_executor_primary_action() -> None:
            if worker_state.get("running") or exec_install_control.get("running"):
                return
            if bool(exec_preflight_state.get("ready")) and not bool(exec_preflight_state.get("problems")):
                export_executor_package()
                return
            if bool(exec_preflight_state.get("installable")) and bool(exec_preflight_state.get("problems")):
                install_executor_dependencies()
                return
            run_executor_preflight(show_dialog=True)

        exec_buttons["primary"] = ttk.Button(actions_exec, text="Sprawdź gotowość eksportu", command=run_executor_primary_action)
        exec_buttons["primary"].grid(row=0, column=1, sticky="e", padx=(0, 7), ipadx=12, ipady=3)
        exec_buttons["pause"] = ttk.Button(actions_exec, text="Pauza", command=toggle_executor_pause, state=tk.DISABLED)
        exec_buttons["pause"].grid(row=0, column=2, sticky="e", padx=(0, 7), ipadx=8, ipady=3)
        exec_buttons["close"] = ttk.Button(actions_exec, text="Zamknij", command=executor_modal.destroy)
        exec_buttons["close"].grid(row=0, column=3, sticky="e", ipadx=8, ipady=3)

        _set_executor_rows(
            [
                (
                    "-",
                    str(state.get("marker") or "-"),
                    "Czeka na sprawdzenie",
                    _executor_formats_description(state),
                )
                for state in model_states
            ]
        )
        _sync_executor_buttons()

    def export_package() -> None:
        if worker_state.get("running"):
            return
        try:
            preview_request = build_preflight_request()
        except Exception as exc:
            set_status(f"Eksport nie może jeszcze ruszyć: {exc}", "warning")
            return messagebox.showerror("Błąd eksportu mobilnego", str(exc), parent=dialog)
        if not run_preflight(show_dialog=False, update_status=True):
            problems = list(preflight_state.get("problems") or [])
            if preflight_state.get("installable"):
                _set_dependency_tool(
                    "Eksport zatrzymany podczas sprawdzenia gotowości. Zainstaluj brakujące zależności albo zmień format eksportu.",
                    "warning",
                    eta="po instalacji",
                    progress_value=0.0,
                )
                return
            show_preflight_problems_dialog(
                problems,
                parent=dialog,
                title="Sprawdzenie gotowości modelu",
                context="Eksport wymaga korekty ustawień wskazanych poniżej.",
            )
            return
        destination = _ask_mobile_package_destination(
            Path(preview_request.destination),
            parent=dialog,
            title="Eksportuj model mobilny (.alprmodel)",
        )
        if destination is None:
            set_status("Eksport anulowany. Plik nie został zapisany.", "info")
            return
        destination_var.set(str(destination))
        if not run_preflight(show_dialog=True):
            return
        request = build_request()
        progress_var.set(0.0)
        progress_percent_var.set("0%")
        set_running(True)
        set_status(
            _mobile_export_verbose_progress_message(
                0.0,
                "Eksportuję model mobilny. To może potrwać, szczególnie dla LiteRT/TFLite.",
            ),
            "info",
        )

        def progress_cb(percent, message: str) -> None:
            def apply_update() -> None:
                if percent is None:
                    progress.configure(mode="indeterminate")
                    progress.start(18)
                    progress_percent_var.set("...")
                else:
                    progress.stop()
                    progress.configure(mode="determinate")
                    progress_value = _mobile_export_progress_value(percent)
                    progress_var.set(progress_value)
                    progress_percent_var.set(_mobile_export_progress_label(progress_value))
                set_status(_mobile_export_verbose_progress_message(percent, message), "info")

            try:
                dialog.after(0, apply_update)
            except Exception:
                pass

        def worker() -> None:
            try:
                package_path = exporter.export(request, progress=progress_cb)

                def done() -> None:
                    set_running(False)
                    progress.stop()
                    progress.configure(mode="determinate")
                    progress_var.set(100.0)
                    progress_percent_var.set("100%")
                    set_status(f"Model mobilny gotowy: {package_path}", "success")
                    try:
                        self._append_train_log(f"[MOBILE EXPORT] Utworzono model mobilny .alprmodel: {package_path}")
                    except Exception:
                        pass
                    messagebox.showinfo(
                        "Model mobilny gotowy",
                        f"Utworzono model mobilny dla klienta Android:\n{package_path}",
                        parent=dialog,
                    )

                dialog.after(0, done)
            except Exception as exc:
                logger.exception("Nie udało się wyeksportować modelu mobilnego")
                error_text = str(exc)

                def failed() -> None:
                    set_running(False)
                    progress.stop()
                    progress.configure(mode="determinate")
                    set_status(f"Nie udało się wyeksportować modelu mobilnego: {error_text}", "error")
                    messagebox.showerror("Błąd eksportu mobilnego", error_text, parent=dialog)

                try:
                    dialog.after(0, failed)
                except Exception:
                    pass

        threading.Thread(target=worker, name="mobile-model-export", daemon=True).start()

    def open_complete_alpr_package_dialog() -> None:
        if worker_state.get("running"):
            return
        plate_candidates = [candidate for candidate in candidates if _mobile_export_target_marker(candidate) == "MT"]
        char_candidates = [candidate for candidate in candidates if _mobile_export_target_marker(candidate) == "MZ"]
        if not plate_candidates or not char_candidates:
            missing = []
            if not plate_candidates:
                missing.append("MT")
            if not char_candidates:
                missing.append("MZ")
            return messagebox.showwarning(
                "Brak kandydatów",
                "Kompletny pakiet ALPR wymaga modelu tablic MT i modelu znaków MZ.\n"
                f"Brakuje: {', '.join(missing)}.",
                parent=dialog,
            )

        picker = tk.Toplevel(dialog)
        try:
            self.app.style_dialog_window(
                picker,
                title="Pakiet ALPR MT+MZ",
                geometry="760x470",
                parent=dialog,
            )
        except Exception:
            picker.title("Pakiet ALPR MT+MZ")
            picker.geometry("760x470")
        try:
            picker.transient(dialog)
            picker.resizable(True, True)
            picker.minsize(680, 420)
        except Exception:
            pass
        _bind_mobile_export_child_window_motion(picker, name="package_picker_configure")
        try:
            picker.after_idle(lambda w=picker: _install_mobile_export_win32_move_guard(w, guard_key=f"picker-{id(w)}"))
        except Exception:
            pass

        picker_bg = bg
        picker_card = card_bg
        root_picker = tk.Frame(
            picker,
            bg=picker_bg,
            padx=18,
            pady=16,
            highlightthickness=1,
            highlightbackground=border,
        )
        root_picker.pack(fill=tk.BOTH, expand=True)
        root_picker.grid_columnconfigure(0, weight=1)
        root_picker.grid_rowconfigure(3, weight=1)

        tk.Label(
            root_picker,
            text="Pakiet ALPR do detekcji mobilnej",
            bg=picker_bg,
            fg=fg,
            font=("Segoe UI", 13, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            root_picker,
            text=(
                "Wybierz parę modeli dla aplikacji mobilnej. Eksporter zbuduje dwa zwalidowane "
                "pakiety modeli i połączy je w jeden manifest systemowy MT+MZ."
            ),
            bg=picker_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=690,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        def _option_label(candidate: dict) -> str:
            metric = _candidate_metric_label(candidate)
            date = _format_mobile_export_datetime(
                candidate.get("finished_at") or candidate.get("started_at") or candidate.get("created_at")
            )
            yolo = _mobile_export_compact_yolo_label(candidate.get("model_version"))
            model = str(candidate.get("model_label") or candidate.get("run_label") or candidate.get("iid") or "-")
            return f"{_mobile_export_target_marker(candidate)} | {model} | {yolo} | {metric} | {date}"

        def _unique_options(items: list[dict]) -> tuple[list[str], dict[str, dict]]:
            values: list[str] = []
            mapping: dict[str, dict] = {}
            seen: dict[str, int] = {}
            for item in items:
                label = _option_label(item)
                seen[label] = int(seen.get(label) or 0) + 1
                if seen[label] > 1:
                    label = f"{label} | {item.get('iid')}"
                values.append(label)
                mapping[label] = item
            return values, mapping

        plate_values, plate_by_label = _unique_options(plate_candidates)
        char_values, char_by_label = _unique_options(char_candidates)
        current_candidate = selected_state.get("candidate") if isinstance(selected_state.get("candidate"), dict) else {}
        plate_default = plate_values[0]
        char_default = char_values[0]
        if current_candidate and _mobile_export_target_marker(current_candidate) == "MT":
            current_label = next((label for label, item in plate_by_label.items() if item is current_candidate), "")
            if current_label:
                plate_default = current_label
        if current_candidate and _mobile_export_target_marker(current_candidate) == "MZ":
            current_label = next((label for label, item in char_by_label.items() if item is current_candidate), "")
            if current_label:
                char_default = current_label

        plate_var = tk.StringVar(value=plate_default)
        char_var = tk.StringVar(value=char_default)
        package_destination_state = {"last": "", "selection": f"{plate_default}\0{char_default}"}
        package_status_var = tk.StringVar(value="-")

        form_grid = tk.Frame(root_picker, bg=picker_bg)
        form_grid.grid(row=2, column=0, sticky="ew")
        form_grid.grid_columnconfigure(1, weight=1)

        tk.Label(form_grid, text="Model tablic MT", bg=picker_bg, fg=muted, font=("Segoe UI", 9, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=(0, 7),
        )
        plate_combo = ttk.Combobox(form_grid, textvariable=plate_var, values=plate_values, state="readonly")
        plate_combo.grid(row=0, column=1, sticky="ew", pady=(0, 7))

        tk.Label(form_grid, text="Model znaków MZ", bg=picker_bg, fg=muted, font=("Segoe UI", 9, "bold")).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=(0, 7),
        )
        char_combo = ttk.Combobox(form_grid, textvariable=char_var, values=char_values, state="readonly")
        char_combo.grid(row=1, column=1, sticky="ew", pady=(0, 7))

        def _selected_package_candidates() -> tuple[dict, dict]:
            plate_candidate = plate_by_label.get(str(plate_var.get()))
            char_candidate = char_by_label.get(str(char_var.get()))
            if not plate_candidate or not char_candidate:
                raise MobileExportError("Wybierz model MT i model MZ.")
            return plate_candidate, char_candidate

        def _default_complete_package_destination(plate_candidate: dict, char_candidate: dict) -> Path:
            previous = str(package_destination_state.get("last") or "").strip()
            if previous:
                return Path(previous)
            try:
                return self._build_mobile_alpr_package_export_path(plate_candidate, char_candidate)
            except Exception:
                return Path.cwd() / "alpr_package.alprmodel"

        details = tk.Text(
            root_picker,
            height=8,
            bg=picker_card,
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )
        details.grid(row=3, column=0, sticky="nsew", pady=(12, 8))
        details.configure(state=tk.DISABLED)

        def _candidate_line(label: str, candidate: dict) -> str:
            return (
                f"{label}: {candidate.get('model_label') or '-'} | "
                f"{_mobile_export_compact_yolo_label(candidate.get('model_version'))} | "
                f"{_candidate_metric_label(candidate)} | "
                f"łącznie epok: {_mobile_export_candidate_epochs_label(candidate)}"
            )

        def _refresh_package_preview(*_args) -> None:
            try:
                plate_candidate, char_candidate = _selected_package_candidates()
                selection_key = f"{plate_var.get()}\0{char_var.get()}"
                if selection_key != str(package_destination_state.get("selection") or ""):
                    package_destination_state["selection"] = selection_key
                    package_destination_state["last"] = ""
                lines = [
                    _candidate_line("MT", plate_candidate),
                    _candidate_line("MZ", char_candidate),
                    "",
                    "Pakiet zawiera manifest systemowy alpr.package.v1 oraz manifesty obu modeli.",
                    "Aplikacja mobilna dostanie parę MT+MZ, pipeline i metadane potrzebne do inferencji.",
                    "Po kliknięciu eksportu wybierzesz nazwę i miejsce zapisu pliku .alprmodel.",
                ]
                details.configure(state=tk.NORMAL)
                details.delete("1.0", tk.END)
                details.insert("1.0", "\n".join(lines))
                details.configure(state=tk.DISABLED)
                package_status_var.set("Gotowe do sprawdzenia wymagań i eksportu pakietu.")
            except Exception as exc:
                package_status_var.set(str(exc))

        for variable in (plate_var, char_var):
            try:
                variable.trace_add("write", _refresh_package_preview)
            except Exception:
                pass
        _refresh_package_preview()

        status_label = tk.Label(
            root_picker,
            textvariable=package_status_var,
            bg=picker_bg,
            fg=muted,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=690,
            font=("Segoe UI", 9),
        )
        status_label.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        actions_picker = tk.Frame(root_picker, bg=picker_bg)
        actions_picker.grid(row=5, column=0, sticky="ew")
        actions_picker.grid_columnconfigure(0, weight=1)

        def export_complete_package() -> None:
            try:
                plate_candidate, char_candidate = _selected_package_candidates()
                default_destination = _default_complete_package_destination(plate_candidate, char_candidate)
                preview_package_id_seed = f"ALPR-{plate_candidate.get('model_label')}-{char_candidate.get('model_label')}"
                preview_plate_request = _build_mobile_export_request_for_candidate(
                    plate_candidate,
                    destination=default_destination.parent / "_mt_single_for_bundle.alprmodel",
                    include_metadata=False,
                )
                preview_char_request = _build_mobile_export_request_for_candidate(
                    char_candidate,
                    destination=default_destination.parent / "_mz_single_for_bundle.alprmodel",
                    include_metadata=False,
                )
                preview_request = MobileAlprPackageRequest(
                    destination=default_destination,
                    plate_request=preview_plate_request,
                    character_request=preview_char_request,
                    package_id=self._safe_model_export_slug(preview_package_id_seed, fallback="ALPR-package")[:80],
                    name=f"ALPR MT+MZ | {plate_candidate.get('model_label') or 'MT'} + {char_candidate.get('model_label') or 'MZ'}",
                    version="1",
                    metadata={"source": "mobile_export_center", "purpose": "complete_alpr_mobile_detection_package"},
                )
                preview_problems = package_exporter.preflight(preview_request)
                if preview_problems:
                    text = "Eksport pakietu nie może jeszcze ruszyć:\n" + "\n".join(f"- {item}" for item in preview_problems)
                    package_status_var.set(text)
                    set_status(text, "warning")
                    show_preflight_problems_dialog(
                        preview_problems,
                        parent=picker,
                        title="Sprawdzenie gotowości pakietu ALPR MT+MZ",
                        context=(
                            "Najpierw sprawdzamy oba modele, formaty i biblioteki. "
                            "Jeśli brakuje zależności eksportu, zainstaluj je tutaj przed wyborem miejsca zapisu."
                        ),
                    )
                    return
                destination = _ask_mobile_package_destination(
                    default_destination,
                    parent=picker,
                    title="Eksportuj komplet MT+MZ jako pakiet ALPR",
                )
                if destination is None:
                    package_status_var.set("Eksport anulowany. Plik nie został zapisany.")
                    return
                package_destination_state["last"] = str(destination)
                package_id_seed = f"ALPR-{plate_candidate.get('model_label')}-{char_candidate.get('model_label')}"
                plate_request = _build_mobile_export_request_for_candidate(
                    plate_candidate,
                    destination=destination.parent / "_mt_single_for_bundle.alprmodel",
                    include_metadata=True,
                )
                char_request = _build_mobile_export_request_for_candidate(
                    char_candidate,
                    destination=destination.parent / "_mz_single_for_bundle.alprmodel",
                    include_metadata=True,
                )
                request = MobileAlprPackageRequest(
                    destination=destination,
                    plate_request=plate_request,
                    character_request=char_request,
                    package_id=self._safe_model_export_slug(package_id_seed, fallback="ALPR-package")[:80],
                    name=f"ALPR MT+MZ | {plate_candidate.get('model_label') or 'MT'} + {char_candidate.get('model_label') or 'MZ'}",
                    version="1",
                    metadata={
                        "source": "mobile_export_center",
                        "purpose": "complete_alpr_mobile_detection_package",
                        "plate_candidate_iid": str(plate_candidate.get("iid") or ""),
                        "character_candidate_iid": str(char_candidate.get("iid") or ""),
                    },
                )
                problems = package_exporter.preflight(request)
                if problems:
                    text = "Eksport pakietu nie może jeszcze ruszyć:\n" + "\n".join(f"- {item}" for item in problems)
                    package_status_var.set(text)
                    set_status(text, "warning")

                    def _refresh_package_after_install(success: bool) -> None:
                        if not success:
                            return
                        try:
                            refreshed = package_exporter.preflight(request)
                        except Exception as exc:
                            refreshed = [str(exc)]
                        if refreshed:
                            package_status_var.set(
                                "Wymagania zainstalowane, ale pakiet nadal wymaga korekty:\n"
                                + "\n".join(f"- {item}" for item in refreshed)
                            )
                        else:
                            package_status_var.set("Wymagania gotowe. Możesz ponownie uruchomić eksport pakietu MT+MZ.")

                    show_preflight_problems_dialog(
                        problems,
                        parent=picker,
                        title="Sprawdzenie gotowości pakietu ALPR MT+MZ",
                        context=(
                            "Sprawdzamy oba modele pakietu, wybrane formaty i zależności eksportu. "
                            "Jeśli brakuje bibliotek, możesz zainstalować je tutaj i ponowić eksport kompletu."
                        ),
                        after_install=_refresh_package_after_install,
                    )
                    return
            except Exception as exc:
                package_status_var.set(str(exc))
                return messagebox.showerror("Pakiet ALPR", str(exc), parent=picker)

            set_running(True)
            start_message = _mobile_export_verbose_progress_message(
                0.0,
                "Buduję pakiet ALPR MT+MZ do detekcji mobilnej.",
            )
            package_status_var.set(start_message)
            set_status(start_message, "info")
            progress_var.set(0.0)
            progress_percent_var.set("0%")

            def progress_cb(percent, message: str) -> None:
                def apply_update() -> None:
                    if percent is not None:
                        progress.stop()
                        progress.configure(mode="determinate")
                        progress_value = _mobile_export_progress_value(percent)
                        progress_var.set(progress_value)
                        progress_percent_var.set(_mobile_export_progress_label(progress_value))
                    else:
                        progress_percent_var.set("...")
                    verbose_message = _mobile_export_verbose_progress_message(percent, message)
                    package_status_var.set(verbose_message)
                    set_status(verbose_message, "info")

                try:
                    dialog.after(0, apply_update)
                except Exception:
                    pass

            def worker() -> None:
                try:
                    package_path = package_exporter.export(request, progress=progress_cb)

                    def done() -> None:
                        set_running(False)
                        progress_var.set(100.0)
                        progress_percent_var.set("100%")
                        package_status_var.set(f"Gotowe: {package_path}")
                        set_status(f"Kompletny pakiet ALPR gotowy: {package_path}", "success")
                        try:
                            self._append_train_log(f"[MOBILE EXPORT] Utworzono kompletny pakiet ALPR MT+MZ: {package_path}")
                        except Exception:
                            pass
                        messagebox.showinfo(
                            "Pakiet ALPR gotowy",
                            f"Utworzono kompletny pakiet MT+MZ:\n{package_path}",
                            parent=picker,
                        )

                    dialog.after(0, done)
                except Exception as exc:
                    logger.exception("Nie udało się zbudować kompletnego pakietu ALPR")
                    error_text = str(exc)

                    def failed() -> None:
                        set_running(False)
                        package_status_var.set(f"Błąd: {error_text}")
                        set_status(f"Nie udało się zbudować kompletnego pakietu ALPR: {error_text}", "error")
                        messagebox.showerror("Błąd pakietu ALPR", error_text, parent=picker)

                    try:
                        dialog.after(0, failed)
                    except Exception:
                        pass

            threading.Thread(target=worker, name="mobile-alpr-package-export", daemon=True).start()

        ttk.Button(actions_picker, text="Eksportuj komplet MT+MZ", command=export_complete_package).grid(
            row=0,
            column=1,
            sticky="e",
            padx=(0, 8),
            ipadx=10,
            ipady=3,
        )
        ttk.Button(actions_picker, text="Zamknij", command=picker.destroy).grid(row=0, column=2, sticky="e", ipadx=8, ipady=3)

    def on_candidate_select(_event=None) -> None:
        selected = candidate_tree.selection()
        if not selected:
            return
        candidate = candidate_by_iid.get(str(selected[0]))
        if candidate:
            apply_candidate(candidate)

    candidate_tree.bind("<<TreeviewSelect>>", on_candidate_select, add="+")
    candidate_tree.bind("<Button-1>", on_candidate_export_click, add="+")
    candidate_tree.bind("<ButtonRelease-1>", lambda _event: dialog.after_idle(on_candidate_select), add="+")
    candidate_tree.bind("<KeyRelease-Up>", lambda _event: dialog.after_idle(on_candidate_select), add="+")
    candidate_tree.bind("<KeyRelease-Down>", lambda _event: dialog.after_idle(on_candidate_select), add="+")

    export_button = ttk.Button(
        actions,
        text="Eksportuj zaznaczone",
        command=open_selected_mobile_export_executor,
        state=tk.DISABLED,
    )
    ttk.Button(
        actions,
        text="Raporty z telefonu",
        command=lambda: open_mobile_report_browser(self, parent=dialog),
    ).grid(row=0, column=0, sticky="w", padx=(0, 8), ipadx=8, ipady=2)
    export_button.grid(row=0, column=1, sticky="e", padx=(0, 7), ipadx=10, ipady=2)
    close_button = ttk.Button(actions, text="Zamknij", command=dialog.destroy)
    close_button.grid(row=0, column=2, sticky="e", ipadx=7, ipady=2)
    try:
        dependency_tool_shell.grid_remove()
    except Exception:
        pass
    _update_dependency_buttons()

    selected_iid = initial_iid or (str(candidates[0].get("iid") or "") if candidates else "")
    if selected_iid:
        if initial_iid and initial_iid in candidate_by_iid:
            try:
                marker = _mobile_export_target_marker(candidate_by_iid[initial_iid])
                if marker in set(export_marker_order):
                    export_selection_state[marker] = initial_iid
            except Exception:
                pass
        try:
            candidate_tree.selection_set(selected_iid)
            candidate_tree.focus(selected_iid)
            candidate_tree.see(selected_iid)
        except Exception:
            pass
    _update_export_selection_view()
    if selected_iid and selected_iid in candidate_by_iid:
        apply_candidate(candidate_by_iid.get(selected_iid), preflight_after=False)
    else:
        set_status("Brak lokalnych kandydatów. Dodaj model pojazdów z Ultralytics albo dodaj model z treningu.", "warning")
    _set_mobile_export_loader("Finalizuję widok eksportu...", 88.0)
    if perf_enabled:
        try:
            dialog.after(350, _mobile_export_log_widget_tree)
        except Exception:
            pass
        try:
            dialog.after(900, _start_mobile_export_event_loop_monitor)
        except Exception:
            pass

    def _show_mobile_export_dialog() -> None:
        _set_mobile_export_loader("Pokazuję gotowe okno...", 96.0, force_paint=True)
        try:
            dialog.update_idletasks()
        except Exception:
            pass
        try:
            dialog.deiconify()
        except Exception:
            pass
        try:
            dialog.update_idletasks()
        except Exception:
            pass
        try:
            dialog.lift(dialog_parent)
        except Exception:
            try:
                dialog.lift()
            except Exception:
                pass
        try:
            dialog.focus_force()
        except Exception:
            pass
        _set_mobile_export_loader("Gotowe.", 100.0, force_paint=True)
        try:
            dialog.after(120, _close_mobile_export_loader)
        except Exception:
            _close_mobile_export_loader()
        try:
            _mobile_export_perf_record(
                "mobile_export_window_show",
                0.0,
                mapped=int(bool(dialog.winfo_ismapped())),
                size=f"{int(dialog.winfo_width() or 0)}x{int(dialog.winfo_height() or 0)}",
            )
        except Exception:
            pass
        try:
            _install_mobile_export_win32_move_guard()
        except Exception:
            pass
        if windowing_system == "win32" and str(os.environ.get("AAT_MOBILE_EXPORT_TOPMOST_FLASH", "")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                dialog.attributes("-topmost", True)

                def _release_topmost() -> None:
                    try:
                        dialog.attributes("-topmost", False)
                    except Exception:
                        pass

                dialog.after(280, _release_topmost)
            except Exception:
                pass

    _show_mobile_export_dialog()

    return dialog


def _export_selected_run_model_to_mobile_package(self):
    run = self._selected_run()
    if run is None:
        return messagebox.showwarning(
            "Brak runu",
            "Najpierw wybierz run treningu z historii albo otwórz eksport mobilny z menu głównego.",
        )
    return _open_mobile_model_export_center(self, initial_run=run)
