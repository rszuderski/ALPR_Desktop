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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None
def _get_datasets_base_dir(self) -> Path:
    """Zwraca bazowy katalog datasetów dla aktywnego projektu albo globalny fallback."""
    if self._campaign_datasets_dir:
        return Path(self._campaign_datasets_dir)
    target = getattr(self, "_step4_dataset_mode", getattr(self, "_campaign_training_target", "char"))
    return Path(CONFIG.get_datasets_dir(target))

def _get_manual_plate_stage_dir(self) -> Path:
    if CAMPAIGN.get_active_project_name():
        base_dir = CAMPAIGN.get_staging_dir("plate_stage")
        if base_dir is not None:
            iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
            return Path(base_dir) / f"Iteracja_{iter_num:03d}"
    return Path(CONFIG.get_auto_annotations_dir("plate")) / "_manual_stage"

def _iter_dataset_search_roots(self, base_dir: Path | None = None) -> list[Path]:
    root = Path(base_dir or self._get_datasets_base_dir())
    candidates = [root, root / "plates", root / "chars", root / "vehicles"]
    unique: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen or not candidate.exists() or not candidate.is_dir():
            continue
        seen.add(resolved)
        unique.append(candidate)

    return unique

def _find_dataset_source_candidates(self, base_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    for search_root in self._iter_dataset_search_roots(base_dir):
        try:
            for path in search_root.iterdir():
                if not path.is_dir() or "_Split_" in path.name or not (path / "images").exists():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(path)
        except Exception:
            continue

    return candidates

def _find_ready_dataset_candidates(self, base_dir: Path | None = None) -> list[tuple[Path, str, float]]:
    candidates: list[tuple[Path, str, float]] = []
    seen: set[str] = set()

    for search_root in self._iter_dataset_search_roots(base_dir):
        try:
            for path in search_root.iterdir():
                if not path.is_dir():
                    continue

                yaml_path = path / "data.yaml"
                if not yaml_path.exists():
                    continue

                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                target = self._infer_dataset_target(str(path)) or "char"
                candidates.append((path, target, path.stat().st_mtime))
        except Exception:
            continue

    return candidates

def _get_free_dataset_variant_choices(self) -> list[dict]:
    selected_target = self._get_selected_training_target()
    candidates = self._find_ready_dataset_candidates(self._get_datasets_base_dir())
    variants: list[dict] = []
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    approved_plate_images = 0
    stored_plate_dataset = ""
    if campaign_active and selected_target == "plate":
        try:
            approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            approved_plate_images = int(approved_stats.get("images", 0) or 0)
        except Exception:
            approved_plate_images = 0
        try:
            stored = dict(CAMPAIGN.get_last_plate_training_source() or {})
            stored_raw = str(stored.get("dataset_path", "") or "").strip()
            stored_plate_dataset = str(Path(stored_raw).resolve()) if stored_raw else ""
        except Exception:
            stored_plate_dataset = ""

    for path, target, stamp in candidates:
        if target != selected_target:
            continue
        readiness = get_training_dataset_readiness(path, target=target)
        if not bool(readiness.get("ok", True)):
            continue
        counts = self._get_dataset_split_image_counts(path)
        total = int(counts.get("total", 0) or 0)
        if total <= 0:
            # PZ2 wybiera warianty splitu utworzone w PZ1. Katalog źródłowy
            # może mieć data.yaml, ale bez images/train|val|test nie jest
            # wariantem treningowym i nie powinien mieszać się z listą.
            continue
        if campaign_active and selected_target == "plate" and approved_plate_images > 0 and total != approved_plate_images:
            try:
                path_resolved = str(Path(path).resolve())
            except Exception:
                path_resolved = str(path)
            is_stored_campaign_variant = bool(stored_plate_dataset and path_resolved == stored_plate_dataset)
            is_current_manifest_variant = False
            try:
                manifest = self._load_plate_dataset_source_manifest(Path(path))
                manifest_project = str(manifest.get("project") or "").strip()
                manifest_iteration = int(manifest.get("iteration", 0) or 0)
                current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
                current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
                manifest_images = manifest.get("approved_set_images", None)
                if manifest_images is not None:
                    is_current_manifest_variant = (
                        manifest_project == current_project
                        and manifest_iteration == current_iteration
                        and int(manifest_images or 0) == approved_plate_images
                    )
                else:
                    # Starsze warianty augmentowane nie mają jeszcze liczników ApprovedSet.
                    is_current_manifest_variant = (
                        manifest_project == current_project
                        and manifest_iteration == current_iteration
                    )
            except Exception:
                is_current_manifest_variant = False
            if not is_stored_campaign_variant and not is_current_manifest_variant:
                continue
        try:
            label_path = self._format_workspace_relative_path(path)
        except Exception:
            label_path = str(path)
        display_ref = build_dataset_display_ref(path, target_hint=target, counts=counts)
        label = display_ref.combo_label
        variants.append(
            {
                "label": label,
                "path": str(path),
                "display_path": label_path,
                "dataset_id": display_ref.id,
                "dataset_label": display_ref.detail_label,
                "stamp": float(stamp or 0),
            }
        )

    variants.sort(key=lambda item: float(item.get("stamp", 0) or 0), reverse=True)
    return variants

def _refresh_dataset_variant_choices(self):
    combo = getattr(self, "dataset_variant_combo", None)
    if combo is None:
        return

    variants = self._get_free_dataset_variant_choices()
    try:
        current = str(getattr(self, "dataset_var", tk.StringVar()).get() or "").strip()
    except Exception:
        current = ""
    if current:
        try:
            current_root = Path(current)
            if current_root.is_file() and current_root.name.lower() == "data.yaml":
                current_root = current_root.parent
            current_yaml = current_root / "data.yaml"
            current_key = str(current_root.resolve())
            known_keys = {
                self._normalize_dataset_variant_root(item.get("path"))
                for item in list(variants or [])
                if str(item.get("path") or "").strip()
            }
            current_target = self._infer_dataset_target(str(current_root)) or self._get_selected_training_target()
            if (
                current_yaml.exists()
                and current_key not in known_keys
                and current_target == self._get_selected_training_target()
                and bool(get_training_dataset_readiness(current_root, target=current_target).get("ok", True))
            ):
                counts = self._get_dataset_split_image_counts(current_root)
                display_ref = build_dataset_display_ref(current_root, target_hint=current_target, counts=counts)
                try:
                    stamp = float(current_root.stat().st_mtime)
                except Exception:
                    stamp = time.time()
                try:
                    label_path = self._format_workspace_relative_path(current_root)
                except Exception:
                    label_path = str(current_root)
                variants.insert(
                    0,
                    {
                        "label": display_ref.combo_label,
                        "path": str(current_root),
                        "display_path": label_path,
                        "dataset_id": display_ref.id,
                        "dataset_label": display_ref.detail_label,
                        "stamp": stamp,
                    },
                )
        except Exception:
            pass
    self._dataset_variant_choices = variants
    labels = [str(item.get("label") or "") for item in variants]

    try:
        combo.configure(values=labels)
    except Exception:
        pass

    try:
        combo.configure(state=("readonly" if labels else tk.DISABLED))
    except Exception:
        pass

    self._sync_dataset_variant_selection()

def _normalize_dataset_variant_root(self, value: str | Path | None) -> str:
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

def _is_free_training_dataset_variant_selected(self, dataset_value: str | Path | None = None) -> bool:
    if CAMPAIGN.get_active_project_name():
        return True

    current_root = self._normalize_dataset_variant_root(
        dataset_value if dataset_value is not None else getattr(self, "dataset_var", tk.StringVar()).get()
    )
    if not current_root:
        return False

    for item in list(self._get_free_dataset_variant_choices() or []):
        item_root = self._normalize_dataset_variant_root(item.get("path"))
        if item_root and item_root == current_root:
            return True
    return False

def _sync_dataset_variant_selection(self):
    var = getattr(self, "dataset_variant_var", None)
    if var is None:
        return

    current = str(getattr(self, "dataset_var", tk.StringVar()).get() or "").strip()
    current_resolved = self._normalize_dataset_variant_root(current)

    selected_label = ""
    for item in list(getattr(self, "_dataset_variant_choices", []) or []):
        item_path = str(item.get("path") or "")
        item_resolved = self._normalize_dataset_variant_root(item_path)
        if current_resolved and item_resolved == current_resolved:
            selected_label = str(item.get("label") or "")
            break

    try:
        var.set(selected_label)
    except Exception:
        pass

def _on_dataset_variant_selected(self, event=None):
    selected = str(getattr(self, "dataset_variant_var", tk.StringVar()).get() or "").strip()
    if not selected:
        return

    for item in list(getattr(self, "_dataset_variant_choices", []) or []):
        if str(item.get("label") or "") != selected:
            continue
        path = str(item.get("path") or "").strip()
        if path:
            try:
                accepted = self._accept_training_input_context(
                    source="pz2_variant",
                    target=self._get_selected_training_target(),
                    dataset_path=path,
                    select_training=False,
                )
            except Exception:
                accepted = False
            if not accepted:
                try:
                    self.dataset_var.set(path)
                except Exception:
                    pass
                try:
                    self._last_training_source = self._build_step4_dataset_training_source(
                        path,
                        target=self._get_selected_training_target(),
                        provenance="pz2_variant",
                    )
                except Exception:
                    pass
        return

def _get_dataset_split_image_counts(self, dataset_path: Path | str | None) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "total": 0}
    if dataset_path is None:
        return counts

    try:
        root = Path(dataset_path)
    except Exception:
        return counts

    if root.is_file():
        root = root.parent

    images_root = root / "images"
    if not images_root.exists() or not images_root.is_dir():
        return counts

    started_at = time.perf_counter()
    signature_parts: list[str] = []
    split_dirs: list[tuple[str, Path, bool]] = []
    for split_name in ("train", "val", "test"):
        split_dir = images_root / split_name
        exists = False
        mtime_ns = 0
        try:
            exists = split_dir.exists() and split_dir.is_dir()
            if exists:
                mtime_ns = int(split_dir.stat().st_mtime_ns)
        except Exception:
            exists = False
            mtime_ns = 0
        split_dirs.append((split_name, split_dir, exists))
        signature_parts.append(f"{split_name}:{mtime_ns}")

    try:
        cache_root = str(root.resolve())
    except Exception:
        cache_root = str(root)
    cache_key = f"{cache_root}|{'|'.join(signature_parts)}"
    cache = getattr(self, "_dataset_split_image_counts_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            self._dataset_split_image_counts_cache = cache
        except Exception:
            pass
    cached_counts = cache.get(cache_key) if isinstance(cache, dict) else None
    if isinstance(cached_counts, dict):
        return dict(cached_counts)

    total = 0
    scan_failed = False
    for split_name, split_dir, exists in split_dirs:
        if not exists:
            continue
        try:
            # DirEntry reuses directory-listing metadata on Windows instead of
            # issuing a separate stat for every image in every candidate split.
            with os.scandir(split_dir) as entries:
                split_count = sum(
                    1 for entry in entries
                    if os.path.splitext(entry.name)[1].lower() in CONFIG.IMAGE_EXTENSIONS and entry.is_file()
                )
        except Exception:
            split_count = 0
            scan_failed = True
        counts[split_name] = int(split_count)
        total += int(split_count)

    counts["total"] = int(total)
    try:
        if len(cache) > 256:
            cache.clear()
        if not scan_failed:
            cache[cache_key] = dict(counts)
    except Exception:
        pass
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    if elapsed_ms >= 250:
        try:
            logger.info(
                f"[Z4/PZ1 PERF] split_counts total={elapsed_ms}ms "
                f"images={int(counts.get('total', 0) or 0)} path={root}"
            )
        except Exception:
            pass
    return counts

def _format_training_source_provenance(provenance: str | None) -> str:
    raw = str(provenance or "").strip()
    key = raw.lower()
    labels = {
        "pz1": "Utworzony w PZ1",
        "pz2_variant": "Wybrany wariant",
        "campaign_ready_dataset": "Dataset kampanii",
        "active": "Aktywny split",
    }
    return labels.get(key, raw or "Aktywny split")

def _build_step4_dataset_training_source(
    self,
    dataset_path: str | Path | None,
    *,
    target: str = "",
    provenance: str = "Aktywny split",
    source_stage: str = "Z4/PZ1",
) -> TrainingSource:
    dataset_text = str(dataset_path or "").strip()
    normalized_target = str(target or getattr(self, "_step4_dataset_mode", "char") or "char").strip().lower()
    try:
        normalized_target = CONFIG.normalize_task_target(normalized_target)
    except Exception:
        normalized_target = normalized_target if normalized_target in {"plate", "char"} else "char"

    if not dataset_text:
        return TrainingSource(
            target=normalized_target,
            kind="yolo_dataset",
            provenance=self._format_training_source_provenance(provenance),
            source_stage=str(source_stage or ""),
            message="Brak wybranego datasetu.",
        )

    dataset_dir = dataset_text
    yaml_path = ""
    validated = False
    stats = TrainingSourceStats()
    message = ""

    try:
        root = Path(dataset_text)
        if root.is_file() and root.name.lower() == "data.yaml":
            yaml_candidate = root
            root = root.parent
        else:
            yaml_candidate = root / "data.yaml"
        dataset_dir = str(root)
        yaml_path = str(yaml_candidate)
        validated = bool(yaml_candidate.exists())
        if validated:
            stats = TrainingSourceStats.from_mapping(self._get_dataset_split_image_counts(root))
            message = "Dataset ma plik data.yaml i może być użyty jako wariant treningowy."
            try:
                inferred_target = str(self._infer_dataset_target(str(root)) or "").strip().lower()
            except Exception:
                inferred_target = ""
            if inferred_target in {"plate", "char"}:
                normalized_target = inferred_target
            readiness = get_training_dataset_readiness(root, target=normalized_target)
            if not bool(readiness.get("ok", True)):
                validated = False
                message = str(readiness.get("message") or "Dataset nie jest gotowy do treningu.")
        else:
            message = "Dataset wymaga pliku data.yaml."
    except Exception:
        dataset_dir = dataset_text
        yaml_path = ""
        validated = False
        message = "Nie udało się odczytać źródła datasetu."

    return TrainingSource(
        target=normalized_target,
        kind="yolo_dataset",
        dataset_dir=dataset_dir,
        yaml_path=yaml_path,
        validated=validated,
        stats=stats,
        provenance=self._format_training_source_provenance(provenance),
        source_stage=str(source_stage or ""),
        message=message,
    )

def _resolve_step4_dataset_summary_source(self) -> TrainingSource:
    dataset_path = ""
    provenance = "Aktywny split"
    target = str(getattr(self, "_step4_dataset_mode", "char") or "char").strip().lower()
    try:
        target = CONFIG.normalize_task_target(target)
    except Exception:
        target = target if target in {"plate", "char"} else "char"

    try:
        current_tab = str(self.main_nb.select())
        dataset_tab = str(getattr(self, "tab_dataset", ""))
    except Exception:
        current_tab = ""
        dataset_tab = ""
    pending_source = getattr(self, "_pending_step4_input_training_source", None)
    if (
        dataset_tab
        and current_tab == dataset_tab
        and isinstance(pending_source, TrainingSource)
        and str(pending_source.display_path() or "").strip()
    ):
        pending_target = str(getattr(pending_source, "target", "") or "").strip().lower()
        try:
            pending_target = CONFIG.normalize_task_target(pending_target)
        except Exception:
            pending_target = pending_target if pending_target in {"plate", "char"} else ""
        if pending_target == target:
            return pending_source

    # W PZ1 trybu swobodnego lewy kafel ma osobne źródła dla tablic i znaków.
    # Nie wolno więc pokazywać w tabeli podsumowania źródła przejętego z drugiego
    # typu datasetu, nawet jeśli zostało wcześniej poprawnie wybrane.
    if dataset_tab and current_tab == dataset_tab:
        try:
            creator_ready_mode = target == "plate" and callable(getattr(self, "_get_creator_source_mode", None)) and self._get_creator_source_mode() == "ready"
        except Exception:
            creator_ready_mode = False
        if not creator_ready_mode:
            resolver_name = "_resolve_dataset_creator_inputs" if target == "plate" else "_resolve_dataset_split_inputs"
            try:
                resolver = getattr(self, resolver_name, None)
                info = dict(resolver() or {}) if callable(resolver) else {}
            except Exception:
                info = {}
            active_source = info.get("training_source")
            if isinstance(active_source, TrainingSource):
                active_target = str(getattr(active_source, "target", "") or "").strip().lower()
                try:
                    active_target = CONFIG.normalize_task_target(active_target)
                except Exception:
                    active_target = active_target if active_target in {"plate", "char"} else ""
                if active_target == target and str(active_source.display_path() or "").strip():
                    return active_source
            return self._build_step4_dataset_training_source(
                "",
                target=target,
                provenance=provenance,
            )

    try:
        dataset_path = str(getattr(self, "dataset_var", tk.StringVar()).get() or "").strip()
    except Exception:
        dataset_path = ""

    cached_source = getattr(self, "_last_training_source", None)
    if isinstance(cached_source, TrainingSource) and dataset_path:
        cached_path = str(cached_source.dataset_dir or cached_source.yaml_path or "").strip()
        try:
            cached_root = Path(cached_path)
            dataset_root = Path(dataset_path)
            if dataset_root.is_file() and dataset_root.name.lower() == "data.yaml":
                dataset_root = dataset_root.parent
            if cached_root.is_file() and cached_root.name.lower() == "data.yaml":
                cached_root = cached_root.parent
            if cached_root and dataset_root and cached_root.resolve() == dataset_root.resolve():
                return cached_source
        except Exception:
            if cached_path == dataset_path:
                return cached_source

    source = self._build_step4_dataset_training_source(
        dataset_path,
        target=target,
        provenance=provenance,
    )
    try:
        self._last_training_source = source
    except Exception:
        pass
    return source

def _refresh_step4_dataset_summary_table(self):
    frame = getattr(self, "step4_dataset_summary_frame", None)
    rows = getattr(self, "_step4_dataset_summary_rows", None)
    if frame is None or not rows:
        return

    if CAMPAIGN.get_active_project_name():
        try:
            frame.pack_forget()
        except Exception:
            pass
        return

    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    field = palette.get("field", panel_alt)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#4ec9b0")
    warning = palette.get("warning", "#d7ba7d")
    danger = palette.get("danger", "#f48771")
    header_bg = panel_alt
    header_fg = success

    source = self._resolve_step4_dataset_summary_source()
    dataset_path = str(source.dataset_dir or source.yaml_path or "").strip()
    target = str(source.target or getattr(self, "_step4_dataset_mode", "char") or "char").strip().lower()
    source_label = str(source.provenance or "Aktywny split")
    target = CONFIG.normalize_task_target(target)
    target_label = self._format_training_target_label(target)
    display_path = "Brak wybranego datasetu"
    dataset_id_text = "-"
    data_yaml_text = "-"
    counts = source.stats.split_counts()
    ready = bool(source.validated)

    if not dataset_path:
        try:
            frame.pack_forget()
        except Exception:
            pass
        return

    try:
        if str(frame.winfo_manager()) != "pack":
            before_widget = getattr(self, "ds_mode_scroll_host", None)
            if before_widget is not None:
                frame.pack(fill=tk.X, pady=(6, 0), before=before_widget)
            else:
                frame.pack(fill=tk.X, pady=(6, 0))
    except Exception:
        pass

    if dataset_path:
        try:
            root = Path(dataset_path)
            if root.is_file() and root.name.lower() == "data.yaml":
                root = root.parent
            data_yaml = Path(source.yaml_path) if str(source.yaml_path or "").strip() else root / "data.yaml"
            ready = bool(source.validated or data_yaml.exists())
            data_yaml_text = "jest" if ready else "brak"
            if int(counts.get("total", 0) or 0) <= 0:
                counts = self._get_dataset_split_image_counts(root)
            display_ref = build_dataset_display_ref(root, target_hint=target, counts=counts)
            dataset_id_text = display_ref.id
            display_path = display_ref.name or self._format_workspace_relative_path(root)
        except Exception:
            display_path = str(dataset_path)
            data_yaml_text = "brak"

    status_text = "Split gotowy" if ready else "Brak wybranego datasetu"
    status_fg = success if ready else warning
    if source.kind in {"annotation_xml_images", "yolo_dataset"} and source.provenance == "Źródło PZ1":
        status_text = "Źródło gotowe" if ready else "Źródło niegotowe"
        status_fg = success if ready else warning
    if dataset_path and not ready:
        status_text = "Wymaga data.yaml"
        status_fg = danger
    total_count = int(counts.get("total", 0) or 0)
    if total_count > 0:
        status_text = f"{status_text} | razem {total_count}"

    values = {
        "status": status_text,
        "dataset_id": dataset_id_text,
        "target": f"{target_label} | {source_label}" if source_label else target_label,
        "source": source_label if dataset_path else "-",
        "path": display_path,
        "yaml": data_yaml_text,
        "total": str(total_count),
    }
    self._step4_dataset_summary_split_counts = dict(counts)

    try:
        frame.configure(bg=panel, highlightbackground=border, highlightcolor=border)
        self.step4_dataset_summary_title_lbl.configure(bg=panel, fg=fg)
        self.step4_dataset_summary_grid.configure(bg=panel)
    except Exception:
        pass

    for index, (key, widgets) in enumerate(rows.items()):
        label_widget, value_widget = widgets
        bg = field if index % 2 == 0 else panel_alt
        value_fg = status_fg if key == "status" else fg
        if key in {"source", "path"} and not dataset_path:
            value_fg = muted
        try:
            label_widget.configure(bg=bg, fg=muted, highlightbackground=border, highlightcolor=border)
            if key == "split":
                value_widget.configure(bg=bg, highlightbackground=border, highlightcolor=border)
                self._draw_step4_dataset_split_bar(value_widget)
                continue
            value_widget.configure(
                bg=bg,
                fg=value_fg,
                highlightbackground=border,
                highlightcolor=border,
                wraplength=620 if key == "path" else 260,
            )
        except Exception:
            pass

    header_widgets = getattr(self, "_step4_dataset_summary_header_widgets", ())
    for widget in header_widgets:
        try:
            widget.configure(bg=header_bg, fg=header_fg, highlightbackground=border, highlightcolor=border)
        except Exception:
            pass

    for key, value in values.items():
        if key in rows:
            try:
                rows[key][1].configure(text=value)
            except Exception:
                pass

def _schedule_step4_dataset_summary_refresh(self):
    try:
        self._refresh_step4_dataset_summary_table()
    except Exception:
        pass

def _draw_step4_dataset_split_bar(self, canvas=None):
    if canvas is None:
        try:
            canvas = self._step4_dataset_summary_rows["split"][1]
        except Exception:
            return

    try:
        counts = dict(getattr(self, "_step4_dataset_summary_split_counts", {}) or {})
    except Exception:
        counts = {}

    palette = getattr(self.app, "palette", {})
    panel_alt = palette.get("panel_alt", "#2d2d30")
    field = palette.get("field", panel_alt)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#4ec9b0")
    warning = palette.get("warning", "#d7ba7d")
    accent = palette.get("accent", "#569cd6")

    try:
        width = max(180, int(canvas.winfo_width() or 0))
        height = max(20, int(canvas.winfo_height() or 0))
    except Exception:
        width, height = 360, 22

    try:
        canvas.delete("all")
        canvas.configure(bg=field, highlightbackground=border, highlightcolor=border)
    except Exception:
        return

    train = int(counts.get("train", 0) or 0)
    val = int(counts.get("val", 0) or 0)
    test = int(counts.get("test", 0) or 0)
    total = int(counts.get("total", 0) or (train + val + test))

    if total <= 0:
        canvas.create_text(
            10,
            height // 2,
            text="brak danych splitu",
            fill=muted,
            anchor="w",
            font=("Segoe UI", 8),
        )
        return

    if train <= 0 and val <= 0 and test <= 0:
        canvas.create_text(
            10,
            height // 2,
            text=f"do splitu: {total}",
            fill=fg,
            anchor="w",
            font=("Segoe UI", 8),
        )
        return

    usable = max(1, width - 2)
    min_segment = max(18, min(46, usable // 10))
    segments = [
        ("train", train, success),
        ("val", val, warning),
        ("test", test, accent),
    ]
    widths = {
        name: (float(count) / float(total)) * usable if count > 0 else 0.0
        for name, count, _color in segments
    }

    for name, count, _color in segments:
        if count > 0 and widths[name] < min_segment:
            widths[name] = float(min_segment)

    overflow = sum(widths.values()) - usable
    while overflow > 0.5:
        adjustable = [
            name
            for name, count, _color in segments
            if count > 0 and widths.get(name, 0.0) > min_segment
        ]
        if not adjustable:
            scale = usable / max(1.0, sum(widths.values()))
            for name in widths:
                widths[name] *= scale
            break
        largest = max(adjustable, key=lambda item: widths.get(item, 0.0))
        cut = min(overflow, widths[largest] - min_segment)
        widths[largest] -= cut
        overflow -= cut

    x = 1.0
    bar_top = 3
    bar_bottom = max(bar_top + 12, height - 3)
    for name, count, color in segments:
        segment_width = widths.get(name, 0.0)
        if count <= 0 or segment_width <= 0:
            continue
        x2 = min(float(width - 1), x + segment_width)
        canvas.create_rectangle(x, bar_top, x2, bar_bottom, fill=color, outline=field)
        text = f"{name} {count}"
        text_fill = "#111111" if name in {"train", "val"} else fg
        if x2 - x >= 58:
            canvas.create_text(
                (x + x2) / 2,
                (bar_top + bar_bottom) / 2,
                text=text,
                fill=text_fill,
                anchor="center",
                font=("Segoe UI", 7),
            )
        else:
            canvas.create_text(
                x + 3,
                (bar_top + bar_bottom) / 2,
                text=str(count),
                fill=text_fill,
                anchor="w",
                font=("Segoe UI", 7),
            )
        x = x2

def _median_int(values: list[int]) -> int:
    cleaned = sorted(int(value) for value in (values or []) if int(value) > 0)
    if not cleaned:
        return 0
    middle = len(cleaned) // 2
    if len(cleaned) % 2 == 1:
        return int(cleaned[middle])
    return int(round((cleaned[middle - 1] + cleaned[middle]) / 2.0))

def _nearest_training_imgsz(value: int, *, minimum: int = 384, maximum: int = 1280) -> int:
    allowed = [256, 320, 384, 416, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280]
    minimum = max(32, int(minimum or 32))
    maximum = max(minimum, int(maximum or minimum))
    target = max(minimum, min(maximum, int(value or minimum)))
    candidates = [candidate for candidate in allowed if minimum <= candidate <= maximum]
    if not candidates:
        snapped = int(round(target / 32.0) * 32)
        return max(minimum, min(maximum, max(32, snapped)))
    return min(candidates, key=lambda candidate: (abs(candidate - target), candidate))

def _parse_training_model_params_millions(raw_value) -> float:
    text = str(raw_value or "").strip().lower().replace(",", ".")
    if not text:
        return 0.0
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except Exception:
        return 0.0

def _normalize_training_model_catalog_key(self, model_value, catalog: dict[str, dict] | None = None) -> str:
    catalog = dict(catalog or {})
    if not catalog:
        return ""

    raw_text = str(model_value or "").strip()
    if not raw_text:
        return ""

    lowered = raw_text.lower()
    try:
        stem = Path(raw_text).stem.lower()
    except Exception:
        stem = lowered

    for key, meta in catalog.items():
        key_text = str(key or "").strip().lower()
        file_text = str((meta or {}).get("file", "") or "").strip().lower()
        try:
            file_stem = Path(file_text).stem.lower()
        except Exception:
            file_stem = file_text

        if lowered in {key_text, file_text}:
            return str(key)
        if stem in {key_text, file_stem}:
            return str(key)

    return ""

def _infer_training_model_bucket(
    model_label: str = "",
    *,
    params_m: float = 0.0,
    file_size_mb: float = 0.0,
) -> str:
    label = str(model_label or "").strip().lower()
    try:
        stem = Path(label).stem.lower()
    except Exception:
        stem = label

    match = re.search(r"([nsmxl])(?:-pose)?$", stem)
    if match:
        return str(match.group(1))

    if params_m > 0:
        if params_m <= 4.5:
            return "n"
        if params_m <= 12.0:
            return "s"
        if params_m <= 24.0:
            return "m"
        if params_m <= 40.0:
            return "l"
        return "x"

    if file_size_mb > 0:
        if file_size_mb <= 8.0:
            return "n"
        if file_size_mb <= 20.0:
            return "s"
        if file_size_mb <= 40.0:
            return "m"
        if file_size_mb <= 80.0:
            return "l"
        return "x"

    return "s"

def _infer_training_model_version_size_from_text(raw_text: str | Path | None) -> tuple[str, str]:
    text = str(raw_text or "").strip().lower()
    if not text:
        return "", ""
    try:
        text = f"{text} {Path(text).stem.lower()}"
    except Exception:
        pass

    match = re.search(r"yolo(?:v)?(8|11|26)([nsmxl])", text)
    if not match:
        return "", ""
    return str(match.group(1) or "").strip(), str(match.group(2) or "").strip().lower()

def _format_training_model_version_label(raw_version: str | None) -> str:
    version = str(raw_version or "").strip().lower()
    if version.startswith("v"):
        version = version[1:]
    if not version:
        return ""
    if version == "8":
        return "YOLOv8"
    return f"YOLO{version}"

def _resolve_selected_training_base_model_profile(self) -> dict:
    target = self._get_selected_training_target()
    catalog = AVAILABLE_POSE_MODELS if target == "plate" else AVAILABLE_DETECT_MODELS
    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    custom_model_path = self._resolve_selected_training_base_model_path()
    inspection_path, inspection_info = self._resolve_selected_training_base_model_info()

    resolved_key = ""
    params_m = 0.0
    file_size_mb = 0.0
    source_label = base_key

    if not self._is_custom_base_model_key(base_key):
        resolved_key = self._normalize_training_model_catalog_key(base_key, catalog)
        source_label = resolved_key or base_key
    else:
        source_run = self._resolve_training_run_from_model_path(custom_model_path)
        source_base_model = str(getattr(source_run, "base_model", "") or "").strip() if source_run is not None else ""
        if source_base_model:
            resolved_key = self._normalize_training_model_catalog_key(source_base_model, catalog)
            source_label = resolved_key or source_base_model
        elif custom_model_path is not None:
            resolved_key = self._normalize_training_model_catalog_key(custom_model_path.name, catalog)
            source_label = resolved_key or custom_model_path.name

        if custom_model_path is not None:
            try:
                file_size_mb = round(float(custom_model_path.stat().st_size) / (1024.0 ** 2), 2)
            except Exception:
                file_size_mb = 0.0

    catalog_meta = catalog.get(resolved_key, {}) if resolved_key in catalog else {}
    catalog_params_text = str((catalog_meta or {}).get("params") or "").strip()
    catalog_version = str((catalog_meta or {}).get("version") or "").strip()
    detected_label = format_yolo_model_identity(inspection_info)
    detected_source_label = str(inspection_info.get("source_architecture_label") or "").strip()
    detected_scale = str(inspection_info.get("model_scale") or inspection_info.get("yolo_size") or "").strip().lower()
    if self._is_custom_base_model_key(base_key):
        if detected_source_label:
            source_label = detected_source_label
        elif detected_label:
            source_label = detected_label

    if resolved_key in catalog:
        params_m = self._parse_training_model_params_millions(catalog_params_text)

    model_version = str(inspection_info.get("yolo_version") or "").strip()
    model_size = detected_scale if detected_scale in {"n", "s", "m", "l", "x"} else ""
    identity_candidates = [
        detected_label,
        detected_source_label,
        source_label,
        resolved_key,
        base_key,
        custom_model_path.name if custom_model_path is not None else "",
        str(inspection_path.name if inspection_path is not None else ""),
    ]
    if not model_version and catalog_version:
        model_version = catalog_version
    for candidate in identity_candidates:
        inferred_version, inferred_size = self._infer_training_model_version_size_from_text(candidate)
        if not model_version and inferred_version:
            model_version = inferred_version
        if not model_size and inferred_size:
            model_size = inferred_size
        if model_version and model_size:
            break

    if model_size in {"n", "s", "m", "l", "x"}:
        bucket = model_size
    else:
        bucket = self._infer_training_model_bucket(
            source_label,
            params_m=params_m,
            file_size_mb=file_size_mb,
        )
        model_size = bucket

    return {
        "target": target,
        "base_key": base_key,
        "catalog_key": resolved_key,
        "label": source_label,
        "bucket": bucket,
        "params_m": params_m,
        "file_size_mb": file_size_mb,
        "custom": bool(self._is_custom_base_model_key(base_key)),
        "detected_label": detected_label,
        "detected_source_label": detected_source_label,
        "detected_scale": detected_scale,
        "model_version": model_version,
        "model_version_label": self._format_training_model_version_label(model_version),
        "model_size": model_size,
        "model_size_label": self._format_training_model_size_label(model_size),
        "params_text": catalog_params_text,
        "inspection_path": str(inspection_path) if inspection_path is not None else "",
    }

def _schedule_training_dataset_profile(self, yaml_path: Path, cache_key: str) -> None:
    if getattr(self, "_training_dataset_profile_job", False):
        return
    self._training_dataset_profile_job = True
    requested_path = str(yaml_path.resolve())

    def finish(profile, completed_key):
        self._training_dataset_profile_job = False
        current_yaml = self._resolve_training_dataset_yaml_path()
        current_path = str(current_yaml.resolve()) if current_yaml is not None else ""
        if current_path == requested_path and completed_key == cache_key:
            self._training_dataset_profile_cache_key = cache_key
            self._training_dataset_profile_cache = dict(profile)
        # A changed selection discards the old result and schedules the current one.
        for name in ("_refresh_training_dataset_quality_summary", "_refresh_training_recommendation_table",
                     "_refresh_training_execution_summary"):
            callback = getattr(self, name, None)
            if callable(callback):
                callback()

    def worker():
        from types import SimpleNamespace

        # No widget, Tk variable, campaign manager or live tab is passed to the reader.
        reader = SimpleNamespace(_training_dataset_profile_sync=True, _median_int=_median_int)
        reader._get_dataset_split_image_counts = lambda root: _get_dataset_split_image_counts(reader, root)
        try:
            profile = _get_training_dataset_profile(reader, yaml_path)
            completed_key = getattr(reader, "_training_dataset_profile_cache_key", cache_key)
        except Exception as exc:
            profile = {"error": str(exc)}
            completed_key = cache_key
        self._ui(lambda: finish(profile, completed_key))

    try:
        threading.Thread(target=worker, daemon=True, name="z4-dataset-profile").start()
    except Exception as exc:
        finish({"error": str(exc)}, cache_key)


def _get_training_dataset_profile(self, dataset_yaml_path: Path | None = None) -> dict:
    result = {
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_images": 0,
        "train_label_files": 0,
        "val_label_files": 0,
        "test_label_files": 0,
        "total_label_files": 0,
        "train_objects": 0,
        "val_objects": 0,
        "test_objects": 0,
        "total_objects": 0,
        "sampled_images": 0,
        "median_width": 0,
        "median_height": 0,
        "median_long_edge": 0,
        "max_long_edge": 0,
        "created_at": "",
        "created_at_timestamp": 0,
    }

    yaml_path = dataset_yaml_path or self._resolve_training_dataset_yaml_path()
    if yaml_path is None or not yaml_path.exists():
        return result

    dataset_root = yaml_path.parent
    created_timestamp = 0.0
    created_text = ""
    try:
        created_timestamp = float(dataset_root.stat().st_ctime or 0)
    except Exception:
        created_timestamp = 0.0
    if created_timestamp <= 0:
        try:
            created_timestamp = float(yaml_path.stat().st_ctime or yaml_path.stat().st_mtime or 0)
        except Exception:
            created_timestamp = 0.0
    if created_timestamp > 0:
        try:
            created_text = datetime.datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d %H:%M")
        except Exception:
            created_text = ""
    result["created_at"] = created_text
    result["created_at_timestamp"] = int(created_timestamp or 0)

    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    test_dir = dataset_root / "images" / "test"
    train_labels_dir = dataset_root / "labels" / "train"
    val_labels_dir = dataset_root / "labels" / "val"
    test_labels_dir = dataset_root / "labels" / "test"

    try:
        cache_root = str(dataset_root.resolve())
    except Exception:
        cache_root = str(dataset_root)

    try:
        yaml_mtime = int(yaml_path.stat().st_mtime_ns)
    except Exception:
        yaml_mtime = 0
    try:
        train_mtime = int(train_dir.stat().st_mtime_ns) if train_dir.exists() else 0
    except Exception:
        train_mtime = 0
    try:
        val_mtime = int(val_dir.stat().st_mtime_ns) if val_dir.exists() else 0
    except Exception:
        val_mtime = 0
    try:
        test_mtime = int(test_dir.stat().st_mtime_ns) if test_dir.exists() else 0
    except Exception:
        test_mtime = 0
    label_mtimes = []
    for label_dir in (train_labels_dir, val_labels_dir, test_labels_dir):
        try:
            label_mtimes.append(int(label_dir.stat().st_mtime_ns) if label_dir.exists() else 0)
        except Exception:
            label_mtimes.append(0)

    cache_key = (
        f"{cache_root}|{yaml_mtime}|{train_mtime}|{val_mtime}|{test_mtime}|"
        + "|".join(str(value) for value in label_mtimes)
    )
    cached_key = str(getattr(self, "_training_dataset_profile_cache_key", "") or "")
    cached_profile = getattr(self, "_training_dataset_profile_cache", None)
    if cache_key == cached_key and isinstance(cached_profile, dict):
        return dict(cached_profile)

    if not bool(getattr(self, "_training_dataset_profile_sync", False)):
        _schedule_training_dataset_profile(self, yaml_path, cache_key)
        if getattr(self, "_training_dataset_profile_cache_key", None) == cache_key:
            return dict(self._training_dataset_profile_cache)
        return {**result, "pending": True}

    counts = self._get_dataset_split_image_counts(dataset_root)
    result["train_images"] = int(counts.get("train", 0) or 0)
    result["val_images"] = int(counts.get("val", 0) or 0)
    result["test_images"] = int(counts.get("test", 0) or 0)
    result["total_images"] = int(counts.get("total", 0) or 0)

    for split_name, label_dir in (
        ("train", train_labels_dir),
        ("val", val_labels_dir),
        ("test", test_labels_dir),
    ):
        label_files = 0
        object_count = 0
        if label_dir.exists() and label_dir.is_dir():
            try:
                with os.scandir(label_dir) as entries:
                    paths = [Path(entry.path) for entry in entries
                             if entry.name.lower().endswith(".txt") and entry.is_file()]
            except Exception:
                paths = []
            label_files = len(paths)
            for label_path in paths:
                try:
                    with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
                        object_count += sum(1 for line in handle if str(line or "").strip())
                except Exception:
                    continue
        result[f"{split_name}_label_files"] = int(label_files)
        result[f"{split_name}_objects"] = int(object_count)
        result["total_label_files"] += int(label_files)
        result["total_objects"] += int(object_count)

    sample_paths: list[Path] = []
    for split_dir in (train_dir, val_dir, test_dir):
        if len(sample_paths) >= 24:
            break
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        try:
            sample_paths.extend(list(get_image_files(split_dir))[: max(0, 24 - len(sample_paths))])
        except Exception:
            continue

    widths: list[int] = []
    heights: list[int] = []
    long_edges: list[int] = []
    if PIL_AVAILABLE:
        for image_path in sample_paths[:24]:
            try:
                with Image.open(image_path) as image_obj:
                    width, height = image_obj.size
            except Exception:
                continue
            width = int(width or 0)
            height = int(height or 0)
            if width <= 0 or height <= 0:
                continue
            widths.append(width)
            heights.append(height)
            long_edges.append(max(width, height))

    result["sampled_images"] = len(long_edges)
    result["median_width"] = self._median_int(widths)
    result["median_height"] = self._median_int(heights)
    result["median_long_edge"] = self._median_int(long_edges)
    result["max_long_edge"] = max(long_edges) if long_edges else 0

    self._training_dataset_profile_cache_key = cache_key
    self._training_dataset_profile_cache = dict(result)
    return dict(result)

def _get_campaign_plate_builder_source(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}
    if str(self.get_campaign_training_target() or "").strip().lower() != "plate":
        return {}

    try:
        annotation_tab = self.app.tabs.get("annotation")
    except Exception:
        annotation_tab = None

    if annotation_tab is None or not hasattr(annotation_tab, "_build_campaign_plate_approved_export_source"):
        return {}

    try:
        source = annotation_tab._build_campaign_plate_approved_export_source()
    except Exception:
        source = {}

    return dict(source or {}) if isinstance(source, dict) else {}

def _load_step4_training_ui_state(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}
    try:
        entry = dict(CAMPAIGN.get_iteration_state() or {})
    except Exception:
        return {}
    payload = entry.get("step4_ui_state", {})
    return payload if isinstance(payload, dict) else {}

def _save_step4_training_ui_state(self, payload: dict) -> None:
    if not CAMPAIGN.get_active_project_name():
        return
    try:
        CAMPAIGN.upsert_iteration_state(
            iteration_num=CAMPAIGN.get_current_iteration_num(),
            updates={"step4_ui_state": dict(payload or {})},
        )
    except Exception as e:
        logger.debug(f"Nie udało się zapisać stanu UI treningu Z4 do rejestru iteracji: {e}")

def _get_preferred_plate_pose_base_model(self) -> str:
    for preferred in ("yolo26s-pose", "yolo11s-pose", "yolov8s-pose"):
        if preferred in self._get_base_model_choices_for_mode("plate"):
            return preferred
    return self._get_default_base_model_for_mode("plate")

def _resolve_saved_step4_training_model_selection(self, target: str | None = None) -> dict:
    normalized_target = str(target or self.get_campaign_training_target() or "").strip().lower()
    if normalized_target not in ("char", "plate"):
        return {}
    payload = self._load_step4_training_ui_state()
    models = payload.get("models", {}) if isinstance(payload, dict) else {}
    if not isinstance(models, dict):
        return {}
    entry = models.get(normalized_target, {})
    return dict(entry or {}) if isinstance(entry, dict) else {}

def _apply_saved_step4_training_model_selection(self, target: str | None = None) -> bool:
    normalized_target = str(target or self.get_campaign_training_target() or "").strip().lower()
    if normalized_target not in ("char", "plate"):
        return False

    saved = self._resolve_saved_step4_training_model_selection(normalized_target)
    base_key = str(saved.get("base_model_key", "") or "").strip()
    custom_path = str(saved.get("base_custom_path", "") or "").strip()
    if not base_key:
        return False

    if self._is_custom_base_model_key(base_key):
        if not custom_path or not Path(custom_path).exists():
            return False
        try:
            source_run = self._resolve_training_run_from_model_path(Path(custom_path))
        except Exception:
            source_run = None
        if source_run is not None:
            status_value = str(getattr(source_run, "status", "") or "").strip().lower()
            if status_value and status_value != TrainingStatus.COMPLETED.value:
                return False
        self.base_model_var.set(self._get_custom_base_model_label())
        self.base_custom_var.set(custom_path)
        return True

    choices = self._get_base_model_choices_for_mode(normalized_target)
    if base_key not in choices:
        return False
    self.base_model_var.set(base_key)
    self.base_custom_var.set("")
    return True

def _remember_current_step4_training_model_selection(self) -> None:
    if not CAMPAIGN.get_active_project_name():
        return
    if bool(getattr(self, "_step4_suppress_base_model_state_save", False)):
        return

    target = str(self.get_campaign_training_target() or "").strip().lower()
    if target not in ("char", "plate"):
        return

    base_key = str(getattr(self, "base_model_var", tk.StringVar()).get() or "").strip()
    custom_value = str(getattr(self, "base_custom_var", tk.StringVar()).get() or "").strip()
    payload = self._load_step4_training_ui_state()
    if not isinstance(payload, dict):
        payload = {}
    models = payload.get("models", {})
    if not isinstance(models, dict):
        models = {}
    models[target] = {
        "base_model_key": base_key,
        "base_custom_path": custom_value if self._is_custom_base_model_key(base_key) else "",
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    payload["models"] = models
    self._save_step4_training_ui_state(payload)

def _resolve_campaign_plate_ready_dataset(self, datasets_dir: Path | None = None) -> dict:
    result = {
        "path": None,
        "counts": {"train": 0, "val": 0, "test": 0, "total": 0},
        "stale": False,
    }

    if not CAMPAIGN.get_active_project_name():
        return result

    try:
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}
    approved_images = int(approved_stats.get("images", 0) or 0)
    current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0

    candidate_paths: list[Path] = []
    seen: set[str] = set()

    stored = dict(CAMPAIGN.get_last_plate_training_source() or {})
    stored_dataset_path = str(stored.get("dataset_path", "") or "").strip()
    stored_dataset_resolved = ""
    if stored_dataset_path:
        try:
            stored_path = Path(stored_dataset_path)
        except Exception:
            stored_path = None
        if stored_path is not None and stored_path.exists():
            key = str(stored_path.resolve())
            stored_dataset_resolved = key
            seen.add(key)
            candidate_paths.append(stored_path)

    if datasets_dir is not None and datasets_dir.exists():
        try:
            ready_candidates = self._find_ready_dataset_candidates(datasets_dir)
        except Exception:
            ready_candidates = []
        ready_candidates.sort(key=lambda rec: rec[2], reverse=True)
        for path, target, _stamp in ready_candidates:
            if target != "plate":
                continue
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            candidate_paths.append(path)

    stale_path = None
    stale_counts = None

    def _accept_ready_dataset(path: Path, counts: dict) -> dict:
        result["path"] = path
        result["counts"] = counts
        try:
            resolver = getattr(self, "_resolve_plate_training_source_from_dataset", None)
            source_info = resolver(path) if callable(resolver) else {"dataset_path": str(path)}
            CAMPAIGN.set_last_plate_training_source(
                dataset_path=str(source_info.get("dataset_path", str(path)) or str(path)),
                source_run_path=str(source_info.get("source_run_path", "") or ""),
                source_xml_path=str(source_info.get("source_xml_path", "") or ""),
            )
        except Exception:
            pass
        return result

    for path in candidate_paths:
        counts = self._get_dataset_split_image_counts(path)
        total_images = int(counts.get("total", 0) or 0)
        if total_images <= 0:
            continue
        try:
            path_resolved = str(Path(path).resolve())
        except Exception:
            path_resolved = str(path)
        is_stored_campaign_variant = bool(stored_dataset_resolved and path_resolved == stored_dataset_resolved)
        manifest_matches_current_approved_set = False
        manifest_matches_current_context_without_counts = False
        try:
            manifest = self._load_plate_dataset_source_manifest(Path(path))
            manifest_project = str(manifest.get("project") or "").strip()
            manifest_iteration = int(manifest.get("iteration", 0) or 0)
            manifest_images = manifest.get("approved_set_images", None)
            manifest_context_ok = (
                bool(manifest_project)
                and manifest_project == current_project
                and manifest_iteration == current_iteration
            )
            if manifest_images is not None:
                manifest_matches_current_approved_set = (
                    manifest_context_ok
                    and int(manifest_images or 0) == approved_images
                )
            else:
                # Kompatybilność wsteczna: starsze warianty augmentowane kopiowały manifest
                # bez liczników ApprovedSet, ale nadal mają projekt i iterację źródłową.
                manifest_matches_current_context_without_counts = manifest_context_ok
        except Exception:
            pass
        if is_stored_campaign_variant and (
            manifest_matches_current_approved_set
            or manifest_matches_current_context_without_counts
            or (approved_images > 0 and total_images == approved_images)
        ):
            return _accept_ready_dataset(path, counts)
        if manifest_matches_current_approved_set:
            return _accept_ready_dataset(path, counts)
        if manifest_matches_current_context_without_counts and total_images >= approved_images:
            return _accept_ready_dataset(path, counts)
        if approved_images > 0 and total_images == approved_images:
            return _accept_ready_dataset(path, counts)
        if stale_path is None:
            stale_path = path
            stale_counts = counts

    if stale_path is not None and approved_images > 0:
        result["path"] = stale_path
        result["counts"] = stale_counts or result["counts"]
        result["stale"] = True

    return result
