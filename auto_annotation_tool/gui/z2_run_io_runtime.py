#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 run import/export and approval-state helpers extracted from tab_annotation.py."""

import copy
import csv
import json
from collections import deque
import os
import re
import tkinter as tk
import tkinter.font as tkfont
import xml.etree.ElementTree as ET
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import datetime
import threading
import logging
import math
import shutil
import time
import queue
import numpy as np

import cv2
from PIL import Image
from .inertial_scroll import InertialScrollController
from .zoomable_canvas import ZoomableCanvas
from .section_header_label import SectionHeaderLabel
from . import z2_workflow_methods
from . import z2_canvas_overlays
from . import z2_canvas_interaction
from . import z2_preview_editor
from . import z2_preview_state
from . import z2_miniflow_runtime
from . import z2_model_runtime
from . import z2_session_runtime
from . import z2_context_runtime
from .z2_main_widgets import create_annotation_widgets
from .z2_auto_scope_modal import prompt_plate_auto_scope_choice
from .z2_canvas_metrics_ui import (
    render_preview_metrics_grab_handle as z2_render_preview_metrics_grab_handle,
    render_preview_metrics_table as z2_render_preview_metrics_table,
)
from .z2_overlay_dock_ui import (
    get_preview_overlay_dock_theme as z2_get_preview_overlay_dock_theme,
    render_preview_campaign_gate_overlay as z2_render_preview_campaign_gate_overlay,
    render_preview_image_status_overlay as z2_render_preview_image_status_overlay,
    render_preview_overlay_dock as z2_render_preview_overlay_dock,
)
from .z2_legend_assets import (
    clear_preview_legend_image_cache as z2_clear_preview_legend_image_cache,
    get_preview_legend_compass_photo as z2_get_preview_legend_compass_photo,
    get_preview_legend_group_shell_photo as z2_get_preview_legend_group_shell_photo,
    get_preview_legend_interaction_marker_photo as z2_get_preview_legend_interaction_marker_photo,
    get_preview_legend_keycap_photo as z2_get_preview_legend_keycap_photo,
    get_preview_legend_shell_photo as z2_get_preview_legend_shell_photo,
    get_preview_legend_text_color as z2_get_preview_legend_text_color,
    hex_to_rgba as z2_hex_to_rgba,
    legend_color_is_light as z2_legend_color_is_light,
    normalize_preview_legend_interaction as z2_normalize_preview_legend_interaction,
    trim_preview_legend_image_cache as z2_trim_preview_legend_image_cache,
)
from .z2_legend_ui import (
    build_preview_controls_context_rows as z2_build_preview_controls_context_rows,
    build_preview_legend_sections as z2_build_preview_legend_sections,
    get_preview_controls_legend_target_height as z2_get_preview_controls_legend_target_height,
    get_preview_controls_legend_target_width as z2_get_preview_controls_legend_target_width,
    get_preview_legend_theme as z2_get_preview_legend_theme,
    preview_controls_row_is_multi_plate_badge as z2_preview_controls_row_is_multi_plate_badge,
    refresh_preview_controls_legend as z2_refresh_preview_controls_legend,
)
from .z2_model_quality_ui import (
    auto_model_metric_tone as z2_auto_model_metric_tone,
    describe_auto_model_quality as z2_describe_auto_model_quality,
    format_model_quality_epoch as z2_format_model_quality_epoch,
    format_model_quality_number as z2_format_model_quality_number,
    get_model_identity_caption as z2_get_model_identity_caption,
    model_quality_float as z2_model_quality_float,
    model_quality_value as z2_model_quality_value,
    render_auto_annotation_model_quality_table as z2_render_auto_annotation_model_quality_table,
    shorten_model_quality_text as z2_shorten_model_quality_text,
)
from .z2_actions import Z2ActionContext, build_z2_primary_actions, build_z2_secondary_actions
from .z2_campaign_flow import (
    apply_campaign_step2_workflow_preset,
    build_z2_cta_state_campaign,
    build_z2_layout_state_campaign,
    build_z2_left_panel_copy_payload_campaign,
    open_campaign_step2_entry as dispatch_open_campaign_step2_entry,
    open_existing_run_for_campaign_review,
    prepare_campaign_workflow_runtime,
)
from .z2_free_mode_flow import (
    AUTO_REVIEW_FOLLOWUP_TEXT,
    AUTO_REVIEW_FOLLOWUP_TITLE,
    MANUAL_REVIEW_FOLLOWUP_TEXT,
    MANUAL_REVIEW_FOLLOWUP_TITLE,
    build_z2_cta_state_free_mode,
    build_z2_layout_state_free_mode,
    build_z2_left_panel_copy_payload_free_mode,
    get_manual_review_history_display_entries as dispatch_get_manual_review_history_display_entries,
    jump_to_export_section as dispatch_jump_to_export_section,
    open_existing_run_for_manual_review as dispatch_open_existing_run_for_manual_review,
    prepare_free_mode_workflow_runtime,
    refresh_manual_review_history_ui as dispatch_refresh_manual_review_history_ui,
    remember_manual_review_run as dispatch_remember_manual_review_run,
    select_free_mode_route,
    set_manual_entry_mode as dispatch_set_manual_entry_mode,
)
from .z2_flow_models import Z2CopyPayload, Z2LeftPanelCopyContext
from .z2_shared_ui import (
    apply_z2_workflow_left_layout as dispatch_apply_z2_workflow_left_layout,
    apply_z2_workflow_cta_ui as dispatch_apply_z2_workflow_cta_ui,
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel

from ..config import CONFIG, logger, YOLO_AVAILABLE, AVAILABLE_DETECT_MODELS, SESSION
from ..annotators.runtime_factory import validate_pt_model_path_for_runtime
from ..icons import IconManager
from ..exporters import CVATExporter, ReportGenerator
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..project_cache import PROJECT_CACHE
from ..training import DatasetCreator
from ..utils import cleanup_gpu_memory, count_images_in_directory, format_duration, get_image_files, get_image_size
from ..data_models import Detection, AnnotationStatus, ImageAnnotation, AnnotationReport
from ..rectification.polygon_validator import PolygonValidator
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .canvas_progress_overlay import CanvasProgressOverlay

NAV_BUTTON_WIDTH = 18


def _path_value_to_path_safe(path_value) -> Path | None:
    if path_value is None:
        return None

    value = path_value
    try:
        if hasattr(value, "get") and callable(value.get):
            value = value.get()
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _path_value_to_path_safe(item)
            if candidate is not None:
                return candidate
        return None

    try:
        text = os.fsdecode(os.fspath(value)).strip()
    except Exception:
        try:
            text = str(value or "").strip()
        except Exception:
            text = ""

    if not text:
        return None
    try:
        return Path(text)
    except Exception:
        return None


def _set_plate_export_status(self, text: str, tone: str = "muted"):
    self._set_inline_label_state(
        self.plate_export_status_lbl,
        text=text,
        tone=tone,
        emphasis=False
    )
    try:
        self._refresh_bound_label_wraplength(self.plate_export_status_lbl)
    except Exception:
        pass


def _set_post_annotation_hint(self, text: str = "", tone: str = "muted"):
    label = getattr(self, "post_annotation_hint_lbl", None)
    if label is None:
        return

    has_text = bool(str(text or "").strip())
    try:
        if has_text:
            if not str(label.winfo_manager()):
                label.pack(fill=tk.X, pady=(6, 0))
        else:
            if str(label.winfo_manager()):
                label.pack_forget()
    except Exception:
        pass

    self._set_inline_label_state(
        label,
        text=str(text or ""),
        tone=tone,
        emphasis=False
    )


def _resolve_existing_run_dir(path_value) -> Path | None:
    candidate = _path_value_to_path_safe(path_value)
    if candidate is None:
        return None

    try:
        if candidate.exists() and candidate.is_dir():
            return candidate
    except Exception:
        return None
    return None


def _resolve_existing_dir(path_value) -> Path | None:
    candidate = _path_value_to_path_safe(path_value)
    if candidate is None:
        return None

    try:
        if candidate.exists() and candidate.is_dir():
            return candidate
    except Exception:
        return None
    return None


def _normalize_preview_metric_threshold(value, *, default: float = 0.0) -> float:
    try:
        normalized = float(value)
    except Exception:
        normalized = float(default)
    return max(0.0, min(1.0, normalized))


def _normalize_annotation_image_name(value) -> str:
    raw_value = str(value or "").strip().replace("\\", "/")
    if not raw_value:
        return ""
    try:
        return str(Path(raw_value).name or "").strip().lower()
    except Exception:
        return str(raw_value.rsplit("/", 1)[-1] or "").strip().lower()


def _get_cached_input_dir_image_names(self, image_dir) -> set[str]:
    safe_dir = self._resolve_existing_dir(image_dir)
    if safe_dir is None:
        return set()

    try:
        resolved = str(safe_dir.resolve()).strip().lower()
        stat = safe_dir.stat()
        cache_key = (
            "dir",
            resolved,
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )
    except Exception:
        cache_key = ("dir", str(safe_dir).strip().lower(), 0, 0)

    cache = getattr(self, "_z2_input_image_name_set_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._z2_input_image_name_set_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        return set(cached)

    names: set[str] = set()
    try:
        for item in safe_dir.iterdir():
            try:
                if item.is_file() and item.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                    normalized = _normalize_annotation_image_name(item.name)
                    if normalized:
                        names.add(normalized)
            except Exception:
                continue
    except Exception:
        names = set()

    cache.clear()
    cache[cache_key] = set(names)
    return names


def _get_cached_run_xml_image_names(self, run_dir) -> set[str]:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return set()

    xml_path = safe_run_dir / "annotations.xml"
    try:
        stat = xml_path.stat()
        cache_key = (
            "xml",
            str(xml_path.resolve()).strip().lower(),
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )
    except Exception:
        cache_key = ("xml", str(xml_path).strip().lower(), 0, 0)

    cache = getattr(self, "_z2_run_xml_image_name_set_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._z2_run_xml_image_name_set_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        cache[cache_key] = cache.pop(cache_key)
        return set(cached)

    names: set[str] = set()
    try:
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "image":
                normalized = _normalize_annotation_image_name(elem.get("name"))
                if normalized:
                    names.add(normalized)
                elem.clear()
    except Exception:
        return set()
    try:
        final_stat = xml_path.stat()
        if (final_stat.st_mtime_ns, final_stat.st_size) != cache_key[2:]:
            return set()
    except OSError:
        return set()

    # Keep multiple XMLs: a one-entry cache reparsed every run on each scan.
    # Bound both the number of runs and the retained names (large source pools).
    for old_key in list(cache):
        if old_key[1] == cache_key[1]:
            del cache[old_key]
    if len(names) <= 200_000:
        cache[cache_key] = frozenset(names)
        retained = sum(len(value) for value in cache.values())
        while len(cache) > 256 or retained > 200_000:
            retained -= len(cache.pop(next(iter(cache))))
    return names


def _annotation_run_matches_expected_image_names(self, run_dir, expected_input_dir) -> bool:
    expected_names = _get_cached_input_dir_image_names(self, expected_input_dir)
    if not expected_names:
        return False
    run_names = _get_cached_run_xml_image_names(self, run_dir)
    if not expected_names or not run_names:
        return False

    overlap = len(expected_names & run_names)
    if overlap <= 0:
        return False

    expected_coverage = overlap / max(1, len(expected_names))
    run_coverage = overlap / max(1, len(run_names))
    return bool(expected_coverage >= 0.98 and run_coverage >= 0.85)


def _annotation_run_matches_expected_input_dir(self, run_dir, expected_input_dir) -> bool:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    expected_dir = self._resolve_existing_dir(expected_input_dir) or self._path_value_to_path(expected_input_dir)
    if safe_run_dir is None or expected_dir is None:
        return True

    try:
        manifest = self._load_annotation_run_manifest(safe_run_dir)
    except Exception:
        return True

    declared_inputs = [
        str(manifest.get("source_input_dir") or "").strip(),
        str(manifest.get("imported_source_input_dir") or "").strip(),
        str(manifest.get("input_dir") or "").strip(),
    ]
    declared_inputs = [value for value in declared_inputs if value]
    if not declared_inputs:
        return True

    for raw_value in declared_inputs:
        try:
            if self._paths_equivalent(raw_value, expected_dir):
                return True
        except Exception:
            continue
    try:
        if _annotation_run_matches_expected_image_names(self, safe_run_dir, expected_dir):
            return True
    except Exception:
        pass
    return False


def _repair_campaign_step2_generated_state_for_input(self, expected_input_dir) -> bool:
    if self._is_free_mode_session_context():
        return False

    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return False
        if str(CAMPAIGN.get_iteration_target() or "").strip().lower() != "plate":
            return False

        current_step2_run = self._resolve_safe_annotation_run_dir(
            CAMPAIGN.get_step2_staging_run(),
            require_xml=True,
        )
        if current_step2_run is None:
            return False

        if self._annotation_run_matches_expected_input_dir(current_step2_run, expected_input_dir):
            return False

        self._append_z2_trace(
            "repair-step2-stale-run",
            f"run={current_step2_run} expected={expected_input_dir}",
        )
        CAMPAIGN.reset_step2()
        return True
    except Exception:
        return False


def _allocate_annotation_run_dir(self, base_out_dir: Path, *, suffix: str = "") -> Path:
    base_out_dir = self._coerce_annotation_output_dir(base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_text = f"_{suffix}" if str(suffix or "").strip() else ""
    counter = 1
    while True:
        run_dir = base_out_dir / f"run_{counter:03d}_{timestamp}{suffix_text}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            PROJECT_CACHE.invalidate_annotation_run_dirs(base_out_dir)
            return run_dir
        counter += 1


def _get_manual_review_import_initial_dir(self) -> str:
    try:
        base_dir = self._resolve_existing_dir(self._get_annotation_output_base_dir())
    except Exception:
        base_dir = None
    if base_dir is not None:
        return str(base_dir)

    candidates = [
        self._get_selected_manual_review_history_run_dir(),
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.input_dir_var.get() or "").strip(),
        Path.cwd(),
    ]

    for candidate in candidates:
        existing_run = self._resolve_existing_run_dir(candidate)
        if existing_run is not None:
            parent_dir = self._resolve_existing_dir(existing_run.parent)
            if parent_dir is not None:
                return str(parent_dir)

        existing_dir = self._resolve_existing_dir(candidate)
        if existing_dir is not None:
            return str(existing_dir)

    try:
        return str(Path.home())
    except Exception:
        return str(Path.cwd())


def _build_safe_imported_image_relative_path(
    path_value,
    *,
    index: int,
    used_paths: set[str],
) -> Path:
    try:
        raw_path = Path(str(path_value or "").strip())
    except Exception:
        raw_path = Path("")

    clean_parts: list[str] = []
    anchor = str(getattr(raw_path, "anchor", "") or "").strip()
    for part in getattr(raw_path, "parts", ()):
        normalized = str(part or "").strip()
        if not normalized or normalized in {".", "..", "/", "\\"}:
            continue
        if anchor and normalized == anchor:
            continue
        normalized = normalized.replace(":", "")
        if not normalized:
            continue
        clean_parts.append(normalized)

    if not clean_parts:
        clean_parts = [f"image_{int(index) + 1:05d}.jpg"]

    candidate = Path(*clean_parts)
    if candidate.is_absolute():
        candidate = Path(candidate.name or f"image_{int(index) + 1:05d}.jpg")

    key = str(candidate).lower()
    if key in used_paths:
        stem = candidate.stem or f"image_{int(index) + 1:05d}"
        suffix = candidate.suffix
        duplicate_counter = 1
        while True:
            deduped = candidate.with_name(f"{stem}_{duplicate_counter:02d}{suffix}")
            key = str(deduped).lower()
            if key not in used_paths:
                candidate = deduped
                break
            duplicate_counter += 1

    used_paths.add(str(candidate).lower())
    return candidate


def _get_external_run_image_roots(
    self,
    source_run_dir: Path,
    source_manifest: dict | None = None,
    compatible_images_dir: Path | str | None = None,
) -> list[Path]:
    manifest = source_manifest if isinstance(source_manifest, dict) else {}
    candidates = [
        compatible_images_dir,
        str(manifest.get("input_dir") or "").strip(),
        source_run_dir / "images",
        source_run_dir,
    ]
    return self._dedupe_paths(candidates)


def _get_external_run_images_initial_dir(self, source_run_dir: Path, source_manifest: dict | None = None) -> str:
    for candidate in self._get_external_run_image_roots(source_run_dir, source_manifest):
        try:
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        except Exception:
            continue

    try:
        return str(source_run_dir.parent if source_run_dir.parent.exists() else source_run_dir)
    except Exception:
        return self._get_manual_review_import_initial_dir()


def _get_external_run_expected_image_count(self, source_run_dir: Path | str | None) -> int:
    source_run_dir = self._resolve_existing_run_dir(source_run_dir)
    if source_run_dir is None:
        return 0

    source_xml_path = source_run_dir / "annotations.xml"
    if not source_xml_path.exists():
        return 0

    try:
        return len(self._parse_cvat_preview_annotations(source_xml_path))
    except Exception:
        return 0


def _resolve_external_run_source_images(
    self,
    annotations: list[ImageAnnotation],
    image_roots: list[Path],
) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    resolved_images: list[tuple[str, Path, Path]] = []
    missing_images: list[str] = []
    basename_index: dict[str, Path | None] = {}

    for idx, ann in enumerate(annotations):
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            missing_images.append("<brak_nazwy>")
            continue

        raw_path = Path(filename)
        source_image_path = None
        try:
            if raw_path.is_absolute() and raw_path.exists() and raw_path.is_file():
                source_image_path = raw_path
        except Exception:
            source_image_path = None

        if source_image_path is None:
            for image_root in image_roots:
                candidate_path = image_root / raw_path
                try:
                    if candidate_path.exists() and candidate_path.is_file():
                        source_image_path = candidate_path
                        break
                except Exception:
                    continue

        if source_image_path is None:
            basename = raw_path.name.lower()
            indexed_candidate = basename_index.get(basename, "__missing__")
            if indexed_candidate == "__missing__":
                matches: list[Path] = []
                for image_root in image_roots:
                    try:
                        if not image_root.exists() or not image_root.is_dir():
                            continue
                    except Exception:
                        continue
                    try:
                        matches.extend(path for path in image_root.rglob(raw_path.name) if path.is_file())
                    except Exception:
                        continue
                    if len(matches) > 1:
                        break

                if len(matches) == 1:
                    basename_index[basename] = matches[0]
                else:
                    basename_index[basename] = None

                indexed_candidate = basename_index.get(basename)

            if indexed_candidate not in {None, "__missing__"}:
                source_image_path = indexed_candidate

        if source_image_path is None:
            missing_images.append(filename)
            continue

        resolved_images.append((filename, raw_path, source_image_path))

    return resolved_images, missing_images


def _import_external_annotation_run_to_workspace(self, *args, **kwargs):
    return z2_workflow_methods._import_external_annotation_run_to_workspace(self, *args, **kwargs)


def _import_or_open_manual_review_run_from_dialog(self):
    initialdir = self._get_manual_review_import_initial_dir()
    path = filedialog.askdirectory(initialdir=initialdir)
    if not path:
        return

    selected_run_dir = self._resolve_existing_run_dir(path)
    if selected_run_dir is None or not (selected_run_dir / "annotations.xml").exists():
        messagebox.showerror(
            "Nieprawidlowy run anotacji",
            "Wskaz run Z2 zawierajacy plik annotations.xml i zgodne obrazy.\n\n"
            f"{self._get_annotation_run_definition_text()}\n\n"
            f"Domyslny katalog runow Z2: {self._get_annotation_run_storage_display_path()}",
        )
        return

    safe_run_dir = self._resolve_safe_annotation_run_dir(selected_run_dir, require_xml=True)
    if safe_run_dir is not None:
        self._open_existing_run_for_manual_review(
            run_dir=safe_run_dir,
            allow_fallback=False,
            show_dialog=True,
            entry_mode="import",
        )
        return

    imported_run_dir, error_message, needs_image_dir = self._import_external_annotation_run_to_workspace(selected_run_dir)
    compatible_images_dir = None
    if imported_run_dir is None and needs_image_dir:
        source_manifest = self._load_annotation_run_manifest(selected_run_dir)
        expected_image_count = self._get_external_run_expected_image_count(selected_run_dir)
        expected_count_hint = ""
        if expected_image_count > 0:
            expected_count_hint = (
                f"\n\nProgram oczekuje katalogu z okolo {expected_image_count} zgodnymi obrazami "
                "wynikajacymi z annotations.xml."
            )
        messagebox.showinfo(
            "Wskaz folder obrazow",
            "Wybrano zewnetrzny run Z2, ale nie ma on kompletu zgodnych obrazow "
            "w standardowych lokalizacjach.\n\n"
            f"{self._get_annotation_run_definition_text()}\n\n"
            "Wskaz folder z kompatybilnymi zdjeciami. Program sprawdzi zgodnosc z annotations.xml "
            f"i skopiuje poprawny zestaw do lokalnego runu w workspace Z2.{expected_count_hint}\n\n"
            f"Domyslny katalog runow Z2: {self._get_annotation_run_storage_display_path()}",
        )
        compatible_images_dir = filedialog.askdirectory(
            initialdir=self._get_external_run_images_initial_dir(selected_run_dir, source_manifest)
        )
        if not compatible_images_dir:
            return
        imported_run_dir, error_message, _needs_image_dir = self._import_external_annotation_run_to_workspace(
            selected_run_dir,
            compatible_images_dir=compatible_images_dir,
        )
    if imported_run_dir is None:
        messagebox.showerror(
            "Blad importu runu anotacji",
            error_message or "Nie udalo sie zaimportowac wskazanego runu anotacji do workspace Z2.",
        )
        return

    if not self._open_existing_run_for_manual_review(
        run_dir=imported_run_dir,
        allow_fallback=False,
        show_dialog=False,
        entry_mode="import",
    ):
        return

    messagebox.showinfo(
        "Run anotacji zaimportowany",
        (
            "Zaimportowano run Z2 do workspace i otwarto go do recznej korekty.\n\n"
            f"{self._get_annotation_run_definition_text()}\n\n"
            f"Run zrodlowy: {selected_run_dir}\n"
            + (
                f"Folder zgodnych obrazow: {compatible_images_dir}\n"
                if compatible_images_dir
                else ""
            )
            + f"Lokalna kopia runu: {imported_run_dir}\n"
            + f"Domyslny katalog runow Z2: {self._get_annotation_run_storage_display_path()}"
        ),
    )


def _count_plate_annotations(self, annotations: list[ImageAnnotation] | None = None) -> tuple[int, int]:
    images_with_plates = 0
    total_plates = 0

    for ann in list(annotations if annotations is not None else (self.current_annotations or [])):
        plate_count = len(self._get_plate_detections(ann))
        if plate_count > 0:
            images_with_plates += 1
            total_plates += plate_count

    return images_with_plates, total_plates


def _get_plate_annotation_exports_base_dir(self) -> Path:
    return Path(CONFIG.DIR_2_AUTO_ANN) / "plate_annotation_exports"


def _get_plate_annotation_export_annotations(self, run_dir: Path) -> list[ImageAnnotation]:
    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(run_dir, current_run_dir)
    ):
        source_annotations = list(self.current_annotations or [])
    else:
        source_annotations = self._parse_cvat_preview_annotations(run_dir / "annotations.xml")

    export_annotations = []
    for ann in list(source_annotations or []):
        try:
            if len(self._get_plate_detections(ann)) <= 0:
                continue
        except Exception:
            continue
        export_annotations.append(ann)
    return export_annotations


def _get_plate_annotation_export_approved_filenames(self, run_dir: Path) -> set[str]:
    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(run_dir, current_run_dir)
    ):
        approved_names = self._get_preview_approved_filenames()
    else:
        approved_names = self._load_annotation_run_approved_filenames(run_dir)

    return {
        str(name or "").strip().replace("\\", "/").lower()
        for name in set(approved_names or set())
        if str(name or "").strip()
    }


def _plate_annotation_filename_matches_lookup(self, filename: str, lookup: set[str]) -> bool:
    if not lookup:
        return False
    safe_name = str(filename or "").strip().replace("\\", "/").lower()
    if not safe_name:
        return False
    if safe_name in lookup:
        return True
    basename = Path(safe_name).name.lower()
    return bool(basename and basename in lookup)


def _filter_plate_annotation_export_annotations_by_approved(
    self,
    run_dir: Path,
    annotations: list[ImageAnnotation],
) -> list[ImageAnnotation]:
    approved_lookup = self._get_plate_annotation_export_approved_filenames(run_dir)
    if not approved_lookup:
        return []
    return [
        ann
        for ann in list(annotations or [])
        if self._plate_annotation_filename_matches_lookup(
            str(getattr(ann, "filename", "") or ""),
            approved_lookup,
        )
    ]


def _get_plate_annotation_package_export_state(self, run_dir: Path | None = None) -> dict:
    result = {
        "ok": False,
        "run_dir": None,
        "images_with_plates": 0,
        "total_plates": 0,
        "message": "Brak gotowego runu anotacji tablic.",
    }
    if run_dir is None:
        run_dir = self._get_preferred_annotation_run_dir(require_xml=True)
    else:
        run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if run_dir is None:
        return result

    result["run_dir"] = run_dir
    try:
        annotations = self._get_plate_annotation_export_annotations(run_dir)
    except Exception as e:
        result["message"] = f"Nie można odczytać annotations.xml: {e}"
        return result

    images_with_plates, total_plates = self._count_plate_annotations(annotations)
    result.update(
        images_with_plates=int(images_with_plates),
        total_plates=int(total_plates),
    )
    if total_plates <= 0:
        result["message"] = "Eksport anotacji jest dostępny po zapisaniu co najmniej jednej tablicy w annotations.xml."
        return result

    result["ok"] = True
    result["message"] = (
        f"Gotowe do eksportu anotacji XML: {images_with_plates} obrazów / {total_plates} tablic."
    )
    return result


def _set_plate_annotation_export_button_state(self) -> None:
    button = getattr(self, "export_plate_annotations_btn", None)
    if button is None:
        return
    try:
        state = self._get_plate_annotation_package_export_state()
        button.configure(state=(tk.NORMAL if state.get("ok") and not self.is_processing else tk.DISABLED))
    except Exception:
        try:
            button.configure(state=tk.DISABLED)
        except Exception:
            pass


def _prompt_z2_export_choice(self, *args, **kwargs):
    return z2_workflow_methods._prompt_z2_export_choice(self, *args, **kwargs)


def _start_z2_export_choice_flow(self):
    if not self._is_free_mode_session_context():
        self._start_plate_dataset_export()
        return
    state = self._build_z2_free_export_status_state()
    if bool(state.get("dataset_ready") or state.get("annotation_ready")):
        try:
            self._manual_review_export_ready = True
            self._dataset_export_completed = False
            self.free_mode_screen_var.set("export")
            self._refresh_plate_dataset_export_sources()
            self._refresh_left_panel_route_copy()
            self._refresh_detection_configuration_ui()
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            self._scroll_left_panel_to_widget(getattr(self, "export_section", None))
            try:
                self.frame.update_idletasks()
            except Exception:
                pass
        except Exception:
            pass
    choice = self._prompt_z2_export_choice(state)
    if choice == "dataset":
        self._start_plate_dataset_export()
    elif choice == "annotations":
        self._start_plate_annotation_package_export()


def _prompt_plate_annotation_package_export_options(self, *args, **kwargs):
    return z2_workflow_methods._prompt_plate_annotation_package_export_options(self, *args, **kwargs)


def _start_plate_annotation_package_export(self, *args, **kwargs):
    return z2_workflow_methods._start_plate_annotation_package_export(self, *args, **kwargs)


def _get_current_preview_plate_count_state(self) -> dict:
    annotations = list(getattr(self, "current_annotations", []) or [])
    approved_set = getattr(self, "_preview_approved_filenames", None)
    campaign_approved_set = getattr(self, "_campaign_pending_approved_filenames", None)
    approval_cache_token = (
        int(id(approved_set)) if approved_set is not None else 0,
        int(len(approved_set or set())),
        int(id(campaign_approved_set)) if campaign_approved_set is not None else 0,
        int(len(campaign_approved_set or set())),
        int(getattr(self, "_preview_approval_version", 0) or 0),
    )
    cache = getattr(self, "_current_preview_plate_count_cache", None)
    if (
        isinstance(cache, dict)
        and int(cache.get("annotations_id", -1) or -1) == int(id(getattr(self, "current_annotations", None)))
        and int(cache.get("total_images", -1) or -1) == int(len(annotations))
        and tuple(cache.get("approval_cache_token", ())) == approval_cache_token
    ):
        return cache

    approved_lookup = {
        str(name or "").strip().lower()
        for name in set(self._get_preview_approved_filenames() or set())
        if str(name or "").strip()
    }
    images_with_plates = 0
    total_plates = 0
    approved_images = 0
    approved_plates = 0
    for ann in annotations:
        plate_count = len(self._get_plate_detections(ann))
        if plate_count > 0:
            images_with_plates += 1
            total_plates += int(plate_count)
        image_name = str(getattr(ann, "filename", "") or "").strip().lower()
        try:
            is_approved = bool(
                self._preview_annotation_is_explicitly_approved(ann, approved_names=approved_lookup)
            )
        except Exception:
            is_approved = bool(image_name and image_name in approved_lookup)
        if image_name and is_approved and plate_count > 0:
            approved_images += 1
            approved_plates += int(plate_count)

    cache = {
        "annotations_id": int(id(getattr(self, "current_annotations", None))),
        "total_images": int(len(annotations)),
        "approval_cache_token": approval_cache_token,
        "images_with_plates": int(images_with_plates),
        "total_plates": int(total_plates),
        "approved_images": int(approved_images),
        "approved_plates": int(approved_plates),
    }
    self._current_preview_plate_count_cache = cache
    return cache


def _get_run_plate_annotation_counts(self, run_dir: Path | None) -> tuple[int, int]:
    if run_dir is None:
        return 0, 0

    try:
        run_dir = Path(run_dir)
    except Exception:
        return 0, 0

    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(run_dir, current_run_dir)
    ):
        state = self._get_current_preview_plate_count_state()
        return int(state.get("images_with_plates", 0) or 0), int(state.get("total_plates", 0) or 0)

    xml_path = run_dir / "annotations.xml"
    if not xml_path.exists():
        return 0, 0

    try:
        cache_run_key = str(run_dir.resolve())
    except Exception:
        cache_run_key = str(run_dir)

    cache_stamp = None
    try:
        stat = xml_path.stat()
        cache_stamp = (int(getattr(stat, "st_mtime_ns", 0) or 0), int(getattr(stat, "st_size", 0) or 0))
    except Exception:
        cache_stamp = None

    cache_entry = self._run_plate_count_cache.get(cache_run_key)
    if (
        isinstance(cache_entry, dict)
        and cache_stamp is not None
        and cache_entry.get("stamp") == cache_stamp
    ):
        return (
            int(cache_entry.get("images_with_plates", 0) or 0),
            int(cache_entry.get("total_plates", 0) or 0),
        )

    try:
        annotations = self._parse_cvat_preview_annotations(xml_path)
    except Exception:
        return 0, 0

    images_with_plates, total_plates = self._count_plate_annotations(annotations)
    if cache_stamp is not None:
        self._run_plate_count_cache[cache_run_key] = {
            "stamp": cache_stamp,
            "images_with_plates": int(images_with_plates),
            "total_plates": int(total_plates),
        }
    return images_with_plates, total_plates


def _get_campaign_project_total_image_count(self) -> int:
    if self._is_free_mode_session_context():
        return 0

    try:
        from ..campaign_manager import CAMPAIGN
        iteration_raw_dir = CAMPAIGN.get_iteration_image_source_dir() or CAMPAIGN.get_iteration_raw_dir()
        iteration_manifest_count = int(CAMPAIGN.get_iteration_image_count() or 0)
        ingest_manifest = CAMPAIGN.load_ingest_manifest() or {}
        approved_entries = list(CAMPAIGN.list_plate_approved_entries() or [])
    except Exception:
        iteration_raw_dir = None
        iteration_manifest_count = 0
        ingest_manifest = {}
        approved_entries = []

    try:
        iter_dir = Path(iteration_raw_dir) if iteration_raw_dir is not None else None
    except Exception:
        iter_dir = None

    try:
        cache_key = str(iter_dir.resolve()) if iter_dir is not None else ""
    except Exception:
        cache_key = str(iter_dir or "")
    cache_key = f"{cache_key}|approved:{len(approved_entries)}"

    cache_entry = getattr(self, "_campaign_project_image_count_cache", None)
    if isinstance(cache_entry, dict) and cache_entry.get("path") == cache_key:
        try:
            return int(cache_entry.get("count", 0) or 0)
        except Exception:
            return 0

    image_names: set[str] = set()
    for item in list((ingest_manifest or {}).get("selected_images") or []):
        if not isinstance(item, dict):
            continue
        safe_name = str(item.get("name", "") or "").strip().lower()
        if not safe_name:
            for key in ("target_path", "source_path", "iteration_target_path"):
                candidate = str(item.get(key, "") or "").strip()
                if candidate:
                    safe_name = Path(candidate).name.strip().lower()
                    break
        if safe_name:
            image_names.add(safe_name)

    try:
        if not image_names and iter_dir is not None and iter_dir.exists() and iter_dir.is_dir():
            for image_path in iter_dir.iterdir():
                if not image_path.is_file() or image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
                safe_name = str(image_path.name or "").strip().lower()
                if safe_name:
                    image_names.add(safe_name)
    except Exception:
        pass

    if not image_names and iteration_manifest_count > 0:
        return int(iteration_manifest_count)

    for entry in approved_entries:
        if not isinstance(entry, dict):
            continue
        safe_name = str(entry.get("image_name", "") or "").strip().lower()
        if safe_name:
            image_names.add(safe_name)

    image_count = int(len(image_names))

    try:
        self._campaign_project_image_count_cache = {
            "path": cache_key,
            "count": int(image_count),
        }
    except Exception:
        pass
    return int(image_count)


def _get_run_plate_approved_counts(self, run_dir: Path | None) -> tuple[int, int]:
    if run_dir is None:
        return 0, 0

    try:
        run_dir = Path(run_dir)
    except Exception:
        return 0, 0

    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    manifest = self._load_annotation_run_manifest(run_dir)
    explicit_approval_enabled = isinstance(manifest, dict) and "approved_filenames" in manifest

    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(run_dir, current_run_dir)
    ):
        state = self._get_current_preview_plate_count_state()
        if not explicit_approval_enabled and self._is_free_mode_session_context():
            return int(state.get("images_with_plates", 0) or 0), int(state.get("total_plates", 0) or 0)
        return int(state.get("approved_images", 0) or 0), int(state.get("approved_plates", 0) or 0)
    else:
        xml_path = run_dir / "annotations.xml"
        if not xml_path.exists():
            return 0, 0
        try:
            annotations = self._parse_cvat_preview_annotations(xml_path)
        except Exception:
            return 0, 0
        approved_filenames = self._load_annotation_run_approved_filenames(run_dir)
        if not approved_filenames:
            try:
                from ..campaign_manager import CAMPAIGN

                state_dir = CAMPAIGN.get_project_state_dir()
                state_path = Path(state_dir) / "annotation_ui_state.json" if state_dir is not None else None
                if state_path is not None and state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    state_run = str(state.get("plate_dataset_run") or state.get("last_preview_run_dir") or "").strip()
                    if state_run and self._paths_equivalent(Path(state_run), run_dir):
                        approved_filenames = {
                            str(name or "").strip().lower()
                            for name in list(state.get("preview_approved_filenames") or [])
                            if str(name or "").strip()
                        }
            except Exception:
                approved_filenames = set()

    if not annotations:
        return 0, 0

    if not explicit_approval_enabled:
        return self._count_plate_annotations(annotations)

    approved_images = 0
    approved_plates = 0
    for ann in annotations:
        image_name = str(getattr(ann, "filename", "") or "").strip().lower()
        if not image_name or image_name not in approved_filenames:
            continue
        plate_count = len(self._get_plate_detections(ann))
        if plate_count <= 0:
            continue
        approved_images += 1
        approved_plates += int(plate_count)

    return approved_images, approved_plates


def _get_current_preview_plate_approved_counts(self) -> tuple[int, int]:
    state = self._get_current_preview_plate_count_state()
    return int(state.get("approved_images", 0) or 0), int(state.get("approved_plates", 0) or 0)


def _get_run_plate_strict_approved_state(self, run_dir: Path | None) -> dict:
    safe_run_dir = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    state = {
        "ok": False,
        "run_dir": safe_run_dir,
        "approved_filenames": set(),
        "approved_images": 0,
        "approved_plates": 0,
        "total_images": 0,
        "total_plates": 0,
    }
    if safe_run_dir is None:
        return state

    current_run_dir = getattr(self, "current_annotation_run_dir", None)
    if (
        current_run_dir is not None
        and self.current_annotations
        and self._paths_equivalent(safe_run_dir, current_run_dir)
    ):
        count_state = self._get_current_preview_plate_count_state()
        approved_lookup = {
            str(name or "").strip().lower()
            for name in set(self._get_preview_approved_filenames() or set())
            if str(name or "").strip()
        }
        state.update(
            {
                "ok": bool(
                    int(count_state.get("approved_images", 0) or 0) > 0
                    and int(count_state.get("approved_plates", 0) or 0) > 0
                ),
                "approved_filenames": approved_lookup,
                "approved_images": int(count_state.get("approved_images", 0) or 0),
                "approved_plates": int(count_state.get("approved_plates", 0) or 0),
                "total_images": int(count_state.get("total_images", 0) or 0),
                "total_plates": int(count_state.get("total_plates", 0) or 0),
            }
        )
        return state
    else:
        try:
            annotations = self._parse_cvat_preview_annotations(safe_run_dir / "annotations.xml")
        except Exception:
            annotations = []
        approved_filenames = self._load_annotation_run_approved_filenames(safe_run_dir)
        if not approved_filenames:
            try:
                from ..campaign_manager import CAMPAIGN

                state_dir = CAMPAIGN.get_project_state_dir()
                state_path = Path(state_dir) / "annotation_ui_state.json" if state_dir is not None else None
                if state_path is not None and state_path.exists():
                    ui_state = json.loads(state_path.read_text(encoding="utf-8"))
                    state_run = str(ui_state.get("plate_dataset_run") or ui_state.get("last_preview_run_dir") or "").strip()
                    if state_run and self._paths_equivalent(Path(state_run), safe_run_dir):
                        approved_filenames = {
                            str(name or "").strip().lower()
                            for name in list(ui_state.get("preview_approved_filenames") or [])
                            if str(name or "").strip()
                        }
            except Exception:
                approved_filenames = set()

    total_images = len(annotations)
    _images_with_plates, total_plates = self._count_plate_annotations(annotations)
    approved_lookup = {
        str(name or "").strip().lower()
        for name in set(approved_filenames or set())
        if str(name or "").strip()
    }

    approved_images = 0
    approved_plates = 0
    for ann in annotations:
        image_name = str(getattr(ann, "filename", "") or "").strip().lower()
        if not image_name or image_name not in approved_lookup:
            continue
        plate_count = len(self._get_plate_detections(ann))
        if plate_count <= 0:
            continue
        approved_images += 1
        approved_plates += int(plate_count)

    state.update(
        {
            "ok": bool(approved_images > 0 and approved_plates > 0),
            "approved_filenames": approved_lookup,
            "approved_images": int(approved_images),
            "approved_plates": int(approved_plates),
            "total_images": int(total_images),
            "total_plates": int(total_plates),
        }
    )
    return state


def _get_plate_dataset_export_approval_state(self, run_dir: Path | None = None) -> dict:
    candidate = run_dir
    if candidate is None:
        candidate = self._resolve_safe_annotation_run_dir(str(self.plate_dataset_run_var.get() or "").strip(), require_xml=True)
    if candidate is None:
        candidate = self._get_preferred_annotation_run_dir(require_xml=True)
    return self._get_run_plate_strict_approved_state(candidate)


def _is_plate_dataset_export_allowed_for_current_selection(self) -> bool:
    state = dict(self._get_plate_dataset_export_approval_state() or {})
    if self._is_free_mode_session_context():
        min_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        return bool(int(state.get("approved_plates", 0) or 0) >= int(min_plates))
    return bool(state.get("ok"))


def _warn_plate_dataset_export_requires_ok(self, approval_state: dict | None = None) -> None:
    state = dict(approval_state or self._get_plate_dataset_export_approval_state())
    approved_images = int(state.get("approved_images", 0) or 0)
    approved_plates = int(state.get("approved_plates", 0) or 0)
    total_images = int(state.get("total_images", 0) or 0)
    total_plates = int(state.get("total_plates", 0) or 0)
    min_dataset_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    missing_dataset_plates = max(0, int(min_dataset_plates) - int(approved_plates or 0))
    message = (
        "Eksport datasetu YOLO Pose jest zablokowany, bo bramka datasetu nie jest spełniona.\n\n"
        f"Minimum datasetu: {min_dataset_plates} zatwierdzonych tablic [OK]. "
        f"Brakuje jeszcze: {missing_dataset_plates}.\n\n"
        "Warunek nadania [OK]: obraz musi mieć co najmniej jedną poprawną ramkę/poligon tablicy zapisaną w XML. "
        "Samo zaznaczenie obrazu bez ramki nie wystarczy.\n\n"
        "Po autoanotacji albo anotacji ręcznej zaznacz poprawne obrazy na liście, kliknij PPM "
        "i wybierz „Oznacz zaznaczone jako OK”. Eksport samych anotacji XML może być nadal dostępny, "
        "jeśli XML zawiera zapisane tablice.\n\n"
        f"Status teraz: OK obrazy={approved_images}, OK tablice={approved_plates}, "
        f"wszystkie obrazy w runie={total_images}, wszystkie tablice={total_plates}."
    )
    self._set_plate_export_status(
        (
            "Eksport datasetu YOLO zablokowany: bramka [OK] nie jest spełniona.\n\n"
            f"Brakuje do minimum: {missing_dataset_plates} tablic [OK].\n\n"
            "Warunek [OK]: obraz musi mieć zapisaną ramkę/poligon tablicy.\n\n"
            "Zaznacz obraz lub grupę obrazów na liście, kliknij PPM i wybierz „Oznacz zaznaczone jako OK”.\n\n"
            f"OK: {approved_images} obrazów / {approved_plates} tablic."
        ),
        "warning",
    )
    try:
        messagebox.showwarning(
            "Dataset YOLO wymaga bramki [OK]",
            message,
            parent=self.frame.winfo_toplevel(),
        )
    except Exception:
        pass


def _warn_plate_cut_requires_ok(self, approval_state: dict | None = None) -> None:
    state = dict(approval_state or self._get_plate_dataset_export_approval_state())
    approved_images = int(state.get("approved_images", 0) or 0)
    approved_plates = int(state.get("approved_plates", 0) or 0)
    total_images = int(state.get("total_images", 0) or 0)
    total_plates = int(state.get("total_plates", 0) or 0)
    message = (
        "Wycinanie tablic do Z3 jest zablokowane, bo żadna pozycja w tym runie nie ma statusu OK.\n\n"
        "Warunek nadania [OK]: obraz musi mieć co najmniej jedną poprawną ramkę/poligon tablicy zapisaną w XML. "
        "Samo zaznaczenie obrazu bez ramki nie wystarczy i nie może być źródłem wycinania.\n\n"
        "Zaznacz poprawne obrazy na liście, kliknij PPM i wybierz „Oznacz zaznaczone jako OK”. "
        "Możesz też zatwierdzać bieżące zdjęcie spacją w podglądzie. Dopiero po tym Z2 przygotuje "
        "dla PZ1 źródło zawierające wyłącznie zatwierdzone pozycje.\n\n"
        f"Status teraz: OK obrazy={approved_images}, OK tablice={approved_plates}, "
        f"wszystkie obrazy w runie={total_images}, wszystkie tablice={total_plates}."
    )
    try:
        messagebox.showwarning(
            "Wycinanie wymaga statusu OK",
            message,
            parent=self.frame.winfo_toplevel(),
        )
    except Exception:
        pass


def _get_campaign_step2_approval_context(self) -> dict:
    context = {
        "project_active": False,
        "current_step": 0,
        "iteration_target": "",
        "run_dir": None,
        "source_kind": "",
        "repair_mode": False,
    }

    try:
        from ..campaign_manager import CAMPAIGN

        context["project_active"] = bool(CAMPAIGN.get_active_project_name())
        context["current_step"] = int(CAMPAIGN.get_current_step() or 0)
        context["iteration_target"] = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        context["repair_mode"] = bool(
            (
                context["current_step"] == 3
                and context["iteration_target"] == "char"
            )
            or self._is_campaign_plate_step4_repair_return_mode()
        )
    except Exception:
        return context

    if not context["project_active"] or not (
        context["current_step"] == 2 or bool(context.get("repair_mode"))
    ):
        return context

    staging_root = None
    auto_root = None
    allowed_roots = []
    try:
        staging_root = CAMPAIGN.get_staging_dir("auto_ann")
        if staging_root is not None:
            staging_root = Path(staging_root)
            allowed_roots.append(staging_root)
    except Exception:
        staging_root = None
    try:
        auto_root = CAMPAIGN.get_dir("auto_ann")
        if auto_root is not None:
            auto_root = Path(auto_root)
            allowed_roots.append(auto_root)
    except Exception:
        auto_root = None

    def _classify_campaign_annotation_run(candidate: Path | None) -> str:
        if candidate is None:
            return ""
        try:
            if staging_root is not None and self._path_is_within(candidate, staging_root):
                return "staging"
        except Exception:
            pass
        try:
            if auto_root is not None and self._path_is_within(candidate, auto_root):
                return "existing"
        except Exception:
            pass
        return ""

    staging_candidate = self._resolve_existing_run_dir(CAMPAIGN.get_step2_staging_run())
    if staging_candidate is not None and (staging_candidate / "annotations.xml").exists():
        source_kind = _classify_campaign_annotation_run(staging_candidate)
        if source_kind:
            context["run_dir"] = staging_candidate
            context["source_kind"] = source_kind
            return context

    if context["iteration_target"] == "plate":
        try:
            approved_state = self._get_campaign_latest_approved_plate_run_state(
                input_dir=getattr(self, "current_input_dir", None)
            )
        except Exception:
            approved_state = {}

        approved_run_dir = self._resolve_existing_run_dir(approved_state.get("run_dir"))

        for raw_candidate in (
            getattr(self, "current_annotation_run_dir", None),
            self._get_preferred_annotation_run_dir(require_xml=True),
            getattr(self, "last_staging_run_dir", None),
            approved_run_dir,
        ):
            candidate = self._resolve_existing_run_dir(raw_candidate)
            if candidate is None or not (candidate / "annotations.xml").exists():
                continue
            if allowed_roots and not any(self._path_is_within(candidate, root) for root in allowed_roots):
                continue
            context["run_dir"] = candidate
            context["source_kind"] = "existing"
            return context

    if context["iteration_target"] != "char":
        return context

    for raw_candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
    ):
        candidate = self._resolve_existing_run_dir(raw_candidate)
        if candidate is None or not (candidate / "annotations.xml").exists():
            continue
        if allowed_roots and not any(self._path_is_within(candidate, root) for root in allowed_roots):
            continue
        context["run_dir"] = candidate
        context["source_kind"] = "existing"
        return context

    bootstrap_candidate = None
    try:
        bootstrap = self._get_campaign_auto_annotation_bootstrap("char")
        bootstrap_candidate = bootstrap.get("restore_run_dir")
    except Exception:
        bootstrap_candidate = None

    candidate = self._resolve_existing_run_dir(bootstrap_candidate)
    if candidate is not None and (candidate / "annotations.xml").exists():
        if not allowed_roots or any(self._path_is_within(candidate, root) for root in allowed_roots):
            context["run_dir"] = candidate
            context["source_kind"] = "existing"
            return context

    return context


def _refresh_step2_action_states(self, *args, **kwargs):
    return z2_workflow_methods._refresh_step2_action_states(self, *args, **kwargs)


def _build_annotation_success_next_steps(self) -> str:
    annotations = list(getattr(self, "current_annotations", []) or [])
    total_plates = sum(len(self._get_plate_detections(ann)) for ann in annotations)
    campaign_context = not self._is_free_mode_session_context()

    if getattr(self, "_current_run_manual_template", False):
        return (
            "Nowy run ręcznej anotacji Z2 jest gotowy. Poprawiaj polygony bezpośrednio w podglądzie "
            "i zapisuj korekty na bieżąco do annotations.xml."
            + (
                " Po zapisaniu zmian domknij E2; dataset i trening wykonasz potem w [Z4]."
                if campaign_context
                else ""
            )
        )

    if total_plates <= 0:
        return (
            "Run anotacji Z2 został zapisany, ale bez gotowych tablic. Możesz przejrzeć wyniki, poprawić je ręcznie "
            "albo uruchomić autoanotację ponownie z innymi ustawieniami."
        )

    if not campaign_context:
        return (
            "Run anotacji Z2 jest gotowy. Kliknij Dalej, aby przejść do korekty i decyzji po autoanotacji. "
            "Tam wybierzesz wyodrębnianie tablic, eksport anotacji XML albo eksport datasetu YOLO Pose."
        )

    return (
        "Run anotacji Z2 jest gotowy. Sprawdź lub popraw tablice w podglądzie, a po domknięciu E2 "
        "przygotujesz dataset i trening w [Z4]."
    )


def _show_auto_annotation_success_dialog(self, *args, **kwargs):
    return z2_workflow_methods._show_auto_annotation_success_dialog(self, *args, **kwargs)


def _load_plate_dataset_context_from_run(self, run_dir: Path, force_images_update: bool = False):
    run_dir = self._resolve_safe_annotation_run_dir(run_dir)
    if run_dir is None:
        return

    try:
        self.plate_dataset_run_var.set(str(run_dir))
    except Exception:
        pass

    self.plate_dataset_out_var.set(self._plate_dataset_output_preview(run_dir))

    current_images = Path(self.plate_dataset_images_var.get().strip()) if self.plate_dataset_images_var.get().strip() else None
    current_valid = bool(current_images and current_images.exists())

    if current_valid and not force_images_update:
        return

    manifest = self._load_annotation_run_manifest(run_dir)
    input_dir = str(manifest.get("input_dir") or "").strip()
    if input_dir and Path(input_dir).exists():
        self.plate_dataset_images_var.set(input_dir)
        return

    if getattr(self, "current_input_dir", None):
        try:
            current_input = Path(self.current_input_dir)
            if current_input.exists():
                self.plate_dataset_images_var.set(str(current_input))
        except Exception:
            pass


def _refresh_plate_dataset_export_sources(self, *args, **kwargs):
    return z2_workflow_methods._refresh_plate_dataset_export_sources(self, *args, **kwargs)


def _select_plate_dataset_run_dir(self):
    initialdir = self.plate_dataset_run_var.get().strip()
    if not initialdir:
        try:
            from ..campaign_manager import CAMPAIGN

            if CAMPAIGN.get_active_project_name() and not self._is_free_mode_session_context():
                auto_dir = CAMPAIGN.get_dir("auto_ann")
                staging_dir = CAMPAIGN.get_staging_dir("auto_ann")
                initialdir = str(auto_dir or staging_dir or "")
        except Exception:
            initialdir = ""
    if not initialdir:
        initialdir = self.output_dir_var.get().strip() or str(CONFIG.get_auto_annotations_dir("plate"))
    path = filedialog.askdirectory(initialdir=initialdir)
    if not path:
        return

    run_dir = self._resolve_safe_annotation_run_dir(path, require_xml=True)
    if run_dir is None:
        return messagebox.showerror(
            "Błędny run anotacji",
            "Wybrany folder runu anotacji musi lezec w aktywnym workspace Z2 i zawierac annotations.xml.",
        )
    self._load_plate_dataset_context_from_run(run_dir, force_images_update=True)
    self._refresh_plate_dataset_export_sources()


def _select_plate_dataset_images_dir(self):
    initialdir = self.plate_dataset_images_var.get().strip() or self.input_dir_var.get().strip() or str(CONFIG.DIR_1_RAW)
    path = filedialog.askdirectory(initialdir=initialdir)
    if not path:
        return

    self.plate_dataset_images_var.set(path)
    self._refresh_plate_dataset_export_sources()


def _update_plate_dataset_ratio_labels(self):
    train = float(self.plate_train_pct.get())
    val = float(self.plate_val_pct.get())
    max_val = max(0.0, 100.0 - train)
    try:
        self.plate_val_scale.configure(to=max_val)
    except Exception:
        pass
    if val > max_val:
        val = max_val
        self.plate_val_pct.set(val)
    elif val < 0.0:
        val = 0.0
        self.plate_val_pct.set(val)

    test = max(0.0, 100.0 - train - val)
    self.plate_train_lbl.configure(text=f"{train:.0f}%")
    self.plate_val_lbl.configure(text=f"{val:.0f}%")
    self.plate_test_lbl.configure(text=f"Test: {test:.0f}%")


def _start_plate_dataset_export(self, *args, **kwargs):
    return z2_workflow_methods._start_plate_dataset_export(self, *args, **kwargs)
