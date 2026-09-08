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
from dataclasses import replace
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
    AugmentationProfile,
    DatasetCreator,
    DatasetSplitter,
    TrainingHistory,
    TrainingStatus,
    YOLOPoseTrainer,
    analyze_character_class_distribution,
    augment_yolo_dataset_train_split,
    build_character_dataset_file_fingerprint,
    build_character_training_variant_manifest,
    compare_character_val_test_unchanged,
    describe_augmentation_randomness_mode,
    ensure_yolo_dataset_yaml_points_to_root,
    get_albumentations_status,
    install_albumentations,
    is_albumentations_available,
    plan_character_train_augmentation,
    save_character_distribution_artifacts,
    update_yolo_dataset_class_names,
)
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_augmentation_modal import ask_step4_augmentation_profile
from .z4_dataset_preview import open_yolo_dataset_preview
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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None
def _pick_file(self, var, ext, initialdir=None):
    kwargs = {"filetypes": [("File", ext)]}
    if initialdir and Path(initialdir).exists():
        kwargs["initialdir"] = str(initialdir)
        
    p = filedialog.askopenfilename(**kwargs)
    if p:
        var.set(p)
        try:
            self._refresh_dataset_creator_cta_state()
        except Exception:
            pass
        try:
            self._refresh_dataset_split_cta_state()
        except Exception:
            pass
    return p

def _pick_dir(self, var, initialdir=None):
    kwargs = {}
    if initialdir and Path(initialdir).exists():
        kwargs["initialdir"] = str(initialdir)
        
    p = filedialog.askdirectory(**kwargs)
    if p:
        var.set(p)
        try:
            self._refresh_dataset_creator_cta_state()
        except Exception:
            pass
        try:
            self._refresh_dataset_split_cta_state()
        except Exception:
            pass
    return p

def _get_step4_creator_annotated_image_count(self) -> int:
    """Return a lightweight cached count of images that can enter the plate dataset."""

    def _fallback_from_creator() -> int:
        try:
            return max(0, len(set(self.creator.get_annotated_image_names() or set())))
        except Exception:
            return 0

    xml_raw = ""
    try:
        xml_raw = str(getattr(self, "cvat_xml_var", tk.StringVar()).get() or "").strip()
    except Exception:
        xml_raw = ""
    if not xml_raw:
        return _fallback_from_creator()

    try:
        xml_path = Path(xml_raw)
    except Exception:
        return _fallback_from_creator()
    if not xml_path.exists() or not xml_path.is_file():
        return _fallback_from_creator()

    try:
        images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
    except Exception:
        images_raw = ""
    if images_raw:
        try:
            exact_count = int(_get_step4_augmentation_source_count(self, "plate") or 0)
        except Exception:
            exact_count = 0
        if exact_count > 0:
            return exact_count

    try:
        stat = xml_path.stat()
        signature = (str(xml_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        signature = (str(xml_path), 0, 0)

    cache = getattr(self, "_creator_split_preview_count_cache", None)
    if isinstance(cache, dict) and cache.get("signature") == signature:
        try:
            return max(0, int(cache.get("count", 0) or 0))
        except Exception:
            pass

    count = 0
    try:
        root = ET.parse(xml_path).getroot()
        plate_labels = {str(label or "").strip().lower() for label in CONFIG.PLATE_LABELS}
        for image in root.findall(".//image"):
            image_has_plate = False
            for poly in image.findall("polygon"):
                label = str(poly.get("label", "") or "").strip().lower()
                if label not in plate_labels:
                    continue
                points_str = str(poly.get("points", "") or "").strip()
                if not points_str:
                    continue
                points = [part for part in points_str.split(";") if "," in part]
                if len(points) == 4:
                    image_has_plate = True
                    break
            if image_has_plate:
                count += 1
    except Exception:
        count = _fallback_from_creator()

    try:
        self._creator_split_preview_count_cache = {"signature": signature, "count": int(count)}
    except Exception:
        pass
    return max(0, int(count or 0))


def _estimate_step4_creator_split_counts(self, train: float, val: float, test: float) -> dict[str, int]:
    total_count = _get_step4_creator_annotated_image_count(self)
    if total_count <= 0:
        return {}
    ratios = {
        "train": max(0.0, float(train or 0.0)),
        "val": max(0.0, float(val or 0.0)),
        "test": max(0.0, float(test or 0.0)),
    }
    try:
        counts = DatasetCreator._allocate_split_counts(int(total_count), ratios)
    except Exception:
        counts = {}
    if not counts:
        train_count = int(round(total_count * ratios["train"] / 100.0))
        val_count = int(round(total_count * ratios["val"] / 100.0))
        train_count = max(0, min(total_count, train_count))
        val_count = max(0, min(total_count - train_count, val_count))
        counts = {
            "train": train_count,
            "val": val_count,
            "test": max(0, total_count - train_count - val_count),
        }
    counts["total"] = int(total_count)
    return {key: max(0, int(value or 0)) for key, value in counts.items()}


def _draw_step4_split_preview_bar(self, train: float, val: float, test: float, counts: dict[str, int] | None = None):
    canvas = getattr(self, "creator_split_bar", None)
    if canvas is None:
        return
    try:
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
    except Exception:
        return
    if width <= 2 or height <= 2:
        return

    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    bg = palette.get("panel", "#252526")
    success = palette.get("success", "#4ec9b0")
    warning = palette.get("warning", "#d7ba7d")
    accent = palette.get("accent", "#0e639c")
    border = blend_hex_colors(success, palette.get("panel_border", "#3c3c3c"), 0.62)
    colors = {
        "train": success,
        "val": warning,
        "test": accent,
    }
    labels = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
    }
    counts = dict(counts or {})
    values = (
        ("train", max(0.0, float(train or 0.0))),
        ("val", max(0.0, float(val or 0.0))),
        ("test", max(0.0, float(test or 0.0))),
    )
    total = sum(value for _name, value in values) or 100.0
    try:
        canvas.delete("all")
        canvas.configure(bg=bg)
        x = 1
        inner_w = max(1, width - 2)
        for index, (name, value) in enumerate(values):
            next_x = width - 1 if index == len(values) - 1 else int(1 + inner_w * (sum(v for _n, v in values[: index + 1]) / total))
            canvas.create_rectangle(
                x,
                2,
                max(x + 1, next_x),
                height - 2,
                fill=colors.get(name, success),
                outline="",
            )
            segment_w = max(0, next_x - x)
            count_value = int(counts.get(name, 0) or 0)
            if counts and segment_w >= 30:
                if segment_w >= 92:
                    label = f"{labels.get(name, name)} {value:.0f}% / {count_value}"
                elif segment_w >= 58:
                    label = f"{value:.0f}% / {count_value}"
                else:
                    label = str(count_value)
            elif segment_w >= 58:
                label = f"{labels.get(name, name)} {value:.0f}%"
            else:
                label = ""
            if label:
                canvas.create_text(
                    (x + next_x) / 2,
                    height / 2,
                    text=label,
                    fill=palette.get("bg", "#1e1e1e"),
                    font=("Segoe UI Semibold", 8),
                )
            x = next_x
        canvas.create_rectangle(1, 2, width - 1, height - 2, outline=border, width=1)
    except Exception:
        pass

def _update_ratio_labels(self):
    train = float(self.train_pct.get())
    val = float(self.val_pct.get())
    max_train_plus_val = 95.0
    if train + val > max_train_plus_val:
        val = max(5.0, max_train_plus_val - train)
        self.val_pct.set(val)

    test = max(5.0, 100.0 - train - val)
    try:
        normalized_mode = CONFIG.normalize_task_target(getattr(self, "_step4_dataset_mode", "plate"))
    except Exception:
        normalized_mode = "plate"
    if normalized_mode == "char":
        ratio_key = (round(train, 4), round(val, 4), round(test, 4))
        previous_ratio_key = getattr(self, "_pending_character_balance_ratio_key", None)
        try:
            previous_ratio_key = tuple(previous_ratio_key) if previous_ratio_key is not None else None
        except Exception:
            previous_ratio_key = None
        if previous_ratio_key is not None and previous_ratio_key != ratio_key:
            _invalidate_pending_character_balance_plan(
                self,
                "Proporcje splitu zmieniły się - przelicz reprezentację MZ.",
            )
        try:
            setattr(self, "_pending_character_balance_ratio_key", ratio_key)
        except Exception:
            pass
    for attr_name, value in (
        ("train_lbl", f"{train:.0f}%"),
        ("val_lbl", f"{val:.0f}%"),
        ("test_lbl", f"Test: {test:.0f}%"),
        ("creator_train_lbl", f"{train:.0f}%"),
        ("creator_val_lbl", f"{val:.0f}%"),
        ("creator_test_lbl", f"Test: {test:.0f}%"),
    ):
        widget = getattr(self, attr_name, None)
        if widget is not None:
            try:
                widget.configure(text=value)
            except Exception:
                pass
            self._style_training_success_label(widget)
    split_counts = _estimate_step4_creator_split_counts(self, train, val, test)
    _draw_step4_split_preview_bar(self, train, val, test, split_counts)
    try:
        self._refresh_step4_augmentation_summary(getattr(self, "_step4_dataset_mode", "plate"))
    except Exception:
        pass
    try:
        self._refresh_step4_creator_decision_summary()
    except Exception:
        pass

def _refresh_step4_creator_decision_summary(self):
    var = getattr(self, "creator_decision_summary_var", None)
    try:
        train = float(self.train_pct.get())
    except Exception:
        train = 80.0
    try:
        val = float(self.val_pct.get())
    except Exception:
        val = 10.0
    test = max(5.0, 100.0 - train - val)

    def _estimate_counts(target: str) -> dict[str, int]:
        normalized = CONFIG.normalize_task_target(target)
        if normalized == "plate":
            return _estimate_step4_creator_split_counts(self, train, val, test)
        try:
            total_count = int(_get_step4_augmentation_source_count(self, normalized) or 0)
        except Exception:
            total_count = 0
        if total_count <= 0:
            return {}
        ratios = {"train": train, "val": val, "test": test}
        try:
            counts = DatasetCreator._allocate_split_counts(total_count, ratios)
        except Exception:
            counts = {}
        if not counts:
            train_count = int(round(total_count * train / 100.0))
            val_count = int(round(total_count * val / 100.0))
            train_count = max(0, min(total_count, train_count))
            val_count = max(0, min(total_count - train_count, val_count))
            counts = {
                "train": train_count,
                "val": val_count,
                "test": max(0, total_count - train_count - val_count),
            }
        counts["total"] = total_count
        return {key: max(0, int(value or 0)) for key, value in counts.items()}

    def _refresh_table(prefix: str, target: str) -> tuple[str, str, str, str]:
        normalized = CONFIG.normalize_task_target(target)
        unit = "tablic" if normalized == "char" else "zdj\u0119\u0107"
        counts = _estimate_counts(normalized)
        try:
            profile = self._get_step4_augmentation_profile(normalized)
        except Exception:
            profile = None
        try:
            extra = max(0, int(getattr(profile, "extra_count", 0) or 0))
        except Exception:
            extra = 0
        augmentation_enabled = bool(getattr(profile, "enabled", False)) and extra > 0

        values = {
            "train": f"{train:.0f}%\n{counts.get('train', 0) or '-'} {unit}",
            "val": f"{val:.0f}%\n{counts.get('val', 0) or '-'} {unit}",
            "test": f"{test:.0f}%\n{counts.get('test', 0) or '-'} {unit}",
            "augmentation": f"+{extra}\ntrain" if augmentation_enabled else "0\nwy\u0142.",
        }
        for key, text in values.items():
            value_var = getattr(self, f"{prefix}_decision_{key}_var", None)
            if value_var is not None:
                try:
                    value_var.set(text)
                except Exception:
                    pass

        palette = getattr(getattr(self, "app", None), "palette", {}) or {}
        color_map = {
            "train": palette.get("success", "#4ec9b0"),
            "val": palette.get("warning", "#d7ba7d"),
            "test": palette.get("accent", "#0e639c"),
            "augmentation": palette.get("success", "#4ec9b0") if augmentation_enabled else palette.get("muted", "#c7c7c7"),
        }
        for key, widget in (getattr(self, f"_{prefix}_decision_value_widgets", {}) or {}).items():
            try:
                widget.configure(fg=color_map.get(key, palette.get("fg", "#f3f3f3")))
            except Exception:
                pass
        return values["train"], values["val"], values["test"], values["augmentation"]

    creator_values = _refresh_table("creator", "plate")
    _refresh_table("split", "char")
    if var is not None:
        try:
            var.set(
                f"Train {train:.0f}% | Val {val:.0f}% | Test {test:.0f}% | {creator_values[3].replace(chr(10), ' ')}"
            )
        except Exception:
            pass

def _on_base_model_change(self):
    try:
        self._refresh_training_base_model_selection_ui()
    except Exception:
        pass
    try:
        self._refresh_training_recommendation_table()
    except Exception:
        pass
    try:
        self._refresh_training_start_state()
    except Exception:
        pass

def _on_cvat_xml_source_changed(self):
    try:
        self._refresh_dataset_creator_cta_state()
    except Exception:
        pass
    try:
        self._cvat_source_binding_generation = int(getattr(self, "_cvat_source_binding_generation", 0) or 0) + 1
    except Exception:
        self._cvat_source_binding_generation = 1
    try:
        self._schedule_cvat_source_binding_refresh()
    except Exception:
        pass

def _on_cvat_images_source_changed(self):
    try:
        self._refresh_dataset_creator_cta_state()
    except Exception:
        pass
    try:
        self._cvat_source_binding_generation = int(getattr(self, "_cvat_source_binding_generation", 0) or 0) + 1
    except Exception:
        self._cvat_source_binding_generation = 1
    pending = getattr(self, "_cvat_source_binding_after_id", None)
    if pending and hasattr(self, "frame") and self.frame is not None:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._cvat_source_binding_after_id = None

def _schedule_cvat_source_binding_refresh(self, delay_ms: int = 0):
    if not hasattr(self, "frame") or self.frame is None:
        return
    try:
        if self._get_creator_source_mode() != "xml":
            return
    except Exception:
        pass

    pending = getattr(self, "_cvat_source_binding_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass

    try:
        token = int(getattr(self, "_cvat_source_binding_generation", 0) or 0)
    except Exception:
        token = 0

    self._cvat_source_binding_after_id = self.frame.after(
        max(0, int(delay_ms)),
        lambda expected_token=token: self._auto_bind_cvat_images_dir_from_xml(expected_token),
    )

def _close_cvat_matching_progress_modal(self):
    window = getattr(self, "_cvat_matching_progress_window", None)
    self._cvat_matching_progress_window = None
    if window is None:
        return
    try:
        window.destroy()
    except Exception:
        pass

def _show_cvat_matching_progress_modal(self, total_images: int | None = None):
    _close_cvat_matching_progress_modal(self)

    parent = getattr(self, "frame", None)
    try:
        root = parent.winfo_toplevel() if parent is not None else None
    except Exception:
        root = None

    try:
        window = tk.Toplevel(root or parent)
        window.title("Dopasowywanie obrazów do XML")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: None)
        if root is not None:
            try:
                window.transient(root)
            except Exception:
                pass

        shell = ttk.Frame(window, padding=(18, 14))
        shell.pack(fill="both", expand=True)

        ttk.Label(
            shell,
            text="Szukam zgodnego katalogu zdjęć dla wybranego pliku XML.",
            font=("Segoe UI", 10, "bold"),
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

        details = "Program sprawdza stałe katalogi workspace i porównuje nazwy obrazów zapisane w XML."
        if total_images:
            details += f"\nW XML wykryto {total_images} obrazów do dopasowania."
        ttk.Label(
            shell,
            text=details,
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(8, 10))

        progress = ttk.Progressbar(shell, mode="indeterminate", length=360)
        progress.pack(fill="x")
        try:
            progress.start(12)
        except Exception:
            pass

        self._cvat_matching_progress_window = window

        try:
            window.update_idletasks()
            width = window.winfo_width()
            height = window.winfo_height()
            if root is not None:
                root.update_idletasks()
                x = root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2)
                y = root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2)
                window.geometry(f"+{x}+{y}")
            window.lift()
            window.update()
        except Exception:
            pass
    except Exception:
        self._cvat_matching_progress_window = None

def _normalize_cvat_xml_image_relpath(self, raw_name: str) -> str:
    raw = str(raw_name or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]

    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    return "/".join(parts)

def _cvat_xml_relpath_to_path(self, rel_path: str) -> Path:
    parts = [part for part in PurePosixPath(rel_path).parts if part not in ("", ".")]
    return Path(*parts) if parts else Path()

def _read_cvat_xml_image_names(self, xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    names: list[str] = []
    for image_el in tree.getroot().findall(".//image"):
        normalized = self._normalize_cvat_xml_image_relpath(image_el.get("name") or "")
        if normalized:
            names.append(normalized)
    return list(dict.fromkeys(names))

def _evaluate_images_dir_for_cvat_xml(self, images_dir: Path, xml_image_names: list[str]) -> dict:
    matched = 0
    missing: list[str] = []
    for rel_name in xml_image_names:
        candidate = Path(images_dir) / self._cvat_xml_relpath_to_path(rel_name)
        if candidate.exists() and candidate.is_file():
            matched += 1
        else:
            missing.append(rel_name)
    return {
        "images_dir": Path(images_dir),
        "total": len(xml_image_names),
        "matched": matched,
        "missing_count": len(missing),
        "missing": missing,
    }

def _get_cvat_image_source_roots(self) -> list[Path]:
    roots: list[Path] = []

    for getter in (
        lambda: CAMPAIGN.get_iteration_image_source_dir(),
        lambda: CAMPAIGN.get_iteration_raw_dir(),
        lambda: CAMPAIGN.get_master_pool_dir(),
        lambda: CAMPAIGN.get_dir("raw"),
    ):
        try:
            candidate = getter()
            if candidate:
                roots.append(Path(candidate))
        except Exception:
            pass

    try:
        roots.append(Path(CONFIG.DIR_1_RAW))
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            if root.exists() and root.is_dir():
                unique.append(root)
        except Exception:
            pass
    return unique

def _derive_cvat_candidate_root_for_match(self, found_file: Path, xml_rel_name: str) -> Path | None:
    parts = [part for part in PurePosixPath(xml_rel_name).parts if part not in ("", ".")]
    if not parts:
        return None

    ascend_levels = len(parts) - 1
    parents = found_file.parents
    if ascend_levels >= len(parents):
        return None

    candidate_root = parents[ascend_levels]
    try:
        expected_path = (candidate_root / self._cvat_xml_relpath_to_path(xml_rel_name)).resolve()
        if expected_path != found_file.resolve():
            return None
    except Exception:
        return None

    return candidate_root

def _find_matching_images_dir_for_cvat_xml(self, xml_image_names: list[str]) -> dict | None:
    if not xml_image_names:
        return None

    search_roots = self._get_cvat_image_source_roots()
    if not search_roots:
        return None

    sample_names = xml_image_names[: min(24, len(xml_image_names))]
    sample_by_basename: dict[str, list[str]] = {}
    for rel_name in sample_names:
        sample_by_basename.setdefault(PurePosixPath(rel_name).name, []).append(rel_name)

    candidate_hits: dict[str, dict] = {}
    for search_root in search_roots:
        try:
            for file_path in search_root.rglob("*"):
                if not file_path.is_file():
                    continue
                candidate_rel_names = sample_by_basename.get(file_path.name)
                if not candidate_rel_names:
                    continue
                for rel_name in candidate_rel_names:
                    candidate_root = self._derive_cvat_candidate_root_for_match(file_path, rel_name)
                    if candidate_root is None:
                        continue
                    try:
                        key = str(candidate_root.resolve())
                    except Exception:
                        key = str(candidate_root)
                    entry = candidate_hits.setdefault(
                        key,
                        {"images_dir": candidate_root, "sample_hits": set()},
                    )
                    entry["sample_hits"].add(rel_name)
        except Exception as exc:
            logger.debug(f"Nie udało się przeskanować {search_root} przy dopasowaniu obrazów do XML: {exc}")

    if not candidate_hits:
        return None

    ranked_candidates = sorted(
        candidate_hits.values(),
        key=lambda item: (len(item["sample_hits"]), -len(str(item["images_dir"]))),
        reverse=True,
    )

    best_match = None
    best_score = None
    for candidate in ranked_candidates[:8]:
        stats = self._evaluate_images_dir_for_cvat_xml(candidate["images_dir"], xml_image_names)
        stats["sample_hits"] = len(candidate["sample_hits"])
        score = (stats["matched"], -stats["missing_count"], stats["sample_hits"])
        if best_match is None or score > best_score:
            best_match = stats
            best_score = score
    return best_match

def _finish_auto_bind_cvat_images_dir_from_xml(
    self,
    expected_token: int | None,
    original_images_raw: str,
    best_candidate: dict | None,
):
    _close_cvat_matching_progress_modal(self)

    try:
        current_token = int(getattr(self, "_cvat_source_binding_generation", 0) or 0)
        if expected_token is not None and int(expected_token) != current_token:
            return
    except Exception:
        pass

    try:
        if self._get_creator_source_mode() != "xml":
            return
    except Exception:
        pass

    if not best_candidate or best_candidate.get("matched") != best_candidate.get("total"):
        try:
            status = getattr(self, "ds_status", None)
            if status is not None:
                self._style_training_success_label(status)
                self._set_training_widget_text(
                    status,
                    "Nie znaleziono automatycznie zgodnego katalogu obrazów. Wskaż katalog ręcznie.",
                )
        except Exception:
            pass
        return

    latest_images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
    if latest_images_raw and latest_images_raw != original_images_raw:
        return

    candidate_dir = Path(best_candidate["images_dir"])
    try:
        display_dir = self._format_workspace_relative_path(candidate_dir)
    except Exception:
        display_dir = str(candidate_dir)

    attach = messagebox.askyesno(
        "Znaleziono zgodny folder obrazów",
        (
            "Na podstawie pliku XML znaleziono folder zawierający komplet zgodnych obrazów.\n\n"
            f"Dopasowanie: {best_candidate['matched']}/{best_candidate['total']} plików z XML\n"
            f"Folder: {display_dir}\n\n"
            "Czy podpiąć ten folder jako źródło obrazów dla kreatora PZ1?"
        ),
        parent=getattr(self, "frame", None),
    )
    if not attach:
        try:
            status = getattr(self, "ds_status", None)
            if status is not None:
                self._style_training_success_label(status)
                self._set_training_widget_text(
                    status,
                    "Znaleziono zgodny folder obrazów, ale podpięcie anulowano. Wskaż folder ręcznie albo wybierz XML ponownie.",
                )
        except Exception:
            pass
        return

    try:
        self.cvat_images_var.set(str(candidate_dir.resolve()))
    except Exception:
        self.cvat_images_var.set(str(candidate_dir))

    try:
        status = getattr(self, "ds_status", None)
        if status is not None:
            self._style_training_success_label(status)
            self._set_training_widget_text(
                status,
                f"Podpięto zgodny folder obrazów: {best_candidate['matched']}/{best_candidate['total']} plików z XML w {display_dir}.",
            )
    except Exception:
        pass

def _auto_bind_cvat_images_dir_from_xml(self, expected_token: int | None = None):
    self._cvat_source_binding_after_id = None

    try:
        current_token = int(getattr(self, "_cvat_source_binding_generation", 0) or 0)
        if expected_token is not None and int(expected_token) != current_token:
            return
    except Exception:
        pass

    try:
        if self._get_creator_source_mode() != "xml":
            return
    except Exception:
        pass

    xml_raw = str(getattr(self, "cvat_xml_var", tk.StringVar()).get() or "").strip()
    if not xml_raw:
        return
    try:
        xml_path = Path(xml_raw)
    except Exception:
        return
    if not xml_path.exists() or not xml_path.is_file():
        return

    try:
        xml_image_names = self._read_cvat_xml_image_names(xml_path)
    except Exception as exc:
        logger.debug(f"Nie udało się odczytać nazw obrazów z XML dla PZ1: {exc}")
        return
    if not xml_image_names:
        return

    images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
    if images_raw:
        try:
            current_dir = Path(images_raw)
            if current_dir.exists() and current_dir.is_dir():
                current_stats = self._evaluate_images_dir_for_cvat_xml(current_dir, xml_image_names)
                if current_stats["matched"] == current_stats["total"]:
                    return
        except Exception:
            pass

    _show_cvat_matching_progress_modal(self, len(xml_image_names))

    def worker():
        try:
            best_candidate = self._find_matching_images_dir_for_cvat_xml(xml_image_names)
        except Exception as exc:
            logger.debug(f"Nie udało się dopasować katalogu obrazów do XML w tle: {exc}")
            best_candidate = None

        frame = getattr(self, "frame", None)
        if frame is None:
            return

        def finish():
            _finish_auto_bind_cvat_images_dir_from_xml(
                self,
                expected_token,
                images_raw,
                best_candidate,
            )

        try:
            frame.after(0, finish)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()
    return

    try:
        best_candidate = self._find_matching_images_dir_for_cvat_xml(xml_image_names)
    finally:
        _close_cvat_matching_progress_modal(self)

    if not best_candidate or best_candidate.get("matched") != best_candidate.get("total"):
        try:
            status = getattr(self, "ds_status", None)
            if status is not None:
                self._style_training_success_label(status)
                self._set_training_widget_text(
                    status,
                    "Nie znaleziono automatycznie zgodnego katalogu obrazów. Wskaż katalog ręcznie.",
                )
        except Exception:
            pass
        return

    try:
        current_token = int(getattr(self, "_cvat_source_binding_generation", 0) or 0)
        if expected_token is not None and int(expected_token) != current_token:
            return
    except Exception:
        pass

    # Użytkownik mógł wskazać folder ręcznie w czasie oczekiwania na automatyczne dopasowanie.
    # Wtedy nie pokazujemy spóźnionego modala, nawet jeśli folder wymaga dalszej walidacji.
    latest_images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
    if latest_images_raw and latest_images_raw != images_raw:
        return

    candidate_dir = Path(best_candidate["images_dir"])
    try:
        display_dir = self._format_workspace_relative_path(candidate_dir)
    except Exception:
        display_dir = str(candidate_dir)

    attach = messagebox.askyesno(
        "Znaleziono zgodny folder obrazów",
        (
            "Na podstawie pliku XML znaleziono folder zawierający komplet zgodnych obrazów.\n\n"
            f"Dopasowanie: {best_candidate['matched']}/{best_candidate['total']} plików z XML\n"
            f"Folder: {display_dir}\n\n"
            "Czy podpiąć ten folder jako źródło obrazów dla kreatora PZ1?"
        ),
        parent=getattr(self, "frame", None),
    )
    if not attach:
        try:
            status = getattr(self, "ds_status", None)
            if status is not None:
                self._style_training_success_label(status)
                self._set_training_widget_text(
                    status,
                    "Znaleziono zgodny folder obrazów, ale podpięcie anulowano. Wskaż folder ręcznie albo wybierz XML ponownie.",
                )
        except Exception:
            pass
        return

    try:
        self.cvat_images_var.set(str(candidate_dir.resolve()))
    except Exception:
        self.cvat_images_var.set(str(candidate_dir))

    try:
        status = getattr(self, "ds_status", None)
        if status is not None:
            self._style_training_success_label(status)
            self._set_training_widget_text(
                status,
                f"Podpięto zgodny folder obrazów: {best_candidate['matched']}/{best_candidate['total']} plików z XML w {display_dir}.",
            )
    except Exception:
        pass

def _get_creator_source_mode(self) -> str:
    try:
        mode = str(self.creator_source_mode_var.get() or "").strip().lower()
    except Exception:
        mode = ""
    return mode if mode in {"ready", "xml"} else "xml"

def _set_creator_source_mode(self, mode: str):
    normalized = str(mode or "").strip().lower()
    if normalized not in {"ready", "xml"}:
        normalized = "xml"
    try:
        self.creator_source_mode_var.set(normalized)
    except Exception:
        pass

def _on_creator_source_mode_change(self):
    try:
        self._refresh_step4_dataset_mode_ui()
    except Exception:
        pass
    try:
        self._refresh_dataset_creator_cta_state()
    except Exception:
        pass

def _resolve_dataset_creator_inputs(self) -> dict:
    result = {
        "ok": False,
        "xml": None,
        "images_dir": None,
        "message": "W trybie budowy z XML wymagane są dwa zgodne źródła: plik XML i folder obrazów.",
    }

    xml_raw = str(getattr(self, "cvat_xml_var", tk.StringVar()).get() or "").strip()
    images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
    adapter = PlateXmlImagesSourceAdapter(
        provenance=self._format_training_source_provenance("Źródło PZ1"),
        source_stage="Z4/PZ1",
    )
    validation = adapter.validate(xml_raw, images_raw)
    result["message"] = validation.message
    result["training_source"] = validation.source
    if not validation.ok:
        return result

    result.update(
        {
            "ok": True,
            "xml": validation.raw.get("xml"),
            "images_dir": validation.raw.get("images_dir"),
            "message": validation.message,
            "training_source": validation.source,
        }
    )
    return result

def _normalize_xml_image_name(image_name: str) -> str:
    return str(image_name or "").strip().replace("\\", "/")

def _validate_dataset_creator_xml_images_alignment(self, images_dir: Path) -> dict:
    result = {
        "ok": False,
        "message": "Nie udało się porównać XML z folderem obrazów.",
        "missing": [],
        "extra_count": 0,
    }

    try:
        images_dir = Path(images_dir)
    except Exception:
        return result

    if not images_dir.exists() or not images_dir.is_dir():
        result["message"] = "Folder obrazów nie istnieje."
        return result

    xml_names = {
        self._normalize_xml_image_name(name)
        for name in set(getattr(self.creator, "xml_image_names", set()) or set())
        if str(name or "").strip()
    }
    annotated_names = {
        self._normalize_xml_image_name(name)
        for name in set(self.creator.get_annotated_image_names() or set())
        if str(name or "").strip()
    }
    required_names = xml_names or annotated_names
    if not required_names:
        result["message"] = "XML nie zawiera obrazów do walidacji."
        return result

    missing = []
    for image_name in sorted(required_names):
        candidate = images_dir / Path(image_name)
        if not candidate.exists() or not candidate.is_file():
            missing.append(image_name)

    if missing:
        sample = ", ".join(missing[:8])
        more = "" if len(missing) <= 8 else f" oraz {len(missing) - 8} więcej"
        result["missing"] = missing
        result["message"] = (
            "Plik XML i folder obrazów nie są zgodne.\n\n"
            f"W XML jest {len(required_names)} obraz(ów), ale w wybranym folderze brakuje {len(missing)}.\n"
            f"Przykłady brakujących: {sample}{more}.\n\n"
            "Wskaż folder, względem którego nazwy obrazów z XML istnieją dokładnie tak samo."
        )
        return result

    try:
        direct_image_names = {path.name for path in get_image_files(images_dir)}
        xml_basenames = {Path(name).name for name in required_names}
        extra_count = len([name for name in direct_image_names if name not in xml_basenames])
    except Exception:
        extra_count = 0

    result.update(
        {
            "ok": True,
            "message": (
                f"XML i folder obrazów są zgodne: {len(required_names)} obraz(ów) z XML "
                "ma odpowiadający plik źródłowy."
            ),
            "extra_count": int(extra_count),
        }
    )
    return result

def _refresh_dataset_creator_cta_state(self):
    btn = getattr(self, "btn_step4_create", None)
    if btn is None:
        return

    try:
        in_campaign = bool(CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False
    if not in_campaign and self._get_creator_source_mode() == "ready":
        try:
            self._pending_step4_input_training_source = None
        except Exception:
            pass
        self._schedule_step4_dataset_summary_refresh()
        try:
            btn.configure(state=tk.DISABLED)
        except Exception:
            pass
        status = getattr(self, "ds_status", None)
        if status is not None:
            try:
                self._style_training_success_label(status)
                self._set_training_widget_text(
                    status,
                    "Gotowy dataset jest już wejściem treningowym. Przejdź dalej do PZ2.",
                )
            except Exception:
                pass
        return

    info = self._resolve_dataset_creator_inputs()
    try:
        busy = bool(self._step4_has_active_operation())
    except Exception:
        busy = False

    enabled = bool(info.get("ok")) and not busy
    try:
        btn.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
    except Exception:
        pass

    try:
        if bool(info.get("ok")):
            self._pending_step4_input_training_source = info.get("training_source")
        elif not busy:
            self._pending_step4_input_training_source = None
    except Exception:
        pass
    self._schedule_step4_dataset_summary_refresh()

    status = getattr(self, "ds_status", None)
    if status is None:
        return
    try:
        if busy:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, "Przygotowanie wariantu jest w toku...")
        elif enabled:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, str(info.get("message") or "Gotowy."))
        else:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, str(info.get("message") or "Uzupełnij źródło datasetu."))
    except Exception:
        pass

def _resolve_dataset_split_inputs(self) -> dict:
    result = {
        "ok": False,
        "src": None,
        "message": (
            "Wskaż źródłowy dataset YOLO Detect dla znaków: katalog images/labels bez splitu "
            "albo gotowy katalog z data.yaml. Dataset OCR/klasyfikacyjny z manifest.json "
            "nie jest wejściem tego kroku."
        ),
    }

    src_raw = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    adapter = CharYoloDatasetSourceAdapter(
        validator=self._validate_char_yolo_split_source,
        provenance=self._format_training_source_provenance("Źródło PZ1"),
        source_stage="Z3/PZ3",
    )
    validation = adapter.validate(src_raw)
    result["message"] = validation.message or result["message"]
    result["training_source"] = validation.source
    if not validation.ok:
        return result

    result.update(
        {
            "ok": True,
            "src": validation.raw.get("src") or validation.source.dataset_dir,
            "message": validation.message or "Gotowy do utworzenia wariantu datasetu treningowego.",
            "training_source": validation.source,
        }
    )
    return result

def _refresh_dataset_split_cta_state(self):
    btn = getattr(self, "btn_step4_split", None)
    if btn is None:
        return

    info = self._resolve_dataset_split_inputs()
    try:
        busy = bool(self._step4_has_active_operation())
    except Exception:
        busy = False

    enabled = bool(info.get("ok")) and not busy
    try:
        btn.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
    except Exception:
        pass

    try:
        if bool(info.get("ok")):
            self._pending_step4_input_training_source = info.get("training_source")
        elif not busy:
            self._pending_step4_input_training_source = None
    except Exception:
        pass
    self._schedule_step4_dataset_summary_refresh()

    status = getattr(self, "split_status", None)
    if status is None:
        return
    try:
        if busy:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, "Przygotowanie wariantu jest w toku...")
        elif enabled:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, str(info.get("message") or "Gotowy."))
        else:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, str(info.get("message") or "Uzupełnij źródło datasetu."))
    except Exception:
        pass

def _can_open_pz2_after_dataset_result(self) -> bool:
    if CAMPAIGN.get_active_project_name():
        return bool(getattr(self, "_step4_train_unlocked", False))
    return True

def _show_step4_dataset_result_modal(
    self,
    *,
    title: str,
    body: str,
    allow_pz2: bool = True,
    primary_text: str = "Przejdź do PZ2",
    secondary_text: str = "Zostań w PZ1",
    preview_path: str | Path | None = None,
    summary_rows: list[tuple[str, str]] | None = None,
    hero_text: str = "",
    hero_subtitle: str = "",
    guidance_text: str = "",
    allow_delete: bool = False,
    delete_text: str = "Usuń wariant",
    return_action: bool = False,
) -> bool | str:
    parent = getattr(self, "frame", None)
    root = parent.winfo_toplevel() if parent is not None else None
    dialog = tk.Toplevel(root or parent)
    dialog.title(title)
    dialog.resizable(False, False)
    try:
        dialog.transient(root)
        dialog.grab_set()
    except Exception:
        pass

    result = {"go_pz2": False, "action": "close"}
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    field_bg = palette.get("field", "#1f1f23")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#a6a6a6")
    success = palette.get("success", "#4ec9b0")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent = palette.get("accent", "#0e639c")
    hero_bg = blend_hex_colors(success, bg, 0.84)
    row_alt = blend_hex_colors(panel_alt, bg, 0.42)

    try:
        dialog.configure(bg=bg)
    except Exception:
        pass

    container = tk.Frame(dialog, bg=bg, padx=18, pady=16)
    container.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        container,
        text=title,
        font=("Segoe UI Semibold", 14),
        justify=tk.LEFT,
        bg=bg,
        fg=fg,
    ).pack(anchor=tk.W, fill=tk.X)

    intro = str(body or "").strip()
    if intro:
        tk.Label(
            container,
            text=intro,
            justify=tk.LEFT,
            wraplength=560,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

    if hero_text or hero_subtitle:
        hero = tk.Frame(
            container,
            bg=hero_bg,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(success, bg, 0.32),
            padx=12,
            pady=10,
        )
        hero.pack(anchor=tk.W, fill=tk.X, pady=(0, 12))
        if hero_text:
            tk.Label(
                hero,
                text=hero_text,
                bg=hero_bg,
                fg=success,
                font=("Segoe UI Semibold", 15),
                anchor=tk.W,
            ).pack(anchor=tk.W, fill=tk.X)
        if hero_subtitle:
            tk.Label(
                hero,
                text=hero_subtitle,
                bg=hero_bg,
                fg=fg,
                font=("Segoe UI", 9),
                anchor=tk.W,
                wraplength=540,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, fill=tk.X, pady=(3, 0))

    rows = [(str(k or "").strip(), str(v or "").strip()) for k, v in (summary_rows or []) if str(k or "").strip()]
    if rows:
        table = tk.Frame(
            container,
            bg=field_bg,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(success, border, 0.42),
            padx=1,
            pady=1,
        )
        table.pack(anchor=tk.W, fill=tk.X, pady=(0, 12))
        for idx, (label, value) in enumerate(rows):
            row_bg = field_bg if idx % 2 == 0 else row_alt
            row = tk.Frame(table, bg=row_bg)
            row.pack(fill=tk.X)
            tk.Label(
                row,
                text=label,
                bg=row_bg,
                fg=muted,
                font=("Segoe UI Semibold", 8),
                anchor=tk.W,
                width=16,
                padx=10,
                pady=6,
            ).pack(side=tk.LEFT, fill=tk.Y)
            tk.Label(
                row,
                text=value or "—",
                bg=row_bg,
                fg=fg,
                font=("Segoe UI", 9),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=430,
                padx=8,
                pady=6,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    if guidance_text:
        tk.Label(
            container,
            text=guidance_text,
            justify=tk.LEFT,
            wraplength=560,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, fill=tk.X, pady=(0, 14))

    buttons = tk.Frame(container, bg=bg)
    buttons.pack(anchor=tk.E, fill=tk.X)

    def close(go_pz2: bool = False, action: str | None = None):
        result["go_pz2"] = bool(go_pz2)
        result["action"] = str(action or ("train" if go_pz2 else "close"))
        try:
            dialog.grab_release()
        except Exception:
            pass
        dialog.destroy()

    pz2_btn = tk.Button(
        buttons,
        text=primary_text,
        command=lambda: close(True),
        state=(tk.NORMAL if allow_pz2 else tk.DISABLED),
        bg=blend_hex_colors(success, field_bg, 0.18),
        fg=fg,
        activebackground=blend_hex_colors(success, field_bg, 0.04),
        activeforeground=fg,
        disabledforeground=muted,
        relief=tk.FLAT,
        bd=0,
        padx=14,
        pady=7,
        font=("Segoe UI Semibold", 9),
        cursor="hand2" if allow_pz2 else "arrow",
    )
    pz2_btn.pack(side=tk.RIGHT)
    tk.Button(
        buttons,
        text=secondary_text,
        command=lambda: close(False),
        bg=field_bg,
        fg=fg,
        activebackground=blend_hex_colors(field_bg, accent, 0.18),
        activeforeground=fg,
        relief=tk.FLAT,
        bd=0,
        padx=12,
        pady=7,
        font=("Segoe UI", 9),
        cursor="hand2",
    ).pack(side=tk.RIGHT, padx=(0, 8))
    if allow_delete and preview_path:
        def delete_variant_from_result_modal(path=preview_path):
            try:
                deleted = self._delete_step4_dataset_variant(path, parent=dialog)
            except Exception as exc:
                logger.exception("Nie udało się usunąć wariantu datasetu z modala podsumowania")
                try:
                    messagebox.showerror(
                        "Usuwanie wariantu",
                        f"Nie udało się usunąć wariantu datasetu:\n{exc}",
                        parent=dialog,
                    )
                except Exception:
                    pass
                deleted = False
            if deleted:
                close(False, "deleted")

        tk.Button(
            buttons,
            text=delete_text,
            command=delete_variant_from_result_modal,
            bg=blend_hex_colors(palette.get("danger", "#f48771"), field_bg, 0.20),
            fg=fg,
            activebackground=blend_hex_colors(palette.get("danger", "#f48771"), field_bg, 0.06),
            activeforeground=fg,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            font=("Segoe UI", 9),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 8))
    if preview_path:
        def open_preview_from_result_modal(path=preview_path):
            try:
                dialog.grab_release()
            except Exception:
                pass
            preview_window = open_yolo_dataset_preview(self, path)
            if preview_window is None:
                try:
                    if dialog.winfo_exists():
                        dialog.grab_set()
                except Exception:
                    pass
                return
            try:
                preview_window.transient(dialog)
            except Exception:
                pass
            try:
                preview_window.grab_set()
            except Exception:
                pass
            try:
                dialog.wait_window(preview_window)
            except Exception:
                pass
            finally:
                try:
                    if dialog.winfo_exists():
                        dialog.grab_set()
                        dialog.lift()
                except Exception:
                    pass

        tk.Button(
            buttons,
            text="Przeglądaj utworzony dataset",
            command=open_preview_from_result_modal,
            bg=field_bg,
            fg=fg,
            activebackground=blend_hex_colors(field_bg, accent, 0.18),
            activeforeground=fg,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            font=("Segoe UI", 9),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 8))

    dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
    try:
        dialog.update_idletasks()
        if root is not None:
            x = root.winfo_rootx() + max(40, (root.winfo_width() - dialog.winfo_width()) // 2)
            y = root.winfo_rooty() + max(40, (root.winfo_height() - dialog.winfo_height()) // 2)
            dialog.geometry(f"+{x}+{y}")
    except Exception:
        pass

    try:
        dialog.wait_window()
    except Exception:
        pass
    if return_action:
        return str(result.get("action") or "close")
    return bool(result.get("go_pz2"))

def _delete_step4_dataset_variant(self, dataset_path: str | Path | None, *, parent=None) -> bool:
    raw = str(dataset_path or "").strip()
    if not raw:
        return False

    dataset_root = Path(raw)
    if dataset_root.is_file() and dataset_root.name.lower() == "data.yaml":
        dataset_root = dataset_root.parent

    try:
        resolved_root = dataset_root.resolve()
    except Exception:
        resolved_root = dataset_root

    try:
        base_root = Path(self._get_datasets_base_dir()).resolve()
    except Exception:
        base_root = resolved_root.parent

    try:
        resolved_root.relative_to(base_root)
        inside_dataset_root = True
    except Exception:
        inside_dataset_root = False

    if (
        not inside_dataset_root
        or not resolved_root.exists()
        or not resolved_root.is_dir()
        or resolved_root == base_root
    ):
        try:
            messagebox.showerror(
                "Usuwanie wariantu",
                "Ten wariant nie może zostać usunięty automatycznie, bo nie wygląda na katalog "
                "utworzony w obszarze datasetów projektu.",
                parent=parent,
            )
        except Exception:
            pass
        return False

    try:
        confirmed = messagebox.askyesno(
            "Usuń wariant datasetu",
            "Usunąć utworzony wariant datasetu?\n\n"
            f"{resolved_root}\n\n"
            "Tej operacji nie można cofnąć.",
            parent=parent,
        )
    except Exception:
        confirmed = False
    if not confirmed:
        return False

    def same_dataset_path(value: str | Path | None) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        candidate = Path(text)
        if candidate.is_file() and candidate.name.lower() == "data.yaml":
            candidate = candidate.parent
        try:
            return candidate.resolve() == resolved_root
        except Exception:
            return str(candidate) == str(resolved_root)

    try:
        shutil.rmtree(resolved_root)
    except Exception as exc:
        logger.exception("Nie udało się usunąć wariantu datasetu")
        try:
            messagebox.showerror(
                "Usuwanie wariantu",
                f"Nie udało się usunąć wariantu datasetu:\n{exc}",
                parent=parent,
            )
        except Exception:
            pass
        return False

    for attr_name in ("dataset_var", "dataset_variant_var", "split_out_var"):
        try:
            var = getattr(self, attr_name, None)
            if var is not None and same_dataset_path(var.get()):
                var.set("")
        except Exception:
            pass

    try:
        source = getattr(self, "_last_training_source", None)
        if source is not None and (
            same_dataset_path(getattr(source, "dataset_dir", ""))
            or same_dataset_path(getattr(source, "yaml_path", ""))
        ):
            self._last_training_source = None
    except Exception:
        pass

    try:
        self._step4_train_unlocked = False
    except Exception:
        pass

    if CAMPAIGN.get_active_project_name():
        try:
            iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        except Exception:
            iteration_num = 1
        cleared_payload = {
            "target": CONFIG.normalize_task_target(getattr(self, "_step4_dataset_mode", "char")),
            "dataset_path": "",
            "yaml_path": "",
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "total_images": 0,
            "source_stage": "Z4/PZ1",
            "deleted": True,
            "deleted_path": str(resolved_root),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            CAMPAIGN.upsert_iteration_state(
                iteration_num=iteration_num,
                updates={"step4_dataset": cleared_payload},
            )
        except Exception:
            pass
        try:
            CAMPAIGN.upsert_iteration_artifact_bundle(
                iteration_num=iteration_num,
                updates={"step4_dataset": cleared_payload},
            )
        except Exception:
            pass

    for callback_name in (
        "_refresh_dataset_variant_choices",
        "_refresh_step4_dataset_mode_ui",
        "_refresh_step4_campaign_navigation_ui",
        "_update_training_dataset_hint",
        "_refresh_training_start_state",
        "_schedule_step4_dataset_summary_refresh",
    ):
        try:
            callback = getattr(self, callback_name, None)
            if callable(callback):
                callback()
        except Exception:
            pass

    try:
        status = getattr(self, "split_status", None) or getattr(self, "ds_status", None)
        if status is not None:
            self._style_training_success_label(status)
            self._set_training_widget_text(status, "Wariant datasetu usunięty. Utwórz nowy wariant albo wybierz inny z listy.")
    except Exception:
        pass
    return True

def _format_step4_dataset_result_path(self, dataset_path: str | Path | None) -> str:
    raw = str(dataset_path or "").strip()
    if not raw:
        return ""
    try:
        return self._format_workspace_relative_path(raw)
    except Exception:
        return raw

def _open_pz2_from_dataset_result(self, dataset_path: str | Path | None = None, target: str | None = None):
    dataset_text = str(dataset_path or "").strip()
    normalized_target = ""
    try:
        normalized_target = CONFIG.normalize_task_target(target or getattr(self, "_step4_dataset_mode", "char"))
    except Exception:
        normalized_target = str(target or getattr(self, "_step4_dataset_mode", "char") or "char").strip().lower()
        if normalized_target not in {"plate", "char"}:
            normalized_target = "char"

    if dataset_text:
        try:
            accepted = self._accept_training_input_context(
                source="pz1",
                target=normalized_target,
                dataset_path=dataset_text,
                select_training=False,
            )
        except Exception:
            accepted = False
        if not accepted:
            try:
                self.dataset_var.set(dataset_text)
            except Exception:
                pass
            try:
                self._step4_dataset_mode = normalized_target
            except Exception:
                pass
            try:
                self._step4_train_unlocked = True
            except Exception:
                pass

    try:
        self._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass
    try:
        self._step4_dataset_go_next()
    except Exception:
        try:
            self.main_nb.select(self.tab_train)
        except Exception:
            pass
    for callback_name in (
        "_refresh_dataset_variant_choices",
        "_sync_dataset_variant_selection",
        "_update_training_dataset_hint",
        "_refresh_training_start_state",
        "_refresh_training_recommendation_table",
    ):
        try:
            callback = getattr(self, callback_name, None)
            if callable(callback):
                callback()
        except Exception:
            pass

def _get_step4_augmentation_prefix(target: str) -> str:
    return "creator" if CONFIG.normalize_task_target(target) == "plate" else "split"

def _default_step4_augmentation_profile(target: str) -> AugmentationProfile:
    normalized = CONFIG.normalize_task_target(target)
    return AugmentationProfile(
        enabled=False,
        sample_size=1,
        extra_count=0,
        rotation_limit=0.0,
        translate_limit=0.0,
        scale_limit=0.0,
        brightness_limit=0.0,
        contrast_limit=0.0,
        saturation_limit=1.0,
        noise_strength=0.0,
        noise_grain_size=1,
        rain_strength=0.0,
        rain_drop_size=0.07,
        rain_drop_size_min=0.01,
        rain_drop_size_max=0.13,
        rain_vector_field_strength=0.0,
        rain_vortex_strength=0.0,
        rain_alpha=0.22,
        rain_lens_strength=0.0,
        rain_edge_mist_strength=0.0,
        rain_edge_mist_radius=0.45,
        wet_reflection_strength=0.0,
        vehicle_speed=0.0,
        night_strength=0.0,
        night_luma_min=0.56,
        night_luma_max=0.92,
        night_light_strength=0.0,
        night_bloom_strength=0.0,
        night_iso_noise_strength=0.0,
        night_light_warmth=0.35,
        flare_strength=0.0,
        overexposure_strength=0.0,
        dirt_streak_strength=0.0,
        dirt_flow_strength=0.0,
        dirt_flow_points=0,
        dirt_flow_mass_min=0.18,
        dirt_flow_mass_max=1.0,
        dirt_flow_splash_scale=0.45,
        dirt_flow_trail_length=0.55,
        dirt_flow_humidity=0.45,
        dirt_flow_stickiness=0.45,
        dirt_flow_stickiness_min=0.30,
        dirt_flow_stickiness_max=0.70,
        dirt_flow_air_angle=0.0,
        dirt_flow_wind_strength=0.45,
        dirt_flow_gravity_angle=90.0,
        dirt_flow_gravity_strength=1.0,
        dirt_flow_opacity_min=0.25,
        dirt_flow_opacity_max=0.80,
        dirt_flow_stop_on_dark_contour=False,
        dark_relief_strength=0.0,
        dark_relief_light_angle=135.0,
        light_normal_strength=0.0,
        overhang_shadow_strength=0.0,
        overhang_shadow_depth=0.60,
        overhang_shadow_skew=0.0,
        plate_reflect_gradient_strength=0.0,
        plate_reflect_glare_strength=0.0,
        plate_reflect_curve_strength=0.0,
        blur_strength=0.0,
        blur_enabled=False,
        randomness_mode="realistic",
        class_name=("plate" if normalized == "plate" else ""),
        task_target=normalized,
    )

def _ensure_step4_augmentation_profile(self, target: str) -> AugmentationProfile:
    normalized = CONFIG.normalize_task_target(target)
    profiles = getattr(self, "_step4_augmentation_profiles", None)
    if not isinstance(profiles, dict):
        profiles = {}
        self._step4_augmentation_profiles = profiles
    if normalized not in profiles:
        profiles[normalized] = _default_step4_augmentation_profile(normalized)
    return (profiles.get(normalized) or _default_step4_augmentation_profile(normalized)).normalized()

def _set_step4_augmentation_profile(self, target: str, profile: AugmentationProfile):
    normalized = CONFIG.normalize_task_target(target)
    profiles = getattr(self, "_step4_augmentation_profiles", None)
    if not isinstance(profiles, dict):
        profiles = {}
        self._step4_augmentation_profiles = profiles
    saved_profile = (profile or _default_step4_augmentation_profile(normalized)).normalized()
    profiles[normalized] = saved_profile
    prefix = _get_step4_augmentation_prefix(normalized)
    for attr_name, value in (
        (f"{prefix}_aug_enabled_var", bool(getattr(saved_profile, "extra_count", 0) or 0)),
        (f"{prefix}_aug_extra_var", int(getattr(saved_profile, "extra_count", 0) or 0)),
        (f"{prefix}_aug_sample_var", max(1, int(getattr(saved_profile, "sample_size", 1) or 1))),
    ):
        var = getattr(self, attr_name, None)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            pass
    if normalized == "plate":
        class_var = getattr(self, f"{prefix}_class_name_var", None)
        if class_var is not None:
            try:
                class_var.set(str(getattr(saved_profile, "class_name", "") or "plate"))
            except Exception:
                pass
    self._refresh_step4_augmentation_summary(normalized)

def _clear_step4_augmentation_profile(self, target: str):
    normalized = CONFIG.normalize_task_target(target)
    profile = _default_step4_augmentation_profile(normalized)
    prefix = _get_step4_augmentation_prefix(normalized)
    for attr_name, value in (
        (f"{prefix}_aug_enabled_var", False),
        (f"{prefix}_aug_sample_var", 1),
        (f"{prefix}_aug_extra_var", 0),
        (f"{prefix}_class_name_var", str(getattr(profile, "class_name", "") or "")),
    ):
        var = getattr(self, attr_name, None)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            pass
    self._set_step4_augmentation_profile(normalized, profile)

def _format_step4_augmentation_summary(profile: AugmentationProfile, target: str) -> str:
    profile = (profile or _default_step4_augmentation_profile(target)).normalized()
    if int(getattr(profile, "extra_count", 0) or 0) <= 0:
        return (
            "Syntetyki: 0. "
            "Dataset zostanie utworzony tylko z materia\u0142u \u017ar\u00f3d\u0142owego projektu."
        )
    class_suffix = ""
    if CONFIG.normalize_task_target(target) == "plate" and profile.class_name:
        class_suffix = f" Klasa: {profile.class_name}."
    randomness_suffix = f" Losowość: {describe_augmentation_randomness_mode(getattr(profile, 'randomness_mode', 'realistic'))}."
    return (
        f"Powi\u0119kszenie syntetyczne train: +{profile.extra_count} obraz\u00f3w "
        f"do aktualnego wariantu train.{class_suffix} "
        f"{randomness_suffix} "
        "Procenty splitu dotycz\u0105 bazy przed powi\u0119kszeniem; po dodaniu syntetyk\u00f3w finalny udzia\u0142 train wzro\u015bnie. "
        "Syntetyki pozostaj\u0105 tylko w tym wariancie treningowym i nie s\u0105 baz\u0105 kolejnej iteracji."
    )


def _format_step4_synthetic_count_origin(self, target: str, extra_count: int) -> str:
    normalized = CONFIG.normalize_task_target(target)
    try:
        extra_count = max(0, int(extra_count or 0))
    except Exception:
        extra_count = 0
    if extra_count <= 0:
        return "bez syntetyków"
    if normalized != "char":
        return "ręcznie"
    pending_plan = getattr(self, "_pending_character_balance_plan", None)
    try:
        planned = max(0, int(getattr(pending_plan, "planned_images", 0) or 0))
    except Exception:
        planned = 0
    if pending_plan is not None and planned > 0 and extra_count == planned:
        return "z histogramu MZ"
    if pending_plan is not None and planned > 0:
        return "ręcznie po korekcie"
    return "ręcznie"


def _apply_step4_synthetic_count_state(self, target: str, extra_count: int) -> None:
    normalized = CONFIG.normalize_task_target(target)
    prefix = _get_step4_augmentation_prefix(normalized)
    enabled_var = getattr(self, f"{prefix}_aug_enabled_var", None)
    if enabled_var is not None:
        try:
            enabled_var.set(int(extra_count or 0) > 0)
        except Exception:
            pass
    origin_var = getattr(self, f"{prefix}_aug_count_origin_var", None)
    if origin_var is not None:
        try:
            origin_var.set(_format_step4_synthetic_count_origin(self, normalized, extra_count))
        except Exception:
            pass


def _legacy_step4_augmentation_summary_details(profile: AugmentationProfile, target: str) -> str:
    parts = []
    if profile.extra_count > 0:
        parts.append(f"włączone: +{profile.extra_count} dodatkowych obrazów train")
        parts.append(f"próbka {profile.sample_size}")
        if abs(profile.rotation_limit) > 0.001:
            parts.append(f"obrót {profile.rotation_limit:+.0f}°")
        if profile.rain_strength > 0:
            parts.append(f"gęstość deszczu {profile.rain_strength:.2f}")
            parts.append(f"kropla {profile.rain_drop_size_min:.2f}-{profile.rain_drop_size_max:.2f}")
            parts.append(f"alfa {profile.rain_alpha:.2f}")
            if getattr(profile, "rain_vector_field_strength", 0.0) > 0:
                parts.append(f"pole deszczu {profile.rain_vector_field_strength:.2f}")
            if getattr(profile, "rain_vortex_strength", 0.0) > 0:
                parts.append(f"wiry {profile.rain_vortex_strength:.2f}")
            if getattr(profile, "rain_lens_strength", 0.0) > 0:
                parts.append(f"soczewka {profile.rain_lens_strength:.2f}")
            if getattr(profile, "rain_edge_mist_strength", 0.0) > 0:
                parts.append(f"mgła {profile.rain_edge_mist_strength:.2f}/{profile.rain_edge_mist_radius:.2f}")
            if profile.dirt_flow_wind_strength > 0:
                parts.append(f"wiatr deszczu {profile.dirt_flow_wind_strength:.2f}/{profile.dirt_flow_air_angle:+.0f}°")
        if profile.vehicle_speed > 0:
            parts.append(f"prędkość auta {profile.vehicle_speed:.2f}")
        if profile.night_strength > 0:
            parts.append(f"noc {profile.night_strength:.2f}")
            if getattr(profile, "night_light_strength", 0.0) > 0:
                parts.append(f"reflektor {profile.night_light_strength:.2f}")
            if getattr(profile, "night_bloom_strength", 0.0) > 0:
                parts.append(f"poświata {profile.night_bloom_strength:.2f}")
            if getattr(profile, "night_iso_noise_strength", 0.0) > 0:
                parts.append(f"ISO {profile.night_iso_noise_strength:.2f}")
        effective_dirt_points = profile.dirt_flow_points or (max(1, int(round(24 * (0.20 + 0.80 * profile.dirt_flow_strength)))) if profile.dirt_flow_strength > 0 else 0)
        if effective_dirt_points > 0:
            parts.append(f"grudki {effective_dirt_points}")
            parts.append(f"masa {profile.dirt_flow_mass_min:.2f}-{profile.dirt_flow_mass_max:.2f}")
            parts.append(f"smugi {profile.dirt_flow_trail_length:.2f}")
            parts.append(f"wilgoć {profile.dirt_flow_humidity:.2f}")
            parts.append(f"lepkość {getattr(profile, 'dirt_flow_stickiness_min', 0.30):.2f}-{getattr(profile, 'dirt_flow_stickiness_max', 0.70):.2f}")
            parts.append(f"wiatr {profile.dirt_flow_wind_strength:.2f}")
            parts.append(f"krycie {profile.dirt_flow_opacity_min:.2f}-{profile.dirt_flow_opacity_max:.2f}")
            if getattr(profile, "dirt_flow_stop_on_dark_contour", False):
                parts.append("stop na konturze")
            parts.append(f"natarcie {profile.dirt_flow_air_angle:+.0f}°")
        if profile.dark_relief_strength > 0:
            parts.append(f"relief {profile.dark_relief_strength:.2f}")
            parts.append(f"światło {profile.dark_relief_light_angle:.0f}°")
            if profile.light_normal_strength > 0:
                parts.append(f"oś kamery {profile.light_normal_strength:.2f}")
        if getattr(profile, "overhang_shadow_strength", 0.0) > 0:
            parts.append(f"cień daszka {profile.overhang_shadow_strength:.2f}/{profile.overhang_shadow_depth:.2f}, skos {getattr(profile, 'overhang_shadow_skew', 0.0):+.2f}")
        if getattr(profile, "plate_reflect_gradient_strength", 0.0) > 0 or getattr(profile, "plate_reflect_glare_strength", 0.0) > 0:
            parts.append(
                "odbicie tablicy "
                f"grad {getattr(profile, 'plate_reflect_gradient_strength', 0.0):.2f}, "
                f"odblask {getattr(profile, 'plate_reflect_glare_strength', 0.0):.2f}, "
                f"łuk {getattr(profile, 'plate_reflect_curve_strength', 0.0):.2f}"
            )
    else:
        return "Syntetyki: 0. Profil efektów nie zostanie użyty."
    if parts and "dodatkowych" in str(parts[0]):
        parts[0] = f"dataset train zostanie powi\u0119kszony o +{profile.extra_count} obraz\u00f3w"
    if CONFIG.normalize_task_target(target) == "plate" and profile.class_name:
        parts.append(f"klasa: {profile.class_name}")
    return "Powi\u0119kszenie syntetyczne train: " + ", ".join(parts)
    return "Profil syntetycznego zwiększania: " + ", ".join(parts)

def _get_step4_augmentation_source_signature(self, target: str) -> str:
    normalized = CONFIG.normalize_task_target(target)
    if normalized == "plate":
        images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
        xml_raw = str(getattr(self, "cvat_xml_var", tk.StringVar()).get() or "").strip()
        return f"plate|{images_raw}|{xml_raw}"
    raw = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    return f"char|{raw}"

def _get_step4_augmentation_source_count(self, target: str) -> int:
    normalized = CONFIG.normalize_task_target(target)
    prefix = _get_step4_augmentation_prefix(normalized)
    signature = _get_step4_augmentation_source_signature(self, normalized)
    parts = signature.split("|")
    source_incomplete = (
        len(parts) < 3 or not parts[1] or not parts[2]
        if normalized == "plate"
        else len(parts) < 2 or not parts[1]
    )
    if source_incomplete:
        try:
            setattr(self, f"{prefix}_aug_source_count", 0)
            setattr(self, f"{prefix}_aug_source_count_signature", signature)
        except Exception:
            pass
        return 0
    cache_attr = f"{prefix}_aug_source_count_signature"
    count_attr = f"{prefix}_aug_source_count"
    should_refresh = getattr(self, cache_attr, None) != signature or getattr(self, count_attr, None) is None
    if should_refresh:
        try:
            self._get_step4_augmentation_preview_images(normalized)
        except Exception:
            pass
        try:
            setattr(self, cache_attr, signature)
        except Exception:
            pass
    try:
        return max(0, int(getattr(self, count_attr, 0) or 0))
    except Exception:
        return 0

def _get_step4_augmentation_char_source_images(self) -> list[Path]:
    raw = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    if not raw:
        return []
    src = Path(raw)
    root = src.parent if src.is_file() and src.name.lower() == "data.yaml" else src
    roots = [
        root / "images" / "train",
        root / "images",
        root / "train" / "images",
        root / "images" / "val",
        root / "val" / "images",
    ]
    images: list[Path] = []
    seen: set[str] = set()
    for root_dir in roots:
        try:
            source_images = get_image_files(Path(root_dir))
        except Exception:
            continue
        for image_path in source_images:
            try:
                key = str(image_path.resolve())
            except Exception:
                key = str(image_path)
            if key in seen:
                continue
            seen.add(key)
            images.append(image_path)
    return images

def _step4_yolo_label_candidates_for_image(image_path: Path) -> list[Path]:
    candidates: list[Path] = []
    parts = image_path.parts
    for index, part in enumerate(parts):
        if str(part).lower() != "images":
            continue
        try:
            candidate = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
        except Exception:
            continue
        candidates.append(candidate)
    candidates.append(image_path.with_suffix(".txt"))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique

def _count_step4_yolo_label_objects_for_images(images: list[Path]) -> int:
    count = 0
    seen_labels: set[str] = set()
    for image_path in images:
        label_path = None
        for candidate in _step4_yolo_label_candidates_for_image(Path(image_path)):
            try:
                key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            except Exception:
                key = str(candidate)
            if key in seen_labels:
                continue
            if candidate.exists() and candidate.is_file():
                label_path = candidate
                seen_labels.add(key)
                break
        if label_path is None:
            continue
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        count += sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return max(0, int(count))

def _get_step4_augmentation_char_label_count(self, source_count: int) -> int:
    try:
        source_total = max(0, int(source_count or 0))
    except Exception:
        source_total = 0
    if source_total <= 0:
        return 0
    signature = _get_step4_augmentation_source_signature(self, "char")
    cache = getattr(self, "_step4_aug_char_label_count_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("signature") == signature
        and int(cache.get("source_count", -1) or -1) == source_total
    ):
        try:
            return max(0, int(cache.get("count", 0) or 0))
        except Exception:
            return 0
    images = _get_step4_augmentation_char_source_images(self)
    count = _count_step4_yolo_label_objects_for_images(images)
    try:
        setattr(
            self,
            "_step4_aug_char_label_count_cache",
            {
                "signature": signature,
                "source_count": source_total,
                "count": int(count),
            },
        )
    except Exception:
        pass
    return count

def _format_step4_augmentation_char_balance(
    train_count: int,
    source_count: int,
    source_char_count: int,
    extra_count: int,
) -> tuple[str, str, str]:
    if train_count <= 0:
        return "-", f"+{max(0, int(extra_count or 0))}", "-"
    try:
        source_total = max(1, int(source_count or 1))
        train_chars = int(round(max(0, int(source_char_count or 0)) * (train_count / source_total)))
    except Exception:
        train_chars = 0
    try:
        avg_chars = (train_chars / train_count) if train_count > 0 else 0.0
        extra_chars = int(round(max(0, int(extra_count or 0)) * avg_chars))
    except Exception:
        extra_chars = 0
    total_count = train_count + max(0, int(extra_count or 0))
    return (
        f"{train_count} tablic\n{train_chars} znak\u00f3w",
        f"+{max(0, int(extra_count or 0))} tablic\n+{extra_chars} znak\u00f3w",
        f"{total_count} tablic\n{train_chars + extra_chars} znak\u00f3w",
    )

def _refresh_step4_augmentation_balance(self, target: str, profile: AugmentationProfile | None = None):
    normalized = CONFIG.normalize_task_target(target)
    prefix = _get_step4_augmentation_prefix(normalized)
    current_var = getattr(self, f"{prefix}_aug_balance_current_var", None)
    extra_var = getattr(self, f"{prefix}_aug_balance_extra_var", None)
    total_var = getattr(self, f"{prefix}_aug_balance_total_var", None)
    if current_var is None or extra_var is None or total_var is None:
        return
    if profile is None:
        profile = self._get_step4_augmentation_profile(normalized)
    try:
        extra = max(0, int(getattr(profile, "extra_count", 0) or 0))
    except Exception:
        extra = 0
    if not bool(getattr(profile, "enabled", False)):
        try:
            current_var.set("-")
            extra_var.set("+0")
            total_var.set("-")
        except Exception:
            pass
        return
    source_count = _get_step4_augmentation_source_count(self, normalized)
    train_count = self._estimate_step4_augmentation_train_pool_limit(normalized, source_count) if source_count > 0 else 0
    sample_var = getattr(self, f"{prefix}_aug_sample_var", None)
    if sample_var is not None and train_count > 0:
        try:
            current_sample = int(sample_var.get() or 0)
        except Exception:
            current_sample = 0
        if current_sample != train_count:
            try:
                sample_var.set(train_count)
            except Exception:
                pass
    try:
        if normalized == "char":
            char_count = _get_step4_augmentation_char_label_count(self, source_count)
            current_text, extra_text, total_text = _format_step4_augmentation_char_balance(
                train_count,
                source_count,
                char_count,
                extra,
            )
            current_var.set(current_text)
            extra_var.set(extra_text)
            total_var.set(total_text)
        else:
            current_var.set(str(train_count) if train_count > 0 else "-")
            extra_var.set(f"+{extra}")
            total_var.set(str(train_count + extra) if train_count > 0 else "-")
    except Exception:
        pass

def _refresh_step4_augmentation_summary(self, target: str | None = None):
    targets = ["plate", "char"] if target is None else [target]
    for item in targets:
        normalized = CONFIG.normalize_task_target(item)
        prefix = _get_step4_augmentation_prefix(normalized)
        var = getattr(self, f"{prefix}_aug_summary_var", None)
        if var is None:
            continue
        try:
            profile = self._get_step4_augmentation_profile(normalized)
            var.set(_format_step4_augmentation_summary(profile, normalized))
            _refresh_step4_augmentation_balance(self, normalized, profile)
            extra_var = getattr(self, f"{prefix}_aug_extra_var", None)
            try:
                extra_count = int(float(extra_var.get())) if extra_var is not None else 0
            except Exception:
                extra_count = 0
            _apply_step4_synthetic_count_state(self, normalized, extra_count)
            configure_button = getattr(self, f"{prefix}_aug_configure_button", None)
            if configure_button is not None:
                configure_button.configure(
                    state=(tk.NORMAL if extra_count > 0 else tk.DISABLED)
                )
            self._refresh_step4_creator_decision_summary()
        except Exception:
            pass

def _get_step4_augmentation_preview_images(self, target: str) -> list[Path]:
    normalized = CONFIG.normalize_task_target(target)
    roots: list[Path] = []
    source_count = 0
    if normalized == "plate":
        images_raw = str(getattr(self, "cvat_images_var", tk.StringVar()).get() or "").strip()
        xml_raw = str(getattr(self, "cvat_xml_var", tk.StringVar()).get() or "").strip()
        if images_raw and xml_raw:
            try:
                images_dir = Path(images_raw)
                xml_path = Path(xml_raw)
                xml_names = []
                tree = ET.parse(xml_path)
                for image_el in tree.getroot().findall(".//image"):
                    image_name = self._normalize_cvat_xml_image_relpath(image_el.get("name") or "")
                    if not image_name:
                        continue
                    has_plate = False
                    for poly_el in image_el.findall("polygon"):
                        label = str(poly_el.get("label", "") or "").strip().lower()
                        points = str(poly_el.get("points", "") or "").strip()
                        if label in CONFIG.PLATE_LABELS and len([p for p in points.split(";") if "," in p]) == 4:
                            has_plate = True
                            break
                    if has_plate:
                        xml_names.append(image_name)
                exact_images = []
                for image_name in dict.fromkeys(xml_names):
                    candidate = images_dir / self._cvat_xml_relpath_to_path(image_name)
                    if candidate.exists() and candidate.is_file():
                        exact_images.append(candidate)
                source_count = len(exact_images)
                try:
                    setattr(self, "creator_aug_source_count", int(source_count))
                except Exception:
                    pass
                return exact_images[:80]
            except Exception:
                pass
        if images_raw:
            roots.append(Path(images_raw))
    else:
        raw = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
        if raw:
            src = Path(raw)
            root = src.parent if src.is_file() and src.name.lower() == "data.yaml" else src
            roots.extend([
                root / "images" / "train",
                root / "images",
                root / "train" / "images",
                root / "images" / "val",
                root / "val" / "images",
            ])

    images: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for image_path in get_image_files(Path(root)):
            key = str(image_path.resolve()) if image_path.exists() else str(image_path)
            if key in seen:
                continue
            seen.add(key)
            if len(images) < 80:
                images.append(image_path)
    source_count = len(seen)
    try:
        setattr(self, f"{_get_step4_augmentation_prefix(normalized)}_aug_source_count", int(source_count))
    except Exception:
        pass
    return images

def _estimate_step4_augmentation_train_pool_limit(self, target: str, source_count: int) -> int:
    try:
        total = max(0, int(source_count or 0))
    except Exception:
        total = 0
    if total <= 0:
        return 0
    try:
        train_pct = float(getattr(self, "train_pct", tk.DoubleVar(value=80)).get() or 80.0)
    except Exception:
        train_pct = 80.0
    try:
        val_pct = float(getattr(self, "val_pct", tk.DoubleVar(value=10)).get() or 10.0)
    except Exception:
        val_pct = 10.0
    ratios = {
        "train": max(0.0, min(100.0, train_pct)),
        "val": max(0.0, min(100.0, val_pct)),
        "test": max(0.0, 100.0 - max(0.0, min(100.0, train_pct)) - max(0.0, min(100.0, val_pct))),
    }
    try:
        split_counts = DatasetCreator._allocate_split_counts(total, ratios)
        train_count = int(split_counts.get("train", 0) or 0)
    except Exception:
        train_count = int(total * max(0.0, min(100.0, train_pct)) / 100.0)
    return max(1, min(total, train_count))

def _clamp_step4_augmentation_profile_to_pool(
    self,
    target: str,
    profile: AugmentationProfile,
    pool_limit: int,
) -> AugmentationProfile:
    profile = (profile or _default_step4_augmentation_profile(target)).normalized()
    try:
        limit = int(pool_limit or 0)
    except Exception:
        limit = 0
    if limit <= 0:
        return profile
    sample_size = max(1, min(int(getattr(profile, "sample_size", 1) or 1), limit))
    if sample_size == int(getattr(profile, "sample_size", 1) or 1):
        return profile
    return replace(profile, sample_size=sample_size).normalized()

def _open_step4_augmentation_modal(self, target: str):
    normalized = CONFIG.normalize_task_target(target)
    prefix = _get_step4_augmentation_prefix(normalized)
    extra_var = getattr(self, f"{prefix}_aug_extra_var", None)
    try:
        extra_count = int(float(extra_var.get())) if extra_var is not None else 0
    except Exception:
        extra_count = 0
    enabled_var = getattr(self, f"{prefix}_aug_enabled_var", None)
    if enabled_var is not None:
        try:
            enabled_var.set(extra_count > 0)
        except Exception:
            pass
    if extra_count <= 0:
        try:
            messagebox.showinfo(
                "Syntetyczne powiększenie train",
                "Wpisz liczbę syntetycznych obrazów train większą od zera. Wartość 0 oznacza wariant bez augmentacji.",
                parent=getattr(self, "frame", None),
            )
        except Exception:
            pass
        return
    profile = self._get_step4_augmentation_profile(normalized)
    sample_images = self._get_step4_augmentation_preview_images(normalized)
    try:
        source_count = int(getattr(self, f"{prefix}_aug_source_count", len(sample_images)) or 0)
    except Exception:
        source_count = len(sample_images)
    pool_limit = self._estimate_step4_augmentation_train_pool_limit(normalized, source_count)
    profile = self._clamp_step4_augmentation_profile_to_pool(normalized, profile, pool_limit)
    result = ask_step4_augmentation_profile(
        self.frame,
        target=normalized,
        profile=profile,
        sample_images=sample_images,
        sample_pool_limit=(pool_limit if pool_limit > 0 else None),
        install_callback=lambda: self._install_step4_albumentations_dependency(normalized),
    )
    if result is not None:
        saved_profile = self._clamp_step4_augmentation_profile_to_pool(normalized, result, pool_limit)
        self._set_step4_augmentation_profile(normalized, saved_profile)
        requested_extra = max(0, int(getattr(saved_profile, "extra_count", 0) or 0))
        status = getattr(self, "ds_status", None) if normalized == "plate" else getattr(self, "split_status", None)
        if _step4_profile_requests_augmentation(saved_profile):
            guide = (
                f"Profil zwiększania zapisany: +{requested_extra} obrazów train. "
                "Następny krok: użyj przycisku „Utwórz split treningowy”."
            )
            if status is not None:
                try:
                    self._style_training_success_label(status)
                    self._set_training_widget_text(status, guide)
                except Exception:
                    pass
            try:
                messagebox.showinfo(
                    "Syntetyczne zwiększanie datasetu",
                    (
                        "Profil zwiększania syntetycznego został zapisany.\n\n"
                        "Teraz kliknij „Utwórz split treningowy”. Program najpierw zbuduje zwykły wariant datasetu, "
                        "a następnie utworzy osobny, powiększony katalog z dopiskiem "
                        f"Aug_plus{requested_extra}.\n\n"
                        "Wygenerowane obrazy pozostaną tylko w tym wariancie treningowym. "
                        "Nie zostaną dopisane do puli obrazów projektu ani użyte jako baza kolejnej iteracji.\n\n"
                        "W podsumowaniu zobaczysz liczniki: oryginalne, syntetyczne i razem."
                    ),
                )
            except Exception:
                pass

def _refresh_step4_augmentation_dependency_state(self, target: str | None = None):
    targets = ["plate", "char"] if target is None else [target]
    installing = bool(getattr(self, "_step4_albumentations_install_running", False))
    status = get_albumentations_status()
    available = bool(status.get("available"))
    for item in targets:
        prefix = _get_step4_augmentation_prefix(item)
        label = getattr(self, f"{prefix}_aug_dep_status_lbl", None)
        button = getattr(self, f"{prefix}_aug_dep_install_btn", None)
        if label is not None:
            try:
                if installing:
                    label.configure(text="Albumentations: instalacja w toku...")
                elif available:
                    label.configure(text="Albumentations: dostępne")
                else:
                    label.configure(text="Albumentations: brak, możesz doinstalować na żądanie")
            except Exception:
                pass
        if button is not None:
            try:
                button.configure(state=(tk.DISABLED if installing or available else tk.NORMAL))
            except Exception:
                pass

def _install_step4_albumentations_dependency(self, target: str | None = None):
    if is_albumentations_available():
        self._refresh_step4_augmentation_dependency_state(target)
        try:
            messagebox.showinfo("Albumentations", "Albumentations jest już dostępne.")
        except Exception:
            pass
        return

    try:
        confirmed = messagebox.askyesno(
            "Instalacja Albumentations",
            (
                "Program doinstaluje bibliotekę Albumentations w bieżącym interpreterze Pythona.\n\n"
                "Ta operacja wymaga połączenia z internetem i może potrwać kilka minut.\n\n"
                "Czy rozpocząć instalację teraz?"
            ),
        )
    except Exception:
        confirmed = True
    if not confirmed:
        return

    if bool(getattr(self, "_step4_albumentations_install_running", False)):
        return
    self._step4_albumentations_install_running = True
    self._refresh_step4_augmentation_dependency_state(None)

    def log_line(line: str):
        text = str(line or "").strip()
        if not text:
            return
        try:
            self._append_train_log(f"[ALBUMENTATIONS] {text}")
        except Exception:
            logger.info(f"[ALBUMENTATIONS] {text}")

    def worker():
        ok, msg = install_albumentations(log_line)

        def finish():
            self._step4_albumentations_install_running = False
            self._refresh_step4_augmentation_dependency_state(None)
            try:
                if ok:
                    messagebox.showinfo("Albumentations", msg)
                else:
                    messagebox.showerror("Albumentations", msg)
            except Exception:
                pass

        self._ui(finish)

    threading.Thread(target=worker, daemon=True).start()

def _read_step4_var(self, name: str, default=None):
    var = getattr(self, name, None)
    if var is None:
        return default
    try:
        return var.get()
    except Exception:
        return default

def _get_step4_augmentation_profile(self, target: str) -> AugmentationProfile:
    normalized_target = CONFIG.normalize_task_target(target)
    prefix = _get_step4_augmentation_prefix(normalized_target)

    def as_int(name: str, default: int) -> int:
        try:
            return int(float(_read_step4_var(self, name, default)))
        except Exception:
            return int(default)

    def as_float(name: str, default: float) -> float:
        try:
            return float(_read_step4_var(self, name, default))
        except Exception:
            return float(default)

    class_name = ""
    if normalized_target == "plate":
        class_name = str(_read_step4_var(self, f"{prefix}_class_name_var", "plate") or "plate").strip()

    profiles = getattr(self, "_step4_augmentation_profiles", None)
    if isinstance(profiles, dict) and normalized_target in profiles:
        profile = (profiles.get(normalized_target) or _default_step4_augmentation_profile(normalized_target)).normalized()
        extra_count = as_int(f"{prefix}_aug_extra_var", int(getattr(profile, "extra_count", 0) or 0))
        return replace(
            profile,
            enabled=extra_count > 0,
            sample_size=as_int(f"{prefix}_aug_sample_var", int(getattr(profile, "sample_size", 1) or 1)),
            extra_count=extra_count,
            class_name=(class_name if normalized_target == "plate" else str(getattr(profile, "class_name", "") or "")),
            task_target=normalized_target,
        ).normalized()

    extra_count = as_int(f"{prefix}_aug_extra_var", 0)
    return AugmentationProfile(
        enabled=extra_count > 0,
        sample_size=as_int(f"{prefix}_aug_sample_var", 1),
        extra_count=extra_count,
        rotation_limit=as_float(f"{prefix}_aug_rotation_var", 0.0),
        translate_limit=0.0,
        scale_limit=0.0,
        brightness_limit=as_float(f"{prefix}_aug_brightness_var", 0.0),
        contrast_limit=as_float(f"{prefix}_aug_contrast_var", as_float(f"{prefix}_aug_brightness_var", 0.0)),
        saturation_limit=as_float(f"{prefix}_aug_saturation_var", 1.0),
        noise_strength=as_float(f"{prefix}_aug_noise_var", 0.0),
        noise_grain_size=as_int(f"{prefix}_aug_noise_grain_var", 1),
        rain_strength=as_float(f"{prefix}_aug_rain_var", 0.0),
        rain_drop_size=as_float(f"{prefix}_aug_rain_drop_size_var", 0.07),
        rain_drop_size_min=as_float(f"{prefix}_aug_rain_drop_size_min_var", as_float(f"{prefix}_aug_rain_drop_size_var", 0.07)),
        rain_drop_size_max=as_float(f"{prefix}_aug_rain_drop_size_max_var", as_float(f"{prefix}_aug_rain_drop_size_var", 0.07)),
        rain_vector_field_strength=as_float(f"{prefix}_aug_rain_vector_field_var", 0.0),
        rain_vortex_strength=as_float(f"{prefix}_aug_rain_vortex_var", 0.0),
        rain_alpha=as_float(f"{prefix}_aug_rain_alpha_var", 0.22),
        rain_lens_strength=as_float(f"{prefix}_aug_rain_lens_var", 0.0),
        rain_edge_mist_strength=as_float(f"{prefix}_aug_rain_edge_mist_var", 0.0),
        rain_edge_mist_radius=as_float(f"{prefix}_aug_rain_edge_mist_radius_var", 0.45),
        wet_reflection_strength=as_float(f"{prefix}_aug_wet_reflection_var", 0.0),
        vehicle_speed=as_float(f"{prefix}_aug_vehicle_speed_var", 0.0),
        night_strength=as_float(f"{prefix}_aug_night_var", 0.0),
        night_luma_min=as_float(f"{prefix}_aug_night_luma_min_var", 0.56),
        night_luma_max=as_float(f"{prefix}_aug_night_luma_max_var", 0.92),
        night_light_strength=as_float(f"{prefix}_aug_night_light_var", 0.0),
        night_bloom_strength=as_float(f"{prefix}_aug_night_bloom_var", 0.0),
        night_iso_noise_strength=as_float(f"{prefix}_aug_night_iso_noise_var", 0.0),
        night_light_warmth=as_float(f"{prefix}_aug_night_warmth_var", 0.35),
        flare_strength=as_float(f"{prefix}_aug_flare_var", 0.0),
        overexposure_strength=as_float(f"{prefix}_aug_overexposure_var", 0.0),
        dirt_streak_strength=as_float(f"{prefix}_aug_dirt_streak_var", 0.0),
        dirt_flow_strength=as_float(f"{prefix}_aug_dirt_flow_var", 0.0),
        dirt_flow_points=as_int(f"{prefix}_aug_dirt_flow_points_var", 0),
        dirt_flow_mass_min=as_float(f"{prefix}_aug_dirt_flow_mass_min_var", 0.18),
        dirt_flow_mass_max=as_float(f"{prefix}_aug_dirt_flow_mass_max_var", 1.0),
        dirt_flow_splash_scale=as_float(f"{prefix}_aug_dirt_flow_splash_scale_var", 0.45),
        dirt_flow_trail_length=as_float(f"{prefix}_aug_dirt_flow_trail_length_var", 0.55),
        dirt_flow_humidity=as_float(f"{prefix}_aug_dirt_flow_humidity_var", 0.45),
        dirt_flow_stickiness=as_float(f"{prefix}_aug_dirt_flow_stickiness_var", 0.45),
        dirt_flow_stickiness_min=as_float(f"{prefix}_aug_dirt_flow_stickiness_min_var", as_float(f"{prefix}_aug_dirt_flow_stickiness_var", 0.30)),
        dirt_flow_stickiness_max=as_float(f"{prefix}_aug_dirt_flow_stickiness_max_var", as_float(f"{prefix}_aug_dirt_flow_stickiness_var", 0.70)),
        dirt_flow_air_angle=as_float(f"{prefix}_aug_dirt_flow_air_angle_var", 0.0),
        dirt_flow_wind_strength=as_float(f"{prefix}_aug_dirt_flow_wind_strength_var", 0.45),
        dirt_flow_gravity_angle=as_float(f"{prefix}_aug_dirt_flow_gravity_angle_var", 90.0),
        dirt_flow_gravity_strength=as_float(f"{prefix}_aug_dirt_flow_gravity_strength_var", 1.0),
        dirt_flow_opacity_min=as_float(f"{prefix}_aug_dirt_flow_opacity_min_var", 0.25),
        dirt_flow_opacity_max=as_float(f"{prefix}_aug_dirt_flow_opacity_max_var", 0.80),
        dirt_flow_stop_on_dark_contour=bool(_read_step4_var(self, f"{prefix}_aug_dirt_flow_stop_on_contour_var", False)),
        dark_relief_strength=as_float(f"{prefix}_aug_dark_relief_var", 0.0),
        dark_relief_light_angle=as_float(f"{prefix}_aug_dark_relief_light_angle_var", 135.0),
        light_normal_strength=as_float(f"{prefix}_aug_light_normal_var", 0.0),
        overhang_shadow_strength=as_float(f"{prefix}_aug_overhang_shadow_var", 0.0),
        overhang_shadow_depth=as_float(f"{prefix}_aug_overhang_shadow_depth_var", 0.60),
        overhang_shadow_skew=as_float(f"{prefix}_aug_overhang_shadow_skew_var", 0.0),
        plate_reflect_gradient_strength=as_float(f"{prefix}_aug_plate_reflect_gradient_var", 0.0),
        plate_reflect_glare_strength=as_float(f"{prefix}_aug_plate_reflect_glare_var", 0.0),
        plate_reflect_curve_strength=as_float(f"{prefix}_aug_plate_reflect_curve_var", 0.0),
        blur_strength=as_float(f"{prefix}_aug_blur_strength_var", 0.18 if bool(_read_step4_var(self, f"{prefix}_aug_blur_var", False)) else 0.0),
        blur_enabled=(
            bool(_read_step4_var(self, f"{prefix}_aug_blur_var", False))
            or as_float(f"{prefix}_aug_blur_strength_var", 0.0) > 0.001
        ),
        randomness_mode="realistic",
        class_name=class_name,
        task_target=normalized_target,
    ).normalized()

def _step4_profile_requests_augmentation(profile: AugmentationProfile | None) -> bool:
    profile = (profile or AugmentationProfile()).normalized()
    return int(profile.extra_count or 0) > 0

def _build_step4_augmented_dataset_dir(
    self,
    dataset_path: str | Path,
    profile: AugmentationProfile,
    target: str | None = None,
) -> Path:
    source_dir = Path(dataset_path)
    normalized_target = CONFIG.normalize_task_target(target)
    try:
        destination_root = (
            Path(getattr(self, "_campaign_datasets_dir"))
            if getattr(self, "_campaign_datasets_dir", None)
            else Path(CONFIG.get_datasets_dir(normalized_target))
        )
    except Exception:
        destination_root = source_dir.parent
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        destination_root = source_dir.parent
    extra_count = max(0, int(getattr(profile, "extra_count", 0) or 0))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"Aug_plus{extra_count}_{timestamp}" if extra_count > 0 else f"Aug_{timestamp}"
    base_name = f"{source_dir.name}_{suffix}"
    candidate = destination_root / base_name
    index = 2
    while candidate.exists():
        candidate = destination_root / f"{base_name}_{index:02d}"
        index += 1
    return candidate

def _write_step4_augmentation_scope_manifest(
    self,
    *,
    source_dataset_dir: Path,
    augmented_dataset_dir: Path,
    target: str,
    requested_extra: int,
    generated: int,
    base_total: int,
    augmented_total: int,
) -> dict:
    payload = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "target": CONFIG.normalize_task_target(target),
        "source_dataset_dir": str(Path(source_dataset_dir).resolve()),
        "augmented_dataset_dir": str(Path(augmented_dataset_dir).resolve()),
        "split_scope": "train_only",
        "synthetic_scope": "training_dataset_only",
        "synthetic_images_project_base": False,
        "synthetic_images_next_iteration_base": False,
        "validation_test_original": True,
        "requested_extra": int(requested_extra or 0),
        "generated": int(generated or 0),
        "base_total": int(base_total or 0),
        "augmented_total": int(augmented_total or 0),
    }
    try:
        Path(augmented_dataset_dir, "augmentation_scope.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać manifestu zakresu augmentacji: {exc}")

    try:
        manifest_path = self._plate_dataset_source_manifest_path(Path(augmented_dataset_dir))
        if manifest_path.exists():
            try:
                source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                source_manifest = {}
            if not isinstance(source_manifest, dict):
                source_manifest = {}
            source_manifest.update(
                {
                    "dataset_dir": str(Path(augmented_dataset_dir).resolve()),
                    "source_kind": f"{str(source_manifest.get('source_kind') or 'z4_dataset')}_augmented_train",
                    "base_dataset_dir": str(Path(source_dataset_dir).resolve()),
                    "synthetic_scope": "training_dataset_only",
                    "synthetic_images_project_base": False,
                    "synthetic_images_next_iteration_base": False,
                    "validation_test_original": True,
                    "synthetic_requested_extra": int(requested_extra or 0),
                    "synthetic_generated": int(generated or 0),
                }
            )
            manifest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug(f"Nie udało się uzupełnić manifestu źródła datasetu augmentowanego: {exc}")
    return payload

def _step4_balance_plan_matches_dataset(plan: object, source_dir: Path) -> bool:
    base_dataset = str(getattr(plan, "base_dataset", "") or "").strip()
    if not base_dataset:
        return False
    try:
        return Path(base_dataset).resolve() == Path(source_dir).resolve()
    except Exception:
        return os.path.normcase(os.path.normpath(base_dataset)) == os.path.normcase(os.path.normpath(str(source_dir)))


def _step4_plan_attr(plan: object, key: str, default=None):
    if plan is None:
        return default
    if isinstance(plan, dict):
        return plan.get(key, default)
    return getattr(plan, key, default)


def _step4_distribution_deficits(distribution) -> dict[str, int]:
    rows = getattr(distribution, "classes", []) or []
    deficits: dict[str, int] = {}
    for row in rows:
        symbol = str(getattr(row, "symbol", "") or "").strip()
        if not symbol:
            continue
        count = max(0, int(getattr(row, "deficit_count", 0) or 0))
        if count > 0:
            deficits[symbol] = count
    return deficits


def _step4_distribution_diversity_warnings(distribution) -> list[str]:
    rows = getattr(distribution, "classes", []) or []
    warnings: list[str] = []
    for row in rows:
        symbol = str(getattr(row, "symbol", "") or "").strip()
        if not symbol:
            continue
        if str(getattr(row, "diversity_status", "") or "").upper() == "LOW_DIVERSITY":
            warnings.append(symbol)
    return warnings


def _step4_mz_completion_status_from_distribution(distribution) -> tuple[str, dict[str, int], list[str]]:
    deficits = _step4_distribution_deficits(distribution)
    diversity_warnings = _step4_distribution_diversity_warnings(distribution)
    if deficits:
        return "TARGET_NOT_REACHED", deficits, diversity_warnings
    if diversity_warnings:
        return "REPRESENTATION_OK_WITH_DIVERSITY_WARNING", deficits, diversity_warnings
    return "REPRESENTATION_OK", deficits, diversity_warnings


def _step4_mz_execution_status(generated: int, requested: int, *, generation_complete: bool | None = None) -> str:
    generated = max(0, int(generated or 0))
    requested = max(0, int(requested or 0))
    if generation_complete is None:
        generation_complete = bool(requested <= 0 or generated >= requested)
    if bool(generation_complete):
        return "COMPLETED"
    if generated > 0:
        return "PARTIAL"
    return "NO_OUTPUT"


def _step4_mz_freeze_status(val_test_guard: dict | None) -> str:
    return "UNCHANGED" if bool((val_test_guard or {}).get("unchanged")) else "VAL_TEST_CHANGED"


def _step4_mz_ready_for_training(
    *,
    execution_status: str,
    representation_status: str,
    freeze_status: str,
) -> bool:
    return (
        str(execution_status or "").upper() == "COMPLETED"
        and str(freeze_status or "").upper() == "UNCHANGED"
        and str(representation_status or "").upper()
        in {"REPRESENTATION_OK", "REPRESENTATION_OK_WITH_DIVERSITY_WARNING"}
    )


def _step4_mz_final_completion_status(
    *,
    representation_status: str,
    freeze_status: str,
    execution_status: str,
    plan_feasible: bool = True,
    generated: int = 0,
    deficits: dict | None = None,
) -> str:
    if str(freeze_status or "").upper() != "UNCHANGED":
        return "VAL_TEST_CHANGED"
    if not bool(plan_feasible) and max(0, int(generated or 0)) <= 0 and dict(deficits or {}):
        return "PLAN_NOT_FEASIBLE"
    if str(execution_status or "").upper() not in {"", "COMPLETED"}:
        return "PARTIAL_AUGMENTATION"
    return str(representation_status or "").strip() or "TARGET_NOT_REACHED"


def _format_step4_mz_plan_status(plan: object | None) -> str:
    if plan is None:
        return "Nie ustawiono syntetycznego podbicia znaków. Otwórz histogram MZ albo wpisz liczbę syntetyków ręcznie."
    target_count = int(_step4_plan_attr(plan, "target_count", 0) or 0)
    planned_images = int(_step4_plan_attr(plan, "planned_images", 0) or 0)
    deficits = dict(_step4_plan_attr(plan, "deficit_by_symbol", {}) or {})
    requested_extras = dict(_step4_plan_attr(plan, "requested_extra_by_symbol", {}) or {})
    if requested_extras:
        requested_text = ", ".join(f"{key}+{value}" for key, value in sorted(requested_extras.items()))
        remaining = dict(_step4_plan_attr(plan, "predicted_deficit_after", {}) or {})
        suffix = ""
        if remaining:
            suffix = " Część znaków nie ma źródła w train: " + ", ".join(sorted(remaining.keys())) + "."
        return (
            f"Histogram MZ: {requested_text}. "
            f"Planowane syntetyczne obrazy train: +{planned_images}.{suffix}"
        )
    if not deficits:
        return f"Miarka referencyjna: {target_count} przykładów znaku w train. Reprezentacja jest wystarczająca. Syntetyki: +0."
    status = str(_step4_plan_attr(plan, "completion_status", "") or "").strip()
    suffix = "" if status != "PLAN_NOT_FEASIBLE" else " Przy aktualnym materiale nie da się osiągnąć tej miarki syntetycznie."
    return (
        f"Miarka referencyjna: {target_count} przykładów znaku w train. "
        f"Do uzupełnienia: {', '.join(deficits.keys())}. "
        f"Syntetyczne uzupełnienie train: +{planned_images} obrazów.{suffix}"
    )


def _format_step4_mz_completion_label(status: str, deficits: dict | None = None, warnings: list | tuple | None = None) -> str:
    normalized = str(status or "").strip().upper()
    deficits = dict(deficits or {})
    warnings = list(warnings or [])
    if normalized == "REPRESENTATION_OK":
        return "OK - próg AUTO spełniony"
    if normalized == "REPRESENTATION_OK_WITH_DIVERSITY_WARNING":
        return "OK - próg spełniony, ale sprawdź różnorodność: " + (", ".join(map(str, warnings)) or "wybrane klasy")
    if normalized == "PLAN_NOT_FEASIBLE":
        return "Nieosiągalna przy aktualnym materiale"
    if normalized == "TARGET_NOT_REACHED":
        return "Nie osiągnięto progu: " + (", ".join(f"{key}: {value}" for key, value in sorted(deficits.items())) or "pozostały braki")
    if normalized == "VAL_TEST_CHANGED":
        return "Błąd freeze - zmienił się split val/test"
    return "Status zapisany w manifeście"


def _invalidate_pending_character_balance_plan(self, message: str | None = None) -> None:
    _clear_pending_character_balance_plan(self)
    for attr_name, value in (
        ("split_aug_enabled_var", False),
        ("split_aug_extra_var", 0),
        ("split_aug_sample_var", 1),
    ):
        var = getattr(self, attr_name, None)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            pass
    status_var = getattr(self, "split_mz_representation_status_var", None)
    if status_var is not None:
        try:
            status_var.set(message or _format_step4_mz_plan_status(None))
        except Exception:
            pass
    origin_var = getattr(self, "split_aug_count_origin_var", None)
    if origin_var is not None:
        try:
            origin_var.set("bez syntetyków")
        except Exception:
            pass


def _step4_pending_plan_fingerprint_matches(plan: object, source_dir: Path) -> bool:
    expected = str(_step4_plan_attr(plan, "base_dataset_fingerprint_sha256", "") or "").strip()
    if not expected:
        return True
    try:
        current = str(build_character_dataset_file_fingerprint(source_dir).get("sha256") or "").strip()
    except Exception:
        return False
    return bool(current and current == expected)


def _take_pending_character_balance_plan(self, source_dir: Path) -> tuple[object | None, dict, bool]:
    plan = getattr(self, "_pending_character_balance_plan", None)
    if (
        plan is None
        or not _step4_balance_plan_matches_dataset(plan, source_dir)
        or not _step4_pending_plan_fingerprint_matches(plan, source_dir)
    ):
        return None, {}, False
    return plan, {}, True

def _clear_pending_character_balance_plan(self) -> None:
    for attr_name in (
        "_pending_character_balance_plan",
        "_pending_character_balance_real_sources",
        "_pending_character_balance_ratio_key",
    ):
        try:
            setattr(self, attr_name, None)
        except Exception:
            pass

def _summarize_character_real_source_search(real_sources: dict) -> dict:
    rows = dict(real_sources or {})
    total_available = 0
    by_symbol: dict[str, int] = {}
    for symbol, row in rows.items():
        if not isinstance(row, dict):
            continue
        count = int(row.get("available_unused_real_sources", 0) or 0)
        by_symbol[str(symbol)] = count
        total_available += count
    return {
        "symbols": len(by_symbol),
        "available_unused_real_sources": total_available,
        "real_source_candidates_found": total_available,
        "real_sources_added": 0,
        "by_symbol": by_symbol,
        "policy": "reported_only_requires_user_selection",
    }


def _finalize_step4_mz_representation_variant(
    self,
    *,
    dataset_path: str | Path,
    plan: object,
    profile: AugmentationProfile,
    counts: dict | None = None,
    generated: int = 0,
    requested_extra: int = 0,
    augmented: bool = False,
    pending_real_sources: dict | None = None,
) -> dict:
    dataset_dir = Path(dataset_path)
    counts = dict(counts or self._get_dataset_split_image_counts(dataset_dir))
    generated = max(0, int(generated or 0))
    requested_extra = max(0, int(requested_extra or 0))
    target_ratio = float(_step4_plan_attr(plan, "target_ratio", 0.50) or 0.50)
    base_fingerprint = build_character_dataset_file_fingerprint(dataset_dir)
    val_test_before = build_character_dataset_file_fingerprint(dataset_dir, splits=("val", "test"))
    before_distribution = analyze_character_class_distribution(dataset_dir, target_ratio=target_ratio)
    after_distribution = analyze_character_class_distribution(dataset_dir, target_ratio=target_ratio)
    val_test_after = build_character_dataset_file_fingerprint(dataset_dir, splits=("val", "test"))
    val_test_guard = compare_character_val_test_unchanged(val_test_before, val_test_after)
    representation_status, remaining_deficits, diversity_warnings = _step4_mz_completion_status_from_distribution(after_distribution)
    freeze_status = _step4_mz_freeze_status(val_test_guard)
    plan_feasible = bool(_step4_plan_attr(plan, "feasible", True))
    if not plan_feasible and remaining_deficits and generated <= 0:
        execution_status = "NOT_RUN"
    else:
        execution_status = _step4_mz_execution_status(generated, requested_extra)
    completion_status = _step4_mz_final_completion_status(
        representation_status=representation_status,
        freeze_status=freeze_status,
        execution_status=execution_status,
        plan_feasible=plan_feasible,
        generated=generated,
        deficits=remaining_deficits,
    )
    ready_for_training = _step4_mz_ready_for_training(
        execution_status=execution_status,
        representation_status=representation_status,
        freeze_status=freeze_status,
    )

    analysis_dir = dataset_dir / "analysis"
    before_refs = save_character_distribution_artifacts(
        before_distribution,
        analysis_dir,
        prefix="character_class_distribution_before",
    )
    after_refs = save_character_distribution_artifacts(
        after_distribution,
        analysis_dir,
        prefix="character_class_distribution_after",
    )

    manifest = build_character_training_variant_manifest(
        base_dataset=dataset_dir,
        before_distribution=before_refs.get("json", {}).get("path", ""),
        after_distribution=after_refs.get("json", {}).get("path", ""),
        plan=plan,
        sources={
            "real": int(counts.get("train", 0) or 0),
            "added_real": 0,
            "augmented_real": generated,
            "synthetic": 0,
            "augmented_synthetic": 0,
        },
        base_dataset_sha_or_fingerprint=base_fingerprint,
        val_test_unchanged=val_test_guard,
        augmentation_mode="mz_auto_representation",
        requested_images=requested_extra,
        planned_images=max(0, int(_step4_plan_attr(plan, "planned_images", requested_extra) or 0)),
        generated_images=generated,
        completion_status=completion_status,
    )
    manifest["augmentation"]["seed"] = int(getattr(profile, "seed", 42) or 42)
    manifest["augmentation"]["randomness_mode"] = str(getattr(profile, "randomness_mode", "") or "")
    manifest["deficit_after"] = dict(remaining_deficits)
    manifest["diversity_warnings"] = list(diversity_warnings)
    manifest["approved_balance_plan"] = True
    manifest["execution_status"] = execution_status
    manifest["representation_status"] = representation_status
    manifest["freeze_status"] = freeze_status
    manifest["ready_for_training"] = bool(ready_for_training)
    (dataset_dir / "mz_training_variant_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok = bool(ready_for_training)
    if completion_status == "REPRESENTATION_OK":
        status_message = "Reprezentacja MZ: OK"
    elif completion_status == "REPRESENTATION_OK_WITH_DIVERSITY_WARNING":
        status_message = "Reprezentacja MZ: OK, ale sprawdź różnorodność klas " + ", ".join(diversity_warnings)
    elif completion_status == "PLAN_NOT_FEASIBLE":
        status_message = "Reprezentacja MZ: nieosiągalna przy aktualnym materiale"
    else:
        status_message = "Reprezentacja MZ: wymaga poprawy"
    if remaining_deficits:
        missing_text = ", ".join(f"{symbol}: {count}" for symbol, count in sorted(remaining_deficits.items()))
        message = f"{status_message}. Pozostałe braki: {missing_text}."
    else:
        message = f"{status_message}. Próg AUTO: {int(_step4_plan_attr(plan, 'target_count', 0) or 0)}."
    if not bool(val_test_guard.get("unchanged")):
        message += " Walidacja freeze: split val/test zmienił się podczas finalizacji."

    return {
        "ok": ok,
        "dataset_path": str(dataset_dir),
        "message": message,
        "counts": counts,
        "augmentation_stats": {
            "enabled": bool(augmented),
            "generated": generated,
            "requested": requested_extra,
            "completion_ok": ok,
            "completion_status": completion_status,
        },
        "augmented": bool(augmented),
        "base_total": int(sum(int(counts.get(key, 0) or 0) for key in ("train", "val", "test"))),
        "generated": generated,
        "requested_extra": requested_extra,
        "augmented_total": int(counts.get("total", 0) or 0),
        "generation_complete": bool(generated >= requested_extra),
        "status_message": message,
        "synthetic_scope": "training_dataset_only",
        "used_pending_character_balance_plan": True,
        "synthetic_images_project_base": False,
        "synthetic_images_next_iteration_base": False,
        "mz_training_variant_manifest": manifest,
    }


def _create_step4_augmented_dataset_variant(
    self,
    *,
    source_dataset_path: str | Path,
    target: str,
    profile: AugmentationProfile,
    progress_var_name: str,
    status_attr_name: str,
    balance_plan_override: object | None = None,
    pending_real_sources_override: dict | None = None,
    augmentation_mode: str | None = None,
) -> dict:
    source_dir = Path(source_dataset_path)
    profile = (profile or AugmentationProfile()).normalized()
    normalized_target = CONFIG.normalize_task_target(target)
    auto_representation_mode = str(augmentation_mode or "").strip() == "mz_auto_representation"
    balance_plan = None
    pending_real_sources: dict = {}
    used_pending_balance_plan = False
    if not _step4_profile_requests_augmentation(profile):
        if normalized_target == "char" and auto_representation_mode:
            try:
                if balance_plan_override is not None:
                    balance_plan = balance_plan_override
                    pending_real_sources = dict(pending_real_sources_override or {})
                else:
                    balance_plan, pending_real_sources, _used = _take_pending_character_balance_plan(
                        self,
                        source_dir,
                    )
                if balance_plan is None:
                    return {
                        "ok": False,
                        "dataset_path": str(source_dir),
                        "message": "PLAN_MISSING: brak planu reprezentacji MZ dla finalnego splitu.",
                        "counts": self._get_dataset_split_image_counts(source_dir),
                        "augmentation_stats": {},
                        "augmented": False,
                    }
                if (
                    not _step4_balance_plan_matches_dataset(balance_plan, source_dir)
                    or not _step4_pending_plan_fingerprint_matches(balance_plan, source_dir)
                ):
                    return {
                        "ok": False,
                        "dataset_path": str(source_dir),
                        "message": "PLAN_DATASET_MISMATCH: plan MZ nie pasuje do datasetu przekazanego do augmentacji.",
                        "counts": self._get_dataset_split_image_counts(source_dir),
                        "augmentation_stats": {},
                        "augmented": False,
                    }
                planned_extra = max(0, int(_step4_plan_attr(balance_plan, "planned_images", 0) or 0))
                if not bool(_step4_plan_attr(balance_plan, "feasible", True)) and planned_extra <= 0:
                    return _finalize_step4_mz_representation_variant(
                        self,
                        dataset_path=source_dir,
                        plan=balance_plan,
                        profile=profile,
                        counts=self._get_dataset_split_image_counts(source_dir),
                        generated=0,
                        requested_extra=planned_extra,
                        augmented=False,
                        pending_real_sources=pending_real_sources,
                    )
                if planned_extra > 0:
                    profile = replace(profile, enabled=True, extra_count=planned_extra).normalized()
                    used_pending_balance_plan = True
                else:
                    return _finalize_step4_mz_representation_variant(
                        self,
                        dataset_path=source_dir,
                        plan=balance_plan,
                        profile=profile,
                        counts=self._get_dataset_split_image_counts(source_dir),
                        generated=0,
                        requested_extra=0,
                        augmented=False,
                        pending_real_sources=pending_real_sources,
                    )
            except Exception as exc:
                logger.exception("Nie udało się sfinalizować wariantu reprezentacji MZ bez augmentacji")
                return {
                    "ok": False,
                    "dataset_path": str(source_dir),
                    "message": f"Nie udało się sfinalizować wariantu reprezentacji MZ: {exc}",
                    "counts": self._get_dataset_split_image_counts(source_dir),
                    "augmentation_stats": {},
                    "augmented": False,
                }
        return {
            "ok": True,
            "dataset_path": str(source_dir),
            "message": "",
            "counts": self._get_dataset_split_image_counts(source_dir),
            "augmentation_stats": {},
            "augmented": False,
        }

    requested_extra = max(0, int(getattr(profile, "extra_count", 0) or 0))
    before_distribution = None
    base_dataset_fingerprint: dict = {}
    val_test_before: dict = {}
    if normalized_target == "char":
        try:
            if balance_plan is None:
                if balance_plan_override is not None:
                    balance_plan = balance_plan_override
                    pending_real_sources = dict(pending_real_sources_override or {})
                    used_pending_balance_plan = True
                else:
                    balance_plan, pending_real_sources, used_pending_balance_plan = _take_pending_character_balance_plan(
                        self,
                        source_dir,
                    )
            if auto_representation_mode and balance_plan is None:
                return {
                    "ok": False,
                    "dataset_path": str(source_dir),
                    "message": "PLAN_MISSING: brak planu reprezentacji MZ dla finalnego splitu.",
                    "counts": self._get_dataset_split_image_counts(source_dir),
                    "augmentation_stats": {},
                    "augmented": False,
                }
            if balance_plan is not None and (
                not _step4_balance_plan_matches_dataset(balance_plan, source_dir)
                or not _step4_pending_plan_fingerprint_matches(balance_plan, source_dir)
            ):
                return {
                    "ok": False,
                    "dataset_path": str(source_dir),
                    "message": "PLAN_DATASET_MISMATCH: plan MZ nie pasuje do datasetu przekazanego do augmentacji.",
                    "counts": self._get_dataset_split_image_counts(source_dir),
                    "augmentation_stats": {},
                    "augmented": False,
                }
            target_ratio = float(_step4_plan_attr(balance_plan, "target_ratio", 0.50) or 0.50) if balance_plan is not None else 0.50
            before_distribution = analyze_character_class_distribution(source_dir, target_ratio=target_ratio)
            if balance_plan is not None:
                planned_extra = max(0, int(_step4_plan_attr(balance_plan, "planned_images", 0) or 0))
                if auto_representation_mode and not bool(_step4_plan_attr(balance_plan, "feasible", True)) and planned_extra <= 0:
                    return _finalize_step4_mz_representation_variant(
                        self,
                        dataset_path=source_dir,
                        plan=balance_plan,
                        profile=profile,
                        counts=self._get_dataset_split_image_counts(source_dir),
                        generated=0,
                        requested_extra=planned_extra,
                        augmented=False,
                        pending_real_sources=pending_real_sources,
                    )
                if auto_representation_mode and planned_extra > 0 and requested_extra != planned_extra:
                    profile = replace(profile, enabled=True, extra_count=planned_extra).normalized()
                    requested_extra = planned_extra
            base_dataset_fingerprint = build_character_dataset_file_fingerprint(source_dir)
            val_test_before = build_character_dataset_file_fingerprint(source_dir, splits=("val", "test"))
        except Exception as exc:
            logger.exception("Nie udało się przygotować uzupełnienia reprezentacji MZ")
            return {
                "ok": False,
                "dataset_path": str(source_dir),
                "message": f"Nie udało się przygotować uzupełnienia reprezentacji MZ: {exc}",
                "counts": self._get_dataset_split_image_counts(source_dir),
                "augmentation_stats": {},
                "augmented": False,
            }
    augmented_dir = self._build_step4_augmented_dataset_dir(source_dir, profile, target=target)
    try:
        self._ui(
            lambda requested=requested_extra: self._set_training_widget_text(
                getattr(self, status_attr_name, None),
                f"Krok 2/2: tworzę powiększony wariant datasetu (+{requested} syntetycznych obrazów train)...",
            )
        )
        shutil.copytree(source_dir, augmented_dir)
        ok_yaml, yaml_msg, yaml_changed = ensure_yolo_dataset_yaml_points_to_root(augmented_dir)
        if not ok_yaml:
            raise RuntimeError(yaml_msg)
        if yaml_changed:
            logger.info(f"Poprawiono data.yaml wariantu augmentowanego: {augmented_dir}")
    except Exception as exc:
        logger.exception("Nie udało się utworzyć katalogu powiększonego wariantu datasetu")
        try:
            if augmented_dir.exists():
                shutil.rmtree(augmented_dir, ignore_errors=True)
        except Exception:
            pass
        return {
            "ok": False,
            "dataset_path": str(source_dir),
            "message": f"Nie udało się utworzyć powiększonego wariantu datasetu: {exc}",
            "counts": self._get_dataset_split_image_counts(source_dir),
            "augmentation_stats": {},
            "augmented": False,
        }

    base_counts = self._get_dataset_split_image_counts(source_dir)
    try:
        postprocess = self._apply_step4_dataset_postprocessing(
            dataset_path=augmented_dir,
            target=target,
            profile=profile,
            progress_var_name=progress_var_name,
            status_attr_name=status_attr_name,
            balance_plan=balance_plan,
        )
    except Exception as exc:
        logger.exception("Nie udało się zwiększyć syntetycznie datasetu")
        try:
            if augmented_dir.exists():
                shutil.rmtree(augmented_dir, ignore_errors=True)
        except Exception:
            pass
        return {
            "ok": False,
            "dataset_path": str(source_dir),
            "message": f"Nie udało się zwiększyć syntetycznie datasetu: {exc}",
            "counts": base_counts,
            "augmentation_stats": {},
            "augmented": False,
        }
    augmented_counts = self._get_dataset_split_image_counts(augmented_dir)
    stats = dict(postprocess.get("augmentation_stats") or {})
    generated = int(stats.get("generated", 0) or 0)
    base_total = int(base_counts.get("total", 0) or 0)
    augmented_total = int(augmented_counts.get("total", 0) or 0)
    generation_complete = bool(requested_extra <= 0 or generated >= requested_extra)
    mz_variant_manifest: dict = {}
    mz_completion_status = ""
    mz_remaining_deficits: dict[str, int] = {}
    mz_diversity_warnings: list[str] = []
    if normalized_target == "char" and before_distribution is not None:
        try:
            analysis_dir = augmented_dir / "analysis"
            before_refs = save_character_distribution_artifacts(
                before_distribution,
                analysis_dir,
                prefix="character_class_distribution_before",
            )
            after_distribution = analyze_character_class_distribution(
                augmented_dir,
                target_ratio=float(_step4_plan_attr(balance_plan, "target_ratio", 0.50) or 0.50),
            )
            after_refs = save_character_distribution_artifacts(
                after_distribution,
                analysis_dir,
                prefix="character_class_distribution_after",
            )
            val_test_after = build_character_dataset_file_fingerprint(augmented_dir, splits=("val", "test"))
            val_test_guard = compare_character_val_test_unchanged(val_test_before, val_test_after)
            representation_status, mz_remaining_deficits, mz_diversity_warnings = _step4_mz_completion_status_from_distribution(after_distribution)
            execution_status = _step4_mz_execution_status(
                generated,
                requested_extra,
                generation_complete=generation_complete,
            )
            freeze_status = _step4_mz_freeze_status(val_test_guard)
            if used_pending_balance_plan:
                mz_completion_status = _step4_mz_final_completion_status(
                    representation_status=representation_status,
                    freeze_status=freeze_status,
                    execution_status=execution_status,
                    plan_feasible=bool(_step4_plan_attr(balance_plan, "feasible", True)),
                    generated=generated,
                    deficits=mz_remaining_deficits,
                )
                ready_for_training = _step4_mz_ready_for_training(
                    execution_status=execution_status,
                    representation_status=representation_status,
                    freeze_status=freeze_status,
                )
            else:
                mz_completion_status = str(stats.get("completion_status") or execution_status or "").strip()
                if freeze_status != "UNCHANGED":
                    mz_completion_status = "VAL_TEST_CHANGED"
                ready_for_training = bool(postprocess.get("ok", True)) and bool(generation_complete) and freeze_status == "UNCHANGED"
            if used_pending_balance_plan and generated > 0:
                _clear_pending_character_balance_plan(self)
            mz_variant_manifest = build_character_training_variant_manifest(
                base_dataset=source_dir,
                before_distribution=before_refs.get("json", {}).get("path", ""),
                after_distribution=after_refs.get("json", {}).get("path", ""),
                plan=balance_plan,
                sources={
                    "real": int(base_counts.get("train", 0) or 0),
                    "added_real": 0,
                    "augmented_real": generated,
                    "synthetic": 0,
                    "augmented_synthetic": 0,
                },
                base_dataset_sha_or_fingerprint=base_dataset_fingerprint,
                val_test_unchanged=val_test_guard,
                augmentation_mode=("mz_auto_representation" if used_pending_balance_plan else "manual_train_augmentation"),
                requested_images=requested_extra,
                planned_images=max(0, int(_step4_plan_attr(balance_plan, "planned_images", requested_extra) or requested_extra)),
                generated_images=generated,
                completion_status=(mz_completion_status or str(stats.get("completion_status") or "")),
                stop_reason=str(stats.get("stop_reason") or ""),
            )
            mz_variant_manifest["augmentation"]["seed"] = int(getattr(profile, "seed", 42) or 42)
            mz_variant_manifest["augmentation"]["randomness_mode"] = str(getattr(profile, "randomness_mode", "") or "")
            mz_variant_manifest["deficit_after"] = dict(mz_remaining_deficits)
            mz_variant_manifest["diversity_warnings"] = list(mz_diversity_warnings)
            mz_variant_manifest["execution_status"] = execution_status
            mz_variant_manifest["representation_status"] = representation_status
            mz_variant_manifest["freeze_status"] = freeze_status
            mz_variant_manifest["ready_for_training"] = bool(ready_for_training)
            stats["completion_status"] = mz_completion_status
            stats["completion_ok"] = bool(ready_for_training)
            if used_pending_balance_plan:
                mz_variant_manifest["approved_balance_plan"] = True
            (augmented_dir / "mz_training_variant_manifest.json").write_text(
                json.dumps(mz_variant_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if not bool(val_test_guard.get("unchanged")):
                postprocess["ok"] = False
                postprocess["message"] = (
                    str(postprocess.get("message") or "").strip()
                    + "\nWalidacja freeze: split val/test zmienił się podczas augmentacji."
                ).strip()
            if used_pending_balance_plan and mz_remaining_deficits:
                missing_text = ", ".join(f"{symbol}: {count}" for symbol, count in sorted(mz_remaining_deficits.items()))
                postprocess["message"] = (
                    str(postprocess.get("message") or "").strip()
                    + f"\nReprezentacja MZ po augmentacji nadal ma braki: {missing_text}."
                ).strip()
        except Exception as exc:
            logger.exception("Nie udało się zapisać manifestu wariantu MZ")
            postprocess["ok"] = False
            postprocess["message"] = (
                str(postprocess.get("message") or "").strip()
                + f"\nNie udało się zapisać manifestu wariantu MZ: {exc}"
            ).strip()
    summary_prefix = "Wariant utworzono" if generation_complete else "Wariant nie został ukończony"
    summary = (
        f"{summary_prefix}: "
        f"oryginalne={base_total}, syntetyczne={generated}/{requested_extra}, razem={augmented_total}.\n"
        f"Katalog: {augmented_dir.name}"
    )
    process_message = str(postprocess.get("message") or "").strip()
    message = summary if not process_message else f"{summary}\n{process_message}"
    status_message = (
        f"{summary_prefix}: "
        f"oryginalne={base_total}, syntetyczne={generated}/{requested_extra}, razem={augmented_total}"
    )
    scope_meta = _write_step4_augmentation_scope_manifest(
        self,
        source_dataset_dir=source_dir,
        augmented_dataset_dir=augmented_dir,
        target=target,
        requested_extra=requested_extra,
        generated=generated,
        base_total=base_total,
        augmented_total=augmented_total,
    )
    return {
        "ok": bool(postprocess.get("ok", True)) and generated > 0 and generation_complete,
        "dataset_path": str(augmented_dir),
        "message": message,
        "counts": augmented_counts,
        "augmentation_stats": stats,
        "augmented": True,
        "base_total": base_total,
        "generated": generated,
        "requested_extra": requested_extra,
        "augmented_total": augmented_total,
        "generation_complete": generation_complete,
        "status_message": status_message,
        "synthetic_scope": "training_dataset_only",
        "used_pending_character_balance_plan": used_pending_balance_plan,
        "synthetic_images_project_base": False,
        "synthetic_images_next_iteration_base": False,
        "augmentation_scope": scope_meta,
        "mz_training_variant_manifest": mz_variant_manifest,
    }

def _apply_step4_dataset_postprocessing(
    self,
    *,
    dataset_path: str | Path,
    target: str,
    profile: AugmentationProfile,
    progress_var_name: str,
    status_attr_name: str,
    balance_plan=None,
) -> dict:
    dataset_dir = Path(dataset_path)
    normalized_target = CONFIG.normalize_task_target(target)
    profile = (profile or AugmentationProfile()).normalized()
    messages: list[str] = []
    result = {
        "ok": True,
        "message": "",
        "augmentation_stats": {},
    }

    if normalized_target == "plate" and profile.class_name:
        try:
            ok_names, names_msg = update_yolo_dataset_class_names(dataset_dir, {0: profile.class_name})
            if not ok_names:
                result["ok"] = False
                messages.append(f"Nazwy klas: {names_msg}")
        except Exception as exc:
            result["ok"] = False
            messages.append(f"Nazwy klas: nie udało się zaktualizować data.yaml ({exc}).")

    if not profile.enabled and profile.extra_count > 0:
        profile = replace(profile, enabled=True).normalized()

    if not profile.enabled or profile.extra_count <= 0:
        result["message"] = "\n".join(messages)
        return result

    self._ui(
        lambda requested=max(0, int(getattr(profile, "extra_count", 0) or 0)): self._set_training_widget_text(
            getattr(self, status_attr_name, None),
            f"Krok 2/2: zwiększanie syntetyczne train 0/{requested}...",
        )
    )
    def reset_progress():
        progress_var = getattr(self, progress_var_name, None)
        if progress_var is not None:
            try:
                progress_var.set(0)
            except Exception:
                pass

    self._ui(reset_progress)

    def progress(current: int, total: int, name: str):
        pct = (float(current) / max(1.0, float(total))) * 100.0

        def apply_progress():
            progress_var = getattr(self, progress_var_name, None)
            if progress_var is not None:
                try:
                    progress_var.set(pct)
                except Exception:
                    pass
            status = getattr(self, status_attr_name, None)
            if status is not None:
                try:
                    self._style_training_success_label(status)
                    self._set_training_widget_text(status, f"Krok 2/2: zwiększanie syntetyczne {int(current)}/{int(total)}...")
                except Exception:
                    pass

        self._ui(apply_progress)

    train_pool_count = 0
    for train_images_dir in (
        dataset_dir / "images" / "train",
        dataset_dir / "train" / "images",
        dataset_dir / "train",
    ):
        try:
            if train_images_dir.exists():
                train_pool_count = len(get_image_files(train_images_dir))
                if train_pool_count > 0:
                    break
        except Exception:
            continue
    if train_pool_count > 0:
        try:
            if int(getattr(profile, "sample_size", 0) or 0) != train_pool_count:
                profile = replace(profile, sample_size=train_pool_count).normalized()
        except Exception:
            pass

    ok_aug, aug_msg, aug_stats = augment_yolo_dataset_train_split(
        dataset_dir,
        profile,
        progress_callback=progress,
        balance_plan=balance_plan,
    )
    result["augmentation_stats"] = dict(aug_stats or {})
    if ok_aug:
        messages.append(str(aug_msg or "Zwiększanie syntetyczne train zakończone."))
    else:
        result["ok"] = False
        messages.append(str(aug_msg or "Zwiększanie syntetyczne train nie zostało ukończone."))
    result["message"] = "\n".join([msg for msg in messages if str(msg or "").strip()])
    return result

def _handle_step4_dataset_success_result(
    self,
    *,
    dataset_path: str | Path,
    message: str,
    target: str,
    counts: dict | None = None,
    result_meta: dict | None = None,
):
    counts = counts or {}
    result_meta = dict(result_meta or {})
    try:
        dataset_root = Path(dataset_path)
        yaml_path = dataset_root / "data.yaml" if dataset_root.is_dir() else dataset_root
        source_stats = TrainingSourceStats.from_mapping(counts or self._get_dataset_split_image_counts(dataset_root))
        self._last_training_source = TrainingSource(
            target=CONFIG.normalize_task_target(target),
            kind="yolo_dataset",
            dataset_dir=str(yaml_path.parent if yaml_path.name.lower() == "data.yaml" else dataset_root),
            yaml_path=str(yaml_path),
            validated=bool(yaml_path.exists()),
            stats=source_stats,
            provenance="Utworzony w PZ1",
            source_stage="Z4/PZ1",
            message=str(message or "").strip(),
        )
    except Exception:
        pass
    target_label = self._format_training_target_label(target)
    path_text = self._format_step4_dataset_result_path(dataset_path)
    if len(path_text) > 96:
        path_text = f"{path_text[:44]}...{path_text[-44:]}"
    dataset_name = ""
    try:
        dataset_name = Path(dataset_path).name
    except Exception:
        dataset_name = str(dataset_path or "").strip()

    train_count = int(counts.get("train", 0) or 0)
    val_count = int(counts.get("val", 0) or 0)
    test_count = int(counts.get("test", 0) or 0)
    total_count = int(counts.get("total", 0) or 0)
    if total_count <= 0:
        total_count = train_count + val_count + test_count

    augmented = bool(result_meta.get("augmented"))
    generated = int(result_meta.get("generated", 0) or 0)
    requested_extra = int(result_meta.get("requested_extra", 0) or 0)
    base_total = int(result_meta.get("base_total", 0) or 0)
    augmented_total = int(result_meta.get("augmented_total", 0) or 0)
    if augmented_total <= 0:
        augmented_total = total_count

    split_text = ""
    if counts:
        split_text = (
            f"train {train_count}  |  "
            f"val {val_count}  |  "
            f"test {test_count}"
        )

    summary_rows: list[tuple[str, str]] = []
    if target_label:
        summary_rows.append(("Tor", target_label))
    if dataset_name:
        summary_rows.append(("Wariant", dataset_name))
    if split_text:
        summary_rows.append(("Split", split_text))
    if augmented:
        augmentation_text = f"+{generated} syntetycznych obrazów train"
        if requested_extra > 0:
            augmentation_text = f"+{generated}/{requested_extra} syntetycznych obrazów train"
        if base_total > 0:
            augmentation_text += f"  |  oryginalnie {base_total}, razem {augmented_total}"
        summary_rows.append(("Powiększenie", augmentation_text))
        summary_rows.append(("Zakres syntetyków", "tylko train tego wariantu; nie baza kolejnej iteracji"))
    elif total_count > 0:
        summary_rows.append(("Razem", f"{total_count} obrazów"))
    mz_manifest = result_meta.get("mz_training_variant_manifest") if isinstance(result_meta, dict) else None
    if CONFIG.normalize_task_target(target) == "char" and isinstance(mz_manifest, dict) and mz_manifest:
        status = str(mz_manifest.get("completion_status") or "").strip()
        target_count = int(mz_manifest.get("target_count", 0) or 0)
        planned_images = int(mz_manifest.get("planned_images", 0) or 0)
        generated_images = int(mz_manifest.get("generated_images", 0) or 0)
        deficits_after = dict(mz_manifest.get("deficit_after") or {})
        diversity_warnings = list(mz_manifest.get("diversity_warnings") or [])
        summary_rows.append((
            "Reprezentacja MZ",
            _format_step4_mz_completion_label(status, deficits_after, diversity_warnings),
        ))
        if target_count > 0:
            summary_rows.append(("Miarka reprezentacji", f"{target_count} przykładów znaku w train"))
        summary_rows.append(("Uzupełnienie train", f"ustawiono +{planned_images} | wygenerowano +{generated_images}"))
    if path_text:
        summary_rows.append(("Lokalizacja", path_text))

    if augmented and generated > 0:
        hero_text = "WARIANT UTWORZONO"
        hero_subtitle = (
            f"Dodano +{generated} syntetycznych obrazów train. "
            f"Wariant ma teraz {augmented_total or total_count} obrazów i jest wejściem treningu. "
            "Syntetyki nie zasilają puli projektu ani kolejnej iteracji."
        )
    else:
        hero_text = "WARIANT UTWORZONO"
        hero_subtitle = f"Wariant ma {total_count} obrazów i jest ustawiony jako bieżące wejście treningu." if total_count else "Wariant został zapisany i ustawiony jako bieżące wejście treningu."

    body = "Sprawdź krótkie podsumowanie i wybierz, co zrobić z utworzonym wariantem."
    guidance = "Jeśli wariant wygląda dobrze, możesz od razu trenować model. Jeśli nie, usuń wariant i zbuduj go ponownie."

    result_action = self._show_step4_dataset_result_modal(
        title="Wariant datasetu gotowy",
        body=body,
        allow_pz2=True,
        primary_text="Trenuj na tym wariancie",
        secondary_text="Zamknij podsumowanie",
        preview_path=dataset_path,
        summary_rows=summary_rows,
        hero_text=hero_text,
        hero_subtitle=hero_subtitle,
        guidance_text=guidance,
        allow_delete=True,
        delete_text="Usuń wariant",
        return_action=True,
    )

    if result_action == "deleted":
        return

    if CONFIG.normalize_task_target(target) == "plate" and CAMPAIGN.get_active_project_name():
        try:
            self._remember_campaign_plate_training_source(dataset_path)
        except Exception:
            pass

    self._mark_step4_dataset_ready(dataset_path, target=target)
    try:
        self._pending_step4_input_training_source = None
    except Exception:
        pass
    self._schedule_step4_dataset_summary_refresh()

    if result_action == "train":
        self._open_pz2_from_dataset_result(dataset_path=dataset_path, target=target)

def _handle_step4_dataset_failure_result(
    self,
    *,
    message: str,
    target: str,
    critical: bool = False,
    allow_pz2_override: bool | None = None,
):
    target_label = self._format_training_target_label(target)
    allow_pz2 = (
        self._can_open_pz2_after_dataset_result()
        if allow_pz2_override is None
        else bool(allow_pz2_override)
    )
    reason = str(message or "Nieznany błąd").strip()
    body = (
        "Dataset nie został utworzony.\n\n"
        f"Przyczyna niepowodzenia: {reason}"
    )

    if self._show_step4_dataset_result_modal(
        title="Dataset nie został utworzony",
        body=body,
        allow_pz2=allow_pz2,
    ):
        self._open_pz2_from_dataset_result()

def _create_dataset_thread(self):
    source_info = self._resolve_dataset_creator_inputs()
    if not bool(source_info.get("ok")):
        self._refresh_dataset_creator_cta_state()
        self._handle_step4_dataset_failure_result(
            message=str(source_info.get("message") or "Najpierw wskaż plik anotacji XML i folder obrazów."),
            target="plate",
            critical=False,
        )
        return
    try:
        self._pending_step4_input_training_source = source_info.get("training_source")
    except Exception:
        pass

    xml = Path(source_info["xml"])
    images_dir = Path(source_info["images_dir"])
    stage_source_images_dir = images_dir
    stage_source_image_paths = None

    try:
        if bool(CAMPAIGN.get_active_project_name()) and self._get_selected_training_target() == "plate":
            campaign_iter_raw = CAMPAIGN.get_iteration_image_source_dir() or CAMPAIGN.get_iteration_raw_dir()
            if campaign_iter_raw is not None and Path(campaign_iter_raw).exists():
                stage_source_images_dir = Path(campaign_iter_raw)
            try:
                manifest_count = int(CAMPAIGN.get_iteration_manifest_image_count() or 0)
            except Exception:
                manifest_count = 0
            if manifest_count > 0:
                stage_source_image_paths = list(
                    CAMPAIGN.get_iteration_manifest_image_paths(base_dir=stage_source_images_dir) or []
                )
    except Exception:
        stage_source_images_dir = images_dir
        stage_source_image_paths = None

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset zapisuj w katalogu projektu albo w przestrzeni globalnej.
    base_datasets_dir = self._get_datasets_base_dir()
    out_dir = base_datasets_dir / f"Plates_CVAT_{timestamp}"

    # Pokaż użytkownikowi docelową ścieżkę zapisu.
    self.ds_out_var.set(str(out_dir))

    # Wyczyść poprzedni stan parsera przed nowym odczytem XML.
    try:
        if hasattr(self.creator, "annotations"):
            self.creator.annotations = []
    except Exception:
        pass

    ok, msg, _ = self.creator.parse_cvat_xml(xml)
    if not ok:
        self._handle_step4_dataset_failure_result(
            message=str(msg or "Nie udało się odczytać pliku XML."),
            target="plate",
            critical=False,
        )
        return

    alignment = self._validate_dataset_creator_xml_images_alignment(images_dir)
    if not bool(alignment.get("ok")):
        self._style_training_error_label(self.ds_status)
        self._set_training_widget_text(self.ds_status, "XML i folder obrazów nie są zgodne.")
        self._handle_step4_dataset_failure_result(
            message=str(alignment.get("message") or "Wskaż zgodny plik XML i folder obrazów."),
            target="plate",
            critical=False,
        )
        return
    try:
        self._append_step4_builder_log(f"[WALIDACJA] {alignment.get('message')}")
    except Exception:
        pass

    train = float(self.train_pct.get()) / 100.0
    val = float(self.val_pct.get()) / 100.0
    ratios = {
        "train": train,
        "val": val,
        "test": max(0.0, 1.0 - train - val)
    }

    augmentation_profile = self._get_step4_augmentation_profile("plate")

    if not self._begin_step4_operation("z4.dataset.build", "Z4: przygotowanie wariantu treningowego tablic"):
        return
    self.dataset_build_is_running = True

    self._configure_train_progress_styles()
    self.ds_progress_var.set(0)
    self._style_training_success_label(self.ds_status)
    self._set_training_widget_text(self.ds_status, "Rozpoczynam przygotowanie datasetu...")

    def worker():
        try:
            stage_result = {
                "enabled": bool(CAMPAIGN.get_active_project_name()) and self._get_selected_training_target() == "plate",
                "ok": False,
                "message": "",
                "pending_count": 0,
                "stage_images_total": 0,
                "stage_images_dir": "",
            }
            progress_state = {"last": 0}

            def prog(c, t, n):
                c_int = int(c)
                t_int = int(t)
                step = 1 if t_int <= 100 else max(10, t_int // 100)
                is_final = bool(t_int > 0 and c_int >= t_int)
                if not is_final and c_int > 1 and (c_int - int(progress_state.get("last", 0))) < step:
                    return
                progress_state["last"] = c_int
                pct = (c_int / t_int) * 100 if t_int > 0 else 0

                def apply_progress():
                    self.ds_progress_var.set(pct)
                    self._style_training_success_label(self.ds_status)
                    if _step4_profile_requests_augmentation(augmentation_profile):
                        requested = max(0, int(getattr(augmentation_profile, "extra_count", 0) or 0))
                        self._set_training_widget_text(
                            self.ds_status,
                            f"Krok 1/2: buduję bazę datasetu {c_int}/{t_int}. Następnie dodam +{requested} syntetycznych obrazów train.",
                        )
                    else:
                        self._set_training_widget_text(self.ds_status, f"{c_int}/{t_int} obrazów...")

                self._ui(apply_progress)

            ok2, msg2, create_stats = self.creator.create_dataset(images_dir, out_dir, ratios, prog)

            if ok2:
                dataset_counts = dict(create_stats or {})
                if int(dataset_counts.get("total", 0) or 0) <= 0:
                    dataset_counts = self._get_dataset_split_image_counts(out_dir)
                else:
                    dataset_counts = self._get_dataset_split_image_counts(out_dir) or dataset_counts
                result_dataset_path = Path(out_dir)
                final_status_message = None
                result_meta: dict = {}
                try:
                    source_run_dir = xml.parent if xml.name.lower() == "annotations.xml" else None
                    self._write_plate_dataset_source_manifest(
                        out_dir,
                        source_kind="z4_cvat_builder",
                        source_run_dir=source_run_dir,
                        source_xml_path=xml,
                        source_images_dir=images_dir,
                    )
                except Exception as e:
                    logger.debug(f"Nie udało się zapisac manifestu źródła datasetu Z4: {e}")

                if _step4_profile_requests_augmentation(augmentation_profile):
                    variant = self._create_step4_augmented_dataset_variant(
                        source_dataset_path=out_dir,
                        target="plate",
                        profile=augmentation_profile,
                        progress_var_name="ds_progress_var",
                        status_attr_name="ds_status",
                    )
                    if str(variant.get("message") or "").strip():
                        msg2 = f"{msg2}\n{variant.get('message')}"
                    if bool(variant.get("ok", True)):
                        result_dataset_path = Path(str(variant.get("dataset_path") or out_dir))
                        dataset_counts = dict(variant.get("counts") or self._get_dataset_split_image_counts(result_dataset_path))
                        final_status_message = str(variant.get("status_message") or "").strip() or None
                        result_meta = dict(variant)
                        self._ui(lambda p=str(result_dataset_path): self.ds_out_var.set(p))
                    else:
                        failure_msg = str(variant.get("message") or "Syntetyczne zwiększanie datasetu tablic nie zostało wykonane.")
                        logger.warning(failure_msg)
                        self._ui(lambda: self._style_training_error_label(self.ds_status))
                        self._ui(lambda msg=failure_msg: self._set_training_widget_text(self.ds_status, msg))
                        self._ui(
                            lambda msg=failure_msg: self._handle_step4_dataset_failure_result(
                                message=msg,
                                target="plate",
                                critical=False,
                            )
                        )
                        return
                else:
                    postprocess = self._apply_step4_dataset_postprocessing(
                        dataset_path=out_dir,
                        target="plate",
                        profile=augmentation_profile,
                        progress_var_name="ds_progress_var",
                        status_attr_name="ds_status",
                    )
                    if str(postprocess.get("message") or "").strip():
                        msg2 = f"{msg2}\n{postprocess.get('message')}"

                if stage_result["enabled"]:
                    stage_ok, stage_msg, stage_stats = self.creator.sync_pending_stage(
                        stage_source_images_dir,
                        self._get_manual_plate_stage_dir(),
                        source_image_paths=stage_source_image_paths,
                    )
                    stage_result["ok"] = bool(stage_ok)
                    stage_result["message"] = str(stage_msg or "").strip()
                    if isinstance(stage_stats, dict):
                        stage_result["pending_count"] = int(stage_stats.get("pending_count", 0) or 0)
                        stage_result["stage_images_total"] = int(stage_stats.get("stage_images_total", 0) or 0)
                        stage_result["stage_images_dir"] = str(stage_stats.get("stage_images_dir") or "").strip()
                self._ui(lambda: self._style_training_success_label(self.ds_status))
                self._ui(lambda: self._set_training_widget_text(self.ds_status, "Dataset treningowy został przygotowany!"))
                if stage_result["enabled"] and (not stage_result["ok"]) and stage_result["message"]:
                    logger.debug(f"Synchronizacja zdjęć oczekujących po utworzeniu datasetu: {stage_result['message']}")

                self._ui(lambda: self._style_training_success_label(self.ds_status))
                self._ui(
                    lambda counts=dict(dataset_counts), final_msg=final_status_message: self._set_training_widget_text(
                        self.ds_status,
                        final_msg or (
                            "Dataset został utworzony: "
                            f"train={int(counts.get('train', 0) or 0)}, "
                            f"val={int(counts.get('val', 0) or 0)}, "
                            f"test={int(counts.get('test', 0) or 0)}"
                        ),
                    )
                )

                self._ui(
                    lambda p=str(result_dataset_path), msg=str(msg2), counts=dict(dataset_counts), meta=dict(result_meta): self._handle_step4_dataset_success_result(
                        dataset_path=p,
                        message=msg,
                        target="plate",
                        counts=counts,
                        result_meta=meta,
                    )
                )
            else:
                self._ui(lambda: self._style_training_error_label(self.ds_status))
                self._ui(lambda: self._set_training_widget_text(self.ds_status, "Błąd przygotowania datasetu"))
                self._ui(
                    lambda msg=str(msg2): self._handle_step4_dataset_failure_result(
                        message=msg,
                        target="plate",
                        critical=False,
                    )
                )

        except Exception as e:
            logger.exception("Krytyczny błąd przygotowania datasetu tablic w Z4/PZ1")
            self._ui(lambda: self._style_training_error_label(self.ds_status))
            self._ui(lambda err=str(e): self._set_training_widget_text(self.ds_status, f"Krytyczny błąd przygotowania datasetu: {err}"))
            self._ui(
                lambda err=str(e): self._handle_step4_dataset_failure_result(
                    message=err,
                    target="plate",
                    critical=True,
                )
            )

        finally:
            self.dataset_build_is_running = False
            self._end_step4_operation("z4.dataset.build")
            self._ui(self._refresh_dataset_creator_cta_state)
            self._ui(self._refresh_training_start_state)

    threading.Thread(target=worker, daemon=True).start()

def _split_dataset_thread(self):
    source_info = self._resolve_dataset_split_inputs()
    if not bool(source_info.get("ok")):
        self._refresh_dataset_split_cta_state()
        self._handle_step4_dataset_failure_result(
            message=str(source_info.get("message") or "Najpierw wskaż źródłowy dataset znaków."),
            target="char",
            critical=False,
            allow_pz2_override=False,
        )
        return
    try:
        self._pending_step4_input_training_source = source_info.get("training_source")
    except Exception:
        pass

    src = Path(source_info["src"])

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Wynik splitu zapisuj obok innych datasetów projektu.
    base_datasets_dir = self._get_datasets_base_dir()
    out = base_datasets_dir / f"{src.name}_Split_{timestamp}"

    # Pokaż użytkownikowi docelową ścieżkę splitu.
    self.split_out_var.set(str(out))
    self._set_split_feedback_visibility(True)

    train = float(self.train_pct.get()) / 100.0
    val = float(self.val_pct.get()) / 100.0
    ratios = {
        "train": train,
        "val": val,
        "test": max(0.0, 1.0 - train - val)
    }

    augmentation_profile = self._get_step4_augmentation_profile("char")
    try:
        profile_requested_extra = max(0, int(getattr(augmentation_profile, "extra_count", 0) or 0))
    except Exception:
        profile_requested_extra = 0
    pending_balance_plan = getattr(self, "_pending_character_balance_plan", None)
    try:
        pending_planned_extra = max(0, int(getattr(pending_balance_plan, "planned_images", 0) or 0))
    except Exception:
        pending_planned_extra = 0
    try:
        pending_target_count_by_symbol = dict(getattr(pending_balance_plan, "target_count_by_symbol", {}) or {})
    except Exception:
        pending_target_count_by_symbol = {}
    try:
        pending_extra_count_by_symbol = dict(getattr(pending_balance_plan, "requested_extra_by_symbol", {}) or {})
    except Exception:
        pending_extra_count_by_symbol = {}
    try:
        pending_target_count = int(getattr(pending_balance_plan, "target_count", 0) or 0)
    except Exception:
        pending_target_count = 0
    manual_augmentation_requested = bool(
        _step4_profile_requests_augmentation(augmentation_profile)
        and (pending_balance_plan is None or profile_requested_extra != pending_planned_extra)
    )

    if not self._begin_step4_operation("z4.dataset.split", "Z4: przygotowanie wariantu treningowego znaków"):
        return
    self.dataset_split_is_running = True

    self._set_split_feedback_visibility(True)
    self.split_progress_var.set(0)
    self._style_training_success_label(self.split_status)
    self._set_training_widget_text(self.split_status, "Rozpoczynam przygotowanie wariantu treningowego...")
    try:
        self.frame.update_idletasks()
    except Exception:
        pass

    def worker():
        nonlocal augmentation_profile
        try:
            progress_state = {"last": 0}

            def prog(c, t, n):
                c_int = int(c)
                t_int = int(t)
                step = 1 if t_int <= 100 else max(10, t_int // 100)
                is_final = bool(t_int > 0 and c_int >= t_int)
                if not is_final and c_int > 1 and (c_int - int(progress_state.get("last", 0))) < step:
                    return
                progress_state["last"] = c_int
                pct = (c_int / t_int) * 100 if t_int > 0 else 0

                def apply_progress():
                    self.split_progress_var.set(pct)
                    self._style_training_success_label(self.split_status)
                    if _step4_profile_requests_augmentation(augmentation_profile):
                        requested = max(0, int(getattr(augmentation_profile, "extra_count", 0) or 0))
                        self._set_training_widget_text(
                            self.split_status,
                            f"Krok 1/2: buduję bazę datasetu {c_int}/{t_int}. Następnie dodam +{requested} syntetycznych obrazów train.",
                        )
                    else:
                        self._set_training_widget_text(self.split_status, f"Kopiowanie {c_int}/{t_int}...")

                self._ui(apply_progress)

            ok, msg, split_stats = self.splitter.split_dataset(src, out, ratios, prog)

            if ok:
                dataset_counts = dict(split_stats or {})
                if int(dataset_counts.get("total", 0) or 0) <= 0:
                    dataset_counts = self._get_dataset_split_image_counts(out)
                else:
                    dataset_counts = self._get_dataset_split_image_counts(out) or dataset_counts
                result_dataset_path = Path(out)
                final_status_message = None
                result_meta: dict = {}
                self._ui(lambda: self._set_training_widget_text(self.split_status, "Krok 2/3: liczę reprezentację znaków MZ dla finalnego splitu..."))
                try:
                    exact_balance_plan = plan_character_train_augmentation(
                        out,
                        target_count=(
                            0
                            if pending_extra_count_by_symbol
                            else (pending_target_count if pending_target_count > 0 else None)
                        ),
                        target_count_by_symbol=(None if pending_extra_count_by_symbol else (pending_target_count_by_symbol or None)),
                        extra_count_by_symbol=pending_extra_count_by_symbol or None,
                        respect_existing_augmented_variants=False,
                    )
                except Exception as exc:
                    failure_msg = f"Nie udało się policzyć planu reprezentacji MZ dla finalnego splitu: {exc}"
                    logger.exception(failure_msg)
                    self._ui(lambda: self._style_training_error_label(self.split_status))
                    self._ui(lambda msg=failure_msg: self._set_training_widget_text(self.split_status, msg))
                    self._ui(
                        lambda msg=failure_msg: self._handle_step4_dataset_failure_result(
                            message=msg,
                            target="char",
                            critical=False,
                            allow_pz2_override=False,
                        )
                    )
                    return
                exact_planned_images = max(0, int(getattr(exact_balance_plan, "planned_images", 0) or 0))
                exact_feasible = bool(getattr(exact_balance_plan, "feasible", True))
                self._ui(
                    lambda plan=exact_balance_plan: self._set_training_widget_text(
                        self.split_status,
                        _format_step4_mz_plan_status(plan),
                    )
                )
                try:
                    _clear_pending_character_balance_plan(self)
                except Exception:
                    pass
                if not exact_feasible and exact_planned_images <= 0:
                    try:
                        result_meta = _finalize_step4_mz_representation_variant(
                            self,
                            dataset_path=out,
                            plan=exact_balance_plan,
                            profile=augmentation_profile,
                            counts=dataset_counts,
                            generated=0,
                            requested_extra=exact_planned_images,
                            augmented=False,
                        )
                    except Exception as exc:
                        logger.exception("Nie udało się zapisać manifestu niewykonalnego planu MZ")
                        result_meta = {"message": f"Nie można osiągnąć reprezentacji MZ i nie udało się zapisać manifestu: {exc}"}
                    failure_msg = str(result_meta.get("message") or "Nie można osiągnąć progu reprezentacji MZ przy aktualnym materiale.")
                    self._ui(lambda: self._style_training_error_label(self.split_status))
                    self._ui(lambda msg=failure_msg: self._set_training_widget_text(self.split_status, msg))
                    self._ui(
                        lambda msg=failure_msg: self._handle_step4_dataset_failure_result(
                            message=msg,
                            target="char",
                            critical=False,
                            allow_pz2_override=False,
                        )
                    )
                    return

                if manual_augmentation_requested or exact_planned_images > 0:
                    augmentation_mode = "manual_train_augmentation"
                    balance_plan_for_variant = None
                    if not manual_augmentation_requested:
                        augmentation_profile = replace(
                            augmentation_profile,
                            enabled=True,
                            extra_count=exact_planned_images,
                        ).normalized()
                        augmentation_mode = "mz_auto_representation"
                        balance_plan_for_variant = exact_balance_plan
                    variant = self._create_step4_augmented_dataset_variant(
                        source_dataset_path=out,
                        target="char",
                        profile=augmentation_profile,
                        progress_var_name="split_progress_var",
                        status_attr_name="split_status",
                        balance_plan_override=balance_plan_for_variant,
                        augmentation_mode=augmentation_mode,
                    )
                    if str(variant.get("message") or "").strip():
                        msg = f"{msg}\n{variant.get('message')}"
                    if bool(variant.get("ok", True)):
                        result_dataset_path = Path(str(variant.get("dataset_path") or out))
                        dataset_counts = dict(variant.get("counts") or self._get_dataset_split_image_counts(result_dataset_path))
                        final_status_message = str(variant.get("status_message") or "").strip() or None
                        result_meta = dict(variant)
                        self._ui(lambda p=str(result_dataset_path): self.split_out_var.set(p))
                    else:
                        failure_msg = str(variant.get("message") or "Syntetyczne zwiększanie datasetu znaków nie zostało wykonane.")
                        logger.warning(failure_msg)
                        self._ui(lambda: self._style_training_error_label(self.split_status))
                        self._ui(lambda msg=failure_msg: self._set_training_widget_text(self.split_status, msg))
                        self._ui(
                            lambda msg=failure_msg: self._handle_step4_dataset_failure_result(
                                message=msg,
                                target="char",
                                critical=False,
                                allow_pz2_override=False,
                            )
                        )
                        return
                else:
                    variant = _finalize_step4_mz_representation_variant(
                        self,
                        dataset_path=out,
                        plan=exact_balance_plan,
                        profile=augmentation_profile,
                        counts=dataset_counts,
                        generated=0,
                        requested_extra=0,
                        augmented=False,
                    )
                    if str(variant.get("message") or "").strip():
                        msg = f"{msg}\n{variant.get('message')}"
                    if bool(variant.get("ok", True)):
                        result_meta = dict(variant)
                        final_status_message = str(variant.get("status_message") or "").strip() or None
                    else:
                        failure_msg = str(variant.get("message") or "Reprezentacja MZ nie została potwierdzona.")
                        self._ui(lambda: self._style_training_error_label(self.split_status))
                        self._ui(lambda msg=failure_msg: self._set_training_widget_text(self.split_status, msg))
                        self._ui(
                            lambda msg=failure_msg: self._handle_step4_dataset_failure_result(
                                message=msg,
                                target="char",
                                critical=False,
                                allow_pz2_override=False,
                            )
                        )
                        return
                self._ui(lambda: self._style_training_success_label(self.split_status))
                self._ui(
                    lambda counts=dict(dataset_counts), final_msg=final_status_message: self._set_training_widget_text(
                        self.split_status,
                        final_msg or (
                            "Dataset treningowy został przygotowany: "
                            f"train={int(counts.get('train', 0) or 0)}, "
                            f"val={int(counts.get('val', 0) or 0)}, "
                            f"test={int(counts.get('test', 0) or 0)}"
                        ),
                    )
                )
                self._ui(
                    lambda p=str(result_dataset_path), msg=str(msg), counts=dict(dataset_counts), meta=dict(result_meta): self._handle_step4_dataset_success_result(
                        dataset_path=p,
                        message=msg,
                        target="char",
                        counts=counts,
                        result_meta=meta,
                    )
                )
            else:
                self._ui(lambda: self._style_training_error_label(self.split_status))
                self._ui(lambda: self._set_training_widget_text(self.split_status, "Błąd przygotowania wariantu"))
                self._ui(
                    lambda msg=str(msg): self._handle_step4_dataset_failure_result(
                        message=msg,
                        target="char",
                        critical=False,
                        allow_pz2_override=False,
                    )
                )

        except Exception as e:
            logger.exception("Krytyczny błąd przygotowania wariantu treningowego znaków w Z4/PZ1")
            self._ui(lambda: self._style_training_error_label(self.split_status))
            self._ui(lambda err=str(e): self._set_training_widget_text(self.split_status, f"Krytyczny błąd przygotowania wariantu: {err}"))
            self._ui(
                lambda err=str(e): self._handle_step4_dataset_failure_result(
                    message=err,
                    target="char",
                    critical=True,
                    allow_pz2_override=False,
                )
            )

        finally:
            self.dataset_split_is_running = False
            self._end_step4_operation("z4.dataset.split")
            self._ui(self._refresh_dataset_split_cta_state)
            self._ui(self._refresh_training_start_state)

    threading.Thread(target=worker, daemon=True).start()
