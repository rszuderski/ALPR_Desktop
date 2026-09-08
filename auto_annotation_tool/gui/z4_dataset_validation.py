#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 dataset picker and character YOLO source validation helpers."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None



def _get_current_picker_parent(self, var_name: str) -> str:
    var = getattr(self, var_name, None)
    if var is None:
        return ""
    try:
        raw = str(var.get() or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return ""
    try:
        path = Path(raw)
        if path.is_file():
            path = path.parent
        elif not path.exists() and path.suffix:
            path = path.parent
        if path.exists() and path.is_dir():
            return str(path.resolve())
    except Exception:
        pass
    return ""


def _get_first_picker_dir(self, *candidates: Path | str | None) -> str:
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            path = Path(candidate)
            if path.exists() and path.is_dir():
                return str(path.resolve())
        except Exception:
            continue
    for candidate in candidates:
        if candidate is not None:
            return self._prepare_picker_initial_dir(candidate)
    return ""


def _get_plate_xml_picker_dir(self) -> str:
    current = self._get_current_picker_parent("cvat_xml_var")
    if current:
        return current

    campaign_auto_dir = None
    campaign_staging_auto_dir = None
    try:
        campaign_auto_dir = CAMPAIGN.get_dir("auto_ann")
    except Exception:
        campaign_auto_dir = None
    try:
        campaign_staging_auto_dir = CAMPAIGN.get_staging_dir("auto_ann")
    except Exception:
        campaign_staging_auto_dir = None

    return self._get_first_picker_dir(
        campaign_auto_dir,
        campaign_staging_auto_dir,
        CONFIG.get_auto_annotations_dir("plate"),
    )


def _get_plate_images_picker_dir(self) -> str:
    current = self._get_current_picker_parent("cvat_images_var")
    if current:
        return current

    iteration_images_dir = None
    iteration_raw_dir = None
    master_pool_dir = None
    campaign_raw_dir = None
    try:
        iteration_images_dir = CAMPAIGN.get_iteration_image_source_dir()
    except Exception:
        iteration_images_dir = None
    try:
        iteration_raw_dir = CAMPAIGN.get_iteration_raw_dir()
    except Exception:
        iteration_raw_dir = None
    try:
        master_pool_dir = CAMPAIGN.get_master_pool_dir()
    except Exception:
        master_pool_dir = None
    try:
        campaign_raw_dir = CAMPAIGN.get_dir("raw")
    except Exception:
        campaign_raw_dir = None

    return self._get_first_picker_dir(
        iteration_images_dir,
        iteration_raw_dir,
        master_pool_dir,
        campaign_raw_dir,
        CONFIG.DIR_1_RAW,
    )


def _get_char_dataset_source_picker_dir(self) -> str:
    current = self._get_current_picker_parent("split_src_var")
    if current:
        return current

    campaign_datasets_dir = None
    try:
        campaign_datasets_dir = CAMPAIGN.get_dir("datasets")
    except Exception:
        campaign_datasets_dir = None

    return self._get_first_picker_dir(
        campaign_datasets_dir,
        CONFIG.get_datasets_dir("char"),
    )


def _count_yolo_image_label_pairs(self, source_dir: Path) -> dict:
    source_dir = Path(source_dir)
    stats = {
        "total": 0,
        "flat": 0,
        "train": 0,
        "val": 0,
        "test": 0,
        "images_without_labels": 0,
    }

    started_at = time.perf_counter()
    signature_parts: list[str] = []
    for rel_image_dir, rel_label_dir in (
        (Path("images") / "train", Path("labels") / "train"),
        (Path("images") / "val", Path("labels") / "val"),
        (Path("images") / "test", Path("labels") / "test"),
        (Path("train") / "images", Path("train") / "labels"),
        (Path("val") / "images", Path("val") / "labels"),
        (Path("test") / "images", Path("test") / "labels"),
        (Path("images"), Path("labels")),
    ):
        for rel_dir in (rel_image_dir, rel_label_dir):
            path = source_dir / rel_dir
            try:
                mtime_ns = int(path.stat().st_mtime_ns) if path.exists() else 0
            except Exception:
                mtime_ns = 0
            signature_parts.append(f"{rel_dir.as_posix()}:{mtime_ns}")

    try:
        cache_root = str(source_dir.resolve())
    except Exception:
        cache_root = str(source_dir)
    cache_key = f"{cache_root}|{'|'.join(signature_parts)}"
    cache = getattr(self, "_char_yolo_pair_counts_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            self._char_yolo_pair_counts_cache = cache
        except Exception:
            pass
    cached_stats = cache.get(cache_key) if isinstance(cache, dict) else None
    if isinstance(cached_stats, dict):
        return dict(cached_stats)

    seen_images: set[str] = set()

    for split in ("train", "val", "test", ""):
        candidates = []
        if split:
            candidates.extend(
                [
                    (source_dir / "images" / split, source_dir / "labels" / split),
                    (source_dir / split / "images", source_dir / split / "labels"),
                ]
            )
        else:
            candidates.append((source_dir / "images", source_dir / "labels"))

        for image_dir, label_dir in candidates:
            if not image_dir.exists() or not image_dir.is_dir():
                continue

            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
                try:
                    image_key = os.path.normcase(os.path.abspath(os.fspath(image_path)))
                except Exception:
                    image_key = str(image_path)
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)
                label_path = label_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    key = split or "flat"
                    stats[key] = int(stats.get(key, 0) or 0) + 1
                    stats["total"] += 1
                else:
                    stats["images_without_labels"] += 1

    try:
        if len(cache) > 256:
            cache.clear()
        cache[cache_key] = dict(stats)
    except Exception:
        pass
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    if elapsed_ms >= 250:
        try:
            logger.info(
                f"[Z4/PZ1 PERF] char_pair_counts total={elapsed_ms}ms "
                f"pairs={int(stats.get('total', 0) or 0)} path={source_dir}"
            )
        except Exception:
            pass
    return stats


def _format_char_yolo_source_layout_message(self, stats: dict) -> str:
    total = int(stats.get("total", 0) or 0)
    flat = int(stats.get("flat", 0) or 0)
    train = int(stats.get("train", 0) or 0)
    val = int(stats.get("val", 0) or 0)
    test = int(stats.get("test", 0) or 0)
    missing = int(stats.get("images_without_labels", 0) or 0)
    split_total = train + val + test

    lines = [
        f"Zgodne pary obraz + etykieta: {total}",
        "Układ źródła:",
    ]
    if flat:
        lines.append(f"- źródło bez splitu (images/labels): {flat} par")
    if split_total:
        lines.append(f"- gotowy split train/val/test: train {train}, val {val}, test {test}")
    if not flat and not split_total:
        lines.append("- nie rozpoznano układu images/labels")
    if missing:
        lines.append(f"- obrazy bez etykiet: {missing} (nie wejdą do wariantu treningowego)")

    lines.extend(
        [
            "",
            "Źródło bez splitu to materiał wejściowy YOLO zapisany jako katalog images/labels, "
            "jeszcze bez podziału na train/val/test. PZ1 tworzy z niego wariant splitu treningowego.",
        ]
    )
    return "\n".join(lines)


def _has_supported_yolo_label_layout(self, source_dir: Path) -> bool:
    source_dir = Path(source_dir)
    if (source_dir / "images").exists() and (source_dir / "labels").exists():
        return True
    for split in ("train", "val", "test"):
        if (source_dir / split / "images").exists() and (source_dir / split / "labels").exists():
            return True
    return False


def _find_char_yolo_dataset_root_candidates(self, source_dir: Path, *, max_depth: int = 2) -> list[dict]:
    source_dir = Path(source_dir)
    candidates: list[dict] = []
    seen: set[str] = set()

    def visit(path: Path, depth: int):
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            return
        seen.add(resolved)

        if self._has_supported_yolo_label_layout(path) and not self._looks_like_char_classification_dataset(path):
            stats = self._count_yolo_image_label_pairs(path)
            if int(stats.get("total", 0) or 0) > 0:
                try:
                    stamp = float(path.stat().st_mtime)
                except Exception:
                    stamp = 0.0
                candidates.append({"path": path, "stats": stats, "stamp": stamp})
                return

        if depth >= max_depth:
            return
        try:
            children = [child for child in path.iterdir() if child.is_dir()]
        except Exception:
            children = []
        for child in children:
            visit(child, depth + 1)

    visit(source_dir, 0)
    candidates.sort(key=lambda item: float(item.get("stamp", 0) or 0), reverse=True)
    return candidates


def _write_char_yolo_data_yaml_if_missing(self, source_dir: Path, *, overwrite: bool = False) -> dict:
    source_dir = Path(source_dir)
    yaml_path = source_dir / "data.yaml"
    if yaml_path.exists() and not overwrite:
        return {"ok": True, "created": False, "path": yaml_path, "message": ""}

    has_canonical_split_layout = any((source_dir / "images" / split).exists() for split in ("train", "val", "test"))
    has_nested_split_layout = any((source_dir / split / "images").exists() for split in ("train", "val", "test"))
    lines = [f"path: {source_dir.resolve().as_posix()}"]
    if has_nested_split_layout:
        lines.append("train: train/images")
        lines.append("val: val/images")
        if (source_dir / "test" / "images").exists():
            lines.append("test: test/images")
    elif has_canonical_split_layout:
        lines.append("train: images/train")
        lines.append("val: images/val")
        if (source_dir / "images" / "test").exists():
            lines.append("test: images/test")
    else:
        lines.append("train: images")
        lines.append("val: images")

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines.append(f"nc: {len(alphabet)}")
    lines.append("names:")
    for class_id, char in enumerate(alphabet):
        lines.append(f"  {class_id}: '{char}'")

    try:
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        return {
            "ok": False,
            "created": False,
            "path": yaml_path,
            "message": f"Dataset wygląda poprawnie, ale nie udało się dopisać data.yaml:\n{exc}",
        }

    return {"ok": True, "created": True, "path": yaml_path, "message": ""}


def _validate_char_yolo_split_source(
    self,
    source_raw: str | Path | None = None,
    *,
    create_missing_yaml: bool = False,
    overwrite_incompatible_yaml: bool = False,
    resolve_nested_dataset: bool = False,
) -> dict:
    result = {
        "ok": False,
        "src": None,
        "stats": {},
        "reason": "",
        "yaml_created": False,
        "yaml_rewritten": False,
        "candidate": None,
        "candidate_count": 0,
        "message": (
            "Wskaż dataset znaków: katalog z obrazami i etykietami albo plik data.yaml."
        ),
    }

    raw = str(source_raw if source_raw is not None else getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    if not raw:
        return result

    try:
        src = Path(raw)
    except Exception:
        result["message"] = "Nie udało się odczytać ścieżki źródłowego datasetu."
        return result

    if src.is_file() and src.name.lower() == "data.yaml":
        src = src.parent
    elif src.is_file() and src.name.lower() == "manifest.json":
        result["message"] = self._char_classification_dataset_message()
        return result

    if self._looks_like_char_classification_dataset(src):
        result["message"] = self._char_classification_dataset_message()
        return result

    if not src.exists() or not src.is_dir():
        result["message"] = "Wskazana ścieżka nie jest katalogiem datasetu znaków."
        return result

    if not self._has_supported_yolo_label_layout(src):
        nested_candidates = self._find_char_yolo_dataset_root_candidates(src)
        if nested_candidates:
            best = dict(nested_candidates[0])
            if not resolve_nested_dataset:
                result.update(
                    {
                        "src": src,
                        "stats": dict(best.get("stats") or {}),
                        "reason": "nested_yolo_dataset_found",
                        "candidate": best.get("path"),
                        "candidate_count": len(nested_candidates),
                        "message": (
                            "Wskazany katalog nie jest bezpośrednim katalogiem datasetu znaków, "
                            "ale program znalazł poprawny dataset wewnątrz."
                        ),
                    }
                )
                return result
            src = Path(best.get("path"))
        else:
            result["message"] = (
                "Wskaż dataset znaków: katalog musi zawierać obrazy oraz odpowiadające im etykiety."
            )
            return result

    if not self._has_supported_yolo_label_layout(src):
        result["message"] = (
            "Wskaż dataset znaków: katalog musi zawierać obrazy oraz odpowiadające im etykiety."
        )
        return result

    stats = self._count_yolo_image_label_pairs(src)
    if int(stats.get("total", 0) or 0) <= 0:
        result["message"] = (
            "Dataset ma katalogi obrazów i etykiet, ale nie znaleziono zgodnych par plików. "
            "Sprawdź, czy etykiety mają te same nazwy bazowe co obrazy."
        )
        return result

    yaml_created = False
    if not (src / "data.yaml").exists():
        if not create_missing_yaml:
            result.update(
                {
                    "src": src,
                    "stats": stats,
                    "reason": "missing_yaml",
                    "message": (
                        "Dataset ma poprawne pary obraz + etykieta, ale brakuje pliku data.yaml. "
                        "Ten plik jest wymagany do przygotowania splitu treningowego."
                    ),
                }
            )
            return result

        yaml_state = self._write_char_yolo_data_yaml_if_missing(src)
        if not bool(yaml_state.get("ok")):
            result["message"] = str(yaml_state.get("message") or "Nie udało się przygotować data.yaml.")
            return result
        yaml_created = bool(yaml_state.get("created"))

    yaml_rewritten = False
    try:
        inferred_target = str(self._infer_dataset_target(str(src)) or "").strip().lower()
    except Exception:
        inferred_target = ""
    if inferred_target in {"plate", "vehicle"}:
        if not overwrite_incompatible_yaml:
            result.update(
                {
                    "src": src,
                    "stats": stats,
                    "reason": "incompatible_yaml",
                    "message": (
                        "Folder ma poprawne pary obraz + etykieta, ale istniejący data.yaml "
                        f"opisuje tor {self._format_training_target_label(inferred_target)}, a nie znaki."
                    ),
                }
            )
            return result
        yaml_state = self._write_char_yolo_data_yaml_if_missing(src, overwrite=True)
        if not bool(yaml_state.get("ok")):
            result["message"] = str(yaml_state.get("message") or "Nie udało się zastąpić data.yaml.")
            return result
        yaml_rewritten = True

    result.update(
        {
            "ok": True,
            "src": src,
            "stats": stats,
            "yaml_created": yaml_created,
            "yaml_rewritten": yaml_rewritten,
            "message": (
                "Dataset znaków jest poprawny. "
                f"Znaleziono {int(stats.get('total', 0) or 0)} par obraz + etykieta."
                + (" Dopisano brakujący plik data.yaml." if yaml_created else "")
                + (" Zastąpiono niezgodny plik data.yaml." if yaml_rewritten else "")
            ),
        }
    )
    return result


def _show_char_split_source_validation_modal(self):
    info = self._validate_char_yolo_split_source()
    validation_ok = False
    created_yaml_after_confirmation = False
    rewritten_yaml_after_confirmation = False
    nested_dataset_after_confirmation = False

    while str(info.get("reason") or "") in {"nested_yolo_dataset_found", "missing_yaml", "incompatible_yaml"}:
        reason = str(info.get("reason") or "")
        stats = dict(info.get("stats") or {})
        src = info.get("src")
        try:
            display_path = self._format_workspace_relative_path(src)
        except Exception:
            display_path = str(src or "")
        if reason == "nested_yolo_dataset_found":
            candidate = info.get("candidate")
            try:
                candidate_display = self._format_workspace_relative_path(candidate)
            except Exception:
                candidate_display = str(candidate or "")
            title = "Znaleziono dataset znaków"
            body = (
                "Wskazany katalog nie jest bezpośrednim katalogiem datasetu znaków, "
                "ale program znalazł poprawny dataset wewnątrz.\n\n"
                f"Wskazany folder: {display_path}\n"
                f"Proponowany dataset: {candidate_display}\n"
                f"Liczba znalezionych datasetów w środku: {int(info.get('candidate_count', 0) or 0)}\n"
                f"Pary obraz + etykieta: {int(stats.get('total', 0) or 0)}\n\n"
                "Czy podpiąć ten znaleziony dataset jako źródło PZ1?"
            )
            if not messagebox.askyesno(title, body, parent=getattr(self, "frame", None)):
                messagebox.showinfo(
                    "Nie podpięto datasetu",
                    "Źródło nie zostało zmienione. Wskaż bezpośredni katalog datasetu znaków "
                    "albo wybierz proponowany dataset przy kolejnym wskazaniu.",
                    parent=getattr(self, "frame", None),
                )
                try:
                    self._refresh_dataset_split_cta_state()
                except Exception:
                    pass
                return False
            try:
                self.split_src_var.set(str(candidate))
            except Exception:
                pass
            nested_dataset_after_confirmation = True
            info = self._validate_char_yolo_split_source(candidate)
            continue

        if reason == "missing_yaml":
            title = "Brak pliku data.yaml"
            body = (
                "Wybrany katalog wygląda na dataset znaków, ale nie ma pliku data.yaml.\n\n"
                f"Folder: {display_path}\n"
                f"Pary obraz + etykieta: {int(stats.get('total', 0) or 0)}\n\n"
                "data.yaml opisuje klasy znaków oraz podział train/val/test.\n\n"
                "Czy program ma utworzyć brakujący plik data.yaml w tym folderze?"
            )
            decline_title = "Nie utworzono data.yaml"
            decline_body = (
                "Plik data.yaml nie został utworzony, więc to źródło pozostaje niegotowe dla PZ1/Z4.\n\n"
                "Możesz wybrać inny dataset albo wrócić do tego folderu i ponownie zaakceptować utworzenie pliku."
            )
        else:
            title = "Niezgodny plik data.yaml"
            body = (
                "Wybrany katalog ma pary obraz + etykieta, ale istniejący data.yaml nie opisuje toru znaków.\n\n"
                f"Folder: {display_path}\n"
                f"Pary obraz + etykieta: {int(stats.get('total', 0) or 0)}\n\n"
                "To mogło powstać po dawnym fallbacku splitu albo po wskazaniu datasetu z innego toru.\n\n"
                "Czy program ma zastąpić data.yaml poprawnym opisem datasetu znaków?"
            )
            decline_title = "Nie zastąpiono data.yaml"
            decline_body = (
                "Plik data.yaml nie został zmieniony, więc to źródło pozostaje niegotowe dla toru znaków.\n\n"
                "Możesz wybrać inny dataset albo ponownie zaakceptować zastąpienie pliku."
            )
        should_create = messagebox.askyesno(
            title,
            body,
            parent=getattr(self, "frame", None),
        )
        if should_create:
            info = self._validate_char_yolo_split_source(
                create_missing_yaml=(reason == "missing_yaml"),
                overwrite_incompatible_yaml=(reason == "incompatible_yaml"),
            )
            created_yaml_after_confirmation = bool(info.get("ok")) and bool(info.get("yaml_created"))
            rewritten_yaml_after_confirmation = bool(info.get("ok")) and bool(info.get("yaml_rewritten"))
            continue
        else:
            messagebox.showinfo(
                decline_title,
                decline_body,
                parent=getattr(self, "frame", None),
            )
            try:
                self._refresh_dataset_split_cta_state()
            except Exception:
                pass
            return False

    if bool(info.get("ok")):
        validation_ok = True
        stats = dict(info.get("stats") or {})
        src = info.get("src")
        try:
            display_path = self._format_workspace_relative_path(src)
        except Exception:
            display_path = str(src or "")
        yaml_line = (
            "\nPlik data.yaml: zastąpiony poprawnym opisem znaków."
            if bool(info.get("yaml_rewritten"))
            else
            "\nPlik data.yaml: utworzony po potwierdzeniu."
            if bool(info.get("yaml_created"))
            else "\nPlik data.yaml: obecny."
        )
        title = (
            "Zastąpiono data.yaml"
            if rewritten_yaml_after_confirmation
            else "Utworzono data.yaml"
            if created_yaml_after_confirmation
            else "Podpięto dataset znaków"
            if nested_dataset_after_confirmation
            else "Dataset znaków OK"
        )
        intro = (
            "Zastąpiono niezgodny plik data.yaml i źródło datasetu znaków jest gotowe."
            if rewritten_yaml_after_confirmation
            else
            "Utworzono brakujący plik data.yaml i źródło datasetu znaków jest gotowe."
            if created_yaml_after_confirmation
            else
            "Podpięto znaleziony dataset i źródło datasetu znaków jest gotowe."
            if nested_dataset_after_confirmation
            else "Źródło datasetu znaków jest poprawne."
        )
        layout_summary = _format_char_yolo_source_layout_message(self, stats)
        messagebox.showinfo(
            title,
            f"{intro}\n\n"
            f"Folder: {display_path}\n"
            f"{layout_summary}"
            f"{yaml_line}\n\n"
            "Możesz teraz utworzyć split treningowy.",
            parent=getattr(self, "frame", None),
        )
    else:
        messagebox.showerror(
            "Dataset znaków niegotowy",
            str(info.get("message") or "Wybrane źródło nie przeszło walidacji."),
            parent=getattr(self, "frame", None),
        )

    try:
        self._refresh_dataset_split_cta_state()
    except Exception:
        pass
    try:
        self.frame.after_idle(self._refresh_dataset_split_cta_state)
    except Exception:
        pass
    return validation_ok


def _pick_char_split_source_dir(self):
    previous_value = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    selected = self._pick_dir(
        self.split_src_var,
        initialdir=self._get_char_dataset_source_picker_dir(),
    )
    if selected:
        if not self._show_char_split_source_validation_modal():
            try:
                self.split_src_var.set(previous_value)
            except Exception:
                pass
            try:
                self._refresh_dataset_split_cta_state()
            except Exception:
                pass


def _pick_char_split_source_yaml(self):
    previous_value = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
    selected = self._pick_file(
        self.split_src_var,
        "*.yaml",
        initialdir=self._get_char_dataset_source_picker_dir(),
    )
    if selected:
        if not self._show_char_split_source_validation_modal():
            try:
                self.split_src_var.set(previous_value)
            except Exception:
                pass
            try:
                self._refresh_dataset_split_cta_state()
            except Exception:
                pass
