from __future__ import annotations

"""
Zakładka: Panel kampanii i etapow projektu.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
import xml.etree.ElementTree as ET
import shutil
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
def _build_model_status(self, parent, title, model_type, initial_dir: Path):
    palette = getattr(self.app, "palette", {})

    f = ttk.Frame(parent)
    f.pack(fill=tk.X, pady=10)

    title_lbl = tk.Label(
        f,
        text=title,
        font=("Segoe UI", 10, "bold"),
        fg=palette.get("fg", "#f3f3f3"),
        bg=palette.get("panel", "#252526")
    )
    title_lbl.pack(anchor=tk.W)
    self._model_status_title_labels.append(title_lbl)

    row = ttk.Frame(f)
    row.pack(fill=tk.X)

    lbl_val = tk.Label(
        row,
        text="Domyslny/Brak",
        fg=palette.get("accent", "#4fc1ff"),
        bg=palette.get("panel", "#252526"),
        font=("Consolas", 10)
    )
    lbl_val.pack(side=tk.LEFT, expand=True, anchor=tk.W)

    setattr(self, f"lbl_model_{model_type}", lbl_val)

    lbl_meta = tk.Label(
        f,
        text="Utworzono: -",
        fg=palette.get("muted", "#b0b0b0"),
        bg=palette.get("panel", "#252526"),
        font=("Segoe UI", 8),
    )
    lbl_meta.pack(anchor=tk.W, pady=(2, 0))
    self._model_status_meta_labels.append(lbl_meta)
    setattr(self, f"lbl_model_{model_type}_meta", lbl_meta)
    setattr(self, f"btn_model_{model_type}", None)

def _format_model_created_label(self, model_path: str | Path | None) -> str:
    path_text = str(model_path or "").strip()
    if not path_text:
        return "Utworzono: -"

    try:
        path = Path(path_text)
    except Exception:
        return "Utworzono: -"

    if not path.exists():
        return "Utworzono: -"

    cache = getattr(self, "_dashboard_perf_cache", {})
    model_cache = cache.get("model_created", {}) if isinstance(cache, dict) else {}
    cache_token = self._build_cache_token_for_path(path)
    cache_key = ("created_label", cache_token)
    cached = model_cache.get(cache_key) if isinstance(model_cache, dict) else None
    if isinstance(cached, str):
        return cached

    try:
        created_at = datetime.fromtimestamp(path.stat().st_ctime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Utworzono: -"
    label = f"Utworzono: {created_at}"
    if isinstance(model_cache, dict):
        if len(model_cache) > 128:
            model_cache.clear()
        model_cache[cache_key] = label
    return label

def _get_model_identity_label(self, model_path: str | Path | None) -> str:
    path_text = str(model_path or "").strip()
    if not path_text:
        return ""

    try:
        path = Path(path_text)
    except Exception:
        return ""
    if not path.exists():
        return ""

    cache = getattr(self, "_dashboard_perf_cache", {})
    model_cache = cache.get("model_identity", {}) if isinstance(cache, dict) else {}
    cache_token = self._build_cache_token_for_path(path)
    cache_key = ("identity_label", cache_token)
    cached = model_cache.get(cache_key) if isinstance(model_cache, dict) else None
    if isinstance(cached, str):
        return cached

    try:
        _ok, _message, info = validate_model_file(path)
    except Exception:
        info = {}
    label = format_yolo_model_identity(info)
    if label and isinstance(model_cache, dict):
        if len(model_cache) > 128:
            model_cache.clear()
        model_cache[cache_key] = label
    return label

def _format_model_meta_label(self, model_path: str | Path | None) -> str:
    identity_label = self._get_model_identity_label(model_path)
    created_label = self._format_model_created_label(model_path)
    if identity_label and created_label != "Utworzono: -":
        return f"{identity_label} | {created_label}"
    if identity_label:
        return identity_label
    return created_label

def _count_images_in_dir(self, directory: Path | None, recursive: bool = True) -> int:
    if directory is None or not directory.exists() or not directory.is_dir():
        return 0

    cache = getattr(self, "_dashboard_perf_cache", {})
    image_cache = cache.get("image_counts", {}) if isinstance(cache, dict) else {}
    cache_key = (self._build_cache_token_for_path(directory), bool(recursive))
    cached = image_cache.get(cache_key) if isinstance(image_cache, dict) else None
    if isinstance(cached, int):
        return cached

    try:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        count = sum(
            1
            for image_path in iterator
            if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
        )
        if isinstance(image_cache, dict):
            if len(image_cache) > 256:
                image_cache.clear()
            image_cache[cache_key] = int(count)
        return int(count)
    except Exception:
        return 0

def _collect_image_names_in_dir(self, directory: Path | None, recursive: bool = True) -> set[str]:
    if directory is None or not directory.exists() or not directory.is_dir():
        return set()

    cache = getattr(self, "_dashboard_perf_cache", {})
    name_cache = cache.get("image_name_sets", {}) if isinstance(cache, dict) else {}
    cache_key = (self._build_cache_token_for_path(directory), bool(recursive))
    cached = name_cache.get(cache_key) if isinstance(name_cache, dict) else None
    if isinstance(cached, set):
        return set(cached)

    names: set[str] = set()
    try:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        for image_path in iterator:
            if not image_path.is_file() or image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                continue
            filename = str(image_path.name or "").strip().lower()
            if filename:
                names.add(filename)
    except Exception:
        return set()

    if isinstance(name_cache, dict):
        if len(name_cache) > 128:
            name_cache.clear()
        name_cache[cache_key] = set(names)
    return names

def _collect_previous_project_image_names_for_step1(self) -> set[str]:
    try:
        current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iter = None
    try:
        registry = CAMPAIGN.get_project_packet_filename_registry(exclude_iteration_num=current_iter)
    except Exception:
        return set()
    return {
        str(name or "").strip().lower()
        for name in list((registry or {}).get("filenames") or [])
        if str(name or "").strip()
    }
