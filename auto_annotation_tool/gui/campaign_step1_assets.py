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
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_iteration_paths import (
    default_iteration_path_for_target,
    iteration_path_target,
    normalize_iteration_path,
)
from ..campaign_resource_contracts import build_resource_contract_meta
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from . import campaign_model_status
from .help_manager import HELP
from .run_display import build_run_display_ref
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
def _normalize_project_start_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower()
    if not value:
        return ""
    return "assets" if value == "assets" else "fresh"

def _is_first_iteration_start_context(self) -> bool:
    if not CAMPAIGN.get_active_project_name():
        return False
    try:
        return int(CAMPAIGN.get_current_iteration_num() or 1) == 1 and CAMPAIGN.get_step1_status() != "approved"
    except Exception:
        return False

def _is_step1_operational_context(self) -> bool:
    if not CAMPAIGN.get_active_project_name():
        return False
    try:
        return (
            int(CAMPAIGN.get_current_step() or 1) == 1
            and str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved"
        )
    except Exception:
        return False

def _get_step1_presentation_mode(self) -> str:
    if not CAMPAIGN.get_active_project_name():
        return "summary"
    try:
        current_step = int(CAMPAIGN.get_current_step() or 1)
    except Exception:
        current_step = 1
    step1_status = str(CAMPAIGN.get_step1_status() or "").strip().lower()
    if current_step != 1 or step1_status == "approved":
        return "summary"
    return "operational_assets"

def _get_project_start_mode(self) -> str:
    try:
        mode = CAMPAIGN.get_project_start_mode()
    except Exception:
        mode = ""
    mode = self._normalize_project_start_mode(mode)
    if not mode and self._is_step1_operational_context():
        mode = "fresh"
        try:
            CAMPAIGN.set_project_start_mode(mode)
        except Exception:
            pass
    try:
        self.project_start_mode_var.set(mode)
    except Exception:
        pass
    return mode

def _get_active_step1_draft_plan(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}
    if str(CAMPAIGN.get_step1_status() or "").strip().lower() == "approved":
        return {}
    plan = getattr(self, "current_ingest_plan", None)
    if not isinstance(plan, dict) or not plan:
        return {}
    try:
        current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iter = 1
    current_proj = str(CAMPAIGN.get_active_project_name() or "").strip()
    try:
        plan_iter = int(plan.get("iteration", 0) or 0)
    except Exception:
        plan_iter = 0
    plan_proj = str(plan.get("project", "") or "").strip()
    if plan_iter != current_iter or plan_proj != current_proj:
        return {}
    if plan.get("selected") is None:
        return {}
    return plan

def _set_project_start_mode(self, mode: str | None, *, refresh: bool = True) -> None:
    normalized = self._normalize_project_start_mode(mode)
    try:
        self.project_start_mode_var.set(normalized)
    except Exception:
        pass

    try:
        CAMPAIGN.set_project_start_mode(normalized)
    except Exception:
        pass

    if refresh:
        self._refresh_ingest_panel()

def _project_start_model_candidate_roots(self, model_type: str, scope_override: str | None = None) -> list[Path]:
    normalized_type = "plate" if str(model_type or "").strip().lower() == "plate" else "char"
    row_key = "plate_model" if normalized_type == "plate" else "char_model"
    scope_value = str(scope_override or "").strip().lower()
    if scope_value not in {"project", "freemode"}:
        scope_value = self._get_project_start_asset_scope(row_key)
    scope = "freemode" if scope_value == "freemode" else "project"
    project_root = CAMPAIGN.get_active_project_root_dir()
    project_models_dir = CAMPAIGN.get_dir("models")
    roots: list[Path] = []

    def _add_root(candidate) -> None:
        if candidate is None:
            return
        try:
            path = Path(candidate)
        except Exception:
            return
        if not path.exists() or not path.is_dir():
            return
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if any(str(existing.resolve()).lower() == key for existing in roots if existing.exists()):
            return
        roots.append(path)

    current_path = str(CAMPAIGN.get_global_model(normalized_type) or "").strip()
    if current_path:
        try:
            current_file = Path(current_path)
            if current_file.exists() and current_file.is_file():
                current_dir = current_file.parent
                if scope == "project" and self._path_is_inside_root(current_dir, project_root):
                    _add_root(current_dir)
                elif scope == "freemode" and not self._path_is_inside_root(current_dir, project_root):
                    _add_root(current_dir)
        except Exception:
            pass

    if scope == "project":
        _add_root(project_models_dir)
        if project_root is not None:
            _add_root(Path(project_root) / "6_models")
    else:
        for candidate in CONFIG.get_model_search_dirs(normalized_type):
            _add_root(candidate)
        _add_root(CONFIG.DIR_6_MODELS)

    return roots


def _project_start_model_sidecar_candidates(model_path: Path) -> list[Path]:
    candidates: list[Path] = []

    def _add(candidate: Path | None) -> None:
        if candidate is None:
            return
        try:
            normalized = candidate.resolve()
        except Exception:
            normalized = candidate
        if all(str(existing) != str(normalized) for existing in candidates):
            candidates.append(normalized)

    suffix = str(model_path.suffix or "").strip()
    if suffix:
        _add(model_path.with_suffix(f"{suffix}.metadata.json"))
    _add(model_path.with_suffix(".metadata.json"))
    _add(model_path.with_suffix(".json"))
    _add(model_path.with_name(f"{model_path.stem}_metadata.json"))
    _add(model_path.with_name("model_metadata.json"))
    _add(model_path.with_name("metadata.json"))
    return candidates


def _read_project_start_model_extra_metadata(model_path: Path) -> dict:
    for metadata_path in _project_start_model_sidecar_candidates(Path(model_path)):
        if not metadata_path.exists() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if not metrics and isinstance(extra.get("metrics"), dict):
            metrics = extra.get("metrics") or {}
        training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
        if not training and isinstance(extra.get("training"), dict):
            training = extra.get("training") or {}
        model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        return {
            "metadata_path": str(metadata_path),
            "created_at": str(payload.get("created_at") or "").strip(),
            "metrics": dict(metrics or {}),
            "training": dict(training or {}),
            "model": dict(model_payload or {}),
            "schema": str(payload.get("schema") or "").strip(),
        }
    return {}


def _project_start_metric_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _project_start_metric_percent(value) -> str:
    numeric = _project_start_metric_float(value)
    if numeric is None:
        return "-"
    if abs(numeric) <= 1.000001:
        numeric *= 100.0
    return f"{numeric:.1f}%"


def _project_start_metric_value(metrics: dict, keys: tuple[str, ...]):
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key in metrics and metrics.get(key) not in (None, ""):
            return metrics.get(key)
    latest = metrics.get("latest") if isinstance(metrics.get("latest"), dict) else {}
    best_row = metrics.get("best_row") if isinstance(metrics.get("best_row"), dict) else {}
    for source in (best_row, latest):
        for key in keys:
            if key in source and source.get(key) not in (None, ""):
                return source.get(key)
    return None


def _project_start_model_class_names(info: dict) -> list[str]:
    return [str(name).strip() for name in (info.get("classes") or []) if str(name).strip()]


def _project_start_looks_like_character_model_classes(class_names: list[str]) -> bool:
    normalized = [str(name).strip().upper() for name in class_names if str(name).strip()]
    if len(normalized) < 8:
        return False
    allowed = set(CHAR_ALPHABET)
    return all(len(token) == 1 and token in allowed for token in normalized)


def _project_start_is_pose_model_info(info: dict) -> bool:
    task = str(info.get("task") or "").strip().lower()
    inferred_type = str(info.get("type") or "").strip().lower()
    kpt_shape = info.get("kpt_shape")
    has_keypoints = bool(info.get("keypoints")) or bool(kpt_shape)
    if has_keypoints or task == "pose" or inferred_type == "pose":
        return True
    if task in {"detect", "detection"} or inferred_type in {"detect", "detection"}:
        return False
    architecture_text = " ".join(
        str(info.get(key) or "").strip().lower()
        for key in ("architecture_label", "source_architecture_label")
    )
    return "pose" in architecture_text


def _summarize_project_start_model_candidate(self, model_path: Path, model_type: str, root: Path | None = None) -> dict:
    normalized_type = "plate" if str(model_type or "").strip().lower() == "plate" else "char"
    safe_path = Path(model_path)
    try:
        stat = safe_path.stat()
        mtime = float(getattr(stat, "st_mtime", 0.0) or 0.0)
        size_mb = float(getattr(stat, "st_size", 0) or 0) / (1024 * 1024)
    except Exception:
        mtime = 0.0
        size_mb = 0.0

    try:
        ok_light, message, info = validate_model_file(
            safe_path,
            allow_heavy_load=False,
            write_sidecar=False,
        )
    except Exception as exc:
        ok_light, message, info = False, str(exc), {}
    info = dict(info or {})
    extra = _read_project_start_model_extra_metadata(safe_path)
    metrics = dict(extra.get("metrics") or {})
    training = dict(extra.get("training") or {})
    model_payload = extra.get("model") if isinstance(extra.get("model"), dict) else {}
    model_info = model_payload.get("info") if isinstance(model_payload.get("info"), dict) else {}
    if model_info:
        for key, value in model_info.items():
            if key not in info or info.get(key) in (None, "", [], {}):
                info[key] = value

    task = str(info.get("task") or info.get("type") or "").strip().lower()
    identity = str(format_yolo_model_identity(info) or "").strip()
    if not identity:
        identity = str(info.get("architecture_label") or "").strip()
    if not identity:
        identity = "YOLO Pose" if "pose" in safe_path.name.lower() else "YOLO Detect" if safe_path.suffix.lower() == ".pt" else "Nieustalone"

    class_names = _project_start_model_class_names(info)
    looks_like_char_model = _project_start_looks_like_character_model_classes(class_names)
    is_pose = _project_start_is_pose_model_info(info)
    if normalized_type == "plate":
        target_match = is_pose and not looks_like_char_model
        expected_text = "MT / tablice"
    else:
        target_match = (not is_pose) and looks_like_char_model
        expected_text = "MZ / znaki"

    map50 = _project_start_metric_value(
        metrics,
        ("pose_map50", "box_map50", "map50", "best_map50", "metrics/mAP50(P)", "metrics/mAP50(B)", "metrics/mAP50"),
    )
    map5095 = _project_start_metric_value(
        metrics,
        (
            "pose_map50_95",
            "box_map50_95",
            "map50_95",
            "best_map50_95",
            "metrics/mAP50-95(P)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ),
    )
    precision = _project_start_metric_value(
        metrics,
        ("pose_precision", "box_precision", "precision", "metrics/precision(P)", "metrics/precision(B)"),
    )
    recall = _project_start_metric_value(
        metrics,
        ("pose_recall", "box_recall", "recall", "metrics/recall(P)", "metrics/recall(B)"),
    )

    if target_match and (ok_light or extra):
        tone = "success"
        status = "Pasuje"
    elif target_match:
        tone = "warning"
        status = "Brak lekkich metryk"
    else:
        tone = "error"
        status = "Inny typ"

    try:
        rel_path = self._format_project_start_asset_source(str(safe_path))
    except Exception:
        rel_path = str(safe_path)
    try:
        root_label = self._format_project_start_asset_source(str(root)) if root is not None else ""
    except Exception:
        root_label = str(root or "")

    return {
        "path": str(safe_path),
        "name": safe_path.name,
        "root": root_label,
        "relative_path": rel_path,
        "identity": identity,
        "expected": expected_text,
        "status": status,
        "tone": tone,
        "message": str(message or "").strip(),
        "target_match": bool(target_match),
        "map50": map50,
        "map50_95": map5095,
        "precision": precision,
        "recall": recall,
        "map50_text": _project_start_metric_percent(map50),
        "map50_95_text": _project_start_metric_percent(map5095),
        "precision_text": _project_start_metric_percent(precision),
        "recall_text": _project_start_metric_percent(recall),
        "size_mb": float(size_mb),
        "size_text": f"{size_mb:.1f} MB" if size_mb > 0 else "-",
        "mtime": mtime,
        "created_at": str(extra.get("created_at") or training.get("finished_at") or training.get("created_at") or "").strip(),
        "metadata_path": str(extra.get("metadata_path") or "").strip(),
    }


def _find_project_start_model_candidates(
    self,
    model_type: str,
    *,
    scope_override: str | None = None,
) -> tuple[list[dict], list[Path]]:
    normalized_type = "plate" if str(model_type or "").strip().lower() == "plate" else "char"
    roots = self._project_start_model_candidate_roots(normalized_type, scope_override=scope_override)
    candidates: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        try:
            model_paths = list(Path(root).rglob("*.pt"))
        except Exception:
            model_paths = []
        for model_path in model_paths:
            try:
                key = str(Path(model_path).resolve()).lower()
            except Exception:
                key = str(model_path).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(self._summarize_project_start_model_candidate(Path(model_path), normalized_type, root))
            if len(candidates) >= 260:
                break
        if len(candidates) >= 260:
            break

    def _score(item: dict) -> tuple[int, float, float, float]:
        map5095 = _project_start_metric_float(item.get("map50_95")) or -1.0
        map50 = _project_start_metric_float(item.get("map50")) or -1.0
        if abs(map5095) <= 1.000001:
            map5095 *= 100.0
        if abs(map50) <= 1.000001:
            map50 *= 100.0
        return (
            1 if bool(item.get("target_match")) else 0,
            float(map5095),
            float(map50),
            float(item.get("mtime", 0.0) or 0.0),
        )

    candidates.sort(key=_score, reverse=True)
    return candidates, roots


def _open_project_start_model_candidate_browser(self, model_type: str, parent=None) -> bool:
    normalized_type = "plate" if str(model_type or "").strip().lower() == "plate" else "char"
    row_key = "plate_model" if normalized_type == "plate" else "char_model"
    title_label = "model tablic MT" if normalized_type == "plate" else "model znaków MZ"
    model_scope_state = {
        "scope": "freemode" if self._get_project_start_asset_scope(row_key) == "freemode" else "project",
    }
    candidates: list[dict] = []
    try:
        roots = self._project_start_model_candidate_roots(
            normalized_type,
            scope_override=str(model_scope_state.get("scope") or "project"),
        )
    except Exception:
        roots = []
    model_load_state: dict[str, object] = {
        "loading": True,
        "error": "",
        "elapsed_ms": 0,
        "token": 0,
    }
    palette = getattr(self.app, "palette", {})
    card_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", "#1a1a1a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f39c12")
    error = palette.get("error", "#e74c3c")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    browser = tk.Toplevel(parent or self.frame)
    try:
        self.app.style_dialog_window(browser, title=f"Wybierz {title_label}", geometry="1180x680", parent=parent or self.frame)
    except Exception:
        browser.title(f"Wybierz {title_label}")
    build_surface = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(build_surface):
        body = build_surface(browser, tone="info")
    else:
        body = tk.Frame(browser, bg=card_bg)
        body.pack(fill=tk.BOTH, expand=True)
    body_bg = str(body.cget("bg") or card_bg)

    title_lbl = tk.Label(
        body,
        text=f"Wybierz {title_label} z dostępnych kandydatów",
        fg=fg,
        bg=body_bg,
        font=("Segoe UI", 13, "bold"),
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=16, pady=(14, 4))
    model_scope_button_refs: dict[str, tk.Button] = {}
    model_scope_info_var = tk.StringVar(value="")

    def _refresh_model_scope_info() -> None:
        active_scope = "freemode" if str(model_scope_state.get("scope") or "").strip().lower() == "freemode" else "project"
        scope_label = "Swobodny" if active_scope == "freemode" else "Projekt"
        roots_text = ", ".join(str(root.name or root) for root in roots[:4]) or "brak katalogów"
        if len(roots) > 4:
            roots_text += f" +{len(roots) - 4}"
        try:
            model_scope_info_var.set(
                campaign_ui_helpers._repair_polish_text(
                    f"Tryb: {scope_label}. Skanowane katalogi: {roots_text}. "
                    "Tabela pokazuje zgodność typu modelu i skrótowe metryki, jeśli plik ma lekkie metadane."
                )
            )
        except Exception:
            pass
        for scope_value, button in list(model_scope_button_refs.items()):
            try:
                active_button = scope_value == active_scope
                button.config(
                    bg=blend_hex_colors(field_bg, success, 0.24) if active_button else field_bg,
                    fg=fg if active_button else muted,
                    font=("Segoe UI", 8, "bold" if active_button else "normal"),
                )
            except Exception:
                pass

    def _switch_model_candidate_scope(next_scope: str) -> None:
        nonlocal candidates, roots
        normalized_scope = "freemode" if str(next_scope or "").strip().lower() == "freemode" else "project"
        if normalized_scope == str(model_scope_state.get("scope") or "").strip().lower():
            return
        try:
            self._set_project_start_asset_scope(row_key, normalized_scope, persist=True)
        except Exception:
            pass
        model_scope_state["scope"] = normalized_scope
        candidates = []
        try:
            roots = self._project_start_model_candidate_roots(
                normalized_type,
                scope_override=normalized_scope,
            )
        except Exception:
            roots = []
        load_token = int(model_load_state.get("token", 0) or 0) + 1
        model_load_state["token"] = load_token
        model_load_state["loading"] = True
        model_load_state["error"] = ""
        model_load_state["elapsed_ms"] = 0
        try:
            model_sort_state["column"] = None
            model_sort_state["reverse"] = True
        except Exception:
            pass
        _refresh_model_scope_info()
        _set_model_loading_progress(True)
        try:
            _refresh_light_model_table()
            tree.yview_moveto(0)
        except Exception:
            try:
                _render_model_table()
                canvas.yview_moveto(0)
            except Exception:
                pass
        _start_model_candidate_loading(load_token=load_token)

    tk.Label(
        body,
        textvariable=model_scope_info_var,
        fg=muted,
        bg=body_bg,
        justify=tk.LEFT,
        anchor="w",
        wraplength=1100,
    ).pack(fill=tk.X, padx=16, pady=(0, 12))
    model_scope_shell = tk.Frame(
        body,
        bg=blend_hex_colors(body_bg, success, 0.045),
        bd=0,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(border, success, 0.18),
        highlightcolor=blend_hex_colors(border, success, 0.18),
    )
    model_scope_shell.pack(fill=tk.X, padx=16, pady=(0, 12))
    tk.Label(
        model_scope_shell,
        text="Zakres źródeł modelu",
        fg=fg,
        bg=str(model_scope_shell.cget("bg") or body_bg),
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(side=tk.LEFT, padx=(10, 12), pady=8)
    for scope_value, scope_text in (("project", "Projekt"), ("freemode", "Swobodny")):
        button = tk.Button(
            model_scope_shell,
            text=scope_text,
            command=lambda value=scope_value: _switch_model_candidate_scope(value),
            cursor="hand2",
            bg=field_bg,
            fg=muted,
            activebackground=blend_hex_colors(field_bg, success, 0.30),
            activeforeground=fg,
            relief=tk.FLAT,
            font=("Segoe UI", 8),
            padx=12,
            pady=5,
        )
        button.pack(side=tk.LEFT, padx=(0, 6), pady=7)
        model_scope_button_refs[scope_value] = button
    tk.Label(
        model_scope_shell,
        text="Zakres dotyczy tylko tej listy kandydatów.",
        fg=muted,
        bg=str(model_scope_shell.cget("bg") or body_bg),
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(side=tk.LEFT, padx=(8, 10), pady=8)
    _refresh_model_scope_info()

    model_progress_state = {"visible": False, "running": False}
    model_progress_var = tk.StringVar(value="")
    model_progress_shell = tk.Frame(
        body,
        bg=blend_hex_colors(body_bg, success, 0.045),
        bd=0,
        highlightthickness=1,
        highlightbackground=blend_hex_colors(border, success, 0.20),
        highlightcolor=blend_hex_colors(border, success, 0.20),
    )
    model_progress_label = tk.Label(
        model_progress_shell,
        textvariable=model_progress_var,
        fg=success,
        bg=str(model_progress_shell.cget("bg") or body_bg),
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    )
    model_progress_label.pack(side=tk.LEFT, padx=(10, 12), pady=8)
    model_progress_bar = ttk.Progressbar(
        model_progress_shell,
        mode="indeterminate",
        length=220,
    )
    model_progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(0, 10), pady=9)

    def _set_model_loading_progress(visible: bool, text: str = "") -> None:
        if visible:
            model_progress_var.set(
                campaign_ui_helpers._repair_polish_text(
                    text or "Wczytuję modele i metryki..."
                )
            )
            if not bool(model_progress_state.get("visible")):
                try:
                    model_progress_shell.pack(fill=tk.X, padx=16, pady=(0, 12))
                except Exception:
                    pass
                model_progress_state["visible"] = True
            if not bool(model_progress_state.get("running")):
                try:
                    model_progress_bar.start(14)
                except Exception:
                    pass
                model_progress_state["running"] = True
            return
        if bool(model_progress_state.get("running")):
            try:
                model_progress_bar.stop()
            except Exception:
                pass
            model_progress_state["running"] = False
        if bool(model_progress_state.get("visible")):
            try:
                model_progress_shell.pack_forget()
            except Exception:
                pass
            model_progress_state["visible"] = False

    _set_model_loading_progress(True)

    table_shell = tk.Frame(
        body,
        bg=body_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    table_shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))
    canvas = tk.Canvas(table_shell, bg=body_bg, bd=0, highlightthickness=0)
    scroll = ttk.Scrollbar(table_shell, orient=tk.VERTICAL, command=canvas.yview)
    rows_frame = tk.Frame(canvas, bg=body_bg)
    canvas_window = canvas.create_window((0, 0), window=rows_frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_sync_after_id = {"id": None}

    def _browser_move_active() -> bool:
        try:
            return bool(getattr(browser, "_aat_dialog_move_active", False))
        except Exception:
            return False

    def _sync_scrollregion(_event=None) -> None:
        scroll_sync_after_id["id"] = None
        if _browser_move_active():
            try:
                scroll_sync_after_id["id"] = browser.after(140, _sync_scrollregion)
            except Exception:
                pass
            return
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(canvas_window, width=max(1, int(canvas.winfo_width() or 1)))
        except Exception:
            pass

    def _schedule_sync_scrollregion(_event=None) -> None:
        if scroll_sync_after_id.get("id") is not None:
            return
        try:
            if _browser_move_active():
                scroll_sync_after_id["id"] = browser.after(140, _sync_scrollregion)
            else:
                scroll_sync_after_id["id"] = browser.after_idle(_sync_scrollregion)
        except Exception:
            _sync_scrollregion()

    rows_frame.bind("<Configure>", _schedule_sync_scrollregion)
    canvas.bind("<Configure>", _schedule_sync_scrollregion)

    header = ("Model", "Typ", "Status", "mAP50-95", "mAP50", "Precision", "Recall", "Rozmiar", "Ścieżka", "Wskaż")
    column_weights = (4, 3, 1, 0, 0, 0, 0, 0, 1, 0)
    for col, weight in enumerate(column_weights):
        rows_frame.grid_columnconfigure(col, weight=weight, minsize=0)
    header_bg = blend_hex_colors(field_bg, success, 0.10)
    model_sort_state = {"column": None, "reverse": True}

    def _choose_candidate(candidate: dict) -> None:
        model_path = Path(str(candidate.get("path") or ""))
        is_valid, error_message = self._validate_project_model_selection(normalized_type, model_path)
        candidate_matches_table_filter = (
            bool(candidate.get("target_match"))
            and str(candidate.get("status") or "").strip().lower() == "pasuje"
        )
        if (
            not is_valid
            and candidate_matches_table_filter
            and model_path.exists()
            and model_path.suffix.lower() == ".pt"
        ):
            # The table already accepted this model from lightweight metadata.
            # Keep selection consistent with what the user sees in the browser.
            is_valid = True
            error_message = ""
        if not is_valid:
            error_title = campaign_ui_helpers._repair_polish_text("Nieprawidłowy model")
            error_body = campaign_ui_helpers._repair_polish_text(str(error_message or "Wybrany plik nie jest właściwym modelem."))
            try:
                self.app.themed_error(error_title, error_body, parent=browser)
            except Exception:
                messagebox.showerror(error_title, error_body, parent=browser)
            return
        try:
            CAMPAIGN.set_global_model(normalized_type, str(model_path))
            self._set_project_start_asset_scope(
                row_key,
                self._infer_project_start_asset_scope_from_path(row_key, model_path),
            )
            self._clear_dashboard_perf_cache()
            self._sync_iteration_artifact_registry_from_project_start()
            self._refresh_dashboard()
            self.app.update_status(f"Wybrano {title_label}: {model_path.name}", "info")
        except Exception as exc:
            logger.debug(f"Nie udało się ustawić modelu startowego {normalized_type}: {exc}")
        _close_browser()

    def _model_sort_metric(value) -> float:
        numeric = _project_start_metric_float(value)
        if numeric is None:
            return -1.0
        if abs(numeric) <= 1.000001:
            numeric *= 100.0
        return float(numeric)

    def _model_sort_value(candidate: dict, col: int):
        if col == 0:
            return str(candidate.get("name") or "").strip().lower()
        if col == 1:
            return str(candidate.get("identity") or "").strip().lower()
        if col == 2:
            tone_rank = {"success": 3, "warning": 2, "error": 1}
            return (
                int(tone_rank.get(str(candidate.get("tone") or "").strip(), 0)),
                str(candidate.get("status") or "").strip().lower(),
            )
        if col == 3:
            return _model_sort_metric(candidate.get("map50_95"))
        if col == 4:
            return _model_sort_metric(candidate.get("map50"))
        if col == 5:
            return _model_sort_metric(candidate.get("precision"))
        if col == 6:
            return _model_sort_metric(candidate.get("recall"))
        if col == 7:
            return float(candidate.get("size_mb", 0.0) or 0.0)
        if col == 8:
            return str(candidate.get("relative_path") or "").strip().lower()
        return ""

    def _sorted_model_candidates() -> list[dict]:
        col = model_sort_state.get("column")
        if col is None:
            return list(candidates)
        return sorted(
            list(candidates),
            key=lambda item: (
                _model_sort_value(item, int(col)),
                float(item.get("mtime", 0.0) or 0.0),
                str(item.get("name") or "").strip().lower(),
            ),
            reverse=bool(model_sort_state.get("reverse")),
        )

    def _set_model_sort(col: int) -> None:
        if col >= len(header) - 1:
            return
        if model_sort_state.get("column") == col:
            model_sort_state["reverse"] = not bool(model_sort_state.get("reverse"))
        else:
            model_sort_state["column"] = col
            model_sort_state["reverse"] = col in {2, 3, 4, 5, 6, 7}
        _render_model_table()
        try:
            canvas.yview_moveto(0)
        except Exception:
            pass

    def _render_model_table() -> None:
        for child in list(rows_frame.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        active_col = model_sort_state.get("column")
        reverse = bool(model_sort_state.get("reverse"))
        for col, text in enumerate(header):
            suffix = " ↓" if active_col == col and reverse else " ↑" if active_col == col else ""
            if col < len(header) - 1:
                tk.Button(
                    rows_frame,
                    text=f"{text}{suffix}",
                    command=lambda column=col: _set_model_sort(column),
                    cursor="hand2",
                    bg=header_bg,
                    fg=fg,
                    relief=tk.FLAT,
                    font=("Segoe UI", 8, "bold"),
                    anchor="w",
                    padx=8,
                    pady=7,
                ).grid(row=0, column=col, sticky="nsew")
            else:
                tk.Label(
                    rows_frame,
                    text=text,
                    fg=fg,
                    bg=header_bg,
                    font=("Segoe UI", 8, "bold"),
                    anchor="w",
                    padx=8,
                    pady=7,
                ).grid(row=0, column=col, sticky="nsew")

        if not candidates:
            tk.Label(
                rows_frame,
                text=campaign_ui_helpers._repair_polish_text(
                    "Nie znaleziono plików .pt w wybranym trybie. Możesz wskazać model ręcznie."
                ),
                fg=warning,
                bg=body_bg,
                font=("Segoe UI", 9),
                anchor="w",
                justify=tk.LEFT,
                padx=10,
                pady=16,
                wraplength=1020,
            ).grid(row=1, column=0, columnspan=len(header), sticky="nsew")
            return

        for row_idx, candidate in enumerate(_sorted_model_candidates(), start=1):
            row_bg = blend_hex_colors(field_bg, card_bg, 0.25) if row_idx % 2 else field_bg
            tone = str(candidate.get("tone") or "").strip()
            status_color = success if tone == "success" else warning if tone == "warning" else error
            values = (
                str(candidate.get("name") or "-"),
                str(candidate.get("identity") or "-"),
                str(candidate.get("status") or "-"),
                str(candidate.get("map50_95_text") or "-"),
                str(candidate.get("map50_text") or "-"),
                str(candidate.get("precision_text") or "-"),
                str(candidate.get("recall_text") or "-"),
                str(candidate.get("size_text") or "-"),
                shorten(str(candidate.get("relative_path") or "-"), width=24, placeholder="..."),
            )
            for col, value in enumerate(values):
                color = status_color if col == 2 else success if col in {3, 4} and value != "-" else fg
                tk.Label(
                    rows_frame,
                    text=campaign_ui_helpers._repair_polish_text(str(value or "")),
                    fg=color,
                    bg=row_bg,
                    font=("Segoe UI", 8, "bold" if col in {2, 3, 4} else "normal"),
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=(320 if col == 0 else 220 if col == 1 else 110 if col == 8 else 90),
                    padx=8,
                    pady=7,
                ).grid(row=row_idx, column=col, sticky="nsew")
            tk.Button(
                rows_frame,
                text="[Wskaż]",
                command=lambda item=candidate: _choose_candidate(item),
                cursor="hand2",
                bg=blend_hex_colors(field_bg, success, 0.18),
                fg=fg,
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=4,
            ).grid(row=row_idx, column=9, sticky="nsew", padx=4, pady=4)
        _schedule_sync_scrollregion()

    _render_model_table()
    try:
        table_shell.destroy()
    except Exception:
        pass

    table_shell = tk.Frame(
        body,
        bg=body_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    table_shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))
    tree_columns = tuple(f"c{i}" for i in range(len(header)))
    tree = ttk.Treeview(table_shell, columns=tree_columns, show="headings", selectmode="browse", height=16)
    tree_scroll = ttk.Scrollbar(table_shell, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
    try:
        style = ttk.Style(tree)
        style.configure(
            "CampaignModelCandidates.Treeview",
            background=field_bg,
            fieldbackground=field_bg,
            foreground=fg,
            rowheight=28,
        )
        style.map(
            "CampaignModelCandidates.Treeview",
            background=[("selected", field_bg)],
            foreground=[("selected", fg)],
        )
        style.configure("CampaignModelCandidates.Treeview.Heading", font=("Segoe UI", 8, "bold"))
        tree.configure(style="CampaignModelCandidates.Treeview")
    except Exception:
        pass
    column_widths = (238, 150, 72, 68, 52, 64, 50, 56, 78, 62)
    light_candidate_by_iid: dict[str, dict] = {}
    light_model_hover_state: dict[str, str] = {"row_id": "", "column_id": ""}
    light_model_hover = tk.Label(
        tree,
        text="",
        fg=fg,
        bg=blend_hex_colors(field_bg, success, 0.20),
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 8),
        anchor="center",
        cursor="hand2",
    )

    def _hide_light_model_hover() -> None:
        try:
            light_model_hover.place_forget()
        except Exception:
            pass
        light_model_hover_state["row_id"] = ""
        light_model_hover_state["column_id"] = ""

    def _light_model_hover_target(event=None) -> tuple[str, str] | None:
        try:
            row_id = str(tree.identify_row(event.y) or "")
            column_id = str(tree.identify_column(event.x) or "")
            if row_id and row_id in light_candidate_by_iid and column_id == f"#{len(header)}":
                return row_id, column_id
        except Exception:
            return None
        return None

    def _show_light_model_hover(row_id: str, column_id: str) -> None:
        try:
            if (
                row_id == str(light_model_hover_state.get("row_id") or "")
                and column_id == str(light_model_hover_state.get("column_id") or "")
            ):
                return
            bbox = tree.bbox(row_id, column_id)
            if not bbox:
                _hide_light_model_hover()
                return
            x, y, width, height = bbox
            text = str(tree.set(row_id, tree_columns[-1]) or "")
            if not text:
                _hide_light_model_hover()
                return
            light_model_hover.config(text=text)
            light_model_hover.place(
                x=int(x) + 1,
                y=int(y) + 1,
                width=max(1, int(width) - 2),
                height=max(1, int(height) - 2),
            )
            light_model_hover_state["row_id"] = row_id
            light_model_hover_state["column_id"] = column_id
        except Exception:
            _hide_light_model_hover()

    def _activate_light_model_hover(_event=None) -> str:
        try:
            row_id = str(light_model_hover_state.get("row_id") or "")
            if row_id and row_id in light_candidate_by_iid:
                tree.selection_set(row_id)
                tree.focus(row_id)
                _choose_candidate(light_candidate_by_iid[row_id])
        except Exception:
            pass
        return "break"

    def _update_light_model_hover(event=None) -> None:
        target = _light_model_hover_target(event)
        try:
            if target:
                tree.configure(cursor="hand2")
                _show_light_model_hover(target[0], target[1])
            else:
                tree.configure(cursor="")
                _hide_light_model_hover()
        except Exception:
            pass

    def _refresh_light_model_table() -> None:
        _hide_light_model_hover()
        for item in tree.get_children(""):
            tree.delete(item)
        light_candidate_by_iid.clear()
        active_col = model_sort_state.get("column")
        reverse = bool(model_sort_state.get("reverse"))
        for col, col_id in enumerate(tree_columns):
            suffix = " ↓" if active_col == col and reverse else " ↑" if active_col == col else ""
            tree.heading(
                col_id,
                text=campaign_ui_helpers._repair_polish_text(f"{header[col]}{suffix}"),
                anchor="w",
            )
        if bool(model_load_state.get("loading")) or str(model_load_state.get("error") or "").strip():
            error_text = str(model_load_state.get("error") or "").strip()
            message = (
                f"Nie udało się wczytać listy modeli: {error_text}"
                if error_text
                else "Wczytuję listę modeli i metryki..."
            )
            tree.insert(
                "",
                "end",
                values=(
                    campaign_ui_helpers._repair_polish_text(message),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ),
            )
            return
        if not candidates:
            tree.insert(
                "",
                "end",
                values=(
                    "Nie znaleziono plików .pt w wybranym trybie.",
                    "",
                    "[Wskaż ręcznie]",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ),
            )
            return
        for row_idx, candidate in enumerate(_sorted_model_candidates(), start=1):
            values = (
                str(candidate.get("name") or "-"),
                str(candidate.get("identity") or "-"),
                str(candidate.get("status") or "-"),
                str(candidate.get("map50_95_text") or "-"),
                str(candidate.get("map50_text") or "-"),
                str(candidate.get("precision_text") or "-"),
                str(candidate.get("recall_text") or "-"),
                str(candidate.get("size_text") or "-"),
                shorten(str(candidate.get("relative_path") or "-"), width=24, placeholder="..."),
                "[Wskaż]",
            )
            iid = f"model-{row_idx}"
            light_candidate_by_iid[iid] = candidate
            tree.insert(
                "",
                "end",
                iid=iid,
                values=tuple(campaign_ui_helpers._repair_polish_text(str(value or "")) for value in values),
            )

    def _start_model_candidate_loading(load_token: int | None = None) -> None:
        nonlocal candidates, roots
        if load_token is None:
            load_token = int(model_load_state.get("token", 0) or 0) + 1
            model_load_state["token"] = load_token
        active_scope = "freemode" if str(model_scope_state.get("scope") or "").strip().lower() == "freemode" else "project"
        model_load_state["loading"] = True
        model_load_state["error"] = ""
        _set_model_loading_progress(True)

        def _worker() -> None:
            result_candidates: list[dict] = []
            result_roots: list[Path] = list(roots or [])
            error_text = ""
            started = perf_counter()
            try:
                result_candidates, result_roots = self._find_project_start_model_candidates(
                    normalized_type,
                    scope_override=active_scope,
                )
            except Exception as exc:
                error_text = str(exc)
                try:
                    logger.exception("Nie udało się przygotować listy modeli startowych")
                except Exception:
                    pass
            elapsed_ms = int((perf_counter() - started) * 1000)

            def _finish() -> None:
                nonlocal candidates, roots
                try:
                    if not browser.winfo_exists():
                        return
                except Exception:
                    return
                if int(model_load_state.get("token", 0) or 0) != int(load_token or 0):
                    return
                candidates = list(result_candidates or [])
                roots = list(result_roots or [])
                model_load_state["loading"] = False
                model_load_state["error"] = error_text
                model_load_state["elapsed_ms"] = elapsed_ms
                _set_model_loading_progress(False)
                _refresh_model_scope_info()
                _refresh_light_model_table()
                try:
                    tree.yview_moveto(0)
                except Exception:
                    pass

            try:
                browser.after(0, _finish)
            except Exception:
                pass

        try:
            threading.Thread(
                target=_worker,
                daemon=True,
                name=f"campaign-{normalized_type}-model-candidates",
            ).start()
        except Exception as exc:
            model_load_state["loading"] = False
            model_load_state["error"] = str(exc)
            _set_model_loading_progress(False)
            _refresh_light_model_table()

    def _set_light_model_sort(col: int) -> None:
        if col >= len(header) - 1:
            return
        if model_sort_state.get("column") == col:
            model_sort_state["reverse"] = not bool(model_sort_state.get("reverse"))
        else:
            model_sort_state["column"] = col
            model_sort_state["reverse"] = col in {2, 3, 4, 5, 6, 7}
        _refresh_light_model_table()
        try:
            tree.yview_moveto(0)
        except Exception:
            pass

    def _choose_light_model_selection() -> None:
        try:
            selection = tree.selection()
            if not selection:
                return
            candidate = light_candidate_by_iid.get(str(selection[0]))
            if candidate:
                _choose_candidate(candidate)
        except Exception:
            pass

    def _on_light_model_tree_click(event=None):
        try:
            row_id = str(tree.identify_row(event.y) or "")
            if not row_id or row_id not in light_candidate_by_iid:
                return
            tree.selection_set(row_id)
            tree.focus(row_id)
            if str(tree.identify_column(event.x) or "") == f"#{len(header)}":
                _choose_candidate(light_candidate_by_iid[row_id])
        except Exception:
            pass

    for col, col_id in enumerate(tree_columns):
        tree.heading(col_id, command=lambda column=col: _set_light_model_sort(column))
        anchor = "center" if col in {3, 4, 5, 6, 7, 9} else "w"
        tree.column(col_id, width=column_widths[col], minwidth=38, anchor=anchor, stretch=(col in {0, 1}))
    tree.bind("<Double-1>", lambda _event: _choose_light_model_selection(), add="+")
    tree.bind("<Return>", lambda _event: _choose_light_model_selection(), add="+")
    tree.bind("<ButtonRelease-1>", _on_light_model_tree_click, add="+")
    tree.bind("<Motion>", _update_light_model_hover, add="+")
    tree.bind("<Leave>", lambda _event: (_hide_light_model_hover(), tree.configure(cursor="")), add="+")
    light_model_hover.bind("<ButtonRelease-1>", _activate_light_model_hover, add="+")
    light_model_hover.bind("<Leave>", lambda _event: _hide_light_model_hover(), add="+")
    _refresh_light_model_table()
    try:
        browser.after(80, _start_model_candidate_loading)
    except Exception:
        _start_model_candidate_loading()

    def _close_browser() -> None:
        _set_model_loading_progress(False)
        pending = scroll_sync_after_id.get("id")
        if pending is not None:
            try:
                browser.after_cancel(pending)
            except Exception:
                pass
            scroll_sync_after_id["id"] = None
        try:
            browser.destroy()
        except Exception:
            pass

    if False:
        tk.Label(
            rows_frame,
            text=campaign_ui_helpers._repair_polish_text(
                "Nie znaleziono plików .pt w wybranym trybie. Możesz wskazać model ręcznie."
            ),
            fg=warning,
            bg=body_bg,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            padx=10,
            pady=16,
            wraplength=1020,
        ).grid(row=1, column=0, columnspan=len(header), sticky="nsew")
    elif False:
        for row_idx, candidate in enumerate(candidates, start=1):
            row_bg = blend_hex_colors(field_bg, card_bg, 0.25) if row_idx % 2 else field_bg
            tone = str(candidate.get("tone") or "").strip()
            status_color = success if tone == "success" else warning if tone == "warning" else error
            values = (
                str(candidate.get("name") or "-"),
                str(candidate.get("identity") or "-"),
                str(candidate.get("status") or "-"),
                str(candidate.get("map50_95_text") or "-"),
                str(candidate.get("map50_text") or "-"),
                str(candidate.get("precision_text") or "-"),
                str(candidate.get("recall_text") or "-"),
                str(candidate.get("size_text") or "-"),
                shorten(str(candidate.get("relative_path") or "-"), width=24, placeholder="..."),
            )
            for col, value in enumerate(values):
                color = status_color if col == 2 else success if col in {3, 4} and value != "-" else fg
                tk.Label(
                    rows_frame,
                    text=campaign_ui_helpers._repair_polish_text(str(value or "")),
                    fg=color,
                    bg=row_bg,
                    font=("Segoe UI", 8, "bold" if col in {2, 3, 4} else "normal"),
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=(210 if col == 0 else 140 if col == 1 else 260 if col == 8 else 90),
                    padx=8,
                    pady=7,
                ).grid(row=row_idx, column=col, sticky="nsew")
            tk.Button(
                rows_frame,
                text="[Wskaż]",
                command=lambda item=candidate: _choose_candidate(item),
                cursor="hand2",
                bg=blend_hex_colors(field_bg, success, 0.18),
                fg=fg,
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=4,
            ).grid(row=row_idx, column=9, sticky="nsew", padx=4, pady=4)

    def _manual_choose() -> None:
        _close_browser()
        self._set_model(normalized_type, initial_dir=self._get_project_start_asset_initial_dir(row_key))

    footer = tk.Frame(body, bg=body_bg)
    footer.pack(fill=tk.X, padx=16, pady=(0, 14))
    tk.Button(
        footer,
        text="Wskaż plik ręcznie",
        command=_manual_choose,
        cursor="hand2",
        bg=field_bg,
        fg=fg,
        relief=tk.FLAT,
        padx=10,
        pady=6,
    ).pack(side=tk.LEFT)
    tk.Button(
        footer,
        text="Zamknij",
        command=_close_browser,
        cursor="hand2",
        bg=field_bg,
        fg=fg,
        relief=tk.FLAT,
        padx=10,
        pady=6,
    ).pack(side=tk.RIGHT)
    try:
        browser.protocol("WM_DELETE_WINDOW", _close_browser)
        browser.transient(parent or self.frame)
        browser.grab_set()
    except Exception:
        pass
    try:
        browser.wait_window()
    except Exception:
        pass
    return True


def _choose_project_start_model(self, model_type: str, parent=None) -> None:
    self._set_project_start_mode("assets", refresh=False)
    normalized_type = "plate" if str(model_type or "").strip().lower() == "plate" else "char"
    try:
        if self._open_project_start_model_candidate_browser(normalized_type, parent=parent or self.frame):
            return
    except Exception as exc:
        logger.debug(f"Nie udało się otworzyć przeglądarki modeli startowych: {exc}")
    row_key = "plate_model" if normalized_type == "plate" else "char_model"
    self._set_model(normalized_type, initial_dir=self._get_project_start_asset_initial_dir(row_key))

def _resolve_project_start_run_images_dir(self, run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return None

    try:
        manifest = annotation_tab._load_annotation_run_manifest(run_dir)
    except Exception:
        manifest = {}

    for raw_path in (
        str(manifest.get("imported_source_input_dir") or "").strip(),
        str(manifest.get("input_dir") or "").strip(),
    ):
        if not raw_path:
            continue
        try:
            candidate = Path(raw_path)
        except Exception:
            continue
        if candidate.exists() and candidate.is_dir() and self._count_images_in_dir(candidate, recursive=True) > 0:
            return candidate

    return None

def _normalize_project_start_asset_full_path(path_value) -> str:
    raw_text = str(path_value or "").strip()
    if not raw_text:
        return ""
    try:
        return str(Path(raw_text).resolve())
    except Exception:
        return raw_text

def _format_project_start_asset_source(self, path_value) -> str:
    normalized_full_path = self._normalize_project_start_asset_full_path(path_value)
    if not normalized_full_path:
        return "Nie wskazano"
    try:
        return self._format_project_relative_path(normalized_full_path)
    except Exception:
        try:
            return self._format_workspace_relative_path(normalized_full_path)
        except Exception:
            return Path(normalized_full_path).name

def _get_project_start_effective_model_state(self, model_type: str) -> dict:
    normalized_type = str(model_type or "").strip().lower()
    if normalized_type not in {"plate", "char"}:
        return {}
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iteration = 1
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip() or None
    except Exception:
        project_name = None
    try:
        model_info = dict(
            CAMPAIGN.get_effective_project_model(
                normalized_type,
                before_iteration=current_iteration,
                project_name=project_name,
            )
            or {}
        )
    except Exception:
        model_info = {}
    model_path = str(model_info.get("path") or "").strip()
    if not model_path:
        return {}
    try:
        path_obj = Path(model_path)
    except Exception:
        path_obj = None
    if path_obj is None or not path_obj.exists() or not path_obj.is_file():
        return {}

    try:
        model_name = path_obj.name
    except Exception:
        model_name = model_path
    source_scope = str(model_info.get("scope") or "").strip()
    try:
        source_iteration = int(model_info.get("iteration", 0) or 0)
    except Exception:
        source_iteration = 0
    if source_scope == "trained_previous_iteration" and source_iteration > 0:
        source_prefix = f"Model wytrenowany w IT{source_iteration}"
    else:
        source_prefix = "Model wskazany w projekcie"
    relative_path = self._format_project_start_asset_source(model_path)
    source_text = f"{source_prefix}: {model_name}"
    if relative_path and relative_path not in {model_name, model_path}:
        source_text = f"{source_text} | {relative_path}"

    metric_parts: list[str] = []
    for metric_key, metric_label in (("best_map50", "mAP50"), ("best_map50_95", "mAP50-95")):
        raw_value = model_info.get(metric_key)
        if raw_value in (None, ""):
            continue
        try:
            metric_parts.append(f"{metric_label}: {float(raw_value) * 100:.1f}%")
        except Exception:
            metric_parts.append(f"{metric_label}: {raw_value}")

    identity = str(model_info.get("identity") or "").strip()
    if not identity:
        try:
            identity = str(self._get_model_identity_label(model_path) or "").strip()
        except Exception:
            identity = ""
    try:
        is_valid, validation_message = self._validate_project_model_selection(normalized_type, path_obj)
    except Exception:
        is_valid, validation_message = True, ""

    validation_parts: list[str] = []
    if identity:
        validation_parts.append(identity)
    validation_parts.extend(metric_parts)
    if is_valid:
        validation_text = "OK | " + " | ".join(validation_parts) if validation_parts else "OK | gotowy model projektu."
        tone = "success"
    else:
        validation_text = str(validation_message or "Model nie przeszedł walidacji.").strip()
        tone = "error"

    detail_lines = [
        f"Źródło: {source_text}",
        f"Plik: {relative_path or model_path}",
    ]
    if identity:
        detail_lines.append(f"Typ modelu: {identity}")
    if metric_parts:
        detail_lines.append("Metryki: " + " | ".join(metric_parts))
    run_id = str(model_info.get("run_id") or "").strip()
    if run_id:
        try:
            run_display = build_run_display_ref({"run_id": run_id}, kind_hint="training").id
        except Exception:
            run_display = run_id
        detail_lines.append(f"Trening: {run_display}")
    created_at = str(model_info.get("finished_at") or model_info.get("created_at") or "").strip()
    if created_at:
        detail_lines.append(f"Data: {created_at}")
    detail_lines.append("Walidacja: " + ("OK" if is_valid else validation_text))

    return {
        "path": model_path,
        "source": source_text,
        "validation": validation_text,
        "tone": tone,
        "identity": identity,
        "metrics": metric_parts,
        "details": detail_lines,
        "scope": source_scope,
        "iteration": str(source_iteration) if source_iteration > 0 else "",
    }

def _collect_project_start_image_names(
    self,
    images_dir: Path | None = None,
    manifest: dict | None = None,
    progress_callback=None,
) -> list[str]:
    image_names: list[str] = []

    for item in list((self.current_ingest_plan or {}).get("selected", []) or []):
        name = str(item.get("name") or "").strip()
        if not name:
            try:
                name = str(Path(str(item.get("source_path") or "").strip()).name or "").strip()
            except Exception:
                name = ""
        if name:
            image_names.append(name)

    if not image_names and isinstance(manifest, dict):
        for item in list(manifest.get("selected_images", []) or []):
            name = str((item or {}).get("name") or "").strip()
            if name:
                image_names.append(name)

    if not image_names and isinstance(images_dir, Path):
        try:
            last_progress = perf_counter()
            for image_path in images_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                    image_names.append(image_path.name)
                    now = perf_counter()
                    if callable(progress_callback) and (
                        len(image_names) == 1
                        or len(image_names) % 250 == 0
                        or now - last_progress >= 0.45
                    ):
                        last_progress = now
                        try:
                            progress_callback(
                                len(image_names),
                                f"Zebrano {len(image_names)} nazw obrazów do kontraktu O.",
                            )
                        except Exception:
                            pass
        except Exception:
            pass

    return image_names

def _build_project_start_image_set_token(
    self,
    images_dir: Path | None = None,
    manifest: dict | None = None,
    progress_callback=None,
) -> tuple[str, int]:
    image_names = self._collect_project_start_image_names(
        images_dir=images_dir,
        manifest=manifest,
        progress_callback=progress_callback,
    )
    token = CAMPAIGN.build_image_name_set_token(image_names, images_dir=images_dir)
    unique_count = len(
        {
            CAMPAIGN._normalize_image_set_name(name)
            for name in image_names
            if CAMPAIGN._normalize_image_set_name(name)
        }
    )
    return token, int(unique_count or 0)

def _build_project_start_plate_xml_image_set_token(self, run_dir: Path | None) -> tuple[str, int]:
    if run_dir is None:
        return "", 0

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return "", 0

    try:
        safe_run_dir = annotation_tab._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    except Exception:
        safe_run_dir = None
    if safe_run_dir is None:
        return "", 0

    try:
        annotations = annotation_tab._parse_cvat_preview_annotations(safe_run_dir / "annotations.xml")
    except Exception:
        annotations = []
    image_names = [str(getattr(ann, "filename", "") or "").strip() for ann in annotations if str(getattr(ann, "filename", "") or "").strip()]
    token = CAMPAIGN.build_image_name_set_token(image_names)
    unique_count = len(
        {
            CAMPAIGN._normalize_image_set_name(name)
            for name in image_names
            if CAMPAIGN._normalize_image_set_name(name)
        }
    )
    return token, int(unique_count or 0)

def _show_project_start_asset_source_context_menu(self, event, row_key: str) -> None:
    rows = getattr(self, "ingest_start_asset_row_widgets", {}) or {}
    row = rows.get(str(row_key or "").strip())
    if not isinstance(row, dict):
        return

    source_full_path = str(row.get("source_full_path") or "").strip()
    if not source_full_path:
        return

    menu = tk.Menu(self.frame, tearoff=0)
    menu.add_command(
        label="Kopiuj pełną ścieżkę",
        command=lambda value=source_full_path: self._copy_project_start_asset_source_path(value),
    )
    menu.add_separator()
    menu.add_command(
        label="Wyczyść wybór",
        command=lambda key=str(row_key or "").strip(): self._clear_project_start_asset(key),
        state=("normal" if self._is_project_start_asset_clearable(row_key) else "disabled"),
    )
    try:
        menu.tk_popup(int(event.x_root), int(event.y_root))
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass

def _copy_project_start_asset_source_path(self, full_path: str) -> None:
    safe_value = str(full_path or "").strip()
    if not safe_value:
        return
    try:
        self.frame.clipboard_clear()
        self.frame.clipboard_append(safe_value)
        self.frame.update_idletasks()
    except Exception:
        return
    try:
        self.app.update_status("Skopiowano pełną ścieżkę zasobu.", "info")
    except Exception:
        pass

def _is_project_start_asset_clearable(self, row_key: str) -> bool:
    row_key = str(row_key or "").strip()
    if row_key == "images":
        plan = getattr(self, "current_ingest_plan", None)
        if isinstance(plan, dict):
            try:
                if int(plan.get("selected_total", 0) or 0) > 0:
                    return True
            except Exception:
                pass
            try:
                if len(list(plan.get("selected", []) or [])) > 0:
                    return True
            except Exception:
                pass

        try:
            master_pool = CAMPAIGN.get_master_pool_dir()
        except Exception:
            master_pool = None
        if master_pool is None:
            return False
        try:
            master_pool_path = Path(master_pool)
            return bool(master_pool_path.exists() and master_pool_path.is_dir())
        except Exception:
            return False
    if row_key == "plate_run":
        try:
            source = CAMPAIGN.get_project_start_plate_source() or {}
        except Exception:
            source = {}
        return bool(
            str(source.get("source_run_path", "") or "").strip()
            or str(source.get("source_xml_path", "") or "").strip()
            or str(source.get("source_input_path", "") or "").strip()
        )
    if row_key == "plate_model":
        try:
            return bool(str(CAMPAIGN.get_global_model("plate") or "").strip())
        except Exception:
            return False
    if row_key == "char_model":
        try:
            return bool(str(CAMPAIGN.get_global_model("char") or "").strip())
        except Exception:
            return False
    return False

def _refresh_project_start_clear_buttons(self) -> None:
    buttons = getattr(self, "btn_ingest_clear_asset", {}) or {}
    if not isinstance(buttons, dict):
        return
    for row_key, button in buttons.items():
        if button is None:
            continue
        try:
            self._set_project_start_badge_button_state(
                button,
                enabled=self._is_project_start_asset_clearable(row_key),
            )
        except Exception:
            pass

def _clear_project_start_asset(self, row_key: str) -> None:
    row_key = str(row_key or "").strip()
    if row_key not in {"images", "plate_run", "plate_model", "char_model"}:
        return
    if not self._is_project_start_asset_clearable(row_key):
        return

    label_map = {
        "images": "katalogu zdjęć",
        "plate_run": "anotacji tablic",
        "plate_model": "modelu tablic",
        "char_model": "modelu znaków",
    }
    label = label_map.get(row_key, "zasobu")

    try:
        confirmed = messagebox.askyesno(
            "Wyczyść wybór",
            f"Czy wyczyścić wybór {label} w E1?\n\n"
            "Operacja usuwa wskazanie z projektu, ale nie kasuje plików z dysku.",
            parent=self.frame,
        )
    except Exception:
        confirmed = True
    if not confirmed:
        return

    try:
        if row_key == "images":
            try:
                CAMPAIGN.clear_master_pool_dir()
            except Exception:
                pass
            try:
                CAMPAIGN.clear_latest_ingest_plan()
            except Exception:
                pass
            self.current_ingest_plan = {}
            self.ingest_plan_items = []
            try:
                self._existing_iteration_ingest_plan_signature = None
            except Exception:
                pass
        elif row_key == "plate_run":
            try:
                CAMPAIGN.clear_project_start_plate_source()
            except Exception:
                pass
            self._set_project_start_asset_scope("plate_run", "", persist=True)
        elif row_key == "plate_model":
            try:
                CAMPAIGN.set_global_model("plate", "")
            except Exception:
                pass
            self._set_project_start_asset_scope("plate_model", "", persist=True)
        elif row_key == "char_model":
            try:
                CAMPAIGN.set_global_model("char", "")
            except Exception:
                pass
            self._set_project_start_asset_scope("char_model", "", persist=True)

        if self._get_iteration_target() == "char" and self._get_step1_char_route_block_reason():
            try:
                CAMPAIGN.clear_iteration_target()
            except Exception:
                pass

        try:
            self._clear_dashboard_perf_cache()
        except Exception:
            pass
        try:
            self._sync_iteration_artifact_registry_from_project_start()
        except Exception:
            pass
        self._refresh_dashboard()
        try:
            self.app.update_status(f"Wyczyszczono wybór {label} w E1.", "info")
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Nie udało się wyczyścić zasobu E1 ({row_key}): {e}")

def _short_project_start_asset_validation(message: str, width: int = 72) -> str:
    text = str(message or "").replace("\n", " ").strip()
    if not text:
        return "Brak"
    return shorten(text, width=max(24, int(width or 72)), placeholder="...")

def _get_step1_ingest_frame_bg(self) -> str:
    palette = getattr(self.app, "palette", {})
    fallback_bg = palette.get("panel", "#252526")
    if self.step1_ingest_host_item is not None:
        extra_frame = self.step1_ingest_host_item.get("extra_actions_frame")
        if extra_frame is not None:
            try:
                return str(extra_frame.cget("bg") or fallback_bg)
            except Exception:
                return fallback_bg
    return fallback_bg

def _get_project_start_asset_scope(self, row_key: str) -> str:
    row_key = str(row_key or "").strip()
    if row_key == "images":
        return "na"
    vars_map = getattr(self, "project_start_asset_scope_vars", {}) or {}
    var = vars_map.get(row_key)
    try:
        value = str(var.get() or "").strip().lower()
    except Exception:
        value = ""
    if value in {"project", "freemode", "na"}:
        return value
    try:
        persisted = str(CAMPAIGN.get_project_start_asset_scope(row_key) or "").strip().lower()
    except Exception:
        persisted = ""
    if persisted in {"project", "freemode", "na"}:
        return persisted
    return "project"

def _set_project_start_asset_scope(self, row_key: str, scope: str, *, persist: bool = True) -> None:
    row_key = str(row_key or "").strip()
    vars_map = getattr(self, "project_start_asset_scope_vars", {}) or {}
    var = vars_map.get(row_key)
    normalized = str(scope or "").strip().lower()
    if row_key == "images":
        normalized = "na"
    elif normalized not in {"project", "freemode", "na"}:
        normalized = "project"
    if var is None:
        try:
            if persist and row_key != "images":
                CAMPAIGN.set_project_start_asset_scope(row_key, normalized)
        except Exception:
            pass
        return
    try:
        var.set(normalized)
    except Exception:
        pass
    try:
        if persist and row_key != "images":
            CAMPAIGN.set_project_start_asset_scope(row_key, normalized)
    except Exception:
        pass

def _restore_project_start_asset_scopes_from_state(self) -> None:
    try:
        imported_plate_source = CAMPAIGN.get_project_start_plate_source() or {}
    except Exception:
        imported_plate_source = {}
    restore_specs = (
        (
            "plate_run",
            str(CAMPAIGN.get_project_start_asset_scope("plate_run") or "").strip().lower(),
            str(imported_plate_source.get("source_xml_path") or "").strip()
            or str(imported_plate_source.get("source_run_path") or "").strip(),
        ),
        (
            "plate_model",
            str(CAMPAIGN.get_project_start_asset_scope("plate_model") or "").strip().lower(),
            str(CAMPAIGN.get_global_model("plate") or "").strip(),
        ),
        (
            "char_model",
            str(CAMPAIGN.get_project_start_asset_scope("char_model") or "").strip().lower(),
            str(CAMPAIGN.get_global_model("char") or "").strip(),
        ),
    )

    for row_key, persisted_scope, fallback_path in restore_specs:
        scope_value = persisted_scope or self._infer_project_start_asset_scope_from_path(row_key, fallback_path)
        self._set_project_start_asset_scope(row_key, scope_value, persist=False)

def _select_project_start_asset_scope(self, row_key: str, scope: str) -> None:
    if self._get_project_start_asset_scope(row_key) == str(scope or "").strip().lower():
        return
    self._set_project_start_asset_scope(row_key, scope)
    try:
        self._refresh_project_start_assets_table_theme(self._get_step1_ingest_frame_bg())
        self._sync_ingest_wraps()
        self.frame.update_idletasks()
    except Exception:
        pass

def _path_is_inside_root(candidate: Path | None, root: Path | None) -> bool:
    if candidate is None or root is None:
        return False
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def _infer_project_start_asset_scope_from_path(self, row_key: str, path_value) -> str:
    row_key = str(row_key or "").strip()
    if row_key == "images":
        return "na"
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return self._get_project_start_asset_scope(row_key)
    try:
        candidate = Path(raw_path)
    except Exception:
        return self._get_project_start_asset_scope(row_key)
    project_root = CAMPAIGN.get_active_project_root_dir()
    if self._path_is_inside_root(candidate, project_root):
        return "project"
    return "freemode"

def _get_project_start_plate_source_info(self) -> dict:
    stored_plate_source = dict(CAMPAIGN.get_project_start_plate_source() or {})

    run_path = str(
        stored_plate_source.get("source_run_path")
        or ""
    ).strip()
    xml_path = str(stored_plate_source.get("source_xml_path") or "").strip()
    images_path = str(stored_plate_source.get("source_input_path") or "").strip()
    source_mode = str(stored_plate_source.get("source_mode") or "").strip().lower()
    if source_mode not in {"approved", "draft"}:
        source_mode = ""

    run_dir = None
    if run_path:
        try:
            candidate = Path(run_path)
            if candidate.exists() and candidate.is_dir():
                run_dir = candidate
        except Exception:
            run_dir = None

    if not xml_path and run_dir is not None:
        try:
            candidate_xml = run_dir / "annotations.xml"
            if candidate_xml.exists() and candidate_xml.is_file():
                xml_path = str(candidate_xml.resolve())
        except Exception:
            pass

    if run_dir is None and xml_path:
        try:
            xml_candidate = Path(xml_path)
            if xml_candidate.exists() and xml_candidate.is_file():
                run_dir = xml_candidate.parent
                run_path = str(run_dir.resolve())
        except Exception:
            run_dir = None

    return {
        "run_path": str(run_path or "").strip(),
        "xml_path": str(xml_path or "").strip(),
        "images_path": str(images_path or "").strip(),
        "source_mode": source_mode,
        "run_dir": run_dir,
    }

def _get_project_start_asset_initial_dir(self, row_key: str) -> Path:
    row_key = str(row_key or "").strip()
    scope = self._get_project_start_asset_scope(row_key)
    project_root = CAMPAIGN.get_active_project_root_dir()
    project_models_dir = CAMPAIGN.get_dir("models")
    project_auto_ann_dir = CAMPAIGN.get_dir("auto_ann")

    if row_key == "plate_run":
        plate_source_info = self._get_project_start_plate_source_info()
        current_path = str(
            plate_source_info.get("xml_path")
            or plate_source_info.get("run_path")
            or ""
        ).strip()
        if current_path:
            try:
                current_candidate = Path(current_path)
                current_dir = current_candidate.parent if current_candidate.is_file() else current_candidate
                if current_dir.exists() and current_dir.is_dir():
                    if scope == "project" and self._path_is_inside_root(current_dir, project_root):
                        return current_dir
                    if scope == "freemode" and not self._path_is_inside_root(current_dir, project_root):
                        return current_dir
            except Exception:
                pass
        if scope == "project" and project_auto_ann_dir is not None:
            candidate = Path(project_auto_ann_dir) / "plates"
            return candidate if candidate.exists() else Path(project_auto_ann_dir)
        return Path(CONFIG.get_auto_annotations_dir("plate"))

    if row_key == "plate_model":
        current_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
        if current_path:
            try:
                current_file = Path(current_path)
                if current_file.exists() and current_file.is_file():
                    current_dir = current_file.parent
                    if scope == "project" and self._path_is_inside_root(current_dir, project_root):
                        return current_dir
                    if scope == "freemode" and not self._path_is_inside_root(current_dir, project_root):
                        return current_dir
            except Exception:
                pass
        if scope == "project" and project_models_dir is not None:
            return Path(project_models_dir)
        for candidate in CONFIG.get_model_search_dirs("plate"):
            if candidate.exists():
                return candidate
        return Path(CONFIG.DIR_6_MODELS)

    if row_key == "char_model":
        current_path = str(CAMPAIGN.get_global_model("char") or "").strip()
        if current_path:
            try:
                current_file = Path(current_path)
                if current_file.exists() and current_file.is_file():
                    current_dir = current_file.parent
                    if scope == "project" and self._path_is_inside_root(current_dir, project_root):
                        return current_dir
                    if scope == "freemode" and not self._path_is_inside_root(current_dir, project_root):
                        return current_dir
            except Exception:
                pass
        if scope == "project" and project_models_dir is not None:
            return Path(project_models_dir)
        for candidate in CONFIG.get_model_search_dirs("char"):
            if candidate.exists():
                return candidate
        return Path(CONFIG.DIR_6_MODELS)

    return Path(CONFIG.WORKSPACE_DIR)

def _sync_iteration_artifact_registry_from_project_start(self, progress_callback=None) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1

    master_pool = CAMPAIGN.get_iteration_image_source_dir(iteration_num) or CAMPAIGN.get_master_pool_dir()
    if master_pool is None:
        master_pool = CAMPAIGN.get_iteration_raw_dir(iteration_num)
    if master_pool is None:
        return

    try:
        master_pool_path = Path(master_pool)
    except Exception:
        return

    try:
        manifest_path = CAMPAIGN.get_ingest_manifest_path(iteration_num)
    except Exception:
        manifest_path = None

    try:
        manifest = CAMPAIGN.load_ingest_manifest(iteration_num) or {}
    except Exception:
        manifest = {}

    image_set_token, image_set_count = self._build_project_start_image_set_token(
        images_dir=master_pool_path,
        manifest=manifest,
        progress_callback=progress_callback,
    )
    package_id = str(
        CAMPAIGN.build_iteration_artifact_package_id(
            master_pool_path,
            iteration_num=iteration_num,
            image_set_token=image_set_token,
        ) or ""
    ).strip()

    stored_plate_source = dict(CAMPAIGN.get_project_start_plate_source() or {})
    existing_bundle = dict(
        CAMPAIGN.get_iteration_artifact_bundle(
            images_dir=master_pool_path,
            iteration_num=iteration_num,
        ) or {}
    )
    existing_char_effective = dict(existing_bundle.get("char_effective_source") or {})

    plate_run_dir_raw = str(
        stored_plate_source.get("source_run_path")
        or ""
    ).strip()
    plate_xml_raw = str(stored_plate_source.get("source_xml_path") or "").strip()
    plate_images_raw = str(stored_plate_source.get("source_input_path") or "").strip()
    plate_source_mode = str(stored_plate_source.get("source_mode") or "").strip().lower()
    if plate_source_mode not in {"approved", "draft"}:
        plate_source_mode = ""

    plate_run_dir = None
    if plate_run_dir_raw:
        try:
            candidate = Path(plate_run_dir_raw)
            if candidate.exists() and candidate.is_dir():
                plate_run_dir = candidate
        except Exception:
            plate_run_dir = None
    if plate_run_dir is not None and not plate_xml_raw:
        try:
            candidate_xml = plate_run_dir / "annotations.xml"
            if candidate_xml.exists():
                plate_xml_raw = str(candidate_xml.resolve())
        except Exception:
            pass
    if plate_run_dir is not None and not plate_images_raw:
        try:
            resolved_images = self._resolve_project_start_run_images_dir(plate_run_dir)
            if resolved_images is not None:
                plate_images_raw = str(resolved_images.resolve())
        except Exception:
            pass

    plate_xml_image_set_token, plate_xml_image_count = self._build_project_start_plate_xml_image_set_token(
        plate_run_dir
    )
    plate_image_set_match = bool(
        image_set_token
        and plate_xml_image_set_token
        and image_set_token == plate_xml_image_set_token
    )

    plate_images_with_plates = 0
    plate_total_plates = 0
    if plate_run_dir is not None:
        try:
            annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
            if annotation_tab is not None and hasattr(annotation_tab, "_get_run_plate_annotation_counts"):
                plate_images_with_plates, plate_total_plates = annotation_tab._get_run_plate_annotation_counts(plate_run_dir)
        except Exception:
            plate_images_with_plates, plate_total_plates = 0, 0

    plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    char_model_path = str(CAMPAIGN.get_global_model("char") or "").strip()
    plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())
    plate_run_ready = bool(plate_run_dir is not None and plate_xml_raw)
    plate_run_approved = bool(plate_run_ready and plate_source_mode == "approved")

    plate_entry_mode = "manual_template"
    if plate_run_ready:
        plate_entry_mode = "ready_run"
    elif plate_model_ready:
        plate_entry_mode = "auto"

    char_images_with_plates = int(existing_char_effective.get("images_with_plates", 0) or 0)
    char_total_plates = int(existing_char_effective.get("total_plates", 0) or 0)
    min_char_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
    char_entry_mode = "blocked"
    if char_total_plates >= min_char_plates:
        char_entry_mode = "ready"
    elif (plate_run_approved and plate_total_plates > 0) or char_total_plates > 0:
        char_entry_mode = "needs_more_tables"

    updates = {
        "image_source": {
            "package_id": package_id,
            "master_pool_dir": str(master_pool_path.resolve()) if master_pool_path.exists() else str(master_pool_path),
            "master_pool_token": CAMPAIGN._build_registry_path_token(master_pool_path),
            "iteration_raw_dir": str(CAMPAIGN.get_iteration_raw_dir(iteration_num) or ""),
            "ingest_manifest_path": str(manifest_path or ""),
            "ingest_manifest_token": CAMPAIGN._build_registry_path_token(manifest_path),
            "selected_count": int(manifest.get("selected_count", 0) or 0),
            "image_set_token": str(image_set_token or "").strip(),
            "image_set_count": int(image_set_count or 0),
            "selection_mode": str(manifest.get("selection_mode") or "").strip(),
        },
        "plate_source": {
            "package_id": package_id,
            "run_dir": str(plate_run_dir.resolve()) if plate_run_dir is not None else "",
            "run_token": CAMPAIGN._build_registry_path_token(plate_run_dir),
            "xml_path": plate_xml_raw,
            "xml_token": CAMPAIGN._build_registry_path_token(plate_xml_raw),
            "xml_image_set_token": str(plate_xml_image_set_token or "").strip(),
            "xml_image_count": int(plate_xml_image_count or 0),
            "images_dir": plate_images_raw,
            "images_token": CAMPAIGN._build_registry_path_token(plate_images_raw),
            "expected_image_set_token": str(image_set_token or "").strip(),
            "expected_image_count": int(image_set_count or 0),
            "image_set_match": bool(plate_image_set_match),
            "scope": self._get_project_start_asset_scope("plate_run"),
            "import_mode": plate_source_mode,
            "input_source": "project_start_import" if plate_run_ready else "",
            "images_with_plates": int(plate_images_with_plates or 0),
            "total_plates": int(plate_total_plates or 0),
        },
        "plate_model": {
            "path": plate_model_path,
            "token": CAMPAIGN._build_registry_path_token(plate_model_path),
            "scope": self._get_project_start_asset_scope("plate_model"),
            "identity": self._get_model_identity_label(plate_model_path) if plate_model_ready else "",
        },
        "char_model": {
            "path": char_model_path,
            "token": CAMPAIGN._build_registry_path_token(char_model_path),
            "scope": self._get_project_start_asset_scope("char_model"),
            "identity": self._get_model_identity_label(char_model_path) if (char_model_path and Path(char_model_path).exists()) else "",
        },
        "route_hints": {
            "plate_entry_mode": plate_entry_mode,
            "char_entry_mode": char_entry_mode,
            "char_ready": bool(char_entry_mode == "ready"),
            "char_has_source": bool(plate_run_approved or char_total_plates > 0),
            "needs_more_tables": bool(char_entry_mode == "needs_more_tables"),
            "images_with_plates": int(char_images_with_plates or (plate_images_with_plates if plate_run_approved else 0) or 0),
            "total_plates": int(char_total_plates or (plate_total_plates if plate_run_approved else 0) or 0),
        },
    }
    try:
        CAMPAIGN.upsert_iteration_artifact_bundle(
            images_dir=master_pool_path,
            iteration_num=iteration_num,
            image_set_token=image_set_token,
            updates=updates,
        )
    except Exception as e:
        logger.debug(f"Nie udało się zsynchronizować rejestru artefaktów E1: {e}")

def _set_project_start_asset_row_state(
    self,
    row_key: str,
    *,
    source_text: str = "",
    source_path: str = "",
    validation_text: str = "",
    tone: str = "muted",
    requirement: str = "",
    counter_text: str = "",
    meta: dict | None = None,
) -> None:
    rows = getattr(self, "ingest_start_asset_row_widgets", {}) or {}
    row = rows.get(str(row_key or "").strip())
    if not isinstance(row, dict):
        return

    try:
        display_text = self._format_tail_text(str(source_text or "").strip() or "Nie wskazano", width=26)
        row["source_var"].set(display_text or "Nie wskazano")
    except Exception:
        pass
    row["source_full_text"] = str(source_text or "").strip() or "Nie wskazano"
    row["source_full_path"] = self._normalize_project_start_asset_full_path(source_path)

    row["tone"] = str(tone or "muted").strip().lower() or "muted"
    requirement_key = str(requirement or "").strip().lower()
    row["requirement"] = requirement_key
    row["counter_text"] = str(counter_text or "").strip()
    row["meta"] = dict(meta or {})
    try:
        base_label = str(row.get("base_label") or "").strip()
        counter = str(row.get("counter_text") or "").strip()
        marker_map = {
            "required": "WYMAGANE",
            "route_required": "WARUNEK TORU",
            "route_required_plate": "WARUNEK: TABLICE",
            "route_required_char": "WARUNEK: ZNAKI",
            "alternative": "UZUPEŁNIAJĄCE",
            "disabled": "NIE DOTYCZY",
        }
        marker = marker_map.get(requirement_key, "")
        counter_suffix = f"  [{counter}]" if counter else ""
        marker_suffix = f"  [{marker}]" if marker else ""
        row["name_lbl"].config(text=f"{base_label}{counter_suffix}{marker_suffix}")
    except Exception:
        pass
    try:
        row["validation_full_text"] = str(validation_text or "").strip() or "Brak"
        row["validation_lbl"].config(text=self._short_project_start_asset_validation(validation_text))
    except Exception:
        pass

def _refresh_project_start_assets_table_theme(self, frame_bg: str | None = None) -> None:
    rows = getattr(self, "ingest_start_asset_row_widgets", {}) or {}
    if not isinstance(rows, dict) or not rows:
        return

    palette = getattr(self.app, "palette", {})
    base_bg = str(frame_bg or self._get_step1_ingest_frame_bg())
    panel_bg = palette.get("field", "#1a1a1a")
    muted = palette.get("muted", "#c7c7c7")
    fg = palette.get("fg", "#f3f3f3")
    accent = palette.get("accent", "#4fc1ff")
    success = palette.get("success", "#27ae60")
    grid_border = blend_hex_colors(
        success,
        base_bg,
        0.80,
    )
    header_bg = blend_hex_colors(success, base_bg, 0.16)
    header_fg = self._pick_readable_text_color(header_bg)
    tone_map = {
        "muted": muted,
        "info": accent,
        "success": success,
        "warning": palette.get("warning", "#d35400"),
        "error": palette.get("error", "#c0392b"),
    }

    shell = getattr(self, "ingest_start_assets_table_shell", None)
    table = getattr(self, "ingest_start_assets_table", None)
    title_lbl = getattr(self, "ingest_start_assets_title_lbl", None)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    if shell is not None:
        try:
            shell.config(
                bg=grid_border,
                highlightthickness=1,
                highlightbackground=grid_border,
                highlightcolor=grid_border,
            )
        except Exception:
            pass
    if table is not None:
        try:
            table.config(bg=grid_border)
        except Exception:
            pass
    if title_lbl is not None:
        try:
            title_lbl.config(bg=base_bg, fg=fg)
        except Exception:
            pass
    try:
        for header_lbl in list(getattr(self, "ingest_start_assets_header_labels", []) or []):
            if header_lbl is None:
                continue
            header_lbl.config(
                bg=header_bg,
                fg=header_fg,
                highlightthickness=1,
                highlightbackground=grid_border,
                highlightcolor=grid_border,
            )
    except Exception:
        pass

    for row_index, row_key in enumerate(("images", "plate_model", "char_model", "plate_run", "char_run"), start=1):
        row = rows.get(row_key)
        if not isinstance(row, dict):
            continue
        row_bg = blend_hex_colors(base_bg, panel_bg, 0.10 if (row_index % 2 == 1) else 0.18)
        requirement = str(row.get("requirement", "") or "").strip().lower()
        requirement_color = {
            "required": palette.get("warning", "#f39c12"),
            "route_required": palette.get("warning", "#f39c12"),
            "route_required_plate": palette.get("warning", "#f39c12"),
            "route_required_char": palette.get("warning", "#f39c12"),
            "alternative": accent,
            "disabled": palette.get("muted_dim", "#777777"),
        }.get(requirement, "")
        if requirement in {"required", "route_required", "route_required_plate", "route_required_char", "alternative"} and requirement_color:
            row_bg = blend_hex_colors(row_bg, requirement_color, 0.08)
        row_fg = self._pick_readable_text_color(row_bg)
        validation_fg = tone_map.get(str(row.get("tone", "muted") or "muted").strip().lower(), muted)
        row_border = (
            blend_hex_colors(requirement_color, row_bg, 0.28)
            if requirement in {"required", "route_required", "route_required_plate", "route_required_char", "alternative"} and requirement_color
            else grid_border
        )
        name_fg = (
            requirement_color
            if requirement in {"required", "route_required", "route_required_plate", "route_required_char", "alternative", "disabled"} and requirement_color
            else row_fg
        )

        for widget_name, fg_value in (
            ("name_lbl", name_fg),
            ("source_lbl", accent),
            ("validation_lbl", validation_fg),
        ):
            widget = row.get(widget_name)
            if widget is None:
                continue
            try:
                widget.config(
                    bg=row_bg,
                    fg=fg_value,
                    highlightthickness=1,
                    highlightbackground=row_border,
                    highlightcolor=row_border,
                )
            except Exception:
                pass

        source_scope_host = row.get("source_scope_host")
        if source_scope_host is not None:
            try:
                source_scope_host.config(
                    bg=row_bg,
                    highlightthickness=1,
                    highlightbackground=row_border,
                    highlightcolor=row_border,
                )
            except Exception:
                pass
        current_scope = self._get_project_start_asset_scope(row_key)
        managed_scope_widgets = set()
        for badge in row.get("scope_badges", []) or []:
            badge_value = str(badge.get("value") or "").strip().lower()
            shell_widget = badge.get("shell")
            lamp_widget = badge.get("lamp")
            lamp_id = badge.get("lamp_id")
            label_widget = badge.get("label")
            active = bool(badge_value == current_scope)
            shell_bg = (
                blend_hex_colors(row_bg, palette.get("success", "#27ae60"), 0.20)
                if active
                else blend_hex_colors(row_bg, palette.get("field", "#1a1a1a"), 0.12)
            )
            lamp_fill = palette.get("success", "#27ae60") if active else palette.get("muted_dim", "#4a4a4a")
            text_fg = fg if active else muted
            border_color = (
                blend_hex_colors(palette.get("success", "#27ae60"), row_bg, 0.18)
                if active
                else blend_hex_colors(row_bg, palette.get("field", "#1a1a1a"), 0.26)
            )
            for managed_widget in (shell_widget, lamp_widget, label_widget):
                if managed_widget is not None:
                    managed_scope_widgets.add(managed_widget)
            if shell_widget is not None:
                try:
                    shell_widget.config(
                        bg=shell_bg,
                        highlightthickness=1,
                        highlightbackground=border_color,
                        highlightcolor=border_color,
                    )
                except Exception:
                    pass
            if lamp_widget is not None:
                try:
                    lamp_widget.config(bg=shell_bg)
                    if lamp_id is not None:
                        lamp_widget.itemconfig(lamp_id, fill=lamp_fill)
                except Exception:
                    pass
            if label_widget is not None:
                try:
                    label_widget.config(bg=shell_bg, fg=text_fg)
                except Exception:
                    pass
        for widget in row.get("scope_widgets", []) or []:
            if widget in managed_scope_widgets:
                continue
            try:
                widget_class = str(widget.winfo_class() or "").lower()
            except Exception:
                widget_class = ""
            try:
                if "label" in widget_class:
                    widget.config(bg=row_bg, fg=muted)
                elif "canvas" in widget_class:
                    widget.config(bg=row_bg)
                else:
                    widget.config(bg=row_bg)
            except Exception:
                pass

        action_host = row.get("action_host")
        if action_host is not None:
            try:
                action_host.config(
                    bg=row_bg,
                    highlightthickness=1,
                    highlightbackground=row_border,
                    highlightcolor=row_border,
                )
            except Exception:
                pass
        for slot_name in ("action_primary_slot", "action_clear_slot"):
            slot = row.get(slot_name)
            if slot is not None:
                try:
                    slot.config(bg=row_bg)
                except Exception:
                    pass
        details_host = row.get("details_host")
        if details_host is not None:
            try:
                details_host.config(
                    bg=row_bg,
                    highlightthickness=1,
                    highlightbackground=row_border,
                    highlightcolor=row_border,
                )
            except Exception:
                pass
        for button in (
            getattr(self, "btn_ingest_start_fresh", None) if row_key == "images" else None,
            getattr(self, "btn_ingest_import_plate_run", None) if row_key == "plate_run" else None,
            getattr(self, "btn_ingest_pick_plate_model", None) if row_key == "plate_model" else None,
            getattr(self, "btn_ingest_pick_char_model", None) if row_key == "char_model" else None,
            dict(getattr(self, "btn_ingest_clear_asset", {}) or {}).get(row_key),
        ):
            if button is not None:
                self._style_project_start_badge_button(button, host_bg=row_bg)
        more_button = dict(getattr(self, "btn_ingest_asset_more", {}) or {}).get(row_key)
        if more_button is not None:
            self._style_project_start_badge_button(more_button, host_bg=row_bg)

def _render_ingest_balance_chart(
    self,
    canvas: tk.Canvas | None,
    summary_label: tk.Label | None,
    package_balance: Counter | dict | None,
    package_images: int = 0,
    raw_package_images: int = 0,
) -> None:
    if canvas is None:
        return

    palette = getattr(self.app, "palette", {})
    field_bg = palette.get("field", "#1a1a1a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#b8b8b8")
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent_color = "#f29f05"
    top_color = "#ffbf47"

    balance = Counter(package_balance or {})
    total = sum(int(v) for v in balance.values())

    try:
        canvas.config(
            bg=field_bg,
            highlightbackground=panel_border,
            highlightcolor=panel_border,
        )
    except Exception:
        pass
    canvas.delete("all")

    if total <= 0:
        width = max(int(canvas.winfo_width() or 280), 280)
        canvas.create_text(
            width // 2,
            95,
            text="Brak danych E1 do pokazania.",
            fill=muted,
            justify=tk.CENTER,
            font=("Segoe UI", 9),
        )
        if summary_label is not None:
            try:
                summary_label.config(
                    text=(
                        "Histogram pokazuje liczbę wystąpień każdego znaku w nazwach tablic zdjęć "
                        "należących do aktualnie wybranej puli E1."
                    )
                )
            except Exception:
                pass
        return

    width = max(int(canvas.winfo_width() or 560), 560)
    height = max(int(canvas.winfo_height() or 220), 220)
    left_margin = 26
    right_margin = 16
    top_margin = 18
    bottom_margin = 42
    plot_height = max(height - top_margin - bottom_margin, 80)
    plot_width = max(width - left_margin - right_margin, 180)
    max_value = max([int(balance.get(ch, 0)) for ch in CHAR_ALPHABET] + [1])
    cell_width = plot_width / float(len(CHAR_ALPHABET))
    top_chars = {
        ch for ch, value in sorted(
            ((ch, int(balance.get(ch, 0))) for ch in CHAR_ALPHABET),
            key=lambda entry: (-entry[1], entry[0])
        )[:4]
        if value > 0
    }

    base_y = top_margin + plot_height
    canvas.create_line(left_margin, base_y, width - right_margin, base_y, fill=panel_border)
    for grid_ratio, label in ((1.0, str(max_value)), (0.5, str(max(1, round(max_value / 2)))), (0.0, "0")):
        y = top_margin + int((1.0 - grid_ratio) * plot_height)
        canvas.create_line(left_margin, y, width - right_margin, y, fill=panel_border)
        canvas.create_text(4, y, text=label, anchor="w", fill=muted, font=("Segoe UI", 7))

    for idx, ch in enumerate(CHAR_ALPHABET):
        count = int(balance.get(ch, 0))
        x0 = left_margin + (idx * cell_width) + 1
        x1 = left_margin + ((idx + 1) * cell_width) - 1
        if x1 <= x0:
            x1 = x0 + 2
        bar_height = int(plot_height * (count / max_value)) if max_value > 0 else 0
        y0 = base_y - bar_height
        fill = top_color if ch in top_chars else accent_color
        if count > 0:
            canvas.create_rectangle(x0, y0, x1, base_y, fill=fill, width=0)
        else:
            canvas.create_line(x0, base_y - 1, x1, base_y - 1, fill=panel_border)
        canvas.create_text((x0 + x1) / 2, base_y + 10, text=ch, fill=fg, font=("Consolas", 7))
        if count > 0 and ch in top_chars:
            canvas.create_text((x0 + x1) / 2, max(y0 - 8, 8), text=str(count), fill=muted, font=("Segoe UI", 7))

    dominant = [
        (ch, int(balance.get(ch, 0)))
        for ch in CHAR_ALPHABET
        if int(balance.get(ch, 0)) > 0
    ]
    dominant.sort(key=lambda entry: (-entry[1], entry[0]))
    dominant_text = ", ".join(
        f"{ch}:{count} ({(count / total) * 100:.1f}%)"
        for ch, count in dominant[:6]
    ) or "brak"
    missing = [ch for ch in CHAR_ALPHABET if int(balance.get(ch, 0)) == 0]
    missing_text = ", ".join(missing[:10]) if missing else "brak"
    summary = (
        "Wysokość słupka oznacza liczbę wystąpień danego znaku w nazwach tablic zdjęć należących do "
        "aktualnie wybranej puli E1.\n"
        f"Wybrana pula zawiera teraz {int(raw_package_images or package_images)} zdjęć"
        + (
            f", z czego {int(package_images or 0)} weszło do planu E1"
            if int(raw_package_images or package_images) != int(package_images or 0)
            else ""
        )
        + f", oraz {total} znaków GT w planie. "
        + f"Najczęstsze znaki: {dominant_text}.\n"
        + f"Brakujące znaki w tej puli: {missing_text}."
    )
    if summary_label is not None:
        try:
            summary_label.config(text=summary)
        except Exception:
            pass

def _ensure_project_start_analysis_plan(self) -> bool:
    plan = getattr(self, "current_ingest_plan", {}) or {}
    if isinstance(plan, dict) and int(plan.get("selected_total", 0) or 0) > 0 and plan.get("selected"):
        return True

    ensure_plan = getattr(self, "_ensure_current_ingest_plan_from_master_pool", None)
    if not callable(ensure_plan):
        return False
    try:
        return bool(ensure_plan())
    except Exception as exc:
        logger.debug(f"Nie udalo sie odbudowac planu analizy E1 z zapisanej sciezki zdjec: {exc}")
        return False

def _show_project_start_images_analysis_dialog(self, parent=None) -> None:
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#b8b8b8")
    panel_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    accent = palette.get("accent", "#4fc1ff")
    perf_start = perf_counter()
    perf_last = perf_start

    def _log_hist_perf(stage: str) -> None:
        nonlocal perf_last
        now = perf_counter()
        elapsed_ms = (now - perf_last) * 1000.0
        total_ms = (now - perf_start) * 1000.0
        perf_last = now
        if elapsed_ms >= 120.0 or total_ms >= 250.0:
            logger.info(
                "[T02 HIST PERF] stage=%s elapsed=%.1fms total=%.1fms",
                stage,
                elapsed_ms,
                total_ms,
            )

    manifest = self._load_ingest_manifest_cached()
    manifest_selected_images = list(manifest.get("selected_images", []) or []) if isinstance(manifest, dict) else []
    manifest_histogram = Counter()
    if isinstance(manifest, dict):
        try:
            manifest_histogram.update(
                {
                    str(ch): int(value)
                    for ch, value in dict(manifest.get("char_histogram") or {}).items()
                    if int(value or 0) > 0
                }
            )
        except Exception:
            manifest_histogram = Counter()
        if not manifest_histogram:
            for item in manifest_selected_images:
                if not isinstance(item, dict):
                    continue
                try:
                    manifest_histogram.update(
                        {
                            str(ch): int(value)
                            for ch, value in dict(item.get("char_histogram") or {}).items()
                            if int(value or 0) > 0
                        }
                    )
                except Exception:
                    continue
        if not manifest_histogram and manifest_selected_images:
            planner = CampaignIngestPlanner()
            for item in manifest_selected_images:
                if not isinstance(item, dict):
                    continue
                true_texts = list(item.get("ground_truth_texts") or [])
                if not true_texts:
                    true_texts = planner.extract_true_texts_from_filename(str(item.get("name") or ""))
                try:
                    manifest_histogram.update(planner.build_char_histogram(true_texts))
                except Exception:
                    continue
    _log_hist_perf("manifest")

    active_project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
    try:
        active_iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        active_iteration_num = 1
    try:
        manifest_signature = (
            "manifest_analysis",
            active_project_name,
            active_iteration_num,
            self._build_cache_token_for_path(CAMPAIGN.get_ingest_manifest_path()),
        )
    except Exception:
        manifest_signature = None

    current_plan = self.current_ingest_plan if isinstance(getattr(self, "current_ingest_plan", {}), dict) else {}
    try:
        current_plan_iteration = int(current_plan.get("iteration", 0) or 0)
    except Exception:
        current_plan_iteration = 0
    current_plan_project = str(current_plan.get("project", "") or "").strip()
    current_plan_matches = bool(
        current_plan
        and (not current_plan_project or current_plan_project == active_project_name)
        and (current_plan_iteration <= 0 or current_plan_iteration == active_iteration_num)
    )
    has_manifest_display_plan = bool(
        manifest_signature
        and current_plan_matches
        and getattr(self, "_existing_iteration_ingest_plan_signature", None) == manifest_signature
        and current_plan.get("selected_balance")
    )

    current_balance = Counter((self.last_ingest_snapshot or {}).get("char_balance", {}) or {})
    if manifest_selected_images and manifest_histogram and not has_manifest_display_plan:
        manifest_plan = self._build_ingest_plan_from_manifest_for_display(
            manifest,
            current_balance=current_balance,
        )
        if manifest_plan:
            self.current_ingest_plan = manifest_plan
            current_plan = manifest_plan
            current_plan_matches = True
            try:
                self._existing_iteration_ingest_plan_signature = manifest_signature
            except Exception:
                pass
    _log_hist_perf("plan")

    try:
        manifest_selected_count = int(manifest.get("selected_count", 0) or len(manifest_selected_images) or 0) if isinstance(manifest, dict) else 0
    except Exception:
        manifest_selected_count = len(manifest_selected_images)
    plan_selected_count = int(current_plan.get("selected_total", 0) or 0) if current_plan_matches else 0
    latest_summary = {}
    try:
        latest_summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
    except Exception:
        latest_summary = {}
    try:
        latest_summary_iteration = int(latest_summary.get("iteration", 0) or 0)
    except Exception:
        latest_summary_iteration = 0
    latest_summary_project = str(latest_summary.get("project", "") or "").strip()
    latest_summary_matches = bool(
        latest_summary
        and (not latest_summary_project or latest_summary_project == active_project_name)
        and (latest_summary_iteration <= 0 or latest_summary_iteration == active_iteration_num)
    )
    latest_summary_balance = Counter()
    if latest_summary_matches:
        for raw_balance in (
            latest_summary.get("selected_balance"),
            latest_summary.get("char_histogram"),
        ):
            if not isinstance(raw_balance, dict) or not raw_balance:
                continue
            try:
                latest_summary_balance.update(
                    {
                        str(ch): int(value)
                        for ch, value in dict(raw_balance).items()
                        if int(value or 0) > 0
                    }
                )
            except Exception:
                latest_summary_balance = Counter()
            if latest_summary_balance:
                break
    try:
        master_pool = CAMPAIGN.get_master_pool_dir()
    except Exception:
        master_pool = None
    master_pool_images = int(
        (latest_summary or {}).get("raw_total", 0)
        or (latest_summary or {}).get("selected_total", 0)
        or manifest_selected_count
        or plan_selected_count
        or 0
    )
    # Read-only details must never rescan the image directory; histogram is
    # produced when the image resource is added and read here from manifest/cache.
    context_builder = getattr(self, "_get_step1_manifest_context_lightweight", None)
    if callable(context_builder):
        step1_context = context_builder(
            latest_summary,
            iter_image_count=manifest_selected_count,
            plan_count=plan_selected_count,
        )
    else:
        step1_context = self._get_step1_manifest_context(manifest)
    _log_hist_perf("context")
    source_total = int(step1_context.get("source_total", 0) or 0)
    project_overlap = int(step1_context.get("project_overlap_filenames", 0) or 0)
    new_to_project = int(step1_context.get("new_to_project_count", 0) or 0)
    approved_overlap = int(step1_context.get("skipped_duplicate_approved", 0) or 0)
    selected_total = int(
        step1_context.get("current_iteration_package_count", 0)
        or (latest_summary.get("selected_total", 0) if latest_summary_matches else 0)
        or plan_selected_count
        or manifest_selected_count
        or 0
    )
    source_diverged = bool(
        master_pool_images > 0
        and selected_total > 0
        and master_pool_images != selected_total
    )
    selected_balance = Counter()
    if manifest_histogram:
        selected_balance = Counter(manifest_histogram)
    elif latest_summary_balance:
        selected_balance = Counter(latest_summary_balance)
    elif current_plan_matches:
        selected_balance = Counter(current_plan.get("selected_balance", {}) or {})

    parent_widget = parent or getattr(self, "frame", None) or getattr(self.app, "root", None)
    try:
        parent_window = parent_widget.winfo_toplevel()
    except Exception:
        parent_window = getattr(self.app, "root", None)

    dialog = tk.Toplevel(parent_window or parent_widget)
    self.app.style_dialog_window(
        dialog,
        title="Analiza obrazów iteracji",
        geometry="820x660",
        parent=parent_window or parent_widget,
    )

    build_surface = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(build_surface):
        body = build_surface(dialog, tone="info")
    else:
        body = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        body.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        body,
        text="Analiza obrazów iteracji",
        font=("Segoe UI", 11, "bold"),
        fg=fg,
        bg=panel_bg,
        anchor="w",
    ).pack(fill=tk.X, padx=14, pady=(14, 4))

    tk.Label(
        body,
        text=(
            "Tutaj sprawdzisz bieżącą pulę E1: źródło obrazów, liczebność iteracji i histogram znaków, "
            "który pokazuje rozkład tablic w wybranym katalogu zdjęć."
        ),
        justify=tk.LEFT,
        anchor="w",
        wraplength=760,
        fg=muted,
        bg=panel_bg,
    ).pack(fill=tk.X, padx=14, pady=(0, 10))

    summary_shell = tk.Frame(body, bg=panel_bg)
    summary_shell.pack(fill=tk.X, padx=14, pady=(0, 10))
    summary_shell.grid_columnconfigure(1, weight=1)

    summary_rows = [
        ("Katalog źródłowy projektu", self._format_project_start_asset_source(master_pool)),
        ("Obrazy w katalogu iteracji", str(selected_total)),
        ("Faktycznie nowe dla projektu", str(new_to_project)),
        ("Już wcześniej w projekcie", str(project_overlap)),
    ]
    if source_diverged:
        summary_rows.insert(1, ("Obrazy w obecnym katalogu źródłowym", str(master_pool_images)))
    if approved_overlap > 0:
        summary_rows.append(("W tym już w pudełku zatwierdzonych", str(approved_overlap)))
    if source_total > 0 and source_total != selected_total:
        summary_rows.append(("Historyczny zapis źródła zdjęć", str(source_total)))

    for row_idx, (label_text, value_text) in enumerate(summary_rows):
        tk.Label(
            summary_shell,
            text=f"{label_text}:",
            font=("Segoe UI", 9, "bold"),
            fg=fg,
            bg=panel_bg,
            anchor="w",
        ).grid(row=row_idx, column=0, sticky="w", padx=(0, 10), pady=1)
        tk.Label(
            summary_shell,
            text=value_text,
            fg=accent,
            bg=panel_bg,
            justify=tk.LEFT,
            anchor="w",
            wraplength=560,
        ).grid(row=row_idx, column=1, sticky="ew", pady=1)

    tk.Label(
        body,
        text="Histogram znaków w aktualnym planie E1",
        font=("Segoe UI", 9, "bold"),
        fg=fg,
        bg=panel_bg,
        anchor="w",
    ).pack(fill=tk.X, padx=14, pady=(2, 4))

    chart_canvas = tk.Canvas(
        body,
        height=240,
        bg=palette.get("field", "#1a1a1a"),
        bd=0,
        highlightthickness=1,
        highlightbackground=panel_border,
        highlightcolor=panel_border,
    )
    chart_canvas.pack(fill=tk.X, padx=14)

    chart_summary_lbl = tk.Label(
        body,
        text="",
        justify=tk.LEFT,
        anchor="w",
        wraplength=760,
        fg=muted,
        bg=panel_bg,
    )
    chart_summary_lbl.pack(fill=tk.X, padx=14, pady=(8, 0))

    top_chars_text = self._format_histogram_compact(selected_balance, limit=8) if selected_balance else "brak danych"
    project_chars_text = self._format_histogram_compact(current_balance, limit=8) if current_balance else "brak danych"
    tk.Label(
        body,
        text=(
            f"Najczęstsze znaki w tej iteracji: {top_chars_text}\n"
            f"Przegląd znaków w dotychczas zatwierdzonym zbiorze projektu: {project_chars_text}"
        ),
        justify=tk.LEFT,
        anchor="w",
        wraplength=760,
        fg=muted,
        bg=panel_bg,
    ).pack(fill=tk.X, padx=14, pady=(8, 0))

    if source_diverged:
        tk.Label(
            body,
            text=(
                f"Uwaga: katalog zdjęć iteracji ma {selected_total} zdjęć, ale obecny katalog źródłowy projektu ma teraz {master_pool_images}. "
                "Dalsza praca tej iteracji opiera się na katalogu zdjęć iteracji."
            ),
            justify=tk.LEFT,
            anchor="w",
            wraplength=760,
            fg=muted,
            bg=panel_bg,
        ).pack(fill=tk.X, padx=14, pady=(8, 0))

    tk.Label(
        body,
        text=(
            "Obrazy iteracji są wymagane. Pozostałe zasoby E1 są opcjonalne: anotacje tablic "
            "mogą przenieść do projektu wcześniej wykonaną ręczną pracę, a modele wskazują punkt startowy, "
            "który chcesz dalej wykorzystywać i dotrenowywać w kolejnych iteracjach."
        ),
        justify=tk.LEFT,
        anchor="w",
        wraplength=760,
        fg=muted,
        bg=panel_bg,
    ).pack(fill=tk.X, padx=14, pady=(10, 0))

    button_row = tk.Frame(body, bg=panel_bg)
    button_row.pack(fill=tk.X, padx=14, pady=(14, 14))
    ttk.Button(button_row, text="Zamknij", command=dialog.destroy, width=14).pack(side=tk.RIGHT)

    def _refresh_modal_chart(_event=None):
        if not dialog.winfo_exists():
            return
        self._render_ingest_balance_chart(
            chart_canvas,
            chart_summary_lbl,
            selected_balance,
            package_images=selected_total,
            raw_package_images=max(source_total, selected_total, master_pool_images),
        )

    chart_canvas.bind("<Configure>", _refresh_modal_chart)
    dialog.bind("<Escape>", lambda _e: dialog.destroy())
    _refresh_modal_chart()
    _log_hist_perf("render")

    fit_dialog = getattr(self.app, "_fit_dialog_to_content", None)
    if callable(fit_dialog):
        fit_dialog(dialog, parent=parent_window or parent_widget, min_width=820, min_height=660)
    _log_hist_perf("fit")
    try:
        dialog.transient(parent_window or parent_widget)
    except Exception:
        pass
    try:
        dialog.lift()
        dialog.grab_set()
        dialog.focus_force()
    except Exception:
        pass
    def _raise_analysis_dialog() -> None:
        try:
            if dialog.winfo_exists():
                dialog.lift()
                dialog.focus_force()
        except Exception:
            pass

    try:
        dialog.after_idle(_raise_analysis_dialog)
    except Exception:
        pass

def _get_project_start_effective_images_source(self) -> dict:
    def _same_path(left, right) -> bool:
        if left is None or right is None:
            return False
        try:
            return Path(left).resolve() == Path(right).resolve()
        except Exception:
            return str(left or "").strip().lower() == str(right or "").strip().lower()

    master_pool = CAMPAIGN.get_master_pool_dir()
    master_pool_exists = bool(master_pool and master_pool.exists() and master_pool.is_dir())
    master_pool_count = 0
    try:
        plan = getattr(self, "current_ingest_plan", None)
        plan_source = str(
            (plan or {}).get("master_pool_dir")
            or (plan or {}).get("source_dir")
            or ""
        ).strip() if isinstance(plan, dict) else ""
        if (
            isinstance(plan, dict)
            and int(plan.get("selected_total", 0) or 0) > 0
            and (not plan_source or _same_path(plan_source, master_pool))
        ):
            master_pool_count = int(plan.get("raw_total", 0) or plan.get("selected_total", 0) or 0)
    except Exception:
        master_pool_count = 0
    if master_pool_count <= 0:
        try:
            summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
        except Exception:
            summary = {}
        if summary:
            try:
                summary_matches = (
                    int(summary.get("iteration", 0) or 0) == int(CAMPAIGN.get_current_iteration_num() or 1)
                    and str(summary.get("project", "") or "").strip() == str(CAMPAIGN.get_active_project_name() or "").strip()
                )
            except Exception:
                summary_matches = False
            summary_source = str(
                summary.get("master_pool_dir")
                or summary.get("source_dir")
                or ""
            ).strip()
            summary_source_matches = (
                not master_pool_exists
                or bool(summary_source and _same_path(summary_source, master_pool))
            )
            if summary_matches and summary_source_matches:
                master_pool_count = int(summary.get("raw_total", 0) or summary.get("selected_total", 0) or 0)

    try:
        iteration_images_dir = CAMPAIGN.get_iteration_raw_dir()
    except Exception:
        iteration_images_dir = None
    try:
        logical_iteration_dir = CAMPAIGN.get_iteration_image_source_dir()
    except Exception:
        logical_iteration_dir = iteration_images_dir
    try:
        visible_iteration_dir = Path(logical_iteration_dir) if logical_iteration_dir is not None else None
    except Exception:
        visible_iteration_dir = Path(iteration_images_dir) if iteration_images_dir is not None else None
    iteration_images_exists = bool(
        visible_iteration_dir
        and visible_iteration_dir.exists()
        and visible_iteration_dir.is_dir()
    )
    try:
        iteration_images_count = int(CAMPAIGN.get_iteration_image_count() or 0)
    except Exception:
        iteration_images_count = (
            self._count_images_in_dir(visible_iteration_dir, recursive=True)
            if iteration_images_exists
            else 0
        )

    try:
        step1_approved = str(CAMPAIGN.get_step1_status() or "").strip().lower() == "approved"
    except Exception:
        step1_approved = False
    prefer_master_pool = bool(master_pool_exists and not step1_approved)

    effective_images_dir = (
        master_pool
        if prefer_master_pool
        else
        Path(logical_iteration_dir)
        if iteration_images_count > 0 and logical_iteration_dir is not None
        else master_pool
    )
    effective_images_count = (
        master_pool_count
        if prefer_master_pool
        else iteration_images_count
        if iteration_images_count > 0
        else master_pool_count
    )
    effective_images_exists = bool(
        effective_images_dir
        and Path(effective_images_dir).exists()
        and Path(effective_images_dir).is_dir()
    )

    return {
        "master_dir": master_pool,
        "master_count": int(master_pool_count or 0),
        "master_exists": bool(master_pool_exists),
        "iteration_dir": visible_iteration_dir,
        "physical_iteration_dir": Path(iteration_images_dir) if iteration_images_dir is not None else None,
        "iteration_count": int(iteration_images_count or 0),
        "iteration_exists": bool(iteration_images_exists),
        "effective_dir": effective_images_dir,
        "effective_count": int(effective_images_count or 0),
        "effective_exists": bool(effective_images_exists),
    }

def _scan_project_start_normalized_image_names(self, images_dir: Path | None) -> set[str]:
    if images_dir is None:
        return set()

    try:
        safe_dir = Path(images_dir)
    except Exception:
        return set()

    image_names: set[str] = set()
    try:
        if not safe_dir.exists() or not safe_dir.is_dir():
            return set()
    except Exception:
        return set()

    try:
        for image_path in safe_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                continue
            normalized = CAMPAIGN._normalize_image_set_name(image_path.name)
            if normalized:
                image_names.add(normalized)
    except Exception:
        return set()
    return image_names

def _get_project_start_approved_normalized_image_names(self) -> set[str]:
    approved_names: set[str] = set()

    try:
        registry = CAMPAIGN.get_used_image_registry() or {}
        for name in list(registry.get("filenames") or []):
            normalized = CAMPAIGN._normalize_image_set_name(name)
            if normalized:
                approved_names.add(normalized)
    except Exception:
        pass

    try:
        for entry in list(CAMPAIGN.list_plate_approved_entries() or []):
            if not isinstance(entry, dict):
                continue
            for raw_name in (
                entry.get("image_name", ""),
                Path(str(entry.get("source_image_path", "") or "")).name if str(entry.get("source_image_path", "") or "").strip() else "",
            ):
                normalized = CAMPAIGN._normalize_image_set_name(raw_name)
                if normalized:
                    approved_names.add(normalized)
    except Exception:
        pass

    return approved_names

def _get_project_start_adoptable_normalized_image_names(self, images_dir: Path | None) -> set[str]:
    all_names = self._scan_project_start_normalized_image_names(images_dir)
    if not all_names:
        all_names = self._get_project_start_normalized_image_names(images_dir)
    approved_names = self._get_project_start_approved_normalized_image_names()
    return set(all_names) - set(approved_names)

def _show_project_start_asset_details(self, row_key: str, parent=None) -> None:
    row_key = str(row_key or "").strip()
    if not row_key:
        return

    if row_key == "images":
        self._show_project_start_images_analysis_dialog(parent=parent)
        return

    image_source = self._get_project_start_effective_images_source()
    master_pool = CAMPAIGN.get_master_pool_dir()
    master_pool_exists = bool(master_pool and master_pool.exists() and master_pool.is_dir())
    master_pool_images = int(image_source.get("master_count", 0) or 0)
    if master_pool_images <= 0 and master_pool_exists:
        try:
            summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
            master_pool_images = int(summary.get("raw_total", 0) or summary.get("selected_total", 0) or 0)
        except Exception:
            master_pool_images = 0
    plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    char_model_path = str(CAMPAIGN.get_global_model("char") or "").strip()

    title = "Szczegóły zasobu"
    body_lines: list[str] = []

    if row_key == "plate_run":
        title = "Anotacje tablic"
        plate_source_info = self._get_project_start_plate_source_info()
        plate_run_path = str(plate_source_info.get("run_path") or "").strip()
        plate_xml_path = str(plate_source_info.get("xml_path") or "").strip()
        plate_source_mode = str(plate_source_info.get("source_mode") or "").strip().lower()
        body_lines.append(f"Źródło: {self._format_project_start_asset_source(plate_xml_path or plate_run_path)}")
        if plate_source_mode == "draft":
            body_lines.append("Status: AT do ręcznej kontroli w Z2. Nie spełnia jeszcze warunku T03.")
        elif plate_source_mode == "approved":
            body_lines.append("Status: AT zatwierdzone po kontroli. Może spełniać warunek T03, jeśli liczba tablic osiąga minimum.")
        if plate_run_path:
            try:
                plate_run_dir = Path(plate_run_path)
            except Exception:
                plate_run_dir = None
            effective_images_dir = image_source.get("effective_dir")
            effective_images_count = int(image_source.get("effective_count", 0) or 0)
            compatibility = (
                self._check_project_start_run_compatibility(plate_run_dir, effective_images_dir, adoptable_only=True)
                if plate_run_dir is not None and effective_images_dir is not None and effective_images_count > 0
                else {}
            )
            if compatibility.get("checked"):
                approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
                incomplete_count = int(compatibility.get("incomplete", 0) or 0)
                body_lines.extend(
                    [
                        f"Porównuję z aktualnym zbiorem obrazów E1: {self._format_project_start_asset_source(effective_images_dir)}",
                        self._format_project_start_annotation_adoption_summary(compatibility),
                    ]
                )
                if approved_overlap > 0:
                    body_lines.append(f"Pominięte, bo już zatwierdzone w projekcie: {approved_overlap}")
                if incomplete_count > 0:
                    body_lines.append(
                        f"Pominięte, bo anotacje nie obejmują wszystkich tablic zapisanych w nazwie pliku: {incomplete_count}"
                    )
                missing_preview = [str(name) for name in (compatibility.get("missing_names") or []) if str(name).strip()]
                if missing_preview:
                    body_lines.append("")
                    body_lines.append("Przykłady brakujących plików:")
                    body_lines.extend(f"- {name}" for name in missing_preview[:5])
            else:
                body_lines.append("Walidacja zgodności pojawi się po wskazaniu obrazów iteracji.")
        else:
            body_lines.append("Plik annotations.xml nie został jeszcze wskazany.")
        body_lines.append("")
        body_lines.append(
            "Ten zasób ma sens wtedy, gdy dla części albo całości wybranego katalogu zdjęć E1 istnieją już dobre, "
            "ręczne anotacje tablic. Program porównuje nazwy obrazów z pliku annotations.xml z obrazami "
            "w aktualnym zbiorze E1: zgodne po nazwie można przyjąć, pozostałe są pomijane."
        )
        body_lines.append("")
        body_lines.append(
            "Po imporcie AT trafia do kontroli w Z2. Dopiero zatwierdzenie w Z2 nadaje status [OK] "
            "i może odblokować dalszy tor pracy."
        )
    elif row_key == "char_run":
        title = "AZ - anotacje znaków"
        body_lines.append("Źródło: Nie wskazano")
        body_lines.append("")
        body_lines.append(
            "AZ oznacza gotowe anotacje znaków. To piąty możliwy zasób wejściowy iteracji, "
            "obok obrazów, modeli i anotacji tablic."
        )
        body_lines.append("")
        body_lines.append(
            "Import anotacji znaków nie jest jeszcze podłączony jako operacja E1. "
            "Na tym etapie zasób jest widoczny w grafie jako element przyszłego modelu warunków wejściowych."
        )
    elif row_key in {"plate_model", "char_model"}:
        is_plate = row_key == "plate_model"
        title = "Model tablic" if is_plate else "Model znaków"
        model_state = self._get_project_start_effective_model_state("plate" if is_plate else "char")
        model_path = str(model_state.get("path") or (plate_model_path if is_plate else char_model_path) or "").strip()
        if model_state:
            body_lines.extend(str(line or "").strip() for line in model_state.get("details", []) if str(line or "").strip())
        else:
            source_text = self._format_project_start_asset_source(model_path)
            body_lines.append(f"Źródło: {source_text}")
        if model_path and not model_state:
            identity = self._get_model_identity_label(model_path)
            created = self._format_model_created_label(model_path)
            if identity:
                body_lines.append(f"Typ modelu: {identity}")
            if created and created != "Utworzono: -":
                body_lines.append(created)
            try:
                is_valid, validation_message = self._validate_project_model_selection("plate" if is_plate else "char", Path(model_path))
            except Exception:
                is_valid, validation_message = False, "Nie udało się zweryfikować modelu."
            body_lines.append("")
            body_lines.append("Walidacja: " + ("OK" if is_valid else validation_message))
        elif not model_path:
            body_lines.append("Model nie został jeszcze wskazany.")
        body_lines.append("")
        if is_plate:
            body_lines.append(
                "Model tablic w E1 jest opcjonalnym zasobem startowym projektu. Wybierz go wtedy, "
                "gdy chcesz używać go do autoanotacji tablic w Z2 i traktować jako model, który "
                "zamierzasz dalej poprawiać treningiem w kolejnych iteracjach."
            )
            body_lines.append("")
            body_lines.append(
                "Jeśli później w Z2 wskażesz inny model tylko dla bieżącego runu, "
                "nie nadpisze to automatycznie modelu projektu."
            )
        else:
            body_lines.append(
                "Model znaków w E1 jest opcjonalnym zasobem startowym projektu. Wybierz go wtedy, "
                "gdy chcesz używać go w Z3 do detekcji znaków i traktować jako model, który "
                "zamierzasz dalej dotrenowywać na kolejnych datasetach znaków."
            )
            body_lines.append("")
            body_lines.append(
                "Ten model pozostaje też dostępny później jako projektowy punkt startowy pracy nad znakami."
            )
    else:
        body_lines.append("Brak dodatkowych szczegółów dla tego zasobu.")

    message = "\n".join(str(line or "").strip() for line in body_lines if str(line or "").strip())
    self.app.themed_info(title, message or "Brak danych.", parent=parent or self.frame, tone="info")

def _count_project_start_plate_detections(annotation_tab, ann) -> int:
    getter = getattr(annotation_tab, "_get_plate_detections", None)
    if callable(getter):
        try:
            return int(len(list(getter(ann) or [])))
        except Exception:
            pass

    plate_labels = {str(label or "").strip().lower() for label in getattr(CONFIG, "PLATE_LABELS", [])}
    count = 0
    for det in list(getattr(ann, "detections", []) or []):
        label = str(getattr(det, "label", "") or "").strip().lower()
        if label and label in plate_labels:
            count += 1
    return int(count)

def _get_project_start_filename_plate_texts(filename: str) -> list[str]:
    try:
        planner = CampaignIngestPlanner()
        return [
            str(text or "").strip().upper()
            for text in planner.extract_true_texts_from_filename(str(filename or ""))
            if str(text or "").strip()
        ]
    except Exception:
        return []

def _get_project_start_filename_plate_signature(filename: str) -> tuple[str, ...]:
    texts = [
        str(text or "").strip().upper()
        for text in _get_project_start_filename_plate_texts(filename)
        if str(text or "").strip()
    ]
    return tuple(texts)

def _project_start_annotation_covers_filename_plates(self, filename: str, plate_count: int) -> tuple[bool, int]:
    expected_texts = self._get_project_start_filename_plate_texts(filename)
    expected_count = int(len(expected_texts) or 0)
    if expected_count <= 0:
        return True, 0
    return bool(int(plate_count or 0) >= expected_count), expected_count

def _format_project_start_annotation_adoption_summary(compatibility: dict | None) -> str:
    payload = dict(compatibility or {})
    matched = int(payload.get("package_matched", payload.get("matched", 0)) or 0)
    plates = int(payload.get("package_matched_plate_count", payload.get("matched_plate_count", 0)) or 0)
    total_plates = int(payload.get("plate_count", 0) or 0)
    rejected_plates = max(0, total_plates - plates)
    return (
        f"Do kontroli: {plates} AT z {matched} obrazów. "
        f"Odrzucamy: {rejected_plates} AT."
    )

def _is_project_start_manual_plate_detection(det) -> bool:
    attrs = dict(getattr(det, "attributes", {}) or {})
    source = str(getattr(det, "_cvat_source", "") or attrs.get("source", "") or "").strip().lower()
    manual_source = str(attrs.get("manual_source", "") or "").strip().lower()
    manually_edited = str(attrs.get("manually_edited", "") or "").strip().lower() == "true"
    return bool(source == "manual" or manual_source or manually_edited)

def _load_project_start_annotation_run_manifest(self, run_dir: Path | None) -> dict:
    if run_dir is None:
        return {}
    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is not None:
        loader = getattr(annotation_tab, "_load_annotation_run_manifest", None)
        if callable(loader):
            try:
                payload = loader(Path(run_dir))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
    try:
        manifest_path = Path(run_dir) / "run_manifest.json"
        payload = PROJECT_CACHE.load_json(manifest_path, default={})
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

def _summarize_project_start_annotation_import_origin(
    self,
    run_dir: Path | None,
    compatibility: dict | None = None,
) -> dict:
    summary = {
        "scope_images": 0,
        "scope_plates": 0,
        "manual_plates": 0,
        "auto_plates": 0,
        "model_label": "",
        "model_path": "",
        "model_used_label": "Nie ustalono",
        "run_type": "",
    }
    if run_dir is None:
        return summary

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return summary

    xml_path = Path(run_dir) / "annotations.xml"
    try:
        annotations = annotation_tab._parse_cvat_preview_annotations(xml_path)
    except Exception:
        annotations = []

    matched_names = {
        str(name or "").strip().lower()
        for name in list(dict(compatibility or {}).get("matched_normalized_names") or [])
        if str(name or "").strip()
    }
    use_matched_scope = bool(matched_names)

    scope_images = 0
    scope_plates = 0
    manual_plates = 0
    auto_plates = 0
    for ann in list(annotations or []):
        filename = str(getattr(ann, "filename", "") or "").strip()
        normalized = CAMPAIGN._normalize_image_set_name(filename)
        if use_matched_scope and str(normalized or "").strip().lower() not in matched_names:
            continue
        plate_detections = [
            det for det in list(getattr(ann, "detections", []) or [])
            if str(getattr(det, "label", "") or "").strip().lower() in CONFIG.PLATE_LABELS
        ]
        if not plate_detections:
            continue
        scope_images += 1
        scope_plates += len(plate_detections)
        for det in plate_detections:
            if self._is_project_start_manual_plate_detection(det):
                manual_plates += 1
            else:
                auto_plates += 1

    manifest = self._load_project_start_annotation_run_manifest(run_dir)
    model_path = str(manifest.get("plate_model_path") or "").strip()
    model_label = str(manifest.get("plate_model_identity") or "").strip()
    if not model_label and model_path:
        try:
            model_label = self._get_model_identity_label(model_path)
        except Exception:
            model_label = ""
    if not model_label and model_path:
        try:
            model_label = Path(model_path).name
        except Exception:
            model_label = model_path
    run_type = str(manifest.get("annotation_run_type") or "").strip()

    if auto_plates > 0 and model_label:
        model_used_label = f"Tak: {model_label}"
    elif auto_plates > 0:
        model_used_label = "Prawdopodobnie tak, ale brak metadanych modelu"
    elif manual_plates > 0:
        model_used_label = "Nie - zgodny podzbiór wygląda na ręczny"
    else:
        model_used_label = "Nie ustalono"

    summary.update(
        scope_images=int(scope_images),
        scope_plates=int(scope_plates),
        manual_plates=int(manual_plates),
        auto_plates=int(auto_plates),
        model_label=str(model_label or ""),
        model_path=str(model_path or ""),
        model_used_label=str(model_used_label or ""),
        run_type=str(run_type or ""),
    )
    return summary

def _format_project_start_annotation_import_origin_table(self, summary: dict | None) -> str:
    payload = dict(summary or {})
    model_label = str(payload.get("model_used_label") or "Nie ustalono").strip()
    run_type = str(payload.get("run_type") or "").strip()
    if run_type == "manual_template":
        run_type_label = "Ręczny XML"
    elif run_type == "auto_annotation":
        run_type_label = "Autoanotacja"
    else:
        run_type_label = "Brak danych"

    rows = [
        ("Zakres importu", f"{int(payload.get('scope_images', 0) or 0)} obrazów / {int(payload.get('scope_plates', 0) or 0)} tablic"),
        ("Run źródłowy", run_type_label),
        ("Model tablic", model_label),
        ("Tablice ręczne", str(int(payload.get("manual_plates", 0) or 0))),
        ("Tablice z auto", str(int(payload.get("auto_plates", 0) or 0))),
    ]
    label_width = max(len(label) for label, _value in rows)
    lines = ["Podsumowanie importowanych anotacji:"]
    for label, value in rows:
        lines.append(f"  {label:<{label_width}} | {value}")
    return "\n".join(lines)

def _show_project_start_annotation_import_modal(
    self,
    *,
    xml_label: str,
    images_label: str,
    compatibility: dict,
    origin_summary: dict | None,
    partial_line: str,
    parent=None,
) -> str | None:
    palette = dict(getattr(self.app, "palette", {}) or {})
    panel = palette.get("panel", "#111827")
    panel_alt = palette.get("panel_alt", "#18212f")
    card = palette.get("card", panel_alt)
    fg = palette.get("fg", "#f3f4f6")
    muted = palette.get("muted", "#9ca3af")
    border = palette.get("border", "#334155")
    accent = palette.get("accent", "#22c55e")
    success = palette.get("success", accent)
    warning = palette.get("warning", "#f59e0b")
    error = palette.get("error", "#ef4444")

    dialog_parent = parent or getattr(self, "frame", None)
    dialog = tk.Toplevel(dialog_parent or getattr(self.app, "root", None))
    try:
        self.app.style_dialog_window(
            dialog,
            title="Import anotacji tablic",
            geometry="900x760",
            parent=dialog_parent,
        )
        body = self.app._build_themed_dialog_surface(dialog, tone="info")
    except Exception:
        dialog.title("Import anotacji tablic")
        dialog.configure(bg=panel)
        body = tk.Frame(dialog, bg=panel, bd=0, highlightthickness=0)
        body.pack(fill=tk.BOTH, expand=True)

    result = {"value": None}

    def _short_path(value: str, limit: int = 82) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text or "-"
        return "..." + text[-max(8, limit - 3):]

    def _run_type_label(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "manual_template":
            return "Ręczny XML"
        if normalized == "auto_annotation":
            return "Autoanotacja"
        return "Brak danych"

    def _route_label(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "plate":
            return "tor tablic"
        if normalized == "char":
            return "tor znaków"
        return "tor nie jest jeszcze wybrany"

    def _blend(color_a: str, color_b: str, factor: float) -> str:
        try:
            return blend_hex_colors(color_a, color_b, factor)
        except Exception:
            return color_b

    def _tone_color(tone: str) -> str:
        normalized = str(tone or "").strip().lower()
        if normalized == "success":
            return success
        if normalized == "warning":
            return warning
        if normalized == "error":
            return error
        if normalized == "accent":
            return accent
        return muted

    def _make_card(parent_widget, title: str, *, tone: str = "neutral"):
        tone_color = {
            "success": success,
            "warning": warning,
            "error": error,
            "accent": accent,
        }.get(tone, border)
        shell = tk.Frame(
            parent_widget,
            bg=card,
            bd=0,
            highlightthickness=1,
            highlightbackground=tone_color,
            highlightcolor=tone_color,
        )
        shell.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            shell,
            text=title,
            bg=card,
            fg=tone_color if tone != "neutral" else fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 4))
        return shell

    def _add_note(parent_widget, text: str, *, color: str = ""):
        tk.Label(
            parent_widget,
            text=text,
            bg=card,
            fg=color or fg,
            font=("Segoe UI", 9),
            wraplength=680,
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(0, 10))

    def _add_table(parent_widget, rows: list[tuple[str, str]], *, value_color: str = ""):
        table = tk.Frame(parent_widget, bg=card, bd=0, highlightthickness=1, highlightbackground=border)
        table.pack(fill=tk.X, padx=12, pady=(2, 12))
        table.grid_columnconfigure(0, weight=0, minsize=230)
        table.grid_columnconfigure(1, weight=1)
        for index, (label, value) in enumerate(rows):
            row_bg = panel_alt if index % 2 else card
            tk.Label(
                table,
                text=str(label),
                bg=row_bg,
                fg=muted,
                font=("Segoe UI", 9),
                anchor="w",
                padx=10,
                pady=6,
            ).grid(row=index, column=0, sticky="nsew")
            tk.Label(
                table,
                text=str(value),
                bg=row_bg,
                fg=value_color or fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                justify=tk.LEFT,
                wraplength=410,
                padx=10,
                pady=6,
            ).grid(row=index, column=1, sticky="nsew")
        return table

    def _add_color_summary(parent_widget, rows: list[dict]):
        table = tk.Frame(parent_widget, bg=card, bd=0, highlightthickness=1, highlightbackground=border)
        table.pack(fill=tk.X, padx=12, pady=(2, 12))
        table.grid_columnconfigure(0, weight=0, minsize=92)
        table.grid_columnconfigure(1, weight=0, minsize=180)
        table.grid_columnconfigure(2, weight=0, minsize=180)
        table.grid_columnconfigure(3, weight=1, minsize=280)
        header_bg = _blend(card, accent, 0.13)
        for column, text in enumerate(("Stan", "Co sprawdzam", "Wynik", "Znaczenie")):
            tk.Label(
                table,
                text=text,
                bg=header_bg,
                fg=fg,
                font=("Segoe UI", 8, "bold"),
                anchor="w",
                padx=9,
                pady=6,
            ).grid(row=0, column=column, sticky="nsew")
        for index, row in enumerate(list(rows or []), start=1):
            tone = str(row.get("tone", "neutral") or "neutral").strip().lower()
            tone_color = _tone_color(tone)
            base_bg = panel_alt if index % 2 else card
            row_bg = _blend(base_bg, tone_color, 0.07 if tone != "neutral" else 0.0)
            badge_bg = _blend(row_bg, tone_color, 0.20 if tone != "neutral" else 0.08)
            tk.Label(
                table,
                text=str(row.get("status", "") or "-"),
                bg=badge_bg,
                fg=tone_color if tone != "neutral" else muted,
                font=("Segoe UI", 8, "bold"),
                anchor="center",
                padx=8,
                pady=6,
            ).grid(row=index, column=0, sticky="nsew")
            tk.Label(
                table,
                text=str(row.get("label", "") or "-"),
                bg=row_bg,
                fg=fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                justify=tk.LEFT,
                padx=9,
                pady=6,
            ).grid(row=index, column=1, sticky="nsew")
            tk.Label(
                table,
                text=str(row.get("value", "") or "-"),
                bg=row_bg,
                fg=tone_color if tone != "neutral" else fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                justify=tk.LEFT,
                wraplength=170,
                padx=9,
                pady=6,
            ).grid(row=index, column=2, sticky="nsew")
            tk.Label(
                table,
                text=str(row.get("note", "") or "-"),
                bg=row_bg,
                fg=muted if tone != "error" else error,
                font=("Segoe UI", 8),
                anchor="w",
                justify=tk.LEFT,
                wraplength=340,
                padx=9,
                pady=6,
            ).grid(row=index, column=3, sticky="nsew")
        return table

    def _close_with(value):
        result["value"] = value
        try:
            dialog.destroy()
        except Exception:
            pass

    content = tk.Frame(body, bg=panel, bd=0, highlightthickness=0)
    content.pack(fill=tk.BOTH, expand=True, padx=18, pady=(16, 8))

    tk.Label(
        content,
        text="Import anotacji tablic AT do kontroli",
        bg=panel,
        fg=fg,
        font=("Segoe UI", 15, "bold"),
        anchor="w",
    ).pack(fill=tk.X)
    tk.Label(
        content,
        text=(
            "Porównuję wybrane AT z aktualnym zbiorem obrazów. "
            "Importujemy tylko to, co ma obraz i nie było jeszcze zatwierdzone."
        ),
        bg=panel,
        fg=muted,
        font=("Segoe UI", 9),
        wraplength=840,
        justify=tk.LEFT,
        anchor="w",
    ).pack(fill=tk.X, pady=(4, 12))

    source_card = _make_card(content, "Źródła")
    _add_table(
        source_card,
        [
            ("Plik anotacji", _short_path(xml_label)),
            ("Zbiór obrazów", _short_path(images_label)),
        ],
    )

    payload = dict(origin_summary or {})
    compatibility_payload = dict(compatibility or {})

    package_count = int(compatibility_payload.get("package_image_count", 0) or 0)
    xml_total = int(compatibility_payload.get("total", 0) or 0)
    xml_plate_count = int(compatibility_payload.get("plate_count", payload.get("scope_plates", 0)) or 0)
    package_matched = int(compatibility_payload.get("package_matched", compatibility_payload.get("matched", 0)) or 0)
    package_matched_plates = int(
        compatibility_payload.get("package_matched_plate_count", compatibility_payload.get("matched_plate_count", 0)) or 0
    )
    to_control_images = int(compatibility_payload.get("adoptable_matched", package_matched) or 0)
    to_control_plates = int(compatibility_payload.get("adoptable_matched_plate_count", package_matched_plates) or 0)
    already_ok_images = int(compatibility_payload.get("approved_overlap", 0) or 0)
    already_ok_plates = int(compatibility_payload.get("approved_overlap_plates", 0) or 0)
    true_missing = int(compatibility_payload.get("missing", 0) or 0)
    true_incomplete = int(compatibility_payload.get("incomplete", 0) or 0)
    rejected_images = max(0, int(xml_total or 0) - int(package_matched or 0))
    rejected_plates = max(0, int(xml_plate_count or 0) - int(package_matched_plates or 0))
    can_import = bool(to_control_plates > 0)
    conclusion_tone = (
        "success"
        if can_import and true_missing <= 0 and true_incomplete <= 0
        else "warning"
        if can_import or package_matched > 0
        else "error"
    )
    conclusion_text = (
        "Pasujące AT, które nie były [OK], trafią do kontroli w Z2."
        if can_import
        else "Pasujące AT są już zatwierdzone [OK], więc niczego nie importujemy."
        if package_matched > 0
        else "Brak AT pasujących do aktualnego zbioru obrazów."
    )
    match_card = _make_card(content, "Podsumowanie importu AT", tone=conclusion_tone)
    _add_note(
        match_card,
        "Obrazy służą wyłącznie do dopasowania nazw. Importujemy anotacje tablic AT, nie obrazy.",
        color=fg,
    )
    _add_color_summary(
        match_card,
        [
            {
                "status": "ZBIÓR",
                "tone": "neutral",
                "label": "Obrazy w zasobach",
                "value": f"{package_count} obrazów",
                "note": "Zbiór użyty do sprawdzenia zgodności nazw.",
            },
            {
                "status": "XML",
                "tone": "neutral",
                "label": "Wybrane anotacje tablic",
                "value": f"{xml_plate_count} AT",
                "note": f"Anotacje zapisane w annotations.xml dla {xml_total} obrazów.",
            },
            {
                "status": "PASUJE" if package_matched > 0 else "BRAK",
                "tone": "success" if package_matched > 0 else "error",
                "label": "Pasujące",
                "value": f"{package_matched_plates} AT",
                "note": (
                    f"Do kontroli: {to_control_plates} AT z {to_control_images} obrazów. "
                    f"Pomijamy już [OK]: {already_ok_plates} AT z {already_ok_images} obrazów."
                ),
            },
            {
                "status": "NIE" if rejected_plates > 0 else "OK",
                "tone": "error" if rejected_plates > 0 else "success",
                "label": "Niepasujące",
                "value": f"{rejected_plates} AT",
                "note": f"Brakuje obrazu w aktualnym zbiorze: {rejected_images} obrazów z XML.",
            },
            {
                "status": "WNIOSEK",
                "tone": conclusion_tone,
                "label": "Decyzja",
                "value": conclusion_text,
                "note": "Import zawsze trafia do kontroli. Status [OK] nadajesz dopiero po sprawdzeniu w Z2.",
            },
        ],
    )

    if str(partial_line or "").strip():
        partial_card = _make_card(content, "Podsumowanie decyzji", tone="warning")
        _add_color_summary(
            partial_card,
            [
                {
                    "status": "INFO",
                    "tone": "warning",
                    "label": "Zakres importu",
                    "value": "Częściowy wynik",
                    "note": str(partial_line).strip(),
                }
            ],
        )

    btn_row = tk.Frame(body, bg=panel, bd=0, highlightthickness=0)
    btn_row.pack(fill=tk.X, padx=18, pady=(4, 16))
    ttk.Button(
        btn_row,
        text="Anuluj",
        command=lambda: _close_with(None),
        style="TButton",
    ).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(
        btn_row,
        text="Importuj pasujące AT do kontroli w Z2",
        command=lambda: _close_with("draft"),
        style="Accent.TButton",
        state=(tk.NORMAL if can_import else tk.DISABLED),
    ).pack(side=tk.RIGHT, padx=(8, 0))

    try:
        self.app._fit_dialog_to_content(dialog, parent=dialog_parent, min_width=900, min_height=720)
    except Exception:
        pass
    dialog.bind("<Escape>", lambda _event: _close_with(None))
    dialog.bind("<Return>", lambda _event: _close_with("draft") if can_import else None)
    try:
        dialog.focus_set()
        dialog.grab_set()
    except Exception:
        pass
    dialog.wait_window()
    return result["value"]

def _choose_project_start_annotation_import_mode(
    self,
    *,
    xml_label: str,
    images_label: str,
    compatibility: dict,
    origin_table: str,
    origin_summary: dict | None = None,
    partial_line: str,
    parent=None,
) -> str | None:
    try:
        choice = self._show_project_start_annotation_import_modal(
            xml_label=xml_label,
            images_label=images_label,
            compatibility=compatibility,
            origin_summary=origin_summary,
            partial_line=partial_line,
            parent=parent,
        )
        if choice in {"approved", "draft"}:
            return choice
        if choice is None:
            return None
    except Exception as exc:
        logger.debug(f"Nie udało się zbudować graficznego modala importu E1: {exc}")

    message = (
        "Program znalazł AT pasujące do aktualnego zbioru obrazów.\n\n"
        f"Plik anotacji: {xml_label}\n"
        f"Zbiór obrazów: {images_label}\n"
        f"{self._format_project_start_annotation_adoption_summary(compatibility)}\n\n"
        f"{origin_table}\n\n"
        f"{partial_line}\n\n"
        "Pasujące AT, które nie były [OK], trafią do kontroli w Z2. Status [OK] nadajesz dopiero po sprawdzeniu.\n"
        "AT bez obrazu o tej samej nazwie w aktualnym zbiorze zostaną odrzucone."
    )
    dialog = getattr(self.app, "themed_message_dialog", None)
    if callable(dialog):
        choice = dialog(
            "Import anotacji tablic",
            message,
            parent=parent or getattr(self, "frame", None),
            buttons=[
                "Anuluj",
                "Importuj AT do kontroli w Z2",
            ],
            default_button="Importuj AT do kontroli w Z2",
            tone="info",
            wraplength=680,
        )
    else:
        choice = messagebox.askokcancel(
            "Import anotacji tablic",
            message,
            parent=parent or getattr(self, "frame", None),
        )
        if choice is True:
            choice = "Importuj AT do kontroli w Z2"
        else:
            choice = None

    if choice in {"Importuj do kontroli", "Sprawdź roboczo w Z2", "Importuj AT do kontroli w Z2"}:
        return "draft"
    return None

def _check_project_start_run_compatibility(
    self,
    run_dir: Path | None,
    expected_images_dir: Path | None,
    *,
    adoptable_only: bool = False,
) -> dict:
    result = {
        "ok": False,
        "checked": False,
        "total": 0,
        "matched": 0,
        "missing": 0,
        "missing_names": [],
        "approved_overlap": 0,
        "approved_overlap_plates": 0,
        "approved_overlap_names": [],
        "adoptable_image_count": 0,
        "adoptable_matched": 0,
        "adoptable_matched_plate_count": 0,
        "adoptable_matched_names": [],
        "adoptable_matched_normalized_names": [],
        "package_matched": 0,
        "package_matched_plate_count": 0,
        "package_matched_names": [],
        "package_matched_normalized_names": [],
        "matched_plate_count": 0,
        "incomplete": 0,
        "incomplete_names": [],
        "matched_names": [],
        "matched_normalized_names": [],
        "missing_normalized_names": [],
        "images_dir": None,
        "package_image_set_token": "",
        "package_image_count": 0,
        "xml_image_set_token": "",
        "xml_image_count": 0,
        "plate_images": 0,
        "plate_count": 0,
        "token_match": False,
    }

    if run_dir is None or expected_images_dir is None:
        return result

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return result

    try:
        safe_run_dir = annotation_tab._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    except Exception:
        safe_run_dir = None

    if safe_run_dir is None:
        try:
            external_run_dir = Path(run_dir)
            if external_run_dir.exists() and external_run_dir.is_dir() and (external_run_dir / "annotations.xml").exists():
                safe_run_dir = external_run_dir
            else:
                return result
        except Exception:
            return result

    try:
        images_dir = Path(expected_images_dir)
    except Exception:
        return result

    try:
        if not images_dir.exists() or not images_dir.is_dir():
            return result
    except Exception:
        return result

    if adoptable_only:
        package_normalized_names = self._scan_project_start_normalized_image_names(images_dir)
        if not package_normalized_names:
            package_normalized_names = self._get_project_start_normalized_image_names(images_dir)
        package_image_count = len(package_normalized_names)
        package_image_set_token = CAMPAIGN.build_image_name_set_token(package_normalized_names, images_dir=images_dir)
    else:
        package_image_set_token, package_image_count = self._build_project_start_image_set_token(
            images_dir=images_dir,
            manifest=self._load_ingest_manifest_cached(),
        )
    result["package_image_set_token"] = str(package_image_set_token or "").strip()
    result["package_image_count"] = int(package_image_count or 0)

    try:
        annotations = annotation_tab._parse_cvat_preview_annotations(safe_run_dir / "annotations.xml")
    except Exception:
        return result

    plate_annotations = []
    plate_count = 0
    plate_count_by_normalized: dict[str, int] = {}
    incomplete_name_by_normalized: dict[str, str] = {}
    for ann in annotations:
        ann_plate_count = self._count_project_start_plate_detections(annotation_tab, ann)
        if ann_plate_count <= 0:
            continue
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        normalized_name = CAMPAIGN._normalize_image_set_name(filename)
        if not normalized_name:
            continue
        covers_filename, expected_plate_count = self._project_start_annotation_covers_filename_plates(
            filename,
            ann_plate_count,
        )
        if not covers_filename:
            try:
                incomplete_name_by_normalized[normalized_name] = (
                    f"{Path(filename).name} ({int(ann_plate_count)} z {int(expected_plate_count)} tablic z nazwy)"
                )
            except Exception:
                incomplete_name_by_normalized[normalized_name] = str(filename)
        plate_annotations.append(ann)
        plate_count += int(ann_plate_count)
        plate_count_by_normalized[normalized_name] = (
            int(plate_count_by_normalized.get(normalized_name, 0) or 0)
            + int(ann_plate_count)
        )

    xml_image_names = [
        str(getattr(ann, "filename", "") or "").strip()
        for ann in plate_annotations
        if str(getattr(ann, "filename", "") or "").strip()
    ]
    xml_name_by_normalized: dict[str, str] = {}
    for name in xml_image_names:
        normalized_name = CAMPAIGN._normalize_image_set_name(name)
        if normalized_name and normalized_name not in xml_name_by_normalized:
            try:
                xml_name_by_normalized[normalized_name] = Path(name).name
            except Exception:
                xml_name_by_normalized[normalized_name] = name
    xml_normalized_names = set(xml_name_by_normalized.keys())
    complete_xml_normalized_names = xml_normalized_names - set(incomplete_name_by_normalized.keys())

    xml_image_set_token = CAMPAIGN.build_image_name_set_token(xml_image_names)
    xml_image_count = len(xml_normalized_names)
    result["xml_image_set_token"] = str(xml_image_set_token or "").strip()
    result["xml_image_count"] = int(xml_image_count or 0)
    result["plate_images"] = int(xml_image_count or 0)
    result["plate_count"] = int(plate_count or 0)
    result["token_match"] = bool(
        package_image_set_token
        and xml_image_set_token
        and package_image_set_token == xml_image_set_token
    )

    if not plate_annotations:
        result["checked"] = True
        result["images_dir"] = images_dir
        return result

    if not adoptable_only:
        package_image_names = self._collect_project_start_image_names(
            images_dir=images_dir,
            manifest=self._load_ingest_manifest_cached(),
        )
        package_normalized_names = {
            CAMPAIGN._normalize_image_set_name(name)
            for name in package_image_names
            if CAMPAIGN._normalize_image_set_name(name)
        }

    approved_normalized_names = (
        self._get_project_start_approved_normalized_image_names()
        if adoptable_only
        else set()
    )
    package_normalized_names = set(package_normalized_names)
    approved_normalized_names = set(approved_normalized_names)
    adoption_candidate_names = package_normalized_names - approved_normalized_names
    package_matched_normalized_names = complete_xml_normalized_names & package_normalized_names
    adoptable_matched_normalized_names = package_matched_normalized_names & adoption_candidate_names
    matched_normalized_names = package_matched_normalized_names
    missing_normalized_names = sorted(xml_normalized_names - package_matched_normalized_names)
    approved_overlap_names = sorted(package_matched_normalized_names & approved_normalized_names)
    package_plate_signatures = {
        _get_project_start_filename_plate_signature(name)
        for name in package_normalized_names
        if _get_project_start_filename_plate_signature(name)
    }
    same_plate_text_normalized_names = set()
    if package_plate_signatures:
        for name in xml_normalized_names:
            if name in package_normalized_names:
                continue
            signature = _get_project_start_filename_plate_signature(name)
            if signature and signature in package_plate_signatures:
                same_plate_text_normalized_names.add(name)
    missing_names = [
        xml_name_by_normalized.get(name, name)
        for name in missing_normalized_names
        if name not in incomplete_name_by_normalized
        and name not in set(approved_overlap_names)
    ]
    approved_overlap_preview = [
        xml_name_by_normalized.get(name, name)
        for name in approved_overlap_names
    ]
    incomplete_names = [
        incomplete_name_by_normalized.get(name, xml_name_by_normalized.get(name, name))
        for name in sorted(set(incomplete_name_by_normalized.keys()) & xml_normalized_names)
    ]
    matched_plate_count = sum(
        int(plate_count_by_normalized.get(name, 0) or 0)
        for name in set(matched_normalized_names)
    )
    adoptable_matched_plate_count = sum(
        int(plate_count_by_normalized.get(name, 0) or 0)
        for name in set(adoptable_matched_normalized_names)
    )
    approved_overlap_plate_count = sum(
        int(plate_count_by_normalized.get(name, 0) or 0)
        for name in set(approved_overlap_names)
    )
    same_plate_text_plate_count = sum(
        int(plate_count_by_normalized.get(name, 0) or 0)
        for name in set(same_plate_text_normalized_names)
    )

    result.update(
        checked=True,
        total=len(xml_normalized_names),
        matched=len(matched_normalized_names),
        missing=len(missing_names),
        missing_names=list(missing_names[:5]),
        approved_overlap=len(approved_overlap_names),
        approved_overlap_plates=int(approved_overlap_plate_count or 0),
        approved_overlap_names=list(approved_overlap_preview[:5]),
        adoptable_image_count=len(adoption_candidate_names),
        adoptable_matched=len(adoptable_matched_normalized_names),
        adoptable_matched_plate_count=int(adoptable_matched_plate_count or 0),
        adoptable_matched_names=[
            xml_name_by_normalized.get(name, name)
            for name in sorted(adoptable_matched_normalized_names)
        ],
        adoptable_matched_normalized_names=sorted(adoptable_matched_normalized_names),
        package_matched=len(package_matched_normalized_names),
        package_matched_plate_count=int(matched_plate_count or 0),
        package_matched_names=[
            xml_name_by_normalized.get(name, name)
            for name in sorted(package_matched_normalized_names)
        ],
        package_matched_normalized_names=sorted(package_matched_normalized_names),
        same_plate_text_images=len(same_plate_text_normalized_names),
        same_plate_text_plate_count=int(same_plate_text_plate_count or 0),
        same_plate_text_names=[
            xml_name_by_normalized.get(name, name)
            for name in sorted(same_plate_text_normalized_names)
        ],
        same_plate_text_normalized_names=sorted(same_plate_text_normalized_names),
        matched_plate_count=int(matched_plate_count or 0),
        incomplete=len(incomplete_names),
        incomplete_names=list(incomplete_names[:5]),
        matched_names=[
            xml_name_by_normalized.get(name, name)
            for name in sorted(matched_normalized_names)
        ],
        matched_normalized_names=sorted(matched_normalized_names),
        missing_normalized_names=sorted(missing_normalized_names),
        images_dir=images_dir,
    )
    result["ok"] = bool(xml_normalized_names and not missing_normalized_names)
    return result

def _get_project_start_normalized_image_names(self, images_dir: Path | None) -> set[str]:
    if images_dir is None:
        return set()

    try:
        manifest = CAMPAIGN.load_ingest_manifest() or {}
    except Exception:
        manifest = {}
    if isinstance(manifest, dict):
        selected_manifest_names = {
            CAMPAIGN._normalize_image_set_name(str(item.get("name", "") or "").strip())
            for item in list(manifest.get("selected_images") or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        }
        selected_manifest_names = {name for name in selected_manifest_names if name}
        if selected_manifest_names:
            try:
                manifest_source_dir = CAMPAIGN.get_iteration_image_source_dir()
            except Exception:
                manifest_source_dir = None
            candidate_dirs = [
                manifest_source_dir,
                manifest.get("source_dir"),
                manifest.get("master_pool_dir"),
                manifest.get("target_dir"),
            ]
            for candidate_dir in candidate_dirs:
                if candidate_dir and self._campaign_paths_equivalent(images_dir, candidate_dir):
                    return selected_manifest_names

    image_names: list[str] = []
    try:
        for image_path in Path(images_dir).rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                image_names.append(str(image_path.name or "").strip())
    except Exception:
        image_names = []
    return {
        CAMPAIGN._normalize_image_set_name(name)
        for name in image_names
        if CAMPAIGN._normalize_image_set_name(name)
    }

def _summarize_project_start_xml_match(self, xml_path: Path | None, images_dir: Path | None) -> dict:
    summary = {
        "checked": False,
        "total": 0,
        "matched": 0,
        "missing": 0,
        "missing_names": [],
        "package_image_count": 0,
        "plate_images": 0,
        "plate_count": 0,
    }
    if xml_path is None or images_dir is None:
        return summary

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        return summary

    try:
        annotations = annotation_tab._parse_cvat_preview_annotations(Path(xml_path))
    except Exception:
        return summary

    xml_name_by_normalized: dict[str, str] = {}
    plate_count = 0
    for ann in annotations:
        ann_plate_count = self._count_project_start_plate_detections(annotation_tab, ann)
        if ann_plate_count <= 0:
            continue
        filename = str(getattr(ann, "filename", "") or "").strip()
        normalized_name = CAMPAIGN._normalize_image_set_name(filename)
        if normalized_name and normalized_name not in xml_name_by_normalized:
            xml_name_by_normalized[normalized_name] = Path(filename).name
        plate_count += int(ann_plate_count)

    package_names = self._get_project_start_normalized_image_names(images_dir)
    missing_names = sorted(set(xml_name_by_normalized.keys()) - package_names)
    summary.update(
        checked=True,
        total=len(xml_name_by_normalized),
        matched=max(0, len(xml_name_by_normalized) - len(missing_names)),
        missing=len(missing_names),
        missing_names=[xml_name_by_normalized.get(name, name) for name in missing_names[:5]],
        package_image_count=len(package_names),
        plate_images=len(xml_name_by_normalized),
        plate_count=int(plate_count or 0),
    )
    return summary

def _import_project_start_plate_run(
    self,
    selected_xml_path: Path | str | None = None,
    *,
    progress_callback=None,
    parent=None,
    refresh_dashboard_after_import: bool = True,
    confirm_import: bool = True,
) -> bool:
    def _notify_progress(percent: float, message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            value = max(0.0, min(100.0, float(percent or 0.0)))
        except Exception:
            value = 0.0
        try:
            progress_callback(value, str(message or ""))
        except Exception:
            pass

    if not CAMPAIGN.get_active_project_name():
        return

    dialog_parent = parent or getattr(self, "frame", None)

    _notify_progress(4, "Sprawdzam wybrany zbiór obrazów O...")
    try:
        image_source = dict(self._get_project_start_effective_images_source() or {})
    except Exception:
        image_source = {}
    images_dir = image_source.get("effective_dir")
    try:
        images_ready = bool(images_dir is not None and Path(images_dir).exists() and Path(images_dir).is_dir())
    except Exception:
        images_ready = False
    if not images_ready:
        message = (
            "Najpierw wskaż katalog obrazów O dla tej iteracji.\n\n"
            "AT jest zasobem zależnym od obrazów: program musi porównać nazwy zdjęć z annotations.xml "
            "z aktualną pulą O, żeby nie przyjąć anotacji z innej bazy."
        )
        try:
            self.app.themed_info(
                "Najpierw wskaż obrazy",
                message,
                parent=dialog_parent,
                tone="warning",
            )
        except Exception:
            messagebox.showwarning("Najpierw wskaż obrazy", message, parent=dialog_parent)
        return

    annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
    if annotation_tab is None:
        self.app.themed_error(
            "Brak Z2",
            "Nie udało się odnalezc zakładki Z2 potrzebnej do importu gotowych anotacji tablic.",
            parent=dialog_parent,
        )
        return

    self._set_project_start_mode("assets", refresh=False)

    selected_xml = str(selected_xml_path or "").strip()
    if not selected_xml:
        try:
            initial_dir = self._get_project_start_asset_initial_dir("plate_run")
        except Exception:
            initial_dir = Path(CONFIG.get_auto_annotations_dir("plate"))

        selected_xml = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Wskaż plik annotations.xml z gotowymi anotacjami tablic",
            filetypes=[
                ("Plik anotacji CVAT", "annotations.xml"),
                ("Pliki XML", "*.xml"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
    if not selected_xml:
        return

    try:
        selected_xml_path = Path(selected_xml)
    except Exception:
        self.app.themed_error("Błąd importu", "Nieprawidłowa ścieżka pliku annotations.xml.", parent=dialog_parent)
        return

    if str(selected_xml_path.name or "").strip().lower() != "annotations.xml":
        self.app.themed_error(
            "Błąd importu",
            "Wskaż właściwy plik annotations.xml z katalogu runu anotacji tablic.",
            parent=dialog_parent,
        )
        return

    try:
        if not selected_xml_path.exists() or not selected_xml_path.is_file():
            raise FileNotFoundError
    except Exception:
        self.app.themed_error(
            "Błąd importu",
            "Nie znaleziono wskazanego pliku annotations.xml.",
            parent=dialog_parent,
        )
        return
    _notify_progress(10, "Sprawdzam plik annotations.xml...")
    selected_run_dir = selected_xml_path.parent
    package_images_dir = selected_run_dir / "images"
    try:
        if (
            not package_images_dir.exists()
            or not package_images_dir.is_dir()
            or self._count_images_in_dir(package_images_dir, recursive=True) <= 0
        ):
            package_images_dir = None
    except Exception:
        package_images_dir = None

    project_images_dir = (
        image_source.get("effective_dir")
        or CAMPAIGN.get_master_pool_dir()
        or CAMPAIGN.get_iteration_image_source_dir()
        or CAMPAIGN.get_iteration_raw_dir()
    )

    try:
        safe_run_dir = annotation_tab._resolve_safe_annotation_run_dir(selected_run_dir, require_xml=True)
    except Exception:
        safe_run_dir = None

    # Import AT is always a local review draft. Even if the source run already
    # belongs to our workspace, its previous OK decisions must not open the
    # current gate before Z2 control approves them for the active O set.
    final_run_dir = None
    selected_images_dir = Path(project_images_dir) if project_images_dir is not None else None
    if selected_images_dir is not None:
        try:
            if not selected_images_dir.exists() or not selected_images_dir.is_dir() or self._count_images_in_dir(selected_images_dir, recursive=True) <= 0:
                selected_images_dir = None
        except Exception:
            selected_images_dir = None
    if selected_images_dir is None and package_images_dir is not None:
        selected_images_dir = package_images_dir
    selected_image_names = self._get_project_start_adoptable_normalized_image_names(selected_images_dir)
    original_xml_match = self._summarize_project_start_xml_match(selected_xml_path, selected_images_dir)
    pre_import_compatibility = {}

    if selected_images_dir is not None:
        _notify_progress(16, "Liczymy zgodność AT z aktualnym zbiorem O...")
        pre_import_compatibility = self._check_project_start_run_compatibility(
            selected_run_dir,
            selected_images_dir,
            adoptable_only=True,
        )

    def _workspace_import_progress(percent: float, message: str) -> None:
        try:
            raw_value = max(0.0, min(100.0, float(percent or 0.0)))
        except Exception:
            raw_value = 0.0
        _notify_progress(18.0 + raw_value * 0.52, message)

    package_matched_count = int(
        pre_import_compatibility.get("package_matched", pre_import_compatibility.get("matched", 0)) or 0
    )
    if selected_images_dir is not None and not selected_image_names and package_matched_count <= 0:
        self.app.themed_error(
            "Import anotacji tablic",
            (
                "W wybranym katalogu nie ma obrazów pasujących do annotations.xml.\n\n"
                "Wszystkie rozpoznane obrazy z tego zbioru są już zatwierdzone w projekcie albo katalog nie zawiera "
                "obrazów możliwych do powiązania z annotations.xml."
            ),
            parent=dialog_parent,
        )
        return

    if final_run_dir is None:
        allowed_import_names = selected_image_names or {
            str(name or "").strip()
            for name in list(pre_import_compatibility.get("matched_normalized_names") or [])
            if str(name or "").strip()
        }
        imported_run_dir, error_message, needs_image_dir = annotation_tab._import_external_annotation_run_to_workspace(
            selected_run_dir,
            compatible_images_dir=selected_images_dir,
            allowed_normalized_names=(allowed_import_names or None),
            copy_images=False,
            progress_callback=_workspace_import_progress,
        )
        if imported_run_dir is None and needs_image_dir:
            prompt_dir = selected_images_dir or package_images_dir or Path(CONFIG.DIR_1_RAW)
            compatible_dir = filedialog.askdirectory(
                initialdir=str(prompt_dir),
                title="Wskaż folder obrazów zgodnych z annotations.xml",
            )
            if not compatible_dir:
                return
            selected_images_dir = Path(compatible_dir)
            selected_image_names = self._get_project_start_adoptable_normalized_image_names(selected_images_dir)
            original_xml_match = self._summarize_project_start_xml_match(selected_xml_path, selected_images_dir)
            prompt_compatibility = self._check_project_start_run_compatibility(
                selected_run_dir,
                selected_images_dir,
                adoptable_only=True,
            )
            prompt_matched_count = int(
                prompt_compatibility.get("package_matched", prompt_compatibility.get("matched", 0)) or 0
            )
            if not selected_image_names and prompt_matched_count <= 0:
                self.app.themed_error(
                    "Import anotacji tablic",
                    (
                        "W wybranym katalogu nie ma obrazów pasujących do annotations.xml.\n\n"
                        "Import E1 pomija obrazy już zatwierdzone w projekcie."
                    ),
                    parent=dialog_parent,
                )
                return
            allowed_import_names = selected_image_names or {
                str(name or "").strip()
                for name in list(prompt_compatibility.get("matched_normalized_names") or [])
                if str(name or "").strip()
            }
            imported_run_dir, error_message, _needs_image_dir = annotation_tab._import_external_annotation_run_to_workspace(
                selected_run_dir,
                compatible_images_dir=selected_images_dir,
                allowed_normalized_names=(allowed_import_names or None),
                copy_images=False,
                progress_callback=_workspace_import_progress,
            )
        if imported_run_dir is None:
            self.app.themed_error(
                "Import anotacji tablic",
                error_message or "Nie udało się zaimportowac wskazanych anotacji tablic.",
                parent=dialog_parent,
            )
            return
        final_run_dir = imported_run_dir

    if final_run_dir is None:
        return

    resolved_images_dir = selected_images_dir or self._resolve_project_start_run_images_dir(final_run_dir)
    if resolved_images_dir is not None:
        try:
            if not resolved_images_dir.exists() or not resolved_images_dir.is_dir() or self._count_images_in_dir(resolved_images_dir, recursive=True) <= 0:
                resolved_images_dir = None
        except Exception:
            resolved_images_dir = None
    if resolved_images_dir is None:
        self.app.themed_error(
            "Import anotacji tablic",
            (
                "Nie udało się ustalić zbioru obrazów zgodnego z annotations.xml.\n\n"
                "Wskaż najpierw zbiór obrazów w E1 albo wybierz run anotacji, który zawiera obrazy lub manifest z input_dir."
            ),
            parent=dialog_parent,
        )
        return

    if not selected_image_names:
        selected_image_names = self._get_project_start_adoptable_normalized_image_names(resolved_images_dir)
        original_xml_match = self._summarize_project_start_xml_match(selected_xml_path, resolved_images_dir)

    _notify_progress(72, "Weryfikuję finalny zakres AT do kontroli...")
    compatibility = self._check_project_start_run_compatibility(final_run_dir, resolved_images_dir, adoptable_only=True)
    final_package_matched_count = int(
        compatibility.get("package_matched", compatibility.get("matched", 0)) or 0
    )
    if not selected_image_names and final_package_matched_count <= 0:
        self.app.themed_error(
            "Import anotacji tablic",
            (
                "W wybranym zbiorze projektu nie ma obrazów, do których można dopasować AT.\n\n"
                "Import E1 pomija obrazy, które są już zatwierdzone w projekcie. "
                "Wskaż zbiór obrazów zgodny z annotations.xml albo usuń błędnie wybrane źródło."
            ),
            parent=dialog_parent,
        )
        return

    if compatibility.get("checked") and not compatibility.get("ok"):
        matched_count = int(compatibility.get("matched", 0) or 0)
        if matched_count > 0:
            matched_names = {
                str(name or "").strip()
                for name in list(compatibility.get("matched_normalized_names") or [])
                if str(name or "").strip()
            }
            imported_run_dir, error_message, _needs_image_dir = annotation_tab._import_external_annotation_run_to_workspace(
                final_run_dir,
                compatible_images_dir=resolved_images_dir,
                allowed_normalized_names=(matched_names or selected_image_names),
                copy_images=False,
                progress_callback=_workspace_import_progress,
            )
            if imported_run_dir is None:
                self.app.themed_error(
                    "Import anotacji tablic",
                    error_message or "Nie udało się przygotować zgodnego podzbioru anotacji.",
                    parent=dialog_parent,
                )
                return
            final_run_dir = imported_run_dir
            compatibility = self._check_project_start_run_compatibility(final_run_dir, resolved_images_dir, adoptable_only=True)
        else:
            images_dir = compatibility.get("images_dir")
            missing_preview = "\n".join(str(name) for name in (compatibility.get("missing_names") or []))
            approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
            missing_count = int(compatibility.get("missing", 0) or 0)
            incomplete_count = int(compatibility.get("incomplete", 0) or 0)
            missing_suffix = ""
            remaining_missing = int(compatibility.get("missing", 0) or 0) - len(compatibility.get("missing_names") or [])
            if remaining_missing > 0:
                missing_suffix = f"\n... i jeszcze {remaining_missing} plikow."
            approved_line = (
                f"\nPominięte, bo już zatwierdzone w projekcie: {approved_overlap}"
                if approved_overlap > 0
                else ""
            )
            missing_line = f"\nAnotacje bez dopasowania w aktualnym zbiorze: {missing_count}" if missing_count > 0 else ""
            incomplete_line = (
                f"\nPominięte, bo anotacje nie obejmują wszystkich tablic zapisanych w nazwie pliku: {incomplete_count}"
                if incomplete_count > 0
                else ""
            )
            self.app.themed_error(
                "Import anotacji tablic",
                (
                    "Wybrane anotacje tablic nie pasują do aktualnego zbioru obrazów projektu.\n\n"
                    f"Zbiór obrazów E1: {Path(images_dir).name if images_dir is not None else 'brak'}\n"
                    f"{self._format_project_start_annotation_adoption_summary(compatibility)}"
                    f"{missing_line}{approved_line}{incomplete_line}\n\n"
                    "Nie znaleziono żadnego zgodnego wpisu do importu."
                    + (f"\n\nPrzykłady brakujących plików:\n{missing_preview}{missing_suffix}" if missing_preview else "")
                ),
                parent=dialog_parent,
            )
            return

    if compatibility.get("checked") and not compatibility.get("ok"):
        images_dir = compatibility.get("images_dir")
        missing_preview = "\n".join(str(name) for name in (compatibility.get("missing_names") or []))
        approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
        missing_count = int(compatibility.get("missing", 0) or 0)
        incomplete_count = int(compatibility.get("incomplete", 0) or 0)
        missing_suffix = ""
        remaining_missing = int(compatibility.get("missing", 0) or 0) - len(compatibility.get("missing_names") or [])
        if remaining_missing > 0:
            missing_suffix = f"\n... i jeszcze {remaining_missing} plikow."
        approved_line = (
            f"\nPominięte, bo już zatwierdzone w projekcie: {approved_overlap}"
            if approved_overlap > 0
            else ""
        )
        missing_line = f"\nAnotacje bez dopasowania w aktualnym zbiorze: {missing_count}" if missing_count > 0 else ""
        incomplete_line = (
            f"\nPominięte, bo anotacje nie obejmują wszystkich tablic zapisanych w nazwie pliku: {incomplete_count}"
            if incomplete_count > 0
            else ""
        )
        self.app.themed_error(
            "Import anotacji tablic",
            (
                "Wybrane anotacje tablic nie pasują do aktualnego zbioru obrazów projektu.\n\n"
                f"Zbiór obrazów E1: {Path(images_dir).name if images_dir is not None else 'brak'}\n"
                f"{self._format_project_start_annotation_adoption_summary(compatibility)}"
                f"{missing_line}{approved_line}{incomplete_line}\n\n"
                "Najpierw wskaż zbiór obrazów zgodny z annotations.xml."
                + (f"\n\nPrzyklady brakujacych plikow:\n{missing_preview}{missing_suffix}" if missing_preview else "")
            ),
            parent=dialog_parent,
        )
        return
    if compatibility.get("checked") and compatibility.get("ok"):
        images_dir = compatibility.get("images_dir")
        matched_count = int(compatibility.get("matched", 0) or 0)
        total_count = int(compatibility.get("total", 0) or 0)
        adoptable_matched = int(compatibility.get("adoptable_matched", matched_count) or 0)
        adoptable_matched_plates = int(compatibility.get("adoptable_matched_plate_count", 0) or 0)
        approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
        approved_overlap_plates = int(compatibility.get("approved_overlap_plates", 0) or 0)
        original_total_count = int(original_xml_match.get("total", 0) or total_count)
        original_missing_count = max(0, original_total_count - matched_count)
        xml_label = self._format_project_start_asset_source(selected_xml_path)
        images_label = (
            self._format_project_start_asset_source(images_dir)
            if images_dir is not None
            else "brak"
        )
        if adoptable_matched_plates <= 0:
            message = (
                "Nie ma nowych AT do kontroli.\n\n"
                f"Pasujące anotacje dotyczą obrazów, które są już zatwierdzone [OK]: "
                f"{approved_overlap} obrazów / {approved_overlap_plates} AT.\n\n"
                "Nie tworzymy duplikatów i nie nadpisujemy zatwierdzonych anotacji."
            )
            try:
                self.app.themed_info("Import anotacji tablic", message, parent=dialog_parent, tone="info")
            except Exception:
                messagebox.showinfo("Import anotacji tablic", message, parent=dialog_parent)
            return
        partial_line = (
            f"{original_missing_count} anotowanych obrazów z XML nie ma dopasowania w aktualnym zbiorze. "
            "Powiązane z nimi AT zostaną odrzucone."
            if original_missing_count > 0
            else (
                "Wszystkie anotowane obrazy z XML pasują po nazwie do aktualnego zbioru."
            )
        )
        if bool(confirm_import):
            origin_summary = self._summarize_project_start_annotation_import_origin(final_run_dir, compatibility)
            origin_table = self._format_project_start_annotation_import_origin_table(origin_summary)
            import_mode = self._choose_project_start_annotation_import_mode(
                xml_label=xml_label,
                images_label=images_label,
                compatibility=compatibility,
                origin_table=origin_table,
                origin_summary=origin_summary,
                partial_line=partial_line,
                parent=dialog_parent,
            )
            if import_mode is None:
                _notify_progress(0, "Import AT anulowany. Wybierz źródło ponownie albo zamknij okno.")
                return
            import_mode = "draft"
        else:
            _notify_progress(80, "Potwierdzono import wybranego AT do kontroli w Z2...")
            import_mode = "draft"
    else:
        import_mode = "draft"

    _notify_progress(86, "Zapisuję AT jako materiał do kontroli w Z2...")
    if resolved_images_dir is not None and resolved_images_dir.exists() and resolved_images_dir.is_dir():
        try:
            CAMPAIGN.set_master_pool_dir(resolved_images_dir)
        except Exception:
            pass

    try:
        annotation_tab._remember_campaign_manual_plate_source(
            run_dir=final_run_dir,
            input_dir=resolved_images_dir,
        )
    except Exception as e:
        logger.debug(f"Nie udało się zapamietac importowanego runu tablic dla kampanii: {e}")

    try:
        CAMPAIGN.set_project_start_plate_source(
            source_run_path=str(final_run_dir.resolve()),
            source_xml_path=str((final_run_dir / "annotations.xml").resolve()),
            source_input_path=str(resolved_images_dir.resolve()),
            source_mode=import_mode,
        )
    except Exception as e:
        logger.debug(f"Nie udało się zapamietac źródła anotacji startowych E1: {e}")

    self.current_ingest_plan = {}
    self._set_project_start_asset_scope(
        "plate_run",
        self._infer_project_start_asset_scope_from_path("plate_run", final_run_dir),
    )
    try:
        self._sync_iteration_artifact_registry_from_project_start()
    except Exception:
        pass
    _notify_progress(94, "Odświeżam zasoby bramki po imporcie AT...")
    if resolved_images_dir is not None and self._get_iteration_image_count() == 0:
        self._generate_ingest_plan()
    elif bool(refresh_dashboard_after_import):
        self._refresh_dashboard()

    try:
        run_name = final_run_dir.name
    except Exception:
        run_name = "anotacje tablic"
    status_text = (
        f"Zaimportowano AT do kontroli z pliku {run_name}/annotations.xml. "
        "Sprawdź je w Z2 i dopiero tam nadaj status [OK] właściwym obrazom."
    )
    try:
        self.app.update_status(status_text, "info")
    except Exception:
        pass
    _notify_progress(100, "AT zaimportowane do kontroli w Z2.")
    return True

def _get_step1_assets_intro_text() -> str:
    return (
        "Wskaż obrazy i dostępne zasoby. AT dopasowujemy do obrazów O, a modele sprawdzamy według ich typu. "
        "Tor wybierzesz w polu Praca bramki; określi on wykorzystanie zasobów, nie ich rodzaj."
    )

def _refresh_project_start_panel(self) -> None:
    shell = getattr(self, "ingest_start_shell", None)
    if shell is None:
        return

    try:
        CAMPAIGN.ensure_step1_image_source_restored_from_previous_iteration()
    except Exception:
        pass

    try:
        self._restore_project_start_asset_scopes_from_state()
    except Exception:
        pass

    step1_mode = self._get_step1_presentation_mode()
    panel_visible = step1_mode in {"operational_assets", "operational_summary"}
    show_summary_operational = bool(step1_mode == "operational_summary")
    show_assets_operational = bool(step1_mode == "operational_assets")
    try:
        self._set_pack_visibility(shell, panel_visible, fill=tk.X, padx=10, pady=(0, 6))
    except Exception:
        pass
    if not panel_visible:
        return

    mode = self._get_project_start_mode()
    palette = getattr(self.app, "palette", {})
    mode_selected = mode in {"fresh", "assets"}
    mode_is_assets = mode == "assets"

    image_source = self._get_project_start_effective_images_source()
    master_pool = image_source.get("master_dir")
    master_pool_exists = bool(image_source.get("master_exists"))
    master_pool_images = int(image_source.get("master_count", 0) or 0)
    iteration_images_dir = image_source.get("iteration_dir")
    iteration_images_exists = bool(image_source.get("iteration_exists"))
    iteration_images_count = int(image_source.get("iteration_count", 0) or 0)
    effective_images_dir = image_source.get("effective_dir")
    effective_images_count = int(image_source.get("effective_count", 0) or 0)
    effective_images_exists = bool(image_source.get("effective_exists"))
    effective_images_ready_for_annotations = bool(
        effective_images_dir is not None
        and effective_images_exists
    )
    plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    char_model_path = str(CAMPAIGN.get_global_model("char") or "").strip()
    plate_model_ready = bool(plate_model_path and Path(plate_model_path).exists())
    char_model_ready = bool(char_model_path and Path(char_model_path).exists())

    header_text = "Panel E1: Zasoby startowe iteracji"
    intro_text = ""
    master_title = "Glowne obrazy iteracji:"
    choose_label = "Wybierz obrazy"

    if mode_is_assets:
        intro_text = ""
        master_title = "Obrazy tej iteracji:"
    elif mode_selected:
        intro_text = ""

    try:
        self.ingest_header_lbl.config(text=header_text)
        self.ingest_intro_lbl.config(text=intro_text, fg=palette.get("muted", "#c7c7c7"))
        self.lbl_ingest_master_title.config(text=master_title)
        self.btn_choose_master_pool.config(text=choose_label)
    except Exception:
        pass

    try:
        self._set_pack_visibility(self.ingest_header_lbl, False)
        self._set_pack_visibility(
            self.ingest_intro_lbl,
            False,
        )
        self._set_pack_visibility(
            self.ingest_top_section,
            bool(mode_selected and show_summary_operational),
            fill=tk.X,
            padx=10,
            pady=(0, 6),
            before=self.ingest_body,
        )
        self._set_pack_visibility(
            self.ingest_body,
            bool(mode_selected and show_assets_operational),
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(0, 6),
            before=self.ingest_insights_toggle_shell,
        )
        self._set_pack_visibility(
            self.ingest_insights_toggle_shell,
            False,
            fill=tk.X,
            padx=10,
            pady=(0, 6),
            before=self.ingest_insights_shell,
        )
        self._set_pack_visibility(self.ingest_start_title_lbl, mode_selected, fill=tk.X)
        self._set_pack_visibility(self.ingest_start_summary_lbl, False)
        self._set_pack_visibility(self.ingest_status_shell, False)
        self._set_pack_visibility(self.btn_choose_master_pool, show_summary_operational, side=tk.RIGHT, padx=(8, 0))
        self._set_pack_visibility(self.btn_ingest_master_analysis, show_summary_operational, side=tk.RIGHT, padx=(0, 8))
    except Exception:
        pass

    try:
        if mode_selected and self.ingest_body is not None and show_summary_operational:
            self.ingest_top_section.pack_configure(before=self.ingest_body)
        if mode_selected and self.ingest_insights_toggle_shell is not None and show_assets_operational:
            self.ingest_body.pack_configure(before=self.ingest_insights_toggle_shell)
        if mode_selected and self.ingest_insights_shell is not None:
            self.ingest_insights_toggle_shell.pack_configure(before=self.ingest_insights_shell)
    except Exception:
        pass

    if self.ingest_start_title_lbl is not None:
        self.ingest_start_title_lbl.config(text="")
        self._set_pack_visibility(self.ingest_start_title_lbl, False)
    if self.ingest_start_summary_lbl is not None:
        self.ingest_start_summary_lbl.config(text="")
        self._set_pack_visibility(self.ingest_start_summary_lbl, False)
    if self.ingest_start_assets_title_lbl is not None:
        self.ingest_start_assets_title_lbl.config(text="Wymagania E1: zasoby zależne od wybranej ścieżki")

    try:
        self._set_project_start_badge_button_state(self.btn_ingest_start_fresh, text="Wybierz", enabled=True)
    except Exception:
        pass

    try:
        current_route_target = self._get_iteration_target()
    except Exception:
        current_route_target = ""
    route_selected = current_route_target in {"plate", "char"}
    char_preflight_state = {}
    if current_route_target == "char":
        try:
            char_preflight_state = dict(self._get_step1_char_route_preflight_state() or {})
        except Exception:
            char_preflight_state = {}
    char_min_images = int(
        char_preflight_state.get("min_images", getattr(self, "STEP1_CHAR_MIN_IMAGES", 10)) or 10
    )
    char_min_plates = int(
        char_preflight_state.get("min_plates", getattr(self, "STEP3_CHAR_MIN_PLATES", 10)) or 10
    )
    char_plate_material_count = int(char_preflight_state.get("plate_material_count", 0) or 0)
    char_material_ready = bool(char_preflight_state.get("material_ready"))
    char_image_potential = bool(char_preflight_state.get("image_potential"))
    try:
        current_iteration_path = normalize_iteration_path(CAMPAIGN.get_iteration_path())
    except Exception:
        current_iteration_path = ""
    if current_iteration_path and iteration_path_target(current_iteration_path) != current_route_target:
        current_iteration_path = ""
    if route_selected and not current_iteration_path:
        if current_route_target == "char" and char_material_ready:
            current_iteration_path = "char_from_ready_plates"
        else:
            current_iteration_path = default_iteration_path_for_target(current_route_target)
        # A resource refresh may describe a legacy route, never select it.

    if current_iteration_path == "plate_training":
        contract_summary = (
            "Wybrano tor tablic. E1 wymaga katalogu zdjęć, a E2/Z2 przygotuje anotacje tablic "
            "do datasetu i treningu modelu tablic."
        )
        assets_title = "2. Wymagania toru tablic: wejście do E2/Z2"
    elif current_iteration_path == "char_from_ready_plates":
        contract_summary = (
            f"Wybrano tor znaków z gotowych tablic. E1 wymaga źródła anotacji tablic "
            f"z minimum {char_min_plates} tablicami; katalog zdjęć jest tylko uzupełnieniem."
        )
        assets_title = "2. Wymagania toru znaków: istniejące źródło tablic jako wejście do E3/Z3"
    elif current_iteration_path == "char_from_images":
        contract_summary = (
            f"Wybrano tor znaków z nowych zdjęć. E1 wymaga katalogu zdjęć, a E2/Z2 przygotuje "
            f"minimum {char_min_plates} tablic do pracy nad znakami w E3/Z3."
        )
        assets_title = "2. Wymagania toru znaków: zdjęcia jako wejście do E2/Z2"
    else:
        contract_summary = (
            "Możesz wskazać obrazy, zaimportować pasujące AT i wybrać modele bez wyboru toru. "
            "Tor tablic albo znaków wybierzesz później w polu Praca bramki."
        )
        assets_title = "2. Zasoby E1: wybór niezależny od toru"

    try:
        if self.ingest_start_title_lbl is not None:
            self.ingest_start_title_lbl.config(text="Kontrakt E1: cel iteracji i warunki wejścia")
            self._set_pack_visibility(self.ingest_start_title_lbl, mode_selected, fill=tk.X)
        if self.ingest_start_summary_lbl is not None:
            self.ingest_start_summary_lbl.config(text=contract_summary, fg=palette.get("muted", "#c7c7c7"))
            self._set_pack_visibility(self.ingest_start_summary_lbl, mode_selected, fill=tk.X, pady=(4, 8))
        if self.ingest_start_assets_title_lbl is not None:
            self.ingest_start_assets_title_lbl.config(text=assets_title)
    except Exception:
        pass

    for widget, enabled in (
        (
            self.btn_ingest_import_plate_run,
            bool(
                mode_selected
                and show_assets_operational
                and effective_images_ready_for_annotations
            ),
        ),
        (self.btn_ingest_pick_plate_model, bool(mode_selected and show_assets_operational)),
        (self.btn_ingest_pick_char_model, bool(mode_selected and show_assets_operational)),
    ):
        if widget is None:
            continue
        try:
            self._set_project_start_badge_button_state(widget, enabled=enabled)
        except Exception:
            pass
    try:
        self._set_pack_visibility(
            self.ingest_start_assets_row,
            bool(mode_selected and show_assets_operational),
            fill=tk.X,
            pady=(10, 0),
        )
    except Exception:
        pass
    try:
        self._set_pack_visibility(getattr(self, "ingest_route_shell", None), False)
    except Exception:
        pass

    if not mode_selected:
        try:
            self._sync_ingest_wraps()
        except Exception:
            pass
        return

    images_source_text = self._format_project_start_asset_source(effective_images_dir)
    images_requirement = ""
    if current_iteration_path == "plate_training":
        images_requirement = "route_required_plate"
    elif current_iteration_path == "char_from_images":
        images_requirement = "route_required_char"
    elif current_iteration_path == "char_from_ready_plates":
        images_requirement = ""

    if current_iteration_path == "char_from_ready_plates":
        if effective_images_count > 0 and effective_images_dir is not None:
            images_validation_text = (
                f"Opcjonalne | wybrano {effective_images_count} obrazów. "
                "Ta ścieżka opiera się jednak na gotowych anotacjach tablic."
            )
            images_validation_tone = "muted"
        elif effective_images_exists and effective_images_count <= 0:
            images_validation_text = "Opcjonalne | wskazany katalog nie zawiera obrazów."
            images_validation_tone = "warning"
        else:
            images_validation_text = (
                "Opcjonalne | katalog zdjęć nie jest wymagany, jeśli źródło gotowych tablic spełnia minimum."
            )
            images_validation_tone = "muted"
    elif current_iteration_path == "char_from_images":
        if effective_images_count > 0 and effective_images_dir is not None:
            if char_image_potential:
                images_validation_text = (
                    f"Warunek toru znaków | OK | {effective_images_count}/{char_min_images} obrazów. "
                    f"W E2/Z2 przygotujesz minimum {char_min_plates} tablic."
                )
                images_validation_tone = "success"
            elif char_material_ready:
                images_validation_text = (
                    f"Uzupełniające | wybrano {effective_images_count} obrazów. "
                    f"Warunek toru znaków spełnia istniejące źródło tablic: {char_plate_material_count}/{char_min_plates}."
                )
                images_validation_tone = "muted"
            else:
                images_validation_text = (
                    f"Warunek toru znaków | za mało obrazów: {effective_images_count}/{char_min_images}. "
                    f"Alternatywnie importuj AT do kontroli i zatwierdź {char_min_plates} tablic w Z2."
                )
                images_validation_tone = "warning"
        elif effective_images_exists and effective_images_count <= 0:
            images_validation_text = "Warunek toru znaków | katalog nie zawiera obrazów."
            images_validation_tone = "error"
        elif char_material_ready:
            images_validation_text = (
                f"Uzupełniające | istniejące źródło tablic spełnia warunek: {char_plate_material_count}/{char_min_plates}. "
                "Zbiór obrazów wybierz tylko wtedy, gdy chcesz dodać nowe obrazy."
            )
            images_validation_tone = "muted"
        else:
            images_validation_text = (
                f"Warunek toru znaków | wybierz katalog min. {char_min_images} obrazów "
                f"albo importuj AT do kontroli i zatwierdź {char_min_plates} tablic w Z2."
            )
            images_validation_tone = "warning"
    elif current_iteration_path == "plate_training" and effective_images_count > 0 and effective_images_dir is not None:
        images_validation_text = f"Wymagane dla toru tablic | OK | {effective_images_count} obrazów"
        images_validation_tone = "success"
    elif current_iteration_path == "plate_training" and effective_images_exists and effective_images_count <= 0:
        images_validation_text = "Wymagane dla toru tablic | katalog nie zawiera obrazów."
        images_validation_tone = "error"
    elif not route_selected and effective_images_exists and effective_images_count > 0:
        images_validation_text = f"Obrazy wskazane | {effective_images_count} obrazów. Tor wybierzesz w polu Praca."
        images_validation_tone = "success"
    else:
        images_validation_text = (
            "Wymagane dla toru tablic | wskaż katalog obrazów tej iteracji."
            if current_iteration_path == "plate_training"
            else "Wskaż katalog obrazów O. Tor wybierzesz w polu Praca bramki."
        )
        images_validation_tone = "warning"
    images_meta = build_resource_contract_meta(
        "O",
        contract_ready=bool(
            (not route_selected or current_iteration_path in {"plate_training", "char_from_images"})
            and effective_images_count > 0
            and effective_images_dir is not None
            and effective_images_exists
        ),
        image_count=int(effective_images_count or 0),
        image_dir=str(effective_images_dir or ""),
        iteration_path=current_iteration_path,
    )
    self._set_project_start_asset_row_state(
        "images",
        source_text=images_source_text,
        source_path=str(effective_images_dir or ""),
        validation_text=images_validation_text,
        tone=images_validation_tone,
        requirement=images_requirement,
        counter_text=str(int(effective_images_count or 0)),
        meta=images_meta,
    )

    plate_source_info = self._get_project_start_plate_source_info()
    plate_run_path = str(plate_source_info.get("run_path") or "").strip()
    plate_xml_path = str(plate_source_info.get("xml_path") or "").strip()
    plate_source_mode = str(plate_source_info.get("source_mode") or "").strip().lower()
    plate_run_source_text = self._format_project_start_asset_source(plate_xml_path or plate_run_path)
    plate_run_counter_images = 0
    plate_run_counter_plates = 0
    plate_run_meta = build_resource_contract_meta(
        "O->AT",
        contract_ready=bool(
            char_material_ready
            and int(char_plate_material_count or 0) >= int(char_min_plates or 0)
        ),
        requires_rematch=False,
        source_mode=plate_source_mode,
        source_run_path=plate_run_path,
        source_xml_path=plate_xml_path,
        source_input_path=str(plate_source_info.get("images_path") or "").strip(),
        expected_images_dir=str(effective_images_dir or ""),
        matched_images=0,
        matched_plates=0,
        adoptable_images=0,
        adoptable_plates=0,
        approved_overlap_images=0,
        approved_overlap_plates=0,
        iteration_path=current_iteration_path,
    )
    if current_iteration_path == "char_from_ready_plates":
        plate_run_requirement = "route_required_char"
        if char_material_ready:
            plate_run_counter_plates = max(plate_run_counter_plates, int(char_plate_material_count or 0))
            plate_run_validation_text = (
                f"Warunek tej ścieżki | OK | źródło tablic: {char_plate_material_count}/{char_min_plates}."
            )
            plate_run_validation_tone = "success"
        else:
            plate_run_validation_text = (
                f"Warunek tej ścieżki | importuj AT do kontroli i zatwierdź minimum {char_min_plates} tablic w Z2."
            )
            plate_run_validation_tone = "warning"
    elif current_iteration_path == "char_from_images":
        plate_run_requirement = "alternative"
        if char_material_ready:
            plate_run_counter_plates = max(plate_run_counter_plates, int(char_plate_material_count or 0))
            plate_run_validation_text = (
                f"Alternatywa | istniejące źródło tablic jest dostępne: {char_plate_material_count}/{char_min_plates}. "
                "Możesz przełączyć graf na ścieżkę z istniejącym zbiorem tablic."
            )
            plate_run_validation_tone = "success"
        elif char_image_potential:
            plate_run_validation_text = (
                f"Alternatywa | import AT do kontroli może przygotować {char_min_plates} gotowych tablic po zatwierdzeniu w Z2."
            )
            plate_run_validation_tone = "muted"
        else:
            plate_run_validation_text = (
                f"Alternatywa | możesz importować AT do kontroli, ale ta ścieżka startuje ze zbioru obrazów."
            )
            plate_run_validation_tone = "muted"
    elif current_iteration_path == "plate_training":
        plate_run_validation_text = "Opcjonalne | import wcześniejszych anotacji do materiału toru tablic."
        plate_run_requirement = ""
        plate_run_validation_tone = "muted"
    else:
        plate_run_validation_text = "Opcjonalne | import wcześniejszych anotacji."
        plate_run_requirement = ""
        plate_run_validation_tone = "muted"
    if plate_run_path:
        try:
            plate_run_dir = Path(plate_run_path)
        except Exception:
            plate_run_dir = None
        if plate_run_dir is None or not plate_run_dir.exists() or not plate_run_dir.is_dir():
            plate_run_validation_text = "Nie znaleziono pliku annotations.xml."
            plate_run_validation_tone = "error"
            plate_run_meta.update(
                contract_ready=False,
                requires_rematch=False,
                contract_message=plate_run_validation_text,
            )
        elif not (plate_run_dir / "annotations.xml").exists():
            plate_run_validation_text = "Nie znaleziono pliku annotations.xml."
            plate_run_validation_tone = "error"
            plate_run_meta.update(
                contract_ready=False,
                requires_rematch=False,
                contract_message=plate_run_validation_text,
            )
        else:
            try:
                annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
                if annotation_tab is not None and hasattr(annotation_tab, "_get_run_plate_annotation_counts"):
                    xml_images, xml_plates = annotation_tab._get_run_plate_annotation_counts(plate_run_dir)
                    plate_run_counter_images = max(plate_run_counter_images, int(xml_images or 0))
                    plate_run_counter_plates = max(plate_run_counter_plates, int(xml_plates or 0))
            except Exception:
                pass
            if effective_images_dir is None or not effective_images_exists:
                plate_run_validation_text = "Najpierw wskaż obrazy tej iteracji."
                plate_run_validation_tone = "warning"
                plate_run_meta.update(
                    contract_ready=False,
                    requires_rematch=bool(plate_run_path),
                    contract_message=plate_run_validation_text,
                )
            else:
                compatibility = self._check_project_start_run_compatibility(plate_run_dir, effective_images_dir, adoptable_only=True)
                if compatibility.get("checked") and int(compatibility.get("total", 0) or 0) <= 0:
                    plate_run_validation_text = "XML nie zawiera anotacji tablic do importu."
                    plate_run_validation_tone = "warning"
                    plate_run_meta.update(
                        checked=True,
                        contract_ready=False,
                        requires_rematch=False,
                        contract_message=plate_run_validation_text,
                    )
                elif compatibility.get("checked") and compatibility.get("ok"):
                    plate_run_counter_images = int(compatibility.get("matched", 0) or 0)
                    plate_run_counter_plates = int(compatibility.get("matched_plate_count", 0) or 0)
                    approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
                    approved_overlap_plates = int(compatibility.get("approved_overlap_plates", 0) or 0)
                    adoptable_matched = int(compatibility.get("adoptable_matched", plate_run_counter_images) or 0)
                    adoptable_matched_plates = int(compatibility.get("adoptable_matched_plate_count", 0) or 0)
                    adoption_note = f" | do kontroli: {adoptable_matched} | już [OK]: {approved_overlap}"
                    plate_run_validation_text = (
                        f"OK | {int(compatibility.get('matched', 0) or 0)} anotacji -> "
                        f"{int(compatibility.get('matched', 0) or 0)} obrazów / "
                        f"{int(compatibility.get('matched_plate_count', 0) or 0)} tablic"
                        f"{adoption_note}"
                    )
                    plate_run_validation_tone = "success"
                    plate_run_meta.update(
                        checked=True,
                        contract_ready=bool(
                            char_material_ready
                            or approved_overlap_plates >= int(char_min_plates or 0)
                            or (
                                plate_source_mode == "approved"
                                and plate_run_counter_plates >= int(char_min_plates or 0)
                            )
                        ),
                        requires_rematch=False,
                        matched_images=int(compatibility.get("matched", 0) or 0),
                        matched_plates=plate_run_counter_plates,
                        adoptable_images=adoptable_matched,
                        adoptable_plates=adoptable_matched_plates,
                        approved_overlap_images=approved_overlap,
                        approved_overlap_plates=approved_overlap_plates,
                        contract_message=plate_run_validation_text,
                    )
                elif compatibility.get("checked") and int(compatibility.get("matched", 0) or 0) > 0:
                    approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
                    approved_overlap_plates = int(compatibility.get("approved_overlap_plates", 0) or 0)
                    incomplete_count = int(compatibility.get("incomplete", 0) or 0)
                    plate_run_counter_images = int(compatibility.get("matched", 0) or 0)
                    plate_run_counter_plates = int(compatibility.get("matched_plate_count", 0) or 0)
                    overlap_text = f" | pominięte zatwierdzone: {approved_overlap}" if approved_overlap > 0 else ""
                    incomplete_text = f" | niepełne wg nazwy: {incomplete_count}" if incomplete_count > 0 else ""
                    plate_run_validation_text = (
                        f"Wymaga ponownego dopasowania do O | {int(compatibility.get('matched', 0) or 0)} anotacji -> "
                        f"{int(compatibility.get('matched', 0) or 0)} obrazów / "
                        f"{int(compatibility.get('matched_plate_count', 0) or 0)} tablic"
                        f"{overlap_text}{incomplete_text}"
                    )
                    plate_run_validation_tone = "warning"
                    plate_run_meta.update(
                        checked=True,
                        contract_ready=bool(char_material_ready and approved_overlap_plates >= int(char_min_plates or 0)),
                        requires_rematch=not bool(char_material_ready and approved_overlap_plates >= int(char_min_plates or 0)),
                        matched_images=plate_run_counter_images,
                        matched_plates=plate_run_counter_plates,
                        approved_overlap_images=approved_overlap,
                        approved_overlap_plates=approved_overlap_plates,
                        incomplete_images=incomplete_count,
                        contract_message=plate_run_validation_text,
                    )
                elif compatibility.get("checked"):
                    approved_overlap = int(compatibility.get("approved_overlap", 0) or 0)
                    approved_overlap_plates = int(compatibility.get("approved_overlap_plates", 0) or 0)
                    incomplete_count = int(compatibility.get("incomplete", 0) or 0)
                    missing_count = int(compatibility.get("missing", 0) or 0)
                    if approved_overlap > 0 and missing_count <= 0 and incomplete_count <= 0:
                        plate_run_counter_images = int(approved_overlap or 0)
                        plate_run_validation_text = (
                            f"Zgodne | pasujące obrazy są już zatwierdzone [OK]: {approved_overlap}."
                        )
                        plate_run_validation_tone = "success"
                        plate_run_meta.update(
                            checked=True,
                            contract_ready=bool(char_material_ready or approved_overlap_plates >= int(char_min_plates or 0)),
                            requires_rematch=False,
                            approved_overlap_images=approved_overlap,
                            approved_overlap_plates=approved_overlap_plates,
                            contract_message=plate_run_validation_text,
                        )
                    elif approved_overlap > 0:
                        plate_run_counter_images = int(approved_overlap or 0)
                        plate_run_validation_text = (
                            f"Częściowo zgodne | już zatwierdzone [OK]: {approved_overlap} obrazów | "
                            f"do sprawdzenia: {missing_count + incomplete_count}"
                        )
                        plate_run_validation_tone = "warning"
                        plate_run_meta.update(
                            checked=True,
                            contract_ready=bool(char_material_ready and approved_overlap_plates >= int(char_min_plates or 0)),
                            requires_rematch=not bool(char_material_ready and approved_overlap_plates >= int(char_min_plates or 0)),
                            approved_overlap_images=approved_overlap,
                            approved_overlap_plates=approved_overlap_plates,
                            missing_images=missing_count,
                            incomplete_images=incomplete_count,
                            contract_message=plate_run_validation_text,
                        )
                    elif incomplete_count > 0:
                        plate_run_validation_text = (
                            f"Brak importu | {incomplete_count} anotacji nie obejmuje wszystkich tablic z nazwy pliku."
                        )
                        plate_run_validation_tone = "error"
                        plate_run_meta.update(
                            checked=True,
                            contract_ready=False,
                            requires_rematch=True,
                            incomplete_images=incomplete_count,
                            contract_message=plate_run_validation_text,
                        )
                    else:
                        plate_run_validation_text = (
                            f"Niezgodne z aktualnym zbiorem E1 | brak {missing_count} "
                            f"z {int(compatibility.get('total', 0) or 0)} anotowanych obrazów."
                        )
                        plate_run_validation_tone = "error"
                        plate_run_meta.update(
                            checked=True,
                            contract_ready=False,
                            requires_rematch=True,
                            missing_images=missing_count,
                            contract_message=plate_run_validation_text,
                        )
                else:
                    plate_run_validation_text = "Run zapisany. Sprawdź zgodność po wskazaniu aktualnego zbioru E1."
                    plate_run_validation_tone = "warning"
                    plate_run_meta.update(
                        contract_ready=False,
                        requires_rematch=True,
                        contract_message=plate_run_validation_text,
                    )
    if plate_source_mode == "draft" and plate_run_path:
        if bool(plate_run_meta.get("requires_rematch")):
            plate_run_validation_tone = "warning"
        elif current_iteration_path == "char_from_ready_plates":
            plate_run_validation_text = (
                f"AT do kontroli | {plate_run_counter_images} obrazów / {plate_run_counter_plates} tablic. "
                "To jeszcze nie spełnia T03. Sprawdź AT w Z2 i zatwierdź właściwe obrazy."
            )
            plate_run_validation_tone = "warning"
        else:
            plate_run_validation_text = (
                f"AT do kontroli | {plate_run_counter_images} obrazów / {plate_run_counter_plates} tablic. "
                "Sprawdź je w Z2; status [OK] nadajesz dopiero po kontroli."
            )
            plate_run_validation_tone = "muted"
        plate_run_meta.update(
            contract_ready=bool(
                char_material_ready
                and int(char_plate_material_count or 0) >= int(char_min_plates or 0)
            ),
            source_mode="draft",
            contract_message=plate_run_validation_text,
        )
    self._set_project_start_asset_row_state(
        "plate_run",
        source_text=plate_run_source_text,
        source_path=(plate_xml_path or plate_run_path),
        validation_text=plate_run_validation_text,
        tone=plate_run_validation_tone,
        requirement=plate_run_requirement,
        counter_text=(
            f"{plate_run_counter_images} obrazów / {plate_run_counter_plates} tablic"
            if plate_run_counter_images > 0 or plate_run_counter_plates > 0
            else "0"
        ),
        meta=plate_run_meta,
    )

    plate_model_source_text = self._format_project_start_asset_source(plate_model_path)
    plate_model_requirement = ""
    if current_route_target == "char":
        plate_model_requirement = "alternative"
    if not plate_model_ready:
        if current_route_target == "char":
            plate_model_validation_text = "Opcjonalne | może przyspieszyć przygotowanie tablic w E2/Z2."
        else:
            plate_model_validation_text = "Opcjonalne | model do autoanotacji i dalszego dotrenowania tablic."
        plate_model_validation_tone = "muted"
    else:
        try:
            plate_model_ok, plate_model_error = self._validate_project_model_selection("plate", Path(plate_model_path))
        except Exception:
            plate_model_ok, plate_model_error = False, "Nie udało się zweryfikować modelu."
        if plate_model_ok:
            plate_model_identity = self._get_model_identity_label(plate_model_path)
            plate_model_validation_text = plate_model_identity or "OK | poprawny model YOLO Pose."
            plate_model_validation_tone = "success"
        else:
            plate_model_validation_text = plate_model_error or "Model tablic nie przeszedł walidacji."
            plate_model_validation_tone = "error"
    plate_effective_model_state = self._get_project_start_effective_model_state("plate")
    if plate_effective_model_state:
        plate_model_path = str(plate_effective_model_state.get("path") or plate_model_path)
        plate_model_source_text = str(plate_effective_model_state.get("source") or plate_model_source_text)
        plate_model_validation_text = str(plate_effective_model_state.get("validation") or plate_model_validation_text)
        plate_model_validation_tone = str(plate_effective_model_state.get("tone") or plate_model_validation_tone)
    self._set_project_start_asset_row_state(
        "plate_model",
        source_text=plate_model_source_text,
        source_path=plate_model_path,
        validation_text=plate_model_validation_text,
        tone=plate_model_validation_tone,
        requirement=plate_model_requirement,
    )

    char_model_source_text = self._format_project_start_asset_source(char_model_path)
    char_model_requirement = ""
    if not char_model_ready:
        char_model_validation_text = "Opcjonalne | wskaż model znaków MZ. Jego wybór nie ustawia toru iteracji."
        char_model_validation_tone = "muted"
    else:
        try:
            char_model_ok, char_model_error = self._validate_project_model_selection("char", Path(char_model_path))
        except Exception:
            char_model_ok, char_model_error = False, "Nie udało się zweryfikować modelu."
        if char_model_ok:
            char_model_identity = self._get_model_identity_label(char_model_path)
            char_model_validation_text = char_model_identity or "OK | poprawny model YOLO Detect."
            char_model_validation_tone = "success"
        else:
            char_model_validation_text = char_model_error or "Model znaków nie przeszedł walidacji."
            char_model_validation_tone = "error"
    char_effective_model_state = self._get_project_start_effective_model_state("char")
    if char_effective_model_state:
        char_model_path = str(char_effective_model_state.get("path") or char_model_path)
        char_model_source_text = str(char_effective_model_state.get("source") or char_model_source_text)
        char_model_validation_text = str(char_effective_model_state.get("validation") or char_model_validation_text)
        char_model_validation_tone = str(char_effective_model_state.get("tone") or char_model_validation_tone)
    if current_route_target == "plate":
        char_model_validation_text += " Model znaków nie jest używany w torze tablic."
    self._set_project_start_asset_row_state(
        "char_model",
        source_text=char_model_source_text,
        source_path=char_model_path,
        validation_text=char_model_validation_text,
        tone=char_model_validation_tone,
        requirement=char_model_requirement,
    )

    char_run_validation_text = "Planowane | import AZ nie jest jeszcze dostępny w zasobach bramki."
    char_run_validation_tone = "muted"
    char_run_requirement = ""
    char_run_meta = build_resource_contract_meta(
        "O/AT->AZ",
        contract_ready=False,
        requires_rematch=False,
        source_mode="",
        iteration_path=current_iteration_path,
    )
    self._set_project_start_asset_row_state(
        "char_run",
        source_text="Nie wskazano",
        source_path="",
        validation_text=char_run_validation_text,
        tone=char_run_validation_tone,
        requirement=char_run_requirement,
        counter_text="0",
        meta=char_run_meta,
    )

    try:
        self._refresh_project_start_clear_buttons()
    except Exception:
        pass
    try:
        self._refresh_project_start_assets_table_theme(self._get_step1_ingest_frame_bg())
    except Exception:
        pass

    try:
        if show_assets_operational:
            self.ingest_insights_expanded = False
            self._set_pack_visibility(self.ingest_insights_toggle_shell, False)
            self._set_pack_visibility(self.ingest_insights_shell, False)
            self._set_pack_visibility(self.ingest_insights_toggle_btn, False)
            self._set_pack_visibility(self.ingest_insights_hint_lbl, False)
        else:
            self._refresh_ingest_insights_visibility(mode_selected=mode_selected)
        self._sync_ingest_wraps()
    except Exception:
        pass

    try:
        self._sync_iteration_artifact_registry_from_project_start()
    except Exception:
        pass
