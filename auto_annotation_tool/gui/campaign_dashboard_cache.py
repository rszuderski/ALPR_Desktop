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
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
def _clear_dashboard_perf_cache(self) -> None:
    cache = getattr(self, "_dashboard_perf_cache", None)
    if not isinstance(cache, dict):
        self._dashboard_perf_cache = {
            "image_counts": {},
            "json_payloads": {},
            "model_created": {},
            "approved_stats": {},
            "step2_source_states": {},
            "step2_view_models": {},
        }
        return
    for key in (
        "image_counts",
        "json_payloads",
        "model_created",
        "approved_stats",
        "step2_source_states",
        "step2_view_models",
        "image_name_sets",
    ):
        value = cache.get(key)
        if isinstance(value, dict):
            value.clear()
        else:
            cache[key] = {}

def _build_cache_token_for_path(path: Path | None):
    if path is None:
        return ("missing", "")
    try:
        candidate = Path(path)
    except Exception:
        return ("invalid", str(path))
    if not candidate.exists():
        return ("missing", str(candidate))
    try:
        stat = candidate.stat()
        resolved = str(candidate.resolve())
        return (resolved, int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return ("exists", str(candidate))

def _get_dashboard_cache_bucket(self, key: str) -> dict:
    cache = getattr(self, "_dashboard_perf_cache", None)
    if not isinstance(cache, dict):
        self._clear_dashboard_perf_cache()
        cache = getattr(self, "_dashboard_perf_cache", {})

    bucket = cache.get(key)
    if isinstance(bucket, dict):
        return bucket

    bucket = {}
    cache[key] = bucket
    return bucket

def _get_project_view_cache_path(self) -> Path | None:
    try:
        state_dir = CAMPAIGN.get_project_state_dir()
    except Exception:
        state_dir = None
    if state_dir is None:
        return None
    try:
        return Path(state_dir) / "wizard_view_cache.json"
    except Exception:
        return None

def _normalize_project_view_cache_value(cls, value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): cls._normalize_project_view_cache_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [cls._normalize_project_view_cache_value(item) for item in value]
    if isinstance(value, set):
        return sorted(cls._normalize_project_view_cache_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _serialize_project_view_cache_signature(cls, payload: dict) -> str:
    try:
        normalized = cls._normalize_project_view_cache_value(payload)
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return ""

def _build_directory_children_signature(self, path_value: Path | None, *, dirs_only: bool = False) -> list:
    if path_value is None:
        return ["missing", ""]
    try:
        root = Path(path_value)
    except Exception:
        return ["invalid", str(path_value)]
    try:
        if not root.exists() or not root.is_dir():
            return ["missing", str(root)]
    except Exception:
        return ["missing", str(root)]

    items: list[list] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: str(item.name).lower())
    except Exception:
        return ["error", str(root)]

    for child in children:
        try:
            is_dir = bool(child.is_dir())
            if dirs_only and not is_dir:
                continue
            stat = child.stat()
            items.append(
                [
                    "d" if is_dir else "f",
                    str(child.name),
                    int(getattr(stat, "st_mtime_ns", 0) or 0),
                    int(getattr(stat, "st_size", 0) or 0),
                ]
            )
        except Exception:
            continue

    return [str(root), items]

def _load_project_view_cache_store(self) -> dict:
    default_store = {
        "version": int(self._PROJECT_VIEW_CACHE_SCHEMA_VERSION),
        "entries": {},
    }
    cache_path = self._get_project_view_cache_path()
    if cache_path is None:
        return dict(default_store)

    loaded = PROJECT_CACHE.load_json(cache_path, default=default_store)
    if not isinstance(loaded, dict):
        return dict(default_store)
    if int(loaded.get("version", 0) or 0) != int(self._PROJECT_VIEW_CACHE_SCHEMA_VERSION):
        return dict(default_store)
    entries = loaded.get("entries", {})
    if not isinstance(entries, dict):
        loaded["entries"] = {}
    return loaded

def _save_project_view_cache_store(self, store: dict) -> None:
    if not isinstance(store, dict):
        return
    cache_path = self._get_project_view_cache_path()
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                self._normalize_project_view_cache_value(store),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        PROJECT_CACHE.invalidate_json(cache_path)
    except Exception as e:
        logger.debug(f"Nie udało się zapisać cache widoków projektu: {e}")

def _get_project_view_cache_entry(self, scope: str, signature: str) -> dict | None:
    if not scope or not signature:
        return None
    store = self._load_project_view_cache_store()
    entries = store.get("entries", {})
    if not isinstance(entries, dict):
        return None
    cached = entries.get(str(scope), {})
    if not isinstance(cached, dict):
        return None
    if str(cached.get("signature", "") or "") != str(signature or ""):
        return None
    payload = cached.get("payload")
    if not isinstance(payload, dict):
        return None
    return dict(payload)

def _set_project_view_cache_entry(self, scope: str, signature: str, payload: dict) -> None:
    if not scope or not signature or not isinstance(payload, dict):
        return
    store = self._load_project_view_cache_store()
    entries = store.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        store["entries"] = entries
    entries[str(scope)] = {
        "signature": str(signature or ""),
        "payload": self._normalize_project_view_cache_value(payload),
    }
    self._save_project_view_cache_store(store)

def _build_step1_manifest_context_signature(self, manifest: dict | None, iter_num: int) -> str:
    manifest = manifest if isinstance(manifest, dict) else {}
    try:
        manifest_path = CAMPAIGN.get_ingest_manifest_path(iter_num)
    except Exception:
        manifest_path = None
    try:
        approved_set_path = CAMPAIGN.get_plate_approved_set_path()
    except Exception:
        approved_set_path = None
    try:
        raw_root = CAMPAIGN.get_dir("raw")
    except Exception:
        raw_root = None
    try:
        target_iter_dir = CAMPAIGN.get_iteration_raw_dir(iter_num)
    except Exception:
        target_iter_dir = None
    try:
        master_pool_dir = CAMPAIGN.get_master_pool_dir()
    except Exception:
        master_pool_dir = None
    try:
        latest_plan_path = CAMPAIGN.get_latest_ingest_plan_path()
    except Exception:
        latest_plan_path = None
    try:
        latest_plan_summary_path = CAMPAIGN.get_latest_ingest_plan_summary_path()
    except Exception:
        latest_plan_summary_path = None

    signature_payload = {
        "scope": "step1_manifest_context",
        "iteration": int(iter_num or 0),
        "selection_mode": str(manifest.get("selection_mode", "") or "").strip().lower(),
        "selected_count": int(manifest.get("selected_count", 0) or 0),
        "manifest_token": list(self._build_cache_token_for_path(manifest_path)),
        "approved_set_token": list(self._build_cache_token_for_path(approved_set_path)),
        "raw_root_dirs_token": self._build_directory_children_signature(
            Path(raw_root) if raw_root is not None else None,
            dirs_only=True,
        ),
        "target_iteration_dir_token": list(
            self._build_cache_token_for_path(
                Path(target_iter_dir) if target_iter_dir is not None else None
            )
        ),
        "target_iteration_children_token": self._build_directory_children_signature(
            Path(target_iter_dir) if target_iter_dir is not None else None,
            dirs_only=False,
        ),
        "master_pool_dir_token": list(
            self._build_cache_token_for_path(
                Path(master_pool_dir) if master_pool_dir is not None else None
            )
        ),
        "latest_plan_token": list(self._build_cache_token_for_path(latest_plan_path)),
        "latest_plan_summary_token": list(self._build_cache_token_for_path(latest_plan_summary_path)),
    }
    return self._serialize_project_view_cache_signature(signature_payload)

def _build_step2_source_state_signature(self, target: str) -> str:
    normalized_target = self._normalize_iteration_target(target)
    if normalized_target not in {"plate", "char"}:
        return ""

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
        step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
        try:
            plate_model_info = dict(
                CAMPAIGN.get_effective_project_model(
                    "plate",
                    before_iteration=int(current_iteration or 1),
                )
                or {}
            )
        except Exception:
            plate_model_info = {}
        plate_model_path = str(plate_model_info.get("path") or "").strip()
        ingest_manifest_path = CAMPAIGN.get_ingest_manifest_path(current_iteration)
        approved_set_path = CAMPAIGN.get_plate_approved_set_path()
        artifact_registry_path = CAMPAIGN.get_artifact_registry_path()
        iter_image_source_dir = CAMPAIGN.get_iteration_image_source_dir(current_iteration)
        raw_iter_dir = CAMPAIGN.get_iteration_raw_dir(current_iteration)
        auto_ann_dir = CAMPAIGN.get_dir("auto_ann")
        step2_staging_run = CAMPAIGN.get_step2_staging_run()
        project_state_dir = CAMPAIGN.get_project_state_dir()
        project_start_mode = str(
            getattr(CAMPAIGN, "get_project_start_mode", lambda *_a, **_k: "fresh")() or "fresh"
        ).strip().lower()
        last_manual_source = dict(CAMPAIGN.get_last_plate_manual_source() or {})
    except Exception:
        return ""

    char_effective_state_path = None
    if project_state_dir is not None:
        try:
            char_effective_state_path = Path(project_state_dir) / "char_effective_source" / "source_state.json"
        except Exception:
            char_effective_state_path = None

    signature_payload = {
        "scope": "step2_source_state",
        "source_state_version": 3,
        "target": normalized_target,
        "iteration": int(current_iteration or 0),
        "current_step": int(current_step or 0),
        "step2_status": step2_status,
        "step3_status": step3_status,
        "project_start_mode": project_start_mode,
        "plate_model_token": list(
            self._build_cache_token_for_path(Path(plate_model_path) if plate_model_path else None)
        ),
        "ingest_manifest_token": list(self._build_cache_token_for_path(ingest_manifest_path)),
        "approved_set_token": list(self._build_cache_token_for_path(approved_set_path)),
        "artifact_registry_token": list(self._build_cache_token_for_path(artifact_registry_path)),
        "iter_image_source_dir_token": list(
            self._build_cache_token_for_path(Path(iter_image_source_dir) if iter_image_source_dir is not None else None)
        ),
        "raw_iter_dir_token": list(
            self._build_cache_token_for_path(Path(raw_iter_dir) if raw_iter_dir is not None else None)
        ),
        "auto_ann_dirs_token": self._build_directory_children_signature(
            Path(auto_ann_dir) if auto_ann_dir is not None else None,
            dirs_only=True,
        ),
        "step2_staging_run_token": list(
            self._build_cache_token_for_path(Path(step2_staging_run) if str(step2_staging_run or "").strip() else None)
        ),
        "char_effective_state_token": list(self._build_cache_token_for_path(char_effective_state_path)),
        "last_manual_source": {
            "source_run_path": str(last_manual_source.get("source_run_path") or "").strip(),
            "source_xml_path": str(last_manual_source.get("source_xml_path") or "").strip(),
            "source_input_path": str(last_manual_source.get("source_input_path") or "").strip(),
        },
    }
    return self._serialize_project_view_cache_signature(signature_payload)

def _elapsed_ms(started_at: float) -> float:
    try:
        return max(0.0, (perf_counter() - float(started_at)) * 1000.0)
    except Exception:
        return 0.0

def _log_perf(self, label: str, started_at: float, *, threshold_ms: float = 40.0, extra: str = "") -> None:
    elapsed_ms = self._elapsed_ms(started_at)
    if elapsed_ms < float(threshold_ms):
        return

    extra_text = f" | {extra}" if str(extra or "").strip() else ""
    logger.debug(f"[CampaignTab][PERF] {label}: {elapsed_ms:.1f} ms{extra_text}")

# ======================================================
# UI BUILD
# ======================================================
