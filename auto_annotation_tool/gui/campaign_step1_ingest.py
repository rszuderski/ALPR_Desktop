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
from . import campaign_model_status
from . import campaign_step1_assets
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
def _refresh_ingest_insights_visibility(self, mode_selected: bool | None = None) -> None:
    if mode_selected is None:
        mode_selected = bool(CAMPAIGN.get_active_project_name())

    expanded = bool(self.ingest_insights_expanded) and bool(mode_selected)
    btn = getattr(self, "ingest_insights_toggle_btn", None)
    hint = getattr(self, "ingest_insights_hint_lbl", None)

    if btn is not None:
        try:
            btn.config(text=("Ukryj analizę puli" if expanded else "Pokaż analizę puli"))
        except Exception:
            pass

    if hint is not None:
        hint_text = (
            "Histogram i rozkład znaków pomagają ocenić wybrany katalog zdjęć, ale nie są wymagane do zatwierdzenia E1."
            if expanded
            else "Histogram i rozkład znaków są dostępne jako sekcja dodatkowa."
        )
        try:
            hint.config(text=hint_text)
        except Exception:
            pass

    try:
        self._set_pack_visibility(self.ingest_insights_shell, expanded, fill=tk.X, padx=10, pady=(0, 6))
    except Exception:
        pass

    if expanded:
        try:
            self._refresh_ingest_balance_chart()
        except Exception:
            pass

def _get_ingest_summary_style(self) -> dict:
    palette = getattr(self.app, "palette", {})
    return {
        "bg": palette.get("panel", "#252526"),
        "border": palette.get("panel_border", palette.get("border", "#3c3c3c")),
        "fg": palette.get("fg", "#f3f3f3"),
        "muted": palette.get("muted", "#c7c7c7"),
    }

def _sync_ingest_wraps(self, _event=None):
    try:
        panel_width = int(getattr(self, "ingest_panel_frame").winfo_width() or 0)
    except Exception:
        panel_width = 680
    if panel_width <= 1:
        panel_width = 680

    try:
        asset_table_width = int(getattr(self, "ingest_start_assets_table_shell").winfo_width() or 0)
    except Exception:
        asset_table_width = 0
    if asset_table_width <= 1:
        asset_table_width = max(0, panel_width - 10)
    asset_table_width = max(asset_table_width, 620)

    narrow_assets_table = bool(asset_table_width < 800)
    if asset_table_width >= 940:
        asset_col0 = 132
        asset_col1 = 190
        asset_col3 = 146
        asset_col4 = 144
        asset_col5 = 70
    elif asset_table_width >= 800:
        asset_col0 = 118
        asset_col1 = 168
        asset_col3 = 128
        asset_col4 = 136
        asset_col5 = 66
    else:
        asset_col0 = 100
        asset_col1 = 140
        asset_col3 = 104
        asset_col4 = 128
        asset_col5 = 60
    fixed_asset_width = asset_col0 + asset_col1 + asset_col3 + asset_col4 + asset_col5
    asset_col2 = max(190, int(asset_table_width) - fixed_asset_width - 12)

    assets_table = getattr(self, "ingest_start_assets_table", None)
    if assets_table is not None:
        try:
            assets_table.grid_columnconfigure(0, minsize=asset_col0, weight=0)
            assets_table.grid_columnconfigure(1, minsize=asset_col1, weight=1)
            assets_table.grid_columnconfigure(2, minsize=asset_col2, weight=3)
            assets_table.grid_columnconfigure(3, minsize=asset_col3, weight=1)
            assets_table.grid_columnconfigure(4, minsize=asset_col4, weight=0)
            assets_table.grid_columnconfigure(5, minsize=asset_col5, weight=0)
            assets_table.grid_rowconfigure(0, minsize=34, weight=0)
        except Exception:
            pass

    try:
        header_texts = ("Zasób", "Źródło", "Walidacja", "Źródło ścieżki danych", "Akcja", "Więcej")
        for idx, header_lbl in enumerate(list(getattr(self, "ingest_start_assets_header_labels", []) or [])):
            if header_lbl is None or idx >= len(header_texts):
                continue
            header_lbl.config(
                text=header_texts[idx],
                font=("Segoe UI", 8, "bold"),
                wraplength=(max(88, asset_col3 - 12) if idx == 3 else 0),
                height=2,
                pady=0,
            )
    except Exception:
        pass

    try:
        info_width = max(int(getattr(self, "ingest_info_panel").winfo_width() or 0) - 18, 260)
    except Exception:
        info_width = 320

    try:
        chart_width = max(int(getattr(self, "ingest_chart_panel").winfo_width() or 0) - 18, 320)
    except Exception:
        chart_width = 460

    master_wrap = max(280, panel_width - 220)
    intro_wrap = max(420, panel_width - 40)
    asset_validation_wrap = max(136, asset_col2 - 18)

    for widget_name, wrap_value in (
        ("lbl_ingest_master_value", master_wrap),
        ("ingest_intro_lbl", intro_wrap),
        ("ingest_start_summary_lbl", intro_wrap),
        ("ingest_insights_hint_lbl", intro_wrap),
        ("ingest_logic_lbl", info_width),
        ("ingest_selection_lbl", info_width),
        ("ingest_balance_summary_lbl", chart_width),
    ):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.config(wraplength=wrap_value)
        except Exception:
            pass

    for row in list(getattr(self, "ingest_start_asset_row_widgets", {}).values() or []):
        if not isinstance(row, dict):
            continue
        row_key = str(row.get("row_key", "") or "").strip()
        try:
            source_lbl = row.get("source_lbl")
            if source_lbl is not None:
                source_lbl.config(wraplength=0, height=1, width=24)
        except Exception:
            pass
        try:
            validation_lbl = row.get("validation_lbl")
            if validation_lbl is not None:
                validation_lbl.config(wraplength=asset_validation_wrap)
        except Exception:
            pass
        try:
            source_scope_host = row.get("source_scope_host")
            if source_scope_host is not None:
                source_scope_host.config(width=max(94, asset_col3 - 6), height=30)
            action_host = row.get("action_host")
            if action_host is not None:
                action_host.config(width=max(136, asset_col4 - 4), height=30)
            primary_slot = row.get("action_primary_slot")
            if primary_slot is not None:
                primary_slot.config(width=62, height=28)
            clear_slot = row.get("action_clear_slot")
            if clear_slot is not None:
                clear_slot.config(width=74, height=28)
            details_host = row.get("details_host")
            if details_host is not None:
                details_host.config(width=max(58, asset_col5 - 4), height=30)
        except Exception:
            pass
        try:
            badge_list = list(row.get("scope_badges", []) or [])
            for idx, badge in enumerate(badge_list):
                shell = badge.get("shell")
                lamp = badge.get("lamp")
                lamp_id = badge.get("lamp_id")
                label = badge.get("label")
                compact_text = str(badge.get("compact_label") or badge.get("full_label") or "").strip()
                full_text = str(badge.get("full_label") or compact_text).strip()
                if shell is not None:
                    try:
                        shell.pack_forget()
                    except Exception:
                        pass
                    shell.pack(
                        side=tk.LEFT,
                        anchor="center",
                        padx=(0 if idx == 0 else 6, 0),
                        pady=0,
                    )
                    shell.config(padx=6, pady=2)
                if lamp is not None:
                    lamp.config(
                        width=8,
                        height=8,
                    )
                    if lamp_id is not None:
                        lamp.coords(
                            lamp_id,
                            1,
                            1,
                            7,
                            7,
                        )
                if label is not None:
                    label.config(
                        text=(compact_text if narrow_assets_table else full_text),
                        font=("Segoe UI", 8),
                    )
        except Exception:
            pass

    try:
        if self.btn_ingest_start_fresh is not None:
            self._set_project_start_badge_button_state(self.btn_ingest_start_fresh, text="Wybierz")
        if self.btn_ingest_import_plate_run is not None:
            self._set_project_start_badge_button_state(self.btn_ingest_import_plate_run, text="Import")
        if self.btn_ingest_pick_plate_model is not None:
            self._set_project_start_badge_button_state(self.btn_ingest_pick_plate_model, text="Wskaż")
        if self.btn_ingest_pick_char_model is not None:
            self._set_project_start_badge_button_state(self.btn_ingest_pick_char_model, text="Wskaż")
        for row_key, button in dict(getattr(self, "btn_ingest_asset_more", {}) or {}).items():
            if button is None:
                continue
            if row_key == "images":
                self._set_project_start_badge_button_state(button, text="Analiza")
            else:
                self._set_project_start_badge_button_state(button, text="Więcej")
    except Exception:
        pass

    status_wrap = max(440, panel_width - 70)
    for lbl in getattr(self, "ingest_status_labels", []):
        try:
            lbl.config(wraplength=status_wrap)
        except Exception:
            pass
    for value_lbl in list(getattr(self, "ingest_status_table_value_labels", []) or []):
        try:
            value_lbl.config(wraplength=max(180, status_wrap - 220))
        except Exception:
            pass

def _set_ingest_status_lines(self, rows=None, meta=""):
    table = getattr(self, "ingest_status_table", None)
    if table is not None:
        try:
            for child in list(table.winfo_children()):
                child.destroy()
        except Exception:
            pass
    self.ingest_status_table_value_labels = []
    labels = list(getattr(self, "ingest_status_labels", []) or [])
    for lbl in labels:
        try:
            lbl.config(text="")
        except Exception:
            pass
    try:
        self._set_pack_visibility(
            getattr(self, "ingest_status_shell", None),
            False,
        )
    except Exception:
        pass

def _load_ingest_manifest_cached(self) -> dict:
    try:
        manifest_path = CAMPAIGN.get_ingest_manifest_path()
    except Exception:
        manifest_path = None

    cache = getattr(self, "_dashboard_perf_cache", {})
    json_cache = cache.get("json_payloads", {}) if isinstance(cache, dict) else {}
    cache_key = ("ingest_manifest", self._build_cache_token_for_path(manifest_path))
    cached = json_cache.get(cache_key) if isinstance(json_cache, dict) else None
    if isinstance(cached, dict):
        return dict(cached)

    try:
        manifest = CAMPAIGN.load_ingest_manifest()
    except Exception:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}

    if isinstance(json_cache, dict):
        if len(json_cache) > 128:
            json_cache.clear()
        json_cache[cache_key] = dict(manifest)
    return dict(manifest)

def _load_previous_iteration_ingest_manifest(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}
    try:
        current_iter = max(1, int(CAMPAIGN.get_current_iteration_num() or 1))
    except Exception:
        current_iter = 1
    if current_iter <= 1:
        return {}

    for iter_num in range(current_iter - 1, 0, -1):
        try:
            manifest_path = CAMPAIGN.get_ingest_manifest_path(iter_num)
        except Exception:
            manifest_path = None
        cache = getattr(self, "_dashboard_perf_cache", {})
        json_cache = cache.get("json_payloads", {}) if isinstance(cache, dict) else {}
        cache_key = ("ingest_manifest_prev", iter_num, self._build_cache_token_for_path(manifest_path))
        cached = json_cache.get(cache_key) if isinstance(json_cache, dict) else None
        if isinstance(cached, dict) and cached:
            return dict(cached)
        try:
            manifest = CAMPAIGN.load_ingest_manifest(iter_num)
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict) or not manifest:
            continue
        if isinstance(json_cache, dict):
            if len(json_cache) > 128:
                json_cache.clear()
            json_cache[cache_key] = dict(manifest)
        return dict(manifest)
    return {}

def _load_latest_ingest_plan_for_current_iteration(self) -> dict:
    try:
        plan_path = CAMPAIGN.get_latest_ingest_plan_path()
    except Exception:
        plan_path = None

    cache = getattr(self, "_dashboard_perf_cache", {})
    json_cache = cache.get("json_payloads", {}) if isinstance(cache, dict) else {}
    cache_key = ("latest_ingest_plan", self._build_cache_token_for_path(plan_path))
    cached = json_cache.get(cache_key) if isinstance(json_cache, dict) else None
    if isinstance(cached, dict):
        plan = dict(cached)
    else:
        plan = CAMPAIGN.load_latest_ingest_plan()
        if not isinstance(plan, dict):
            plan = {}
        if isinstance(json_cache, dict):
            if len(json_cache) > 128:
                json_cache.clear()
            json_cache[cache_key] = dict(plan)
    if not isinstance(plan, dict):
        return {}
    if int(plan.get("iteration", 0) or 0) != CAMPAIGN.get_current_iteration_num():
        return {}
    if str(plan.get("project", "") or "").strip() != str(CAMPAIGN.get_active_project_name() or "").strip():
        return {}
    current_master_pool = CAMPAIGN.get_master_pool_dir()
    plan_master_pool = str(plan.get("master_pool_dir", "") or "").strip()
    if current_master_pool is None and plan_master_pool:
        return {}
    if current_master_pool is not None and plan_master_pool and plan_master_pool != str(current_master_pool):
        return {}
    return plan

def _normalize_ingest_plan_for_display(self, plan: dict, current_balance: dict | None = None) -> dict:
    if not isinstance(plan, dict):
        return {}

    selected_items = list(plan.get("selected", []) or [])
    if not selected_items:
        return {}

    normalized = dict(plan)
    try:
        selected_total = int(normalized.get("selected_total", 0) or 0)
    except Exception:
        selected_total = 0
    if selected_total <= 0:
        normalized["selected_total"] = len(selected_items)

    try:
        raw_total = int(normalized.get("raw_total", 0) or 0)
    except Exception:
        raw_total = 0
    if raw_total <= 0:
        normalized["raw_total"] = int(normalized.get("selected_total", len(selected_items)) or len(selected_items))

    if current_balance:
        normalized["current_balance"] = {
            ch: int((current_balance or {}).get(ch, 0) or 0)
            for ch in CHAR_ALPHABET
        }
    elif not isinstance(normalized.get("current_balance"), dict):
        normalized["current_balance"] = {ch: 0 for ch in CHAR_ALPHABET}

    normalized["selected"] = selected_items
    return normalized

def _build_ingest_plan_from_manifest_for_display(
    self,
    manifest: dict,
    current_balance: dict | None = None,
) -> dict:
    if not isinstance(manifest, dict) or not manifest:
        return {}

    selected_images = list(manifest.get("selected_images", []) or [])
    if not selected_images:
        return {}

    planner = CampaignIngestPlanner()
    selected_items = []
    selected_hist = Counter()
    master_pool_dir = CAMPAIGN.get_master_pool_dir()
    target_dir_text = str(manifest.get("target_dir", "") or "").strip()
    target_dir = Path(target_dir_text) if target_dir_text else CAMPAIGN.get_iteration_raw_dir()

    for item in selected_images:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue

        target_path_text = str(item.get("target_path", "") or "").strip()
        if not target_path_text and target_dir is not None:
            target_path_text = str((Path(target_dir) / name).resolve())

        source_path_text = str(item.get("source_path", "") or "").strip() or target_path_text
        ground_truth_texts = list(item.get("ground_truth_texts", []) or [])
        if not ground_truth_texts:
            ground_truth_texts = planner.extract_true_texts_from_filename(name)

        char_hist = item.get("char_histogram", {}) or {}
        if not isinstance(char_hist, dict) or not char_hist:
            char_hist = planner.build_char_histogram(ground_truth_texts)
        if not char_hist:
            continue

        selected_hist.update(char_hist)
        selected_items.append(
            {
                "name": name,
                "source_path": source_path_text,
                "source_key": str(item.get("source_key", "") or ""),
                "target_path": target_path_text,
                "ground_truth_texts": list(ground_truth_texts),
                "char_histogram": {ch: int(value) for ch, value in char_hist.items()},
                "score": 0.0,
                "score_details": {},
            }
        )

    if not selected_items:
        return {}

    current_counter = Counter()
    for ch in CHAR_ALPHABET:
        current_counter[ch] = int((current_balance or {}).get(ch, 0) or 0)
    predicted_counter = Counter(current_counter)
    predicted_counter.update(selected_hist)

    try:
        selected_count = int(manifest.get("selected_count", len(selected_items)) or len(selected_items))
    except Exception:
        selected_count = len(selected_items)

    return {
        "ok": True,
        "planner_version": "manifest_display_v1",
        "generated_at": str(manifest.get("created_at", "") or datetime.now().isoformat()),
        "project": str(manifest.get("project", CAMPAIGN.get_active_project_name() or "") or ""),
        "iteration": int(manifest.get("iteration", CAMPAIGN.get_current_iteration_num() or 1) or 1),
        "master_pool_dir": str(manifest.get("master_pool_dir", "") or ""),
        "source_dir": str(manifest.get("source_dir", "") or ""),
        "selection_mode": str(manifest.get("selection_mode", "") or ""),
        "manifest_only": bool(manifest.get("manifest_only", False)),
        "batch_size": 0,
        "raw_total": selected_count,
        "candidates_total": selected_count,
        "selected_total": len(selected_items),
        "new_to_project_total": int((manifest.get("proposal_summary") or {}).get("new_to_project_total", len(selected_items)) or len(selected_items)),
        "skipped_used": int((manifest.get("proposal_summary") or {}).get("skipped_used", 0) or 0),
        "skipped_duplicate_filenames": int((manifest.get("proposal_summary") or {}).get("skipped_duplicate_filenames", 0) or 0),
        "skipped_duplicate_approved_filenames": int((manifest.get("proposal_summary") or {}).get("skipped_duplicate_approved_filenames", 0) or 0),
        "project_overlap_filenames": int((manifest.get("proposal_summary") or {}).get("project_overlap_filenames", 0) or 0),
        "skipped_invalid_ground_truth": 0,
        "current_balance": {ch: int(current_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
        "selected_balance": {ch: int(selected_hist.get(ch, 0)) for ch in CHAR_ALPHABET},
        "predicted_balance_after": {ch: int(predicted_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
        "selected": selected_items,
    }

def _build_ingest_plan_from_iteration_dir_for_display(
    self,
    iteration_dir: Path,
    current_balance: dict | None = None,
) -> dict:
    if iteration_dir is None:
        return {}
    iteration_dir = Path(iteration_dir)
    if not iteration_dir.exists() or not iteration_dir.is_dir():
        return {}

    planner = CampaignIngestPlanner()
    selected_items = []
    selected_hist = Counter()
    image_paths = sorted(
        (
            image_path
            for image_path in iteration_dir.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )

    master_pool_dir = CAMPAIGN.get_master_pool_dir()
    for image_path in image_paths:
        ground_truth_texts = planner.extract_true_texts_from_filename(image_path.name)
        char_hist = planner.build_char_histogram(ground_truth_texts)
        if not char_hist:
            continue

        selected_hist.update(char_hist)
        try:
            source_path = str(image_path.resolve())
        except Exception:
            source_path = str(image_path.absolute())
        selected_items.append(
            {
                "name": image_path.name,
                "source_path": source_path,
                "source_key": planner.make_source_key(image_path, master_pool_dir=master_pool_dir),
                "target_path": source_path,
                "ground_truth_texts": list(ground_truth_texts),
                "char_histogram": dict(char_hist),
                "score": 0.0,
                "score_details": {},
            }
        )

    if not selected_items:
        return {}

    current_counter = Counter()
    for ch in CHAR_ALPHABET:
        current_counter[ch] = int((current_balance or {}).get(ch, 0) or 0)
    predicted_counter = Counter(current_counter)
    predicted_counter.update(selected_hist)

    return {
        "ok": True,
        "planner_version": "iteration_dir_display_v1",
        "generated_at": datetime.now().isoformat(),
        "project": CAMPAIGN.get_active_project_name() or "",
        "iteration": int(CAMPAIGN.get_current_iteration_num() or 1),
        "master_pool_dir": str(master_pool_dir.resolve()) if master_pool_dir else "",
        "batch_size": 0,
        "raw_total": len(image_paths),
        "candidates_total": len(image_paths),
        "selected_total": len(selected_items),
        "new_to_project_total": len(selected_items),
        "skipped_used": 0,
        "skipped_duplicate_filenames": 0,
        "skipped_duplicate_approved_filenames": 0,
        "project_overlap_filenames": 0,
        "skipped_invalid_ground_truth": len(image_paths) - len(selected_items),
        "current_balance": {ch: int(current_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
        "selected_balance": {ch: int(selected_hist.get(ch, 0)) for ch in CHAR_ALPHABET},
        "predicted_balance_after": {ch: int(predicted_counter.get(ch, 0)) for ch in CHAR_ALPHABET},
        "selected": selected_items,
    }

def _restore_ingest_plan_from_existing_iteration(
    self,
    current_iter: int,
    current_proj: str,
    snapshot: dict | None = None,
) -> bool:
    try:
        iteration_dir = CAMPAIGN.get_iteration_raw_dir(current_iter, current_proj)
    except Exception:
        iteration_dir = None

    try:
        manifest_path = CAMPAIGN.get_ingest_manifest_path(current_iter, current_proj)
    except Exception:
        manifest_path = None
    try:
        plan_path = CAMPAIGN.get_latest_ingest_plan_path(current_proj)
    except Exception:
        plan_path = None

    signature = (
        str(current_proj or "").strip(),
        int(current_iter or 0),
        self._build_cache_token_for_path(iteration_dir),
        self._build_cache_token_for_path(manifest_path),
        self._build_cache_token_for_path(plan_path),
    )
    if (
        getattr(self, "_existing_iteration_ingest_plan_signature", None) == signature
        and isinstance(self.current_ingest_plan, dict)
        and self.current_ingest_plan.get("selected")
    ):
        return True

    current_balance = (snapshot or {}).get("char_balance", {}) if isinstance(snapshot, dict) else {}
    plan = self._normalize_ingest_plan_for_display(
        self._load_latest_ingest_plan_for_current_iteration(),
        current_balance=current_balance,
    )

    if not plan:
        try:
            manifest = CAMPAIGN.load_ingest_manifest(current_iter, current_proj)
        except Exception:
            manifest = {}
        plan = self._build_ingest_plan_from_manifest_for_display(
            manifest,
            current_balance=current_balance,
        )

    if not plan:
        plan = self._build_ingest_plan_from_iteration_dir_for_display(
            iteration_dir,
            current_balance=current_balance,
        )

    if not plan:
        self._existing_iteration_ingest_plan_signature = None
        return False

    plan["project"] = str(current_proj or plan.get("project", "") or "")
    plan["iteration"] = int(current_iter or plan.get("iteration", 1) or 1)
    self.current_ingest_plan = plan
    self._recalculate_current_ingest_plan()
    self._existing_iteration_ingest_plan_signature = signature
    return True

def _recalculate_current_ingest_plan(self):
    if not isinstance(self.current_ingest_plan, dict):
        self.current_ingest_plan = {}
        return

    selected_items = self.current_ingest_plan.get("selected", []) or []
    current_balance = Counter(self.current_ingest_plan.get("current_balance", {}) or {})
    selected_balance = Counter()
    for item in selected_items:
        if not isinstance(item, dict):
            continue
        selected_balance.update(item.get("char_histogram", {}) or {})

    predicted = Counter(current_balance)
    predicted.update(selected_balance)

    self.current_ingest_plan["selected_total"] = len(selected_items)
    self.current_ingest_plan["selected_balance"] = {
        ch: int(selected_balance.get(ch, 0))
        for ch in CHAR_ALPHABET
    }
    self.current_ingest_plan["predicted_balance_after"] = {
        ch: int(predicted.get(ch, 0))
        for ch in CHAR_ALPHABET
    }

def _get_selected_ingest_indices(self) -> list[int]:
    if self.ingest_plan_listbox is None:
        return []
    try:
        indices = []
        for raw_idx in self.ingest_plan_listbox.curselection():
            idx = int(raw_idx)
            if 0 <= idx < len(self.ingest_plan_items):
                indices.append(idx)
        return sorted(set(indices))
    except Exception:
        return []

def _refresh_ingest_logic_text(self):
    if self.ingest_logic_lbl is None:
        return

    plan_count = int(self.current_ingest_plan.get("selected_total", 0) or 0)
    selected_balance = Counter(self.current_ingest_plan.get("selected_balance", {}) or {})
    snapshot_balance = Counter((self.last_ingest_snapshot or {}).get("char_balance", {}) or {})
    rare_chars = [
        ch for ch in CHAR_ALPHABET
        if int(snapshot_balance.get(ch, 0)) > 0
    ]
    rare_chars.sort(key=lambda ch: (int(snapshot_balance.get(ch, 0)), ch))
    rare_preview = ", ".join(
        f"{ch}:{int(snapshot_balance.get(ch, 0))}"
        for ch in rare_chars[:6]
    ) or "brak danych"

    if plan_count > 0:
        text = (
            "Załadowany został aktualny wybrany folder zdjęć dla E1.\n"
            "1. System wczytuje wszystkie poprawne zdjęcia z głównej puli.\n"
            "2. Z nazwy każdego pliku odczytuje tekst tablic i buduje histogram znaków.\n"
            "3. Możesz zatwierdzić zestaw zdjęć jako wejście do iteracji.\n"
            f"4. Najczęstsze znaki w tej chwili: {self._format_histogram_compact(selected_balance, limit=6)}."
        )
    else:
        text = ""

    try:
        self.ingest_logic_lbl.config(text=text)
    except Exception:
        pass

def _refresh_ingest_selection_info(self):
    if self.ingest_selection_lbl is None:
        return

    if not self.current_ingest_plan or not self.ingest_plan_items:
        text = (
            "Po załadowaniu wybranego folderu zdjęć zaznacz jedno albo kilka zdjęć na liście. "
            "Panel pokaże, jaki wpływ będzie miało ich usunięcie z E1."
        )
        try:
            self.ingest_selection_lbl.config(text=text)
        except Exception:
            pass
        return

    selected_indices = self._get_selected_ingest_indices()
    if not selected_indices:
        text = (
            "Zaznacz jedno albo kilka zdjęć na liście. "
            "Możesz używać Ctrl i Shift jak w systemie Windows."
        )
        try:
            self.ingest_selection_lbl.config(text=text)
        except Exception:
            pass
        return

    selected_items = [
        self.ingest_plan_items[idx]
        for idx in selected_indices
        if 0 <= idx < len(self.ingest_plan_items)
    ]
    if not selected_items:
        return

    selected_texts = []
    removed_hist = Counter()
    for item in selected_items:
        for text_value in item.get("ground_truth_texts", []) or []:
            text_value = str(text_value).strip()
            if text_value and text_value not in selected_texts:
                selected_texts.append(text_value)
        removed_hist.update(item.get("char_histogram", {}) or {})

    current_hist = Counter(self.current_ingest_plan.get("selected_balance", {}) or {})
    after_hist = Counter(current_hist)
    for ch, value in removed_hist.items():
        after_hist[ch] = max(0, int(after_hist.get(ch, 0)) - int(value))

    impact_rows = []
    for ch, removed_count in removed_hist.items():
        current_value = int(current_hist.get(ch, 0))
        after_value = int(after_hist.get(ch, 0))
        if removed_count > 0:
            impact_rows.append((str(ch), int(removed_count), current_value, after_value))
    impact_rows.sort(key=lambda entry: (-entry[1], entry[0]))

    zeroed_chars = [
        ch for ch, removed_count, current_value, after_value in impact_rows
        if current_value > 0 and after_value == 0
    ]

    texts_preview = ", ".join(selected_texts[:6])
    if len(selected_texts) > 6:
        texts_preview += f" +{len(selected_texts) - 6} więcej"
    if not texts_preview:
        texts_preview = "brak ground truth"

    impact_preview = ", ".join(
        f"{ch}: {before}->{after}"
        for ch, _removed, before, after in impact_rows[:6]
    ) or "brak danych"

    text = (
        f"Zaznaczono {len(selected_items)} zdjęć.\n"
        f"Tablice z nazw plików: {texts_preview}\n"
        f"Znaki w zaznaczeniu: {self._format_histogram_compact(removed_hist, limit=8)}\n"
        f"Po usunięciu z wybranego folderu zdjęć najbardziej spadną: {impact_preview}"
    )
    if zeroed_chars:
        text += f"\nPo usunięciu całkiem znikną z E1: {', '.join(zeroed_chars[:6])}."
    try:
        self.ingest_selection_lbl.config(text=text)
    except Exception:
        pass

def _refresh_ingest_balance_chart(self):
    self._render_ingest_balance_chart(
        self.ingest_balance_canvas,
        self.ingest_balance_summary_lbl,
        self.current_ingest_plan.get("selected_balance", {}) or {},
        package_images=int(self.current_ingest_plan.get("selected_total", 0) or 0),
        raw_package_images=int(
            self.current_ingest_plan.get(
                "raw_total",
                self.current_ingest_plan.get("selected_total", 0),
            ) or self.current_ingest_plan.get("selected_total", 0) or 0
        ),
    )

def _theme_step1_ingest_panel(self):
    palette = getattr(self.app, "palette", {})
    frame_bg = self._get_step1_ingest_frame_bg()

    try:
        if self.ingest_panel_frame is not None:
            self.ingest_panel_frame.config(
                bg=frame_bg,
                highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c"))
            )
    except Exception:
        pass

    for widget_name, fg_value in (
        ("ingest_header_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_intro_lbl", palette.get("muted", "#c7c7c7")),
        ("ingest_start_title_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_start_summary_lbl", palette.get("muted", "#c7c7c7")),
        ("ingest_insights_hint_lbl", palette.get("muted", "#c7c7c7")),
        ("lbl_ingest_master_title", palette.get("fg", "#f3f3f3")),
        ("lbl_ingest_master_value", palette.get("accent", "#4fc1ff")),
        ("lbl_ingest_batch_title", palette.get("fg", "#f3f3f3")),
        ("ingest_list_title_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_logic_title_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_logic_lbl", palette.get("muted", "#c7c7c7")),
        ("ingest_balance_title_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_balance_summary_lbl", palette.get("muted", "#c7c7c7")),
        ("ingest_selection_title_lbl", palette.get("fg", "#f3f3f3")),
        ("ingest_selection_lbl", palette.get("muted", "#c7c7c7")),
    ):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.config(bg=frame_bg, fg=fg_value)
        except Exception:
            pass

    for widget_name in (
        "ingest_start_shell",
        "ingest_start_panel",
        "ingest_route_shell",
        "ingest_route_body",
        "ingest_start_assets_row",
        "ingest_top_section",
        "ingest_body",
        "ingest_left_col",
        "ingest_right_col",
        "ingest_master_row",
        "ingest_master_value_row",
        "ingest_config_row",
        "ingest_actions_row",
        "ingest_approve_row",
        "ingest_footer_row",
        "ingest_insights_toggle_shell",
        "ingest_insights_shell",
        "ingest_chart_panel",
        "ingest_info_panel",
    ):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.config(bg=frame_bg)
        except Exception:
            pass

    bordered_widgets = ("ingest_route_shell",)
    for widget_name in bordered_widgets:
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            widget.config(
                highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
            )
        except Exception:
            pass

    try:
        if getattr(self, "ingest_insights_shell", None) is not None:
            self.ingest_insights_shell.config(
                bg=frame_bg,
                highlightthickness=0,
                highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c"))
            )
    except Exception:
        pass

    try:
        summary_style = self._get_ingest_summary_style()
        if getattr(self, "ingest_status_shell", None) is not None:
            self.ingest_status_shell.config(
                bg=summary_style["bg"],
                highlightthickness=1,
                highlightbackground=summary_style["border"],
                highlightcolor=summary_style["border"],
            )
        if getattr(self, "ingest_status_panel", None) is not None:
            self.ingest_status_panel.config(bg=summary_style["bg"])
        if getattr(self, "ingest_status_table", None) is not None:
            self.ingest_status_table.config(
                bg=summary_style["border"],
                highlightbackground=summary_style["border"],
                highlightcolor=summary_style["border"],
                highlightthickness=1,
            )
        for idx, lbl in enumerate(getattr(self, "ingest_status_labels", [])):
            lbl.config(
                bg=summary_style["bg"],
                fg=(summary_style["fg"] if idx == 0 else summary_style["muted"]),
                font=("Segoe UI", 9, "normal"),
            )
    except Exception:
        pass

    self._sync_ingest_wraps()
    self._refresh_ingest_balance_chart()

def _populate_ingest_plan_list(self):
    self.ingest_plan_items = list(self.current_ingest_plan.get("selected", []) or [])
    if self.ingest_plan_listbox is None:
        return

    self.ingest_plan_listbox.delete(0, tk.END)
    for index, item in enumerate(self.ingest_plan_items, start=1):
        name = str(item.get("name", "") or "").strip()
        texts = ", ".join(item.get("ground_truth_texts", []) or [])
        left = texts if texts else "brak GT"
        label = (
            f"{index:03d}. "
            f"{shorten(left, width=28, placeholder='...')} | "
            f"{shorten(name, width=70, placeholder='...')}"
        )
        self.ingest_plan_listbox.insert(tk.END, label)

    if self.ingest_plan_items:
        try:
            self.ingest_plan_listbox.selection_clear(0, tk.END)
            self.ingest_plan_listbox.selection_set(0)
            self.ingest_plan_listbox.activate(0)
        except Exception:
            pass

def _get_step1_manifest_context_lightweight(
    self,
    latest_plan_summary: dict | None = None,
    *,
    iter_image_count: int = 0,
    plan_count: int = 0,
) -> dict:
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iteration = 1

    cache_scope = f"step1_manifest_context:iter_{int(current_iteration or 0):03d}"
    try:
        store = self._load_project_view_cache_store()
        cached = dict((store.get("entries", {}) or {}).get(cache_scope, {}) or {})
        payload = cached.get("payload")
        if isinstance(payload, dict) and payload:
            result = dict(payload)
        else:
            result = {}
    except Exception:
        result = {}

    summary = latest_plan_summary if isinstance(latest_plan_summary, dict) else {}
    try:
        summary_matches = (
            int(summary.get("iteration", 0) or 0) == int(current_iteration or 1)
            and (
                not str(summary.get("project", "") or "").strip()
                or str(summary.get("project", "") or "").strip()
                == str(CAMPAIGN.get_active_project_name() or "").strip()
            )
        )
    except Exception:
        summary_matches = False

    if summary_matches:
        source_total = int(summary.get("raw_total", 0) or summary.get("selected_total", 0) or 0)
        current_count = int(
            summary.get("selected_total", 0)
            or iter_image_count
            or plan_count
            or source_total
            or 0
        )
        result.update(
            iteration=int(current_iteration),
            selection_mode=str(summary.get("selection_mode", result.get("selection_mode", "")) or ""),
            selected_count=int(current_count or 0),
            source_total=int(source_total or 0),
            skipped_duplicate_filenames=int(summary.get("skipped_duplicate_filenames", result.get("skipped_duplicate_filenames", 0)) or 0),
            skipped_duplicate_approved=int(summary.get("skipped_duplicate_approved_filenames", result.get("skipped_duplicate_approved", 0)) or 0),
            project_overlap_filenames=int(summary.get("project_overlap_filenames", result.get("project_overlap_filenames", 0)) or 0),
            project_pool_total_before=int(summary.get("project_pool_total_before_iteration", result.get("project_pool_total_before", 0)) or 0),
            project_pool_total_after=int(summary.get("project_pool_total_after_iteration", result.get("project_pool_total_after", 0)) or 0),
            project_pool_total=int(summary.get("project_pool_total_after_iteration", result.get("project_pool_total", 0)) or 0),
            current_iteration_package_count=int(current_count or 0),
            new_to_project_count=int(summary.get("new_to_project_total", result.get("new_to_project_count", current_count)) or 0),
        )
    elif not result:
        result = {
            "iteration": int(current_iteration),
            "selection_mode": "",
            "selected_count": int(iter_image_count or plan_count or 0),
            "source_total": int(iter_image_count or plan_count or 0),
            "skipped_duplicate_filenames": 0,
            "skipped_duplicate_approved": 0,
            "project_overlap_filenames": 0,
            "project_pool_total": int(iter_image_count or plan_count or 0),
            "project_pool_total_before": 0,
            "project_pool_total_after": int(iter_image_count or plan_count or 0),
            "current_iteration_package_count": int(iter_image_count or plan_count or 0),
            "new_to_project_count": int(iter_image_count or plan_count or 0),
        }

    try:
        approved_stats = CAMPAIGN.get_plate_approved_set_stats() or {}
    except Exception:
        approved_stats = {}
    result["approved_images"] = int(approved_stats.get("images", result.get("approved_images", 0)) or 0)
    result["approved_plates"] = int(approved_stats.get("plates", result.get("approved_plates", 0)) or 0)
    return result

def _refresh_ingest_panel(self, snapshot_override: dict = None):
    has_project = bool(CAMPAIGN.get_active_project_name())
    lightweight_open = bool(getattr(self, "_project_open_lightweight_refresh", False))
    step1_status = CAMPAIGN.get_step1_status() if has_project else "pending"
    step1_approved = step1_status == "approved"
    master_pool = CAMPAIGN.get_master_pool_dir()
    master_pool_text = str(master_pool) if master_pool else "Brak ustawionej głównej puli zdjęć"
    self.ingest_master_pool_var.set(master_pool_text)
    master_pool_exists = bool(master_pool and master_pool.exists() and master_pool.is_dir())
    frame_bg = self._get_step1_ingest_frame_bg()
    try:
        current_step_value = int(CAMPAIGN.get_current_step() or 1)
    except Exception:
        current_step_value = 1
    skip_operational_step1_refresh = bool(lightweight_open and has_project and current_step_value != 1)
    if not skip_operational_step1_refresh:
        self._theme_step1_ingest_panel()
        self._refresh_project_start_panel()
        try:
            self._refresh_wizard_transition_graph()
        except Exception as e:
            logger.debug(f"Nie udało się odświeżyć wyboru toru E1: {e}")
        try:
            self._refresh_project_start_assets_table_theme(frame_bg)
        except Exception:
            pass
    project_start_mode = self._get_project_start_mode()

    snapshot = snapshot_override or CAMPAIGN.load_ingest_balance_snapshot()
    if has_project and not snapshot and not lightweight_open:
        snapshot = CAMPAIGN.refresh_ingest_balance_snapshot()
    self.last_ingest_snapshot = snapshot or {}

    iter_image_count = self._get_iteration_image_count() if has_project else 0
    try:
        latest_plan_summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
    except Exception:
        latest_plan_summary = {}
    summary_matches_current = False
    if has_project and latest_plan_summary:
        try:
            summary_matches_current = (
                int(latest_plan_summary.get("iteration", 0) or 0) == int(CAMPAIGN.get_current_iteration_num() or 1)
                and str(latest_plan_summary.get("project", "") or "").strip() == str(CAMPAIGN.get_active_project_name() or "").strip()
            )
        except Exception:
            summary_matches_current = False
    if summary_matches_current:
        master_pool_image_count = int(
            latest_plan_summary.get("raw_total", 0)
            or latest_plan_summary.get("selected_total", 0)
            or 0
        )
    else:
        master_pool_image_count = 0

    if has_project and lightweight_open:
        self.current_ingest_plan = {}
        self._existing_iteration_ingest_plan_signature = None
        self.ingest_plan_items = []
        if self.ingest_plan_listbox is not None:
            try:
                self.ingest_plan_listbox.delete(0, tk.END)
                self.ingest_plan_listbox.insert(
                    tk.END,
                    "Lista zdjęć zostanie załadowana po wejściu w zasoby E1.",
                )
                self.ingest_plan_listbox.itemconfig(0, foreground="#888888")
            except Exception:
                pass
    elif has_project:
        current_iter = CAMPAIGN.get_current_iteration_num()
        current_proj = CAMPAIGN.get_active_project_name()
        if iter_image_count > 0:
            restored = self._restore_ingest_plan_from_existing_iteration(
                current_iter=current_iter,
                current_proj=current_proj,
                snapshot=snapshot,
            )
            if not restored:
                self.current_ingest_plan = {}
        elif (
            isinstance(self.current_ingest_plan, dict)
            and int(self.current_ingest_plan.get("iteration", 0) or 0) == current_iter
            and str(self.current_ingest_plan.get("project", "") or "").strip() == str(current_proj or "").strip()
            and self.current_ingest_plan.get("selected") is not None
        ):
            self._recalculate_current_ingest_plan()
            self._existing_iteration_ingest_plan_signature = None
        else:
            self.current_ingest_plan = {}
            self._existing_iteration_ingest_plan_signature = None
        self._populate_ingest_plan_list()
    else:
        self.current_ingest_plan = {}
        self._existing_iteration_ingest_plan_signature = None
        self._populate_ingest_plan_list()

    plan_count = int(self.current_ingest_plan.get("selected_total", 0) or 0)
    raw_plan_count = int(self.current_ingest_plan.get("raw_total", plan_count) or plan_count)
    if has_project and lightweight_open:
        step1_context = self._get_step1_manifest_context_lightweight(
            latest_plan_summary,
            iter_image_count=iter_image_count,
            plan_count=plan_count,
        )
    else:
        manifest = self._load_ingest_manifest_cached() if has_project else {}
        step1_context = self._get_step1_manifest_context(manifest if isinstance(manifest, dict) else None)
    project_pool_total = int(step1_context.get("project_pool_total_after", step1_context.get("project_pool_total", 0)) or 0)
    source_total = int(step1_context.get("source_total", 0) or 0)
    approved_images = int(step1_context.get("approved_images", 0) or 0)
    approved_plates = int(step1_context.get("approved_plates", 0) or 0)
    current_iteration_package = int(
        step1_context.get("current_iteration_package_count", 0)
        or iter_image_count
        or plan_count
        or 0
    )
    new_to_project_count = int(step1_context.get("new_to_project_count", 0) or 0)
    project_overlap_count = int(step1_context.get("project_overlap_filenames", 0) or 0)
    skipped_approved_count = int(step1_context.get("skipped_duplicate_approved", 0) or 0)

    if not step1_approved and plan_count > 0:
        project_pool_before = int(step1_context.get("project_pool_total_before", 0) or 0)
        new_to_project_count = int(
            self.current_ingest_plan.get(
                "new_to_project_total",
                plan_count,
            ) or 0
        )
        project_pool_total = max(project_pool_total, project_pool_before + new_to_project_count)
        current_iteration_package = max(current_iteration_package, plan_count)
        project_overlap_count = int(
            self.current_ingest_plan.get("project_overlap_filenames", self.current_ingest_plan.get("skipped_duplicate_filenames", 0)) or 0
        )
        skipped_approved_count = int(
            self.current_ingest_plan.get("skipped_duplicate_approved_filenames", 0) or 0
        )
        source_total = max(source_total, raw_plan_count)

    if project_pool_total <= 0:
        project_pool_total = max(int(source_total or 0), int(raw_plan_count or 0), int(current_iteration_package or 0))

    summary_rows = [
        ("Pudełko: tablice zatwierdzone", f"{approved_images} zdjęć / {approved_plates} tablic"),
        ("Zdjęcia tej iteracji", f"{current_iteration_package} zdjęć"),
        ("Nowe względem projektu", f"{new_to_project_count} zdjęć"),
        ("Pula projektu (informacyjnie)", f"{project_pool_total} zdjęć"),
    ]
    if source_total > 0:
        summary_rows.insert(0, ("Źródło zdjęć", f"{source_total} zdjęć"))
    if project_overlap_count > 0:
        summary_rows.append(("Już wcześniej w projekcie", f"{project_overlap_count} zdjęć"))
    if skipped_approved_count > 0:
        summary_rows.append(("W tym już zatwierdzone", f"{skipped_approved_count} zdjęć"))

    self._set_ingest_status_lines(summary_rows, "")

    char_material_ready_without_new_images = False
    try:
        # Przy lekkim otwieraniu projektu nie uruchamiamy preflightu toru znakow
        # dla E1, jesli projekt stoi juz na dalszym kroku. Dla duzych projektow
        # taki preflight potrafi kaskadowo dotknac Z2/Z3 i niepotrzebnie spowalnia
        # samo pokazanie grafu kampanii.
        should_check_char_preflight = bool(
            self._get_iteration_target() == "char"
            and not step1_approved
            and not skip_operational_step1_refresh
        )
        if should_check_char_preflight:
            char_state = dict(self._get_step1_char_route_preflight_state() or {})
            char_material_ready_without_new_images = bool(char_state.get("material_ready"))
    except Exception:
        char_material_ready_without_new_images = False
    enable_apply = bool(
        (plan_count > 0 and iter_image_count == 0)
        or (iter_image_count > 0 and not step1_approved)
        or char_material_ready_without_new_images
    )
    for widget, enabled in (
        (getattr(self, "btn_ingest_master_analysis", None), has_project),
        (getattr(self, "btn_apply_ingest_plan", None), enable_apply),
    ):
        if widget is None:
            continue
        try:
            widget.config(state="normal" if enabled else "disabled")
        except Exception:
            pass

    self._refresh_ingest_logic_text()
    self._refresh_ingest_selection_info()
    self._refresh_ingest_balance_chart()

def _mark_ingest_plan_generation_pending(self, master_pool: Path) -> None:
    self.current_ingest_plan = {}
    self.ingest_plan_items = []
    if getattr(self, "ingest_plan_listbox", None) is not None:
        try:
            self.ingest_plan_listbox.delete(0, tk.END)
            self.ingest_plan_listbox.insert(tk.END, "Analizuję wybrany katalog zdjęć...")
            self.ingest_plan_listbox.insert(tk.END, "Okno pozostaje aktywne; lista pojawi się po zakończeniu analizy.")
            self.ingest_plan_listbox.itemconfig(0, foreground="#4f8de3")
            self.ingest_plan_listbox.itemconfig(1, foreground="#888888")
        except Exception:
            pass
    for widget in (
        getattr(self, "btn_apply_ingest_plan", None),
        getattr(self, "btn_ingest_master_analysis", None),
    ):
        if widget is None:
            continue
        try:
            widget.config(state="disabled")
        except Exception:
            pass
    try:
        self.ingest_logic_lbl.config(
            text=(
                "Analizuję wybrany katalog zdjęć w tle. "
                "Po zakończeniu pojawi się lista obrazów gotowych do użycia w tej iteracji."
            )
        )
    except Exception:
        pass
    try:
        self.app.update_status(f"Analizuję katalog zdjęć: {master_pool}", "info")
    except Exception:
        pass
    try:
        self.frame.update_idletasks()
    except Exception:
        pass


def _show_ingest_plan_progress_dialog(
    self,
    master_pool: Path | str | None = None,
    parent=None,
    *,
    title: str = "Przygotowanie zbioru obrazów O",
    eyebrow: str = "DODAWANIE OBRAZÓW",
    initial_detail: str = "Weryfikuję wskazany katalog i przygotowuję analizę.",
) -> None:
    existing = getattr(self, "_ingest_plan_progress_dialog", None)
    try:
        if existing is not None and existing.winfo_exists():
            try:
                getattr(self, "_ingest_plan_progress_title_var", None).set(
                    campaign_ui_helpers._repair_polish_text(str(title or "").strip())
                )
                getattr(self, "_ingest_plan_progress_detail_var", None).set(
                    campaign_ui_helpers._repair_polish_text(str(initial_detail or "").strip())
                )
                getattr(self, "_ingest_plan_progress_step_var", None).set(
                    campaign_ui_helpers._repair_polish_text("Start przetwarzania...")
                )
                getattr(self, "_ingest_plan_progress_var", None).set(4.0)
            except Exception:
                pass
            existing.lift()
            return
    except Exception:
        pass

    owner = parent or getattr(self, "frame", None)
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    bg = palette.get("panel", "#102016")
    panel = blend_hex_colors(bg, palette.get("accent", "#8fbf79"), 0.045)
    border = blend_hex_colors(palette.get("accent", "#8fbf79"), palette.get("panel_border", "#314233"), 0.28)
    fg = palette.get("fg", "#e8f2dc")
    muted = palette.get("muted", "#bac8ae")
    accent = palette.get("accent", "#8fbf79")

    try:
        dialog = tk.Toplevel(owner)
    except Exception:
        return
    dialog.withdraw()
    dialog.title(campaign_ui_helpers._repair_polish_text("Przetwarzanie obrazów"))
    dialog.configure(bg=bg)
    dialog.resizable(False, False)
    try:
        dialog.overrideredirect(True)
    except Exception:
        pass
    try:
        dialog.attributes("-toolwindow", True)
    except Exception:
        pass
    try:
        dialog.transient(owner)
    except Exception:
        pass
    try:
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    except Exception:
        pass

    shell = tk.Frame(
        dialog,
        bg=panel,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_columnconfigure(1, weight=1)
    tk.Frame(shell, width=3, bg=accent, bd=0, highlightthickness=0).grid(row=0, column=0, sticky="nsw")
    content = tk.Frame(shell, bg=panel, bd=0, highlightthickness=0, padx=24, pady=20)
    content.grid(row=0, column=1, sticky="nsew")
    content.grid_columnconfigure(0, weight=1)

    title_var = tk.StringVar(master=dialog, value=campaign_ui_helpers._repair_polish_text(str(title or "").strip()))
    detail_var = tk.StringVar(master=dialog, value=campaign_ui_helpers._repair_polish_text(str(initial_detail or "").strip()))
    step_var = tk.StringVar(master=dialog, value=campaign_ui_helpers._repair_polish_text("Start przetwarzania..."))
    progress_var = tk.DoubleVar(master=dialog, value=4.0)

    tk.Label(
        content,
        text=campaign_ui_helpers._repair_polish_text(str(eyebrow or "").strip() or "DODAWANIE OBRAZÓW"),
        bg=panel,
        fg=blend_hex_colors(accent, fg, 0.10),
        font=("Segoe UI", 8, "bold"),
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    tk.Label(
        content,
        textvariable=title_var,
        bg=panel,
        fg=fg,
        font=("Segoe UI Semibold", 15),
        anchor=tk.W,
    ).grid(row=1, column=0, sticky="ew", pady=(7, 0))
    tk.Label(
        content,
        textvariable=detail_var,
        bg=panel,
        fg=fg,
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=430,
    ).grid(row=2, column=0, sticky="ew", pady=(10, 0))
    ttk.Progressbar(
        content,
        mode="determinate",
        maximum=100.0,
        variable=progress_var,
        style="Horizontal.TProgressbar",
    ).grid(row=3, column=0, sticky="ew", pady=(16, 0))
    tk.Label(
        content,
        textvariable=step_var,
        bg=panel,
        fg=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=430,
    ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

    width = 520
    height = 190
    dialog.geometry(f"{width}x{height}")
    center_dialog = getattr(getattr(self, "app", None), "_center_dialog_window", None)
    if callable(center_dialog):
        try:
            center_dialog(dialog, parent=owner, width=width, height=height)
        except Exception:
            pass
    try:
        dialog.deiconify()
        dialog.lift()
        dialog.update_idletasks()
    except Exception:
        pass

    self._ingest_plan_progress_dialog = dialog
    self._ingest_plan_progress_var = progress_var
    self._ingest_plan_progress_title_var = title_var
    self._ingest_plan_progress_detail_var = detail_var
    self._ingest_plan_progress_step_var = step_var
    try:
        if master_pool:
            _update_ingest_plan_progress_dialog(
                self,
                4,
                "Weryfikuję katalog obrazów.",
                f"Źródło: {Path(master_pool).name or master_pool}",
            )
    except Exception:
        pass


def _update_ingest_plan_progress_dialog(
    self,
    progress: float | int | None,
    message: str = "",
    detail: str = "",
) -> None:
    dialog = getattr(self, "_ingest_plan_progress_dialog", None)
    try:
        if dialog is None or not dialog.winfo_exists():
            return
    except Exception:
        return

    try:
        value = max(0.0, min(100.0, float(progress if progress is not None else 0.0)))
    except Exception:
        value = 0.0
    try:
        getattr(self, "_ingest_plan_progress_var", None).set(value)
    except Exception:
        pass
    try:
        if message:
            getattr(self, "_ingest_plan_progress_detail_var", None).set(
                campaign_ui_helpers._repair_polish_text(str(message))
            )
    except Exception:
        pass
    try:
        visible_value = int(round(value))
        step_text = str(detail or "").strip()
        if step_text:
            step_text = f"{visible_value}% · {step_text}"
        else:
            step_text = f"{visible_value}%"
        getattr(self, "_ingest_plan_progress_step_var", None).set(
            campaign_ui_helpers._repair_polish_text(step_text)
        )
    except Exception:
        pass
    try:
        dialog.lift()
        dialog.update_idletasks()
    except Exception:
        pass


def _hide_ingest_plan_progress_dialog(self, *, delay_ms: int = 0) -> None:
    target_dialog = getattr(self, "_ingest_plan_progress_dialog", None)

    def _destroy() -> None:
        dialog = getattr(self, "_ingest_plan_progress_dialog", None)
        if target_dialog is not None and dialog is not target_dialog:
            return
        if dialog is not None:
            try:
                dialog.destroy()
            except Exception:
                pass
        self._ingest_plan_progress_dialog = None
        self._ingest_plan_progress_var = None
        self._ingest_plan_progress_title_var = None
        self._ingest_plan_progress_detail_var = None
        self._ingest_plan_progress_step_var = None

    if int(delay_ms or 0) > 0:
        try:
            self.frame.after(int(delay_ms), _destroy)
            return
        except Exception:
            pass
    _destroy()


def _same_ingest_source_path(left, right) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return str(left or "").strip().lower() == str(right or "").strip().lower()


def _finish_generated_ingest_plan(self, plan: dict, snapshot: dict | None = None) -> None:
    if not isinstance(plan, dict) or not plan.get("ok", False):
        messagebox.showwarning(
            "Brak wybranego folderu zdjęć E1",
            str(plan.get("error", "Nie udało się załadować wybranego folderu zdjęć E1."))
            if isinstance(plan, dict)
            else "Nie udało się załadować wybranego folderu zdjęć E1.",
        )
        return

    self.current_ingest_plan = plan
    self._recalculate_current_ingest_plan()
    selected_total = int(plan.get("selected_total", 0) or 0)
    raw_total = int(plan.get("raw_total", selected_total) or selected_total)
    skipped_invalid = int(plan.get("skipped_invalid_ground_truth", 0) or 0)
    skipped_duplicates = int(plan.get("skipped_duplicate_filenames", plan.get("skipped_used", 0)) or 0)
    try:
        CAMPAIGN.save_latest_ingest_plan(self.current_ingest_plan)
    except Exception:
        pass

    try:
        self._refresh_project_start_assets_table_theme(self._get_step1_ingest_frame_bg())
    except Exception:
        pass
    self._refresh_ingest_panel(snapshot_override=snapshot)

    if selected_total <= 0:
        if skipped_duplicates > 0 and skipped_invalid <= 0:
            messagebox.showwarning(
                "Brak nowych zdjęć do iteracji",
                (
                    f"Wybrany folder zawiera {raw_total} zdjęć, ale wszystkie zostały odrzucone jako duble po nazwie.\n\n"
                    "Ten zestaw zdjęć nie wnosi nowych obrazów do projektu."
                ),
            )
            return
        messagebox.showwarning(
            "Brak poprawnych pozycji w wybranym folderze zdjęć",
            (
                "Nie znaleziono zdjęć z poprawnym ground truth w nazwie pliku.\n"
                "Sprawdź nazewnictwo plików w głównej puli."
            ),
        )
        return

    try:
        status_text = f"Załadowano wybrany folder zdjęć E1: {raw_total} zdjęć w folderze."
        if raw_total != selected_total:
            status_text += f" Do planu E1 weszło {selected_total}."
        if skipped_duplicates > 0:
            status_text += f" Pominięto {skipped_duplicates} dubli po nazwie."
        if skipped_invalid > 0:
            status_text += f" Pominięto {skipped_invalid} plików bez poprawnego GT w nazwie."
        self.app.update_status(status_text, "info")
    except Exception:
        pass


def _generate_ingest_plan(self, *, async_mode: bool = True, parent=None):
    if not CAMPAIGN.get_active_project_name():
        return

    master_pool = CAMPAIGN.get_master_pool_dir()
    if master_pool is None:
        messagebox.showwarning("Brak wybranego folderu zdjęć", "Najpierw wskaż główną pulę zdjęć.")
        return
    if not master_pool.exists() or not master_pool.is_dir():
        messagebox.showwarning("Brak wybranego folderu zdjęć", f"Katalog nie istnieje:\n{master_pool}")
        return

    if async_mode:
        try:
            project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
            iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
        except Exception:
            project_name = ""
            iteration_num = 1
        try:
            master_pool = Path(master_pool)
            master_token_path = str(master_pool.resolve())
        except Exception:
            master_token_path = str(master_pool)
        token = (project_name, int(iteration_num or 1), master_token_path, perf_counter())
        self._ingest_plan_generation_token = token
        _show_ingest_plan_progress_dialog(self, master_pool, parent=parent)
        _update_ingest_plan_progress_dialog(
            self,
            10,
            "Przygotowuję analizę wybranego zbioru obrazów.",
            "Sprawdzam stan projektu przed skanowaniem katalogu.",
        )
        _mark_ingest_plan_generation_pending(self, master_pool)

        last_progress_emit = 0.0

        def _emit_progress(
            progress: float,
            message: str = "",
            *,
            detail: str = "",
            force: bool = False,
        ) -> None:
            nonlocal last_progress_emit
            now = perf_counter()
            if not force and now - last_progress_emit < 0.12:
                return
            last_progress_emit = now

            def _apply() -> None:
                if getattr(self, "_ingest_plan_generation_token", None) != token:
                    return
                _update_ingest_plan_progress_dialog(self, progress, message, detail)

            try:
                self.frame.after(0, _apply)
            except Exception:
                pass

        def _worker() -> None:
            started = perf_counter()
            snapshot_result: dict = {}
            plan_result: dict = {}
            error_text = ""
            try:
                _emit_progress(
                    18,
                    "Liczenie dotychczasowego bilansu projektu.",
                    detail="Odczytuję istniejące obrazy i rozkład znaków.",
                    force=True,
                )
                snapshot_result = CAMPAIGN.refresh_ingest_balance_snapshot(project_name) or {}
                _emit_progress(
                    26,
                    "Skanowanie nowego zbioru obrazów.",
                    detail="Szukam plików graficznych w wybranym katalogu.",
                    force=True,
                )
                plan_result = self._build_main_pack_plan(
                    master_pool_dir=master_pool,
                    current_balance=(snapshot_result or {}).get("char_balance", {}),
                    progress_callback=_emit_progress,
                )
                _emit_progress(
                    94,
                    "Finalizuję plan wejścia E1.",
                    detail="Zapisuję podsumowanie i odświeżam zasoby bramki.",
                    force=True,
                )
            except Exception as exc:
                error_text = str(exc)
                try:
                    logger.exception("Nie udało się przygotować planu E1 w tle")
                except Exception:
                    pass

            elapsed_ms = int((perf_counter() - started) * 1000)

            def _finish() -> None:
                if getattr(self, "_ingest_plan_generation_token", None) != token:
                    return
                self._ingest_plan_generation_token = None
                try:
                    active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
                    active_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
                    active_pool = CAMPAIGN.get_master_pool_dir()
                except Exception:
                    active_project = ""
                    active_iter = 0
                    active_pool = None
                if (
                    active_project != project_name
                    or int(active_iter or 0) != int(iteration_num or 1)
                    or not _same_ingest_source_path(active_pool, master_pool)
                ):
                    _hide_ingest_plan_progress_dialog(self)
                    return
                try:
                    logger.info(
                        "[E1 PERF] generate_ingest_plan_async total=%sms raw=%s selected=%s source=%s",
                        elapsed_ms,
                        int((plan_result or {}).get("raw_total", 0) or 0),
                        int((plan_result or {}).get("selected_total", 0) or 0),
                        master_pool,
                    )
                except Exception:
                    pass
                if error_text:
                    _hide_ingest_plan_progress_dialog(self)
                    try:
                        self._refresh_ingest_panel(snapshot_override=snapshot_result)
                    except Exception:
                        pass
                    messagebox.showerror("Błąd ładowania wybranego folderu zdjęć E1", error_text)
                    return
                selected_total = int((plan_result or {}).get("selected_total", 0) or 0)
                if selected_total <= 0:
                    _hide_ingest_plan_progress_dialog(self)
                    _finish_generated_ingest_plan(self, plan_result, snapshot_result)
                    return
                _finish_generated_ingest_plan(self, plan_result, snapshot_result)
                _update_ingest_plan_progress_dialog(
                    self,
                    100,
                    "Zbiór obrazów został przeanalizowany.",
                    (
                        f"Gotowe: {int((plan_result or {}).get('selected_total', 0) or 0)} "
                        f"z {int((plan_result or {}).get('raw_total', 0) or 0)} obrazów trafi do kontroli tej iteracji."
                    ),
                )
                _hide_ingest_plan_progress_dialog(self, delay_ms=650)

            try:
                self.frame.after(0, _finish)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="campaign-e1-ingest-plan").start()
        return

    _show_ingest_plan_progress_dialog(self, master_pool, parent=parent)
    _update_ingest_plan_progress_dialog(
        self,
        18,
        "Liczenie dotychczasowego bilansu projektu.",
        "Odczytuję istniejące obrazy i rozkład znaków.",
    )
    snapshot = CAMPAIGN.refresh_ingest_balance_snapshot()

    try:
        plan = self._build_main_pack_plan(
            master_pool_dir=master_pool,
            current_balance=(snapshot or {}).get("char_balance", {}),
            progress_callback=lambda value, message="", **kwargs: _update_ingest_plan_progress_dialog(
                self,
                value,
                message,
                str(kwargs.get("detail", "") or ""),
            ),
        )
    except Exception as e:
        _hide_ingest_plan_progress_dialog(self)
        messagebox.showerror("Błąd ładowania wybranego folderu zdjęć E1", str(e))
        return

    if int((plan or {}).get("selected_total", 0) or 0) <= 0:
        _hide_ingest_plan_progress_dialog(self)
    else:
        _update_ingest_plan_progress_dialog(self, 100, "Zbiór obrazów został przeanalizowany.", "Gotowe.")
        _hide_ingest_plan_progress_dialog(self, delay_ms=450)
    _finish_generated_ingest_plan(self, plan, snapshot)

def _should_reuse_step1_source_despite_duplicate_plan(self, plan: dict) -> bool:
    if not isinstance(plan, dict) or not plan.get("ok", False):
        return False
    try:
        current_iter = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iter = 1
    if current_iter <= 1:
        return False
    try:
        raw_total = int(plan.get("raw_total", 0) or 0)
        selected_total = int(plan.get("selected_total", 0) or 0)
        project_overlap = int(plan.get("project_overlap_filenames", 0) or 0)
        pending_overlap = int(plan.get("pending_iteration_overlap_filenames", 0) or 0)
        invalid_gt = int(plan.get("skipped_invalid_ground_truth", 0) or 0)
    except Exception:
        return False
    if raw_total <= 0 or selected_total > 0:
        return False
    if invalid_gt > 0:
        return False
    return bool(project_overlap + pending_overlap >= raw_total)

def _build_step1_source_reuse_plan(self, base_plan: dict | None = None) -> dict:
    master_pool = CAMPAIGN.get_master_pool_dir()
    if master_pool is None:
        return {}
    try:
        master_pool = Path(master_pool)
    except Exception:
        return {}
    if not master_pool.exists() or not master_pool.is_dir():
        return {}

    image_paths = sorted(
        (
            image_path
            for image_path in master_pool.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
        ),
        key=lambda p: p.as_posix().lower(),
    )
    if not image_paths:
        return {}

    planner = CampaignIngestPlanner()
    selected_hist = Counter()
    selected_items = []
    skipped_invalid_gt = 0
    for image_path in image_paths:
        gt_texts = planner.extract_true_texts_from_filename(image_path.name)
        char_hist = planner.build_char_histogram(gt_texts)
        if not gt_texts or not char_hist:
            skipped_invalid_gt += 1
        selected_hist.update(char_hist)
        try:
            source_path = str(image_path.resolve())
        except Exception:
            source_path = str(image_path.absolute())
        selected_items.append(
            {
                "name": image_path.name,
                "source_path": source_path,
                "source_key": planner.make_source_key(image_path, master_pool_dir=master_pool),
                "ground_truth_texts": list(gt_texts or []),
                "char_histogram": dict(char_hist or {}),
                "score": 0.0,
                "score_details": {"source_reuse": True},
            }
        )

    base_plan = dict(base_plan or {})
    raw_total = int(len(image_paths))
    project_overlap = int(base_plan.get("project_overlap_filenames", raw_total) or 0)
    pending_overlap = int(base_plan.get("pending_iteration_overlap_filenames", 0) or 0)
    source_new_total = max(0, raw_total - project_overlap)

    return {
        "ok": True,
        "planner_version": "source_reuse_v1",
        "generated_at": datetime.now().isoformat(),
        "project": CAMPAIGN.get_active_project_name() or "",
        "iteration": int(CAMPAIGN.get_current_iteration_num() or 1),
        "master_pool_dir": str(master_pool.resolve()),
        "batch_size": 0,
        "raw_total": raw_total,
        "candidates_total": raw_total,
        "selected_total": len(selected_items),
        "new_to_project_total": source_new_total,
        "source_new_to_project_total": source_new_total,
        "skipped_used": 0,
        "skipped_duplicate_filenames": 0,
        "skipped_duplicate_approved_filenames": int(base_plan.get("skipped_duplicate_approved_filenames", 0) or 0),
        "project_overlap_filenames": project_overlap,
        "pending_iteration_overlap_filenames": pending_overlap,
        "project_pool_total_before_iteration": int(base_plan.get("project_pool_total_before_iteration", project_overlap) or project_overlap),
        "project_pool_total_after_iteration": int(base_plan.get("project_pool_total_after_iteration", project_overlap + source_new_total) or (project_overlap + source_new_total)),
        "skipped_invalid_ground_truth": skipped_invalid_gt,
        "source_reuse": True,
        "current_balance": dict(base_plan.get("current_balance", {}) or {}),
        "selected_balance": {ch: int(selected_hist.get(ch, 0)) for ch in CHAR_ALPHABET},
        "predicted_balance_after": {ch: int(selected_hist.get(ch, 0)) for ch in CHAR_ALPHABET},
        "selected": selected_items,
    }

def _ensure_current_ingest_plan_from_master_pool(self) -> bool:
    if (
        isinstance(self.current_ingest_plan, dict)
        and int(self.current_ingest_plan.get("selected_total", 0) or 0) > 0
        and self.current_ingest_plan.get("selected")
    ):
        return True

    plan = self._load_latest_ingest_plan_for_current_iteration()
    if isinstance(plan, dict) and int(plan.get("selected_total", 0) or 0) > 0 and plan.get("selected"):
        self.current_ingest_plan = dict(plan)
        self._recalculate_current_ingest_plan()
        return True

    manifest = self._load_ingest_manifest_cached()
    manifest_plan = self._build_ingest_plan_from_manifest_for_display(
        manifest,
        current_balance=(self.last_ingest_snapshot or {}).get("char_balance", {}),
    )
    if (
        isinstance(manifest_plan, dict)
        and int(manifest_plan.get("selected_total", 0) or 0) > 0
        and manifest_plan.get("selected")
    ):
        self.current_ingest_plan = manifest_plan
        self._recalculate_current_ingest_plan()
        return True

    master_pool = CAMPAIGN.get_master_pool_dir()
    if master_pool is None:
        return False
    try:
        master_pool = Path(master_pool)
    except Exception:
        return False
    if not master_pool.exists() or not master_pool.is_dir():
        return False
    if self._count_images_in_dir(master_pool, recursive=True) <= 0:
        return False

    snapshot = CAMPAIGN.refresh_ingest_balance_snapshot()
    try:
        plan = self._build_main_pack_plan(
            master_pool_dir=master_pool,
            current_balance=(snapshot or {}).get("char_balance", {}),
        )
    except Exception as e:
        logger.debug(f"Nie udało się odbudować planu E1 z wybranego katalogu: {e}")
        return False

    if not isinstance(plan, dict) or not plan.get("ok", False):
        return False
    if int(plan.get("selected_total", 0) or 0) <= 0 or not plan.get("selected"):
        if self._should_reuse_step1_source_despite_duplicate_plan(plan):
            reuse_plan = self._build_step1_source_reuse_plan(plan)
            if int(reuse_plan.get("selected_total", 0) or 0) > 0 and reuse_plan.get("selected"):
                self.current_ingest_plan = reuse_plan
                self._recalculate_current_ingest_plan()
                try:
                    CAMPAIGN.save_latest_ingest_plan(self.current_ingest_plan)
                except Exception:
                    pass
                return True
        return False

    self.current_ingest_plan = plan
    self._recalculate_current_ingest_plan()
    try:
        CAMPAIGN.save_latest_ingest_plan(self.current_ingest_plan)
    except Exception:
        pass
    return True

def _ensure_step1_iteration_target_selected(self) -> bool:
    if self._get_iteration_target() in {"plate", "char"}:
        return True

    try:
        self.step1_panel_expanded = True
        self.request_wizard_stage_focus(step_num=1)
        self._refresh_active_project_wizard_only()
    except Exception:
        pass

    message = (
        "Zanim zatwierdzisz E1, wybierz tor tej iteracji w panelu E1.\n\n"
        "E2 jest etapem pracy na tablicach: w torze tablic prowadzi dalej do treningu tablic, "
        "a w torze znaków przygotowuje źródło dla E3/Z3."
    )
    try:
        self.app.themed_info(
            "Wybierz tor iteracji E1",
            message,
            parent=self.frame,
            tone="warning",
        )
    except Exception:
        messagebox.showwarning("Wybierz tor iteracji E1", message)
    return False

def _continue_to_step3_after_step1_if_char_ready(self) -> bool:
    if self._get_iteration_target() != "char":
        return False
    if str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved":
        return False
    if int(CAMPAIGN.get_current_step() or 1) < 2:
        return False

    preflight = self._get_step1_char_route_preflight_state()
    if not bool(preflight.get("material_ready")):
        return False

    ready_source = self._get_char_route_ready_source()
    if not ready_source:
        return False

    logger.debug(
        "[CampaignTab] Tor znaków ma gotowe źródło tablic po E1, ale E2 nie jest zatwierdzane automatycznie. "
        "Użytkownik musi jawnie użyć badge'a E2."
    )
    return False

def _format_step1_adopted_annotations_z2_scope_notice(self, selected_source_files) -> str:
    try:
        approved_names = set(self._get_project_start_approved_normalized_image_names() or set())
    except Exception:
        approved_names = set()
    if not approved_names:
        return ""

    selected_names: set[str] = set()
    for raw_path in list(selected_source_files or []):
        try:
            image_name = Path(raw_path).name
        except Exception:
            image_name = str(raw_path or "").strip()
        normalized = CAMPAIGN._normalize_image_set_name(image_name)
        if normalized:
            selected_names.add(normalized)

    if not selected_names:
        return ""

    adopted_count = len(selected_names & approved_names)
    if adopted_count <= 0:
        return ""

    remaining_count = max(0, len(selected_names) - adopted_count)
    if remaining_count > 0:
        return (
            "\n\nImport/adopcja anotacji: "
            f"{adopted_count} zdjęć z wybranego katalogu ma już status [OK] i jest gotowe "
            "do eksportu datasetu YOLO albo wyodrębniania tablic. "
            f"W E2/Z2 do dalszej pracy pokażemy tylko pozostałe {remaining_count} zdjęć bez "
            "zaadoptowanych anotacji."
        )

    return (
        "\n\nImport/adopcja anotacji: wszystkie zdjęcia z wybranego katalogu mają już status [OK]. "
        "E2/Z2 nie musi pokazywać ich ponownie do anotacji; są gotowe do eksportu datasetu YOLO "
        "albo wyodrębniania tablic."
    )

def _approve_step1_with_existing_char_material(self, state: dict | None = None) -> bool:
    """Approve E1 when the char route already has enough plate material."""
    if self._get_iteration_target() != "char":
        return False
    state = dict(state or self._get_step1_char_route_preflight_state() or {})
    if not bool(state.get("material_ready")):
        return False

    plate_count = int(state.get("plate_material_count", 0) or 0)
    min_plates = int(state.get("min_plates", getattr(self, "STEP3_CHAR_MIN_PLATES", 10)) or 10)
    image_count = int(state.get("source_image_count", 0) or state.get("approved_image_count", 0) or 0)
    message = (
        "Tor znaków ma już spełniony warunek wejścia bez wskazywania nowego katalogu zdjęć.\n\n"
        f"Gotowe tablice: {plate_count}/{min_plates}\n"
        f"Zdjęcia z gotowymi anotacjami: {image_count}\n\n"
        "Po zatwierdzeniu tej bramki program pominie E2 i przejdzie bezpośrednio do E3/Z3, "
        "czyli do pracy nad znakami na gotowych tablicach."
    )
    try:
        should_approve = self.app.themed_confirm(
            "Zatwierdzenie E1: tor znaków",
            message,
            parent=self.frame,
            confirm_label="Zatwierdź E1",
            cancel_label="Wróć",
            tone="info",
        )
    except Exception:
        should_approve = messagebox.askyesno("Zatwierdzenie E1: tor znaków", message, parent=self.frame)
    if not should_approve:
        return True

    try:
        CAMPAIGN.approve_step1()
        CAMPAIGN.set_current_step(max(3, int(CAMPAIGN.get_current_step() or 1)))
    except Exception as e:
        logger.error(f"Nie udało się zatwierdzić E1 na gotowych tablicach: {e}")
        return True

    try:
        self.app.update_status(
            f"Zatwierdzono E1 dla toru znaków na gotowych tablicach: {plate_count}/{min_plates}.",
            "info",
        )
    except Exception:
        pass
    try:
        self._refresh_active_project_wizard_only()
    except Exception:
        try:
            self._refresh_dashboard()
        except Exception:
            pass
    return True

def _apply_current_ingest_plan(self):
    if not CAMPAIGN.get_active_project_name():
        return
    if CAMPAIGN.get_step1_status() != "approved" and not self._ensure_step1_iteration_target_selected():
        return
    if not self.current_ingest_plan:
        try:
            self._ensure_current_ingest_plan_from_master_pool()
        except Exception as e:
            logger.debug(f"Nie udało się przygotować planu E1 przed zatwierdzeniem: {e}")
    if not self.current_ingest_plan:
        raw_dir = CAMPAIGN.get_dir("raw")
        if raw_dir is None:
            return
        iter_num = CAMPAIGN.get_current_iteration_num()
        target_iter_dir = CAMPAIGN.get_iteration_raw_dir(iter_num) or (Path(raw_dir) / f"Iteracja_{iter_num:03d}")

        try:
            manifest = CAMPAIGN.load_ingest_manifest(iter_num) or {}
        except Exception:
            manifest = {}
        manifest_images = []
        if isinstance(manifest, dict):
            try:
                manifest_images = list(CAMPAIGN.get_iteration_manifest_image_paths(iter_num) or [])
            except Exception:
                manifest_images = []

        source_dir = CAMPAIGN.get_iteration_image_source_dir(iter_num) or target_iter_dir
        selected_images = list(manifest_images or [])
        if not selected_images and target_iter_dir.exists() and target_iter_dir.is_dir():
            selected_images = [
                image_path for image_path in target_iter_dir.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
            ]
            source_dir = target_iter_dir

        if not selected_images:
            try:
                char_state = self._get_step1_char_route_preflight_state()
            except Exception:
                char_state = {}
            if self._approve_step1_with_existing_char_material(char_state):
                return
            messagebox.showwarning("Brak wybranego folderu zdjęć E1", "Załaduj najpierw wybrany folder zdjęć E1.")
            return
        if not self._validate_step1_char_preflight_for_approval():
            return
        if CAMPAIGN.get_step1_status() != "approved":
            selection_mode = (
                str((manifest or {}).get("selection_mode", "") or "").strip()
                if isinstance(manifest, dict)
                else ""
            ) or ("planned_manifest" if manifest_images else "existing")
            adoption_scope_notice = self._format_step1_adopted_annotations_z2_scope_notice(selected_images)
            should_approve = self.app.themed_confirm(
                "Zatwierdzenie E1",
                (
                    f"Wybrany katalog zdjęć zawiera {len(selected_images)} obrazów dla bieżącej iteracji:\n{source_dir}\n\n"
                    "Czy zatwierdzić ten zestaw zdjęć jako E1 i odblokować E2?"
                    f"{adoption_scope_notice}"
                ),
                parent=self.frame,
                confirm_label="Zatwierdź",
                tone="info",
            )
            if not should_approve:
                return
            _show_ingest_plan_progress_dialog(
                self,
                source_dir,
                parent=self.frame,
                title="Zatwierdzanie E1",
                eyebrow="ZATWIERDZANIE E1",
                initial_detail="Zapisuję manifest obrazów i aktualizuję graf kampanii.",
            )
            try:
                self._approve_current_iteration_package(
                    target_iter_dir=target_iter_dir,
                    source_dir=source_dir,
                    selected_source_files=selected_images,
                    selection_mode=selection_mode,
                    progress_callback=lambda value, message="", **kwargs: _update_ingest_plan_progress_dialog(
                        self,
                        value,
                        message,
                        str(kwargs.get("detail", "") or ""),
                    ),
                )
                _update_ingest_plan_progress_dialog(
                    self,
                    100,
                    "E1 zatwierdzone.",
                    "Manifest został zapisany, a graf odświeżony.",
                )
            finally:
                _hide_ingest_plan_progress_dialog(self, delay_ms=650)
        return

    selected_items = list(self.current_ingest_plan.get("selected", []) or [])
    if not selected_items:
        try:
            char_state = self._get_step1_char_route_preflight_state()
        except Exception:
            char_state = {}
        if self._approve_step1_with_existing_char_material(char_state):
            return
        messagebox.showwarning("Brak wybranego folderu zdjęć E1", "Załaduj najpierw wybrany folder zdjęć E1.")
        return
    if not self._validate_step1_char_preflight_for_approval():
        return

    raw_dir = CAMPAIGN.get_dir("raw")
    if raw_dir is None:
        logger.error("Brak katalogu raw dla aktywnego projektu.")
        return

    iter_num = CAMPAIGN.get_current_iteration_num()
    target_iter_dir = CAMPAIGN.get_iteration_raw_dir(iter_num) or (Path(raw_dir) / f"Iteracja_{iter_num:03d}")
    target_iter_dir.mkdir(parents=True, exist_ok=True)

    _show_ingest_plan_progress_dialog(
        self,
        self.current_ingest_plan.get("master_pool_dir") or CAMPAIGN.get_master_pool_dir() or target_iter_dir,
        parent=self.frame,
        title="Zatwierdzanie E1",
        eyebrow="ZATWIERDZANIE E1",
        initial_detail="Przygotowuję manifest wybranego zbioru obrazów.",
    )
    _update_ingest_plan_progress_dialog(
        self,
        6,
        "Przygotowuję listę obrazów do manifestu.",
        f"Do sprawdzenia: {len(selected_items)} pozycji z planu E1.",
    )
    selected_source_files = []
    last_source_progress = perf_counter()
    for processed_count, item in enumerate(selected_items, start=1):
        source_path = Path(str(item.get("source_path", "") or "").strip())
        if not source_path.exists() or not source_path.is_file():
            now = perf_counter()
            if processed_count == 1 or processed_count == len(selected_items) or processed_count % 250 == 0 or now - last_source_progress >= 0.35:
                last_source_progress = now
                _update_ingest_plan_progress_dialog(
                    self,
                    6.0 + 10.0 * (processed_count / max(1, len(selected_items))),
                    "Przygotowuję listę obrazów do manifestu.",
                    f"Sprawdzono {processed_count}/{len(selected_items)}. Poprawne: {len(selected_source_files)}.",
                )
            continue
        selected_source_files.append(source_path)
        now = perf_counter()
        if processed_count == 1 or processed_count == len(selected_items) or processed_count % 250 == 0 or now - last_source_progress >= 0.35:
            last_source_progress = now
            _update_ingest_plan_progress_dialog(
                self,
                6.0 + 10.0 * (processed_count / max(1, len(selected_items))),
                "Przygotowuję listę obrazów do manifestu.",
                f"Sprawdzono {processed_count}/{len(selected_items)}. Poprawne: {len(selected_source_files)}.",
            )

    if not selected_source_files:
        _hide_ingest_plan_progress_dialog(self)
        messagebox.showwarning(
            "Brak obrazów w manifeście E1",
            "Nie udało się odczytać obrazów z wybranego katalogu zdjęć. Wybierz katalog ponownie.",
        )
        return

    selection_mode = str(self.current_ingest_plan.get("selection_mode") or "").strip()
    if not selection_mode:
        selection_mode = "source_reuse" if bool(self.current_ingest_plan.get("source_reuse")) else "planned_manifest"
    source_dir_text = str(self.current_ingest_plan.get("source_dir") or self.current_ingest_plan.get("master_pool_dir") or "").strip()
    source_dir = Path(source_dir_text) if source_dir_text else (CAMPAIGN.get_master_pool_dir() or target_iter_dir)
    try:
        if source_dir is None or not Path(source_dir).exists() or not Path(source_dir).is_dir():
            source_dir = selected_source_files[0].parent
    except Exception:
        source_dir = selected_source_files[0].parent
    try:
        self._approve_current_iteration_package(
            target_iter_dir=target_iter_dir,
            source_dir=source_dir,
            selected_source_files=selected_source_files,
            selection_mode=selection_mode,
            selected_source_metadata=selected_items,
            proposal_summary={
                "planner_version": self.current_ingest_plan.get("planner_version", ""),
                "generated_at": self.current_ingest_plan.get("generated_at", ""),
                "source_total": self.current_ingest_plan.get("raw_total", 0),
                "selected_total": self.current_ingest_plan.get("selected_total", 0),
                "current_iteration_package_count": self.current_ingest_plan.get("selected_total", 0),
                "batch_size": self.current_ingest_plan.get("batch_size", 0),
                "skipped_duplicate_filenames": self.current_ingest_plan.get("skipped_duplicate_filenames", self.current_ingest_plan.get("skipped_used", 0)),
                "skipped_duplicate_approved_filenames": self.current_ingest_plan.get("skipped_duplicate_approved_filenames", 0),
                "project_overlap_filenames": self.current_ingest_plan.get("project_overlap_filenames", 0),
                "new_to_project_count": self.current_ingest_plan.get("new_to_project_total", 0),
                "skipped_invalid_ground_truth": self.current_ingest_plan.get("skipped_invalid_ground_truth", 0),
            },
            progress_callback=lambda value, message="", **kwargs: _update_ingest_plan_progress_dialog(
                self,
                value,
                message,
                str(kwargs.get("detail", "") or ""),
            ),
        )
        _update_ingest_plan_progress_dialog(
            self,
            100,
            "E1 zatwierdzone.",
            "Manifest został zapisany, a graf odświeżony.",
        )
    finally:
        _hide_ingest_plan_progress_dialog(self)

    try:
        self.app.update_status(
            f"Zatwierdzono E1 manifestem: {len(selected_source_files)} zdjęć z wybranego katalogu. Odblokowano Krok 2.",
            "info",
        )
    except Exception:
        pass

    adoption_scope_notice = self._format_step1_adopted_annotations_z2_scope_notice(selected_source_files)
    messagebox.showinfo(
        "E1 zatwierdzone",
        f"Zapisano manifest iteracji {iter_num:03d}: {len(selected_source_files)} zdjęć.\n\n"
        "Zdjęcia nie są kopiowane do kolejnej iteracji. Program będzie korzystał z wybranego katalogu źródłowego."
        f"{adoption_scope_notice}",
    )

def _format_step1_selection_mode_label(selection_mode: str) -> str:
    normalized = str(selection_mode or "").strip().lower()
    labels = {
        "planned": "Wybrano zdjęcia z głównego katalogu zdjęć",
        "manual": "Wskazano katalog zdjęć ręcznie",
        "existing": "Użyto gotowego katalogu iteracji",
        "iteration_reuse": "Użyto tego samego zestawu zdjęć co poprzednio",
        "pool_reuse": "Przygotowano kolejny zestaw zdjęć z tej samej puli projektu",
        "stage_reuse": "Przejęto zdjęcia oczekujące w stage po poprzedniej iteracji",
    }
    return labels.get(normalized, "Tryb przygotowania nie jest jeszcze znany")

def _get_step1_manifest_context(self, manifest: dict | None = None) -> dict:
    manifest = manifest if isinstance(manifest, dict) else {}
    current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    iter_num = int(manifest.get("iteration", current_iteration) or current_iteration)
    cache_scope = f"step1_manifest_context:iter_{int(iter_num or 0):03d}"
    cache_signature = self._build_step1_manifest_context_signature(manifest, iter_num)
    cached_context = self._get_project_view_cache_entry(cache_scope, cache_signature)
    if isinstance(cached_context, dict) and cached_context:
        return dict(cached_context)
    manifest_mode = str(manifest.get("selection_mode", "") or "").strip().lower()
    planned_manifest_modes = {"planned", "planned_manifest", "manual", "existing", "source_reuse"}
    selected_count = int(manifest.get("selected_count", 0) or 0)

    proposal_summary = manifest.get("proposal_summary", {})
    if not isinstance(proposal_summary, dict):
        proposal_summary = {}
    source_total = int(proposal_summary.get("source_total", 0) or 0)
    skipped_duplicate_filenames = int(proposal_summary.get("skipped_duplicate_filenames", 0) or 0)
    skipped_duplicate_approved = int(proposal_summary.get("skipped_duplicate_approved_filenames", 0) or 0)
    project_overlap_filenames = int(proposal_summary.get("project_overlap_filenames", 0) or 0)
    current_iteration_package_count = int(proposal_summary.get("current_iteration_package_count", selected_count) or selected_count)
    new_to_project_count = int(
        proposal_summary.get(
            "new_to_project_count",
            (
                current_iteration_package_count
                if manifest_mode in planned_manifest_modes
                else max(0, current_iteration_package_count - project_overlap_filenames)
            ),
        ) or 0
    )
    project_pool_total_before = int(proposal_summary.get("project_pool_total_before_iteration", 0) or 0)
    project_pool_total_after = int(proposal_summary.get("project_pool_total_after_iteration", 0) or 0)
    approved_images_before = int(proposal_summary.get("approved_images_before_iteration", 0) or 0)
    approved_plates_before = int(proposal_summary.get("approved_plates_before_iteration", 0) or 0)

    source_iteration = int(proposal_summary.get("source_iteration", 0) or 0)
    source_kind = str(proposal_summary.get("source_kind", "") or "").strip().lower()

    if source_iteration <= 0:
        source_dir = str(manifest.get("source_dir", "") or "").strip()
        for raw_part in reversed(str(source_dir).replace("\\", "/").split("/")):
            lowered = str(raw_part).strip().lower()
            if lowered.startswith("iteracja_"):
                try:
                    source_iteration = int(str(raw_part).split("_", 1)[1])
                except Exception:
                    source_iteration = 0
                break

    source_label = f"Iteracja_{source_iteration:03d}" if source_iteration > 0 else ""

    approved_stats = {}
    try:
        approved_stats = CAMPAIGN.get_plate_approved_set_stats() or {}
    except Exception:
        approved_stats = {}
    if approved_images_before <= 0:
        approved_images_before = int(approved_stats.get("images", 0) or 0)
    if approved_plates_before <= 0:
        approved_plates_before = int(approved_stats.get("plates", 0) or 0)
    if project_pool_total_before <= 0:
        project_pool_total_before = approved_images_before
    if project_pool_total_after <= 0:
        project_pool_total_after = max(project_pool_total_before, approved_images_before) + new_to_project_count

    try:
        master_pool_dir = CAMPAIGN.get_master_pool_dir()
    except Exception:
        master_pool_dir = None
    try:
        target_iter_dir = CAMPAIGN.get_iteration_raw_dir(iter_num)
    except Exception:
        target_iter_dir = None

    def _collect_image_names_in_dir(dir_path: Path | None, *, recursive: bool = False) -> set[str]:
        names: set[str] = set()
        if dir_path is None or not dir_path.exists() or not dir_path.is_dir():
            return names
        try:
            iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()
            for image_path in iterator:
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
                filename = str(image_path.name or "").strip().lower()
                if filename:
                    names.add(filename)
        except Exception:
            return names
        return names

    if target_iter_dir is not None:
        try:
            target_iter_dir = Path(target_iter_dir)
            registry = CAMPAIGN.get_project_packet_filename_registry(exclude_iteration_num=iter_num)
            previous_project_names: set[str] = {
                str(name or "").strip().lower()
                for name in list((registry or {}).get("filenames") or [])
                if str(name or "").strip()
            }

            selected_names: set[str] = set()
            for selected_item in list(manifest.get("selected_images") or []):
                if not isinstance(selected_item, dict):
                    continue
                filename = str(selected_item.get("name", "") or "").strip().lower()
                if filename:
                    selected_names.add(filename)
            if not selected_names:
                selected_names = _collect_image_names_in_dir(target_iter_dir, recursive=False)

            if selected_names:
                overlap_actual = len(selected_names & previous_project_names)
                new_actual = max(0, len(selected_names) - overlap_actual)
                project_pool_total_before = max(project_pool_total_before, len(previous_project_names))
                project_pool_total_after = max(project_pool_total_after, len(previous_project_names | selected_names))
                current_iteration_package_count = int(len(selected_names))
                selected_count = int(len(selected_names))
                if manifest_mode in planned_manifest_modes:
                    source_total = int(len(selected_names))

                stale_overlap = int(project_overlap_filenames or 0)
                stale_new = int(new_to_project_count or 0)
                if (
                    manifest_mode in planned_manifest_modes
                    and (
                        stale_overlap != overlap_actual
                        or stale_new != new_actual
                    )
                ):
                    project_overlap_filenames = int(overlap_actual)
                    new_to_project_count = int(new_actual)
        except Exception:
            pass

    if manifest_mode == "stage_reuse":
        if source_label:
            source_summary = f"Stage po {source_label}"
        else:
            source_summary = "Stage po poprzedniej iteracji"
    elif manifest_mode == "pool_reuse":
        source_summary = "Ta sama pula projektu"
    elif manifest_mode == "iteration_reuse":
        source_summary = "Ten sam zestaw zdjęć co poprzednio"
    elif manifest_mode in {"planned", "planned_manifest"}:
        source_summary = "Wybrany zestaw zdjęć z głównej puli projektu"
    elif manifest_mode == "source_reuse":
        source_summary = "Wybrany katalog zdjęć użyty ponownie w tej iteracji"
    elif manifest_mode == "manual":
        source_summary = "Ręcznie wskazany katalog zdjęć wejściowych"
    elif manifest_mode == "existing":
        source_summary = "Gotowy katalog bieżącej iteracji"
    else:
        source_summary = "Źródło zdjęć nie jest jeszcze znane"

    if manifest_mode in planned_manifest_modes and int(current_iteration_package_count or 0) > 0:
        source_total = max(int(source_total or 0), int(current_iteration_package_count or 0))

    result = {
        "iteration": iter_num,
        "selection_mode": manifest_mode,
        "selected_count": selected_count,
        "source_kind": source_kind,
        "source_iteration": source_iteration,
        "source_label": source_label,
        "source_summary": source_summary,
        "source_total": source_total,
        "skipped_duplicate_filenames": skipped_duplicate_filenames,
        "skipped_duplicate_approved": skipped_duplicate_approved,
        "project_overlap_filenames": project_overlap_filenames,
        "project_pool_total": int(project_pool_total_after or 0),
        "project_pool_total_before": int(project_pool_total_before or 0),
        "project_pool_total_after": int(project_pool_total_after or 0),
        "current_iteration_package_count": int(current_iteration_package_count or selected_count or 0),
        "new_to_project_count": int(new_to_project_count or 0),
        "approved_images": approved_images_before,
        "approved_plates": approved_plates_before,
    }
    self._set_project_view_cache_entry(cache_scope, cache_signature, result)
    return result

def _build_step1_summary_payload(self) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}

    manifest = self._load_ingest_manifest_cached()
    if not isinstance(manifest, dict) or not manifest:
        draft_plan = self._get_active_step1_draft_plan()
        if isinstance(draft_plan, dict) and draft_plan:
            latest_selected_total = int(draft_plan.get("selected_total", 0) or 0)
            if latest_selected_total > 0:
                iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
                target_dir = CAMPAIGN.get_iteration_raw_dir(iter_num)
                manifest = {
                    "iteration": iter_num,
                    "selection_mode": "planned",
                    "source_dir": str(draft_plan.get("master_pool_dir") or ""),
                    "target_dir": str(target_dir or ""),
                    "master_pool_dir": str(draft_plan.get("master_pool_dir") or ""),
                    "selected_count": latest_selected_total,
                    "selected_images": list(draft_plan.get("selected") or []),
                    "char_histogram": dict(draft_plan.get("selected_balance") or {}),
                    "created_at": str(draft_plan.get("generated_at") or ""),
                    "proposal_summary": {
                        "source_total": int(draft_plan.get("raw_total", latest_selected_total) or latest_selected_total),
                        "selected_total": latest_selected_total,
                        "current_iteration_package_count": latest_selected_total,
                        "project_overlap_filenames": int(draft_plan.get("project_overlap_filenames", 0) or 0),
                        "new_to_project_count": int(draft_plan.get("new_to_project_total", 0) or 0),
                        "skipped_duplicate_filenames": int(draft_plan.get("skipped_duplicate_filenames", draft_plan.get("skipped_used", 0)) or 0),
                        "skipped_duplicate_approved_filenames": int(draft_plan.get("skipped_duplicate_approved_filenames", 0) or 0),
                        "project_pool_total_before_iteration": int(draft_plan.get("project_pool_total_before_iteration", 0) or 0),
                        "project_pool_total_after_iteration": int(draft_plan.get("project_pool_total_after_iteration", 0) or 0),
                        "skipped_invalid_ground_truth": int(draft_plan.get("skipped_invalid_ground_truth", 0) or 0),
                    },
                }
        if not isinstance(manifest, dict) or not manifest:
            try:
                latest_summary = dict(CAMPAIGN.load_latest_ingest_plan_summary() or {})
            except Exception:
                latest_summary = {}
            try:
                latest_iter = int(latest_summary.get("iteration", 0) or 0)
            except Exception:
                latest_iter = 0
            latest_project = str(latest_summary.get("project", "") or "").strip()
            current_project = str(CAMPAIGN.get_active_project_name() or "").strip()
            latest_selected_total = int(latest_summary.get("selected_total", 0) or 0) if latest_summary else 0
            if latest_selected_total > 0 and latest_iter == int(CAMPAIGN.get_current_iteration_num() or 1) and (
                not latest_project or latest_project == current_project
            ):
                iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
                target_dir = CAMPAIGN.get_iteration_raw_dir(iter_num)
                manifest = {
                    "iteration": iter_num,
                    "selection_mode": "planned",
                    "source_dir": str(latest_summary.get("master_pool_dir") or latest_summary.get("source_dir") or ""),
                    "target_dir": str(target_dir or latest_summary.get("target_dir") or ""),
                    "master_pool_dir": str(latest_summary.get("master_pool_dir") or ""),
                    "selected_count": latest_selected_total,
                    "selected_images": [],
                    "char_histogram": dict(latest_summary.get("selected_balance") or {}),
                    "created_at": str(latest_summary.get("generated_at") or ""),
                    "proposal_summary": {
                        "source_total": int(latest_summary.get("raw_total", latest_selected_total) or latest_selected_total),
                        "selected_total": latest_selected_total,
                        "current_iteration_package_count": latest_selected_total,
                        "project_overlap_filenames": int(latest_summary.get("project_overlap_filenames", 0) or 0),
                        "new_to_project_count": int(latest_summary.get("new_to_project_total", 0) or 0),
                        "skipped_duplicate_filenames": int(latest_summary.get("skipped_duplicate_filenames", latest_summary.get("skipped_used", 0)) or 0),
                        "skipped_duplicate_approved_filenames": int(latest_summary.get("skipped_duplicate_approved_filenames", 0) or 0),
                        "project_pool_total_before_iteration": int(latest_summary.get("project_pool_total_before_iteration", 0) or 0),
                        "project_pool_total_after_iteration": int(latest_summary.get("project_pool_total_after_iteration", 0) or 0),
                        "skipped_invalid_ground_truth": int(latest_summary.get("skipped_invalid_ground_truth", 0) or 0),
                    },
                }
        if not isinstance(manifest, dict) or not manifest:
            if (
                int(CAMPAIGN.get_current_step() or 1) == 1
                and str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved"
            ):
                manifest = self._load_previous_iteration_ingest_manifest()
        if not isinstance(manifest, dict) or not manifest:
            iter_num = int(CAMPAIGN.get_current_iteration_num() or 1)
            target_dir = CAMPAIGN.get_iteration_raw_dir(iter_num)
            source_dir = CAMPAIGN.get_iteration_image_source_dir(iter_num) or target_dir
            try:
                selected_count = int(CAMPAIGN.get_iteration_image_count(iter_num) or 0)
            except Exception:
                selected_count = 0
            if selected_count <= 0 and target_dir is not None and target_dir.exists() and target_dir.is_dir():
                selected_count = sum(
                    1
                    for image_path in target_dir.iterdir()
                    if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
                )
            if selected_count <= 0:
                return {}
            manifest = {
                "iteration": iter_num,
                "selection_mode": "existing",
                "source_dir": str(source_dir or target_dir or ""),
                "target_dir": str(target_dir or ""),
                "master_pool_dir": str(CAMPAIGN.get_master_pool_dir() or ""),
                "selected_count": selected_count,
                "selected_images": [],
                "char_histogram": {},
                "created_at": "",
                "proposal_summary": {},
            }

    selected_count = int(manifest.get("selected_count", 0) or 0)
    target_dir = str(manifest.get("target_dir", "") or "").strip()
    if selected_count <= 0 and not target_dir:
        return {}
    step1_context = self._get_step1_manifest_context(manifest)
    package_count = int(
        step1_context.get("current_iteration_package_count", 0)
        or self._get_iteration_image_count()
        or selected_count
        or 0
    )
    source_total = int(step1_context.get("source_total", 0) or 0)
    project_overlap_count = int(step1_context.get("project_overlap_filenames", 0) or 0)
    approved_overlap_count = int(step1_context.get("skipped_duplicate_approved", 0) or 0)
    project_pool_total = int(step1_context.get("project_pool_total_after", step1_context.get("project_pool_total", 0)) or 0)
    approved_images = int(step1_context.get("approved_images", 0) or 0)
    new_to_project_count = int(step1_context.get("new_to_project_count", 0) or 0)
    source_summary = str(step1_context.get("source_summary", "") or "").strip()
    source_label = str(step1_context.get("source_label", "") or "").strip()
    if source_label and str(step1_context.get("selection_mode", "") or "").strip().lower() == "stage_reuse":
        source_summary = f"{source_summary} ({source_label})"

    manifest_mode = str(manifest.get("selection_mode", "") or "").strip().lower()
    manifest_image_names = list(manifest.get("selected_images") or [])
    manifest_count = 0
    try:
        manifest_count = int(CAMPAIGN.get_iteration_manifest_image_count() or 0)
    except Exception:
        manifest_count = 0
    if manifest_count <= 0:
        manifest_count = int(len(manifest_image_names) or 0)
    manifest_contract_modes = {
        "planned",
        "planned_manifest",
        "source_reuse",
        "pool_reuse",
        "stage_reuse",
        "iteration_reuse",
    }
    uses_manifest_contract = bool(
        manifest.get("manifest_only", False)
        or manifest_mode in manifest_contract_modes
        or manifest_image_names
        or manifest_count > 0
    )
    source_contract_text = (
        "Manifestowy zestaw zdjęć"
        if uses_manifest_contract
        else "Fizyczny katalog iteracji"
    )
    if uses_manifest_contract and manifest_count > 0:
        source_contract_text = f"{source_contract_text} ({manifest_count} w manifeście)"
    selection_mode_text = self._format_step1_selection_mode_label(manifest_mode)
    current_iteration = int(manifest.get("iteration", CAMPAIGN.get_current_iteration_num()) or CAMPAIGN.get_current_iteration_num())
    total_pool_images = max(int(project_pool_total or 0), int(package_count or 0))
    previous_approved_images = max(0, int(approved_images or 0))
    current_iteration_package = int(package_count or 0)
    current_source_dir_count = int(source_total or step1_context.get("source_total", 0) or 0)
    plate_source_info = self._get_project_start_plate_source_info()
    plate_xml_path = str(plate_source_info.get("xml_path") or "").strip()
    plate_run_path = str(plate_source_info.get("run_path") or "").strip()
    plate_annotations_text = self._format_project_start_asset_source(plate_xml_path or plate_run_path)
    plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    plate_model_identity = self._get_model_identity_label(plate_model_path)
    plate_model_text = plate_model_identity or self._format_project_start_asset_source(plate_model_path)
    char_model_path = str(CAMPAIGN.get_global_model("char") or "").strip()
    char_model_identity = self._get_model_identity_label(char_model_path)
    char_model_text = char_model_identity or self._format_project_start_asset_source(char_model_path)
    source_diverged = bool(
        current_source_dir_count > 0
        and current_iteration_package > 0
        and current_source_dir_count != current_iteration_package
    )

    rows = [
        ("Tryb wejścia", source_contract_text),
        ("Sposób wyboru", selection_mode_text),
        (
            "Pudełko: tablice zatwierdzone",
            f"{previous_approved_images} zdjęć / {int(step1_context.get('approved_plates', 0) or 0)} tablic",
        ),
        ("Anotacje tablic", plate_annotations_text),
        ("Model tablic", plate_model_text),
        ("Model znaków", char_model_text),
        ("Zdjęcia iteracji", f"{current_iteration_package} zdjęć"),
        ("Nowe w historii projektu", f"{new_to_project_count} zdjęć"),
        ("Duble względem wcześniejszych iteracji", f"{project_overlap_count} zdjęć"),
        ("Pula projektu po E1 (informacyjnie)", f"{total_pool_images} zdjęć"),
    ]
    if source_diverged:
        rows.insert(1, ("Obecny katalog źródłowy", f"{current_source_dir_count} zdjęć"))

    return {
        "title": "Co wybrano w E1",
        "meta": "",
        "rows": rows,
        "source_diverged": bool(source_diverged),
        "current_source_dir_count": int(current_source_dir_count or 0),
        "package_count": int(current_iteration_package or 0),
        "uses_manifest_contract": bool(uses_manifest_contract),
        "source_contract_text": source_contract_text,
    }
