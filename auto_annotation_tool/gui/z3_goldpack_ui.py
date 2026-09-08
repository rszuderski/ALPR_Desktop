#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold pack status and filter UI helpers for Z3/PZ3."""

import datetime
import json
import copy
import os
import shutil
import tkinter as tk
import uuid
from pathlib import Path
from .z3_metadata_cache import read_preview_metadata

import cv2

from ..campaign_manager import CAMPAIGN
from .web_slim_scrollbar import blend_hex_colors


def set_preview_record_source_info(host, text: str, tone: str = "muted") -> None:
    label = getattr(host, "preview_record_source_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def set_gold_export_scope_info(host, text: str, tone: str = "muted") -> None:
    label = getattr(host, "gold_export_scope_lbl", None)
    if isinstance(label, tk.Label):
        current_text = str(label.cget("text") or "")
        current_tone = str(getattr(label, "_inline_status_tone", "") or "")
        current_emphasis = bool(getattr(label, "_inline_status_emphasis", False))
        if current_text == str(text) and current_tone == str(tone).strip().lower() and current_emphasis is False:
            return
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def format_gold_export_split_summary(host) -> str:
    return "wariant/split: Z4"


def empty_perfect_strategy_counts(perfect_strategy_buckets) -> dict:
    return {key: 0 for key, _ in perfect_strategy_buckets}


def empty_gold_source_counts(gold_source_buckets) -> dict:
    return {key: 0 for key, _ in gold_source_buckets}


def format_perfect_strategy_counts(counts: dict, perfect_strategy_buckets) -> str:
    safe_counts = counts if isinstance(counts, dict) else {}
    parts = [
        f"{label}: {int(safe_counts.get(key, 0))}"
        for key, label in perfect_strategy_buckets
    ]
    return "Tablice perfect wg strategii: " + " | ".join(parts)


def empty_plate_layout_counts() -> dict:
    return {"1R": 0, "1R*": 0, "2R": 0, "2R?": 0, "2R*": 0, "?": 0}


def increment_plate_layout_counts(host, counts: dict, data: dict | None, *, amount: int = 1) -> None:
    if not isinstance(counts, dict):
        return
    label = host._get_plate_layout_count_label(data)
    if label not in counts:
        counts[label] = 0
    counts[label] = int(counts.get(label, 0) or 0) + int(amount)


def format_plate_layout_counts(counts: dict | None, *, prefix: str = "Układ") -> str:
    safe_counts = counts if isinstance(counts, dict) else {}
    order = ("1R", "1R*", "2R", "2R?", "2R*", "?")
    keys = [key for key in order if int(safe_counts.get(key, 0) or 0) > 0]
    keys.extend(sorted(key for key, value in safe_counts.items() if key not in keys and int(value or 0) > 0))
    if not keys:
        return f"{prefix}: brak"
    return f"{prefix}: " + " | ".join(f"{key}: {int(safe_counts.get(key, 0) or 0)}" for key in keys)


def get_selected_gold_export_strategy_buckets(host) -> set:
    selected = set()
    if bool(getattr(host, "gold_include_ocr_exact_var", None).get() if hasattr(host, "gold_include_ocr_exact_var") else True):
        selected.add("ocr_exact")
    if bool(getattr(host, "gold_include_yolo_exact_var", None).get() if hasattr(host, "gold_include_yolo_exact_var") else True):
        selected.add("yolo_exact")
    if bool(getattr(host, "gold_include_ocr_yolo_rescue_var", None).get() if hasattr(host, "gold_include_ocr_yolo_rescue_var") else True):
        selected.add("ocr_yolo_rescue")
    if bool(getattr(host, "gold_include_yolo_box_ocr_var", None).get() if hasattr(host, "gold_include_yolo_box_ocr_var") else True):
        selected.add("yolo_box_ocr")
    if bool(getattr(host, "gold_include_other_perfect_var", None).get() if hasattr(host, "gold_include_other_perfect_var") else True):
        selected.add("other_perfect")
    return selected


def format_selected_gold_export_strategy_labels(host, perfect_strategy_buckets) -> str:
    selected = host._get_selected_gold_export_strategy_buckets()
    labels = [label for key, label in perfect_strategy_buckets if key in selected]
    return ", ".join(labels) if labels else "brak"


def get_selected_gold_export_source_buckets(host) -> set:
    selected = {"cvat_manual"}
    if bool(getattr(host, "gold_include_source_auto_var", None).get() if hasattr(host, "gold_include_source_auto_var") else True):
        selected.add("auto_preview")
    if bool(getattr(host, "gold_include_source_local_manual_var", None).get() if hasattr(host, "gold_include_source_local_manual_var") else True):
        selected.add("local_manual")
    return selected


def format_selected_gold_export_source_labels(host, gold_source_buckets) -> str:
    selected = host._get_selected_gold_export_source_buckets()
    labels = [label for key, label in gold_source_buckets if key in selected]
    return ", ".join(labels) if labels else "brak"


def get_gold_export_split_percentages(host) -> tuple[float, float, float]:
    train = float(getattr(host, "gold_export_train_pct_var", tk.DoubleVar(value=80.0)).get())
    val = float(getattr(host, "gold_export_val_pct_var", tk.DoubleVar(value=10.0)).get())
    max_train_plus_val = 95.0
    if train + val > max_train_plus_val:
        val = max(5.0, max_train_plus_val - train)
        try:
            host.gold_export_val_pct_var.set(val)
        except Exception:
            pass

    test = max(5.0, 100.0 - train - val)
    return train, val, test


def build_gold_export_plate_unique_key(data: dict) -> str:
    if not isinstance(data, dict):
        return ""

    source_image = str(data.get("source_image", "u") or "u")
    source_bbox = data.get("source_bbox") or []
    try:
        bbox_bucket = int(float(source_bbox[0]) // 10) if source_bbox else 0
    except Exception:
        bbox_bucket = 0
    return f"{source_image}_{bbox_bucket}"


def _resolve_gold_export_meta_path(raw_path) -> Path | None:
    if not raw_path:
        return None
    try:
        path = Path(raw_path)
    except Exception:
        return None
    if path.is_dir():
        path = path / "metadata.json"
    if path.name != "metadata.json":
        return None
    images_dir = path.parent / "images"
    if path.exists() and images_dir.exists():
        return path
    return None


def collect_gold_export_plate_candidates(host, selected_buckets, selected_sources=None, *,
                                         prepare_records: bool = True) -> tuple[list[dict], dict, dict, dict, dict]:
    selected = set(selected_buckets or [])
    selected_source_buckets = set(selected_sources or [])
    total_strategy_counts = host._empty_perfect_strategy_counts()
    selected_strategy_counts = host._empty_perfect_strategy_counts()
    total_source_counts = host._empty_gold_source_counts()
    selected_source_counts = host._empty_gold_source_counts()
    plate_entries = []
    seen = set()

    meta_candidates = host._get_gold_export_meta_candidates()

    for meta in meta_candidates:
        run_dir = meta.parent
        metadata = read_preview_metadata(host, meta)
        with os.scandir(run_dir / "images") as image_entries:
            image_names = {entry.name for entry in image_entries if entry.is_file()}

        for pid, source_data in metadata.items():
            if not isinstance(source_data, dict) or source_data.get("status") != "perfect":
                continue
            data = source_data
            if prepare_records:
                data = _copy_gold_candidate_for_normalization(source_data)
                ensure_plate_source_metadata(host, data, plate_id=str(pid or ""), meta_path=meta, include_diagnostics=False)
                chars = list(data.get("characters", []) or []) if isinstance(data.get("characters", []), list) else []
                host._update_preview_plate_layout_metadata(data, chars)
                if chars:
                    data["characters"] = host._annotate_preview_character_reading_positions(
                        host._sort_character_records_by_x(chars, data=data),
                        data=data,
                    )
            char_count = host._count_exportable_characters_in_data(data)
            if char_count <= 0:
                continue
            strategy_bucket = host._get_perfect_strategy_bucket(data)
            source_bucket = host._get_plate_source_bucket(data, meta_path=meta)
            total_strategy_counts[strategy_bucket] += 1
            if source_bucket in total_source_counts:
                total_source_counts[source_bucket] += 1
            if selected and strategy_bucket not in selected:
                continue
            if selected_source_buckets and source_bucket not in selected_source_buckets:
                continue

            unique_key = host._build_gold_export_plate_unique_key(data)
            if unique_key in seen:
                continue

            img_src = run_dir / "images" / f"{pid}.jpg"
            if img_src.name not in image_names:
                continue

            seen.add(unique_key)
            selected_strategy_counts[strategy_bucket] += 1
            if source_bucket in selected_source_counts:
                selected_source_counts[source_bucket] += 1
            plate_entries.append(
                {
                    "pid": str(pid),
                    "data": data,
                    "run_dir": run_dir,
                    "img_src": img_src,
                    "strategy_bucket": strategy_bucket,
                    "source_bucket": source_bucket,
                }
            )

    return plate_entries, total_strategy_counts, selected_strategy_counts, total_source_counts, selected_source_counts


def count_exportable_characters_in_data(host, data: dict) -> int:
    if not isinstance(data, dict):
        return 0

    count = 0
    for rec in list(data.get("characters", []) or []):
        if host._is_exportable_character_record(rec):
            count += 1
    return count


def count_exportable_perfect_plates_in_metadata(host, metadata_map) -> int:
    count = 0
    for _pid, data in (metadata_map or {}).items():
        if not isinstance(data, dict):
            continue
        if str(data.get("status", "unknown") or "unknown").strip().lower() != "perfect":
            continue
        if host._count_exportable_characters_in_data(data) > 0:
            count += 1
    return int(count)


def _get_contextual_gold_metadata(host) -> dict:
    metadata_map = getattr(host, "preview_metadata", None)
    if isinstance(metadata_map, dict) and metadata_map:
        return metadata_map

    try:
        context = host._get_active_preview_context()
    except Exception:
        context = {}
    if not bool(context.get("ready")):
        return {}

    meta_path = context.get("meta_path")
    try:
        safe_meta_path = Path(meta_path)
    except Exception:
        return {}
    if not safe_meta_path.exists() or not safe_meta_path.is_file():
        return {}

    try:
        loaded = read_preview_metadata(host, safe_meta_path)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}

    return loaded


def build_contextual_gold_export_counts(host, *, selected_strategies=None, selected_sources=None):
    metadata = _get_contextual_gold_metadata(host)
    revision = getattr(host, "_preview_metadata_revision", None)
    key = (id(metadata), revision, tuple(sorted(selected_strategies or ())), tuple(sorted(selected_sources or ())))
    cached = getattr(host, "_preview_contextual_counts_cache", None)
    if isinstance(revision, int) and isinstance(cached, tuple) and cached[0] == key and cached[2] is metadata:
        return copy.deepcopy(cached[1])
    result = _compute_contextual_gold_export_counts(host, selected_strategies=selected_strategies,
                                                   selected_sources=selected_sources)
    if isinstance(revision, int):
        host._preview_contextual_counts_cache = (key, copy.deepcopy(result), metadata)
    return result


def _compute_contextual_gold_export_counts(host, *, selected_strategies=None, selected_sources=None):
    strategy_filter = set(selected_strategies or [])
    source_filter = set(selected_sources or [])
    use_strategy_filter = bool(strategy_filter)
    use_source_filter = bool(source_filter)

    strategy_counts = host._empty_perfect_strategy_counts()
    strategy_char_counts = host._empty_perfect_strategy_counts()
    source_counts = host._empty_gold_source_counts()
    source_char_counts = host._empty_gold_source_counts()
    selected_layout_counts = host._empty_plate_layout_counts()
    selected_layout_char_counts = host._empty_plate_layout_counts()
    selected_plate_count = 0
    selected_char_count = 0

    metadata_map = _get_contextual_gold_metadata(host)
    for _pid, data in (metadata_map or {}).items():
        if not isinstance(data, dict):
            continue
        if str(data.get("status", "unknown") or "unknown").strip().lower() != "perfect":
            continue

        strategy_bucket = host._get_perfect_strategy_bucket(data)
        source_bucket = host._get_plate_source_bucket(data)
        char_count = host._count_exportable_characters_in_data(data)
        if char_count <= 0:
            continue
        strategy_match = (not use_strategy_filter) or strategy_bucket in strategy_filter
        source_match = (not use_source_filter) or source_bucket in source_filter

        if source_match:
            if strategy_bucket in strategy_counts:
                strategy_counts[strategy_bucket] += 1
                strategy_char_counts[strategy_bucket] += char_count

        if strategy_match:
            if source_bucket in source_counts:
                source_counts[source_bucket] += 1
                source_char_counts[source_bucket] += char_count

        if strategy_match and source_match:
            selected_plate_count += 1
            selected_char_count += char_count
            host._increment_plate_layout_counts(selected_layout_counts, data)
            host._increment_plate_layout_counts(selected_layout_char_counts, data, amount=char_count)

    return {
        "strategy_counts": strategy_counts,
        "strategy_char_counts": strategy_char_counts,
        "source_counts": source_counts,
        "source_char_counts": source_char_counts,
        "selected_layout_counts": selected_layout_counts,
        "selected_layout_char_counts": selected_layout_char_counts,
        "selected_plate_count": int(selected_plate_count),
        "selected_char_count": int(selected_char_count),
    }


def count_statuses_in_metadata_mapping(host, metadata_map):
    revision = getattr(host, "_preview_metadata_revision", None)
    key = (id(metadata_map), revision)
    cached = getattr(host, "_preview_counts_cache", None)
    if isinstance(revision, int) and isinstance(cached, tuple) and cached[0] == key and cached[2] is metadata_map:
        return copy.deepcopy(cached[1])
    result = _compute_statuses_in_metadata_mapping(host, metadata_map)
    if isinstance(revision, int):
        host._preview_counts_cache = (key, copy.deepcopy(result), metadata_map)
    return result


def _compute_statuses_in_metadata_mapping(host, metadata_map):
    perfect = 0
    needs_fix = 0
    unknown = 0
    char_boxes = 0
    strategy_counts = host._empty_perfect_strategy_counts()
    strategy_char_counts = host._empty_perfect_strategy_counts()
    source_counts = host._empty_gold_source_counts()
    source_char_counts = host._empty_gold_source_counts()
    layout_counts = {}
    layout_perfect_counts = {}

    for _, data in (metadata_map or {}).items():
        if not isinstance(data, dict):
            unknown += 1
            continue

        try:
            char_boxes += len(list(data.get("characters", []) or []))
        except Exception:
            pass

        layout_label = host._get_plate_layout_count_label(data)
        layout_counts[layout_label] = int(layout_counts.get(layout_label, 0) or 0) + 1

        status = str(data.get("status", "unknown")).strip().lower()
        if status == "perfect":
            perfect += 1
            layout_perfect_counts[layout_label] = int(layout_perfect_counts.get(layout_label, 0) or 0) + 1
            bucket = host._get_perfect_strategy_bucket(data)
            source_bucket = host._get_plate_source_bucket(data)
            strategy_counts[bucket] += 1
            strategy_char_counts[bucket] += host._count_exportable_characters_in_data(data)
            if source_bucket in source_counts:
                source_counts[source_bucket] += 1
                source_char_counts[source_bucket] += host._count_exportable_characters_in_data(data)
        elif status == "needs_fix":
            needs_fix += 1
        else:
            unknown += 1

    return {
        "perfect": perfect,
        "needs_fix": needs_fix,
        "unknown": unknown,
        "total": perfect + needs_fix + unknown,
        "char_boxes": int(char_boxes),
        "strategy_counts": strategy_counts,
        "strategy_char_counts": strategy_char_counts,
        "source_counts": source_counts,
        "source_char_counts": source_char_counts,
        "layout_counts": layout_counts,
        "layout_perfect_counts": layout_perfect_counts,
    }


def build_merged_gold_export_counts(host, *, selected_strategies=None, selected_sources=None):
    strategy_filter = set(selected_strategies or [])
    source_filter = set(selected_sources or [])
    use_strategy_filter = bool(strategy_filter)
    use_source_filter = bool(source_filter)

    strategy_counts = host._empty_perfect_strategy_counts()
    strategy_char_counts = host._empty_perfect_strategy_counts()
    source_counts = host._empty_gold_source_counts()
    source_char_counts = host._empty_gold_source_counts()
    selected_plate_count = 0
    selected_char_count = 0
    selected_layout_counts = host._empty_plate_layout_counts()
    selected_layout_char_counts = host._empty_plate_layout_counts()

    meta_candidates = host._get_gold_export_meta_candidates()

    seen = set()
    for meta in meta_candidates:
        try:
            metadata = read_preview_metadata(host, meta)
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue

        for pid, data in metadata.items():
            if not isinstance(data, dict):
                continue
            if str(data.get("status", "unknown") or "unknown").strip().lower() != "perfect":
                continue
            data = _copy_gold_candidate_for_normalization(data)

            try:
                ensure_plate_source_metadata(host, data, plate_id=str(pid or ""), meta_path=meta, include_diagnostics=False)
            except Exception:
                pass

            unique_key = host._build_gold_export_plate_unique_key(data)
            if unique_key in seen:
                continue
            seen.add(unique_key)

            strategy_bucket = host._get_perfect_strategy_bucket(data)
            source_bucket = host._get_plate_source_bucket(data, meta_path=meta)
            char_count = host._count_exportable_characters_in_data(data)
            if char_count <= 0:
                continue
            strategy_match = (not use_strategy_filter) or (strategy_bucket in strategy_filter)
            source_match = (not use_source_filter) or (source_bucket in source_filter)

            if source_match and strategy_bucket in strategy_counts:
                strategy_counts[strategy_bucket] += 1
                strategy_char_counts[strategy_bucket] += char_count

            if strategy_match and source_bucket in source_counts:
                source_counts[source_bucket] += 1
                source_char_counts[source_bucket] += char_count

            if strategy_match and source_match:
                selected_plate_count += 1
                selected_char_count += char_count
                host._increment_plate_layout_counts(selected_layout_counts, data)
                host._increment_plate_layout_counts(selected_layout_char_counts, data, amount=char_count)

    return {
        "strategy_counts": strategy_counts,
        "strategy_char_counts": strategy_char_counts,
        "source_counts": source_counts,
        "source_char_counts": source_char_counts,
        "selected_plate_count": int(selected_plate_count),
        "selected_char_count": int(selected_char_count),
        "selected_layout_counts": selected_layout_counts,
        "selected_layout_char_counts": selected_layout_char_counts,
    }


def build_campaign_aware_gold_export_counts(host, *, selected_strategies=None, selected_sources=None):
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False

    if in_campaign:
        contextual = host._build_contextual_gold_export_counts(
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )
        strategy_counts = dict(contextual.get("strategy_counts", {}) or {})
        strategy_char_counts = dict(contextual.get("strategy_char_counts", {}) or {})
        source_counts = dict(contextual.get("source_counts", {}) or {})
        source_char_counts = dict(contextual.get("source_char_counts", {}) or {})
        selected_layout_counts = dict(contextual.get("selected_layout_counts", {}) or {})
        selected_layout_char_counts = dict(contextual.get("selected_layout_char_counts", {}) or {})
        selected_plate_count = int(contextual.get("selected_plate_count", 0) or 0)
        selected_char_count = int(contextual.get("selected_char_count", 0) or 0)
        return {
            "strategy_counts": strategy_counts,
            "strategy_char_counts": strategy_char_counts,
            "source_counts": source_counts,
            "source_char_counts": source_char_counts,
            "selected_plate_count": int(selected_plate_count),
            "selected_char_count": int(selected_char_count),
            "selected_layout_counts": selected_layout_counts,
            "selected_layout_char_counts": selected_layout_char_counts,
        }

    return host._build_merged_gold_export_counts(
        selected_strategies=selected_strategies,
        selected_sources=selected_sources,
    )


def get_gold_export_meta_candidates(host) -> list[Path]:
    in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())

    if not in_campaign:
        preview_dir_raw = host._get_preferred_step3_preview_dir(require_plates=True, allow_fallback=True)
        if not preview_dir_raw:
            return []
        try:
            preview_dir = Path(preview_dir_raw)
        except Exception:
            return []
        meta_path = preview_dir / "metadata.json"
        images_dir = preview_dir / "images"
        if meta_path.exists() and images_dir.exists():
            if str(host.preview_dir_var.get() or "").strip() != str(preview_dir):
                try:
                    host.preview_dir_var.set(str(preview_dir))
                    host._save_local_setting("char_preview_dir", str(preview_dir))
                except Exception:
                    pass
            return [meta_path]
        return []

    meta_candidates: list[Path] = []

    def add_candidate(raw_path) -> bool:
        meta_path = _resolve_gold_export_meta_path(raw_path)
        if meta_path is None:
            return False
        if meta_path not in meta_candidates:
            meta_candidates.append(meta_path)
        return True

    try:
        current_preview_dir = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        current_preview_dir = ""
    if add_candidate(current_preview_dir):
        return meta_candidates

    if add_candidate(getattr(host, "_loaded_meta_path", None)):
        return meta_candidates

    try:
        saved_preview_dir = host._get_saved_step3_preview_dir(require_plates=True)
    except Exception:
        saved_preview_dir = ""
    if add_candidate(saved_preview_dir):
        return meta_candidates

    try:
        preferred_preview_dir = host._get_preferred_step3_preview_dir(require_plates=True, allow_fallback=False)
    except Exception:
        preferred_preview_dir = ""
    if add_candidate(preferred_preview_dir):
        return meta_candidates

    return meta_candidates


def run_yolo_gold_export(
    host,
    *,
    char_class_alphabet: str,
    gold_source_buckets,
    gold_source_labels: dict,
) -> None:
    selected_buckets = host._get_selected_gold_export_strategy_buckets()
    selected_labels = host._format_selected_gold_export_strategy_labels()
    selected_sources = host._get_selected_gold_export_source_buckets()
    selected_source_labels = host._format_selected_gold_export_source_labels()
    split_enabled = False
    train_pct, val_pct, test_pct = host._get_gold_export_split_percentages()

    if not selected_buckets:
        host._set_console_text(
            host.export_console,
            "❌ Nie wybrano żadnej strategii perfect do eksportu gold packa.\n\n"
            "Zaznacz co najmniej jedną z opcji: OCR exact, YOLO exact, OCR + YOLO rescue, YOLO boxy + OCR lub Manual / inne perfect."
        )
        return
    if not selected_sources:
        host._set_console_text(
            host.export_console,
            "❌ Nie wybrano żadnego źródła do eksportu gold packa.\n\n"
            "Zaznacz Auto z runu lub Ręczne poprawki lokalne. Poprawki CVAT po imporcie są dołączane automatycznie."
        )
        return

    split_line = "Wariant/split: przygotujesz w Z4"
    host._set_console_text(
        host.export_console,
        f"⏳ Zbieranie idealnych tablic...\nFiltr strategii: {selected_labels}\nŹródła: {selected_source_labels}\n{split_line}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_datasets_dir = host._get_step3_datasets_root_dir()
    yolo_out = base_datasets_dir / f"YOLO_MegaDataset_Chars_{timestamp}"

    try:
        char_map = {c: i for i, c in enumerate(char_class_alphabet)}
        (
            plate_entries,
            total_strategy_counts,
            copied_strategy_counts,
            total_source_counts,
            copied_source_counts,
        ) = host._collect_gold_export_plate_candidates(selected_buckets, selected_sources)
        if not host._confirm_export_with_uncertain_layouts(plate_entries, export_label="YOLO Detect znaków"):
            host._set_console_text(
                host.export_console,
                "Eksport przerwany. Wróć do PZ2, użyj filtra 2R? i potwierdź układ tablic badge’em „Układ”."
            )
            return
        export_entries = []

        for plate_entry in plate_entries:
            pid = str(plate_entry.get("pid", "") or "")
            img_src = plate_entry.get("img_src")
            data = plate_entry.get("data", {})
            strategy_bucket = str(plate_entry.get("strategy_bucket", "") or "")
            source_bucket = str(plate_entry.get("source_bucket", "") or "")
            layout_meta = host._build_plate_layout_export_metadata(data)
            img = cv2.imread(str(img_src))
            if img is None:
                continue

            ih, iw = img.shape[:2]
            new_pid = f"mega_{uuid.uuid4().hex[:8]}_{pid}"
            txt = []
            exported_chars_meta = []
            for c in data.get("characters", []):
                ch = str(c.get("character", "")).upper()
                if ch not in char_map:
                    continue
                x1, y1, x2, y2 = c.get("bbox", [0, 0, 0, 0])
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = max(0.0, min(1.0, ((x1 + x2) / 2.0) / iw))
                yc = max(0.0, min(1.0, ((y1 + y2) / 2.0) / ih))
                w = max(0.0, min(1.0, (x2 - x1) / iw))
                h = max(0.0, min(1.0, (y2 - y1) / ih))
                txt.append(f"{char_map[ch]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                exported_chars_meta.append(
                    {
                        "character": ch,
                        "class_id": int(char_map[ch]),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "reading_row": int(c.get("reading_row", 0) or 0),
                        "reading_col": int(c.get("reading_col", 0) or 0),
                        "reading_index": int(c.get("reading_index", 0) or 0),
                        "source_tag": str(host._get_character_source_tag(c, data=data) or ""),
                        "confidence": float(c.get("confidence", 0.0) or 0.0),
                    }
                )

            if not txt:
                continue

            export_entries.append(
                {
                    "pid": new_pid,
                    "img_src": img_src,
                    "label_text": "\n".join(txt),
                    "strategy_bucket": strategy_bucket,
                    "source_bucket": source_bucket,
                    "source_pid": pid,
                    "source_image": str(data.get("source_image", "") or ""),
                    "layout": layout_meta,
                    "characters": exported_chars_meta,
                }
            )

        copied = len(export_entries)
        if copied == 0:
            msg = (
                "❌ NIE UDAŁO SIĘ UTWORZYĆ DATASETU YOLO.\n\n"
                f"Powód: po filtrze strategii ({selected_labels}) i źródeł ({selected_source_labels}) "
                "nie znaleziono ani jednej tablicy ze statusem perfect.\n\n"
                "Tablica perfect musi mieć przynajmniej jeden poprawny box znaku z etykietą. Puste tablice nie są eksportowane.\n\n"
                f"Dostępne tablice perfect wg strategii przed deduplikacją:\n{host._format_perfect_strategy_counts(total_strategy_counts)}\n"
                f"Dostępne tablice perfect wg źródeł: "
                f"{' | '.join(f'{gold_source_labels[key]}: {int(total_source_counts.get(key, 0) or 0)}' for key, _ in gold_source_buckets)}\n\n"
                "CO DALEJ:\n"
                "1. Możesz rozszerzyć zaznaczone strategie w pz3.\n"
                "2. Możesz wrócić do pz2 i poprawić OCR / Laboratorium.\n"
                "3. Możesz wykonać eksport do CVAT i później zaimportować poprawki.\n"
                "4. Możesz też wrócić do grafu i skorzystać z trybów naprawczych."
            )
            host._set_console_text(host.export_console, msg)

            try:
                if host._step3_linear_mode and CAMPAIGN.get_active_project_name():
                    CAMPAIGN.set_step3_needs_rework()
            except Exception:
                pass

            try:
                host._update_step3_finish_button_state()
            except Exception:
                pass

            try:
                host.app.update_status(
                    "Nie utworzono gold packa — pozostajesz w z3/pz3, aby kontynuować pracę.",
                    "warning"
                )
            except Exception:
                pass

            return

        split_stats = {}
        if split_enabled:
            ratios = {
                "train": train_pct / 100.0,
                "val": val_pct / 100.0,
                "test": test_pct / 100.0,
            }
            split_entries, split_preview_counts = host._build_split_entries(export_entries, ratios, shuffle_seed=42)
            preview_train = int(split_preview_counts.get("train", 0) or 0)
            preview_val = int(split_preview_counts.get("val", 0) or 0)
            preview_test = int(split_preview_counts.get("test", 0) or 0)
            if preview_train <= 0 or preview_val <= 0:
                msg = (
                    "❌ NIE UDAŁO SIĘ PRZYGOTOWAĆ POPRAWNEGO DATASETU DO TRENINGU.\n\n"
                    f"Do gold packa przeszło tylko {copied} tablic(y), więc przy splicie "
                    f"{train_pct:.0f}/{val_pct:.0f}/{test_pct:.0f} dostaniesz:\n"
                    f"train={preview_train}, val={preview_val}, test={preview_test}\n\n"
                    "To nie wystarczy do treningu, bo dataset musi mieć co najmniej 1 obraz w train i 1 obraz w val.\n\n"
                    "CO DALEJ:\n"
                    "1. Dodaj więcej tablic perfect do gold packa.\n"
                    "2. Zmień proporcje splitu tak, aby train i val nie były puste.\n"
                    "3. Wróć do PZ2 / CVAT i popraw więcej tablic."
                )
                host._set_console_text(host.export_console, msg)
                try:
                    failure_summary = host._build_step3_export_summary(
                        gold_dataset_path="",
                        review_pack_path="",
                        retry_pack_path="",
                        note="Gold pack znaków jest za mały, by przygotować poprawny split train / val / test."
                    )
                    failure_summary["gold_dataset_valid"] = False
                    failure_summary["gold_dataset_validation_message"] = msg
                    host._write_step3_export_summary(failure_summary)
                except Exception:
                    pass

                try:
                    if host._step3_linear_mode and CAMPAIGN.get_active_project_name():
                        CAMPAIGN.set_step3_needs_rework()
                except Exception:
                    pass

                try:
                    host._update_step3_finish_button_state()
                except Exception:
                    pass

                try:
                    host.app.update_status(
                        "Gold pack znaków jest jeszcze zbyt mały do poprawnego splitu train / val / test.",
                        "warning"
                    )
                except Exception:
                    pass

                return

            for split_name, items in split_entries.items():
                (yolo_out / "images" / split_name).mkdir(parents=True, exist_ok=True)
                (yolo_out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
                split_stats[split_name] = len(items)
                for item in items:
                    item["split"] = split_name
                    shutil.copy2(item["img_src"], yolo_out / "images" / split_name / f"{item['pid']}.jpg")
                    (yolo_out / "labels" / split_name / f"{item['pid']}.txt").write_text(
                        item["label_text"],
                        encoding="utf-8"
                    )
        else:
            img_out, lbl_out = yolo_out / "images", yolo_out / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for item in export_entries:
                item["split"] = "flat"
                shutil.copy2(item["img_src"], img_out / f"{item['pid']}.jpg")
                (lbl_out / f"{item['pid']}.txt").write_text(item["label_text"], encoding="utf-8")

        yaml_content = f"path: {yolo_out.absolute().as_posix()}\n"
        if split_enabled:
            yaml_content += "train: images/train\nval: images/val\ntest: images/test\n"
        else:
            yaml_content += "train: images\nval: images\n"
        yaml_content += "nc: 36\nnames:\n"
        for char, class_id in char_map.items():
            yaml_content += f"  {class_id}: '{char}'\n"
        (yolo_out / "data.yaml").write_text(yaml_content, encoding="utf-8")

        layout_counts = {}
        manifest_items = []
        for item in export_entries:
            layout_meta = item.get("layout", {}) if isinstance(item.get("layout", {}), dict) else {}
            layout_label = str(layout_meta.get("plate_layout_label", "?") or "?")
            layout_counts[layout_label] = int(layout_counts.get(layout_label, 0) or 0) + 1
            split_name = str(item.get("split", "flat") or "flat")
            manifest_items.append(
                {
                    "pid": str(item.get("pid", "")),
                    "source_pid": str(item.get("source_pid", "")),
                    "source_image": str(item.get("source_image", "")),
                    "split": split_name,
                    "image_path": (
                        f"images/{split_name}/{item.get('pid')}.jpg"
                        if split_enabled
                        else f"images/{item.get('pid')}.jpg"
                    ),
                    "label_path": (
                        f"labels/{split_name}/{item.get('pid')}.txt"
                        if split_enabled
                        else f"labels/{item.get('pid')}.txt"
                    ),
                    "strategy_bucket": str(item.get("strategy_bucket", "")),
                    "source_bucket": str(item.get("source_bucket", "")),
                    "layout": layout_meta,
                    "characters": list(item.get("characters", []) or []),
                }
            )
        exported_plate_count = int(len(manifest_items))
        exported_char_count = int(
            sum(len(item.get("characters", []) or []) for item in manifest_items)
        )
        layout_counts_text = (
            " | ".join(
                f"{str(label)}: {int(count)}"
                for label, count in sorted(layout_counts.items())
            )
            if layout_counts
            else "brak danych"
        )

        host._atomic_write_json(
            yolo_out / "metadata_manifest.json",
            {
                "dataset_type": "char_yolo_detect",
                "created_at": timestamp,
                "split_enabled": bool(split_enabled),
                "selected_strategies": sorted(selected_buckets),
                "selected_sources": sorted(selected_sources),
                "plate_count": int(exported_plate_count),
                "character_count": int(exported_char_count),
                "layout_counts": {str(key): int(value) for key, value in sorted(layout_counts.items())},
                "items": manifest_items,
            },
        )

        split_info_text = (
            f"train={split_stats.get('train', 0)}, val={split_stats.get('val', 0)}, test={split_stats.get('test', 0)}"
            if split_enabled
            else "źródło bez splitu (images/labels)"
        )

        export_summary = "\n".join(
            [
                "Proces zakończył się sukcesem: utworzono źródłowy dataset YOLO znaków.",
                f"Wyeksportowane tablice: {exported_plate_count}",
                f"Wyeksportowane znaki: {exported_char_count}",
                f"Układ tablic: {layout_counts_text}",
                f"Strategie: {selected_labels}",
                f"Źródła: {selected_source_labels}",
                f"Split: {split_info_text}",
                f"Ścieżka: {yolo_out}",
                (
                    "Dalej: przejdź do Z4, aby przygotować wariant treningowy i split, "
                    "albo zostań w Z3 / PZ3 i wykonaj kolejny eksport źródłowy."
                ),
            ]
        )
        host._set_console_text(host.export_console, export_summary)
        host._show_pz3_operation_summary_modal(
            "Eksport datasetu znaków zakończony",
            export_summary,
            tone="success",
        )
        try:
            success_summary = host._build_step3_export_summary(
                gold_dataset_path=str(yolo_out),
                review_pack_path="",
                retry_pack_path="",
                note="Bieżący eksport źródłowego gold packa znaków zakończył się powodzeniem. Wariant treningowy i split przygotuj w Z4."
            )
            success_summary["exportable_plate_count"] = int(exported_plate_count)
            success_summary["exportable_char_count"] = int(exported_char_count)
            success_summary["layout_counts"] = {
                str(key): int(value) for key, value in sorted(layout_counts.items())
            }
            success_summary["gold_dataset_valid"] = True
            success_summary["gold_dataset_validation_message"] = ""
            host._write_step3_export_summary(success_summary)
            try:
                from . import z3_campaign_flow

                z3_campaign_flow.mark_step3_dataset_exported_for_campaign(
                    host,
                    success_summary,
                    reason="dataset_exported",
                )
            except Exception:
                pass
        except Exception:
            pass
        host._log(
            host.export_console,
            f"[INFO] Dostępne tablice perfect we wszystkich zestawach przed deduplikacją i filtrem: {host._format_perfect_strategy_counts(total_strategy_counts)}",
            "INFO"
        )
        host._log(
            host.export_console,
            "[INFO] Dostępne tablice perfect wg źródeł: "
            + " | ".join(
                f"{gold_source_labels[key]}={int(total_source_counts.get(key, 0) or 0)}"
                for key, _ in gold_source_buckets
            ),
            "INFO"
        )
        try:
            host._update_step3_finish_button_state()
            host._set_button_emphasis("btn_finish_step3_frame", True)
        except Exception:
            pass

        try:
            host.app.update_status(
                "Źródłowy dataset YOLO został utworzony. Wariant treningowy i split przygotujesz w Z4.",
                "info"
            )
        except Exception:
            pass
    except Exception as e:
        host._set_console_text(host.export_console, f"❌ BŁĄD EKSPORTU YOLO:\n{e}")


def run_pz3_existing_dataset_split(host) -> None:
    src_raw = str(host.pz3_existing_dataset_var.get() or "").strip()
    source_dir = Path(src_raw) if src_raw else None
    source_info = host._inspect_pz3_dataset_source_dir(source_dir)

    if not source_info.get("ok"):
        host._set_console_text(
            host.export_console,
            "❌ Nie można wykonać nowego splitu.\n\n"
            + str(source_info.get("message") or "Wskaż poprawny dataset YOLO znaków.")
        )
        return

    split_enabled = bool(host.gold_export_split_var.get())
    train_pct, val_pct, test_pct = host._get_gold_export_split_percentages()
    if not split_enabled:
        host._set_console_text(
            host.export_console,
            "❌ Dla wskazanego datasetu włącz split train / val / test, aby utworzyć nowy podział."
        )
        return

    if source_dir is None:
        host._set_console_text(
            host.export_console,
            "❌ Brak źródła datasetu do podziału."
        )
        return

    ratios = {
        "train": train_pct / 100.0,
        "val": val_pct / 100.0,
        "test": test_pct / 100.0,
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_datasets_dir = host._get_step3_datasets_root_dir()
    base_datasets_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_datasets_dir / f"{source_dir.name}_Split_{timestamp}"

    host._set_console_text(
        host.export_console,
        "⌛ Przygotowuję nowy split datasetu...\n"
        f"Źródło: {source_dir}\n"
        f"Rozklad: train={train_pct:.0f}, val={val_pct:.0f}, test={test_pct:.0f}\n"
        f"Par obraz + etykieta: {int(source_info.get('total_pairs', 0) or 0)}"
    )

    try:
        ok, msg, stats = host._pz3_dataset_splitter.split_dataset(source_dir, out_dir, ratios)
        if not ok:
            host._set_console_text(
                host.export_console,
                f"❌ Nowy split nie powiódł się:\n{msg}"
            )
            return

        split_summary = host._build_step3_local_success_message(
            "Utworzono nowy split datasetu.",
            out_dir,
            "Przejdź do Z4, aby użyć nowego splitu w treningu, albo zostań w Z3 / PZ3 i wykonaj kolejny eksport.",
        )
        host._set_console_text(host.export_console, split_summary)
        host._show_pz3_operation_summary_modal(
            "Split datasetu zakończony",
            split_summary,
            tone="success",
        )
        try:
            success_summary = host._build_step3_export_summary(
                gold_dataset_path=str(out_dir),
                review_pack_path="",
                retry_pack_path="",
                note="Bieżący split datasetu znaków zakończył się powodzeniem."
            )
            success_summary["gold_dataset_valid"] = True
            success_summary["gold_dataset_validation_message"] = ""
            host._write_step3_export_summary(success_summary)
            try:
                from . import z3_campaign_flow

                z3_campaign_flow.mark_step3_dataset_exported_for_campaign(
                    host,
                    success_summary,
                    reason="dataset_split_exported",
                )
            except Exception:
                pass
        except Exception:
            pass
        try:
            host._update_step3_finish_button_state()
            host._set_button_emphasis("btn_finish_step3_frame", True)
        except Exception:
            pass
        try:
            host.app.update_status(
                "Nowy split datasetu znaków został zapisany w katalogu projektu.",
                "info"
            )
        except Exception:
            pass
    except Exception as exc:
        host._set_console_text(
            host.export_console,
            f"❌ Błąd nowego splitu datasetu:\n{exc}"
        )


def normalize_classification_char_crop_bbox(
    bbox,
    image_shape,
    *,
    pad_x_ratio: float = 0.12,
    pad_y_ratio: float = 0.16,
    min_pad: int = 2,
    min_size: int = 4,
):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except Exception:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    img_h, img_w = image_shape[:2]
    width = max(float(min_size), x2 - x1)
    height = max(float(min_size), y2 - y1)
    pad_x = max(int(round(width * float(pad_x_ratio))), int(min_pad))
    pad_y = max(int(round(height * float(pad_y_ratio))), int(min_pad))

    crop_x1 = max(0, min(int(img_w), int(x1) - pad_x))
    crop_y1 = max(0, min(int(img_h), int(y1) - pad_y))
    crop_x2 = max(0, min(int(img_w), int(x2) + pad_x))
    crop_y2 = max(0, min(int(img_h), int(y2) + pad_y))

    if crop_x2 - crop_x1 < int(min_size):
        crop_x2 = min(int(img_w), crop_x1 + int(min_size))
    if crop_y2 - crop_y1 < int(min_size):
        crop_y2 = min(int(img_h), crop_y1 + int(min_size))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]


def run_char_classification_export(
    host,
    *,
    char_class_alphabet: str,
    gold_source_buckets,
    gold_source_labels: dict,
) -> None:
    selected_buckets = host._get_selected_gold_export_strategy_buckets()
    selected_labels = host._format_selected_gold_export_strategy_labels()
    selected_sources = host._get_selected_gold_export_source_buckets()
    selected_source_labels = host._format_selected_gold_export_source_labels()
    split_enabled = False
    train_pct, val_pct, test_pct = host._get_gold_export_split_percentages()

    if not selected_buckets:
        host._set_console_text(
            host.export_console,
            "❌ Nie wybrano żadnej strategii perfect do eksportu znaków.\n\n"
            "Zaznacz co najmniej jedną z opcji: OCR exact, YOLO exact, OCR + YOLO rescue, YOLO boxy + OCR lub Manual / inne perfect."
        )
        return
    if not selected_sources:
        host._set_console_text(
            host.export_console,
            "❌ Nie wybrano żadnego źródła do eksportu znaków.\n\n"
            "Zaznacz Auto z runu lub Ręczne poprawki lokalne. Poprawki CVAT po imporcie są dołączane automatycznie."
        )
        return

    split_line = "Podział klasyfikacyjny: bez splitu w PZ3"
    host._set_console_text(
        host.export_console,
        f"⏳ Zbieranie poprawnych znaków...\nFiltr strategii: {selected_labels}\nŹródła: {selected_source_labels}\n{split_line}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_datasets_dir = host._get_step3_char_classification_datasets_root_dir(ensure_exists=True)
    out_dir = base_datasets_dir / f"CharClassificationDataset_{timestamp}"

    try:
        (
            plate_entries,
            total_strategy_counts,
            exported_plate_strategy_counts,
            total_source_counts,
            exported_source_counts,
        ) = host._collect_gold_export_plate_candidates(selected_buckets, selected_sources)
        if not plate_entries:
            msg = (
                "❌ NIE UDAŁO SIĘ UTWORZYĆ DATASETU ZNAKÓW.\n\n"
                f"Powód: po filtrze strategii ({selected_labels}) i źródeł ({selected_source_labels}) "
                "nie znaleziono ani jednej tablicy ze statusem perfect.\n\n"
                f"Dostępne tablice perfect wg strategii przed deduplikacją:\n{host._format_perfect_strategy_counts(total_strategy_counts)}\n"
                f"Dostępne tablice perfect wg źródeł: "
                f"{' | '.join(f'{gold_source_labels[key]}: {int(total_source_counts.get(key, 0) or 0)}' for key, _ in gold_source_buckets)}\n\n"
                "CO DALEJ:\n"
                "1. Możesz rozszerzyć zaznaczone strategie w pz3.\n"
                "2. Możesz wrócić do pz2 i poprawić wykrycia lub odczyt.\n"
                "3. Możesz wykonać eksport do CVAT i później zaimportować poprawki."
            )
            host._set_console_text(host.export_console, msg)
            return
        if not host._confirm_export_with_uncertain_layouts(plate_entries, export_label="klasyfikacja znaków"):
            host._set_console_text(
                host.export_console,
                "Eksport przerwany. Wróć do PZ2, użyj filtra 2R? i potwierdź układ tablic badge’em „Układ”."
            )
            return

        char_entries = []
        skipped_chars = 0

        for plate_entry in plate_entries:
            data = plate_entry.get("data", {})
            img_src = plate_entry.get("img_src")
            pid = str(plate_entry.get("pid", "") or "")
            strategy_bucket = str(plate_entry.get("strategy_bucket", "") or "")
            source_bucket = str(plate_entry.get("source_bucket", "") or "")
            layout_meta = host._build_plate_layout_export_metadata(data)

            img = cv2.imread(str(img_src))
            if img is None:
                continue

            for idx, rec in enumerate(list(data.get("characters", []) or [])):
                if not isinstance(rec, dict):
                    skipped_chars += 1
                    continue

                symbol = host._sanitize_preview_char_symbol(rec.get("character", ""))
                if not symbol:
                    skipped_chars += 1
                    continue

                crop_bbox = normalize_classification_char_crop_bbox(rec.get("bbox"), img.shape)
                if crop_bbox is None:
                    skipped_chars += 1
                    continue

                x1, y1, x2, y2 = crop_bbox
                crop = img[y1:y2, x1:x2]
                if crop is None or getattr(crop, "size", 0) == 0:
                    skipped_chars += 1
                    continue

                crop_name = f"{symbol}_{uuid.uuid4().hex[:10]}_{pid}_{idx:02d}.png"
                source_tag = host._get_character_source_tag(rec, data=data, fallback_index=idx)
                confidence = float(rec.get("confidence", 0.0) or 0.0)

                char_entries.append(
                    {
                        "symbol": symbol,
                        "filename": crop_name,
                        "image": crop,
                        "plate_id": pid,
                        "source_image": str(data.get("source_image", "") or ""),
                        "strategy_bucket": strategy_bucket,
                        "fusion_strategy": str(data.get("fusion_strategy", "") or ""),
                        "source_tag": str(source_tag or ""),
                        "source_bucket": source_bucket,
                        "source_kind": str(host._get_character_source_kind(rec, data=data)),
                        "confidence": confidence,
                        "bbox": [float(v) for v in (rec.get("bbox", []) or [])[:4]],
                        "reading_row": int(rec.get("reading_row", 0) or 0),
                        "reading_col": int(rec.get("reading_col", 0) or 0),
                        "reading_index": int(rec.get("reading_index", 0) or 0),
                        "layout": layout_meta,
                    }
                )

        if not char_entries:
            msg = (
                "❌ NIE UDAŁO SIĘ UTWORZYĆ DATASETU ZNAKÓW.\n\n"
                "Wybrano tablice perfect, ale nie udało się wyciąć żadnego poprawnego znaku z końcowych ramek."
            )
            host._set_console_text(host.export_console, msg)
            return

        split_entries = {}
        split_stats = {}
        if split_enabled:
            ratios = {
                "train": train_pct / 100.0,
                "val": val_pct / 100.0,
                "test": test_pct / 100.0,
            }
            split_entries, split_stats = host._build_split_entries(char_entries, ratios, shuffle_seed=42)
        else:
            split_entries = {"train": list(char_entries)}

        manifest_items = []
        saved_class_counts = {}
        for split_name, items in split_entries.items():
            split_stats[split_name] = len(items)
            for item in items:
                class_dir = out_dir / split_name / str(item["symbol"])
                class_dir.mkdir(parents=True, exist_ok=True)
                target_path = class_dir / str(item["filename"])
                if not cv2.imwrite(str(target_path), item["image"]):
                    continue

                saved_symbol = str(item["symbol"])
                saved_class_counts[saved_symbol] = int(saved_class_counts.get(saved_symbol, 0) or 0) + 1

                manifest_items.append(
                    {
                        "path": str(target_path.relative_to(out_dir).as_posix()),
                        "split": split_name,
                        "character": str(item["symbol"]),
                        "plate_id": str(item["plate_id"]),
                        "source_image": str(item["source_image"]),
                        "strategy_bucket": str(item["strategy_bucket"]),
                        "fusion_strategy": str(item["fusion_strategy"]),
                        "source_tag": str(item["source_tag"]),
                        "source_bucket": str(item["source_bucket"]),
                        "source_kind": str(item["source_kind"]),
                        "confidence": float(item["confidence"]),
                        "bbox": list(item["bbox"]),
                        "reading_row": int(item.get("reading_row", 0) or 0),
                        "reading_col": int(item.get("reading_col", 0) or 0),
                        "reading_index": int(item.get("reading_index", 0) or 0),
                        "layout": dict(item.get("layout", {}) or {}),
                    }
                )

        if not manifest_items:
            host._set_console_text(
                host.export_console,
                "❌ NIE UDAŁO SIĘ UTWORZYĆ DATASETU ZNAKÓW.\n\nNie udało się zapisać żadnego cropa znaku na dysku."
            )
            return

        present_classes = [char for char in char_class_alphabet if int(saved_class_counts.get(char, 0) or 0) > 0]
        layout_counts = {}
        for item in manifest_items:
            layout_meta = item.get("layout", {}) if isinstance(item.get("layout", {}), dict) else {}
            layout_label = str(layout_meta.get("plate_layout_label", "?") or "?")
            layout_counts[layout_label] = int(layout_counts.get(layout_label, 0) or 0) + 1
        manifest = {
            "dataset_type": "char_classification",
            "created_at": timestamp,
            "selected_strategies": sorted(selected_buckets),
            "selected_sources": sorted(selected_sources),
            "split_enabled": bool(split_enabled),
            "split_stats": {key: int(value) for key, value in split_stats.items()},
            "plate_count": int(len(plate_entries)),
            "character_count": int(len(manifest_items)),
            "skipped_characters": int(skipped_chars),
            "present_classes": list(present_classes),
            "class_counts": {char: int(saved_class_counts.get(char, 0) or 0) for char in char_class_alphabet if int(saved_class_counts.get(char, 0) or 0) > 0},
            "layout_counts": {str(key): int(value) for key, value in sorted(layout_counts.items())},
            "exported_plate_strategy_counts": dict(exported_plate_strategy_counts),
            "available_plate_strategy_counts": dict(total_strategy_counts),
            "exported_plate_source_counts": dict(exported_source_counts),
            "available_plate_source_counts": dict(total_source_counts),
            "items": manifest_items,
        }
        host._atomic_write_json(out_dir / "manifest.json", manifest)
        (out_dir / "classes.txt").write_text("\n".join(present_classes), encoding="utf-8")

        split_info = "Podział klasyfikacyjny: wszystkie znaki zapisano w train/"
        classifier_summary = host._build_step3_local_success_message(
            "Utworzono dataset znaków do klasyfikacji.",
            out_dir,
            "To dataset OCR/klasyfikacji znaków (manifest.json), nie wejście Z4/PZ1/PZ2 YOLO. "
            "Zostań w Z3 / PZ3 i wykonaj kolejny eksport albo użyj go w module klasyfikacyjnym.",
        )
        host._set_console_text(host.export_console, classifier_summary)
        host._show_pz3_operation_summary_modal(
            "Eksport datasetu klasyfikacyjnego zakończony",
            classifier_summary,
            tone="success",
        )
        host._log(
            host.export_console,
            f"[INFO] Dostępne tablice perfect we wszystkich zestawach przed deduplikacją i filtrem: {host._format_perfect_strategy_counts(total_strategy_counts)}",
            "INFO"
        )
        host._log(
            host.export_console,
            "[INFO] Dostępne tablice perfect wg źródeł: "
            + " | ".join(
                f"{gold_source_labels[key]}={int(total_source_counts.get(key, 0) or 0)}"
                for key, _ in gold_source_buckets
            ),
            "INFO"
        )
        host._log(
            host.export_console,
            f"[INFO] Rozkład klas i metadane źródła zapisano w: {out_dir / 'manifest.json'}",
            "INFO"
        )

        try:
            host._update_step3_finish_button_state()
            host._set_button_emphasis("btn_finish_step3_frame", True)
        except Exception:
            pass

        try:
            host.app.update_status(
                "Dataset znaków do klasyfikacji został utworzony poprawnie.",
                "info"
            )
        except Exception:
            pass
    except Exception as e:
        host._set_console_text(host.export_console, f"❌ BŁĄD EKSPORTU ZNAKÓW:\n{e}")


def refresh_gold_export_scope_label(host) -> None:
    selected_strategies = host._get_selected_gold_export_strategy_buckets()
    selected_sources = host._get_selected_gold_export_source_buckets()
    if not selected_strategies:
        set_gold_export_scope_info(host, "Do eksportu gold packa nie wybrano żadnej strategii.", "warning")
        return
    if not selected_sources:
        set_gold_export_scope_info(host, "Do eksportu gold packa nie wybrano żadnego źródła.", "warning")
        return

    selected_labels = host._format_selected_gold_export_strategy_labels()
    selected_source_labels = host._format_selected_gold_export_source_labels()
    contextual = host._build_campaign_aware_gold_export_counts(
        selected_strategies=selected_strategies,
        selected_sources=selected_sources,
    )
    selected_count = int(contextual.get("selected_plate_count", 0) or 0)
    selected_chars = int(contextual.get("selected_char_count", 0) or 0)
    layout_summary = host._format_plate_layout_counts(
        contextual.get("selected_layout_counts", {}),
        prefix="układ",
    )

    set_gold_export_scope_info(
        host,
        f"Do gold packa: strategie={selected_labels} | źródła={selected_source_labels} | "
        f"perfect={selected_count} | znaki={selected_chars} | {layout_summary} | "
        f"{format_gold_export_split_summary(host)}",
        "muted",
    )


def update_gold_export_split_labels(host) -> None:
    train_pct, val_pct, test_pct = host._get_gold_export_split_percentages()

    for widget_name, text in (
        ("gold_export_train_pct_lbl", f"{train_pct:.0f}%"),
        ("gold_export_val_pct_lbl", f"{val_pct:.0f}%"),
        ("gold_export_test_pct_lbl", f"Test: {test_pct:.0f}%"),
    ):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(text=text)
        except Exception:
            pass

    enabled = bool(getattr(host, "gold_export_split_var", None).get() if hasattr(host, "gold_export_split_var") else False)
    state = tk.NORMAL if enabled else tk.DISABLED
    for widget_name in ("gold_export_train_scale", "gold_export_val_scale"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(state=state)
        except Exception:
            pass

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    label_fg = palette.get("muted", palette.get("fg", "#f3f3f3"))

    for widget_name in ("gold_export_train_pct_lbl", "gold_export_val_pct_lbl", "gold_export_test_pct_lbl"):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        try:
            widget.configure(foreground=label_fg)
        except Exception:
            try:
                widget.configure(fg=label_fg, bg=panel_bg)
            except Exception:
                pass

    helper = getattr(host, "gold_export_test_hint_lbl", None)
    if helper is not None:
        try:
            helper.configure(foreground=label_fg)
        except Exception:
            try:
                helper.configure(fg=label_fg, bg=panel_bg)
            except Exception:
                pass


def update_gold_export_total_row(host, row_info: dict | None, *, plate_count: int, char_count: int) -> None:
    if not isinstance(row_info, dict):
        return

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f4c27a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#a0a0a0")
    has_data = int(plate_count or 0) > 0 or int(char_count or 0) > 0

    label_widget = row_info.get("label")
    plate_widget = row_info.get("plate_label")
    char_widget = row_info.get("char_label")
    badge_widget = row_info.get("badge")

    try:
        if label_widget is not None:
            label_widget.configure(text=str(row_info.get("base_label", "Suma wybranych") or "Suma wybranych"))
    except Exception:
        pass
    for widget, value in ((plate_widget, int(plate_count or 0)), (char_widget, int(char_count or 0))):
        if widget is None:
            continue
        try:
            widget.configure(text=str(value))
        except Exception:
            pass
    if badge_widget is not None:
        try:
            badge_widget.configure(text=("W EKSPORCIE" if has_data else "PUSTE"))
        except Exception:
            pass

    row_info["fg"] = fg if has_data else muted
    row_info["badge_bg"] = blend_hex_colors(success if has_data else warning, panel_bg, 0.74)
    row_info["badge_fg"] = success if has_data else warning
    row_info["border_color"] = blend_hex_colors(success if has_data else warning, panel_bg, 0.45)
    host._refresh_selection_row(row_info)


def refresh_gold_export_filter_labels(host) -> None:
    contextual = host._build_campaign_aware_gold_export_counts(
        selected_sources=host._get_selected_gold_export_source_buckets(),
    )
    strategy_counts = contextual.get("strategy_counts", {}) if isinstance(contextual, dict) else {}
    strategy_char_counts = contextual.get("strategy_char_counts", {}) if isinstance(contextual, dict) else {}
    palette = getattr(host.app, "palette", {})
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f4c27a")
    muted = palette.get("muted", "#a0a0a0")
    panel_bg = palette.get("panel", "#252526")

    for row_info in getattr(host, "gold_export_filter_rows", []):
        if not isinstance(row_info, dict):
            continue

        label_widget = row_info.get("label")
        plate_widget = row_info.get("plate_label")
        char_widget = row_info.get("char_label")
        badge_widget = row_info.get("badge")
        bucket_key = str(row_info.get("bucket_key", "") or "").strip()
        base_label = str(row_info.get("base_label", "") or "").strip()
        if label_widget is None or not bucket_key or not base_label:
            continue

        plate_count = int(strategy_counts.get(bucket_key, 0) or 0)
        char_count = int(strategy_char_counts.get(bucket_key, 0) or 0)
        selected = bool(row_info.get("selected_getter", lambda: False)())
        if not selected:
            badge_text = "POMINIĘTE"
            badge_bg = blend_hex_colors(muted, panel_bg, 0.82)
            badge_fg = muted
        elif plate_count > 0:
            badge_text = "WŁĄCZONE"
            badge_bg = blend_hex_colors(success, panel_bg, 0.76)
            badge_fg = success
        else:
            badge_text = "BRAK DANYCH"
            badge_bg = blend_hex_colors(warning, panel_bg, 0.78)
            badge_fg = warning

        try:
            label_widget.configure(text=base_label)
        except Exception:
            pass
        for widget, value in ((plate_widget, plate_count), (char_widget, char_count)):
            if widget is None:
                continue
            try:
                widget.configure(text=str(value))
            except Exception:
                pass
        if badge_widget is not None:
            try:
                badge_widget.configure(text=badge_text)
            except Exception:
                pass
        row_info["badge_bg"] = badge_bg
        row_info["badge_fg"] = badge_fg
        host._refresh_selection_row(row_info)

    selected_strategy_keys = host._get_selected_gold_export_strategy_buckets()
    update_gold_export_total_row(
        host,
        getattr(host, "gold_export_filter_total_row", None),
        plate_count=sum(int(strategy_counts.get(key, 0) or 0) for key in selected_strategy_keys),
        char_count=sum(int(strategy_char_counts.get(key, 0) or 0) for key in selected_strategy_keys),
    )


def refresh_gold_export_source_labels(host) -> None:
    contextual = host._build_campaign_aware_gold_export_counts(
        selected_strategies=host._get_selected_gold_export_strategy_buckets(),
    )
    source_counts = contextual.get("source_counts", {}) if isinstance(contextual, dict) else {}
    source_char_counts = contextual.get("source_char_counts", {}) if isinstance(contextual, dict) else {}
    palette = getattr(host.app, "palette", {})
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f4c27a")
    muted = palette.get("muted", "#a0a0a0")
    panel_bg = palette.get("panel", "#252526")

    for row_info in getattr(host, "gold_export_source_rows", []):
        if not isinstance(row_info, dict):
            continue

        label_widget = row_info.get("label")
        plate_widget = row_info.get("plate_label")
        char_widget = row_info.get("char_label")
        badge_widget = row_info.get("badge")
        bucket_key = str(row_info.get("bucket_key", "") or "").strip()
        base_label = str(row_info.get("base_label", "") or "").strip()
        if label_widget is None or not bucket_key or not base_label:
            continue

        plate_count = int(source_counts.get(bucket_key, 0) or 0)
        char_count = int(source_char_counts.get(bucket_key, 0) or 0)
        selected = bool(row_info.get("selected_getter", lambda: False)())
        locked = bool(row_info.get("locked", False))
        if locked and (plate_count > 0 or char_count > 0):
            badge_text = "AUTO"
            badge_bg = blend_hex_colors(success, panel_bg, 0.76)
            badge_fg = success
        elif locked:
            badge_text = "AUTO / BRAK"
            badge_bg = blend_hex_colors(muted, panel_bg, 0.84)
            badge_fg = muted
        elif not selected:
            badge_text = "POMINIĘTE"
            badge_bg = blend_hex_colors(muted, panel_bg, 0.82)
            badge_fg = muted
        elif plate_count > 0:
            badge_text = "WŁĄCZONE"
            badge_bg = blend_hex_colors(success, panel_bg, 0.76)
            badge_fg = success
        else:
            badge_text = "BRAK DANYCH"
            badge_bg = blend_hex_colors(warning, panel_bg, 0.78)
            badge_fg = warning

        try:
            label_widget.configure(text=base_label)
        except Exception:
            pass
        for widget, value in ((plate_widget, plate_count), (char_widget, char_count)):
            if widget is None:
                continue
            try:
                widget.configure(text=str(value))
            except Exception:
                pass
        if badge_widget is not None:
            try:
                badge_widget.configure(text=badge_text)
            except Exception:
                pass
        row_info["badge_bg"] = badge_bg
        row_info["badge_fg"] = badge_fg
        host._refresh_selection_row(row_info)

    selected_source_keys = host._get_selected_gold_export_source_buckets()
    update_gold_export_total_row(
        host,
        getattr(host, "gold_export_source_total_row", None),
        plate_count=sum(int(source_counts.get(key, 0) or 0) for key in selected_source_keys),
        char_count=sum(int(source_char_counts.get(key, 0) or 0) for key in selected_source_keys),
    )

    cvat_label = getattr(host, "gold_cvat_source_status_lbl", None)
    if cvat_label is not None:
        cvat_plate_count = int(source_counts.get("cvat_manual", 0) or 0)
        cvat_char_count = int(source_char_counts.get("cvat_manual", 0) or 0)
        if cvat_plate_count > 0 or cvat_char_count > 0:
            cvat_text = (
                f"Poprawki CVAT: {cvat_plate_count} tablic | {cvat_char_count} znaków. "
                "Zaimportowane poprawki są automatycznie dołączane do budowy datasetu."
            )
            cvat_tone = "success"
            cvat_emphasis = True
        else:
            cvat_text = (
                "Poprawki CVAT: brak zaimportowanych poprawek. "
                "Jeśli wczytasz XML z CVAT, ten wkład zostanie automatycznie użyty w datasecie."
            )
            cvat_tone = "muted"
            cvat_emphasis = False
        host._set_inline_status_label_state(
            cvat_label,
            text=cvat_text,
            tone=cvat_tone,
            emphasis=cvat_emphasis,
        )


def _copy_gold_candidate_for_normalization(data):
    # Raw detector proposals are immutable diagnostics, often 300 per plate.
    # Preserve them in the export payload without copying/reclassifying them
    # for every status query; normalization only edits the final characters.
    editable = {"characters", "source_info", "gold_state", "plate_attributes", "layout_separator"}
    return {key: copy.deepcopy(value) if key in editable else value for key, value in data.items()}


def ensure_plate_source_metadata(
    host,
    data: dict,
    *,
    plate_id: str = "",
    meta_path: Path | None = None,
    default_bucket: str | None = None,
    default_origin: str | None = None,
    import_batch_id: str = "",
    review_manifest_id: str = "",
    modified_by: str = "",
    include_diagnostics: bool = True,
) -> bool:
    self = host
    if not isinstance(data, dict):
        return False

    changed = False
    source_info = data.get("source_info")
    if not isinstance(source_info, dict):
        source_info = {}
        data["source_info"] = source_info
        changed = True

    bucket = self._normalize_plate_source_bucket(
        default_bucket if default_bucket is not None else source_info.get("bucket", ""),
        data=data,
        meta_path=meta_path,
    )
    if str(source_info.get("bucket", "") or "") != bucket:
        source_info["bucket"] = bucket
        changed = True

    origin = self._normalize_plate_source_origin(
        default_origin if default_origin is not None else source_info.get("origin", ""),
        bucket=bucket,
    )
    if str(source_info.get("origin", "") or "") != origin:
        source_info["origin"] = origin
        changed = True

    run_id = str(source_info.get("run_id", "") or "").strip()
    if not run_id:
        candidate_run_id = ""
        if meta_path is not None:
            try:
                candidate_run_id = str(meta_path.parent.name or "").strip()
            except Exception:
                candidate_run_id = ""
        if not candidate_run_id:
            candidate_run_id = str(getattr(self, "preview_dir_var", None).get() if hasattr(self, "preview_dir_var") else "").strip()
            candidate_run_id = Path(candidate_run_id).name if candidate_run_id else ""
        if candidate_run_id:
            source_info["run_id"] = candidate_run_id
            changed = True

    if review_manifest_id and str(source_info.get("review_manifest_id", "") or "").strip() != str(review_manifest_id):
        source_info["review_manifest_id"] = str(review_manifest_id)
        changed = True
    elif "review_manifest_id" not in source_info:
        source_info["review_manifest_id"] = str(source_info.get("review_manifest_id", "") or "")
        changed = True

    if import_batch_id and str(source_info.get("import_batch_id", "") or "").strip() != str(import_batch_id):
        source_info["import_batch_id"] = str(import_batch_id)
        changed = True
    elif "import_batch_id" not in source_info:
        source_info["import_batch_id"] = str(source_info.get("import_batch_id", "") or "")
        changed = True

    if "last_modified_at" not in source_info:
        source_info["last_modified_at"] = str(source_info.get("last_modified_at", "") or "")
        changed = True
    if "last_modified_by" not in source_info:
        source_info["last_modified_by"] = str(source_info.get("last_modified_by", "") or "")
        changed = True

    if modified_by:
        from datetime import datetime
        now = datetime.now().isoformat(timespec="seconds")
        if str(source_info.get("last_modified_by", "") or "") != str(modified_by):
            source_info["last_modified_by"] = str(modified_by)
            changed = True
        if str(source_info.get("last_modified_at", "") or "") != now:
            source_info["last_modified_at"] = now
            changed = True

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
        changed = True

    status = str(data.get("status", "unknown") or "unknown").strip().lower()
    candidate = bool(status == "perfect")
    if bool(gold_state.get("candidate", False)) != candidate:
        gold_state["candidate"] = candidate
        changed = True

    approved = bool(gold_state.get("approved", False))
    if meta_path is not None:
        path_parts = {part.lower() for part in meta_path.parts}
        if "gold_char_pool" in path_parts:
            approved = True
    if bool(gold_state.get("approved", False)) != approved:
        gold_state["approved"] = approved
        changed = True

    approved_from_bucket = str(gold_state.get("approved_from_bucket", "") or "").strip()
    if approved and not approved_from_bucket:
        gold_state["approved_from_bucket"] = bucket
        changed = True
    elif not approved and "approved_from_bucket" not in gold_state:
        gold_state["approved_from_bucket"] = approved_from_bucket
        changed = True

    excluded = bool(gold_state.get("excluded", False))
    if bool(gold_state.get("excluded", False)) != excluded:
        gold_state["excluded"] = excluded
        changed = True
    if "excluded_reason" not in gold_state:
        gold_state["excluded_reason"] = str(gold_state.get("excluded_reason", "") or "")
        changed = True

    list_keys = ("characters", "yolo_detections", "yolo_nms_detections", "yolo_raw_detections") if include_diagnostics else ("characters",)
    for list_key in list_keys:
        records = data.get(list_key)
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            source_kind = self._normalize_character_source_kind(rec, data=data, plate_source_bucket=bucket)
            if str(rec.get("source_kind", "") or "") != source_kind:
                rec["source_kind"] = source_kind
                changed = True
            if "source_batch_id" not in rec:
                rec["source_batch_id"] = ""
                changed = True
            if import_batch_id and source_kind == "cvat_manual" and str(rec.get("source_batch_id", "") or "") != str(import_batch_id):
                rec["source_batch_id"] = str(import_batch_id)
                changed = True

    return changed
