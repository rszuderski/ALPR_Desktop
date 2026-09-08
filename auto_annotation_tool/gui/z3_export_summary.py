#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read/write helpers for Z3 export summary artifacts."""

import json
from pathlib import Path
from datetime import datetime

from ..campaign_manager import CAMPAIGN


def _with_summary_location(summary: dict, summary_path: Path) -> dict:
    enriched = dict(summary or {})
    try:
        enriched["_summary_path"] = str(summary_path)
        enriched["_summary_dir"] = str(summary_path.parent)
    except Exception:
        pass
    return enriched


def _path_is_inside(path_like, root_like) -> bool:
    if not path_like or not root_like:
        return True
    try:
        Path(path_like).resolve().relative_to(Path(root_like).resolve())
        return True
    except Exception:
        return False


def _summary_matches_active_project(summary: dict, *, active_project: str = "", project_root=None) -> bool:
    if not active_project:
        return True
    data = dict(summary or {})
    summary_project = str(data.get("project", "") or "").strip()
    if summary_project and summary_project != active_project:
        return False
    for key in ("gold_dataset_path", "_summary_path", "_summary_dir"):
        raw = str(data.get(key) or "").strip()
        if raw and not _path_is_inside(raw, project_root):
            return False
    return True


def build_step3_export_summary(
    host,
    gold_dataset_path: str | None = None,
    review_pack_path: str | None = None,
    retry_pack_path: str | None = None,
    note: str = "",
) -> dict:
    counts = host._count_preview_statuses()
    exportable_char_count = int(
        sum(int(value or 0) for value in dict(counts.get("strategy_char_counts", {}) or {}).values())
    )
    exportable_plate_count = host._count_exportable_perfect_plates_in_metadata(host.preview_metadata)

    gold_exists = bool(gold_dataset_path and Path(gold_dataset_path).exists())
    review_exists = bool(review_pack_path and Path(review_pack_path).exists())
    retry_exists = bool(retry_pack_path and Path(retry_pack_path).exists())
    try:
        campaign_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        campaign_iteration = 0
    try:
        campaign_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        campaign_project = ""
    try:
        campaign_project_root = str(CAMPAIGN.get_active_project_root_dir() or "")
    except Exception:
        campaign_project_root = ""

    return {
        "project": campaign_project,
        "project_root": campaign_project_root,
        "gold_dataset_created": gold_exists,
        "gold_dataset_path": str(gold_dataset_path or ""),
        "review_pack_created": review_exists,
        "review_pack_path": str(review_pack_path or ""),
        "retry_pack_created": retry_exists,
        "retry_pack_path": str(retry_pack_path or ""),
        "perfect_count": counts["perfect"],
        "needs_fix_count": counts["needs_fix"],
        "unknown_count": counts["unknown"],
        "total_count": counts["total"],
        "exportable_plate_count": int(exportable_plate_count),
        "exportable_char_count": int(exportable_char_count),
        "perfect_strategy_counts": counts.get("strategy_counts", host._empty_perfect_strategy_counts()),
        "perfect_strategy_char_counts": counts.get("strategy_char_counts", host._empty_perfect_strategy_counts()),
        "selected_gold_export_strategies": sorted(host._get_selected_gold_export_strategy_buckets()),
        "iteration": int(campaign_iteration or 0),
        "source_iteration": int(campaign_iteration or 0),
        "created_iteration": int(campaign_iteration or 0),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }


def write_step3_export_summary(host, summary: dict) -> Path:
    summary_dir = host._get_step3_summary_dir()
    summary_path = summary_dir / "export_summary.json"
    host._atomic_write_json(summary_path, summary)
    return summary_path


def read_step3_export_summary(host) -> dict:
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False
    try:
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip() if in_campaign else ""
    except Exception:
        active_project = ""
    try:
        project_root = CAMPAIGN.get_active_project_root_dir() if in_campaign else None
    except Exception:
        project_root = None

    summary_path = host._get_step3_summary_dir() / "export_summary.json"
    try:
        if summary_path.exists():
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded:
                enriched = _with_summary_location(loaded, summary_path)
                if _summary_matches_active_project(
                    enriched,
                    active_project=active_project,
                    project_root=project_root,
                ):
                    return enriched
    except Exception:
        pass

    try:
        campaign_chars_dir = getattr(host, "_campaign_chars_dir", None)
        if campaign_chars_dir and (
            not in_campaign or _path_is_inside(campaign_chars_dir, project_root)
        ):
            chars_root = Path(campaign_chars_dir)
        elif in_campaign:
            chars_root = CAMPAIGN.get_dir("chars")
        else:
            chars_root = host._get_step3_chars_root_dir(ensure_exists=False)
    except Exception:
        chars_root = None

    if chars_root is None or not Path(chars_root).exists():
        return {}

    try:
        candidates = sorted(
            Path(chars_root).rglob("export_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return {}

    for candidate in candidates:
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(loaded, dict) or not loaded:
            continue
        if bool(loaded.get("gold_dataset_created")) and bool(loaded.get("gold_dataset_valid", True)):
            enriched = _with_summary_location(loaded, candidate)
            if _summary_matches_active_project(
                enriched,
                active_project=active_project,
                project_root=project_root,
            ):
                return enriched
    return {}


def inspect_yolo_dataset_label_objects(dataset_dir: Path | None) -> dict:
    result = {"label_files": 0, "label_files_with_objects": 0, "objects": 0}
    if dataset_dir is None:
        return result
    try:
        labels_root = Path(dataset_dir) / "labels"
        if not labels_root.exists() or not labels_root.is_dir():
            return result
        for label_path in labels_root.rglob("*.txt"):
            if not label_path.is_file():
                continue
            result["label_files"] += 1
            try:
                lines = [
                    line.strip()
                    for line in label_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            except Exception:
                lines = []
            if lines:
                result["label_files_with_objects"] += 1
                result["objects"] += len(lines)
    except Exception:
        return result
    return {key: int(value or 0) for key, value in result.items()}
