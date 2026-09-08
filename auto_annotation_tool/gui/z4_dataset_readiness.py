#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared readiness checks for Z4 training dataset variants."""

from __future__ import annotations

import json
from pathlib import Path


MZ_TRAINING_VARIANT_MANIFEST = "mz_training_variant_manifest.json"
MZ_READY_STATUSES = {
    "REPRESENTATION_OK",
    "REPRESENTATION_OK_WITH_DIVERSITY_WARNING",
}
MZ_BLOCKING_STATUSES = {
    "PLAN_MISSING",
    "PLAN_DATASET_MISMATCH",
    "PLAN_NOT_FEASIBLE",
    "TARGET_NOT_REACHED",
    "VAL_TEST_CHANGED",
    "PARTIAL_AUGMENTATION",
    "AUGMENTATION_FAILED",
}


def normalize_dataset_variant_root(value: str | Path | None) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
        if path.is_file() and path.name.lower() == "data.yaml":
            path = path.parent
        return path
    except Exception:
        return None


def load_mz_training_variant_manifest(dataset_path: str | Path | None) -> dict:
    root = normalize_dataset_variant_root(dataset_path)
    if root is None:
        return {}
    manifest_path = root / MZ_TRAINING_VARIANT_MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def get_training_dataset_readiness(dataset_path: str | Path | None, *, target: str | None = None) -> dict:
    """Return whether a dataset variant is allowed to be used by Z4/PZ2."""

    normalized_target = str(target or "").strip().lower()
    if normalized_target not in {"char", "znaki", "mz"}:
        return {"ok": True, "message": "", "manifest": {}}

    manifest = load_mz_training_variant_manifest(dataset_path)
    if not manifest:
        # Backward compatible: older character datasets did not yet have this manifest.
        return {"ok": True, "message": "", "manifest": {}}

    ready_raw = manifest.get("ready_for_training", None)
    completion_status = str(manifest.get("completion_status") or "").strip().upper()
    representation_status = str(manifest.get("representation_status") or "").strip().upper()
    freeze_status = str(manifest.get("freeze_status") or "").strip().upper()
    execution_status = str(manifest.get("execution_status") or "").strip().upper()
    deficits = dict(manifest.get("deficit_after") or {})
    augmentation_mode = str(manifest.get("augmentation_mode") or "").strip().lower()
    balance_managed = (
        augmentation_mode == "mz_auto_representation"
        or bool(manifest.get("approved_balance_plan"))
        or completion_status in MZ_BLOCKING_STATUSES
        or representation_status in {"TARGET_NOT_REACHED"}
        or freeze_status == "VAL_TEST_CHANGED"
    )

    if ready_raw is False:
        return {
            "ok": False,
            "message": _format_mz_readiness_block_message(manifest),
            "manifest": manifest,
        }
    if ready_raw is True:
        return {"ok": True, "message": "", "manifest": manifest}

    # Compatibility path for manifests created just before ready_for_training existed.
    if not balance_managed:
        return {"ok": True, "message": "", "manifest": manifest}
    if completion_status in MZ_BLOCKING_STATUSES:
        return {"ok": False, "message": _format_mz_readiness_block_message(manifest), "manifest": manifest}
    if freeze_status and freeze_status != "UNCHANGED":
        return {"ok": False, "message": _format_mz_readiness_block_message(manifest), "manifest": manifest}
    if representation_status and representation_status not in MZ_READY_STATUSES:
        return {"ok": False, "message": _format_mz_readiness_block_message(manifest), "manifest": manifest}
    if execution_status and execution_status != "COMPLETED":
        return {"ok": False, "message": _format_mz_readiness_block_message(manifest), "manifest": manifest}
    if deficits:
        return {"ok": False, "message": _format_mz_readiness_block_message(manifest), "manifest": manifest}
    return {"ok": True, "message": "", "manifest": manifest}


def _format_mz_readiness_block_message(manifest: dict) -> str:
    status = str(manifest.get("completion_status") or manifest.get("representation_status") or "").strip().upper()
    deficits = dict(manifest.get("deficit_after") or {})
    if status == "VAL_TEST_CHANGED" or str(manifest.get("freeze_status") or "").strip().upper() == "VAL_TEST_CHANGED":
        return "Ten wariant nie jest gotowy do treningu, bo naruszono freeze splitu val/test."
    if status == "PLAN_NOT_FEASIBLE":
        return "Ten wariant jest raportem diagnostycznym: reprezentacji MZ nie da się osiągnąć przy aktualnym materiale."
    if status == "PLAN_MISSING":
        return "Ten wariant nie ma zatwierdzonego planu reprezentacji MZ dla finalnego splitu."
    if status == "PLAN_DATASET_MISMATCH":
        return "Ten wariant ma plan MZ policzony dla innego datasetu."
    if deficits:
        missing = ", ".join(f"{key}: {value}" for key, value in sorted(deficits.items()))
        return f"Ten wariant nie jest gotowy do treningu, bo po augmentacji nadal ma braki MZ: {missing}."
    if str(manifest.get("execution_status") or "").strip().upper() not in {"", "COMPLETED"}:
        return "Ten wariant nie jest gotowy do treningu, bo augmentacja nie została wykonana w całości."
    return "Ten wariant nie jest oznaczony jako gotowy do treningu."
