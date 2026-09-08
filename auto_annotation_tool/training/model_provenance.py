#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training provenance builder for exported ALPR models.

The desktop application owns the training lineage.  Android stores and reports
this payload, but must not reconstruct parent runs or epoch totals.
"""

from __future__ import annotations

import hashlib
import datetime as _dt
import csv
import json
import re
import zipfile
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import CONFIG, get_torch_module
from ..utils import safe_load_yaml


PROVENANCE_VERSION = 2
TOTAL_EPOCHS_SCOPE = "project_training_after_pretrained_base"
_RUN_ID_RE = re.compile(r"(20\d{6}_\d{6})")
_IMAGE_SUFFIXES = {str(ext).lower() for ext in getattr(CONFIG, "IMAGE_EXTENSIONS", (".jpg", ".jpeg", ".png", ".bmp", ".webp"))}
_SIDE_CAR_NAMES = (
    "{stem}{suffix}.metadata.json",
    "{stem}.metadata.json",
    "{stem}_metadata.json",
    "model_metadata.json",
    "metadata.json",
)
_DATASET_MANIFEST_NAMES = (
    "training_variant_manifest.json",
    "dataset_manifest.json",
    "augmentation_manifest.json",
    "stage_manifest.json",
    "manifest.json",
)


@dataclass
class _EpochLineageResult:
    total_epochs: int | None
    known_epochs_minimum: int
    total_epochs_known: bool
    lineage_stage_count_known: bool
    known_stage_count_minimum: int
    provenance_status: str
    lineage: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_model_training_provenance(
    run_like: Any,
    *,
    history_index: Mapping[str, Any] | None = None,
    checkpoint: Path | str | None = None,
    dataset_path: Path | str | None = None,
    target: str = "",
    model_sidecar: Mapping[str, Any] | None = None,
    include_dataset_fingerprint: bool = True,
) -> dict[str, Any]:
    """Return the canonical training/provenance payload for a model manifest."""

    run = _run_like_dict(run_like)
    sidecar = dict(model_sidecar or {}) if isinstance(model_sidecar, Mapping) else {}
    if not run:
        run = _legacy_run_from_sidecar(sidecar)

    index = _normalize_history_index(history_index)
    run = _resolve_legacy_parent(run, index)
    resolved_target = _normalize_target(target or _value(run, "training_target") or _value(run, "parent_model_target") or _infer_target_from_text(
        " ".join(str(_value(run, key, "")) for key in ("dataset_path", "base_model", "name", "output_dir"))
    ))
    resolved_dataset_path = str(dataset_path or _value(run, "dataset_path", "") or "").strip()
    dataset, provenance_capture = _dataset_provenance_for_run(
        run,
        fallback_dataset_path=resolved_dataset_path,
        target=resolved_target,
        include_content_fingerprint=include_dataset_fingerprint,
    )

    checkpoint_path = Path(checkpoint) if checkpoint else _path_or_none(_value(run, "best_weights") or _value(run, "last_weights"))
    lineage_result = _lineage_total_epochs(
        run,
        history_index=index,
        current_checkpoint=checkpoint_path,
        visited=set(),
    ) if run else _EpochLineageResult(
        total_epochs=None,
        known_epochs_minimum=0,
        total_epochs_known=False,
        lineage_stage_count_known=False,
        known_stage_count_minimum=0,
        provenance_status="legacy_unknown",
        lineage=[],
        warnings=["Brak historii runu treningowego dla modelu."],
    )

    run_epochs_completed = _completed_epoch_count(run)
    run_epochs_planned = _int_or_none(_value(run, "epochs"))
    lineage_mode = _normalize_lineage_mode(_value(run, "lineage_mode", "new"))
    pretrained_origin = _pretrained_origin(run)
    status = lineage_result.provenance_status
    warnings = list(lineage_result.warnings)
    dataset_status = str(dataset.get("provenance_status") or "").strip().lower()
    if provenance_capture != "frozen_at_training_start":
        status = _weaken_status(status)
        if provenance_capture == "reconstructed_at_export":
            warnings.append("Dataset treningowy odtworzono z aktualnego katalogu podczas eksportu.")
        elif provenance_capture == "reconstructed_from_training_artifacts":
            warnings.append("Dataset treningowy odtworzono z historycznych artefaktow, nie z zamrozonego snapshotu.")
        elif provenance_capture == "legacy_unknown":
            status = "legacy_unknown" if not run else _weaken_status(status)
            warnings.append("Brak wiarygodnego snapshotu datasetu treningowego.")
    if dataset_status == "legacy_unknown":
        status = "legacy_unknown" if not run else _weaken_status(status)
    elif dataset_status == "partial":
        status = _weaken_status(status)
    if resolved_dataset_path and not dataset.get("manifest_sha256"):
        status = _weaken_status(status)
        warnings.append("Dataset treningowy nie ma jawnego manifestu generatora.")
    if include_dataset_fingerprint and resolved_dataset_path and not dataset.get("split_sha256"):
        status = _weaken_status(status)
        warnings.append("Nie udało się policzyć fingerprintu splitu datasetu treningowego.")

    input_checkpoint_snapshot = _checkpoint_snapshot_from_run(run, "input_checkpoint_snapshot")
    output_checkpoint_snapshot = _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
    input_sha = _checkpoint_snapshot_sha(input_checkpoint_snapshot) or _file_sha256(
        _path_or_none(_value(run, "parent_model_path") or _value(run, "base_model"))
    )
    if run and not input_sha:
        status = _weaken_status(status)
        warnings.append("Nie udalo sie zamrozic sumy SHA-256 wejsciowego checkpointu.")
    output_mismatch_warning = _checkpoint_mismatch_warning(
        output_checkpoint_snapshot,
        checkpoint_path,
        label="best.pt",
    )
    if output_mismatch_warning:
        status = _weaken_status(status)
        warnings.append(output_mismatch_warning)
    frozen_best_sha = _checkpoint_snapshot_sha(output_checkpoint_snapshot, key="best") or _checkpoint_snapshot_sha(output_checkpoint_snapshot)
    frozen_last_sha = _checkpoint_snapshot_sha(output_checkpoint_snapshot, key="last")
    current_best_sha = _file_sha256(checkpoint_path)
    current_last_sha = _file_sha256(_path_or_none(_value(run, "last_weights")))
    output_best_sha = frozen_best_sha or current_best_sha
    output_last_sha = frozen_last_sha or current_last_sha
    if run and not frozen_best_sha and current_best_sha:
        status = _weaken_status(status)
        warnings.append("SHA-256 checkpointu best.pt odtworzono z aktualnego pliku, bez zamrozonego snapshotu po treningu.")
    elif run and not output_best_sha:
        status = _weaken_status(status)
        warnings.append("Brak wiarygodnego SHA-256 checkpointu wynikowego best.pt.")

    run_train_images = _train_images_from_dataset(dataset)
    run_nominal_sample_presentations = _nominal_sample_presentations(run_train_images, run_epochs_completed if run else None)
    (
        lineage_nominal_sample_presentations,
        sample_presentations_known,
        known_sample_presentations_minimum,
    ) = _lineage_sample_presentation_summary(lineage_result.lineage)
    if not lineage_result.total_epochs_known:
        sample_presentations_known = False
        lineage_nominal_sample_presentations = None
    checkpoint_metrics = build_checkpoint_metric_summary(run, checkpoint=checkpoint_path)
    best_epoch = checkpoint_metrics["best_epoch"]
    best_epoch_source = checkpoint_metrics["best_epoch_source"]
    if best_epoch is not None and best_epoch_source == "checkpoint" and not output_mismatch_warning:
        if output_checkpoint_snapshot.get("best_epoch") not in (None, best_epoch):
            output_checkpoint_snapshot["historical_best_epoch"] = output_checkpoint_snapshot["best_epoch"]
        output_checkpoint_snapshot["best_epoch"] = best_epoch
        output_checkpoint_snapshot["best_epoch_source"] = best_epoch_source

    parent_run_id = _explicit_parent_run_id(run)
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "run_id": str(_value(run, "id", "") or ""),
        "run_name": str(_value(run, "name", "") or ""),
        "mode": "fine_tune" if lineage_mode == "fine_tune" else ("legacy_unknown" if not run else "new"),
        "lineage_mode": lineage_mode,
        "run_epochs_planned": run_epochs_planned,
        "run_epochs_completed": run_epochs_completed if run else None,
        "epochs": run_epochs_planned,
        "current_epoch": run_epochs_completed if run else None,
        "best_epoch": best_epoch,
        "best_epoch_source": best_epoch_source,
        "total_epochs": lineage_result.total_epochs,
        "total_epochs_known": bool(lineage_result.total_epochs_known),
        "total_epochs_scope": TOTAL_EPOCHS_SCOPE,
        "known_epochs_minimum": int(lineage_result.known_epochs_minimum or 0),
        "lineage_total_epochs": lineage_result.total_epochs,
        "lineage_total_epochs_known": bool(lineage_result.total_epochs_known),
        "pretrained": bool(pretrained_origin),
        "pretrained_origin": pretrained_origin,
        "parent_run_id": parent_run_id,
        "parent_resolution": str(run.get("_parent_resolution") or ("explicit" if parent_run_id else "")),
        "parent_model_path": str(_value(run, "parent_model_path", "") or ""),
        "parent_model_name": str(_value(run, "parent_model_name", "") or ""),
        "lineage_depth": len(lineage_result.lineage),
        "lineage_stage_count": len(lineage_result.lineage),
        "lineage_stage_count_known": bool(lineage_result.lineage_stage_count_known),
        "known_stage_count_minimum": int(lineage_result.known_stage_count_minimum or 0),
        "run_train_images": run_train_images,
        "run_nominal_sample_presentations": run_nominal_sample_presentations,
        "lineage_nominal_sample_presentations": lineage_nominal_sample_presentations,
        "sample_presentations_known": bool(sample_presentations_known),
        "known_sample_presentations_minimum": int(known_sample_presentations_minimum or 0),
        "provenance_capture": provenance_capture,
        "lineage": lineage_result.lineage,
        "dataset": dataset,
        "dataset_id": str(dataset.get("dataset_id") or ""),
        "dataset_path": str(dataset.get("local_path_hint") or resolved_dataset_path),
        "input_dataset_snapshot": _mapping_copy(_value(run, "training_dataset_input_snapshot")),
        "dataset_preparation": _mapping_copy(_value(run, "dataset_preparation")),
        "input_checkpoint": input_checkpoint_snapshot,
        "input_checkpoint_sha256": input_sha,
        "output_checkpoint": output_checkpoint_snapshot,
        "best_checkpoint_sha256": output_best_sha,
        "last_checkpoint_sha256": output_last_sha,
        "base_model": str(_value(run, "base_model", "") or ""),
        "img_size": _int_or_none(_value(run, "img_size")),
        "batch_size": _int_or_none(_value(run, "batch_size")),
        "started_at": str(_value(run, "started_at", "") or ""),
        "finished_at": str(_value(run, "finished_at", "") or ""),
        "created_at": str(_value(run, "created_at", "") or ""),
        "provenance_status": status,
        "warnings": warnings,
    }
    return _json_safe(payload)


def build_dataset_training_provenance(
    dataset_path: Path | str | None,
    *,
    target: str = "",
    include_content_fingerprint: bool = True,
) -> dict[str, Any]:
    """Describe the training dataset without using local paths as the identity."""

    root = _dataset_root(dataset_path)
    if root is None:
        return {
            "dataset_id": "",
            "name": "",
            "target": _normalize_target(target),
            "provenance_status": "legacy_unknown",
        }
    yaml_path = root / "data.yaml"
    cfg = _safe_yaml(yaml_path)
    resolved_target = _normalize_target(target) or _infer_dataset_target(root, cfg)
    counts = _dataset_split_counts(root, cfg)
    data_yaml_sha = _file_sha256(yaml_path)
    manifests = _dataset_manifest_refs(root)
    manifest_sha = _json_sha256(manifests) if manifests else ""
    split_sha = ""
    split_file_count = 0
    if include_content_fingerprint:
        split_fingerprint = _dataset_split_fingerprint(root, cfg)
        split_sha = str(split_fingerprint.get("sha256") or "")
        split_file_count = int(split_fingerprint.get("file_count", 0) or 0)
    augmentation_meta = _dataset_augmentation_summary(root, manifests)
    identity_payload = {
        "target": resolved_target or "unknown",
        "manifest_sha256": manifest_sha,
        "data_yaml_sha256": data_yaml_sha,
        "split_sha256": split_sha,
    }
    identity_seed = _json_sha256(identity_payload) or str(root.resolve() if root.exists() else root)
    dataset_id = f"DS-{_target_code(resolved_target)}-{identity_seed[:10].upper()}" if identity_seed else ""
    total_images = int(counts.get("train", 0) or 0) + int(counts.get("val", 0) or 0) + int(counts.get("test", 0) or 0)
    return _json_safe(
        {
            "dataset_id": dataset_id,
            "name": root.name,
            "target": resolved_target or "unknown",
            "manifest_sha256": manifest_sha,
            "data_yaml_sha256": data_yaml_sha,
            "split_sha256": split_sha,
            "split_file_count": split_file_count,
            "train_images": int(counts.get("train", 0) or 0),
            "val_images": int(counts.get("val", 0) or 0),
            "test_images": int(counts.get("test", 0) or 0),
            "total_images": total_images,
            "source_images": int(augmentation_meta.get("source_images", 0) or total_images),
            "source_objects": int(augmentation_meta.get("source_objects", 0) or 0),
            "offline_augmentation_train_added": int(augmentation_meta.get("offline_augmentation_train_added", 0) or 0),
            "split_seed": augmentation_meta.get("split_seed"),
            "manifests": manifests,
            "local_path_hint": str(root),
            "data_yaml": str(yaml_path) if yaml_path.exists() else "",
            "dataset_id_strategy": "composite_v2",
            "provenance_status": "complete" if manifest_sha and split_sha else "partial",
        }
    )


def build_training_dataset_snapshot(
    dataset_path: Path | str | None,
    *,
    target: str = "",
) -> dict[str, Any]:
    """Freeze dataset identity before training starts."""

    snapshot = dict(
        build_dataset_training_provenance(
            dataset_path,
            target=target,
            include_content_fingerprint=True,
        )
        or {}
    )
    snapshot.setdefault("schema", "alpr.training_dataset_snapshot.v1")
    snapshot.setdefault("captured_at", _utc_now_iso())
    snapshot.setdefault("snapshot_source", "frozen_at_training_start")
    return _json_safe(snapshot)


def training_dataset_snapshots_match(
    stored_snapshot: Mapping[str, Any] | None,
    current_snapshot: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Compare the frozen dataset identity with the current dataset identity."""

    stored = dict(stored_snapshot or {}) if isinstance(stored_snapshot, Mapping) else {}
    current = dict(current_snapshot or {}) if isinstance(current_snapshot, Mapping) else {}
    if not stored:
        return False, "Brak zamrozonego snapshotu datasetu w historii runu."
    if not current:
        return False, "Nie udalo sie zbudowac aktualnego snapshotu datasetu."

    stored_split = str(stored.get("split_sha256") or "").strip()
    current_split = str(current.get("split_sha256") or "").strip()
    if not stored_split:
        return False, "Historyczny run nie ma fingerprintu splitu datasetu."
    if not current_split:
        return False, "Aktualny dataset nie ma fingerprintu splitu."
    if stored_split != current_split:
        return False, "Fingerprint splitu datasetu jest inny niz przed przerwaniem treningu."

    for key, label in (
        ("manifest_sha256", "manifest datasetu"),
        ("data_yaml_sha256", "data.yaml"),
    ):
        left = str(stored.get(key) or "").strip()
        right = str(current.get(key) or "").strip()
        if (left or right) and left != right:
            return False, f"Fingerprint {label} jest inny niz przed przerwaniem treningu."

    return True, "Dataset zgodny z zamrozonym snapshotem."


def build_checkpoint_training_snapshot(
    checkpoint_path: Path | str | None,
    *,
    name: str = "",
    kind: str = "",
) -> dict[str, Any]:
    """Freeze a checkpoint reference without using the path as identity."""

    path = _path_or_none(checkpoint_path)
    exists = bool(path is not None and path.exists() and path.is_file())
    sha = _file_sha256(path)
    size = _file_size(path) if exists and path is not None else 0
    payload = {
        "schema": "alpr.training_checkpoint_snapshot.v1",
        "captured_at": _utc_now_iso(),
        "name": str(name or (path.name if path is not None else "") or "").strip(),
        "kind": str(kind or "unknown").strip() or "unknown",
        "path_hint": str(path or ""),
        "sha256": sha,
        "size": size,
        "exists_at_capture": exists,
        "provenance_status": "complete" if sha else "partial",
    }
    return _json_safe(payload)


def normalize_epoch_index_to_completed_epoch(raw_epoch: Any) -> int | None:
    """Convert a zero-based checkpoint epoch index to completed epoch count."""

    parsed = _int_or_none(raw_epoch)
    if parsed is None:
        return None
    if parsed < 0:
        return 0
    return int(parsed) + 1


def build_output_checkpoint_training_snapshot(
    *,
    best_checkpoint: Path | str | None = None,
    last_checkpoint: Path | str | None = None,
    best_epoch: int | None = None,
    best_epoch_source: str = "",
) -> dict[str, Any]:
    """Freeze output checkpoint hashes after training finishes."""

    best_snapshot = build_checkpoint_training_snapshot(best_checkpoint, name="best.pt", kind="best_checkpoint")
    last_snapshot = build_checkpoint_training_snapshot(last_checkpoint, name="last.pt", kind="last_checkpoint")
    best_sha = str(best_snapshot.get("sha256") or "")
    last_sha = str(last_snapshot.get("sha256") or "")
    checkpoint_best_epoch = _checkpoint_completed_epoch(best_checkpoint)
    resolved_best_epoch = checkpoint_best_epoch if checkpoint_best_epoch is not None else _int_or_none(best_epoch)
    resolved_best_epoch_source = "checkpoint" if checkpoint_best_epoch is not None else str(best_epoch_source or "").strip()
    if not resolved_best_epoch_source:
        resolved_best_epoch_source = "metrics_history" if resolved_best_epoch is not None else "unknown"
    payload = {
        "schema": "alpr.output_checkpoint_snapshot.v1",
        "captured_at": _utc_now_iso(),
        "best": best_snapshot,
        "last": last_snapshot,
        "best_checkpoint_sha256": best_sha,
        "last_checkpoint_sha256": last_sha,
        "best_epoch": resolved_best_epoch,
        "best_epoch_source": resolved_best_epoch_source,
        "provenance_status": "complete" if best_sha or last_sha else "partial",
    }
    return _json_safe(payload)


def _dataset_provenance_for_run(
    run: Mapping[str, Any],
    *,
    fallback_dataset_path: Path | str | None = None,
    target: str = "",
    include_content_fingerprint: bool = True,
) -> tuple[dict[str, Any], str]:
    snapshot = _mapping_copy(_value(run, "training_dataset_snapshot"))
    if snapshot:
        capture = str(snapshot.get("snapshot_source") or "frozen_at_training_start")
        if capture not in {"frozen_at_training_start", "reconstructed_from_training_artifacts"}:
            capture = "frozen_at_training_start"
        return _normalized_dataset_snapshot(snapshot, target=target), capture

    embedded = _mapping_copy(_value(run, "dataset"))
    if embedded and (
        embedded.get("dataset_id")
        or embedded.get("train_images") is not None
        or embedded.get("split_sha256")
        or embedded.get("manifest_sha256")
    ):
        capture = str(embedded.get("provenance_capture") or embedded.get("snapshot_source") or "").strip()
        if capture not in {
            "frozen_at_training_start",
            "reconstructed_from_training_artifacts",
            "reconstructed_at_export",
            "legacy_unknown",
        }:
            capture = "reconstructed_from_training_artifacts"
        return _normalized_dataset_snapshot(embedded, target=target), capture

    dataset_path = str(fallback_dataset_path or _value(run, "dataset_path", "") or "").strip()
    if dataset_path:
        return (
            build_dataset_training_provenance(
                dataset_path,
                target=target,
                include_content_fingerprint=include_content_fingerprint,
            ),
            "reconstructed_at_export",
        )
    return (
        _normalized_dataset_snapshot(
            {
                "dataset_id": "",
                "name": "",
                "target": _normalize_target(target),
                "provenance_status": "legacy_unknown",
            },
            target=target,
        ),
        "legacy_unknown",
    )


def _normalized_dataset_snapshot(snapshot: Mapping[str, Any], *, target: str = "") -> dict[str, Any]:
    payload = dict(snapshot or {})
    payload.setdefault("schema", "alpr.training_dataset_snapshot.v1")
    if not payload.get("target"):
        payload["target"] = _normalize_target(target)
    if not payload.get("provenance_status"):
        payload["provenance_status"] = "complete" if payload.get("manifest_sha256") and payload.get("split_sha256") else "partial"
    return _json_safe(payload)


def _checkpoint_path_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).resolve()).replace("\\", "/").rstrip("/").casefold()


def _resolve_legacy_parent(run: dict[str, Any], history_index) -> dict[str, Any]:
    """Read legacy custom starts by exact checkpoint reference, never by model name."""
    if _explicit_parent_run_id(run) or str(run.get("lineage_mode") or "").lower() in {"resume", "resumed"}:
        return run
    source = str(run.get("parent_model_path") or run.get("base_model") or "").strip()
    if not source or (not run.get("parent_model_path") and _pretrained_origin(run)):
        return run
    key = _checkpoint_path_key(source)
    matches = []
    input_sha = _checkpoint_snapshot_sha(_checkpoint_snapshot_from_run(run, "input_checkpoint_snapshot"))
    for parent_id, parent in history_index.items():
        if parent_id == _run_id(run):
            continue
        for role in ("best", "last"):
            path = str(parent.get(f"{role}_weights") or "").strip()
            if not path or _checkpoint_path_key(path) != key:
                continue
            output = _checkpoint_snapshot_from_run(parent, "output_checkpoint_snapshot")
            output_sha = _checkpoint_snapshot_sha(output, key=role)
            if input_sha and not output_sha:
                output_sha = _file_sha256(Path(path))
            if input_sha and output_sha and input_sha != output_sha:
                continue
            matches.append((parent_id, role))
            break
    if len(matches) != 1:
        return run
    parent_id, role = matches[0]
    return {**run, "lineage_mode": "fine_tune", "parent_run_id": parent_id,
            "parent_model_path": source, "_parent_resolution": f"history_checkpoint_path:{role}"}


def _lineage_total_epochs(
    run: dict[str, Any],
    *,
    history_index: Mapping[str, dict[str, Any]],
    current_checkpoint: Path | None,
    visited: set[str],
) -> _EpochLineageResult:
    run = _resolve_legacy_parent(run, history_index)
    run_id = _run_id(run)
    if run_id and run_id in visited:
        return _EpochLineageResult(
            total_epochs=None,
            known_epochs_minimum=0,
            total_epochs_known=False,
            lineage_stage_count_known=False,
            known_stage_count_minimum=0,
            provenance_status="partial",
            lineage=[],
            warnings=[f"Wykryto cykl rodowodu treningu przy runie {run_id}."],
        )
    if run_id:
        visited.add(run_id)

    completed = _completed_epoch_count(run)
    lineage_mode = _normalize_lineage_mode(_value(run, "lineage_mode", "new"))
    entry = _lineage_entry(run, completed, checkpoint=current_checkpoint)

    if lineage_mode == "fine_tune":
        parent_id = _explicit_parent_run_id(run) or _run_id_from_text(_value(run, "parent_model_path", "")) or _run_id_from_text(_value(run, "parent_model_name", ""))
        parent = dict(history_index.get(parent_id) or {}) if parent_id else {}
        if not parent:
            sidecar_total = _read_parent_sidecar_epochs(run)
            parent_total = int(sidecar_total[1] or 0)
            if sidecar_total[0]:
                return _EpochLineageResult(
                    total_epochs=parent_total + completed,
                    known_epochs_minimum=parent_total + completed,
                    total_epochs_known=True,
                    lineage_stage_count_known=False,
                    known_stage_count_minimum=2,
                    provenance_status="partial",
                    lineage=[
                        {
                            "run_id": parent_id,
                            "mode": "external_known",
                            "epochs_completed": parent_total,
                            "dataset_id": "",
                            "dataset_manifest_sha256": "",
                            "dataset_split_sha256": "",
                            "train_images": None,
                            "nominal_sample_presentations": None,
                            "dataset_snapshot_source": "legacy_unknown",
                            "output_checkpoint_sha256": _file_sha256(_path_or_none(_value(run, "parent_model_path"))),
                        },
                        entry,
                    ],
                    warnings=[f"Rodzic fine-tune ma znaną sumę epok, ale niepełny rodowód etapów: {parent_id or 'nieznany'}."],
                )
            if parent_total > 0:
                return _EpochLineageResult(
                    total_epochs=None,
                    known_epochs_minimum=parent_total + completed,
                    total_epochs_known=False,
                    lineage_stage_count_known=False,
                    known_stage_count_minimum=2,
                    provenance_status="partial",
                    lineage=[
                        {
                            "run_id": parent_id,
                            "mode": "external_partial",
                            "epochs_completed": parent_total,
                            "dataset_id": "",
                            "dataset_manifest_sha256": "",
                            "dataset_split_sha256": "",
                            "train_images": None,
                            "nominal_sample_presentations": None,
                            "dataset_snapshot_source": "legacy_unknown",
                            "output_checkpoint_sha256": _file_sha256(_path_or_none(_value(run, "parent_model_path"))),
                        },
                        entry,
                    ],
                    warnings=[f"Rodzic fine-tune ma tylko częściowy rodowód: {parent_id or 'nieznany'}."],
                )
            return _EpochLineageResult(
                total_epochs=None,
                known_epochs_minimum=completed,
                total_epochs_known=False,
                lineage_stage_count_known=False,
                known_stage_count_minimum=1,
                provenance_status="partial",
                lineage=[entry],
                warnings=[f"Brak rodzica fine-tune: {parent_id or 'nieznany'}."],
            )

        parent_result = _lineage_total_epochs(
            parent,
            history_index=history_index,
            current_checkpoint=_path_or_none(_value(parent, "best_weights") or _value(parent, "last_weights")),
            visited=visited,
        )
        known_minimum = int(parent_result.known_epochs_minimum or 0) + completed
        if parent_result.total_epochs_known and parent_result.total_epochs is not None:
            total = int(parent_result.total_epochs) + completed
            known = True
            status = parent_result.provenance_status
        else:
            total = None
            known = False
            status = _weaken_status(parent_result.provenance_status)
        return _EpochLineageResult(
            total_epochs=total,
            known_epochs_minimum=known_minimum,
            total_epochs_known=known,
            lineage_stage_count_known=bool(parent_result.lineage_stage_count_known),
            known_stage_count_minimum=int(parent_result.known_stage_count_minimum or len(parent_result.lineage)) + 1,
            provenance_status=status,
            lineage=[*parent_result.lineage, entry],
            warnings=[*parent_result.warnings, *(
                [f"Rodzica runu {run_id} odtworzono z zapisanej ścieżki checkpointu: {parent_id}."]
                if run.get("_parent_resolution") else []
            )],
        )

    if _pretrained_origin(run) or not _starts_from_external_or_custom_checkpoint(run):
        return _EpochLineageResult(
            total_epochs=completed,
            known_epochs_minimum=completed,
            total_epochs_known=True,
            lineage_stage_count_known=True,
            known_stage_count_minimum=1,
            provenance_status="complete",
            lineage=[entry],
        )

    sidecar_total = _read_parent_sidecar_epochs(run)
    parent_total = int(sidecar_total[1] or 0)
    if sidecar_total[0]:
        return _EpochLineageResult(
            total_epochs=parent_total + completed,
            known_epochs_minimum=parent_total + completed,
            total_epochs_known=True,
            lineage_stage_count_known=False,
            known_stage_count_minimum=2,
            provenance_status="partial",
            lineage=[entry],
            warnings=["Run startuje z niestandardowego checkpointu o znanej sumie epok, ale niepełnym rodowodzie etapów."],
        )
    if parent_total > 0:
        return _EpochLineageResult(
            total_epochs=None,
            known_epochs_minimum=parent_total + completed,
            total_epochs_known=False,
            lineage_stage_count_known=False,
            known_stage_count_minimum=2,
            provenance_status="partial",
            lineage=[entry],
            warnings=["Run startuje z niestandardowego checkpointu o częściowo znanym rodowodzie."],
        )
    return _EpochLineageResult(
        total_epochs=None,
        known_epochs_minimum=completed,
        total_epochs_known=False,
        lineage_stage_count_known=False,
        known_stage_count_minimum=1,
        provenance_status="partial",
        lineage=[entry],
        warnings=["Run startuje z niestandardowego checkpointu bez znanego rodowodu."],
    )


def _lineage_entry(run: Mapping[str, Any], completed: int, *, checkpoint: Path | None) -> dict[str, Any]:
    target = _normalize_target(_value(run, "training_target", "") or _value(run, "parent_model_target", "")) or _infer_target_from_text(
        " ".join(str(_value(run, key, "")) for key in ("dataset_path", "base_model", "name", "output_dir"))
    )
    dataset, dataset_capture = _dataset_provenance_for_run(
        run,
        fallback_dataset_path=_value(run, "dataset_path", ""),
        target=target,
        include_content_fingerprint=False,
    )
    train_images = _train_images_from_dataset(dataset)
    nominal_presentations = _nominal_sample_presentations(train_images, completed)
    input_snapshot = _checkpoint_snapshot_from_run(run, "input_checkpoint_snapshot")
    output_snapshot = _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
    input_checkpoint = _path_or_none(_value(run, "parent_model_path") or _value(run, "base_model"))
    output_checkpoint = checkpoint or _path_or_none(_value(run, "best_weights") or _value(run, "last_weights"))
    return _json_safe(
        {
            "run_id": _run_id(run),
            "mode": "fine_tune" if _normalize_lineage_mode(_value(run, "lineage_mode", "new")) == "fine_tune" else "new",
            "epochs_completed": completed,
            "dataset_id": str(dataset.get("dataset_id") or ""),
            "dataset_manifest_sha256": str(dataset.get("manifest_sha256") or ""),
            "dataset_split_sha256": str(dataset.get("split_sha256") or ""),
            "train_images": train_images,
            "nominal_sample_presentations": nominal_presentations,
            "dataset_snapshot_source": dataset_capture,
            "input_checkpoint_sha256": _checkpoint_snapshot_sha(input_snapshot) or _file_sha256(input_checkpoint),
            "output_checkpoint_sha256": _checkpoint_snapshot_sha(output_snapshot, key="best")
            or _checkpoint_snapshot_sha(output_snapshot, key="last")
            or _checkpoint_snapshot_sha(output_snapshot)
            or _file_sha256(output_checkpoint),
        }
    )


def _metric_rows_completed_epoch(rows: Any) -> int | None:
    if not isinstance(rows, list) or not rows:
        return None
    parsed_epochs: list[int] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        raw_epoch = row.get("epoch")
        if raw_epoch is None:
            raw_epoch = row.get("Epoch")
        parsed = _int_or_none(raw_epoch)
        if parsed is not None:
            parsed_epochs.append(max(0, parsed))
        elif any(str(value or "").strip() for value in row.values()):
            parsed_epochs.append(index)
    if not parsed_epochs:
        return None
    if min(parsed_epochs) == 0:
        return max(parsed_epochs) + 1
    return max(parsed_epochs)


def _results_csv_completed_epoch(run: Mapping[str, Any]) -> int | None:
    output_dir = str(_value(run, "output_dir", "") or "").strip()
    if not output_dir:
        return None
    results_path = Path(output_dir) / "train" / "results.csv"
    if not results_path.exists() or not results_path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                rows.append({str(key or "").strip(): value for key, value in dict(raw or {}).items()})
    except Exception:
        return None
    return _metric_rows_completed_epoch(rows)


def _checkpoint_completed_epoch(checkpoint_path: Path | str | None) -> int | None:
    path = _path_or_none(checkpoint_path)
    if path is None or not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    return _cached_checkpoint_completed_epoch(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=64)
def _cached_checkpoint_completed_epoch(path: str, size: int, mtime_ns: int, ctime_ns: int) -> int | None:
    torch = get_torch_module()
    if torch is None:
        return None
    checkpoint = None
    try:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            return None
        raw_epoch = _int_or_none(checkpoint.get("epoch"))
        if raw_epoch is None:
            return None
        if raw_epoch >= 0:
            return normalize_epoch_index_to_completed_epoch(raw_epoch)
        # Ultralytics replaces epoch with -1 when finalizing weights. best.pt
        # then carries the entire run's train_results, including later epochs.
        # Locate its own saved metrics instead of treating -1 as zero, or using
        # the last results row as the best epoch.
        metrics = checkpoint.get("train_metrics")
        results = checkpoint.get("train_results")
        if not isinstance(metrics, Mapping) or not isinstance(results, Mapping):
            return None
        metric_keys = [key for key in metrics if str(key).startswith("metrics/")]
        epochs = results.get("epoch")
        if not metric_keys or not isinstance(epochs, (list, tuple)):
            return None
        if any(not isinstance(results.get(key), (list, tuple)) or len(results[key]) != len(epochs)
               for key in metric_keys):
            return None
        matches = []
        for index, epoch in enumerate(epochs):
            if all(_float_or_none(results[key][index]) is not None
                   and _float_or_none(results[key][index]) == _float_or_none(metrics[key])
                   for key in metric_keys):
                completed_epoch = _int_or_none(epoch)
                if completed_epoch is not None and completed_epoch > 0:
                    matches.append(completed_epoch)
        return matches[0] if len(matches) == 1 else None
    except Exception:
        return None
    finally:
        try:
            del checkpoint
        except Exception:
            pass


def _completed_epoch_count(run: Mapping[str, Any]) -> int:
    # Historical completion handlers sometimes copied the requested budget into
    # current_epoch after early stopping. Finished epoch rows are execution evidence.
    if (not _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
            and str(_value(run, "status", "") or "").lower() in {"completed", "failed", "interrupted", "cancelled", "stopped"}):
        evidence = [value for value in (
            _results_csv_completed_epoch(run),
            _metric_rows_completed_epoch(_value(run, "metrics_history", [])),
        ) if value is not None]
        if evidence:
            return max(evidence)
    for key in ("current_epoch", "completed_epochs", "trained_epochs"):
        value = _int_or_none(_value(run, key))
        if value is not None and value > 0:
            return max(0, value)

    csv_epoch = _results_csv_completed_epoch(run)
    if csv_epoch is not None:
        return max(0, csv_epoch)

    metric_epoch = _metric_rows_completed_epoch(_value(run, "metrics_history", []))
    if metric_epoch is not None:
        return max(0, metric_epoch)

    checkpoint_epoch = _checkpoint_completed_epoch(_value(run, "last_weights") or _value(run, "best_weights"))
    if checkpoint_epoch is not None:
        return max(0, checkpoint_epoch)

    return 0


def _legacy_run_from_sidecar(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    raw = sidecar.get("raw") if isinstance(sidecar.get("raw"), Mapping) else sidecar
    training = raw.get("training") if isinstance(raw, Mapping) and isinstance(raw.get("training"), Mapping) else {}
    run_snapshot = raw.get("run_snapshot") if isinstance(raw, Mapping) and isinstance(raw.get("run_snapshot"), Mapping) else {}
    if run_snapshot:
        return dict(run_snapshot)
    if training:
        payload = dict(training)
        if "run_id" in payload and "id" not in payload:
            payload["id"] = payload.get("run_id")
        if "run_name" in payload and "name" not in payload:
            payload["name"] = payload.get("run_name")
        return payload
    return {}


def _read_parent_sidecar_epochs(run: Mapping[str, Any]) -> tuple[bool, int | None]:
    parent_path = _path_or_none(_value(run, "parent_model_path"))
    if parent_path is None:
        return False, None
    if parent_path.exists() and parent_path.is_file() and parent_path.suffix.lower() == ".alprmodel":
        try:
            with zipfile.ZipFile(parent_path, "r") as archive:
                payload = json.loads(archive.read("manifest.json").decode("utf-8-sig", errors="replace"))
            known, total = _training_payload_total_epochs(payload.get("training") if isinstance(payload, Mapping) else {})
            if known or total:
                return known, total
        except Exception:
            pass
    for sidecar_path in _sidecar_candidates(parent_path):
        if not sidecar_path.exists() or not sidecar_path.is_file():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        training = payload.get("training") if isinstance(payload, Mapping) and isinstance(payload.get("training"), Mapping) else {}
        if not training and isinstance(payload.get("raw"), Mapping):
            raw = payload.get("raw") or {}
            training = raw.get("training") if isinstance(raw.get("training"), Mapping) else {}
        known, total = _training_payload_total_epochs(training)
        if known or total:
            return known, total
    return False, None


def _training_payload_total_epochs(training: Any) -> tuple[bool, int | None]:
    if not isinstance(training, Mapping):
        return False, None
    known_raw = training.get("total_epochs_known")
    known = known_raw is True or str(known_raw).strip().lower() in {"1", "true", "tak", "yes"}
    explicitly_unknown = known_raw is False or str(known_raw).strip().lower() in {"0", "false", "nie", "no"}
    total = _int_or_none(training.get("total_epochs"))
    minimum = _int_or_none(training.get("known_epochs_minimum"))
    if known and total is not None:
        return True, total
    if total is not None and total > 0 and not explicitly_unknown:
        return True, total
    if minimum is not None and minimum > 0:
        return False, minimum
    return False, None


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _train_images_from_dataset(dataset: Mapping[str, Any]) -> int | None:
    parsed = _int_or_none(_value(dataset, "train_images"))
    if parsed is None:
        return None
    return max(0, parsed)


def _nominal_sample_presentations(train_images: int | None, epochs_completed: int | None) -> int | None:
    if train_images is None or epochs_completed is None:
        return None
    if train_images < 0 or epochs_completed < 0:
        return None
    return int(train_images) * int(epochs_completed)


def _lineage_sample_presentation_summary(lineage: list[dict[str, Any]]) -> tuple[int | None, bool, int]:
    if not lineage:
        return None, False, 0
    known = True
    total = 0
    minimum = 0
    for entry in lineage:
        value = _int_or_none(entry.get("nominal_sample_presentations"))
        if value is None:
            known = False
            continue
        value = max(0, int(value))
        total += value
        minimum += value
    return (total if known else None), known, minimum


def _checkpoint_snapshot_from_run(run: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _mapping_copy(_value(run, key))


def _checkpoint_snapshot_sha(snapshot: Mapping[str, Any], *, key: str = "") -> str:
    if not isinstance(snapshot, Mapping):
        return ""
    if key:
        nested = snapshot.get(key)
        if isinstance(nested, Mapping):
            value = str(nested.get("sha256") or "").strip()
            if value:
                return value
        direct = str(snapshot.get(f"{key}_checkpoint_sha256") or "").strip()
        if direct:
            return direct
    return str(snapshot.get("sha256") or "").strip()


def _checkpoint_mismatch_warning(snapshot: Mapping[str, Any], checkpoint_path: Path | None, *, label: str) -> str:
    if not isinstance(snapshot, Mapping) or checkpoint_path is None:
        return ""
    frozen = _checkpoint_snapshot_sha(snapshot, key="best") or _checkpoint_snapshot_sha(snapshot)
    if not frozen:
        return ""
    current = _file_sha256(checkpoint_path)
    if current and current != frozen:
        return f"Ostrzezenie: {label} nie odpowiada checkpointowi zarejestrowanemu po treningu."
    return ""


def build_checkpoint_metric_summary(run_like: Any, *, checkpoint: Path | str | None = None,
                                    verify_checkpoint: bool = True) -> dict[str, Any]:
    """Describe the selected weights, never the maximum of unrelated epoch metrics."""
    run = _run_like_dict(run_like)
    path = _path_or_none(checkpoint or _value(run, "best_weights"))
    rows = [dict(row) for row in (run.get("metrics_history") or []) if isinstance(row, Mapping)]
    snapshot = _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
    frozen_sha = _checkpoint_snapshot_sha(snapshot, key="best") or _checkpoint_snapshot_sha(snapshot)
    actual_sha = _file_sha256(path)
    mismatch = bool(frozen_sha and actual_sha and frozen_sha != actual_sha)
    epoch = None
    source = "unknown"
    frozen_epoch = _int_or_none(snapshot.get("best_epoch"))
    frozen_source = str(snapshot.get("best_epoch_source") or "")
    if (frozen_sha and actual_sha == frozen_sha and frozen_epoch is not None and frozen_epoch > 0
            and frozen_source.startswith("checkpoint")):
        epoch, source = frozen_epoch, frozen_source
    if run and actual_sha and (verify_checkpoint or epoch is None):
        checkpoint_epoch = _checkpoint_completed_epoch(path)
        if checkpoint_epoch is not None:
            epoch, source = checkpoint_epoch, "checkpoint"
    if epoch is None and run and not mismatch:
        epoch = _best_epoch_from_run(run)
        source = _best_epoch_source_from_run(run) if epoch is not None else "unknown"
        if epoch is not None and epoch <= 0:
            epoch, source = None, "unknown"
    # The final validation callback can repeat the last epoch number while
    # evaluating earlier best.pt weights. Keep the original epoch row.
    best_row = next((row for row in rows if _int_or_none(row.get("epoch")) == epoch), {}) if epoch else {}
    if mismatch:
        best_row = {}
    def metric(*keys):
        for key in keys:
            value = _float_or_none(best_row.get(key))
            if value is not None:
                return value
        return None
    return {
        "best_epoch": epoch, "best_epoch_source": source,
        "checkpoint_sha256": actual_sha, "checkpoint_mismatch": mismatch,
        "best_map50": metric("map50", "box_map50", "metrics/mAP50(B)", "metrics/mAP50"),
        "best_map50_95": metric("map50_95", "box_map50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "best_row": best_row, "latest": rows[-1] if rows else {}, "history_rows": rows,
        "metrics_source": "checkpoint_epoch_history" if best_row else "unknown",
    }


def _best_epoch_from_run(run: Mapping[str, Any]) -> int | None:
    output_snapshot = _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
    best_epoch = _int_or_none(output_snapshot.get("best_epoch")) if output_snapshot else None
    if best_epoch is not None:
        return max(0, best_epoch)
    explicit = _int_or_none(_value(run, "best_epoch"))
    if explicit is not None:
        return max(0, explicit)

    metrics = _value(run, "metrics_history", [])
    if not isinstance(metrics, list):
        return None
    best_score: tuple[float, float] | None = None
    best_row_epoch: int | None = None
    for row in metrics:
        if not isinstance(row, Mapping):
            continue
        epoch = _int_or_none(row.get("epoch") or row.get("Epoch"))
        if epoch is None:
            continue
        score_95 = _float_or_none(
            row.get("map50_95")
            or row.get("box_map50_95")
            or row.get("pose_map50_95")
            or row.get("metrics/mAP50-95(B)")
            or row.get("metrics/mAP50-95")
        )
        score_50 = _float_or_none(
            row.get("map50")
            or row.get("box_map50")
            or row.get("pose_map50")
            or row.get("metrics/mAP50(B)")
            or row.get("metrics/mAP50")
        )
        score = (float(score_95 if score_95 is not None else -1.0), float(score_50 if score_50 is not None else -1.0))
        if best_score is None or score > best_score:
            best_score = score
            best_row_epoch = max(0, int(epoch))
    return best_row_epoch


def _best_epoch_source_from_run(run: Mapping[str, Any]) -> str:
    output_snapshot = _checkpoint_snapshot_from_run(run, "output_checkpoint_snapshot")
    source = str(output_snapshot.get("best_epoch_source") or "").strip() if output_snapshot else ""
    if source:
        return source
    if _int_or_none(_value(run, "best_epoch")) is not None:
        return "run_state"
    metrics = _value(run, "metrics_history", [])
    if isinstance(metrics, list) and any(isinstance(row, Mapping) for row in metrics):
        return "metrics_history"
    return "unknown"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sidecar_candidates(model_path: Path) -> list[Path]:
    suffix = "".join(model_path.suffixes) if model_path.suffixes else model_path.suffix
    stem_path = model_path.with_suffix("") if model_path.suffix else model_path
    result: list[Path] = []
    for template in _SIDE_CAR_NAMES:
        name = template.format(stem=stem_path.name, suffix=suffix)
        candidate = model_path.parent / name
        if candidate not in result:
            result.append(candidate)
    return result


def _dataset_root(path_value: Path | str | None) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.name.lower() == "data.yaml":
        return path.parent
    return path


def _dataset_split_counts(root: Path, cfg: Mapping[str, Any]) -> dict[str, int]:
    return {
        split: sum(_count_images_in_source(source) for source in _split_sources(root, cfg, split))
        for split in ("train", "val", "test")
    }


def _dataset_split_fingerprint(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for source in _split_sources(root, cfg, split):
            entries.extend(_fingerprint_entries_for_source(root, source, split, image=True))
        for label_dir in _label_dirs_for_split(root, split):
            entries.extend(_fingerprint_entries_for_source(root, label_dir, split, image=False))
    entries.sort(key=lambda item: (str(item.get("split")), str(item.get("relative_path"))))
    return {
        "sha256": _json_sha256(entries),
        "file_count": len(entries),
    }


def _split_sources(root: Path, cfg: Mapping[str, Any], split: str) -> list[Path]:
    raw = str(cfg.get(split) or f"images/{split}").strip() if isinstance(cfg, Mapping) else f"images/{split}"
    candidates = [Path(raw)]
    resolved_root = _yaml_root(root, cfg)
    result: list[Path] = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = resolved_root / candidate
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        if candidate.exists() and candidate not in result:
            result.append(candidate)
    if result:
        return result
    for candidate in (root / "images" / split, root / split / "images"):
        if candidate.exists() and candidate not in result:
            result.append(candidate)
    return result


def _label_dirs_for_split(root: Path, split: str) -> list[Path]:
    result: list[Path] = []
    for candidate in (root / "labels" / split, root / split / "labels"):
        if candidate.exists() and candidate.is_dir() and candidate not in result:
            result.append(candidate)
    return result


def _fingerprint_entries_for_source(root: Path, source: Path, split: str, *, image: bool) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if source.is_file():
        paths.append(source)
        if image and source.suffix.lower() not in _IMAGE_SUFFIXES:
            try:
                for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
                    text = str(line or "").strip()
                    if not text or text.startswith("#"):
                        continue
                    child = Path(text)
                    if not child.is_absolute():
                        child = source.parent / child
                    if child.exists() and child.suffix.lower() in _IMAGE_SUFFIXES:
                        paths.append(child)
            except Exception:
                pass
    elif source.is_dir():
        suffixes = _IMAGE_SUFFIXES if image else {".txt"}
        try:
            paths.extend(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
        except Exception:
            paths = []
    entries: list[dict[str, Any]] = []
    for path in paths:
        entries.append(
            {
                "split": split,
                "relative_path": _relative_path(root, path),
                "sha256": _file_sha256(path),
                "size": _file_size(path),
            }
        )
    return entries


def _count_images_in_source(source: Path) -> int:
    if source.is_dir():
        try:
            return sum(1 for path in source.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)
        except Exception:
            return 0
    if source.is_file() and source.suffix.lower() in _IMAGE_SUFFIXES:
        return 1
    if source.is_file():
        count = 0
        try:
            for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
                text = str(line or "").strip()
                if text and not text.startswith("#"):
                    count += 1
        except Exception:
            return 0
        return count
    return 0


def _dataset_manifest_refs(root: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for name in _DATASET_MANIFEST_NAMES:
        path = root / name
        if path.exists() and path.is_file():
            refs.append({"name": name, "sha256": _file_sha256(path), "size": _file_size(path)})
    return refs


def _dataset_augmentation_summary(root: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "source_images": 0,
        "source_objects": 0,
        "offline_augmentation_train_added": 0,
        "split_seed": None,
    }
    for ref in manifests:
        name = str(ref.get("name") or "")
        path = root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for key in ("source_images", "base_images", "input_images"):
            value = _find_first_int(payload, key)
            if value:
                result["source_images"] = max(int(result["source_images"]), value)
        for key in ("source_objects", "source_plates", "source_labels", "char_count", "plate_count"):
            value = _find_first_int(payload, key)
            if value:
                result["source_objects"] = max(int(result["source_objects"]), value)
        for key in ("offline_augmentation_train_added", "generated_images", "planned_images", "augmentation_train_added"):
            value = _find_first_int(payload, key)
            if value:
                result["offline_augmentation_train_added"] = max(int(result["offline_augmentation_train_added"]), value)
        seed = _find_first_value(payload, "split_seed", "seed", "random_seed")
        if seed not in (None, "") and result["split_seed"] in (None, ""):
            result["split_seed"] = seed
    return result


def _find_first_int(value: Any, key: str) -> int:
    found = _find_first_value(value, key)
    parsed = _int_or_none(found)
    return int(parsed or 0)


def _find_first_value(value: Any, *keys: str) -> Any:
    wanted = {str(key) for key in keys}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in wanted:
                return child
        for child in value.values():
            nested = _find_first_value(child, *keys)
            if nested not in (None, ""):
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _find_first_value(child, *keys)
            if nested not in (None, ""):
                return nested
    return None


def _safe_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = safe_load_yaml(path)
        return dict(payload or {}) if isinstance(payload, Mapping) else {}
    except Exception:
        return {}


def _yaml_root(root: Path, cfg: Mapping[str, Any]) -> Path:
    raw = str(cfg.get("path") or "").strip() if isinstance(cfg, Mapping) else ""
    if not raw:
        return root
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except Exception:
        return path


def _infer_dataset_target(root: Path, cfg: Mapping[str, Any]) -> str:
    if isinstance(cfg, Mapping) and cfg.get("kpt_shape"):
        return "plate"
    return _infer_target_from_text(str(root))


def _infer_target_from_text(text: str) -> str:
    lower = str(text or "").lower()
    if "pose" in lower or any(token in lower for token in ("plate", "plates", "tablica", "tablic")):
        return "plate"
    if any(token in lower for token in ("char", "chars", "character", "znak", "znaki")):
        return "char"
    if any(token in lower for token in ("vehicle", "vehicles", "pojazd", "pojazdy", "car", "cars")):
        return "vehicle"
    return ""


def _target_code(target: str) -> str:
    return {"plate": "MT", "char": "MZ", "character": "MZ", "vehicle": "MP"}.get(_normalize_target(target), "XX")


def _normalize_target(target: Any) -> str:
    raw = str(target or "").strip().lower()
    if raw in {"plate", "plates", "tablica", "tablice", "pose", "mt"}:
        return "plate"
    if raw in {"char", "chars", "character", "characters", "znak", "znaki", "mz"}:
        return "char"
    if raw in {"vehicle", "vehicles", "pojazd", "pojazdy", "car", "cars", "mp"}:
        return "vehicle"
    return raw


def _normalize_lineage_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"fine_tune", "finetune", "fine-tune", "continue", "continued"}:
        return "fine_tune"
    if raw in {"resume", "resumed"}:
        return "new"
    return "new"


def _pretrained_origin(run: Mapping[str, Any]) -> str:
    base = str(_value(run, "base_model", "") or "").strip()
    if not base:
        return ""
    name = Path(base).name.lower()
    if re.match(r"^yolo(v?\d+|\d+)[a-z0-9_-]*(?:-pose)?(?:\.pt)?$", name):
        return Path(base).name
    return ""


def _starts_from_external_or_custom_checkpoint(run: Mapping[str, Any]) -> bool:
    base = str(_value(run, "base_model", "") or "").strip()
    parent = str(_value(run, "parent_model_path", "") or "").strip()
    if parent:
        return True
    if not base:
        return False
    return not bool(_pretrained_origin(run))


def _explicit_parent_run_id(run: Mapping[str, Any]) -> str:
    return str(_value(run, "parent_run_id", "") or "").strip()


def _run_id(run: Mapping[str, Any]) -> str:
    explicit = str(_value(run, "id", "") or _value(run, "run_id", "") or "").strip()
    return explicit or _run_id_from_text(_value(run, "output_dir", "") or _value(run, "best_weights", ""))


def _run_id_from_text(text: Any) -> str:
    match = _RUN_ID_RE.search(str(text or ""))
    return str(match.group(1) or "") if match else ""


def _normalize_history_index(history_index: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(history_index, Mapping):
        return result
    for key, value in history_index.items():
        payload = _run_like_dict(value)
        run_id = str(payload.get("id") or key or "").strip()
        if run_id:
            payload.setdefault("id", run_id)
            result[run_id] = payload
    return result


def _run_like_dict(run_like: Any) -> dict[str, Any]:
    if not run_like:
        return {}
    if isinstance(run_like, Mapping):
        return dict(run_like)
    try:
        to_dict = getattr(run_like, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
    except Exception:
        pass
    result: dict[str, Any] = {}
    for key in (
        "id",
        "name",
        "created_at",
        "status",
        "dataset_path",
        "base_model",
        "epochs",
        "batch_size",
        "img_size",
        "current_epoch",
        "best_epoch",
        "best_map50",
        "best_map50_95",
        "output_dir",
        "best_weights",
        "last_weights",
        "started_at",
        "finished_at",
        "metrics_history",
        "lineage_mode",
        "parent_run_id",
        "parent_model_path",
        "parent_model_name",
        "parent_model_target",
        "parent_dataset_path",
        "training_target",
        "training_dataset_snapshot",
        "training_dataset_input_snapshot",
        "dataset_preparation",
        "input_checkpoint_snapshot",
        "output_checkpoint_snapshot",
    ):
        try:
            value = getattr(run_like, key)
        except Exception:
            continue
        result[key] = value
    return result


def _value(run: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return run.get(key, default) if isinstance(run, Mapping) else default


def _int_or_none(value: Any) -> int | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _path_or_none(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def _file_sha256(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        safe_path = Path(path)
        if not safe_path.exists() or not safe_path.is_file():
            return ""
        stat = safe_path.stat()
        return _cached_file_sha256(str(safe_path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except Exception:
        return ""


@lru_cache(maxsize=128)
def _cached_file_sha256(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except Exception:
        return 0


def _relative_path(root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except Exception:
        return str(path)


def _json_sha256(value: Any) -> str:
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ""


def _weaken_status(status: str) -> str:
    raw = str(status or "").strip().lower()
    if raw == "legacy_unknown":
        return "legacy_unknown"
    return "partial"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value
