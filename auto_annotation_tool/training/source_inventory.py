"""Exact scene inventory, separate from the dataset-level identity fingerprint.

Snapshots freeze this payload for new runs. Old runs can be reconstructed only
after the current files match their frozen fingerprint. No dataset is modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "alpr.training_source_hashes.v1"
FILENAME = "training_source_hashes.json"
SNAPSHOT_KEY = "training_source_hash_inventory"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value):
    value = str(value or "").lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


def _source_sha(row):
    provenance = row.get("provenance") or {}
    # Prefer the immutable scene identity, never crop_identity or artifact SHA.
    identity = str(provenance.get("source_image_id") or row.get("source_image_id") or "")
    if identity.startswith("img-sha256-") and _sha(identity[11:]):
        return _sha(identity[11:])
    for data in (provenance, row):
        if _sha(data.get("source_image_sha256")):
            return _sha(data["source_image_sha256"])
    group = str(row.get("group_id") or "")
    prefix = "source_image_id:img-sha256-"
    return _sha(group[len(prefix):]) if group.startswith(prefix) else ""


def unavailable(dataset_id, role, count=0, reason="missing_frozen_dataset"):
    return {"schema": SCHEMA, "dataset_id": str(dataset_id or ""), "role": role,
            "source_scene_hashes": [], "artifact_hashes": [],
            "sample_count": max(0, int(count or 0)), "unresolved_source_count": max(0, int(count or 0)),
            "unresolved_dataset_count": 0, "status": "unavailable", "capture": reason}


def _resolve_scene(path, *, root, rows, role, visited=frozenset()):
    if path in visited:
        return ""
    candidates = rows.get(path, [])
    hashes = {_source_sha(row) for row in candidates} - {""}
    if hashes:
        return next(iter(hashes)) if len(hashes) == 1 else ""
    source_rows = [row for row in candidates if row.get("source_image")]
    if source_rows:
        resolved = set()
        for row in source_rows:
            source = (root / row["source_image"]).resolve()
            if source == path:
                continue
            if row.get("_derived_artifact") or source in rows:
                sha = _resolve_scene(source, root=root, rows=rows, role=role, visited=visited | {path})
            else:
                try:
                    sha = file_sha256(source)
                except OSError:
                    sha = ""
            if not sha:
                return ""
            resolved.add(sha)
        return next(iter(resolved)) if len(resolved) == 1 else ""
    if role in {"plate", "vehicle"}:
        try:
            return file_sha256(path)
        except OSError:
            pass
    return ""


def build_source_inventory(dataset_path, *, role, dataset_id=""):
    """Enumerate real split inputs; join metadata by exact path, not basename."""
    from .model_provenance import _safe_yaml, _yaml_root

    role = {"char": "character", "car": "vehicle"}.get(role, role)
    root = Path(dataset_path)
    if root.is_file():
        root = root.parent
    cfg = _safe_yaml(root / "data.yaml")
    data_root = _yaml_root(root, cfg)
    rows = {}
    errors = 0
    for name in ("metadata_manifest.json", "split_assignment_manifest.json", "scene_split_assignment.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            for row in json.loads(path.read_text(encoding="utf-8-sig")).get("items", []):
                raw = str(row.get("image_path") or "")
                if raw:
                    key = (root / raw).resolve()
                    rows.setdefault(key, []).append(row)
        except (OSError, ValueError, TypeError, AttributeError):
            errors += 1
    augmentation = root / "augmentation_manifest.json"
    if augmentation.exists():
        try:
            for row in json.loads(augmentation.read_text(encoding="utf-8-sig")).get("generated_files", []):
                key = (root / row["image"]).resolve()
                rows.setdefault(key, []).append({"source_image": row["source_image"], "_derived_artifact": True})
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            errors += 1

    scenes = {}
    artifacts = []
    unresolved = samples = 0
    for split in SPLITS:
        raw_sources = cfg.get(split, f"images/{split}")
        if not raw_sources:
            continue
        sources = raw_sources if isinstance(raw_sources, list) else [raw_sources]
        paths = set()
        for raw in sources:
            source = data_root / str(raw)
            if source.is_dir():
                paths.update(p.resolve() for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            elif source.is_file() and source.suffix.lower() == ".txt":
                for line in source.read_text(encoding="utf-8-sig").splitlines():
                    if line.strip():
                        # YOLO resolves ./ entries relative to the list file.
                        p = source.parent / line[2:] if line.startswith("./") else data_root / line
                        paths.add(p.resolve())
            elif source.is_file():
                paths.add(source.resolve())
            elif split in cfg:
                errors += 1
        for path in sorted(paths):
            samples += 1
            try:
                artifact = file_sha256(path)
                artifacts.append((artifact, split, "crop" if role == "character" else "image"))
            except OSError:
                artifact = ""
                errors += 1
            source_hash = _resolve_scene(path, root=root, rows=rows, role=role)
            if source_hash:
                scenes.setdefault(source_hash, set()).add(split)
            else:
                unresolved += 1
    result = unavailable(dataset_id, role, samples, "frozen_at_training_start")
    result.update(source_scene_hashes=[{"sha256": sha, "splits": sorted(splits)} for sha, splits in sorted(scenes.items())],
                  artifact_hashes=[{"sha256": sha, "split": split, "kind": kind} for sha, split, kind in sorted(artifacts)],
                  unresolved_source_count=unresolved, enumeration_error_count=errors,
                  status="complete" if samples and not unresolved and not errors else ("partial" if scenes else "unavailable"))
    return result


def inventory_for_export(metadata, *, role):
    """Use frozen exact data, or verify a legacy snapshot before reconstructing."""
    from .model_provenance import build_dataset_training_provenance, training_dataset_snapshots_match

    training = metadata.get("training") or {}
    snapshot = training.get("dataset") or training.get("training_dataset_snapshot") or {}
    dataset_id = snapshot.get("dataset_id") or training.get("dataset_id") or ""
    inventory = snapshot.get(SNAPSHOT_KEY)
    if inventory:
        inventory = copy.deepcopy(inventory)
        validate_inventory(inventory, role=role)
        if inventory["dataset_id"] != dataset_id:
            raise ValueError("Inventory dataset_id differs from training snapshot")
    else:
        inventory = unavailable(dataset_id, role, snapshot.get("total_images", 0))
        path = snapshot.get("local_path_hint") or training.get("dataset_path") or ""
        capture = training.get("provenance_capture") or snapshot.get("snapshot_source")
        if path and Path(path).exists() and snapshot.get("split_sha256") and capture in {
            "frozen_at_training_start", "reconstructed_from_training_artifacts"
        }:
            current = build_dataset_training_provenance(path, target={"character": "char"}.get(role, role))
            if training_dataset_snapshots_match(snapshot, current)[0]:
                inventory = build_source_inventory(path, role=role, dataset_id=dataset_id)
                after = build_dataset_training_provenance(path, target={"character": "char"}.get(role, role))
                if training_dataset_snapshots_match(snapshot, after)[0]:
                    inventory["capture"] = "reconstructed_from_verified_snapshot"
                else:
                    inventory = unavailable(dataset_id, role, snapshot.get("total_images", 0), "dataset_changed_during_export")
            else:
                inventory["capture"] = "dataset_changed_since_training"
    # A final run's dataset cannot certify earlier fine-tuning datasets.
    other_datasets = {str(row.get("dataset_id") or "") for row in training.get("lineage", [])
                      if isinstance(row, dict)} - {str(dataset_id)}
    inventory["unresolved_dataset_count"] = len(other_datasets)
    if other_datasets and inventory["status"] == "complete":
        inventory["status"] = "partial"
    return inventory


def strip_embedded_inventory(value):
    """Full hash arrays live only in the sidecar, including in reproducibility."""
    if isinstance(value, dict):
        return {k: strip_embedded_inventory(v) for k, v in value.items() if k != SNAPSHOT_KEY}
    if isinstance(value, (tuple, list)):
        return [strip_embedded_inventory(v) for v in value]
    return value


def validate_inventory(payload, *, role):
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("role") != role:
        raise ValueError("Invalid training source inventory schema/role")
    if payload.get("status") not in {"complete", "partial", "unavailable"}:
        raise ValueError("Invalid training source inventory status")
    for key in ("sample_count", "unresolved_source_count", "unresolved_dataset_count"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"Invalid inventory count: {key}")
    seen = set()
    for row in payload["source_scene_hashes"]:
        sha, splits = row.get("sha256"), row.get("splits")
        if not _sha(sha) or _sha(sha) != sha or sha in seen or not isinstance(splits, list) or not splits or not set(splits) <= set(SPLITS):
            raise ValueError("Invalid or duplicate source scene hash/split")
        seen.add(sha)
    for row in payload["artifact_hashes"]:
        if not _sha(row.get("sha256")) or _sha(row.get("sha256")) != row.get("sha256") or row.get("split") not in SPLITS or row.get("kind") not in {"image", "crop"}:
            raise ValueError("Invalid artifact hash/split/kind")
    if payload["status"] == "complete" and (not seen or payload["unresolved_source_count"] or payload["unresolved_dataset_count"] or payload.get("enumeration_error_count", 0)):
        raise ValueError("Incomplete inventory cannot claim complete status")


def write_inventory(package_root, inventory):
    validate_inventory(inventory, role=inventory["role"])
    data = (json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (Path(package_root) / FILENAME).write_bytes(data)
    return {"schema": SCHEMA, "file": FILENAME, "sha256": hashlib.sha256(data).hexdigest(),
            "source_scene_hash_count": len(inventory["source_scene_hashes"]),
            "artifact_hash_count": len(inventory["artifact_hashes"])}


def validate_archive_inventory(archive, manifest):
    ref = manifest.get("training_source_inventory")
    if ref is None:  # legacy package
        return
    if not isinstance(ref, dict) or ref.get("schema") != SCHEMA or ref.get("file") != FILENAME:
        raise ValueError("Invalid training_source_inventory reference")
    data = archive.read(FILENAME)
    if hashlib.sha256(data).hexdigest() != ref.get("sha256"):
        raise ValueError("Training source inventory SHA-256 mismatch")
    payload = json.loads(data)
    validate_inventory(payload, role=manifest["role"])
    training = manifest.get("training") or {}
    dataset_id = training.get("dataset_id") or (training.get("dataset") or {}).get("dataset_id")
    if dataset_id and dataset_id != payload.get("dataset_id"):
        raise ValueError("Training source inventory dataset_id mismatch")
    for key, rows in (("source_scene_hash_count", "source_scene_hashes"), ("artifact_hash_count", "artifact_hashes")):
        if ref.get(key) != len(payload[rows]):
            raise ValueError(f"Training source inventory {key} mismatch")
