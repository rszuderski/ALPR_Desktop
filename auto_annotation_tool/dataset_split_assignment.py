"""Deterministic source-group assignment; independent of GUI and training."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import PurePosixPath

from .dataset_split_allocation import build_split_entries

ASSIGNMENT_SCHEMA = "alpr.dataset_split_assignment.v1"
GROUPING_SCHEMA = "alpr.dataset_source_groups.v1"
GROUPING_PRIORITY = ["dataset_group", "session_group", "provenance.source_image_id", "crop_identity"]


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def crop_identity(item: dict) -> str:
    provenance = item.get("provenance") or {}
    value = item.get("crop_id") or provenance.get("crop_id") or provenance.get("crop_identity_sha256")
    value = value or item.get("source_pid") or item.get("pid")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Source item lacks a stable crop identity")
    return value.strip()


def source_group_id(item: dict) -> str:
    provenance = item.get("provenance") or {}
    for name, value in [("dataset_group", item.get("dataset_group")),
        ("session_group", item.get("session_group") or provenance.get("session_group")),
        ("source_image_id", provenance.get("source_image_id"))]:
        if value is not None and str(value).strip():
            return name + ":" + str(value).strip()
    return "crop:" + crop_identity(item)


def relative_path(value) -> str:
    path = PurePosixPath(str(value or "").replace("\\", "/"))
    if not str(value or "") or path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("Manifest paths must be relative and contained in the dataset")
    return path.as_posix()


def assignment_sha256(manifest: dict) -> str:
    # Capture time is evidence, not assignment identity. Paths are relative.
    return canonical_sha256({key: value for key, value in manifest.items()
        if key not in {"created_at", "assignment_sha256"}})


def build_group_assignment(source_manifest: dict, *, source_dataset_id: str,
                           source_manifest_sha256: str, ratios: dict, seed: int = 42) -> dict:
    if not ratios or any(name not in {"train", "val", "test"} for name in ratios):
        raise ValueError("Expected train/val/test ratios")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in ratios.values()) or abs(sum(ratios.values())-1)>1e-9:
        raise ValueError("Ratios must be finite, nonnegative and sum to one")
    items = source_manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Source manifest has no items")
    prepared = []
    identities = set()
    names = set()
    for item in items:
        identity = crop_identity(item)
        if identity in identities:
            raise ValueError("Duplicate source crop identity: " + identity)
        identities.add(identity)
        image = relative_path(item.get("image_path"))
        label = relative_path(item.get("label_path"))
        if PurePosixPath(image).name in names:
            raise ValueError("Ambiguous source image basename")
        names.add(PurePosixPath(image).name)
        prepared.append({"crop_identity": identity, "source_pid": item.get("source_pid") or item.get("pid"),
            "source_item_id": item.get("pid"), "group_id": source_group_id(item),
            "dataset_group": source_group_id(item), "source_image_path": image, "source_label_path": label})
    prepared.sort(key=lambda row: row["crop_identity"])
    splits, image_counts = build_split_entries(prepared, ratios, shuffle_seed=seed)
    rows = []
    counts = {"groups_total": len({row["group_id"] for row in prepared})}
    for split, entries in splits.items():
        counts[split + "_groups"] = len({row["group_id"] for row in entries})
        counts[split + "_images"] = image_counts[split]
        for row in entries:
            row = {k:v for k,v in row.items() if k!="dataset_group"}
            row.update(split=split,
                image_path=f"images/{split}/{PurePosixPath(row['source_image_path']).name}",
                label_path=f"labels/{split}/{PurePosixPath(row['source_label_path']).name}")
            rows.append(row)
    rows.sort(key=lambda row: row["crop_identity"])
    manifest = {"schema": ASSIGNMENT_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_id": source_dataset_id, "source_manifest_sha256": source_manifest_sha256,
        "source_contract_sha256": str(source_manifest.get("gold_source_contract_sha256") or ""),
        "seed": int(seed), "ratios": dict(ratios), "grouping_schema": GROUPING_SCHEMA,
        "grouping_priority": list(GROUPING_PRIORITY), "items": rows, "counts": counts}
    manifest["assignment_sha256"] = assignment_sha256(manifest)
    result = validate_group_assignment(manifest, source_manifest=source_manifest)
    if not result["ok"]:
        raise ValueError("Invalid assignment: " + "; ".join(result["errors"]))
    return manifest


def validate_group_assignment(manifest: dict, *, source_manifest: dict | None = None) -> dict:
    errors = []
    groups = defaultdict(set)
    seen = Counter()
    image_paths = set()
    label_paths = set()
    for row in manifest.get("items") or []:
        identity = row.get("crop_identity")
        group = row.get("group_id")
        split = row.get("split")
        seen[identity] += 1
        if not identity or not group or split not in {"train", "val", "test"}:
            errors.append("Invalid identity/group/split")
        groups[group].add(split)
        try:
            image = relative_path(row.get("image_path"))
            label = relative_path(row.get("label_path"))
            if image in image_paths or label in label_paths:
                errors.append("Duplicate destination path")
            image_paths.add(image)
            label_paths.add(label)
            if not image.startswith("images/"+str(split)+"/") or not label.startswith("labels/"+str(split)+"/"):
                errors.append("Destination path does not match split")
        except ValueError as exc:
            errors.append(str(exc))
    leakage = {str(group): sorted(map(str,splits)) for group,splits in groups.items() if len(splits)>1}
    if leakage:
        errors.append("Source group occurs in more than one split")
    duplicates = [identity for identity,count in seen.items() if count!=1]
    if duplicates:
        errors.append("Duplicate crop identities")
    missing = []
    unexpected = []
    if source_manifest is not None:
        expected = {crop_identity(item): item for item in source_manifest.get("items") or []}
        missing = sorted(set(expected)-set(seen))
        unexpected = sorted(set(seen)-set(expected))
        if missing or unexpected or len(expected)!=len(source_manifest.get("items") or []):
            errors.append("Assignment is not an exact partition of the source")
        for row in manifest.get("items") or []:
            item = expected.get(row.get("crop_identity"))
            if item and row.get("group_id")!=source_group_id(item):
                errors.append("Group identity differs from source contract")
    counts = {"groups_total":len(groups)}
    for split in ["train","val","test"]:
        counts[split+"_groups"] = sum(split in values for values in groups.values())
        counts[split+"_images"] = sum(row.get("split")==split for row in manifest.get("items") or [])
    declared = manifest.get("counts") or {}
    if any(counts.get(key)!=value for key,value in declared.items()) or not declared:
        errors.append("Assignment counts do not match items")
    if manifest.get("schema")!=ASSIGNMENT_SCHEMA or manifest.get("grouping_schema")!=GROUPING_SCHEMA:
        errors.append("Unsupported assignment schema")
    if manifest.get("assignment_sha256")!=assignment_sha256(manifest):
        errors.append("Assignment fingerprint mismatch")
    return {"ok":not errors,"errors":errors,"cross_split_group_leakage":leakage,
        "duplicate_crop_identities":duplicates,"missing_items":missing,"unexpected_items":unexpected,"counts":counts}
