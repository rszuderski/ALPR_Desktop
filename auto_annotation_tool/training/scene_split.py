"""Split full scenes by identity, including known augmentation ancestry."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from ..dataset_split_allocation import allocate_split_counts
from .source_inventory import _resolve_scene, file_sha256


def dataset_scene_hashes(root: Path, paths: list[Path], *, role="plate") -> dict[Path, str]:
    """Resolve declared ancestry; unreadable or ambiguous provenance fails closed."""
    root = Path(root).resolve()
    rows = {}
    for filename in ("metadata_manifest.json", "split_assignment_manifest.json", "scene_split_assignment.json"):
        path = root / filename
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
            for row in manifest.get("items", []):
                if row.get("image_path"):
                    rows.setdefault((root / row["image_path"]).resolve(), []).append(row)
    path = root / "augmentation_manifest.json"
    if path.exists():
        for row in json.loads(path.read_text(encoding="utf-8-sig"))["generated_files"]:
            rows.setdefault((root / row["image"]).resolve(), []).append(
                {"source_image": row["source_image"], "_derived_artifact": True})
    result = {}
    for path in paths:
        identity = _resolve_scene(path.resolve(), root=root, rows=rows, role=role)
        if not identity and role != "character":
            raise ValueError(f"Nie można ustalić źródłowej sceny: {path.name}")
        result[path.resolve()] = identity
    return result


def assign_scene_splits(items: list[dict], ratios: dict, *, seed: int = 42,
                        source_hashes: dict[Path, str] | None = None) -> tuple[dict, dict]:
    """Preserve every sample; assign connected exact/source identities together."""
    if (not ratios or set(ratios) - {"train", "val", "test"}
            or any(not math.isfinite(float(v)) or float(v) < 0 for v in ratios.values())
            or abs(sum(ratios.values()) - 1) > .000001):
        raise ValueError("Proporcje train/val/test muszą być nieujemne i sumować się do 1.")
    if not items:
        raise ValueError("Brak obrazów do podziału.")
    prepared, destinations, parent = [], set(), {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for item in sorted(items, key=lambda row: str(row["name"]).casefold()):
        name = str(item["name"])
        if Path(name).name != name or Path(name).stem.casefold() in destinations:
            raise ValueError(f"Niejednoznaczna nazwa obrazu/etykiety: {name}")
        destinations.add(Path(name).stem.casefold())
        artifact = file_sha256(item["image"])
        declared = artifact if source_hashes is None else source_hashes.get(Path(item["image"]).resolve(), "")
        source = declared or artifact
        left, right = find(artifact), find(source)
        parent[max(left, right)] = min(left, right)
        prepared.append({**item, "artifact_sha256": artifact, "source_image_sha256": source,
                         "source_identity_known": bool(declared)})
    groups = {}
    for item in prepared:
        groups.setdefault(find(item["source_image_sha256"]), []).append(item)
    required = [name for name in ("train", "val") if ratios.get(name, 0) > 0]
    if len(groups) < len(required):
        raise ValueError("Za mało niezależnych scen SHA-256, aby wypełnić train i val bez przecieku.")
    targets = allocate_split_counts(len(items), ratios)
    for name in required:
        targets[name] = max(1, targets[name])
    eligible = [name for name in ratios if ratios[name] > 0 and targets[name] > 0]
    split_items = {name: [] for name in ratios}
    ordered = sorted(groups.items())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda pair: len(pair[1]), reverse=True)
    for index, (identity, members) in enumerate(ordered):
        empty_required = [name for name in required if not split_items[name]]
        choices = empty_required if len(ordered) - index == len(empty_required) else eligible
        target = min(choices, key=lambda name: len(split_items[name]) / targets[name])
        split_items[target].extend({**item, "group_id": identity} for item in members)
    manifest = {"schema": "alpr.scene_split_assignment.v1", "seed": seed,
                "grouping": "connected_source_scene_and_exact_file_sha256", "ratios": dict(ratios),
                "groups_total": len(groups), "cross_split_group_leakage": 0,
                "counts": {name: len(rows) for name, rows in split_items.items()}, "items": []}
    for split, rows in split_items.items():
        for row in rows:
            manifest["items"].append({"image_path": f"images/{split}/{row['name']}",
                "split": split, "group_id": row["group_id"],
                **({"source_image_sha256": row["source_image_sha256"]} if row["source_identity_known"] else {}),
                "artifact_sha256": row["artifact_sha256"]})
    return split_items, manifest


def check_new_split_destination(source: Path, output: Path):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Nowy split musi być poza katalogiem źródłowym.")
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise ValueError("Nowy split wymaga pustego katalogu; historyczne dane nie są nadpisywane.")


def save_scene_assignment(output: Path, manifest: dict):
    (Path(output) / "scene_split_assignment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
