"""Ground-truth companions for an image resource O.

A campaign image resource is the owner of optional portable GT packs.  This
module is intentionally UI-neutral: it discovers direct ``*.alprgt`` sidecars,
imports them into a project, stores the binding in the existing artifact
registry and resolves the binding again when Z2 opens the same image resource.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .gt_pack import ALPRGTPack, PACK_SCHEMA, normalize_polygon
from .registration_text import normalize_registration


COMPANION_SCHEMA = "alpr.image-resource.companions.v1"
GT_BINDING_SCHEMA = "alpr.gt.resource-binding.v1"
DEFAULT_WORKING_PACK_NAME = "current_work.alprgt"


def _path_key(path: Path | str | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path or "").replace("\\", "/").strip().lower()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _safe_name(value: str) -> str:
    prepared = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip(".-")
    return prepared or "ground-truth"


def _pack_semantic_counts(pack: ALPRGTPack) -> dict[str, int]:
    gt_set = 0
    gt_cleared = 0
    gt_conflicts = 0
    geometry_conflicts = 0
    layout_set = 0
    layout_conflicts = 0

    for plate in pack.list_plates():
        plate_id = str(plate.get("plate_id") or "").strip()
        if not plate_id:
            continue
        gt_state = dict(pack.resolve_ground_truth(plate_id) or {})
        if gt_state.get("conflict"):
            gt_conflicts += 1
        elif gt_state.get("resolved"):
            if str(gt_state.get("operation") or "").strip().lower() == "set" and gt_state.get("has_ground_truth"):
                gt_set += 1
            elif str(gt_state.get("operation") or "").strip().lower() == "clear":
                gt_cleared += 1

        geometry_state = dict(pack.resolve_plate_geometry(plate_id) or {})
        if geometry_state.get("conflict"):
            geometry_conflicts += 1

        resolver = getattr(pack, "resolve_plate_layout_gt", None)
        if callable(resolver):
            layout_state = dict(resolver(plate_id) or {})
            if layout_state.get("conflict"):
                layout_conflicts += 1
            elif layout_state.get("resolved") and layout_state.get("has_layout"):
                layout_set += 1

    return {
        "gt_set": int(gt_set),
        "gt_cleared": int(gt_cleared),
        "gt_conflicts": int(gt_conflicts),
        "geometry_conflicts": int(geometry_conflicts),
        "layout_set": int(layout_set),
        "layout_conflicts": int(layout_conflicts),
    }


def inspect_gt_pack(path: Path | str) -> dict[str, Any]:
    """Inspect one pack without deep blob validation.

    The pack API is required to keep ``open``/``validate(deep=False)`` read-only;
    discovery must be safe for source media mounted without write permission.
    """
    pack_path = Path(path)
    manifest_path = pack_path / "manifest.json"
    manifest = _read_json(manifest_path)
    result: dict[str, Any] = {
        "path": str(pack_path),
        "name": pack_path.name,
        "schema": str(manifest.get("schema") or ""),
        "version": int(manifest.get("version", 0) or 0),
        "valid": False,
        "issues": [],
    }
    if not pack_path.is_dir() or not manifest_path.is_file():
        result["issues"] = ["Brak katalogu packa lub manifest.json."]
        return result
    if result["schema"] != PACK_SCHEMA:
        result["issues"] = [f"Nieobsługiwany schemat: {result['schema'] or '-'}"]
        return result

    try:
        pack = ALPRGTPack.open(pack_path)
        validation = dict(pack.validate(deep=False) or {})
        summary_fn = getattr(pack, "summary")
        try:
            summary = dict(summary_fn(refresh_manifest=False) or {})
        except TypeError:
            summary = dict(summary_fn() or {})
        result.update(summary)
        result.update(_pack_semantic_counts(pack))
        result["valid"] = bool(validation.get("ok"))
        result["issues"] = [str(v) for v in list(validation.get("issues", []) or [])]
    except Exception as exc:
        result["issues"] = [str(exc)]
    return result



def ensure_working_gt_pack(
    image_dir: Path | str,
    *,
    producer: str = "auto_annotation_tool.desktop.z2",
) -> dict[str, Any]:
    """Create or safely reopen the one writable GT pack for an image folder.

    ``current_work.alprgt`` is the only automatic write target. Named packs are
    source/export artifacts and are not created here. Existing data is never
    removed or replaced by this operation.
    """
    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Nieprawidłowy katalog obrazów: {root}")

    target = root / DEFAULT_WORKING_PACK_NAME
    created = False

    if target.exists():
        if not target.is_dir() or not (target / "manifest.json").is_file():
            raise ValueError(
                f"{DEFAULT_WORKING_PACK_NAME} istnieje, ale nie jest poprawnym ALPR GT Pack."
            )
        try:
            pack = ALPRGTPack.open(target)
        except Exception as exc:
            raise ValueError(
                f"Nie można otworzyć {DEFAULT_WORKING_PACK_NAME}: {exc}"
            ) from exc
    else:
        pack = ALPRGTPack.create(target, producer=producer)
        created = True

    validation = dict(pack.validate(deep=False) or {})
    if not validation.get("ok"):
        issues = "; ".join(str(v) for v in list(validation.get("issues", []) or [])[:3])
        raise ValueError(
            f"{DEFAULT_WORKING_PACK_NAME} nie przeszedł walidacji"
            + (f": {issues}" if issues else ".")
        )

    try:
        summary = dict(pack.summary(refresh_manifest=False) or {})
    except TypeError:
        summary = dict(pack.summary() or {})

    return {
        "created": bool(created),
        "path": target,
        "summary": summary,
        "validation": validation,
    }

def discover_gt_pack_companions(
    image_dir: Path | str,
    *,
    include_default_working_pack: bool = False,
) -> dict[str, Any]:
    """Discover direct-child ``*.alprgt`` packs for one image resource.

    Discovery is deliberately non-recursive.  Nested training/run directories
    must never become accidental GT sources for the parent image resource.
    """
    root = Path(image_dir)
    result: dict[str, Any] = {
        "root": str(root),
        "schema": COMPANION_SCHEMA,
        "ground_truth": [],
        "invalid": [],
        "signature": "",
    }
    if not root.exists() or not root.is_dir():
        return result

    candidates: list[Path] = []
    try:
        children = sorted(root.iterdir(), key=lambda value: value.name.lower())
    except OSError:
        children = []
    for child in children:
        if not child.is_dir() or child.suffix.lower() != ".alprgt":
            continue
        if not include_default_working_pack and child.name.lower() == DEFAULT_WORKING_PACK_NAME:
            continue
        if not (child / "manifest.json").is_file():
            continue
        candidates.append(child)

    signature_parts: list[str] = []
    for candidate in candidates:
        manifest = candidate / "manifest.json"
        try:
            stat = manifest.stat()
            signature_parts.append(
                f"{candidate.name.lower()}:{int(stat.st_size)}:{int(getattr(stat, 'st_mtime_ns', 0) or 0)}"
            )
        except OSError:
            signature_parts.append(candidate.name.lower())

        info = inspect_gt_pack(candidate)
        if info.get("valid"):
            result["ground_truth"].append(info)
        else:
            result["invalid"].append(info)

    result["signature"] = hashlib.sha1("\n".join(signature_parts).encode("utf-8")).hexdigest() if signature_parts else ""
    return result


def _copy_pack(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".pack.lock"),
    )


def import_gt_pack_into_project(
    source_path: Path | str,
    project_root: Path | str,
) -> Path:
    source = Path(source_path)
    root = Path(project_root)
    manifest = source / "manifest.json"
    if not source.is_dir() or not manifest.is_file():
        raise ValueError(f"Nieprawidłowy GT Pack: {source}")

    try:
        source_identity = str(source.resolve())
    except Exception:
        source_identity = str(source)
    digest = hashlib.sha1(source_identity.encode("utf-8")).hexdigest()[:10]
    stem = _safe_name(source.stem)
    target = root / "_campaign_state" / "ground_truth" / "imported" / f"{stem}-{digest}.alprgt"
    _copy_pack(source, target)

    imported = inspect_gt_pack(target)
    if not imported.get("valid"):
        issues = "; ".join(imported.get("issues", [])[:3])
        raise ValueError(f"Skopiowany GT Pack nie przeszedł walidacji: {issues}")
    return target


def _project_relpath(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return ""


def _binding_entry(path: Path, project_root: Path, info: dict | None = None) -> dict[str, Any]:
    summary = dict(info or {})
    return {
        "schema": GT_BINDING_SCHEMA,
        "pack_schema": str(summary.get("schema") or PACK_SCHEMA),
        "name": path.name,
        "project_relpath": _project_relpath(path, project_root),
        "mode": "read_only",
        "origin": "image_resource_companion",
        "images": int(summary.get("images", 0) or 0),
        "plates": int(summary.get("plates", 0) or 0),
        "gt_set": int(summary.get("gt_set", 0) or 0),
        "layout_set": int(summary.get("layout_set", 0) or 0),
    }


def bind_campaign_gt_companions(
    campaign,
    *,
    image_dir: Path | str,
    imported_packs: Iterable[Path | str],
    iteration_num: int | None = None,
    info_by_path: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Bind imported source packs to the current image-resource package."""
    project_root = campaign.get_active_project_root_dir()
    if project_root is None:
        return {}
    project_root = Path(project_root)
    image_dir = Path(image_dir)
    try:
        iteration = int(iteration_num or campaign.get_current_iteration_num() or 1)
    except Exception:
        iteration = 1

    bundle = dict(
        campaign.get_iteration_artifact_bundle(
            images_dir=image_dir,
            iteration_num=iteration,
        )
        or {}
    )
    image_source = dict(bundle.get("image_source") or {})
    image_set_token = str(
        image_source.get("image_set_token")
        or campaign.get_iteration_image_set_token(iteration)
        or ""
    ).strip()
    package_id = str(bundle.get("package_id") or image_source.get("package_id") or "").strip()

    info_map = info_by_path or {}
    entries: list[dict[str, Any]] = []
    for raw in imported_packs:
        path = Path(raw)
        key = _path_key(path)
        info = info_map.get(key)
        if info is None:
            try:
                info = inspect_gt_pack(path)
            except Exception:
                info = {}
        entries.append(_binding_entry(path, project_root, info))

    companions = {
        "schema": COMPANION_SCHEMA,
        "package_id": package_id,
        "image_set_token": image_set_token,
        "ground_truth": entries,
    }
    return dict(
        campaign.upsert_iteration_artifact_bundle(
            images_dir=image_dir,
            iteration_num=iteration,
            image_set_token=image_set_token,
            updates={"image_source": {"companions": companions}},
        )
        or {}
    )


def import_and_bind_campaign_gt_companions(
    campaign,
    *,
    image_dir: Path | str,
    discoveries: Iterable[dict],
    iteration_num: int | None = None,
) -> dict[str, Any]:
    project_root = campaign.get_active_project_root_dir()
    if project_root is None:
        return {}
    imported: list[Path] = []
    info_by_path: dict[str, dict] = {}
    for item in list(discoveries or []):
        raw_source = str(item.get("path") or "").strip()
        if not raw_source:
            continue
        source = Path(raw_source)
        target = import_gt_pack_into_project(source, project_root)
        imported.append(target)
        info_by_path[_path_key(target)] = dict(item)

    bundle = bind_campaign_gt_companions(
        campaign,
        image_dir=image_dir,
        imported_packs=imported,
        iteration_num=iteration_num,
        info_by_path=info_by_path,
    )
    return {
        "bundle": bundle,
        "paths": imported,
    }


def get_campaign_gt_companion_paths(
    campaign,
    *,
    image_dir: Path | str,
    iteration_num: int | None = None,
) -> list[Path]:
    project_root = campaign.get_active_project_root_dir()
    if project_root is None:
        return []
    project_root = Path(project_root)
    try:
        iteration = int(iteration_num or campaign.get_current_iteration_num() or 1)
    except Exception:
        iteration = 1

    bundle = dict(
        campaign.get_iteration_artifact_bundle(
            images_dir=Path(image_dir),
            iteration_num=iteration,
        )
        or {}
    )
    image_source = dict(bundle.get("image_source") or {})
    companions = dict(image_source.get("companions") or {})
    if companions.get("schema") not in (None, "", COMPANION_SCHEMA):
        return []

    bundle_package_id = str(bundle.get("package_id") or image_source.get("package_id") or "").strip()
    bound_package_id = str(companions.get("package_id") or "").strip()
    if bundle_package_id and bound_package_id and bundle_package_id != bound_package_id:
        return []

    expected_token = str(image_source.get("image_set_token") or "").strip()
    bound_token = str(companions.get("image_set_token") or "").strip()
    if expected_token and bound_token and expected_token != bound_token:
        return []

    # get_iteration_artifact_bundle() has a legacy iteration-index fallback.
    # Reject such a fallback when it points at a different image resource.
    requested_key = _path_key(image_dir)
    known_resource_paths = {
        _path_key(value)
        for value in (
            bundle.get("images_dir"),
            image_source.get("master_pool_dir"),
        )
        if str(value or "").strip()
    }
    if requested_key and known_resource_paths and requested_key not in known_resource_paths:
        return []

    result: list[Path] = []
    seen: set[str] = set()
    for entry in list(companions.get("ground_truth", []) or []):
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("project_relpath") or "").strip()
        if not rel:
            continue
        path = project_root / Path(rel)
        key = _path_key(path)
        if key and key not in seen and path.is_dir() and (path / "manifest.json").is_file():
            seen.add(key)
            result.append(path)
    return result


def collect_image_resource_gt_pack_paths(
    image_dir: Path | str,
    *,
    campaign=None,
    iteration_num: int | None = None,
) -> list[Path]:
    """Collect readable GT packs that belong to one image resource.

    Direct-child packs (including ``current_work.alprgt``) are considered first.
    In campaign mode imported O-companions and the project working pack are added.
    Returned paths are deduplicated and must contain a manifest.
    """
    root = Path(image_dir)
    candidates: list[Path] = []

    try:
        discovery = discover_gt_pack_companions(
            root,
            include_default_working_pack=True,
        )
    except Exception:
        discovery = {}

    for item in list(discovery.get("ground_truth", []) or []):
        raw = str((item or {}).get("path") or "").strip()
        if raw:
            candidates.append(Path(raw))

    if campaign is not None:
        try:
            active_name = str(campaign.get_active_project_name() or "").strip()
        except Exception:
            active_name = ""
        if active_name:
            try:
                candidates.extend(
                    get_campaign_gt_companion_paths(
                        campaign,
                        image_dir=root,
                        iteration_num=iteration_num,
                    )
                )
            except Exception:
                pass
            try:
                project_root = campaign.get_active_project_root_dir()
            except Exception:
                project_root = None
            if project_root is not None:
                working = (
                    Path(project_root)
                    / "_campaign_state"
                    / "ground_truth"
                    / DEFAULT_WORKING_PACK_NAME
                )
                candidates.append(working)

    result: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        path = Path(raw)
        key = _path_key(path)
        if not key or key in seen:
            continue
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        seen.add(key)
        result.append(path)
    return result


def open_gt_pack_sources(paths: Iterable[Path | str]) -> list[tuple[Path, ALPRGTPack]]:
    """Open validated read sources once for batch provenance resolution."""
    result: list[tuple[Path, ALPRGTPack]] = []
    seen: set[str] = set()
    for raw in list(paths or []):
        path = Path(raw)
        key = _path_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            pack = ALPRGTPack.open(path)
            validation = dict(pack.validate(deep=False) or {})
        except Exception:
            continue
        if validation.get("ok"):
            result.append((path, pack))
    return result


def _normalized_points_equal(left, right, *, tolerance: float = 1e-7) -> bool:
    left_points = list(left or [])
    right_points = list(right or [])
    if len(left_points) != len(right_points) or not left_points:
        return False
    try:
        return all(
            abs(float(lp[0]) - float(rp[0])) <= tolerance
            and abs(float(lp[1]) - float(rp[1])) <= tolerance
            for lp, rp in zip(left_points, right_points)
        )
    except Exception:
        return False


def resolve_plate_revision_provenance(
    opened_sources,
    *,
    image_id: str,
    plate_id: str,
    polygon,
    image_width: int,
    image_height: int,
    ground_truth_text: str | None = None,
) -> dict[str, Any]:
    """Resolve revision heads only when portable pack state matches PZ1 input.

    Geometry IDs are exposed only when the current source polygon exactly matches
    the resolved pack geometry. GT revision IDs are exposed only for a resolved
    SET whose normalized value equals the current Z2 GT. Any relevant conflict
    suppresses that provenance instead of choosing a silent winner.
    """
    safe_image_id = str(image_id or "").strip()
    safe_plate_id = str(plate_id or "").strip()
    expected_gt = normalize_registration(ground_truth_text)

    result = {
        "geometry_revision_ids": [],
        "ground_truth_revision_ids": [],
        "source_geometry_revision_id": None,
        "source_gt_revision_id": None,
        "geometry_conflict": False,
        "ground_truth_conflict": False,
        "matched_packs": 0,
    }
    if not safe_image_id or not safe_plate_id:
        return result

    try:
        normalized_polygon = normalize_polygon(
            polygon,
            image_width=int(image_width),
            image_height=int(image_height),
        )
    except Exception:
        normalized_polygon = []

    geometry_ids: set[str] = set()
    gt_ids: set[str] = set()

    for _path, pack in list(opened_sources or []):
        try:
            plate = pack.get_plate(safe_plate_id)
        except Exception:
            plate = None
        if not isinstance(plate, dict):
            continue
        if str(plate.get("image_id") or "").strip() != safe_image_id:
            continue

        result["matched_packs"] += 1

        try:
            geometry = dict(pack.resolve_plate_geometry(safe_plate_id) or {})
        except Exception:
            geometry = {}
        if geometry.get("conflict"):
            result["geometry_conflict"] = True
        elif (
            normalized_polygon
            and geometry.get("resolved")
            and _normalized_points_equal(
                geometry.get("points", []),
                normalized_polygon,
            )
        ):
            for value in list(geometry.get("geometry_ids", []) or []):
                prepared = str(value or "").strip()
                if prepared:
                    geometry_ids.add(prepared)

        try:
            gt_state = dict(pack.resolve_ground_truth(safe_plate_id) or {})
        except Exception:
            gt_state = {}
        if gt_state.get("conflict"):
            result["ground_truth_conflict"] = True
        elif (
            expected_gt
            and gt_state.get("resolved")
            and str(gt_state.get("operation") or "").strip().lower() == "set"
            and normalize_registration(gt_state.get("text")) == expected_gt
        ):
            for value in list(gt_state.get("revision_ids", []) or []):
                prepared = str(value or "").strip()
                if prepared:
                    gt_ids.add(prepared)

    if not result["geometry_conflict"]:
        result["geometry_revision_ids"] = sorted(geometry_ids)
        if len(geometry_ids) == 1:
            result["source_geometry_revision_id"] = next(iter(geometry_ids))

    if not result["ground_truth_conflict"]:
        result["ground_truth_revision_ids"] = sorted(gt_ids)
        if len(gt_ids) == 1:
            result["source_gt_revision_id"] = next(iter(gt_ids))

    return result
