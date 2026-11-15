"""Z2 adapter for portable ALPR GT Pack v1."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..campaign_manager import CAMPAIGN
from ..config import SESSION, logger
from ..gt_pack import (
    ALPRGTPack,
    IMAGE_ID_PREFIX,
    denormalize_polygon,
    normalize_polygon,
)
from ..data_models import Detection
from ..plate_ground_truth import (
    GROUND_TRUTH_SOURCE_MANUAL_Z2,
    PLATE_ANNOTATION_ID_ATTR,
    ensure_plate_detection_contract,
    get_plate_ground_truth,
    get_plate_layout_gt,
    has_explicit_plate_layout_gt,
    normalize_plate_ground_truth_text,
    normalize_plate_layout_gt,
    set_plate_ground_truth,
    set_plate_layout_gt,
)

PRODUCER = "auto_annotation_tool.desktop.z2"
OUTBOX_SCHEMA = "alpr.gt.sync_outbox.v1"
MOUNTS_SCHEMA = "alpr.gt.mounts.v1"
PROJECT_REF_PREFIX = "project://"
GT_PACK_STATUS_IO_CACHE_TTL_S = 0.45


def _safe_path(value):
    raw = str(value or "").strip()
    return Path(raw) if raw else None


def _path_key(path):
    if path is None:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path or "").replace("\\", "/").strip().lower()



def _active_project_root() -> Path | None:
    try:
        name = str(CAMPAIGN.get_active_project_name() or "").strip()
        root = CAMPAIGN.get_active_project_root_dir() if name else None
    except Exception:
        root = None
    return Path(root) if root is not None else None


def _mount_config_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "_campaign_state"
        / "ground_truth"
        / "mounts.json"
    )


def _encode_mount_path(
    path: Path | str | None,
    project_root: Path | None,
) -> str:
    if path is None or not str(path).strip():
        return ""
    value = Path(path)
    if project_root is not None:
        try:
            relative = value.resolve().relative_to(project_root.resolve())
            return PROJECT_REF_PREFIX + relative.as_posix()
        except Exception:
            pass
    try:
        return str(value.resolve())
    except Exception:
        return str(value)


def _decode_mount_path(
    value,
    project_root: Path | None,
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.startswith(PROJECT_REF_PREFIX):
        if project_root is None:
            return None
        relative = raw[len(PROJECT_REF_PREFIX):].lstrip("/\\")
        return Path(project_root) / Path(relative)
    return Path(raw)


def _default_project_working_pack(
    project_root: Path | None,
) -> Path | None:
    if project_root is None:
        return None
    return (
        Path(project_root)
        / "_campaign_state"
        / "ground_truth"
        / "current_work.alprgt"
    )


def _load_mount_payload(host) -> dict:
    project_root = _active_project_root()
    if project_root is not None:
        path = _mount_config_path(project_root)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
    else:
        try:
            payload = (
                SESSION.get("annotation", "gt_pack_mounts", {})
                if SESSION
                else {}
            )
        except Exception:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}
    if payload.get("schema") not in (None, "", MOUNTS_SCHEMA):
        payload = {}

    source_values = payload.get("source_paths", [])
    if not isinstance(source_values, list):
        source_values = []
    working_value = payload.get("working_path", "")

    sources = []
    seen = set()
    for raw in source_values:
        path = _decode_mount_path(raw, project_root)
        if path is None:
            continue
        key = _path_key(path)
        if key and key not in seen:
            seen.add(key)
            sources.append(path)

    working = _decode_mount_path(working_value, project_root)
    if working is None:
        working = _default_project_working_pack(project_root)

    return {
        "schema": MOUNTS_SCHEMA,
        "source_paths": sources,
        "working_path": working,
    }


def _persist_mount_payload(host) -> None:
    project_root = _active_project_root()
    sources = [
        Path(path)
        for path in list(
            getattr(host, "_z2_gt_pack_source_paths", []) or []
        )
        if str(path or "").strip()
    ]
    working_raw = str(
        getattr(host, "_z2_gt_working_pack_path", "") or ""
    ).strip()
    working = Path(working_raw) if working_raw else None

    payload = {
        "schema": MOUNTS_SCHEMA,
        "source_paths": [
            _encode_mount_path(path, project_root)
            for path in sources
        ],
        "working_path": _encode_mount_path(
            working,
            project_root,
        ),
    }

    if project_root is not None:
        path = _mount_config_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
        return

    try:
        if SESSION:
            SESSION.set("annotation", "gt_pack_mounts", payload)
            SESSION.save_session()
    except Exception as exc:
        logger.debug("Nie udało się zapisać mountów GT Pack: %s", exc)


def _mount_context_key() -> str:
    project_root = _active_project_root()
    if project_root is None:
        return "free"
    return f"project:{_path_key(project_root)}"


def _ensure_mount_config_loaded(host) -> None:
    """Load persisted manual mounts for the current project/free context.

    The previous implementation cached mounts for the lifetime of AnnotationTab,
    which allowed a project switch to retain another project's PACK state.
    Explicit values injected before the first load remain authoritative, while a
    context switch always reloads the proper project/session payload.
    """
    context_key = _mount_context_key()
    loaded = bool(getattr(host, "_z2_gt_mount_config_loaded", False))
    loaded_key = str(getattr(host, "_z2_gt_mount_context_key", "") or "")
    # Backward compatibility: callers/tests from the pre-context runtime may
    # mark the payload as loaded without recording a context key. Adopt the
    # current context once instead of discarding an explicitly injected setup.
    if loaded and not loaded_key:
        host._z2_gt_mount_context_key = context_key
        return
    if loaded and loaded_key == context_key:
        return
    if loaded and loaded_key != context_key:
        host._z2_gt_pack_source_paths = []
        host._z2_gt_working_pack_path = ""
        host._z2_gt_resource_source_paths = []
        host._z2_gt_resource_working_pack_path = ""
        host._z2_gt_resource_key = ""
        host._z2_gt_mount_config_loaded = False

    raw_sources = getattr(host, "_z2_gt_pack_source_paths", None)
    raw_working = getattr(host, "_z2_gt_working_pack_path", None)

    if isinstance(raw_sources, (list, tuple, set)):
        explicit_sources = [
            str(Path(value))
            for value in raw_sources
            if str(value or "").strip()
        ]
    elif str(raw_sources or "").strip():
        explicit_sources = [str(Path(raw_sources))]
    else:
        explicit_sources = []

    explicit_working = str(raw_working or "").strip()

    if explicit_sources or explicit_working:
        host._z2_gt_pack_source_paths = explicit_sources
        host._z2_gt_working_pack_path = (
            str(Path(explicit_working))
            if explicit_working
            else ""
        )
        host._z2_gt_mount_config_loaded = True
        host._z2_gt_mount_context_key = context_key
        return

    payload = _load_mount_payload(host)
    host._z2_gt_pack_source_paths = [
        str(path)
        for path in payload.get("source_paths", [])
    ]
    working = payload.get("working_path")
    host._z2_gt_working_pack_path = (
        str(working) if working is not None else ""
    )
    host._z2_gt_mount_config_loaded = True
    host._z2_gt_mount_context_key = context_key

def get_configured_source_pack_paths(host) -> list[Path]:
    _ensure_mount_config_loaded(host)
    result = []
    seen = set()
    for raw in list(
        getattr(host, "_z2_gt_pack_source_paths", []) or []
    ):
        path = _safe_path(raw)
        if path is None:
            continue
        key = _path_key(path)
        if key and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def get_resource_source_pack_paths(host) -> list[Path]:
    result = []
    seen = set()
    for raw in list(getattr(host, "_z2_gt_resource_source_paths", []) or []):
        path = _safe_path(raw)
        if path is None:
            continue
        key = _path_key(path)
        if key and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _get_all_source_pack_paths(host) -> list[Path]:
    values = list(get_configured_source_pack_paths(host)) + list(get_resource_source_pack_paths(host))
    result = []
    seen = set()
    for path in values:
        key = _path_key(path)
        if key and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def set_gt_resource_binding(
    host,
    *,
    source_paths=None,
    working_path=None,
    resource_key: str = "",
) -> None:
    """Set non-global GT sources owned by the current image resource O."""
    host._z2_gt_resource_source_paths = [
        str(Path(path))
        for path in list(source_paths or [])
        if str(path or "").strip()
    ]
    host._z2_gt_resource_working_pack_path = (
        str(Path(working_path))
        if str(working_path or "").strip()
        else ""
    )
    host._z2_gt_resource_key = str(resource_key or "").strip()
    invalidate_gt_pack_status_cache(host)


def get_working_gt_pack_path(
    host,
    *,
    create_parent: bool = False,
) -> Path | None:
    _ensure_mount_config_loaded(host)

    explicit = _safe_path(
        getattr(host, "_z2_gt_working_pack_path", None)
    )
    if explicit is None:
        explicit = _safe_path(
            getattr(host, "_z2_gt_resource_working_pack_path", None)
        )
    if explicit is None:
        explicit = _default_project_working_pack(
            _active_project_root()
        )
        if explicit is not None:
            host._z2_gt_working_pack_path = str(explicit)

    if explicit is not None and create_parent:
        explicit.parent.mkdir(parents=True, exist_ok=True)
    return explicit

def get_mounted_gt_pack_paths(host) -> list[Path]:
    _ensure_mount_config_loaded(host)
    values = list(_get_all_source_pack_paths(host))

    working = get_working_gt_pack_path(
        host,
        create_parent=False,
    )
    if working is not None:
        values.append(working)

    result = []
    seen = set()
    for raw in values:
        path = _safe_path(raw)
        if path is None:
            continue
        key = _path_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        if path.exists() and (path / "manifest.json").is_file():
            result.append(path)
    return result

def set_gt_pack_mounts(
    host,
    *,
    source_paths=None,
    working_path=None,
    persist: bool = True,
) -> None:
    host._z2_gt_pack_source_paths = [
        str(Path(path))
        for path in list(source_paths or [])
        if str(path or "").strip()
    ]
    host._z2_gt_working_pack_path = (
        str(Path(working_path))
        if str(working_path or "").strip()
        else ""
    )
    host._z2_gt_mount_config_loaded = True
    host._z2_gt_mount_context_key = _mount_context_key()
    invalidate_gt_pack_status_cache(host)
    if persist:
        _persist_mount_payload(host)


def invalidate_gt_pack_status_cache(host) -> None:
    host._z2_gt_pack_status_io_cache = None


def _get_gt_pack_status_io_snapshot(host) -> dict:
    now = time.monotonic()
    cached = getattr(host, "_z2_gt_pack_status_io_cache", None)
    if isinstance(cached, dict):
        try:
            age = now - float(cached.get("created_at", 0.0) or 0.0)
        except Exception:
            age = GT_PACK_STATUS_IO_CACHE_TTL_S + 1.0
        if 0.0 <= age <= GT_PACK_STATUS_IO_CACHE_TTL_S:
            return dict(cached)

    sources = _get_all_source_pack_paths(host)
    working = get_working_gt_pack_path(host, create_parent=False)
    try:
        pending_count = len(dict(_load_outbox(host).get("items", {}) or {}))
    except Exception:
        pending_count = 0

    snapshot = {
        "created_at": now,
        "source_count": int(len(sources)),
        "working_path": str(working or ""),
        "working_enabled": bool(working is not None),
        "pending_count": int(pending_count),
    }
    host._z2_gt_pack_status_io_cache = dict(snapshot)
    return snapshot


def get_gt_pack_status(host) -> dict:
    _ensure_mount_config_loaded(host)
    io_state = _get_gt_pack_status_io_snapshot(host)
    source_count = int(io_state.get("source_count", 0) or 0)
    working_path = str(io_state.get("working_path", "") or "")
    working_enabled = bool(io_state.get("working_enabled", False))
    pending_count = int(io_state.get("pending_count", 0) or 0)

    report = dict(
        getattr(host, "_z2_gt_last_restore_report", {}) or {}
    )
    conflict_count = len(
        list(report.get("conflicts", []) or [])
    )

    enabled = bool(source_count > 0 or working_enabled)
    if conflict_count > 0:
        tone = "conflict"
        text = f"PACK C{conflict_count}"
    elif pending_count > 0:
        tone = "pending"
        text = f"PACK !{pending_count}"
    elif enabled:
        tone = "ok"
        text = "PACK OK"
    else:
        tone = "off"
        text = "PACK OFF"

    return {
        "enabled": enabled,
        "source_count": int(source_count),
        "working_path": working_path,
        "pending_count": int(pending_count),
        "conflict_count": int(conflict_count),
        "tone": tone,
        "text": text,
    }


def is_gt_pack_status_hit(
    host,
    canvas_x: float,
    canvas_y: float,
) -> bool:
    bbox = getattr(host, "_z2_gt_pack_status_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return (
        x1 <= float(canvas_x) <= x2
        and y1 <= float(canvas_y) <= y2
    )


def _open_mounted_packs(host):
    result = []
    for path in get_mounted_gt_pack_paths(host):
        try:
            result.append((path, ALPRGTPack.open(path)))
        except Exception as exc:
            logger.debug("Nie udało się zamontować GT Pack %s: %s", path, exc)
    return result


def _fingerprint_cache(host):
    cache = getattr(host, "_z2_gt_image_fingerprint_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        host._z2_gt_image_fingerprint_cache = cache
    return cache


def fingerprint_preview_image(host, image_path):
    path = Path(image_path)
    stat = path.stat()
    cache_key = (
        _path_key(path),
        int(stat.st_size),
        int(getattr(stat, "st_mtime_ns", 0) or 0),
    )
    cache = _fingerprint_cache(host)
    if isinstance(cache.get(cache_key), dict):
        return dict(cache[cache_key])

    # Restoring annotations needs exact file identity and oriented dimensions,
    # not pixel/perceptual fingerprints used when writing a portable GT pack.
    # Reading those fingerprints decoded a full image again on every cold Q/E.
    import hashlib
    from PIL import Image

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as source:
        width, height = source.size
        if source.getexif().get(274) in {5, 6, 7, 8}:
            width, height = height, width
    source_hash = digest.hexdigest()
    value = {"image_id": IMAGE_ID_PREFIX + source_hash, "source_file_sha256": source_hash,
             "width": width, "height": height}
    cache[cache_key] = dict(value)
    if len(cache) > 256:
        for key in list(cache.keys())[:-192]:
            cache.pop(key, None)
    return dict(value)


def _resolve_annotation_image_path(host, ann):
    try:
        resolved = host._resolve_preview_image_path(ann)
    except Exception:
        resolved = None
    if resolved is None:
        return None
    path = Path(resolved)
    return path if path.exists() and path.is_file() else None


def _detection_polygon(host, det):
    try:
        points = list(host._detection_polygon(det) or [])
    except Exception:
        points = list(getattr(det, "polygon", None) or [])
    result = []
    for point in points[:4]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            result.append((float(point[0]), float(point[1])))
    return result


def _normalized_detection_polygon(host, det, *, width, height):
    polygon = _detection_polygon(host, det)
    if len(polygon) < 4:
        return None
    try:
        return normalize_polygon(
            polygon,
            image_width=int(width),
            image_height=int(height),
        )
    except Exception:
        return None


def _points_equal(left, right, *, tolerance=1e-7):
    left = list(left or [])
    right = list(right or [])
    if len(left) != len(right) or not left:
        return False
    try:
        return all(
            abs(float(lp[0]) - float(rp[0])) <= tolerance
            and abs(float(lp[1]) - float(rp[1])) <= tolerance
            for lp, rp in zip(left, right)
        )
    except Exception:
        return False


def _pack_gt_state(pack, plate_id):
    state = dict(pack.resolve_ground_truth(plate_id) or {})
    if state.get("conflict"):
        return {"kind": "conflict", "state": state}

    if not state.get("resolved"):
        return {"kind": "missing", "state": state}

    operation = str(state.get("operation") or "").strip().lower()
    has_gt = bool(state.get("has_ground_truth", False))
    if operation == "clear" or not has_gt:
        return {
            "kind": "clear",
            "text": "",
            "revision_ids": list(state.get("revision_ids", []) or []),
            "state": state,
        }

    text = normalize_plate_ground_truth_text(state.get("text"))
    revision_ids = list(state.get("revision_ids", []) or [])
    source = ""
    if revision_ids:
        try:
            revision = pack.get_revision(revision_ids[0]) or {}
            source = str(revision.get("source") or "").strip()
        except Exception:
            source = ""

    return {
        "kind": "set",
        "text": text,
        "source": source or GROUND_TRUTH_SOURCE_MANUAL_Z2,
        "revision_ids": revision_ids,
        "state": state,
    }


def _resolve_same_plate_across_packs(mounted, *, image_id, plate_id):
    observations = []
    for pack_path, pack in mounted:
        plate = pack.get_plate(plate_id)
        if not isinstance(plate, dict):
            continue
        if str(plate.get("image_id") or "").strip() != image_id:
            observations.append({
                "pack": str(pack_path),
                "kind": "identity_conflict",
            })
            continue
        item = _pack_gt_state(pack, plate_id)
        item["pack"] = str(pack_path)
        observations.append(item)

    if not observations:
        return {"found": False, "conflict": False, "observations": []}

    if any(
        item.get("kind") in {"conflict", "identity_conflict"}
        for item in observations
    ):
        return {
            "found": True,
            "conflict": True,
            "reason": "mounted_pack_conflict",
            "observations": observations,
        }

    semantic = set()
    for item in observations:
        if item.get("kind") == "set":
            semantic.add(("set", str(item.get("text") or "")))
        elif item.get("kind") == "clear":
            semantic.add(("clear", ""))

    if len(semantic) > 1:
        return {
            "found": True,
            "conflict": True,
            "reason": "mounted_pack_semantic_conflict",
            "observations": observations,
        }

    if not semantic:
        return {
            "found": True,
            "conflict": False,
            "kind": "missing",
            "observations": observations,
        }

    operation, text = next(iter(semantic))
    source = ""
    revision_ids = []
    if operation == "set":
        for item in observations:
            if item.get("kind") == "set" and item.get("text") == text:
                source = str(item.get("source") or "").strip()
                revision_ids.extend(item.get("revision_ids", []) or [])

    return {
        "found": True,
        "conflict": False,
        "kind": operation,
        "text": text,
        "source": source or GROUND_TRUTH_SOURCE_MANUAL_Z2,
        "revision_ids": sorted(set(str(v) for v in revision_ids if str(v))),
        "observations": observations,
    }


def _find_exact_geometry_candidates(mounted, *, image_id, normalized_points):
    result = {}
    for pack_path, pack in mounted:
        image = pack.get_image(image_id)
        if not isinstance(image, dict):
            continue
        for raw_plate_id in list(image.get("plate_ids", []) or []):
            plate_id = str(raw_plate_id or "").strip()
            if not plate_id:
                continue
            geometry = dict(pack.resolve_plate_geometry(plate_id) or {})
            if not geometry.get("resolved"):
                continue
            if not _points_equal(geometry.get("points", []), normalized_points):
                continue
            item = result.setdefault(
                plate_id,
                {"plate_id": plate_id, "packs": []},
            )
            item["packs"].append(str(pack_path))
    return sorted(result.values(), key=lambda item: item["plate_id"])


def _pack_layout_state(pack, plate_id):
    resolver = getattr(pack, "resolve_plate_layout_gt", None)
    if not callable(resolver):
        return {"kind": "missing", "state": {}}
    state = dict(resolver(plate_id) or {})
    if state.get("conflict"):
        return {"kind": "conflict", "state": state}
    if not state.get("resolved"):
        return {"kind": "missing", "state": state}
    operation = str(state.get("operation") or "").strip().lower()
    has_layout = bool(state.get("has_layout", False))
    if operation == "clear" or not has_layout:
        return {"kind": "clear", "layout": "", "state": state}
    layout = normalize_plate_layout_gt(state.get("layout"), default="")
    return {"kind": "set", "layout": layout, "state": state}


def _resolve_same_plate_layout_across_packs(mounted, *, image_id, plate_id):
    observations = []
    for pack_path, pack in mounted:
        plate = pack.get_plate(plate_id)
        if not isinstance(plate, dict):
            continue
        if str(plate.get("image_id") or "").strip() != image_id:
            observations.append({"pack": str(pack_path), "kind": "identity_conflict"})
            continue
        item = _pack_layout_state(pack, plate_id)
        item["pack"] = str(pack_path)
        observations.append(item)

    if not observations:
        return {"found": False, "conflict": False, "observations": []}
    if any(item.get("kind") in {"conflict", "identity_conflict"} for item in observations):
        return {
            "found": True,
            "conflict": True,
            "reason": "mounted_pack_layout_conflict",
            "observations": observations,
        }

    semantic = set()
    for item in observations:
        if item.get("kind") == "set":
            semantic.add(("set", str(item.get("layout") or "")))
        elif item.get("kind") == "clear":
            semantic.add(("clear", ""))
    if len(semantic) > 1:
        return {
            "found": True,
            "conflict": True,
            "reason": "mounted_pack_layout_semantic_conflict",
            "observations": observations,
        }
    if not semantic:
        return {"found": True, "conflict": False, "kind": "missing", "observations": observations}
    operation, layout = next(iter(semantic))
    return {
        "found": True,
        "conflict": False,
        "kind": operation,
        "layout": layout,
        "observations": observations,
    }


def _collect_hydration_candidates(mounted, *, image_id):
    by_plate = {}
    conflicts = []
    for pack_path, pack in mounted:
        image = pack.get_image(image_id)
        if not isinstance(image, dict):
            continue
        for raw_plate_id in list(image.get("plate_ids", []) or []):
            plate_id = str(raw_plate_id or "").strip()
            if not plate_id:
                continue
            plate = pack.get_plate(plate_id)
            if not isinstance(plate, dict) or str(plate.get("image_id") or "").strip() != image_id:
                conflicts.append({
                    "plate_id": plate_id,
                    "reason": "pack_plate_image_identity",
                    "pack": str(pack_path),
                })
                continue
            geometry = dict(pack.resolve_plate_geometry(plate_id) or {})
            if geometry.get("conflict"):
                conflicts.append({
                    "plate_id": plate_id,
                    "reason": "pack_geometry_conflict",
                    "pack": str(pack_path),
                })
                continue
            if not geometry.get("resolved"):
                continue
            points = list(geometry.get("points", []) or [])
            if len(points) < 4:
                continue
            by_plate.setdefault(plate_id, []).append({
                "pack": str(pack_path),
                "points": points,
            })

    candidates = []
    for plate_id, observations in sorted(by_plate.items()):
        unique_points = []
        for observation in observations:
            points = observation.get("points", [])
            if not any(_points_equal(points, existing) for existing in unique_points):
                unique_points.append(points)
        if len(unique_points) > 1:
            conflicts.append({
                "plate_id": plate_id,
                "reason": "mounted_pack_geometry_semantic_conflict",
                "packs": [item.get("pack") for item in observations],
            })
            continue
        if unique_points:
            candidates.append({"plate_id": plate_id, "points": unique_points[0]})
    return candidates, conflicts


def _geometries_overlap(left, right, *, threshold=0.8):
    """Recognize a corrected copy of the same plate without rebinding its GT."""
    if _points_equal(left, right):
        return True
    import cv2
    import numpy as np

    a = cv2.convexHull(np.asarray(left, dtype=np.float32).reshape(-1, 2))
    b = cv2.convexHull(np.asarray(right, dtype=np.float32).reshape(-1, 2))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return bool(union > 0 and intersection / union >= threshold)


def hydrate_annotations_from_gt_pack(host, ann):
    """Create missing plate detections from portable pack geometry.

    Hydration is idempotent.  Existing plate IDs win; an exact local geometry
    match is left in place so the normal restore pass can rebind its ID without
    duplicating the polygon.
    """
    report = {
        "enabled": False,
        "changed": False,
        "created": 0,
        "skipped_overlapping": [],
        "conflicts": [],
        "image_id": "",
    }
    mounted = _open_mounted_packs(host)
    if not mounted:
        return report
    report["enabled"] = True

    image_path = _resolve_annotation_image_path(host, ann)
    if image_path is None:
        report["reason"] = "missing_image"
        return report
    try:
        fingerprint = fingerprint_preview_image(host, image_path)
    except Exception as exc:
        report["reason"] = "fingerprint_failed"
        report["error"] = str(exc)
        return report

    image_id = str(fingerprint.get("image_id") or "")
    report["image_id"] = image_id
    width = int(fingerprint.get("width", 0) or 0)
    height = int(fingerprint.get("height", 0) or 0)
    candidates, conflicts = _collect_hydration_candidates(mounted, image_id=image_id)
    report["conflicts"].extend(conflicts)

    try:
        local_plates = list(host._get_plate_detections(ann) or [])
    except Exception:
        local_plates = []
    local_ids = {
        str(dict(getattr(det, "attributes", {}) or {}).get(PLATE_ANNOTATION_ID_ATTR) or "").strip()
        for det in local_plates
    }
    local_geometries = []
    for det in local_plates:
        normalized = _normalized_detection_polygon(host, det, width=width, height=height)
        if normalized:
            local_geometries.append(normalized)

    detections = list(getattr(ann, "detections", []) or [])
    for candidate in candidates:
        plate_id = str(candidate.get("plate_id") or "").strip()
        normalized_points = list(candidate.get("points", []) or [])
        if not plate_id or plate_id in local_ids:
            continue
        if any(_points_equal(normalized_points, local) for local in local_geometries):
            # restore_gt_for_annotation will perform the canonical ID rebind.
            continue
        if any(_geometries_overlap(normalized_points, local) for local in local_geometries):
            # Identical image bytes can have several filename aliases and plate
            # IDs, with slightly different manual corrections in each run.
            # Keep the current polygon/GT; do not append its old pack geometry.
            report["skipped_overlapping"].append(plate_id)
            continue
        try:
            pixel_points = denormalize_polygon(
                normalized_points,
                image_width=width,
                image_height=height,
            )
        except Exception as exc:
            report["conflicts"].append({
                "plate_id": plate_id,
                "reason": "hydrate_denormalize_failed",
                "error": str(exc),
            })
            continue
        xs = [float(point[0]) for point in pixel_points]
        ys = [float(point[1]) for point in pixel_points]
        attributes = {PLATE_ANNOTATION_ID_ATTR: plate_id}

        gt_state = _resolve_same_plate_across_packs(mounted, image_id=image_id, plate_id=plate_id)
        if gt_state.get("conflict"):
            report["conflicts"].append({"plate_id": plate_id, "reason": gt_state.get("reason")})
        elif gt_state.get("found") and gt_state.get("kind") == "set":
            set_plate_ground_truth(
                attributes,
                gt_state.get("text"),
                source=str(gt_state.get("source") or GROUND_TRUTH_SOURCE_MANUAL_Z2),
            )

        layout_state = _resolve_same_plate_layout_across_packs(
            mounted, image_id=image_id, plate_id=plate_id
        )
        if layout_state.get("conflict"):
            report["conflicts"].append({"plate_id": plate_id, "reason": layout_state.get("reason")})
        elif layout_state.get("found") and layout_state.get("kind") == "set":
            set_plate_layout_gt(attributes, layout_state.get("layout"))

        det = Detection(
            label="plate",
            confidence=1.0,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            polygon=[(float(x), float(y)) for x, y in pixel_points],
            attributes=attributes,
        )
        detections.append(det)
        local_plates.append(det)
        local_ids.add(plate_id)
        local_geometries.append(normalized_points)
        report["created"] += 1
        report["changed"] = True

    if report["changed"]:
        ann.detections = detections
    return report


def restore_gt_for_annotation(host, ann):
    report = {
        "enabled": False,
        "changed": False,
        "restored": 0,
        "restored_layouts": 0,
        "rebound_plate_ids": 0,
        "conflicts": [],
        "image_id": "",
    }
    mounted = _open_mounted_packs(host)
    if not mounted:
        return report
    report["enabled"] = True

    image_path = _resolve_annotation_image_path(host, ann)
    if image_path is None:
        report["reason"] = "missing_image"
        return report

    try:
        fingerprint = fingerprint_preview_image(host, image_path)
    except Exception as exc:
        report["reason"] = "fingerprint_failed"
        report["error"] = str(exc)
        return report

    image_id = str(fingerprint.get("image_id") or "")
    report["image_id"] = image_id
    width = int(fingerprint.get("width", 0) or 0)
    height = int(fingerprint.get("height", 0) or 0)

    try:
        plates = list(host._get_plate_detections(ann) or [])
    except Exception:
        plates = []

    for index, det in enumerate(plates):
        attributes = dict(getattr(det, "attributes", {}) or {})
        local_gt = get_plate_ground_truth(attributes)
        local_plate_id = str(
            attributes.get(PLATE_ANNOTATION_ID_ATTR) or ""
        ).strip()

        resolved = (
            _resolve_same_plate_across_packs(
                mounted,
                image_id=image_id,
                plate_id=local_plate_id,
            )
            if local_plate_id
            else {"found": False, "conflict": False}
        )

        if resolved.get("conflict"):
            report["conflicts"].append({
                "plate_index": index,
                "plate_id": local_plate_id,
                "reason": resolved.get("reason"),
            })
            continue

        if not resolved.get("found"):
            normalized_points = _normalized_detection_polygon(
                host,
                det,
                width=width,
                height=height,
            )
            candidates = (
                _find_exact_geometry_candidates(
                    mounted,
                    image_id=image_id,
                    normalized_points=normalized_points,
                )
                if normalized_points
                else []
            )
            if len(candidates) > 1:
                report["conflicts"].append({
                    "plate_index": index,
                    "plate_id": local_plate_id,
                    "reason": "multiple_exact_geometry_candidates",
                    "candidates": [item["plate_id"] for item in candidates],
                })
                continue
            if len(candidates) == 1:
                candidate_id = candidates[0]["plate_id"]
                resolved = _resolve_same_plate_across_packs(
                    mounted,
                    image_id=image_id,
                    plate_id=candidate_id,
                )
                if resolved.get("conflict"):
                    report["conflicts"].append({
                        "plate_index": index,
                        "plate_id": candidate_id,
                        "reason": resolved.get("reason"),
                    })
                    continue
                if resolved.get("found"):
                    if local_plate_id and local_plate_id != candidate_id and local_gt:
                        report["conflicts"].append({
                            "plate_index": index,
                            "plate_id": local_plate_id,
                            "candidate_plate_id": candidate_id,
                            "reason": "local_gt_blocks_rebind",
                        })
                        continue
                    attributes[PLATE_ANNOTATION_ID_ATTR] = candidate_id
                    local_plate_id = candidate_id
                    report["rebound_plate_ids"] += 1
                    report["changed"] = True

        if not resolved.get("found"):
            det.attributes = attributes
            continue

        kind = str(resolved.get("kind") or "")
        if kind == "set":
            pack_gt = normalize_plate_ground_truth_text(resolved.get("text"))
            if local_gt and local_gt != pack_gt:
                report["conflicts"].append({
                    "plate_index": index,
                    "plate_id": local_plate_id,
                    "reason": "local_gt_differs_from_pack",
                    "local_gt": local_gt,
                    "pack_gt": pack_gt,
                })
                det.attributes = attributes
                continue
            if not local_gt and pack_gt:
                set_plate_ground_truth(
                    attributes,
                    pack_gt,
                    source=str(
                        resolved.get("source")
                        or GROUND_TRUTH_SOURCE_MANUAL_Z2
                    ),
                )
                report["restored"] += 1
                report["changed"] = True
        elif kind == "clear" and local_gt:
            report["conflicts"].append({
                "plate_index": index,
                "plate_id": local_plate_id,
                "reason": "pack_clear_vs_local_gt",
                "local_gt": local_gt,
            })

        if local_plate_id:
            layout_resolved = _resolve_same_plate_layout_across_packs(
                mounted,
                image_id=image_id,
                plate_id=local_plate_id,
            )
            if layout_resolved.get("conflict"):
                report["conflicts"].append({
                    "plate_index": index,
                    "plate_id": local_plate_id,
                    "reason": layout_resolved.get("reason"),
                })
            elif layout_resolved.get("found") and layout_resolved.get("kind") == "set":
                pack_layout = normalize_plate_layout_gt(
                    layout_resolved.get("layout"), default=""
                )
                local_layout_explicit = has_explicit_plate_layout_gt(attributes)
                local_layout = (
                    get_plate_layout_gt(attributes, default="")
                    if local_layout_explicit
                    else ""
                )
                if local_layout_explicit and local_layout != pack_layout:
                    report["conflicts"].append({
                        "plate_index": index,
                        "plate_id": local_plate_id,
                        "reason": "local_layout_differs_from_pack",
                        "local_layout": local_layout,
                        "pack_layout": pack_layout,
                    })
                elif not local_layout_explicit and pack_layout:
                    set_plate_layout_gt(attributes, pack_layout)
                    report["restored_layouts"] += 1
                    report["changed"] = True

        det.attributes = attributes

    return report


def _outbox_path(host):
    try:
        xml_path = host._get_current_annotation_xml_path()
    except Exception:
        xml_path = None
    if xml_path is not None:
        return Path(xml_path).parent / "gt_sync_pending.json"

    run_dir = _safe_path(getattr(host, "current_annotation_run_dir", None))
    return (run_dir / "gt_sync_pending.json") if run_dir is not None else None


def _load_outbox(host):
    path = _outbox_path(host)
    if path is None or not path.exists():
        return {"schema": OUTBOX_SCHEMA, "items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    items = payload.get("items") if isinstance(payload, dict) else {}
    return {
        "schema": OUTBOX_SCHEMA,
        "items": dict(items or {}) if isinstance(items, dict) else {},
    }


def _write_outbox(host, payload):
    path = _outbox_path(host)
    if path is None:
        return
    items = dict(payload.get("items", {}) or {})
    if not items:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        invalidate_gt_pack_status_cache(host)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(
            {"schema": OUTBOX_SCHEMA, "items": items},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)
    invalidate_gt_pack_status_cache(host)


def _outbox_key(item):
    return "|".join((
        str(item.get("plate_id") or ""),
        str(item.get("field") or "ground_truth"),
        str(item.get("operation") or ""),
    ))


def _queue_outbox_item(host, item):
    payload = _load_outbox(host)
    payload["items"][_outbox_key(item)] = dict(item)
    _write_outbox(host, payload)


def _sync_item_to_pack(item):
    working_path = _safe_path(item.get("working_pack_path"))
    image_path = _safe_path(item.get("image_path"))
    plate_id = str(item.get("plate_id") or "").strip()
    polygon = list(item.get("polygon", []) or [])
    operation = str(item.get("operation") or "").strip().lower()
    field = str(item.get("field") or "ground_truth").strip().lower()
    text = normalize_plate_ground_truth_text(item.get("text"))
    layout = normalize_plate_layout_gt(item.get("layout"), default="")
    source = str(
        item.get("source") or GROUND_TRUTH_SOURCE_MANUAL_Z2
    ).strip()

    if working_path is None:
        return {"ok": False, "reason": "no_working_pack"}
    if image_path is None or not image_path.exists():
        return {"ok": False, "reason": "missing_image"}
    if not plate_id or len(polygon) < 4:
        return {"ok": False, "reason": "incomplete_record"}

    pack = (
        ALPRGTPack.open(working_path)
        if (working_path / "manifest.json").is_file()
        else ALPRGTPack.create(working_path, producer=PRODUCER)
    )

    image = pack.add_image(
        image_path,
        alias=str(item.get("image_name") or image_path.name),
        producer=PRODUCER,
    )
    pack.ensure_plate(
        image_id=image["image_id"],
        polygon=polygon,
        plate_id=plate_id,
    )

    if field == "layout":
        if operation == "clear":
            pack.clear_plate_layout_gt(
                plate_id,
                source=source,
                producer=PRODUCER,
            )
        elif operation == "set" and layout:
            pack.set_plate_layout_gt(
                plate_id,
                layout,
                source=source,
                producer=PRODUCER,
            )
        else:
            return {"ok": False, "reason": "invalid_layout_operation"}
    elif field == "ground_truth":
        if operation == "clear":
            pack.clear_ground_truth(
                plate_id,
                source=source,
                producer=PRODUCER,
            )
        elif operation == "set" and text:
            pack.set_ground_truth(
                plate_id,
                text,
                source=source,
                producer=PRODUCER,
            )
        else:
            return {"ok": False, "reason": "invalid_operation"}
        # A text edit in Z2 materializes the visible layout GT as well.
        if layout:
            pack.set_plate_layout_gt(
                plate_id,
                layout,
                source=source,
                producer=PRODUCER,
            )
    else:
        return {"ok": False, "reason": "invalid_field"}

    return {
        "ok": True,
        "plate_id": plate_id,
        "image_id": image["image_id"],
        "field": field,
        "operation": operation,
    }


def retry_pending_gt_sync(host):
    payload = _load_outbox(host)
    items = dict(payload.get("items", {}) or {})
    if not items:
        return {"ok": True, "retried": 0, "remaining": 0}

    remaining = {}
    for key, item in items.items():
        try:
            result = _sync_item_to_pack(item)
        except Exception:
            result = {"ok": False}
        if not result.get("ok"):
            remaining[key] = item

    payload["items"] = remaining
    _write_outbox(host, payload)
    return {
        "ok": not remaining,
        "retried": len(items),
        "remaining": len(remaining),
    }


def prepare_plate_gt_sync_item(host, ann, det, requested_text):
    """Capture paths and values on the UI thread; workers receive plain data."""
    working_path = get_working_gt_pack_path(host, create_parent=True)
    if working_path is None:
        return None
    image_path = _resolve_annotation_image_path(host, ann)
    if image_path is None:
        raise ValueError("Brak obrazu do zapisania GT Pack")
    attributes = ensure_plate_detection_contract(det)
    polygon = _detection_polygon(host, det)
    normalized = normalize_plate_ground_truth_text(requested_text)
    return {
        "working_pack_path": str(working_path), "image_path": str(image_path),
        "image_name": str(getattr(ann, "filename", "") or image_path.name),
        "plate_id": str(attributes.get(PLATE_ANNOTATION_ID_ATTR) or ""),
        "polygon": [[float(x), float(y)] for x, y in polygon[:4]],
        "field": "ground_truth", "operation": "set" if normalized else "clear",
        "text": normalized, "layout": get_plate_layout_gt(attributes, default=""),
        "source": str(attributes.get("ground_truth_source") or GROUND_TRUTH_SOURCE_MANUAL_Z2),
    }


def sync_gt_items_after_xml_save(xml_path, incoming_items):
    """Worker-safe sync with a durable outbox and newest-value precedence."""
    from types import SimpleNamespace

    # This detached adapter has no Tk objects or access to a changing UI run.
    adapter = SimpleNamespace(_get_current_annotation_xml_path=lambda: Path(xml_path))
    payload = _load_outbox(adapter)
    items = dict(payload.get("items", {}) or {})
    for item in incoming_items:
        identity = (item.get("plate_id"), item.get("field", "ground_truth"))
        items = {key: old for key, old in items.items()
                 if (old.get("plate_id"), old.get("field", "ground_truth")) != identity}
        items[_outbox_key(item)] = dict(item)
    _write_outbox(adapter, {"items": items})
    for key, item in list(items.items()):
        try:
            result = _sync_item_to_pack(item)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        if result.get("ok"):
            items.pop(key, None)
        else:
            item["last_error"] = str(result.get("error") or result.get("reason") or "unknown")
    _write_outbox(adapter, {"items": items})
    return {"ok": not items, "queued": len(items)}


def sync_plate_gt_after_xml_save(
    host,
    ann,
    det,
    requested_text,
    *,
    retry_pending: bool = True,
):
    if bool(retry_pending):
        try:
            retry_pending_gt_sync(host)
        except Exception:
            pass

    working_path = get_working_gt_pack_path(host, create_parent=True)
    if working_path is None:
        return {
            "ok": True,
            "enabled": False,
            "reason": "no_working_pack",
        }

    image_path = _resolve_annotation_image_path(host, ann)
    if image_path is None:
        return {
            "ok": False,
            "enabled": True,
            "reason": "missing_image",
        }

    attributes = ensure_plate_detection_contract(det)
    plate_id = str(
        attributes.get(PLATE_ANNOTATION_ID_ATTR) or ""
    ).strip()
    polygon = _detection_polygon(host, det)
    normalized = normalize_plate_ground_truth_text(requested_text)
    operation = "set" if normalized else "clear"

    item = {
        "working_pack_path": str(working_path),
        "image_path": str(image_path),
        "image_name": str(getattr(ann, "filename", "") or image_path.name),
        "plate_id": plate_id,
        "polygon": [[float(x), float(y)] for x, y in polygon[:4]],
        "field": "ground_truth",
        "operation": operation,
        "text": normalized,
        "layout": get_plate_layout_gt(attributes, default=""),
        "source": str(
            attributes.get("ground_truth_source")
            or GROUND_TRUTH_SOURCE_MANUAL_Z2
        ).strip(),
    }

    try:
        result = _sync_item_to_pack(item)
    except Exception as exc:
        result = {
            "ok": False,
            "reason": "exception",
            "error": str(exc),
        }

    if result.get("ok"):
        payload = _load_outbox(host)
        items = dict(payload.get("items", {}) or {})
        for key, pending in list(items.items()):
            if (
                str((pending or {}).get("plate_id") or "").strip() == plate_id
                and str((pending or {}).get("field") or "ground_truth").strip().lower() == "ground_truth"
            ):
                items.pop(key, None)
        payload["items"] = items
        _write_outbox(host, payload)
        result["enabled"] = True
        result["working_pack_path"] = str(working_path)
        try:
            refresher = getattr(host, "_refresh_gt_completion_ui", None)
            if callable(refresher):
                refresher()
        except Exception:
            pass
        return result

    item["last_error"] = str(
        result.get("error")
        or result.get("reason")
        or "unknown"
    )
    _queue_outbox_item(host, item)
    return {
        **result,
        "enabled": True,
        "queued": True,
        "working_pack_path": str(working_path),
    }


def sync_plate_layout_after_xml_save(
    host,
    ann,
    det,
    requested_layout,
    *,
    retry_pending: bool = True,
):
    if bool(retry_pending):
        try:
            retry_pending_gt_sync(host)
        except Exception:
            pass

    working_path = get_working_gt_pack_path(host, create_parent=True)
    if working_path is None:
        return {"ok": True, "enabled": False, "reason": "no_working_pack"}

    image_path = _resolve_annotation_image_path(host, ann)
    if image_path is None:
        return {"ok": False, "enabled": True, "reason": "missing_image"}

    attributes = ensure_plate_detection_contract(det)
    plate_id = str(attributes.get(PLATE_ANNOTATION_ID_ATTR) or "").strip()
    polygon = _detection_polygon(host, det)
    layout = normalize_plate_layout_gt(requested_layout, default="")
    operation = "set" if layout else "clear"
    item = {
        "working_pack_path": str(working_path),
        "image_path": str(image_path),
        "image_name": str(getattr(ann, "filename", "") or image_path.name),
        "plate_id": plate_id,
        "polygon": [[float(x), float(y)] for x, y in polygon[:4]],
        "field": "layout",
        "operation": operation,
        "layout": layout,
        "source": GROUND_TRUTH_SOURCE_MANUAL_Z2,
    }

    try:
        result = _sync_item_to_pack(item)
    except Exception as exc:
        result = {"ok": False, "reason": "exception", "error": str(exc)}

    if result.get("ok"):
        payload = _load_outbox(host)
        items = dict(payload.get("items", {}) or {})
        for key, pending in list(items.items()):
            if (
                str((pending or {}).get("plate_id") or "").strip() == plate_id
                and str((pending or {}).get("field") or "ground_truth").strip().lower() == "layout"
            ):
                items.pop(key, None)
        payload["items"] = items
        _write_outbox(host, payload)
        result["enabled"] = True
        result["working_pack_path"] = str(working_path)
        try:
            refresher = getattr(host, "_refresh_gt_completion_ui", None)
            if callable(refresher):
                refresher()
        except Exception:
            pass
        return result

    item["last_error"] = str(result.get("error") or result.get("reason") or "unknown")
    _queue_outbox_item(host, item)
    return {
        **result,
        "enabled": True,
        "queued": True,
        "working_pack_path": str(working_path),
    }
