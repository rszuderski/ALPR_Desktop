"""Ground-truth readiness helpers for the Z2 -> character-model path."""

from __future__ import annotations

from typing import Any
import os
import math

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG
from ..plate_ground_truth import normalize_plate_ground_truth_text


def gt_required_for_current_route(host, *, campaign=None) -> bool:
    """Z2 approves geometry. Text may be authored later in Z3/PZ2."""
    return False


def plate_geometry_valid(plate) -> bool:
    polygon = plate.get("polygon") if isinstance(plate, dict) else getattr(plate, "polygon", None)
    try:
        if polygon:
            points = [(float(point[0]), float(point[1])) for point in polygon]
            if len(points) < 4 or not all(math.isfinite(v) for point in points for v in point):
                return False
            area = sum(x1*y2 - x2*y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))
            return abs(area) > 0
        bbox = plate.get("bbox") if isinstance(plate, dict) else getattr(plate, "bbox", None)
        x1, y1, x2, y2 = map(float, bbox)
        return all(map(math.isfinite, (x1, y1, x2, y2))) and x2 > x1 and y2 > y1
    except (TypeError, ValueError, IndexError):
        return False


def annotation_gt_readiness(ann) -> dict:
    """Read only explicit per-polygon GT; no filenames, packs or image I/O."""
    plates = [
        det for det in getattr(ann, "detections", []) or []
        if str(getattr(det, "label", "") or "").strip().lower() in CONFIG.PLATE_LABELS
    ]
    missing = [
        index for index, det in enumerate(plates)
        if not normalize_plate_ground_truth_text(
            (getattr(det, "attributes", None) or {}).get("ground_truth_text")
        )
    ]
    return {"total": len(plates), "filled": len(plates) - len(missing),
            "missing": len(missing), "missing_indices": missing}


def missing_required_gt(host, ann) -> int:
    if not gt_required_for_current_route(host):
        return 0
    return annotation_gt_readiness(ann)["missing"]


def approval_block_reason(host, ann) -> str:
    state = annotation_gt_readiness(ann)
    if not state["total"]:
        return "Najpierw dodaj ramkę tablicy."
    if any(not plate_geometry_valid(det) for det in getattr(ann, "detections", []) or []
           if str(getattr(det, "label", "") or "").strip().lower() in CONFIG.PLATE_LABELS):
        return "Popraw geometrię ramek tablic przed zatwierdzeniem."
    return ""


def _entry_identity(entry: dict, fallback_index: int) -> str:
    key = str(entry.get("entry_key") or "").strip()
    if key:
        return f"key:{key}"

    image_name = str(entry.get("image_name") or "").strip().lower()
    source_path = str(entry.get("source_image_path") or "").strip().lower()
    if image_name or source_path:
        return f"image:{image_name}|{source_path}"
    return f"fallback:{fallback_index}"


def _plate_entry_ground_truth(plate_entry: dict) -> str:
    if not isinstance(plate_entry, dict):
        return ""

    direct = normalize_plate_ground_truth_text(
        plate_entry.get("ground_truth_text")
    )
    if direct:
        return direct

    attrs = plate_entry.get("attributes")
    if isinstance(attrs, dict):
        return normalize_plate_ground_truth_text(
            attrs.get("ground_truth_text")
        )
    return ""


def summarize_char_gt_entries(
    project_entries,
    run_entries=None,
) -> dict[str, Any]:
    """Summarize GT completeness.

    Entries from ``run_entries`` override project entries with the same
    identity so a correction in the current Z2 run wins over older project
    metadata.
    """
    merged: dict[str, dict] = {}

    for group in (list(project_entries or []), list(run_entries or [])):
        for index, raw_entry in enumerate(group):
            if not isinstance(raw_entry, dict):
                continue
            merged[_entry_identity(raw_entry, index)] = raw_entry

    total_plates = 0
    geometry_plates = 0
    gt_plates = 0
    missing_images: list[str] = []

    for entry in merged.values():
        image_name = str(entry.get("image_name") or "").strip()
        entry_missing = 0

        for plate_entry in list(entry.get("plates") or []):
            if not isinstance(plate_entry, dict):
                continue
            polygon = list(plate_entry.get("polygon") or [])
            if polygon and len(polygon) < 4:
                continue

            total_plates += 1
            geometry_plates += int(plate_geometry_valid(plate_entry))
            if _plate_entry_ground_truth(plate_entry):
                gt_plates += 1
            else:
                entry_missing += 1

        if entry_missing > 0 and image_name:
            missing_images.append(image_name)

    missing_gt = max(0, int(total_plates) - int(gt_plates))
    return {
        "total_plates": int(total_plates),
        "gt_plates": int(gt_plates),
        "missing_gt": int(missing_gt),
        "geometry_ready_plates": geometry_plates,
        "plate_geometry_count": geometry_plates,
        "plate_gt_present_count": gt_plates,
        "plate_gt_missing_count": missing_gt,
        "geometry_ready": bool(geometry_plates > 0 and geometry_plates == total_plates),
        "gt_complete": bool(total_plates > 0 and missing_gt == 0),
        "ready": bool(geometry_plates > 0 and geometry_plates == total_plates),
        "missing_images": list(dict.fromkeys(missing_images)),
    }


def _path_identity(value):
    return os.path.normcase(os.path.abspath(str(value))) if value else ""


def _live_char_gt_entries(host, run_dir, project_entries):
    """GT is annotation metadata: do not resolve/stat every image to count it."""
    current_run = getattr(host, "current_annotation_run_dir", None)
    annotations = getattr(host, "current_annotations", None)
    if not current_run or not run_dir or not annotations:
        return None
    run_key = _path_identity(run_dir)
    if _path_identity(current_run) != run_key:
        return None
    approved = {str(name).strip().lower() for name in host._get_preview_approved_filenames()}
    image_map = getattr(host, "_preview_image_path_map", {}) or {}
    by_name = {}
    for entry in project_entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("image_name") or "").strip().lower()
        by_name.setdefault(name, []).append(entry)
    result = []
    for ann in annotations:
        name = str(getattr(ann, "filename", "") or "").strip()
        if name.lower() not in approved:
            continue
        mapped_path = image_map.get(name)
        mapped_key = _path_identity(mapped_path)
        matches = [entry for entry in by_name.get(name.lower(), [])
                   if _path_identity(entry.get("approved_from_run")) == run_key
                   or (mapped_key and _path_identity(entry.get("source_image_path")) == mapped_key)]
        identity = matches[0] if matches else {
            "entry_key": mapped_key or f"run:{run_key}|{name.lower()}",
            "source_image_path": str(mapped_path or ""),
        }
        result.append({**identity, "image_name": name, "plates": [
            {"attributes": dict(getattr(det, "attributes", {}) or {}),
             "polygon": getattr(det, "polygon", None), "bbox": getattr(det, "bbox", None)}
            for det in host._get_plate_detections(ann)
        ]})
    return result


def build_char_gt_readiness(
    host,
    *,
    run_dir=None,
    force_parse_xml: bool = False,
) -> dict[str, Any]:
    """Build GT readiness for the effective character-source pool."""
    try:
        project_entries = list(
            CAMPAIGN.list_plate_approved_entries() or []
        )
    except Exception:
        project_entries = []

    if not force_parse_xml:
        live_entries = _live_char_gt_entries(host, run_dir, project_entries)
        if live_entries is not None:
            return summarize_char_gt_entries(project_entries, live_entries)

    run_entries = []
    if run_dir is not None:
        builder = getattr(
            host,
            "_build_campaign_plate_approved_entries_from_run",
            None,
        )
        if callable(builder):
            try:
                run_entries = list(
                    builder(
                        run_dir,
                        force_parse_xml=bool(force_parse_xml),
                    )
                    or []
                )
            except TypeError:
                try:
                    run_entries = list(builder(run_dir) or [])
                except Exception:
                    run_entries = []
            except Exception:
                run_entries = []

    return summarize_char_gt_entries(project_entries, run_entries)
