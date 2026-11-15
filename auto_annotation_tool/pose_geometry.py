"""Working Z2 geometry and immutable Pose evidence, persisted inside CVAT XML.

This module never reads or rewrites frozen GT. A display cycle is not a claim
about the learned meaning of keypoint indices. Legacy XML has UNKNOWN raw Pose.
"""
from __future__ import annotations

import copy
import json
import math

from .rectification.quad_geometry import analyze_quad, canonicalize_polygon

GEOMETRY_SCHEMA = "alpr.pose_geometry.v1"
GEOMETRY_XML_ATTR = "alpr_quad_geometry"


def _xy(points):
    try:
        return [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError):
        return []


def _bbox_polygon(bbox):
    try:
        x1, y1, x2, y2 = map(float, bbox)
        if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    except (ValueError, TypeError):
        pass
    return None


def _display_evidence(points, *, image_size=None):
    result = canonicalize_polygon(points, anchor_index=0, image_size=image_size)
    return {
        "geometry_validity": result.input_analysis.to_dict(),
        "display_validity": result.output_analysis.to_dict() if result.output_analysis else None,
        "display_polygon": copy.deepcopy(result.polygon),
        "reorder_permutation": list(result.permutation) if result.permutation is not None else None,
        "reordered": bool(result.reordered),
        "corner_identity_status": result.corner_identity_status,
        "pose_corner_identity": "UNKNOWN",
    }


def attach_pose_geometry(det, raw_keypoints, *, image_size=None, offset=(0, 0)):
    """Attach exact [N,2]/[N,3] Pose evidence; translate XY only for ROI.

    The legacy keypoints field may contain synthesized confidence for [4,2].
    It is deliberately not the authority; this evidence retains original shape.
    """
    raw = None
    if raw_keypoints is not None:
        raw = [[float(value) for value in point] for point in raw_keypoints]
    scene = copy.deepcopy(raw)
    if scene is not None:
        for point in scene:
            if len(point) >= 2:
                point[0] += float(offset[0])
                point[1] += float(offset[1])
    geometry = {
        "schema": GEOMETRY_SCHEMA,
        "raw_pose_keypoints": scene,
        "raw_pose_roi_keypoints": raw if tuple(offset) != (0, 0) else None,
        "roi_offset": list(offset),
        "source_bbox": list(det.bbox),
        "source": "raw_pose" if raw is not None else "bbox_fallback",
        "manual_edit": False,
        "bbox_fallback": False,
        "raw_pose_available": raw is not None,
    }
    geometry.update(_display_evidence(_xy(scene or []), image_size=image_size))
    if geometry["display_polygon"] is None:
        geometry["display_polygon"] = _bbox_polygon(det.bbox)
        geometry["bbox_fallback"] = True
        geometry["source"] = "bbox_fallback"
        geometry["display_validity"] = analyze_quad(geometry["display_polygon"] or [], image_size=image_size).to_dict()
    geometry["stored_polygon_points"] = copy.deepcopy(geometry["display_polygon"])
    det.polygon = copy.deepcopy(geometry["display_polygon"])
    det.geometry = geometry
    return geometry


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_restore(value):
    if isinstance(value, dict):
        if set(value) == {"__nonfinite_float__"}:
            return float(value["__nonfinite_float__"])
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


def geometry_xml_payload(det):
    """Return metadata without mutating live attributes during background save."""
    geometry = getattr(det, "geometry", None)
    if not isinstance(geometry, dict):
        return None
    evidence = copy.deepcopy(geometry)
    if not geometry_matches_stored_polygon(det):
        current = _display_evidence(det.polygon or [])
        evidence["previous_stored_polygon_points"] = evidence.get("stored_polygon_points")
        evidence["working_geometry_validity"] = current["geometry_validity"]
        evidence["display_polygon"] = current["display_polygon"]
        evidence["display_validity"] = current["display_validity"]
        evidence["corner_identity_status"] = current["corner_identity_status"]
        evidence["source"] = "untracked_geometry_edit"
        evidence["geometry_edit_untracked"] = True
        evidence.pop("stored_polygon_xml", None)
    evidence["stored_polygon_points"] = copy.deepcopy(det.polygon)
    return json.dumps(_json_safe(evidence), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def geometry_matches_stored_polygon(det):
    geometry = getattr(det, "geometry", None) or {}
    return (_json_safe(_xy(det.polygon or [])) ==
            _json_safe(_xy(geometry.get("stored_polygon_points") or [])))


def hydrate_stored_geometry(det, *, image_size=None, points_text=None):
    """Load provenance while leaving source coordinates and order unchanged."""
    stored = copy.deepcopy(det.polygon or [])
    encoded = (getattr(det, "attributes", None) or {}).get(GEOMETRY_XML_ATTR)
    geometry = None
    if encoded:
        try:
            candidate = _json_restore(json.loads(encoded))
            if isinstance(candidate, dict) and candidate.get("schema") == GEOMETRY_SCHEMA:
                geometry = candidate
        except (ValueError, TypeError):
            pass
    if geometry is None:
        geometry = {
            "schema": GEOMETRY_SCHEMA, "raw_pose_keypoints": None,
            "raw_pose_available": False, "source_bbox": None,
            "source": "legacy_xml", "manual_edit": False, "bbox_fallback": False,
            "stored_polygon_points": stored, "raw_pose_status": "UNKNOWN",
        }
        geometry.update(_display_evidence(stored, image_size=image_size))
        if encoded:
            geometry["provenance_error"] = "INVALID_OR_UNSUPPORTED_METADATA"
    elif _xy(geometry.get("stored_polygon_points") or []) != _xy(stored):
        # Another editor changed the XML: preserve the original Pose evidence,
        # but discard stale display/permutation claims.
        geometry.update(_display_evidence(stored, image_size=image_size))
        geometry["external_polygon_edit"] = True
        geometry["source"] = "xml_external"
        geometry["stored_polygon_points"] = stored
    if geometry.get("display_polygon") is not None:
        geometry["display_polygon"] = _xy(geometry["display_polygon"])
    geometry["stored_polygon_xml"] = points_text
    det.geometry = geometry
    if (geometry.get("raw_pose_available") and not geometry.get("manual_edit")
            and not geometry.get("external_polygon_edit") and not geometry.get("geometry_edit_untracked")
            and geometry.get("source_bbox")):
        det.bbox = tuple(geometry["source_bbox"])
    raw = geometry.get("raw_pose_keypoints")
    det.keypoints = ([(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 1.0)
                      for p in raw if len(p) >= 2] if raw is not None else None)
    return geometry


def geometry_display_polygon(det):
    """Read the cached display shape; no sorting/validation in mousemove."""
    geometry = getattr(det, "geometry", None) or {}
    polygon = geometry.get("display_polygon")
    if polygon is None:
        polygon = getattr(det, "polygon", None)
    points = _xy(polygon) if polygon else (_bbox_polygon(det.bbox) or [])
    # Invalid historical coordinates remain in the evidence/XML, but must not
    # reach Tcl canvas coordinates (NaN/Inf raises inside rendering).
    if any(not math.isfinite(value) for point in points for value in point):
        return _bbox_polygon(det.bbox) or []
    return points


def set_manual_display_point(det, polygon):
    """Commit the explicitly moved handle, deferring analysis until release."""
    if not isinstance(getattr(det, "geometry", None), dict):
        hydrate_stored_geometry(det)
    det.polygon = copy.deepcopy(polygon)
    det.geometry["display_polygon"] = copy.deepcopy(polygon)
    det.geometry["manual_edit"] = True
    det.geometry["manual_pending"] = True
    det.geometry["source"] = "manual"
    det.geometry.pop("stored_polygon_xml", None)


def finalize_manual_geometry(det, polygon=None, *, image_size=None, normalize=True):
    """Finalize only a genuinely edited detection, leaving raw Pose immutable."""
    if not isinstance(getattr(det, "geometry", None), dict):
        hydrate_stored_geometry(det, image_size=image_size)
    geometry = det.geometry
    points = _xy(polygon if polygon is not None else geometry_display_polygon(det))
    if normalize:
        evidence = _display_evidence(points, image_size=image_size)
        fixed = evidence["display_polygon"]
        # Invalid manual geometry remains visible for correction; never replace
        # the operator's coordinates with a fake Pose or bbox.
        if fixed is None:
            evidence["display_polygon"] = points
        geometry["manual_reorder_permutation"] = evidence.pop("reorder_permutation")
        geometry["manual_geometry_validity"] = evidence.pop("geometry_validity")
        geometry["manual_reordered"] = evidence.pop("reordered")
        geometry.update(evidence)
    else:
        geometry["display_polygon"] = points
    det.polygon = copy.deepcopy(geometry["display_polygon"])
    if det.polygon:
        xs, ys = zip(*det.polygon)
        det.bbox = (min(xs), min(ys), max(xs), max(ys))
    geometry["source"] = "manual"
    geometry["manual_edit"] = True
    geometry["manual_pending"] = not normalize
    geometry["stored_polygon_points"] = copy.deepcopy(det.polygon)
    geometry.pop("stored_polygon_xml", None)
    det.attributes["manually_edited"] = "true"
    det.attributes["manual_source"] = "preview"
    return det.polygon


def geometry_warning(det):
    geometry = getattr(det, "geometry", None) or {}
    if geometry.get("provenance_error"):
        return "Nieznane metadane geometrii"
    if geometry.get("geometry_edit_untracked"):
        return "Zmieniona geometria — sprawdź narożniki"
    if geometry.get("manual_edit"):
        validity = geometry.get("display_validity") or {}
        if not validity.get("is_geometrically_valid", False):
            return "Sprawdź narożniki"
        return ""
    if geometry.get("bbox_fallback"):
        return "Obrys z bbox — brak poprawnego Pose"
    validity = geometry.get("geometry_validity") or {}
    if geometry.get("reordered"):
        return "Połączenia poprawione; punkty zachowane"
    if validity and not validity.get("is_geometrically_valid", False):
        return "Nieprawidłowa geometria — sprawdź narożniki"
    if geometry.get("corner_identity_status") == "CORNER_IDENTITY_AMBIGUOUS":
        return "Indeksy narożników wymagają odbioru"
    return ""
