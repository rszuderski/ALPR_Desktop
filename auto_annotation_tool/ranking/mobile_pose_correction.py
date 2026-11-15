"""Z4MOBREV026: source-pinned, prediction-assisted MT polygon corrections.

This is not independent GT. No source prediction or historical Pose GT is changed.
The API uses MT-input coordinates, never GUI Canvas pixels.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math

from .mobile_pose_metrics import compare_mt_pose_to_gt, validate_mt_quad

CORRECTION_SCHEMA = "alpr.mobile.pose_assisted_correction.v1"
CORRECTION_METHOD = "assisted_correction_from_mt_v1"


def _pointlist(quad, size):
    points = validate_mt_quad(quad, float(size[0]), float(size[1]))
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]


def _digest(points):
    data = json.dumps([[float(x), float(y)] for x, y in points],
                      separators=(",", ":"), ensure_ascii=True)
    return sha256(data.encode("ascii")).hexdigest()


def correction_for_attempt(annotation, *, attempt_id, source_sha, image_entry, size, prediction):
    """Return trusted persisted correction or None; reject mismatched provenance."""
    stored = (annotation or {}).get("pose_mt_correction_v1")
    if stored in (None, ""):
        return None
    if not isinstance(stored, dict) or stored.get("schema") != CORRECTION_SCHEMA:
        raise ValueError("Nieprawidłowy schemat korekty Pose MT")
    if (str(stored.get("attempt_id")) != str(attempt_id) or
            str(stored.get("source_archive_sha256")) != str(source_sha) or
            str(stored.get("mt_input_evidence_entry")) != str(image_entry) or
            list(stored.get("input_size_px") or []) != list(map(float, size))):
        raise ValueError("Korekta Pose dotyczy innej próby, obrazu albo archiwum")
    model_points = _pointlist(prediction, size)
    if stored.get("model_prediction_sha256") != _digest(model_points):
        raise ValueError("Korekta Pose nie odpowiada oryginalnym narożnikom MT")
    points = _pointlist(stored.get("operator_quad_mt_input_px"), size)
    if not moved(model_points, points):
        raise ValueError("Korekta nie zawiera rzeczywistej zmiany obrysu")
    if stored.get("method") != CORRECTION_METHOD:
        raise ValueError("Nieznane pochodzenie korekty Pose")
    return points


def moved(a, b, *, tolerance=0.0005):
    return any(abs(float(p) - float(q)) > tolerance
               for point, other in zip(a, b) for p, q in zip(point, other))


def make_correction(*, attempt_id, source_sha, image_entry, size, prediction, corrected):
    model_points = _pointlist(prediction, size)
    points = _pointlist(corrected, size)
    if not moved(model_points, points):
        raise ValueError("Najpierw zmień położenie przynajmniej jednego narożnika")
    return {
        "schema": CORRECTION_SCHEMA,
        "method": CORRECTION_METHOD,
        "attempt_id": str(attempt_id),
        "source_archive_sha256": str(source_sha),
        "mt_input_evidence_entry": str(image_entry),
        "input_size_px": list(map(float, size)),
        "model_prediction_sha256": _digest(model_points),
        "operator_quad_mt_input_px": points,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def inside_polygon(point, points):
    """Convex polygon hit-test in any orientation (Canvas coordinates)."""
    signs = []
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        cross = (x2-x1)*(point[1]-y1) - (y2-y1)*(point[0]-x1)
        if abs(cross) > 1e-8:
            signs.append(cross > 0)
    return bool(signs) and all(s == signs[0] for s in signs)


def hit_target(cursor, corners_canvas, *, radius=15):
    if len(corners_canvas) != 4:
        return None
    distances = [math.dist(cursor, corner) for corner in corners_canvas]
    near = min(range(4), key=distances.__getitem__)
    if distances[near] <= radius:
        return ("corner", near)
    return ("body", None) if inside_polygon(cursor, corners_canvas) else None


def move_polygon(points, target, start, end, size):
    """Move one vertex or entire quad; fail closed on invalid geometry."""
    if target is None or len(points) != 4:
        raise ValueError("Brak aktywnej geometrii do korekty")
    dx, dy = float(end[0])-float(start[0]), float(end[1])-float(start[1])
    if not all(math.isfinite(v) for v in (dx,dy)):
        raise ValueError("Nieprawidłowe współrzędne myszy")
    out = [list(map(float, p)) for p in points]
    kind, index = target
    if kind == "corner" and isinstance(index,int) and 0 <= index < 4:
        out[index][0] += dx
        out[index][1] += dy
    elif kind == "body":
        out = [[x+dx, y+dy] for x,y in out]
    else:
        raise ValueError("Nieznany uchwyt korekty")
    return _pointlist(out, size)


def correction_report(session):
    """Separate assisted-delta results from independently annotated Pose GT."""
    count, values, invalid = 0, [], 0
    for aid, attempt in getattr(session,"attempts",{}).items():
        decision = getattr(session.review,"attempt_annotations",{}).get(aid,{})
        if not decision.get("pose_mt_correction_v1"):
            continue
        # Import lazily: the GUI knows the provenance transform of the mobile MT quad.
        try:
            from ..gui.z4_mobile_sample_review import _linked_pose_quad_on_mt_input
            if getattr(session,"cancelled",lambda row:False)(attempt):
                continue
            group = session.mt_invocations.get(session.invocation_by_attempt.get(aid))
            if not group or not group.executed or group.cancelled or group.execution_failed:
                continue
            pred = _linked_pose_quad_on_mt_input(session, attempt)
            if pred is None:
                raise ValueError("Brak źródłowego Pose")
            size=(float(attempt["input_width"]),float(attempt["input_height"]))
            points=correction_for_attempt(decision,attempt_id=aid,
                      source_sha=session.review.source_archive_sha256,
                      image_entry=attempt["mt_input_evidence_entry"],size=size,prediction=pred)
            if points is None: continue
            # Old corrections kept for audit when the active decision changes.
            if decision.get("pose_quality") != "imprecise" or decision.get("is_plate") is not True:
                continue
            metrics=compare_mt_pose_to_gt(pred,points,*size)
            values.append(metrics)
            count+=1
        except (KeyError, TypeError, ValueError):
            invalid+=1
    return {"assisted_correction_n":count,
            "assisted_correction_invalid_n":invalid,
            "assisted_delta_iou_mean_pct": (sum(v["polygon_iou_pct"] for v in values)/count if count else None),
            "assisted_delta_nmce_mean_pct": (sum(v["nmce_pct"] for v in values)/count if count else None),
            "assisted_measurement_note": "Korekta wspomagana predykcją MT; nie jest niezależnym GT do IoU/NMCE."}
