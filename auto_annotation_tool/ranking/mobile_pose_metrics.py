"""Auditable MT Pose corner metrics. Source evidence is read-only.

The order of both quads is the Android Pose contract TL, TR, BR, BL.
All coordinates are on the *same* MT input image in pixels (before GUI Zoom/Pan).
NMCE normalizes mean corresponding-corner distance by the diagonal of GT's
axis-aligned envelope, not by an arbitrary photo/display resolution.
"""
from __future__ import annotations

import math

POSE_GT_SCHEMA = "manual_mt_input_quad_v1"


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area(points):
    return 0.5 * sum(points[i][0] * points[(i + 1) % len(points)][1]
                     - points[(i + 1) % len(points)][0] * points[i][1]
                     for i in range(len(points)))


def validate_mt_quad(points, width, height):
    """Reject malformed, off-image, degenerate and crossed four-point GT/prediction.

    Returns a validated tuple of four (x,y) floats without silently reordering.
    """
    try:
        w, h = float(width), float(height)
        if not math.isfinite(w) or not math.isfinite(h) or w <= 0 or h <= 0:
            raise ValueError("Nieprawidłowy rozmiar wejścia MT")
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            raise ValueError("Polygon wymaga czterech punktów")
        quad = tuple((float(p[0]), float(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) == 2)
        if len(quad) != 4 or any(not math.isfinite(v) for xy in quad for v in xy):
            raise ValueError("Współrzędne punktów są nieprawidłowe")
        if any(x < 0 or x > w or y < 0 or y > h for x, y in quad):
            raise ValueError("Narożniki muszą znajdować się na wejściu MT")
        area = abs(_signed_area(quad))
        if area <= max(1e-6, w * h * 1e-8):
            raise ValueError("Polygon ma zbyt małą powierzchnię")
        turns = [_cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
        if not (all(v > 1e-9 for v in turns) or all(v < -1e-9 for v in turns)):
            raise ValueError("Narożniki muszą tworzyć wypukły, nieprzecinający się polygon")
        return quad
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(str(exc) or "Błędne cztery narożniki") from exc


def _clip_convex(subject, clip):
    """Sutherland–Hodgman intersection of two convex quads, either winding."""
    output = list(subject)
    orientation = 1.0 if _signed_area(clip) > 0 else -1.0
    for i in range(4):
        a, b = clip[i], clip[(i + 1) % 4]
        if not output:
            break
        seq, output = output, []
        start = seq[-1]
        start_inside = orientation * _cross(a, b, start) >= -1e-9
        for end in seq:
            end_inside = orientation * _cross(a, b, end) >= -1e-9
            if end_inside != start_inside:
                d_x, d_y = end[0] - start[0], end[1] - start[1]
                c_x, c_y = b[0] - a[0], b[1] - a[1]
                denominator = d_x * c_y - d_y * c_x
                if abs(denominator) > 1e-12:
                    t = ((a[0] - start[0]) * c_y - (a[1] - start[1]) * c_x) / denominator
                    output.append((start[0] + t * d_x, start[1] + t * d_y))
            if end_inside:
                output.append(end)
            start, start_inside = end, end_inside
    return output


def compare_mt_pose_to_gt(predicted, ground_truth, width, height):
    """Numerical polygon assessment for an MT attempt with operator-defined GT.

    NMCE_pct = mean Euclidean distance between corresponding corners / diagonal
    of the GT envelope * 100. Polygon IoU is the true polygon area intersection
    divided by union (not the enclosing bounding-box IoU). No synthetic scores.
    """
    pred = validate_mt_quad(predicted, width, height)
    gt = validate_mt_quad(ground_truth, width, height)
    gt_width = max(x for x, _ in gt) - min(x for x, _ in gt)
    gt_height = max(y for _, y in gt) - min(y for _, y in gt)
    diagonal = math.hypot(gt_width, gt_height)
    if diagonal <= 0:
        raise ValueError("Nieprawidłowa przekątna polygonu GT")
    distances = [math.dist(a, b) for a, b in zip(pred, gt)]
    mean_distance = sum(distances) / 4
    area_pred, area_gt = abs(_signed_area(pred)), abs(_signed_area(gt))
    intersection = abs(_signed_area(_clip_convex(pred, gt)))
    union = area_pred + area_gt - intersection
    if union <= 0:
        raise ValueError("Nieprawidłowe pole sumy polygonów")
    iou = max(0.0, min(1.0, intersection / union))
    return {
        "nmce_pct": mean_distance / diagonal * 100,
        "polygon_iou": iou,
        "polygon_iou_pct": iou * 100,
        "mean_corner_error_px": mean_distance,
        "per_corner_errors_px": distances,
        "gt_envelope_diagonal_px": diagonal,
        "gt_area_px2": area_gt,
        "pred_area_px2": area_pred,
    }
