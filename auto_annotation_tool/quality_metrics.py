#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pomocnicze metryki jakości geometrii dla tablic i boxów znaków.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .rectification.polygon_validator import PolygonValidator


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _polygon_bbox(points: List[Tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [float(x) for x, _y in points]
    ys = [float(y) for _x, y in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b[:4]]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _edge_lengths(points: List[Tuple[float, float]]) -> list[float]:
    if len(points) != 4:
        return []
    result: list[float] = []
    for idx in range(4):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % 4]
        result.append(float(np.hypot(float(x2) - float(x1), float(y2) - float(y1))))
    return result


def _ratio_score(value: float, *, ideal_min: float, ideal_max: float, hard_min: float, hard_max: float) -> float:
    value = _safe_float(value, 0.0)
    if value <= 0.0:
        return 0.0
    if ideal_min <= value <= ideal_max:
        return 1.0
    if value < hard_min or value > hard_max:
        return 0.0
    if value < ideal_min:
        span = max(1e-6, ideal_min - hard_min)
        return _clamp01((value - hard_min) / span)
    span = max(1e-6, hard_max - ideal_max)
    return _clamp01((hard_max - value) / span)


def compute_plate_polygon_fit_metrics(
    detection_confidence: float,
    polygon: List[Tuple[float, float]] | None,
    bbox: tuple[float, float, float, float] | None,
    *,
    keypoints: Iterable[tuple[float, float, float]] | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict:
    polygon = list(polygon or [])
    bbox = tuple(bbox or (0.0, 0.0, 0.0, 0.0))
    det_conf = _clamp01(detection_confidence)

    keypoint_confs: list[float] = []
    for kp in list(keypoints or [])[:4]:
        if len(kp) < 2:
            continue
        kp_conf = _safe_float(kp[2], 1.0) if len(kp) >= 3 else 1.0
        keypoint_confs.append(_clamp01(kp_conf))

    if keypoint_confs:
        kp_mean = sum(keypoint_confs) / float(len(keypoint_confs))
        kp_min = min(keypoint_confs)
        keypoint_score = _clamp01((0.7 * kp_mean) + (0.3 * kp_min))
    else:
        keypoint_score = _clamp01(det_conf * 0.85)

    ordered_polygon = list(polygon)
    try:
        ordered_polygon = PolygonValidator.order_points(polygon[:4]) if len(polygon) >= 4 else polygon
        ordered_polygon = [(float(x), float(y)) for x, y in ordered_polygon[:4]]
    except Exception:
        ordered_polygon = [(float(x), float(y)) for x, y in polygon[:4]]

    valid_quad = bool(len(ordered_polygon) == 4 and PolygonValidator.is_valid_quad(ordered_polygon))
    polygon_bbox = _polygon_bbox(ordered_polygon) if len(ordered_polygon) == 4 else bbox
    polygon_area = _polygon_area(ordered_polygon) if len(ordered_polygon) == 4 else 0.0
    bbox_area = _bbox_area(bbox)
    bbox_fill_ratio = (polygon_area / bbox_area) if bbox_area > 1e-6 else 0.0
    fill_score = _ratio_score(
        bbox_fill_ratio,
        ideal_min=0.62,
        ideal_max=1.02,
        hard_min=0.30,
        hard_max=1.15,
    )

    lengths = _edge_lengths(ordered_polygon)
    if len(lengths) == 4:
        top, right, bottom, left = lengths
        width_consistency = min(top, bottom) / max(top, bottom, 1e-6)
        height_consistency = min(left, right) / max(left, right, 1e-6)
        avg_width = (top + bottom) * 0.5
        avg_height = (left + right) * 0.5
        aspect_ratio = (avg_width / max(avg_height, 1e-6)) if avg_height > 0 else 0.0
        side_consistency_score = _clamp01((width_consistency + height_consistency) * 0.5)
        aspect_score = _ratio_score(
            aspect_ratio,
            ideal_min=1.7,
            ideal_max=7.5,
            hard_min=1.0,
            hard_max=10.0,
        )
    else:
        side_consistency_score = 0.0
        aspect_score = 0.0

    shape_score = (
        0.15 * (1.0 if valid_quad else 0.0)
        + 0.35 * fill_score
        + 0.25 * side_consistency_score
        + 0.25 * aspect_score
    )
    shape_score = _clamp01(shape_score)

    bbox_alignment_score = _clamp01(_bbox_iou(bbox, polygon_bbox))

    if image_size is not None:
        try:
            width, height = int(image_size[0]), int(image_size[1])
        except Exception:
            width, height = 0, 0
        image_area = max(1.0, float(width * height))
        relative_area = polygon_area / image_area
        size_score = _ratio_score(
            relative_area,
            ideal_min=0.0015,
            ideal_max=0.18,
            hard_min=0.0002,
            hard_max=0.45,
        )
    else:
        relative_area = 0.0
        size_score = 1.0

    fit_score = _clamp01(
        0.45 * keypoint_score
        + 0.25 * shape_score
        + 0.20 * bbox_alignment_score
        + 0.10 * size_score
    )
    if fit_score >= 0.85:
        fit_label = "wysokie"
    elif fit_score >= 0.65:
        fit_label = "srednie"
    else:
        fit_label = "niskie"

    return {
        "fit_score": fit_score,
        "fit_label": fit_label,
        "keypoint_score": _clamp01(keypoint_score),
        "shape_score": _clamp01(shape_score),
        "bbox_alignment_score": _clamp01(bbox_alignment_score),
        "size_score": _clamp01(size_score),
        "relative_area": max(0.0, float(relative_area)),
        "bbox_fill_ratio": _clamp01(min(1.5, float(bbox_fill_ratio))) if bbox_fill_ratio >= 0.0 else 0.0,
        "keypoint_count": int(len(keypoint_confs)),
    }


def compute_character_box_fit_metrics(
    detection_confidence: float,
    bbox: tuple[float, float, float, float] | None,
    *,
    plate_size: tuple[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict:
    bbox = tuple(bbox or (0.0, 0.0, 0.0, 0.0))
    det_conf = _clamp01(detection_confidence)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    aspect_ratio = (height / max(width, 1e-6)) if width > 0 else 0.0
    aspect_score = _ratio_score(
        aspect_ratio,
        ideal_min=1.15,
        ideal_max=5.5,
        hard_min=0.6,
        hard_max=8.0,
    )

    if plate_size is not None:
        try:
            plate_w, plate_h = int(plate_size[0]), int(plate_size[1])
        except Exception:
            plate_w, plate_h = 0, 0
        rel_h = (height / max(float(plate_h), 1.0)) if plate_h > 0 else 0.0
        rel_w = (width / max(float(plate_w), 1.0)) if plate_w > 0 else 0.0
        height_score = _ratio_score(rel_h, ideal_min=0.22, ideal_max=0.95, hard_min=0.08, hard_max=1.05)
        width_score = _ratio_score(rel_w, ideal_min=0.03, ideal_max=0.34, hard_min=0.01, hard_max=0.50)
        size_score = _clamp01((height_score + width_score) * 0.5)
    elif image_size is not None:
        try:
            image_w, image_h = int(image_size[0]), int(image_size[1])
        except Exception:
            image_w, image_h = 0, 0
        rel_area = area / max(1.0, float(image_w * image_h))
        size_score = _ratio_score(rel_area, ideal_min=0.0008, ideal_max=0.08, hard_min=0.0001, hard_max=0.18)
    else:
        size_score = 1.0

    fit_score = _clamp01((0.55 * det_conf) + (0.25 * aspect_score) + (0.20 * size_score))
    if fit_score >= 0.85:
        fit_label = "wysokie"
    elif fit_score >= 0.65:
        fit_label = "srednie"
    else:
        fit_label = "niskie"

    return {
        "fit_score": fit_score,
        "fit_label": fit_label,
        "aspect_score": _clamp01(aspect_score),
        "size_score": _clamp01(size_score),
    }
