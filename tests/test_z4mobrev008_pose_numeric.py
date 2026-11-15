"""Z4MOBREV008: deterministic corner geometry; no inferred or synthetic GT."""
from __future__ import annotations

import math
import pytest

from auto_annotation_tool.ranking.mobile_pose_metrics import (
    POSE_GT_SCHEMA, compare_mt_pose_to_gt, validate_mt_quad,
)

GT = ((10, 20), (90, 20), (90, 40), (10, 40))


def test_schema_declares_manual_annotation_not_inference():
    assert POSE_GT_SCHEMA == "manual_mt_input_quad_v1"


def test_identical_four_corner_gt_yields_nmce_zero_and_iou_one():
    m = compare_mt_pose_to_gt(GT, GT, 200, 200)
    assert m["nmce_pct"] == pytest.approx(0, abs=1e-12)
    assert m["polygon_iou"] == pytest.approx(1, abs=1e-12)
    assert m["per_corner_errors_px"] == [0, 0, 0, 0]


def test_known_shift_proves_normalization_uses_gt_diagonal_not_image_diagonal():
    pred = tuple((x + 5, y) for x, y in GT)
    m = compare_mt_pose_to_gt(pred, GT, 200, 200)
    assert m["mean_corner_error_px"] == pytest.approx(5)
    assert m["nmce_pct"] == pytest.approx(100 * 5 / math.hypot(80, 20))
    assert m["polygon_iou"] == pytest.approx(75 / 85)


def test_polygon_iou_not_equivalent_to_bbox_iou():
    gt = ((20, 20), (100, 20), (90, 50), (30, 50))
    pred = ((30, 10), (90, 10), (100, 60), (20, 60))
    m = compare_mt_pose_to_gt(pred, gt, 200, 200)
    assert 0 < m["polygon_iou"] < 1
    # Distinct polygon/rectangle areas, independent check on bounding boxes.
    box_inter = 80 * 30
    box_union = 80 * 50
    assert m["polygon_iou"] != pytest.approx(box_inter / box_union)


def test_non_overlapping_pose_has_zero_polygon_iou():
    far = tuple((x + 110, y) for x, y in GT)
    assert compare_mt_pose_to_gt(far, GT, 250, 200)["polygon_iou"] == 0


def test_convexity_and_bounds_are_required_for_operator_gt():
    for polygon in (
        ((10, 10), (90, 90), (90, 10), (10, 90)),  # crossed
        ((10, 10), (20, 20), (30, 30), (40, 40)),  # no area
        ((10, 10), (90, 10), (110, 20), (10, 50)), # beyond source
        ((10, 10), (60, 10), (15, 15), (10, 50)),  # concave
    ):
        with pytest.raises(ValueError):
            validate_mt_quad(polygon, 100, 100)


def test_float_nan_inf_and_missing_gt_never_become_metrics():
    for invalid in (None, (), GT[:3], ((float('nan'), 10), *GT[1:]),
                    ((float('inf'), 10), *GT[1:])):
        with pytest.raises(ValueError):
            compare_mt_pose_to_gt(GT, invalid, 200, 200)


def test_clockwise_and_counterclockwise_valid_geometry_iou_is_stable():
    ccw = tuple(reversed(GT))
    m = compare_mt_pose_to_gt(ccw, GT, 200, 200)
    assert m["polygon_iou"] == pytest.approx(1.0)
    # Matching indexed corners is a different contract from shape overlap.
    assert m["nmce_pct"] > 0


def test_preserve_fractional_precision_and_image_scale():
    gt = tuple((x * 1.5, y * 1.5) for x, y in GT)
    pred = tuple((x * 1.5, y * 1.5) for x, y in GT)
    assert compare_mt_pose_to_gt(pred, gt, 300, 300)["nmce_pct"] == 0
