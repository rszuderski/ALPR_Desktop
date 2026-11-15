# -*- coding: utf-8 -*-
"""AKT89E: MT false negatives must not require a text GT or an MZ inference.

The main application's fixture deliberately uses a minimal 2-image Canvas
review. This test does not write any real experiment data.
"""
from __future__ import annotations

from copy import deepcopy
from auto_annotation_tool.gui.mobile_compare_review_gate import (
    inspect_canvas, paired_geometry_text,
)

QUAD = [[10.0, 10.0], [110.0, 10.0], [110.0, 30.0], [10.0, 30.0]]


def example():
    return {
        "provenance": "ASSISTED_CANVAS_SOURCE_ROI", "independent_gt": False,
        "scene_count": 2, "checked_scenes": 2, "pending_scenes": 0,
        "excluded_scenes": 0, "incomplete_checked_scenes": 0,
        "operator_errors": 0, "expected_plates": 2, "linked_plates": 1,
        "missed_plates": 1, "false_positive_mt": 0, "geometry_n": 1,
        "evaluated_reads": 1, "exact_reads": 1,
        "gt_characters": 4, "text_errors": 0, "cer": 0.0,
        "scenes": [{"status": "CHECKED"}, {"status": "CHECKED"}],
        "rows": [
            {"included": True, "outcome": "LINKED_MT", "post_nms_suppressed_raw": False,
             "original_mt_quad": QUAD, "operator_quad": QUAD,
             "gt_text": "AB12", "mz_status": "READ", "mz_prediction": "AB12",
             "text_exclusion_reason": "", "polygon_iou_pct": 94.0,
             "nmce_pct": 2.0, "cer": 0.0},
            {"included": True, "outcome": "MISSED_MT", "post_nms_suppressed_raw": False,
             "original_mt_quad": None, "operator_quad": QUAD,
             "gt_text": "", "mz_status": "NOT_RUN", "mz_prediction": "",
             "text_exclusion_reason": "", "polygon_iou_pct": None,
             "nmce_pct": None, "cer": None},
        ],
    }


def check(data):
    return inspect_canvas(data, "COMPLETED")


def test_approved_session_with_one_mt_fn_and_no_gt_is_ready():
    c = example()
    ok, detail = check(c)
    assert ok, detail
    assert "1 pominięć" in detail and "1 odczytów" in detail
    assert len(list(paired_geometry_text(c))) == 1
    assert c["gt_characters"] == 4 and c["geometry_n"] == 1


def test_old_historical_gt_of_missed_mt_is_allowed_but_not_evaluated():
    c = example()
    c["rows"][1]["gt_text"] = "GXW9"
    ok, reason = check(c)
    assert ok, reason
    assert c["gt_characters"] == 4
    assert len(list(paired_geometry_text(c))) == 1


def test_linked_mt_still_requires_gt_or_documented_exclusion():
    c = example()
    c["rows"][0]["gt_text"] = ""
    assert not check(c)[0]


def test_missed_mt_with_mz_read_is_inconsistent():
    c = example()
    c["rows"][1]["mz_status"] = "READ"
    assert not check(c)[0]


def test_missed_mt_with_prediction_is_inconsistent():
    c = example()
    c["rows"][1]["mz_prediction"] = "GXW9"
    assert not check(c)[0]


def test_missed_mt_with_cer_is_inconsistent():
    c = example()
    c["rows"][1]["cer"] = 1.0
    assert not check(c)[0]


def test_missed_mt_requires_operator_reference_quad_for_fn():
    c = example()
    c["rows"][1]["operator_quad"] = None
    assert not check(c)[0]


def test_unapproved_session_cannot_be_compared():
    assert not inspect_canvas(example(), "IN_PROGRESS")[0]


def test_missing_scene_decision_cannot_be_compared():
    c = example()
    c["pending_scenes"] = 1
    assert not check(c)[0]
