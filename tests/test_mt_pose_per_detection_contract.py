"""Z4MOBPOSE017 — provenance and strict MT Pose per-detection contract.

Run inside the Desktop repository. These tests never read or mutate real sessions.
"""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from auto_annotation_tool.gui import z4_mobile_sample_review as review


def attempt(attempt_id="A-2"):
    return {
        "id": attempt_id,
        "mt_status": "VALID_QUAD",
        "mt_invocation_id": "mt-invoke-1",
        "mt_pose_status": "VALID_QUAD",
        "mt_pose_corners_norm": "0.1:0.1;0.6:0.1;0.6:0.6;0.1:0.6",
        "mt_pose_source_width_px": "200",
        "mt_pose_source_height_px": "100",
        "mt_pose_coordinate_space": "source_frame_normalized_v1",
        "mt_pose_corner_order": "TL_TR_BR_BL",
        "roi_left": "0", "roi_top": "0", "roi_right": "200", "roi_bottom": "100",
        "input_width": "512", "input_height": "512", "input_scale": "2.56",
        "input_pad_x": "0", "input_pad_y": "128",
        "plate_left": "20", "plate_top": "10", "plate_right": "120", "plate_bottom": "60",
        "mt_input_evidence_entry": "samples/evidence/A-2.jpg",
    }


def session(samples=None):
    return SimpleNamespace(samples=samples or {}, sample_ids_by_attempt={})


def test_pose_can_be_read_from_mt_record_without_any_crop():
    points = review._linked_pose_quad_on_mt_input(session(), attempt())
    expected = ((51.2, 153.6), (307.2, 153.6), (307.2, 281.6), (51.2, 281.6))
    assert points is not None
    for actual, wanted in zip(points, expected):
        assert actual == pytest.approx(wanted)


def test_status_provenance_is_direct_mt():
    status, msg = review._pose_source_status(session(), attempt())
    assert status == "mt_record_pose"
    assert "MT" in msg


def test_new_invalid_quad_is_not_replaced_with_legacy_crop():
    row = attempt()
    row["mt_pose_status"] = "INVALID_QUAD"
    linked = {"S": {"attempt_id": "A-2", "plate_geometry": {
        "available": True, "plate_corners_norm": [[.1,.1],[.6,.1],[.6,.6],[.1,.6]],
        "source_width_px":200, "source_height_px":100}}}
    assert review._linked_pose_quad_on_mt_input(session(linked), row) is None


def test_new_valid_status_without_points_is_not_repaired_by_crop():
    row = attempt()
    row["mt_pose_corners_norm"] = ""
    linked = {"S": {"attempt_id": "A-2", "plate_geometry": {
        "available":True,"plate_corners_norm":[[.1,.1],[.6,.1],[.6,.6],[.1,.6]],
        "source_width_px":200,"source_height_px":100}}}
    assert review._linked_pose_quad_on_mt_input(session(linked), row) is None
    assert review._pose_source_status(session(linked), row)[0] == "mt_pose_record_inconsistent"


def test_bbox_without_pose_never_becomes_four_corners():
    row = attempt()
    row["mt_pose_status"] = "MISSING_KEYPOINTS"
    row["mt_pose_corners_norm"] = ""
    assert review._linked_pose_quad_on_mt_input(session(), row) is None


def test_each_detection_uses_its_own_points_even_if_same_mt_invoke():
    first = attempt("A-1")
    second = attempt("A-2")
    first["mt_pose_corners_norm"] = "0.2:0.2;0.4:0.2;0.4:0.4;0.2:0.4"
    first["plate_left"],first["plate_top"],first["plate_right"],first["plate_bottom"] = "40","20","80","40"
    p1 = review._linked_pose_quad_on_mt_input(session(), first)
    p2 = review._linked_pose_quad_on_mt_input(session(), second)
    assert p1 and p2 and p1 != p2


@pytest.mark.parametrize("field,value", [
    ("mt_pose_source_width_px", "0"),
    ("mt_pose_corner_order", "TR_TL_BL_BR"),
    ("mt_pose_coordinate_space", "mt_input_normalized"),
    ("mt_pose_corners_norm", "0.1:0.1;0.6:0.1;0.6:0.6"),
    ("mt_pose_corners_norm", "NaN:0.1;0.6:0.1;0.6:0.6;0.1:0.6"),
    ("mt_pose_corners_norm", "0.1:0.1;1.1:0.1;0.6:0.6;0.1:0.6"),
    ("mt_pose_corners_norm", "0.1:0.1;0.6:0.6;0.6:0.1;0.1:0.6"),
    ("mt_pose_corners_norm", "0.1:0.1;0.1:0.1;0.6:0.6;0.1:0.6"),
    ("input_width", "0"),
    ("input_scale", "0"),
    ("roi_left", "220"),
])
def test_rejects_corrupt_geometry(field, value):
    row=attempt()
    row[field]=value
    assert review._pose_quad_from_mt_attempt(row) is None


def test_legacy_crop_read_works_only_for_matching_attempt_id():
    row = attempt()
    for key in list(row):
        if key.startswith("mt_pose_"):
            row.pop(key)
    sample = {"attempt_id":"A-2", "source_width_px":"200", "source_height_px":"100",
              "plate_corners_norm":"0.1:0.1;0.6:0.1;0.6:0.6;0.1:0.6"}
    s = session({"S":sample})
    assert review._linked_pose_quad_on_mt_input(s,row) is not None
    assert review._pose_source_status(s,row)[0] == "pose_available"
    sample["attempt_id"] = "another"
    assert review._linked_pose_quad_on_mt_input(s,row) is None


def test_new_pose_numerical_metric_does_not_need_mz():
    row=attempt()
    points=review._linked_pose_quad_on_mt_input(session(),row)
    metrics=review.compare_mt_pose_to_gt(points,points,512,512)
    assert metrics["nmce_pct"] == pytest.approx(0.0)
    assert metrics["polygon_iou_pct"] == pytest.approx(100.0)


def test_historical_no_crop_has_no_fabricated_pose():
    row = attempt()
    for key in list(row):
        if key.startswith("mt_pose_"):
            row.pop(key)
    assert review._linked_pose_quad_on_mt_input(session(),row) is None
    assert review._pose_source_status(session(),row)[0] == "no_linked_crop"


def test_model_bbox_is_not_drawn_as_pose_overlay():
    import inspect
    source = inspect.getsource(review.MobileSampleReviewWindow._paint_detection_overlay)
    assert '"detection_box"' not in source
    assert '"detection_polygon"' in source


def test_declared_new_schema_never_falls_back_to_legacy_crop():
    row=attempt()
    for key in list(row):
        if key.startswith("mt_pose_"):
            row.pop(key)
    linked={"S": {"attempt_id":"A-2", "source_width_px":"200", "source_height_px":"100",
                  "plate_corners_norm":"0.1:0.1;0.6:0.1;0.6:0.6;0.1:0.6"}}
    s=session(linked)
    s.bundle=SimpleNamespace(sample_schema={"capabilities":{"mt_pose_per_detection":True}})
    assert review._linked_pose_quad_on_mt_input(s,row) is None
    assert review._pose_source_status(s,row)[0] == "mt_pose_unavailable"


def test_bbox_coordinates_do_not_decide_if_per_detection_pose_is_valid():
    row=attempt()
    row.update(plate_left="0",plate_top="0",plate_right="199",plate_bottom="99")
    assert review._pose_quad_from_mt_attempt(row) is not None
