from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from auto_annotation_tool.gui.z3_gt_geometry_recovery import recover_z2_gt_geometry
from auto_annotation_tool.gui import z3_review_runtime as review
from test_z3_edit_gt_runtime import quality_host


def fixture():
    image = np.full((64, 256, 3), 240, dtype=np.uint8)
    image[:, :20] = (180, 65, 10)
    boxes = [(30, 8, 50, 56), (60, 8, 80, 56), (95, 8, 114, 56),
             (125, 8, 145, 56), (155, 8, 170, 56), (180, 8, 200, 56), (220, 8, 240, 56)]
    for x1, y1, x2, y2 in boxes:
        image[y1:y2, x1:x2] = 20
    seeds = [[0, 0, 21, 64], [29, 6, 51, 58], [59, 6, 81, 58],
             [94, 6, 146, 58], [154, 6, 171, 58], [179, 7, 201, 34], [219, 6, 241, 58]]
    nms = [{"character": "", "bbox": list(box), "box_source": "yolo_box", "method": "yolo_box",
            "confidence": 0.1, "sign_source": ""} for box in seeds]
    raw_chars = [deepcopy(nms[i]) for i in (0, 1, 2, 4, 5)]
    data = {"ground_truth_text": "KR5EY24", "ground_truth_source": "manual_z2",
            "plate_layout": "single_row", "plate_image_width": 256, "plate_image_height": 64,
            "characters": [], "raw_detection": {"contract": "gt_blind.v1", "result_hash": "raw", "characters": raw_chars,
                                                  "detection_method": "yolo_box", "pipeline_blocks": ["yolo_box"]},
            "yolo_nms_detections": nms, "status": "needs_fix"}
    return image, data


def test_recovers_distinct_image_components_and_excludes_eu_strip_without_mutating_inputs():
    image, data = fixture()
    before = deepcopy(data)
    raw = data["raw_detection"]["characters"]
    records, report = recover_z2_gt_geometry(SimpleNamespace(), data, raw, plate_image=image,
                                            settings={"seq_max_w": 1.2, "seq_min_h": .7})
    assert report["recovered_count"] == 7
    assert report["blue_strip_excluded"]
    assert report["requires_review"]
    assert report["threshold_checks"] >= 2
    assert len(records) == 7
    assert min(r["bbox"][0] for r in records) >= 29
    assert records[3]["bbox"][2] - records[3]["bbox"][0] <= 24  # E alone.
    assert records[5]["bbox"][3] - records[5]["bbox"][1] >= 45  # Complete 2.
    assert records[3]["box_source"] == "generated_box"
    assert records[5]["box_source"] == "generated_box"
    assert data == before


def test_full_preparation_follows_yb_raw_without_inserting_segmentation():
    image, data = fixture()
    host = quality_host(data)
    before = deepcopy(data["raw_detection"])
    with patch("auto_annotation_tool.gui.z3_gt_geometry_recovery.recover_z2_gt_geometry",
               side_effect=AssertionError("pipeline ran an undeclared segmentation")) as segmentation:
        assert review.prepare_working_annotation_from_raw(host, data, plate_id="plate", plate_image=image,
                                                         geometry_settings={"seq_max_w": 1.2, "seq_min_h": .7})
    segmentation.assert_not_called()
    assert len(data["characters"]) == 5
    assert all(rec["box_source"] == "yolo_box" for rec in data["characters"])
    assert data["live_gt_assist"]["reason"] == "box_count_mismatch"
    assert data["raw_detection"] == before
    assert data["review_state"]["status"] == "in_progress"
    assert data["status"] == "needs_fix"
    assert not data["gold_state"]["approved"]
    saved = host._serialize_character_records(data["characters"], data=data)
    assert all(rec["box_source"] == "yolo_box" for rec in saved)
    assert "gt_geometry_recovery" not in data


@pytest.mark.parametrize("change", ["missing_component", "blank", "manual_gt", "two_row", "manual_box", "unanchored"])
def test_no_recovery_for_ambiguous_or_protected_input(change):
    image, data = fixture()
    raw = data["raw_detection"]["characters"]
    if change == "missing_component":
        image[:, 95:114] = 240
    elif change == "blank":
        image[:] = 240
    elif change == "manual_gt":
        data["ground_truth_source"] = "manual_z3"
    elif change == "two_row":
        data["plate_layout"] = "two_row"
    elif change == "manual_box":
        raw[0]["box_source"] = "manual_box"
    elif change == "unanchored":
        data["yolo_nms_detections"] = data["yolo_nms_detections"][:3]
    before = deepcopy(data)
    result, report = recover_z2_gt_geometry(SimpleNamespace(), data, raw, plate_image=image)
    assert report is None
    assert result is raw
    assert data == before


def test_rerun_keeps_manual_work_even_with_recoverable_image():
    image, data = fixture()
    host = quality_host(data)
    data["characters"] = [{"character": "X", "bbox": [30, 8, 50, 56], "box_source": "manual_box"}]
    data["review_state"] = {"status": "in_progress", "human_edited": True}
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="plate", plate_image=image)
    assert data == before


def test_identical_raw_can_refresh_old_automatic_preparation_once():
    image, data = fixture()
    host = quality_host(data)
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="plate")
    assert len(data["characters"]) == 5
    legacy_chars, report = recover_z2_gt_geometry(host, data, data["raw_detection"]["characters"], plate_image=image)
    data["characters"] = legacy_chars
    data["gt_geometry_recovery"] = report
    host._apply_live_gt_assist(data, prepare=True)
    data["working_annotation"].pop("pipeline_preparation_version")
    data["working_annotation"]["gt_geometry_preparation_version"] = "z2_gt_image_recovery.v1"
    data["working_annotation"]["automatic_content_hash"] = review._working_annotation_fingerprint(data)
    assert len(data["characters"]) == 7
    raw = deepcopy(data["raw_detection"])
    assert review.can_refresh_automatic_working_annotation(data)
    assert review.prepare_working_annotation_from_raw(host, data, plate_id="plate", plate_image=image)
    assert len(data["characters"]) == 5
    assert all(rec["box_source"] == "yolo_box" for rec in data["characters"])
    assert data["raw_detection"] == raw
    assert not review.can_refresh_automatic_working_annotation(data)
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data, plate_id="plate", plate_image=image)
    assert data == before
