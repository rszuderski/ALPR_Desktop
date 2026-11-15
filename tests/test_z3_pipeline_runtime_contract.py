from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.z3_detection_runtime import resolve_runtime_pipeline
from auto_annotation_tool.gui.z3_detection_runtime import build_yolo_box_only_records
from auto_annotation_tool.character_recognition.char_detector import DetectionMethod
from auto_annotation_tool.gui.z3_detection_pipeline_ui import compile_detection_pipeline_blocks
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui.z3_list_review import clear_automatic_boxes
from test_z3_edit_gt_runtime import quality_host


PATTERNS = [("OCR", ["ocr_symbol"], "generated_box"),
            ("YOLO_BOX", ["yolo_box"], "yolo_box"),
            ("YOLO_SYMBOL", ["yolo_symbol"], "generated_box"),  # YS changes symbols in existing boxes.
            ("YOLO", ["yolo_box", "yolo_symbol"], "yolo_box"),
            ("BOTH", ["ocr_symbol", "yolo_box"], "yolo_box"),
            ("BOTH", ["ocr_symbol", "yolo_box", "yolo_symbol"], "yolo_box"),
            ("YOLO_OCR", ["yolo_box", "ocr_symbol"], "yolo_box")]


@pytest.mark.parametrize("method,blocks,source", PATTERNS)
def test_committed_builder_blocks_override_stale_method_card(method, blocks, source):
    host = SimpleNamespace(local_session={"char_detection_pipeline_blocks": blocks},
                           _get_detection_method_key=lambda:"OCR",
                           _get_detection_pipeline_blocks=Mock(return_value=["ocr_symbol"]))
    plan = resolve_runtime_pipeline(host)
    assert plan["method_key"] == method
    assert plan["blocks"] == blocks
    assert plan == compile_detection_pipeline_blocks(blocks)
    host._get_detection_pipeline_blocks.assert_not_called()


def test_unsupported_builder_chain_is_rejected_instead_of_running_ocr():
    host = SimpleNamespace(local_session={"char_detection_pipeline_blocks": ["yolo_symbol", "ocr_symbol"]})
    with pytest.raises(ValueError, match="łańcuch"):
        resolve_runtime_pipeline(host)


def test_pipeline_snapshot_is_used_even_if_current_builder_has_changed():
    host = SimpleNamespace(local_session={"char_detection_pipeline_blocks": ["ocr_symbol"]})
    assert resolve_runtime_pipeline(host, ["yolo_box"])["method_key"] == "YOLO_BOX"


@pytest.mark.parametrize("method,blocks,source", PATTERNS)
def test_working_preparation_and_render_preserve_pipeline_box_geometry(method, blocks, source):
    rec = {"character": "", "bbox": [10., 5., 30., 50.], "confidence": .8,
           "method": "yolo_box" if source == "yolo_box" else "ocr", "box_source": source, "sign_source": ""}
    raw = {"contract": "gt_blind.v1", "result_hash": "raw", "detection_method": method.lower(),
           "pipeline_blocks": blocks, "characters": [rec]}
    data = {"ground_truth_text": "A", "ground_truth_source": "manual_z2", "plate_layout": "single_row",
            "plate_image_width": 100, "plate_image_height": 64, "characters": [], "raw_detection": raw,
            "status": "needs_fix"}
    host = quality_host(data)
    host._get_preview_box_mode_key = lambda:"AUTO"
    before = deepcopy(raw)
    assert review.prepare_working_annotation_from_raw(host, data)
    assert data["characters"][0]["bbox"] == rec["bbox"]
    assert data["characters"][0]["box_source"] == source
    assert data["characters"][0]["sign_source"] == "gt_assisted"
    assert host._get_preview_box_records(data)[0][0]["box_source"] == source
    assert host._serialize_character_records(data["characters"], data=data)[0]["box_source"] == source
    assert data["raw_detection"] == before
    assert data["working_annotation"]["pipeline_blocks"] == blocks


def test_runtime_stores_and_executes_the_same_declared_blocks():
    source = (Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z3_detection_runtime.py").read_text(encoding="utf-8")
    assert 'method_key = pipeline["method_key"]' in source
    assert source.count('"pipeline_blocks": list(pipeline["blocks"])') >= 3
    assert 'method = DetectionMethod(method_key.lower())' in source


def test_single_yb_builder_runs_real_box_filter_then_renders_yb_only_without_ocr():
    import numpy as np
    from test_z3_yb_sequence_guards import make_detector
    plan = resolve_runtime_pipeline(SimpleNamespace(local_session={"char_detection_pipeline_blocks": ["yolo_box"]}))
    detector = make_detector(DetectionMethod(plan["method_key"].lower()))
    detector._detect_with_ocr = Mock(side_effect=AssertionError("YB ran OCR"))
    detector._detect_with_yolo_boxes_and_ocr = Mock(side_effect=AssertionError("YB ran OCR on boxes"))
    image = np.full((64, 256, 3), 240, dtype=np.uint8)
    detections = detector.detect(image)
    data = {"ground_truth_text": "KR5EY24", "ground_truth_source": "manual_z2", "plate_layout": "single_row",
            "plate_image_width": 256, "plate_image_height": 64, "characters": [], "status": "needs_fix"}
    host = quality_host(data)
    host._get_preview_box_mode_key = lambda:"AUTO"
    records = build_yolo_box_only_records(host, detections)
    data["raw_detection"] = {"contract": "gt_blind.v1", "result_hash": "raw", "detection_method": "yolo_box",
                             "pipeline_blocks": plan["blocks"], "characters": deepcopy(records)}
    data["yolo_nms_detections"] = [det.to_dict() for det in detector.last_yolo_nms_detections]
    raw = deepcopy(data["raw_detection"])
    assert review.prepare_working_annotation_from_raw(host, data, from_detection=True, plate_image=image)
    assert len(data["characters"]) == len(detections) == 6
    assert all(rec["box_source"] == "yolo_box" for rec in host._get_preview_box_records(data)[0])
    assert data["raw_detection"] == raw
    assert data["live_gt_assist"]["reason"] == "box_count_mismatch"
    detector._detect_with_ocr.assert_not_called()
    detector._detect_with_yolo_boxes_and_ocr.assert_not_called()


@pytest.mark.parametrize("approved", [False, True])
def test_clear_legacy_segmentation_then_yb_rerun_preserves_manual_geometry(approved):
    manual = {"character": "M", "bbox": [10., 5., 30., 50.], "confidence": 1.,
              "method": "manual", "box_source": "manual_box", "sign_source": "manual_sign"}
    generated = {"character": "B", "bbox": [40., 5., 60., 50.], "confidence": .8,
                 "method": "segment", "box_source": "generated_box", "sign_source": "gt_assisted"}
    detected = {**generated, "character": "", "method": "yolo_box", "box_source": "yolo_box", "sign_source": ""}
    data = {"ground_truth_text": "MB", "ground_truth_source": "manual_z2", "plate_layout": "single_row",
            "plate_image_width": 100, "plate_image_height": 64, "characters": [manual, generated],
            "raw_detection": {"contract": "gt_blind.v1", "result_hash": "yb-raw", "detection_method": "yolo_box",
                              "pipeline_blocks": ["yolo_box"], "characters": [detected]},
            "status": "perfect" if approved else "needs_fix", "gold_state": {"approved": approved},
            "review_state": {"status": "approved" if approved else "in_progress", "human_edited": True},
            "gt_geometry_recovery": {"recovered_count": 2}}
    host = quality_host(data)
    host._get_preview_box_mode_key = lambda:"AUTO"
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data, from_detection=True)
    assert data == before
    assert clear_automatic_boxes(host, "plate")["removed_boxes"] == 1
    assert review.prepare_working_annotation_from_raw(host, data, from_detection=True)
    rendered = host._get_preview_box_records(data)[0]
    assert [rec["box_source"] for rec in rendered] == ["manual_box", "yolo_box"]
    assert rendered[0]["bbox"] == before["characters"][0]["bbox"]
    assert rendered[0]["character"] == "M"
    assert data["raw_detection"] == before["raw_detection"]
    assert "gt_geometry_recovery" not in data
