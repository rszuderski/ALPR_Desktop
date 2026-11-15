import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from auto_annotation_tool.character_recognition.char_detector import CharacterDetector, DetectionMethod
from auto_annotation_tool.gui.z3_gt_assist_runtime import apply_live_gt_assist


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def make_detector(method, min_height=0.60, max_width=1.20, confidence=0.8):
    boxes = [[i * 25, 0, i * 25 + 10, 40] for i in range(8)]
    boxes[3][2] = boxes[3][0] + 20  # E spans two normal character widths.
    boxes[5][3] = 22  # 2 is 55% of the reference height.
    result = SimpleNamespace(boxes=SimpleNamespace(
        xyxy=Tensor(boxes), conf=Tensor([confidence] * 8), cls=Tensor(list(range(8)))))
    model = Mock(return_value=[result])
    model.names = dict(enumerate("KR5EY24X"))
    return CharacterDetector(method=method, yolo_model=model,
                             yolo_box_confidence=0.01, yolo_symbol_confidence=0.50,
                             yolo_sequence_min_height_ratio=min_height,
                             yolo_sequence_max_width_ratio=max_width)


@pytest.mark.parametrize("method", [DetectionMethod.YOLO_BOX, DetectionMethod.YOLO_OCR])
def test_geometry_guards_reach_yb_and_ocr_crops_and_preserve_nms(method):
    detector = make_detector(method)
    detector._detect_with_yolo_boxes_and_ocr = Mock(side_effect=lambda image, boxes: list(boxes))
    result = detector.detect(np.zeros((64, 256, 3), dtype=np.uint8))
    assert [r.character for r in result] == list("KR5Y4X")
    assert [r.character for r in detector.last_yolo_nms_detections] == list("KR5EY24X")
    assert len(detector.last_yolo_raw_detections) == 8
    assert result == detector.last_yolo_box_detections
    if method == DetectionMethod.YOLO_OCR:
        assert detector._detect_with_yolo_boxes_and_ocr.call_args.args[1] == result


def test_box_confidence_remains_independent_of_symbol_confidence():
    detector = make_detector(DetectionMethod.YOLO_BOX, confidence=0.1)
    result = detector.detect(np.zeros((64, 256, 3), dtype=np.uint8))
    assert len(result) == 6
    assert detector.last_yolo_detections == []


def test_changing_pipeline_guard_values_changes_box_result():
    detector = make_detector(DetectionMethod.YOLO_BOX)
    image = np.zeros((64, 256, 3), dtype=np.uint8)
    assert len(detector.detect(image)) == 6
    detector.yolo_sequence_min_height_ratio = 0.5
    detector.yolo_sequence_max_width_ratio = 2.1
    assert len(detector.detect(image)) == 8


def test_empty_guarded_result_does_not_restore_rejected_nms_boxes():
    detector = make_detector(DetectionMethod.YOLO_BOX)
    detector._filter_yolo_sequence_consistency = Mock(return_value=[])
    assert detector.detect(np.zeros((64, 256, 3), dtype=np.uint8)) == []
    assert len(detector.last_yolo_nms_detections) == 8


@pytest.mark.parametrize("box_count,reason", [(8, "ready"), (6, "box_count_mismatch")])
def test_z2_gt_limits_working_count_after_raw_and_never_fills_missing_geometry(box_count, reason):
    chars = [{"character": "", "bbox": [i * 25, 0, i * 25 + 10, 40],
              "box_source": "yolo_box", "confidence": 0.8} for i in range(box_count)]
    raw = {"contract": "gt_blind.v1", "characters": copy.deepcopy(chars)}
    before = copy.deepcopy(raw)
    data = {"ground_truth_text": "KR5EY24", "ground_truth_source": "manual_z2",
            "characters": chars, "raw_detection": raw, "review_state": {"status": "in_progress"}}
    host = SimpleNamespace(
        _sort_character_records_by_x=lambda records, data=None: sorted(records, key=lambda r:r["bbox"][0]),
        _get_preview_ground_truth_text=lambda data:data["ground_truth_text"],
        _update_preview_plate_layout_metadata=Mock(),
        _annotate_preview_character_reading_positions=lambda records, data=None:records,
        _preview_layout_separator_conflicts_with_chars=lambda data, records:False)
    result = apply_live_gt_assist(host, data, prepare=True)
    assert result["reason"] == reason
    assert len(data["characters"]) == min(box_count, 7)
    assert data["raw_detection"] == before
    if reason == "ready":
        assert ''.join(r["character"] for r in data["characters"]) == "KR5EY24"


def test_pz2_runtime_uses_guarded_box_stream():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z3_detection_runtime.py").read_text(encoding="utf-8")
    assert "list(detector.last_yolo_box_detections)" in source
    assert "yolo_nms_chars or yolo_chars or yolo_raw_chars" not in source
