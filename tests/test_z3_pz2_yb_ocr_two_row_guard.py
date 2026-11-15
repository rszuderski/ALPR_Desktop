from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui.z3_detection_runtime import (
    guard_yolo_box_candidates_for_plate_layout,
)


def _host(two_row=True):
    return SimpleNamespace(
        _should_preview_use_two_row_layers=lambda _data: bool(two_row),
        _char_record_bbox=lambda rec: rec.get("bbox"),
        _normalize_preview_layout_separator=lambda sep, **_kwargs: dict(sep) if isinstance(sep, dict) else None,
        _ensure_preview_layout_separator=lambda *_args, **_kwargs: None,
    )


def _data():
    return {
        "plate_layout": "two_row",
        "plate_layout_override": "two_row",
        "layout_separator": {
            "x1": 0.0,
            "y1": 50.0,
            "x2": 120.0,
            "y2": 50.0,
            "source": "manual",
        },
    }


def test_row_guard_rejects_cross_row_yb_without_mutating_nms_records():
    host = _host()
    records = [
        {"id": "top", "bbox": [10, 8, 30, 45]},
        {"id": "cross", "bbox": [40, 12, 65, 88]},
        {"id": "bottom", "bbox": [75, 55, 95, 92]},
    ]
    before = [dict(rec, bbox=list(rec["bbox"])) for rec in records]

    accepted, details = guard_yolo_box_candidates_for_plate_layout(
        host,
        _data(),
        records,
        image_shape=(100, 120, 3),
    )

    assert records == before
    assert [rec["id"] for rec in accepted] == ["top", "bottom"]
    assert details["row_guard_enabled"] is True
    assert details["row_guard_input_count"] == 3
    assert details["row_guard_accepted_count"] == 2
    assert details["row_guard_rejected_count"] == 1
    assert details["row_guard_separator_cross_count"] == 1


def test_row_guard_handles_sloped_separator_across_whole_box_width():
    host = _host()
    data = _data()
    data["layout_separator"] = {
        "x1": 0.0,
        "y1": 40.0,
        "x2": 120.0,
        "y2": 60.0,
        "source": "manual",
    }
    records = [
        {"id": "safe_top", "bbox": [10, 5, 40, 37]},
        {"id": "edge_cross", "bbox": [60, 42, 110, 58]},
    ]

    accepted, details = guard_yolo_box_candidates_for_plate_layout(
        host,
        data,
        records,
        image_shape=(100, 120, 3),
    )

    assert [rec["id"] for rec in accepted] == ["safe_top"]
    assert details["row_guard_separator_cross_count"] == 1


def test_row_guard_is_noop_for_single_row_layout():
    records = [{"id": "tall", "bbox": [10, 5, 30, 95]}]
    accepted, details = guard_yolo_box_candidates_for_plate_layout(
        _host(two_row=False),
        {"plate_layout": "single_row"},
        records,
        image_shape=(100, 120, 3),
    )

    assert accepted == records
    assert details["row_guard_enabled"] is False
    assert details["row_guard_rejected_count"] == 0


def test_detection_runtime_applies_2r_guard_before_yolo_ocr_crop():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_detection_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("ocr_chars = self._sort_character_records_by_x")
    end = source.index("c_clean = self._serialize_character_records", start)
    block = source[start:end]

    guard_call = "guard_yolo_box_candidates_for_plate_layout("
    ocr_crop_call = "detector._detect_with_yolo_boxes_and_ocr("
    assert guard_call in block
    assert ocr_crop_call in block
    assert block.index(guard_call) < block.index(ocr_crop_call)
    assert "yolo_box_backend_chars, row_guard_details =" in block


def test_diagnostic_nms_stays_serialized_from_detector():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_detection_runtime.py"
    ).read_text(encoding="utf-8-sig")

    assert 'getattr(detector, "last_yolo_nms_detections", [])' in source
    assert 'local_meta[pid]["yolo_nms_detections"] = yolo_nms_clean' in source
