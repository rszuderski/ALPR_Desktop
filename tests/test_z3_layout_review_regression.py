"""Layout changes must preserve the visible prediction/review across undo."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk

import pytest

from auto_annotation_tool.gui import z3_detection_runtime as detection
from auto_annotation_tool.gui import z3_plate_layout_runtime as layouts
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui import z3_preview_events as events
from test_z3_edit_gt_runtime import plate, quality_host


@pytest.fixture
def host():
    data = plate("AT37", "AT37", editing=False)
    obj = quality_host(data)
    obj.preview_box_mode_var = SimpleNamespace(get=lambda: obj.mode, set=lambda value: setattr(obj, "mode", value))
    obj.mode = "GT_RESULT"
    obj._get_preview_box_mode_label = lambda value: value
    for name in ("_schedule_preview_metadata_save", "_schedule_preview_info_refresh",
                 "_refresh_preview_layout_override_ui_light", "_refresh_preview_live_metadata_ui",
                 "_redraw_preview_character_overlays_light"):
        setattr(obj, name, Mock())
    return obj


@pytest.mark.parametrize("override", ["single_row", "two_row", ""])
def test_layout_opens_visible_prediction_before_marking_review(host, override):
    data = host.preview_metadata["plate"]
    raw = deepcopy(data["raw_detection"])
    before = sorted(tuple(rec["bbox"]) for rec in host._get_preview_box_records(data)[0])
    layouts._apply_preview_plate_layout_override(host, override)
    assert sorted(tuple(rec["bbox"]) for rec in data["characters"]) == before
    assert len(data["characters"]) == 4
    assert host._get_preview_box_records(data)[1] == "FINAL"
    assert data["review_state"]["status"] == "in_progress"
    assert data["raw_detection"] == raw
    assert len(host._get_preview_history_stack("undo")) == 1


def test_layout_then_undo_redo_preserves_prediction_and_corrected_geometry(host):
    data = host.preview_metadata["plate"]
    prediction = deepcopy(data)
    layouts._apply_preview_plate_layout_override(host, "two_row")
    host._undo_preview_edit()
    restored = host.preview_metadata["plate"]
    assert len(host._get_preview_box_records(restored)[0]) == 4
    assert not review.get_review_state_status(restored)
    assert restored["raw_detection"] == prediction["raw_detection"]
    host._redo_preview_edit()
    restored = host.preview_metadata["plate"]
    assert len(restored["characters"]) == 4
    assert restored["plate_layout_override"] == "two_row"
    restored["characters"][0]["bbox"][0] += 2
    boxes = sorted(tuple(rec["bbox"]) for rec in restored["characters"])
    for layout in ("single_row", "two_row", "single_row"):
        layouts._apply_preview_plate_layout_override(host, layout)
        assert sorted(tuple(rec["bbox"]) for rec in restored["characters"]) == boxes


def test_layout_does_not_resurrect_intentionally_deleted_boxes(host):
    data = host.preview_metadata["plate"]
    data["review_state"] = {"status": "in_progress"}
    for layout in ("two_row", "single_row"):
        layouts._apply_preview_plate_layout_override(host, layout)
        assert data["characters"] == []
    host._undo_preview_edit()
    assert host._get_preview_box_records(host.preview_metadata["plate"])[0] == []


def test_explicit_empty_review_recovery_is_undoable_and_gt_limited(host):
    data = host.preview_metadata["plate"]
    data["review_state"] = {"status": "in_progress"}
    data["raw_detection"]["characters"].append({"character": "X", "bbox": [210, 2, 230, 60]})
    raw = deepcopy(data["raw_detection"])
    result = review.open_or_restore_review(host)
    assert result["ok"] and len(data["characters"]) == 4
    assert data["raw_detection"] == raw
    assert not data["gold_state"]["approved"]
    host._undo_preview_edit()
    restored = host.preview_metadata["plate"]
    assert restored["characters"] == []
    assert host._get_preview_box_records(restored)[0] == []
    assert restored["raw_detection"] == raw


def test_recovery_does_not_overwrite_existing_correction(host, monkeypatch):
    data = host.preview_metadata["plate"]
    data["characters"] = [{"character": "M", "bbox": [2, 3, 20, 40]}]
    data["review_state"] = {"status": "in_progress"}
    before = deepcopy(data)
    monkeypatch.setattr(review.messagebox, "showinfo", Mock())
    assert not review.open_or_restore_review(host)["ok"]
    assert data == before


@pytest.mark.parametrize("busy", [False, True])
def test_empty_review_exposes_recovery_button(host, busy):
    host.preview_metadata["plate"]["review_state"] = {"status": "in_progress"}
    host.is_processing = busy
    host.btn_start_review_from_raw = Mock()
    detection.refresh_detection_review_controls(host)
    assert host.btn_start_review_from_raw.config.call_args.kwargs == {
        "text": "Przywróć ramki", "state": tk.DISABLED if busy else tk.NORMAL}


def test_layout_is_ignored_during_detection_or_bulk_review(host):
    before = deepcopy(host.preview_metadata)
    host._preview_review_batch_running = True
    layouts._apply_preview_plate_layout_override(host, "two_row")
    assert host.preview_metadata == before


def test_separator_edit_materializes_prediction_and_keeps_raw(host):
    data = host.preview_metadata["plate"]
    raw = deepcopy(data["raw_detection"])
    host._preview_layout_separator_drag_state = {
        "dirty": True,
        "preview_separator": {"x1": 0, "x2": 240, "y1": 62, "y2": 62},
    }
    assert events.on_preview_canvas_release(host, SimpleNamespace(x=100, y=62)) == "break"
    assert len(data["characters"]) == 4
    assert data["raw_detection"] == raw
    assert host._get_preview_box_records(data)[1] == "FINAL"
