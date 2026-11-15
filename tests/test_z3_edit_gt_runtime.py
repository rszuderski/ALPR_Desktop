from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import tkinter as tk

import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_preview_ui as ui
from auto_annotation_tool.gui import z3_preview_events as events
from auto_annotation_tool.gui import z3_gt_assist_runtime as assist
from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui import z3_preview_dock_ui as dock
from test_z3_raw_result_canvas import canvas_host


def records(text):
    return [{"character": char, "bbox": [15 + i*40, 4, 40 + i*40, 60],
             "confidence": .9, "method": "yolo", "box_source": "yolo_box", "sign_source": "yolo_symbol"}
            for i, char in enumerate(text)]


def plate(text="AXC", gt="ABC", editing=True):
    data = {"ground_truth_text": gt, "source_annotation_id": "annotation-1",
            "source_gt_hash": "gt-1", "source_gt_revision_ids": ["rev-1"],
            "plate_layout": "single_row", "plate_layout_override": "single_row",
            "plate_image_width": 240, "plate_image_height": 64,
            "characters": records(text) if editing else [],
            "raw_detection": {"contract": "gt_blind.v1", "characters": records(text)},
            "status": "needs_fix"}
    if editing:
        data["review_state"] = {"status": "in_progress"}
    return data


def quality_host(data):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.preview_metadata = {"plate": data}
    host._preview_active_pid = "plate"
    host.app = SimpleNamespace(palette={})
    for name in ("_ensure_plate_source_metadata", "_persist_preview_metadata",
                 "_refresh_preview_listbox_row", "_refresh_detection_review_controls",
                 "_on_preview_select", "_update_preview_edit_status"):
        setattr(host, name, Mock())
    return host


def test_real_validation_can_approve_matching_gt_and_preserves_raw():
    data = plate("ABC")
    host = quality_host(data)
    raw = deepcopy(data["raw_detection"])
    assert host._derive_preview_status_from_data(data, data["characters"]) == "needs_fix"
    assert host._get_review_quality_status(data) == "perfect"
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert host._derive_preview_status_from_data(data, data["characters"]) == "perfect"
    assert host._review_approval_is_current(data)
    assert data["raw_detection"] == raw


@pytest.mark.parametrize("text", ["AB", ""])
def test_approval_still_rejects_wrong_box_count(text):
    data = plate(text)
    host = quality_host(data)
    assert not review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert data["review_state"]["status"] == "in_progress"


def test_explicit_preparation_reduces_extras_before_validation():
    data = plate("ABCD")
    host = quality_host(data)
    raw = deepcopy(data["raw_detection"])
    assert not review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assist.apply_live_gt_assist(host, data, prepare=True)
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert len(data["characters"]) == 3
    assert data["raw_detection"] == raw
    assert data["review_source"] == "gt_assist_confirmed"


def test_live_assist_labels_existing_geometry_and_records_gt_provenance():
    data = plate()
    host = quality_host(data)
    raw = deepcopy(data["raw_detection"])
    boxes = [deepcopy(rec["bbox"]) for rec in data["characters"]]
    assert assist.apply_live_gt_assist(host, data)["changed"]
    assert host._characters_to_text(data["characters"], data=data) == "ABC"
    assert data["characters"][1]["sign_source"] == "gt_assisted"
    assert data["characters"][1]["gt_assist_original_character"] == "X"
    assert [rec["bbox"] for rec in data["characters"]] == boxes
    assert data["raw_detection"] == raw
    assert not assist.apply_live_gt_assist(host, data)["changed"]
    assert data["review_state"]["status"] == "in_progress"
    assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    assert data["review_source"] == "gt_assist_confirmed"


def test_gt_symbol_badge_and_serialization_keep_provenance():
    data = plate()
    host = quality_host(data)
    assist.apply_live_gt_assist(host, data)
    rec = data["characters"][1]
    assert host._get_character_sign_source_tag(rec, data=data) == "gt_assisted"
    layers = host._get_preview_source_badge_layers("yolo_box", box_source="yolo_box",
                                                  sign_source="gt_assisted", include_confidence=False)
    assert [layer["text"] for layer in layers] == ["YB", "GT"]
    saved = host._serialize_character_records(data["characters"], data=data)
    assert saved[1]["correction_source"] == "gt_assisted"
    assert saved[1]["gt_assist_original_character"] == "X"


def test_live_assist_rechecks_count_after_manual_add_and_delete():
    data = plate("XX")
    host = quality_host(data)
    assert assist.apply_live_gt_assist(host, data)["reason"] == "box_count_mismatch"
    assert host._characters_to_text(data["characters"]) == "XX"
    data["characters"].append(records("XXX")[-1])
    assert assist.apply_live_gt_assist(host, data)["changed"]
    assert host._characters_to_text(data["characters"]) == "ABC"
    data["characters"].pop()
    assert assist.apply_live_gt_assist(host, data)["reason"] == "box_count_mismatch"
    assert "ramki 2/3" in assist.get_live_gt_assist_presentation(host, data)["text"]


def test_assist_does_not_invent_geometry_for_invalid_or_missing_gt():
    for data in (plate(gt=""), plate()):
        data["characters"][0]["bbox"] = [5, 5, 5, 10]
        before = deepcopy(data["characters"])
        assert not assist.apply_live_gt_assist(quality_host(data), data)["changed"]
        assert data["characters"] == before


def test_ready_review_is_distinguished_from_errors_before_human_approval():
    data = plate("ABC")
    host = quality_host(data)
    presentation = host._get_preview_status_presentation(data, data["characters"])
    assert presentation["ready_for_approval"]
    assert presentation["canvas_text"] == "Gotowa do zatwierdzenia"
    label = host._format_plate_listbox_label("plate", data, ordinal=1)
    assert "ZATWIERDŹ" in label
    assert data["review_state"]["status"] == "in_progress"
    state = dock._get_preview_dock_row_runtime_state(host, "plate_status", {})
    assert state["status_text"] == "GOTOWA"
    assert state["tone"] == "info"


def test_assist_uses_two_row_reading_order_and_preserves_geometry():
    data = plate("XXX")
    data["plate_layout"] = data["plate_layout_override"] = "two_row"
    data["plate_layout_separator_y"] = 32
    data["characters"] = [
        {**records("X")[0], "bbox": [60, 37, 80, 60]},
        {**records("X")[0], "bbox": [100, 3, 120, 28]},
        {**records("X")[0], "bbox": [10, 3, 30, 28]},
    ]
    boxes = deepcopy([rec["bbox"] for rec in data["characters"]])
    host = quality_host(data)
    result = assist.apply_live_gt_assist(host, data)
    assert result["reason"] == "ready"
    assert [rec["character"] for rec in data["characters"]] == ["C", "B", "A"]
    assert [rec["bbox"] for rec in data["characters"]] == boxes


@pytest.mark.parametrize("editing", [False, True])
def test_zoom_keeps_visible_layer_and_correct_runtime_keys(canvas_host, monkeypatch, editing):
    host = canvas_host
    data = plate(editing=editing)
    monkeypatch.setattr(host, "_schedule_preview_metadata_save", Mock())
    host.preview_metadata = {"plate": data}
    host.preview_box_mode_var.set("AUTO")
    host._preview_active_pid = None
    host._preview_char_label_mode = False
    host._preview_char_label_active_index = None
    ui.on_preview_select(host)
    source = "FINAL" if editing else "RAW_RESULT"
    raw = deepcopy(data["raw_detection"])
    for zoom in (1.2, .8, 1.4, 1.0):
        host._preview_zoom_level = zoom
        assert events._render_preview_zoom_frame(host)
        assert len(host._preview_char_runtime) == 3
        assert set(host._preview_char_runtime) == {f"{source}:{i}" for i in range(3)}
        for entry in host._preview_char_runtime.values():
            assert host.preview_canvas.type(entry["box_id"]) == "rectangle"
    assert data["raw_detection"] == raw


def test_s_opens_review_selects_visible_box_after_zoom_and_can_delete(canvas_host, monkeypatch):
    host = canvas_host
    data = plate(editing=False)
    host.preview_metadata = {"plate": data}
    host.preview_box_mode_var.set("AUTO")
    host._preview_active_pid = None
    host._preview_char_label_mode = False
    host._preview_char_label_active_index = None
    monkeypatch.setattr(host, "_schedule_preview_metadata_save", Mock())
    monkeypatch.setattr(host, "_schedule_preview_info_refresh", Mock())
    monkeypatch.setattr(host, "_refresh_preview_listbox_row", Mock())
    monkeypatch.setattr(host, "_refresh_preview_live_metadata_ui", Mock())
    ui.on_preview_select(host)
    host._preview_zoom_level = 1.25
    assert events._render_preview_zoom_frame(host)
    raw = deepcopy(data["raw_detection"])
    x, y = host._preview_image_to_canvas_point(27, 32)
    event = SimpleNamespace(keysym="s", char="s", x=x, y=y, state=0, widget=host.preview_canvas)
    assert events.on_preview_canvas_keypress(host, event) == "break"
    assert host._preview_char_selected_index == 0
    assert data["review_state"]["status"] == "in_progress"
    selected = host._preview_char_runtime["FINAL:0"]
    assert host.preview_canvas.type(selected["selection_id"]) == "rectangle"
    assert host._characters_to_text(data["characters"]) == "ABC"
    assert host._delete_selected_preview_char_box() == "break"
    assert len(data["characters"]) == 2
    assert len(host._preview_char_runtime) == 2
    assert data["raw_detection"] == raw


def test_qe_after_edit_and_zoom_shows_next_raw_boxes_and_s_still_works(canvas_host, monkeypatch):
    host = canvas_host
    directory = Path(host.preview_dir_var.get())
    (directory / "images/next.jpg").write_bytes((directory / "images/plate.jpg").read_bytes())
    first, second = plate(), plate("YY", "ZZ", editing=False)
    host.preview_metadata = {"plate": first, "next": second}
    host._preview_base_plate_ids = host.preview_plate_ids = host._listbox_pid_by_index = ["plate", "next"]
    host.plates_listbox.delete(0, "end")
    host.plates_listbox.insert("end", "plate", "next")
    host.plates_listbox.selection_set(0)
    host.preview_box_mode_var.set("AUTO")
    host._preview_active_pid = None
    host._preview_char_label_mode = False
    host._preview_char_label_active_index = None
    monkeypatch.setattr(host, "_schedule_preview_metadata_save", Mock())
    monkeypatch.setattr(host, "_schedule_preview_info_refresh", Mock())
    monkeypatch.setattr(host, "_refresh_preview_listbox_row", Mock())
    ui.on_preview_select(host)
    host._preview_zoom_level = 1.2
    assert events._render_preview_zoom_frame(host)
    assert events.on_preview_canvas_keypress(host, SimpleNamespace(keysym="e", char="e", state=0)) == "break"
    done = tk.BooleanVar(host.frame, False)
    host.frame.after(450, lambda: done.set(True))
    host.frame.wait_variable(done)
    assert host._preview_active_pid == "next"
    assert set(host._preview_char_runtime) == {"RAW_RESULT:0", "RAW_RESULT:1"}
    x, y = host._preview_image_to_canvas_point(27, 32)
    assert events.on_preview_canvas_keypress(host, SimpleNamespace(keysym="s", char="s", state=0, x=x, y=y)) == "break"
    assert host._preview_char_selected_index == 0
    assert set(host._preview_char_runtime) == {"FINAL:0", "FINAL:1"}
    assert host._characters_to_text(second["characters"]) == "ZZ"


def test_stale_zoom_frame_cannot_replace_next_plate(canvas_host):
    host = canvas_host
    host._preview_render_state = {"plate_id": "plate"}
    host._preview_active_pid = "next"
    before = host.preview_canvas.find_all()
    assert not events._render_preview_zoom_frame(host)
    assert host.preview_canvas.find_all() == before
