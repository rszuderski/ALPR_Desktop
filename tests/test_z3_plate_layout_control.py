import copy
from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_preview_layout_control as control
from auto_annotation_tool.gui import z3_preview_events as events
from auto_annotation_tool.gui import z3_plate_layout_runtime as layouts


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def host(root):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.app = SimpleNamespace(palette=get_theme_palette())
    host.preview_canvas = tk.Canvas(root, width=500, height=300)
    host.preview_canvas.winfo_width = lambda: 500
    host.preview_canvas.winfo_height = lambda: 300
    host._preview_active_pid = "plate"
    host._preview_render_state = {"image_left": 60, "image_top": 90, "image_right": 420,
                                  "image_bottom": 240, "orig_w": 120, "orig_h": 100}
    host.preview_metadata = {"plate": {"plate_layout": "single_row", "plate_image_width": 120,
        "plate_image_height": 100, "status": "needs_fix", "characters": [
            {"character": "A", "bbox": [10, 10, 30, 35]},
            {"character": "C", "bbox": [10, 65, 30, 90]},
            {"character": "B", "bbox": [70, 10, 90, 35]},
            {"character": "D", "bbox": [70, 65, 90, 90]},
        ]}}
    host._get_preview_active_data = lambda **kwargs: host.preview_metadata["plate"]
    host._get_preview_active_character_records = lambda **kwargs: host.preview_metadata["plate"]["characters"]
    host._derive_preview_status_from_data = Mock(return_value="needs_fix")
    host._push_preview_history_snapshot = Mock()
    host._schedule_preview_metadata_save = Mock()
    host._persist_preview_metadata = Mock(side_effect=AssertionError("Layout must not synchronously save the whole project"))
    host._schedule_preview_info_refresh = Mock()
    host._focus_preview_canvas = Mock()
    host._refresh_preview_layout_override_ui_light = lambda **kwargs: host._draw_preview_plate_status_frame()
    yield host
    control.hide_layout_tip(host)
    host.preview_canvas.destroy()


@pytest.mark.parametrize("layout,override,expected_rows,next_override", [
    ("single_row", None, 1, "two_row"), ("two_row", None, 2, "single_row"),
    ("two_row_candidate", None, 2, "single_row"), ("two_row", "single_row", 1, "two_row"),
    ("single_row", "two_row", 2, "single_row"),
])
def test_icon_reflects_effective_layout_and_first_click_changes_it(host, layout, override, expected_rows, next_override):
    data = host._get_preview_active_data()
    data["plate_layout"] = layout
    if override:
        data["plate_layout_override"] = override
    before = {rec["character"]: list(rec["bbox"]) for rec in data["characters"]}
    assert host._draw_preview_plate_status_frame()
    assert host.preview_canvas.find_withtag(f"preview_plate_layout_rows::{expected_rows}")
    assert control.toggle_plate_rows(host) == "break"
    assert data["plate_layout_override"] == next_override
    assert data["layout_override_source"] == "frame_toggle"
    assert {rec["character"]: rec["bbox"] for rec in data["characters"]} == before
    assert host.preview_canvas.find_withtag(f"preview_plate_layout_rows::{3 - expected_rows}")
    if next_override == "two_row":
        assert data["layout_separator"]["y1"] == data["layout_separator"]["y2"] == 50
        assert [rec["reading_row"] for rec in data["characters"]] == [1, 1, 2, 2]
    else:
        assert "layout_separator" not in data
        assert all(rec["reading_row"] == 1 for rec in data["characters"])
    host._persist_preview_metadata.assert_not_called()
    host._schedule_preview_metadata_save.assert_called_once_with(delay_ms=450)
    host._push_preview_history_snapshot.assert_called_once()


def test_selection_stays_on_same_character_after_reordering(host):
    host._preview_char_selected_index = 2  # B, before sorting into two rows.
    control.toggle_plate_rows(host)
    assert host._preview_char_selected_index == 1
    assert host._get_preview_active_data()["characters"][host._preview_char_selected_index]["character"] == "B"


def test_frame_control_stays_inside_canvas_and_does_not_accumulate_callbacks(host):
    canvas = host.preview_canvas
    data = host._get_preview_active_data()
    control.draw_plate_layout_control(host, data, right=490, top=-40)
    initial = len(canvas._tclCommands)
    for _ in range(15):
        control.draw_plate_layout_control(host, data, right=700, top=40)
    assert len(canvas._tclCommands) == initial
    x1, y1, x2, y2 = canvas.bbox(control.CONTROL_TAG)
    assert 0 <= x1 < x2 <= 500 and 0 <= y1 < y2 <= 300
    for item in canvas.find_withtag(control.CONTROL_TAG):
        assert "preview_action::toggle_plate_rows" in canvas.gettags(item)


def test_header_status_refresh_keeps_hover_control_and_tooltip_timer(host):
    canvas = host.preview_canvas
    host._draw_preview_plate_status_frame()
    items = canvas.find_withtag(control.CONTROL_TAG)
    binding_ids = list(host._preview_layout_control_bindings)
    pending = canvas.after(5000, lambda: None)
    host._preview_layout_tip_after_id = pending
    canvas.create_text(300, 40, text="hint", tags=(control.TIP_TAG,))
    canvas.delete("preview_overlay")
    canvas.delete("preview_overlay_action")
    host._draw_preview_plate_status_frame()
    assert canvas.find_withtag(control.CONTROL_TAG) == items
    assert host._preview_layout_control_bindings == binding_ids
    assert host._preview_layout_tip_after_id == pending
    assert canvas.find_withtag(control.TIP_TAG)


def test_action_click_does_not_pan_or_select_a_box_and_ppm_opens_layout_menu(host):
    host._extract_preview_action_from_current_item = lambda: control.ACTION
    host._find_preview_character_box_hit = Mock(side_effect=AssertionError("Icon is not a character box"))
    host._cycle_preview_plate_layout_override = Mock(return_value="break")
    event = SimpleNamespace(x=440, y=68, x_root=440, y_root=68)
    assert events.on_preview_canvas_press(host, event) == "break"
    assert host._get_preview_active_data()["plate_layout_override"] == "two_row"
    assert events.on_preview_canvas_secondary_press(host, event) == "break"
    host._cycle_preview_plate_layout_override.assert_called_once_with(event)


def test_auto_option_removes_override_and_restores_detected_rows(host):
    control.toggle_plate_rows(host)
    control.toggle_plate_rows(host)
    assert host._get_preview_active_data()["plate_layout_override"] == "single_row"
    layouts._apply_preview_plate_layout_override(host, "", source="menu")
    data = host._get_preview_active_data()
    assert "plate_layout_override" not in data
    assert data["plate_layout"] == "two_row"
    assert host.preview_canvas.find_withtag("preview_plate_layout_rows::2")


def test_layout_before_detection_still_has_frame_control(host):
    data = host._get_preview_active_data()
    data.update(status="unknown", characters=[], plate_layout="unknown")
    host._derive_preview_status_from_data.return_value = "unknown"
    assert host._draw_preview_plate_status_frame()
    assert host.preview_canvas.find_withtag(control.CONTROL_TAG)
