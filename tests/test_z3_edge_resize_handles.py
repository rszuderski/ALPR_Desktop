from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk

import pytest

from auto_annotation_tool.gui.z3_preview_grip_geometry import (
    CORNER_HANDLE_ORDER, EDGE_HANDLE_ORDER, get_edge_handle_rects, resize_box_edge,
)
from auto_annotation_tool.gui import z3_preview_ui as ui
from auto_annotation_tool.gui import z3_preview_events as events
from test_z3_raw_result_canvas import canvas_host


def center(rect):
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


@pytest.mark.parametrize("radius", [7.0, 10.0, 13.0])
def test_side_rectangles_and_hover_outlines_are_outside_and_centered(radius):
    x1, y1, x2, y2 = 100.0, 80.0, 220.0, 180.0
    rects = get_edge_handle_rects(x1, y1, x2, y2, radius)
    assert rects["n"][3] + 1.5 <= y1
    assert rects["s"][1] - 1.5 >= y2
    assert rects["w"][2] + 1.5 <= x1
    assert rects["e"][0] - 1.5 >= x2
    assert center(rects["n"])[0] == center(rects["s"])[0] == (x1 + x2) / 2
    assert center(rects["w"])[1] == center(rects["e"])[1] == (y1 + y2) / 2


@pytest.mark.parametrize("side,axis,expected", [("w", 0, 12), ("n", 1, 12), ("e", 2, 80), ("s", 3, 80)])
def test_resize_changes_exactly_one_edge_even_with_diagonal_pointer_motion(side, axis, expected):
    bbox = [20, 20, 60, 60]
    result = resize_box_edge(bbox, side, 12 if side in ("w", "n") else 80,
                             12 if side in ("w", "n") else 80)
    assert result[axis] == expected
    assert [v for i, v in enumerate(result) if i != axis] == [v for i, v in enumerate(bbox) if i != axis]


@pytest.mark.parametrize("side,expected", [("w", [56, 20, 60, 60]), ("n", [20, 56, 60, 60]),
                                          ("e", [20, 20, 24, 60]), ("s", [20, 20, 60, 24])])
def test_crossing_opposite_side_stops_at_minimum_size(side, expected):
    value = 100 if side in ("w", "n") else -100
    assert resize_box_edge([20, 20, 60, 60], side, value, value) == expected


def test_real_canvas_draws_four_circles_and_four_rectangles_with_matching_hit_regions(canvas_host):
    host = canvas_host
    rec = {"character": "A", "bbox": [20., 10., 80., 50.], "confidence": .8,
           "method": "manual", "box_source": "manual_box"}
    host.preview_metadata = {"plate": {"characters": [rec], "review_state": {"status": "in_progress"}}}
    host._preview_active_pid = "plate"
    host._preview_char_selected_index = 0
    host._preview_render_state = {"scale": 2., "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    cx1, cy1 = host._preview_image_to_canvas_point(20., 10.)
    cx2, cy2 = host._preview_image_to_canvas_point(80., 50.)
    ids, move = ui._draw_preview_character_edit_grips(host, host.preview_canvas, tags=("test_grips",),
                                                     cx1=cx1, cy1=cy1, cx2=cx2, cy2=cy2, selection_color="#ff8a00")
    assert len(ids) == 8
    assert [host.preview_canvas.type(item) for item in ids] == ["oval"] * 4 + ["rectangle"] * 4
    rects = get_edge_handle_rects(cx1, cy1, cx2, cy2, host._get_preview_char_handle_radius())
    for side, item in zip(EDGE_HANDLE_ORDER, ids[4:]):
        assert host.preview_canvas.coords(item) == list(rects[side])
        hit = host._find_preview_character_grip_hit(*center(rects[side]))
        assert hit == {"index": 0, "kind": "edge", "handle": side, "key": f"edge:{side}"}
    host._preview_char_hover_grip = "edge:e"
    assert ui._style_preview_character_edit_grips_fast(host, host.preview_canvas,
                                                      {"handle_ids": ids, "move_handle_id": move}, selection_color="#ff8a00")
    assert host.preview_canvas.itemcget(ids[5], "width") == "3.0"
    host.preview_canvas.delete("test_grips")


@pytest.mark.parametrize("side,axis,delta", [("w", 0, -7.), ("n", 1, -5.), ("e", 2, 7.), ("s", 3, 5.)])
@pytest.mark.parametrize("scale", [0.5, 2., 5.])
def test_actual_drag_preserves_other_three_edges_and_pointer_offset(canvas_host, monkeypatch, side, axis, delta, scale):
    host = canvas_host
    bbox = [30., 15., 60., 45.]
    rec = {"character": "A", "bbox": list(bbox), "confidence": .8, "method": "yolo",
           "box_source": "yolo_box", "sign_source": "yolo_symbol"}
    raw = {"contract": "gt_blind.v1", "characters": [deepcopy(rec)]}
    data = {"characters": [rec], "plate_layout": "single_row", "plate_layout_override": "single_row",
            "review_state": {"status": "in_progress"}, "raw_detection": raw}
    raw_before = deepcopy(raw)
    host.preview_metadata = {"plate": data}
    host._preview_active_pid = "plate"
    host._preview_char_selected_index = 0
    host._preview_char_geometry_inherit_down = False
    host._preview_render_state = {"scale": scale, "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    monkeypatch.setattr(host, "_bind_preview_char_drag_session", Mock())
    monkeypatch.setattr(host, "_update_preview_character_drag_visual", Mock(return_value=True))
    x1, y1 = host._preview_image_to_canvas_point(*bbox[:2])
    x2, y2 = host._preview_image_to_canvas_point(*bbox[2:])
    x, y = center(get_edge_handle_rects(x1, y1, x2, y2, host._get_preview_char_handle_radius())[side])
    assert events.start_preview_character_box_drag(host, 0, "resize", SimpleNamespace(x=x, y=y), handle_name=side)
    # A zero-motion event at the center of the external grab must not jump.
    assert events.on_preview_canvas_drag(host, SimpleNamespace(x=x, y=y)) == "break"
    assert rec["bbox"] == bbox
    # Both pointer coordinates change; only the grabbed edge may move.
    dx = delta * scale if side in ("w", "e") else 12 * scale
    dy = delta * scale if side in ("n", "s") else 12 * scale
    events.on_preview_canvas_drag(host, SimpleNamespace(x=x + dx, y=y + dy))
    expected = list(bbox)
    expected[axis] += delta
    assert rec["bbox"] == pytest.approx(expected)
    assert rec["box_source"] == "manual_box"
    assert data["raw_detection"] == raw_before
    host._preview_char_drag_state = None


def test_live_drag_repositions_edge_rectangles_using_new_bbox(canvas_host):
    host = canvas_host
    host._preview_render_state = {"scale": 2., "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    ids, move = ui._draw_preview_character_edit_grips(host, host.preview_canvas, tags=("test_live_grips",),
                                                     cx1=60., cy1=45., cx2=140., cy2=125., selection_color="#ff8a00")
    host._preview_char_drag_state = {"handle_radius": host._get_preview_char_handle_radius(), "using_live_visual": False,
                                    "visual_ids": {"box": None, "selection": None, "handles": ids, "move_handle": move}}
    bbox = [20., 10., 90., 50.]
    assert ui.update_preview_character_drag_visual(host, 0, {"character": "A", "bbox": bbox}, bbox)
    rects = get_edge_handle_rects(60., 45., 200., 125., host._get_preview_char_handle_radius())
    for side, item in zip(EDGE_HANDLE_ORDER, ids[4:]):
        assert host.preview_canvas.coords(item) == list(rects[side])
    host._preview_char_drag_state = None
    host.preview_canvas.delete("test_live_grips")


@pytest.mark.parametrize("side,cursor", [("w", "sb_h_double_arrow"), ("e", "sb_h_double_arrow"),
                                       ("n", "sb_v_double_arrow"), ("s", "sb_v_double_arrow")])
def test_press_on_side_handle_starts_orthogonal_resize(canvas_host, monkeypatch, side, cursor):
    host = canvas_host
    rec = {"character": "A", "bbox": [30., 15., 60., 45.], "confidence": .8, "method": "manual", "box_source": "manual_box"}
    host.preview_metadata = {"plate": {"characters": [rec], "review_state": {"status": "in_progress"}, "plate_layout": "single_row"}}
    host._preview_active_pid = "plate"
    host._preview_char_selected_index = 0
    host._preview_char_edit_mode = True
    host._preview_char_label_mode = False
    host._preview_char_label_active_index = None
    host._preview_char_add_state = None
    host._preview_char_add_click_armed = False
    host._preview_char_add_modifier_down = False
    host._preview_char_add_mode = False
    host._preview_char_geometry_inherit_down = False
    host._preview_render_state = {"scale": 2., "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    host.preview_canvas.delete("all")
    monkeypatch.setattr(host, "_bind_preview_char_drag_session", Mock())
    start = Mock(wraps=host._start_preview_character_box_drag)
    monkeypatch.setattr(host, "_start_preview_character_box_drag", start)
    rects = get_edge_handle_rects(80., 55., 140., 115., host._get_preview_char_handle_radius())
    x, y = center(rects[side])
    assert events.on_preview_canvas_press(host, SimpleNamespace(x=x, y=y)) == "break"
    assert start.call_args.args[1] == "resize"
    assert start.call_args.kwargs == {"handle_name": side, "cursor": cursor}
    assert host._preview_char_drag_state["handle"] == side
    host._preview_char_drag_state = None
    host._preview_char_edit_mode = False


@pytest.mark.parametrize("side,axis,target", [("w", 0, -100.), ("n", 1, -100.), ("e", 2, 1000.), ("s", 3, 1000.)])
def test_edge_drag_reaches_image_boundary_without_moving_other_edges(canvas_host, monkeypatch, side, axis, target):
    host = canvas_host
    bbox = [30., 15., 60., 45.]
    rec = {"character": "A", "bbox": list(bbox), "confidence": .8, "method": "manual", "box_source": "manual_box"}
    host.preview_metadata = {"plate": {"characters": [rec], "review_state": {"status": "in_progress"}, "plate_layout": "single_row"}}
    host._preview_active_pid = "plate"
    host._preview_char_geometry_inherit_down = False
    host._preview_render_state = {"scale": 2., "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    monkeypatch.setattr(host, "_bind_preview_char_drag_session", Mock())
    monkeypatch.setattr(host, "_update_preview_character_drag_visual", Mock(return_value=True))
    rects = get_edge_handle_rects(80., 55., 140., 115., host._get_preview_char_handle_radius())
    x, y = center(rects[side])
    assert events.start_preview_character_box_drag(host, 0, "resize", SimpleNamespace(x=x, y=y), handle_name=side)
    dx = (target - bbox[axis]) * 2 if side in ("w", "e") else 0
    dy = (target - bbox[axis]) * 2 if side in ("n", "s") else 0
    events.on_preview_canvas_drag(host, SimpleNamespace(x=x + dx, y=y + dy))
    expected = list(bbox)
    expected[axis] = {"w": 0., "n": 0., "e": 240., "s": 64.}[side]
    assert rec["bbox"] == expected
    host._preview_char_drag_state = None


def test_side_drag_release_uses_existing_save_and_undo_contract(canvas_host, monkeypatch):
    host = canvas_host
    bbox = [30., 15., 60., 45.]
    rec = {"character": "A", "bbox": list(bbox), "confidence": .8, "method": "yolo",
           "box_source": "yolo_box", "sign_source": "yolo_symbol"}
    raw = {"contract": "gt_blind.v1", "characters": [deepcopy(rec)]}
    host.preview_metadata = {"plate": {"characters": [rec], "review_state": {"status": "in_progress"},
                                        "plate_layout": "single_row", "raw_detection": raw}}
    host._preview_active_pid = "plate"
    host._preview_char_geometry_inherit_down = False
    host._preview_history_undo = {}
    host._preview_history_redo = {}
    host._preview_render_state = {"scale": 2., "image_left": 20., "image_top": 25., "orig_w": 240., "orig_h": 64.}
    monkeypatch.setattr(host, "_bind_preview_char_drag_session", Mock())
    monkeypatch.setattr(host, "_unbind_preview_char_drag_session", Mock())
    monkeypatch.setattr(host, "_update_preview_character_drag_visual", Mock(return_value=True))
    save = Mock()
    monkeypatch.setattr(host, "_persist_active_preview_characters", save)
    x, y = center(get_edge_handle_rects(80., 55., 140., 115., host._get_preview_char_handle_radius())["e"])
    assert events.start_preview_character_box_drag(host, 0, "resize", SimpleNamespace(x=x, y=y), handle_name="e")
    assert ui.resolve_preview_canvas_cursor(host) == "sb_h_double_arrow"
    events.on_preview_canvas_drag(host, SimpleNamespace(x=x + 20, y=y + 30))
    events.on_preview_canvas_release(host, SimpleNamespace(x=x + 20, y=y + 30))
    assert rec["bbox"] == [30., 15., 70., 45.]
    save.assert_called_once()
    assert save.call_args.kwargs["selected_record"] is rec
    assert host._preview_history_undo["plate"][-1]["characters"][0]["bbox"] == bbox
    assert host._preview_char_drag_state is None
    assert raw["characters"][0]["bbox"] == bbox
