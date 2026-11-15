import math
from types import SimpleNamespace

from auto_annotation_tool.gui.z3_preview_grip_geometry import (
    CORNER_HANDLE_ORDER,
    get_corner_handle_centers,
)
from auto_annotation_tool.gui.z3_preview_ui import (
    update_preview_character_drag_visual,
)


class FakeCanvas:
    def __init__(self):
        self.coords_calls = {}
        self.raise_calls = []

    def coords(self, item_id, *coords):
        self.coords_calls[item_id] = tuple(float(v) for v in coords)

    def tag_raise(self, *args):
        self.raise_calls.append(args)


def _oval_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def test_corner_handle_centers_are_outside_box_and_tangent_to_corner():
    radius = 10.0
    x1, y1, x2, y2 = 100.0, 200.0, 180.0, 300.0

    centers = get_corner_handle_centers(x1, y1, x2, y2, radius)

    assert centers["nw"][0] < x1 and centers["nw"][1] < y1
    assert centers["ne"][0] > x2 and centers["ne"][1] < y1
    assert centers["sw"][0] < x1 and centers["sw"][1] > y2
    assert centers["se"][0] > x2 and centers["se"][1] > y2

    corners = {
        "nw": (x1, y1),
        "ne": (x2, y1),
        "sw": (x1, y2),
        "se": (x2, y2),
    }
    for name in CORNER_HANDLE_ORDER:
        hx, hy = centers[name]
        cx, cy = corners[name]
        assert math.isclose(
            math.hypot(hx - cx, hy - cy),
            radius,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


def test_live_drag_keeps_corner_handle_centers_outside_box():
    canvas = FakeCanvas()
    handle_ids = [101, 102, 103, 104]
    move_handle_id = 201
    radius = 10.0

    host = SimpleNamespace(
        preview_canvas=canvas,
        _preview_render_state={
            "scale": 2.0,
            "image_left": 10.0,
            "image_top": 20.0,
        },
        _preview_char_drag_state={
            "handle_radius": radius,
            "visual_ids": {
                "box": 1,
                "selection": 2,
                "handles": handle_ids,
                "move_handle": move_handle_id,
            },
            "using_live_visual": False,
        },
        _preview_badge_runtime={},
        _get_preview_char_handle_radius=lambda: radius,
        _get_preview_char_move_handle_radius=lambda: 15.0,
    )

    bbox = [5.0, 7.0, 25.0, 37.0]
    rec = {"bbox": bbox, "character": "A"}

    assert update_preview_character_drag_visual(host, 0, rec, bbox) is True

    cx1 = 5.0 * 2.0 + 10.0
    cy1 = 7.0 * 2.0 + 20.0
    cx2 = 25.0 * 2.0 + 10.0
    cy2 = 37.0 * 2.0 + 20.0

    expected = get_corner_handle_centers(
        cx1, cy1, cx2, cy2, radius
    )

    for name, item_id in zip(CORNER_HANDLE_ORDER, handle_ids):
        actual_center = _oval_center(canvas.coords_calls[item_id])
        assert math.isclose(
            actual_center[0], expected[name][0],
            rel_tol=1e-9, abs_tol=1e-9,
        )
        assert math.isclose(
            actual_center[1], expected[name][1],
            rel_tol=1e-9, abs_tol=1e-9,
        )

    # Move handle jest celowo w środku boxa — fix dotyczy narożników.
    move_center = _oval_center(canvas.coords_calls[move_handle_id])
    assert math.isclose(move_center[0], (cx1 + cx2) / 2.0)
    assert math.isclose(move_center[1], (cy1 + cy2) / 2.0)


def test_live_drag_source_does_not_reintroduce_raw_corner_centers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_preview_ui.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def update_preview_character_drag_visual")
    end = source.index("\ndef draw_preview_plate_status_frame", start)
    body = source[start:end]

    assert "get_corner_handle_centers(" in body
    assert "zip(CORNER_HANDLE_ORDER, handle_ids)" in body
    assert "handle_points = ((cx1, cy1)" not in body
