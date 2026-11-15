import math
from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui.z3_preview_grip_geometry import (
    CORNER_HANDLE_ORDER,
    get_box_corner_point,
    get_corner_handle_centers,
)
from auto_annotation_tool.gui import z3_preview_editor_runtime


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def test_corner_handles_are_external_and_tangent_at_box_corners():
    x1, y1, x2, y2 = 100.0, 80.0, 220.0, 180.0
    radius = 12.0
    centers = get_corner_handle_centers(x1, y1, x2, y2, radius)

    expected_signs = {
        "nw": (-1, -1),
        "ne": (+1, -1),
        "sw": (-1, +1),
        "se": (+1, +1),
    }

    for name in CORNER_HANDLE_ORDER:
        corner = get_box_corner_point(name, x1, y1, x2, y2)
        center = centers[name]
        dx = center[0] - corner[0]
        dy = center[1] - corner[1]

        assert abs(abs(dx) - abs(dy)) < 1e-9
        sx, sy = expected_signs[name]
        assert math.copysign(1.0, dx) == float(sx)
        assert math.copysign(1.0, dy) == float(sy)
        assert abs(_distance(center, corner) - radius) < 1e-9

        nearest_x = min(max(center[0], x1), x2)
        nearest_y = min(max(center[1], y1), y2)
        assert abs(_distance(center, (nearest_x, nearest_y)) - radius) < 1e-9


def test_handle_hit_uses_same_external_centers_as_renderer():
    rec = {"bbox": [100.0, 80.0, 220.0, 180.0]}
    host = SimpleNamespace(
        _get_preview_selected_char_record=lambda: (0, rec),
        _char_record_bbox=lambda record: record["bbox"],
        _preview_image_to_canvas_point=lambda x, y: (float(x), float(y)),
        _get_preview_char_handle_radius=lambda: 10.0,
    )

    centers = get_corner_handle_centers(100.0, 80.0, 220.0, 180.0, 10.0)
    for name, center in centers.items():
        assert z3_preview_editor_runtime._find_preview_character_handle_hit(
            host, center[0], center[1]
        ) == (0, name)


def test_resize_drag_preserves_pointer_to_corner_offset_contract():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_events.py"
    ).read_text(encoding="utf-8-sig")

    assert "pointer_corner_offset_img_x" in source
    assert "pointer_corner_offset_img_y" in source
    assert "corner_img_x = float(img_x) - pointer_offset_x" in source
    assert "corner_img_y = float(img_y) - pointer_offset_y" in source
