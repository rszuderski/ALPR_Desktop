from pathlib import Path

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


def _host():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host._preview_render_state = {
        "scale": 2.0,
        "image_left": 100.0,
        "image_top": 50.0,
        "orig_w": 256.0,
        "orig_h": 128.0,
    }
    return host


def test_canvas_to_image_point_keeps_default_clamp():
    host = _host()
    assert host._preview_canvas_to_image_point(90.0, 40.0) == (0.0, 0.0)
    assert host._preview_canvas_to_image_point(700.0, 400.0) == (256.0, 128.0)


def test_canvas_to_image_point_can_be_unclamped_for_external_resize_grip():
    host = _host()
    x, y = host._preview_canvas_to_image_point(90.0, 40.0, clamp=False)
    assert x == -5.0
    assert y == -5.0


def test_resize_drag_uses_unclamped_pointer_but_move_stays_clamped():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_events.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def start_preview_character_box_drag")
    end = source.index("\ndef on_preview_canvas_secondary_press", start)
    start_drag = source[start:end]
    assert "resize_mode =" in start_drag
    assert "clamp=not resize_mode" in start_drag

    drag_start = source.index("def on_preview_canvas_drag")
    drag_end = source.index("\ndef on_preview_canvas_release", drag_start)
    drag = source[drag_start:drag_end]
    assert 'mode = str(char_drag_state.get("mode", "move") or "move").lower()' in drag
    assert 'clamp=(mode == "move")' in drag


def test_normalizer_still_clamps_actual_bbox_corner_to_image_edge():
    host = _host()
    normalized = host._normalize_preview_char_bbox(
        [-7.0, -5.0, 80.0, 70.0],
        min_size=4.0,
    )
    assert normalized[0] == 0.0
    assert normalized[1] == 0.0
    assert normalized[2] == 80.0
    assert normalized[3] == 70.0
