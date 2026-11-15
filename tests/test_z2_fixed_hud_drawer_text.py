from types import SimpleNamespace
import inspect

from auto_annotation_tool.gui.zoomable_canvas import ZoomableCanvas
from auto_annotation_tool.gui import z2_preview_editor
from auto_annotation_tool.gui import z2_overlay_dock_ui


def test_buffered_pan_keeps_viewport_fixed_hud_stationary():
    moves = []
    state = {"configured": False}

    def move(tag, dx, dy):
        moves.append((tag, float(dx), float(dy)))

    def configure(**kwargs):
        state["configured"] = True

    fake = SimpleNamespace(
        VIEWPORT_FIXED_TAG="preview_viewport_fixed",
        _can_reuse_buffered_pan=lambda: True,
        canvasx=lambda _x: 10.0 if state["configured"] else 2.0,
        canvasy=lambda _y: 7.0 if state["configured"] else 1.0,
        move=move,
        configure=configure,
        _render_region={"draw_x": 0.0, "draw_y": 0.0},
        original_image=SimpleNamespace(width=100, height=50),
        zoom_level=2.0,
        pan_data={"x": -20.0, "y": -10.0},
        _pan_buffered_move_active=False,
    )

    result = ZoomableCanvas._apply_buffered_pan_move(fake, 4.0, 3.0)

    assert result is True
    assert moves[0] == ("all", 4.0, 3.0)
    assert moves[1] == ("preview_viewport_fixed", 4.0, 3.0)


def test_plate_combo_is_tagged_as_viewport_fixed():
    source = inspect.getsource(z2_preview_editor._draw_preview_plate_combo_overlay)
    assert "preview_plate_combo_overlay" in source
    assert "VIEWPORT_FIXED_TAG" in source
    assert "addtag_withtag" in source


def test_drawer_gate_text_is_forced_white():
    source = inspect.getsource(z2_overlay_dock_ui.render_preview_overlay_dock)
    assert 'drawer_text_fg = "#ffffff"' in source
    assert 'status_text = drawer_text_fg' in source
    assert 'have_text = drawer_text_fg' in source
    assert 'missing_text = drawer_text_fg' in source
    assert 'quality_text = drawer_text_fg' in source
    assert 'approval_text = drawer_text_fg' in source
