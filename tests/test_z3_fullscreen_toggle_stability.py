import inspect
from types import SimpleNamespace
import tkinter as tk

import pytest

from auto_annotation_tool.gui import z3_preview_ui as preview_ui
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


@pytest.fixture(scope="module")
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


def _host(root, fullscreen=True):
    canvas = tk.Canvas(root, width=600, height=360)
    canvas.winfo_width = lambda: 600
    canvas.winfo_height = lambda: 360
    host = SimpleNamespace(
        app=SimpleNamespace(palette=get_theme_palette()),
        preview_canvas=canvas,
        _preview_fullscreen_active=bool(fullscreen),
        _get_preview_legend_theme=lambda: {
            "panel_fill": "#1f2933",
            "badge_plate_outline": "#2fbf71",
            "entry_text": "#f8fafc",
        },
    )
    return host


def test_fs_toggle_is_inside_fixed_top_bar(root):
    host = _host(root, fullscreen=True)
    try:
        rect = preview_ui.draw_preview_fullscreen_toggle(
            host,
            host.preview_canvas,
            600,
            bar_height=74,
        )
        assert rect is not None
        x1, y1, x2, y2 = rect
        assert 0 <= y1 < y2 <= 74
        assert x2 <= 600
        assert host.preview_canvas.find_withtag(
            preview_ui.PREVIEW_FULLSCREEN_TOGGLE_TAG
        )
        assert host.preview_canvas.find_withtag(
            "preview_action::toggle_fullscreen"
        )
    finally:
        host.preview_canvas.destroy()


def test_fullscreen_transition_redraws_fs_before_stabilized_rerender():
    source = inspect.getsource(preview_ui.set_preview_fullscreen)
    redraw = source.rfind("draw_preview_fullscreen_toggle")
    schedule = source.rfind("_schedule_preview_stabilized_rerender")
    assert redraw >= 0
    assert schedule > redraw


def test_fullscreen_chrome_refreshes_fs_before_drawer_placement():
    source = inspect.getsource(preview_ui.apply_preview_fullscreen_chrome)
    redraw = source.find("draw_preview_fullscreen_toggle")
    drawer = source.find("_place_preview_overlay_dock")
    assert redraw >= 0
    assert drawer > redraw


def test_info_overlay_uses_shared_fs_renderer():
    source = inspect.getsource(preview_ui.draw_preview_canvas_info_overlay)
    assert "draw_preview_fullscreen_toggle(" in source
