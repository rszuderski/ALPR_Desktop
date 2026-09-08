from types import SimpleNamespace
import tkinter as tk
import tkinter.font as tkfont
from unittest.mock import patch

import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui.web_slim_scrollbar import WebSlimScrollbar


@pytest.fixture(scope="module")
def root():
    root = tk.Tk()
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    yield root
    root.destroy()
    assert not errors


@pytest.fixture
def compass(root):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    host.frame = root
    host.app = SimpleNamespace(palette={})
    host._get_preview_legend_context = lambda: {"filename": "ABC1234.jpg", "image_text": "Tablica: 1/1245",
                                               "plate_text": "Boxy: 7", "vehicle_text": "Tryb: podgląd"}
    host._preview_render_state = {"image": True}
    host.preview_canvas_host = tk.Frame(root)
    host.preview_canvas_host.place(x=0, y=0, width=960, height=600)
    host.preview_hint_frame = tk.Frame(host.preview_canvas_host)
    host.preview_hint_frame.grid_columnconfigure(0, weight=1)
    host.preview_hint_frame.grid_rowconfigure(0, weight=1)
    host.preview_controls_canvas = tk.Canvas(
        host.preview_hint_frame, width=32, height=32, bd=0, highlightthickness=0,
    )
    host.preview_controls_canvas.grid(row=0, column=0, sticky="nsew")
    host.preview_controls_vbar = WebSlimScrollbar(
        host.preview_hint_frame, orient=tk.VERTICAL,
        command=host.preview_controls_canvas.yview, auto_hide=False, thickness=6,
    )
    host.preview_controls_canvas.configure(yscrollcommand=host.preview_controls_vbar.set)
    host.preview_overlay_dock = tk.Frame(host.preview_canvas_host)
    host.preview_overlay_dock.place(x=758, y=50, width=190, height=220)
    host._preview_fullscreen_toggle_rect = (916, 10, 948, 42)
    host.preview_controls_canvas.bind("<Configure>", host._on_preview_controls_legend_configure)
    root.update_idletasks()
    host._place_preview_hint_overlay(refresh=True)
    root.update_idletasks()
    yield host
    pending = getattr(host, "_preview_controls_legend_configure_after_id", None)
    if pending:
        host.preview_controls_canvas.after_cancel(pending)
    host.preview_canvas_host.destroy()


def bounds(widget):
    return (widget.winfo_x(), widget.winfo_y(),
            widget.winfo_x() + widget.winfo_width(),
            widget.winfo_y() + widget.winfo_height())


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def event(host, x, y, dx=0, dy=0):
    canvas = host.preview_controls_canvas
    return SimpleNamespace(widget=canvas, x=x, y=y,
                           x_root=canvas.winfo_rootx() + x + dx,
                           y_root=canvas.winfo_rooty() + y + dy)


@pytest.mark.parametrize("fullscreen", [False, True])
@pytest.mark.parametrize("width,height", [(960, 600), (640, 360), (380, 600)])
def test_expanded_compass_does_not_cover_drawer(compass, fullscreen, width, height):
    compass.preview_canvas_host.place_configure(width=width, height=height)
    compass.preview_overlay_dock.place_configure(x=width - 202)
    compass.frame.update_idletasks()
    compass._preview_fullscreen_active = fullscreen
    compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    assert not overlaps(bounds(compass.preview_hint_frame), bounds(compass.preview_overlay_dock))
    assert compass._preview_controls_legend_grab_bbox is not None
    x1, y1, x2, y2 = bounds(compass.preview_hint_frame)
    assert 0 <= x1 < x2 <= width
    assert 46 <= y1 < y2 <= height


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("fullscreen", [False, True])
def test_drag_uses_visible_position_and_survives_refresh(compass, expanded, fullscreen):
    compass._preview_fullscreen_active = fullscreen
    if expanded:
        compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    before = bounds(compass.preview_hint_frame)
    x, y = (16, 16)
    if expanded:
        x1, y1, x2, y2 = compass._preview_controls_legend_grab_bbox
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
    press = event(compass, x, y)
    # A one-column fullscreen legend fills the available height.
    dy = 0 if fullscreen and expanded else 20
    drag = event(compass, x, y, dx=60, dy=dy)
    compass._on_preview_controls_legend_press(press)
    compass._on_preview_controls_legend_drag(drag)
    compass._on_preview_controls_legend_release(drag)
    compass.frame.update_idletasks()
    after = bounds(compass.preview_hint_frame)
    expected_y = min(before[1] + dy, 600 - (before[3] - before[1]))
    assert after[:2] == (before[0] + 60, expected_y)
    compass._place_preview_hint_overlay(refresh=True)
    compass.frame.update_idletasks()
    assert bounds(compass.preview_hint_frame) == after
    assert compass._preview_controls_legend_drag_state is None
    assert compass.frame.grab_current() is None


def test_scrollbar_does_not_shrink_compass_on_each_refresh(compass):
    compass.preview_canvas_host.place_configure(height=360)
    compass.frame.update_idletasks()
    compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    assert compass.preview_controls_vbar.winfo_manager() == "place"
    initial = bounds(compass.preview_hint_frame)
    for _ in range(8):
        compass._refresh_preview_controls_legend()
        compass.frame.update_idletasks()
        assert bounds(compass.preview_hint_frame) == initial


def settle(host, milliseconds=250):
    done = tk.BooleanVar(host.frame, False)
    host.frame.after(milliseconds, lambda: done.set(True))
    host.frame.wait_variable(done)


def test_expand_scroll_collapse_and_resize_leave_event_loop_idle(compass):
    with patch.object(compass, "_refresh_preview_controls_legend",
                      wraps=compass._refresh_preview_controls_legend) as refresh:
        for height in (360, 850, 600):
            compass.preview_canvas_host.place_configure(height=height)
            compass.frame.update_idletasks()
            compass._toggle_preview_controls_legend()
            # Withdrawn Tk windows do not emit native Configure events.
            compass._on_preview_controls_legend_configure()
            settle(compass)
            count = refresh.call_count
            settle(compass)
            assert refresh.call_count == count
            assert compass._preview_controls_legend_configure_after_id is None
            compass._on_preview_controls_legend_mousewheel(SimpleNamespace(delta=-120))
            compass.preview_controls_canvas.yview_moveto(0)
            compass._on_preview_controls_legend_press(event(compass, 20, 20))
            compass._on_preview_controls_legend_release(event(compass, 20, 20))
            settle(compass)
            assert not compass._is_preview_controls_legend_expanded()
            assert bounds(compass.preview_hint_frame)[2] - bounds(compass.preview_hint_frame)[0] == 360
            assert compass.preview_controls_canvas.bbox("preview_legend_context") is not None
            assert compass.preview_controls_canvas.yview() == (0.0, 1.0)
            assert compass.frame.grab_current() is None


def test_collapse_restores_manually_moved_compact_card(compass):
    compass._on_preview_controls_legend_press(event(compass, 16, 16))
    drag = event(compass, 16, 16, dx=120, dy=70)
    compass._on_preview_controls_legend_drag(drag)
    compass._on_preview_controls_legend_release(drag)
    compass.frame.update_idletasks()
    before = bounds(compass.preview_hint_frame)
    compass._toggle_preview_controls_legend()
    compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    assert bounds(compass.preview_hint_frame) == before


def test_hiding_compass_clears_interaction_and_allows_reopening(compass):
    compass._focus_preview_canvas = lambda: None
    compass._on_preview_controls_legend_press(event(compass, 16, 16))
    compass._toggle_preview_overlay_dock_tool("legend")
    assert compass._preview_controls_legend_current_bounds is None
    assert compass._preview_controls_legend_click_state is None
    assert compass._preview_controls_legend_drag_state is None
    assert compass.preview_hint_frame.winfo_manager() == ""
    compass._toggle_preview_overlay_dock_tool("legend")
    compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    assert compass._is_preview_controls_legend_expanded()
    assert not overlaps(bounds(compass.preview_hint_frame), bounds(compass.preview_overlay_dock))


def test_fullscreen_keeps_compass_expansion_and_shortcut_clicks_do_not_collapse(compass):
    compass._toggle_preview_controls_legend()
    for fullscreen in (True, False, True):
        compass._preview_fullscreen_active = fullscreen
        compass._place_preview_hint_overlay(refresh=True)
        assert compass._is_preview_controls_legend_expanded()
    compass._on_preview_controls_legend_press(event(compass, 80, 240))
    compass._on_preview_controls_legend_release(event(compass, 80, 240))
    assert compass._is_preview_controls_legend_expanded()
    compass._toggle_preview_controls_legend()
    for fullscreen in (False, True):
        compass._preview_fullscreen_active = fullscreen
        assert not compass._is_preview_controls_legend_expanded()


def test_compact_context_updates_without_reopening(compass):
    compass._get_preview_legend_context = lambda: {"filename": "SECOND_1234.jpg", "image_text": "Tablica: 2/1245"}
    compass._place_preview_hint_overlay()
    canvas = compass.preview_controls_canvas
    text = " ".join(canvas.itemcget(item, "text") for item in canvas.find_withtag("preview_legend_context")
                    if canvas.type(item) == "text")
    assert "SECOND_1234.jpg" in text and "2/1245" in text
    assert not compass._is_preview_controls_legend_expanded()


def test_expanded_header_stays_reachable_after_scrolling(compass):
    compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    canvas = compass.preview_controls_canvas
    canvas.yview_moveto(1)
    compass.frame.update_idletasks()
    bounds = canvas.bbox("preview_legend_toggle")
    assert bounds is not None
    assert bounds[1] - canvas.canvasy(0) < 15
    compass._on_preview_controls_legend_press(event(compass, 20, 20))
    compass._on_preview_controls_legend_release(event(compass, 20, 20))
    assert not compass._is_preview_controls_legend_expanded()


@pytest.mark.parametrize("expanded", [False, True])
def test_compass_has_no_right_gutter_and_uses_readable_counters(compass, expanded):
    if expanded:
        compass._toggle_preview_controls_legend()
    compass.frame.update_idletasks()
    canvas = compass.preview_controls_canvas
    assert canvas.winfo_width() == compass.preview_hint_frame.winfo_width()
    for item in canvas.find_withtag("preview_legend_counter"):
        assert tkfont.Font(root=compass.frame, font=canvas.itemcget(item, "font")).cget("size") >= 12
    canvas.yview_moveto(1)
    compass.frame.update_idletasks()
    x1, y1, x2, y2 = canvas.bbox("preview_legend_action")
    viewport_y = (y1 + y2) / 2 - canvas.canvasy(0)
    assert 0 <= x1 < x2 <= canvas.winfo_width()
    assert 0 < viewport_y < 50
    click = event(compass, (x1 + x2) / 2, viewport_y)
    compass._on_preview_controls_legend_press(click)
    compass._on_preview_controls_legend_release(click)
    assert compass._is_preview_controls_legend_expanded() is not expanded
