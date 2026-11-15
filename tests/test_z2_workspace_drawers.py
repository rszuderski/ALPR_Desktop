import tkinter as tk
from tkinter import ttk, font as tkfont
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.z2_workspace_drawers import WorkspaceDrawers
from auto_annotation_tool.gui import z2_canvas_interaction as interaction
from auto_annotation_tool.gui import z2_preview_editor as editor
from auto_annotation_tool.gui import z2_layout_ui_runtime as layout
from auto_annotation_tool.gui import z2_panel_workflow as workflow
from auto_annotation_tool.gui import z2_preview_state as state


@pytest.fixture
def scene():
    root = tk.Tk()
    root.geometry("1100x680+30+30")
    pane = ttk.Panedwindow(root, orient="horizontal")
    pane.pack(fill="both", expand=True)
    left, center, right = [tk.Frame(pane) for _ in range(3)]
    for frame in (left, center, right):
        pane.add(frame)
    entry = tk.Entry(right)
    entry.insert(0, "Ręczna wartość GT")
    entry.pack()
    images = tk.Listbox(left)
    images.pack(fill="both", expand=True)
    images.insert("end", *[f"{i}.jpg" for i in range(100)])
    canvas = tk.Canvas(center, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.original_image = object()
    root.update()
    pane.sashpos(0, 270)
    pane.sashpos(1, 860)
    root.update()
    host = SimpleNamespace(
        frame=root, app=SimpleNamespace(root=Mock()), main_pane=pane,
        main_left_frame=left, main_center_frame=center, main_right_frame=right,
        canvas_frame=center, preview_canvas=canvas, preview_listbox=images,
        _preview_fullscreen_active=False, _preview_fullscreen_transition_active=False,
        _preview_list_selection_sync_pending=False, current_preview_index=12,
        _annotation_right_panel_visible=True,
        _get_preview_display_index=lambda index: index,
        _clear_listbox_selection_fast=lambda widget: widget.selection_clear(0, "end"),
        _get_preview_legend_theme=lambda: {}, _manual_xml_template_enabled=lambda: False,
        _get_preview_bottom_hint_text=lambda: "Tablica 1/2 | Q/E zdjęcia | Spacja OK/NOK | F dopasuj | Ctrl+S zapis",
        _get_preview_legend_font=lambda size, weight: tkfont.Font(root, family="Segoe UI", size=size, weight=weight),
        _is_free_mode_session_context=lambda: False,
    )
    for name in ("_set_annotation_process_log_visibility", "_apply_preview_fullscreen_chrome",
                 "_hide_preview_overlay_dock_stack", "_hide_preview_controls_legend_overlay",
                 "_cancel_preview_controls_legend_animation", "_push_preview_debug_event",
                 "_update_preview_toolbar_state", "_update_preview_edit_status",
                 "_refresh_preview_list_row_for_actual_index", "_refresh_preview_list_summary",
                 "_place_preview_legend_overlay", "_place_preview_overlay_dock",
                 "_place_preview_campaign_gate_overlay", "_refresh_preview_canvas",
                 "_cancel_preview_layout_restore_jobs",
                 "_schedule_preview_layout_restore_after_resize", "_refresh_step2_action_states",
                 "_schedule_right_panel_content_restore_after_fullscreen",
                 "_schedule_main_pane_layout_refresh"):
        setattr(host, name, Mock())
    host._draw_preview_bottom_hint = lambda target: editor._draw_preview_bottom_hint(host, target)
    host._should_show_right_panel = lambda: layout._should_show_right_panel(host)
    host.app.root.attributes.return_value = False
    host.app.root.state.return_value = "normal"
    host.app.root.geometry.return_value = "1100x680+30+30"
    clock = [0.0]
    drawer = host._preview_workspace_drawers = WorkspaceDrawers(host, clock=lambda: clock[0])
    yield root, host, drawer, clock, entry
    root.destroy()


def advance(scene, seconds=.25):
    root, host, drawer, clock, _ = scene
    drawer._cancel()
    clock[0] += seconds
    drawer._tick()
    root.update()


def enter(scene):
    root, host, drawer, clock, _ = scene
    interaction._set_preview_fullscreen(host, True)
    root.update()
    host._draw_preview_bottom_hint(host.preview_canvas)
    advance(scene)


def test_round_trip_preserves_widgets_values_selection_and_custom_widths(scene):
    root, host, drawer, clock, entry = scene
    before = (str(entry), entry.get(), host.main_left_frame.winfo_width(), host.main_right_frame.winfo_width())
    enter(scene)
    assert drawer.detached
    assert tuple(map(str, host.main_pane.panes())) == (str(host.main_center_frame),)
    assert host.main_left_frame.winfo_x() == -before[2]
    assert host.main_right_frame.winfo_x() == host.main_pane.winfo_width()
    assert drawer.bottom_button.winfo_manager() == "place"
    assert host.preview_canvas.bbox("preview_bottom_hint")[1] >= host.preview_canvas.winfo_height()
    host._preview_list_selection_sync_pending = True
    drawer.toggle("left")
    advance(scene)
    assert host.preview_listbox.curselection() == (12,)
    assert host.main_left_frame.winfo_x() == 0
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    assert not drawer.detached
    assert entry.get() == before[1] and str(entry) == before[0]
    assert abs(host.main_left_frame.winfo_width() - before[2]) <= 2
    assert abs(host.main_right_frame.winfo_width() - before[3]) <= 2
    host._refresh_step2_action_states.assert_not_called()
    host._schedule_right_panel_content_restore_after_fullscreen.assert_not_called()
    host._schedule_main_pane_layout_refresh.assert_not_called()


def test_each_drawer_moves_without_resizing_or_redrawing_preview(scene):
    root, host, drawer, clock, _ = scene
    enter(scene)
    dimensions = (host.preview_canvas.winfo_width(), host.preview_canvas.winfo_height())
    host._refresh_preview_canvas.reset_mock()
    for side in ("left", "right", "bottom"):
        drawer.toggle(side)
        advance(scene, .08)
        assert 0 < drawer.visible[side] < 1
        assert dimensions == (host.preview_canvas.winfo_width(), host.preview_canvas.winfo_height())
        advance(scene)
        assert drawer.visible[side] == 1
    host._refresh_preview_canvas.assert_not_called()


def test_quick_reversal_continues_from_current_position(scene):
    root, host, drawer, clock, _ = scene
    enter(scene)
    drawer.toggle("left")
    advance(scene, .08)
    position = host.main_left_frame.winfo_x()
    drawer.toggle("left")
    root.update()
    assert host.main_left_frame.winfo_x() == position
    advance(scene)
    assert drawer.visible["left"] == 0


def test_rapid_fs_reversal_keeps_original_window_restore_and_cancels_old_completion(scene):
    root, host, drawer, clock, _ = scene
    interaction._set_preview_fullscreen(host, True)
    root.update()
    advance(scene, .05)
    interaction._set_preview_fullscreen(host, False)
    advance(scene, .05)
    host.app.root.geometry.return_value = "1920x1080+0+0"
    interaction._set_preview_fullscreen(host, True)
    advance(scene)
    assert host._preview_fullscreen_restore_geometry == "1100x680+30+30"
    assert host._preview_fullscreen_active and drawer.detached
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    assert not drawer.detached
    assert len(host.main_pane.panes()) == 3
    host.app.root.geometry.assert_called_with("1100x680+30+30")


def test_hidden_right_panel_is_not_created_by_fs(scene):
    root, host, drawer, clock, _ = scene
    host.main_pane.forget(host.main_right_frame)
    host._annotation_right_panel_visible = False
    root.update()
    enter(scene)
    assert not drawer.right_allowed
    assert drawer.buttons["right"].winfo_manager() == ""
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    assert len(host.main_pane.panes()) == 2


def test_background_layout_updates_keep_detached_panel_content(scene):
    root, host, drawer, clock, entry = scene
    enter(scene)
    assert layout._should_show_right_panel(host)
    for _ in range(5):
        layout._sync_main_pane_right_panel_visibility(host)
        workflow._apply_main_pane_layout(host, force_defaults=True)
    assert len(host.main_pane.panes()) == 1
    assert entry.winfo_manager() == "pack"
    assert entry.get() == "Ręczna wartość GT"


def test_bottom_drawer_survives_overlay_redraw_and_can_hide_in_window(scene):
    root, host, drawer, clock, _ = scene
    enter(scene)
    canvas = host.preview_canvas
    for _ in range(3):
        host._draw_preview_bottom_hint(canvas)
        assert canvas.bbox("preview_bottom_hint")[1] >= canvas.winfo_height()
    drawer.toggle("bottom")
    advance(scene)
    assert canvas.bbox("preview_bottom_hint")[3] < canvas.winfo_height()
    assert all("preview_viewport_fixed" in canvas.gettags(item) for item in canvas.find_withtag("preview_bottom_hint"))
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    host._draw_preview_bottom_hint(canvas)
    drawer.toggle("bottom")
    advance(scene)
    assert drawer.visible["bottom"] == 0
    assert canvas.bbox("preview_bottom_hint")[1] >= canvas.winfo_height()


def test_destroy_cancels_animation(scene):
    root, host, drawer, clock, _ = scene
    interaction._set_preview_fullscreen(host, True)
    job = drawer._job
    host.main_pane.destroy()
    root.update()
    assert drawer.destroyed and drawer._job is None
    assert job not in root.tk.call("after", "info")


def test_qe_updates_selection_while_image_drawer_is_open(scene):
    root, host, drawer, clock, _ = scene
    host.current_annotations = [SimpleNamespace(filename=f"{i}.jpg") for i in range(100)]
    for name in ("_cancel_preview_selection_render", "_defer_preview_autosave_for_navigation",
                 "_load_current_preview_selection", "_schedule_preview_resume_persist",
                 "_refresh_plate_auto_scope_modal_selection_state"):
        setattr(host, name, Mock())
    enter(scene)
    state._select_preview_index(host, 7)
    assert host._preview_list_selection_sync_pending
    drawer.toggle("left")
    advance(scene)
    assert host.preview_listbox.curselection() == (7,)
    state._select_preview_index(host, 8)
    assert host.preview_listbox.curselection() == (8,)
    assert not host._preview_list_selection_sync_pending


def test_resize_during_slide_keeps_panels_at_current_window_edges(scene):
    root, host, drawer, clock, _ = scene
    enter(scene)
    drawer.toggle("right")
    advance(scene, .07)
    root.geometry("900x550")
    root.update()
    advance(scene)
    assert host.main_right_frame.winfo_x() + host.main_right_frame.winfo_width() == host.main_pane.winfo_width()
    drawer.toggle("right")
    advance(scene)
    assert host.main_right_frame.winfo_x() == host.main_pane.winfo_width()


def test_status_handle_stays_clear_of_tool_drawer_through_slide_and_resize(scene):
    from auto_annotation_tool.gui.z2_drawer_slide import PreviewDrawerSlide
    from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette

    root, host, drawer, clock, _ = scene
    enter(scene)
    host.app.palette = get_theme_palette()
    host.preview_overlay_dock = tk.Frame(host.canvas_frame)
    slide = host._preview_drawer_slide = PreviewDrawerSlide(host)

    def bounds(widget):
        return (widget.winfo_rootx(), widget.winfo_rooty(),
                widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())

    def separate(a, b):
        return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]

    for size in ("1100x680", "900x550"):
        root.geometry(size)
        root.update()
        width = host.canvas_frame.winfo_width()
        slide.place(width, width - 280, 68, 270, 430, fullscreen=True, toggle_y=68)
        for _ in range(2):
            drawer.toggle("right")
            for elapsed in (.07, .25):
                advance(scene, elapsed)
                handle = bounds(drawer.buttons["right"])
                assert separate(handle, bounds(slide.panel))
                assert separate(handle, bounds(slide.button))
                assert separate(handle, bounds(host.main_right_frame))
                left_handle = bounds(drawer.buttons["left"])
                assert abs((handle[1] + handle[3]) - (left_handle[1] + left_handle[3])) <= 2
                if size == "1100x680":
                    assert handle[1] >= bounds(slide.panel)[3] + 6
    drawer.sync_right_visibility(False)
    root.update()
    left_handle = bounds(drawer.buttons["left"])
    assert left_handle[3] <= host.main_pane.winfo_rooty() + host.main_pane.winfo_height()


def test_right_panel_does_not_absorb_extra_window_width_after_fs(scene):
    root, host, drawer, clock, _ = scene
    enter(scene)
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    before = host.main_right_frame.winfo_width()
    root.geometry("1400x680")
    root.update()
    assert host.main_right_frame.winfo_width() == before


def test_layout_reclaims_canvas_space_from_a_restored_528px_sidebar(scene):
    root, host, drawer, clock, _ = scene
    root.geometry("1492x702")
    host.main_pane.pane(host.main_center_frame, weight=6)
    host.main_pane.pane(host.main_left_frame, weight=2)
    host.main_pane.pane(host.main_right_frame, weight=0)
    root.update()
    host.main_pane.sashpos(1, 959)
    host.main_pane.sashpos(0, 318)
    root.update()
    assert host.main_right_frame.winfo_width() == 528
    before_canvas = host.main_center_frame.winfo_width()
    host._main_pane_layout_initialized = True
    host._pane_has_child = lambda pane, child: str(child) in tuple(map(str, pane.panes()))
    host._get_main_pane_width_limits = lambda: layout._get_main_pane_width_limits(host)
    host._sync_main_pane_right_panel_visibility = lambda: layout._sync_main_pane_right_panel_visibility(host)
    workflow._apply_main_pane_layout(host, force_defaults=False)
    root.update()
    assert host.main_right_frame.winfo_width() <= 280
    assert host.main_center_frame.winfo_width() >= before_canvas + 248
    assert host.main_left_frame.winfo_width() == 318
    enter(scene)
    interaction._set_preview_fullscreen(host, False)
    advance(scene)
    assert host.main_right_frame.winfo_width() <= 280
