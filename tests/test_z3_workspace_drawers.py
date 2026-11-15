import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.z3_workspace_drawers import WorkspaceDrawers
from auto_annotation_tool.gui import z3_preview_ui as preview
from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab


@pytest.fixture
def scene():
    root = tk.Tk()
    root.geometry("1200x740+20+20")
    content = ttk.Frame(root)
    content.pack(fill="both", expand=True)
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)
    outer = tk.PanedWindow(content, orient="horizontal", sashwidth=7, sashpad=2, bd=0)
    outer.grid(row=0, column=0, sticky="nsew")
    left, right = ttk.Frame(outer), ttk.Frame(outer)
    outer.add(left, minsize=560, stretch="always")
    outer.add(right, minsize=280, width=310, stretch="never")
    inner = tk.PanedWindow(left, orient="horizontal", sashwidth=6, bd=0)
    inner.pack(fill="both", expand=True)
    listing, center = ttk.Frame(inner), ttk.Frame(inner)
    inner.add(listing, minsize=240, width=265)
    inner.add(center, minsize=320)
    canvas = tk.Canvas(center, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    images = tk.Listbox(listing)
    images.pack(fill="both", expand=True)
    images.insert("end", *[str(i) for i in range(100)])
    images.selection_set(18)
    images.see(18)
    entry = tk.Entry(right)
    entry.insert(0, "0.25")
    entry.pack()
    footer = ttk.Frame(content, height=95)
    footer.grid(row=1, column=0, sticky="ew", pady=(4, 2))
    footer.grid_propagate(False)
    action = ttk.Button(footer, text="Zapisz PZ2")
    action.grid()
    root.update()
    app_root = Mock()
    app_root.attributes.return_value = False
    app_root.state.return_value = "normal"
    app_root.geometry.return_value = "1200x740+20+20"
    owner = SimpleNamespace(
        frame=root, app=SimpleNamespace(root=app_root), detect_content_frame=content,
        detect_split=outer, detect_right_panel=right, preview_vertical_split=inner,
        preview_list_lf=listing, preview_lf=center, detect_footer_nav=footer,
        preview_canvas=canvas, preview_canvas_host=center,
        plates_listbox=images, _preview_render_state={"plate_id": "plate"},
        _preview_fullscreen_active=False, _get_preview_mode_overlay_default_position=lambda **kw: {},
        _preview_controls_legend_visible=True,
        _preview_controls_legend_current_bounds=None,
    )
    for name in ("_focus_preview_canvas", "_update_preview_edit_status", "_sync_preview_edit_status_visibility",
                 "_update_preview_toolbar_state", "_apply_preview_fullscreen_chrome",
                 "_schedule_preview_stabilized_rerender"):
        setattr(owner, name, Mock())
    clock = [0.0]
    drawer = owner._preview_workspace_drawers = WorkspaceDrawers(owner, clock=lambda: clock[0])
    yield root, owner, drawer, clock, entry
    root.destroy()


def advance(scene, seconds=.25):
    root, owner, drawer, clock, _ = scene
    drawer._cancel()
    clock[0] += seconds
    drawer._tick()
    root.update()


def enter(scene):
    preview.set_preview_fullscreen(scene[1], True)
    scene[0].update()
    advance(scene)


def test_round_trip_preserves_panels_values_selection_widths_and_footer_grid(scene):
    root, owner, drawer, clock, entry = scene
    before = (owner.preview_list_lf.winfo_width(), owner.detect_right_panel.winfo_width(),
              owner.detect_footer_nav.grid_info(), str(entry), entry.get())
    enter(scene)
    assert drawer.detached and not drawer.mode_transition
    assert len(owner.detect_split.panes()) == len(owner.preview_vertical_split.panes()) == 1
    assert owner.preview_list_lf.winfo_x() == -before[0]
    assert owner.detect_right_panel.winfo_x() == owner.detect_split.winfo_width()
    assert owner.detect_footer_nav.winfo_y() == owner.detect_content_frame.winfo_height()
    for button in drawer.buttons.values():
        assert button.winfo_manager() == "place"
    owner._schedule_preview_stabilized_rerender.assert_called_once_with(delay_ms=35)
    preview.set_preview_fullscreen(owner, False)
    advance(scene)
    assert not drawer.detached and not drawer.mode_transition
    assert tuple(map(str, owner.preview_vertical_split.panes())) == (str(owner.preview_list_lf), str(owner.preview_lf))
    assert abs(owner.preview_list_lf.winfo_width() - before[0]) <= 2
    assert abs(owner.detect_right_panel.winfo_width() - before[1]) <= 2
    assert owner.detect_footer_nav.grid_info() == before[2]
    assert (str(entry), entry.get()) == before[3:]
    assert owner.plates_listbox.curselection() == (18,)


def test_drawers_move_without_resizing_canvas_or_rerendering(scene):
    root, owner, drawer, clock, entry = scene
    enter(scene)
    before = (owner.preview_canvas.winfo_width(), owner.preview_canvas.winfo_height())
    owner._schedule_preview_stabilized_rerender.reset_mock()
    for side in drawer.panels:
        drawer.toggle(side)
        advance(scene, .08)
        assert 0 < drawer.visible[side] < 1
        assert before == (owner.preview_canvas.winfo_width(), owner.preview_canvas.winfo_height())
        advance(scene)
        assert drawer.visible[side] == 1
    owner._schedule_preview_stabilized_rerender.assert_not_called()


def test_shared_tool_drawer_animates_with_pz2_workspace_controller(scene):
    from auto_annotation_tool.gui.z2_drawer_slide import PreviewDrawerSlide
    from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette

    root, owner, drawer, clock, entry = scene
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    enter(scene)
    owner.app.palette = get_theme_palette()
    owner.preview_overlay_dock = tk.Frame(owner.preview_lf)
    slide = PreviewDrawerSlide(owner, host=owner.preview_lf, clock=lambda: clock[0])
    slide.place(1200, 1000, 68, 180, 260, fullscreen=True, toggle_y=68)
    for _ in range(4):
        # Exercise the actual Tk button callback used by PZ2.
        slide.button._invoke()
        slide._cancel()
        clock[0] += .3
        slide._tick()
        root.update()
        assert slide.visible == (0.0 if slide.hidden else 1.0)
    assert not errors
    assert entry.get() == "0.25"


def test_reversal_continues_from_current_position_and_preserves_original_window(scene):
    root, owner, drawer, clock, entry = scene
    preview.set_preview_fullscreen(owner, True)
    advance(scene, .06)
    preview.set_preview_fullscreen(owner, False)
    advance(scene, .05)
    position = owner.preview_list_lf.winfo_x()
    owner.app.root.geometry.return_value = "1920x1080+0+0"
    preview.set_preview_fullscreen(owner, True)
    root.update()
    assert owner.preview_list_lf.winfo_x() == position
    advance(scene)
    assert owner._preview_fullscreen_restore_geometry == "1200x740+20+20"
    preview.set_preview_fullscreen(owner, False)
    advance(scene)
    assert len(owner.detect_split.panes()) == 2
    owner.app.root.geometry.assert_called_with("1200x740+20+20")


def test_hidden_panel_is_not_reintroduced_by_fullscreen(scene):
    root, owner, drawer, clock, entry = scene
    owner.detect_split.forget(owner.detect_right_panel)
    root.update()
    enter(scene)
    assert drawer.buttons["right"].winfo_manager() == ""
    preview.set_preview_fullscreen(owner, False)
    advance(scene)
    assert len(owner.detect_split.panes()) == 1


def test_resize_anchors_open_and_closed_drawers_to_edges(scene):
    root, owner, drawer, clock, entry = scene
    enter(scene)
    drawer.toggle("right")
    advance(scene, .07)
    root.geometry("1000x600")
    root.update()
    advance(scene)
    panel = owner.detect_right_panel
    assert panel.winfo_x() + panel.winfo_width() == owner.detect_split.winfo_width()
    drawer.toggle("right")
    advance(scene)
    assert panel.winfo_x() == owner.detect_split.winfo_width()


def test_destroy_cancels_animation(scene):
    root, owner, drawer, clock, entry = scene
    preview.set_preview_fullscreen(owner, True)
    job = drawer._job
    owner.detect_content_frame.destroy()
    root.update()
    assert drawer.destroyed and drawer._job is None
    assert job not in root.tk.call("after", "info")


def test_stabilized_render_waits_for_drawer_animation(scene):
    root, owner, drawer, clock, entry = scene
    owner._preview_active_pid = "plate"
    owner._on_preview_select = Mock()
    drawer.mode_transition = True
    CharacterAnnotationTab._schedule_preview_stabilized_rerender(owner, delay_ms=1)
    done = tk.BooleanVar(root, False)
    root.after(30, lambda: done.set(True))
    root.wait_variable(done)
    owner._on_preview_select.assert_not_called()
    assert owner._preview_stabilized_render_after_id is None

def test_open_side_tabs_do_not_cover_panel_contents(scene):
    root, owner, drawer, clock, entry = scene
    enter(scene)

    # Po wejściu w fullscreen panele są schowane. Otwieramy oba.
    drawer.toggle("left")
    advance(scene)
    drawer.toggle("right")
    advance(scene)
    root.update_idletasks()

    left_panel = owner.preview_list_lf
    left_button = drawer.buttons["left"]
    right_panel = owner.detect_right_panel
    right_button = drawer.buttons["right"]

    left_panel_right = left_panel.winfo_rootx() + left_panel.winfo_width()
    left_button_left = left_button.winfo_rootx()
    assert left_button_left >= left_panel_right

    right_panel_left = right_panel.winfo_rootx()
    right_button_right = right_button.winfo_rootx() + right_button.winfo_width()
    assert right_button_right <= right_panel_left

def test_bottom_actions_tab_centers_in_reserved_source_legend_gap(scene):
    root, owner, drawer, clock, entry = scene
    enter(scene)

    canvas = owner.preview_canvas
    root.update_idletasks()

    # Nowy kontrakt fullscreen:
    # legenda sama zostawia centralną szczelinę dla zakładki Akcje.
    gap_left = 360.0
    gap_right = 500.0
    owner._preview_source_legend_gap_bounds = (
        gap_left,
        max(0.0, float(canvas.winfo_height() - 36)),
        gap_right,
        max(1.0, float(canvas.winfo_height() - 8)),
    )

    drawer.place_status_toggle()
    root.update_idletasks()

    button = drawer.buttons["bottom"]
    actual_center_root = (
        float(button.winfo_rootx())
        + float(button.winfo_width()) / 2.0
    )
    expected_center_root = (
        float(canvas.winfo_rootx())
        + (gap_left + gap_right) / 2.0
    )

    assert abs(actual_center_root - expected_center_root) <= 2.0
