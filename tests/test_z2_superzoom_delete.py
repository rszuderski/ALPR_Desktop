from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.data_models import AnnotationStatus, Detection, ImageAnnotation
from auto_annotation_tool.gui import z2_canvas_interaction as interaction
from auto_annotation_tool.gui import z2_preview_editor as editor
from auto_annotation_tool.gui import z2_preview_state as state


def make_owner(count, selected, *, super_mode=True):
    plates = [Detection(
        "plate", 1.0, (i * 100, 10, i * 100 + 80, 40),
        polygon=[(i * 100, 10), (i * 100 + 80, 10), (i * 100 + 80, 40), (i * 100, 40)],
        attributes={"plate_annotation_id": f"plate-{i}", "ground_truth_text": f"WX{i}123"},
    ) for i in range(count)]
    annotations = [ImageAnnotation("current.jpg", 600, 400, plates),
                   ImageAnnotation("next.jpg", 600, 400, [Detection("plate", 1.0, (0, 0, 80, 20))])]
    canvas = Mock(original_image=object())
    canvas.view = {"polygon": tuple(plates[selected].polygon)}
    def set_view(view, *, redraw):
        canvas.view = view
        return True
    canvas.set_view_state.side_effect = set_view
    host = SimpleNamespace(
        current_annotations=annotations, current_preview_index=0, preview_canvas=canvas,
        _preview_super_correction_active=super_mode,
        _preview_selected_plate_by_image={"current.jpg": selected},
        _preview_focus_target={"filename": "current.jpg", "kind": "plate", "index": selected},
        _preview_polygon_focus_restore_state={"stale": True},
        _preview_focus_zoom_restore_state={"stale": True},
        _preview_focus_zoom_history=[{"stale": True}], _preview_focus_zoom_click_stage=1,
        _preview_delete_mode=True, _preview_delete_candidate_idx=selected,
        _preview_dirty_images=set(), _preview_drag_state=None, _preview_pending_vertex_hit=None,
        _push_preview_history_snapshot=Mock(), _push_preview_debug_event=Mock(),
        _refresh_preview_list_row_for_actual_index=Mock(), _refresh_preview_canvas_interactive=Mock(),
        _update_preview_edit_status=Mock(), _schedule_preview_autosave=Mock(),
        _cancel_preview_layout_restore_jobs=Mock(), _refresh_preview_plate_context_overlays=Mock(),
        _prime_preview_corner_drag_history=Mock(),
        _preview_focus_restore_matches_current_image=lambda: True,
        _compute_preview_plate_focus_view_state=lambda p: {"polygon": tuple(p)},
        _detection_polygon=lambda det: det.polygon,
    )
    host._get_preview_annotation = lambda: host.current_annotations[host.current_preview_index]
    host._get_preview_focus_image_key = lambda: host._get_preview_annotation().filename
    host._get_plate_detections = lambda ann: ann.plates if ann else []
    host._mark_preview_image_dirty = lambda ann, **kw: host._preview_dirty_images.add(ann.filename)
    for name in ("_get_selected_plate_index_for_ann", "_set_selected_plate_index_for_ann", "_set_preview_focus_target"):
        setattr(host, name, MethodType(getattr(state, name), host))
    host._focus_preview_plate = MethodType(interaction._focus_preview_plate, host)
    return host


@pytest.mark.parametrize("super_mode", [False, True])
def test_a_fits_plate_at_full_quality_without_waiting_for_idle(super_mode):
    host = make_owner(2, 0, super_mode=super_mode)
    host._preview_shortcuts_enabled = lambda *args, **kwargs: True
    host._preview_shortcut_is_duplicate = lambda *args: False
    host._mark_preview_user_interaction = Mock()
    host._preview_draw_mode = False
    host._capture_preview_view_state = Mock(return_value={})

    assert interaction._on_preview_cycle_plate_shortcut(host) == "break"

    assert host._get_selected_plate_index_for_ann(host._get_preview_annotation()) == 1
    host.preview_canvas._update_display.assert_called_once_with(interaction_fast=False)
    host.preview_canvas._schedule_final_quality_display.assert_not_called()
    if super_mode:
        assert not host.preview_canvas._navigation_rendering


@pytest.mark.parametrize("count,selected", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_delete_in_superzoom_shows_remaining_plate_on_same_image(count, selected):
    host = make_owner(count, selected)
    ann = host._get_preview_annotation()
    removed = ann.plates[selected]
    remaining = [det for det in ann.plates if det is not removed]
    expected_index = min(selected, len(remaining) - 1)

    assert editor._delete_preview_polygon(host, selected)

    assert host.current_preview_index == 0
    assert ann.plates == remaining
    assert host._get_selected_plate_index_for_ann(ann) == expected_index
    assert host.preview_canvas.view["polygon"] == tuple(remaining[expected_index].polygon)
    assert host._preview_focus_target == {"filename": "current.jpg", "kind": "plate", "index": expected_index}
    assert host._preview_dirty_images == {"current.jpg"}
    assert host._preview_focus_zoom_restore_state is None
    assert host._preview_focus_zoom_history == []
    assert host._preview_delete_mode is False
    assert host._preview_delete_candidate_idx is None
    assert len(host.current_annotations[1].plates) == 1
    host._schedule_preview_autosave.assert_called_once()


def test_delete_last_plate_keeps_image_and_clears_superzoom_target():
    host = make_owner(1, 0)
    assert editor._delete_preview_polygon(host, 0)
    assert host.current_preview_index == 0
    assert host._get_preview_annotation().status == AnnotationStatus.NO_PLATE
    assert host._get_preview_annotation().plates == []
    assert host._preview_focus_target is None
    host.preview_canvas.set_view_state.assert_not_called()


def test_delete_outside_superzoom_preserves_the_view_and_manual_save_option():
    host = make_owner(2, 0, super_mode=False)
    view = host.preview_canvas.view.copy()
    assert editor._delete_preview_polygon(host, 0, autosave=False)
    assert host.current_preview_index == 0
    assert host.preview_canvas.view == view
    host.preview_canvas.set_view_state.assert_not_called()
    host._schedule_preview_autosave.assert_not_called()


def test_invalid_delete_does_not_change_selection_or_schedule_save():
    host = make_owner(2, 1)
    before = list(host._get_preview_annotation().plates)
    assert not editor._delete_preview_polygon(host, 2)
    assert host._get_preview_annotation().plates == before
    assert host._get_selected_plate_index_for_ann(host._get_preview_annotation()) == 1
    host._schedule_preview_autosave.assert_not_called()


@pytest.mark.parametrize("selected", [0, 1])
def test_right_click_delete_keeps_real_tk_list_selection(selected):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        host = make_owner(2, selected)
        host._preview_draw_mode = False
        host.preview_canvas.canvasx.side_effect = float
        host.preview_canvas.canvasy.side_effect = float
        host.preview_listbox = tk.Listbox(root, exportselection=False)
        host.preview_listbox.insert(tk.END, "current.jpg", "next.jpg")
        host.preview_listbox.selection_set(0)
        host.preview_listbox.activate(0)
        host._preview_list_display_indices = [0, 1]
        host._get_preview_display_index = lambda i: i
        host._get_preview_actual_index_from_display = lambda i: i
        host._preview_list_item_text = lambda ann, **kw: ann.filename
        host._preview_list_item_color = lambda ann: "black"
        host._clear_listbox_selection_fast = lambda box: box.selection_clear(0, tk.END)
        host._mark_preview_user_interaction = Mock()
        host._defer_preview_autosave_for_navigation = Mock()
        host._refresh_plate_auto_scope_modal_selection_state = Mock()
        host._update_preview_toolbar_state = Mock()
        host._refresh_preview_list_row_for_actual_index = MethodType(editor._refresh_preview_list_row_for_actual_index, host)
        host._delete_preview_polygon = MethodType(editor._delete_preview_polygon, host)
        host.preview_listbox.bind("<<ListboxSelect>>", lambda e: editor._on_preview_select(host, e))

        assert editor._on_preview_canvas_right_click(host, SimpleNamespace(x=10, y=20)) == "break"
        root.update()
        editor._on_preview_select(host, SimpleNamespace(widget=host.preview_listbox))

        assert host.current_preview_index == 0
        assert host.preview_listbox.curselection() == (0,)
        assert host.preview_listbox.index(tk.ACTIVE) == 0
        assert len(host._get_preview_annotation().plates) == 1
        assert host.preview_canvas.view["polygon"] == tuple(host._get_preview_annotation().plates[0].polygon)
    finally:
        root.destroy()
