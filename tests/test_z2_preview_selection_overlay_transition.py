import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_preview_editor


def test_transition_clear_removes_old_overlay_and_hides_gt():
    canvas = Mock()
    host = SimpleNamespace(
        preview_canvas=canvas,
        _preview_fullscreen_toggle_bbox=(1, 2, 3, 4),
        _plate_gt_mode_toggle_bbox=(1, 2, 3, 4),
        _z2_gt_pack_status_bbox=(1, 2, 3, 4),
        _preview_bottom_hint_bbox=(1, 2, 3, 4),
        _preview_bottom_hint_move_bbox=(1, 2, 3, 4),
        _preview_bottom_hint_collapse_bbox=(1, 2, 3, 4),
        _preview_bottom_hint_restore_bbox=(1, 2, 3, 4),
        _preview_super_correction_badge_bbox=(1, 2, 3, 4),
        _preview_super_correction_handle_bbox=(1, 2, 3, 4),
    )

    old_hide = z2_preview_editor.z2_plate_gt_inline.hide_inline_plate_gt_editors
    hide = Mock()
    z2_preview_editor.z2_plate_gt_inline.hide_inline_plate_gt_editors = hide
    try:
        z2_preview_editor._clear_preview_selection_transition_overlay(host)
    finally:
        z2_preview_editor.z2_plate_gt_inline.hide_inline_plate_gt_editors = old_hide

    canvas.delete.assert_any_call("preview_overlay")
    hide.assert_called_once_with(host, destroy=False)

    assert host._preview_fullscreen_toggle_bbox is None
    assert host._plate_gt_mode_toggle_bbox is None
    assert host._z2_gt_pack_status_bbox is None
    assert host._preview_bottom_hint_bbox is None
    assert host._preview_bottom_hint_move_bbox is None
    assert host._preview_bottom_hint_collapse_bbox is None
    assert host._preview_bottom_hint_restore_bbox is None
    assert host._preview_super_correction_badge_bbox is None
    assert host._preview_super_correction_handle_bbox is None


def test_deferred_selection_clears_old_overlay_before_scheduling_new_render():
    source = inspect.getsource(z2_preview_editor._on_preview_select)

    clear_pos = source.find("_clear_preview_selection_transition_overlay")
    schedule_pos = source.find("_schedule_preview_selection_render")

    assert clear_pos >= 0
    assert schedule_pos >= 0
    assert clear_pos < schedule_pos


def test_load_selection_also_clears_transition_overlay_on_real_change():
    source = inspect.getsource(z2_preview_editor._load_current_preview_selection)

    selection_pos = source.find("if selection_changed:")
    clear_pos = source.find(
        "_clear_preview_selection_transition_overlay",
        selection_pos,
    )

    assert selection_pos >= 0
    assert clear_pos > selection_pos
