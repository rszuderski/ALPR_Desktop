import inspect

from auto_annotation_tool.gui import z2_canvas_interaction


def test_enter_fullscreen_forces_fit_after_resize():
    source = inspect.getsource(
        z2_canvas_interaction._set_preview_fullscreen
    )

    active_marker = "self._preview_fullscreen_active = True"
    force_marker = "self._preview_force_fit_after_resize = True"
    schedule_marker = "self._schedule_preview_layout_restore_after_resize()"

    active_pos = source.find(active_marker)
    force_pos = source.find(force_marker, active_pos)
    schedule_pos = source.find(schedule_marker, active_pos)

    assert active_pos >= 0
    assert force_pos > active_pos
    assert schedule_pos > force_pos


def test_force_fit_restore_path_centers_via_fit():
    source = inspect.getsource(
        z2_canvas_interaction._restore_preview_layout_view_after_resize
    )

    assert '_preview_force_fit_after_resize' in source
    assert 'self._fit_preview_image_to_view()' in source
