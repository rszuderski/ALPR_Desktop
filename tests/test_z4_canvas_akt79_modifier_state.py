"""AKT79 regression based on real AKT78 Windows log: S entered with state=8.

This test must FAIL on AKT77 and PASS on AKT79. Neither test writes GT
or reports: it uses a synthetic frame from the earlier Canvas fixtures.
"""
from copy import deepcopy
from types import SimpleNamespace

from test_z4_canvas_akt70_selection_drag import setup, event, key_event, overlay_handles


def test_real_log_state8_s_selects_quad_and_second_s_deselects():
    _mod, editor, canvas, _host, scene, messages, _ = setup()
    before = deepcopy(scene)
    canvas.pointer = (35, 35)
    editor.hover_motion(event(35, 35))
    assert editor._s_hover_hit is not None
    assert editor.selection_key_press(key_event(canvas, state=8)) == 'break'
    assert editor.selected == 'o1' and editor.mode == 'edit', messages
    assert len(overlay_handles(canvas)) == 4
    editor.reset_selection_key()
    assert editor.selection_key_press(key_event(canvas, state=8)) == 'break'
    assert editor.selected == '' and editor.mode == 'select'
    assert scene == before


def test_ctrl_s_does_not_select_quad():
    _mod, editor, canvas, _host, _scene, _messages, _ = setup()
    canvas.pointer = (35, 35)
    editor.hover_motion(event(35, 35))
    assert editor.select_mode(key_event(canvas, state=4)) == 'break'
    assert editor.selected == ''


def test_state8_a_zoom_and_second_press_restores_previous_view():
    _mod, editor, canvas, host, scene, messages, _ = setup()
    before = deepcopy(scene)
    host._image = SimpleNamespace(size=(100, 100), width=100, height=100)
    host._view_fit_scale = 3.0
    host._view_zoom = 1.0
    old_center = host._view_center
    editor.plate_fit_previous = None
    editor.plate_fit_active = False
    editor.plate_fit_gid = ''
    canvas.pointer=(35,35)
    editor.hover_motion(event(35,35))
    assert editor.fit_plate_toggle(SimpleNamespace(widget=canvas, state=8, keysym='a'))=='break'
    assert editor.plate_fit_active and host._view_zoom > 1.0, messages
    assert editor.fit_plate_toggle(SimpleNamespace(widget=canvas, state=8, keysym='a'))=='break'
    assert not editor.plate_fit_active and host._view_zoom == 1.0
    assert host._view_center == old_center
    assert scene == before


def test_ctrl_a_does_not_zoom():
    _mod, editor, canvas, host, _scene, _messages, _ = setup()
    host._image = SimpleNamespace(size=(100, 100), width=100, height=100)
    host._view_zoom = 1.0
    editor.plate_fit_active = False
    editor.plate_fit_previous = None
    editor.plate_fit_gid = ''
    assert editor.fit_plate_toggle(SimpleNamespace(widget=canvas, state=4, keysym='a')) == 'break'
    assert not editor.plate_fit_active and host._view_zoom == 1.0


def test_native_tk_mod1_s_reaches_selection_and_handles():
    """Generate the same state=8 that AKT78 recorded on Windows."""
    import tkinter as tk
    import pytest
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display unavailable')
    try:
        root.geometry('680x510+80+80')
        canvas = tk.Canvas(root, width=560, height=390, takefocus=1)
        canvas.pack()
        root.update()
        _mod, editor, canvas, host, scene, messages, _ = setup(canvas)
        host.window = root
        editor.bindings()
        root.update()
        canvas.event_generate('<Motion>',x=35,y=35,warp=1)
        root.update()
        # OS pointer warping is not guaranteed in all CI/window managers.
        px = canvas.winfo_pointerx()-canvas.winfo_rootx()
        py = canvas.winfo_pointery()-canvas.winfo_rooty()
        if not (0 <= px < canvas.winfo_width() and 0 <= py < canvas.winfo_height()):
            pytest.skip('Cannot warp pointer into Canvas in this environment')
        canvas.focus_force()
        root.update()
        canvas.event_generate('<KeyPress-s>',state=8)
        root.update()
        assert editor.selected == 'o1' and editor.mode == 'edit', messages
        assert len(overlay_handles(canvas)) == 4
        canvas.event_generate('<KeyRelease-s>',state=8)
        root.update()
    finally:
        try:
            editor.dispose()
        except (NameError, AttributeError):
            pass
        root.destroy()
