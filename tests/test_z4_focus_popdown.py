"""Native Tcl-created dropdown focus must not crash or dismiss review badges."""
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui import mobile_canvas_info_badge as badge


@pytest.fixture
def focus_ui():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Native Tk display required')
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    window = tk.Toplevel(root, name='review')
    combo = ttk.Combobox(window, values=('R0', 'R1', 'R2'), state='readonly')
    combo.pack()
    canvas = tk.Canvas(window, width=200, height=100)
    canvas.pack()
    ed = SimpleNamespace(h=SimpleNamespace(window=window, _closed=False), c=canvas,
                         selected='retained-selection', request_overlay=lambda: None)
    badge.install_focus_cleanup(ed)
    root.update()
    canvas.focus_force()
    root.update()
    try:
        yield root, ed, combo, errors
    finally:
        ed.h._closed = True
        root.destroy()


def post(root, combo):
    # This is the real ttk implementation's listbox, not a Python widget.
    root.tk.call('ttk::combobox::Post', str(combo))
    popup = str(root.tk.call('ttk::combobox::PopdownWindow', str(combo)))
    root.tk.call('focus', '-force', popup + '.f.l')
    root.update()
    assert str(root.tk.call('focus')) == popup + '.f.l'
    return popup


@pytest.mark.parametrize('own_window', [True, False])
def test_native_combobox_popdown_belongs_to_its_review_window(focus_ui, own_window):
    root, ed, combo, errors = focus_ui
    if not own_window:
        # A similar pathname is insufficient: it must be an actual ancestor.
        other = tk.Toplevel(root, name='review_extra')
        combo = ttk.Combobox(other, values=('A', 'B'), state='readonly')
        combo.pack()
        root.update()
    post(root, combo)
    try:
        with pytest.raises(KeyError, match='popdown'):
            root.focus_get()  # Reproduce the original Tkinter conversion failure.
        assert badge.focus_in_window(ed) is own_window
        assert not errors
    finally:
        root.tk.call('ttk::combobox::Unpost', str(combo))


def test_idle_cleanup_preserves_badge_during_dropdown_and_clears_on_other_window(focus_ui):
    root, ed, combo, errors = focus_ui
    mark = ed.c.create_rectangle(5, 5, 25, 25, tags=('active_layer_badge',))
    for _ in range(3):
        post(root, combo)
        # Exercise the exact deferred callback in the reported traceback.
        ed.h.window.event_generate('<FocusOut>')
        root.update()
        assert not errors and not ed._badge_window_inactive
        assert ed.c.type(mark) == 'rectangle' and ed.selected == 'retained-selection'
        root.tk.call('ttk::combobox::Unpost', str(combo))
        ed.c.focus_force()
        root.update()
    other = tk.Toplevel(root)
    entry = ttk.Entry(other)
    entry.pack()
    root.update()
    entry.focus_force()
    ed.h.window.event_generate('<FocusOut>')
    root.update()
    assert not errors and ed._badge_window_inactive and not ed.c.find_withtag('active_layer_badge')
    assert ed.selected == 'retained-selection'


def test_destroyed_window_is_inactive(focus_ui):
    root, ed, combo, errors = focus_ui
    ed.h._closed = True
    ed.h.window.destroy()
    assert not badge.focus_in_window(ed)
    root.update()
    assert not errors
