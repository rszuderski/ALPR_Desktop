"""AKT74: real Tk class-event routing and view transform, not direct method calls.

Synthetic GT only: no session files, no model results, no writes to operator data.
"""
from __future__ import annotations
from types import SimpleNamespace
import math
import tkinter as tk
from tkinter import ttk
import pytest
from test_z4_canvas_akt70_selection_drag import setup


def native_fixture():
    root = tk.Tk()
    root.geometry('920x720+150+120')
    tree = ttk.Treeview(root, height=3)
    tree.insert('', 'end', text='Obraz 01')
    tree.pack(fill='x')
    c = tk.Canvas(root, width=740, height=540, background='#142028', takefocus=1)
    c.pack(fill='both', expand=True)
    entry = tk.Entry(root)
    entry.pack()
    root.update()
    mod, editor, canvas, host, scene, messages, corrected = setup(c)
    host.window = root
    host.subject_tree = tree
    host._image.size = (100, 100)
    host._view_zoom = 1.25
    host._view_center = (50., 50.)
    host._view_fit_scale = 1.0
    host._view_pixel_scale = 1.0
    editor.to_canvas = mod.CanvasReviewEditor.to_canvas.__get__(editor)
    editor.to_source = mod.CanvasReviewEditor.to_source.__get__(editor)
    editor.plate_fit_previous = None
    editor.shortcut_tag = None
    editor.shortcut_original_tags = {}
    editor.fit_key_down = False
    editor.fit_release_after = None
    def paint():
        iw, ih = host._image.size
        cw, ch = canvas.winfo_width(),canvas.winfo_height()
        host._view_fit_scale = min((cw-16)/iw, (ch-16)/ih)
        host._view_pixel_scale = host._view_fit_scale * host._view_zoom
        cx,cy=host._view_center
        scale=host._view_pixel_scale
        half_x=cw/(2*scale);half_y=ch/(2*scale)
        cx = iw/2 if iw<=2*half_x else max(half_x,min(iw-half_x,cx))
        cy = ih/2 if ih<=2*half_y else max(half_y,min(ih-half_y,cy))
        host._view_center=(cx,cy)
        host.paints+=1
    host._paint_image = paint
    paint()
    editor.bindings()
    # Real OS mouse inside canvas; we replace ONLY the pointer query to ensure
    # Tk/WM cursor-warp limitations do not mask binding delivery correctness.
    editor._pointer_for_selection=lambda event=None: editor.to_canvas((35,30))
    return root, tree, c, entry, editor, host, scene, messages


def test_shortcuts_work_before_treeview_class_can_consume_keypress():
    root, tree, c, entry, e, h, s, msg = native_fixture()
    try:
        # DELIBERATELY consume the event on the Treeview widget. Old
        # Toplevel-based code never receives such a KeyPress.
        count=[]
        tree.bind('<KeyPress-s>', lambda event:(count.append('S'),'break')[-1])
        tree.bind('<KeyPress-a>', lambda event:(count.append('A'),'break')[-1])
        root.lift();tree.focus_force();root.update()
        # Does NOT call select_mode or fit_plate_toggle directly.
        tree.event_generate('<KeyPress-s>');root.update()
        assert e.selected == 'o1' and e.mode=='edit'
        assert count==[]  # custom bindtag handles before widget/class
        tree.event_generate('<KeyRelease-s>');root.update()
        tree.event_generate('<KeyPress-s>');root.update()
        assert e.selected == ''
        tree.event_generate('<KeyRelease-s>');root.update()
        previous=(h._view_zoom,h._view_center)
        tree.event_generate('<KeyPress-a>');root.update()
        assert e.plate_fit_active and h._view_zoom > previous[0]
        xys=[e.to_canvas(point) for point in s['objects'][0]['quad']]
        min_x=min(x for x,y in xys);max_x=max(x for x,y in xys)
        min_y=min(y for x,y in xys);max_y=max(y for x,y in xys)
        assert min_x>10 and max_x<c.winfo_width()-10
        assert min_y>10 and max_y<c.winfo_height()-10
        assert max_x-min_x>c.winfo_width()*0.30 or max_y-min_y>c.winfo_height()*0.30
        tree.event_generate('<KeyPress-a>');root.update()
        assert e.plate_fit_active  # autorepeat key must NOT toggle off
        tree.event_generate('<KeyRelease-a>');root.update()
        tree.event_generate('<KeyPress-a>');root.update()
        assert not e.plate_fit_active and h._view_zoom==previous[0]
        assert h._view_center==previous[1]
    finally:
        e.dispose();root.destroy()


def test_no_shortcut_when_typing_GT_and_no_binding_to_other_windows():
    root, tree, c, entry, e, h, s, msg = native_fixture()
    try:
        entry.focus_force();root.update()
        assert e._canvas_hotkey_pointer(SimpleNamespace(widget=entry, state=0)) is None
        entry.event_generate('<KeyPress-s>');root.update()
        assert e.selected==''
        entry.event_generate('<KeyPress-a>');root.update()
        assert not e.plate_fit_active
        foreign = tk.Toplevel(root)
        foreign_entry = tk.Entry(foreign); foreign_entry.pack();foreign.update()
        assert e.shortcut_tag not in foreign_entry.bindtags()
        foreign.destroy()
    finally:
        e.dispose();root.destroy()


def test_zoom_unique_plate_without_hover_hit_and_fit_restoration():
    root, tree, c, entry, e, h, s, msg = native_fixture()
    try:
        e._pointer_for_selection=lambda event=None: (5.,5.)
        root.lift();tree.focus_force();root.update()
        prev=(h._view_zoom,h._view_center)
        tree.event_generate('<KeyPress-a>');root.update()
        assert e.plate_fit_active
        assert h._view_zoom>prev[0]
        tree.event_generate('<KeyRelease-a>');root.update()
        tree.event_generate('<KeyPress-a>');root.update()
        assert h._view_center==prev[1] and h._view_zoom==prev[0]
    finally:
        e.dispose();root.destroy()


def test_A_fits_only_plate_even_with_pointer_outside_canvas():
    root, tree, c, entry, e, h, s, msg = native_fixture()
    try:
        e._pointer_for_selection=lambda event=None: None
        root.lift();tree.focus_force();root.update()
        initial_zoom=h._view_zoom
        tree.event_generate('<KeyPress-a>');root.update()
        assert e.plate_fit_active
        assert h._view_zoom>initial_zoom
        tree.event_generate('<KeyRelease-a>');root.update()
        tree.event_generate('<KeyPress-a>');root.update()
        assert not e.plate_fit_active
        assert h._view_zoom==initial_zoom
    finally:
        e.dispose();root.destroy()
