"""AKT82: R reset/remove, T ROI-only, N cycle, PPM toggles once.

No file or session data is changed in these tests. The fixture uses in-memory
GT and preserves immutable MT parent geometry.
"""
from copy import deepcopy
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import pytest
from test_z4_canvas_akt70_selection_drag import setup, event, key_event


def make_event(widget, key, state=8):
    return SimpleNamespace(widget=widget, keysym=key, state=state)


def test_ppm_first_click_toggles_flag_without_prior_selection():
    mod, editor, canvas, _host, scene, messages, corrected = setup()
    before_mt = deepcopy(scene['parents'])
    calls = []
    def toggle(_session, _gid, oid):
        calls.append(oid)
        match = next(x for x in scene['objects'] if x['object_id'] == oid)
        match['error'] = not match['error']
    mod.core.toggle_error = toggle
    assert editor.selected == '' and editor.mode == 'select'
    assert editor.right_press(event(35, 35)) == 'break'
    assert calls == ['o1']
    assert scene['objects'][0]['error'] is True
    assert editor.selected == 'o1' and editor.mode == 'edit'
    assert editor.h._view_drag_at is None
    assert scene['parents'] == before_mt
    assert editor.right_press(event(35, 35)) == 'break'
    assert calls == ['o1','o1']
    assert scene['objects'][0]['error'] is False
    assert scene['parents'] == before_mt


def test_ppm_can_flag_a_different_frame_without_s():
    mod, editor, canvas, host, scene, messages, corrected = setup()
    another = deepcopy(scene['objects'][0]); another['object_id'] = 'o2'
    another['quad'] = [[75,75],[95,75],[95,85],[75,85]]
    another['parent_id'] = ''
    scene['objects'].append(another)
    editor.selected = 'o1'; editor.mode = 'edit'
    calls=[]
    def toggle(_session, _gid, oid):
        calls.append(oid)
        obj=next(x for x in scene['objects'] if x['object_id']==oid)
        obj['error']=not obj['error']
    mod.core.toggle_error = toggle
    editor.right_press(event(84,80))
    assert calls == ['o2']
    assert editor.selected == 'o2' and editor.mode == 'edit'
    assert scene['objects'][0]['error'] is False
    assert scene['objects'][1]['error'] is True


def test_ppm_outside_frame_pans_without_toggling_flag():
    mod, editor, canvas, host, scene, messages, corrected = setup()
    calls=[]
    mod.core.toggle_error=lambda *args:calls.append(args)
    editor.right_press(event(300,190))
    assert not calls
    assert editor.selected == ''
    assert host._view_drag_at is not None


def test_r_resets_selected_mt_child_even_on_windows_state8():
    mod, editor, canvas, host, scene, messages, corrected=setup()
    original_mt=deepcopy(scene['parents'])
    scene['objects'][0]['quad']=[[18,27],[52,30],[52,41],[20,40]]
    editor.selected='o1';editor.mode='edit'
    editor.prepare_navigation=lambda:True
    editor.refresh_tree=lambda:None
    removed=[]
    def remove(_session, _gid, oid):
        removed.append(oid)
        scene['objects'][0]['quad']=deepcopy(scene['parents'][0]['quad'])
        return 'reset_child_to_mt_original'
    mod.core.remove_operator_geometry=remove
    assert editor.remove_hotkey(make_event(canvas,'r',8)) == 'break'
    assert removed==['o1']
    assert scene['objects'][0]['quad']==original_mt[0]['quad']
    assert scene['parents']==original_mt
    assert editor.selected=='o1'


def test_r_removes_only_manual_frame():
    mod, editor, canvas, host, scene, messages, corrected=setup()
    parent_before=deepcopy(scene['parents'])
    scene['objects'][0]['parent_id']=''
    editor.selected='o1';editor.mode='edit'
    editor.prepare_navigation=lambda:True
    editor.refresh_tree=lambda:None
    mod.core.remove_operator_geometry=lambda sess,gid,oid:(scene['objects'].clear() or 'remove_manual_plate')
    assert editor.remove_hotkey(make_event(canvas,'r',8)) == 'break'
    assert scene['objects']==[]
    assert scene['parents']==parent_before
    assert editor.selected == ''


def test_ctrl_r_is_ignored():
    mod,e,c,h,s,m,cor=setup()
    e.selected='o1';e.mode='edit'
    e.prepare_navigation=lambda:True
    e.refresh_tree=lambda:None
    touched=[]
    mod.core.remove_operator_geometry=lambda *args:touched.append(args)
    assert e.remove_hotkey(make_event(c,'r',4)) is None
    assert touched==[]


def test_t_only_toggles_roi_without_changing_review_data():
    mod,e,c,h,s,m,cor=setup()
    before=deepcopy(s)
    assert e.show_roi_overlay is True
    assert e.roi_hotkey(make_event(c,'t',8)) == 'break'
    assert e.show_roi_overlay is False
    e.overlay()
    assert not any('analyzed_roi' in op[2].get('tags',()) for op in c.ops)
    assert e.roi_hotkey(make_event(c,'t',8)) == 'break'
    assert e.show_roi_overlay is True
    assert s==before


def test_key_typed_in_gt_field_never_activates_r_or_t():
    mod,e,c,h,s,m,cor=setup()
    class TypedWidget: pass
    # `tk.Entry` typing is excluded from Canvas commands.
    root = None
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    try:
        field=tk.Entry(root)
        h.window=root
        assert e.remove_hotkey(make_event(field,'r',8)) is None
        assert e.roi_hotkey(make_event(field,'t',8)) is None
        assert e.show_roi_overlay is True
    finally:
        root.destroy()


def test_native_tk_bindtag_routes_r_and_t_from_tree_focus():
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    editor=None
    try:
        root.geometry('720x510+100+100')
        tree=ttk.Treeview(root,height=2)
        tree.insert('','end',iid='img',text='Obraz 19')
        tree.pack()
        canvas=tk.Canvas(root,width=540,height=350,takefocus=True)
        canvas.pack(fill='both',expand=True)
        root.update()
        mod, editor, c, host, scene, messages, corrected = setup(canvas)
        host.window=root
        editor._canvas_hotkey_pointer=lambda e:(35,35)
        editor.prepare_navigation=lambda:True
        editor.refresh_tree=lambda:None
        original_mt=deepcopy(scene['parents'][0]['quad'])
        editor.selected='o1'; editor.mode='edit'
        scene['objects'][0]['quad']=[[18,27],[52,30],[52,41],[20,40]]
        calls=[]
        def remove(sess,gid,oid):
            calls.append(oid)
            scene['objects'][0]['quad']=deepcopy(original_mt)
            return 'reset_child_to_mt_original'
        mod.core.remove_operator_geometry=remove
        editor.bindings()
        root.update()
        tree.focus_force(); root.update()
        if root.focus_get() is not tree:
            pytest.skip('Cannot focus Treeview in this Tk environment')
        tree.event_generate('<KeyPress-r>',state=8)
        root.update()
        assert calls==['o1']
        assert scene['objects'][0]['quad']==original_mt
        assert editor.show_roi_overlay is True
        tree.event_generate('<KeyPress-t>',state=8)
        root.update()
        assert editor.show_roi_overlay is False
    finally:
        if editor is not None:
            try: editor.dispose()
            except (AttributeError,tk.TclError):pass
        root.destroy()


def test_native_tk_one_right_mouse_press_changes_flag_once():
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    editor=None
    try:
        root.geometry('640x480+90+90')
        canvas=tk.Canvas(root,width=540,height=350,takefocus=True)
        canvas.pack()
        root.update()
        mod, editor, c, host, scene, messages, corrected=setup(canvas)
        host.window=root
        calls=[]
        def toggle(_session,_gid,oid):
            calls.append(oid)
            scene['objects'][0]['error'] = not scene['objects'][0]['error']
        mod.core.toggle_error=toggle
        editor.bindings()
        root.update()
        canvas.event_generate('<ButtonPress-3>',x=35,y=35)
        root.update()
        assert calls==['o1']
        assert editor.selected=='o1'
        assert scene['objects'][0]['error'] is True
        canvas.event_generate('<ButtonRelease-3>',x=35,y=35)
        root.update()
        assert calls==['o1']  # releasing must not toggle a second time
    finally:
        if editor is not None:
            try: editor.dispose()
            except (AttributeError,tk.TclError):pass
        root.destroy()
