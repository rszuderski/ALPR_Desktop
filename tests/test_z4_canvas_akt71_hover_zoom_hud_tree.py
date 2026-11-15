"""AKT71 interaction regression: hover+S, A fit/restoration, local metrics HUD, active row.

Uses synthetic scenes; does not modify mobile results, raw MT or persisted operator GT.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
import math
import tkinter as tk
from tkinter import ttk
import pytest
from test_z4_canvas_akt70_selection_drag import setup, event, key_event
from test_z4_canvas_linked_measurement_tree import fixture


def setup_zoom():
    mod,e,c,h,s,msg,_=setup()
    h._image.size=(100,100)
    h._view_zoom=2.0
    h._view_fit_scale=min((c.winfo_width()-16)/100,(c.winfo_height()-16)/100)
    h._view_pixel_scale=h._view_zoom*h._view_fit_scale
    h._view_center=(48.,53.)
    def actual_paint():
        h._view_fit_scale=max(.0001,min((c.winfo_width()-16)/100,(c.winfo_height()-16)/100))
        h._view_pixel_scale=h._view_fit_scale*h._view_zoom
        h.paints += 1
    h._paint_image=actual_paint
    e.to_canvas=mod.CanvasReviewEditor.to_canvas.__get__(e)
    e.to_source=mod.CanvasReviewEditor.to_source.__get__(e)
    e.plate_fit_previous=None
    return mod,e,c,h,s,msg


def test_hover_plus_s_selects_even_when_keypress_has_wrong_coordinates():
    mod,e,c,h,s,msg,_=setup()
    original=copy.deepcopy(s)
    c.pointer=(35,30)
    e.selection_key_press(key_event(c,x=-500,y=-50))
    assert e.selected=='o1' and e.mode=='edit'
    assert s==original
    c.after_cancel=lambda token:None
    e.selection_key_release(key_event(c))
    assert e.selection_key_down is False
    e.selection_key_press(key_event(c,x=0,y=0))
    assert e.selected=='' and e.mode=='select'
    assert s==original


def test_hover_s_works_with_focus_on_tree_when_mouse_is_over_canvas():
    mod,e,c,h,s,msg,_=setup()
    outside=SimpleNamespace(winfo_toplevel=lambda: h.window)
    c.pointer=(35,30)
    e.selection_key_press(SimpleNamespace(widget=outside,keysym='s',state=0,x=-99,y=-99))
    assert e.selected=='o1'
    c.after_cancel=lambda token:None
    e.selection_key_release(SimpleNamespace(widget=outside,keysym='s',state=0))
    assert not e.selection_key_down


def test_hover_outside_canvas_does_not_select_or_reuse_old_coordinates():
    mod,e,c,h,s,msg,_=setup()
    c.pointer=(800,801)
    e.pointer_canvas=(35,30)
    assert e.selection_key_press(key_event(c))=='break'
    assert e.selected==''
    c.pointer=(35,30)
    e.selection_key_press(key_event(c))
    assert e.selected=='o1'


def test_hotkey_from_entry_does_not_steal_typing():
    mod,e,c,h,s,msg,_=setup()
    fake_entry=mod.tk.Entry.__new__(mod.tk.Entry)
    # create an actual Entry under a real Tk to avoid tk's internal widget id
    try:
        window=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    try:
        entry=tk.Entry(window)
        h.window=window
        c.pointer=(35,30)
        assert e.selection_key_press(SimpleNamespace(widget=entry,keysym='s',state=0)) is None
        assert e.selected==''
    finally:
        window.destroy()


def test_a_fits_plate_with_visible_margin_and_second_a_restores_prior_view():
    mod,e,c,h,s,msg=setup_zoom()
    c.pointer=tuple(e.to_canvas((35,30)))
    prev=(h._view_zoom,h._view_center)
    e.fit_plate_toggle(key_event(c))
    assert e.plate_fit_active
    assert h._view_zoom > prev[0]
    points=[e.to_canvas(p) for p in s['objects'][0]['quad']]
    assert min(p[0] for p in points) >= 25
    assert max(p[0] for p in points) <= c.winfo_width()-25
    assert min(p[1] for p in points) >= 25
    assert max(p[1] for p in points) <= c.winfo_height()-25
    assert 'Dopasowano' in msg[-1]
    e.fit_plate_toggle(key_event(c))
    assert not e.plate_fit_active and e.plate_fit_previous is None
    assert h._view_zoom==prev[0] and h._view_center==prev[1]
    assert 'Przywrócono' in msg[-1]


def test_a_hovered_without_prior_s_selection_fits_and_keeps_gt_untouched():
    mod,e,c,h,s,msg=setup_zoom()
    before=copy.deepcopy(s)
    # actual mapped pointer after previous non-centered view
    c.pointer=tuple(e.to_canvas((35,30)))
    assert e.selected==''
    assert e.fit_plate_toggle(key_event(c))=='break'
    assert e.plate_fit_active and e.selected==''
    assert s==before


def test_a_when_focus_elsewhere_but_pointer_over_canvas():
    mod,e,c,h,s,msg=setup_zoom()
    other=SimpleNamespace(winfo_toplevel=lambda:h.window)
    c.pointer=tuple(e.to_canvas((35,30)))
    e.fit_plate_toggle(SimpleNamespace(widget=other,state=0,keysym='a'))
    assert e.plate_fit_active


def test_hud_is_anchored_above_corrected_frame_not_fixed_top_left():
    mod,e,c,h,s,msg,_=setup()
    e.selected='o1';e.mode='edit'
    e.to_canvas=lambda p:(p[0]*5+100,p[1]*4+80)
    e.overlay()
    badge=[op for op in c.ops if op[0]=='rect' and 'overlap_hud_badge' in op[2].get('tags',())]
    assert len(badge)==1
    left,top,right,bottom=badge[0][1]
    frame_top=min(e.to_canvas(p)[1] for p in s['objects'][0]['quad'])
    assert left > 10 and top > 20 and bottom <= frame_top
    assert right <= c.winfo_width()-7
    assert any(op[0]=='text' and 'Pokrycie' in op[2].get('text','') for op in c.ops)


def test_hud_updates_from_live_uncommitted_corner_position_without_saving_gt():
    mod,e,c,h,s,msg,_=setup()
    original=copy.deepcopy(s)
    calls=[]
    def measure(before,after,*size):
        calls.append(copy.deepcopy(after))
        return {'polygon_iou_pct':80.-len(calls),'nmce_pct':2.+len(calls)*.5}
    mod.core.compare_mt_pose_to_gt=measure
    e.selected='o1';e.mode='edit'
    e.overlay()
    assert len(calls)==1
    e.drag=dict(oid='o1',index=0,quad=copy.deepcopy(s['objects'][0]['quad']),before=copy.deepcopy(s['objects'][0]['quad']))
    e.drag['quad'][0]=[24.,27.]
    e.overlay()
    assert calls[-1][0]==[24.,27.]
    assert 'na żywo' in ' '.join(op[2].get('text','') for op in c.ops if op[0]=='text')
    assert s==original


def test_active_image_list_row_has_strong_tag_and_previous_is_cleared():
    mod,e,tree,s,events=fixture(4,with_alias=False)
    e.refresh_tree()
    first='subject-02'
    assert 'canvas_active' in tree.item(first,'tags')
    s.inventory['groups'] = s.inventory['groups']
    e.h.subject_key='key00'
    e.h.current_invocation='G00'
    e.sync_tree_selection()
    assert 'canvas_active' in tree.item('subject-00','tags')
    assert 'canvas_active' not in tree.item(first,'tags')
    assert tree.selection()==('subject-00',)


def test_tk_native_theme_highlight_after_focus_is_moved_to_canvas():
    try:
        window=tk.Tk()
    except tk.TclError:
        pytest.skip('No display')
    try:
        c=tk.Canvas(window,width=320,height=240)
        tree=ttk.Treeview(window,columns=('Nr',),height=2)
        tree.insert('','end',iid='im1',text='Obraz 1')
        tree.pack();c.pack()
        style=ttk.Style(window)
        style.configure('MobileReview.Observations.Treeview',rowheight=35)
        style.map('MobileReview.Observations.Treeview',
            background=[('selected','!focus','#135481'),('selected','#135481')],
            foreground=[('selected','!focus','#FFFFFF'),('selected','#FFFFFF')])
        tree.configure(style='MobileReview.Observations.Treeview')
        tree.selection_set('im1')
        c.focus_set()
        window.update()
        bg=style.lookup('MobileReview.Observations.Treeview','background',('selected','!focus'))
        fg=style.lookup('MobileReview.Observations.Treeview','foreground',('selected','!focus'))
        assert bg.lower()=='#135481' and fg.lower()=='#ffffff'
    finally:
        window.destroy()


def test_native_tk_tree_focus_hover_s_twice_and_a_twice():
    """Real Tk dispatch from Treeview should reach toplevel hover shortcuts."""
    try:
        window=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    try:
        window.geometry('840x660+50+50')
        tree=ttk.Treeview(window,height=2)
        tree.insert('','end',iid='test',text='Obraz 01')
        tree.pack()
        c=tk.Canvas(window,width=700,height=480,background='#142028')
        c.pack(fill='both',expand=True)
        window.update()
        mod,e,c,h,s,msg,_=setup(canvas=c)
        e.h.window=window
        h._image.size=(100,100)
        h._view_zoom=1.2
        h._view_fit_scale=1.0
        h._view_pixel_scale=1.0
        h._view_center=(50.,50.)
        e.to_canvas=mod.CanvasReviewEditor.to_canvas.__get__(e)
        e.to_source=mod.CanvasReviewEditor.to_source.__get__(e)
        e.plate_fit_previous=None
        # The synthetic scene is 100x100 but mouse hit must use true geometry.
        h._view_fit_scale=min((c.winfo_width()-16)/100,(c.winfo_height()-16)/100)
        h._view_pixel_scale=h._view_fit_scale*h._view_zoom
        def paint():
            h._view_fit_scale=min((c.winfo_width()-16)/100,(c.winfo_height()-16)/100)
            h._view_pixel_scale=h._view_fit_scale*h._view_zoom
            h.paints+=1
        h._paint_image=paint
        e.bindings()
        # The module-level polygon-containment test is intentionally stubbed
        # by the general-purpose unit fixture. Aim at an actual *edge* so the
        # native test exercises hit-testing without relying on that stub.
        ideal_x, ideal_y = e.to_canvas((35, 30))
        warp_x, warp_y = round(ideal_x), round(ideal_y)
        inside = False
        for _ in range(4):
            c.event_generate('<Motion>', x=int(warp_x), y=int(warp_y), warp=True)
            window.update()
            observed = e._pointer_for_selection(None)
            if observed is None:
                break  # OS/WM may reject synthetic pointer movement.
            if abs(observed[0]-ideal_x) <= 12 and abs(observed[1]-ideal_y) <= 4:
                inside = True
                break
            warp_x += round(ideal_x - observed[0])
            warp_y += round(ideal_y - observed[1])
        if not inside:
            pytest.skip('Native Tk pointer warp unavailable or outside test plate')
        # A real hit must still work; never fake the application's selection.
        hit = e.hit(*observed, badge=False)
        assert hit is not None and hit[0] == 'o1', (observed, hit)
        # focus_set() requests focus only if the test window is already active.
        # Windows often leaves focus_get() as None. Force focus for this native
        # key-routing test; no change to production focus management.
        window.lift()
        tree.focus_force()
        window.update()
        if window.focus_get() is not tree:
            pytest.skip('Window manager denied native Tk keyboard focus')
        tree.event_generate('<KeyPress-s>');window.update()
        assert e.selected=='o1' and e.mode=='edit'
        tree.event_generate('<KeyRelease-s>');window.update()
        tree.event_generate('<KeyPress-s>');window.update()
        assert e.selected==''
        tree.event_generate('<KeyRelease-s>');window.update()
        orig=(h._view_zoom,h._view_center)
        tree.event_generate('<KeyPress-a>');window.update()
        assert e.plate_fit_active and h._view_zoom>orig[0]
        tree.event_generate('<KeyRelease-a>');window.update()
        tree.event_generate('<KeyPress-a>');window.update()
        assert not e.plate_fit_active
        assert h._view_zoom==orig[0] and h._view_center==orig[1]
    finally:
        window.destroy()


def test_production_tree_style_is_applied_and_tags_are_configured():
    try:
        window=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display')
    try:
        from test_z4_canvas_linked_measurement_tree import load_editor
        mod=load_editor()
        parent=tk.Frame(window)
        parent.pack()
        frame=tk.Frame(parent)
        frame.pack()
        tree=ttk.Treeview(frame,columns=('Nr',),height=2)
        tree.pack()
        e=mod.CanvasReviewEditor.__new__(mod.CanvasReviewEditor)
        e.h=SimpleNamespace(subject_tree=tree,window=window)
        e.p=dict(surface='#F6F9FB',fg='#182532', red_bg='#FFE9EE',red='#AA1231',
                 green_bg='#DEF7E8',green='#105F35',purple_bg='#F4EAFE',purple='#7037A1')
        e.left_width=360
        e._configure_tree()
        assert tree.cget('style')=='MobileReview.Observations.Treeview'
        assert tree.tag_has('canvas_active')==()
        style=ttk.Style(window)
        maps=style.map('MobileReview.Observations.Treeview','background')
        assert any('#135481' in str(item).lower() for item in maps)
        assert style.lookup('MobileReview.Observations.Treeview','rowheight') in ('35',35)
    finally:
        window.destroy()
