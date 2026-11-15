"""Smooth viewport contracts on real Tk; all editing fixtures are private."""
import copy,math,time
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_one_navigator import gui,pick,frozen
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_zoom as zoom

@pytest.mark.parametrize('delta,num,expected',[(120,0,1.25),(-120,0,.8),(15,0,1.25**.125),(-15,0,1.25**-.125),
    (240,0,1.25**2),(0,4,1.25),(0,5,.8),(0,0,None),(float('nan'),0,None),(float('inf'),0,None)])
def test_wheel_fraction_and_direction(delta,num,expected):
    value=zoom.wheel_factor(NS(delta=delta,num=num))
    assert value is None if expected is None else value==pytest.approx(expected)

def settle(root,h):
    native.pump(root,lambda:not h._smooth_zoom.active,seconds=4)
    root.update_idletasks()

def start(root,h,factor=1.25,anchor=None):
    native.focus_canvas(h)
    anchor=anchor or (h.canvas.winfo_width()/2,h.canvas.winfo_height()/2)
    zoom.request_zoom(h,factor,anchor)

def test_native_intermediate_frames_anchor_and_exact_target(gui,monkeypatch):
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed);frozen_before=frozen(s)
    anchor=(h.canvas.winfo_width()*.57,h.canvas.winfo_height()*.43);point=ed.to_source(*anchor)
    frames=[];paint=h._paint_image
    def observed(*a,**kw):
        paint(*a,**kw);frames.append((h._view_zoom,getattr(h,'_zoom_preview',False),ed.to_source(*anchor)))
    monkeypatch.setattr(h,'_paint_image',observed)
    start(root,h,1.25,anchor);assert h._view_zoom==4  # Input handler does no raster work.
    settle(root,h)
    assert len(frames)>=3 and all(4<v[0]<=5 for v in frames)
    assert frames[-1][0]==pytest.approx(5) and not frames[-1][1]
    assert all(v[2]==pytest.approx(point,abs=1e-7) for v in frames)
    assert frozen(s)==frozen_before

def test_fractional_burst_accumulates_in_one_timer(gui):
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed)
    start(root,h,1.25**.125);timer=h._smooth_zoom.after_id
    for _ in range(7):zoom.request_zoom(h,1.25**.125,(h.canvas.winfo_width()/2,h.canvas.winfo_height()/2))
    assert h._smooth_zoom.after_id==timer and h._smooth_zoom.target==pytest.approx(5)
    settle(root,h);assert h._view_zoom==pytest.approx(5)

def test_reversal_reacts_to_current_view(gui):
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed);start(root,h,2)
    native.pump(root,lambda:h._view_zoom>4);current=h._view_zoom
    zoom.request_zoom(h,.8,(h.canvas.winfo_width()/2,h.canvas.winfo_height()/2))
    assert h._smooth_zoom.target==pytest.approx(current*.8)
    settle(root,h);assert h._view_zoom==pytest.approx(current*.8)

def test_zoom_does_not_rebuild_scientific_panel(gui,monkeypatch):
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed);before=frozen(s);context=ed.contexts.fields['object'].cget('text')
    def heavy(*a,**kw):pytest.fail('Viewport animation must not rebuild scientific diagnostics')
    # Focus recovery schedules a normal overlay. Complete that unrelated paint
    # before guarding the viewport animation, then restore the method for close.
    native.focus_canvas(h);root.update_idletasks()
    with monkeypatch.context() as guard:
        guard.setattr(ed,'_render_contexts',heavy)
        zoom.request_zoom(h,1.25,(h.canvas.winfo_width()/2,h.canvas.winfo_height()/2))
        settle(root,h)
    assert ed.contexts.fields['object'].cget('text')==context and frozen(s)==before

def test_fit_resize_and_image_change_cancel_pending_callback(gui):
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed);start(root,h);h.fit_image()
    assert not h._smooth_zoom.active and h._view_zoom==1 and h._smooth_zoom.after_id is None
    start(root,h);h._image_token+=1;settle(root,h);assert h._view_zoom==1
    start(root,h);h.window.geometry('1180x760');root.update();assert not h._smooth_zoom.active

def test_click_interrupts_before_corner_edit_and_drawing_keeps_points(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');h.zoom_image(2);ready(root,h,ed)
    native.key(h,'d');native.click(h,ed,(450,480));ready(root,h,ed);points=copy.deepcopy(ed.points)
    start(root,h);settle(root,h);assert ed.points==points
    start(root,h);h.canvas.event_generate('<ButtonPress-1>',x=30,y=30)
    assert not h._smooth_zoom.active and h._smooth_zoom.after_id is None
    h.canvas.event_generate('<ButtonRelease-1>',x=30,y=30);ed.escape()

def test_boundaries_and_zoom_out_from_A_scale(gui):
    root,h,ed,s=gui;h._view_zoom=100;h._paint_image();ready(root,h,ed)
    start(root,h,.8);settle(root,h);assert h._view_zoom==pytest.approx(80)
    start(root,h,1000);settle(root,h);assert h._view_zoom==zoom.MAX_ZOOM
    start(root,h,1e-6);settle(root,h);assert h._view_zoom==1

def test_entry_selection_and_source_geometry_survive_zoom(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');h.zoom_image(2);ready(root,h,ed)
    oid=ed.selected;ed.start_entry(oid);native.pump(root);assert ed.entry is not None
    entry=ed.entry;before=frozen(s)
    zoom.request_zoom(h,1.25,(h.canvas.winfo_width()/2,h.canvas.winfo_height()/2));settle(root,h)
    assert ed.entry is entry and entry.winfo_ismapped() and ed.selected==oid and frozen(s)==before

def test_close_cancels_timer_without_late_callback(gui):
    root,h,ed,s=gui;start(root,h);h._closed=True
    try:
        settle(root,h);assert not h._smooth_zoom.active and h._smooth_zoom.after_id is None
    finally:h._closed=False

def test_bitmap_is_reused_but_new_image_pixels_replace_old(gui):
    from PIL import Image,ImageTk
    root,h,ed,s=gui;h.zoom_image(4);ready(root,h,ed)
    item=h.canvas.find_withtag('evidence_image');photo=h._photo
    for _ in range(4):
        zoom.paint_viewport(h,preview=True)
        assert h.canvas.find_withtag('evidence_image')==item and h._photo is photo
    original=h._image
    try:
        h._image=Image.new('RGB',original.size,(220,10,20));h._paint_image()
        image=ImageTk.getimage(h._photo);assert image.getpixel((image.width//2,image.height//2))[:3]==(220,10,20)
        h._image=None;h._paint_image();assert not h.canvas.find_withtag('evidence_image') and h._photo is None
    finally:h._image=original;h._paint_image()

def test_sidebar_wheel_routes_before_starting_animation(gui,monkeypatch):
    root,h,ed,s=gui;h.window.geometry('1180x760');ready(root,h,ed);h.zoom_image(2)
    panel=h.review_sidebar_scroll;assert panel.winfo_ismapped()
    monkeypatch.setattr(h.window,'winfo_pointerxy',lambda:(panel.winfo_rootx()+20,panel.winfo_rooty()+30))
    before=h._view_zoom
    assert h._wheel_zoom(NS(delta=-120,num=0,x=20,y=30,widget=h.canvas))=='break'
    assert h._view_zoom==before and not getattr(h,'_smooth_zoom',None)
