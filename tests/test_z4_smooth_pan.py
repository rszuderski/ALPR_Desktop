"""Pointer-following pan and cancellation, exercised in real private Tk fixtures."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_one_navigator import gui,pick,frozen
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_pan as pan,mobile_canvas_zoom as zoom
from auto_annotation_tool.gui.mobile_canvas_visibility import contours

def begin(root,h,ed,button=1):
    h.zoom_image(4);ready(root,h,ed);native.focus_canvas(h)
    x,y=20,30
    h.canvas.event_generate(f'<ButtonPress-{button}>',x=x,y=y);root.update()
    assert h._view_drag_at and ed.drag is None
    return x,y,h._view_center,h._view_pixel_scale

def move(h,x,y,button=1):h.canvas.event_generate(f'<B{button}-Motion>',x=x,y=y)
def end(h,x,y,button=1):h.canvas.event_generate(f'<ButtonRelease-{button}>',x=x,y=y)

@pytest.mark.parametrize('button',[1,3])
def test_pointer_burst_coalesces_and_release_uses_exact_last_position(gui,monkeypatch,button):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed,button);before=frozen(s);selected=ed.selected
    calls=[];paint=h._paint_image
    def observed():calls.append(bool(getattr(h,'_zoom_preview',False)));return paint()
    monkeypatch.setattr(h,'_paint_image',observed)
    for delta in range(1,31):move(h,x+delta,y+delta,button)
    timer=h._smooth_pan.after_id
    assert not calls and h._view_center==center and timer
    for delta in range(31,61):move(h,x+delta,y+delta,button)
    assert h._smooth_pan.after_id==timer and not calls
    end(h,x+75,y+55,button)
    assert calls==[False] and h._view_center==pytest.approx((center[0]-75/scale,center[1]-55/scale))
    assert h._smooth_pan.after_id is None and not h._smooth_pan.active and not h._view_drag_at
    assert ed.selected==selected and frozen(s)==before and h._view_zoom==4

def test_preview_and_refinement_keep_reference_mapping_without_context_work(gui,monkeypatch):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed);before=frozen(s);frames=[];paint=h._paint_image
    bitmap=h.canvas.find_withtag('evidence_image');photo=h._photo
    def observed():
        paint();frames.append(bool(getattr(h,'_zoom_preview',False)))
        assert h.canvas.find_withtag('evidence_image')==bitmap and h._photo is photo
        items=contours(ed)
        for item,canvas_id in zip(items,h.canvas.find_withtag('navigator_geometry')):
            assert h.canvas.coords(canvas_id)==pytest.approx([v for p in item['quad'] for v in ed.to_canvas(p)])
    monkeypatch.setattr(h,'_paint_image',observed)
    monkeypatch.setattr(ed,'_render_contexts',lambda *a,**k:pytest.fail('Pan must not rebuild scientific context'))
    for dx,dy in ((30,15),(-30,-10),(45,20)):
        move(h,x+dx,y+dy);native.pump(root,lambda:h._smooth_pan.after_id is None)
    end(h,x+45,y+20)
    assert frames==[True,True,True,False] and frozen(s)==before

@pytest.mark.parametrize('ref,inf',[(True,True),(True,False),(False,True),(False,False)])
def test_pan_preserves_visibility_selection_and_review(gui,ref,inf):
    root,h,ed,s=gui;ed.ref_visible.set(ref);ed.inf_visible.set(inf)
    x,y,center,scale=begin(root,h,ed);before=frozen(s);selected=ed.selected
    move(h,x+45,y+20);end(h,x+45,y+20);ready(root,h,ed)
    assert (ed.ref_visible.get(),ed.inf_visible.get())==(ref,inf)
    assert len(h.canvas.find_withtag('operator_quad'))==int(ref)
    assert len(h.canvas.find_withtag('mt_original'))==int(inf)
    assert ed.selected==selected and frozen(s)==before

def test_reversal_and_boundaries_do_not_queue_old_positions(gui):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed)
    move(h,x+80,y+20);native.pump(root,lambda:h._smooth_pan.after_id is None)
    move(h,x-50,y-10);native.pump(root,lambda:h._smooth_pan.after_id is None)
    assert h._view_center==pytest.approx((center[0]+50/scale,center[1]+10/scale))
    move(h,x+100000,y+100000);native.pump(root,lambda:h._smooth_pan.after_id is None)
    half=(h.canvas.winfo_width()/(2*scale),h.canvas.winfo_height()/(2*scale))
    assert h._view_center==pytest.approx(half)
    end(h,x+5,y+5);assert h._view_center==pytest.approx((center[0]-5/scale,center[1]-5/scale))

@pytest.mark.parametrize('action',['image','resize','navigate','dispose','close'])
def test_pending_pan_cannot_leak_into_other_context(gui,action):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed);move(h,x+60,y+25)
    controller=h._smooth_pan;assert controller.after_id
    if action=='image':h._image_token+=1
    elif action=='resize':h.window.geometry('1180x760')
    elif action=='navigate':assert ed.prepare_navigation()
    elif action=='dispose':ed.dispose()
    elif action=='close':h._closed=True
    try:
        native.pump(root,lambda:controller.after_id is None)
        assert not controller.active and not h._view_drag_at
        if action!='resize':assert h._view_center==center
    finally:h._closed=False

def test_focus_loss_finishes_once_and_does_not_leave_preview(gui,monkeypatch):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed);move(h,x+60,y+25)
    calls=[];paint=h._paint_image
    monkeypatch.setattr(h,'_paint_image',lambda:(calls.append(bool(getattr(h,'_zoom_preview',False))),paint())[-1])
    h.canvas.event_generate('<FocusOut>');root.update()
    assert calls==[False] and not h._smooth_pan.active and h._smooth_pan.after_id is None
    assert h._view_center==pytest.approx((center[0]-60/scale,center[1]-25/scale))

def test_stationary_click_does_not_pan_or_force_bitmap_render(gui,monkeypatch):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed)
    monkeypatch.setattr(h,'_paint_image',lambda:pytest.fail('Stationary click must not repaint the bitmap'))
    end(h,x,y);assert h._view_center==center and not getattr(h._smooth_pan,'active',False)

def test_zoom_behavior_after_pan_and_while_dragging_is_preserved(gui):
    root,h,ed,s=gui;x,y,center,scale=begin(root,h,ed);before=frozen(s)
    move(h,x+60,y+25)
    zoom.request_zoom(h,1.25,(x,y));assert not getattr(h,'_smooth_zoom',None)
    end(h,x+60,y+25);anchor=(h.canvas.winfo_width()/2,h.canvas.winfo_height()/2);point=ed.to_source(*anchor)
    zoom.request_zoom(h,1.25,anchor);native.pump(root,lambda:not h._smooth_zoom.active)
    assert h._view_zoom==pytest.approx(5) and ed.to_source(*anchor)==pytest.approx(point) and frozen(s)==before

def test_corner_drag_is_not_deferred_pan(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');before=copy.deepcopy(ed._read()['parents']);q=contours(ed)[0]['quad']
    x,y=map(round,ed.to_canvas(q[0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag is not None
    h.canvas.event_generate('<B1-Motion>',x=x+3,y=y+2)
    assert not getattr(h,'_smooth_pan',None)
    h.canvas.event_generate('<ButtonRelease-1>',x=x+3,y=y+2);ready(root,h,ed)
    assert ed._read()['parents']==before and ed.drag is None
