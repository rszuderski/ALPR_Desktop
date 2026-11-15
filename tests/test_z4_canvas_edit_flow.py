"""One-step Escape, stable viewport, isolated drawing, sequential cues."""
import copy,time
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_draw_shift_gtz_fix import gui
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_reservation as reservation
from auto_annotation_tool.ranking import mobile_manual_frame_guard as guard
G='mt-one';OID='GT:a0'
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())
def view(h,ed):
    return (ed.c.winfo_width(),ed.c.winfo_height(),h._view_zoom,h._view_pixel_scale,h._view_center,ed.to_canvas((150,140)),ed.to_canvas((300,210)))

@pytest.mark.parametrize('initial',['selection','focus','entry','corner','draw'])
def test_one_Escape_leaves_entire_frame_and_cancels_pending_work(gui,initial):
    root,h,ed,s=gui;native.focus_canvas(h);nav.select_layer(ed,'GTR');ready(root,h,ed)
    if initial in ('focus','entry'):ed.draw_toggle();ready(root,h,ed)
    if initial=='entry':ed.start_entry(OID);ready(root,h,ed);ed.entry_var.set('UNSAVED')
    elif initial=='corner':
        x,y=map(round,ed.to_canvas(ed._read()['objects'][0]['quad'][0]));ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
        ed.c.event_generate('<B1-Motion>',x=x+10,y=y+5)
        assert ed.drag['quad']!=ed.drag['before']
    elif initial=='draw':ed.escape();ed.draw_toggle();ready(root,h,ed)
    before=frozen(s)
    widget=ed.entry if ed.entry is not None else ed.c
    widget.event_generate('<KeyPress-Escape>')
    if widget.winfo_exists():widget.event_generate('<KeyRelease-Escape>')
    ready(root,h,ed)
    assert ed.mode=='select' and not ed.selected and not ed.gt_focus_id and ed.entry is None
    assert not ed.points and ed.drag is None and ed.draw_target is None and ed._reserved_edit is None
    assert not ed.corner_pulse.after_ids and ed.corner_pulse.pending is None and not ed.collision_warning.after_ids
    assert frozen(s)==before

@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_enter_leave_and_first_point_preserve_canvas_and_mapping(gui,size):
    root,h,ed,s=gui;h.window.state('normal');h.window.geometry(size+'+10+10');ready(root,h,ed)
    h.zoom_image(2);h._view_center=(350,250);h._paint_image();ready(root,h,ed);before=frozen(s)
    nav.select_layer(ed,'GTR');ready(root,h,ed);baseline=view(h,ed)
    ed.draw_toggle();ready(root,h,ed);assert view(h,ed)==baseline
    ed.escape();ready(root,h,ed);assert view(h,ed)==baseline
    ed.draw_toggle();ready(root,h,ed);assert ed.mode=='draw' and view(h,ed)==baseline
    native.click(h,ed,(350,250));ready(root,h,ed);assert len(ed.points)==1 and view(h,ed)==baseline
    ed.escape();ready(root,h,ed);assert view(h,ed)==baseline and frozen(s)==before

@pytest.mark.parametrize('gui',['many'],indirect=True)
def test_focused_GT_hides_unrelated_and_inferred_geometry(gui):
    root,h,ed,s=gui;before=frozen(s);nav.select_layer(ed,'GTR');ready(root,h,ed)
    assert len(ed.c.find_withtag('navigator_geometry'))>1
    ed.draw_toggle();ready(root,h,ed)
    items=ed.c.find_withtag('navigator_geometry')
    assert len(items)==1 and 'GTR' in ed.c.gettags(items[0]) and OID in ed.c.gettags(items[0])
    ed.escape();ready(root,h,ed);assert len(ed.c.find_withtag('navigator_geometry'))>1 and frozen(s)==before

@pytest.mark.parametrize('gui',['many'],indirect=True)
def test_drawing_suppresses_Shift_and_removes_every_legacy_overlay(gui):
    root,h,ed,s=gui;ed.escape();ed.draw_toggle();ready(root,h,ed);before=frozen(s)
    bitmap=ed.c.find_withtag('evidence_image')
    ed.c.create_rectangle(20,20,100,100,outline='red',tags=('legacy_unknown_box',))
    ed.c.create_polygon(50,50,80,50,80,80,50,80,fill='',outline='yellow',tags=('old_pose_renderer',))
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert not ed.shift_preview.active and not ed.c.find_withtag('shift_source_preview')
    assert not ed.c.find_withtag('legacy_unknown_box') and not ed.c.find_withtag('old_pose_renderer')
    assert ed.c.find_withtag('evidence_image')==bitmap
    for i in ed.c.find_all():
        assert ed.c.type(i)=='image' or set(ed.c.gettags(i)) & {'gtr_guides','drawing_line','drawing_corner'}
    ed.escape();ready(root,h,ed);ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert len(ed.c.find_withtag('shift_source_preview'))==8 and frozen(s)==before

def test_background_refresh_waits_while_D_has_no_points(gui,monkeypatch):
    root,h,ed,s=gui;ed.escape();ed.draw_toggle();ready(root,h,ed)
    with monkeypatch.context() as patch:
        patch.setattr(h,'_change',lambda *a,**kw:pytest.fail('Reload during drawing'))
        ed.refresh_results();assert ed.refresh_after is not None and ed.mode=='draw'
    ed.escape()

def test_red_warning_finishes_before_green_vertex_starts(gui,monkeypatch):
    root,h,ed,s=gui;native.focus_canvas(h);ed.escape();ed.draw_toggle();ready(root,h,ed);before=frozen(s)
    events=[]
    for obj,name in ((ed.collision_warning,'paint'),(ed.corner_pulse,'draw')):
        original=getattr(obj,name)
        def observed(original=original):
            original();events.append((time.monotonic(),ed.collision_warning.flashing,ed.collision_warning.phase,ed.corner_pulse.green,bool(ed.c.find_withtag('gtr_vertex_1_green'))))
        monkeypatch.setattr(obj,name,observed)
    native.click(h,ed,(160,150));ready(root,h,ed)
    try:native.pump(root,lambda:any(e[4] for e in events),seconds=3)
    except AssertionError:pytest.fail(str(dict(events=events,pending=ed.corner_pulse.pending,selected=ed.selected,focus=ed.gt_focus_id,warning_active=ed.collision_warning.active)))
    native.pump(root,lambda:ed.corner_pulse.target is None,seconds=3)
    red=[e for e in events if e[1] and e[2]=='red'];green=[e for e in events if e[4]]
    assert red and green and max(e[0] for e in red)<min(e[0] for e in green)
    assert not any(e[1] and e[3] for e in events)
    assert not ed.c.find_withtag('gtr_collision_warning') and frozen(s)==before

def test_Escape_cancels_green_waiting_for_red_sequence(gui):
    root,h,ed,s=gui;before=frozen(s)
    reference=guard.find_point_collision(ed._read(),(160,150),session=s,gid=G)
    reservation.handle(ed,reference)
    assert ed.corner_pulse.pending is not None and not ed.corner_pulse.green
    ed.escape();ready(root,h,ed);deadline=time.monotonic()+1.3
    native.pump(root,lambda:time.monotonic()>=deadline,seconds=3)
    assert not ed.selected and not ed.c.find_withtag('gtr_corner_pulse') and not ed.collision_warning.active
    assert ed.corner_pulse.pending is None and not ed.corner_pulse.after_ids and frozen(s)==before

def test_stale_source_does_not_fall_back_to_other_frames_during_D(gui,monkeypatch):
    root,h,ed,s=gui;ed.escape();ed.draw_toggle();ready(root,h,ed);entry=h._image_entry
    try:
        h._image_entry='old-frame';ed.c.create_rectangle(10,10,80,80,tags=('stale_box',))
        with monkeypatch.context() as patch:
            patch.setattr(h,'_paint_source_frame_overlays',lambda *a:pytest.fail('Legacy frames during D'))
            assert ed.paint(ed.c.winfo_width(),ed.c.winfo_height())
        assert not ed.c.find_withtag('stale_box')
    finally:h._image_entry=entry

@pytest.mark.parametrize('gui',['many'],indirect=True)
def test_new_GTR_selection_removes_stale_red_before_green(gui):
    root,h,ed,s=gui;before=frozen(s)
    reference=guard.find_point_collision(ed._read(),(160,150),session=s,gid=G)
    reservation.handle(ed,reference);assert ed.c.find_withtag('gtr_collision_warning')
    from auto_annotation_tool.gui.mobile_canvas_clean_view import records
    row=next(r for r in records(ed) if r['gt'] and not r['candidate'])
    ed._choose_plate(row['oid']);ed.corner_pulse.trigger()
    assert ed.corner_pulse.green and not ed.c.find_withtag('gtr_collision_warning') and frozen(s)==before
