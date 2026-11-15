"""First stored GT corner flashes on selection, including collision redirection."""
import copy
import time
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_click_stack_priority import gui
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_candidate_stacks as data
G='mt-one';OID='GT:a0'
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())

@pytest.mark.parametrize('action',['menu','x','z','click','D','collision'])
def test_GTR_activation_flashes_vertex_one_without_review_write(gui,action,monkeypatch):
    root,h,ed,s=gui;native.focus_canvas(h)
    nav.select_layer(ed,'BBOX' if action=='z' else 'POSE');ready(root,h,ed)
    trace=[];draw=ed.corner_pulse.draw
    def observed():
        draw()
        if ed.corner_pulse.green and ed.corner_pulse.target:
            items=ed.c.find_withtag('gtr_vertex_1_green')
            if items:trace.append((ed.c.coords(items[0]),ed.to_canvas(clean.selected(ed)['gt'][0])))
    monkeypatch.setattr(ed.corner_pulse,'draw',observed);before=frozen(s)
    if action=='menu':ed.navigator.menu.invoke(2)
    elif action in ('x','z','D'):native.key(h,action.lower())
    elif action=='click':native.click(h,ed,(160,150))
    else:
        ed.escape();ready(root,h,ed);native.key(h,'d');ready(root,h,ed);native.click(h,ed,(160,150))
    ready(root,h,ed)
    if action in ('D','collision'):native.pump(root,lambda:bool(trace),seconds=3)
    assert trace and ed.selected==OID and frozen(s)==before
    for (l,t,r,b),(x,y) in trace:
        assert ((l+r)/2,(t+b)/2)==pytest.approx((x,y))
    if action in ('D','collision'):assert ed.gt_focus_id
    else:assert not ed.gt_focus_id

def test_pulse_expires_and_viewport_does_not_restart_it(gui):
    root,h,ed,s=gui;native.focus_canvas(h);before=frozen(s)
    nav.select_layer(ed,'GTR');pulse=ed.corner_pulse;generation=pulse.generation
    h._view_center=(h._view_center[0]+4,h._view_center[1]+3)
    from auto_annotation_tool.gui.mobile_canvas_zoom import paint_viewport
    paint_viewport(h,preview=True)
    assert pulse.generation==generation
    if pulse.green:
        l,t,r,b=ed.c.coords(ed.c.find_withtag('gtr_vertex_1_green')[0]);x,y=ed.to_canvas(clean.selected(ed)['gt'][0])
        assert ((l+r)/2,(t+b)/2)==pytest.approx((x,y))
    deadline=time.monotonic()+.9;native.pump(root,lambda:time.monotonic()>=deadline,seconds=3)
    assert pulse.target is None and not pulse.after_ids and not ed.c.find_withtag('gtr_corner_pulse')
    generation=pulse.generation;paint_viewport(h,preview=True)
    assert pulse.generation==generation and pulse.target is None and frozen(s)==before

@pytest.mark.parametrize('action',['layer','Escape','draw','source_change','dispose'])
def test_pending_pulse_cannot_survive_leaving_its_context(gui,action):
    root,h,ed,s=gui;native.focus_canvas(h);nav.select_layer(ed,'GTR');assert ed.corner_pulse.after_ids
    before=frozen(s);entry=h._image_entry
    try:
        if action=='layer':nav.select_layer(ed,'POSE');ed.overlay()
        elif action=='Escape':ed.escape()
        elif action=='draw':ed.escape();native.key(h,'d');ed.overlay();assert ed.mode=='draw'
        elif action=='source_change':h._image_entry='other-source';ed.overlay()
        else:ed.dispose()
        assert ed.corner_pulse.target is None and not ed.corner_pulse.after_ids and not ed.c.find_withtag('gtr_corner_pulse')
        assert frozen(s)==before
    finally:h._image_entry=entry

def test_vertex_one_is_stored_GT_index_zero_not_a_bbox_corner(gui):
    root,h,ed,s=gui;q=[[260,155],[260,195],[190,195],[190,155]]
    data.edit_gt(s,G,OID,quad=q);ready(root,h,ed);native.focus_canvas(h);before=frozen(s)
    nav.select_layer(ed,'GTR');ed.corner_pulse.draw()
    l,t,r,b=ed.c.coords(ed.c.find_withtag('gtr_vertex_1_green')[0])
    assert ((l+r)/2,(t+b)/2)==pytest.approx(ed.to_canvas(q[0])) and frozen(s)==before

@pytest.mark.parametrize('gui',['manual'],indirect=True)
def test_independent_manual_GTR_also_gets_vertex_one_cue(gui):
    root,h,ed,s=gui;native.focus_canvas(h);before=frozen(s)
    row=next(r for r in clean.records(ed) if r['gt'] and not r['candidate'])
    ed._choose_plate(row['oid']);ed.overlay()
    assert ed.c.find_withtag('gtr_vertex_1_green') and frozen(s)==before

def test_hidden_REF_keeps_flags_and_still_marks_vertex_one_briefly(gui):
    root,h,ed,s=gui;native.focus_canvas(h);ed.ref_visible.set(False);ed.inf_visible.set(False);before=frozen(s)
    nav.select_layer(ed,'GTR');ed.overlay()
    assert ed.c.find_withtag('gtr_vertex_1_green') and not ed.c.find_withtag('operator_quad')
    assert not ed.ref_visible.get() and not ed.inf_visible.get() and frozen(s)==before
