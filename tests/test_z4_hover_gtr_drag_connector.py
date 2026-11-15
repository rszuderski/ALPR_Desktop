"""GTR-first S selection and immediate, view-only corner preview."""
import copy,math
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_gtr_reapproval_gui import gui
from test_z4_gtr_reapproval import G,OID,pid,pending
from test_z4_frame_stacks import ready
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks

def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())
def hover_s(monkeypatch,h,ed,q):
    x,y=ed.to_canvas(tuple(sum(p[i] for p in q)/4 for i in (0,1)))
    ed.hover_motion(NS(x=x,y=y));monkeypatch.setattr(ed,'_canvas_hotkey_pointer',lambda event=None:(x,y))
    native.key(h,'s');return x,y

@pytest.mark.parametrize('kind',['POSE','BBOX'])
def test_S_over_GTR_activates_GTR_even_when_same_OID_source_layer_is_selected(gui,monkeypatch,kind):
    root,h,ed,s=gui;nav.select_layer(ed,kind,OID);ready(root,h,ed);before=frozen(s)
    hover_s(monkeypatch,h,ed,clean.selected(ed)['gt']);ready(root,h,ed)
    assert ed.selected==OID and ed.mode=='edit' and layers.model(ed)['active']=='GTR'
    assert len(ed.c.find_withtag('corner_handle'))==4 and frozen(s)==before
    native.key(h,'s');ready(root,h,ed)
    assert not ed.selected and ed.mode=='select' and frozen(s)==before

def test_S_prefers_independent_GTR_inside_unlinked_model_over_selected_MT(tmp_path,monkeypatch):
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path/'data');q=core.scene(s,G)['parents'][0]['quad'];cx,cy=(sum(p[i] for p in q)/4 for i in (0,1))
    inner=[[(x+cx)/2,(y+cy)/2] for x,y in q];manual=core.add_plate(s,G,inner,partial_review=True)
    monkeypatch.setattr(native,'source_session',lambda _:s);gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);nav.select_layer(ed,'POSE',OID);ready(root,h,ed);before=frozen(s)
        hover_s(monkeypatch,h,ed,inner);ready(root,h,ed)
        assert ed.selected==manual and ed.mode=='edit' and clean.selected(ed)['pid']==stacks.snapshot(s,G)['links'][manual]['plate_id']
        assert len(ed.c.find_withtag('corner_handle'))==4 and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass

def grab(root,h,ed):
    nav.select_layer(ed,'GTR',OID);ready(root,h,ed)
    p=clean.selected(ed)['gt'][0];x,y=map(round,ed.to_canvas(p))
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag
    return x,y

def test_corner_moves_on_motion_before_idle_and_without_writing(gui):
    root,h,ed,s=gui;x,y=grab(root,h,ed);before=frozen(s)
    ed.c.event_generate('<B1-Motion>',x=x+14,y=y+6)
    shape=next(i for i in ed.c.find_withtag('navigator_geometry') if 'GTR' in ed.c.gettags(i))
    expected=ed.to_canvas(ed.drag['quad'][0])
    assert math.dist(ed.c.coords(shape)[:2],expected)<.01
    handles=ed.c.find_withtag('corner_handle');l,t,r,b=ed.c.coords(handles[0])
    assert math.dist(((l+r)/2,(t+b)/2),expected)<.01 and frozen(s)==before

def test_drag_motion_does_not_rebuild_overlay_or_diagnostics(gui,monkeypatch):
    root,h,ed,s=gui;x,y=grab(root,h,ed);ready(root,h,ed)
    monkeypatch.setattr(clean,'paint',lambda _:pytest.fail('Full overlay in corner motion'))
    monkeypatch.setattr(stacks,'diagnostics',lambda *a,**k:pytest.fail('Diagnostics rebuilt in corner motion'))
    before=frozen(s)
    for delta in (4,8,12):
        ed.c.event_generate('<B1-Motion>',x=x+delta,y=y+2);root.update_idletasks()
    assert frozen(s)==before

@pytest.mark.parametrize('finish',['release','escape'])
def test_fast_preview_commits_once_on_release_or_discards_on_Escape(gui,finish):
    root,h,ed,s=gui;before=frozen(s);parents=copy.deepcopy(core.scene(s,G)['parents']);identity=pid(s)
    x,y=grab(root,h,ed);ed.c.event_generate('<B1-Motion>',x=x+8,y=y+3)
    moved=copy.deepcopy(ed.drag['quad']);assert frozen(s)==before
    if finish=='release':
        revision=s.review.review_revision;ed.c.event_generate('<ButtonRelease-1>',x=x+8,y=y+3);ready(root,h,ed)
        assert s.review.review_revision==revision+1 and pending(s)==[identity]
        assert stacks.snapshot(s,G)['plates'][identity]['gt_quad']==moved
    else:
        ed.escape();ready(root,h,ed);assert frozen(s)==before and not ed.selected
    assert core.scene(s,G)['parents']==parents

@pytest.mark.parametrize('kind',['GTR','POSE','BBOX'])
def test_tether_connector_marks_exact_contact_with_its_own_contour(gui,kind):
    root,h,ed,s=gui;nav.select_layer(ed,kind,OID);ready(root,h,ed);before=frozen(s)
    circles=ed.c.find_withtag('badge_tether_connector');assert circles
    for circle in circles:
        tags=ed.c.gettags(circle);oid=next(o['object_id'] for o in core.scene(s,G)['objects'] if o['object_id'] in tags)
        l,t,r,b=ed.c.coords(circle);center=((l+r)/2,(t+b)/2)
        lines=[i for i in ed.c.find_withtag('badge_tether_line') if oid in ed.c.gettags(i)]
        assert any(math.dist(ed.c.coords(line)[-2:],center)<.01 for line in lines)
        assert 4<=r-l<=9 and 4<=b-t<=9
    assert frozen(s)==before

def test_connector_follows_corner_preview_and_clears_for_drawing(gui):
    root,h,ed,s=gui;x,y=grab(root,h,ed)
    ed.c.event_generate('<B1-Motion>',x=x+11,y=y-6)
    line=ed.c.find_withtag('badge_tether_line')[0];circle=ed.c.find_withtag('badge_tether_connector')[0]
    l,t,r,b=ed.c.coords(circle)
    assert math.dist(ed.c.coords(line)[-2:],((l+r)/2,(t+b)/2))<.01
    ed.escape();ready(root,h,ed);native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='draw' and not ed.c.find_withtag('badge_tether_connector')
