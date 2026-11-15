"""Reserved GT/source stacks redirect new drawing; layer count never covers UI."""
import copy
from types import SimpleNamespace as NS
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_mt_geometry_layers import source_session
from test_z4_frame_stacks import ready
from test_z4_gtr_collisions_replace import advance
from test_z4_badge4_hud2 import bbox as bbox_gui
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_manual_frame_guard as guard,mobile_mt_geometry_layers as evidence,mobile_canvas_partial_review as partial
G='mt-one';OID='GT:a0';GT=[[190,155],[260,155],[260,195],[190,195]]
def rect(l,t,r,b):return [[l,t],[r,t],[r,b],[l,b]]
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())
def state(s):return core.scene(s,G)
def seeded(path,linked=True):
    s=source_session(path,n=1)
    oid=evidence.create_reference(s,G,OID,GT) if linked else core.add_plate(s,G,GT,partial_review=True)
    data.edit_gt(s,G,oid,text='SAVEDGT');return s,oid
@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    s,oid=seeded(tmp_path/'data',linked=getattr(request,'param',True))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen);ready(root,h,ed)
    try:yield root,h,ed,s,oid
    finally:
        try:next(gen)
        except StopIteration:pass
def start_new(root,h,ed):ed.escape();ready(root,h,ed);native.key(h,'d');ready(root,h,ed);assert ed.mode=='draw'

def test_source_area_outside_GT_is_reserved_without_changing_links(tmp_path):
    s,oid=seeded(tmp_path);before=frozen(s);point=(160,150)
    assert not guard.contains(GT,point)
    hit=guard.find_point_collision(state(s),point,session=s,gid=G)
    assert hit and {c['kind'] for c in hit['contours']}=={'GTR','BBOX','POSE'}
    with pytest.raises(guard.GTRCollisionError):core.add_plate(s,G,rect(155,142,175,152),partial_review=True)
    assert frozen(s)==before

def test_unlinked_manual_GT_reserves_model_without_automatic_association(tmp_path):
    s,oid=seeded(tmp_path,linked=False);before=copy.deepcopy(data.snapshot(s,G)['links'])
    assert guard.find_point_collision(state(s),(160,150),session=s,gid=G)
    assert data.snapshot(s,G)['links']==before and before[OID]['plate_id'] is None

def test_no_GT_does_not_reserve_any_inferred_area(tmp_path):
    s=source_session(tmp_path,n=1);assert not guard.find_point_collision(state(s),(160,150),session=s,gid=G)
    evidence.create_reference(s,G,OID,GT)
    assert len(data.snapshot(s,G)['plates'])==1

@pytest.mark.parametrize('action',['edit','accept'])
def test_other_GT_creation_APIs_cannot_bypass_reservation(tmp_path,action):
    s,oid=seeded(tmp_path,linked=False);before=frozen(s)
    with pytest.raises(guard.GTRCollisionError):
        if action=='edit':data.edit_gt(s,G,OID,quad=rect(155,142,175,152))
        else:data.accept_reference(s,G,OID)
    assert frozen(s)==before

def test_any_positive_GT_intersection_including_crossing_is_rejected(tmp_path):
    s,oid=seeded(tmp_path);before=frozen(s)
    crossing=rect(170,170,280,180)
    assert all(not guard.contains(GT,p) for p in crossing)
    with pytest.raises(guard.GTRCollisionError):guard.require_free_manual_area(state(s),crossing,session=s,gid=G)
    assert frozen(s)==before

def test_REMOVED_manual_GT_frees_source_area(tmp_path):
    s,oid=seeded(tmp_path,linked=False);core.remove_operator_geometry(s,G,oid)
    assert guard.find_point_collision(state(s),(160,150),session=s,gid=G) is None
    core.add_plate(s,G,GT,partial_review=True)

def test_multiple_existing_GT_in_one_source_area_require_explicit_choice(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=1)
    first=core.add_plate(s,G,GT,partial_review=True)
    second=core.add_plate(s,G,rect(650,350,800,420),partial_review=True)
    # Seed a legacy duplicate in this private fixture. The current edit API
    # correctly refuses moving a separate GT into another reserved source area.
    changed=copy.deepcopy(s.review);scene=changed.invocation_annotations[G][core.KEY]
    pid=data.snapshot(s,G)['links'][second]['plate_id']
    scene[data.KEY]['plates'][pid]['gt_quad']=rect(270,155,290,185);s._save_change(changed)
    hit=guard.find_point_collision(state(s),(160,150),session=s,gid=G)
    assert len(hit['choices'])==2
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);start_new(root,h,ed);before=frozen(s)
        native.click(h,ed,(160,150));ready(root,h,ed)
        assert ed.mode=='select' and not ed.points and not ed.selected and not ed.gt_focus_id
        assert len(ed.collision_warning.active['choices'])==2 and 'Wybierz' in h.status_var.get()
        assert frozen(s)==before and len(data.snapshot(s,G)['plates'])==2
        assert ed._choose_plate(first);ready(root,h,ed)
        assert clean.selected(ed)['pid']==data.snapshot(s,G)['links'][first]['plate_id']
        assert frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass


@pytest.mark.parametrize('ref,inf',[(True,True),(True,False),(False,True),(False,False)])
def test_NEW_collision_edits_same_GT_and_flashes_all_layers(gui,ref,inf,monkeypatch):
    root,h,ed,s,oid=gui;start_new(root,h,ed);ed.ref_visible.set(ref);ed.inf_visible.set(inf);ed.overlay();before=frozen(s)
    phases=[];paint=ed.collision_warning.paint
    def observed():
        paint();phases.append(ed.collision_warning.phase)
    monkeypatch.setattr(ed.collision_warning,'paint',observed)
    native.click(h,ed,(160,150));ready(root,h,ed)
    assert ed.mode=='edit' and not ed.points and clean.selected(ed)['plate']['gt_text']=='SAVEDGT'
    assert ed.gt_focus_id==ed.selected and len(h.canvas.find_withtag('corner_handle'))==4
    assert {f['kind'] for f in ed.collision_warning.frames()}=={'GTR','BBOX','POSE'}
    assert 'red' in phases and len(h.canvas.find_withtag('gtr_collision_warning'))==4
    assert (ed.ref_visible.get(),ed.inf_visible.get())==(ref,inf) and frozen(s)==before
    assert len(data.snapshot(s,G)['plates'])==1

@pytest.mark.parametrize('gui',[False],indirect=True)
def test_D_on_unlinked_occupied_source_redirects_without_creating_link(gui):
    root,h,ed,s,oid=gui;before=frozen(s);nav.select_layer(ed,'BBOX');ready(root,h,ed)
    native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='edit' and clean.selected(ed)['plate']['gt_text']=='SAVEDGT'
    # Current spatial contract exposes the source immediately, but D remains
    # read-only and does not persist a canonical relation by itself.
    selected=clean.selected(ed)
    assert selected['candidate']['role']=='FINAL' and selected['candidate']['canonical_link']['plate_id'] is None
    assert frozen(s)==before

def test_timer_expires_but_edit_preview_is_usable_until_Escape(gui):
    root,h,ed,s,oid=gui;start_new(root,h,ed);ed.ref_visible.set(False);ed.inf_visible.set(False);ed.overlay()
    before=frozen(s);native.click(h,ed,(160,150));ready(root,h,ed)
    advance(root,2.7)
    assert not ed.collision_warning.active and not ed.collision_warning.after_ids
    assert h.canvas.find_withtag('operator_quad') and len(h.canvas.find_withtag('corner_handle'))==4
    native.key(h,'Escape');ready(root,h,ed)
    assert not h.canvas.find_withtag('operator_quad') and not ed.ref_visible.get() and frozen(s)==before

def test_D_edits_existing_GT_without_replacing_its_identity(gui):
    root,h,ed,s,oid=gui;nav.select_layer(ed,'GTR');ready(root,h,ed);before=data.snapshot(s,G);pid=next(iter(before['plates']))
    native.key(h,'d');ready(root,h,ed);assert ed.mode=='edit' and ed.draw_target is None
    x,y=map(round,ed.to_canvas(clean.selected(ed)['gt'][0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
    h.canvas.event_generate('<B1-Motion>',x=x+2,y=y+1)
    h.canvas.event_generate('<ButtonRelease-1>',x=x+2,y=y+1)
    ready(root,h,ed);after=data.snapshot(s,G)
    assert set(after['plates'])=={pid} and after['plates'][pid]['gt_text']=='SAVEDGT'
    assert after['plates'][pid]['gt_quad']!=before['plates'][pid]['gt_quad'] and after['links']==before['links']

@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_counter_is_three_available_layers_and_covers_nothing(gui,kind):
    root,h,ed,s,oid=gui;nav.select_layer(ed,kind);ready(root,h,ed);before=frozen(s);counter=ed._stack_counter
    assert counter and counter['count']==3 and len(h.canvas.find_withtag('stack_count_text'))==1
    assert h.canvas.itemcget(h.canvas.find_withtag('stack_count_text')[0],'text')=='3'
    obstacles=ed.clean_geometry_boxes+[ed._context_badge['rect']]
    assert not any(clean.intersects(counter['rect'],b) for b in obstacles)
    x,y,r,b=counter['rect'];assert 0<=x<r<=h.canvas.winfo_width() and 0<=y<b<=h.canvas.winfo_height()
    assert frozen(s)==before

def test_counter_includes_unsaved_seed_but_keeps_independent_GT_separate(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=1);monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);assert ed._stack_counter['count']==3
        assert not data.snapshot(s,G)['plates'] # The third layer is an unsaved starting outline.
        oid=core.add_plate(s,G,rect(650,350,800,420),partial_review=True);ed._choose_plate(oid);ready(root,h,ed)
        assert ed._stack_counter['count']==1
        ed.escape();ready(root,h,ed);assert ed._stack_counter is None and not h.canvas.find_withtag('stack_count_badge')
    finally:
        try:next(gen)
        except StopIteration:pass

def test_counter_hides_with_its_active_layer(gui):
    root,h,ed,s,oid=gui;nav.select_layer(ed,'POSE');ready(root,h,ed)
    ed.inf_visible.set(False);ed.overlay();assert not ed._stack_counter and not h.canvas.find_withtag('stack_count_badge')
    nav.select_layer(ed,'GTR');ready(root,h,ed);assert ed._stack_counter['count']==3

def test_reservation_selection_and_frame_edit_do_not_mark_model_BAD(gui):
    root,h,ed,s,oid=gui;parents=copy.deepcopy(state(s)['parents']);start_new(root,h,ed)
    native.click(h,ed,(160,150));ready(root,h,ed);q=clean.selected(ed)['gt'];x,y=map(round,ed.to_canvas(q[0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);h.canvas.event_generate('<B1-Motion>',x=x+2,y=y+1);h.canvas.event_generate('<ButtonRelease-1>',x=x+2,y=y+1);ready(root,h,ed)
    assert state(s)['parents']==parents and not core._object(state(s),OID)['error'] and len(data.snapshot(s,G)['plates'])==1

def test_INVALID_QUAD_reserves_only_real_bbox_and_GT(bbox_gui):
    root,h,ed,s=bbox_gui;data.edit_gt(s,G,OID,quad=GT);ready(root,h,ed)
    start_new(root,h,ed);before=frozen(s);native.click(h,ed,(160,150));ready(root,h,ed)
    assert ed.mode=='edit' and {f['kind'] for f in ed.collision_warning.frames()}=={'GTR','BBOX'}
    assert ed._stack_counter['count']==2 and frozen(s)==before
    assert data.diagnostics(s,[G])['rows'][0]['polygon_iou_pct'] is None

def test_GT_absent_text_only_bbox_can_be_drawn_without_prior_Space(tmp_path,monkeypatch):
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
                       plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    partial.toggle_plate(s,G,OID);data.edit_gt(s,G,OID,text='KEEPGTZ');partial.toggle_plate(s,G,OID)
    before=data.snapshot(s,G);pid=before['links'][OID]['plate_id'];assert before['plates'][pid]['gt_quad'] is None
    monkeypatch.setattr(native,'source_session',lambda _:s);gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);native.key(h,'d');ready(root,h,ed);assert ed.draw_target['kind']=='complete'
        for point in GT:native.click(h,ed,point)
        ready(root,h,ed);after=data.snapshot(s,G)
        assert after['plates'][pid]['gt_quad'] and after['plates'][pid]['gt_text']=='KEEPGTZ' and set(after['plates'])==set(before['plates'])
        assert partial.progress(state(s))['object_decisions'][OID]==partial.UNRESOLVED
    finally:
        try:next(gen)
        except StopIteration:pass
