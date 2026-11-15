"""One physical reference: unsaved source seed, explicit links, labelled RAW."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_mt_geometry_layers import source_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers,mobile_canvas_navigator as nav
from auto_annotation_tool.gui.mobile_canvas_shift_preview import ShiftPreview
from auto_annotation_tool.gui.mobile_canvas_reference_seed import project

OUT=Path(__file__).resolve().parents[1]/'output/Z4_ONE_GTR_SHIFT_IDENTITY_20261031_001'
G='mt-one';OID='GT:a0'

def scene(s,gid=G):return core.scene(s,gid)
def snapshot(s,gid=G):return stacks.snapshot(s,gid)
def candidate(s,oid=OID,gid=G):return next(c for c in snapshot(s,gid)['candidates'] if c['object_id']==oid)
def seed(s,oid=OID,gid=G):return project(s,gid,scene(s,gid),snapshot(s,gid),candidate(s,oid,gid))

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    if getattr(request,'param',None)=='bbox':
        from test_z4_space_gtr_hud_shift import fixture_archive
        def invalid(rows,trace,payload):
            rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
                plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
        s=fixture_archive(tmp_path/'seed',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    else:s=source_session(tmp_path/'seed',n=2,mz=False)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def drag(root,h,ed,dx=5,dy=2,release=True):
    row=clean.selected(ed);x,y=map(round,ed.to_canvas(row['gt'][0]))
    hit=ed.hit(x,y,prefer_stack=True)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y);assert ed.drag,(row['oid'],ed.mode,hit,x,y,ed.c.winfo_width(),ed.c.winfo_height(),h.status_var.get())
    ed.c.event_generate('<B1-Motion>',x=x+dx,y=y+dy)
    if release:ed.c.event_generate('<ButtonRelease-1>',x=x+dx,y=y+dy);ready(root,h,ed)

def test_seed_is_source_aligned_but_not_a_saved_GT(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();parents=copy.deepcopy(scene(s)['parents'])
    assert not snapshot(s)['plates']
    assert clean.selected(ed)['seed'] and clean.selected(ed)['gt']==[list(p) for p in candidate(s)['mt_quad']]
    assert layers.model(ed)['gtr_count']==0 and len(layers.model(ed)['layers'])==3
    for kind in ('BBOX','POSE','GTR'):
        assert nav.select_layer(ed,kind);ready(root,h,ed)
    assert ed.c.find_withtag('operator_seed')
    assert s.sidecar_path.read_bytes()==before and scene(s)['parents']==parents and not snapshot(s)['plates']

def test_two_unsaved_seeds_are_separate_navigation_targets_and_clickable_GTR(gui):
    from auto_annotation_tool.gui.mobile_canvas_plate_cycle import targets,next_target
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();rows=targets(ed)
    assert len(rows)==2 and len({r['key'] for r in rows})==2
    assert next_target(ed,rows)['oid']=='GT:a1'
    r=clean.selected(ed);point=tuple(sum(p[i] for p in r['gt'])/4 for i in (0,1));x,y=ed.to_canvas(point)
    assert nav.hit(ed,x,y,prefer_stack=True)[1]=='layer:GTR'
    native.key(h,'a');ready(root,h,ed);assert ed.selected=='GT:a1'
    native.key(h,'a');ready(root,h,ed);assert ed.selected==OID
    assert not snapshot(s)['plates'] and s.sidecar_path.read_bytes()==before

def test_D_edits_seed_and_first_release_creates_only_one_linked_GTR(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();parents=copy.deepcopy(scene(s)['parents'])
    native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='edit' and ed.selected==OID and not ed.points and not ed.draw_target
    assert s.sidecar_path.read_bytes()==before
    drag(root,h,ed,release=False)
    assert s.sidecar_path.read_bytes()==before
    ed.c.event_generate('<ButtonRelease-1>');ready(root,h,ed)
    state=snapshot(s);pid=state['links'][OID]['plate_id']
    assert len(state['plates'])==1 and pid and not clean.selected(ed)['seed']
    assert scene(s)['parents']==parents
    native.key(h,'d');ready(root,h,ed);drag(root,h,ed,dx=3,dy=1)
    assert len(snapshot(s)['plates'])==1 and snapshot(s)['links'][OID]['plate_id']==pid

def test_Esc_discards_seed_drag_without_saving_GT(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes()
    native.key(h,'d');ready(root,h,ed);drag(root,h,ed,release=False)
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.drag and not snapshot(s)['plates'] and s.sidecar_path.read_bytes()==before

def test_fast_second_corner_press_is_not_dropped_by_double_click_binding(gui):
    root,h,ed,s=gui;native.key(h,'d');ready(root,h,ed);before=s.sidecar_path.read_bytes()
    x,y=map(round,ed.to_canvas(clean.selected(ed)['gt'][0]))
    ed.double_press(NS(x=x,y=y,widget=ed.c,state=0))
    assert ed.drag and s.sidecar_path.read_bytes()==before
    ed.escape();ready(root,h,ed)

def test_two_real_plates_keep_separate_reference_ids(gui):
    root,h,ed,s=gui;ids=[]
    for oid in (OID,'GT:a1'):
        assert nav.select_layer(ed,'GTR',oid);ready(root,h,ed);native.key(h,'d');ready(root,h,ed);drag(root,h,ed)
        ids.append(snapshot(s)['links'][oid]['plate_id']);ed.escape();ready(root,h,ed)
    assert len(set(ids))==2 and len(snapshot(s)['plates'])==2

def test_removed_GTR_never_regenerates_operator_seed(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);partial.confirm_stack(s,G,oid=OID)
    stacks.remove_linked_geometry(s,G,OID)
    assert seed(s) is None

def test_bbox_seed_does_not_invent_source_pose_or_review(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
            plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);before=s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None
    value=seed(s)
    assert value['origin']=='BBOX' and value['quad']==[[150,140],[300,140],[300,210],[150,210]]
    assert candidate(s)['mt_quad'] is None and not snapshot(s)['plates']
    assert (s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None)==before

@pytest.mark.parametrize('gui',['bbox'],indirect=True)
def test_bbox_seed_corner_release_saves_GTR_without_manufacturing_Pose(gui):
    root,h,ed,s=gui;before=copy.deepcopy(scene(s)['parents']);native.key(h,'d');ready(root,h,ed)
    assert clean.selected(ed)['seed']['origin']=='BBOX';drag(root,h,ed)
    assert len(snapshot(s)['plates'])==1 and snapshot(s)['links'][OID]['plate_id']
    assert scene(s)['parents']==before and candidate(s)['mt_quad'] is None

@pytest.fixture
def actual(tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    if not (OUT/'diagnosis_before.json').is_file():pytest.skip('Local F4 audit fixture required')
    shutil.copyfile(OUT/'private/source.alprsession',tmp_path/'source.alprsession')
    shutil.copyfile(OUT/'private/exp-1792699864138-add3e282.review.json',tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')
    gid=list(s.mt_invocations)[4]
    return s,gid

def preview(s,gid):
    p=ShiftPreview.__new__(ShiftPreview);p.ed=NS(h=NS(session=s,current_invocation=gid),_read=lambda:scene(s,gid));return p

def test_real_image05_has_one_manual_GTR_FINAL_and_suppressed_candidate(actual):
    s,gid=actual;state=snapshot(s,gid);before=s.sidecar_path.read_bytes()
    assert len(state['plates'])==1
    assert [c['role'] for c in state['candidates']]==['FINAL','SUPPRESSED']
    oid=state['candidates'][0]['object_id'];assert not state['links'][oid]['plate_id'] and seed(s,oid,gid) is None
    frames=preview(s,gid).frames()
    assert len(frames)==5
    assert len([f for f in frames if f['kind']=='GTR'])==1
    assert len([f for f in frames if f['role']=='SUPPRESSED'])==2
    assert all(f['label'] for f in frames) and all('NMS' in f['label'] for f in frames if f['role']=='SUPPRESSED')
    assert s.sidecar_path.read_bytes()==before

def test_existing_real_GTR_blocks_enclosing_or_overlapping_second_GTR(actual):
    from auto_annotation_tool.ranking.mobile_manual_frame_guard import GTRCollisionError
    s,gid=actual;state=snapshot(s,gid);p=next(iter(state['plates'].values()));q=p['gt_quad'];before=s.sidecar_path.read_bytes()
    l,t,r,b=min(x for x,y in q)-3,min(y for x,y in q)-3,max(x for x,y in q)+3,max(y for x,y in q)+3
    for points in (q,[[l,t],[r,t],[r,b],[l,b]]):
        with pytest.raises(GTRCollisionError):core.add_plate(s,gid,points,partial_review=True)
        assert s.sidecar_path.read_bytes()==before
    final=next(c for c in state['candidates'] if c['role']=='FINAL')
    for action in (lambda:partial.confirm_stack(s,gid,oid=final['object_id']),
                   lambda:stacks.accept_reference(s,gid,final['object_id'])):
        with pytest.raises(GTRCollisionError):action()
        assert s.sidecar_path.read_bytes()==before and len(snapshot(s,gid)['plates'])==1

def test_explicit_link_keeps_real_GTR_identity_geometry_and_RAW_excluded(actual):
    s,gid=actual;state=snapshot(s,gid);pid,p=next(iter(state['plates'].items()));before=copy.deepcopy(p)
    final,raw=state['candidates'];parents=copy.deepcopy(scene(s,gid)['parents'])
    stacks.associate(s,gid,final['object_id'],pid)
    after=snapshot(s,gid)
    assert list(after['plates'])==[pid] and after['plates'][pid]['gt_quad']==before['gt_quad']
    assert after['plates'][pid]['gt_text']==before['gt_text']
    assert after['links'][final['object_id']]['plate_id']==pid and not after['links'][raw['object_id']]['plate_id']
    assert scene(s,gid)['parents']==parents
    from auto_annotation_tool.ranking.mobile_mt_geometry_layers import snapshot as geometry
    assert [l['kind'] for l in geometry(s,gid,final['object_id'])['layers']]==['BBOX','POSE','GTR']

@pytest.fixture
def actual_gui(actual,tmp_path,monkeypatch):
    s,gid=actual;monkeypatch.setattr(native,'source_session',lambda _:s)
    cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
        iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==scene(s,gid)['image_entry'] and not h._saving)
        ready(root,h,ed);yield root,h,ed,s,gid
    finally:
        try:next(gen)
        except StopIteration:pass

def test_Shift_labels_every_real_frame_including_suppressed_RAW(actual_gui):
    root,h,ed,s,gid=actual_gui;before=s.sidecar_path.read_bytes()
    native.focus_canvas(h);h.canvas.event_generate('<KeyPress-Shift_L>');ready(root,h,ed)
    assert ed.shift_preview.active and len(ed.shift_preview.frames())==5
    labels=[ed.c.itemcget(i,'text') for i in ed.c.find_withtag('shift_identity_text')]
    assert len(labels)==5 and len(set(labels))==5
    assert sum('odrzucony przez NMS' in text for text in labels)==2
    assert sum('bez powiązania MT' in text for text in labels)==1
    assert not ed.c.find_withtag('context_badge_title')
    assert s.sidecar_path.read_bytes()==before
    h.canvas.event_generate('<KeyRelease-Shift_L>');ready(root,h,ed)
    assert not ed.c.find_withtag('shift_identity_text') and s.sidecar_path.read_bytes()==before

def test_D_focuses_existing_real_GTR_then_explicit_button_links_same_identity(actual_gui):
    root,h,ed,s,gid=actual_gui;state=snapshot(s,gid);pid,p=next(iter(state['plates'].items()));oid=state['candidates'][0]['object_id']
    before=s.sidecar_path.read_bytes();native.key(h,'d');ready(root,h,ed)
    assert ed.mode=='edit' and clean.selected(ed)['pid']==pid and not ed.draw_target
    assert s.sidecar_path.read_bytes()==before and len(snapshot(s,gid)['plates'])==1
    ed.escape();ready(root,h,ed);assert nav.select_layer(ed,'POSE',oid);ready(root,h,ed)
    assert len(ed.existing_reference_buttons)==1 and ed.existing_reference_bar.winfo_manager()
    ed.existing_reference_buttons[0].invoke();ready(root,h,ed)
    state=snapshot(s,gid)
    assert state['links'][oid]['plate_id']==pid and len(state['plates'])==1
    assert state['plates'][pid]['gt_quad']==p['gt_quad'] and state['plates'][pid]['gt_text']==p['gt_text']
    assert len(layers.model(ed)['layers'])==3 and layers.model(ed)['gtr_count']==1
    assert partial.progress(scene(s,gid))['decisions']==1
