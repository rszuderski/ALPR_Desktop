"""K01-K27: active GTR collision feedback and explicit, atomic redraw."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,time
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data
from auto_annotation_tool.ranking import mobile_manual_frame_guard as guard,mobile_canvas_partial_review as partial

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/Z4_GTR_COLLISIONS_REPLACE_20261027_001'
G='mt-one';F4='exp-1792699303016-c8b05efb';FG=F4+'-a00000008';FOID='GT:'+FG
Q1=[[650,350],[800,350],[800,420],[650,420]]
Q2=[[350,400],[500,400],[500,470],[350,470]]
NEW=[[665,360],[785,360],[785,410],[665,410]]
def rect(l,t,r,b):return [[l,t],[r,t],[r,b],[l,b]]
def center(q):return tuple(sum(p[i] for p in q)/4 for i in (0,1))
def baseline():return json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
def seeded(path):
    s=source_session(path);a=core.add_plate(s,G,Q1,partial_review=True);core.set_text(s,G,a,'WA12345')
    b=core.add_plate(s,G,Q2,partial_review=True);core.set_text(s,G,b,'WB67890')
    return s,a,b
def pid(s,oid,g=G):
    return next(k for k,p in data.snapshot(s,g)['plates'].items() if oid in (p['origin_object_id'],p.get('primary_object_id'),k+':reference'))
def advance(root,seconds):
    end=time.monotonic()+seconds;native.pump(root,lambda:time.monotonic()>=end,seconds=seconds+3)
def start(root,h,ed,oid=None):
    ed.escape();ready(root,h,ed)
    if oid:ed._choose_plate(oid);ready(root,h,ed)
    native.key(h,'d');assert ed.mode=='draw'
def click(root,h,ed,p):native.click(h,ed,p);native.pump(root)
def draw(root,h,ed,q=NEW):
    for p in q:click(root,h,ed,p)
    ready(root,h,ed)

@pytest.fixture
def scene(tmp_path):return seeded(tmp_path)

@pytest.fixture
def gui(tmp_path,monkeypatch):
    s,a,b=seeded(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen);ready(root,h,ed)
    try:yield root,h,ed,s,a,b
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.fixture
def bbox_gui(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    # A fresh review of a read-only source copy, never deleting existing GT.
    archive=ROOT/'output/Z4_GTR_GTZ_GUIDES_20261027_001/read_only_case.alprsession'
    s=MobileReviewSession.open(archive,sidecar_path=tmp_path/'fresh.review.json');s.set_review_mode('assisted')
    partial.toggle_plate(s,FG,FOID);data.edit_gt(s,FG,FOID,text='WU1092S')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    iid=next(i for i in h.subject_tree.get_children('') if FG in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==FG and h._image_entry==core.scene(s,FG)['image_entry'] and not h._saving)
    ready(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def test_K01_baseline_and_K27_originals_unchanged():
    b=baseline();assert len(b['sessions'])==12 and b['free_bytes']>700*1024**2
    for n,v in b['protected'].items():
        p=Path(n);assert p.exists()==v['exists']
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in b['prior_reports'].items())

def test_K03_boundary_and_one_screen_pixel_tolerance(scene):
    s,a,b=scene;state=core.scene(s,G)
    hit=guard.find_point_collision(state,(650,380),tolerance=0)
    assert hit['plate_id']==pid(s,a) and hit['number']==1
    assert guard.find_point_collision(state,(649.5,380),tolerance=1)['plate_id']==hit['plate_id']
    assert guard.find_point_collision(state,(648,380),tolerance=1) is None

def test_K08_stable_hit_not_z_order_and_self_exclusion(scene):
    s,a,b=scene;state=core.scene(s,G);aid=pid(s,a)
    assert guard.find_point_collision(state,center(Q1))['plate_id']==aid
    assert guard.find_point_collision(state,center(Q1),exclude_plate_id=aid) is None
    state[data.KEY]['plates']=dict(reversed(list(state[data.KEY]['plates'].items())))
    assert guard.find_point_collision(state,center(Q1))['plate_id']==aid

def test_K11_containing_and_K12_crossing_are_distinguished(scene):
    s,a,b=scene;state=core.scene(s,G)
    with pytest.raises(ValueError):guard.require_free_manual_area(state,rect(620,330,830,440))
    cross=rect(620,375,830,390);before=set(data.snapshot(s,G)['plates'])
    oid=core.add_plate(s,G,cross,partial_review=True)
    assert len(data.snapshot(s,G)['plates'])==len(before)+1 and oid

def test_K13_unreviewed_MT_pose_is_not_a_GTR(tmp_path):
    s=source_session(tmp_path);state=core.scene(s,G)
    assert guard.find_point_collision(state,center(state['parents'][0]['quad'])) is None
    before=copy.deepcopy(data.snapshot(s,G)['links'])
    core.add_plate(s,G,state['parents'][0]['quad'],partial_review=True)
    assert data.snapshot(s,G)['links']['GT:a0']==before['GT:a0']

def test_K13_presence_only_MT_pose_is_not_an_accepted_GTR(tmp_path):
    s=source_session(tmp_path);partial.toggle_plate(s,G,'GT:a0');state=core.scene(s,G)
    p=data.snapshot(s,G)['plates'][data.snapshot(s,G)['links']['GT:a0']['plate_id']]
    assert p['geometry_state']=='unconfirmed'
    assert guard.find_point_collision(state,center(state['parents'][0]['quad'])) is None

def test_K23_deleted_GTR_in_audit_is_not_an_obstacle(scene):
    s,a,b=scene;core.remove_operator_geometry(s,G,a);state=core.scene(s,G)
    assert state.get(data.HISTORY) and guard.find_point_collision(state,center(Q1)) is None

def test_K02_K04_K05_block_preserves_points_and_wakes_hidden_GTR(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed)
    assert not h.canvas.find_withtag('operator_quad')
    for good in ((540,520),(580,520)):
        click(root,h,ed,good);before=copy.deepcopy(ed.points)
        click(root,h,ed,center(Q1))
        assert ed.points==before and ed.collision_warning.active['plate_id']==pid(s,a)
        assert h.canvas.find_withtag('gtr_collision_warning') and not h.canvas.find_withtag('operator_quad')
        assert h.canvas.cget('cursor')=='crosshair' and s.sidecar_path.read_bytes()==frozen

def test_K06_two_pulses_and_2500ms_timeout_real_Tk(gui,monkeypatch):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed)
    warning=ed.collision_warning;trace=[];original=warning.paint
    def observed():trace.append((time.monotonic(),warning.phase));return original()
    monkeypatch.setattr(warning,'paint',observed)
    click(root,h,ed,center(Q1));t0=time.monotonic()
    assert warning.phase=='red';advance(root,.68);assert warning.phase=='normal' and warning.active
    advance(root,1.4);assert warning.active
    advance(root,.6);assert warning.active is None and not warning.after_ids
    transitions=[]
    for _,phase in trace:
        if not transitions or phase!=transitions[-1]:transitions.append(phase)
    assert transitions[:4]==['red','normal','red','normal'] and transitions.count('red')==2
    assert time.monotonic()-t0>=2.5 and s.sidecar_path.read_bytes()==frozen

def test_K07_repeated_click_renews_expiry_without_strobing(gui):
    root,h,ed,s,a,b=gui;start(root,h,ed);click(root,h,ed,center(Q1))
    started=ed.collision_warning.started_at;advance(root,.7)
    for _ in range(4):click(root,h,ed,center(Q1))
    warning=ed.collision_warning
    assert warning.started_at==started and warning.phase=='normal' and len(warning.after_ids)<=4
    click(root,h,ed,(540,520));assert len(ed.points)==1
    advance(root,1.95);assert warning.active
    advance(root,.65);assert warning.active is None and not warning.after_ids

def test_K08_new_hit_switches_to_concrete_other_ID(gui):
    root,h,ed,s,a,b=gui;start(root,h,ed);click(root,h,ed,center(Q1));first=pid(s,a)
    click(root,h,ed,center(Q2));warning=ed.collision_warning
    assert warning.active['plate_id']==pid(s,b)!=first and len(h.canvas.find_withtag('gtr_collision_warning'))==2
    assert ed.selected=='' and ed.points==[]

def test_K09_Escape_cancels_callbacks_and_restores_data(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed)
    click(root,h,ed,(540,520));click(root,h,ed,center(Q1));native.key(h,'Escape');ready(root,h,ed)
    assert not ed.collision_warning.active and not ed.collision_warning.after_ids
    assert not h.canvas.find_withtag('gtr_collision_warning') and not ed.points and s.sidecar_path.read_bytes()==frozen

def test_K10_dispose_cancels_warning(gui):
    root,h,ed,s,a,b=gui;start(root,h,ed);click(root,h,ed,center(Q1))
    ed.dispose();assert not ed.collision_warning.active and not ed.collision_warning.after_ids

def test_K10_actual_image_change_cancels_warning(bbox_gui):
    root,h,ed,s=bbox_gui
    # Deliberate new annotation on a test copy, in this otherwise fresh review.
    oid=core.add_plate(s,FG,rect(550,610,630,630),partial_review=True);ready(root,h,ed)
    start(root,h,ed);click(root,h,ed,(590,620));assert ed.collision_warning.active
    native.key(h,'e');native.pump(root,lambda:h.current_invocation!=FG and not h._saving);ready(root,h,ed)
    assert not ed.collision_warning.active and not ed.collision_warning.after_ids
    assert not h.canvas.find_withtag('gtr_collision_warning')

def test_K11_enclosing_commit_rejected_and_three_points_preserved(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed)
    draw(root,h,ed,rect(620,330,830,440))
    assert len(ed.points)==3 and ed.collision_warning.active['plate_id']==pid(s,a)
    assert s.sidecar_path.read_bytes()==frozen

def test_K14_bbox_completion_is_same_id_text_and_absent_pose(bbox_gui):
    root,h,ed,s=bbox_gui;before=data.snapshot(s,FG);p=before['links'][FOID]['plate_id'];rev=s.review.review_revision
    start(root,h,ed,FOID);assert ed.draw_reference_target==(FG,FOID,p)
    draw(root,h,ed,[[557,613],[628,612],[629,626],[559,627]])
    after=data.snapshot(s,FG);row=data.diagnostics(s,[FG])['rows'][0]
    assert len(after['plates'])==1 and after['links'][FOID]['plate_id']==p and after['plates'][p]['gt_text']=='WU1092S'
    assert s.review.review_revision==rev+1 and row['mt_quad'] is None and row['mz_status']=='NOT_RUN'
    assert row['polygon_iou_pct'] is row['nmce_pct'] is row['cer'] is None

def test_K15_K16_selected_GTR_redraw_is_atomic_and_same_identity(gui):
    root,h,ed,s,a,b=gui;before=copy.deepcopy(data.snapshot(s,G));rev=s.review.review_revision;parents=copy.deepcopy(ed._read()['parents'])
    start(root,h,ed,a);assert ed.draw_target['kind']=='replace' and 'Poprawiasz GTR 1' in ed.navigator.hint.cget('text')
    draw(root,h,ed)
    after=data.snapshot(s,G);p=pid(s,a)
    assert set(after['plates'])==set(before['plates']) and after['plates'][p]['gt_quad']!=before['plates'][p]['gt_quad']
    assert after['plates'][p]['gt_text']=='WA12345' and after['links']==before['links']
    assert s.review.review_revision==rev+1 and ed._read()['parents']==parents and not ed.collision_warning.active

def test_K17_replace_excludes_self_but_blocks_other_GTR(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed,a)
    click(root,h,ed,center(Q1));assert len(ed.points)==1 and not ed.collision_warning.active
    click(root,h,ed,center(Q2));assert len(ed.points)==1 and ed.collision_warning.active['plate_id']==pid(s,b)
    assert s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('n',[0,1,2,3])
def test_K18_cancel_replacement_is_bitwise_read_only(gui,n):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed,a)
    for p in NEW[:n]:click(root,h,ed,p)
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.points and ed.draw_target is None and s.sidecar_path.read_bytes()==frozen
    assert any(h.canvas.find_withtag('operator_quad'))

def test_K19_K27_undo_redo_and_restart_keep_old_and_new_same_GT(gui):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    root,h,ed,s,a,b=gui;old=copy.deepcopy(data.snapshot(s,G)['plates']);start(root,h,ed,a);draw(root,h,ed)
    new=copy.deepcopy(data.snapshot(s,G)['plates']);ed.undo();ready(root,h,ed);assert data.snapshot(s,G)['plates']==old
    ed.undo(redo=True);ready(root,h,ed);assert data.snapshot(s,G)['plates']==new
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert data.snapshot(reopened,G)['plates']==new

def test_K20_invalid_quad_never_replaces_saved_reference(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed,a)
    draw(root,h,ed,[[700,380]]*4)
    assert s.sidecar_path.read_bytes()==frozen and len(ed.points)==3

def test_K21_collision_timeout_never_changes_scoring_or_checked(gui):
    from auto_annotation_tool.ranking.mobile_multi_plate_verification import multi_scientific_summary
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();before=multi_scientific_summary(s)
    start(root,h,ed);click(root,h,ed,center(Q1));advance(root,2.7)
    assert multi_scientific_summary(s)==before and s.sidecar_path.read_bytes()==frozen

def test_K22_linked_MT_redraw_changes_geometry_only(gui):
    root,h,ed,s,a,b=gui;data.associate(s,G,'GT:a0',a);ready(root,h,ed)
    parents=copy.deepcopy(ed._read()['parents']);before=data.diagnostics(s,[G])['rows'][0];old_links=copy.deepcopy(data.snapshot(s,G)['links'])
    start(root,h,ed,'GT:a0');draw(root,h,ed)
    after=data.diagnostics(s,[G])['rows'][0]
    assert ed._read()['parents']==parents and data.snapshot(s,G)['links']==old_links
    assert after['nmce_pct']!=before['nmce_pct'] and after['gt_text']=='WA12345'

def test_K24_Entry_D_is_text_and_not_a_draw_command(gui):
    root,h,ed,s,a,b=gui
    data.associate(s,G,'GT:a0',a);ready(root,h,ed)
    ed._choose_plate('GT:a0');ready(root,h,ed);native.focus_canvas(h)
    ed.start_entry('GT:a0');native.pump(root);assert ed.entry is not None,dict(status=h.status_var.get(),policy=ed._gt_policy('GT:a0'),saving=h._saving)
    ed.entry.focus_force();ed.entry.selection_clear();ed.entry.icursor('end');ed.entry.event_generate('<KeyPress-d>');ed.entry.event_generate('<KeyRelease-d>');native.pump(root)
    assert ed.mode=='edit' and ed.entry_var.get().endswith('D') and not ed.collision_warning.active
    ed.escape();ready(root,h,ed)

def test_K25_zoom_pan_keeps_warning_source_quad_and_guide_palette(gui):
    root,h,ed,s,a,b=gui;start(root,h,ed);click(root,h,ed,center(Q1))
    h._view_zoom*=1.4;h._view_center=(h._view_center[0]+10,h._view_center[1]+8);h._paint_image();ready(root,h,ed)
    warning=ed.collision_warning;assert warning.active['plate_id']==pid(s,a)
    item=h.canvas.find_withtag('gtr_collision_core')[0]
    assert h.canvas.coords(item)==pytest.approx([v for p in Q1 for v in ed.to_canvas(p)])
    for tag,color in (('gtr_guide_core','#165D8C'),('gtr_guide_halo','#EAF5FC')):
        assert h.canvas.find_withtag(tag)
        assert all(h.canvas.itemcget(i,'fill')==color for i in h.canvas.find_withtag(tag))

def test_K08_overlapping_saved_GTR_lookup_uses_nearest_edge_then_ID(scene):
    s,a,b=scene;state=copy.deepcopy(core.scene(s,G));aid,bid=pid(s,a),pid(s,b)
    state[data.KEY]['plates'][bid]['gt_quad']=rect(660,360,790,410)
    assert guard.find_point_collision(state,(670,380))['plate_id']==bid
    state[data.KEY]['plates'][bid]['gt_quad']=copy.deepcopy(Q1)
    assert guard.find_point_collision(state,center(Q1))['plate_id']==min(aid,bid)

def test_K10_destroyed_Canvas_has_no_late_after_callbacks(scene):
    import tkinter as tk
    from auto_annotation_tool.gui.mobile_canvas_collision_warning import CollisionWarning
    s,a,b=scene;state=core.scene(s,G);root=tk.Tk();root.withdraw();errors=[]
    root.report_callback_exception=lambda *args:errors.append(args)
    canvas=tk.Canvas(root);canvas.pack()
    ed=NS(c=canvas,h=NS(session=s,current_invocation=G,_image_entry=state['image_entry']),
          mode='draw',_read=lambda:state,to_canvas=lambda point:point,message=lambda text:None)
    warning=CollisionWarning(ed);warning.show(guard.find_point_collision(state,center(Q1)))
    assert warning.after_ids
    canvas.destroy();advance(root,2.7)
    assert not warning.after_ids and warning.active is None and not errors
    root.destroy()

def test_K20_outside_ROI_and_stale_target_never_overwrite_reference(gui):
    root,h,ed,s,a,b=gui;start(root,h,ed,a);target=copy.deepcopy(ed.draw_target);frozen=s.sidecar_path.read_bytes()
    click(root,h,ed,(50,50));assert not ed.points and s.sidecar_path.read_bytes()==frozen
    data.edit_gt(s,G,a,text='NEW123');current=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='zmieni'):
        data.redraw_reference(s,G,target,NEW)
    assert s.sidecar_path.read_bytes()==current

def test_K27_current_F4_review_redraw_preserves_other_twenty_images(tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    row=next(r for r in baseline()['sessions'] if r['session_id']==F4)
    path=tmp_path/'copy.review.json';path.write_text(json.dumps(row['payload']),encoding='utf-8')
    archive=ROOT/'output/Z4_GTR_GTZ_GUIDES_20261027_001/read_only_case.alprsession'
    s=MobileReviewSession.open(archive,sidecar_path=path);before=copy.deepcopy(s.review.to_dict())
    scene=core.scene(s,FG);v=data.snapshot(s,FG)
    pgt,p=next((k,p) for k,p in v['plates'].items() if p['gt_quad'])
    target=dict(gid=FG,plate_id=pgt,oid=p['origin_object_id'],binding=copy.deepcopy(scene['source_binding']),
        revision=s.review.review_revision,quad=copy.deepcopy(p['gt_quad']))
    middle=center(p['gt_quad']);quad=[[middle[0]+.94*(x-middle[0]),middle[1]+.94*(y-middle[1])] for x,y in p['gt_quad']]
    data.redraw_reference(s,FG,target,quad);after=s.review.to_dict()
    assert {k:v for k,v in before['invocation_annotations'].items() if k!=FG}=={k:v for k,v in after['invocation_annotations'].items() if k!=FG}
    assert s.review.review_revision==before['review_revision']+1 and data.snapshot(s,FG)['plates'][pgt]['gt_text']==p['gt_text']

def test_K03_actual_edge_and_outside_across_zoom(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed)
    for zoom in (.8,1.4,2.3):
        h._view_zoom=zoom;h._view_center=center(Q1);h._paint_image();ready(root,h,ed)
        click(root,h,ed,(650,380));assert not ed.points and ed.collision_warning.active['plate_id']==pid(s,a)
        x,y=ed.to_canvas((650,380));outside=ed.to_source(round(x)-3,round(y))
        click(root,h,ed,outside);assert len(ed.points)==1
        native.key(h,'BackSpace');assert not ed.points
    assert s.sidecar_path.read_bytes()==frozen

def test_K14_four_points_inside_source_BBOX_keep_one_GT(bbox_gui):
    root,h,ed,s=bbox_gui;before=data.snapshot(s,FG);p=before['links'][FOID]['plate_id'];rev=s.review.review_revision
    start(root,h,ed,FOID);draw(root,h,ed,rect(570,600,620,620))
    after=data.snapshot(s,FG)
    assert len(after['plates'])==1 and after['links'][FOID]['plate_id']==p
    assert after['plates'][p]['gt_text']=='WU1092S' and s.review.review_revision==rev+1
    assert not ed.collision_warning.active and core.scene(s,FG)['parents'][0]['quad'] is None

def test_K20_concave_candidate_does_not_replace_saved_GTR(gui):
    root,h,ed,s,a,b=gui;frozen=s.sidecar_path.read_bytes();start(root,h,ed,a)
    draw(root,h,ed,[[650,350],[800,350],[700,360],[650,420]])
    assert s.sidecar_path.read_bytes()==frozen and len(ed.points)==3
