"""D01-D20/P01-P13: presentation units, not a replacement scientific scorer."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
OUT=Path(__file__).resolve().parents[1]/'output/Z4_SINGLE_TASK_PANEL_20261031_001'
SID='exp-1792699864138-add3e282';GID=SID+'-a00000005'

@pytest.fixture
def actual(tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    source=OUT/'private'/SID
    if not source.exists():pytest.skip('Local current F4 audit fixture required')
    shutil.copyfile(source/'source.alprsession',tmp_path/'source.alprsession');shutil.copyfile(source/'review.json',tmp_path/'review.json')
    return MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json'),GID

def test_D02_D03_D06_D12_existing_pair_has_one_inventory_task(actual):
    s,gid=actual;before=s.sidecar_path.read_bytes();ed=NS(h=NS(session=s),inventory_key=None)
    inv=CanvasReviewEditor.inventory(ed);row=inv['groups'][gid]
    count=row.get('review_tasks',{}).get('count',row['progress']['decisions'])
    assert count==1
    assert s.sidecar_path.read_bytes()==before and not core.scene(s,gid)['checked']

@pytest.fixture
def gui(actual,tmp_path,monkeypatch):
    from test_z4_spatial_gtz_edit import gui as prior
    yield from prior.__wrapped__(actual,tmp_path,monkeypatch,None)

def test_P01_P04_image_cards_contain_real_GT_TP_FP_FN(gui):
    root,h,ed,s,gid=gui
    assert {'gt','tp','fp','fn'}<=set(ed.contexts.counters)
    assert not {'mt','rejected','unlinked'}&set(ed.contexts.counters)
    assert ed.contexts.counters['gt'].cget('text')=='1'
    assert all(ed.contexts.counters[k].cget('text')=='—' for k in ('tp','fp','fn'))

def tasks(s,gid='mt-one'):
    from auto_annotation_tool.gui.mobile_canvas_review_tasks import for_scene
    return for_scene(s,gid)

def box(l,t,r,b):return [[l,t],[r,t],[r,b],[l,b]]

def test_D01_current_source_reasons_are_MT_and_GTR_not_RAW_or_GTZ(actual):
    s,gid=actual;p=partial.progress(core.scene(s,gid));value=tasks(s,gid)
    assert len(p['pending_ids'])==len(p['pending_gtr_ids'])==1 and p['decisions']==2
    assert len(value['units'][0]['reasons'])==2 and value['units'][0]['kind']=='PAIR'
    assert {r.split(':',1)[0] for r in value['units'][0]['reasons']}=={'pending_mt','pending_gtr'}
    assert [c['role'] for c in stacks.snapshot(s,gid)['candidates']]==['FINAL','SUPPRESSED']

@pytest.mark.parametrize('n,quads,expected',[
    (2,[box(150,140,520,210)],2),
    (2,[box(150,140,300,210),box(370,140,520,210)],2),
    (0,[box(150,140,300,210)],1),
    (1,[],1),
    (0,[],0)])
def test_D07_D08_D09_D10_D11_population_units(tmp_path,n,quads,expected):
    from test_z4_mt_geometry_layers import source_session
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    s=source_session(tmp_path,n=n,mz=False)
    for q in quads:core.add_plate(s,'mt-one',q,partial_review=True)
    before=s.sidecar_path.read_bytes();science=evaluate_unit(s,'mt-one');value=tasks(s)
    assert value['count']==expected and s.sidecar_path.read_bytes()==before and evaluate_unit(s,'mt-one')==science
    assert len(stacks.snapshot(s,'mt-one')['plates'])==len(quads)
    assert len({u['final_mt_id'] for u in value['units'] if u['final_mt_id']})==n
    assert 'SCENE' in {v['kind'] for v in value['steps']}
    if not n and not quads:
        assert [r['kind'] for r in value['steps']]==['SCENE']
        assert not core.scene(s,'mt-one')['checked'];partial.confirm_empty(s,'mt-one')
        assert not tasks(s)['needs_review'] and core.scene(s,'mt-one')['checked']
    if not n and quads:
        partial.confirm_gtr(s,'mt-one',next(iter(stacks.snapshot(s,'mt-one')['plates'])))
        assert evaluate_unit(s,'mt-one')['fn']==1 and not tasks(s)['count']

def test_D04_D05_P05_P06_P08_canonical_metrics_and_separate_text_step(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    root,h,ed,s,gid=gui;pid=clean.selected(ed)['pid'];source=clean.selected(ed)['oid']
    native.key(h,'space');ready(root,h,ed)
    raw=evaluate_unit(s,gid);assert (raw['tp'],raw['fp'],raw['fn'])==(1,0,0)
    assert tasks(s,gid)['count']==0 and 'GTZ' in {r['kind'] for r in tasks(s,gid)['steps']}
    assert all(ed.contexts.counters[k].cget('text')==str(raw[k]) for k in ('tp','fp','fn'))
    assert 'GTZ' in ed.contexts.fields['image'].cget('text')
    q=box(283.5,500,289,530);stacks.edit_gt(s,gid,source,quad=q);ready(root,h,ed)
    assert tasks(s,gid)['count']==1
    native.key(h,'space');ready(root,h,ed);raw=evaluate_unit(s,gid)
    assert (raw['tp'],raw['fp'],raw['fn'])==(0,1,1)
    assert all(ed.contexts.counters[k].cget('text')==str(raw[k]) for k in ('tp','fp','fn'))
    for k in ('fp','fn'):
        ed.contexts.counters[k].event_generate('<Button-1>');ready(root,h,ed)
        assert ed.selected and clean.selected(ed)['pid']==pid

def test_D13_P07_invalid_Pose_remains_NA_with_bbox_pair(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def invalid(rows,trace,payload):rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    core.add_plate(s,'mt-one',box(160,145,290,205),partial_review=True)
    assert tasks(s)['count']==1 and tasks(s)['units'][0]['kind']=='PAIR'
    assert core.scene(s,'mt-one')['parents'][0]['quad'] is None
    assert all(r['polygon_iou_pct'] is None and r['nmce_pct'] is None for r in stacks.diagnostics(s)['rows'])

def test_D13_P13_unavailable_context_is_a_separate_source_step():
    from auto_annotation_tool.gui.mobile_canvas_review_tasks import pending_review_units
    value=pending_review_units(None,None)
    assert not value['count'] and value['steps'][0]['kind']=='SOURCE' and not value['needs_review']

def test_D13_blocking_source_reason_precedes_a_pending_pair(actual):
    from auto_annotation_tool.gui.mobile_canvas_review_tasks import pending_review_units
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import project
    s,gid=actual;scene=core.scene(s,gid);view=project(s,gid,scene,stacks.snapshot(s,gid))
    scene=copy.deepcopy(scene);scene['_post_nms_unknown']=True
    value=pending_review_units(scene,view)
    assert value['count']==1 and value['steps'][0]['kind']=='SOURCE'
    assert value['next_action']==value['steps'][0]['text']

def test_D15_no_read_can_complete_whole_image(actual):
    s,gid=actual;before=s.sidecar_path.read_bytes();rev=s.review.review_revision
    for _ in range(3):assert 'SCENE' in {r['kind'] for r in tasks(s,gid)['steps']}
    assert not core.scene(s,gid)['checked'] and s.review.review_revision==rev and s.sidecar_path.read_bytes()==before

def test_D16_D17_P11_space_checkbox_undo_redo_keep_one_task(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    root,h,ed,s,gid=gui;pid=tasks(s,gid)['units'][0]['physical_gt_id'];parents=copy.deepcopy(core.scene(s,gid)['parents'])
    assert tasks(s,gid)['count']==1
    native.key(h,'space');ready(root,h,ed);assert tasks(s,gid)['count']==0
    ed.undo();ready(root,h,ed);assert tasks(s,gid)['count']==1
    ed.undo(redo=True);ready(root,h,ed);assert tasks(s,gid)['count']==0
    ed.approve.invoke();ready(root,h,ed);assert tasks(s,gid)['count']==1
    assert list(stacks.snapshot(s,gid)['plates'])==[pid] and core.scene(s,gid)['parents']==parents

def test_D18_D20_P10_P12_shared_inventory_drives_list_panel_and_next(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    root,h,ed,s,gid=gui
    for target in (SID+'-a00000004',GID,SID+'-a00000007',GID):
        assert ed.open_result_image(target);native.pump(root,lambda:h.current_invocation==target and h._image_entry==core.scene(s,target)['image_entry']);ready(root,h,ed)
        inv=ed.inventory();value=inv['groups'][target]['review_tasks']
        assert inv['task_decisions']==sum(r['review_tasks']['count'] for r in inv['groups'].values())
        assert ed.contexts.counters['decisions'].cget('text')==str(inv['task_decisions'])
        iid=next(i for i,k in h._subject_ids.items() if target in subject_groups(s,k))
        assert ed._image_action([inv['groups'][target]])[0] in h.subject_tree.item(iid,'text')
        assert value['next_action'] in ed.contexts.fields['image'].cget('text')
    expected=next(g for g,r in ed.inventory()['groups'].items() if r['review_tasks']['needs_review'] and r.get('approval_issues',r['issues']))
    ed.next_pending();native.pump(root,lambda:h.current_invocation==expected);ready(root,h,ed)
    assert h.current_invocation==expected

def test_D19_P09_view_and_layer_gestures_are_read_only(gui):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_navigator as nav
    root,h,ed,s,gid=gui;raw=s.sidecar_path.read_bytes();rev=s.review.review_revision
    for kind in ('BBOX','POSE','GTR'):
        nav.select_layer(ed,kind);ready(root,h,ed)
        assert ed.contexts.counters['gt'].cget('text')=='1' and tasks(s,gid)['count']==1
    native.focus_canvas(h);ed.c.event_generate('<KeyPress-Shift_L>');ready(root,h,ed)
    assert len(ed.c.find_withtag('shift_identity_text'))==5
    ed.c.event_generate('<KeyRelease-Shift_L>');ready(root,h,ed)
    from auto_annotation_tool.gui.mobile_canvas_clean_view import selected
    points=selected(ed)['gt'];point=ed.to_canvas(tuple(sum(p[i] for p in points)/4 for i in (0,1)))
    for _ in range(2):native.key(h,'s',xy=point);ready(root,h,ed)
    for key in ('z','x','a','f'):native.key(h,key);ready(root,h,ed)
    ed.c.event_generate('<MouseWheel>',delta=60,x=150,y=150);ready(root,h,ed)
    ed.c.event_generate('<ButtonPress-3>',x=10,y=10);ed.c.event_generate('<B3-Motion>',x=20,y=20);ed.c.event_generate('<ButtonRelease-3>',x=20,y=20);ready(root,h,ed)
    assert s.sidecar_path.read_bytes()==raw and s.review.review_revision==rev

def test_D08_explicit_distinct_and_conflicting_relations_are_not_coalesced(actual):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import keep_separate
    s,gid=actual;row=tasks(s,gid)['units'][0];pid=row['physical_gt_id'];ref=stacks.snapshot(s,gid)['plates'][pid]['origin_object_id']
    keep_separate(s,gid,ref,row['object_id'])
    assert tasks(s,gid)['count']==2

def test_D11_P11_empty_image_has_no_claimed_zero_until_explicit_check(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from test_z4_mt_geometry_layers import source_session
    s=source_session(tmp_path/'source',n=0,mz=False);monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        ready(root,h,ed)
        assert ed.contexts.counters['gt'].cget('text')=='—'
        assert tasks(s)['count']==0 and not core.scene(s,'mt-one')['checked']
        ed.clean_empty_button.invoke();ready(root,h,ed)
        assert core.scene(s,'mt-one')['checked']
        assert all(ed.contexts.counters[k].cget('text')=='0' for k in ('gt','tp','fp','fn'))
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.mark.parametrize('change',['removed','moved'])
def test_explicit_reference_identity_still_groups_pending_aliases(tmp_path,change):
    from test_z4_gtr_reapproval import verified
    s=verified(tmp_path);gid='mt-one';oid='GT:a0'
    pid=stacks.snapshot(s,gid)['links'][oid]['plate_id']
    partial.toggle_stack(s,gid,plate_id=pid)
    if change=='removed':core.remove_operator_geometry(s,gid,oid)
    else:stacks.edit_gt(s,gid,oid,quad=box(650,350,800,420))
    value=tasks(s);assert value['count']==1 and value['units'][0]['physical_gt_id']==pid
    assert len(value['units'][0]['reasons'])==2
