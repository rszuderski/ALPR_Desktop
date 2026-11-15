"""A01-A26: automatic spatial presentation is not scientific TP or a read write."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers

OUT=Path(__file__).resolve().parents[1]/'output/Z4_AUTO_GTR_MT_20261031_001'
SID='exp-1792699864138-add3e282'

@pytest.fixture
def actual(tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    if not (OUT/'private/source.alprsession').exists():pytest.skip('Local F4 acceptance fixture required')
    shutil.copyfile(OUT/'private/source.alprsession',tmp_path/'source.alprsession')
    shutil.copyfile(OUT/'private'/(SID+'.review.json'),tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')
    return s,list(s.mt_invocations)[4]

def editor(s,gid,oid=None):
    sc=lambda:core.scene(s,gid)
    oid=oid or next(c['object_id'] for c in stacks.snapshot(s,gid)['candidates'] if c['role']=='FINAL')
    return NS(h=NS(session=s,current_invocation=gid),_read=sc,selected=oid,mode='edit',gt_focus_id='',
        cache_key=(id(s),gid,s.review.review_revision),show_fp_var=NS(get=lambda:True),clean_bar=True)

def test_A14_existing_real_GTR_is_in_stack_on_first_read_without_link_CTA(actual):
    s,gid=actual;ed=editor(s,gid);before=s.sidecar_path.read_bytes();rev=s.review.review_revision
    v=layers.model(ed)
    assert v['gtr_count']==1
    assert [l['kind'] for l in v['layers']]==['BBOX','POSE','GTR']
    assert s.sidecar_path.read_bytes()==before and s.review.review_revision==rev

def box(l,t,r,b):return [[l,t],[r,t],[r,b],[l,b]]
def source(bbox=(10,10,30,20),pose=None,oid='MT1',role='FINAL'):
    return dict(object_id=oid,role=role,bbox=list(bbox) if bbox else None,pose=pose)
def reference(q=None,pid='GT1',**kw):return dict(plate_id=pid,quad=q or box(10,10,30,20),**kw)
def rank(sources,refs):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import rank
    return rank(sources,refs)

def test_A01_positive_bbox_automatic_choice_is_not_a_write():
    sources=[source(pose=box(11,11,29,19))];refs=[reference()];before=copy.deepcopy((sources,refs))
    r=rank(sources,refs)['MT1'];assert r['plate_id']=='GT1' and r['candidates'][0]['bbox_overlap_area']==200
    assert (sources,refs)==before

def test_A02_pose_only_overlap_reports_source_geometry_disagreement():
    r=rank([source(pose=box(100,100,120,110))],[reference(box(101,101,119,109))])['MT1']
    assert r['plate_id']=='GT1' and r['candidates'][0]['pose_only_overlap']
    assert r['candidates'][0]['bbox_overlap_area']==0 and r['candidates'][0]['pose_overlap_area']==144

@pytest.fixture
def synthetic(tmp_path):
    from test_z4_mt_geometry_layers import source_session
    return source_session(tmp_path/'synthetic',n=1,mz=False),'mt-one'

def manual(s,gid,q):
    oid=core.add_plate(s,gid,q,partial_review=True)
    return oid,stacks.snapshot(s,gid)['links'][oid]['plate_id']

def relation(s,gid,oid='GT:a0'):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import relations
    return relations(s,gid)['relations'][oid]

def test_A03_bbox_with_invalid_Pose_links_without_manufacturing_pose(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
            plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted');gid='mt-one'
    manual(s,gid,box(180,150,260,190));before=s.sidecar_path.read_bytes()
    value=layers.model(editor(s,gid));assert [l['kind'] for l in value['layers']]==['BBOX','GTR']
    assert value['pose_unavailable'] and s.sidecar_path.read_bytes()==before
    assert core.scene(s,gid)['parents'][0]['quad'] is None

def test_A04_new_saved_manual_reference_is_available_immediately(synthetic):
    s,gid=synthetic;ed=editor(s,gid);assert layers.model(ed)['gtr_count']==0
    _,pid=manual(s,gid,box(180,150,260,190));ed.cache_key=(gid,s.review.review_revision)
    assert layers.model(ed)['gtr_id']==pid
    assert not stacks.snapshot(s,gid)['links']['GT:a0']['plate_id']

def test_A05_move_in_undo_and_redo_update_view_without_migration(synthetic):
    s,gid=synthetic;oid,pid=manual(s,gid,box(650,350,800,420));assert relation(s,gid)['plate_id'] is None
    stacks.edit_gt(s,gid,oid,quad=box(180,150,260,190));assert relation(s,gid)['plate_id']==pid
    stacks.restore_history(s,gid);assert relation(s,gid)['plate_id'] is None
    stacks.restore_history(s,gid,redo=True);assert relation(s,gid)['plate_id']==pid

def test_A06_move_out_removes_spatial_relation_but_preserves_MT(synthetic):
    s,gid=synthetic;oid,pid=manual(s,gid,box(180,150,260,190));parents=copy.deepcopy(core.scene(s,gid)['parents'])
    assert relation(s,gid)['plate_id']==pid
    stacks.edit_gt(s,gid,oid,quad=box(650,350,800,420));assert relation(s,gid)['plate_id'] is None
    assert core.scene(s,gid)['parents']==parents

@pytest.mark.parametrize('q',[box(30,10,40,20),box(30,20,40,30)])
def test_A07_zero_area_touch_is_not_an_association(q):
    assert rank([source()],[reference(q)])['MT1']['plate_id'] is None

def test_A08_small_positive_overlap_is_link_but_frozen_scorer_returns_FP_FN():
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
    u=unit();u['finals'][0]['bbox']=[298,140,448,210];before=copy.deepcopy(u)
    r=rank([source(u['finals'][0]['bbox'])],[reference(box(*u['gt'][0]['bbox']))])['MT1']
    assert r['plate_id']=='GT1' and 0<r['candidates'][0]['bbox_iou']<.01
    score=mt.score_unit(u);assert (score['tp'],score['fp'],score['fn'])==(0,1,1) and u==before

def test_A09_multiple_GT_ranking_conflict_and_stable_tie_break():
    a=reference(box(10,10,20,20),'A');b=reference(box(10,10,29,20),'B')
    r=rank([source()],[a,b])['MT1'];assert r['plate_id']=='B' and r['conflict'] and len(r['candidates'])==2
    a['quad']=b['quad'];r=rank([source()],[b,a])['MT1'];assert r['plate_id']=='A'
    assert rank([source()],[a,b])['MT1']==r

def test_A10_many_FINAL_one_GTR_keeps_scientific_one_to_one():
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
    u=unit(final=2);u['finals'][1]['bbox']=list(u['finals'][0]['bbox'])
    r=rank([source(f['bbox'],oid=f['attempt_id']) for f in u['finals']],[reference(box(*u['gt'][0]['bbox']))])
    assert {v['plate_id'] for v in r.values()}=={'GT1'}
    score=mt.score_unit(u);assert (score['tp'],score['fp'],score['fn'])==(1,1,0)

def test_A11_RAW_suppressed_cannot_acquire_FINAL_reference(actual):
    s,gid=actual;ed=editor(s,gid);v=clean.source(ed)
    final,raw=v['candidates'];assert final['plate_id'] and not raw['plate_id'] and raw['role']=='SUPPRESSED'
    assert v['spatial_reference']['relations'][raw['object_id']]['reason']=='NOT_FINAL'

def test_A12_source_coordinates_do_not_depend_on_Canvas_zoom_or_DPI(actual):
    s,gid=actual;ed=editor(s,gid);before=s.sidecar_path.read_bytes();base=clean.source(ed)['spatial_reference']
    for zoom,dpi,pan in ((.1,96,(0,0)),(7,144,(200,-20)),(23,192,(-80,900))):
        ed.h._view_zoom=zoom;ed.dpi=dpi;ed.h._view_center=pan;ed.ref_visible=NS(get=lambda:False);ed.inf_visible=NS(get=lambda:False)
        assert clean.source(ed)['spatial_reference']==base
    assert s.sidecar_path.read_bytes()==before

@pytest.mark.parametrize('field,value',[('image_entry','other.jpg'),('roi',[0,0,2,2]),('mt_invocation_id','different'),('source_binding','wrong')])
def test_A13_incompatible_source_context_has_reason_and_no_link(actual,field,value):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import relations
    s,gid=actual;sc=core.scene(s,gid);sc[field]=value;before=s.sidecar_path.read_bytes()
    r=relations(s,gid,sc);assert r['status']=='UNAVAILABLE' and field in r['reason'] and not r['relations']
    assert s.sidecar_path.read_bytes()==before

def test_A15_unsaved_seed_is_not_a_reference_or_scoring_GT(synthetic):
    s,gid=synthetic;before=s.sidecar_path.read_bytes();r=relation(s,gid)
    assert r['plate_id'] is None and not r['candidates']
    assert layers.model(editor(s,gid))['gtr_count']==0 and not stacks.snapshot(s,gid)['plates']
    assert s.sidecar_path.read_bytes()==before

def test_A16_deleted_reference_does_not_reappear_as_GT(synthetic):
    s,gid=synthetic;oid,pid=manual(s,gid,box(180,150,260,190));core.remove_operator_geometry(s,gid,oid)
    before=s.sidecar_path.read_bytes();assert relation(s,gid)['plate_id'] is None
    assert pid not in stacks.snapshot(s,gid)['plates'] and layers.model(editor(s,gid))['gtr_count']==0
    assert s.sidecar_path.read_bytes()==before

def test_A17_completed_F4_review_and_science_are_unchanged_on_view(tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_session
    rows=json.loads((OUT/'source_inventory.json').read_text(encoding='utf-8'));r=next(r for r in rows if r['session_id']=='exp-1792701720679-5318a7d2')
    shutil.copyfile(r['path'],tmp_path/'original.alprsession');shutil.copyfile(OUT/'private'/(r['session_id']+'.review.json'),tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/'original.alprsession',sidecar_path=tmp_path/'review.json');before=s.sidecar_path.read_bytes()
    reviewed=s.review.to_dict();science=evaluate_session(s)
    for gid in s.mt_invocations:
        if stacks.snapshot(s,gid)['candidates']:layers.model(editor(s,gid))
    assert s.sidecar_path.read_bytes()==before and s.review.to_dict()==reviewed and evaluate_session(s)==science

def test_A18_explicit_distinct_decision_survives_spatial_view_and_edit(synthetic):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import keep_separate
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    s,gid=synthetic;oid,pid=manual(s,gid,box(180,150,260,190));keep_separate(s,gid,oid,'GT:a0')
    before=s.sidecar_path.read_bytes();assert relation(s,gid)['plate_id']==pid and s.sidecar_path.read_bytes()==before
    canonical_action(s,gid,'GT:a0',pid,operation='edit',text='DISTINCT')
    assert not stacks.snapshot(s,gid)['links']['GT:a0']['plate_id']
    assert 'GT:a0' in next(o for o in core.scene(s,gid)['objects'] if o['object_id']==oid)['separate_from_mt']

def test_A19_D_and_edit_target_existing_physical_reference_once(actual):
    from auto_annotation_tool.gui.mobile_canvas_draw_target import resolve
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    s,gid=actual;ed=editor(s,gid);row=clean.selected(ed);pid=row['pid'];before=copy.deepcopy(stacks.snapshot(s,gid)['plates'][pid]);rev=s.review.review_revision
    assert resolve(ed)['reference']['plate_id']==pid
    q=copy.deepcopy(before['gt_quad']);q[0][0]+=1
    canonical_action(s,gid,ed.selected,pid,operation='edit',quad=q)
    v=stacks.snapshot(s,gid);assert list(v['plates'])==[pid] and v['plates'][pid]['gt_quad']==q
    assert v['plates'][pid]['gt_text']==before['gt_text'] and s.review.review_revision==rev+1
    stacks.restore_history(s,gid);assert stacks.snapshot(s,gid)['plates'][pid]['gt_quad']==before['gt_quad']

def test_A20_foreign_GTR_collision_guard_still_blocks_duplicate(actual):
    from auto_annotation_tool.ranking.mobile_manual_frame_guard import GTRCollisionError
    s,gid=actual;q=next(iter(stacks.snapshot(s,gid)['plates'].values()))['gt_quad'];before=s.sidecar_path.read_bytes()
    with pytest.raises(GTRCollisionError):core.add_plate(s,gid,q,partial_review=True)
    assert s.sidecar_path.read_bytes()==before

def test_A21_visibility_and_Shift_use_same_read_only_relations(actual):
    from auto_annotation_tool.gui.mobile_canvas_shift_preview import ShiftPreview
    s,gid=actual;ed=editor(s,gid);before=s.sidecar_path.read_bytes();pid=layers.model(ed)['gtr_id']
    for visible in (False,True):
        ed.ref_visible=NS(get=lambda:visible);ed.inf_visible=NS(get=lambda:visible)
        assert layers.model(ed)['gtr_id']==pid
    preview=ShiftPreview.__new__(ShiftPreview);preview.ed=ed
    frames=preview.frames();assert len(frames)==5 and any(f['kind']=='GTR' and 'MT 1' in f['label'] for f in frames)
    assert s.sidecar_path.read_bytes()==before

def test_A22_BAD_remains_only_on_FINAL_when_GTR_is_spatially_visible(actual):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    s,gid=actual;oid=editor(s,gid).selected;stacks.toggle_unassociated(s,gid,oid,partial_review=True)
    ed=editor(s,gid);row=clean.selected(ed);assert row['fp'] and row['pid']
    pid=row['pid'];canonical_action(s,gid,oid,pid,operation='edit',text='KEEPBAD')
    assert stacks.snapshot(s,gid)['links'][oid]['status']=='UNASSOCIATED_NO_PLATE'
    ref=next(o for o in core.scene(s,gid)['objects'] if not o.get('parent_id'));assert not ref['error']
    assert stacks.snapshot(s,gid)['plates'][pid]['gt_text']=='KEEPBAD'

def test_A23_switching_images_never_reuses_another_frame_relation(actual):
    s,gid=actual;ed=editor(s,gid);pid=layers.model(ed)['gtr_id'];gid6=list(s.mt_invocations)[5]
    ed.h.current_invocation=gid6;ed._read=lambda:core.scene(s,gid6);ed.selected=stacks.snapshot(s,gid6)['candidates'][0]['object_id'];ed.cache_key=(gid6,s.review.review_revision)
    assert layers.model(ed)['gtr_count']==0
    assert layers.model(editor(s,gid))['gtr_id']==pid

def test_A24_real_image05_has_measured_positive_source_area_and_one_GT(actual):
    s,gid=actual;v=clean.source(editor(s,gid));r=v['spatial_reference']['relations'][v['candidates'][0]['object_id']]
    assert len(v['plates'])==1 and [c['role'] for c in v['candidates']]==['FINAL','SUPPRESSED']
    chosen=r['candidates'][0];assert chosen['bbox_overlap_area']>0 and chosen['pose_overlap_area']>0
    (OUT/'image05_overlap.json').write_text(json.dumps(v['spatial_reference'],ensure_ascii=False,indent=2),encoding='utf-8')

def test_A25_distinct_physical_ids_are_not_merged_by_overlapping_model_boxes():
    refs=[reference(box(10,10,20,20),'LEFT',scene_identity='DISTINCT'),reference(box(21,10,31,20),'RIGHT',scene_identity='DISTINCT')]
    sources=[source((9,9,28,21),oid='MT1'),source((15,9,32,21),oid='MT2')];before=copy.deepcopy(refs)
    values=rank(sources,refs);assert values['MT1']['plate_id']=='LEFT' and values['MT2']['plate_id']=='RIGHT'
    assert all(v['conflict'] for v in values.values()) and refs==before

def test_A25_existing_distinct_canonical_ID_is_not_reassigned_by_spatial_choice(tmp_path):
    from test_z4_gtr_reapproval import verified
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    s=verified(tmp_path,n=2);gid='mt-one'
    for oid in ('GT:a0','GT:a1'):partial.confirm_stack(s,gid,oid=oid)
    original=stacks.snapshot(s,gid);p1=original['links']['GT:a0']['plate_id'];p2=original['links']['GT:a1']['plate_id']
    stacks.edit_gt(s,gid,'GT:a0',quad=box(650,350,800,420))
    stacks.edit_gt(s,gid,'GT:a1',quad=box(180,150,260,190))
    before=copy.deepcopy(stacks.snapshot(s,gid));assert relation(s,gid)['plate_id']==p2
    canonical_action(s,gid,'GT:a0',p2,operation='edit',text='SECOND')
    after=stacks.snapshot(s,gid)
    assert set(after['plates'])=={p1,p2} and after['links']==before['links']
    assert after['plates'][p2]['gt_text']=='SECOND' and after['plates'][p1]==before['plates'][p1]

def test_A26_repeated_view_is_idempotent_and_explicit_confirmation_is_one_undoable_write(actual):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    s,gid=actual;ed=editor(s,gid);before=s.sidecar_path.read_bytes();rev=s.review.review_revision
    values=[copy.deepcopy(clean.source(ed)['spatial_reference']) for _ in range(4)]
    assert all(v==values[0] for v in values) and s.sidecar_path.read_bytes()==before
    pid=layers.model(ed)['gtr_id'];canonical_action(s,gid,ed.selected,pid,operation='toggle')
    assert s.review.review_revision==rev+1 and stacks.snapshot(s,gid)['links'][ed.selected]['plate_id']==pid
    assert len(stacks.snapshot(s,gid)['plates'])==1
    stacks.restore_history(s,gid);assert not stacks.snapshot(s,gid)['links'][ed.selected]['plate_id']
    stacks.restore_history(s,gid,redo=True);assert stacks.snapshot(s,gid)['links'][ed.selected]['plate_id']==pid

def test_A26_failed_final_write_does_not_publish_intermediate_association(actual,monkeypatch):
    from auto_annotation_tool.gui.mobile_canvas_spatial_reference import canonical_action
    from auto_annotation_tool.ranking import mobile_human_review
    s,gid=actual;ed=editor(s,gid);pid=layers.model(ed)['gtr_id'];before=s.sidecar_path.read_bytes();review=copy.deepcopy(s.review.to_dict())
    def fail(*a,**kw):raise OSError('Injected private-copy write failure')
    monkeypatch.setattr(mobile_human_review,'_atomic_write',fail)
    with pytest.raises(OSError):canonical_action(s,gid,ed.selected,pid,operation='toggle')
    assert s.sidecar_path.read_bytes()==before and s.review.to_dict()==review
