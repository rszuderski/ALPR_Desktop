"""E02-E22/E27 and M02-M15: approved FINAL population, pure 1:1 and N/A."""
import copy
import json
import hashlib
import csv
import io
import zipfile
import shutil
from pathlib import Path
import pytest
from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
from auto_annotation_tool.ranking import mobile_canvas_review as core, mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_z4_p0_physical_gt import session_fixture
from test_z4_p0_physical_gt import csv_bytes
from test_mobile_p2b2_canvas_review import MANUAL

G='mt-one'
BOX=[150,140,300,210]

def unit(*,gt=1,final=1):
    return dict(session_id='s',source_archive_sha256='a'*64,review_revision=3,
        unit_id='a'*64+':g',source_frame_id='frame',mt_invocation_id='g',source_binding='binding',
        image_entry='samples/frames/f.jpg',image_size=[1000,700],roi=[100,60,900,640],
        execution_status='COMPLETED',source_verified=True,complete=True,n_raw_suppressed=0,
        gt=[dict(gt_id=f'G{i}',bbox=[150+220*i,140,300+220*i,210],confirmed=True) for i in range(gt)],
        finals=[dict(attempt_id=f'A{i}',index=i,confidence=.9-i*.1,bbox=[150+220*i,140,300+220*i,210],decision='PLATE') for i in range(final)])

def counts(result):return tuple(result[k] for k in ('n_gt','tp','fp','fn'))

def approved_session(path,n=1,*,suppressed=None):
    original=session_fixture(path/'source',n=n,suppressed=suppressed)
    # Older generic fixtures intentionally omit detector confidence. Supply
    # actual source fields in our own new archive, never a production fallback.
    with zipfile.ZipFile(original.path) as z:payload={n:z.read(n) for n in z.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode('utf-8'))))
    for row in rows:row['mt_confidence']='0.90'
    payload['samples/attempts.csv']=csv_bytes(rows)
    archive=path/'scoring.alprsession'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    s=MobileReviewSession.open(archive,sidecar_path=path/'scoring.review.json')
    s.set_review_mode('assisted')
    for o in core.scene(s,G)['objects']:
        if o.get('post_nms_suppressed'):continue
        partial.toggle_plate(s,G,o['object_id'])
        core.correct_plate(s,G,o['object_id'],o['quad'])
    return s

def with_source_confidence(original,path):
    with zipfile.ZipFile(original.path) as z:payload={n:z.read(n) for n in z.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode('utf-8'))))
    for row in rows:row['mt_confidence']='0.9'
    payload['samples/attempts.csv']=csv_bytes(rows)
    archive=path/'confidence.alprsession'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    s=MobileReviewSession.open(archive,sidecar_path=path/'confidence.review.json')
    s.set_review_mode('assisted');return s

def test_E02_one_match():assert counts(mt.score_unit(unit()))==(1,1,0,0)

def test_E04_duplicate_final_is_fp_without_changing_positive_decision():
    u=unit(final=2);u['finals'][1]['bbox']=BOX
    before=copy.deepcopy(u);r=mt.score_unit(u)
    assert counts(r)==(1,1,1,0) and r['matched_pairs'][0]['attempt_id']=='A0'
    assert u==before and all(c['decision']=='PLATE' for c in u['finals'])
    assert r['final_outcomes'][1]['reason']=='DUPLICATE_FINAL'

@pytest.mark.parametrize('gt,final,expected',[(2,1,(2,1,0,1)),(1,0,(1,0,0,1)),
                                           (0,1,(0,0,1,0)),(2,2,(2,2,0,0)),(0,0,(0,0,0,0))])
def test_E05_E08_E14_cardinality(gt,final,expected):
    r=mt.score_unit(unit(gt=gt,final=final));assert counts(r)==expected
    assert 'tn' not in r and 'true_negative' not in r

def test_E09_positive_with_bad_location_is_fp_and_fn():
    u=unit();u['finals'][0]['bbox']=[650,400,800,470]
    assert counts(mt.score_unit(u))==(1,0,1,1)

def test_E10_one_good_one_bad_final():
    u=unit(final=2);assert counts(mt.score_unit(u))==(1,1,1,0)

def test_E11_raw_is_diagnostic_only_and_low_confidence_final_is_not_refiltered():
    u=unit();u['n_raw_suppressed']=12;u['finals'][0]['confidence']=0
    r=mt.score_unit(u);assert counts(r)==(1,1,0,0) and r['n_raw_suppressed']==12

@pytest.mark.parametrize('bbox',[None,[1,2,1,3],[float('nan'),1,2,3],[-1,140,300,210]])
def test_E12_missing_or_bad_bbox_is_not_a_false_tp(bbox):
    u=unit();u['finals'][0]['bbox']=bbox;r=mt.score_unit(u)
    assert r['tp'] is None and 'FINAL_SOURCE_BBOX_UNAVAILABLE' in r['unavailable_reasons']

def test_E12_pose_not_used_for_bbox_detection():
    u=unit();u['finals'][0]['mt_quad']=None
    r=mt.score_unit(u);assert counts(r)==(1,1,0,0)
    assert 'polygon_iou' not in r and 'nmce' not in r

@pytest.mark.parametrize('field,value',[('complete',False),('source_verified',False),('execution_status','CANCELLED'),
                                        ('execution_status','NOT_RUN'),('execution_status','ERROR')])
def test_E15_E16_incomplete_is_null(field,value):
    u=unit();u[field]=value;r=mt.score_unit(u)
    assert r['status']=='NOT_EVALUABLE' and all(r[k] is None for k in ('tp','fp','fn','precision','recall','f1'))

def test_E16_one_unresolved_final_blocks_the_unit():
    u=unit(final=2);u['finals'][1]['decision']='UNRESOLVED'
    assert mt.score_unit(u)['tp'] is None

def test_E21_aliases_are_deduplicated_and_conflicts_do_not_choose_a_winner():
    r=mt.score_unit(unit());a=mt.aggregate_units([r,copy.deepcopy(r)])
    assert a['n_total']==a['n_qualified']==1 and a['tp']==1
    other=copy.deepcopy(r);other['review_revision']+=1
    a=mt.aggregate_units([r,other]);assert a['n_qualified']==0 and a['tp'] is None
    assert a['alias_conflicts']==[r['unit_id']]
    assert a['units'][0]['tp'] is None and a['units'][0]['status']=='NOT_EVALUABLE'

def test_E22_distinct_invocations_remain_distinct():
    u=unit();r=mt.score_unit(u);u['unit_id']='b';u['mt_invocation_id']='other'
    a=mt.aggregate_units([r,mt.score_unit(u)]);assert a['n_total']==2 and a['tp']==2

def test_E27_missing_or_changed_policy_cannot_produce_formal_results():
    for policy in (None,{},dict(mt.POLICY,match_iou=.1)):
        r=mt.score_unit(unit(),policy=policy)
        assert r['status']=='BLOCKED_METHODOLOGY' and r['tp'] is None and r['precision'] is None

def test_tie_order_is_confidence_then_index_then_gt_id_and_threshold_is_inclusive():
    u=unit(gt=2,final=2)
    for g in u['gt']:g['bbox']=BOX
    for c in u['finals']:c.update(confidence=.8,bbox=BOX)
    u['gt'].reverse();u['finals'].reverse()
    r=mt.score_unit(u);assert [(p['attempt_id'],p['gt_id']) for p in r['matched_pairs']]==[('A0','G0'),('A1','G1')]
    u=unit();u['finals'][0]['bbox']=[150,140,225,210]
    assert mt.score_unit(u)['tp']==1  # IoU exactly .50

def test_greedy_is_not_replaced_with_maximum_cardinality_matching():
    u=unit(gt=2,final=2)
    u['gt'][0]['bbox']=[200,140,300,210];u['gt'][1]['bbox']=[300,140,400,210]
    u['finals'][0]['bbox']=[200,140,400,210]
    u['finals'][1]['bbox']=[200,140,300,210]
    r=mt.score_unit(u);assert counts(r)==(2,1,1,1)

def test_gt_outside_roi_excluded_but_partial_boundary_is_unavailable():
    u=unit(gt=2);u['gt'][1]['bbox']=[910,140,990,210]
    r=mt.score_unit(u);assert counts(r)==(1,1,0,0) and r['excluded_gt_ids']==['G1']
    u['gt'][1]['bbox']=[880,140,990,210]
    assert 'GT_PARTIAL_ROI_COVERAGE' in mt.score_unit(u)['unavailable_reasons']

@pytest.mark.parametrize('tp,fp,fn,expected',[(8,2,0,(.8,1,8/9)),(8,0,2,(1,.8,8/9)),
    (0,0,0,(None,None,None)),(0,0,1,(None,0,0)),(0,1,0,(0,None,0)),(1,1,1,(.5,.5,.5))])
def test_M02_M07_ratios_and_zero_denominators(tp,fp,fn,expected):
    r=mt.rates(tp,fp,fn)
    for k,v in zip(('precision','recall','f1'),expected):
        assert r[k] is None if v is None else r[k]==pytest.approx(v)

@pytest.mark.parametrize('values',[(-1,0,0),(True,0,0),(1.2,0,0),(None,1,0)])
def test_inconsistent_counts_do_not_become_zero(values):
    assert mt.rates(*values,n_gt=3)['f1'] is None
    assert mt.rates(1,0,0,n_gt=2)['f1'] is None

def test_M12_aggregation_sums_counts_before_division():
    a=unit(gt=1,final=1);b=unit(gt=2,final=10);b['unit_id']='b'
    for c in b['finals']:c['bbox']=BOX  # one TP, nine FP, one FN
    r=mt.aggregate_units([mt.score_unit(a),mt.score_unit(b)])
    assert (r['tp'],r['fp'],r['fn'])==(2,9,1)
    assert r['precision']==pytest.approx(2/11) and r['precision']!=pytest.approx((1+.1)/2)

def test_M14_partial_cohort_excludes_incomplete_units():
    a=unit();b=unit();b.update(unit_id='b',complete=False)
    r=mt.aggregate_units([mt.score_unit(a),mt.score_unit(b)])
    assert r['status']=='PARTIAL' and (r['n_qualified'],r['n_total'],r['tp'])==(1,2,1)

@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),-1,2,True])
def test_format_rate_never_publishes_nan_inf_or_invalid_ratios(value):
    assert mt.format_rate(value)=='—'

def test_E19_shared_gt_correction_changes_all_pairs_but_not_source_parents(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    s=approved_session(tmp_path,n=2)
    from auto_annotation_tool.ranking.mobile_canvas_physical_gt import mark_duplicate
    mark_duplicate(s,G,'GT:a1','GT:a0')
    parents=copy.deepcopy(core.scene(s,G)['parents'])
    before=stacks.diagnostics(s)['rows']
    quad=[[150,400],[300,400],[300,470],[150,470]]
    stacks.edit_gt(s,G,'GT:a0',quad=quad)
    r=mt.evaluate_unit(s,G)
    assert counts(r)==(1,0,2,1),r
    after=stacks.diagnostics(s)['rows']
    assert {tuple(map(tuple,row['gt_quad'])) for row in after}=={tuple(map(tuple,quad))}
    assert [row['polygon_iou_pct'] for row in after]!=[row['polygon_iou_pct'] for row in before]
    assert core.scene(s,G)['parents']==parents

def test_M13_different_sessions_cannot_be_pooled_silently():
    a=unit();b=unit();b.update(session_id='other',unit_id='b')
    r=mt.aggregate_units([mt.score_unit(a),mt.score_unit(b)])
    assert r['tp'] is None and 'MIXED_SESSION_COHORT' in r['unavailable_reasons']

def test_approved_policy_is_frozen_against_mutating_the_public_dict():
    before=copy.deepcopy(mt.POLICY)
    try:
        mt.POLICY['match_iou']=.1
        assert mt.score_unit(unit())['status']=='BLOCKED_METHODOLOGY'
    finally:mt.POLICY.clear();mt.POLICY.update(before)

def test_E12_invalid_pose_source_bbox_with_manual_gt_and_MZ_not_run(tmp_path):
    from test_z4_akt89lm_resolution import invalid_source
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    s=with_source_confidence(invalid_source(tmp_path/'source'),tmp_path)
    quad=[[150,140],[300,140],[300,210],[150,210]]
    manual=core.add_plate(s,G,quad,partial_review=True)
    link_manual_to_mt(s,G,manual,'GT:a0')
    r=mt.evaluate_unit(s,G);assert counts(r)==(1,1,0,0),r
    assert not core.scene(s,G)['parents'][0]['quad']
    assert core.summary(s)['geometry_n']==core.summary(s)['evaluated_reads']==0

def test_E06_E14_explicit_zero_detection_and_negative_need_no_object_tn(tmp_path):
    from test_mobile_p2b2_canvas_review import source_session
    s=source_session(tmp_path,n=0,mz=False)
    partial.confirm_empty(s,G)
    assert counts(mt.evaluate_unit(s,G))==(0,0,0,0)
    assert mt.evaluate_unit(s,G)['f1'] is None
    core.add_plate(s,G,MANUAL,partial_review=True)
    assert counts(mt.evaluate_unit(s,G))==(1,0,0,1)

def test_E03_E17_E19_real_toyota_copy_decision_undo_restart_export(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    baseline=json.loads((repo/'output/Z4_P0_P1_APPROVED_20261026_001/baseline_f4.json').read_text(encoding='utf-8'))
    row=next(r for r in baseline['sessions'] if r['session_id']=='exp-1792702658896-47b4e60b')
    archive,sidecar=Path(row['path']),Path(row['sidecar'])
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (archive,sidecar)}
    shutil.copyfile(archive,tmp_path/archive.name);shutil.copyfile(sidecar,tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/archive.name,sidecar_path=tmp_path/'review.json')
    gid=list(s.mt_invocations)[4]
    v=stacks.snapshot(s,gid)
    assert [c['role'] for c in v['candidates']]==['FINAL','SUPPRESSED','FINAL']
    first,raw,third=v['candidates']
    # Explicit acceptance/correction is confined to this private test sidecar.
    if partial.progress(core.scene(s,gid))['object_decisions'].get(first['object_id'])!='PLATE':
        partial.toggle_plate(s,gid,first['object_id'])
    quad=core._object(core.scene(s,gid),first['object_id'])['quad']
    core.correct_plate(s,gid,first['object_id'],quad)
    stacks.toggle_unassociated(s,gid,third['object_id'],partial_review=True)
    result=mt.evaluate_unit(s,gid)
    assert counts(result)==(1,1,1,0),result
    assert result['n_raw_suppressed']==1 and raw['attempt_id'] not in str(result['matched_pairs'])
    corrected=copy.deepcopy(result)
    assert stacks.restore_history(s,gid)
    assert mt.evaluate_unit(s,gid)!=corrected
    assert stacks.restore_history(s,gid,redo=True)
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert counts(mt.evaluate_unit(reopened,gid))==(1,1,1,0)
    paths=mt.export(reopened,tmp_path/'export')
    exported=json.loads(paths[0].read_text(encoding='utf-8'))
    assert counts(next(r for r in exported['units'] if r['mt_invocation_id']==gid))==(1,1,1,0)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())

def test_E08_two_real_plates_keep_distinct_gt_ids_on_real_photo_fixture(tmp_path):
    real=Path(__file__).resolve().parents[1]/'output/p0_obraz05_20261025'
    protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (real/'two_real_plates.jpg',real/'two_real_plates_gt.json')}
    original=session_fixture(tmp_path/'source',n=2,suppressed=None,real=real)
    s=with_source_confidence(original,tmp_path)
    for obj in core.scene(s,G)['objects']:
        partial.toggle_plate(s,G,obj['object_id']);core.correct_plate(s,G,obj['object_id'],obj['quad'])
    r=mt.evaluate_unit(s,G)
    assert counts(r)==(2,2,0,0) and len({p['gt_id'] for p in r['matched_pairs']})==2
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in protected.items())

def test_adapter_reads_without_migration_and_MZ_text_not_required(tmp_path):
    s=approved_session(tmp_path);before=s.sidecar_path.read_bytes();archive=s.path.read_bytes()
    r=mt.evaluate_unit(s,G);assert counts(r)==(1,1,0,0),r
    assert core.group_issues(s,G)  # missing text required by the old MT+MZ gate
    assert s.sidecar_path.read_bytes()==before and s.path.read_bytes()==archive

def test_space_alone_does_not_accept_unconfirmed_gt_geometry(tmp_path):
    s=session_fixture(tmp_path,n=1,suppressed=None);partial.toggle_plate(s,G,'GT:a0')
    r=mt.evaluate_unit(s,G)
    assert r['tp'] is None and 'GT_GEOMETRY_UNCONFIRMED' in r['unavailable_reasons']

def test_E15_actual_bitmap_dimensions_are_verified_not_just_declared(tmp_path):
    from PIL import Image
    original=session_fixture(tmp_path/'source',n=1,suppressed=None)
    with zipfile.ZipFile(original.path) as z:payload={n:z.read(n) for n in z.namelist()}
    stream=io.BytesIO();Image.new('RGB',(500,350)).save(stream,format='JPEG')
    payload['samples/frames/source.jpg']=stream.getvalue()
    archive=tmp_path/'wrong_dimensions.alprsession'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    s=MobileReviewSession.open(archive,sidecar_path=tmp_path/'wrong.review.json');s.set_review_mode('assisted')
    before=s.sidecar_path.read_bytes()
    r=mt.evaluate_unit(s,G)
    assert r['tp'] is None and any('Rozmiar klatki' in reason for reason in r['unavailable_reasons'])
    assert s.sidecar_path.read_bytes()==before

def test_manual_gt_is_fn_until_linked_without_double_gt(tmp_path):
    s=approved_session(tmp_path)
    manual=core.add_plate(s,G,MANUAL,partial_review=True)
    assert counts(mt.evaluate_unit(s,G))==(2,1,0,1)
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    before=s.sidecar_path.read_bytes()
    # The occupied MT must keep its existing physical reference. New P0
    # contract rejects assigning the independent manual GTR as a second one.
    with pytest.raises(ValueError,match='GTR'):link_manual_to_mt(s,G,manual,'GT:a0')
    assert s.sidecar_path.read_bytes()==before
    # The original purpose (one GT after explicit linking) still applies to
    # an unoccupied MT, on a fresh isolated review of the same source.
    s=MobileReviewSession.open(s.path,sidecar_path=tmp_path/'unlinked.review.json');s.set_review_mode('assisted')
    manual=core.add_plate(s,G,MANUAL,partial_review=True)
    link_manual_to_mt(s,G,manual,'GT:a0')
    r=mt.evaluate_unit(s,G)
    assert r['n_gt']==1 and r['tp']==0 and r['fn']==1  # manual GT is elsewhere

def test_restart_and_undo_redo_keep_source_and_refresh_revision_cache(tmp_path):
    s=approved_session(tmp_path);source=s.path.read_bytes()
    assert mt.evaluate_unit(s,G)['tp']==1
    stacks.toggle_unassociated(s,G,'GT:a0',partial_review=True)
    assert counts(mt.evaluate_unit(s,G))==(0,0,1,0)
    stacks.restore_history(s,G);assert mt.evaluate_unit(s,G)['tp']==1
    stacks.restore_history(s,G,redo=True);assert mt.evaluate_unit(s,G)['fp']==1
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert mt.evaluate_session(reopened)==mt.evaluate_session(s)
    assert s.path.read_bytes()==source

def test_export_is_versioned_read_only_and_sha_bound(tmp_path):
    s=approved_session(tmp_path/'data');before=s.sidecar_path.read_bytes()
    data=mt.evaluate_session(s);root=tmp_path/'exports'
    first=mt.export(s,root);frozen={p:p.read_bytes() for p in first}
    second=mt.export(s,root);assert first[0].parent!=second[0].parent
    assert all(p.read_bytes()==v for p,v in frozen.items())
    assert json.loads(first[0].read_text(encoding='utf-8'))==data
    manifest=json.loads(next(p for p in first if p.name=='manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((first[0].parent/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
    assert s.sidecar_path.read_bytes()==before

def test_revision_change_during_session_calculation_cannot_relabel_old_counts(tmp_path,monkeypatch):
    s=approved_session(tmp_path);original=mt.evaluate_unit;old_revision=s.review.review_revision
    def changed(session,gid,**kwargs):
        result=original(session,gid,**kwargs)
        session.review.review_revision+=1  # test-only interleaving, no file write
        return result
    monkeypatch.setattr(mt,'evaluate_unit',changed)
    r=mt.evaluate_session(s)
    assert r['status']=='STALE_REVIEW' and r['tp'] is None
    assert r['review_revision']==old_revision
    with pytest.raises(ValueError,match='rewizję'):mt.export(s,tmp_path/'export')
    assert not (tmp_path/'export').exists()
