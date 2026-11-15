"""F4-EVAL-2.0 acceptance 01-27, 35-41, 43; no production writes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from auto_annotation_tool.ranking import mobile_mt_object_evaluation as mt
from auto_annotation_tool.ranking import mobile_mt_object_export as exporting
from auto_annotation_tool.ranking import mobile_object_mz_quality as mz
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_z4_mt_detection_evaluation import approved_session

G = 'mt-one'
Q = [[150, 140], [300, 140], [300, 210], [150, 210]]


def unit(n_gt=1, n_final=1):
    gt = [dict(gt_id=f'G{i}', quad=[[x+220*i, y] for x,y in Q], confirmed=True) for i in range(n_gt)]
    finals = []
    for i in range(n_final):
        pid = f'G{min(i, n_gt-1)}' if n_gt else None
        decision = 'PLATE' if pid else 'NOT_PLATE'
        finals.append(dict(attempt_id=f'A{i}', object_id=f'O{i}', index=i, role='FINAL', confidence=.9-i*.1,
            bbox=[150+220*i, 140, 300+220*i, 210], bbox_source_verified=True,
            pose_quad=[[x+220*i, y] for x,y in Q], pose_source_verified=True,
            decision=decision, operator_confirmed=True, gt_id=pid, relation_confirmed=bool(pid),
            operator_review_evidence=dict(kind='PERSISTED_FRAME_DECISION', attempt_id=f'A{i}',
                gt_id=pid, source_binding='binding', decision=decision)))
    return dict(session_id='s', source_archive_sha256='a'*64, review_revision=3,
        unit_id='a'*64+':g', source_frame_id='frame', mt_invocation_id='g', source_binding='binding',
        image_entry='samples/frames/frame.jpg', image_size=[1000,700], roi=[100,60,900,640],
        source_verified=True, execution_status='COMPLETED', complete=True, gt_complete=True,
        n_raw_suppressed=0, gt=gt, finals=finals)


def counts(r):
    return tuple(r[k] for k in ('tp','fp','fn'))


def negative(c):
    c.update(decision='NOT_PLATE',gt_id=None,relation_confirmed=False)
    c['operator_review_evidence'].update(decision='NOT_PLATE',gt_id=None)


def readout(**changes):
    row=dict(executed=True,parent_proven=True,source_mt_role='FINAL',mt_outcome='TP',
        operator_decision='PLATE',crop_parent_proven=True,readout_unambiguous=True,
        gtz_confirmed=True,gtz='ABC123',raw_prediction='ABC123',attempt_id='a',
        crop_entry='c.jpg',timing={'character_inference_ms':10},source_mz_ms=10)
    row.update(changes)
    return row


@pytest.fixture
def session(tmp_path):
    s=approved_session(tmp_path)
    v=stacks.snapshot(s,G)
    partial.confirm_stack(s,G,plate_id=next(iter(v['plates'])))
    core.set_scene_review_state(s,G,checked=True)  # Inspection independent of optional OCR GTZ.
    assert partial.progress(core.scene(s,G))['complete']
    return s


def test_T01_one_FINAL_with_two_source_geometries_is_one_TP():
    r=mt.score_unit_v2(unit())
    assert counts(r)==(1,0,0) and r['n_final_mt']==1 and len(r['matched_pairs'])==1


def test_T02_low_bbox_IoU_does_not_overrule_confirmed_object():
    u=unit();u['finals'][0]['bbox']=[225,140,375,210]
    r=mt.score_unit_v2(u)
    assert counts(r)==(1,0,0) and r['matched_pairs'][0]['bbox_iou']==pytest.approx(1/3)
    assert 'BBOX_POSE_THRESHOLD_DISAGREEMENT' in r['matched_pairs'][0]['disagreement_codes']


def test_T03_zero_bbox_has_NA_geometry_and_one_TP():
    u=unit();u['finals'][0]['bbox']=[150,140,300,140]
    r=mt.score_unit_v2(u)
    assert counts(r)==(1,0,0) and r['matched_pairs'][0]['bbox_iou'] is None
    assert r['matched_pairs'][0]['polygon_iou']==1


def test_T04_zero_bbox_rejected_fence_and_unmatched_true_plate():
    u=unit();negative(u['finals'][0]);u['finals'][0]['bbox']=[400,300,450,300]
    r=mt.score_unit_v2(u)
    assert counts(r)==(0,1,1) and r['unmatched_gt_ids']==['G0']


def test_T05_car_PLATE_and_fence_NOT_PLATE():
    u=unit(n_final=2);negative(u['finals'][1]);assert counts(mt.score_unit_v2(u))==(1,1,0)


def test_T06_duplicate_FINAL_is_FP_without_rewriting_operator_history():
    u=unit(n_final=2);before=copy.deepcopy(u);r=mt.score_unit_v2(u)
    assert counts(r)==(1,1,0) and r['final_outcomes'][1]['reason']=='DUPLICATE_FINAL'
    assert r['matched_pairs'][0]['attempt_id']=='A0' and u==before


def test_T07_two_distinct_physical_references_are_not_spatially_merged():
    u=unit(2,2);u['gt'][1]['quad']=copy.deepcopy(u['gt'][0]['quad'])
    assert counts(mt.score_unit_v2(u))==(2,0,0)


@pytest.mark.parametrize('n_gt,n_final,expected',[(1,0,(0,0,1)),(0,1,(0,1,0)),(0,0,(0,0,0))],ids=['T08','T09','T10'])
def test_T08_T10_cardinality(n_gt,n_final,expected):
    r=mt.score_unit_v2(unit(n_gt,n_final));assert counts(r)==expected
    assert 'tn' not in r
    if n_gt==n_final==0:assert r['precision'] is r['recall'] is r['f1'] is None


def test_T11_RAW_is_outside_FINAL_without_confidence_refilter():
    u=unit();u['n_raw_suppressed']=4;u['finals'][0]['confidence']=0
    r=mt.score_unit_v2(u);assert counts(r)==(1,0,0) and r['n_raw_suppressed']==4 and r['n_final_mt']==1


def test_T12_edit_one_GTR_preserves_original_bbox_pose_and_attempt(session):
    before=core.scene(session,G);aid=before['parents'][0]['attempt_id'];source=copy.deepcopy(session.attempts[aid])
    oid=before['objects'][0]['object_id'];q=copy.deepcopy(before['objects'][0]['quad']);q[0][0]+=1
    core.correct_plate(session,G,oid,q)
    partial.confirm_stack(session,G,plate_id=next(iter(stacks.snapshot(session,G)['plates'])))
    core.set_scene_review_state(session,G,checked=True)
    after=core.scene(session,G);r=mt.evaluate_unit(session,G)
    assert before['parents']==after['parents'] and session.attempts[aid]==source
    assert counts(r)==(1,0,0) and r['n_final_mt']==1 and len(r['physical_gt'])==1


@pytest.mark.parametrize('change,reason',[
    ('unconfirmed','GT_GEOMETRY_UNCONFIRMED'),('incomplete','REVIEW_INCOMPLETE'),('partial_roi','GT_PARTIAL_ROI_COVERAGE')])
def test_T13_unconfirmed_or_partial_GT_is_unavailable(change,reason):
    u=unit()
    if change=='unconfirmed':u['gt'][0]['confirmed']=False
    elif change=='incomplete':u['gt_complete']=False
    else:u['roi']=[200,60,900,640]
    r=mt.score_unit_v2(u);assert counts(r)==(None,None,None) and reason in r['unavailable_reasons']


def test_T14_unresolved_has_only_confirmed_FP_lower_bound():
    u=unit(n_final=2);negative(u['finals'][1]);u['complete']=False;u['finals'][0]['decision']='UNRESOLVED'
    r=mt.score_unit_v2(u)
    assert counts(r)==(None,None,None) and r['partial_fp_lower_bound']==1 and r['status']=='REVIEW_REQUIRED'


def test_T15_spatial_overlap_without_operator_evidence_is_never_TP():
    u=unit();u['finals'][0].update(operator_confirmed=False,operator_review_evidence=None)
    assert counts(mt.score_unit_v2(u))==(None,None,None)


@pytest.mark.parametrize('field,value',[('gt_id','missing'),('relation_confirmed',False)])
def test_T16_missing_stable_relation_is_NA(field,value):
    u=unit();u['finals'][0][field]=value;r=mt.score_unit_v2(u)
    assert 'MT_GTR_RELATION_UNPROVEN' in r['unavailable_reasons'] and counts(r)==(None,None,None)


def test_T17_conflicting_source_index_is_not_resolved_by_geometry():
    u=unit(2,2);u['finals'][1]['index']=0
    assert 'FINAL_INDEX_CONFLICT' in mt.score_unit_v2(u)['unavailable_reasons']


def test_T18_actual_source_SHA_drift_blocks_scoring(session):
    with session.path.open('ab') as handle:handle.write(b'drift')
    r=mt.evaluate_unit(session,G,cache=False)
    assert counts(r)==(None,None,None) and 'SOURCE_NOT_VERIFIED' in r['unavailable_reasons']


@pytest.mark.parametrize('key,value',[('confidence',None),('index',None),('confidence',float('nan'))])
def test_T19_confidence_and_source_index_have_no_default_zero(key,value):
    u=unit();u['finals'][0][key]=value;assert counts(mt.score_unit_v2(u))==(None,None,None)


def test_T20_GT_straddling_ROI_cannot_be_a_false_FN():
    u=unit(n_final=0);u['roi'][0]=151
    assert counts(mt.score_unit_v2(u))==(None,None,None)


def test_T21_executed_fence_MZ_excluded_keeps_raw_crop_and_cost():
    row=readout(mt_outcome='FP',operator_decision='NOT_PLATE',raw_prediction='FENCE')
    out=mz.score_readout(row)
    assert out['quality_state']=='EXECUTED_EXCLUDED_MT_FP' and out['cer'] is None
    for k in ('raw_prediction','crop_entry','timing','source_mz_ms','attempt_id'):assert out[k]==row[k]


def test_T23_wrong_OCR_on_true_plate_is_scored_error():
    out=mz.score_readout(readout(raw_prediction='ABC124'))
    assert out['canonical_ocr_eligible'] and out['edit_distance']==1 and not out['exact_match']


def test_T24_empty_executed_OCR_has_full_edit_distance():
    out=mz.score_readout(readout(raw_prediction=''))
    assert out['quality_state']=='EXECUTED_ELIGIBLE' and out['edit_distance']==6 and out['cer']==1


def test_T25_NOT_RUN_is_outside_conditional_OCR():
    out=mz.score_readout(readout(executed=False,raw_prediction=''))
    assert out['quality_state']=='NOT_RUN' and out['edit_distance'] is None


@pytest.mark.parametrize('changes,reason',[({'parent_proven':False},'MZ_PARENT_UNPROVEN'),
    ({'crop_parent_proven':False},'MZ_PARENT_UNPROVEN'),({'gtz_confirmed':False},'GTZ_UNAVAILABLE'),
    ({'gtz':''},'GTZ_UNAVAILABLE')])
def test_T26_missing_crop_or_GTZ_is_NA(changes,reason):
    out=mz.score_readout(readout(**changes));assert not out['canonical_ocr_eligible'] and out['quality_reason']==reason


def test_T27_one_trace_two_MZ_calls_is_not_a_single_call_percentile():
    traces=[dict(status='preliminary',mz_runs='2',mt_runs_this_frame='1',character_inference_ms='500')]
    session=SimpleNamespace(bundle=SimpleNamespace(trace_rows=traces,trace_total=1))
    out=mz._execution_counters(session)
    assert out['n_mz_calls']==2 and out['n_traces']==1 and out['mz_single_call_p95_ms'] is None
    assert traces[0]['character_inference_ms']=='500'


def test_T35_T36_export_is_replayable_additive_and_deterministic(session,tmp_path):
    a=mt.export(session,tmp_path/'new');first={p.name:p.read_bytes() for p in a}
    data=exporting.replay(a[0].parent)
    b=mt.export(session,tmp_path/'new');second={p.name:p.read_bytes() for p in b}
    assert a[0].parent!=b[0].parent and first==second
    assert counts(data)==(1,0,0) and first=={p.name:p.read_bytes() for p in a}
    (b[0].parent/'pairs.csv').write_text('corrupt',encoding='utf-8')
    with pytest.raises(ValueError,match='EXPORT_FILE_MISMATCH'):exporting.replay(b[0].parent)


def test_T40_source_mapping_order_does_not_change_export_manifest(session,tmp_path):
    a=mt.export(session,tmp_path/'new')
    def reorder(value):
        if isinstance(value,dict):return {k:reorder(v) for k,v in reversed(list(value.items()))}
        if isinstance(value,list):return [reorder(v) for v in value]
        return value
    session.review.provenance=reorder(session.review.provenance)
    session.review.invocation_annotations=reorder(session.review.invocation_annotations)
    mt._cache.pop(session,None)
    b=mt.export(session,tmp_path/'new')
    assert {p.name:p.read_bytes() for p in a}=={p.name:p.read_bytes() for p in b}


def test_T35_replay_rejects_inconsistent_OCR_summary_even_with_rehashed_file(session,tmp_path):
    paths=mt.export(session,tmp_path/'new');root=paths[0].parent
    p=root/'mt_object_evaluation.json';d=json.loads(p.read_text(encoding='utf-8'));d['ocr_summary']['n_reads']+=1
    p.write_text(json.dumps(d),encoding='utf-8')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    manifest['files'][p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size)
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='EXPORT_OCR_SUMMARY_REPLAY_MISMATCH'):exporting.replay(root)


def test_T37_missing_GTZ_keeps_MT_full_and_OCR_unavailable(session):
    result=mt.evaluate_session(session)
    assert result['status']=='COMPLETE' and counts(result)==(1,0,0)
    assert result['ocr_summary']['n_reads']==0 and result['ocr_summary']['micro_cer'] is None
    assert mz.score_readout(readout(gtz_confirmed=False))['quality_reason']=='GTZ_UNAVAILABLE'


@pytest.mark.parametrize('external',[False,True],ids=['in-memory-revision','external-sidecar'])
def test_T38_change_during_export_cannot_publish(session,tmp_path,monkeypatch,external):
    guard=exporting._publish_guard
    calls=[]
    def change(s,*args):
        calls.append(1)
        if len(calls)>1:
            if external:
                doc=json.loads(s.sidecar_path.read_text(encoding='utf-8'));doc['review_revision']+=1
                s.sidecar_path.write_text(json.dumps(doc),encoding='utf-8')
            else:s.review.review_revision+=1
        guard(s,*args)
    monkeypatch.setattr(exporting,'_publish_guard',change)
    with pytest.raises(ValueError,match='REVIEW_REVISION_CHANGED'):mt.export(session,tmp_path/'new')
    assert list((tmp_path/'new').iterdir())==[]


def test_T38_stale_loaded_review_is_rejected_before_export(session,tmp_path):
    doc=json.loads(session.sidecar_path.read_text(encoding='utf-8'));doc['review_revision']+=1
    session.sidecar_path.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises(ValueError,match='STALE_LOADED_REVIEW'):mt.export(session,tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_T40_ties_order_aliases_and_mixed_policy():
    u=unit(n_final=2);u['finals'][1]['confidence']=u['finals'][0]['confidence']
    a=mt.score_unit_v2(u);u['finals'].reverse();b=mt.score_unit_v2(u)
    # The source input ordering is retained evidence; normalized scoring is stable.
    for k in ('matched_pairs','final_outcomes','geometry_diagnostics','tp','fp','fn'):assert a[k]==b[k]
    assert mt.aggregate_units([a,a])['n_total']==1
    changed=copy.deepcopy(a);changed['matching_policy_sha256']='b'*64
    assert mt.aggregate_units([changed])['n_qualified']==0


def test_T41_disjoint_source_geometries_are_diagnosed_without_repair():
    u=unit();u['finals'][0]['pose_quad']=[[x+400,y] for x,y in Q];before=copy.deepcopy(u)
    r=mt.score_unit_v2(u);assert counts(r)==(1,0,0) and u==before
    p=r['matched_pairs'][0]
    assert p['polygon_iou']==0 and 'SOURCE_GEOMETRY_DISJOINT_FROM_CONFIRMED_GTR' in p['disagreement_codes']


def test_T43_retrospective_policy_is_explicit_and_frozen():
    assert mt.POLICY['adopted_date']=='2026-11-01' and mt.POLICY['chronology']=='RETROSPECTIVE_NOT_PREREGISTERED'
    canonical=json.dumps(mt.POLICY,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    assert hashlib.sha256(canonical).hexdigest()==mt.POLICY_SHA256
    with_threshold=copy.deepcopy(mt.POLICY);with_threshold['bbox_threshold_for_primary']=.5
    assert counts(mt.score_unit_v2(unit(),policy=with_threshold))==(None,None,None)


def test_T32_existing_history_caption_uses_same_v2_sums_without_changing_v1(session):
    from auto_annotation_tool.ranking.mobile_session_metrics import canvas_metric_projection
    from auto_annotation_tool.ranking.mobile_canvas_presentation import summary_caption
    original=core.summary(session);before=copy.deepcopy(original);result=mt.evaluate_session(session)
    # Legacy complete-scene values can differ from the independent v2 domain.
    original.update(geometry_n=0,polygon_iou_mean_pct=None,nmce_mean_pct=None)
    projected=canvas_metric_projection(original,result)
    assert projected['geometry_n']==result['geometry_summary']['n_geometry']==1
    assert original['geometry_n']==0 and before['source_archive_sha256']==original['source_archive_sha256']
    text=summary_caption(dict(original,mt_object_evaluation_v2=result),dict(inspected=1,total=1))
    assert '(n=1)' in text


def test_T24_T32_empty_read_count_comes_from_conditional_v2_rows(session):
    from auto_annotation_tool.ranking.mobile_session_metrics import canvas_metric_projection
    result=mt.evaluate_session(session);result['ocr_summary']['rows']=[mz.score_readout(readout(raw_prediction='')),
        mz.score_readout(readout(mt_outcome='FP',operator_decision='NOT_PLATE',raw_prediction=''))]
    result['ocr_summary'].update(mz.summarize_readouts(result['ocr_summary']['rows']))
    assert canvas_metric_projection(core.summary(session),result)['conditional_empty_reads']==1
