"""F5A/12C T18-T27: actual producer archives, private copies, native Desktop."""
from pathlib import Path
from _private_test_data import private_test_path
import hashlib
import json
import shutil

import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_scene_frame_review as scenes, mobile_canvas_review as core

PRODUCER = private_test_path('F5A_12A_ANDROID_DOWODY_20261104')
GOOD = sorted(p.stem for p in (PRODUCER/'fixtures').glob('*.alprsession'))
BAD = sorted(p.stem for p in (PRODUCER/'negative').glob('*.alprsession'))

def private_session(tmp_path, name):
    root=tmp_path/name;root.mkdir(parents=True,exist_ok=True)
    source=PRODUCER/'fixtures'/(name+'.alprsession')
    path=root/'source.alprsession';shutil.copyfile(source,path)
    expected=json.loads((PRODUCER/'delivery_manifest.json').read_text(encoding='utf-8'))['entry_sha256']['fixtures/'+name+'.alprsession']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    session=MobileReviewSession.open(path,sidecar_path=root/'review.json')
    session.set_review_mode('assisted')
    return session

@pytest.mark.parametrize('name',GOOD)
def test_T18_native_producer_positive_and_legacy_import(tmp_path,name):
    session=private_session(tmp_path,name)
    assert session.bundle.validation.ok
    if name.startswith('legacy'):
        assert not session.scene_frames and not session.review.scene_annotations
        assert session.bundle.scene_evidence['status']=='ABSENT'
    else:
        assert len(session.scene_frames)==1
        assert not session.review.scene_annotations
    assert not session.review.invocation_annotations

@pytest.mark.parametrize('name',BAD)
def test_T19_native_producer_negative_is_rejected(name):
    try:
        bundle=read_mobile_report_bundle(PRODUCER/'negative'/(name+'.alprsession'))
    except (ValueError, OSError):
        return
    assert not bundle.validation.ok, name

def test_T20_zero_roi_plate_is_outside_without_mt_fn(tmp_path):
    s=private_session(tmp_path,'zero-roi-with-plate');key=next(iter(s.scene_frames))
    assert len(s.mt_invocations)==len(s.attempts)==0 and not scenes.scene(s,key)['checked']
    oid=core.add_plate(s,key,[(10,40),(80,40),(80,60),(10,60)])
    core.set_text(s,key,oid,'','Obraz syntetyczny: nieczytelny tekst')
    result=scenes.coverage(s,key)
    assert result['n_physical_gt']==1 and result['rows'][0]['coverage']=='OUTSIDE_ROI'
    assert result['rows'][0]['text_exclusion_reason']
    assert result['mt_fn'] is None and result['tv_e2e_reason']=='SCENE_UNMAPPED'
    stats=s.statistics()
    assert stats['mt_object_evaluation_v2']['n_total']==0
    from auto_annotation_tool.ranking.mobile_session_metrics import current,result_key
    assert current(stats['live_session_metrics_v2'],result_key(s))
    assert stats['live_session_metrics_v2']['mt']['precision'] is None

def test_T21_presence_only_never_fabricates_corners_or_ocr(tmp_path):
    s=private_session(tmp_path,'zero-roi-with-plate');key=next(iter(s.scene_frames))
    oid=scenes.add_plate(s,key,geometry_reason='Zasloniete rogi')
    scenes.edit(s,key,oid,text='',reason='Nieczytelne znaki')
    obj=scenes.scene(s,key)['objects'][0]
    assert obj['quad'] is None and obj['physical_gt_id'] and not obj['gt_text']
    assert scenes.coverage(s,key)['rows'][0]['geometry_status']=='N/A'

def test_T22_explicit_zero_is_one_saved_decision_and_undoable(tmp_path):
    s=private_session(tmp_path,'zero-roi');key=next(iter(s.scene_frames))
    assert not scenes.scene(s,key)['checked'] and scenes.issues(s)
    before=s.review.review_revision
    scenes.confirm_empty(s,key)
    assert s.review.review_revision==before+1 and not scenes.issues(s)
    assert scenes.scene(s,key)['presence']=='NO_PLATES'
    scenes.restore(s,key)
    assert scenes.scene(s,key)['presence']=='UNRESOLVED' and scenes.issues(s)
    scenes.restore(s,key,redo=True)
    assert scenes.scene(s,key)['presence']=='NO_PLATES'
    result=scenes.coverage(s,key)
    assert result['n_physical_gt']==0 and result['reason']=='ZERO_DENOMINATOR'

def test_T19_roundtrip_source_scene_identity_history_and_export(tmp_path):
    from auto_annotation_tool.ranking.mobile_scientific_review import make_scientific_summary,write_scientific_companion
    from auto_annotation_tool.gui.mobile_review_export_layout import finalize_export_manifest
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export
    s=private_session(tmp_path,'two-roi');key=next(iter(s.scene_frames))
    raw_sha=hashlib.sha256(s.path.read_bytes()).hexdigest()
    oid=scenes.add_plate(s,key,[(10,30),(60,30),(60,50),(10,50)])
    scenes.edit(s,key,oid,text='SYN123')
    original_id=scenes.scene(s,key)['objects'][0]['physical_gt_id']
    p=scenes.scene(s,key)['objects'][0]['quad'];p[0][0]+=1
    scenes.edit(s,key,oid,quad=p)
    scenes.restore(s,key);scenes.restore(s,key,redo=True)
    before=json.loads(json.dumps(s.review.to_dict()))
    restored=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert restored.review.to_dict()==before
    out=tmp_path/'export';out.mkdir()
    paths=restored.export(out)
    for name,content in [('mt_pose_geometry.csv','attempt_id,metric_status\n'),('mt_multi_plate_verified.json','{}')]:
        path=out/name;path.write_text(content,encoding='utf-8');paths.append(path)
    paths.extend(write_scientific_companion(make_scientific_summary(restored,restored.statistics(),[]),out))
    finalize_export_manifest(restored,out,paths)
    record=load_session_export(out)
    assert record.scene_assessment['scene_annotations']==before['scene_annotations']
    imported=MobileReviewSession.open(s.path,sidecar_path=out/'review.json')
    assert imported.review.scene_annotations==before['scene_annotations']
    assert scenes.scene(imported,key)['objects'][0]['physical_gt_id']==original_id
    assert imported.mt_invocations.keys()==s.mt_invocations.keys()
    assert imported.mt_detection_roles==s.mt_detection_roles
    assert len(imported.mt_invocations)==2 and len(imported.scene_frames)==1
    assert imported.review.review_revision==s.review.review_revision
    assert hashlib.sha256(s.path.read_bytes()).hexdigest()==raw_sha

@pytest.mark.parametrize('name',['one-roi','two-roi','dedup','fallback','shared-image','cancelled'])
def test_T19_exact_per_call_roles_and_shared_images(tmp_path,name):
    from auto_annotation_tool.ranking.mobile_mt_postdedup import source_roles_v2
    s=private_session(tmp_path,name)
    for gid,role in s.mt_detection_roles.items():
        proof=source_roles_v2(s,gid)
        assert proof['recorded_final_ids']==role['final_attempt_ids']
        assert proof['recorded_suppressed_ids']==role['suppressed_attempt_ids']
        if not role['stale_or_cancelled']:
            assert proof['proof_status']=='PROVEN'
    if name=='shared-image':
        assert (len(s.attempts),len(s.mt_invocations),len({r['source_frame_entry'] for r in s.attempts.values()}))==(47,40,40)
        assert len(s.scene_frames)==1
    if name=='fallback':
        assert len(s.mt_invocations)==3
        assert [r['roi'] for r in s.mt_detection_roles.values()][-1]==[0,0,160,100]

def test_T23_missing_is_na_and_not_a_negative(tmp_path):
    s=private_session(tmp_path,'missing-frame');key=next(iter(s.scene_frames))
    assert core.context(s,key) is None
    with pytest.raises(ValueError,match='N/A'):
        scenes.confirm_empty(s,key)
    result=scenes.summary(s)
    assert result['n_missing']==1 and result['n_assessed']==0
    assert result['evidence_status']=='PARTIAL' and not result['scene_population_complete']
    assert result['scenes'][0]['n_physical_gt'] is None
    assert not s.review.scene_annotations


def test_T19_one_physical_reference_can_explicitly_link_two_roi_calls(tmp_path):
    from auto_annotation_tool.ranking.mobile_mt_object_evaluation import evaluate_unit
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    s=private_session(tmp_path,'two-roi');key=next(iter(s.scene_frames))
    oid=scenes.add_plate(s,key,[(72,40),(78,40),(78,50),(72,50)])
    aid1,aid2=[r['final_attempt_ids'][0] for r in s.mt_detection_roles.values()]
    scenes.decide_mt(s,key,aid1,reference=oid)
    assert evaluate_unit(s,list(s.mt_invocations)[1])['n_qualified']==0
    scenes.decide_mt(s,key,aid2,reference=oid)
    ids=[]
    for gid in s.mt_invocations:
        result=evaluate_unit(s,gid)
        assert result['n_qualified']==1 and (result['tp'],result['fp'],result['fn'])==(1,0,0)
        ids.append(result['matched_pairs'][0]['gt_id'])
    assert len(set(ids))==1 and len(scenes.scene(s,key)['objects'])==1
    assert build_inventory(s)['operator_assessed']==2
    assert scenes.summary(s)['n_assessed']==1
    live=s.statistics()['live_session_metrics_v2']
    from auto_annotation_tool.ranking.mobile_session_metrics import current,result_key
    assert current(live,result_key(s))
    assert live['mt']['precision']==live['mt']['recall']==live['mt']['f1']==1.0
    assert live['geometry']['n_geometry']==2
    restored=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert restored.review.scene_annotations==s.review.scene_annotations



def test_T20_executed_empty_roi_can_be_fn_but_not_run_never_is(tmp_path):
    from auto_annotation_tool.ranking.mobile_mt_object_evaluation import evaluate_unit
    s=private_session(tmp_path,'fallback');key=next(iter(s.scene_frames))
    scenes.add_plate(s,key,[(20,40),(60,40),(60,60),(20,60)])
    first,second,_=s.mt_invocations
    result=evaluate_unit(s,first)
    assert result['n_qualified']==1 and result['fn']==1
    negative=evaluate_unit(s,second)
    assert negative['n_qualified']==1 and negative['n_gt']==0
    assert negative['precision'] is None and negative['recall'] is None
    assert s.scene_frames[key]['mp_observations']==[]
    assert scenes.coverage(s,key)['reason']=='ROI_COVERAGE_UNPROVEN'

def test_T19_history_snapshots_remain_immutable_across_later_edits(tmp_path):
    import copy
    s=private_session(tmp_path,'zero-roi-with-plate');key=next(iter(s.scene_frames))
    oid=scenes.add_plate(s,key,[(20,30),(60,30),(60,50),(20,50)])
    old_review=s.review
    old=copy.deepcopy(old_review.scene_annotations)
    scenes.edit(s,key,oid,text='SYN001')
    assert old_review.scene_annotations==old
    scenes.restore(s,key)
    assert old_review.scene_annotations==old



def test_T22_scene_sidecar_has_no_fictional_invocation_or_selected_roi(tmp_path):
    s=private_session(tmp_path,'zero-roi');key=next(iter(s.scene_frames))
    scenes.confirm_empty(s,key)
    persisted=s.review.scene_annotations[key]
    assert 'mt_invocation_id' not in persisted and 'roi' not in persisted
    assert persisted['edit_bounds']==[0,0,160,100]
    assert len(s.mt_invocations)==0 and len(s.attempts)==0
    s.complete()
    assert s.review.review_status=='COMPLETED'
    assert MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path).review.review_status=='COMPLETED'


def test_T27_partial_export_rejected_without_backfill(tmp_path):
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export,InvalidExport
    # Native manifest validation refuses missing dependencies before reading metrics.
    out=tmp_path/'partial';out.mkdir()
    (out/'export_manifest.json').write_text(json.dumps(dict(schema='alpr_mobile_review_export_manifest_v1',
        files=[dict(path='missing/scene_frame_assessment.json',bytes=1,sha256='a'*64)])))
    with pytest.raises(InvalidExport,match='Missing'):
        load_session_export(out)
    assert not (out/'missing').exists()
