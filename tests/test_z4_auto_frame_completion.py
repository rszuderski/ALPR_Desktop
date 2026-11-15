"""A01-D08: source-bound assisted frame completion without an image gesture."""
from pathlib import Path
import copy,hashlib,json,shutil,sys
from types import SimpleNamespace as NS
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_canvas_partial_review as partial,mobile_candidate_stacks as stacks,mobile_mt_object_evaluation as mt,mobile_mt_object_export as export
from auto_annotation_tool.ranking import mobile_frame_review_state as frame
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
from auto_annotation_tool.ranking.mobile_verification_policy import verification_issues
from auto_annotation_tool.gui.mobile_canvas_review_tasks import for_scene,image_action
from test_z4_mt_geometry_layers import source_session
from test_mobile_p2b2_canvas_review import MANUAL
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/Z4_AUTO_FRAME_COMPLETION_20261102_001'
SID='exp-1792702658896-47b4e60b';G7=SID+'-a00000012';G5=SID+'-a00000007';G='mt-one';OID='GT:a0'
P1='PGT:935580407c757a47393f51d2';P2='PGT:2092896f8c02ea9acf99642b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
def frozen(s):return digest(s.review.to_dict()),s.sidecar_path.read_bytes(),s.sidecar_path.stat().st_mtime_ns
def ready(s,g=G):
    sc=core.scene(s,g);r=frame.for_session(s,g,sc)
    assert r['state']==frame.READY and r['metric_complete'] and r['complete']
    assert not any(t['kind']=='SCENE' for t in for_scene(s,g)['steps'])
    assert build_inventory(s)['groups'][g]['status']=='READY'
    return sc,r,mt.evaluate_unit(s,g)
def accepted(path,n=1,mz=False):
    s=source_session(path,n=n,mz=mz)
    for i in range(n):partial.confirm_stack(s,G,oid=f'GT:a{i}')
    return s
@pytest.fixture
def actual(tmp_path):
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'));row=next(x for x in b['sessions'] if x['session_id']==SID)
    shutil.copyfile(row['private_source'],tmp_path/'source.alprsession');shutil.copyfile(row['private_sidecar'],tmp_path/'review.json')
    return MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')
@pytest.fixture
def red(tmp_path):
    for n in ('source.alprsession','review.json'):shutil.copyfile(OUT/'private/red07'/n,tmp_path/n)
    return MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

def test_A01_real_image07_ready_from_existing_decisions_without_check(red):
    s=red;before=frozen(s);sc,r,score=ready(s,G7)
    assert not sc['checked'] and set(r['object_decisions'].values())=={'PLATE','NOT_PLATE'}
    assert (score['tp'],score['fp'],score['fn'])==(1,1,0) and frozen(s)==before
    row=build_inventory(s)['groups'][G7];assert image_action([dict(row,review_tasks=for_scene(s,G7))])[0]=='Zweryfikowany'

def test_A02_real_image05_keeps_cleaned_GTR_and_result(actual):
    s=actual;before=frozen(s);sc,r,score=ready(s,G5);v=stacks.snapshot(s,G5)
    assert set(v['plates'])=={P1} and P2 not in v['active_plate_ids'] and (score['tp'],score['fp'],score['fn'])==(1,1,0)
    assert frozen(s)==before

def test_A03_one_Space_one_revision_no_second_reference_confirmation(tmp_path):
    s=source_session(tmp_path,mz=False);r=s.review.review_revision;partial.toggle_stack(s,G,oid=OID)
    sc,f,score=ready(s);assert s.review.review_revision==r+1 and not sc['checked'] and score['tp']==1
    assert not f['pending_gtr_ids'] and sc['frame_completion_v1']['completion_basis']==frame.BASIS

def test_A04_one_BAD_zero_GT_is_operator_scope_FP_without_scene_task(tmp_path):
    s=source_session(tmp_path,mz=False);r=s.review.review_revision;stacks.toggle_unassociated(s,G,OID,partial_review=True)
    sc,f,score=ready(s);assert s.review.review_revision==r+1 and not sc['checked']
    assert (score['n_gt'],score['tp'],score['fp'],score['fn'])==(0,0,1,0)
    assert score['recall'] is None and score['completion_eligibility']['fn_scope']=='saved_physical_GT_only'

def test_A05_two_FINAL_one_unresolved_stays_one_decision(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,G,oid=OID)
    f=partial.progress(core.scene(s,G));assert f['state']==frame.PENDING and f['decisions']==1
    assert mt.evaluate_unit(s,G)['status']=='REVIEW_REQUIRED'

def test_A06_last_BAD_makes_ready_immediately_in_same_write(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,G,oid=OID);r=s.review.review_revision
    stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True);sc,f,score=ready(s)
    assert s.review.review_revision==r+1 and (score['tp'],score['fp'],score['fn'])==(1,1,0) and not sc['checked']

def test_A07_BBOX_and_Pose_are_one_FINAL_decision(tmp_path):
    s=accepted(tmp_path);sc,f,score=ready(s)
    assert f['total']==1 and len(score['source_finals'])==1 and score['n_final_mt']==1 and score['tp']==1

def test_A08_RAW_suppressed_is_diagnostic_and_never_a_task(actual):
    sc,f,r=ready(actual,G5);v=stacks.snapshot(actual,G5)
    assert sum(c['role']=='SUPPRESSED' for c in v['candidates'])==r['n_raw_suppressed']==1
    assert f['total']==r['n_final_mt']==2 and not for_scene(actual,G5)['units']

def test_A09_BAD_retires_auto_GTR_atomically_without_ghost_task(tmp_path):
    s=accepted(tmp_path);r=s.review.review_revision;stacks.toggle_unassociated(s,G,OID,partial_review=True)
    sc,f,score=ready(s)
    assert s.review.review_revision==r+1 and not stacks.snapshot(s,G)['plates'] and not for_scene(s,G)['units']
    assert score['fp']==1 and not f['pending_gtr_ids']

def test_A10_independent_GTR_survives_model_BAD_without_extra_approval(tmp_path):
    s=accepted(tmp_path);stacks.edit_gt(s,G,OID,text='INDEPENDENTQA');before=digest(stacks.snapshot(s,G)['plates'])
    stacks.toggle_unassociated(s,G,OID,partial_review=True);sc,f,score=ready(s)
    assert digest(stacks.snapshot(s,G)['plates'])==before and (score['tp'],score['fp'],score['fn'])==(0,1,1)

def test_B01_manual_D_is_positive_reference_and_FN_in_one_write(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);r=s.review.review_revision;core.add_plate(s,G,MANUAL,partial_review=True)
    sc,f,score=ready(s);assert s.review.review_revision==r+1 and score['fn']==1 and not f['pending_gtr_ids']

def test_B02_D_corner_correction_keeps_ID_and_auto_readiness(tmp_path):
    s=accepted(tmp_path);v=stacks.snapshot(s,G);pid=next(iter(v['plates']));q=copy.deepcopy(v['plates'][pid]['gt_quad']);q[0][0]+=1
    r=s.review.review_revision;stacks.edit_gt(s,G,OID,quad=q);sc,f,score=ready(s)
    assert set(stacks.snapshot(s,G)['plates'])=={pid} and s.review.review_revision==r+1 and score['tp']==1

def test_B03_R_keeps_real_geometry_task_without_SCENE(tmp_path):
    s=accepted(tmp_path);stacks.edit_gt(s,G,OID,text='REALQA');stacks.remove_linked_geometry(s,G,OID)
    f=partial.progress(core.scene(s,G));task=for_scene(s,G)
    assert f['state']==frame.PENDING and f['pending_gtr_ids'] and task['count']==1
    assert 'D' in task['next_action'] and all(t['kind']!='SCENE' for t in task['steps'])

def test_B04_shared_stack_has_one_positive_and_no_layer_confirmations(tmp_path):
    s=accepted(tmp_path);v=stacks.snapshot(s,G);pid=next(iter(v['plates']));before=frozen(s)
    assert partial.confirm_stack(s,G,plate_id=pid) is False and frozen(s)==before
    assert ready(s)[2]['tp']==1

def test_B05_Space_unconfirms_then_Undo_Redo_restores_readiness(tmp_path):
    s=accepted(tmp_path);partial.toggle_stack(s,G,oid=OID)
    assert partial.progress(core.scene(s,G))['state']==frame.PENDING
    stacks.restore_history(s,G);assert ready(s)[2]['tp']==1
    stacks.restore_history(s,G,redo=True);assert partial.progress(core.scene(s,G))['state']==frame.PENDING

def test_B06_second_PPM_returns_to_unresolved_not_an_old_OK(tmp_path):
    s=accepted(tmp_path);stacks.toggle_unassociated(s,G,OID,partial_review=True);ready(s)
    stacks.toggle_unassociated(s,G,OID,partial_review=True);f=partial.progress(core.scene(s,G))
    assert f['state']==frame.PENDING and f['object_decisions'][OID]==partial.UNRESOLVED

def test_B07_Undo_Redo_last_FINAL_agrees_with_inventory(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,G,oid=OID);stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True)
    assert not verification_issues(s) and build_inventory(s)['remaining']==0
    stacks.restore_history(s,G);assert build_inventory(s)['remaining']==1 and verification_issues(s)
    stacks.restore_history(s,G,redo=True);assert build_inventory(s)['remaining']==0 and not verification_issues(s)

def test_B08_GTZ_edit_changes_OCR_only_without_scene_task(tmp_path):
    s=accepted(tmp_path,mz=True);before=ready(s)[2]
    stacks.edit_gt(s,G,OID,text='QUALITYQA');after=ready(s)[2]
    assert (before['tp'],before['fp'],before['fn'])==(after['tp'],after['fp'],after['fn'])
    assert not any(x['kind']=='SCENE' for x in for_scene(s,G)['steps'])

def test_B09_Missing_GTZ_does_not_block_MT(tmp_path):
    s=accepted(tmp_path,mz=True);ready(s)
    assert mt.evaluate_session(s)['ocr_summary']['n_reads']==0
    assert all(x['kind']=='GTZ' for x in for_scene(s,G)['steps'])

def test_B10_repeat_read_inventory_and_open_are_read_only(red):
    s=red;before=frozen(s)
    for _ in range(2):ready(s,G7);for_scene(s,G7)
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);ready(reopened,G7)
    assert frozen(s)==before

def test_B11_next_navigation_needs_no_scene_step(red):
    before=frozen(red);inv=build_inventory(red)
    assert inv['groups'][G7]['status']=='READY' and not for_scene(red,G7)['needs_review']
    assert frozen(red)==before

def test_B12_two_plates_two_FINALs_count_each_once(tmp_path):
    s=accepted(tmp_path,n=2);sc,f,score=ready(s)
    assert (score['n_gt'],score['n_final_mt'],score['tp'],score['fp'],score['fn'])==(2,2,2,0,0)

def test_C01_empty_is_NO_REFERENCE_not_TN_or_full_recall(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);before=frozen(s);f=frame.for_session(s,G);r=mt.evaluate_unit(s,G)
    assert f['state']==frame.NO_REFERENCE and f['complete'] and not f['metric_complete']
    assert r['status']=='NOT_EVALUABLE' and all(r[k] is None for k in ('tp','fp','fn','recall')) and 'tn' not in r
    assert not for_scene(s,G)['needs_review'] and frozen(s)==before

def test_C02_no_FINAL_one_manual_reference_is_observable_FN(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);core.add_plate(s,G,MANUAL,partial_review=True)
    assert ready(s)[2]['fn']==1

def test_C03_technical_NOT_RUN_is_NA_without_operator_task(tmp_path):
    s=source_session(tmp_path,mz=False);g=s.mt_invocations[G]
    s.mt_invocations[G]=NS(executed=False,cancelled=True,execution_failed=False,records=g.records,subject_keys=g.subject_keys)
    r=frame.for_session(s,G);v=mt.evaluate_unit(s,G,cache=False)
    assert r['state']==frame.SOURCE_NA and r['complete'] and not r['decisions'] and v['tp'] is v['fp'] is v['fn'] is None
    assert build_inventory(s)['remaining']==0

def test_C04_bad_ROI_unknown_FINAL_are_NA_not_zero(tmp_path):
    s=accepted(tmp_path);sc=core.scene(s,G);sc['roi']=[999,999,1000,1000]
    assert frame.from_scene(sc)['state']==frame.SOURCE_NA
    sc=core.scene(s,G);sc['_post_nms_unknown']='UNPROVEN'
    assert frame.from_scene(sc)['state']==frame.SOURCE_NA and not frame.from_scene(sc)['metric_complete']

def test_C05_zero_area_BBOX_valid_Pose_keeps_object_TP(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def zero(rows,trace,payload):
        for row in rows:row.update(plate_left=150,plate_top=140,plate_right=300,plate_bottom=140,mt_confidence='0.9')
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=zero);s.set_review_mode('assisted');partial.confirm_stack(s,G,oid=OID)
    sc,f,r=ready(s);assert r['tp']==1 and r['matched_pairs'][0]['bbox_iou'] is None and r['matched_pairs'][0]['polygon_iou_pct'] is not None

def test_C06_BAD_MZ_execution_timing_kept_but_not_CER(actual):
    from auto_annotation_tool.gui.mobile_mz_quality_context import snapshot
    s=actual;oid='GT:'+SID+'-a00000009';aid=oid[3:]
    before=next(r for r in snapshot(s)['rows'] if r['attempt_id']==aid)
    stacks.toggle_unassociated(s,G5,oid,partial_review=True);stacks.toggle_unassociated(s,G5,oid,partial_review=True)
    after=next(r for r in snapshot(s)['rows'] if r['attempt_id']==aid)
    assert after['executed'] and after['quality_state']=='EXECUTED_EXCLUDED_MT_FP' and not after['canonical_ocr_eligible']
    assert digest(after['timing'])==digest(before['timing']) and digest(after['raw_prediction'])==digest(before['raw_prediction'])

def test_C07_missing_Pose_with_manual_GTR_has_TP_and_geometry_NA(tmp_path):
    from test_z4_akt89lm_resolution import invalid_source
    from test_z4_mt_detection_evaluation import with_source_confidence
    from auto_annotation_tool.ranking.mobile_mt_geometry_layers import create_reference
    original=invalid_source(tmp_path/'original');s=with_source_confidence(original,tmp_path)
    q=[[150,140],[300,140],[300,210],[150,210]];create_reference(s,G,OID,q)
    sc,f,r=ready(s);assert r['tp']==1 and r['matched_pairs'][0]['polygon_iou_pct'] is None and r['matched_pairs'][0]['nmce_pct'] is None

def test_C08_missing_GTZ_preserves_MT_and_separate_OCR_denominator(tmp_path):
    s=accepted(tmp_path,mz=True);r=mt.evaluate_session(s)
    assert r['n_qualified']==1 and r['tp']==1 and r['ocr_summary']['n_reads']==0
    assert not core.scene(s,G)['checked']

def test_C09_historical_checked_true_remains_byte_exact(actual):
    before=frozen(actual);sc=core.scene(actual,G7);assert sc['checked']
    ready(actual,G7);assert frozen(actual)==before

def test_C10_historical_false_ready_has_retrospective_basis_without_write(red):
    before=frozen(red);sc,f,r=ready(red,G7);e=r['completion_eligibility']
    assert not sc['checked'] and e['completion_basis']==frame.BASIS and e['gt_scope']=='operator_reference_scope'
    assert e['adopted_date']=='2026-11-02' and not e['independent_exhaustive_scene_gt'] and frozen(red)==before

def test_C11_unresolved_FINAL_can_never_be_ready(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);f=frame.for_session(s,G)
    assert f['state']==frame.PENDING and not f['complete'] and f['decisions']==1

def test_C12_twelve_saved_reviews_never_migrated_on_read():
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    for row in b['sessions']:
        p=Path(row['private_sidecar']);before=p.read_bytes() if p.exists() else None
        s=MobileReviewSession.open(row['private_source'],sidecar_path=p);mt.evaluate_session(s);build_inventory(s)
        assert before==(p.read_bytes() if p.exists() else None)
        if row.get('review_revision') is not None:assert s.review.review_revision==row['review_revision']

def test_D01_real_image05_policy_and_counts_unchanged(actual):
    sc,f,r=ready(actual,G5)
    assert mt.POLICY_SHA256=='c5461594eae9a070d5af37d64a20ba23abd006c4a999a59e2f3d17e3fe4be3f0'
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0)

def test_D02_matching_1to1_unchanged_for_duplicate_FINAL():
    from test_f4_eval2_object_scoring import unit
    u=unit(n_gt=1,n_final=2);r=mt.score_unit_v2(u)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0) and r['final_outcomes'][1]['reason']=='DUPLICATE_FINAL'

def test_D03_new_export_has_eligibility_review_SHA_and_replay(red,tmp_path):
    s=red;before=frozen(s);paths=export.export(s,tmp_path/'exports');root=paths[0].parent;m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert m['completion_eligibility']['sha256']==frame.ELIGIBILITY_SHA256 and m['source_review_sha256']==sha(s.sidecar_path)
    assert m['review_revision']==s.review.review_revision and export.replay(root)['matching_policy_sha256']==mt.POLICY_SHA256
    assert frozen(s)==before

def test_D04_old_checked_only_failure_now_measured_with_explicit_contract(red):
    before=json.loads((OUT/'red_before.json').read_text(encoding='utf-8'));r=mt.evaluate_unit(red,G7)
    assert before['score_status']=='REVIEW_REQUIRED' and not before['checked'] and r['status']=='MEASURED'
    assert r['completion_eligibility']['completion_basis']==frame.BASIS

def test_D05_export_after_last_FINAL_adds_no_revision(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,G,oid=OID);r=s.review.review_revision
    stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True);assert s.review.review_revision==r+1
    before=frozen(s);paths=export.export(s,tmp_path/'export');assert frozen(s)==before and export.replay(paths[0].parent)['tp']==1

def test_D06_no_image_inspection_controller_or_active_checkbox_source():
    assert not (ROOT/'auto_annotation_tool/gui/mobile_canvas_image_inspection.py').exists()
    text=(ROOT/'auto_annotation_tool/gui/mobile_canvas_clean_view.py').read_text(encoding='utf-8')
    assert 'Sprawdź cały obraz' not in text and 'ed.approve=tk.Checkbutton' not in text and 'clean_empty_button' not in text

def test_D07_no_SCENE_task_can_create_hidden_action(red):
    before=frozen(red);task=for_scene(red,G7)
    assert not task['units'] and not task['steps'] and task['next_action']=='Obraz zweryfikowany' and frozen(red)==before

def test_D08_wrong_source_and_conflicting_alias_rejected(tmp_path):
    s=accepted(tmp_path);r=mt.evaluate_unit(s,G);bad=copy.deepcopy(r);bad['tp']=500
    result=mt.aggregate_units([r,bad]);assert result['tp'] is None and result['alias_conflicts']
    before=s.sidecar_path.read_bytes()
    with s.path.open('ab') as f:f.write(b'PRIVATE_QA_SOURCE_DRIFT')
    r=mt.evaluate_unit(s,G,cache=False);assert all(r[k] is None for k in ('tp','fp','fn')) and s.sidecar_path.read_bytes()==before
