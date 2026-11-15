"""N01-N36. Actual unknown archive plus explicitly labelled source fixtures.

The synthetic per-ID protocol is deliberately absent from the real archive.
Tests never invent source roles for production data or write its sidecar.
"""
from pathlib import Path
import copy,csv,io,json,shutil,zipfile,hashlib
import pytest
from auto_annotation_tool.ranking import mobile_mt_postdedup as nms,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_frame_review_state as frame,mobile_mt_object_evaluation as mt,mobile_mt_object_export as exporting
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui.mobile_canvas_review_tasks import for_scene
from test_z4_mt_geometry_layers import source_session
from test_mobile_p2b2_canvas_review import MANUAL
from test_z4_auto_frame_completion import OUT,G,OID,frozen,sha
SID='exp-1792702907090-d3edcf48';G6=SID+'-a00000008'

def csv_bytes(rows):
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);return b.getvalue().encode()

def synthetic(path,*,raw=4,final=(0,1),explicit=True,mutate=None):
    template=source_session(path/'template',n=raw,mz=True)
    with zipfile.ZipFile(template.path) as z:payload={n:z.read(n) for n in z.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode())))
    samples=list(csv.DictReader(io.StringIO(payload['samples/index.csv'].decode())))
    for i,r in enumerate(rows):
        r.update(mz_executed='true' if i in final else 'false',rectification_status='OK' if i in final else 'NOT_RUN',
            mz_status='READ' if i in final else 'NOT_RUN')
    samples=[r for r in samples if r['attempt_id'] in {f'a{i}' for i in final}]
    trace=json.loads(payload['telemetry/traces.jsonl']);trace['frame_id']=7
    trace['counters'].update(plate_detections_raw=raw,plate_detections_after_dedup=len(final),plate_detections_suppressed=raw-len(final),mz_runs=len(final))
    if explicit:
        trace['source_roles_v2']=dict(schema='alpr.source_nms_roles.v2',complete=True,mt_invocation_id=G,
            source_frame_entry=rows[0]['source_frame_entry'],roi=[100,60,900,640],
            final_ids=[f'a{i}' for i in final],suppressed_ids=[f'a{i}' for i in range(raw) if i not in final])
    if mutate:mutate(rows,trace,samples,payload)
    payload['samples/attempts.csv']=csv_bytes(rows);payload['samples/index.csv']=csv_bytes(samples)
    payload['telemetry/traces.jsonl']=json.dumps(trace).encode()
    target=path/'labelled_source.alprsession'
    with zipfile.ZipFile(target,'w') as z:
        for n,b in payload.items():z.writestr(n,b)
    s=MobileReviewSession.open(target,sidecar_path=path/'review.json');s.set_review_mode('assisted');return s

@pytest.fixture
def actual06(tmp_path):
    for n in ('source.alprsession','review.json'):shutil.copyfile(OUT/'private/annex_image06'/n,tmp_path/n)
    return MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

def proven(s,finals,raws):
    before=frozen(s);p=nms.canvas_post_nms_proof(s,G);v=stacks.snapshot(s,G)
    assert set(nms.final_ids(p))==set(finals) and set(nms.suppressed_ids(p))==set(raws)
    assert {c['attempt_id'] for c in v['candidates'] if c['role']=='FINAL'}==set(finals)
    assert {c['attempt_id'] for c in v['candidates'] if c['role']=='SUPPRESSED'}==set(raws)
    assert frozen(s)==before;return p

def unknown(s,g=G):
    before=frozen(s);d=nms.source_roles_v2(s,g)
    assert d['proof_status'] in {'UNPROVEN','CONFLICT'} and not nms.canvas_post_nms_proof(s,g)
    f=frame.for_session(s,g);r=mt.evaluate_unit(s,g,cache=False)
    assert f['state']==frame.SOURCE_NA and not f['metric_complete'] and r['status']=='NOT_EVALUABLE'
    assert r['tp'] is r['fp'] is r['fn'] is None and frozen(s)==before
    return d,r

def finish(s):
    for c in stacks.snapshot(s,G)['candidates']:
        if c['role']=='FINAL':partial.confirm_stack(s,G,oid=c['object_id'])
    assert frame.for_session(s,G)['state']==frame.READY and not core.scene(s,G)['checked']
    return mt.evaluate_unit(s,G,cache=False)

def test_N01_real_archive_SID_SHA_frame_ROI_is_read_only(actual06):
    before=frozen(actual06);sc=core.scene(actual06,G6)
    assert sha(actual06.path)=='9972226234c6ac08f49580d1f70a5eb79000fab1f00b7051121ca3b99f77f351'
    assert actual06.session_id==SID and list(actual06.mt_invocations)[5]==G6
    assert sc['image_entry']==f'samples/frames/{G6}.jpg' and sc['roi']==[0,0,960,1280]
    assert sc['source_binding']=='a078033eb9c6dcd7c50b61b79837db77a14a502e56fe250fe8636d72269968fb' and frozen(actual06)==before

def test_N02_real_trace_counters_are_known_but_IDs_are_not(actual06):
    d,r=unknown(actual06,G6)
    assert (d['raw_count'],d['final_count'],d['suppressed_count'],d['mz_runs'])==(4,2,2,2)
    assert d['final_identity_count']==d['suppressed_identity_count']==0
    assert d['missing_final_id_count']==d['missing_suppressed_id_count']==2

def test_N03_real_four_attempts_indices_are_distinct(actual06):
    rows=actual06.mt_invocations[G6].records
    assert [int(r['mt_detection_index']) for r in rows]==list(range(4))
    assert {r['id'] for r in rows}=={SID+f'-a{i:08d}' for i in range(8,12)}

def test_N04_two_MZ_never_establish_FINAL(actual06):
    assert sum(str(r.get('mz_executed')).lower()=='true' for r in actual06.mt_invocations[G6].records)==2
    d,r=unknown(actual06,G6);assert not r['source_finals'] and 'SOURCE_FINAL_ROLE_UNPROVEN' in r['unavailable_reasons']

def test_N05_counters_without_per_ID_evidence_stay_NA(tmp_path):
    s=synthetic(tmp_path,explicit=False);d,r=unknown(s);assert d['final_count']==2

def test_N06_explicit_4_to_2_proves_all_IDs(tmp_path):
    s=synthetic(tmp_path);p=proven(s,['a0','a1'],['a2','a3']);assert p['suppressed']==2

def test_N07_legacy_2_to_1_proof_is_stable(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path,n=2);p=nms.canvas_post_nms_proof(s,G)
    assert p['suppressed_id']=='a1' and nms.suppressed_ids(p)==['a1'] and 'suppressed_ids' not in p

def test_N08_legacy_3_to_2_still_proven(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path);p=proven(s,['a0','a2'],['a1']);assert p['schema'].endswith('.v1')

def test_N09_explicit_3_to_1(tmp_path):
    proven(synthetic(tmp_path,raw=3,final=(1,)),['a1'],['a0','a2'])

def test_N10_legacy_4_to_3(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    proven(session_fixture(tmp_path,n=4),['a0','a2','a3'],['a1'])

def test_N11_partial_MZ_does_not_make_remaining_FINAL_RAW(tmp_path):
    def change(rows,t,s,p):rows[1].update(mz_executed='false',mz_status='NOT_RUN',rectification_status='FAILED');s[:]=[r for r in s if r['attempt_id']!='a1'];t['counters']['mz_runs']=1
    s=synthetic(tmp_path,mutate=change);proven(s,['a0','a1'],['a2','a3'])

def test_N12_invalid_Pose_FINAL_does_not_become_suppressed(tmp_path):
    def change(rows,t,s,p):rows[1].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',mz_executed='false',mz_status='NOT_RUN',rectification_status='FAILED');s[:]=[r for r in s if r['attempt_id']!='a1'];t['counters']['mz_runs']=1
    s=synthetic(tmp_path,mutate=change);proven(s,['a0','a1'],['a2','a3'])

def test_N13_NOT_RUN_without_source_contract_proves_no_role(tmp_path):
    s=synthetic(tmp_path,raw=3,final=(0,),explicit=False);unknown(s)

def test_N14_conflicting_counters_rejected(tmp_path):
    s=synthetic(tmp_path,mutate=lambda r,t,s,p:t['counters'].update(plate_detections_suppressed=1));d,r=unknown(s)
    assert d['proof_status']=='CONFLICT' and 'NMS_COUNTERS_CONFLICT' in d['reasons']

def test_N15_duplicate_trace_and_wrong_epoch_rejected(tmp_path,monkeypatch):
    s=synthetic(tmp_path);original=nms._trace_index
    for kind in ('duplicate','epoch'):
        def index(*a):
            v=copy.deepcopy(original(*a));key=next(iter(v))
            if kind=='duplicate':v[key].append(copy.deepcopy(v[key][0]))
            else:v[key][0]['visual_epoch']=999
            return v
        monkeypatch.setattr(nms,'_trace_index',index);unknown(s)
    monkeypatch.setattr(nms,'_trace_index',original)
    def old_compatible(rows,t,s,p):
        for key in ('plate_left','plate_top','plate_right','plate_bottom'):rows[1][key]=rows[0][key]
        t['visual_epoch']=999
    two=synthetic(tmp_path/'two',raw=2,final=(0,),mutate=old_compatible)
    assert nms.canvas_post_nms_proof(two,G) is None

def test_N16_incomplete_attempt_population_rejected(tmp_path):
    with pytest.raises(ValueError,match='Niepełny zestaw'):
        synthetic(tmp_path,mutate=lambda r,t,s,p:r.pop())

def test_N17_cancelled_MT_is_not_a_miss(tmp_path):
    from dataclasses import replace
    s=synthetic(tmp_path);s.mt_invocations[G]=replace(s.mt_invocations[G],cancelled=True)
    r=mt.evaluate_unit(s,G,cache=False);assert r['tp'] is r['fp'] is r['fn'] is None
    assert nms.source_roles_v2(s,G)['reasons']==['MT_NOT_COMPLETED']

def test_N18_other_ROI_or_invocation_cannot_supply_role_identity(tmp_path):
    for key,value in [('roi',[0,0,1000,700]),('mt_invocation_id','another-call')]:
        s=synthetic(tmp_path/key,mutate=lambda r,t,s,p:t['source_roles_v2'].update({key:value}));unknown(s)

def test_N19_nearby_RAW_never_becomes_FP_or_TP(tmp_path):
    def overlap(rows,t,s,p):
        for key in ('plate_left','plate_top','plate_right','plate_bottom','mt_pose_corners_norm'):rows[2][key]=rows[0][key]
    s=synthetic(tmp_path,mutate=overlap);r=finish(s)
    assert (r['tp'],r['fp'],r['fn'],r['n_raw_suppressed'])==(2,0,0,2)
    assert len(stacks.snapshot(s,G)['plates'])==2 and {r['attempt_id'] for r in r['source_finals']}=={'a0','a1'}

def test_N20_existing_manual_GTR_keeps_every_field_and_history(actual06,tmp_path,monkeypatch):
    before=frozen(actual06);sc=core.scene(actual06,G6);v=stacks.snapshot(actual06,G6)
    pid='PGT:e4f2e29411a24a1725f26869';plate=copy.deepcopy(v['plates'][pid])
    unknown(actual06,G6);assert stacks.snapshot(actual06,G6)['plates'][pid]==plate
    assert frozen(actual06)==before and len(sc['candidate_stack_history_v1']['entries'])>0
    s=synthetic(tmp_path/'proven-later');original=nms._explicit_role_proof
    monkeypatch.setattr(nms,'_explicit_role_proof',lambda *a:None)
    core.add_plate(s,G,MANUAL,partial_review=True);old=stacks.snapshot(s,G)['plates'];saved=frozen(s)
    monkeypatch.setattr(nms,'_explicit_role_proof',original)
    assert stacks.snapshot(s,G)['plates']==old and frame.for_session(s,G)['decisions']==2
    assert frozen(s)==saved and len(stacks.snapshot(s,G)['plates'])==1

def test_N21_proven_FINAL_decisions_finish_without_image_check(tmp_path):
    s=synthetic(tmp_path);manual=core.add_plate(s,G,MANUAL,partial_review=True)
    pid=next(iter(stacks.snapshot(s,G)['plates']));before=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid])
    stacks.associate(s,G,OID,manual);partial.confirm_stack(s,G,oid=OID);stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True)
    assert frame.for_session(s,G)['state']==frame.READY and for_scene(s,G)['count']==0
    r=mt.evaluate_unit(s,G);assert (r['tp'],r['fp'],r['fn'])==(1,1,0) and not core.scene(s,G)['checked']
    assert set(stacks.snapshot(s,G)['plates'])=={pid} and stacks.snapshot(s,G)['plates'][pid]['gt_quad']==before['gt_quad']

def test_N22_UNKNOWN_preserves_GTR_and_has_no_source_task(actual06):
    before=frozen(actual06);unknown(actual06,G6);task=for_scene(actual06,G6)
    assert task['units']==task['steps']==[] and not task['needs_review'] and 'N/A' in task['next_action']
    assert len(stacks.snapshot(actual06,G6)['plates'])==1 and frozen(actual06)==before

def test_N23_role_is_independent_of_BBOX_Pose_view(tmp_path):
    from auto_annotation_tool.ranking.mobile_mt_geometry_layers import snapshot
    s=synthetic(tmp_path);before=frozen(s)
    for oid in ('GT:a0','GT:a1'):snapshot(s,G,oid)
    with pytest.raises(ValueError,match='FINAL'):snapshot(s,G,'GT:a2')
    proven(s,['a0','a1'],['a2','a3']);assert frozen(s)==before

def test_N24_Shift_has_all_four_identities_read_only(tmp_path):
    from auto_annotation_tool.gui.mobile_canvas_shift_preview import ShiftPreview
    from types import SimpleNamespace as NS
    s=synthetic(tmp_path);before=frozen(s);ed=NS(h=NS(session=s,current_invocation=G),_read=lambda:core.scene(s,G))
    preview=ShiftPreview.__new__(ShiftPreview);preview.ed=ed;rows=preview.frames()
    assert {r['attempt_id'] for r in rows if r['kind']=='BBOX'}=={'a0','a1','a2','a3'}
    assert {r['attempt_id'] for r in rows if r['role']=='SUPPRESSED'}=={'a2','a3'} and frozen(s)==before

def test_N25_UNKNOWN_Space_and_PPM_cannot_save_FINAL(actual06):
    before=frozen(actual06)
    for action in (lambda:partial.confirm_stack(actual06,G6,oid='GT:'+G6),lambda:stacks.toggle_unassociated(actual06,G6,'GT:'+G6,partial_review=True)):
        with pytest.raises(ValueError):action()
        assert frozen(actual06)==before

def test_N26_next_other_image_is_accessible_without_writing(actual06):
    before=frozen(actual06);inv=build_inventory(actual06)
    assert inv['groups'][G6]['status']==frame.SOURCE_NA and inv['groups'][G6]['frame_review']['complete']
    other=next(g for g in actual06.mt_invocations if g!=G6);core.scene(actual06,other);assert frozen(actual06)==before

def test_N27_other_scene_auto_completion_remains_available(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);partial.confirm_stack(s,G,oid=OID)
    assert build_inventory(s)['groups'][G]['status']=='READY' and not core.scene(s,G)['checked']

def test_N28_partial_FP_lower_bound_only_from_proven_FINAL(tmp_path,actual06):
    s=synthetic(tmp_path/'synthetic');stacks.toggle_unassociated(s,G,OID,partial_review=True);r=mt.evaluate_unit(s,G)
    assert r['partial_fp_lower_bound']==1 and r['status']=='REVIEW_REQUIRED'
    assert mt.evaluate_unit(actual06,G6)['partial_fp_lower_bound'] is None

def test_N29_real_NA_export_replays_and_keeps_other_units(actual06,tmp_path):
    before=frozen(actual06);data=mt.evaluate_session(actual06);paths=exporting.export(actual06,tmp_path/'export');replay=exporting.replay(paths[0].parent)
    u=next(u for u in replay['units'] if u['mt_invocation_id']==G6)
    assert 'SOURCE_FINAL_ROLE_UNPROVEN' in u['unavailable_reasons'] and u['tp'] is u['fp'] is u['fn'] is None
    assert json.loads(json.dumps(data))==replay and frozen(actual06)==before

def test_N30_proven_export_counts_and_SHA_replay(tmp_path):
    s=synthetic(tmp_path);finish(s);before=frozen(s);paths=exporting.export(s,tmp_path/'export');r=exporting.replay(paths[0].parent)
    assert r['n_final_mt']==2 and r['n_raw_suppressed']==2 and r['source_archive_sha256']==sha(s.path)
    assert r['units'][0]['source_role_evidence']['proof_status']=='PROVEN' and frozen(s)==before

def test_N31_Undo_Redo_keeps_candidate_and_GT_IDs(tmp_path):
    s=synthetic(tmp_path);finish(s);sc=core.scene(s,G);state=stacks.snapshot(s,G)
    stacks.restore_history(s,G);assert not frame.for_session(s,G)['complete']
    stacks.restore_history(s,G,redo=True);after=stacks.snapshot(s,G)
    assert after['plates']==state['plates'] and core.scene(s,G)['parents']==sc['parents'] and frame.for_session(s,G)['complete']

def test_N32_twelve_archives_and_reviews_are_read_only():
    baseline=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    for row in baseline['sessions']:
        path=Path(row['private_sidecar']);raw=path.read_bytes() if path.exists() else None
        s=MobileReviewSession.open(row['private_source'],sidecar_path=path)
        for gid in s.mt_invocations:nms.source_roles_v2(s,gid)
        mt.evaluate_session(s)
        assert raw==(path.read_bytes() if path.exists() else None) and sha(s.path)==baseline['protected'][row['path']]['sha256']

def test_N33_zero_FINAL_and_GT_is_not_TN_or_full_Recall(tmp_path):
    s=synthetic(tmp_path,raw=3,final=());proven(s,[],['a0','a1','a2']);r=mt.evaluate_unit(s,G)
    assert frame.for_session(s,G)['state']==frame.NO_REFERENCE and r['recall'] is None and r['fn'] is None
    assert 'NO_REFERENCE_EVIDENCE' in r['unavailable_reasons']
    zero=source_session(tmp_path/'zero',n=0,mz=False);d=nms.source_roles_v2(zero,G)
    assert d['proof_status']=='PROVEN' and d['raw_count']==d['final_count']==d['suppressed_count']==0
    assert d['final_ids']==[] and d['attempt_ids']==[] and mt.evaluate_unit(zero,G)['recall'] is None

def test_N34_MZ_trace_count_is_not_FINAL_count(tmp_path):
    s=synthetic(tmp_path,raw=3,final=(0,),mutate=lambda r,t,s,p:t['counters'].update(mz_runs=2))
    proven(s,['a0'],['a1','a2']);d=nms.source_roles_v2(s,G)
    assert d['mz_runs']==2 and d['final_count']==1 and not d['mz_count_is_final_count']

def test_N35_zero_area_BBOX_does_not_discard_proven_FINAL(tmp_path):
    s=synthetic(tmp_path,mutate=lambda r,t,s,p:r[0].update(plate_bottom=r[0]['plate_top']));r=finish(s)
    assert r['tp']==2 and next(p for p in r['matched_pairs'] if p['attempt_id']=='a0')['bbox_iou'] is None

def test_N36_historical_suppressed_id_remains_byte_exact(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path);partial.confirm_stack(s,G,oid='GT:a0');sc=core.scene(s,G);proof=copy.deepcopy(sc['post_nms_evidence']);before=frozen(s)
    assert 'suppressed_id' in proof and 'suppressed_ids' not in proof
    nms.source_roles_v2(s,G);core.scene(s,G);assert frozen(s)==before and core.scene(s,G)['post_nms_evidence']==proof
