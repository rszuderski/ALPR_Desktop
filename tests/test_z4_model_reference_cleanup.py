"""P0 model-only reference lifecycle and audited production-repair dry runs.

All mutations use private copies. Historical GTR/GTZ and F4 scoring stay intact.
"""
from pathlib import Path
import copy,hashlib,json,os,shutil,subprocess,sys
import pytest
from auto_annotation_tool.ranking import mobile_human_review as review,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_object_evaluation as mt
from auto_annotation_tool.ranking import mobile_reference_lifecycle as life,mobile_reference_reconciliation as repair,mobile_reference_disposition as legacy
from auto_annotation_tool.gui import mobile_canvas_review_tasks as tasks,mobile_canvas_review_feedback as feedback
from test_z4_mt_geometry_layers import source_session

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Z4_GHOST_CLEANUP_20261102_001'
SID='exp-1792702658896-47b4e60b';G=SID+'-a00000007';GOOD='GT:'+G;BAD='GT:'+SID+'-a00000009';RAW='GT:'+SID+'-a00000008'
P1='PGT:935580407c757a47393f51d2';P2='PGT:2092896f8c02ea9acf99642b'
SHA='aab5df72a72ab422e8c7edcae4cc6b323d68599be87cb2f21ab01ff95fca562f'
BINDING='f5d24e043cdbd34161b3cc6c955df69d332010b702e3953dad496d1b333c620a'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
def frozen(s):return digest(s.review.to_dict()),s.sidecar_path.read_bytes(),s.sidecar_path.stat().st_mtime_ns
def state(s,g=G):return core._legacy_scene(s,g),stacks.snapshot(s,g)
def cleaned(s):
    target=repair.prepare(s,G,P2);backup=s.sidecar_path.parent/'exact.before.json';backup.write_bytes(s.sidecar_path.read_bytes())
    repair.apply(s,target,backup_path=backup);return s
def accepted(path,n=1):
    s=source_session(path,n=n,mz=False)
    for i in range(n):partial.confirm_stack(s,'mt-one',oid=f'GT:a{i}')
    return s
def before_withdrawal(s):
    while legacy.withdrawn_ids(state(s)[0]):assert stacks.restore_history(s,G)
    return s

@pytest.fixture
def actual(tmp_path):
    row=next(r for r in json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    shutil.copyfile(row['private_source'],tmp_path/'source.alprsession');shutil.copyfile(row['backup'],tmp_path/'review.json')
    return review.MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

def test_T01_exact_latest_source_binding_revision(actual):
    s=actual;b=frozen(s);t=repair.prepare(s,G,P2)
    assert s.session_id==SID and t['source_archive_sha256']==SHA and t['source_binding']==BINDING
    assert s.review.review_revision==108 and t['review_revision']==108 and t['sidecar_sha256']==hashlib.sha256(b[1]).hexdigest()
    assert frozen(s)==b

def test_T02_FINAL_RAW_source_population_unchanged(actual):
    sc,v=state(actual)
    assert [(c['object_id'],c['role']) for c in v['candidates']]==[(GOOD,'FINAL'),(RAW,'SUPPRESSED'),(BAD,'FINAL')]
    assert len(sc['parents'])==3

def test_T03_T04_full_origin_history_and_independent_GTR1(actual):
    s=actual;sc,v=state(s);before=copy.deepcopy(v['plates'][P1]);t=repair.prepare(s,G,P2)
    assert t['evidence']['created_revision_before']==88
    events={(e['action'],e['review_revision_before']) for e in sc['events']}
    assert {('operator_toggle_frame_presence',88),('remove_operator_reference_geometry',95),('operator_candidate_no_plate',96)}<=events
    cleaned(s);assert state(s)[1]['plates'][P1]==before

def test_T05_independent_GTZ_blocks_cleanup(actual):
    s=before_withdrawal(actual);stacks.edit_gt(s,G,BAD,text='INDEPENDENTQA');before=frozen(s)
    with pytest.raises(ValueError):repair.prepare(s,G,P2)
    assert frozen(s)==before and state(s)[1]['plates'][P2]['gt_text']=='INDEPENDENTQA'

def test_T06_shared_reference_blocks_cleanup(tmp_path):
    s=accepted(tmp_path);g='mt-one';sc,v=state(s,g);pid=v['links']['GT:a0']['plate_id']
    # A second explicit source relation, even to a diagnostic source, protects
    # the real reference; no geometry-based decision is used.
    v['links']['OTHER']=dict(plate_id=pid,status='ASSOCIATED_FOR_GEOMETRY')
    with pytest.raises(ValueError,match='SHARED'):life.dependency_evidence(sc,v,pid)

def test_T07_check_command_writes_no_input_or_workspace_file(actual):
    s=actual;before=frozen(s);paths={p.name:p.stat().st_mtime_ns for p in s.sidecar_path.parent.iterdir()}
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reconcile_false_model_reference.py'),'--check','--source',str(s.path),'--sidecar',str(s.sidecar_path),'--invocation',G,'--plate-id',P2],cwd=ROOT,capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    assert b'READY' in result.stdout and frozen(s)==before
    assert paths=={p.name:p.stat().st_mtime_ns for p in s.sidecar_path.parent.iterdir()}

def test_T08_existing_inspection_is_not_fabricated_or_reset(actual):
    s=actual;old=state(s)[0];assert old['checked']
    cleaned(s);assert state(s)[0]['checked']
    assert state(s)[0]['events'][:-1]==old['events']
    assert state(s)[0]['events'][-1]['actor']=='PROGRAM_RECONCILIATION'

def test_T09_byte_backup_and_original_metadata(actual):
    row=next(r for r in json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    baseline=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))['protected'][row['sidecar']]
    assert hashlib.sha256(Path(row['backup']).read_bytes()).hexdigest()==baseline['sha256']
    assert len(Path(row['backup']).read_bytes())==baseline['size'] and row['review_revision']==108
    assert row['review_id']==actual.review.review_id and baseline['mtime_ns']>0

def test_T10_same_revision_CAS_race_preserves_new_data(actual):
    s=actual;t=repair.prepare(s,G,P2);backup=s.sidecar_path.parent/'before.json';backup.write_bytes(s.sidecar_path.read_bytes())
    d=json.loads(s.sidecar_path.read_bytes());d['reviewer_id']='OTHER_WRITER';s.sidecar_path.write_text(json.dumps(d),encoding='utf-8');before=frozen(s)
    with pytest.raises(ValueError):repair.apply(s,t,backup_path=backup)
    assert frozen(s)==before

def test_T10_CAS_rechecked_at_atomic_transaction(actual,monkeypatch):
    s=actual;t=repair.prepare(s,G,P2);backup=s.sidecar_path.parent/'before.json';backup.write_bytes(s.sidecar_path.read_bytes())
    original=s._save_change_locked;external=[];memory=digest(s.review.to_dict())
    def race(*a,**kw):
        d=json.loads(s.sidecar_path.read_bytes());d['reviewer_id']='CONCURRENT_QA'
        s.sidecar_path.write_text(json.dumps(d),encoding='utf-8');external.append(s.sidecar_path.read_bytes())
        return original(*a,**kw)
    monkeypatch.setattr(s,'_save_change_locked',race)
    with pytest.raises(ValueError):repair.apply(s,t,backup_path=backup)
    assert s.sidecar_path.read_bytes()==external[0] and digest(s.review.to_dict())==memory
    assert not s.sidecar_path.with_suffix(s.sidecar_path.suffix+'.write.lock').exists()

def test_T11_writer_lock_is_preserved(actual):
    s=actual;t=repair.prepare(s,G,P2);backup=s.sidecar_path.parent/'before.json';backup.write_bytes(s.sidecar_path.read_bytes())
    lock=s.sidecar_path.with_suffix(s.sidecar_path.suffix+'.write.lock');lock.write_text('OTHER_WRITER');before=frozen(s)
    with pytest.raises(ValueError,match='innym oknie'):repair.apply(s,t,backup_path=backup)
    assert frozen(s)==before and lock.read_text()=='OTHER_WRITER'

def test_T12_atomic_write_failure_changes_neither_memory_nor_disk(actual,monkeypatch):
    s=actual;t=repair.prepare(s,G,P2);backup=s.sidecar_path.parent/'before.json';backup.write_bytes(s.sidecar_path.read_bytes());before=frozen(s)
    def fail(*a,**kw):raise OSError('injected atomic failure')
    monkeypatch.setattr(review,'_atomic_write',fail)
    with pytest.raises(OSError):repair.apply(s,t,backup_path=backup)
    assert frozen(s)==before

def test_T13_T14_single_revision_event_and_idempotent_reapply(actual):
    s=actual;old=state(s)[0];r=s.review.review_revision;cleaned(s);after=state(s)[0]
    assert s.review.review_revision==r+1 and len(after['events'])==len(old['events'])+1
    assert after['events'][-1]['action']==repair.ACTION and after['events'][-1]['backup_sha256']==hashlib.sha256(Path(s.sidecar_path.parent/'exact.before.json').read_bytes()).hexdigest()
    before=frozen(s);assert repair.apply(s,repair.prepare(s,G,P2),backup_path='unused')['status']=='NOOP_ALREADY_RECONCILED'
    assert frozen(s)==before

def test_T14_exact_same_apply_command_is_an_idempotent_NOOP(actual):
    s=actual;raw=s.sidecar_path.read_bytes();sha=hashlib.sha256(raw).hexdigest();backup=s.sidecar_path.parent/'cli.before.json'
    args=[sys.executable,'-B',str(ROOT/'scripts/reconcile_false_model_reference.py'),'--apply','--source',str(s.path),'--sidecar',str(s.sidecar_path),'--invocation',G,'--plate-id',P2,'--expected-sha256',sha,'--expected-revision',str(s.review.review_revision),'--backup',str(backup)]
    first=subprocess.run(args,cwd=ROOT,capture_output=True);assert first.returncode==0,first.stderr.decode(errors='replace')
    before=(s.sidecar_path.read_bytes(),s.sidecar_path.stat().st_mtime_ns)
    second=subprocess.run(args,cwd=ROOT,capture_output=True);assert second.returncode==0,second.stderr.decode(errors='replace')
    assert b'NOOP_ALREADY_RECONCILED' in second.stdout and (s.sidecar_path.read_bytes(),s.sidecar_path.stat().st_mtime_ns)==before
    assert backup.read_bytes()==raw and backup.with_name(backup.name+'.manifest.json').exists()

def test_T15_T16_no_active_GTR2_but_history_is_preserved(actual):
    s=actual;sc,v=state(s);old=copy.deepcopy(sc);cleaned(s)
    reopened=review.MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);sc,v=state(reopened)
    assert set(v['plates'])==set(v['active_plate_ids'])=={P1}
    assert P2 not in partial.progress(sc)['pending_gtr_ids'] and tasks.for_scene(reopened,G)['count']==0
    assert sc['events'][:-1]==old['events'] and sc[stacks.HISTORY]['entries'][:-1]==old[stacks.HISTORY]['entries']
    assert life.archived_reference(v,P2)['plate']==old[stacks.KEY]['plates'][P2]

def test_T17_T18_complete_frames_count_without_new_inspection(actual):
    s=cleaned(actual);core.set_scene_review_state(s,G,checked=False);sc,v=state(s);r=mt.evaluate_unit(s,G)
    assert feedback.operator_inventory(sc,v)['counts']['gt']==1
    assert sc['checked'] is False and partial.progress(sc)['complete']
    assert r['status']=='MEASURED' and (r['tp'],r['fp'],r['fn'])==(1,1,0)

def test_T19_no_withdrawal_UI_or_restoration_task(actual):
    s=cleaned(actual);t=tasks.for_scene(s,G)
    assert not t['units'] and 'GTR 2' not in t['next_action']
    assert not (ROOT/'auto_annotation_tool/gui/mobile_canvas_reference_resolution.py').exists()
    for f in ('mobile_canvas_clean_view.py','mobile_canvas_editor.py','mobile_canvas_task_locator.py'):
        text=(ROOT/'auto_annotation_tool/gui'/f).read_text(encoding='utf-8')
        assert 'reference_resolution' not in text and 'Wycofaj błędnie' not in text

def test_T20_frames_without_image_check_produce_1_1_0(actual):
    s=cleaned(actual);core.set_scene_review_state(s,G,checked=False);r=mt.evaluate_unit(s,G)
    assert not state(s)[0]['checked'] and r['status']=='MEASURED' and (r['n_gt'],r['tp'],r['fp'],r['fn'])==(1,1,1,0)

def test_T21_unpersisted_Pose_seed_has_no_plate_id(tmp_path):
    s=source_session(tmp_path,mz=False);before=frozen(s)
    for _ in range(2):
        assert not stacks.snapshot(s,'mt-one')['plates'];core.scene(s,'mt-one');tasks.for_scene(s,'mt-one')
    assert frozen(s)==before

def test_T22_one_Space_creates_one_audited_dependent_reference(tmp_path):
    s=source_session(tmp_path,mz=False);r=s.review.review_revision
    partial.toggle_stack(s,'mt-one',oid='GT:a0');sc,v=state(s,'mt-one')
    assert s.review.review_revision==r+1 and len(v['plates'])==1
    p=next(iter(v['plates'].values()));assert p[life.ORIGIN]['kind']=='MODEL_DERIVED'
    assert partial.progress(sc)['object_decisions']['GT:a0']==partial.PLATE and not sc['checked']
    assert not partial.progress(sc)['pending_gtr_ids']

def test_T23_one_BAD_retires_only_automatic_reference_atomically(tmp_path):
    s=accepted(tmp_path);sc,v=state(s,'mt-one');pid=next(iter(v['plates']));r=s.review.review_revision
    stacks.toggle_unassociated(s,'mt-one','GT:a0',partial_review=True);sc,v=state(s,'mt-one')
    assert s.review.review_revision==r+1 and not v['plates'] and not partial.progress(sc)['pending_gtr_ids']
    assert v['links']['GT:a0']['status']=='UNASSOCIATED_NO_PLATE' and life.archived_reference(v,pid)
    assert tasks.for_scene(s,'mt-one')['count']==0 and mt.evaluate_unit(s,'mt-one')['partial_fp_lower_bound']==1

def test_T24_T35_manual_GTR_survives_unrelated_FINAL_rejection(tmp_path):
    from test_mobile_p2b2_canvas_review import MANUAL
    s=source_session(tmp_path,mz=False);core.add_plate(s,'mt-one',MANUAL,partial_review=True)
    before=copy.deepcopy(stacks.snapshot(s,'mt-one')['plates']);stacks.toggle_unassociated(s,'mt-one','GT:a0',partial_review=True)
    assert stacks.snapshot(s,'mt-one')['plates']==before

def test_T25_independent_GTZ_and_corners_survive_BAD(tmp_path):
    s=accepted(tmp_path);g='mt-one';stacks.edit_gt(s,g,'GT:a0',text='KEEPQA');v=stacks.snapshot(s,g);before=copy.deepcopy(v['plates'])
    stacks.toggle_unassociated(s,g,'GT:a0',partial_review=True);sc,v=state(s,g)
    assert v['plates']==before and set(v['active_plate_ids'])==set(before)
    assert next(iter(v['plates'].values()))['gt_text']=='KEEPQA'
    assert not any(life.RETIRED in x for x in v['links'].values())
    core.check_scene(s,g);r=mt.evaluate_unit(s,g)
    assert (r['n_gt'],r['tp'],r['fp'],r['fn'])==(1,0,1,1)

def test_T26_shared_physical_GTR_survives_one_rejected_FINAL(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,'mt-one',oid='GT:a0');v=stacks.snapshot(s,'mt-one');pid=next(iter(v['plates']))
    stacks.associate(s,'mt-one','GT:a1',pid);before=copy.deepcopy(stacks.snapshot(s,'mt-one')['plates'])
    stacks.toggle_unassociated(s,'mt-one','GT:a0',partial_review=True)
    v=stacks.snapshot(s,'mt-one')
    # The existing primary-source selector now uses the remaining correct MT.
    assert v['plates'][pid]['primary_object_id']=='GT:a1' and pid in v['active_plate_ids']
    assert {k:v for k,v in v['plates'][pid].items() if k!='primary_object_id'}=={k:v for k,v in before[pid].items() if k!='primary_object_id'}

def test_T27_RAW_suppressed_is_never_a_BAD_target(actual):
    before=frozen(actual)
    with pytest.raises(ValueError):stacks.toggle_unassociated(actual,G,RAW,partial_review=True)
    assert frozen(actual)==before

def test_T28_Space_after_BAD_restores_same_ID_without_extra_task(tmp_path):
    s=accepted(tmp_path);g='mt-one';pid=next(iter(stacks.snapshot(s,g)['plates']))
    stacks.toggle_unassociated(s,g,'GT:a0',partial_review=True);partial.toggle_stack(s,g,oid='GT:a0');sc,v=state(s,g)
    assert set(v['plates'])=={pid} and partial.progress(sc)['object_decisions']['GT:a0']==partial.PLATE
    assert not partial.progress(sc)['pending_gtr_ids']
    stacks.toggle_unassociated(s,g,'GT:a0',partial_review=True);assert not stacks.snapshot(s,g)['plates']

def test_T29_Undo_Redo_restores_source_decision_and_reference_together(tmp_path):
    s=accepted(tmp_path);g='mt-one';before=stacks.snapshot(s,g)
    stacks.toggle_unassociated(s,g,'GT:a0',partial_review=True);after=stacks.snapshot(s,g)
    stacks.restore_history(s,g);v=stacks.snapshot(s,g);assert v['plates']==before['plates'] and v['links']==before['links']
    stacks.restore_history(s,g,redo=True);v=stacks.snapshot(s,g);assert v['plates']==after['plates'] and v['links']==after['links']

def test_T32_R_removes_only_real_GTR_geometry(tmp_path):
    s=accepted(tmp_path);g='mt-one';stacks.edit_gt(s,g,'GT:a0',text='REALQA');pid=next(iter(stacks.snapshot(s,g)['plates']))
    stacks.remove_linked_geometry(s,g,'GT:a0');sc,v=state(s,g)
    assert v['plates'][pid]['gt_quad'] is None and v['plates'][pid]['gt_text']=='REALQA'
    assert partial.progress(sc)['object_decisions']['GT:a0']==partial.PLATE
    assert 'Przywróć' in tasks.for_scene(s,g)['next_action']

def test_T33_D_redraw_preserves_real_identity(tmp_path):
    s=accepted(tmp_path);g='mt-one';sc,v=state(s,g);pid=next(iter(v['plates']));p=v['plates'][pid];q=copy.deepcopy(p['gt_quad']);q[0][0]+=1
    target=dict(plate_id=pid,oid='GT:a0',gid=g,binding=sc['source_binding'],revision=s.review.review_revision,quad=copy.deepcopy(p['gt_quad']))
    stacks.redraw_reference(s,g,target,q);v=stacks.snapshot(s,g)
    assert set(v['plates'])=={pid} and v['plates'][pid]['gt_quad']==q

def test_T34_two_real_plates_keep_two_TP(tmp_path):
    s=accepted(tmp_path,n=2);core.check_scene(s,'mt-one');r=mt.evaluate_unit(s,'mt-one')
    assert r['status']=='MEASURED' and (r['n_gt'],r['tp'],r['fp'],r['fn'])==(2,2,0,0)

def test_T36_multiple_FINALs_on_one_GTR_remain_1to1(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.confirm_stack(s,'mt-one',oid='GT:a0');pid=next(iter(stacks.snapshot(s,'mt-one')['plates']))
    stacks.associate(s,'mt-one','GT:a1',pid);core.check_scene(s,'mt-one');r=mt.evaluate_unit(s,'mt-one')
    assert r['status']=='MEASURED' and (r['n_gt'],r['tp'],r['fp'],r['fn'])==(1,1,1,0)

def test_T37_legacy_without_dispositions_is_read_only(tmp_path):
    s=accepted(tmp_path);before=frozen(s);reopened=review.MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    core.scene(reopened,'mt-one');assert frozen(s)==before and not legacy.withdrawn_ids(core.scene(reopened,'mt-one'))

def test_T38_historical_withdrawal_reads_unchanged_without_new_workflow(actual):
    before=frozen(actual);sc,v=state(actual);legacy.validate_audit(sc,session_id=SID)
    assert legacy.withdrawn_ids(sc)=={P2} and mt.evaluate_unit(actual,G)['n_gt']==1 and frozen(actual)==before

def test_T39_matching_policy_and_OCR_BBOX_algorithms_unchanged():
    baseline=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    assert mt.POLICY_SHA256=='c5461594eae9a070d5af37d64a20ba23abd006c4a999a59e2f3d17e3fe4be3f0'
    for f in ('mobile_object_mz_quality.py','mobile_mt_detection_evaluation.py'):
        rel='auto_annotation_tool/ranking/'+f
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==baseline['sources'][rel]['sha256']

def test_T40_MZ_execution_prediction_timing_kept_but_FP_excluded(actual):
    from auto_annotation_tool.gui.mobile_mz_quality_context import snapshot
    s=actual;before={r['attempt_id']:r for r in snapshot(s)['rows']};cleaned(s);after={r['attempt_id']:r for r in snapshot(s)['rows']}
    assert after[BAD[3:]]['executed'] and after[BAD[3:]]['quality_state']=='EXECUTED_EXCLUDED_MT_FP'
    assert not after[BAD[3:]]['canonical_ocr_eligible']
    for aid in (GOOD[3:],BAD[3:]):
        for k in ('timing','raw_prediction'):assert after[aid][k]==before[aid][k]

def test_T41_T42_versioned_export_replay_preserves_historical_result(actual,tmp_path):
    from auto_annotation_tool.ranking import mobile_mt_object_export as ex
    s=actual;root=tmp_path/'exports';before=ex.export(s,root);files={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    cleaned(s);after=ex.export(s,root)
    assert before!=after
    assert all((Path(p).read_bytes(),Path(p).stat().st_mtime_ns)==value for p,value in files.items())
    for result in (before,after):
        assert result and (result[0].parent/'manifest.json').exists()
        ex.replay(result[0].parent)
    old_manifest=json.loads((before[0].parent/'manifest.json').read_text(encoding='utf-8'))
    new_manifest=json.loads((after[0].parent/'manifest.json').read_text(encoding='utf-8'))
    assert old_manifest['review_revision']+1==new_manifest['review_revision']
    # Replay from the exact canonical input, independent of the live sidecar.
    r=mt.evaluate_unit(s,G);replay=mt.score_unit_v2(r['evaluation_input'])
    assert replay==r and replay['matching_policy_sha256']==mt.POLICY_SHA256

def test_T48_model_acceptance_qualifies_only_registered_operator_GT(tmp_path):
    s=accepted(tmp_path);g='mt-one';sc,v=state(s,g);r=mt.evaluate_unit(s,g)
    assert not sc['checked'] and r['status']=='MEASURED' and r['tp']==1
    assert r['completion_eligibility']['gt_scope']=='operator_reference_scope'
    assert r['completion_eligibility']['independent_exhaustive_scene_gt'] is False

def test_T48_existing_inspection_survives_decision_only_toggle(tmp_path):
    s=accepted(tmp_path);g='mt-one';core.check_scene(s,g)
    partial.toggle_stack(s,g,oid='GT:a0');assert state(s,g)[0]['checked'] and not partial.progress(state(s,g)[0])['complete']
    partial.toggle_stack(s,g,oid='GT:a0');assert state(s,g)[0]['checked'] and mt.evaluate_unit(s,g)['tp']==1

def test_T20_T39_full_image_inspection_and_MT_remain_independent_of_GTZ(tmp_path):
    s=source_session(tmp_path,mz=True);g='mt-one';partial.confirm_stack(s,g,oid='GT:a0')
    core.set_scene_review_state(s,g,checked=True)
    assert mt.evaluate_unit(s,g)['tp']==1 and state(s,g)[0]['checked']
    assert mt.evaluate_session(s)['ocr_summary']['n_reads']==0
    stacks.edit_gt(s,g,'GT:a0',text='QUALITYQA')
    assert mt.evaluate_unit(s,g)['tp']==1 and state(s,g)[0]['checked']
