"""A-C: bounded source audit, strict negative guards, export/protection. No show."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import asdict
import ast,json,os,zipfile
import pytest
from auto_annotation_tool.ranking import final180_mobile as mobile,final180_reference as ref,photo_reference as photo
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession


def save(out,name,data):(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf-8')


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    value=os.environ.get('F6_FIRST_MT_FIX_EVIDENCE')
    if not value:pytest.skip('Private immutable real-session evidence required')
    out=Path(value);paths=photo.read_json(out/'paths.json');old=photo.read_json(paths['old_report'])
    session=MobileReviewSession.open(Path(paths['archive']),sidecar_path=tmp_path_factory.mktemp('readonly')/'absent.json')
    assert session.bundle.validation.ok and session.bundle.source_archive_sha256==paths['archive_sha256']
    target=mobile.analyze_import(paths['archive'],paths['run'],output=out/'analysis_results')
    report=photo.read_json(target/'evaluation.json');manifest=photo.read_json(Path(paths['prepared'])/'manifest.json')
    save(out,'delivery.json',dict(result=str(target),evaluation_sha256=photo.digest(target/'evaluation.json'),
        result_manifest_sha256=photo.digest(target/'manifest.json'),source_archive_sha256=paths['archive_sha256']))
    yield NS(out=out,paths=paths,old=old,session=session,report=report,manifest=manifest,target=target)
    session.verify_source(force=True);assert not session.sidecar_path.exists()


def test_A_real_session_four_first_mt_fifth_na_and_no_gt(source):
    r=source.report;rows=r['rows'];pre=r['readability_preflight']['first_mt']
    assert r['first_mt_scope']['status']=='BOUNDED' and len(r['first_mt_scope']['mt_invocation_ids'])==11
    assert [x['evidence'] is not None for x in rows]==[True,True,True,True,False]
    assert [x['status'] for x in pre]==['PASS_SOURCE_ORDER']*4+['N/A']
    assert r['summary']['selected_mt_exposures']==r['readability_preflight']['summary']['first_mt_order_proven']==4
    assert [x['evidence']['mt_invocation_ids'][0].rsplit('-',1)[-1] for x in rows[:4]]==['a00000003','a00000005','a00000007','a00000008']
    assert all(x['first_mt_risk']['reason']=='UNKNOWN_MT_STRICTLY_LATER' for x in rows[:4])
    assert rows[4]['first_mt_risk']['reason']=='UNKNOWN_MT_MAY_BE_EARLIER'
    assert rows[4]['evidence_reason']=='EARLIER_MT_INPUT_MAY_BE_UNLOCATED'
    assert rows[4]['first_mt_risk']['blocking_mt_ids']==['exp-1794662253625-4b7c0c8b-a00000009']
    assert all(x['status']=='N/A' and x['reason']=='NOT_AVAILABLE_NO_GT' and x['tp'] is None for x in rows)
    assert r['metrics_status']=='NOT_AVAILABLE_NO_GT' and r['summary']['system_full_read']['rate'] is None
    unknown=next(f for f in r['frames'] if f['source_stamp']['source_sequence']==86927)
    assert unknown['execution_status']=='CANCELLED' and unknown['marker_decoding']['decoded_tokens']==['','R005TN5JOX']
    assert unknown['exposure_id'] is None and unknown['marker_decoding']['pair_verified'] is False
    assert r['policy']==source.old['policy']==ref.POLICY
    save(source.out,'A.json',dict(status='PASS',qualified=4,total=5,fifth='N/A_EARLIER_UNIDENTIFIED_STARTED_MT',
        metrics='NOT_AVAILABLE_NO_GT',unknown_phase_inferred=False,cancelled_start_retained=True))


def test_B_earlier_unknown_domains_stamp_order_loss_and_candidate_locality(source):
    original=source.report;faults=['source_earlier','start_earlier','both_earlier','sensor_domain','android_domain','event_domain',
        'capture_domain','missing_stamp','candidate_stamp','missing_start','inconsistent_start','different_session','different_archive',
        'camera_generation','tied_capture','tied_start','unbounded_loss','missing_frame','missing_inventory']
    results=[]
    for fault in faults:
        frames=deepcopy(original['frames']);scope=deepcopy(original['first_mt_scope'])
        candidate,_=mobile.select_first('e0001',frames);u=next(f for f in frames if f['source_stamp']['source_sequence']==86927)
        start=u['mt_lifecycle'][0]['events']['MT_STARTED']
        if fault in ('source_earlier','both_earlier'):
            u['captured_elapsed_nanos']=candidate['captured_elapsed_nanos']-1
            for key in ('source_sequence','source_timestamp_nanos'):u['source_stamp'][key]=candidate['source_stamp'][key]-1
            u['mt_lifecycle'][0]['source_stamp']=deepcopy(u['source_stamp'])
        if fault in ('start_earlier','both_earlier'):
            u['mt_started_elapsed_nanos']=start['elapsed_nanos']=candidate['mt_started_elapsed_nanos']-1
            u['mt_start_event_sequence']=start['event_sequence']=candidate['mt_start_event_sequence']-1
        if fault=='sensor_domain':u['source_stamp']['source_timestamp_domain']='UNKNOWN';u['mt_lifecycle'][0]['source_stamp']=deepcopy(u['source_stamp'])
        elif fault=='android_domain':u['android_clock_domain']='PC_PERF_COUNTER'
        elif fault=='event_domain':start['time_domain']='UNKNOWN'
        elif fault=='capture_domain':u['capture_clock']='UNKNOWN'
        elif fault=='missing_stamp':u['source_stamp']={}
        elif fault=='candidate_stamp':candidate['source_stamp']={}
        elif fault=='missing_start':u['mt_lifecycle'][0]['events'].pop('MT_STARTED')
        elif fault=='inconsistent_start':u['mt_started_elapsed_nanos']+=1
        elif fault=='different_session':u['session_id']='FOREIGN'
        elif fault=='different_archive':u['source_archive_sha256']='f'*64
        elif fault=='camera_generation':u['source_stamp']['camera_transform_generation']+=1;u['mt_lifecycle'][0]['source_stamp']=deepcopy(u['source_stamp'])
        elif fault=='tied_capture':u['captured_elapsed_nanos']=candidate['captured_elapsed_nanos']
        elif fault=='tied_start':u['mt_started_elapsed_nanos']=start['elapsed_nanos']=candidate['mt_started_elapsed_nanos']
        elif fault=='unbounded_loss':
            f6=deepcopy(source.session.f6_evidence);f6['unlocated_loss_possible']=True
            scope=mobile.first_mt_scope(frames,f6,source.session.bundle.collection_session)
        elif fault=='missing_frame':
            frames.remove(u);scope=mobile.first_mt_scope(frames,source.session.f6_evidence,source.session.bundle.collection_session)
        elif fault=='missing_inventory':scope=None
        risk=mobile.first_mt_risk(candidate,frames,scope)
        assert risk['status']=='N/A',(fault,risk)
        results.append(dict(fault=fault,reason=risk['reason']))
    for fault in ('sequence_gaps','malformed_records','binding_gaps'):
        f6=deepcopy(source.session.f6_evidence);f6[fault]=['TEST_ONLY_MISSING_EVIDENCE']
        scope=mobile.first_mt_scope(original['frames'],f6,source.session.bundle.collection_session)
        candidate,_=mobile.select_first('e0001',original['frames'])
        assert mobile.first_mt_risk(candidate,original['frames'],scope)['status']=='N/A'
        results.append(dict(fault=fault,reason='UNBOUNDED_MT_SOURCE_LOSS'))
    # A faulty candidate does not invalidate an unrelated earlier candidate.
    frames=deepcopy(original['frames']);first,_=mobile.select_first('e0001',frames);second,_=mobile.select_first('e0002',frames)
    second['source_stamp']={}
    assert mobile.first_mt_risk(first,frames,original['first_mt_scope'])['status']=='PASS'
    assert mobile.first_mt_risk(second,frames,original['first_mt_scope'])['status']=='N/A'
    save(source.out,'B.json',dict(status='PASS',negative_cases=results,candidate_failure_is_local=True))


def test_C_consistent_native_export_unchanged_sources_and_frozen_selection(source):
    r=source.report
    for row,pre in zip(r['rows'],r['readability_preflight']['first_mt']):
        assert row['exposure_id']==pre['exposure_id'] and row['first_mt_risk']==pre['first_mt_risk']
        assert bool(row['evidence'])==(pre['status']=='PASS_SOURCE_ORDER')
    for key in ('frames','raw_mt','raw_mz','presentation_timeline','policy','geometry_policy','recording_completeness'):
        assert r[key]==source.old[key],key
    assert r['readability_preflight']['frames']==source.old['readability_preflight']['frames']
    assert r['summary']['unlocated_mt_calls']==1 and r['summary']['raw_mt_calls']==11 and r['summary']['raw_mz_calls']==2
    with zipfile.ZipFile(source.paths['archive']) as z:assert z.testzip() is None
    v=source.session.bundle.validation;assert v.checked_hashes==86 and v.skipped_hashes==0
    before_ast=ast.parse((source.out/'before/final180_mobile.py').read_text('utf-8'));after_ast=ast.parse(Path(mobile.__file__).read_text('utf-8'))
    funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(before_ast),funcs(after_ast)
    for name in ('_order','select_first','score_frame','_native_frames','configuration','reference_report'):assert a[name]==b[name],name
    protected=photo.read_json(source.out/'protection_before.json');changed=[]
    for name,old in protected.items():
        if name==source.paths['code']:continue
        p=Path(name);stat=p.stat()
        if stat.st_size!=old['bytes'] or stat.st_mtime_ns!=old['mtime_ns'] or photo.digest(p)!=old['sha256']:changed.append(name)
    assert not changed,changed
    m,_,_=ref.load(source.paths['final']);pf=ref.reference_preflight(source.paths['final'])
    assert len(m['exposures'])==176 and pf['complete_gt_plates']==209
    assert 'Nieznane MT istnieje, ale' in (source.target/'report.md').read_text('utf-8')
    assert 'Nieznane MT może być wcześniejsze' in (source.target/'report.md').read_text('utf-8')
    save(source.out,'C.json',dict(status='PASS',qualified=4,preflight_qualified=4,CRC='PASS',validation=asdict(v),
        protected_files_unchanged=len(protected)-1,changed_protected_files=changed,final_photographs=176,gt_plates=209,
        source_frames_and_raw_calls_unchanged=True,frozen_selection_and_scoring_unchanged=True))
