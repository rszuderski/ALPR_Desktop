"""TEST_ONLY controlled PC clock and producer contract acceptance, no device run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

import pytest
from test_final180_desktop import frozen
from auto_annotation_tool.gui import final180_tv as tv
from auto_annotation_tool.ranking import final180_reference as ref, photo_reference as photo
from auto_annotation_tool.ranking import final180_timeline as timeline, final180_mobile as mobile
from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle

from _private_test_data import private_test_path
FIXTURES=private_test_path('output/F6_DESKTOP_TIMELINE_20261112_001/producer_snapshot/fixtures')
FINAL_FIXTURES=FIXTURES.parents[1]/'producer_final_snapshot'


class Clock:
    def __init__(self):self.ns=1000;self.wall='2026-11-12T10:00:00+00:00'
    def __call__(self):return self.ns
    def advance(self,ns):self.ns+=ns
    def utc(self):return self.wall


def setup_run(tmp_path,**kwargs):
    d=frozen(tmp_path,**kwargs);m,_,_=ref.load(d);clock=Clock()
    return d,m,clock,tv.PresentationRun(d,m,clock=clock,utc=clock.utc)


def projected(run):return timeline.project(run.manifest,run.events)


def test_software_interval_io_delay_render_delay_and_utc_jump(tmp_path,monkeypatch):
    d,m,c,r=setup_run(tmp_path)
    real=tv.os.fsync
    def slow(fd):c.advance(2_000_000_000);real(fd)
    monkeypatch.setattr(tv.os,'fsync',slow)
    r.plan_origin_ns=c();c.advance(5_000_000_000)
    r.present(0,{},lambda:c.advance(500_000_000))
    start=projected(r)[0]['start_pc_ns'];assert start is not None
    assert r.deadline==start+30_000_000_000
    c.wall='2026-11-11T03:00:00+00:00'
    c.ns=r.deadline-1;assert not r.due()
    c.advance(1);r.end_photo()
    assert projected(r)[0]['end_pc_ns'] is None and r.current==0
    r.replace(lambda:c.advance(700_000_000))
    row=projected(r)[0]
    assert row['duration_pc_ns']>=30_000_000_000 and row['status']=='COMPLETED'
    assert row['draw_duration_ns']==500_000_000
    assert row['request_pc_ns']<row['start_pc_ns']
    r.stop()
    timeline.read_run(r.directory/'run.json',m,photo.digest(d/'manifest.json'))


def test_stop_partial_and_not_shown(tmp_path):
    _,m,c,r=setup_run(tmp_path);r.present(0,{},lambda:None);c.advance(15_000_000_000)
    r.stop(clear=lambda:None)
    rows=projected(r)
    assert rows[0]['status']=='PARTIAL' and rows[0]['duration_pc_ns']==15_000_000_000
    assert all(x['status']=='NOT_SHOWN' and x['duration_pc_ns'] is None for x in rows[1:])


def test_missing_end_and_restart_clock_never_subtract_domains(tmp_path):
    d,m,c,r=setup_run(tmp_path);r.present(0,{},lambda:None);r.stop()
    old=(r.directory/'events.jsonl').read_bytes();c.ns=5
    resumed=tv.PresentationRun(d,m,resume=r.directory,clock=c)
    assert resumed.next_index==1 and resumed.clock_id!=r.clock_id
    resumed.present(1,{},lambda:None);c.advance(2_000_000_000);resumed.stop(clear=lambda:None)
    assert (r.directory/'events.jsonl').read_bytes()==old
    rows=projected(resumed);assert rows[0]['status']=='UNKNOWN' and rows[0]['end_pc_ns'] is None
    assert rows[1]['duration_pc_ns']==2_000_000_000


def test_recovery_retains_broken_tail_and_consumes_intent(tmp_path):
    d,m,c,r=setup_run(tmp_path)
    r.append('PHOTO_DRAW_INTENT',position=1,status='INTENT');r.journal.close()
    with r.journal_path.open('ab') as stream:stream.write(b'{"event":"PHOTO_')
    damaged=r.journal_path.read_bytes()
    with pytest.raises(ValueError,match='UNCHECKPOINTED'):timeline.read_run(r.directory/'run.json',m,r.manifest_sha)
    recovered=tv.PresentationRun(d,m,resume=r.directory,clock=c)
    assert recovered.next_index==1 and r.journal_path.read_bytes()==damaged
    assert recovered.state['recoveries'][0]['reason']=='MALFORMED_TAIL_RETAINED'
    recovered.stop()
    _,events,_=timeline.read_run(r.directory/'run.json',m,r.manifest_sha)
    row=timeline.project(m,events)[0]
    assert row['status']=='UNKNOWN' and row['duration_pc_ns'] is None


@pytest.mark.parametrize('fault',['fsync','atomic'])
def test_write_failure_is_visible_stops_and_never_completes(tmp_path,monkeypatch,fault):
    d,m,c,r=setup_run(tmp_path)
    def broken(*args,**kwargs):raise OSError('TEST_ONLY disk full')
    if fault=='fsync':monkeypatch.setattr(tv.os,'fsync',broken)
    else:monkeypatch.setattr(tv,'_atomic_write_json',broken)
    with pytest.raises(OSError,match='LOG_ERROR'):r.present(0,{},lambda:None)
    assert r.status=='LOG_ERROR' and r.journal.closed
    assert photo.read_json(r.directory/'run.json')['status']!='COMPLETED'
    assert all(x.get('event')!='RUN_COMPLETED' for x in r.events)


def test_failed_render_cannot_claim_view_update(tmp_path):
    _,_,c,r=setup_run(tmp_path)
    def fail():raise RuntimeError('TEST_ONLY renderer')
    with pytest.raises(RuntimeError):r.present(0,{},fail)
    r.stop();row=projected(r)[0]
    assert row['status']=='UNKNOWN' and row['start_pc_ns'] is None and r.next_index==1


def test_prepared_binding_rejects_neighbor_before_drawing(tmp_path):
    d,m,c,r=setup_run(tmp_path);_,mat,_=ref.load(d)
    cache=tv.PreparedFrames(mat,m,r.manifest_sha,r.state['run_id'])
    try:
        cache.request(1);image,placement=cache.get(1)
        with pytest.raises(ValueError,match='IDENTITY_CONFLICT'):r.present(0,placement,lambda:pytest.fail('must not draw'),require_binding=True)
        assert r.next_index==0
    finally:cache.close();r.stop()


@pytest.mark.parametrize('change',['run_id','photo_sha256','position','clock'])
def test_resealed_journal_still_requires_event_identity(tmp_path,change):
    d,m,c,r=setup_run(tmp_path);r.present(0,{},lambda:None);r.stop()
    events=deepcopy(r.events);event=next(e for e in events if e['event']=='PHOTO_REQUESTED')
    if change=='clock':event['pc_perf_counter_ns']=-1
    else:event[change]={'position':2,'run_id':'foreign','photo_sha256':'0'*64}[change]
    raw=('\n'.join(json.dumps(x) for x in events)+'\n').encode();r.journal_path.write_bytes(raw)
    state=photo.read_json(r.directory/'run.json');state['journals'][0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (r.directory/'run.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):timeline.read_run(r.directory/'run.json',m,r.manifest_sha)


def test_176_plan_and_duplicate_content_are_not_collapsed(tmp_path):
    d,m,c,r=setup_run(tmp_path,n=166,negative=10,duplicate=True)
    r.plan_origin_ns=c();c.advance(5_000_000_000)
    for i in range(176):
        r.present(i,{},lambda:None);c.advance(30_000_000_000);r.end_photo()
        if i==175:r.complete(lambda:None)
        else:r.replace(lambda:None);c.advance(500_000_000)
    rows=projected(r)
    assert len(rows)==176 and all(x['status']=='COMPLETED' for x in rows)
    assert sum(x['duration_pc_ns'] for x in rows)==5280_000_000_000
    assert rows[-1]['planned_start_s']+30==5372.5
    assert len([e for e in r.events if e['event']=='REPLACEMENT_VIEW_UPDATED' and e['phase']=='T'])==175
    assert rows[0]['photo_sha256']==rows[1]['photo_sha256'] and rows[0]['exposure_id']!=rows[1]['exposure_id']


@pytest.mark.parametrize('name',sorted(p.stem for p in FIXTURES.glob('*.alprsession')))
def test_actual_producer_fixtures_native_reader(name):
    p=FIXTURES/(name+'.alprsession');before=photo.digest(p);b=read_mobile_report_bundle(p)
    assert b.validation.ok,b.validation.errors
    assert b.f6_evidence['available'] and photo.digest(p)==before
    if name.startswith('interrupted') or name in ('queue-full','r0-cancelled','r0-delayed-old-frame','r0-mt-error','r0-mz-error'):
        assert b.f6_evidence['status']=='PARTIAL'
    if name=='interrupted-start':
        assert len(b.f6_evidence['mt_invocations'])==1 and not b.scene_evidence['roles']
    if name=='long-session-4100-frames':
        assert len(b.tv_bbox_evidence['rows'])==4100
        with zipfile.ZipFile(p) as z:assert z.getinfo('samples/f6_execution.jsonl').file_size>32*1024*1024
    if name=='r0-positive':
        call=next(iter(b.f6_evidence['mt_invocations'].values()))
        assert call['source_stamp']['source_timestamp_nanos']==9007199254740994
        assert call['inputs'][0]['roi_camera_xy']==[128,88,525,288]
        assert call['inputs'][0]['status']=='WRITTEN' and call['inputs'][0]['input_width']==64
    if name=='r0-no-detection':
        call=next(iter(b.f6_evidence['mt_invocations'].values()))
        assert call['final_population_complete'] and call['events']['MT_DECODED']['details']['raw_detection_count']==0


@pytest.mark.parametrize('path',sorted(FINAL_FIXTURES.rglob('*.alprsession')))
def test_completed_producer_delivery(path):
    b=read_mobile_report_bundle(path)
    if path.stem=='long-session-4100-frames':
        # Completed producer delivery has 4100 truncated MP observations. Do not
        # fabricate their absent trace IDs/ROI fields just to accept the fixture.
        assert not b.validation.ok and any('MP_OBSERVATION_FIELDS_MISSING' in e for e in b.validation.errors)
        assert b.f6_evidence['events_persisted']==12300
        from auto_annotation_tool.ranking.mobile_f6_execution_evidence import checked_rows
        with zipfile.ZipFile(path) as z:
            assert z.getinfo('samples/scene_evidence.jsonl').file_size>32*1024*1024
            scenes=list(checked_rows(z,'samples/scene_evidence.jsonl',json.loads(z.read('manifest.json'))))
        assert len(scenes[0]['mp_observations'])==4100
        return
    assert b.validation.ok,b.validation.errors
    assert b.f6_evidence['available']
    if 'fatal' in path.stem:assert b.f6_evidence['status']=='PARTIAL'


def mutate_fixture(target,modify):
    with zipfile.ZipFile(FIXTURES/'r0-positive.alprsession') as z:payload={n:z.read(n) for n in z.namelist()}
    modify(payload)
    manifest=json.loads(payload['manifest.json'])
    for name in manifest['entry_sha256']:manifest['entry_sha256'][name]=hashlib.sha256(payload[name]).hexdigest()
    payload['manifest.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():z.writestr(name,raw)
    return target


@pytest.mark.parametrize('change',['missing_start','foreign_stamp','false_zero','wrong_input','bad_roi','false_complete'])
@pytest.mark.skipif(not (FIXTURES/'r0-positive.alprsession').is_file(), reason='Producer fixture not supplied; set ALPR_TEST_DATA_DIR.')
def test_producer_cross_evidence_conflicts_are_rejected(tmp_path,change):
    def modify(data):
        rows=[json.loads(x) for x in data['samples/f6_execution.jsonl'].splitlines() if x.strip()]
        start=next(r for r in rows if r['event']=='MT_STARTED')
        if change=='missing_start':start['event']='SYNTHETIC_LOAD'
        elif change=='foreign_stamp':start['source_stamp']['source_sequence']+=1
        elif change=='false_zero':next(r for r in rows if r['event']=='MT_DECODED')['details']['raw_detection_count']=0
        elif change=='false_complete':rows[-1]['event_sequence']+=1
        else:
            attempts=[json.loads(x) for x in data['samples/attempts.jsonl'].splitlines() if x.strip()]
            if change=='bad_roi':attempts[0]['roi_left']+=1
            else:attempts[0]['mt_input_evidence_entry']=attempts[0]['source_frame_entry']
            data['samples/attempts.jsonl']=('\n'.join(json.dumps(a) for a in attempts)+'\n').encode()
        data['samples/f6_execution.jsonl']=('\n'.join(json.dumps(x) for x in rows)+'\n').encode()
    b=read_mobile_report_bundle(mutate_fixture(tmp_path/'bad.alprsession',modify))
    assert not b.validation.ok


def test_source_and_actual_start_order_conflict_is_not_best_read():
    first=dict(frame_key='old',exposure_id='e',identity_status='PASS',mt_invocation_ids=['a'],
        captured_elapsed_nanos=10,source_stamp=dict(source_sequence=1),f6_lifecycle_required=True,mt_started_elapsed_nanos=100)
    later=dict(first,frame_key='new',captured_elapsed_nanos=20,mt_started_elapsed_nanos=50)
    assert mobile.select_first('e',[later,first])==(None,'FIRST_SOURCE_ORDER_VS_MT_START_CONFLICT')
    later['mt_started_elapsed_nanos']=200;first['execution_status']='ERROR'
    assert mobile.select_first('e',[later,first])[0] is first


def test_attempt_flag_without_actual_backend_start_never_proves_first():
    frame=dict(frame_key='legacy',exposure_id='e',identity_status='PASS',mt_invocation_ids=['declared'],
        captured_elapsed_nanos=1,source_stamp=dict(source_sequence=1),f6_lifecycle_required=True)
    assert mobile.select_first('e',[frame])==(None,'ACTUAL_MT_START_UNPROVEN')


def test_subset_is_not_complete_system_rate():
    plate=dict(text_state='FULL_TEXT')
    row=dict(status='MEASURED',tp=1,fp=0,fn=0,readable_gt=1,exact_reads=1,unknown_reads=0,conditional_mz_n=1,
        conditional_mz_exact=1,wrong_reads=0,missing_reads=0,plates=[plate],evidence={},reason='',negative_false_positive=None,
        negative_text_only=None,negative_correct_silence=None,system_failure=False)
    other=dict(row,status='N/A',reason='NOT_SHOWN',readable_gt=0,evidence=None)
    s=mobile.summarize([row,other],[],dict(exposures=[dict(source_role='POSITIVE')]*2))
    assert s['system_full_read']['rate'] is None and s['system_full_read']['denominator']==2
    assert s['evaluated_subset_full_read']['rate']==1


@pytest.mark.skipif(not (FIXTURES/'r0-positive.alprsession').is_file(), reason='Producer fixture not supplied; set ALPR_TEST_DATA_DIR.')
def test_native_f6_adapter_separates_tv_context_roi_and_mz_identity(tmp_path,monkeypatch):
    """Geometry stub is explicit: producer patches have no actual FINAL codes."""
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.ranking import final180_markers as markers
    d,m,c,r=setup_run(tmp_path);r.present(0,{},lambda:None);r.stop(clear=lambda:None)
    session=MobileReviewSession.open(FIXTURES/'r0-positive.alprsession',sidecar_path=tmp_path/'absent.review.json')
    l,t,right,bottom=m['layout']['photo_rect'];sx=397/(right-l);sy=200/(bottom-t)
    decoded=dict(status='PASS',exposure_id='e0001',matrix_tv_to_camera=[[sx,0,128-sx*l],[0,sy,88-sy*t],[0,0,1]],
        test_only='GEOMETRY_STUB_NOT_PRODUCER_MARKER_ACCEPTANCE')
    monkeypatch.setattr(markers,'decode',lambda *args,**kwargs:deepcopy(decoded))
    frames,gaps=mobile._native_frames(session,m,r.manifest_sha,r.state['run_id'])
    f=frames[0];assert not gaps and f['input_region']==[128,88,525,288]
    assert f['geometry_status']=='PASS',f['geometry_reason']
    assert markers.input_exclusion(decoded,m['layout'],[80,60,560,340])['status']=='N/A'
    assert f['execution_status']=='COMPLETED' and f['final_population_complete']
    assert f['finals'][0]['bbox']==pytest.approx([160,150,280,190],abs=.001)
    mz=f['finals'][0]['mz'];assert mz['lifecycle']['status']=='RETURNED'
    assert mz['source_bound'] and mz['executed'] and mz['raw_prediction']=='A123'
    # Repeated offline import uses the same journal and sources; clocks are not aligned.
    a=mobile.analyze_archive(d,session.path,r.directory/'run.json')
    b=mobile.analyze_archive(d,session.path,r.directory/'run.json')
    assert a==b and a['rows'][0]['evidence']['frame_key']==f['frame_key']
    assert a['rows'][0]['presentation']['status']=='PARTIAL'
    assert all(x['reason']=='NOT_SHOWN' for x in a['rows'][1:])
    assert a['acceptance']['producer_contract']=='NATIVE_F6_FORMAT_VALIDATED'
    assert a['go']=='NO_GO'


def test_desktop_entry_uses_same_arguments_and_toplevel(tmp_path,monkeypatch):
    # Each native Tk smoke test owns its Tcl interpreter in a fresh process.
    # Repeated Tk roots in the Windows aggregate runner have intermittent Tcl
    # library initialization failures; this must not become an unconditional skip.
    import os,subprocess,sys
    if os.environ.get('F6_ISOLATED_TK')!='1':
        process=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_desktop_entry_uses_same_arguments_and_toplevel','-q'],
            env=dict(os.environ,F6_ISOLATED_TK='1'),capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stdout+process.stderr
        return
    import tkinter as tk
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    d=frozen(tmp_path);root=tk.Tk();root.withdraw()
    args=NS(xml=None,positives=None,negatives=None,output=tmp_path,dwell=30,negatives_declared=True,unreadable_ids=[])
    app=tv.make_app(tv.legacy_module(tv.LEGACY_DEFAULT),args,parent=root);app.window.withdraw()
    try:
        app.loaded(d);assert isinstance(app.window,tk.Toplevel) and app.viewer is None
        assert '91' in app.plan.get() or '96' in app.plan.get()  # 3*30 + 5 + 2*.5 = 96
        called=[];monkeypatch.setattr(tv,'main',lambda argv,**kw:called.append((argv,kw)))
        tv.open_from_desktop(root)
        assert called[0]==(tv.accepted_arguments(),dict(parent=root))
        assert callable(MobileReportBrowser.open_final_presentation)
    finally:app._close();root.destroy()


@pytest.mark.skipif(not (FIXTURES/'interrupted-start.alprsession').is_file(), reason='Producer fixture not supplied; set ALPR_TEST_DATA_DIR.')
def test_unlocated_lifecycle_loss_prevents_later_success_selection(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    d,m,c,r=setup_run(tmp_path);r.present(0,{},lambda:None);r.stop(clear=lambda:None)
    session=MobileReviewSession.open(FIXTURES/'interrupted-start.alprsession',sidecar_path=tmp_path/'absent.review.json')
    assert session.f6_evidence['unlocated_loss_possible']
    monkeypatch.setattr(MobileReviewSession,'open',classmethod(lambda cls,*args,**kwargs:session))
    monkeypatch.setattr(mobile,'_native_frames',lambda *args,**kwargs:([dict(frame_key='later',exposure_id='e0001',
        identity_status='PASS',mt_invocation_ids=['later-call'],captured_elapsed_nanos=900,source_stamp=dict(source_sequence=90))],[]))
    report=mobile.analyze_archive(d,session.path,r.directory/'run.json')
    assert report['rows'][0]['reason']=='EARLIER_MT_INPUT_MAY_BE_UNLOCATED'
    assert report['rows'][0]['evidence'] is None and report['summary']['system_full_read']['rate'] is None
