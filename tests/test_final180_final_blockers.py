"""Exactly three focused Desktop checks. All writes are private; no TV show."""
from pathlib import Path
from types import SimpleNamespace as NS
from copy import deepcopy
from collections import Counter
import ast, hashlib, io, json, os, shutil, time, zipfile
import cv2
import pytest
from PIL import Image
from auto_annotation_tool.gui import final180_tv as tv
from auto_annotation_tool.ranking import final180_mobile as mobile, final180_markers as mk, final180_reference as ref, final180_timeline as tl, photo_reference as photo


@pytest.fixture(scope='module')
def evidence():
    value=os.environ.get('F6_BLOCKERS_EVIDENCE')
    if not value:pytest.skip('Private real-session evidence not supplied')
    out=Path(value);return NS(out=out,paths=photo.read_json(out/'paths.json'))


def save(out,name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')


def plan(tmp_path,n=1,version=mk.F2_SAFE_LAYOUT):
    source=tmp_path/'source';source.mkdir(parents=True)
    for i in range(n):Image.new('RGB',(100,80),(i+25,60,80)).save(source/f'{i}.png')
    prepared=ref.freeze_photographs(source,tmp_path/'plans',screen=(1920,1080),technical=True,layout_version=version)
    return prepared,photo.read_json(prepared/'manifest.json')


def finished(prepared,m):
    clock=NS(value=1000);r=tv.PresentationRun(prepared,m,clock=lambda:clock.value)
    for i in range(len(m['exposures'])):
        r.present(i,{},lambda:None);clock.value=r.deadline;r.end_photo()
        if i+1<len(m['exposures']):r.replace(lambda:None)
    token=mk.payload(m,r.manifest_sha,r.state['run_id'],'POSTROLL','T') if m['layout'].get('layout_version')==mk.F2_SAFE_LAYOUT else None
    r.complete(lambda:None,display_token=token);return r


def picture(m,r):return mk.render(None,m['layout'],mk.payload(m,r.manifest_sha,r.state['run_id'],'e0001','P'))[0]


def archive_image(tmp_path,image):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from test_final180_timeline import FIXTURES,mutate_fixture
    tmp_path.mkdir(parents=True,exist_ok=True)
    session=MobileReviewSession.open(FIXTURES/'r0-positive.alprsession',sidecar_path=tmp_path/'absent.json')
    record=session.tv_bbox_evidence['rows'][0]['source_image'];old=record['sha256'];buf=io.BytesIO()
    image.resize((record['width_px'],record['height_px']),Image.Resampling.LANCZOS).save(buf,format='JPEG',quality=96)
    raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest()
    def replace(data):
        for k in data:
            if hashlib.sha256(data[k]).hexdigest()==old:data[k]=raw
            elif k.endswith(('.json','.jsonl')):data[k]=data[k].replace(old.encode(),sha.encode())
    return mutate_fixture(tmp_path/'source.alprsession',replace)


def app_fixture(tmp_path,monkeypatch):
    import tkinter as tk
    from auto_annotation_tool import config
    from auto_annotation_tool.session import SessionManager
    monkeypatch.setattr(config,'SESSION',SessionManager(tmp_path/'preferences.json'))
    monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda:[dict(name='HOST_ONLY',x=0,y=0,w=1920,h=1080,verified=True,geometry_source='ISOLATED_TEST')])
    root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *a:errors.append(str(a))
    app=tv.make_app(tv.legacy_module(tv.LEGACY_DEFAULT),NS(xml=None,positives=None,negatives=None,output=tmp_path,dwell=5,negatives_declared=False,unreadable_ids=[]),parent=root)
    app.window.withdraw()
    def hidden(screen):
        app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
        app.viewer.bind('<Escape>',lambda _:app.abort());app.viewer.protocol('WM_DELETE_WINDOW',app.abort)
    monkeypatch.setattr(app,'_make_viewer',hidden)
    return root,app,errors


def close_app(root,app):
    for token in root.tk.call('after','info'):
        if root.tk.call('after','info',token)[0] in (app.window._tclCommands or []):app.window.after_cancel(token)
    app._close();root.destroy()


def test_D1_completed_postroll_close_and_failed_journal(evidence,tmp_path,monkeypatch):
    prepared,m=plan(tmp_path,5);clock=NS(value=1_000_000_000);now=lambda:clock.value
    monkeypatch.setattr(tv,'time',NS(perf_counter_ns=now,monotonic=time.monotonic))
    controller=tv.PresentationRun;monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:controller(*a,clock=now,**kw))
    root,app,errors=app_fixture(tmp_path,monkeypatch);drawn=[];draw=app.draw
    monkeypatch.setattr(app,'draw',lambda image:(drawn.append(image.copy()),draw(image)))
    def advance():
        clock.value=app.count_end
        for i in range(5):
            deadline=time.monotonic()+10
            while app.mode!='RUNNING' and time.monotonic()<deadline:
                app._tick();root.update();time.sleep(.01)
            assert app.mode=='RUNNING',app.status.get()
            clock.value=app.run.deadline;app._tick()
            if i<4:clock.value=app.transition_end
    try:
        app.loaded(prepared);app.begin_countdown();advance()
        assert app.mode=='COMPLETED' and app._post_roll_hold and '5/5' in app.status.get()
        assert app.stop_btn.cget('text')=='Zamknij planszę T'
        state,events,_=tl.read_run(app.run.directory/'run.json',m,app.run.manifest_sha)
        assert state['status']=='COMPLETED' and events[-1]['event']=='RUN_COMPLETED'
        assert [p['status'] for p in tl.project(m,events)]==['COMPLETED']*5
        context=tl.marker_context(state,events,m,app.run.manifest_sha,prepared=prepared)
        post=mk.decode(drawn[-1],m,app.run.manifest_sha,state['run_id'],run_context=context)
        assert post['reason']=='TECHNICAL_TRANSITION_FRAME' and len(set(post['decoded_tokens']))==1
        before={p.name:photo.digest(p) for p in app.run.directory.iterdir() if p.is_file()}
        clock.value+=60_000_000_000;app._tick()
        # Native Tk dispatch of the window-close command uses the same handler
        # bound to Escape, without mapping any window onto the TV.
        assert app.viewer.bind('<Escape>')
        app.viewer.tk.call(app.viewer.protocol('WM_DELETE_WINDOW'))
        assert app.mode=='COMPLETED' and app.viewer is None and not app._post_roll_hold
        assert '5/5' in app.status.get() and 'Przerwany' not in app.status.get()
        assert before=={p.name:photo.digest(p) for p in app.run.directory.iterdir() if p.is_file()}
        app.stop_btn.configure(state='normal');app.stop_btn.invoke();assert app.mode=='COMPLETED'
        # A failed final atomic checkpoint must not remain COMPLETED in memory
        # or be accepted from the uncheckpointed journal tail on disk.
        original=tv._atomic_write_json
        def fail(path,value):
            if Path(path).name=='run.json' and value.get('status')=='COMPLETED':raise OSError('TEST_ONLY_DISK_FULL')
            return original(path,value)
        monkeypatch.setattr(tv,'_atomic_write_json',fail)
        app.begin_countdown();advance()
        assert app.mode=='ERROR' and app.run.status==app.run.state['status']=='LOG_ERROR'
        assert photo.read_json(app.run.directory/'run.json')['status']!='COMPLETED'
        with pytest.raises(ValueError,match='UNCHECKPOINTED'):tl.read_run(app.run.directory/'run.json',m,app.run.manifest_sha)
        assert 'niepotwierdzone' in app.status.get() and not errors
        save(evidence.out,'D1.json',dict(status='PASS',completed=5,postroll='T',late_close_preserves_completed=True,journal_failure='ERROR_NO_COMPLETED',native_tk=True,no_tv=True))
    finally:close_app(root,app)


def test_D2_exact_auto_match_rejections_and_single_dialog(evidence,tmp_path,monkeypatch):
    prepared,m=plan(tmp_path/'one');r=finished(prepared,m);image=picture(m,r);archive=archive_image(tmp_path/'archive',image)
    scopes=[prepared.parent];match=mobile.locate_run(archive,scopes)
    assert match['status']=='PASS',match
    assert match['run_path']==str((r.directory/'run.json').resolve()) and match['run_proof']['locator_used']
    index=next((prepared.parent/'run_index').glob('*.json'));assert photo.read_json(index)['run_id']==r.state['run_id']
    foreign,fm=plan(tmp_path/'foreign');finished(foreign,fm)
    assert mobile.locate_run(archive,[foreign.parent])['status']=='N/A'
    clone=tmp_path/'clone'/prepared.name;shutil.copytree(prepared,clone)
    ambiguous=mobile.locate_run(archive,[prepared.parent,clone.parent]);assert ambiguous['reason']=='AMBIGUOUS_RUNS',ambiguous
    other=picture(fm,finished(foreign,fm));l,t,right,b=m['layout']['qr_rects'][1]
    conflict=image.copy();conflict.paste(other.crop((l,t,right,b)),(l,t))
    assert mobile.locate_run(archive_image(tmp_path/'conflict',conflict),scopes)['status']=='N/A'
    one_qr=image.copy();one_qr.paste('white',(l,t,right,b))
    assert mobile.locate_run(archive_image(tmp_path/'one_qr',one_qr),scopes)['status']=='N/A'
    # Index corruption, log corruption and TOCTOU are all rejected.
    raw=index.read_bytes();entry=photo.read_json(index);entry['run_id']='f'*32;save(index.parent,index.name,entry)
    assert mobile.locate_run(archive,scopes)['status']=='N/A';index.write_bytes(raw)
    log=r.directory/'events.jsonl';raw=log.read_bytes();log.write_bytes(raw+b'{}\n')
    with pytest.raises(ValueError):tl.verify_run_proof(match['run_proof'])
    assert mobile.locate_run(archive,scopes)['status']=='N/A';log.write_bytes(raw)
    # Real worker + native Tk queue/button flow: only the archive dialog on
    # success; the run dialog is deferred until the fallback button is clicked.
    root,app,errors=app_fixture(tmp_path,monkeypatch);dialogs=[];exports=[];panels=[]
    monkeypatch.setattr(tv,'DEFAULT_EXPERIMENT',prepared.parent)
    def dialog(**kwargs):dialogs.append(kwargs['title']);return str(archive) if len(dialogs)==1 else str(r.directory/'run.json')
    monkeypatch.setattr(tv.filedialog,'askopenfilename',dialog)
    monkeypatch.setattr(tv,'FinalResultsWindow',lambda *a,**kw:panels.append(a[1]))
    def export(archive_path,run_path,**kwargs):
        if kwargs.get('run_match'):tl.verify_run_proof(kwargs['run_match']['run_proof'])
        exports.append((str(archive_path),str(run_path)));return tmp_path/'private_result'
    monkeypatch.setattr(mobile,'analyze_import',export)
    def wait_for(predicate):
        deadline=time.monotonic()+40
        while not predicate() and time.monotonic()<deadline:root.update();time.sleep(.02)
        assert predicate(),app.status.get()
    try:
        app.import_btn.invoke();wait_for(lambda:bool(panels))
        assert len(dialogs)==1 and len(exports)==1 and not app.manual_run_btn.winfo_manager()
        monkeypatch.setattr(mobile,'locate_run',lambda *a:dict(status='N/A',reason='TEST_ONLY_UNMATCHED'))
        app.import_android(archive=archive);wait_for(lambda:bool(app.manual_run_btn.winfo_manager()))
        assert len(dialogs)==1 and 'Nie udało się jednoznacznie' in app.status.get()
        app.manual_run_btn.invoke();wait_for(lambda:len(exports)==2)
        assert len(dialogs)==2 and not errors
        save(evidence.out,'D2.json',dict(status='PASS',exact_match=match,ambiguous=ambiguous['reason'],foreign_rejected=True,one_qr_rejected=True,conflict_rejected=True,corrupt_index_and_journal_rejected=True,dialogs_success=1,fallback_explicit=True))
    finally:close_app(root,app)


def test_D3_legacy_actual_diagnostics_and_unchanged_first_mt(evidence,tmp_path):
    from test_final180_desktop import frozen
    # Historical F6v1 and F2 v1 use their actual independent pixel decoders.
    legacy=frozen(tmp_path/'legacy',n=1,negative=0,dwell_seconds=5);lm=photo.read_json(legacy/'manifest.json');lr=finished(legacy,lm)
    legacy_image=picture(lm,lr)
    decoded=mk.decode(legacy_image,lm,lr.manifest_sha,lr.state['run_id'])
    assert decoded['status']=='PASS' and len(set(decoded['decoded_tokens']))==1,decoded
    candidates,issues=tl.discover_runs([legacy.parent]);assert not issues and decoded['decoded_tokens'][0] in candidates[0]['tokens']
    # Long historical QR readability varies after this fixture's downsampling.
    # Full-raster identity is mandatory; a compressed unreadable QR stays N/A.
    old_archive=archive_image(tmp_path/'legacy_archive',legacy_image);old=mobile.locate_run(old_archive,[legacy.parent])
    if old['status']=='PASS':assert old['run_proof']['run_id']==lr.state['run_id'] and old['observations']
    else:assert old['reason']=='NO_VERIFIED_RUN_TOKEN' and not old['observations']
    prior,pm=plan(tmp_path/'f2v1',version=mk.F2_LAYOUT);pr=finished(prior,pm)
    prior_archive=archive_image(tmp_path/'f2_archive',picture(pm,pr));assert mobile.locate_run(prior_archive,[prior.parent])['status']=='PASS'
    paths=evidence.paths;archive=Path(paths['private_archive']);prepared=Path(paths['prepared']);run=Path(paths['run'])
    baseline=photo.read_json(evidence.out/'private/baseline.json')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    match=mobile.locate_run(archive,[Path(paths['experiment']),prepared.parent,prepared.parents[2]/'F6_REFERENCE_AWARE_V3_20261113_001/technical_plans'])
    assert match['status']=='PASS' and match['run_path']==str(run),match
    target=mobile.analyze_import(archive,run,output=evidence.out/'analysis_results',run_match=match)
    actual=photo.read_json(target/'evaluation.json')
    for key in ('summary','rows','frames','raw_mt','raw_mz','presentation_timeline','integration_gaps'):
        assert actual[key]==baseline[key],key
    # Static semantic equality supplements actual report equality; no decoder,
    # first-MT, geometry or scoring changes can hide behind this one fixture.
    before=ast.parse((evidence.out/'before/final180_mobile.py').read_text('utf-8'))
    after=ast.parse(Path(mobile.__file__).read_text('utf-8'))
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(before),functions(after)
    for name in ('select_first','score_frame','_native_frames','analyze_archive','reference_report','summarize'):assert a[name]==b[name],name
    protected=photo.read_json(evidence.out/'protected_before.json')
    expected=next(r['sha256'] for r in protected if r['path']==str(Path(mk.__file__)))
    assert photo.digest(mk.__file__)==expected
    s=actual['summary'];assert (s['identified_p_frames'],s['technical_frames'],s['unidentified_frames'])==(6,5,3)
    assert (s['raw_mt_calls'],s['raw_mz_calls'],s['unlocated_mt_calls'],s['selected_mt_exposures'])==(11,2,2,0)
    assert actual['metrics_status']=='NOT_AVAILABLE_NO_GT' and s['system_full_read']['rate'] is None
    rows=[];reasons=Counter()
    for f in sorted(actual['frames'],key=lambda x:x['source_stamp']['source_sequence']):
        d=f.get('marker_decoding',{});tokens=d.get('decoded_tokens',[])
        reason=d.get('reason','');phase='P' if f['identity_status']=='PASS' else 'T' if reason=='TECHNICAL_TRANSITION_FRAME' else 'N/A'
        if len(tokens)==2:
            if not tokens[0]:reasons['LEFT_QR_UNREADABLE']+=1
            if not tokens[1]:reasons['RIGHT_QR_UNREADABLE']+=1
            if all(tokens) and tokens[0]!=tokens[1]:reasons['TOKEN_CONFLICT']+=1
        if phase=='N/A':reasons[reason]+=1
        rows.append(dict(sequence=f['source_stamp']['source_sequence'],frame_key=f['frame_key'],phase=phase,exposure_id=f.get('exposure_id'),
            reason=reason,source_image=f['source_image'],decoded_tokens=tokens,mt_invocation_ids=f['mt_invocation_ids']))
    assert reasons=={'ARUCO_MISSING_OR_DUPLICATED':3}
    save(evidence.out,'D3.json',dict(status='PASS',source_archive_sha256=actual['source_archive_sha256'],session_id=actual['session_id'],
        match=match,legacy_f6v1='PASS_FULL_RASTER_AND_JOURNAL_TOKEN',legacy_f6v1_compressed=old['status'],legacy_f2='PASS',first_mt_unchanged=True,decoder_unchanged=True,report=str(target),
        P=6,T=5,unassigned=3,identified_photographs=4,planned_photographs=5,raw_mt=11,raw_mz=2,unlocated_mt=2,qualifying_mt=0,
        failures=dict(reasons),left_qr_unreadable=0,right_qr_unreadable=0,token_conflicts=0,qr_not_attempted_no_aruco=3,frames=rows,
        metrics='N/A_NO_GT_AND_UNLOCATED_EARLIER_MT',A12_TV='NOT_VERIFIED'))
