"""Scene GT: real A2 membership plus private/synthetic save and native Tk tests."""
import copy, gc, hashlib, json, os, shutil, zipfile
from pathlib import Path
from types import SimpleNamespace as NS
import tkinter as tk
import pytest

from auto_annotation_tool.ranking import mobile_f5a_scene_gt as scene
from auto_annotation_tool.ranking import mobile_f5a_assisted as a3
from auto_annotation_tool.ranking import photo_reference as photo
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui.mobile_f5a_scene_gt import SceneReviewWindow
from auto_annotation_tool.gui.mobile_f5a_scene_gt import timeline_records
import test_mobile_p2b2_canvas_gui as native


def test_operator_k02_all_raw_consensus_history_and_frozen_c(k02,ref):
    store=scene.SceneReference(k02,ref);ev=scene.timeline_records(store.index)
    assert len(ev['calls'])==44 and len(ev['mz'])==51
    assert sum(r['readouts']!=r['consensus'] for r in ev['mz'].values())==21
    assert sum(r['event_type']=='consensus_updated' for r in ev['consensus_events'])==16
    assert sum(r['event_type']=='consensus_confirmed' for r in ev['consensus_events'])==5
    assert {r['event_seq'] for r in ev['consensus_events']}=={r['event_seq'] for m in ev['readouts'].values() for r in m['events']}
    assert any(not r['executed'] and any(r['consensus']) for r in ev['readouts'].values())
    result=scene.operator_results(store,evidence=ev)
    assert len(result['exposures'])==10 and result['progress']['unreviewed']==10
    assert all(r['operator_decision'] is None and r['C'] is not None for r in result['exposures'])
    assert sum(r['automatic']['counts']['mt'] for r in result['exposures'])==44
    assert sum(r['automatic']['counts']['mz'] for r in result['exposures'])==51
    assert result['exposures'][3]['automatic']['no_mt_evidence']
    assert result['exposures'][3]['automatic']['no_mt_system_exact'] is False
    assert result['exposures'][4]['automatic']['no_mt_system_exact'] is True
    assert all(r['automatic']['iou'] is None for r in result['exposures'])
    assert result['verified_summary']['accepted_accounting_exposures']==0 and result['verified_summary']['B']['denominator']==0
    assert not k02.sidecar_path.exists()


def test_multiset_duplicate_missing_partial_no_physical_guess():
    def plate(text,kind='FULL_TEXT'):return dict(gt_text=text,text_kind=kind,text_state=kind,decision_version=3)
    gt=dict(presence='PRESENT',objects=[plate('AA1'),plate('BB2')])
    r=scene.text_multiset(gt,['AA1','AA1'])
    assert r['exact'] is False and r['known_missing']=={'BB2':1} and r['unaccounted']=={'AA1':1}
    assert r['duplicate_readouts']=={'AA1':2} and r['physical_plate_association'].startswith('N/A')
    gt['objects'][1]=plate('BB','PARTIAL_TEXT');r=scene.text_multiset(gt,['AA1'])
    assert r['exact'] is None and r['known_recovered']=={'AA1':1}
    assert scene.text_multiset(dict(presence='NO_PLATES',objects=[]),[])['exact'] is True


@pytest.mark.parametrize('variant',['R0','R1','R2'])
def test_exposure_decisions_autosave_resume_history_export_private(tmp_path,ref,variant):
    sid=next(sid for sid,r in ref.sessions.items() if r['variant']==variant and r['replicate']==2 and r['block']=='BLOK_B')
    s=private(ref,sid,tmp_path/variant);ref.attach(s);eid=next(e['exposure_id'] for e in ref.exposures.values() if e['session_id']==sid)
    a3.decide_identity(s,eid,'CONFIRMED') # isolated historical data must survive
    old=copy.deepcopy(s.review.to_dict());store=scene.SceneReference(s,ref);eid=store.material.order[0]
    with pytest.raises(ValueError,match='protokół'):store.decide_exposure(eid,'CONFIRMED')
    store.accept_protocol();store.decide_exposure(eid,'CONFIRMED');rev=s.review.review_revision
    store.decide_exposure(eid,'CONFIRMED');assert s.review.review_revision==rev
    gid=next(g for g,r in store.index.calls.items() if r['exposure_id']==eid)
    aid=s.mt_invocations[gid].records[0]['id'];ctx=('attempt',gid,aid)
    store.decide_exposure(eid,'ERROR_REPORTED',reason='Sprawdź komplet',context=ctx)
    store.undo_exposure(eid);assert store.operator_decision(eid)['decision']=='CONFIRMED'
    store.undo_exposure(eid,redo=True);assert store.operator_decision(eid)['decision']=='ERROR_REPORTED'
    store.decide_exposure(eid,'NOT_ASSESSABLE',reason='Brak dowodu w źródle',context=ctx)
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);again=scene.SceneReference(reopened,ref)
    assert again.data==store.data and scene.exposure_review_state(again.data)['last_context']==list(ctx)
    assert again.data['reviews']=={} and not again.approved(eid)
    assert reopened.review.provenance[a3.KEY]==old['provenance'][a3.KEY]
    for key in ('subjects','attempt_annotations','sample_annotations','invocation_annotations','scene_annotations','review_mode','review_status'):
        assert reopened.review.to_dict().get(key)==old.get(key)
    out=tmp_path/'out';out.mkdir();reopened.export(out)
    result=json.loads((out/'f5a_exposure_reviews.json').read_text('utf-8'))
    assert result['progress']==dict(total=10,reviewed=1,confirmed=0,corrected=0,errors=0,na=1,unreviewed=9)
    assert result['history']==scene.exposure_review_state(again.data)
    assert (out/'f5a_operator_decisions.csv').is_file() and (out/'f5a_source_timings.csv').is_file()
    before=copy.deepcopy(again.index.ref.gt_states)
    other=again.index.exposures[again.material.order[1]]['photo_sha256']
    again.decide_exposure(eid,'CORRECTED',photo_sha=other,reason='Inne zdjęcie',diagnostic_b=False,override_b=True)
    result=scene.operator_results(again)
    assert result['exposures'][0]['automatic']['photo_sha256']!=result['exposures'][0]['corrected']['photo_sha256']
    assert result['exposures'][0]['corrected']['B']['value'] is False and again.index.ref.gt_states==before


def test_operator_save_failure_retries_without_false_state(k02,ref,monkeypatch):
    store=scene.SceneReference(k02,ref);store.accept_protocol();eid=store.material.order[0]
    before=copy.deepcopy(store.data);saved=k02.sidecar_path.read_bytes();fn=k02._save_change
    monkeypatch.setattr(k02,'_save_change',lambda _:(_ for _ in ()).throw(PermissionError('locked')))
    with pytest.raises(PermissionError):store.decide_exposure(eid,'CONFIRMED')
    assert store.data==before and k02.sidecar_path.read_bytes()==saved
    monkeypatch.setattr(k02,'_save_change',fn);store.decide_exposure(eid,'CONFIRMED')
    assert scene.exposure_progress(store.data)['confirmed']==1


@pytest.fixture
def operator_gui(k02,ref):
    import traceback
    gc.collect();root=tk.Tk();root.withdraw();errors=[]
    root.report_callback_exception=lambda *args:errors.append(''.join(traceback.format_exception(*args)))
    h=SceneReviewWindow(NS(window=root),k02,ref)
    native.pump(root,lambda:h.canvas.winfo_ismapped() and h._view_center is not None)
    try:
        yield root,h,h._canvas_editor,k02
        assert not errors,'\n'.join(errors)
    finally:
        if not h._closed:
            h.source.pending_action=None;h._canvas_editor.escape();h.close()
        root.destroy()


def test_native_operator_photo_mt_mz_protocol_and_no_auto_save(operator_gui,tmp_path,monkeypatch):
    root,h,ed,s=operator_gui;eid=h.current_invocation;canvas=h.canvas
    from auto_annotation_tool.gui import mobile_f5a_scene_gt as ui
    assert len(h.subject_tree.get_children(''))==10 and h._image_entry.startswith('A1:')
    assert h.source.read_only and not s.sidecar_path.exists() and h.canvas.find_withtag('a1_readonly')
    native.focus_canvas(h);h.canvas.event_generate('<Control-s>');native.pump(root);assert not s.sidecar_path.exists()
    native.key(h,'d');assert ed.mode!='draw';native.key(h,'r');assert not s.sidecar_path.exists()
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**k:False)
    native.key(h,'space');assert not s.sidecar_path.exists()
    gid=next(g for g,r in h.evidence['calls'].items() if r['exposure_id']==eid);m=h.evidence['calls'][gid]['mz'][0]
    h.subject_tree.selection_set('mz:'+m);h._selected();native.pump(root)
    assert h._nav_target==('mz',gid,m) and h.preview['gid']==gid
    assert 'Raw:' in h.current_status.get() and 'Konsensus:' in h.current_status.get()
    assert h._image_entry==h.evidence['frames'][h.preview['frame_id']]['full_frame_entry']
    assert not s.sidecar_path.exists()
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**k:True)
    native.key(h,'space');native.pump(root)
    assert h.session.operator_decision(eid)['origin']=='OPERATOR_EXPLICIT' and '1 z 10' in h.progress.get()
    saved=s.sidecar_path.read_bytes();h.source.navigate(1);native.pump(root)
    assert s.sidecar_path.read_bytes()==saved and h.canvas is canvas and h.preview is None
    h.select(eid);native.focus_canvas(h)
    h.canvas.event_generate('<Control-z>');native.pump(root)
    assert h.session.operator_decision(eid) is None
    h.canvas.event_generate('<Control-y>');native.pump(root);assert h.session.operator_decision(eid)
    h.canvas.event_generate('<Control-s>');native.pump(root)
    assert h.session.operator_decision(eid)['decision']=='CONFIRMED'
    original=h._image_entry
    other=h.session.index.exposures[h.session.material.order[1]]['photo_sha256']
    h.decide('CORRECTED',reason='Prywatna próba korekty',photo_sha=other);native.pump(root)
    assert h._image_entry=='A1:'+other and h.canvas.find_withtag('a1_readonly')
    h.source.restore();native.pump(root);assert h._image_entry==original
    from PIL import ImageGrab
    ImageGrab.grab(window=h.window.winfo_id()).save(tmp_path/'operator_k02.png')


@pytest.mark.parametrize('failure',['replace','json','lock'])
def test_native_operator_retry_resume_mz_and_na(operator_gui,monkeypatch,failure):
    root,h,ed,s=operator_gui
    from auto_annotation_tool.gui import mobile_f5a_scene_gt as ui
    from auto_annotation_tool.ranking import mobile_human_review as review
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:True)
    assert h.ensure_protocol();eid=h.current_invocation
    gid=next(g for g,r in h.evidence['calls'].items() if r['exposure_id']==eid);aid=h.evidence['calls'][gid]['mz'][0]
    timeline_click(root,h,('mz',gid,aid));saved=s.sidecar_path.read_bytes()
    with monkeypatch.context() as fault:
        if failure=='replace':fault.setattr(review.os,'replace',lambda *a:(_ for _ in ()).throw(PermissionError('private locked file')))
        elif failure=='json':fault.setattr(review,'_atomic_write',lambda *a:(_ for _ in ()).throw(TypeError('private serialization failure')))
        else:fault.setattr(s,'_save_change',lambda *a:(_ for _ in ()).throw(ValueError('Trwa zapis weryfikacji w innym oknie.')))
        h.decide('ERROR_REPORTED',reason='Prywatna próba zapisu');native.pump(root)
        assert h.source.pending_action and not h.session.operator_decision(eid)
        assert s.sidecar_path.read_bytes()==saved and '0 z 10' in h.progress.get()
        assert not h.select(h.session.material.order[1])
    h.retry_button.invoke();native.pump(root)
    assert h.source.pending_action is None and h.session.operator_decision(eid)['decision']=='ERROR_REPORTED'
    h.decide('NOT_ASSESSABLE',reason='Prywatny brak dowodu');native.pump(root)
    before=copy.deepcopy(h.session.data);h.close()
    again=SceneReviewWindow(NS(window=root),MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path),h.session.index.ref)
    try:
        native.pump(root)
        assert again.session.data==before and again._nav_target==('mz',gid,aid)
        assert again.preview['aid']==aid and '1 z 10' in again.progress.get()
        assert again.session.operator_decision(eid)['decision']=='NOT_ASSESSABLE'
    finally:again.close()


@pytest.mark.parametrize('size',['1000x700','1450x780'])
def test_native_operator_timeline_zoom_pan_same_canvas(operator_gui,tmp_path,size):
    test_native_timeline_photo_frame_mt_mz_and_viewport_are_read_only(operator_gui,tmp_path,size)


def test_native_operator_negative_and_missing_frame_keep_correct_exposure(operator_gui,monkeypatch):
    root,h,ed,s=operator_gui
    from auto_annotation_tool.gui import mobile_f5a_scene_gt as ui
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:True)
    eid=h.session.material.order[4];h.select(eid);native.pump(root)
    assert 'NO_PLATES' in h.reference_label.cget('text') and not s.sidecar_path.exists()
    h.decide('CONFIRMED');assert h.session.operator_decision(eid)['decision']=='CONFIRMED'
    other=h.session.material.order[1];gid=next(g for g,r in h.evidence['calls'].items() if r['exposure_id']==other)
    h.evidence['calls'][gid].update(frame_id=None,reason='PRIVATE_MISSING_SOURCE')
    h.subject_tree.selection_set('call:'+gid);h._selected();native.pump(root)
    assert h.current_invocation==other and h._image is None
    h.decide('NOT_ASSESSABLE',reason='Prywatny brak dowodu');native.pump(root)
    assert h.session.operator_decision(other)['decision']=='NOT_ASSESSABLE'
    assert h.session.operator_decision(eid)['decision']=='CONFIRMED'
    assert not h.source.pending_action and h.session.data['reviews']=={}


@pytest.fixture(scope='module')
def ref():
    path=os.environ.get('ALPR_A3_A2')
    if not path:pytest.skip('Read-only real A2 fixture required')
    return a3.Reference(path)


def private(ref,sid,path):
    path.mkdir(parents=True,exist_ok=True);archive=path/'source.alprsession'
    shutil.copyfile(ref.sessions[sid]['archive_path'],archive)
    return MobileReviewSession.open(archive,sidecar_path=path/'review.json')


@pytest.fixture
def k02(tmp_path,ref):return private(ref,'exp-1794265037188-43dc06b8',tmp_path/'k02')


def synthetic(tmp_path,*,n=1,second_frame=False):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path,n=n,suppressed=None,two_calls=second_frame)
    # Explicit synthetic source evidence, not inferred from a model outcome.
    for i,(aid,row) in enumerate(s.attempts.items()):
        row['mt_confidence']=.9-i*.01
        samples=s.sample_ids_by_attempt.get(aid,[])
        if samples:row.update(evidence_entry=s.samples[samples[0]]['image_entry'],evidence_kind='plate_crop')
    frames={};calls={};exposure='exposure';photo_sha='f'*64
    with zipfile.ZipFile(s.path) as z:digest=hashlib.sha256(z.read('samples/frames/source.jpg')).hexdigest()
    for i,(gid,g) in enumerate(s.mt_invocations.items()):
        row=g.records[0];fid='frame'+str(i)
        stamp={k:row.get(k) for k in a3.STAMP}
        frames[fid]=dict(frame_id=fid,session_id=s.session_id,source_stamp=stamp,full_frame_entry='samples/frames/source.jpg',
            full_frame_sha256=digest,verified=True,status='POTWIERDZONE',photo_sha256=photo_sha,capture_elapsed_nanos=i+1)
        calls[(s.session_id,gid)]=dict(frame_ids=[fid],status='POTWIERDZONE',photo_sha256=photo_sha)
        s.mt_detection_roles[gid]=dict(role_proof_status='COMPLETE',source_stamp=copy.deepcopy(stamp),
            final_attempt_ids=[r['id'] for r in g.records],suppressed_attempt_ids=[])
    objects=[]
    for i in range(n):
        obj=dict(object_id='PGT:test'+str(i),quad=None,gt_text='SYN'+str(i),text_exclusion_reason='',geometry_exclusion_reason='Nieustalona')
        photo._explicit(obj);photo._explicit_geometry(obj);objects.append(obj)
    ref=NS(verify=lambda:None,sessions={s.session_id:dict(archive_sha256=s.review.source_archive_sha256)},
        exposures={exposure:dict(exposure_id=exposure,session_id=s.session_id,planned_position=1,photo_sha256=photo_sha,image_id='img',evidence_frame_ids=list(frames))},
        frames=frames,calls=calls,frame_exposure={fid:exposure for fid in frames},
        gt_states={'img':dict(presence='PRESENT' if n else 'NO_PLATES',objects=objects)},binding=dict(test_fixture=True))
    ref.call_frame=lambda session,gid:next(fid for fid,f in frames.items() if f['source_stamp']['source_sequence']==session.mt_invocations[gid].records[0]['source_sequence'])
    ref.frame_key=lambda fid:a3.fingerprint([s.session_id,frames[fid]['full_frame_sha256'],frames[fid]['source_stamp']])
    return s,scene.SceneReference(s,ref),exposure


def test_proposals_never_save_or_copy_jpeg_geometry(k02,ref):
    store=scene.SceneReference(k02,ref);eid=store.material.order[0];state=store.state(eid)
    assert len(store.material.order)==10 and len(store.index.calls)==44
    assert not k02.sidecar_path.exists() and not store.approved(eid)
    canonical=ref.plates(store.index.exposures[eid]['photo_sha256'])
    assert [photo.display_text(p) for p in state['objects']]==[photo.display_text(p) for p in canonical]
    assert all(p['quad'] is None and p['localization_quad'] is None for p in state['objects'])
    assert state['frame']['full_frame_entry'].startswith('samples/frames/')
    result=scene.analyze(store)
    assert result['denominators']['exposures']==10 and result['denominators']['mt_calls']==44
    assert result['denominators']['approved_scenes']==0
    assert len(result['units'])==44 and result['main_alpr_success_rate'] is None
    assert not k02.sidecar_path.exists()


def test_timeline_real_k02_readouts_frames_and_id_clock_separation(k02,ref):
    before=copy.deepcopy(k02.review.to_dict());index=scene.Index(k02,ref);ev=timeline_records(index)
    assert len(ev['frames'])==56 and len(ev['calls'])==44 and len(ev['mz'])==51
    assert len(ev['aliases'])==66 and len(ev['events'])==10
    assert any(not f['calls'] for f in ev['frames'].values())
    assert all(m['trace_ids'] and all(t in ev['traces'] for t in m['trace_ids']) for m in ev['mz'].values())
    for a,m in ev['mz'].items():
        assert m['readouts']==[str(k02.samples[sid]['raw_prediction'] or '') for sid in m['sample_ids']]
    # A different measurement clock cannot change photograph membership.
    for trace in k02.bundle.trace_rows:trace['elapsed_ms']='999999999999';trace['timestamp_ms']='1'
    assert timeline_records(index)==ev and k02.review.to_dict()==before
    assert not k02.sidecar_path.exists()


def test_timeline_no_fabricated_mz_missing_evidence_and_late_source(tmp_path):
    s,store,eid=synthetic(tmp_path,n=2);index=store.index;gid=next(iter(index.calls))
    frozen=s.sidecar_path.read_bytes()
    aids=[r['id'] for r in s.mt_invocations[gid].records]
    for a in aids:s.attempts[a]['mz_executed']='false';s.attempts[a]['mz_status']='NOT_RUN'
    assert not timeline_records(index)['mz']
    a=aids[0];s.attempts[a].update(mz_executed='true',mz_status='READ',raw_prediction='OLD',
        stale_or_cancelled='true',cancel_reason='scene_superseded')
    ev=timeline_records(index)
    assert len(ev['mz'])==1 and ev['mz'][a]['stale']
    assert ev['mz'][a]['exposure_id']==eid and ev['mz'][a]['cancel_reason']=='scene_superseded'
    index.calls[gid].update(frame_id=None,exposure_id=None,reason='MISSING_SOURCE')
    index.ref.call_frame=lambda *a:(_ for _ in ()).throw(ValueError('missing'))
    ev=timeline_records(index)
    assert ev['calls'][gid]['frame_id'] is None and ev['mz'][a]['frame_id'] is None
    assert ev['mz'][a]['exposure_id'] is None and s.sidecar_path.read_bytes()==frozen


@pytest.mark.parametrize('variant',['R0','R1','R2'])
def test_real_variants_one_save_many_calls_resume_export_no_old_decision_changes(tmp_path,ref,variant):
    sid=next(sid for sid,r in ref.sessions.items() if r['variant']==variant and r['replicate']==2 and r['block']=='BLOK_B')
    s=private(ref,sid,tmp_path/variant);ref.attach(s)
    eid=next(e['exposure_id'] for e in ref.exposures.values() if e['session_id']==sid)
    a3.decide_identity(s,eid,'CONFIRMED')  # private historical A3 fixture only
    old=copy.deepcopy(s.review.to_dict());source=copy.deepcopy((s.attempts,s.scene_frames,s.mt_detection_roles));digest=a3.sha(s.path)
    store=scene.SceneReference(s,ref);eid=store.material.order[0]
    store.approve(eid);revision=s.review.review_revision;store.approve(eid)
    assert s.review.review_revision==revision and store.approved(eid)
    assert s.review.provenance[a3.KEY]==old['provenance'][a3.KEY]
    for key in ('subjects','attempt_annotations','sample_annotations','invocation_annotations','scene_annotations','review_mode','review_status'):
        assert s.review.to_dict().get(key)==old.get(key)
    result=scene.analyze(store)
    assert len(result['units'])==len(s.mt_invocations)
    assert not s.review.invocation_annotations and not s.review.scene_annotations
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);again=scene.SceneReference(reopened,ref)
    assert again.data==store.data and again.approved(eid)
    out=tmp_path/'export';out.mkdir();frozen=s.sidecar_path.read_bytes();reopened.export(out)
    exported=json.loads((out/'f5a_scene_gt.json').read_text(encoding='utf-8'))
    assert exported['gt_layer']==store.data and len(exported['units'])==len(s.mt_invocations)
    assert exported['policy']==scene.POLICY and exported['policy_sha256']==scene.fingerprint(scene.POLICY)
    assert (out/'f5a_calls.csv').is_file() and (out/'f5a_scenes.csv').is_file()
    assert s.sidecar_path.read_bytes()==frozen and s.review.provenance[a3.KEY]==old['provenance'][a3.KEY]
    assert (s.attempts,s.scene_frames,s.mt_detection_roles)==source and a3.sha(s.path)==digest


def test_content_propagates_geometry_does_not_and_main_rule_ignores_gt(k02,ref):
    store=scene.SceneReference(k02,ref);eid=store.material.order[0];before=copy.deepcopy(store.index.main)
    oid=store.state(eid)['objects'][0]['object_id']
    store.edit(eid,oid,quad=[(476,542),(494,540),(495,550),(477,552)],text='',reason='Nieczytelne na tej klatce')
    store.approve(eid);result=scene.analyze(store)
    units=[u for u in result['units'] if u['exposure_id']==eid]
    assert len(units)>1 and all(u['content_reference']['plate_ids']==[oid] for u in units)
    exact=[u for u in units if u['geometry_status']=='EXACT_CAMERA_BYTES'];assert exact
    other=[u for u in units if u['source_frame']['full_frame_sha256']!=store.state(eid)['frame']['full_frame_sha256']];assert other
    assert all(u['metric_tp_iou50'] is None and not any(r['geometry'] for r in u['rows']) for u in other)
    assert all(not r['mz']['canonical_ocr_eligible'] for u in units for r in u['rows'])
    assert store.index.main==before and not k02.review.invocation_annotations


def test_history_representative_change_never_transports_geometry(k02,ref):
    store=scene.SceneReference(k02,ref);eid=store.material.order[0];oid=store.state(eid)['objects'][0]['object_id']
    q=[(476,542),(494,540),(495,550),(477,552)];store.edit(eid,oid,quad=q);store.approve(eid)
    saved=store.state(eid);fid=next(f for f in store.index.frame_ids(eid) if f!=saved['frame']['frame_id'])
    store.choose_frame(eid,fid);changed=store.state(eid)
    assert not store.approved(eid) and changed['objects'][0]['quad'] is None
    assert changed['objects'][0]['gt_text']==saved['objects'][0]['gt_text']
    store.restore(eid);assert store.state(eid)==saved
    store.restore(eid,redo=True);assert store.state(eid)==changed


def test_no_plates_applies_to_many_frames_but_cancelled_stays_na(tmp_path):
    s,store,eid=synthetic(tmp_path,n=1)
    store.remove(eid,'PGT:test0');assert store.state(eid)['presence']=='UNRESOLVED'
    store.empty(eid);result=scene.analyze(store);u=result['units'][0]
    assert u['metric_fp_iou50']==1 and u['rows'][0]['operational_outcome']=='FP'
    assert result['exposures'][0]['main_alpr_success'] is False
    before=s.review.to_dict();store.index.ref.frames['frame0']['full_frame_sha256']='0'*64
    with pytest.raises(ValueError):store.index.verify_frame('frame0')
    assert s.review.to_dict()==before


def test_many_plates_match_geometry_not_ocr_and_partial_text_stays_na(tmp_path):
    s,store,eid=synthetic(tmp_path,n=2)
    for i in range(2):store.edit(eid,'PGT:test'+str(i),quad=[(150+220*i,140),(300+220*i,140),(300+220*i,210),(150+220*i,210)],text='WRONG')
    store.edit(eid,'PGT:test1',text='SY',text_state='PARTIAL_TEXT');store.approve(eid)
    result=scene.analyze(store);u=result['units'][0]
    assert u['metric_tp_iou50']==2 and u['metric_fp_iou50']==u['metric_fn_iou50']==0
    assert [r['gt_id'] for r in u['rows']]==['PGT:test0','PGT:test1']
    assert u['rows'][0]['mz']['canonical_ocr_eligible'] and not u['rows'][0]['mz']['exact_match']
    assert u['rows'][1]['mz']['cer'] is None
    assert result['exposures'][0]['main_alpr_success'] is None


def test_roi_loss_not_mt_fn_and_ambiguous_ocr_does_not_choose_best(tmp_path):
    s,store,eid=synthetic(tmp_path,n=1)
    store.edit(eid,'PGT:test0',quad=[(5,5),(40,5),(40,30),(5,30)])
    store.approve(eid);u=scene.analyze(store)['units'][0]
    assert u['coverage'][0]['coverage']=='OUTSIDE_ROI' and u['metric_fn_iou50']==0
    assert scene.analyze(store)['exposures'][0]['main_alpr_success'] is False
    store.edit(eid,'PGT:test0',quad=[(150,140),(300,140),(300,210),(150,210)],text='WA01234');store.approve(eid)
    sid=s.sample_ids_by_attempt['a0'][0];s.samples['extra']=dict(s.samples[sid],prediction='WRONG')
    s.sample_ids_by_attempt['a0'].append('extra')
    r=scene.analyze(store)['units'][0]['rows'][0]
    assert not r['mz']['canonical_ocr_eligible'] and r['mz']['quality_reason']=='AMBIGUOUS_SOURCE_READOUTS'
    assert len(r['mz']['source_sample_ids'])==2 and len(r['mz']['raw_source_readouts'])==2


def test_one_gt_state_many_identical_camera_frames_never_pool_best_result(tmp_path):
    s,store,eid=synthetic(tmp_path,n=2,second_frame=True)
    for i in range(2):store.edit(eid,'PGT:test'+str(i),quad=[(150+220*i,140),(300+220*i,140),(300+220*i,210),(150+220*i,210)])
    store.approve(eid);result=scene.analyze(store)
    assert len(store.data['reviews'])==1 and len(result['units'])==2
    assert all(u['geometry_status']=='EXACT_CAMERA_BYTES' for u in result['units'])
    assert result['exposures'][0]['main_call_ids']==['g0']
    assert result['exposures'][0]['main_alpr_success'] is False
    assert result['exposures'][0]['any_detected_at_least_once'] is True
    assert not s.review.invocation_annotations


def test_cancelled_and_partial_roi_do_not_become_false_fn(tmp_path):
    from dataclasses import replace
    s,store,eid=synthetic(tmp_path,n=1);gid=next(iter(s.mt_invocations))
    store.edit(eid,'PGT:test0',quad=[(90,100),(120,100),(120,130),(90,130)]);store.approve(eid)
    u=scene.analyze(store)['units'][0]
    assert u['coverage'][0]['coverage']=='PARTIAL_ROI' and u['metric_fn_iou50'] is None
    s.mt_invocations[gid]=replace(s.mt_invocations[gid],cancelled=True)
    u=scene.analyze(store)['units'][0]
    assert u['execution_status']=='CANCELLED' and u['operational_status']=='N/A'
    assert u['metric_tp_iou50'] is u['metric_fp_iou50'] is u['metric_fn_iou50'] is None
    with pytest.raises(ValueError):store.exception(eid,gid,'a0','TP',gt_id='PGT:test0')


def test_exception_is_operational_only_and_atomic_stale_writers(k02,ref):
    store=scene.SceneReference(k02,ref);eid=store.material.order[0];store.approve(eid)
    other=scene.SceneReference(MobileReviewSession.open(k02.path,sidecar_path=k02.sidecar_path),ref)
    u=next(u for u in scene.analyze(store)['units'] if u['exposure_id']==eid and u['exceptions'])
    aid=u['exceptions'][0];pid=store.state(eid)['objects'][0]['object_id']
    store.exception(eid,u['mt_invocation_id'],aid,'TP',gt_id=pid,text_readable=False,reason='Nieczytelne znaki')
    row=next(r for r in scene.analyze(store)['units'] if r['mt_invocation_id']==u['mt_invocation_id'])['rows'][0]
    assert row['operational_outcome']=='TP' and row['origin']=='OPERATOR_EXCEPTION'
    assert row['automatic_iou50_outcome'] is None and row['geometry'] is None and row['mz']['cer'] is None
    frozen=k02.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='innym oknie'):other.edit(eid,pid,text='STALE')
    assert k02.sidecar_path.read_bytes()==frozen


def test_all_actual_a2_calls_mixed_events_delay_and_airplanes(tmp_path,ref):
    mixed=[e for e in ref.data['scene_events'] if e['multiple_photographs_observed']]
    assert len(mixed)==8
    total=0;assigned=0;all_main={};outside=[];diagnostic=[]
    for sid in ref.sessions:
        s=private(ref,sid,tmp_path/sid);index=scene.Index(s,ref)
        timeline=timeline_records(index)
        assert len(timeline['calls'])==len(index.calls)
        assert sum(len(f['aliases']) for f in timeline['frames'].values())==len(index.frames)
        for aid,m in timeline['mz'].items():
            assert s.attempts[aid]['mz_executed']=='true'
            assert m['gid'] in index.calls
        assert len(index.exposures)==10 and len(index.calls)+len(index.diagnostics)==len(s.mt_invocations)
        diagnostic.extend(index.diagnostics)
        total+=len(index.calls);assigned+=sum(bool(r['exposure_id']) for r in index.calls.values())
        all_main[sid]=copy.deepcopy(index.main)
        for gid,row in index.calls.items():
            if row['exposure_id']:
                f=ref.frames[row['frame_id']]
                assert index.exposures[row['exposure_id']]['photo_sha256']==f['photo_sha256']
            else:outside.append((sid,gid,row['reason']))
        # A delayed completion timestamp cannot move an old input into a new photo.
        for a in s.attempts.values():a['finished_elapsed_nanos']=10**25;a['finished_at_utc']='2099-01-01'
        delayed=scene.Index(s,ref)
        assert delayed.calls==index.calls and delayed.main==all_main[sid]
        assert timeline_records(delayed)==timeline
        for event in (e for e in mixed if e['session_id']==sid):
            frames=[ref.frames[fid] for fid in event['all_frame_ids']]
            assert len({f['photo_sha256'] for f in frames if f.get('photo_sha256')})==2
            for gid,row in index.calls.items():
                if row['frame_id'] in event['all_frame_ids']:
                    assert ref.exposures[row['exposure_id']]['photo_sha256']==ref.frames[row['frame_id']]['photo_sha256']
            assert timeline['events'][event['scene_event_id']]['multiple_photographs_observed']
        assert not s.sidecar_path.exists()
    assert total==1281 and assigned==1280 and len(outside)==1
    assert len(diagnostic)==2 and all(not r['source_record']['mt_invocation_id'] and r['source_record']['mt_executed']=='false' for r in diagnostic)
    airplanes=[f for f in ref.frames.values() if f.get('content_membership')=='OUTSIDE_GT_PHOTOGRAPH_SET']
    assert len(airplanes)==2 and all(f['frame_id'] not in ref.frame_exposure for f in airplanes)


@pytest.fixture
def gui(k02,ref):
    gc.collect();root=tk.Tk();root.withdraw();errors=[]
    root.report_callback_exception=lambda *args:errors.append(args)
    h=SceneReviewWindow(NS(window=root),k02,ref,operator_mode=False)
    native.pump(root,lambda:h.canvas.winfo_ismapped() and h._view_center is not None)
    try:
        yield root,h,h._canvas_editor,k02
        assert not errors,errors
    finally:
        if not h._closed:
            h.source.pending_action=None;h._canvas_editor.escape();h.close()
        root.destroy()


def test_native_one_scene_draw_gtz_approve_next_exceptions_and_same_canvas(gui,tmp_path,monkeypatch):
    root,h,ed,s=gui;eid=h.current_invocation;canvas=h.canvas
    assert len(h.subject_tree.get_children(''))==10 and not s.sidecar_path.exists()
    assert ed.reference_source is h.source and not hasattr(ed,'clean_bar')
    h.show_exceptions();assert not h.show_exception_rows and h.preview is None
    assert all(not h.subject_tree.item(i,'open') for i in h.subject_tree.get_children(''))
    native.key(h,'d')
    for p in [(476,542),(494,540),(495,550),(477,552)]:native.click(h,ed,p)
    assert ed.entry is not None
    oid=ed.entry_target[1]
    h.source.unreadable(oid);native.pump(root)
    assert len(h.session.state(eid)['objects'])==1 and not h.session.approved(eid)
    assert not h.canvas.find_withtag('scene_source_mt')
    native.key(h,'space');native.pump(root)
    assert h.session.approved(eid) and '1/10' in h.progress.get()
    revision=s.review.review_revision;native.key(h,'space');assert s.review.review_revision==revision
    h.source.navigate(1);native.pump(root)
    assert h.current_invocation!=eid and not h.session.approved(h.current_invocation)
    h.select(eid);h.show_exceptions();native.pump(root)
    assert h.canvas is canvas and h.preview and h.canvas.find_withtag('scene_source_mt')
    h.zoom_image(1.5);before_view=(h._view_zoom,h._view_center,id(h._image))
    p=copy.deepcopy(h.preview);h.inspect(p['frame_id'],gid=p['gid'],aid=None)
    assert (h._view_zoom,h._view_center,id(h._image))==before_view
    h.inspect(p['frame_id'],gid=p['gid'],aid=p['aid'])
    frozen=s.sidecar_path.read_bytes();native.key(h,'d');assert ed.mode!='draw' and s.sidecar_path.read_bytes()==frozen
    # Actual native Tk menu command; don't enter the blocking Windows popup loop.
    menu=h.source.context_menu(show=False)
    assert 'Trafne' in menu.entrycget(0,'label');menu.invoke(0);menu.destroy();native.pump(root)
    assert h.session.state(eid)['exceptions']
    h.select(eid);assert h.preview is None and not h.canvas.find_withtag('scene_source_mt')
    # Zoom/pan retain the exact same renderer and source coordinates.
    native.focus_canvas(h);h.canvas.event_generate('<MouseWheel>',delta=120,x=400,y=300)
    native.pump(root,lambda:not getattr(h,'_smooth_zoom',None) or not h._smooth_zoom.active)
    point=(480,545);assert ed.to_source(*ed.to_canvas(point))==pytest.approx(point)
    frozen=s.sidecar_path.read_bytes();h.fit_image();native.pump(root);assert s.sidecar_path.read_bytes()==frozen
    from PIL import ImageGrab
    ImageGrab.grab(window=h.window.winfo_id()).save(tmp_path/'k02_scene_gt.png')
    # Frame/exception children from another photo change the active GT scope.
    before=s.sidecar_path.read_bytes();other=h.session.material.order[1]
    h.inspect(h.session.index.frame_ids(other)[0]);native.pump(root)
    assert h.current_invocation==other and h.source.key==other
    h.source.navigate(1);native.pump(root)
    assert h.current_invocation==h.session.material.order[2]
    assert s.sidecar_path.read_bytes()==before


def test_native_partial_geometry_and_text_keep_old_a1_and_f4_contracts(gui):
    root,h,ed,s=gui;eid=h.current_invocation;oid=h.session.state(eid)['objects'][0]['object_id']
    before=copy.deepcopy(h.session.index.ref.gt_states)
    h.source.set_text_kind(oid,'PARTIAL_TEXT');ed.entry_var.set('CTR');ed.entry_enter()
    h.source.set_geometry_status(oid,'OCCLUDED');native.key(h,'space');native.pump(root)
    state=h.session.state(eid)
    assert h.session.approved(eid) and photo.text_kind(state['objects'][0])=='PARTIAL_TEXT'
    assert photo.geometry_status(state['objects'][0])=='OCCLUDED' and state['objects'][0]['quad'] is None
    assert h.session.index.ref.gt_states==before
    ed.undo();native.pump(root);assert not h.session.approved(eid)
    ed.undo(redo=True);native.pump(root);assert h.session.approved(eid)


def test_native_browser_entry_resume_export_and_f4_gate(tmp_path,k02,ref,monkeypatch):
    from auto_annotation_tool.gui import mobile_f5a_scene_gt as ui,z4_mobile_sample_review as entry
    from auto_annotation_tool.ranking import mobile_human_review
    cls=MobileReviewSession
    monkeypatch.setattr(mobile_human_review,'MobileReviewSession',lambda bundle:cls.open(k02.path,sidecar_path=k02.sidecar_path))
    monkeypatch.setattr(a3,'registered',lambda:ref)
    assert ui.open_if_supported(NS(),NS(experiment_index={'series_id':'F4'})) is None
    gc.collect();root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *e:errors.append(e)
    browser=NS(window=root)
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:True) # private protocol acceptance fixture
    h=entry.open_mobile_sample_review(browser,k02.bundle.report,k02.bundle)
    try:
        native.pump(root,lambda:h._image is not None)
        assert isinstance(h,SceneReviewWindow)
        assert entry.open_mobile_sample_review(browser,k02.bundle.report,k02.bundle) is h
        eid=h.session.material.order[1];h.select(eid);h.source.confirm();native.pump(root)
        assert h.operator_mode and h.session.operator_decision(eid)['decision']=='CONFIRMED'
        assert not h.session.approved(eid);saved=copy.deepcopy(h.session.data)
        h.close()
        h=entry.open_mobile_sample_review(browser,k02.bundle.report,k02.bundle);native.pump(root)
        assert h.current_invocation==eid and h.session.data==saved
        monkeypatch.setattr(ui.filedialog,'askdirectory',lambda **kw:str(tmp_path/'export'))
        h.export();native.pump(root)
        output=next((tmp_path/'export').glob('F5A_scenes_*'))
        manifest=json.loads((output/'f5a_scene_export_manifest.json').read_text(encoding='utf-8'))
        assert len(manifest['files'])==7
        assert all(a3.sha(output/f['path'])==f['sha256'] for f in manifest['files'])
        from auto_annotation_tool.gui.z4_mobile_report_browser_ux import sidecar_preview
        root_store=tmp_path/'catalog';(root_store/'human_reviews').mkdir(parents=True)
        shutil.copyfile(k02.sidecar_path,root_store/'human_reviews'/(k02.review.source_archive_sha256+'.review.json'))
        preview=sidecar_preview(root_store,k02.review.source_archive_sha256,k02.session_id)
        assert preview['title']=='Oceniono 1 z 10 zdjęć' and preview['state']!='completed'
        assert not errors,errors
    finally:
        if not h._closed:h.close()
        root.destroy()


def test_native_negative_proposal_is_not_approved_until_one_explicit_click(tmp_path,ref):
    eid=next(eid for eid,e in ref.exposures.items() if ref.gt_states[e['image_id']]['presence']=='NO_PLATES')
    s=private(ref,ref.exposures[eid]['session_id'],tmp_path/'negative')
    gc.collect();root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *e:errors.append(e)
    h=SceneReviewWindow(NS(window=root),s,ref,operator_mode=False)
    try:
        h.select(eid);native.pump(root)
        assert h.empty_button.cget('text')=='Brak tablic' and not s.sidecar_path.exists()
        assert '0/10' in h.progress.get()
        h.empty_button.invoke();native.pump(root)
        assert h.session.approved(eid) and s.review.review_revision==1
        assert h.empty_button.cget('text')=='Brak tablic ✓' and '1/10' in h.progress.get()
        h.source.restore();native.pump(root);assert not h.session.approved(eid)
        h.source.restore(redo=True);native.pump(root);assert h.session.approved(eid)
        assert not errors,errors
    finally:h.close();root.destroy()


def test_native_failed_initial_read_is_not_a_dirty_save(k02,ref,monkeypatch,caplog):
    """Replay the actual smoke-test mistake: full frame sent to crop reader."""
    from auto_annotation_tool.gui import z4_photo_reference as ui
    reader=scene.Index.frame_image
    monkeypatch.setattr(scene.Index,'frame_image',lambda index,fid:
        index.session.image(dict(image_entry=index.frames[fid]['full_frame_entry'])))
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:pytest.fail('Untouched window must close without warning'))
    gc.collect();root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *e:errors.append(e)
    h=SceneReviewWindow(NS(window=root),k02,ref,operator_mode=False)
    try:
        native.pump(root)
        assert h._image is None and h.current_invocation==''
        assert 'Nie można otworzyć' in h.status_var.get()
        assert 'Obraz musi należeć do cropów' in caplog.text and 'Traceback' in caplog.text
        h.confirm_button.invoke();h.empty_button.invoke();native.focus_canvas(h)
        h.canvas.event_generate('<Control-KeyPress-s>');native.pump(root)
        h.source.context_menu(show=False);h.show_exceptions()
        assert h.source.pending_action is None and not h.source.editing()
        assert not h.retry_button.winfo_ismapped() and 'NIE ZAPISANO' not in h.status_var.get()
        assert not k02.sidecar_path.exists() and '0/10' in h.progress.get()
        diagnostic=(Path.home()/'.auto_annotation_tool/diagnostics/z4_gt_errors.log').read_text(encoding='utf-8')
        assert 'Traceback' in diagnostic and 'ValueError: Obraz musi należeć do cropów' in diagnostic
        # Closing even an unreadable first scene must never claim lost edits.
        h.close();assert h._closed
        monkeypatch.setattr(scene.Index,'frame_image',reader)
        h=SceneReviewWindow(NS(window=root),k02,ref,operator_mode=False);native.pump(root)
        assert h._image.size==(960,1280) and h.current_invocation==h.session.material.order[0]
        assert h._image_entry.startswith('samples/frames/') and not k02.sidecar_path.exists()
        assert not errors
    finally:h.close();root.destroy()


def test_native_read_retry_by_selecting_photo(k02,ref,monkeypatch):
    reader=scene.Index.frame_image
    monkeypatch.setattr(scene.Index,'frame_image',lambda *a:(_ for _ in ()).throw(FileNotFoundError('private missing frame')))
    gc.collect();root=tk.Tk();root.withdraw();h=SceneReviewWindow(NS(window=root),k02,ref,operator_mode=False)
    try:
        native.pump(root);assert h._image is None and h.source.pending_action is None
        monkeypatch.setattr(scene.Index,'frame_image',reader)
        eid=h.session.material.order[0];h.subject_tree.selection_set(eid);native.pump(root)
        assert h.current_invocation==eid and h._image.size==(960,1280)
        assert h.source.pending_action is None and not k02.sidecar_path.exists()
    finally:h.close();root.destroy()


@pytest.mark.parametrize('failure',['replace','json','lock'])
def test_native_failed_approval_retains_state_and_retries_once(gui,monkeypatch,caplog,failure):
    from auto_annotation_tool.ranking import mobile_human_review as review
    root,h,ed,s=gui;eid=h.current_invocation
    # Persist a draft first: the previous on-disk revision must survive failure.
    oid=h.session.state(eid)['objects'][0]['object_id'];h.session.edit(eid,oid,text='PRIVATE123')
    saved=s.sidecar_path.read_bytes();before=copy.deepcopy(h.session.data);revision=s.review.review_revision
    with monkeypatch.context() as fault:
        if failure=='replace':
            fault.setattr(review.os,'replace',lambda *a:(_ for _ in ()).throw(PermissionError('synthetic sharing violation')))
        elif failure=='json':
            dumps=review.json.dumps
            def serialize(value,*args,**kwargs):
                if isinstance(value,dict) and value.get('schema')==review.REVIEW_SCHEMA:raise TypeError('private non-JSON value')
                return dumps(value,*args,**kwargs)
            fault.setattr(review.json,'dumps',serialize)
        else:
            lock=s.sidecar_path.with_suffix(s.sidecar_path.suffix+'.write.lock');lock.write_text('private lock')
        try:
            h.source.confirm();native.pump(root)
            assert h.source.pending_action and 'NIE ZAPISANO:' in h.status_var.get()
            assert h.status_var.get()!="NIE ZAPISANO: ''" and 'Traceback' not in h.status_var.get()
            assert 'Traceback' in caplog.text
            assert not h.session.approved(eid) and '0/10' in h.progress.get()
            assert h.session.data==before and s.sidecar_path.read_bytes()==saved and s.review.review_revision==revision
            h.source.navigate(1);assert h.current_invocation==eid
            assert not list(s.sidecar_path.parent.glob('*.tmp'))
            if failure=='lock':assert lock.read_text()=='private lock'
        finally:
            if failure=='lock':lock.unlink()
    h.retry_button.invoke();native.pump(root)
    assert h.source.pending_action is None and h.session.approved(eid) and '1/10' in h.progress.get()
    assert s.review.review_revision==revision+1 and not h.retry_button.winfo_ismapped()
    again=scene.SceneReference(MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path),h.session.index.ref)
    assert again.data==h.session.data and again.state(eid)['objects'][0]['gt_text']=='PRIVATE123'


def test_native_drawing_failure_inline_save_close_resume(gui,monkeypatch,tmp_path):
    from auto_annotation_tool.ranking import mobile_human_review as review
    from auto_annotation_tool.gui import z4_photo_reference as ui
    root,h,ed,s=gui;eid=h.current_invocation;q=[(476,542),(494,540),(495,550),(477,552)]
    # Only private destination; exercise mkdir as well as native drawing.
    s.sidecar_path=tmp_path/'new'/'nested'/'review.json';h.session.path=s.sidecar_path
    writer=review._atomic_write
    monkeypatch.setattr(review,'_atomic_write',lambda *a:(_ for _ in ()).throw(OSError('private disk failure')))
    native.key(h,'d')
    for p in q:native.click(h,ed,p)
    assert len(ed.points)==4 and h.source.pending_action and not s.sidecar_path.exists()
    assert not h.session.approved(eid)
    # Refusing to discard on close must retain all four draft points.
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:False)
    h.close();assert not h._closed and len(ed.points)==4
    monkeypatch.setattr(review,'_atomic_write',writer);h.retry_button.invoke();native.pump(root)
    assert not ed.points and ed.entry is not None
    oid=ed.entry_target[1];ed.entry.focus_force();native.pump(root);ed.entry_var.set('PRIVATE123')
    ed.entry.event_generate('<KeyRelease-3>')
    native.pump(root,lambda:h.session.state(eid)['objects'][0]['gt_text']=='PRIVATE123')
    assert not h.session.approved(eid)  # autosave is not approval
    ed.entry_enter();native.focus_canvas(h);h.canvas.event_generate('<Control-KeyPress-s>');native.pump(root)
    assert h.session.approved(eid)
    before=copy.deepcopy(h.session.data);monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:pytest.fail('Saved window must close cleanly'))
    h.close();again=SceneReviewWindow(NS(window=root),MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path),h.session.index.ref,operator_mode=False)
    try:
        native.pump(root);assert again.current_invocation==eid and again.session.data==before
        plate=again.session.state(eid)['objects'][0]
        assert plate['object_id']==oid and plate['gt_text']=='PRIVATE123'
        for actual,expected in zip(plate['quad'],q):assert actual==pytest.approx(expected,abs=2)
        # An unflushed inline change is saved by close, without approving it.
        again._canvas_editor.start_entry(oid);again._canvas_editor.entry_var.set('PRIVATE456');again.close()
        resumed=scene.SceneReference(MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path),h.session.index.ref)
        assert resumed.state(eid)['objects'][0]['gt_text']=='PRIVATE456' and not resumed.approved(eid)
    finally:
        if not again._closed:again.close()


def test_native_failed_change_of_approved_scene_is_not_shown_ready(gui,monkeypatch):
    from auto_annotation_tool.ranking import mobile_human_review as review
    root,h,ed,s=gui;eid=h.current_invocation;h.source.confirm()
    old=s.sidecar_path.read_bytes();oid=h.session.state(eid)['objects'][0]['object_id']
    ed.start_entry(oid);ed.entry_var.set('PRIVATE789')
    with monkeypatch.context() as fault:
        fault.setattr(review,'_atomic_write',lambda *a:(_ for _ in ()).throw(PermissionError('private write failure')))
        assert not ed.finish_entry()
        assert ed.entry.get()=='PRIVATE789' and h.source.pending_action
        assert 'Niezapisane' in h.current_status.get() and '0/10' in h.progress.get()
        assert h.subject_tree.item(eid,'values')[-1]=='Niezapisane'
        assert s.sidecar_path.read_bytes()==old  # preserve the earlier approval
    h.retry_button.invoke();native.pump(root)
    assert h.session.state(eid)['objects'][0]['gt_text']=='PRIVATE789'
    assert not h.session.approved(eid) and h.source.pending_action is None
    native.key(h,'space');assert h.session.approved(eid)
    h.show_exceptions();native.pump(root);preview=copy.deepcopy(h.preview)
    assert preview and preview['gid'] and preview['aid']
    old=s.sidecar_path.read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(review,'_atomic_write',lambda *a:(_ for _ in ()).throw(PermissionError('private exception failure')))
        menu=h.source.context_menu(show=False);menu.invoke(0);menu.destroy()
        assert h.source.pending_action
        assert not h.select(h.session.material.order[1])
        assert h.preview==preview and h.current_invocation==eid and s.sidecar_path.read_bytes()==old
    h.retry_button.invoke();native.pump(root)
    assert h.source.pending_action is None and h.session.state(eid)['exceptions']


def timeline_click(root,h,target):
    photos=target[0] in ('exposure','extra');canvas=h.timeline_photos if photos else h.timeline
    hit=next(v for v in (h._photo_hits if photos else h._timeline_hits) if v[-1]==target)
    x=(hit[0]+hit[2])/2
    if not photos and not canvas.canvasx(0)<=x<=canvas.canvasx(canvas.winfo_width()):
        extent=float(canvas.cget('scrollregion').split()[2]);canvas.xview_moveto(max(0,x-70)/extent)
    canvas.event_generate('<Button-1>',x=round(x-canvas.canvasx(0)),y=round((hit[1]+hit[3])/2))
    native.pump(root)
    assert h._nav_target==target
    assert h.targets[h.subject_tree.selection()[0]]==target


@pytest.mark.parametrize('size',['1000x700','1450x780'])
def test_native_timeline_photo_frame_mt_mz_and_viewport_are_read_only(gui,tmp_path,size):
    root,h,ed,s=gui;before=copy.deepcopy(s.review.to_dict());h.window.geometry(size);native.pump(root)
    eid=h.session.material.order[0];timeline_click(root,h,('exposure',eid))
    gid=next(g for g,r in h.evidence['calls'].items() if r['exposure_id']==eid)
    timeline_click(root,h,('call',gid))
    assert h.preview['gid']==gid and h.canvas.find_withtag('scene_source_mt')
    assert not h.session.approved(eid) and h.source.read_only
    fid=h.evidence['calls'][gid]['frame_id'];assert h._image_entry==h.evidence['frames'][fid]['full_frame_entry']
    aid=h.evidence['calls'][gid]['mz'][0]
    h.zoom_image(2);view=(h._view_zoom,h._view_center,id(h._image))
    timeline_click(root,h,('mz',gid,aid));assert (h._view_zoom,h._view_center,id(h._image))==view
    assert h.evidence['mz'][aid]['readouts'][0] in h.current_status.get()
    assert any(h.canvas.itemcget(i,'outline')=='#FFD36B' for i in h.canvas.find_withtag(aid))
    timeline_click(root,h,('frame',fid));assert h._view_zoom==view[0]
    # A native wheel burst still uses the existing anchored renderer.
    native.focus_canvas(h);anchor=(h.canvas.winfo_width()//2,h.canvas.winfo_height()//2);point=ed.to_source(*anchor)
    for delta in [120]*10+[-120]*10:
        h.canvas.event_generate('<MouseWheel>',delta=delta,x=anchor[0],y=anchor[1])
        native.pump(root,lambda:not getattr(h,'_smooth_zoom',None) or not h._smooth_zoom.active)
        assert ed.to_source(*anchor)==pytest.approx(point,abs=1e-6)
    assert h._view_zoom==pytest.approx(view[0])
    center=h._view_center
    h.canvas.event_generate('<ButtonPress-1>',x=20,y=30)
    h.canvas.event_generate('<B1-Motion>',x=50,y=50)
    h.canvas.event_generate('<ButtonRelease-1>',x=50,y=50);native.pump(root)
    assert h._view_center!=center
    h.fit_image();native.pump(root);assert h._view_zoom==1
    other=h.session.material.order[3];timeline_click(root,h,('exposure',other))
    assert h.preview is None and not h.canvas.find_withtag('scene_source_mt')
    free=next(f for f,r in h.evidence['frames'].items() if r['exposure_id']==other and not r['calls'])
    timeline_click(root,h,('frame',free));assert not h.canvas.find_withtag('scene_source_mt')
    assert s.review.to_dict()==before and not s.sidecar_path.exists()
    timeline_click(root,h,('exposure',eid));timeline_click(root,h,('mz',gid,aid));h.fit_image();native.pump(root)
    assert 'GT oryginalnego zdjęcia' in h.reference_label.cget('text')
    from PIL import ImageGrab
    ImageGrab.grab(window=h.window.winfo_id()).save(tmp_path/('timeline_'+size+'.png'))
    h._toggle_details();assert 'nie jest czasem pobrania' in h.timeline_details.cget('text')
    h.close();assert not s.sidecar_path.exists()


def test_native_tree_timeline_missing_frame_does_not_reuse_previous_bitmap(gui):
    root,h,ed,s=gui;gid=next(iter(h.evidence['calls']));eid=h.current_invocation
    h.evidence['calls'][gid].update(frame_id=None,reason='MISSING_SOURCE')
    h.subject_tree.selection_set('call:'+gid);native.pump(root)
    assert h._nav_target==('call',gid) and h._image is None
    assert 'N/A' in h.current_status.get() and not h.canvas.find_withtag('scene_source_mt')
    assert h.reference_label.cget('image')=='' and 'N/A' in h.reference_label.cget('text')
    h.window.geometry('1100x720');native.pump(root)
    assert not s.sidecar_path.exists() and h.source.pending_action is None
    timeline_click(root,h,('exposure',eid));assert h._image is not None and h.preview is None


@pytest.mark.parametrize('variant',['R0','R2'])
@pytest.mark.parametrize('operator_mode',[False,True])
def test_native_variant_evidence_many_roi_and_trace_ambiguity(tmp_path,ref,variant,operator_mode):
    sid=next(sid for sid,r in ref.sessions.items() if r['variant']==variant and r['replicate']==2 and r['block']=='BLOK_B')
    s=private(ref,sid,tmp_path/variant);before=copy.deepcopy(s.review.to_dict())
    gc.collect();root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *e:errors.append(e)
    h=SceneReviewWindow(NS(window=root),s,ref,operator_mode=operator_mode)
    try:
        native.pump(root)
        # A genuine CameraX input can have two MT calls; both remain selectable.
        fid,f=max(h.evidence['frames'].items(),key=lambda item:len(item[1]['calls']))
        eid=f['exposure_id'];timeline_click(root,h,('exposure',eid))
        for gid in f['calls']:
            timeline_click(root,h,('call',gid))
            assert h._image_entry==f['full_frame_entry'] and h.preview['gid']==gid
            for item in h.canvas.find_withtag('scene_source_roi'):
                from auto_annotation_tool.ranking.mobile_canvas_review import context
                l,t,r,b=context(s,gid)['roi'];ed=h._canvas_editor
                assert h.canvas.coords(item)==pytest.approx((*ed.to_canvas((l,t)),*ed.to_canvas((r,b))))
        if len(f['calls'])>1:
            assert variant=='R2'
            tids=set.intersection(*(set(h.evidence['calls'][g]['trace_ids']) for g in f['calls']))
            assert tids
            h._timing_selected(next(iter(tids)),'plate_inference_ms');native.pump(root)
            assert h._nav_target==('frame',fid) and 'rekordy' in h.status_var.get()
        assert not errors and not s.sidecar_path.exists() and s.review.to_dict()==before
    finally:h.close();root.destroy()


def test_native_existing_timing_chart_bidirectional_ids_and_no_pan_selection(gui,tmp_path):
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    from tkinter import ttk
    root,h,ed,s=gui;b=MobileReportBrowser.__new__(MobileReportBrowser)
    b.window=tk.Toplevel(root);b.window.geometry('1200x650');b.current_report=s.bundle.report;b.current_bundle=s.bundle
    b.panel='#eef2e9';b.border='#94a487';b.fg='#18251b';b.muted='#4c5c50'
    b._latency_x_view=None;b._latency_full_x=None;b._latency_pan_origin=None;b._latency_effective_mode=None
    b._latency_connect_neighbors_var=tk.BooleanVar(root,value=True);b._latency_axis_mode_var=tk.StringVar(root,value='time')
    b._latency_axis_footer_var=tk.StringVar(root);b._latency_context_var=tk.StringVar(root)
    b.notebook=ttk.Notebook(b.window);b.notebook.pack(fill='both',expand=True)
    tabs=[ttk.Frame(b.notebook) for _ in range(4)]
    for t in tabs:b.notebook.add(t,text='Dane')
    b.technical_notebook=ttk.Notebook(tabs[3]);b.technical_notebook.pack(fill='both',expand=True)
    tab=ttk.Frame(b.technical_notebook);b.technical_notebook.add(tab,text='Czasy szczegółowe')
    b.latency_split=ttk.PanedWindow(tab);b.latency_split.pack(fill='both',expand=True)
    b.latency_canvas=tk.Canvas(b.latency_split);b.latency_split.add(b.latency_canvas)
    b.latency_canvas.bind('<ButtonPress-1>',b._begin_latency_pan)
    b.latency_canvas.bind('<B1-Motion>',b._move_latency_pan)
    b.latency_canvas.bind('<ButtonRelease-1>',b._end_latency_pan)
    h.browser=b;gid=next(iter(h.evidence['calls']));aid=h.evidence['calls'][gid]['mz'][0]
    try:
        timeline_click(root,h,('mz',gid,aid));h._open_timing();native.pump(root);b._draw_latency_chart()
        assert b.latency_canvas.find_withtag('linked_evidence_timing')
        assert b._evidence_timing_link()['ids']==frozenset(h.evidence['mz'][aid]['trace_ids'])
        point=next(p for p in b._latency_hit_points if p[3]=='character_inference_ms' and
            str(s.bundle.trace_rows[p[2]]['frame_id']) in h.evidence['mz'][aid]['trace_ids'])
        h.select(h.session.material.order[1]);native.pump(root)
        b.latency_canvas.event_generate('<ButtonPress-1>',x=round(point[0]),y=round(point[1]))
        b.latency_canvas.event_generate('<ButtonRelease-1>',x=round(point[0]),y=round(point[1]));native.pump(root)
        assert h._nav_target==('mz',gid,aid)
        old=h._nav_target
        b.latency_canvas.event_generate('<ButtonPress-1>',x=round(point[0]),y=round(point[1]))
        b.latency_canvas.event_generate('<ButtonRelease-1>',x=round(point[0])+40,y=round(point[1]));native.pump(root)
        assert h._nav_target==old and not s.sidecar_path.exists()
        from PIL import ImageGrab
        b.window.lift();native.pump(root)
        ImageGrab.grab(window=b.window.winfo_id()).save(tmp_path/'existing_timing_ids.png')
        # A different report must not receive this session's selections.
        report=b.current_report;b.current_report=NS(source_archive_sha256='f'*64)
        assert b._evidence_timing_link() is None and not b.link_evidence_timing(s.review.source_archive_sha256,[],lambda *a:None)
        b.current_report=report;h.close();assert b._linked_evidence_timing is None
    finally:
        h.browser=NS(window=root);b.window.destroy()
