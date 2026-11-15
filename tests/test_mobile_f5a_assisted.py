"""A3 decisions only on private copies; real A2 and GT are read-only inputs."""
import copy,json,os,shutil
from pathlib import Path
import pytest

from auto_annotation_tool.ranking import mobile_f5a_assisted as a3
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

@pytest.fixture(scope='module')
def ref():
    path=os.environ.get('ALPR_A3_A2')
    if not path:pytest.skip('Read-only A2 fixture not configured')
    return a3.Reference(path)

@pytest.fixture
def session(tmp_path,ref):
    source=os.environ.get('ALPR_OBSERVATION_K02')
    if not source:pytest.skip('Private K02 fixture not configured')
    dest=tmp_path/'source.alprsession';shutil.copyfile(source,dest)
    s=MobileReviewSession.open(dest,sidecar_path=tmp_path/'review.json');ref.attach(s)
    return s

def exps(ref,s):return sorted([e for e in ref.exposures.values() if e['session_id']==s.session_id],key=lambda e:e['planned_position'])
def populated_call(ref,s):
    for gid,group in s.mt_invocations.items():
        if not group.executed or group.cancelled or group.execution_failed:continue
        fid=ref.call_frame(s,gid);photo=ref.frames[fid]['photo_sha256'];plates=ref.plates(photo)
        aids=[str(r['id']) for r in group.records if r['mt_status'] in ('VALID_QUAD','DETECTION_INVALID_QUAD') and a3.detection_role(s,gid,r['id'])=='FINAL']
        if plates and aids:return gid,fid,plates,aids
    raise AssertionError('No eligible fixture call')

def test_actual_36_360_386_sha_and_proposals_not_operator_approval(ref,session):
    s=session
    assert len(ref.sessions)==36 and len(ref.exposures)==360 and len(ref.data['scene_events'])==386
    assert ref.binding['gt_sha256']=='e7af93284b427925fa072e5f5315a4caecb5c90bb4095038d72fddaf6c1c7e82'
    assert not s.sidecar_path.exists() and not s.review.provenance.get(a3.KEY)
    assert all(not a3.unit_state(s,g)['complete'] for g in s.mt_invocations)
    assert a3.identity(s,exps(ref,s)[0]['evidence_frame_ids'][0])['decision']=='PROPOSED'

def test_identity_correction_rejection_na_undo_resume(ref,session):
    s=session;e=exps(ref,s)[0];fid=e['evidence_frame_ids'][0]
    a3.decide_identity(s,e['exposure_id'],'CONFIRMED');first=a3.identity(s,fid)
    other=next(h for h in ref.photos if h!=e['photo_sha256'])
    a3.decide_identity(s,fid,'CORRECTED',photo=other)
    assert a3.identity(s,fid)['photo_sha256']==other
    a3.undo(s);assert a3.identity(s,fid)==first
    a3.undo(s,redo=True);assert a3.identity(s,fid)['photo_sha256']==other
    with pytest.raises(ValueError):a3.decide_identity(s,fid,'REJECTED')
    a3.decide_identity(s,fid,'REJECTED',reason='Błędna fotografia')
    assert not a3.identity(s,fid)['approved'] and a3.identity(s,fid)['photo_sha256'] is None
    a3.decide_identity(s,fid,'N/A',reason='Brak pewności tożsamości')
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);ref.attach(reopened)
    assert a3.state(reopened)==a3.state(s)
    assert s.review.review_status=='IN_PROGRESS' and s.review.review_mode=='assisted'
    assert 'legacy_review_header' in s.review.provenance[a3.KEY]
    with pytest.raises(ValueError):s.set_review_mode('blinded_gt_v1')

def test_reviewed_block_never_approves_mt_or_overwrites_correction(ref,session):
    s=session;items=exps(ref,s)
    with pytest.raises(ValueError):a3.confirm_block(s,[items[0]['exposure_id']])
    a3.decide_identity(s,items[0]['exposure_id'],'N/A',reason='Do wyjaśnienia')
    a3.confirm_block(s,[e['exposure_id'] for e in items])
    st=a3.state(s);assert st['identities'][items[0]['exposure_id']]['decision']=='N/A'
    assert len(st['identities'])==10 and not st['detections'] and not st['coverage']
    assert st['blocks'][s.session_id]['mt_mz_approved'] is False
    assert all(not a3.unit_state(s,g)['complete'] for g in s.mt_invocations)

def test_tp_roi_text_and_geometric_na_are_separate(ref,session,tmp_path):
    s=session;gid,fid,plates,aids=populated_call(ref,s);pid=plates[0]['object_id'];aid=aids[0]
    with pytest.raises(ValueError):a3.decide_detection(s,gid,aid,'TP',gt_id=pid)
    a3.decide_identity(s,ref.frame_exposure[fid],'CONFIRMED')
    a3.decide_detection(s,gid,aid,'TP',gt_id=pid)
    assert not a3.state(s)['coverage']
    result=a3.unit_state(s,gid);assert 'ROI:'+pid in result['pending']
    row=next(r for r in result['rows'] if r['attempt_id']==aid)
    assert row['operational_outcome']=='TP' and row['polygon_iou'] is None and row['tp_at_iou_050'] is None and row['nmce'] is None
    a3.decide_coverage(s,gid,pid,'COVERED')
    with pytest.raises(ValueError):a3.decide_missing(s,gid,pid)
    with pytest.raises(ValueError):a3.decide_coverage(s,gid,pid,'ROI_COVERAGE_UNPROVEN')
    a3.unit_na(s,gid,'Tymczasowe wyłączenie');assert a3.unit_state(s,gid)['status']=='N/A'
    a3.decide_detection(s,gid,aid,'TP',gt_id=pid)
    assert a3.unit_state(s,gid)['status']!='N/A' and gid not in a3.state(s)['units']
    a3.decide_coverage(s,gid,pid,'OUTSIDE_ROI');assert 'ROI_CONFLICT:'+pid in a3.unit_state(s,gid)['pending']
    a3.decide_coverage(s,gid,pid,'COVERED')
    a3.text_na(s,aid,'Nie można ocenić odczytu w tej klatce')
    row=next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aid)
    assert row['cer'] is None and 'OPERATOR_NA' in row['quality_reason']
    export=tmp_path/'export';export.mkdir();paths=a3.export(s,export)
    payload=json.loads(paths[0].read_text(encoding='utf-8'));assert not payload['operator_complete']
    assert payload['layer']['binding']==ref.binding

def test_canonical_partial_never_becomes_full_ocr_and_identity_changes_invalidate(ref,session):
    s=session;gid,fid,plates,aids=populated_call(ref,s);aid=aids[0]
    partial=next((h,p) for h in ref.photos for p in ref.plates(h) if p.get('text_state')=='PARTIAL_TEXT')
    a3.decide_identity(s,fid,'CORRECTED',photo=partial[0])
    a3.decide_detection(s,gid,aid,'TP',gt_id=partial[1]['object_id'],coverage='COVERED')
    row=next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aid)
    assert row['gtz']=='' and row['cer'] is None and row['exact_match'] is None and row['quality_reason']=='PARTIAL_TEXT_NOT_FULL_GT'
    a3.decide_identity(s,fid,'CORRECTED',photo=ref.frames[fid]['photo_sha256'])
    row=next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aid)
    assert row['operational_outcome']=='UNRESOLVED'
    assert a3.state(s)['detections'][aid]['outcome']=='TP'  # audit remains, not silently overwritten

def test_multi_call_batch_preserves_each_invocation_and_source(ref,session):
    s=session;items=exps(ref,s);a3.confirm_block(s,[e['exposure_id'] for e in items])
    photo_calls={}
    for gid,g in s.mt_invocations.items():
        if not g.executed or g.cancelled or g.execution_failed:continue
        fid=ref.call_frame(s,gid);photo=ref.frames[fid]['photo_sha256']
        aids=[r['id'] for r in g.records if r['mt_status'] in ('VALID_QUAD','DETECTION_INVALID_QUAD')]
        if len(aids)==1 and a3.detection_role(s,gid,aids[0])=='FINAL' and ref.plates(photo):photo_calls.setdefault(photo,[]).append(gid)
    photo,gids=next((p,g) for p,g in photo_calls.items() if len(g)>1);gids=gids[:2];pid=ref.plates(photo)[0]['object_id']
    source=copy.deepcopy(s.attempts);rev=s.review.review_revision
    a3.batch_single_tp(s,gids,pid)
    assert s.review.review_revision==rev+1
    rows=[r for r in a3.state(s)['detections'].values() if r['gid'] in gids]
    assert len(rows)==2 and len({r['aid'] for r in rows})==2 and {r['gt_id'] for r in rows}=={pid}
    assert s.attempts==source
    a3.undo(s);assert not a3.state(s)['detections']
    a3.undo(s,redo=True);assert len(a3.state(s)['detections'])==2

def test_missing_requires_actual_roi_review_not_exposure_approval(ref,session):
    s=session;gid,fid,plates,aids=populated_call(ref,s);pid=plates[0]['object_id']
    a3.decide_identity(s,fid,'CONFIRMED')
    with pytest.raises(ValueError):a3.decide_missing(s,gid,pid)
    a3.decide_coverage(s,gid,pid,'OUTSIDE_ROI')
    with pytest.raises(ValueError):a3.decide_missing(s,gid,pid)
    a3.decide_coverage(s,gid,pid,'COVERED');a3.decide_missing(s,gid,pid)
    assert a3.state(s)['missing'][gid+'|'+pid]['outcome']=='FN'

def test_mixed_events_are_split_by_real_frame_and_airplanes_have_no_fake_gt(ref):
    mixed=next(e for e in ref.data['scene_events'] if e['multiple_photographs_observed'])
    frames=[ref.frames[f] for f in mixed['all_frame_ids']]
    assert len({f['photo_sha256'] for f in frames if f['photo_sha256']})==2
    assert len({ref.frame_exposure.get(f['frame_id']) for f in frames if f['photo_sha256']})==2
    outside=[f for f in ref.frames.values() if f.get('content_membership')=='OUTSIDE_GT_PHOTOGRAPH_SET']
    assert len(outside)==2 and all(f['photo_sha256'] is None for f in outside)

def test_no_automatic_save_and_changed_a2_sha_blocks_loading(ref,tmp_path):
    paths=(ref.path,ref.gt_path,ref.root/'artifacts_manifest.json',ref.root/'inventory.json',ref.root/'source_integrity_before.json')
    for p in paths:shutil.copyfile(p,tmp_path/p.name)
    path=tmp_path/'mapping.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA'):a3.Reference(path)

def test_stale_writer_and_io_error_are_atomic(ref,session,monkeypatch):
    s=session;e=exps(ref,s)[0]
    other=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);ref.attach(other)
    a3.decide_identity(s,e['exposure_id'],'CONFIRMED');before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='innym oknie'):a3.decide_identity(other,e['exposure_id'],'N/A',reason='test')
    assert s.sidecar_path.read_bytes()==before
    from auto_annotation_tool.ranking import mobile_human_review
    monkeypatch.setattr(mobile_human_review,'_atomic_write',lambda *a:(_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):a3.decide_identity(s,e['exposure_id'],'N/A',reason='test')
    assert s.sidecar_path.read_bytes()==before

def open_private(ref,sid,tmp_path):
    source=Path(ref.sessions[sid]['archive_path']);dest=tmp_path/'extra.alprsession';shutil.copyfile(source,dest)
    s=MobileReviewSession.open(dest,sidecar_path=tmp_path/'extra.review.json');ref.attach(s);return s

def test_mixed_frame_correction_does_not_approve_other_photo(ref,tmp_path):
    event=next(e for e in ref.data['scene_events'] if e['multiple_photographs_observed'])
    s=open_private(ref,event['session_id'],tmp_path);frames=[ref.frames[f] for f in event['all_frame_ids']]
    one=frames[0];other=next(f for f in frames if f['photo_sha256']!=one['photo_sha256'])
    a3.decide_identity(s,one['frame_id'],'CONFIRMED')
    assert a3.identity(s,one['frame_id'])['approved']
    assert not a3.identity(s,other['frame_id'])['approved']
    assert not s.review.invocation_annotations and not s.review.scene_annotations

def test_airplanes_na_and_negative_empty_unit_remain_explicit(ref,tmp_path):
    frame=next(f for f in ref.frames.values() if f.get('content_membership')=='OUTSIDE_GT_PHOTOGRAPH_SET')
    s=open_private(ref,frame['session_id'],tmp_path)
    gid=next(g for (sid,g),m in ref.calls.items() if sid==s.session_id and any(ref.frame_key(fid)==ref.frame_key(frame['frame_id']) for fid in m['frame_ids']))
    assert not a3.unit_state(s,gid)['complete']
    with pytest.raises(ValueError):a3.decide_identity(s,frame['frame_id'],'CONFIRMED')
    a3.decide_identity(s,frame['frame_id'],'N/A',reason='Samoloty poza materiałem GT');a3.unit_na(s,gid,'Brak referencji dla tego materiału')
    assert a3.unit_state(s,gid)['status']=='N/A' and not a3.unit_state(s,gid)['rows']
    assert not s.review.scene_annotations and not s.review.invocation_annotations
    negative=next(g for g,group in s.mt_invocations.items() if group.executed and not group.cancelled and not group.execution_failed and
        not ref.plates(ref.frames[ref.call_frame(s,g)]['photo_sha256']) and ref.frames[ref.call_frame(s,g)]['photo_sha256'] is not None and
        all(r['mt_status']=='NO_DETECTION' for r in group.records))
    fid=ref.call_frame(s,negative);a3.decide_identity(s,fid,'CONFIRMED')
    assert not a3.unit_state(s,negative)['complete']
    a3.confirm_empty_unit(s,negative);assert a3.unit_state(s,negative)['complete']

def test_report_approval_requires_all_decisions_and_is_never_automatic(ref,session,tmp_path):
    s=session
    with pytest.raises(ValueError):a3.approve_session(s)
    a3.confirm_block(s,[e['exposure_id'] for e in exps(ref,s)])
    with pytest.raises(ValueError):a3.approve_session(s)
    # Explicit reasoned exclusions in a PRIVATE fixture, not production review.
    for gid in s.mt_invocations:a3.unit_na(s,gid,'Prywatny test stanu N/A')
    assert not a3.state(s)['blocks'].get('approval:'+s.session_id)
    a3.approve_session(s);assert a3.state(s)['blocks']['approval:'+s.session_id]['status']=='OPERATOR_APPROVED_A3'
    e=exps(ref,s)[0];a3.decide_identity(s,e['exposure_id'],'N/A',reason='Korekta testowa')
    assert not a3.state(s)['blocks'].get('approval:'+s.session_id)

def test_shared_export_includes_a3_without_mutating_sources(ref,session,tmp_path):
    s=session;source=a3.sha(s.path);gt=a3.sha(ref.gt_path);mapping=a3.sha(ref.path)
    a3.decide_identity(s,exps(ref,s)[0]['exposure_id'],'CONFIRMED')
    directory=tmp_path/'shared_export';directory.mkdir();paths=s.export(directory)
    assert any(p.name=='f5a_a3_operator_review.json' for p in paths)
    payload=json.loads((directory/'review.json').read_text(encoding='utf-8'))
    assert a3.KEY in payload['provenance']
    assert (source,gt,mapping)==(a3.sha(s.path),a3.sha(ref.gt_path),a3.sha(ref.path))

def test_series_binding_reopens_without_creating_reviews(ref,session,tmp_path):
    path=tmp_path/'bindings.json';a3.register(ref,path)
    loaded=a3.registered(path)
    assert loaded.binding==ref.binding and len(loaded.sessions)==36
    assert set(json.loads(path.read_text(encoding='utf-8')))=={'schema','series'}
    assert not session.sidecar_path.exists()

def test_cancelled_calls_require_explicit_na_and_not_fn(ref,session):
    s=session
    gid=next(g for g,unit in s.mt_invocations.items() if unit.cancelled or unit.execution_failed or not unit.executed)
    fid=ref.call_frame(s,gid);a3.decide_identity(s,fid,'CONFIRMED')
    assert a3.unit_state(s,gid)['status']=='EXECUTION_UNASSESSABLE'
    with pytest.raises(ValueError,match='N/A'):a3.confirm_empty_unit(s,gid)
    plates=ref.plates(a3.identity(s,fid)['photo_sha256'])
    if plates:
        with pytest.raises(ValueError,match='N/A'):a3.decide_missing(s,gid,plates[0]['object_id'],confirm_coverage=True)
    a3.unit_na(s,gid,'Anulowane wywołanie — brak poprawnego wyniku')
    assert a3.unit_state(s,gid)['complete'] and a3.unit_state(s,gid)['status']=='N/A'

def test_duplicate_is_not_second_tp_and_loses_support_when_tp_rejected(ref,session):
    s=session
    for gid,group in s.mt_invocations.items():
        fid=ref.call_frame(s,gid);plates=ref.plates(ref.frames[fid]['photo_sha256'])
        aids=[r['id'] for r in group.records if r['mt_status'] in ('VALID_QUAD','DETECTION_INVALID_QUAD') and a3.detection_role(s,gid,r['id'])=='FINAL']
        if group.executed and not group.cancelled and not group.execution_failed and plates and len(aids)>1:break
    else:raise AssertionError('Fixture must contain several FINAL detections')
    a3.decide_identity(s,fid,'CONFIRMED');pid=plates[0]['object_id']
    a3.decide_detection(s,gid,aids[0],'TP',gt_id=pid)
    with pytest.raises(ValueError,match='duplikat'):a3.decide_detection(s,gid,aids[1],'TP',gt_id=pid)
    a3.decide_detection(s,gid,aids[1],'DUPLICATE',gt_id=pid)
    row=next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aids[1]);assert row['operational_outcome']=='DUPLICATE' and row['cer'] is None
    a3.decide_detection(s,gid,aids[0],'FP')
    assert next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aids[1])['operational_outcome']=='UNRESOLVED'

def test_ocr_requires_single_source_crop_and_never_chooses_best_read(ref,session):
    s=session
    for gid,g in s.mt_invocations.items():
        if not g.executed or g.cancelled or g.execution_failed:continue
        fid=ref.call_frame(s,gid);plates=ref.plates(ref.frames[fid]['photo_sha256'])
        full=next((p for p in plates if (p.get('text_state') or 'FULL_TEXT')=='FULL_TEXT' and p.get('gt_text')),None)
        if not full:continue
        for r in g.records:
            aid=r['id']
            if str(r.get('mz_executed')).lower() in ('true','1') and a3.detection_role(s,gid,aid)=='FINAL' and len(s.sample_ids_by_attempt.get(aid,()))==1:
                break
        else:continue
        break
    else:raise AssertionError('Fixture must contain full GT and a recorded MZ read')
    a3.decide_identity(s,fid,'CONFIRMED');a3.decide_detection(s,gid,aid,'TP',gt_id=full['object_id'])
    def scored():return next(r for r in a3.unit_state(s,gid)['rows'] if r['attempt_id']==aid)
    original=scored();assert original['canonical_ocr_eligible'] and original['cer'] is not None
    sample=s.sample_ids_by_attempt[aid][0];s.samples['private-extra']=dict(s.samples[sample],raw_prediction=full['gt_text'])
    s.sample_ids_by_attempt[aid].append('private-extra')
    assert scored()['cer'] is None and not scored()['canonical_ocr_eligible']
    s.sample_ids_by_attempt[aid].pop();s.attempts[aid]['evidence_kind']='unproven'
    assert scored()['cer'] is None and scored()['quality_reason']=='MZ_PARENT_UNPROVEN'

@pytest.fixture
def native_gui(tmp_path,monkeypatch,session,ref):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui import mobile_f5a_assisted as adapter
    install=adapter.install
    def opt_in(h,*args):
        h.browser._f5a_assistance_requested=True
        return install(h,*args)
    monkeypatch.setattr(adapter,'install',opt_in)
    monkeypatch.setattr(native,'source_session',lambda _:session)
    monkeypatch.setattr(a3,'registered',lambda:ref)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(gen)
    yield root,h,ed,s
    if h._closed:monkeypatch.setattr(ed,'escape',lambda:None)  # The close/reopen test already destroyed this Tk tree.
    try:next(gen)
    except StopIteration:pass

def test_native_existing_canvas_exposure_navigation_and_no_passive_write(native_gui,ref):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3
    assert ui.active and len(ui.exposures)==10
    assert not s.sidecar_path.exists()
    canvases=[]
    def scan(w):
        if w.winfo_class()=='Canvas':canvases.append(w)
        for c in w.winfo_children():scan(c)
    scan(h.window);assert h.canvas in canvases  # original Canvas, GT uses a Label
    assert ui.reference_label.winfo_class()=='Label'
    for _ in range(10):
        native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[ui.current_frame]['full_frame_entry'])
        ui.painted();ui.navigate(1)
    native.pump(root,lambda:h._image is not None)
    assert len(ui.seen)==10 and not s.sidecar_path.exists()
    ui.confirm_block();root.update()
    assert len(a3.state(s)['identities'])==10 and not a3.state(s)['detections']
    assert h.progress_label.cget('text')=='Zdjęcia: 10/10     MT: 0/44'


def picker_select(root,ui,sha):
    """Deliberate activation, distinct from arrow/search-only browsing."""
    win=ui.photo_picker;win.tree.selection_set(sha);win.tree.focus(sha);win.tree.see(sha)
    win.tree.focus_force();root.update();win.tree.event_generate('<Return>');root.update()


def test_native_photo_panel_and_ppm_execute_same_three_commands(native_gui,ref,monkeypatch):
    import tkinter as tk
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3;e=ui.exposures[0]
    ui.select('a3:exp:'+e['exposure_id']);native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[ui.current_frame]['full_frame_entry'])
    expected=['Zgadza się','Wybierz inne zdjęcie','Nie można ustalić']
    assert [b.cget('text') for b in ui.decision_buttons]==expected
    assert all(b.winfo_ismapped() for b in ui.decision_buttons)
    assert 'Dotyczy: zdjęcia 1' in ui.scope_label.cget('text') and 'Propozycja:' in ui.scope_label.cget('text')
    if os.environ.get('ALPR_A3_UX_SCREENSHOTS'):
        from PIL import ImageGrab
        from z4_tk_visibility_support import foreground_test_window,assert_native_widgets_visible
        with foreground_test_window(h.window,focus=h.canvas):
            assert_native_widgets_visible(root,h.window,list(ui.decision_buttons))
            ImageGrab.grab(window=h.window.winfo_id()).save(Path(os.environ['ALPR_A3_UX_SCREENSHOTS'])/'photo_actions.png')
    # Native Button-3 binding builds the actual Tk Menu. Invoke its registered
    # Tcl command; skip only the blocking Win32 popup message loop.
    posted=[];monkeypatch.setattr(tk.Menu,'tk_popup',lambda m,x,y:posted.append(m))
    h.canvas.event_generate('<Button-3>',x=10,y=10);root.update()
    assert posted and isinstance(posted[-1],tk.Menu)
    m=posted[-1];labels=[m.entrycget(i,'label') for i in range(m.index('end')+1)]
    assert labels==expected and all(m.type(i)=='command' for i in range(3))
    m.invoke(0);root.update()
    assert a3.state(s)['identities'][e['exposure_id']]['decision']=='CONFIRMED'
    assert not a3.state(s)['detections']


def test_native_picker_all_40_thumbnails_search_and_browse_without_save(native_gui,ref):
    from PIL import Image,ImageOps,ImageTk
    root,h,ed,s=native_gui;ui=h._f5a_a3;before=s.review.to_dict()
    win=ui.choose_photo();root.update();assert len(win.tree.get_children(''))==40
    if os.environ.get('ALPR_A3_UX_SCREENSHOTS'):
        from PIL import ImageGrab
        from z4_tk_visibility_support import foreground_test_window,assert_native_widgets_visible
        with foreground_test_window(win,focus=win.search_entry):
            assert_native_widgets_visible(root,win,[win.tree,win.search_entry,win.reject_button])
            ImageGrab.grab(window=win.winfo_id()).save(Path(os.environ['ALPR_A3_UX_SCREENSHOTS'])/'photo_picker.png')
    for sha,p in ref.photos.items():
        label=win.tree.item(sha,'text');plates=ref.plates(sha)
        assert label.startswith(f"GT {p['global_position']} · ") and sha not in label and Path(p['sciezka_zrodla']).name not in label
        if not plates:assert 'Bez tablic' in label
        if len(plates)>1:assert 'Kilka tablic' in label
        for plate in plates:
            if plate.get('text_state')=='PARTIAL_TEXT':assert 'Fragment: '+plate['partial_text'] in label
            elif plate.get('gt_text'):assert plate['gt_text'] in label
        with Image.open(ref.photo_root/p['sciezka_zrodla']) as source:im=ImageOps.exif_transpose(source).convert('RGB')
        im.thumbnail((84,56),Image.Resampling.LANCZOS)
        assert ImageTk.getimage(ui._gt_thumbnails[sha]).tobytes()==im.convert('RGBA').tobytes()
    target=next(sha for sha,p in ref.photos.items() if p['global_position']==11)
    win.query.set('11');root.update();assert win.tree.get_children('')==(target,)
    full=next((sha,p['gt_text']) for sha in ref.photos for p in ref.plates(sha) if p.get('gt_text'))
    win.query.set(full[1].lower());root.update();assert full[0] in win.tree.get_children('')
    fragment=Path(ref.photos[target]['sciezka_zrodla']).stem[:8]
    win.query.set(fragment);root.update();assert target in win.tree.get_children('')
    win.query.set('no-such-photo-private-test');root.update();assert not win.tree.get_children('')
    win.query.set('');root.update();assert len(win.tree.get_children(''))==40
    win.tree.selection_set(target);win.tree.focus_force();win.tree.event_generate('<Down>');root.update()
    assert s.review.to_dict()==before and not s.sidecar_path.exists()
    win.event_generate('<Escape>');root.update();assert ui.photo_picker is None
    assert s.review.to_dict()==before and not s.sidecar_path.exists()


def test_native_picker_correction_invalidates_dependents_keeps_history_export_resume(native_gui,ref,tmp_path):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3;gid,fid,plates,aids=populated_call(ref,s);eid=ref.frame_exposure[fid]
    ui.select('a3:mt:'+gid);native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    ui.confirm_identity();ui.judge(gid,aids[0],'TP',plates[0]['object_id'])
    old=a3.identity(s,fid);source=copy.deepcopy(s.attempts)
    ui.select('a3:exp:'+eid);native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[ui.current_frame]['full_frame_entry'])
    other=next(sha for sha in ref.photos if sha!=old['photo_sha256'])
    ui.more.invoke();root.update();assert ui.photo_picker is not None
    picker_select(root,ui,other)
    assert ui.photo_picker is None and a3.identity(s,fid)['decision']=='CORRECTED'
    assert a3.identity(s,fid)['photo_sha256']==other and ui._thumbnail_key[0]==other
    assert 'Poprawione' in ui.scope_label.cget('text')
    assert all(r['operational_outcome']=='UNRESOLVED' for r in a3.unit_state(s,gid)['rows'])
    assert old['decision_id'] in json.dumps(s.review.provenance[a3.KEY]['versions']) and s.attempts==source
    ui.undo();root.update();assert a3.identity(s,fid)==old
    ui.undo(True);root.update();assert a3.identity(s,fid)['photo_sha256']==other
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);ref.attach(reopened);assert a3.state(reopened)==a3.state(s)
    dest=tmp_path/'picker_export';dest.mkdir();reopened.export(dest)
    payload=json.loads((dest/'f5a_a3_operator_review.json').read_text(encoding='utf-8'));assert payload['layer']['state']==a3.state(s)


@pytest.mark.parametrize('decision',['REJECTED','N/A'])
def test_native_photo_rejected_and_na_require_reason_and_remain_distinct(native_gui,ref,monkeypatch,decision):
    root,h,ed,s=native_gui;ui=h._f5a_a3;e=ui.exposures[0]
    reason='Poza zestawem GT' if decision=='REJECTED' else 'Za mało dowodów'
    monkeypatch.setattr(ui,'_reason',lambda *args,**kwargs:None)
    if decision=='REJECTED':
        win=ui.choose_photo();root.update();win.reject_button.invoke();root.update()
    else:ui.third.invoke();root.update()
    assert not s.sidecar_path.exists()
    monkeypatch.setattr(ui,'_reason',lambda *args,**kwargs:reason)
    if decision=='REJECTED':win.reject_button.invoke()
    else:ui.third.invoke()
    root.update();row=a3.state(s)['identities'][e['exposure_id']]
    assert row['decision']==decision and row['photo_sha256'] is None and row['reason']==reason
    expected='Odrzucone' if decision=='REJECTED' else 'Nieustalone'
    assert expected in ui.scope_label.cget('text') and expected in ui.reference_label.cget('text')


def test_native_frame_picker_override_survives_general_photo_confirmation(native_gui,ref):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3;e=ui.exposures[0]
    fid=e['evidence_frame_ids'][0];other_frame=next(f for f in e['evidence_frame_ids'] if ref.frame_key(f)!=ref.frame_key(fid))
    iid=next(i for i,(kind,f) in ui.targets.items() if kind=='frame' and ref.frame_key(f)==ref.frame_key(fid))
    ui.select(iid);native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[ui.current_frame]['full_frame_entry'])
    assert 'Dotyczy: tylko tej klatki' in ui.scope_label.cget('text')
    replacement=next(sha for sha in ref.photos if sha!=e['photo_sha256'])
    ui.more.invoke();root.update();picker_select(root,ui,replacement)
    frame_decision=a3.identity(s,fid)
    assert frame_decision['decision']=='CORRECTED' and not a3.identity(s,other_frame)['approved']
    ui.select('a3:exp:'+e['exposure_id']);native.pump(root,lambda:h._image is not None)
    ed.clean_gtr_confirm_button.invoke();root.update()
    assert a3.identity(s,fid)==frame_decision
    assert a3.identity(s,other_frame)['decision']=='CONFIRMED' and a3.identity(s,other_frame)['photo_sha256']==e['photo_sha256']


def test_native_mt_buttons_duplicate_and_roi_fn_na_keep_existing_meanings(native_gui,ref,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui import mobile_f5a_assisted as ui_module
    root,h,ed,s=native_gui;ui=h._f5a_a3
    for gid,g in s.mt_invocations.items():
        fid=ref.call_frame(s,gid);plates=ref.plates(ref.frames[fid]['photo_sha256'])
        aids=[r['id'] for r in ui.detections(gid) if a3.detection_role(s,gid,r['id'])=='FINAL']
        if plates and len(aids)>1 and g.executed and not g.cancelled and not g.execution_failed:break
    else:raise AssertionError('Multiple actual detections required')
    ui.select('a3:mt:'+gid);native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    ui.confirm_identity();identity=copy.deepcopy(a3.state(s)['identities']);pid=plates[0]['object_id']
    ui.judge(gid,aids[0],'TP',pid)
    asked=[];monkeypatch.setattr(ui_module.messagebox,'askyesno',lambda *a,**k:asked.append(True) or True)
    ui.judge(gid,aids[1],'TP',pid)
    assert asked==[True] and a3.state(s)['detections'][aids[1]]['outcome']=='DUPLICATE'
    ui.select_detection(aids[1]);root.update()
    assert [b.cget('text') for b in ui.decision_buttons]==['Trafne','Błędne','Nie można ocenić']
    ui.more.invoke();root.update();assert a3.state(s)['detections'][aids[1]]['outcome']=='FP'
    monkeypatch.setattr(ui,'_reason',lambda *a,**k:'Brak dowodu odczytu')
    ui.third.invoke();root.update();assert a3.state(s)['detections'][aids[1]]['outcome']=='N/A'
    assert a3.state(s)['identities']==identity
    ui.judge(gid,aids[0],'FP')
    win=ui.unit_review(gid);root.update()
    next(w for w in win.winfo_children() if w.winfo_class()=='TButton' and w.cget('text')=='W ROI, ale pominięta przez MT').invoke();root.update()
    assert a3.state(s)['missing'][gid+'|'+pid]['outcome']=='FN'
    assert a3.state(s)['coverage'][gid+'|'+pid]['status']=='COVERED'

def test_native_call_context_menu_and_reference_text(native_gui,ref,monkeypatch):
    import tkinter as tk
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3;gid,fid,plates,aids=populated_call(ref,s)
    ui.select('a3:mt:'+gid);native.pump(root,lambda:h.current_invocation==gid and h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    ui.confirm_identity();root.update()
    ed.selected='GT:'+aids[0]
    menu=ui.menu(show=False);labels=[menu.entrycget(i,'label') for i in range(menu.index('end')+1) if menu.type(i)!='separator']
    assert labels==['Trafne','Błędne','Nie można ocenić']
    assert not a3.state(s)['detections']
    assert ui.photo is not None and h.canvas.find_withtag('evidence_image')
    assert not ed.ref_visible.get() and ed.inf_visible.get()
    posted=[];monkeypatch.setattr(tk.Menu,'tk_popup',lambda m,x,y:posted.append(m))
    source=next(o for o in ed._read()['objects'] if o['attempt_id']==aids[0])
    if source['quad']:
        point=[sum(p[j] for p in source['quad'])/4 for j in (0,1)];x,y=map(round,ed.to_canvas(point))
        h.canvas.event_generate('<Button-3>',x=x,y=y);root.update();assert posted
        menu=posted[-1]
    index=next(i for i in range(menu.index('end')+1) if menu.type(i)!='separator' and menu.entrycget(i,'label')=='Trafne')
    menu.invoke(index)
    if len(plates)>1:posted[-1].invoke(0)
    root.update()
    assert a3.state(s)['detections'][aids[0]]['outcome']=='TP'
    assert not s.review.invocation_annotations and not s.review.scene_annotations
    native.key(h,'d');assert ed.mode!='draw'
    ed.start_entry('GT:'+aids[0]);assert ed.entry is None
    assert not ed._commit(lambda:pytest.fail('Legacy write in A3'),'test')

def test_native_corrected_photo_survives_space_close_and_reopen(native_gui,ref,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui import z4_mobile_sample_review as module
    root,h,ed,s=native_gui;ui=h._f5a_a3
    frame=exps(ref,s)[-1]['evidence_frame_ids'][0]
    a3.decide_identity(s,frame,'CORRECTED',photo=next(p for p in ref.photos if p!=ref.frames[frame]['photo_sha256']))
    ui.attach(ref)
    native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[ui.current_frame]['full_frame_entry'])
    assert ref.frame_key(ui.current_frame)==ref.frame_key(frame)
    previous=a3.identity(s,frame);revision=s.review.review_revision
    native.key(h,'space');assert a3.identity(s,frame)==previous and s.review.review_revision==revision
    h.close();native.pump(root,lambda:h._closed);h._executor.shutdown(wait=True,cancel_futures=True)
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    monkeypatch.setattr(module,'MobileReviewSession',lambda bundle:reopened)
    h2=module.MobileSampleReviewWindow(h.browser,reopened.bundle.report,reopened.bundle)
    try:
        native.pump(root,lambda:h2.session is not None and h2._image is not None and getattr(h2,'_f5a_a3',None).active)
        assert ref.frame_key(h2._f5a_a3.current_frame)==ref.frame_key(frame)
        assert a3.identity(reopened,frame)==previous
    finally:
        h2.close();native.pump(root,lambda:h2._closed);h2._executor.shutdown(wait=True,cancel_futures=True)


def test_native_simple_tree_counts_and_navigation_preserve_every_source(native_gui,ref):
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3;tree=h.subject_tree
    assert h.session_label.cget('text')=='R1 · Powtórzenie 2 · Blok B'
    assert h.progress_label.cget('text')=='Zdjęcia: 0/10     MT: 0/44'
    assert not any(w.winfo_class()=='TCombobox' for w in ui.navigation.winfo_children())
    assert not ui.connect_button.winfo_ismapped() and not ed.clean_exclude_button.winfo_ismapped()
    assert not ui.left.grid_slaves(row=3)
    calls=[];frames=set()
    for n,e in enumerate(ui.exposures,1):
        iid='a3:exp:'+e['exposure_id'];children=tree.get_children(iid)
        assert tree.item(iid,'text').startswith(f'Zdjęcie {n} · GT ')
        folder=children[-1];assert folder=='a3:frames:'+e['exposure_id']
        assert tree.item(folder,'text').startswith('Klatki telefonu (') and not tree.item(folder,'open')
        calls.extend(ui.targets[i][1] for i in children[:-1])
        frames.update(ref.frame_key(ui.targets[i][1]) for i in tree.get_children(folder))
    assert len(calls)==len(set(calls))==44 and set(calls)==set(s.mt_invocations)
    assert frames=={ref.frame_key(f) for e in ui.exposures for f in e['evidence_frame_ids']}
    for iid in ui.targets:
        assert not any(value in tree.item(iid,'text') for value in ('seq','scene_event_id','source_stamp','ekspozycj','obserwacj'))
    ui.select('a3:mt:'+calls[0]);ui.navigate(1)
    native.pump(root,lambda:h.current_invocation==calls[1] and h._image is not None)
    assert ui.selection_kind=='mt' and not s.sidecar_path.exists()
    assert h.progress_label.cget('text')=='Zdjęcia: 0/10     MT: 0/44'


def test_native_single_gt_needs_no_choice_and_saved_decision_is_not_repeated(native_gui,ref,monkeypatch):
    import tkinter as tk
    import test_mobile_p2b2_canvas_gui as native
    root,h,ed,s=native_gui;ui=h._f5a_a3
    for gid,g in s.mt_invocations.items():
        fid=ref.call_frame(s,gid);plates=ref.plates(ref.frames[fid]['photo_sha256'])
        rows=ui.detections(gid)
        if len(plates)==1 and len(rows)==1 and g.executed and not g.cancelled and not g.execution_failed and a3.detection_role(s,gid,rows[0]['id'])=='FINAL':break
    else:raise AssertionError('Single GT/detection fixture required')
    ui.select('a3:mt:'+gid);native.pump(root,lambda:h.current_invocation==gid and h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    ui.confirm_identity();root.update();rev=s.review.review_revision
    monkeypatch.setattr(tk.Menu,'tk_popup',lambda *args:pytest.fail('No choice is needed for one GT'))
    ui.primary();root.update()
    assert a3.state(s)['detections'][rows[0]['id']]['outcome']=='TP' and s.review.review_revision==rev+1
    saved=s.sidecar_path.read_bytes();ui.primary();root.update();assert s.sidecar_path.read_bytes()==saved


def test_native_simple_layout_keeps_canvas_largest_and_details_collapsed(native_gui,ref):
    import test_mobile_p2b2_canvas_gui as native
    from PIL import ImageGrab
    from z4_tk_visibility_support import foreground_test_window,assert_native_widgets_visible
    root,h,ed,s=native_gui;ui=h._f5a_a3;gid,fid,plates,aids=populated_call(ref,s)
    ui.select('a3:mt:'+gid);native.pump(root,lambda:h.current_invocation==gid and h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    for size in ('1649x896','1180x760'):
        h.window.state('normal');h.window.geometry(size+'+10+10');h.window.lift();root.update();ed.request_overlay();root.update()
        assert h.canvas.winfo_width()>ui.reference_label.master.winfo_width()
        assert h.canvas.winfo_width()>h.review_sidebar_scroll.winfo_width()
        assert not ed.contexts.sections['session'].winfo_ismapped() and not ed.contexts.object_details.winfo_ismapped()
        assert not ed.source_hud.winfo_ismapped() and ui.more.winfo_ismapped()
        assert 'Nie można odczytać źródła wywołań' not in h.status_var.get()
        assert not ed.navigator.layer.winfo_ismapped() and not ed.navigator.hint.winfo_ismapped()
        assert ed.clean_gtr_confirm_button.winfo_ismapped()
        path=os.environ.get('ALPR_A3_UX_SCREENSHOTS')
        if path:
            with foreground_test_window(h.window,focus=h.canvas):
                assert_native_widgets_visible(root,h.window,[h.canvas,ui.more,ed.clean_gtr_confirm_button])
                ImageGrab.grab(window=h.window.winfo_id()).save(Path(path)/(size+'.png'))
    assert not s.sidecar_path.exists()
