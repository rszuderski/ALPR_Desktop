"""AKT89G: archival inspection never changes review/metric eligibility."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from test_mobile_p2b2_canvas_review import source_session,MANUAL
import test_mobile_p2b2_canvas_gui as native
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_mt_history import history_evidence,history_assessment
from auto_annotation_tool.ranking.mobile_canvas_presentation import execution_state,focused_assessment
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.gui.mobile_compare_review_gate import inspect_canvas


def historical(tmp_path,*,n=1,trace=None,new_sidecar=False):
    s=source_session(tmp_path,n=n,mz=False)
    if trace is not None:
        with zipfile.ZipFile(s.path,'a') as z:
            z.writestr('telemetry/traces.jsonl','\n'.join(json.dumps(r) for r in (trace if isinstance(trace,list) else [trace])))
    if trace is not None or new_sidecar:
        s=MobileReviewSession.open(s.path,sidecar_path=tmp_path/'never_written.review.json')
    for r in s.attempts.values():r.update(stale_or_cancelled=True,cancel_reason='scene_superseded')
    g=s.mt_invocations['mt-one']
    s.mt_invocations['mt-one']=replace(g,cancelled=True,records=tuple(s.attempts.values()))
    return s


def trace_complete(**overrides):
    return dict(source_sequence=17,source_timestamp_ns=100000,scene_generation=1,
        visual_epoch=1,camera_transform_generation=1,frame_id=8,status='scene_superseded',
        stage_ms={'plate_inference':701.,'plate_postprocess':2.,'total':900.},
        counters=dict(source_width=1000,source_height=700,mt_runs_this_frame=1,
            plate_detections_raw=1,plate_detections_after_dedup=1,plate_detections_suppressed=0),**overrides)


def viewed(s,**kw):
    return history_evidence(s,'mt-one',image_entry=kw.get('entry','samples/frames/source.jpg'),
                            image_size=kw.get('size',(1000,700)))


def frozen(s):
    return (s.review.to_dict(),s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None,
            hashlib.sha256(s.path.read_bytes()).hexdigest(),core.summary(s))


def test_recorded_cancelled_quad_is_bound_but_completion_is_unknown_and_not_scored(tmp_path):
    s=historical(tmp_path);before=frozen(s);e=viewed(s)
    assert e['view_bound'] and e['detections'][0]['attempt_id']=='a0'
    assert e['list_label']=='Anulowany potok'
    assert 'Pełna klatka' in e['image_status'] and 'zachowany wynik' in e['mt_status']
    assert not e['mt_completion_confirmed'] and not e['final_detection_confirmed']
    assert e['cancellation_stage']=='NIEUSTALONY' and e['cancellation_timestamp'] is None
    assert core.context(s,'mt-one') is None and not e['metric_eligible']
    model=history_assessment(focused_assessment(None,'',execution_state(s,'mt-one')),e,'a0')
    assert model['execution']['state']=='MT_CANCELLED'
    assert model['cer'] is model['polygon_iou_pct'] is model['nmce_pct'] is None
    assert 'attempt_id: a0' in model['history_details'] and 'NOT_RUN' not in model['text']
    assert 'Zakończenie inferencji: NIEUSTALONE' in model['frame']
    assert frozen(s)==before


def test_completed_mt_and_cancelled_pipeline_remain_separate_without_relaxing_gate(tmp_path):
    s=historical(tmp_path,trace=trace_complete());before=frozen(s);e=viewed(s)
    assert e['mt_completion_confirmed'] and e['final_detection_confirmed']
    assert 'zakończona inferencja' in e['mt_status'] and 'Potok: anulowano' in e['pipeline_status']
    assert len(e['detections'])==1 and not e['metric_eligible']
    assert core.context(s,'mt-one') is None
    assert frozen(s)==before


@pytest.mark.parametrize('change',['other_sequence','other_epoch','other_dimensions','duplicate'])
def test_trace_requires_exact_identity_and_uniqueness(tmp_path,change):
    t=trace_complete()
    if change=='other_sequence':t['source_sequence']=18
    elif change=='other_epoch':t['visual_epoch']=2
    elif change=='other_dimensions':t['counters']['source_width']=999
    else:t=[t,copy.deepcopy(t)]
    s=historical(tmp_path,trace=t);e=viewed(s)
    assert not e['mt_completion_confirmed'] and not e['final_detection_confirmed']
    assert len(e['detections'])==1  # local recorded geometry is still historical


@pytest.mark.parametrize('duration',[True,0,-1,float('nan'),None])
def test_invalid_or_missing_inference_time_never_proves_completion(tmp_path,duration):
    t=trace_complete();t['stage_ms']['plate_inference']=duration
    s=historical(tmp_path,trace=t);e=viewed(s)
    assert not e['mt_completion_confirmed'] and not e['final_detection_confirmed']


def test_inference_duration_alone_is_not_completion_proof(tmp_path):
    t=trace_complete();del t['stage_ms']['plate_postprocess']
    s=historical(tmp_path,trace=t);e=viewed(s)
    assert not e['mt_completion_confirmed'] and not e['final_detection_confirmed']


def test_cancelled_after_completed_zero_detection_is_still_not_fn(tmp_path):
    t=trace_complete();t['counters'].update(plate_detections_raw=0,plate_detections_after_dedup=0)
    s=historical(tmp_path,n=0,trace=t);e=viewed(s)
    assert e['mt_completion_confirmed'] and not e['detections']
    assert not e['metric_eligible'] and core.summary(s)['missed_plates']==0


@pytest.mark.parametrize('change',['input_only','missing_images','wrong_view','wrong_size','wrong_invocation','bad_roi'])
def test_missing_or_incompatible_source_never_borrows_a_frame(tmp_path,change):
    s=historical(tmp_path);before=frozen(s);kw={}
    if change=='input_only':s.entry_names.remove('samples/frames/source.jpg')
    elif change=='missing_images':s.entry_names=set()
    elif change=='wrong_view':kw['entry']='samples/frames/other.jpg'
    elif change=='wrong_size':kw['size']=(512,512)
    elif change=='wrong_invocation':s.attempts['a0']['mt_invocation_id']='other'
    else:s.attempts['a0']['roi_right']=1001
    e=viewed(s,**kw)
    assert not e['view_bound'] and not e['detections'] and not e['metric_eligible']
    if change=='input_only':assert e['list_label']=='Tylko podgląd ROI'
    if change=='missing_images':assert e['list_label']=='Brak zdjęcia'
    assert s.review.to_dict()==before[0]
    assert (s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None)==before[1]


@pytest.mark.parametrize('kind',['partial_raw','missing_quad','no_detection'])
def test_partial_and_absent_detection_are_not_final_or_quality_results(tmp_path,kind):
    s=historical(tmp_path,n=0 if kind=='no_detection' else 1,trace=trace_complete())
    if kind=='partial_raw':
        s.attempts['a0']['mt_status']='RAW'
    elif kind=='missing_quad':s.attempts['a0']['mt_pose_corners_norm']=''
    e=viewed(s)
    assert not e['mt_completion_confirmed'] and not e['final_detection_confirmed'] and not e['detections'] and not e['metric_eligible']
    d=core.summary(s)
    assert d['missed_plates']==d['linked_plates']==d['false_positive_mt']==d['geometry_n']==d['evaluated_reads']==0


def test_zero_completed_mt_with_manual_reference_stays_fn(tmp_path):
    s=source_session(tmp_path,n=0,mz=False)
    core.add_plate(s,'mt-one',MANUAL);core.check_scene(s,'mt-one')
    assert history_evidence(s,'mt-one') is None
    d=core.summary(s)
    assert d['missed_plates']==1 and d['expected_plates']==1
    assert d['geometry_n']==d['evaluated_reads']==0 and d['cer'] is None


def test_opening_history_creates_no_sidecar_or_revision(tmp_path):
    s=historical(tmp_path,new_sidecar=True);before=frozen(s)
    assert not s.sidecar_path.exists()
    for _ in range(3):viewed(s)
    assert frozen(s)==before and not s.sidecar_path.exists()


@pytest.fixture
def history_gui(tmp_path,monkeypatch):
    s=historical(tmp_path,new_sidecar=True)
    before=frozen(s)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(gen)
    assert frozen(s)==before and not s.sidecar_path.exists()
    try:
        yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass


def test_native_history_click_never_enters_edit_or_metrics(history_gui):
    root,h,ed,s=history_gui;before=frozen(s)
    assert ed.approve.cget('state')=='disabled'
    assert h.complete_button.cget('state')=='disabled' and not s.sidecar_path.exists()
    assert h.form_status_label.cget('text')=='TYLKO HISTORIA'
    assert ed._read() is None and h.canvas.find_withtag('mt_history_frame')
    x,y=ed._history_to_canvas((225,175))
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y))
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));native.pump(root)
    assert ed.history_selected=='a0' and not ed.selected and ed.mode=='select'
    assert not h.canvas.find_withtag('corner_handle')
    assert ed.assessment.history.opened
    assert 'a0' in ed.assessment.model['history_details']
    for key in ('s','a','t','n'):
        native.key(h,key,xy=(x,y));native.pump(root)
    ed.start_entry('GT:a0');ed.approve.invoke();ed.undo();ed.undo(redo=True);h.complete()
    assert ed.entry is None and not ed.selected and frozen(s)==before
    native.key(h,'v');native.pump(root,lambda:h._image is not None and h._image_entry=='samples/evidence/a0.jpg')
    assert not getattr(ed,'history_selected','')
    assert not h.canvas.find_withtag('mt_history_frame')
    native.key(h,'v');native.pump(root,lambda:h._image is not None and h._image_entry=='samples/frames/source.jpg')
    assert h.canvas.find_withtag('mt_history_frame') and frozen(s)==before


def test_native_roi_only_has_preview_label_and_no_edit(tmp_path,monkeypatch):
    s=historical(tmp_path,new_sidecar=True)
    s.entry_names.remove('samples/frames/source.jpg')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        before=frozen(s)
        assert h._image is not None and h._image_entry=='samples/evidence/a0.jpg'
        assert 'Tylko podgląd ROI' in h.subject_tree.item(h.subject_tree.get_children('')[0],'text')
        assert ed.approve.cget('state')=='disabled' and not h.canvas.find_withtag('mt_history_frame')
        ed.start_entry('GT:a0');ed.approve.invoke()
        assert ed.entry is None and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass


def test_native_navigation_clears_history_without_touching_valid_editor(tmp_path,monkeypatch):
    from test_mobile_p2b2_fix01_canvas_ux import two_scene_session
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    s=two_scene_session(tmp_path)
    s.attempts['t0'].update(stale_or_cancelled=True,cancel_reason='scene_superseded')
    s.mt_invocations['g0']=replace(s.mt_invocations['g0'],cancelled=True,records=(s.attempts['t0'],))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        before=frozen(s)
        def navigate(gid):
            iid=next(i for i in h.subject_tree.get_children('') if gid in subject_groups(s,h._subject_ids[i]))
            h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
            native.pump(root,lambda:h.current_invocation==gid and h._image is not None and
                        h._image_entry==f'samples/frames/source{gid[-1]}.jpg')
        navigate('g0');x,y=ed._history_to_canvas((225,175))
        h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y));h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y))
        native.pump(root);assert ed.history_selected=='t0'
        navigate('g1')
        assert ed._read() is not None and not ed.history_selected
        assert ed.assessment.model['execution']['state']=='MT_EXECUTED'
        assert not ed.assessment.history.winfo_ismapped() and not h.canvas.find_withtag('mt_history_frame')
        assert ed.approve.cget('state')=='normal'
        navigate('g0')
        assert not ed.history_selected and ed.selected=='' and not ed.assessment.history.winfo_ismapped()
        assert frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass


REAL_ROOT=Path.home()/'Desktop/ALPR_EKSPERYMENTY/E-MOB-QINT8-01/02_sesje/DAY_FP32/PAIR_DAY_01'


def real_session(repeat):
    sources=list((REAL_ROOT/repeat).glob('*.alprsession'))
    if len(sources)!=1:pytest.skip('Operator source archive not available')
    return MobileReviewSession.open(sources[0])


def test_real_r03_exact_image02_evidence_is_read_only():
    s=real_session('r03');before=frozen(s);gid='exp-1792702342177-89128c1a-a00000002'
    e=history_evidence(s,gid,image_entry='samples/frames/'+gid+'.jpg',image_size=(960,1280))
    assert e['view_bound'] and e['trace']['frame_id']==2608
    assert e['trace']['source_sequence']==81932 and e['trace']['source_timestamp_ns']==552161388860000
    assert e['detections'][0]['attempt_id']==gid and e['detections'][0]['detection_index']==0
    assert len(e['images'])==2 and not e['mt_completion_confirmed'] and not e['final_detection_confirmed']
    assert e['trace']['status']=='scene_superseded' and e['mz_status']=='MZ: nie uruchomiono'
    assert frozen(s)==before


@pytest.mark.parametrize('repeat,revision,errors,exact',[('r01',247,4,15),('r02',181,2,16)])
def test_real_approved_canvas_quality_and_gate_contract_remain_unchanged(repeat,revision,errors,exact):
    s=real_session(repeat);before=frozen(s);d=core.summary(s)
    assert s.review.review_revision==revision and s.review.review_status=='COMPLETED'
    assert (d['linked_plates'],d['expected_plates'],d['missed_plates'],d['geometry_n'])==(18,19,1,18)
    assert (d['text_errors'],d['gt_characters'],d['exact_reads'],d['evaluated_reads'])==(errors,127,exact,18)
    assert inspect_canvas(d,s.review.review_status)[0]
    for gid in s.mt_invocations:history_evidence(s,gid)
    assert frozen(s)==before
