"""F5A reviews plates, not vehicles; source ROI is immutable evidence.

Real-archive native tests require ALPR_F5A_VARIANT_FIXTURES (a JSON mapping of
R0/R1/R2 to private archive paths). All decisions are private test fixtures,
not accepted scientific annotations. No inference or production writes.
"""
import copy
import json
import os
import re
import shutil
from pathlib import Path

import pytest
import test_mobile_p2b2_canvas_gui as native
from test_f5a_manual_review import draw, hidden, sha
from test_f5a12c_scene_evidence import private_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_scene_frame_review as scenes
from auto_annotation_tool.ranking.mobile_mt_object_evaluation import evaluate_unit


def actions(widget):
    if widget.winfo_class() in {'Button','TButton','Checkbutton','TCheckbutton','Radiobutton','TRadiobutton','Menubutton'}:
        if widget.winfo_ismapped():yield str(widget.cget('text'))
    for child in widget.winfo_children():yield from actions(child)


def menu_labels(menu):
    for i in range((menu.index('end') or 0)+1):
        if menu.type(i)!='separator':yield str(menu.entrycget(i,'label'))


@pytest.mark.parametrize('variant,max_rois',[('R0',0),('R1',1),('R2',2)])
def test_native_real_variants_have_no_mp_actions_roi_is_readonly_and_reviews_resume(tmp_path,monkeypatch,variant,max_rois):
    fixture=os.environ.get('ALPR_F5A_VARIANT_FIXTURES')
    if not fixture:pytest.skip('Read-only R0/R1/R2 fixture inventory required')
    item=json.loads(Path(fixture).read_text(encoding='utf-8'))[variant]
    original=Path(item['archive']);before_sha=sha(original)
    source=tmp_path/'source.alprsession';shutil.copyfile(original,source)
    session=MobileReviewSession.open(source,sidecar_path=tmp_path/'review.json')
    assert session.session_id==item['session_id']
    assert session.bundle.report.experiment_index['experiment_variant']==variant
    observations=[m for frame in session.scene_frames.values() for m in frame['mp_observations']]
    assert max(len(m['selected_rois']) for m in observations)==max_rois
    if variant=='R0':assert all(m['execution_status']=='NOT_RUN' for m in observations)
    source_state=copy.deepcopy((session.attempts,session.mt_detection_roles,session.scene_frames))
    monkeypatch.setattr(native,'source_session',lambda _:session)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        assert ed.manual_gt_first and not getattr(h,'_f5a_a3',None)
        gid=next(g for g,u in s.mt_invocations.items() if u.executed and not u.cancelled
            and not u.execution_failed and core.context(s,g))
        aid=s.mt_invocations[gid].records[0]['id'];h._multi_focus_attempt(aid);h._select_record()
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==s.attempts[aid]['source_frame_entry'])
        hidden(root,h,ed)
        labels=[*actions(h.window),*menu_labels(ed.clean_menu)]
        assert not any(re.search(r'\bMP\b|pojazd|vehicle',text,re.I) for text in labels)
        assert not any(re.search(r'(zatwierd|popraw|edyt).*ROI|ROI.*(zatwierd|popraw|edyt)',text,re.I) for text in labels)
        roi=list(core.context(s,gid)['roi']);ed.show_roi_overlay=True;ed.escape();ready(root,h,ed)
        if variant=='R0':
            local=next(link['roi_local_xy'] for frame in s.tv_bbox_evidence['rows']
                for link in frame['attempt_links'] if link['attempt_id']==aid)
            descriptor=s.tv_bbox_evidence['descriptor']
            assert [local[k] for k in ('left','top','right','bottom')]==[0,0,descriptor['input_width_px'],descriptor['input_height_px']]
        items=h.canvas.find_withtag('analyzed_roi');assert len(items)==1
        assert h.canvas.coords(items[0])==pytest.approx([*ed.to_canvas(roi[:2]),*ed.to_canvas(roi[2:])])
        # An ROI corner is not an editable GT handle. Dragging there pans only.
        before=s.review.to_dict();x,y=map(round,ed.to_canvas(roi[:2]))
        h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
        h.canvas.event_generate('<B1-Motion>',x=x+8,y=y+5)
        h.canvas.event_generate('<ButtonRelease-1>',x=x+8,y=y+5);ready(root,h,ed)
        assert s.review.to_dict()==before and list(core.context(s,gid)['roi'])==roi
        # Geometry outside this MT input cannot be inserted as its missed GT.
        with pytest.raises(ValueError):core.add_plate(s,gid,[(1,1),(5,1),(5,5),(1,5)],partial_review=True)
        assert s.review.to_dict()==before
        # Private UI mechanics only: independently draw in this exact frame.
        l,t,r,b=roi;w=r-l;hh=b-t
        pid=draw(h,ed,[(l+.2*w,t+.2*hh),(l+.4*w,t+.2*hh),(l+.4*w,t+.4*hh),(l+.2*w,t+.4*hh)])
        ed.entry.delete(0,'end');ed.entry.insert(0,'SYNF5A')
        ed.entry.event_generate('<KeyRelease-a>')
        native.pump(root,lambda:stacks.snapshot(s,gid)['plates'][pid]['gt_text']=='SYNF5A')
        if ed.entry is not None:ed.entry.event_generate('<Return>')
        hidden(root,h,ed)
        frozen=s.sidecar_path.read_bytes();ed.clean_menu.invoke(1);ready(root,h,ed)
        assert ed.inf_visible.get() and s.sidecar_path.read_bytes()==frozen
        assert (s.attempts,s.mt_detection_roles,s.scene_frames)==source_state
        assert not s.review.scene_annotations  # no MP, ROI or extra scene approval
        saved=copy.deepcopy(s.review.to_dict())
    finally:
        try:next(gen)
        except StopIteration:pass
    reopened=MobileReviewSession.open(source,sidecar_path=session.sidecar_path)
    # Check all persisted review fields/history/identity. Existing source
    # metadata text in provenance is refreshed by the importer on opening.
    assert {k:v for k,v in reopened.review.to_dict().items() if k!='provenance'}=={k:v for k,v in saved.items() if k!='provenance'}
    out=tmp_path/'export';out.mkdir();reopened.export(out)
    assert json.loads((out/'review.json').read_text(encoding='utf-8'))['invocation_annotations']==saved['invocation_annotations']
    exported=json.loads((out/'scene_frame_assessment.json').read_text(encoding='utf-8'))
    assert exported['source_scene_records']==list(source_state[2].values())
    assert exported['source_mt_roles']==list(source_state[1].values())
    assert exported['scene_annotations']=={} and reopened.sidecar_path.read_bytes()==frozen
    assert sha(source)==sha(original)==before_sha


def synthetic_roi_observation(s,key):
    """Unit-test input only: older producer fixtures omit MP observations.

    Do not repair their archives or treat this as recovered real evidence.
    Actual R0/R1/R2 source preservation is tested separately above.
    """
    frame=s.scene_frames[key]
    rois=[dict(zip(('left','top','right','bottom'),r['roi']))
          for r in s.mt_detection_roles.values() if r['source_stamp']==frame['source_stamp']]
    frame['mp_observations']=[dict(source_stamp=copy.deepcopy(frame['source_stamp']),
        execution_status='EXECUTED',selected_rois=rois)]


@pytest.mark.parametrize('fixture', ['one-roi','two-roi'])
def test_exact_frame_plate_coverage_is_automatic_not_a_vehicle_review(tmp_path,fixture):
    s=private_session(tmp_path,fixture);key=next(iter(s.scene_frames))
    synthetic_roi_observation(s,key)
    source=copy.deepcopy((s.mt_detection_roles,s.scene_frames));archive_sha=sha(s.path)
    left=scenes.add_plate(s,key,[(15,30),(35,30),(35,50),(15,50)])
    right=scenes.add_plate(s,key,[(115,30),(135,30),(135,50),(115,50)])
    # These synthetic source detections are explicitly rejected, not inferred FP.
    for role in s.mt_detection_roles.values():
        for aid in role['final_attempt_ids']:scenes.decide_mt(s,key,aid,is_plate=False)
    frozen=s.sidecar_path.read_bytes();cover=scenes.coverage(s,key)
    assert [r['coverage'] for r in cover['rows']]==(['COVERED','OUTSIDE_ROI'] if fixture=='one-roi' else ['COVERED','COVERED'])
    assert s.sidecar_path.read_bytes()==frozen  # coverage needs no extra acceptance
    for gid in s.mt_invocations:
        result=evaluate_unit(s,gid)
        assert result['n_qualified']==1 and result['fn']==1 and result['n_gt']==1
        assert len(result['excluded_gt_ids'])==1  # other ROI is never this MT's FN
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    synthetic_roi_observation(reopened,key)
    assert scenes.coverage(reopened,key)==cover
    out=tmp_path/'export';out.mkdir();reopened.export(out)
    exported=json.loads((out/'scene_frame_assessment.json').read_text(encoding='utf-8'))
    assert exported['scenes'][0]['rows']==cover['rows']
    assert cover['mt_fn'] is None and cover['tv_e2e_status']=='N/A'
    assert (s.mt_detection_roles,s.scene_frames)==source and sha(s.path)==archive_sha


def test_missing_matching_stamp_and_partial_roi_stay_na(tmp_path):
    s=private_session(tmp_path,'one-roi');key=next(iter(s.scene_frames));gid=next(iter(s.mt_invocations))
    synthetic_roi_observation(s,key)
    oid=scenes.add_plate(s,key,[(72,30),(88,30),(88,50),(72,50)])
    for aid in s.mt_detection_roles[gid]['final_attempt_ids']:scenes.decide_mt(s,key,aid,is_plate=False)
    assert scenes.coverage(s,key)['rows'][0]['coverage']=='PARTIAL_ROI'
    result=evaluate_unit(s,gid)
    assert result['fn'] is None and 'GT_PARTIAL_ROI_COVERAGE' in result['unavailable_reasons']
    # In-memory negative evidence fixture; never modify any archived JSON.
    for m in s.scene_frames[key]['mp_observations']:m['source_stamp']['source_sequence']+=1
    assert scenes.coverage(s,key)['status']=='N/A'
    assert scenes.coverage(s,key)['reason']=='ROI_COVERAGE_UNPROVEN'
    s.mt_detection_roles[gid]['source_stamp']['source_sequence']+=1
    with pytest.raises(ValueError,match='exact scene bitmap'):
        scenes.unit_for_mt(s,gid)  # stale decisions cannot transport GT to another frame


def test_zero_rois_without_mt_is_alpr_coverage_loss_not_mt_fn(tmp_path):
    s=private_session(tmp_path,'zero-roi-with-plate');key=next(iter(s.scene_frames))
    scenes.add_plate(s,key,[(10,30),(40,30),(40,50),(10,50)])
    cover=scenes.coverage(s,key)
    assert cover['n_outside']==1 and cover['n_covered']==0 and cover['mt_fn'] is None
    assert not s.mt_invocations and cover['tv_e2e_status']=='N/A'
    assert all('vehicle' not in obj and 'mp_decision' not in obj for obj in scenes.scene(s,key)['objects'])
