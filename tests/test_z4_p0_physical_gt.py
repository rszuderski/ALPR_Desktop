"""Physical identity, final detections and raw evidence have separate counts."""
import copy,csv,io,json,os,zipfile
from pathlib import Path
from dataclasses import replace
import pytest
from PIL import Image
from test_mobile_p2b2_canvas_review import source_session,text_all
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_canvas_physical_gt import mark_duplicate,share_across_invocations,declare_distinct,KEY
from auto_annotation_tool.ranking.mobile_mt_postdedup import canvas_post_nms_proof
from auto_annotation_tool.gui.mobile_compare_review_gate import inspect_canvas


def csv_bytes(rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader();writer.writerows(rows);return out.getvalue().encode()


def session_fixture(tmp_path,*,n=3,suppressed=1,two_calls=False,real=None):
    source=source_session(tmp_path/'template',n=n)
    target=tmp_path/'case.alprsession'
    with zipfile.ZipFile(source.path) as src:
        payload={name:src.read(name) for name in src.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode())))
    samples=list(csv.DictReader(io.StringIO(payload['samples/index.csv'].decode())))
    if real:
        data=json.loads((real/'two_real_plates_gt.json').read_text(encoding='utf-8'))
        frame=(real/'two_real_plates.jpg').read_bytes()
        with Image.open(io.BytesIO(frame)) as image:w,h=image.size
        payload['samples/frames/source.jpg']=frame
    for i,row in enumerate(rows):
        points=[tuple(map(float,p.split(':'))) for p in row['mt_pose_corners_norm'].split(';')]
        if real:
            points=data[i]['geometry']['points']
            row.update(source_frame_width_px=w,source_frame_height_px=h,mt_pose_source_width_px=w,mt_pose_source_height_px=h,
                       roi_left=0,roi_top=0,roi_right=w,roi_bottom=h,mt_pose_corners_norm=';'.join(f'{x}:{y}' for x,y in points))
        else:w,h=1000,700
        row.update(plate_left=min(p[0] for p in points)*w,plate_top=min(p[1] for p in points)*h,
                   plate_right=max(p[0] for p in points)*w,plate_bottom=max(p[1] for p in points)*h,
                   mz_executed='false' if i==suppressed else 'true',rectification_status='NOT_RUN' if i==suppressed else 'OK')
        if i==suppressed:
            row.update(mz_status='NOT_RUN',prediction='')
        if two_calls:
            row.update(mt_invocation_id=f'g{i}',mt_detection_index=0,mt_detection_count=1,source_sequence=17+i,source_timestamp_nanos=100000+i)
    samples=[r for r in samples if r['attempt_id']!=f'a{suppressed}']
    payload['samples/attempts.csv']=csv_bytes(rows);payload['samples/index.csv']=csv_bytes(samples)
    traces=[]
    for i in range(n if two_calls else 1):
        raw=1 if two_calls else n;reject=int(suppressed is not None and not two_calls)
        traces.append(dict(frame_id=7+i,source_sequence=17+i,source_timestamp_ns=100000+i,scene_generation=1,
            visual_epoch=1,camera_transform_generation=1,counters=dict(plate_detections_raw=raw,
            plate_detections_after_dedup=raw-reject,plate_detections_suppressed=reject,mt_runs_this_frame=1,mz_runs=raw-reject)))
    payload['telemetry/traces.jsonl']='\n'.join(json.dumps(t) for t in traces).encode()
    with zipfile.ZipFile(target,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    s=load(target,tmp_path);s.set_review_mode('assisted');return s


def prepare(s,gid='mt-one'):
    for o in core.scene(s,gid)['objects']:
        if o.get('post_nms_suppressed') or o['error'] or o.get('duplicate_of'):continue
        parent=next((p for p in core.scene(s,gid)['parents'] if p['parent_id']==o['parent_id']),None)
        if parent and parent['mz_ran']:core.set_text(s,gid,o['object_id'],parent['mz_prediction'] or 'ABC123')
        if parent:partial.confirm_stack(s,gid,oid=o['object_id'])
    # Per-frame decisions are the completion action; checking the whole image
    # is no longer part of this fixture's operator workflow.


def test_three_raw_two_final_has_one_gt_one_fp_and_one_suppressed(tmp_path):
    s=session_fixture(tmp_path);proof=canvas_post_nms_proof(s,'mt-one')
    assert proof['suppressed_id']=='a1' and proof['final_ids']==['a0','a2']
    scene=core.scene(s,'mt-one');assert scene['objects'][1]['post_nms_suppressed']
    core.toggle_error(s,'mt-one','GT:a2');prepare(s);d=core.summary(s)
    assert (d['expected_plates'],d['linked_plates'],d['false_positive_mt'],d['missed_plates'],d['raw_suppressed_mt'])==(1,1,1,0,1)
    assert len(d['physical_plates'])==1 and d['rows'][1]['physical_gt_id'] is None
    assert inspect_canvas(d,'COMPLETED')[0]
    source=core.scene(s,'mt-one')['parents'];reopened=load(s.path,tmp_path)
    assert core.summary(reopened)==d and core.scene(reopened,'mt-one')['parents']==source
    exported=core.export(s,tmp_path/'export')
    assert json.loads(exported[0].read_text(encoding='utf-8'))==d
    with (tmp_path/'export/canvas_physical_plates.csv').open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==1


@pytest.mark.parametrize('damage',['no_trace','ambiguous_trace','two_calls','mz_missing','sample_missing','epoch'])
def test_not_run_or_overlap_alone_cannot_prove_suppression(tmp_path,monkeypatch,damage):
    from auto_annotation_tool.ranking import mobile_mt_postdedup as module
    s=session_fixture(tmp_path);assert canvas_post_nms_proof(s,'mt-one')
    original=module._trace_index
    def traces(*args):
        value=copy.deepcopy(original(*args));key=next(iter(value));t=value[key][0]
        if damage=='no_trace':return {}
        if damage=='ambiguous_trace':value[key].append(copy.deepcopy(t))
        if damage=='two_calls':t['counters']['mt_runs_this_frame']=2
        if damage=='epoch':t['visual_epoch']=99
        return value
    monkeypatch.setattr(module,'_trace_index',traces)
    if damage=='mz_missing':s.attempts['a2']['mz_executed']='false'
    if damage=='sample_missing':s.sample_ids_by_attempt['a2']=()
    before=copy.deepcopy(s.review.to_dict())
    assert canvas_post_nms_proof(s,'mt-one') is None
    assert not any(o.get('post_nms_suppressed') for o in core.scene(s,'mt-one')['objects'])
    assert s.review.to_dict()==before


def test_missing_final_proof_blocks_new_research_review_without_changing_source(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking import mobile_mt_postdedup as module
    s=session_fixture(tmp_path)
    s.entry_names=set(s.entry_names)|{'session.json'}  # full research archive contract
    monkeypatch.setattr(module,'_trace_index',lambda *a:{})
    source=s.path.read_bytes()
    core.set_scene_review_state(s,'mt-one',checked=True)
    assert any('śladu' in issue for issue in core.group_issues(s,'mt-one'))
    assert core.summary(s)['checked_scenes']==core.summary(s)['expected_plates']==0
    assert s.path.read_bytes()==source


def test_two_final_duplicates_one_gt_one_tp_one_fp_no_duplicate_metrics(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None)
    before=core.scene(s,'mt-one')
    mark_duplicate(s,'mt-one','GT:a1','GT:a0');prepare(s);d=core.summary(s)
    assert core.scene(s,'mt-one')['parents']==before['parents']
    assert (d['physical_gt_scene_n'],d['expected_plates'],d['linked_plates'],d['false_positive_mt'],d['geometry_n'],d['evaluated_reads'])==(1,1,1,1,1,1)
    duplicate=d['rows'][1];assert duplicate['false_positive_subtype']=='duplicate_final'
    assert duplicate['physical_gt_id']==d['rows'][0]['physical_gt_id']
    assert duplicate['cer'] is duplicate['nmce_pct'] is duplicate['polygon_iou_pct'] is None
    assert inspect_canvas(d,'COMPLETED')[0]


def test_fp_toggle_never_deletes_confirmed_physical_gt(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None)
    core.toggle_error(s,'mt-one','GT:a1');prepare(s)
    initial=copy.deepcopy(core.scene(s,'mt-one')[KEY]);assert len(initial['plates'])==1
    core.toggle_error(s,'mt-one','GT:a0');core.check_scene(s,'mt-one');d=core.summary(s)
    assert (d['expected_plates'],d['missed_plates'],d['false_positive_mt'])==(1,1,2)
    current=core.scene(s,'mt-one')[KEY]['plates']
    assert [{k:p[k] for k in ('gt_id','quad','scene_identity')} for p in current]==[{k:p[k] for k in ('gt_id','quad','scene_identity')} for p in initial['plates']]
    # The same GT now has an independent FN reference; it cannot point at the FP.
    assert inspect_canvas(d,'COMPLETED')[0]
    assert d['geometry_n']==d['evaluated_reads']==0


def test_three_final_one_duplicate_and_one_nonplate_fp_have_only_one_gt(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=None)
    core.toggle_error(s,'mt-one','GT:a2')
    mark_duplicate(s,'mt-one','GT:a1','GT:a0');prepare(s);d=core.summary(s)
    assert (d['expected_plates'],d['linked_plates'],d['false_positive_mt'],d['missed_plates'],d['raw_suppressed_mt'])==(1,1,2,0,0)
    assert d['geometry_n']==d['evaluated_reads']==d['physical_gt_scene_n']==1
    assert inspect_canvas(d,'COMPLETED')[0]


def test_two_calls_one_physical_gt_keep_two_stage_results(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None,two_calls=True)
    prepare(s,'g0');prepare(s,'g1')
    share_across_invocations(s,'g1','GT:a1','g0','GT:a0');core.check_scene(s,'g1');d=core.summary(s)
    assert (d['physical_gt_scene_n'],d['expected_plates'],d['linked_plates'],d['false_positive_mt'])==(1,2,2,0)
    assert d['geometry_n']==d['evaluated_reads']==2
    assert d['rows'][0]['physical_gt_id']==d['rows'][1]['physical_gt_id']
    assert inspect_canvas(d,'COMPLETED')[0]
    assert core.summary(load(s.path,tmp_path))==d


def test_two_calls_never_claim_two_scene_gt_without_an_identity_decision(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None,two_calls=True)
    prepare(s,'g0');prepare(s,'g1');d=core.summary(s)
    assert d['physical_gt_scene_n'] is None and d['false_positive_mt']==0
    assert core.group_issues(s,'g1') and not inspect_canvas(d,'COMPLETED')[0]
    declare_distinct(s,'g1','GT:a1');d=core.summary(s)
    assert (d['physical_gt_scene_n'],d['expected_plates'],d['linked_plates'],d['false_positive_mt'])==(2,2,2,0)
    assert inspect_canvas(d,'COMPLETED')[0]


def test_unknown_physical_identity_state_cannot_become_ready(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None);prepare(s);d=core.summary(s)
    d['scenes'][0]['physical_gt']['plates'][0]['scene_identity']='UNKNOWN'
    assert not inspect_canvas(d,'COMPLETED')[0]


def test_raw_cannot_be_called_a_duplicate_and_other_scene_cannot_share_gt(tmp_path):
    s=session_fixture(tmp_path/'raw')
    before=copy.deepcopy(s.review.to_dict())
    with pytest.raises(ValueError):mark_duplicate(s,'mt-one','GT:a1','GT:a0')
    assert s.review.to_dict()==before
    s=session_fixture(tmp_path/'calls',n=2,suppressed=None,two_calls=True)
    prepare(s,'g0')
    s.attempts['a1']['scene_generation']='other-scene'
    before=copy.deepcopy(s.review.to_dict())
    with pytest.raises(ValueError):share_across_invocations(s,'g1','GT:a1','g0','GT:a0')
    assert s.review.to_dict()==before


def test_two_distinct_close_plates_are_not_auto_merged(tmp_path):
    s=session_fixture(tmp_path,n=2,suppressed=None);prepare(s);d=core.summary(s)
    assert d['expected_plates']==d['physical_gt_scene_n']==d['linked_plates']==2
    assert len({r['physical_gt_id'] for r in d['rows']})==2 and d['false_positive_mt']==0


def test_real_photo_two_gt_remain_distinct(tmp_path):
    path=os.environ.get('P0_TWO_PLATES_FIXTURE')
    if not path:pytest.skip('Real photo acceptance is enabled with P0_TWO_PLATES_FIXTURE')
    s=session_fixture(tmp_path,n=2,suppressed=None,real=Path(path));prepare(s);d=core.summary(s)
    assert (d['expected_plates'],d['physical_gt_scene_n'],d['linked_plates'],d['false_positive_mt'])==(2,2,2,0)
    assert len({r['physical_gt_id'] for r in d['rows']})==2
    assert inspect_canvas(d,'COMPLETED')[0]


@pytest.mark.parametrize('damage',['duplicate_gt','double_tp','raw_gt','scene_count'])
def test_export_gate_rejects_tampered_physical_assignments(tmp_path,damage):
    s=session_fixture(tmp_path);core.toggle_error(s,'mt-one','GT:a2');prepare(s);d=core.summary(s)
    assert inspect_canvas(d,'COMPLETED')[0]
    if damage=='duplicate_gt':d['scenes'][0]['physical_gt']['plates']*=2
    elif damage=='double_tp':d['rows'].append(copy.deepcopy(d['rows'][0]))
    elif damage=='raw_gt':d['rows'][1]['physical_gt_id']=d['rows'][0]['physical_gt_id']
    else:d['physical_gt_scene_n']+=1
    assert not inspect_canvas(d,'COMPLETED')[0]


import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_gui import gui,pump
from test_z4_akt89lm_gui import ready


@pytest.fixture
def final_pair_gui(tmp_path,monkeypatch,request):
    monkeypatch.setattr(native,'source_session',lambda p:session_fixture(p,n=2,suppressed=None))
    return request.getfixturevalue('gui')


def test_native_explicit_duplicate_undo_redo_and_selection_preserve_one_gt(final_pair_gui,monkeypatch):
    root,h,ed,s=final_pair_gui
    ed._choose_plate('GT:a1');ready(root,h,ed)
    before=copy.deepcopy(s.review.to_dict())
    before_scene=core.scene(s,'mt-one')
    assert ed.tally_cards['expected'].value.cget('text')=='do ustalenia'
    assert str(ed.decision_panel.physical_same.cget('state'))=='disabled'
    ed.decision_panel.physical_pick.current(0)
    ed.decision_panel.physical_pick.event_generate('<<ComboboxSelected>>')
    assert s.review.to_dict()==before
    import tkinter.messagebox as messagebox
    monkeypatch.setattr(messagebox,'askyesno',lambda *a,**k:pytest.fail('No decision modal'))
    ed.decision_panel.physical_same.invoke();ready(root,h,ed)
    assert core.counts(core.scene(s,'mt-one'))['expected']==1
    assert core.scene(s,'mt-one')['objects'][1]['duplicate_of']=='GT:a0'
    assert 'IoU ' in ed.contexts.data['object']  # candidate diagnostics, not a second final TP
    assert ed.assessment.model['cer'] is None and core.summary(s)['geometry_n']==0
    ed.undo();ready(root,h,ed)
    assert core.scene(s,'mt-one')['objects']==before_scene['objects']
    assert not core.scene(s,'mt-one')[KEY]['plates'] and core.counts(core.scene(s,'mt-one'))['expected']==0
    ed.undo(redo=True);ready(root,h,ed)
    assert core.scene(s,'mt-one')['objects'][1]['duplicate_of']=='GT:a0'
    ed.start_entry('GT:a0');ed.entry_var.set('WA01234');ed.entry_enter();ready(root,h,ed)
    ed.approve.invoke();ready(root,h,ed)
    ids=copy.deepcopy(core.scene(s,'mt-one')[KEY]['plates'])
    frozen=s.review.to_dict()
    for oid in ('GT:a0','GT:a1',''):
        ed._choose_plate(oid);ready(root,h,ed)
        assert ed.tally_cards['expected'].value.cget('text')=='1'
        assert core.scene(s,'mt-one')[KEY]['plates']==ids
    assert s.review.to_dict()==frozen
    assert core.scene(load(s.path,s.sidecar_path.parent),'mt-one')[KEY]['plates']==ids
