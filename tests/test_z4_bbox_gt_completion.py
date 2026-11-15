"""P0: explicit four-corner GT for a proven FINAL BBOX without source Pose."""
from pathlib import Path
import copy, hashlib, json, shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core, mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial, mobile_mt_detection_evaluation as mt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Z4_P0_BBOX_POSE_GT_20261027_001'
SID='exp-1792699303016-c8b05efb';G=SID+'-a00000008';OID='GT:'+G

def private_case(path):
    path.mkdir(parents=True,exist_ok=True)
    row=next(r for r in json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    archive=path/'case.alprsession';shutil.copyfile(row['path'],archive)
    sidecar=path/'private.review.json'
    sidecar.write_text(json.dumps(row['sidecar_payload'],ensure_ascii=False,indent=2),encoding='utf-8')
    return MobileReviewSession.open(archive,sidecar_path=sidecar)

def reference_quad():
    old=ROOT/'output/Z4_GT_EDIT_RESPONSIVENESS_20261026_001/baseline_f4.json'
    row=next(r for r in json.loads(old.read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    scene=row['sidecar_payload']['invocation_annotations'][G][core.KEY]
    return copy.deepcopy(next(o['quad'] for o in scene['objects'] if not o['parent_id']))

@pytest.fixture
def case_gui(tmp_path,monkeypatch):
    s=private_case(tmp_path/'case')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    import tkinter as tk
    original=tk.Tk
    def scaled_root(*args,**kwargs):
        root=original(*args,**kwargs);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',scaled_root)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
        iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving)
        ready(root,h,ed)
        yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def select_bbox(root,h,ed):
    l,t,r,b,_=ed.stack_badges[OID]
    x,y=round((l+r)/2),round((t+b)/2)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y,time=10000)
    h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y,time=10010)
    ready(root,h,ed);assert ed.selected==OID

def draw_reference(root,h,ed,quad=None):
    native.key(h,'d');assert ed.mode=='draw'
    for point in quad or reference_quad():native.click(h,ed,point)
    ready(root,h,ed)

def complete(s,quad=None):
    pid=stacks.snapshot(s,G)['links'][OID]['plate_id']
    stacks.complete_bbox_reference(s,G,OID,quad or reference_quad(),expected_plate_id=pid)
    return pid

def unit_counts(s):return tuple(mt.evaluate_unit(s,G)[k] for k in ('n_gt','tp','fp','fn'))

def test_P01_readonly_baseline_and_storage():
    b=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    assert len(b['sessions'])==12 and b['disk_free_bytes']>2*1024**3
    assert not b['index'] and b['approved_policy_candidate_sha256']==mt.POLICY['approval']['approved_candidate_sha256']
    for row in b['sessions']:
        path=Path(row['sidecar'])
        assert path.exists()==row['sidecar_file']['exists']
        if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sidecar_file']['sha256']

def test_P02_exact_source_frame_trace_and_separate_statuses(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import model_location,verify_frame
    s=private_case(tmp_path);scene=core.scene(s,G);verify_frame(s,G)
    location=model_location(s,G,scene,core._object(scene,OID))
    assert scene['image_size']==[960,1280] and scene['roi']==[0,0,960,1280]
    assert location['kind']=='BBOX' and location['bbox']==[565.01526,594.10974,625.43317,624.31866]
    assert location['image_entry']==scene['image_entry'] and location['mt_invocation_id']==G
    assert location['attempt_id']==G and scene['parents'][0]['quad'] is None
    row=s.attempts[G]
    assert (row['mt_status'],row['mt_pose_status'],row['mz_status'])==('DETECTION_INVALID_QUAD','INVALID_QUAD','NOT_RUN')
    assert not row['mt_pose_corners_norm']

def test_P03_P05_selected_ok_bbox_D_completes_same_gt_once(case_gui):
    root,h,ed,s=case_gui
    before=stacks.snapshot(s,G);pid=before['links'][OID]['plate_id']
    assert len(before['plates'])==1 and before['plates'][pid]['gt_quad'] is None
    assert before['plates'][pid]['gt_text']=='WU1092S'
    revision=s.review.review_revision;parents=copy.deepcopy(core.scene(s,G)['parents'])
    select_bbox(root,h,ed)
    expected=[ed.to_source(*(round(v) for v in ed.to_canvas(point))) for point in reference_quad()]
    expected=core.ordered_quad(expected,ed._read()['image_size'],ed._read()['roi'])
    draw_reference(root,h,ed)
    after=stacks.snapshot(s,G)
    assert len(after['plates'])==1,'D must complete the selected physical GT, not create GT2'
    assert after['links'][OID]['plate_id']==pid
    assert after['plates'][pid]['gt_text']=='WU1092S'
    assert after['plates'][pid]['gt_quad']==expected
    assert s.review.review_revision==revision+1
    assert core.scene(s,G)['parents']==parents

def test_P06_existing_manual_gt_can_be_explicitly_linked_without_duplicate(case_gui):
    from auto_annotation_tool.gui import mobile_canvas_stacks as ui
    root,h,ed,s=case_gui;ed.escape();ready(root,h,ed)
    draw_reference(root,h,ed)
    before=stacks.snapshot(s,G);manual=next(p for p in before['plates'].values() if p['origin_object_id'].startswith('MANUAL:'))
    select_bbox(root,h,ed);menu=ui.context_menu(ed,show=False)
    index=next(i for i in range(menu.index('end')+1) if menu.type(i)=='command' and menu.entrycget(i,'label').startswith('Powiąż z ręczną GTR'))
    menu.invoke(index);ready(root,h,ed)
    after=stacks.snapshot(s,G)
    assert len(after['plates'])==1 and after['links'][OID]['plate_id']==manual['plate_id']
    assert after['plates'][manual['plate_id']]['gt_text']=='WU1092S'
    assert after['candidates'][0]['mt_quad'] is None

def test_P07_unselected_D_is_an_independent_missed_plate(case_gui):
    root,h,ed,s=case_gui
    complete(s);ready(root,h,ed);before=unit_counts(s)
    ed.escape();ready(root,h,ed)
    other=[[350,600],[420,600],[420,620],[350,620]]
    draw_reference(root,h,ed,other)
    after=stacks.snapshot(s,G)
    assert len(after['plates'])==2 and unit_counts(s)==(2,before[1],before[2],before[3]+1)
    assert after['links'][OID]['plate_id']!=next(p['plate_id'] for p in after['plates'].values() if p['origin_object_id'].startswith('MANUAL:'))

def test_P08_P09_bbox_readonly_and_gt_indexed_corner_drag(case_gui):
    root,h,ed,s=case_gui;select_bbox(root,h,ed)
    assert not h.canvas.find_withtag('corner_handle') and ed._current_quad(core._object(ed._read(),OID)) is None
    parents=copy.deepcopy(ed._read()['parents']);row=copy.deepcopy(s.attempts[G])
    draw_reference(root,h,ed)
    assert len(h.canvas.find_withtag('model_location_bbox'))==1
    assert len(h.canvas.find_withtag('corner_handle'))==4 and OID in ed.badges
    obj=core._object(ed._read(),OID);points=copy.deepcopy(obj['quad'])
    x,y=ed.to_canvas(points[0]);xx,yy=ed.to_canvas((points[0][0]-2,points[0][1]+1))
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y))
    h.canvas.event_generate('<B1-Motion>',x=round(xx),y=round(yy));native.pump(root)
    assert not h.canvas.find_withtag('model_location_bbox')
    assert OID in ed.badges and ed.clean_counter_boxes.keys()=={OID}
    h.canvas.event_generate('<ButtonRelease-1>',x=round(xx),y=round(yy));ready(root,h,ed)
    after=core._object(ed._read(),OID)['quad']
    assert after[0]!=points[0] and after[1:]==points[1:]
    assert ed._read()['parents']==parents and s.attempts[G]==row
    ed.escape();ready(root,h,ed)
    assert h.canvas.find_withtag('model_location_bbox')

def test_P10_bbox_iou_independent_continuous_math():
    assert mt.box_iou([0,0,10,10],[5,0,15,10])==pytest.approx(1/3)
    assert mt.box_iou([0,0,10,10],[0,0,5,10])==.5
    q=reference_quad();gt=[min(x for x,y in q),min(y for x,y in q),max(x for x,y in q),max(y for x,y in q)]
    b=[565.01526,594.10974,625.43317,624.31866]
    intersection=(min(b[2],gt[2])-max(b[0],gt[0]))*(min(b[3],gt[3])-max(b[1],gt[1]))
    expected=intersection/((b[2]-b[0])*(b[3]-b[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-intersection)
    assert mt.box_iou(b,gt)==pytest.approx(expected)

def test_P11_good_bbox_can_be_tp_without_pose(tmp_path):
    s=private_case(tmp_path)
    # Explicit synthetic operator reference, not a predicted MT Pose.
    complete(s,[[570,600],[620,600],[620,625],[570,625]])
    assert unit_counts(s)==(1,1,0,0)
    row=stacks.diagnostics(s,[G])['rows'][0]
    assert row['polygon_iou_pct'] is row['nmce_pct'] is None
    assert row['not_measurable_reason']=='MT_POSE_INVALID'

def test_P12_real_historical_reference_is_fp_fn_while_operator_ok_is_preserved(tmp_path):
    s=private_case(tmp_path);complete(s)
    assert unit_counts(s)==(1,0,1,1)
    assert partial.progress(core.scene(s,G))['object_decisions'][OID]==partial.PLATE
    result=mt.evaluate_unit(s,G)
    assert result['final_outcomes'][0]['reason']=='LOCALIZATION_CRITERION_NOT_MET'

@pytest.mark.parametrize('damage',['source','bbox','gt','transform'])
def test_P13_missing_evidence_is_na_not_zero(damage):
    from test_z4_mt_detection_evaluation import unit
    u=unit()
    if damage=='source':u['source_verified']=False
    elif damage=='bbox':u['finals'][0]['bbox']=None
    elif damage=='gt':u['gt'][0]['bbox']=None
    else:u['roi']=[0,0,2000,2000]
    r=mt.score_unit(u)
    assert r['status']=='NOT_EVALUABLE' and r['unavailable_reasons']
    assert r['tp'] is r['fp'] is r['fn'] is r['precision'] is None

def test_P14_P15_not_run_invalid_pose_survive_restart_and_additive_export(tmp_path):
    s=private_case(tmp_path/'data');complete(s)
    saved=s.sidecar_path.read_bytes();reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    paths=mt.export(reopened,tmp_path/'export')
    exported=json.loads(paths[0].read_text(encoding='utf-8'))
    r=next(u for u in exported['units'] if u['mt_invocation_id']==G)
    assert (r['tp'],r['fp'],r['fn'])==(0,1,1) and r['gt_mode']=='ASSISTED'
    diag=stacks.diagnostics(reopened,[G])['rows'][0]
    assert diag['mt_quad'] is None and diag['polygon_iou_pct'] is diag['nmce_pct'] is diag['cer'] is None
    assert not diag['mz_ran'] and diag['mz_status']=='NOT_RUN'
    assert reopened.attempts[G]['mt_pose_status']=='INVALID_QUAD'
    assert s.sidecar_path.read_bytes()==saved

def test_P16_two_final_share_one_gt_raw_stays_diagnostic():
    from test_z4_mt_detection_evaluation import unit
    u=unit(final=2);u['finals'][1]['bbox']=u['finals'][0]['bbox'];u['n_raw_suppressed']=3
    r=mt.score_unit(u)
    assert (r['n_gt'],r['tp'],r['fp'],r['fn'])==(1,1,1,0)
    assert len(r['matched_pairs'])==1 and r['n_raw_suppressed']==3

def test_P17_text_or_proximity_never_merges_second_plate(tmp_path):
    s=private_case(tmp_path);other=[[551,602],[553,602],[553,607],[551,607]]
    manual=core.add_plate(s,G,other,partial_review=True);core.set_text(s,G,manual,'WU1092S')
    before=copy.deepcopy(stacks.snapshot(s,G));pid=before['links'][OID]['plate_id']
    complete(s);after=stacks.snapshot(s,G)
    assert len(after['plates'])==2 and after['links'][OID]['plate_id']==pid
    mpid=before['links'][manual]['plate_id'];assert after['plates'][mpid]==before['plates'][mpid]

def test_P18_fp_D_does_not_silently_link_and_decisions_are_reversible(case_gui):
    root,h,ed,s=case_gui;parents=copy.deepcopy(ed._read()['parents'])
    select_bbox(root,h,ed)
    ed._commit(lambda:stacks.toggle_unassociated(s,G,OID,partial_review=True),'Test FP');ready(root,h,ed)
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.NOT_PLATE
    frozen=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):stacks.complete_bbox_reference(s,G,OID,reference_quad())
    assert s.sidecar_path.read_bytes()==frozen
    draw_reference(root,h,ed)
    assert stacks.snapshot(s,G)['links'][OID]['status']=='UNASSOCIATED_NO_PLATE'
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.NOT_PLATE
    ed._commit(lambda:partial.toggle_plate(s,G,OID),'Test OK');ready(root,h,ed)
    assert partial.progress(ed._read())['object_decisions'][OID]==partial.PLATE
    assert ed._read()['parents']==parents and s.attempts[G]['mt_pose_status']=='INVALID_QUAD'

def test_P19_D_in_entry_is_text_enter_and_escape_keep_their_roles(case_gui):
    root,h,ed,s=case_gui;select_bbox(root,h,ed);draw_reference(root,h,ed)
    ed.start_entry(OID);native.pump(root);entry=ed.entry
    entry.focus_force();entry.selection_clear();entry.icursor('end');revision=s.review.review_revision
    entry.event_generate('<KeyPress-d>');entry.event_generate('<KeyRelease-d>');native.pump(root)
    assert ed.mode=='edit' and not ed.points and ed.entry_var.get().endswith('D')
    assert s.review.review_revision==revision
    entry.event_generate('<KeyPress-Return>');ready(root,h,ed)
    pid=stacks.snapshot(s,G)['links'][OID]['plate_id'];assert stacks.snapshot(s,G)['plates'][pid]['gt_text']=='WU1092SD'
    ed.start_entry(OID);ed.entry_var.set('CANCEL');ed.escape();ready(root,h,ed)
    assert stacks.snapshot(s,G)['plates'][pid]['gt_text']=='WU1092SD'

@pytest.mark.parametrize('n',[1,2,3])
def test_P20_partial_drawing_escape_has_zero_writes(case_gui,n):
    root,h,ed,s=case_gui;select_bbox(root,h,ed);before=s.sidecar_path.read_bytes();revision=s.review.review_revision
    native.key(h,'d')
    for p in reference_quad()[:n]:native.click(h,ed,p)
    assert len(ed.points)==n and not h.canvas.find_withtag('stack_badge')
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.points and ed.mode=='select' and s.review.review_revision==revision
    assert s.sidecar_path.read_bytes()==before and len(stacks.snapshot(s,G)['plates'])==1

def test_P21_persistent_undo_redo_draw_text_link_corner(tmp_path):
    s=private_case(tmp_path);original=copy.deepcopy(stacks.snapshot(s,G));parents=copy.deepcopy(core.scene(s,G)['parents'])
    pid=complete(s);stacks.edit_gt(s,G,OID,text='WU1092SNEW')
    quad=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid]['gt_quad']);quad[0][0]-=1
    core.correct_plate(s,G,OID,quad)
    final=copy.deepcopy(stacks.snapshot(s,G)['plates'])
    s=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    for _ in range(3):stacks.restore_history(s,G)
    assert stacks.snapshot(s,G)['plates']==original['plates']
    for _ in range(3):stacks.restore_history(s,G,redo=True)
    assert stacks.snapshot(s,G)['plates']==final and core.scene(s,G)['parents']==parents
    assert len(final)==1 and final[pid]['gt_quad']==quad

def test_P22_gt_correction_updates_bbox_and_session_rates_only(tmp_path):
    s=private_case(tmp_path);complete(s)
    before=mt.evaluate_session(s);summary=core.summary(s)
    stacks.edit_gt(s,G,OID,quad=[[570,600],[620,600],[620,625],[570,625]])
    after=mt.evaluate_session(s)
    assert (after['tp'],after['fp'],after['fn'])==(before['tp']+1,before['fp']-1,before['fn']-1)
    assert after['precision']>before['precision'] and after['recall']>before['recall'] and after['f1']>before['f1']
    assert core.summary(s)['geometry_n']==summary['geometry_n'] and core.summary(s)['evaluated_reads']==summary['evaluated_reads']
    assert stacks.diagnostics(s,[G])['rows'][0]['polygon_iou_pct'] is None

def test_P23_selection_tab_zoom_scroll_panel_reopen_are_readonly(case_gui):
    root,h,ed,s=case_gui;frozen=s.sidecar_path.read_bytes();revision=s.review.review_revision
    select_bbox(root,h,ed);native.key(h,'Tab')
    for zoom in (1.7,1):h._view_zoom=zoom;h._paint_image();native.pump(root)
    h.canvas.event_generate('<MouseWheel>',delta=120);native.pump(root)
    h.review_sidebar_scroll.yview_moveto(.5);ed.sync();ed.update_report();native.pump(root)
    re=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert re.review.review_revision==revision and s.sidecar_path.read_bytes()==frozen

def test_P24_complete_operator_review_and_na_gt_are_independent(tmp_path):
    s=private_case(tmp_path);before=copy.deepcopy(s.review.invocation_annotations)
    assert partial.progress(core.scene(s,G))['complete'] and mt.evaluate_unit(s,G)['tp'] is None
    assert len(s.mt_invocations)==21
    complete(s)
    assert partial.progress(core.scene(s,G))['complete'] and mt.evaluate_unit(s,G)['status']=='MEASURED'
    assert {k:v for k,v in s.review.invocation_annotations.items() if k!=G}=={k:v for k,v in before.items() if k!=G}

def test_P25_native_two_sizes_and_minimal_context_without_new_widgets(case_gui):
    root,h,ed,s=case_gui
    h.window.maxsize(3000,2400);assert root.winfo_fpixels('1i')==pytest.approx(120,abs=.1)
    tabs=list(h.notebook.tabs()) if hasattr(h,'notebook') else None
    select_bbox(root,h,ed)
    for width,height in ((1649,896),(1180,760)):
        h.window.geometry(f'{width}x{height}+20+20');native.pump(root);ready(root,h,ed)
        assert h.canvas.winfo_width()>300 and h.canvas.winfo_height()>300
        assert 'GTR: brak obrysu' in ed.contexts.data['object'] and 'D:' in ed.contexts.data['object']
        assert OID not in ed.badges and not h.canvas.find_withtag('corner_handle')
    draw_reference(root,h,ed)
    for width,height in ((1649,896),(1180,760)):
        h.window.geometry(f'{width}x{height}+20+20');native.pump(root);ready(root,h,ed)
        assert 'BBOX IoU:' in ed.contexts.data['object'] and 'MT: FP' in ed.contexts.data['object']
        assert OID in ed.badges and h.canvas.find_withtag('model_location_bbox')
        assert [ed.contexts.counters[k].cget('text') for k in ('gt','mt','fp','fn')]==['1','0','1','1']
    if tabs is not None:assert list(h.notebook.tabs())==tabs

def test_P26_ui_export_same_canonical_results_and_reasons(case_gui,tmp_path):
    root,h,ed,s=case_gui
    before=mt.evaluate_unit(s,G);assert 'GT_GEOMETRY_UNCONFIRMED' in before['unavailable_reasons']
    assert all(ed.contexts.counters[k].cget('text')=='—' for k in ('gt','mt','fp','fn'))
    select_bbox(root,h,ed);draw_reference(root,h,ed)
    ui=[ed.contexts.counters[k].cget('text') for k in ('gt','mt','fp','fn')]
    paths=mt.export(s,tmp_path/'export');snapshot=json.loads(paths[0].read_text(encoding='utf-8'))
    row=next(u for u in snapshot['units'] if u['mt_invocation_id']==G)
    assert ui==[str(row[k]) for k in ('n_gt','tp','fp','fn')]
    assert row['matching_policy_id']=='f4_mt_final_bbox_iou50_greedy_v1'

def test_P27_all_f4_payloads_status_revision_and_absence_preserved():
    b=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    for r in b['sessions']:
        s=MobileReviewSession.open(Path(r['path']),sidecar_path=Path(r['sidecar']))
        assert s.review.review_revision==r['review_revision'] and s.review.review_status==r['review_status']
        assert core.summary(s)==r['canvas_summary']
        payload=json.loads(s.sidecar_path.read_text(encoding='utf-8')) if s.sidecar_path.exists() else None
        assert payload==r['sidecar_payload']

def test_target_binding_or_concurrent_gt_change_cannot_write(tmp_path):
    s=private_case(tmp_path);frozen=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):stacks.complete_bbox_reference(s,G,OID,reference_quad(),expected_plate_id='PGT:wrong')
    assert s.sidecar_path.read_bytes()==frozen
    complete(s);frozen=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):complete(s)
    assert s.sidecar_path.read_bytes()==frozen

def test_P13_damaged_source_binding_cannot_complete_gt(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking import mobile_canvas_resolution as resolution
    s=private_case(tmp_path);frozen=s.sidecar_path.read_bytes()
    def rejected(*args,**kwargs):raise ValueError('test: invalid source/ROI binding')
    monkeypatch.setattr(resolution,'model_location',rejected)
    with pytest.raises(ValueError):complete(s)
    assert s.sidecar_path.read_bytes()==frozen

def test_P25_unselected_native_layout_equals_frozen_baseline(case_gui):
    import time
    root,h,ed,s=case_gui;h.window.maxsize(3000,2400)
    def rect(widget):return [widget.winfo_rootx(),widget.winfo_rooty(),widget.winfo_width(),widget.winfo_height()]
    rows=[]
    for width,height in ((1649,896),(1180,760)):
        h.window.geometry(f'{width}x{height}+20+20');native.pump(root);ready(root,h,ed)
        settled=time.monotonic()+.4
        native.pump(root,lambda:time.monotonic()>=settled)
        base=json.loads((OUT/f'native_layout_baseline_{width}.json').read_text(encoding='utf-8'))
        row=dict(size=[width,height],dpi=root.winfo_fpixels('1i'),canvas=rect(h.canvas),tree=rect(h.subject_tree),sidebar=rect(h.review_sidebar_scroll),
                 image_cards={k:rect(w.master) for k,w in ed.contexts.counters.items()})
        (OUT/f'native_layout_after_attempt_{width}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ('canvas','tree','sidebar','image_cards'):assert row[key]==base[key],(key,row[key],base[key])
        rows.append(row)
    (OUT/'native_layout_after.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

def test_P28_same_four_baseline_failures_and_no_new_failures():
    import xml.etree.ElementTree as ET
    def results(name):
        root=ET.parse(OUT/name).getroot()
        return {(t.get('classname'),t.get('name')):('FAIL' if t.find('failure') is not None else
                'ERROR' if t.find('error') is not None else 'SKIP' if t.find('skipped') is not None else 'PASS') for t in root.iter('testcase')}
    before,after=results('baseline_regression.xml'),results('final_regression.xml')
    assert set(before)==set(after)
    # Retain raw broad results and overlay only explicitly rerun nodes. This
    # covers the restored N/A caption and the transient Tk startup skip.
    for key,value in results('final_followup.xml').items():
        if key in after:after[key]=value
    expected={key for key,value in before.items() if value=='FAIL'}
    assert len(expected)==4 and {key for key,value in after.items() if value in ('FAIL','ERROR')}==expected
    assert not {key for key,value in after.items() if value=='SKIP' and before[key]!='SKIP'}
