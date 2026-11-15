"""S/B/H/V/L/R acceptance: physical references vs decision/metric projection."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,shutil,time
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/Z4_SPACE_GTR_STACK_HUD_SHIFT_20261027_001'
F='exp-1792699303016-c8b05efb';G1=F+'-a00000001';G8=F+'-a00000008';G10=F+'-a00000010'

def current_copy(path):
    path.mkdir(parents=True,exist_ok=True);p=path/'private.review.json'
    shutil.copyfile(OUT/'private/reference_baseline.review.json',p)
    return MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=p)

def settled(root,h,ed):
    native.pump(root,lambda:ed.refresh_after is None and not ed.in_refresh and not h._saving and not ed.busy,seconds=15)

def modifier(h,key,pressed):
    """Native Tk on Windows lacks a synthetic Shift_R keysym mapping.

    Post the actual left/right scan code to this test Canvas's own HWND.
    These are programmatic Windows messages, not physical operator input.
    """
    import sys
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        post=ctypes.windll.user32.PostMessageW
        post.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        scan=0x36 if key=='Shift_R' else 0x2A
        param=(scan<<16)|1|(0 if pressed else 0xC0000000)
        assert post(h.canvas.winfo_id(),0x100 if pressed else 0x101,0x10,param)
    else:h.canvas.event_generate('<KeyPress-'+key+'>' if pressed else '<KeyRelease-'+key+'>')
    h.window.update()

def identity(s,gid):
    v=data.snapshot(s,gid)
    return copy.deepcopy((v['plates'],v['links'],core.scene(s,gid)['parents']))

@pytest.fixture
def ref_gui(tmp_path,monkeypatch):
    s=current_copy(tmp_path/'data');oid='GT:'+G1
    if partial.progress(core.scene(s,G1))['object_decisions'][oid]!=partial.PLATE:partial.toggle_plate(s,G1,oid)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.fixture
def gui(tmp_path,monkeypatch):
    monkeypatch.setattr(native,'source_session',lambda _:source_session(tmp_path/'data'))
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_S02_S03_S04_space_keeps_saved_GTR_and_GTZ_visible(ref_gui):
    root,h,ed,s=ref_gui;oid='GT:'+G1;ed._choose_plate(oid);ready(root,h,ed)
    before=identity(s,G1);rev=s.review.review_revision
    native.key(h,'space');ready(root,h,ed)
    assert partial.progress(core.scene(s,G1))['object_decisions'][oid]==partial.UNRESOLVED
    assert identity(s,G1)==before and s.review.review_revision==rev+1
    assert h.canvas.find_withtag('operator_quad')
    assert any('DWR8339U' in h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('gt_badge') if h.canvas.type(i)=='text')
    assert relation(s,G1)['count']==1

def test_H01_H02_H11_source_HUD_separate_from_FINAL_and_metrics(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();summary=core.summary(s)
    assert hasattr(ed,'source_hud')
    value=ed._source_hud_value
    assert (value['mt'],value['final'],value['mz'])==(1,2,2)
    assert ed.source_hud.cget('text').splitlines()==['Model tablic MT — uruchomień: 1/klatka','Model znaków MZ — uruchomień: 2/klatka']
    assert s.sidecar_path.read_bytes()==before and core.summary(s)==summary

def test_V01_V02_V03_Shift_is_transient_and_read_only(gui):
    root,h,ed,s=gui;before=s.sidecar_path.read_bytes();native.key(h,'d');ready(root,h,ed)
    assert not h.canvas.find_withtag('shift_source_preview')
    h.canvas.event_generate('<KeyPress-Shift_L>');root.update();ed.overlay()
    assert h.canvas.find_withtag('shift_source_preview')
    h.canvas.event_generate('<KeyRelease-Shift_L>');root.update();ed.overlay()
    assert not h.canvas.find_withtag('shift_source_preview')
    assert ed.mode=='draw' and not ed.points and s.sidecar_path.read_bytes()==before

def relation(s,gid,aid=None):
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import linked_references
    v=data.snapshot(s,gid);c=next(c for c in v['candidates'] if aid is None or c['attempt_id']==aid)
    return linked_references(core.scene(s,gid),v,c)

def fixture_archive(path,*,n=1,mz=True,mutate=None):
    import csv,io,zipfile
    from test_z4_p0_physical_gt import csv_bytes
    from test_mobile_human_review import load
    source=source_session(path/'template',n=n,mz=mz)
    with zipfile.ZipFile(source.path) as z:payload={p:z.read(p) for p in z.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode())))
    trace=json.loads(payload['telemetry/traces.jsonl'])
    if mutate:mutate(rows,trace,payload)
    payload['samples/attempts.csv']=csv_bytes(rows)
    if 'telemetry/traces.jsonl' in payload:payload['telemetry/traces.jsonl']=json.dumps(trace).encode()
    p=path/'fixture.alprsession'
    with zipfile.ZipFile(p,'w') as z:
        for key,value in payload.items():z.writestr(key,value)
    return load(p,path)

def test_S01_S12_real_current_identity_and_source_immutable(tmp_path):
    s=current_copy(tmp_path);v=data.snapshot(s,G1);p=next(iter(v['plates'].values()))
    assert p['gt_text']=='DWR8339U' and p['geometry_state']=='operator_corrected'
    assert relation(s,G1)['ids']==[p['plate_id']] and relation(s,G1)['count']==1
    assert v['candidates'][0]['operator_decision']==partial.UNRESOLVED
    assert v['candidates'][0]['plate_id'] is None  # scientific projection unchanged
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import display_snapshot
    before=s.sidecar_path.read_bytes();ui=display_snapshot(core.scene(s,G1),v)
    assert ui['candidates'][0]['plate_id']==p['plate_id']
    assert s.sidecar_path.read_bytes()==before

def test_S05_existing_link_provenance_is_not_replaced_on_positive_Space(tmp_path):
    s=current_copy(tmp_path);before=identity(s,G10)
    assert partial.progress(core.scene(s,G10))['object_decisions']['GT:'+G10]==partial.UNRESOLVED
    partial.toggle_plate(s,G10,'GT:'+G10)
    after=identity(s,G10)
    assert after[1:]==before[1:]  # exact link metadata and immutable source
    fields=('plate_id','origin_object_id','primary_object_id','gt_quad','gt_text','geometry_state','text_confirmed')
    assert {pid:{k:p[k] for k in fields} for pid,p in after[0].items()}=={pid:{k:p[k] for k in fields} for pid,p in before[0].items()}
    # Existing record_scene_identity may materialize its legacy DISTINCT
    # default on an explicit write; no change to that pre-existing policy.

@pytest.fixture
def bbox_gui(tmp_path,monkeypatch):
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=tmp_path/'fresh.review.json');s.set_review_mode('assisted')
    partial.toggle_plate(s,G8,'GT:'+G8);data.edit_gt(s,G8,'GT:'+G8,text='WU1092S')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    iid=next(i for i in h.subject_tree.get_children('') if G8 in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==G8 and h._image_entry==core.scene(s,G8)['image_entry'] and not h._saving)
    ready(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def test_B03_B05_B06_R06_bbox_completion_and_pending_redraw_same_GT(bbox_gui):
    root,h,ed,s=bbox_gui;oid='GT:'+G8
    ed._choose_plate(oid);ready(root,h,ed);before=data.snapshot(s,G8);pid=before['links'][oid]['plate_id']
    assert relation(s,G8)['count']==0
    quad=[[566,599],[623,599],[623,622],[566,622]]
    native.key(h,'d');assert ed.draw_target['kind']=='complete'
    for p in quad:native.click(h,ed,p)
    ready(root,h,ed);assert relation(s,G8)['count']==1
    committed=copy.deepcopy(data.snapshot(s,G8)['plates'][pid]['gt_quad'])
    assert list(data.snapshot(s,G8)['plates'])==[pid] and data.snapshot(s,G8)['plates'][pid]['gt_text']=='WU1092S'
    assert any('GTR 1' in h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('mt_relation_text'))
    settled(root,h,ed);native.key(h,'space');ready(root,h,ed)
    assert relation(s,G8)['count']==1
    native.key(h,'d');assert ed.draw_target['kind']=='replace'
    replacement=[[568,601],[621,601],[621,620],[568,620]]
    for p in replacement:native.click(h,ed,p)
    ready(root,h,ed);settled(root,h,ed)
    replaced=copy.deepcopy(data.snapshot(s,G8)['plates'][pid]['gt_quad']);assert replaced!=committed
    ed.undo();ready(root,h,ed);assert data.snapshot(s,G8)['plates'][pid]['gt_quad']==committed
    ed.undo(redo=True);ready(root,h,ed);assert data.snapshot(s,G8)['plates'][pid]['gt_quad']==replaced
    assert relation(s,G8)['count']==1 and list(data.snapshot(s,G8)['plates'])==[pid]
    row=data.diagnostics(s,[G8])['rows'][0]
    assert row['mt_quad'] is None and row['mz_status']=='NOT_RUN' and row['polygon_iou_pct'] is row['nmce_pct'] is row['cer'] is None

def test_V02_corner_correction_also_allows_preview(ref_gui):
    root,h,ed,s=ref_gui;oid='GT:'+G1;ed._choose_plate(oid);ready(root,h,ed);before=identity(s,G1)
    point=core._object(ed._read(),oid)['quad'][0];x,y=map(round,ed.to_canvas(point))
    native.focus_canvas(h);h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);root.update();assert ed.drag
    modifier(h,'Shift_L',True);root.update();assert ed.shift_preview.active and h.canvas.find_withtag('shift_source_preview')
    modifier(h,'Shift_L',False);root.update();assert not h.canvas.find_withtag('shift_source_preview')
    ed.escape();ready(root,h,ed);assert identity(s,G1)==before

def test_S05_S06_S07_S08_round_trip_history_restart_and_other_twenty(ref_gui):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    root,h,ed,s=ref_gui;oid='GT:'+G1;ed._choose_plate(oid);ready(root,h,ed)
    other={k:copy.deepcopy(v) for k,v in s.review.invocation_annotations.items() if k!=G1}
    before=identity(s,G1)
    for _ in range(2):native.key(h,'space');ready(root,h,ed);assert identity(s,G1)==before
    settled(root,h,ed)
    ed.undo();ready(root,h,ed);assert identity(s,G1)==before
    ed.undo(redo=True);ready(root,h,ed);assert identity(s,G1)==before
    settled(root,h,ed)
    for gid in (G10,G1):
        iid=next(i for i in h.subject_tree.get_children('') if gid in subject_groups(s,h._subject_ids[i]))
        h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry'] and not h._saving)
        ready(root,h,ed)
    assert identity(s,G1)==before
    assert {k:v for k,v in s.review.invocation_annotations.items() if k!=G1}==other and len(other)==20
    frozen=s.sidecar_path.read_bytes();reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert identity(reopened,G1)==before and s.sidecar_path.read_bytes()==frozen

def test_S09_pending_metrics_do_not_keep_old_eligibility(ref_gui):
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    root,h,ed,s=ref_gui;ed._choose_plate('GT:'+G1);ready(root,h,ed)
    before=identity(s,G1);assert evaluate_unit(s,G1)['status']=='MEASURED'
    native.key(h,'space');ready(root,h,ed)
    assert evaluate_unit(s,G1)['status']!='MEASURED' and identity(s,G1)==before
    assert all(v=='—' for v in clean.context(ed,ed.assessment.model)['image_counts'].values())

def test_S10_V07_entry_keeps_text_Space_and_Shift(ref_gui):
    root,h,ed,s=ref_gui;oid='GT:'+G1;ed._choose_plate(oid);ready(root,h,ed);ed.start_entry(oid);root.update()
    assert ed.entry is not None
    before=identity(s,G1);decision=partial.progress(core.scene(s,G1))['object_decisions']
    ed.entry.focus_force();root.update()
    ed.entry.event_generate('<KeyPress-Shift_L>');ed.entry.event_generate('<KeyRelease-Shift_L>')
    ed.entry.event_generate('<KeyPress-space>');ed.entry.event_generate('<KeyRelease-space>');root.update()
    assert not ed.shift_preview.active and partial.progress(core.scene(s,G1))['object_decisions']==decision
    ed.escape();ready(root,h,ed);assert identity(s,G1)==before

def test_S11_selection_changes_no_GTZ_or_links(ref_gui):
    root,h,ed,s=ref_gui;before=identity(s,G1)
    for _ in range(3):ed._choose_plate('GT:'+G1);ready(root,h,ed);ed.select_mode();ready(root,h,ed)
    assert identity(s,G1)==before

def test_B01_B04_no_overlap_auto_link(tmp_path):
    s=source_session(tmp_path);g='mt-one';q=core.scene(s,g)['parents'][0]['quad']
    assert relation(s,g,'a0')['count']==0
    core.add_plate(s,g,q,partial_review=True)
    assert relation(s,g,'a0')['count']==0

def test_B02_H04_L02_fresh_invalid_Pose_not_invented(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=tmp_path/'fresh.json')
    assert relation(s,G8)['count']==0 and core.scene(s,G8)['parents'][0]['quad'] is None
    v=frame_execution(s,G8);assert (v['mt'],v['final'],v['mz'])==(1,1,0)
    assert not s.sidecar_path.exists()

def test_B07_B08_shared_reference_is_distinct_ID_once_per_source_MT(tmp_path):
    from test_z4_frame_stacks import linked
    s=linked(tmp_path,data,n=3,suppressed=None)
    values=[relation(s,'mt-one','a'+str(i)) for i in range(3)]
    assert all(v['count']==1 for v in values) and len({v['ids'][0] for v in values})==1

def test_B05_R04_stack_cycles_persisted_links_after_Space(tmp_path):
    from test_z4_frame_stacks import linked
    from auto_annotation_tool.gui import mobile_canvas_stacks as ui
    s=linked(tmp_path,data,n=3,suppressed=None);partial.toggle_plate(s,'mt-one','GT:a0')
    before=s.sidecar_path.read_bytes();chosen=[];canvas=NS()
    ed=NS(_read=lambda:core.scene(s,'mt-one'),h=NS(session=s,current_invocation='mt-one'),selected='GT:a0',
        entry=None,drag=None,points=[],c=canvas,clean_bar=True,_is_typing_shortcut=lambda w:False)
    def choose(oid):chosen.append(oid);ed.selected=oid
    ed._choose_plate=choose
    for _ in range(3):assert ui.cycle(ed,NS(widget=canvas),False)=='break'
    assert chosen==['GT:a1','GT:a2','GT:a0'] and s.sidecar_path.read_bytes()==before

def test_S04_B02_text_only_BBOX_reference_retains_GTZ_after_Space(tmp_path):
    from auto_annotation_tool.gui.mobile_canvas_draw_target import resolve
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=tmp_path/'fresh.json');s.set_review_mode('assisted')
    oid='GT:'+G8;partial.toggle_plate(s,G8,oid);data.edit_gt(s,G8,oid,text='WU1092S');partial.toggle_plate(s,G8,oid)
    before=s.sidecar_path.read_bytes();ed=NS(_read=lambda:core.scene(s,G8),h=NS(session=s,current_invocation=G8),
        selected=oid,show_fp_var=NS(get=lambda:True),cache_key=None)
    row=clean.selected(ed)
    assert row['plate']['gt_text']=='WU1092S' and row['gt'] is None and relation(s,G8)['count']==0
    with pytest.raises(ValueError,match='Spacj'):resolve(ed)
    assert s.sidecar_path.read_bytes()==before and len(data.snapshot(s,G8)['plates'])==1

def test_B09_RAW_suppressed_has_no_FINAL_badge(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import linked_references
    s=session_fixture(tmp_path,n=2,suppressed=1);v=data.snapshot(s,'mt-one')
    c=next(c for c in v['candidates'] if c['attempt_id']=='a1');assert c['role']=='SUPPRESSED'
    assert linked_references(core.scene(s,'mt-one'),v,c)['count'] is None

@pytest.mark.parametrize('fault',('binding','role','orphan','many_ids','wrong_detection'))
def test_B10_unproven_relationship_never_fake_zero(tmp_path,fault):
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import linked_references
    s=source_session(tmp_path);sc=core.scene(s,'mt-one');v=data.snapshot(s,'mt-one');c=v['candidates'][0]
    if fault=='binding':v['source_binding']='wrong'
    elif fault=='role':c['role']='UNKNOWN'
    elif fault=='orphan':v['links'][c['object_id']]=dict(plate_id='missing',status='ASSOCIATED_FOR_GEOMETRY')
    elif fault=='many_ids':v['links'][c['object_id']]=dict(plate_id=['g1','g2'],status='ASSOCIATED_FOR_GEOMETRY')
    else:c['mt_detection_index']=999
    assert linked_references(sc,v,c)['count'] is None

@pytest.mark.parametrize('n,mz,want',[(1,True,(1,1,1)),(2,True,(1,2,2)),(2,False,(1,2,0)),(0,False,(1,0,0))])
def test_H01_H02_H03_H04_H07_actual_calls_not_detections(tmp_path,n,mz,want):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    s=source_session(tmp_path,n=n,mz=mz);before=s.sidecar_path.read_bytes()
    v=frame_execution(s,'mt-one');assert (v['mt'],v['final'],v['mz'])==want
    assert len(v['mt_invocation_ids'])==1 and s.sidecar_path.read_bytes()==before

def test_H05_unknown_no_trace_never_zero(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution,caption
    def change(rows,trace,payload):
        payload.pop('telemetry/traces.jsonl')
        for row in rows:row.update(mz_status='UNKNOWN',mz_executed='',prediction='')
    s=fixture_archive(tmp_path,mutate=change);v=frame_execution(s,'mt-one')
    assert v['mz'] is None and 'MZ wyw.: —' in caption(v) and 'brak wiarygodnego śladu' in caption(v)
    assert v['mt']==1 and v['final'] is None  # lost NMS proof is not FINAL=1

def test_H06_duplicate_sample_rows_one_execution(tmp_path):
    import csv,io
    from test_z4_p0_physical_gt import csv_bytes
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    def change(rows,trace,payload):
        samples=list(csv.DictReader(io.StringIO(payload['samples/index.csv'].decode())))
        samples.append(dict(samples[0],capture_id='duplicate'))
        payload['samples/index.csv']=csv_bytes(samples);payload['samples/crops/duplicate.jpg']=payload['samples/crops/c0.jpg']
    s=fixture_archive(tmp_path,mutate=change)
    assert len(s.sample_ids_by_attempt['a0'])==2 and frame_execution(s,'mt-one')['mz']==1

def test_H07_explicit_MZ_execution_IDs_without_aggregate(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    def change(rows,trace,payload):
        trace['counters'].pop('mz_runs')
        for index,row in enumerate(rows):row.update(mz_executed='true',mz_execution_id='mz-'+str(index))
    s=fixture_archive(tmp_path,n=2,mutate=change);v=frame_execution(s,'mt-one')
    assert v['mz']==2 and v['mz_execution_ids']==['mz-0','mz-1']

@pytest.mark.parametrize('fault',('cancelled','error','missing_id','source_conflict','not_run','incomplete_trace','duplicate_trace'))
def test_H08_not_run_cancelled_error_and_contradictions(tmp_path,fault):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    def change(rows,trace,payload):
        row=rows[0]
        if fault=='cancelled':row.update(stale_or_cancelled='true',cancel_reason='cancelled')
        elif fault=='error':row.update(execution_error='backend failure')
        elif fault=='missing_id':row.pop('mt_invocation_id')
        elif fault=='source_conflict':row['source_frame_expected']='false'
        elif fault=='not_run':
            row.update(mt_executed='false',mt_status='NOT_RUN',mt_detection_count=0,mt_detection_index='',mz_status='NOT_RUN',mz_executed='false')
            trace['counters'].update(mt_runs_this_frame=0,plate_detections_after_dedup=0,mz_runs=0)
        elif fault=='incomplete_trace':trace['counters']['mt_runs_this_frame']=2
    s=fixture_archive(tmp_path,mutate=change)
    if fault=='duplicate_trace':
        # An ambiguous frame-key must not be resolved by its first record.
        import zipfile
        with zipfile.ZipFile(s.path) as z:payload={n:z.read(n) for n in z.namelist()}
        payload['telemetry/traces.jsonl']+=b'\n'+payload['telemetry/traces.jsonl']
        p=tmp_path/'duplicate.alprsession'
        with zipfile.ZipFile(p,'w') as z:
            for n,v in payload.items():z.writestr(n,v)
        s=MobileReviewSession.open(p,sidecar_path=tmp_path/'duplicate.review.json')
    v=frame_execution(s,'mt-one')
    if fault=='not_run':assert (v['mt'],v['final'],v['mz'])==(0,0,0)
    elif fault=='duplicate_trace':assert v['mz'] is None
    else:assert v['mt'] is None and v['mz'] is None

def test_H09_H10_per_frame_and_session_scope_and_no_selection_dependency(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    s=current_copy(tmp_path);before=s.sidecar_path.read_bytes()
    values=[frame_execution(s,g) for g in (G1,G10,G1)]
    assert all((v['mt'],v['final'],v['mz'])==(1,1,1) for v in values)
    assert values[0]==values[2] and values[0]['frame']!=values[1]['frame']
    assert all(len(v['mt_invocation_ids'])==1 for v in values) and s.sidecar_path.read_bytes()==before

def test_H02_two_actual_MT_invocations_share_one_source_frame(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
    def change(rows,trace,payload):
        for i,row in enumerate(rows):row.update(mt_invocation_id=f'g{i}',mt_detection_index=0,mt_detection_count=1)
        trace['counters']['mt_runs_this_frame']=2
    s=fixture_archive(tmp_path,n=2,mutate=change)
    for gid in ('g0','g1'):
        v=frame_execution(s,gid)
        assert (v['mt'],v['final'],v['mz'])==(2,2,2) and set(v['mt_invocation_ids'])=={'g0','g1'}

def test_R04_PPM_keeps_saved_GTR_but_selected_MT_is_FP(tmp_path):
    s=current_copy(tmp_path);oid='GT:'+G1
    if partial.progress(core.scene(s,G1))['object_decisions'][oid]!=partial.PLATE:partial.toggle_plate(s,G1,oid)
    physical=copy.deepcopy(data.snapshot(s,G1)['plates'])
    data.toggle_unassociated(s,G1,oid,partial_review=True)
    ed=NS(_read=lambda:core.scene(s,G1),h=NS(session=s,current_invocation=G1),selected=oid,show_fp_var=NS(get=lambda:True))
    assert clean.selected(ed)['fp']
    assert data.snapshot(s,G1)['plates']==physical

def test_V04_V11_both_Shifts_repeat_and_thirty_cycles_no_repaint(gui,monkeypatch):
    root,h,ed,s=gui;native.key(h,'d');ready(root,h,ed);before=s.sidecar_path.read_bytes();paint=[];latencies=[]
    monkeypatch.setattr(h,'_paint_image',lambda:paint.append(True))
    for index in range(30):
        started=time.perf_counter()
        key='Shift_R' if index%2 else 'Shift_L'
        for _ in range(3):modifier(h,key,True)
        root.update();assert ed.shift_preview.active and len(h.canvas.find_withtag('shift_mt_pose'))==2
        modifier(h,key,False);root.update();assert not h.canvas.find_withtag('shift_source_preview')
        latencies.append(time.perf_counter()-started)
    assert not paint and s.sidecar_path.read_bytes()==before
    measured=dict(cycles=30,cycle_ms=[round(x*1000,3) for x in latencies],p95_cycle_ms=sorted(latencies)[28]*1000,full_image_repaints=len(paint))
    (OUT/'shift_latency.json').write_text(json.dumps(measured,indent=2),encoding='utf-8')

def test_V05_V10_preview_click_is_same_drawing_and_not_source_collision(gui):
    root,h,ed,s=gui;native.key(h,'d');ready(root,h,ed);h.canvas.event_generate('<KeyPress-Shift_L>');root.update()
    native.click(h,ed,(200,180));ready(root,h,ed)
    assert len(ed.points)==1 and not ed.collision_warning.active and ed.selected==''
    h.canvas.event_generate('<KeyRelease-Shift_L>');root.update();native.click(h,ed,(400,180));ready(root,h,ed)
    assert len(ed.points)==2 and not ed.collision_warning.active

@pytest.mark.parametrize('action',('focus_out','escape','navigate','tab_change','dispose'))
def test_V06_preview_cleanup_without_stuck_state(gui,action):
    root,h,ed,s=gui;native.key(h,'d');ready(root,h,ed);modifier(h,'Shift_R',True);root.update();assert ed.shift_preview.active
    if action=='focus_out':h.canvas.event_generate('<FocusOut>')
    elif action=='escape':ed.escape()
    elif action=='navigate':ed.prepare_navigation()
    elif action=='tab_change':ed.shift_preview.notebook.select(1)
    else:ed.shift_preview.dispose()
    root.update();ed.overlay();assert not ed.shift_preview.active and not h.canvas.find_withtag('shift_source_preview')

def test_B05_S03_R04_provisional_reference_remains_linked_and_redrawable(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import display_snapshot
    from auto_annotation_tool.ranking.mobile_manual_frame_guard import find_point_collision
    from auto_annotation_tool.gui.mobile_canvas_draw_target import resolve
    s=source_session(tmp_path,n=1);gid='mt-one';oid='GT:a0'
    partial.toggle_plate(s,gid,oid);p=data.snapshot(s,gid)['plates'];pid=next(iter(p))
    assert p[pid]['geometry_state']=='unconfirmed' and relation(s,gid)['count']==1
    partial.toggle_plate(s,gid,oid);sc=core.scene(s,gid);v=data.snapshot(s,gid);ui=display_snapshot(sc,v)
    assert pid in ui['active_plate_ids'] and relation(s,gid)['count']==1
    assert find_point_collision(sc,(225,175)) is None
    ed=NS(_read=lambda:core.scene(s,gid),h=NS(session=s,current_invocation=gid),selected=oid,show_fp_var=NS(get=lambda:True),cache_key=None)
    target=resolve(ed);assert target['kind']=='replace' and target['plate_id']==pid
    data.redraw_reference(s,gid,target,[[152,142],[298,142],[298,208],[152,208]])
    assert list(data.snapshot(s,gid)['plates'])==[pid]
    assert partial.progress(core.scene(s,gid))['object_decisions'][oid]==partial.UNRESOLVED

def test_V08_V09_guides_and_warning_survive_preview(tmp_path,monkeypatch):
    from test_z4_gtr_collisions_replace import seeded,center,Q1,advance
    s,a,b=seeded(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        native.key(h,'d');ready(root,h,ed);native.click(h,ed,center(Q1));root.update();w=ed.collision_warning
        assert w.active;started=w.started_at;h.canvas.event_generate('<KeyPress-Shift_L>');root.update()
        assert w.active and w.started_at==started
        assert all(h.canvas.itemcget(i,'fill')=='#165D8C' for i in h.canvas.find_withtag('gtr_guide_core'))
        assert all(h.canvas.itemcget(i,'fill')=='#EAF5FC' for i in h.canvas.find_withtag('gtr_guide_halo'))
        h.canvas.event_generate('<KeyRelease-Shift_L>');root.update();assert w.active and w.started_at==started
        advance(root,2.8);assert not w.active and not w.after_ids
    finally:
        try:next(gen)
        except StopIteration:pass

def test_L01_L03_source_label_typed_and_no_phantom_GTZ(gui):
    root,h,ed,s=gui;ready(root,h,ed)
    labels=[h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('mt_relation_text')]
    assert len(labels)==2 and all('POSE' in label and 'decyzja oczekuje' in label and 'GTR 0' in label for label in labels)
    assert not h.canvas.find_withtag('gt_badge')
    ed._choose_plate('GT:a0');ready(root,h,ed)
    assert 'Brak jawnej' in clean.context(ed,ed.assessment.model)['object'].replace('brak jawnej','Brak jawnej')
    assert 'GTZ' not in clean.context(ed,ed.assessment.model)['object']

def test_L04_real_image10_relation_belongs_to_GTR1_not_independent_GTR2(tmp_path):
    s=current_copy(tmp_path);v=data.snapshot(s,G10);c=v['candidates'][0];rel=relation(s,G10)
    assert len(v['plates'])==2 and rel['count']==1
    assert v['links'][c['object_id']]['plate_id']==rel['ids'][0]
    other=next(p for pid,p in v['plates'].items() if pid not in rel['ids'])
    assert other['origin_object_id'].startswith('MANUAL:') and other['gt_text']==''
    # The supplied screenshot hides pending GTR1; it does not prove GTR2 linked.

def assert_L07_BBOX_value_comes_only_from_frozen_matching(tmp_path):
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=tmp_path/'bbox.json');s.set_review_mode('assisted')
    oid='GT:'+G8;partial.toggle_plate(s,G8,oid)
    inventory=json.loads((OUT/'source_reference_inventory.json').read_text(encoding='utf-8'))
    quad=next(p['gt_quad'] for p in inventory[G8]['snapshot']['plates'].values() if p['gt_quad'])
    data.complete_bbox_reference(s,G8,oid,quad)
    result=evaluate_unit(s,G8);assert result['status']=='MEASURED' and result['fp']==1 and not result['matched_pairs']
    frozen=s.sidecar_path.read_bytes()
    ed=NS(_read=lambda:core.scene(s,G8),h=NS(session=s,current_invocation=G8,report=None),selected=oid,
        entry=None,show_fp_var=NS(get=lambda:True),cache_key=None,
        inventory=lambda:dict(decisions=0,inspected=1,total=21,remaining=20))
    text=clean.context(ed,{})['object']
    assert 'BBOX IoU: —' in text and 'Wynik dopasowania BBOX: FP' in text
    assert 'Polygon IoU / NMCE: —' in text and s.sidecar_path.read_bytes()==frozen

def test_L05_L06_L07_operator_presence_and_bbox_outcome_distinct(ref_gui,tmp_path):
    root,h,ed,s=ref_gui;ed._choose_plate('GT:'+G1);ready(root,h,ed)
    text=clean.context(ed,ed.assessment.model)['object']
    assert 'DWR8339U' in text and 'Decyzja operatora: poprawna' in text
    assert 'Wynik dopasowania BBOX: FP' in text and 'IoU' in text
    assert_L07_BBOX_value_comes_only_from_frozen_matching(tmp_path)

def test_R01_R02_R05_R09_current_baseline_protected_and_metrics_unchanged():
    b=json.loads((OUT/'baseline_worktree.json').read_text(encoding='utf-8'))
    assert len(b['sessions'])==12
    for n,value in b['protected'].items():
        p=Path(n);assert p.exists()==value['exists']
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256']
    for n,h in b['prior_reports'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
    for n in ('auto_annotation_tool/ranking/mobile_mt_detection_evaluation.py','auto_annotation_tool/ranking/mobile_pose_metrics.py'):
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==b['source_files'][n]

def test_R04_native_operator_workflow_on_current_source_contract(gui,monkeypatch):
    """Real Tk handlers with a deterministic canvas pointer for S/A.

    Programmatic keys cannot move the OS cursor. The menu is constructed
    without blocking Windows menu tracking; no menu action is selected.
    """
    import tkinter as tk
    root,h,ed,s=gui;source=copy.deepcopy(core.scene(s,'mt-one')['parents']);pointer=[(0,0)]
    monkeypatch.setattr(ed,'_pointer_for_selection',lambda event=None:pointer[0])
    for oid,point in (('GT:a0',(225,175)),('GT:a1',(445,175))):
        pointer[0]=ed.to_canvas(point);native.key(h,'s',xy=pointer[0]);ready(root,h,ed);assert ed.selected==oid
        native.key(h,'space');ready(root,h,ed);settled(root,h,ed)
    pointer[0]=ed.to_canvas((225,175));native.key(h,'s',xy=pointer[0]);ready(root,h,ed);assert ed.selected=='GT:a0'
    rev=s.review.review_revision;roi=ed.show_roi_overlay
    native.key(h,'t');ready(root,h,ed);assert ed.show_roi_overlay!=roi
    native.key(h,'t');ready(root,h,ed);assert ed.show_roi_overlay==roi and s.review.review_revision==rev
    old=(h._view_center,h._view_zoom);native.key(h,'a',xy=pointer[0]);ready(root,h,ed);assert h._view_zoom!=old[1]
    native.key(h,'a',xy=pointer[0]);ready(root,h,ed);assert (h._view_center,h._view_zoom)==old
    native.key(h,'n');ready(root,h,ed);assert ed.selected=='GT:a1'
    native.click(h,ed,(225,175));ready(root,h,ed);assert ed.selected=='GT:a0'
    ed.start_entry('GT:a0');root.update();assert ed.entry is not None
    ed.entry_var.set('WATEST0');ed.entry.event_generate('<Return>');ready(root,h,ed);settled(root,h,ed)
    pid=data.snapshot(s,'mt-one')['links']['GT:a0']['plate_id'];assert data.snapshot(s,'mt-one')['plates'][pid]['gt_text']=='WATEST0'
    ed.start_entry('GT:a0');ed.entry_var.set('DRAFT01');ed.entry.event_generate('<Escape>');ready(root,h,ed)
    assert data.snapshot(s,'mt-one')['plates'][pid]['gt_text']=='WATEST0'
    # Actual corner drag, then the unchanged source R reset preserves GTZ.
    pointer[0]=ed.to_canvas((225,175));native.key(h,'s',xy=pointer[0]);ready(root,h,ed);assert ed.selected=='GT:a0'
    original=copy.deepcopy(core.scene(s,'mt-one')['parents'][0]['quad']);x,y=map(round,ed.to_canvas(original[0]))
    nx,ny=map(round,ed.to_canvas((original[0][0]+6,original[0][1]+4)))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);root.update();assert ed.drag
    h.canvas.event_generate('<B1-Motion>',x=nx,y=ny)
    h.canvas.event_generate('<ButtonRelease-1>',x=nx,y=ny);ready(root,h,ed);settled(root,h,ed)
    assert data.snapshot(s,'mt-one')['plates'][pid]['gt_quad']!=original
    native.key(h,'r');ready(root,h,ed);settled(root,h,ed)
    assert data.snapshot(s,'mt-one')['plates'][pid]['gt_quad']==original and data.snapshot(s,'mt-one')['plates'][pid]['gt_text']=='WATEST0'
    native.key(h,'Escape');native.key(h,'Escape');ready(root,h,ed)
    pointer[0]=ed.to_canvas((225,175));native.key(h,'s',xy=pointer[0]);ready(root,h,ed);assert ed.selected=='GT:a0'
    for want in (partial.NOT_PLATE,partial.PLATE):
        from test_z4_bad_only_mt_final import source_badge,ppm
        ppm(h,*source_badge(h,ed));ready(root,h,ed);settled(root,h,ed)
        assert partial.progress(core.scene(s,'mt-one'))['object_decisions']['GT:a0']==want
    popups=[];monkeypatch.setattr(tk.Menu,'tk_popup',lambda menu,*a,**kw:popups.append(menu))
    x,y=map(round,ed.to_canvas((225,175)))
    h.canvas.event_generate('<ButtonPress-3>',x=x,y=y,state=4);root.update()
    assert popups and popups[0].index('end') is not None
    native.key(h,'Escape');ready(root,h,ed);assert not ed.selected
    native.key(h,'d');assert ed.mode=='draw' and ed.draw_target is None
    for p in ((600,500),(790,500),(790,580),(600,580)):native.click(h,ed,p)
    ready(root,h,ed);settled(root,h,ed);manual=ed.selected;assert manual.startswith('MANUAL:')
    native.key(h,'r');ready(root,h,ed);settled(root,h,ed);assert not ed.selected
    native.focus_canvas(h);h.canvas.event_generate('<KeyPress-z>',state=4);h.canvas.event_generate('<KeyRelease-z>',state=4)
    ready(root,h,ed);settled(root,h,ed)
    assert any(p['origin_object_id']==manual for p in data.snapshot(s,'mt-one')['plates'].values())
    native.focus_canvas(h);h.canvas.event_generate('<KeyPress-y>',state=4);h.canvas.event_generate('<KeyRelease-y>',state=4)
    ready(root,h,ed);assert not any(p['origin_object_id']==manual for p in data.snapshot(s,'mt-one')['plates'].values())
    assert core.scene(s,'mt-one')['parents']==source and data.snapshot(s,'mt-one')['plates'][pid]['gt_text']=='WATEST0'
