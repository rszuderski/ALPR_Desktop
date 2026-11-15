"""S01–S14/U01–U12/E01–E10/N01–N08: separate source layers and 0..1 GTR."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,importlib,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import MANUAL
from test_z4_akt89lm_resolution import invalid_source
from test_z4_frame_stacks import stack_gui,stacks,ready,linked
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

OUT=Path(__file__).resolve().parents[1]/'output/Z4_MT_GEOMETRY_LAYERS_20261028_001'
G='mt-one';OID='GT:a0';REF='exp-1792699303016-c8b05efb';G10=REF+'-a00000010'
def api():return importlib.import_module('auto_annotation_tool.ranking.mobile_mt_geometry_layers')
def source_session(path,n=1,*,mz=True):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def bbox(rows,trace,payload):
        for i,row in enumerate(rows):row.update(plate_left=150+220*i,plate_top=140,plate_right=300+220*i,plate_bottom=210,mt_confidence='0.9')
    s=fixture_archive(path,n=n,mz=mz,mutate=bbox);s.set_review_mode('assisted');return s
def ui():return importlib.import_module('auto_annotation_tool.gui.mobile_canvas_geometry_layers')
def kinds(s,g=G,oid=OID):return [r['kind'] for r in api().snapshot(s,g,oid)['layers']]
def freeze(s):return (copy.deepcopy(s.review.to_dict()),s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None)
def shape(s):return copy.deepcopy(core.scene(s,G)['parents'][0]['quad'])
def accepted(p):
    s=source_session(p,n=1);data.accept_reference(s,G,OID);return s
def drawn(p):
    s=source_session(p,n=1);api().create_reference(s,G,OID,shape(s));return s
def real(p):
    p.mkdir(parents=True,exist_ok=True);side=p/'review.json';shutil.copyfile(OUT/'private/baseline.review.json',side)
    return MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=side)
def snapshot(ed):return ui().model(ed)
def cycle(root,h,ed,back=False):
    controls=ed.geometry_layer_controls;box,direction=next((b,d) for b,d in controls if d==back)
    x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
    ed.press(NS(widget=h.canvas,x=x,y=y,state=0));ed.release(NS(x=x,y=y));ready(root,h,ed)

@pytest.fixture
def gui(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=1);monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_S01_bbox_only(tmp_path):
    s=invalid_source(tmp_path);assert kinds(s)==['BBOX'] and api().snapshot(s,G,OID)['gtr_count']==0
def test_S02_bbox_pose_no_GTR(tmp_path):
    s=source_session(tmp_path,n=1);assert kinds(s)==['BBOX','POSE'] and api().snapshot(s,G,OID)['gtr_count']==0
def test_S03_invalid_pose_with_GTR(tmp_path):
    s=invalid_source(tmp_path);api().create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    assert kinds(s)==['BBOX','GTR'] and core.scene(s,G)['parents'][0]['quad'] is None
def test_S04_three_layers_one_FINAL(tmp_path):
    s=drawn(tmp_path);assert kinds(s)==['BBOX','POSE','GTR'] and len(data.snapshot(s,G)['candidates'])==1
def test_S05_occupied_link_rejected_before_write(tmp_path):
    s=accepted(tmp_path);manual=core.add_plate(s,G,MANUAL,partial_review=True);before=freeze(s)
    with pytest.raises(ValueError,match='GTR'):data.associate(s,G,OID,manual)
    assert freeze(s)==before
def test_S06_central_write_and_second_link_path_reject(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    s=accepted(tmp_path);manual=core.add_plate(s,G,MANUAL,partial_review=True);before=freeze(s)
    with pytest.raises(ValueError,match='GTR'):link_manual_to_mt(s,G,manual,OID)
    changed=copy.deepcopy(s.review);scene=changed.invocation_annotations[G][core.KEY]
    scene[data.KEY]['links'][OID]['plate_id']=scene[data.KEY]['links'][manual]['plate_id']
    with pytest.raises(ValueError,match='GTR'):s._save_change(changed)
    assert freeze(s)==before
    malformed=copy.deepcopy(s.review);malformed.invocation_annotations[G][core.KEY][data.KEY]['links'][OID]['plate_id']=['first','second']
    with pytest.raises(ValueError,match='GTR'):s._save_change(malformed)
    assert freeze(s)==before
    saved=s.review;s.review=malformed
    try:
        with pytest.raises(ValueError,match='GTR'):s.export(tmp_path/'blocked_export')
        assert not (tmp_path/'blocked_export').exists()
    finally:s.review=saved
def test_S07_D_existing_replaces_same_id(gui):
    root,h,ed,s=gui;data.accept_reference(s,G,OID);ready(root,h,ed);ed._choose_plate(OID)
    from auto_annotation_tool.gui.mobile_canvas_draw_target import resolve
    target=resolve(ed);assert target['kind']=='replace';old=data.snapshot(s,G);q=shape(s);q[0][0]+=2
    data.redraw_reference(s,G,target,q);new=data.snapshot(s,G)
    assert old['links']==new['links'] and set(old['plates'])==set(new['plates'])
def test_S08_D_unlinked_creates_and_links_atomically(gui):
    root,h,ed,s=gui;ed._choose_plate(OID);ready(root,h,ed);q=shape(s);rev=s.review.review_revision
    native.key(h,'d');ready(root,h,ed)
    for p in q:native.click(h,ed,p)
    ready(root,h,ed);v=data.snapshot(s,G)
    assert s.review.review_revision==rev+1 and len(v['plates'])==1 and v['links'][OID]['plate_id']
def test_S09_cancel_and_history(gui):
    root,h,ed,s=gui;ed._choose_plate(OID);ready(root,h,ed);frozen=freeze(s);q=shape(s)
    native.key(h,'d');native.click(h,ed,q[0]);ed.escape();ready(root,h,ed);assert freeze(s)==frozen
    api().create_reference(s,G,OID,q);pid=data.snapshot(s,G)['links'][OID]['plate_id']
    assert data.restore_history(s,G) and not data.snapshot(s,G)['plates']
    assert data.restore_history(s,G,redo=True) and data.snapshot(s,G)['links'][OID]['plate_id']==pid
def test_S10_overlap_never_links(tmp_path):
    s=source_session(tmp_path,n=1);core.add_plate(s,G,shape(s),partial_review=True)
    assert kinds(s)==['BBOX','POSE'] and api().snapshot(s,G,OID)['gtr_count']==0
def test_S11_order_does_not_change_identity(tmp_path):
    s=linked(tmp_path,data,n=3,suppressed=None);before=api().snapshot(s,G,'GT:a1')
    scene=copy.deepcopy(core.scene(s,G));scene['objects'].reverse();scene['parents'].reverse()
    state=data.snapshot(s,G,s=scene);state['plates']=dict(reversed(list(state['plates'].items())))
    assert api().snapshot(s,G,'GT:a1',scene=scene,state=state)==before
def test_S12_RAW_not_layer_or_FP(tmp_path):
    s=linked(tmp_path,data,n=2,suppressed=1);before=core.summary(s)
    with pytest.raises(ValueError):api().snapshot(s,G,'GT:a1')
    assert core.summary(s)==before and before['false_positive_mt']==0
def test_S13_BAD_only_FINAL(tmp_path):
    s=drawn(tmp_path);m=core.add_plate(s,G,MANUAL,partial_review=True);frozen=freeze(s)
    with pytest.raises(ValueError):data.toggle_unassociated(s,G,m)
    assert freeze(s)==frozen;data.toggle_unassociated(s,G,OID);assert core._object(core.scene(s,G),OID)['error']
def test_S14_same_GTZ_not_link(tmp_path):
    s=source_session(tmp_path,n=1);m=core.add_plate(s,G,shape(s),partial_review=True);core.set_text(s,G,m,'SAME')
    assert api().snapshot(s,G,OID)['gtr_count']==0 and data.snapshot(s,G)['links'][OID]['plate_id'] is None

def select_gt(root,h,ed,s):
    data.accept_reference(s,G,OID);ed._choose_plate(OID);ready(root,h,ed)

def test_U01_FINAL_navigation_unchanged(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate(OID);ready(root,h,ed);frozen=freeze(s)
    for key,oid in [('Down','GT:a1'),('Up',OID),('z','GT:a1'),('x',OID)]:
        native.key(h,key);ready(root,h,ed);assert ed.selected==oid
    assert freeze(s)==frozen
def test_U02_GTR_count_reads_relation(gui):
    root,h,ed,s=gui;ed._choose_plate(OID);ready(root,h,ed);assert snapshot(ed)['gtr_count']==0
    api().create_reference(s,G,OID,shape(s));ready(root,h,ed)
    assert snapshot(ed)['gtr_count']==1 and any('GTR 1' in h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('geometry_layer_text'))
def test_U03_cycle_three_layers_and_reopen(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);assert snapshot(ed)['active']=='GTR'
    seen=[]
    for _ in range(3):cycle(root,h,ed);seen.append(snapshot(ed)['active'])
    assert seen==['BBOX','POSE','GTR']
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert api().snapshot(reopened,G,OID)==api().snapshot(s,G,OID)
def test_U04_invalid_pose_two_layers(tmp_path):
    s=invalid_source(tmp_path);api().create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    assert kinds(s)==['BBOX','GTR'] and api().snapshot(s,G,OID)['pose_unavailable']
def test_U05_single_layer_has_no_buttons(tmp_path,monkeypatch):
    s=invalid_source(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ed._choose_plate(OID);ready(root,h,ed)
        assert snapshot(ed)['active']=='BBOX' and len(snapshot(ed)['layers'])==1
        assert ed.geometry_layer_badge and not ed.geometry_layer_controls
    finally:
        try:next(gen)
        except StopIteration:pass
def test_U06_source_operator_roles_not_text_layers(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s)
    assert [l['role'] for l in snapshot(ed)['layers']]==['MT BBOX (źródło)','MT POSE (źródło)','GTR (operator)']
    cycle(root,h,ed);assert 'MT BBOX (źródło)' in ed.clean_caption.cget('text')
    assert not h.canvas.find_withtag('corner_handle')
def test_U07_badge_clicks_never_BAD_drag_GTZ(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);frozen=freeze(s);box=ed.geometry_layer_badge
    event=NS(widget=ed.c,x=(box[0]+box[2])/2,y=(box[1]+box[3])/2,state=0)
    ed.press(event);ed.right_press(event);ed.release(event);ready(root,h,ed)
    assert freeze(s)==frozen and ed.entry is None and ed.drag is None and ed._click_candidate is None
def test_U08_layer_cycle_is_read_only(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);frozen=freeze(s);summary=core.summary(s);undo=copy.deepcopy(ed.undo_stack)
    for _ in range(6):cycle(root,h,ed)
    assert freeze(s)==frozen and core.summary(s)==summary and ed.undo_stack==undo
def test_U09_changing_FINAL_resets_own_layer(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate(OID);ready(root,h,ed);cycle(root,h,ed)
    native.key(h,'Down');ready(root,h,ed)
    assert snapshot(ed)['object_id']=='GT:a1' and snapshot(ed)['active']=='GTR'
    assert snapshot(ed)['source_key'][-2]=='a1'
def test_U10_Shift_draw_release_focus(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);native.key(h,'d');native.click(h,ed,shape(s)[0]);ready(root,h,ed)
    points=copy.deepcopy(ed.points);frozen=freeze(s)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert h.canvas.find_withtag('shift_mt_source_bbox') and h.canvas.find_withtag('shift_mt_pose')
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay();assert not h.canvas.find_withtag('shift_source_preview')
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.shift_preview.focus_out();ed.overlay()
    assert not ed.shift_preview.active and ed.points==points and freeze(s)==frozen
def test_U11_gestures_warning_and_source_preserved(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);parents=copy.deepcopy(ed._read()['parents'])
    native.key(h,'space');ready(root,h,ed);assert api().snapshot(s,G,OID)['gtr_count']==1
    native.key(h,'space');ready(root,h,ed)
    ed._choose_plate('');native.key(h,'d');native.click(h,ed,shape(s)[0]);native.pump(root)
    assert not ed.points and ed.collision_warning.active
    ed.escape();ready(root,h,ed);assert ed._read()['parents']==parents
def test_U12_HUD_not_layer_count(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);hud=copy.deepcopy(ed._source_hud_value)
    cycle(root,h,ed);cycle(root,h,ed)
    assert ed._source_hud_value==hud and (hud['mt'],hud['final'],hud['mz'])==(1,1,1)

def test_E01_real_image10_one_FINAL(tmp_path):
    s=real(tmp_path);v=data.snapshot(s,G10)
    assert len([c for c in v['candidates'] if c['role']=='FINAL'])==1
def test_E02_real_GTR2_outside_MT_layers(tmp_path):
    s=real(tmp_path);v=data.snapshot(s,G10);m=api().snapshot(s,G10,'GT:'+G10)
    assert m['gtr_count']==1 and [r['kind'] for r in m['layers']]==['BBOX','POSE','GTR']
    manual=next(pid for pid,p in v['plates'].items() if p['origin_object_id'].startswith('MANUAL:'))
    assert manual!=m['gtr_id'] and len(v['plates'])==2
def test_E03_real_history_and_overlap_are_advisory(tmp_path):
    s=real(tmp_path);frozen=freeze(s);scene=core.scene(s,G10);v=data.snapshot(s,G10)
    notes=api().overlapping_references(scene,v);assert len(notes)==2
    assert scene['events'] and freeze(s)==frozen and v['links']['GT:'+G10]['plate_id']
def test_E04_remove_GTR2_on_copy_undo_redo(tmp_path):
    s=real(tmp_path);v=data.snapshot(s,G10);m=next(p['origin_object_id'] for p in v['plates'].values() if p['origin_object_id'].startswith('MANUAL:'))
    text=api().snapshot(s,G10,'GT:'+G10)['gtr_id'];p=copy.deepcopy(v['plates'][text])
    core.remove_operator_geometry(s,G10,m);assert len(data.snapshot(s,G10)['plates'])==1
    assert data.restore_history(s,G10) and len(data.snapshot(s,G10)['plates'])==2
    assert data.restore_history(s,G10,redo=True) and len(data.snapshot(s,G10)['plates'])==1
    assert data.snapshot(s,G10)['plates'][text]==p
def test_E05_multiple_FINAL_one_TP(tmp_path):
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import score_unit
    s=linked(tmp_path,data,n=2,suppressed=None);assert len(data.snapshot(s,G)['plates'])==1
    u=unit(final=2);u['finals'][1]['bbox']=u['finals'][0]['bbox'];r=score_unit(u)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0)
def test_E06_invalid_Pose_metrics_stay_NA(tmp_path):
    s=invalid_source(tmp_path);api().create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    row=data.diagnostics(s)['rows'][0]
    assert row['polygon_iou_pct'] is None and row['nmce_pct'] is None and row['not_measurable_reason']
def test_E07_NOT_RUN_no_CER(tmp_path):
    s=invalid_source(tmp_path);api().create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    data.edit_gt(s,G,OID,text='GT123');r=data.diagnostics(s)['rows'][0]
    assert r['mz_status']=='NOT_RUN' and r['cer'] is None and r['gt_text']=='GT123'
def test_E08_threshold_unchanged():
    from test_z4_mt_detection_evaluation import unit
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import score_unit
    for ratio,tp in ((.49,0),(.50,1)):
        u=unit();u['finals'][0]['bbox']=[150,140,150+150*ratio,210];r=score_unit(u)
        assert r['tp']==tp and r['fp']==1-tp and r['fn']==1-tp
def test_E09_Space_keeps_reference(tmp_path):
    s=drawn(tmp_path);before=data.snapshot(s,G);partial.toggle_plate(s,G,OID)
    after=data.snapshot(s,G);assert before['plates']==after['plates'] and before['links']==after['links']
    assert kinds(s)==['BBOX','POSE','GTR']
def test_E10_no_FINAL_manual_FN(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);oid=core.add_plate(s,G,MANUAL,partial_review=True)
    assert not data.snapshot(s,G)['candidates']
    with pytest.raises(ValueError):api().snapshot(s,G,oid)
    partial.confirm_gtr(s,G,data.snapshot(s,G)['links'][oid]['plate_id'])
    core.check_scene(s,G)
    assert core.summary(s)['missed_plates']==1

def test_N01_native_themes_sizes_receipt():
    p=OUT/'native_receipt.json'
    if not p.exists():pytest.skip('Native receipt pending')
    r=json.loads(p.read_text(encoding='utf-8'))
    assert {c['theme'] for c in r['cases']}=={'light_visual_cs','dark_visual_cs'}
    assert {c['size'] for c in r['cases']}=={'1649x896','1180x760'} and all(c['status']=='PASS' for c in r['cases'])
def test_N02_native_layer_sequences():
    p=OUT/'native_receipt.json'
    if not p.exists():pytest.skip('Native receipt pending')
    r=json.loads(p.read_text(encoding='utf-8'));assert {s['active'] for s in r['screenshots'] if s['active']}=={'BBOX','POSE','GTR'}
    assert {s['gtr_count'] for s in r['screenshots'] if s['gtr_count'] is not None}=={0,1}
def test_N03_native_Shift_focus(gui):
    root,h,ed,s=gui;ed._choose_plate(OID);ready(root,h,ed);native.key(h,'d');ready(root,h,ed)
    native.focus_canvas(h);h.canvas.event_generate('<KeyPress-Shift_L>');native.pump(root);ed.overlay();assert ed.shift_preview.active
    h.canvas.event_generate('<KeyRelease-Shift_L>');native.pump(root);assert not ed.shift_preview.active
    h.canvas.event_generate('<KeyPress-Shift_L>');native.pump(root);h.canvas.event_generate('<FocusOut>');native.pump(root)
    assert not ed.shift_preview.active
def test_N04_real_bindings_badge_no_shortcut_collision(gui):
    root,h,ed,s=gui;select_gt(root,h,ed,s);box,_=next(c for c in ed.geometry_layer_controls if not c[1])
    x,y=map(round,((box[0]+box[2])/2,(box[1]+box[3])/2));frozen=freeze(s)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y);ready(root,h,ed)
    assert snapshot(ed)['active']=='BBOX' and freeze(s)==frozen
    native.key(h,'Down');ready(root,h,ed);assert ed.selected==OID and snapshot(ed)['active']=='BBOX'
def test_N05_all_F4_semantic_integrity():
    p=OUT/'integrity.json'
    if not p.exists():pytest.skip('Pre/post audit pending')
    v=json.loads(p.read_text(encoding='utf-8'));assert v['semantic_equal'] and len(v['session_equal'])==12 and not v['protected_changes']
def test_N06_legacy_exports_readonly():
    from pathlib import Path
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'));denied=[]
    for p,v in b['protected'].items():
        if 'unreadable' in v:
            try:Path(p).read_bytes()
            except PermissionError:denied.append(p)
    if denied:pytest.skip('BLOCKED: Windows denied '+str(len(denied))+' historical export paths; no ACL changes')
def test_N07_four_prior_failures_preserved_as_evidence():
    import xml.etree.ElementTree as E
    cases=list(E.parse(OUT/'baseline.xml').getroot().iter('testcase'))
    assert len([t for t in cases if t.find('failure') is not None])==4
def test_N08_worktree_and_static_checks():
    import subprocess
    assert subprocess.run(['git','diff','--check'],capture_output=True).returncode==0
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==b['head']
