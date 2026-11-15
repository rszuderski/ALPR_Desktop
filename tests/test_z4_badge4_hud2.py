"""H01–H26: UI projection only; exact counts and immutable source records."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,importlib,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_one_navigator import gui,multiple,pick,keys,frozen
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_canvas_source_hud import frame_execution
OUT=Path(__file__).resolve().parents[1]/'output/Z4_BADGE4_HUD2_20261028_001'
F='exp-1792699303016-c8b05efb';G='mt-one';OID='GT:a0'
def ui():return importlib.import_module('auto_annotation_tool.gui.mobile_canvas_info_badge')
def rows(ed):return ui().badge_rows(ed)
def texts(ed):return [v for _,v in rows(ed)]

@pytest.fixture
def real_gui(tmp_path,monkeypatch):
    side=tmp_path/'private.review.json';shutil.copyfile(OUT/'private/baseline.review.json',side)
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.fixture
def bbox(tmp_path,monkeypatch):
    # The older geometry-only invalid_source fixture deliberately omits run
    # counters. HUD acceptance needs actual complete source execution proof.
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
                       plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    from auto_annotation_tool.ranking.mobile_mt_geometry_layers import create_reference
    create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_H01_four_fields_in_panel_one_context_badge(real_gui):
    root,h,ed,s=real_gui;ready(root,h,ed)
    assert [k for k,v in rows(ed)]==['GTR 1','Model tablic MT 1','GTZ','Model znaków MZ']
    assert len(h.canvas.find_withtag('context_badge_background'))==1 and not h.canvas.find_withtag('badge4_background')
    assert all(name+': '+value in clean.context(ed,{})['object'] for name,value in rows(ed))
def test_H02_real_reference_not_prediction(real_gui):
    root,h,ed,s=real_gui;ready(root,h,ed);v=texts(ed)
    assert v[2]=='DWR8339U' and v[3]=='DNR8339U'
def test_H03_actual_bbox_FP(real_gui):
    root,h,ed,s=real_gui;assert texts(ed)[1]=='BBOX + Pose · FP'
def test_H04_exact_two_HUD_lines(real_gui):
    root,h,ed,s=real_gui;ready(root,h,ed)
    assert ed.source_hud.lines==['Model tablic MT — uruchomień: 1/klatka','Model znaków MZ — uruchomień: 1/klatka']
def test_H05_no_FINAL_in_HUD(real_gui):
    root,h,ed,s=real_gui;ready(root,h,ed);text='\n'.join(ed.source_hud.lines)
    assert all(w not in text for w in ('FINAL','MT wyw.','wskazań końcowych','wywołano raz'))
    assert ed._source_hud_value['final']==1
def test_H06_no_recorded_MT_is_zero(tmp_path,monkeypatch):
    def change(rows,trace,payload):
        rows[0].update(mt_executed='false',mt_status='NOT_RUN',mt_detection_count=0,mt_detection_index='',mz_status='NOT_RUN',mz_executed='false')
        trace['counters'].update(mt_runs_this_frame=0,plate_detections_after_dedup=0,mz_runs=0)
    s=fixture_archive(tmp_path,mutate=change);v=frame_execution(s,G);before=frozen(s)
    assert ui().hud_lines(v)[0]=='Model tablic MT — uruchomień: 0/klatka' and v['mt']==0 and frozen(s)==before
    # Absence of registered MZ is a UI zero, never a rewrite of UNKNOWN.
    absent=dict(v,mt=1,mz=None,reason='brak wiarygodnego śladu',mz_execution_ids=[])
    assert ui().hud_lines(absent)[1].endswith('0/klatka') and absent['mz'] is None
    for bad in (dict(v,frame=None),dict(v,reason='sprzeczny ślad MZ')):
        with pytest.raises(ValueError):ui().hud_lines(bad)
    # A source-load error uses the existing status path, never invented zeroes.
    shown=[];messages=[]
    hud=NS(display=lambda lines:shown.append(lines),unavailable=lambda:shown.append('hidden'))
    h=NS(session=s,current_invocation=G,_image=object(),_image_entry=v['frame']['entry'],_image_source_size=v['frame']['size'])
    ed=NS(h=h,source_hud=hud,message=messages.append)
    def broken():raise OSError('test: source unreadable')
    monkeypatch.setattr(s,'verify_source',broken);ui().update_hud(ed)
    assert shown==['hidden'] and 'source unreadable' in messages[-1]
def test_H07_NOT_RUN_zero_and_badge(bbox):
    root,h,ed,s=bbox;ready(root,h,ed)
    assert texts(ed)[3]=='nie uruchomiono' and ed.source_hud.lines[1].endswith('0/klatka')
    assert data.diagnostics(s)['rows'][0]['cer'] is None
def test_H08_invalid_pose_not_replaced_by_GT(bbox):
    root,h,ed,s=bbox;assert 'BBOX' in texts(ed)[1] and 'brak poprawnego Pose' in texts(ed)[1]
    assert [l['kind'] for l in layers.model(ed)['layers']]==['BBOX','GTR']
    assert data.diagnostics(s)['rows'][0]['nmce_pct'] is None
def test_H09_executed_empty_text(tmp_path,monkeypatch):
    def change(rows,trace,payload):rows[0].update(mz_status='NO_CHARACTERS',prediction='',mz_executed='true')
    s=fixture_archive(tmp_path/'data',mutate=change);s.set_review_mode('assisted');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);assert texts(ed)[3]=='brak odczytu' and ed.source_hud.lines[1].endswith('1/klatka')
        uncertain=copy.deepcopy(clean.selected(ed));uncertain['candidate'].update(mz_status='UNKNOWN',mz_prediction='AMBIGUOUS')
        monkeypatch.setattr(clean,'selected',lambda _:uncertain)
        assert texts(ed)[3]=='wynik niejednoznaczny: AMBIGUOUS'
    finally:
        try:next(gen)
        except StopIteration:pass
def test_H10_empty_GTZ_not_MZ(multiple):
    root,h,ed,s=multiple;nav.select_result(ed,2);ready(root,h,ed)
    assert texts(ed)[2]=='nie wpisano' and texts(ed)[3]=='WA21234'
def test_H11_unreadable_reference(gui):
    root,h,ed,s=gui;data.edit_gt(s,G,OID,text='',reason='nieczytelna');ready(root,h,ed)
    assert texts(ed)[2]=='nieczytelny' and texts(ed)[3]=='WA01234'
def test_H12_no_GTR_no_new_GT(multiple):
    root,h,ed,s=multiple;nav.select_result(ed,2);ready(root,h,ed);before=frozen(s)
    assert texts(ed)[0]=='brak obrysu' and frozen(s)==before
def test_H13_independent_manual_reference(gui):
    root,h,ed,s=gui;manual=core.add_plate(s,G,[[650,350],[800,350],[800,420],[650,420]],partial_review=True)
    ed._choose_plate(manual);ready(root,h,ed);v=texts(ed)
    assert v[1]=='brak powiązanej detekcji' and v[3]=='brak powiązanego odczytu'
    assert 'TP' not in v[1] and 'FP' not in v[1]
def test_H14_image10_manual_does_not_link(real_gui):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    root,h,ed,s=real_gui;gid=F+'-a00000010';iid=next(i for i in h.subject_tree.get_children('') if gid in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);ready(root,h,ed)
    before=frozen(s);manual=next(r for r in clean.records(ed) if not r['candidate']);ed._choose_plate(manual['oid']);ready(root,h,ed)
    assert texts(ed)[1]=='brak powiązanej detekcji' and len(h.canvas.find_withtag('context_badge_background'))==1 and frozen(s)==before
def test_H15_same_summary_data_all_layers(real_gui):
    root,h,ed,s=real_gui;before=rows(ed)
    for kind in ('BBOX','POSE','GTR'):
        pick(root,h,ed,kind);assert rows(ed)==before and ed._context_badge['kind']==kind
        assert all(name+': '+value in clean.context(ed,{})['object'] for name,value in rows(ed))
def test_H16_keyboard_and_menu_still_readonly(gui):
    root,h,ed,s=gui;before=frozen(s);pick(root,h,ed,'BBOX')
    keys(root,h,ed,'x');assert layers.model(ed)['active']=='POSE'
    ed.navigator.menu.invoke(2);ready(root,h,ed);assert layers.model(ed)['active']=='GTR' and frozen(s)==before
def test_H17_different_FINAL_own_MZ(multiple):
    root,h,ed,s=multiple;assert texts(ed)[3]=='WA01234';keys(root,h,ed,'Down')
    assert ed.selected=='GT:a1' and texts(ed)[3]=='WA11234'
def test_H18_many_FINAL_one_MT(multiple):
    root,h,ed,s=multiple;ready(root,h,ed);assert len(nav.finals(ed))==3
    assert ed.source_hud.lines[0].endswith('1/klatka') and ed.source_hud.lines[1].endswith('3/klatka')
def test_H19_MZ_execution_ID_dedup(tmp_path):
    def change(rows,trace,payload):
        trace['counters'].pop('mz_runs')
        for row in rows:row.update(mz_execution_id='same-execution',mz_executed='true')
    s=fixture_archive(tmp_path,n=2,mutate=change);v=frame_execution(s,G)
    assert v['mz']==1 and len(v['mz_execution_ids'])==1 and ui().hud_lines(v)[1].endswith('1/klatka')
    def distinct(rows,trace,payload):
        trace['counters'].pop('mz_runs')
        for i,row in enumerate(rows):row.update(mz_execution_id='execution-'+str(i),mz_executed='true')
    other=fixture_archive(tmp_path/'distinct',n=2,mutate=distinct);v=frame_execution(other,G)
    assert v['mz']==2 and ui().hud_lines(v)[1].endswith('2/klatka')
def test_H20_space_keeps_reference(real_gui):
    root,h,ed,s=real_gui;before=rows(ed);keys(root,h,ed,'space');after=rows(ed)
    assert after[0]==before[0] and after[2]==before[2] and after[3]==before[3] and 'oczekuje' in after[1][1]
def test_H21_text_geometry_refresh(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');before=copy.deepcopy(ed._read()['parents'])
    box=ed.badges[OID];x,y=round((box[0]+box[2])/2),round((box[1]+box[3])/2)
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-1>',x=x,y=y);native.pump(root)
    assert ed.entry is not None;ed.entry_var.set('CHANGED');ed.entry_enter();ready(root,h,ed)
    assert texts(ed)[2]=='CHANGED' and texts(ed)[3]=='WA01234'
    q=copy.deepcopy(core._object(ed._read(),OID)['quad']);q[0][0]+=2;data.edit_gt(s,G,OID,quad=q);ready(root,h,ed)
    assert texts(ed)[2]=='CHANGED' and ed._read()['parents']==before
def test_H22_D_Shift_badge_hidden(gui):
    root,h,ed,s=gui;keys(root,h,ed,'d');assert ed.mode=='draw'
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay()
    assert not h.canvas.find_withtag('context_badge_background') and h.canvas.find_withtag('shift_source_preview')
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.escape();ready(root,h,ed);nav.select_result(ed,0);ready(root,h,ed)
    assert h.canvas.find_withtag('context_badge_background')
def test_H23_badge_click_is_not_BAD_and_history(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');before=frozen(s)
    box=ed._context_badge['rect'];x,y=round(box[0]+4),round(box[1]+4)
    h.canvas.event_generate('<ButtonPress-3>',x=x,y=y);h.canvas.event_generate('<ButtonRelease-3>',x=x,y=y);ready(root,h,ed)
    assert frozen(s)==before
    data.edit_gt(s,G,OID,text='NEXT');ready(root,h,ed);ed.undo();ready(root,h,ed);assert texts(ed)[2]=='WA12345'
    ed.undo(redo=True);ready(root,h,ed);assert texts(ed)[2]=='NEXT'
def test_H24_native_layout_receipt():
    p=OUT/'native_receipt.json'
    if not p.exists():pytest.skip('Native acceptance pending')
    r=json.loads(p.read_text(encoding='utf-8'));assert len(r['cases'])==4 and all(c['status']=='PASS' for c in r['cases'])
    assert {c['size'] for c in r['cases']}=={'1649x896','1180x760'}
def test_H25_contrast_and_long_text(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');long='LONG1234567890'*12;data.edit_gt(s,G,OID,text=long);ready(root,h,ed)
    assert rows(ed)[2][1]==long and long in clean.context(ed,{})['object']
    assert ed._context_badge['displayed'][1].endswith('…') and long not in ed._context_badge['displayed'][1]
    for box in ed.clean_label_boxes:assert all(not clean.intersects(box,q) for q in ed.clean_geometry_boxes)
    for dark in (False,True):
        p=ui().badge_palette(dark)
        assert ui().contrast(p['name'],p['bg'])>=4.5 and ui().contrast(p['fg'],p['bg'])>=4.5
def test_H26_F4_unchanged():
    p=OUT/'integrity.json'
    if not p.exists():pytest.skip('Post-change audit pending')
    r=json.loads(p.read_text(encoding='utf-8'));assert r['semantic_equal'] and len(r['session_equal'])==12 and not r['protected_changes']
