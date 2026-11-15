"""U01–U34: one descriptive navigator; UI changes cannot write research data."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_mt_geometry_layers import source_session,invalid_source
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_clean_view as clean
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data,mobile_mt_geometry_layers as evidence,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
OUT=Path(__file__).resolve().parents[1]/'output/Z4_ONE_NAVIGATOR_ZX_20261028_001'
G='mt-one';OID='GT:a0';F='exp-1792699303016-c8b05efb'

def frozen(s):return copy.deepcopy(s.review.to_dict()),s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None
def pick(root,h,ed,kind):assert nav.select_layer(ed,kind);ready(root,h,ed)
def keys(root,h,ed,key):native.key(h,key);ready(root,h,ed)
def render_quad(ed):
    return [(ed.c.gettags(i),ed.c.coords(i)) for i in ed.c.find_withtag('navigator_geometry') if ed.selected in ed.c.gettags(i)]
def available(ed):return [r['kind'] for r in layers.model(ed)['layers']]

@pytest.fixture
def gui(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=1);evidence.create_reference(s,G,OID,core.scene(s,G)['parents'][0]['quad']);data.edit_gt(s,G,OID,text='WA12345')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.fixture
def bbox(tmp_path,monkeypatch):
    s=invalid_source(tmp_path/'data');evidence.create_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]])
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.fixture
def multiple(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=3);evidence.create_reference(s,G,OID,core.scene(s,G)['parents'][0]['quad'])
    data.associate(s,G,'GT:a1',OID)  # a2 intentionally has no linked GTR.
    monkeypatch.setattr(native,'source_session',lambda _:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def test_U01_heading_result(gui):
    root,h,ed,s=gui;ready(root,h,ed)
    assert ed.clean_caption.cget('text')==nav.TITLE and ed.navigator.result.cget('text')=='Wynik 1 z 1'
def test_U02_full_layer_names(gui):
    root,h,ed,s=gui;ready(root,h,ed)
    assert [ed.navigator.menu.entrycget(i,'label') for i in range(3)]==['Prostokąt modelu (BBOX)','Narożniki modelu (Pose)','Obrys operatora (GTR 1)']
def test_U03_no_old_canvas_navigators(gui):
    root,h,ed,s=gui;ready(root,h,ed)
    assert not any(h.canvas.find_withtag(t) for t in ('mt_stack_control','geometry_layer_badge','mt_relation_badge'))
def test_U04_single_selected_badge(gui):
    root,h,ed,s=gui
    for kind in available(ed):
        pick(root,h,ed,kind)
        # Only the selected geometry gets a compact two-line Canvas badge.
        assert len(h.canvas.find_withtag('context_badge_background'))==1
        assert len(h.canvas.find_withtag('context_badge_title'))==len(h.canvas.find_withtag('context_badge_detail'))==1
        assert not h.canvas.find_withtag('badge4_background')
        assert ed.navigator.selection.get()==kind and len(render_quad(ed))==2
def test_U05_BBOX_from_menu_is_read_only(gui):
    root,h,ed,s=gui;ed.navigator.menu.invoke(0);ready(root,h,ed)
    assert layers.model(ed)['active']=='BBOX' and not h.canvas.find_withtag('corner_handle')
    assert any('BBOX' in tags for tags,_ in render_quad(ed)) and h.canvas.find_withtag('operator_quad')
def test_U06_Pose_source_from_menu(gui):
    root,h,ed,s=gui;before=frozen(s);ed.navigator.menu.invoke(1);ready(root,h,ed)
    assert layers.model(ed)['active']=='POSE' and not h.canvas.find_withtag('corner_handle') and frozen(s)==before
    assert next(coords for tags,coords in render_quad(ed) if 'POSE' in tags)==[v for p in core.scene(s,G)['parents'][0]['quad'] for v in ed.to_canvas(p)]
def test_U07_GTR_editable(gui):
    root,h,ed,s=gui;ed.navigator.menu.invoke(2);ready(root,h,ed)
    assert len(h.canvas.find_withtag('corner_handle'))==4 and OID in ed.badges
def test_U08_each_choice_changes_geometry(gui):
    root,h,ed,s=gui;roles=[]
    for kind in ('BBOX','POSE','GTR','BBOX'):
        pick(root,h,ed,kind)
        roles.append([t for t in next(tags for tags,_ in render_quad(ed) if kind in tags) if t in ('BBOX','POSE','GTR')])
    assert roles==[['BBOX'],['POSE'],['GTR'],['BBOX']]
def test_U09_X_next_layer(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX')
    for expected in ('POSE','GTR','BBOX'):keys(root,h,ed,'x');assert layers.model(ed)['active']==expected
def test_U10_Z_previous_layer(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX')
    for expected in ('GTR','POSE','BBOX'):keys(root,h,ed,'z');assert layers.model(ed)['active']==expected
def test_U11_single_FINAL_still_cycles(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX');keys(root,h,ed,'X');assert layers.model(ed)['active']=='POSE'
    assert not ed.navigator.previous.winfo_ismapped() and not ed.navigator.following.winfo_ismapped()
def test_U12_focus_on_navigator_and_canvas(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX');ed.navigator.layer.focus_force();native.pump(root)
    ed.navigator.layer.event_generate('<KeyPress-x>');ed.navigator.layer.event_generate('<KeyRelease-x>');ready(root,h,ed)
    assert layers.model(ed)['active']=='POSE'
    native.focus_canvas(h);keys(root,h,ed,'x');assert layers.model(ed)['active']=='GTR'
    ed._choose_plate('');ready(root,h,ed);ed.navigator.result.invoke();ready(root,h,ed)
    assert ed.selected==OID and layers.model(ed)['active']=='BBOX'
def test_U13_typing_and_other_windows_do_not_navigate(gui):
    import tkinter as tk
    root,h,ed,s=gui;pick(root,h,ed,'GTR');ed.start_entry(OID);ed.entry.focus_force();native.pump(root)
    for key in ('Z','X','z','x'):ed.entry.event_generate('<KeyPress-'+key+'>')
    native.pump(root);assert layers.model(ed)['active']=='GTR' and 'ZXzx' in ed.entry_var.get()
    dialog=tk.Toplevel(h.window);text=tk.Text(dialog);text.pack();text.focus_force();native.pump(root)
    assert ed.stack_hotkey(NS(widget=text,keysym='x',state=0)) is None;dialog.destroy()
def test_U14_INVALID_QUAD_missing_Pose(bbox):
    root,h,ed,s=bbox;assert available(ed)==['BBOX','GTR']
    assert ed.navigator.menu.entrycget(1,'state')=='disabled' and 'INVALID_QUAD' in ed.navigator.hint.cget('text')
def test_U15_NOT_RUN_stays_NA(bbox):
    root,h,ed,s=bbox;pick(root,h,ed,'BBOX');context=clean.context(ed,{})
    assert 'NOT_RUN' in context['object'] and 'N/A' in context['object']
    assert data.diagnostics(s)['rows'][0]['cer'] is None
def test_U16_no_GTR_stays_unlinked(multiple):
    root,h,ed,s=multiple;nav.select_result(ed,2);ready(root,h,ed)
    assert available(ed)==['BBOX','POSE'] and 'brak referencji' in ed.navigator.hint.cget('text')
def test_U17_BBOX_only_noop(tmp_path,monkeypatch):
    s=invalid_source(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s);gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);before=frozen(s);keys(root,h,ed,'x');keys(root,h,ed,'z');assert available(ed)==['BBOX'] and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass
def test_U18_zero_FINAL_manual_available(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data',n=0,mz=False);oid=core.add_plate(s,G,[[650,350],[800,350],[800,420],[650,420]],partial_review=True)
    monkeypatch.setattr(native,'source_session',lambda _:s);gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        ready(root,h,ed);assert ed.navigator.result.cget('text')=='Brak wskazań modelu'
        ed._choose_plate(oid);ready(root,h,ed);assert len(h.canvas.find_withtag('corner_handle'))==4 and not nav.finals(ed)
    finally:
        try:next(gen)
        except StopIteration:pass
def test_U19_arrows_all_FINAL_ZX_layers(multiple):
    root,h,ed,s=multiple
    for oid in ('GT:a1','GT:a2',OID):keys(root,h,ed,'Down');assert ed.selected==oid
    pick(root,h,ed,'BBOX');keys(root,h,ed,'x');assert ed.selected==OID and layers.model(ed)['active']=='POSE'
    assert ed.navigator.result.cget('text')=='Wynik 1 z 3'
def test_U20_missing_layer_falls_back(multiple):
    root,h,ed,s=multiple;pick(root,h,ed,'GTR');nav.select_result(ed,2);ready(root,h,ed)
    assert ed.selected=='GT:a2' and layers.model(ed)['active']=='BBOX'
def test_U21_independent_GTR2_not_fourth_layer(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    side=tmp_path/'review.json';shutil.copyfile(OUT/'private/baseline.review.json',side);s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda _:s);gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen);gid=F+'-a00000010'
    try:
        iid=next(i for i in h.subject_tree.get_children('') if gid in subject_groups(s,h._subject_ids[i]));h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);ready(root,h,ed)
        assert len(available(ed))==3 and len(nav.finals(ed))==1
        manual=next(r for r in clean.records(ed) if not r['candidate']);ed._choose_plate(manual['oid']);ready(root,h,ed)
        assert layers.model(ed) is None and len(h.canvas.find_withtag('corner_handle'))==4 and ed.navigator.result.cget('text')=='Wybór ręczny'
    finally:
        try:next(gen)
        except StopIteration:pass
def test_U22_navigation_does_not_create_GT(multiple):
    root,h,ed,s=multiple;before=copy.deepcopy(data.snapshot(s,G)['plates']);nav.select_result(ed,2);ready(root,h,ed);keys(root,h,ed,'x')
    assert data.snapshot(s,G)['plates']==before and layers.model(ed)['gtr_count']==0
def test_U23_navigation_no_review_write(gui):
    root,h,ed,s=gui;before=frozen(s);history=copy.deepcopy(ed.undo_stack)
    for key in ('x','z','X','Z','Down','Up'):keys(root,h,ed,key)
    assert frozen(s)==before and ed.undo_stack==history
def test_U24_Space_keeps_reference(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');before=data.snapshot(s,G);keys(root,h,ed,'space');after=data.snapshot(s,G)
    assert before['plates']==after['plates'] and before['links']==after['links'] and 'WA12345' in clean.context(ed,{})['object']
def test_U25_PPM_only_model_source(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');q=nav.quad(layers.model(ed));point=tuple(sum(p[i] for p in q)/4 for i in (0,1));before=frozen(s)
    native.click(h,ed,point,3);ready(root,h,ed);assert frozen(s)==before
    pick(root,h,ed,'BBOX');point=tuple(sum(p[i] for p in nav.quad(layers.model(ed)))/4 for i in (0,1));native.click(h,ed,point,3);ready(root,h,ed)
    assert core._object(ed._read(),OID)['error']
    assert len(render_quad(ed))==2 and len(h.canvas.find_withtag('fp_diagonal'))==2
def test_U26_D_from_source_preserves_MT(gui):
    root,h,ed,s=gui;pick(root,h,ed,'POSE');parents=copy.deepcopy(ed._read()['parents']);pids=set(data.snapshot(s,G)['plates'])
    keys(root,h,ed,'d');assert ed.draw_target['kind']=='replace';ed.escape();ready(root,h,ed)
    assert ed._read()['parents']==parents and set(data.snapshot(s,G)['plates'])==pids
def test_U27_GTR_drag_history_R(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');q=copy.deepcopy(core._object(ed._read(),OID)['quad']);parents=copy.deepcopy(ed._read()['parents'])
    x,y=map(round,ed.to_canvas(q[0]));h.canvas.event_generate('<ButtonPress-1>',x=x,y=y);h.canvas.event_generate('<B1-Motion>',x=x+3,y=y+2);h.canvas.event_generate('<ButtonRelease-1>',x=x+3,y=y+2);ready(root,h,ed)
    assert ed._read()['parents']==parents and core._object(ed._read(),OID)['quad']!=q
    ed.undo();ready(root,h,ed);assert core._object(ed._read(),OID)['quad']==q
    ed.undo(redo=True);ready(root,h,ed);ed.escape();pick(root,h,ed,'GTR');ed.remove_selected();ready(root,h,ed);assert len(data.snapshot(s,G)['plates'])==1
def test_U28_Shift_hold_release_focus(gui):
    root,h,ed,s=gui;pick(root,h,ed,'BBOX');before=frozen(s)
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.overlay();assert h.canvas.find_withtag('shift_source_preview')
    ed.shift_preview.release(NS(keysym='Shift_L'));ed.overlay();assert not h.canvas.find_withtag('shift_source_preview')
    ed.shift_preview.press(NS(widget=ed.c,keysym='Shift_L'));ed.shift_preview.focus_out();ed.overlay();assert layers.model(ed)['active']=='BBOX' and frozen(s)==before
def test_U29_guides_collision_unchanged(gui):
    root,h,ed,s=gui;pick(root,h,ed,'GTR');q=core._object(ed._read(),OID)['quad'];ed._choose_plate('');keys(root,h,ed,'d');native.click(h,ed,q[0]);native.pump(root)
    assert not ed.points and ed.collision_warning.active and ed.collision_warning.phase in ('red','normal')
    from auto_annotation_tool.gui import mobile_canvas_guides as g
    assert '#165D8C' in Path(g.__file__).read_text(encoding='utf-8') and '#EAF5FC' in Path(g.__file__).read_text(encoding='utf-8')
def test_U30_panel_follows_role_and_retains_full_summary(gui):
    root,h,ed,s=gui
    for kind in ('BBOX','POSE'):
        pick(root,h,ed,kind);context=clean.context(ed,{})
        assert kind.upper() in context['object_title'].upper() and 'GTZ' in context['object'] and 'WA12345' in context['object']
        assert 'GTZ' not in ed._context_badge['detail']
    pick(root,h,ed,'GTR');assert 'WA12345' in clean.context(ed,{})['object']
def test_U31_HUD_session_numbers_stable(gui):
    root,h,ed,s=gui;before=copy.deepcopy(ed._source_hud_value);summary=core.summary(s)
    keys(root,h,ed,'x');keys(root,h,ed,'z');assert ed._source_hud_value==before and core.summary(s)==summary
def test_U32_native_receipt():
    p=OUT/'native_receipt.json'
    if not p.exists():pytest.skip('Native acceptance pending')
    r=json.loads(p.read_text(encoding='utf-8'));assert len(r['cases'])>=8 and all(c['status']=='PASS' for c in r['cases'])
    assert len({c['dpi'] for c in r['cases']})>=2
def test_U33_zoom_no_pan_and_label_clear(gui):
    root,h,ed,s=gui;h._view_zoom=2;h._paint_image();ready(root,h,ed);center=h._view_center
    for kind in ('POSE','GTR','BBOX'):
        pick(root,h,ed,kind);assert h._view_center==center and h._view_zoom==2
        for box in ed.clean_label_boxes:assert all(not clean.intersects(box,q) for q in ed.clean_geometry_boxes)
def test_U34_exports_and_aggregates_unchanged(gui,tmp_path):
    root,h,ed,s=gui;before=core.summary(s);core.export(s,tmp_path/'before');keys(root,h,ed,'x');keys(root,h,ed,'z');core.export(s,tmp_path/'after')
    assert core.summary(s)==before
    assert {p.name:p.read_bytes() for p in (tmp_path/'before').iterdir()}=={p.name:p.read_bytes() for p in (tmp_path/'after').iterdir()}
