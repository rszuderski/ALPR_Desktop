"""Source-bound task navigation: transient Windows Tk UI, unchanged reviews."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import settled
from test_z4_mt_geometry_layers import source_session
from auto_annotation_tool.ranking import mobile_human_review as review,mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_detection_evaluation as mt
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers,mobile_canvas_review_tasks as tasks

OUT=Path(__file__).resolve().parents[1]/'output/Z4_LOCATE_PENDING_MT_20261101_001'
SID='exp-1792699864138-add3e282';G=SID+'-a00000008';GOOD='GT:'+G;BAD='GT:'+SID+'-a00000009'
def api():
    import importlib
    return importlib.import_module('auto_annotation_tool.gui.mobile_canvas_task_locator')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,default=str).encode()).hexdigest()
def frozen(s):return digest(s.review.to_dict()),hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()
def pure(s,gid='mt-one',selected='GT:a0'):
    sc=core.scene(s,gid);ed=NS(h=NS(session=s),selected=selected)
    return api().task_target(ed,tasks.for_scene(s,gid,sc),sc,stacks.snapshot(s,gid,sc))
def move(root,h,ed,s,gid):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry']);settled(root,h,ed)

@pytest.fixture
def actual(tmp_path):
    src=OUT/'private'/SID
    shutil.copyfile(src/'source.alprsession',tmp_path/'source.alprsession');shutil.copyfile(src/'review.json',tmp_path/'review.json')
    s=review.MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')
    # Current G0 is rev88: MT1 was already accepted. Only this private fixture
    # restores pending MT1, preserving the saved reference and BAD MT2.
    assert s.review.review_revision==88
    partial.toggle_plate(s,G,GOOD)
    assert partial.progress(core.scene(s,G))['object_decisions'][GOOD]=='UNRESOLVED'
    return s

@pytest.fixture
def gui(tmp_path,monkeypatch,actual):
    monkeypatch.setattr(native,'source_session',lambda _:actual)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    h.window.state('normal');h.window.geometry('1649x896+10+10');h.window.deiconify();root.update()
    move(root,h,ed,s,G)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass

def target(ed):return ed.contexts.fields['image']._target
def locate(gui):
    root,h,ed,s=gui;settled(root,h,ed)
    assert ed.task_locator.activate(target(ed));ready(root,h,ed)
    return ed.task_locator.signal

def test_L01_inline_action(gui):
    root,h,ed,s=gui;line=ed.contexts.fields['image']
    assert 'Do zrobienia' in line.cget('text')
    assert line.button.cget('text')=='Pokaż MT1' and line.button.winfo_ismapped()
    assert len(line.window_names())==1 and line.button.master is line

def test_L02_dynamic_MT2(tmp_path):
    s=source_session(tmp_path,n=2);text,value=pure(s,selected='GT:a1')
    assert value['index']==1 and value['attempt_id']=='a1' and 'MT 2' in text

def test_L03_GTZ_does_not_locate_MT(tmp_path):
    s=source_session(tmp_path);partial.confirm_stack(s,'mt-one',oid='GT:a0')
    sc=core.scene(s,'mt-one');t=tasks.pending_review_units(sc,stacks.snapshot(s,'mt-one'),issues=['Brak GTZ'])
    text,value=api().task_target(NS(h=NS(session=s),selected='GT:a0'),t,sc,stacks.snapshot(s,'mt-one'))
    assert 'GTZ' in text and value is None

def test_L04_scene_review_has_no_MT_action(tmp_path):
    s=source_session(tmp_path,mz=False);partial.confirm_stack(s,'mt-one',oid='GT:a0')
    sc=core.scene(s,'mt-one');sc['checked']=False;t=tasks.pending_review_units(sc,stacks.snapshot(s,'mt-one'),issues=[])
    text,value=api().task_target(NS(h=NS(session=s),selected='GT:a0'),t,sc,stacks.snapshot(s,'mt-one'))
    assert text=='Sprawdź cały obraz' and value is None

def test_L05_exact_ID_not_display_text(gui):
    root,h,ed,s=gui;t=copy.deepcopy(target(ed));t['attempt_id']=BAD[3:];before=frozen(s)
    assert not ed.task_locator.activate(t) and frozen(s)==before
    assert locate(gui)['target']['attempt_id']==G and ed.selected==GOOD

def test_L06_real_F4_target_and_BAD_unchanged(gui):
    root,h,ed,s=gui;before=frozen(s);assert nav.select_layer(ed,'POSE',BAD);ready(root,h,ed)
    v=locate(gui);p=partial.progress(core.scene(s,G))
    assert v['target']['attempt_id']==G and ed.selected==GOOD
    assert p['object_decisions'][BAD]=='NOT_PLATE' and p['object_decisions'][GOOD]=='UNRESOLVED' and frozen(s)==before

def test_L07_overlap_keeps_distinct_source_IDs(tmp_path):
    from test_z4_space_gtr_hud_shift import fixture_archive
    s=fixture_archive(tmp_path,n=2,mz=False);s.set_review_mode('assisted')
    a=pure(s,selected='GT:a0')[1];b=pure(s,selected='GT:a1')[1]
    assert a['attempt_id']=='a0' and b['attempt_id']=='a1' and a!=b

def test_L08_RAW_is_never_a_target(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    s=session_fixture(tmp_path,n=3,suppressed=2);sc=core.scene(s,'mt-one');state=stacks.snapshot(s,'mt-one');t=tasks.for_scene(s,'mt-one',sc)
    assert all(u['final_mt_id']!='a2' for u in t['units'])
    _,v=api().task_target(NS(h=NS(session=s),selected='GT:a2'),t,sc,state)
    assert v is None or v['attempt_id']!='a2'

def test_L09_layers_one_target(gui):
    root,h,ed,s=gui;locate(gui)
    assert {v['kind'] for v in layers.model(ed)['layers']}=={'BBOX','POSE','GTR'}
    assert len(ed.c.find_withtag('pending_mt_label'))==1
    assert {ed.c.gettags(i)[3] for i in ed.c.find_withtag('pending_mt_contour')}=={G}

def test_L10_no_GTR_created(tmp_path,monkeypatch):
    s=source_session(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        settled(root,h,ed);before=frozen(s);assert not stacks.snapshot(s,'mt-one')['plates']
        assert ed.task_locator.activate(target(ed));ready(root,h,ed)
        assert frozen(s)==before and not stacks.snapshot(s,'mt-one')['plates']
    finally:
        try:next(gen)
        except StopIteration:pass

def test_L11_untrusted_loaded_frame_rejected(gui):
    root,h,ed,s=gui;before=frozen(s);old=h._image_entry;h._image_entry='wrong_frame'
    try:assert not ed.task_locator.activate(target(ed)) and frozen(s)==before and ed.task_locator.signal is None
    finally:h._image_entry=old

@pytest.mark.parametrize('box',[(150,170,300,170),(150,170,150,170)])
def test_L12_zero_area_source_BBOX(tmp_path,monkeypatch,box):
    from test_z4_bbox_spatial_marker import source
    s=source(tmp_path/'data',bbox=box);monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        settled(root,h,ed);assert nav.select_layer(ed,'BBOX','GT:a0');ready(root,h,ed);before=frozen(s)
        assert ed.task_locator.activate(target(ed));ready(root,h,ed)
        assert ed.task_locator.signal['marker'] and ed.task_locator.signal['quad']==[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]
        assert ed.c.find_withtag(api().TAG) and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass

def test_L13_visible_preserves_view(gui):
    root,h,ed,s=gui;h.fit_image();ready(root,h,ed);before=(h._view_zoom,h._view_center)
    locate(gui);assert (h._view_zoom,h._view_center)==before

def test_L14_offscreen_fits_exact_source(gui):
    root,h,ed,s=gui;h._view_zoom=30;h._view_center=(0,0);h._paint_image();ready(root,h,ed)
    v=locate(gui)
    assert all(0<=ed.to_canvas(p)[0]<=ed.c.winfo_width() and 0<=ed.to_canvas(p)[1]<=ed.c.winfo_height() for p in v['quad'])
    assert ed.selected==GOOD and h._view_center!=(0,0)

def test_L15_hidden_INF_only_transient(gui):
    root,h,ed,s=gui;ed.inf_visible.set(False);ed.ref_visible.set(False);ed.show_fp_var.set(False);ed.sync();ready(root,h,ed)
    before=frozen(s);locate(gui)
    assert not ed.inf_visible.get() and not ed.ref_visible.get() and not ed.show_fp_var.get()
    assert ed.c.find_withtag(api().TAG);ed.task_locator.clear();ed.overlay()
    assert not ed.c.find_withtag(api().TAG) and frozen(s)==before

def test_L16_neutral_halo_label(gui):
    root,h,ed,s=gui;locate(gui);ids=ed.c.find_withtag('pending_mt_contour')
    assert len(ids)==3 and ed.c.itemcget(ids[-1],'outline')==api().COLOR
    assert [ed.c.itemcget(i,'text') for i in ed.c.find_withtag('pending_mt_label')]==['MT1 · do oceny']

def test_L17_expiry_and_disposal(gui):
    root,h,ed,s=gui;locate(gui);timer=ed.task_locator.after_id
    native.pump(root,lambda:ed.task_locator.signal is None,seconds=3)
    assert not ed.c.find_withtag(api().TAG) and timer not in ed.c.tk.call('after','info')
    locate(gui);ed.task_locator.clear();assert ed.task_locator.after_id is None

def test_L18_scene_change_clears(gui):
    root,h,ed,s=gui;locate(gui);move(root,h,ed,s,SID+'-a00000001')
    assert ed.task_locator.signal is None and not ed.c.find_withtag(api().TAG)

def test_L19_stale_pending_guard(gui):
    root,h,ed,s=gui;t=copy.deepcopy(target(ed));locate(gui);partial.confirm_stack(s,G,oid=GOOD);settled(root,h,ed)
    before=frozen(s);assert ed.task_locator.signal is None and not ed.task_locator.activate(t) and frozen(s)==before

def test_L20_repeat_replaces_timer(gui):
    root,h,ed,s=gui;before=frozen(s);locate(gui);old=ed.task_locator.after_id
    for _ in range(5):locate(gui)
    assert len(ed.c.find_withtag('pending_mt_label'))==1 and old not in ed.c.tk.call('after','info') and frozen(s)==before

def test_L21_navigation_gestures_remain_read_only(gui):
    root,h,ed,s=gui;locate(gui);before=frozen(s)
    for key in ('z','x','Shift_L','a','Escape'):
        native.key(h,key);ready(root,h,ed)
    assert frozen(s)==before and not ed.task_locator.signal

def test_L22_GTZ_entry_not_committed_or_blurred(gui):
    root,h,ed,s=gui;locate(gui);assert nav.select_layer(ed,'GTR',GOOD);ready(root,h,ed);ed.start_entry(GOOD);ready(root,h,ed)
    assert ed.entry is not None;ed.entry.delete(0,'end');ed.entry.insert(0,'DRAFTQA');ed.entry.focus_force();root.update()
    before=frozen(s);focus=h.window.focus_get();entry=ed.entry
    assert not ed.task_locator.activate(target(ed))
    ed.contexts.fields['image'].button.event_generate('<ButtonPress-1>');ed.contexts.fields['image'].button.event_generate('<ButtonRelease-1>');root.update()
    assert ed.entry is entry and entry.get()=='DRAFTQA' and h.window.focus_get() is focus and frozen(s)==before
    ed.escape();ready(root,h,ed)

@pytest.mark.parametrize('field,value',[('drag',('corner',0)),('mode','draw'),('points',[(1,1)]),('busy',True)])
def test_L23_busy_edit_untouched(gui,field,value):
    root,h,ed,s=gui;old=getattr(ed,field);setattr(ed,field,value);before=frozen(s)
    try:assert not ed.task_locator.activate(target(ed)) and getattr(ed,field)==value and frozen(s)==before
    finally:setattr(ed,field,old)

def test_L24_L25_L26_state_science_MZ_counters_unchanged(gui):
    root,h,ed,s=gui;before=frozen(s);science=digest(mt.evaluate_session(s));attempts=digest(s.attempts);hud=ed.source_hud.cget('text')
    counters={k:w.cget('text') for k,w in ed.contexts.counters.items()};assert counters['fp']=='≥1' and counters['tp']==counters['fn']=='—'
    locate(gui)
    assert frozen(s)==before and digest(mt.evaluate_session(s))==science and digest(s.attempts)==attempts and ed.source_hud.cget('text')==hud
    assert {k:w.cget('text') for k,w in ed.contexts.counters.items()}==counters

def test_L27_Enter_and_Space_on_action(gui):
    root,h,ed,s=gui;b=ed.contexts.fields['image'].button;before=frozen(s);b.focus_force();root.update()
    b.event_generate('<KeyPress-Tab>');root.update();assert h.window.focus_get() is not b
    b.focus_force();root.update()
    b.event_generate('<KeyPress-space>');b.event_generate('<KeyRelease-space>');root.update();assert frozen(s)==before and not ed.task_locator.signal
    b.event_generate('<KeyPress-Return>');b.event_generate('<KeyRelease-Return>');ready(root,h,ed)
    assert ed.selected==GOOD and ed.task_locator.signal and frozen(s)==before

def test_current_G0_GTR_only_task_has_no_false_MT_action(tmp_path):
    src=OUT/'private'/SID;s=review.MobileReviewSession.open(src/'source.alprsession',sidecar_path=src/'review.json')
    before=frozen(s);text,value=pure(s,G,GOOD)
    assert 'GTR' in text and value is None and frozen(s)==before

def test_L02_L07_L19_overlapping_FINAL_buttons_switch_exact_target(tmp_path,monkeypatch):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def overlap(rows,trace,payload):
        for row in rows:
            row.update(plate_left=150,plate_top=140,plate_right=300,plate_bottom=210,mt_confidence=.9)
            row['mt_pose_corners_norm']=rows[0]['mt_pose_corners_norm']
    s=fixture_archive(tmp_path/'data',n=2,mz=False,mutate=overlap);s.set_review_mode('assisted');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        settled(root,h,ed);before=frozen(s)
        for oid,index in (('GT:a1',2),('GT:a0',1)):
            assert nav.select_layer(ed,'POSE',oid);ready(root,h,ed)
            assert ed.task_locator.signal is None
            line=ed.contexts.fields['image'];assert line.button.cget('text')==f'Pokaż MT{index}' and f'MT {index}' in line.cget('text')
            line.button.invoke();ready(root,h,ed)
            assert ed.task_locator.signal['target']['attempt_id']==oid[3:] and ed.selected==oid
            assert len(ed.c.find_withtag('pending_mt_label'))==1
            assert all(oid in ed.c.gettags(i) for i in ed.c.find_withtag('pending_mt_contour'))
        assert frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass

def test_L11_verify_frame_failure_cannot_choose_neighbor(gui,monkeypatch):
    from auto_annotation_tool.ranking import mobile_canvas_resolution as resolution
    root,h,ed,s=gui;before=frozen(s);selected=ed.selected
    def fail(*a,**k):raise ValueError('Brak udokumentowanego full-frame/ROI')
    monkeypatch.setattr(resolution,'verify_frame',fail)
    assert not ed.task_locator.activate(target(ed)) and ed.selected==selected and frozen(s)==before

def test_L12_zero_BBOX_without_Pose_or_GTR(tmp_path,monkeypatch):
    from test_z4_space_gtr_hud_shift import fixture_archive
    def change(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',
            plate_left=150,plate_top=170,plate_right=300,plate_bottom=170,mt_confidence=.9)
    s=fixture_archive(tmp_path/'data',n=1,mz=False,mutate=change);s.set_review_mode('assisted');monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    try:
        settled(root,h,ed);before=frozen(s)
        assert ed.task_locator.activate(target(ed));ready(root,h,ed)
        assert ed.task_locator.signal['kind']=='BBOX' and ed.task_locator.signal['marker']
        assert not stacks.snapshot(s,'mt-one')['plates'] and frozen(s)==before
    finally:
        try:next(gen)
        except StopIteration:pass

def test_L17_close_cancels_own_callback(gui):
    root,h,ed,s=gui;locate(gui);timer=ed.task_locator.after_id
    ed.dispose();assert ed.task_locator.after_id is None and timer not in ed.c.tk.call('after','info')

def test_L23_save_in_progress_does_not_locate(gui):
    root,h,ed,s=gui;before=frozen(s);h._saving=True
    try:assert not ed.task_locator.activate(target(ed)) and frozen(s)==before
    finally:h._saving=False

def test_L28_inline_link_reuses_spacing_on_narrow_resize(gui):
    root,h,ed,s=gui;before=frozen(s);line=ed.contexts.fields['image']
    for geometry,compact in (('1180x760+10+10',True),('1649x896+10+10',False)):
        h.window.geometry(geometry);root.update();ready(root,h,ed)
        assert line.compact is compact
        b=line.button
        assert b.winfo_rooty()+b.winfo_height()<=line.winfo_rooty()+line.winfo_height()
    assert frozen(s)==before
