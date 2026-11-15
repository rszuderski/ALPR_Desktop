"""U01–U26: operational visibility and presentation, never scientific mutation."""
import copy,hashlib,importlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_z4_frame_stacks import linked,ready
from test_z4_p0_physical_gt import session_fixture
from test_mobile_p2b2_canvas_review import source_session,MANUAL
import test_mobile_p2b2_canvas_gui as native

G='mt-one';P=Path(__file__).resolve().parents[1]/'output/P0_Z4_CLEAN_UX_20261025_001'
@pytest.fixture
def clean():return importlib.import_module('auto_annotation_tool.gui.mobile_canvas_clean_view')

def attach(s,p,mp):
    mp.setattr(native,'source_session',lambda _:s)
    return native.gui.__wrapped__(p,mp)

@pytest.fixture
def night(tmp_path,monkeypatch):
    source=P/'copies/exp-1792702658896-47b4e60b';archive=next(source.glob('*.alprsession'))
    # The audited archive below output/copies is already isolated and read-only;
    # each test writes its own sidecar, without another identical 8 MB archive.
    shutil.copy2(source/'review.json',tmp_path/'review.json')
    s=MobileReviewSession.open(archive,sidecar_path=tmp_path/'review.json')
    gen=attach(s,tmp_path,monkeypatch);root,h,ed,_=next(gen)
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    gid=list(s.mt_invocations)[4];iid=next(i for i in h.subject_tree.get_children('') if gid in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry'] and not h._saving);ready(root,h,ed)
    try:yield root,h,ed,s,gid
    finally:
        try:next(gen)
        except StopIteration:pass

@pytest.fixture
def pair(tmp_path,monkeypatch):
    s=linked(tmp_path/'data',stacks,n=2,suppressed=None)
    gen=attach(s,tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:yield root,h,ed,s,G
    finally:
        try:next(gen)
        except StopIteration:pass

def pick(ui,index=0):
    root,h,ed,s,gid=ui;oid=core.scene(s,gid)['objects'][index]['object_id']
    ed._choose_plate(oid);ready(root,h,ed);return oid

def focus(ui):
    root,h,ed,s,gid=ui;oid=pick(ui);q=core._object(core.scene(s,gid),oid)['quad'];x,y=ed.to_canvas(q[0])
    # These are separate corrections, not a Double-Button gesture (which the
    # established bindings intentionally consume). Use separated event times.
    ed._test_click_time=getattr(ed,'_test_click_time',1000)+1000
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y),time=ed._test_click_time);native.pump(root);ed.overlay()
    assert ed.drag is not None
    return oid,(x,y)

def frozen(s):return (s.sidecar_path.read_bytes(),copy.deepcopy(core.summary(s)))
def unchanged(s,before):assert (s.sidecar_path.read_bytes(),core.summary(s))==before
def drawn(h,tag):return list(h.canvas.find_withtag(tag))

def test_U01_real_night_no_RAW(night,clean):
    root,h,ed,s,gid=night;v=stacks.snapshot(s,gid);raw=next(c for c in v['candidates'] if c['role']=='SUPPRESSED')
    assert raw['object_id'] not in clean.visible_ids(ed)
    assert all(raw['object_id'] not in h.canvas.gettags(i) for i in h.canvas.find_all())
    assert any(c['role']=='FINAL' for c in v['candidates']) and drawn(h,'mt_original')

def test_U02_saved_FP_visible_and_scored(night,clean):
    root,h,ed,s,gid=night;b=frozen(s)
    fp=next(c for c in stacks.snapshot(s,gid)['candidates'] if c['association_status']=='UNASSOCIATED_NO_PLATE')
    assert fp['object_id'] in clean.visible_ids(ed) and core.counts(core.scene(s,gid))['false_positive']==1
    assert len(drawn(h,'fp_diagonal'))==2
    unchanged(s,b)

def test_U03_selected_final_PPM_marks_and_keeps_it(pair,clean):
    root,h,ed,s,gid=pair;oid=pick(pair);before=copy.deepcopy(core.scene(s,gid)['parents'])
    q=core._object(core.scene(s,gid),oid)['quad'];native.click(h,ed,tuple(sum(p[i] for p in q)/4 for i in (0,1)),3);ready(root,h,ed)
    assert oid in clean.visible_ids(ed) and core._object(core.scene(s,gid),oid)['error']
    assert core.scene(s,gid)['parents']==before and len(stacks.snapshot(s,gid)['plates'])==1

def test_U04_show_FP_after_restart_and_reverse(night,clean):
    root,h,ed,s,gid=night;fp=next(c for c in stacks.snapshot(s,gid)['candidates'] if c['association_status']=='UNASSOCIATED_NO_PLATE')
    b=frozen(s);assert fp['object_id'] in clean.visible_ids(ed)
    clean.toggle_fp(ed);ready(root,h,ed);assert fp['object_id'] not in clean.visible_ids(ed);unchanged(s,b)
    clean.toggle_fp(ed);ready(root,h,ed);assert fp['object_id'] in clean.visible_ids(ed);unchanged(s,b)
    ed._choose_plate(fp['object_id']);q=fp['mt_quad'];native.click(h,ed,tuple(sum(p[i] for p in q)/4 for i in (0,1)),3);ready(root,h,ed)
    assert not core._object(core.scene(s,gid),fp['object_id'])['error']
    ed.undo();ready(root,h,ed);assert core._object(core.scene(s,gid),fp['object_id'])['error']
    re=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path);assert core.summary(re)==core.summary(s)

def test_U05_RAW_not_in_standard_stack(night,clean):
    from auto_annotation_tool.gui.mobile_canvas_stacks import members
    root,h,ed,s,gid=night;v=stacks.snapshot(s,gid);b=frozen(s)
    for pid in v['plates']:assert all(c['role']=='FINAL' for c in members(v,pid))
    assert any(r['role']=='SUPPRESSED' for r in stacks.diagnostics(s,[gid])['rows']);unchanged(s,b)

def test_U06_single_final_has_no_fake_counter(night,clean):
    root,h,ed,s,gid=night;oid=pick(night);b=frozen(s)
    assert '1/1' not in clean.bar_caption(ed)
    native.key(h,'z');ready(root,h,ed);assert ed.selected==oid;unchanged(s,b)

def test_U07_two_FINAL_cycle_shared_GT(pair,clean):
    root,h,ed,s,gid=pair;pick(pair);b=frozen(s)
    assert '1/2' in clean.bar_caption(ed)
    native.key(h,'z');ready(root,h,ed);assert ed.selected=='GT:a1' and '2/2' in clean.bar_caption(ed)
    native.key(h,'x');ready(root,h,ed)
    assert ed.selected=='GT:a0';unchanged(s,b)

def test_U08_entry_tab_is_native(pair,clean):
    root,h,ed,s,gid=pair;pick(pair);ed.start_entry('GT:a0');native.pump(root)
    entry=ed.entry;entry.focus_force();native.pump(root);target=entry.tk_focusNext();ed.entry_var.set('WA01234X')
    entry.event_generate('<KeyPress-Tab>');ready(root,h,ed)
    assert ed.selected=='GT:a0' and h.window.focus_get() is target and ed.entry is None
    assert all(p['gt_text']=='WA01234X' for p in stacks.snapshot(s,gid)['plates'].values())
    ed.start_entry('GT:a0');native.pump(root);entry=ed.entry;entry.focus_force();native.pump(root);target=entry.tk_focusPrev()
    entry.event_generate('<KeyPress-Tab>',state=1);ready(root,h,ed)
    assert ed.selected=='GT:a0' and h.window.focus_get() is target and ed.entry is None

def test_U09_unselected_PPM_targets_visible_FINAL(pair,clean):
    root,h,ed,s,gid=pair;before=copy.deepcopy(core.scene(s,gid)['parents']);rev=s.review.review_revision
    native.click(h,ed,(225,175),3);ready(root,h,ed)
    assert ed.selected=='GT:a0' and core._object(core.scene(s,gid),'GT:a0')['error']
    assert s.review.review_revision==rev+1 and core.scene(s,gid)['parents']==before

def test_U10_zoom_labels_never_cover_quads(night,clean):
    root,h,ed,s,gid=night;pick(night);b=frozen(s)
    for z in (1,3,8):
        h._view_zoom=z;h._paint_image();native.pump(root)
        for box in ed.clean_label_boxes:
            assert all(not clean.intersects(box,q) for q in ed.clean_geometry_boxes)
    assert not drawn(h,'overlap_hud');unchanged(s,b)

def test_U11_focus_only_GT_and_four_handles(night,clean):
    root,h,ed,s,gid=night;b=frozen(s);oid,_=focus(night)
    assert ed.gt_focus_id==oid and len(drawn(h,'operator_quad'))==1 and len(drawn(h,'corner_handle'))==4
    assert oid in ed.badges and drawn(h,'gt_badge') and drawn(h,'stack_badge')
    for tag in ('mt_original','overlap_hud','analyzed_roi','model_location_bbox'):assert not drawn(h,tag)
    assert clean.bar_caption(ed).startswith('Korekta GT');ed.escape();unchanged(s,b)

def test_U12_focus_survives_zoom_pan_release(night,clean):
    root,h,ed,s,gid=night;oid,(x,y)=focus(night);b=frozen(s)
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    h._view_zoom*=1.3;h._view_center=(h._view_center[0]+5,h._view_center[1]+5);h._paint_image();native.pump(root)
    assert ed.gt_focus_id==oid and len(drawn(h,'corner_handle'))==4 and not drawn(h,'mt_original');unchanged(s,b)
    center=h._view_center
    h.canvas.event_generate('<ButtonPress-3>',x=10,y=10);h.canvas.event_generate('<B3-Motion>',x=18,y=16);h.canvas.event_generate('<ButtonRelease-3>',x=18,y=16);ready(root,h,ed)
    assert h._view_center!=center and ed.gt_focus_id==oid and not drawn(h,'mt_original');unchanged(s,b)

def test_U13_cancel_live_draft_preserves_saved(night,clean):
    root,h,ed,s,gid=night;oid,(x,y)=focus(night);b=frozen(s)
    h.canvas.event_generate('<B1-Motion>',x=round(x+3),y=round(y+2));native.pump(root)
    ed.escape();ready(root,h,ed)
    assert not ed.gt_focus_id and ed.selected==oid and ed.drag is None;unchanged(s,b)
    # A completed corner correction survives cancelling the NEXT unsaved drag.
    oid,(x,y)=focus(night)
    h.canvas.event_generate('<B1-Motion>',x=round(x+3),y=round(y+2));h.canvas.event_generate('<ButtonRelease-1>',x=round(x+3),y=round(y+2));ready(root,h,ed)
    saved=frozen(s);assert saved!=b
    oid,(x,y)=focus(night);h.canvas.event_generate('<B1-Motion>',x=round(x+2),y=round(y+1));ed.escape();ready(root,h,ed);unchanged(s,saved)
    # D changes selection; a stale focus must not hide the newly drawn GT.
    focus(night);ed.release();ed.draw_toggle();ready(root,h,ed);assert not ed.gt_focus_id and ed.mode=='draw';ed.escape();unchanged(s,saved)

def test_U14_shared_correction_one_write(pair,clean):
    root,h,ed,s,gid=pair;oid,(x,y)=focus(pair);parents=copy.deepcopy(core.scene(s,gid)['parents']);rev=s.review.review_revision
    h.canvas.event_generate('<B1-Motion>',x=round(x+3),y=round(y+2));h.canvas.event_generate('<ButtonRelease-1>',x=round(x+3),y=round(y+2));ready(root,h,ed)
    assert s.review.review_revision==rev+1 and core.scene(s,gid)['parents']==parents
    rows=stacks.diagnostics(s,[gid])['rows'];assert rows[0]['gt_quad']==rows[1]['gt_quad'] and rows[0]['mz_prediction']!=rows[1]['mz_prediction']

def test_U15_views_never_change_saved_review(pair,clean):
    root,h,ed,s,gid=pair;b=frozen(s);pick(pair);clean.toggle_fp(ed);native.key(h,'z');ready(root,h,ed)
    focus(pair);ed.escape();h._view_zoom=1.2;h._paint_image();ed._choose_plate('');ready(root,h,ed);unchanged(s,b)
    entry=h._image_entry;h._image_entry='another/frame.png';ed.overlay()
    assert not drawn(h,'mt_original') and not drawn(h,'operator_quad') and ed.hit(100,100) is None
    h._image_entry=entry;ed.overlay();unchanged(s,b)

def test_U16_no_selection_no_stale_result(pair,clean):
    root,h,ed,s,gid=pair;pick(pair);ed._choose_plate('');ready(root,h,ed)
    assert ed.contexts.data['object']=='Wybierz ramkę na obrazie'

def test_U17_compact_three_contexts_own_MZ(pair,clean):
    root,h,ed,s,gid=pair;pick(pair,1)
    assert set(ed.contexts.fields)=={'object','image','session'}
    assert 'WA11234' in ed.contexts.data['object'] and 'IoU' in ed.contexts.data['object']
    assert len(ed.contexts.data['object'].splitlines())<=5 and not ed.panel.winfo_ismapped()

def test_U18_FP_has_no_metrics(night,clean):
    root,h,ed,s,gid=night;pick(night,2)
    text=ed.contexts.data['object'];assert 'błędne wykrycie' in text
    assert not any(word in text for word in ('IoU','NMCE','CER','GT:'))

def test_U19_missing_pose_NOT_RUN_is_not_zero(tmp_path,monkeypatch,clean):
    from test_z4_akt89lm_resolution import invalid_source
    s=invalid_source(tmp_path);gen=attach(s,tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        ed._choose_plate('GT:a0');ready(root,h,ed)
        text=ed.contexts.data['object'];assert 'nie można zmierzyć' in text.lower() and 'Nie wykonano odczytu' in text
        assert '0,0%' not in text and 'Rozstrzygnij ramki' in ed.contexts.data['image']
        assert all(ed.contexts.counters[k].cget('text')=='—' for k in ('gt','mt','fp','fn'))
    finally:
        try:next(gen)
        except StopIteration:pass

def test_U20_concrete_next_action(pair,clean):
    root,h,ed,s,gid=pair;core.set_text(s,gid,'GT:a0','');ready(root,h,ed)
    assert 'Zweryfikuj tablicę GTR 1' in ed.contexts.data['image']
    pick(pair);native.key(h,'space');ready(root,h,ed)
    assert 'Obraz zweryfikowany' in ed.contexts.fields['image'].cget('text')
    assert 'uzupełnij numer GT' in ed.contexts.fields['image'].cget('text')

def test_U21_space_decides_selected_frame_and_derives_whole_image(pair,clean):
    root,h,ed,s,gid=pair;core.set_scene_review_state(s,gid,checked=False);ready(root,h,ed)
    pick(pair)
    native.key(h,'space');ready(root,h,ed);assert core.scene(s,gid)['checked']
    assert ed.approve.winfo_ismapped() and ed.approve.cget('state')=='disabled'
    native.key(h,'space');ready(root,h,ed);assert not core.scene(s,gid)['checked']

def test_U22_exclusion_requires_reason(pair,clean):
    root,h,ed,s,gid=pair;b=frozen(s);clean.show_exclusion(ed);native.pump(root);unchanged(s,b)
    ed.clean_reason.set('');assert not clean.save_exclusion(ed);unchanged(s,b)
    ed.clean_reason.set('Test: klatka zasłonięta');assert clean.save_exclusion(ed);ready(root,h,ed)
    assert core.scene(s,gid)['exclusion_reason']=='Test: klatka zasłonięta' and core.scene(s,gid)['checked']

def test_U23_no_global_duplicate_progress(pair,clean):
    root,h,ed,s,gid=pair;assert not h.progress_label.winfo_ismapped()
    text=' '.join(ed.contexts.data.values())
    assert not any(t in text for t in ('surowych','NMS','RAW','Braki dowodów','zakres roboczy'))
    report=h.report
    h.browser._comparison_verification={(str(report.report_id),report.source_archive_sha256.lower()):SimpleNamespace(state='STALE')}
    ready(root,h,ed);assert 'eksport nieaktualny' in ed.contexts.data['session'].lower()

def test_U24_native_sizes_all_contexts_visible(pair,clean):
    from z4_sidebar_assertions import assert_sidebar_reaches
    root,h,ed,s,gid=pair;pick(pair)
    for size in ('1649x896','1180x760'):
        h.window.geometry(size+'+20+20');ready(root,h,ed)
        assert_sidebar_reaches(root,h,list(ed.contexts.fields.values()))
        assert h.canvas.winfo_width()>=380

def test_U25_UNKNOWN_is_local_not_hidden_as_RAW(night,clean,monkeypatch):
    root,h,ed,s,gid=night;v=copy.deepcopy(stacks.snapshot(s,gid));v['candidates'][0]['role']='UNKNOWN'
    import auto_annotation_tool.gui.mobile_canvas_stacks as ui
    monkeypatch.setattr(ui,'view',lambda _:v);pick(night)
    assert 'Nie można ustalić pochodzenia' in ed.contexts.data['object']
    assert v['candidates'][0]['object_id'] in clean.visible_ids(ed)

def test_U26_all_F4_records_unchanged():
    rows=json.loads((P/'baseline_f4.json').read_text(encoding='utf-8'))
    for r in rows:
        s=MobileReviewSession.open(r['path'],sidecar_path=r['sidecar']);p=Path(r['sidecar']);b=p.read_bytes() if p.exists() else None
        assert s.review.review_revision==r['revision'] and core.summary(s)==r['summary']
        assert (p.read_bytes() if p.exists() else None)==b
    for name,sha in json.loads((P/'baseline_hashes.json').read_text()).items():
        p=Path(name);assert (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)==sha
