"""T29–T43: Canvas establishes GT; inspected is never a count decision."""
import copy,json,hashlib,shutil
from pathlib import Path
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_presentation as presentation
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_z4_frame_stacks import stack_gui,ready,stack_key,linked
from test_z4_p0_physical_gt import session_fixture
from test_mobile_p2b2_canvas_review import source_session,MANUAL
import test_mobile_p2b2_canvas_gui as native

G='mt-one'
OUT=Path(__file__).resolve().parents[1]/'output/P0_GT_CANVAS_ONLY_20261025_001'

@pytest.fixture(name='stacks')
def stack_module():return stacks

def attach(s,tmp_path,monkeypatch):
    monkeypatch.setattr(native,'source_session',lambda p:s)
    return native.gui.__wrapped__(tmp_path,monkeypatch)

@pytest.fixture
def night19(tmp_path,monkeypatch):
    source=OUT/'copies/exp-1792702658896-47b4e60b'
    archive=next(source.glob('*.alprsession'));shutil.copy2(archive,tmp_path/archive.name);shutil.copy2(source/'review.json',tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/archive.name,sidecar_path=tmp_path/'review.json')
    gen=attach(s,tmp_path,monkeypatch);root,h,ed,s=next(gen)
    gid=list(s.mt_invocations)[18];scene=core.scene(s,gid)
    iid=next(i for i in h.subject_tree.get_children('') if gid in presentation.subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==scene['image_entry']);ready(root,h,ed)
    try:yield root,h,ed,s,gid
    finally:
        try:next(gen)
        except StopIteration:pass

def no_count_widget(ed):
    def texts(w):
        try:yield str(w.cget('text'))
        except Exception:pass
        for child in w.winfo_children():yield from texts(child)
    assert not any('Potwierdzam odrębne' in t or 'Potwierdzam jedną tablicę' in t for t in texts(ed.h.window))
    assert not ed.decision_panel.winfo_ismapped()  # no competing sidebar GT decisions

def test_T29_night19_unresolved_is_not_two_gt(night19,monkeypatch):
    root,h,ed,s,gid=night19;before=s.sidecar_path.read_bytes()
    assert len(core.scene(s,gid)['parents'])==2
    no_count_widget(ed)
    assert not stacks.snapshot(s,gid)['plates']
    assert 'Nieustalone' in ed.contexts.data['image'] and 'Rozstrzygnij wskazanie' in ed.contexts.data['image']
    assert s.sidecar_path.read_bytes()==before
    # The existing rare identity actions are contextual to Canvas, not forms
    # in the sidebar. Opening the native menu itself is read-only.
    oid=core.scene(s,gid)['objects'][0]['object_id'];ed._choose_plate(oid);ready(root,h,ed)
    q=core.scene(s,gid)['objects'][0]['quad'];x,y=ed.to_canvas(q[0])
    # Windows native popup posting enters an OS menu loop. The real popup has
    # a separate native receipt; this repeatable regression constructs the
    # actual Tk Menu through the real Ctrl+PPM binding, then invokes its command.
    from auto_annotation_tool.gui import mobile_canvas_stacks as ui
    original=ui.context_menu
    def build(editor,event):
        editor._canvas_decision_menu=original(editor,event,show=False)
        return editor._canvas_decision_menu
    monkeypatch.setattr(ui,'context_menu',build)
    h.canvas.event_generate('<ButtonPress-3>',x=round(x),y=round(y),state=4);ready(root,h,ed)
    menu=ed._canvas_decision_menu
    assert menu.entrycget(0,'label')=='Ustanów GT z tego obrysu MT'
    assert s.sidecar_path.read_bytes()==before
    menu.invoke(0);menu.destroy();ready(root,h,ed)
    assert len(stacks.snapshot(s,gid)['plates'])==1 and not core.scene(s,gid)['checked']

def test_T30_night19_ppm_resolves_only_selected_final(night19):
    root,h,ed,s,gid=night19;scene=core.scene(s,gid);oid=scene['objects'][1]['object_id']
    before=copy.deepcopy(scene['parents']);ed._choose_plate(oid);ready(root,h,ed)
    q=scene['objects'][1]['quad'];point=tuple(sum(p[i] for p in q)/4 for i in (0,1))
    native.click(h,ed,point,button=3);ready(root,h,ed);v=stacks.snapshot(s,gid)
    assert sum(c['association_status']=='UNRESOLVED' for c in v['candidates'])==1
    assert sum(c['association_status']=='UNASSOCIATED_NO_PLATE' for c in v['candidates'])==1
    assert not v['plates'] and core.scene(s,gid)['parents']==before
    assert core.group_issues(s,gid);no_count_widget(ed)

def test_T31_four_canvas_points_create_one_stable_gt(night19):
    root,h,ed,s,gid=night19;scene=core.scene(s,gid);l,t,r,b=scene['roi'];w=r-l;hh=b-t
    points=[(l+w*.1,t+hh*.1),(l+w*.25,t+hh*.1),(l+w*.25,t+hh*.2),(l+w*.1,t+hh*.2)]
    revision=s.review.review_revision;native.key(h,'d')
    for p in points:native.click(h,ed,p)
    ready(root,h,ed);v=stacks.snapshot(s,gid)
    assert len(v['plates'])==core.counts(core.scene(s,gid))['expected']==1
    assert s.review.review_revision==revision+1 and not core.scene(s,gid)['checked']
    assert '1' in ed.contexts.data['image'] and no_count_widget(ed) is None
    re=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert stacks.snapshot(re,gid)['plates']==v['plates']

def test_T32_partial_D_escape_has_no_revision(night19):
    root,h,ed,s,gid=night19;before=s.sidecar_path.read_bytes();scene=core.scene(s,gid)
    l,t,r,b=scene['roi'];points=[(l+40,t+40),(l+120,t+40),(l+120,t+100)]
    native.key(h,'d')
    for p in points:native.click(h,ed,p)
    native.key(h,'Escape');ready(root,h,ed)
    assert not ed.points and s.sidecar_path.read_bytes()==before and not stacks.snapshot(s,gid)['plates']

def test_T33_manual_correction_is_not_another_gt(tmp_path):
    s=session_fixture(tmp_path,n=1,suppressed=None);oid=core.add_plate(s,G,MANUAL)
    ids=set(stacks.snapshot(s,G)['plates']);q=[(x+1,y+1) for x,y in MANUAL]
    core.correct_plate(s,G,oid,q)
    assert set(stacks.snapshot(s,G)['plates'])==ids and core.counts(core.scene(s,G))['expected']==1
    assert core._object(core.scene(s,G),oid)['quad']==[list(p) for p in q]
    from auto_annotation_tool.ranking.mobile_canvas_resolution import keep_separate
    keep_separate(s,G,oid,'GT:a0')
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert core._object(core.scene(reopened,G),oid)['separate_from_mt']==['GT:a0']

def test_T34_one_gt_three_candidates_view_only(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed);frozen=s.sidecar_path.read_bytes()
    stack_key(h,'Tab');ready(root,h,ed);assert ed.selected=='GT:a1'
    stack_key(h,'ISO_Left_Tab');ready(root,h,ed);assert ed.selected=='GT:a0'
    assert core.counts(core.scene(s,G))['expected']==1 and s.sidecar_path.read_bytes()==frozen
    no_count_widget(ed)

def test_T35_manual_FN_without_MZ_or_count_confirmation(tmp_path):
    s=source_session(tmp_path,n=0);oid=core.add_plate(s,G,MANUAL)
    assert not core.scene(s,G)['checked'];core.check_scene(s,G);d=core.summary(s)
    assert d['expected_plates']==d['missed_plates']==1 and d['evaluated_reads']==0 and d['cer'] is None
    assert d['checked_scenes']==1 and not core.group_issues(s,G)
    assert next(r for r in d['rows'] if r['object_id']==oid)['mz_status']=='NOT_RUN'

def test_T36_link_manual_keeps_plate_id(tmp_path):
    s=session_fixture(tmp_path,n=1,suppressed=None);oid=core.add_plate(s,G,MANUAL)
    pid=next(iter(stacks.snapshot(s,G)['plates']));stacks.associate(s,G,'GT:a0',oid)
    core.set_text(s,G,'GT:a0','WA01234');core.check_scene(s,G)
    assert set(stacks.snapshot(s,G)['plates'])=={pid}
    assert (core.summary(s)['expected_plates'],core.summary(s)['linked_plates'],core.summary(s)['missed_plates'])==(1,1,0)

def test_T37_two_real_plates_no_implicit_merge(tmp_path):
    real=Path(__file__).resolve().parents[1]/'output/p0_obraz05_20261025'
    s=session_fixture(tmp_path,n=2,suppressed=None,real=real);parents=copy.deepcopy(core.scene(s,G)['parents'])
    for p in parents:core.add_plate(s,G,p['quad'])
    v=stacks.snapshot(s,G);assert len(v['plates'])==2 and core.counts(core.scene(s,G))['expected']==2
    assert all(c['plate_id'] is None for c in v['candidates'])  # overlap never implies a link
    assert core.scene(s,G)['parents']==parents and not core.scene(s,G)['checked']

def test_T38_explicit_complete_empty_image_is_TN(tmp_path):
    s=source_session(tmp_path,n=0);before=copy.deepcopy(s.review.to_dict())
    assert not core.scene(s,G)['checked'] and not core.summary(s).get('true_negative_frames')
    assert s.review.to_dict()==before
    core.check_scene(s,G);d=core.summary(s)
    assert d['expected_plates']==0 and d['true_negative_frames']==1 and d['checked_scenes']==1

def test_T39_all_final_FP_then_image_completion_is_zero_GT(tmp_path):
    s=session_fixture(tmp_path,n=1,suppressed=None);core.toggle_error(s,G,'GT:a0')
    assert not stacks.snapshot(s,G)['plates'] and not core.scene(s,G)['checked']
    core.check_scene(s,G);d=core.summary(s)
    assert (d['expected_plates'],d['false_positive_mt'],d['negative_frames_with_fp'])==(0,1,1)
    assert not core.group_issues(s,G)

def test_T40_unresolved_RAW_UNKNOWN_never_imply_GT(tmp_path,monkeypatch):
    s=session_fixture(tmp_path,n=3,suppressed=1)
    core.set_scene_review_state(s,G,checked=True)
    assert not stacks.snapshot(s,G)['plates'] and core.group_issues(s,G)
    assert core.summary(s)['linked_plates']==core.summary(s)['false_positive_mt']==0
    assert stacks.snapshot(s,G)['candidates'][1]['role']=='SUPPRESSED'
    from auto_annotation_tool.ranking import mobile_mt_postdedup as nms
    other=session_fixture(tmp_path/'unknown',n=2,suppressed=None)
    monkeypatch.setattr(nms,'canvas_post_nms_proof',lambda *a:None);monkeypatch.setattr(nms,'_trace_index',lambda *a:{})
    before=other.sidecar_path.read_bytes()
    assert all(c['role']=='UNKNOWN' for c in stacks.snapshot(other,G)['candidates'])
    with pytest.raises(ValueError):stacks.accept_reference(other,G,'GT:a0')
    assert other.sidecar_path.read_bytes()==before

def test_T41_legacy_READY_read_only():
    rows=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'));ready_rows=[r for r in rows if r['gate']['state']=='READY']
    assert len(ready_rows)==5
    for r in ready_rows:
        s=MobileReviewSession.open(r['path'],sidecar_path=r['sidecar']);before=Path(r['sidecar']).read_bytes()
        assert core.summary(s)==r['summary'] and s.review.review_revision==r['revision']
        for gid in s.mt_invocations:
            if core.has_scene(s,gid):core.scene(s,gid)
        assert Path(r['sidecar']).read_bytes()==before

def test_T42_persistent_history_D_link_PPM_restart(tmp_path):
    s=session_fixture(tmp_path,n=1,suppressed=None);oid=core.add_plate(s,G,MANUAL)
    p=copy.deepcopy(stacks.snapshot(s,G)['plates']);assert len(p)==1
    stacks.associate(s,G,'GT:a0',oid);stacks.toggle_unassociated(s,G,'GT:a0')
    after=copy.deepcopy(stacks.snapshot(s,G)['links'])
    for _ in range(3):assert stacks.restore_history(s,G)
    assert not stacks.snapshot(s,G)['plates'] and core.counts(core.scene(s,G))['expected']==0
    assert core.summary(s)['scenes'][0]['counts']['expected']==0
    s=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    for _ in range(3):assert stacks.restore_history(s,G,redo=True)
    assert stacks.snapshot(s,G)['links']==after and set(stacks.snapshot(s,G)['plates'])==set(p)

def test_T43_F4_hashes_and_new_export_without_distinct_gate(tmp_path):
    hashes=json.loads((OUT/'baseline_hashes.json').read_text())
    for name,digest in hashes.items():
        p=Path(name);assert (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)==digest
    s=source_session(tmp_path,n=0);core.add_plate(s,G,MANUAL);core.check_scene(s,G)
    from auto_annotation_tool.gui.mobile_compare_review_gate import inspect_canvas
    d=core.summary(s);assert inspect_canvas(d,'COMPLETED')[0]
    files=core.export(s,tmp_path/'export');assert json.loads(files[0].read_text(encoding='utf-8'))==d
    assert d['expected_plates']==1 and d['missed_plates']==1
