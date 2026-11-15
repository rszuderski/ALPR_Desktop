"""B01-B16: BAD belongs to a proven source FINAL, never an operator GTR."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,shutil
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session,MANUAL
from test_z4_frame_stacks import ready
from test_z4_p0_physical_gt import session_fixture
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean

OUT=Path(__file__).resolve().parents[1]/'output/Z4_BAD_ONLY_MT_FINAL_20261027_001'
G='mt-one';REF='exp-1792699303016-c8b05efb';G8=REF+'-a00000008'
LINKED=[[155,145],[295,145],[295,205],[155,205]]

def frozen(s,ed=None):
    value=dict(file=s.sidecar_path.read_bytes() if s.sidecar_path.exists() else None,review=copy.deepcopy(s.review.to_dict()),
        scene=copy.deepcopy(core.scene(s,G)),detection=evaluate_unit(s,G,cache=False))
    if ed:value['ui']=copy.deepcopy(dict(selected=ed.selected,mode=ed.mode,points=ed.points,drag=ed.drag,
        undo=ed.undo_stack,redo=ed.redo_stack,source_selection=ed._bad_mt_selection))
    return value

def legacy(s):
    change=copy.deepcopy(s.review);r=change.invocation_annotations[G][core.KEY]
    for k in (stacks.KEY,stacks.HISTORY,partial.KEY,'physical_gt_v1'):r.pop(k,None)
    s._save_change(change)

def case(path,linked=False):
    s=source_session(path,n=1);oid=core.add_plate(s,G,LINKED if linked else MANUAL,partial_review=True)
    stacks.edit_gt(s,G,oid,text='REALGT')
    if linked:stacks.associate(s,G,'GT:a0',oid)
    return s,oid

@pytest.fixture
def gui(tmp_path,monkeypatch,request):
    s,oid=case(tmp_path/'data',linked=bool(getattr(request,'param',False)))
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    ready(root,h,ed)
    yield root,h,ed,s,oid
    try:next(gen)
    except StopIteration:pass

def ppm(h,x,y):
    h.canvas.event_generate('<ButtonPress-3>',x=round(x),y=round(y));h.canvas.event_generate('<ButtonRelease-3>',x=round(x),y=round(y));h.window.update()

def source_badge(h,ed,oid='GT:a0'):
    # Current four-row badge exposes the explicit source hit area directly.
    if oid in getattr(ed,'mt_source_badge_hits',{}):
        l,t,r,b=ed.mt_source_badge_hits[oid];return (l+r)/2,(t+b)/2
    rects=[h.canvas.coords(i) for i in h.canvas.find_withtag('mt_relation_badge') if h.canvas.type(i)=='rectangle' and oid in h.canvas.gettags(i)]
    assert len(rects)==1
    l,t,r,b=rects[0];return (l+r)/2,(t+b)/2

@pytest.mark.parametrize('old',[False,True])
def test_B07_core_rejects_manual_without_any_write(tmp_path,old):
    s,oid=case(tmp_path)
    if old:legacy(s)
    before=frozen(s)
    with pytest.raises(ValueError):core.toggle_error(s,G,oid)
    assert frozen(s)==before

@pytest.mark.parametrize('previous',[False,True])
def test_B08_store_cannot_bypass_manual_BAD_validation(tmp_path,previous):
    s,oid=case(tmp_path)
    if previous:
        legacy(s);change=copy.deepcopy(s.review);core._object(change.invocation_annotations[G][core.KEY],oid)['error']=True;s._save_change(change)
    before=frozen(s);bad=core._legacy_scene(s,G);core._object(bad,oid)['error']=not previous
    with pytest.raises(ValueError):core._store(s,G,bad,'internal_BAD_attempt',oid)
    assert frozen(s)==before

def test_B05_RAW_cannot_receive_BAD_through_either_API(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=1)
    if core.scene(s,G).get('_post_nms_pending'):core.accept_post_nms_raw_only(s,G)
    before=frozen(s)
    for action in (core.toggle_error,stacks.toggle_unassociated):
        with pytest.raises(ValueError):action(s,G,'GT:a1')
        assert frozen(s)==before

def test_B05_explicit_negative_never_marks_suppressed_RAW_BAD(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=1);stacks.confirm_negative(s,G)
    raw=core._object(core.scene(s,G),'GT:a1')
    assert raw['post_nms_suppressed'] and not raw['error']
    assert all(core._object(core.scene(s,G),oid)['error'] for oid in ('GT:a0','GT:a2'))

def test_B03_B11_FINAL_round_trip_keeps_physical_GT_and_source(tmp_path):
    s,oid=case(tmp_path,linked=True);before=stacks.snapshot(s,G);parents=copy.deepcopy(core.scene(s,G)['parents'])
    stacks.toggle_unassociated(s,G,'GT:a0',partial_review=True)
    bad=stacks.snapshot(s,G);assert bad['links']['GT:a0']['status']=='UNASSOCIATED_NO_PLATE'
    assert bad['plates']==before['plates'] and bad['links']['GT:a0']['previous_link']==before['links']['GT:a0']
    stacks.toggle_unassociated(s,G,'GT:a0',partial_review=True)
    after=stacks.snapshot(s,G)
    assert after['plates']==before['plates'] and after['links']==before['links'] and core.scene(s,G)['parents']==parents

def test_B04_BBOX_FINAL_BAD_keeps_INVALID_QUAD_and_MZ_NOT_RUN(tmp_path):
    s=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=tmp_path/'bbox.json');s.set_review_mode('assisted')
    before=copy.deepcopy(core.scene(s,G8)['parents']);core.toggle_error(s,G8,'GT:'+G8)
    assert core.scene(s,G8)['parents']==before and before[0]['quad'] is None and before[0]['mz_status']=='NOT_RUN'
    core.toggle_error(s,G8,'GT:'+G8);assert core.scene(s,G8)['parents']==before

@pytest.mark.parametrize('gui',[False,True],indirect=True)
def test_B01_B02_B06_GTR_PPM_is_read_only_even_when_linked(gui):
    root,h,ed,s,oid=gui;row=next(r for r in clean.records(ed) if r['plate'] and r['plate']['gt_text']=='REALGT')
    q=row['gt'];p=tuple(sum(v[i] for v in q)/4 for i in (0,1));before=frozen(s,ed)
    native.click(h,ed,p,button=3);ready(root,h,ed)
    assert frozen(s,ed)==before and not h.canvas.find_withtag('fp_diagonal')
    assert 'GTR' in h.status_var.get()

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_B03_B06_source_badge_explicitly_targets_MT_and_reverses(gui):
    root,h,ed,s,oid=gui;before=stacks.snapshot(s,G);ppm(h,*source_badge(h,ed));ready(root,h,ed)
    assert stacks.snapshot(s,G)['links']['GT:a0']['status']=='UNASSOCIATED_NO_PLATE'
    assert h.canvas.find_withtag('fp_diagonal') and stacks.snapshot(s,G)['plates']==before['plates']
    ppm(h,*source_badge(h,ed));ready(root,h,ed)
    assert stacks.snapshot(s,G)['links']==before['links'] and not h.canvas.find_withtag('fp_diagonal')
    x,y=source_badge(h,ed);h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y));h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    assert ed._bad_mt_selection==(id(s),G,'GT:a0')
    ed.decision_panel.toggle_fp();ready(root,h,ed)
    assert stacks.snapshot(s,G)['links']['GT:a0']['status']=='UNASSOCIATED_NO_PLATE'

def test_B09_refusal_preserves_D_points_selection_and_undo(gui):
    root,h,ed,s,oid=gui;native.key(h,'d');native.click(h,ed,(450,480));ready(root,h,ed)
    assert len(ed.points)==1;before=frozen(s,ed)
    native.click(h,ed,(710,385),button=3);ready(root,h,ed)
    assert frozen(s,ed)==before

@pytest.mark.parametrize('gui',[False,True],indirect=True)
def test_B08_GTR_decision_panel_cannot_redirect_BAD_to_paired_MT(gui):
    root,h,ed,s,oid=gui;row=next(r for r in clean.records(ed) if r['plate'] and r['plate']['gt_text']=='REALGT')
    ed._choose_plate(row['oid']);ready(root,h,ed)
    panel=ed.decision_panel;assert panel.obj and ed._bad_mt_selection is None
    before=frozen(s,ed);panel.toggle_fp();ready(root,h,ed)
    assert frozen(s,ed)==before and not panel.fp.winfo_manager()

def test_B10_manual_reference_is_FN_never_FP(tmp_path):
    s=source_session(tmp_path,n=0,mz=False);oid=core.add_plate(s,G,MANUAL,partial_review=True)
    assert evaluate_unit(s,G,cache=False)['status']=='NOT_EVALUABLE'
    partial.confirm_gtr(s,G,stacks.snapshot(s,G)['links'][oid]['plate_id'])
    core.check_scene(s,G)
    result=evaluate_unit(s,G,cache=False)
    assert result['status']=='MEASURED' and (result['tp'],result['fp'],result['fn'])==(0,0,1)
    assert not core._object(core.scene(s,G),oid)['error']

def test_B12_legacy_manual_BAD_requires_explicit_correction_without_migration(tmp_path):
    s,oid=case(tmp_path);legacy(s);change=copy.deepcopy(s.review);r=change.invocation_annotations[G][core.KEY]
    core._object(r,oid)['error']=True;r['checked']=True;s._save_change(change)
    before=frozen(s);issues=core.group_issues(s,G)
    assert any('GTR' in i and 'BAD' in i for i in issues)
    assert evaluate_unit(s,G,cache=False)['status']!='MEASURED'
    with pytest.raises(ValueError):core.check_scene(s,G)
    assert frozen(s)==before

def test_B13_reopen_and_export_preserve_GT_text_links_and_BAD_scope(tmp_path):
    s,oid=case(tmp_path/'data',linked=True);stacks.toggle_unassociated(s,G,'GT:a0',partial_review=True)
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    for field in ('invocation_annotations','attempt_annotations','review_revision','review_status','review_mode'):
        assert getattr(reopened.review,field)==getattr(s.review,field)
    assert core.scene(reopened,G)==core.scene(s,G)
    before=frozen(s);target=tmp_path/'export';core.export(s,target)
    assert frozen(s)==before and any(target.iterdir())
    assert stacks.snapshot(reopened,G)['plates']==stacks.snapshot(s,G)['plates']
    current_path=tmp_path/'current.review.json';shutil.copyfile(OUT/'private/reference_baseline.review.json',current_path)
    current=MobileReviewSession.open(OUT/'private/reference.alprsession',sidecar_path=current_path)
    raw=current_path.read_bytes();record=copy.deepcopy(current.review.invocation_annotations)
    core.export(current,tmp_path/'current_export')
    reopened=MobileReviewSession.open(current.path,sidecar_path=current_path)
    assert current_path.read_bytes()==raw and reopened.review.invocation_annotations==record

def test_B14_current_protected_F4_and_previous_reports_unchanged():
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    for n,value in b['protected'].items():
        p=Path(n);assert p.exists()==value['exists']
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256']
    for n,h in b['prior_reports'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
