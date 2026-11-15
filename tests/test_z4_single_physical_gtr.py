"""One physical plate, one verification unit; explicit legacy reconciliation."""
from pathlib import Path
import copy,shutil
import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_z4_mt_geometry_layers import source_session
from test_z4_gtr_reapproval import verified,MANUAL

OUT=Path(__file__).resolve().parents[1]/'output/Z4_SINGLE_PHYSICAL_GTR_20261030_001'
F='exp-1792699303016-c8b05efb';G=F+'-a00000010'

def real(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True);p=tmp_path/'review.json'
    shutil.copyfile(OUT/'private'/(F+'.review.json'),p)
    return MobileReviewSession.open(OUT/'private/day.alprsession',sidecar_path=p)

def pair(s):return tuple(stacks.snapshot(s,G)['plates'])
def merge(s):
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import merge_manual_duplicate
    canonical,duplicate=pair(s)
    return merge_manual_duplicate(s,G,duplicate,canonical,expected_revision=s.review.review_revision)

def test_one_stack_is_one_decision_even_with_pending_MT_and_GTR(tmp_path):
    s=verified(tmp_path);partial.toggle_stack(s,'mt-one',oid='GT:a0');v=partial.progress(core.scene(s,'mt-one'))
    assert len(v['pending_ids'])==len(v['pending_gtr_ids'])==1
    assert v['decisions']==1

def test_shared_FINALs_have_one_unit_but_two_physical_plates_have_two(tmp_path):
    s=source_session(tmp_path,n=3,mz=False)
    partial.confirm_stack(s,'mt-one',oid='GT:a0');stacks.associate(s,'mt-one','GT:a1','GT:a0')
    partial.confirm_stack(s,'mt-one',oid='GT:a2');partial.toggle_stack(s,'mt-one',oid='GT:a0')
    assert partial.progress(core.scene(s,'mt-one'))['decisions']==1
    partial.toggle_stack(s,'mt-one',oid='GT:a2');assert partial.progress(core.scene(s,'mt-one'))['decisions']==2

def test_real_image10_merge_retains_identity_text_geometry_and_auditable_duplicate(tmp_path):
    s=real(tmp_path);a,b=pair(s);old=copy.deepcopy(stacks.snapshot(s,G));review=copy.deepcopy(s.review.to_dict());rev=s.review.review_revision
    merge(s);state=stacks.snapshot(s,G);scene=core.scene(s,G)
    assert list(state['plates'])==[a] and state['plates'][a]==old['plates'][a]
    assert state['retired_references'][b]['plate']==old['plates'][b]
    assert state['retired_references'][b]['merged_into']==a
    assert scene['parents']==core.scene(real(tmp_path/'original'),G)['parents']
    assert s.review.review_revision==rev+1 and partial.progress(scene)['decisions']==1
    assert partial.progress(scene)['object_decisions']==partial.progress(core.scene(real(tmp_path/'other'),G))['object_decisions']
    assert {k:v for k,v in s.review.invocation_annotations.items() if k!=G}=={k:v for k,v in review['invocation_annotations'].items() if k!=G}
    partial.toggle_stack(s,G,plate_id=a);assert partial.progress(core.scene(s,G))['complete']
    partial.toggle_stack(s,G,plate_id=a);assert partial.progress(core.scene(s,G))['decisions']==1 and len(stacks.snapshot(s,G)['plates'])==1

def test_merge_undo_redo_and_reopen_preserve_both_full_historical_references(tmp_path):
    s=real(tmp_path);before=copy.deepcopy(stacks.snapshot(s,G)['plates']);merge(s);rev=s.review.review_revision;frozen=s.sidecar_path.read_bytes()
    s=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert s.sidecar_path.read_bytes()==frozen and s.review.review_revision==rev
    assert stacks.restore_history(s,G) and stacks.snapshot(s,G)['plates']==before
    assert stacks.restore_history(s,G,redo=True) and len(stacks.snapshot(s,G)['plates'])==1
    assert len(core.scene(s,G)[stacks.HISTORY]['entries'])>=11

def test_read_and_suggestions_never_merge_or_write(tmp_path):
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import suggestions
    s=real(tmp_path);before=s.sidecar_path.read_bytes();a,b=pair(s)
    proposals=suggestions(core.scene(s,G),stacks.snapshot(s,G))
    assert [(x['canonical_id'],x['duplicate_id']) for x in proposals]==[(a,b)]
    assert s.sidecar_path.read_bytes()==before and len(stacks.snapshot(s,G)['plates'])==2

def test_true_second_plate_is_not_suggested_or_merged_and_keeps_its_number(tmp_path):
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import suggestions,merge_manual_duplicate
    from auto_annotation_tool.ranking.mobile_canvas_operator_view import reference_numbers
    s=real(tmp_path);a,b=pair(s);oid=core.add_plate(s,G,[[100,150],[220,150],[220,190],[100,190]],partial_review=True)
    state=stacks.snapshot(s,G);p=next(p for p,v in state['plates'].items() if v['origin_object_id']==oid)
    assert reference_numbers(state)[p]==3
    assert all(x['duplicate_id']!=p for x in suggestions(core.scene(s,G),state))
    before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):merge_manual_duplicate(s,G,p,a,expected_revision=s.review.review_revision)
    assert s.sidecar_path.read_bytes()==before
    merge_manual_duplicate(s,G,b,a,expected_revision=s.review.review_revision)
    state=stacks.snapshot(s,G);assert len(state['plates'])==2 and reference_numbers(state)[p]==3
    assert partial.progress(core.scene(s,G))['decisions']==2

def test_stale_revision_and_conflicting_GT_text_refuse_without_write(tmp_path):
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import merge_manual_duplicate
    s=real(tmp_path);a,b=pair(s);before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):merge_manual_duplicate(s,G,b,a,expected_revision=s.review.review_revision-1)
    assert s.sidecar_path.read_bytes()==before
    oid=stacks.snapshot(s,G)['plates'][b]['origin_object_id'];stacks.edit_gt(s,G,oid,text='OTHER123');before=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError):merge_manual_duplicate(s,G,b,a,expected_revision=s.review.review_revision)
    assert s.sidecar_path.read_bytes()==before


def test_confirmed_canonical_reference_does_not_need_reverification_after_merge(tmp_path):
    s=real(tmp_path);a,b=pair(s);partial.confirm_stack(s,G,plate_id=a);partial.confirm_stack(s,G,plate_id=b)
    before=copy.deepcopy(stacks.snapshot(s,G)['plates'][a]);assert partial.progress(core.scene(s,G))['complete']
    merge(s)
    assert partial.progress(core.scene(s,G))['complete'] and stacks.snapshot(s,G)['plates'][a]==before


def test_explicit_separate_identity_is_not_overruled_by_overlap(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import keep_separate
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import suggestions
    s=real(tmp_path);a,b=pair(s);state=stacks.snapshot(s,G)
    keep_separate(s,G,state['plates'][b]['origin_object_id'],state['plates'][a]['origin_object_id'])
    before=s.sidecar_path.read_bytes();assert suggestions(core.scene(s,G),stacks.snapshot(s,G))==[]
    assert s.sidecar_path.read_bytes()==before


def test_real_two_plate_photo_keeps_two_independent_GT_and_verification_units(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    from test_z4_mt_detection_evaluation import with_source_confidence
    from auto_annotation_tool.ranking.mobile_gtr_reconciliation import suggestions
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    photo=OUT.parent/'p0_obraz05_20261025'
    raw=session_fixture(tmp_path/'source',n=2,suppressed=None,real=photo)
    s=with_source_confidence(raw,tmp_path)
    for oid in ('GT:a0','GT:a1'):partial.confirm_stack(s,'mt-one',oid=oid)
    before=copy.deepcopy(stacks.snapshot(s,'mt-one')['plates']);ids=list(before)
    assert len(ids)==2 and evaluate_unit(s,'mt-one')['tp']==2
    assert suggestions(core.scene(s,'mt-one'),stacks.snapshot(s,'mt-one'))==[]
    for n,pid in enumerate(ids,1):
        partial.toggle_stack(s,'mt-one',plate_id=pid)
        assert partial.progress(core.scene(s,'mt-one'))['decisions']==n
    partial.toggle_stack(s,'mt-one',plate_id=ids[0])
    assert partial.progress(core.scene(s,'mt-one'))['decisions']==1
    assert stacks.snapshot(s,'mt-one')['plates']==before


@pytest.fixture
def gui(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
    s=real(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s)
    cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    iid=next(i for i in h.subject_tree.get_children('') if G in subject_groups(s,h._subject_ids[i]))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==G and h._image_entry==core.scene(s,G)['image_entry'] and not h._saving);ready(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass


@pytest.mark.parametrize('kind',['BBOX','POSE','GTR'])
def test_native_button_merges_once_then_Space_toggles_one_plate(gui,kind):
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean
    root,h,ed,s=gui;a,b=pair(s);before=copy.deepcopy(stacks.snapshot(s,G)['plates'][a]);ed.escape();ready(root,h,ed)
    assert len(ed.gtr_reconciliation_buttons)==1 and ed.gtr_reconciliation_bar.winfo_manager()
    ed.gtr_reconciliation_buttons[0].invoke();ready(root,h,ed)
    assert not ed.gtr_reconciliation_bar.winfo_manager() and len(clean.records(ed))==1
    assert clean.context(ed,{})['image_counts']==dict(gt='1',mt='1',rejected='0',unlinked='0')
    assert partial.progress(core.scene(s,G))['decisions']==1
    nav.select_layer(ed,kind);ready(root,h,ed)
    for expected in (True,False,True):
        native.key(h,'space');ready(root,h,ed)
        assert partial.progress(core.scene(s,G))['complete'] is expected
        assert partial.progress(core.scene(s,G))['decisions']==int(not expected)
        assert len(stacks.snapshot(s,G)['plates'])==1 and stacks.snapshot(s,G)['plates'][a]==before
        assert ed.checked_var.get() is expected
