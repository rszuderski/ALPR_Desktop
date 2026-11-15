"""Geometry changes invalidate verification until that physical GT is approved."""
import copy
import pytest
from test_z4_mt_geometry_layers import source_session
from test_z4_space_gtr_hud_shift import fixture_archive
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_mt_geometry_layers import create_reference
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
G='mt-one';OID='GT:a0';MANUAL=[[650,350],[800,350],[800,420],[650,420]]
def scene(s):return core.scene(s,G)
def pending(s):return partial.progress(scene(s))['pending_gtr_ids']
def pid(s,oid=OID):return stacks.snapshot(s,G)['links'][oid]['plate_id']
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())
def verified(path,n=1):
    s=source_session(path,n=n,mz=False)
    for i in range(n):partial.toggle_plate(s,G,f'GT:a{i}')
    assert scene(s)['checked'] and not pending(s);return s
def edit(s,oid=OID,delta=3):
    q=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid(s,oid)]['gt_quad']);q[0][0]+=delta
    stacks.edit_gt(s,G,oid,quad=q);return q

@pytest.mark.parametrize('api',['stack','core'])
def test_edit_requires_explicit_reapproval_of_same_GT_without_losing_presence(tmp_path,api):
    s=verified(tmp_path);before=stacks.snapshot(s,G);parents=copy.deepcopy(scene(s)['parents']);q=copy.deepcopy(before['plates'][pid(s)]['gt_quad']);q[0][0]+=3
    if api=='stack':stacks.edit_gt(s,G,OID,quad=q)
    else:core.correct_plate(s,G,OID,q)
    assert not scene(s)['checked'] and pending(s)==[pid(s)]
    assert partial.progress(scene(s))['object_decisions'][OID]==partial.PLATE
    assert stacks.snapshot(s,G)['links']==before['links'] and scene(s)['parents']==parents
    assert build_inventory(s)['inspected']==0
    assert partial.confirm_gtr(s,G,pid(s))
    assert scene(s)['checked'] and not pending(s) and build_inventory(s)['inspected']==1
    assert stacks.snapshot(s,G)['plates'][pid(s)]['gt_quad']==q

def test_new_manual_GT_is_pending_until_confirmed(tmp_path):
    s=verified(tmp_path);oid=core.add_plate(s,G,MANUAL,partial_review=True);manual=pid(s,oid)
    assert pending(s)==[manual] and not scene(s)['checked']
    assert partial.progress(scene(s))['manual_added']==1
    assert partial.confirm_gtr(s,G,manual) and scene(s)['checked']

def test_many_changed_GTs_require_each_approval(tmp_path):
    s=verified(tmp_path,n=2);edit(s);edit(s,'GT:a1');a,b=pid(s),pid(s,'GT:a1')
    assert set(pending(s))=={a,b} and partial.progress(scene(s))['decisions']==2
    partial.confirm_gtr(s,G,a);assert pending(s)==[b] and not scene(s)['checked']
    partial.confirm_gtr(s,G,b);assert not pending(s) and scene(s)['checked']

def test_GT_text_and_source_presence_cannot_clear_pending_geometry(tmp_path):
    s=verified(tmp_path);edit(s);p=pid(s)
    stacks.edit_gt(s,G,OID,text='KEEPGTZ');assert pending(s)==[p] and not scene(s)['checked']
    partial.toggle_plate(s,G,OID);partial.toggle_plate(s,G,OID)
    assert pending(s)==[p] and not scene(s)['checked']
    partial.confirm_gtr(s,G,p);assert scene(s)['checked']
    assert stacks.snapshot(s,G)['plates'][p]['gt_text']=='KEEPGTZ'

def test_read_reopen_and_unrelated_actions_never_reapprove(tmp_path):
    s=verified(tmp_path);edit(s);p=pid(s);before=frozen(s)
    for _ in range(3):partial.progress(scene(s));core.summary(s);stacks.diagnostics(s);build_inventory(s)
    assert frozen(s)==before
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert pending(reopened)==[p] and not scene(reopened)['checked']

def test_another_edit_after_approval_invalidates_again_but_noop_does_not(tmp_path):
    s=verified(tmp_path);q=copy.deepcopy(stacks.snapshot(s,G)['plates'][pid(s)]['gt_quad'])
    stacks.edit_gt(s,G,OID,quad=q);assert scene(s)['checked'] and not pending(s)
    edit(s);partial.confirm_gtr(s,G,pid(s));assert scene(s)['checked']
    edit(s);assert not scene(s)['checked'] and pending(s)==[pid(s)]

def test_image_checkbox_and_complete_review_cannot_bypass_GT_approval(tmp_path):
    s=verified(tmp_path);edit(s);before=frozen(s)
    for action in (lambda:core.check_scene(s,G),lambda:core.set_scene_review_state(s,G,checked=True)):
        with pytest.raises(ValueError,match='GTR'):action()
        assert frozen(s)==before

def test_initial_explicit_Space_still_confirms_unchanged_source(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);partial.toggle_plate(s,G,OID)
    assert scene(s)['checked'] and not pending(s)

def test_drawn_source_reference_needs_confirmation(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);q=copy.deepcopy(scene(s)['parents'][0]['quad'])
    create_reference(s,G,OID,q)
    assert pending(s)==[pid(s)] and not scene(s)['checked']
    partial.confirm_gtr(s,G,pid(s));assert scene(s)['checked']

def test_missing_pose_completion_keeps_GTZ_and_requires_approval(tmp_path):
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted')
    partial.toggle_plate(s,G,OID);stacks.edit_gt(s,G,OID,text='SAVEDGTZ');p=pid(s)
    assert scene(s)['checked']
    stacks.complete_bbox_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]],expected_plate_id=p)
    assert pending(s)==[p] and not scene(s)['checked']
    partial.confirm_gtr(s,G,p);assert scene(s)['checked'] and stacks.snapshot(s,G)['plates'][p]['gt_text']=='SAVEDGTZ'
    assert scene(s)['parents'][0]['quad'] is None

def test_undo_redo_restores_exact_verification_state(tmp_path):
    s=verified(tmp_path);edit(s);assert pending(s)
    assert stacks.restore_history(s,G);assert scene(s)['checked'] and not pending(s)
    assert stacks.restore_history(s,G,redo=True);assert not scene(s)['checked'] and pending(s)
    partial.confirm_gtr(s,G,pid(s));assert scene(s)['checked']
    stacks.restore_history(s,G);assert not scene(s)['checked'] and pending(s)
    stacks.restore_history(s,G,redo=True);assert scene(s)['checked'] and not pending(s)

def test_removing_pending_manual_reference_restores_prior_approved_state(tmp_path):
    s=verified(tmp_path);oid=core.add_plate(s,G,MANUAL,partial_review=True);assert pending(s)
    core.remove_operator_geometry(s,G,oid);assert scene(s)['checked'] and not pending(s)

def test_legacy_checked_review_keeps_other_decisions_when_geometry_changes(tmp_path):
    s=verified(tmp_path,n=2);raw=copy.deepcopy(s.review);raw.invocation_annotations[G][core.KEY].pop(partial.KEY);s._save_change(raw)
    before=copy.deepcopy(scene(s)['parents']);edit(s)
    assert pending(s)==[pid(s)] and not scene(s)['checked']
    assert all(d==partial.PLATE for d in partial.progress(scene(s))['object_decisions'].values())
    partial.confirm_gtr(s,G,pid(s));assert scene(s)['checked'] and scene(s)['parents']==before

def test_invalid_or_already_approved_GT_has_no_side_effect(tmp_path):
    s=verified(tmp_path);before=frozen(s)
    assert not partial.confirm_gtr(s,G,pid(s))
    with pytest.raises(ValueError):partial.confirm_gtr(s,G,'PGT:missing')
    assert frozen(s)==before
