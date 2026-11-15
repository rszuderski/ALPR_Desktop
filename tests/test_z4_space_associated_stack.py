"""Space positively confirms a physical stack in one durable action."""
import copy
import pytest
from test_z4_gtr_reapproval import verified,edit,pid,MANUAL,G,OID
from test_z4_mt_geometry_layers import source_session
from test_z4_space_gtr_hud_shift import fixture_archive
from test_z4_p0_physical_gt import session_fixture
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial

def scene(s):return core.scene(s,G)
def frozen(s):return s.sidecar_path.read_bytes(),copy.deepcopy(s.review.to_dict())

def test_first_positive_stack_accepts_source_geometry_without_fabricating_text(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);parents=copy.deepcopy(scene(s)['parents']);revision=s.review.review_revision
    assert partial.confirm_stack(s,G,oid=OID)
    p=stacks.snapshot(s,G)['plates'][pid(s)]
    assert scene(s)['checked'] and not partial.progress(scene(s))['pending_gtr_ids']
    assert p['geometry_state']=='operator_confirmed_unchanged' and p['gt_quad']==parents[0]['quad']
    assert p['gt_text']=='' and not p['text_confirmed'] and scene(s)['parents']==parents
    assert s.review.review_revision==revision+1 and scene(s)['events'][-1]['action']=='operator_confirm_associated_frames'

def test_pending_geometry_and_unresolved_presence_are_approved_together_and_repeat_is_noop(tmp_path):
    s=verified(tmp_path);q=edit(s);partial.toggle_plate(s,G,OID);revision=s.review.review_revision;p=pid(s)
    assert not scene(s)['checked'] and partial.progress(scene(s))['object_decisions'][OID]==partial.UNRESOLVED
    assert partial.confirm_stack(s,G,plate_id=p)
    assert scene(s)['checked'] and partial.progress(scene(s))['object_decisions'][OID]==partial.PLATE
    assert not partial.progress(scene(s))['pending_gtr_ids'] and stacks.snapshot(s,G)['plates'][p]['gt_quad']==q
    assert s.review.review_revision==revision+1
    before=frozen(s);assert not partial.confirm_stack(s,G,plate_id=p) and frozen(s)==before

def test_shared_GT_confirms_only_explicitly_linked_FINALs(tmp_path):
    s=source_session(tmp_path,n=3,mz=False);partial.toggle_plate(s,G,OID);stacks.associate(s,G,'GT:a1',OID)
    partial.toggle_plate(s,G,OID);partial.toggle_plate(s,G,'GT:a1');edit(s);before=copy.deepcopy(stacks.snapshot(s,G)['links'])
    partial.confirm_stack(s,G,plate_id=pid(s));progress=partial.progress(scene(s))
    assert progress['object_decisions']=={OID:partial.PLATE,'GT:a1':partial.PLATE,'GT:a2':partial.UNRESOLVED}
    assert not progress['pending_gtr_ids'] and not scene(s)['checked'] and stacks.snapshot(s,G)['links']==before

def test_other_pending_GT_is_not_approved(tmp_path):
    s=verified(tmp_path,n=2);edit(s);edit(s,'GT:a1');partial.confirm_stack(s,G,plate_id=pid(s))
    assert partial.progress(scene(s))['pending_gtr_ids']==[pid(s,'GT:a1')] and not scene(s)['checked']

def test_manual_GT_approval_does_not_associate_overlapping_model(tmp_path):
    s=source_session(tmp_path,n=1,mz=False);quad=[[190,155],[260,155],[260,195],[190,195]]
    oid=core.add_plate(s,G,quad,partial_review=True);manual=pid(s,oid)
    partial.confirm_stack(s,G,plate_id=manual)
    assert not partial.progress(scene(s))['pending_gtr_ids'] and not scene(s)['checked']
    assert partial.progress(scene(s))['object_decisions'][OID]==partial.UNRESOLVED and stacks.snapshot(s,G)['links'][OID]['plate_id'] is None
    before=frozen(s)
    with pytest.raises(ValueError):partial.confirm_stack(s,G,oid=OID)
    assert frozen(s)==before

def test_manual_stack_is_confirmed_without_changing_existing_text_or_source(tmp_path):
    s=verified(tmp_path);oid=core.add_plate(s,G,MANUAL,partial_review=True);stacks.edit_gt(s,G,oid,text='KEEPGTZ')
    parents=copy.deepcopy(scene(s)['parents']);p=pid(s,oid);partial.confirm_stack(s,G,plate_id=p)
    assert scene(s)['checked'] and stacks.snapshot(s,G)['plates'][p]['gt_text']=='KEEPGTZ' and scene(s)['parents']==parents

def test_BBOX_without_Pose_uses_identity_only_then_accepts_drawn_GT(tmp_path):
    def invalid(rows,trace,payload):
        rows[0].update(mt_status='DETECTION_INVALID_QUAD',mt_pose_status='INVALID_QUAD',mt_pose_corners_norm='',plate_left=150,plate_top=140,plate_right=300,plate_bottom=210)
    s=fixture_archive(tmp_path,n=1,mz=False,mutate=invalid);s.set_review_mode('assisted');parents=copy.deepcopy(scene(s)['parents'])
    partial.confirm_stack(s,G,oid=OID);p=pid(s)
    assert scene(s)['checked'] and stacks.snapshot(s,G)['plates'][p]['gt_quad'] is None and scene(s)['parents']==parents
    stacks.complete_bbox_reference(s,G,OID,[[150,140],[300,140],[300,210],[150,210]],expected_plate_id=p)
    assert not scene(s)['checked'];partial.confirm_stack(s,G,oid=OID)
    assert scene(s)['checked'] and not partial.progress(scene(s))['pending_gtr_ids'] and scene(s)['parents']==parents

def test_confirming_GT_does_not_undo_unlinked_BAD_but_explicit_source_acceptance_can(tmp_path):
    s=source_session(tmp_path,n=2,mz=False);partial.toggle_plate(s,G,OID);stacks.associate(s,G,'GT:a1',OID)
    stacks.toggle_unassociated(s,G,'GT:a1',partial_review=True);bad=copy.deepcopy(stacks.snapshot(s,G)['links']['GT:a1'])
    partial.confirm_stack(s,G,plate_id=pid(s))
    assert partial.progress(scene(s))['object_decisions']['GT:a1']==partial.NOT_PLATE and stacks.snapshot(s,G)['links']['GT:a1']==bad
    partial.confirm_stack(s,G,oid='GT:a1')
    assert partial.progress(scene(s))['object_decisions']['GT:a1']==partial.PLATE and pid(s,'GT:a1')==pid(s)

def test_one_undo_redo_restores_both_geometry_approval_and_presence(tmp_path):
    s=verified(tmp_path);edit(s);partial.toggle_plate(s,G,OID);before=copy.deepcopy(scene(s)[partial.KEY])
    partial.confirm_stack(s,G,plate_id=pid(s));assert scene(s)['checked']
    stacks.restore_history(s,G);assert scene(s)[partial.KEY]==before and not scene(s)['checked']
    stacks.restore_history(s,G,redo=True);assert scene(s)['checked'] and not partial.progress(scene(s))['pending_gtr_ids']

def test_RAW_unknown_ID_and_deleted_GT_cannot_be_accepted(tmp_path):
    s=session_fixture(tmp_path,n=3,suppressed=2);before=frozen(s)
    for args in ({'oid':'GT:a2'},{'oid':'GT:missing'},{'plate_id':'PGT:missing'}):
        with pytest.raises(ValueError):partial.confirm_stack(s,G,**args)
        assert frozen(s)==before
