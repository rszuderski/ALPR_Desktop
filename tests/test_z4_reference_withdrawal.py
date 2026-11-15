"""Read compatibility for the retired explicit-withdrawal workflow.

The original action/UI tests are superseded by model_reference_cleanup tests.
Legacy evidence remains strict and readable; no withdrawal writer is exposed.
"""
import copy,json
import pytest
from test_z4_model_reference_cleanup import actual,G,SID,P1,P2,state,frozen,cleaned,core,stacks,legacy,review

def test_historical_withdrawal_opens_read_only(actual):
    before=frozen(actual);sc,v=state(actual)
    legacy.validate_audit(sc,session_id=SID)
    assert legacy.withdrawn_ids(sc)=={P2} and P1 in v['active_plate_ids'] and P2 not in v['active_plate_ids']
    assert frozen(actual)==before

def test_no_new_withdrawal_writer_or_confirmation_workflow():
    assert not hasattr(legacy,'withdraw') and not hasattr(legacy,'prepare_withdrawal')
    assert not hasattr(legacy,'confirm_image_review')

def test_original_disposition_is_archived_exactly_after_cleanup(actual):
    old=copy.deepcopy(state(actual)[0][stacks.KEY][legacy.KEY]['records'][P2]);cleaned(actual)
    from auto_annotation_tool.ranking.mobile_reference_lifecycle import archived_reference
    record=archived_reference(state(actual)[1],P2)
    assert record['legacy_disposition']==old
    reopened=review.MobileReviewSession.open(actual.path,sidecar_path=actual.sidecar_path)
    assert P2 not in state(reopened)[1]['plates']

def test_missing_historical_event_is_still_rejected(actual):
    data=json.loads(actual.sidecar_path.read_bytes());scene=data['invocation_annotations'][G][core.KEY]
    scene['events']=[e for e in scene['events'] if e.get('action')!=legacy.ACTION]
    actual.sidecar_path.write_text(json.dumps(data),encoding='utf-8');before=actual.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='historii'):review.MobileReviewSession.open(actual.path,sidecar_path=actual.sidecar_path)
    assert actual.sidecar_path.read_bytes()==before

def test_cleanup_Undo_Redo_preserves_legacy_disposition_and_no_task(actual):
    from auto_annotation_tool.gui.mobile_canvas_review_tasks import for_scene
    cleaned(actual);stacks.restore_history(actual,G)
    assert legacy.withdrawn_ids(state(actual)[0])=={P2} and for_scene(actual,G)['count']==0
    stacks.restore_history(actual,G,redo=True)
    assert P2 not in state(actual)[1]['plates'] and for_scene(actual,G)['count']==0
