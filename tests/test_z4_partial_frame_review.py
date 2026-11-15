"""Partial MT presence decisions are durable, reversible and independent of OCR GT."""
import copy

import pytest
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory, review_progress
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from test_mobile_p2b2_canvas_review import source_session, MANUAL
from test_z4_p0_physical_gt import session_fixture
from test_z4_akt89lm_resolution import invalid_source

G = 'mt-one'


def state(session):
    return partial.progress(core.scene(session, G))


def frozen(session):
    return session.sidecar_path.read_bytes(), copy.deepcopy(session.review.to_dict())


def test_reading_never_confirms_or_migrates_unopened_predictions(tmp_path):
    s = source_session(tmp_path)
    before = frozen(s)
    for _ in range(3):
        assert state(s)['decisions'] == 2
        assert not state(s)['complete']
        assert build_inventory(s)['inspected'] == 0
        assert partial.KEY not in core.scene(s, G)
    assert frozen(s) == before


def test_two_individual_decisions_verify_image_but_do_not_invent_ocr_gt(tmp_path):
    s = source_session(tmp_path)
    parents = copy.deepcopy(core.scene(s, G)['parents'])
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['decisions'] == 1 and not core.scene(s, G)['checked']
    partial.toggle_plate(s, G, 'GT:a1')
    scene = core.scene(s, G)
    assert state(s)['complete'] and scene['checked']
    assert state(s)['correct'] == 2 and core.counts(scene)['expected'] == 2
    assert scene['parents'] == parents
    assert all(not p['text_confirmed'] and p['gt_text'] == '' for p in stacks.snapshot(s, G)['plates'].values())
    assert core.summary(s)['evaluated_reads'] == core.summary(s)['geometry_n'] == 0
    assert core.group_issues(s, G)  # GT text remains a separate export requirement.
    inv = build_inventory(s)
    assert inv['inspected'] == 1 and inv['remaining'] == inv['decisions'] == 0
    assert inv['groups'][G]['status'] == 'VERIFIED'


def test_second_press_is_unresolved_without_losing_text_geometry_or_source(tmp_path):
    s = source_session(tmp_path, n=1)
    partial.toggle_plate(s, G, 'GT:a0')
    core.set_text(s, G, 'GT:a0', 'ABC123')
    quad = copy.deepcopy(core.scene(s, G)['objects'][0]['quad'])
    quad[0][0] += 4
    core.correct_plate(s, G, 'GT:a0', quad)
    partial.confirm_gtr(s, G, stacks.snapshot(s, G)['links']['GT:a0']['plate_id'])
    accepted = copy.deepcopy(stacks.snapshot(s, G)['plates'])
    parents = copy.deepcopy(core.scene(s, G)['parents'])
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['decisions'] == 1 and not core.scene(s, G)['checked']
    assert core.counts(core.scene(s, G))['expected'] == 0
    assert stacks.snapshot(s, G)['plates'] == accepted
    assert core.scene(s, G)['parents'] == parents
    assert core._object(core.scene(s, G), 'GT:a0')['gt_text'] == 'ABC123'
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['complete'] and core.counts(core.scene(s, G))['expected'] == 1
    assert core._object(core.scene(s, G), 'GT:a0')['quad'] == quad


def test_all_false_positive_decisions_finish_image_and_count_no_phantom_gt(tmp_path):
    s = source_session(tmp_path)
    parents = copy.deepcopy(core.scene(s, G)['parents'])
    for oid in ('GT:a0', 'GT:a1'):
        stacks.toggle_unassociated(s, G, oid, partial_review=True)
    assert state(s)['complete'] and core.scene(s, G)['checked']
    assert core.counts(core.scene(s, G))['expected'] == 0
    assert core.counts(core.scene(s, G))['false_positive'] == 2
    assert core.scene(s, G)['roi_presence'] == 'NO_PLATES'
    result = core.summary(s)
    assert result['false_positive_mt'] == 2 and result['negative_frames_with_fp'] == 1
    assert result['true_negative_frames'] == result['missed_plates'] == 0
    assert core.scene(s, G)['parents'] == parents


def test_false_positive_can_withdraw_previously_accepted_presence_without_a_false_fn(tmp_path):
    s = source_session(tmp_path, n=1)
    partial.toggle_plate(s, G, 'GT:a0')
    stored = copy.deepcopy(stacks.snapshot(s, G)['plates'])
    stacks.toggle_unassociated(s, G, 'GT:a0', partial_review=True)
    assert state(s)['complete'] and state(s)['plate_count'] == 0
    assert core.summary(s)['missed_plates'] == 0 and core.summary(s)['false_positive_mt'] == 1
    assert stacks.snapshot(s, G)['plates'] == stored
    stacks.toggle_unassociated(s, G, 'GT:a0', partial_review=True)
    assert state(s)['complete'] and state(s)['correct'] == 1
    assert core.counts(core.scene(s, G))['expected'] == 1


def test_manual_plate_requires_approval_when_added_to_verified_image(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['complete']
    oid = core.add_plate(s, G, MANUAL, partial_review=True)
    assert not state(s)['complete'] and state(s)['manual_added'] == 1
    assert not core.scene(s, G)['checked']
    partial.confirm_gtr(s, G, stacks.snapshot(s, G)['links'][oid]['plate_id'])
    assert state(s)['complete']
    assert core.scene(s, G)['checked']
    assert core.counts(core.scene(s, G))['expected'] == 2
    assert core.summary(s)['missed_plates'] == 1
    assert core._object(core.scene(s, G), oid)['parent_id'] is None
    core.remove_operator_geometry(s, G, oid)
    assert state(s)['complete'] and state(s)['manual_added'] == 0
    assert core.counts(core.scene(s, G))['expected'] == 1


def test_empty_image_requires_explicit_decision_and_manual_addition_recomputes_it(tmp_path):
    s = source_session(tmp_path, n=0, mz=False)
    assert not state(s)['complete'] and state(s)['empty_eligible']
    assert core.summary(s).get('true_negative_frames', 0) == 0
    partial.confirm_empty(s, G)
    assert state(s)['complete'] and core.summary(s)['true_negative_frames'] == 1
    oid = core.add_plate(s, G, MANUAL, partial_review=True)
    assert not state(s)['complete'] and state(s)['manual_added'] == 1
    partial.confirm_gtr(s, G, stacks.snapshot(s, G)['links'][oid]['plate_id'])
    assert state(s)['complete']
    assert not state(s)['empty_confirmed'] and core.summary(s)['true_negative_frames'] == 0
    core.remove_operator_geometry(s, G, oid)
    assert state(s)['empty_eligible'] and not state(s)['complete']
    assert not core.scene(s, G)['checked']


def test_empty_confirmation_cannot_resolve_nonempty_or_unresolved_frames(tmp_path):
    s = source_session(tmp_path, n=1)
    before = frozen(s)
    with pytest.raises(ValueError):
        partial.confirm_empty(s, G)
    assert frozen(s) == before


def test_restart_undo_redo_restore_each_partial_state_and_preserve_source(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    archive = s.path.read_bytes()
    partial.toggle_plate(s, G, 'GT:a0')
    partial.toggle_plate(s, G, 'GT:a0')
    reopened = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert state(reopened)['decisions'] == 1
    assert stacks.restore_history(reopened, G)
    assert state(reopened)['complete'] and core.scene(reopened, G)['checked']
    assert stacks.restore_history(reopened, G, redo=True)
    assert not state(reopened)['complete'] and not core.scene(reopened, G)['checked']
    assert reopened.review.review_status == 'IN_PROGRESS'
    assert s.path.read_bytes() == archive


@pytest.mark.parametrize('field,value', [('source_binding', 'other-source'), ('schema', 'other'),
                                        ('decisions', {'GT:missing': partial.PLATE})])
def test_tampered_decision_binding_fails_closed_without_writes(tmp_path, field, value):
    s = source_session(tmp_path, n=1, mz=False)
    partial.toggle_plate(s, G, 'GT:a0')
    scene = copy.deepcopy(core.scene(s, G))
    scene[partial.KEY][field] = value
    before = frozen(s)
    with pytest.raises(ValueError):
        partial.validate(s, G, scene)
    assert frozen(s) == before


def test_proven_suppressed_raw_never_becomes_an_extra_decision(tmp_path):
    s = session_fixture(tmp_path, n=3, suppressed=2)
    assert state(s)['total'] == state(s)['decisions'] == 2
    before = frozen(s)
    with pytest.raises(ValueError):
        partial.toggle_plate(s, G, 'GT:a2')
    assert frozen(s) == before
    partial.toggle_plate(s, G, 'GT:a0')
    stacks.toggle_unassociated(s, G, 'GT:a1', partial_review=True)
    assert state(s)['complete'] and state(s)['total'] == 2
    assert core.scene(s, G)['parents'] == core._legacy_scene(s, G)['parents']


def test_invalid_pose_identity_is_an_explicit_decision_without_fabricated_pose_metrics(tmp_path):
    s = invalid_source(tmp_path)
    parent = copy.deepcopy(core.scene(s, G)['parents'])
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['complete']
    assert not core.scene(s, G)['objects'][0]['quad']
    assert core.scene(s, G)['parents'] == parent
    assert core.summary(s)['geometry_n'] == 0
    assert core.group_issues(s, G)  # A missing GT outline still blocks geometry/export.


def test_typing_gt_before_space_is_not_implicit_presence_acceptance(tmp_path):
    s = source_session(tmp_path, n=1)
    core.set_text(s, G, 'GT:a0', 'ABC123')
    assert state(s)['decisions'] == 1 and not state(s)['complete']
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['complete'] and core._object(core.scene(s, G), 'GT:a0')['gt_text'] == 'ABC123'


def test_whole_image_flag_cannot_bypass_an_unresolved_object(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    partial.toggle_plate(s, G, 'GT:a0')
    partial.toggle_plate(s, G, 'GT:a0')
    core.set_scene_review_state(s, G, checked=True)
    assert not state(s)['complete'] and not core.scene(s, G)['checked']
    assert core.summary(s)['checked_scenes'] == 0


def test_shared_physical_plate_keeps_positive_candidate_when_primary_is_withdrawn(tmp_path):
    s = source_session(tmp_path, n=2, mz=False)
    core.set_text(s, G, 'GT:a0', 'ABC123')
    stacks.associate(s, G, 'GT:a1', 'GT:a0')
    core.check_scene(s, G)
    parents = copy.deepcopy(core.scene(s, G)['parents'])
    partial.toggle_plate(s, G, 'GT:a0')
    assert state(s)['decisions'] == 1
    assert core.counts(core.scene(s, G))['expected'] == 1
    assert core.counts(core.scene(s, G))['false_positive'] == 0
    stacks.toggle_unassociated(s, G, 'GT:a0', partial_review=True)
    assert state(s)['complete'] and state(s)['correct'] == 1
    assert core.counts(core.scene(s, G))['false_positive'] == 1
    assert core.counts(core.scene(s, G))['expected'] == 1
    assert core.scene(s, G)['parents'] == parents


def test_operator_added_counter_is_provenance_not_the_fn_count(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    manual = core.add_plate(s, G, MANUAL, partial_review=True)
    stacks.associate(s, G, 'GT:a0', manual)
    assert not state(s)['complete']
    partial.confirm_gtr(s, G, stacks.snapshot(s, G)['links'][manual]['plate_id'])
    assert state(s)['complete'] and state(s)['manual_added'] == 1
    assert core.counts(core.scene(s, G))['missed'] == 0
    assert core.counts(core.scene(s, G))['expected'] == 1


def test_positive_decision_cannot_point_to_a_false_positive_link(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    stacks.toggle_unassociated(s, G, 'GT:a0', partial_review=True)
    scene = copy.deepcopy(core.scene(s, G))
    scene[partial.KEY]['decisions']['GT:a0'] = partial.PLATE
    before = frozen(s)
    with pytest.raises(ValueError):
        partial.validate(s, G, scene)
    assert frozen(s) == before
