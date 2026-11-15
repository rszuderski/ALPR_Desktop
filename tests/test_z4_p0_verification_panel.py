"""P0: source location is not Pose, and inspection is not a saved decision."""
import copy
import json

import pytest

from test_z4_akt89lm_resolution import invalid_source
from test_mobile_p2b2_canvas_review import MANUAL, source_session, text_all
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_resolution import (
    link_manual_to_mt, model_location, mark_identity_unresolved,
)
from auto_annotation_tool.ranking.mobile_canvas_presentation import (
    build_inventory, execution_state, operator_context, review_progress,
)

GID = 'mt-one'


@pytest.mark.parametrize('damage', ['missing_bbox','nan','outside_roi','wrong_space','wrong_size','wrong_frame',
                                   'missing_trace','duplicate_trace','wrong_epoch'])
def test_unproven_location_cannot_link_or_correct_into_a_true_positive(tmp_path, damage, monkeypatch):
    s = invalid_source(tmp_path)
    manual = core.add_plate(s,GID,MANUAL)
    row = s.attempts['a0']
    if damage == 'missing_bbox':row.pop('plate_left')
    elif damage == 'nan':row['plate_top'] = 'nan'
    elif damage == 'outside_roi':row['plate_left'] = 0
    elif damage == 'wrong_space':row['mt_pose_coordinate_space'] = 'input_pixels'
    elif damage == 'wrong_size':row['mt_pose_source_width_px'] = 512
    elif damage == 'wrong_frame':row['source_frame_entry'] = 'samples/frames/wrong.jpg'
    else:
        from auto_annotation_tool.ranking import mobile_mt_postdedup as dedup
        trace = dict(visual_epoch=2 if damage == 'wrong_epoch' else 1,camera_transform_generation=1,
                     counters=dict(plate_detections_after_dedup=1))
        matches = [] if damage == 'missing_trace' else [trace,trace] if damage == 'duplicate_trace' else [trace]
        monkeypatch.setattr(dedup,'_trace_index',lambda *args:{(17,100000,1):matches})
    before = copy.deepcopy(s.review.to_dict())
    for action in (lambda:link_manual_to_mt(s,GID,manual,'GT:a0'),
                   lambda:core.correct_plate(s,GID,'GT:a0',MANUAL)):
        with pytest.raises(ValueError):action()
        assert s.review.to_dict() == before


@pytest.mark.parametrize('missing_evidence', [False,True])
def test_bbox_preview_and_explicit_link_never_create_pose_metrics(tmp_path, missing_evidence):
    s = invalid_source(tmp_path,missing_evidence=missing_evidence)
    manual = core.add_plate(s,GID,MANUAL)
    core.set_scene_review_state(s,GID,checked=True)
    before = copy.deepcopy(s.review.to_dict());scene = core.scene(s,GID)
    context = operator_context(s,GID,scene,'',execution_state(s,GID))
    assert context['focus_id'] == manual  # the existing visible reference, no selection write
    assert context['location']['kind'] == 'BBOX'
    assert context['location']['bbox'] == [150,140,300,210]
    assert 'quad' not in context['location'] and 'Narysuj' not in context['question']
    assert 'Odczytu numeru nie wykonano' in context['numbers']
    assert s.review.to_dict() == before
    progress = build_inventory(s)
    assert (progress['inspected'],progress['assessed'],progress['decisions'],progress['evidence_gaps']) == (1,0,1,1)
    link_manual_to_mt(s,GID,manual,'GT:a0')
    after = core.scene(s,GID);summary = core.summary(s)
    assert after['parents'] == scene['parents'] and after['parents'][0]['quad'] is None
    assert summary['linked_plates'] == 1 and summary['geometry_n'] == summary['evaluated_reads'] == 0
    assert summary['rows'][0]['pose_status'] == ('EVIDENCE_UNAVAILABLE' if missing_evidence else 'INVALID')
    assert summary['rows'][0]['polygon_iou_pct'] is summary['rows'][0]['nmce_pct'] is summary['rows'][0]['cer'] is None
    progress = build_inventory(s)
    assert (progress['assessed'],progress['decisions'],progress['evidence_gaps']) == (1,0,1)
    assert core.scene(load(s.path,tmp_path),GID) == after


def test_cannot_establish_is_a_persistent_unknown_not_fp_fn_or_exclusion(tmp_path):
    s = invalid_source(tmp_path)
    manual = core.add_plate(s,GID,MANUAL)
    core.set_scene_review_state(s,GID,checked=True)
    before = core.scene(s,GID)
    mark_identity_unresolved(s,GID,'GT:a0')
    after = core.scene(s,GID)
    assert after['parents'] == before['parents'] and len(after['objects']) == 2
    assert not after['exclusion_reason'] and not any(o['error'] for o in after['objects'])
    assert after['objects'][0]['identity_unresolved']['decision'] == 'CANNOT_ESTABLISH'
    assert review_progress(after)['decisions'] == 0 and review_progress(after)['unresolved'] == 1
    summary = core.summary(s)
    assert summary['linked_plates'] == summary['missed_plates'] == summary['false_positive_mt'] == 0
    assert summary['geometry_n'] == 0 and core.group_issues(s,GID)
    assert all(r['outcome'] == 'UNRESOLVED_MT_IDENTITY' for r in summary['rows'])
    assert summary['operator_progress']['decisions'] == 0
    assert core.scene(load(s.path,tmp_path),GID) == after
    exports = core.export(s,tmp_path/'export')
    assert json.loads(exports[0].read_text(encoding='utf-8'))['rows'][0]['identity_unresolved']
    assert 'Decyzje do podjęcia: 0' in exports[2].read_text(encoding='utf-8')
    # The explicit uncertainty can later be resolved with actual source evidence.
    link_manual_to_mt(s,GID,manual,'GT:a0')
    assert 'identity_unresolved' not in core.scene(s,GID)['objects'][0]


def test_multiple_targets_require_specific_identity_without_reassigning_another_plate(tmp_path):
    s = source_session(tmp_path,n=2);text_all(s)
    manual = core.add_plate(s,GID,MANUAL)
    before = core.scene(s,GID)
    context = operator_context(s,GID,before,manual,execution_state(s,GID))
    assert len(context['pairs']) == 2 and context['target'] is None
    explicit = operator_context(s,GID,before,manual,execution_state(s,GID),target_id='GT:a1')
    assert explicit['location']['attempt_id'] == 'a1'
    link_manual_to_mt(s,GID,manual,'GT:a1')
    after = core.scene(s,GID)
    assert after['objects'][0] == before['objects'][0] and len(after['objects']) == 2


from test_z4_akt89lm_gui import invalid_gui, ready
from test_mobile_p2b2_canvas_gui import pump, gui


def test_native_preview_uncertainty_undo_redo_and_focus_do_not_fake_decisions(invalid_gui):
    root,h,ed,s = invalid_gui
    manual = core.add_plate(s,GID,MANUAL)
    core.set_scene_review_state(s,GID,checked=True);ready(root,h,ed)
    ed._choose_plate('');pump(root)
    before = copy.deepcopy(s.review.to_dict())
    assert not ed.decision_panel.same.winfo_ismapped()
    assert ed.assessment.model['focus_id'] == ''
    assert 'Wybierz ramkę' in ed.contexts.data['object']
    assert 'Narysuj' not in ed.decision_panel.detail.cget('text')
    assert not ed.assessment.technical.opened
    ed.show_model_location('GT:a0');pump(root)
    assert h.canvas.find_withtag('model_location_bbox')
    assert not h.canvas.find_withtag('mt_original')  # rectangle is not a Pose polygon
    ed.sync();ed.overlay();pump(root)
    assert s.review.to_dict() == before and ed.selected == ''
    from auto_annotation_tool.gui.mobile_canvas_stacks import context_menu
    ed._choose_plate('GT:a0');menu=context_menu(ed,show=False)
    index=next(i for i in range(menu.index('end')+1) if menu.entrycget(i,'label')=='Nie można ustalić tożsamości wskazania')
    menu.invoke(index);menu.destroy();ready(root,h,ed)
    assert s.review.review_revision == before['review_revision'] + 1
    saved = core.scene(s,GID)
    assert 'Nie można ustalić' in ed.status.cget('text')
    ed.undo();ready(root,h,ed)
    assert core.scene(s,GID)['objects'] == before['invocation_annotations'][GID][core.KEY]['objects']
    ed.undo(redo=True);ready(root,h,ed)
    assert core.scene(s,GID)['objects'] == saved['objects']
    assert core.scene(load(s.path,s.sidecar_path.parent),GID)['objects'] == saved['objects']


def test_native_missing_location_shows_reason_and_only_allows_unknown(invalid_gui):
    root,h,ed,s = invalid_gui
    core.add_plate(s,GID,MANUAL)
    s.attempts['a0'].pop('plate_left')
    ready(root,h,ed)
    assert str(ed.decision_panel.same.cget('state')) == 'disabled'
    assert not ed.decision_panel.choose_mt.winfo_ismapped()
    assert str(ed.decision_panel.unknown.cget('state')) == 'normal'
    assert 'Nie mamy wiarygodnego położenia' in ed.decision_panel.detail.cget('text')
