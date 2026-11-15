"""Z4MOBREV013: old MT positive decision never starts editing on reload.

Verify guided answer semantics, legitimate skip after partial clicks, and
human-facing wording while leaving CER/NMCE/IoU computation untouched.
"""
from __future__ import annotations
import inspect
from auto_annotation_tool.gui import z4_mobile_sample_review as ui
from test_z4mobrev012_guided_form import view


def test_restored_yes_does_not_enter_corner_editor():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._sync_pose_gt_controls()
    assert not w.pose_edit_var.get()
    assert w.pose_gt_answer.get()==''
    assert not w.pose_edit_button.shown
    assert 'Wybierz odpowiedź poniżej' in w.pose_gt_hint.text


def test_load_and_presence_handlers_do_not_start_corners_implicitly():
    assert 'self._maybe_begin_guided_pose_gt()' not in inspect.getsource(ui.MobileSampleReviewWindow._load_image)
    assert 'self._maybe_begin_guided_pose_gt()' not in inspect.getsource(ui.MobileSampleReviewWindow.save_detection_choice)
    assert 'self._maybe_begin_guided_pose_gt()' in inspect.getsource(ui.MobileSampleReviewWindow._resume_pose_after_answer)


def test_only_explicit_second_answer_activates_drawing():
    w=view()
    w.detection_choice.set('yes')
    w.save_detection_choice()
    assert w.canvas.cursor=='hand2'
    assert not w.pose_edit_var.get()
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    assert w.pose_edit_var.get()
    assert w.canvas.cursor=='crosshair'
    assert len(w.session.calls)==1


def test_unmarkable_choice_after_two_clicked_corners_discards_only_draft():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._pose_gt_points=[(1.,1.), (4.,1.)]
    w.pose_edit_var.set(True)
    w.pose_gt_answer.set('unavailable')
    w._choose_pose_gt_answer()
    assert w._pose_gt_points==[]
    assert not w.pose_edit_var.get()
    decision=w.session.review.attempt_annotations['a1']
    assert decision['pose_gt_unavailable_reason']=='corners_not_distinguishable'
    assert decision['is_plate'] is True
    assert 'pose_gt_quad_mt_input_px' not in decision


def test_cancel_without_marking_preserves_model_preview():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w.pose_gt_answer.set('mark')
    w._choose_pose_gt_answer()
    assert w.pose_edit_var.get()
    w._toggle_pose_gt_edit()
    assert not w.pose_edit_var.get()
    w._sync_pose_gt_controls()
    assert 'Wybierz odpowiedź poniżej' in w.pose_gt_hint.text
    assert w.pose_gt_answer.get()==''


def test_operator_copy_describes_purpose_not_metric_acronyms():
    src=inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert 'sprawdzić wskazanie modelu' in src
    assert 'Czy możesz wskazać cztery rogi tablicy?' in src
    assert '03  ·  ODCZYT NUMERU' in src
    assert 'NMCE' not in src
    assert 'Polygon IoU' not in src
    assert 'CER' not in src


def test_human_readable_geometric_scores_use_same_calculation():
    src=inspect.getsource(ui.MobileSampleReviewWindow._sync_pose_gt_controls)
    assert 'compare_mt_pose_to_gt' in src
    assert "measured['nmce_pct']" in src
    assert "measured['polygon_iou_pct']" in src
    assert 'Błąd położenia rogów' in src and 'Zgodność obrysu' in src


def test_no_geometry_assessment_stays_separate_from_mz_review():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w.pose_gt_answer.set('unavailable')
    w._choose_pose_gt_answer()
    assert w.session.review.attempt_annotations['a1']['is_plate'] is True
    assert w.session.review.sample_annotations['c1']=={}
    assert w.session.review.subjects['subject']=={'evaluable':False,'ground_truth':''}
