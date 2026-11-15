"""Z4MOBREV014 — clear, neutral, three-stage questionnaire, no data mutations."""
from __future__ import annotations
import inspect
import tkinter as tk
from types import SimpleNamespace

from auto_annotation_tool.gui import z4_mobile_sample_review as ui
from test_z4mobrev012_guided_form import view


def test_operator_skip_is_a_choice_not_an_assertion_of_failure():
    src = inspect.getsource(ui.MobileSampleReviewWindow._sync_pose_gt_controls)
    build = inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert 'Nie udało się wskazać wszystkich rogów' not in src
    assert 'Wybrano pominięcie oceny obrysu' in src
    assert 'Nie mam pewności — pomiń ocenę obrysu' in build
    assert 'Czy możesz wskazać cztery rogi tablicy?' in build


def test_geometry_skip_still_leaves_mt_and_mz_independent():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w.pose_gt_answer.set('unavailable')
    w._choose_pose_gt_answer()
    w._sync_pose_gt_controls()
    assert 'Wybrano pominięcie' in w.pose_gt_hint.text
    assert w.session.review.attempt_annotations['a1']['pose_gt_unavailable_reason'] == 'corners_not_distinguishable'
    assert w.session.review.attempt_annotations['a1']['is_plate'] is True
    assert w.session.review.sample_annotations['c1'] == {}


def test_user_facing_explanations_are_short_and_scores_remain_untouched():
    build=inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert '01  ·  WYKRYCIE TABLICY' in build
    assert '02  ·  OBRYS TABLICY' in build
    assert '03  ·  ODCZYT NUMERU' in build
    assert 'self._review_section(right)' in build
    assert 'self.pose_gt_instruction.grid_remove()' in build
    assert 'CER' not in build and 'NMCE' not in build and 'Polygon IoU' not in build
    calc=inspect.getsource(ui.MobileSampleReviewWindow._sync_pose_gt_controls)
    assert 'compare_mt_pose_to_gt' in calc
    assert "measured['nmce_pct']" in calc
    assert "measured['polygon_iou_pct']" in calc


def test_questionnaire_skips_placeholders_until_they_are_relevant():
    w=view()
    w.session.review.attempt_annotations['a1'].update(is_plate=True,evaluable=True)
    w._sync_pose_gt_controls()
    assert w.pose_gt_progress.text in ('', 'Zaznaczone rogi: 0/4')
    assert w.pose_gt_metric.text == ''
    assert not w.pose_edit_var.get()


def test_card_surfaces_and_control_labels_are_theme_safe():
    assert ui._review_color_mix('#ffffff','#000000',0.5) == '#808080'
    assert ui._review_color_mix('#202020','#eeeeee',0.04) != '#202020'
    assert ui._review_color_mix('SystemButtonFace','#eeeeee',0.04) == 'SystemButtonFace'
    build = inspect.getsource(ui.MobileSampleReviewWindow._build)
    assert 'review_card_bg' in build
    assert 'review_border' not in build or 'self._review_section' in build


def test_layout_structure_real_tk():
    root=tk.Tk()
    root.withdraw()
    w=ui.MobileSampleReviewWindow.__new__(ui.MobileSampleReviewWindow)
    try:
        w.browser=SimpleNamespace(palette={'panel':'#f1f3e9','fg':'#1b241e','muted':'#5e6c60','accent':'#447c36'}, error='#b22b2b', warning='#b78b20')
        w.bg='#f1f3e9';w.fg='#1b241e';w.muted='#5e6c60';w.accent='#447c36'
        w.review_card_bg=ui._review_color_mix(w.bg,w.fg,0.035)
        w.review_border=ui._review_color_mix(w.bg,w.fg,0.19)
        w.window=tk.Toplevel(root); w.actions=[]
        w._image=None;w._image_source_size=None;w._view_center=None
        w.window.geometry('1450x850+0+0')
        w._build()
        root.update()
        assert w.pose_gt_host.winfo_manager() == 'grid'
        assert w.pose_gt_host.cget('highlightthickness') == 1
        assert w.pose_gt_host.master != w.pose_gt_progress.master
        assert not w.pose_gt_instruction.winfo_ismapped()
        assert w.pose_gt_question.cget('text') == 'Czy możesz wskazać cztery rogi tablicy?'
    finally:
        w.window.destroy()
        root.destroy()
