"""Z4MOBREV027: four manual corners on original MT image, no source mutation.

These tests use the exact shipped GUI method AST on a controlled Tk-independent
harness (not an approximation of the implementation). An optional real Tk smoke
is exercised separately if DISPLAY is available.
"""
from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_z4mobrev026_pose_editor_and_table import window, PRED, SIZE, ENTRY, SHA

GUI_FILE = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_sample_review.py'
GUI = GUI_FILE.read_text('utf-8')


def _ordered_operator_gt_quad(points, w, h):
    # Controlled order that models the production manual click sorter for
    # convex rectangles; actual project PolygonValidator is used in production.
    from auto_annotation_tool.ranking.mobile_pose_metrics import validate_mt_quad
    center = (sum(p[0] for p in points) / len(points),
              sum(p[1] for p in points) / len(points))
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    i = min(range(len(ordered)), key=lambda j: (sum(ordered[j]), ordered[j][1], ordered[j][0]))
    return validate_mt_quad(ordered[i:] + ordered[:i], w, h)


def methods():
    tree=ast.parse(GUI)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MobileSampleReviewWindow')
    targets={'_resume_mt_correction','_start_mt_redraw','_add_mt_redraw_point','_cancel_mt_correction'}
    selected={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in targets}
    assert set(selected)==targets
    scope={'_linked_pose_quad_on_mt_input':lambda s,a:PRED,
           'selected_visual_state':lambda a:a.get('pose_quality') or '',
           '_ordered_operator_gt_quad':_ordered_operator_gt_quad,
           'POSE_GT_SCHEMA':'manual_mt_input_quad_v1',
           'math':math,
           'messagebox':SimpleNamespace(askyesno=lambda *args,**kwargs:True)}
    from auto_annotation_tool.ranking.mobile_pose_correction import make_correction
    scope['make_correction']=make_correction
    mod=ast.Module(body=list(selected.values()),type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod,'gui-027-under-test','exec'),scope)
    return scope


def make_window():
    w=window()
    for name,func in methods().items():
        if name.startswith('_') and callable(func):
            setattr(w,name,types.MethodType(func,w))
    w._mt_redraw_active=False
    w._mt_redraw_attempt=''
    w._mt_redraw_points=[]
    w._image_source_size=SIZE
    w._sync_mt_visual_choice=lambda:None
    w._pose_mt_to_canvas=lambda p,a:tuple(p)
    w._pose_canvas_to_mt_input=lambda e,a:(e.x,e.y)
    w._view_center=(50.,25.)
    w.session.review.attempt_annotations['a']={'pose_quality':'imprecise','is_plate':True,
                                                'evaluable':True,'pose_correction_required_v1':True}
    w.msg=[]
    w.status_var=SimpleNamespace(set=lambda msg:w.msg.append(msg))
    return w


def click(w,x,y):
    return w._add_mt_redraw_point(SimpleNamespace(x=x,y=y))


def test_canvas_editing_replaces_artificial_cta_buttons():
    assert 'self.mt_quad_mode_bar = None' in GUI
    assert 'self.mt_drag_button = ttk.Button' not in GUI
    assert 'self.mt_redraw_button = ttk.Button' not in GUI
    assert 'self.canvas.bind("<KeyPress-d>"' not in GUI or '_start_mt_redraw' in GUI
    assert 'self.canvas.bind(sequence, self._start_mt_redraw)' in GUI
    assert 'self.canvas.bind(sequence, self._delete_mt_operator_correction)' in GUI


def test_start_four_click_drawing_is_empty_and_saves_nothing():
    w=make_window();w._start_mt_redraw()
    assert w._mt_redraw_active and w._mt_redraw_attempt=='a'
    assert w._mt_redraw_points==[]
    assert not w.session.saved
    assert w._mt_corr_active is False
    assert '0/4' in w.msg[-1]


def test_four_corners_save_one_assisted_correction_without_imputing_gt():
    w=make_window();w._start_mt_redraw()
    for point in [(32,11),(9,21),(9,11),(32,21)]:
        assert click(w,*point)
    assert not w._mt_redraw_active
    assert len(w.session.saved)==1
    saved=w.session.saved[0]
    assert 'pose_gt_quad_mt_input_px' not in saved
    assert 'pose_gt_schema' not in saved
    record=saved['pose_mt_correction_v1']
    assert record['operator_quad_mt_input_px']==[[9.,11.],[32.,11.],[32.,21.],[9.,21.]]
    assert record['drawing_method']=='four_manual_corner_clicks_v1'
    assert record['model_prediction_sha256'] != SHA
    assert saved['pose_correction_required_v1'] is True
    assert list(PRED[0])==[10.,10.]


def test_three_clicks_are_only_a_local_draft():
    w=make_window();w._start_mt_redraw()
    for point in [(9,11),(32,11),(9,21)]:click(w,*point)
    assert len(w._mt_redraw_points)==3 and w.session.saved==[]
    assert '3/4' in w.msg[-1]
    assert w._cancel_mt_correction()=='break'
    assert not w._mt_redraw_active and w.session.saved==[]
    assert w._mt_redraw_points==[]


def test_double_click_does_not_add_same_corner_twice():
    w=make_window();w._start_mt_redraw()
    click(w,9,11); click(w,9,11)
    assert len(w._mt_redraw_points)==1


def test_invalid_fourth_corner_keeps_three_and_never_writes():
    w=make_window();w._start_mt_redraw()
    for point in [(9,11),(32,11),(9,21),(19,11)]:click(w,*point)
    # fourth interior point has no convex quadrilateral
    assert len(w._mt_redraw_points)==3
    assert not w.session.saved
    assert 'Nie zapisano obrysu' in w.msg[-1]


def test_existing_independent_gt_allows_a_separate_assisted_correction():
    w=make_window()
    independent=[[10,10],[30,10],[30,20],[10,20]]
    w.session.review.attempt_annotations['a'].update(
        pose_gt_quad_mt_input_px=independent,pose_gt_review_mode='independent_manual_v1')
    w._start_mt_redraw()
    assert w._mt_redraw_active
    for point in [(32,11),(9,21),(9,11),(32,21)]:
        click(w,*point)
    assert w.session.saved and 'pose_gt_quad_mt_input_px' not in w.session.saved[0]
    assert w.session.review.attempt_annotations['a']['pose_gt_quad_mt_input_px']==independent


def test_broken_source_or_wrong_active_attempt_rejected():
    w=make_window()
    w._image_entry='other.jpg'
    w._start_mt_redraw();assert not w._mt_redraw_active
    w._image_entry=ENTRY
    w._start_mt_redraw();assert w._mt_redraw_active
    w._visual_target=lambda:('b',{'id':'b','mt_input_evidence_entry':ENTRY,'input_width':100.,'input_height':50.})
    assert click(w,10,10) is False
    assert not w._mt_redraw_active and not w.session.saved


def test_nonimprecise_selection_never_starts_redraw():
    w=make_window();w.session.review.attempt_annotations['a']['pose_quality']='accurate'
    w._start_mt_redraw()
    assert not w._mt_redraw_active and not w.session.saved


def test_redraw_ui_preserves_original_mt_reference_and_zoom_pan():
    assert 'if self._mt_redraw_active and self._mt_redraw_attempt == selected_aid' in GUI
    assert 'return  # Both geometries are visualized; model data remain immutable.' in GUI
    assert 'tags=("source_mt_quad", "detection_overlay")' in GUI
    assert 'if not self._mt_redraw_active:\n            self._paint_multi_overlays' in GUI
    assert 'if not force_pan and self._mt_redraw_active:' in GUI
    assert '_pose_canvas_to_mt_input(event, aid)' in GUI
    assert 'if self._mt_redraw_active and (event is None or getattr(event, "num", 1) == 1):' in GUI


def test_stored_GT_provenance_never_appears_as_blinded_in_science():
    scientific=(Path(__file__).resolve().parents[1]/'auto_annotation_tool/ranking/mobile_scientific_review.py').read_text('utf-8')
    assert 'in {"independent_manual_v1", "blinded_manual_v1", "manual_blinded_v1"}' in scientific
    assert 'assisted_manual_redraw_from_mt_v1' not in scientific


def test_older_026_drag_mode_still_edits_without_restarting_annotation():
    w=make_window();w._start_mt_redraw()
    w._resume_mt_correction()
    assert not w._mt_redraw_active
    assert w._mt_corr_active
    assert w._mt_corr_points == [list(p) for p in PRED]
    assert not w.session.saved


def test_native_tk_has_direct_editing_without_mode_buttons():
    import tkinter as tk
    from tkinter import ttk
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('Native Tk display unavailable')
    try:
        program=ast.parse(GUI)
        cls=next(n for n in program.body if isinstance(n,ast.ClassDef) and n.name=='MobileSampleReviewWindow')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sync_mt_visual_choice')
        mod=ast.Module(body=[fn],type_ignores=[])
        ast.fix_missing_locations(mod)
        scope={'selected_visual_state':lambda a:a.get('pose_quality') or ''}
        exec(compile(mod,'native-tk-027','exec'),scope)
        self=SimpleNamespace()
        self._saving=True # no restart while a save is pending
        self._mt_redraw_active=False
        self._mt_corr_active=False
        self.pose_edit_var=SimpleNamespace(get=lambda:False)
        self._image=SimpleNamespace(size=SIZE)
        self._image_entry=ENTRY
        self._visual_target=lambda:('a',{'mt_input_evidence_entry':ENTRY})
        self.session=SimpleNamespace(review=SimpleNamespace(attempt_annotations={
            'a':{'pose_quality':'imprecise'}}))
        self.mt_visual_choice=tk.StringVar(master=root)
        self.mt_visual_host=ttk.Frame(root)
        self.mt_visual_status=ttk.Label(root)
        self.mt_visual_overlay_buttons=[]
        self.mt_quad_mode_bar=None
        scope['_sync_mt_visual_choice'](self)
        root.update_idletasks()
        assert self.mt_quad_mode_bar is None
        assert self.mt_visual_choice.get()=='imprecise'
    finally:
        root.destroy()


def test_redraw_preserves_previous_assisted_correction_in_history():
    from test_z4mobrev026_pose_editor_and_table import record
    w=make_window()
    old=record()
    w.session.review.attempt_annotations['a']['pose_mt_correction_v1']=old
    w._start_mt_redraw()
    assert w._mt_redraw_active
    for x,y in [(9,11),(32,11),(32,21),(9,21)]: click(w,x,y)
    assert len(w.session.saved)==1
    new=w.session.saved[0]
    assert new['pose_mt_correction_v1']['operator_quad_mt_input_px'] != old['operator_quad_mt_input_px']
    assert new['pose_mt_correction_history_v1']==[old]


def test_existing_correction_requires_explicit_consent_to_redraw():
    from test_z4mobrev026_pose_editor_and_table import record
    w=make_window();w.session.review.attempt_annotations['a']['pose_mt_correction_v1']=record()
    scope=w._start_mt_redraw.__func__.__globals__
    original=scope['messagebox']
    scope['messagebox']=SimpleNamespace(askyesno=lambda *args,**kwargs:False)
    try:
        w._start_mt_redraw()
        assert not w._mt_redraw_active and not w.session.saved
    finally:
        scope['messagebox']=original
