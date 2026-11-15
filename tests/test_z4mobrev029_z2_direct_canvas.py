"""Z4MOBREV029: directly editable selected polygon; one operator contour, no CTA.

The exercised methods come directly from the shipped GUI source, with no
writes to .alprsession or training data.
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from auto_annotation_tool.ranking.mobile_pose_correction import moved
from test_z4mobrev026_pose_editor_and_table import window, record, PRED, ENTRY, SIZE
from test_z4mobrev027_manual_quad import make_window, click, methods
from test_z4mobrev028_wheel_dual_pose_ui import viewer, method, Canvas, Label
from auto_annotation_tool.ranking.mobile_pose_metrics import compare_mt_pose_to_gt

GUI = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_sample_review.py'
SRC = GUI.read_text('utf-8')
TREE = ast.parse(SRC)
CLS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'MobileSampleReviewWindow')


def getmethod(name, **bindings):
    target=next(n for n in CLS.body if isinstance(n,ast.FunctionDef) and n.name==name)
    mod=ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns=dict(bindings)
    exec(compile(mod,str(GUI),'exec'),ns)
    return ns[name]


def test_only_canvas_based_editor_and_its_shortcuts_are_installed():
    assert 'self.mt_quad_mode_bar = None' in SRC
    assert 'self.mt_drag_button = ttk.Button' not in SRC
    assert 'self.mt_redraw_button = ttk.Button' not in SRC
    assert 'self.canvas.bind(sequence, self._start_mt_redraw)' in SRC
    assert 'self.canvas.bind(sequence, self._delete_mt_operator_correction)' in SRC
    assert 'self.canvas.bind("<B1-Motion>", self._pan_move)' in SRC
    assert 'self.canvas.bind("<ButtonRelease-1>", self._pan_end)' in SRC
    assert 'self.canvas.bind("<ButtonPress-3>", lambda event: self._pan_begin(event, force_pan=True))' in SRC


def test_imprecise_row_automatically_activates_drag_editor_on_source_photo():
    calls=[]
    fake=SimpleNamespace(_saving=False,_mt_corr_active=False,_mt_corr_attempt='',
        _mt_redraw_active=False, pose_edit_var=SimpleNamespace(get=lambda:False),
        _image=object(),_image_entry=ENTRY,
        _visual_target=lambda:('a',{'mt_input_evidence_entry':ENTRY}),
        mt_visual_choice=SimpleNamespace(set=lambda s:calls.append(('choice',s)),get=lambda:'imprecise'),
        mt_visual_radios=[],mt_visual_overlay_buttons=[],
        mt_visual_host=SimpleNamespace(grid_remove=lambda:None),
        mt_visual_status=SimpleNamespace(configure=lambda **kwargs:None),
        session=SimpleNamespace(review=SimpleNamespace(attempt_annotations={
            'a':{'pose_quality':'imprecise'}})),
        _start_mt_correction=lambda aid:calls.append(('start',aid)))
    sync=getmethod('_sync_mt_visual_choice',selected_visual_state=lambda a:a.get('pose_quality'))
    sync(fake)
    assert ('start','a') in calls
    calls.clear();fake._saving=True;sync(fake)
    assert ('start','a') not in calls
    calls.clear();fake._saving=False;fake._mt_redraw_active=True;sync(fake)
    assert ('start','a') not in calls
    calls.clear();fake._mt_redraw_active=False;fake._mt_corr_active=True;fake._mt_corr_attempt='a'
    sync(fake)
    assert ('start','a') not in calls


def test_shift_click_starts_redraw_and_records_the_first_corner():
    calls=[]
    model=SimpleNamespace(
        _image=object(), _saving=False, _mt_redraw_active=False,
        _mt_corr_active=True, _mt_corr_attempt='a',
        _image_entry=ENTRY,
        _visual_target=lambda:('a',{'mt_input_evidence_entry':ENTRY}),
        canvas=SimpleNamespace(focus_set=lambda:None),
        _start_mt_redraw=lambda:calls.append('start'),
        _add_mt_redraw_point=lambda event:calls.append((event.x,event.y)))
    def started():
        calls.append('start');model._mt_redraw_active=True
    model._start_mt_redraw=started
    press=getmethod('_pan_begin',tk=__import__('tkinter'))
    outcome=press(model,SimpleNamespace(x=11,y=17,state=0x0001))
    assert outcome=='break' and calls==['start',(11,17)]


def test_regular_click_drag_remains_direct_corner_edit_not_second_polygon():
    model=SimpleNamespace(_image=object(),_saving=False,_mt_redraw_active=False,
        _mt_corr_active=True,_mt_corr_attempt='a',_multi_bbox_edit=False,
        pose_edit_var=SimpleNamespace(get=lambda:False),
        _visual_target=lambda:('a',{'mt_input_evidence_entry':ENTRY}),_image_entry=ENTRY,
        canvas=SimpleNamespace(focus_set=lambda:None),
        _mt_corr_begin_drag=lambda ev:True)
    press=getmethod('_pan_begin',tk=__import__('tkinter'))
    assert press(model,SimpleNamespace(x=11,y=17,state=0))=='break'


def _delete_harness(*, independent=False, mirrored=False, confirm=True):
    model=window()
    model._image=object()
    model._image_entry=ENTRY
    model._mt_redraw_active=False
    model._mt_corr_active=True
    model._mt_corr_attempt='a'
    model._mt_corr_last_saved=[[12.,10.],[30.,10.],[30.,20.],[10.,20.]]
    model.session.review.attempt_annotations['a']['pose_mt_correction_v1']=record()
    if independent:
        model.session.review.attempt_annotations['a'].update(
            pose_gt_quad_mt_input_px=[[4,4],[35,4],[35,26],[4,26]],
            pose_gt_review_mode='independent_manual_v1')
    if mirrored:
        model.session.review.attempt_annotations['a'].update(
            pose_gt_quad_mt_input_px=model._mt_corr_last_saved,
            pose_gt_review_mode='assisted_manual_redraw_from_mt_v1')
    cleared=[]
    model._cancel_mt_correction=lambda *args:cleared.append(True)
    alerts=[]
    class Messagebox:
        @staticmethod
        def askyesno(*args,**kwargs):
            alerts.append(True)
            return confirm
    erase=getmethod('_delete_mt_operator_correction',
        selected_visual_state=lambda ann:ann.get('pose_quality'),messagebox=Messagebox)
    return model,erase,cleared,alerts


def test_delete_removes_only_operator_correction_and_preserves_provenance():
    model,erase,cleared,alerts=_delete_harness(independent=True)
    original=model.session.review.attempt_annotations['a']['pose_gt_quad_mt_input_px'][:]
    assert erase(model)=='break'
    assert cleared and alerts
    recorded=model.session.saved[-1]
    assert recorded['pose_mt_correction_v1'] is None
    assert recorded['pose_correction_required_v1'] is True
    assert len(recorded['pose_mt_correction_history_v1'])==1
    assert 'pose_gt_quad_mt_input_px' not in recorded
    assert model.session.review.attempt_annotations['a']['pose_gt_quad_mt_input_px']==original
    assert list(PRED[0])==[10.,10.]


def test_delete_legacy_mirrored_assisted_gt_together_with_its_correction():
    model,erase,_,_=_delete_harness(mirrored=True)
    assert erase(model)=='break'
    recorded=model.session.saved[-1]
    assert recorded['pose_gt_quad_mt_input_px'] is None
    assert recorded['pose_gt_review_mode'] is None
    assert recorded['pose_mt_correction_v1'] is None


def test_delete_can_be_cancelled_without_any_write():
    model,erase,_,_=_delete_harness(confirm=False)
    assert erase(model)=='break' and not model.session.saved


def _draw_dual(*, independent=False, editing=True, moved_draft=True):
    v=viewer(corrected=not editing,editing=editing)
    v.canvas=Canvas();v.image_help=Label()
    v.session.review.attempt_annotations['a']['pose_quality']='imprecise'
    if independent:
        v.session.review.attempt_annotations['a'].update(
            pose_gt_quad_mt_input_px=[[3,3],[34,3],[34,25],[3,25]],
            pose_gt_review_mode='independent_manual_v1')
    if editing and not moved_draft:
        v._mt_corr_points=[list(p) for p in PRED]
    calls=[]
    v._draw_pose_gt_overlay=lambda aid,fn:calls.append(aid)
    f=getmethod('_paint_detection_overlay',
        DETECTION_STATUSES={'VALID_QUAD','DETECTION_INVALID_QUAD'},
        detection_box_on_mt_input=lambda row:None,
        _linked_pose_quad_on_mt_input=lambda s,r,sample=None:PRED,
        _pose_source_status=lambda s,r:('',''),
        selected_visual_state=lambda ann:ann.get('pose_quality'),
        compare_mt_pose_to_gt=compare_mt_pose_to_gt,moved=moved)
    f(v,700,420,0,0)
    return v,calls


def test_at_most_one_model_and_one_operator_outline_during_correction():
    v,calls=_draw_dual(independent=True)
    model=[kw for _,kw in v.canvas.polygon if 'source_mt_quad' in kw.get('tags',())]
    operator=[kw for _,kw in v.canvas.polygon if 'operator_quad' in kw.get('tags',())]
    assert len(model)==1 and len(operator)==1 and not calls


def test_no_second_operator_line_if_editing_has_not_moved_any_corner():
    v,calls=_draw_dual(editing=True,moved_draft=False)
    assert sum(1 for _,kw in v.canvas.polygon if 'source_mt_quad' in kw.get('tags',()))==1
    assert not [kw for _,kw in v.canvas.polygon if 'operator_quad' in kw.get('tags',())]
    assert not calls


def test_keyboard_shortcuts_are_canvas_scoped_never_gt_fields():
    assert 'self.canvas.bind(sequence, self._start_mt_redraw)' in SRC
    assert 'self.canvas.bind(sequence, self._delete_mt_operator_correction)' in SRC
    assert 'self.window.bind("<KeyPress-d>"' not in SRC
    assert 'self.window.bind("<Delete>"' not in SRC
