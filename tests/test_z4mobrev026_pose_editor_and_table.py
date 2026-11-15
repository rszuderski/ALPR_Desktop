"""Z4MOBREV026 — guarded Pose correction and readable verification matrix.

Uses real editor methods extracted from the shipped GUI (not reimplemented
logic); all source/mobile measurements remain read-only in these tests.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from auto_annotation_tool.ranking.mobile_pose_correction import (
    CORRECTION_SCHEMA, CORRECTION_METHOD, correction_for_attempt,
    hit_target, move_polygon, moved, make_correction,
)
from auto_annotation_tool.ranking.mobile_verification_policy import (
    visual_decision, subject_verification_issues,
)

PRED=((10.,10.),(30.,10.),(30.,20.),(10.,20.))
SIZE=(100.,50.)
SHA='c'*64
ENTRY='samples/evidence/a.jpg'


def record(corrected=None):
    return make_correction(attempt_id='a',source_sha=SHA,image_entry=ENTRY,
        size=SIZE,prediction=PRED,corrected=corrected or ((12.,10.),(30.,10.),(30.,20.),(10.,20.)))


def load_record(value, **changes):
    data=dict(attempt_id='a',source_sha=SHA,image_entry=ENTRY,size=SIZE,prediction=PRED)
    data.update(changes)
    return correction_for_attempt({'pose_mt_correction_v1': value},**data)


def test_correction_has_explicit_assisted_provenance():
    v=record()
    assert v['schema']==CORRECTION_SCHEMA
    assert v['method']==CORRECTION_METHOD
    assert v['attempt_id']=='a' and v['source_archive_sha256']==SHA
    assert v['operator_quad_mt_input_px'][0]==[12.,10.]
    assert v['model_prediction_sha256'] != SHA
    assert v['mt_input_evidence_entry']==ENTRY

@pytest.mark.parametrize('field,invalid',[
    ('attempt_id','b'),('source_archive_sha256','f'*64),
    ('mt_input_evidence_entry','other.jpg'),('input_size_px',[150.,50.]),
    ('model_prediction_sha256','e'*64),('method','independent_manual')])
def test_mismatched_saved_provenance_fails_closed(field,invalid):
    v=record();v[field]=invalid
    with pytest.raises(ValueError):load_record(v)

def test_not_reviewed_does_not_impute_gt():
    assert correction_for_attempt({},attempt_id='a',source_sha=SHA,
           image_entry=ENTRY,size=SIZE,prediction=PRED) is None

def test_source_predicted_quad_cannot_be_saved_as_manual_correction():
    with pytest.raises(ValueError,match='Najpierw'):
        make_correction(attempt_id='a',source_sha=SHA,image_entry=ENTRY,
                        size=SIZE,prediction=PRED,corrected=PRED)

def test_invalid_and_crossed_corrections_rejected():
    for points in [((10,10),(30,20),(30,10),(10,20)),
                   ((-1,10),(30,10),(30,20),(10,20)),
                   ((10,10),(10,10),(30,20),(10,20))]:
        with pytest.raises(ValueError):
            make_correction(attempt_id='a',source_sha=SHA,image_entry=ENTRY,
                            size=SIZE,prediction=PRED,corrected=points)

def test_handle_hit_precedes_polygon_pan():
    assert hit_target((11,11),PRED,radius=4)==('corner',0)
    assert hit_target((20,15),PRED,radius=4)==('body',None)
    assert hit_target((42,42),PRED,radius=4) is None

def test_move_corner_and_move_entire_body():
    point=move_polygon(PRED,('corner',0),(10,10),(12,10),SIZE)
    assert point[0]==[12,10] and point[1]==[30,10]
    full=move_polygon(PRED,('body',None),(20,15),(22,18),SIZE)
    assert full==[[12,13],[32,13],[32,23],[12,23]]
    assert moved(PRED,full)
    assert PRED[0]==(10,10)  # immutable original prediction

def test_drag_outside_image_is_rejected():
    with pytest.raises(ValueError):
        move_polygon(PRED,('body',None),(20,15),(-100,15),SIZE)

def test_evil_nonfinite_drag_rejected():
    with pytest.raises(ValueError):
        move_polygon(PRED,('corner',0),(10,10),(math.nan,20),SIZE)

def source_methods():
    src=(ROOT/'auto_annotation_tool/gui/z4_mobile_sample_review.py').read_text(encoding='utf8')
    nodes=ast.parse(src)
    cls=next(x for x in nodes.body if isinstance(x,ast.ClassDef) and x.name=='MobileSampleReviewWindow')
    names={'save_mt_visual_choice','_valid_mt_correction','_start_mt_correction',
           '_cancel_mt_correction','_mt_corr_canvas_points','_mt_corr_begin_drag',
           '_mt_corr_move_drag','_mt_corr_finish_drag',
           '_status_yscroll','_schedule_status_pills','_render_status_pills','_choose_status_row'}
    selected=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in selected}==names
    module=ast.Module(body=[ast.ClassDef(name='EditorHarness',bases=[],keywords=[],
                              body=selected,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(module)
    return module,src


def window():
    module,src=source_methods()
    import tkinter as tk
    from tkinter import ttk
    namespace={'tk':tk,'ttk':ttk,'math':math,'correction_for_attempt':correction_for_attempt,
               'make_correction':make_correction,'move_polygon':move_polygon,'moved':moved,
               'hit_target':hit_target,'visual_decision':visual_decision,
               'selected_visual_state':lambda a: a.get('pose_quality') or '',
               '_linked_pose_quad_on_mt_input':lambda session,attempt: PRED}
    exec(compile(module,'gui-under-test','exec'),namespace)
    h=namespace['EditorHarness']()
    h._saving=False
    h._image=SimpleNamespace(width=100,height=50)
    h._image_entry=ENTRY
    h._view_center=(50.,25.)
    h._view_pixel_scale=1.
    h._view_fit_scale=1.
    h._mt_corr_active=False
    h._mt_corr_attempt=''
    h._mt_corr_points=[]
    h._mt_corr_drag=None
    h._mt_corr_last_saved=[]
    h._multi_bbox_edit=False
    h.mt_visual_choice=SimpleNamespace(get=lambda:'imprecise')
    h.status_var=SimpleNamespace(set=lambda s:None)
    h.pose_edit_var=SimpleNamespace(get=lambda:False)
    h._leave_pose_gt_edit_on_navigation=lambda:None
    h._paint_image=lambda:None
    h._pose_canvas_to_mt_input=lambda event,aid:(event.x,event.y)
    h.canvas=SimpleNamespace(winfo_width=lambda:100,winfo_height=lambda:50,
                              configure=lambda **k:None)
    h.window=None
    attempt={'id':'a','mt_input_evidence_entry':ENTRY,'input_width':100.,'input_height':50.}
    h._visual_target=lambda:('a',attempt)
    annotation={'is_plate':True,'pose_quality':'imprecise','pose_correction_required_v1':True}
    review=SimpleNamespace(source_archive_sha256=SHA,attempt_annotations={'a':annotation})
    class Session:
        def __init__(self): self.review=review;self.saved=[]
        def annotate(self,kind,aid,**fields):
            assert kind=='attempt' and aid=='a'
            self.review.attempt_annotations[aid].update(fields)
            self.saved.append(fields)
    h.session=Session()
    h._change=lambda operation,save_draft=False,after=None:(operation(),after() if after else None)
    return h


def test_editor_does_not_save_by_simply_selecting_imprecise():
    h=window()
    h._start_mt_correction('a')
    assert h._mt_corr_active
    assert h._mt_corr_points==[list(x) for x in PRED]
    assert not h.session.saved


def test_direct_drag_saves_once_and_not_the_model_or_old_gt():
    h=window();h._start_mt_correction('a')
    assert h._mt_corr_begin_drag(SimpleNamespace(x=10,y=10))
    assert h._mt_corr_move_drag(SimpleNamespace(x=12,y=10))
    assert h._mt_corr_finish_drag()
    assert len(h.session.saved)==1
    assert h.session.saved[0]['pose_mt_correction_v1']['operator_quad_mt_input_px'][0]==[12,10]
    assert 'pose_gt_quad_mt_input_px' not in h.session.saved[0]
    assert PRED[0]==(10,10)


def test_click_without_movement_never_writes():
    h=window();h._start_mt_correction('a')
    h._mt_corr_begin_drag(SimpleNamespace(x=10,y=10))
    h._mt_corr_finish_drag()
    assert not h.session.saved


def test_resume_uses_saved_quad_and_escape_discards_draft():
    h=window();h.session.review.attempt_annotations['a']['pose_mt_correction_v1']=record()
    h._start_mt_correction('a')
    assert h._mt_corr_points[0]==[12,10]
    h._mt_corr_points[0]=[16,10]  # uncommitted local edit
    assert h._cancel_mt_correction()=='break'
    h._start_mt_correction('a')
    assert h._mt_corr_points[0]==[12,10]


def test_separate_attempt_never_resumes_another_plate_correction():
    h=window();h.session.review.attempt_annotations['a']['pose_mt_correction_v1']=record()
    h._start_mt_correction('a')
    assert h._mt_corr_attempt=='a'
    h._visual_target=lambda:('b',{'id':'b','mt_input_evidence_entry':ENTRY,
                                  'input_width':100.,'input_height':50.})
    assert h._mt_corr_begin_drag(SimpleNamespace(x=10,y=10)) is False
    h._cancel_mt_correction()
    assert h._mt_corr_attempt==''


def test_explicit_new_correction_pending_vs_legacy_imprecise(tmp_path,monkeypatch):
    from test_z4mobrev022_form_verification import FakeSession
    session=FakeSession(tmp_path,annotation=dict(visual_decision('imprecise'),pose_correction_required_v1=True))
    # Our lightweight test source does not include full dependencies of the GUI;
    # monkeypatch only the immutable Pose lookup used by the policy.
    import sys,types
    stub=types.ModuleType('auto_annotation_tool.gui.z4_mobile_sample_review')
    stub._linked_pose_quad_on_mt_input=lambda s,a:PRED
    monkeypatch.setitem(sys.modules,'auto_annotation_tool.gui.z4_mobile_sample_review',stub)
    assert any('Niezakończona korekta' in issue for issue in subject_verification_issues(session,'subject1'))
    session.review.attempt_annotations['attempt1']=visual_decision('imprecise')
    assert not subject_verification_issues(session,'subject1')


def test_saved_current_correction_satisfies_new_guard(tmp_path,monkeypatch):
    from test_z4mobrev022_form_verification import FakeSession
    import sys,types
    stub=types.ModuleType('auto_annotation_tool.gui.z4_mobile_sample_review')
    stub._linked_pose_quad_on_mt_input=lambda s,a:PRED
    monkeypatch.setitem(sys.modules,'auto_annotation_tool.gui.z4_mobile_sample_review',stub)
    session=FakeSession(tmp_path,annotation=dict(visual_decision('imprecise'),pose_correction_required_v1=True))
    a=session.attempts['attempt1'];a.update(input_width=100.,input_height=50.)
    session.review.attempt_annotations['attempt1']['pose_mt_correction_v1']=make_correction(
        attempt_id='attempt1',source_sha='a'*64,image_entry='evidence/1.jpg',
        size=SIZE,prediction=PRED,corrected=((12,10),(30,10),(30,20),(10,20)))
    assert not subject_verification_issues(session,'subject1')


def test_GUI_contract_one_choice_and_no_duplicate_sections():
    _,src=source_methods()
    assert 'tk.Radiobutton(self.mt_visual_overlay' in src
    assert 'self.mt_visual_host.grid_remove()' in src
    assert 'self.pose_gt_host.grid_remove()' in src
    assert '_mt_corr_begin_drag' in src and '_mt_corr_finish_drag' in src
    assert 'KOREKTA OPERATORA' in src and '#40DCFF' in src
    assert 'self._paint_multi_overlays(width, height)' in src
    assert 'def _focus_multi_gt' in src


def test_tk_per_cell_pills_and_blue_selected_row():
    import tkinter as tk
    from tkinter import ttk
    try: root=tk.Tk()
    except tk.TclError: pytest.skip('Tk display unavailable')
    try:
        root.geometry('650x290')
        columns=('Nr','MT','Wsk.','GT','Rogi','Form.')
        tree=ttk.Treeview(root,columns=columns,show='tree headings',height=5)
        tree.column('#0',width=210)
        for col in columns:tree.heading(col,text=col);tree.column(col,width=60)
        tree.pack(fill='both',expand=True)
        tree.insert('','end',iid='a',text='Obserwacja 01',values=('01','✓','○','—','✓','○'))
        tree.insert('','end',iid='b',text='Obserwacja 02',values=('02','—','—','—','—','✓'))
        root.update()
        mod,_=source_methods()
        ns={'tk':tk,'ttk':ttk}
        exec(compile(mod,'gui-under-test','exec'),ns)
        obj=ns['EditorHarness']()
        obj.subject_tree=tree
        obj._status_pills={};obj._status_paint_scheduled=False
        obj._select_subject=lambda:None
        obj._render_status_pills()
        assert obj._status_pills[('a','MT')].cget('bg')=='#E1F3E6'
        assert obj._status_pills[('a','Wsk.')].cget('bg')=='#FFF1D4'
        tree.selection_set('a');root.update()
        obj._render_status_pills()
        assert obj._status_pills[('a','MT')].cget('bg')=='#C2E9FD'
        assert obj._status_pills[('a','MT')].winfo_manager()=='place'
        assert len(obj._status_pills)==10
    finally:root.destroy()


def test_multi_plate_guided_correction_blocks_both_ready_and_scientific_count(tmp_path,monkeypatch):
    from test_z4mobrev024_multi_plate import MultiSession, populated
    from auto_annotation_tool.ranking.mobile_multi_plate_verification import (
        group_issues, multi_scientific_summary)
    from auto_annotation_tool.ranking.mobile_verification_policy import verification_progress
    import sys,types
    stub=types.ModuleType("auto_annotation_tool.gui.z4_mobile_sample_review")
    stub._linked_pose_quad_on_mt_input=lambda session,attempt:PRED
    monkeypatch.setitem(sys.modules,"auto_annotation_tool.gui.z4_mobile_sample_review",stub)
    s=populated(MultiSession(tmp_path,n_pred=4,mz=False))
    s.review.attempt_annotations["at1"] = dict(visual_decision("imprecise"), pose_correction_required_v1=True)
    assert any("Niezakończona korekta" in issue for issue in group_issues(s,"gid"))
    assert verification_progress(s)["status"] == "IN_PROGRESS"
    assert multi_scientific_summary(s)["verified_multi_invocations"] == 0
    s.review.attempt_annotations["at1"]["pose_mt_correction_v1"] = make_correction(
        attempt_id="at1",source_sha="a"*64,image_entry="evidence.jpg",size=(640.,480.),
        prediction=PRED,corrected=((12,10),(30,10),(30,20),(10,20)))
    assert group_issues(s,"gid")==[]
    assert verification_progress(s)["status"] == "READY_TO_APPROVE"
    assert multi_scientific_summary(s)["verified_multi_invocations"] == 1
