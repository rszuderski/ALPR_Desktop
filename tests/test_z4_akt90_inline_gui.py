"""Actual Windows Tk transitions for explicit negative ROI decisions."""
import copy
import tkinter as tk
import tkinter.messagebox as messagebox
import pytest
import test_mobile_p2b2_canvas_gui as base
from test_mobile_p2b2_canvas_gui import gui, pump
from test_mobile_p2b2_canvas_review import source_session
from test_z4_akt89lm_gui import ready
from test_z4_akt89lm_gui import invalid_gui
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core


@pytest.fixture
def empty_gui(tmp_path,monkeypatch,request):
    monkeypatch.setattr(base,'source_session',lambda p:source_session(p,n=0,mz=False))
    return request.getfixturevalue('gui')


def test_negative_decision_is_one_click_no_modal_and_persists(empty_gui,monkeypatch):
    root,h,ed,s=empty_gui
    before=copy.deepcopy(s.review.to_dict())
    windows=[w for w in h.window.winfo_children() if isinstance(w,tk.Toplevel)]
    for _ in range(3):ed.sync();ed.overlay();pump(root)
    assert s.review.to_dict()==before and not ed.decision_panel.negative.winfo_ismapped()
    monkeypatch.setattr(messagebox,'askyesno',lambda *a,**k:pytest.fail('No confirmation modal'))
    base.key(h,'space');ready(root,h,ed)
    assert s.review.review_revision==before['review_revision']+1
    assert core.summary(s)['true_negative_frames']==1
    assert 'Wybierz ramkę' in ed.contexts.data['object']
    assert 'Tablice na zdjęciu: 0' in ed.contexts.data['image']
    assert 'Obraz gotowy' in ed.contexts.data['image']
    ed.report_host.render(core.summary(s),ed.inventory())
    assert 'TN (ROI): 1' in ed.report_host.mt_counts.cget('text')
    assert ed.report_host.cards['iou'].value.cget('text')=='—'
    assert [w for w in h.window.winfo_children() if isinstance(w,tk.Toplevel)]==windows
    state=core.scene(s,'mt-one')
    assert core.scene(load(s.path,s.sidecar_path.parent),'mt-one')==state
    ed.undo();ready(root,h,ed)
    assert core.summary(s)['true_negative_frames']==0
    ed.undo(redo=True);ready(root,h,ed)
    assert core.scene(s,'mt-one')['roi_presence']=='NO_PLATES'
    assert core.scene(s,'mt-one')['checked'] and core.summary(s)['true_negative_frames']==1
    assert s.review.review_status=='IN_PROGRESS'  # redo restores the image, not session approval
    assert 'Wybierz ramkę' in ed.contexts.data['object']
    assert 'Obraz gotowy' in ed.contexts.data['image']
    frozen=copy.deepcopy(s.review.to_dict())
    ed.sync();ed.overlay();pump(root)
    assert s.review.to_dict()==frozen


def test_negative_roi_with_two_final_detections_keeps_source_and_gt(gui,monkeypatch):
    root,h,ed,s=gui
    core.set_text(s,'mt-one','GT:a0','HISTORY');ready(root,h,ed)
    before=core.scene(s,'mt-one');ed._choose_plate('GT:a0');pump(root)
    monkeypatch.setattr(messagebox,'askyesno',lambda *a,**k:pytest.fail('No confirmation modal'))
    assert not ed.decision_panel.negative.winfo_ismapped()
    # Canvas Undo explicitly withdraws the mistaken GT; PPM by itself must
    # preserve an established plate. The prior number stays in the journal.
    ed.undo();ready(root,h,ed)
    for oid,point in [('GT:a0',(225,175)),('GT:a1',(445,175))]:
        ed._choose_plate(oid);base.click(h,ed,point,button=3);ready(root,h,ed)
    preview=core.scene(s,'mt-one')
    base.key(h,'space');ready(root,h,ed)
    result=core.summary(s)
    assert result['false_positive_mt']==2 and result['true_negative_frames']==0
    assert result['geometry_n']==result['evaluated_reads']==0
    assert core.scene(s,'mt-one')['parents']==before['parents']
    assert any((e.get('object_after') or {}).get('gt_text')=='HISTORY' for e in core.scene(s,'mt-one')['events'])
    ed.undo();ready(root,h,ed)
    assert core.scene(s,'mt-one')['objects']==preview['objects']


def test_missing_pose_summary_shows_independent_count_and_no_geometry(invalid_gui):
    from test_mobile_p2b2_canvas_review import MANUAL
    root,h,ed,s=invalid_gui
    manual=core.add_plate(s,'mt-one',MANUAL)
    core.set_scene_review_state(s,'mt-one',checked=True);ready(root,h,ed)
    ed._choose_plate(manual);ed.decision_panel.same.invoke();ready(root,h,ed)
    ed.report_host.render(core.summary(s),ed.inventory())
    assert 'Powiązane MT z błędem Pose: 1' in ed.report_host.mt_counts.cget('text')
    assert 'bez dowodu Pose: 0' in ed.report_host.mt_counts.cget('text')
    assert ed.report_host.cards['linked'].value.cget('text')=='1'
    assert ed.report_host.cards['missed'].value.cget('text')=='0'
    assert ed.report_host.cards['iou'].value.cget('text')=='—'
    assert ed.report_host.cards['nmce'].value.cget('text')=='—'
