"""Z4MOBREV028: focus-safe wheel, dual immutable/model + editable/operator polygons.

The AST harness executes the final GUI methods, without changing an archive.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from auto_annotation_tool.ranking.mobile_pose_metrics import compare_mt_pose_to_gt
from auto_annotation_tool.ranking.mobile_pose_correction import moved
from test_z4mobrev026_pose_editor_and_table import PRED, SIZE, ENTRY, window, record
from test_z4mobrev027_manual_quad import make_window as manual_window, click

GUI = Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_sample_review.py'
SRC = GUI.read_text(encoding='utf8')

def method(name, **extra):
    tree=ast.parse(SRC)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MobileSampleReviewWindow')
    func=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    namespace=dict(__name__='auto_annotation_tool.gui.z4_mobile_sample_review',__package__='auto_annotation_tool.gui',**extra)
    module=ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(GUI), 'exec'), namespace)
    return namespace[name]

class Canvas:
    def __init__(self):
        self.polygon=[]; self.oval=[]; self.label=[]; self.cursor=[]
        self.size=(700,420); self.pos=(200,90); self.pointer=(450,300)
    def winfo_width(self): return self.size[0]
    def winfo_height(self): return self.size[1]
    def winfo_rootx(self): return self.pos[0]
    def winfo_rooty(self): return self.pos[1]
    def winfo_pointerxy(self): return self.pointer
    def create_polygon(self, *coords, **kw): self.polygon.append((coords,kw));return len(self.polygon)
    def create_line(self, *coords, **kw): self.polygon.append((coords,kw));return len(self.polygon)
    def create_oval(self, *coords, **kw): self.oval.append((coords,kw));return len(self.oval)
    def create_text(self, *coords, **kw): self.label.append((coords,kw));return len(self.label)
    def bbox(self, iid): return (0,0,140,20)
    def move(self, *args): pass
    def configure(self,**kw): self.cursor.append(kw.get('cursor'))

class Label:
    def __init__(self): self.messages=[]
    def configure(self,**kw): self.messages.append(kw.get('text',''))

def viewer(*, corrected=True, redraw=False, editing=False):
    import math
    canvas=Canvas();hint=Label()
    sample=SimpleNamespace(width=100,height=50)
    attempt={'id':'a','mt_status':'VALID_QUAD','mt_detection_index':0,
             'mt_input_evidence_entry':ENTRY,'input_width':100.,'input_height':50.}
    session=SimpleNamespace(attempts={'a':attempt},samples={},
        review=SimpleNamespace(attempt_annotations={'a':{'pose_quality':'imprecise'}}),
        mt_invocations={'inv':SimpleNamespace(evidence_entries=(ENTRY,),detection_count=1,records=(attempt,))})
    obj=SimpleNamespace(canvas=canvas,current_record=('attempt','a'), current_invocation='inv',
                        session=session,_image=sample,_image_source_size=SIZE,_image_entry=ENTRY,
                        _view_center=(50.,25.),_view_pixel_scale=1.,_mt_redraw_active=redraw,
                        _mt_redraw_attempt='a' if redraw else '',
                        _mt_redraw_points=[[12.,10.],[30.,10.]] if redraw else [],
                        _mt_corr_active=editing,_mt_corr_attempt='a' if editing else '',
                        _mt_corr_points=[[12.,10.],[30.,10.],[30.,20.],[10.,20.]],
                        image_help=hint,
                        pose_edit_var=SimpleNamespace(get=lambda:False),
                        _pose_gt_points=[],
                        _pose_gt_attempt_context=lambda:'a',
                        _ensure_pose_gt_points=lambda aid:None,
                        _valid_mt_correction=lambda aid,row,quad:[list(p) for p in ((12,10),(30,10),(30,20),(10,20))] if corrected else None,
                        _draw_pose_gt_overlay=lambda aid,to_canvas:None,
                        _canvas_text_background=lambda *args:None,
                        _paint_mt_visual_overlay=lambda:None)
    scope={'DETECTION_STATUSES':{'VALID_QUAD','DETECTION_INVALID_QUAD'},
           'detection_box_on_mt_input':lambda row:None,
           '_linked_pose_quad_on_mt_input':lambda session,row,sample=None:PRED,
           '_pose_source_status':lambda s,row:('',''),
           'selected_visual_state':lambda annot:annot.get('pose_quality'),
           'compare_mt_pose_to_gt':compare_mt_pose_to_gt,
           'moved':moved}
    draw=method('_paint_detection_overlay',**scope)
    draw(obj,canvas.size[0],canvas.size[1],0,0)
    return obj

def test_source_and_correction_visible_together_as_two_different_polygons():
    h=viewer(corrected=True)
    original=[kw for _,kw in h.canvas.polygon if 'source_mt_quad' in kw.get('tags',())]
    revised=[kw for _,kw in h.canvas.polygon if 'operator_quad' in kw.get('tags',())]
    assert len(original)==len(revised)==1
    assert original[0]['outline']=='#40DCFF' and original[0]['dash']
    assert revised[0]['outline']=='#FFBD59'
    assert max(x['width'] for _,x in h.canvas.polygon)<=2
    assert any('IoU MT ↔ korekta operatora' in message for message in h.image_help.messages)
    assert any('asystowane' in message for message in h.image_help.messages)

def test_original_is_also_shown_during_new_four_click_draft():
    h=viewer(redraw=True)
    assert any('source_mt_quad' in kw.get('tags',()) for _,kw in h.canvas.polygon)
    assert not any('operator_quad' in kw.get('tags',()) for _,kw in h.canvas.polygon)
    assert h.canvas.oval and len(h.canvas.oval)==2
    assert all(kw.get('width',0)<=2 for _,kw in h.canvas.polygon)

def test_unmodified_drag_draft_does_not_publish_misleading_iou_100():
    h=viewer(editing=True)
    h._mt_corr_points=[list(p) for p in PRED]
    h.canvas=Canvas();h.image_help=Label()
    f=method('_paint_detection_overlay',DETECTION_STATUSES={'VALID_QUAD','DETECTION_INVALID_QUAD'},
        detection_box_on_mt_input=lambda row:None,
        _linked_pose_quad_on_mt_input=lambda s,row,sample=None:PRED,
        _pose_source_status=lambda s,row:('',''),
        selected_visual_state=lambda annot:annot.get('pose_quality'),
        compare_mt_pose_to_gt=compare_mt_pose_to_gt,moved=moved)
    f(h,700,420,0,0)
    assert not any('100.00%' in m for m in h.image_help.messages)
    assert any('Korekta jeszcze niezmieniona' in m for m in h.image_help.messages)

def test_wheel_routes_from_focused_other_widget_if_pointer_is_over_image(monkeypatch):
    from auto_annotation_tool.gui import mobile_canvas_zoom
    h=SimpleNamespace(canvas=Canvas(),_image=object())
    zoom=[];monkeypatch.setattr(mobile_canvas_zoom,'request_zoom',lambda host,fac,at:zoom.append((fac,at)))
    h._wheel_zoom=lambda event:method('_wheel_zoom')(h,event)
    route=method('_route_image_wheel')
    assert route(h,SimpleNamespace(widget=object(),delta=120,num=0))=='break'
    assert zoom==[(1.25,(250,210))]
    h.canvas.pointer=(999,999)
    assert route(h,SimpleNamespace(widget=object(),delta=120,num=0)) is None
    assert len(zoom)==1
    h.canvas.pointer=(450,300)
    assert route(h,SimpleNamespace(widget=h.canvas,delta=120,num=0)) is None
    assert len(zoom)==1

def test_wheel_event_has_linux_and_windows_directions_and_no_fake_delta(monkeypatch):
    from auto_annotation_tool.gui import mobile_canvas_zoom
    monkeypatch.setattr(mobile_canvas_zoom,'request_zoom',lambda host,fac,at:states.append((fac,at)))
    h=SimpleNamespace(_image=object(),zoom_image=lambda fac,at:states.append((fac,at)))
    states=[];zoom=method('_wheel_zoom')
    assert zoom(h,SimpleNamespace(delta=-120,num=0,x=30,y=40))=='break'
    assert zoom(h,SimpleNamespace(delta=0,num=4,x=3,y=9))=='break'
    assert zoom(h,SimpleNamespace(delta=0,num=0,x=2,y=2))=='break'
    assert states==[(0.8,(30,40)),(1.25,(3,9))]

def test_manual_redraw_enters_corner_edit_without_another_cta():
    h=manual_window();h._start_mt_redraw()
    for point in [(32,11),(9,21),(9,11),(32,21)]:click(h,*point)
    assert not h._mt_redraw_active
    assert h._mt_corr_active
    assert h._mt_corr_points==[[9.,11.],[32.,11.],[32.,21.],[9.,21.]]

def test_post_draw_corner_drag_updates_assisted_record_but_not_model():
    h=window()
    original=record()
    quad=original['operator_quad_mt_input_px']
    decision=h.session.review.attempt_annotations['a']
    decision.update(pose_mt_correction_v1=original,
                    pose_gt_quad_mt_input_px=[list(p) for p in quad],
                    pose_gt_review_mode='assisted_manual_redraw_from_mt_v1')
    h._start_mt_correction('a')
    h._mt_corr_drag=(('corner',0),(12.,10.),[list(p) for p in quad])
    h._mt_corr_points=[[14.,10.],[30.,10.],[30.,20.],[10.,20.]]
    assert h._mt_corr_finish_drag()
    assert len(h.session.saved)==1
    payload=h.session.saved[0]
    assert payload['pose_gt_quad_mt_input_px']==[[14.,10.],[30.,10.],[30.,20.],[10.,20.]]
    assert payload['pose_mt_correction_v1']['operator_quad_mt_input_px']==payload['pose_gt_quad_mt_input_px']
    assert payload['pose_mt_correction_history_v1']==[original]
    assert PRED[0]==(10.,10.)

def test_independent_gt_never_overwritten_by_assisted_drag():
    h=window();old_gt=[[5,8],[31,8],[31,21],[5,21]]
    decision=h.session.review.attempt_annotations['a']
    decision['pose_gt_quad_mt_input_px']=old_gt
    decision['pose_gt_review_mode']='independent_manual_v1'
    h._start_mt_correction('a')
    h._mt_corr_drag=(('corner',0),(10.,10.),[list(p) for p in PRED])
    h._mt_corr_points=[[12.,10.],[30.,10.],[30.,20.],[10.,20.]]
    assert h._mt_corr_finish_drag()
    assert 'pose_gt_quad_mt_input_px' not in h.session.saved[0]
    assert decision['pose_gt_quad_mt_input_px']==old_gt

def test_conflicting_assisted_gt_rejects_drag_instead_of_desynchronizing():
    h=window();original=record()
    decision=h.session.review.attempt_annotations['a']
    decision.update(pose_mt_correction_v1=original,
                    pose_gt_quad_mt_input_px=[[5,5],[30,5],[30,20],[5,20]],
                    pose_gt_review_mode='assisted_manual_redraw_from_mt_v1')
    statuses=[];h.status_var=SimpleNamespace(set=lambda t:statuses.append(t))
    h._start_mt_correction('a')
    h._mt_corr_drag=(('corner',0),(12.,10.),[list(p) for p in original['operator_quad_mt_input_px']])
    h._mt_corr_points=[[14.,10.],[30.,10.],[30.,20.],[10.,20.]]
    assert h._mt_corr_finish_drag()
    assert not h.session.saved
    assert any('Konflikt' in s for s in statuses)

def test_hover_uses_corner_then_body_cursor():
    h=window()
    h._start_mt_correction('a')
    h._saving=False
    h.canvas.cursor=[]
    h.canvas.configure=lambda **kw:h.canvas.cursor.append(kw.get('cursor'))
    hover=method('_update_mt_edit_cursor',hit_target=__import__('auto_annotation_tool.ranking.mobile_pose_correction',fromlist=['hit_target']).hit_target)
    h._mt_corr_canvas_points=lambda _attempt:[(50.,50.),(150.,50.),(150.,100.),(50.,100.)]
    hover(h,SimpleNamespace(x=50,y=50))
    hover(h,SimpleNamespace(x=100,y=75))
    hover(h,SimpleNamespace(x=300,y=200))
    assert h.canvas.cursor==['hand2','fleur','crosshair']

def test_scientific_independence_was_not_changed():
    s=(GUI.parents[1]/'ranking/mobile_scientific_review.py').read_text('utf8')
    assert 'manual_assisted_gt_n' in s
    assert 'independent_manual_v1' in s
    assert 'assisted_delta_iou_mean_pct' in s

def test_native_tk_canvas_renders_thin_two_color_outlines():
    import tkinter as tk
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('native Tk not available')
    try:
        root.geometry('760x500')
        canvas=tk.Canvas(root,width=700,height=420)
        canvas.pack()
        root.update()
        obj=viewer(corrected=True)
        obj.canvas=canvas
        obj.image_help=tk.Label(root,text='')
        draw=method('_paint_detection_overlay',
            DETECTION_STATUSES={'VALID_QUAD','DETECTION_INVALID_QUAD'},
            detection_box_on_mt_input=lambda r:None,
            _linked_pose_quad_on_mt_input=lambda s,r,sample=None:PRED,
            _pose_source_status=lambda s,r:('',''),
            selected_visual_state=lambda annot:annot.get('pose_quality'),
            compare_mt_pose_to_gt=compare_mt_pose_to_gt,moved=moved)
        draw(obj,700,420,0,0)
        orig=canvas.find_withtag('source_mt_quad')
        corr=canvas.find_withtag('operator_quad')
        assert len(orig)==1
        assert len(corr)>=1
        assert canvas.itemcget(orig[0],'outline')=='#40DCFF'
        assert canvas.itemcget(orig[0],'dash')
        assert any(canvas.itemcget(i,'outline')=='#FFBD59' for i in corr)
        assert 'IoU MT' in obj.image_help.cget('text')
    finally:
        root.destroy()
