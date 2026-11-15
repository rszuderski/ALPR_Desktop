"""AKT70: pointer/key selection, handles and editable drag without losing pan.

Pure synthetic scene + one actual Tk dispatch. No .alprsession or GT file writes.
"""
from __future__ import annotations
import copy
import math
from types import SimpleNamespace
import tkinter as tk
import pytest
from test_z4_canvas_linked_measurement_tree import load_editor


class DummyCanvas:
    def __init__(self):
        self.ops=[]
        self.props={'cursor':'arrow'}
        self.focused=False
        self.pointer=(35,30)
        self.origin=(210,140)
    def winfo_width(self): return 500
    def winfo_height(self): return 350
    def winfo_rootx(self): return self.origin[0]
    def winfo_rooty(self): return self.origin[1]
    def winfo_pointerx(self): return self.origin[0]+self.pointer[0]
    def winfo_pointery(self): return self.origin[1]+self.pointer[1]
    def configure(self,**kw): self.props.update(kw)
    def cget(self,k):return self.props.get(k)
    def delete(self,*tags): self.ops=[]
    def tag_raise(self,*tags): pass
    def create_rectangle(self,*args,**kw): self.ops.append(('rect',args,kw))
    def create_polygon(self,*args,**kw): self.ops.append(('polygon',args,kw))
    def create_line(self,*args,**kw): self.ops.append(('line',args,kw))
    def create_oval(self,*args,**kw): self.ops.append(('oval',args,kw))
    def create_text(self,*args,**kw): self.ops.append(('text',args,kw))
    def focus_set(self):self.focused=True
    def after_idle(self,cb):cb();return 'idle'


def scene_fixture():
    quad=[[20.,30.],[50.,30.],[50.,40.],[20.,40.]]
    return dict(image_entry='source.png',image_size=(100,100),roi=(0,0,100,100),
        parents=[dict(parent_id='m1',quad=copy.deepcopy(quad),mz_prediction='EZ123',detection_index=0,mz_ran=True)],
        objects=[dict(object_id='o1',parent_id='m1',quad=copy.deepcopy(quad),
                      post_nms_suppressed=False,gt_text='EZ123',error=False,
                      text_exclusion_reason='')], checked=False,exclusion_reason='')


def setup(canvas=None):
    mod=load_editor()
    s=scene_fixture()
    c=canvas or DummyCanvas()
    messages=[];corrected=[]
    def report(msg):messages.append(msg)
    h=SimpleNamespace(current_invocation='G', session=SimpleNamespace(),
         _image=SimpleNamespace(width=100,height=100),_image_entry='source.png',
         _image_source_size=(100,100),_view_center=(50.,50.),_view_pixel_scale=1.0,
         _view_drag_at=None,_closed=False,_saving=False,
         image_help=SimpleNamespace(configure=lambda **kw:None),
         status_var=SimpleNamespace(set=report),
         window=SimpleNamespace(bind=lambda *a,**kw:None))
    h.paints=0
    def paint():h.paints+=1
    h._paint_image=paint
    e=mod.CanvasReviewEditor.__new__(mod.CanvasReviewEditor)
    e.h=h; e.c=c
    e.mode='select';e.selected='';e.drag=None;e._click_candidate=None
    e.entry=None;e.pointer=None;e.pointer_canvas=(35,30);e.points=[];e.hover=''
    e.badges={};e.card_positions={};e.paint_scheduled=False;e.paint_after=None
    e.roi_overlay_visible=True;e.show_roi_overlay=True
    e.right_press_state=None
    e.selection_key_down=False;e.selection_release_after=None
    e.read_error='';e.plate_fit_active=False;e.plate_fit_gid=''
    e._read=lambda:s
    e.to_canvas=lambda p:tuple(p)
    e.to_source=lambda x,y:(x,y)
    e.message=report
    e.available=lambda:True
    e.finish_entry=lambda:True
    e.clear_tree_selection=lambda:None
    e.sync=lambda:None
    e.request_overlay=lambda:e.overlay()
    def correct(_session,_gid,oid,quad):
        assert oid=='o1'
        corrected.append(copy.deepcopy(quad))
        s['objects'][0]['quad']=copy.deepcopy(quad)
        return True
    mod.core.correct_plate=correct
    mod.core.compare_mt_pose_to_gt=lambda *args:{'polygon_iou_pct':95.0,'nmce_pct':1.2}
    e._commit=lambda action,description:action()
    return mod,e,c,h,s,messages,corrected


def key_event(c, *, x=-100, y=-150, state=0):
    return SimpleNamespace(widget=c,x=x,y=y,state=state,keysym='s')


def event(x,y):return SimpleNamespace(x=x,y=y)


def overlay_handles(c):
    if isinstance(c,DummyCanvas):
        return [op for op in c.ops if op[0]=='oval' and 'corner_handle' in op[2].get('tags',())]
    return c.find_withtag('corner_handle')


def test_s_ignores_invalid_key_event_xy_and_selects_under_actual_pointer():
    mod,e,c,h,s,msg,corrected=setup()
    original=copy.deepcopy(s)
    e.overlay();assert not overlay_handles(c)
    assert e.select_mode(key_event(c))=='break'
    assert e.selected=='o1' and e.mode=='edit'
    assert len(overlay_handles(c))==4
    assert s==original
    assert 'Ramka wybrana' in msg[-1]


def test_s_toggles_off_and_hides_corner_handles():
    mod,e,c,h,s,msg,corrected=setup()
    e.select_mode(key_event(c));assert len(overlay_handles(c))==4
    e.select_mode(key_event(c));assert e.selected=='' and e.mode=='select'
    assert not overlay_handles(c)


def test_click_selects_and_displays_four_draggable_corners():
    mod,e,c,h,s,msg,corrected=setup()
    old=copy.deepcopy(s)
    e.press(event(35,30));assert e.selected==''
    e.release(event(35,30))
    assert e.selected=='o1' and e.mode=='edit'
    assert len(overlay_handles(c))==4
    assert s==old


def test_click_on_frame_does_not_deselect_and_corner_drag_changes_only_gt():
    mod,e,c,h,s,msg,corrected=setup()
    e.press(event(35,30));e.release(event(35,30))
    original_mt=copy.deepcopy(s['parents'][0]['quad'])
    e.press(event(20,30));assert e.drag and e.drag['index']==0
    e.motion(event(25,28));assert s['objects'][0]['quad'][0]==[20,30]
    assert e.drag['quad'][0]==[25,28]
    assert len(overlay_handles(c))==4
    e.release(event(25,28))
    assert e.drag is None and e.selected=='o1'
    assert s['objects'][0]['quad'][0]==[25,28]
    assert s['parents'][0]['quad']==original_mt
    assert len(corrected)==1
    assert len(overlay_handles(c))==4


def test_drag_on_frame_pans_without_selecting_or_modifying_gt():
    mod,e,c,h,s,msg,corrected=setup()
    original=copy.deepcopy(s)
    e.press(event(35,30))
    assert e._click_candidate is not None
    e.motion(event(55,60))
    assert e._click_candidate is None and h.paints>0
    e.release(event(55,60))
    assert e.selected=='' and s==original and corrected==[]


def test_pan_after_selection_retains_selected_and_does_not_move_corner():
    mod,e,c,h,s,msg,corrected=setup()
    e.select_mode(key_event(c))
    original=copy.deepcopy(s)
    e.press(event(35,30))
    e.motion(event(56,62))
    e.release(event(56,62))
    assert e.selected=='o1'
    assert s==original
    assert len(overlay_handles(c))==4


def test_badge_click_keeps_text_editor_behavior():
    mod,e,c,h,s,msg,corrected=setup()
    called=[]
    e.hit=lambda x,y:('o1','badge',None)
    e.start_entry=lambda oid:called.append(oid)
    e.press(event(42,80)); e.release(event(42,80))
    assert called==['o1'] and e.selected==''


def test_no_editing_suppressed_raw_candidate():
    mod,e,c,h,s,msg,corrected=setup()
    s['objects'][0]['post_nms_suppressed']=True
    e.overlay()
    e.select_mode(key_event(c))
    assert e.selected==''
    assert not overlay_handles(c)


def test_real_tk_canvas_keyboard_xy_can_disagree_with_mouse_pointer():
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('No Tk display available')
    try:
        root.geometry('650x480+160+160')
        c=tk.Canvas(root,width=540,height=370, bg='white',takefocus=1)
        c.pack()
        root.update()
        mod,e,c,h,s,msg,corrected=setup(c)
        h.window=root
        root.update()
        c.focus_force()
        # Move the real pointer INSIDE the test plate, but do not assume
        # pixel-exact Tk ``warp`` coordinates. On Windows, the OS-reported
        # pointer can differ by several pixels because of DPI/scaling.
        target_x, target_y = 35, 35  # comfortably inside x=20..50, y=30..40
        c.event_generate('<Motion>', x=target_x, y=target_y, warp=1)
        root.update()
        def actual_pointer():
            return (int(c.winfo_pointerx()-c.winfo_rootx()),
                    int(c.winfo_pointery()-c.winfo_rooty()))
        pointer_x, pointer_y = actual_pointer()
        for _ in range(3):
            if 22 <= pointer_x <= 48 and 31 <= pointer_y <= 39:
                break
            # Correct for a display-specific offset, then check the *actual*
            # pointer again. This keeps the hit test meaningful on Windows.
            c.event_generate('<Motion>',
                x=int(target_x + target_x - pointer_x),
                y=int(target_y + target_y - pointer_y),
                warp=1)
            root.update()
            pointer_x, pointer_y = actual_pointer()
        assert 22 <= pointer_x <= 48 and 31 <= pointer_y <= 39, (
            'Tk warp did not move the pointer onto the test frame',
            (pointer_x, pointer_y))
        # Capture an actual Tk key event; its x/y should NOT be used for hover.
        observed=[]
        c.bind('<KeyPress-s>',lambda ev:observed.append((ev.x,ev.y)))
        c.event_generate('<KeyPress-s>',when='tail')
        root.update()
        assert observed and observed[-1] != (pointer_x, pointer_y)
        e.bindings()
        c.focus_force()
        c.event_generate('<KeyPress-s>',when='tail')
        root.update()
        assert e.selected=='o1' and e.mode=='edit'
        assert len(overlay_handles(c))==4
        # Corner drag is now actually dispatched by Tk's canvas bindings.
        c.event_generate('<ButtonPress-1>',x=20,y=30)
        root.update()
        assert e.drag and e.drag['index']==0
        c.event_generate('<B1-Motion>',x=25,y=28)
        root.update()
        c.event_generate('<ButtonRelease-1>',x=25,y=28)
        root.update()
        assert corrected and corrected[-1][0]==[25,28]
    finally:
        root.destroy()
