"""A3 viewport only: native wheel/pan events on a private K02 archive."""
import copy
from types import SimpleNamespace

import pytest

from test_mobile_f5a_assisted import ref,session,native_gui,populated_call
import test_mobile_p2b2_canvas_gui as native
from auto_annotation_tool.gui import mobile_canvas_zoom as zoom
from auto_annotation_tool.gui.mobile_canvas_visibility import contours
from auto_annotation_tool.gui import mobile_canvas_navigator as nav


def settle(root,h):
    native.pump(root,lambda:not getattr(getattr(h,'_smooth_zoom',None),'active',False),seconds=5)
    root.update_idletasks()


def prepare(gui,ref,monkeypatch,size='1180x760'):
    root,h,ed,s=gui;ui=h._f5a_a3;gid,fid,plates,aids=populated_call(ref,s)
    ui.select('a3:mt:'+gid)
    native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
    h.window.state('normal');h.window.geometry(size);root.update();h.fit_image();native.focus_canvas(h);root.update_idletasks()
    point=[h.canvas.winfo_width()//2,h.canvas.winfo_height()//2]
    pointer=lambda:(h.canvas.winfo_rootx()+point[0],h.canvas.winfo_rooty()+point[1])
    monkeypatch.setattr(h.window,'winfo_pointerxy',pointer);monkeypatch.setattr(h.canvas,'winfo_pointerxy',pointer)
    return root,h,ed,s,ui,gid,aids,point


def wheel(root,h,point,delta,*,wait=True,widget=None):
    (widget or h.canvas).event_generate('<MouseWheel>',x=point[0],y=point[1],delta=delta)
    if wait:settle(root,h)


def mapped(h,p):
    """Independent source->viewport equation, also used by the bitmap crop."""
    sx=h._image.width/h._image_source_size[0];sy=h._image.height/h._image_source_size[1]
    return ((p[0]*sx-h._view_center[0])*h._view_pixel_scale+h.canvas.winfo_width()/2,
            (p[1]*sy-h._view_center[1])*h._view_pixel_scale+h.canvas.winfo_height()/2)


def assert_overlays(h,ed):
    items=sorted(contours(ed),key=lambda item:item['kind']=='GTR')
    drawn=h.canvas.find_withtag('navigator_geometry')
    assert items and drawn
    matched=set()
    for model in items:
        # One source contour can have contrast/selection strokes. All must
        # share the same geometry; these strokes are not extra detections.
        handles=[i for i in drawn if model['oid'] in h.canvas.gettags(i) and model['kind'] in h.canvas.gettags(i)]
        assert handles;matched.update(handles)
        for handle in handles:assert h.canvas.coords(handle)==pytest.approx([v for p in model['quad'] for v in mapped(h,p)],abs=1e-7)
    assert matched==set(drawn)
    l,t,r,b=ed._read()['roi'];expected=(*mapped(h,(l,t)),*mapped(h,(r,b)))
    assert h.canvas.find_withtag('analyzed_roi')
    for handle in h.canvas.find_withtag('analyzed_roi'):assert h.canvas.coords(handle)==pytest.approx(expected,abs=1e-7)
    f=h._f5a_a3.ref.frames[h._f5a_a3.current_frame]
    scene=h.session.scene_frames.get(f['scene_event_id'],{})
    observations=[m['selected_rois'] for m in scene.get('mp_observations',[]) if m['source_stamp']==f['source_stamp']]
    if observations and all(r==observations[0] for r in observations):
        assert len(h.canvas.find_withtag('a3_source_mp'))==len(observations[0])
        for roi,handle in zip(observations[0],h.canvas.find_withtag('a3_source_mp')):
            assert h.canvas.coords(handle)==pytest.approx((*mapped(h,(roi['left'],roi['top'])),*mapped(h,(roi['right'],roi['bottom']))),abs=1e-7)


@pytest.mark.parametrize('size',['1180x760','1649x896'])
@pytest.mark.parametrize('where',['plate','edge'])
def test_native_10_in_10_out_anchor_layout_bitmap_and_overlay(native_gui,ref,monkeypatch,size,where):
    root,h,ed,s,ui,gid,aids,point=prepare(native_gui,ref,monkeypatch,size)
    obj=next(o for o in ed._read()['objects'] if o['attempt_id']==aids[0])
    source=[sum(p[j] for p in obj['quad'])/4 for j in (0,1)] if where=='plate' else [h._image_source_size[0]*.015,h._image_source_size[1]*.985]
    point[:]=[round(v) for v in mapped(h,source)];anchor=ed.to_source(*point)
    assert 0<=point[0]<h.canvas.winfo_width() and 0<=point[1]<h.canvas.winfo_height()
    before=copy.deepcopy(s.review.to_dict());source_before=copy.deepcopy(s.attempts);center=h._view_center;fit=h._view_fit_scale
    photo=h._photo;bitmap=h.canvas.find_withtag('evidence_image');dimensions=(h.canvas.winfo_width(),h.canvas.winfo_height());frames=[];resizes=[];requests=[];errors=[]
    paint=h._paint_image;request=zoom.request_zoom
    def observed(*a,**k):
        paint(*a,**k)
        try:
            assert (h.canvas.winfo_width(),h.canvas.winfo_height())==dimensions
            assert h.canvas.find_withtag('evidence_image')==bitmap and h._photo is photo
            assert h._view_fit_scale==fit and ed.to_source(*point)==pytest.approx(anchor,abs=1e-7)
            assert_overlays(h,ed)
        except AssertionError as exc:errors.append(str(exc))
        frames.append(h._view_zoom)
    def requested(host,factor,at):requests.append((factor,at));return request(host,factor,at)
    h.canvas.bind('<Configure>',lambda e:resizes.append((e.width,e.height)),add='+')
    with monkeypatch.context() as guard:
        guard.setattr(h,'_paint_image',observed);guard.setattr(zoom,'request_zoom',requested)
        guard.setattr(h,'_load_image',lambda *a:pytest.fail('Wheel must not reload source/archive'))
        guard.setattr(ui,'refresh_reference',lambda:pytest.fail('Wheel must not rebuild the GT preview'))
        guard.setattr(ui,'update_header',lambda:pytest.fail('Wheel must not rebuild progress/header'))
        for direction in (1,-1):
            for index in range(10):
                start=len(frames);prior=h._view_zoom;wheel(root,h,point,120*direction)
                assert not errors,errors
                expected=1.12**(index+1 if direction==1 else 9-index)
                assert h._view_zoom==pytest.approx(expected)
                segment=[prior,*frames[start:]]
                assert len(segment)>2 and all((b-a)*direction>=-1e-10 for a,b in zip(segment,segment[1:]))
    assert not resizes and len(requests)==20 and h._view_center==pytest.approx(center,abs=1e-7)
    assert s.review.to_dict()==before and s.attempts==source_before and not s.sidecar_path.exists()


def test_native_fractional_burst_routing_and_reversal_do_not_duplicate_events(native_gui,ref,monkeypatch):
    root,h,ed,s,ui,gid,aids,point=prepare(native_gui,ref,monkeypatch)
    for _ in range(8):wheel(root,h,point,15,wait=False)
    controller=h._smooth_zoom;assert controller.target==pytest.approx(1.12)
    timer=controller.after_id;assert timer;settle(root,h);assert h._view_zoom==pytest.approx(1.12)
    # Wheel delivered to the focused Treeview must route once to the Canvas.
    before=h._view_zoom;wheel(root,h,point,120,widget=h.subject_tree);assert h._view_zoom==pytest.approx(before*1.12)
    wheel(root,h,point,360,wait=False);native.pump(root,lambda:h._view_zoom>before*1.12)
    current=h._view_zoom;wheel(root,h,point,-120,wait=False)
    assert controller.target==pytest.approx(max(1,current/1.12));settle(root,h)
    assert h._view_zoom==pytest.approx(max(1,current/1.12)) and not s.sidecar_path.exists()


def test_native_pan_bbox_pose_mt_navigation_and_fit(native_gui,ref,monkeypatch):
    root,h,ed,s,ui,gid,aids,point=prepare(native_gui,ref,monkeypatch)
    # Exercise an already saved operator assessment, not only an empty sidecar.
    ui.confirm_identity();ui.judge(gid,aids[0],'TP',ref.plates(ref.frames[ui.current_frame]['photo_sha256'])[0]['object_id'])
    root.update();saved=s.sidecar_path.read_bytes()
    assert nav.select_layer(ed,'BBOX');root.update_idletasks()
    wheel(root,h,point,120);assert_overlays(h,ed)
    assert nav.select_layer(ed,'POSE');root.update_idletasks()
    wheel(root,h,point,120);assert_overlays(h,ed)
    before=copy.deepcopy(s.review.to_dict());center=h._view_center;scale=h._view_pixel_scale;level=h._view_zoom
    h.canvas.event_generate('<ButtonPress-1>',x=20,y=30)
    h.canvas.event_generate('<B1-Motion>',x=77,y=63)
    h.canvas.event_generate('<ButtonRelease-1>',x=77,y=63);root.update()
    assert h._view_center==pytest.approx((center[0]-57/scale,center[1]-33/scale)) and h._view_zoom==level
    assert_overlays(h,ed)
    view=(h._view_zoom,h._view_center);ui.update_header();ed._render_contexts({});root.update()
    assert (h._view_zoom,h._view_center)==view
    # Same existing F shortcut restores the whole frame after zoom and pan.
    native.key(h,'f');assert h._view_zoom==1 and h._view_center==pytest.approx((h._image.width/2,h._image.height/2))
    eid=ui.current_exposure;calls=[ui.targets[i][1] for i in h.subject_tree.get_children('a3:exp:'+eid) if i in ui.targets and ui.targets[i][0]=='mt']
    assert len(calls)>=2
    for number in range(2):
        ui.select('a3:mt:'+calls[number]);fid=ref.call_frame(s,calls[number])
        native.pump(root,lambda:h._image is not None and h._image_entry==ref.frames[fid]['full_frame_entry'])
        h.fit_image();root.update_idletasks();point[:]=[h.canvas.winfo_width()//2,h.canvas.winfo_height()//2]
        wheel(root,h,point,120)
        assert h.current_invocation==calls[number] and h._view_zoom==pytest.approx(1.12)
        assert_overlays(h,ed)
    assert s.review.to_dict()==before and s.sidecar_path.read_bytes()==saved


def test_a3_steps_leave_default_zoom_contract_unchanged():
    for delta in (15,120,240,-15,-120,-240):
        event=SimpleNamespace(delta=delta,num=0)
        assert zoom.wheel_factor(event)==pytest.approx(1.25**(delta/120))
        assert zoom.wheel_factor(event,step_factor=1.12)==pytest.approx(1.12**(delta/120))
