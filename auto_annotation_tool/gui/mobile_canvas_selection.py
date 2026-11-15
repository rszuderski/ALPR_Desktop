"""Transient active contour and a short outside click, independent of review."""
import math,time

CLICK_SECONDS=.25
MOVE_PIXELS=5

def contains(ed,quad,x,y):
    from .mobile_canvas_editor import quad_contains_point
    from ..ranking.mobile_manual_frame_guard import edge_distance
    return quad_contains_point(quad,*ed.to_source(x,y)) or edge_distance((x,y),[ed.to_canvas(p) for p in quad])<=7

def context(ed):
    revision=(ed.reference_source.data.data['revision'] if getattr(ed,'reference_source',None)
              else ed.h.session.review.review_revision)
    return (id(ed.h.session),ed.h.current_invocation,ed.h._image_entry,ed.selected,
            revision,ed.h._view_zoom,ed.c.winfo_width(),ed.c.winfo_height())

def outside(ed,x,y):
    if getattr(ed,'reference_source',None):
        return not ed.hit(x,y) and not any(o['quad'] and contains(ed,o['quad'],x,y) for o in ed._read()['objects'])
    from .mobile_canvas_navigator import hit
    from .mobile_canvas_visibility import contours
    if hit(ed,x,y) is not None:return False
    shapes=ed.shift_preview.frames() if ed.shift_preview.active else contours(ed)
    return not any(contains(ed,f['quad'],x,y) for f in shapes)

def arm(ed,event,*,had_entry=False):
    if (not had_entry and ed.selected and ed.mode=='edit' and not ed.drag and not ed.points
            and outside(ed,event.x,event.y)):
        ed._outside_click=dict(start=time.monotonic(),point=(event.x,event.y),context=context(ed),moved=False)

def motion(ed,event):
    state=getattr(ed,'_outside_click',None)
    if state and math.dist(state['point'],(event.x,event.y))>MOVE_PIXELS:state['moved']=True

def release(ed,event):
    state=getattr(ed,'_outside_click',None);ed._outside_click=None
    if (state and event is not None and not state['moved'] and time.monotonic()-state['start']<=CLICK_SECONDS
            and math.dist(state['point'],(event.x,event.y))<=MOVE_PIXELS and state['context']==context(ed)
            and outside(ed,event.x,event.y)):
        ed.escape();return True
    return False

def active(ed,frame):
    from .mobile_canvas_clean_view import selected
    from .mobile_canvas_geometry_layers import model
    if not ed.selected or ed.mode!='edit' or ed.drag or frame.get('gid',ed.h.current_invocation)!=ed.h.current_invocation:return False
    value=model(ed);row=selected(ed)
    kind='GTR' if ed.gt_focus_id or not value else value['active']
    if frame['kind']!=kind:return False
    return frame['oid']==ed.selected or bool(kind=='GTR' and row and row['pid'] and frame['oid']==row['pid'])

def accent(ed,frame,width=3):
    """Thin white halo with dark outer edge; semantic color stays unchanged."""
    from .mobile_canvas_editor import TAG
    if not active(ed,frame):return
    tags=(TAG,'active_frame_accent','navigator_geometry',frame['kind'],ed.selected)
    marker=frame.get('marker')
    if marker:
        from .mobile_canvas_bbox_marker import dpi
        l,t,r,b=marker['bbox'];a=ed.to_canvas((l,t));z=ed.to_canvas((r,b));scale=dpi(ed)
        segments=[(*a,*z)] if a!=z else [(a[0]-5*scale,a[1],a[0]+5*scale,a[1]),(a[0],a[1]-5*scale,a[0],a[1]+5*scale)]
        for segment in segments:
            for color,w in (('#080808',10),('#FFFFFF',8)):ed.c.create_line(*segment,fill=color,width=w*scale,tags=tags)
    else:
        points=[v for p in frame['quad'] for v in ed.to_canvas(p)]
        for color,w in (('#080808',width+6),('#FFFFFF',width+4)):
            ed.c.create_polygon(*points,fill='',outline=color,width=w,tags=tags)
