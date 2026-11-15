"""Exact source segments/points, separate from positive-area metric BBOXes."""
from __future__ import annotations
import math

REASONS = dict(MISSING='Brak współrzędnych BBOX', NONFINITE='Nieskończone lub nieliczbowe współrzędne BBOX',
    UNTRUSTED='Niepotwierdzona przestrzeń klatki / ROI', OUT_OF_BOUNDS='BBOX poza granicami obrazu; bez przycinania',
    REVERSED='Odwrócone granice BBOX', ZERO_HEIGHT='wysokość 0 px (h=0)', ZERO_WIDTH='szerokość 0 px (w=0)',
    POINT='punkt (w=0, h=0)', VALID='Poprawny prostokąt BBOX')

def describe(session, gid, scene, candidate):
    from ..ranking import mobile_canvas_review as core
    from ..ranking.mobile_canvas_resolution import verify_frame
    row=session.attempts.get(candidate['attempt_id'], {})
    result=dict(marker=False, bbox=None, validity='MISSING', reason=REASONS['MISSING'], identity=None)
    def finish(status):
        validity={'ZERO_HEIGHT':'INVALID_ZERO_HEIGHT','ZERO_WIDTH':'INVALID_ZERO_WIDTH','POINT':'INVALID_POINT',
                  'NONFINITE':'INVALID_NONFINITE','UNTRUSTED':'UNTRUSTED_COORDINATE_SPACE'}.get(status,status)
        result.update(validity=validity, reason=REASONS[status]); return result
    fields=('plate_left','plate_top','plate_right','plate_bottom')
    if any(row.get(k) in (None,'') for k in fields):return result
    try: box=[float(row[k]) for k in fields]
    except (ValueError,TypeError,OverflowError):return finish('NONFINITE')
    if not all(math.isfinite(v) for v in box):return finish('NONFINITE')
    result['bbox']=box
    try:
        ctx=core.context(session,gid)
        if (not ctx or row.get('source_frame_coordinate_space')!='mt_source_bitmap_xy'
            or row.get('source_frame_origin')!='pipeline_bitmap_before_roi'
            or scene['source_binding']!=core._binding(session,ctx,scene['parents'])
            or list(ctx['size'])!=scene['image_size'] or list(ctx['roi'])!=scene['roi']
            or ctx['entry']!=scene['image_entry']):return finish('UNTRUSTED')
        stamp=session.path.stat()
        cachekey=(gid,scene['source_binding'],stamp.st_size,stamp.st_mtime_ns)
        cache=getattr(session,'_bbox_marker_verified_frames',set())
        if cachekey not in cache:
            verify_frame(session,gid); cache.add(cachekey); session._bbox_marker_verified_frames=cache
    except (ValueError,KeyError,TypeError,OSError):return finish('UNTRUSTED')
    l,t,r,b=box; w,h=scene['image_size']
    if l>r or t>b:return finish('REVERSED')
    if not (0<=l<=r<=w and 0<=t<=b<=h):return finish('OUT_OF_BOUNDS')
    status='POINT' if l==r and t==b else 'ZERO_HEIGHT' if t==b else 'ZERO_WIDTH' if l==r else 'VALID'
    result.update(marker=status!='VALID', identity=dict(session_id=session.session_id,
        source_archive_sha256=session.bundle.source_archive_sha256, source_binding=scene['source_binding'],
        invocation_id=gid, attempt_id=candidate['attempt_id'], detection_index=candidate['mt_detection_index'],
        geometry_kind='BBOX', image_entry=ctx['entry'], image_size=list(ctx['size']),roi=list(ctx['roi']),
        source_sequence=ctx['source_sequence'],source_timestamp_nanos=ctx['source_timestamp_nanos']))
    return finish(status)

def enrich(session,gid,scene,candidate,value):
    descriptor=describe(session,gid,scene,candidate)
    value=dict(value,bbox_diagnostic=descriptor)
    if descriptor['marker'] and not any(l['kind']=='BBOX' for l in value['layers']):
        value['layers']=[dict(kind='BBOX',role='BBOX niepoprawny — '+descriptor['reason'],
            bbox=descriptor['bbox'],quad=None,marker=descriptor),*value['layers']]
        value['bbox_marker']=descriptor
    return value

def quad(marker):
    l,t,r,b=marker['bbox'];return [(l,t),(r,t),(r,b),(l,b)]

def dpi(ed):
    return max(1.,float(ed.c.winfo_fpixels('1i'))/96.)

def paint(ed,marker,*,color,tags):
    l,t,r,b=marker['bbox'];a=ed.to_canvas((l,t));z=ed.to_canvas((r,b));scale=dpi(ed)
    tags=(*tags,'bbox_zero_marker')
    if a==z:
        radius=5*scale
        for dx,dy in ((radius,0),(0,radius)):
            ed.c.create_line(a[0]-dx,a[1]-dy,a[0]+dx,a[1]+dy,fill='#0C1117',width=6*scale,tags=tags)
            ed.c.create_line(a[0]-dx,a[1]-dy,a[0]+dx,a[1]+dy,fill=color,width=2.5*scale,tags=tags+('bbox_marker_cross',))
    else:
        ed.c.create_line(*a,*z,fill='#0C1117',width=6*scale,tags=tags)
        ed.c.create_line(*a,*z,fill=color,width=2.5*scale,tags=tags+('bbox_marker_segment',))
    for x,y in (a,z) if a!=z else (a,):
        radius=3*scale
        ed.c.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,outline='#0C1117',width=scale,tags=tags+('bbox_marker_endpoint',))

def hit(ed,x,y):
    from .mobile_canvas_visibility import contours
    frames=ed.shift_preview.frames() if getattr(ed,'shift_preview',None) and ed.shift_preview.active else contours(ed)
    hits=[]
    for frame in frames:
        if not frame.get('marker') or frame.get('role','FINAL')!='FINAL' or frame.get('gid',ed.h.current_invocation)!=ed.h.current_invocation:continue
        l,t,r,b=frame['marker']['bbox'];a=ed.to_canvas((l,t));z=ed.to_canvas((r,b))
        vx,vy=z[0]-a[0],z[1]-a[1];length=vx*vx+vy*vy
        f=max(0,min(1,((x-a[0])*vx+(y-a[1])*vy)/length)) if length else 0
        distance=math.hypot(x-a[0]-f*vx,y-a[1]-f*vy)
        if distance<=8*dpi(ed):hits.append((distance,frame['oid']))
    return (min(hits)[1],'layer:BBOX',None) if hits else None

def verdict(ed,candidate):
    from . import mobile_bbox_spatial_review as review
    try:
        value=review.record(ed.h.session,ed.h.current_invocation,candidate)
        return value['spatial_verdict'] if value else 'UNREVIEWED'
    except (ValueError,OSError,KeyError):return 'NOT_ASSESSABLE_SOURCE'

def apply(ed,operation):
    try:
        operation();ed.sync();ed.request_overlay();ed.message('Zapisano ocenę położenia BBOX. Wyniki IoU, Pose i weryfikacja MT pozostają bez zmian.')
    except (ValueError,OSError,KeyError) as exc:ed.message('Nie zapisano oceny BBOX: '+str(exc))

def popup(ed,event):
    import tkinter as tk
    from . import mobile_canvas_clean_view as clean, mobile_bbox_spatial_review as review
    row=clean.selected(ed)
    if not row or not row['candidate'] or ed.busy or ed.h._saving or ed.drag or ed.mode=='draw':return
    session=ed.h.session;gid=ed.h.current_invocation;oid=row['candidate']['object_id']
    try:revision=review.load(session)['diagnostic_revision']
    except (ValueError,OSError) as exc:ed.message(str(exc));return
    previous=getattr(ed,'_bbox_spatial_menu',None)
    if previous is not None:
        try:previous.destroy()
        except tk.TclError:pass
    menu=tk.Menu(ed.c,tearoff=False,**clean.menu_colors(ed.p));ed._bbox_spatial_menu=menu
    def choose(value):
        menu.unpost();menu.destroy();ed.c.focus_set()
        if ed.h.session is not session or ed.h.current_invocation!=gid:return
        apply(ed,lambda:review.set_verdict(session,gid,oid,value,expected_revision=revision))
    for value in ('ON_PLATE','OFF_PLATES_BAD','UNDECIDABLE','UNREVIEWED'):
        menu.add_command(label=review.LABELS[value],command=lambda value=value:choose(value))
    try:menu.tk_popup(ed.c.winfo_rootx()+event.x,ed.c.winfo_rooty()+event.y)
    finally:
        try:menu.grab_release()
        except tk.TclError:pass  # The chosen command may have destroyed it.
