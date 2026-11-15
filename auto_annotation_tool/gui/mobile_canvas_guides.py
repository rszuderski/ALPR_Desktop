"""Source-aligned, view-only rulers. Motion touches only these Canvas items."""
from __future__ import annotations
import tkinter as tk

TAG='gtr_guides'
CORE='#165D8C'
HALO='#EAF5FC'

def clear(ed):
    ed.c.delete(TAG);ed._gtr_guide_items=[]

def image_bounds(ed):
    s=ed._read();h=ed.h
    if (not s or h._image is None or h._view_center is None or h._image_entry!=s['image_entry']
            or tuple(h._image_source_size or ())!=tuple(s['image_size'])):return None
    a=ed.to_canvas((0,0));b=ed.to_canvas(s['image_size'])
    box=(max(0,a[0]),max(0,a[1]),min(ed.c.winfo_width(),b[0]),min(ed.c.winfo_height(),b[1]))
    return box if box[0]<box[2] and box[1]<box[3] else None

def _spans(lo,hi,cuts):
    spans=[(lo,hi)]
    for a,b in cuts:
        spans=[piece for l,r in spans for piece in ((l,min(r,a)),(max(l,b),r)) if piece[1]>piece[0]]
    return spans

def update(ed):
    active=(ed.mode=='draw' or bool(ed.drag)) and ed.entry is None
    if active:ed.c.configure(cursor='crosshair')
    elif ed.c.cget('cursor')=='crosshair':ed.c.configure(cursor='arrow')
    box=image_bounds(ed) if active else None;pointer=ed.pointer_canvas
    if not box or pointer is None or not(box[0]<=pointer[0]<=box[2] and box[1]<=pointer[1]<=box[3]):
        clear(ed);return
    x,y=(ed.to_canvas(ed.drag['quad'][ed.drag['index']]) if ed.drag else pointer)
    l,t,r,b=box
    obstacles=getattr(ed,'clean_label_boxes',[])
    horizontal=_spans(l,r,[(a-2,c+2) for a,bb,c,d in obstacles if bb-2<=y<=d+2])
    vertical=_spans(t,b,[(bb-2,d+2) for a,bb,c,d in obstacles if a-2<=x<=c+2])
    segments=[(a,y,bb,y) for a,bb in horizontal]+[(x,a,x,bb) for a,bb in vertical]
    items=getattr(ed,'_gtr_guide_items',[])
    if len(items)!=2*len(segments) or any(not ed.c.type(i) for i in items):
        clear(ed);items=[]
        # Canvas widths are device pixels in the DPI-aware app. Integer,
        # continuous strokes keep a one-pixel centre. A two-pixel underlay
        # leaves just one light edge, rather than a three-pixel-wide ruler.
        for color,width,tag in ((HALO,2,'gtr_guide_halo'),(CORE,1,'gtr_guide_core')):
            for coords in segments:
                items.append(ed.c.create_line(*coords,fill=color,width=width,capstyle=tk.BUTT,tags=(TAG,tag)))
        ed._gtr_guide_items=items
    else:
        for i,coords in zip(items,segments*2):ed.c.coords(i,*coords)
    ed.c.tag_raise(TAG)
    for tag in ('drawing_corner','corner_handle'):
        if ed.c.find_withtag(tag):ed.c.tag_raise(tag)

def update_draft(ed):
    """The pending segment follows the same current source mapping as a click."""
    from .mobile_canvas_editor import TAG as OVERLAY
    box=image_bounds(ed);xy=ed.pointer_canvas
    ed.pointer=(ed.to_source(*xy) if box and xy and box[0]<=xy[0]<=box[2] and box[1]<=xy[1]<=box[3] else None)
    pts=list(ed.points)+([ed.pointer] if ed.points and ed.pointer is not None else [])
    line=ed.c.find_withtag('drawing_line')
    if len(pts)>1:
        coords=[v for p in pts for v in ed.to_canvas(p)]
        if line:ed.c.coords(line[0],*coords)
        else:ed.c.create_line(*coords,fill='#FFD400',width=2,tags=(OVERLAY,'drawing_line'))
    elif line:ed.c.delete('drawing_line')
    update(ed)
