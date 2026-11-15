"""Immediate Canvas-only corner preview; the existing release owns the save."""
from __future__ import annotations
import copy


def _prepare(ed,key):
    from .mobile_canvas_editor import TAG
    from . import mobile_canvas_clean_view as clean,mobile_canvas_guides as guides
    from .mobile_canvas_approval_visuals import PENDING,polygon
    row=clean.selected(ed)
    if not row or not row['gt']:return None
    if ed.paint_after is not None:
        ed.c.after_cancel(ed.paint_after);ed.paint_after=None;ed.paint_scheduled=False
    oid=ed.drag['oid'];context=getattr(ed,'_context_badge',None)
    if not context or context['oid']!=oid or context['kind']!='GTR':context=None
    badge_items=[]
    for item in ed.c.find_withtag(TAG):
        tags=ed.c.gettags(item)
        if any(t in tags for t in ('image_verification_border','image_border_contrast')):continue
        if context and oid in tags and any(t in tags for t in ('active_layer_badge','stack_position_badges','badge_tether')):
            if 'badge_tether' not in tags:badge_items.append(item)
            continue
        ed.c.delete(item)
    guides.clear(ed)
    points=[ed.to_canvas(p) for p in ed.drag['quad']];coords=[v for p in points for v in p]
    shape=polygon(ed,coords,color=PENDING,width=3,tags=(TAG,'operator_quad','navigator_geometry','GTR',oid,row['pid']))
    underlay=ed.c.find_withtag('navigator_contrast_underlay')[-1]
    handles=[ed.c.create_oval(x-5,y-5,x+5,y+5,fill='white',outline=PENDING,
                             tags=(TAG,'corner_handle',oid)) for x,y in points]
    ed.reference_identity_hits={};ed.mt_source_badge_hits={}
    if context:ed.c.itemconfigure('context_badge_background',outline=PENDING)
    else:ed.badges={};ed._stack_counter=None
    return dict(key=key,shape=shape,underlay=underlay,handles=handles,context=context,badge_items=badge_items,
                hidden=False,published_quad=copy.deepcopy(ed.drag['quad']))


def _badge(ed,state,box):
    from . import mobile_canvas_clean_view as clean
    context=state['context']
    if not context:return
    context['quad']=copy.deepcopy(ed.drag['quad'])
    counter=getattr(ed,'_stack_counter',None)
    rects=[context['rect']]+([counter['group_rect']] if counter else [])
    group=(min(r[0] for r in rects),min(r[1] for r in rects),max(r[2] for r in rects),max(r[3] for r in rects))
    clipped=group[0]<2 or group[1]<2 or group[2]>ed.c.winfo_width()-2 or group[3]>ed.c.winfo_height()-2
    if state['hidden'] or clipped or clean.intersects(group,box):
        ed.clean_label_boxes=[]
        target=clean.badge_position(ed,box,group[2]-group[0],group[3]-group[1])
        state['hidden']=target is None
        for item in state['badge_items']:ed.c.itemconfigure(item,state='hidden' if state['hidden'] else 'normal')
        if target:
            dx,dy=target[0]-group[0],target[1]-group[1]
            for item in state['badge_items']:ed.c.move(item,dx,dy)
            def shift(rect):return tuple(v+(dx if i%2==0 else dy) for i,v in enumerate(rect))
            context['rect']=shift(context['rect'])
            ed.badges={oid:shift(rect) for oid,rect in ed.badges.items()}
            if counter:
                for name in ('rect','current_rect','group_rect'):counter[name]=shift(counter[name])
    ed.clean_label_boxes=[] if state['hidden'] else [context['rect']]+([counter['group_rect']] if counter else [])
    contact=clean.tether_points(ed,ed.drag['quad'],context['rect'])
    if contact:
        start,end=contact;x,y=end
        for item in ed.c.find_withtag('badge_tether'):
            tags=ed.c.gettags(item)
            if context['oid'] not in tags:continue
            ed.c.itemconfigure(item,state='hidden' if state['hidden'] else 'normal')
            if 'badge_tether_line' in tags:ed.c.coords(item,*start,*end)
            elif 'badge_tether_connector' in tags:ed.c.coords(item,x-3,y-3,x+3,y+3)


def paint(ed,*,publish=True):
    """Move the visible handles in the Motion handler, before the next idle."""
    if not hasattr(ed,'navigator') or not ed.drag:return False
    from . import mobile_canvas_clean_view as clean,mobile_canvas_guides as guides
    scene=ed._read()
    if not scene:return False
    key=(ed.cache_key,ed.c.winfo_width(),ed.c.winfo_height(),ed.to_canvas((0,0)),ed.to_canvas(scene['image_size']))
    state=ed.drag.get('_canvas_preview')
    if not state or state['key']!=key or not ed.c.type(state['shape']):
        state=_prepare(ed,key)
        if state is None:return False
        ed.drag['_canvas_preview']=state
    points=[ed.to_canvas(p) for p in ed.drag['quad']];coords=[v for p in points for v in p]
    for item in (state['shape'],state['underlay']):ed.c.coords(item,*coords)
    for item,(x,y) in zip(state['handles'],points):ed.c.coords(item,x-5,y-5,x+5,y+5)
    box=clean.bounds(ed,ed.drag['quad']);ed.clean_geometry_boxes=[box]
    _badge(ed,state,box)
    ed.c.tag_raise('badge_tether_connector')
    guides.update(ed)
    if publish and state['published_quad']!=ed.drag['quad']:
        ed._render_drag_geometry(scene)
        state['published_quad']=copy.deepcopy(ed.drag['quad'])
    return True
