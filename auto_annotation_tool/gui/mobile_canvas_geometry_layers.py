"""Transient ◀/▶ view of one source MT; independent of the FINAL stack."""
from __future__ import annotations

def model(ed):
    from . import mobile_canvas_clean_view as clean
    from ..ranking import mobile_mt_geometry_layers as data
    row=clean.selected(ed)
    if not row or not row['candidate'] or row['role']!='FINAL':return None
    key=(ed.cache_key,row['oid'])
    if getattr(ed,'_geometry_model_key',None)!=key:
        ed._geometry_model_key=key
        try:
            ed._geometry_model=data.snapshot(ed.h.session,ed.h.current_invocation,row['candidate']['object_id'],scene=ed._read(),state=clean.source(ed))
            from .mobile_canvas_bbox_marker import enrich
            ed._geometry_model=enrich(ed.h.session,ed.h.current_invocation,ed._read(),row['candidate'],ed._geometry_model)
        except (ValueError,KeyError,TypeError,OSError):ed._geometry_model=None
    value=ed._geometry_model
    if not value or not value['layers']:return None
    value=dict(value,bad=row['fp'])
    if row.get('seed'):
        value=dict(value,layers=[*value['layers'],dict(kind='GTR',role='GTR startowy (niezapisany szkic)',
            quad=row['gt'],plate_id=None,provisional=True)],gtr_seed=True)
    identity=(id(ed.h.session),tuple(value['source_key']),row['oid'])
    selection=getattr(ed,'_geometry_layer_state',None)
    kinds=[l['kind'] for l in value['layers']]
    # Keep the familiar editable GT on selection; BBOX is first in the cycle.
    default='GTR' if 'GTR' in kinds else 'POSE' if 'POSE' in kinds else kinds[0]
    kind=selection[1] if selection and selection[0]==identity and selection[1] in kinds else default
    return dict(value,identity=identity,active=kind,index=kinds.index(kind),role=next(l['role'] for l in value['layers'] if l['kind']==kind))

def editable(ed):
    value=model(ed)
    return not value or value['active']=='GTR'

def contains(ed,x,y):
    box=getattr(ed,'geometry_layer_badge',None)
    return bool(box and box[0]<=x<=box[2] and box[1]<=y<=box[3])

def click(ed,event):
    if not contains(ed,event.x,event.y):return False
    if ed.entry is not None or ed.drag or ed.points or ed.mode=='draw' or ed.gt_focus_id:return True
    value=model(ed)
    if value and len(value['layers'])>1:
        direction=next((back for (l,t,r,b),back in ed.geometry_layer_controls if l<=event.x<=r and t<=event.y<=b),None)
        if direction is not None:
            index=(value['index']+(-1 if direction else 1))%len(value['layers'])
            ed._geometry_layer_state=(value['identity'],value['layers'][index]['kind'])
            ed.c.focus_set();ed.sync();ed.request_overlay()
            ed.message(value['layers'][index]['role']+' · przegląd warstw; ▲/▼ zmienia detekcję MT')
    return True

def bbox_quad(value):
    layer=next((l for l in (value or {}).get('layers',[]) if l['kind']=='BBOX'),None)
    if not layer:return None
    l,t,r,b=layer['bbox'];return [(l,t),(r,t),(r,b),(l,b)]

def paint_geometry(ed,value):
    if not value or ed.gt_focus_id or ed.mode=='draw':return
    if hasattr(ed,'navigator'):
        return paint_navigator_geometry(ed)
    from .mobile_canvas_editor import TAG
    q=bbox_quad(value)
    if q and value.get('bbox_marker'):
        from .mobile_canvas_bbox_marker import paint
        paint(ed,value['bbox_marker'],color='#EF5262' if value['bad'] else '#58B8D5',
              tags=(TAG,'mt_source_bbox_layer',value['object_id']))
        q=None
    if q:
        x,y=ed.to_canvas(q[0]);r,b=ed.to_canvas(q[2]);active=value['active']=='BBOX'
        existing=[i for i in ed.c.find_withtag('model_location_bbox') if value['object_id'] in ed.c.gettags(i)]
        style=dict(outline='#EF5262' if value['bad'] else '#58B8D5',width=2.5 if active else 1,dash=() if active else (4,4))
        if existing:
            ed.c.itemconfigure(existing[0],**style);ed.c.addtag_withtag('mt_source_bbox_layer',existing[0])
        else:ed.c.create_rectangle(x,y,r,b,**style,tags=(TAG,'mt_source_bbox_layer',value['object_id']))
    for kind,tag in (('POSE','mt_original'),('GTR','operator_quad')):
        for item in ed.c.find_withtag(tag):
            if value['object_id'] in ed.c.gettags(item):
                ed.c.itemconfigure(item,width=2.5 if value['active']==kind else 1)
                if value['active']==kind:ed.c.tag_raise(item)
    ed.c.tag_raise('corner_handle')
    for tag in ('clean_label','mt_source_badge','mt_stack_control'):ed.c.tag_raise(tag)


def paint_reference_identities(ed):
    """Name saved GTRs beside their own contours, also with a source selected."""
    from . import mobile_canvas_clean_view as clean
    from .mobile_canvas_visibility import contours
    from .mobile_canvas_editor import TAG
    from .mobile_canvas_approval_visuals import GREEN,PENDING,DARK
    import tkinter.font as tkfont
    if not hasattr(ed,'navigator') or ed.mode=='draw' or ed.shift_preview.active:return
    if not hasattr(ed,'_reference_identity_font'):
        ed._reference_identity_font=tkfont.Font(root=ed.c,font=('Segoe UI',8,'bold'))
    font=ed._reference_identity_font;rows=clean.records(ed);value=model(ed)
    active_kind='GTR' if ed.gt_focus_id or not value else value['active']
    for frame in contours(ed):
        if frame['kind']!='GTR' or (frame['oid']==ed.selected and active_kind=='GTR'):continue
        row=next((r for r in rows if r['oid']==frame['oid'] and r.get('seed')),None) if frame.get('seed') else next((r for r in rows if r['pid']==frame['plate_id']),None)
        if not row:continue
        text=(f"Szkic GTR · MT {row['candidate']['mt_detection_index']+1} · niezapisany" if frame.get('seed') else f"GTR {row['n']} · operator")
        w=font.measure(text)+12;h=font.metrics('linespace')+8
        rect=clean.badge_position(ed,clean.bounds(ed,frame['quad']),w,h)
        if rect is None:continue
        color=GREEN if frame['approved'] else PENDING;tags=(TAG,'reference_identity_label',frame['oid'])
        clean.badge_tether(ed,frame['quad'],rect,frame['oid'],tag='reference_identity_tether')
        ed.c.create_rectangle(*rect,fill=DARK,outline=color,tags=tags)
        ed.c.create_text(rect[0]+6,(rect[1]+rect[3])/2,text=text,anchor='w',font=font,fill=color,tags=tags+('reference_identity_text',))
        ed.clean_label_boxes.append(rect);ed.reference_identity_hits[frame['oid']]=rect

def paint_badge(ed,value):
    if hasattr(ed,'navigator'):return paint_navigator_badge(ed,value)
    if not value or ed.gt_focus_id or ed.mode=='draw':return
    import tkinter.font as tkfont
    from .mobile_canvas_clean_view import bounds,badge_position,badge_tether,intersects
    from .mobile_canvas_editor import TAG
    anchor=bbox_quad(value) or next((l['quad'] for l in value['layers'] if l.get('quad')),None)
    if not anchor:return
    if not hasattr(ed,'_mt_badge_font'):ed._mt_badge_font=tkfont.Font(root=ed.c,font=('Segoe UI',8,'bold'))
    font=ed._mt_badge_font;label=f"{value['active']} {value['index']+1}/{len(value['layers'])} · GTR {value['gtr_count']}"
    h=max(23,font.metrics('linespace')+6);button=h if len(value['layers'])>1 else 0;w=font.measure(label)+14+2*button
    box=bounds(ed,anchor);rect=badge_position(ed,box,w,h)
    if rect is None:
        cw,ch=ed.c.winfo_width(),ed.c.winfo_height()
        for x,y in ((2,2),(cw-w-2,2),(2,ch-h-2),(cw-w-2,ch-h-2)):
            proposed=(x,y,x+w,y+h)
            if x>=2 and y>=2 and not any(intersects(proposed,b) for b in ed.clean_geometry_boxes+ed.clean_label_boxes):rect=proposed;break
    if rect is None:return
    ed.geometry_layer_badge=rect;ed.clean_label_boxes.append(rect);x,y,r,b=rect
    tags=(TAG,'geometry_layer_badge',value['object_id']);badge_tether(ed,anchor,rect,value['object_id'],tag='geometry_layer_tether')
    ed.c.create_rectangle(*rect,fill='#14252D',outline='#58B8D5',tags=tags)
    ed.c.create_text(x+button+7,(y+b)/2,text=label,anchor='w',font=font,fill='#D9F1FA',tags=tags+('geometry_layer_text',))
    if button:
        for l,glyph,back in ((x,'◀',True),(r-button,'▶',False)):
            cell=(l,y,l+button,b);ed.c.create_rectangle(*cell,fill='#145A7A',outline='#58B8D5',tags=tags)
            ed.c.create_text(l+button/2,(y+b)/2,text=glyph,font=font,fill='white',tags=tags)
            ed.geometry_layer_controls.append((cell,back))

def paint_navigator_geometry(ed):
    from .mobile_canvas_visibility import contours
    from .mobile_canvas_approval_visuals import GREEN,PENDING,RED,POSE,BBOX,polygon
    from .mobile_canvas_editor import TAG
    if ed.mode=='draw':return
    for tag in ('mt_original','operator_quad','model_location_bbox','mt_source_bbox_layer','fp_diagonal','corner_handle','navigator_contrast_underlay','clean_contrast_underlay','bbox_zero_marker','active_frame_accent'):ed.c.delete(tag)
    if ed.shift_preview.active:return
    for item in sorted(contours(ed),key=lambda item:item['kind']=='GTR'):
        oid,kind,q=item['oid'],item['kind'],item['quad'];active=oid==ed.selected
        color=(GREEN if item['approved'] else PENDING) if kind=='GTR' else RED if item['bad'] else BBOX if kind=='BBOX' else POSE
        tag={'GTR':'operator_quad','POSE':'mt_original','BBOX':'model_location_bbox'}[kind]
        from .mobile_canvas_selection import accent
        accent(ed,item,{'GTR':3,'POSE':2.5,'BBOX':2}[kind])
        if item.get('marker'):
            from .mobile_canvas_bbox_marker import paint as paint_marker,verdict
            from .mobile_canvas_clean_view import source
            candidate=next(c for c in source(ed)['candidates'] if c['object_id']==oid)
            if verdict(ed,candidate)=='OFF_PLATES_BAD':color=RED
            paint_marker(ed,item['marker'],color=color,tags=(TAG,tag,'mt_source_bbox_layer','navigator_geometry',kind,oid))
            continue
        coords=[v for p in q for v in ed.to_canvas(p)]
        polygon(ed,coords,color=color,width={'GTR':3,'POSE':2.5,'BBOX':2}[kind],
                dash=(4,3) if kind=='BBOX' else (2,2) if item.get('seed') else (),
                tags=(TAG,tag,'navigator_geometry',kind,oid,item['plate_id'] or 'unlinked',
                      'operator_seed' if item.get('seed') else 'saved_or_source'))
        if kind=='BBOX':ed.c.addtag_withtag('mt_source_bbox_layer',ed.c.find_withtag(tag)[-1])
        if kind!='GTR' and item['bad']:
            for a,b in ((0,2),(1,3)):ed.c.create_line(*ed.to_canvas(q[a]),*ed.to_canvas(q[b]),fill=RED,width=2,tags=(TAG,'fp_diagonal',oid))
        if active and kind=='GTR' and ed.mode=='edit' and (ed.gt_focus_id or editable(ed)):
            for p in q:
                x,y=ed.to_canvas(p);ed.c.create_oval(x-5,y-5,x+5,y+5,fill='white',outline=color,tags=(TAG,'corner_handle',oid))
    value=model(ed);kind='GTR' if ed.gt_focus_id or not value else value['active']
    for item in ed.c.find_withtag('navigator_geometry'):
        tags=ed.c.gettags(item)
        if ed.selected in tags and kind in tags:ed.c.tag_raise(item)
    ed.c.tag_raise('corner_handle')
    paint_reference_identities(ed)

def paint_navigator_badge(ed,value):
    from .mobile_canvas_info_badge import paint
    return paint(ed,value)
