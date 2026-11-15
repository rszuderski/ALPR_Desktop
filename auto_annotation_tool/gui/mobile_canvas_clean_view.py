"""Read-only operational Canvas projection over the existing scientific records."""
from __future__ import annotations
import copy,math,tkinter as tk
from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as data
from ..ranking import mobile_canvas_partial_review as partial
from ..ranking.mobile_canvas_operator_view import reference_numbers

def source(ed):
    from .mobile_canvas_stacks import view
    from ..ranking.mobile_canvas_operator_view import display_snapshot
    snapshot=view(ed)
    if not snapshot:return None
    from .mobile_canvas_spatial_reference import view as spatial_view
    value=spatial_view(ed,ed._read(),display_snapshot(ed._read(),snapshot))
    from .mobile_canvas_visibility import gt_only
    if gt_only(ed):value=dict(value,candidates=[])
    return value

def show_fp(ed):return bool(getattr(ed,'show_fp_var',None) and ed.show_fp_var.get())

def candidates(v,pid):
    return [c for c in v['candidates'] if c['role']=='FINAL' and c['plate_id']==pid and c['association_status']=='ASSOCIATED_FOR_GEOMETRY']

def records(ed):
    s=ed._read();v=source(ed)
    if not s or not v:return []
    result=[];conflicted={c['plate_id'] for c in v['conflicts']}
    numbers=reference_numbers(v)
    for pid,p in v['plates'].items():
        if pid not in v.get('active_plate_ids', v['plates']):continue
        members=[c for c in candidates(v,pid) if show_fp(ed) or not c.get('operator_bad')]
        active=next((c for c in members if c['object_id']==ed.selected),members[0] if members else None)
        if active is None and not p['gt_quad']:
            # Retain the saved ID/GTZ and explicit removal state without
            # pretending the missing corners form a spatial GTR relation.
            active=next((c for c in v['candidates'] if c['role']=='FINAL' and not c['plate_id']
                and c.get('canonical_link',{}).get('plate_id')==pid and (show_fp(ed) or not c.get('operator_bad'))),None)
        ref=next((x['reference_object_id'] for x in s.get('physical_gt_v1',{}).get('plates',[]) if x['gt_id']==pid),p['origin_object_id'])
        if active is None:
            active=next((c for c in v['candidates'] if c['object_id']==ref and c['role']=='UNKNOWN'),None)
        oid=active['object_id'] if active else ref
        result.append(dict(oid=oid,pid=pid,n=numbers.get(pid),reference_id=ref,candidate=active,plate=p,fp=bool(active and active.get('operator_bad')),role=active['role'] if active else 'GT',
            mt=copy.deepcopy(active['mt_quad']) if active else None,gt=copy.deepcopy(p['gt_quad']) if pid not in conflicted else None,
            members=members,conflict=pid in conflicted))
    for c in v['candidates']:
        if c['role']=='SUPPRESSED' or (c['role']=='FINAL' and c['plate_id']):continue
        if any(r['candidate'] and r['candidate']['object_id']==c['object_id'] for r in result):continue
        fp=c['role']=='FINAL' and (c.get('operator_bad') or c['association_status']=='UNASSOCIATED_NO_PLATE')
        if fp and not show_fp(ed):continue
        result.append(dict(oid=c['object_id'],pid=None,n=None,candidate=c,plate=None,fp=fp,role=c['role'],
                           mt=copy.deepcopy(c['mt_quad']),gt=None,members=[],conflict=False))
    decisions=partial.progress(s)['object_decisions']
    from .mobile_canvas_reference_seed import for_editor
    seeds=for_editor(ed,s,v)
    for row in result:
        row['seed']=seeds.get(row['oid']) if not row['plate'] else None
        if row['seed']:row['gt']=copy.deepcopy(row['seed']['quad'])
        row['decision']=(partial.NOT_PLATE if row['fp'] else decisions.get(row['oid'],partial.PLATE)
                         if row['role'] in ('GT','FINAL') else partial.UNRESOLVED)
    return result

def visible_ids(ed):return [r['oid'] for r in records(ed)]

def eligible(ed,oid):
    if not oid:return True
    v=source(ed)
    if not v:return False
    c=next((c for c in v['candidates'] if c['object_id']==oid),None)
    if c:return c['role']!='SUPPRESSED' and (show_fp(ed) or not(c.get('operator_bad') or c['association_status']=='UNASSOCIATED_NO_PLATE'))
    return oid in visible_ids(ed)

def prune(ed):
    if getattr(ed,'gt_focus_id','') and ed.gt_focus_id!=ed.selected:ed.gt_focus_id=''
    if ed.selected and not eligible(ed,ed.selected):
        alias=next((r for r in records(ed) if r.get('reference_id')==ed.selected and r['candidate']),None)
        if alias:
            if ed.gt_focus_id==ed.selected:ed.gt_focus_id=alias['oid']
            ed.selected=alias['oid'];return
        ed.selected=ed.hover='';ed.mode='select';ed.gt_focus_id='';ed._s_hover_hit=None

def toggle_fp(ed):
    ed.show_fp_var.set(not ed.show_fp_var.get());prune(ed);ed._s_hover_hit=None
    ed.sync();ed.request_overlay()

def refresh_visibility(ed):
    prune(ed);ed._s_hover_hit=None;ed.sync();ed.request_overlay()

def refresh_contours(ed):
    # A view toggle must not prune selection, rebuild metrics or finish GTZ.
    ed._s_hover_hit=None
    if getattr(ed,'manual_gt_first',False):
        ed._navigator_initialized=None
        ed.sync()
    ed.request_overlay()

def selected(ed):
    items=records(ed);rows=[r for r in items if r['oid']==ed.selected]
    from .mobile_canvas_reservation import focused_row
    reserved=focused_row(ed,items)
    if reserved:
        if reserved['oid']==ed.selected and reserved['candidate'] and not reserved['fp']:return reserved
        return dict(reserved,oid=ed.selected,candidate=None,role='GT',mt=None,fp=False)
    # A retained, currently inactive GTR can share its historical origin OID
    # with a source MT now marked FP. Selection still targets that source MT.
    return next((r for r in rows if r['candidate']),rows[0] if rows else None)

def location_only(row):
    return bool(row and row['role']=='FINAL' and row['candidate'] and not row['mt'] and not row['gt'])

def source_location(ed,r):
    """Verified source location, cached only for this source/review revision."""
    if not r or not r['candidate'] or r['role']!='FINAL':return None
    try:stat=ed.h.session.path.stat()
    except OSError:return None
    key=(ed.cache_key,r['oid'],stat.st_size,stat.st_mtime_ns)
    cache=getattr(ed,'_gtr_source_locations',{})
    if key not in cache:
        from ..ranking.mobile_canvas_resolution import model_location
        try:value=model_location(ed.h.session,ed.h.current_invocation,ed._read(),core._object(ed._read(),r['oid']))
        except (ValueError,KeyError,TypeError,OSError):value=None
        cache={key:value};ed._gtr_source_locations=cache
    return cache[key]

def drawing_context(ed):
    target=getattr(ed,'draw_reference_target',None)
    if not target or target[0]!=ed.h.current_invocation:return None
    row=next((r for r in records(ed) if r['oid']==target[1] and r['pid']==target[2]),None)
    location=source_location(ed,row)
    if not location or location['kind']!='BBOX':return None
    visible=[pid for pid,p in source(ed)['plates'].items() if p['gt_quad'] or pid==target[2]]
    return dict(row=row,location=location,number=visible.index(target[2])+1)

def bar_caption(ed,r=None):
    s=ed._read()
    if s and s.get('scope')=='SCENE_FRAME':
        obj=next((o for o in s['objects'] if o['object_id']==ed.selected),None)
        if obj and obj['quad'] is None:return 'GTR · obecność potwierdzona · narożniki niedostępne'
    r=r or selected(ed)
    if not r:return ''
    if location_only(r):return f"MT {r['candidate']['mt_detection_index']+1} · brak narożników"
    if getattr(ed,'gt_focus_id',''):
        return f"Korekta GTR {r['n']}" if r['n'] else 'Korekta nowej GTR'
    if r['pid'] and r['n'] is not None:
        location=source_location(ed,r)
        label=(f"MT {r['candidate']['mt_detection_index']+1} ({location['kind']}) ↔ GTR {r['n']}" if location else
               f"GTR {r['n']} · "+('brak wiarygodnej lokalizacji' if r['candidate'] else 'MT: niepowiązany'))
        if len(r['members'])>1:label+=f" · MT {r['members'].index(r['candidate'])+1}/{len(r['members'])}"
        return label
    return f"MT {r['candidate']['mt_detection_index']+1}"+(' · błędne wskazanie' if r['fp'] else '')

def intersects(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]

def bounds(ed,q):
    points=[ed.to_canvas(p) for p in q]
    # Leave the entire 14 px handle hit area clear, not only its painted oval.
    return (min(x for x,y in points)-16,min(y for x,y in points)-16,max(x for x,y in points)+16,max(y for x,y in points)+16)

def badge_position(ed,box,w,h):
    """Prefer a short vertical tether above the frame, keeping GT fields clear."""
    cw,ch=ed.c.winfo_width(),ed.c.winfo_height()
    x=max(2,min((box[0]+box[2]-w)/2,cw-w-2))
    y=max(2,min(box[1],ch-h-2))
    proposals=[(x,box[1]-h-6),(max(2,box[0]),box[1]-h-6),
               (min(cw-w-2,box[2]-w),box[1]-h-6),
               (x,box[3]+6),(box[2]+6,y),(box[0]-w-6,y)]
    # A source BBOX may enclose both GT and Pose. Keep the editable GT badge
    # above/below that enclosing contour (or another label), not in the toolbar
    # merely because the first six positions are occupied.
    obstacles=ed.clean_geometry_boxes+ed.clean_label_boxes
    extra=[(x,b[1]-h-6) for b in obstacles]+[(x,b[3]+6) for b in obstacles]
    proposals+=sorted(extra,key=lambda p:abs(p[1]-(box[1]-h-6)))
    for x,y in proposals:
        rect=(x,y,x+w,y+h)
        if x<2 or y<2 or x+w>cw-2 or y+h>ch-2:continue
        if not any(intersects(rect,b) for b in ed.clean_geometry_boxes+ed.clean_label_boxes):return rect
    return None

def tether_points(ed,anchor,rect):
    """Nearest contact on the actual polygon, including a sloping edge."""
    points=[ed.to_canvas(p) for p in anchor];cx=(rect[0]+rect[2])/2;cy=(rect[1]+rect[3])/2
    nearest=[]
    for a,b in zip(points,points[1:]+points[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
        u=max(0,min(1,((cx-a[0])*dx+(cy-a[1])*dy)/den)) if den else 0
        p=(a[0]+u*dx,a[1]+u*dy);nearest.append((math.dist((cx,cy),p),p))
    if not nearest:return None
    end=min(nearest)[1];start=(max(rect[0],min(end[0],rect[2])),max(rect[1],min(end[1],rect[3])))
    return start,end


def badge_tether(ed,anchor,rect,oid,*,tag='badge_tether'):
    """Attach to the nearest edge and mark the exact frame contact."""
    from .mobile_canvas_editor import TAG
    points=tether_points(ed,anchor,rect)
    if points is None:return
    start,end=points;tags=(TAG,tag,oid)
    ed.c.create_line(*start,*end,fill='#ADBCC5',width=1,tags=tags+('badge_tether_line',))
    x,y=end;ed.c.create_oval(x-3,y-3,x+3,y+3,fill='#ADBCC5',outline='#080808',width=1.3,
                           tags=tags+('badge_tether_connector',))

def paint_badge(ed,r,anchor,text_editable,number):
    from .mobile_canvas_editor import TAG
    c=ed.c;oid=r['oid'];bbox=location_only(r)
    label=(f"MT {r['candidate']['mt_detection_index']+1} · BBOX" if bbox else
           f"GTR {r['n']}" if r['pid'] and r['n'] is not None else f"MT {r['candidate']['mt_detection_index']+1}")
    if bbox:text_editable=False;number=''
    reference=bool(r['pid'] and r['n'] is not None)
    label+=' · '+('Błędna' if r['fp'] else 'OK' if reference or r['decision']==partial.PLATE else 'decyzja oczekuje')
    if not r['candidate'] and r['pid'] in getattr(ed,'_gtr_overlap_notes',{}):label+=' · sprawdź nałożenie'
    color='#F17987' if r['fp'] else '#61E6A2' if reference or r['decision']==partial.PLATE else '#EDF5F7'
    limit=max(80,min(320,c.winfo_width()-12));header_h=23;gt_h=29 if text_editable or number else 0
    gt_label='GTZ: '+(number or 'wpisz numer')
    while ed._badge_font.measure(gt_label)+12>limit:gt_label=gt_label[:-2]+'…'
    w=min(limit,max(ed._badge_font.measure(label)+12,ed._badge_font.measure(gt_label)+12 if gt_h else 0,150 if text_editable else 0))
    box=bounds(ed,anchor);rect=badge_position(ed,box,w,header_h+gt_h)
    if rect is None:
        # Keep the compact decision/count visible even when zoom fills the
        # viewport. GT typing still uses the existing safe toolbar fallback.
        gt_h=0;w=min(limit,ed._badge_font.measure(label)+12);rect=badge_position(ed,box,w,header_h)
    if rect is None:
        x=max(2,min((box[0]+box[2]-w)/2,c.winfo_width()-w-2))
        y=max(2,min(box[1]-header_h-6,c.winfo_height()-header_h-2));rect=(x,y,x+w,y+header_h)
    else:ed.clean_label_boxes.append(rect)
    x,y,rr,b=rect;header=(x,y,rr,y+header_h)
    ed.clean_counter_boxes[oid]=header;badge_tether(ed,anchor,rect,oid)
    tags=(TAG,'clean_label','stack_badge',oid)
    c.create_rectangle(*header,fill='#18352C' if r['decision']==partial.PLATE else '#3B222B' if r['fp'] else '#14252D',outline=color,tags=tags)
    c.create_text(x+6,y+header_h/2,text=label,anchor='w',fill=color,font=ed._badge_font,tags=tags+('clean_label_text','stack_badge_text',))
    if gt_h:
        gt_rect=(x,y+header_h,rr,b);gt_tags=(TAG,'clean_label','gt_badge',oid)
        c.create_rectangle(*gt_rect,fill='#302618',outline='#FFD400',tags=gt_tags)
        c.create_text(x+6,y+header_h+gt_h/2,text=gt_label,anchor='w',fill='#FFE2AE',font=ed._badge_font,tags=gt_tags+('clean_label_text',))
        if text_editable:ed.badges[oid]=gt_rect
    ed.stack_badges[oid]=(*rect,r['pid'])

def mt_label(ed,r,*,compact=False):
    from ..ranking.mobile_canvas_operator_view import linked_references
    c=r['candidate'];location=source_location(ed,r)
    kind=location['kind'] if location else ('POSE' if c['mt_quad'] else 'BBOX' if c['source_pose_status']=='INVALID_QUAD' else 'źródło —')
    decision=('FP · Błędna' if r['decision']==partial.NOT_PLATE else 'OK' if r['decision']==partial.PLATE else 'decyzja oczekuje')
    relation=linked_references(ed._read(),source(ed),c)
    count='—' if relation['count'] is None else str(relation['count'])
    label=f"MT {c['mt_detection_index']+1} · {kind}"
    if c['role']!='FINAL':label+=' · FINAL —'
    if not compact and kind=='BBOX':label+=' · brak Pose'
    return label+('' if compact else ' · '+decision)+f' · GTR {count}'

def paint_mt_badge(ed,r,anchor,*,preview=False):
    """Small source-owned label, never an unassigned GTZ input."""
    from .mobile_canvas_editor import TAG
    c=ed.c;label=mt_label(ed,r,compact=preview);font=('Segoe UI',8,'bold')
    import tkinter.font as tkfont
    if not hasattr(ed,'_mt_badge_font'):ed._mt_badge_font=tkfont.Font(root=c,font=font)
    w=min(max(30,c.winfo_width()-8),ed._mt_badge_font.measure(label)+10);h=21
    box=bounds(ed,anchor);rect=badge_position(ed,box,w,h)
    if rect is None:
        # Never cover the source digits. The selected context/bar retains text
        # if extreme zoom leaves no safe label space.
        return
    ed.clean_label_boxes.append(rect);x,y,rr,b=rect
    oid=r['oid'];tags=(TAG,'shift_source_preview' if preview else 'mt_relation_badge','mt_source_badge',oid,r['candidate']['attempt_id'])
    if not preview:ed.mt_source_badge_hits[oid]=rect
    badge_tether(ed,anchor,rect,oid,tag='mt_source_tether')
    fp=r['decision']==partial.NOT_PLATE and not preview
    c.create_rectangle(*rect,fill='#3B222B' if fp else '#14252D',outline='#F17987' if fp else '#58B8D5',tags=tags)
    c.create_text(x+5,(y+b)/2,text=label,anchor='w',fill='#F17987' if fp else '#D9F1FA',font=ed._mt_badge_font,
                  tags=tags+('mt_relation_text','clean_label_text')+(('stack_badge_text',) if not preview and not r['pid'] else ()))
    if not preview and not r['pid']:
        ed.clean_counter_boxes[oid]=rect;ed.stack_badges[oid]=(*rect,None)

def entry_matches(ed,r):
    if ed.entry is None or not ed.entry_target or ed.entry_target[0]!=ed.h.current_invocation:return False
    p=r['plate'] or {}
    return ed.entry_target[1] in (r['oid'],r.get('reference_id'),p.get('primary_object_id'),p.get('origin_object_id'))

def text_editable(ed,oid):
    """Cache display permission only; entering/saving still revalidates source."""
    key=(ed.cache_key,id(getattr(ed.h.session,'review',None)),ed.h._image_entry,
         tuple(ed.h._image_source_size or ()),id(ed.h._image))
    if getattr(ed,'_clean_text_policy_key',None)!=key:
        ed._clean_text_policy_key=key;ed._clean_text_policies={}
    if oid not in ed._clean_text_policies:
        ed._clean_text_policies[oid]=ed._gt_policy(oid)['gt_editable']
    return ed._clean_text_policies[oid]

def place_entry(ed):
    """Keep the existing GT editor next to its physical reference, also in stacks."""
    if ed.entry is None:return
    if ed.mode=='draw':ed.entry.place_forget();return
    r=next((r for r in records(ed) if entry_matches(ed,r)),None)
    box=ed.badges.get(r['oid']) if r else None
    if box:
        l,t,rr,b=box
        if getattr(ed,'_gtz_entry_right',None) is not None:rr=min(rr,ed._gtz_entry_right)
        prefix=getattr(ed,'_navigator_badge_prefix',ed._badge_font.measure('GTZ: ')+6)
        ed.entry.place(in_=ed.c,x=round(l+prefix),y=round(t+1),relwidth=0,relheight=0,width=round(rr-l-prefix-2),height=round(b-t-2))
    else:
        # A viewport completely filled by geometry has no safe nearby space.
        ed.clean_gt_slot.grid();ed.clean_gt_slot.lift();ed.entry.place(in_=ed.clean_gt_slot,x=0,y=0,width=0,relwidth=1,relheight=0,height=27)
    ed.entry.lift()

def paint(ed):
    from .mobile_canvas_editor import TAG
    from . import mobile_canvas_guides as guides
    s=ed._read();c=ed.c
    from .mobile_canvas_info_badge import update_hud
    update_hud(ed)
    ed.badges={};ed.card_positions={};ed.card_widths={};ed.card_heights={};ed.card_header_heights={}
    ed.stack_badges={};ed.stack_hit=[];ed.stack_bbox_hit=[];ed.clean_label_boxes=[];ed.clean_geometry_boxes=[];ed.clean_counter_boxes={};ed.mt_source_badge_hits={}
    ed.mt_stack_badges={};ed.mt_stack_controls=[]
    ed.geometry_layer_badge=None;ed.geometry_layer_controls=[]
    ed._badge4_rows=[];ed._badge4_rect=None
    ed._context_badge=None
    ed._stack_counter=None
    ed._navigator_badge_prefix=0
    ed.reference_identity_hits={}
    if (not s or ed.h._view_center is None or ed.h._image_entry!=s['image_entry']
            or tuple(ed.h._image_source_size or ())!=tuple(s['image_size'])):
        guides.clear(ed);return True
    prune(ed);items=records(ed);focus=getattr(ed,'gt_focus_id','')
    from .mobile_canvas_approval_visuals import paint_image_border,approved_ids,GREEN,PENDING,polygon,RED,POSE,BBOX
    approved=approved_ids(s,source(ed))
    paint_image_border(ed,s)
    from . import mobile_canvas_geometry_layers as geometry_layers
    from ..ranking.mobile_mt_geometry_layers import overlapping_references
    ed._gtr_overlap_notes=overlapping_references(s,source(ed))
    layer_view=geometry_layers.model(ed)
    for frame in ed.collision_warning.frames():ed.clean_geometry_boxes.append(bounds(ed,frame['quad']))
    if focus:items=[r for r in items if r['oid']==focus]
    def poly(q,color,tag,oid,width=1):
        if q:polygon(ed,[v for p in q for v in ed.to_canvas(p)],color=color,width=width,tags=(TAG,tag,oid),underlay='clean_contrast_underlay')
    for r in items:
        if r['gt']:ed.clean_geometry_boxes.append(bounds(ed,r['gt']))
        if r['mt']:ed.clean_geometry_boxes.append(bounds(ed,r['mt']))
    if not focus and geometry_layers.bbox_quad(layer_view):ed.clean_geometry_boxes.append(bounds(ed,geometry_layers.bbox_quad(layer_view)))
    if ed.mode=='draw':
        context=drawing_context(ed)
        for p in ed.points:
            x,y=ed.to_canvas(p);c.create_oval(x-5,y-5,x+5,y+5,fill='#FFD400',tags=(TAG,'drawing_corner'))
        caption=(f"GTR {context['number']} ← MT {context['row']['candidate']['mt_detection_index']+1} (BBOX)" if context else 'Nowa GTR')
        target=getattr(ed,'draw_target',None)
        if target and target['kind']=='replace':caption=f"Poprawiasz GTR {target['number']}"
        elif target and target['kind']=='create':caption='Nowa GTR dla wybranego MT'
        ed.clean_caption.configure(text=caption+f' · narożnik {min(4,len(ed.points)+1)}/4')
        guides.update_draft(ed)
        if hasattr(ed,'navigator'):ed.navigator.update()
        return True
    if ed.show_roi_overlay and not focus:
        l,t,r,b=s['roi'];c.create_rectangle(*ed.to_canvas((l,t)),*ed.to_canvas((r,b)),outline='#6E7880',tags=(TAG,'analyzed_roi'))
    for r in items:
        oid=r['oid'];active=oid==ed.selected and (r['candidate'] is not None or not any(x['oid']==oid and x['candidate'] for x in items))
        obj=next((o for o in s['objects'] if o['object_id']==oid),None)
        q=copy.deepcopy(ed.drag['quad']) if ed.drag and ed.drag['oid']==oid else r['gt']
        if focus and q is None:q=ed._current_quad(obj) if obj else None
        if not focus:
            poly(r['mt'],RED if r['fp'] else GREEN if r['decision']==partial.PLATE
                 else POSE if r['role']=='FINAL' else PENDING,'mt_original',oid,2 if r['fp'] else 1.4 if active else 1)
            if r['fp'] and r['mt']:
                for a,b in ((0,2),(1,3)):
                    c.create_line(*ed.to_canvas(r['mt'][a]),*ed.to_canvas(r['mt'][b]),fill=RED,width=2,tags=(TAG,'fp_diagonal',oid))
        gtr_color=GREEN if r['pid'] in approved else PENDING
        poly(q,gtr_color,'operator_quad',oid,1.6 if active else 1)
        editable=not r['fp'] and not r['conflict'] and r['role'] in ('GT','FINAL')
        handle=q or (r['mt'] if editable else None)
        if active and ed.mode=='edit' and editable and handle and (focus or geometry_layers.editable(ed)):
            for x,y in [ed.to_canvas(p) for p in handle]:c.create_oval(x-5,y-5,x+5,y+5,fill='white',outline=gtr_color,tags=(TAG,'corner_handle',oid))
        anchor=q or r['mt']
        if not focus and r['candidate'] and not r['mt']:
            from ..ranking.mobile_canvas_resolution import model_location
            try:location=model_location(ed.h.session,ed.h.current_invocation,s,obj)
            except (ValueError,KeyError,TypeError):location=None
            if location and location['kind']=='BBOX':
                l,t,rr,b=location['bbox'];polygon(ed,[v for p in ((l,t),(rr,t),(rr,b),(l,b)) for v in ed.to_canvas(p)],color=RED if r['fp'] else BBOX,width=2,dash=(4,3),tags=(TAG,'model_location_bbox',oid),underlay='clean_contrast_underlay')
                ed.stack_bbox_hit.append((oid,l,t,rr,b))
                box_anchor=[(l,t),(rr,t),(rr,b),(l,b)]  # Badge/hit area only; never Pose or GT.
                ed.clean_geometry_boxes.append(bounds(ed,box_anchor))
                anchor=anchor or box_anchor
                if r['fp']:
                    for a,z in (((l,t),(rr,b)),((rr,t),(l,b))):
                        c.create_line(*ed.to_canvas(a),*ed.to_canvas(z),fill='#EF5262',width=2,tags=(TAG,'fp_diagonal',oid))
        if not anchor:continue
        editable_text=bool(r['plate']) and not r['fp'] and not r['conflict'] and r['role'] in ('FINAL','GT') and text_editable(ed,r.get('reference_id') or oid)
        number=ed.entry_var.get() if entry_matches(ed,r) else (r['plate']['gt_text'] if r['plate'] else '')
        if not hasattr(ed,'navigator') and (r['plate'] or r['role'] not in ('FINAL','UNKNOWN')):paint_badge(ed,r,anchor,editable_text,number)
        if not hasattr(ed,'navigator') and r['candidate'] and r['role'] in ('FINAL','UNKNOWN') and not focus:
            mt_anchor=r['mt']
            if not mt_anchor:
                location=source_location(ed,r)
                if location and location['kind']=='BBOX':
                    l,t,rr,b=location['bbox'];mt_anchor=[(l,t),(rr,t),(rr,b),(l,b)]
            if mt_anchor:
                from .mobile_canvas_stacks import paint_candidate_badge
                paint_candidate_badge(ed,r,mt_anchor)
                paint_mt_badge(ed,r,mt_anchor)
    ed.clean_caption.configure(text=bar_caption(ed))
    if hasattr(ed,'navigator'):
        geometry_layers.paint_navigator_geometry(ed)
        if ed.shift_preview.active:ed.shift_preview.paint()
        else:geometry_layers.paint_navigator_badge(ed,layer_view)
        ed.navigator.update()
    elif not focus:
        geometry_layers.paint_geometry(ed,layer_view)
        geometry_layers.paint_badge(ed,layer_view)
        if layer_view:ed.clean_caption.configure(text=bar_caption(ed)+' · '+layer_view['role'])
    r=selected(ed);number=(r['plate']['gt_text'] if r and r['plate'] else '')
    editable=bool(r and r['plate'] and not r['fp'] and text_editable(ed,r.get('reference_id') or r['oid']))
    ed.clean_gt_label.configure(text=('GTZ: '+(number or ('wpisz' if editable else 'nie dotyczy'))) if r and not r['fp'] else '',
                               cursor='hand2' if editable else 'arrow')
    entry_row=next((row for row in items if entry_matches(ed,row)),None)
    entry_fallback=ed.entry is not None and (entry_row is None or entry_row['oid'] not in ed.badges)
    source_layer=layer_view and layer_view['active']!='GTR'
    from .mobile_canvas_visibility import enabled
    if entry_fallback or (editable and not source_layer and enabled(ed,'ref')
            and not getattr(ed,'_badge_window_inactive',False) and r['oid'] not in ed.badges):ed.clean_gt_slot.grid();ed.clean_gt_slot.lift()
    else:ed.clean_gt_slot.grid_remove()
    place_entry(ed)
    guides.update(ed)
    return True

def hit(ed,x,y,*,prefer_stack=False):
    from .mobile_canvas_editor import quad_contains_point
    s=ed._read()
    if (not s or ed.h._view_center is None or ed.h._image_entry!=s['image_entry']
            or tuple(ed.h._image_source_size or ())!=tuple(s['image_size'])):return None
    if not(0<=x<=ed.c.winfo_width() and 0<=y<=ed.c.winfo_height()):return None
    if hasattr(ed,'navigator'):
        from .mobile_canvas_navigator import hit as navigator_hit
        return navigator_hit(ed,x,y,prefer_stack=prefer_stack)
    items=records(ed);focus=getattr(ed,'gt_focus_id','')
    if focus:items=[r for r in items if r['oid']==focus]
    r=next((r for r in items if r['oid']==ed.selected),None)
    from .mobile_canvas_geometry_layers import editable as layer_editable
    if r and not r['fp'] and not r['conflict'] and r['role'] in ('FINAL','GT') and ed.mode=='edit' and (focus or layer_editable(ed)):
        obj=core._object(ed._read(),r['oid'])
        for i,p in enumerate(ed._current_quad(obj) or []):
            if math.dist((x,y),ed.to_canvas(p))<=14:return r['oid'],'corner',i
    for oid,(l,t,rr,b) in list(getattr(ed,'mt_stack_badges',{}).items())+list(getattr(ed,'mt_source_badge_hits',{}).items()):
        if l<=x<=rr and t<=y<=b:return oid,'mz_badge',None
    for oid,(l,t,rr,b,pid) in ed.stack_badges.items():
        if l<=x<=rr and t<=y<=b:
            gt=ed.badges.get(oid)
            return oid,('badge' if gt and gt[1]<=y<=gt[3] else 'mz_badge'),None
    point=ed.to_source(x,y);hits=[]
    for r in items:
        for q in ([r['gt'] or r['mt']] if focus else [r['gt'],r['mt']]):
            if not q:continue
            pts=[ed.to_canvas(p) for p in q];inside=quad_contains_point(q,*point)
            for a,b in zip(pts,pts[1:]+pts[:1]):
                dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy;u=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/den)) if den else 0
                inside=inside or math.dist((x,y),(a[0]+u*dx,a[1]+u*dy))<=8
            if inside:hits.append((r['oid'],'body',None));break
    if not focus:
        for oid,l,t,rr,b in ed.stack_bbox_hit:
            if l<=point[0]<=rr and t<=point[1]<=b:hits.append((oid,'body',None))
    visible_fp={r['oid'] for r in items if r['fp']}
    default=next((h for h in hits if h[0] in visible_fp),hits[0] if hits else None)
    return next((h for h in hits if h[0]==ed.selected),default)

def right_hit(ed,x,y):
    """Resolve the clicked layer; a shared OID never makes GTR into MT."""
    from .mobile_canvas_editor import quad_contains_point
    if not ed.available():return None
    if hasattr(ed,'navigator'):
        from .mobile_canvas_navigator import right_hit as navigator_right_hit
        return navigator_right_hit(ed,x,y)
    items=records(ed);point=ed.to_source(x,y)
    def contains(q):
        if not q:return False
        if quad_contains_point(q,*point):return True
        pts=[ed.to_canvas(p) for p in q]
        for a,b in zip(pts,pts[1:]+pts[:1]):
            dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
            u=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/den)) if den else 0
            if math.dist((x,y),(a[0]+u*dx,a[1]+u*dy))<=8:return True
        return False
    if ed.mode!='draw' and not ed.gt_focus_id:
        badge=list({oid for oid,(l,t,r,b) in list(getattr(ed,'mt_stack_badges',{}).items())+list(getattr(ed,'mt_source_badge_hits',{}).items()) if l<=x<=r and t<=y<=b})
        if len(badge)==1:
            core.require_source_final(ed.h.session,ed.h.current_invocation,badge[0]);return badge[0]
    if any(l<=x<=r and t<=y<=b and pid for l,t,r,b,pid in ed.stack_badges.values()):
        raise ValueError(core.GTR_BAD_MESSAGE)
    if any(contains(r['gt']) for r in items):raise ValueError(core.GTR_BAD_MESSAGE)
    if ed.gt_focus_id or ed.mode=='draw':return None
    hits=set()
    for r in items:
        if not r['candidate'] or r['role']!='FINAL':continue
        q=r['mt']
        if not q:
            # BBOX is a hit area only: it never supplies Pose or GT corners.
            box=next((b[1:] for b in ed.stack_bbox_hit if b[0]==r['oid']),None)
            if not box:continue
            l,t,rr,b=box;q=[(l,t),(rr,t),(rr,b),(l,b)]
        if contains(q):hits.add(r['oid'])
    if len(hits)>1:raise ValueError('Niejednoznaczna warstwa MT. PPM na badge wybranego źródłowego MT FINAL.')
    return next(iter(hits),None)

def context(ed,model):
    from .mobile_canvas_visibility import gt_only
    if gt_only(ed):
        rows=records(ed);row=selected(ed);s=ed._read() or {}
        text='Ręczne GT na tej klatce. D: narysuj obrys; wpisz tekst i zatwierdź Enter.'
        if row and row['plate']:
            plate=row['plate'];text='GT: '+(plate['gt_text'] or plate['text_exclusion_reason'] or 'uzupełnij tekst lub powód nieczytelności')
        return dict(object=text,image='Wyniki modeli ukryte',
            image_note='Po ustaleniu GT: Widok → Pokaż wyniki modeli (MT/MZ).',
            image_counts=dict(gt=str(len(rows)) if rows or s.get('presence')=='NO_PLATES' else '—',tp='—',fp='—',fn='—'),
            session='Ręczne GT · zapis automatyczny',session_note='Zapisane oceny można wznowić po ponownym otwarciu sesji.',
            session_counts=dict(inspected='—',remaining='—',decisions='—'),locator_target=None)
    if (ed._read() or {}).get('scope') == 'SCENE_FRAME':
        from .mobile_canvas_scene_frames import context as scene_context
        return scene_context(ed, model)
    s=ed._read();v=source(ed);r=next((row for row in records(ed) if entry_matches(ed,row)),None) or selected(ed)
    inv=ed.inventory();obj='Wybierz ramkę na obrazie';title='WYBRANA TABLICA'
    if r:
        if r['fp']:
            obj=mt_label(ed,r)+'\nDecyzja operatora: błędna\nPoza tablicą';title='WYBRANE WSKAZANIE'
            obj+='\nŹródło: '+str(r['candidate']['source_pose_status'] or 'brak danych')+' · MZ: '+r['candidate']['mz_status']
        elif r['role']=='UNKNOWN':obj=mt_label(ed,r)+'\nNiepotwierdzone pochodzenie FINAL\nGTR: brak wiarygodnej relacji';title='WYBRANE WSKAZANIE'
        elif not r['plate']:
            title='WYBRANE WSKAZANIE MT';obj=mt_label(ed,r)
            obj+='\nGTR: brak jawnej relacji GTR'
            obj+='\nDecyzja operatora: '+('błędna' if r['decision']==partial.NOT_PLATE else 'poprawna' if r['decision']==partial.PLATE else 'oczekuje')
            obj+='\nŹródło: '+str(r['candidate']['source_pose_status'] or 'brak danych')
            obj+='\nMZ: '+('nie wykonano (NOT_RUN)' if r['candidate']['mz_status']=='NOT_RUN' else r['candidate']['mz_status'])
        else:
            p=r['plate'];gt=(p['gt_text'] if p and p['text_confirmed'] else '')
            geometry=('obrys niepotwierdzony' if p and p['geometry_state']=='unconfirmed' else 'obrys zapisany') if r['gt'] else 'brak obrysu'
            obj=f"GTR {r['n']}: "+geometry if r['n'] is not None else 'GTR: '+geometry
            location=source_location(ed,r)
            if location:obj+=f" · MT {r['candidate']['mt_detection_index']+1} {location['kind']} powiązany"
            elif not r['candidate']:obj+=' · MT: niepowiązany'
            obj+='\nGTZ: '+(gt[:35] or 'nieuzupełniony')
            from .mobile_canvas_info_badge import selected_diagnostic
            row=selected_diagnostic(ed,r)
            ed._clean_geometry_preview=(ed.cache_key,r['oid'],row)
            if r['conflict']:geo='Nie można ustalić referencji GT'
            elif row and row['polygon_iou_pct'] is not None:
                geo=f"Polygon IoU {row['polygon_iou_pct']:.1f}% · NMCE {row['nmce_pct']:.1f}%".replace('.',',')
            elif r['candidate'] and not r['candidate']['mt_quad']:geo='nie można zmierzyć — brak 4 narożników MT'
            elif not r['candidate']:geo='nie można zmierzyć — brak wskazania MT'
            else:geo='jeszcze nieocenione'
            obj+='\nPołożenie: '+geo
            obj+='\nDecyzja ramki: '+('poprawna' if r['decision']==partial.PLATE else 'nieustalona')
            if r['pid'] in partial.progress(s)['pending_gtr_ids']:
                obj+='\nNowa lub zmieniona GTR — zatwierdź Spacją'
            if r['candidate'] and r['candidate']['mz_ran']:
                obj+='\nMZ: '+(r['candidate']['mz_prediction'][:32] or 'brak odczytanego numeru')
                if row and row['cer'] is not None:obj+=f" · CER {100*row['cer']:.1f}%".replace('.',',')
            elif not r['candidate'] or r['candidate']['mz_status']=='NOT_RUN':obj+='\nNie wykonano odczytu'
            else:obj+='\nOdczyt: brak potwierdzonych danych'
    detection=None
    if ed.h.session and ed.h.current_invocation:
        from ..ranking.mobile_mt_object_evaluation import evaluate_unit, reason_text
        detection=evaluate_unit(ed.h.session,ed.h.current_invocation)
        measured=detection['status']=='MEASURED'
        if r and r['fp']:
            outcome=next((p for p in detection['final_outcomes'] if p['attempt_id']==r['candidate']['attempt_id']),None)
            obj+='\nWynik MT v2: '+(outcome['outcome'] if outcome else '—')
        if r and not r['fp'] and r['role']!='UNKNOWN':
            outcome=next((p for p in detection['final_outcomes'] if r['candidate']
                          and p['attempt_id']==r['candidate']['attempt_id']),None)
            status=(outcome['outcome']+(' (duplikat)' if outcome.get('reason')=='DUPLICATE_FINAL' else '')
                    if outcome else 'FN' if measured and r['pid'] in detection.get('unmatched_gt_ids',[]) else '—')
            decision='poprawna' if r['decision']==partial.PLATE else 'nieustalona'
            obj=obj.replace('Decyzja ramki: '+decision,'Decyzja operatora: '+('poprawna' if decision=='poprawna' else 'oczekuje')+'\nWynik MT v2: '+status)
            if r['candidate'] and not r['mt'] and r['gt']:
                overlap=None
                if measured:
                    pair=next((p for p in detection['matched_pairs'] if p['attempt_id']==r['candidate']['attempt_id']),None)
                    overlap=pair.get('bbox_iou') if pair else None
                obj+='\nBBOX IoU: '+(f'{100*overlap:.1f}%'.replace('.',',') if overlap is not None else '—')+' · MT: '+status
                obj+='\nPolygon IoU / NMCE: — · brak narożników MT'
    if location_only(r) and r['plate']:
        title='WYBRANE WSKAZANIE MT'
        label=f"MT {r['candidate']['mt_detection_index']+1} · BBOX · brak narożników"
        saved=r['plate']['gt_text'] if r['plate'] else ''
        obj=label+'\nGTR: brak obrysu · Pose: nie można zmierzyć'
        if r['fp']:obj+='\nDecyzja: błędne wskazanie'
        else:
            obj+=('\nD: uzupełnij GTR' if r['decision']==partial.PLATE else '\nSpacja: potwierdź tablicę')
            obj+=' · Ctrl+PPM: powiąż z GTR'
            obj+=('\nMZ: '+(r['candidate']['mz_prediction'][:32] or 'brak odczytanego numeru')
                  if r['candidate']['mz_ran'] else '\nNie wykonano odczytu' if r['candidate']['mz_status']=='NOT_RUN'
                  else '\nOdczyt: brak potwierdzonych danych')
        obj+='\nGTZ: '+(saved[:35] or 'nieuzupełniony')
    from ..ranking.mobile_review_status import approval_caption
    status=(approval_caption(inv,ed.h.session.review.review_status,ed.h.session.review.review_revision)
            if ed.h.session else 'Wczytywanie oceny…')
    report=getattr(ed.h,'report',None);stale=False
    if ed.h.session and report and report.source_archive_sha256.lower()==ed.h.session.review.source_archive_sha256.lower():
        gate=getattr(ed.h.browser,'_comparison_verification',{}).get((str(report.report_id),report.source_archive_sha256.lower()))
        stale=bool(gate and gate.state=='STALE')
    pending=inv.get('task_decisions',inv.get('approval_decisions',inv['decisions']))
    progress=(f"Eksport nieaktualny · decyzje: {pending}" if stale else f"{status} · Pozostałe decyzje: {pending}")
    session=f"Sprawdzone: {inv['inspected']}/{inv['total']} · Pozostało: {inv['remaining']}\n{progress}"
    assessed = inv['inspected']
    if inv.get('experiment_family') == 'F5A':
        from ..ranking.mobile_review_status import approval_caption
        assessed = inv.get('operator_assessed', inv['assessed'])
        session = approval_caption(inv, ed.h.session.review.review_status, ed.h.session.review.review_revision)
    session_counts = dict(inspected=f"{assessed}/{inv['total']}", remaining=str(inv['remaining']), decisions=str(inv.get('decisions',pending)))
    if inv.get('operator_review_progress'):
        from ..ranking.mobile_operator_progress import image_counts
        human = image_counts(inv['operator_review_progress'])
        session = approval_caption(inv, ed.h.session.review.review_status, ed.h.session.review.review_revision)
        session_counts.update(inspected=human['fraction'], remaining=str(human['remaining']) if human['remaining'] is not None else 'Nie ustalono')
    if r and r['pid'] and not r['candidate']:
        from ..ranking.mobile_mt_geometry_layers import overlapping_references
        overlap=overlapping_references(s,v).get(r['pid'],[])
        if overlap:obj+='\nNakłada się z GTR '+', '.join(map(str,overlap))+'. R: usuń; Ctrl+Z: cofnij.'
    if hasattr(ed,'navigator'):
        from .mobile_canvas_geometry_layers import model as layer_model
        from .mobile_canvas_navigator import source_context,label
        from .mobile_canvas_info_badge import full_text_context
        value=layer_model(ed)
        if value and value['active']!='GTR':title,obj=source_context(ed,value)
        elif r:title=label(ed,'GTR').upper()
        obj+=full_text_context(ed)
    from .mobile_canvas_review_feedback import image_feedback,operator_view
    feedback=image_feedback(ed,detection);operational=operator_view(ed)
    operational_counts={k:str(v) if v is not None else '—' for k,v in operational['counts'].items()}
    measured=bool(detection and detection['status']=='MEASURED')
    bbox_counts={k:str(detection[k]) if measured else '—' for k in ('tp','fp','fn')}
    bbox_scope=f"W ocenie: {detection['n_gt']} GT · wskazania MT: {detection['n_final_mt']}" if measured else 'Wynik oczekuje na kompletną ocenę'
    bbox_note=feedback['text'] if measured else reason_text(detection) if detection else 'Brak zgodnego źródła'
    from .mobile_canvas_review_tasks import for_scene,pending_review_units
    task=inv.get('groups',{}).get(ed.h.current_invocation,{}).get('review_tasks')
    if task is None:task=for_scene(ed.h.session,ed.h.current_invocation,s) if s else pending_review_units(None,None)
    from .mobile_canvas_task_locator import task_target
    action_text,locator_target=task_target(ed,task,s,v)
    image_note=('Do zrobienia: ' if task['needs_review'] else '')+action_text
    image_counts=dict(gt=operational_counts['gt'],**bbox_counts)
    # An untouched image still cannot claim a confirmed empty scene. Once an
    # explicit rejection is shown, zero means zero saved references, alongside
    # the partial-review explanation, not an assertion that no plate exists.
    if feedback.get('partial_fp_lower_bound') is not None:
        image_counts['fp']=f"≥{feedback['partial_fp_lower_bound']}"
    if feedback.get('partial_note'):image_note=feedback['partial_note']+'\n'+image_note
    if operational.get('gt_reason'):image_note='Zapisane GT: '+operational['gt_reason']+'.\n'+image_note
    image=' · '.join(k.upper()+': '+v for k,v in image_counts.items())+'\n'+image_note
    obj+='\n\nBieżący obraz · szczegóły: FINAL '+operational_counts['mt']+' · odrzucone przez operatora '+operational_counts['rejected']+' · GT bez powiązania '+operational_counts['unlinked']
    obj+='\n'+bbox_scope+'\n'+bbox_note
    if task['steps']:obj+='\nOsobne kroki: '+'; '.join(step['text'] for step in task['steps'])
    if task.get('diagnostic_issues'):obj+='\nWarunki oceny: '+'; '.join(task['diagnostic_issues'])
    for unit in task['units']:obj+='\n'+unit['text']+' · '+', '.join(unit['reasons'])
    return dict(object=obj,image=image,session=session,object_title=title,image_counts=image_counts,image_note=image_note,
                bbox_counts=bbox_counts,bbox_note=bbox_note,bbox_scope=bbox_scope,locator_target=locator_target,
                session_counts=session_counts,
                session_note='Eksport nieaktualny' if stale else status)

class CompactContexts(tk.Frame):
    def __init__(self,parent,p):
        super().__init__(parent,bg=p['bg'],padx=10,pady=5);self.fields={};self.titles={};self.data={};self.sections={};self.counters={};self.counter_labels={}
        for key,title in (('object','WYBRANA TABLICA'),('image','BIEŻĄCY OBRAZ'),('session','CAŁA SESJA')):
            frame=tk.Frame(self,bg=p['bg']);frame.pack(fill='x',pady=(0,6));self.sections[key]=frame
            heading=tk.Label(frame,text=title,bg=p['bg'],fg=p['blue'],anchor='w',font=('Segoe UI',10,'bold'));heading.pack(fill='x');self.titles[key]=heading
            specs={'image':(('gt','Zapisane GT'),('tp','Trafne MT (TP)'),('fp','Błędne MT (FP)'),('fn','Pominięte GT (FN)')),
                   'session':(('inspected','Ocenione przez operatora'),('remaining','Pozostało obrazów'),('decisions','Decyzje tablic'))}.get(key,())
            if specs:
                cards=tk.Frame(frame,bg=p['bg']);cards.pack(fill='x',pady=(5,0));columns=2 if key=='image' else 3
                for i,(name,caption) in enumerate(specs):
                    cards.columnconfigure(i%columns,weight=(7,6,7)[i] if key=='session' else 1,uniform=key)
                    card=tk.Frame(cards,bg=p['surface'],padx=5 if key=='image' else 2,pady=3,highlightthickness=1,highlightbackground=p['border'])
                    card.grid(row=i//columns,column=i%columns,sticky='nsew',padx=(0,4 if i%columns<columns-1 else 0),pady=(0,3))
                    number=tk.Label(card,text='—',bg=p['surface'],fg=p['fg'],font=('Segoe UI',20,'bold'),anchor='w',width=0 if key=='image' else 1)
                    caption_label=tk.Label(card,text=caption,bg=p['surface'],fg=p['fg'],font=('Segoe UI',10),anchor='w',justify='left',width=1)
                    if key=='image':
                        number.pack(side='left',padx=(0,6));caption_label.pack(side='left',fill='x',expand=True)
                    else:number.pack(fill='x');caption_label.pack(fill='x')
                    caption_label.bind('<Configure>',lambda e,w=caption_label:w.configure(wraplength=max(55,e.width)) if int(str(w.cget('wraplength')))!=max(55,e.width) else None)
                    self.counters[name]=number;self.counter_labels[name]=caption_label
            host=frame
            if key=='object':
                self.object_details=tk.Frame(frame,bg=p['bg']);host=self.object_details
                self.details_open=False
                self.details_button=tk.Button(frame,text='Szczegóły wskazania ▸',command=self.toggle_details,
                    bg=p['bg'],fg=p['blue'],activebackground=p['bg'],activeforeground=p['blue'],
                    font=('Segoe UI',10),anchor='w',relief='flat',cursor='hand2')
                self.details_button.pack(fill='x',pady=(2,0))
            from .mobile_canvas_task_locator import TaskText
            label=(TaskText(host,p) if key=='image' else tk.Label(host,bg=p['bg'],fg=p['fg'],anchor='nw',justify='left',font=('Segoe UI',10),wraplength=290,width=1))
            label.pack(fill='x',pady=(5,0));self.fields[key]=label
            if key=='image':
                # Anchor for existing image actions, not a second metrics section.
                self.image_actions=tk.Frame(frame,bg=p['bg'],height=0)
                self.image_actions.pack(fill='x')
            if key=='session':
                from .mobile_session_metrics_ui import SessionMetrics
                self.live_metrics=SessionMetrics(frame,p=p)
                self.live_metrics.pack(fill='x')
        self.bind('<Configure>',lambda e:[w.configure(wraplength=max(150,e.width-24)) for w in self.fields.values()])
    def render(self,value):
        self.data={k:value[k] for k in self.fields}
        for k,w in self.fields.items():w.configure(text=value.get(k+'_note',value[k]))
        self.fields['image'].set_target(value.get('locator_target'))
        for group in ('image_counts','session_counts'):
            for name,number in value.get(group,{}).items():self.counters[name].configure(text=number)
        self.titles['object'].configure(text='AKCJE DLA WSKAZANIA')
        self.live_metrics.render(value.get('session_metrics'),value.get('session_metrics_key'))

    def toggle_details(self):
        self.details_open=not self.details_open
        if self.details_open:self.object_details.pack(fill='x',after=self.details_button)
        else:self.object_details.pack_forget()
        self.details_button.configure(text='Szczegóły wskazania ▾' if self.details_open else 'Szczegóły wskazania ▸')

def show_exclusion(ed):
    if not ed.available():return
    ed.clean_reason_kind='exclude'
    ed.clean_reason_label.configure(text='Przyczyna wyłączenia całego obrazu:')
    ed.clean_reason_save.configure(text='Wyłącz obraz')
    ed.clean_exclusion_context=(id(ed.h.session),ed.h.current_invocation)
    ed.clean_reason.set(ed._read().get('exclusion_reason','') if ed._read() else '')
    ed.clean_exclusion.pack(fill='x',pady=(5,0),before=ed.contexts.image_actions);sync_image_actions(ed);ed.clean_reason_entry.focus_set()

def close_exclusion(ed):
    ed.clean_exclusion.pack_forget();sync_image_actions(ed)

def save_exclusion(ed):
    if not ed.available() or ed.clean_exclusion_context!=(id(ed.h.session),ed.h.current_invocation):return False
    reason=ed.clean_reason.get().strip()
    if getattr(ed,'clean_reason_kind','exclude')=='presence' and (ed._read() or {}).get('scope')=='SCENE_FRAME':
        if not reason:ed.message('Podaj przyczynę braku narożników tablicy.');return False
        from ..ranking.mobile_scene_frame_review import add_plate
        oid=ed._commit(lambda:add_plate(ed.h.session,ed.h.current_invocation,geometry_reason=reason),'Zapisano fizyczną tablicę bez fikcyjnych narożników')
        if oid:
            close_exclusion(ed);ed.selected=oid;ed.mode='edit';ed.sync();ed.start_entry(oid)
        return bool(oid)
    if not reason:ed.message('Podaj przyczynę wyłączenia całego obrazu.');return False
    result=ed._commit(lambda:core.exclude_scene(ed.h.session,ed.h.current_invocation,reason),'Zapisano przyczynę wyłączenia obrazu')
    if result:close_exclusion(ed)
    return bool(result)

def restore_image(ed):
    if ed.available():
        ed._commit(lambda:core.set_scene_review_state(ed.h.session,ed.h.current_invocation,checked=True),'Przywrócono ocenę obrazu')
        close_exclusion(ed)

def sync_image_actions(ed):
    context=(id(ed.h.session),ed.h.current_invocation)
    if getattr(ed,'clean_exclusion_context',None)!=context:
        ed.clean_exclusion.pack_forget();ed.clean_reason.set('');ed.clean_exclusion_context=context
    note=ed.contexts.fields['image']
    editing_reason=bool(ed.clean_exclusion.winfo_manager())
    if editing_reason:
        note.pack_forget();ed.clean_exclude_button.pack_forget()
    else:
        if not note['text'].strip():note.pack_forget()
        elif not note.winfo_manager():note.pack(fill='x',pady=(0 if getattr(note,'compact',False) else 5,0),before=ed.contexts.image_actions)
        if not ed.clean_exclude_button.winfo_manager():ed.clean_exclude_button.pack(fill='x',pady=(0 if getattr(note,'compact',False) else 5,0),before=ed.contexts.image_actions)
    scene=ed._read() or {};enabled=ed.available() and not ed.busy and not ed.h._saving
    frame=partial.progress(scene) if scene else None
    row=selected(ed)
    target=bool(frame and row and ((row['pid'] and row['gt']) or (row['candidate'] and row['role']=='FINAL')))
    from .mobile_canvas_geometry_layers import model as layer_model
    layer=layer_model(ed)
    bad_source=bool(row and row['fp'] and row['candidate'] and not
                    (row['plate'] and (ed.gt_focus_id or (layer and layer['active']=='GTR'))))
    verified=False;removed=False
    if target:
        pid=row['pid'] if row['gt'] and not bad_source else None
        verified=partial.stack_is_verified(scene,oid=row['oid'] if not pid else None,plate_id=pid)
        if not row['fp'] and row['candidate'] and row['candidate'].get('spatial_relation',{}).get('plate_id')==pid:
            verified=partial.stack_is_verified(scene,oid=row['candidate']['object_id'])
        removed=bool(row['plate'] and row['plate']['geometry_state']=='operator_removed' and not row['gt'])
        ed.clean_gtr_confirm_button.configure(text='Uznaj wskazanie MT · Spacja' if bad_source else 'Narysuj obrys GTR · D' if removed else 'Cofnij weryfikację · Spacja' if verified else 'Zatwierdź ramki · Spacja',
                                            command=ed.draw_toggle if removed and not bad_source else ed.confirm,
                                            disabledforeground=ed.p['fg'],state='normal' if enabled and ed.mode!='draw' else 'disabled')
        ed.clean_gtr_confirm_button.pack(fill='x',pady=(4,0))
    else:ed.clean_gtr_confirm_button.pack_forget()
    from .mobile_canvas_scene_frames import sync_empty_action
    sync_empty_action(ed, scene, enabled)
    # One stack may be verified while another plate still awaits a decision.
    # Read-only stack state; no image-level confirmation widget is installed.
    ed.frame_checked_var.set(verified)
    from .mobile_canvas_review_feedback import sync_actions
    sync_actions(ed)
    from .mobile_canvas_gtr_reconciliation import sync as sync_reconciliation
    sync_reconciliation(ed)
    ed.clean_exclude_button.configure(state='normal' if enabled else 'disabled',text='Zmień przyczynę wyłączenia…' if scene.get('exclusion_reason') else 'Nie można ocenić całego obrazu…')
    ed.clean_restore_button.configure(state='normal' if enabled else 'disabled')
    if scene.get('exclusion_reason') and not editing_reason:ed.clean_restore_button.pack(fill='x',pady=(4,0),before=ed.contexts.image_actions)
    else:ed.clean_restore_button.pack_forget()

def menu_colors(p):
    return dict(background=p['surface'],foreground=p['fg'],activebackground='#145A7A',activeforeground='#FFFFFF',
                disabledforeground=p['muted'],activeborderwidth=0,borderwidth=1,relief='solid')

def install(ed,viewer):
    p=ed.p;ed.gt_focus_id='';ed.show_fp_var=tk.BooleanVar(master=ed.h.window,value=True)
    from .mobile_canvas_info_badge import install_focus_cleanup
    install_focus_cleanup(ed)
    ed.ref_visible=tk.BooleanVar(master=ed.h.window,value=True)
    ed.inf_visible=tk.BooleanVar(master=ed.h.window,value=True)
    parent=ed.contexts.master;ed.contexts.grid_remove()
    # Reuse the original right-panel scroll. Keep its scrollbar hidden, as in
    # the clean view, so the existing cards retain their exact width/positions.
    body=ed.panel.master;body.configure(padx=0,pady=0)
    ed.contexts=CompactContexts(body,p)
    ed.contexts.grid(row=0,column=0,sticky='new')
    parent.rowconfigure(0,weight=1);parent.rowconfigure(1,weight=0)
    ed.h.review_sidebar_scroll.grid(row=0,column=0,columnspan=2,sticky='nsew')
    ed._sidebar_bar_width=0
    for w in parent.winfo_children():
        if w.winfo_class()=='TScrollbar':
            w.grid_remove()
    ed.h.review_sidebar_scroll.configure(width=max(1,ed.sidebar_width+12-ed._sidebar_bar_width))
    ed.h._bind_sidebar_wheel(ed.contexts)
    ed.panel.grid_remove()
    ed.frame_checked_var=tk.BooleanVar(master=ed.h.window,value=False)
    button_options=dict(bg=p['surface'],fg=p['fg'],activebackground='#145A7A',activeforeground='#FFFFFF',font=('Segoe UI',10),relief='solid',bd=1,pady=3)
    ed.clean_gtr_confirm_button=tk.Button(ed.contexts.sections['object'],text='Zatwierdź ramki · Spacja',command=ed.confirm,**button_options)
    from .mobile_canvas_review_feedback import install as install_feedback
    install_feedback(ed,button_options)
    from .mobile_canvas_gtr_reconciliation import install as install_reconciliation
    install_reconciliation(ed,button_options)
    ed.clean_exclude_button=tk.Button(ed.contexts.sections['image'],text='Nie można ocenić całego obrazu…',command=lambda:show_exclusion(ed),**button_options)
    ed.clean_exclude_button.pack(fill='x',pady=(5,0),before=ed.contexts.image_actions)
    ed.clean_restore_button=tk.Button(ed.contexts.sections['image'],text='Przywróć ocenę obrazu',command=lambda:restore_image(ed),**button_options)
    ed.clean_exclusion_context=None
    ed.clean_exclusion=tk.Frame(ed.contexts.sections['image'],bg=p['bg']);ed.clean_reason=tk.StringVar(master=ed.h.window)
    ed.clean_reason_label=tk.Label(ed.clean_exclusion,text='Przyczyna wyłączenia całego obrazu:',bg=p['bg'],fg=p['fg'],anchor='w',font=('Segoe UI',10));ed.clean_reason_label.pack(fill='x')
    ed.clean_reason_entry=tk.Entry(ed.clean_exclusion,textvariable=ed.clean_reason,font=('Segoe UI',10),bg=p['surface'],fg=p['fg'],insertbackground=p['fg']);ed.clean_reason_entry.pack(fill='x')
    ed.clean_reason_entry.bind('<Return>',lambda e:save_exclusion(ed))
    ed.clean_reason_save=tk.Button(ed.clean_exclusion,text='Wyłącz obraz',command=lambda:save_exclusion(ed),**button_options);ed.clean_reason_save.pack(side='left')
    tk.Button(ed.clean_exclusion,text='Anuluj',command=lambda:close_exclusion(ed),**button_options).pack(side='right')
    ed.clean_bar=tk.Frame(viewer,bg=p['surface'],height=30);ed.clean_bar.grid(row=0,column=0,sticky='ew',pady=(0,3))
    ed.clean_bar.columnconfigure(1,weight=1)
    ed.clean_caption=tk.Label(ed.clean_bar,bg=p['surface'],fg=p['fg'],font=('Segoe UI',10,'bold'),anchor='w');ed.clean_caption.grid(row=0,column=0,padx=5)
    from .mobile_canvas_info_badge import ExecutionHUD
    ed.source_hud=ExecutionHUD(ed.clean_bar,p)
    ed.source_hud.grid(row=1,column=0,columnspan=3,sticky='ew',padx=5)
    ed.clean_gt_slot=tk.Frame(ed.clean_bar,bg=p['surface'],height=28);ed.clean_gt_slot.grid(row=0,column=1,sticky='ew',padx=5)
    ed.clean_gt_slot.grid_propagate(False)
    ed.clean_gt_label=tk.Label(ed.clean_gt_slot,bg=p['surface'],fg='#FFD400',font=('Segoe UI',10),anchor='w',cursor='hand2')
    ed.clean_gt_label.place(x=0,y=0,relwidth=1,height=27)
    ed.clean_gt_label.bind('<Button-1>',lambda e:ed.start_entry(ed.selected) if ed.selected else None)
    menu_button=tk.Menubutton(ed.clean_bar,text='Widok ⋯',bg=p['surface'],fg=p['fg'],activebackground='#145A7A',activeforeground='#FFFFFF',relief='flat');menu_button.grid(row=0,column=2,padx=5)
    menu=tk.Menu(menu_button,tearoff=False,**menu_colors(p));menu_button.configure(menu=menu)
    menu.add_checkbutton(label='Pokaż ramki referencyjne (REF/GTR)',variable=ed.ref_visible,command=lambda:refresh_contours(ed))
    menu.add_checkbutton(label='Pokaż ramki inferowane (INF/MT)',variable=ed.inf_visible,command=lambda:refresh_contours(ed))
    menu.add_separator()
    menu.add_checkbutton(label='Pokaż odrzucone wskazania (PPM)',variable=ed.show_fp_var,command=lambda:refresh_visibility(ed))
    menu.add_command(label='Pokaż zapisany crop MZ wybranego wskazania',command=ed.show_crop,state='disabled')
    ed.clean_mz_crop_index=menu.index('end')
    ed.clean_menu=menu
    help_menu=tk.Menu(menu,tearoff=False,**menu_colors(p));menu.add_cascade(label='Skróty i obsługa',menu=help_menu);ed.clean_help_menu=help_menu
    for text in ('S / klik: wybierz obiekt','Spacja: przełącz weryfikację skojarzonych ramek','GTR: zielona — zatwierdzona; pomarańczowa — oczekuje','Obwódka obrazu: zielona — zweryfikowany; czerwona — oczekuje','Z/X: poprzednia/następna warstwa','↑/↓: poprzednie/następne wskazanie modelu','Warstwę wybierzesz także z listy nad zdjęciem','LPM na narożniku: korekta GTR; Esc: koniec','Klik w badge obrysu operatora: edycja GTZ','PPM: błędne wskazanie; Ctrl+Z/Y: historia','Ctrl+PPM: powiązania i pochodzenie','D: cztery narożniki GTR · R: usuń/cofnij referencję','A: następna tablica + dopasowanie · F: cały obraz · rolka: zoom','Q/E: poprzedni/następny obraz · N: następny obrys'):
        help_menu.add_command(label=text,command=lambda message=text:ed.message(message))
    ed.h.progress_label.grid_remove()
    from .mobile_canvas_navigator import Navigator
    ed.navigator=Navigator(ed)
    from .mobile_canvas_task_locator import Locator
    ed.task_locator=Locator(ed)
    if ed.h.form_status_label.winfo_manager()=='pack':ed.h.form_status_label.pack_forget()
    elif ed.h.form_status_label.winfo_manager()=='grid':ed.h.form_status_label.grid_remove()
