"""Canvas helpers for the existing editor; no second controller or canvas."""
import copy, math
import tkinter as tk
from ..ranking import mobile_candidate_stacks as stacks
from ..ranking import mobile_canvas_review as core

def view(ed):
    if getattr(ed,'reference_source',None) is not None:return None
    s=ed._read()
    if not s:return None
    key=(id(ed.h.session),ed.h.session.review.source_archive_sha256,ed.h.current_invocation,ed.h.session.review.review_revision)
    if getattr(ed,'_stack_view_key',None)!=key:
        ed._stack_view=stacks.snapshot(ed.h.session,ed.h.current_invocation)
        ed._stack_view_key=key
    return ed._stack_view

def enhanced(ed):
    v=view(ed)
    if hasattr(ed,'clean_bar'):return bool(v)
    return bool(v and not v['conflicts'] and (any(c['role']=='SUPPRESSED' or not c['mt_quad'] or c['association_status']=='UNASSOCIATED_NO_PLATE' for c in v['candidates'])
                or any(len(members(v,pid))>1 for pid in v['plates'])))

def write_reference(ed,oid,*,quad=None,text=None,reason=''):
    if getattr(ed,'reference_source',None) is not None:
        return ed.reference_source.edit(oid,quad=quad,text=text,reason=reason)
    if hasattr(ed,'clean_bar') and oid==ed.selected:
        from .mobile_canvas_clean_view import selected
        row=selected(ed)
        if row and row['pid'] and not row['candidate']:
            return stacks.edit_gt(ed.h.session,ed.h.current_invocation,row['pid']+':reference',quad=quad,text=text,reason=reason)
    from .mobile_canvas_spatial_reference import selected_relation,canonical_action
    relation=selected_relation(ed,oid) if hasattr(ed,'clean_bar') else None
    if relation:
        source,pid=relation
        return canonical_action(ed.h.session,ed.h.current_invocation,source,pid,operation='edit',quad=quad,text=text,reason=reason)
    v=view(ed);pid=(v or {}).get('links',{}).get(oid,{}).get('plate_id')
    shared=stacks.KEY in ed._read() or (pid and len(members(v,pid))>1)
    if shared:return stacks.edit_gt(ed.h.session,ed.h.current_invocation,oid,quad=quad,text=text,reason=reason)
    if quad is not None:return core.correct_plate(ed.h.session,ed.h.current_invocation,oid,quad)
    return core.set_text(ed.h.session,ed.h.current_invocation,oid,text,reason)

def members(v,pid):
    return [c for c in v['candidates'] if c['plate_id']==pid and c['role']=='FINAL'
            and c['association_status']=='ASSOCIATED_FOR_GEOMETRY']

def active(ed,v,pid):
    choices=members(v,pid)
    if not choices:return None
    return next((c for c in choices if c['object_id']==ed.selected),choices[0])

def cycle(ed,event,back=False,*,oid=None):
    if ed.entry is not None or ed._is_typing_shortcut(getattr(event,'widget',ed.c)):return None
    if ed.drag or ed.points or getattr(ed,'mode','')=='draw':return 'break'
    if hasattr(ed,'clean_bar'):
        from .mobile_canvas_clean_view import source
        v=source(ed)
    else:v=view(ed)
    if v and v['conflicts']:return None
    selected=next((c for c in (v or {}).get('candidates',[]) if c['object_id']==(oid or ed.selected)),None)
    if not selected or not selected['plate_id']:return None
    choices=members(v,selected['plate_id'])
    if len(choices)<2 or selected not in choices:return None
    i=choices.index(selected)
    ed._choose_plate(choices[(i+(-1 if back else 1))%len(choices)]['object_id'])
    return 'break'


def control_hit(ed,x,y):
    return next(((oid,back) for (l,t,r,b),oid,back in getattr(ed,'mt_stack_controls',())
                 if l<=x<=r and t<=y<=b),None)


def paint_candidate_badge(ed,row,anchor):
    """Current FINAL / FINAL count for one shared GT, attached to source MT.

    A single candidate is explicitly 1/1, with inactive arrows. This count
    never includes GTR contours, unrelated GT, history or suppressed RAW.
    """
    if not row['members'] or row['conflict'] or row['candidate'] not in row['members']:return
    import tkinter.font as tkfont
    from .mobile_canvas_editor import TAG
    from .mobile_canvas_clean_view import bounds,intersects,badge_tether
    if not hasattr(ed,'_mt_badge_font'):ed._mt_badge_font=tkfont.Font(root=ed.c,font=('Segoe UI',8,'bold'))
    font=ed._mt_badge_font;count=len(row['members']);index=row['members'].index(row['candidate'])+1
    label=f'MT {index}/{count}';h=max(23,font.metrics('linespace')+6);button=h
    w=font.measure(label)+14+2*button;box=bounds(ed,anchor)
    cw,ch=ed.c.winfo_width(),ed.c.winfo_height();rect=None
    # Place at an end of the original contour; protect digits and handles.
    for x,y in ((box[2]-w,box[3]+4),(box[2]+4,box[3]-h),
                (box[2]-w,box[1]-h-4),(box[0]-w-4,box[1])):
        proposal=(x,y,x+w,y+h)
        if x<2 or y<2 or x+w>cw-2 or y+h>ch-2:continue
        if not any(intersects(proposal,b) for b in ed.clean_geometry_boxes+ed.clean_label_boxes):
            rect=proposal;break
    if rect is None:
        # Search free viewport edges at extreme zoom; never cover digits or
        # GT handles. If geometry fills the entire viewport, keyboard controls
        # and the existing toolbar count remain available.
        for y in (2,ch-h-2):
            for x in (2,cw-w-2,(cw-w)/2):
                proposal=(x,y,x+w,y+h)
                if x<2 or y<2 or x+w>cw-2 or y+h>ch-2:continue
                if not any(intersects(proposal,b) for b in ed.clean_geometry_boxes+ed.clean_label_boxes):
                    rect=proposal;break
            if rect:break
    if rect is None:return
    ed.clean_label_boxes.append(rect);oid=row['oid'];ed.mt_stack_badges[oid]=rect
    x,y,r,b=rect;tags=(TAG,'mt_stack_control',oid,row['candidate']['attempt_id'])
    badge_tether(ed,anchor,rect,oid,tag='mt_stack_tether')
    ed.c.create_rectangle(*rect,fill='#14252D',outline='#58B8D5',tags=tags)
    ed.c.create_text(x+7,(y+b)/2,text=label,anchor='w',font=font,fill='#D9F1FA',tags=tags+('mt_stack_count','clean_label_text'))
    for i,(glyph,back) in enumerate((('▲',True),('▼',False))):
        l=r-(2-i)*button;edge=(l,y,l+button,b)
        ed.c.create_rectangle(*edge,fill='#145A7A' if count>1 else '#233844',outline='#58B8D5',tags=tags)
        ed.c.create_text(l+button/2,(y+b)/2,text=glyph,font=font,fill='white' if count>1 else '#9BAAB3',tags=tags)
        if count>1:ed.mt_stack_controls.append((edge,oid,back))

def paint(ed):
    if hasattr(ed,'clean_bar'):
        from .mobile_canvas_clean_view import paint as clean_paint
        return clean_paint(ed)
    from .mobile_canvas_editor import TAG,overlay_label_position
    if not enhanced(ed):return False
    s=ed._read();v=view(ed);c=ed.c
    if ed.h._view_center is None:return False
    ed.badges={};ed.card_positions={};ed.card_widths={};ed.card_heights={};ed.card_header_heights={}
    ed.stack_badges={};ed.stack_hit=[];ed.stack_bbox_hit=[]
    w,h=c.winfo_width(),c.winfo_height();placed=[]
    def poly(quad,color,tag,oid):
        if quad:c.create_polygon(*[x for p in quad for x in ed.to_canvas(p)],outline=color,fill='',width=1.5,tags=(TAG,tag,oid))
    def badge(quad,title,number,oid,pid,color):
        pts=[ed.to_canvas(p) for p in quad] if quad else [(8,8),(20,8),(20,20),(8,20)]
        bounds=(min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts))
        bw=min(265,max(175,len(title)*7),max(120,w-16));bh=50
        pos=overlay_label_position(bounds,(bw,bh),(w,h),placed+[bounds])
        if pos is None:pos=(8,max(8,min(h-bh-8,8+len(placed)*56)))
        x,y=pos;placed.append((x,y,x+bw,y+bh))
        c.create_rectangle(x,y,x+bw,y+24,fill='#102C3A',outline=color,tags=(TAG,'stack_badge',oid))
        c.create_text(x+5,y+3,text=title,anchor='nw',fill='#EBF7FF',font=('Segoe UI',9,'bold'),width=bw-10,tags=(TAG,'stack_badge_text','mz_badge',oid))
        c.create_rectangle(x,y+25,x+bw,y+bh,fill='#302618',outline=color,tags=(TAG,'gt_badge',oid))
        c.create_text(x+5,y+28,text=number,anchor='nw',fill=color,font=('Segoe UI',9,'bold'),width=bw-10,tags=(TAG,'gt_badge',oid))
        ed.card_positions[oid]=(x,y);ed.card_widths[oid]=bw;ed.card_header_heights[oid]=24;ed.card_heights[oid]=bh
        editable=pid or any(k['object_id']==oid and k['role']=='FINAL' and k['mt_quad'] and
                           k['association_status']!='UNASSOCIATED_NO_PLATE' for k in v['candidates'])
        if editable:ed.badges[oid]=(x,y+25,x+bw,y+bh)
        ed.stack_badges[oid]=(x,y,x+bw,y+bh,pid)
        reference=next((p['reference_object_id'] for p in s.get('physical_gt_v1',{}).get('plates',[]) if p['gt_id']==pid),None)
        if ed.entry is not None and ed.entry_target and ed.entry_target[1] in (oid,(v['plates'].get(pid) or {}).get('origin_object_id'),reference):
            ed.entry.place(x=x,y=y+25,width=bw,height=26);ed.entry.lift()
    if ed.show_roi_overlay:
        l,t,r,b=s['roi'];a=ed.to_canvas((l,t));z=ed.to_canvas((r,b))
        c.create_rectangle(*a,*z,outline='#FFD16C',tags=(TAG,'analyzed_roi'))
    from ..ranking.mobile_canvas_operator_view import reference_numbers
    numbers=reference_numbers(v)
    for index,(pid,p) in enumerate(v['plates'].items(),1):
        n=numbers.get(pid,index)
        candidate=active(ed,v,pid);choices=members(v,pid)
        oid=candidate['object_id'] if candidate else p['origin_object_id']
        quad=copy.deepcopy(p['gt_quad'])
        if ed.drag and ed.drag['oid']==oid:quad=ed.drag['quad']
        if candidate:poly(candidate['mt_quad'],'#00E5FF','mt_original',candidate['attempt_id'])
        poly(quad,'#FFD400','operator_quad',oid)
        if quad:
            ed.stack_hit.append((oid,quad,'body'))
            if candidate and candidate['mt_quad']:ed.stack_hit.append((oid,candidate['mt_quad'],'body'))
            if ed.selected==oid and ed.mode=='edit':
                for x,y in [ed.to_canvas(p) for p in quad]:c.create_oval(x-6,y-6,x+6,y+6,fill='white',outline='#FFD400',tags=(TAG,'corner_handle',oid))
        index=choices.index(candidate)+1 if candidate else 0
        title=f'Tablica {n} · {len(choices)} MT'+(f' · {n}.{index}/{len(choices)}' if candidate and ed.selected==oid else '')
        number=ed.entry_var.get() if ed.entry is not None and ed.entry_target[1] in (oid,p['origin_object_id']) else p['gt_text']
        if len(choices)==1 and candidate['role']=='FINAL':
            from ..ranking.mobile_canvas_presentation import plate_review_state
            state=plate_review_state(s,core._object(s,oid),ed._canvas_execution)
            title=f'Tablica {n} · '+state['canvas_frame'].replace('MT ·',f"MT {candidate['mt_detection_index']+1} ·")
        badge(quad,title,('GT: ' if len(choices)>1 else 'Numer: ')+(number or 'niepodany'),oid,pid,'#FFD400')
    for candidate in v['candidates']:
        if candidate['plate_id']:continue
        oid=candidate['object_id'];quad=candidate['mt_quad'];raw=candidate['role']=='SUPPRESSED'
        anchor=quad
        if quad is None:
            from ..ranking.mobile_canvas_resolution import model_location
            try:location=model_location(ed.h.session,ed.h.current_invocation,s,core._object(s,oid))
            except (ValueError,KeyError,TypeError):location=None
            if location and location['kind']=='BBOX':
                l,t,r,b=location['bbox'];a=ed.to_canvas((l,t));z=ed.to_canvas((r,b))
                c.create_rectangle(*a,*z,outline='#77CBF0',dash=(6,3),tags=(TAG,'model_location_bbox',oid))
                ed.stack_bbox_hit.append((oid,l,t,r,b));anchor=[(l,t),(r,t),(r,b),(l,b)]  # placement only, never Pose/metrics
        if not raw or ed.selected==oid:poly(quad,'#FF7C8C' if candidate['association_status']=='UNASSOCIATED_NO_PLATE' else '#00E5FF','mt_original',candidate['attempt_id'])
        if (quad and ed.selected==oid and ed.mode=='edit' and candidate['role']=='FINAL'
                and candidate['association_status']!='UNASSOCIATED_NO_PLATE'):
            for x,y in [ed.to_canvas(p) for p in quad]:c.create_oval(x-6,y-6,x+6,y+6,fill='white',outline='#FFD400',tags=(TAG,'corner_handle',oid))
        if quad and (not raw or ed.selected==oid):ed.stack_hit.append((oid,quad,'body'))
        role='RAW odrzucona' if raw else candidate['role']+(' · FP' if candidate['association_status']=='UNASSOCIATED_NO_PLATE' else '')
        badge(anchor,f"MT {candidate['mt_detection_index']+1} · {role}",'FP — poza tablicą' if candidate['association_status']=='UNASSOCIATED_NO_PLATE' else 'bez przypisania GT',oid,None,'#FF7C8C' if candidate['association_status']=='UNASSOCIATED_NO_PLATE' else '#77CBF0')
    ed.h.image_help.configure(text='Z / X: w dół / w górę stosu · PPM: poza tablicą · Ctrl+klik GT: powiąż · Ctrl+PPM: decyzje GT')
    selected=next((k for k in v['candidates'] if k['object_id']==ed.selected),None)
    if selected and selected['role']=='FINAL' and (not selected['plate_id'] or len(members(v,selected['plate_id']))==1):
        ed._paint_live_overlap_hud(s,ed._canvas_execution)
    return True

def hit(ed,x,y):
    if hasattr(ed,'clean_bar'):
        from .mobile_canvas_clean_view import hit as clean_hit
        return clean_hit(ed,x,y)
    from .mobile_canvas_editor import quad_contains_point
    if not enhanced(ed):return None
    if ed.selected and ed.mode=='edit':
        obj=next((o for o in ed._read()['objects'] if o['object_id']==ed.selected),None)
        selected=next((c for c in view(ed)['candidates'] if c['object_id']==ed.selected),None)
        if obj and (not selected or selected['plate_id'] or (selected['role']=='FINAL' and selected['association_status']!='UNASSOCIATED_NO_PLATE')):
            for i,p in enumerate(ed._current_quad(obj) or []):
                if math.dist((x,y),ed.to_canvas(p))<=14:return ed.selected,'corner',i
    for oid,(l,t,r,b,pid) in reversed(list(getattr(ed,'stack_badges',{}).items())):
        if l<=x<=r and t<=y<=b:return oid,('badge' if oid in ed.badges and y>t+24 else 'mz_badge'),None
    source=ed.to_source(x,y)
    for oid,l,t,r,b in getattr(ed,'stack_bbox_hit',[]):
        if l<=source[0]<=r and t<=source[1]<=b:return oid,'body',None
    hits=[]
    for oid,quad,kind in reversed(getattr(ed,'stack_hit',[])):
        inside=quad_contains_point(quad,*source)
        points=[ed.to_canvas(p) for p in quad]
        for a,b in zip(points,points[1:]+points[:1]):
            dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
            u=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/den)) if den else 0
            inside=inside or math.dist((x,y),(a[0]+u*dx,a[1]+u*dy))<=11
        if inside:hits.append((oid,kind,None))
    return next((hit for hit in hits if hit[0]==ed.selected),hits[0] if hits else None)

def ambiguous(ed,x,y,target):
    """PPM must never choose among overlapping independent GT/candidates."""
    from .mobile_canvas_editor import quad_contains_point
    if ed.selected:return False
    v=view(ed)
    if not v:return False
    candidate=next((c for c in v['candidates'] if c['object_id']==target[0]),None)
    if candidate and candidate['plate_id'] and len(members(v,candidate['plate_id']))>1:return True
    if target[1] in ('badge','mz_badge'):return False
    point=ed.to_source(x,y);hits=set()
    for obj in ed._read()['objects']:
        if obj.get('quad') and quad_contains_point(obj['quad'],*point):hits.add(obj['object_id'])
    for c in v['candidates']:
        if c['mt_quad'] and quad_contains_point(c['mt_quad'],*point):hits.add(c['object_id'])
    return len(hits)>1

def associate_click(ed,event):
    if not getattr(event,'state',0)&4 or not ed.selected:return False
    v=view(ed)
    for oid,(l,t,r,b,pid) in getattr(ed,'stack_badges',{}).items():
        if l<=event.x<=r and t<=event.y<=b:
            if pid:ed._commit(lambda:stacks.associate(ed.h.session,ed.h.current_invocation,ed.selected,pid),'Powiązano kandydat z jedną referencją GT')
            elif oid==ed.selected:ed._commit(lambda:stacks.accept_reference(ed.h.session,ed.h.current_invocation,oid),'Przyjęto wskazany obrys jako jedną referencję GT — ASSISTED')
            else:ed.message('Najpierw wybierz docelowy MT i Ctrl+kliknij jego badge, aby przyjąć referencję GT.')
            return True
    if not enhanced(ed):
        target=ed.hit(event.x,event.y)
        if target and target[0]==ed.selected and target[1] in ('badge','mz_badge'):
            ed._commit(lambda:stacks.accept_reference(ed.h.session,ed.h.current_invocation,ed.selected),'Przyjęto wskazaną referencję GT — ASSISTED')
            return True
    return False

def context_menu(ed,event=None,*,show=True):
    """Existing identity decisions, contextual to the explicitly selected Canvas object."""
    from ..ranking import mobile_canvas_resolution as resolution,mobile_canvas_physical_gt as physical
    if ed.points or ed.drag or getattr(ed,'gt_focus_id',''):
        ed.message('Dokończ rysowanie lub anuluj szkic przez Esc.');return None
    scene_scope=(ed._read() or {}).get('scope')=='SCENE_FRAME'
    if (not ed.selected and not scene_scope) or not ed.available():
        ed.message('Najpierw wybierz obiekt na Canvas.');return None
    previous=getattr(ed,'_canvas_decision_menu',None)
    if previous is not None:
        try:previous.destroy()
        except tk.TclError:pass
    s=ed._read();v=view(ed);oid=ed.selected;gid=ed.h.current_invocation;session=ed.h.session
    obj=core._object(s,oid) if oid else None;candidate=next((c for c in v['candidates'] if c['object_id']==oid),None)
    menu=tk.Menu(ed.c,tearoff=False)
    def add(label,action):menu.add_command(label=label,command=lambda:ed._commit(action,'Zapisano decyzję wskazanego obiektu Canvas'))
    if s.get('scope') == 'SCENE_FRAME':
        from ..ranking.mobile_scene_frame_review import role_for_final, decide_mt
        from .mobile_canvas_scene_frames import show_presence_only
        menu.add_command(label='Tablica widoczna, narożniki niedostępne…',command=lambda:show_presence_only(ed))
        for item in s['objects']:
            if item['quad'] is None:
                menu.add_command(label='Wybierz tablicę bez narożników: '+(item['gt_text'] or 'GTZ nieuzupełniony'),command=lambda target=item['object_id']:ed._choose_plate(target))
        index=0
        for role in session.mt_detection_roles.values():
            for aid in role['final_attempt_ids']:
                try:role_for_final(session,gid,aid)
                except ValueError:continue
                index+=1
                if oid:add(f'Ta GTR odpowiada FINAL MT {index}',lambda target=aid:decide_mt(session,gid,target,reference=oid))
                add(f'FINAL MT {index} jest błędnym wskazaniem',lambda target=aid:decide_mt(session,gid,target,is_plate=False))
    if candidate:
        if candidate['role']=='FINAL' and not candidate['plate_id'] and not obj['error'] and candidate['mt_quad']:
            add('Ustanów GT z tego obrysu MT',lambda:stacks.accept_reference(session,gid,oid))
        from ..ranking.mobile_canvas_operator_view import reference_numbers
        numbers=reference_numbers(v)
        for pid,p in ((pid,p) for pid,p in v['plates'].items() if p['gt_quad']):
            n=numbers[pid]
            if pid==candidate['plate_id']:continue
            manual=next((o for o in s['objects'] if o['object_id']==p['origin_object_id'] and not o.get('parent_id')),None)
            if manual and candidate['role']=='FINAL':
                add(f'Powiąż z ręczną GTR {n}',lambda target=manual['object_id']:resolution.link_manual_to_mt(session,gid,target,oid))
            else:add(f'Powiąż z GTR {n}',lambda target=pid:stacks.associate(session,gid,oid,target))
        if obj in resolution.missing_mt_geometry(s):
            add('Nie można ustalić tożsamości wskazania',lambda:resolution.mark_identity_unresolved(session,gid,oid))
        for other_gid,other_oid,label in getattr(ed.decision_panel,'physical_options',()):
            if other_gid==gid:
                add('Ta sama tablica: '+label,lambda target=other_oid:physical.mark_duplicate(session,gid,oid,target))
            else:
                add('Ta sama tablica: '+label,lambda target=other_oid,call=other_gid:physical.share_across_invocations(session,gid,oid,call,target))
        if candidate['plate_id'] and v['plates'][candidate['plate_id']].get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS':
            add('To odrębna tablica względem innych wywołań',lambda:physical.declare_distinct(session,gid,oid))
    else:
        for c in v['candidates']:
            target=c['object_id']
            if c['role']=='FINAL' and not core._object(s,target)['error']:
                add(f"Ta GTR odpowiada MT {c['mt_detection_index']+1}",lambda target=target:resolution.link_manual_to_mt(session,gid,oid,target))
                add(f"Ta GTR jest inna niż MT {c['mt_detection_index']+1}",lambda target=target:resolution.keep_separate(session,gid,oid,target))
    if menu.index('end') is None:menu.add_command(label='D: nowa tablica · PPM: błędna · Spacja: ramka OK / nieustalona',state='disabled')
    if show:
        ed._canvas_decision_menu=menu
        try:menu.tk_popup(event.x_root,event.y_root)
        finally:menu.grab_release()
    return menu


def present(ed,model):
    if hasattr(ed,'clean_bar'):return model
    if not enhanced(ed) or not ed.selected:return model
    v=view(ed);candidate=next((c for c in v['candidates'] if c['object_id']==ed.selected),None)
    if not candidate:return model
    row=next(r for r in stacks.diagnostics(ed.h.session,[ed.h.current_invocation])['rows'] if r['object_id']==ed.selected)
    p=v['plates'].get(candidate['plate_id']);role=candidate['role']
    choices=members(v,candidate['plate_id']) if p else []
    role_caption={'FINAL':'FINAL','SUPPRESSED':'RAW odrzucona przez NMS','UNKNOWN':'status nieustalony'}[role]
    label=f"MT {candidate['mt_detection_index']+1} · {role_caption}"
    if p:
        n=list(v['plates']).index(candidate['plate_id'])+1;i=choices.index(candidate)+1
        label=f'Tablica GT {n} · MT {n}.{i} z {len(choices)} · {role_caption}'
    reason={'MT_POSE_INVALID':'niepoprawny wynik Pose','MT_POSE_EVIDENCE_UNAVAILABLE':'brak dowodu narożników',
            'GT_UNCONFIRMED':'referencja GT niepotwierdzona','UNASSOCIATED':'brak powiązania z GT','LEGACY_CONFLICT':'sprzeczne referencje historyczne','SCENE_EXCLUDED':'obraz wyłączony z oceny'}.get(row['not_measurable_reason'],row['not_measurable_reason'])
    geom=(f"IoU {row['polygon_iou_pct']:.2f}% · NMCE {row['nmce_pct']:.2f}%" if row['polygon_iou_pct'] is not None else 'IoU/NMCE: N/A · '+reason)
    model.update(focus_id=ed.selected,focus_mode='SELECTED',focus=label,frame=label,
        gt='GT: '+(p['gt_text'] or 'niepodany') if p else 'GT: bez przypisania',
        mz='MZ: '+(candidate['mz_prediction'] or 'brak odczytu') if candidate['mz_ran'] else 'MZ: nie wykonano',
        geometry=geom,polygon_iou_pct=row['polygon_iou_pct'],nmce_pct=row['nmce_pct'],
        stack_caption=label+'\n'+('Prawdziwy numer: '+(p['gt_text'] or 'niepodany')+'\n' if p else '')+
            ('Model odczytał: '+(candidate['mz_prediction'] or 'brak znaków') if candidate['mz_ran'] else 'MZ: nie wykonano')+'\n'+geom)
    model['reference']='Źródłowy Pose MT: '+str(candidate['mt_quad'])+'\nGT ID: '+str(candidate['plate_id'])+'\nMT ID: '+candidate['attempt_id']+'\nPewność źródłowa: '+str(candidate['score'] if candidate['score'] is not None else 'brak danych')
    model['cer']=row['cer']
    model['text']='Diagnostyka MZ ↔ wspólne GT: '+(f"CER {row['cer']*100:.2f}%" if row['cer'] is not None else 'CER nie dotyczy')
    if not p and candidate['association_status']=='UNASSOCIATED_NO_PLATE':
        model['stack_caption']=label+'\n'+('RAW: diagnostycznie poza tablicą; bez końcowego FP.' if role=='SUPPRESSED' else 'Błędne wskazanie — obszar bez tablicy.')
    if role=='SUPPRESSED':
        model.update(state='DIAGNOSTIC_RAW',summary_note='Diagnostyka RAW — poza końcowym TP/FP/FN i mianownikami raportu MT+MZ.',
                     outcome='Odrzucony kandydat RAW; geometrię porównujemy tylko diagnostycznie.')
    elif p and p.get('primary_object_id')!=candidate['object_id']:
        model['summary_note']='Diagnostyka względem wspólnej GT; końcowy duplikat pozostaje FP w dopasowaniu 1:1.'
    return model
