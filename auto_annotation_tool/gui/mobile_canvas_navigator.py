"""One descriptive, transient navigator. No review/metric writes."""
from __future__ import annotations
import math,tkinter as tk
from . import mobile_canvas_geometry_layers as layers

TITLE='Wskazanie modelu lokalizacji tablic'

def finals(ed):
    from .mobile_canvas_clean_view import source
    v=source(ed)
    return [c for c in (v or {}).get('candidates',[]) if c['role']=='FINAL']

def ensure_selection(ed):
    key=(id(ed.h.session),ed.h.current_invocation)
    if getattr(ed,'_navigator_initialized',None)==key or not ed._read():return
    ed._navigator_initialized=key
    choices=finals(ed)
    if choices and not ed.selected and ed.mode!='draw':
        ed.selected=choices[0]['object_id'];ed.mode='edit'
        value=layers.model(ed)
        if value:
            kinds=[layer['kind'] for layer in value['layers']]
            ed._geometry_layer_state=(value['identity'],'POSE' if 'POSE' in kinds else kinds[0])

def scope(ed,widget):
    if widget is ed.c:return True
    nav=getattr(ed,'navigator',None)
    while widget is not None:
        if nav and widget is nav.controls:return True
        widget=getattr(widget,'master',None)
    return False

def select_layer(ed,kind,oid=None):
    if ed.entry is not None or ed.drag or ed.mode=='draw' or ed.points or ed.busy:return False
    if oid and oid!=ed.selected:
        if not ed._choose_plate(oid,source=kind!='GTR'):return False
    value=layers.model(ed)
    if not value or kind not in [l['kind'] for l in value['layers']]:return False
    ed._geometry_layer_state=(value['identity'],kind);ed.gt_focus_id='';ed.mode='edit'
    ed._bad_mt_selection=(id(ed.h.session),ed.h.current_invocation,ed.selected) if kind!='GTR' else None
    ed.c.focus_set();ed.sync();ed.request_overlay()
    if kind=='GTR':ed.corner_pulse.trigger()
    ed.message(label(ed,kind)+' · Z/X: poprzednia/następna warstwa · ↑/↓: wskazanie modelu')
    return True

def cycle_layer(ed,back=False):
    value=layers.model(ed)
    if not value:return None
    kinds=[l['kind'] for l in value['layers']]
    if len(kinds)>1:select_layer(ed,kinds[(value['index']+(-1 if back else 1))%len(kinds)])
    return 'break'

def select_result(ed,index):
    if ed.entry is not None or ed.drag or ed.mode=='draw' or ed.points or ed.busy:return False
    choices=finals(ed)
    if not choices:return False
    prior=layers.model(ed);kind=prior['active'] if prior else 'BBOX';c=choices[index%len(choices)]
    if c['association_status']=='UNASSOCIATED_NO_PLATE':ed.show_fp_var.set(True)
    if not ed._choose_plate(c['object_id'],source=True):return False
    value=layers.model(ed)
    if not value:return False
    available=[l['kind'] for l in value['layers']]
    return select_layer(ed,kind if kind in available else available[0])

def cycle_result(ed,back=False):
    choices=finals(ed)
    if len(choices)<2:return 'break'
    ids=[c['object_id'] for c in choices];i=ids.index(ed.selected) if ed.selected in ids else 0
    select_result(ed,i+(-1 if back else 1));return 'break'

def label(ed,kind):
    from .mobile_canvas_clean_view import selected
    row=selected(ed)
    if kind=='BBOX':
        value=layers.model(ed)
        if value and value.get('bbox_marker'):return 'BBOX (niepoprawny — '+('punkt' if value['bbox_marker']['validity']=='INVALID_POINT' else 'odcinek')+')'
    if kind=='GTR' and row and row.get('seed'):return 'GTR startowy (niezapisany szkic)'
    return {'BBOX':'Prostokąt modelu (BBOX)','POSE':'Narożniki modelu (Pose)',
            'GTR':f"Obrys operatora (GTR {row['n'] if row else '—'})"}.get(kind,'Wybierz warstwę')

def quad(value,kind=None):
    layer=next((l for l in (value or {}).get('layers',[]) if l['kind']==(kind or value['active'])),None)
    if not layer:return None
    if layer['kind']=='BBOX':
        l,t,r,b=layer['bbox'];return [(l,t),(r,t),(r,b),(l,b)]
    return layer['quad']

def show_geometry(ed,points,*,only_if_needed=False):
    """Fit source geometry using the counter inspector's existing viewport rules."""
    width,height=ed.c.winfo_width(),ed.c.winfo_height()
    visible=[ed.to_canvas(p) for p in points]
    if only_if_needed and all(12<=x<=width-12 and 12<=y<=height-12 for x,y in visible):return False
    sw,sh=ed._read()['image_size'];sx,sy=ed.h._image.width/sw,ed.h._image.height/sh
    xs=[p[0]*sx for p in points];ys=[p[1]*sy for p in points]
    scale=min(width*.65/max(1,max(xs)-min(xs)),height*.45/max(1,max(ys)-min(ys)))
    ed.h._view_zoom=max(1,min(256,scale/max(.0001,ed.h._view_fit_scale)))
    ed.h._view_center=((min(xs)+max(xs))/2,(min(ys)+max(ys))/2)
    ed.h._paint_image()
    return True

def geometries(ed):
    from .mobile_canvas_visibility import contours
    return [(item['oid'],item['kind'],item['quad']) for item in contours(ed)]

def selection_hit(ed,x,y):
    """S edits a visible GTR before considering any overlapping MT layer."""
    from .mobile_canvas_editor import quad_contains_point
    from ..ranking.mobile_manual_frame_guard import edge_distance
    from .mobile_canvas_bbox_marker import hit as marker_hit
    marker=marker_hit(ed,x,y)
    if marker:return marker
    point=ed.to_source(x,y);hits=[]
    for oid,kind,q in geometries(ed):
        if kind!='GTR':continue
        distance=edge_distance((x,y),[ed.to_canvas(p) for p in q])
        if distance<=7 or quad_contains_point(q,*point):hits.append((distance,oid))
    if hits:
        # MT edge distance and active source inspection do not own S here.
        hits.sort();chosen=hits[0]
        if layers.editable(ed):
            chosen=next((h for h in hits if h[1]==ed.selected and h[0]<=hits[0][0]+2),chosen)
        return chosen[1],'layer:GTR',None
    return hit(ed,x,y,prefer_stack=True)


def hit(ed,x,y,*,prefer_stack=False):
    from .mobile_canvas_clean_view import selected
    from .mobile_canvas_editor import quad_contains_point
    row=selected(ed);value=layers.model(ed)
    from .mobile_canvas_visibility import enabled
    from .mobile_canvas_reservation import preview
    if row and (enabled(ed,'ref') or preview(ed,row['pid'])) and layers.editable(ed) and ed.mode=='edit':
        for i,p in enumerate(row['gt'] or []):
            if math.dist((x,y),ed.to_canvas(p))<=14:return row['oid'],'corner',i
    for oid,(l,t,r,b) in ed.badges.items():
        if l<=x<=r and t<=y<=b:return oid,'badge',None
    for oid,(l,t,r,b) in getattr(ed,'reference_identity_hits',{}).items():
        if l<=x<=r and t<=y<=b:return oid,'layer:GTR',None
    for oid,(l,t,r,b) in getattr(ed,'bbox_spatial_hits',{}).items():
        if l<=x<=r and t<=y<=b:return oid,'bbox_spatial_menu',None
    for oid,(l,t,r,b) in ed.mt_source_badge_hits.items():
        if l<=x<=r and t<=y<=b:
            badge=getattr(ed,'_context_badge',None)
            kind=value['active'] if value else badge['kind'] if badge and badge['oid']==oid else None
            if kind:return oid,'layer:'+kind,None
    badge=getattr(ed,'_context_badge',None)
    if badge and badge['kind']=='GTR':
        l,t,r,b=badge['rect']
        if l<=x<=r and t<=y<=b:return badge['oid'],'layer:GTR',None
    from .mobile_canvas_bbox_marker import hit as marker_hit
    marker=marker_hit(ed,x,y)
    if marker:return marker
    hits=[];point=ed.to_source(x,y);shapes=geometries(ed);models={}
    if prefer_stack:
        from .mobile_canvas_clean_view import source
        from ..ranking.mobile_mt_geometry_layers import snapshot
        # All source layers of a visible stack share the same selection area.
        # Explicit links alone determine its GTR; overlap never creates a link.
        state=source(ed)
        for oid,kind,_ in tuple(shapes):
            if kind=='GTR' or oid in models:continue
            try:models[oid]=snapshot(ed.h.session,ed.h.current_invocation,oid,scene=ed._read(),state=state)
            except (ValueError,KeyError,TypeError,OSError):continue
            # A visible, unsaved operator seed belongs to this exact source
            # object too, even though it is absent from the scientific GT.
            seed=next((q for key,role,q in shapes if key==oid and role=='GTR'),None)
            if seed and not any(l['kind']=='GTR' for l in models[oid]['layers']):
                models[oid]['layers'].append(dict(kind='GTR',quad=seed,provisional=True))
            for layer in models[oid]['layers']:
                if layer['kind'] in ('BBOX','POSE'):shapes.append((oid,layer['kind'],quad(models[oid],layer['kind'])))
    from ..ranking.mobile_manual_frame_guard import edge_distance
    # Screen-space edge distance distinguishes adjacent source/reference lines.
    for oid,kind,q in shapes:
        pts=[ed.to_canvas(p) for p in q];distance=edge_distance((x,y),pts)
        if distance<=7 or quad_contains_point(q,*point):hits.append((distance,oid,kind))
    if prefer_stack:
        # Several contours of one stack are one hit, even after deselection.
        grouped={}
        for h in sorted(hits):grouped.setdefault(h[1],h)
        hits=list(grouped.values())
    edges=sorted([h for h in hits if h[0]<=7])
    active_kind=value['active'] if value else 'GTR'
    selected_hit=next((h for h in hits if h[1]==ed.selected and h[2]==active_kind),
                      next((h for h in hits if h[1]==ed.selected),None))
    best=edges[0] if edges and (len(edges)==1 or edges[1][0]-edges[0][0]>2) else selected_hit or (hits[0] if len(hits)==1 else None)
    if not best:return None
    _,oid,kind=best
    if prefer_stack and oid in models:
        kinds=[layer['kind'] for layer in models[oid]['layers']]
        kind=next(k for k in ('GTR','POSE','BBOX') if k in kinds)
    return oid,'layer:'+kind,None

def right_hit(ed,x,y):
    from ..ranking import mobile_canvas_review as core
    if getattr(getattr(ed,'shift_preview',None),'active',False):
        for label in getattr(ed,'shift_preview_labels',[]):
            l,t,r,b=label['rect']
            if not (l<=x<=r and t<=y<=b):continue
            frame=label['frame']
            if frame['kind']=='GTR' and not frame.get('seed'):raise ValueError(core.GTR_BAD_MESSAGE)
            if frame.get('role') not in ('FINAL','SEED') or frame['gid']!=ed.h.current_invocation:
                raise ValueError('RAW lub inne wywołanie: tylko podgląd. Wybierz źródłowy MT FINAL w nawigatorze.')
            core.require_source_final(ed.h.session,ed.h.current_invocation,frame['oid'])
            return frame['oid']
    from .mobile_canvas_reference_seed import right_target
    seed_source=right_target(ed,x,y)
    if seed_source:return seed_source
    if ed.mode=='draw':
        # Drawing intentionally hides contours; a saved GTR remains protected
        # from PPM/BAD and must not turn the click into draft cancellation.
        from ..ranking.mobile_manual_frame_guard import find_point_collision
        if find_point_collision(ed._read(),ed.to_source(x,y)):
            raise ValueError(core.GTR_BAD_MESSAGE)
        return None
    target=hit(ed,x,y)
    if not target:return None
    oid,kind,_=target
    if kind in ('badge','corner','layer:GTR'):raise ValueError(core.GTR_BAD_MESSAGE)
    if ed.mode=='draw' or ed.gt_focus_id:return None
    # With REF and INF visible together, automatic initial selection is not
    # an explicit choice of the source underneath an operator reference.
    source_badge=any(l<=x<=r and t<=y<=b for l,t,r,b in ed.mt_source_badge_hits.values())
    explicit=getattr(ed,'_bad_mt_selection',None)==(id(ed.h.session),ed.h.current_invocation,oid)
    if not source_badge and not explicit:
        from .mobile_canvas_editor import quad_contains_point
        from ..ranking.mobile_manual_frame_guard import edge_distance
        point=ed.to_source(x,y)
        if any(role=='GTR' and (quad_contains_point(q,*point) or edge_distance((x,y),[ed.to_canvas(p) for p in q])<=7)
               for _,role,q in geometries(ed)):
            raise ValueError(core.GTR_BAD_MESSAGE)
    core.require_source_final(ed.h.session,ed.h.current_invocation,oid)
    return oid

def source_context(ed,value):
    from .mobile_canvas_clean_view import selected
    from ..ranking.mobile_mt_object_evaluation import evaluate_unit
    from ..ranking import mobile_candidate_stacks as data
    row=selected(ed);c=row['candidate'];kind=value['active'];det=evaluate_unit(ed.h.session,ed.h.current_invocation)
    outcome=next((r['outcome'] for r in det['final_outcomes'] if r['attempt_id']==c['attempt_id']),'—')
    decision={'PLATE':'poprawna','NOT_PLATE':'błędna','UNRESOLVED':'oczekuje'}.get(row['decision'],'oczekuje')
    text='Źródło: model lokalizacji tablic\nDecyzja operatora: '+decision+'\nWynik MT v2: '+outcome
    from .mobile_canvas_info_badge import measurements,percent
    values=measurements(ed,row)
    if kind=='BBOX':
        if value.get('bbox_marker'):
            from .mobile_canvas_bbox_marker import verdict
            from .mobile_bbox_spatial_review import LABELS
            marker=value['bbox_marker']
            text+='\nBBOX niepoprawny: '+marker['reason']+'\nBBOX IoU: N/A\nPołożenie: '+LABELS[verdict(ed,c)]
            text+='\nOcena położenia BBOX jest niezależna od Pose i decyzji MT.'
            return label(ed,kind).upper(),text
        text+='\nPomocnicza ocena BBOX v1: '+(values['outcome'] or 'ocena oczekuje')
        text+='\nBBOX IoU: '+percent(100*values['bbox_iou'] if values['bbox_iou'] is not None else None)+'\nProstokąt źródłowy · tylko inspekcja'
        comparison=values.get('reference_comparison')
        if comparison and values['bbox_iou'] is None:
            value_text=f"{100*comparison['iou']:.2f}%".replace('.',',')
            text=text.replace('BBOX IoU: —',f"BBOX IoU z GTR {comparison['number']}: {value_text}")
            text+=f"\nPróg dopasowania BBOX: {100*comparison['threshold']:g}%"
            if values['bbox_reason']=='LOCALIZATION_CRITERION_NOT_MET':text+=' · poniżej progu'
        if not values['outcome']:text+='\n'+str(values['bbox_reason'] or 'Brak ukończonej oceny')
    else:
        diagnostic=values
        f=lambda v:'—' if v is None else f'{v:.1f}%'.replace('.',',')
        text+='\nPolygon IoU: '+f(diagnostic.get('polygon_iou_pct'))+'\nNMCE: '+f(diagnostic.get('nmce_pct'))+'\nNarożniki źródłowe · tylko inspekcja'
        if values['pose_reason']:
            reason=values['pose_reason']
            text+='\n'+('Porównanie oczekuje na zatwierdzenie ramek' if reason=='UNASSOCIATED' and value['gtr_count'] else
                        {'UNASSOCIATED':'Brak powiązanej referencji GTR','GT_UNCONFIRMED':'Referencja GTR niepotwierdzona',
                         'SPATIAL_REFERENCE_NOT_SCIENTIFIC_PAIR':'Relacja przestrzenna; brak zatwierdzonej pary metrycznej dla tej GTR',
                         'SCENE_EXCLUDED':'Obraz wyłączony z oceny','LEGACY_CONFLICT':'Konflikt referencji GTR'}.get(reason,'Nie można porównać geometrii z referencją'))
    text+='\nReferencja operatora: '+(f"GTR {row['n']}" if value['gtr_count'] else 'brak powiązania')
    if row['plate'] and row['plate']['geometry_state']=='operator_removed' and not row['gt']:
        text=text.replace('Referencja operatora: brak powiązania','Obrys operatora: usunięty; ID i GTZ zachowane')
        text+='\nD: narysuj nowy obrys · Ctrl+Z: przywróć usunięty'
    from .mobile_canvas_info_badge import mz_readout
    text+='\nMZ: '+mz_readout(c)
    return label(ed,kind).upper(),text

class Navigator:
    def __init__(self,ed):
        self.ed=ed;p=ed.p;bar=ed.clean_bar
        ed.clean_caption.configure(text=TITLE,font=('Segoe UI',9,'bold'),wraplength=300)
        ed.clean_caption.grid(row=0,column=0,columnspan=2,sticky='w')
        ed.clean_gt_slot.grid_configure(row=2,column=0,columnspan=3);ed.clean_gt_slot.grid_remove()
        ed.source_hud.grid_configure(row=4,column=0,columnspan=3)
        self.controls=tk.Frame(bar,bg=p['surface']);self.controls.grid(row=1,column=0,columnspan=3,sticky='ew',padx=4,pady=(1,2))
        self.controls.columnconfigure(3,weight=1)
        opts=dict(bg=p['surface'],fg=p['fg'],activebackground='#145A7A',activeforeground='white',font=('Segoe UI',9),bd=1,relief='solid',takefocus=1)
        self.previous=tk.Button(self.controls,text='◀',command=lambda:cycle_result(ed,True),**opts)
        self.result=tk.Button(self.controls,bg=p['surface'],fg=p['fg'],disabledforeground=p['fg'],font=('Segoe UI',9),
            relief='flat',bd=0,padx=2,command=lambda:select_result(ed,0));self.result.grid(row=0,column=1,padx=3)
        self.following=tk.Button(self.controls,text='▶',command=lambda:cycle_result(ed),**opts)
        self.layer=tk.Menubutton(self.controls,text='Wybierz warstwę',anchor='w',width=1,**opts)
        self.layer.grid(row=0,column=3,sticky='ew',padx=(5,0));self.layer.bind('<Configure>',self.resize)
        from .mobile_canvas_clean_view import menu_colors
        self.menu=tk.Menu(self.layer,tearoff=False,**menu_colors(p));self.layer.configure(menu=self.menu)
        self.selection=tk.StringVar(master=bar);self.signature=None
        self.hint=tk.Label(bar,bg=p['surface'],fg=p['muted'],font=('Segoe UI',8),anchor='nw',height=2,wraplength=400)
        self.hint.grid(row=2,column=0,columnspan=3,sticky='ew',padx=5)
        bar.rowconfigure(2,minsize=max(28,self.hint.winfo_reqheight()))
        bar.bind('<Configure>',self.resize,add='+')

    def resize(self,event=None):
        self.layer.configure(wraplength=max(90,self.layer.winfo_width()-22))
        self.ed.clean_caption.configure(wraplength=max(160,self.ed.clean_bar.winfo_width()-85))
        self.hint.configure(wraplength=max(150,self.ed.clean_bar.winfo_width()-12))

    def update(self):
        from .mobile_canvas_visibility import gt_only
        if gt_only(self.ed):
            self.controls.grid_remove();self.hint.configure(text='Wyniki modeli ukryte · najpierw ręczne GT');self.hint.grid()
            self.ed.clean_caption.configure(text='Ręczne GT · D: obrys · Enter: tekst · Spacja: zatwierdź')
            self.ed.clean_menu.entryconfigure(self.ed.clean_mz_crop_index,state='disabled')
            return
        self.controls.grid()
        ed=self.ed;value=layers.model(ed);choices=finals(ed);ids=[c['object_id'] for c in choices]
        from .mobile_canvas_clean_view import selected,source
        row=selected(ed);manual=bool(row and not row['candidate'])
        unknown=sum(c['role']=='UNKNOWN' for c in (source(ed) or {}).get('candidates',[]))
        absent=f'Nieudowodnione role MT: {unknown}' if unknown else 'Brak wskazań modelu'
        ed.clean_caption.configure(text=TITLE)
        if len(ids)>1:
            self.previous.grid(row=0,column=0);self.following.grid(row=0,column=2)
        else:self.previous.grid_remove();self.following.grid_remove()
        self.result.configure(text=f'Wynik {ids.index(ed.selected)+1} z {len(ids)}' if ed.selected in ids else 'Wybór ręczny' if manual else absent if not ids else 'Wybierz wskazanie')
        self.result.configure(state='normal' if ids and ed.selected not in ids and ed.mode!='draw' else 'disabled')
        kinds=[l['kind'] for l in value['layers']] if value else []
        signature=(tuple(kinds),value['identity'] if value else None,row['n'] if row else None,bool(row and row.get('seed')))
        if signature!=self.signature:
            self.signature=signature;self.menu.delete(0,'end')
            for kind in ('BBOX','POSE','GTR'):
                self.menu.add_radiobutton(label=label(ed,kind),value=kind,variable=self.selection,
                    state='normal' if kind in kinds else 'disabled',command=lambda kind=kind:select_layer(ed,kind))
        self.selection.set(value['active'] if value else '')
        self.layer.configure(text=(label(ed,value['active']) if value else label(ed,'GTR') if manual else 'Brak dostępnych warstw')+' ▾',
            state='normal' if kinds and not ed.drag and ed.mode!='draw' else 'disabled',bg='#145A7A' if value else ed.p['surface'],fg='white' if value else ed.p['fg'])
        hints=[]
        if value:
            hints.append(('Szkic GTR · niezapisany · D: przesuń narożniki' if row.get('seed') and value['active']=='GTR' else
                          f"GTR {row['n']} · obrys operatora" if value['active']=='GTR' else
                          f"{value['active'].title()} MT {row['candidate']['mt_detection_index']+1} · zapis źródłowy"))
            pose=quad(value,'POSE')
            if pose and row['gt']==pose and not row.get('seed'):
                hints.append(f"Pose MT i GTR {row['n']}: te same narożniki")
            if value.get('bbox_marker'):hints.append('BBOX niepoprawny: '+value['bbox_marker']['reason'])
            elif 'BBOX' not in kinds:hints.append(value.get('bbox_diagnostic',{}).get('reason','Brak wiarygodnego prostokąta modelu'))
            if 'POSE' not in kinds:
                status=(row['candidate'].get('source_pose_status') or '') if row and row['candidate'] else ''
                hints.append('Narożniki modelu: niepoprawne (INVALID_QUAD)' if status=='INVALID_QUAD' else 'Narożniki modelu: brak potwierdzonych danych')
            if 'GTR' not in kinds:
                hints.append('Obrys operatora: brak referencji · GTR usunięta' if row['plate'] and row['plate']['geometry_state']=='operator_removed'
                             else 'Brak powiązanej GTR dla tego MT')
        elif manual:hints.append(f"GTR {row['n']} · niezależny obrys operatora · brak powiązania MT/MZ")
        elif any(c['role']=='UNKNOWN' for c in (source(ed) or {}).get('candidates',[])):hints.append('Nie potwierdzono pochodzenia FINAL — brak dostępnych warstw')
        if ed.mode=='draw':
            target=getattr(ed,'draw_target',None)
            name=f"Poprawiasz GTR {target['number']}" if target and target['kind']=='replace' else 'Rysowanie obrysu operatora'
            hints=[name+f' · narożnik {min(4,len(ed.points)+1)}/4 · Esc: anuluj']
            self.layer.configure(text='Rysowanie obrysu operatora')
        self.hint.configure(text=' · '.join(hints))
        self.hint.grid()
        candidate=row['candidate'] if row else None
        session=ed.h.session
        crop_available=bool(candidate and candidate['mz_ran'] and session and any(
            session.samples[sid].get('attempt_id')==candidate['attempt_id'] and
            session.samples[sid].get('image_entry') in session.entry_names
            for sid in session.sample_ids_by_attempt.get(candidate['attempt_id'],()) if sid in session.samples))
        ed.clean_menu.entryconfigure(ed.clean_mz_crop_index,state='normal' if crop_available and ed.mode!='draw' and not ed.drag else 'disabled')
