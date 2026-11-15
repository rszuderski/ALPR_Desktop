"""Badge/HUD projection only; source counts and research data stay untouched."""
from __future__ import annotations
import copy
import tkinter as tk

def badge_palette(dark):
    return (dict(bg='#14252D',fg='#F2F5F7',name='#A5D9F2',border='#718E9F') if dark else
            dict(bg='#F5F7FA',fg='#182A38',name='#165D8C',border='#49677C'))

def luminance(color):
    channels=[int(color[i:i+2],16)/255 for i in (1,3,5)]
    linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in channels]
    return sum(a*b for a,b in zip(linear,(.2126,.7152,.0722)))

def contrast(a,b):
    values=sorted((luminance(a),luminance(b)));return (values[1]+.05)/(values[0]+.05)

def dark_theme(ed):
    red,green,blue=ed.c.winfo_rgb(ed.p['surface'])
    return (.2126*red+.7152*green+.0722*blue)/65535<.5

def badge_rows(ed):
    from . import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers
    row=clean.selected(ed)
    if not row:return []
    plate=row['plate'];candidate=row['candidate'];value=layers.model(ed)
    reference=bool(plate and row['gt']);gtr=f"GTR {row['n']}" if reference else 'GTR'
    gtz=(ed.entry_var.get() if clean.entry_matches(ed,row) else plate['gt_text'] if plate else '')
    if plate and plate['text_exclusion_reason']:gtz='nieczytelny'
    elif not gtz:gtz='nie wpisano'
    if row['fp'] and not plate:gtz='nie dotyczy — odrzucone MT'
    mt_name='Model tablic MT';mt='brak powiązanej detekcji';mz='brak powiązanego odczytu'
    if candidate:
        mt_name+=f" {candidate['mt_detection_index']+1}"
        kinds=[l['kind'] for l in (value or {}).get('layers',[])]
        mt=('BBOX + Pose' if 'BBOX' in kinds and 'POSE' in kinds else
            'BBOX · brak poprawnego Pose' if 'BBOX' in kinds else
            'Pose · brak BBOX' if 'POSE' in kinds else 'brak potwierdzonej geometrii')
        mt+=' · '+{'PLATE':'wskazanie zatwierdzone','NOT_PLATE':'odrzucone przez operatora',
                   'UNRESOLVED':'oczekuje na decyzję'}.get(row['decision'],'oczekuje na decyzję')
        status=candidate['mz_status'];prediction=candidate['mz_prediction'] or ''
        if status=='NOT_RUN':mz='nie uruchomiono'
        elif status in ('READ','NO_CHARACTERS') and candidate['mz_ran']:mz=prediction or 'brak odczytu'
        else:mz='wynik niejednoznaczny'+(': '+prediction if prediction else '')
    from .mobile_mz_quality_context import note
    mz+=note(candidate)
    return [(gtr,'niezapisany szkic z '+row['seed']['origin'] if row.get('seed') else 'obrys operatora' if reference else 'brak obrysu'),(mt_name,mt),('GTZ',gtz),('Model znaków MZ',mz)]

def full_text_context(ed):
    """The complete physical/source summary belongs in the existing sidebar."""
    rows=badge_rows(ed)
    from .mobile_canvas_spatial_reference import diagnostic
    from .mobile_canvas_clean_view import selected
    detail=diagnostic(selected(ed))
    return '\n\n'+'\n'.join([*(name+': '+value for name,value in rows),detail]) if rows else ''


def selected_diagnostic(ed,row):
    """One source/revision-bound diagnostic shared by the panel and badges."""
    from ..ranking import mobile_candidate_stacks as data
    key=(ed.cache_key,row['oid'],row.get('pid'))
    if getattr(ed,'_selected_diagnostic_key',None)!=key:
        candidate=row['candidate']
        ed._selected_diagnostic=next((r for r in data.diagnostics(ed.h.session,[ed.h.current_invocation])['rows']
            if r['object_id']==row['oid']),{}) if candidate else {}
        ed._selected_diagnostic_key=key
    return ed._selected_diagnostic


def measurements(ed,row):
    """Read existing results once per review/selection, never per pan frame."""
    from ..ranking.mobile_mt_detection_evaluation import evaluate_unit,reason_text
    from ..ranking import mobile_candidate_stacks as data
    key=(ed.cache_key,row['oid'],row.get('pid'))
    if getattr(ed,'_badge_measurement_key',None)!=key:
        result=evaluate_unit(ed.h.session,ed.h.current_invocation)
        candidate=row['candidate'];aid=candidate['attempt_id'] if candidate else None
        outcome=next((r for r in result['final_outcomes'] if r['attempt_id']==aid),{})
        pair=next((r for r in result['matched_pairs'] if r['attempt_id']==aid),{})
        diagnostic=selected_diagnostic(ed,row)
        measured=result['status']=='MEASURED'
        from .mobile_canvas_review_feedback import reference_comparison
        comparison=reference_comparison(ed,row,result)
        same_bbox_reference=not pair or pair.get('gt_id')==row.get('pid')
        same_pose_reference=diagnostic.get('plate_id')==row.get('pid')
        ed._badge_measurement=dict(bbox_iou=pair.get('bbox_iou') if measured and same_bbox_reference else None,
            reference_comparison=comparison,
            outcome=outcome.get('outcome') if measured and same_bbox_reference else None,
            bbox_reason='SPATIAL_REFERENCE_NOT_SCIENTIFIC_PAIR' if not same_bbox_reference else reason_text(result) if not measured else outcome.get('reason'),
            polygon_iou_pct=diagnostic.get('polygon_iou_pct') if same_pose_reference else None,
            nmce_pct=diagnostic.get('nmce_pct') if same_pose_reference else None,
            pose_reason=diagnostic.get('not_measurable_reason') if same_pose_reference else 'SPATIAL_REFERENCE_NOT_SCIENTIFIC_PAIR')
        ed._badge_measurement_key=key
    return ed._badge_measurement


def percent(value):
    return '—' if value is None else f'{value:.1f}%'.replace('.',',')


def mz_readout(candidate):
    if not candidate:return 'brak powiązanego odczytu'
    if candidate['mz_status']=='NOT_RUN':return 'nie uruchomiono (NOT_RUN)'
    if candidate['mz_status'] in ('READ','NO_CHARACTERS') and candidate['mz_ran']:
        from .mobile_mz_quality_context import note
        return (candidate['mz_prediction'] or 'brak odczytu')+note(candidate)
    return 'wynik niejednoznaczny'+(': '+candidate['mz_prediction'] if candidate['mz_prediction'] else '')


def context_model(ed,layer_view=None,*,row=None,kind=None):
    """Derive identity, text and source anchor from the navigator selection."""
    from . import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers,mobile_canvas_navigator as nav
    row=row if row is not None else clean.selected(ed)
    if not row:return None
    layer_view=layer_view or layers.model(ed)
    kind=kind or ('GTR' if ed.gt_focus_id or not row['candidate'] else layer_view['active'] if layer_view else None)
    q=row['gt'] if kind=='GTR' else nav.quad(layer_view,kind) if layer_view else None
    if ed.drag and ed.drag['oid']==row['oid']:q=ed.drag['quad']
    if not q:return None
    model=dict(kind=kind,oid=row['oid'],plate_id=row['pid'],attempt_id=row['candidate']['attempt_id'] if row['candidate'] else None,
               quad=copy.deepcopy(q),reference_id=row.get('reference_id'))
    a3=getattr(ed.h,'_f5a_a3',None)
    if a3 and a3.active and row['candidate'] and kind!='GTR':
        model.update(title=f"MT · wskazanie {row['candidate']['mt_detection_index']+1}",
                     detail='Wynik: '+(row['candidate']['mz_prediction'] or 'brak odczytu'))
        return model
    if kind=='GTR':
        if row.get('seed'):
            model.update(title='GTR startowy · niezapisany szkic',
                detail='Narożniki z '+row['seed']['origin']+' · przeciągnij, aby ustalić GTR',
                mz_detail='MZ: '+mz_readout(row['candidate']))
            return model
        p=row['plate'];text=ed.entry_var.get() if clean.entry_matches(ed,row) else p['gt_text']
        text='nieczytelny' if p['text_exclusion_reason'] else text or 'nie wpisano'
        model.update(title=f"Obrys operatora · GTR {row['n']}",detail='GTZ: '+text,
                     mz_detail='MZ: '+mz_readout(row['candidate']))
    else:
        values=measurements(ed,row);index=row['candidate']['mt_detection_index']+1
        if kind=='BBOX':
            if layer_view.get('bbox_marker'):
                from .mobile_canvas_bbox_marker import verdict
                from .mobile_bbox_spatial_review import LABELS
                model.update(title=f"MT {index} · BBOX niepoprawny: "+layer_view['bbox_marker']['reason'],
                    detail='Położenie: '+LABELS[verdict(ed,row['candidate'])],spatial_action=True)
                return model
            iou=values['bbox_iou'];outcome=values['outcome']
            comparison=values.get('reference_comparison')
            detail=('BBOX IoU z GTR '+str(comparison['number'])+': '+percent(100*comparison['iou'])
                    if iou is None and comparison else 'BBOX IoU: '+percent(100*iou if iou is not None else None))
            model.update(title=f'Prostokąt modelu · wskazanie {index}',
                detail=detail+' · '+(outcome or 'ocena oczekuje'))
        else:
            model.update(title=f'Narożniki modelu · wskazanie {index}',
                detail='Polygon IoU: '+percent(values['polygon_iou_pct'])+' · NMCE: '+percent(values['nmce_pct']))
    return model


def clear(ed):
    ed.c.delete('active_layer_badge');ed.c.delete('badge_tether');ed.c.delete('stack_count_badge');ed.c.delete('stack_position_badges')
    ed._context_badge=None;ed._badge4_rows=[];ed._badge4_rect=None
    ed._stack_counter=None
    ed.badges={};ed.mt_source_badge_hits={};ed.stack_badges={};ed.bbox_spatial_hits={}


def focus_in_window(ed):
    window=ed.h.window
    # ttk dropdowns are Tcl-created widgets without Python wrappers. Calling
    # focus_get() on their listbox raises KeyError('popdown'). Walk Tcl parents
    # instead; this also keeps menus/inline entries inside their review window.
    try:
        widget=str(window.tk.call('focus','-displayof',window._w))
        while widget:
            if widget==window._w:return True
            widget=str(window.tk.call('winfo','parent',widget))
    except tk.TclError:
        # The focused widget/window may disappear before an idle check runs.
        return False
    return False


def install_focus_cleanup(ed):
    """Invalidate visual hit areas outside this window, retaining selection."""
    window=ed.h.window;canvas=ed.c;pending=[None];bindings=[]
    ed._badge_window_inactive=False
    def focus_in(event):
        ed._badge_window_inactive=False;ed.request_overlay()
    def check():
        pending[0]=None
        if ed.h._closed:return
        if not focus_in_window(ed):
            ed._badge_window_inactive=True;clear(ed)
    def focus_out(event):
        if pending[0] is None:pending[0]=canvas.after_idle(check)
    def destroy(event):
        if event.widget is not canvas:return
        if pending[0] is not None:
            canvas.after_cancel(pending[0]);pending[0]=None
        for sequence,token in bindings:window.unbind(sequence,token)
    for sequence,callback in (('<FocusIn>',focus_in),('<FocusOut>',focus_out)):
        bindings.append((sequence,window.bind(sequence,callback,add='+')))
    canvas.bind('<Destroy>',destroy,add='+')


def fitted(text,font,width):
    if font.measure(text)<=width:return text
    lo,hi=0,len(text)
    while lo<hi:
        mid=(lo+hi+1)//2
        if font.measure(text[:mid]+'…')<=width:lo=mid
        else:hi=mid-1
    return text[:lo]+'…'


def inspection_key(ed):
    scene=ed._read()
    return (id(ed.h.session),ed.h.current_invocation,scene.get('source_binding') if scene else None)


def remember_on_escape(ed):
    """Keep only a display identity; Escape still deselects and ends editing."""
    from .mobile_canvas_clean_view import selected
    from .mobile_canvas_geometry_layers import model
    row=selected(ed)
    if not row:return
    value=model(ed)
    kind='GTR' if ed.gt_focus_id or not row['candidate'] else value['active'] if value else None
    if kind:ed._idle_badge=dict(context=inspection_key(ed),oid=row['oid'],pid=row['pid'],kind=kind,
                              candidate_oid=row['candidate']['object_id'] if row['candidate'] else None)


def display_context(ed,layer_view):
    from .mobile_canvas_clean_view import selected,records,source
    row=selected(ed)
    if row:
        ed._idle_badge=None
        return row,layer_view,None
    saved=getattr(ed,'_idle_badge',None)
    if ed.mode!='select' or not saved or saved['context']!=inspection_key(ed):return None,None,None
    rows=records(ed)
    row=next((r for r in rows if r['pid']==saved['pid'] and r['gt']),None) if saved['kind']=='GTR' else next(
        (r for r in rows if (r['candidate'] and r['candidate']['object_id']==saved['candidate_oid'])
         or any(c['object_id']==saved['candidate_oid'] for c in r['members'])),None)
    if not row:return None,None,None
    candidate=next((c for c in [*row['members'],*([row['candidate']] if row['candidate'] else [])]
                    if c['object_id']==saved['candidate_oid']),None)
    row=dict(row,oid=saved['oid'],candidate=candidate,mt=copy.deepcopy(candidate['mt_quad']) if candidate else None,
             role=candidate['role'] if candidate else 'GT')
    value=None
    if row['candidate']:
        from ..ranking.mobile_mt_geometry_layers import snapshot
        key=(ed.cache_key,row['oid'])
        if getattr(ed,'_idle_layer_key',None)!=key:
            try:
                ed._idle_layer_value=snapshot(ed.h.session,ed.h.current_invocation,row['oid'],scene=ed._read(),state=source(ed))
                from .mobile_canvas_bbox_marker import enrich
                ed._idle_layer_value=enrich(ed.h.session,ed.h.current_invocation,ed._read(),row['candidate'],ed._idle_layer_value)
            except (ValueError,KeyError,TypeError,OSError):ed._idle_layer_value=None
            ed._idle_layer_key=key
        value=ed._idle_layer_value
        if not value or not any(l['kind']==saved['kind'] for l in value['layers']):return None,None,None
        value=dict(value,active=saved['kind'])
    return row,value,saved['kind']


def activate_inspection(ed,oid):
    badge=getattr(ed,'_context_badge',None)
    if not badge or not badge.get('inspection') or badge['oid']!=oid:return True
    from .mobile_canvas_plate_cycle import select
    return select(ed,dict(oid=oid,kind=badge['kind'],pid=badge['plate_id'],quad=badge['quad']))

def paint(ed,layer_view):
    from . import mobile_canvas_clean_view as clean
    from .mobile_canvas_visibility import enabled
    from .mobile_canvas_editor import TAG
    import tkinter.font as tkfont
    ed._context_badge=None;ed._badge4_rows=[];ed._badge4_rect=None;ed._gtz_entry_right=None;ed.bbox_spatial_hits={}
    if ed.mode=='draw':return
    if getattr(ed,'_badge_window_inactive',False):
        if not focus_in_window(ed):return
        ed._badge_window_inactive=False
    row,layer_view,inspection=display_context(ed,layer_view)
    if not row:return
    model=context_model(ed,layer_view,row=row,kind=inspection)
    if not model:return
    kind=model['kind'];q=model['quad'];oid=model['oid']
    from .mobile_canvas_reservation import preview
    editing=kind=='GTR' and preview(ed,row['pid'])
    if not enabled(ed,'ref' if kind=='GTR' else 'inf') and not ed.shift_preview.active and not editing:return
    if ed.drag:ed.clean_geometry_boxes.append(clean.bounds(ed,q))
    if not hasattr(ed,'_badge4_value_font'):ed._badge4_value_font=tkfont.Font(root=ed.c,font=('Segoe UI',9))
    names=ed._badge_font;values=ed._badge4_value_font;line=max(names.metrics('linespace'),values.metrics('linespace'))+5
    editable=kind=='GTR' and row['plate'] and clean.text_editable(ed,row.get('reference_id') or oid)
    action_width=names.measure('Edytuj')+18 if editable else 0
    edit_width=0
    if editable:
        if not hasattr(ed,'_gtz_edit_font'):ed._gtz_edit_font=tkfont.Font(root=ed.c,font=('Segoe UI',11,'bold'))
        text=ed.entry_var.get() if clean.entry_matches(ed,row) else row['plate']['gt_text']
        edit_width=ed._gtz_edit_font.measure(text or 'WW00000WW')+values.measure('GTZ: ')+action_width+28
    width=min(max(names.measure(model['title']),values.measure(model['detail'])+action_width,
                  values.measure(model.get('mz_detail','')),edit_width)+20,430,ed.c.winfo_width()-12)
    if width<80:return
    title=fitted(model['title'],names,width-14);detail=fitted(model['detail'],values,width-action_width-20)
    rect=clean.badge_position(ed,clean.bounds(ed,q),width,3*line+14 if kind=='GTR' or model.get('spatial_action') else 2*line+8)
    if rect is None:return
    ed.clean_label_boxes.append(rect);ed._context_badge=dict(model,rect=rect,displayed=[title,detail],inspection=bool(inspection))
    x,y,r,b=rect;p=badge_palette(dark_theme(ed));tags=(TAG,'active_layer_badge','context_badge','clean_label',oid)
    if kind=='GTR':
        from .mobile_canvas_approval_visuals import color,DARK
        p['border']=color(ed._read(),clean.source(ed),row['pid'])
        ed.c.create_rectangle(*rect,fill=p['bg'],outline=DARK,width=4,tags=(TAG,'active_layer_badge','badge_contrast_underlay',oid))
    ed.c.create_rectangle(*rect,fill=p['bg'],outline=p['border'],width=2 if kind=='GTR' else 1,tags=tags+('context_badge_background',))
    ed.c.create_text(x+7,y+5,text=title,font=names,anchor='nw',fill=p['name'],tags=tags+('context_badge_title',))
    if kind=='GTR':
        top=y+line+2;bottom=top+line+5
        dark=dark_theme(ed);field='#203541' if dark else '#FFFFFF';border='#83D1F0' if dark else '#145A7A'
        if editable:
            ed.c.create_rectangle(x+5,top,r-5,bottom,fill=field,outline=border,width=1.5,tags=tags+('gtz_edit_field',))
            button_left=r-5-action_width
            ed.c.create_rectangle(button_left,top,r-5,bottom,fill='#145A7A',outline=border,tags=tags+('gtz_edit_action',))
            ed.c.create_text((button_left+r-5)/2,(top+bottom)/2,text='Edytuj',font=names,fill='#FFFFFF',tags=tags+('gtz_edit_action',))
            ed.badges[oid]=(x+5,top,r-5,bottom);ed._navigator_badge_prefix=values.measure('GTZ: ')+2
            ed._gtz_entry_right=button_left-2
        ed.c.create_text(x+7,top+3,text=detail,font=values,anchor='nw',fill=p['fg'],tags=tags+('context_badge_detail','clean_label_text','gt_badge'))
        mz=fitted(model['mz_detail'],values,width-14);ed._context_badge['displayed'].append(mz)
        ed.c.create_text(x+7,y+5+2*line+8,text=mz,font=values,anchor='nw',fill=p['fg'],tags=tags+('context_badge_mz',))
        if row['pid']:ed.stack_badges[oid]=(x,y,r,y+4+line,row['pid'])
    else:
        ed.c.create_text(x+7,y+5+line,text=detail,font=values,anchor='nw',fill=p['fg'],tags=tags+('context_badge_detail','clean_label_text'))
        ed.mt_source_badge_hits[oid]=rect
        if model.get('spatial_action'):
            top=y+5+2*line
            ed.c.create_text(x+7,top,text='Oceń położenie ▾ · BBOX IoU: N/A',font=names,anchor='nw',fill=p['name'],tags=tags+('bbox_spatial_action',))
            ed.bbox_spatial_hits[oid]=(x,top,r,b)
    clean.badge_tether(ed,q,rect,oid)
    from .mobile_canvas_stack_counter import paint as paint_counter
    paint_counter(ed,ed._context_badge,layer_view)

def hud_lines(value):
    if not value.get('frame'):raise ValueError('Nie wczytano zweryfikowanej klatki źródłowej.')
    reason=value.get('reason','')
    if any(word in reason for word in ('sprzecz','niejednoznacz','niekomplet','anulowano','błąd')):
        raise ValueError('Błąd źródła wywołań modeli: '+reason)
    # UI counts registered calls. None remains None in the scientific adapter.
    number=lambda key:0 if value.get(key) is None else value[key]
    return [f"Model tablic MT — uruchomień: {number('mt')}/klatka",f"Model znaków MZ — uruchomień: {number('mz')}/klatka"]

class ExecutionHUD(tk.Frame):
    def __init__(self,master,p):
        super().__init__(master,bg=p['surface']);self.lines=[];self.names=[];self.values=[]
        red,green,blue=self.winfo_rgb(p['surface']);dark=(.2126*red+.7152*green+.0722*blue)/65535<.5
        color='#A5D9F2' if dark else '#165D8C'
        for i,name in enumerate(('Model tablic MT','Model znaków MZ')):
            label=tk.Label(self,text=name,bg=p['surface'],fg=color,font=('Segoe UI',9,'bold'),anchor='w',padx=0,pady=0)
            label.grid(row=i,column=0,sticky='w');self.names.append(label)
            value=tk.Label(self,text='',bg=p['surface'],fg=p['fg'],font=('Segoe UI',9),anchor='w',padx=0,pady=0)
            value.grid(row=i,column=1,sticky='w');self.values.append(value)
    def display(self,lines):
        self.lines=lines
        for label,text in zip(self.values,lines):label.configure(text=' — '+text.split(' — ',1)[1])
        self.grid()
    def unavailable(self):self.lines=[];self.grid_remove()
    def cget(self,key):return '\n'.join(self.lines) if key=='text' else super().cget(key)

def update_hud(ed):
    from .mobile_canvas_visibility import gt_only
    if gt_only(ed):ed.source_hud.unavailable();return
    a3=getattr(ed.h,'_f5a_a3',None)
    if a3 and a3.active:
        ed.source_hud.unavailable();return  # A3 uses photo/MT progress, not legacy frame counters.
    from ..ranking.mobile_canvas_source_hud import frame_execution
    h=ed.h
    if h._image is None or not h._image_entry:ed.source_hud.unavailable();return
    try:
        stamp=h.session.path.stat() if h.session else None
        key=(id(h.session),h.current_invocation,getattr(stamp,'st_size',None),getattr(stamp,'st_mtime_ns',None),h._image_entry,tuple(h._image_source_size or ()))
        if getattr(ed,'_source_hud_key',None)!=key:
            ed._source_hud_value=frame_execution(h.session,h.current_invocation,image_entry=h._image_entry,image_size=h._image_source_size)
            ed._source_hud_key=key
        if ed._source_hud_value.get('mz') is None:
            gids=ed._source_hud_value.get('mt_invocation_ids') or [h.current_invocation]
            records=[r for gid in gids if gid in h.session.mt_invocations for r in h.session.mt_invocations[gid].records]
            if any(str(r.get('mz_executed')).lower() in ('true','1') or r.get('mz_execution_id') for r in records):
                raise ValueError('Niekompletny rejestr wykonanych wywołań MZ — nie można podać ich liczby.')
        ed.source_hud.display(hud_lines(ed._source_hud_value));ed._source_hud_error=None
    except (ValueError,KeyError,TypeError,OSError) as exc:
        ed.source_hud.unavailable();message='Nie można odczytać źródła wywołań modeli: '+str(exc)
        if getattr(ed,'_source_hud_error',None)!=message:ed.message(message)
        ed._source_hud_error=message
