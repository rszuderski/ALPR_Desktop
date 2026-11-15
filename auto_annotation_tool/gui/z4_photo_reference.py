"""Two-column source-photograph host for the existing Z4 CanvasReviewEditor.

No mobile report, inference widget, secondary Canvas controller or scoring path.
"""
from pathlib import Path
import math
import traceback
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font

from ..ranking.photo_reference import (Material, PhotoReference, complete, display_text, text_kind,
    geometry_status, localization_outline, GEOMETRY_REASONS)
from .mobile_canvas_editor import CanvasReviewEditor, TAG
from .mobile_canvas_ui import palette
from .z4_mobile_sample_review import MobileSampleReviewWindow
from ..config import logger


class PhotographCanvasSource:
    def __init__(self, host):
        self.h = host
        self.error = ''
        self.entry_after_draw = None
        self.pending_action = None
        self.location_draft = None
        self.exact_target = None
        self.menu_open = False

    display_text = staticmethod(display_text)
    localization_outline = staticmethod(localization_outline)

    @property
    def data(self):return self.h.session

    @property
    def key(self):return self.h.current_invocation

    def install(self, ed, viewer):
        self.ed = ed
        ed.p = palette(self.h.window, self.h.bg)
        from .mobile_canvas_collision_warning import CollisionWarning
        from .mobile_canvas_corner_pulse import CornerPulse
        from .mobile_canvas_shift_preview import ShiftPreview
        ed.collision_warning=CollisionWarning(ed);ed.corner_pulse=CornerPulse(ed);ed.shift_preview=ShiftPreview(ed)
        ed._badge_font=font.Font(root=self.h.window,family='Segoe UI',size=9,weight='bold')
        ed.tree_targets={};ed.linked_tree_rows={};ed.pending_tree_target=None;ed.tree_refreshing=False
        ed.gt_focus_id='';ed.model_location_target='';ed.history_selected=''
        ed.show_roi_overlay=False;ed.read_error='';ed._geometry_layer_state=None
        ed.bindings()

    def scene(self):
        return self.data.scene(self.key) if self.key else None

    def sync(self):
        ed=self.ed
        ed.cache_key=(self.key,self.data.data['revision'])
        if ed.skey!=self.key:
            ed.skey=self.key;ed.selected=ed.hover='';ed.mode='select';ed.points=[];ed.drag=None
        if ed.selected and not any(o['object_id']==ed.selected for o in self.scene()['objects']):
            ed.selected='';ed.mode='select'
        self.refresh_list()
        if self.entry_after_draw and ed.mode=='edit' and not ed.points:
            oid,self.entry_after_draw=self.entry_after_draw,None
            ed.start_entry(oid)

    def refresh_list(self):
        tree=self.h.subject_tree
        editing=self.editing()
        for i,key in enumerate(self.data.material.order,1):
            row=self.data.material.images[key];name=Path(row['relative_path']).stem
            limit=self.h.subject_tree.column('photo','width')-38
            if self.h.list_font.measure(name)>limit:
                while name and self.h.list_font.measure(name+'…')>limit:name=name[:-1]
                name+='…'
            status='W trakcie' if key==self.key and editing else self.data.status(key)
            short='Uzupełnij' if status=='Wymaga uzupełnienia' else status
            values=(f'{i:02d}  {name}',short)
            tag='ready' if status=='Gotowe' else 'pending'
            if tree.exists(key):tree.item(key,values=values,tags=(tag,))
            else:tree.insert('','end',iid=key,values=values,tags=(tag,))
        self.h.progress.set(f"Gotowe: {sum(complete(self.data.state(k)) and not (k==self.key and editing) for k in self.data.material.order)}/{len(self.data.material.order)}")
        if self.key:
            self.h.current_status.set('W trakcie' if editing else self.data.status(self.key))
            self.h.empty_button.configure(text='Brak tablic ✓' if self.scene()['presence']=='NO_PLATES' else 'Brak tablic')

    def caption(self):
        ed=self.ed
        key=(self.key,self.editing(),self.data.data['revision'])
        if getattr(self,'_display_key',None)!=key:
            self._display_key=key;self.refresh_list()
        draft=self.location_draft
        self.h.image_help.configure(text=(('Przeciągnij po widocznym fragmencie tablicy' if draft['kind']=='APPROXIMATE_REGION' else 'Wskaż miejsce na fotografii')+' · Esc: anuluj' if draft else
            f'Wskaż narożnik {len(ed.points)+1} z 4 · Esc: anuluj' if ed.mode=='draw'
            else 'D: obrys · PPM: numer i geometria · R: usuń · Ctrl+Z/Y: cofnij/ponów · rolka: zoom · przeciągnij tło: przesuń'))

    def editing(self):
        ed=self.ed
        return bool(ed.mode=='draw' or ed.points or ed.drag or self.pending_action or self.location_draft
            or (ed.entry is not None and ed.entry_var.get()!=ed.entry_original))

    def available(self):
        ok=bool(self.key and self.h._image is not None and not self.h._saving)
        if not ok:self.h.status_var.set('Nie można edytować fotografii. Sprawdź komunikat odczytu.')
        return ok

    def policy(self, oid):
        ok=any(o['object_id']==oid for o in self.scene()['objects'])
        return dict(gt_editable=ok,gt_reason='' if ok else 'Wybierz tablicę.',gt_note='')

    def report_error(self, exc, operation):
        """Keep diagnostics out of the workspace UI and out of the GT store."""
        context=f'Z4 GT {operation}: key={self.key!r}, destination={self.data.path}'
        logger.exception(context)
        try:
            from alpr_runtime import user_data_root
            path=user_data_root()/'diagnostics/z4_gt_errors.log'
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',encoding='utf-8') as out:
                out.write(f'\n{datetime.now(timezone.utc).isoformat()} {context}\n')
                traceback.print_exception(type(exc),exc,exc.__traceback__,file=out)
        except OSError:
            logger.exception('Nie udało się zapisać pliku diagnostyki Z4 GT')
        if isinstance(exc,PermissionError):reason='Brak dostępu do pliku lub plik jest zajęty.'
        elif isinstance(exc,FileNotFoundError):reason='Nie znaleziono wymaganego pliku lub katalogu.'
        elif isinstance(exc,KeyError):reason='Nie znaleziono wybranej sceny lub tablicy.'
        elif isinstance(exc,TypeError):reason='Nie można zapisać danych oceny w wymaganym formacie.'
        elif isinstance(exc,OSError):reason='Błąd odczytu lub zapisu pliku. Sprawdź dostęp i wolne miejsce.'
        elif str(exc).startswith(('Trwa zapis weryfikacji','Weryfikacja została zmieniona','Uzupełnij GT')):reason=str(exc)
        else:reason='Nie można wykonać operacji: dane są niezgodne lub zmienione.'
        return reason

    def commit(self, action):
        try:
            value=action()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.error=self.report_error(exc,'zapis')
            self.pending_action=action
            self.h.status_var.set('NIE ZAPISANO: '+self.error)
            self.h.status_label.configure(fg='#b02020')
            self.h.retry_button.grid(row=4,column=0,sticky='e',padx=10)
            return False
        self.error=''
        self.pending_action=None
        self.h.retry_button.grid_remove()
        self.h.status_var.set('Zapisano')
        self.h.status_label.configure(fg=self.h.muted)
        self.ed.cache_key=None
        self.sync();self.ed.request_overlay()
        return value if value is not None else True

    def retry(self):
        if self.location_draft:return self.location_release()
        if self.ed.entry is not None:
            return self.ed.finish_entry()
        if self.ed.drag:
            return self.ed.release()
        if self.pending_action is not None:
            value=self.commit(self.pending_action)
            if isinstance(value,str) and value.startswith('PGT:'):
                self.ed.points=[];self.ed.pointer=None;self.ed.draw_target=None
                self.ed.selected=value;self.ed.mode='edit';self.entry_after_draw=value
                self.sync()
            return value

    def draw_target(self):
        if self.exact_target:
            oid,self.exact_target=self.exact_target,None
            return dict(kind='photo_complete',oid=oid,plate_id=oid)
        obj=next((o for o in self.scene()['objects'] if o['object_id']==self.ed.selected and not localization_outline(o)),None)
        return dict(kind='photo_complete',oid=obj['object_id'],plate_id=obj['object_id']) if obj else None

    def outline(self, quad):
        target=self.ed.draw_target
        oid=(self.data.edit(self.key,target['oid'],quad=quad) if target else self.data.add(self.key,quad))
        self.entry_after_draw=oid
        return oid

    def edit(self, oid, **kwargs):
        obj=next(o for o in self.scene()['objects'] if o['object_id']==oid)
        if kwargs.get('quad') is not None and obj.get('localization_quad') and geometry_status(obj)!='EXACT':
            # Dragging a visible corner must not silently restore Pose accuracy.
            kwargs['localization_quad']=kwargs.pop('quad')
        return self.data.edit(self.key,oid,**kwargs)

    def confirm(self, event=None):
        if event is not None and self.ed._is_typing_shortcut(getattr(event,'widget',None)):
            return None
        if self.ed.prepare_navigation():
            self.h.status_var.set('Zapisano' if complete(self.data.state(self.key)) else 'Uzupełnij numer lub zaznacz, że jest nieczytelny.')
        return 'break'

    def empty(self):
        if self.scene()['objects']:
            self.h.status_var.set('Na zdjęciu są oznaczone tablice. Usuń błędne oznaczenia przed potwierdzeniem ich braku.')
            return 'break'
        if self.ed.prepare_navigation():self.commit(lambda:self.data.empty(self.key))
        return 'break'

    def navigate(self, direction):
        order=self.data.material.order;i=order.index(self.key)
        self.h.select(order[max(0,min(len(order)-1,i+direction))])
        return 'break'

    def restore(self, *, redo=False):
        if self.ed.prepare_navigation():self.commit(lambda:self.data.restore(self.key,redo=redo))
        return 'break'

    def remove(self):
        oid=self.ed.selected
        if oid and self.ed.prepare_navigation():self.commit(lambda:self.data.remove(self.key,oid))
        return 'break'

    def unreadable(self, oid=None):
        oid=oid or (self.ed.entry_target[1] if self.ed.entry_target else self.ed.selected)
        if oid and self.ed.finish_entry():
            self.commit(lambda:self.data.edit(self.key,oid,text='',text_state='UNREADABLE_TEXT',reason='Numer nieczytelny na fotografii'))
        return 'break'

    def without_outline(self, reason):
        if not self.ed.prepare_navigation():return
        oid=self.commit(lambda:self.data.add(self.key,geometry_reason=reason))
        if oid:
            self.ed.selected=oid;self.ed.mode='edit';self.ed.start_entry(oid)

    def geometry_na(self, oid, reason):
        status=geometry_status(dict(quad=None,geometry_exclusion_reason=reason))
        return self.set_geometry_status(oid,status)

    def _end_drawing(self, oid):
        """A status command is complete now; no next-click drawing intent."""
        ed=self.ed
        self.location_draft=None;self.exact_target=None;self.entry_after_draw=None
        ed.points=[];ed.pointer=None;ed.drag=None
        ed.draw_target=None;ed.draw_reference_target=None
        ed._click_candidate=None;ed._click_layer=None;ed._outside_click=None
        ed._s_hover_hit=None;ed.gt_focus_id='';ed._geometry_layer_state=None
        self.h._view_drag_at=None
        ed.collision_warning.clear();ed.corner_pulse.clear();ed.shift_preview.clear()
        ed.selected=oid;ed.hover='';ed.mode='edit'
        menu=getattr(self,'_context_menu',None)
        if menu is not None:
            try:menu.unpost()
            except tk.TclError:pass
        self.menu_open=False
        ed.c.focus_set();ed.request_overlay()

    def set_geometry_status(self, oid, status):
        if self.pending_action is not None:
            self.h.status_var.set('Najpierw ponów niezapisany zapis albo anuluj szkic przez Esc.')
            return 'break'
        if not self.ed.finish_entry():return 'break'
        key=self.key
        self._end_drawing(oid)
        self.commit(lambda:self.data.set_geometry_status(key,oid,status))
        self.ed.request_overlay()
        return 'break'

    def geometry_style(self, obj):
        status=geometry_status(obj)
        color,label,dash={
            'EXACT':('#38E4D3','Dokładna',()),
            'CUT_OFF':('#FF9F43','Ucięta',(6,3)),
            'OCCLUDED':('#CF8BFF','Zasłonięta',(6,3)),
            'BLURRED':('#65B5FF','Niewyraźna',(6,3)),
            'UNDETERMINED':('#B8C0CC','Obrys N/A',(2,4)),
        }[status]
        if not localization_outline(obj) and not obj.get('approximate_region') and not obj.get('approximate_point'):
            label='Lokalizacja N/A'
        return dict(color=color,label=label,dash=dash)

    def text_caption(self, obj, value=None):
        value=display_text(obj) if value is None else value
        if text_kind(obj)=='UNREADABLE_TEXT':return 'Numer nieczytelny'
        prefix='Numer: fragment ' if text_kind(obj)=='PARTIAL_TEXT' else 'Numer: '
        return prefix+(value or ('wpisz fragment' if text_kind(obj)=='PARTIAL_TEXT' else 'wpisz numer'))

    def set_text_kind(self, oid, kind):
        if not self.ed.prepare_navigation():return
        if kind=='UNREADABLE_TEXT':return self.unreadable(oid)
        if self.commit(lambda:self.data.edit(self.key,oid,text_state=kind)):
            self.ed.selected=oid;self.ed.start_entry(oid)
            self.h.status_var.set('Wpisz tylko pewny fragment numeru' if kind=='PARTIAL_TEXT' else 'Wpisz pełny numer')

    def source_point(self, x, y, *, clamp=False):
        p=self.ed.to_source(x,y);w,h=self.scene()['image_size']
        if not all(math.isfinite(v) for v in p):return None
        if clamp:return [max(0,min(w,p[0])),max(0,min(h,p[1]))]
        return list(p) if 0<=p[0]<=w and 0<=p[1]<=h else None

    def begin_geometry(self, kind, oid=None, *, reason='', point=None):
        ed=self.ed
        if not ed.prepare_navigation():return
        ed.selected=oid or '';ed.mode='select';ed.draw_target=None
        if kind=='EXACT_QUAD':
            self.exact_target=oid
            return ed.draw_toggle()
        self.location_draft=dict(kind=kind,oid=oid,reason=reason,image_id=self.key,
            anchor=None,value=None,before=None)
        ed.c.focus_set()
        if kind=='APPROXIMATE_POINT' and point is not None:
            self.location_draft['value']=point
            self.location_release()
        ed.request_overlay()

    def location_hit(self, x, y):
        for obj in reversed(self.scene()['objects']):
            point=obj.get('approximate_point');region=obj.get('approximate_region')
            if point is not None and math.dist((x,y),self.ed.to_canvas(point))<=12:
                return obj['object_id'],'photo_location',None
            if region is not None:
                l,t=self.ed.to_canvas(region[:2]);r,b=self.ed.to_canvas(region[2:])
                if l-5<=x<=r+5 and t-5<=y<=b+5:return obj['object_id'],'photo_location',None
        return None

    def location_press(self, event):
        ed=self.ed;draft=self.location_draft
        if not draft and ed.mode!='draw':
            hit=ed.hit(event.x,event.y)
            if hit and hit[1]=='photo_location':
                if not ed.finish_entry() or not self.available():return True
                obj=next(o for o in self.scene()['objects'] if o['object_id']==hit[0])
                value=obj.get('approximate_point') or obj.get('approximate_region')
                draft=dict(kind=obj['geometry_kind'],oid=obj['object_id'],reason=obj['geometry_exclusion_reason'],
                    image_id=self.key,anchor=self.source_point(event.x,event.y,clamp=True),value=list(value),before=list(value))
                self.location_draft=draft;ed.selected=obj['object_id'];ed.mode='edit'
        if not draft:return False
        if self.pending_action:return True
        if draft['before'] is None:
            p=self.source_point(event.x,event.y)
            if p is None:
                self.h.status_var.set('Wskaż miejsce na fotografii, poza pustym marginesem.');return True
            draft['anchor']=p
            draft['value']=p if draft['kind']=='APPROXIMATE_POINT' else [*p,*p]
        ed.c.focus_set();ed._click_candidate=None;self.h._view_drag_at=None
        ed.request_overlay();return True

    def location_motion(self, event):
        draft=self.location_draft
        if not draft or draft['anchor'] is None:return bool(draft)
        before=draft['before'];a=draft['anchor']
        p=self.ed.to_source(event.x,event.y) if before is not None else self.source_point(event.x,event.y,clamp=True)
        if before is not None:
            w,h=self.scene()['image_size'];dx=p[0]-a[0];dy=p[1]-a[1]
            dx=max(-before[0],min(w-before[-2],dx));dy=max(-before[1],min(h-before[-1],dy))
            draft['value']=[v+(dx if i%2==0 else dy) for i,v in enumerate(before)]
        elif draft['kind']=='APPROXIMATE_POINT':draft['value']=p
        else:draft['value']=[min(a[0],p[0]),min(a[1],p[1]),max(a[0],p[0]),max(a[1],p[1])]
        self.ed.request_overlay();return True

    def location_release(self, event=None):
        draft=self.location_draft
        if not draft:return False
        if event is not None and draft['anchor'] is not None:self.location_motion(event)
        value=draft['value']
        if value is None:return True
        if draft['image_id']!=self.key:return True
        if draft['kind']=='APPROXIMATE_REGION' and (value[0]>=value[2] or value[1]>=value[3]):
            self.h.status_var.set('Przeciągnij, aby wskazać obszar widocznego fragmentu.');return True
        if value==draft['before']:
            self.location_draft=None;self.ed.request_overlay();return True
        kwargs={'approximate_point' if draft['kind']=='APPROXIMATE_POINT' else 'approximate_region':list(value),
            'geometry_reason':draft['reason']}
        oid=draft['oid'];key=draft['image_id']
        result=self.commit(lambda:self.data.edit(key,oid,**kwargs) if oid else self.data.add(key,**kwargs))
        if result:
            self.location_draft=None;self.ed.selected=result;self.ed.mode='edit'
            self.ed.request_overlay();self.refresh_list()
            if not oid:self.ed.start_entry(result)
        return True

    def prepare_location(self):
        if not self.location_draft:return True
        if self.location_draft['value'] is not None:self.location_release()
        if self.location_draft:
            if not self.error:self.h.status_var.set('Dokończ lokalizację albo naciśnij Esc przed zmianą fotografii.')
            return False
        return True

    def cancel_location(self):
        if self.location_draft:
            self.location_draft=None;self.pending_action=None;self.error=''
            self.h.retry_button.grid_remove();self.h.status_var.set('Anulowano niezapisane położenie')
        self.exact_target=None

    def _paint_marker(self, kind, value, oid, index, *, draft=False):
        ed=self.ed;c=ed.c;tags=(TAG,'photo_location_draft' if draft else 'photo_location',oid)
        obj=next((o for o in self.scene()['objects'] if o['object_id']==oid),None)
        style=self.geometry_style(obj) if obj else dict(color='#B8C0CC',dash=(6,3))
        color=style['color']
        if kind=='APPROXIMATE_REGION':
            l,t=ed.to_canvas(value[:2]);r,b=ed.to_canvas(value[2:])
            c.create_rectangle(l,t,r,b,outline=color,width=2,dash=style['dash'],tags=tags)
            return ((l+r)/2,t,(l,t,r,b))
        x,y=ed.to_canvas(value)
        c.create_oval(x-5,y-5,x+5,y+5,outline=color,fill='#302618',width=2,tags=tags)
        c.create_text(x+8,y,text=str(index),anchor='w',fill=color,font=('Segoe UI',9,'bold'),tags=tags)
        return x,y,(x-6,y-6,x+6,y+6)

    def paint_location_draft(self):
        draft=self.location_draft
        if draft and draft['value'] is not None and not draft['oid']:
            self._paint_marker(draft['kind'],draft['value'],'draft',len(self.scene()['objects'])+1,draft=True)
        elif draft and draft['value'] is not None:
            # Exact-to-approximate conversions still show the saved quad until
            # release. The dashed/point draft is only a preview, never GT.
            obj=next(o for o in self.scene()['objects'] if o['object_id']==draft['oid'])
            if obj['quad'] is not None:self._paint_marker(draft['kind'],draft['value'],draft['oid'],'',draft=True)

    def paint_presence_only(self, obj, index):
        ed=self.ed;oid=obj['object_id'];draft=self.location_draft
        kind=obj.get('geometry_kind');value=obj.get('approximate_point') or obj.get('approximate_region')
        if draft and draft['oid']==oid and draft['value'] is not None:kind,value=draft['kind'],draft['value']
        editing=ed.entry is not None and ed.entry_target==(self.key,oid)
        label=self.text_caption(obj,ed.entry_var.get() if editing else None)
        style=self.geometry_style(obj)
        title=f"Tablica {index} · {style['label']}"
        w=min(260,max(150,ed._badge_font.measure(title)+12,ed._badge_font.measure(label)+12),max(80,ed.c.winfo_width()-24))
        height=70 if editing else 46
        if value is not None:
            ax,ay,bounds=self._paint_marker(kind,value,oid,index)
            # Keep the small badge adjacent to its real location, inside the
            # visible photo whenever possible (also at borders/letterboxing).
            iw,ih=self.scene()['image_size'];l,t=ed.to_canvas((0,0));r,b=ed.to_canvas((iw,ih))
            l,t=max(4,l),max(4,t);r,b=min(ed.c.winfo_width()-4,r),min(ed.c.winfo_height()-4,b)
            if r<l or b<t:return
            x=max(l,min(r-w,ax-w/2))
            y=ay-height-12 if ay-height-12>=t else bounds[3]+12
            y=max(t,min(b-height,y))
            ed.c.create_line(ax,ay,max(x,min(x+w,ax)),max(y,min(y+height,ay)),fill='#DDE7EF',tags=(TAG,'badge_tether',oid))
        else:x,y=12,12+(index-1)*64
        ed.c.create_rectangle(x,y,x+w,y+height,fill='#302618',outline=style['color'],tags=(TAG,'gt_badge',oid))
        ed.c.create_text(x+6,y+4,text=title+'\n'+label,anchor='nw',fill=style['color'],font=('Segoe UI',9),width=w-12,tags=(TAG,'gt_badge',oid))
        ed.badges[oid]=(x,y,x+w,y+height);ed.card_positions[oid]=(x,y);ed.card_widths[oid]=w
        ed.stack_badges[oid]=(x,y,x+w,y+height,None)
        if editing:
            ed.entry.place(x=x+1,y=y+height-25,width=w-2,height=24);ed.entry.lift()

    def context_menu(self, event=None, *, show=True):
        ed=self.ed;oid=ed.selected if event is None else None;point=None
        if event is not None:
            if getattr(event,'widget',None) is ed.entry and ed.entry_target:
                oid=ed.entry_target[1]
            else:
                x,y=event.x,event.y;hit=ed.hit(x,y,prefer_stack=True)
                oid=hit[0] if hit else None
                if not hit or hit[1] in ('body','corner','photo_location'):point=self.source_point(x,y)
                if not oid and point is None:
                    self.h.status_var.set('Menu dodawania tablic jest dostępne na fotografii.');return 'break'
        if oid and ed.mode!='draw' and not self.location_draft:
            ed.selected=oid;ed.mode='edit';ed.request_overlay()
        previous=getattr(self,'_context_menu',None)
        if previous is not None:previous.destroy()
        menu=tk.Menu(self.h.window,tearoff=False);self._context_menu=menu;key=self.key
        def command(fn):
            return lambda:fn() if self.key==key else None
        if oid:
            obj=next(o for o in self.scene()['objects'] if o['object_id']==oid)
            number=tk.Menu(menu,tearoff=False)
            for label,kind in (('Pełny numer','FULL_TEXT'),('Fragment numeru','PARTIAL_TEXT'),('Numer nieczytelny','UNREADABLE_TEXT')):
                number.add_command(label=label,command=command(lambda k=kind:self.set_text_kind(oid,k)))
            menu.add_cascade(label='Numer',menu=number)
            geometry=tk.Menu(menu,tearoff=False)
            for status,label in GEOMETRY_REASONS.items():
                geometry.add_command(label=label,command=command(lambda s=status:self.set_geometry_status(oid,s)))
            if geometry_status(obj)!='EXACT' and localization_outline(obj):
                geometry.add_separator()
                geometry.add_command(label='Przywróć dokładną geometrię',command=command(lambda:self.set_geometry_status(oid,'EXACT')))
            elif not localization_outline(obj):
                geometry.add_separator()
                geometry.add_command(label='Uzupełnij obrys (D)',command=command(lambda:self.begin_geometry('EXACT_QUAD',oid)))
                if not obj.get('approximate_region') and not obj.get('approximate_point'):
                    geometry.add_command(label='Wskaż położenie punktem',command=command(lambda:self.begin_geometry('APPROXIMATE_POINT',oid,
                        reason=obj['geometry_exclusion_reason'] or GEOMETRY_REASONS['UNDETERMINED'],point=point)))
            menu.add_cascade(label='Geometria',menu=geometry)
            menu.add_command(label='Usuń tablicę',command=command(lambda:self._remove_context(oid)))
        else:
            add=tk.Menu(menu,tearoff=False)
            add.add_command(label='Narysuj obrys (D)',command=command(lambda:self.begin_geometry('EXACT_QUAD')))
            add.add_command(label='Nie mogę wskazać obszaru — punkt',command=command(lambda:self.begin_geometry('APPROXIMATE_POINT',
                reason=GEOMETRY_REASONS['UNDETERMINED'],point=point)))
            menu.add_cascade(label='Dodaj tablicę',menu=add)
            menu.add_command(label='Na zdjęciu nie ma tablic',state='disabled' if self.scene()['objects'] else 'normal',command=command(self.empty))
        if event is None:
            menu.add_separator();menu.add_command(label='Cofnij',command=self.restore)
            menu.add_command(label='Ponów',command=lambda:self.restore(redo=True))
            menu.add_command(label='Informacje o zapisie',command=lambda:messagebox.showinfo('Zapis referencji',
                'Plik ocen:\n'+str(self.data.path)+'\n\nReferencja bez predykcji. Fragment numeru oraz orientacyjna lokalizacja nie są pełnym GT tekstu ani dokładnej geometrii.',parent=self.h.window))
        if not show:return menu
        self.menu_open=True
        menu.bind('<Unmap>',lambda e:self.h.window.after_idle(self._menu_closed))
        try:menu.tk_popup(event.x_root if event else self.h.more_button.winfo_rootx(),event.y_root if event else self.h.more_button.winfo_rooty()+self.h.more_button.winfo_height())
        finally:menu.grab_release()
        return 'break'

    def _menu_closed(self):
        self.menu_open=False
        if self.ed.entry is not None:self.ed.entry.focus_set()

    def _remove_context(self, oid):
        self.ed.selected=oid
        return self.remove()


class PhotoReferenceWindow:
    # Reuse the exact source bitmap/viewport renderer and animated zoom from Z4.
    fit_image=MobileSampleReviewWindow.fit_image
    zoom_image=MobileSampleReviewWindow.zoom_image
    _wheel_zoom=MobileSampleReviewWindow._wheel_zoom

    def __init__(self, parent, material, *, output=None, reference=None, source_factory=PhotographCanvasSource):
        self.session=reference if reference is not None else PhotoReference(material,output)
        self.window=tk.Toplevel(parent);self.window.title('GT fotografii — Z4')
        self.window._aat_skip_window_recovery=True
        self.window.geometry(f'{min(1450,self.window.winfo_screenwidth()-60)}x{min(900,self.window.winfo_screenheight()-100)}+25+25')
        self.window.minsize(900,600)
        self.bg='#eef2e9';self.fg='#18251b';self.muted='#4c5c50'
        self.window.configure(bg=self.bg)
        self._closed=False;self._saving=False;self.current_invocation=''
        self._image=None;self._photo=None;self._image_entry='';self._image_source_size=None;self._image_token=0
        self._view_zoom=1.;self._view_center=None;self._view_fit_scale=1.;self._view_pixel_scale=1.;self._view_drag_at=None
        self._viewport_paint=False
        self.status_var=tk.StringVar(value='D: narysuj tablicę albo wybierz „Brak tablic”.')
        self.progress=tk.StringVar();self.current_status=tk.StringVar()
        self.window.columnconfigure(0,weight=1);self.window.rowconfigure(1,weight=1)
        header=tk.Frame(self.window,bg=self.bg,padx=12,pady=8);header.grid(row=0,column=0,sticky='ew')
        self.header_title=tk.Label(header,text='GT fotografii',font=('Segoe UI',14,'bold'),bg=self.bg,fg=self.fg)
        self.header_title.pack(side='left')
        tk.Label(header,textvariable=self.progress,font=('Segoe UI',11,'bold'),bg=self.bg,fg=self.fg).pack(side='right')
        work=tk.Frame(self.window,bg=self.bg);work.grid(row=1,column=0,sticky='nsew',padx=10)
        work.columnconfigure(1,weight=1);work.rowconfigure(0,weight=1)
        self.list_font=font.Font(root=self.window,family='Segoe UI',size=9)
        status_width=self.list_font.measure('Do oznaczenia')+16
        left=tk.Frame(work,bg=self.bg,width=168+status_width);left.grid(row=0,column=0,sticky='ns',padx=(0,8))
        self.left=left
        left.grid_propagate(False);left.rowconfigure(0,weight=1);left.columnconfigure(0,weight=1)
        style=f'PhotoReference{id(self)}.Treeview'
        ttk.Style(self.window).configure(style,font=self.list_font,rowheight=self.list_font.metrics('linespace')+9)
        self.subject_tree=ttk.Treeview(left,columns=('photo','status'),show='headings',selectmode='browse',style=style)
        self.subject_tree.heading('photo',text='Fotografia');self.subject_tree.heading('status',text='Status')
        self.subject_tree.column('photo',width=150,minwidth=110);self.subject_tree.column('status',width=status_width,minwidth=status_width,stretch=False)
        self.subject_tree.grid(row=0,column=0,sticky='nsew')
        bar=ttk.Scrollbar(left,orient='vertical',command=self.subject_tree.yview);bar.grid(row=0,column=1,sticky='ns')
        self.subject_tree.configure(yscrollcommand=bar.set)
        self.subject_tree.tag_configure('ready',foreground='#28652c')
        viewer=tk.Frame(work,bg='#101820');viewer.grid(row=0,column=1,sticky='nsew')
        viewer.columnconfigure(0,weight=1);viewer.rowconfigure(1,weight=1)
        tk.Label(viewer,textvariable=self.current_status,bg=self.bg,fg=self.fg,anchor='w',padx=8,pady=4).grid(row=0,column=0,sticky='ew')
        self.canvas=tk.Canvas(viewer,bg='#101820',highlightthickness=0)
        self.canvas.grid(row=1,column=0,sticky='nsew')
        toolbar=tk.Frame(self.window,bg=self.bg,padx=10,pady=5);toolbar.grid(row=2,column=0,sticky='ew')
        self.toolbar=toolbar
        self.source=source_factory(self)
        self._canvas_editor=CanvasReviewEditor(self,None,work,viewer,reference_source=self.source)
        for text,command in [('Dodaj tablicę (D)',self._canvas_editor.draw_toggle)]:
            ttk.Button(toolbar,text=text,command=command).pack(side='left',padx=3)
        self.empty_button=ttk.Button(toolbar,text='Brak tablic',command=self.source.empty);self.empty_button.pack(side='left',padx=3)
        self.more_button=ttk.Button(toolbar,text='Więcej ▾',command=self.source.context_menu);self.more_button.pack(side='left',padx=3)
        self.pending_button=ttk.Button(toolbar,text='Następne niegotowe',command=self.next_pending)
        self.pending_button.pack(side='right',padx=3)
        ttk.Button(toolbar,text='Dalej →',command=lambda:self.source.navigate(1)).pack(side='right',padx=3)
        ttk.Button(toolbar,text='← Wstecz',command=lambda:self.source.navigate(-1)).pack(side='right',padx=3)
        self.image_help=tk.Label(self.window,bg=self.bg,fg=self.muted,anchor='w',font=('Segoe UI',9),padx=12)
        self.image_help.grid(row=3,column=0,sticky='ew')
        self.status_label=tk.Label(self.window,textvariable=self.status_var,bg=self.bg,fg=self.muted,anchor='w',padx=12,pady=5)
        self.status_label.grid(row=4,column=0,sticky='ew')
        self.retry_button=ttk.Button(self.window,text='Ponów zapis',command=self.source.retry)
        self.subject_tree.bind('<<TreeviewSelect>>',self._selected)
        self.canvas.bind('<Configure>',lambda e:self._paint_image())
        for event in ('<MouseWheel>','<Button-4>','<Button-5>'):self.canvas.bind(event,self._wheel_zoom)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.source.refresh_list()
        self.select(self.session.data['last_image_id'],initial=True)

    def _is_source_frame_view(self):return False
    def _toggle_source_frame_view(self):return 'break'

    def _paint_image(self, message=''):
        MobileSampleReviewWindow._paint_image(self,message or 'Nie można wyświetlić fotografii.')
        if hasattr(self,'image_help'):self.source.caption()

    def _selected(self, event=None):
        choices=self.subject_tree.selection()
        if choices and choices[0]!=self.current_invocation:self.select(choices[0])

    def select(self, key, *, initial=False):
        ed=self._canvas_editor
        if key==self.current_invocation:return True
        if not initial and not ed.prepare_navigation():
            self.subject_tree.selection_set(self.current_invocation)
            return False
        try:
            image=self.session.material.image(key)
            if not initial:self.session.remember(key)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            self.source.error=self.source.report_error(exc,'odczyt')
            self.status_var.set('Nie można otworzyć fotografii: '+self.source.error);self.status_label.configure(fg='#b02020')
            if self.current_invocation:self.subject_tree.selection_set(self.current_invocation)
            return False
        self.current_invocation=key;self._image=image;self._image_source_size=image.size
        self._image_entry=self.session.material.images[key]['relative_path'];self._image_token+=1
        self._view_zoom=1.;self._view_center=None
        ed.sync();self.subject_tree.selection_set(key);self.subject_tree.focus(key);self.subject_tree.see(key)
        self._paint_image();self.source.error='';self.status_label.configure(fg=self.muted)
        return True

    def next_pending(self):
        target=self.session.next_pending(self.current_invocation)
        if target:self.select(target)
        else:self.status_var.set('Wszystkie fotografie mają zapisane, kompletne oznaczenia.')

    def close(self):
        if self._closed:return
        if not self._canvas_editor.prepare_navigation():
            if self.source.pending_action is None:return
            if not messagebox.askyesno('Niezapisane zmiany',
                'Zapis nie powiódł się. Zamknąć okno bez niezapisanych zmian? Wcześniejsze zapisane oceny pozostaną zachowane.',parent=self.window):return
            self.source.pending_action=None;self._canvas_editor.destroy_entry()
            self._canvas_editor.points=[];self._canvas_editor.drag=None
        self._closed=True;self._canvas_editor.dispose()
        from .mobile_canvas_zoom import cancel_zoom
        cancel_zoom(self)
        self.window.destroy()


def open_photo_reference(parent, *, source=None, output=None):
    from alpr_runtime import workspace_root
    default=workspace_root()
    if source is None:
        source=filedialog.askdirectory(parent=parent,title='Wybierz folder fotografii zatwierdzonych w manifestach',initialdir=str(default))
    if not source:return None
    try:return PhotoReferenceWindow(parent,Material(source),output=output)
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        messagebox.showerror('Nie można otworzyć fotografii',str(exc),parent=parent)
        return None
