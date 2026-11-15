"""One source-bound navigation action and a transient cue; no review writes."""
from __future__ import annotations
import copy, tkinter as tk

TAG='pending_mt_locator'
COLOR='#C29AFF'

def task_target(ed,task,scene,state):
    """Keep the displayed pending unit and its exact FINAL identity together."""
    units=task.get('units',[])
    if not scene or not state or not units or task.get('canonical_progress',{}).get('source_unknown'):
        return task.get('next_action',''),None
    unit=units[0]
    if task['next_action']!=unit['text']:return task['next_action'],None
    selected=next((u for u in units if u['object_id']==ed.selected and 'pending_mt:'+u['object_id'] in u['reasons']),None)
    if selected:unit=selected
    if 'pending_mt:'+unit['object_id'] not in unit['reasons']:return unit['text'],None
    matches=[c for c in state['candidates'] if c['role']=='FINAL' and c['attempt_id']==unit['final_mt_id'] and c['object_id']==unit['object_id']]
    if len(matches)!=1:return unit['text'],None
    c=matches[0]
    return unit['text'],dict(session=id(ed.h.session),session_id=ed.h.session.session_id,
        sha=scene['source_archive_sha256'],binding=scene['source_binding'],gid=scene['mt_invocation_id'],
        revision=ed.h.session.review.review_revision,oid=c['object_id'],attempt_id=c['attempt_id'],index=c['mt_detection_index'])

def current_target(ed):
    from .mobile_canvas_review_tasks import for_scene
    from .mobile_canvas_clean_view import source
    s=ed._read()
    if not s:return None
    return task_target(ed,for_scene(ed.h.session,ed.h.current_invocation,s),s,source(ed))[1]

def resolve(ed,target):
    from ..ranking import mobile_canvas_review as core,mobile_mt_geometry_layers as geometry
    from ..ranking.mobile_canvas_resolution import verify_frame
    from . import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers
    from .mobile_canvas_bbox_marker import enrich
    if not target or target!=current_target(ed):raise ValueError('Zadanie zmieniło się. Wybierz aktualne Pokaż MT.')
    s=core.scene(ed.h.session,target['gid'],include_history=False)
    if ed.h._image_entry!=s['image_entry'] or tuple(ed.h._image_source_size or ())!=tuple(s['image_size']):
        raise ValueError('Nie można wskazać MT: pełny obraz nie jest wczytany.')
    verify_frame(ed.h.session,target['gid']);core.require_source_final(ed.h.session,target['gid'],target['oid'])
    state=clean.source(ed)
    matches=[c for c in state['candidates'] if c['object_id']==target['oid'] and c['attempt_id']==target['attempt_id'] and c['role']=='FINAL']
    if len(matches)!=1:raise ValueError('Nie można wskazać MT: niejednoznaczne źródło.')
    value=enrich(ed.h.session,target['gid'],s,matches[0],geometry.snapshot(ed.h.session,target['gid'],target['oid'],scene=s,state=state))
    available={v['kind']:v for v in value['layers'] if v['kind'] in ('POSE','BBOX')}
    old=layers.model(ed) if ed.selected==target['oid'] else None
    kind=old['active'] if old and old['active'] in available else 'POSE' if 'POSE' in available else 'BBOX'
    if kind not in available:raise ValueError('Nie można wskazać MT w tym obrazie: brak wiarygodnej geometrii źródłowej.')
    from .mobile_canvas_navigator import quad
    return dict(target=copy.deepcopy(target),kind=kind,quad=copy.deepcopy(quad(value,kind)),marker=available[kind].get('marker'))

class ActionLink(tk.Label):
    """A focusable inline action with text-line height on Windows Tk."""
    def __init__(self,parent,command,**kwargs):
        super().__init__(parent,**kwargs);self.command=command;self.pressed=False
        self.bind('<ButtonPress-1>',self.press)
        self.bind('<ButtonRelease-1>',self.release)
    def invoke(self):
        if self.cget('state')!='disabled':return self.command()
    def press(self,event):
        self.pressed=self.cget('state')!='disabled'
        return 'break'
    def release(self,event):
        pressed=self.pressed;self.pressed=False
        if pressed and 0<=event.x<self.winfo_width() and 0<=event.y<self.winfo_height():self.invoke()
        return 'break'

class TaskText(tk.Text):
    """Label-compatible flowing task text with one embedded keyboard action."""
    def __init__(self,parent,p):
        self._text='';self._target=None;self.locator=None;self._resize_after=None;self.compact=False
        super().__init__(parent,bg=p['bg'],fg=p['fg'],font=('Segoe UI',10),wrap='word',width=1,height=1,
            bd=0,highlightthickness=0,padx=0,pady=0,takefocus=0,state='disabled',cursor='arrow')
        self.bindtags((str(self),))
        self.button=ActionLink(self,command=self.invoke,text='',font=('Segoe UI',9,'underline'),bg=p['bg'],fg=p['blue'],
            activebackground=p['bg'],activeforeground=p['blue'],disabledforeground=p['muted'],
            relief='flat',bd=0,padx=2,pady=0,highlightthickness=1,highlightbackground=p['bg'],highlightcolor=p['blue'],takefocus=1,cursor='hand2')
        self.button.bindtags((str(self.button),'Label'))
        self.button.bind('<Return>',self.invoke);self.button.bind('<KP_Enter>',self.invoke)
        self.button.bind('<space>',lambda e:'break')
        self.button.bind('<Tab>',lambda e:self.traverse(False))
        self.button.bind('<Shift-Tab>',lambda e:self.traverse(True))
        self.bind('<Configure>',lambda e:self.schedule_resize());self.bind('<Destroy>',self.destroyed)

    def configure(self,cnf=None,**kwargs):
        if isinstance(cnf,dict):kwargs={**cnf,**kwargs};cnf=None
        text=kwargs.pop('text',None);kwargs.pop('wraplength',None)
        result=super().configure(cnf,**kwargs) if cnf is not None or kwargs else None
        if text is not None and str(text)!=self._text:self._text=str(text);self.rebuild()
        return result
    config=configure
    def cget(self,key):return self._text if key=='text' else super().cget(key)
    __getitem__=cget
    def set_target(self,target):
        if self._target!=target:
            self._target=copy.deepcopy(target);self.rebuild()
        self.sync_state()
    def sync_state(self):
        enabled=bool(self._target and self.locator and not self.locator.blocked())
        self.button.configure(state='normal' if enabled else 'disabled')
    def invoke(self,event=None):
        if self._target and self.locator:self.locator.activate(copy.deepcopy(self._target))
        return 'break'
    def traverse(self,back):
        (self.button.tk_focusPrev() if back else self.button.tk_focusNext()).focus_set()
        return 'break'
    def rebuild(self):
        super().configure(state='normal')
        # Deleting an embedded Text window also destroys its Tk widget. Detach
        # it first so changing/finishing a task can reuse the same button.
        for name in self.window_names():self.window_configure(name,window='')
        self.delete('1.0','end')
        self.insert('1.0',self._text)
        if self._target:
            self.button.configure(text=f"Pokaż MT{self._target['index']+1}")
            self.insert('end',' · ');self.window_create('end',window=self.button,align='center')
        super().configure(state='disabled');self.schedule_resize()
    def schedule_resize(self):
        if self._resize_after is None:self._resize_after=self.after_idle(self.resize)
    def resize(self):
        self._resize_after=None
        if self.winfo_width()<10:return
        count=self.count('1.0','end','displaylines')
        lines=max(1,count[0] if count else 1)
        if int(super().cget('height'))!=lines:super().configure(height=lines)
        wrapped=bool(self._target and self.window_names() and not self.get(self.index(str(self.button)+' display linestart'),str(self.button)).strip())
        if self.compact!=wrapped:
            self.compact=wrapped
            if self.locator:self.locator.reserve_line_space(wrapped)
    def destroyed(self,event):
        if event.widget is self and self._resize_after is not None:
            self.after_cancel(self._resize_after);self._resize_after=None

class Locator:
    def __init__(self,ed):
        self.ed=ed;self.signal=None;self.after_id=None
        self.line=ed.contexts.fields['image'];self.line.locator=self
        ed.c.bind('<Destroy>',lambda e:self.clear() if e.widget is ed.c else None,add='+')
    def reserve_line_space(self,wrapped):
        # A wrapped inline link uses the adjacent existing whitespace instead
        # of pushing session cards further down. Fonts/cards/sections stay put.
        ed=self.ed
        if self.line.winfo_manager():self.line.pack_configure(pady=(0 if wrapped else 5,0))
        ed.approve.pack_configure(pady=(0 if wrapped else 6,0))
        if ed.clean_exclude_button.winfo_manager():ed.clean_exclude_button.pack_configure(pady=(0 if wrapped else 5,0))
        ed.contexts.sections['image'].pack_configure(pady=(0,5 if wrapped else 6))
    def blocked(self):
        ed=self.ed
        return bool(ed.entry is not None or ed.drag or ed.points or ed.mode=='draw' or ed.busy or ed.h._saving or ed.in_refresh)
    def activate(self,target):
        ed=self.ed
        if self.blocked():ed.message('Zakończ edycję przed wskazaniem MT.');return False
        try:signal=resolve(ed,target)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            self.clear();ed.message('Nie można wskazać MT w tym obrazie. '+str(exc));return False
        from .mobile_canvas_navigator import select_layer,show_geometry
        self.clear();ed._reserved_edit=None;ed.gt_focus_id=''
        if not select_layer(ed,signal['kind'],target['oid']):
            ed.message('Nie można wskazać MT w tym obrazie.');return False
        show_geometry(ed,signal['quad'],only_if_needed=True)
        self.signal=signal;self.after_id=ed.c.after(2000,self.clear)
        self.paint();ed.message(f"MT{target['index']+1} · do oceny — wskazano źródło bez zmiany oceny.")
        return True
    def clear(self):
        if self.after_id is not None:
            try:self.ed.c.after_cancel(self.after_id)
            except tk.TclError:pass
        self.after_id=None;self.signal=None
        try:self.ed.c.delete(TAG)
        except tk.TclError:pass
    def sync(self):
        self.line.sync_state()
        if not self.signal:return
        ed=self.ed;target=self.signal['target'];s=ed._read()
        from .mobile_canvas_geometry_layers import model
        value=model(ed)
        if (self.blocked() or not s or ed.h._closed or id(ed.h.session)!=target['session'] or ed.h.current_invocation!=target['gid']
                or s['source_binding']!=target['binding'] or ed.h.session.review.review_revision!=target['revision']
                or ed.selected!=target['oid'] or ed.mode!='edit' or not value or value['active']!=self.signal['kind']):self.clear()
    def paint(self):
        self.sync();ed=self.ed;ed.c.delete(TAG)
        if not self.signal:return
        from .mobile_canvas_editor import TAG as OVERLAY
        from . import mobile_canvas_clean_view as clean
        signal=self.signal;target=signal['target'];q=signal['quad'];tags=(OVERLAY,TAG,target['oid'],target['attempt_id'])
        if signal['marker']:
            from .mobile_canvas_bbox_marker import paint
            paint(ed,signal['marker'],color=COLOR,tags=tags)
        else:
            coords=[v for p in q for v in ed.to_canvas(p)]
            for color,width in (('#080808',7),('#FFFFFF',5),(COLOR,3)):
                ed.c.create_polygon(*coords,fill='',outline=color,width=width,tags=tags+('pending_mt_contour',))
        box=clean.bounds(ed,q);cw,ch=ed.c.winfo_width(),ed.c.winfo_height()
        if box[2]<0 or box[0]>cw or box[3]<0 or box[1]>ch:return
        label=f"MT{target['index']+1} · do oceny";font=ed._badge_font
        width=font.measure(label)+12;height=font.metrics('linespace')+8
        rect=clean.badge_position(ed,box,width,height)
        if rect is None:
            # A zoomed contour may fill the viewport. Keep the cue visible in
            # its interior, avoiding every visible handle and existing label.
            handles=[(x-16,y-16,x+16,y+16) for r in clean.records(ed) for p in (r['gt'] or []) for x,y in [ed.to_canvas(p)]]
            obstacles=handles+ed.clean_label_boxes
            for y in range(6,max(7,ch-height-6),height+6):
                for x in (6,max(6,(cw-width)/2),max(6,cw-width-6)):
                    candidate=(x,y,x+width,y+height)
                    if x+width<cw and y+height<ch and not any(clean.intersects(candidate,b) for b in obstacles):rect=candidate;break
                if rect:break
        if rect:
            ed.c.create_rectangle(*rect,fill='#241A37',outline=COLOR,tags=tags)
            ed.c.create_text(rect[0]+6,(rect[1]+rect[3])/2,text=label,font=font,anchor='w',fill='#FFFFFF',tags=tags+('pending_mt_label',))
            clean.badge_tether(ed,q,rect,target['oid'],tag=TAG)
        ed.c.tag_raise(TAG)
