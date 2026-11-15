"""Small A3 adapter for the existing Z4 window/tree/Canvas and sidecar."""
from pathlib import Path
import copy, hashlib, tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageOps, ImageDraw, ImageTk
from ..ranking import mobile_f5a_assisted as data
from ..ranking.photo_reference import display_text, text_kind, geometry_status

def active(host):
    ui=getattr(host,'_f5a_a3',None)
    return ui if ui and ui.active else None

class AssistedReview:
    def __init__(self,h,header,left):
        self.h=h;self.ed=h._canvas_editor;self.active=False;self.ref=None
        self.current_exposure=None;self.current_frame=None;self.targets={};self.seen=set();self._render_key=None;self._thumbnail_key=None
        self.bar=tk.Frame(header,bg=h.bg);self.bar.grid(row=2,column=0,columnspan=2,sticky='ew')
        self.connect_button=ttk.Button(self.bar,text='Połącz GT i A2 dla serii F5A',command=self.connect);self.connect_button.pack(side='left')
        self.navigation=tk.Frame(self.bar,bg=h.bg)
        ttk.Button(self.navigation,text='Poprzednie',command=lambda:self.navigate(-1)).pack(side='left',padx=3)
        ttk.Button(self.navigation,text='Dalej',command=lambda:self.navigate(1)).pack(side='left')
        area=self.ed.contexts.sections['object']
        self.more=ttk.Button(area)
        self.third=ttk.Button(area)
        self.decision_buttons=(self.ed.clean_gtr_confirm_button,self.more,self.third)
        self.scope_label=tk.Label(area,bg=self.ed.p['bg'],fg=h.fg,justify='left',anchor='w',wraplength=235)
        self.tools=tk.Button(area,text='Historia',command=self.aux_menu,relief='flat',anchor='w',bg=self.ed.p['bg'],fg=self.ed.p['blue'])
        self.selection_kind='exposure';self.selection_key=None
        self.reference_label=tk.Label(left,bg=h.review_card_bg,fg=h.fg,font=('Segoe UI',9),justify='left',compound='top',anchor='w',cursor='hand2')
        self.reference_label.bind('<Button-3>',self.identity_menu)
        self.left=left
        h.subject_tree.bind('<Button-3>',self.tree_menu,add='+')

    def loaded(self):
        if not self.h.session:return
        try:
            ref=getattr(self.h.browser,'_f5a_reference',None) or data.registered()
            if ref:self.attach(ref)
            else:
                self.candidates=data.discover(self.h.session.path)
                if len(self.candidates)==1:self.connect_button.configure(text='Połącz znalezione A2 i GT dla całej serii')
        except (ValueError,OSError,KeyError) as exc:self.h.status_var.set('A3: '+str(exc))

    def connect(self):
        if not self.h.session or self.h._saving:return
        candidates=getattr(self,'candidates',data.discover(self.h.session.path))
        path=str(candidates[0]) if len(candidates)==1 and not self.active else filedialog.askopenfilename(parent=self.h.window,title='Wskaż mapping.json A2',filetypes=[('Mapowanie A2','*.json')])
        if not path:return
        try:
            ref=data.Reference(path)
            ref.attach(self.h.session)
            data.register(ref);self.attach(ref)
        except (OSError,ValueError,KeyError) as exc:self.h.status_var.set('Nie połączono A3: '+str(exc))

    def attach(self,ref):
        ref.attach(self.h.session);self.ref=ref;self.active=True;self.h.browser._f5a_reference=ref
        self.connect_button.pack_forget();self.navigation.pack(side='left')
        s=ref.sessions[self.h.session.session_id]
        self.h.session_label.configure(text=f"{s['variant']} · Powtórzenie {s['replicate']} · Blok {s['block'][-1]}")
        self.ed.left_width=300;self.ed.sidebar_width=260;self.ed._review_columns=None
        self.h.progress_label.grid();self.h.form_status_label.grid_remove()
        for w in self.left.grid_slaves(row=0):
            if isinstance(w,tk.Label):w.configure(text='ZDJĘCIA')
        for w in self.left.grid_slaves(row=3):w.grid_remove()
        if not getattr(self,'_view_menu_ready',False):
            self.ed.clean_menu.add_cascade(label='Geometria MT',menu=self.ed.navigator.menu);self._view_menu_ready=True
        self.reference_label.grid(row=1,column=0,sticky='ew',pady=4)
        # Canonical JPEG geometry stays on the photograph thumbnail. On the
        # phone show source inference/ROI, not provisional legacy REF seeds.
        self.ed.ref_visible.set(False);self.ed.inf_visible.set(True);self.ed.show_roi_overlay=True
        self.h.subject_tree.configure(selectmode='extended')
        self.h.complete_button.configure(text='Zatwierdź sesję A3',command=lambda:self.save(lambda:data.approve_session(self.h.session)))
        self.exposures=sorted([e for e in ref.exposures.values() if e['session_id']==self.h.session.session_id],key=lambda e:e['planned_position'])
        self.seen.update(data.state(self.h.session)['blocks'].get(self.h.session.session_id,{}).get('viewed_exposures',[]))
        self.render_tree()
        resume=self.h.session.review.provenance.get(data.KEY,{}).get('resume',{})
        if resume.get('gid') in self.h.session.mt_invocations:self.select('a3:mt:'+resume['gid'])
        else:
            scope=resume.get('scope')
            frame=next((iid for iid,target in self.targets.items() if scope in ref.frames and target[0]=='frame' and ref.frame_key(target[1])==ref.frame_key(scope)),None)
            self.select(frame or 'a3:exp:'+(scope if scope in ref.exposures else self.exposures[0]['exposure_id']))
        self.h.status_var.set('Oceń zdjęcie lub wybierz MT.')

    def _progress(self):
        key=(id(self.h.session),self.h.session.review.review_revision)
        if getattr(self,'_progress_key',None)!=key:
            self._progress_key=key
            self.units={g:data.unit_state(self.h.session,g) for g in self.h.session.mt_invocations}
        return self.units

    def photo_done(self,e):
        key=(id(self.h.session),self.h.session.review.review_revision)
        if getattr(self,'_photos_key',None)!=key:
            st=data.state(self.h.session);self._photos_key=key
            self._photos_done={r['exposure_id']:bool(r['evidence_frame_ids']) and all(data.identity(self.h.session,f,st)['decision']!='PROPOSED' for f in r['evidence_frame_ids']) for r in self.exposures}
        return self._photos_done[e['exposure_id']]

    def detections(self,gid):
        return [r for r in self.h.session.mt_invocations[gid].records if r.get('mt_status') in ('VALID_QUAD','DETECTION_INVALID_QUAD')]

    def render_tree(self):
        if not self.active:return False
        h=self.h;tree=h.subject_tree;ed=self.ed;state=data.state(h.session)
        units=self._progress();key=(id(h.session),h.session.review.review_revision)
        if key==self._render_key:self.update_header();return True
        self._render_key=key;previous=tree.selection();desired=set();targets={};self.mt_numbers={}
        ed.tree_refreshing=True
        def node(iid,parent,text,target=None):
            desired.add(iid)
            if tree.exists(iid):tree.move(iid,parent,'end');tree.item(iid,text=text,values=())
            else:tree.insert(parent,'end',iid=iid,text=text,values=(),open=False)
            if target:targets[iid]=target
        try:
            tree.configure(displaycolumns=(),show='tree');tree.heading('#0',text='Zdjęcia i MT')
            represented=set()
            for n,e in enumerate(self.exposures,1):
                eid=e['exposure_id'];root='a3:exp:'+eid
                photos={data.identity(h.session,f)['photo_sha256'] for f in e['evidence_frame_ids']}
                gt=' / '.join(str(self.ref.photos[p]['global_position']) for p in sorted(photos-{None})) or '—'
                gids=[gid for (sid,gid),m in self.ref.calls.items() if sid==h.session.session_id and any(f in e['evidence_frame_ids'] for f in m['frame_ids'])]
                node(root,'',f"Zdjęcie {n} · GT {gt}"+(' · ✓' if self.photo_done(e) else ''),('exposure',eid))
                for j,gid in enumerate(gids,1):
                    if gid not in h.session.mt_invocations:continue
                    represented.add(gid);self.mt_numbers[gid]=j;count=len(self.detections(gid))
                    word='wskazanie' if count==1 else 'wskazania' if 2<=count%10<=4 and not 12<=count%100<=14 else 'wskazań'
                    node('a3:mt:'+gid,root,f"MT {j} · {count} {word}"+(' · ✓' if units[gid]['complete'] else ''),('mt',gid))
                # Same CameraX bitmap can have SCENE/TV representations. Show it
                # once but retain every A2 ID and every event in the source model.
                shown=set();folder='a3:frames:'+eid;node(folder,root,'Klatki telefonu')
                for fid in sorted(e['evidence_frame_ids'],key=lambda f:(self.ref.frames[f]['capture_elapsed_nanos'],f)):
                    f=self.ref.frames[fid];fk=self.ref.frame_key(fid)
                    if fk in shown:continue
                    shown.add(fk)
                    node('a3:frame:'+fid,folder,f"Klatka telefonu {len(shown)}",('frame',fid))
                tree.item(folder,text=f'Klatki telefonu ({len(shown)})')
            extra='a3:extra'
            extras=[f for f in self.ref.frames.values() if f['session_id']==h.session.session_id and f['frame_id'] not in self.ref.frame_exposure]
            if extras or any(g not in represented for g in h.session.mt_invocations):node(extra,'','Poza zdjęciami GT')
            extra_frames=set()
            for f in extras:
                fid=f['frame_id'];physical=self.ref.frame_key(fid)
                if physical in extra_frames:continue
                extra_frames.add(physical)
                node('a3:frame:'+fid,extra,f'Klatka telefonu {len(extra_frames)}',('frame',fid))
            for gid in h.session.mt_invocations:
                if gid not in represented:
                    self.mt_numbers[gid]=len(self.mt_numbers)+1;node('a3:mt:'+gid,extra,f"MT {self.mt_numbers[gid]} · bez zdjęcia GT",('mt',gid))
            def prune(parent=''):
                for iid in tree.get_children(parent):
                    if iid not in desired:tree.delete(iid)
                    else:prune(iid)
            prune();self.targets=targets
            ed.observation_model=None;ed.scene_tree_ids={};ed.scene_mt_tree_ids={};ed.linked_tree_rows={}
            surviving=[i for i in previous if tree.exists(i)]
            if surviving:tree.selection_set(surviving)
        finally:ed.tree_refreshing=False
        h._schedule_status_pills();self.update_header();return True

    def select(self,iid):
        if not self.active:return False
        if self.ed.tree_refreshing or self.h._saving:return True
        target=self.targets.get(iid)
        if not target:return True
        if not self.ed.prepare_navigation():return True
        kind,key=target
        self.selection_kind=kind;self.selection_key=key
        if self.h.subject_tree.exists(iid) and iid not in self.h.subject_tree.selection():self.h.subject_tree.selection_set(iid)
        if kind=='exposure':
            e=self.ref.exposures[key];self.current_exposure=key
            if not e['evidence_frame_ids']:return True
            self.open_frame(e['evidence_frame_ids'][0])
        elif kind=='frame':self.open_frame(key)
        else:self.open_call(key)
        if self.h.subject_tree.exists(iid):self.h.subject_tree.focus(iid);self.h.subject_tree.see(iid)
        # Scope can change from photo to its already displayed frame without
        # reloading the bitmap. Rebind all three actions immediately.
        self.decorate();return True

    def open_frame(self,fid):
        if (self.current_frame==fid and self.h.current_record and self.h.current_record[0] in ('scene','a3frame')
                and self.h._image is not None and self.h._image_entry==self.ref.frames[fid]['full_frame_entry']):return
        f=self.ref.frames[fid];self.current_frame=fid;self.current_exposure=self.ref.frame_exposure.get(fid)
        h=self.h
        if f['kind']=='SCENE_FRAME':
            from .mobile_canvas_scene_frames import open_scene
            open_scene(self.ed,f['scene_event_id'])
        else:
            h.subject_key='';h.current_record=('a3frame',fid);h.current_invocation=''
            h._record_ids={};h.record_tree.delete(*h.record_tree.get_children());h.gt_var.set('');h.note_var.set('')
            h._image_rows=[('Klatka telefonu',dict(image_entry=f['full_frame_entry'],source_frame_view=True))]
            h.image_choice.configure(values=['Klatka telefonu']);h.image_choice.current(0);h._load_image();self.ed.sync()
        self.refresh_reference()

    def open_call(self,gid):
        fid=self.ref.call_frame(self.h.session,gid);self.current_frame=fid;self.current_exposure=self.ref.frame_exposure.get(fid)
        group=self.h.session.mt_invocations[gid]
        aid=str(group.records[0].get('id') or group.records[0].get('attempt_id'))
        if self.h.current_invocation!=gid:
            self.ed.scene_mt_focus=None;self.h._multi_focus_attempt(aid);self.h._select_record()
        self.refresh_reference()

    def sync_selection(self):
        if not self.active:return False
        self.refresh_reference();return True

    def navigate(self,direction):
        ids=[iid for iid,(kind,key) in self.targets.items() if kind==self.selection_kind]
        current=next((iid for iid in ids if self.targets[iid][1]==self.selection_key),None)
        if not ids:return 'break'
        index=ids.index(current) if current in ids else 0;target=ids[max(0,min(len(ids)-1,index+direction))]
        self.h.subject_tree.selection_set(target);self.select(target);return 'break'

    def next_pending(self):
        e=next((e for e in self.exposures if not self.photo_done(e)),None)
        if e:self.select('a3:exp:'+e['exposure_id']);return
        gid=next((g for g,u in self._progress().items() if not u['complete']),None)
        if gid:self.select('a3:mt:'+gid)
        else:self.h.status_var.set('Zdjęcia i MT ocenione. Możesz zatwierdzić sesję A3.')

    def save(self,operation):
        if self.h._saving or not self.ed.prepare_navigation():return False
        try:
            operation();self._render_key=None;self._thumbnail_key=None
            self.ed.cache_key=None;self.ed.sync();self.render_tree();self.refresh_reference();self.ed.request_overlay()
            self.h.status_var.set('Ocena zapisana.')
            return True
        except (ValueError,OSError,KeyError) as exc:
            self.h.status_var.set('Nie zapisano A3: '+str(exc));return False

    def confirm_identity(self,event=None):
        if event is not None and self.ed._is_typing_shortcut(getattr(event,'widget',None)):return None
        if not self.current_frame or self.h._image is None or self.h._image_entry!=self.ref.frames[self.current_frame]['full_frame_entry']:
            self.h.status_var.set('Poczekaj na klatkę telefonu.');return 'break'
        scope=self.current_frame if self.selection_kind=='frame' else self.current_exposure or self.current_frame
        saved=data.state(self.h.session)['identities'].get(scope,{}) if scope in self.ref.exposures else data.identity(self.h.session,self.current_frame)
        if saved.get('decision') in ('CONFIRMED','CORRECTED'):
            self.h.status_var.set('Zdjęcie jest już ocenione.');return 'break'
        if scope:self.save(lambda:data.decide_identity(self.h.session,scope,'CONFIRMED'))
        return 'break'

    def confirm_block(self):self.save(lambda:data.confirm_block(self.h.session,self.seen))
    def undo(self,redo=False):self.save(lambda:data.undo(self.h.session,redo=redo));return 'break'

    def _reason(self,title,*,parent=None):return simpledialog.askstring('Powód oceny',title,parent=parent or self.h.window)
    def _identity_action(self,scope,decision,photo=None,*,parent=None):
        prompt='Dlaczego to nie jest żadne z 40 zdjęć?' if decision=='REJECTED' else 'Dlaczego nie można ustalić zdjęcia?'
        reason=self._reason(prompt,parent=parent) if decision in ('REJECTED','N/A') else ''
        if reason is None:return False
        return self.save(lambda:data.decide_identity(self.h.session,scope,decision,photo=photo,reason=reason))

    PHOTO_STATUS={'PROPOSED':'Do sprawdzenia','CONFIRMED':'Potwierdzone','CORRECTED':'Poprawione','REJECTED':'Odrzucone','N/A':'Nieustalone'}

    def photo_scope(self):
        return self.current_frame if self.selection_kind=='frame' else self.current_exposure or self.current_frame

    def scope_text(self,scope):
        return f"Dotyczy: zdjęcia {self.ref.exposures[scope]['planned_position']}" if scope in self.ref.exposures else 'Dotyczy: tylko tej klatki'

    def photo_caption(self,sha):
        if not sha:return 'Brak propozycji zdjęcia GT'
        plates=self.ref.plates(sha);number=self.ref.photos[sha]['global_position']
        texts=[('Fragment: '+display_text(p)) if text_kind(p)=='PARTIAL_TEXT' else display_text(p) or 'Tekst nieczytelny' for p in plates]
        return f'GT {number} · '+('Bez tablic' if not plates else 'Kilka tablic\n'+' · '.join(texts) if len(plates)>1 else texts[0])

    def photo_actions(self,scope):
        saved=data.state(self.h.session)['identities'].get(scope if scope in self.ref.exposures else self.ref.frame_key(scope),{})
        proposed=self.ref.exposures[scope]['photo_sha256'] if scope in self.ref.exposures else self.ref.frames[scope]['photo_sha256']
        return [('Zgadza się',lambda:self._identity_action(scope,'CONFIRMED'),bool(proposed) and saved.get('decision')!='CONFIRMED'),
                ('Wybierz inne zdjęcie',lambda:self.choose_photo(scope),True),
                ('Nie można ustalić',lambda:self._identity_action(scope,'N/A'),True)]

    @staticmethod
    def add_actions(menu,actions):
        for label,command,enabled in actions:menu.add_command(label=label,command=command,state='normal' if enabled else 'disabled')

    def _identity_items(self,menu,scope):
        self.add_actions(menu,self.photo_actions(scope))

    def choose_photo(self,scope=None):
        """One read-only catalogue; only Enter/double-click commits a choice."""
        scope=scope or self.photo_scope()
        old=getattr(self,'photo_picker',None)
        if old is not None and old.winfo_exists():old.lift();return old
        session=self.h.session;revision=session.review.review_revision
        try:
            self.ref.verify();catalog=[];thumbnails=getattr(self,'_gt_thumbnails',{})
            for sha,p in sorted(self.ref.photos.items(),key=lambda item:item[1]['global_position']):
                if sha not in thumbnails:
                    path=self.ref.photo_root/p['sciezka_zrodla']
                    if data.sha(path)!=sha:raise ValueError('Niezgodne źródło zdjęcia GT.')
                    with Image.open(path) as image:im=ImageOps.exif_transpose(image).convert('RGB')
                    im.thumbnail((84,56),Image.Resampling.LANCZOS);thumbnails[sha]=ImageTk.PhotoImage(im,master=self.h.window)
                caption=self.photo_caption(sha)
                catalog.append((sha,caption,(caption+' '+Path(p['sciezka_zrodla']).stem).casefold()))
            self._gt_thumbnails=thumbnails
        except (OSError,ValueError,KeyError) as exc:self.h.status_var.set('Nie otwarto zdjęć GT: '+str(exc));return
        win=tk.Toplevel(self.h.window);self.photo_picker=win;win.title('Wybierz właściwe zdjęcie');win.transient(self.h.window)
        win.geometry('660x650');win.minsize(500,420);win.columnconfigure(0,weight=1);win.rowconfigure(3,weight=1)
        ttk.Label(win,text='WYBIERZ WŁAŚCIWE ZDJĘCIE',font=('Segoe UI',11,'bold')).grid(row=0,column=0,sticky='w',padx=12,pady=(12,3))
        ttk.Label(win,text=self.scope_text(scope)).grid(row=1,column=0,sticky='w',padx=12)
        search=ttk.LabelFrame(win,text='Szukaj zdjęcia lub numeru tablicy');search.grid(row=2,column=0,sticky='ew',padx=12,pady=8)
        query=tk.StringVar(master=win);entry=ttk.Entry(search,textvariable=query);entry.pack(fill='x',padx=5,pady=5)
        body=ttk.Frame(win);body.grid(row=3,column=0,sticky='nsew',padx=12);body.rowconfigure(0,weight=1);body.columnconfigure(0,weight=1)
        style='A3.PhotoPicker.Treeview';ttk.Style(win).configure(style,rowheight=66,font=('Segoe UI',10))
        tree=ttk.Treeview(body,show='tree',selectmode='browse',style=style);tree.grid(row=0,column=0,sticky='nsew');tree.column('#0',width=580,minwidth=300)
        scroll=ttk.Scrollbar(body,orient='vertical',command=tree.yview);scroll.grid(row=0,column=1,sticky='ns');tree.configure(yscrollcommand=scroll.set)
        for sha,caption,_ in catalog:tree.insert('','end',iid=sha,text=caption,image=thumbnails[sha])
        notice=tk.StringVar(master=win,value='40 zdjęć · Dwuklik lub Enter wybiera zdjęcie. Esc anuluje.')
        ttk.Label(win,textvariable=notice,wraplength=620).grid(row=4,column=0,sticky='w',padx=12,pady=8)
        def close():
            win.grab_release();win.destroy();self.photo_picker=None
        def commit(sha=None,decision='CORRECTED'):
            if self.h.session is not session or session.review.review_revision!=revision:
                notice.set('Ocena zmieniła się. Zamknij wybór i otwórz go ponownie.');return
            if self._identity_action(scope,decision,sha,parent=win):close()
            else:notice.set(self.h.status_var.get());win.grab_set()
        def activate(event=None):
            iid=tree.identify_row(event.y) if event is not None and getattr(event,'num',None)==1 else next(iter(tree.selection()),None)
            if iid in tree.get_children(''):commit(iid)
            return 'break'
        def filter_rows(*_):
            term=''.join(c for c in query.get().casefold() if c.isalnum())
            number=int(query.get().strip()) if query.get().strip().isdigit() and 1<=int(query.get().strip())<=len(catalog) else None
            for sha,caption,words in catalog:
                match=self.ref.photos[sha]['global_position']==number if number is not None else term in ''.join(c for c in words if c.isalnum())
                if match:tree.move(sha,'','end')
                else:tree.detach(sha)
            hidden=[i for i in tree.selection() if i not in tree.get_children('')]
            if hidden:tree.selection_remove(*hidden)
            notice.set(f'{len(tree.get_children(""))} z 40 zdjęć · Dwuklik lub Enter wybiera zdjęcie. Esc anuluje.')
        query.trace_add('write',filter_rows);tree.bind('<Double-1>',activate);tree.bind('<Return>',activate)
        actions=ttk.Frame(win);actions.grid(row=5,column=0,sticky='ew',padx=12,pady=(0,12))
        reject=ttk.Button(actions,text='To nie jest żadne z 40 zdjęć',command=lambda:commit(decision='REJECTED'));reject.pack(side='left')
        ttk.Button(actions,text='Anuluj',command=close).pack(side='right')
        win.protocol('WM_DELETE_WINDOW',close);win.bind('<Escape>',lambda e:close());win.grab_set();entry.focus_set()
        win.tree=tree;win.query=query;win.search_entry=entry;win.reject_button=reject;win.close=close
        return win

    def identity_menu(self,event=None):
        if not self.active:return
        menu=tk.Menu(self.h.window,tearoff=False);scope=self.current_frame if self.selection_kind=='frame' else self.current_exposure or self.current_frame
        if scope:self._identity_items(menu,scope)
        return self._popup(menu,event)

    def tree_menu(self,event):
        if not self.active:return
        iid=self.h.subject_tree.identify_row(event.y)
        if iid not in self.targets:return 'break'
        self.select(iid);return self.menu(event)

    def _popup(self,menu,event=None):
        old=getattr(self,'_menu',None)
        if old:old.destroy()
        self._menu=menu
        try:menu.tk_popup(event.x_root if event else self.more.winfo_rootx(),event.y_root if event else self.more.winfo_rooty()+25)
        finally:menu.grab_release()
        return 'break'

    @staticmethod
    def plate_label(n,p):
        return f"Tablica {n} · "+(display_text(p) or 'nieczytelna')+(' (fragment)' if text_kind(p)=='PARTIAL_TEXT' else '')

    def selected_aid(self,event=None):
        gid=self.h.current_invocation
        if gid not in self.h.session.mt_invocations:return None
        oid=self.ed.selected
        if event and getattr(event,'widget',self.h.canvas) is self.h.canvas:
            hit=self.ed.hit(event.x,event.y,prefer_stack=True)
            if hit:oid=hit[0];self.ed.selected=oid
        aid=next((o.get('attempt_id') for o in (self.ed._read() or {}).get('objects',[]) if o['object_id']==oid),None)
        rows=self.detections(gid);ids=[r['id'] for r in rows]
        return aid if aid in ids else ids[0] if len(ids)==1 else None

    def choose_plate(self,command):
        ident=data.identity(self.h.session,self.current_frame);plates=self.ref.plates(ident['photo_sha256'])
        if len(plates)==1:return command(plates[0]['object_id'])
        menu=tk.Menu(self.h.window,tearoff=False)
        for n,p in enumerate(plates,1):menu.add_command(label=self.plate_label(n,p),command=lambda pid=p['object_id']:command(pid))
        if not plates:menu.add_command(label='GT: brak tablic — sprawdź zdjęcie',command=self.show_photo)
        return self._popup(menu)

    def judge(self,gid,aid,outcome,gt_id=None):
        session=self.h.session;row=next((r for r in self._progress()[gid]['rows'] if r['attempt_id']==aid),{})
        if row.get('operational_outcome')==outcome and row.get('gt_id')==gt_id:
            self.h.status_var.set('Ocena jest już zapisana.');return
        if outcome=='TP' and any(r['attempt_id']!=aid and r['operational_outcome']=='TP' and r['gt_id']==gt_id for r in self._progress()[gid]['rows']):
            if not messagebox.askyesno('Ta sama tablica', 'Ta tablica ma już trafne wskazanie w tym MT. Zapisać to wskazanie jako duplikat?',parent=self.h.window):return
            outcome='DUPLICATE'
        reason=self._reason('Dlaczego tego wskazania nie można ocenić?') if outcome=='N/A' else ''
        if reason is None:return
        self.save(lambda:data.decide_detection(session,gid,aid,outcome,gt_id=gt_id,reason=reason,coverage='COVERED' if outcome=='TP' else None))

    def show_photo(self):
        if self.current_exposure:self.select('a3:exp:'+self.current_exposure)
        else:self.identity_menu()

    def primary(self,event=None):
        if event is not None and self.ed._is_typing_shortcut(getattr(event,'widget',None)):return None
        gid=self.h.current_invocation
        if gid not in self.h.session.mt_invocations:
            if self.selection_kind=='exposure' and self.current_exposure and self.photo_done(self.ref.exposures[self.current_exposure]):return 'break'
            if self.current_frame and not data.identity(self.h.session,self.current_frame)['photo_sha256']:
                self._identity_action(self.current_frame,'N/A');return 'break'
            return self.confirm_identity(event)
        if not data.identity(self.h.session,self.current_frame)['approved']:self.show_photo();return 'break'
        unit=self._progress()[gid];aid=self.selected_aid()
        if unit['status']=='EXECUTION_UNASSESSABLE':self.mark_unit_na();return 'break'
        if any(r['attempt_id']==aid and r['operational_outcome']!='UNRESOLVED' for r in unit['rows']):return 'break'
        if aid and data.detection_role(self.h.session,gid,aid)=='SUPPRESSED':self.judge(gid,aid,'N/A');return 'break'
        if aid:self.choose_plate(lambda pid:self.judge(gid,aid,'TP',pid))
        elif self._progress()[gid]['pending']==['EMPTY_UNIT_INSPECTION']:self.save(lambda:data.confirm_empty_unit(self.h.session,gid))
        elif not unit['complete']:self.menu()
        return 'break'

    def select_detection(self,aid):
        obj=next((o for o in (self.ed._read() or {}).get('objects',[]) if o.get('attempt_id')==aid),None)
        if obj:self.ed.selected=obj['object_id'];self.ed.sync();self.ed.request_overlay()

    def unit_review(self,gid):
        """Optional call-level ROI/FN decisions, without cascading menus."""
        session=self.h.session;ident=data.identity(session,self.ref.call_frame(session,gid));plates=self.ref.plates(ident['photo_sha256'])
        if not ident['approved'] or not plates:return
        win=tk.Toplevel(self.h.window);win.title('Ocena MT — tablice i ROI');win.transient(self.h.window)
        revision=session.review.review_revision;labels=[self.plate_label(n,p) for n,p in enumerate(plates,1)]
        ttk.Label(win,text=f"MT {self.mt_numbers[gid]} · Tablica GT",font=('Segoe UI',11,'bold')).pack(anchor='w',padx=12,pady=10)
        choice=ttk.Combobox(win,values=labels,state='readonly',width=48);choice.current(0)
        if len(plates)>1:choice.pack(fill='x',padx=12,pady=(0,8))
        else:ttk.Label(win,text=labels[0]).pack(anchor='w',padx=12,pady=(0,8))
        def close():win.grab_release();win.destroy()
        def apply(status):
            if self.h.session is not session or session.review.review_revision!=revision:
                self.h.status_var.set('Ocena zmieniła się. Otwórz ocenę MT ponownie.');close();return
            pid=plates[choice.current()]['object_id'];reason=self._reason('Dlaczego nie można ustalić pokrycia ROI?',parent=win) if status=='ROI_COVERAGE_UNPROVEN' else ''
            if reason is None:win.grab_set();return
            if self.save(lambda:data.decide_missing(session,gid,pid,confirm_coverage=True) if status=='FN' else data.decide_coverage(session,gid,pid,status,reason)):close()
            else:win.grab_set()
        for label,status in [('W ROI','COVERED'),('Poza ROI','OUTSIDE_ROI'),('Częściowo w ROI','PARTIAL_ROI'),('Nie można ustalić','ROI_COVERAGE_UNPROVEN'),('W ROI, ale pominięta przez MT','FN')]:
            ttk.Button(win,text=label,command=lambda value=status:apply(value)).pack(fill='x',padx=12,pady=3)
        ttk.Label(win,textvariable=self.h.status_var,wraplength=400).pack(fill='x',padx=12,pady=6)
        ttk.Button(win,text='Anuluj',command=close).pack(anchor='e',padx=12,pady=8)
        win.protocol('WM_DELETE_WINDOW',close);win.bind('<Escape>',lambda e:close());win.grab_set()
        return win

    def mt_actions(self,gid,aid):
        session=self.h.session;ident=data.identity(session,self.current_frame);group=session.mt_invocations[gid]
        enabled=ident['approved'] and group.executed and not group.cancelled and not group.execution_failed and data.detection_role(session,gid,aid)!='SUPPRESSED'
        return [('Trafne',lambda:self.choose_plate(lambda pid:self.judge(gid,aid,'TP',pid)),enabled),
                ('Błędne',lambda:self.judge(gid,aid,'FP'),enabled),
                ('Nie można ocenić',lambda:self.judge(gid,aid,'N/A'),ident['approved'])]

    def basic_actions(self,event=None):
        gid=self.h.current_invocation
        if gid not in self.h.session.mt_invocations:
            scope=self.photo_scope();return self.photo_actions(scope) if scope else []
        aid=self.selected_aid(event)
        if aid:return self.mt_actions(gid,aid)
        unit=self._progress()[gid]
        if unit['pending']==['EMPTY_UNIT_INSPECTION']:
            return [('Brak wskazań — zgadza się',lambda:self.save(lambda:data.confirm_empty_unit(self.h.session,gid)),True),('Nie można ocenić MT',self.mark_unit_na,True)]
        return [('Oceń tablice i ROI',lambda:self.unit_review(gid),data.identity(self.h.session,self.current_frame)['approved']),('Nie można ocenić MT',self.mark_unit_na,True)]

    def menu(self,event=None,*,show=True,unit=False):
        if not self.active:return False
        if unit:return self.unit_review(self.h.current_invocation)
        menu=tk.Menu(self.h.window,tearoff=False);self.add_actions(menu,self.basic_actions(event))
        if not show:return menu
        return self._popup(menu,event)

    def aux_menu(self,*,show=True):
        gid=self.h.current_invocation;session=self.h.session;menu=tk.Menu(self.h.window,tearoff=False)
        if gid in session.mt_invocations:
            ident=data.identity(session,self.current_frame);plates=self.ref.plates(ident['photo_sha256']);aid=self.selected_aid()
            menu.add_command(label='Ocena tablic i ROI…',command=lambda:self.unit_review(gid),state='normal' if ident['approved'] and plates else 'disabled')
            if aid:
                def na_text():
                    reason=self._reason('Dlaczego odczytu nie można ocenić?')
                    if reason:self.save(lambda:data.text_na(session,aid,reason))
                menu.add_command(label='Nie można ocenić odczytu',command=na_text)
            menu.add_command(label='Nie można ocenić całego MT',command=self.mark_unit_na)
            chosen=[self.targets[i][1] for i in self.h.subject_tree.selection() if i in self.targets and self.targets[i][0]=='mt']
            if len(chosen)>1:
                menu.add_command(label=f'Oceń zaznaczone MT ({len(chosen)}): trafne',command=lambda:self.choose_plate(lambda pid:self.save(lambda:data.batch_single_tp(session,chosen,pid))))
            menu.add_separator()
        elif len(self.seen)==len(self.exposures) and not all(self.photo_done(e) for e in self.exposures):
            menu.add_command(label='Potwierdź zdjęcia bloku',command=self.confirm_block);menu.add_separator()
        menu.add_command(label='Cofnij ocenę',command=self.undo);menu.add_command(label='Ponów ocenę',command=lambda:self.undo(True))
        if not show:return menu
        return self._popup(menu)

    def mark_missing(self,gid,pid):
        # One explicitly worded action, two independently stored decisions.
        self.save(lambda:data.decide_missing(self.h.session,gid,pid,confirm_coverage=True))
    def mark_unit_na(self):
        gid=self.h.current_invocation
        if gid not in self.h.session.mt_invocations:return
        reason=self._reason('Dlaczego tego MT nie można ocenić?')
        if reason:self.save(lambda:data.unit_na(self.h.session,gid,reason))

    def refresh_reference(self):
        if not self.active or not self.current_frame:return
        ident=data.identity(self.h.session,self.current_frame);photo=ident.get('photo_sha256')
        key=(photo,ident['decision'],self.h.session.review.review_revision)
        if key==self._thumbnail_key:return
        self._thumbnail_key=key
        if not photo:
            self.reference_label.configure(image='',text='Zdjęcie GT · '+self.PHOTO_STATUS[ident['decision']]);return
        record=self.ref.photos[photo];path=self.ref.photo_root/record['sciezka_zrodla']
        if data.sha(path)!=photo:self.reference_label.configure(image='',text='Nie można odczytać potwierdzonego zdjęcia GT.');return
        with Image.open(path) as image:im=ImageOps.exif_transpose(image).convert('RGB')
        w,h=im.size;im.thumbnail((260,165),Image.Resampling.LANCZOS);sx,sy=im.width/w,im.height/h
        draw=ImageDraw.Draw(im);lines=[]
        for n,p in enumerate(self.ref.plates(photo),1):
            q=p.get('localization_quad') or p.get('quad')
            color={'EXACT':'#38E4D3','CUT_OFF':'#FF9F43','OCCLUDED':'#CF8BFF','BLURRED':'#65B5FF','UNDETERMINED':'#B8C0CC'}[geometry_status(p)]
            if q:
                points=[(x*sx,y*sy) for x,y in q];draw.line(points+[points[0]],fill=color,width=2);draw.text(points[0],str(n),fill=color)
            lines.append(f"{n}. "+(display_text(p) or 'nieczytelny')+(' [fragment]' if text_kind(p)=='PARTIAL_TEXT' else '')+(' · geometria N/A' if p['quad'] is None else ''))
        self.photo=ImageTk.PhotoImage(im,master=self.h.window)
        prefix=self.PHOTO_STATUS[ident['decision']]
        self.reference_label.configure(image=self.photo,text=f"Zdjęcie · GT {record['global_position']} · {prefix}\n"+('\n'.join(lines) or 'Brak tablic'),wraplength=270)

    def update_header(self):
        if not self.active:return False
        s=data.state(self.h.session);done=sum(self.photo_done(e) for e in self.exposures);units=self._progress();mt_done=sum(u['complete'] for u in units.values())
        approved=bool(s['blocks'].get('approval:'+self.h.session.session_id))
        self.h.progress_label.configure(text=f"Zdjęcia: {done}/{len(self.exposures)}     MT: {mt_done}/{len(units)}")
        extras_done=all(data.identity(self.h.session,fid,s)['decision']!='PROPOSED' for fid,f in self.ref.frames.items() if f['session_id']==self.h.session.session_id and fid not in self.ref.frame_exposure)
        self.h.complete_button.configure(text='Sesja A3 zatwierdzona' if approved else 'Zatwierdź sesję A3',state='normal' if not approved and done==len(self.exposures) and mt_done==len(units) and extras_done else 'disabled')
        self.h.form_status_label.configure(text='Ocena sesji zapisana' if approved else 'Ocena w toku')
        return True

    def context(self):
        gid=self.h.current_invocation;units=self._progress();unit=units.get(gid);aid=self.selected_aid();ident=data.identity(self.h.session,self.current_frame) if self.current_frame else {}
        details='';note='Oceń, czy zdjęcie GT odpowiada klatce telefonu.'
        if unit:
            from ..ranking.mobile_canvas_review import _prediction
            rows=[r for r in unit['rows'] if not aid or r['attempt_id']==aid]
            names={'TP':'Trafne','FP':'Błędne','DUPLICATE':'Duplikat','N/A':'Nie można ocenić','UNRESOLVED':'Do sprawdzenia'}
            note='\n'.join('Wynik modelu: '+(_prediction(self.h.session,a)[0] or 'brak odczytu') for a in ([aid] if aid else [r['id'] for r in self.detections(gid)])) or 'MT: brak wskazań.'
            note+='\nOcena: '+(', '.join(names[r['operational_outcome']] for r in rows) or ('Nie można ocenić' if unit['status']=='N/A' else 'Do sprawdzenia'))
            if not ident.get('approved'):note+='\nNajpierw oceń zdjęcie.'
            elif unit['pending']:
                labels=[label for prefix,label in [('MT:','wskazania MT'),('ROI:','tablice i ROI'),('TP_OR_FN:','pominięte tablice'),('ROI_CONFLICT:','sprzeczna ocena ROI'),('EXECUTION_NA_REQUIRED','brak wyniku MT'),('EMPTY_UNIT_INSPECTION','brak tablic')] if any(p.startswith(prefix) for p in unit['pending'])]
                note+='\nDo sprawdzenia: '+', '.join(labels)+'.'
                if aid and any(r['attempt_id']==aid and r['operational_outcome']=='UNRESOLVED' for r in rows):note+='\nTrafne: właściwa tablica w ROI.'
            else:note+='\nOcena MT zapisana.'
            details=str(unit)+'\nMT: '+gid
        elif ident.get('decision')!='PROPOSED':note='Ocena zdjęcia zapisana.'+(' Nie można ustalić zdjęcia GT.' if not ident.get('approved') else '')
        if self.current_frame:details+='\n'+str(self.ref.frames[self.current_frame])
        details+='\n'+str(self.ref.binding)+'\nPolygon IoU / NMCE: N/A — brak GT w geometrii telefonu.'
        return dict(object=details,image=note,session='',session_metrics=None,session_metrics_key=None,locator_target=None)

    def decorate(self):
        ed=self.ed;contexts=ed.contexts;gid=self.h.current_invocation;unit=self._progress().get(gid);aid=self.selected_aid()
        # Reuse the existing decision area and its collapsed technical details.
        keep={contexts.titles['object'],contexts.details_button,contexts.object_details,*self.decision_buttons,self.scope_label,self.tools}
        for w in contexts.sections['object'].winfo_children():
            if w not in keep:w.pack_forget()
        contexts.sections['session'].pack_forget()
        for number in contexts.counters.values():number.master.master.pack_forget()
        ed.clean_exclude_button.pack_forget();ed.clean_restore_button.pack_forget();ed.clean_exclusion.pack_forget()
        indication=next((i for i,r in enumerate(self.detections(gid),1) if r['id']==aid),None) if unit else None
        contexts.titles['object'].configure(text='OCENA WSKAZANIA' if indication else 'OCENA MT' if unit else 'OCENA ZDJĘCIA')
        contexts.titles['image'].configure(text='MT' if unit else 'ZDJĘCIE')
        n=next((i for i,e in enumerate(self.exposures,1) if e['exposure_id']==self.current_exposure),None)
        self.viewport_chrome()
        if unit:
            self.scope_label.configure(text=f"MT {self.mt_numbers.get(gid,'—')}"+(f' · wskazanie {indication}' if indication else ''))
            contexts.sections['image'].pack(fill='x',pady=(0,6))
        elif self.current_frame:
            scope=self.photo_scope();st=data.state(self.h.session)
            saved=st['identities'].get(scope,{}) if scope in self.ref.exposures else data.identity(self.h.session,scope,st)
            proposed=self.ref.exposures[scope]['photo_sha256'] if scope in self.ref.exposures else self.ref.frames[scope]['photo_sha256']
            text=(f'Zdjęcie {n} z {len(self.exposures)}\n' if n else '')+self.scope_text(scope)+'\n\nPropozycja:\n'+self.photo_caption(proposed)
            text+='\n\n'+self.PHOTO_STATUS[saved.get('decision','PROPOSED')]
            if saved.get('decision')=='CORRECTED':text+='\nWybrane: '+self.photo_caption(saved['photo_sha256'])
            if scope in self.ref.exposures and self.ref.frame_key(self.current_frame) in st['identities']:text+='\nTa klatka ma osobną ocenę.'
            self.scope_label.configure(text=text);contexts.sections['image'].pack_forget()
        self.scope_label.pack(fill='x',pady=(4,6),before=contexts.details_button)
        actions=self.basic_actions()
        for index,button in enumerate(self.decision_buttons):
            if index>=len(actions):button.pack_forget();continue
            label,command,enabled=actions[index];button.configure(text=label,command=command,state='normal' if enabled and not self.h._saving else 'disabled')
            button.pack(fill='x',pady=(3,0),before=contexts.details_button)
        self.tools.configure(text='Ocena całego MT i historia…' if unit else 'Historia…')
        self.tools.pack(fill='x',pady=(4,0),before=contexts.details_button)
        contexts.details_button.configure(text='Szczegóły ▾' if contexts.details_open else 'Szczegóły ▸')
        self.update_header()

    def viewport_chrome(self):
        """Keep layout stable also when zoom/pan skips assessment rendering."""
        ed=self.ed;gid=self.h.current_invocation
        rows=self.detections(gid) if gid in self.h.session.mt_invocations else []
        aid=self.selected_aid();indication=next((i for i,r in enumerate(rows,1) if r['id']==aid),None)
        n=self.ref.exposures[self.current_exposure]['planned_position'] if self.current_exposure else None
        caption=f'Zdjęcie {n}' if n else 'Poza zdjęciami GT'
        if gid in self.h.session.mt_invocations:caption+=f" · MT {self.mt_numbers.get(gid,'—')}"
        ed.clean_caption.configure(text=caption+' · Klatka telefonu');ed.clean_gt_label.configure(text='');ed.source_hud.grid_remove()
        ed.navigator.hint.grid_remove();ed.clean_bar.rowconfigure(2,minsize=0);ed.navigator.layer.grid_remove()
        if len(rows)>1:
            ed.navigator.controls.grid();ed.navigator.result.configure(text=f'Wskazanie {indication or "—"} / {len(rows)}')
        else:ed.navigator.controls.grid_remove()
        for item in ed.c.find_withtag('tv_bbox_domain'):
            if ed.c.type(item)=='text':ed.c.itemconfigure(item,state='hidden')
        for tag in ('stack_count_badge','stack_position_badges'):ed.c.itemconfigure(tag,state='hidden')
        ed._stack_counter=None

    def painted(self):
        if not self.active or not self.current_frame or self.h._image is None:return
        self.viewport_chrome()
        f=self.ref.frames[self.current_frame]
        if self.h._image_entry!=f['full_frame_entry']:return
        if not getattr(self.h,'_viewport_paint',False):
            if self.current_exposure:self.seen.add(self.current_exposure)
            self.refresh_reference();self.update_header()
        c=self.h.canvas;c.delete('a3_source_mp')
        scene=self.h.session.scene_frames.get(f['scene_event_id'],{})
        observations=[m for m in scene.get('mp_observations',[]) if m['source_stamp']==f['source_stamp']]
        rois=[m['selected_rois'] for m in observations]
        if not rois or any(r!=rois[0] for r in rois):return
        for r in rois[0]:
            p=self.ed.to_canvas((r['left'],r['top']),image_size=self.h._image_source_size);q=self.ed.to_canvas((r['right'],r['bottom']),image_size=self.h._image_source_size)
            c.create_rectangle(*p,*q,outline='#D4A03A',dash=(5,3),tags=('a3_source_mp',))
        c.tag_raise('a3_source_mp','evidence_image')

def install(h,header,left):
    index=getattr(h.report,'experiment_index',{}) or {}
    if index.get('series_id')!=data.SERIES:return
    # A saved A2 binding is archival context, not permission to replace the
    # ordinary verifier. Retain an explicit opt-in for inspecting the adapter.
    if getattr(h.browser,'_f5a_assistance_requested',False):
        h._f5a_a3=AssistedReview(h,header,left)
    else:
        ed=h._canvas_editor;ed.manual_gt_first=True;ed.inf_visible.set(False)
        ed.clean_menu.entryconfigure(1,label='Pokaż wyniki modeli (MT/MZ)')
