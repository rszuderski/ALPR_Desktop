"""F5A exposure adapter for the accepted Z4 GT Canvas and mobile sidecar.

No new Canvas or gesture controller. The standard per-call verifier and A3
remain intact; the report entry selects this scene workflow when A2 is proven.
"""
from pathlib import Path
import copy
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageOps, ImageTk
from .z4_photo_reference import PhotoReferenceWindow, PhotographCanvasSource
from ..ranking import mobile_f5a_scene_gt as data, mobile_f5a_assisted as a2
from ..ranking import photo_reference as photo


def timeline_records(index):
    return data.timeline_records(index)


class SceneCanvasSource(PhotographCanvasSource):
    def install(self,ed,viewer):
        super().install(ed,viewer)
        # The early S selection bindtag must let the Canvas Ctrl+S save through.
        for key in ('s','S'):
            ed.c.bind_class(ed.shortcut_tag,'<KeyPress-'+key+'>',
                lambda event:None if event.state & 4 else ed.selection_key_press(event))

    def scene(self):
        preview=self.h.preview
        if self.h.operator_mode and not preview:
            size=list(self.h._image.size) if self.h._image is not None else [1,1]
            return dict(scope='F5A_A1_READ_ONLY',image_entry=self.h._image_entry,image_size=size,
                roi=[0,0,*size],objects=[],parents=[],presence='UNRESOLVED',checked=False,exclusion_reason='')
        if not preview:return super().scene()
        if not preview.get('frame_id'):
            return dict(scope='F5A_SOURCE_INSPECTION',image_entry='',image_size=[1,1],roi=[0,0,1,1],
                objects=[],parents=[],presence='UNRESOLVED',checked=False,exclusion_reason='Brak klatki')
        f=self.data.index.verify_frame(preview['frame_id'])
        return dict(scope='F5A_SOURCE_INSPECTION',image_entry=f['full_frame_entry'],image_size=f['image_size'],
            roi=[0,0,*f['image_size']],objects=[],parents=[],presence='UNRESOLVED',checked=False,exclusion_reason='')

    @property
    def read_only(self):return self.h.operator_mode or bool(self.h.preview)
    def available(self):
        if self.h.operator_mode:return self.key in self.data.index.exposures and not self.h._saving
        if self.key not in self.data.index.exposures or self.h._image is None:
            self.h.status_var.set('Nie wczytano klatki CameraX. Wybierz zdjęcie z listy, aby ponowić odczyt.')
            return False
        return super().available()
    def edit(self,oid,**kwargs):
        if self.read_only:raise ValueError('Klatka dowodowa jest tylko do odczytu.')
        return super().edit(oid,**kwargs)
    def outline(self,quad):
        if self.read_only:raise ValueError('Wróć do GT sceny, aby rysować obrys.')
        return super().outline(quad)
    def sync(self):
        changed=self.ed.skey!=self.key
        super().sync()
        if changed and not self.h.preview and self.scene()['objects']:
            self.ed.selected=self.scene()['objects'][0]['object_id'];self.ed.mode='edit'
    def refresh_list(self):self.h.refresh_scenes()
    def caption(self):
        if self.h.operator_mode:
            self.h.image_help.configure(text='GT A1 i dowody tylko do odczytu · Spacja: potwierdź ocenę · Ctrl+Z/Y: cofnij/ponów · rolka: zoom · przeciągnij tło: przesuń')
            return
        if self.h.preview:
            self.h.image_help.configure(text='Dowód źródłowy · geometria GT nie jest przenoszona. PPM na wskazaniu MT: rozstrzygnij wyjątek.')
        else:
            super().caption()
            if self.ed.mode!='draw':self.h.image_help.configure(text='D: obrys tablicy · PPM: tekst/geometria · Spacja: potwierdź tablice i czytelność na tej klatce · Dalej: bez automatycznej oceny')
    def confirm(self,event=None):
        if event is not None and self.ed._is_typing_shortcut(getattr(event,'widget',None)):return None
        if self.h.operator_mode and event is not None and event.state & 4:
            if self.pending_action:self.retry()
            else:self.h.status_var.set('Ocena zapisuje się automatycznie po decyzji. Spacja: Potwierdź ocenę.')
            return 'break'
        if self.h.operator_mode:return self.h.decide('CONFIRMED')
        if self.h.preview or not self.available():return 'break'
        key=self.key
        if self.ed.prepare_navigation():self.commit(lambda:self.data.approve(key))
        return 'break'
    def empty(self):
        if self.h.operator_mode:return 'break'
        if not self.h.preview and self.available():return super().empty()
        return 'break'
    def unreadable(self,oid=None):
        if self.h.operator_mode:return 'break'
        oid=oid or (self.ed.entry_target[1] if self.ed.entry_target else self.ed.selected)
        if oid and self.ed.finish_entry():
            self.commit(lambda:self.data.edit(self.key,oid,text='',text_state='UNREADABLE_TEXT',reason='Numer nieczytelny na tej klatce CameraX'))
        return 'break'
    def commit(self,action):
        if not self.available():return False
        result=super().commit(action)
        if result is False:self.refresh_list()
        else:self.h.show_result_status()
        return result
    def context_menu(self,event=None,*,show=True):
        if not self.available():return 'break'
        if self.h.operator_mode:return self.h.operator_menu(event,show=show)
        if self.h.preview:return self.h.exception_menu(event,show=show)
        return super().context_menu(event,show=show)
    def navigate(self,direction):
        if self.key not in self.data.index.exposures:
            self.h.select(self.data.material.order[0]);return 'break'
        return super().navigate(direction)
    def restore(self,*,redo=False):
        if self.h.operator_mode:
            if self.ed.prepare_navigation():
                eid=self.key;ctx=self.h._nav_target
                self.commit(lambda:self.data.undo_exposure(eid,redo=redo,context=ctx))
            return 'break'
        if self.h.preview and self.h.preview['frame_id'] not in self.data.index.ref.frame_exposure:
            self.h.status_var.set('Ten dowód nie ma przypisanej sceny GT.');return 'break'
        return super().restore(redo=redo)

    def remove(self):
        return 'break' if self.h.operator_mode else super().remove()


class SceneReviewWindow(PhotoReferenceWindow):
    def __init__(self,browser,mobile,reference,*,operator_mode=True):
        self.operator_mode=operator_mode;self.viewed={};self._original_cache={};self._accounting_cache={}
        self.browser=browser;self.mobile=mobile;self.preview=None;self.targets={};self.show_exception_rows=False
        self._results_key=None;self._scene_widgets_ready=False;self._nav_target=None
        store=data.SceneReference(mobile,reference)
        self.evidence=timeline_records(store.index)
        super().__init__(browser.window,store.material,reference=store,source_factory=SceneCanvasSource)
        self.window.title('F5A — GT sceny · Z4')
        meta=reference.sessions[mobile.session_id]
        self.header_title.configure(text=f"{meta['variant']} · Powtórzenie {meta['replicate']} · Blok {meta['block'][-1]}")
        if operator_mode:
            if store.protocol_approval():self.browser._f5a_operator_protocol=copy.deepcopy(store.protocol_approval())
            # Stable text slots: changing evidence must not resize the Canvas
            # during the accepted wheel animation (which cancels on resize).
            self.canvas.master.grid_slaves(row=0,column=0)[0].configure(height=4)
            self.status_label.configure(height=4)
            from .mobile_report_operator_metadata import phone_session_metadata,phone_note_caption
            note=phone_note_caption(phone_session_metadata(mobile.bundle.report,mobile.bundle))
            self.header_title.configure(text=self.header_title.cget('text')+'\n'+note,font=('Segoe UI',11,'bold'),justify='left',wraplength=850)
            for w in self.toolbar.winfo_children():
                if w.cget('text') in ('Dodaj tablicę (D)','Brak tablic','Następne niegotowe'):w.pack_forget()
            self.frozen=data.frozen_first_input(reference)
        self.left.configure(width=280)
        self.subject_tree.configure(show='tree',displaycolumns=('status',))
        self.subject_tree.column('#0',width=175,minwidth=140);self.subject_tree.column('status',width=80,minwidth=65,stretch=False)
        self.reference_label=tk.Label(self.left,bg=self.bg,fg=self.fg,compound='top',justify='left',wraplength=265)
        self.reference_label.grid(row=1,column=0,columnspan=2,sticky='ew',pady=5)
        self.confirm_button=ttk.Button(self.toolbar,text='Potwierdź ocenę · Spacja' if operator_mode else 'Zatwierdź GT sceny · Spacja',command=self.source.confirm)
        self.confirm_button.pack(side='left',padx=3)
        if operator_mode:
            ttk.Button(self.toolbar,text='Zgłoś błąd',command=lambda:self.decide('ERROR_REPORTED')).pack(side='left',padx=3)
            ttk.Button(self.toolbar,text='Nie można ocenić',command=lambda:self.decide('NOT_ASSESSABLE')).pack(side='left',padx=3)
        self.export_button=ttk.Button(self.toolbar,text='Eksport',command=self.export)
        self.export_button.pack(side='right',padx=3)
        self.pending_button.configure(text='Sprawdź wyjątki',command=self.show_exceptions)
        self._build_timeline()
        self._scene_widgets_ready=True;self.refresh_scenes();self.refresh_reference();self._sync_navigation(('exposure',self.current_invocation))
        if operator_mode:
            self.show_result_status()
            context=data.exposure_review_state(store.data)['last_context']
            if context and store.valid_context(context):
                iid=next((i for i,t in self.targets.items() if tuple(context)==t),None)
                if iid:self.subject_tree.selection_set(iid);self._selected()

    def results(self):
        self.session.index.ref.verify();self.mobile.verify_source()
        if self._results_key!=self.session.data['revision']:
            self._results=data.analyze(self.session);self._results_key=self.session.data['revision']
        return self._results

    def _build_timeline(self):
        strip=tk.Frame(self.canvas.master,bg=self.bg)
        strip.grid(row=2,column=0,sticky='ew');strip.columnconfigure(0,weight=1)
        head=tk.Frame(strip,bg=self.bg);head.grid(row=0,column=0,sticky='ew')
        tk.Label(head,text='Przebieg · zdjęcia wg A2 / kolejność klatek CameraX',bg=self.bg,fg=self.fg,
            font=('Segoe UI',9)).pack(side='left',padx=6)
        ttk.Button(head,text='Czasy szczegółowe',command=self._open_timing).pack(side='right')
        self.details_button=ttk.Button(head,text='Szczegóły ▸',command=self._toggle_details)
        self.details_button.pack(side='right')
        self.timeline_photos=tk.Canvas(strip,height=27,bg=self.bg,highlightthickness=0)
        self.timeline_photos.grid(row=1,column=0,sticky='ew')
        self.timeline=tk.Canvas(strip,height=112,bg='#f8faf5',highlightthickness=0)
        self.timeline.grid(row=2,column=0,sticky='ew')
        scroll=ttk.Scrollbar(strip,orient='horizontal',command=self.timeline.xview)
        scroll.grid(row=3,column=0,sticky='ew');self.timeline.configure(xscrollcommand=scroll.set)
        self.timeline_details=tk.Label(strip,bg=self.bg,fg=self.muted,justify='left',anchor='w',font=('Segoe UI',9))
        self.timeline_details.grid(row=4,column=0,sticky='ew',padx=6);self.timeline_details.grid_remove()
        self._details_open=False;self._timeline_hits=[];self._photo_hits=[]
        self.timeline_photos.bind('<Configure>',lambda e:self._draw_timeline())
        self.timeline.bind('<Button-1>',lambda e:self._timeline_click(e,False))
        self.timeline_photos.bind('<Button-1>',lambda e:self._timeline_click(e,True))
        self.subject_tree.tag_configure('linked',background='#dfebd9')

    def _timeline_click(self,event,photos):
        canvas=self.timeline_photos if photos else self.timeline
        x=canvas.canvasx(event.x);y=canvas.canvasy(event.y)
        for left,top,right,bottom,target in (self._photo_hits if photos else self._timeline_hits):
            if left<=x<=right and top<=y<=bottom:
                iid=next((i for i,t in self.targets.items() if t==target),None)
                if iid:
                    self.subject_tree.selection_set(iid);self.subject_tree.see(iid);self._selected()
                return 'break'

    def _sync_navigation(self,target):
        if not self._scene_widgets_ready:return
        self._nav_target=target
        eid=self._target_exposure(target)
        if eid and self.session.valid_context(target):self.viewed.setdefault(eid,set()).add(tuple(target))
        iid=next((i for i,t in self.targets.items() if t==target),None)
        if iid:
            if self.subject_tree.selection()!=(iid,):self.subject_tree.selection_set(iid)
            self.subject_tree.focus(iid);self.subject_tree.see(iid)
        self._draw_timeline();self._update_timing_link()

    def _target_exposure(self,target):
        kind,key,*rest=target
        if kind in ('exposure','folder'):return key
        if kind=='frame':return self.evidence['frames'][key]['exposure_id']
        if kind in ('call','attempt','mz','consensus'):return self.evidence['calls'][key]['exposure_id']
        return None

    def _draw_timeline(self):
        if not self._scene_widgets_ready:return
        target=self._nav_target or ('exposure',self.current_invocation);eid=self._target_exposure(target)
        for iid,t in self.targets.items():
            self.subject_tree.item(iid,tags=('linked',) if eid and self._target_exposure(t)==eid else ())
        pc=self.timeline_photos;pc.delete('all');self._photo_hits=[]
        order=self.session.material.order;extra=any(not r['exposure_id'] for r in self.evidence['frames'].values()) or any(not r['exposure_id'] for r in self.evidence['calls'].values())
        width=max(400,pc.winfo_width());step=width/(len(order)+int(extra))
        for n,key in enumerate(order+([None] if extra else [])):
            x=n*step;text=f'Zdjęcie {n+1}' if key else 'N/A'
            pc.create_rectangle(x+2,2,x+step-2,25,fill='#c9dfbd' if key==eid else '#e7ebdf',outline='#94a487')
            pc.create_text(x+step/2,13,text=text,fill=self.fg,font=('Segoe UI',9))
            self._photo_hits.append((x,0,x+step,27,('exposure',key) if key else ('extra',None)))
        c=self.timeline;c.delete('all');self._timeline_hits=[]
        frames=[(f,r) for f,r in self.evidence['frames'].items() if r['exposure_id']==eid]
        frames.sort(key=lambda item:(item[1].get('capture_elapsed_nanos') is None,item[1].get('capture_elapsed_nanos') or 0,item[0]))
        times=[r.get('capture_elapsed_nanos') for r in self.evidence['frames'].values() if type(r.get('capture_elapsed_nanos')) is int]
        origin=min(times) if times else None
        for y,name in [(14,'Klatka'),(47,'MT'),(78,'MZ')]:c.create_text(5,y,text=name,anchor='w',fill=self.muted,font=('Segoe UI',9))
        def chip(x,y,w,label,t,warning=False):
            active=t==target;kind=t[0]
            confirmed=kind in ('mz','consensus') and self.evidence['readouts'][t[2]]['confirmed'] is True
            fill='#f2b1b1' if warning else '#b9dfbb' if confirmed else '#d3d7da' if kind=='frame' else '#8cdbd4' if kind=='call' else '#ffd397'
            c.create_rectangle(x,y,x+w,y+22,fill=fill,outline='#153d7c' if active else '#b0bac0',width=3 if active else 1)
            c.create_text(x+w/2,y+11,text=label,fill=self.fg,font=('Segoe UI',9))
            self._timeline_hits.append((x,y,x+w,y+22,t))
        x=55;call_numbers={g:i for i,g in enumerate((g for g,r in self.evidence['calls'].items() if r['exposure_id']==eid),1)}
        for n,(fid,f) in enumerate(frames,1):
            groups=f['calls'];mids=[a for g in groups for a in self.evidence['calls'][g]['readout_ids']
                if self.evidence['readouts'][a]['executed'] or any(self.evidence['readouts'][a]['consensus'])]
            w=max(112,85*len(groups),85*len(mids));stamp=f.get('capture_elapsed_nanos')
            seconds=f' · {(stamp-origin)/1e9:.1f} s' if type(stamp) is int and origin is not None else ' · czas N/A'
            mixed=self.evidence['events'].get(f.get('scene_event_id'),{}).get('multiple_photographs_observed',False)
            chip(x,2,w-10,f'Klatka {n}'+seconds+(' ≠' if mixed else ''),('frame',fid))
            for j,gid in enumerate(groups):
                pos=x+j*(w-10)/len(groups);cw=(w-10)/len(groups)
                late=self.evidence['calls'][gid]['stale'] or self.evidence['calls'][gid]['failed']
                chip(pos,35,cw-3,f'MT {call_numbers[gid]}'+(' !' if late else ''),('call',gid),late)
            for j,aid in enumerate(mids):
                m=self.evidence['readouts'][aid];pos=x+j*(w-10)/len(mids);cw=(w-10)/len(mids)
                kind='mz' if m['executed'] else 'consensus'
                label=f'MZ {j+1}' if m['executed'] else 'Pamięć'
                chip(pos,66,cw-3,label+(' !' if m['stale'] else ' ✓' if m['confirmed'] is True else ''),(kind,m['gid'],aid),m['stale'])
            if not groups:c.create_text(x+(w-10)/2,46,text='brak MT',fill=self.muted,font=('Segoe UI',8))
            elif not mids:c.create_text(x+(w-10)/2,77,text='MZ nie wykonano' if all(self.evidence['calls'][g]['mz_not_run'] for g in groups) else 'MZ: brak dowodu',fill=self.muted,font=('Segoe UI',8))
            x+=w
        # Calls without a proven frame remain selectable and never borrow a bitmap.
        for gid,r in self.evidence['calls'].items():
            if r['exposure_id']==eid and not r['frame_id']:
                chip(x,35,110,f'MT {call_numbers[gid]} · N/A',('call',gid),True)
                for j,aid in enumerate(r['mz']):chip(x+j*115,66,110,'MZ · N/A',('mz',gid,aid),True)
                x+=max(115,len(r['mz'])*115)
        if not frames and x==55:c.create_text(65,46,anchor='w',text='Brak klatek potwierdzonych w A2 · N/A',fill=self.muted)
        c.create_text(55,109,anchor='sw',text='Klatki: czas pobrania. MT/MZ: relacja po ID. ✓ potwierdzenie Androida, nie zgodność z GT; ! anulowanie.',fill=self.muted,font=('Segoe UI',8))
        c.configure(scrollregion=(0,0,max(x,c.winfo_width()),112))
        selected=next((hit for hit in self._timeline_hits if hit[-1]==target),None)
        if selected and (selected[0]<c.canvasx(0) or selected[2]>c.canvasx(c.winfo_width())):
            c.xview_moveto(max(0,selected[0]-55)/max(x,c.winfo_width()))
        self._refresh_details()

    def _toggle_details(self):
        self._details_open=not self._details_open
        self.details_button.configure(text='Szczegóły ▾' if self._details_open else 'Szczegóły ▸')
        if self._details_open:self.timeline_details.grid();self._refresh_details()
        else:self.timeline_details.grid_remove()

    def _selected_trace_ids(self):
        t=self._nav_target or ('exposure',self.current_invocation);kind,key,*rest=t
        if kind in ('mz','consensus'):return self.evidence['readouts'][rest[0]]['trace_ids']
        if kind in ('call','attempt'):return self.evidence['calls'][key]['trace_ids']
        if kind=='frame':gids=self.evidence['frames'][key]['calls']
        else:gids=[g for g,r in self.evidence['calls'].items() if r['exposure_id']==self._target_exposure(t)]
        return list(dict.fromkeys(v for g in gids for v in self.evidence['calls'][g]['trace_ids']))

    def _refresh_details(self):
        if not self._details_open:return
        target=self._nav_target or ('exposure',self.current_invocation);kind,key,*rest=target
        eid=self._target_exposure(target);lines=[]
        if eid:
            e=self.session.index.exposures[eid]
            lines.append('Zdjęcie: '+e.get('relative_path',eid)+' · SHA-256: '+e['photo_sha256'])
        fid=key if kind=='frame' else self.evidence['calls'][key]['frame_id'] if kind in ('call','attempt','mz','consensus') else None
        if fid:
            f=self.evidence['frames'][fid];stamp=f['source_stamp'];event=self.evidence['events'].get(f.get('scene_event_id'),{})
            lines.extend(['Klatka: '+f['full_frame_entry']+' · SHA-256: '+f['full_frame_sha256'],
                f"Pobranie: {f.get('capture_elapsed_nanos')} ns · sensor: {stamp.get('source_timestamp_nanos')} ({stamp.get('source_timestamp_domain')})",
                'Zdarzenie: '+str(f.get('scene_event_id','N/A'))+(' · obejmuje dwie fotografie; przypisanie tylko po klatce A2' if event.get('multiple_photographs_observed') else '')])
        if kind in ('call','attempt','mz','consensus'):
            r=self.evidence['calls'][key];lines.append('MT: '+key+(' · N/A: '+r['reason'] if r['reason'] else ''))
            if r['stale']:lines.append('MT oznaczone w archiwum jako spóźnione/anulowane; pozostaje przy swoim wejściu.')
            if kind in ('mz','consensus'):
                m=self.evidence['readouts'][rest[0]]
                lines.append('MZ · odczyty: '+repr(m['readouts'])+' · próba: '+m['attempt_id'])
                lines.append('Konsensus: '+repr(m['consensus'])+' · potwierdzony: '+str(m['confirmed']))
                for event in m['events']:
                    lines.append(f"Zdarzenie {event['event_seq']} · zegar rejestratora {event.get('elapsed_ms')} ms · {event['event_type']}: "+
                        str(event.get('prediction_before',''))+' → '+str(event.get('prediction_after',event.get('prediction',''))))
                lines.append('Próbki źródłowe: '+(', '.join(m['sample_ids']) or 'odczyt w rekordzie próby'))
                if m['stale']:lines.append('Archiwum oznaczyło wynik jako spóźniony/anulowany: '+m['cancel_reason'])
        tids=self._selected_trace_ids();lines.append('ID pomiarów: '+(', '.join(tids) or 'N/A · brak jawnego powiązania'))
        missing=[t for t in tids if t not in self.evidence['traces']]
        if missing:lines.append('Brak punktu na wykresie dla ID: '+', '.join(missing))
        lines.append('Zmiana zdjęcia: A2 nie podaje chwili zmiany w zegarze CameraX. Opóźnienie względem zmiany: N/A bez jawnego dowodu.\nCzasy szczegółowe mają oddzielną skalę pomiarową; znacznik pomiaru nie jest czasem pobrania klatki.')
        self.timeline_details.configure(text='\n'.join(lines),wraplength=max(500,self.timeline.winfo_width()-15))

    def _update_timing_link(self,*,open_tab=False):
        bind=getattr(self.browser,'link_evidence_timing',None)
        if bind:return bind(self.mobile.review.source_archive_sha256,self._selected_trace_ids(),self._timing_selected,open_tab=open_tab)
        return False

    def _open_timing(self):
        if not self._update_timing_link(open_tab=True):
            self.status_var.set('Czasy szczegółowe: wybierz tę samą sesję w przeglądarce raportów.')

    def _timing_selected(self,trace_id,stage):
        if self._closed:return
        if stage.startswith('character'):
            targets=[('mz',m['gid'],a) for a,m in self.evidence['mz'].items() if trace_id in m['trace_ids']]
        else:targets=[('call',g) for g,r in self.evidence['calls'].items() if trace_id in r['trace_ids']]
        if len(targets)==1:
            iid=next(i for i,t in self.targets.items() if t==targets[0])
            self.subject_tree.selection_set(iid);self._selected();self.window.lift()
        elif targets:
            # A trace may aggregate two ROI calls; never choose one arbitrarily.
            fids={self.evidence['calls'][t[1]]['frame_id'] for t in targets}
            if len(fids)==1 and None not in fids:self.inspect(next(iter(fids)))
            self.status_var.set(f'Pomiar łączy {len(targets)} rekordy. Wybierz MT lub MZ na linii.');self.window.lift()
        else:self.status_var.set('N/A · punkt pomiarowy bez jawnego powiązania MT/MZ')

    def refresh_scenes(self):
        tree=self.subject_tree;current=tree.selection();self.targets={};wanted=set();store=self.session
        pending=self.current_invocation if self.source.pending_action else None
        def status(eid):return 'Niezapisane' if eid==pending else store.operator_status(eid) if self.operator_mode else store.status(eid)
        def node(iid,parent,label,target,status=''):
            wanted.add(iid);self.targets[iid]=target
            if tree.exists(iid):tree.item(iid,text=label,values=('',status))
            else:tree.insert(parent,'end',iid=iid,text=label,values=('',status),open=False)
        for n,eid in enumerate(store.material.order,1):
            node(eid,'',f'Zdjęcie {n} z {len(store.material.order)}',('exposure',eid),status(eid))
            folder='frames:'+eid;ids=store.index.frame_ids(eid)
            node(folder,eid,f'Klatki telefonu ({len(ids)})',('folder',eid))
            for i,fid in enumerate(ids,1):
                fid=self.evidence['aliases'][fid]
                node('frame:'+fid,folder,f'Klatka {i}',('frame',fid))
            for i,(gid,u) in enumerate(((g,r) for g,r in self.evidence['calls'].items() if r['exposure_id']==eid),1):
                node('call:'+gid,eid,f'MT {i}',('call',gid))
                for j,r in enumerate(u['source_candidates'],1):
                    node('attempt:'+r['attempt_id'],'call:'+gid,f'Wskazanie {j}',('attempt',gid,r['attempt_id']))
                for j,aid in enumerate(u['mz'],1):
                    m=self.evidence['mz'][aid]
                    node('mz:'+aid,'call:'+gid,f'MZ {j}'+(' ✓' if m['confirmed'] is True else ' !' if m['stale'] else ''),('mz',gid,aid))
                for aid,m in u['readouts'].items():
                    if not m['executed'] and any(m['consensus']):
                        node('consensus:'+aid,'call:'+gid,'Konsensus · bez nowego MZ',('consensus',gid,aid))
        extras=[f for f,r in self.evidence['frames'].items() if not r['exposure_id']]
        unknown=[g for g,r in store.index.calls.items() if not r['exposure_id']]
        if extras or unknown or store.index.diagnostics:
            node('extra','','Dowody poza zdjęciami',('extra',None),'N/A')
            for i,fid in enumerate(extras,1):node('frame:'+fid,'extra',f'Klatka {i}',('frame',fid),'N/A')
            for i,gid in enumerate(unknown,1):
                node('call:'+gid,'extra',f'MT {i}',('call',gid),'N/A')
                for j,aid in enumerate(self.evidence['calls'][gid]['mz'],1):
                    node('mz:'+aid,'call:'+gid,f'MZ {j}',('mz',gid,aid),'N/A')
            for i,r in enumerate(store.index.diagnostics,1):node('diagnostic:'+r['attempt_id'],'extra',f'Próba bez wywołania MT {i}',('diagnostic',r['attempt_id']),'N/A')
        def prune(parent=''):
            for iid in tree.get_children(parent):
                if iid not in wanted:tree.delete(iid)
                else:prune(iid)
        prune()
        if self.operator_mode:
            p=data.exposure_progress(store.data);self.progress.set(f"Oceniono {p['reviewed']} z {p['total']} zdjęć")
        else:self.progress.set(f"GT sceny: {sum(store.approved(k) and k!=pending for k in store.material.order)}/{len(store.material.order)}")
        if self.current_invocation in store.index.exposures:
            n=store.material.order.index(self.current_invocation)+1
            self.current_status.set(f'Zdjęcie {n} z {len(store.material.order)} · '+('GT oryginalnego zdjęcia' if self.operator_mode else 'GT na klatce telefonu')+' · '+status(self.current_invocation))
            self.empty_button.configure(text='Brak tablic ✓' if store.approved(self.current_invocation) and store.state(self.current_invocation)['presence']=='NO_PLATES' else 'Brak tablic')
        if current and tree.exists(current[0]):tree.selection_set(current)
        if self._scene_widgets_ready:
            self._draw_timeline()
            if self.preview:self.inspection_status()

    def _selected(self,event=None):
        choice=self.subject_tree.selection();target=self.targets.get(choice[0]) if choice else None
        if not target:return
        kind,key,*rest=target
        if target==self._nav_target:return
        if kind=='exposure':self.select(key)
        elif kind=='frame':self.inspect(key)
        elif kind in ('call','attempt','mz','consensus'):
            row=self.evidence['calls'][key]
            if row['frame_id']:self.inspect(row['frame_id'],gid=key,aid=rest[0] if rest else None,target=target)
            else:
                if not self._canvas_editor.prepare_navigation():
                    self._sync_navigation(self._nav_target);return
                self.preview=dict(frame_id=None,gid=key,aid=rest[0] if rest else None)
                if self.operator_mode:
                    self.current_invocation=row['exposure_id'] or ''
                    self.confirm_button.configure(state='normal' if row['exposure_id'] else 'disabled')
                self._image=None;self._image_source_size=None;self._image_entry='';self._image_token+=1
                self._view_center=None
                self.canvas.delete('all');self.current_status.set('N/A · brak udowodnionej klatki tego MT')
                self._paint_image();self.refresh_reference();self.status_var.set('N/A: '+row['reason']);self._sync_navigation(target)
        elif kind=='diagnostic':
            raw=next(r['source_record'] for r in self.session.index.diagnostics if r['attempt_id']==key)
            fid=next((f for f,r in self.session.index.frames.items() if r['full_frame_entry']==raw.get('source_frame_entry')
                and all(str(raw.get(k))==str(r['source_stamp'].get(k)) for k in a2.STAMP)),None)
            if fid:self.inspect(fid)
            self.status_var.set('Niewykonana próba bez identyfikatora wywołania MT · N/A, nie FN')
        elif kind=='extra':self._sync_navigation(target)

    def select(self,key,*,initial=False):
        # A resume cursor is view state, never a reason to save or approve GT.
        if initial and key not in self.session.index.exposures:key=self.session.material.order[0]
        if self.operator_mode:return self.select_original(key,initial=initial)
        if self.preview:
            if not self._canvas_editor.prepare_navigation():
                self._sync_navigation(self._nav_target);return False
            self.preview=None;self.current_invocation=''
        result=super().select(key,initial=initial)
        if self._scene_widgets_ready:
            self.confirm_button.configure(text='Zatwierdź GT sceny · Spacja',command=self.source.confirm,state='normal')
            self.refresh_reference()
            if result:self._sync_navigation(('exposure',key))
        return result

    def refresh_reference(self):
        if not self._scene_widgets_ready:return
        if self.preview and not self.evidence['frames'].get(self.preview.get('frame_id'),{}).get('exposure_id'):
            self.reference_label.configure(image='',text='GT oryginalnego zdjęcia\nN/A · brak przypisania klatki w A2')
            self._reference_photo=None;return
        if self.current_invocation not in self.session.index.exposures:return
        e=self.session.index.exposures[self.current_invocation];ref=self.session.index.ref
        sha=self.session.operator_photo(self.current_invocation) if self.operator_mode else e['photo_sha256']
        thumb=ImageOps.contain(self.original_image(sha),(250,135))
        self._reference_photo=ImageTk.PhotoImage(thumb,master=self.window)
        plates=ref.plates(sha)
        lines=[f"{i}. "+('Nieczytelna' if photo.text_kind(p)=='UNREADABLE_TEXT' else 'Fragment: '+photo.display_text(p) if photo.text_kind(p)=='PARTIAL_TEXT' else photo.display_text(p) or 'Tekst N/A') for i,p in enumerate(plates,1)]
        presence=ref.gt_states['img-sha256-'+sha]['presence']
        self.reference_label.configure(image=self._reference_photo,text='GT oryginalnego zdjęcia\n'+
            ('NO_PLATES · brak tablic' if presence=='NO_PLATES' else f'Tablice: {len(plates)}\n'+'\n'.join(lines)))

    def inspect(self,fid,*,gid=None,aid=None,target=None):
        fid=self.evidence['aliases'].get(fid,fid)
        target=target or (('attempt',gid,aid) if aid else ('call',gid) if gid else ('frame',fid))
        if self.preview==dict(frame_id=fid,gid=gid,aid=aid):self._sync_navigation(target);return
        if not self._canvas_editor.prepare_navigation():
            self._sync_navigation(self._nav_target);return
        same_image=self.preview and self._image_entry==self.session.index.frames[fid]['full_frame_entry']
        try:im=self._image if same_image else self.session.index.frame_image(fid)
        except (ValueError,OSError) as exc:
            self._sync_navigation(self._nav_target);self.status_var.set('Nie można otworzyć klatki: '+str(exc));return
        self.preview=dict(frame_id=fid,gid=gid,aid=aid)
        eid=self.session.index.ref.frame_exposure.get(fid)
        if eid:self.current_invocation=eid
        if not same_image:
            self._image=im;self._image_source_size=im.size;self._image_entry=self.session.index.frames[fid]['full_frame_entry'];self._image_token+=1
            self._view_zoom=1.;self._view_center=None
        self._canvas_editor.escape();self._paint_image()
        self.refresh_reference()
        if not self.operator_mode:
            if gid or not eid:self.confirm_button.configure(text='Wróć do GT sceny',command=lambda:self.select(self.current_invocation))
            else:self.confirm_button.configure(text='Użyj tej klatki do GT',command=lambda:self.choose_frame(eid,fid))
        else:self.confirm_button.configure(state='normal' if eid else 'disabled')
        self._sync_navigation(target)
        self.status_var.set('Klatka telefonu · tylko odczyt · bez zatwierdzania'+(' · N/A: poza fotografiami GT' if not eid else ''))
        if not gid:
            n=self.session.material.order.index(eid)+1 if eid else None
            self.current_status.set((f'Zdjęcie {n} · ' if n else '')+'Klatka telefonu · '+str(len(self.evidence['frames'][fid]['calls']))+' MT')

    def choose_frame(self,eid,fid):
        old=self.session.state(eid)
        if old['frame'] and old['frame']['frame_id']!=fid and any(photo.localization_outline(p) for p in old['objects']):
            if not messagebox.askyesno('Zmiana klatki GT','Zmieniasz klatkę odniesienia. Poprzednie obrysy pozostaną w historii; na nowej klatce ich geometria będzie N/A. Kontynuować?',parent=self.window):return
        if self.source.commit(lambda:self.session.choose_frame(eid,fid)) is not False:
            self.select(eid)
            self.status_var.set('Wybrano klatkę GT bez przenoszenia obrysów. Reguła wyboru wyniku głównego pozostała bez zmian.')

    def _paint_image(self,message=''):
        if self.preview and not self.preview.get('frame_id'):
            self.canvas.delete('all')
            self.canvas.create_text(max(100,self.canvas.winfo_width()/2),max(60,self.canvas.winfo_height()/2),
                text='N/A · brak udowodnionej klatki tego MT',fill='#dfe8ed')
            return
        super()._paint_image(message)
        self.canvas.delete('a1_readonly')
        if self.operator_mode and not self.preview and self._view_center is not None:
            for p in self.session.index.ref.plates(self.session.operator_photo(self.current_invocation)):
                q=photo.localization_outline(p)
                if q:self.canvas.create_polygon(*[v for pt in q for v in self._canvas_editor.to_canvas(pt)],fill='',outline='#38E4D3',width=2,tags=('a1_readonly',))
        if not self.preview or not self.preview.get('frame_id') or self._view_center is None:return
        from ..ranking.mobile_canvas_review import context
        gids=[self.preview['gid']] if self.preview.get('gid') else self.evidence['frames'][self.preview['frame_id']]['calls']
        for gid in gids:self._paint_source_call(gid,context(self.mobile,gid))
        self.inspection_status()

    def _paint_source_call(self,gid,ctx):
        if ctx and self.evidence['calls'][gid]['frame_id']==self.preview['frame_id'] and tuple(ctx['size'])==self._image_source_size:
            l,t,r,b=ctx['roi'];self.canvas.create_rectangle(*self._canvas_editor.to_canvas((l,t)),*self._canvas_editor.to_canvas((r,b)),outline='#909BA5',tags=('scene_source_roi',))
        eid=self.evidence['calls'][gid]['exposure_id']
        number=[g for g,r in self.evidence['calls'].items() if r['exposure_id']==eid].index(gid)+1
        for i,c in enumerate(self.evidence['calls'][gid]['source_candidates'],1):
            box=c['bbox'];valid=bool(box and all(math.isfinite(v) for v in box) and box[0]<box[2] and box[1]<box[3])
            q=c['pose'] or ([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])] if valid else None)
            if not q:continue
            coords=[v for p in q for v in self._canvas_editor.to_canvas(p)]
            selected=self.preview.get('aid')==c['attempt_id']
            self.canvas.create_polygon(*coords,fill='',outline='#FFD36B' if selected else '#55CAFF',width=3 if selected else 1,tags=('scene_source_mt',c['attempt_id']))
            self.canvas.create_text(*self._canvas_editor.to_canvas(q[0]),text=f'MT {number} · {i}',anchor='sw',fill='#55CAFF',tags=('scene_source_mt',))

    def inspection_status(self):
        if not self.preview or not self.preview.get('frame_id'):return
        if not self.preview.get('gid'):
            fid=self.preview['frame_id'];f=self.evidence['frames'][fid];eid=f['exposure_id']
            prefix=f'Zdjęcie {self.session.material.order.index(eid)+1} · ' if eid else ''
            self.current_status.set(prefix+'Klatka telefonu · '+str(len(f['calls']))+' MT')
            return
        gid=self.preview['gid'];aid=self.preview.get('aid');row=self.evidence['calls'][gid]
        eid=row['exposure_id'];fid=row['frame_id']
        prefix=''
        if eid:
            ids=[self.evidence['aliases'][f] for f in self.session.index.frame_ids(eid)]
            prefix=f'Zdjęcie {self.session.material.order.index(eid)+1} · Klatka {ids.index(fid)+1} · '
        number=[g for g,r in self.evidence['calls'].items() if r['exposure_id']==eid].index(gid)+1
        mids=[aid] if aid in row['readouts'] else row['readout_ids'] if not aid else []
        lines=[]
        for a in mids:
            m=row['readouts'][a];raw=' / '.join(m['readouts']) or 'brak odczytu'
            consensus=' / '.join(m['consensus']) if m['consensus'] else 'N/A · brak pola'
            state='✓ potwierdzony przez Androida' if m['confirmed'] is True else 'niepotwierdzony' if m['confirmed'] is False else 'potwierdzenie N/A'
            lines.append(('Raw: '+raw if m['executed'] else 'MZ nie wykonano')+' → Konsensus: '+consensus+' · '+state)
        self.current_status.set(prefix+f'MT {number} · Wskazania: '+str(len(row['source_candidates']))+
            (' · ! anulowane' if row['stale'] else '')+'\n'+('\n'.join(lines) or 'Brak zapisanego odczytu'))
        if self.operator_mode and not self.source.pending_action:self.show_accounting()

    def show_exceptions(self):
        if not self.source.available():return
        if not self._canvas_editor.prepare_navigation():return
        if not self.session.approved(self.current_invocation):
            self.status_var.set('Najpierw zatwierdź GT sceny przy ukrytych predykcjach.');return
        self.show_exception_rows=True;self.refresh_scenes()
        eid=self.current_invocation
        if self.subject_tree.exists(eid):self.subject_tree.item(eid,open=True)
        units=[u for u in self.results()['units'] if u['exposure_id']==eid and u['exceptions']]
        if units:
            u=units[0];iid='attempt:'+u['exceptions'][0];self.subject_tree.item('call:'+u['mt_invocation_id'],open=True)
            self.subject_tree.selection_set(iid);self.subject_tree.see(iid);self._selected()
        else:self.status_var.set('Brak nierozstrzygniętych wskazań w tej scenie. Wszystkie źródłowe MT pozostają dostępne na liście.')

    def exception_menu(self,event=None,*,show=True):
        menu=tk.Menu(self.window,tearoff=False);p=self.preview or {};gid=p.get('gid');aid=p.get('aid')
        eid=self.session.index.calls.get(gid,{}).get('exposure_id')
        if gid and event:
            from ..ranking.mobile_source_frame_roi import quad_contains_point
            point=self._canvas_editor.to_source(event.x,event.y)
            hits=[]
            for c in data.source_candidates(self.mobile,gid):
                b=c['bbox'];q=c['pose'] or ([(b[0],b[1]),(b[2],b[1]),(b[2],b[3]),(b[0],b[3])] if b else None)
                if q and quad_contains_point(q,*point):hits.append(c['attempt_id'])
            if len(hits)==1:aid=hits[0]
        def save(decision,pid=None,readable=None,reason=''):
            return self.source.commit(lambda:self.session.exception(eid,gid,aid,decision,gt_id=pid,text_readable=readable,reason=reason))
        if eid and aid and self.session.approved(eid):
            for i,obj in enumerate(self.session.state(eid)['objects'],1):
                menu.add_command(label=f'Trafne · tablica {i}',command=lambda pid=obj['object_id']:save('TP',pid))
            menu.add_command(label='Błędne wskazanie MT',command=lambda:save('FP'))
            def na():
                reason=simpledialog.askstring('N/A','Dlaczego nie można ocenić tego wskazania?',parent=self.window)
                if reason:save('N/A',reason=reason)
            menu.add_command(label='Nie można ocenić',command=na)
            old=self.session.state(eid)['exceptions'].get(aid,{})
            if old.get('decision')=='TP':
                menu.add_separator()
                menu.add_command(label='Tekst czytelny na tej klatce',command=lambda:save('TP',old['gt_id'],True))
                menu.add_command(label='Tekst nieczytelny · MZ N/A',command=lambda:save('TP',old['gt_id'],False,'Nieczytelny tekst na tej klatce'))
        else:menu.add_command(label='N/A · wybierz wskazanie po zatwierdzeniu GT sceny',state='disabled')
        if not show:return menu
        try:menu.tk_popup(event.x_root if event else self.more_button.winfo_rootx(),event.y_root if event else self.more_button.winfo_rooty())
        finally:menu.grab_release()
        return 'break'

    def show_result_status(self):
        if self.operator_mode:
            if self._scene_widgets_ready:
                if not self.preview and self.current_invocation and self._image_entry!='A1:'+self.session.operator_photo(self.current_invocation):
                    self.select_original(self.current_invocation)
                self.refresh_reference();self.refresh_scenes();self.show_accounting()
            return
        if not self.session.approved(self.current_invocation):return
        results=self.results();units=[u for u in results['units'] if u['exposure_id']==self.current_invocation]
        self.status_var.set(f"GT sceny zapisane · MT ocenialne: {sum(u['operational_status']=='ASSESSED' for u in units)}/{len(units)} · pozostałe N/A; sprawdź wyjątki w razie potrzeby")
        self.inspection_status()

    def original_image(self,sha):
        ref=self.session.index.ref;ref.verify()
        path=ref.photo_root/ref.photos[sha]['sciezka_zrodla'];stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if sha not in self._original_cache or self._original_cache[sha][0]!=stamp:
            if a2.sha(path)!=sha:raise ValueError('SHA oryginalnej fotografii nie odpowiada A1.')
            with Image.open(path) as im:image=ImageOps.exif_transpose(im).convert('RGB')
            self._original_cache[sha]=(stamp,image)
        return self._original_cache[sha][1]

    def select_original(self,eid,*,initial=False):
        ed=self._canvas_editor
        if not initial and not ed.prepare_navigation():
            self._sync_navigation(self._nav_target);return False
        try:image=self.original_image(self.session.operator_photo(eid))
        except (OSError,ValueError,KeyError) as exc:
            if self.current_invocation:self.subject_tree.selection_set(self.current_invocation)
            self.status_var.set('Nie można otworzyć GT oryginalnego zdjęcia: '+str(exc));return False
        self.preview=None;self.current_invocation=eid
        if self._image is not image:
            self._image=image;self._image_source_size=image.size;self._image_entry='A1:'+self.session.operator_photo(eid)
            self._image_token+=1;self._view_zoom=1.;self._view_center=None
        ed.escape();ed.sync();self.subject_tree.selection_set(eid);self._paint_image()
        if self._scene_widgets_ready:
            self.confirm_button.configure(state='normal');self.refresh_reference();self._sync_navigation(('exposure',eid));self.show_accounting()
        return True

    def ensure_protocol(self):
        if self.session.protocol_approval():return True
        approval=getattr(self.browser,'_f5a_operator_protocol',None)
        if not approval or approval.get('binding')!=self.session.index.ref.binding or approval.get('protocol_sha256')!=data.fingerprint(data.REVIEW_PROTOCOL):
            if not messagebox.askyesno('Protokół oceny F5A',data.PROTOCOL_TEXT+'\n\nZatwierdzasz ten protokół?',parent=self.window):return False
            approval=None
        if self.source.commit(lambda:self.session.accept_protocol(approval=approval)) is False:return False
        self.browser._f5a_operator_protocol=copy.deepcopy(self.session.protocol_approval())
        return True

    def decide(self,decision,*,reason=None,photo_sha=None,diagnostic_b=None,override_b=False):
        if not self.source.available() or not self._canvas_editor.prepare_navigation():return 'break'
        target=self._nav_target or ('exposure',self.current_invocation)
        eid=self._target_exposure(target)
        if eid not in self.session.index.exposures:
            self.status_var.set('Dowód poza przypisanymi zdjęciami — wybierz zdjęcie do oceny.');return 'break'
        if not self.ensure_protocol():return 'break'
        if reason is None and decision!='CONFIRMED':
            reason=simpledialog.askstring('Ocena zdjęcia','Krótki powód (wybrany dowód zostanie dołączony):',parent=self.window)
            if not reason or not reason.strip():return 'break'
        viewed=tuple(self.viewed.get(eid,()))
        self.source.commit(lambda:self.session.decide_exposure(eid,decision,reason=reason or '',context=target,
            viewed=viewed,photo_sha=photo_sha,diagnostic_b=diagnostic_b,override_b=override_b))
        return 'break'

    def operator_menu(self,event=None,*,show=True):
        menu=tk.Menu(self.window,tearoff=False)
        menu.add_command(label='Cofnij ocenę · Ctrl+Z',command=self.source.restore)
        menu.add_command(label='Ponów ocenę · Ctrl+Y',command=lambda:self.source.restore(redo=True))
        menu.add_separator()
        menu.add_command(label='Popraw przypisanie zdjęcia…',command=self.correct_photo)
        menu.add_command(label='Popraw ocenę diagnostyczną B…',command=self.correct_diagnostic)
        menu.add_command(label='Protokół oceny…',command=lambda:messagebox.showinfo('Protokół oceny F5A',data.PROTOCOL_TEXT,parent=self.window))
        if not show:return menu
        try:menu.tk_popup(event.x_root if event else self.more_button.winfo_rootx(),event.y_root if event else self.more_button.winfo_rooty())
        finally:menu.grab_release()
        return 'break'

    def correct_photo(self):
        number=simpledialog.askinteger('Popraw przypisanie','Numer oryginalnej fotografii GT (1–40):',minvalue=1,maxvalue=40,parent=self.window)
        if number is None:return
        matches={e['photo_sha256'] for e in self.session.index.ref.exposures.values() if e.get('photo_global_position')==number}
        if len(matches)!=1:
            messagebox.showerror('Przypisanie','Nie ma jednoznacznej fotografii o tym numerze w A2.',parent=self.window);return
        sha=next(iter(matches));self.decide('CORRECTED',photo_sha=sha)
        if self.session.operator_photo(self.current_invocation)==sha:self.select(self.current_invocation)

    def correct_diagnostic(self):
        value=messagebox.askyesnocancel('Popraw ocenę B','Czy wystąpił zapisany, potwierdzony przez Androida odczyt zgodny z pełnym GT?\nTak / Nie — korekta B; Anuluj — bez zmian.',parent=self.window)
        if value is not None:self.decide('CORRECTED',diagnostic_b=value,override_b=True)

    def show_accounting(self):
        if not self.operator_mode or self.source.pending_action:return
        eid=self.current_invocation
        if eid not in self.session.index.exposures:return
        sha=self.session.operator_photo(eid);key=(eid,sha)
        if key not in self._accounting_cache:self._accounting_cache[key]=data.exposure_accounting(self.session,eid,evidence=self.evidence,photo_sha=sha)
        r=self._accounting_cache[key];decision=self.session.operator_decision(eid);seq=r['A']
        value=r['B']['value']
        if decision and decision['decision']=='CORRECTED' and decision['correction']['diagnostic_b_override']:value=decision['correction']['diagnostic_b']
        b='tak' if value is True else 'nie' if value is False else 'N/A'
        c=self.frozen['exposures'].get(eid,{})
        c_label={'FAILURE':'niepowodzenie','SUCCESS':'zgodny','N/A':'N/A'}.get(c.get('result'),'N/A')
        line=f"A · {r['counts']['mt']} MT / {r['counts']['mz']} MZ; każdy komplet osobno. B · potwierdzony odczyt zgodny z GT: {b}. C · pierwsze MT: {c_label}."
        if value is None:line+=' B: '+('brak pełnego numeru GT' if r['B']['reason']=='NO_FULL_GT_NUMBER' else 'brak wystarczającego dowodu potwierdzenia')+'.'
        if not seq:line='Model MT nie został uruchomiony · zapisano zero wybranych ROI.' if r['no_mt_evidence'] else 'MT: brak wystarczającego dowodu wykonania · N/A.'
        target=self._nav_target or ();gid=target[1] if target and target[0] in ('call','attempt','mz','consensus') else None
        chosen=next((v for v in seq if gid in v['mt_invocation_ids']),None)
        if chosen:
            reasons={'EXECUTION_FAILURE_NO_PUBLISHED_OUTPUT':'anulowanie/błąd wykonania; brak opublikowanego wyniku',
                'CANCELLED_OUTPUT':'wynik anulowany','PARTIAL_OR_UNREADABLE_GT_TEXT':'GT częściowe/nieczytelne',
                'CONSENSUS_FIELD_MISSING':'brak konsensusu w archiwum','RAW_READOUT_AMBIGUOUS':'niejednoznaczny odczyt',
                'SOURCE_FRAME_UNPROVEN':'brak dowodu klatki','FINAL_ROLE_UNPROVEN':'brak dowodu roli FINAL'}
            def caption(label,x):
                return label+': '+('komplet zgodny' if x['exact'] is True else 'komplet różny' if x['exact'] is False else 'N/A ('+reasons.get(x['reason'],'brak wystarczających dowodów')+')')+\
                    ' · braki '+(', '.join(f'{k} ×{v}' for k,v in x['known_missing'].items()) or '0')+\
                    ' · dodatkowe '+(', '.join(f'{k} ×{v}' for k,v in x['unaccounted'].items()) or '0')
            line+='\n'+caption('Raw',chosen['raw'])+' | '+caption('Konsensus',chosen['consensus'])
        elif seq:
            line+='\nKomplety zgodne z pełnym GT: '+', '.join(
                f"{label} {sum(v[k]['exact'] is True for v in seq)}/{sum(v[k]['exact'] is not None for v in seq)} ocenialnych; N/A {sum(v[k]['exact'] is None for v in seq)}"
                for label,k in [('raw','raw'),('konsensus','consensus')])+'. To przebieg, nie końcowy HUD.'
        if decision:line+='\n'+self.session.operator_status(eid)+((': '+decision['reason']) if decision['reason'] else '')
        if not self.session.protocol_approval():line+=' · Protokół do zatwierdzenia przy pierwszej ocenie.'
        self.status_label.configure(wraplength=max(600,self.window.winfo_width()-40),justify='left');self.status_var.set(line)

    def close(self):
        super().close()
        if self._closed:
            link=getattr(self.browser,'_linked_evidence_timing',None)
            if link and link['callback']==self._timing_selected:
                self.browser._linked_evidence_timing=None
                self.browser._draw_latency_chart()

    def export(self):
        if self.source.editing() or self._canvas_editor.entry is not None:
            self.status_var.set('Zakończ albo anuluj bieżącą edycję przed eksportem.');return
        if self.operator_mode and not data.exposure_progress(self.session.data)['reviewed']:
            self.status_var.set('Brak zapisanej oceny operatora do eksportu.');return
        target=filedialog.askdirectory(parent=self.window,title='Katalog eksportu GT scen i wyników F5A')
        if not target:return
        dest=Path(target)/('F5A_scenes_'+self.mobile.session_id+'_r'+str(self.mobile.review.review_revision))
        if dest.exists():messagebox.showerror('Eksport','Ten eksport już istnieje. Wybierz inny katalog.',parent=self.window);return
        try:
            dest.mkdir(parents=True);self.mobile.export(dest)
            self.status_var.set('Zapisano eksport: '+str(dest))
        except (OSError,ValueError) as exc:messagebox.showerror('Eksport',str(exc),parent=self.window)


def open_if_supported(browser,report,bundle=None):
    if (getattr(report,'experiment_index',{}) or {}).get('series_id')!=a2.SERIES:return None
    if getattr(browser,'_f5a_assistance_requested',False):return None
    from ..ranking.mobile_human_review import MobileReviewSession
    from ..ranking.mobile_package_experiments import read_mobile_report_bundle
    mobile=MobileReviewSession(bundle or read_mobile_report_bundle(report.source_path))
    layer=mobile.review.provenance.get(data.KEY)
    if layer:
        b=layer['binding'];ref=a2.Reference(b['mapping_path'],b['gt_path'],expected=b)
    else:
        ref=getattr(browser,'_f5a_reference',None) or a2.registered()
        if ref is None:
            found=a2.discover(mobile.path)
            ref=a2.Reference(found[0]) if len(found)==1 else None
    if ref is None:return None  # keep the existing ordinary verifier available
    return SceneReviewWindow(browser,mobile,ref)
