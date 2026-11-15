"""Optional FINAL mode; old F5A block viewer is imported unchanged.

python -m auto_annotation_tool.gui.final180_tv --help
"""
import argparse
from copy import deepcopy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid
import windows_dpi as monitor_dpi

from PIL import Image, ImageDraw, ImageFont, ImageTk
from ..ranking import final180_reference as reference
from ..ranking import final180_markers as markers
from ..ranking import photo_reference as photo
from ..gt_pack import _atomic_write_json
from .photo_material_ui import MaterialPresentationActions

from alpr_runtime import workspace_root
LEGACY_DEFAULT=Path(__file__).with_name('f5a_tv_core.py')
DEFAULT_EXPERIMENT=workspace_root()/'experiments/F6-FINAL180-A12-01'
ACTIVE_MODES=('PREPARING','COUNTDOWN','RUNNING','TRANSITION')


def preferred_dwell():
    from ..config import SESSION
    value=SESSION.get('final_presentation','dwell_seconds',reference.DEFAULT_DWELL_SECONDS) if SESSION else reference.DEFAULT_DWELL_SECONDS
    return reference.parse_dwell(value)


def remember_selection(seconds,prepared=None):
    from ..config import SESSION
    seconds=reference.parse_dwell(seconds)
    if SESSION is not None:
        SESSION.set('final_presentation','dwell_seconds',seconds)
        if prepared is not None:SESSION.set('final_presentation','prepared_path',str(Path(prepared).resolve()))
        SESSION.save_session()


class PreparedSelectionError(ValueError):
    """A normal folder-choice rejection, not an unexpected application failure."""


def read_prepared_selection(path):
    """Validate existing material and reference before changing any Tk state."""
    path=Path(path).resolve()
    if not path.is_dir():raise PreparedSelectionError('Wybrana lokalizacja nie jest dostępnym folderem: '+str(path))
    if not (path/'manifest.json').is_file():
        hint=('Otwórz jeden z podfolderów prepared_*, nie katalog nadrzędny.'
            if next(path.glob('prepared_*/manifest.json'),None) is not None else 'Wybierz gotowy folder prepared_* zawierający manifest.json.')
        raise PreparedSelectionError('Brak manifest.json w wybranym folderze:\n'+str(path)+'\n'+hint)
    before=photo.digest(path/'manifest.json')
    manifest,material,_=reference.load(path,strict_reference=False)
    gt=reference.reference_preflight(path)
    if photo.digest(path/'manifest.json')!=before:raise ValueError('Manifest zmienił się podczas wczytywania.')
    return path,manifest,material,gt


def dwell_argument(value):
    try:return reference.parse_dwell(value)
    except ValueError as exc:raise argparse.ArgumentTypeError(str(exc)) from None


def legacy_module(path=None):
    if path is not None and Path(path).resolve()!=LEGACY_DEFAULT.resolve():
        raise ValueError('F6 używa wewnętrznego modułu przeglądarki; zewnętrzny skrypt nie jest wymagany ani ładowany.')
    from . import f5a_tv_core as module
    for name in ('TvApp','choose_monitors','set_dpi_aware'):
        if not hasattr(module,name):raise ValueError('LEGACY_VIEWER_API_MISSING:'+name)
    return module


def calibration_frame(manifest,manifest_sha,technical_id=None):
    """Decorate a neutral full photo field, then use the unchanged FINAL renderer."""
    board=manifest['layout'];l,t,r,b=board['photo_rect'];width,height=r-l,b-t
    if board.get('layout_version') in markers.BOUND_LAYOUTS:
        technical_id=technical_id or uuid.uuid4().hex
        payload=markers.payload(manifest,manifest_sha,technical_id,'TEST_ONLY','T',test_only=True)
        image,_=markers.render(None,board,payload);draw=ImageDraw.Draw(image)
        draw.rectangle((l,t,r-1,b-1),fill='#343a40',outline='white',width=max(2,round(board['height']/360)))
        def label(text,y,size):
            try:font=ImageFont.truetype(str(Path(os.environ.get('WINDIR',r'C:\Windows'))/'Fonts/segoeui.ttf'),size)
            except OSError:font=ImageFont.load_default()
            draw.text((board['width']/2,y),text,font=font,anchor='mm',fill='white')
        size=max(16,round(board['height']/42))
        label('USTAWIENIE KADRU F6',t*.28,size)
        label('Wskaż cztery rogi jasnej ramki',t*.68,size)
        label('Pole fotografii F6 ustaw RAZ · zewnętrzna kalibracja TV jest osobna',b+(board['height']-b)*.30,max(14,round(size*.72)))
        label('TEST_ONLY · bez STARTU badania · Esc: zamknij planszę',b+(board['height']-b)*.72,max(14,round(size*.72)))
        return image,dict(purpose='TEST_ONLY',phase='T',technical_id=technical_id,exposure_id='TEST_ONLY',
            manifest_sha256=manifest_sha,token=payload,photo_rect=list(board['photo_rect']),
            test_binding=markers.binding(manifest,manifest_sha,technical_id,test_only=True))
    field=Image.new('RGB',(width,height),'#343a40');draw=ImageDraw.Draw(field)
    edge=max(3,round(min(width,height)/140))
    draw.rectangle((0,0,width-1,height-1),outline='#ffdc58',width=edge)
    arm=max(18,round(min(width,height)/12))
    for x,y,dx,dy in ((0,0,1,1),(width-1,0,-1,1),(width-1,height-1,-1,-1),(0,height-1,1,-1)):
        draw.line((x,y,x+dx*arm,y),fill='white',width=edge*2)
        draw.line((x,y,x,y+dy*arm),fill='white',width=edge*2)
    def font(size):
        try:return ImageFont.truetype(str(Path(os.environ.get('WINDIR',r'C:\Windows'))/'Fonts/segoeui.ttf'),size)
        except OSError:return ImageFont.load_default()
    size=max(15,min(40,round(height/16)))
    for label,y,scale in (
            ('USTAWIENIE KADRU F6',.30,1),
            ('CAŁE STAŁE POLE FOTOGRAFII',.45,1),
            ('Wskaż cztery rogi tej wewnętrznej ramki.',.59,.72),
            ('TEST_ONLY · bez STARTU badania',.73,.72),
            ('Esc: zamknij planszę',.85,.65)):
        draw.text((width/2,height*y),label,font=font(round(size*scale)),anchor='mm',fill='white')
    technical_id=technical_id or uuid.uuid4().hex
    payload=markers.token(manifest_sha,technical_id,'TEST_ONLY','T')
    image,placement=markers.render(field,board,payload)
    if placement['rect']!=list(board['photo_rect']):raise ValueError('CALIBRATION_PHOTO_FIELD_MISMATCH')
    return image,dict(purpose='TEST_ONLY',phase='T',technical_id=technical_id,
        exposure_id='TEST_ONLY',manifest_sha256=manifest_sha,token=payload,photo_rect=list(board['photo_rect']))


class PresentationRun:
    """Durable PC operations; only an observed replacement closes an interval."""
    def __init__(self, prepared, manifest, *, resume=None, clock=time.perf_counter_ns, utc=reference.utc):
        from ..ranking import final180_timeline as timeline
        self.prepared=Path(prepared);self.manifest=deepcopy(manifest);self.clock=clock;self.utc=utc
        if self.manifest!=photo.read_json(self.prepared/'manifest.json'):raise ValueError('RUN_FROZEN_MANIFEST_MISMATCH')
        self.dwell_seconds=reference.parse_dwell(self.manifest['dwell_seconds'])
        self.manifest_sha=photo.digest(self.prepared/'manifest.json');self.log_error=None
        recovery=[]
        if resume:
            self.directory=Path(resume).resolve()
            self.state,self.events,recovery=timeline.read_run(self.directory/'run.json',manifest,self.manifest_sha,recover=True)
            if self.state.get('dwell_seconds',self.dwell_seconds)!=self.dwell_seconds:raise ValueError('RESUME_DWELL_MISMATCH')
            consumed=[e['position'] for e in self.events if e['event'] in ('PHOTO_DRAW_INTENT','PHOTO_REQUESTED')]
            self.next_index=max(consumed,default=0)
            self.segment=max((e['segment'] for e in self.events),default=0)+1
        else:
            self.directory=self.prepared/'runs'/('run_'+uuid.uuid4().hex)
            self.directory.mkdir(parents=True,exist_ok=False)
            self.state=dict(schema='alpr.final.presentation_run.v1',run_id=uuid.uuid4().hex,
                manifest_sha256=self.manifest_sha,prepared_path=str(self.prepared.resolve()),created_utc=self.utc(),
                clock_domain='PC_PERF_COUNTER_NS_PER_SEGMENT',display_evidence='TK_VIEW_UPDATE_NOT_PANEL_REFRESH',
                interval_basis='VIEW_UPDATE_COMPLETED_TO_OBSERVED_REPLACEMENT_REQUEST_PC',
                android_start='NOT_OBSERVED',status='PREPARED',journals=[])
            self.events=[];self.next_index=0;self.segment=1
            if manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS:
                self.state['qr_binding']=markers.binding(manifest,self.manifest_sha,self.state['run_id'])
                self.state['run_tag_registry_scope']=timeline.check_run_tag(self.prepared,self.state)
        self.state.update(dwell_seconds=self.dwell_seconds,countdown_seconds=self.manifest['countdown_seconds'],
            transition_seconds=self.manifest['transition_seconds'])
        self.segment_id=uuid.uuid4().hex;self.clock_id=uuid.uuid4().hex
        self.event_seq=max((e.get('event_seq',0) for e in self.events),default=0)
        self.current=None;self.deadline=None;self.status='PREPARED';self.phase='PREPARATION';self.segment_start=self.clock()
        self.segment_first_index=self.next_index;self.plan_origin_ns=None
        name='events.jsonl' if not resume else f'events_segment_{self.segment:03d}_{self.segment_id[:8]}.jsonl'
        self.journal_path=self.directory/name
        self.journal=self.journal_path.open('x',encoding='utf-8',newline='\n')
        self.state['journals'].append(dict(file=name,bytes=0,sha256=__import__('hashlib').sha256(b'').hexdigest()))
        if recovery:
            self.state.setdefault('recoveries',[]).extend(recovery)
            _atomic_write_json(self.directory/f'recovery_{self.segment_id}.json',dict(recovery=recovery,action='ORIGINAL_BYTES_RETAINED_NEW_SEGMENT'))
        self.append('SEGMENT_PREPARED',next_position=self.next_index+1,recovery=recovery)
        try:timeline.register_run(self.prepared,self.directory/'run.json',self.state)
        except (OSError,ValueError):
            # The locator is optional; read-only legacy discovery remains valid.
            logging.exception('FINAL: nie zapisano indeksu lokalizacji runu')

    def point(self):return dict(pc_perf_counter_ns=self.clock(),pc_utc=self.utc())

    def append(self,event,*,point=None,**fields):
        if self.log_error:raise OSError('LOG_ERROR: '+self.log_error)
        point=point or self.point();self.event_seq+=1
        record=dict(schema='alpr.final.presentation_event.v1',run_id=self.state['run_id'],manifest_sha256=self.manifest_sha,
            event=event,event_seq=self.event_seq,segment=self.segment,segment_id=self.segment_id,clock_id=self.clock_id,
            clock_domain='PC_PERF_COUNTER',pc_unit='ns',segment_elapsed_ns=point['pc_perf_counter_ns']-self.segment_start,
            phase=self.phase,**point)
        record.update(status='OBSERVED');record.update(fields)
        if record.get('position'):
            exposure=self.manifest['exposures'][record['position']-1]
            record.update(exposure_id=exposure['exposure_id'],photo_sha256=exposure['photo_sha256'],
                manifest_record=f"exposures/{record['position']-1}")
        try:
            self.journal.write(json.dumps(record,ensure_ascii=False)+'\n');self.journal.flush();os.fsync(self.journal.fileno())
            self.events.append(record)
            self.state['journals'][-1].update(bytes=self.journal_path.stat().st_size,sha256=photo.digest(self.journal_path))
            self.state.update(status=self.status,last_event=record,next_index=self.next_index,
                journal_sha256=self.state['journals'][0]['sha256'],segments=self.segment)
            _atomic_write_json(self.directory/'run.json',self.state)
        except (OSError,ValueError) as exc:
            self.log_error=str(exc);self.status='LOG_ERROR';self.state['status']='LOG_ERROR'
            try:self.journal.close()
            except OSError:pass
            raise OSError('LOG_ERROR: '+str(exc)) from exc
        return record

    def present(self,index,placement,draw,*,require_binding=False):
        if index!=self.next_index or self.current is not None:raise ValueError('NO_SKIPPING_OR_REPLAYING_EXPOSURES')
        exposure=self.manifest['exposures'][index]
        expected=dict(manifest_sha256=self.manifest_sha,run_id=self.state['run_id'],exposure_id=exposure['exposure_id'],photo_sha256=exposure['photo_sha256'])
        if require_binding and placement.get('binding')!=expected:raise ValueError('PREPARED_FRAME_IDENTITY_CONFLICT')
        token_fields={}
        if self.manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS:
            value=markers.payload(self.manifest,self.manifest_sha,self.state['run_id'],exposure['exposure_id'],'P')
            if require_binding and placement.get('display_token')!=value:raise ValueError('F2_PREPARED_TOKEN_CONFLICT')
            token_fields['display_token']=value
        self.phase='P';self.status='RUNNING';self.next_index=index+1
        self.append('PHOTO_DRAW_INTENT',position=index+1,status='INTENT')
        request=self.point()
        try:draw()
        except Exception as exc:
            self.append('PHOTO_DRAW_FAILED',position=index+1,reason=str(exc),status='FAILED');raise
        updated=self.point();self.current=index
        self.deadline=updated['pc_perf_counter_ns']+round(self.dwell_seconds*1e9)
        if self.plan_origin_ns is None:
            self.plan_origin_ns=request['pc_perf_counter_ns']-round(float(self.manifest['countdown_seconds'])*1e9)
        planned=round((float(self.manifest['countdown_seconds'])+(index-self.segment_first_index)*
            (self.dwell_seconds+float(self.manifest['transition_seconds'])))*1e9)
        self.append('PHOTO_REQUESTED',point=request,position=index+1,photo_placement=placement,
            dwell_seconds=self.dwell_seconds,requested_deadline_ns=self.deadline,
            schedule_delay_ns=request['pc_perf_counter_ns']-self.plan_origin_ns-planned,
            prepare_duration_ns=placement.get('prepare_duration_ns'),read_duration_ns=placement.get('read_duration_ns'),actual_panel_refresh='NOT_OBSERVED')
        self.append('PHOTO_VIEW_UPDATED',point=updated,position=index+1,
            draw_duration_ns=updated['pc_perf_counter_ns']-request['pc_perf_counter_ns'],actual_panel_refresh='NOT_OBSERVED',**token_fields)

    def show(self,index,placement):
        # Retained for programmatic callers; this records no observed GUI completion.
        if index!=self.next_index or self.current is not None:raise ValueError('NO_SKIPPING_OR_REPLAYING_EXPOSURES')
        self.phase='P';self.current=index;self.next_index=index+1;self.status='RUNNING'
        self.deadline=self.clock()+round(self.dwell_seconds*1e9)
        self.append('PHOTO_REQUESTED',position=index+1,photo_placement=placement,observation='REQUEST_ONLY')

    def due(self):return self.deadline is not None and self.clock()>=self.deadline

    def end_photo(self):
        if self.current is None:return
        if not self.due():raise ValueError('EXPOSURE_DURATION_NOT_COMPLETED')
        self.append('PHOTO_DWELL_COMPLETED',position=self.current+1,deadline_lateness_ns=self.clock()-self.deadline)
        # Timer completion is not replacement. current stays until replace().

    def replace(self,draw,*,phase='T',reason='ADVANCE',display_token=None):
        previous=self.manifest['exposures'][self.current]['exposure_id'] if self.current is not None else None
        position=self.current+1 if self.current is not None else None
        self.phase=phase;self.append('REPLACEMENT_INTENT',position=position,replaced_exposure_id=previous,reason=reason,status='INTENT')
        request=self.point()
        try:draw()
        except Exception as exc:
            self.append('REPLACEMENT_FAILED',position=position,replaced_exposure_id=previous,reason=str(exc),status='FAILED');raise
        updated=self.point()
        self.append('REPLACEMENT_REQUESTED',point=request,position=position,replaced_exposure_id=previous,reason=reason)
        self.append('REPLACEMENT_VIEW_UPDATED',point=updated,position=position,replaced_exposure_id=previous,
            replacement_requested_ns=request['pc_perf_counter_ns'],reason=reason,
            draw_duration_ns=updated['pc_perf_counter_ns']-request['pc_perf_counter_ns'],
            **({'display_token':display_token} if display_token is not None else {}))
        self.current=None;self.deadline=None
        return updated['pc_perf_counter_ns']

    def complete(self,draw,*,display_token=None):
        if self.next_index!=len(self.manifest['exposures']) or not self.due():raise ValueError('RUN_NOT_FINISHED')
        self.replace(draw,phase='T' if display_token is not None else 'END',display_token=display_token)
        self.status='COMPLETED';self.append('RUN_COMPLETED');self.journal.close()

    def stop(self,reason='OPERATOR_STOP',clear=None):
        if self.journal.closed:return
        partial=self.current+1 if self.current is not None else None
        if clear is not None:self.replace(clear,phase='END',reason=reason)
        self.status='STOPPED';self.phase='END'
        self.append('RUN_STOPPED',reason=reason,partial_position=partial);self.journal.close()


class PreparedFrames:
    """One worker, at most current + next PIL frame; Tk images stay on UI thread."""
    def __init__(self,material,manifest,manifest_sha,run_id):
        self.material,self.manifest,self.manifest_sha,self.run_id=material,manifest,manifest_sha,run_id
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='final-tv-prefetch');self.pending={}
        self.max_resident=0

    def request(self,index):
        if index>=len(self.manifest['exposures']):return
        for old in list(self.pending):
            if old<index-1:self.pending.pop(old).cancel()
        if index not in self.pending:
            exposure=self.manifest['exposures'][index]
            def load():
                started=time.perf_counter_ns()
                image=self.material.image(exposure['image_id'])
                read=time.perf_counter_ns()
                value=markers.payload(self.manifest,self.manifest_sha,self.run_id,exposure['exposure_id'],'P')
                frame,placement=markers.render(image,self.manifest['layout'],value)
                if self.manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS:placement['display_token']=value
                placement.update(binding=dict(manifest_sha256=self.manifest_sha,run_id=self.run_id,
                    exposure_id=exposure['exposure_id'],photo_sha256=exposure['photo_sha256']),
                    read_duration_ns=read-started,prepare_duration_ns=time.perf_counter_ns()-started)
                return frame,placement
            self.pending[index]=self.pool.submit(load)
        self.max_resident=max(self.max_resident,len(self.pending))
        if len(self.pending)>2:raise RuntimeError('PREFETCH_CACHE_LIMIT')

    def ready(self,index):return index in self.pending and self.pending[index].done()
    def get(self,index):return self.pending[index].result()
    def close(self):
        for task in self.pending.values():task.cancel()
        self.pool.shutdown(wait=False,cancel_futures=True);self.pending.clear()


class FinalResultsWindow:
    """Read-only result view; images/overlays, no annotation engine or decisions."""
    FILTERS=('Wszystkie','Poprawne','Błędne','Pominięte GT','FP','Negatywy','Nieczytelne','Brak dowodów','QR / ArUco','Anulowane','N/A / brak GT')
    def __init__(self,parent,result_path,*,on_reference=None,on_manual=None):
        self.path=Path(result_path);self.report=photo.read_json(self.path/'evaluation.json')
        proof=photo.read_json(self.path/'manifest.json')
        if proof['files']['evaluation.json']!=photo.digest(self.path/'evaluation.json'):raise ValueError('Nieprawidłowy SHA raportu analizy.')
        self.rows={r['exposure_id']:r for r in self.report['rows']};self.frames={f['frame_key']:f for f in self.report['frames']}
        self.prepared=Path(self.report['prepared_path']);self.manifest,self.material,self.reference=reference.load(self.prepared,strict_reference=False,
            reference_path=self.report['reference_preflight'].get('reference_path') if self.report['reference_preflight']['provenance']=='EXPLICIT_ANALYSIS_OVERRIDE' else None)
        self.exposures={e['exposure_id']:e for e in self.manifest['exposures']};self.pictures=[];self.thumbnails={}
        self.window=tk.Toplevel(parent);self.window.title('Wyniki pokazu i ocena względem GT');self.window.geometry('1380x900+35+35')
        self.window.rowconfigure(2,weight=1);self.window.columnconfigure(0,weight=1)
        s=self.report['summary'];gt=self.report['reference_preflight'];pc=sum(p['status']=='COMPLETED' for p in self.report.get('presentation_timeline',[]))
        status=dict(complete='kompletne',partial='częściowe',none='brak',invalid='niezgodne')[gt['mode']]
        rate=s['system_full_read']['rate'];rate_text='N/A' if rate is None else f'{rate:.1%}'
        self.summary_text=(f"{self.report.get('session_id','Import niekompletny')} · run {self.report.get('run_id','N/A')[:12]}\n"
            f"POKAZ PC: {pc}/{len(self.rows)} COMPLETED | ANDROID: {s['identified_exposures']} fotografii | GT: {status} | SKUTECZNOŚĆ: {rate_text}\n"
            f"MT: {s.get('raw_mt_calls',0)} wywołań · Kwalifikujące pierwsze MT: {s['selected_mt_exposures']} · MZ: {s.get('raw_mz_calls',0)} wywołań\n"
            f"Klatki P: {s.get('identified_p_frames',0)} · T: {s.get('technical_frames',0)} · bez identyfikacji: {s.get('unidentified_frames',0)}\n"
            +(f"N/A pierwszego MT: {s.get('unlocated_mt_calls',0)} wywołania bez identyfikacji — brak dowodu, które wejście było pierwsze.\n" if 'EARLIER_MT_INPUT_MAY_BE_UNLOCATED' in self.report.get('integration_gaps',[]) else '')
            +('Automatyczna ocena względem istniejącego GT.' if gt['mode'] in ('complete','partial') else 'Brak poprawnej referencji — detekcje dostępne, metryki wymagają GT.')
            +f"  Metryki: {self.report['metrics_status']} · {self.manifest['layout'].get('layout_version','F6v1')} / {self.manifest['layout'].get('qr_version','F6v1')}")
        if self.report.get('readability_preflight'):
            qr=self.report['readability_preflight']['summary']
            self.summary_text+=f"\nF3: poprawne pary L/R + geometria {qr['verified_pairs']}/{qr['source_frames']} klatek · szczegóły w readability_preflight.json · A12/AF: NOT_VERIFIED"
        ttk.Label(self.window,text=self.summary_text,wraplength=1300,justify='left',padding=10).grid(row=0,column=0,sticky='ew')
        bar=ttk.Frame(self.window,padding=6);bar.grid(row=1,column=0,sticky='ew')
        self.filter=tk.StringVar(value='Wszystkie');combo=ttk.Combobox(bar,textvariable=self.filter,values=self.FILTERS,state='readonly',width=22);combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda _:self.populate())
        if on_reference:
            button=ttk.Button(bar,text='Utwórz / uzupełnij GT',command=on_reference);button.pack(side='left',padx=8)
            def choose_reference():
                path=filedialog.askopenfilename(parent=self.window,title='Przelicz z jawnie wybraną referencją PhotoReference',filetypes=[('PhotoReference JSON','*.json')])
                if path:on_reference(Path(path))
            menu=tk.Menu(button,tearoff=False);menu.add_command(label='Przelicz z zapisanym GT JSON…',command=choose_reference)
            button.bind('<Button-3>',lambda event:menu.tk_popup(event.x_root,event.y_root))
        if on_manual:ttk.Button(bar,text='Opcjonalna kontrola ręczna',command=on_manual).pack(side='left')
        ttk.Label(bar,text='Raport: '+str(self.path),wraplength=650).pack(side='right')
        work=ttk.Panedwindow(self.window,orient='horizontal');work.grid(row=2,column=0,sticky='nsew',padx=8,pady=6)
        left=ttk.Frame(work);left.rowconfigure(0,weight=1);left.columnconfigure(0,weight=1);work.add(left,weight=1)
        self.tree=ttk.Treeview(left,columns=('gt','mt','mz','status'),show='tree headings',selectmode='browse')
        self.tree.heading('#0',text='Zdjęcie');self.tree.column('#0',width=100,stretch=False)
        for key,label,width in [('gt','GT',120),('mt','MT',38),('mz','MZ',38),('status','Ocena',145)]:self.tree.heading(key,text=label);self.tree.column(key,width=width,stretch=key=='status')
        style=ttk.Style(self.window);style.configure('FinalResult.Treeview',rowheight=42);self.tree.configure(style='FinalResult.Treeview')
        self.tree.grid(row=0,column=0,sticky='nsew');scroll=ttk.Scrollbar(left,command=self.tree.yview);scroll.grid(row=0,column=1,sticky='ns');self.tree.configure(yscrollcommand=scroll.set)
        right=ttk.Frame(work);right.columnconfigure(0,weight=1);right.rowconfigure(1,weight=1);work.add(right,weight=2)
        self.selection_info=tk.StringVar();ttk.Label(right,textvariable=self.selection_info,wraplength=820,justify='left').grid(row=0,column=0,sticky='ew')
        photos=ttk.Frame(right);photos.grid(row=1,column=0,sticky='nsew');photos.columnconfigure(0,weight=1);photos.columnconfigure(1,weight=1)
        ttk.Label(photos,text='Oryginalne zdjęcie').grid(row=0,column=0);ttk.Label(photos,text='Klatka telefonu / dowód źródłowy').grid(row=0,column=1)
        self.original_label=ttk.Label(photos);self.original_label.grid(row=1,column=0,sticky='n');self.camera_label=ttk.Label(photos);self.camera_label.grid(row=1,column=1,sticky='n')
        self.details=tk.Text(right,height=10,wrap='word');self.details.grid(row=2,column=0,sticky='ew')
        self.tree.bind('<<TreeviewSelect>>',self.select);self.populate()

    def matches(self,row):
        f=self.filter.get();reason=(row.get('reason','')+' '+row.get('evidence_reason','')).upper()
        checks={'Wszystkie':True,'Poprawne':row['status']=='MEASURED' and not any(row.get(k) for k in ('fp','fn','wrong_reads','missing_reads')),
            'Błędne':any(row.get(k) for k in ('fp','fn','wrong_reads','missing_reads')) or row['status']=='SYSTEM_FAILURE',
            'Pominięte GT':bool(row.get('fn')),'FP':bool(row.get('fp')),'Negatywy':row.get('gt_plate_count')==0,
            'Nieczytelne':any(p.get('text_state')=='UNREADABLE_TEXT' for p in row['plates']),
            'Brak dowodów':row['status']=='N/A','QR / ArUco':any(x in reason for x in ('QR','ARUCO','UNLOCATED','UNASSIGNED','UNREADABLE')),
            'Anulowane':'CANCELLED' in reason or any(self.frames[k].get('stale_or_cancelled') for k in row['identified_frame_keys']),
            'N/A / brak GT':row['status']=='N/A'}
        return bool(checks[f])

    def populate(self):
        selection=self.tree.selection();previous=selection[0] if selection else None
        self.tree.delete(*self.tree.get_children())
        for eid,row in self.rows.items():
            if not self.matches(row):continue
            if eid not in self.thumbnails:
                try:
                    im=self.material.image(self.exposures[eid]['image_id']);im.thumbnail((50,36));self.thumbnails[eid]=ImageTk.PhotoImage(im,master=self.window)
                except (OSError,ValueError):self.thumbnails[eid]=None
            gt={'GT_COMPLETE':str(row['gt_plate_count'])+' · kompletne','GT_MISSING':'GT niedostępne','GT_INCOMPLETE':'GT niepełne','GT_CONFLICT':'Konflikt GT'}[row['gt_status']]
            label='Brak GT' if row['gt_status']=='GT_MISSING' else 'Błąd systemu' if row['status']=='SYSTEM_FAILURE' else 'Ocenione' if row['status']=='MEASURED' else 'Niezidentyfikowane' if not row['identified_frame_keys'] else 'N/A'
            kwargs=dict(text=str(row['position']),values=(gt,row['known_mt_count'],row['known_mz_count'],label))
            if self.thumbnails[eid] is not None:kwargs['image']=self.thumbnails[eid]
            self.tree.insert('','end',iid=eid,**kwargs)
        children=self.tree.get_children()
        if children:self.tree.selection_set(previous if previous in children else children[0]);self.select()
        else:
            self.selection_info.set('Brak ekspozycji spełniających filtr.');self.pictures=[]
            self.original_label.configure(image='',text='');self.camera_label.configure(image='',text='');self.details.delete('1.0','end')

    def select(self,_event=None):
        chosen=self.tree.selection()
        if not chosen or chosen[0] not in self.rows:return
        row=self.rows[chosen[0]];e=self.exposures[chosen[0]];frame=row.get('evidence') or self.frames.get(row.get('diagnostic_frame_key'))
        qualified=row['status']=='MEASURED' and row.get('evidence') is not None
        reasons=[row['reason'] or row.get('evidence_reason','')]
        if row.get('gt_reason'):reasons.append('GT: '+row['gt_reason'])
        self.selection_info.set(f"Zdjęcie {row['position']} · {row['gt_status']} · {row['status']}\n"
            +('Automatyczna ocena GT / FINAL / MZ' if qualified else 'Dowód diagnostyczny — bez przypisywania TP/FP/FN')
            +('\n'+' · '.join(filter(None,reasons)) if any(reasons) else ''))
        self.pictures=[];self.original_label.configure(image='',text='');self.camera_label.configure(image='',text='Brak przypisanej klatki')
        def display(label,image):
            image.thumbnail((385,480));picture=ImageTk.PhotoImage(image,master=self.window);label.configure(image=picture,text='');self.pictures.append(picture)
        try:
            original=self.material.image(e['image_id'])
            if self.reference is not None and row['gt_status']=='GT_COMPLETE':
                draw=ImageDraw.Draw(original)
                for obj in self.reference.state(e['image_id'])['objects']:
                    if obj['quad']:q=[tuple(p) for p in obj['quad']];draw.line(q+[q[0]],fill='#22bb44',width=3)
            display(self.original_label,original)
            if frame:
                import io,zipfile
                record=frame['source_image']
                with zipfile.ZipFile(self.report['source_archive_path']) as archive:raw=archive.read(record['entry'])
                if __import__('hashlib').sha256(raw).hexdigest()!=record['sha256']:raise ValueError('SHA klatki niezgodny z raportem.')
                with Image.open(io.BytesIO(raw)) as image:camera=image.convert('RGB')
                draw=ImageDraw.Draw(camera);region=frame.get('input_region')
                if region:draw.rectangle(tuple(region),outline='#ffdd00',width=3)
                finals={c['attempt_id']:c for c in frame.get('finals',[])}
                for c in finals.values():draw.rectangle(tuple(c['bbox']),outline='#ff8833',width=3)
                if qualified:
                    for p in row['plates']:
                        geom=p.get('geometry') or {};q=geom.get('quad')
                        if q:q=[tuple(x) for x in q];draw.line(q+[q[0]],fill='#22dd66',width=3)
                        c=finals.get(p.get('attempt_id'))
                        if c and q:
                            b=c['bbox'];draw.line((sum(x for x,y in q)/4,sum(y for x,y in q)/4,(b[0]+b[2])/2,(b[1]+b[3])/2),fill='#33ddff',width=2)
                display(self.camera_label,camera)
        except (ValueError,OSError,KeyError) as exc:self.selection_info.set(self.selection_info.get()+'\nNie odczytano dowodu: '+str(exc))
        lines=[]
        def shown(value):return 'N/A' if value is None else value
        for p in row['plates']:
            pair=next((x for x in row.get('matching',{}).get('matched',[]) if x['plate_id']==p['plate_id']),{})
            lines.append(f"{p['detection']} · GT: {p.get('gt_text') or p['text_state']} → MZ: {shown(p.get('raw_prediction'))} · IoU: {shown(pair.get('iou'))} · CER: {shown(p.get('cer'))}")
        if not qualified:
            lines.extend('Wynik źródłowy MZ (bez oceny): '+str(v.get('raw_prediction') if v.get('raw_prediction') else 'pusty odczyt' if v.get('raw_prediction')=='' else 'N/A') for v in self.report.get('raw_mz',[]) if v.get('exposure_id')==e['exposure_id'])
        lines.extend(['',f"Zdjęcie SHA: {e['photo_sha256']}",f"Klatka: {frame['frame_key'] if frame else 'N/A'}",f"MT: {', '.join(frame.get('mt_invocation_ids',[])) if frame else 'N/A'}"])
        self.details.delete('1.0','end');self.details.insert('1.0','\n'.join(lines))


def make_app(legacy, args, parent=None):
    class FinalApp(MaterialPresentationActions, legacy.TvApp):
        def __init__(self):
            self.tk,self.ttk=tk,ttk
            # The main Desktop has already fixed its process DPI policy. Only
            # standalone startup may configure it, before creating any Tk root.
            if parent is None and tk._default_root is None:monitor_dpi.enable_windows_dpi_awareness()
            self.window=tk.Toplevel(parent) if parent is not None else tk.Tk();self.window.title('Pokaz fotografii i ocena klatki źródłowej')
            self.window.geometry('1240x780');self.window.minsize(960,620)
            self.monitors=[];self.monitor_names=[];self._selected_device=None;self._monitor_error=''
            self._refreshing_monitors=False;self._last_monitor_probe=0;self._viewer_geometry=None;self._opening_viewer=False
            self._post_roll_hold=False
            self.screen_choice=tk.StringVar(value='');self.monitor=None
            self.viewer=None;self.canvas=None;self.photos=[];self.mode='IDLE';self.run=None;self.cache=None
            self.current_index=0;self.pending_id=None;self.stage=None;self.transition_end=0;self.closed=False
            self.audit_result=None;self.prepared=None;self.manifest=None;self.material=None
            self.gt_preflight=None;self.result_windows=[];self.analysis_reference=None
            self.source_material=None;self.recursive_photographs=tk.BooleanVar(value=False)
            self._loading=False;self._pending_start=False;self._generation=0
            self._selection_generation=0;self._selection_loading=False;self._selection_error=''
            self.calibration_preview=None
            self.queue=queue.SimpleQueue();self.workers=ThreadPoolExecutor(max_workers=1)
            self.paths={k:tk.StringVar(value=str(getattr(args,k) or '')) for k in ('xml','positives','negatives','output')}
            self.dwell=tk.StringVar(value=str(args.dwell));self.negative_confirmed=tk.BooleanVar(value=args.negatives_declared)
            self._locked_dwell=args.dwell
            self.status=tk.StringVar(value='Wybierz katalog fotografii lub otwórz zapisany pokaz. Dane referencyjne są opcjonalne.')
            self.clock=tk.StringVar(value='')
            self.plan=tk.StringVar(value='Plan zostanie odczytany z zamrożonego manifestu.')
            self.reference_info=tk.StringVar(value='Materiał: nie wybrano fotografii | GT: nieprzypisane')
            self.set_info=tk.StringVar(value='Wybrany zestaw: brak');self.selected_folder=tk.StringVar(value='Wybierz katalog ze zdjęciami lub otwórz zapisany pokaz.')
            self.advanced_visible=tk.BooleanVar(value=False)
            self.ignore_reference=tk.BooleanVar(value=False)
            self.unreadable_ids=set(args.unreadable_ids)
            self.unreadable_xml_sha=getattr(args,'unreadable_xml_sha',None)
            def changed(*_):
                if not self._loading and self.mode not in ACTIVE_MODES:
                    self.audit_result=None;self.source_material=None
                    self.close_calibration();self._selection_generation+=1;self._selection_loading=False;self._selection_error=''
                    self.prepared=None;self.manifest=None;self.material=None;self.gt_preflight=None
                    self.set_info.set('Wybrany zestaw: brak — nowe materiały');self.selected_folder.set('Zaawansowane: przygotowanie nowego zestawu.')
                    self.reference_info.set('Materiał zmieniony — wykonaj kontrolę | GT: niezweryfikowane')
                    self.refresh_plan()
            for key in ('xml','positives','negatives'):self.paths[key].trace_add('write',changed)
            self.paths['negatives'].trace_add('write',lambda *_:self.negative_confirmed.set(False))
            self._build_final();self.refresh_monitors();self.window.protocol('WM_DELETE_WINDOW',self._close);self._poll()
            self.dwell.trace_add('write',self.dwell_changed)
            self.dwell_entry.bind('<FocusOut>',self.remember_dwell)
            self.dwell_entry.bind('<Return>',self.remember_dwell)
            self.screen_choice.trace_add('write',self.monitor_changed)
            self.ignore_reference.trace_add('write',lambda *_:None if self._loading else self.refresh_plan())

        def _build_final(self):
            from .photo_presentation_layout import build
            build(self)

        def toggle_advanced(self):
            if self.mode in ACTIVE_MODES or self._post_roll_hold:return
            visible=not self.advanced_visible.get();self.advanced_visible.set(visible)
            if visible:
                self.material_body.grid_remove();self.advanced.grid()
            else:
                self.advanced.grid_remove();self.material_body.grid()
            self.advanced_btn.configure(text='Wróć do pokazu' if visible else 'Zaawansowane')

        def dwell_changed(self,*_):
            if self._loading:return
            if self.mode in ACTIVE_MODES:
                self._loading=True
                self.dwell.set(str(self._locked_dwell))
                self._loading=False
                return
            self.refresh_plan()

        def remember_dwell(self,*_):
            if self._loading or self.mode in ACTIVE_MODES:return
            try:remember_selection(reference.parse_dwell(self.dwell.get()))
            except ValueError as exc:self.status.set(str(exc))

        def refresh_plan(self):
            self.calibration_btn.configure(state='normal' if self.prepared and self.mode not in ACTIVE_MODES else 'disabled')
            try:seconds=reference.parse_dwell(self.dwell.get())
            except ValueError as exc:
                self.plan.set(str(exc));self.friendly_plan.set(str(exc));self.start_btn.configure(state='disabled');return
            count=len(self.manifest['exposures']) if self.manifest else len((self.audit_result or {}).get('images',[]))
            times=reference.plan_times(count,seconds,countdown_seconds=self.manifest['countdown_seconds'] if self.manifest else 5,
                transition_seconds=self.manifest['transition_seconds'] if self.manifest else .5)
            dirty=self.manifest is not None and seconds!=self.manifest['dwell_seconds']
            note=(f" Zmiana czasu: zapisany plan {self.manifest['dwell_seconds']:g} s; Przygotuj/START utworzy nową wersję." if dirty else '')
            geometry='';geometry_ok=True
            try:
                monitor=self._current_monitor();monitor_dpi.validate_final_monitor(monitor,self.manifest['layout'] if self.manifest else None)
                if self.manifest:geometry=f"Plan: {self.manifest['layout']['width']}×{self.manifest['layout']['height']} · {self.manifest['layout'].get('layout_version','F6v1')}; {monitor_dpi.monitor_description(monitor)}.\n"
            except ValueError as exc:geometry_ok=False;geometry=str(exc)+'\n'
            self.plan.set(geometry+f"{count} fotografii × {seconds:g} s; zdjęcia: {times['planned_photo_seconds']:g} s ({reference.format_seconds(times['planned_photo_seconds'])}); plan: {times['planned_total_seconds']:g} s ({reference.format_seconds(times['planned_total_seconds'])})."+note)
            self.friendly_plan.set(f"{count} zdjęć · łączny czas {reference.format_seconds(times['planned_total_seconds'])}\nZ odliczaniem i przerwami." if count else 'Wybierz fotografie, aby obliczyć czas.')
            enabled=(bool(self.source_material) or (self.prepared and reference.display_ready(self.manifest))) and self.mode not in ACTIVE_MODES and not self._post_roll_hold
            if enabled and self.gt_preflight and self.gt_preflight['mode']=='invalid':enabled=self.ignore_reference.get()
            selection_ok=not self._selection_loading and not self._selection_error
            self.start_btn.configure(state='normal' if enabled and geometry_ok and selection_ok else 'disabled')
            self.calibration_btn.configure(state='normal' if self.calibration_preview is not None or (self.prepared and geometry_ok and selection_ok and self.mode not in ACTIVE_MODES and not self._post_roll_hold) else 'disabled')
            self.choose_set_btn.configure(state='disabled' if self.mode in ACTIVE_MODES or self._post_roll_hold else 'normal')
            self.choose_material_btn.configure(state='disabled' if self.mode in ACTIVE_MODES or self._post_roll_hold or self._selection_loading else 'normal')
            self.reference_menu_btn.configure(state='normal' if self.source_material is not None and self.mode=='IDLE' and not self._post_roll_hold else 'disabled')
            self.import_btn.configure(state='disabled' if self.mode in ACTIVE_MODES or self._selection_loading else 'normal')
            self.sync_presentation_ui(geometry_ok=geometry_ok,selection_ok=selection_ok)

        def _current_monitor(self):
            for m in self.monitors:
                if m['name']==self._selected_device:return m
            raise ValueError('NO_GO: '+(self._selected_device or 'Monitor')+' — odłączony lub niedostępny. '+self._monitor_error)

        def monitor_changed(self,*_):
            if self._refreshing_monitors:return
            if self.screen_choice.get() in self.monitor_names:
                self._selected_device=self.monitors[self.monitor_names.index(self.screen_choice.get())]['name']
            self.close_calibration();self.refresh_plan()

        def refresh_monitors(self):
            self._last_monitor_probe=time.monotonic()
            try:self.monitors=monitor_dpi.final_monitors();self._monitor_error=''
            except (OSError,ValueError,AttributeError) as exc:self.monitors=[];self._monitor_error=str(exc)
            if self._selected_device is None and self.monitors:self._selected_device=self.monitors[-1]['name']
            self.monitor_names=[monitor_dpi.monitor_description(m) for m in self.monitors]
            current=next((m for m in self.monitors if m['name']==self._selected_device),None)
            self._refreshing_monitors=True
            try:
                self.monitor_combo.configure(values=self.monitor_names)
                self.screen_choice.set(monitor_dpi.monitor_description(current) if current else (self._selected_device or 'Monitor')+' · niedostępny')
            finally:self._refreshing_monitors=False
            def identity(m):
                return tuple(m.get(k) for k in ('name','x','y','w','h','hmonitor','verified'))+(m.get('display_path',{}).get('device_path'),)
            if self.viewer is not None and self.monitor and (current is None or identity(current)!=identity(self.monitor)):
                self.abort();self.status.set('NO_GO: zmienił się monitor lub jego geometria. Widok zamknięty; wybierz zgodny monitor/plan.')
            self.refresh_plan()

        def _select_monitor(self):
            self.refresh_monitors()
            return monitor_dpi.validate_final_monitor(self._current_monitor())

        def _position_viewer(self,screen):
            # Tk's pack allocates child pixels on MapNotify. Verify HWND before
            # showing; the full Canvas check below runs before drawing any raster.
            self._viewer_geometry=monitor_dpi.final_window_geometry(self.viewer,self.canvas,screen,position=True,check_canvas=False)

        def _make_viewer(self,screen):
            monitor_dpi.validate_final_monitor(screen,self.manifest['layout'])
            self._opening_viewer=True
            try:
                with monitor_dpi.physical_pixels():
                    super()._make_viewer(screen)
                    self._viewer_geometry=monitor_dpi.final_window_geometry(self.viewer,self.canvas,screen)
            except Exception:
                self._dispose_viewer();raise
            finally:self._opening_viewer=False

        def _dispose_viewer(self):
            super()._dispose_viewer();self._viewer_geometry=None;self._post_roll_hold=False

        def toggle_calibration(self):
            if self.calibration_preview is not None:self.close_calibration()
            else:self.show_calibration()

        def show_calibration(self):
            if self.mode in ACTIVE_MODES:return
            if self._post_roll_hold:self.status.set('Plansza T pozostaje widoczna. Zakończ nagrywanie na telefonie, potem naciśnij STOP.');return
            if self._selection_loading or self._selection_error:self.status.set(self._selection_error or 'Trwa sprawdzanie wybranego zestawu.');return
            if not self.prepared:self.status.set('Wczytaj przygotowanie FINAL, aby ustawić kadr.');return
            if self.calibration_preview is not None:return
            try:
                # Read the currently loaded plan; no selection, run or acceptance is changed.
                manifest,_,_=reference.load(self.prepared,strict_reference=False)
                monitor=self._select_monitor()
                monitor_dpi.validate_final_monitor(monitor,manifest['layout'])
                frame,identity=calibration_frame(manifest,photo.digest(self.prepared/'manifest.json'))
                previous=dict(mode=self.mode,status=self.status.get(),clock=self.clock.get())
                self._dispose_viewer();self.monitor=monitor
                self.calibration_preview=dict(identity=identity,previous=previous);self.mode='CALIBRATION'
                self._make_viewer(monitor)
                # Legacy viewer's space key starts a show; it is deliberately absent here.
                self.viewer.unbind('<space>')
                self.viewer.bind('<Escape>',lambda _:self.close_calibration())
                self.viewer.protocol('WM_DELETE_WINDOW',self.close_calibration)
                self.draw(frame)
                if manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS:
                    from ..ranking.final180_timeline import register_calibration
                    register_calibration(self.prepared,identity)
                self.calibration_btn.configure(text='Zamknij planszę')
                self.status.set('Plansza TEST_ONLY — bez odliczania i przebiegu badawczego. Ustaw w Androidzie rogi wewnętrznego pola fotografii; status odbioru pozostaje bez zmian.')
                self.clock.set('Nieruchoma plansza · Esc lub „Zamknij planszę” · właściwy START wyłącznie z panelu')
            except Exception as exc:
                self.close_calibration();self.status.set('Nie otwarto planszy: '+str(exc))

        def close_calibration(self):
            if self.calibration_preview is None:return
            previous=self.calibration_preview['previous']
            self.calibration_preview=None;self._dispose_viewer();self.mode=previous['mode']
            self.calibration_btn.configure(text='Sprawdź kadr na ekranie')
            self.status.set(previous['status']);self.clock.set(previous['clock'])
            self.refresh_plan()

        def browse(self,key):
            if self.mode in ACTIVE_MODES:return
            path=filedialog.askopenfilename(parent=self.window,filetypes=[('XML','*.xml')]) if key=='xml' else filedialog.askdirectory(parent=self.window)
            if path and key=='output' and (Path(path)/'manifest.json').is_file():
                self.status.set('To gotowy zestaw. Otwórz go przyciskiem „Wybierz zestaw”. Folder bazowy służy wyłącznie do zapisu.');return
            if path:self.paths[key].set(path);self.audit_result=None

        def background(self,work,done,failed=None):
            if self.mode in ACTIVE_MODES:return
            self.status.set('Pracuję… źródła pozostają tylko do odczytu.')
            def run():
                try:self.queue.put((done,work(),None))
                except Exception as exc:
                    if isinstance(exc,PreparedSelectionError):logging.getLogger(__name__).warning('FINAL selection rejected: %s',exc)
                    else:logging.getLogger(__name__).exception('FINAL background operation failed')
                    self.queue.put((failed,str(exc),None) if failed else (None,None,str(exc)))
            self.workers.submit(run)

        def _poll(self):
            if self.closed:return
            try:
                while True:
                    done,value,error=self.queue.get_nowait()
                    if error:self.status.set('Nie wykonano: '+error)
                    else:done(value)
            except queue.Empty:pass
            except Exception as exc:
                logging.getLogger(__name__).exception('FINAL UI callback failed')
                self.status.set('Nie wykonano: '+str(exc))
            finally:
                if not self.closed:self.window.after(60,self._poll)
            if not self.closed and time.monotonic()-self._last_monitor_probe>=2:self.refresh_monitors()

        def show_details(self,value):
            self.details.delete('1.0','end');self.details.insert('1.0',value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,default=str))

        def check(self):
            if self.mode in ACTIVE_MODES or self._selection_loading:return
            if self.prepared:
                try:
                    self.loaded(self.prepared);self.status.set('Materiał sprawdzony. Gotowość GT i dowodów Androida jest oceniana osobno.')
                except Exception as exc:self.status.set(str(exc))
                return
            values={k:v.get() for k,v in self.paths.items()};negative=self.negative_confirmed.get()
            if not values['xml'].strip() or self.ignore_reference.get():
                def check_photos():
                    if not values['positives'].strip():raise ValueError('Najpierw wskaż katalog fotografii.')
                    photos=reference.files(values['positives'])
                    if values['negatives'].strip():photos=list(dict.fromkeys(photos+reference.files(values['negatives'])))
                    if not photos:raise ValueError('Nie znaleziono fotografii.')
                    return dict(no_reference=True,source=values['positives'],summary=dict(photographs=len(photos)),errors=[],images=photos)
                def checked(value):
                    self.audit_result=value;self.reference_info.set(f"Materiał: {len(value['images'])} fotografii | GT: brak")
                    self.show_details('GT nieprzypisane. Pokaz dostępny, metryki niedostępne do czasu anotacji.');self.refresh_plan()
                self.background(check_photos,checked);return
            ids=self.unreadable_ids if self.unreadable_xml_sha and Path(values['xml']).is_file() and photo.digest(values['xml'])==self.unreadable_xml_sha else set()
            def done(a):
                self.audit_result=a;self.status.set('Materiał: '+a['status']+'; żadne puste rekordy nie zostały pominięte.')
                gt=reference.audit_reference_preflight(a);self.gt_preflight=gt
                label=dict(complete='kompletne',partial='częściowe',none='brak',invalid='niezgodne')[gt['mode']]
                self.reference_info.set(f"Materiał: {len(a['images'])} fotografii | GT: {label}")
                self.refresh_plan()
                self.show_details(f"GT: {label}\nKompletne: {gt['counts']['GT_COMPLETE']} · niepełne: {gt['counts']['GT_INCOMPLETE']} · brak: {gt['counts']['GT_MISSING']} · konflikt: {gt['counts']['GT_CONFLICT']}\n"
                    +'Istnienie ramek w XML nie dowodzi zatwierdzenia całej fotografii.\n'+gt['error']+'\n'+ '\n'.join(a['errors']))
            self.background(lambda:reference.audit(values['xml'],values['positives'],values['negatives'] or None,
                negative_declared=negative,unreadable_plate_ids=ids,expected_negative=None),done)

        def prepare(self,*,start_after=False):
            if self.mode in ACTIVE_MODES or self._selection_loading:return
            if self.source_material is not None and self.prepared is None:
                return self.prepare_material(start_after=start_after)
            self.close_calibration()
            if not self.prepared and not self.audit_result:self.status.set('Najpierw wykonaj kontrolę materiału.');return
            try:
                seconds=reference.parse_dwell(self.dwell.get());screen=self._select_monitor();root=self.paths['output'].get()
                if (Path(root)/'manifest.json').is_file():raise ValueError('Folder bazowy nie może być gotowym prepared_*. Wskaż katalog nadrzędny; gotowy zestaw otwiera „Wybierz zestaw”.')
                if start_after and self.prepared:monitor_dpi.validate_final_monitor(screen,self.manifest['layout'])
            except (ValueError,TypeError) as exc:self.status.set('Nieprawidłowy czas lub monitor: '+str(exc));return
            source=self.prepared;audited=self.audit_result;ignore=self.ignore_reference.get();source_dir=self.paths['positives'].get();additional_dir=self.paths['negatives'].get()
            self._locked_dwell=seconds
            self._generation+=1;generation=self._generation;self._pending_start=start_after
            self.mode='PREPARING';self.dwell_entry.configure(state='disabled');self.start_btn.configure(state='disabled');self.stop_btn.configure(state='normal')
            self.calibration_btn.configure(state='disabled')
            self.status.set('Przygotowuję nową wersję planu; poprzednia pozostaje zachowana.')
            def done(path):
                if generation!=self._generation:return
                self.mode='IDLE';self.dwell_entry.configure(state='normal');self.stop_btn.configure(state='disabled')
                self.loaded(path);remember_selection(seconds,path)
                if self._pending_start:self._pending_start=False;self.begin_countdown()
            def failed(error):
                if generation!=self._generation:return
                self.mode='IDLE';self._pending_start=False;self.dwell_entry.configure(state='normal');self.stop_btn.configure(state='disabled')
                self.status.set('Nie przygotowano planu: '+error);self.refresh_plan()
            def work():
                try:
                    if source:
                        upgrade=ignore or (not start_after and (self.manifest.get('schema')!=reference.SCHEMA_V2 or self.manifest['layout'].get('layout_version')!=markers.F3_LAYOUT or [screen['w'],screen['h']]!=[self.manifest['layout']['width'],self.manifest['layout']['height']]))
                        path=reference.relayout(source,(screen['w'],screen['h']),root=root,dwell_seconds=seconds,reference_aware=True,ignore_reference=ignore,layout_version=markers.F3_LAYOUT) if upgrade else reference.replan(source,seconds,root=root)
                    elif audited.get('no_reference') or ignore:path=reference.freeze_photographs(source_dir,root,screen=(screen['w'],screen['h']),dwell_seconds=seconds,additional_source=additional_dir or None,layout_version=markers.F3_LAYOUT)
                    else:path=reference.freeze(audited,root,screen=(screen['w'],screen['h']),dwell_seconds=seconds,layout_version=markers.F3_LAYOUT,reference_aware=True)
                    self.queue.put((done,path,None))
                except Exception as exc:self.queue.put((failed,str(exc),None))
            self.workers.submit(work)

        def loaded(self,path,*,validated=None):
            if self.mode in ACTIVE_MODES:return False
            path,manifest,material,gt=validated or read_prepared_selection(path)
            self.close_calibration()
            self._dispose_viewer()
            # No state mutation before all source/GT validation has succeeded.
            self.prepared=Path(path);self.manifest=manifest;self.material=material
            self.gt_preflight=gt;self.audit_result=None;self.run=None
            if self.cache:self.cache.close();self.cache=None
            self._loading=True
            try:
                self.ignore_reference.set(False);self.dwell.set(str(manifest['dwell_seconds']));self.paths['output'].set(str(path.parent))
            finally:self._loading=False
            self._selection_loading=False;self._selection_error=''
            variant='F3' if manifest['layout'].get('layout_version')==markers.F3_LAYOUT else 'F2' if manifest['layout'].get('layout_version') in markers.F2_LAYOUTS else 'F6v1'
            name=f"FINAL{len(manifest['exposures'])}" if manifest.get('mode')=='FINAL' else f"Zestaw {len(manifest['exposures'])}"
            self.set_info.set(f'Wybrany zestaw: {name} · {variant}');self.selected_folder.set(path.name)
            self.status.set(f'Wczytano {variant}: {path.name}. Gotowość A12/TV: '+manifest['acceptance']['a12_tv'])
            self.clock.set('')
            names=dict(complete='kompletne',partial='częściowe',none='brak',invalid='niezgodne');gt=self.gt_preflight
            self.reference_info.set(f"Materiał: {len(manifest['exposures'])} fotografii | GT: {names[gt['mode']]}")
            notes=dict(complete='Gotowe do automatycznej oceny po imporcie, o ile uda się przypisać klatki i dowody.',
                partial='Pokaz dostępny. Ocena tylko kompletnego podzbioru; pełna skuteczność N/A.',
                none='GT nieprzypisane. Pokaz dostępny; po imporcie detekcje bez metryk. GT można uzupełnić później.',
                invalid='Niezgodne GT: ocena zablokowana. Możesz świadomie utworzyć nowy plan bez przypiętego GT.')
            self.show_details(notes[gt['mode']]+'\n\n'+f"Kompletne: {gt['counts']['GT_COMPLETE']} · niepełne: {gt['counts']['GT_INCOMPLETE']} · bez GT: {gt['counts']['GT_MISSING']} · konflikt: {gt['counts']['GT_CONFLICT']}\n"
                +(f"Tablice w kompletnym podzbiorze GT: {gt['complete_gt_plates']} · jawne negatywy: {gt['negative_images']} · pełne teksty: {gt['full_text']} · nieczytelne: {gt['unreadable']}\n" if gt['mode'] not in ('none','invalid') else 'Liczba tablic GT: N/A. Nie stwierdzono automatycznie braku tablic.\n')
                +f"Źródło GT: {gt['reference_path'] or 'nieprzypisane'}\nSHA GT: {(gt['reference_sha256'] or 'N/A')[:16]}\n{gt['error']}\n\nA12/TV: {manifest['acceptance']['a12_tv']} — niezależne od gotowości materiału i GT.")
            from ..ranking.photo_material import material_for_presentation
            self.source_material=material_for_presentation(path)
            if self.source_material:
                self.recursive_photographs.set(self.source_material.data['recursive'])
                self.set_info.set(self.source_material.root.name)
                self.selected_folder.set(str(self.source_material.root))
                self.refresh_material_reference()
            self.refresh_plan()
            return True

        def selection_failed(self,path,error):
            self._selection_loading=False
            reason=str(error) or type(error).__name__
            retained=f' Poprzedni zestaw pozostaje wybrany: {self.prepared.name}.' if self.prepared else ''
            self._selection_error=f'Nie wczytano „{Path(path).name}”: {reason}.{retained} Wybierz poprawny zestaw, aby odblokować planszę i START.'
            self.status.set(self._selection_error);self.refresh_plan()

        def open_prepared(self):
            if self.mode in ACTIVE_MODES or self._post_roll_hold:return
            self.close_calibration()
            path=filedialog.askdirectory(parent=self.window,title='Otwórz folder prepared_* i kliknij „Wybierz folder”',initialdir=str(self.prepared if self.prepared else DEFAULT_EXPERIMENT),mustexist=True)
            if not path:return
            self._selection_generation+=1;generation=self._selection_generation
            self._selection_loading=True;self.refresh_plan()
            def failed(error):
                if generation==self._selection_generation and not self.closed:self.selection_failed(path,error)
            def chosen(value):
                if generation!=self._selection_generation or self.closed:return
                try:
                    if self.loaded(path,validated=value):remember_selection(self.manifest['dwell_seconds'],self.prepared)
                except Exception as exc:
                    logging.getLogger(__name__).exception('FINAL preparation activation failed');failed(exc)
            self.background(lambda:read_prepared_selection(path),chosen,failed)
            self.status.set('Sprawdzam zestaw: '+Path(path).name+'…')

        def begin_countdown(self,resume=None):
            if self.mode in ACTIVE_MODES or self._opening_viewer:return
            if self._post_roll_hold:self.status.set('Zakończ nagrywanie na telefonie i naciśnij STOP przed kolejnym pokazem.');return
            if self._selection_loading or self._selection_error:self.status.set(self._selection_error or 'Trwa sprawdzanie wybranego zestawu.');return
            if not resume and self.prepared is None and self.source_material is not None:
                return self.prepare_material(start_after=True)
            reuse_preview=bool(not resume and self.calibration_preview and self.manifest['layout'].get('layout_version') in markers.EDGE_LAYOUTS)
            if not reuse_preview:self.close_calibration()
            try:
                if resume:
                    state=photo.read_json(Path(resume)/'run.json')
                    self.loaded(state['prepared_path'])
                if not self.prepared:return
                if not resume and self.ignore_reference.get() and self.gt_preflight and self.gt_preflight['mode']!='none':self.prepare(start_after=True);return
                seconds=reference.parse_dwell(self.dwell.get())
                if not resume and seconds!=self.manifest['dwell_seconds']:
                    self.prepare(start_after=True);return
                self.manifest,self.material,_=reference.load(self.prepared,strict_reference=False)
                self.gt_preflight=reference.reference_preflight(self.prepared,reference_path=self.analysis_reference)
                if self.gt_preflight['mode']=='invalid':raise ValueError('Niezgodne GT. Wybierz jawnie nowy plan bez GT albo wskaż poprawne przygotowanie.')
                if not reference.display_ready(self.manifest):raise ValueError('NO_GO: nierozstrzygnięte wyjątki referencji.')
                self.monitor=self._select_monitor()
                monitor_dpi.validate_final_monitor(self.monitor,self.manifest['layout'])
                # No new research run until the actual HWND/Canvas raster is proven.
                if reuse_preview and self.calibration_preview and self.viewer is not None:
                    previous=self.calibration_preview['previous'];self.calibration_preview=None;self.mode=previous['mode']
                    self.calibration_btn.configure(text='Sprawdź kadr na ekranie');self.viewer.bind('<Escape>',lambda _:self.abort());self.viewer.protocol('WM_DELETE_WINDOW',self.abort)
                    if self._viewer_geometry is not None:monitor_dpi.final_window_geometry(self.viewer,self.canvas,self.monitor)
                else:self._dispose_viewer();self._make_viewer(self.monitor)
                self.run=PresentationRun(self.prepared,self.manifest,resume=resume)
                self._locked_dwell=self.manifest['dwell_seconds']
                if self.run.next_index>=len(self.manifest['exposures']):self.run.stop('NOTHING_TO_RESUME');self._dispose_viewer();return
                self.cache=PreparedFrames(self.material,self.manifest,self.run.manifest_sha,self.run.state['run_id'])
                self.cache.request(self.run.next_index);self.cache.request(self.run.next_index+1)
                self.mode='COUNTDOWN';self.stop_btn.configure(text='STOP')
                self.calibration_btn.configure(state='disabled')
                self.dwell_entry.configure(state='disabled')
                origin=self.run.replace(lambda:self._show_countdown(self.manifest['countdown_seconds']),phase='COUNTDOWN',reason='COUNTDOWN',display_token=self._countdown_token())
                self.run.plan_origin_ns=origin
                self.last_countdown=self.manifest['countdown_seconds']
                self.count_end=origin+int(self.manifest['countdown_seconds']*1e9)
                self.start_btn.configure(state='disabled');self.stop_btn.configure(state='normal');self.sync_presentation_ui();self._tick()
            except Exception as exc:self.abort();self.status.set(str(exc))

        def _tick(self):
            if self.closed or not self.run or self.mode not in ('COUNTDOWN','RUNNING','TRANSITION'):return
            now=time.perf_counter_ns()
            try:
                if self.mode=='COUNTDOWN':
                    left=max(0,(self.count_end-now+999999999)//1000000000)
                    if left!=self.last_countdown:
                        self.run.replace(lambda:self._show_countdown(left),phase='COUNTDOWN',reason='COUNTDOWN',display_token=self._countdown_token())
                        self.last_countdown=left
                    if now>=self.count_end:self.mode='TRANSITION';self.transition_end=now
                if self.mode=='RUNNING' and self.run.due():
                    self.run.end_photo()
                    if self.run.next_index==len(self.manifest['exposures']):
                        hold=self.manifest['layout'].get('layout_version') in markers.EDGE_LAYOUTS
                        if hold:
                            value=markers.payload(self.manifest,self.run.manifest_sha,self.run.state['run_id'],'POSTROLL','T')
                            image,_=markers.render(None,self.manifest['layout'],value)
                            self.run.complete(lambda:self.draw(image),display_token=value);self._post_roll_hold=True
                        else:self.run.complete(self.clear_view)
                        self.mode='COMPLETED';self.dwell_entry.configure(state='normal');self.refresh_plan()
                        total=len(self.manifest['exposures'])
                        self.status.set(f'Pokaz zakończony {total}/{total}. '+('Zatrzymaj pomiar na telefonie; możesz zamknąć planszę T.' if hold else 'Dziennik zapisany.'))
                        self.stop_btn.configure(text='Zamknij planszę' if hold else 'Zatrzymaj',state='normal' if hold else 'disabled');self.cache.close();self.cache=None
                        self.clock.set('Zakończono wszystkie fotografie. Czasy PC w osi czasu.');return
                    self.mode='TRANSITION'
                    next_e=self.manifest['exposures'][self.run.next_index]
                    value=markers.payload(self.manifest,self.run.manifest_sha,self.run.state['run_id'],next_e['exposure_id'],'T')
                    image,_=markers.render(None,self.manifest['layout'],value)
                    updated=self.run.replace(lambda:self.draw(image),display_token=value if self.manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS else None)
                    self.transition_end=updated+int(self.manifest['transition_seconds']*1e9)
                if self.mode=='TRANSITION' and time.perf_counter_ns()>=self.transition_end:
                    index=self.run.next_index
                    if self.cache.ready(index):
                        image,placement=self.cache.get(index);self.run.present(index,placement,lambda:self.draw(image),require_binding=True)
                        self.cache.request(index+1);self.mode='RUNNING'
                    else:self.status.set('Czekam na dekodowanie obrazu; żadna fotografia nie zostanie pominięta.')
                index=self.run.current if self.run.current is not None else min(self.run.next_index,len(self.manifest['exposures'])-1)
                name=self.material.images[self.manifest['exposures'][index]['image_id']]['source_relative_path']
                remaining=max(0,((self.run.deadline or time.perf_counter_ns())-time.perf_counter_ns())/1e9)
                phase={'COUNTDOWN':'Odliczanie','RUNNING':'Prezentacja','TRANSITION':'Przejście / przygotowanie'}[self.mode]
                self.clock.set(f"{phase} · Fotografia {index+1}/{len(self.manifest['exposures'])} · {name} · pozostało PC: {remaining:.1f} s")
                self.pending_id=self.window.after(30,self._tick)
            except Exception as exc:
                logging.exception('FINAL: błąd przebiegu');self.abort();self.mode='ERROR'
                self.status.set('Błąd przebiegu — zakończenie niepotwierdzone: '+str(exc))

        def _countdown_token(self):
            if self.manifest['layout'].get('layout_version') in markers.BOUND_LAYOUTS:
                return markers.payload(self.manifest,self.run.manifest_sha,self.run.state['run_id'],'COUNTDOWN','T')
            return None

        def _show_countdown(self,left):
            value=self._countdown_token()
            if value is None:return super()._show_countdown(left)
            image,_=markers.render(None,self.manifest['layout'],value)
            draw=ImageDraw.Draw(image);draw.text((image.width//2,image.height//2),str(left),anchor='mm',fill='white',font=ImageFont.load_default(size=48))
            self.draw(image)

        def draw(self,image):
            if self._viewer_geometry is not None:monitor_dpi.final_window_geometry(self.viewer,self.canvas,self.monitor)
            picture=ImageTk.PhotoImage(image,master=self.viewer);self.canvas.delete('all')
            self.canvas.create_image(0,0,image=picture,anchor='nw');self.photos=[picture]
            self.canvas.update_idletasks()

        def clear_view(self):
            if self.canvas is not None and self.viewer is not None:
                self.canvas.delete('all');self.canvas.configure(bg='black');self.canvas.update_idletasks();self.photos=[]

        def abort(self):
            if self.calibration_preview is not None:
                self.close_calibration();return
            self._generation+=1;self._pending_start=False
            if self.pending_id:
                self.window.after_cancel(self.pending_id);self.pending_id=None
            completed=bool(self.run and self.run.status=='COMPLETED' and not self.run.log_error)
            error=None
            try:
                if self.run:self.run.stop(clear=self.clear_view if self.viewer is not None else None)
            except Exception as exc:error=str(exc)
            if self.cache:self.cache.close();self.cache=None
            self.mode='COMPLETED' if completed else 'STOPPED';self.stop_btn.configure(state='disabled')
            self.dwell_entry.configure(state='normal')
            total=len(self.manifest['exposures']) if self.manifest else 0
            self.status.set(f'Pokaz zakończony {total}/{total}. Plansza zamknięta; dziennik COMPLETED zachowany.' if completed else 'Przerwany przebieg zachowany. Wznowienie tworzy jawny segment; nie powtarza częściowej ekspozycji.')
            self._dispose_viewer()
            self.refresh_plan()
            if error or (self.run and self.run.log_error):
                self.mode='ERROR';self.status.set('Błąd trwałego zapisu — zakończenie niepotwierdzone: '+str(error or self.run.log_error))

        def show_timeline(self):
            if not self.prepared:return
            from ..ranking import final180_timeline as timeline
            selected=filedialog.askopenfilename(parent=self.window,title='Dziennik pokazu',filetypes=[('Run','run.json')])
            if not selected:return
            try:
                state,events,_=timeline.read_run(selected,self.manifest,photo.digest(self.prepared/'manifest.json'))
                rows=timeline.project(self.manifest,events)
                for r in rows:r['file']=self.material.images[self.manifest['exposures'][r['position']-1]['image_id']]['source_relative_path']
                win=tk.Toplevel(self.window);win.title('Oś czasu FINAL — pomiar programu PC');win.geometry('1150x500')
                labels=dict(position='Pozycja',file='Plik',status='Stan',segment='Segment',start_segment_s='Początek PC [s]',
                    end_segment_s='Koniec PC [s]',duration_pc_s='Czas PC [s]',planned_duration_s='Plan [s]',schedule_delay_ns='Opóźnienie [ns]',reason='Przyczyna')
                cols=tuple(labels)
                table=ttk.Treeview(win,columns=cols,show='headings')
                for col in cols:table.heading(col,text=labels[col]);table.column(col,width=130,stretch=True)
                for r in rows:table.insert('', 'end',values=[r.get(c) if r.get(c) is not None else 'N/A' for c in cols])
                scroll=ttk.Scrollbar(win,orient='horizontal',command=table.xview);table.configure(xscrollcommand=scroll.set)
                table.pack(fill='both',expand=True);scroll.pack(fill='x')
                def save():
                    path=filedialog.asksaveasfilename(parent=win,initialfile='presentation_timeline.csv',defaultextension='.csv')
                    if path:timeline.export_csv(path,rows)
                ttk.Button(win,text='Eksportuj CSV',command=save).pack(pady=8)
            except Exception as exc:self.status.set(str(exc))

        def resume(self):
            if self.mode in ACTIVE_MODES:return
            path=filedialog.askdirectory(parent=self.window,title='Katalog istniejącego run.json i events.jsonl')
            if path:self.begin_countdown(resume=path)

        def import_android(self,*,archive=None,run=None,reference_path=None):
            if self.mode in ACTIVE_MODES:return
            archive=archive or filedialog.askopenfilename(parent=self.window,title='Eksport Androida',filetypes=[('ALPR','*.alprsession *.zip')])
            if not archive:return
            self.manual_run_btn.pack_forget();self._unmatched_archive=None
            from ..ranking.final180_mobile import analyze_import,locate_run
            scopes=[DEFAULT_EXPERIMENT,Path(self.paths['output'].get())]
            from ..ranking.photo_material import storage_root,material_for_presentation,reference_for_presentation
            scopes.extend(p for p in storage_root().glob('material-*/presentations') if p.is_dir())
            if self.prepared:scopes.append(self.prepared.parent)
            scopes.extend(p for p in (Path(__file__).resolve().parents[2]/'output').glob('*/technical_plans') if p.is_dir())
            self.status.set('Sprawdzam klatki źródłowe, dwa QR, ArUco i dziennik pokazu…')
            def work():
                match=locate_run(archive,scopes) if run is None else None
                if match is not None and match['status']!='PASS':return dict(match=match)
                selected=run or match['run_path']
                chosen_reference=reference_path
                if chosen_reference is None:
                    run_file=Path(selected);run_file=run_file/'run.json' if run_file.is_dir() else run_file
                    prepared=Path(photo.read_json(run_file)['prepared_path'])
                    material=material_for_presentation(prepared)
                    if material is not None:
                        chosen_reference=(material.default_output if material.reference_status()['state']=='invalid'
                            else reference_for_presentation(material,prepared))
                return dict(path=analyze_import(archive,selected,reference_path=chosen_reference,run_match=match),run=selected)
            def done(result):
                if 'path' not in result:
                    self._unmatched_archive=archive;self.manual_run_btn.pack(side='left',padx=8)
                    self.status.set('Nie udało się jednoznacznie odnaleźć pokazu. Możesz użyć „Wskaż ręcznie”; nieudowodnione przypisania pozostaną N/A.')
                    logging.warning('FINAL: automatyczne powiązanie runu: %s',result['match']);return
                path=result['path'];selected=result['run']
                self.status.set('Wyniki pokazu i ocena GT: '+str(path))
                panel=FinalResultsWindow(self.window,path,on_reference=lambda chosen=None:self.import_android(archive=archive,run=selected,reference_path=chosen) if chosen else self.edit_analysis_gt(path,archive,selected),on_manual=getattr(self,'manual_review_callback',None))
                self.result_windows.append(panel)
                from ..ranking.photo_material import note_predictions_exposed
                material=material_for_presentation(panel.report['prepared_path'])
                if material is not None:note_predictions_exposed(material,result_path=path)
            self.background(work,done)

        def choose_manual_run(self):
            if not self._unmatched_archive or self.mode in ACTIVE_MODES:return
            run=filedialog.askopenfilename(parent=self.window,title='run.json pokazu — klatki bez dowodu pozostaną N/A',filetypes=[('Run JSON','run.json')])
            if run:self.import_android(archive=self._unmatched_archive,run=run)

        def edit_analysis_gt(self,result_path,archive,run):
            """Existing Z4 editor, separate native reference version; no predictions."""
            import shutil
            from .z4_photo_reference import PhotoReferenceWindow
            report=photo.read_json(Path(result_path)/'evaluation.json');prepared=Path(report['prepared_path'])
            _,material,current=reference.load(prepared,strict_reference=False,reference_path=report['reference_preflight'].get('reference_path') if report['reference_preflight']['provenance']=='EXPLICIT_ANALYSIS_OVERRIDE' else None)
            folder=prepared.parent/'reference_work';folder.mkdir(exist_ok=True)
            target=folder/('reference_'+uuid.uuid4().hex+'.json')
            working=reference.working_reference(prepared,target,source_reference=current.path if current is not None else None,predictions_exposed=True)
            initial=photo.digest(target) if target.exists() else None
            editor=PhotoReferenceWindow(self.window,working.material,reference=working)
            def closed(event):
                if event.widget is not editor.window:return
                if target.is_file() and photo.digest(target)!=initial:
                    self.window.after_idle(lambda:self.import_android(archive=archive,run=run,reference_path=target))
            editor.window.bind('<Destroy>',closed,add='+')
            return editor

        def _close(self):
            self.abort();self.closed=True;self.material_preview.dispose();self.workers.shutdown(wait=False,cancel_futures=True);self.window.destroy()
    return FinalApp()


def accepted_arguments(overrides=None):
    """Shared defaults for the existing launcher and Desktop action."""
    evidence=Path(__file__).resolve().parents[2]/'output/F6_FINAL180_DESKTOP_20261112_001'
    from ..config import SESSION
    default_pointer=evidence.parent/'F6_OPERATOR_DWELL_20261112_001/prepared_path.txt'
    prepared=SESSION.get('final_presentation','prepared_path','') if SESSION else ''
    if not prepared and default_pointer.is_file():prepared=default_pointer.read_text('utf-8').strip()
    # A missing saved preparation must be reported, not silently replaced.
    overrides=list(overrides or [])
    if any(s in ('--positives','--xml') or s.startswith(('--positives=','--xml=')) for s in overrides):prepared=''
    arguments=['--output',str(Path(prepared).parent if prepared else DEFAULT_EXPERIMENT)]
    if prepared:arguments.extend(['--prepared',prepared])
    # A preparation is authoritative. Do not inject unrelated historical XML or
    # negative folders into a new generic material selection.
    if prepared and (evidence/'accepted_unreadable_declaration.json').is_file():arguments.extend(['--unreadable-declaration',str(evidence/'accepted_unreadable_declaration.json')])
    if not any(s=='--prepared' or s.startswith('--prepared=') for s in overrides):
        arguments.extend(['--dwell',str(preferred_dwell())])
    return arguments+overrides


def open_from_desktop(parent):
    return main(accepted_arguments(),parent=parent)


def authentic_final_bundle(bundle):
    if bundle is None or not getattr(getattr(bundle,'validation',None),'ok',False):return False
    collection=getattr(bundle,'collection_session',{}) or {};execution=collection.get('execution') or {}
    return execution.get('capture_profile',collection.get('capture_profile'))=='f6_final180_v1' and (getattr(bundle,'f6_evidence',{}) or {}).get('available') is True


def main(argv=None,*,parent=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-viewer',type=Path,default=LEGACY_DEFAULT)
    parser.add_argument('--xml',type=Path);parser.add_argument('--positives',type=Path);parser.add_argument('--negatives',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT_EXPERIMENT);parser.add_argument('--dwell',type=dwell_argument)
    parser.add_argument('--negatives-declared',action='store_true');parser.add_argument('--unreadable-declaration',type=Path)
    parser.add_argument('--check',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--layout',choices=markers.BOUND_LAYOUTS,help='Nowe przygotowanie F2/F3; nigdy nie zmienia istniejącego planu w miejscu.')
    parser.add_argument('--prepared',type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--reference',type=Path,help='Jawnie wybrana wersja PhotoReference tylko dla nowej analizy istniejącego runu.')
    parser.add_argument('--screen',nargs=2,type=int,default=(1920,1080));parser.add_argument('--configuration',type=Path)
    args=parser.parse_args(argv)
    explicit_dwell=args.dwell is not None
    if args.dwell is None:
        args.dwell=reference.load(args.prepared,strict_reference=False)[0]['dwell_seconds'] if args.prepared and (args.check or args.freeze) else preferred_dwell()
    declaration=photo.read_json(args.unreadable_declaration) if args.unreadable_declaration else {}
    args.unreadable_ids=declaration.get('plate_ids',[]);args.unreadable_xml_sha=declaration.get('xml_sha256')
    if args.archive:
        from ..ranking.final180_mobile import analyze_import
        if not args.run:parser.error('--archive wymaga --run')
        print(analyze_import(args.archive,args.run,output=args.output,reference_path=args.reference));return
    if args.check or args.freeze:
        if args.prepared:
            manifest,_,_=reference.load(args.prepared,strict_reference=False)
            if args.freeze:
                path=reference.relayout(args.prepared,args.screen,root=args.output,dwell_seconds=args.dwell,reference_aware=True,layout_version=args.layout) if args.layout else reference.replan(args.prepared,args.dwell,root=args.output)
                remember_selection(args.dwell,path);print(path)
            else:
                result=dict(summary=manifest['input_audit'],errors=manifest['input_errors'],status='PASS' if not manifest['input_errors'] else 'NO_GO',
                    prepared_path=str(args.prepared.resolve()),manifest_sha256=photo.digest(args.prepared/'manifest.json'),
                    frozen_dwell_seconds=manifest['dwell_seconds'],requires_new_plan=args.dwell!=manifest['dwell_seconds'],
                    plan=reference.plan_times(len(manifest['exposures']),args.dwell,countdown_seconds=manifest['countdown_seconds'],transition_seconds=manifest['transition_seconds']))
                result['reference_preflight']=reference.reference_preflight(args.prepared);result['display_ready']=reference.display_ready(manifest)
                print(json.dumps(result,ensure_ascii=False,indent=2))
            return
        if not args.positives:parser.error('--check/--freeze wymaga --positives (GT XML jest opcjonalne)')
        if not args.xml:
            if args.freeze:print(reference.freeze_photographs(args.positives,args.output,screen=args.screen,dwell_seconds=args.dwell,additional_source=args.negatives,configuration=photo.read_json(args.configuration) if args.configuration else None,layout_version=args.layout or markers.F3_LAYOUT))
            else:
                paths=reference.files(args.positives)+(reference.files(args.negatives) if args.negatives else [])
                print(json.dumps(dict(photographs=len(dict.fromkeys(paths)),reference_mode='none',metrics_status='NOT_AVAILABLE_NO_GT'),ensure_ascii=False))
            return
        if args.unreadable_ids and photo.digest(args.xml)!=args.unreadable_xml_sha:
            parser.error('Deklaracja nieczytelności dotyczy innego XML; nie rozszerzono jej na nowe dane.')
        result=reference.audit(args.xml,args.positives,args.negatives,negative_declared=args.negatives_declared,unreadable_plate_ids=args.unreadable_ids,expected_negative=None)
        if args.freeze:
            path=reference.freeze(result,args.output,screen=args.screen,dwell_seconds=args.dwell,
                configuration=photo.read_json(args.configuration) if args.configuration else {},layout_version=args.layout or markers.F3_LAYOUT,reference_aware=True)
            remember_selection(args.dwell,path);print(path)
        else:print(json.dumps(dict(summary=result['summary'],errors=result['errors'],status=result['status'],
            plan=reference.plan_times(len(result['images']),args.dwell)),ensure_ascii=False,indent=2))
        return
    app=make_app(legacy_module(args.legacy_viewer),args,parent=parent)
    if args.prepared:
        try:app.loaded(args.prepared)
        except Exception as exc:
            if isinstance(exc,PreparedSelectionError):logging.getLogger(__name__).warning('FINAL saved selection rejected: %s',exc)
            else:logging.getLogger(__name__).exception('FINAL saved preparation could not be loaded')
            app.selection_failed(args.prepared,exc)
    if explicit_dwell:app.dwell.set(str(args.dwell))
    if parent is None:app.window.mainloop()
    return app


if __name__=='__main__':main()
