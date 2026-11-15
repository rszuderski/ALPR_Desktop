"""Photograph-first layout for the existing presentation controller.

This preview is read-only. Annotation and presentation rendering remain in
their existing Canvas and F3 owners.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import weakref
from PIL import Image, ImageTk

from ..gt_pack import _atomic_write_json

BG = '#f1f4f3'
CARD = '#ffffff'
TEXT = '#182b25'
MUTED = '#586b63'
ACCENT = '#22684f'


def wrapping(label):
    label.bind('<Configure>', lambda e: label.configure(wraplength=max(80,e.width-4)), add='+')
    return label


class MaterialPreview:
    def __init__(self, app, parent):
        self.app=app;self.material=None;self.order=[];self.index=0
        self.generation=0;self.future=None;self.image=None;self.photo=None;self.resize_job=None
        self.message='Wybierz katalog fotografii, aby zobaczyć podgląd.'
        self.canvas=tk.Canvas(parent,bg='#e7ede9',highlightthickness=0,takefocus=True)
        self.canvas.grid(row=2,column=0,columnspan=2,sticky='nsew',pady=(12,8))
        self.canvas.bind('<Configure>',self.resize)
        self.canvas.bind('<Button-1>',lambda e:self.canvas.focus_set())
        self.canvas.bind('<Left>',lambda e:self.move(-1))
        self.canvas.bind('<Right>',lambda e:self.move(1))
        nav=ttk.Frame(parent,style='PhotoShow.Card.TFrame')
        nav.grid(row=3,column=0,columnspan=2,sticky='ew');nav.columnconfigure(1,weight=1)
        self.previous=ttk.Button(nav,text='←',width=4,command=lambda:self.move(-1),state='disabled')
        self.previous.grid(row=0,column=0,padx=(0,10))
        self.caption=tk.StringVar(value='Brak wybranych zdjęć')
        self.caption_label=ttk.Label(nav,textvariable=self.caption,anchor='center',width=1,style='PhotoShow.Muted.TLabel')
        self.caption_label.grid(row=0,column=1,sticky='ew')
        self.next=ttk.Button(nav,text='→',width=4,command=lambda:self.move(1),state='disabled')
        self.next.grid(row=0,column=2,padx=(10,0))

    def set_material(self, material):
        identity=getattr(material,'material_id',str(getattr(material,'directory','')))
        old_identity=getattr(self.material,'material_id',str(getattr(self.material,'directory','')))
        if self.material is not None and identity==old_identity and self.material.root==material.root:
            self.material=material
            return
        self.material=material;self.order=list(material.order) if material is not None else []
        index=0
        if material is not None and getattr(material,'material_id',None):
            try:
                from ..ranking.photo_reference import read_json
                saved=read_json(material.directory/'view_state.json')
                if saved.get('material_id')==material.material_id and saved.get('image_id') in self.order:
                    index=self.order.index(saved['image_id'])
            except (OSError,ValueError,KeyError):pass
        self.select(index,persist=False)

    def select(self,index,*,persist=True):
        self.generation+=1;generation=self.generation
        if self.future is not None:self.future.cancel()
        if not self.order:
            self.image=None;self.caption.set('Brak wybranych zdjęć');self.paint();self.controls();return
        self.index=max(0,min(len(self.order)-1,index));key=self.order[self.index]
        material=self.material
        row=material.images[key]
        name=row.get('source_relative_path',row['relative_path'])
        self.caption.set(f'Podgląd {self.index+1} / {len(self.order)}  ·  {name}')
        self.image=None;self.message='Wczytuję fotografię…';self.paint();self.controls()
        if persist and getattr(material,'material_id',None):
            _atomic_write_json(material.directory/'view_state.json',dict(schema='alpr.photo_material_view.v1',
                material_id=material.material_id,image_id=key))
        target=weakref.ref(self)
        dispatch=self.app.queue
        def ready(result):
            preview=target()
            if preview is None or preview.app is None or preview.app.closed or generation!=preview.generation:return
            image,error=result
            preview.image=image;preview.message=error or '';preview.paint()
            if error:preview.app.status.set('Nie można odczytać fotografii: '+error)
        def work():
            try:result=(material.image(key),'')
            except (OSError,ValueError,KeyError) as exc:result=(None,str(exc))
            dispatch.put((ready,result,None))
        self.future=self.app.workers.submit(work)

    def move(self,offset):
        if not self.app.material_actions_available() or not self.order:return
        self.select(self.index+offset)

    def controls(self):
        enabled=self.app.material_actions_available()
        self.previous.configure(state='normal' if enabled and self.index>0 else 'disabled')
        self.next.configure(state='normal' if enabled and self.index+1<len(self.order) else 'disabled')

    def resize(self,event=None):
        if self.app is None:return
        if self.resize_job is not None:
            try:self.canvas.after_cancel(self.resize_job)
            except tk.TclError:pass
        self.resize_job=self.canvas.after(50,self.paint)

    def paint(self):
        self.resize_job=None
        if self.app is None or self.app.closed or not self.canvas.winfo_exists():return
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        if w<4 or h<4:return
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.create_text(w/2,h/2,text=self.message,font=('Segoe UI',11),fill=MUTED,width=max(80,w-60))
            return
        image=self.image.copy();image.thumbnail((max(1,w-24),max(1,h-24)),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(image,master=self.canvas)
        self.canvas.create_image(w/2,h/2,image=self.photo,anchor='center')

    def dispose(self):
        self.generation+=1
        if self.future is not None:self.future.cancel()
        if self.resize_job is not None:
            try:self.canvas.after_cancel(self.resize_job)
            except tk.TclError:pass
        self.photo=None;self.image=None
        self.app=None


def build(app):
    style=ttk.Style(app.window)
    style.configure('PhotoShow.TFrame',background=BG)
    style.configure('PhotoShow.Card.TFrame',background=CARD)
    style.configure('PhotoShow.TLabel',background=CARD,foreground=TEXT,font=('Segoe UI',10))
    style.configure('PhotoShow.Muted.TLabel',background=CARD,foreground=MUTED,font=('Segoe UI',9))
    style.configure('PhotoShow.Title.TLabel',background=BG,foreground=TEXT,font=('Segoe UI',18,'bold'))
    style.configure('PhotoShow.Section.TLabel',background=CARD,foreground=TEXT,font=('Segoe UI',11,'bold'))
    style.configure('PhotoShow.TCheckbutton',background=CARD,foreground=MUTED,font=('Segoe UI',9))
    app.window.configure(background=BG)
    shell=ttk.Frame(app.window,padding=20,style='PhotoShow.TFrame');shell.pack(fill='both',expand=True)
    shell.columnconfigure(0,weight=1);shell.rowconfigure(1,weight=1)
    heading=ttk.Frame(shell,style='PhotoShow.TFrame');heading.grid(row=0,column=0,sticky='ew',pady=(0,14))
    heading.columnconfigure(0,weight=1)
    ttk.Label(heading,text='Pokaz fotografii',style='PhotoShow.Title.TLabel').grid(row=0,column=0,sticky='w')
    app.choose_set_btn=ttk.Button(heading,text='Zapisane pokazy…',command=app.open_prepared)
    app.choose_set_btn.grid(row=0,column=1,padx=(16,0))
    app.advanced_btn=ttk.Button(heading,text='Zaawansowane',command=app.toggle_advanced)
    app.advanced_btn.grid(row=0,column=2,padx=(8,0))
    body=ttk.Frame(shell,style='PhotoShow.TFrame');body.grid(row=1,column=0,sticky='nsew')
    app.material_body=body
    body.columnconfigure(0,weight=1,minsize=360);body.columnconfigure(1,minsize=310);body.rowconfigure(0,weight=1)
    left=ttk.Frame(body,padding=16,style='PhotoShow.Card.TFrame');left.grid(row=0,column=0,sticky='nsew',padx=(0,16))
    left.columnconfigure(0,weight=1);left.rowconfigure(2,weight=1)
    app.preview_title=tk.StringVar(value='Twoje fotografie')
    ttk.Label(left,textvariable=app.preview_title,style='PhotoShow.Section.TLabel',width=1).grid(row=0,column=0,sticky='ew')
    app.choose_material_btn=ttk.Button(left,text='Wybierz katalog…',command=app.choose_photographs)
    app.choose_material_btn.grid(row=0,column=1,padx=(10,0),sticky='e')
    wrapping(ttk.Label(left,textvariable=app.selected_folder,style='PhotoShow.Muted.TLabel',width=1)).grid(row=1,column=0,columnspan=2,sticky='ew',pady=(6,0))
    app.material_preview=MaterialPreview(app,left)
    app.recursive_checkbox=ttk.Checkbutton(left,text='Uwzględniaj podkatalogi przy wyborze materiału',
        variable=app.recursive_photographs,style='PhotoShow.TCheckbutton')
    app.recursive_checkbox.grid(row=4,column=0,columnspan=2,sticky='w',pady=(12,0))
    sidebar=ttk.Frame(body,style='PhotoShow.TFrame');sidebar.grid(row=0,column=1,sticky='nsew');sidebar.columnconfigure(0,weight=1)
    settings=ttk.Frame(sidebar,padding=16,style='PhotoShow.Card.TFrame');settings.grid(row=0,column=0,sticky='ew',pady=(0,12));settings.columnconfigure(0,weight=1)
    ttk.Label(settings,text='Ustawienia pokazu',style='PhotoShow.Section.TLabel').grid(row=0,column=0,sticky='w',pady=(0,12))
    ttk.Label(settings,text='Ekran wyświetlania',style='PhotoShow.TLabel').grid(row=1,column=0,sticky='w')
    app.monitor_combo=ttk.Combobox(settings,textvariable=app.screen_choice,values=app.monitor_names,state='readonly',width=30,postcommand=app.refresh_monitors)
    app.monitor_combo.grid(row=2,column=0,sticky='ew',pady=(5,14))
    timing=ttk.Frame(settings,style='PhotoShow.Card.TFrame');timing.grid(row=3,column=0,sticky='ew');timing.columnconfigure(0,weight=1)
    ttk.Label(timing,text='Czas zdjęcia',style='PhotoShow.TLabel').grid(row=0,column=0,sticky='w')
    app.dwell_entry=ttk.Entry(timing,textvariable=app.dwell,width=7,justify='right')
    app.dwell_entry.grid(row=0,column=1,padx=(12,6))
    ttk.Label(timing,text='s',style='PhotoShow.TLabel').grid(row=0,column=2)
    app.friendly_plan=tk.StringVar(value='Wybierz fotografie, aby obliczyć czas.')
    wrapping(ttk.Label(settings,textvariable=app.friendly_plan,style='PhotoShow.Muted.TLabel',width=1)).grid(row=4,column=0,sticky='ew',pady=(10,12))
    app.calibration_btn=ttk.Button(settings,text='Sprawdź kadr na ekranie',command=app.preview_calibration,state='disabled')
    app.calibration_btn.grid(row=5,column=0,sticky='ew')
    refs=ttk.Frame(sidebar,padding=16,style='PhotoShow.Card.TFrame');refs.grid(row=1,column=0,sticky='ew',pady=(0,12));refs.columnconfigure(0,weight=1)
    ttk.Label(refs,text='Dane referencyjne',style='PhotoShow.Section.TLabel').grid(row=0,column=0,sticky='w')
    app.reference_caption=tk.StringVar(value='Opcjonalne — pokaz działa także bez oznaczeń.')
    wrapping(ttk.Label(refs,textvariable=app.reference_caption,style='PhotoShow.Muted.TLabel',width=1)).grid(row=1,column=0,sticky='ew',pady=(6,12))
    app.edit_reference_btn=ttk.Button(refs,text='Utwórz oznaczenia',command=app.edit_material_reference,state='disabled')
    app.edit_reference_btn.grid(row=2,column=0,sticky='ew')
    app.reference_menu_btn=ttk.Menubutton(refs,text='Wczytaj / eksportuj…',state='disabled');app.reference_menu_btn.grid(row=3,column=0,sticky='ew',pady=(6,0))
    menu=tk.Menu(app.reference_menu_btn,tearoff=False);app.reference_menu_btn.configure(menu=menu)
    for title,command in (('Wczytaj istniejące GT',app.import_material_reference),('Pokaż powiązanie i zgodność',app.show_material_binding),('Eksportuj GT',app.export_material_reference)):
        menu.add_command(label=title,command=command)
    app.advanced=ttk.Frame(shell,padding=16,style='PhotoShow.Card.TFrame');app.advanced.grid(row=1,column=0,sticky='nsew');app.advanced.columnconfigure(1,weight=1);app.advanced.rowconfigure(6,weight=1)
    for row,(key,title) in enumerate((('xml','GT XML (opcjonalnie)'),('positives','Fotografie'),('negatives','Zdjęcia bez tablic (opcjonalnie)'),('output','Zapis nowych pokazów i raportów'))):
        ttk.Label(app.advanced,text=title,style='PhotoShow.TLabel').grid(row=row,column=0,sticky='w',padx=4,pady=3)
        ttk.Entry(app.advanced,textvariable=app.paths[key]).grid(row=row,column=1,sticky='ew',padx=8)
        ttk.Button(app.advanced,text='Wybierz…',command=lambda k=key:app.browse(k)).grid(row=row,column=2,padx=4)
    checks=ttk.Frame(app.advanced,style='PhotoShow.Card.TFrame');checks.grid(row=4,column=0,columnspan=3,sticky='ew',pady=6)
    ttk.Checkbutton(checks,text='Potwierdzam brak tablic w katalogu negatywów',variable=app.negative_confirmed,style='PhotoShow.TCheckbutton').pack(side='left')
    ttk.Checkbutton(checks,text='Nowy plan bez GT',variable=app.ignore_reference,style='PhotoShow.TCheckbutton').pack(side='right')
    extra=ttk.Frame(app.advanced,style='PhotoShow.Card.TFrame');extra.grid(row=5,column=0,columnspan=3,sticky='ew',pady=4)
    for text,command in (('Sprawdź materiał',app.check),('Przygotuj nowy plan',app.prepare),('Wznów zapisany przebieg…',app.resume),('Oś czasu / CSV…',app.show_timeline)):
        ttk.Button(extra,text=text,command=command).pack(side='left',padx=(0,6))
    app.details=tk.Text(app.advanced,wrap='word',height=3,font=('Segoe UI',9),background=CARD,foreground=MUTED,relief='flat')
    app.details.grid(row=6,column=0,columnspan=3,sticky='nsew',pady=6)
    wrapping(ttk.Label(app.advanced,textvariable=app.plan,style='PhotoShow.Muted.TLabel',width=1)).grid(row=7,column=0,columnspan=3,sticky='ew')
    legacy_actions=ttk.Frame(app.advanced,style='PhotoShow.Card.TFrame');legacy_actions.grid(row=8,column=0,columnspan=3,sticky='ew',pady=(6,0))
    app.import_btn=ttk.Button(legacy_actions,text='Wczytaj sesję telefonu…',command=app.import_android);app.import_btn.pack(side='left')
    app.manual_run_btn=ttk.Button(legacy_actions,text='Wskaż zapis pokazu…',command=app.choose_manual_run);app._unmatched_archive=None
    app.advanced.grid_remove()
    footer=ttk.Frame(shell,padding=(0,14,0,0),style='PhotoShow.TFrame');footer.grid(row=3,column=0,sticky='ew');footer.columnconfigure(0,weight=1)
    status=tk.Label(footer,textvariable=app.status,bg=BG,fg=MUTED,font=('Segoe UI',10),justify='left',anchor='w',width=1)
    wrapping(status).grid(row=0,column=0,sticky='ew',padx=(0,20))
    app.clock_label=tk.Label(footer,textvariable=app.clock,bg=BG,fg=TEXT,font=('Segoe UI',10,'bold'),anchor='w',width=1)
    wrapping(app.clock_label).grid(row=1,column=0,sticky='ew',padx=(0,20))
    app.stop_btn=ttk.Button(footer,text='Zatrzymaj',command=app.abort,state='disabled');app.stop_btn.grid(row=0,column=1,rowspan=2,padx=(0,10));app.stop_btn.grid_remove()
    app.start_btn=tk.Button(footer,text='Rozpocznij pokaz',command=app.begin_countdown,state='disabled',
        bg=ACCENT,fg='white',activebackground='#194e3c',activeforeground='white',disabledforeground='#b7c8bf',
        font=('Segoe UI',11,'bold'),relief='flat',borderwidth=0,padx=22,pady=12,cursor='hand2')
    app.start_btn.grid(row=0,column=2,rowspan=2,sticky='e')
    def reference_changed(*_):
        value=app.reference_info.get().split('| GT:')[-1].strip()
        app.reference_caption.set(('Oznaczenia: '+value+'\n') if value else '' )
        app.reference_caption.set(app.reference_caption.get()+'Opcjonalne — nie blokują pokazu zdjęć.')
    app.reference_info.trace_add('write',reference_changed)
    reference_changed()
