"""NMCE/CER presentation and read-only inspection of the existing paired rows.

Coordinates and eligibility come exclusively from paired_geometry_text. Pixel
distance is used only for picking; neither exports nor markers are jittered.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import tkinter as tk
from tkinter import ttk

from .mobile_compare_review_gate import paired_geometry_text
from .mobile_model_identity import recorded_model_identity

MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '<', '>')
COLORS = ('#1764a0', '#b45b13', '#278065', '#8c4e96', '#ad3b51')


def inspection_records(group):
    """Retain the exact qualified row, including literal (not normalized) GT/MZ."""
    result = []
    for session in group:
        if not session.canvas_ready:
            continue
        for raw in (session.canvas or {}).get('rows', ()):
            for pair in paired_geometry_text({'rows': [raw]}):
                result.append(dict(pair, session_id=session.session_id,
                    source_sha256=session.source_sha256, review_revision=session.review_revision,
                    repeat=session.repeat_label, scenario=session.scenario,
                    attempt_id=raw.get('attempt_id', ''), mt_invocation_id=raw.get('mt_invocation_id', ''),
                    mz_prediction=raw.get('mz_prediction', ''), gt_text=raw.get('gt_text', ''),
                    mz_status=raw.get('mz_status', ''), provenance='ASSISTED_CANVAS_SOURCE_ROI',
                    review_export_manifest_sha256=session.manifest_sha256))
    return result


def _identity_key(session):
    identity = session.model_identity or recorded_model_identity({})
    return json.dumps(dict(package=identity.get('package_id'), models=identity.get('models'),
        mt_checkpoint=session.mt_checkpoint, mz_checkpoint=session.mz_checkpoint,
        mt_model_sha=session.mt_model_sha, mz_model_sha=session.mz_model_sha), sort_keys=True)


def _wrap(text, renderer, font, width):
    """Wrap using actual font metrics, also on a DPI-scaled Tk canvas."""
    lines = []
    for paragraph in text.splitlines():
        line = ''
        for word in paragraph.split():
            candidate = (line+' '+word).strip()
            if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > width:
                lines.append(line); line = word
            else:
                line = candidate
        lines.append(line)
    return '\n'.join(lines)


def fit_geometry_scatter(fig):
    """Size margins from rendered header/footer; leave the data axes central."""
    model = fig._alpr_geometry_scatter
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    unit = fig.dpi / 100
    title, identity, footer = (model[k] for k in ('title', 'identity', 'footer'))
    cursor = height - 7*unit
    for artist, full in ((title, model['title_text']), (identity, model['identity_text'])):
        artist.set_text(_wrap(full, renderer, artist.get_fontproperties(), width-30*unit))
        artist.set_position((.5, cursor/height))
        cursor -= artist.get_window_extent(renderer).height + 5*unit
    legend = model['legend']
    if legend is not None:
        legend.set_bbox_to_anchor((.5, cursor/height), transform=fig.transFigure)
        cursor -= legend.get_window_extent(renderer).height + 6*unit
    footer.set_text(_wrap(model['footer_text'], renderer, footer.get_fontproperties(), width-30*unit))
    footer.set_position((.5, 7*unit/height))
    bottom = footer.get_window_extent(renderer).y1 + 47*unit
    ax = model['axes']
    tick_width = max((tick.get_window_extent(renderer).width for tick in ax.get_yticklabels()), default=25*unit)
    left = max(65*unit, tick_width+37*unit)
    fig.subplots_adjust(left=left/width, right=1-17*unit/width,
                        bottom=bottom/height, top=max(bottom+50*unit, cursor)/height)


def nmce_cer_figure(group, *, scenario):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    rows = inspection_records(group)
    counts = Counter(r['session_id'] for r in rows)
    keys = list(dict.fromkeys(_identity_key(s) for s in group))
    identities = []
    for key in keys:
        session = next(s for s in group if _identity_key(s) == key)
        identity = session.model_identity or recorded_model_identity({})
        caption = identity['caption']+' · '+identity['variant_caption']
        if len(keys) > 1:
            caption = f'M{keys.index(key)+1}: '+caption
            # Equal display names must not hide differing recorded checkpoints.
            caption += f" · MT {session.mt_checkpoint[:8] or '?'} / MZ {session.mz_checkpoint[:8] or '?'}"
        identities.append(caption)
    duplicate_repeats = Counter(s.repeat_label for s in group)
    series = []
    for i, session in enumerate(group):
        data = [r for r in rows if r['session_id'] == session.session_id]
        if not data:
            continue
        label = session.repeat_label
        if duplicate_repeats[label] > 1:
            label += ' · '+session.session_id[-7:]
        if len(keys) > 1:
            label += f' · M{keys.index(_identity_key(session))+1}'
        label += f' · n={counts[session.session_id]}'
        color = COLORS[i % len(COLORS)]
        artist = ax.scatter([r['nmce_pct'] for r in data], [r['cer_pct'] for r in data],
            label=label, marker=MARKERS[i % len(MARKERS)], s=53,
            facecolors=[to_rgba(color, .20)], edgecolors=[to_rgba(color, .95)], linewidths=1.25, clip_on=False)
        series.append(artist)
    ax.set_xlabel('NMCE narożników MT [%]', fontsize=10)
    ax.set_ylabel('CER tej samej tablicy [%]', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-1)
    ax.grid(alpha=.20, linewidth=.7)
    ax.set_axisbelow(True)
    if not rows:
        ax.text(.5,.5,'Brak ocenialnych par NMCE + CER',transform=ax.transAxes,ha='center',va='center')
    title_text = 'NMCE ramki MT a CER odczytu MZ · '+scenario
    title = fig.suptitle(title_text, fontsize=12, fontweight='semibold', va='top')
    identity_text = '\n'.join(identities)
    identity = fig.text(.5,.94,identity_text,ha='center',va='top',fontsize=9)
    legend = fig.legend(handles=series,loc='upper center',ncol=min(3,len(series)),
        fontsize=9,frameon=False,borderaxespad=0,columnspacing=1.8,handletextpad=.5) if series else None
    footer_text = (f'N={len(rows)} par · Canvas ASSISTED; powtórzenia scen nie są niezależne.\n'
                   'Współwystępowanie, bez wniosków przyczynowych. Wartości bez przesunięć.')
    footer = fig.text(.5,.01,footer_text,ha='center',va='bottom',fontsize=8.5,color='#333333')
    fig._alpr_geometry_scatter = dict(records=rows, axes=ax, series=series, title=title,
        identity=identity, footer=footer, legend=legend, title_text=title_text,
        identity_text=identity_text, footer_text=footer_text)
    # Agg is available before Tk attachment and during scientific exports.
    fig.canvas.draw()
    fit_geometry_scatter(fig)
    return fig


def hit_indices(fig, x, y, *, radius_points=8):
    """All records within the pointer's screen radius, sorted by true distance."""
    model = fig._alpr_geometry_scatter
    radius = radius_points * fig.dpi / 72
    found = []
    for i, row in enumerate(model['records']):
        px, py = model['axes'].transData.transform((row['nmce_pct'], row['cer_pct']))
        distance = math.hypot(px-x, py-y)
        if distance <= radius:
            found.append((distance, i))
    return [i for _, i in sorted(found)]


class PointInspector(ttk.Frame):
    """Lists every picked observation; selecting a row only changes local UI."""
    def __init__(self, parent, canvas):
        super().__init__(parent)
        self.canvas = canvas
        self.fig = canvas.figure
        self.model = self.fig._alpr_geometry_scatter
        self.records = self.model['records']
        self.visible_indices = []
        self.current_record = None
        self.columnconfigure(0,weight=1)
        self.rowconfigure(2,weight=1)
        ttk.Label(self,text='Inspekcja punktów',font=('Segoe UI',10,'bold')).grid(row=0,column=0,sticky='w',pady=(8,3))
        controls=ttk.Frame(self);controls.grid(row=1,column=0,sticky='ew')
        ttk.Button(controls,text=f'Wszystkie: {len(self.records)}',command=self.show_all).pack(side='left')
        count=sum(r['cer_pct']>0 for r in self.records)
        ttk.Button(controls,text=f'CER > 0: {count}',command=self.show_errors).pack(side='left',padx=(4,0))
        table=ttk.Frame(self);table.grid(row=2,column=0,sticky='nsew',pady=3)
        table.columnconfigure(0,weight=1);table.rowconfigure(1,weight=1)
        self.caption=tk.StringVar()
        ttk.Label(table,textvariable=self.caption,wraplength=270).grid(row=0,column=0,columnspan=2,sticky='w')
        self.tree=ttk.Treeview(table,columns=('repeat','nmce','cer','image'),show='headings',height=4,selectmode='browse')
        for name,label,width in [('repeat','Seria',45),('nmce','NMCE %',65),('cer','CER %',62),('image','Attempt',72)]:
            self.tree.heading(name,text=label);self.tree.column(name,width=width,minwidth=30,stretch=True)
        self.tree.grid(row=1,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview);scroll.grid(row=1,column=1,sticky='ns')
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind('<<TreeviewSelect>>',self._select)
        detail_host=ttk.Frame(self);detail_host.grid(row=3,column=0,sticky='ew')
        detail_host.columnconfigure(0,weight=1)
        self.detail=tk.Text(detail_host,height=7,width=32,wrap='word',font=('Segoe UI',9),relief='flat',state='disabled')
        self.detail.grid(row=0,column=0,sticky='ew')
        detail_scroll=ttk.Scrollbar(detail_host,orient='vertical',command=self.detail.yview)
        detail_scroll.grid(row=0,column=1,sticky='ns')
        self.detail.configure(yscrollcommand=detail_scroll.set)
        self.highlight=self.model['axes'].scatter([],[],s=165,facecolors='none',edgecolors='#171717',linewidths=1.7,zorder=20)
        self.highlight.set_gid('inspection-highlight')
        self._cid=canvas.mpl_connect('button_press_event',self._click)
        self.show_all()

    def _text(self, text):
        self.detail.configure(state='normal');self.detail.delete('1.0','end')
        self.detail.insert('1.0',text);self.detail.configure(state='disabled')

    def _show(self, indices, caption, select=False):
        self.visible_indices=list(indices)
        self.tree.delete(*self.tree.get_children())
        for i in self.visible_indices:
            r=self.records[i]
            short=r['attempt_id'].rsplit('-',1)[-1] or r['object_id'][-10:]
            if short.startswith('a') and short[1:].isdecimal():short=f'a{int(short[1:]):02d}'
            self.tree.insert('', 'end', iid=str(i), values=(r['repeat'],f"{r['nmce_pct']:.4f}",f"{r['cer_pct']:.4f}",short))
        self.caption.set(f'{caption} · n={len(self.visible_indices)}')
        self.current_record=None
        self.highlight.set_offsets(__import__('numpy').empty((0,2)))
        self._text('Kliknij punkt wykresu, aby zobaczyć wszystkie obserwacje pod kursorem. Wybierz rekord z listy; strzałki przechodzą między rekordami. Dane bez przesunięć.')
        if select and self.visible_indices:self.select_record(self.visible_indices[0])
        self.canvas.draw_idle()

    def show_all(self):
        self._show(range(len(self.records)), 'Wszystkie pary')

    def show_errors(self):
        self._show([i for i,r in enumerate(self.records) if r['cer_pct']>0], 'Odczyty z CER > 0', True)

    def _click(self, event):
        if event.button != 1 or event.inaxes is not self.model['axes']:
            return
        self._show(hit_indices(self.fig,event.x,event.y),'Pod kliknięciem (bliskie punkty)',True)

    def _select(self, event=None):
        selected=self.tree.selection()
        if selected:self.select_record(int(selected[0]))

    def select_record(self, index):
        if index not in self.visible_indices:return
        r=self.records[index];self.current_record=r
        if self.tree.selection() != (str(index),):self.tree.selection_set(str(index))
        self.tree.see(str(index))
        self.highlight.set_offsets([[r['nmce_pct'],r['cer_pct']]])
        self._text(f"{r['repeat']} · rewizja {r['review_revision']} · ASSISTED\n"
            f"NMCE: {r['nmce_pct']!r}% · CER: {r['cer_pct']!r}%\n"
            f"IoU: {r['iou_pct']!r}%\nMZ: {r['mz_prediction']}\nGT: {r['gt_text']}\n"
            f"Sesja: {r['session_id']}\nObiekt: {r['object_id']}\nObraz: {r['image_entry']}\n"
            f"Attempt: {r['attempt_id']}\nSHA źródła: {r['source_sha256']}\n"
            f"SHA manifestu oceny: {r['review_export_manifest_sha256']}")
        self.canvas.draw_idle()

    def destroy(self):
        self.canvas.mpl_disconnect(self._cid)
        self.highlight.remove()
        super().destroy()
