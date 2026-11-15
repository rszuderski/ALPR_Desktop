# -*- coding: utf-8 -*-
"""AKT85: the chart comparison window opened from the existing Z4 imports list."""
from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from .mobile_comparison_core import figures, validate_selection
from .mobile_import_comparison import export_imported_charts
from .mobile_import_trace_charts import trace_figures
from .mobile_model_identity import chart_identity_caption
from .mobile_plot_choices import plot_choices


class ImportedMobileComparisonWindow(tk.Toplevel):
    """No reimport: records originate in verified bundles supplied by Browser."""

    def __init__(self, parent, records, rows_by_record, results_root=None, imported_sources=None, *, comparison_scope=None):
        super().__init__(parent)
        self.title("ALPR Desktop — porównaj importowane raporty")
        self.geometry("1210x760")
        self.minsize(940,610)
        self.records=list(records)
        from .mobile_comparison_scope import scope_value
        self._comparison_scope=scope_value(comparison_scope,len(self.records))
        self._strict_scope=comparison_scope is not None
        self.traces=list(rows_by_record)
        self.root_path=Path(results_root) if results_root else None
        self.imported_sources=list(imported_sources or [])
        self.figure_items=[]
        self.drawn=None
        self.point_inspector=None
        self.configure(background="#f1f5f9")
        self._build()
        self._make_charts()
        self.protocol("WM_DELETE_WINDOW",self._close)

    @property
    def comparison_scope(self):
        return self._comparison_scope

    def _build(self):
        top=ttk.Frame(self,padding=12)
        top.pack(fill="both",expand=True)
        top.columnconfigure(0,weight=0,minsize=280)
        top.columnconfigure(1,weight=1,minsize=480)
        top.rowconfigure(1,weight=1)
        heading = ttk.Label(top,text="Porównanie raportów z telefonu\n" + chart_identity_caption(self.records),
            font=("Segoe UI",10,"bold"), wraplength=1120, justify="left")
        heading.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,10))
        self.heading=heading
        self.bind("<Configure>", lambda e: heading.configure(wraplength=max(600,e.width-40))
                  if e.widget is self else None, add="+")
        left=ttk.Frame(top)
        self.left_panel=left
        left.grid(row=1,column=0,sticky="nsew",padx=(0,12))
        left.columnconfigure(0,weight=1)
        left.rowconfigure(1,weight=1)
        ttk.Label(left,text="Wybrane sesje (każda liczona raz)",font=("Segoe UI",10,"bold")).grid(row=0,column=0,sticky="w")
        self.selected_list=tk.Listbox(left,selectmode="extended",width=28,height=12,activestyle="dotbox")
        self.selected_list.grid(row=1,column=0,sticky="nsew",pady=(6,8))
        for record in self.records:
            self.selected_list.insert(tk.END,record.label.replace("\n", " · "))
        controls=ttk.Frame(left)
        horizontal = ttk.Scrollbar(left, orient="horizontal", command=self.selected_list.xview)
        horizontal.grid(row=2,column=0,sticky="ew")
        self.selected_list.configure(xscrollcommand=horizontal.set)
        controls.grid(row=3,column=0,sticky="ew")
        ttk.Button(controls,text="Usuń z porównania",command=self.remove_selected).pack(side=tk.LEFT)
        self.root_var=tk.StringVar(value=str(self.root_path) if self.root_path else "Wskaż katalog 03_wyniki do zapisania wykresów")
        ttk.Label(left,text="Katalog docelowy",font=("Segoe UI",9,"bold")).grid(row=4,column=0,sticky="w",pady=(18,3))
        ttk.Label(left,textvariable=self.root_var,wraplength=265,justify="left").grid(row=5,column=0,sticky="w")
        ttk.Button(left,text="Wskaż 03_wyniki",command=self.pick_root).grid(row=6,column=0,sticky="ew",pady=(5,12))
        ttk.Button(left,text="Zapisz PNG + SVG + CSV",command=self.save_figures).grid(row=7,column=0,sticky="ew",ipady=5)
        right=ttk.Frame(top)
        right.grid(row=1,column=1,sticky="nsew")
        right.columnconfigure(0,weight=1)
        right.rowconfigure(2,weight=1)
        ttk.Label(right,text="Kontrola porównywalności",font=("Segoe UI",10,"bold")).grid(row=0,column=0,sticky="w")
        self.notes=tk.Text(right,height=6,wrap="word",state="disabled",font=("Segoe UI",9),relief="flat")
        self.notes.grid(row=1,column=0,sticky="ew",pady=(5,8))
        self.notes_toggle=ttk.Button(right,text="Rozwiń kontrolę",command=self._toggle_notes)
        self.notes_toggle.grid(row=0,column=0,sticky="e")
        self.notes_toggle.grid_remove()
        plot=ttk.Frame(right)
        plot.grid(row=2,column=0,sticky="nsew")
        plot.columnconfigure(0,weight=1)
        plot.rowconfigure(1,weight=1)
        selector=ttk.Frame(plot)
        selector.grid(row=0,column=0,sticky="ew",pady=(0,6))
        selector.columnconfigure(1,weight=1)
        ttk.Label(selector,text="Wybierz wykres",font=("Segoe UI",9,"bold")).grid(row=0,column=0,sticky="w",padx=(0,8))
        self.plot_choice=ttk.Combobox(selector,state="readonly",height=8)
        self.plot_choice.grid(row=0,column=1,sticky="ew")
        self.plot_choice.bind("<<ComboboxSelected>>",lambda _ev:self.show_plot())
        self.chart_host=ttk.Frame(plot)
        self.chart_host.grid(row=1,column=0,sticky="nsew")
        self.status=tk.StringVar(value="Czasy pochodzą z oryginalnych śladów; brak GT nie jest 0%.")
        ttk.Label(top,textvariable=self.status,wraplength=1080).grid(row=2,column=0,columnspan=2,sticky="ew",pady=(12,0))

    def _set_notes(self,text):
        self.notes.configure(state="normal")
        self.notes.delete("1.0",tk.END)
        self.notes.insert(tk.END,text)
        self.notes.configure(state="disabled")

    def _close_charts(self):
        if self.point_inspector is not None:
            self.point_inspector.destroy()
            self.point_inspector=None
        try:
            if self.drawn is not None:
                self.drawn.get_tk_widget().destroy()
        except tk.TclError:
            pass
        self.drawn=None
        import matplotlib.pyplot as plt
        for _,fig in self.figure_items:
            plt.close(fig)
        self.figure_items=[]

    def _make_charts(self):
        selected_key=self.current_plot_key()
        self._close_charts()
        self.plot_choice.configure(values=[])
        self.plot_choice.set("")
        self._update_heading()
        if not self.records:
            self._set_notes("Wybierz przynajmniej jedną sesję.")
            return
        from .mobile_comparison_scope import validate_comparison, comparison_warnings
        notes=comparison_warnings(self.records,self.comparison_scope)
        self._set_notes("\n".join("• "+n for n in notes) if notes else "Sprawdzone wspólne warunki według dostępnych metadanych. Parowanie pojedynczych zdjęć wymaga osobnego manifestu.")
        try:
            validate_comparison(self.records,self.traces,comparison_scope=self.comparison_scope,require_sources=self._strict_scope)
        except (ValueError,OSError) as exc:
            self.status.set('Zablokowano porównanie: '+str(exc));return
        try:
            plots=list(figures(self.records,comparison_scope=self.comparison_scope) if self._strict_scope else figures(self.records))
            if len(self.records)==1:
                plots += [("oś czasu · "+title,fig) for title,fig in trace_figures(self.traces[0],session_label=self.records[0].label)]
            choices=plot_choices(plots)
            kept={id(choice.figure) for choice in choices}
            import matplotlib.pyplot as plt
            for _,fig in plots:
                if id(fig) not in kept:
                    plt.close(fig)
            self.figure_items=[(choice.key,choice.figure) for choice in choices]
        except (ImportError,ValueError,RuntimeError,TypeError) as exc:
            self.status.set("Nie udało się przygotować wykresu (sprawdź matplotlib): "+str(exc)[:180]);return
        labels=[choice.label for choice in choices]
        self.plot_choice.configure(values=labels)
        if labels:
            keys=[key for key,_ in self.figure_items]
            self.plot_choice.current(keys.index(selected_key) if selected_key in keys else 0)
            self.show_plot()
        self.status.set(f"Wybrane sesje: {len(self.records)}. Porównanie opisowe; MZ może sumować wywołania w śladzie. "
            + ('Pełna jakość poza zakresem.' if self.comparison_scope=='performance' else 'Canvas ASSISTED nie jest niezależnym GT.'))

    def current_plot_key(self):
        index=self.plot_choice.current()
        return self.figure_items[index][0] if 0<=index<len(self.figure_items) else None

    def _update_heading(self, scatter=False):
        scenarios=list(dict.fromkeys(str(s.scenario) for s in self.records))
        from .mobile_comparison_scope import SCOPES
        title=SCOPES.get(self.comparison_scope,"Wykresy sesji z telefonu")
        if scenarios:title+=" · "+" / ".join(scenarios)
        if self.records and not scatter:
            from .mobile_measurement_availability import polish_model_caption
            title+="\n"+polish_model_caption(chart_identity_caption(self.records))
        self.heading.configure(text=title)

    def show_plot(self):
        i=self.plot_choice.current()
        if not (0<=i<len(self.figure_items)):
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            if self.point_inspector is not None:
                self.point_inspector.destroy()
                self.point_inspector=None
            if self.drawn is not None:
                self.drawn.get_tk_widget().destroy()
            fig=self.figure_items[i][1]
            scatter=hasattr(fig,"_alpr_geometry_scatter")
            self._update_heading(scatter)
            self.notes.configure(height=2 if scatter else 6)
            self.notes_toggle.configure(text="Rozwiń kontrolę")
            if scatter:self.notes_toggle.grid()
            else:self.notes_toggle.grid_remove()
            self.left_panel.rowconfigure(1,weight=0 if scatter else 1)
            self.left_panel.rowconfigure(8,weight=1 if scatter else 0)
            self.selected_list.configure(height=3 if scatter else 12)
            selection=self.selected_list.curselection()
            self.selected_list.delete(0,tk.END)
            counts={s.session_id:0 for s in self.records}
            if scatter:
                for row in fig._alpr_geometry_scatter['records']:counts[row['session_id']]+=1
            for s in self.records:
                label=(f"{s.repeat_label} · n={counts[s.session_id]} · {s.session_id}" if scatter else
                    f'{s.repeat_label} · {s.precision.upper()} · {s.session_id[-8:]}' if self.comparison_scope=='performance' else s.label.replace('\n',' · '))
                self.selected_list.insert(tk.END,label)
            for selected in selection:self.selected_list.selection_set(selected)
            if not hasattr(fig,"_alpr_preview_dpi"):
                fig._alpr_preview_dpi=fig.dpi
            # Reattaching the same figure must not multiply Windows DPI again.
            fig.set_dpi(fig._alpr_preview_dpi)
            self.drawn=FigureCanvasTkAgg(fig,master=self.chart_host)
            if scatter:
                from .mobile_geometry_scatter import PointInspector
                self.point_inspector=PointInspector(self.left_panel,self.drawn)
                self.point_inspector.grid(row=8,column=0,sticky="nsew")
            self.drawn.get_tk_widget().pack(fill="both",expand=True)
            self.drawn.get_tk_widget().bind("<Configure>", self._fit_preview_text, add="+")
            # TkAgg updates DPI on Map; a constrained host need not emit another
            # Configure event, so explicitly reconcile its backing image then.
            self.drawn.get_tk_widget().bind("<Map>",
                lambda _event: self.after_idle(self._fit_preview_text), add="+")
            self._fit_preview_text()
            self.drawn.draw()
        except (RuntimeError,ValueError,tk.TclError) as exc:
            self.status.set("Błąd podglądu: "+str(exc)[:160])

    def _toggle_notes(self):
        expanded=int(self.notes.cget('height'))>2
        self.notes.configure(height=2 if expanded else 6)
        self.notes_toggle.configure(text="Rozwiń kontrolę" if expanded else "Zwiń kontrolę")

    def _fit_preview_text(self, event=None):
        """Wrap preview annotations after Tk resizes the existing figure."""
        if self.drawn is None:
            return
        import textwrap
        from types import SimpleNamespace
        fig = self.drawn.figure
        widget = self.drawn.get_tk_widget()
        w, h = widget.winfo_width(), widget.winfo_height()
        if min(w,h)>1 and (abs(fig.bbox.width-w)>1 or abs(fig.bbox.height-h)>1):
            self.drawn.resize(SimpleNamespace(width=w,height=h))
        if hasattr(fig,"_alpr_geometry_scatter"):
            from .mobile_geometry_scatter import fit_geometry_scatter
            fit_geometry_scatter(fig)
            self.drawn.draw_idle()
            return
        width = fig.bbox.width
        fig.subplots_adjust(top=.72)
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)
            for label in ax.texts:
                if not hasattr(label,"_alpr_full_label"):
                    label._alpr_full_label=label.get_text()
                label.set_fontsize(7)
                label.set_text(textwrap.fill(label._alpr_full_label,
                    width=max(22,int(ax.bbox.width/(7*fig.dpi/72*.55)))))
            if not hasattr(ax, "_alpr_full_title"):
                ax._alpr_full_title = ax.get_title()
            ax.set_title("\n".join(textwrap.wrap(ax._alpr_full_title,
                         width=max(25, int(width*.55/7)))), fontsize=9)
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    if not hasattr(text, "_alpr_full_label"):
                        text._alpr_full_label = text.get_text()
                    text.set_text("\n".join(textwrap.fill(line, width=max(22, int(width*.36/6)))
                                            for line in text._alpr_full_label.splitlines()))
        if fig._suptitle is not None:
            title = fig._suptitle
            if not hasattr(title, "_alpr_full_label"):
                title._alpr_full_label = title.get_text()
            title.set_text("\n".join(textwrap.fill(line, width=max(35,int(width/7)))
                                     for line in title._alpr_full_label.splitlines()))
        for note in fig.texts:
            if note is fig._suptitle:
                continue
            if not hasattr(note, "_alpr_full_label"):
                note._alpr_full_label = note.get_text()
            note.set_fontsize(6.5)
            chars = max(28, int(width*.88 / max(5, note.get_fontsize()*fig.dpi/72*.55)))
            note.set_text(textwrap.fill(" ".join(note._alpr_full_label.splitlines()), width=chars))
        # Reserve the actual footer height plus tick/x-label clearance. Small
        # Tk previews have a different aspect ratio from the exported figures.
        renderer = self.drawn.get_renderer()
        footer_top = max((note.get_window_extent(renderer).y1 for note in fig.texts
                          if note is not fig._suptitle), default=0)
        fig.subplots_adjust(bottom=max(.24, min(.46, (footer_top + 56)/fig.bbox.height)))
        self.drawn.draw_idle()

    def remove_selected(self):
        gone=set(self.selected_list.curselection())
        if not gone:
            return
        self.records=[r for i,r in enumerate(self.records) if i not in gone]
        self.traces=[t for i,t in enumerate(self.traces) if i not in gone]
        if self.imported_sources:
            self.imported_sources=[v for i,v in enumerate(self.imported_sources) if i not in gone]
        self.selected_list.delete(0,tk.END)
        for rec in self.records:
            self.selected_list.insert(tk.END,rec.label.replace("\n", " · "))
        self._make_charts()

    def pick_root(self):
        initial=str(self.root_path) if self.root_path and self.root_path.is_dir() else str(Path.home())
        value=filedialog.askdirectory(parent=self,title="Wskaż istniejący katalog 03_wyniki",initialdir=initial)
        if value:
            self.root_path=Path(value)
            self.root_var.set(str(self.root_path))
            if self.imported_sources:
                from .mobile_import_comparison import record_from_import
                revised=[]; traces=[]
                try:
                    for report,bundle in self.imported_sources:
                        if self._strict_scope and self.comparison_scope!='single':
                            from .mobile_comparison_scope import assess_import
                            access=assess_import(report,bundle.path,self.root_path)
                            if not access.allowed(self.comparison_scope):raise ValueError(access.reason(self.comparison_scope))
                            record,rows=access.record,list(access.traces)
                        else:record,rows,_=record_from_import(report,bundle,self.root_path)
                        revised.append(record);traces.append(rows)
                except (ValueError,OSError,RuntimeError,KeyError) as exc:
                    messagebox.showerror("Nie uzgodniono raportów","Nie można wczytać ocen Canvas: "+str(exc),parent=self)
                    return
                self.records,self.traces=revised,traces
                self.selected_list.delete(0,tk.END)
                for rec in self.records:self.selected_list.insert(tk.END,rec.label.replace("\n", " · "))
                self._make_charts()

    def save_figures(self):
        if not self.records:
            messagebox.showwarning("Brak sesji","Dodaj raporty do porównania.",parent=self);return
        if self.root_path is None or not self.root_path.is_dir():
            self.pick_root()
        if self.root_path is None:
            return
        try:
            destination=export_imported_charts(self.records,self.traces,self.root_path,
                comparison_scope=self.comparison_scope if self._strict_scope else None)
        except (ImportError,OSError,ValueError,RuntimeError,PermissionError) as exc:
            messagebox.showerror("Nie zapisano wykresów",str(exc),parent=self);return
        self.status.set("Zapisano wykresy: "+str(destination))
        messagebox.showinfo("Eksport zakończony",f"Wykresy PNG i SVG, CSV oraz manifest SHA zapisano w:\n{destination}",parent=self)

    def _close(self):
        self._close_charts()
        self.destroy()
