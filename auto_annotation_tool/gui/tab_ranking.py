#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka rankingu.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from ..config import CONFIG, logger
from ..icons import IconManager
from ..ranking import ModelRanking, AnnotationComparator
from ..validators import validate_cvat_xml
from .web_slim_scrollbar import WebSlimScrollbar


class RankingTab:
    """Zakładka rankingu modeli."""
    
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.icon_manager = IconManager
        
        self.frame = ttk.Frame(parent)
        
        self.ranking = ModelRanking()
        self.comparator = AnnotationComparator()
        
        self._create_widgets()
        self._load_ranking()
    
    def _create_widgets(self):
        """Tworzy interfejs z użyciem pionowego PanedWindow."""
        
        # GŁÓWNY KONTENER Z PIONOWYM PODZIAŁEM (Poziome belki)
        self.paned_window = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # =========================================================
        # 1. GÓRNY PANEL: Wczytywanie plików
        # =========================================================
        top_frame = ttk.LabelFrame(self.paned_window, text="Porównaj model z poprawionymi anotacjami (Ground Truth)", padding=10)
        self.paned_window.add(top_frame, weight=0) # weight=0 - domyślnie zajmuje tylko tyle miejsca, ile potrzebuje
        
        input_frame = ttk.Frame(top_frame)
        input_frame.pack(fill=tk.X, expand=True)
        input_frame.columnconfigure(1, weight=1) # Entry będzie się rozciągać poziomo
        
        # Model .pt
        ttk.Label(input_frame, text="Oceniany model (.pt):").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        self.model_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.model_var).grid(row=0, column=1, sticky=tk.EW, pady=2, padx=5)
        ttk.Button(input_frame, text=f"{self.icon_manager.get('file')} Wybierz", 
                   command=self._select_model).grid(row=0, column=2, pady=2, padx=5)
        
        # Auto XML
        ttk.Label(input_frame, text="Automatyczne anotacje (XML):").grid(row=1, column=0, sticky=tk.W, pady=2, padx=5)
        self.auto_xml_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.auto_xml_var).grid(row=1, column=1, sticky=tk.EW, pady=2, padx=5)
        ttk.Button(input_frame, text=f"{self.icon_manager.get('file')} Wybierz", 
                   command=self._select_auto_xml).grid(row=1, column=2, pady=2, padx=5)
        
        # Poprawione XML (Ground Truth)
        ttk.Label(input_frame, text="Poprawione anotacje (XML):").grid(row=2, column=0, sticky=tk.W, pady=2, padx=5)
        self.corr_xml_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.corr_xml_var).grid(row=2, column=1, sticky=tk.EW, pady=2, padx=5)
        ttk.Button(input_frame, text=f"{self.icon_manager.get('file')} Wybierz", 
                   command=self._select_corr_xml).grid(row=2, column=2, pady=2, padx=5)
        
        ttk.Button(top_frame, text=f"{self.icon_manager.get('chart')} Przeprowadź ocenę modelu", 
                   command=self._evaluate_model).pack(pady=10)
        
        # =========================================================
        # 2. ŚRODKOWY PANEL: Tabela Rankingu
        # =========================================================
        middle_frame = ttk.LabelFrame(self.paned_window, text="Ranking przetestowanych modeli", padding=10)
        self.paned_window.add(middle_frame, weight=2) # weight=2 - mocno się rozciąga podczas powiększania okna
        
        tree_frame = ttk.Frame(middle_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("#", "Model", "Dokładność %", "Precyzja %", "Czułość %", "F1", "Data")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        for col in columns:
            self.tree.heading(col, text=col)
            stretch = True if col == "Model" else False
            self.tree.column(col, width=120 if stretch else 90, stretch=stretch)
        
        scrollbar = WebSlimScrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_frame = ttk.Frame(middle_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Odśwież listę", command=self._load_ranking).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Generuj raport tekstowy", command=self._generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Usuń wybrany", command=self._delete_selected).pack(side=tk.LEFT, padx=5)
        
        # =========================================================
        # 3. DOLNY PANEL: Raport tekstowy
        # =========================================================
        bottom_frame = ttk.LabelFrame(self.paned_window, text="Szczegółowy raport", padding=5)
        self.paned_window.add(bottom_frame, weight=1) # weight=1 - rozciąga się, ale słabiej niż tabela
        
        self.report_text = tk.Text(bottom_frame, wrap=tk.WORD, state=tk.DISABLED, height=6)
        report_scroll = WebSlimScrollbar(bottom_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scroll.set)
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def apply_theme(self):
        palette = getattr(self.app, "palette", {})
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

        try:
            self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
        except Exception:
            pass

        try:
            self.app.style_text_widget(self.report_text)
        except Exception:
            pass

        try:
            self.tree.configure(style="Treeview")
        except Exception:
            pass

        try:
            self.report_text.configure(highlightbackground=border, highlightcolor=border)
        except Exception:
            pass
    
    def _select_model(self):
        file_path = filedialog.askopenfilename(title="Wybierz oceniany model .pt", filetypes=[("PyTorch models", "*.pt")])
        if file_path:
            self.model_var.set(file_path)
    
    def _select_auto_xml(self):
        file_path = filedialog.askopenfilename(title="Automatyczne anotacje z narzędzia (XML)", filetypes=[("CVAT XML", "*.xml")])
        if file_path:
            self.auto_xml_var.set(file_path)
            self._validate_xml(file_path, "Automatyczny")
    
    def _select_corr_xml(self):
        file_path = filedialog.askopenfilename(title="Poprawione anotacje przez człowieka (XML)", filetypes=[("CVAT XML", "*.xml")])
        if file_path:
            self.corr_xml_var.set(file_path)
            self._validate_xml(file_path, "Poprawiony (GT)")
    
    def _validate_xml(self, path: str, type_: str):
        success, msg, stats = validate_cvat_xml(Path(path))
        if success:
            self.app.update_status(f"{type_} XML OK: {stats['images']} obrazów, {stats['plates']} tablic", "check")
        else:
            self.app.update_status(f"{type_} XML BŁĄD: {msg}", "error")
    
    def _evaluate_model(self):
        model_path = self.model_var.get()
        auto_xml = self.auto_xml_var.get()
        corr_xml = self.corr_xml_var.get()
        
        if not all([model_path, auto_xml, corr_xml]):
            messagebox.showerror("Błąd", "Wypełnij wszystkie 3 pola (Model, Auto XML, Poprawione XML)")
            return
        
        if not Path(auto_xml).exists() or not Path(corr_xml).exists():
            messagebox.showerror("Błąd", "Wskazane pliki XML nie istnieją na dysku.")
            return
        
        try:
            self.app.set_processing(True)
            self.app.update_status("Porównywanie bounding boxów/polygonów (IoU)...", "chart")
            
            stats = self.comparator.compare(Path(auto_xml), Path(corr_xml))
            model_name = Path(model_path).name
            
            entry = self.ranking.add_entry(model_name, model_path, stats)
            
            self.app.set_processing(False)
            self.app.update_status("Ocena modelu zakończona", "success")
            
            messagebox.showinfo("Sukces", f"Model został oceniony!\nWynik F1-score: {entry.f1_score:.2f}%")
            self._load_ranking()
            
        except Exception as e:
            self.app.set_processing(False)
            messagebox.showerror("Błąd oceny", str(e))
            logger.error(f"Błąd modułu oceny: {e}")
    
    def _load_ranking(self):
        entries = self.ranking.get_ranking()
        self.tree.delete(*self.tree.get_children())
        
        for i, entry in enumerate(entries, 1):
            self.tree.insert("", tk.END, values=(
                i,
                entry.model_name,
                f"{entry.accuracy:.1f}",
                f"{entry.precision:.1f}",
                f"{entry.recall:.1f}",
                f"{entry.f1_score:.2f}",
                entry.date_evaluated[:10]
            ))
        
        report = self.ranking.generate_report()
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)
        self.report_text.config(state=tk.DISABLED)
    
    def _generate_report(self):
        report = self.ranking.generate_report()
        if not report.strip():
            messagebox.showinfo("Raport", "Brak danych do wygenerowania raportu.")
            return
        messagebox.showinfo("Szczegółowy Raport Rankingu", report)
    
    def _delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        
        if messagebox.askyesno("Potwierdź", "Na pewno chcesz usunąć ten wynik z rankingu?"):
            idx = self.tree.index(selection[0])
            self.ranking.delete_entry(idx)
            self._load_ranking()
