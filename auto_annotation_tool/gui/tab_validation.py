#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka walidacji.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from ..config import CONFIG, logger
from ..icons import IconManager
from ..validators import (
    validate_yolo_dataset, validate_model_file, 
    validate_cvat_xml, validate_coco_file
)
from ..utils import get_image_size
from .web_slim_scrollbar import WebSlimScrollbar


class ValidationTab:
    """Zakładka walidacji plików i datasetów."""
    
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.icon_manager = IconManager
        
        self.frame = ttk.Frame(parent)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Tworzy interfejs z w pełni responsywnym sub-notebookiem."""
        # Wewnętrzne zakładki dla różnych typów plików (Pełny expand)
        notebook = ttk.Notebook(self.frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- Walidacja datasetu YOLO ---
        ds_frame = ttk.Frame(notebook)
        notebook.add(ds_frame, text="Dataset YOLO (Folder)")
        
        input_ds_frame = ttk.Frame(ds_frame)
        input_ds_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_ds_frame, text="Ścieżka do folderu z datasetem:").pack(side=tk.LEFT)
        self.ds_var = tk.StringVar()
        ttk.Entry(input_ds_frame, textvariable=self.ds_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_ds_frame, text=f"{self.icon_manager.get('folder')} Wybierz", 
                   command=lambda: self._select_file_or_dir(self.ds_var, "dataset")).pack(side=tk.LEFT)
        
        self.ds_btn = ttk.Button(ds_frame, text=f"{self.icon_manager.get('check')} Uruchom walidację Datasetu", 
                                 command=self._validate_dataset)
        self.ds_btn.pack(pady=5)
        
        # TextBox na wynik z pełnym expandem
        self.ds_result = tk.Text(ds_frame, wrap=tk.WORD)
        ds_scroll = WebSlimScrollbar(ds_frame, orient=tk.VERTICAL, command=self.ds_result.yview)
        self.ds_result.configure(yscrollcommand=ds_scroll.set)
        self.ds_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        ds_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0,10))
        
        # --- Walidacja modelu ---
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model YOLO (.pt)")
        
        input_mod_frame = ttk.Frame(model_frame)
        input_mod_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_mod_frame, text="Ścieżka do pliku wagi (.pt):").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        ttk.Entry(input_mod_frame, textvariable=self.model_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_mod_frame, text=f"{self.icon_manager.get('file')} Wybierz", 
                   command=lambda: self._select_file_or_dir(self.model_var, ".pt")).pack(side=tk.LEFT)
        
        self.model_btn = ttk.Button(model_frame, text=f"{self.icon_manager.get('check')} Uruchom walidację Modelu", 
                                    command=self._validate_model)
        self.model_btn.pack(pady=5)
        
        self.model_result = tk.Text(model_frame, wrap=tk.WORD)
        model_scroll = WebSlimScrollbar(model_frame, orient=tk.VERTICAL, command=self.model_result.yview)
        self.model_result.configure(yscrollcommand=model_scroll.set)
        self.model_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        model_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0,10))
        
        # --- Walidacja CVAT XML ---
        cvat_frame = ttk.Frame(notebook)
        notebook.add(cvat_frame, text="CVAT XML (Anotacje)")
        
        input_cvat_frame = ttk.Frame(cvat_frame)
        input_cvat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_cvat_frame, text="Ścieżka do wyeksportowanego pliku .xml:").pack(side=tk.LEFT)
        self.cvat_var = tk.StringVar()
        ttk.Entry(input_cvat_frame, textvariable=self.cvat_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_cvat_frame, text=f"{self.icon_manager.get('file')} Wybierz", 
                   command=lambda: self._select_file_or_dir(self.cvat_var, ".xml")).pack(side=tk.LEFT)
        
        self.cvat_btn = ttk.Button(cvat_frame, text=f"{self.icon_manager.get('check')} Uruchom walidację XML", 
                                   command=self._validate_cvat)
        self.cvat_btn.pack(pady=5)
        
        self.cvat_result = tk.Text(cvat_frame, wrap=tk.WORD)
        cvat_scroll = WebSlimScrollbar(cvat_frame, orient=tk.VERTICAL, command=self.cvat_result.yview)
        self.cvat_result.configure(yscrollcommand=cvat_scroll.set)
        self.cvat_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        cvat_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0,10))
        
        # --- Narzędzia dodatkowe (Na samym dole głównego frame'a zakładki) ---
        tools_frame = ttk.LabelFrame(self.frame, text="Szybkie Narzędzia", padding=10)
        tools_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Button(tools_frame, text="Szybko sprawdź rozdzielczość obrazu", 
                   command=self._check_image_size).pack(side=tk.LEFT, padx=5)

    def apply_theme(self):
        palette = getattr(self.app, "palette", {})

        try:
            self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
        except Exception:
            pass

        for widget_name in ("ds_result", "model_result", "cvat_result"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                self.app.style_text_widget(widget)
            except Exception:
                pass
    
    def _select_file_or_dir(self, var: tk.StringVar, filter_: str):
        if filter_ == "dataset":
            path = filedialog.askdirectory(title="Wybierz folder zawierający dataset YOLO")
        else:
            types = [("Modele PyTorch", "*.pt")] if filter_ == ".pt" else [(f"Pliki {filter_}", f"*{filter_}")]
            path = filedialog.askopenfilename(title=f"Wybierz plik {filter_}", filetypes=types)
        if path:
            var.set(path)
    
    def _show_result(self, text_widget: tk.Text, success: bool, msg: str, stats: dict):
        text_widget.delete(1.0, tk.END)
        icon = self.icon_manager.get("check" if success else "error")
        text_widget.insert(tk.END, f"{icon} WYNIK: {msg}\n")
        text_widget.insert(tk.END, "="*50 + "\n\n")
        
        if stats:
            text_widget.insert(tk.END, "SZCZEGÓŁOWE STATYSTYKI:\n")
            for k, v in stats.items():
                if isinstance(v, dict):
                    text_widget.insert(tk.END, f"- {k}:\n")
                    for kk, vv in v.items():
                        text_widget.insert(tk.END, f"    {kk}: {vv}\n")
                elif isinstance(v, list):
                    text_widget.insert(tk.END, f"- {k}: {', '.join(map(str, v))}\n")
                else:
                    text_widget.insert(tk.END, f"- {k}: {v}\n")
        
        self.app.update_status(f"Zakończono proces walidacji ({'Sukces' if success else 'Błędy'})", "check" if success else "error")
    
    def _validate_dataset(self):
        path = Path(self.ds_var.get())
        if not path.exists():
            messagebox.showerror("Błąd", "Wskazana ścieżka do datasetu nie istnieje na dysku.")
            return
        self.app.update_status("Trwa walidacja folderu z datasetem...", "search")
        success, msg, stats = validate_yolo_dataset(path)
        self._show_result(self.ds_result, success, msg, stats)
    
    def _validate_model(self):
        path = Path(self.model_var.get())
        if not path.exists() or not path.is_file():
            messagebox.showerror("Błąd", "Plik modelu .pt nie istnieje.")
            return
        self.app.update_status("Ładowanie nagłówków modelu .pt...", "search")
        success, msg, stats = validate_model_file(path)
        self._show_result(self.model_result, success, msg, stats)
    
    def _validate_cvat(self):
        path = Path(self.cvat_var.get())
        if not path.exists() or not path.is_file():
            messagebox.showerror("Błąd", "Wskazany plik XML nie istnieje.")
            return
        self.app.update_status("Parsowanie pliku CVAT XML...", "search")
        success, msg, stats = validate_cvat_xml(path)
        self._show_result(self.cvat_result, success, msg, stats)
    
    def _check_image_size(self):
        file_path = filedialog.askopenfilename(title="Wybierz obraz (JPG/PNG)", filetypes=[("Obrazy", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            size = get_image_size(Path(file_path))
            messagebox.showinfo("Informacja o obrazie", f"Wybrany plik: {Path(file_path).name}\n\nRozdzielczość: {size[0]} x {size[1]} pikseli")
