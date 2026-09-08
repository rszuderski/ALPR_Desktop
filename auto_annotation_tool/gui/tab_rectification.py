#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka: Prostowanie tablic (warpPerspective) z precyzyjnymi liniami podglądu.
"""

from __future__ import annotations
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..config import CONFIG, CV2_AVAILABLE, cv2, PIL_AVAILABLE, np, SESSION
from ..icons import IconManager
from ..rectification import PlateRectifier
from .inertial_scroll import InertialScrollController
from .web_slim_scrollbar import WebSlimScrollbar
from .zoomable_canvas import ZoomableCanvas

if PIL_AVAILABLE:
    from PIL import Image, ImageTk

from ..rectification.character_segmentation import CharacterSegmenter
from ..rectification.polygon_validator import PolygonValidator

class ScrollableFrame(ttk.Frame):
    """Pomocnicza klasa dla scrollowanego panelu bocznego."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self._inertial_scroll = InertialScrollController(self.canvas)
        self.vscroll = WebSlimScrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")
        
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.inner_id, width=e.width))
        
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        units = self._inertial_scroll.mousewheel_units(event)
        if units == 0:
            return None
        self._inertial_scroll.queue_canvas_by_units(
            self.canvas,
            units,
            magnitude=self._inertial_scroll.mousewheel_magnitude(event),
        )
        return "break"


class RectificationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.icon = IconManager
        self.frame = ttk.Frame(parent)
        self.current_segments = []  # Przechowuj segmenty
        self.current_bboxes = []    # Przechowuj współrzędne

        # ==================== ZAPAMIĘTYWANIE ŚCIEŻEK ====================
        # Wczytaj ostatnio używane ścieżki z sesji (jeśli dostępna)
        if SESSION:
            rectif_session = SESSION.get_rectification()
            default_output = str(Path.home() / "Auto-Annotation-Tool-Output" / "rectified_plates")
        else:
            rectif_session = {"images_dir": "", "xml_path": "", "output_dir": str(Path(CONFIG.DEFAULT_OUTPUT_DIR) / "rectified_plates")}
            default_output = str(Path(CONFIG.DEFAULT_OUTPUT_DIR) / "rectified_plates")
        
        # Zmienne ścieżek
        self.images_dir = tk.StringVar(value=rectif_session.get("images_dir", ""))
        self.xml_path = tk.StringVar(value=rectif_session.get("xml_path", ""))
        self.output_dir = tk.StringVar(value=rectif_session.get("output_dir", default_output))

        # Zmienne parametrów (bezpieczne)
        self.px_per_mm = tk.DoubleVar(value=2.0)
        self.plate_w_mm = tk.DoubleVar(value=520.0)
        self.plate_h_mm = tk.DoubleVar(value=110.0)
        self.auto_aspect = tk.BooleanVar(value=True)
        self.interp = tk.StringVar(value="lanczos4")
        self.two_stage = tk.BooleanVar(value=True)
        self.oversample = tk.DoubleVar(value=2.0)
        self.sharpen_var = tk.DoubleVar(value=0.0)
        self.contrast_var = tk.BooleanVar(value=False)
        self.deskew_var = tk.BooleanVar(value=False)
        self.manual_angle_var = tk.DoubleVar(value=0.0)

        # Dane i stan
        self.ann: Dict[str, List[List[Tuple[float, float]]]] = {}
        self.current_img_bgr = None
        self.current_img_name: Optional[str] = None
        self.current_plate_index = tk.IntVar(value=0)
        self._last_rectified_bgr = None

        self._build_ui()
        
        # Zarejestruj callback do zapisania sesji przy zamknięciu
        self.frame.bind("<Destroy>", self._on_closing)

    def apply_theme(self):
        palette = getattr(self.app, "palette", {})
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

        try:
            self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
        except Exception:
            pass

        try:
            self.app.style_listbox_widget(self.listbox, bordercolor=border)
        except Exception:
            pass

        for canvas_name in ("canvas_left", "canvas_right"):
            widget = getattr(self, canvas_name, None)
            if widget is None:
                continue
            try:
                self.app.style_canvas_widget(widget, background=palette.get("panel", "#252526"), bordercolor=border)
            except Exception:
                pass

    def _get_safe_int(self, var: tk.IntVar, default: int = 0) -> int:
        try: return var.get()
        except: return default

    def _get_safe_float(self, var: tk.DoubleVar, default: float = 0.0) -> float:
        try: return var.get()
        except: return default

    def _build_ui(self):
        pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_outer = ttk.Frame(pane)
        pane.add(left_outer, weight=0)
        left_scroll = ScrollableFrame(left_outer)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left = ttk.LabelFrame(left_scroll.inner, text="Parametry rektyfikacji", padding=10)
        left.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(pane, text="Podgląd", padding=10)
        pane.add(right, weight=1)

        # --- SEKCJA PLIKÓW ---
        ttk.Label(left, text="Folder obrazów:").pack(anchor=tk.W)
        r1 = ttk.Frame(left); r1.pack(fill=tk.X, pady=2)
        ttk.Entry(r1, textvariable=self.images_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r1, text="Wybierz", command=self._pick_images_dir).pack(side=tk.LEFT, padx=5)

        ttk.Label(left, text="CVAT XML:").pack(anchor=tk.W, pady=(5,0))
        r2 = ttk.Frame(left); r2.pack(fill=tk.X, pady=2)
        ttk.Entry(r2, textvariable=self.xml_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r2, text="Wybierz", command=self._pick_xml).pack(side=tk.LEFT, padx=5)
        ttk.Button(left, text="Wczytaj anotacje", command=self._load_annotations).pack(fill=tk.X, pady=5)

        self.listbox = tk.Listbox(left, height=6)
        self.listbox.pack(fill=tk.X, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_image)

        idx_row = ttk.Frame(left); idx_row.pack(fill=tk.X, pady=5)
        ttk.Label(idx_row, text="Index tablicy:").pack(side=tk.LEFT)
        ttk.Spinbox(idx_row, from_=0, to=99, textvariable=self.current_plate_index, width=5, command=self._update_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(idx_row, text="Podgląd", command=self._update_preview).pack(side=tk.RIGHT)

        # --- SEKCJA ROZMIARU I SKALI ---
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Skala (Px/mm):").pack(anchor=tk.W)
        
        row_pxmm = ttk.Frame(left); row_pxmm.pack(fill=tk.X, pady=2)
        ttk.Scale(row_pxmm, from_=0.5, to=10.0, variable=self.px_per_mm, orient=tk.HORIZONTAL, 
                  command=lambda e: self._update_preview()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ent_pxmm = ttk.Entry(row_pxmm, textvariable=self.px_per_mm, width=6)
        ent_pxmm.pack(side=tk.RIGHT, padx=5)
        ent_pxmm.bind("<Return>", lambda e: self._update_preview())
        
        dim = ttk.Frame(left); dim.pack(fill=tk.X, pady=5)
        ttk.Label(dim, text="Rozmiar mm:").pack(side=tk.LEFT)
        ttk.Entry(dim, textvariable=self.plate_w_mm, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(dim, text="x").pack(side=tk.LEFT)
        ttk.Entry(dim, textvariable=self.plate_h_mm, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(left, text="Auto proporcje", variable=self.auto_aspect, command=self._update_preview).pack(anchor=tk.W)

        # --- SEKCJA JAKOŚCI ---
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Interpolacja:").pack(anchor=tk.W)
        cb = ttk.Combobox(left, textvariable=self.interp, values=["nearest", "linear", "cubic", "lanczos4"], state="readonly")
        cb.pack(fill=tk.X, pady=2); cb.bind("<<ComboboxSelected>>", lambda e: self._update_preview())
        
        ttk.Label(left, text="Wyostrzanie (Sharpen):").pack(anchor=tk.W, pady=(5,0))
        row_sharp = ttk.Frame(left); row_sharp.pack(fill=tk.X, pady=2)
        ttk.Scale(row_sharp, from_=0.0, to=2.0, variable=self.sharpen_var, orient=tk.HORIZONTAL, 
                  command=lambda e: self._update_preview()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ent_sharp = ttk.Entry(row_sharp, textvariable=self.sharpen_var, width=6)
        ent_sharp.pack(side=tk.RIGHT, padx=5)
        ent_sharp.bind("<Return>", lambda e: self._update_preview())
        
        ttk.Checkbutton(left, text="Prostuj znaki (Auto Deskew)", variable=self.deskew_var, command=self._update_preview).pack(anchor=tk.W)
        
        ttk.Label(left, text="Korekta kąta:").pack(anchor=tk.W, pady=(5,0))
        ang_r = ttk.Frame(left); ang_r.pack(fill=tk.X)
        ttk.Scale(ang_r, from_=-15.0, to=15.0, variable=self.manual_angle_var, orient=tk.HORIZONTAL, 
                  command=lambda e: self._update_preview()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ent_ang = ttk.Entry(ang_r, textvariable=self.manual_angle_var, width=6)
        ent_ang.pack(side=tk.LEFT, padx=5)
        ent_ang.bind("<Return>", lambda e: self._update_preview())
        ttk.Button(ang_r, text="0", width=3, command=lambda: [self.manual_angle_var.set(0.0), self._update_preview()]).pack(side=tk.RIGHT)

        ttk.Checkbutton(left, text="Kontrast CLAHE", variable=self.contrast_var, command=self._update_preview).pack(anchor=tk.W)

        # --- SEKCJA ZAPISU ---
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Folder zapisu:").pack(anchor=tk.W)
        r_out = ttk.Frame(left); r_out.pack(fill=tk.X, pady=2)
        ttk.Entry(r_out, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(r_out, text="Wybierz", command=self._pick_output_dir).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left, text="Zapisz bieżącą", command=self._save_current).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Zapisz wszystkie", command=self._save_all_thread).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Segmentacja znaków:").pack(anchor=tk.W)
        
        seg_row = ttk.Frame(left); seg_row.pack(fill=tk.X, pady=5)
        ttk.Label(seg_row, text="Liczba znaków:").pack(side=tk.LEFT)
        self.num_segments_var = tk.IntVar(value=8)
        ttk.Spinbox(seg_row, from_=4, to=12, textvariable=self.num_segments_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left, text="📊 Segmentuj znaki", command=self._segment_characters).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="💾 Zapisz segmenty", command=self._save_segments).pack(fill=tk.X, pady=2)
        
        self.segments_status = ttk.Label(left, text="Brak segmentów", wraplength=250)
        self.segments_status.pack(anchor=tk.W, pady=5)

        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(left, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=5)
        self.status = ttk.Label(left, text="Gotowy", wraplength=250); self.status.pack(anchor=tk.W)

        # ==================== SEKCJA PODGLĄDU Z ZOOMEM ====================
        vpane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpane.pack(fill=tk.BOTH, expand=True)

        canvas_left_frame = ttk.LabelFrame(vpane, text="Oryginał", padding=5)
        self.canvas_left = ZoomableCanvas(canvas_left_frame, bg="gray20", highlightthickness=0)
        self.canvas_left.pack(fill=tk.BOTH, expand=True)

        canvas_right_frame = ttk.LabelFrame(vpane, text="Wynik", padding=5)
        self.canvas_right = ZoomableCanvas(canvas_right_frame, bg="gray20", highlightthickness=0)
        self.canvas_right.pack(fill=tk.BOTH, expand=True)

        vpane.add(canvas_left_frame, weight=1)
        vpane.add(canvas_right_frame, weight=1)

    def _pick_images_dir(self):
        p = filedialog.askdirectory(initialdir=self.images_dir.get() or str(Path.home()))
        if p:
            self.images_dir.set(p)
            if SESSION:
                SESSION.set("rectification", "images_dir", p)
                SESSION.save_session()

    def _pick_xml(self):
        initial_dir = str(Path(self.xml_path.get()).parent) if self.xml_path.get() else str(Path.home())
        p = filedialog.askopenfilename(
            filetypes=[("XML", "*.xml")],
            initialdir=initial_dir
        )
        if p:
            self.xml_path.set(p)
            if SESSION:
                SESSION.set("rectification", "xml_path", p)
                SESSION.save_session()

    def _pick_output_dir(self):
        p = filedialog.askdirectory(initialdir=self.output_dir.get() or str(Path.home()))
        if p:
            self.output_dir.set(p)
            if SESSION:
                SESSION.set("rectification", "output_dir", p)
                SESSION.save_session()

    def _update_preview(self):
        if self.current_img_bgr is None or self.current_img_name not in self.ann: return
        plates = self.ann[self.current_img_name]
        idx = self._get_safe_int(self.current_plate_index, 0)
        if idx >= len(plates): return

        pts = plates[idx]
        
        # Uporządkuj kolejność punktów przed rektyfikacją.
        pts = PolygonValidator.fix_polygon(pts)
        
        try:
            w_px, h_px = PlateRectifier.polygon_wh_px(pts)
            h_mm = self._get_safe_float(self.plate_h_mm, 110.0)
            w_mm = h_mm * (w_px/h_px) if self.auto_aspect.get() and h_px > 0 else self._get_safe_float(self.plate_w_mm, 520.0)
            
            ppm = self._get_safe_float(self.px_per_mm, 2.0)
            out_w, out_h = max(1, int(w_mm * ppm)), max(1, int(h_mm * ppm))

            rect = PlateRectifier.rectify(
                self.current_img_bgr, pts, out_w, out_h,
                interpolation=self.interp.get(),
                sharpen=self._get_safe_float(self.sharpen_var, 0.0),
                enhance_contrast=self.contrast_var.get(),
                do_deskew=self.deskew_var.get(),
                manual_angle=self._get_safe_float(self.manual_angle_var, 0.0)
            )
            self._last_rectified_bgr = rect

            if PIL_AVAILABLE:
                tmp_vis = self.current_img_bgr.copy()
                pts_arr = np.array(pts, np.int32)
                cv2.polylines(tmp_vis, [pts_arr], True, (0, 255, 0), 1)
                for pt in pts:
                    cv2.circle(tmp_vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
                
                img_l = Image.fromarray(cv2.cvtColor(tmp_vis, cv2.COLOR_BGR2RGB))
                
                if self.canvas_left.original_image is None:
                    self.canvas_left.set_image(img_l)
                else:
                    self.canvas_left.update_image_preserve_zoom(img_l)
                
                img_r = Image.fromarray(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
                
                if self.canvas_right.original_image is None:
                    self.canvas_right.set_image(img_r)
                else:
                    self.canvas_right.update_image_preserve_zoom(img_r)
                    
        except Exception as e: 
            self.status.config(text=f"Błąd: {str(e)}")

    def _load_annotations(self):
        xml_path_str = self.xml_path.get().strip()
        if not xml_path_str: return
        xml = Path(xml_path_str)
        if not xml.exists(): 
            messagebox.showerror("Błąd", "Plik XML nie istnieje.")
            return
            
        try:
            tree = ET.parse(xml)
            root = tree.getroot()
            ann = {}
            
            valid_labels = set(CONFIG.PLATE_LABELS) 
            valid_labels.update(['plate', 'license-plate', 'rejestracja', 'tablica', 'lp'])

            for image in root.findall(".//image"):
                name = image.get("name", "")
                plates = []
                
                for poly in image.findall("polygon"):
                    label = (poly.get("label") or "").lower().strip()
                    if label in valid_labels:
                        pts_raw = poly.get("points", "").replace('\n', '').replace(' ', '')
                        pts = [tuple(map(float, p.split(","))) for p in pts_raw.split(";") if "," in p]
                        if len(pts) >= 4:
                            # Uporządkuj poligon odczytany z XML.
                            pts = PolygonValidator.fix_polygon(pts[:4])
                            plates.append(pts)
                
                if not plates:
                    for box in image.findall("box"):
                        label = (box.get("label") or "").lower().strip()
                        if label in valid_labels:
                            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
                            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
                            pts = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
                            # Uporządkuj prostokąt zbudowany z bboxa.
                            pts = PolygonValidator.fix_polygon(pts)
                            plates.append(pts)

                if plates:
                    ann[name] = plates
            
            self.ann = ann
            self.listbox.delete(0, tk.END)
            for k in sorted(self.ann.keys()):
                self.listbox.insert(tk.END, f"{k} ({len(self.ann[k])})")
            
            self.status.config(text=f"Sukces! Wczytano {len(self.ann)} obrazów z tablicami.")
            if not self.ann:
                messagebox.showwarning("Uwaga", "W pliku XML nie znaleziono etykiet pasujących do tablic (plate, tablica itp.)")
                
        except Exception as e:
            messagebox.showerror("Błąd parsowania XML", f"Szczegóły: {e}")
    def _on_select_image(self, event=None):
        sel = self.listbox.curselection()
        if not sel: return
        img_name = self.listbox.get(sel[0]).split(" (")[0]
        img_path = Path(self.images_dir.get()) / img_name
        self.current_img_bgr = cv2.imread(str(img_path))
        self.current_img_name = img_name
        self.current_plate_index.set(0)
        self._update_preview()
        
    def _segment_characters(self):
        """Segmentuj znaki na bieżącym obrazie tablicy."""
        if self._last_rectified_bgr is None:
            messagebox.showwarning("Błąd", "Brak wyrównanego obrazu tablicy do segmentacji.")
            return
        
        try:
            num_segs = self.num_segments_var.get()
            self.current_segments, self.current_bboxes = CharacterSegmenter.segment_characters(
                self._last_rectified_bgr,
                num_segments=num_segs
            )
            
            if not self.current_segments:
                messagebox.showwarning("Błąd", "Nie udało się segmentować znaków.")
                return
            
            # Pokaż segmentację na canvas'ie
            vis_image = CharacterSegmenter.visualize_segments(
                self._last_rectified_bgr,
                self.current_bboxes
            )
            
            if PIL_AVAILABLE:
                img_vis = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                self.canvas_right.set_image(img_vis)
            
            self.segments_status.config(
                text=f"✅ Znaleziono {len(self.current_segments)} segmentów"
            )
            self.status.config(text=f"Segmentacja: {len(self.current_segments)} znaków")
            
        except Exception as e:
            messagebox.showerror("Błąd segmentacji", f"Szczegóły: {e}")
            self.status.config(text=f"Błąd: {str(e)}")
    
    def _save_segments(self):
        """Zapisz każdy segment jako osobny plik."""
        if not self.current_segments:
            messagebox.showwarning("Błąd", "Brak segmentów do zapisania. Najpierw segmentuj znaki.")
            return
        
        try:
            out_p = Path(self.output_dir.get()); out_p.mkdir(parents=True, exist_ok=True)
            segments_dir = out_p / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = f"{Path(self.current_img_name).stem}_p{self.current_plate_index.get()}"
            
            for i, segment in enumerate(self.current_segments):
                filename = f"{base_name}_seg_{i+1:02d}.png"
                cv2.imwrite(str(segments_dir / filename), segment)
            
            self.status.config(
                text=f"✅ Zapisano {len(self.current_segments)} segmentów do {segments_dir}"
            )
            messagebox.showinfo("Sukces", f"Segmenty zapisane do:\n{segments_dir}")
            
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Szczegóły: {e}")

    def _save_current(self):
        if self._last_rectified_bgr is None: return
        out_p = Path(self.output_dir.get()); out_p.mkdir(parents=True, exist_ok=True)
        name = f"{Path(self.current_img_name).stem}_p{self.current_plate_index.get()}.png"
        cv2.imwrite(str(out_p / name), self._last_rectified_bgr)
        self.status.config(text=f"Zapisano: {name}")

    def _save_all_thread(self):
        if not self.ann: return
        threading.Thread(target=self._save_all_worker, daemon=True).start()

    def _save_all_worker(self):
        out_p = Path(self.output_dir.get()); out_p.mkdir(parents=True, exist_ok=True)
        img_dir = Path(self.images_dir.get())
        items = list(self.ann.items()); total = len(items)
        ppm = self._get_safe_float(self.px_per_mm, 2.0)
        h_f = self._get_safe_float(self.plate_h_mm, 110.0)
        w_f = self._get_safe_float(self.plate_w_mm, 520.0)
        
        for i, (name, plates) in enumerate(items):
            img = cv2.imread(str(img_dir / name))
            if img is None: continue
            for p_idx, pts in enumerate(plates):
                w_px, h_px = PlateRectifier.polygon_wh_px(pts)
                w_mm = h_f * (w_px/h_px) if self.auto_aspect.get() and h_px > 0 else w_f
                rect = PlateRectifier.rectify(img, pts, max(1, int(w_mm*ppm)), max(1, int(h_f*ppm)),
                    interpolation=self.interp.get(), sharpen=self._get_safe_float(self.sharpen_var, 0.0),
                    enhance_contrast=self.contrast_var.get(), do_deskew=self.deskew_var.get(),
                    manual_angle=self._get_safe_float(self.manual_angle_var, 0.0))
                cv2.imwrite(str(out_p / f"{Path(name).stem}_p{p_idx}.png"), rect)
            self.frame.after(0, lambda p=((i+1)/total*100): self.progress_var.set(p))
        self.frame.after(0, lambda: messagebox.showinfo("Sukces", "Zapisano wszystko."))
    
    def _on_closing(self, event=None):
        """Callback przy zamknięciu zakładki - zapisz ostatnie ścieżki."""
        if SESSION:
            SESSION.set_rectification(
                images_dir=self.images_dir.get(),
                xml_path=self.xml_path.get(),
                output_dir=self.output_dir.get()
            )
            SESSION.save_session()
