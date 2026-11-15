"""E2E campaign panel in the existing native EVAL396 Notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageDraw

from .zoomable_canvas import ZoomableCanvas
from ..image_orientation import read_oriented_bgr
from ..ranking.eval396 import contained
from ..rectification.plate_rectifier import PlateRectifier


def runner_api():
    from ..ranking import e2e_eval_runner
    return e2e_eval_runner


def report_api():
    from ..ranking import e2e_eval_report
    return e2e_eval_report


def pct(value):
    return "—" if value is None else f"{100*value:.2f}%".replace(".", ",")


class E2EMeasurementPanel(ttk.Frame):
    def __init__(self, parent, selection_dir):
        super().__init__(parent, padding=10)
        self.selection_dir = Path(selection_dir).resolve()
        if self.selection_dir.name == "selection_manifest.json":
            self.selection_dir = self.selection_dir.parent
        self.output_root = self.selection_dir.parent.parent / "evaluation_runs"
        self.alive, self.busy, self.running = True, False, False
        self._scene_progress = None
        self.prepared = self.approval = self.campaign = None
        self.messages, self.cancelled = queue.Queue(), threading.Event()
        self.epoch, self.poll_job = 0, None
        self.history, self.error_rows = [], []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        ttk.Label(self, text="E2E — pełna scena → tekst", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        choices = ttk.LabelFrame(self, text="1. Wybierz zapis MT i pary modeli — propozycja bez podpisu", padding=7)
        choices.grid(row=1, column=0, sticky="ew", pady=6)
        choices.columnconfigure(1, weight=1)
        ttk.Label(choices, text="Cache MT:").grid(row=0, column=0, sticky="w")
        caches = sorted(self.output_root.glob("EVAL396_MT_run_*"), reverse=True)
        caches = [str(path) for path in caches if path.is_dir() and not path.name.endswith('.partial')]
        self.cache = tk.StringVar(value=caches[0] if caches else "")
        self.cache_entry = ttk.Combobox(choices, textvariable=self.cache, values=caches)
        self.cache_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5)
        self.browse_button = ttk.Button(choices, text="Wybierz katalog…", command=self.browse)
        self.browse_button.grid(row=0, column=4)
        self.device = tk.StringVar(value="0")
        ttk.Label(choices, text="Urządzenie:").grid(row=1, column=0, sticky="w")
        self.device_combo = ttk.Combobox(choices, textvariable=self.device, values=("0", "cpu"), state="readonly", width=8)
        self.device_combo.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(choices, text="0 = CUDA:0 • FP32 • MT z cache • MZ na przewidzianych cropach").grid(row=1, column=2, columnspan=3, sticky="w")
        pair_frame = ttk.Frame(choices)
        pair_frame.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(5, 0))
        self.pair_vars, self.pair_buttons = {}, []
        for i, plate in enumerate(("MT-n", "MT-DAY", "MT-NIGHT")):
            ttk.Label(pair_frame, text=plate + " →").grid(row=i, column=0, sticky="w", padx=(0, 8))
            for j, chars in enumerate(("MZ-s", "MZ-DAY", "MZ-NIGHT")):
                var = tk.BooleanVar(value=i == j)
                button = ttk.Checkbutton(pair_frame, text=chars, variable=var)
                button.grid(row=i, column=j+1, sticky="w", padx=16)
                self.pair_vars[(plate, chars)] = var
                self.pair_buttons.append(button)
        self.signature_frame = ttk.LabelFrame(self, text="2. Odbiór protokołu — nie uruchamia pomiaru", padding=7)
        self.signature_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        self.signature_frame.columnconfigure(1, weight=1)
        self.operator = tk.StringVar()
        self.acknowledged = tk.BooleanVar(value=False)
        ttk.Label(self.signature_frame, text="Podpis operatora:").grid(row=0, column=0, sticky="w")
        self.operator_entry = ttk.Entry(self.signature_frame, textvariable=self.operator)
        self.operator_entry.grid(row=0, column=1, sticky="ew", padx=6)
        self.approve_button = ttk.Button(self.signature_frame, text="Zatwierdź protokół i pary", command=self.approve)
        self.approve_button.grid(row=0, column=2, rowspan=2, padx=5)
        self.ack_button = ttk.Checkbutton(self.signature_frame, text="Akceptuję widoczny protokół, pary, rektyfikację i metryki.",
                                          variable=self.acknowledged, command=self.signature_changed)
        self.ack_button.grid(row=1, column=0, columnspan=2, sticky="w")
        self.signature_frame.grid_remove()
        self.views = ttk.Notebook(self)
        self.views.grid(row=3, column=0, sticky="nsew")
        results, protocol_view, preview, history = (ttk.Frame(self.views, padding=6) for _ in range(4))
        for frame, title in ((results, "Wyniki i wykres"), (protocol_view, "Protokół i modele"), (preview, "Podgląd błędów"), (history, "Historia E2E")):
            self.views.add(frame, text=title)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(2, weight=1)
        self.domain = tk.StringVar(value="ALL")
        domain_combo = ttk.Combobox(results, textvariable=self.domain, values=("ALL", "DAY", "NIGHT"), state="readonly", width=10)
        domain_combo.grid(row=0, column=0, sticky="w")
        domain_combo.bind("<<ComboboxSelected>>", lambda _: self.refresh_results())
        self.table = self._tree(results, ("pair", "exact", "cer", "no_read", "precision", "recall", "f1", "iou", "conditional", "fp"),
            ("Para", "Exact (licznik / GT)", "Micro-CER", "NO_READ", "bbox Precision", "bbox Recall", "bbox F1", "Śr. IoU (TP)", "Conditional Exact", "FP OCR"), height=4)
        self.table.master.grid(row=1, column=0, sticky="ew", pady=4)
        self.table.column("pair", width=210)
        self.chart = tk.Canvas(results, background="#fafbf7", highlightthickness=0, height=170)
        self.chart.grid(row=2, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        protocol_view.rowconfigure(0, weight=1)
        protocol_view.columnconfigure(0, weight=1)
        self.protocol_text = tk.Text(protocol_view, wrap="word", height=10, font=("Segoe UI", 10))
        self.protocol_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(protocol_view, command=self.protocol_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.protocol_text.configure(yscrollcommand=scrollbar.set, state="disabled")
        preview.rowconfigure(1, weight=1)
        preview.columnconfigure(0, weight=1)
        self.errors = self._tree(preview, ("pair", "plate", "status", "text"), ("Para", "GT / indeks FP", "Przyczyna", "Odczyt"), height=3)
        self.errors.master.grid(row=0, column=0, sticky="ew")
        self.errors.bind("<<TreeviewSelect>>", self.select_error)
        canvases = ttk.Panedwindow(preview, orient="horizontal")
        canvases.grid(row=1, column=0, sticky="nsew", pady=5)
        self.scene_canvas = ZoomableCanvas(canvases, bg="#101820", highlightthickness=0)
        self.crop_canvas = ZoomableCanvas(canvases, bg="#101820", highlightthickness=0)
        canvases.add(self.scene_canvas, weight=3)
        canvases.add(self.crop_canvas, weight=2)
        self.preview_note = tk.StringVar(value="Podgląd jest dostępny po odczycie wyników. Nie uruchamia modeli.")
        ttk.Label(preview, textvariable=self.preview_note, wraplength=1000).grid(row=2, column=0, sticky="ew")
        history.rowconfigure(1, weight=1)
        history.columnconfigure(0, weight=1)
        self.history_button = ttk.Button(history, text="Odśwież historię E2E", command=self.refresh_history)
        self.history_button.grid(row=0, column=0, sticky="w")
        self.history_tree = self._tree(history, ("date", "status", "pairs", "path"), ("Data UTC", "Stan zapisu", "Pary", "Katalog"), height=5)
        self.history_tree.master.grid(row=1, column=0, sticky="nsew", pady=5)
        self.history_tree.bind("<<TreeviewSelect>>", self.select_history)
        footer = ttk.Frame(self)
        footer.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        footer.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="NOT_RUN — sprawdź źródła i protokół. F3-C wymaga osobnej dyspozycji operatora.")
        self.status_label = ttk.Label(footer, textvariable=self.status, wraplength=1050)
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=5)
        buttons = ttk.Frame(footer)
        buttons.grid(row=2, column=0, sticky="ew")
        self.prepare_button = ttk.Button(buttons, text="Sprawdź protokół E2E", command=self.prepare)
        self.start_button = ttk.Button(buttons, text="[ E2E ] Uruchom pomiar", command=self.launch)
        self.resume_button = ttk.Button(buttons, text="Wznów zapisany przebieg", command=self.resume)
        self.export_button = ttk.Button(buttons, text="Eksport CSV / SVG / PNG / MD", command=self.export)
        self.cancel_button = ttk.Button(buttons, text="Anuluj", command=self.cancel)
        for i, button in enumerate((self.prepare_button, self.start_button, self.resume_button, self.export_button, self.cancel_button)):
            button.grid(row=0, column=i, sticky="ew", padx=(0, 5))
            buttons.columnconfigure(i, weight=1)
        self.variable_traces = []
        for variable, callback in ((self.cache, self.invalidate), (self.device, self.invalidate), (self.operator, self.signature_changed)):
            self.variable_traces.append((variable, variable.trace_add("write", callback)))
        for variable in self.pair_vars.values():
            self.variable_traces.append((variable, variable.trace_add("write", self.invalidate)))
        self.bind("<Destroy>", self.destroyed, add="+")
        self.bind("<Configure>", self.resized, add="+")
        self.update_controls()
        self.draw_chart()
        self.poll_job = self.after(80, self.poll)

    @staticmethod
    def _tree(parent, columns, headings, height):
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height, selectmode="browse")
        for col, heading in zip(columns, headings):
            tree.heading(col, text=heading)
            tree.column(col, width=130, minwidth=75, stretch=True)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        return tree

    def chosen_pairs(self):
        return [pair for pair, var in self.pair_vars.items() if var.get()]

    @property
    def can_start(self):
        return bool(not self.busy and self.prepared and self.approval and self.campaign is None)

    @property
    def can_resume(self):
        return bool(not self.busy and self.campaign and self.campaign['status'] in {'PARTIAL', 'RUNNING'} and self.prepared and self.approval)

    def update_controls(self):
        def state(widget, yes):
            widget.configure(state="normal" if yes else "disabled")
        state(self.prepare_button, not self.busy and bool(self.cache.get()) and bool(self.chosen_pairs()))
        state(self.approve_button, not self.busy and self.prepared is not None and self.acknowledged.get() and bool(self.operator.get().strip()) and self.approval is None)
        state(self.start_button, self.can_start)
        state(self.resume_button, self.can_resume)
        state(self.export_button, not self.busy and self.campaign is not None and self.campaign['status'] != 'RUNNING')
        state(self.cancel_button, self.busy)
        for widget in (self.cache_entry, self.browse_button, self.operator_entry, self.ack_button, self.history_button, *self.pair_buttons):
            state(widget, not self.busy)
        self.device_combo.configure(state="disabled" if self.busy else "readonly")

    def resized(self, event):
        if event.widget is self:
            self.status_label.configure(wraplength=max(400, event.width-30))

    def browse(self):
        value = filedialog.askdirectory(parent=self, title="Ukończony przebieg MT EVAL396", initialdir=str(self.output_root))
        if value:
            self.cache.set(value)

    def invalidate(self, *_):
        self.epoch += 1
        self.prepared = self.approval = self.campaign = None
        self.acknowledged.set(False)
        self.signature_frame.grid_remove()
        self.clear_protocol()
        self.status.set("NOT_RUN — wybór zmieniony. Sprawdź aktualny protokół przed podpisem.")
        self.refresh_results()
        self.update_controls()

    def signature_changed(self, *_):
        was_signed = self.approval is not None
        self.approval = None
        self.epoch += 1
        if was_signed:
            self.status.set('Podpis zmieniony — zatwierdź ponownie protokół przed uruchomieniem.')
            if self.prepared:
                self.show_protocol()
        self.update_controls()

    def work(self, kind, operation, text):
        if self.busy:
            return
        self.busy, self.running = True, kind == "run"
        self._scene_progress = None
        self.cancelled.clear()
        self.status.set(text)
        self._show_busy_progress(reset=True)
        self.update_controls()
        epoch = self.epoch
        messages = self.messages
        def worker():
            try:
                messages.put((kind, epoch, operation()))
            except Exception as exc:
                messages.put(("error", epoch, (type(exc).__name__, str(exc), kind)))
        threading.Thread(target=worker, name="z4-e2e-" + kind, daemon=True).start()

    def _show_busy_progress(self, *, reset=False):
        # Repeated stage messages must not reset the moving marker. A previous
        # maximum=1 cannot be reused either: it makes the marker jump end to end.
        if reset or str(self.progress['mode']) != 'indeterminate':
            self.progress.stop()
            self.progress.configure(mode='indeterminate', maximum=200, value=0)
            self.progress.start(20)

    def _show_progress(self, value):
        if 'total' in value:
            completed, total = value.get('completed', 0), value['total']
            if self.running:
                self._scene_progress = (completed, total)
            if completed < total:
                self.progress.stop()
                self.progress.configure(mode='determinate', maximum=max(1, total), value=completed)
            else:
                # The last raw scene is not the end of source verification,
                # scoring and loading the verified report. No invented percent.
                self._show_busy_progress()
        elif 'detections' not in value:
            self._show_busy_progress()
        # Detection-only messages retain the current scene count. Uncounted
        # phases animate, using the native runner's events without text matching.
        message = value['message']
        if self._scene_progress:
            completed, total = self._scene_progress
            if total > 0 and completed >= total:
                message = f'Predykcje: {completed}/{total} scen-par. ' + message
        self.status.set(message)

    def progress_callback(self, *, checking=False):
        messages, epoch, cancelled = self.messages, self.epoch, self.cancelled
        def callback(value):
            if checking and cancelled.is_set():
                raise InterruptedError("Sprawdzanie anulowane")
            messages.put(("progress", epoch, value))
        return callback

    def prepare(self):
        if self.busy:
            return
        selection, cache, pairs, device = self.selection_dir, self.cache.get(), self.chosen_pairs(), self.device.get()
        self.prepared = self.approval = self.campaign = None
        self.signature_frame.grid_remove()
        self.clear_protocol()
        self.refresh_results()
        progress = self.progress_callback(checking=True)
        self.work("prepared", lambda: runner_api().prepare_campaign(selection, cache, pairs=pairs, device=device, progress=progress),
                  "Sprawdzam źródła, EXIF, raw MT i kontrakty MZ. Inferencja nie jest uruchamiana.")

    def approve(self):
        if self.busy or not self.prepared or not self.acknowledged.get() or not self.operator.get().strip():
            return
        prepared, operator = self.prepared, self.operator.get()
        self.work("approved", lambda: runner_api().save_approval(prepared, operator=operator, acknowledged=True), "Weryfikuję aktualność i zapisuję podpis protokołu…")

    def launch(self):
        if self.can_start:
            prepared, approval = self.prepared, self.approval
            progress, cancelled = self.progress_callback(), self.cancelled.is_set
            self.work("run", lambda: runner_api().run_campaign(prepared, approval, progress=progress, cancelled=cancelled),
                      "Uruchomiono E2E na jawne polecenie operatora…")

    def resume(self):
        if self.can_resume:
            prepared, approval, directory = self.prepared, self.approval, self.campaign['run_dir']
            progress, cancelled = self.progress_callback(), self.cancelled.is_set
            self.work("run", lambda: runner_api().run_campaign(prepared, approval, resume_dir=directory,
                      progress=progress, cancelled=cancelled), "Weryfikuję zapisane detekcje przed wznowieniem…")

    def cancel(self):
        self.cancelled.set()
        self.status.set("Zgłoszono anulowanie. Ukończona detekcja zostanie zapisana; przebieg pozostanie PARTIAL.")

    def refresh_history(self):
        output_root = self.output_root
        self.work("history", lambda: runner_api().list_campaigns(output_root), "Odczytuję historię E2E bez uruchamiania modeli…")

    def select_history(self, _event=None):
        selected = self.history_tree.selection()
        if self.busy or not selected:
            return
        path = self.history[int(selected[0])]['run_dir']
        self.work("loaded", lambda: runner_api().load_campaign(path), "Weryfikuję zapis E2E i jego źródła…")

    def export(self):
        if self.busy or self.campaign is None:
            return
        output = filedialog.askdirectory(parent=self, title="Katalog nowego raportu E2E", initialdir=str(self.output_root.parent))
        if output:
            directory = self.campaign['run_dir']
            progress = self.progress_callback(checking=True)
            self.work("exported", lambda: report_api().export_report(directory, output, progress=lambda text: progress({"message": text})), "Eksportuję zapisane wyniki…")

    def show_protocol(self):
        p = self.prepared['protocol']
        text = ("PROPOZYCJA / ODBIÓR — podpis nie uruchamia pomiaru. F3-C wymaga osobnej dyspozycji.\n"
            f"Dane: {p['evidence_kind']}\nProtokół SHA: {p['protocol_sha256']}\n"
            f"Podpis: {(self.approval or {}).get('operator_signature', 'BRAK — propozycja')}\n"
            "Kolejność: pełna scena EXIF → wszystkie raw MT → quad z kotwicą k0 → crop → MZ → tekst.\n"
            "Bez GT assist, bbox fallback, ROI, AutoZoom, EasyOCR i przełączania według domeny.\n"
            "Matching: bbox IoU ≥ 0,50, confidence/index → IoU/plate_id. Micro-CER zawiera NO_READ.\n\n" +
            json.dumps({k:p[k] for k in ('pairs','counts','rectification','inference','runtime','mt_models','mz_models')}, ensure_ascii=False, indent=2))
        self.protocol_text.configure(state='normal')
        self.protocol_text.delete('1.0','end')
        self.protocol_text.insert('1.0',text)
        self.protocol_text.configure(state='disabled')

    def clear_protocol(self):
        self.protocol_text.configure(state='normal')
        self.protocol_text.delete('1.0','end')
        self.protocol_text.insert('1.0','Brak aktualnego, zweryfikowanego protokołu E2E.')
        self.protocol_text.configure(state='disabled')

    def refresh_results(self):
        self.table.delete(*self.table.get_children())
        if self.campaign:
            for row in report_api().summary_rows(self.campaign, self.domain.get()):
                self.table.insert('', 'end', values=(row['pair'].replace('__',' + '),
                    f"{row['exact_matches']}/{row['plates']} ({pct(row['exact_match_rate'])})", pct(row['cer']), row['no_read'], pct(row['box_precision']), pct(row['box_recall']), pct(row['box_f1']), pct(row['box_iou_mean']),
                    f"{row['conditional_exact_matches']}/{row['conditional_plates']}", row['fp_ocr']))
        self.errors.delete(*self.errors.get_children())
        self.error_rows = []
        if self.campaign and self.campaign.get('scoring'):
            for pair, score in self.campaign['scoring'].items():
                for row in score['gt_rows']:
                    if row['failure']:
                        self.error_rows.append({'pair': pair, **row})
                for row in score['fp_rows']:
                    self.error_rows.append({'pair':pair, 'plate_id':'FP', **row})
            for index, row in enumerate(self.error_rows):
                self.errors.insert('', 'end', iid=str(index), values=(row['pair'], row['plate_id'], row['failure'] or 'FP_OCR', row['prediction']))
        self.scene_canvas.clear_image()
        self.crop_canvas.clear_image()
        self.draw_chart()

    def draw_chart(self):
        canvas = self.chart
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 300), max(canvas.winfo_height(), 80)
        rows = report_api().summary_rows(self.campaign, self.domain.get()) if self.campaign else []
        if not rows:
            canvas.create_text(width/2, height/2, text="Brak wyników E2E. Metryki pojawią się po rzeczywistym pomiarze i scoringu.", width=width-40, fill='#425646')
            return
        left, right, top, bottom = 190, width-65, 34, height-32
        for value in (0,25,50,75,100):
            x = left+(right-left)*value/100
            canvas.create_line(x,top,x,bottom,fill='#d6ded3')
            canvas.create_text(x,bottom+15,text=f'{value}%',fill='#425646')
        canvas.create_text(12,12,anchor='w',text=f"Exact Match — {self.domain.get()} • {self.campaign['protocol']['evidence_kind']}",fill='#214535')
        step = (bottom-top)/max(1,len(rows))
        for index,row in enumerate(rows):
            y = top+step*(index+.5)
            canvas.create_text(left-12,y,anchor='e',text=row['pair'].replace('__',' + '),fill='#214535')
            x = left+(right-left)*(row['exact_match_rate'] or 0)
            canvas.create_rectangle(left,y-9,x,y+9,fill='#36886b',outline='')
            canvas.create_text(x+7,y,anchor='w',text=pct(row['exact_match_rate']),fill='#214535')

    def select_error(self, _event=None):
        selected = self.errors.selection()
        if not selected or not self.campaign or self.busy:
            return
        row = self.error_rows[int(selected[0])]
        try:
            report = self.campaign
            pair = next(p for p in report['protocol']['pairs'] if p['id'] == row['pair'])
            source = next(s for s in report['prepared']['inputs'][pair['mt']] if s['scene_sha256'] == row['scene_sha256'])
            file = contained(report['protocol']['source_root'],source['file'])
            require_sha = hashlib.sha256(file.read_bytes()).hexdigest()
            if require_sha != source['scene_sha256']:
                raise ValueError('Zmieniono obraz źródłowy')
            pixels = read_oriented_bgr(file)
            if hashlib.sha256(pixels.tobytes()).hexdigest() != source['decoded_pixel_sha256']:
                raise ValueError('Zmieniono piksele EXIF')
            image = Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            scene = next(s for s in report['reference'] if s['sha256'] == row['scene_sha256'])
            for plate in scene['plates']:
                draw.rectangle(plate['bbox'], outline='#4be0d1', width=3)
            index = row['prediction_index']
            self.crop_canvas.clear_image()
            text = 'Brak dopasowanej predykcji MT; GT pokazano tylko do oceny.'
            if index is not None:
                record = next(r for s in report['pairs'][pair['id']] if s['scene_sha256'] == source['scene_sha256'] for r in s['detections'] if r['mt']['index'] == index)
                points = record['mt']['keypoints']
                if points and len(points)==4 and all(isinstance(v,(int,float)) for point in points for v in point):
                    draw.line([tuple(p) for p in points]+[tuple(points[0])], fill='#ffb34b',width=3)
                    for i,point in enumerate(points):
                        draw.text(tuple(point),f'k{i}',fill='#ffefcf')
                if record['rectification']['status']=='RECTIFIED':
                    crop, evidence = PlateRectifier.rectify_predicted_quad(pixels,points)
                    if crop is None or evidence['pixel_sha256'] != record['rectification']['pixel_sha256']:
                        raise ValueError('Podgląd cropu nie odpowiada zapisanej polityce / SHA')
                    preview = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                    overlay = ImageDraw.Draw(preview)
                    for char in record['reading']['ordered_predictions']:
                        overlay.rectangle(char['bbox'],outline='#ffc34a',width=1)
                        overlay.text((char['bbox'][0],max(0,char['bbox'][1]-10)),f"{char['reading_index']}:{char['character']}",fill='#ff453f')
                    self.crop_canvas.set_image_fit_to_view(preview)
                text=f"Raw indeks {index}; {record['rectification']['raw_quad_status']}; odczyt: {record['reading']['prediction'] or 'NO_READ'}"
            self.scene_canvas.set_image_fit_to_view(image)
            gt = (row.get('scoring') or {}).get('ground_truth','brak czytelnego GT')
            self.preview_note.set(f"{text} • GT: {gt} • {row['failure'] or 'FP'} • turkus: GT, pomarańcz: raw MT")
        except Exception as exc:
            self.preview_note.set('Błąd podglądu: '+str(exc))

    def poll(self):
        if not self.alive:
            return
        try:
            while True:
                kind, epoch, value = self.messages.get_nowait()
                if kind=='progress':
                    if epoch==self.epoch and self.busy:
                        self._show_progress(value)
                    continue
                self.busy = self.running = False
                self.progress.stop()
                self.progress.configure(mode='determinate', maximum=1, value=0)
                if epoch != self.epoch:
                    self.update_controls()
                    continue
                if kind=='prepared':
                    self.prepared, self.approval = value, None
                    self.acknowledged.set(False)
                    self.signature_frame.grid()
                    self.show_protocol()
                    self.views.select(1)
                    self.status.set('PASS preflight — zapoznaj się z protokołem i podpisz wybrane pary. Pomiar NOT_RUN.')
                    self.progress.configure(value=1)
                elif kind=='approved':
                    self.approval=value['approval']
                    self.show_protocol()
                    self.status.set('PROTOKÓŁ PODPISANY — pomiar NOT_RUN. Uruchomienie F3-C wymaga osobnej dyspozycji operatora.')
                elif kind in {'loaded','run'}:
                    self.campaign=value
                    self.prepared, self.approval=value['prepared'],value['approval']
                    self.show_protocol()
                    self.refresh_results()
                    self.views.select(0)
                    completed, total = value['completed_scenes'], value['total_scenes']
                    unfinished = value['status'] != 'COMPLETE' and total > 0 and completed >= total
                    self.progress.configure(maximum=max(1, total), value=0 if unfinished else completed)
                    message = f"{value['status']} — {completed}/{total} scen-par • {value['protocol']['evidence_kind']}"
                    if unfinished:
                        message += ' • Predykcje zapisane, ocena nieukończona.'
                    self.status.set(message)
                elif kind=='history':
                    self.history=value
                    self.history_tree.delete(*self.history_tree.get_children())
                    for i,row in enumerate(value):
                        self.history_tree.insert('','end',iid=str(i),values=(row['created_at'],row['status'],row['pairs'],row['run_dir']))
                    self.status.set(f'Historia E2E: {len(value)} zapisów. Otwarcie rekordu sprawdzi jego integralność.')
                elif kind=='exported':
                    self.status.set('Raport E2E zapisany: '+str(value))
                elif kind=='error':
                    self.prepared = self.approval = self.campaign = None
                    self.signature_frame.grid_remove()
                    self.clear_protocol()
                    self.refresh_results()
                    blocked = value[0] in {'PermissionError', 'FileNotFoundError'} or value[2] in {'prepared', 'approved'}
                    self.status.set('BLOCKED — '+value[0]+': '+value[1]+' • Start jest zablokowany; błąd źródeł nie jest NO_READ.' if blocked else 'BŁĄD / PARTIAL — '+value[1]+' • Zapisany przebieg można odszukać w historii.')
                self.update_controls()
        except queue.Empty:
            pass
        self.poll_job=self.after(80,self.poll)

    def destroyed(self,event):
        if event.widget is self:
            self.alive=False
            self.cancelled.set()
            for variable, trace in self.variable_traces:
                variable.trace_remove('write', trace)
            self.variable_traces.clear()
            if self.poll_job:
                self.after_cancel(self.poll_job)


def install_e2e_panel(dialog, notebook, selection_dir):
    if dialog.winfo_exists():
        panel = E2EMeasurementPanel(notebook, selection_dir)
        notebook.add(panel, text="E2E — pełna scena → tekst")
        dialog.e2e_panel = panel
