"""Native MT campaign controls, results and raw-error review inside Z4.

Opening this frame never starts a model. Preparation verifies the approved
geometry and freezes the shared protocol; only an explicit Start/Resume click
calls the campaign executor. All expensive work runs outside the Tk thread.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image

from ..image_orientation import inspect_oriented_image
from ..ranking.eval396 import contained
from .zoomable_canvas import ZoomableCanvas


def runner_api():
    from ..ranking import mt_eval_runner
    return mt_eval_runner


def report_api():
    from ..ranking import mt_eval_report
    return mt_eval_report


def number(value, *, percent=False, digits=3):
    if value is None:
        return "—"
    try:
        value = float(value)
        if not math.isfinite(value):
            return "—"
        return ((f"{100 * value:.2f}%" if percent else f"{value:.{digits}f}").replace(".", ","))
    except (TypeError, ValueError):
        return "—"


def finite_xy(points):
    try:
        return all(len(point) >= 2 and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
                   for point in points)
    except (TypeError, ValueError, OverflowError):
        return False


class MTMeasurementPanel(ttk.Frame):
    COLORS = ("#277da1", "#f8961e", "#43aa8b")
    CHARTS = ("F1 bbox", "Precyzja bbox", "Czułość bbox", "Pokrycie Pose",
              "Rozkład IoU", "Rozkład błędu rogu [px]", "Rozkład błędu rogu / przekątna")

    def __init__(self, parent, selection_dir, *, geometry_host=None):
        super().__init__(parent, padding=12)
        self.selection_dir = Path(selection_dir)
        if self.selection_dir.name == "selection_manifest.json":
            self.selection_dir = self.selection_dir.parent
        self.output_root = self.selection_dir.parent.parent / "evaluation_runs"
        self.geometry_host = geometry_host
        self.prepared = self.campaign = None
        self.history = []
        self.error_rows = []
        self.active_error = None
        self.export_path = None
        self.busy = self.running = False
        self.alive = True
        self.cancelled = threading.Event()
        self.messages = queue.Queue()
        self.image_epoch = 0
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        ttk.Label(self, text="MT — pomiar EVAL396", font=("Segoe UI", 15, "bold")).grid(row=0, sticky="w")
        self.status = tk.StringVar(value="Pomiar nie został uruchomiony. Przygotuj protokół na podstawie zapisanego odbioru EXIF/GT.")
        self.status_label = ttk.Label(self, textvariable=self.status, wraplength=1000)
        self.status_label.grid(row=1, sticky="ew", pady=(4, 6))
        controls = ttk.LabelFrame(self, text="Wspólny protokół MT-n → MT-DAY → MT-NIGHT", padding=8)
        controls.grid(row=2, sticky="ew")
        controls.columnconfigure(0, weight=1)
        buttons = ttk.Frame(controls)
        buttons.grid(row=0, sticky="ew")
        ttk.Label(buttons, text="Urządzenie:").pack(side="left")
        self.device = tk.StringVar(value="0")
        self.device_choice = ttk.Combobox(buttons, textvariable=self.device, values=("0", "cpu"), state="readonly", width=7)
        self.device_choice.pack(side="left", padx=(5, 8))
        self.device_choice.bind("<<ComboboxSelected>>", self.invalidate_prepared)
        self.prepare_button = ttk.Button(buttons, text="Przygotuj i sprawdź protokół", command=self.prepare)
        self.prepare_button.pack(side="left")
        self.start_button = ttk.Button(buttons, text="[ MT ] Uruchom pomiar", command=self.launch, state="disabled")
        self.start_button.pack(side="left", padx=6)
        self.resume_button = ttk.Button(buttons, text="Wznów wybrany pomiar", command=self.resume, state="disabled")
        self.resume_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Anuluj po bieżącej scenie", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right")
        self.protocol_text = tk.StringVar(value="396 scen • bbox: 479 GT • indeksowane Pose: 478 GT • OCR: 449 (bez ponownego pomiaru MZ)")
        self.protocol_label = ttk.Label(controls, textvariable=self.protocol_text, wraplength=1000)
        self.protocol_label.grid(row=1, sticky="ew", pady=(6, 3))
        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=1188)
        self.progress.grid(row=2, sticky="ew")
        self.progress_text = tk.StringVar(value="0 / 1188 obrazów do przetworzenia • trzy modele wykonywane kolejno")
        ttk.Label(controls, textvariable=self.progress_text).grid(row=3, sticky="w", pady=(3, 0))

        self.results_notebook = ttk.Notebook(self)
        self.results_notebook.grid(row=3, sticky="nsew", pady=(8, 0))
        results = ttk.Frame(self.results_notebook, padding=8)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(3, weight=1)
        self.results_notebook.add(results, text="Tabela i wykresy")
        result_controls = ttk.Frame(results)
        result_controls.grid(row=0, sticky="ew")
        ttk.Label(result_controls, text="Warunki:").pack(side="left")
        self.domain = tk.StringVar(value="ALL")
        self.domain_choice = ttk.Combobox(result_controls, textvariable=self.domain, values=("ALL", "DAY", "NIGHT"), state="readonly", width=8)
        self.domain_choice.pack(side="left", padx=6)
        self.domain_choice.bind("<<ComboboxSelected>>", lambda _e: self.refresh_results())
        self.chart_kind = tk.StringVar(value=self.CHARTS[0])
        self.chart_choice = ttk.Combobox(result_controls, textvariable=self.chart_kind, values=self.CHARTS, state="readonly", width=29)
        self.chart_choice.pack(side="left")
        self.chart_choice.bind("<<ComboboxSelected>>", lambda _e: self.draw_chart())
        self.export_button = ttk.Button(result_controls, text="Eksport CSV / SVG / Markdown", command=self.export, state="disabled")
        self.export_button.pack(side="right")
        columns = ("model", "status", "precision", "recall", "f1", "counts", "iou", "pose", "coverage", "corner", "normalized")
        self.result_tree = ttk.Treeview(results, columns=columns, show="headings", height=3)
        for column, caption, width in zip(columns,
            ("Model", "Stan / sceny", "Precyzja", "Czułość", "F1", "TP / FP / FN", "IoU średnie", "Pose n / GT", "Pokrycie", "Błąd rogu px", "Róg / przekątna"),
            (88, 145, 78, 78, 75, 110, 85, 92, 85, 96, 110)):
            self.result_tree.heading(column, text=caption)
            self.result_tree.column(column, width=width, minwidth=width, stretch=column in ("model", "counts"), anchor="w" if column in ("model", "status") else "e")
        self.result_tree.grid(row=1, sticky="ew", pady=5)
        scroll = ttk.Scrollbar(results, orient="horizontal", command=self.result_tree.xview)
        self.result_tree.configure(xscrollcommand=scroll.set)
        scroll.grid(row=2, sticky="ew")
        self.chart = tk.Canvas(results, background="#101820", highlightthickness=0, height=210)
        self.chart.grid(row=3, sticky="nsew", pady=(5, 0))
        self.chart.bind("<Configure>", lambda _e: self.draw_chart())
        self.metrics_note = tk.StringVar(value="Brak pomiaru. Oficjalne AP nie są zastępowane średnim IoU ani błędem narożników.")
        self.metrics_note_label = ttk.Label(results, textvariable=self.metrics_note, wraplength=1000)
        self.metrics_note_label.grid(row=4, sticky="ew", pady=(5, 0))

        errors = ttk.Frame(self.results_notebook, padding=8)
        errors.columnconfigure(1, weight=1)
        errors.rowconfigure(1, weight=1)
        self.results_notebook.add(errors, text="Błędy i surowe predykcje")
        error_controls = ttk.Frame(errors)
        error_controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ttk.Label(error_controls, text="Model:").pack(side="left")
        self.error_model = tk.StringVar()
        self.error_model_choice = ttk.Combobox(error_controls, textvariable=self.error_model, state="readonly", width=16)
        self.error_model_choice.pack(side="left", padx=6)
        self.error_model_choice.bind("<<ComboboxSelected>>", lambda _e: self.refresh_errors())
        self.only_errors = tk.BooleanVar(value=True)
        ttk.Checkbutton(error_controls, text="Tylko FP / FN / ostrzeżenia Pose", variable=self.only_errors, command=self.refresh_errors).pack(side="left")
        self.error_tree = ttk.Treeview(errors, columns=("scene", "domain", "fp", "fn"), show="headings", height=7)
        for column, caption, width in (("scene", "Scena", 165), ("domain", "Domena", 65), ("fp", "FP", 35), ("fn", "FN", 35)):
            self.error_tree.heading(column, text=caption)
            self.error_tree.column(column, width=width, stretch=column == "scene")
        self.error_tree.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self.error_tree.bind("<<TreeviewSelect>>", self.select_error)
        self.error_canvas = ZoomableCanvas(errors, background="#101820", highlightthickness=0, width=650, height=280)
        self.error_canvas.min_zoom = .01
        self.error_canvas.show_info = False
        self.error_canvas.set_overlay_renderer(self.draw_error_overlay)
        self.error_canvas.grid(row=1, column=1, sticky="nsew")
        self.error_note = tk.StringVar(value="Turkus: GT bbox • zielony/pomarańczowy: GT • magenta: surowe bbox i k0–k3 modelu. Bez przestawiania punktów.")
        self.error_note_label = ttk.Label(errors, textvariable=self.error_note, wraplength=1000)
        self.error_note_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        history_frame = ttk.Frame(self.results_notebook, padding=8)
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        self.results_notebook.add(history_frame, text="Historia pomiarów")
        self.history_tree = ttk.Treeview(history_frame, columns=("date", "status", "models", "scenes", "directory"), show="headings", height=8)
        for column, caption, width in (("date", "Utworzono", 140), ("status", "Stan", 95), ("models", "Modele", 60),
                                       ("scenes", "Przetworzone", 90), ("directory", "Katalog", 260)):
            self.history_tree.heading(column, text=caption)
            self.history_tree.column(column, width=width, minwidth=width, stretch=column == "directory")
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        self.history_tree.bind("<<TreeviewSelect>>", self.select_history)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        history_scroll.grid(row=0, column=1, sticky="ns")
        self.refresh_button = ttk.Button(history_frame, text="Odśwież", command=self.refresh_history)
        self.refresh_button.grid(row=0, column=2, padx=(6, 0), sticky="n")
        self.bind("<Configure>", self.wrap_text, add="+")
        self.bind("<Destroy>", self.on_destroy, add="+")
        if isinstance(parent, ttk.Notebook):
            parent.bind("<<NotebookTabChanged>>", self.on_selected, add="+")
        self.history_loaded = False
        self.poll_job = self.after(80, self.poll)
        self.update_controls()

    @property
    def can_start(self):
        if not self.prepared or self.busy or self.running:
            return False
        protocol, decision = self.prepared["protocol"], self.prepared["decision"]
        valid = (decision.get("decision") == "APPROVED" and bool(protocol.get("protocol_sha256"))
                 and bool(decision.get("contract_sha256"))
                 and protocol.get("geometry_approval_contract_sha256") == decision["contract_sha256"]
                 and self.prepared["preflight"].get("review_sha256") == decision.get("review_sha256"))
        if self.geometry_host is not None:
            current = self.geometry_host.report
            valid = valid and not self.geometry_host.busy and (not current or current.get("review_sha256") == decision.get("review_sha256"))
        return bool(valid)

    @property
    def can_resume(self):
        return bool(self.can_start and self.campaign and self.campaign.get("status") in {"PARTIAL", "RUNNING"}
                    and self.campaign["protocol"].get("protocol_sha256") == self.prepared["protocol"].get("protocol_sha256"))

    def update_controls(self):
        if not self.alive:
            return
        self.prepare_button.configure(state="disabled" if self.busy else "normal")
        self.device_choice.configure(state="disabled" if self.busy else "readonly")
        self.sync_gate_controls()
        self.cancel_button.configure(state="normal" if self.running else "disabled")
        self.refresh_button.configure(state="disabled" if self.busy else "normal")
        self.export_button.configure(state="normal" if self.campaign and not self.busy else "disabled")
        if self.geometry_host is not None and self.geometry_host.winfo_exists():
            self.geometry_host.update_controls()

    def sync_gate_controls(self):
        if self.alive:
            self.start_button.configure(state="normal" if self.can_start else "disabled")
            self.resume_button.configure(state="normal" if self.can_resume else "disabled")

    def wrap_text(self, event):
        if event.widget is self:
            width = max(260, event.width - 30)
            self.status_label.configure(wraplength=width)
            self.protocol_label.configure(wraplength=width - 16)
            self.metrics_note_label.configure(wraplength=width - 16)
            self.error_note_label.configure(wraplength=width - 16)

    def on_selected(self, _event=None):
        if self.alive and isinstance(self.master, ttk.Notebook) and self.master.select() == str(self) and not self.history_loaded:
            self.refresh_history()

    def invalidate_prepared(self, _event=None, *, reason="Zmieniono urządzenie. Ponownie przygotuj i sprawdź wspólny protokół."):
        if not self.busy:
            self.prepared = None
            self.status.set(reason)
            self.update_controls()

    def work(self, kind, operation, message):
        if self.busy:
            return
        self.busy = True
        self.running = kind == "campaign_run"
        self.status.set(message)
        if not self.running:
            self.progress.configure(mode="indeterminate")
            self.progress.start(40)
        else:
            self.progress.configure(mode="determinate", maximum=1188, value=0)
        self.update_controls()

        def worker():
            try:
                result = operation()
                self.messages.put((kind, result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, name="z4-mt-" + kind, daemon=True).start()

    def progress_message(self, value):
        self.messages.put(("progress", value))

    def prepare(self):
        if self.busy:
            return
        self.prepared = None
        device = self.device.get()
        self.work("prepared", lambda: runner_api().prepare_campaign(self.selection_dir, device=device, progress=self.progress_message),
                  "Sprawdzam odbiór EXIF/GT, źródła i kontrakty modeli. Inferencja nie jest uruchamiana.")

    def launch(self):
        if not self.can_start:
            return
        if isinstance(self.master, ttk.Notebook):
            self.master.select(self)
        self.cancelled.clear()
        prepared = self.prepared
        self.work("campaign_run", lambda: runner_api().run_campaign(prepared, progress=self.progress_message, cancelled=self.cancelled.is_set),
                  "Uruchamiam trzy pomiary kolejno zgodnie z przygotowanym protokołem…")

    def resume(self):
        if not self.can_resume:
            return
        self.cancelled.clear()
        prepared, directory = self.prepared, self.campaign["run_dir"]
        self.work("campaign_run", lambda: runner_api().run_campaign(prepared, progress=self.progress_message,
                  cancelled=self.cancelled.is_set, resume_dir=directory), "Weryfikuję zapisane rekordy przed wznowieniem…")

    def cancel(self):
        if self.running:
            self.cancelled.set()
            self.status.set("Anulowanie zgłoszone. Bieżąca scena zostanie zapisana; pomiar pozostanie PARTIAL.")
            self.cancel_button.configure(state="disabled")

    def refresh_history(self):
        if not self.busy:
            self.work("history", lambda: runner_api().list_campaigns(self.output_root), "Odczytuję historię pomiarów MT…")

    def select_history(self, _event=None):
        selection = self.history_tree.selection()
        if self.busy or not selection:
            return
        directory = self.history[int(selection[0])]["run_dir"]
        self.campaign = None
        self.refresh_results()
        self.work("campaign", lambda: runner_api().load_campaign(directory), "Weryfikuję zapisany pomiar i jego rekordy…")

    def export(self):
        if self.busy or not self.campaign:
            return
        destination = filedialog.askdirectory(parent=self, title="Katalog eksportu raportu MT",
                                             initialdir=str(self.output_root.parent))
        if destination:
            directory = self.campaign["run_dir"]
            self.work("exported", lambda: report_api().export_mt_report(directory, Path(destination), progress=self.progress_message),
                      "Eksportuję CSV, SVG, Markdown i proweniencję…")

    def protocol_summary(self):
        if not self.prepared:
            return
        p = self.prepared["protocol"]
        settings = p["inference"]
        labels = " → ".join(m["label"] for m in p["models"])
        counts = p["metric_counts"]
        postprocess = (f"natywny end2end; IoU NMS={settings['iou']} nieaktywny" if settings.get("end2end")
                       else f"NMS IoU={settings['iou']}")
        self.protocol_text.set(f"{labels} | {counts['scenes']} scen • bbox {counts['bbox']} • Pose {counts['pose']} • archiwalne OCR {counts['ocr']}\n"
            f"imgsz={settings['imgsz']} • conf={settings['conf']} • {postprocess} • max_det={settings['max_det']} • "
            f"urządzenie={settings['device']} • {'FP16' if settings.get('half') else 'FP32'} • augment={settings.get('augment', False)} • "
            f"batch={settings.get('batch', 1)} • end2end={settings.get('end2end', '—')}\n"
            f"Protokół {p['protocol_sha256'][:16]} • zatwierdzony odbiór EXIF/GT: {p['geometry_approval_contract_sha256'][:16]}")

    def refresh_results(self):
        self.result_tree.delete(*self.result_tree.get_children())
        if not self.campaign:
            self.metrics_note.set("Brak pomiaru. Oficjalne AP nie są zastępowane średnim IoU ani błędem narożników.")
            self.draw_chart()
            return
        for index, model in enumerate(self.campaign["models"]):
            metrics = (model.get("metrics") or {}).get("domains", {}).get(self.domain.get())
            if metrics is None:
                values = [model["label"], model.get("status", "NOT_RUN")] + ["—"] * 9
            else:
                values = [model["label"], f"{model.get('status', 'PARTIAL')} {metrics.get('processed_scenes', 0)}/{metrics.get('cohort_scenes', 0)}", number(metrics.get("box_precision"), percent=True),
                    number(metrics.get("box_recall"), percent=True), number(metrics.get("box_f1"), percent=True),
                    f"{metrics.get('box_tp', 0)} / {metrics.get('box_fp', 0)} / {metrics.get('box_fn', 0)}",
                    number((metrics.get("box_iou") or {}).get("mean")),
                    f"{metrics.get('pose_valid_matched', 0)} / {metrics.get('pose_gt', 0)}",
                    number(metrics.get("pose_coverage"), percent=True), number((metrics.get("corner_error_px") or {}).get("mean"), digits=2),
                    number((metrics.get("corner_error_normalized") or {}).get("mean"), digits=4)]
            self.result_tree.insert("", "end", iid=str(index), values=values)
        labels = [model["label"] for model in self.campaign["models"]]
        self.error_model_choice.configure(values=labels)
        if self.error_model.get() not in labels and labels:
            self.error_model.set(labels[0])
        scope_note = ("COMPLETE: wszystkie sceny przetworzone. " if self.campaign["status"] == "COMPLETE" else
                      f"{self.campaign['status']}: miary tylko przetworzonych scen; oczekujące nie są FN. ")
        self.metrics_note.set(scope_note + "Pokrycie Pose: poprawne dopasowania / ocenione GT. "
            "Oficjalne Box AP i Pose AP: NIEPOLICZONE. Historyczne błędy etykiet MT ograniczają ocenę indeksów narożników.")
        self.draw_chart()
        self.refresh_errors()

    def draw_chart(self):
        if not self.alive:
            return
        canvas = self.chart
        canvas.delete("all")
        width, height = max(320, canvas.winfo_width()), max(70, canvas.winfo_height())
        if not self.campaign:
            canvas.create_text(width/2, height/2, text="Brak zapisanych wyników MT", fill="#c7d5df")
            return
        distribution = self.chart_kind.get().startswith("Rozkład")
        compact = height < 160
        left, right, top, bottom = 65, width - 25, 24 if compact else 32, height - (26 if compact else 42)
        for i in range(5):
            y = bottom - (bottom-top) * i/4
            canvas.create_line(left, y, right, y, fill="#31434e")
            canvas.create_text(left-8, y, text=f"{i*25}%", fill="#c7d5df", anchor="e", font=("Segoe UI", 8), tags="axis_tick")
        models = self.campaign["models"]
        if not distribution:
            key = {"F1 bbox": "box_f1", "Precyzja bbox": "box_precision", "Czułość bbox": "box_recall", "Pokrycie Pose": "pose_coverage"}[self.chart_kind.get()]
            stride = (right-left) / max(1, len(models))
            for index, model in enumerate(models):
                metric = (model.get("metrics") or {}).get("domains", {}).get(self.domain.get(), {}).get(key)
                x = left + stride * (index+.5)
                canvas.create_text(x, bottom+(13 if compact else 17), text=model["label"], fill="#e3eef4")
                if metric is not None:
                    y = bottom - (bottom-top) * float(metric)
                    canvas.create_rectangle(x-stride*.24, y, x+stride*.24, bottom, fill=self.COLORS[index % 3], outline="")
                    canvas.create_text(x, max(15, y-10), text=number(metric, percent=True), fill="#ffffff")
                else:
                    canvas.create_text(x, (top+bottom)/2, text="NIEPOLICZONE", fill="#b6c1c8")
        else:
            series = []
            is_iou = self.chart_kind.get() == "Rozkład IoU"
            error_key = "corner_errors_normalized" if "przekątna" in self.chart_kind.get() else "corner_errors_px"
            for model in models:
                values = []
                for scene in (model.get("metrics") or {}).get("scenes", []):
                    if self.domain.get() != "ALL" and scene["domain"] != self.domain.get():
                        continue
                    for match in scene.get("matched", []):
                        candidates = [match.get("iou")] if is_iou else (match.get(error_key) or [])
                        values.extend(float(v) for v in candidates if v is not None and math.isfinite(float(v)))
                series.append(sorted(values))
            maximum = 1.0 if is_iou else max([v for values in series for v in values] or [1.0])
            maximum = maximum if maximum > 0 else 1.0
            for i in range(5):
                x = left+(right-left)*i/4
                canvas.create_text(x, bottom+(13 if compact else 16), text=number(maximum*i/4, digits=2), fill="#c7d5df", tags="distribution_tick")
            for index, (model, values) in enumerate(zip(models, series)):
                color = self.COLORS[index % 3]
                canvas.create_text(left+index*190, 13, text=f"{model['label']} (n={len(values)})", fill=color, anchor="w")
                if values:
                    points = [left, bottom]
                    for i, value in enumerate(values, 1):
                        x = left+(right-left)*value/maximum
                        points.extend((x, bottom-(bottom-top)*(i-1)/len(values), x, bottom-(bottom-top)*i/len(values)))
                    points.extend((right, top))
                    canvas.create_line(*points, fill=color, width=2)
        if not compact:
            canvas.create_text(right, height-4, text="ECDF: odsetek próbek ≤ wartości na osi X" if distribution else self.chart_kind.get(),
                               fill="#c7d5df", anchor="se")

    def refresh_errors(self):
        self.error_tree.delete(*self.error_tree.get_children())
        self.error_rows = []
        self.active_error = None
        self.image_epoch += 1
        self.error_canvas.clear_image()
        if not self.campaign:
            return
        model = next((m for m in self.campaign["models"] if m["label"] == self.error_model.get()), None)
        for scene in ((model or {}).get("metrics") or {}).get("scenes", []):
            if self.domain.get() != "ALL" and scene["domain"] != self.domain.get():
                continue
            fp, fn = len(scene.get("unmatched_prediction_indices", [])), len(scene.get("unmatched_gt_ids", []))
            pose_problem = any(p.get("pose_status") not in (None, "VALID", "NOT_EVALUATED", "EXCLUDED", "EXCLUDED_BY_OPERATOR")
                               or p.get("raw_quad_status") == "REVERSED_WINDING"
                               for p in scene.get("predictions", []))
            if self.only_errors.get() and not (fp or fn or pose_problem):
                continue
            self.error_rows.append(scene)
            self.error_tree.insert("", "end", iid=str(len(self.error_rows)-1), values=(Path(scene["file"]).name, scene["domain"], fp, fn))

    def select_error(self, _event=None):
        selected = self.error_tree.selection()
        if not selected or not self.campaign:
            return
        detail = self.error_rows[int(selected[0])]
        preflight = self.campaign["preflight"]
        scene = next(s for s in preflight["scenes"] if s["sha256"] == detail["scene_sha256"])
        path = contained(preflight["source_scene_root"], scene["file"])
        self.image_epoch += 1
        epoch = self.image_epoch
        self.error_note.set("Weryfikuję SHA obrazu przed pokazaniem surowych predykcji…")

        def worker():
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
                    raise ValueError("ERROR_PREVIEW_SOURCE_SHA_CHANGED")
                pixels, info = inspect_oriented_image(path)
                if info != scene["image"] or hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
                    raise ValueError("ERROR_PREVIEW_PIXELS_CHANGED")
                preview = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                self.messages.put(("error_image", (epoch, scene, detail, preview)))
            except Exception as exc:
                self.messages.put(("image_error", (epoch, str(exc))))
        threading.Thread(target=worker, daemon=True, name="z4-mt-error-preview").start()

    def draw_error_overlay(self, canvas):
        if not self.active_error:
            return
        scene, detail = self.active_error
        for plate in scene["plates"]:
            box = plate.get("bbox")
            if box and len(box) == 4 and finite_xy([box[:2], box[2:]]):
                canvas.create_rectangle(*canvas.image_to_canvas_coords(*box[:2]), *canvas.image_to_canvas_coords(*box[2:]),
                                        outline="#39d9ff", width=2, dash=(5, 3), tags="preview_overlay")
            quad = plate.get("quad")
            if quad and len(quad) == 4 and finite_xy(quad):
                color = "#ffb347" if plate.get("pose_exclusion") else "#52ff87"
                points = [canvas.image_to_canvas_coords(float(p[0]), float(p[1])) for p in quad]
                canvas.create_line(*[v for p in points+points[:1] for v in p], fill=color, width=2, tags="preview_overlay")
                names = ("P1", "P2", "P3", "P4") if plate.get("pose_exclusion") else ("TL", "TR", "BR", "BL")
                for (x, y), name in zip(points, names):
                    canvas.create_text(x-6, y+8, text=name, fill=color, anchor="e", tags="preview_overlay")
        for prediction in detail.get("predictions", []):
            box = prediction.get("bbox")
            if box and len(box) == 4 and finite_xy([box[:2], box[2:]]):
                canvas.create_rectangle(*canvas.image_to_canvas_coords(*box[:2]), *canvas.image_to_canvas_coords(*box[2:]),
                                        outline="#ff63db", width=2, tags="preview_overlay")
            points = prediction.get("keypoints") or []
            coordinates = []
            for index, point in enumerate(points):
                if not finite_xy([point]):
                    continue
                x, y = canvas.image_to_canvas_coords(float(point[0]), float(point[1]))
                coordinates.append((x, y))
                canvas.create_oval(x-3, y-3, x+3, y+3, fill="#ff63db", outline="#101820", tags="preview_overlay")
                canvas.create_text(x+6, y-7, text=f"k{index}", fill="#ffb3ee", anchor="w", tags="preview_overlay")
            if len(coordinates) == 4 and len(points) == 4:
                canvas.create_line(*[v for p in coordinates+coordinates[:1] for v in p], fill="#ff63db", dash=(3, 3), tags="preview_overlay")

    def finish(self):
        self.busy = self.running = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.update_controls()

    def poll(self):
        if not self.alive:
            return
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "progress":
                    if isinstance(data, dict):
                        self.status.set(data.get("message", ""))
                        if data.get("stage") == "inference":
                            self.progress.configure(mode="determinate", maximum=data.get("total", 1188), value=data.get("completed", 0))
                            self.progress_text.set(f"{data.get('model') or 'MT'} — scena {data.get('scene', 0)}/{data.get('total_scenes', 396)} • "
                                                   f"łącznie {data.get('completed', 0)}/{data.get('total', 1188)}")
                    else:
                        self.status.set(str(data))
                elif kind == "prepared":
                    self.prepared = data
                    if self.geometry_host is not None:
                        self.geometry_host.adopt_verified_measurement_context(data)
                    self.finish()
                    self.protocol_summary()
                    self.progress.configure(value=0)
                    self.progress_text.set("0 / 1188 obrazów do przetworzenia • trzy modele wykonywane kolejno")
                    self.status.set("GOTOWE DO STARTU — protokół i zapisany odbiór EXIF/GT zweryfikowane. Pomiar rozpoczyna wyłącznie przycisk Uruchom pomiar.")
                elif kind in ("campaign", "campaign_run"):
                    self.campaign = data
                    self.finish()
                    completed = sum(len(model.get("rows", [])) for model in data["models"])
                    total = data["protocol"]["metric_counts"]["scenes"] * len(data["protocol"]["models"])
                    self.progress.configure(maximum=total, value=completed)
                    self.progress_text.set(f"Zapisane rekordy: {completed}/{total} • {data['status']}")
                    self.refresh_results()
                    self.results_notebook.select(0)
                    details = data.get("status_detail") or {}
                    reason = details.get("error") or details.get("reason") or ""
                    self.status.set(f"{data['status']} — {reason + ' — ' if reason else ''}{data['run_dir']}")
                    self.history_loaded = False
                elif kind == "history":
                    self.history = data
                    self.history_loaded = True
                    self.history_tree.delete(*self.history_tree.get_children())
                    for index, row in enumerate(data):
                        self.history_tree.insert("", "end", iid=str(index), values=(row.get("created_at", ""), row["status"],
                            f"{row.get('models_completed', 0)}/3", f"{row.get('scenes_completed', 0)}/1188", Path(row["run_dir"]).name))
                    self.finish()
                    self.status.set(f"Historia: {len(data)} pomiarów.")
                elif kind == "exported":
                    self.export_path = Path(data)
                    self.finish()
                    self.status.set(f"Raport zapisany: {self.export_path}")
                elif kind == "error_image":
                    epoch, scene, detail, preview = data
                    if epoch == self.image_epoch:
                        self.active_error = (scene, detail)
                        self.error_canvas.set_image_fit_to_view(preview)
                        self.error_note.set(f"{scene['file']} | SHA {scene['sha256']}\nTurkus: GT bbox • zielony/pomarańczowy: GT • magenta: surowe bbox i k0–k3. Bez sortowania i fallbacku.")
                elif kind == "image_error":
                    epoch, message = data
                    if epoch == self.image_epoch:
                        self.active_error = None
                        self.error_canvas.clear_image()
                        self.error_note.set("FAIL / STOP — " + message)
                elif kind == "error":
                    self.finish()
                    self.status.set("FAIL / STOP — " + data)
        except queue.Empty:
            pass
        self.poll_job = self.after(80, self.poll)

    def on_destroy(self, event):
        if event.widget is self:
            self.alive = False
            self.cancelled.set()
            self.after_cancel(self.poll_job)
