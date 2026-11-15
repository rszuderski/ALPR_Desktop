"""Read-only MT geometry review inside the EVAL396 Z4 result window."""
from __future__ import annotations

import hashlib
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image

from ..image_orientation import inspect_oriented_image
from ..ranking.eval396 import contained
from ..ranking.mt_geometry_review import (
    CORNER_NAMES, preflight_mt_geometry, save_operator_decision, load_operator_decision,
)
from .zoomable_canvas import ZoomableCanvas


def pose_excluded(plate):
    """Only an explicit metric exception can change the UI review semantics."""
    exception = plate.get("pose_exclusion")
    return bool(isinstance(exception, dict) and exception.get("plate_id") == plate.get("plate_id")
                and exception.get("policy_id") and exception.get("reason")
                and plate.get("bbox_eligible") is True and plate.get("pose_eligible") is False)


def blocking_issues(report):
    if not report:
        return []
    # Old preflights have no authorized metric exceptions.
    return report.get("blocking_geometry_issues", report.get("geometry_issues", []))


class GeometryReviewPanel(ttk.Frame):
    def __init__(self, parent, selection_dir):
        super().__init__(parent, padding=12)
        self.selection_dir = Path(selection_dir)
        if self.selection_dir.name == "selection_manifest.json":
            self.selection_dir = self.selection_dir.parent
        self.output_root = self.selection_dir.parent.parent / "evaluation_runs"
        self.report = self.scene = self.decision = None
        self.measurement_panel = None
        self.measurement_install_job = None
        self.reviewed = set()
        self.messages = queue.Queue()
        self.cancelled = threading.Event()
        self.busy = False
        self.epoch = 0
        self._wrapped_labels = []
        self._wrap_width = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        ttk.Label(self, text="MT — lokalizacja i geometria", font=("Segoe UI", 15, "bold")).grid(row=0, sticky="w")
        self.status = tk.StringVar(value="WERYFIKACJA EXIF/GT WYMAGANA — rozpocznij kontrolę źródeł i geometrii.")
        status_label = ttk.Label(self, textvariable=self.status, wraplength=1040)
        status_label.grid(row=1, sticky="ew", pady=(5, 4))
        self._wrapped_labels.append(status_label)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=2, sticky="ew", pady=(0, 6))
        toolbar = ttk.Frame(self)
        toolbar.grid(row=3, sticky="ew")
        self.check_button = ttk.Button(toolbar, text="[ MT ] Sprawdź geometrię i EXIF", command=self.check)
        self.check_button.pack(side="left")
        self.preview_button = ttk.Button(toolbar, text="[ MT ] Otwórz podgląd GT", command=self.show_scene, state="disabled")
        self.preview_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(toolbar, text="Anuluj sprawdzanie", command=self.cancelled.set, state="disabled")
        self.cancel_button.pack(side="right")
        source = ttk.Frame(self)
        source.grid(row=4, sticky="ew", pady=6)
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="Scena do odbioru:").grid(row=0, column=0, padx=(0, 8))
        self.scene_choice = ttk.Combobox(source, state="disabled")
        self.scene_choice.grid(row=0, column=1, sticky="ew")
        self.scene_choice.bind("<<ComboboxSelected>>", lambda _event: self.show_scene())
        self.plate_choice = ttk.Combobox(source, state="disabled", width=34)
        self.plate_choice.grid(row=0, column=2, padx=(8, 0))
        self.plate_choice.bind("<<ComboboxSelected>>", lambda _event: self.focus_plate())
        self.image_info = tk.StringVar(value="Zakres protokołu: 396 scen MT • 479 bbox • 478 Pose • 449 OCR. GT jest tylko do odczytu.")
        image_info_label = ttk.Label(source, textvariable=self.image_info, wraplength=1040, justify="left")
        image_info_label.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self._wrapped_labels.append(image_info_label)
        images = ttk.Frame(self)
        images.grid(row=5, sticky="nsew")
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=2)
        images.rowconfigure(1, weight=1)
        ttk.Label(images, text="Cały obraz wejściowy MT — po orientacji EXIF").grid(row=0, column=0, sticky="w")
        self.plate_info = tk.StringVar(value="Powiększenie GT — rolka: zoom, przeciąganie: przesuwanie")
        ttk.Label(images, textvariable=self.plate_info, wraplength=650).grid(row=0, column=1, sticky="w")
        self.overview = ZoomableCanvas(images, background="#101820", highlightthickness=0, width=340, height=300)
        self.detail = ZoomableCanvas(images, background="#101820", highlightthickness=0, width=620, height=300)
        for column, canvas in enumerate((self.overview, self.detail)):
            canvas.min_zoom = .01
            canvas.show_info = False
            canvas.set_overlay_renderer(self.draw_gt)
            canvas.grid(row=1, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
        legend_label = ttk.Label(self, text="Turkus: bbox GT • Zielony: TL/TR/BR/BL • Pomarańczowy: wyłączone Pose, oryginalne P1–P4 • Czerwony: błąd blokujący. GT bez edycji.",
                                 wraplength=1040)
        legend_label.grid(row=6, sticky="w", pady=5)
        self._wrapped_labels.append(legend_label)
        approval = ttk.LabelFrame(self, text="Odbiór operatora — granice obrazu nie potwierdzają położenia GT", padding=8)
        approval.grid(row=7, sticky="ew")
        approval.columnconfigure(1, weight=1)
        self.confirmed = tk.BooleanVar(value=False)
        self.confirm_button = ttk.Checkbutton(approval, text="Obejrzałem tę scenę: bbox i TL/TR/BR/BL pokrywają właściwą tablicę",
                                              variable=self.confirmed, command=self.confirm_scene, state="disabled")
        self.confirm_button.grid(row=0, column=0, columnspan=3, sticky="w")
        self.review_count = tk.StringVar(value="Potwierdzone sceny: 0/3")
        ttk.Label(approval, textvariable=self.review_count).grid(row=1, column=0, sticky="w", pady=4)
        self.operator = tk.StringVar()
        ttk.Label(approval, text="Podpis operatora:").grid(row=1, column=1, sticky="e", padx=8)
        ttk.Entry(approval, textvariable=self.operator, width=26).grid(row=1, column=2, sticky="e")
        self.operator.trace_add("write", lambda *_args: self.update_controls())
        ttk.Label(approval, text="Powód odrzucenia:").grid(row=2, column=0, sticky="w")
        self.reason = tk.StringVar()
        ttk.Entry(approval, textvariable=self.reason).grid(row=2, column=1, columnspan=2, sticky="ew")
        footer = ttk.Frame(self)
        footer.grid(row=8, sticky="ew", pady=(8, 0))
        self.approve_button = ttk.Button(footer, text="Zatwierdź protokół EXIF/GT", command=lambda: self.submit(True), state="disabled")
        self.approve_button.pack(side="left")
        self.reject_button = ttk.Button(footer, text="Odrzuć geometrię — STOP", command=lambda: self.submit(False), state="disabled")
        self.reject_button.pack(side="left", padx=6)
        self.measure_button = ttk.Button(footer, text="[ MT ] Uruchom pomiar", state="disabled")
        self.measure_button.pack(side="right")
        self.policy_info = tk.StringVar(value="Zatwierdzenie zapisuje kontrakt EXIF/GT. Osobna karta MT — pomiar i wyniki służy do przygotowania protokołu i jawnego uruchomienia modeli.")
        policy_label = ttk.Label(self, textvariable=self.policy_info, wraplength=1040)
        policy_label.grid(row=9, sticky="w", pady=(6, 0))
        self._wrapped_labels.append(policy_label)
        self.bind("<Configure>", self.wrap_review_text, add="+")
        self.bind("<Destroy>", self.on_destroy, add="+")
        self.poll_job = self.after(80, self.poll)
        if isinstance(parent, ttk.Notebook):
            self.measurement_install_job = self.after_idle(self.install_measurement_panel)

    def install_measurement_panel(self):
        self.measurement_install_job = None
        if self.measurement_panel is None and self.winfo_exists():
            from .z4_mt_results import MTMeasurementPanel
            self.measurement_panel = MTMeasurementPanel(self.master, self.selection_dir, geometry_host=self)
            self.master.add(self.measurement_panel, text="MT — pomiar i wyniki")
            self.update_controls()

    def adopt_verified_measurement_context(self, prepared):
        """Display an existing verified signature; never author a new decision."""
        self.report = prepared["preflight"]
        self.decision = prepared["decision"]
        self.reviewed = set(self.decision["reviewed_scene_hashes"])
        self.scene = None
        self.confirmed.set(False)
        self.epoch += 1
        self.overview.clear_image()
        self.detail.clear_image()
        self.set_review_choices()
        self.image_info.set(self.population_text() + " — zapisany odbiór EXIF/GT został zweryfikowany. Podgląd GT pozostaje dostępny.")
        self.status.set(self.gate_status() + " — kontrakt: " + str(prepared["approval_dir"]))
        self.update_controls()

    def set_review_choices(self):
        by_sha = {s["sha256"]: s for s in self.report["scenes"]}
        self.scene_choice.configure(values=[f"{'! GT / ' if any(p.get('geometry_issue') and not pose_excluded(p) for p in by_sha[sha]['plates']) else 'Pose: wyjątek / ' if any(pose_excluded(p) for p in by_sha[sha]['plates']) else ''}EXIF {by_sha[sha]['image']['exif_orientation']} — {by_sha[sha]['file']}"
                                           for sha in self.report["review_scene_hashes"]])
        if self.report["review_scene_hashes"]:
            self.scene_choice.current(0)

    def wrap_review_text(self, event):
        if event.widget is not self:
            return
        width = max(240, int(event.width) - 24)
        if width != self._wrap_width:
            self._wrap_width = width
            for label in self._wrapped_labels:
                label.configure(wraplength=width)

    def update_controls(self):
        measurement_busy = self.measurement_panel is not None and self.measurement_panel.busy
        self.check_button.configure(state="disabled" if self.busy or measurement_busy else "normal")
        ready = self.report is not None and not self.busy
        self.preview_button.configure(state="normal" if ready else "disabled")
        self.scene_choice.configure(state="readonly" if ready else "disabled")
        self.plate_choice.configure(state="readonly" if ready and self.scene else "disabled")
        valid_scene = self.scene_reviewable()
        self.confirm_button.configure(state="normal" if ready and valid_scene else "disabled")
        excluded = [p["plate_id"] for p in (self.scene or {}).get("plates", []) if pose_excluded(p)]
        self.confirm_button.configure(text=(
            "Obejrzałem bbox i zakres wyłączenia Pose; pozostałe narożniki są zgodne "
            f"({', '.join(excluded)}: bez oceny kolejności P1–P4)"
            if excluded else "Obejrzałem tę scenę: bbox i TL/TR/BR/BL pokrywają właściwą tablicę"))
        signed = bool(self.operator.get().strip())
        all_seen = (ready and self.report["status"] == "READY_FOR_REVIEW" and not blocking_issues(self.report)
                    and set(self.report["review_scene_hashes"]) == self.reviewed)
        self.approve_button.configure(state="normal" if all_seen and signed and not measurement_busy else "disabled")
        self.reject_button.configure(state="normal" if ready and signed and not measurement_busy else "disabled")
        measurement_ready = self.measurement_panel is not None and self.measurement_panel.can_start
        self.measure_button.configure(state="normal" if measurement_ready else "disabled",
            command=self.measurement_panel.launch if measurement_ready else "")
        if self.measurement_panel is not None:
            self.measurement_panel.sync_gate_controls()
        self.cancel_button.configure(state="normal" if self.busy and self.report is None else "disabled")
        self.review_count.set(f"Potwierdzone sceny: {len(self.reviewed)}/{len(self.report['review_scene_hashes']) if self.report else 3}")
        self.update_policy_info()

    def scene_reviewable(self):
        if not self.scene:
            return False
        if any(issue.get("sha256", issue.get("source_scene_sha256")) == self.scene["sha256"]
               for issue in blocking_issues(self.report)):
            return False
        return all(p.get("bbox_eligible") is not False
                   and (not p.get("geometry_issue") or pose_excluded(p)) for p in self.scene["plates"])

    def population_text(self):
        report = self.report or {}
        counts = report.get("metric_counts")
        if counts:
            return f"{counts['scenes']} scen MT • {counts['bbox']} bbox • {counts['pose']} Pose • {counts['ocr']} OCR"
        counts = report.get("counts", {})
        scenes = counts.get("scenes", len(report.get("scenes", [])))
        plates = counts.get("plates", sum(len(s["plates"]) for s in report.get("scenes", [])))
        return f"{scenes} scen MT • {plates} tablic GT"

    def update_policy_info(self):
        policy = (self.report or {}).get("metric_policy")
        if not policy:
            self.policy_info.set("Zatwierdzenie zapisuje kontrakt EXIF/GT. Osobna karta MT — pomiar i wyniki służy do przygotowania protokołu i jawnego uruchomienia modeli.")
            return
        models = " / ".join(m.get("label", "MT") for m in self.report.get("models", [])) or "MT-n / MT-DAY / MT-NIGHT"
        exclusions = self.report.get("metric_exclusions", [])
        reasons = "; ".join(f"{item['plate_id']}: {item['reason']}" for item in exclusions)
        authorization = policy.get("authorization", {})
        self.policy_info.set(f"Wspólna polityka {models}: {policy['id']} | SHA {self.report.get('metric_policy_sha256', '')[:12]}\n"
            f"Wyłączenie wyłącznie Pose — decyzja operatora {authorization.get('date', '')}: {reasons}. "
            "Bbox i sceny pozostają w ocenie. Modele uruchamiasz osobno w karcie MT — pomiar i wyniki.")

    def begin(self, text):
        self.busy = True
        self.status.set(text)
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(40)
        self.update_controls()

    def end(self, success=True):
        self.busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=100 if success else 0)
        self.update_controls()

    def check(self):
        if self.busy or (self.measurement_panel is not None and self.measurement_panel.busy):
            return
        if self.measurement_panel is not None:
            self.measurement_panel.invalidate_prepared(reason="Rozpoczęto nową weryfikację EXIF/GT. Przygotuj protokół pomiaru ponownie.")
        self.report = self.scene = self.decision = None
        self.reviewed.clear()
        self.confirmed.set(False)
        self.plate_info.set("Powiększenie GT — rolka: zoom, przeciąganie: przesuwanie")
        self.image_info.set("Zakres protokołu: 396 scen MT • 479 bbox • 478 Pose • 449 OCR. Sprawdzam źródła…")
        self.epoch += 1
        self.overview.clear_image()
        self.detail.clear_image()
        self.cancelled.clear()
        self.begin("WERYFIKACJA EXIF/GT WYMAGANA — sprawdzam niezmienność źródeł…")
        def worker():
            try:
                result = preflight_mt_geometry(self.selection_dir, cancelled=self.cancelled.is_set,
                    progress=lambda text: self.messages.put(("progress", text)))
                saved = load_operator_decision(result, self.output_root)
                self.messages.put(("checked", (result, saved)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def show_scene(self):
        if self.busy or self.report is None or self.scene_choice.current() < 0:
            return
        sha = self.report["review_scene_hashes"][self.scene_choice.current()]
        scene = next(s for s in self.report["scenes"] if s["sha256"] == sha)
        self.scene = None
        self.epoch += 1
        epoch = self.epoch
        path = contained(self.report["source_scene_root"], scene["file"])
        self.begin("Otwieram faktyczne piksele wejściowe MT i niezmienione GT…")
        def worker():
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
                    raise ValueError("PREVIEW_SOURCE_SHA_CHANGED")
                image, info = inspect_oriented_image(path)
                if info != scene["image"] or hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
                    raise ValueError("PREVIEW_PIXELS_CHANGED_SINCE_PREFLIGHT")
                preview = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.messages.put(("image", (epoch, scene, preview)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def gate_status(self):
        issues = blocking_issues(self.report)
        if issues:
            issue = issues[0]
            return f"FAIL / STOP — {len(issues)} problem GT: {issue['plate_id']} — {issue['reason']}. Podgląd pozostaje dostępny."
        if self.report and self.report.get("status") != "READY_FOR_REVIEW":
            return "FAIL / STOP — preflight nie jest gotowy do odbioru."
        if self.decision:
            return ("POLITYKA EXIF ZATWIERDZONA" if self.decision["decision"] == "APPROVED"
                    else "FAIL / STOP — " + self.decision["reason"])
        exclusion_note = " Wyłączenie Pose jest uwzględnione w protokole." if (self.report or {}).get("metric_exclusions") else ""
        return f"GOTOWE DO ODBIORU — {self.population_text()}.{exclusion_note} Wymagana wzrokowa akceptacja operatora."

    def draw_gt(self, canvas):
        if self.scene is None:
            return
        for plate in self.scene["plates"]:
            if not isinstance(plate["bbox"], (list, tuple)) or len(plate["bbox"]) != 4:
                continue
            x1, y1, x2, y2 = plate["bbox"]
            canvas.create_rectangle(*canvas.image_to_canvas_coords(x1, y1), *canvas.image_to_canvas_coords(x2, y2),
                                    outline="#39d9ff", width=2, dash=(5, 3), tags="preview_overlay")
            if not isinstance(plate["quad"], (list, tuple)) or len(plate["quad"]) != 4:
                continue
            xy = [canvas.image_to_canvas_coords(x, y) for x, y in plate["quad"]]
            excluded = pose_excluded(plate)
            names = tuple(f"P{i + 1}" for i in range(4)) if excluded or plate["geometry_issue"] else CORNER_NAMES
            color = "#ffb347" if excluded else "#ff8060" if plate["geometry_issue"] else "#52ff87"
            canvas.create_line(*[v for point in xy + xy[:1] for v in point], fill=color, width=3, tags="preview_overlay")
            for index, ((x, y), name) in enumerate(zip(xy, names)):
                canvas.create_oval(x-4, y-4, x+4, y+4, fill="#fff05a", outline="#101820", tags="preview_overlay")
                dx = -9 if index in (0, 3) else 9
                dy = -13 if index in (0, 1) else 13
                anchor = "e" if dx < 0 else "w"
                label = canvas.create_text(x+dx, y+dy, text=name, anchor=anchor, fill="#fff05a",
                                           font=("Segoe UI", 10, "bold"), tags="preview_overlay")
                background = canvas.create_rectangle(*canvas.bbox(label), fill="#101820", outline="", tags="preview_overlay")
                canvas.tag_lower(background, label)

    def focus_plate(self):
        if not self.scene or self.detail.original_image is None:
            return
        index = max(0, self.plate_choice.current())
        plate = self.scene["plates"][index]
        if pose_excluded(plate):
            self.plate_info.set(f"{plate['plate_id']} — bbox: w ocenie • Pose: wyłączone • oryginalne P1–P4")
        elif plate.get("geometry_issue"):
            self.plate_info.set(f"{plate['plate_id']} — błąd blokujący: {plate['geometry_issue']}")
        else:
            self.plate_info.set(f"{plate['plate_id']} — bbox i Pose: w ocenie • TL/TR/BR/BL")
        if not isinstance(self.scene["plates"][index]["bbox"], (tuple, list)) or len(self.scene["plates"][index]["bbox"]) != 4:
            return
        x1, y1, x2, y2 = self.scene["plates"][index]["bbox"]
        margin = max(50, (x2-x1) * .20)
        zoom = min(self.detail.max_zoom, self.detail.winfo_width() / (x2-x1 + 2*margin),
                   self.detail.winfo_height() / (y2-y1 + 2*margin))
        self.detail.set_view(zoom, self.detail.winfo_width()/2 - ((x1+x2)/2)*zoom,
                             self.detail.winfo_height()/2 - ((y1+y2)/2)*zoom)

    def confirm_scene(self):
        if self.scene_reviewable() and not self.busy:
            sha = self.scene["sha256"]
            self.reviewed.add(sha) if self.confirmed.get() else self.reviewed.discard(sha)
            self.update_controls()

    def submit(self, accepted):
        if self.busy or self.report is None or (self.measurement_panel is not None and self.measurement_panel.busy):
            return
        operator, reason = self.operator.get().strip(), self.reason.get().strip()
        if not operator or (accepted and (self.report["status"] != "READY_FOR_REVIEW" or blocking_issues(self.report)
                or self.reviewed != set(self.report["review_scene_hashes"]))) or (not accepted and not reason):
            return messagebox.showwarning("Odbiór geometrii", "Wymagany podpis i potwierdzenie wszystkich scen albo powód odrzucenia.", parent=self)
        report, reviewed = self.report, set(self.reviewed)
        if self.measurement_panel is not None:
            self.measurement_panel.invalidate_prepared(reason="Zmieniono decyzję EXIF/GT. Przygotuj protokół pomiaru ponownie.")
        self.begin("Ponownie sprawdzam SHA przed zapisaniem decyzji operatora…")
        def worker():
            try:
                result = save_operator_decision(report, self.output_root, operator=operator,
                    reviewed_scenes=reviewed, accepted=accepted, reason=reason)
                self.messages.put(("decision", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "progress":
                    self.status.set(data)
                elif kind == "checked":
                    self.report, saved = data
                    if saved:
                        _path, self.decision = saved
                        self.reviewed = set(self.decision["reviewed_scene_hashes"])
                    self.set_review_choices()
                    self.end()
                    self.show_scene()
                elif kind == "image":
                    epoch, scene, preview = data
                    if epoch != self.epoch:
                        continue
                    self.scene = scene
                    info = scene["image"]
                    self.image_info.set(f"{self.population_text()}\n{scene['file']}\nSHA: {scene['sha256']}\n"
                        f"EXIF={info['exif_orientation']} | raw {info['raw_size'][0]}×{info['raw_size'][1]} → "
                        f"wejście MT {info['oriented_size'][0]}×{info['oriented_size'][1]} | "
                        f"GT: {len(scene['plates'])} | MAE={info['decoder_comparison']['mae']:.2f}, p95={info['decoder_comparison']['p95']:.2f}")
                    self.confirmed.set(scene["sha256"] in self.reviewed)
                    self.plate_choice.configure(values=[p["plate_id"] + (" — bbox TAK / Pose NIE" if pose_excluded(p) else " — ! GT" if p["geometry_issue"] else "") for p in scene["plates"]])
                    self.plate_choice.current(next((i for i, p in enumerate(scene["plates"]) if p["geometry_issue"]), 0))
                    self.overview.set_image_fit_to_view(preview)
                    self.detail.set_image(preview)
                    self.focus_plate()
                    self.end()
                    self.status.set(self.gate_status())
                elif kind == "decision":
                    path, self.decision = data
                    self.end()
                    self.status.set(self.gate_status() + " — kontrakt: " + str(path))
                elif kind == "error":
                    if self.measurement_panel is not None:
                        self.measurement_panel.invalidate_prepared(reason="Weryfikacja EXIF/GT nie powiodła się. Pomiar pozostaje zablokowany.")
                    self.report = self.scene = self.decision = None
                    self.reviewed.clear()
                    self.epoch += 1
                    self.overview.clear_image()
                    self.detail.clear_image()
                    self.image_info.set("Nieudany preflight — dane podglądu niedostępne.")
                    self.plate_info.set("Podgląd GT niedostępny")
                    self.end(False)
                    self.status.set("FAIL / STOP — " + data)
        except queue.Empty:
            pass
        self.poll_job = self.after(80, self.poll)

    def on_destroy(self, event):
        if event.widget is self:
            if self.measurement_install_job is not None:
                self.after_cancel(self.measurement_install_job)
            self.cancelled.set()
            if self.progress.winfo_exists():
                self.progress.stop()
            self.after_cancel(self.poll_job)
