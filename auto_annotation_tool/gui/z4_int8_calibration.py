"""Native Z4 preparation of calibration evidence; never starts a converter."""
from __future__ import annotations

import json
import gc
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ..exporters.int8_calibration import build_manifest
from ..exporters import exploratory_calibration as exploratory
from .z4_calibration_context import CalibrationContext, verify_selection, run_label, assert_manifest_context


class Int8CalibrationDialog(tk.Toplevel):
    def __init__(self, parent, *, context=None, on_saved=None, mode=None):
        super().__init__(parent)
        self.title("Bezpieczna kalibracja INT8 — MT / MZ")
        self.geometry("1080x800")
        self.minsize(800, 600)
        # A transient Windows dialog loses the standard minimize/maximize buttons.
        # Keep this tool modeless and respect intentional minimization on root focus.
        self.wm_transient("")
        self.resizable(True, True)
        self._aat_skip_window_recovery = True
        if self.tk.call("tk", "windowingsystem") == "win32":
            self.attributes("-toolwindow", False)
        self.result = None
        self._closed = self._busy = False
        self._poll_id = None
        self._events = queue.Queue()
        self._revision = 0
        self._lifecycle = {"revision": 0, "closed": False}
        self.context, self.on_saved = context, on_saved
        self._verified = self._source_audit = None
        self._saved_path = None
        self._job_kind = "manifest"
        self._internal_change = False
        self._run_records = {}
        self.run_display = tk.StringVar(self, "")
        self.card_text = tk.StringVar(self, "Model nieustalony — wybierz ukończony trening.")
        self.next_text = tk.StringVar(self, "")
        self.source_text = tk.StringVar(self, "EVAL396 i liczności train: jeszcze niezweryfikowane.")
        self._error = ""
        root = Path(__file__).resolve().parents[2]
        self.values = {key: tk.StringVar(self, value) for key, value in {
            "role": "MT — sceny", "history_path": str(root / "Workspace/5_training_runs/plates/training_history.json"),
            "run_id": "", "eval_inventory": str(root / "Workspace/E-MZ-DN-01/evaluation_selections/E-MZ-DN-01_EVAL396_20261009/selection_manifest.json"),
            "final_inventory": "", "image_size": "640", "limit": "300", "approved_by": "",
        }.items()}
        self.approval = tk.BooleanVar(self, False)
        self.mode = tk.StringVar(self, "Ścisła kalibracja INT8 — wymaga final_test")
        self.research_values = {k: tk.StringVar(self, v) for k, v in {
            "checkpoint_sha256": "", "confidence_threshold": "0.25", "iou_threshold": "0.45"}.items()}
        self.status = tk.StringVar(self, "Wybierz ukończony trening i inwentarze SHA. Brak final_test oznacza BLOCKED.")
        body = self._build_layout(root)
        for key, var in self.values.items():
            var.trace_add("write", lambda *_, k=key: self.settings_changed(k))
        self.approval.trace_add("write", self.invalidate)
        self.mode.trace_add("write", self.mode_changed)
        for key, var in self.research_values.items():
            var.trace_add("write", lambda *_, k=key: self.settings_changed(k))
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _: self.destroy())
        for widget in self._walk(body):
            if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton, tk.Checkbutton, tk.Text)):
                widget.bind("<FocusIn>", self._reveal, add="+")
            if not isinstance(widget, (tk.Text, ttk.Scrollbar)):
                widget.bind("<MouseWheel>", self._scroll, add="+")
        if context:
            self.values["role"].set("MT — sceny" if context.role == "plate" else "MZ — cropy")
            self.values["history_path"].set(context.history_path)
            self.values["run_id"].set(context.run_id)
            self.values["image_size"].set(str(context.image_size))
            self.research_values["confidence_threshold"].set(str(context.confidence_threshold))
            self.research_values["iou_threshold"].set(str(context.iou_threshold))
            self.run_display.set(f"{context.model_label or 'nazwa nieustalona'} · Run: {context.run_id}")
            self.role_combo.configure(state="disabled")
            for key in ("history_path", "run_id"):
                for widget in self.field_widgets[key][1:]:
                    widget.configure(state="readonly" if isinstance(widget, ttk.Entry) and not isinstance(widget, ttk.Combobox) else "disabled")
        if mode == "exploratory":
            self.mode.set(exploratory.LABEL)
        self._update_guidance()
        if context:
            self.read_checkpoint_sha()

    @staticmethod
    def _wrap(widget, inset=4):
        widget.bind("<Configure>", lambda e: widget.configure(wraplength=max(120, e.width - inset)))
        return widget

    def _build_layout(self, root):
        """Keep guidance/actions fixed; scroll only the grouped form and disclosures."""
        shell = ttk.Frame(self, padding=(16, 14, 16, 12))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        heading = ttk.Frame(shell)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="Kalibracja INT8", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        self.identity_badge = ttk.Label(heading, padding=(10, 4))
        self.identity_badge.grid(row=1, column=1, sticky="e", padx=(16, 0))
        self.identity_toggle = ttk.Button(heading, text="Pokaż tożsamość modelu i SHA", command=self.toggle_identity)
        self.identity_toggle.grid(row=0, column=1, sticky="e", padx=(16, 0))
        self.summary_title = self._wrap(ttk.Label(heading, font=("Segoe UI", 10, "bold")))
        self.summary_title.grid(row=1, column=0, sticky="ew", pady=(6, 3))
        self.summary_meta = self._wrap(ttk.Label(heading))
        self.summary_meta.grid(row=2, column=0, columnspan=2, sticky="ew")

        guide = ttk.LabelFrame(shell, text="Następny krok", padding=(12, 8))
        guide.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        guide.columnconfigure(0, weight=1)
        self.next_label = self._wrap(ttk.Label(guide, textvariable=self.next_text, justify="left", wraplength=650, style="Panel.TLabel"))
        self.next_label.grid(row=0, column=0, sticky="ew")
        self.next_button = ttk.Button(guide, command=self.next_action, padding=(12, 6), style="Accent.TButton")
        self.next_button.grid(row=0, column=1, sticky="e", padx=(16, 0))

        form = ttk.Frame(shell)
        form.grid(row=2, column=0, sticky="nsew")
        background = ttk.Style(self).lookup("TFrame", "background") or self.cget("background")
        self.canvas = tk.Canvas(form, background=background, highlightthickness=0, yscrollincrement=20)
        scroll = ttk.Scrollbar(form, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=(8, 0))
        self.canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(self.canvas)
        body.columnconfigure(0, weight=1)
        window = self.canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.entries, self.field_widgets = {}, {}

        self.identity_frame = ttk.LabelFrame(body, text="Tożsamość modelu i źródło treningu", padding=12)
        self.identity_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.identity_frame.columnconfigure(1, weight=1)
        ttk.Label(self.identity_frame, text="Tor modelu").grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.role_combo = ttk.Combobox(self.identity_frame, textvariable=self.values["role"], width=18,
                                       values=("MT — sceny", "MZ — cropy"), state="readonly")
        self.role_combo.grid(row=0, column=1, sticky="w", pady=(0, 8))
        self.role_combo.bind("<<ComboboxSelected>>", lambda _: self._role_changed(root))
        self._path_row(self.identity_frame, 1, "history_path", "Historia treningu")
        title = ttk.Label(self.identity_frame, text="Trening / run")
        title.grid(row=2, column=0, sticky="w", padx=(0, 16), pady=4)
        self.run_combo = ttk.Combobox(self.identity_frame, textvariable=self.run_display, state="readonly", width=20)
        self.run_combo.grid(row=2, column=1, sticky="ew", pady=4)
        self.run_combo.bind("<<ComboboxSelected>>", self.select_run)
        load = ttk.Button(self.identity_frame, text="Wczytaj treningi", command=self.load_runs)
        load.grid(row=2, column=2, sticky="ew", padx=(8, 0), pady=4)
        self.entries["run_id"] = self.run_combo
        self.field_widgets["run_id"] = [title, self.run_combo, load]
        self.card = self._wrap(ttk.Label(self.identity_frame, textvariable=self.card_text, justify="left", wraplength=850))
        self.card.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        if self.context:
            self.identity_frame.grid_remove()
        else:
            self.identity_toggle.configure(text="Ukryj tożsamość modelu i SHA")

        sources = ttk.LabelFrame(body, text="Tryb i niezależność danych", padding=10)
        sources.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        sources.columnconfigure(1, weight=1)
        ttk.Label(sources, text="Tryb kalibracji").grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.mode_combo = ttk.Combobox(sources, textvariable=self.mode, state="readonly", width=25,
            values=("Ścisła kalibracja INT8 — wymaga final_test", exploratory.LABEL))
        self.mode_combo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 4))
        self._path_row(sources, 1, "eval_inventory", "Inwentarz EVAL396")
        self._path_row(sources, 2, "final_inventory", "Inwentarz final_test")
        self.final_notice = self._wrap(ttk.Label(sources,
            text="final_test nie jest wymagany w trybie badawczym · FINAL_TEST_NOT_VERIFIED", wraplength=700))
        self.final_notice.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        self.final_notice.grid_remove()
        self.source_label = self._wrap(ttk.Label(sources, textvariable=self.source_text, wraplength=850))
        self.source_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        params = self.params_frame = ttk.LabelFrame(body, text="Parametry i zatwierdzenie", padding=10)
        params.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        params.columnconfigure(0, weight=1)
        numbers = ttk.Frame(params, style="Panel.TFrame")
        numbers.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for col, (key, label) in enumerate((("image_size", "Wejście (imgsz)"), ("limit", "Limit próbek"))):
            title = ttk.Label(numbers, text=label)
            title.grid(row=0, column=col * 2, sticky="w", padx=(0 if col == 0 else 24, 8))
            entry = ttk.Entry(numbers, textvariable=self.values[key], width=8)
            entry.grid(row=0, column=col * 2 + 1, sticky="w")
            self.entries[key] = entry
            self.field_widgets[key] = [title, entry]
        operator = self.operator_frame = ttk.Frame(params, style="Panel.TFrame")
        operator.grid(row=1, column=0, sticky="ew")
        title = ttk.Label(operator, text="Operator")
        title.pack(side="left", padx=(0, 12))
        entry = ttk.Entry(operator, textvariable=self.values["approved_by"], width=32)
        entry.pack(side="left")
        self.entries["approved_by"] = entry
        self.field_widgets["approved_by"] = [title, entry]
        # Created after the operator entry to retain the limit → operator Tab order.
        self.suggest_button = ttk.Button(params, text="Wstaw 300", command=lambda: self.values["limit"].set("300"))
        # Layout in the numeric row, but retain limit → operator in the Tab order.
        self.suggest_button.grid(in_=numbers, row=0, column=4, sticky="w", padx=(8, 0))
        self.approval_check = self._check(params, self.approval,
            "Zatwierdzam źródło treningowe i wybór wyłącznie splitu train")
        self.approval_check.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.research_frame = ttk.LabelFrame(params, text="Badawcza kalibracja · LiteRT CPU", padding=8)
        self.research_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.research_frame.columnconfigure(1, weight=1)
        ttk.Label(self.research_frame, text="SHA checkpointu").grid(row=0, column=0, sticky="w", padx=(0, 12))
        entry = ttk.Entry(self.research_frame, textvariable=self.research_values["checkpoint_sha256"], width=20)
        entry.grid(row=0, column=1, sticky="ew")
        self.entries["checkpoint_sha256"] = entry
        self.sha_button = ttk.Button(self.research_frame, text="Sprawdź SHA", command=self.read_checkpoint_sha)
        self.sha_button.grid(row=0, column=2, padx=(8, 0))
        thresholds = ttk.Frame(self.research_frame, style="Panel.TFrame")
        thresholds.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        for col, (key, label) in enumerate((("confidence_threshold", "conf"), ("iou_threshold", "IoU"))):
            ttk.Label(thresholds, text=label).grid(row=0, column=col * 2, padx=(0 if col == 0 else 24, 8))
            entry = ttk.Entry(thresholds, textvariable=self.research_values[key], width=8)
            entry.grid(row=0, column=col * 2 + 1)
            self.entries[key] = entry
        self.research_notice = ttk.Label(self.research_frame,
            text="EXPLORATORY_ONLY · FINAL_TEST_NOT_VERIFIED\nWynik badawczy, nie końcowy.", wraplength=500)
        self.research_notice.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.research_frame.grid_remove()
        for section in (self.identity_frame, sources, params):
            for widget in self._walk(section):
                if isinstance(widget, ttk.Label):
                    widget.configure(style="Panel.TLabel")

        report_header = ttk.Frame(body)
        report_header.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.report_toggle = ttk.Button(report_header, text="Pokaż szczegóły kontroli", command=self.toggle_report)
        self.report_toggle.pack(side="left")
        self.report_summary = ttk.Label(report_header, text="Manifest: nieprzygotowany")
        self.report_summary.pack(side="left", padx=(12, 0))
        self.report_frame = ttk.Frame(body)
        self.report_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.report_frame.columnconfigure(0, weight=1)
        self.status_label = self._wrap(ttk.Label(self.report_frame, textvariable=self.status, wraplength=850))
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.details = scrolledtext.ScrolledText(self.report_frame, width=40, height=8, wrap="word", state="disabled")
        self.details.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.details.grid_remove()
        self.report_frame.grid_remove()

        ttk.Separator(shell).grid(row=3, column=0, sticky="ew", pady=(12, 10))
        actions = ttk.Frame(shell)
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(2, weight=1)
        self.prepare_button = ttk.Button(actions, text="Sprawdź i przygotuj manifest", command=self.prepare, padding=(12, 6))
        self.prepare_button.grid(row=0, column=0, sticky="w")
        self.save_button = ttk.Button(actions, text="Zapisz manifest / raport", command=self.save, state="disabled", padding=(12, 6))
        self.save_button.grid(row=0, column=1, padx=(8, 0))
        self.close_button = ttk.Button(actions, text="Zamknij", command=self.destroy, padding=(12, 6))
        self.close_button.grid(row=0, column=3, sticky="e")
        return body

    def _path_row(self, parent, row, key, label):
        title = ttk.Label(parent, text=label)
        title.grid(row=row, column=0, sticky="w", padx=(0, 16), pady=4)
        entry = ttk.Entry(parent, textvariable=self.values[key], width=20)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(parent, text="Wybierz JSON", command=lambda: self.choose(key))
        button.grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=4)
        self.entries[key] = entry
        self.field_widgets[key] = [title, entry, button]

    def _check(self, parent, variable, text):
        style = ttk.Style(self)
        bg = style.lookup("TLabelframe", "background") or self.cget("background")
        fg = style.lookup("TLabel", "foreground") or "black"
        return self._wrap(tk.Checkbutton(parent, variable=variable, text=text, anchor="w", justify="left",
            background=bg, foreground=fg, activebackground=bg, activeforeground=fg, selectcolor=bg,
            wraplength=700, borderwidth=0, highlightthickness=0), inset=30)

    def toggle_identity(self):
        shown = bool(self.identity_frame.winfo_manager())
        self.identity_frame.grid_remove() if shown else self.identity_frame.grid()
        self.identity_toggle.configure(text=("Pokaż" if shown else "Ukryj") + " tożsamość modelu i SHA")

    def toggle_report(self):
        shown = bool(self.report_frame.winfo_manager())
        self.report_frame.grid_remove() if shown else self.report_frame.grid()
        self.report_toggle.configure(text=("Pokaż" if shown else "Ukryj") + " szczegóły kontroli")

    def _role_changed(self, root):
        mt = self.values["role"].get().startswith("MT")
        self.values["history_path"].set(str(root / "Workspace/5_training_runs" / ("plates" if mt else "chars") / "training_history.json"))
        self.values["image_size"].set(("512" if mt else "320") if self.is_research() else ("640" if mt else "256"))
        self.research_values["checkpoint_sha256"].set("")
        self.values["run_id"].set("")
        self.run_combo.configure(values=())

    def invalidate(self, *_):
        self._revision += 1
        self._lifecycle["revision"] = self._revision
        self.result = None
        self._saved_path = None
        self.save_button.configure(state="disabled")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.configure(state="disabled")
        self.details.grid_remove()
        self.status.set("Zmieniono ustawienia — ponownie sprawdź manifest. " +
                       ("FINAL_TEST_NOT_VERIFIED; wyłącznie badania eksploracyjne." if self.is_research()
                        else "Brak inwentarza final_test blokuje kalibrację."))
        self._update_guidance()

    def settings_changed(self, key="ustawienia", *_):
        if self._internal_change:
            return
        self._error = ""
        if key in {"role", "history_path", "run_id", "checkpoint_sha256"}:
            self._verified = None
            if key in {"role", "history_path"}:
                self._run_records = {}
                self.run_display.set("")
                self.run_combo.configure(values=())
            if key != "checkpoint_sha256":
                self._internal_change = True
                self.research_values["checkpoint_sha256"].set("")
                self._internal_change = False
        if key in {"role", "history_path", "run_id", "eval_inventory", "final_inventory", "mode", "checkpoint_sha256"}:
            self._source_audit = None
            self.source_text.set("Źródła i liczności wymagają ponownego sprawdzenia.")
        self.approval.set(False)
        self.invalidate()
        names = {"role": "rolę", "run_id": "trening", "history_path": "historię", "image_size": "imgsz",
                 "limit": "limit próbek", "eval_inventory": "EVAL396", "final_inventory": "final_test",
                 "checkpoint_sha256": "SHA checkpointu", "mode": "tryb", "approved_by": "operatora"}
        self.status.set(f"Zmieniono {names.get(key, key)} — wynik i fingerprint unieważnione." +
                        (" FINAL_TEST_NOT_VERIFIED." if self.is_research() else ""))
        self._update_guidance()

    def is_research(self):
        return self.mode.get() == exploratory.LABEL

    def mode_changed(self, *_):
        self.settings_changed("mode")
        if self.is_research():
            self.operator_frame.grid_remove()
            self.approval_check.grid_remove()
            self.params_frame.configure(text="Parametry kalibracji")
            self.research_frame.grid()
            if not self.context:
                self.values["image_size"].set("512" if self.values["role"].get().startswith("MT") else "320")
            self.values["limit"].set("")
            for widget in self.field_widgets["final_inventory"]:
                widget.grid_remove()
            self.final_notice.grid()
        else:
            self.operator_frame.grid()
            self.approval_check.grid()
            self.params_frame.configure(text="Parametry i zatwierdzenie")
            self.research_frame.grid_remove()
            self.final_notice.grid_remove()
            for widget in self.field_widgets["final_inventory"]:
                widget.grid()
        self._update_guidance()

    def read_checkpoint_sha(self):
        if self._busy or self._closed:
            return
        self.approval.set(False)
        self.invalidate()
        history, run_id = self.values["history_path"].get(), self.values["run_id"].get()
        role = "plate" if self.values["role"].get().startswith("MT") else "character"
        expected = self.research_values["checkpoint_sha256"].get().strip()
        context = self.context
        self._begin_job("sha", lambda: verify_selection(history, run_id, role, context=context,
                                                       expected_sha256=expected),
                        "Sprawdzam SHA wag wybranego runu — bez uruchamiania modelu…")

    def _begin_job(self, kind, operation, message):
        if self._busy or self._closed:
            return
        # Tk objects left by previously closed windows must be finalized on the
        # UI thread before a data-only worker can trigger cyclic collection.
        gc.collect()
        self._error = ""
        self._job_kind, self._busy = kind, True
        revision = self._revision
        self.prepare_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.status.set(message)
        self._update_guidance()
        events = self._events
        def worker():
            try:
                events.put((revision, operation(), None))
            except Exception as exc:
                events.put((revision, None, str(exc)))
        threading.Thread(target=worker, name="z4-calibration-" + kind, daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def check_sources(self):
        try:
            builder, config = self._configuration(audit=True)
            self._begin_job("sources", lambda: builder(**config), "Sprawdzam źródła train, EVAL396 i dopuszczalne liczności…")
        except Exception as exc:
            self._show_error(str(exc))

    def _configuration(self, *, audit=False):
        config = {k: v.get().strip() for k, v in self.values.items()}
        config["role"] = "plate" if config["role"].startswith("MT") else "character"
        config["image_size"] = int(config["image_size"])
        config["limit"] = 1 if audit else int(config["limit"])
        if audit or not self.approval.get():
            config["approved_by"] = ""
        if self.context:
            for key, value in (("role", self.context.role), ("run_id", self.context.run_id),
                               ("history_path", self.context.history_path), ("image_size", self.context.image_size)):
                if config[key] != value:
                    raise ValueError(f"Konflikt {key}: formularz={config[key]}, kontekst={value}. Przywróć wartość lub wróć do wyboru modelu.")
            for key in ("confidence_threshold", "iou_threshold"):
                if float(self.research_values[key].get()) != getattr(self.context, key):
                    raise ValueError(f"Konflikt {key} z kontekstem eksportu. Przywróć parametr lub wróć do slotu.")
        builder = build_manifest
        if self.is_research():
            builder = exploratory.build_manifest
            config.pop("final_inventory")
            config.pop("approved_by")
            config.update({k: v.get().strip() for k, v in self.research_values.items()})
            config["confidence_threshold"] = float(config["confidence_threshold"])
            config["iou_threshold"] = float(config["iou_threshold"])
        return builder, config

    def _show_error(self, message):
        self._error = message
        self.status.set("BLOCKED: " + message)
        self._update_guidance()

    def choose(self, key):
        path = filedialog.askopenfilename(parent=self, title="Wybierz dokument SHA / historię treningu",
                                         filetypes=(("JSON", "*.json"),))
        if path:
            self.values[key].set(path)
            if key == "history_path":
                self.values["run_id"].set("")
                self.run_combo.configure(values=())

    def load_runs(self):
        try:
            payload = json.loads(Path(self.values["history_path"].get()).read_text(encoding="utf-8-sig"))
            role = "plate" if self.values["role"].get().startswith("MT") else "character"
            self._run_records = {run_label(k, v, v.get("training_dataset_snapshot", {}).get(
                "training_source_hash_inventory", {}).get("role", "")): k for k, v in sorted(payload["runs"].items())
                                 if v.get("status") == "completed"}
            self.run_combo.configure(values=list(self._run_records))
            self.run_display.set("")
            self.values["run_id"].set("")
            self.status.set(f"Ukończone treningi: {len(self._run_records)}. Wskaż czytelny rekord; raw ID pozostaje tożsamością danych.")
            self._update_guidance()
        except Exception as exc:
            self._show_error(str(exc))

    def select_run(self, _event=None):
        self.values["run_id"].set(self._run_records.get(self.run_display.get(), ""))

    @staticmethod
    def _walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from Int8CalibrationDialog._walk(child)

    def _scroll(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _reveal(self, event):
        if not self.winfo_ismapped():
            return
        try:
            y = event.widget.winfo_rooty() - self.canvas.winfo_rooty()
            if y < 0 or y + event.widget.winfo_height() > self.canvas.winfo_height():
                region = self.canvas.bbox("all")
                self.canvas.yview_moveto(max(0, (self.canvas.canvasy(0) + y - 10) / max(1, region[3])))
        except tk.TclError:
            pass

    def _update_guidance(self):
        if self._closed:
            return
        role = "MT — cała scena" if self.values["role"].get().startswith("MT") else "MZ — crop tablicy"
        verified = self._verified or {}
        label = (self.context.model_label if self.context else "") or verified.get("label") or self.run_display.get() or "nazwa nieustalona"
        if len(label) > 110:
            label = label[:107] + "…"
        def short_path(value):
            return str(value) if len(str(value)) <= 100 else "…/" + "/".join(Path(value).parts[-5:])
        self.card_text.set(f"{self.context.source if self.context else 'Wybór ręczny'} | {role} | {label}\n"
            f"Run: {self.values['run_id'].get() or 'nie wybrano'} | imgsz: {self.values['image_size'].get()} | "
            f"conf: {self.research_values['confidence_threshold'].get()} | IoU: {self.research_values['iou_threshold'].get()}\n"
            f"Historia: {short_path(self.values['history_path'].get())}\n"
            f"Checkpoint: {short_path(self.context.checkpoint if self.context else verified.get('checkpoint', 'jeszcze niezweryfikowany'))}\n"
            f"SHA oczekiwany: {verified.get('expected_sha256') or (self.context.expected_sha256 if self.context else '') or 'brak historycznego dowodu'}\n"
            f"SHA zweryfikowany: {verified.get('sha256', 'niezweryfikowany')} — "
            f"{'run ↔ checkpoint zgodne' if verified else 'wymagana kontrola'}")
        self.summary_title.configure(text=f"{role}  ·  {label[:87] + '…' if len(label) > 90 else label}")
        self.summary_meta.configure(text=f"Run: {self.values['run_id'].get() or 'nie wybrano'}   ·   "
            f"imgsz: {self.values['image_size'].get()}   ·   conf: {self.research_values['confidence_threshold'].get()}   ·   "
            f"IoU: {self.research_values['iou_threshold'].get()}")
        self.identity_badge.configure(text="Run ↔ checkpoint zgodne" if verified else "SHA: wymagana kontrola")
        self.report_summary.configure(text="Manifest: " + (self.result['status'] if self.result else "nieprzygotowany"))
        key = ""
        if self._busy:
            action, title, why = "wait", "Trwa sprawdzanie — poczekaj", self.status.get()
        elif self._error:
            action, title, why = "fix", "Popraw wskazane pole", self._error
            key = next((k for k, token in (("image_size", "imgsz"), ("limit", "limit"),
                ("checkpoint_sha256", "SHA"), ("eval_inventory", "EVAL396"), ("run_id", "run"))
                if token in self._error), "history_path")
            if self.context and (key in {"history_path", "run_id"} or "kontekst" in self._error):
                action, title = "return", "Wróć do wyboru modelu / slotu"
        elif self._saved_path and self.result and self.result["status"] in {"READY", "EXPLORATORY_ONLY"}:
            action, title, why = "return", "Wróć do eksportu", "Przygotuj drugi slot albo sprawdź gotowość eksportu. Zapis manifestu nie uruchomił konwersji."
        elif self.result and self.result["status"] in {"READY", "EXPLORATORY_ONLY"}:
            action, title, why = "save", "Zapisz manifest", "Zapisz trwały dowód dla tego modelu. " + ("Przypiszemy go tylko do niezmienionego kontekstu eksportu." if self.on_saved else "Konwersja pozostaje osobnym etapem.")
        elif not self.values["run_id"].get():
            action, title, why = ("runs", "Wczytaj treningi", "Wybierz ukończony model; checkboxy Eksport nie wskazują runu kalibracji.") if not self._run_records else (
                "fix", "Wskaż model MT/MZ", "Wybierz nazwę, datę i Run z listy. Identyfikator backendu nie zmienia się.")
            key = "run_id"
        elif not self._verified:
            action, title, why = "sha", "Oblicz SHA checkpointu", "Porównujemy wagę wybranego treningu z historią; unikamy kalibracji innego modelu. Odczyt odbywa się w tle."
        elif not self.values["eval_inventory"].get() or (self._source_audit and
                self._source_audit.get("source_audit", self._source_audit)["gates"]["eval396_inventory"] != "VERIFIED"):
            action, title, why, key = "fix", "Popraw inwentarz EVAL396", "Wymagany jest rzeczywisty, zgodny dokument SHA — sam wpis ścieżki nie jest zatwierdzeniem.", "eval_inventory"
        elif not self.is_research() and (not self.values["final_inventory"].get() or (self._source_audit and
                self._source_audit.get("source_audit", self._source_audit)["gates"]["final_test_inventory"] != "VERIFIED")):
            action, title, why, key = "fix", "Wskaż zweryfikowany final_test", "Tryb ścisły nadal wymaga inwentarza. FINAL_TEST_INVENTORY_NOT_VERIFIED blokuje manifest READY.", "final_inventory"
        elif not self._source_audit:
            action, title, why = "sources", "Sprawdź źródła i liczności train", "Ustalimy bieżącą liczbę dopuszczalnych próbek bez kalibracji ani konwersji."
        elif self._source_problems():
            action, title, why, key = "fix", "Popraw źródła treningu", "; ".join(self._source_problems()), "history_path"
            if self.context:
                action, title = "return", "Wróć do wyboru modelu — źródła zablokowane"
        elif not self.values["limit"].get().isdigit() or int(self.values["limit"].get()) <= 0:
            available = self._source_audit["counts"]["eligible_unique"]
            action, title, why, key = "fix", "Ustaw maksymalną liczbę próbek", f"Dopuszczalne próbki train: {available}. Ustaw limit dla tego manifestu.", "limit"
        elif not self.is_research() and (not self.values["approved_by"].get().strip() or not self.approval.get()):
            action, title, why, key = "fix", "Wpisz operatora i zatwierdź deklaracje", "Tryb ścisły wymaga zatwierdzenia źródła train. Zgoda nie jest zaznaczana automatycznie.", "approved_by"
        else:
            action, title, why = "prepare", "Sprawdź i przygotuj manifest", "Sprawdzimy ponownie split train, powiązania crop → scena i SHA aktualnych wejść. To jeszcze nie konwersja."
        self._next_action, self._next_field = action, key
        self.next_text.set(title + "\n" + why)
        button_text = {"wait": "Sprawdzanie…", "runs": "Wczytaj treningi", "sha": "Sprawdź SHA",
            "sources": "Sprawdź źródła", "prepare": "Przygotuj manifest", "save": "Zapisz manifest",
            "return": "Wróć do eksportu" if self._saved_path else "Wróć do wyboru"}.get(action, "Przejdź do pola")
        self.next_button.configure(text=button_text, state="disabled" if action == "wait" else "normal")

    def _source_problems(self):
        return [p for p in self._source_audit.get("blockers", [])
                if p not in {"Brak jawnego zatwierdzenia źródła przez operatora.", "EXPLORATORY_OPERATOR_CONFIRMATION_REQUIRED"}
                and not (self.is_research() and p.startswith("FINAL_TEST_INVENTORY_NOT_VERIFIED"))]

    def next_action(self):
        action = self._next_action
        if action == "runs": self.load_runs()
        elif action == "sha": self.read_checkpoint_sha()
        elif action == "sources": self.check_sources()
        elif action == "prepare": self.prepare()
        elif action == "save": self.save()
        elif action == "return": self.destroy()
        elif action == "fix":
            widget = self.entries.get(self._next_field)
            if widget:
                if self._next_field in {"history_path", "run_id"} and not self.identity_frame.winfo_manager():
                    self.toggle_identity()
                    self.update_idletasks()
                widget.focus_set()
                self._reveal(type("Focus", (), {"widget": widget})())

    def prepare(self):
        if self._busy or self._closed:
            return
        self.invalidate()
        try:
            builder, config = self._configuration()
        except ValueError as exc:
            self._show_error("Popraw imgsz / limit / progi. " + str(exc))
            return
        context = self.context
        def prepare_bound():
            binding = verify_selection(config["history_path"], config["run_id"], config["role"], context=context,
                                       expected_sha256=config.get("checkpoint_sha256", ""))
            return {"binding": binding, "manifest": builder(**config)}
        self._begin_job("manifest", prepare_bound, "Kontrola pochodzenia, SHA i niezależności train…")

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            revision, result, error = self._events.get_nowait()
        except queue.Empty:
            self._poll_id = self.after(50, self._poll)
            return
        self._busy = False
        self.prepare_button.configure(state="normal")
        if revision != self._revision:
            self.status.set("Ustawienia zmieniły się podczas kontroli. Przygotuj manifest ponownie.")
            self._update_guidance()
            return
        if error:
            self._show_error(error)
            if self._job_kind == "save":
                messagebox.showerror("Nie zapisano / nie przypisano manifestu", error, parent=self)
            return
        if self._job_kind == "sha":
            self._verified = result
            self._internal_change = True
            self.research_values["checkpoint_sha256"].set(result["sha256"])
            self._internal_change = False
            self.run_display.set(result["label"])
            self.check_sources()
            return
        if self._job_kind == "save":
            self._saved_path = result
            try:
                if self.on_saved and self.result["status"] in {"READY", "EXPLORATORY_ONLY"}:
                    self.on_saved(result, self.result)
                self.status.set(f"Zapisano {self.result['status']}: {result}. Nie uruchomiono konwersji.")
            except (ValueError, RuntimeError, tk.TclError) as exc:
                self._saved_path = None
                self._show_error(f"Manifest zapisany, ale nie przypisany: {exc}")
                return
            self.save_button.configure(state="normal")
            self._update_guidance()
            return
        if self._job_kind == "manifest":
            self._verified = result["binding"]
            result = result["manifest"]
        self._source_audit = result
        audit = result.get("source_audit", result)
        counts = result["counts"]
        self.source_text.set(f"Audyt train: {counts['eligible_unique']} dopuszczalnych, "
                             f"{counts['cross_split_rejected']} odrzuconych cross-split, {counts['duplicates_removed']} duplikatów. "
                             f"EVAL396: {audit['gates']['eval396_inventory']}. To nie jest kalibracja ani wynik grafu INT8.")
        if self._job_kind == "sources":
            self.status.set(("Sprawdzono źródła; ustaw limit i przygotuj manifest. " if self.is_research()
                             else "Sprawdzono źródła; limit i zgoda pozostają decyzją operatora. ") + "; ".join(result["blockers"]))
            self._update_guidance()
            return
        self.result = result
        counts = result["counts"]
        self.status.set(f"{result['status']} | train: {counts['train_candidates']}, wybrane: {counts['selected']}, "
                        f"sceny: {counts['selected_source_scenes']}, odrzucone val/test: {counts['cross_split_rejected']}, "
                        f"duplikaty: {counts['duplicates_removed']}\n" + "; ".join(result["blockers"])
                        + "\n" + "; ".join(result.get("warnings", []))
                        + "\nFingerprint wejścia: " + result["input_fingerprint"])
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2))
        self.details.configure(state="disabled")
        self.details.grid()
        self.save_button.configure(state="normal")
        self._update_guidance()

    def save(self):
        if not self.result or self._busy:
            return
        name = filedialog.asksaveasfilename(parent=self, title="Zapisz nowy manifest kalibracji",
                                          initialfile=f"calibration_{self.result['role']}_{self.result['status']}.json",
                                          defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if not name:
            return
        payload = self.result
        revision = self._revision
        context, lifecycle = self.context, self._lifecycle
        def write_verified():
            p = Path(name).resolve()
            refs = payload["source_files"] + [payload["checkpoint"]]
            refs += [ref for exclusion in payload["exclusions"] for ref in exclusion["files"]]
            if p.suffix.lower() != ".json" or any(p.is_relative_to(Path(ref["path"]).parent) for ref in refs):
                raise ValueError("Wybierz nowy JSON poza źródłami treningu i chronionymi zbiorami.")
            rebuilt = (exploratory.rebuild_manifest(payload) if payload["schema"] in exploratory.SUPPORTED_SCHEMAS
                       else build_manifest(**payload["config"]))
            if rebuilt != payload:
                raise ValueError("Zmieniono wejścia od przygotowania manifestu; przygotuj go ponownie.")
            if context:
                verify_selection(context.history_path, context.run_id, context.role, context=context,
                                 expected_sha256=payload["checkpoint"]["sha256"])
                assert_manifest_context(payload, context)
            if lifecycle["closed"] or lifecycle["revision"] != revision:
                raise ValueError("Zamknięto okno lub zmieniono ustawienia — zapis anulowany.")
            with p.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            return p
        self._begin_job("save", write_verified, "Ponownie weryfikuję manifest przed zapisem i przypisaniem…")

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        self._lifecycle["closed"] = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        owned = [*self.values.values(), *self.research_values.values(), self.mode, self.approval,
                 self.run_display, self.card_text, self.next_text, self.source_text, self.status]
        for variable in owned:
            for modes, token in variable.trace_info():
                variable.trace_remove(modes, token)
        super().destroy()
        self.values.clear()
        self.research_values.clear()
        for name in ("mode", "approval", "run_display", "card_text", "next_text", "source_text", "status"):
            setattr(self, name, None)
        owned.clear()
        del variable
        gc.collect()


def open_int8_calibration_dialog(parent, *, context=None, on_saved=None, mode=None):
    return Int8CalibrationDialog(parent, context=context, on_saved=on_saved, mode=mode)


def calibration_request_options(variables):
    """Explicit UI mode; never infer authorization from a manifest filename."""
    def value(key, default):
        var = variables.get(key)
        return var.get() if hasattr(var, "get") else default if var is None else var
    return {"calibration_mode": value("calibration_mode", "strict")}


class CalibrationModeControl(ttk.Frame):
    """Shared by the existing single and MT+MZ export dialogs."""
    def __init__(self, parent, variables):
        super().__init__(parent)
        self.variables = variables
        if "calibration_mode" not in variables:
            variables["calibration_mode"] = tk.StringVar(self, "strict")
        self.choice = tk.StringVar(self, exploratory.LABEL if variables["calibration_mode"].get() == "exploratory" else "Ścisła — wymaga final_test")
        self.combo = ttk.Combobox(self, textvariable=self.choice, state="readonly",
                                 values=("Ścisła — wymaga final_test", exploratory.LABEL))
        self.combo.pack(fill="x")
        self.notice = ttk.Label(self, text="", wraplength=420)
        self.notice.pack(fill="x")
        self.notice.bind("<Configure>", lambda event: self.notice.configure(wraplength=max(180, event.width - 4)))
        self.choice.trace_add("write", self.changed_mode)
        self.changed_mode()

    def changed_mode(self, *_):
        research = self.choice.get() == exploratory.LABEL
        self.variables["calibration_mode"].set("exploratory" if research else "strict")
        self.notice.configure(text=("EXPLORATORY_ONLY · FINAL_TEST_NOT_VERIFIED\nWynik badawczy, nie końcowy."
                                    if research else "Wymagany zweryfikowany final_test; ścisły manifest READY."))

    def destroy(self):
        for modes, token in self.choice.trace_info():
            self.choice.trace_remove(modes, token)
        super().destroy()
