"""Session → subject → evidence review. File IO and metrics run off the Tk thread."""
from __future__ import annotations

import json
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from ..ranking.mobile_human_review import MobileReviewSession, align_plate_text, normalize_registration
from ..ranking.mobile_mt_invocations import DETECTION_STATUSES, detection_box_on_mt_input
from ..ranking.mobile_package_experiments import MobilePackageExperimentStore, read_mobile_report_bundle

REVIEW_LABELS = {"NOT_STARTED": "Nie rozpoczęto", "IN_PROGRESS": "W toku", "COMPLETED": "Zakończona"}
MODE_LABELS = {"blinded_gt_v1": "Zaślepiona GT", "assisted": "Asystowana (diagnostyczna)"}
VISIBILITY_LABELS = {"none": "Brak widocznej tablicy", "one": "Jedna widoczna tablica",
                     "multiple": "Wiele widocznych tablic", "uncertain": "Niejednoznaczne"}
HIDDEN_PREDICTION = "ukryta do czasu GT"
STAGE_LABELS = {"VALID_QUAD": "Poprawny obszar", "NO_DETECTION": "Brak detekcji", "DETECTION_INVALID_QUAD": "Błędna geometria",
                "NOT_RUN": "Nie uruchomiono", "NO_CHARACTERS": "Brak znaków", "READ": "Odczyt"}


def _display(value, *, rate=False):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Tak" if value else "Nie"
    if rate:
        return f"{value * 100:.2f}%"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


class MobileSampleReviewWindow:
    def __init__(self, browser, report, bundle=None):
        self.browser, self.report = browser, report
        self.session = None
        self.stats = {}
        self.subject_key = ""
        self.current_record = None
        self.current_invocation = ""
        self._subject_ids = {}
        self._record_ids = {}
        self._closed = False
        self._saving = False
        self._close_requested = False
        self._pending_subject = ""
        self._rendering = False
        self._autosave_id = None
        self._image_token = 0
        self._image = None
        self._image_source_size = None
        self._image_entry = ""
        self._photo = None
        self._results = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mobile-review")
        palette = browser.palette
        self.bg = palette.get("panel", "#162013")
        self.fg = palette.get("fg", "#e9efdf")
        self.muted = palette.get("muted", self.fg)
        self.window = tk.Toplevel(browser.window)
        self.window._aat_skip_window_recovery = True
        self.window.title("Weryfikacja próbek z telefonu")
        width = min(1380, self.window.winfo_screenwidth() - 80)
        height = min(900, self.window.winfo_screenheight() - 100)
        self.window.geometry(f"{width}x{height}+35+35")
        self.window.minsize(1000, 690)
        self.window.configure(bg=self.bg)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Destroy>", self._destroyed, add="+")
        self.actions = []
        self._build()
        self._poll_id = self.window.after(40, self._poll)
        source = Path(bundle.path if bundle and bundle.source_archive_sha256 == report.source_archive_sha256 else report.source_path)
        expected_hash = report.source_archive_sha256
        def load():
            # Revalidate on opening; the stored report may outlive its archive.
            fresh = read_mobile_report_bundle(source)
            if expected_hash and fresh.source_archive_sha256 != expected_hash:
                raise ValueError("Plik źródłowy nie odpowiada SHA-256 zapisanego raportu. Zaimportuj właściwą sesję.")
            session = MobileReviewSession(fresh)
            store = MobilePackageExperimentStore(browser.store.root_dir)
            store.apply_human_review(session)
            return session, session.statistics()
        self._submit(load, self._loaded)

    def _label(self, parent, text="", **kwargs):
        return tk.Label(parent, text=text, bg=self.bg, fg=self.fg, anchor="w", justify="left", **kwargs)

    def _button(self, parent, text, command, **grid):
        button = ttk.Button(parent, text=text, command=command, state="disabled")
        button.grid(**grid, padx=3, pady=3)
        self.actions.append(button)
        return button

    def _table(self, parent, columns, widths, *, height=8):
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height, selectmode="browse")
        for name, width in zip(columns, widths):
            tree.heading(name, text=name)
            tree.column(name, width=width, minwidth=45, stretch=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        return frame, tree

    def _build(self):
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        header = tk.Frame(self.window, bg=self.bg, padx=12, pady=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.session_label = self._label(header, "Wczytywanie sesji…", font=("Segoe UI", 12, "bold"))
        self.session_label.grid(row=0, column=0, sticky="ew")
        self.progress_label = self._label(header, font=("Segoe UI", 9))
        self.progress_label.grid(row=1, column=0, sticky="ew")
        self._button(header, "Eksport wyników", self.export, row=0, column=1)
        self._button(header, "Zakończ weryfikację", self.complete, row=0, column=2)
        self.notice_label = self._label(self.window, wraplength=1150, font=("Segoe UI", 9))
        self.notice_label.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        tabs = ttk.Notebook(self.window)
        tabs.grid(row=2, column=0, sticky="nsew", padx=10)
        review = tk.Frame(tabs, bg=self.bg)
        review.columnconfigure(1, weight=1)
        review.rowconfigure(0, weight=1)
        tabs.add(review, text="Weryfikacja")
        left = tk.Frame(review, bg=self.bg)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        self._label(left, "Tablice", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=7)
        frame, self.subject_tree = self._table(left, ("Nr", "Stan", "GT"), (40, 85, 100), height=18)
        frame.grid(row=1, column=0, sticky="nsew")
        self.subject_tree.bind("<<TreeviewSelect>>", self._select_subject)
        self._button(left, "Następna nieoceniona", self.next_pending, row=2, column=0, sticky="ew")
        right = tk.Frame(review, bg=self.bg)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1, minsize=120)
        self.subject_label = self._label(right, font=("Segoe UI", 10, "bold"))
        self.subject_label.grid(row=0, column=0, sticky="ew", pady=(6, 2))
        gt = tk.Frame(right, bg=self.bg)
        gt.grid(row=1, column=0, sticky="ew")
        gt.columnconfigure(1, weight=1)
        self._label(gt, "GT").grid(row=0, column=0, padx=(0, 6))
        self.gt_var = tk.StringVar()
        self.gt_entry = ttk.Entry(gt, textvariable=self.gt_var, font=("Segoe UI", 14), width=15)
        self.gt_entry.grid(row=0, column=1, sticky="ew")
        self.gt_entry.bind("<Return>", lambda event: self.save_gt(explicit=True))
        self._button(gt, "Zapisz GT", lambda: self.save_gt(explicit=True), row=0, column=2)
        self.accept_button = self._button(gt, "Zgodne z predykcją", self.accept_prediction, row=0, column=3)
        self.accept_button.grid_remove()
        self._button(gt, "Nie do oceny", self.exclude_subject, row=0, column=4)
        self.note_var = tk.StringVar()
        self._label(gt, "Notatka").grid(row=1, column=0, sticky="w")
        self.note_entry = ttk.Entry(gt, textvariable=self.note_var)
        self.note_entry.grid(row=1, column=1, columnspan=4, sticky="ew", pady=4)
        self.gt_var.trace_add("write", self._schedule_autosave)
        self.note_var.trace_add("write", self._schedule_autosave)
        image_bar = tk.Frame(right, bg=self.bg)
        image_bar.grid(row=2, column=0, sticky="ew")
        image_bar.columnconfigure(0, weight=1)
        self.record_label = self._label(image_bar, font=("Segoe UI", 9))
        self.record_label.grid(row=0, column=0, sticky="ew")
        self.image_choice = ttk.Combobox(image_bar, state="readonly", width=24)
        self.image_choice.grid(row=0, column=1, padx=4)
        self.image_choice.bind("<<ComboboxSelected>>", lambda event: self._load_image())
        self.canvas = tk.Canvas(right, height=210, bg="#101820", highlightthickness=0)
        self.canvas.grid(row=3, column=0, sticky="nsew", pady=5)
        self.canvas.bind("<Configure>", lambda event: self._paint_image())
        comparison = tk.Frame(right, bg=self.bg)
        comparison.grid(row=4, column=0, sticky="ew", pady=4)
        comparison.columnconfigure(0, weight=1)
        self.alignment_label = self._label(comparison, font=("Segoe UI", 11), wraplength=900)
        self.alignment_label.grid(row=0, column=0, sticky="ew")
        self.alignment_text = tk.Text(comparison, height=2, font=("Consolas", 11), bg=self.bg, fg=self.fg,
                                      bd=0, highlightthickness=0, wrap="none", state="disabled", takefocus=False)
        self.alignment_text.grid(row=1, column=0, sticky="ew")
        self.alignment_text.tag_configure("incorrect", foreground=self.browser.error)
        self.alignment_text.tag_configure("missing", foreground=self.browser.warning)
        self.alignment_text.tag_configure("extra", foreground=self.browser.warning)
        controls = tk.Frame(right, bg=self.bg)
        controls.grid(row=5, column=0, sticky="ew")
        invocation_bar = tk.Frame(controls, bg=self.bg)
        invocation_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        invocation_bar.columnconfigure(0, weight=1)
        self.invocation_label = self._label(invocation_bar, "Wywołanie MT: —", font=("Segoe UI", 9))
        self.invocation_label.grid(row=0, column=0, sticky="w")
        self._button(invocation_bar, "Pokaż wejście MT", self.show_invocation_input, row=0, column=1)
        self.invocation_choice = ttk.Combobox(invocation_bar, state="disabled", width=25, values=list(VISIBILITY_LABELS.values()))
        self.invocation_choice.grid(row=0, column=2, padx=3)
        self.invocation_choice.bind("<<ComboboxSelected>>", self.save_invocation_decision)
        self.invocation_error_label = self._label(invocation_bar, font=("Segoe UI", 9), wraplength=800)
        self.invocation_error_label.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.invocation_error_label.grid_remove()
        for column, (title, decision) in enumerate((
            ("To jest tablica", {"plate_visibility": "visible", "is_plate": True, "evaluable": True}),
            ("To nie jest tablica", {"is_plate": False, "evaluable": False}),
            ("Pomiń detekcję / crop", {"evaluable": False}),
        )):
            self._button(controls, title, lambda decision=decision: self.annotate(decision),
                         row=1, column=column, sticky="ew")
            controls.columnconfigure(column, weight=1)
        frame, self.record_tree = self._table(right, ("Rodzaj", "Id", "MT / MZ", "Predykcja", "Ocena"), (65, 130, 140, 130, 120), height=4)
        frame.grid(row=6, column=0, sticky="nsew", pady=6)
        self.record_tree.bind("<<TreeviewSelect>>", self._select_record)
        summary = tk.Frame(tabs, bg=self.bg)
        summary.rowconfigure(0, weight=1)
        summary.columnconfigure(0, weight=1)
        tabs.add(summary, text="Statystyki")
        frame, self.stats_tree = self._table(summary, ("Zakres", "Miara", "Wynik"), (150, 480, 150), height=20)
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        details = tk.Frame(tabs, bg=self.bg)
        details.rowconfigure(1, weight=1)
        details.columnconfigure(0, weight=1)
        tabs.add(details, text="Sesja i modele")
        reviewer = tk.Frame(details, bg=self.bg)
        reviewer.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self._label(reviewer, "Osoba weryfikująca (opcjonalny identyfikator)").grid(row=0, column=0, padx=5)
        self.reviewer_var = tk.StringVar()
        ttk.Entry(reviewer, textvariable=self.reviewer_var).grid(row=0, column=1)
        self._button(reviewer, "Zapisz", self.save_reviewer, row=0, column=2)
        self._label(reviewer, "Tryb weryfikacji").grid(row=1, column=0, padx=5, pady=5)
        self.mode_choice = ttk.Combobox(reviewer, state="disabled", width=29, values=list(MODE_LABELS.values()))
        self.mode_choice.grid(row=1, column=1, columnspan=2, sticky="w")
        self.mode_choice.bind("<<ComboboxSelected>>", self.change_review_mode)
        frame, self.details_tree = self._table(details, ("Pole", "Wartość"), (340, 750), height=20)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.status_var = tk.StringVar(value="Odczyt pełnego indeksu i kontrola SHA-256…")
        self._label(self.window, textvariable=self.status_var, font=("Segoe UI", 9), wraplength=1250).grid(
            row=3, column=0, sticky="ew", padx=12, pady=8)

    def _submit(self, work, done, *, saving=False):
        if self._closed:
            return
        if saving:
            self._saving = True
            self.gt_entry.configure(state="disabled")
            self.note_entry.configure(state="disabled")
            self.mode_choice.configure(state="disabled")
            self.invocation_choice.configure(state="disabled")
            for button in self.actions:
                button.configure(state="disabled")
        def task():
            try:
                self._results.put((done, work(), None, saving))
            except Exception as error:
                self._results.put((done, None, str(error), saving))
        self._executor.submit(task)

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                done, result, error, saving = self._results.get_nowait()
                if saving:
                    self._saving = False
                    self.gt_entry.configure(state="normal")
                    self.note_entry.configure(state="normal")
                    for button in self.actions:
                        button.configure(state="normal")
                if error:
                    self.status_var.set(error)
                    self._close_requested = False
                else:
                    done(result)
                    if saving and self._close_requested:
                        self.close()
                        return
                if self.session:
                    self._sync_review_controls()
        except queue.Empty:
            pass
        self._poll_id = self.window.after(40, self._poll)

    def _loaded(self, result):
        self.session, self.stats = result
        for button in self.actions:
            button.configure(state="normal")
        session = self.session
        self.reviewer_var.set(session.review.reviewer_id)
        self.session_label.configure(text=f"Sesja: {session.session_id}")
        context = dict(session.bundle.metadata, **session.bundle.report_payload)
        context.update(session.bundle.collection_session or context.get("research_collection", {}))
        state = str(context.get("state") or context.get("session_state") or context.get("status") or
                    session.bundle.experiment_session.get("status") or "").upper()
        notices = list(session.warnings)
        if state in {"PARTIAL", "ERROR"}:
            notices.insert(0, f"Sesja niekompletna: {state}. Wyniki obejmują wyłącznie zapisane dowody.")
        elif session.bundle.manifest.get("collection_complete") is False:
            notices.insert(0, "Sesja niekompletna. Wyniki obejmują wyłącznie zapisane dowody.")
        self.notice_label.configure(text="\n".join(notices), fg=self.browser.warning if notices else self.fg)
        self._subject_ids = {f"subject-{index}": key for index, key in enumerate(session.subjects)}
        self._refresh_subject_rows()
        self._refresh_stats()
        self._populate_details()
        self._sync_review_controls()
        self.status_var.set("Gotowe. Zmiany zapisują się automatycznie. Lokalizacja zapisu: zakładka „Sesja i modele”.")
        if self._subject_ids:
            first = next(iter(self._subject_ids))
            self.subject_tree.selection_set(first)
            self.subject_tree.focus(first)

    def _refresh_subject_rows(self):
        self._subject_stats = {row["subject_key"]: row for row in self.stats["subjects"]}
        for index, (iid, key) in enumerate(self._subject_ids.items(), 1):
            decision = self.session.review.subjects.get(key, {})
            state = "Pominięta" if decision.get("evaluable") is False else "Gotowa" if decision.get("ground_truth") else "Oczekuje"
            if decision.get("draft_ground_truth") and not self.session.predictions_visible(key):
                state = "Szkic GT"
            pending = self._subject_stats.get(key, {}).get("pending_decisions", 0)
            if pending and decision.get("ground_truth"):
                state = f"Decyzje: {pending}"
            values = (index, state, self.session.subject_draft(key))
            if self.subject_tree.exists(iid):
                self.subject_tree.item(iid, values=values)
            else:
                self.subject_tree.insert("", "end", iid=iid, values=values)

    def _select_subject(self, event=None):
        selected = self.subject_tree.selection()
        key = self._subject_ids.get(selected[0]) if selected else None
        if not key or key == self.subject_key:
            return
        if self._saving:
            self._pending_subject = key
            return
        self.save_gt()
        if self._saving:
            self._pending_subject = key
            return
        self.subject_key = key
        self._render_subject()

    def _render_subject(self):
        key = self.subject_key
        if not key:
            return
        subject = self.session.subjects[key]
        decision = self.session.review.subjects.get(key, {})
        self._rendering = True
        self.gt_var.set(self.session.subject_draft(key))
        self.note_var.set(decision.get("note", ""))
        self._rendering = False
        number = list(self.session.subjects).index(key) + 1
        legacy = " • tożsamość legacy" if subject["legacy_identity"] else ""
        stats = self._subject_stats.get(key, {})
        invocations = {self.session.invocation_by_attempt[item] for item in subject["attempts"]}
        invocation_count = sum(self.session.mt_invocations[item].executed for item in invocations) if self.session.attempts_available else None
        self.subject_label.configure(text=f"Tablica {number} / {len(self.session.subjects)} • wywołania MT: {_display(invocation_count)} • cropy: {len(subject['samples'])} • "
            f"MT miss: {_display(stats.get('mt_no_detection_invocations'))} • odczyty MZ: {stats.get('evaluable_reads', 0)}{legacy}")
        visible = self.session.predictions_visible(key)
        self._sync_review_controls()
        if not visible:
            self.alignment_label.configure(text="Wpisz GT z obrazu i wybierz „Zapisz GT”.")
            self.alignment_text.configure(state="normal")
            self.alignment_text.delete("1.0", "end")
            self.alignment_text.configure(state="disabled")
        old = self.current_record
        self.record_tree.delete(*self.record_tree.get_children())
        self._record_ids = {}
        for kind, keys, records in (("attempt", subject["attempts"], self.session.attempts), ("sample", subject["samples"], self.session.samples)):
            for record_id in keys:
                row = records[record_id]
                iid = f"record-{len(self._record_ids)}"
                self._record_ids[iid] = (kind, record_id)
                annotation = (self.session.review.attempt_annotations if kind == "attempt" else self.session.review.sample_annotations).get(record_id, {})
                state = "Nie tablica" if annotation.get("is_plate") is False else "Pominięta" if annotation.get("evaluable") is False else annotation.get("plate_visibility", "GT zapisane" if kind == "sample" and decision.get("ground_truth") else "Do oceny")
                state = {"visible": "Widoczna", "invisible": "Niewidoczna", "uncertain": "Niejednoznaczna"}.get(state, state)
                if self.session.cancelled(row):
                    state = "Anulowana"
                invocation = self.session.invocation_by_attempt.get(record_id if kind == "attempt" else row.get("attempt_id"), "")
                group = self.session.mt_invocations.get(invocation)
                if group and group.execution_failed and not group.cancelled:
                    state = "Błąd wykonania MT"
                index = row.get("mt_detection_index")
                label = f"Detekcja {int(index) + 1}" if kind == "attempt" and index not in (None, "") else "Rekord MT" if kind == "attempt" else "Crop"
                identity = f"{invocation} / {record_id}" if invocation and invocation != record_id else record_id
                self.record_tree.insert("", "end", iid=iid, values=(label, identity,
                    self._record_stages(row, group), row.get("prediction", "") if visible else HIDDEN_PREDICTION, state))
        if self._record_ids:
            iid = next((iid for iid, value in self._record_ids.items() if value == old), next(iter(self._record_ids)))
            self.current_record = None
            self.record_tree.selection_set(iid)
            self.record_tree.focus(iid)
        else:
            self.current_record = None
            self._image = None
            self._paint_image()

    def _record_stages(self, row, group):
        mt = ("Anulowane MT" if group and group.cancelled else
              "Błąd wykonania MT" if group and group.execution_failed else STAGE_LABELS.get(row.get("mt_status"), ""))
        # MZ status itself can disclose a no-read before GT, even with OCR hidden.
        mz = STAGE_LABELS.get(row.get("mz_status"), "") if self.session.predictions_visible(self.subject_key) else ""
        return " / ".join(stage for stage in (mt, mz) if stage)

    def _select_record(self, event=None):
        selected = self.record_tree.selection()
        record = self._record_ids.get(selected[0]) if selected else None
        if not record or record == self.current_record:
            return
        self.current_record = record
        kind, key = record
        row = (self.session.attempts if kind == "attempt" else self.session.samples)[key]
        self._image_rows = []
        attempt = row if kind == "attempt" else self.session.attempts.get(row.get("attempt_id"), {})
        self.current_invocation = self.session.invocation_by_attempt.get(attempt.get("id"), "")
        if self.current_invocation:
            group = self.session.mt_invocations[self.current_invocation]
            preferred = attempt.get("mt_input_evidence_entry")
            for index, entry in enumerate(sorted(group.evidence_entries,
                    key=lambda name: (name not in self.session.entry_names, name != preferred))):
                self._image_rows.append((f"Wejście MT {index + 1}", {"image_entry": entry}))
        if attempt.get("evidence_entry"):
            self._image_rows.append(("Dowód próby", attempt))
        if kind == "sample" and row.get("image_entry"):
            self._image_rows.insert(0, ("Crop tablicy", row))
        elif kind == "attempt":
            for sample_id in self.session.subjects[self.subject_key]["samples"]:
                sample = self.session.samples[sample_id]
                if sample.get("attempt_id") == key and sample.get("image_entry"):
                    self._image_rows.append((f"Crop: {sample_id}", sample))
        self.image_choice.configure(values=[title for title, _ in self._image_rows])
        if self._image_rows:
            self.image_choice.current(0)
        else:
            self.image_choice.set("Brak zapisanego obrazu")
        label = "Rekord MT" if kind == "attempt" else "Crop"
        self.record_label.configure(text=f"{label}: {key} • {self._record_stages(row, self.session.mt_invocations.get(self.current_invocation))}")
        gt = self.session.review.subjects.get(self.subject_key, {}).get("ground_truth", "")
        prediction = str(row.get("prediction", ""))
        self.alignment_text.configure(state="normal")
        self.alignment_text.delete("1.0", "end")
        if not self.session.predictions_visible(self.subject_key):
            self.alignment_label.configure(text="Wpisz GT z obrazu i wybierz „Zapisz GT”. Predykcje i porównanie pozostają ukryte; autosave chroni szkic.")
        elif self.session.cancelled(row):
            self.alignment_label.configure(text=f"Próba anulowana — poza metrykami. Zapisana predykcja: {prediction or '—'}")
        elif str(row.get("mz_status", "")).upper() == "NOT_RUN":
            self.alignment_label.configure(text="MZ nie został uruchomiony w tej próbie. Oceń widoczność tablicy dla MT.")
        elif gt:
            match = align_plate_text(gt, prediction)
            outcome = "Brak odczytu" if not match.prediction else "Poprawny odczyt" if match.exact_match else "Niepoprawny odczyt"
            self.alignment_label.configure(text=f"{outcome} • predykcja: {prediction or '—'}\n"
                f"Poprawnie rozpoznane: {match.correct_characters} • błędnie rozpoznane: {match.incorrect_characters} • "
                f"brakujące: {match.missing_characters} • nadmiarowe: {match.extra_characters} • CER: {_display(match.cer, rate=True)}")
            for prefix, field in (("GT    : ", "ground_truth"), ("Odczyt: ", "prediction")):
                self.alignment_text.insert("end", prefix)
                for item in match.alignment:
                    self.alignment_text.insert("end", (item[field] or "·") + " ", item["kind"])
                self.alignment_text.insert("end", "\n")
        else:
            self.alignment_label.configure(text=f"Predykcja: {prediction or 'brak odczytu'} • wpisz GT dla tej tablicy.")
        self.alignment_text.configure(state="disabled")
        self._sync_review_controls()
        self._load_image()

    def _sync_review_controls(self):
        if not self.session:
            return
        if self.subject_key and self.session.predictions_visible(self.subject_key):
            self.accept_button.grid()
            self.accept_button.configure(state="disabled" if self._saving else "normal")
        else:
            self.accept_button.configure(state="disabled")
            self.accept_button.grid_remove()
        self.mode_choice.set(MODE_LABELS[self.session.review.review_mode])
        self.mode_choice.configure(state="disabled" if self._saving or self.session.review.review_mode_locked else "readonly")
        self.invocation_error_label.grid_remove()
        if self.current_invocation:
            group = self.session.mt_invocations[self.current_invocation]
            state = "anulowane" if group.cancelled else "nie uruchomiono" if not group.executed else "błąd wykonania" if group.execution_failed else f"detekcje: {group.detection_count}"
            self.invocation_label.configure(text=f"Wywołanie MT: {self.current_invocation} • {state}")
            if group.execution_failed:
                error = " ".join("; ".join(group.execution_errors).split())
                error = error[:157] + "…" if len(error) > 160 else error
                self.invocation_error_label.configure(text=f"{error}\nWywołanie technicznie nieukończone — poza jakością MT; nie wymaga oceny. Szczegóły: Sesja i modele.", fg=self.browser.warning)
                self.invocation_error_label.grid()
            annotation = self.session.invocation_annotation(self.current_invocation)
            self.invocation_choice.set("Brak dawnej oceny liczby tablic" if annotation.get("decision_source") == "legacy_cardinality_unknown"
                                       else VISIBILITY_LABELS.get(annotation.get("visible_plate_count"), "Oceń całe wejście MT"))
            if group.execution_failed:
                self.invocation_choice.set("Nie wymaga oceny MT")
            self.invocation_choice.configure(state="disabled" if self._saving or group.cancelled or not group.executed or group.execution_failed else "readonly")
        else:
            self.invocation_label.configure(text="Wywołanie MT: niedostępne dla tego cropa")
            self.invocation_choice.set("")
            self.invocation_choice.configure(state="disabled")

    def show_invocation_input(self):
        for index, (label, _) in enumerate(getattr(self, "_image_rows", [])):
            if label.startswith("Wejście MT"):
                self.image_choice.current(index)
                self._load_image()
                return
        self.status_var.set("Brak dowodu całego wejścia MT. Oznacz wywołanie jako niejednoznaczne.")

    def save_invocation_decision(self, event=None):
        key = self.current_invocation
        group = self.session.mt_invocations.get(key) if self.session else None
        if not group or group.cancelled or not group.executed or group.execution_failed:
            self._sync_review_controls()
            return
        value = next((key for key, label in VISIBILITY_LABELS.items() if label == self.invocation_choice.get()), None)
        if key and value:
            self._change(lambda: self.session.annotate_invocation(key, visible_plate_count=value, evaluable=value != "uncertain"))

    def change_review_mode(self, event=None):
        mode = next((key for key, label in MODE_LABELS.items() if label == self.mode_choice.get()), None)
        if mode:
            key, text, note = self.subject_key, self.gt_var.get(), self.note_var.get()
            def change():
                self.session.set_review_mode(mode)
                if key and (text or note):
                    self.session.save_subject_draft(key, ground_truth=text, note=note)
            self._change(change, save_draft=False)

    def _load_image(self):
        self._image_token += 1
        token = self._image_token
        self._image = None
        self._image_source_size = None
        self._image_entry = ""
        self._paint_image("Wczytywanie obrazu…")
        index = self.image_choice.current()
        if not self._image_rows or index < 0:
            self._paint_image("Brak obrazu dowodowego. Możesz oznaczyć próbę jako niejednoznaczną lub pominąć.")
            return
        row = self._image_rows[index][1]
        def work():
            try:
                image = self.session.image(row)
                source_size = image.size
                image.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
                return image, source_size, ""
            except Exception as error:
                return None, None, str(error)
        def done(result):
            if token == self._image_token:
                self._image, self._image_source_size, error = result
                self._image_entry = row.get("image_entry") or row.get("evidence_entry") or ""
                self._paint_image(error)
        self._submit(work, done)

    def _paint_image(self, message=""):
        self.canvas.delete("all")
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self._image is None:
            self._photo = None
            self.canvas.create_text(width / 2, height / 2, text=message or "Wybierz próbę lub crop", fill="#edf4f6", width=max(50, width - 30))
            return
        scale = min(max(1, width - 16) / self._image.width, max(1, height - 16) / self._image.height, 4.0)
        image = self._image.resize((max(1, round(self._image.width * scale)), max(1, round(self._image.height * scale))), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.create_image(width / 2, height / 2, image=self._photo, tags=("evidence_image",))
        self._paint_detection_overlay(image.width, image.height, (width - image.width) / 2, (height - image.height) / 2)

    def _paint_detection_overlay(self, width, height, offset_x, offset_y):
        if not self.current_record or not self.current_invocation:
            return
        kind, key = self.current_record
        row = self.session.attempts.get(key if kind == "attempt" else self.session.samples[key].get("attempt_id"), {})
        group = self.session.mt_invocations[self.current_invocation]
        explicit_inputs = {child.get("mt_input_evidence_entry") for child in group.records}
        # Legacy evidence can be inspected, but never treated as letterbox geometry.
        if (not self._image_entry or self._image_entry not in group.evidence_entries
                or self._image_entry.startswith("samples/crops/") or row.get("mt_status") not in DETECTION_STATUSES):
            return
        box = detection_box_on_mt_input(row) if self._image_entry in explicit_inputs else None
        message = "Brak geometrii detekcji w tej sesji."
        if box is not None and self._image_source_size != (float(row["input_width"]), float(row["input_height"])):
            box = None
            message = "Rozmiar obrazu nie odpowiada geometrii wejścia MT."
        if box is None:
            text_id = self.canvas.create_text(12, 12, anchor="nw", text=message, fill="#ffffff", width=max(50, self.canvas.winfo_width() - 24), tags=("detection_geometry_notice",))
            self._canvas_text_background(text_id, "detection_geometry_notice")
            return
        scale_x, scale_y = width / float(row["input_width"]), height / float(row["input_height"])
        left, top, right, bottom = box
        coords = (offset_x + left * scale_x, offset_y + top * scale_y, offset_x + right * scale_x, offset_y + bottom * scale_y)
        # A dark outer stroke keeps the cyan box readable on bright evidence too.
        self.canvas.create_rectangle(*coords, outline="#10212b", width=6, tags=("detection_overlay",))
        self.canvas.create_rectangle(*coords, outline="#40dcff", width=3, tags=("detection_overlay", "detection_box"))
        ordinal = row.get("mt_detection_index")
        title = f"Detekcja {int(ordinal) + 1} / {group.detection_count}" if ordinal not in (None, "") else "Detekcja"
        if row.get("mt_status") == "DETECTION_INVALID_QUAD":
            title += " • błędna geometria"
        text_id = self.canvas.create_text(coords[0] + 5, max(offset_y + 3, coords[1] - 24),
            anchor="nw", text=title, fill="#ffffff", font=("Segoe UI", 10, "bold"), tags=("detection_overlay", "detection_label"))
        bounds = self.canvas.bbox(text_id)
        self.canvas.move(text_id, min(0, self.canvas.winfo_width() - 8 - bounds[2]), 0)
        self._canvas_text_background(text_id, "detection_overlay")

    def _canvas_text_background(self, text_id, tag):
        bounds = self.canvas.bbox(text_id)
        background = self.canvas.create_rectangle(bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2,
            fill="#10212b", outline="", tags=(tag,))
        self.canvas.tag_lower(background, text_id)

    def _schedule_autosave(self, *args):
        if self._rendering or not self.subject_key:
            return
        self.mode_choice.configure(state="disabled")
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
        self._autosave_id = self.window.after(750, self.save_gt)

    def _draft_operation(self):
        key, gt, note = self.subject_key, self.gt_var.get(), self.note_var.get()
        def save():
            if key:
                old = self.session.review.subjects.get(key, {})
                if normalize_registration(gt) != normalize_registration(self.session.subject_draft(key)) or note != old.get("note", ""):
                    self.session.save_subject_draft(key, ground_truth=gt, note=note)
        return save

    def _change(self, operation, *, save_draft=True, after=None):
        if not self.session or self._saving:
            return
        self.status_var.set("Zapisuję weryfikację…")
        draft = self._draft_operation()
        def work():
            if save_draft:
                draft()
            operation()
            stats = self.session.statistics()
            warning = ""
            updated_report = None
            try:
                store = MobilePackageExperimentStore(self.browser.store.root_dir)
                if store.apply_human_review(self.session):
                    updated_report = next((report for report in store.reports if report.report_id == self.report.report_id
                                           and report.source_archive_sha256 == self.session.review.source_archive_sha256), None)
            except Exception as error:
                warning = f"Decyzje zapisane; ranking wymaga odświeżenia: {error}"
            return stats, warning, updated_report
        def done(result):
            self._changed(result)
            if after:
                after()
        self._submit(work, done, saving=True)

    def _changed(self, result):
        self.stats, warning, updated_report = result
        self._refresh_subject_rows()
        self._refresh_stats()
        if self._pending_subject:
            self.subject_key, self._pending_subject = self._pending_subject, ""
        self._render_subject()
        self.status_var.set(warning or f"Zapisano automatycznie • rewizja {self.session.review.review_revision} • {REVIEW_LABELS[self.session.review.review_status]}")
        if updated_report is not None and self.browser.window.winfo_exists():
            self.browser.store.add_report(updated_report, save=False)
            for iid, report in self.browser.report_by_iid.items():
                if report.report_id == updated_report.report_id and report.source_archive_sha256 == updated_report.source_archive_sha256:
                    self.browser.report_by_iid[iid] = updated_report
            if (getattr(self.browser, "current_report", None) and self.browser.current_report.report_id == updated_report.report_id
                    and self.browser.current_report.source_archive_sha256 == updated_report.source_archive_sha256):
                self.browser._show_report(updated_report, self.browser.current_bundle)

    def save_gt(self, *, explicit=False):
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
            self._autosave_id = None
        if not self.session or not self.subject_key:
            return
        if self._saving:
            self._autosave_id = self.window.after(200, self.save_gt)
            return
        key, gt, note = self.subject_key, self.gt_var.get(), self.note_var.get()
        old = self.session.review.subjects.get(key, {})
        normalized = normalize_registration(gt)
        needs_confirmation = explicit and normalized and (not self.session.predictions_visible(key) or old.get("evaluable") is not True)
        if normalize_registration(self.session.subject_draft(key)) == normalized and old.get("note", "") == note and not needs_confirmation:
            return
        if explicit:
            self._change(lambda: self.session.set_subject(key, ground_truth=gt, note=note, evaluable=True if normalized else None), save_draft=False)
        else:
            self._change(lambda: self.session.save_subject_draft(key, ground_truth=gt, note=note), save_draft=False)

    def accept_prediction(self):
        if not self.session.predictions_visible(self.subject_key):
            self.status_var.set("Najpierw zapisz GT odczytane z obrazu.")
            return
        if self.current_record:
            kind, key = self.current_record
            row = (self.session.attempts if kind == "attempt" else self.session.samples)[key]
            if not normalize_registration(row.get("prediction")):
                self.status_var.set("Ta próba nie zawiera predykcji do przepisania jako GT.")
                return
            self.gt_var.set(row["prediction"])
            self.save_gt(explicit=True)

    def exclude_subject(self):
        key = self.subject_key
        if key:
            gt, note = self.gt_var.get(), self.note_var.get()
            self._change(lambda: self.session.set_subject(key, ground_truth=gt, evaluable=False, note=note), save_draft=False)

    def annotate(self, decision):
        if self.current_record:
            kind, key = self.current_record
            self._change(lambda: self.session.annotate(kind, key, **decision))

    def save_reviewer(self):
        value = self.reviewer_var.get()
        self._change(lambda: self.session.set_reviewer(value))

    def complete(self):
        self._change(self.session.complete)

    def next_pending(self):
        for iid, key in self._subject_ids.items():
            if self._subject_stats.get(key, {}).get("pending_decisions", 0):
                self.subject_tree.selection_set(iid)
                self.subject_tree.see(iid)
                return
        self.status_var.set("Wszystkie tablice mają decyzję. Sprawdź jeszcze widoczność w poszczególnych próbach.")

    def export(self):
        directory = filedialog.askdirectory(parent=self.window, title="Katalog wyników weryfikacji")
        if directory:
            paths = []
            self._change(lambda: paths.extend(self.session.export(Path(directory))),
                         after=lambda: self.status_var.set(f"Wyeksportowano {len(paths)} pliki do {directory}"))

    def _refresh_stats(self):
        summary = self.stats["summary"]
        self.progress_label.configure(text=f"Tablice: {summary['subject_count']} • wywołania MT: {_display(summary['mt_invocation_count'])} • cropy: {summary['crop_count']} • "
            f"ocenione: {summary['reviewed_subjects']} • oczekujące: {summary['not_reviewed_subjects']} • {REVIEW_LABELS[self.session.review.review_status]} • {MODE_LABELS[self.session.review.review_mode]}")
        self.stats_tree.delete(*self.stats_tree.get_children())
        sections = {
            "Tablice / ALPR": {"evaluable_subjects": "Tablice do oceny", "subjects_with_exact_read": "Z poprawnym odczytem", "subjects_without_exact_read": "Bez poprawnego odczytu", "subject_success_rate": "Skuteczność systemu dla tablic"},
            "Odczyty MZ": {"evaluable_reads": "Oceniane cropy", "exact_reads": "Poprawne odczyty", "incorrect_reads": "Niepoprawne odczyty", "no_reads": "Brak odczytu", "exact_read_rate": "Udział poprawnych odczytów", "no_read_rate": "Udział braku odczytu"},
            "Znaki MZ": {"gt_characters": "Znaki GT", "correct_characters": "Poprawnie rozpoznane znaki", "incorrect_characters": "Błędnie rozpoznane znaki", "missing_characters": "Brakujące znaki", "extra_characters": "Znaki nadmiarowe", "cer": "CER — suma błędów / suma znaków GT"},
            "Lokalizacja MT": {"evaluable_mt_invocations": "Oceniane wywołania z jedną widoczną tablicą", "mt_successful_invocations": "Wywołania z poprawną lokalizacją", "mt_no_detection_invocations": "Wywołania bez detekcji", "mt_execution_error_invocations": "Wywołania z błędem wykonania", "mt_invalid_quad_invocations": "Wywołania z błędną geometrią", "mt_multi_plate_invocations": "Wywołania z wieloma tablicami", "mt_uncertain_invocations": "Wywołania niejednoznaczne", "mt_false_detections": "Fałszywe detekcje", "mt_localization_success_rate": "Skuteczność lokalizacji MT w wywołaniach z jedną tablicą"},
            "Czas i próby": {"median_time_to_first_exact_ms": "Mediana czasu do próby z poprawnym odczytem [ms]", "p90_time_to_first_exact_ms": "P90 czasu do próby z poprawnym odczytem [ms]", "median_attempts_to_first_exact": "Mediana liczby prób do poprawnego odczytu"},
        }
        for section, fields in sections.items():
            for key, label in fields.items():
                self.stats_tree.insert("", "end", values=(section, label, _display(summary.get(key), rate=key.endswith("rate") or key == "cer")))
        for row in self.stats["character_confusion"]:
            self.stats_tree.insert("", "end", values=("Pomyłki znaków", f"{row['ground_truth']} → {row['prediction']}", row["count"]))

    def _populate_details(self):
        values = {"session_id": self.session.session_id, "źródło": str(self.session.path),
                  "SHA-256": self.session.review.source_archive_sha256, "weryfikacja": str(self.session.sidecar_path),
                  "schemat próbek": self.session.bundle.sample_schema,
                  "eksperyment": self.session.bundle.experiment_session,
                  "capture": self.session.bundle.report_payload.get("capture", {}),
                  "pochodzenie modeli i warianty": self.session.review.provenance,
                  "błędy wykonania MT": {key: list(group.execution_errors) for key, group in self.session.mt_invocations.items() if group.execution_failed}}
        def append(value, prefix=""):
            for key, item in value.items():
                name = f"{prefix} / {key}" if prefix else str(key)
                if isinstance(item, dict):
                    append(item, name)
                else:
                    self.details_tree.insert("", "end", values=(name, json.dumps(item, ensure_ascii=False) if isinstance(item, list) else str(item)))
        append(values)

    def close(self):
        if self._closed:
            return
        self.save_gt()
        if self._saving:
            self._close_requested = True
            self.status_var.set("Kończę zapis weryfikacji przed zamknięciem…")
            return
        self._closed = True
        self.window.after_cancel(self._poll_id)
        if self._autosave_id:
            self.window.after_cancel(self._autosave_id)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()

    def _destroyed(self, event):
        if event.widget is self.window:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=False)


def open_mobile_sample_review(browser, report, bundle=None):
    reviews = {key: panel for key, panel in getattr(browser, "_sample_reviews", {}).items() if not panel._closed}
    key = report.source_archive_sha256 or report.source_path
    current = reviews.get(key)
    if current is not None and not current._closed:
        browser._sample_review = current
        current.window.lift()
        current.window.focus_set()
        return current
    browser._sample_review = MobileSampleReviewWindow(browser, report, bundle)
    reviews[key] = browser._sample_review
    browser._sample_reviews = reviews
    return browser._sample_review
