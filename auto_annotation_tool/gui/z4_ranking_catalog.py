"""Native ranking catalog windows; selection/execution remain in existing Z4 callbacks."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from . import z4_ranking_identity as evidence


def _workspace(host):
    from ..config import CONFIG
    return Path(getattr(host, "_ranking_workspace", Path(CONFIG.DIR_9_PROJECTS).parent))


def _ranking_evidence(host):
    path = getattr(getattr(host, "ranking_engine", None), "ranking_file", None)
    return evidence.ranking_catalog(_workspace(host), [path] if path else [])


class Catalog:
    """A filtered view never edits the authoritative participant/track selection."""
    def __init__(self, host, title, columns, *, participants=False):
        self.host, self.participants = host, participants
        self.rows, self.visible, self.selected_key = [], {}, ""
        self.sort_column, self.descending = "", False
        self.columns = columns
        self.dialog = tk.Toplevel(host.frame)
        self.dialog.title(title)
        self.dialog.wm_transient("")
        self.dialog.resizable(True, True)
        self.dialog.minsize(920, 580)
        self.dialog.geometry("1320x760")
        # Root window recovery must not undo a user's explicit minimize.
        self.dialog._aat_skip_window_recovery = True
        self.dialog.catalog = self
        self.alive = True
        self.sha_epoch = 0
        self.sha_queue = queue.Queue()
        self.sha_cache, self.sha_pending = {}, set()
        self.poll_job = self.dialog.after(80, self.poll_sha)
        shell = ttk.Frame(self.dialog, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=3)
        shell.rowconfigure(4, weight=2)
        ttk.Label(shell, text=title, font=("Segoe UI", 14, "bold")).grid(row=0, sticky="w")
        self.intro = ttk.Label(shell, wraplength=1200)
        self.intro.grid(row=1, sticky="ew", pady=(4, 8))
        self.toolbar = ttk.Frame(shell)
        self.toolbar.grid(row=2, sticky="ew", pady=(0, 8))
        ttk.Label(self.toolbar, text="Szukaj:").pack(side="left")
        self.query = tk.StringVar(master=self.dialog)
        self.search_entry = ttk.Entry(self.toolbar, textvariable=self.query, width=34)
        self.search_entry.pack(side="left", padx=(6, 12), fill="x", expand=True)
        self.query.trace_add("write", lambda *_: self.render())
        self.refresh_button = ttk.Button(self.toolbar, text="Odśwież", command=lambda: self.reload())
        self.refresh_button.pack(side="right")
        table_frame = ttk.Frame(shell)
        table_frame.grid(row=3, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(table_frame, columns=tuple(c[0] for c in columns),
            show="tree headings" if participants else "headings", selectmode="browse", height=8)
        if participants:
            self.tree.column("#0", width=80, minwidth=75, stretch=False, anchor="center")
            self.tree.heading("#0", text="Start", command=lambda: self.sort("start"))
            self.tree.bind("<Button-1>", self.click_start)
            self.tree.bind("<space>", self.toggle_start)
        for key, label, width in columns:
            self.tree.heading(key, text=label, command=lambda k=key: self.sort(k))
            self.tree.column(key, width=width, minwidth=min(width, 90), stretch=key in {"directory", "checkpoint"})
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.ys = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.xs = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.ys.set, xscrollcommand=self.xs.set)
        self.ys.grid(row=0, column=1, sticky="ns")
        self.xs.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        details = ttk.LabelFrame(shell, text="Szczegóły zaznaczonego wiersza", padding=8)
        details.grid(row=4, sticky="nsew", pady=(10, 0))
        details.columnconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)
        path_row = ttk.Frame(details)
        path_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.path_value = tk.StringVar(master=self.dialog)
        ttk.Entry(path_row, textvariable=self.path_value, state="readonly").pack(side="left", fill="x", expand=True)
        self.copy_button = ttk.Button(path_row, text="Kopiuj pełną ścieżkę", command=self.copy_path, state="disabled")
        self.copy_button.pack(side="right", padx=(8, 0))
        self.details = tk.Text(details, wrap="word", height=8, state="disabled", font=("Segoe UI", 10))
        palette = getattr(getattr(host, "app", None), "palette", {})
        if palette.get("panel") and palette.get("fg"):
            self.details.configure(background=palette["panel"], foreground=palette["fg"])
        self.details.grid(row=1, column=0, sticky="nsew")
        ds = ttk.Scrollbar(details, orient="vertical", command=self.details.yview)
        self.details.configure(yscrollcommand=ds.set)
        ds.grid(row=1, column=1, sticky="ns")
        self.bottom = ttk.Frame(shell)
        self.bottom.grid(row=5, sticky="ew", pady=(9, 0))
        self.status = tk.StringVar(master=self.dialog)
        ttk.Label(self.bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        self.close_button = ttk.Button(self.bottom, text="Zamknij", command=self.close)
        self.close_button.pack(side="right")
        self.dialog.protocol("WM_DELETE_WINDOW", self.close)
        self.dialog.bind("<Escape>", lambda _e: self.close())
        self.dialog.bind("<Destroy>", self.destroyed, add="+")

    def close(self):
        self.dialog.destroy()

    def destroyed(self, event):
        if event.widget is self.dialog:
            self.alive = False
            self.sha_epoch += 1
            if self.poll_job:
                self.dialog.after_cancel(self.poll_job)
                self.poll_job = None

    def set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def selected(self):
        selected = self.tree.selection()
        return self.visible.get(selected[0]) if selected else None

    def render(self):
        if not self.alive:
            return
        self.tree.delete(*self.tree.get_children())
        tokens = evidence.search_text(self.query.get()).split()
        rows = [r for r in self.rows if all(t in r["search"] for t in tokens)]
        from .z4_analysis_ranking import _is_ranking_participant_enabled
        if self.participants:
            for row in self.rows:
                row["start"] = _is_ranking_participant_enabled(self.host, row["path"])
        if self.sort_column:
            def sort_key(row):
                value = row.get("completed") if self.sort_column == "epochs" else row.get(self.sort_column)
                if self.sort_column == "historical_samples":
                    return (0, tuple(row.get("evaluated_counts", [])))
                if isinstance(value, (int, float)):
                    return (0, value)
                return (1, str(value or "").casefold())
            rows.sort(key=sort_key, reverse=self.descending)
        self.visible = {}
        for row in rows:
            key = row["key"]
            self.visible[key] = row
            values = [evidence.UNKNOWN if row.get(c[0]) is None else row.get(c[0]) for c in self.columns]
            if self.participants:
                map_index = next(i for i, c in enumerate(self.columns) if c[0] == "map")
                value = row.get("map")
                if isinstance(value, (int, float)):
                    values[map_index] = f"{value*100 if value <= 1 else value:.2f}%"
            self.tree.insert("", "end", iid=key, values=values,
                             text=("TAK" if row["start"] else "NIE") if self.participants else "")
        for key, caption, _ in self.columns:
            self.tree.heading(key, text=caption + ((" ↓" if self.descending else " ↑") if self.sort_column == key else ""))
        if self.selected_key in self.visible:
            self.tree.selection_set(self.selected_key)
            self.tree.focus(self.selected_key)
            self.tree.see(self.selected_key)
        self.on_select()
        if self.participants:
            count = sum(r["start"] for r in self.rows)
            self.status.set(f"Widoczne: {len(rows)}/{len(self.rows)} • Start TAK: {count}/{len(self.rows)}")
        else:
            self.status.set(f"Widoczne tory: {len(rows)}/{len(self.rows)} • wybór dopiero przez Zastosuj")

    def sort(self, column):
        self.descending = not self.descending if self.sort_column == column else False
        self.sort_column = column
        self.render()

    def copy_path(self):
        if self.selected():
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(self.path_value.get())

    def on_select(self, _event=None):
        row = self.selected()
        self.sha_epoch += 1
        epoch = self.sha_epoch
        self.path_value.set(row["path"] if row else "")
        self.copy_button.configure(state="normal" if row else "disabled")
        if not row:
            self.set_details("Zaznacz wiersz, aby zobaczyć szczegóły.")
            return
        self.selected_key = row["key"]
        if not self.participants:
            self.set_details(evidence.track_details(row))
            return
        self.set_details(evidence.model_details(row))
        try:
            stat = Path(row["path"]).stat()
            token = (row["path"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            token = (row["path"], None)
        if token in self.sha_cache:
            self.set_details(evidence.model_details(row, self.sha_cache[token]))
            return
        self.current_sha_token = token
        if token in self.sha_pending:
            return
        self.sha_pending.add(token)
        def calculate():
            try:
                sha = evidence.actual_sha256(row["path"])
            except (OSError, ValueError) as exc:
                sha = "błąd odczytu: " + str(exc)
            self.sha_queue.put((token, row, sha))
        threading.Thread(target=calculate, name="ranking-model-sha", daemon=True).start()

    def poll_sha(self):
        if not self.alive:
            return
        try:
            while True:
                token, row, sha = self.sha_queue.get_nowait()
                self.sha_pending.discard(token)
                self.sha_cache[token] = sha
                selected = self.selected()
                if selected and selected["key"] == row["key"] and getattr(self, "current_sha_token", None) == token:
                    self.set_details(evidence.model_details(row, sha))
        except queue.Empty:
            pass
        self.poll_job = self.dialog.after(80, self.poll_sha)

    def click_start(self, event):
        if self.tree.identify_region(event.x, event.y) in {"tree", "cell"} and self.tree.identify_column(event.x) == "#0":
            item = self.tree.identify_row(event.y)
            if item:
                self.tree.selection_set(item)
                return self.toggle_start()

    def toggle_start(self, _event=None):
        from .z4_analysis_ranking import _is_ranking_participant_enabled, _set_ranking_participant_enabled
        row = self.selected()
        if row:
            self.selected_key = row["key"]
            _set_ranking_participant_enabled(self.host, row["path"], not _is_ranking_participant_enabled(self.host, row["path"]))
            self.render()
        return "break"


def _existing(host, name):
    dialog = getattr(host, name, None)
    if dialog is not None and dialog.winfo_exists():
        if dialog.state() == "iconic":
            dialog.deiconify()
        dialog.lift()
        return dialog


def open_tracks(host):
    from . import z4_analysis_ranking as legacy
    existing = _existing(host, "_ranking_track_modal")
    if existing:
        return existing
    ui = Catalog(host, "Wybór toru testowego rankingu", (
        ("status", "Gotowość", 135), ("directory", "Rzeczywisty katalog toru", 310),
        ("z2_id", "ID Z2", 205), ("type", "Typ", 105), ("source", "Projekt / globalny", 195),
        ("split", "Split", 70), ("source_count", "Obrazy źródłowe", 120),
        ("historical_samples", "Ocenione historycznie", 150), ("created", "Utworzono", 175),
        ("history_status", "Zapisane wyniki", 165)))
    host._ranking_track_modal = ui.dialog
    ui.intro.configure(text="Liczba obrazów toru i liczba obrazów historycznie ocenionych to odrębne dane. "
                       "Szukaj po ID, katalogu, ścieżce, dacie lub metadanych. Wybór wiersza pokazuje szczegóły.")
    split = tk.StringVar(master=ui.dialog, value=host._get_ranking_split_name())
    if host._get_ranking_task_target() == "char":
        combo = ttk.Combobox(ui.toolbar, textvariable=split, values=("test", "val"), state="readonly", width=7)
        combo.pack(side="left", padx=(0, 6))
        combo.bind("<<ComboboxSelected>>", lambda _e: ui.reload())
    ui.draft_split = split
    ui.selected_key = evidence.path_key(host.rank_data_dir.get())

    def load():
        catalog = _ranking_evidence(host)
        candidates = legacy._collect_ranking_track_candidates(host, split_name=split.get())
        ui.rows = evidence.historical_tracks(candidates, catalog, _workspace(host), host._get_ranking_task_target())
        ui.catalog_errors = catalog["errors"]
        for row in ui.rows:
            row["historical_samples"] = ", ".join(map(str, row["evaluated_counts"])) or "brak rekordu"
            if catalog["errors"]:
                row["metadata_errors"].extend(catalog["errors"])
                if not row["evaluations"]:
                    row["history_status"] = "historia niepełna / błąd"
        ui.render()

    def accept():
        row = ui.selected()
        if not row:
            return messagebox.showwarning("Brak wyboru", "Zaznacz tor testowy w tabeli.", parent=ui.dialog)
        if not row.get("ready"):
            return messagebox.showwarning("Tor nie jest gotowy", "Ten wpis służy do odczytu historii albo nie ma gotowego materiału. "
                                          "Zachowano aktualny tor rankingu.", parent=ui.dialog)
        host.rank_data_dir.set(row["path"])
        if row.get("split") != "EVAL396" and host._get_ranking_task_target() == "char":
            host.rank_split_var.set(split.get())
        host._refresh_ranking_reference_ui()
        ui.close()

    ui.reload = load
    ui.accept_button = ttk.Button(ui.bottom, text="[ TOR ] Zastosuj zaznaczony tor", command=accept)
    ui.accept_button.pack(side="right", padx=8)
    ttk.Button(ui.bottom, text="Zaawansowane", command=host._open_ranking_advanced_modal).pack(side="right")
    def double_click(event):
        if ui.tree.identify_region(event.x, event.y) == "cell" and ui.tree.identify_row(event.y):
            accept()
    ui.double_click = double_click
    ui.tree.bind("<Double-1>", double_click)
    load()
    return ui.dialog


def open_models(host):
    from . import z4_analysis_ranking as legacy
    from ..campaign_manager import CAMPAIGN
    existing = _existing(host, "_ranking_participants_modal")
    if existing:
        return existing
    if not hasattr(host, "rank_scope_var"):
        host.rank_scope_var = tk.StringVar(value="Projekt" if CAMPAIGN.get_active_project_name() else "Globalne")
    ui = Catalog(host, "Uczestnicy rankingu modeli", (
        ("participant", "ID uczestnika", 205), ("role", "Rola", 90), ("architecture", "Architektura", 165),
        ("size", "n / s", 90), ("run_id", "ID treningu", 160), ("checkpoint", "Pełna nazwa checkpointu", 300),
        ("date", "Data treningu", 175), ("epochs", "Epoki ukończone / plan", 165),
        ("scope", "Projekt / globalny", 200), ("map", "mAP50-95 treningu", 140),
        ("train_images", "Obrazy train", 110)), participants=True)
    host._ranking_participants_modal = ui.dialog
    ui.intro.configure(text="Start TAK/NIE określa uczestników. Wyszukiwanie, sortowanie i odświeżanie zachowują ten wybór. "
                       "Architektura i trening pochodzą z metadanych; brak danych oznacza „nieustalone”.")

    def load():
        histories = evidence.history_catalog(_workspace(host))
        ui.sha_cache.clear()
        target = host._get_ranking_task_target()
        raw_dir = host.rank_models_dir.get()
        paths = legacy._collect_ranking_participant_candidates(host, Path(raw_dir) if raw_dir else None, target, legacy._get_ranking_scope(host))
        ui.rows = [evidence.model_evidence(p, histories, _workspace(host), target) for p in paths]
        ui.render()

    def scope_changed():
        host._refresh_ranking_reference_ui()
        reload_ranking = getattr(host, "_load_ranking", None)
        if callable(reload_ranking):
            reload_ranking()
        load()

    for label, value in (("Projektowe", "Projekt"), ("Globalne", "Globalne")):
        ttk.Radiobutton(ui.toolbar, text=label, value=value, variable=host.rank_scope_var, command=scope_changed).pack(side="left")

    def directory():
        chosen = filedialog.askdirectory(parent=ui.dialog, title="Wskaż katalog modeli globalnych", initialdir=host.rank_models_dir.get())
        if chosen:
            host.rank_models_dir.set(chosen)
            host.rank_scope_var.set("Globalne")
            scope_changed()

    ttk.Button(ui.bottom, text="Wskaż katalog modeli", command=directory).pack(side="right", padx=8)
    ui.reload = load
    load()
    return ui.dialog
