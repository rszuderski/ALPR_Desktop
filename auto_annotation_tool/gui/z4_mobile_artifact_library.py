
"""Read-only mobile artifact browser for Z4; no model conversions/imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk, font as tkfont

from .z4_mobile_library_view import inventory_caption
from .z4_mobile_details_dialog import DetailSnapshot, MobileArtifactDetailsDialog
from .z4_mobile_library_view import (
    artifact_state, artifact_sheet, event_sheet, event_tone, short_sha,
    display_utc, file_name, size_mib,
)
from .z4_mobile_artifact_registry import library_snapshot, sha256_file
from .z4_mobile_operation_summary import summarize_operations
from .z4_mobile_artifact_status_ui import artifact_status_label, validation_result_text
from .z4_mobile_artifact_validation_history import (
    record_desktop_check, observe_last_checks, read_check_events, history_key,
    historical_status_label,
)
from .z4_mobile_history_presentation import (
    build_history_presentation, history_source_label,
    history_operation_label, history_outcome_label, history_detail_card,
    SOURCE_EXPORT, SOURCE_CHECK, SOURCE_FILE,
)
# Z4LIB035_HISTORY_UI_20261015
from .z4_mobile_artifact_transfer_ui import transfer_guidance
# Z4LIB036_COPY_READY_UI_20261016
# Z4LIB034_OPERATOR_STATE_GUIDANCE_20261015
from .z4_mobile_artifact_details import (
    artifact_search_text, checkpoint_sha_by_role, format_artifact_card, format_event_card,
)
# Z4LIB027_READABLE_ARTIFACT_LIBRARY_UI_20261015


def verify_unchanged_mobile_package(
    package_path: Path, *, kind: str, model_validator, bundle_validator,
) -> dict:
    """Full Desktop graph validation with an immutable-by-hash observation window.

    This does NOT approve Android import, calibration or F4 final_test. The hash
    is scoped to this validation run; the exporter must revalidate before reuse.
    """
    path = Path(package_path)
    if path.suffix.lower() != ".alprmodel" or path.is_symlink() or not path.is_file():
        raise ValueError("Wskaż istniejący, zwykły plik .alprmodel (bez dowiązania).")
    validator = model_validator if kind == "MODEL" else bundle_validator if kind == "PACKAGE" else None
    if not callable(validator):
        raise ValueError("Nieobsługiwany rodzaj paczki lub brak pełnego walidatora.")
    resolved_path = path.resolve(strict=True)
    before = sha256_file(path)
    validator(path, require_precision=True)
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != resolved_path:
        raise ValueError("Ścieżka lub rodzaj pliku zmieniły się podczas weryfikacji.")
    after = sha256_file(path)
    if before != after:
        raise ValueError("SHA-256 paczki zmienił się podczas pełnej walidacji; wynik unieważniono.")
    return {
        "path": str(path), "kind": kind, "sha256": after,
        "authority": "DESKTOP_EXPORTER_FULL_PRECISION_CHECK_ONLY",
        "android_verified": False, "final_test_verified": False,
    }


class MobileArtifactLibrary:
    def __init__(self, parent, *, packages_root: Path, repo_root: Path,
                 on_choose=None, choose_role: str | None = None,
                 choose_checkpoint: Path | None = None, choose_image_size: int | None = None,
                 choose_mt_precision: str = "int8", choose_mz_precision: str = "int8"):
        self.parent = parent
        self.on_choose = on_choose
        self.choose_role = str(choose_role or "").strip().upper()
        self.choose_checkpoint = Path(choose_checkpoint) if choose_checkpoint is not None else None
        self.choose_image_size = choose_image_size
        self.choose_mt_precision = str(choose_mt_precision or "int8").lower()
        self.choose_mz_precision = str(choose_mz_precision or "int8").lower()
        self.packages_root = Path(packages_root)
        self.repo_root = Path(repo_root)
        self.rows = []
        self.history = []
        self.by_iid = {}
        self.events_by_iid = {}
        # Per-tab sort survives search filtering and reloads. None = source order.
        self.sort_state = {"artifacts": None, "events": None}
        self.sort_labels = {
            "artifacts": {"role": "Rola", "name": "Nazwa", "variants": "Runtime / precyzja",
                          "size": "MiB", "date": "Modyfikacja (UTC)", "state": "Kontrola PC"},
            "events": {"date": "Data (UTC)", "source": "Źródło", "operation": "Operacja",
                        "outcome": "Wynik", "file": "Plik", "sha": "SHA pliku (skrót)"},
        }
        self.active_source = "artifact"
        self._event_view_raw = False
        self.pipe = queue.Queue()
        self.busy = False
        self._verification_seq = 0
        self._verification_active = None
        self._verified_selection_path = None
        self._verification_error = None  # only while THIS selection is active
        self._previous_desktop_checks = {}  # observations from separate history ledger
        self._desktop_check_events = []   # full recorded PC audit, read only
        self._desktop_check_note = ""
        self._history_read_note = ""
        self._closed = False
        self._poll_after_id = None
        self._layout_after_id = None
        self._details_dialog = None
        self._detail_sheet = None
        self._detail_text = ""
        self._detail_json = ""
        self.window = tk.Toplevel(parent)
        self.show_raw_events = tk.BooleanVar(master=self.window, value=False)
        self.event_view_status = tk.StringVar(
            master=self.window, value="Odczytuję operacje, kontrole PC i starsze pakiety..."
        )
        self.history_source_filter = tk.StringVar(master=self.window, value="Wszystkie źródła")
        self.window.title("Biblioteka modeli mobilnych — Z4")
        self.window.geometry("1300x790")
        self.window.minsize(940, 570)
        # Z4LIB002: an independent, modeless tool window must keep normal Win32
        # minimize/maximize/restore controls; a transient dialog can hide them.
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self._init_appearance()
        self.shell = shell = ttk.Frame(self.window, padding=12, style=self._ui + ".TFrame")
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        header = ttk.Frame(shell, style=self._ui + ".TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="Biblioteka modeli mobilnych", style=self._ui + ".Title.TLabel").pack(anchor="w")
        self.intro = ttk.Label(header,
            text="Z4 · Paczki, pochodzenie i kontrole techniczne. Wyniki PC nie są wynikami badań na telefonie.",
            style=self._ui + ".Muted.TLabel", justify="left")
        self.intro.pack(anchor="w", fill="x", pady=(2, 0))

        controls = ttk.Frame(shell, style=self._ui + ".TFrame")
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Szukaj:", style=self._ui + ".TLabel").grid(row=0, column=0, padx=(0, 6))
        self.filter_var = tk.StringVar(master=self.window)
        ent = ttk.Entry(controls, textvariable=self.filter_var, width=28)
        ent.grid(row=0, column=1, sticky="ew")
        self.filter_var.trace_add("write", lambda *_: self.refresh_views())
        ttk.Button(controls, text="Wyczyść", command=lambda: self.filter_var.set("")).grid(row=0, column=2, padx=6)
        self.refresh_button = ttk.Button(controls, text="Odśwież z dysku", command=self.refresh)
        self.refresh_button.grid(row=0, column=3)
        try:
            directory = str(self.packages_root.relative_to(self.repo_root))
        except ValueError:
            directory = "…/" + "/".join(self.packages_root.parts[-3:])
        self.storage_label = ttk.Label(controls, text="Katalog: " + directory,
            style=self._ui + ".Muted.TLabel", justify="left")
        self.storage_label.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))

        # FIX04: the lists own the entire elastic row. Details are on demand,
        # in a separate modal, never another pane competing with table rows.
        notebook = ttk.Notebook(shell, style=self._ui + ".TNotebook")
        notebook.grid(row=2, column=0, sticky="nsew")
        self.packages_tab = ttk.Frame(notebook, padding=(0, 6, 0, 0), style=self._ui + ".TFrame")
        self.events_tab = ttk.Frame(notebook, padding=(0, 6, 0, 0), style=self._ui + ".TFrame")
        notebook.add(self.packages_tab, text="Pakiety i modele")
        notebook.add(self.events_tab, text="Historia i kontrole PC")
        self.notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        self.package_count = tk.StringVar(master=self.window, value="Odczytuję paczki…")
        ttk.Label(self.packages_tab, textvariable=self.package_count, style=self._ui + ".Muted.TLabel").pack(anchor="w", pady=(0, 5))
        self._make_history_header()
        self.art_tree = self.build_tree(self.packages_tab,
            ("role", "name", "variants", "size", "date", "state"),
            ("Rola", "Nazwa", "Runtime / precyzja", "MiB", "Modyfikacja (UTC)", "Kontrola PC"),
            (85, 320, 230, 75, 165, 225),
            on_sort=lambda column: self.sort_column("artifacts", column))
        self.art_tree.bind("<<TreeviewSelect>>", self.artifact_selected)
        self.event_tree = self.build_tree(self.events_tab,
            ("date", "source", "operation", "outcome", "file", "sha"),
            ("Data (UTC)", "Źródło", "Operacja", "Wynik", "Plik", "SHA pliku (skrót)"),
            (190, 150, 180, 190, 285, 190),
            on_sort=lambda column: self.sort_column("events", column))
        self.event_tree.bind("<<TreeviewSelect>>", self.event_selected)
        for tree in (self.art_tree, self.event_tree):
            tree.bind("<Double-Button-1>", self._details_double_click, add="+")
            tree.bind("<Return>", self._details_key, add="+")
            tree.bind("<KP_Enter>", self._details_key, add="+")
            tree.configure(style=self._ui + ".Treeview")
            for tone in ("success", "warning", "danger", "info", "neutral"):
                tree.tag_configure(tone, background=self.colors[tone + "_bg"], foreground=self.colors["ink"])
        self.art_tree.column("size", anchor="e")

        self.footer = ttk.Frame(shell, style=self._ui + ".TFrame")
        self.footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.footer.columnconfigure(0, weight=1)
        self.action_group = ttk.Frame(self.footer, style=self._ui + ".TFrame")
        self.action_group.grid(row=0, column=0, sticky="ew")
        self.open_button = ttk.Button(self.action_group, text="Pokaż wybrany plik w folderze", command=self.open_location)
        self.path_button = ttk.Button(self.action_group, text="Kopiuj ścieżkę", command=self.copy_path)
        self.validate_button = ttk.Button(self.action_group, text="Sprawdź paczkę (pełna kontrola)", command=self.validate_selected)
        self.details_button = ttk.Button(self.action_group, text="Szczegóły…", command=self.show_details)
        self.action_buttons = [self.details_button, self.open_button, self.path_button, self.validate_button]
        self.choose_button = None
        if self.on_choose is not None:
            self.choose_button = ttk.Button(self.action_group, text=f"Wybierz jako {self.choose_role}", command=self.choose_for_export)
            self.action_buttons.append(self.choose_button)
        self.close_button = ttk.Button(self.footer, text="Zamknij", command=self.close)
        self.close_button.grid(row=0, column=1, sticky="se", padx=(10, 0), pady=2)
        self.status = tk.StringVar(master=self.window, value="Odczytuję bibliotekę…")
        self.status_label = ttk.Label(self.footer, textvariable=self.status,
            style=self._ui + ".Muted.TLabel", justify="left")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        shell.bind("<Configure>", self._queue_layout, add="+")
        self.window.bind("<<ThemeChanged>>", self._theme_changed, add="+")
        max_w = max(640, self.window.winfo_screenwidth() - 60)
        max_h = max(420, self.window.winfo_screenheight() - 100)
        if max_w < 1300 or max_h < 790:
            self.window.geometry(f"{min(1300, max_w)}x{min(790, max_h)}")
            self.window.minsize(min(940, max_w), min(570, max_h))
        self._poll_after_id = self.window.after(90, self.poll)
        self._queue_layout()
        self.refresh()

    @staticmethod
    def build_tree(parent, cols, headings, widths, *, on_sort=None):
        host = ttk.Frame(parent)
        host.pack(fill="both", expand=True)
        tree = ttk.Treeview(host, columns=cols, show="headings", selectmode="browse", height=4)
        for key, title, width in zip(cols, headings, widths):
            if on_sort is None:
                tree.heading(key, text=title)
            else:
                tree.heading(key, text=title, command=lambda column=key: on_sort(column))
            tree.column(key, width=width, minwidth={"name": 240, "file": 220, "variants": 180}.get(key, 68), stretch=(key in ("name", "variants", "file")))
        y = ttk.Scrollbar(host, command=tree.yview)
        x = ttk.Scrollbar(host, command=tree.xview, orient="horizontal")
        tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        host.grid_columnconfigure(0, weight=1)
        host.grid_rowconfigure(0, weight=1)
        return tree

    @staticmethod
    def _sort_value(source: str, column: str, item):
        """Use the actual metadata rather than formatted cells (especially MB/date)."""
        if source == "artifacts":
            fields = {
                "role": item.role, "name": item.name, "variants": item.variants,
                "size": item.size_bytes, "date": item.modified_utc, "state": item.state,
            }
            value = fields.get(column, "")
        elif source == "events":
            fields = {
                "date": item.get("_history_sort_date", item.get("observed_utc", "")),
                "source": history_source_label(item),
                "operation": item.get("operation", ""), "outcome": item.get("outcome", ""),
                "file": Path(str(item.get("artifact_path") or "-")).name,
                "sha": item.get("artifact_sha256") or "-",
            }
            value = fields.get(column, "")
        else:
            return ""
        if source == "artifacts" and column == "size":
            return int(value)
        # Both timestamps are generated in UTC ISO-8601 by the artifact registry.
        # The full value is used; the UI renders a shortened 19-char timestamp.
        return str(value or "").casefold()

    def sort_column(self, source: str, column: str) -> None:
        """Click heading: ascending, then descending. Does not change selection."""
        if source not in self.sort_state or column not in self.sort_labels[source]:
            return
        previous = self.sort_state[source]
        descending = bool(previous is not None and previous[0] == column and not previous[1])
        self.sort_state[source] = (column, descending)
        self._apply_sort(source)

    def _apply_sort(self, source: str) -> None:
        state = self.sort_state.get(source)
        if state is None:
            return
        column, descending = state
        tree = self.art_tree if source == "artifacts" else self.event_tree
        lookup = self.by_iid if source == "artifacts" else self.events_by_iid
        for key, label in self.sort_labels[source].items():
            marker = " ↓" if descending else " ↑"
            tree.heading(key, text=label + marker if key == column else label)
        ordered = sorted(
            tree.get_children(""),
            key=lambda iid: (
                self._state_view(lookup[iid]).label.casefold() if source == "artifacts" and column == "state"
                else history_operation_label(lookup[iid]).casefold() if source == "events" and column == "operation"
                else history_outcome_label(lookup[iid]).casefold() if source == "events" and column == "outcome"
                else self._sort_value(source, column, lookup[iid])),
            reverse=descending,
        )
        for position, iid in enumerate(ordered):
            tree.move(iid, "", position)

    def close(self):
        self._closed = True
        dialog = getattr(self, "_details_dialog", None)
        if dialog is not None:
            dialog.close(restore_focus=False)
        for name in ("_poll_after_id", "_layout_after_id"):
            timer = getattr(self, name, None)
            if timer:
                try:
                    self.window.after_cancel(timer)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def refresh(self):
        if self.busy:
            return
        # Re-reading the directory is not a new full graph check.
        self._verified_selection_path = None
        self._verification_error = None
        self.busy = True
        self.refresh_button.configure(state="disabled")
        self.status.set("Odczytuję listę paczek i ich opisy. Nie sprawdzam jeszcze modeli ani nie zmieniam plików...")
        def work():
            try:
                scan_result = library_snapshot(self.packages_root, self.repo_root)
                previous_checks = observe_last_checks(self.packages_root, scan_result[0])
                check_events, check_note = read_check_events(self.packages_root)
                self.pipe.put(("scan", (scan_result, previous_checks, check_events, check_note)))
            except Exception as exc:
                self.pipe.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True, name="z4-artifact-library-readonly-scan").start()

    def poll(self):
        if getattr(self, "_closed", False):
            return
        timer = getattr(self, "_poll_after_id", None)
        if timer is not None:
            try:
                self.window.after_cancel(timer)
            except (AttributeError, tk.TclError):
                pass
            self._poll_after_id = None
        try:
            while True:
                kind, value = self.pipe.get_nowait()
                if kind == "scan":
                    self.rows, self.history, note = value[0]
                    self._previous_desktop_checks = value[1]
                    self._desktop_check_events = value[2]
                    self._desktop_check_note = value[3]
                    self._history_read_note = note
                    self.busy = False
                    self.refresh_button.configure(state="normal")
                    self.refresh_views()
                    self.status.set(
                        inventory_caption(self.rows) + ". "
                        + ("Uwagi o źródłach w zakładce historii." if note or self._desktop_check_note else "")
                    )
                elif kind == "verified":
                    # The background result may arrive after a tab/row/filter change.
                    ticket = value if isinstance(value, dict) else {}
                    stored = ticket.get("history_record")
                    if isinstance(stored, dict) and stored.get("path"):
                        self._previous_desktop_checks[history_key(stored["path"])] = {
                            "record": stored,
                            "comparison": ("MATCH" if stored.get("sha256") else "UNCONFIRMED"),
                        }
                        entry = dict(stored)
                        entry["_ledger_index"] = max(
                            (int(item.get("_ledger_index", -1)) for item in self._desktop_check_events),
                            default=-1,
                        ) + 1
                        self._desktop_check_events.append(entry)
                    active = self._verification_active
                    if not isinstance(active, dict) or any(
                        ticket.get(key) != active.get(key) for key in ("token", "path", "kind")
                    ):
                        # A stale worker must NOT unlock a newer ongoing verification.
                        continue
                    current = self.selected_artifact() if self._active_tab_source() == "artifact" else None
                    still_selected = (
                        current is not None
                        and str(current.path) == ticket.get("path")
                        and current.kind == ticket.get("kind")
                    )
                    self._verification_active = None
                    self.busy = False
                    self.refresh_button.configure(state="normal")
                    if not still_selected:
                        self._verified_selection_path = None
                        self._verification_error = None
                        self.status.set(
                            "Kontrola dotyczyła innego zaznaczenia. Wybierz ponownie paczkę, "
                            "którą chcesz sprawdzić."
                        )
                    elif ticket.get("ok"):
                        self._verified_selection_path = ticket["path"]
                        self._verification_error = None
                        self.status.set(
                            "Kontrola na komputerze zaliczona. To nie jest jeszcze test na telefonie. "
                            "Teraz porównaj modele z planem eksperymentu."
                        )
                        self.show_text(validation_result_text(
                            ok=True, path=ticket["path"], sha256=str(ticket.get("sha256") or ""),
                        ))
                    else:
                        self._verified_selection_path = None
                        self._verification_error = {
                            "path": str(ticket.get("path") or ""),
                            "detail": str(ticket.get("detail") or "brak szczegółów"),
                        }
                        self.status.set("Kontrola na komputerze niezaliczona. Odczytaj przyczynę w szczegółach.")
                        self.show_text(validation_result_text(
                            ok=False, path=str(ticket.get("path") or ""),
                            problem=str(ticket.get("detail") or "brak szczegółów"),
                        ))
                    if ticket.get("history_warning"):
                        self.status.set(self.status.get() + " Nie udało się zapisać historii kontroli: "
                                        + str(ticket["history_warning"])[:240])
                    self._refresh_artifact_status_cells()
                    # Live refresh of both sources; do not present a PC check as export.
                    if isinstance(stored, dict) and stored.get("path"):
                        self.refresh_views()
                    if still_selected and hasattr(self, "colors"):
                        self.artifact_selected()

                else:
                    self.busy = False
                    self.refresh_button.configure(state="normal")
                    self.status.set("Błąd odczytu: " + str(value))
        except queue.Empty:
            pass
        self._sync_actions()
        try:
            if not getattr(self, "_closed", False):
                self._poll_after_id = self.window.after(150, self.poll)
        except tk.TclError:
            return

    @staticmethod
    def _selection_row(tree, lookup):
        """Resolve the selected OBJECT before Treeview rows are rebuilt."""
        try:
            selected = tree.selection()
            return lookup.get(selected[0]) if selected else None
        except (AttributeError, IndexError, tk.TclError):
            return None

    @staticmethod
    def _history_selection_key(event, *, raw: bool):
        if not isinstance(event, dict):
            return None
        origin = event.get("_history_origin")
        if origin == SOURCE_CHECK:
            return ("pc_check", str(event.get("_history_source_index")))
        if origin == SOURCE_FILE:
            return ("disk_file", history_key(event.get("artifact_path") or ""))
        if raw:
            event_id = str(event.get("event_id") or "").strip()
            return ("event", event_id) if event_id else None
        operation = str(event.get("operation") or "").strip()
        operation_id = str(event.get("operation_id") or "").strip()
        return ("operation", operation, operation_id) if operation and operation_id else None

    @staticmethod
    def _restore_row_by_identity(tree, lookup: dict, identity, key_fn):
        """Never restore by positional e0/a0: fresh scans may insert new rows."""
        if identity is None:
            return
        matches = [iid for iid, row in lookup.items() if key_fn(row) == identity]
        if len(matches) != 1:
            return
        try:
            tree.selection_set(matches[0])
            tree.focus(matches[0])
        except (AttributeError, tk.TclError):
            pass

    def refresh_views(self):
        phrase = self.filter_var.get().strip().casefold()
        current_artifact = self._selection_row(self.art_tree, self.by_iid)
        artifact_path = current_artifact.path if current_artifact is not None else None
        raw_view = bool(getattr(getattr(self, "show_raw_events", None), "get", lambda: False)())
        was_raw = bool(getattr(self, "_event_view_raw", raw_view))
        current_event = self._selection_row(self.event_tree, self.events_by_iid)
        event_key = (self._history_selection_key(current_event, raw=raw_view)
                     if raw_view == was_raw or (isinstance(current_event, dict)
                        and current_event.get("_history_origin") in (SOURCE_CHECK, SOURCE_FILE))
                     else None)
        self.by_iid.clear()
        for item in self.art_tree.get_children():
            self.art_tree.delete(item)
        for index, row in enumerate(self.rows):
            if phrase not in (artifact_search_text(row) + " " + artifact_status_label(row)
                              + " " + self._transfer_status_label(row) + " " + self._state_view(row).label).casefold():
                continue
            iid = f"a{index}"
            self.by_iid[iid] = row
            size = size_mib(row.size_bytes)
            state = self._state_view(row)
            self.art_tree.insert("", "end", iid=iid, values=(row.role, row.name, row.variants,
                                size, display_utc(row.modified_utc), state.label))
            if hasattr(self, "colors"):
                self.art_tree.item(iid, tags=(state.tone,))
        self.events_by_iid.clear()
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        # A partial __new__ test fixture without a package root represents the
        # original export-only interface. Real instances always have a root.
        if hasattr(self, "packages_root"):
            visible_history, counts = build_history_presentation(
                self.rows, self.history, self._desktop_check_events, raw_exports=raw_view,
            )
        else:
            visible_history = self.history if raw_view else summarize_operations(self.history)
            counts = {"export_operations": len(summarize_operations(self.history)),
                      "export_events": len(self.history), "pc_checks": 0, "unlogged_packages": 0}
        choice = str(getattr(getattr(self, "history_source_filter", None), "get", lambda: "Wszystkie źródła")())
        source_choices = {"Eksporty": SOURCE_EXPORT, "Kontrole PC": SOURCE_CHECK,
                          "Pakiety z dysku": SOURCE_FILE}
        source_limit = source_choices.get(choice)
        if source_limit is not None:
            visible_history = [row for row in visible_history
                               if row.get("_history_origin", SOURCE_EXPORT) == source_limit]
        history_total = len(visible_history)
        visible_history = [event for event in visible_history if phrase in (
            json.dumps(event, ensure_ascii=False, default=str) + " " + history_source_label(event) +
            " " + history_operation_label(event, raw_exports=raw_view) + " " + history_outcome_label(event) +
            " " + display_utc(event.get("_history_sort_date", event.get("observed_utc"))) +
            " " + short_sha(event.get("artifact_sha256"))).casefold()]
        self._update_history_header(counts, len(visible_history), history_total, choice)
        if hasattr(self, "package_count"):
            self.package_count.set(f"Wyświetlono: {len(self.by_iid)} / {len(self.rows)} artefaktów · MiB = bajty / 1 048 576")
        for index, event in enumerate(visible_history):
            # Stable legacy Treeview IDs for lightweight export-only fixtures;
            # actual unified histories use unique sequential view IDs.
            iid = (f"e{event.get('_source_index', index)}" if
                   not hasattr(self, "packages_root") and not raw_view else f"e{index}")
            self.events_by_iid[iid] = event
            date = display_utc(event.get("observed_utc"))
            if event.get("_history_origin") == SOURCE_FILE:
                date = "plik: " + display_utc(event.get("_file_modified_utc"))
            values = (
                date, history_source_label(event),
                history_operation_label(event, raw_exports=raw_view),
                history_outcome_label(event),
                file_name(event.get("artifact_path")),
                short_sha(event.get("artifact_sha256")),
            )
            # Partial legacy fixtures retain their original five-column shape;
            # real Tk library windows use the explicit six-column source display.
            if not hasattr(self, "packages_root"):
                values = (values[0], *values[2:])
            self.event_tree.insert("", "end", iid=iid, values=values)
            if hasattr(self, "colors"):
                self.event_tree.item(iid, tags=(event_tone(event),))
        # Refreshing/filtering rebuilds the Treeviews; reapply the user's sort.
        self._apply_sort("artifacts")
        self._apply_sort("events")
        self._event_view_raw = raw_view
        self._restore_row_by_identity(self.art_tree, self.by_iid, artifact_path,
                                      lambda row: row.path)
        self._restore_row_by_identity(self.event_tree, self.events_by_iid, event_key,
                                      lambda row: self._history_selection_key(row, raw=raw_view))
        self._refresh_artifact_status_cells()
        if getattr(self, "notebook", None) is not None:
            self._on_tab_changed()

    def _previous_record(self, row):
        if row is None:
            return None
        return getattr(self, "_previous_desktop_checks", {}).get(history_key(row.path))

    def _transfer_status_label(self, row, *, is_pass=False, is_fail=False, in_progress=False):
        """UI status only: copying file, not Android/experiment approval."""
        return transfer_guidance(
            row, desktop_validated=is_pass, validation_failed=is_fail,
            validation_running=in_progress, history_entry=self._previous_record(row),
        ).label

    def _refresh_artifact_status_cells(self) -> None:
        """Present a check result only for the active selection, never a stored approval."""
        tree = getattr(self, "art_tree", None)
        if tree is None or not callable(getattr(tree, "set", None)):
            return
        chosen = self.selected_artifact()
        path = str(chosen.path) if chosen is not None else None
        failure = getattr(self, "_verification_error", None)
        active = getattr(self, "_verification_active", None)
        for iid, row in getattr(self, "by_iid", {}).items():
            is_current = str(row.path) == path
            is_pass = (is_current and str(row.path) == getattr(self, "_verified_selection_path", None))
            is_fail = (is_current and isinstance(failure, dict)
                       and str(row.path) == failure.get("path"))
            in_progress = (is_current and isinstance(active, dict)
                           and str(row.path) == active.get("path")
                           and bool(getattr(self, "busy", False)))
            state = self._state_view(row, validated=is_pass, failed=is_fail, running=in_progress)
            label = state.label
            if hasattr(self, "colors"):
                tree.item(iid, tags=(state.tone,))
            try:
                if tree.set(iid, "state") != label:
                    tree.set(iid, "state", label)
            except (AttributeError, KeyError, tk.TclError):
                pass

    def selected_artifact(self):
        return self._selection_row(self.art_tree, self.by_iid)

    def selected_event(self):
        return self._selection_row(self.event_tree, self.events_by_iid)

    def _active_tab_source(self) -> str:
        notebook = getattr(self, "notebook", None)
        if notebook is not None:
            try:
                selected = str(notebook.select())
                if selected == str(self.events_tab):
                    return "event"
                if selected == str(self.packages_tab):
                    return "artifact"
            except (AttributeError, tk.TclError):
                pass
        return "event" if getattr(self, "active_source", "artifact") == "event" else "artifact"

    def _on_tab_changed(self, _event=None):
        """No stale JSON or copy/check action from the other (hidden) table."""
        self.active_source = self._active_tab_source()
        if self.active_source == "event":
            self.event_selected()
        else:
            self.artifact_selected()
        self._queue_layout()

    def _init_appearance(self):
        """Window-scoped styles; no global theme/font changes for Z1–Z4."""
        self._ui = f"Z4Library{id(self)}"
        self._font_body = tkfont.nametofont("TkDefaultFont", root=self.window).copy()
        self._font_bold = self._font_body.copy()
        self._font_bold.configure(weight="bold")
        self._font_title = self._font_body.copy()
        size = int(self._font_body.cget("size"))
        self._font_title.configure(size=(size + 2 if size > 0 else size - 3), weight="bold")
        self._font_mono = tkfont.nametofont("TkFixedFont", root=self.window).copy()
        self._apply_palette()

    def _apply_palette(self):
        style = ttk.Style(self.window)
        bg = style.lookup("TFrame", "background") or self.window.cget("background")
        try:
            rgb = self.window.winfo_rgb(bg)
            dark = sum(rgb) / (3 * 65535) < .48
        except tk.TclError:
            dark = False
        self.colors = ({
            "bg": "#232B33", "surface": "#29343E", "ink": "#EFF4F8", "muted": "#B8C5D0",
            "border": "#526372", "info": "#99D0F2", "success": "#8EE0B5",
            "warning": "#F5D28B", "danger": "#FFB0B4", "neutral": "#C3CDD7",
            "info_bg": "#263F52", "success_bg": "#243F35", "warning_bg": "#423B2A",
            "danger_bg": "#4B2D35", "neutral_bg": "#303B45",
        } if dark else {
            "bg": "#F1F4F7", "surface": "#FFFFFF", "ink": "#243343", "muted": "#536679",
            "border": "#D2DCE5", "info": "#205E8B", "success": "#216443",
            "warning": "#845A0C", "danger": "#A12D3A", "neutral": "#526275",
            "info_bg": "#EEF5FB", "success_bg": "#EFF7F2", "warning_bg": "#FFF7E7",
            "danger_bg": "#FFF0F1", "neutral_bg": "#F5F7FA",
        })
        c, prefix = self.colors, self._ui
        style.configure(prefix + ".TFrame", background=c["bg"])
        style.configure(prefix + ".TLabel", background=c["bg"], foreground=c["ink"], font=self._font_body)
        style.configure(prefix + ".Muted.TLabel", background=c["bg"], foreground=c["muted"], font=self._font_body)
        style.configure(prefix + ".Title.TLabel", background=c["bg"], foreground=c["ink"], font=self._font_title)
        style.configure(prefix + ".TLabelframe", background=c["bg"])
        style.configure(prefix + ".TLabelframe.Label", background=c["bg"], foreground=c["muted"], font=self._font_bold)
        style.configure(prefix + ".TNotebook", background=c["bg"], tabmargins=(0, 0, 0, 0))
        style.configure(prefix + ".TNotebook.Tab", padding=(10, 5), background=c["bg"], foreground=c["ink"])
        style.map(prefix + ".TNotebook.Tab", background=[("selected", c["surface"])], foreground=[("selected", c["info"])])
        style.configure(prefix + ".TCheckbutton", background=c["bg"], foreground=c["ink"])
        style.map(prefix + ".TCheckbutton", foreground=[("disabled", c["muted"])])
        style.configure(prefix + ".Treeview", background=c["surface"], fieldbackground=c["surface"],
                        foreground=c["ink"], font=self._font_body,
                        rowheight=max(26, self._font_body.metrics("linespace") + 9))
        style.configure(prefix + ".Treeview.Heading", font=self._font_bold, padding=(5, 6))
        style.map(prefix + ".Treeview", background=[("selected", c["info"])],
                  foreground=[("selected", "#FFFFFF" if not dark else "#142330")])

    def _theme_changed(self, _event=None):
        if getattr(self, "_closed", False) or not hasattr(self, "art_tree"):
            return
        self._apply_palette()
        for tone in ("success", "warning", "danger", "info", "neutral"):
            for tree in (self.art_tree, self.event_tree):
                tree.tag_configure(tone, background=self.colors[tone + "_bg"], foreground=self.colors["ink"])
        for key, (frame, value, hint, line) in getattr(self, "_source_cards", {}).items():
            tone = {"export_operations": "info", "pc_checks": "success", "unlogged_packages": "neutral"}[key]
            frame.configure(background=self.colors["surface"], highlightbackground=self.colors["border"])
            value.configure(background=self.colors["surface"], foreground=self.colors[tone])
            hint.configure(background=self.colors["surface"], foreground=self.colors["muted"])
            line.configure(background=self.colors[tone])
        dialog = getattr(self, "_details_dialog", None)
        if dialog is not None:
            dialog.apply_palette()

    def _make_history_header(self):
        self._source_cards = {}
        self._history_cards_frame = box = ttk.Frame(self.events_tab, style=self._ui + ".TFrame")
        box.pack(fill="x", pady=(0, 7))
        for column, (key, title, tone) in enumerate((
            ("export_operations", "Eksporty / konwersje", "info"),
            ("pc_checks", "Kontrole PC", "success"),
            ("unlogged_packages", "Paczki bez wpisu eksportu", "neutral"),
        )):
            box.columnconfigure(column, weight=1, uniform="history")
            frame = tk.Frame(box, background=self.colors["surface"], highlightthickness=1,
                             highlightbackground=self.colors["border"])
            frame.grid(row=0, column=column, sticky="nsew", padx=(0, 6 if column < 2 else 0))
            stripe = tk.Frame(frame, height=3, background=self.colors[tone])
            stripe.pack(fill="x")
            value = tk.Label(frame, text="—  " + title, anchor="w", font=self._font_bold,
                             background=self.colors["surface"], foreground=self.colors[tone])
            value.pack(fill="x", padx=10, pady=(6, 2))
            hint = tk.Label(frame, text="Odczyt…", anchor="w", font=self._font_body,
                            background=self.colors["surface"], foreground=self.colors["muted"])
            hint.pack(fill="x", padx=10, pady=(0, 6))
            self._source_cards[key] = (frame, value, hint, stripe)
        self._event_controls = event_controls = ttk.Frame(self.events_tab, style=self._ui + ".TFrame")
        event_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(event_controls, text="Źródło:", style=self._ui + ".TLabel").pack(side="left", padx=(0, 6))
        self.source_picker = ttk.Combobox(event_controls, textvariable=self.history_source_filter,
            values=("Wszystkie źródła", "Eksporty", "Kontrole PC", "Pakiety z dysku"), state="readonly", width=22)
        self.source_picker.pack(side="left")
        self.source_picker.bind("<<ComboboxSelected>>", lambda _event: self.refresh_views())
        self.audit_checkbox = ttk.Checkbutton(event_controls, text="Pokaż każde zdarzenie eksportu",
            variable=self.show_raw_events, command=self.refresh_views, style=self._ui + ".TCheckbutton")
        self.audit_checkbox.pack(side="left", padx=(12, 0))
        self.history_info_label = ttk.Label(self.events_tab, textvariable=self.event_view_status,
                                            justify="left", style=self._ui + ".Muted.TLabel")
        self.history_info_label.pack(fill="x", pady=(0, 4))
        self.history_note_var = tk.StringVar(master=self.window)
        self.history_note_label = ttk.Label(self.events_tab, textvariable=self.history_note_var,
                                            justify="left", style=self._ui + ".Muted.TLabel")
        self.history_note_label.pack(fill="x", pady=(0, 6))

    def _update_history_header(self, counts, visible, source_total, choice):
        self.history_counts = dict(counts)
        var = getattr(self, "event_view_status", None)
        if var is not None:
            var.set(f"Wiersze po filtrze: {visible} / {source_total} · Wpisy historii, nie liczba plików")
        titles = ({"export_operations": "Operacje", "pc_checks": "Kontrole PC", "unlogged_packages": "Paczki bez wpisu"}
                  if getattr(self, "_compact", False) else
                  {"export_operations": "Eksporty / konwersje", "pc_checks": "Kontrole PC", "unlogged_packages": "Paczki bez wpisu eksportu"})
        hints = {"export_operations": f"{counts['export_events']} zapisanych zdarzeń",
                 "pc_checks": "Wyniki historyczne, nie nowe testy",
                 "unlogged_packages": "Pliki, nie odtworzone operacje"}
        for key, (_, value, hint, _) in getattr(self, "_source_cards", {}).items():
            value.configure(text=f"{counts[key]}  {titles[key]}")
            hint.configure(text=hints[key])
        if hasattr(self, "audit_checkbox"):
            self.audit_checkbox.configure(state="normal" if choice in ("Wszystkie źródła", "Eksporty") else "disabled")
        if hasattr(self, "history_note_var"):
            notes = " ".join(x for x in (getattr(self, "_history_read_note", ""),
                                         getattr(self, "_desktop_check_note", "")) if x)
            if not notes:
                notes = ("Brak odczytanych zdarzeń eksportu. Nie oznacza to braku dawnych eksportów."
                         if counts["export_events"] == 0 else "Daty „plik:” oznaczają modyfikację, nie czas konwersji.")
            self.history_note_var.set(notes)
        self._queue_layout()

    def _queue_layout(self, _event=None):
        if not hasattr(self, "shell") or getattr(self, "_closed", False):
            return
        if getattr(self, "_layout_after_id", None) is None:
            self._layout_after_id = self.window.after_idle(self._layout_content)

    def _layout_content(self):
        self._layout_after_id = None
        if getattr(self, "_closed", False):
            return
        linespace = self._font_body.metrics("linespace")
        # Large system text needs a larger local minimum, not clipped controls.
        min_w = min(940 + max(0, linespace - 17) * 10, max(640, self.window.winfo_screenwidth() - 60))
        min_h = min(570 + max(0, linespace - 17) * 14, max(420, self.window.winfo_screenheight() - 100))
        if self.window.minsize() != (min_w, min_h):
            self.window.minsize(min_w, min_h)
        compact = (self.window.winfo_height() < 740 or
                   (self._font_body.metrics("linespace") > 22 and self.window.winfo_height() < 920))
        if compact != getattr(self, "_compact", None):
            self._compact = compact
            if compact:
                self.intro.pack_forget()
                self.storage_label.grid_remove()
            else:
                self.intro.pack(anchor="w", fill="x", pady=(2, 0))
                self.storage_label.grid()
            for key, (_, value, hint, _) in self._source_cards.items():
                if compact:
                    hint.pack_forget()
                else:
                    hint.pack(fill="x", padx=10, pady=(0, 6))
                titles = ({"export_operations": "Operacje", "pc_checks": "Kontrole PC", "unlogged_packages": "Paczki bez wpisu"}
                          if compact else {"export_operations": "Eksporty / konwersje", "pc_checks": "Kontrole PC", "unlogged_packages": "Paczki bez wpisu eksportu"})
                value.configure(text=f"{getattr(self, 'history_counts', {}).get(key, '—')}  {titles[key]}")
        note = self.history_note_var.get()
        important_note = any(word in note.casefold() for word in ("pominięto", "limit", "przekierow", "nieczytel", "zbyt", "nie uda"))
        if compact and not important_note:
            self.history_note_label.pack_forget()
        elif not self.history_note_label.winfo_manager():
            self.history_note_label.pack(before=self.event_tree.master, fill="x", pady=(0, 6))
        width = max(200, self.shell.winfo_width() - 26)
        for item in (self.intro, self.storage_label, self.status_label,
                     self.history_info_label, self.history_note_label):
            item.configure(wraplength=width)
        # Measure actual themed/DPI-scaled widgets, not hardcoded text widths.
        available = max(160, width - self.close_button.winfo_reqwidth() - 15)
        if not hasattr(self, "_action_rows"):
            self._action_rows = []
        x, row, col = 0, 0, 0
        for button in self.action_buttons:
            need = button.winfo_reqwidth() + 8
            if col and x + need > available:
                x, row, col = 0, row + 1, 0
            while len(self._action_rows) <= row:
                self._action_rows.append(ttk.Frame(self.action_group, style=self._ui + ".TFrame"))
            host = self._action_rows[row]
            host.grid(row=row, column=0, sticky="w")
            button.grid(in_=host, row=0, column=col, sticky="w", padx=(0, 7), pady=2)
            button.lift()
            x, col = x + need, col + 1
        for unused in self._action_rows[row + 1:]:
            unused.grid_remove()

    def _state_view(self, row, *, validated=None, failed=None, running=None):
        path = str(getattr(row, "path", ""))
        failure = getattr(self, "_verification_error", None)
        active = getattr(self, "_verification_active", None)
        if validated is None:
            validated = path == getattr(self, "_verified_selection_path", None)
        if failed is None:
            failed = isinstance(failure, dict) and failure.get("path") == path
        if running is None:
            running = bool(getattr(self, "busy", False) and isinstance(active, dict) and active.get("path") == path)
        return artifact_state(row, validated=validated, failed=failed, running=running, history=self._previous_record(row))

    def _render_sheet(self, sheet):
        """Cache display data without opening a window or resizing the list."""
        self._detail_sheet = sheet
        self._detail_text = "\n".join([sheet.title, sheet.subtitle, sheet.state,
            "Następny krok: " + sheet.next_step] + [f"{k}: {v}" for k, v in sheet.fields] + [sheet.note])

    def _sync_actions(self):
        if not hasattr(self, "validate_button"):
            return
        is_event = self._active_tab_source() == "event"
        row = None if is_event else self.selected_artifact()
        event = self.selected_event() if is_event else None
        details = getattr(self, "details_button", None)
        if details is not None:
            details.configure(state="normal" if row is not None or event is not None else "disabled")
        busy = bool(getattr(self, "busy", False))
        path = self.active_path()
        for button in (self.path_button, self.open_button):
            button.configure(state="normal" if path else "disabled")
        self.validate_button.configure(state="normal" if row is not None and row.kind in ("MODEL", "PACKAGE") and not busy else "disabled")
        if self.choose_button is not None:
            ready = row is not None and not busy and not self._choice_problem(row)
            self.choose_button.configure(state="normal" if ready else "disabled")

    def copy_evidence_sha(self):
        if self._active_tab_source() != "event":
            return self.copy_checkpoint_sha()
        value = str((self.selected_event() or {}).get("artifact_sha256") or "")
        if short_sha(value) == "—":
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(value)
        self.status.set("Skopiowano pełne SHA pliku zapisane we wpisie — nie SHA checkpointu i nie nowy pomiar.")


    def _details_double_click(self, event):
        tree = getattr(event, "widget", None)
        active = self.event_tree if self._active_tab_source() == "event" else self.art_tree
        if tree is not active or tree.identify_region(event.x, event.y) not in ("cell", "tree"):
            return
        iid = tree.identify_row(event.y)
        lookup = self.events_by_iid if tree is self.event_tree else self.by_iid
        if not iid or iid not in lookup:
            return
        tree.selection_set(iid)
        tree.focus(iid)
        self.show_details()
        return "break"

    def _details_key(self, event):
        tree = self.event_tree if self._active_tab_source() == "event" else self.art_tree
        if event.widget is tree and self.window.focus_get() is tree:
            self.show_details()
            return "break"

    def show_details(self):
        """Open a read-only, pinned snapshot for the active selection, once."""
        if getattr(self, "_closed", False):
            return None
        previous = getattr(self, "_details_dialog", None)
        if previous is not None and previous.is_open():
            previous.raise_window()
            return previous
        source = self._active_tab_source()
        item = self.selected_event() if source == "event" else self.selected_artifact()
        if item is None:
            return None
        if source == "event":
            self.event_selected()
            path = str(item.get("artifact_path") or "")
            name = file_name(path) if path else history_operation_label(item)
            sha = str(item.get("artifact_sha256") or "")
            sha = sha if short_sha(sha) != "—" else ""
            sha_label = "Kopiuj SHA pliku z wpisu"
            context = "Wpis historii · bez nowej kontroli pliku"
            json_label = "Wpis źródłowy (JSON)"
            identity = repr(self._history_selection_key(item, raw=bool(self.show_raw_events.get())))
            tree = self.event_tree
        else:
            self.artifact_selected()
            path, name = str(item.path), str(item.name)
            hashes = checkpoint_sha_by_role(item)
            sha = (hashes[0][1] if len(hashes) == 1 else "\n".join(f"{role}: {value}" for role, value in hashes))
            sha_label = "Kopiuj SHA checkpointu"
            context = "Model / pakiet · dane i kontrola PC"
            json_label = "Manifest i źródła (JSON)"
            identity = path
            tree = self.art_tree
        snapshot = DetailSnapshot(source=source, identity=identity, name=name, path=path,
            context=context, json_label=json_label, sha_label=sha_label, sha_value=sha,
            sheet=self._detail_sheet, summary_text=self._detail_text, raw_json=self._detail_json)
        dialog = MobileArtifactDetailsDialog(self, snapshot, return_focus=tree)
        self._details_dialog = dialog
        return dialog

    def _choice_problem(self, row) -> str:
        """Fast, read-only selection gate. NOT equivalent to full graph validation."""
        if not callable(self.on_choose) or self.choose_role not in {"MT", "MZ"}:
            return "Biblioteka nie jest w trybie wyboru modelu."
        if row is None or row.kind != "MODEL" or row.role != self.choose_role:
            return f"Wybierz pojedynczy model {self.choose_role} (.alprmodel), nie pakiet złożony."
        if row.state in {"REGISTERED_SHA_MISMATCH", "RESEARCH_SHA_MISMATCH"} or row.error:
            return "Pakiet ma błąd integralności lub manifestu."
        source = row.manifest.get("source") if isinstance(row.manifest, dict) else None
        declared_sha = str(source.get("checkpoint_sha256") or "").lower() if isinstance(source, dict) else ""
        if self.choose_checkpoint is None or not self.choose_checkpoint.is_file():
            return "Brak checkpointu wybranego w Centrum eksportu."
        try:
            checkpoint_sha = sha256_file(self.choose_checkpoint)
        except OSError:
            return "Nie można odczytać checkpointu do kontroli SHA."
        if not declared_sha or declared_sha != checkpoint_sha:
            return "SHA checkpointu pakietu nie odpowiada wybranemu modelowi w Centrum eksportu."
        input_spec = row.manifest.get("input") if isinstance(row.manifest, dict) else None
        if not isinstance(input_spec, dict) or self.choose_image_size is None:
            return "Nie można potwierdzić rozmiaru wejścia modelu."
        try:
            if int(input_spec.get("width") or 0) != int(self.choose_image_size) or int(input_spec.get("height") or 0) != int(self.choose_image_size):
                return "Rozmiar wejścia (imgsz) pakietu różni się od wybranego checkpointu."
        except (TypeError, ValueError):
            return "Błędny rozmiar wejścia w manifeście."
        if self.choose_role == "MT":
            # The operator explicitly chooses which variant to reuse. No silent
            # FP32 fallback when INT8 was requested (or vice versa).
            expected = getattr(self, 'choose_mt_precision', 'int8')
            if expected not in {'int8', 'fp32'}:
                return "MT: wybierz jawnie INT8 albo FP32."
            variants = row.manifest.get("variants")
            if (not isinstance(variants, list) or len(variants) != 1
                    or not isinstance(variants[0], dict) or variants[0].get("runtime") != "tflite"
                    or variants[0].get("precision") != expected):
                return f"MT: ponowne użycie wymaga pojedynczego wariantu TFLite {expected.upper()}."
            data_type = str(input_spec.get('data_type') or '').upper()
            if expected == 'fp32' and data_type != 'FLOAT32':
                return "MT: FP32 wymaga wejścia FLOAT32."
            if expected == 'int8' and data_type and data_type != 'INT8':
                return "MT: INT8 wymaga wejścia signed INT8."
        if self.choose_role == "MZ":
            expected = getattr(self, "choose_mz_precision", "int8")
            if expected not in {"int8", "fp32"}:
                return "MZ: wybierz jawnie INT8 albo FP32."
            variants = row.manifest.get("variants")
            if (not isinstance(variants, list) or len(variants) != 1
                    or not isinstance(variants[0], dict) or variants[0].get("runtime") != "tflite"
                    or str(variants[0].get("precision") or "").lower() != expected):
                return f"MZ: dla F4 wymagany pojedynczy wariant TFLite {expected.upper()}."
            input_type = str(input_spec.get("data_type") or "").upper()
            # Discovery is a prefilter: legacy INT8 manifests may omit data_type.
            # The mandatory exporter full-graph validator rejects incorrect dtype.
            if ((expected == "fp32" and input_type != "FLOAT32")
                    or (expected == "int8" and input_type and input_type != "INT8")):
                return f"MZ: {expected.upper()} wymaga zgodnego typu wejścia grafu."
        return ""

    def choose_for_export(self):
        if self._active_tab_source() == "event":
            return messagebox.showwarning(
                "Wybór modelu z biblioteki", "Wybierz paczkę w zakładce Pakiety i modele.",
                parent=self.window,
            )
        row = self.selected_artifact()
        problem = self._choice_problem(row)
        if problem:
            return messagebox.showwarning("Wybór modelu z biblioteki", problem, parent=self.window)
        try:
            # UI binds only the path. Export preflight owns authoritative validation.
            self.on_choose(Path(row.path))
        except Exception as exc:
            return messagebox.showerror("Wybór modelu z biblioteki", str(exc), parent=self.window)
        self.close()

    def show_text(self, value: str):
        """Update the pending text only; never mutate an open snapshot modal."""
        self._detail_sheet = None
        self._detail_text = value

    def show_raw(self, value: str):
        """Cache full source JSON. Table selection does not display a detail pane."""
        self._detail_json = value

    def copy_checkpoint_sha(self):
        """Copy source SHA(s) only, never confuse with package or file SHA."""
        if self._active_tab_source() != "artifact":
            return messagebox.showinfo("Biblioteka", "Przejdź do zakładki Pakiety i modele.", parent=self.window)
        row = self.selected_artifact()
        if row is None:
            return messagebox.showinfo("Biblioteka", "Najpierw wskaż paczkę.", parent=self.window)
        hashes = checkpoint_sha_by_role(row)
        if not hashes:
            return messagebox.showinfo("Biblioteka", "Brak wiarygodnego checkpoint_sha256 w manifeście.", parent=self.window)
        payload = hashes[0][1] if len(hashes) == 1 else "\n".join(
            f"{role}: {sha}" for role, sha in hashes
        )
        self.window.clipboard_clear()
        self.window.clipboard_append(payload)
        self.status.set(f"Skopiowano SHA checkpointu: {len(hashes)} pozycji (tylko deklaracje manifestu).")

    def artifact_selected(self, _event=None):
        if self._active_tab_source() != "artifact":
            return
        self.active_source = "artifact"
        row = self.selected_artifact()
        previous = getattr(self, "_verified_selection_path", None)
        if previous and (row is None or str(row.path) != previous):
            self._verified_selection_path = None
            if getattr(self, "status", None) is not None:
                self.status.set("Wybrano inną paczkę. Jej model nie został jeszcze sprawdzony w tej sesji.")
        failure = getattr(self, "_verification_error", None)
        if isinstance(failure, dict) and (row is None or str(row.path) != failure.get("path")):
            self._verification_error = None
        self._refresh_artifact_status_cells()
        if row:
            data = {
                "UWAGA": "ODCZYT Z MANIFESTU — NIE JEST CERTYFIKATEM PRECISION ANI ANDROID",
                "state": row.state, "error": row.error, "path": row.path,
                "size_bytes": row.size_bytes, "schema": row.schema,
                "role": row.role, "model_id": row.model_id, "variants": row.variants,
                "manifest": row.manifest,
                "child_metadata": row.child_metadata,
                "child_metadata_authority": "SHA manifestu potomnego, nie pełna walidacja TFLite",
            }
            self._render_sheet(artifact_sheet(
                row, validated=(str(row.path) == str(getattr(self, "_verified_selection_path", None))),
                failed=bool(isinstance(failure, dict) and failure.get("path") == str(row.path)),
                running=bool(getattr(self, "busy", False) and isinstance(getattr(self, "_verification_active", None), dict)
                             and self._verification_active.get("path") == str(row.path)),
                history=self._previous_record(row),
                failure=str(failure.get("detail") or "") if isinstance(failure, dict) else "",
            ))
            self.show_raw(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            self.show_text("Wybierz paczkę i otwórz „Szczegóły…”.")
            self.show_raw("")
        self._sync_actions()

    def event_selected(self, _event=None):
        if self._active_tab_source() != "event":
            return
        self.active_source = "event"
        event = self.selected_event()
        if event:
            self._render_sheet(event_sheet(event))
            self.show_raw(json.dumps(event, ensure_ascii=False, indent=2))
        else:
            self.show_text("Wybierz wpis historii. Kontrola PC i odnaleziony plik nie są zdarzeniami eksportu.")
            self.show_raw("")
        self._sync_actions()

    def active_path(self) -> Path | None:
        if self._active_tab_source() == "event":
            event = self.selected_event()
            return Path(event["artifact_path"]) if event and event.get("artifact_path") else None
        row = self.selected_artifact()
        return Path(row.path) if row else None

    def copy_path(self):
        path = self.active_path()
        if not path:
            return messagebox.showinfo("Biblioteka", "Najpierw wybierz plik lub zdarzenie.", parent=self.window)
        self.window.clipboard_clear()
        self.window.clipboard_append(str(path))
        self.status.set("Skopiowano ścieżkę: " + str(path))

    def open_location(self):
        path = self.active_path()
        if not path or not path.exists():
            return messagebox.showwarning("Biblioteka", "Brak istniejącego pliku do otwarcia.", parent=self.window)
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer.exe", "/select,", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except OSError as exc:
            messagebox.showerror("Biblioteka", str(exc), parent=self.window)

    def validate_selected(self):
        if self._active_tab_source() == "event":
            return messagebox.showwarning(
                "Biblioteka", "Wybierz paczkę w zakładce Pakiety i modele.", parent=self.window,
            )
        row = self.selected_artifact()
        if not row or row.kind not in ("MODEL", "PACKAGE"):
            return messagebox.showwarning("Biblioteka", "Wskaż właściwy pakiet .alprmodel.", parent=self.window)
        if self.busy:
            return
        self.busy = True
        self.refresh_button.configure(state="disabled")
        self.status.set("Waliduję rzeczywistą paczkę i jej SHA-256. Może to chwilę potrwać...")
        self._verification_seq = int(getattr(self, "_verification_seq", 0)) + 1
        token = self._verification_seq
        path, selected_kind = str(row.path), row.kind
        self._verification_active = {"token": token, "path": path, "kind": selected_kind}
        self._verified_selection_path = None
        self._verification_error = None
        self._refresh_artifact_status_cells()
        self._sync_actions()

        def work():
            try:
                from ..exporters.mobile_model_exporter import MobileModelExporter, MobileAlprPackageExporter
                proof = verify_unchanged_mobile_package(
                    Path(path), kind=selected_kind,
                    model_validator=MobileModelExporter().validate_package,
                    bundle_validator=MobileAlprPackageExporter().validate_package,
                )
                # History is only an observation. Validation before reuse is unchanged.
                history_record, history_warning = None, ""
                try:
                    history_record = record_desktop_check(
                        self.packages_root, path=path, kind=selected_kind,
                        passed=True, sha256=proof["sha256"],
                        detail="Kontrola techniczna na komputerze: PASS")
                except (OSError, ValueError, RuntimeError) as write_exc:
                    history_warning = str(write_exc)
                self.pipe.put(("verified", {
                    "token": token, "path": path, "kind": selected_kind, "ok": True,
                    "sha256": proof["sha256"], "detail": "Weryfikator Desktop: PASS",
                    "history_record": history_record, "history_warning": history_warning,
                }))
            except Exception as exc:
                # A failed graph check is recorded as FAILED, never as permission.
                history_record, history_warning = None, ""
                try:
                    try:
                        failed_sha = sha256_file(Path(path))
                    except (OSError, ValueError, RuntimeError):
                        failed_sha = ""
                    history_record = record_desktop_check(
                        self.packages_root, path=path, kind=selected_kind,
                        passed=False, sha256=failed_sha,
                        detail=f"{type(exc).__name__}: {exc}")
                except (OSError, ValueError, RuntimeError) as write_exc:
                    history_warning = str(write_exc)
                self.pipe.put(("verified", {
                    "token": token, "path": path, "kind": selected_kind, "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "history_record": history_record, "history_warning": history_warning,
                }))
        threading.Thread(target=work, daemon=True, name="z4-artifact-full-verifier").start()


def open_mobile_artifact_library(parent, *, packages_root=None, repo_root=None,
                                 on_choose=None, choose_role=None,
                                 choose_checkpoint=None, choose_image_size=None,
                                 choose_mt_precision="int8", choose_mz_precision="int8"):
    from ..config import CONFIG
    root = Path(packages_root) if packages_root else Path(CONFIG.DIR_6_MODELS_MOBILE_PACKAGES)
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return MobileArtifactLibrary(
        parent, packages_root=root, repo_root=repo,
        on_choose=on_choose, choose_role=choose_role,
        choose_checkpoint=choose_checkpoint, choose_image_size=choose_image_size,
        choose_mt_precision=choose_mt_precision, choose_mz_precision=choose_mz_precision,
    ).window
