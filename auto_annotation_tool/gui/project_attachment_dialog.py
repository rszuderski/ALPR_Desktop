"""Explicit, asynchronous attachment of a copied project directory."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG


class ProjectAttachmentDialog:
    def __init__(self, host):
        self.host = host
        self.manager = CAMPAIGN
        self.plan = None
        self.busy = False
        self.attaching = False
        self.closed = False
        self.poll_after_id = None
        self.results = queue.Queue()
        self.window = tk.Toplevel(host.frame)
        host.app.style_dialog_window(self.window, title="Podłącz istniejący projekt", geometry="760x590", parent=host.frame)
        self.window.resizable(True, True)
        self.window.minsize(600, 450)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        body = ttk.Frame(self.window, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(6, weight=1)
        ttk.Label(body, text="Podłącz projekt skopiowany do katalogu projektów programu.",
                  style="PanelStatusInfo.TLabel", wraplength=690).grid(row=0, column=0, sticky="w", pady=(0, 12))
        candidates = self.manager.list_attachable_projects()
        ttk.Label(body, text="Katalog projektu").grid(row=1, column=0, sticky="w")
        picker = ttk.Frame(body)
        picker.grid(row=2, column=0, sticky="ew", pady=(4, 10))
        picker.columnconfigure(0, weight=1)
        self.folder = tk.StringVar(self.window, str(candidates[0]) if candidates else "")
        self.combo = ttk.Combobox(picker, textvariable=self.folder, values=[str(path) for path in candidates])
        self.combo.grid(row=0, column=0, sticky="ew")
        self.browse_button = ttk.Button(picker, text="Wybierz katalog…", command=self.browse)
        self.browse_button.grid(row=0, column=1, padx=(8, 0))
        self.inspect_button = ttk.Button(picker, text="Sprawdź", command=self.inspect)
        self.inspect_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(body, text="Nazwa na liście projektów").grid(row=3, column=0, sticky="w")
        self.name = tk.StringVar(self.window)
        self.name_entry = ttk.Entry(body, textvariable=self.name)
        self.name_entry.grid(row=4, column=0, sticky="ew", pady=(4, 10))
        self.status = tk.StringVar(self.window, "Wybierz katalog i sprawdź jego zawartość.")
        ttk.Label(body, textvariable=self.status, wraplength=690).grid(row=5, column=0, sticky="w", pady=(0, 8))
        self.details = tk.Text(body, wrap="word", height=15, state="disabled", bd=0)
        self.details.grid(row=6, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.details.yview)
        scrollbar.grid(row=6, column=1, sticky="ns")
        self.details.configure(yscrollcommand=scrollbar.set)
        host.app.style_text_widget(self.details, role="console")
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.grid(row=7, column=0, sticky="ew", pady=10)
        actions = ttk.Frame(body)
        actions.grid(row=8, column=0, sticky="ew")
        ttk.Button(actions, text="Zamknij", command=self.close).pack(side="right")
        self.attach_button = ttk.Button(actions, text="Podłącz projekt", command=self.attach, state="disabled")
        self.attach_button.pack(side="right", padx=(0, 8))
        self.folder.trace_add("write", self._folder_changed)
        self.poll_after_id = self.window.after(60, self.poll)

    def _folder_changed(self, *_args):
        if not self.busy:
            self.plan = None
            self.attach_button.configure(state="disabled")

    def browse(self):
        folder = filedialog.askdirectory(parent=self.window, title="Katalog istniejącego projektu",
                                         initialdir=str(CONFIG.DIR_9_PROJECTS), mustexist=True)
        if folder:
            self.folder.set(folder)
            self.inspect()

    def _set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _run(self, action, work):
        if self.busy:
            return
        self.busy = True
        self.attaching = action == "attach"
        for widget in (self.combo, self.name_entry, self.browse_button, self.inspect_button, self.attach_button):
            widget.configure(state="disabled")
        self.status.set("Podłączam projekt i zapisuję kopię bezpieczeństwa…" if self.attaching else "Sprawdzam projekt i ścieżki do jego zasobów…")
        self.progress.start(15)

        def worker():
            try:
                result = work()
                if not self.closed:
                    self.results.put((action, result, None))
            except Exception as exc:
                if not self.closed:
                    self.results.put((action, None, str(exc)))

        threading.Thread(target=worker, daemon=True, name="ProjectAttachment").start()

    def inspect(self):
        folder = self.folder.get().strip()
        if not folder:
            self.status.set("Najpierw wybierz katalog projektu.")
            return
        self.plan = None
        self._run("inspect", lambda: self.manager.inspect_existing_project(Path(folder)))

    def attach(self):
        if self.plan is None:
            return
        plan, name = self.plan, self.name.get().strip()
        if not name:
            self.status.set("Podaj nazwę projektu.")
            return
        self._run("attach", lambda: self.manager.attach_existing_project(plan, name=name))

    def poll(self):
        self.poll_after_id = None
        if not self.window.winfo_exists():
            return
        try:
            action, result, error = self.results.get_nowait()
        except queue.Empty:
            self.poll_after_id = self.window.after(60, self.poll)
            return
        self.busy = self.attaching = False
        self.progress.stop()
        for widget in (self.combo, self.name_entry, self.browse_button, self.inspect_button):
            widget.configure(state="normal")
        if error:
            self.status.set("Nie udało się podłączyć projektu." if action == "attach" else "Nie można podłączyć tego katalogu.")
            self._set_details(error)
            self.plan = None
            self.attach_button.configure(state="disabled")
        elif action == "inspect":
            self.plan = result
            self.name.set(result.name)
            data = result.project_data
            lines = [f"Projekt: {result.name}", f"Katalog: {result.root}",
                     f"Iteracja: {data.get('current_iteration', 1)}  •  Etap: Z{data.get('current_step', 1)}",
                     f"Treningi: {result.training_runs}  •  Tablice w bieżącym podglądzie: {result.preview_count}", ""]
            if result.state_source == "recovered":
                lines.append("Brak przenośnego opisu. Stan odtworzono z zachowanej historii i artefaktów; wybór aktywnych modeli może wymagać uzupełnienia.")
            else:
                lines.append("Znaleziono zapisany stan projektu.")
            lines += [f"Do dostosowania lokalizacji: {len(result.changes)} plików opisowych.",
                      "Oryginały zmienianych plików zostaną zachowane w kopii bezpieczeństwa."]
            if result.source_copies:
                lines.append(f"Do zabezpieczenia ze źródła roboczego: {len(result.source_copies)} obrazów zatwierdzonych anotacji.")
            if result.unresolved_paths:
                lines += ["", f"Nie znaleziono {len(result.unresolved_paths)} odwołań do zasobów poza projektem.",
                          "Te zasoby nie zostaną zastąpione innymi plikami. Jeśli będą potrzebne, wskaż je po otwarciu projektu."]
                lines.extend(result.unresolved_paths[:8])
                if len(result.unresolved_paths) > 8:
                    lines.append("Pełna lista zostanie zapisana w raporcie podłączenia.")
            self._set_details("\n".join(lines))
            self.status.set("Projekt gotowy do podłączenia. Podłączenie nie zmienia aktywnego projektu.")
            self.attach_button.configure(state="normal")
        else:
            self.host._refresh_projects_list()
            self.host._select_project_in_list(result["name"])
            self.host._on_project_changed()
            self.host.app.update_status(f"Podłączono projekt: {result['name']}. Możesz otworzyć go z listy.", "success")
            self.close()
            return
        self.poll_after_id = self.window.after(60, self.poll)

    def close(self):
        if self.attaching:
            self.status.set("Trwa zapis projektu. Poczekaj na zakończenie podłączania.")
            return
        self.closed = True
        self.plan = None
        if self.poll_after_id:
            self.window.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        if getattr(self.host, "_project_attachment_dialog", None) is self:
            self.host._project_attachment_dialog = None
        self.window.destroy()


def show_project_attachment_dialog(host):
    existing = getattr(host, "_project_attachment_dialog", None)
    if existing is not None and existing.window.winfo_exists():
        existing.window.lift()
        return existing
    dialog = ProjectAttachmentDialog(host)
    host._project_attachment_dialog = dialog
    return dialog
