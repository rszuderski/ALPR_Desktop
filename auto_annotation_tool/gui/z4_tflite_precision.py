"""Read-only graph inspection inside the native Z4 mobile export workflow."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ..exporters.mobile_model_exporter import inspect_mobile_precision


class TflitePrecisionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Precyzja grafu TFLite")
        self.geometry("1020x680")
        self.minsize(700, 420)
        self.transient(parent.winfo_toplevel())
        self.result = None
        self._events = queue.Queue()
        self._closed = False
        self._poll_id = None
        self.status = tk.StringVar(self, "Wybierz istniejący plik .tflite lub .alprmodel.")
        self.fingerprint = tk.StringVar(self, "SHA-256 pliku: —")
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Analiza statyczna pliku. Wykonanie integer na telefonie: UNVERIFIED.").pack(anchor="w")
        ttk.Label(body, textvariable=self.status, wraplength=960).pack(anchor="w", pady=6)
        ttk.Label(body, textvariable=self.fingerprint, wraplength=960).pack(anchor="w")
        self.table = ttk.Treeview(body, columns=("role", "variant", "declared", "graph", "metadata"),
                                  show="headings", height=6)
        for key, text, width in (("role", "Model", 90), ("variant", "Wariant", 170),
                                 ("declared", "Deklaracja", 95), ("graph", "Graf", 110),
                                 ("metadata", "Zgodność deklaracji", 380)):
            self.table.heading(key, text=text)
            self.table.column(key, width=width)
        self.table.pack(fill="x", pady=8)
        self.details = scrolledtext.ScrolledText(body, wrap="none", height=12, state="disabled")
        self.details.pack(fill="both", expand=True)
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(10, 0))
        self.choose_button = ttk.Button(actions, text="Wybierz model lub paczkę", command=self.choose)
        self.choose_button.pack(side="left")
        self.save_button = ttk.Button(actions, text="Zapisz raport analizy", command=self.save, state="disabled")
        self.save_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Zamknij", command=self.destroy).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def choose(self):
        path = filedialog.askopenfilename(parent=self, title="Sprawdź precyzję istniejącego modelu",
                                         filetypes=(("Model TFLite lub pakiet ALPR", "*.tflite *.alprmodel"),))
        if path:
            self.inspect(Path(path))

    def inspect(self, path: Path):
        if self._closed or str(self.choose_button["state"]) == "disabled":
            return
        self.result = None
        self.save_button.configure(state="disabled")
        self.choose_button.configure(state="disabled")
        self.table.delete(*self.table.get_children())
        self.fingerprint.set("SHA-256 pliku: —")
        self._show_details("")
        self.status.set("Odczyt grafów i kontrola SHA-256…")

        def worker():
            try:
                self._events.put((True, inspect_mobile_precision(path)))
            except Exception as exc:
                self._events.put((False, str(exc)))

        threading.Thread(target=worker, name="z4-tflite-precision", daemon=True).start()
        self._poll_id = self.after(50, self._poll)

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            ok, result = self._events.get_nowait()
        except queue.Empty:
            self._poll_id = self.after(50, self._poll)
            return
        self.choose_button.configure(state="normal")
        if not ok:
            self.status.set("Nie udało się zweryfikować pliku: " + result)
            return
        self.result = result
        self.fingerprint.set("SHA-256 pliku: " + result["file_sha256"])
        for index, row in enumerate(result["variants"]):
            self.table.insert("", "end", iid=str(index), values=(row["role"] or "TFLite", row["variant_id"] or row["file"],
                              row["declared_precision"] or "—", row["graph"]["classification"],
                              "; ".join(row["metadata_issues"]) or ("zgodna" if row["declared_precision"] else "brak deklaracji")))
        self.status.set(("Analiza zakończona." if result["variants"] else "Paczka nie zawiera wariantów TFLite.")
                        + " Klasyfikacja grafu nie potwierdza wykonania na telefonie.")
        self._show_details(json.dumps(result, ensure_ascii=False, indent=2))
        self.save_button.configure(state="normal")

    def _show_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def save(self):
        if not self.result:
            return
        source = Path(self.result["path"])
        name = filedialog.asksaveasfilename(parent=self, title="Zapisz raport analizy grafu",
                                          initialfile=source.name + ".precision.json", defaultextension=".json",
                                          filetypes=(("Raport JSON", "*.json"),))
        if not name:
            return
        path = Path(name)
        try:
            if path.resolve() == source.resolve() or path.suffix.lower() != ".json":
                raise ValueError("Raport musi być oddzielnym plikiem .json; model pozostaje niezmieniony.")
            path.write_text(json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Nie zapisano raportu", str(exc), parent=self)
            return
        self.status.set("Zapisano raport analizy: " + str(path))

    def destroy(self):
        self._closed = True
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().destroy()


def open_tflite_precision_dialog(parent):
    return TflitePrecisionDialog(parent)
