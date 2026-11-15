"""Acquisition gallery and human text review before handing crops to Z3."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from PIL import Image, ImageTk

from ..mobile_acquisition import CropReview, export_annotation_preview, read_crop_session


def open_crop_acquisition(host):
    window = getattr(host, "_mobile_acquisition_window", None)
    if window is not None and window.winfo_exists():
        window.lift()
        return window
    window = CropAcquisitionWindow(host)
    host._mobile_acquisition_window = window
    return window


class CropAcquisitionWindow(tk.Toplevel):
    def __init__(self, host):
        super().__init__(host.frame)
        self.host = host
        self.title("Akwizycja Androida")
        self.geometry("1040x720")
        self.minsize(760, 540)
        self.session = None
        self.review = None
        self.current = None
        self.busy = False
        self.source_image = None
        self.photo = None
        self.jobs = queue.Queue()
        self.filter_var = tk.StringVar(self)
        self.gt_var = tk.StringVar(self)
        self.boxes_var = tk.BooleanVar(self, True)
        self.status_var = tk.StringVar(self, "Brak otwartej paczki")
        self.path_var = tk.StringVar(self)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=10)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.grid_columnconfigure(1, weight=1)
        self.open_button = ttk.Button(toolbar, text="Otwórz paczkę", command=self.open_file)
        self.open_button.grid(row=0, column=0, padx=(0, 10))
        ttk.Entry(toolbar, textvariable=self.path_var, state="readonly").grid(row=0, column=1, sticky="ew")

        content = ttk.Panedwindow(self, orient="horizontal")
        content.grid(row=1, column=0, sticky="nsew", padx=10)
        left = ttk.Frame(content, width=310)
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=3)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        ttk.Entry(left, textvariable=self.filter_var).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.tree = ttk.Treeview(left, columns=("number", "count", "status"), show="headings", selectmode="browse")
        for name, text, width in (("number", "Odczyt MZ", 120), ("count", "Obserwacje", 75), ("status", "Ocena", 85)):
            self.tree.heading(name, text=text)
            self.tree.column(name, width=width, minwidth=55, stretch=name == "number")
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.select_group)
        self.filter_var.trace_add("write", lambda *_: self.refresh_list())

        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(3, weight=2)
        self.canvas = tk.Canvas(right, background="#222222", height=210, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 0))
        self.canvas.bind("<Configure>", lambda _event: self.render_image())
        ttk.Checkbutton(right, text="Ramki MZ", variable=self.boxes_var,
                        command=self.render_image).grid(row=1, column=0, sticky="w", padx=10, pady=4)
        review_row = ttk.Frame(right, padding=(10, 6))
        review_row.grid(row=2, column=0, sticky="ew")
        review_row.grid_columnconfigure(1, weight=1)
        ttk.Label(review_row, text="Numer GT").grid(row=0, column=0, padx=(0, 8))
        self.gt_entry = ttk.Entry(review_row, textvariable=self.gt_var)
        self.gt_entry.grid(row=0, column=1, sticky="ew")
        commands = ttk.Frame(review_row)
        commands.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.decision_buttons = []
        for column, (text, status) in enumerate((("Zapisz szkic", "draft"), ("Potwierdź GT", "confirmed"), ("Odrzuć", "rejected"))):
            button = ttk.Button(commands, text=text, command=lambda value=status: self.decide(value))
            button.grid(row=0, column=column, padx=(0, 6))
            self.decision_buttons.append(button)
        details_frame = ttk.Frame(right)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=(10, 0), pady=(6, 0))
        details_frame.grid_columnconfigure(0, weight=1)
        details_frame.grid_rowconfigure(0, weight=1)
        self.details = tk.Text(details_frame, wrap="word", height=9, state="disabled", font=("Consolas", 10))
        self.details.grid(row=0, column=0, sticky="nsew")
        details_scroll = ttk.Scrollbar(details_frame, command=self.details.yview)
        details_scroll.grid(row=0, column=1, sticky="ns")
        self.details.configure(yscrollcommand=details_scroll.set)

        footer = ttk.Frame(self, padding=10)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ttk.Label(footer, textvariable=self.status_var, wraplength=600)
        self.status_label.grid(row=0, column=0, sticky="w")
        self.export_button = ttk.Button(footer, text="Zatwierdzone do anotacji", command=self.export)
        self.export_button.grid(row=0, column=1, padx=(10, 0))
        footer.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(200, event.width - 230)))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.set_busy(False)

    def set_busy(self, busy):
        self.busy = busy
        self.open_button.configure(state="disabled" if busy else "normal")
        state = "normal" if self.current is not None and not busy else "disabled"
        self.gt_entry.configure(state=state)
        for button in self.decision_buttons:
            button.configure(state=state)
        confirmed = self.review and any(item["status"] == "confirmed" for item in self.review.decisions.values())
        self.export_button.configure(state="normal" if confirmed and not busy else "disabled")

    def run_job(self, action, done):
        self.set_busy(True)

        def worker():
            try:
                self.jobs.put((True, action()))
            except Exception as exc:
                self.jobs.put((False, exc))

        def poll():
            try:
                ok, result = self.jobs.get_nowait()
            except queue.Empty:
                self.after(60, poll)
                return
            self.set_busy(False)
            if ok:
                try:
                    done(result)
                except Exception as exc:
                    messagebox.showerror("Akwizycja Androida", str(exc), parent=self)
            else:
                self.status_var.set("Operacja nie powiodła się")
                messagebox.showerror("Akwizycja Androida", str(result), parent=self)

        threading.Thread(target=worker, daemon=True).start()
        self.after(60, poll)

    def save_draft(self):
        if not self.current:
            return True
        text = self.gt_var.get()
        previous = self.review.decisions.get(self.current.plate_id, {"gt": ""})
        if text.upper() == previous["gt"]:
            return True
        try:
            self.review.set_decision(self.current.plate_id, text, "draft")
            if self.tree.exists(self.current.plate_id):
                self.tree.set(self.current.plate_id, "status", "Szkic")
            self.update_status()
            self.set_busy(False)
            return True
        except Exception as exc:
            messagebox.showerror("Zapis oceny", str(exc), parent=self)
            return False

    def open_file(self):
        if self.busy or not self.save_draft():
            return
        path = filedialog.askopenfilename(parent=self, title="Paczka Akwizycji Androida",
                                         filetypes=[("Paczki ZIP", "*.zip *.alprsession"), ("Wszystkie pliki", "*.*")])
        if not path:
            return
        root = self.host._get_step3_chars_root_dir()

        def load():
            session = read_crop_session(Path(path))
            return session, CropReview(session, root / "acquisition_reviews" / f"{session.archive_sha256}.json")

        self.status_var.set("Sprawdzanie paczki i obrazów...")
        self.run_job(load, self.loaded)

    def loaded(self, result):
        self.session, self.review = result
        self.current = None
        self.source_image = None
        self.gt_var.set("")
        self.path_var.set(str(self.session.path))
        self.filter_var.set("")
        self.refresh_list()
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
        self.update_status()

    def refresh_list(self):
        if self.session is None:
            return
        selected = self.current.plate_id if self.current else ""
        self.tree.delete(*self.tree.get_children())
        query = self.filter_var.get().upper()
        labels = {"draft": "Szkic", "confirmed": "GT", "rejected": "Odrzucona"}
        for group in self.session.groups:
            if query and query not in group.key:
                continue
            decision = self.review.decisions.get(group.plate_id, {})
            self.tree.insert("", "end", iid=group.plate_id,
                             values=(group.key, group.observation_count, labels.get(decision.get("status"), "Do oceny")))
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)

    def select_group(self, _event=None):
        selection = self.tree.selection()
        if self.busy or not selection or (self.current and self.current.plate_id == selection[0]):
            return
        if not self.save_draft():
            if self.current and self.tree.exists(self.current.plate_id):
                self.tree.selection_set(self.current.plate_id)
            return
        group = next(group for group in self.session.groups if group.plate_id == selection[0])
        try:
            source = self.session.read_image(group)
        except Exception as exc:
            messagebox.showerror("Obraz Akwizycji", str(exc), parent=self)
            return
        self.current = group
        self.source_image = source
        self.gt_var.set(self.review.decisions.get(group.plate_id, {}).get("gt", ""))
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", json.dumps({"session_id": self.session.session_id,
                            "started_at_ms": self.session.manifest.get("started_at_ms"),
                            "base_image": group.base["image"], "crops": group.crops}, ensure_ascii=False, indent=2))
        self.details.configure(state="disabled")
        self.render_image()
        self.set_busy(False)

    def render_image(self):
        self.canvas.delete("all")
        if self.source_image is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(max(1, width - 20) / self.source_image.width, max(1, height - 20) / self.source_image.height)
        size = (max(1, round(self.source_image.width * scale)), max(1, round(self.source_image.height * scale)))
        self.photo = ImageTk.PhotoImage(self.source_image.resize(size, Image.Resampling.LANCZOS), master=self)
        left, top = (width - size[0]) / 2, (height - size[1]) / 2
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        if self.boxes_var.get() and self.current:
            for character in self.current.base["characters"]:
                self.canvas.create_rectangle(left + character["left"] * size[0], top + character["top"] * size[1],
                                             left + character["right"] * size[0], top + character["bottom"] * size[1],
                                             outline="#e73463", width=2)

    def decide(self, status):
        if self.busy or self.current is None:
            return
        try:
            self.review.set_decision(self.current.plate_id, self.gt_var.get(), status)
            self.gt_var.set(self.review.decisions[self.current.plate_id]["gt"])
        except Exception as exc:
            messagebox.showerror("Ocena Akwizycji", str(exc), parent=self)
            return
        self.refresh_list()
        self.update_status()
        self.set_busy(False)

    def update_status(self):
        confirmed = sum(item["status"] == "confirmed" for item in self.review.decisions.values())
        self.status_var.set(f"Sesja {self.session.session_id}: {len(self.session.groups)} grup, GT: {confirmed}. "
                            "Tylko niepuste odczyty MZ; jakość całego pipeline'u niedostępna.")

    def export(self):
        if self.busy or not self.save_draft():
            return
        root = self.host._get_step3_chars_root_dir()
        destination = root / ("mobile_crops_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        try:
            snapshot = CropReview(self.session, self.review.path)
        except Exception as exc:
            messagebox.showerror("Ocena Akwizycji", str(exc), parent=self)
            return
        self.status_var.set("Przygotowywanie materiału do anotacji...")

        def opened(path):
            self.host._reset_extract_source_inputs()
            if self.host._open_detection_subtab_with_preview(path, force_reload=True):
                self.status_var.set(f"Przekazano do anotacji: {path}")
                self.host._persist_step3_extract_state()
                self.lower()
            else:
                self.status_var.set(f"Zapisano materiał: {path}")
                messagebox.showerror("Anotacja", f"Materiał zapisano w {path}, ale edytor nie został otwarty.", parent=self)

        self.run_job(lambda: export_annotation_preview(snapshot, destination), opened)

    def close(self):
        if not self.busy and self.save_draft():
            self.destroy()
