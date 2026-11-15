"""Z4MOBREV031: accessible scrollable Tk dashboard over canonical ALPR results.

Only presentation: there are no writes to sessions, GT, metrics or sidecars.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from .mobile_science_viewmodel import STATUS_DISPLAY, make_science_dashboard
from .mobile_canvas_ui import palette, Disclosure
from ..ranking.mobile_canvas_presentation import summary_caption


class MobileScienceSummaryPanel:
    """One scroll area (including the table) to avoid nested wheel scrolling."""

    def __init__(self, parent, *, background: str, surface: str, foreground: str,
                 muted: str, accent: str, border: str,
                 on_review: Callable[[], None] | None = None):
        self.ui_p = palette(parent, background)
        background, surface = self.ui_p["bg"], self.ui_p["surface"]
        foreground, muted = self.ui_p["fg"], self.ui_p["muted"]
        accent, border = self.ui_p["blue"], self.ui_p["border"]
        self.background, self.surface = background, surface
        self.foreground, self.muted = foreground, muted
        self.accent, self.border = accent, border
        self.on_review = on_review
        self.model = None
        self._sections_open = {"why": False, "caveats": False,
                               "methods": False, "source": False}
        self.frame = tk.Frame(parent, bg=background)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.frame, bg=background, bd=0, highlightthickness=0,
                                takefocus=True, yscrollincrement=22)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.body = tk.Frame(self.canvas, bg=background)
        self.item = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", self._fit_body_width, add="+")
        self.body.bind("<Configure>", self._refresh_scroll_region, add="+")
        self._canvas_args = None
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.body)
        for key, amount in (("<Up>", -1), ("<Down>", 1),
                            ("<Prior>", -8), ("<Next>", 8)):
            self.canvas.bind(key, lambda _event, step=amount: self.canvas.yview_scroll(step, "units"), add="+")
        self.tree = None
        self._style_name = "MobileScience031.Treeview"
        try:
            style = ttk.Style(parent)
            style.configure(self._style_name, rowheight=29,
                            font=("Segoe UI", 9), background=surface,
                            foreground=foreground, fieldbackground=surface)
            style.configure(self._style_name + ".Heading", font=("Segoe UI", 9, "bold"))
            style.map(self._style_name, background=[("selected", "#DDEBF5")],
                      foreground=[("selected", "#17354C")])
        except tk.TclError:
            pass

    def _fit_body_width(self, event=None):
        width = self.canvas.winfo_width() if event is None else event.width
        if width > 1:
            self.canvas.itemconfigure(self.item, width=width)
        self._refresh_scroll_region()

    def _refresh_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_wheel(self, widget):
        widget.bind("<MouseWheel>", self._wheel, add="+")
        widget.bind("<Button-4>", self._wheel, add="+")
        widget.bind("<Button-5>", self._wheel, add="+")

    def _wheel(self, event):
        direction = (-1 if getattr(event, "num", None) == 4 else
                     1 if getattr(event, "num", None) == 5 else
                     -1 if int(getattr(event, "delta", 0) or 0) > 0 else 1)
        if getattr(event, "delta", None) == 0 and getattr(event, "num", None) not in (4, 5):
            return None
        self.canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _bind_descendants(self, root):
        for child in root.winfo_children():
            self._bind_wheel(child)
            self._bind_descendants(child)

    def _text(self, parent, text: str, *, size=10, bold=False, muted=False, bg=None,
              wrap=800, padx=0, pady=0, anchor="w"):
        w = tk.Label(parent, text=text, bg=bg or self.background,
                     fg=self.muted if muted else self.foreground,
                     font=("Segoe UI", size, "bold" if bold else "normal"),
                     justify="left", anchor=anchor, wraplength=wrap, padx=padx, pady=pady)
        parent.bind("<Configure>", lambda e: w.configure(wraplength=max(120,e.width-24))
                    if w.winfo_exists() else None, add="+")
        return w

    def _section_title(self, text):
        w = self._text(self.body, text, size=10, bold=True, padx=0)
        w.pack(fill="x", padx=18, pady=(13, 5))
        return w
        return w

    def _card(self, host, card: dict, row: int, col: int):
        box = tk.Frame(host, bg=self.surface, highlightbackground=self.border,
                       highlightthickness=1, padx=11, pady=9)
        box.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        self._text(box, card["heading"], bold=True, size=10, bg=self.surface,
                   wrap=370).pack(fill="x")
        value_label = self._text(box, card["value"], size=19, bold=True, bg=self.surface, wrap=370)
        value_label.configure(fg=self.ui_p["blue"])
        value_label.pack(fill="x", pady=(6, 2))
        label = card["reason"]
        self._text(box, label, size=9, muted=True, bg=self.surface, wrap=370).pack(fill="x")
        return box

    def _table(self, model, parent=None):
        rows = model["rows"]
        columns = ("group", "metric", "value", "sample", "status", "reason")
        heading = ("Obszar", "Miara", "Wynik", "Próba", "Stan", "Wyjaśnienie")
        wrapper = tk.Frame(parent or self.body, bg=self.background)
        wrapper.pack(fill="x", padx=4 if parent else 18)
        self.tree = ttk.Treeview(wrapper, columns=columns, show="headings",
                                 selectmode="browse", style=self._style_name,
                                 height=max(1, len(rows)), takefocus=True)
        self.tree.pack(fill="x")
        for key, name in zip(columns, heading):
            self.tree.heading(key, text=name, anchor="w")
        palette = {
            "AVAILABLE": ("#E8F2E9", "#214B2A"),
            "PRELIMINARY": ("#F9F0DA", "#5F4B21"),
            "DIAGNOSTIC": ("#E7EEF4", "#264865"),
            "MISSING": ("#F1F3F4", "#4C5961"),
            "NOT_EVALUABLE": ("#F5E9E9", "#703C3C"),
            "SOURCE_UNVERIFIED": ("#F5E9E9", "#703C3C"),
            "NOT_RUN": ("#F1F3F4", "#4C5961"),
            "INVALID": ("#F5E9E9", "#703C3C"),
        }
        for status, (bg, fg) in palette.items():
            self.tree.tag_configure(status, background=bg, foreground=fg)
        for entry in rows:
            if entry.status == "SOURCE_UNVERIFIED":
                sample = "—"
            elif entry.denominator:
                sample = f"{entry.denominator} · {entry.unit}"
            elif entry.metric_id in {"pose_iou", "pose_nmce"}:
                sample = "0 par GT"
            elif entry.metric_id == "mz_cer" and entry.denominator is not None:
                sample = f"{entry.denominator} znaków GT"
            else:
                sample = "—"
            self.tree.insert("", "end", iid=entry.metric_id, tags=(entry.status,),
                             values=(entry.group, entry.label, entry.formatted_value,
                                     sample, STATUS_DISPLAY.get(entry.status, entry.status),
                                     entry.reason))
        self._size_columns()
        wrapper.bind("<Configure>", lambda _e: self._size_columns(), add="+")
        explanation = self._text(wrapper, "Kliknij wiersz, aby zobaczyć pełne wyjaśnienie.",
                                 size=9, muted=True, wrap=980)
        explanation.pack(fill="x", pady=(4, 2))
        index = {entry.metric_id: entry for entry in rows}
        def select(_event=None):
            selected = self.tree.selection()
            if not selected:
                return
            entry = index.get(selected[0])
            if entry is not None:
                explanation.configure(text=f"{entry.label}: {entry.reason}  •  Źródło: {entry.source_path}")
        self.tree.bind("<<TreeviewSelect>>", select, add="+")
        self._table_description = explanation

    def _size_columns(self):
        if not self.tree or not self.tree.winfo_exists():
            return
        width = self.tree.winfo_width()
        if width <= 1:
            width = self.canvas.winfo_width() - 45
        if width <= 1:
            width = 1080
        shares = (("group", .075, 64), ("metric", .195, 140), ("value", .120, 110),
                  ("sample", .180, 140), ("status", .165, 130), ("reason", .265, 200))
        total = max(735, width - 10)
        for column, weight, minimum in shares:
            self.tree.column(column, width=max(minimum, int(total * weight)),
                             minwidth=minimum, stretch=True, anchor="w")

    def _progress(self, host, item, col: int):
        cell = tk.Frame(host, bg=self.surface, padx=10, pady=8,
                        highlightthickness=1, highlightbackground=self.border)
        cell.grid(row=0, column=col, sticky="nsew", padx=4)
        heading = tk.Frame(cell, bg=self.surface)
        heading.pack(fill="x")
        self._text(heading, item["label"], size=9, bold=True,
                   bg=self.surface, wrap=440).pack(side="left", anchor="w")
        self._text(heading, item["text"], size=9, bg=self.surface).pack(side="right", anchor="e")
        track = tk.Canvas(cell, bg=self.surface, height=13, bd=0, highlightthickness=0)
        track.pack(fill="x", pady=(5, 2))
        ratio = (max(0, min(1, item["done"] / item["total"]))
                 if item["done"] is not None and item["total"] else 0.0)
        def draw(event):
            track.delete("all")
            width = max(1, event.width)
            track.create_rectangle(0, 0, width, 11, outline=self.border, fill="#E5E9EB")
            if ratio > 0:
                track.create_rectangle(1, 1, width * ratio, 10, outline="", fill=self.ui_p["blue"])
        track.bind("<Configure>", draw)

    def _expand(self, key, name, details: list[str]):
        def changed(opened):
            self._sections_open[key] = opened
            self._refresh_scroll_region()
        row = Disclosure(self.body, name, p=self.ui_p,
                         opened=self._sections_open[key], command=changed)
        row.pack(fill="x", padx=18, pady=4)
        for line in details or ["Brak dodatkowych danych."]:
            item = self._text(row.body, str(line), size=10, bg=self.surface, wrap=950)
            item.pack(fill="x", anchor="w", pady=4)
            item.bind("<Configure>", lambda e,w=item: w.configure(wraplength=max(120,e.width-10)))
        return row

    def render_canvas(self, data, inventory, *, refreshing=False, review_status="IN_PROGRESS"):
        self._canvas_args = (data, inventory, refreshing, review_status)
        if not hasattr(self, "canvas_caption") or not self.canvas_caption.winfo_exists():
            return
        from ..ranking.mobile_review_status import approval_caption
        state = approval_caption(inventory, review_status, (data or {}).get('review_revision','?'))
        self.canvas_title.configure(text="Ocena operatora (Canvas · asystowana) · " + state)
        text = ("Aktualizowanie oceny Canvas…" if refreshing else
                summary_caption(data, inventory) if data else
                "Ocena Canvas niedostępna dla tej rewizji")
        self.canvas_caption.configure(text=text)

    def render(self, report: dict, form: dict | None = None, *, source_verified=True):
        model = make_science_dashboard(report, form, source_verified=source_verified)
        preserved = self.canvas.yview()[0]
        for child in self.body.winfo_children():
            child.destroy()
        self.model = model
        banner = tk.Frame(self.body, bg=self.ui_p["blue_bg"], padx=10, pady=8)
        banner.pack(fill="x", padx=18, pady=(10, 3))
        self.canvas_title = self._text(banner, "Ocena operatora (Canvas · asystowana)",
                                      bold=True, bg=self.ui_p["blue_bg"], wrap=1000)
        self.canvas_title.pack(fill="x")
        self.canvas_caption = self._text(banner, "Ocena Canvas niedostępna dla tej rewizji",
                                        bg=self.ui_p["blue_bg"], wrap=1000)
        self.canvas_caption.pack(fill="x", pady=(3, 0))
        for item in (self.canvas_title, self.canvas_caption):
            item.bind("<Configure>", lambda e, w=item: w.configure(wraplength=max(120, e.width - 8)))
        if self._canvas_args:
            data, inventory, refreshing, status = self._canvas_args
            if data and (not source_verified or any(data.get(k) != report.get(k)
                    for k in ("session_id", "source_archive_sha256", "review_revision"))):
                data, refreshing = None, True
            self.render_canvas(data, inventory, refreshing=refreshing, review_status=status)
        independent_missing = not model["pose_pair_count"] and not model["mz_reviewed_count"]
        explanation = ("Osobnej oceny niezależnej nie przeprowadzono. "
                       "Wyniki weryfikacji operatora są dostępne powyżej."
                       if independent_missing else
                       "Poniższy raport historyczny używa odrębnego protokołu. "
                       "Ocena asystowana Canvas nie jest niezależną próbą ślepą.")
        info = self._text(self.body, explanation, size=9, muted=True, wrap=1000)
        info.pack(fill="x", padx=19, pady=(3, 7))
        info.bind("<Configure>", lambda e: info.configure(wraplength=max(120, e.width - 8)))
        header = tk.Frame(self.body, bg=self.background)
        header.pack(fill="x", padx=18, pady=(10, 4))
        header.columnconfigure(0, weight=1)
        status_caption=self._text(header, model["form_status"] + "  ·  " + model["science_status"] +
                   "  ·  " + model["review_mode"], bold=True, size=10, wrap=1070)
        status_caption.grid(row=0, column=0, sticky="ew")
        self.conditions_caption=self._text(header, model["conditions"], size=9, muted=True,wrap=1070)
        self.conditions_caption.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        for caption in (status_caption,self.conditions_caption):
            # The frame owns width. Text must not resize itself repeatedly
            # through its own Configure event while the tab is being built.
            caption.configure(width=1)
        review_button=None
        if self.on_review is not None:
            review_button=tk.Button(header, text="← Weryfikacja", command=self.on_review,
                      bg=self.ui_p["blue_bg"], fg=self.ui_p["blue"],
                      activebackground=self.ui_p["blue_bg"], activeforeground=self.ui_p["blue"],
                      relief="flat", bd=0, cursor="hand2", padx=12, pady=7,
                      font=("Segoe UI", 9, "bold"))
            review_button.grid(row=0,column=1,rowspan=2,sticky="ne",padx=(12,0))
        def fit_header(event):
            reserved=review_button.winfo_reqwidth()+12 if review_button is not None else 0
            wrap=max(120,event.width-reserved-8)
            for caption in (status_caption,self.conditions_caption):
                if int(float(caption.cget('wraplength')))!=wrap:caption.configure(wraplength=wrap)
        header.bind('<Configure>',fit_header)
        self.history_progress = Disclosure(self.body, "01  Postęp protokołu historycznego · osobna ocena", p=self.ui_p,
            opened=self._sections_open.get("history_progress", not independent_missing),
            command=lambda opened: self._sections_open.update(history_progress=opened))
        self.history_progress.pack(fill="x", padx=18, pady=(6, 7))
        progress_box = tk.Frame(self.history_progress.body, bg=self.background)
        progress_box.pack(fill="x", padx=14)
        progress_box.columnconfigure(0, weight=1, uniform="science_progress")
        progress_box.columnconfigure(1, weight=1, uniform="science_progress")
        for i, item in enumerate(model["progress"]):
            self._progress(progress_box, item, i)
        gt_info = (f"Niezależne porównania Pose: {model['pose_pair_count']}" if model["pose_pair_count"] is not None else
                   "Niezależne porównania Pose: brak danych")
        mz_info = (f"Odczyty MZ z GT: {model['mz_reviewed_count']}" if model["mz_reviewed_count"] is not None else
                   "Odczyty MZ z GT: brak danych")
        self._text(self.history_progress.body, gt_info + "    ·    " + mz_info, size=9, muted=True).pack(fill="x", padx=9, pady=(7, 2))
        historical = Disclosure(self.body, "02  Wyniki protokołu historycznego · osobna ocena", p=self.ui_p,
                                opened=self._sections_open.get("historical", not independent_missing),
                                command=lambda opened: self._sections_open.update(historical=opened))
        historical.pack(fill="x", padx=18, pady=(2, 7))
        top = tk.Frame(historical.body, bg=self.background)
        top.pack(fill="x", padx=14, pady=(2, 7))
        top.columnconfigure(0, weight=1, uniform="science_card")
        top.columnconfigure(1, weight=1, uniform="science_card")
        for i, card in enumerate(model["cards"]):
            self._card(top, card, i // 2, i % 2)
        table_section = Disclosure(self.body, "Szczegółowe miary i wiarygodność", p=self.ui_p,
                                   opened=self._sections_open.get("metrics", False),
                                   command=lambda opened: self._sections_open.update(metrics=opened))
        table_section.pack(fill="x", padx=18, pady=(6, 7))
        self._table(model, parent=table_section.body)
        times_heading = self._section_title("CZASY TELEFONU • NIEZALEŻNE OD GT")
        times_heading.pack_configure(before=self.history_progress)
        times = tk.Frame(self.body, bg=self.background)
        times.pack(fill="x", padx=14, before=self.history_progress)
        for i, item in enumerate(model["latencies"]):
            times.columnconfigure(i, weight=1, uniform="science_times")
            cell = tk.Frame(times, bg=self.surface, padx=10, pady=7,
                            highlightbackground=self.border, highlightthickness=1)
            cell.grid(row=0, column=i, sticky="nsew", padx=4)
            tk.Frame(cell, bg=self.ui_p[("blue", "green", "purple")[i % 3]], height=3).pack(fill="x",pady=(0,7))
            self._text(cell, item["label"], bg=self.surface, size=9,
                       bold=True, wrap=320).pack(fill="x")
            self._text(cell, item["value"], bg=self.surface, size=9,
                       wrap=320).pack(fill="x", pady=(4, 0))
        self._text(self.body, "Czasy z raportu telefonu (p50, P95, n), nie pomiary kamera→ekran.",
                   size=9, muted=True).pack(fill="x", padx=19, pady=(5, 7), before=self.history_progress)
        self._section_title("SZCZEGÓŁY BADANIA")
        self._expand("why", "Braki danych i status wyników", model["why"])
        self._expand("caveats", "Ograniczenia", model["caveats"])
        self._expand("methods", "Miary i metodyka", model["methods"])
        self._expand("source", "Źródła danych", [f"{k}: {v}" for k, v in model["source"]])
        self._text(self.body, "Raport prezentacyjny nie zastępuje pełnej tabeli technicznej ani źródłowego eksportu.",
                   size=9, muted=True).pack(fill="x", padx=18, pady=(6, 17))
        self._bind_descendants(self.body)
        self._refresh_scroll_region()
        if preserved > 0:
            self.canvas.after_idle(lambda: self.canvas.yview_moveto(preserved)
                                   if self.canvas.winfo_exists() else None)
        return model
