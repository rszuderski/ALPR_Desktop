"""Export configurator UI only: local theme, role pages and operation feedback.

No filesystem/model I/O, calibration, conversion, approval or journal writes.
The existing exporter and preflight own all validation decisions.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

ROLE_NAMES = {"MP": "Pojazdy", "MT": "Tablice", "MZ": "Znaki"}


def palette_for(widget: tk.Misc, background: str) -> dict[str, str]:
    try:
        rgb = widget.winfo_rgb(background)
        dark = sum(rgb) / 3 < 32768
    except tk.TclError:
        dark = False
    if dark:
        return dict(bg="#192330", card_bg="#222f3e", fg="#f1f5fa", muted="#bcc9d8",
                    border="#455770", accent="#7fc4ff", success="#87d7b1",
                    warning="#f4ce83", error="#ff9da4", select="#24466d")
    return dict(bg="#f0f4f8", card_bg="#ffffff", fg="#182b43", muted="#475d75",
                border="#c9d5e2", accent="#245c90", success="#216446",
                warning="#84580c", error="#a02c39", select="#e2eef9")


def role_plan_text(state: dict) -> str:
    """Safe, non-authoritative preview even while a numeric field is being edited."""
    marker = str(state.get("marker") or "?")
    def value(key, default=""):
        obj = state.get(key)
        if obj is None:
            return default
        try:
            return obj.get()
        except (tk.TclError, ValueError, TypeError):
            return None
    reuse = "reuse_" + marker.lower()
    if value(reuse, False):
        path = str(value(reuse + "_path") or "").replace("\\", "/")
        return f"{marker}: z biblioteki · {str(value(reuse + '_precision') or '?').upper()} · {path.rsplit('/', 1)[-1] if path else 'wybierz paczkę'}"
    labels = (("litert", "TFLite FP32"), ("int8", "TFLite INT8"),
              ("onnx", "ONNX FP32"), ("onnx_int8", "ONNX INT8"), ("ncnn", "NCNN FP32"))
    formats = ", ".join(label for key, label in labels if value(key, False)) or "wybierz format"
    size = value("imgsz")
    return f"{marker}: nowy eksport · {formats} · wejście {size if size is not None else '?'} px"


class ScrolledPage:
    """Independent wheel area; spinboxes/comboboxes do not change via wheel."""
    def __init__(self, parent, colors):
        self.shell = tk.Frame(parent, bg=colors["bg"])
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.shell, bg=colors["bg"], bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set, yscrollincrement=24)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.body = tk.Frame(self.canvas, bg=colors["bg"])
        self.body.grid_columnconfigure(0, weight=1)
        self.window_id = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", self._resize, add="+")
        self.body.bind("<Configure>", self._region, add="+")
        self._fraction = 0.0

    def _region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize(self, event):
        self.canvas.itemconfigure(self.window_id, width=max(1, event.width))
        self._region()

    def wheel(self, event):
        number = getattr(event, "num", None)
        if number in (4, 5):
            steps = -1 if number == 4 else 1
        else:
            delta = float(getattr(event, "delta", 0) or 0)
            self._fraction += -delta / 120.0
            steps = math.trunc(self._fraction)
            self._fraction -= steps
        if steps and self.body.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(steps * 2, "units")
        return "break"

    def bind_wheel(self, widget=None):
        widget = self.body if widget is None else widget
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self.wheel)  # precedes native Spinbox bindings
        for child in widget.winfo_children():
            self.bind_wheel(child)


class ExportConfiguratorView:
    def __init__(self, window, title):
        self.window = window
        self.colors = palette_for(window, window.cget("background"))
        c = self.colors
        self.font_body = tkfont.Font(root=window, family="Segoe UI", size=11)
        self.font_small = tkfont.Font(root=window, family="Segoe UI", size=10)
        self.font_bold = tkfont.Font(root=window, family="Segoe UI", size=11, weight="bold")
        self.font_title = tkfont.Font(root=window, family="Segoe UI", size=16, weight="bold")
        self.style_prefix = f"Z4Export{id(self)}"
        self.style = ttk.Style(window)
        for cls in ("TButton", "TCheckbutton", "TRadiobutton", "TLabel", "TEntry", "TSpinbox", "TCombobox", "TNotebook.Tab"):
            self.style.configure(self.style_prefix + "." + cls, font=self.font_body,
                                 padding=(8, 5) if cls in ("TButton", "TNotebook.Tab") else (2, 3))
        for cls in ("TCheckbutton", "TRadiobutton", "TLabel"):
            self.style.configure(self.style_prefix + "." + cls, foreground=c["fg"], background=c["card_bg"])
        self.style.configure(self.style_prefix + ".TNotebook", background=c["bg"])
        self.root = tk.Frame(window, bg=c["bg"], padx=14, pady=12)
        self.root.pack(fill="both", expand=True)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        tk.Label(self.root, text=title, font=self.font_title, bg=c["bg"], fg=c["fg"], anchor="w").grid(row=0, column=0, sticky="ew")
        self.intro = tk.Frame(self.root, bg=c["bg"])
        self.intro.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        self.intro.grid_columnconfigure(0, weight=1)
        tk.Label(self.intro, text="Ustaw modele → sprawdź gotowość → eksportuj. Kontrola PC nie jest testem na telefonie.",
                 bg=c["bg"], fg=c["muted"], font=self.font_small, anchor="w").grid(row=0, column=0, sticky="ew")
        self.notebook = ttk.Notebook(self.root, style=self.style_prefix + ".TNotebook")
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self.pages = {}
        self._locked = {}
        self._busy = False
        self._geometry_job = None
        self.phase = tk.StringVar(master=window, value="USTAWIENIA · eksport nieuruchomiony")
        self.plan = tk.StringVar(master=window, value="")
        self._status_trace = None
        self._progress_trace = None
        self.install_running = False
        window.bind("<Destroy>", self._destroy, add="+")

    def add_role(self, marker):
        page = ScrolledPage(self.notebook, self.colors)
        self.notebook.add(page.shell, text=f"{marker} · {ROLE_NAMES.get(marker, marker)}")
        self.pages[marker] = page
        return page.body

    def add_checks(self):
        c = self.colors
        self.checks = tk.Frame(self.notebook, bg=c["bg"], padx=12, pady=10)
        self.notebook.add(self.checks, text="Kontrola eksportu")
        return self.checks

    def show_checks(self):
        self.notebook.select(self.checks)

    def prepare_footer(self, status_var, progress_var, percent_var):
        c = self.colors
        self.footer = tk.Frame(self.root, bg=c["bg"])
        self.footer.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        self.footer.grid_columnconfigure(0, weight=1)
        self.phase_label = tk.Label(self.footer, textvariable=self.phase, font=self.font_bold,
                                   bg=c["bg"], fg=c["accent"], anchor="w")
        self.phase_label.grid(row=0, column=0, sticky="ew")
        self.plan_label = tk.Label(self.footer, textvariable=self.plan, font=self.font_small,
                                  bg=c["bg"], fg=c["muted"], justify="left", anchor="w")
        self.plan_label.grid(row=1, column=0, sticky="ew", pady=(3, 2))
        status_host = tk.Frame(self.footer, bg=c["bg"])
        status_host.grid(row=2, column=0, sticky="ew")
        status_host.grid_columnconfigure(0, weight=1)
        self.status_text = tk.Text(status_host, height=2, width=1, wrap="word", font=self.font_small,
                                   bg=c["bg"], fg=c["muted"], bd=0, highlightthickness=0, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="ew")
        bar = ttk.Scrollbar(status_host, orient="vertical", command=self.status_text.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=bar.set)
        def status_changed(*_):
            self.status_text.configure(state="normal")
            self.status_text.delete("1.0", "end")
            self.status_text.insert("1.0", str(status_var.get()))
            self.status_text.configure(state="disabled")
        self._status_var = status_var
        self._status_trace = status_var.trace_add("write", status_changed)
        status_changed()
        self._progress_var = progress_var
        def progress_changed(*_):
            if self.install_running:
                try:
                    value = float(progress_var.get())
                    if math.isfinite(value):
                        percent_var.set(f"{max(0.0, min(100.0, value)):.0f}%")
                except (ValueError, TypeError, tk.TclError):
                    pass
        self._progress_trace = progress_var.trace_add("write", progress_changed)
        self.progress_row = tk.Frame(self.footer, bg=c["bg"])
        self.progress_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.progress_row.grid_columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(self.progress_row, maximum=100, variable=progress_var)
        self.progress.grid(row=0, column=0, sticky="ew")
        tk.Label(self.progress_row, textvariable=percent_var, font=self.font_bold, bg=c["bg"], fg=c["fg"], width=6).grid(row=0, column=1)
        self.progress_row.grid_remove()
        self.actions = tk.Frame(self.root, bg=c["bg"])
        self.actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.action_buttons = []
        self.actions.bind("<Configure>", self._queue_geometry, add="+")
        self.root.bind("<Configure>", self._queue_geometry, add="+")
        return self.footer, self.status_text, self.progress_row, self.progress, self.actions

    def _queue_geometry(self, _event=None):
        if self._geometry_job is None:
            self._geometry_job = self.window.after_idle(self._arrange)

    def _arrange(self):
        self._geometry_job = None
        width = max(1, self.root.winfo_width() - 30)
        self.plan_label.configure(wraplength=width)
        self.phase_label.configure(wraplength=width)
        available = max(1, self.actions.winfo_width())
        for child in self.action_buttons:
            child.grid_forget()
        shown = [b for b in self.action_buttons if not getattr(b, "_hide_when_idle", False) or getattr(self, "install_running", False)]
        row = column = used = 0
        for button in shown:
            need = button.winfo_reqwidth() + 10
            if used and used + need > available:
                row += 1; column = used = 0
            button.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=(0, 3))
            column += 1; used += need

    def feedback(self, *, running=False, installing=False, ready=False, problems=False, completed=False, failed=False):
        busy = bool(running or installing)
        self.install_running = bool(installing)
        self.phase.set("INSTALACJA ZALEŻNOŚCI" if installing else "EKSPORT W TOKU" if running else
                       "BŁĄD EKSPORTU · przeczytaj komunikat" if failed else
                       "EKSPORT ZAKOŃCZONY · plik zapisany" if completed else
                       "WYMAGANA KOREKTA · eksport nieuruchomiony" if problems else
                       "GOTOWOŚĆ SPRAWDZONA · eksport nieuruchomiony" if ready else
                       "USTAWIENIA · eksport nieuruchomiony")
        if hasattr(self, "progress_row"):
            self.progress_row.grid() if busy or completed or failed else self.progress_row.grid_remove()
            self._queue_geometry()
        self.lock_inputs(busy)

    def lock_inputs(self, busy):
        if busy == self._busy:
            return
        self._busy = busy
        roots = [self.intro] + [p.body for p in self.pages.values()]
        if busy:
            for root in roots:
                for widget in self.walk(root):
                    if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Spinbox, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                        self._locked[widget] = tuple(widget.state())
                        widget.state(["disabled"])
        else:
            for widget, states in self._locked.items():
                try:
                    widget.state(["!disabled", "!readonly"])
                    widget.state(states)
                except tk.TclError:
                    pass
            self._locked.clear()

    @staticmethod
    def walk(root):
        yield root
        for child in root.winfo_children():
            yield from ExportConfiguratorView.walk(child)

    def style_widgets(self, root):
        for widget in self.walk(root):
            try:
                cls = widget.winfo_class()
                if isinstance(widget, ttk.Widget) and cls in ("TButton", "TCheckbutton", "TRadiobutton", "TLabel", "TEntry", "TSpinbox", "TCombobox"):
                    style_name = self.style_prefix + "." + cls
                    if cls in ("TLabel", "TCheckbutton", "TRadiobutton"):
                        parent_bg = widget.master.cget("background") if isinstance(widget.master, tk.Widget) and not isinstance(widget.master, ttk.Widget) else self.colors["card_bg"]
                        style_name = self.style_prefix + "." + str(parent_bg).replace("#", "") + "." + cls
                        self.style.configure(style_name, font=self.font_body, background=parent_bg, foreground=self.colors["fg"])
                        self.style.map(style_name, background=[("active", parent_bg), ("!disabled", parent_bg)])
                    widget.configure(style=style_name)
                elif isinstance(widget, tk.Label):
                    old = tkfont.Font(root=self.window, font=widget.cget("font"))
                    if abs(int(old.actual("size"))) < 10:
                        widget.configure(font=self.font_bold if old.actual("weight") == "bold" else self.font_small)
                    if widget.winfo_fpixels(widget.cget("wraplength")) > 0 or len(str(widget.cget("text"))) > 60:
                        widget.configure(wraplength=600)
                        widget.bind("<Configure>", lambda e, w=widget: w.configure(wraplength=max(30, e.width - 8)), add="+")
            except tk.TclError:
                pass
        for page in self.pages.values():
            page.bind_wheel()

    def foldout(self, parent, variables):
        c = self.colors
        frame = tk.Frame(parent, bg=c["card_bg"])
        body = tk.Frame(frame, bg=c["card_bg"])
        body.grid_columnconfigure(0, weight=1, uniform="other_formats")
        body.grid_columnconfigure(1, weight=1, uniform="other_formats")
        frame.grid_columnconfigure(0, weight=1)
        opened = tk.BooleanVar(master=self.window, value=False)
        def render(*_):
            count = sum(bool(v.get()) for v in variables)
            button.configure(text=("▾ " if opened.get() else "▸ ") + f"Inne formaty · ONNX / NCNN (wybrane: {count})")
            body.grid(row=1, column=0, sticky="ew") if opened.get() else body.grid_remove()
        button = ttk.Button(frame, command=lambda: (opened.set(not opened.get()), render()))
        button.grid(row=0, column=0, sticky="w", pady=(2, 5))
        for variable in variables:
            variable.trace_add("write", render)
        render()
        return frame, body

    def build_table(self, parent, columns, **_kwargs):
        """Wrapped native labels, no font shrinking or horizontal clipping."""
        c = self.colors
        page = ScrolledPage(parent, c)
        frame = page.body
        for i, column in enumerate(columns):
            frame.grid_columnconfigure(i, weight=int(column.get("weight") or (1 if i == len(columns)-1 else 0)), minsize=100 if i < 2 else 170 if i == 2 else 0)
        data = []
        labels = []
        def set_rows(rows):
            data[:] = rows
            for child in frame.winfo_children(): child.destroy()
            labels.clear()
            for j, col in enumerate(columns):
                w = tk.Label(frame, text=col["title"], font=self.font_bold, bg=c["select"], fg=c["fg"], anchor="w", justify="left", padx=8, pady=8)
                w.grid(row=0, column=j, sticky="nsew", padx=1, pady=(0, 3)); labels.append(w)
            for i, row in enumerate(rows, 1):
                values = row.get("values", {})
                status = str(values.get("status") or "")
                tone = c["error"] if status in ("Brak", "Błąd", "FAILED") else c["success"] if status in ("Jest", "OK", "PASS") else c["muted"]
                for j, col in enumerate(columns):
                    w = tk.Label(frame, text=str(values.get(col["key"]) or "—"), font=self.font_bold if j == 0 else self.font_body,
                                 bg=c["card_bg"] if i % 2 else c["bg"], fg=tone if j == 0 else c["fg"], anchor="nw", justify="left", padx=8, pady=8)
                    w.grid(row=i, column=j, sticky="nsew", padx=1, pady=1);labels.append(w)
            for w in labels:
                w.configure(wraplength=120)
                w.bind("<Configure>", lambda e, item=w: item.configure(wraplength=max(30, e.width-18)), add="+")
            page.bind_wheel()
        set_rows([])
        return {"shell": page.shell, "set_rows": set_rows, "canvas": page.canvas, "rows": data, "labels": labels}

    def _destroy(self, event):
        if event.widget is not self.window:
            return
        if self._geometry_job:
            try: self.window.after_cancel(self._geometry_job)
            except tk.TclError: pass
        if self._status_trace:
            try: self._status_var.trace_remove("write", self._status_trace)
            except tk.TclError: pass
        if self._progress_trace:
            try: self._progress_var.trace_remove("write", self._progress_trace)
            except tk.TclError: pass
