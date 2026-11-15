# -*- coding: utf-8 -*-
"""On-demand read-only details for Z4. No model, registry or session writes.

The frozen display snapshot owns its source identity and full JSON/SHA strings.
Library scans and completed PC checks cannot replace the open dialog's record.
"""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from .z4_mobile_library_view import Sheet


@dataclass(frozen=True)
class DetailSnapshot:
    source: str
    identity: str
    name: str
    path: str
    context: str
    json_label: str
    sha_label: str
    sha_value: str
    sheet: Sheet | None
    summary_text: str
    raw_json: str


class MobileArtifactDetailsDialog:
    """One resizable local-grab modal, no nested mainloop or global key binding."""

    def __init__(self, owner, snapshot: DetailSnapshot, *, return_focus=None):
        self.owner = owner
        self.snapshot = snapshot
        self.return_focus = return_focus
        self._closed = False
        self._activate_id = None
        self._layout_id = None
        self._previous_grab = None
        self._previous_grab_status = None
        self._owns_grab = False
        self.window = tk.Toplevel(owner.window)
        self.window.withdraw()
        try:
            self.window.title("Szczegóły · " + snapshot.name)
            self.window.transient(owner.window)
            self.window.resizable(True, True)
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            self.window.bind("<Escape>", self._escape, add="+")
            self.window.bind("<Destroy>", self._destroyed, add="+")
            self._build()
            self._place_window()
            self.window.deiconify()
            self._activate_id = self.window.after_idle(self._activate)
        except Exception:
            self._closed = True
            self.window.destroy()
            raise

    def _build(self):
        o, s = self.owner, self.snapshot
        self.shell = ttk.Frame(self.window, padding=14, style=o._ui + ".TFrame")
        self.shell.pack(fill="both", expand=True)
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(2, weight=1)
        self.name_label = ttk.Label(self.shell, text=s.name, justify="left",
                                   style=o._ui + ".Title.TLabel")
        self.name_label.grid(row=0, column=0, sticky="ew")
        self.context_label = ttk.Label(self.shell,
            text=s.context + "\nDane z chwili otwarcia · odczyt bez uruchamiania kontroli.",
            justify="left", style=o._ui + ".Muted.TLabel")
        self.context_label.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.notebook = ttk.Notebook(self.shell, style=o._ui + ".TNotebook")
        self.notebook.grid(row=2, column=0, sticky="nsew")
        summary = ttk.Frame(self.notebook, style=o._ui + ".TFrame")
        raw = ttk.Frame(self.notebook, style=o._ui + ".TFrame")
        self.notebook.add(summary, text="Podsumowanie")
        self.notebook.add(raw, text=s.json_label)
        self.summary = self._text_area(summary, wrap="word", font=o._font_body, horizontal=False)
        self.raw = self._text_area(raw, wrap="none", font=o._font_mono, horizontal=True)
        self.summary.configure(padx=14, pady=12, spacing1=2, spacing3=4)
        self.summary.tag_configure("title", font=o._font_title, spacing3=6)
        self.summary.tag_configure("label", font=o._font_bold)
        self.summary.tag_configure("heading", font=o._font_bold, spacing1=10, spacing3=5)
        for tone in ("success", "warning", "danger", "info", "neutral"):
            self.summary.tag_configure(tone, font=o._font_bold, spacing3=6)
        self.apply_palette()
        self.summary.configure(state="normal")
        sheet = s.sheet
        if sheet is None:
            self.summary.insert("end", s.summary_text)
        else:
            self.summary.insert("end", sheet.title + "\n", "title")
            self.summary.insert("end", sheet.subtitle + "\n", "muted")
            self.summary.insert("end", sheet.state + "\n", sheet.tone)
            self.summary.insert("end", "Następny krok: ", "label")
            self.summary.insert("end", sheet.next_step + "\n")
            self.summary.insert("end", "IDENTYFIKACJA I DOWÓD\n", "heading")
            for name, value in sheet.fields:
                self.summary.insert("end", name + ": ", "label")
                self.summary.insert("end", value + "\n")
            self.summary.insert("end", "\n" + sheet.note, "muted")
        self.summary.configure(state="disabled")
        self.raw.configure(state="normal")
        self.raw.insert("end", s.raw_json)
        self.raw.configure(state="disabled")
        self.footer = ttk.Frame(self.shell, style=o._ui + ".TFrame")
        self.footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.footer.columnconfigure(0, weight=1)
        self.button_group = ttk.Frame(self.footer, style=o._ui + ".TFrame")
        self.button_group.grid(row=0, column=0, sticky="ew")
        self.path_button = ttk.Button(self.button_group, text="Kopiuj ścieżkę",
                                     command=lambda: self._copy(s.path, "Skopiowano ścieżkę."))
        self.sha_button = ttk.Button(self.button_group, text=s.sha_label,
                                    command=lambda: self._copy(s.sha_value, "Skopiowano pełne SHA ze szczegółów."))
        self.path_button.configure(state="normal" if s.path else "disabled")
        self.sha_button.configure(state="normal" if s.sha_value else "disabled")
        self.action_buttons = [self.path_button, self.sha_button]
        self.close_button = ttk.Button(self.footer, text="Zamknij", command=self.close)
        self.close_button.grid(row=0, column=1, sticky="se", padx=(10, 0), pady=2)
        self.status = tk.StringVar(master=self.window, value="Escape zamyka szczegóły. Dane i pliki pozostają bez zmian.")
        self.status_label = ttk.Label(self.footer, textvariable=self.status,
                                      justify="left", style=o._ui + ".Muted.TLabel")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.shell.bind("<Configure>", self._queue_layout, add="+")
        self._queue_layout()

    def _text_area(self, host, *, wrap, font, horizontal):
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        text = tk.Text(host, wrap=wrap, font=font, state="disabled", height=8,
                       width=40, relief="flat", borderwidth=0, exportselection=False)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(host, command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        if horizontal:
            scrollbar_x = ttk.Scrollbar(host, orient="horizontal", command=text.xview)
            scrollbar_x.grid(row=1, column=0, sticky="ew")
            text.configure(xscrollcommand=scrollbar_x.set)
        return text

    def _place_window(self):
        parent = self.owner.window
        parent.update_idletasks()
        screen_w, screen_h = parent.winfo_screenwidth(), parent.winfo_screenheight()
        max_w, max_h = max(320, screen_w - 70), max(260, screen_h - 100)
        size = getattr(self.owner, "_details_window_size", None)
        if size is None:
            size = (max(900, min(1160, int(parent.winfo_width() * .92))),
                    max(600, min(780, int(parent.winfo_height() * .9))))
        width, height = min(size[0], max_w), min(size[1], max_h)
        self.window.minsize(min(700, max_w), min(460, max_h))
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        x, y = max(0, min(x, screen_w - width - 10)), max(0, min(y, screen_h - height - 70))
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def _queue_layout(self, _event=None):
        if not self._closed and self._layout_id is None:
            self._layout_id = self.window.after_idle(self._layout)

    def _layout(self):
        self._layout_id = None
        if self._closed:
            return
        width = max(150, self.shell.winfo_width() - 30)
        for label in (self.name_label, self.context_label, self.status_label):
            label.configure(wraplength=width)
        available = max(100, width - self.close_button.winfo_reqwidth() - 14)
        used, row, column = 0, 0, 0
        for button in self.action_buttons:
            needed = button.winfo_reqwidth() + 8
            if column and used + needed > available:
                used, row, column = 0, row + 1, 0
            button.grid(row=row, column=column, sticky="w", padx=(0, 7), pady=2)
            used += needed
            column += 1

    def apply_palette(self):
        if self._closed or not hasattr(self, "raw"):
            return
        colors = self.owner.colors
        for target in (self.summary, self.raw):
            target.configure(background=colors["surface"], foreground=colors["ink"],
                             selectbackground=colors["info"], insertbackground=colors["ink"])
        self.summary.tag_configure("muted", foreground=colors["muted"])
        self.summary.tag_configure("heading", foreground=colors["info"])
        for tone in ("success", "warning", "danger", "info", "neutral"):
            self.summary.tag_configure(tone, foreground=colors[tone])

    def _activate(self):
        self._activate_id = None
        if self._closed:
            return
        if not self.window.winfo_viewable():
            self._activate_id = self.window.after(20, self._activate)
            return
        self._previous_grab = self.window.grab_current()
        if self._previous_grab is not None:
            self._previous_grab_status = self._previous_grab.grab_status()
        self.window.grab_set()
        self._owns_grab = True
        self.window.lift()
        self.summary.focus_set()

    def is_open(self):
        return not self._closed and bool(self.window.winfo_exists())

    def raise_window(self):
        if self.is_open():
            self.window.lift()
            self.window.focus_set()

    def _copy(self, value: str, notice: str):
        if value and not self._closed:
            self.window.clipboard_clear()
            self.window.clipboard_append(value)
            self.status.set(notice)

    def _escape(self, _event=None):
        self.close()
        return "break"

    def _destroyed(self, event):
        if event.widget is self.window and not self._closed:
            self.close(restore_focus=False, destroy=False)

    def close(self, *, restore_focus=True, destroy=True):
        if self._closed:
            return
        self._closed = True
        for name in ("_activate_id", "_layout_id"):
            token = getattr(self, name, None)
            if token is not None:
                try:
                    self.window.after_cancel(token)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        restore_grab = False
        try:
            current = self.owner.window.grab_current()
            # Direct destruction can release the Tcl grab before <Destroy>.
            # Never steal a different dialog's newer grab.
            restore_grab = self._owns_grab and current in (None, self.window)
            if current is self.window:
                self.window.grab_release()
            size = (self.window.winfo_width(), self.window.winfo_height())
            if size[0] > 1 and size[1] > 1:
                self.owner._details_window_size = size
        except tk.TclError:
            pass
        self._owns_grab = False
        if getattr(self.owner, "_details_dialog", None) is self:
            self.owner._details_dialog = None
        if destroy:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
        if restore_grab and self._previous_grab is not None:
            try:
                if self._previous_grab.winfo_exists():
                    if self._previous_grab_status == "global":
                        self._previous_grab.grab_set_global()
                    else:
                        self._previous_grab.grab_set()
            except tk.TclError:
                pass
        if restore_focus and not getattr(self.owner, "_closed", False):
            try:
                target = self.return_focus
                if target is not None and target.winfo_viewable():
                    target.focus_set()
            except tk.TclError:
                pass
