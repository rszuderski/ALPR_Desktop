#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global terminal window helpers for the main application shell."""

import tkinter as tk

from .web_slim_scrollbar import WebSlimScrollbar

def _apply_global_terminal_visual_state(self):
    palette = getattr(self, "palette", {})
    window = getattr(self, "_global_terminal_window", None)
    shell = getattr(self, "_global_terminal_shell", None)
    header = getattr(self, "_global_terminal_header", None)
    body = getattr(self, "_global_terminal_body", None)
    title_lbl = getattr(self, "_global_terminal_title_lbl", None)
    clear_btn = getattr(self, "_global_terminal_clear_btn", None)
    close_btn = getattr(self, "_global_terminal_close_btn", None)
    text_widget = getattr(self, "_global_terminal_text", None)
    scrollbar = getattr(self, "_global_terminal_scrollbar", None)
    hscrollbar = getattr(self, "_global_terminal_hscrollbar", None)

    if window is not None:
        try:
            window.configure(bg=palette.get("bg", "#1e1e1e"))
        except Exception:
            pass

    if shell is not None:
        try:
            shell.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
            )
        except Exception:
            pass

    for widget in (header, body):
        if widget is None:
            continue
        try:
            widget.configure(bg=palette.get("panel", "#252526"))
        except Exception:
            pass

    if title_lbl is not None:
        try:
            title_lbl.configure(
                bg=palette.get("panel", "#252526"),
                fg=palette.get("fg", "#f3f3f3"),
            )
        except Exception:
            pass

    for button in (clear_btn, close_btn):
        if button is None:
            continue
        try:
            button.configure(
                bg=palette.get("panel_alt", "#2d2d30"),
                fg=palette.get("fg", "#f3f3f3"),
                activebackground=palette.get("button_hover", "#37373d"),
                activeforeground=palette.get("fg", "#f3f3f3"),
                highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
                highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
            )
        except Exception:
            pass

    if text_widget is not None:
        self.style_text_widget(text_widget, role="console")
        self._configure_global_terminal_tags(text_widget)

    if scrollbar is not None:
        try:
            self.style_web_scrollbar(
                scrollbar,
                track_color=palette.get("console_bg", palette.get("panel", "#252526")),
            )
        except Exception:
            pass
    if hscrollbar is not None:
        try:
            self.style_web_scrollbar(
                hscrollbar,
                track_color=palette.get("console_bg", palette.get("panel", "#252526")),
            )
        except Exception:
            pass

def _normalize_global_terminal_entry(self, entry, default_tag: str = "terminal_default"):
    text = ""
    tag = default_tag

    if isinstance(entry, dict):
        text = str(entry.get("text", "") or "")
        tag = str(entry.get("tag", default_tag) or default_tag)
    elif isinstance(entry, (tuple, list)):
        if entry:
            text = str(entry[0] or "")
        if len(entry) > 1:
            tag = str(entry[1] or default_tag)
    else:
        text = str(entry or "")

    return {"text": text, "tag": tag}

def _refresh_global_terminal_plain_lines(self):
    entries = list(getattr(self, "_global_terminal_entries", []) or [])
    self._global_terminal_lines = [str(entry.get("text", "") or "") for entry in entries]

def _configure_global_terminal_tags(self, text_widget):
    if text_widget is None:
        return

    palette = getattr(self, "palette", {})
    default_fg = palette.get("console_fg", palette.get("fg", "#f3f3f3"))
    muted_fg = palette.get("muted", "#b9b9b9")
    info_fg = palette.get("accent", "#4aa3ff")
    success_fg = palette.get("success", "#2ecc71")
    warning_fg = palette.get("guide", palette.get("warning", "#f0b44c"))
    error_fg = palette.get("error", "#ff6b6b")
    header_fg = palette.get("accent_selected", info_fg)
    border_fg = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    try:
        text_widget.tag_configure("terminal_default", foreground=default_fg)
        text_widget.tag_configure("terminal_muted", foreground=muted_fg)
        text_widget.tag_configure("terminal_info", foreground=info_fg)
        text_widget.tag_configure("terminal_success", foreground=success_fg)
        text_widget.tag_configure("terminal_warning", foreground=warning_fg)
        text_widget.tag_configure("terminal_error", foreground=error_fg)
        text_widget.tag_configure("terminal_header", foreground=header_fg, font=("Consolas", 9, "bold"))
        text_widget.tag_configure("terminal_border", foreground=border_fg)
    except Exception:
        pass

def _insert_global_terminal_entry(self, text_widget, entry):
    if text_widget is None:
        return

    normalized = self._normalize_global_terminal_entry(entry)
    text = str(normalized.get("text", "") or "")
    tag = str(normalized.get("tag", "terminal_default") or "terminal_default")

    try:
        text_widget.insert(tk.END, text + "\n", (tag,))
    except Exception:
        try:
            text_widget.insert(tk.END, text + "\n")
        except Exception:
            pass

def _capture_terminal_widget_view_state(text_widget) -> dict:
    state = {
        "insert": None,
        "x_first": 0.0,
        "y_first": 0.0,
        "view_at_end": True,
        "insert_at_end": True,
        "sel_first": None,
        "sel_last": None,
    }
    if text_widget is None:
        return state

    try:
        state["insert"] = text_widget.index(tk.INSERT)
    except Exception:
        state["insert"] = None

    try:
        x_first, _x_last = text_widget.xview()
        state["x_first"] = float(x_first)
    except Exception:
        state["x_first"] = 0.0

    try:
        y_first, y_last = text_widget.yview()
        state["y_first"] = float(y_first)
        state["view_at_end"] = float(y_last) >= 0.999
    except Exception:
        state["y_first"] = 0.0
        state["view_at_end"] = True

    try:
        state["insert_at_end"] = bool(text_widget.compare(tk.INSERT, ">=", "end-2c"))
    except Exception:
        state["insert_at_end"] = True

    try:
        state["sel_first"] = text_widget.index("sel.first")
        state["sel_last"] = text_widget.index("sel.last")
    except Exception:
        state["sel_first"] = None
        state["sel_last"] = None

    return state

def _restore_terminal_widget_view_state(text_widget, state: dict) -> None:
    if text_widget is None or not isinstance(state, dict):
        return

    try:
        insert_index = state.get("insert")
        if insert_index:
            text_widget.mark_set(tk.INSERT, str(insert_index))
    except Exception:
        pass

    try:
        text_widget.xview_moveto(float(state.get("x_first", 0.0) or 0.0))
    except Exception:
        pass

    try:
        text_widget.yview_moveto(float(state.get("y_first", 0.0) or 0.0))
    except Exception:
        pass

    try:
        text_widget.tag_remove(tk.SEL, "1.0", tk.END)
        sel_first = state.get("sel_first")
        sel_last = state.get("sel_last")
        if sel_first and sel_last:
            text_widget.tag_add(tk.SEL, str(sel_first), str(sel_last))
    except Exception:
        pass

def _set_global_terminal_hold_position(self, hold: bool) -> None:
    self._global_terminal_hold_position = bool(hold)

def _sync_global_terminal_hold_position_from_view(self) -> None:
    text_widget = getattr(self, "_global_terminal_text", None)
    if text_widget is None:
        self._set_global_terminal_hold_position(False)
        return

    try:
        _y_first, y_last = text_widget.yview()
        at_end = float(y_last) >= 0.999
    except Exception:
        at_end = True

    self._set_global_terminal_hold_position(not at_end)

def _schedule_global_terminal_hold_position_sync(self, _event=None):
    try:
        self.root.after_idle(self._sync_global_terminal_hold_position_from_view)
    except Exception:
        try:
            self._sync_global_terminal_hold_position_from_view()
        except Exception:
            pass
    return None

def _on_global_terminal_pointer_event(self, event=None):
    self._set_global_terminal_hold_position(True)
    text_widget = getattr(self, "_global_terminal_text", None)
    if text_widget is not None and event is not None:
        try:
            text_widget.focus_set()
        except Exception:
            pass
        try:
            text_widget.mark_set(tk.INSERT, f"@{int(event.x)},{int(event.y)}")
        except Exception:
            pass
    return None

def _on_global_terminal_scroll_event(self, event=None):
    self._set_global_terminal_hold_position(True)
    self._schedule_global_terminal_hold_position_sync()
    return None

def _populate_global_terminal_widget(self):
    text_widget = getattr(self, "_global_terminal_text", None)
    if text_widget is None:
        return

    try:
        state_before = self._capture_terminal_widget_view_state(text_widget)
        follow_end = bool(
            (not bool(getattr(self, "_global_terminal_hold_position", False)))
            and state_before.get("view_at_end")
        )
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        self._configure_global_terminal_tags(text_widget)
        entries = list(getattr(self, "_global_terminal_entries", []) or [])
        if not entries and getattr(self, "_global_terminal_lines", None):
            entries = [self._normalize_global_terminal_entry(line) for line in self._global_terminal_lines]
            self._global_terminal_entries = entries
        for entry in entries:
            self._insert_global_terminal_entry(text_widget, entry)
        if follow_end:
            text_widget.see(tk.END)
        else:
            self._restore_terminal_widget_view_state(text_widget, state_before)
    except Exception:
        pass
    finally:
        try:
            text_widget.configure(state=tk.DISABLED)
        except Exception:
            pass

def _position_global_terminal_window(self, force: bool = False):
    window = getattr(self, "_global_terminal_window", None)
    if window is None:
        return
    if self._global_terminal_geometry_initialized and not force:
        return

    try:
        self.root.update_idletasks()
        root_x = int(self.root.winfo_rootx() or 0)
        root_y = int(self.root.winfo_rooty() or 0)
        root_w = max(900, int(self.root.winfo_width() or self.root.winfo_reqwidth() or 900))
        root_h = max(640, int(self.root.winfo_height() or self.root.winfo_reqheight() or 640))
    except Exception:
        root_x = 80
        root_y = 80
        root_w = 1200
        root_h = 760

    width = min(1080, max(780, int(root_w * 0.78)))
    height = min(420, max(260, int(root_h * 0.36)))
    pos_x = max(8, root_x + int((root_w - width) / 2))
    pos_y = max(8, root_y + int((root_h - height) / 2))

    try:
        window.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self._global_terminal_geometry_initialized = True
    except Exception:
        pass

def _ensure_global_terminal_window(self):
    window = getattr(self, "_global_terminal_window", None)
    try:
        if window is not None and window.winfo_exists():
            return window
    except Exception:
        pass

    palette = getattr(self, "palette", {})
    window = tk.Toplevel(self.root)
    window.withdraw()
    window.title("Terminal procesu")
    try:
        window.transient(self.root)
    except Exception:
        pass
    window.configure(bg=palette.get("bg", "#1e1e1e"))
    window.protocol("WM_DELETE_WINDOW", self.hide_global_terminal)
    try:
        window.bind("<Escape>", lambda _event: self.hide_global_terminal(), add="+")
    except Exception:
        pass

    shell = tk.Frame(
        window,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
    )
    shell.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    header = tk.Frame(shell, bg=palette.get("panel", "#252526"), bd=0, highlightthickness=0)
    header.pack(fill=tk.X, padx=12, pady=(12, 6))

    title_lbl = tk.Label(
        header,
        text="Terminal procesu",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 10, "bold"),
        bg=palette.get("panel", "#252526"),
        fg=palette.get("fg", "#f3f3f3"),
        bd=0,
        highlightthickness=0,
    )
    title_lbl.pack(side=tk.LEFT)

    close_btn = tk.Button(
        header,
        text="Zwin",
        command=self.hide_global_terminal,
        cursor="hand2",
        bd=0,
        relief=tk.FLAT,
        highlightthickness=1,
        padx=10,
        pady=3,
        font=("Segoe UI", 9),
    )
    close_btn.pack(side=tk.RIGHT)

    clear_btn = tk.Button(
        header,
        text="Wyczysc",
        command=self.clear_global_terminal,
        cursor="hand2",
        bd=0,
        relief=tk.FLAT,
        highlightthickness=1,
        padx=10,
        pady=3,
        font=("Segoe UI", 9),
    )
    clear_btn.pack(side=tk.RIGHT, padx=(0, 6))

    body = tk.Frame(shell, bg=palette.get("panel", "#252526"), bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    text_widget = tk.Text(
        body,
        wrap=tk.NONE,
        font=("Consolas", 9),
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
        takefocus=1,
        cursor="xterm",
    )
    hscrollbar = WebSlimScrollbar(
        body,
        orient=tk.HORIZONTAL,
        command=text_widget.xview,
        auto_hide=False,
    )
    hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = WebSlimScrollbar(
        body,
        orient=tk.VERTICAL,
        command=text_widget.yview,
        auto_hide=False,
    )
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.configure(yscrollcommand=scrollbar.set, xscrollcommand=hscrollbar.set)
    text_widget.web_vbar = scrollbar
    text_widget.web_hbar = hscrollbar

    for sequence in ("<Button-1>", "<B1-Motion>"):
        try:
            text_widget.bind(sequence, self._on_global_terminal_pointer_event, add="+")
        except Exception:
            pass

    for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>"):
        try:
            text_widget.bind(sequence, self._on_global_terminal_scroll_event, add="+")
        except Exception:
            pass

    for sequence in (
        "<ButtonRelease-1>",
        "<ButtonRelease-2>",
        "<ButtonRelease-3>",
        "<ButtonRelease-4>",
        "<ButtonRelease-5>",
        "<KeyRelease-Up>",
        "<KeyRelease-Down>",
        "<KeyRelease-Prior>",
        "<KeyRelease-Next>",
        "<KeyRelease-Home>",
        "<KeyRelease-End>",
    ):
        try:
            text_widget.bind(sequence, self._schedule_global_terminal_hold_position_sync, add="+")
        except Exception:
            pass

    for widget in (scrollbar, hscrollbar):
        try:
            widget.bind("<ButtonRelease-1>", self._schedule_global_terminal_hold_position_sync, add="+")
        except Exception:
            pass

    self._global_terminal_window = window
    self._global_terminal_shell = shell
    self._global_terminal_header = header
    self._global_terminal_title_lbl = title_lbl
    self._global_terminal_clear_btn = clear_btn
    self._global_terminal_close_btn = close_btn
    self._global_terminal_body = body
    self._global_terminal_text = text_widget
    self._global_terminal_scrollbar = scrollbar
    self._global_terminal_hscrollbar = hscrollbar

    self._apply_global_terminal_visual_state()
    self._populate_global_terminal_widget()
    self._position_global_terminal_window(force=True)
    return window

def is_global_terminal_visible(self) -> bool:
    window = getattr(self, "_global_terminal_window", None)
    if window is None:
        return False
    try:
        return bool(window.winfo_exists()) and str(window.state()) != "withdrawn"
    except Exception:
        return False

def show_global_terminal(self):
    window = self._ensure_global_terminal_window()
    self._position_global_terminal_window()
    try:
        window.deiconify()
        window.lift()
        window.focus_force()
    except Exception:
        pass
    self._global_terminal_visible = True
    self._sync_global_terminal_toggle_state()

def hide_global_terminal(self):
    window = getattr(self, "_global_terminal_window", None)
    if window is not None:
        try:
            window.withdraw()
        except Exception:
            pass
    self._global_terminal_visible = False
    self._sync_global_terminal_toggle_state()

def toggle_global_terminal(self):
    if self.is_global_terminal_visible():
        self.hide_global_terminal()
    else:
        self.show_global_terminal()

def clear_global_terminal(self):
    self._global_terminal_entries = []
    self._global_terminal_lines = []
    self._set_global_terminal_hold_position(False)
    self._populate_global_terminal_widget()

def append_global_terminal_entries(self, entries):
    normalized_entries = []
    for entry in list(entries or []):
        normalized = self._normalize_global_terminal_entry(entry)
        text = str(normalized.get("text", "") or "")
        if text == "":
            normalized_entries.append(normalized)
        elif text.strip():
            normalized_entries.append(normalized)

    if not normalized_entries:
        return

    current_entries = list(getattr(self, "_global_terminal_entries", []) or [])
    current_entries.extend(normalized_entries)
    overflow = len(current_entries) - int(self._global_terminal_max_lines)
    if overflow > 0:
        current_entries = current_entries[overflow:]
    self._global_terminal_entries = current_entries
    self._refresh_global_terminal_plain_lines()

    def update():
        text_widget = getattr(self, "_global_terminal_text", None)
        if text_widget is None:
            return

        try:
            state_before = self._capture_terminal_widget_view_state(text_widget)
            follow_end = bool(
                (not bool(getattr(self, "_global_terminal_hold_position", False)))
                and state_before.get("view_at_end")
            )
            text_widget.configure(state=tk.NORMAL)
            self._configure_global_terminal_tags(text_widget)
            if overflow > 0:
                text_widget.delete("1.0", tk.END)
                for entry in self._global_terminal_entries:
                    self._insert_global_terminal_entry(text_widget, entry)
            else:
                for entry in normalized_entries:
                    self._insert_global_terminal_entry(text_widget, entry)
            if follow_end:
                text_widget.see(tk.END)
            else:
                self._restore_terminal_widget_view_state(text_widget, state_before)
        except Exception:
            pass
        finally:
            try:
                text_widget.configure(state=tk.DISABLED)
            except Exception:
                pass

    try:
        self.root.after(0, update)
    except Exception:
        pass

def append_global_terminal(self, message: str, source: str = None, tag: str = "terminal_default"):
    text = "" if message is None else str(message)
    if not text:
        return

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    prefix = f"[{str(source).strip()}] " if str(source or "").strip() else ""
    new_entries = []
    for line in normalized.split("\n"):
        if line == "":
            continue
        new_entries.append({"text": f"{prefix}{line}", "tag": tag})

    self.append_global_terminal_entries(new_entries)

