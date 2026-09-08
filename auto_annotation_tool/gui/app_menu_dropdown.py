#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dropdown menu głównego okna aplikacji."""

import tkinter as tk

def _close_menu_dropdown(self, event=None):
    owner = getattr(self, "_menu_dropdown_owner", None)
    bind_id = getattr(self, "_menu_outside_click_bind_id", None)
    if bind_id:
        try:
            self.root.unbind("<ButtonPress-1>", bind_id)
        except Exception:
            pass
    self._menu_outside_click_bind_id = None

    bind_id = getattr(self, "_menu_escape_bind_id", None)
    if bind_id:
        try:
            self.root.unbind("<Escape>", bind_id)
        except Exception:
            pass
    self._menu_escape_bind_id = None

    popup = getattr(self, "_menu_dropdown", None)
    self._menu_dropdown = None
    self._menu_dropdown_owner = None
    self._menu_dropdown_update_items = None

    if owner is not None:
        for item in list(getattr(self, "_menu_buttons", []) or []):
            try:
                if item.get("button") is owner:
                    setter = item.get("set_hover")
                    if callable(setter):
                        setter(False)
                    break
            except Exception:
                pass

    if popup is not None:
        try:
            if popup.winfo_exists():
                popup.destroy()
        except Exception:
            pass

def _toggle_menu_dropdown(self, owner_widget, items, min_width: int = 220):
    current_owner = getattr(self, "_menu_dropdown_owner", None)
    current_popup = getattr(self, "_menu_dropdown", None)
    if current_popup is not None and current_owner is owner_widget:
        self._close_menu_dropdown()
        return

    self._open_menu_dropdown(owner_widget, items, min_width=min_width)

def _open_menu_dropdown(self, owner_widget, items, min_width: int = 220):
    self._close_menu_dropdown()

    palette = self.palette
    popup = tk.Toplevel(self.root)
    popup.overrideredirect(True)
    popup.configure(bg=palette["bg"])

    shell = tk.Frame(
        popup,
        bg=palette["panel"],
        bd=0,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette["border"]),
        highlightcolor=palette.get("panel_border", palette["border"])
    )
    shell.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(shell, bg=palette["panel"])
    body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def close_then_call(command):
        self._close_menu_dropdown()
        if callable(command):
            self.root.after(0, command)

    def make_item_row(item):
        kind = item.get("kind", "command")
        if kind == "separator":
            sep = tk.Frame(body, bg=palette.get("panel_border", palette["border"]), height=1, bd=0, highlightthickness=0)
            sep.pack(fill=tk.X, padx=6, pady=4)
            return

        is_selected = bool(item.get("selected"))
        is_disabled = bool(item.get("disabled"))
        prefix = "✓  " if is_selected else "   "
        text = f"{prefix}{item.get('label', '').strip()}"

        row = tk.Button(
            body,
            text=text,
            anchor="w",
            justify=tk.LEFT,
            bg=palette["panel"],
            fg=(palette.get("muted", palette["fg"]) if is_disabled else (palette["accent"] if is_selected else palette["fg"])),
            activebackground=palette.get("surface_info", palette.get("button_hover", palette["panel_alt"])),
            activeforeground=palette["fg"],
            disabledforeground=palette.get("muted", palette["fg"]),
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            padx=12,
            pady=7,
            font=("Segoe UI", 10, "bold" if is_selected else "normal"),
            cursor=("arrow" if is_disabled else "hand2"),
            state=(tk.DISABLED if is_disabled else tk.NORMAL),
            command=lambda cmd=item.get("command"): close_then_call(cmd)
        )
        row.pack(fill=tk.X)

        base_bg = palette["panel"]
        hover_bg = palette.get("surface_info", palette.get("button_hover", palette["panel_alt"]))
        base_fg = palette["accent"] if is_selected else palette["fg"]

        def _set_row_hover(active: bool, widget=row):
            try:
                widget.configure(
                    bg=hover_bg if active else base_bg,
                    fg=palette["fg"] if active else base_fg,
                )
            except Exception:
                pass

        if not is_disabled:
            row.bind("<Enter>", lambda _event, fn=_set_row_hover: fn(True), add="+")
            row.bind("<Leave>", lambda _event, fn=_set_row_hover: fn(False), add="+")

    def update_items(updated_items):
        if not popup.winfo_exists():
            return
        for child in body.winfo_children():
            child.destroy()
        for item in updated_items:
            make_item_row(item)

        popup.update_idletasks()
        popup_width = max(min_width, shell.winfo_reqwidth())
        popup_height = shell.winfo_reqheight()
        try:
            x = owner_widget.winfo_rootx()
            y = owner_widget.winfo_rooty() + owner_widget.winfo_height() + 4
        except Exception:
            x = 8
            y = 8

        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = max(8, min(x, screen_w - popup_width - 8))
        y = max(8, min(y, screen_h - popup_height - 8))
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

    update_items(items)
    popup.lift()

    self._menu_dropdown = popup
    self._menu_dropdown_owner = owner_widget
    self._menu_dropdown_update_items = update_items
    for item in list(getattr(self, "_menu_buttons", []) or []):
        try:
            if item.get("button") is owner_widget:
                setter = item.get("set_hover")
                if callable(setter):
                    setter(True)
                break
        except Exception:
            pass

    def handle_outside_click(event):
        if self._widget_contains_point(popup, event.x_root, event.y_root):
            return
        if self._widget_contains_point(owner_widget, event.x_root, event.y_root):
            return
        self._close_menu_dropdown()

    try:
        self._menu_outside_click_bind_id = self.root.bind("<ButtonPress-1>", handle_outside_click, add="+")
    except Exception:
        self._menu_outside_click_bind_id = None

    try:
        self._menu_escape_bind_id = self.root.bind("<Escape>", self._close_menu_dropdown, add="+")
    except Exception:
        self._menu_escape_bind_id = None
