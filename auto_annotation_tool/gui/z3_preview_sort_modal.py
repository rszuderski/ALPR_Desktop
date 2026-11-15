from __future__ import annotations

import tkinter as tk

from .web_slim_scrollbar import blend_hex_colors


def open_preview_sort_modal(host, *, sort_options, sort_color_keys):
    """Open a compact single-choice checklist for PZ2 list sorting."""
    existing = getattr(host, "_preview_sort_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return existing
    except Exception:
        pass

    parent = getattr(host, "frame", None)
    dialog = tk.Toplevel(parent)
    host._preview_sort_modal = dialog

    palette = getattr(getattr(host, "app", None), "palette", {}) or {}
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", panel)
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#a0a0a0")
    border = palette.get("border", "#3c3c3c")

    try:
        host.app.style_dialog_window(
            dialog,
            title="Sortowanie listy",
            geometry="340x425",
            parent=parent,
        )
    except Exception:
        dialog.title("Sortowanie listy")
        dialog.configure(bg=panel)
        try:
            dialog.transient(parent.winfo_toplevel() if parent is not None else None)
        except Exception:
            pass

    dialog.resizable(False, False)

    body = tk.Frame(
        dialog,
        bg=panel,
        bd=0,
        highlightthickness=0,
        padx=12,
        pady=10,
    )
    body.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        body,
        text="Sortowanie listy",
        bg=panel,
        fg=fg,
        anchor="w",
        font=("Segoe UI Semibold", 10),
    ).pack(fill=tk.X, pady=(0, 2))

    tk.Label(
        body,
        text="Wybierz jeden sposób uporządkowania tablic.",
        bg=panel,
        fg=muted,
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
    ).pack(fill=tk.X, pady=(0, 8))

    active_key = host._get_preview_sort_mode_key()

    def close():
        try:
            dialog.grab_release()
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass
        if getattr(host, "_preview_sort_modal", None) is dialog:
            host._preview_sort_modal = None

    def choose(target_key):
        host._on_preview_sort_mode_change(target_key)
        close()

    rows = tk.Frame(body, bg=panel, bd=0, highlightthickness=0)
    rows.pack(fill=tk.BOTH, expand=True)

    for mode_key, mode_label in sort_options:
        selected = str(mode_key) == str(active_key)
        tone_key = sort_color_keys.get(mode_key, "muted")
        accent = palette.get(tone_key, palette.get("accent", "#4aa3ff"))
        row_bg = (
            blend_hex_colors(accent, panel_alt, 0.70)
            if selected
            else panel_alt
        )
        active_bg = blend_hex_colors(accent, panel_alt, 0.56)
        mark = "✓" if selected else " "

        button = tk.Button(
            rows,
            text=f"{mark}  {mode_label}",
            anchor="w",
            justify=tk.LEFT,
            bg=row_bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            font=("Segoe UI", 9, "bold" if selected else "normal"),
            bd=0,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=(accent if selected else border),
            highlightcolor=(accent if selected else border),
            padx=10,
            pady=5,
            cursor="hand2",
            takefocus=1,
            command=lambda target_key=mode_key: choose(target_key),
        )
        button.pack(fill=tk.X, pady=(0, 3))

    footer = tk.Frame(body, bg=panel, bd=0, highlightthickness=0)
    footer.pack(fill=tk.X, pady=(5, 0))
    tk.Button(
        footer,
        text="Zamknij",
        command=close,
        bg=panel_alt,
        fg=fg,
        activebackground=blend_hex_colors(
            palette.get("accent", "#4aa3ff"),
            panel_alt,
            0.70,
        ),
        activeforeground=fg,
        bd=0,
        relief=tk.FLAT,
        padx=10,
        pady=4,
        cursor="hand2",
    ).pack(side=tk.RIGHT)

    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda _event: (close(), "break")[1])

    try:
        dialog.update_idletasks()
        dialog_height = max(425, dialog.winfo_reqheight())
        dialog.geometry(f"340x{dialog_height}")
        center = getattr(host.app, "_center_dialog_window", None)
        if callable(center):
            center(dialog, parent=parent, width=340, height=dialog_height)
    except Exception:
        pass

    try:
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
    except Exception:
        pass

    return dialog
