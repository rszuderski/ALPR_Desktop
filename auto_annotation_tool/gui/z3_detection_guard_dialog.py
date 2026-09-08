#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialog ochrony wyników PZ2 przed uruchomieniem detekcji znaków."""

import tkinter as tk

from .web_slim_scrollbar import blend_hex_colors

def prompt_pz2_detection_guard_options(host, method_key: str) -> dict | None:
    self = host
    method_key = self._normalize_detection_method_key(method_key or self._get_detection_method_key())
    method_has_yolo = method_key in {"YOLO", "BOTH", "YOLO_OCR"}
    counts = self._build_pz2_detection_guard_counts()
    result = {"value": None}

    palette = getattr(self.app, "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f1c40f")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    dialog = tk.Toplevel(self.frame)
    try:
        dialog.withdraw()
    except Exception:
        pass

    styler = getattr(self.app, "style_dialog_window", None)
    if callable(styler):
        try:
            styler(
                dialog,
                title="Ochrona i refiner wyników PZ2",
                geometry="780x690",
                parent=self.frame,
            )
        except Exception:
            dialog.title("Ochrona i refiner wyników PZ2")
            dialog.resizable(True, True)
    else:
        dialog.title("Ochrona i refiner wyników PZ2")
        dialog.resizable(True, True)

    try:
        dialog.resizable(True, True)
        dialog.minsize(700, 575)
    except Exception:
        pass

    try:
        dialog.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
    except Exception:
        pass

    surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(surface_builder):
        try:
            surface = surface_builder(dialog, tone="info")
        except Exception:
            surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            surface.pack(fill=tk.BOTH, expand=True)
    else:
        surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        surface.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(surface, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

    window_state = {"normal_geometry": "", "maximized": False}

    def minimize_dialog():
        try:
            dialog.grab_release()
        except Exception:
            pass
        try:
            dialog.iconify()
        except Exception:
            pass

    def toggle_maximize_dialog():
        try:
            if not window_state.get("maximized"):
                window_state["normal_geometry"] = str(dialog.geometry() or "")
                margin = 16
                width = max(700, int(dialog.winfo_screenwidth()) - margin * 2)
                height = max(575, int(dialog.winfo_screenheight()) - margin * 2)
                dialog.geometry(f"{width}x{height}+{margin}+{margin}")
                window_state["maximized"] = True
                try:
                    maximize_btn.configure(text="❐")
                except Exception:
                    pass
            else:
                geometry = str(window_state.get("normal_geometry") or "").strip()
                if geometry:
                    dialog.geometry(geometry)
                window_state["maximized"] = False
                try:
                    maximize_btn.configure(text="□")
                except Exception:
                    pass
        except Exception:
            pass

    header = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    header.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    header.grid_columnconfigure(0, weight=1)
    header.grid_columnconfigure(1, weight=0)

    title_stack = tk.Frame(header, bg=panel_bg, bd=0, highlightthickness=0)
    title_stack.grid(row=0, column=0, sticky="ew")

    tk.Label(
        title_stack,
        text="Detekcja może nadpisać istniejące boxy znaków",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 12, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    controls = tk.Frame(header, bg=panel_bg, bd=0, highlightthickness=0)
    controls.grid(row=0, column=1, sticky="ne", padx=(12, 0))

    def make_window_icon(text, command, tooltip_text=""):
        btn = tk.Button(
            controls,
            text=str(text),
            command=command,
            bg=blend_hex_colors(panel_alt, panel_bg, 0.18),
            fg=fg,
            activebackground=blend_hex_colors(panel_alt, success, 0.18),
            activeforeground=fg,
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=border,
            width=3,
            height=1,
            cursor="hand2",
            takefocus=False,
        )
        if tooltip_text:
            try:
                btn.configure(cursor="hand2")
            except Exception:
                pass
        return btn

    minimize_btn = make_window_icon("_", minimize_dialog, "Minimalizuj")
    minimize_btn.pack(side=tk.LEFT, padx=(0, 6))
    maximize_btn = make_window_icon("□", toggle_maximize_dialog, "Maksymalizuj")
    maximize_btn.pack(side=tk.LEFT)

    tk.Label(
        body,
        text=(
            "Wybierz zakres detekcji i zasady zapisu wyniku. Tablice perfect są domyślnie zachowywane, "
            "a refiner może poprawiać wyłącznie geometrię ich boxów bez zmiany odczytu znaków."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 10),
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 7))

    summary = tk.Frame(body, bg=panel_alt, bd=0, highlightthickness=1, highlightbackground=border)
    summary.pack(fill=tk.X, pady=(0, 8))
    for col, weight in ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)):
        summary.grid_columnconfigure(col, weight=weight)

    summary_rows = [
        ("Wszystkie tablice", int(counts.get("total", 0)), "Tablice cropy w PZ2"),
        ("Ze statusem perfect", int(counts.get("perfect", 0)), "Chronione domyślnie"),
        ("Do korekty", int(counts.get("needs_detection", 0)), "Nie mają statusu perfect"),
        ("Perfect z OCR", int(counts.get("perfect_ocr", 0)), "Perfect uzyskane przez OCR"),
        ("Z ręcznymi boxami", int(counts.get("manual", 0)), f"{int(counts.get('manual_boxes', 0))} boxów"),
    ]
    for idx, (label, value, note) in enumerate(summary_rows):
        cell = tk.Frame(summary, bg=panel_alt, bd=0, highlightthickness=0)
        cell.grid(row=0, column=idx, sticky="nsew", padx=(8 if idx == 0 else 4, 8 if idx == len(summary_rows) - 1 else 4), pady=6)
        tk.Label(
            cell,
            text=f"{int(value)}",
            bg=panel_alt,
            fg=success if int(value) > 0 else muted,
            font=("Segoe UI", 13, "bold"),
            anchor="w",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            cell,
            text=str(label),
            bg=panel_alt,
            fg=muted,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            cell,
            text=str(note),
            bg=panel_alt,
            fg=blend_hex_colors(muted, panel_alt, 0.18),
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(anchor=tk.W, fill=tk.X)

    protect_manual_var = tk.BooleanVar(value=True)
    protect_perfect_var = tk.BooleanVar(value=bool(self.detect_protect_perfect_var.get()))
    refine_perfect_yolo_var = tk.BooleanVar(value=bool(self.detect_refine_perfect_yolo_var.get()))
    process_non_perfect_only_var = tk.BooleanVar(value=False)
    try:
        continuity_guard_default = bool(self.detect_refiner_continuity_guard_var.get())
    except Exception:
        continuity_guard_default = True
    continuity_guard_var = tk.BooleanVar(value=continuity_guard_default)

    options = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    options.pack(fill=tk.X)

    def add_check(text, variable, note, *, disabled=False):
        row = tk.Frame(options, bg=panel_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, pady=(0, 4))
        chk = tk.Checkbutton(
            row,
            text=text,
            variable=variable,
            bg=panel_bg,
            fg=fg,
            activebackground=panel_bg,
            activeforeground=fg,
            selectcolor=panel_alt,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            state=(tk.DISABLED if disabled else tk.NORMAL),
        )
        chk.pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            row,
            text=note,
            bg=panel_bg,
            fg=muted if not disabled else blend_hex_colors(muted, panel_bg, 0.45),
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            wraplength=700,
        ).pack(anchor=tk.W, fill=tk.X, padx=(24, 0), pady=(0, 0))
        return chk

    add_check(
        "Chroń ręczne boxy znaków",
        protect_manual_var,
        "Manualne boxy są chronione bezwzględnie. OCR/YOLO może uzupełniać lub poprawiać pozostałe ramki, ale nie nadpisze ręcznych korekt.",
        disabled=True,
    )
    needs_detection_count = int(counts.get("needs_detection", 0) or 0)
    non_perfect_scope_check = add_check(
        "Przetwarzaj tylko tablice do korekty",
        process_non_perfect_only_var,
        "Ogranicza detekcję do tablic bez statusu perfect. Tablice perfect zostaną pominięte w tym przebiegu.",
        disabled=needs_detection_count <= 0,
    )
    protect_perfect_check = add_check(
        "Zachowaj status i odczyt tablic perfect",
        protect_perfect_var,
        "Perfect oznacza, że tekst znaków zgadza się z oczekiwanym odczytem z nazwy pliku. Ta opcja blokuje przebudowę wyniku od zera.",
    )
    refine_check = add_check(
        "Refiner geometrii boxów perfect",
        refine_perfect_yolo_var,
        "Refiner używa propozycji YOLO tylko jako punktu startowego. Zawęża lub przesuwa box, jeśli poprawia pokrycie właściwego znaku. Opcja działa tylko w pipeline z YOLO.",
        disabled=not method_has_yolo,
    )

    continuity_check = add_check(
        "Pilnuj ciągłości znaku",
        continuity_guard_var,
        "Dodatkowy bezpiecznik refinera: box musi obejmować tę samą spójną strukturę znaku i nie może przejmować sąsiedniego znaku.",
        disabled=not method_has_yolo,
    )

    def sync_refine_state(*_args):
        try:
            non_perfect_scope = bool(process_non_perfect_only_var.get())
            protect_perfect_check.configure(state=(tk.DISABLED if non_perfect_scope else tk.NORMAL))
            refiner_available = method_has_yolo and bool(protect_perfect_var.get()) and not non_perfect_scope
            state = tk.NORMAL if refiner_available else tk.DISABLED
            refine_check.configure(state=state)
            continuity_state = tk.NORMAL if (refiner_available and bool(refine_perfect_yolo_var.get())) else tk.DISABLED
            continuity_check.configure(state=continuity_state)
            non_perfect_scope_check.configure(state=(tk.NORMAL if needs_detection_count > 0 else tk.DISABLED))
        except Exception:
            pass

    try:
        process_non_perfect_only_var.trace_add("write", sync_refine_state)
        protect_perfect_var.trace_add("write", sync_refine_state)
        refine_perfect_yolo_var.trace_add("write", sync_refine_state)
        sync_refine_state()
    except Exception:
        pass

    hint_color = warning if int(counts.get("perfect", 0) or 0) > 0 or int(counts.get("manual", 0) or 0) > 0 else muted
    tk.Label(
        body,
        text=(
            "Jeśli chcesz przebudować automatyczne wyniki od zera, możesz odznaczyć ochronę perfectów. "
            "Ręczne korekty pozostają chronione."
        ),
        bg=panel_bg,
        fg=hint_color,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0, height=46)
    buttons.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
    try:
        buttons.pack_propagate(False)
    except Exception:
        pass

    def close(value):
        if value:
            try:
                self.detect_protect_manual_var.set(True)
                self.detect_protect_perfect_var.set(bool(protect_perfect_var.get()))
                self.detect_refine_perfect_yolo_var.set(bool(refine_perfect_yolo_var.get()))
                self.detect_refiner_continuity_guard_var.set(bool(continuity_guard_var.get()))
                self._save_local_setting("char_detect_protect_manual", True)
                self._save_local_setting("char_detect_protect_perfect", bool(self.detect_protect_perfect_var.get()))
                self._save_local_setting("char_detect_refine_perfect_yolo", bool(self.detect_refine_perfect_yolo_var.get()))
                self._save_local_setting("char_detect_refiner_continuity_guard", bool(self.detect_refiner_continuity_guard_var.get()))
                self._refresh_detection_refiner_guard_label()
            except Exception:
                pass
            use_perfect_refiner = bool(
                method_has_yolo and protect_perfect_var.get() and refine_perfect_yolo_var.get()
            )
            use_continuity_guard = bool(use_perfect_refiner and continuity_guard_var.get())
            if bool(process_non_perfect_only_var.get()):
                process_scope = "non_perfect_only"
            else:
                process_scope = "all"
            result["value"] = {
                "protect_manual_boxes": True,
                "protect_perfect_plates": bool(protect_perfect_var.get()),
                "allow_yolo_geometry_on_perfect": use_perfect_refiner,
                "use_perfect_box_refiner": use_perfect_refiner,
                "use_perfect_refiner_continuity_guard": use_continuity_guard,
                "process_scope": process_scope,
                "counts": dict(counts),
            }
        else:
            result["value"] = None
        try:
            dialog.destroy()
        except Exception:
            pass

    def make_action_button(parent, text, command, *, primary=False):
        fill = (
            blend_hex_colors(success, panel_alt, 0.28)
            if primary
            else blend_hex_colors(panel_alt, panel_bg, 0.18)
        )
        outline = success if primary else border
        text_color = self._get_readable_text_color(fill, preferred=fg if not primary else "#ffffff")
        active_fill = (
            blend_hex_colors(success, fill, 0.18)
            if primary
            else blend_hex_colors(fill, success, 0.08)
        )
        button_host = tk.Frame(
            parent,
            bg=outline,
            bd=0,
            highlightthickness=0,
            width=176 if primary else 104,
            height=36,
        )
        button_host.pack_propagate(False)
        button = tk.Button(
            button_host,
            text=str(text),
            command=command,
            bg=fill,
            fg=text_color,
            activebackground=active_fill,
            activeforeground=self._get_readable_text_color(active_fill, preferred=text_color),
            font=("Segoe UI", 9),
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            padx=10,
            pady=3,
            cursor="hand2",
            takefocus=True,
        )
        button.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        return button_host

    make_action_button(buttons, "Anuluj", lambda: close(False)).pack(side=tk.RIGHT)
    make_action_button(buttons, "Zatwierdź i uruchom", lambda: close(True), primary=True).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
        dialog.bind("<Escape>", lambda _e: close(False))
        dialog.bind("<Return>", lambda _e: close(True))
        dialog.bind(
            "<Map>",
            lambda _e: dialog.grab_set() if str(dialog.state() or "") != "iconic" else None,
            add="+",
        )
    except Exception:
        pass

    try:
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

    dialog.wait_window()
    return result.get("value")


