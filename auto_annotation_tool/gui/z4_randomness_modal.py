"""Category-based, transactional editor for synthetic-series deviations."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import queue
import random
import threading
import tkinter as tk
from tkinter import font as tkfont, ttk

from ..training.augmentation_ranges import (
    GROUP_LABELS, SPECS, RangeDraft, legacy_amount, range_bounds, unavailable_reason,
)
from ..training.dataset_augmentation import preview_augmentation_image
from .app_window_recovery import configure_minimizable_modal


def _fmt(value, spec):
    digits = 0 if spec.integer else 3 if spec.step < .01 else 2
    return f"{value:.{digits}f}".replace(".", ",") + spec.unit


class RandomnessModal:
    def __init__(self, owner, *, show=True):
        self.owner = owner
        self.base = owner._profile_from_vars().normalized()
        self.draft = RangeDraft(self.base)
        self.closed = False
        self.rows = {}
        self._images = []
        self._preview_sources = []
        self._preview_render_keys = []
        self._resize_after = None
        self._sash_after = None
        self._preview_expanded = False
        self._saved_sash = None
        self._preview_after = None
        self._poll_after = None
        self._busy = False
        self._pending = None
        self._revision = 0
        self._cancel_event = threading.Event()
        self._results = queue.Queue()
        self._seed_rng = random.Random(self.base.seed)
        self.preview_seeds = tuple(self._seed_rng.randint(1, 2_147_483_647) for _ in range(3))
        self._previous_grab = owner.window.grab_current()
        self.window = tk.Toplevel(owner.window)
        self.window.withdraw()
        self.window.title("Odchyłki serii syntetyków")
        self.window.resizable(True, True)
        configure_minimizable_modal(self.window)
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        width, height = min(1200, sw - 60), min(900, sh - 80)
        self.window.geometry(f"{width}x{height}+{max(0, (sw-width)//2)}+{max(0, (sh-height)//2)}")
        self.window.minsize(min(740, width), min(560, height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", self._on_escape, add="+")
        self.window.bind("<MouseWheel>", self._wheel, add="+")
        self.window.bind("<Button-4>", self._wheel, add="+")
        self.window.bind("<Button-5>", self._wheel, add="+")
        self.window.bind("<Destroy>", self._destroyed, add="+")
        self._title_font = tkfont.nametofont("TkDefaultFont").copy()
        self._title_font.configure(size=self._title_font.cget("size") + 3, weight="bold")
        self._bold_font = tkfont.nametofont("TkDefaultFont").copy()
        self._bold_font.configure(weight="bold")

        shell = ttk.Frame(self.window, padding=14)
        shell.grid(sticky="nsew")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Odchyłki serii", font=self._title_font).pack(side="left")
        context = "MT · pełna scena" if self.base.task_target == "plate" else "MZ · wycinek tablicy"
        ttk.Label(header, text=context).pack(side="right")
        has_legacy = any(legacy_amount(self.draft.legacy, field) for spec in SPECS for field in spec.fields)
        intro = "Wartość bazowa → odchylenie ± → zakres w serii. Zastosowanie włącza tryb Ręczny."
        if has_legacy:
            intro += " Parametry „Z presetu” zachowują dotychczasowe losowanie, dopóki ich nie zmienisz."
        self.intro = ttk.Label(shell, text=intro, wraplength=950, justify="left")
        self.intro.grid(row=1, column=0, sticky="ew", pady=(6, 12))
        shell.bind("<Configure>", lambda event: self.intro.configure(wraplength=max(300, event.width-28)), add="+")

        self.preview_split = ttk.Panedwindow(shell, orient="vertical")
        self.preview_split.grid(row=2, column=0, sticky="nsew")
        body = self.settings_frame = ttk.Frame(self.preview_split)
        self.preview_split.add(body, weight=2)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        style = ttk.Style(self.window)
        background = style.lookup("TFrame", "background") or self.window.cget("background")
        foreground = style.lookup("TLabel", "foreground") or "#202020"
        self.nav = tk.Listbox(body, width=24, exportselection=False, activestyle="none",
                              borderwidth=0, highlightthickness=0, background=background,
                              foreground=foreground, font="TkDefaultFont")
        self.nav.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        self.groups = [group for group in GROUP_LABELS if any(s.group == group and self.base.task_target in s.targets for s in SPECS)]
        self.nav.bind("<<ListboxSelect>>", self._select_group)
        editor = ttk.Frame(body)
        editor.grid(row=0, column=1, sticky="nsew")
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)
        controls = ttk.Frame(editor)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.group_title = ttk.Label(controls, font=self._bold_font)
        self.group_title.pack(side="left")
        self.advanced = tk.BooleanVar(self.window, value=False)
        ttk.Checkbutton(controls, text="Zaawansowane", variable=self.advanced, command=self._toggle_advanced).pack(side="right")
        self.canvas = tk.Canvas(editor, background=background, highlightthickness=0, width=300)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(editor, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.fields = ttk.Frame(self.canvas)
        self._body_item = self.canvas.create_window(0, 0, window=self.fields, anchor="nw")
        self.fields.columnconfigure(0, weight=1)
        self.fields.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._resize_fields)
        actions = ttk.Frame(editor)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(actions, text="Stałe w tej grupie", command=lambda: self._set_group(False)).pack(side="left")
        ttk.Button(actions, text="Losuj dostępne", command=lambda: self._set_group(True)).pack(side="left", padx=(6, 0))

        preview = ttk.Frame(self.preview_split, padding=(0, 8, 0, 0))
        self.preview_split.add(preview, weight=3)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(2, weight=1)
        top = ttk.Frame(preview)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Podgląd próbek", font=self._bold_font).pack(side="left", padx=(0, 10))
        self._preview_titles = ("Baza", "Próbka 1", "Próbka 2", "Próbka 3")
        self.preview_view = tk.StringVar(self.window, value="Próbka 1")
        view_selector = ttk.Combobox(top, textvariable=self.preview_view, state="readonly", width=19,
                                    values=(*self._preview_titles, "Porównanie 2×2"))
        view_selector.pack(side="left")
        view_selector.bind("<<ComboboxSelected>>", lambda event: self._layout_preview_cells())
        ttk.Button(top, text="Losuj inne próbki", command=self.reroll).pack(side="right")
        self.expand_preview_button = ttk.Button(top, text="Powiększ podgląd", command=self._toggle_preview_expansion)
        self.expand_preview_button.pack(side="right", padx=(6, 8))
        self.preview_status = ttk.Label(preview, text="", wraplength=920)
        self.preview_status.grid(row=1, column=0, sticky="ew", pady=(2, 5))
        self.preview_area = ttk.Frame(preview, height=300)
        self.preview_area.grid(row=2, column=0, sticky="nsew")
        self.preview_area.grid_propagate(False)
        self.preview_area.bind("<Configure>", self._preview_resized)
        self.preview_labels = []
        self.preview_values = []
        self.preview_cells = []
        for i, title in enumerate(self._preview_titles):
            cell = ttk.Frame(self.preview_area, padding=(4, 2))
            cell.columnconfigure(0, weight=1)
            cell.rowconfigure(1, weight=1)
            ttk.Label(cell, text=title, font=self._bold_font).grid(row=0, column=0, sticky="w")
            label = ttk.Label(cell, text="Brak obrazu", anchor="center")
            label.grid(row=1, column=0, sticky="nsew")
            label.bind("<Configure>", self._preview_resized)
            label.bind("<Double-Button-1>", lambda event, index=i: self._focus_preview(index))
            value = ttk.Label(cell, text="", anchor="w", wraplength=180)
            value.grid(row=2, column=0, sticky="ew")
            self.preview_cells.append(cell)
            self.preview_labels.append(label)
            self.preview_values.append(value)
        self._preview_profiles = []
        self._selected_field = None

        footer = ttk.Frame(shell)
        footer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.status = ttk.Label(footer, text="Zmiany są zapisywane dopiero po zastosowaniu.", wraplength=500)
        self.status.grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Button(footer, text="Anuluj", command=self.close).grid(row=0, column=1, padx=(0, 8))
        self.apply_button = ttk.Button(footer, text="Zastosuj odchyłki", command=self.apply)
        self.apply_button.grid(row=0, column=2)
        self.active_group = "weather" if "weather" in self.groups else self.groups[0]
        self._update_nav()
        self._render_group()
        self._layout_preview_cells()
        self._sash_after = self.window.after_idle(self._place_initial_sash)
        owner._manual_randomness_dialog = self
        if show:
            self.window.deiconify()
            self.window.update_idletasks()
            self.window.grab_set()
            self.window.focus_set()
        self._schedule_preview()

    def focus(self):
        self.window.deiconify()
        self.window.lift()
        self.window.grab_set()
        self.window.focus_set()

    def _place_initial_sash(self):
        self._sash_after = None
        if not self.closed and not self._preview_expanded and self.preview_split.winfo_height() > 200:
            self.preview_split.sashpos(0, min(260, max(190, int(self.preview_split.winfo_height() * .40))))

    def _layout_preview_cells(self):
        comparison = self.preview_view.get() == "Porównanie 2×2"
        selected = self._preview_titles.index(self.preview_view.get()) if not comparison else None
        for index in (0, 1):
            weight = 1 if comparison or index == 0 else 0
            self.preview_area.columnconfigure(index, weight=weight, uniform="samples" if comparison else "")
            self.preview_area.rowconfigure(index, weight=weight, uniform="samples" if comparison else "")
        for i, cell in enumerate(self.preview_cells):
            cell.grid_forget()
            if comparison:
                cell.grid(row=i//2, column=i%2, sticky="nsew")
            elif i == selected:
                cell.grid(row=0, column=0, sticky="nsew")
        self._preview_resized()

    def _toggle_preview_expansion(self):
        if self._preview_expanded:
            self.preview_split.insert(0, self.settings_frame, weight=2)
            self._preview_expanded = False
            self.expand_preview_button.configure(text="Powiększ podgląd")
            self.window.update_idletasks()
            if self._saved_sash is not None:
                self.preview_split.sashpos(0, self._saved_sash)
        else:
            if not self._commit_entries():
                return
            self._saved_sash = self.preview_split.sashpos(0)
            self.preview_split.forget(self.settings_frame)
            self._preview_expanded = True
            self.expand_preview_button.configure(text="Pokaż ustawienia")
        self._preview_resized()

    def _focus_preview(self, index):
        self.preview_view.set(self._preview_titles[index])
        self._layout_preview_cells()
        if not self._preview_expanded:
            self._toggle_preview_expansion()

    def _on_escape(self, event=None):
        if self._preview_expanded:
            self._toggle_preview_expansion()
        else:
            self.close()
        return "break"

    def _preview_resized(self, event=None):
        if self.closed:
            return
        if self._resize_after:
            self.window.after_cancel(self._resize_after)
        self._resize_after = self.window.after(60, self._redraw_preview_images)

    def _update_nav(self):
        self.nav.delete(0, "end")
        for group in self.groups:
            specs = [s for s in SPECS if s.group == group and self.base.task_target in s.targets]
            count = sum(self.draft.state(s)[0] != "fixed" and not unavailable_reason(s, self.base) for s in specs)
            self.nav.insert("end", f"{GROUP_LABELS[group]}  ·  {count}")
        index = self.groups.index(self.active_group)
        self.nav.selection_set(index)
        self.nav.activate(index)

    def _select_group(self, event=None):
        selection = self.nav.curselection()
        if not selection:
            return
        group = self.groups[selection[0]]
        if group == self.active_group:
            return
        if not self._commit_entries():
            self._update_nav()
            return
        self.active_group = group
        self._render_group()

    def _toggle_advanced(self):
        if self._commit_entries():
            self._render_group()
        else:
            self.advanced.set(not self.advanced.get())

    def _resize_fields(self, event):
        self.canvas.itemconfigure(self._body_item, width=max(1, event.width))
        for row in self.rows.values():
            row["summary"].configure(wraplength=max(150, event.width - 25))

    def _render_group(self):
        for widget in self.fields.winfo_children():
            widget.destroy()
        self.rows = {}
        self.group_title.configure(text=GROUP_LABELS[self.active_group])
        specs = [s for s in SPECS if s.group == self.active_group and self.base.task_target in s.targets]
        # Configured advanced fields remain visible, even before expanding the rest.
        specs = [s for s in specs if not s.advanced or self.advanced.get() or self.draft.state(s)[0] != "fixed"]
        for index, spec in enumerate(specs):
            frame = ttk.Frame(self.fields, padding=(0, 9, 10, 9))
            frame.grid(row=index*2, column=0, sticky="ew")
            frame.columnconfigure(0, weight=1)
            heading = ttk.Frame(frame)
            heading.grid(row=0, column=0, sticky="ew")
            heading.columnconfigure(0, weight=1)
            ttk.Label(heading, text=spec.label, font=self._bold_font).grid(row=0, column=0, sticky="w")
            mode, delta = self.draft.state(spec)
            enabled = tk.BooleanVar(self.window, value=mode != "fixed")
            toggle = ttk.Checkbutton(heading, text="Losuj", variable=enabled, command=lambda s=spec: self._toggle(s))
            toggle.grid(row=0, column=2, sticky="e")
            base_text = " – ".join(_fmt(float(getattr(self.base, key)), spec) for key in spec.fields)
            ttk.Label(frame, text=f"Baza: {base_text}").grid(row=1, column=0, sticky="w", pady=(2, 3))
            control = ttk.Frame(frame)
            control.grid(row=2, column=0, sticky="ew")
            control.columnconfigure(0, weight=1)
            amount = tk.DoubleVar(self.window, value=delta)
            text = tk.StringVar(self.window, value=f"{delta:.3f}".rstrip("0").rstrip(".").replace(".", ","))
            ceiling = 180 if spec.angle else spec.maximum-spec.minimum
            scale = ttk.Scale(control, from_=0, to=ceiling, variable=amount, command=lambda value, s=spec: self._slide(s, value))
            scale.grid(row=0, column=0, sticky="ew", padx=(0, 10))
            ttk.Label(control, text="±").grid(row=0, column=1, padx=(0, 5))
            entry = ttk.Entry(control, textvariable=text, width=9)
            entry.grid(row=0, column=2)
            if spec.unit:
                ttk.Label(control, text=spec.unit).grid(row=0, column=3, padx=(4, 0))
            summary = ttk.Label(frame, justify="left", wraplength=max(200, self.canvas.winfo_width()-25))
            summary.grid(row=3, column=0, sticky="ew", pady=(4, 0))
            legacy_button = ttk.Button(heading, text="Własny zakres", command=lambda s=spec: self._use_range(s))
            legacy_button.grid(row=0, column=1, padx=(8, 10))
            self.rows[spec.key] = dict(spec=spec, enabled=enabled, amount=amount, text=text, toggle=toggle,
                                       scale=scale, entry=entry, summary=summary, legacy_button=legacy_button,
                                       dirty=False, updating=False)
            text.trace_add("write", lambda *args, s=spec: self._typed(s))
            entry.bind("<Return>", lambda event, s=spec: self._commit_row(s))
            entry.bind("<FocusOut>", lambda event, s=spec: self._commit_row(s))
            self._refresh_row(spec)
            ttk.Separator(self.fields).grid(row=index*2+1, column=0, sticky="ew")
        self.canvas.yview_moveto(0)
        if self._selected_field not in {s.key for s in specs}:
            self._selected_field = specs[0].key if specs else None
        self._show_sample_values()

    def _typed(self, spec):
        row = self.rows.get(spec.key)
        if row and not row["updating"]:
            row["dirty"] = True

    def _refresh_row(self, spec):
        row = self.rows[spec.key]
        mode, delta = self.draft.state(spec)
        reason = unavailable_reason(spec, self.base)
        row["updating"] = True
        row["enabled"].set(mode != "fixed")
        row["amount"].set(delta)
        row["text"].set(f"{delta:.3f}".rstrip("0").rstrip(".").replace(".", ",") or "0")
        row["updating"] = False
        row["dirty"] = False
        row["toggle"].configure(state="disabled" if reason else "normal")
        state = "normal" if mode == "range" and not reason else "disabled"
        row["entry"].configure(state=state)
        row["scale"].configure(state=state)
        row["legacy_button"].grid() if mode == "legacy" and not reason else row["legacy_button"].grid_remove()
        if reason:
            description = reason
        elif mode == "legacy":
            amounts = " / ".join(f"{legacy_amount(self.draft.legacy, key):g}%" for key in spec.fields)
            description = f"Z presetu · efektywna siła {amounts}. Dotychczasowy sposób losowania."
        elif mode == "fixed":
            description = "W serii: stała wartość bazowa."
        else:
            low, high = range_bounds(spec, self.base, delta)
            if spec.angle:
                description = f"W serii: kierunek bazowy ±{_fmt(delta, spec)} · z przejściem przez ±180°."
            elif len(spec.fields) > 1:
                values = [float(getattr(self.base, key)) for key in spec.fields]
                description = f"W serii: od {_fmt(values[0]+low, spec)} do {_fmt(values[-1]+high, spec)} · wspólne przesunięcie min–max, stała szerokość."
            else:
                description = f"W serii: {_fmt(low, spec)} – {_fmt(high, spec)}. Najczęściej blisko bazy."
        row["summary"].configure(text=description)

    def _toggle(self, spec):
        row = self.rows[spec.key]
        enabled = row["enabled"].get()
        delta = self.draft.state(spec)[1] or self.draft.default_delta(spec)
        self._set(spec, enabled, delta)

    def _use_range(self, spec):
        self._set(spec, True, self.draft.default_delta(spec))

    def _slide(self, spec, value):
        row = self.rows.get(spec.key)
        if row is None or row["updating"]:
            return
        step = spec.step
        self._set(spec, True, round(round(float(value)/step)*step, 6))

    def _set(self, spec, enabled, delta):
        try:
            self.draft.set(spec, enabled, delta)
        except (TypeError, ValueError) as exc:
            self.status.configure(text=f"{spec.label}: {exc}")
            return False
        self._selected_field = spec.key
        self._refresh_row(spec)
        self._update_nav()
        self.status.configure(text="Niezapisane zmiany · 0 oznacza stałą wartość.")
        self._schedule_preview()
        return True

    def _commit_row(self, spec):
        row = self.rows.get(spec.key)
        if row is None or not row["dirty"] or self.closed:
            return True
        return self._set(spec, row["enabled"].get(), row["text"].get())

    def _commit_entries(self):
        return all(self._commit_row(row["spec"]) for row in list(self.rows.values()))

    def _set_group(self, enabled):
        if not self._commit_entries():
            return
        for spec in SPECS:
            if spec.group == self.active_group and not unavailable_reason(spec, self.base):
                self.draft.set(spec, enabled, self.draft.state(spec)[1] or self.draft.default_delta(spec))
        self._update_nav()
        self._render_group()
        self.status.configure(text="Niezapisane zmiany w wybranej grupie.")
        self._schedule_preview()

    def _wheel(self, event):
        if self._preview_expanded:
            return "break"
        if event.widget is not self.nav:
            delta = getattr(event, "delta", 0)
            direction = -1 if delta > 0 or getattr(event, "num", None) == 4 else 1
            self.canvas.yview_scroll(direction * max(1, abs(int(delta))//120), "units")
        return "break"

    def _schedule_preview(self):
        if self.closed:
            return
        self._revision += 1
        self._cancel_event.set()
        self._pending = None
        if self._preview_after:
            self.window.after_cancel(self._preview_after)
        self._preview_after = self.window.after(300, self._request_preview)

    def reroll(self):
        if not self._commit_entries():
            return
        self.preview_seeds = tuple(self._seed_rng.randint(1, 2_147_483_647) for _ in range(3))
        self._schedule_preview()

    def _request_preview(self):
        self._preview_after = None
        if self.closed:
            return
        sample = getattr(self.owner, "_current_sample", None)
        if not sample or not Path(sample).is_file():
            self.preview_status.configure(text="Wybierz obraz w podglądzie efektów, aby zobaczyć warianty serii.")
            return
        profile = replace(self.base, manual_randomness=self.draft.export(), randomness_mode="manual")
        self._pending = (self._revision, Path(sample), profile, self.preview_seeds)
        self.preview_status.configure(text="Przeliczam warianty…")
        self._start_preview()

    def _start_preview(self):
        if self._busy or self._pending is None or self.closed:
            return
        if self._pending[0] != self._revision:
            self._pending = None
            return
        revision, path, profile, seeds = self._pending
        self._pending = None
        self._busy = True
        cancelled = self._cancel_event = threading.Event()
        results_queue = self._results

        def worker():
            results, error = [], ""
            try:
                jobs = [(replace(profile, randomness_mode="fixed"), seeds[0])] + [(profile, seed) for seed in seeds]
                for job_profile, seed in jobs:
                    if cancelled.is_set():
                        break
                    ok, message, payload = preview_augmentation_image(path, job_profile, sample_seed=seed, max_pixels=650_000)
                    if not ok:
                        error = message
                        break
                    results.append(payload)
            except Exception as exc:
                error = str(exc)
            results_queue.put((revision, results, error))

        threading.Thread(target=worker, name="z4-deviation-preview", daemon=True).start()
        self._poll_after = self.window.after(60, self._poll_preview)

    def _poll_preview(self):
        self._poll_after = None
        if self.closed:
            return
        try:
            revision, results, error = self._results.get_nowait()
        except queue.Empty:
            self._poll_after = self.window.after(60, self._poll_preview)
            return
        self._busy = False
        if revision == self._revision:
            if error:
                self.preview_status.configure(text=f"Podgląd niedostępny: {error}")
            elif len(results) == 4:
                try:
                    self._display_previews(results)
                except Exception as exc:
                    self.preview_status.configure(text=f"Nie udało się wyświetlić podglądu: {exc}")
        self._start_preview()

    def _display_previews(self, results):
        from PIL import Image
        self._preview_sources = [Image.fromarray(payload["augmented_rgb"]) for payload in results]
        self._preview_render_keys = [None] * len(results)
        self._preview_profiles = [payload["sample_profile"] for payload in results]
        self.preview_status.configure(text="Aktualny podgląd · ten sam mechanizm odchyłek co przy generowaniu serii.")
        self._show_sample_values()
        self._redraw_preview_images()

    def _redraw_preview_images(self):
        if self._resize_after:
            self.window.after_cancel(self._resize_after)
        self._resize_after = None
        if self.closed or not self._preview_sources:
            return
        from PIL import Image, ImageTk
        if len(self._images) != len(self._preview_sources):
            self._images = [None] * len(self._preview_sources)
        comparison = self.preview_view.get() == "Porównanie 2×2"
        fallback_width = max(200, (self.window.winfo_width()-64)//(2 if comparison else 1))
        for index, (label, value, source) in enumerate(zip(self.preview_labels, self.preview_values, self._preview_sources)):
            width = label.winfo_width() if label.winfo_width() > 8 else fallback_width
            height = label.winfo_height() if label.winfo_height() > 8 else 240
            scale = min(max(1, width-8)/source.width, max(1, height-8)/source.height)
            size = (max(1, int(source.width*scale)), max(1, int(source.height*scale)))
            if self._preview_render_keys[index] == size:
                continue
            photo = ImageTk.PhotoImage(source.resize(size, Image.Resampling.LANCZOS), master=self.window)
            self._images[index] = photo
            self._preview_render_keys[index] = size
            label.configure(image=photo, text="")
            value.configure(wraplength=max(150, width-8))
        self.preview_status.configure(wraplength=max(300, self.preview_area.winfo_width()-8))

    def _show_sample_values(self):
        from ..training.augmentation_ranges import SPEC_BY_KEY
        spec = SPEC_BY_KEY.get(self._selected_field)
        if not spec:
            return
        for label, profile in zip(self.preview_values, self._preview_profiles):
            values = " – ".join(_fmt(float(profile[field]), spec) for field in spec.fields)
            label.configure(text=f"{spec.label}: {values}")

    def apply(self):
        if self.closed or not self._commit_entries():
            return
        self.owner._manual_randomness = self.draft.export()
        self.owner.randomness_mode_var.set(self.owner._randomness_mode_to_label("manual"))
        self.close()
        self.owner._refresh_preview(redraw_only=False)

    def _destroyed(self, event):
        if event.widget is self.window and not self.closed:
            self.close(destroy=False)

    def close(self, *, destroy=True):
        if self.closed:
            return
        self.closed = True
        self._cancel_event.set()
        self._pending = None
        for job in (self._preview_after, self._poll_after, self._resize_after, self._sash_after):
            if job:
                try:
                    self.window.after_cancel(job)
                except tk.TclError:
                    pass
        self._preview_after = self._poll_after = None
        self._resize_after = self._sash_after = None
        try:
            if self.window.grab_current() is self.window:
                self.window.grab_release()
            if destroy:
                self.window.destroy()
        except tk.TclError:
            pass
        if getattr(self.owner, "_manual_randomness_dialog", None) is self:
            self.owner._manual_randomness_dialog = None
        try:
            if self._previous_grab is not None and self._previous_grab.winfo_exists() and self._previous_grab.winfo_viewable():
                self._previous_grab.grab_set()
        except tk.TclError:
            pass
