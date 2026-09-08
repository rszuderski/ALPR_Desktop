#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI for MZ character representation diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..config import CONFIG, logger
from ..training import (
    CHARACTER_REPRESENTATION_THRESHOLD_POLICY,
    CharacterClassMapValidationError,
    CharacterClassDistribution,
    analyze_character_class_distribution,
    collect_character_train_augmentation_source_pool,
    plan_character_train_augmentation,
    preview_character_train_synthetic_counts_from_pool,
    save_character_distribution_artifacts,
    save_character_class_distribution_csv,
    save_character_class_distribution_json,
)
from .dataset_display import build_dataset_display_ref


_STATUS_LABELS = {
    "OK": "OK",
    "LOW": "Mało próbek",
    "LOW_DIVERSITY": "Mała różnorodność",
    "LOW+LOW_DIVERSITY": "Mało i wąsko",
    "CRITICAL": "Brak klasy",
}

_STATUS_COLORS = {
    "OK": "#2faa66",
    "LOW": "#d89a20",
    "LOW_DIVERSITY": "#c9a521",
    "LOW+LOW_DIVERSITY": "#e0792f",
    "CRITICAL": "#d94b4b",
}


def open_character_class_distribution_dialog(
    host,
    *,
    dataset_root: Path | str | None = None,
    yaml_path: Path | str | None = None,
    read_only: bool | None = None,
    context: str = "pz2",
) -> None:
    parent = getattr(host, "frame", None) or getattr(getattr(host, "app", None), "root", None)
    if dataset_root is None:
        target = "char"
        try:
            target = CONFIG.normalize_task_target(host._get_selected_training_target())
        except Exception:
            target = "char"
        if target != "char":
            messagebox.showinfo(
                "Uzupełnianie znaków MZ",
                "Ta analiza dotyczy modelu znaków MZ. Przełącz Z4 na tor znaków i wybierz wariant datasetu.",
                parent=parent,
            )
            return

        resolver = getattr(host, "_resolve_training_dataset_yaml_path", None)
        yaml_path = resolver() if callable(resolver) else None
        if yaml_path is None:
            messagebox.showwarning(
                "Brak wariantu datasetu",
                "Najpierw wybierz aktywny wariant datasetu znaków.",
                parent=parent,
            )
            return
        try:
            yaml_path = Path(yaml_path)
            dataset_root = yaml_path.parent if yaml_path.is_file() else yaml_path
        except Exception:
            messagebox.showerror(
                "Błąd datasetu",
                "Nie udało się odczytać ścieżki aktywnego wariantu datasetu.",
                parent=parent,
            )
            return
        if read_only is None:
            read_only = True
    else:
        try:
            dataset_root = Path(dataset_root)
            yaml_path = Path(yaml_path) if yaml_path is not None else dataset_root / "data.yaml"
        except Exception:
            messagebox.showerror(
                "Błąd datasetu",
                "Nie udało się odczytać ścieżki wariantu datasetu.",
                parent=parent,
            )
            return
        if read_only is None:
            read_only = False

    _CharacterClassDistributionDialog(
        host,
        Path(dataset_root),
        Path(yaml_path) if yaml_path is not None else None,
        read_only=bool(read_only),
        context=context,
    ).show()


class _CharacterClassDistributionDialog:
    def __init__(
        self,
        host,
        dataset_root: Path,
        yaml_path: Path | None,
        *,
        read_only: bool = True,
        context: str = "pz2",
    ):
        self.host = host
        self.dataset_root = Path(dataset_root)
        self.yaml_path = Path(yaml_path) if yaml_path is not None else None
        self.read_only = bool(read_only)
        self.context = str(context or "pz2")
        self.app = getattr(host, "app", None)
        self.palette = getattr(self.app, "palette", {}) if self.app is not None else {}
        self.window: tk.Toplevel | None = None
        self.progress: ttk.Progressbar | None = None
        self.status_var = tk.StringVar(value="Przygotowuję analizę reprezentacji znaków MZ...")
        self.dataset_var = tk.StringVar(value=self._dataset_label())
        self.target_ratio_var = tk.StringVar(value="0.50")
        self.reference_count_var = tk.StringVar(value="")
        self.synthetic_total_var = tk.StringVar(value="Syntetyki train: +0")
        self.threshold_policy_var = tk.StringVar(value=f"AUTO ({CHARACTER_REPRESENTATION_THRESHOLD_POLICY})")
        self.requested_extra_by_symbol: dict[str, int] = {}
        self._chart_bar_slots: dict[str, tuple[float, float, float, float, float, float, float]] = {}
        self._chart_reference_line_slot: tuple[float, ...] | None = None
        self._chart_press: dict | None = None
        self._symbol_editor: tk.Toplevel | None = None
        self._preview_plan = None
        self._preview_after_id: str | None = None
        self._preview_serial = 0
        self._source_pool_candidates = ()
        self._source_pool_existing: dict[str, int] = {}
        self._source_pool_ready = False
        self._source_pool_serial = 0
        self._quick_preview_cache_key: tuple[tuple[str, int], ...] | None = None
        self._quick_preview_cache: dict[str, int] = {}
        self.result: CharacterClassDistribution | None = None
        self.summary_value_labels: dict[str, tk.Label] = {}
        self.content_frame: tk.Frame | None = None
        self.table: ttk.Treeview | None = None
        self.table_shell: tk.LabelFrame | None = None
        self.table_toggle_btn: ttk.Button | None = None
        self._table_visible = False
        self.chart: tk.Canvas | None = None
        self.save_before_btn: ttk.Button | None = None
        self.plan_btn: ttk.Button | None = None
        self.export_csv_btn: ttk.Button | None = None
        self.export_json_btn: ttk.Button | None = None

    def show(self) -> None:
        parent = getattr(self.host, "frame", None)
        root = getattr(self.app, "root", None) or (parent.winfo_toplevel() if parent is not None else None)
        self.window = tk.Toplevel(root or parent)
        title_suffix = "raport" if self.read_only else "uzupełnienie PZ1"
        self.window.title(f"Uzupełnianie reprezentacji znaków MZ - {title_suffix}")
        self.window.minsize(980, 620)
        self.window.geometry("1160x740")
        try:
            self.window.resizable(True, True)
            self.window.attributes("-toolwindow", False)
        except Exception:
            pass

        self._build()
        self._center()
        self._start_analysis()

    def _build(self) -> None:
        assert self.window is not None
        bg = self.palette.get("bg", "#1e1f22")
        fg = self.palette.get("fg", "#f3f3f3")
        muted = self.palette.get("muted", "#b7bcc6")
        panel = self.palette.get("panel", "#25262b")
        panel_alt = self.palette.get("panel_alt", "#2c2d33")

        self.window.configure(bg=bg)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        root = tk.Frame(self.window, bg=bg, padx=14, pady=12)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)

        title = tk.Label(
            root,
            text="Uzupełnianie reprezentacji znaków MZ",
            bg=bg,
            fg=fg,
            font=("Segoe UI Semibold", 15),
            anchor=tk.W,
        )
        title.grid(row=0, column=0, sticky="ew")

        intro = tk.Label(
            root,
            text=(
                "Sprawdzamy, które znaki 0-9 i A-Z są za rzadkie w train. "
                "Celem nie jest idealnie równy rozkład, tylko doprowadzenie słabych znaków do progu, "
                "żeby model MZ miał z czego się ich nauczyć."
            ),
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=980,
        )
        intro.grid(row=1, column=0, sticky="ew", pady=(3, 10))

        meta = tk.Frame(root, bg=panel_alt, highlightthickness=1, highlightbackground=self._border())
        meta.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        meta.grid_columnconfigure(1, weight=1)
        tk.Label(
            meta,
            text="Dataset",
            bg=panel_alt,
            fg=muted,
            font=("Segoe UI Semibold", 9),
            padx=8,
            pady=6,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            meta,
            textvariable=self.dataset_var,
            bg=panel_alt,
            fg=fg,
            font=("Segoe UI", 9),
            anchor=tk.W,
            padx=8,
            pady=6,
        ).grid(row=0, column=1, sticky="ew")
        tk.Label(
            meta,
            text="Miarka",
            bg=panel_alt,
            fg=muted,
            font=("Segoe UI Semibold", 9),
            padx=8,
            pady=6,
        ).grid(row=1, column=0, sticky="w")
        reference_shell = tk.Frame(meta, bg=panel_alt)
        reference_shell.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        tk.Label(
            reference_shell,
            textvariable=self.reference_count_var,
            bg=panel_alt,
            fg=fg,
            font=("Segoe UI Semibold", 10),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            reference_shell,
            text="przykładów znaku w train; przesuń poziomą linię na histogramie",
            bg=panel_alt,
            fg=muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
            padx=8,
        ).grid(row=0, column=1, sticky="w")
        tk.Label(
            meta,
            text="Po ustawieniu",
            bg=panel_alt,
            fg=muted,
            font=("Segoe UI Semibold", 9),
            padx=8,
            pady=6,
        ).grid(row=2, column=0, sticky="w")
        tk.Label(
            meta,
            textvariable=self.synthetic_total_var,
            bg=panel_alt,
            fg=self._summary_accent_fg(panel_alt),
            font=("Segoe UI Semibold", 10),
            anchor=tk.W,
            padx=8,
            pady=6,
        ).grid(row=2, column=1, sticky="ew")

        content = tk.Frame(root, bg=bg)
        self.content_frame = content
        content.grid(row=3, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=0, minsize=290)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=0)

        summary = tk.LabelFrame(
            content,
            text=" Podsumowanie ",
            bg=panel,
            fg=fg,
            padx=8,
            pady=8,
            font=("Segoe UI Semibold", 9),
            highlightthickness=1,
            highlightbackground=self._border(),
        )
        summary.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        summary.grid_columnconfigure(1, weight=1)
        self._build_summary_rows(summary)

        chart_shell = tk.LabelFrame(
            content,
            text=" Wykres train ",
            bg=panel,
            fg=fg,
            padx=8,
            pady=8,
            font=("Segoe UI Semibold", 9),
            highlightthickness=1,
            highlightbackground=self._border(),
        )
        chart_shell.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        chart_shell.grid_columnconfigure(0, weight=1)
        chart_shell.grid_rowconfigure(0, weight=1)
        self.chart = tk.Canvas(
            chart_shell,
            height=210,
            bg=panel,
            highlightthickness=0,
            bd=0,
        )
        self.chart.grid(row=0, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())
        if not self.read_only:
            self.chart.bind("<Button-1>", self._on_chart_mouse_down)
            self.chart.bind("<B1-Motion>", self._on_chart_mouse_drag)
            self.chart.bind("<ButtonRelease-1>", self._on_chart_mouse_up)

        table_section = tk.Frame(content, bg=bg)
        table_section.grid(row=1, column=1, sticky="ew")
        table_section.grid_columnconfigure(0, weight=1)
        table_section.grid_rowconfigure(1, weight=1)
        table_toggle_bar = tk.Frame(
            table_section,
            bg=panel,
            highlightthickness=1,
            highlightbackground=self._border(),
        )
        table_toggle_bar.grid(row=0, column=0, sticky="ew")
        table_toggle_bar.grid_columnconfigure(0, weight=1)
        tk.Label(
            table_toggle_bar,
            text="Szczegółowa tabela klas jest ukryta. Otwórz ją tylko wtedy, gdy chcesz sprawdzić liczby w wierszach.",
            bg=panel,
            fg=muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            padx=8,
            pady=6,
        ).grid(row=0, column=0, sticky="ew")
        self.table_toggle_btn = ttk.Button(
            table_toggle_bar,
            text="Pokaż tabelę klas",
            command=self._toggle_table_section,
        )
        self.table_toggle_btn.grid(row=0, column=1, sticky="e", padx=8, pady=5)

        self.table_shell = tk.LabelFrame(
            table_section,
            text=" Klasy znaków ",
            bg=panel,
            fg=fg,
            padx=8,
            pady=8,
            font=("Segoe UI Semibold", 9),
            highlightthickness=1,
            highlightbackground=self._border(),
        )
        self.table_shell.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.table_shell.grid_rowconfigure(0, weight=1)
        self.table_shell.grid_columnconfigure(0, weight=1)
        self._build_table(self.table_shell)
        self.table_shell.grid_remove()

        footer = tk.Frame(root, bg=bg)
        footer.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e", padx=(8, 8))
        actions = tk.Frame(footer, bg=bg)
        actions.grid(row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))
        self.save_before_btn = None
        self.plan_btn = ttk.Button(
            actions,
            text="Zatwierdź",
            command=self._accept_current_settings,
            state=tk.DISABLED,
        )
        if not self.read_only:
            self.plan_btn.grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.export_csv_btn = None
        self.export_json_btn = ttk.Button(
            actions,
            text="Eksport rozkładu",
            command=self._export_distribution,
            state=tk.DISABLED,
        )
        self.export_json_btn.grid(row=0, column=1, sticky="e", padx=(0, 6))
        if self.read_only:
            ttk.Button(actions, text="Zamknij", command=self.window.destroy).grid(row=0, column=2, sticky="e")

    def _toggle_table_section(self) -> None:
        if self.table_shell is None:
            return
        self._table_visible = not self._table_visible
        try:
            if self._table_visible:
                self.table_shell.grid()
                if self.content_frame is not None:
                    self.content_frame.grid_rowconfigure(1, weight=1)
                if self.table_toggle_btn is not None:
                    self.table_toggle_btn.configure(text="Ukryj tabelę klas")
            else:
                self.table_shell.grid_remove()
                if self.content_frame is not None:
                    self.content_frame.grid_rowconfigure(1, weight=0)
                if self.table_toggle_btn is not None:
                    self.table_toggle_btn.configure(text="Pokaż tabelę klas")
            if self.chart is not None:
                self.chart.after_idle(self._draw_chart)
        except Exception:
            pass

    def _build_summary_rows(self, parent: tk.Widget) -> None:
        labels = [
            ("layout", "Układ danych"),
            ("class_map", "Mapa klas"),
            ("diagnostic_split", "Kryterium"),
            ("total_labels", "Etykiety"),
            ("range", "Min / mediana / max"),
            ("target", "Miarka / syntetyki"),
            ("ratio", "Stosunek max/min"),
            ("zero", "Braki krytyczne"),
            ("low", "Mało próbek"),
            ("diversity", "Mała różnorodność"),
            ("files", "Pliki / błędy"),
        ]
        for row_index, (key, label) in enumerate(labels):
            tk.Label(
                parent,
                text=label,
                bg=self.palette.get("panel", "#25262b"),
                fg=self.palette.get("muted", "#b7bcc6"),
                font=("Segoe UI Semibold", 8),
                anchor=tk.W,
                padx=4,
                pady=5,
            ).grid(row=row_index, column=0, sticky="nw")
            value_lbl = tk.Label(
                parent,
                text="-",
                bg=self.palette.get("panel", "#25262b"),
                fg=self.palette.get("fg", "#f3f3f3"),
                font=("Segoe UI", 8),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=165,
                padx=4,
                pady=5,
            )
            value_lbl.grid(row=row_index, column=1, sticky="ew")
            self.summary_value_labels[key] = value_lbl

    def _build_table(self, parent: tk.Widget) -> None:
        columns = (
            "symbol",
            "train",
            "unique_train",
            "count_status",
            "diversity_status",
            "target",
            "deficit",
            "val",
            "test",
            "total",
            "share",
            "status",
        )
        self.table = ttk.Treeview(parent, columns=columns, show="headings", height=16)
        headings = {
            "symbol": "Znak",
            "train": "Train",
            "unique_train": "Unikalne tablice train",
            "count_status": "Liczebność",
            "diversity_status": "Różnorodność",
            "target": "Po ustawieniu",
            "deficit": "Syntetyki",
            "val": "Val",
            "test": "Test",
            "total": "Razem",
            "share": "Udział train",
            "status": "Status",
        }
        widths = {
            "symbol": 54,
            "train": 76,
            "unique_train": 150,
            "count_status": 110,
            "diversity_status": 125,
            "target": 70,
            "deficit": 76,
            "val": 70,
            "test": 70,
            "total": 78,
            "share": 92,
            "status": 145,
        }
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], minwidth=42, stretch=column in {"unique_train", "status"})
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        for status, color in _STATUS_COLORS.items():
            tag = _status_tag(status)
            try:
                self.table.tag_configure(tag, foreground=color)
            except Exception:
                pass

    def _start_analysis(self) -> None:
        target_ratio = self._target_ratio()
        if self.progress is not None:
            try:
                self.progress.grid()
            except Exception:
                pass
            self.progress.start(12)

        def worker() -> None:
            source_pool: tuple[tuple, dict[str, int]] | None = None
            try:
                result = analyze_character_class_distribution(
                    self.dataset_root,
                    target_ratio=target_ratio,
                )
                if not self.read_only:
                    try:
                        source_pool = collect_character_train_augmentation_source_pool(
                            self.dataset_root,
                            respect_existing_augmented_variants=False,
                        )
                    except Exception:
                        logger.exception("Nie udało się zbudować cache źródeł train dla histogramu MZ")
            except CharacterClassMapValidationError as exc:
                logger.warning("Analiza MZ zatrzymana przez niezgodną mapę klas: %s", exc)
                self._after(lambda: self._show_error(str(exc)))
                return
            except Exception as exc:
                logger.exception("Nie udało się przeanalizować reprezentacji znaków MZ")
                self._after(lambda: self._show_error(str(exc)))
                return
            self._after(lambda: self._apply_result(result, source_pool=source_pool))

        threading.Thread(target=worker, daemon=True).start()

    def _restart_analysis(self) -> None:
        self.result = None
        self._source_pool_ready = False
        self._source_pool_candidates = ()
        self._source_pool_existing = {}
        self._quick_preview_cache_key = None
        self._quick_preview_cache = {}
        if self.save_before_btn is not None:
            self.save_before_btn.configure(state=tk.DISABLED)
        if self.plan_btn is not None:
            self.plan_btn.configure(state=tk.DISABLED)
        if self.export_csv_btn is not None:
            self.export_csv_btn.configure(state=tk.DISABLED)
        if self.export_json_btn is not None:
            self.export_json_btn.configure(state=tk.DISABLED)
        if self.table is not None:
            self.table.delete(*self.table.get_children())
        self.status_var.set("Przeliczam reprezentację znaków w train...")
        self._draw_chart()
        self._start_analysis()

    def _target_ratio(self) -> float:
        raw = str(self.target_ratio_var.get() or "0.50").strip().replace(",", ".")
        try:
            value = float(raw)
        except Exception:
            value = 0.50
        if value > 1.0:
            value /= 100.0
        value = max(0.0, min(2.0, value))
        self.target_ratio_var.set(f"{value:.2f}")
        return value

    def _apply_result(self, result: CharacterClassDistribution, source_pool=None) -> None:
        self.result = result
        self._sync_common_target_from_result(result)
        if source_pool is not None:
            try:
                candidates, existing = source_pool
            except Exception:
                candidates, existing = (), {}
            self._source_pool_candidates = tuple(candidates or ())
            self._source_pool_existing = dict(existing or {})
            self._source_pool_ready = True
            self._quick_preview_cache_key = None
            self._quick_preview_cache = {}
        if self.progress is not None:
            self.progress.stop()
            self.progress.grid_remove()
        if self.save_before_btn is not None:
            self.save_before_btn.configure(state=tk.NORMAL)
        if self.plan_btn is not None:
            self.plan_btn.configure(state=(tk.DISABLED if self.read_only else tk.NORMAL))
        if self.export_csv_btn is not None:
            self.export_csv_btn.configure(state=tk.NORMAL)
        if self.export_json_btn is not None:
            self.export_json_btn.configure(state=tk.NORMAL)
        if self.read_only:
            self.status_var.set("Raport gotowy. To tylko podgląd reprezentacji znaków MZ.")
        else:
            self.status_var.set("Analiza gotowa. Kliknij słupek, wpisz +N albo przeciągnij uchwyt dodatku, a potem kliknij Zatwierdź.")
        self._fill_summary(result)
        self._fill_table(result)
        self._draw_chart()
        if not self.read_only and not self._source_pool_ready:
            self._start_source_pool_preview_cache()
        self._schedule_synthetic_preview()

    def _show_error(self, error_text: str) -> None:
        if self.progress is not None:
            self.progress.stop()
            self.progress.grid_remove()
        self.status_var.set("Nie udało się wykonać analizy.")
        if self.window is not None and self.window.winfo_exists():
            messagebox.showerror("Błąd analizy reprezentacji znaków", error_text, parent=self.window)

    def _start_source_pool_preview_cache(self) -> None:
        if self.read_only or self.result is None:
            return
        self._source_pool_ready = False
        self._source_pool_candidates = ()
        self._source_pool_existing = {}
        self._source_pool_serial += 1
        serial = self._source_pool_serial

        def worker() -> None:
            try:
                candidates, existing = collect_character_train_augmentation_source_pool(
                    self.dataset_root,
                    respect_existing_augmented_variants=False,
                )
            except Exception:
                logger.exception("Nie udało się zbudować cache źródeł train dla histogramu MZ")
                return
            self._after(lambda: self._apply_source_pool_preview_cache(serial, candidates, existing))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_source_pool_preview_cache(self, serial: int, candidates, existing) -> None:
        if self.window is None or not self.window.winfo_exists():
            return
        if serial != self._source_pool_serial:
            return
        self._source_pool_candidates = tuple(candidates or ())
        self._source_pool_existing = dict(existing or {})
        self._source_pool_ready = True
        self._quick_preview_cache_key = None
        self._quick_preview_cache = {}
        if self.requested_extra_by_symbol and self.result is not None:
            self._preview_plan = None
            self._fill_summary(self.result)
            self._fill_table(self.result)
            self._draw_chart()

    def _quick_synthetic_counts_by_symbol(self) -> dict[str, int]:
        if not self._source_pool_ready or not self.requested_extra_by_symbol:
            return {}
        overrides = self._extra_overrides_for_plan()
        cache_key = tuple(sorted((str(symbol), int(count)) for symbol, count in overrides.items()))
        if cache_key == self._quick_preview_cache_key:
            return dict(self._quick_preview_cache)
        try:
            counts = preview_character_train_synthetic_counts_from_pool(
                overrides,
                self._source_pool_candidates,
                self._source_pool_existing,
                respect_existing_augmented_variants=False,
            )
        except Exception:
            logger.exception("Nie udało się policzyć szybkiego preview syntetyków MZ")
            return {}
        self._quick_preview_cache_key = cache_key
        self._quick_preview_cache = dict(counts or {})
        return dict(self._quick_preview_cache)

    def _default_target_count(self) -> int:
        if self.result is None:
            return 0
        try:
            return max(0, int(dict(self.result.summary or {}).get("target_class_count", 0) or 0))
        except Exception:
            return 0

    def _common_target_count(self) -> int:
        raw = str(self.reference_count_var.get() or "").strip().replace(",", ".")
        if not raw:
            return self._default_target_count()
        try:
            return max(0, int(round(float(raw))))
        except Exception:
            return self._default_target_count()

    def _sync_common_target_from_result(self, result: CharacterClassDistribution) -> None:
        if str(self.reference_count_var.get() or "").strip():
            return
        try:
            count = max(0, int(dict(result.summary or {}).get("target_class_count", 0) or 0))
        except Exception:
            count = 0
        self.reference_count_var.set(str(count))

    def _target_for_symbol(self, symbol: str, row=None) -> int:
        symbol = str(symbol or "").strip().upper()
        current = self._diagnostic_count_for_row(row) if row is not None else 0
        extra = max(0, int(self.requested_extra_by_symbol.get(symbol, 0) or 0))
        return int(current) + extra

    def _extra_overrides_for_plan(self) -> dict[str, int]:
        if self.result is None:
            return {}
        rows = {str(row.symbol): row for row in self.result.classes}
        return {
            str(symbol): max(0, int(count or 0))
            for symbol, count in sorted(dict(self.requested_extra_by_symbol).items())
            if str(symbol) in rows and int(count or 0) > 0
        }

    def _build_current_balance_plan(self):
        return plan_character_train_augmentation(
            self.dataset_root,
            target_ratio=self._target_ratio(),
            target_count=0,
            extra_count_by_symbol=self._extra_overrides_for_plan(),
            respect_existing_augmented_variants=False,
        )

    def _schedule_synthetic_preview(self) -> None:
        if self.read_only or self.result is None or self.window is None:
            return
        if self._preview_after_id is not None:
            try:
                self.window.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        self._preview_serial += 1
        serial = self._preview_serial
        self.synthetic_total_var.set("Syntetyki train: liczę...")

        def run_preview() -> None:
            try:
                plan = self._build_current_balance_plan()
            except Exception as exc:
                self._after(lambda: self._apply_synthetic_preview_error(serial, str(exc)))
                return
            self._after(lambda: self._apply_synthetic_preview(serial, plan))

        def start_worker() -> None:
            threading.Thread(target=run_preview, daemon=True).start()

        self._preview_after_id = self.window.after(100, start_worker)

    def _apply_synthetic_preview(self, serial: int, plan) -> None:
        if serial != self._preview_serial:
            return
        self._preview_plan = plan
        planned = max(0, int(getattr(plan, "planned_images", 0) or 0))
        remaining = dict(getattr(plan, "predicted_deficit_after", {}) or {})
        suffix = ""
        if remaining:
            suffix = " | braki bez źródła: " + ", ".join(sorted(remaining.keys()))
        self.synthetic_total_var.set(f"Syntetyki train: +{planned}{suffix}")
        if self.result is not None:
            self._fill_summary(self.result)
            self._fill_table(self.result)
        self._draw_chart()

    def _apply_synthetic_preview_error(self, serial: int, error_text: str) -> None:
        if serial != self._preview_serial:
            return
        self._preview_plan = None
        self.synthetic_total_var.set("Syntetyki train: nie policzono")
        self.status_var.set(f"Nie udało się policzyć syntetyków: {error_text}")

    def _planned_synthetic_counts_by_symbol(self, plan=None) -> dict[str, int]:
        plan = plan if plan is not None else self._preview_plan
        result: dict[str, int] = {}
        if plan is None:
            return result
        explicit = dict(getattr(plan, "synthetic_count_by_symbol", {}) or {})
        if explicit:
            return {
                str(symbol): max(0, int(count or 0))
                for symbol, count in sorted(explicit.items())
                if int(count or 0) > 0
            }
        for candidate in getattr(plan, "candidates", ()) or ():
            try:
                variants = max(0, int(getattr(candidate, "planned_variants", 0) or 0))
                symbol_counts = dict(getattr(candidate, "symbol_counts", {}) or {})
            except Exception:
                if not isinstance(candidate, dict):
                    continue
                variants = max(0, int(candidate.get("planned_variants", 0) or 0))
                symbol_counts = dict(candidate.get("symbol_counts", {}) or {})
            if variants <= 0:
                continue
            for symbol, count in symbol_counts.items():
                amount = max(0, int(count or 0)) * variants
                if amount <= 0:
                    continue
                symbol_key = str(symbol or "").strip().upper()
                if not symbol_key:
                    continue
                result[symbol_key] = int(result.get(symbol_key, 0) or 0) + amount
        return result

    def _display_synthetic_counts_by_symbol(self) -> dict[str, int]:
        planned = self._planned_synthetic_counts_by_symbol()
        if planned:
            return planned
        quick = self._quick_synthetic_counts_by_symbol()
        if quick:
            return quick
        return {
            str(symbol): max(0, int(count or 0))
            for symbol, count in dict(self.requested_extra_by_symbol).items()
            if int(count or 0) > 0
        }

    def _set_reference_count(self, value: int) -> None:
        if self.result is None:
            return
        value = max(0, int(value or 0))
        if str(self.reference_count_var.get() or "") == str(value):
            return
        self.reference_count_var.set(str(value))
        self._fill_summary(self.result)
        self._fill_table(self.result)
        self._draw_chart()

    def _set_symbol_extra(self, symbol: str, value: int) -> None:
        if self.result is None:
            return
        symbol = str(symbol or "").strip().upper()
        row = self._row_for_symbol(symbol)
        if row is None:
            return
        if self._diagnostic_count_for_row(row) <= 0:
            self.requested_extra_by_symbol.pop(symbol, None)
            self._preview_plan = None
            self._quick_preview_cache_key = None
            self._quick_preview_cache = {}
            self._announce_zero_symbol_locked(symbol)
            self._fill_summary(self.result)
            self._fill_table(self.result)
            self._draw_chart()
            return
        extra = max(0, int(value or 0))
        current_extra = max(0, int(self.requested_extra_by_symbol.get(symbol, 0) or 0))
        if extra == current_extra:
            return
        if extra <= 0:
            self.requested_extra_by_symbol.pop(symbol, None)
        else:
            self.requested_extra_by_symbol[symbol] = extra
        self._preview_plan = None
        self._quick_preview_cache_key = None
        self._quick_preview_cache = {}
        self.status_var.set(f"Znak {symbol}: ustawiono +{extra}. Zatwierdź, aby przenieść liczbę syntetyków do PZ1.")
        self._fill_summary(self.result)
        self._fill_table(self.result)
        self._draw_chart()
        self._schedule_synthetic_preview()

    def _commit_chart_target_entry(self, symbol: str) -> None:
        return None

    def _commit_all_chart_target_entries(self) -> bool:
        return False

    def _clear_chart_target_entries(self) -> None:
        return None

    def _fill_summary(self, result: CharacterClassDistribution) -> None:
        summary = dict(result.summary or {})
        requested_total = sum(max(0, int(value or 0)) for value in dict(self.requested_extra_by_symbol).values())
        reference_count = self._common_target_count()
        range_text = (
            f"{int(summary.get('minimum_class_count', 0) or 0)} / "
            f"{_format_float(summary.get('median_class_count'))} / "
            f"{int(summary.get('maximum_class_count', 0) or 0)}"
        )
        ratio_value = summary.get("max_min_ratio")
        ratio_text = "nieokreślony przy brakach" if ratio_value is None else _format_float(ratio_value)
        file_errors = int(summary.get("invalid_label_lines", 0) or 0) + int(summary.get("invalid_class_ids", 0) or 0)
        class_map = dict(summary.get("class_map") or {})
        class_map_status = str(class_map.get("status") or "-")
        if class_map_status == "OK":
            class_map_text = "Zgodna z MZ"
        elif class_map_status == "WARNING":
            class_map_text = "Ostrzeżenie: brak data.yaml"
        else:
            class_map_text = class_map.get("message") or "Nie sprawdzono"
        values = {
            "layout": _layout_label(result.layout),
            "class_map": class_map_text,
            "diagnostic_split": "train" if result.diagnostic_split == "train" else "razem",
            "total_labels": str(int(summary.get("total_labels", 0) or 0)),
            "range": range_text,
            "target": f"miarka {reference_count}; zamówiono +{requested_total}",
            "ratio": ratio_text,
            "zero": _symbols(summary.get("zero_classes")),
            "low": _symbols(summary.get("low_count_classes")),
            "diversity": _symbols(summary.get("low_diversity_classes")),
            "files": f"{int(summary.get('files_read', 0) or 0)} plików, błędy: {file_errors}",
        }
        for key, value in values.items():
            label = self.summary_value_labels.get(key)
            if label is None:
                continue
            color = self.palette.get("fg", "#f3f3f3")
            if key == "zero" and value != "brak":
                color = _STATUS_COLORS["CRITICAL"]
            elif key == "target" and requested_total > 0:
                color = _STATUS_COLORS["LOW"]
            elif key == "class_map" and str(class_map_status) == "WARNING":
                color = _STATUS_COLORS["LOW"]
            elif key in {"low", "diversity"} and value != "brak":
                color = _STATUS_COLORS["LOW"]
            try:
                label.configure(text=value, fg=color)
            except Exception:
                label.configure(text=value)

    def _fill_table(self, result: CharacterClassDistribution) -> None:
        if self.table is None:
            return
        synthetic_by_symbol = self._display_synthetic_counts_by_symbol()
        self.table.delete(*self.table.get_children())
        for row in result.classes:
            diagnostic_count = self._diagnostic_count_for_row(row)
            synthetic_count = max(0, int(synthetic_by_symbol.get(str(row.symbol), 0) or 0))
            target_count = diagnostic_count + synthetic_count
            deficit_count = self._effective_deficit_for_row(row)
            status = row.status
            if synthetic_count > 0:
                status = "LOW"
            elif deficit_count > 0:
                status = "CRITICAL" if diagnostic_count <= 0 else "LOW"
            self.table.insert(
                "",
                tk.END,
                values=(
                    row.symbol,
                    row.train_count,
                    row.unique_train_plate_count,
                    _STATUS_LABELS.get(row.count_status, row.count_status),
                    _STATUS_LABELS.get(row.diversity_status, row.diversity_status),
                    target_count,
                    f"+{synthetic_count}" if synthetic_count > 0 else "0",
                    row.val_count,
                    row.test_count,
                    row.total_count,
                    _format_percent(row.train_share),
                    _STATUS_LABELS.get(status, status),
                ),
                tags=(_status_tag(status),),
            )

    def _draw_chart(self) -> None:
        if self.chart is None:
            return
        self._clear_chart_target_entries()
        self.chart.delete("all")
        self._chart_bar_slots = {}
        self._chart_reference_line_slot = None
        width = max(320, int(self.chart.winfo_width() or 760))
        height = max(160, int(self.chart.winfo_height() or 210))
        panel = self.palette.get("panel", "#25262b")
        fg = self.palette.get("fg", "#f3f3f3")
        muted = self.palette.get("muted", "#b7bcc6")
        grid = self._border()
        self.chart.configure(bg=panel)

        if self.result is None:
            self.chart.create_text(width / 2, height / 2, text="Liczenie klas...", fill=muted, font=("Segoe UI", 10))
            return

        result = self.result
        use_train = result.diagnostic_split == "train"
        values = [row.train_count if use_train else row.total_count for row in result.classes]
        targets = [self._effective_target_for_row(row) for row in result.classes]
        common_target = self._common_target_count()
        synthetic_by_symbol = self._display_synthetic_counts_by_symbol()
        totals_after = [
            int(value or 0) + int(synthetic_by_symbol.get(str(row.symbol), 0) or 0)
            for value, row in zip(values, result.classes)
        ]
        max_value = max([*values, *targets, *totals_after, common_target, 1]) if values or targets else 1
        max_value = max(1, int(max_value * 1.15) + 1)
        extra_label_values = [
            max(
                max(0, int(self.requested_extra_by_symbol.get(str(row.symbol), 0) or 0)),
                max(0, int(synthetic_by_symbol.get(str(row.symbol), 0) or 0)),
            )
            for row in result.classes
        ]
        max_extra_label_h = max((_chart_extra_label_height(value) for value in extra_label_values if value > 0), default=0)
        left = 68
        right = 14
        top = max(48, 34 + max_extra_label_h)
        bottom = 34
        plot_w = max(1, width - left - right)
        plot_h = max(1, height - top - bottom)
        base_y = top + plot_h
        self.chart.create_line(left, top, left, base_y, fill=grid)
        self.chart.create_line(left, base_y, width - right, base_y, fill=grid)
        title = "Liczba etykiet w train" if use_train else "Liczba etykiet razem"
        self.chart.create_text(left, 10, anchor=tk.W, text=title, fill=fg, font=("Segoe UI Semibold", 9))
        hint = "" if self.read_only else "kliknij słupek, aby wpisać +N; przeciągnij górny uchwyt; miarkę przesuń bocznym wskaźnikiem"
        self.chart.create_text(width - right, 10, anchor=tk.E, text=f"max: {max_value}  {hint}", fill=muted, font=("Segoe UI", 8))

        if max_value <= 0:
            self.chart.create_text(width / 2, height / 2, text="Brak etykiet do pokazania.", fill=muted, font=("Segoe UI", 10))
            return

        reference_color = self._chart_reference_color(panel)
        synthetic_color = self._chart_synthetic_color(panel)
        common_y = base_y - (float(common_target) / float(max_value)) * plot_h if common_target > 0 else base_y
        self.chart.create_line(left, common_y, width - right, common_y, fill=reference_color, width=2, dash=(7, 4))
        label = str(common_target)
        pointer_h = 20
        pointer_w = max(34, min(52, len(label) * 7 + 18))
        body_x1 = left - 8
        body_x0 = max(4, body_x1 - pointer_w)
        label_y0 = max(2, min(height - pointer_h - 2, common_y - pointer_h / 2))
        label_y1 = label_y0 + pointer_h
        self.chart.create_polygon(
            body_x0,
            label_y0,
            body_x1,
            label_y0,
            left,
            common_y,
            body_x1,
            label_y1,
            body_x0,
            label_y1,
            fill=self._chart_label_bg(panel),
            outline=reference_color,
        )
        self.chart.create_text(
            (body_x0 + body_x1) / 2,
            (label_y0 + label_y1) / 2,
            anchor=tk.CENTER,
            text=label,
            fill=reference_color,
            font=("Segoe UI Semibold", 8),
        )
        self._chart_reference_line_slot = (
            float(body_x0),
            float(left),
            float(label_y0),
            float(label_y1),
            float(plot_h),
            float(max_value),
            float(common_y),
        )

        slot = plot_w / max(1, len(values))
        bar_w = max(4, slot * 0.64)
        for index, row in enumerate(result.classes):
            value = int(values[index] or 0)
            target = self._effective_target_for_row(row)
            synthetic_count = int(synthetic_by_symbol.get(str(row.symbol), 0) or 0)
            total_after = value + synthetic_count
            bar_h = (float(value) / float(max_value)) * plot_h if max_value else 0
            synthetic_h = (float(synthetic_count) / float(max_value)) * plot_h if max_value else 0
            total_h = (float(total_after) / float(max_value)) * plot_h if max_value else 0
            target_h = (float(target) / float(max_value)) * plot_h if max_value else 0
            x_mid = left + slot * index + slot / 2
            x0 = x_mid - bar_w / 2
            x1 = x_mid + bar_w / 2
            y0 = base_y - bar_h
            synthetic_y0 = base_y - total_h
            target_y = base_y - target_h
            color = _STATUS_COLORS.get(row.status, _STATUS_COLORS["OK"])
            if target > total_after:
                self.chart.create_rectangle(x0, target_y, x1, base_y, outline=grid, dash=(2, 3))
            self.chart.create_rectangle(x0, y0, x1, base_y, fill=color, outline="")
            if synthetic_count > 0:
                self.chart.create_rectangle(x0, synthetic_y0, x1, y0, fill=synthetic_color, outline="")
            if bar_h >= 15:
                self.chart.create_text(
                    x_mid,
                    base_y - min(max(9, bar_h / 2), bar_h - 6),
                    text=str(value),
                    fill=self._chart_dark_text_color(panel),
                    font=("Segoe UI Semibold", 7),
                )
            else:
                self.chart.create_text(
                    x_mid,
                    max(top + 10, y0 - 8),
                    text=str(value),
                    fill=fg,
                    font=("Segoe UI Semibold", 7),
                )
            requested_extra = max(0, int(self.requested_extra_by_symbol.get(str(row.symbol), 0) or 0))
            handle_y = synthetic_y0 if synthetic_count > 0 else y0
            if requested_extra > 0 or synthetic_count > 0:
                label_value = max(requested_extra, synthetic_count)
                label_text = "+\n" + "\n".join(str(label_value))
                label_h = _chart_extra_label_height(label_value)
                label_bottom = min(top - 3, top - (4 if index % 2 == 0 else 12))
                label_top = max(20, label_bottom - label_h)
                label_y = label_top + label_h / 2
                label_w = max(18, min(28, slot * 0.72))
                self.chart.create_line(
                    x_mid,
                    handle_y,
                    x_mid,
                    label_top + label_h,
                    fill=synthetic_color,
                    width=1,
                    dash=(2, 3),
                )
                self.chart.create_rectangle(
                    x_mid - label_w / 2,
                    label_y - label_h / 2,
                    x_mid + label_w / 2,
                    label_y + label_h / 2,
                    fill=self._chart_label_bg(panel),
                    outline=synthetic_color,
                )
                self.chart.create_text(
                    x_mid,
                    label_y,
                    text=label_text,
                    fill=self._chart_label_fg(panel),
                    font=("Segoe UI Semibold", 7),
                    justify=tk.CENTER,
                )
            if value > 0:
                handle_color = synthetic_color if (requested_extra > 0 or synthetic_count > 0) else reference_color
                handle_pad = max(3, min(7, slot * 0.12))
                self.chart.create_rectangle(
                    x0 - handle_pad,
                    handle_y - 3,
                    x1 + handle_pad,
                    handle_y + 3,
                    fill=handle_color,
                    outline=self._chart_label_bg(panel),
                    width=1,
                )
            else:
                self.chart.create_text(
                    x_mid,
                    base_y - 8,
                    text="×",
                    fill=muted,
                    font=("Segoe UI Semibold", 8),
                )
            self.chart.create_text(x_mid, base_y + 12, text=row.symbol, fill=muted, font=("Segoe UI", 7))
            self._chart_bar_slots[row.symbol] = (x_mid - slot / 2, x_mid + slot / 2, top, base_y, plot_h, float(max_value), float(value))
        self.chart.create_text(8, top, anchor=tk.W, text=str(max_value), fill=muted, font=("Segoe UI", 7))
        self.chart.create_text(8, base_y, anchor=tk.W, text="0", fill=muted, font=("Segoe UI", 7))

    def _diagnostic_count_for_row(self, row) -> int:
        if self.result is not None and str(self.result.diagnostic_split or "") == "train":
            return int(getattr(row, "train_count", 0) or 0)
        return int(getattr(row, "total_count", 0) or 0)

    def _effective_target_for_row(self, row) -> int:
        return self._target_for_symbol(str(getattr(row, "symbol", "") or ""), row)

    def _effective_deficit_for_row(self, row) -> int:
        return max(0, self._effective_target_for_row(row) - self._diagnostic_count_for_row(row))

    def _row_for_symbol(self, symbol: str):
        if self.result is None:
            return None
        symbol = str(symbol or "").strip().upper()
        return next((item for item in self.result.classes if str(item.symbol) == symbol), None)

    def _can_edit_symbol_extra(self, symbol: str) -> bool:
        row = self._row_for_symbol(symbol)
        return row is not None and self._diagnostic_count_for_row(row) > 0

    def _announce_zero_symbol_locked(self, symbol: str) -> None:
        symbol = str(symbol or "").strip().upper()
        self.status_var.set(
            f"Znak {symbol}: 0 przykładów w train. Nie da się go zwiększyć syntetycznie bez realnej tablicy źródłowej."
        )

    def _on_chart_mouse_down(self, event):
        if self.read_only or self.result is None:
            return
        self._chart_press = {
            "kind": "",
            "symbol": "",
            "x": int(getattr(event, "x", 0) or 0),
            "y": int(getattr(event, "y", 0) or 0),
            "moved": False,
        }
        ref_slot = self._chart_reference_line_slot
        if ref_slot is not None:
            x0, x1, y0, y1, _plot_h, _max_value, _line_y = ref_slot
            if float(x0) <= float(event.x) <= float(x1) and float(y0) - 4 <= float(event.y) <= float(y1) + 4:
                self._chart_press["kind"] = "reference"
                self.chart.configure(cursor="sb_v_double_arrow")
                return "break"
        selected_symbol = ""
        selected_current = 0.0
        for symbol, slot in self._chart_bar_slots.items():
            x0, x1, _top, _base_y, _plot_h, _max_value, _current = slot
            if float(x0) <= float(event.x) <= float(x1):
                selected_symbol = symbol
                selected_current = float(_current)
                break
        if not selected_symbol:
            self._chart_press = None
            return
        if selected_current <= 0 or not self._can_edit_symbol_extra(selected_symbol):
            self._chart_press = None
            self._announce_zero_symbol_locked(selected_symbol)
            return "break"
        self._chart_press["kind"] = "bar"
        self._chart_press["symbol"] = selected_symbol
        self.chart.configure(cursor="sb_v_double_arrow")
        return "break"

    def _on_chart_mouse_drag(self, event):
        if self.read_only or self.result is None or self._chart_press is None:
            return
        start_x = int(self._chart_press.get("x", 0) or 0)
        start_y = int(self._chart_press.get("y", 0) or 0)
        if abs(int(event.x) - start_x) + abs(int(event.y) - start_y) > 3:
            self._chart_press["moved"] = True
        kind = str(self._chart_press.get("kind") or "")
        if kind == "reference":
            ref_slot = self._chart_reference_line_slot
            if ref_slot is None:
                return "break"
            _x0, _x1, _y0, _y1, plot_h, max_value, _line_y = ref_slot
            base_y = self._chart_plot_base_y()
            top = max(1.0, base_y - float(plot_h))
            y = max(float(top), min(float(base_y), float(event.y)))
            value = int(round(((float(base_y) - y) / max(1.0, float(plot_h))) * float(max_value)))
            self._set_reference_count(value)
            return "break"
        if kind == "bar":
            symbol = str(self._chart_press.get("symbol") or "")
            slot = self._chart_bar_slots.get(symbol)
            if slot is None:
                return "break"
            _x0, _x1, top, base_y, plot_h, max_value, current_count = slot
            if float(current_count) <= 0 or not self._can_edit_symbol_extra(symbol):
                self._announce_zero_symbol_locked(symbol)
                return "break"
            y = max(float(top), min(float(base_y), float(event.y)))
            target_total = int(round(((float(base_y) - y) / max(1.0, float(plot_h))) * float(max_value)))
            extra = max(0, int(target_total) - int(round(float(current_count))))
            self._set_symbol_extra(symbol, extra)
            return "break"

    def _on_chart_mouse_up(self, event):
        if self.chart is not None:
            try:
                self.chart.configure(cursor="")
            except Exception:
                pass
        press = self._chart_press
        self._chart_press = None
        if self.read_only or self.result is None or not press:
            return
        if str(press.get("kind") or "") == "bar" and not bool(press.get("moved")):
            symbol = str(press.get("symbol") or "")
            if symbol:
                self._open_symbol_extra_editor(symbol, event)
                return "break"
        return "break"

    def _chart_plot_base_y(self) -> float:
        if self.chart is None:
            return 1.0
        height = max(160, int(self.chart.winfo_height() or 210))
        return float(48 + max(1, height - 48 - 34))

    def _open_symbol_extra_editor(self, symbol: str, event=None) -> None:
        if self.window is None or self.result is None:
            return
        symbol = str(symbol or "").strip().upper()
        row = self._row_for_symbol(symbol)
        if row is None:
            return
        if self._diagnostic_count_for_row(row) <= 0:
            self._announce_zero_symbol_locked(symbol)
            return
        try:
            if self._symbol_editor is not None and self._symbol_editor.winfo_exists():
                self._symbol_editor.destroy()
        except Exception:
            pass
        bg = self.palette.get("panel", "#25262b")
        panel_alt = self.palette.get("panel_alt", "#2c2d33")
        fg = self.palette.get("fg", "#f3f3f3")
        muted = self.palette.get("muted", "#b7bcc6")
        accent = self._chart_synthetic_color(bg)
        editor = tk.Toplevel(self.window)
        self._symbol_editor = editor
        editor.title(f"Znak {symbol}: syntetyki")
        editor.configure(bg=bg)
        try:
            editor.transient(self.window)
            editor.resizable(False, False)
        except Exception:
            pass
        root = tk.Frame(editor, bg=bg, padx=12, pady=10)
        root.pack(fill=tk.BOTH, expand=True)
        current = self._diagnostic_count_for_row(row)
        value_var = tk.StringVar(value=str(max(0, int(self.requested_extra_by_symbol.get(symbol, 0) or 0))))
        tk.Label(
            root,
            text=f"Znak {symbol}",
            bg=bg,
            fg=fg,
            font=("Segoe UI Semibold", 11),
            anchor=tk.W,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(
            root,
            text=f"W train jest teraz: {current}. Wpisz, ile znaków dodać syntetycznie.",
            bg=bg,
            fg=muted,
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            wraplength=260,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        tk.Label(root, text="+", bg=bg, fg=accent, font=("Segoe UI Semibold", 14)).grid(row=2, column=0, sticky="e")
        entry = tk.Entry(
            root,
            textvariable=value_var,
            width=8,
            justify=tk.CENTER,
            bg=panel_alt,
            fg=fg,
            insertbackground=accent,
            highlightthickness=1,
            highlightbackground=accent,
            highlightcolor=accent,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 11),
        )
        entry.grid(row=2, column=1, sticky="w", padx=(6, 0))

        def accept(_event=None):
            raw = str(value_var.get() or "").strip().replace(",", ".")
            try:
                value = int(round(float(raw)))
            except Exception:
                value = 0
            self._set_symbol_extra(symbol, max(0, value))
            try:
                editor.destroy()
            except Exception:
                pass
            return "break"

        def close(_event=None):
            try:
                editor.destroy()
            except Exception:
                pass
            return "break"

        entry.bind("<Return>", accept)
        entry.bind("<KP_Enter>", accept)
        editor.bind("<Escape>", close)
        ttk.Button(root, text="Zatwierdź", command=accept).grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        try:
            editor.update_idletasks()
            if event is not None and self.chart is not None:
                x = int(self.chart.winfo_rootx()) + int(getattr(event, "x", 0) or 0) + 12
                y = int(self.chart.winfo_rooty()) + int(getattr(event, "y", 0) or 0) + 12
            else:
                x = int(self.window.winfo_rootx()) + 120
                y = int(self.window.winfo_rooty()) + 120
            editor.geometry(f"+{max(0, x)}+{max(0, y)}")
            entry.focus_set()
            entry.selection_range(0, tk.END)
            entry.icursor(tk.END)
        except Exception:
            pass

    def _save_before_artifacts(self) -> None:
        if self.result is None or self.window is None:
            return
        try:
            refs = save_character_distribution_artifacts(
                self.result,
                self.dataset_root / "analysis",
                prefix="character_class_distribution_before",
            )
        except Exception as exc:
            messagebox.showerror("Błąd zapisu raportu przed uzupełnieniem", str(exc), parent=self.window)
            return
        json_ref = refs.get("json", {})
        self.status_var.set(f"Zapisano raport przed uzupełnieniem: {json_ref.get('path', '')}")

    def _accept_current_settings(self) -> None:
        if self.window is None:
            return
        if self.read_only:
            self.status_var.set("Ten widok jest raportem. Nie zmieniam ustawień PZ1.")
            return
        if self.result is None:
            self.status_var.set("Najpierw poczekaj na zakończenie analizy.")
            return
        if self._commit_all_chart_target_entries():
            self._fill_summary(self.result)
            self._fill_table(self.result)
            self._draw_chart()
        self.status_var.set("Liczenie syntetyków dla ustawionego rozkładu...")
        if self.plan_btn is not None:
            self.plan_btn.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                plan = self._build_current_balance_plan()
            except Exception as exc:
                self._after(lambda: self._show_plan_error(str(exc)))
                return
            self._after(lambda: self._accept_balance_plan(self.window, plan))

        threading.Thread(target=worker, daemon=True).start()

    def _show_balance_plan(self) -> None:
        self._accept_current_settings()

    def _show_plan_error(self, error_text: str) -> None:
        if self.plan_btn is not None:
            self.plan_btn.configure(state=(tk.NORMAL if self.result is not None and not self.read_only else tk.DISABLED))
        self.status_var.set("Nie udało się policzyć syntetyków.")
        if self.window is not None and self.window.winfo_exists():
            messagebox.showerror("Błąd uzupełnienia", error_text, parent=self.window)

    def _open_plan_dialog(self, plan) -> None:
        if self.plan_btn is not None:
            self.plan_btn.configure(state=tk.NORMAL if self.result is not None else tk.DISABLED)
        if self.window is None or not self.window.winfo_exists():
            return
        try:
            preliminary_unsplit = str(getattr(self.result, "layout", "") or "").strip().lower() != "split"
        except Exception:
            preliminary_unsplit = False
        self.status_var.set(
            "Prognoza uzupełnienia niedoreprezentowanych znaków jest gotowa."
            if preliminary_unsplit
            else "Propozycja uzupełnienia niedoreprezentowanych znaków jest gotowa."
        )
        bg = self.palette.get("bg", "#1e1f22")
        fg = self.palette.get("fg", "#f3f3f3")
        muted = self.palette.get("muted", "#b7bcc6")
        panel = self.palette.get("panel", "#25262b")
        dialog = tk.Toplevel(self.window)
        dialog.title("Prognoza uzupełnienia znaków MZ" if preliminary_unsplit else "Uzupełnianie znaków MZ")
        dialog.minsize(820, 520)
        dialog.geometry("940x600")
        dialog.configure(bg=bg)
        try:
            dialog.transient(self.window)
        except Exception:
            pass
        root = tk.Frame(dialog, bg=bg, padx=14, pady=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)
        tk.Label(
            root,
            text=("Prognoza uzupełnienia znaków MZ" if preliminary_unsplit else "Propozycja uzupełnienia znaków MZ"),
            bg=bg,
            fg=fg,
            font=("Segoe UI Semibold", 13),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        total_deficit = sum(int(value or 0) for value in dict(plan.deficit_by_symbol or {}).values())
        deficient_symbols = list(dict(plan.deficit_by_symbol or {}).keys())
        planned_images = int(getattr(plan, "planned_images", 0) or 0)
        train_sources_by_symbol = dict(getattr(plan, "train_sources_by_symbol", {}) or {})
        missing_source_symbols = [
            symbol for symbol in deficient_symbols
            if int(train_sources_by_symbol.get(symbol, 0) or 0) <= 0
        ]
        plan_label = "Prognoza przed splitem" if preliminary_unsplit else "Syntetyczne uzupełnienie"
        summary = (
            f"Próg: {int(getattr(plan, 'target_count', 0) or 0)} przykładów znaku w train  |  "
            f"Do uzupełnienia: {', '.join(deficient_symbols) if deficient_symbols else 'brak'}  |  "
            f"Brakuje znaków łącznie: {total_deficit}  |  "
            f"{plan_label}: +{planned_images} syntetycznych obrazów train  |  "
            f"Użyte tablice: {int(getattr(plan, 'unique_real_sources_used', 0) or 0)}, "
            f"maks. z jednej: {int(getattr(plan, 'max_augmented_variants_from_single_source', 0) or 0)}"
        )
        if missing_source_symbols:
            summary += "  |  Bez źródła w train: " + ", ".join(missing_source_symbols)
        tk.Label(
            root,
            text=summary,
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=880,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 10))
        tk.Label(
            root,
            text=(
                "PZ1 uzupełnia znaki wyłącznie syntetycznie, na bazie tablic istniejących w train. "
                "Jeśli znaku nie ma w train, trzeba dodać lub oznaczyć realną tablicę w kolejnej iteracji. "
                "Wzrost liczby innych znaków przy okazji nie jest błędem."
            ),
            bg=panel,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=880,
            padx=10,
            pady=8,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        table = ttk.Treeview(
            root,
            columns=("symbol", "train", "sources", "target", "missing_before", "missing_after", "status"),
            show="headings",
            height=16,
        )
        headings = {
            "symbol": "Znak",
            "train": "Train",
            "sources": "Tablice w train",
            "target": "Próg",
            "missing_before": "Brakuje teraz",
            "missing_after": "Po uzupełnieniu",
            "status": "Prognoza",
        }
        widths = {
            "symbol": 70,
            "train": 90,
            "sources": 120,
            "target": 100,
            "missing_before": 120,
            "missing_after": 110,
            "status": 210,
        }
        for column in headings:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], minwidth=60, stretch=column == "status")
        yscroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscrollcommand=yscroll.set)
        table.grid(row=3, column=0, sticky="nsew")
        yscroll.grid(row=3, column=1, sticky="ns")
        result_rows = {str(row.symbol): row for row in (getattr(self.result, "classes", []) or [])}
        predicted_after = dict(getattr(plan, "predicted_deficit_after", {}) or {})
        target_by_symbol = dict(getattr(plan, "target_count_by_symbol", {}) or {})
        default_target_count = int(getattr(plan, "target_count", 0) or 0)
        for symbol in deficient_symbols:
            row = result_rows.get(str(symbol))
            before = int(dict(plan.deficit_by_symbol or {}).get(symbol, 0) or 0)
            after = int(predicted_after.get(symbol, 0) or 0)
            source_count = int(train_sources_by_symbol.get(symbol, 0) or 0)
            if source_count <= 0:
                source_text = "brak"
                status = "Brak źródła w train"
            else:
                source_text = f"{source_count} tablic" if source_count != 1 else "1 tablica"
                status = "OK po uzupełnieniu" if after <= 0 else f"Zostanie brak: {after}"
            table.insert(
                "",
                tk.END,
                values=(
                    symbol,
                    int(getattr(row, "train_count", 0) or 0) if row is not None else "-",
                    source_text,
                    int(target_by_symbol.get(symbol, default_target_count) or default_target_count),
                    before,
                    max(0, after),
                    status,
                ),
            )
        footer = tk.Frame(root, bg=bg)
        footer.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(
            footer,
            text=(
                "To jest prognoza przed finalnym splitem. Ostateczna propozycja zostanie przeliczona przy tworzeniu wariantu."
                if preliminary_unsplit
                else (
                    "Ten krok syntetycznie powiększa tylko train. Val i test pozostają bez zmian. "
                    "Jeśli przy okazji wzrosną także inne znaki, to jest akceptowalne."
                )
            ),
            bg=bg,
            fg=muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        plan_feasible = bool(getattr(plan, "feasible", True))
        can_accept = plan_feasible or planned_images > 0
        ttk.Button(
            footer,
            text=(
                ("Ustaw prognozę syntetyków w PZ1" if preliminary_unsplit else "Ustaw uzupełnienie w PZ1")
                if plan_feasible
                else (
                    "Ustaw częściowe uzupełnienie w PZ1"
                    if planned_images > 0
                    else ("Przelicz po utworzeniu splitu" if preliminary_unsplit else "Brak wykonalnego uzupełnienia")
                )
            ),
            command=lambda: self._accept_balance_plan(dialog, plan),
            state=(tk.NORMAL if can_accept else tk.DISABLED),
        ).grid(row=0, column=1, sticky="e", padx=(8, 6))
        ttk.Button(footer, text="Zamknij", command=dialog.destroy).grid(row=0, column=2, sticky="e")
        try:
            self.window.update_idletasks()
            dialog.update_idletasks()
            width = int(dialog.winfo_width() or 940)
            height = int(dialog.winfo_height() or 600)
            x = int(self.window.winfo_rootx()) + max(20, (int(self.window.winfo_width()) - width) // 2)
            y = int(self.window.winfo_rooty()) + max(20, (int(self.window.winfo_height()) - height) // 2)
            dialog.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _accept_balance_plan(self, dialog: tk.Toplevel, plan) -> None:
        if self.read_only:
            self.status_var.set("Ten widok jest raportem. Nie zmieniam ustawień PZ1.")
            return
        planned_images = max(0, int(getattr(plan, "planned_images", 0) or 0))
        try:
            setattr(self.host, "_pending_character_balance_plan", plan)
            try:
                train = float(getattr(self.host, "train_pct").get())
                val = float(getattr(self.host, "val_pct").get())
                test = max(5.0, 100.0 - train - val)
                setattr(self.host, "_pending_character_balance_ratio_key", (round(train, 4), round(val, 4), round(test, 4)))
            except Exception:
                pass
            enabled_var = getattr(self.host, "split_aug_enabled_var", None)
            extra_var = getattr(self.host, "split_aug_extra_var", None)
            sample_var = getattr(self.host, "split_aug_sample_var", None)
            if enabled_var is not None:
                enabled_var.set(planned_images > 0)
            if extra_var is not None:
                extra_var.set(planned_images)
            if sample_var is not None:
                sample_var.set(max(1, planned_images))
            get_profile = getattr(self.host, "_get_step4_augmentation_profile", None)
            set_profile = getattr(self.host, "_set_step4_augmentation_profile", None)
            if callable(set_profile):
                try:
                    base_profile = get_profile("char") if callable(get_profile) else None
                    if base_profile is not None:
                        set_profile(
                            "char",
                            replace(
                                base_profile,
                                enabled=planned_images > 0,
                                extra_count=planned_images,
                                sample_size=max(1, planned_images),
                                task_target="char",
                            ).normalized(),
                        )
                except Exception:
                    logger.exception("Nie udało się zapisać profilu syntetyków znaków MZ w PZ1")
            status_var = getattr(self.host, "split_mz_representation_status_var", None)
            if status_var is not None:
                deficits = dict(getattr(plan, "deficit_by_symbol", {}) or {})
                requested_extras = dict(getattr(plan, "requested_extra_by_symbol", {}) or {})
                remaining = dict(getattr(plan, "predicted_deficit_after", {}) or {})
                requested_text = (
                    " Zamówione dodatki: "
                    + ", ".join(f"{key}+{value}" for key, value in sorted(requested_extras.items()))
                    + "."
                    if requested_extras
                    else ""
                )
                remaining_text = (
                    " Pozostały braki bez źródła w train: " + ", ".join(sorted(remaining.keys())) + "."
                    if remaining
                    else ""
                )
                status_var.set(
                    f"Miarka referencyjna: {self._common_target_count()} przykładów znaku w train. "
                    f"Znaki do syntetycznego podbicia: {', '.join(deficits.keys()) if deficits else 'brak'}. "
                    f"Ustawiono +{planned_images} syntetycznych obrazów train. "
                    "Przy tworzeniu wariantu liczba zostanie sprawdzona na aktualnym splicie."
                    f"{requested_text}"
                    f"{remaining_text}"
                )
            refresher = getattr(self.host, "_refresh_step4_augmentation_summary", None)
            if callable(refresher):
                refresher("char")
            logger.info(
                "[MZ BALANCE] accepted synthetic_count=%s target=%s overrides=%s remaining=%s",
                planned_images,
                int(getattr(plan, "target_count", 0) or 0),
                dict(getattr(plan, "requested_extra_by_symbol", {}) or {}),
                dict(getattr(plan, "predicted_deficit_after", {}) or {}),
            )
        except Exception as exc:
            logger.exception("Nie udało się przenieść ustawień doreprezentowania MZ do PZ1")
            if self.plan_btn is not None:
                self.plan_btn.configure(state=tk.NORMAL)
            self.status_var.set("Nie udało się ustawić syntetyków w PZ1.")
            if self.window is not None and self.window.winfo_exists():
                messagebox.showerror("Błąd ustawienia syntetyków", str(exc), parent=self.window)
            return
        self.status_var.set("Ustawiono syntetyki znaków w PZ1.")
        try:
            dialog.destroy()
        except Exception:
            pass

    def _export_csv(self) -> None:
        if self.result is None or self.window is None:
            return
        selected = filedialog.asksaveasfilename(
            parent=self.window,
            title="Zapisz raport reprezentacji znaków jako CSV",
            initialdir=str(self.dataset_root),
            initialfile="mz_class_distribution.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
        )
        if not selected:
            return
        try:
            output = save_character_class_distribution_csv(self.result, selected)
        except Exception as exc:
            messagebox.showerror("Błąd zapisu CSV", str(exc), parent=self.window)
            return
        self.status_var.set(f"Zapisano raport CSV: {output}")

    def _export_json(self) -> None:
        if self.result is None or self.window is None:
            return
        selected = filedialog.asksaveasfilename(
            parent=self.window,
            title="Zapisz raport reprezentacji znaków jako JSON",
            initialdir=str(self.dataset_root),
            initialfile="mz_class_distribution.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Wszystkie pliki", "*.*")],
        )
        if not selected:
            return
        try:
            output = save_character_class_distribution_json(self.result, selected)
        except Exception as exc:
            messagebox.showerror("Błąd zapisu JSON", str(exc), parent=self.window)
            return
        self.status_var.set(f"Zapisano raport JSON: {output}")

    def _export_distribution(self) -> None:
        if self.result is None or self.window is None:
            return
        if self._commit_all_chart_target_entries():
            self._fill_summary(self.result)
            self._fill_table(self.result)
            self._draw_chart()
        selected = filedialog.asksaveasfilename(
            parent=self.window,
            title="Eksport rozkładu znaków MZ",
            initialdir=str(self.dataset_root),
            initialfile="mz_rozklad_znakow.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Wszystkie pliki", "*.*")],
        )
        if not selected:
            return
        try:
            plan = self._preview_plan or self._build_current_balance_plan()
            payload = {
                "schema": "alpr.mz.character_distribution_export.v2",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_root": str(self.dataset_root),
                "dataset_yaml": str(self.yaml_path or ""),
                "reference_count": self._common_target_count(),
                "requested_extra_by_symbol": {
                    str(symbol): int(count)
                    for symbol, count in sorted(dict(self.requested_extra_by_symbol).items())
                },
                "planned_synthetic_train_images": max(0, int(getattr(plan, "planned_images", 0) or 0)),
                "remaining_deficit_by_symbol": dict(getattr(plan, "predicted_deficit_after", {}) or {}),
                "distribution": self.result.to_dict(),
                "augmentation_selection": plan.to_dict() if hasattr(plan, "to_dict") else {},
            }
            output = Path(selected)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Błąd eksportu rozkładu", str(exc), parent=self.window)
            return
        self.status_var.set(f"Zapisano rozkład: {output}")

    def _dataset_label(self) -> str:
        try:
            counts = self.host._get_dataset_split_image_counts(self.dataset_root)
        except Exception:
            counts = {}
        try:
            ref = build_dataset_display_ref(self.dataset_root, target_hint="char", counts=counts)
            return ref.detail_label
        except Exception:
            return str(self.dataset_root)

    def _center(self) -> None:
        if self.window is None:
            return
        center = getattr(self.app, "_center_dialog_window", None)
        if callable(center):
            try:
                center(self.window, parent=getattr(self.app, "root", None), width=1160, height=740)
                return
            except Exception:
                pass
        try:
            self.window.update_idletasks()
            width = int(self.window.winfo_width() or 1160)
            height = int(self.window.winfo_height() or 740)
            screen_w = int(self.window.winfo_screenwidth())
            screen_h = int(self.window.winfo_screenheight())
            x = max(0, (screen_w - width) // 2)
            y = max(0, (screen_h - height) // 2)
            self.window.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _border(self) -> str:
        return self.palette.get("panel_border", self.palette.get("border", "#3b3d46"))

    def _summary_accent_fg(self, background: str | None = None) -> str:
        return "#245b1f" if _is_light_hex(background or self.palette.get("panel_alt", "#162113")) else "#c9f27a"

    def _chart_reference_color(self, background: str | None = None) -> str:
        return "#355f24" if _is_light_hex(background or self.palette.get("panel", "#25262b")) else "#b7ff6a"

    def _chart_synthetic_color(self, background: str | None = None) -> str:
        return "#b45f00" if _is_light_hex(background or self.palette.get("panel", "#25262b")) else "#ffd166"

    def _chart_label_bg(self, background: str | None = None) -> str:
        return "#f6ffe9" if _is_light_hex(background or self.palette.get("panel", "#25262b")) else "#071b12"

    def _chart_label_fg(self, background: str | None = None) -> str:
        return "#102010" if _is_light_hex(background or self.palette.get("panel", "#25262b")) else "#f6ffe5"

    def _chart_dark_text_color(self, background: str | None = None) -> str:
        return "#071009" if not _is_light_hex(background or self.palette.get("panel", "#25262b")) else "#f8fff0"

    def _after(self, callback) -> None:
        window = self.window
        if window is None:
            return
        try:
            if window.winfo_exists():
                window.after(0, callback)
        except Exception:
            pass


def _status_tag(status: str) -> str:
    return "class_balance_" + str(status or "OK").lower().replace("+", "_").replace("-", "_")


def _is_light_hex(value: str | None) -> bool:
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return False
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except Exception:
        return False
    return ((0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0) > 0.58


def _chart_extra_label_height(value: int) -> int:
    digits = len(str(max(0, int(value or 0))))
    return max(34, 18 + 11 * (max(1, digits) + 1))


def _format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def _format_float(value) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if abs(numeric - round(numeric)) < 0.0001:
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


def _symbols(values) -> str:
    if not values:
        return "brak"
    try:
        items = [str(item) for item in values if str(item)]
    except Exception:
        items = []
    return ", ".join(items) if items else "brak"


def _layout_label(layout: str) -> str:
    if layout == "split":
        return "train / val / test"
    if layout == "flat":
        return "niesplitowany: liczymy razem"
    return "brak etykiet"
