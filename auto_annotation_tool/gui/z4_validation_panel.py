#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 validation panel helpers extracted from tab_training.py."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import datetime
import time
import webbrowser
import csv
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import (
    CONFIG,
    YOLO_AVAILABLE,
    AVAILABLE_POSE_MODELS,
    AVAILABLE_DETECT_MODELS,
    PIL_AVAILABLE,
    get_torch_module,
    get_yolo_class,
    is_cuda_available,
    logger,
)
from ..icons import IconManager
from ..validators import validate_model_file, format_yolo_model_identity
from ..training import YOLOPoseTrainer, TrainingHistory, TrainingStatus, DatasetCreator, DatasetSplitter
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_train_progress_bar import TrainProgressBar
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    finish_campaign_step4,
    get_campaign_training_target,
    open_campaign_step4_entry,
    poll_training_completion,
    restore_step4_campaign_project_state,
    set_campaign_context,
    set_campaign_training_target,
)
from .z4_flow_models import (
    CharYoloDatasetSourceAdapter,
    PlateXmlImagesSourceAdapter,
    TrainingSource,
    TrainingSourceStats,
)
from .z4_free_mode_flow import (
    refresh_free_training_route_cards,
    refresh_free_training_route_ui,
    update_step4_notebook_mode,
)
from .z4_shared_ui import (
    accept_training_input_context,
    clear_step4_guidance,
    guide_step4_builder_action,
    guide_step4_finish_action,
    guide_step4_next_action,
    guide_step4_route_selection,
    mark_step4_dataset_ready,
    open_step4_dataset_stage,
    refresh_step4_campaign_builder_inputs_ui,
    refresh_step4_analysis_tab_visibility,
    refresh_step4_campaign_navigation_ui,
    refresh_step4_dataset_mode_ui,
    refresh_step4_training_inputs_mode_ui,
    set_step4_dataset_mode,
    sync_step4_analysis_nav_buttons,
    step4_dataset_go_back,
    step4_dataset_go_next,
    step4_train_go_back,
)
from . import z4_dataset_sources
from . import z4_training_metrics
from . import z4_dataset_builder
from . import z4_analysis_ranking
from . import z4_training_runtime
from . import z4_campaign_state
from . import z4_dataset_validation
from . import z4_layout_runtime
from . import z4_device_runtime
from . import z4_model_export
from . import z4_training_progress
from . import z4_ui_runtime
from . import z4_theme_runtime
from . import z4_tab_shell
from . import z4_history_runtime
from . import z4_dataset_panels
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _format_validation_metric_name(raw_name: str) -> str:
    raw = str(raw_name or "").strip()
    mapping = {
        "metrics/precision(B)": "Precision (boxy)",
        "metrics/recall(B)": "Recall (boxy)",
        "metrics/mAP50(B)": "mAP50 (boxy)",
        "metrics/mAP50-95(B)": "mAP50-95 (boxy)",
        "metrics/precision(P)": "Precision (punkty)",
        "metrics/recall(P)": "Recall (punkty)",
        "metrics/mAP50(P)": "mAP50 (punkty)",
        "metrics/mAP50-95(P)": "mAP50-95 (punkty)",
        "fitness": "Fitness",
    }
    if raw in mapping:
        return mapping[raw]

    pretty = raw.replace("metrics/", "").replace("(B)", " (boxy)").replace("(P)", " (punkty)")
    pretty = pretty.replace("_", " ")
    return pretty or "Metryka"

def _format_validation_metric_value(value) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)

def _validation_best_metric_summary(rows: list[tuple[str, str, str, str]] | None) -> str:
    rows = list(rows or [])
    if not rows:
        return "Brak metryk do pokazania."
    preferred = ("mAP50-95", "mAP50", "Precision", "Recall", "Fitness")
    for needle in preferred:
        for row in rows:
            name = str(row[0] if len(row) > 0 else "")
            value = str(row[1] if len(row) > 1 else "")
            band = str(row[2] if len(row) > 2 else "")
            if needle.lower() in name.lower():
                suffix = f" | {band}" if band and band != "-" else ""
                return f"{name}: {value}{suffix}"
    row = rows[0]
    name = str(row[0] if len(row) > 0 else "Metryka")
    value = str(row[1] if len(row) > 1 else "-")
    return f"{name}: {value}"

def _build_validation_panel_v2(self, parent):
    shell = ttk.Frame(parent, padding=10, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        shell,
        text=(
            "Walidacja sprawdza wybrany model na wskazanym torze testowym. "
            "Karta pokazuje krótki werdykt, a pełną tabelę metryk otwierasz na żądanie."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 12))

    form_box = ttk.LabelFrame(shell, text=" Co sprawdzamy ", padding=10)
    form_box.pack(fill=tk.X)

    self.val_model_var = tk.StringVar()
    row1 = ttk.Frame(form_box, style="Panel.TFrame")
    row1.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(row1, text="Model do sprawdzenia (.pt):", style="Panel.TLabel").pack(anchor=tk.W, fill=tk.X)
    row1_input = ttk.Frame(row1, style="Panel.TFrame")
    row1_input.pack(fill=tk.X, pady=(4, 0))
    ttk.Entry(row1_input, textvariable=self.val_model_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(
        row1_input,
        text="[ MODEL ] Wybierz",
        command=lambda: self._pick_file(
            self.val_model_var,
            "*.pt",
            initialdir=self._get_validation_model_picker_dir(),
        ),
    ).pack(side=tk.LEFT, padx=(8, 0))

    self.val_data_var = tk.StringVar()
    row2 = ttk.Frame(form_box, style="Panel.TFrame")
    row2.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(row2, text="Tor walidacji (folder lub data.yaml):", style="Panel.TLabel").pack(anchor=tk.W, fill=tk.X)
    row2_input = ttk.Frame(row2, style="Panel.TFrame")
    row2_input.pack(fill=tk.X, pady=(4, 0))
    ttk.Entry(row2_input, textvariable=self.val_data_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(
        row2_input,
        text="[ TOR ] Wybierz",
        command=lambda: self._pick_dir(
            self.val_data_var,
            initialdir=self._get_validation_dataset_picker_dir(),
        ),
    ).pack(side=tk.LEFT, padx=(8, 0))

    split_row = ttk.Frame(form_box, style="Panel.TFrame")
    split_row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(split_row, text="Split egzaminu:", style="Panel.TLabel").pack(side=tk.LEFT)
    self.val_split_var = tk.StringVar(value="val")
    split_combo = ttk.Combobox(
        split_row,
        textvariable=self.val_split_var,
        values=["val", "test", "train"],
        state="readonly",
        width=12,
    )
    split_combo.pack(side=tk.LEFT, padx=(8, 0))

    action_row = ttk.Frame(form_box, style="Panel.TFrame")
    action_row.pack(fill=tk.X, pady=(10, 0))
    self.btn_run_val = ttk.Button(
        action_row,
        text="[ START ] Uruchom walidację",
        style="Accent.TButton",
        command=self._run_validation,
    )
    self.btn_run_val.pack(side=tk.LEFT)
    self.val_status = ttk.Label(action_row, text="Gotowy", style="PanelMuted.TLabel")
    self.val_status.pack(side=tk.LEFT, padx=(10, 0))

    ttk.Label(
        form_box,
        text="Walidacja nie wybiera zwycięzcy. Sprawdza jeden model na jednym torze i daje tabelę metryk do interpretacji.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=720,
    ).pack(anchor=tk.W, fill=tk.X, pady=(10, 0))

    HELP.bind_help(row1_input, "tr_val_model")
    HELP.bind_help(row2_input, "tr_val_data")
    HELP.bind_help(self.btn_run_val, "tr_val_btn")
    HELP.bind_help(split_combo, "tr_val_split")

    summary_box = ttk.LabelFrame(shell, text=" Wynik walidacji ", padding=10)
    summary_box.pack(fill=tk.X, pady=(12, 0))

    self.val_summary_title_var = tk.StringVar(
        value="Uruchom walidację, aby zobaczyć werdykt."
    )
    self.val_summary_note_var = tk.StringVar(
        value="Szczegółowa tabela metryk będzie dostępna w osobnym oknie."
    )
    self.val_summary_metric_var = tk.StringVar(value="Metryki: -")
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_title_var,
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).pack(anchor=tk.W, fill=tk.X)
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_metric_var,
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_note_var,
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self.btn_open_val_results = ttk.Button(
        summary_box,
        text="[ WYNIKI ] Pokaż tabelę metryk",
        command=self._open_validation_results_modal,
        state=tk.DISABLED,
    )
    self.btn_open_val_results.pack(anchor=tk.W)
    self._set_validation_summary(
        "Uruchom walidację, aby zobaczyć werdykt.",
        [],
        "Szczegółowa tabela metryk będzie dostępna w osobnym oknie.",
    )

def _ensure_contextual_validation_state(self) -> None:
    if not hasattr(self, "val_model_var"):
        self.val_model_var = tk.StringVar()
    if not hasattr(self, "val_data_var"):
        self.val_data_var = tk.StringVar()
    if not hasattr(self, "val_split_var"):
        self.val_split_var = tk.StringVar(value="val")
    if not hasattr(self, "val_summary_title_var"):
        self.val_summary_title_var = tk.StringVar(value="Uruchom walidację, aby zobaczyć werdykt.")
    if not hasattr(self, "val_summary_note_var"):
        self.val_summary_note_var = tk.StringVar(value="Szczegółowa tabela metryk będzie dostępna w osobnym oknie.")
    if not hasattr(self, "val_summary_metric_var"):
        self.val_summary_metric_var = tk.StringVar(value="Metryki: -")

def _default_contextual_validation_dataset(self) -> str:
    for attr in ("val_data_var", "dataset_var", "rank_data_dir"):
        var = getattr(self, attr, None)
        try:
            value = str(var.get() or "").strip()
        except Exception:
            value = ""
        if value:
            return value

    try:
        info = self._resolve_ranking_reference_source()
        for key in ("data_yaml_path", "yaml_path", "reference_dir", "selected_path"):
            value = str(info.get(key) or "").strip()
            if value:
                return value
    except Exception:
        pass

    try:
        return str(CONFIG.get_datasets_dir(self._get_selected_training_target()))
    except Exception:
        return ""

def _open_model_validation_modal(
    self,
    model_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    split: str | None = None,
    context_label: str | None = None,
):
    _ensure_contextual_validation_state(self)

    model_value = str(model_path or "").strip()
    dataset_value = str(dataset_path or "").strip() or _default_contextual_validation_dataset(self)
    split_value = str(split or "").strip().lower()
    if split_value not in {"val", "test", "train"}:
        split_value = "val"

    try:
        if model_value:
            self.val_model_var.set(model_value)
        if dataset_value:
            self.val_data_var.set(dataset_value)
        self.val_split_var.set(split_value)
    except Exception:
        pass

    existing = getattr(self, "_validation_launcher_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            if not bool(getattr(self, "val_is_running", False)):
                self._set_validation_summary(
                    "Uruchom walidację, aby zobaczyć werdykt.",
                    [],
                    "Po zakończeniu testu otworzysz pełną tabelę metryk.",
                )
            existing.lift()
            existing.focus_force()
            return existing
    except Exception:
        pass

    palette = getattr(self.app, "palette", {}) if getattr(self, "app", None) is not None else {}
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._validation_launcher_modal = dialog
    dialog.title("Walidacja modelu")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.minsize(860, 660)
    except Exception:
        pass
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        if bool(getattr(self, "val_is_running", False)):
            return messagebox.showinfo(
                "Walidacja w toku",
                "Poczekaj na zakończenie walidacji. Okno pozostaje aktywne, żeby nie zgubić wyniku.",
            )
        try:
            self._validation_launcher_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=12, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(0, weight=1)

    content_shell = ttk.Frame(shell, style="Panel.TFrame")
    content_shell.grid(row=0, column=0, sticky="nsew")
    content_shell.columnconfigure(0, weight=1)
    content_shell.rowconfigure(0, weight=1)

    content_canvas = tk.Canvas(
        content_shell,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    content_canvas.grid(row=0, column=0, sticky="nsew")
    content_scrollbar = WebSlimScrollbar(content_shell, orient=tk.VERTICAL, command=content_canvas.yview)
    content_scrollbar.grid(row=0, column=1, sticky="ns")
    content_canvas.configure(yscrollcommand=content_scrollbar.set)

    content = ttk.Frame(content_canvas, style="Panel.TFrame")
    content_window = content_canvas.create_window((0, 0), window=content, anchor="nw")
    content.columnconfigure(0, weight=1)

    def _sync_content_scrollregion(_event=None):
        try:
            content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        except Exception:
            pass

    def _sync_content_width(event):
        try:
            content_canvas.itemconfigure(content_window, width=max(1, int(event.width)))
        except Exception:
            pass

    def _on_content_wheel(event):
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            units = -1 if delta > 0 else 1
        else:
            button = int(getattr(event, "num", 0) or 0)
            units = -1 if button == 4 else (1 if button == 5 else 0)
        if units:
            try:
                content_canvas.yview_scroll(units * 3, "units")
            except Exception:
                pass
        return "break"

    content.bind("<Configure>", _sync_content_scrollregion, add="+")
    content_canvas.bind("<Configure>", _sync_content_width, add="+")
    for widget in (dialog, shell, content_shell, content_canvas, content):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(sequence, _on_content_wheel, add="+")
            except Exception:
                pass

    ttk.Label(
        content,
        text="[ TEST ] Walidacja jednego modelu",
        style="PanelTitle.TLabel",
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")

    subtitle_text = str(context_label or "").strip()
    if not subtitle_text:
        subtitle_text = "Sprawdzasz konkretny model na wskazanym torze. To diagnostyka, nie ranking."
    ttk.Label(
        content,
        text=subtitle_text,
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=780,
    ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

    form_box = ttk.LabelFrame(content, text=" Co sprawdzamy ", padding=10)
    form_box.grid(row=2, column=0, sticky="ew")
    form_box.columnconfigure(0, weight=1)

    model_row = ttk.Frame(form_box, style="Panel.TFrame")
    model_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    model_row.columnconfigure(0, weight=1)
    ttk.Label(model_row, text="Model do sprawdzenia (.pt):", style="Panel.TLabel").grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Entry(model_row, textvariable=self.val_model_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ttk.Button(
        model_row,
        text="[ MODEL ] Wybierz",
        command=lambda: self._pick_file(
            self.val_model_var,
            "*.pt",
            initialdir=self._get_validation_model_picker_dir(),
        ),
    ).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(4, 0))

    data_row = ttk.Frame(form_box, style="Panel.TFrame")
    data_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    data_row.columnconfigure(0, weight=1)
    ttk.Label(data_row, text="Tor walidacji (folder lub data.yaml):", style="Panel.TLabel").grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Entry(data_row, textvariable=self.val_data_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ttk.Button(
        data_row,
        text="[ TOR ] Wybierz",
        command=lambda: self._pick_dir(
            self.val_data_var,
            initialdir=self._get_validation_dataset_picker_dir(),
        ),
    ).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(4, 0))

    split_row = ttk.Frame(form_box, style="Panel.TFrame")
    split_row.grid(row=2, column=0, sticky="ew")
    ttk.Label(split_row, text="Split egzaminu:", style="Panel.TLabel").pack(side=tk.LEFT)
    ttk.Combobox(
        split_row,
        textvariable=self.val_split_var,
        values=["val", "test", "train"],
        state="readonly",
        width=12,
    ).pack(side=tk.LEFT, padx=(8, 0))

    action_row = ttk.Frame(form_box, style="Panel.TFrame")
    action_row.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    self.btn_run_val = ttk.Button(
        action_row,
        text="[ START ] Uruchom walidację",
        style="Accent.TButton",
        command=self._run_validation,
    )
    self.btn_run_val.pack(side=tk.LEFT)
    self.val_status = ttk.Label(action_row, text="Gotowy", style="PanelMuted.TLabel")
    self.val_status.pack(side=tk.LEFT, padx=(10, 0))

    summary_box = ttk.LabelFrame(content, text=" Wynik ", padding=10)
    summary_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_title_var,
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=780,
    ).pack(anchor=tk.W, fill=tk.X)
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_metric_var,
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=780,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    ttk.Label(
        summary_box,
        textvariable=self.val_summary_note_var,
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=780,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self.btn_open_val_results = ttk.Button(
        summary_box,
        text="[ WYNIKI ] Pokaż tabelę metryk",
        command=self._open_validation_results_modal,
        state=(tk.NORMAL if getattr(self, "_validation_metric_rows", None) else tk.DISABLED),
    )
    self.btn_open_val_results.pack(anchor=tk.W)

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    ttk.Label(
        bottom,
        text="Ranking wybiera zwycięzcę. Walidacja tylko sprawdza ten jeden model na wybranym torze.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=620,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(bottom, text="[ X ] Zamknij", command=close_dialog).pack(side=tk.RIGHT)

    self._set_validation_summary(
        "Uruchom walidację, aby zobaczyć werdykt.",
        [],
        "Po zakończeniu testu otworzysz pełną tabelę metryk.",
    )

    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(900, int(root.winfo_width() * 0.62)), 1120)
        height = min(max(680, int(root.winfo_height() * 0.72)), 900)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("920x700")

    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass
    return dialog

def _open_selected_run_validation_modal(self, event=None):
    if event is not None and hasattr(self, "tree") and getattr(event, "y", None) is not None:
        try:
            row_id = self.tree.identify_row(event.y)
        except Exception:
            row_id = ""
        if row_id:
            try:
                self.tree.selection_set(row_id)
                self.tree.focus(row_id)
            except Exception:
                pass

    run = None
    try:
        run = self._selected_run()
    except Exception:
        run = None
    if run is None:
        return messagebox.showwarning("Brak runu", "Najpierw wybierz run z historii treningu.")

    model_path = None
    try:
        model_path = self._resolve_history_run_best_weights(run)
    except Exception:
        model_path = None
    if model_path is None:
        return messagebox.showwarning("Brak best.pt", "Wybrany run nie ma dostępnego pliku best.pt do walidacji.")

    dataset_path = str(getattr(run, "dataset_path", "") or "").strip()
    run_label = str(getattr(run, "name", "") or getattr(run, "id", "") or Path(str(model_path)).name)
    return self._open_model_validation_modal(
        model_path=model_path,
        dataset_path=dataset_path,
        context_label=f"Run: {run_label}. Sprawdzasz jego best.pt na wybranym torze walidacji.",
    )

def _open_current_run_validation_modal(self):
    run = None
    run_id = str(getattr(self, "_run_details_current_run_id", "") or "").strip()
    if run_id:
        try:
            run = self.history.get_run(run_id)
        except Exception:
            run = None
    if run is None:
        try:
            run = self._selected_run()
        except Exception:
            run = None
    if run is None:
        return messagebox.showwarning("Brak runu", "Najpierw wybierz run z historii treningu.")
    try:
        if hasattr(self, "tree"):
            run_id = str(getattr(run, "id", "") or "").strip()
            if run_id:
                self.tree.selection_set(run_id)
                self.tree.focus(run_id)
    except Exception:
        pass
    return self._open_selected_run_validation_modal()

def _open_selected_ranking_validation_modal(self):
    tree = getattr(self, "rank_tree", None)
    ref = {}
    if tree is not None:
        try:
            item_id = str(tree.focus() or "")
            if not item_id:
                selection = list(tree.selection() or [])
                item_id = str(selection[0]) if selection else ""
        except Exception:
            item_id = ""
        refs = getattr(self, "_ranking_tree_entry_refs", {}) or {}
        ref = refs.get(item_id, {}) if isinstance(refs, dict) else {}
    ref = dict(ref) if isinstance(ref, dict) else {}

    model_path = str(ref.get("model_path", "") or "").strip()
    if not model_path:
        return messagebox.showwarning("Brak modelu", "Ten wiersz rankingu nie ma przypisanego pliku modelu.")

    dataset_path = ""
    try:
        info = self._resolve_ranking_reference_source()
        for key in ("data_yaml_path", "yaml_path", "reference_dir", "selected_path"):
            dataset_path = str(info.get(key) or "").strip()
            if dataset_path:
                break
    except Exception:
        dataset_path = ""

    split = "val"
    try:
        split = self._get_ranking_split_name()
    except Exception:
        pass
    label = str(ref.get("choice_label", "") or Path(model_path).name)
    return self._open_model_validation_modal(
        model_path=model_path,
        dataset_path=dataset_path,
        split=split,
        context_label=f"Kandydat z rankingu: {label}. Test użyje aktualnie wybranego toru rankingowego.",
    )

def _format_validation_metric_band(self, raw_name: str, value) -> tuple[str, str]:
    raw = str(raw_name or "").strip().lower()
    if "map50-95" in raw:
        return self._format_training_metric_band("map50_95", value)
    if "map50" in raw:
        return self._format_training_metric_band("map50", value)
    if "precision" in raw:
        return self._format_training_metric_band("precision", value)
    if "recall" in raw:
        return self._format_training_metric_band("recall", value)
    if "fitness" in raw:
        return "monitoruj", "syntetyczna"
    return "-", "-"

def _extract_validation_metric_rows(self, metrics) -> list[tuple[str, str, str, str]]:
    preferred_order = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(P)",
        "metrics/recall(P)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
        "fitness",
    ]

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict) and results_dict:
        ordered_keys = [key for key in preferred_order if key in results_dict]
        ordered_keys.extend(key for key in results_dict.keys() if key not in ordered_keys)
        rows: list[tuple[str, str, str, str]] = []
        for key in ordered_keys:
            band, range_text = self._format_validation_metric_band(key, results_dict.get(key))
            rows.append(
                (
                    self._format_validation_metric_name(key),
                    self._format_validation_metric_value(results_dict.get(key)),
                    band,
                    range_text,
                )
            )
        return rows

    rows: list[tuple[str, str, str, str]] = []
    if hasattr(metrics, "box"):
        box_rows = [
            ("mAP50 (boxy)", getattr(metrics.box, "map50", 0)),
            ("mAP50-95 (boxy)", getattr(metrics.box, "map", 0)),
            ("Precision (boxy)", getattr(metrics.box, "mp", 0)),
            ("Recall (boxy)", getattr(metrics.box, "mr", 0)),
        ]
        for name, value in box_rows:
            band, range_text = self._format_validation_metric_band(name, value)
            rows.append((name, self._format_validation_metric_value(value), band, range_text))
    if hasattr(metrics, "pose"):
        pose_rows = [
            ("mAP50 (punkty)", getattr(metrics.pose, "map50", 0)),
            ("mAP50-95 (punkty)", getattr(metrics.pose, "map", 0)),
        ]
        for name, value in pose_rows:
            band, range_text = self._format_validation_metric_band(name, value)
            rows.append((name, self._format_validation_metric_value(value), band, range_text))
    return rows

def _open_validation_results_modal(self):
    rows = list(getattr(self, "_validation_metric_rows", []) or [])
    if not rows:
        return messagebox.showinfo(
            "Wyniki walidacji",
            "Brak tabeli metryk. Najpierw uruchom walidację modelu.",
        )

    existing = getattr(self, "_validation_results_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._validation_results_modal = dialog
    dialog.title("Wyniki walidacji modelu")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        try:
            self._validation_results_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=12, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_rowconfigure(2, weight=1)
    shell.grid_columnconfigure(0, weight=1)

    title = str(getattr(self, "_validation_summary_title", "") or "Wyniki walidacji modelu")
    note = str(getattr(self, "_validation_summary_note", "") or "")
    ttk.Label(
        shell,
        text=title,
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=980,
    ).grid(row=0, column=0, sticky="ew")
    ttk.Label(
        shell,
        text=note or "Tabela pokazuje metryki zwrócone przez walidację YOLO.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=980,
    ).grid(row=1, column=0, sticky="ew", pady=(3, 10))

    table_frame = ttk.Frame(shell, style="Panel.TFrame")
    table_frame.grid(row=2, column=0, sticky="nsew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    cols = ("metric", "value", "rating", "range")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
    headings = {
        "metric": "Metryka",
        "value": "Wartość",
        "rating": "Ocena",
        "range": "Zakres",
    }
    for col in cols:
        tree.heading(col, text=headings[col])
    tree.column("metric", width=360, minwidth=240, anchor=tk.W, stretch=True)
    tree.column("value", width=120, minwidth=90, anchor=tk.CENTER, stretch=False)
    tree.column("rating", width=150, minwidth=110, anchor=tk.CENTER, stretch=False)
    tree.column("range", width=180, minwidth=120, anchor=tk.CENTER, stretch=False)

    try:
        tree.tag_configure(
            "good",
            foreground=palette.get("success", "#2ecc71"),
        )
        tree.tag_configure(
            "warn",
            foreground=palette.get("warning", "#f0b44c"),
        )
        tree.tag_configure(
            "bad",
            foreground=palette.get("error", "#e05d5d"),
        )
    except Exception:
        pass

    def row_tag(row: tuple) -> tuple[str, ...]:
        band = str(row[2] if len(row) > 2 else "").strip().lower()
        if any(token in band for token in ("dobr", "wysok", "ok", "bardzo")):
            return ("good",)
        if any(token in band for token in ("nis", "słab", "blad", "błąd")):
            return ("bad",)
        if any(token in band for token in ("monitor", "śred", "przeci", "uwag")):
            return ("warn",)
        return ()

    for row in rows:
        try:
            tree.insert("", tk.END, values=tuple(row), tags=row_tag(tuple(row)))
        except Exception:
            continue

    yscroll = WebSlimScrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    xscroll = WebSlimScrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")

    def _wheel_units(event) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            return -1 if delta > 0 else 1
        button = int(getattr(event, "num", 0) or 0)
        if button == 4:
            return -1
        if button == 5:
            return 1
        return 0

    def _on_tree_wheel(event):
        units = _wheel_units(event)
        if units:
            try:
                tree.yview_scroll(units * 3, "units")
            except Exception:
                pass
        return "break"

    for widget in (dialog, shell, table_frame, tree):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(sequence, _on_tree_wheel)
            except Exception:
                pass

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    ttk.Label(
        bottom,
        text="Walidacja nie wybiera modelu projektowego. To szybki test jakości na wskazanym splicie.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(bottom, text="[ X ] Zamknij", command=close_dialog).pack(side=tk.RIGHT)

    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(920, int(root.winfo_width() * 0.74)), 1260)
        height = min(max(520, int(root.winfo_height() * 0.66)), 820)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("980x580")
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

def _set_validation_summary(
    self,
    title: str,
    rows: list[tuple[str, str, str, str]] | None = None,
    note: str = "",
):
    rows = list(rows or [])
    self._validation_summary_title = str(title)
    self._validation_summary_note = str(note or "").strip()
    self._validation_metric_rows = rows

    title_var = getattr(self, "val_summary_title_var", None)
    if title_var is not None:
        try:
            title_var.set(str(title))
        except Exception:
            pass

    note_var = getattr(self, "val_summary_note_var", None)
    if note_var is not None:
        try:
            note_var.set(str(note or "").strip())
        except Exception:
            pass

    metric_var = getattr(self, "val_summary_metric_var", None)
    if metric_var is not None:
        try:
            if rows:
                metric_var.set(f"Metryki: {len(rows)} | {_validation_best_metric_summary(rows)}")
            else:
                metric_var.set("Metryki: -")
        except Exception:
            pass

    button = getattr(self, "btn_open_val_results", None)
    if button is not None:
        try:
            button.configure(state=(tk.NORMAL if rows else tk.DISABLED))
        except Exception:
            pass

    existing = getattr(self, "_validation_results_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.destroy()
            self._validation_results_modal = None
    except Exception:
        pass
