#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 dataset creator and splitter panel builders extracted from tab_training.py."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18
STEP4_PZ1_SECTION_TITLE_FONT = ("Segoe UI Semibold", 10)
STEP4_PZ1_SECTION_TITLE_PADY = (8, 7)

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _get_step4_table_colors(self):
    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    field = palette.get("field", panel_alt)
    success = palette.get("success", "#4ec9b0")
    border_base = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    border = blend_hex_colors(success, border_base, 0.62)
    return {
        "panel": panel,
        "header": blend_hex_colors(success, panel_alt, 0.84),
        "row": field,
        "row_alt": blend_hex_colors(panel_alt, panel, 0.45),
        "border": border,
        "fg": palette.get("fg", "#f3f3f3"),
        "muted": palette.get("muted", "#c7c7c7"),
        "accent": success,
        "warning": palette.get("warning", "#d7ba7d"),
    }


def _ensure_step4_pz1_section_styles(self):
    colors = _get_step4_table_colors(self)
    palette = getattr(self.app, "palette", {})
    try:
        style = ttk.Style()
        style.configure(
            "Step4PZ1Section.TLabelframe",
            background=palette.get("panel", "#252526"),
            foreground=colors["fg"],
        )
        style.configure(
            "Step4PZ1Section.TLabelframe.Label",
            font=STEP4_PZ1_SECTION_TITLE_FONT,
            foreground=colors["accent"],
            background=palette.get("panel", "#252526"),
        )
    except Exception:
        pass


def _make_step4_pz1_section_title(self, parent, text: str):
    colors = _get_step4_table_colors(self)
    palette = getattr(self.app, "palette", {})
    label = tk.Label(
        parent,
        text=text,
        font=STEP4_PZ1_SECTION_TITLE_FONT,
        padx=2,
        pady=0,
        anchor="w",
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", "#252526"),
        fg=colors["accent"],
    )
    label.pack(fill=tk.X, pady=STEP4_PZ1_SECTION_TITLE_PADY)
    return label


def _make_step4_action_shell(self, parent, *, title: str, description: str):
    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", panel)
    border = blend_hex_colors(palette.get("success", "#2fa36b"), palette.get("panel_border", "#3c3c3c"), 0.68)
    bg = blend_hex_colors(panel_alt, panel, 0.60)
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#4ec9b0")

    shell = tk.Frame(parent, bg=border, padx=1, pady=1, bd=0, highlightthickness=0)
    inner = tk.Frame(shell, bg=bg, padx=12, pady=10, bd=0, highlightthickness=0)
    inner.pack(fill=tk.X, expand=True)
    setattr(shell, "_step4_action_inner", inner)
    setattr(shell, "_step4_action_bg", bg)

    tk.Label(
        inner,
        text=title,
        font=STEP4_PZ1_SECTION_TITLE_FONT,
        fg=success,
        bg=bg,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    ).pack(anchor=tk.W, fill=tk.X)
    tk.Label(
        inner,
        text=description,
        font=("Segoe UI", 9),
        fg=muted,
        bg=bg,
        wraplength=720,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 10))

    return shell, inner, bg


def _make_step4_flow_strip(self, parent):
    colors = _get_step4_table_colors(self)
    strip = tk.Frame(parent, bg=colors["panel"], bd=0, highlightthickness=0)
    strip.pack(fill=tk.X, pady=(0, 12))
    steps = (
        ("1", "Materia\u0142"),
        ("2", "Split"),
        ("3", "Powi\u0119kszenie"),
        ("4", "Wariant"),
    )
    for index, (number, label) in enumerate(steps):
        pill = tk.Frame(
            strip,
            bg=colors["row_alt"],
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["border"],
            padx=8,
            pady=5,
        )
        pill.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            pill,
            text=number,
            bg=colors["header"],
            fg=colors["accent"],
            font=("Segoe UI Semibold", 8),
            width=2,
            bd=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            pill,
            text=label,
            bg=colors["row_alt"],
            fg=colors["fg"],
            font=("Segoe UI", 8),
            bd=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT)
        if index < len(steps) - 1:
            tk.Label(
                strip,
                text="\u203a",
                bg=colors["panel"],
                fg=colors["muted"],
                font=("Segoe UI Semibold", 11),
                bd=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(0, 6))
    return strip


def _make_step4_decision_table(self, parent, *, prefix: str, bg: str):
    colors = _get_step4_table_colors(self)
    shell = tk.Frame(
        parent,
        bg=colors["border"],
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
    )
    shell.pack(fill=tk.X, pady=(0, 10))
    grid = tk.Frame(shell, bg=colors["border"], bd=0, highlightthickness=0)
    grid.pack(fill=tk.X)
    setattr(self, f"{prefix}_decision_summary_frame", shell)
    setattr(self, f"{prefix}_decision_summary_grid", grid)

    rows = {}
    columns = (
        ("train", "Train"),
        ("val", "Val"),
        ("test", "Test"),
        ("augmentation", "Powiększenie"),
    )
    for column, (_key, title) in enumerate(columns):
        grid.columnconfigure(column, weight=1, uniform=f"{prefix}_decision")
        tk.Label(
            grid,
            text=title,
            anchor=tk.CENTER,
            padx=9,
            pady=5,
            bg=colors["header"],
            fg=colors["accent"],
            font=("Segoe UI Semibold", 8),
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["border"],
        ).grid(row=0, column=column, sticky="nsew")

    for column, (key, _title) in enumerate(columns):
        var = tk.StringVar(value="-")
        setattr(self, f"{prefix}_decision_{key}_var", var)
        cell_bg = colors["row"] if column % 2 == 0 else colors["row_alt"]
        value = tk.Label(
            grid,
            textvariable=var,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            padx=9,
            pady=7,
            bg=cell_bg,
            fg=colors["fg"],
            font=("Segoe UI Semibold", 10),
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["border"],
        )
        value.grid(row=1, column=column, sticky="nsew")
        rows[key] = value
    setattr(self, f"_{prefix}_decision_value_widgets", rows)
    return shell


def _build_step4_augmentation_controls(self, parent, *, target: str):
    _ensure_step4_pz1_section_styles(self)
    normalized_target = CONFIG.normalize_task_target(target)
    prefix = "creator" if normalized_target == "plate" else "split"
    target_label = "tablic" if normalized_target == "plate" else "znaków"

    section_title = (
        " 3. Syntetyczne uzupełnienie train "
        if normalized_target == "char"
        else f" 3. Syntetyczne powiększenie liczby {target_label} zbioru train "
    )
    frame = ttk.LabelFrame(
        parent,
        text=section_title,
        padding=10,
        style="Step4PZ1Section.TLabelframe",
    )
    frame.pack(fill=tk.X, pady=(12, 8))
    setattr(self, f"{prefix}_augmentation_frame", frame)

    try:
        self._ensure_step4_augmentation_profile(normalized_target)
    except Exception:
        pass
    table_colors = _get_step4_table_colors(self)

    augmentation_intro_label = tk.Label(
        frame,
        text=(
            (
                "Opcjonalnie sprawdź niedoreprezentowane znaki MZ. Histogram może ustawić liczbę "
                "syntetyków, ale ostatecznym przełącznikiem jest pole liczby syntetycznych obrazów train. "
                "Val i test pozostają bez zmian."
            )
            if normalized_target == "char"
            else (
                "Opcjonalnie powiększ wyłącznie część train aktualnego wariantu datasetu. "
                "Wygenerowane obrazy służą tylko treningowi tego wariantu: nie trafiają do puli obrazów "
                "projektu i nie są bazą kolejnej iteracji. Val i test pozostają oryginalne."
            )
        ),
        bg=table_colors["row"],
        fg=table_colors["fg"],
        justify=tk.LEFT,
        wraplength=700,
        anchor=tk.W,
        padx=10,
        pady=8,
        bd=0,
        highlightthickness=1,
        highlightbackground=table_colors["border"],
        highlightcolor=table_colors["border"],
    )
    augmentation_intro_label.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 6))
    augmentation_intro_label.bind(
        "<Configure>",
        lambda event, label=augmentation_intro_label: (
            label.configure(wraplength=max(360, int(event.width) - 24))
            if int(float(label.cget("wraplength") or 0)) != max(360, int(event.width) - 24)
            else None
        ),
    )

    try:
        augmentation_profile = self._ensure_step4_augmentation_profile(normalized_target)
    except Exception:
        augmentation_profile = None
    initial_extra = max(0, int(getattr(augmentation_profile, "extra_count", 0) or 0))
    initial_sample = max(1, int(getattr(augmentation_profile, "sample_size", 1) or 1))
    enabled_var = tk.BooleanVar(value=initial_extra > 0)
    extra_var = tk.StringVar(value=str(initial_extra))
    sample_var = tk.IntVar(value=initial_sample)
    class_var = tk.StringVar(value=str(getattr(augmentation_profile, "class_name", "") or ("plate" if normalized_target == "plate" else "")))
    setattr(self, f"{prefix}_aug_enabled_var", enabled_var)
    setattr(self, f"{prefix}_aug_extra_var", extra_var)
    setattr(self, f"{prefix}_aug_sample_var", sample_var)
    if normalized_target == "plate":
        setattr(self, f"{prefix}_class_name_var", class_var)

    def _on_plan_change(*_args):
        try:
            enabled_var.set(int(float(extra_var.get() or 0)) > 0)
        except Exception:
            enabled_var.set(False)
        try:
            self._refresh_step4_augmentation_summary(normalized_target)
        except Exception:
            pass

    for plan_var in (extra_var, class_var):
        try:
            plan_var.trace_add("write", _on_plan_change)
        except Exception:
            pass

    plan_frame = ttk.Frame(frame)
    plan_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 8))
    plan_frame.columnconfigure(1, weight=0)
    plan_frame.columnconfigure(2, weight=0)
    plan_frame.columnconfigure(3, weight=1)
    if normalized_target == "char":
        pending_plan = getattr(self, "_pending_character_balance_plan", None)
        planned_images = max(0, int(getattr(pending_plan, "planned_images", 0) or 0)) if pending_plan is not None else 0
        try:
            pending_requested_extras = dict(getattr(pending_plan, "requested_extra_by_symbol", {}) or {})
        except Exception:
            pending_requested_extras = {}
        if pending_plan is not None:
            try:
                enabled_var.set(planned_images > 0)
                extra_var.set(planned_images)
                sample_var.set(max(1, planned_images))
            except Exception:
                pass
        mz_status_var = tk.StringVar(
            value=(
                (
                    "Histogram MZ: "
                    f"{', '.join(f'{key}+{value}' for key, value in sorted(pending_requested_extras.items())) or 'bez ręcznych dodatków'}. "
                    f"Ustawiono +{planned_images} syntetycznych obrazów train."
                )
                if planned_images > 0 or pending_requested_extras
                else (
                    "Nie ustawiono syntetycznego podbicia znaków. Histogram MZ pozwala wskazać, "
                    "które znaki podnieść syntetycznie."
                )
            )
        )
        setattr(self, "split_mz_representation_status_var", mz_status_var)
        count_origin_var = tk.StringVar(value="bez syntetyków")
        setattr(self, "split_aug_count_origin_var", count_origin_var)

        def _open_char_representation_plan():
            try:
                source_raw = str(getattr(self, "split_src_var", tk.StringVar()).get() or "").strip()
            except Exception:
                source_raw = ""
            if not source_raw:
                messagebox.showwarning(
                    "Brak datasetu",
                    "Najpierw wybierz dataset znaków, dla którego PZ1 ma policzyć reprezentację.",
                    parent=getattr(self, "frame", None),
                )
                return
            from . import z4_character_balance

            z4_character_balance.open_character_class_distribution_dialog(
                self,
                dataset_root=Path(source_raw),
                read_only=False,
                context="pz1",
            )

        tk.Label(
            plan_frame,
            textvariable=mz_status_var,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=620,
            padx=8,
            pady=7,
            bg=table_colors["row"],
            fg=table_colors["fg"],
            font=("Segoe UI", 9, "bold"),
            highlightthickness=1,
            highlightbackground=table_colors["border"],
        ).grid(row=0, column=0, columnspan=4, sticky=tk.EW, pady=(0, 6))
        tk.Label(
            plan_frame,
            text="1. Doreprezentowanie znaków MZ (opcjonalne)",
            bg=table_colors["panel"],
            fg=table_colors["accent"],
            font=("Segoe UI Semibold", 9),
            anchor=tk.W,
        ).grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(2, 2))
        ttk.Button(
            plan_frame,
            text="Sprawdź niedoreprezentowane znaki",
            command=_open_char_representation_plan,
        ).grid(row=2, column=0, sticky=tk.W, pady=(0, 6))
        ttk.Label(
            plan_frame,
            text="Histogram może policzyć propozycję. Ten krok można pominąć.",
            foreground=table_colors["muted"],
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=(10, 0), pady=(0, 6))
        tk.Label(
            plan_frame,
            text="2. Liczba syntetyków train",
            bg=table_colors["panel"],
            fg=table_colors["accent"],
            font=("Segoe UI Semibold", 9),
            anchor=tk.W,
        ).grid(row=3, column=0, columnspan=4, sticky=tk.EW, pady=(4, 2))
        ttk.Label(plan_frame, text="Generuj dodatkowo").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Spinbox(plan_frame, from_=0, to=100000, textvariable=extra_var, width=9).grid(row=4, column=1, sticky=tk.W, pady=2)
        ttk.Label(plan_frame, text="syntetycznych obrazów train").grid(row=4, column=2, sticky=tk.W, padx=(8, 0), pady=2)
        tk.Label(
            plan_frame,
            textvariable=count_origin_var,
            anchor=tk.CENTER,
            padx=10,
            pady=4,
            bg=table_colors["header"],
            fg=table_colors["accent"],
            font=("Segoe UI Semibold", 8),
            highlightthickness=1,
            highlightbackground=table_colors["border"],
        ).grid(row=4, column=3, sticky=tk.W, padx=(12, 0), pady=2)
        ttk.Label(
            plan_frame,
            text="Wpisz 0, aby pominąć syntetyki. Jeśli zmienisz wartość po histogramie, traktujemy ją jako ręczną korektę.",
            foreground=table_colors["muted"],
            wraplength=620,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=(2, 4))
    else:
        ttk.Label(plan_frame, text="Generuj dodatkowo").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        ttk.Spinbox(plan_frame, from_=0, to=100000, textvariable=extra_var, width=9).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(plan_frame, text="syntetycznych obrazów train").grid(row=0, column=2, sticky=tk.W, padx=(8, 0), pady=2)
        ttk.Label(
            plan_frame,
            text="Wpisz 0, aby utworzyć wariant bez syntetycznego powiększenia train.",
            foreground=table_colors["muted"],
            wraplength=620,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(2, 4))
    if normalized_target == "plate":
        ttk.Label(plan_frame, text="Klasa YOLO").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(4, 0))
        ttk.Entry(plan_frame, textvariable=class_var, width=18).grid(row=2, column=1, sticky=tk.W, pady=(4, 0))
        ttk.Label(
            plan_frame,
            text="Etykieta klasy zapisana w data.yaml, np. plate albo pl.",
            foreground=table_colors["muted"],
        ).grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))

    balance_vars = {
        "current": tk.StringVar(value="-"),
        "extra": tk.StringVar(value="+0"),
        "total": tk.StringVar(value="-"),
    }
    for key, var in balance_vars.items():
        setattr(self, f"{prefix}_aug_balance_{key}_var", var)

    table_colors = _get_step4_table_colors(self)
    balance_frame = tk.Frame(
        frame,
        bg=table_colors["border"],
        padx=1,
        pady=1,
        bd=0,
        highlightthickness=0,
    )
    balance_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(2, 8))
    for column in range(3):
        balance_frame.columnconfigure(column, weight=1, uniform=f"{prefix}_aug_balance")
    for column, title in enumerate(("Teraz w train", "Dodajemy", "Po powi\u0119kszeniu")):
        tk.Label(
            balance_frame,
            text=title,
            anchor=tk.CENTER,
            padx=9,
            pady=5,
            bg=table_colors["header"],
            fg=table_colors["accent"],
            font=("Segoe UI Semibold", 8),
            bd=0,
            highlightthickness=1,
            highlightbackground=table_colors["border"],
            highlightcolor=table_colors["border"],
        ).grid(row=0, column=column, sticky=tk.EW)
    for column, key in enumerate(("current", "extra", "total")):
        tk.Label(
            balance_frame,
            textvariable=balance_vars[key],
            anchor=tk.CENTER,
            justify=tk.CENTER,
            padx=9,
            pady=6,
            bg=(table_colors["row"] if column % 2 == 0 else table_colors["row_alt"]),
            fg=table_colors["fg"],
            font=("Segoe UI", 9, "bold"),
            bd=0,
            highlightthickness=1,
            highlightbackground=table_colors["border"],
            highlightcolor=table_colors["border"],
        ).grid(row=1, column=column, sticky=tk.EW)

    summary_var = tk.StringVar(
        value="Syntetyki: 0. Dataset powstanie tylko z materiału źródłowego projektu."
    )
    setattr(self, f"{prefix}_aug_summary_var", summary_var)
    summary_label = tk.Label(
        frame,
        textvariable=summary_var,
        justify=tk.LEFT,
        wraplength=700,
        anchor=tk.W,
        bg=table_colors["row_alt"],
        fg=table_colors["fg"],
        padx=10,
        pady=8,
        bd=0,
        highlightthickness=1,
        highlightbackground=table_colors["border"],
        highlightcolor=table_colors["border"],
    )
    summary_label.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(0, 8))
    summary_label.bind(
        "<Configure>",
        lambda event, label=summary_label: (
            label.configure(wraplength=max(360, int(event.width) - 24))
            if int(float(label.cget("wraplength") or 0)) != max(360, int(event.width) - 24)
            else None
        ),
    )
    configure_aug_button = ttk.Button(
        plan_frame,
        text="Skonfiguruj powiększenie",
        command=lambda t=normalized_target: self._open_step4_augmentation_modal(t),
    )
    if normalized_target == "char":
        tk.Label(
            plan_frame,
            text="3. Efekty syntetyków",
            bg=table_colors["panel"],
            fg=table_colors["accent"],
            font=("Segoe UI Semibold", 9),
            anchor=tk.W,
        ).grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=(6, 2))
        configure_aug_button.grid(row=7, column=0, sticky=tk.W, pady=(0, 2))
        ttk.Label(
            plan_frame,
            text="Efekty zostaną użyte tylko dla syntetycznych obrazów train.",
            foreground=table_colors["muted"],
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=7, column=1, columnspan=3, sticky=tk.W, padx=(10, 0), pady=(0, 2))
        configure_aug_button.configure(text="Edytuj efekty syntetyków")
    else:
        configure_aug_button.grid(row=0, column=3, sticky=tk.W, padx=(12, 0), pady=2)
        configure_aug_button.configure(text="Edytuj efekty bazowe")
    try:
        initial_extra_count = int(float(extra_var.get() or 0))
    except Exception:
        initial_extra_count = 0
    configure_aug_button.configure(state=(tk.NORMAL if initial_extra_count > 0 else tk.DISABLED))
    setattr(self, f"{prefix}_aug_configure_button", configure_aug_button)
    frame.columnconfigure(0, weight=1)
    try:
        self._refresh_step4_augmentation_summary(normalized_target)
    except Exception:
        pass
    try:
        HELP.bind_help(frame, "tr_split_ratios")
    except Exception:
        pass
    return


def _build_creator_ui(self):
    _ensure_step4_pz1_section_styles(self)
    f = self.ds_creator_frame
    palette = getattr(self.app, "palette", {})
    self.creator_intro_lbl = ttk.Label(
        f,
        text=(
            "Sprawdź źródło, ustaw split i utwórz wariant do treningu modelu tablic."
        ),
        font=("Segoe UI", 9),
        wraplength=720,
        justify=tk.LEFT,
    )
    self.creator_intro_lbl.pack(anchor=tk.W, pady=(0, 10))
    self.creator_intro_lbl.configure(
        text=(
            "Sprawdź źródło, ustaw split i utwórz wariant do treningu modelu tablic. "
            "Augmentacja powiększa tylko część train."
        )
    )
    self.creator_flow_strip = _make_step4_flow_strip(self, f)

    summary_colors = _get_step4_table_colors(self)
    self.creator_campaign_summary_title = _make_step4_pz1_section_title(self, f, "1. Materiał projektu")
    self.creator_campaign_summary_frame = tk.Frame(
        f,
        bd=0,
        padx=1,
        pady=1,
        bg=summary_colors["border"],
        highlightthickness=0,
    )
    self.creator_campaign_summary_frame.pack(fill=tk.X, pady=(0, 12))
    self.creator_campaign_summary_grid = tk.Frame(
        self.creator_campaign_summary_frame,
        bd=0,
        highlightthickness=0,
        bg=summary_colors["border"],
    )
    self.creator_campaign_summary_grid.pack(fill=tk.X)
    self.creator_campaign_summary_grid.grid_columnconfigure(0, weight=0, minsize=150)
    self.creator_campaign_summary_grid.grid_columnconfigure(1, weight=1)
    self._creator_campaign_summary_header_widgets = []
    self._creator_campaign_summary_rows = {}
    for column, text in enumerate(("Krok", "Stan")):
        header_cell = tk.Label(
            self.creator_campaign_summary_grid,
            text=text,
            font=("Segoe UI Semibold", 8),
            padx=9,
            pady=5,
            anchor="w",
            bd=0,
            highlightthickness=1,
            bg=summary_colors["header"],
            fg=summary_colors["accent"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        header_cell.grid(row=0, column=column, sticky="nsew")
        self._creator_campaign_summary_header_widgets.append(header_cell)
    for row_index, (key, label_text) in enumerate(
        (
            ("target", "Cel bramki"),
            ("source", "Materiał projektu"),
            ("material", "Dane wejściowe"),
            ("variant", "Wariant datasetu"),
        ),
        start=1,
    ):
        label_cell = tk.Label(
            self.creator_campaign_summary_grid,
            text=label_text,
            font=("Segoe UI", 8),
            padx=9,
            pady=5,
            anchor="w",
            bd=0,
            highlightthickness=1,
            bg=(summary_colors["row"] if row_index % 2 else summary_colors["row_alt"]),
            fg=summary_colors["muted"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        label_cell.grid(row=row_index, column=0, sticky="nsew")
        if key == "material":
            value_cell = tk.Frame(
                self.creator_campaign_summary_grid,
                padx=9,
                pady=4,
                bd=0,
                highlightthickness=1,
                bg=(summary_colors["row"] if row_index % 2 else summary_colors["row_alt"]),
                highlightbackground=summary_colors["border"],
                highlightcolor=summary_colors["border"],
            )
        else:
            value_cell = tk.Label(
                self.creator_campaign_summary_grid,
                text="-",
                font=("Segoe UI", 8),
                padx=9,
                pady=5,
                anchor="w",
                justify=tk.LEFT,
                bd=0,
                highlightthickness=1,
                bg=(summary_colors["row"] if row_index % 2 else summary_colors["row_alt"]),
                fg=summary_colors["fg"],
                highlightbackground=summary_colors["border"],
                highlightcolor=summary_colors["border"],
            )
        value_cell.grid(row=row_index, column=1, sticky="nsew")
        self._creator_campaign_summary_rows[key] = (label_cell, value_cell)

    self.creator_source_mode_var = tk.StringVar(value="xml")
    self.creator_source_mode_frame = ttk.LabelFrame(f, text=" Źródło datasetu ", padding=8)
    ttk.Label(
        self.creator_source_mode_frame,
        text="",
        justify=tk.LEFT,
        wraplength=700,
    )
    self.creator_source_ready_radio = ttk.Radiobutton(
        self.creator_source_mode_frame,
        text="Użyj gotowego splitu tablic",
        value="ready",
        variable=self.creator_source_mode_var,
        command=self._on_creator_source_mode_change,
    )
    self.creator_source_xml_radio = ttk.Radiobutton(
        self.creator_source_mode_frame,
        text="Utwórz split z anotacji XML i zdjęć",
        value="xml",
        variable=self.creator_source_mode_var,
        command=self._on_creator_source_mode_change,
    )

    self.creator_ready_dataset_lbl = ttk.Label(
        f,
        text="",
        justify=tk.LEFT,
        wraplength=720
    )

    row1 = ttk.Frame(f); row1.pack(fill=tk.X, pady=2)
    self.creator_xml_row = row1
    ttk.Label(row1, text="Anotacje tablic XML:").pack(side=tk.LEFT)
    self.cvat_xml_var = tk.StringVar()
    self.cvat_xml_entry = ttk.Entry(row1, textvariable=self.cvat_xml_var)
    self.cvat_xml_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    self.btn_pick_cvat_xml = ttk.Button(
        row1,
        text="Wybierz plik",
        command=lambda: self._pick_file(
            self.cvat_xml_var,
            "*.xml",
            initialdir=self._get_plate_xml_picker_dir(),
        ),
    )
    self.btn_pick_cvat_xml.pack(side=tk.LEFT)

    self.creator_auto_match_hint_lbl = ttk.Label(
        f,
        text=(
            "Po wyborze XML program spróbuje dopasować katalog zdjęć po nazwach plików."
        ),
        justify=tk.LEFT,
        wraplength=720,
    )

    row2 = ttk.Frame(f); row2.pack(fill=tk.X, pady=2)
    self.creator_images_row = row2
    ttk.Label(row2, text="Zdjęcia zgodne z XML:").pack(side=tk.LEFT)
    self.cvat_images_var = tk.StringVar()
    self.cvat_images_entry = ttk.Entry(row2, textvariable=self.cvat_images_var)
    self.cvat_images_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    self.btn_pick_cvat_images = ttk.Button(
        row2,
        text="Wybierz katalog",
        command=lambda: self._pick_dir(
            self.cvat_images_var,
            initialdir=self._get_plate_images_picker_dir(),
        ),
    )
    self.btn_pick_cvat_images.pack(side=tk.LEFT)

    self.creator_source_summary_lbl = ttk.Label(
        f,
        text="",
        justify=tk.LEFT,
        wraplength=720
    )

    row3 = ttk.Frame(f); row3.pack(fill=tk.X, pady=2)
    self.creator_output_row = row3
    ttk.Label(row3, text="Folder wariantu:").pack(side=tk.LEFT)
    # Ścieżka docelowa jest wyliczana automatycznie i pozostaje tylko do odczytu.
    self.ds_out_var = tk.StringVar(value=str(self._get_datasets_base_dir() / "Plates_CVAT_[DATA_I_CZAS]"))
    ttk.Entry(row3, textvariable=self.ds_out_var, state="readonly", foreground="gray").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    creator_ratios = ttk.LabelFrame(
        f,
        text=" 2. Podział tworzonego wariantu ",
        padding=10,
        style="Step4PZ1Section.TLabelframe",
    )
    self.creator_ratios_frame = creator_ratios
    creator_ratios.pack(fill=tk.X, pady=(12, 8))
    ttk.Label(
        creator_ratios,
        text="Ustal proporcje wariantu: train uczy, val kontroluje, test zostaje do oceny.",
        justify=tk.LEFT,
        wraplength=700,
    ).grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 9))
    ttk.Label(creator_ratios, text="Train %").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Scale(
        creator_ratios,
        from_=50,
        to=90,
        variable=self.train_pct,
        command=lambda e: self._update_ratio_labels(),
    ).grid(row=1, column=1, sticky=tk.EW, padx=8, pady=2)
    self.creator_train_lbl = ttk.Label(creator_ratios, text="80%")
    self.creator_train_lbl.grid(row=1, column=2, sticky=tk.W, pady=2)

    ttk.Label(creator_ratios, text="Val %").grid(row=2, column=0, sticky=tk.W, pady=2)
    ttk.Scale(
        creator_ratios,
        from_=5,
        to=50,
        variable=self.val_pct,
        command=lambda e: self._update_ratio_labels(),
    ).grid(row=2, column=1, sticky=tk.EW, padx=8, pady=2)
    self.creator_val_lbl = ttk.Label(creator_ratios, text="10%")
    self.creator_val_lbl.grid(row=2, column=2, sticky=tk.W, pady=2)

    ttk.Label(creator_ratios, text="Test %").grid(row=3, column=0, sticky=tk.W, pady=2)
    ttk.Label(creator_ratios, text="liczony automatycznie").grid(row=3, column=1, sticky=tk.W, padx=8, pady=2)
    self.creator_test_lbl = ttk.Label(creator_ratios, text="Test: 10%")
    self.creator_test_lbl.grid(row=3, column=2, sticky=tk.W, pady=2)
    self.creator_split_bar = tk.Canvas(
        creator_ratios,
        height=26,
        bd=0,
        highlightthickness=0,
        bg=summary_colors["panel"],
    )
    self.creator_split_bar.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(11, 3))
    self.creator_split_bar.bind("<Configure>", lambda _event: self._update_ratio_labels(), add="+")
    self.creator_split_bar_legend = ttk.Label(
        creator_ratios,
        text="Podzia\u0142 zostanie zapisany w tworzonym wariancie datasetu.",
        justify=tk.LEFT,
        wraplength=700,
    )
    self.creator_split_bar_legend.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(1, 0))
    creator_ratios.columnconfigure(1, weight=1)
    try:
        self._update_ratio_labels()
    except Exception:
        pass

    self._build_step4_augmentation_controls(f, target="plate")

    create_shell, create_inner, create_bg = _make_step4_action_shell(
        self,
        f,
        title="4. Utwórz wariant datasetu tablic",
        description=(
            "Zapisuje wariant gotowy do treningu modelu tablic."
        ),
    )
    self.btn_step4_create_frame = create_shell
    self.step4_creator_action_inner = create_inner
    self.btn_step4_create_frame.pack(fill=tk.X, pady=(12, 10))
    self.creator_decision_summary_var = tk.StringVar(
        value="Train 80% | Val 10% | Test 10% | bez syntetycznego powi\u0119kszenia"
    )
    _make_step4_decision_table(self, create_inner, prefix="creator", bg=create_bg)
    create_button_row = tk.Frame(create_inner, bg=create_bg, bd=0, highlightthickness=0)
    create_button_row.pack(anchor=tk.W, fill=tk.X)

    self.btn_step4_create = ttk.Button(
        create_button_row,
        text="Utwórz split treningowy",
        command=self._create_dataset_thread,
        style="Accent.TButton",
    )
    self.btn_step4_create.pack(anchor=tk.W, fill=tk.X, ipady=2)
    self.btn_step4_create.configure(text="Utw\u00f3rz wariant datasetu tablic")
    try:
        self._refresh_step4_creator_decision_summary()
    except Exception:
        pass

    self.ds_progress_var = tk.DoubleVar(value=0.0)
    self.ds_progress = TrainProgressBar(
        create_inner,
        variable=self.ds_progress_var,
        maximum=100,
        thickness=6,
        trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
        fill_color=palette.get("success", "#2ecc71"),
        bg=palette.get("panel", "#252526"),
        height=10,
    )
    self.ds_progress.pack(fill=tk.X, pady=(10, 2))

    self.ds_status = ttk.Label(create_inner, text="Gotowy", style="TrainSplitSuccess.TLabel")
    self.ds_status.pack(anchor=tk.W)
    self._style_training_success_label(self.ds_status)

    try:
        self.cvat_xml_var.trace_add("write", lambda *_args: self._on_cvat_xml_source_changed())
        self.cvat_images_var.trace_add("write", lambda *_args: self._on_cvat_images_source_changed())
    except Exception:
        pass
    self._refresh_dataset_creator_cta_state()

    # Powiązania pomocy dla budowy datasetu z CVAT.
    HELP.bind_help(row1, "tr_cvat_xml")
    HELP.bind_help(self.creator_source_mode_frame, "tr_cvat_source_mode")
    HELP.bind_help(self.creator_source_ready_radio, "tr_cvat_source_mode")
    HELP.bind_help(self.creator_source_xml_radio, "tr_cvat_source_mode")
    HELP.bind_help(self.btn_pick_cvat_xml, "tr_cvat_xml")
    HELP.bind_help(self.creator_auto_match_hint_lbl, "tr_cvat_img")
    HELP.bind_help(row2, "tr_cvat_img")
    HELP.bind_help(self.btn_pick_cvat_images, "tr_cvat_img")
    HELP.bind_help(self.creator_source_summary_lbl, "tr_cvat_source_mode")
    HELP.bind_help(creator_ratios, "tr_split_ratios")
    HELP.bind_help(self.btn_step4_create, "tr_cvat_btn")

def _build_splitter_ui(self):
    _ensure_step4_pz1_section_styles(self)
    f = self.ds_split_frame
    palette = getattr(self.app, "palette", {})
    self.split_intro_lbl = ttk.Label(
        f,
        text=(
            "Wskaż źródło, ustaw split i utwórz wariant do treningu modelu znaków."
        ),
        font=("Segoe UI", 9),
        wraplength=720,
        justify=tk.LEFT,
    )
    self.split_intro_lbl.pack(anchor=tk.W, pady=(0, 10))

    self._step4_char_split_details_visible = False
    self.btn_step4_split_toggle = ttk.Button(
        f,
        text="Popraw split",
        command=self._toggle_step4_char_split_details,
        style="WorkflowCard.TButton"
    )

    self.split_source_hint_lbl = ttk.Label(
        f,
        text="",
        justify=tk.LEFT,
        wraplength=720,
    )

    row1 = ttk.Frame(f); row1.pack(fill=tk.X, pady=2)
    self.split_source_row = row1
    ttk.Label(row1, text="Dataset znaków:").pack(side=tk.LEFT)
    self.split_src_var = tk.StringVar()
    self.split_src_entry = ttk.Entry(row1, textvariable=self.split_src_var)
    self.split_src_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    self.btn_pick_split_src = ttk.Button(
        row1,
        text="Wybierz katalog",
        command=self._pick_char_split_source_dir,
    )
    self.btn_pick_split_src.pack(side=tk.LEFT)
    self.btn_pick_split_yaml = ttk.Button(
        row1,
        text="Wybierz data.yaml",
        command=self._pick_char_split_source_yaml,
    )
    self.btn_pick_split_yaml.pack(side=tk.LEFT, padx=(6, 0))

    self.split_source_summary_lbl = ttk.Label(
        f,
        text="",
        justify=tk.LEFT,
        wraplength=720
    )
    summary_colors = _get_step4_table_colors(self)
    self.split_campaign_summary_title = _make_step4_pz1_section_title(self, f, "1. Źródło i wariant treningowy")
    self.split_campaign_summary_frame = tk.Frame(
        f,
        bd=0,
        padx=1,
        pady=1,
        bg=summary_colors["border"],
        highlightthickness=0,
    )
    self.split_campaign_summary_grid = tk.Frame(
        self.split_campaign_summary_frame,
        bd=0,
        highlightthickness=0,
        bg=summary_colors["border"],
    )
    self.split_campaign_summary_grid.pack(fill=tk.X)
    self.split_campaign_summary_grid.grid_columnconfigure(0, weight=0, minsize=132)
    self.split_campaign_summary_grid.grid_columnconfigure(1, weight=1, minsize=180)
    self.split_campaign_summary_grid.grid_columnconfigure(2, weight=2, minsize=260)
    self._split_campaign_summary_header_widgets = []
    self._split_campaign_summary_rows = {}
    for column, text in enumerate(("Element", "Identyfikator", "Szczegóły")):
        header_cell = tk.Label(
            self.split_campaign_summary_grid,
            text=text,
            font=("Segoe UI Semibold", 8),
            padx=9,
            pady=5,
            anchor="w",
            bd=0,
            highlightthickness=1,
            bg=summary_colors["header"],
            fg=summary_colors["accent"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        header_cell.grid(row=0, column=column, sticky="nsew")
        self._split_campaign_summary_header_widgets.append(header_cell)
    for row_index, (key, label_text) in enumerate(
        (
            ("source", "Źródłowy dataset"),
            ("variant", "Wariant treningowy"),
            ("split", "Podział wariantu"),
        ),
        start=1,
    ):
        row_bg = summary_colors["row"] if row_index % 2 else summary_colors["row_alt"]
        label_cell = tk.Label(
            self.split_campaign_summary_grid,
            text=label_text,
            font=("Segoe UI", 8),
            padx=9,
            pady=5,
            anchor="w",
            bd=0,
            highlightthickness=1,
            bg=row_bg,
            fg=summary_colors["muted"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        label_cell.grid(row=row_index, column=0, sticky="nsew")
        id_cell = tk.Label(
            self.split_campaign_summary_grid,
            text="-",
            font=("Segoe UI", 8),
            padx=9,
            pady=5,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=1,
            bg=row_bg,
            fg=summary_colors["fg"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        id_cell.grid(row=row_index, column=1, sticky="nsew")
        details_cell = tk.Label(
            self.split_campaign_summary_grid,
            text="-",
            font=("Segoe UI", 8),
            padx=9,
            pady=5,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=1,
            bg=row_bg,
            fg=summary_colors["fg"],
            highlightbackground=summary_colors["border"],
            highlightcolor=summary_colors["border"],
        )
        details_cell.grid(row=row_index, column=2, sticky="nsew")
        self._split_campaign_summary_rows[key] = (label_cell, id_cell, details_cell)

    row2 = ttk.Frame(f); row2.pack(fill=tk.X, pady=2)
    self.split_output_row = row2
    ttk.Label(row2, text="Folder wariantu:").pack(side=tk.LEFT)
    # Ścieżka wariantu splitu jest wyliczana automatycznie i pozostaje tylko do odczytu.
    self.split_out_var = tk.StringVar(value=str(self._get_datasets_base_dir() / "[NAZWA_ZRODLA]_Split_[DATA_I_CZAS]"))
    ttk.Entry(row2, textvariable=self.split_out_var, state="readonly", foreground="gray").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    ratios = ttk.LabelFrame(
        f,
        text=" 2. Podział wariantu train / val / test ",
        padding=10,
        style="Step4PZ1Section.TLabelframe",
    )
    ratios.pack(fill=tk.X, pady=(12, 8))
    self.split_ratios_frame = ratios
    ttk.Label(ratios, text="Train %").grid(row=0, column=0, sticky=tk.W)
    ttk.Scale(ratios, from_=50, to=90, variable=self.train_pct, command=lambda e: self._update_ratio_labels()).grid(row=0, column=1, sticky=tk.EW, padx=5)
    self.train_lbl = ttk.Label(ratios, text="80%"); self.train_lbl.grid(row=0, column=2, sticky=tk.W)

    ttk.Label(ratios, text="Val %").grid(row=1, column=0, sticky=tk.W)
    ttk.Scale(ratios, from_=5, to=50, variable=self.val_pct, command=lambda e: self._update_ratio_labels()).grid(row=1, column=1, sticky=tk.EW, padx=5)
    self.val_lbl = ttk.Label(ratios, text="10%"); self.val_lbl.grid(row=1, column=2, sticky=tk.W)

    ttk.Label(ratios, text="Test %").grid(row=2, column=0, sticky=tk.W)
    ttk.Label(ratios, text="liczony automatycznie").grid(row=2, column=1, sticky=tk.W, padx=5)
    self.test_lbl = ttk.Label(ratios, text="Test: 10%"); self.test_lbl.grid(row=2, column=2, sticky=tk.W)
    ratios.columnconfigure(1, weight=1)

    self._build_step4_augmentation_controls(f, target="char")

    split_shell, split_inner, split_bg = _make_step4_action_shell(
        self,
        f,
        title="4. Utwórz wariant treningowy znaków",
        description=(
            "Zapisuje wariant train / val / test, który PZ2 wykorzysta do treningu modelu znaków."
        ),
    )
    self.btn_step4_split_frame = split_shell
    self.step4_split_action_inner = split_inner
    self.btn_step4_split_frame.pack(fill=tk.X, pady=(12, 10))
    _make_step4_decision_table(self, split_inner, prefix="split", bg=split_bg)
    split_button_row = tk.Frame(split_inner, bg=split_bg, bd=0, highlightthickness=0)
    split_button_row.pack(anchor=tk.W, fill=tk.X)

    self.btn_step4_split = ttk.Button(
        split_button_row,
        text="Utwórz split treningowy",
        command=self._split_dataset_thread,
        style="Accent.TButton"
    )
    self.btn_step4_split.pack()

    self.split_progress_var = tk.DoubleVar(value=0.0)
    self.split_feedback_frame = tk.Frame(split_inner, bg=split_bg, bd=0, highlightthickness=0)
    self.split_progress = TrainProgressBar(
        self.split_feedback_frame,
        variable=self.split_progress_var,
        maximum=100,
        thickness=6,
        trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
        fill_color=palette.get("success", "#2ecc71"),
        bg=palette.get("panel", "#252526"),
        height=10,
    )
    self.split_progress.pack(fill=tk.X, pady=2)
    self.split_status = ttk.Label(self.split_feedback_frame, text="Gotowy")
    self.split_status.pack(anchor=tk.W)
    self._set_split_feedback_visibility(False)
    def _on_split_source_change(*_args):
        try:
            self._refresh_dataset_split_cta_state()
        except Exception:
            pass
        try:
            self._refresh_step4_creator_decision_summary()
        except Exception:
            pass
        try:
            self._invalidate_pending_character_balance_plan(
                "Źródło zmieniło się - przelicz reprezentację MZ."
            )
        except Exception:
            pass

    try:
        self.split_src_var.trace_add("write", _on_split_source_change)
    except Exception:
        pass
    self._refresh_dataset_split_cta_state()

    # Powiązania pomocy dla splitu datasetu.
    HELP.bind_help(row1, "tr_split_src")
    HELP.bind_help(self.split_intro_lbl, "tr_split_src")
    HELP.bind_help(self.split_source_hint_lbl, "tr_split_src")
    HELP.bind_help(self.btn_pick_split_src, "tr_split_src")
    HELP.bind_help(self.btn_pick_split_yaml, "tr_split_src")
    HELP.bind_help(self.split_source_summary_lbl, "tr_split_src")
    HELP.bind_help(ratios, "tr_split_ratios")
    HELP.bind_help(self.btn_step4_split_toggle, "tr_split_ratios")
    HELP.bind_help(self.btn_step4_split, "tr_split_btn")
    self._update_ratio_labels()
    self._configure_train_progress_styles()
