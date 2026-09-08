#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 PZ2 training tab builder extracted from tab_training.py."""

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
from . import z4_dataset_tab_builder
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _build_train_tab(self):
    try:
        self._ensure_step4_training_state_vars()
    except Exception:
        pass
    palette = getattr(self.app, "palette", {})
    root = ttk.Frame(self.tab_train, padding=5)
    root.pack(fill=tk.BOTH, expand=True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    self.train_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    self.train_pane.grid(row=0, column=0, sticky="nsew")

    self.left = ttk.LabelFrame(self.train_pane, text=" Trening ", padding=9)
    self.right = ttk.LabelFrame(self.train_pane, text=" Wyniki i narzędzia ", padding=8)
    self.train_pane.add(self.left, weight=5)
    self.train_pane.add(self.right, weight=9)

    self.left.grid_rowconfigure(0, weight=1)
    self.left.grid_columnconfigure(0, weight=1)

    self.train_left_scroll_host = ttk.Frame(self.left, style="Panel.TFrame")
    self.train_left_scroll_host.grid(row=0, column=0, sticky="nsew")
    self.train_left_scroll_host.grid_rowconfigure(0, weight=1)
    self.train_left_scroll_host.grid_columnconfigure(0, weight=1)

    self.train_left_canvas = tk.Canvas(
        self.train_left_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0
    )
    self.train_left_canvas.grid(row=0, column=0, sticky="nsew")

    self.train_left_scrollbar = WebSlimScrollbar(
        self.train_left_scroll_host,
        command=self.train_left_canvas.yview
    )
    self.train_left_scrollbar.grid(row=0, column=1, sticky="ns")
    self.train_left_canvas.configure(yscrollcommand=self.train_left_scrollbar.set)

    self._train_left_content_inset = 14
    self._train_left_hint_inset = 10
    self._train_left_section_gap = 16
    self._train_left_content_max_width = 660
    self.train_left_content = ttk.Frame(self.train_left_canvas, style="Panel.TFrame")
    self.train_left_content.grid_columnconfigure(0, weight=1)
    self.train_left_content_window = self.train_left_canvas.create_window(
        (self._train_left_content_inset, 0),
        window=self.train_left_content,
        anchor="nw"
    )
    self.train_left_content.bind("<Configure>", self._sync_train_left_scrollregion, add="+")
    self.train_left_canvas.bind("<Configure>", self._sync_train_left_canvas_width, add="+")

    self.train_left_settings_col = ttk.Frame(self.train_left_content, style="Panel.TFrame")
    settings_col = self.train_left_settings_col
    settings_col.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

    self.free_training_route_host = SectionHeaderLabel(
        settings_col,
        self.app,
        text="Przepływ PZ1 -> PZ2",
        fade_ratio=0.74,
    )
    self.free_training_route_host.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self.free_training_route_title_lbl = self.free_training_route_host
    self.free_training_route_intro_lbl = None
    self.training_input_summary_frame = None
    self.training_input_badge_lbl = None
    self.training_input_route_lbl = None
    self.training_input_dataset_lbl = None
    self.training_input_source_lbl = None
    self.training_input_change_btn = None
    self.free_training_route_cards_row = None
    self._free_training_route_cards = {}

    self.name_var = getattr(self, "name_var", tk.StringVar())

    cockpit_bg = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.88,
    )
    cockpit_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.45,
    )
    cockpit_card_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
    cockpit_muted = palette.get("muted", "#c7c7c7")

    self.train_cockpit_shell = tk.Frame(
        settings_col,
        bg=cockpit_bg,
        highlightthickness=1,
        highlightbackground=cockpit_border,
        highlightcolor=cockpit_border,
        bd=0,
    )
    self.train_cockpit_shell.pack(fill=tk.X, pady=(0, self._train_left_section_gap))

    self.train_cockpit_header = tk.Frame(
        self.train_cockpit_shell,
        bg=cockpit_bg,
        bd=0,
        highlightthickness=0,
    )
    self.train_cockpit_header.pack(fill=tk.X, padx=10, pady=(8, 5))
    self.train_cockpit_header.grid_columnconfigure(0, weight=1)

    self.train_cockpit_title_lbl = tk.Label(
        self.train_cockpit_header,
        text="Kokpit treningu",
        font=("Segoe UI", 12),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=cockpit_bg,
        fg=palette.get("fg", "#f3f3f3"),
    )
    self.train_cockpit_title_lbl.grid(row=0, column=0, sticky="ew")

    self.train_cockpit_status_lbl = tk.Label(
        self.train_cockpit_header,
        text="Czekam na wariant",
        font=("Segoe UI Semibold", 8),
        anchor=tk.CENTER,
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        padx=8,
        pady=3,
        bg=palette.get("surface_warning", "#3a3425"),
        fg=palette.get("warning", "#f0b44c"),
    )
    self.train_cockpit_status_lbl.grid(row=0, column=1, sticky="e", padx=(8, 0))

    self.train_cockpit_subtitle_lbl = tk.Label(
        self.train_cockpit_shell,
        text="Wybierz wariant splitu i model startowy treningu, aby rozpocząć nowy run.",
        font=("Segoe UI", 9),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=cockpit_bg,
        fg=cockpit_muted,
        wraplength=380,
    )
    self.train_cockpit_subtitle_lbl.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 8))
    self._register_train_left_wrap_target(self.train_cockpit_subtitle_lbl, padding=24, min_wrap=220)

    self.train_cockpit_grid = tk.Frame(
        self.train_cockpit_shell,
        bg=cockpit_bg,
        bd=0,
        highlightthickness=0,
    )
    self.train_cockpit_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
    self.train_cockpit_grid.grid_columnconfigure(0, weight=1)
    self.train_cockpit_grid.grid_columnconfigure(1, weight=1)

    self._train_cockpit_cards = []
    for card_index, card_title in enumerate(("Dataset", "Próbka", "Model", "Sprzęt")):
        card = tk.Frame(
            self.train_cockpit_grid,
            bg=cockpit_card_bg,
            highlightthickness=1,
            highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
            highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
            bd=0,
        )
        card.grid(
            row=card_index // 2,
            column=card_index % 2,
            sticky="nsew",
            padx=(0, 6) if card_index % 2 == 0 else (0, 0),
            pady=(0, 6) if card_index < 2 else (0, 0),
        )
        title_lbl = tk.Label(
            card,
            text=card_title,
            font=("Segoe UI Semibold", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            bg=cockpit_card_bg,
            fg=cockpit_muted,
            padx=8,
            pady=4,
        )
        title_lbl.pack(anchor=tk.W, fill=tk.X)
        value_lbl = tk.Label(
            card,
            text="-",
            font=("Segoe UI Semibold", 10),
            anchor=tk.W,
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            bg=cockpit_card_bg,
            fg=palette.get("fg", "#f3f3f3"),
            padx=8,
            pady=1,
            wraplength=210,
        )
        value_lbl.pack(anchor=tk.W, fill=tk.X)
        hint_lbl = tk.Label(
            card,
            text="",
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            bg=cockpit_card_bg,
            fg=cockpit_muted,
            padx=8,
            pady=4,
            wraplength=210,
        )
        hint_lbl.pack(anchor=tk.W, fill=tk.X)
        self._register_train_left_wrap_target(value_lbl, container=card, padding=18, min_wrap=140)
        self._register_train_left_wrap_target(hint_lbl, container=card, padding=18, min_wrap=140)
        self._train_cockpit_cards.append(
            {
                "frame": card,
                "title": title_lbl,
                "value": value_lbl,
                "hint": hint_lbl,
            }
        )

    section_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.72,
    )

    try:
        dataset_section_style = ttk.Style()
        dataset_section_style.configure(
            "TrainDatasetSection.TLabelframe",
            background=palette.get("panel", "#252526"),
            bordercolor=section_border,
            lightcolor=section_border,
            darkcolor=section_border,
            relief=tk.SOLID,
        )
        dataset_section_style.configure(
            "TrainDatasetSection.TLabelframe.Label",
            font=("Segoe UI", 12, "bold"),
            foreground=palette.get("fg", "#f3f3f3"),
            background=palette.get("panel", "#252526"),
        )
    except Exception:
        pass

    self.train_dataset_section_frame = ttk.LabelFrame(
        settings_col,
        text=" Dataset treningowy ",
        style="TrainDatasetSection.TLabelframe",
        padding=9,
    )
    self.train_dataset_section_frame.pack(fill=tk.X, pady=(0, self._train_left_section_gap))

    self.train_dataset_title_lbl = tk.Label(
        self.train_dataset_section_frame,
        text="Wybierz wariant splitu",
        font=("Segoe UI", 12),
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.train_dataset_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
    self.train_dataset_intro_lbl = ttk.Label(
        self.train_dataset_section_frame,
        text="Trening ruszy na wariancie datasetu przygotowanym w PZ1.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360,
    )
    self.train_dataset_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._register_train_left_wrap_target(
        self.train_dataset_intro_lbl,
        padding=18,
        min_wrap=220,
    )
    self.dataset_var = getattr(self, "dataset_var", tk.StringVar())
    # PZ2 przechowuje dataset_var wewnetrznie. Uzytkownik wybiera juz tylko
    # warianty splitu przygotowane w PZ1, bez recznego podmieniania zrodel.

    variant_card_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    self.dataset_variant_row = tk.Frame(
        self.train_dataset_section_frame,
        bg=variant_card_bg,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        bd=0,
    )
    self.dataset_variant_row.pack(fill=tk.X, pady=(6, 10))
    self.dataset_variant_title_lbl = tk.Label(
        self.dataset_variant_row,
        text="Aktywny wariant",
        font=("Segoe UI Semibold", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        padx=8,
        pady=5,
        bg=variant_card_bg,
        fg=palette.get("muted", "#c7c7c7"),
    )
    self.dataset_variant_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.dataset_variant_var = getattr(self, "dataset_variant_var", tk.StringVar())
    self.dataset_variant_combo = ttk.Combobox(
        self.dataset_variant_row,
        textvariable=self.dataset_variant_var,
        state="readonly",
        values=[],
    )
    self.dataset_variant_combo.pack(fill=tk.X, padx=8, pady=(0, 6))
    self.dataset_variant_combo.bind("<<ComboboxSelected>>", self._on_dataset_variant_selected)
    try:
        self._refresh_dataset_variant_choices()
    except Exception:
        pass
    self.train_dataset_selected_path_lbl = None
    self.train_dataset_hint_lbl = ttk.Label(
        self.train_dataset_section_frame,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=320
    )
    self._register_train_left_wrap_target(
        self.train_dataset_hint_lbl,
        padding=16,
        min_wrap=220,
    )
    self.train_dataset_hint_lbl.bind("<Configure>", self._update_training_dataset_hint_wraplength, add="+")

    self.train_dataset_quality_shell = tk.Frame(
        self.train_dataset_section_frame,
        bd=0,
        highlightthickness=1,
    )
    self.train_dataset_quality_shell.pack(anchor=tk.W, fill=tk.X, pady=(4, 10))
    self.train_dataset_quality_grid = tk.Frame(
        self.train_dataset_quality_shell,
        bd=0,
        highlightthickness=0,
    )
    self.train_dataset_quality_grid.pack(fill=tk.X, padx=1, pady=1)
    self.train_dataset_quality_grid.grid_columnconfigure(0, weight=0, minsize=96)
    self.train_dataset_quality_grid.grid_columnconfigure(1, weight=1)
    self.train_dataset_quality_title_row = tk.Frame(
        self.train_dataset_quality_grid,
        bd=0,
        highlightthickness=0,
    )
    self.train_dataset_quality_title_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 1), pady=(0, 1))
    self.train_dataset_quality_title_lbl = tk.Label(
        self.train_dataset_quality_title_row,
        text="Podsumowanie materiału",
        font=("Segoe UI", 9, "bold"),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=6,
        pady=5,
    )
    self.train_dataset_quality_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self._train_dataset_quality_row_widgets = []

    self.btn_character_class_distribution = ttk.Button(
        self.train_dataset_section_frame,
        text="Pokaż reprezentację MZ",
        command=self._open_character_class_distribution_dialog,
        style="WorkflowCard.TButton",
    )
    self._character_class_distribution_button_visible = False

    self.train_scope_hint_lbl = ttk.Label(
        settings_col,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self.train_scope_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, self._train_left_section_gap))
    self._register_train_left_wrap_target(self.train_scope_hint_lbl, padding=16, min_wrap=220)

    self.train_pose_warning_lbl = ttk.Label(
        settings_col,
        text="",
        style="PanelStatusWarning.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self._register_train_left_wrap_target(self.train_pose_warning_lbl, padding=16, min_wrap=220)

    self._update_training_dataset_hint()
    self._refresh_free_training_route_ui()

    self._build_train_left_separator(settings_col, pady=(0, self._train_left_section_gap))

    self.base_model_var = getattr(self, "base_model_var", tk.StringVar())

    base_values = (
        list(AVAILABLE_DETECT_MODELS.keys())
        + list(AVAILABLE_POSE_MODELS.keys())
        + [self._get_custom_base_model_label()]
    )

    self.train_base_shell = tk.Frame(
        settings_col,
        bd=0,
        highlightthickness=1,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        highlightbackground=section_border,
        highlightcolor=section_border,
    )
    self.train_base_shell.pack(fill=tk.X, pady=(4, 2))

    self.train_base_title_lbl = tk.Label(
        self.train_base_shell,
        text="Model startowy treningu",
        font=("Segoe UI Semibold", 12),
        anchor="w",
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        fg=palette.get("fg", "#f3f3f3"),
    )
    self.train_base_title_lbl.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(9, 4))
    self.train_base_context_row = tk.Frame(
        self.train_base_shell,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        bd=0,
        highlightthickness=0,
    )
    self.train_base_context_row.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 6))
    self.train_base_context_chips = {}
    chip_bg = blend_hex_colors(
        palette.get("accent", "#0e639c"),
        palette.get("panel_alt", palette.get("panel", "#252526")),
        0.84,
    )
    chip_fg = palette.get("fg", "#f3f3f3")
    for chip_key, chip_text in (
        ("iteration", "IT"),
        ("target", "Model"),
        ("backend", "YOLO"),
    ):
        chip_lbl = tk.Label(
            self.train_base_context_row,
            text=chip_text,
            font=("Segoe UI Semibold", 8),
            anchor=tk.CENTER,
            justify=tk.CENTER,
            bd=0,
            highlightthickness=1,
            padx=7,
            pady=2,
            bg=chip_bg,
            fg=chip_fg,
            highlightbackground=section_border,
            highlightcolor=section_border,
        )
        chip_lbl.pack(side=tk.LEFT, padx=(0, 6))
        self.train_base_context_chips[chip_key] = chip_lbl
    self.train_base_status_lbl = tk.Label(
        self.train_base_shell,
        text="PUNKT STARTU",
        font=("Segoe UI Semibold", 8),
        anchor=tk.CENTER,
        justify=tk.CENTER,
        bd=0,
        highlightthickness=1,
        padx=8,
        pady=3,
        bg=palette.get("panel", "#252526"),
        fg=palette.get("muted", "#c7c7c7"),
        highlightbackground=section_border,
        highlightcolor=section_border,
    )
    self.train_base_status_lbl.pack(anchor=tk.W, padx=10, pady=(0, 5))
    self.train_base_caption_lbl = ttk.Label(
        self.train_base_shell,
        text=(
            "To wagi startowe nowego runu. Ukończone modele projektu znajdziesz obok "
            "w Historii treningów; tam wybierzesz wynik bramki albo punkt dotrenowania."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self.train_base_caption_lbl.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 7))
    self._register_train_left_wrap_target(self.train_base_caption_lbl, container=self.train_base_shell, padding=24, min_wrap=220)

    self.train_base_combo_lbl = ttk.Label(
        self.train_base_shell,
        text="Model startowy",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
    )
    self.train_base_combo_lbl.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 3))
    self.base_combo = ttk.Combobox(self.train_base_shell, textvariable=self.base_model_var, values=base_values, state="readonly")
    self.base_combo.pack(fill=tk.X, padx=10, pady=(0, 7))
    if not str(self.base_model_var.get() or "").strip():
        self.base_model_var.set("yolo11n.pt")
    self.base_combo.bind("<<ComboboxSelected>>", lambda e: self._on_base_model_change())

    # Pozwól wskazać własne wagi startowe do fine-tuningu z katalogu modeli.
    self.base_custom_var = getattr(self, "base_custom_var", tk.StringVar())
    self.custom_row = ttk.Frame(self.train_base_shell, style="Panel.TFrame")
    self.custom_row.pack(fill=tk.X, padx=10, pady=(0, 9))

    self.base_custom_lbl = ttk.Label(
        self.custom_row,
        text="Własny .pt:",
        style="PanelMuted.TLabel",
    )
    self.base_custom_lbl.pack(side=tk.LEFT, padx=(0, 6))
    self.base_custom_lbl.configure(text="Plik .pt:")

    self.base_custom_entry = ttk.Entry(self.custom_row, textvariable=self.base_custom_var, state=tk.DISABLED)
    self.base_custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    self.base_custom_btn = ttk.Button(
        self.custom_row,
        text="Wybierz .pt",
        state=tk.DISABLED,
        command=self._pick_base_custom_model
    )
    self.base_custom_btn.pack(side=tk.LEFT, padx=(8, 0))
    self.base_custom_btn.configure(text="Wskaż plik .pt")

    self.train_base_summary_shell = tk.Frame(
        self.train_base_shell,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=1,
        highlightbackground=section_border,
        highlightcolor=section_border,
    )
    self.train_base_summary_shell.pack(fill=tk.X, padx=10, pady=(0, 8))
    self.train_base_summary_shell.grid_columnconfigure(1, weight=1)
    self.train_base_summary_values = {}
    self.train_base_summary_cells = []
    for row_index, (row_key, row_label) in enumerate((
        ("selected", "Wybrano model"),
        ("origin", "Pochodzenie"),
        ("state", "Stan"),
    )):
        row_bg = (
            palette.get("panel", "#252526")
            if row_index % 2 == 0
            else palette.get("panel_alt", palette.get("panel", "#252526"))
        )
        key_lbl = tk.Label(
            self.train_base_summary_shell,
            text=row_label,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bg=row_bg,
            fg=palette.get("muted", "#c7c7c7"),
            padx=8,
            pady=4,
            bd=0,
        )
        key_lbl.grid(row=row_index, column=0, sticky="nsew")
        self.train_base_summary_cells.append((row_index, key_lbl, "key"))
        value_lbl = tk.Label(
            self.train_base_summary_shell,
            text="-",
            font=("Segoe UI Semibold", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bg=row_bg,
            fg=palette.get("fg", "#f3f3f3"),
            padx=8,
            pady=4,
            bd=0,
            wraplength=320,
        )
        value_lbl.grid(row=row_index, column=1, sticky="nsew")
        self.train_base_summary_cells.append((row_index, value_lbl, "value"))
        self.train_base_summary_values[row_key] = value_lbl
        self._register_train_left_wrap_target(
            value_lbl,
            container=self.train_base_summary_shell,
            padding=96,
            min_wrap=180,
        )

    self.train_base_identity_lbl = ttk.Label(
        self.train_base_shell,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360,
    )
    self._register_train_left_wrap_target(self.train_base_identity_lbl, padding=16, min_wrap=220)

    pinned_bg = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_alt", palette.get("panel", "#252526")),
        0.88,
    )
    pinned_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.42,
    )
    self.step4_pinned_result_shell = tk.Frame(
        self.train_base_shell,
        bg=pinned_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=pinned_border,
        highlightcolor=pinned_border,
    )
    pinned_title_row = tk.Frame(self.step4_pinned_result_shell, bg=pinned_bg, bd=0, highlightthickness=0)
    pinned_title_row.pack(fill=tk.X, padx=9, pady=(8, 4))
    pinned_title_row.grid_columnconfigure(0, weight=1)
    self.step4_pinned_result_title_lbl = tk.Label(
        pinned_title_row,
        text="Konfiguracja zablokowana",
        font=("Segoe UI Semibold", 10),
        anchor=tk.W,
        justify=tk.LEFT,
        bg=pinned_bg,
        fg=palette.get("success", "#2ecc71"),
        bd=0,
        padx=0,
        pady=0,
    )
    self.step4_pinned_result_title_lbl.grid(row=0, column=0, sticky="w")
    self.step4_pinned_result_detail_lbl = tk.Label(
        self.step4_pinned_result_shell,
        text="",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=340,
        bg=pinned_bg,
        fg=palette.get("muted", "#c7c7c7"),
        bd=0,
        padx=9,
        pady=0,
    )
    self.step4_pinned_result_detail_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._register_train_left_wrap_target(
        self.step4_pinned_result_detail_lbl,
        container=self.step4_pinned_result_shell,
        padding=24,
        min_wrap=220,
    )
    self.step4_pinned_result_shell.pack_forget()

    def auto_name(*args):
        ds_name = Path(self.dataset_var.get()).name if self.dataset_var.get() else "UnknownDS"
        model_name = self.base_model_var.get()
        if self._is_custom_base_model_key(model_name):
            model_name = Path(self.base_custom_var.get()).stem if self.base_custom_var.get() else "WlasnyModel"
        import datetime
        ts = datetime.datetime.now().strftime("%d%b_%H%M")
        self.name_var.set(f"Train_{model_name}_{ds_name}_{ts}")

    self.dataset_var.trace_add("write", auto_name)
    self.base_model_var.trace_add("write", auto_name)
    self.base_custom_var.trace_add("write", auto_name)
    self.dataset_var.trace_add("write", lambda *args: self._update_training_dataset_hint())
    self.dataset_var.trace_add("write", lambda *args: self._refresh_training_start_state())
    self.dataset_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.dataset_var.trace_add("write", lambda *args: self._refresh_step4_dataset_mode_ui())
    self.dataset_var.trace_add("write", lambda *args: self._sync_dataset_variant_selection())
    self.base_model_var.trace_add("write", self._on_training_base_model_value_write)
    self.base_model_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.base_custom_var.trace_add("write", self._on_training_base_model_value_write)
    self.base_custom_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.base_model_var.trace_add("write", lambda *args: self._remember_current_step4_training_model_selection())
    self.base_custom_var.trace_add("write", lambda *args: self._remember_current_step4_training_model_selection())
    auto_name() # Inicjalizacja pierwszego wpisu
    try:
        self._refresh_training_base_model_identity_ui()
    except Exception:
        pass

    self._build_train_left_separator(settings_col, pady=(self._train_left_section_gap, self._train_left_section_gap))

    self.train_params_caption_lbl = None

    self.epochs_var = getattr(self, "epochs_var", tk.IntVar(value=100))
    self.batch_var = getattr(self, "batch_var", tk.IntVar(value=16))
    self.imgsz_var = getattr(self, "imgsz_var", tk.IntVar(value=640))
    self.lr0_var = getattr(self, "lr0_var", tk.DoubleVar(value=0.01))

    self.train_recommendation_table_shell = tk.Frame(
        settings_col,
        bd=0,
        highlightthickness=1,
        highlightbackground=section_border,
        highlightcolor=section_border,
    )
    self.train_recommendation_table_shell.pack(fill=tk.X, pady=(4, 10))
    self.train_recommendation_grid = tk.Frame(self.train_recommendation_table_shell, bd=0, highlightthickness=0)
    self.train_recommendation_grid.pack(fill=tk.X, padx=1, pady=1)
    self.train_recommendation_grid.grid_columnconfigure(0, weight=1)
    self.train_recommendation_title_row = tk.Frame(
        self.train_recommendation_grid,
        bd=0,
        highlightthickness=0,
    )
    self.train_recommendation_title_row.grid(
        row=0,
        column=0,
        columnspan=1,
        sticky="ew",
        padx=(0, 1),
        pady=(0, 1),
    )
    self.train_recommendation_title_row.grid_columnconfigure(0, weight=1)
    self.train_recommendation_title_label = tk.Label(
        self.train_recommendation_title_row,
        text="Ustawienia treningu",
        font=("Segoe UI", 12),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=10,
        pady=8,
    )
    self.train_recommendation_title_label.grid(row=0, column=0, sticky="w")
    self.btn_apply_training_recommendation = ttk.Button(
        self.train_recommendation_title_row,
        text="Zastosuj rekomendowane",
        command=self._apply_training_device_recommendation,
        width=24,
    )
    self.btn_apply_training_recommendation.grid(row=0, column=1, sticky="e", padx=(8, 10), pady=6)
    self.train_recommendation_hardware_label = tk.Label(
        self.train_recommendation_grid,
        text="Wykryty sprzęt: -",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=6,
        pady=3,
    )
    self.train_recommendation_hardware_label.grid(
        row=1,
        column=0,
        columnspan=1,
        sticky="ew",
        padx=(8, 8),
        pady=(2, 8),
    )
    self.train_recommendation_model_label = tk.Label(
        self.train_recommendation_grid,
        text="Model: -",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=6,
        pady=3,
    )
    self.train_recommendation_model_label.grid(
        row=2,
        column=0,
        columnspan=1,
        sticky="ew",
        padx=(0, 1),
        pady=(0, 1),
    )
    self.train_recommendation_model_label.grid_remove()
    self._train_recommendation_cells = []
    self._train_recommendation_header_labels = []
    self.train_recommendation_cards_grid = tk.Frame(
        self.train_recommendation_grid,
        bd=0,
        highlightthickness=0,
    )
    self.train_recommendation_cards_grid.grid(
        row=3,
        column=0,
        sticky="ew",
        padx=12,
        pady=(0, 10),
    )
    self.train_recommendation_cards_grid.grid_columnconfigure(0, weight=1, uniform="train_param_cards")
    self.train_recommendation_cards_grid.grid_columnconfigure(1, weight=1, uniform="train_param_cards")

    editor_specs = (
        ("Epoki", self.epochs_var, {"from_": 1, "to": 5000, "width": 8}, "epochs"),
        ("Batch", self.batch_var, {"from_": 1, "to": 256, "width": 8}, "batch"),
        ("Rozdzielczość", self.imgsz_var, {"from_": 32, "to": 2048, "increment": 32, "width": 8}, "imgsz"),
        ("LR", self.lr0_var, {"from_": 0.0001, "to": 0.1, "increment": 0.001, "format": "%.4f", "width": 8}, "lr0"),
    )
    for row_index, (param_text, variable, spinbox_kwargs, param_key) in enumerate(editor_specs):
        card = tk.Frame(
            self.train_recommendation_cards_grid,
            bd=0,
            highlightthickness=1,
            padx=6,
            pady=5,
        )
        card.grid(
            row=row_index // 2,
            column=row_index % 2,
            sticky="nsew",
            padx=(0 if row_index % 2 == 0 else 5, 5 if row_index % 2 == 0 else 0),
            pady=(0, 6),
        )
        card.grid_columnconfigure(0, weight=1)

        key_label = tk.Label(
            card,
            text=param_text,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bd=0,
        )
        key_label.grid(row=0, column=0, sticky="ew")

        editor_host = tk.Frame(
            card,
            bd=0,
            highlightthickness=0,
        )
        editor_host.grid(row=1, column=0, sticky="ew", pady=(4, 3))
        editor_host.grid_columnconfigure(0, weight=1)
        editor_host.grid_columnconfigure(1, weight=0)
        editor_host.grid_columnconfigure(2, weight=1)
        editor = ttk.Spinbox(
            editor_host,
            textvariable=variable,
            justify=tk.CENTER,
            **spinbox_kwargs,
        )
        editor.grid(row=0, column=1, sticky="", padx=(2, 2))

        recommended_label = tk.Label(
            card,
            text="-",
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            bd=0,
        )
        recommended_label.grid(row=2, column=0, sticky="ew")

        self._train_recommendation_cells.append(
            {
                "card": card,
                "key": key_label,
                "editor_host": editor_host,
                "editor": editor,
                "recommended": recommended_label,
                "param_key": param_key,
            }
        )
    self.train_recommendation_tree = None
    self.train_recommendation_note_var = tk.StringVar(value="Gdy zabraknie VRAM: najpierw zmniejsz batch, potem rozdzielczość.")
    self.train_recommendation_note_lbl = ttk.Label(
        settings_col,
        textvariable=self.train_recommendation_note_var,
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360,
    )
    self.train_recommendation_note_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 14))
    self._register_train_left_wrap_target(self.train_recommendation_note_lbl, padding=16, min_wrap=220)
    self.epochs_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.epochs_var.trace_add("write", lambda *args: self._refresh_training_execution_summary())
    self.batch_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.imgsz_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self.lr0_var.trace_add("write", lambda *args: self._refresh_training_recommendation_table())
    self._refresh_training_recommendation_table()
    self._apply_training_recommendation_table_theme()

    self.device_var = getattr(
        self,
        "device_var",
        tk.StringVar(value=self._get_global_training_device_choice()),
    )

    self.train_device_hint_lbl = ttk.Label(
        settings_col,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self.train_device_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 14))
    self._register_train_left_wrap_target(self.train_device_hint_lbl, padding=16, min_wrap=220)
    self._refresh_training_device_hint()

    self._build_train_left_separator(settings_col, pady=(4, self._train_left_section_gap))

    self.train_run_summary_shell = None
    self.train_run_summary_title_row = None
    self.train_run_summary_title_lbl = None
    self.train_run_summary_grid = None
    self.train_run_summary_rows_frame = None
    self.train_run_summary_header_key_lbl = None
    self.train_run_summary_header_value_lbl = None
    self._train_run_summary_row_widgets = []

    self.train_start_gate_shell = tk.Frame(
        settings_col,
        bd=0,
        highlightthickness=1,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        highlightbackground=section_border,
        highlightcolor=section_border,
    )
    self.train_start_gate_shell.pack(fill=tk.X, pady=(14, 10))
    self.train_start_gate_header = tk.Frame(
        self.train_start_gate_shell,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
    )
    self.train_start_gate_header.pack(fill=tk.X, padx=12, pady=(10, 5))
    self.train_start_gate_header.grid_columnconfigure(0, weight=1)
    self.train_start_gate_title_lbl = tk.Label(
        self.train_start_gate_header,
        text="Start treningu",
        font=("Segoe UI", 12),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        fg=palette.get("fg", "#f3f3f3"),
    )
    self.train_start_gate_title_lbl.grid(row=0, column=0, sticky="ew")
    self.train_start_gate_status_lbl = tk.Label(
        self.train_start_gate_header,
        text="Brakuje danych",
        font=("Segoe UI Semibold", 8),
        anchor=tk.CENTER,
        justify=tk.CENTER,
        bd=0,
        highlightthickness=0,
        padx=8,
        pady=2,
        bg=palette.get("surface_warning", palette.get("panel", "#252526")),
        fg=palette.get("warning", "#f0b44c"),
    )
    self.train_start_gate_status_lbl.grid(row=0, column=1, sticky="e", padx=(8, 0))
    self.train_start_gate_hint_lbl = tk.Label(
        self.train_start_gate_shell,
        text="Po kliknięciu uruchomimy nowy run na wybranym wariancie datasetu i modelu startowym.",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
        fg=palette.get("muted", "#c7c7c7"),
        wraplength=360,
    )
    self.train_start_gate_hint_lbl.pack(anchor=tk.W, fill=tk.X, padx=12, pady=(0, 10))
    self._register_train_left_wrap_target(self.train_start_gate_hint_lbl, padding=24, min_wrap=220)

    self.btn_step4_start_train_frame = tk.Frame(
        self.train_start_gate_shell,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel_alt", palette.get("panel", "#252526")),
    )
    self.btn_step4_start_train_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
    self.btn_step4_start_train_frame.grid_columnconfigure(0, weight=1)

    self.btn_step4_start_train_pulse_frame = tk.Frame(
        self.btn_step4_start_train_frame,
        bd=0,
        highlightthickness=1
    )
    self.btn_step4_start_train_pulse_frame.grid(row=0, column=0, sticky="ew")

    self.btn_start_train = ttk.Button(
        self.btn_step4_start_train_pulse_frame,
        text="Rozpocznij trening",
        command=self._start_training,
        style="Accent.TButton"
    )
    self.btn_start_train.pack(fill=tk.X)

    self.btn_pause_train = ttk.Button(
        self.btn_step4_start_train_frame,
        text="Pauza",
        command=self._pause_training,
        state=tk.DISABLED,
        width=12,
    )
    self.btn_pause_train.grid(row=0, column=1, sticky="e", padx=(8, 0))

    self.btn_stop_train = ttk.Button(
        self.btn_step4_start_train_frame,
        text="Stop",
        command=self._stop_training,
        state=tk.DISABLED,
        width=12,
    )
    self.btn_stop_train.grid(row=0, column=2, sticky="e", padx=(8, 0))

    self.train_epoch_progress_var = tk.DoubleVar(value=0.0)
    self.train_progress_var = tk.DoubleVar(value=0.0)
    self.train_progress_shell = tk.Frame(
        settings_col,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
    )
    self.train_progress_shell.pack(fill=tk.X, pady=(10, 0))

    self.train_epoch_progress_row = tk.Frame(
        self.train_progress_shell,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
    )
    self.train_epoch_progress_row.pack(fill=tk.X, pady=(0, 3))
    self.train_epoch_progress = TrainProgressBar(
        self.train_epoch_progress_row,
        variable=self.train_epoch_progress_var,
        maximum=100,
        thickness=4,
        trough_color=palette.get("surface_info", palette.get("panel_alt", palette.get("panel", "#252526"))),
        fill_color=palette.get("guide", palette.get("warning", "#f0b44c")),
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        height=8,
    )
    self.train_epoch_progress_measure_lbl = tk.Label(
        self.train_epoch_progress_row,
        text="Bieżąca epoka (partie danych)",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        fg=palette.get("muted", "#c7c7c7"),
        padx=2,
    )
    self.train_epoch_progress_measure_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
    self.train_epoch_progress.pack(anchor=tk.W, fill=tk.X)
    self.train_epoch_progress_hint_lbl = tk.Label(
        self.train_progress_shell,
        text="Pokazuje, ile partii danych zostało wykonanych w aktualnej epoce.",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        fg=palette.get("muted_dim", palette.get("muted", "#9a9a9a")),
        padx=2,
    )
    self.train_epoch_progress_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(1, 6))

    self.train_overall_progress_row = tk.Frame(
        self.train_progress_shell,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
    )
    self.train_overall_progress_row.pack(fill=tk.X)
    self.train_progress = TrainProgressBar(
        self.train_overall_progress_row,
        variable=self.train_progress_var,
        maximum=100,
        thickness=4,
        trough_color=palette.get("surface_info", palette.get("panel_alt", palette.get("panel", "#252526"))),
        fill_color=palette.get("success", "#2ecc71"),
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        height=8,
    )
    self.train_progress_measure_lbl = tk.Label(
        self.train_overall_progress_row,
        text="Cały run (epoki)",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        fg=palette.get("muted", "#c7c7c7"),
        padx=2,
    )
    self.train_progress_measure_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
    self.train_progress.pack(anchor=tk.W, fill=tk.X)
    self.train_progress_hint_lbl = tk.Label(
        self.train_progress_shell,
        text="Pokazuje, ile epok całego runu zostało już domkniętych względem planu.",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", palette.get("bg", "#1f1f1f")),
        fg=palette.get("muted_dim", palette.get("muted", "#9a9a9a")),
        padx=2,
    )
    self.train_progress_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(1, 0))
    self._configure_train_progress_styles()

    self.train_progress_label = ttk.Label(
        settings_col,
        text="Czekam na start...",
        font=("Segoe UI", 9),
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360,
    )
    self.train_progress_label.pack(anchor=tk.W, fill=tk.X, pady=(8, 3))
    self._register_train_left_wrap_target(self.train_progress_label, padding=16, min_wrap=220)
    self.train_resource_label = ttk.Label(
        settings_col,
        text="Zasoby w czasie treningu",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360,
    )
    self.train_resource_label.pack(anchor=tk.W, fill=tk.X, pady=(0, 3))
    self._register_train_left_wrap_target(self.train_resource_label, padding=16, min_wrap=220)

    try:
        compact_table_style = ttk.Style(settings_col)
        compact_table_style.configure(
            "TrainingCompact.Treeview",
            font=("Segoe UI", 8),
            rowheight=21,
        )
        compact_table_style.configure(
            "TrainingCompact.Treeview.Heading",
            font=("Segoe UI", 8),
            padding=(2, 3),
        )
    except Exception:
        pass

    self.train_resource_tree_host = ttk.Frame(settings_col, style="Panel.TFrame")
    self.train_resource_tree_host.pack(fill=tk.X, padx=(0, 4), pady=(0, 8))
    self.train_resource_tree = ttk.Treeview(
        self.train_resource_tree_host,
        columns=("Zasób", "Teraz", "Min", "Max", "Wolne", "Stan"),
        show="headings",
        height=4,
        style="TrainingCompact.Treeview",
    )
    resource_heading_labels = {
        "Zasób": "Zasób / jedn.",
    }
    for name, width, anchor in (
        ("Zasób", 96, tk.W),
        ("Teraz", 62, tk.CENTER),
        ("Min", 52, tk.CENTER),
        ("Max", 52, tk.CENTER),
        ("Wolne", 60, tk.CENTER),
        ("Stan", 70, tk.CENTER),
    ):
        self.train_resource_tree.heading(name, text=resource_heading_labels.get(name, name))
        self.train_resource_tree.column(name, width=width, minwidth=width, anchor=anchor, stretch=(name in {"Zasób", "Stan"}))
    self.train_resource_tree.pack(fill=tk.X, padx=(0, 2))
    try:
        self._configure_training_status_table_tags(self.train_resource_tree)
        self._set_training_resource_sample(None)
    except Exception:
        pass
    self.train_metric_hint_lbl = ttk.Label(
        settings_col,
        text="Najlepsza epoka tego runu pojawi się po pierwszej zakończonej epoce.",
        style="Panel.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self.train_metric_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 3))
    self._register_train_left_wrap_target(self.train_metric_hint_lbl, padding=16, min_wrap=220)

    self.train_live_metrics_tree_host = ttk.Frame(settings_col, style="Panel.TFrame")
    self.train_live_metrics_tree_host.pack(fill=tk.X, padx=(0, 4), pady=(0, 8))
    self.train_live_metrics_tree = ttk.Treeview(
        self.train_live_metrics_tree_host,
        columns=("Metryka", "Wynik", "Ocena", "Próg"),
        show="headings",
        height=3,
        style="TrainingCompact.Treeview",
    )
    for name, width, anchor in (
        ("Metryka", 118, tk.W),
        ("Wynik", 70, tk.CENTER),
        ("Ocena", 82, tk.CENTER),
        ("Próg", 112, tk.CENTER),
    ):
        self.train_live_metrics_tree.heading(name, text=name)
        self.train_live_metrics_tree.column(name, width=width, minwidth=width, anchor=anchor, stretch=(name in {"Metryka", "Próg"}))
    self.train_live_metrics_tree.pack(fill=tk.X, padx=(0, 2))
    try:
        self._configure_training_status_table_tags(self.train_live_metrics_tree)
    except Exception:
        pass
    self.train_metric_reference_lbl = ttk.Label(
        settings_col,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=360
    )
    self.train_metric_reference_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 14))
    self._register_train_left_wrap_target(self.train_metric_reference_lbl, padding=16, min_wrap=220)
    self._refresh_training_metric_reference()
    self._set_train_live_metrics(None)
    self._refresh_training_recommendation_table()
    self._refresh_training_start_state()

    terminal_tools = ttk.Frame(root, style="Panel.TFrame")
    terminal_tools.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    self.btn_toggle_step4_train_log = ttk.Button(
        terminal_tools,
        text="Pokaż terminal",
        command=self._toggle_step4_train_log
    )
    self.btn_toggle_step4_train_log.pack_forget()

    ttk.Label(
        terminal_tools,
        text="Wspólny terminal procesu dla treningu i walidacji jest dostępny na żądanie.",
        foreground="gray"
    ).pack(side=tk.LEFT, padx=(8, 0))

    self.step4_train_log_host = ttk.Frame(root, style="Panel.TFrame")
    self.step4_train_log_host.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    self.step4_train_log_frame = ttk.LabelFrame(
        self.step4_train_log_host,
        text=" Terminal procesu ",
        padding=6
    )
    self.step4_train_log_console_host = ttk.Frame(self.step4_train_log_frame, style="Panel.TFrame")
    self.step4_train_log_console_host.pack(fill=tk.BOTH, expand=True)

    self.train_log_console = tk.Text(
        self.step4_train_log_console_host,
        width=50,
        height=10,
        wrap=tk.NONE,
        font=("Consolas", 10),
        bg=palette.get("console_bg", palette.get("field", "#1e1e1e")),
        fg=palette.get("console_fg", palette.get("fg", "#ecf0f1")),
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
    )
    self.step4_train_log_hscrollbar = WebSlimScrollbar(
        self.step4_train_log_console_host,
        orient=tk.HORIZONTAL,
        command=self.train_log_console.xview,
        auto_hide=False,
    )
    self.step4_train_log_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.train_log_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.step4_train_log_scrollbar = WebSlimScrollbar(
        self.step4_train_log_console_host,
        orient=tk.VERTICAL,
        command=self.train_log_console.yview,
        auto_hide=False,
    )
    self.step4_train_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.train_log_console.configure(
        yscrollcommand=self.step4_train_log_scrollbar.set,
        xscrollcommand=self.step4_train_log_hscrollbar.set,
    )
    self.train_log_console.web_vbar = self.step4_train_log_scrollbar
    self.train_log_console.web_hbar = self.step4_train_log_hscrollbar
    self._set_step4_process_console_text(
        "Oczekuję na rozpoczęcie treningu lub walidacji...\n"
        "Terminal procesu jest gotowy na dane z Ultralytics.\n"
    )
    self._set_step4_train_log_visibility(False)
    try:
        terminal_tools.grid_remove()
    except Exception:
        pass

    self.campaign_training_result_host = ttk.Frame(
        self.right,
        padding=(8, 0, 6, 4),
        style="Panel.TFrame",
    )
    self.campaign_training_result_host.pack(fill=tk.X)

    self.right_nb = ttk.Notebook(self.right)
    self.right_nb.pack(fill=tk.BOTH, expand=True)

    self.hist_tab = ttk.Frame(self.right_nb)
    self.ranking_tab = ttk.Frame(self.right_nb)
    self.right_nb.add(self.hist_tab, text="Historia treningów")
    self.right_nb.add(self.ranking_tab, text="Ranking")
    self._step4_ranking_tab_visible = True
    self.right_nb.bind("<<NotebookTabChanged>>", self._sync_step4_analysis_nav_buttons)

    hist_top = ttk.Frame(self.hist_tab, padding=(8, 0, 6, 0), style="Panel.TFrame")
    hist_top.pack(fill=tk.BOTH, expand=True)

    result_bg = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.82,
    )
    result_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.24,
    )
    self.campaign_training_result_var = getattr(self, "campaign_training_result_var", tk.StringVar())
    self._campaign_training_result_choices = {}
    finish_gate_id = "T06"
    self.campaign_training_result_shell = tk.Frame(
        self.campaign_training_result_host,
        bg=result_bg,
        bd=0,
        highlightthickness=2,
        highlightbackground=result_border,
        highlightcolor=result_border,
    )
    self.campaign_training_result_shell.pack(fill=tk.X, pady=(0, 9))
    result_title_row = tk.Frame(self.campaign_training_result_shell, bg=result_bg, bd=0, highlightthickness=0)
    self.campaign_training_result_title_row = result_title_row
    result_title_row.pack(fill=tk.X, padx=12, pady=(9, 3))
    result_title_row.grid_columnconfigure(0, weight=1)
    self.campaign_training_result_title_lbl = tk.Label(
        result_title_row,
        text=f"Wynik bramki {finish_gate_id}",
        font=("Segoe UI Semibold", 12),
        anchor=tk.W,
        justify=tk.LEFT,
        bg=result_bg,
        fg=palette.get("success", "#2ecc71"),
        bd=0,
        padx=0,
        pady=0,
    )
    self.campaign_training_result_title_lbl.grid(row=0, column=0, sticky="w")
    self.campaign_training_result_status_lbl = tk.Label(
        result_title_row,
        text="SPRAWDZAM",
        font=("Segoe UI Semibold", 9),
        anchor=tk.CENTER,
        justify=tk.CENTER,
        bg=palette.get("panel", "#252526"),
        fg=palette.get("muted", "#c7c7c7"),
        bd=0,
        highlightthickness=1,
        highlightbackground=result_border,
        highlightcolor=result_border,
        padx=10,
        pady=3,
    )
    self.campaign_training_result_status_lbl.grid(row=0, column=1, sticky="e", padx=(8, 0))
    self.campaign_training_result_copy_lbl = tk.Label(
        self.campaign_training_result_shell,
        text="Tu podpinasz model wynikowy bramki. To osobna decyzja od wyboru modelu startowego treningu.",
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bg=result_bg,
        fg=palette.get("muted", "#c7c7c7"),
        bd=0,
        padx=0,
        pady=0,
        wraplength=760,
    )
    self.campaign_training_result_copy_lbl.pack(anchor=tk.W, fill=tk.X, padx=12, pady=(0, 7))
    # Keep the global selector compact so it does not steal vertical space from Ranking.
    result_pick_row = tk.Frame(self.campaign_training_result_shell, bg=result_bg, bd=0, highlightthickness=0)
    self.campaign_training_result_pick_row = result_pick_row
    result_pick_row.pack(fill=tk.X, padx=12, pady=(0, 9))
    result_pick_row.columnconfigure(0, weight=1)
    result_field_bg = palette.get("field", "#1f1f1f")
    result_field_fg = palette.get("fg", "#f3f3f3")
    result_select_bg = blend_hex_colors(palette.get("success", "#2ecc71"), result_field_bg, 0.55)
    result_selector_shell = tk.Frame(
        result_pick_row,
        bg=result_field_bg,
        highlightthickness=1,
        highlightbackground=palette.get("border", "#3c3c3c"),
        highlightcolor=palette.get("success", "#2ecc71"),
        bd=0,
    )
    self.campaign_training_result_selector_shell = result_selector_shell
    result_selector_shell.grid(row=0, column=0, sticky="ew")
    result_selector_shell.grid_columnconfigure(0, weight=1)
    self.campaign_training_result_entry = tk.Entry(
        result_selector_shell,
        textvariable=self.campaign_training_result_var,
        bd=0,
        highlightthickness=0,
        relief=tk.FLAT,
        bg=result_field_bg,
        fg=result_field_fg,
        insertbackground=result_field_fg,
        insertwidth=2,
        selectbackground=result_select_bg,
        selectforeground=result_field_fg,
        disabledbackground=palette.get("panel", "#252526"),
        disabledforeground=palette.get("muted_dim", "#777777"),
        cursor="xterm",
    )
    self.campaign_training_result_entry.grid(row=0, column=0, sticky="ew", padx=(8, 2), pady=4)
    self.campaign_training_result_menu = tk.Menu(
        result_selector_shell,
        tearoff=0,
        bg=result_field_bg,
        fg=result_field_fg,
        activebackground=result_select_bg,
        activeforeground=result_field_fg,
        borderwidth=1,
        relief=tk.SOLID,
    )

    def _open_campaign_result_menu(_event=None):
        menu = getattr(self, "campaign_training_result_menu", None)
        button = getattr(self, "campaign_training_result_dropdown_btn", None)
        selector = getattr(self, "campaign_training_result_selector_shell", None)
        if menu is None or button is None:
            return "break"
        try:
            if str(button.cget("state")) == tk.DISABLED:
                return "break"
        except Exception:
            pass
        try:
            anchor = selector if selector is not None else button
            menu.tk_popup(anchor.winfo_rootx(), anchor.winfo_rooty() + anchor.winfo_height())
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass
        return "break"

    self.campaign_training_result_dropdown_btn = tk.Button(
        result_selector_shell,
        text="v",
        command=_open_campaign_result_menu,
        bd=0,
        highlightthickness=0,
        relief=tk.FLAT,
        padx=8,
        pady=2,
        bg=palette.get("panel_alt", "#2d2d30"),
        fg=result_field_fg,
        activebackground=blend_hex_colors(palette.get("success", "#2ecc71"), result_field_bg, 0.72),
        activeforeground=result_field_fg,
        cursor="hand2",
    )
    self.campaign_training_result_dropdown_btn.grid(row=0, column=1, sticky="ns")
    self.campaign_training_result_combo = self.campaign_training_result_entry
    def _show_campaign_result_start(_event=None):
        entry = getattr(self, "campaign_training_result_entry", None)
        if entry is None:
            return
        try:
            entry.icursor(0)
        except Exception:
            pass
        try:
            entry.xview_moveto(0.0)
        except Exception:
            pass

    def _guard_campaign_result_text_edit(event):
        entry = getattr(self, "campaign_training_result_entry", None)
        keysym = str(getattr(event, "keysym", "") or "")
        ctrl_pressed = bool(int(getattr(event, "state", 0) or 0) & 0x4)
        if ctrl_pressed and keysym.lower() == "a":
            try:
                entry.selection_range(0, tk.END)
                entry.icursor(tk.END)
            except Exception:
                pass
            return "break"
        if ctrl_pressed and keysym.lower() in {"c", "insert"}:
            return None
        if keysym in {"Left", "Right", "Home", "End", "Tab", "Escape"}:
            return None
        return "break"

    self.campaign_training_result_entry.bind("<KeyPress>", _guard_campaign_result_text_edit, add="+")
    self.campaign_training_result_entry.bind("<<Paste>>", lambda _event: "break", add="+")
    self.campaign_training_result_entry.bind("<<Cut>>", lambda _event: "break", add="+")
    self.campaign_training_result_entry.bind("<FocusIn>", _show_campaign_result_start, add="+")
    self.btn_use_campaign_training_result = ttk.Button(
        result_pick_row,
        text=f"Podepnij jako wynik bramki {finish_gate_id}",
        command=self._use_campaign_training_result_choice,
        width=38,
    )
    self.btn_use_campaign_training_result.grid(row=0, column=1, sticky="e", padx=(8, 0))
    self.campaign_training_result_detail_lbl = ttk.Label(
        self.campaign_training_result_shell,
        text="",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    # Detailed guidance is available in the action modals; this header stays one-line.

    history_area = tk.Frame(
        hist_top,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    history_area.pack(fill=tk.BOTH, expand=True)
    history_area.grid_columnconfigure(0, weight=1)
    history_area.grid_rowconfigure(1, weight=1)

    ttk.Label(
        history_area,
        text=(
            "Wybierz trening z listy, aby zobaczyć konfigurację i metryki."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

    hist_tree_shell = ttk.LabelFrame(history_area, text=" Historia treningów ", padding=6)
    hist_tree_shell.grid(row=1, column=0, sticky="nsew")
    columns = ("Tor", "Model", "Dataset", "Run", "Start", "Status", "Epoki", "mAP50-95")
    self.tree = ttk.Treeview(hist_tree_shell, columns=columns, show="headings", height=5, selectmode="extended")
    history_heading_labels = {
        "Model": "Model <- Z4/PZ2",
        "Dataset": "Dataset <- Z4/PZ1",
        "Run": "Run <- Z4/PZ2",
    }
    for c in columns:
        self.tree.heading(c, text=history_heading_labels.get(c, c))
    self.tree.column("Tor", width=62, stretch=False, anchor=tk.CENTER)
    self.tree.column("Model", width=136, minwidth=116, stretch=True, anchor=tk.W)
    self.tree.column("Dataset", width=148, minwidth=126, stretch=True, anchor=tk.W)
    self.tree.column("Run", width=132, minwidth=112, stretch=True, anchor=tk.W)
    self.tree.column("Start", width=90, stretch=False, anchor=tk.CENTER)
    self.tree.column("Status", width=104, stretch=False)
    self.tree.column("Epoki", width=54, stretch=False, anchor=tk.CENTER)
    self.tree.column("mAP50-95", width=76, stretch=False, anchor=tk.CENTER)

    yscroll = WebSlimScrollbar(hist_tree_shell, orient=tk.VERTICAL, command=self.tree.yview)
    self.tree.configure(yscrollcommand=yscroll.set)
    self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.tree.bind("<<TreeviewSelect>>", self._on_run_selected)
    self.tree.bind("<Double-1>", self._open_selected_run_details, add="+")
    self.tree.bind("<Button-3>", self._show_history_context_menu, add="+")
    self.history_context_menu = tk.Menu(self.tree, tearoff=0)
    self.history_context_menu.add_command(label="Wznów trening", command=self._resume_selected_run)
    self.history_context_menu.add_command(label="Dotrenuj od best.pt", command=self._select_selected_run_as_fine_tune_base)
    self.history_context_menu.add_command(label="[ TEST ] Waliduj best.pt", command=self._open_selected_run_validation_modal)
    self.history_context_menu.add_command(label="Szczegóły runu", command=self._open_selected_run_details)
    self.history_context_menu.add_command(label="Porównaj zaznaczone runy", command=self._open_selected_runs_compare)
    self.history_context_menu.add_command(label="Otwórz folder", command=self._open_run_folder)
    self.history_context_menu.add_command(
        label="Eksportuj best.pt do modeli trybu swobodnego",
        command=self._export_selected_run_model_to_free_mode,
    )
    self.history_context_menu.add_command(
        label="Eksportuj model mobilny (.alprmodel)",
        command=self._export_selected_run_model_to_mobile_package,
    )
    self.history_context_menu.add_command(
        label=f"Podepnij jako wynik bramki {finish_gate_id}",
        command=self._promote_selected_run_model_to_campaign,
    )
    self.history_context_menu.add_command(label="Usun", command=self._delete_selected)

    hist_actions = tk.Frame(
        history_area,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    hist_actions.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    tk.Label(
        hist_actions,
        text="Zaznacz jeden run albo kilka runów z Ctrl/Shift, a potem wybierz akcję.",
        bg=palette.get("panel", "#252526"),
        fg=palette.get("muted", "#c7c7c7"),
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        bd=0,
        padx=2,
        pady=0,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 7))

    hist_buttons_row = tk.Frame(
        hist_actions,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    hist_buttons_row.pack(fill=tk.X, padx=(2, 0))
    hist_buttons_row.grid_columnconfigure(0, weight=1, uniform="history_actions")
    hist_buttons_row.grid_columnconfigure(1, weight=1, uniform="history_actions")

    def _history_action_button(text: str, command):
        base_bg = palette.get("panel_alt", "#2d2d30")
        hover_bg = palette.get("button_hover", palette.get("panel_alt", "#2d2d30"))
        border = palette.get("success", "#4ec9b0")
        border_hover = palette.get("accent_hover", palette.get("success", "#4ec9b0"))
        wrapper = tk.Frame(
            hist_buttons_row,
            bg=border,
            bd=0,
            highlightthickness=0,
            padx=1,
            pady=1,
        )
        button = tk.Button(
            wrapper,
            text=text,
            command=command,
            bg=base_bg,
            fg=palette.get("fg", "#f3f3f3"),
            activebackground=hover_bg,
            activeforeground=palette.get("fg", "#f3f3f3"),
            font=("Segoe UI Semibold", 9),
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            padx=16,
            pady=8,
            cursor="hand2",
        )
        button.pack(fill=tk.BOTH, expand=True)

        def _enter(_event=None):
            try:
                wrapper.configure(bg=border_hover)
                button.configure(bg=hover_bg)
            except Exception:
                pass

        def _leave(_event=None):
            try:
                wrapper.configure(bg=border)
                button.configure(bg=base_bg)
            except Exception:
                pass

        wrapper.bind("<Enter>", _enter, add="+")
        wrapper.bind("<Leave>", _leave, add="+")
        button.bind("<Enter>", _enter, add="+")
        button.bind("<Leave>", _leave, add="+")
        return wrapper

    _history_action_button(
        "Szczegóły runu",
        self._open_selected_run_details,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 2))
    _history_action_button(
        "Porównaj wybrane",
        self._open_selected_runs_compare,
    ).grid(row=0, column=1, sticky="ew", pady=(0, 2))
    _history_action_button(
        "Dotrenuj wybrany model",
        self._select_selected_run_as_fine_tune_base,
    ).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(6, 0))
    _history_action_button(
        "[ TEST ] Waliduj model",
        self._open_selected_run_validation_modal,
    ).grid(row=1, column=1, sticky="ew", pady=(6, 0))

    self._build_ranking_panel_v2(self.ranking_tab)
    self._refresh_step4_analysis_tab_visibility()
    self._sync_step4_analysis_nav_buttons()
    self._set_history_run_tables(None)

    self.step4_train_nav = ttk.Frame(root)
    self.step4_train_nav.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    self.btn_step4_train_back = ttk.Button(
        self.step4_train_nav,
        text="Wróć do grafu",
        command=self._step4_train_go_back,
        style="WorkflowCard.TButton"
    )
    self.btn_step4_train_back.pack(side=tk.RIGHT, padx=(8, 0))
    self.btn_step4_train_back.configure(text="Wróć do grafu", width=NAV_BUTTON_WIDTH)

    self.btn_step4_finish_frame = tk.Frame(self.step4_train_nav, bd=0, highlightthickness=0)
    self.btn_step4_finish_frame.pack(side=tk.RIGHT)

    self.btn_step4_complete_project = ttk.Button(
        self.btn_step4_finish_frame,
        text="Zakończ projekt",
        command=self._complete_campaign_project,
        style="WorkflowCard.TButton",
        state=tk.DISABLED
    )
    self.btn_step4_complete_project.pack(side=tk.LEFT, padx=(0, 8))
    self.btn_step4_complete_project.configure(width=NAV_BUTTON_WIDTH)

    self.btn_step4_finish = ttk.Button(
        self.btn_step4_finish_frame,
        text="Stwórz inny wariant datasetu",
        command=self._open_step4_dataset_stage,
        style="Accent.TButton",
        state=tk.DISABLED
    )
    self.btn_step4_finish.pack(side=tk.LEFT)
    self.btn_step4_finish.configure(text="Stwórz inny wariant datasetu", width=28)

    self._refresh_step4_campaign_navigation_ui()

    if self.training_input_summary_frame is not None:
        HELP.bind_help(self.training_input_summary_frame, "tr_train_input_summary")
    if self.training_input_change_btn is not None:
        HELP.bind_help(self.training_input_change_btn, "tr_train_input_summary")
    HELP.bind_help(self.train_dataset_section_frame, "tr_train_ds")
    HELP.bind_help(self.dataset_variant_row, "tr_dataset_variant")
    HELP.bind_help(self.dataset_variant_combo, "tr_dataset_variant")
    HELP.bind_help(self.btn_character_class_distribution, "tr_train_ds")
    HELP.bind_help(self.train_dataset_hint_lbl, "tr_train_ds")
    HELP.bind_help(self.base_combo, "tr_train_base")
    HELP.bind_help(self.base_custom_btn, "tr_train_custom")
    HELP.bind_help(self.train_recommendation_grid, "tr_train_params")
    HELP.bind_help(self.btn_apply_training_recommendation, "tr_train_recommendation")
    HELP.bind_help(self.train_device_hint_lbl, "tr_train_device")
    HELP.bind_help(self.btn_step4_back, "tr_builder_back")

    try:
        recommendation_rows = [
            row for row in list(getattr(self, "_train_recommendation_cells", []) or [])
            if row.get("editor") is not None
        ]
        help_keys = ("tr_train_ep", "tr_train_bs", "tr_train_imgsz", "tr_train_lr0")
        for row, help_key in zip(recommendation_rows, help_keys):
            HELP.bind_help(row.get("editor"), help_key)
    except Exception as e:
        logger.debug(f"Błąd podpinania pomocy do siatki: {e}")

    HELP.bind_help(self.btn_start_train, "tr_train_btn")
    HELP.bind_help(self.btn_pause_train, "tr_train_control")
    HELP.bind_help(self.btn_stop_train, "tr_train_control")
    HELP.bind_help(self.tree, "tr_train_tree")
    HELP.bind_help(self.custom_row, "tr_train_custom")
    HELP.bind_help(self.btn_toggle_step4_train_log, "tr_train_log")
    HELP.bind_help(self.btn_step4_train_back, "tr_train_back")
    HELP.bind_help(self.btn_step4_finish, "tr_train_finish")
    HELP.bind_help(self.btn_step4_complete_project, "tr_train_finish")

    self._bind_scroll_canvas_children(self.train_left_content, self.train_left_canvas)
    self.frame.after_idle(self._sync_train_left_scrollregion)
    self.frame.after_idle(self._update_training_dataset_hint_wraplength)
    self.frame.after_idle(self._sync_train_left_canvas_width)
    self.frame.after_idle(self._init_train_pane_layout)
    self.frame.bind_all("<MouseWheel>", self._on_train_left_global_mousewheel, add="+")
    self.frame.bind_all("<Button-4>", self._on_train_left_global_mousewheel, add="+")
    self.frame.bind_all("<Button-5>", self._on_train_left_global_mousewheel, add="+")
