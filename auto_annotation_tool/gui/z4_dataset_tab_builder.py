#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 PZ1 dataset tab builder extracted from tab_training.py."""

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
from . import z4_validation_panel
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def _build_dataset_tab(self):
    build_started = time.perf_counter()
    root = ttk.Frame(self.tab_dataset, padding=8)
    root.pack(fill=tk.BOTH, expand=True)

    top = ttk.Frame(root)
    top.pack(fill=tk.BOTH, expand=True)

    self.step4_route_panel_frame = tk.Frame(top, bd=0, highlightthickness=1)
    self.step4_route_panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    self.step4_route_panel_frame.configure(width=252)
    self.step4_route_panel_frame.pack_propagate(False)
    self.step4_dataset_top = top

    left = ttk.LabelFrame(self.step4_route_panel_frame, text=" Typ datasetu ", padding=10)
    left.pack(fill=tk.BOTH, expand=True)
    self.step4_route_choice_frame = left

    self.step4_route_intro_lbl = ttk.Label(
        left,
        text="Najpierw wybierz typ datasetu. PZ1 utworzy wariant treningowy, który potem wybierzesz w PZ2.",
        font=("Segoe UI", 9),
        wraplength=205,
        justify=tk.LEFT
    )
    self.step4_route_intro_lbl.pack(anchor=tk.W, pady=(0, 12))

    self._step4_route_choice_cards = {}
    route_palette = getattr(self.app, "palette", {})
    route_initial_badge_bg = route_palette.get("panel", "#252526")
    route_initial_badge_fg = route_palette.get("muted", "#c7c7c7")
    route_initial_badge_border = route_palette.get("panel_border", route_palette.get("border", "#3c3c3c"))

    self.step4_plate_route_card = tk.Frame(left, bd=0, highlightthickness=1, padx=10, pady=8)
    self.step4_plate_route_card.pack(fill=tk.X, pady=(0, 10))
    self.step4_plate_route_title_row = tk.Frame(self.step4_plate_route_card, bd=0, highlightthickness=0)
    self.step4_plate_route_title_row.pack(fill=tk.X)
    self.step4_plate_route_title_lbl = tk.Label(
        self.step4_plate_route_title_row,
        text="Dataset tablic",
        font=("Segoe UI Semibold", 10),
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.step4_plate_route_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.step4_plate_route_badge_lbl = tk.Label(
        self.step4_plate_route_title_row,
        text="DO WYBORU",
        font=("Segoe UI", 8, "bold"),
        width=10,
        anchor=tk.CENTER,
        padx=7,
        pady=2,
        bd=0,
        highlightthickness=1,
        bg=route_initial_badge_bg,
        fg=route_initial_badge_fg,
        highlightbackground=route_initial_badge_border,
        highlightcolor=route_initial_badge_border,
    )
    self.step4_plate_route_badge_lbl.pack(side=tk.RIGHT, padx=(6, 0))
    self.step4_plate_route_desc_lbl = tk.Label(
        self.step4_plate_route_card,
        text="YOLO Pose. Źródłem jest XML z anotacjami tablic i zgodny katalog zdjęć albo gotowy dataset tablic.",
        wraplength=195,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.step4_plate_route_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self.btn_choose_plate = ttk.Button(
        left,
        text="Wybierz dataset tablic",
        command=lambda: self._set_step4_dataset_mode("plate")
    )

    self.step4_char_route_card = tk.Frame(left, bd=0, highlightthickness=1, padx=10, pady=8)
    self.step4_char_route_card.pack(fill=tk.X)
    self.step4_char_route_title_row = tk.Frame(self.step4_char_route_card, bd=0, highlightthickness=0)
    self.step4_char_route_title_row.pack(fill=tk.X)
    self.step4_char_route_title_lbl = tk.Label(
        self.step4_char_route_title_row,
        text="Dataset znaków",
        font=("Segoe UI Semibold", 10),
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.step4_char_route_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.step4_char_route_badge_lbl = tk.Label(
        self.step4_char_route_title_row,
        text="DO WYBORU",
        font=("Segoe UI", 8, "bold"),
        width=10,
        anchor=tk.CENTER,
        padx=7,
        pady=2,
        bd=0,
        highlightthickness=1,
        bg=route_initial_badge_bg,
        fg=route_initial_badge_fg,
        highlightbackground=route_initial_badge_border,
        highlightcolor=route_initial_badge_border,
    )
    self.step4_char_route_badge_lbl.pack(side=tk.RIGHT, padx=(6, 0))
    self.step4_char_route_desc_lbl = tk.Label(
        self.step4_char_route_card,
        text="YOLO Detect. Źródłem jest gotowy dataset znaków, z którego PZ1 utworzy wariant treningowy.",
        wraplength=195,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.step4_char_route_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self.btn_choose_char = ttk.Button(
        left,
        text="Wybierz dataset znaków",
        command=lambda: self._set_step4_dataset_mode("char")
    )

    self._step4_route_choice_cards = {
        "plate": {
            "frame": self.step4_plate_route_card,
            "title_row": self.step4_plate_route_title_row,
            "title": self.step4_plate_route_title_lbl,
            "badge": self.step4_plate_route_badge_lbl,
            "desc": self.step4_plate_route_desc_lbl,
        },
        "char": {
            "frame": self.step4_char_route_card,
            "title_row": self.step4_char_route_title_row,
            "title": self.step4_char_route_title_lbl,
            "badge": self.step4_char_route_badge_lbl,
            "desc": self.step4_char_route_desc_lbl,
        },
    }
    for route_mode, widgets in self._step4_route_choice_cards.items():
        for widget in widgets.values():
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass
            try:
                widget.bind("<Button-1>", lambda _event, mode=route_mode: self._set_step4_dataset_mode(mode), add="+")
            except Exception:
                pass

    right = ttk.Frame(top)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.step4_mode_right_host = right

    def _sync_step4_dataset_right_wraps(_event=None):
        try:
            panel_width = int(right.winfo_width() or 0)
        except Exception:
            panel_width = 0
        wrap = max(360, panel_width - 34)

        def _visit(widget):
            try:
                children = widget.winfo_children()
            except Exception:
                return
            for child in children:
                try:
                    current = int(float(child.cget("wraplength") or 0))
                except Exception:
                    current = 0
                if current > 0:
                    try:
                        child.configure(wraplength=wrap)
                    except Exception:
                        pass
                _visit(child)

        _visit(right)
        try:
            self._sync_dataset_mode_scrollregion()
        except Exception:
            pass

    def _schedule_step4_dataset_right_wraps(_event=None):
        try:
            pending = getattr(self, "_step4_dataset_right_wrap_after_id", None)
            if pending is not None:
                self.frame.after_cancel(pending)
        except Exception:
            pass
        try:
            self._step4_dataset_right_wrap_after_id = self.frame.after(50, _sync_step4_dataset_right_wraps)
        except Exception:
            self._step4_dataset_right_wrap_after_id = None
            _sync_step4_dataset_right_wraps()

    right.bind("<Configure>", _schedule_step4_dataset_right_wraps, add="+")

    header = ttk.LabelFrame(right, text=" Aktywny typ datasetu ", padding=10)
    header.pack(fill=tk.X)
    self.ds_mode_header_frame = header

    self.ds_mode_title_var = tk.StringVar(value="Dataset znaków (YOLO Detect)")
    self.ds_mode_desc_var = tk.StringVar(value="")

    ttk.Label(
        header,
        textvariable=self.ds_mode_title_var,
        font=("Segoe UI", 11, "bold")
    ).pack(anchor=tk.W)

    ttk.Label(
        header,
        textvariable=self.ds_mode_desc_var,
        wraplength=700,
        justify=tk.LEFT
    ).pack(anchor=tk.W, pady=(6, 0))

    summary_palette = getattr(self.app, "palette", {})
    summary_panel = summary_palette.get("panel", "#252526")
    summary_border = summary_palette.get("panel_border", summary_palette.get("border", "#3c3c3c"))
    self.step4_dataset_summary_frame = tk.Frame(
        right,
        bd=0,
        highlightthickness=1,
        padx=8,
        pady=5,
        bg=summary_panel,
        highlightbackground=summary_border,
        highlightcolor=summary_border,
    )
    self.step4_dataset_summary_frame.pack(fill=tk.X, pady=(6, 0))
    self.step4_dataset_summary_title_lbl = tk.Label(
        self.step4_dataset_summary_frame,
        text="Podsumowanie splitu",
        font=("Segoe UI Semibold", 9),
        anchor="w",
        bd=0,
        highlightthickness=0,
        bg=summary_panel,
    )
    self.step4_dataset_summary_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 3))
    self.step4_dataset_summary_grid = tk.Frame(
        self.step4_dataset_summary_frame,
        bd=0,
        highlightthickness=0,
        bg=summary_panel,
    )
    self.step4_dataset_summary_grid.pack(fill=tk.X)
    self.step4_dataset_summary_grid.grid_columnconfigure(0, weight=0, minsize=92)
    self.step4_dataset_summary_grid.grid_columnconfigure(1, weight=1)

    self._step4_dataset_summary_header_widgets = []
    self._step4_dataset_summary_rows = {}
    for column, text in enumerate(("Parametr", "Wartość")):
        header_cell = tk.Label(
            self.step4_dataset_summary_grid,
            text=text,
            font=("Segoe UI Semibold", 8),
            padx=6,
            pady=2,
            anchor="w",
            bd=0,
            highlightthickness=1,
        )
        header_cell.grid(row=0, column=column, sticky="nsew")
        self._step4_dataset_summary_header_widgets.append(header_cell)

    summary_rows = (
        ("status", "Status"),
        ("dataset_id", "ID datasetu"),
        ("target", "Dataset"),
        ("path", "Wariant"),
        ("split", "Train / val / test"),
    )
    for row_index, (key, label_text) in enumerate(summary_rows, start=1):
        label_cell = tk.Label(
            self.step4_dataset_summary_grid,
            text=label_text,
            font=("Segoe UI", 8),
            padx=6,
            pady=2,
            anchor="w",
            bd=0,
            highlightthickness=1,
        )
        label_cell.grid(row=row_index, column=0, sticky="nsew")
        if key == "split":
            value_cell = tk.Canvas(
                self.step4_dataset_summary_grid,
                height=22,
                bd=0,
                highlightthickness=1,
            )
            value_cell.bind(
                "<Configure>",
                lambda _event, canvas=value_cell: self._draw_step4_dataset_split_bar(canvas),
                add="+",
            )
        else:
            value_cell = tk.Label(
                self.step4_dataset_summary_grid,
                text="-",
                font=("Segoe UI", 8),
                padx=6,
                pady=2,
                anchor="w",
                justify=tk.LEFT,
                bd=0,
                highlightthickness=1,
            )
        value_cell.grid(row=row_index, column=1, sticky="nsew")
        self._step4_dataset_summary_rows[key] = (label_cell, value_cell)
    if CAMPAIGN.get_active_project_name():
        try:
            self.step4_dataset_summary_frame.pack_forget()
        except Exception:
            pass

    self.ds_mode_scroll_host = ttk.Frame(right)
    self.ds_mode_scroll_host.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
    self.ds_mode_scroll_host.grid_rowconfigure(0, weight=1)
    self.ds_mode_scroll_host.grid_columnconfigure(0, weight=1)

    palette = getattr(self.app, "palette", {})
    self.ds_mode_canvas = tk.Canvas(
        self.ds_mode_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.ds_mode_canvas.grid(row=0, column=0, sticky="nsew")

    self.ds_mode_scrollbar = WebSlimScrollbar(
        self.ds_mode_scroll_host,
        command=self.ds_mode_canvas.yview,
        auto_hide=False,
    )
    self.ds_mode_scrollbar.grid(row=0, column=1, sticky="ns")
    self.ds_mode_canvas.configure(yscrollcommand=self.ds_mode_scrollbar.set)

    self.ds_mode_host = ttk.Frame(self.ds_mode_canvas)
    self.ds_mode_host_window = self.ds_mode_canvas.create_window(
        (0, 0),
        window=self.ds_mode_host,
        anchor="nw",
    )
    self.ds_mode_host.bind("<Configure>", self._sync_dataset_mode_scrollregion, add="+")
    self.ds_mode_canvas.bind("<Configure>", self._sync_dataset_mode_canvas_width, add="+")

    self.ds_mode_waiting_frame = ttk.LabelFrame(
        self.ds_mode_host,
        text=" Oczekiwanie na wybór toru ",
        padding=18
    )
    ttk.Label(
        self.ds_mode_waiting_frame,
        text=(
            "Panel zostanie odblokowany po wyborze toru po lewej stronie.\n\n"
            "W trybie projektu najpierw wskaż, czy chcesz prowadzić tor tablic, czy tor znaków."
        ),
        foreground="gray",
        justify=tk.LEFT,
        wraplength=520
    ).pack(anchor=tk.W)

    self.ds_creator_frame = ttk.LabelFrame(
        self.ds_mode_host,
        text=" Wariant treningowy tablic (YOLO Pose) ",
        padding=10
    )
    self.ds_split_frame = ttk.LabelFrame(
        self.ds_mode_host,
        text=" Wariant treningowy znaków (YOLO Detect) ",
        padding=10
    )

    self.train_pct = tk.DoubleVar(value=80.0)
    self.val_pct = tk.DoubleVar(value=10.0)

    self._build_creator_ui()
    self._build_splitter_ui()
    try:
        self.frame.after_idle(_sync_step4_dataset_right_wraps)
    except Exception:
        pass
    self._bind_scroll_canvas_children(self.ds_mode_host, self.ds_mode_canvas)
    try:
        self.ds_mode_canvas.bind(
            "<MouseWheel>",
            lambda event, canvas=self.ds_mode_canvas: self._redirect_child_mousewheel_to_canvas(event, canvas),
            add="+",
        )
        self.ds_mode_canvas.bind(
            "<Button-4>",
            lambda event, canvas=self.ds_mode_canvas: self._redirect_child_mousewheel_to_canvas(event, canvas),
            add="+",
        )
        self.ds_mode_canvas.bind(
            "<Button-5>",
            lambda event, canvas=self.ds_mode_canvas: self._redirect_child_mousewheel_to_canvas(event, canvas),
            add="+",
        )
    except Exception:
        pass
    try:
        self.frame.bind_all("<MouseWheel>", self._on_dataset_mode_global_mousewheel, add="+")
        self.frame.bind_all("<Button-4>", self._on_dataset_mode_global_mousewheel, add="+")
        self.frame.bind_all("<Button-5>", self._on_dataset_mode_global_mousewheel, add="+")
    except Exception:
        pass

    self.step4_builder_log_frame = ttk.LabelFrame(root, text=" Terminal procesu ", padding=6)
    self.step4_builder_log_toolbar = ttk.Frame(self.step4_builder_log_frame)
    self.step4_builder_log_toolbar.pack(fill=tk.X, pady=(0, 6))

    self.btn_hide_step4_log = ttk.Button(
        self.step4_builder_log_toolbar,
        text="Ukryj terminal",
        command=self._toggle_step4_builder_log
    )
    self.btn_hide_step4_log.pack(side=tk.LEFT)

    self.step4_builder_log_host = ttk.Frame(self.step4_builder_log_frame, style="Panel.TFrame")
    self.step4_builder_log_host.pack(fill=tk.BOTH, expand=True)

    self.step4_builder_log_text = tk.Text(
        self.step4_builder_log_host,
        wrap=tk.WORD,
        height=10,
        font=("Consolas", 10),
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
    )
    self.step4_builder_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.step4_builder_log_scrollbar = WebSlimScrollbar(
        self.step4_builder_log_host,
        orient=tk.VERTICAL,
        command=self.step4_builder_log_text.yview,
        auto_hide=False,
    )
    self.step4_builder_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.step4_builder_log_text.configure(yscrollcommand=self.step4_builder_log_scrollbar.set)
    self.step4_builder_log_text.web_vbar = self.step4_builder_log_scrollbar
    self.step4_builder_log_text.configure(state=tk.DISABLED)

    self.step4_builder_tools = ttk.Frame(root)
    self.step4_builder_tools.pack_forget()

    self.step4_builder_nav = ttk.Frame(root)
    self.step4_builder_nav.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0), before=self.step4_dataset_top)
    self.step4_builder_nav.grid_columnconfigure(0, weight=0)
    self.step4_builder_nav.grid_columnconfigure(1, weight=1)
    self.step4_builder_nav.grid_columnconfigure(2, weight=0)

    self.btn_step4_back = ttk.Button(
        self.step4_builder_nav,
        text="Wstecz",
        command=self._step4_dataset_go_back,
        style="WorkflowCard.TButton"
    )
    self.btn_step4_back.grid(row=0, column=0, sticky="w")
    self.btn_step4_back.configure(padding=(8, 2), width=NAV_BUTTON_WIDTH)

    self.btn_toggle_step4_log = ttk.Button(
        self.step4_builder_tools,
        text="Pokaż terminal",
        command=self._toggle_step4_builder_log
    )
    self.btn_toggle_step4_log.pack_forget()

    self.btn_step4_next_frame = tk.Frame(self.step4_builder_nav, bd=0, highlightthickness=0)
    self.btn_step4_next_frame.grid(row=0, column=2, sticky="e", padx=(10, 0))

    self.btn_step4_next = ttk.Button(
        self.btn_step4_next_frame,
        text="Dalej: Trening i analiza modelu znaków",
        command=self._step4_dataset_go_next,
        style="Accent.TButton"
    )
    self.btn_step4_next.pack()
    self.btn_step4_next.configure(text="Dalej do treningu", padding=(8, 2), width=NAV_BUTTON_WIDTH)

    HELP.bind_help(self.step4_route_panel_frame, "tr_route_panel")
    HELP.bind_help(self.step4_plate_route_card, "tr_route_plate")
    HELP.bind_help(self.step4_char_route_card, "tr_route_char")
    HELP.bind_help(self.btn_choose_plate, "tr_route_plate")
    HELP.bind_help(self.btn_choose_char, "tr_route_char")
    HELP.bind_help(self.btn_toggle_step4_log, "tr_builder_log")
    HELP.bind_help(self.btn_hide_step4_log, "tr_builder_log")
    HELP.bind_help(self.btn_step4_next, "tr_builder_next")

    if CAMPAIGN.get_active_project_name():
        initial_mode = self.get_campaign_training_target()
    else:
        initial_mode = "plate"
    if initial_mode not in ("char", "plate"):
        initial_mode = "plate"

    self._step4_builder_log_visible = False
    self._step4_route_selected = True
    self._step4_train_unlocked = True
    self._set_step4_builder_log_visibility(False)
    self._set_step4_dataset_mode(initial_mode, show_locked_message=False)
    try:
        def sync_dataset_canvas_after_first_paint():
            try:
                self._sync_dataset_mode_canvas_width()
                self._sync_dataset_mode_scrollregion()
            except Exception:
                pass

        self.frame.after_idle(sync_dataset_canvas_after_first_paint)
    except Exception:
        pass
    elapsed_ms = int((time.perf_counter() - build_started) * 1000)
    if elapsed_ms >= 300:
        try:
            logger.info(f"[Z4/PZ1 PERF] build_dataset_tab total={elapsed_ms}ms mode={initial_mode}")
        except Exception:
            pass
