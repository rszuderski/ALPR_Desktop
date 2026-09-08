#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 theme application helpers extracted from tab_training.py."""

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
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None


def apply_theme(self):
    palette = getattr(self.app, "palette", {})
    console_border = palette.get("console_border", palette.get("border", "#3c3c3c"))
    section_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.72,
    )

    try:
        style = ttk.Style()
        style.configure(
            "TrainDatasetSection.TLabelframe",
            background=palette.get("panel", "#252526"),
            bordercolor=section_border,
            lightcolor=section_border,
            darkcolor=section_border,
            relief=tk.SOLID,
        )
        style.configure(
            "TrainDatasetSection.TLabelframe.Label",
            font=("Segoe UI", 12, "bold"),
            foreground=palette.get("fg", "#f3f3f3"),
            background=palette.get("panel", "#252526"),
        )
    except Exception:
        pass

    try:
        self._configure_train_progress_styles()
    except Exception:
        pass

    for label_name in ("free_training_route_title_lbl",):
        label = getattr(self, label_name, None)
        if label is None or not hasattr(label, "apply_theme"):
            continue
        try:
            label.apply_theme()
        except Exception:
            pass

    for label_name in (
        "train_dataset_title_lbl",
        "train_dataset_intro_lbl",
        "dataset_variant_title_lbl",
        "train_dataset_selected_path_lbl",
    ):
        label = getattr(self, label_name, None)
        if label is None:
            continue
        try:
            label.configure(
                bg=palette.get("panel", "#252526"),
                fg=palette.get("fg", "#f3f3f3"),
            )
        except Exception:
            pass

    variant_row = getattr(self, "dataset_variant_row", None)
    if variant_row is not None:
        try:
            variant_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
            border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
            variant_row.configure(bg=variant_bg, highlightbackground=border, highlightcolor=border)
        except Exception:
            pass
    train_base_shell = getattr(self, "train_base_shell", None)
    base_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    if train_base_shell is not None:
        try:
            base_border = section_border
            train_base_shell.configure(bg=base_bg, highlightbackground=base_border, highlightcolor=base_border)
        except Exception:
            pass
    for label_name in ("train_base_title_lbl", "train_base_caption_lbl"):
        label = getattr(self, label_name, None)
        if label is None:
            continue
        try:
            label.configure(
                background=base_bg,
                foreground=palette.get("fg", "#f3f3f3") if label_name.endswith("title_lbl") else palette.get("muted", "#c7c7c7"),
            )
        except Exception:
            try:
                label.configure(
                    bg=base_bg,
                    fg=palette.get("fg", "#f3f3f3") if label_name.endswith("title_lbl") else palette.get("muted", "#c7c7c7"),
                )
            except Exception:
                pass
    context_row = getattr(self, "train_base_context_row", None)
    if context_row is not None:
        try:
            context_row.configure(bg=base_bg)
        except Exception:
            pass
    try:
        chip_bg = blend_hex_colors(palette.get("accent", "#0e639c"), base_bg, 0.84)
        for chip in (getattr(self, "train_base_context_chips", {}) or {}).values():
            try:
                chip.configure(
                    bg=chip_bg,
                    fg=palette.get("fg", "#f3f3f3"),
                    highlightbackground=section_border,
                    highlightcolor=section_border,
                )
            except Exception:
                pass
    except Exception:
        pass
    combo_label = getattr(self, "train_base_combo_lbl", None)
    if combo_label is not None:
        try:
            combo_label.configure(background=base_bg, foreground=palette.get("muted", "#c7c7c7"))
        except Exception:
            pass
    summary_shell = getattr(self, "train_base_summary_shell", None)
    if summary_shell is not None:
        try:
            summary_shell.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=section_border,
                highlightcolor=section_border,
            )
        except Exception:
            pass
    try:
        for row_index, cell, cell_kind in getattr(self, "train_base_summary_cells", []) or []:
            row_bg = (
                palette.get("panel", "#252526")
                if int(row_index) % 2 == 0
                else palette.get("panel_alt", palette.get("panel", "#252526"))
            )
            cell.configure(
                bg=row_bg,
                fg=palette.get("muted", "#c7c7c7") if cell_kind == "key" else palette.get("fg", "#f3f3f3"),
            )
    except Exception:
        pass
    for label_name in ("dataset_variant_title_lbl", "train_dataset_selected_path_lbl"):
        label = getattr(self, label_name, None)
        if label is None:
            continue
        try:
            label.configure(
                bg=palette.get("panel_alt", palette.get("panel", "#252526")),
                fg=palette.get("muted", "#c7c7c7"),
            )
        except Exception:
            pass

    try:
        self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
    except Exception:
        pass
    try:
        self._configure_train_progress_styles()
    except Exception:
        pass

    for widget_name in ("step4_builder_log_text", "train_log_console"):
        widget = getattr(self, widget_name, None)
        if widget is None:
            continue
        try:
            self.app.style_text_widget(widget, role="console")
        except Exception:
            pass

    try:
        self.app.style_listbox_widget(self.plots_list, bordercolor=console_border)
    except Exception:
        pass

    try:
        self.app.style_canvas_widget(
            self.plot_canvas,
            background=palette.get("panel", "#252526"),
            bordercolor=console_border
        )
    except Exception:
        pass

    try:
        for canvas_name in ("train_left_canvas", "ds_mode_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas is None:
                continue
            canvas.configure(
                bg=palette.get("panel", "#252526"),
                highlightbackground=console_border,
                highlightcolor=console_border
            )
    except Exception:
        pass

    for frame_name in (
        "step4_route_panel_frame",
        "btn_step4_next_frame",
        "btn_step4_create_frame",
        "btn_step4_split_frame",
        "btn_step4_start_train_frame",
        "btn_step4_finish_frame",
    ):
        frame = getattr(self, frame_name, None)
        if frame is None:
            continue
        try:
            action_inner = getattr(frame, "_step4_action_inner", None)
            if action_inner is not None:
                action_border = blend_hex_colors(
                    palette.get("success", "#2fa36b"),
                    palette.get("panel_border", palette.get("border", "#3c3c3c")),
                    0.68,
                )
                action_bg = blend_hex_colors(
                    palette.get("panel_alt", palette.get("panel", "#252526")),
                    palette.get("panel", "#252526"),
                    0.60,
                )
                frame.configure(bg=action_border, bd=0, highlightthickness=0)
                action_inner.configure(bg=action_bg)
                setattr(frame, "_step4_action_bg", action_bg)
                for child in action_inner.winfo_children():
                    try:
                        if isinstance(child, tk.Label):
                            child.configure(bg=action_bg)
                        elif isinstance(child, tk.Frame):
                            child.configure(bg=action_bg)
                    except Exception:
                        pass
                continue
            bg = palette.get("bg", "#1e1e1e") if frame_name == "step4_route_panel_frame" else palette.get("panel", "#252526")
            self.app.style_guidance_frame(frame, background=bg)
        except Exception:
            pass

    try:
        start_shell = getattr(self, "train_start_gate_shell", None)
        start_header = getattr(self, "train_start_gate_header", None)
        start_title = getattr(self, "train_start_gate_title_lbl", None)
        start_hint = getattr(self, "train_start_gate_hint_lbl", None)
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", panel)
        start_bg = panel
        start_border = section_border
        if start_shell is not None:
            start_shell.configure(
                bg=start_bg,
                highlightbackground=start_border,
                highlightcolor=start_border,
            )
        if start_header is not None:
            start_header.configure(bg=start_bg)
        if start_title is not None:
            start_title.configure(bg=start_bg, fg=palette.get("fg", "#f3f3f3"))
        if start_hint is not None:
            start_hint.configure(bg=start_bg, fg=palette.get("muted", "#c7c7c7"))
        start_frame = getattr(self, "btn_step4_start_train_frame", None)
        if start_frame is not None:
            start_frame.configure(bg=start_bg)
    except Exception:
        pass

    try:
        summary_shell = getattr(self, "train_run_summary_shell", None)
        summary_title_row = getattr(self, "train_run_summary_title_row", None)
        summary_title = getattr(self, "train_run_summary_title_lbl", None)
        summary_grid = getattr(self, "train_run_summary_grid", None)
        summary_rows_frame = getattr(self, "train_run_summary_rows_frame", None)
        summary_header_key = getattr(self, "train_run_summary_header_key_lbl", None)
        summary_header_value = getattr(self, "train_run_summary_header_value_lbl", None)
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", "#2d2d30")
        summary_bg = panel
        summary_title_bg = panel
        raw_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        summary_header_bg = blend_hex_colors(panel_alt, panel, 0.42)
        summary_border = blend_hex_colors(raw_border, panel, 0.35)
        if summary_shell is not None:
            summary_shell.configure(
                bg=summary_bg,
                highlightbackground=summary_border,
                highlightcolor=summary_border,
            )
        if summary_title_row is not None:
            summary_title_row.configure(bg=summary_title_bg)
        if summary_title is not None:
            summary_title.configure(
                bg=summary_title_bg,
                fg=palette.get("fg", "#f3f3f3"),
            )
        if summary_grid is not None:
            summary_grid.configure(
                bg=summary_bg,
                highlightbackground=summary_border,
                highlightcolor=summary_border,
            )
        if summary_rows_frame is not None:
            summary_rows_frame.configure(bg=summary_border)
        for header_widget in (summary_header_key, summary_header_value):
            if header_widget is not None:
                header_widget.configure(
                    bg=summary_header_bg,
                    fg=palette.get("fg", "#f3f3f3"),
                )
        for row in list(getattr(self, "_train_run_summary_row_widgets", []) or []):
            row_frame = row.get("frame")
            key_widget = row.get("key")
            value_widget = row.get("value")
            row_index = int(row.get("row_index", 0) or 0)
            row_bg = panel if row_index % 2 == 0 else blend_hex_colors(panel_alt, panel, 0.76)
            key_bg = row_bg
            value_bg = row_bg
            if row_frame is not None:
                row_frame.configure(bg=summary_border)
            if key_widget is not None:
                key_widget.configure(
                    bg=key_bg,
                    fg=palette.get("muted", "#c7c7c7"),
                )
            if value_widget is not None:
                value_widget.configure(
                    bg=value_bg,
                    fg=palette.get("fg", "#f3f3f3"),
                )
    except Exception:
        pass

    try:
        if hasattr(self, "free_training_route_host"):
            self.free_training_route_host.configure(style="TLabelframe")
    except Exception:
        pass

    try:
        cards = getattr(self, "_free_training_route_cards", {})
        for widgets in cards.values():
            frame = widgets.get("frame")
            if frame is not None:
                frame.configure(bg=palette.get("panel_alt", "#2d2d30"))
    except Exception:
        pass

    try:
        if hasattr(self, "free_training_route_cards_row"):
            self.free_training_route_cards_row.configure(bg=palette.get("panel", "#252526"))
    except Exception:
        pass

    menu = getattr(self, "history_context_menu", None)
    if menu is not None:
        try:
            menu.configure(
                bg=palette.get("panel", "#252526"),
                fg=palette.get("fg", "#f3f3f3"),
                activebackground=palette.get("button_hover", "#37373d"),
                activeforeground=palette.get("fg", "#f3f3f3"),
                bd=0,
                relief=tk.FLAT,
            )
        except Exception:
            pass

    try:
        z4_training_runtime._configure_history_tree_tags(self)
    except Exception:
        pass

    try:
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
        for frame_name in (
            "campaign_training_result_shell",
            "campaign_training_result_title_row",
            "campaign_training_result_pick_row",
        ):
            frame = getattr(self, frame_name, None)
            if frame is not None:
                frame.configure(bg=result_bg)
        result_shell = getattr(self, "campaign_training_result_shell", None)
        if result_shell is not None:
            result_shell.configure(
                highlightbackground=result_border,
                highlightcolor=result_border,
            )
        result_title = getattr(self, "campaign_training_result_title_lbl", None)
        if result_title is not None:
            result_title.configure(
                bg=result_bg,
                fg=palette.get("success", "#2ecc71"),
                font=("Segoe UI Semibold", 12),
            )
        result_copy = getattr(self, "campaign_training_result_copy_lbl", None)
        if result_copy is not None:
            result_copy.configure(
                bg=result_bg,
                fg=palette.get("muted", "#c7c7c7"),
            )
    except Exception:
        pass

    for line in getattr(self, "_train_left_section_separators", []):
        if line is None:
            continue
        try:
            if isinstance(line, dict):
                host = line.get("host")
                accent = line.get("accent")
                shadow = line.get("shadow")
                if host is not None:
                    host.configure(bg=palette.get("panel", "#252526"))
                if accent is not None:
                    accent.configure(bg=palette.get("surface_info", palette.get("accent", "#0e639c")))
                if shadow is not None:
                    shadow.configure(bg=palette.get("panel_border", palette.get("border", "#3c3c3c")))
            else:
                line.configure(bg=palette.get("surface_info", palette.get("accent", "#0e639c")))
        except Exception:
            pass

    self._refresh_free_training_route_cards()
    try:
        self._refresh_step4_dataset_mode_ui()
    except Exception:
        pass
    self._apply_training_recommendation_table_theme()
    try:
        self._refresh_training_dataset_quality_summary()
    except Exception:
        pass
    self._style_analysis_dialog()
