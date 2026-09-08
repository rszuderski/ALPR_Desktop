#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 status panels and process-log UI helpers extracted from tab_annotation.py."""

import copy
import csv
import json
from collections import deque
import os
import re
import tkinter as tk
import tkinter.font as tkfont
import xml.etree.ElementTree as ET
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import datetime
import threading
import logging
import math
import shutil
import time
import queue
import numpy as np

import cv2
from PIL import Image
from .inertial_scroll import InertialScrollController
from .zoomable_canvas import ZoomableCanvas
from .section_header_label import SectionHeaderLabel
from . import z2_workflow_methods
from . import z2_canvas_overlays
from . import z2_canvas_interaction
from . import z2_preview_editor
from . import z2_preview_state
from . import z2_miniflow_runtime
from . import z2_model_runtime
from . import z2_session_runtime
from . import z2_context_runtime
from . import z2_run_io_runtime
from .z2_main_widgets import create_annotation_widgets
from .z2_auto_scope_modal import prompt_plate_auto_scope_choice
from .z2_canvas_metrics_ui import (
    render_preview_metrics_grab_handle as z2_render_preview_metrics_grab_handle,
    render_preview_metrics_table as z2_render_preview_metrics_table,
)
from .z2_overlay_dock_ui import (
    get_preview_overlay_dock_theme as z2_get_preview_overlay_dock_theme,
    render_preview_campaign_gate_overlay as z2_render_preview_campaign_gate_overlay,
    render_preview_image_status_overlay as z2_render_preview_image_status_overlay,
    render_preview_overlay_dock as z2_render_preview_overlay_dock,
)
from .z2_legend_assets import (
    clear_preview_legend_image_cache as z2_clear_preview_legend_image_cache,
    get_preview_legend_compass_photo as z2_get_preview_legend_compass_photo,
    get_preview_legend_group_shell_photo as z2_get_preview_legend_group_shell_photo,
    get_preview_legend_interaction_marker_photo as z2_get_preview_legend_interaction_marker_photo,
    get_preview_legend_keycap_photo as z2_get_preview_legend_keycap_photo,
    get_preview_legend_shell_photo as z2_get_preview_legend_shell_photo,
    get_preview_legend_text_color as z2_get_preview_legend_text_color,
    hex_to_rgba as z2_hex_to_rgba,
    legend_color_is_light as z2_legend_color_is_light,
    normalize_preview_legend_interaction as z2_normalize_preview_legend_interaction,
    trim_preview_legend_image_cache as z2_trim_preview_legend_image_cache,
)
from .z2_legend_ui import (
    build_preview_controls_context_rows as z2_build_preview_controls_context_rows,
    build_preview_legend_sections as z2_build_preview_legend_sections,
    get_preview_controls_legend_target_height as z2_get_preview_controls_legend_target_height,
    get_preview_controls_legend_target_width as z2_get_preview_controls_legend_target_width,
    get_preview_legend_theme as z2_get_preview_legend_theme,
    preview_controls_row_is_multi_plate_badge as z2_preview_controls_row_is_multi_plate_badge,
    refresh_preview_controls_legend as z2_refresh_preview_controls_legend,
)
from .z2_model_quality_ui import (
    auto_model_metric_tone as z2_auto_model_metric_tone,
    describe_auto_model_quality as z2_describe_auto_model_quality,
    format_model_quality_epoch as z2_format_model_quality_epoch,
    format_model_quality_number as z2_format_model_quality_number,
    get_model_identity_caption as z2_get_model_identity_caption,
    model_quality_float as z2_model_quality_float,
    model_quality_value as z2_model_quality_value,
    render_auto_annotation_model_quality_table as z2_render_auto_annotation_model_quality_table,
    shorten_model_quality_text as z2_shorten_model_quality_text,
)
from .z2_actions import Z2ActionContext, build_z2_primary_actions, build_z2_secondary_actions
from .z2_campaign_flow import (
    apply_campaign_step2_workflow_preset,
    build_z2_cta_state_campaign,
    build_z2_layout_state_campaign,
    build_z2_left_panel_copy_payload_campaign,
    open_campaign_step2_entry as dispatch_open_campaign_step2_entry,
    open_existing_run_for_campaign_review,
    prepare_campaign_workflow_runtime,
)
from .z2_free_mode_flow import (
    AUTO_REVIEW_FOLLOWUP_TEXT,
    AUTO_REVIEW_FOLLOWUP_TITLE,
    MANUAL_REVIEW_FOLLOWUP_TEXT,
    MANUAL_REVIEW_FOLLOWUP_TITLE,
    build_z2_cta_state_free_mode,
    build_z2_layout_state_free_mode,
    build_z2_left_panel_copy_payload_free_mode,
    get_manual_review_history_display_entries as dispatch_get_manual_review_history_display_entries,
    jump_to_export_section as dispatch_jump_to_export_section,
    open_existing_run_for_manual_review as dispatch_open_existing_run_for_manual_review,
    prepare_free_mode_workflow_runtime,
    refresh_manual_review_history_ui as dispatch_refresh_manual_review_history_ui,
    remember_manual_review_run as dispatch_remember_manual_review_run,
    select_free_mode_route,
    set_manual_entry_mode as dispatch_set_manual_entry_mode,
)
from .z2_flow_models import Z2CopyPayload, Z2LeftPanelCopyContext
from .z2_shared_ui import (
    apply_z2_workflow_left_layout as dispatch_apply_z2_workflow_left_layout,
    apply_z2_workflow_cta_ui as dispatch_apply_z2_workflow_cta_ui,
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel

from ..config import CONFIG, logger, YOLO_AVAILABLE, AVAILABLE_DETECT_MODELS, SESSION
from ..annotators.runtime_factory import validate_pt_model_path_for_runtime
from ..icons import IconManager
from ..exporters import CVATExporter, ReportGenerator
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..project_cache import PROJECT_CACHE
from ..training import DatasetCreator
from ..utils import cleanup_gpu_memory, count_images_in_directory, format_duration, get_image_files, get_image_size
from ..data_models import Detection, AnnotationStatus, ImageAnnotation, AnnotationReport
from ..rectification.polygon_validator import PolygonValidator
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .canvas_progress_overlay import CanvasProgressOverlay

NAV_BUTTON_WIDTH = 18

def _redirect_logs(self):
    if getattr(self, "_annotation_log_handlers_attached", False):
        return

    def append_annotation_log(message: str):
        text = "" if message is None else str(message)
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"

        def update():
            try:
                if hasattr(self.app, "append_global_terminal"):
                    self.app.append_global_terminal(text.rstrip("\n"), source="Z2")
            except Exception:
                pass
            try:
                self.log_text.insert(tk.END, text)
                self.log_text.see(tk.END)
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            try:
                self.frame.after_idle(update)
            except Exception:
                try:
                    update()
                except Exception:
                    pass
        else:
            self._post_to_ui(update)

    class TextHandler(logging.Handler):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def emit(self, record):
            try:
                msg = self.format(record)
                if msg:
                    append_annotation_log(msg)
            except Exception:
                pass

    formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')

    self._annotation_app_log_handler = TextHandler(self)
    self._annotation_app_log_handler.setFormatter(formatter)
    logger.addHandler(self._annotation_app_log_handler)

    try:
        self._annotation_ultralytics_log_handler = TextHandler(self)
        self._annotation_ultralytics_log_handler.setFormatter(formatter)
        self._annotation_ultralytics_logger = logging.getLogger("ultralytics")
        self._annotation_ultralytics_logger.addHandler(self._annotation_ultralytics_log_handler)
    except Exception:
        pass

    self._annotation_log_handlers_attached = True


def apply_theme(self, *args, **kwargs):
    return z2_workflow_methods.apply_theme(self, *args, **kwargs)


def _set_inline_label_state(self, widget, text: str | None = None, tone: str = "neutral", emphasis: bool = False) -> bool:
    if widget is None or not isinstance(widget, tk.Label):
        return False

    palette = getattr(self.app, "palette", {})
    bg = palette.get("bg", "#1f1f1f")

    try:
        parent = widget.nametowidget(widget.winfo_parent())
    except Exception:
        parent = None

    for candidate in (parent, widget):
        if candidate is None:
            continue
        try:
            bg_candidate = candidate.cget("bg")
            if bg_candidate:
                bg = bg_candidate
                break
        except Exception:
            pass
        try:
            bg_candidate = candidate.cget("background")
            if bg_candidate:
                bg = bg_candidate
                break
        except Exception:
            pass
        try:
            style_name = str(candidate.cget("style") or "").strip()
            if style_name:
                bg_candidate = self.app.style.lookup(style_name, "background")
                if bg_candidate:
                    bg = bg_candidate
                    break
        except Exception:
            pass
        try:
            bg_candidate = ttk.Style().lookup(candidate.winfo_class(), "background")
            if bg_candidate:
                bg = bg_candidate
                break
        except Exception:
            pass

    tone_key = str(tone or "").strip().lower()
    fg = {
        "default": palette.get("fg", "#f3f3f3"),
        "neutral": palette.get("muted", "#9a9a9a"),
        "muted": palette.get("muted", "#9a9a9a"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f39c12"),
        "error": palette.get("error", "#e74c3c"),
    }.get(tone_key, palette.get("muted", "#9a9a9a"))

    config_kwargs = {
        "bg": bg,
        "fg": fg,
    }
    if text is not None:
        config_kwargs["text"] = text

    widget._inline_tone = tone_key
    widget._inline_emphasis = bool(emphasis)
    widget.config(**config_kwargs)
    return True


def _set_approve_hint_box_state(self, tone: str = "muted"):
    box = getattr(self, "approve_hint_box", None)
    if box is None:
        return

    palette = getattr(self.app, "palette", {})
    tone_key = str(tone or "").strip().lower()
    bg = palette.get("panel_alt", palette.get("panel", palette.get("bg", "#1f1f1f")))
    border = {
        "default": palette.get("border", "#3a3a3a"),
        "neutral": palette.get("border", "#3a3a3a"),
        "muted": palette.get("border", "#3a3a3a"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f39c12"),
        "error": palette.get("error", "#e74c3c"),
    }.get(tone_key, palette.get("border", "#3a3a3a"))

    try:
        box.configure(bg=bg, highlightbackground=border, highlightcolor=border)
    except Exception:
        pass


def _set_approve_context_box_state(self, tone: str = "info"):
    box = getattr(self, "approve_context_box", None)
    if box is None:
        return

    palette = getattr(self.app, "palette", {})
    tone_key = str(tone or "").strip().lower()
    bg = blend_hex_colors(
        palette.get("panel_alt", palette.get("panel", "#252526")),
        palette.get("field", "#3c3c3c"),
        0.22,
    )
    border = {
        "default": palette.get("border", "#3a3a3a"),
        "neutral": palette.get("border", "#3a3a3a"),
        "muted": palette.get("border", "#3a3a3a"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f39c12"),
        "error": palette.get("error", "#e74c3c"),
    }.get(tone_key, palette.get("border", "#3a3a3a"))

    try:
        box.configure(bg=bg, highlightbackground=border, highlightcolor=border)
    except Exception:
        pass


def _render_compact_info_table(self, *args, **kwargs):
    return z2_workflow_methods._render_compact_info_table(self, *args, **kwargs)


def _set_plate_model_info_box_state(self, tone: str = "info"):
    box = getattr(self, "workflow_plate_model_info_box", None)
    if box is None:
        return

    palette = getattr(self.app, "palette", {})
    tone_key = str(tone or "").strip().lower()
    bg = blend_hex_colors(
        palette.get("panel_alt", palette.get("panel", "#252526")),
        palette.get("field", "#3c3c3c"),
        0.30,
    )
    border = {
        "default": palette.get("border", "#3a3a3a"),
        "neutral": palette.get("border", "#3a3a3a"),
        "muted": palette.get("border", "#3a3a3a"),
        "info": palette.get("info", palette.get("accent", "#4aa3ff")),
        "success": palette.get("success", "#2ecc71"),
        "warning": palette.get("warning", "#f39c12"),
        "error": palette.get("error", "#e74c3c"),
    }.get(tone_key, palette.get("border", "#3a3a3a"))

    try:
        box.configure(bg=bg, highlightbackground=border, highlightcolor=border)
    except Exception:
        pass
    try:
        self.workflow_plate_model_info_title_lbl.configure(bg=bg, fg=palette.get("fg", "#f3f3f3"))
    except Exception:
        pass
    self._set_inline_label_state(
        getattr(self, "workflow_plate_model_info_lbl", None),
        tone=("success" if tone_key == "success" else "muted" if tone_key == "muted" else "info"),
        emphasis=False,
    )


def _collect_preview_category_breakdown(self) -> dict[str, int]:
    annotations = list(self.current_annotations or [])
    manual_images = sum(1 for ann in annotations if self._preview_annotation_is_manually_corrected(ann))
    auto_images = sum(
        1
        for ann in annotations
        if getattr(ann, "is_successful", False) and not self._preview_annotation_is_manually_corrected(ann)
    )
    problem_images = max(0, int(len(annotations)) - int(manual_images) - int(auto_images))
    approved_images = sum(1 for ann in annotations if self._preview_annotation_is_explicitly_approved(ann))
    return {
        "total": int(len(annotations)),
        "manual": int(manual_images),
        "auto": int(auto_images),
        "problem": int(problem_images),
        "approved": int(approved_images),
    }


def _refresh_approval_breakdown_canvas(self):
    box = getattr(self, "approve_breakdown_box", None)
    canvas = getattr(self, "approve_breakdown_canvas", None)
    label = getattr(self, "approve_breakdown_lbl", None)
    if box is None or canvas is None:
        return

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#1f1f1f"))
    panel_border = palette.get("panel_border", palette.get("border", "#3a3a3a"))
    manual_color = palette.get("warning", "#f39c12")
    auto_color = palette.get("success", "#27ae60")
    problem_color = palette.get("error", "#c0392b")
    approved_color = palette.get("accent", "#0e639c")
    fg_color = palette.get("fg", "#f3f3f3")
    muted_color = palette.get("muted", "#c7c7c7")

    try:
        box.configure(bg=panel_bg, highlightbackground=panel_border, highlightcolor=panel_border)
        canvas.configure(bg=panel_bg)
    except Exception:
        pass

    breakdown = self._collect_preview_category_breakdown()
    total = int(breakdown.get("total", 0) or 0)
    manual_images = int(breakdown.get("manual", 0) or 0)
    auto_images = int(breakdown.get("auto", 0) or 0)
    problem_images = int(breakdown.get("problem", 0) or 0)
    approved_images = int(breakdown.get("approved", 0) or 0)

    try:
        width = max(120, int(canvas.winfo_width() or 320))
    except Exception:
        width = 320
    height = 46

    try:
        canvas.delete("all")
    except Exception:
        return

    left = 8
    right = max(left + 1, width - 8)
    bar_width = max(1, right - left)

    if total <= 0:
        try:
            canvas.create_rectangle(left, 10, right, 24, fill=panel_bg, outline=panel_border, width=1)
            canvas.create_text(
                width / 2,
                17,
                text="Brak aktywnej puli obrazów",
                fill=muted_color,
                font=("Segoe UI", 8),
            )
        except Exception:
            pass
        if label is not None:
            try:
                self._set_inline_label_state(
                    label,
                    text="M 0 | A 0 | -- 0 | OK 0",
                    tone="muted",
                    emphasis=False,
                )
            except Exception:
                pass
        return

    segments = [
        ("M", manual_images, manual_color),
        ("A", auto_images, auto_color),
        ("--", problem_images, problem_color),
    ]
    cursor = float(left)
    for badge, count, color in segments:
        if count <= 0:
            continue
        segment_width = max(1.0, bar_width * (float(count) / float(total)))
        x1 = cursor
        x2 = right if badge == segments[-1][0] else min(float(right), cursor + segment_width)
        try:
            canvas.create_rectangle(x1, 10, x2, 24, fill=color, outline="")
            if (x2 - x1) >= 24:
                canvas.create_text(
                    (x1 + x2) / 2,
                    17,
                    text=f"{badge} {count}",
                    fill=("#1b1b1b" if badge == "M" else panel_bg),
                    font=("Segoe UI", 8, "bold"),
                )
        except Exception:
            pass
        cursor = x2

    try:
        canvas.create_rectangle(left, 10, right, 24, outline=panel_border, width=1)
    except Exception:
        pass

    ok_ratio = max(0.0, min(1.0, float(approved_images) / float(total)))
    ok_right = left + (bar_width * ok_ratio)
    try:
        canvas.create_rectangle(left, 31, right, 38, fill=panel_bg, outline=panel_border, width=1)
        if approved_images > 0:
            canvas.create_rectangle(left, 31, ok_right, 38, fill=approved_color, outline="")
            if ok_right - left >= 32:
                canvas.create_text(
                    (left + ok_right) / 2,
                    34,
                    text=f"OK {approved_images}",
                    fill=panel_bg,
                    font=("Segoe UI", 7, "bold"),
                )
    except Exception:
        pass

    if label is not None:
        try:
            self._set_inline_label_state(
                label,
                text=f"M {manual_images} | A {auto_images} | -- {problem_images} | OK {approved_images}",
                tone="muted",
                emphasis=False,
            )
        except Exception:
            pass


def _refresh_manual_stage_export_box_style(self):
    box = getattr(self, "manual_stage_export_box", None)
    if box is None:
        return

    palette = getattr(self.app, "palette", {})
    box_bg = palette.get("panel", palette.get("bg", "#1f1f1f"))
    border = palette.get("panel_border", palette.get("border", "#3a3a3a"))
    title_fg = palette.get("muted", "#9a9a9a")
    muted = palette.get("muted_dim", palette.get("muted", "#9a9a9a"))

    try:
        box.configure(bg=box_bg, highlightbackground=border, highlightcolor=border)
    except Exception:
        pass

    for label, color in (
        (getattr(self, "manual_stage_export_title_lbl", None), title_fg),
        (getattr(self, "manual_stage_export_help_lbl", None), muted),
    ):
        if label is None:
            continue
        try:
            label.configure(bg=box_bg, fg=color)
        except Exception:
            pass


def _set_status_label_state(self, text: str, tone: str = "neutral"):
    if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
        updater = getattr(self, "_update_plate_auto_scope_progress_modal", None)
        if callable(updater):
            try:
                updater(meta_text=str(text or "").strip())
            except Exception:
                pass
        return

    if not self._set_inline_label_state(self.status_label, text=text, tone=tone, emphasis=True):
        style_map = {
            "neutral": "PanelStatusNeutral.TLabel",
            "info": "PanelStatusInfo.TLabel",
            "success": "PanelStatusSuccess.TLabel",
            "warning": "PanelStatusWarning.TLabel",
            "error": "PanelStatusError.TLabel",
        }
        self.status_label.config(
            text=text,
            style=style_map.get(str(tone or "").lower(), "PanelStatusNeutral.TLabel")
        )


def _set_annotation_process_log_visibility(self, visible: bool):
    if not hasattr(self, "annotation_log_overlay"):
        return

    self._annotation_log_visible = bool(visible)
    try:
        self.annotation_log_overlay.place_forget()
    except Exception:
        pass

    if self._annotation_log_visible:
        try:
            if hasattr(self.app, "show_global_terminal"):
                self.app.show_global_terminal()
        except Exception:
            pass

    if hasattr(self, "btn_toggle_annotation_log"):
        try:
            self.btn_toggle_annotation_log.configure(text="Terminal")
        except Exception:
            pass
    return

    if self._annotation_log_visible:
        try:
            self.annotation_log_overlay.place(
                relx=0.015,
                rely=0.02,
                relwidth=0.97,
                relheight=0.96,
            )
            self.annotation_log_overlay.lift()
        except Exception:
            pass
        try:
            self.annotation_log_overlay.update_idletasks()
            self.annotation_log_frame.update_idletasks()
            self.annotation_log_host.update_idletasks()
        except Exception:
            pass
        if hasattr(self, "btn_toggle_annotation_log"):
            self.btn_toggle_annotation_log.configure(text="Ukryj terminal")
    else:
        try:
            self.annotation_log_overlay.place_forget()
        except Exception:
            pass
        if hasattr(self, "btn_toggle_annotation_log"):
            self.btn_toggle_annotation_log.configure(text="Pokaż terminal")

