#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 campaign/free-mode context runtime extracted from tab_annotation.py."""

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

def ensure_campaign_context_ready_for_active_project(self, *, force: bool = False) -> bool:
    if bool(getattr(self, "_campaign_project_restore_in_progress", False)):
        return False

    try:
        from ..campaign_manager import CAMPAIGN
        active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    except Exception:
        active_project = ""

    if not active_project:
        self._campaign_context_project_name = ""
        return False

    if self._is_free_mode_session_context():
        return False

    current_project = str(getattr(self, "_campaign_context_project_name", "") or "").strip()
    if not force and current_project == active_project:
        return False

    return bool(self.restore_campaign_context_from_project())


def ensure_free_mode_session_preview_ready(self, *, force: bool = False) -> bool:
    if not self._is_free_mode_session_context():
        return False
    if getattr(self, "is_processing", False):
        return False

    if bool(getattr(self, "current_annotations", None)) and not force:
        return True

    try:
        has_saved_run = (
            self._resolve_safe_annotation_run_dir(getattr(self, "last_staging_run_dir", None), require_xml=True) is not None
            or self._resolve_safe_annotation_run_dir(str(self.plate_dataset_run_var.get() or "").strip(), require_xml=True) is not None
        )
    except Exception:
        has_saved_run = False

    # Startup/route exit must stay clean; explicit "continue/import" paths open old runs.
    if not has_saved_run:
        return False

    try:
        restored = bool(self._restore_preview_from_session_run())
    except Exception as e:
        logger.debug(f"Nie udalo sie przywrocic podgladu Z2 z sesji free mode: {e}")
        return False

    if restored:
        try:
            route = self._get_workflow_route()
            manifest = self._load_annotation_run_manifest(getattr(self, "current_annotation_run_dir", None))
            manual_template_like = bool(self._annotation_run_manifest_is_manual_template(manifest))
            manual_review = route == "manual" or (not route and manual_template_like)
            if manual_review:
                self.workflow_route_var.set("manual")
                self.free_mode_screen_var.set("manual_review")
                self._manual_review_active = True
                # An auto result can also be opened explicitly for manual correction.
                from_auto = bool(getattr(self, "_manual_review_from_auto", False)) and not manual_template_like
                self._manual_review_from_auto = from_auto
                self._manual_review_origin_route = "auto" if from_auto else "manual"
                self._last_completed_workflow_route = self._manual_review_origin_route
            else:
                self.workflow_route_var.set("auto")
                self.free_mode_screen_var.set("auto_summary")
                self._manual_review_active = False
                self._manual_review_from_auto = False
                self._manual_review_origin_route = ""
                self._last_completed_workflow_route = "auto"
            try:
                if self._get_z2_thematic_route() == "auto":
                    self._restore_plate_model_selection_from_active_run()
            except Exception:
                pass
            self._repair_restored_z2_workflow_completion_state()
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            self._update_preview_edit_status("Przywrócono ostatni run anotacji Z2 z poprzedniej sesji.")
        except Exception:
            pass

    return restored


def clear_campaign_context(self, *, restore_free_mode_preview: bool = True):
    """
    Przywraca neutralny stan zakĹ‚adki Autoanotacji po wyjĹ›ciu z projektu
    i czyĹ›ci wszystkie artefakty poprzedniego projektu z UI.
    """
    self._reset_campaign_step2_runtime_state()
    try:
        self._hide_campaign_step2_splash()
    except Exception:
        pass
    self._set_campaign_paths_lock_state(False)

    try:
        self.project_paths_info_var.set("")
    except Exception:
        pass

    try:
        self.project_paths_rel_var.set("")
    except Exception:
        pass

    self.current_annotations = []
    self._clear_preview_editor_state(clear_dirty=True)
    self.is_processing = False

    try:
        self.current_input_dir = Path(self.input_dir_var.get().strip())
    except Exception:
        self.current_input_dir = None

    try:
        self.preview_listbox.delete(0, tk.END)
    except Exception:
        pass

    try:
        self.preview_canvas.delete("all")
    except Exception:
        pass

    try:
        self.log_text.delete("1.0", tk.END)
    except Exception:
        pass

    try:
        self.progress.configure(value=0)
    except Exception:
        pass

    try:
        self._set_status_label_state("Gotowy do uruchomienia", "neutral")
    except Exception:
        pass

    try:
        self._set_post_annotation_hint("")
    except Exception:
        pass

    try:
        self._set_progress_counters(0, 0, 0)
    except Exception:
        pass

    try:
        self.start_btn.config(state=tk.NORMAL)
    except Exception:
        pass

    try:
        self.stop_btn.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self.approve_btn.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self._set_annotation_process_log_visibility(False)
    except Exception:
        pass

    try:
        snapshot = getattr(self, "_pre_campaign_free_mode_snapshot", None)
        self._pre_campaign_free_mode_snapshot = None
        self._apply_free_mode_session_snapshot(
            session_state=snapshot,
            restore_preview=bool(restore_free_mode_preview and snapshot is None),
        )
        if snapshot is not None:
            try:
                self.flush_free_mode_session_state()
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Nie udalo sie przywrocic ostatniego stanu Z2 po wyjsciu z projektu: {e}")


def _format_project_relative_path(self, path_value: str) -> str:
    try:
        from ..campaign_manager import CAMPAIGN
        root = CAMPAIGN.get_active_project_root_dir()
        p = Path(path_value)

        if root is not None:
            try:
                return str(p.relative_to(root))
            except Exception:
                pass

        return str(p)
    except Exception:
        return str(path_value)


def _set_campaign_paths_lock_state(self, locked: bool):
    self._campaign_paths_locked = bool(locked)

    try:
        self.input_dir_entry.configure(state=("readonly" if locked else "normal"))
    except Exception:
        pass

    try:
        self.input_dir_browse_btn.configure(state=(tk.DISABLED if locked else tk.NORMAL))
    except Exception:
        pass

    if locked:
        try:
            self.input_dir_entry.selection_clear()
        except Exception:
            pass

    if locked:
        self.project_paths_info_var.set(
            "Ścieżki zostały uzupełnione automatycznie z aktywnego projektu."
        )
        self.project_paths_rel_var.set(
            f"IN:  {self._format_project_relative_path(self.input_dir_var.get().strip())}\n"
            f"OUT: {self._format_project_relative_path(self.output_dir_var.get().strip())}"
        )
    else:
        self.project_paths_info_var.set("")
        self.project_paths_rel_var.set("")

def _draw_campaign_step2_splash_bar(
    self,
    *,
    progress_value: float | None,
    accent: str,
    trough_color: str,
    border_color: str,
) -> None:
    progress_widget = getattr(self, "_campaign_step2_splash_progress", None)
    if progress_widget is None:
        return
    try:
        canvas_width = int(progress_widget.winfo_width() or 0)
        if canvas_width <= 1:
            canvas_width = int(progress_widget.winfo_reqwidth() or 520)
    except Exception:
        canvas_width = 520
    canvas_width = max(180, int(canvas_width or 520))
    try:
        canvas_height = int(progress_widget.winfo_height() or 0)
        if canvas_height <= 1:
            canvas_height = int(progress_widget.winfo_reqheight() or 16)
    except Exception:
        canvas_height = 16
    canvas_height = max(12, int(canvas_height or 16))
    bar_top = max(3, int((canvas_height - 8) / 2))
    bar_bottom = min(canvas_height - 2, bar_top + 8)
    left = 1
    right = max(left + 20, canvas_width - 1)
    try:
        progress_widget.delete("all")
        progress_widget.create_rectangle(
            left,
            bar_top,
            right,
            bar_bottom,
            fill=trough_color,
            outline=border_color,
            width=1,
        )
        if progress_value is None:
            fill_width = max(52, int(float(right - left) * 0.24))
            travel = max(1, int(right - left - fill_width))
            try:
                phase = (time.perf_counter() * 0.82) % 1.0
            except Exception:
                phase = 0.0
            fill_x0 = left + int(float(travel) * float(phase))
        else:
            fill_width = max(
                0,
                min(right - left, int((float(progress_value) / 100.0) * float(right - left))),
            )
            fill_x0 = left
        if fill_width > 0:
            progress_widget.create_rectangle(
                fill_x0,
                bar_top + 1,
                min(right, fill_x0 + fill_width),
                bar_bottom - 1,
                fill=accent,
                outline="",
            )
    except Exception:
        pass


def _schedule_campaign_step2_splash_animation(self) -> None:
    try:
        pending = getattr(self, "_campaign_step2_splash_anim_after_id", None)
        if pending:
            self.frame.after_cancel(pending)
    except Exception:
        pass
    self._campaign_step2_splash_anim_after_id = None
    if not bool(getattr(self, "_campaign_step2_splash_visible", False)):
        return
    if not bool(getattr(self, "_campaign_step2_splash_indeterminate", False)):
        return
    style = dict(getattr(self, "_campaign_step2_splash_bar_style", {}) or {})
    _draw_campaign_step2_splash_bar(
        self,
        progress_value=None,
        accent=str(style.get("accent") or "#4f8de3"),
        trough_color=str(style.get("trough_color") or "#1d1d1d"),
        border_color=str(style.get("border_color") or "#3c3c3c"),
    )
    try:
        self._campaign_step2_splash_anim_after_id = self.frame.after(
            70,
            lambda: _schedule_campaign_step2_splash_animation(self),
        )
    except Exception:
        self._campaign_step2_splash_anim_after_id = None


def _show_campaign_step2_splash(
    self,
    *,
    title: str = "Przygotowuję Z2",
    body: str = "",
    tone: str = "info",
    progress: float | None = None,
) -> int:
    if bool(getattr(self, "_plate_auto_scope_progress_modal_active", False)):
        updater = getattr(self, "_update_plate_auto_scope_progress_modal", None)
        if callable(updater):
            try:
                progress_value = None if progress is None else max(0.0, min(100.0, float(progress)))
                updater(
                    pct=progress_value,
                    meta_text=str(body or title or "").strip(),
                )
            except Exception:
                pass
        return int(getattr(self, "_campaign_step2_splash_token", 0) or 0)

    overlay = getattr(self, "_campaign_step2_splash_overlay", None)
    card = getattr(self, "_campaign_step2_splash_card", None)
    title_lbl = getattr(self, "_campaign_step2_splash_title_lbl", None)
    body_lbl = getattr(self, "_campaign_step2_splash_body_lbl", None)
    progress_widget = getattr(self, "_campaign_step2_splash_progress", None)
    progress_pct_lbl = getattr(self, "_campaign_step2_splash_progress_pct_lbl", None)
    progress_var = getattr(self, "_campaign_step2_splash_progress_var", None)
    progress_value = None if progress is None else max(0.0, min(100.0, float(progress)))
    if overlay is None or card is None or title_lbl is None or body_lbl is None or progress_widget is None:
        return 0

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    tone_key = str(tone or "info").strip().lower()
    # Splash is a loading surface, not a final status badge. Keep success in
    # the status text; do not repaint the whole overlay green.
    visual_tone_key = "info" if tone_key == "success" else tone_key
    if visual_tone_key == "error":
        accent = palette.get("error", "#e74c3c")
    elif visual_tone_key == "warning":
        accent = palette.get("warning", palette.get("accent", "#4f8de3"))
    else:
        accent = palette.get("accent", "#4f8de3")

    muted = palette.get("muted", "#c7c7c7")
    overlay_bg = blend_hex_colors(panel_bg, "#000000", 0.18)
    card_bg = blend_hex_colors(panel_bg, accent, 0.045)
    border_color = blend_hex_colors(accent, palette.get("panel_border", palette.get("border", "#3c3c3c")), 0.28)
    trough_color = blend_hex_colors(panel_bg, "#000000", 0.12)

    try:
        overlay.configure(bg=overlay_bg, highlightbackground=overlay_bg, highlightcolor=overlay_bg)
        card.configure(bg=card_bg, highlightbackground=border_color, highlightcolor=border_color)
        title_lbl.configure(text=str(title or "").strip(), bg=card_bg, fg=blend_hex_colors(accent, fg, 0.06))
        body_lbl.configure(text=str(body or "").strip(), bg=card_bg, fg=fg)
        progress_widget.configure(bg=card_bg)
        if progress_pct_lbl is not None:
            progress_pct_lbl.configure(
                bg=card_bg,
                fg=muted,
                text=("" if progress_value is None else f"{int(round(progress_value))}%"),
            )
            try:
                if progress_value is None:
                    progress_pct_lbl.grid_remove()
                else:
                    progress_pct_lbl.grid(row=3, column=0, sticky="e", pady=(5, 0))
            except Exception:
                pass
        if hasattr(self.app, "ensure_adaptive_wrap"):
            self.app.ensure_adaptive_wrap(body_lbl, container=card, padding=44, min_wrap=240)
    except Exception:
        pass

    try:
        if progress_var is not None:
            progress_var.set(0.0 if progress_value is None else progress_value)
    except Exception:
        pass
    try:
        self._campaign_step2_splash_bar_style = {
            "accent": accent,
            "trough_color": trough_color,
            "border_color": border_color,
        }
        self._campaign_step2_splash_indeterminate = bool(progress_value is None)
        _draw_campaign_step2_splash_bar(
            self,
            progress_value=progress_value,
            accent=accent,
            trough_color=trough_color,
            border_color=border_color,
        )
        if progress_value is not None:
            pending = getattr(self, "_campaign_step2_splash_anim_after_id", None)
            if pending:
                self.frame.after_cancel(pending)
            self._campaign_step2_splash_anim_after_id = None
    except Exception:
        pass
    try:
        overlay.place(in_=self.frame, relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
        overlay.lift()
    except Exception:
        pass
    was_visible = bool(getattr(self, "_campaign_step2_splash_visible", False))
    self._campaign_step2_splash_visible = True
    if not was_visible:
        self._campaign_step2_splash_token = int(getattr(self, "_campaign_step2_splash_token", 0) or 0) + 1
    if progress_value is None:
        try:
            _schedule_campaign_step2_splash_animation(self)
        except Exception:
            pass
    try:
        self.frame.update_idletasks()
    except Exception:
        pass
    try:
        self.frame.update()
    except Exception:
        pass
    return int(self._campaign_step2_splash_token)


def _hide_campaign_step2_splash(self, *, token: int | None = None) -> None:
    current_token = int(getattr(self, "_campaign_step2_splash_token", 0) or 0)
    if token is not None and int(token or 0) != current_token:
        return

    overlay = getattr(self, "_campaign_step2_splash_overlay", None)
    progress = getattr(self, "_campaign_step2_splash_progress", None)
    try:
        pending = getattr(self, "_campaign_step2_splash_anim_after_id", None)
        if pending:
            self.frame.after_cancel(pending)
    except Exception:
        pass
    self._campaign_step2_splash_anim_after_id = None
    self._campaign_step2_splash_indeterminate = False
    try:
        if progress is not None:
            progress.stop()
    except Exception:
        pass
    try:
        if overlay is not None:
            overlay.place_forget()
    except Exception:
        pass
    self._campaign_step2_splash_visible = False


def _begin_campaign_step2_transition(self) -> None:
    self._campaign_step2_transition_in_progress = True
    self._campaign_step2_transition_refresh_pending = False
    self._campaign_step2_transition_skip_heavy_finalize = False
    try:
        workflow_shell = getattr(self, "workflow_entry_shell", None)
        if workflow_shell is not None:
            workflow_shell.pack_forget()
    except Exception:
        pass


def _end_campaign_step2_transition(self) -> None:
    finalize_started = time.perf_counter()
    finalize_phase_started = finalize_started

    def _mark_finalize_phase(name: str) -> None:
        nonlocal finalize_phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - finalize_phase_started) * 1000.0
            total_ms = (now - finalize_started) * 1000.0
            if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                logger.info(
                    "[Z2 PERF] end_campaign_step2_transition "
                    f"{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                )
            finalize_phase_started = now
        except Exception:
            pass

    self._campaign_step2_transition_in_progress = False
    if not bool(getattr(self, "_campaign_step2_transition_refresh_pending", False)):
        self._campaign_step2_transition_skip_heavy_finalize = False
        return

    self._campaign_step2_transition_refresh_pending = False
    skip_heavy_finalize = bool(getattr(self, "_campaign_step2_transition_skip_heavy_finalize", False))
    self._campaign_step2_transition_skip_heavy_finalize = False
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    t02_at_review = bool(
        str(graph_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or str(graph_context.get("graph_gate_id") or "").strip().upper() == "T02"
    )
    if t02_at_review and bool(getattr(self, "_campaign_deferred_run_restore_in_progress", False)):
        _mark_finalize_phase("defer_t02_action_states_until_payload")
        try:
            self.frame.after(120, self._sync_right_panel_scrollregion)
        except Exception:
            pass
        return
    if skip_heavy_finalize:
        # A large existing run may still be restored asynchronously. If the
        # payload has already landed while the transition flag was active, the
        # restore callback's full refresh was intentionally deferred by
        # _refresh_step2_action_states(). In that case this finalizer must do
        # the full right-panel refresh now; otherwise the panel can stay stuck
        # in the temporary "loading active Z2 work" message.
        deferred_restore_pending = bool(
            getattr(self, "_campaign_deferred_run_restore_in_progress", False)
        ) and not bool(getattr(self, "_campaign_deferred_run_restore_payload_applied", False))
        if not deferred_restore_pending:
            try:
                self._refresh_step2_action_states(lightweight=False)
            except Exception:
                pass
            _mark_finalize_phase("full_action_states_after_deferred")
            try:
                self._refresh_free_mode_workflow_ui()
            except Exception:
                pass
            _mark_finalize_phase("workflow_ui_after_deferred")
            try:
                self._sync_right_panel_scrollregion()
            except Exception:
                pass
            _mark_finalize_phase("right_scrollregion_after_deferred")
            return

        # The payload is still pending. Do not even run the "lightweight"
        # action-state refresh here: on large campaign runs it still scans
        # enough state to block the transition for several seconds, and the
        # result is stale before the async restore callback lands anyway.
        _mark_finalize_phase("defer_action_states_until_payload")
        try:
            self.frame.after(120, self._sync_right_panel_scrollregion)
        except Exception:
            pass
        _mark_finalize_phase("schedule_scrollregion")
        return

    try:
        self._refresh_left_panel_route_copy()
    except Exception:
        pass
    _mark_finalize_phase("left_panel_route_copy")
    try:
        self._refresh_detection_configuration_ui()
    except Exception:
        pass
    _mark_finalize_phase("detection_configuration")
    try:
        self._refresh_step2_action_states()
    except Exception:
        pass
    _mark_finalize_phase("action_states")
    try:
        self._refresh_free_mode_workflow_ui()
    except Exception:
        pass
    _mark_finalize_phase("workflow_ui")
    try:
        self._refresh_step2_action_states()
    except Exception:
        pass
    _mark_finalize_phase("action_states_2")
    try:
        self._refresh_preview_list_summary()
    except Exception:
        pass
    _mark_finalize_phase("preview_list_summary")
    try:
        self._sync_right_panel_scrollregion()
    except Exception:
        pass
    _mark_finalize_phase("right_scrollregion")


def _reset_campaign_runtime_state(self, *args, **kwargs):
    return z2_workflow_methods._reset_campaign_runtime_state(self, *args, **kwargs)


def _apply_campaign_step2_workflow_preset(
    self,
    *,
    iteration_target: str | None = None,
    manual_template: bool | None = None,
) -> None:
    apply_campaign_step2_workflow_preset(
        self,
        iteration_target=iteration_target,
        manual_template=manual_template,
    )


def restore_campaign_context_from_project(self) -> bool:
    try:
        from ..campaign_manager import CAMPAIGN

        active_project = CAMPAIGN.get_active_project_name()
        if not active_project:
            self._campaign_context_project_name = ""
            return False
        if int(CAMPAIGN.get_current_step() or 1) < 2:
            snapshot_path = self._get_campaign_annotation_state_path(active_project)
            if snapshot_path is None or not snapshot_path.exists():
                return False

        raw_dir = CAMPAIGN.get_dir("raw")
        auto_out = CAMPAIGN.get_staging_dir("auto_ann")
        if raw_dir is None or auto_out is None:
            return False

        iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        campaign_mode_text = "B: Tylko tablice"
        bootstrap = self._get_campaign_auto_annotation_bootstrap(iteration_target)
        input_dir = Path(bootstrap.get("input_dir") or Path(raw_dir))
        manual_template = bool(bootstrap.get("manual_template", iteration_target == "plate"))
        plate_bootstrap_model = str(bootstrap.get("plate_model_path") or "").strip()
        restore_run_dir = bootstrap.get("restore_run_dir")
        snapshot_state = self._load_campaign_project_snapshot()
        try:
            self._repair_campaign_step2_generated_state_for_input(input_dir)
        except Exception:
            pass
        if not self._should_restore_existing_campaign_step2_run(iteration_target):
            restore_run_dir = None
            bootstrap["restore_run_dir"] = None
        elif not self._should_restore_campaign_generated_step2_run_preview(
            restore_run_dir,
            session_state=snapshot_state,
            iteration_target=iteration_target,
        ):
            restore_run_dir = None
            bootstrap["restore_run_dir"] = None

        restored_snapshot = self.apply_campaign_context(
            input_dir,
            Path(auto_out),
            manual_template=manual_template,
            mode_text=campaign_mode_text,
            restore_project_state=True,
        )

        v_mod = CAMPAIGN.get_global_model("vehicle")
        p_mod = CAMPAIGN.get_global_model("plate")

        if not restored_snapshot:
            if v_mod and Path(v_mod).exists():
                self.vehicle_model_var.set("Custom")
                self.vehicle_custom_var.set(v_mod)
            else:
                try:
                    vehicle_values = list(self.vehicle_combo["values"]) if hasattr(self, "vehicle_combo") else []
                except Exception:
                    vehicle_values = []
                detect_values = [value for value in vehicle_values if value != "Custom"]
                default_vehicle = "yolo11s" if "yolo11s" in detect_values else (detect_values[0] if detect_values else "")
                if default_vehicle:
                    self.vehicle_model_var.set(default_vehicle)
                self.vehicle_custom_var.set("")

            if (
                iteration_target == "char" and p_mod and Path(p_mod).exists()
            ) or (
                iteration_target == "plate" and plate_bootstrap_model and Path(plate_bootstrap_model).exists()
            ):
                selected_plate_model = plate_bootstrap_model if iteration_target == "plate" else p_mod
                self.plate_custom_var.set(selected_plate_model)
                self._remember_plate_model_runtime_meta(
                    model_path=selected_plate_model,
                    source="project",
                    scope="project",
                )
            else:
                self.plate_custom_var.set("")
                self._clear_plate_model_runtime_meta()

            self.mode_var.set(campaign_mode_text)
            self._on_mode_change()

            if restore_run_dir is not None:
                self._restore_preview_from_annotation_run(restore_run_dir)
            self._apply_campaign_step2_workflow_preset(
                iteration_target=iteration_target,
                manual_template=manual_template,
            )
        elif not self._get_workflow_route():
            self._apply_campaign_step2_workflow_preset(
                iteration_target=iteration_target,
                manual_template=manual_template,
            )
        try:
            self._reset_main_pane_left_width_for_z2()
        except Exception:
            pass
        self._campaign_context_project_name = str(active_project or "").strip()
        return True
    except Exception as e:
        logger.debug(f"Nie udalo sie przywrocic projektowego kontekstu Z2: {e}")
        return False


def open_campaign_step2_entry(
    self,
    *,
    iteration_target: str | None = None,
    entry_strategy: str | None = None,
    restore_preview: bool = True,
    open_existing_run: bool = True,
    defer_preview_load: bool = False,
    source_context: dict | None = None,
) -> dict:
    return dispatch_open_campaign_step2_entry(
        self,
        iteration_target=iteration_target,
        entry_strategy=entry_strategy,
        restore_preview=restore_preview,
        open_existing_run=open_existing_run,
        defer_preview_load=defer_preview_load,
        source_context=source_context,
    )


def _apply_campaign_iteration_model_defaults(self, iteration_target: str | None = None) -> None:
    try:
        from ..campaign_manager import CAMPAIGN

        if not CAMPAIGN.get_active_project_name():
            return

        target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
        if target not in {"plate", "char"}:
            return

        vehicle_model_path = str(CAMPAIGN.get_global_model("vehicle") or "").strip()
        plate_model_path = str(CAMPAIGN.get_global_model("plate") or "").strip()
    except Exception:
        return

    try:
        vehicle_values = list(self.vehicle_combo["values"]) if hasattr(self, "vehicle_combo") else []
    except Exception:
        vehicle_values = []

    current_vehicle = str(self.vehicle_model_var.get() or "").strip()
    current_vehicle_custom = str(self.vehicle_custom_var.get() or "").strip()
    vehicle_selection_invalid = False

    if current_vehicle == "Custom":
        vehicle_selection_invalid = not current_vehicle_custom or not Path(current_vehicle_custom).exists()
    elif current_vehicle:
        vehicle_selection_invalid = bool(vehicle_values) and current_vehicle not in vehicle_values
    else:
        vehicle_selection_invalid = True

    if vehicle_selection_invalid:
        if vehicle_model_path and Path(vehicle_model_path).exists():
            self.vehicle_model_var.set("Custom")
            self.vehicle_custom_var.set(vehicle_model_path)
        else:
            detect_values = [value for value in vehicle_values if value != "Custom"]
            default_vehicle = "yolo11s" if "yolo11s" in detect_values else (detect_values[0] if detect_values else "")
            if default_vehicle:
                self.vehicle_model_var.set(default_vehicle)
            self.vehicle_custom_var.set("")

    if target in {"char", "plate"} and plate_model_path and Path(plate_model_path).exists():
        current_plate = str(self.plate_custom_var.get() or "").strip()
        if not current_plate or not Path(current_plate).exists():
            self.plate_custom_var.set(plate_model_path)

    try:
        self._on_vehicle_model_change()
    except Exception:
        pass


def _enforce_campaign_plate_only_auto_default(self, iteration_target: str | None = None) -> None:
    if self._is_free_mode_session_context():
        return

    try:
        from ..campaign_manager import CAMPAIGN

        target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
    except Exception:
        target = str(iteration_target or "").strip().lower()

    if target not in {"plate", "char"}:
        return
    try:
        if self._is_campaign_char_repair_return_mode() or self._is_campaign_plate_step4_repair_return_mode():
            return
    except Exception:
        pass
    if self._manual_xml_template_enabled():
        return
    if self._get_workflow_route() != "auto":
        return
    if getattr(self, "current_annotation_xml_path", None) is not None:
        return

    changed = False
    if str(self.mode_var.get() or "").strip() != "B: Tylko tablice":
        self.mode_var.set("B: Tylko tablice")
        changed = True
    if self._get_auto_vehicle_choice() != "skip":
        self._set_auto_vehicle_choice_state(
            "skip",
            campaign_context=self._is_campaign_step2_context(),
        )
        changed = True

    if changed:
        self._refresh_auto_vehicle_choice_ui()
        self._refresh_left_panel_route_copy()
        self._refresh_detection_configuration_ui()
        self._refresh_step2_action_states()
        self._refresh_free_mode_workflow_ui()


def _should_restore_existing_campaign_step2_run(self, iteration_target: str | None = None) -> bool:
    if self._is_free_mode_session_context():
        return True

    return True


def _should_restore_campaign_generated_step2_run_preview(
    self,
    run_dir: Path | None,
    *,
    session_state: dict | None = None,
    iteration_target: str | None = None,
) -> bool:
    if self._is_free_mode_session_context():
        return True

    try:
        from ..campaign_manager import CAMPAIGN

        target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
        campaign_step = int(CAMPAIGN.get_current_step() or 0)
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
        current_step2_run = self._resolve_safe_annotation_run_dir(
            CAMPAIGN.get_step2_staging_run(),
            require_xml=True,
        )
    except Exception:
        return True

    candidate = self._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if (
        target != "plate"
        or campaign_step != 2
        or step2_status != "generated"
        or candidate is None
        or current_step2_run is None
    ):
        return True

    try:
        same_run = self._paths_equivalent(candidate, current_step2_run)
    except Exception:
        same_run = False
    if not same_run:
        return True

    try:
        manifest = self._load_annotation_run_manifest(candidate)
    except Exception:
        manifest = {}

    if self._annotation_run_manifest_has_manual_value(manifest):
        return True

    try:
        images_with_plates, total_plates = self._get_run_plate_annotation_counts(candidate)
    except Exception:
        images_with_plates, total_plates = 0, 0
    if int(total_plates or 0) > 0 or int(images_with_plates or 0) > 0:
        return True

    state = dict(session_state or {})
    if bool(state.get("manual_review_active", False)):
        return True

    workflow_route = self._normalize_workflow_route_value(state.get("workflow_route"))
    if workflow_route == "manual":
        return True

    return False


def _refresh_campaign_workflow_input_lock_state(self, *, campaign_context: bool, current_step: str):
    entry = getattr(self, "workflow_input_entry", None)
    browse_btn = getattr(self, "workflow_input_browse_btn", None)
    title_lbl = getattr(self, "workflow_input_title_lbl", None)
    hint_lbl = getattr(self, "workflow_input_hint_lbl", None)
    prefix_lookup = self._get_z2_thematic_title_prefixes()
    input_dir_value = str(self.input_dir_var.get() or "").strip()

    lock_input = bool(
        campaign_context
        and self._get_workflow_route() == "auto"
        and input_dir_value
    )

    if title_lbl is not None:
        try:
            title_lbl.configure(
                text=self._format_z2_thematic_title(
                    ("Folder obrazów tej iteracji" if lock_input else "Wskaż katalog obrazów"),
                    prefix_lookup.get("workflow_input"),
                )
            )
        except Exception:
            pass

    if hint_lbl is not None and lock_input:
        try:
            self._set_inline_label_state(
                hint_lbl,
                text=(
                    "To jest zestaw obrazów przypisany już do tej iteracji przez E1. "
                    "W Z2 nie trzeba wskazywać jej ponownie."
                ),
                tone="muted",
                emphasis=False,
            )
        except Exception:
            pass

    if entry is not None:
        try:
            entry.configure(state=("readonly" if lock_input else "normal"))
        except Exception:
            pass

    if browse_btn is not None:
        try:
            if lock_input:
                browse_btn.grid_remove()
            else:
                browse_btn.grid()
        except Exception:
            try:
                browse_btn.configure(state=(tk.DISABLED if lock_input else tk.NORMAL))
            except Exception:
                pass


def apply_campaign_context(
    self,
    input_dir: Path,
    output_dir: Path,
    *,
    manual_template: bool = False,
    mode_text: str = "C: Pojazdy + tablice",
    restore_project_state: bool = True,
    restore_preview: bool = True,
    defer_preview_load: bool = False,
    refresh_export_sources: bool = True,
    refresh_character_models: bool = True,
):
    context_started = time.perf_counter()
    context_phase_started = context_started

    def _mark_context_phase(name: str) -> None:
        nonlocal context_phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - context_phase_started) * 1000.0
            total_ms = (now - context_started) * 1000.0
            if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                logger.info(
                    "[Z2 PERF] apply_campaign_context "
                    f"{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                )
            context_phase_started = now
        except Exception:
            pass

    self._cancel_deferred_campaign_source_preview_load()
    self._campaign_project_restore_in_progress = True
    restored_snapshot = False
    try:
        self.input_dir_var.set(str(input_dir))
        self.output_dir_var.set(str(self._coerce_annotation_output_dir(output_dir)))
        self.plate_dataset_images_var.set(str(input_dir))
        self.plate_dataset_run_var.set("")
        self.plate_dataset_out_var.set(self._plate_dataset_output_preview())
        self.plate_custom_var.set("")
        self.manual_xml_template_var.set(bool(manual_template))
        self.manual_vehicle_assist_var.set(bool(manual_template))
        self.mode_var.set(self._normalize_mode_value(mode_text))
        self.character_model_var.set("Brak / OCR")
        self.character_custom_var.set("")
        _mark_context_phase("set_vars")
        self._reset_campaign_runtime_state(input_dir)
        _mark_context_phase("reset_runtime")
        self._update_manual_xml_template_ui()
        _mark_context_phase("manual_template_ui")
        self._on_mode_change()
        _mark_context_phase("mode_change")
        self._set_campaign_paths_lock_state(True)
        _mark_context_phase("lock_paths")
        if refresh_export_sources:
            self._refresh_plate_dataset_export_sources()
            _mark_context_phase("export_sources")
        else:
            _mark_context_phase("skip_export_sources")
        if hasattr(self, "character_combo") and refresh_character_models:
            self._refresh_character_model_choices()
            _mark_context_phase("character_models")
        elif hasattr(self, "character_combo"):
            previous_after_id = getattr(self, "_campaign_deferred_character_models_after_id", None)
            if previous_after_id:
                try:
                    self.frame.after_cancel(previous_after_id)
                except Exception:
                    pass
            self._campaign_deferred_character_models_after_id = None

            def _refresh_character_models_later() -> None:
                self._campaign_deferred_character_models_after_id = None
                try:
                    self._refresh_character_model_choices()
                except Exception:
                    pass

            try:
                self._campaign_deferred_character_models_after_id = self.frame.after(
                    750,
                    _refresh_character_models_later,
                )
            except Exception:
                pass
            _mark_context_phase("defer_character_models")
    finally:
        self._campaign_project_restore_in_progress = False

    if restore_project_state:
        restored_snapshot = self._apply_campaign_project_snapshot(restore_preview=restore_preview)
        _mark_context_phase("project_snapshot")

    if restore_preview and not restored_snapshot:
        self._restore_preview_from_session_run()
        _mark_context_phase("session_preview")

    _mark_context_phase("before_source_preview")
    if not restored_snapshot and not getattr(self, "current_annotations", None):
        if defer_preview_load:
            # The deferred loader owns the visible progress/status. Avoid a
            # costly status-label layout pass before Z2 is visible.
            pass
        else:
            try:
                self._prime_campaign_source_preview(input_dir)
            except Exception as e:
                logger.debug(f"Nie udało się przygotować podglądu zestawu Z2: {e}")

    _mark_context_phase("source_preview")

    try:
        self._apply_campaign_iteration_model_defaults()
    except Exception:
        pass
    _mark_context_phase("iteration_model_defaults")

    try:
        self._enforce_campaign_plate_only_auto_default()
    except Exception:
        pass
    _mark_context_phase("plate_auto_default")

    self._refresh_step2_action_states()
    _mark_context_phase("action_states")
    return restored_snapshot


def _prepare_campaign_source_preview_payload(self, *args, **kwargs):
    return z2_workflow_methods._prepare_campaign_source_preview_payload(self, *args, **kwargs)


def _apply_campaign_source_preview_payload(self, *args, **kwargs):
    return z2_workflow_methods._apply_campaign_source_preview_payload(self, *args, **kwargs)


def _ensure_free_mode_input_workspace_preview(self, *args, **kwargs):
    return z2_workflow_methods._ensure_free_mode_input_workspace_preview(self, *args, **kwargs)


def _schedule_free_mode_input_workspace_preview_load(
    self,
    *,
    expected_route: str = "auto",
    expected_step: str | None = None,
    require_auto_pending: bool = False,
) -> bool:
    if not self._is_free_mode_session_context():
        return False
    expected_route = str(expected_route or "").strip().lower()
    if str(self._get_workflow_route() or "").strip().lower() != expected_route:
        return False
    if str(self._coerce_free_mode_screen() or "").strip().lower() != "workflow":
        return False
    expected_step = str(expected_step or f"{expected_route}_start").strip().lower()
    if str(self._coerce_workflow_step() or "").strip().lower() != expected_step:
        return False
    if require_auto_pending and not bool(getattr(self, "_auto_route_settings_pending", False)):
        return False
    if self.is_processing:
        return False

    images_value = str(self.input_dir_var.get() or "").strip()
    if not images_value:
        return False

    resolved_input_dir = self._resolve_annotation_input_images_dir(Path(images_value))
    if resolved_input_dir is None:
        return False

    try:
        current_input_dir = getattr(self, "current_input_dir", None)
        if (
            current_input_dir is not None
            and self._paths_equivalent(Path(current_input_dir), resolved_input_dir)
            and bool(getattr(self, "current_annotations", None))
        ):
            return False
    except Exception:
        pass

    if bool(getattr(self, "_free_mode_auto_workspace_preview_load_in_progress", False)):
        return True

    token = int(getattr(self, "_free_mode_auto_workspace_preview_load_token", 0) or 0) + 1
    self._free_mode_auto_workspace_preview_load_token = token
    self._free_mode_auto_workspace_preview_load_in_progress = True

    try:
        self._show_campaign_step2_splash(
            title="Ładowanie katalogu obrazów",
            body="Przygotowuję workspace Z2 dla wybranego zestawu zdjęć.",
            tone="info",
            progress=0.0,
        )
    except Exception:
        pass

    def _run() -> None:
        if token != int(getattr(self, "_free_mode_auto_workspace_preview_load_token", 0) or 0):
            return
        self._free_mode_auto_workspace_preview_load_after_id = None
        try:
            self._ensure_free_mode_input_workspace_preview(
                expected_route=expected_route,
                expected_step=expected_step,
            )
        except Exception as e:
            logger.debug(f"Nie udało się przygotować odroczonego workspace Z2 dla freemode {expected_route}: {e}")
            try:
                self._hide_campaign_step2_splash()
            except Exception:
                pass
        finally:
            if token == int(getattr(self, "_free_mode_auto_workspace_preview_load_token", 0) or 0):
                self._free_mode_auto_workspace_preview_load_in_progress = False

    try:
        self._free_mode_auto_workspace_preview_load_after_id = self.frame.after(1, _run)
    except Exception:
        self._free_mode_auto_workspace_preview_load_after_id = None
        _run()
    return True
