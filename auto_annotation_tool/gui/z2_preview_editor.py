#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 preview editor helpers extracted from tab_annotation.py."""

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
from .web_slim_scrollbar import blend_hex_colors
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
PREVIEW_CORRECTION_AUTOSAVE_DELAY_MS = 45000
PREVIEW_IMAGE_CACHE_LIMIT = 24
SUPER_CORRECTION_AUTOSAVE_DELAY_MS = 12000

def _preview_super_perf_active(self) -> bool:
    try:
        return time.perf_counter() <= float(getattr(self, "_preview_super_perf_until", 0.0) or 0.0)
    except Exception:
        return False


def _log_preview_super_perf(self, operation: str, elapsed_ms: float, *, threshold_ms: float = 80.0, **details) -> None:
    if not _preview_super_perf_active(self) and float(elapsed_ms) < float(threshold_ms):
        return
    try:
        suffix = " ".join(f"{key}={value}" for key, value in details.items())
        if suffix:
            suffix = " " + suffix
        logger.info(
            "[Z2 DRAG PERF] %s %.1fms label=%s%s",
            operation,
            float(elapsed_ms),
            str(getattr(self, "_preview_super_perf_label", "") or "-"),
            suffix,
        )
    except Exception:
        pass


def _is_free_mode_manual_xml_waiting_for_review(self) -> bool:
    if not self._is_free_mode_session_context():
        return False
    try:
        if self._get_workflow_route() != "manual":
            return False
        if self._get_manual_entry_mode() != "new":
            return False
        if str(self._coerce_free_mode_screen() or "").strip().lower() != "workflow":
            return False
        if str(self._coerce_workflow_step() or "").strip().lower() != "manual_start":
            return False
        return bool(getattr(self, "_manual_template_ready_for_review", False))
    except Exception:
        return False


def _get_step2_start_action_label(self) -> str:
    try:
        route = self._get_workflow_route()
    except Exception:
        route = ""

    if route == "manual":
        try:
            if self._is_free_mode_manual_xml_waiting_for_review():
                return "Dalej"
            if self._get_manual_entry_mode() == "new":
                return "Utwórz XML + boxy" if self._manual_vehicle_assist_enabled() else "Utwórz XML"
        except Exception:
            pass
    elif route == "auto":
        try:
            return (
                "Uruchom autoanotację tablic"
                if self._get_auto_vehicle_choice() == "skip"
                else "Uruchom autoanotację tablic ze wsparciem pojazdów"
            )
        except Exception:
            pass

    try:
        label = str(self.start_btn.cget("text") or "").strip()
    except Exception:
        label = ""
    generic_labels = {
        "",
        "START",
        "Start",
        "Wybierz tor",
        "Wybierz model",
        "Wybierz obrazy",
        "Run istnieje",
    }
    return "" if label in generic_labels else label


def _get_step2_start_action_reference(self) -> str:
    label = self._get_step2_start_action_label()
    if label:
        return f"„{label}”"
    return "główny przycisk po lewej stronie"


def _on_preview_canvas_motion(self, event=None):
    if getattr(self, "_preview_drag_state", None) is not None:
        return None
    if bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            try:
                if getattr(canvas, "current_cursor", None) != "crosshair":
                    canvas.current_cursor = "crosshair"
                    canvas.config(cursor="crosshair")
            except Exception:
                pass
        return None
    self._sync_preview_canvas_cursor()
    return None


def _on_preview_canvas_leave(self, event=None):
    self._sync_preview_canvas_cursor()
    return None


def _update_preview_toolbar_state(self, *, refresh_summary: bool = True):
    total = len(self.current_annotations)
    has_selection = self.current_preview_index is not None and total > 0
    has_image = has_selection and getattr(self.preview_canvas, "original_image", None) is not None
    can_edit = bool(has_image and self._preview_is_editable())
    display_total = len(getattr(self, "_preview_list_display_indices", []) or [])
    current_display_index = self._get_preview_display_index(self.current_preview_index) if has_selection else None
    navigation_total = max(display_total, total)
    can_go_prev = has_selection and current_display_index is not None and int(current_display_index) > 0
    can_go_next = (
        has_selection
        and current_display_index is not None
        and int(current_display_index) < (navigation_total - 1)
    )
    has_dirty = bool(self._preview_dirty_images)
    scope_selection_mode_active = bool(getattr(self, "_plate_auto_scope_selection_mode_active", False))
    selected_actual_indices = self._get_selected_preview_actual_indices()
    has_group_selection = bool(selected_actual_indices and self._preview_is_editable())
    can_rename_single = bool(
        can_edit
        and len(selected_actual_indices) == 1
        and getattr(self, "current_input_dir", None) is not None
    )
    approved_names = self._get_preview_approved_filenames_base()
    any_selected_unapproved = False
    any_selected_approved = False
    any_selected_auto = False
    for actual_index in selected_actual_indices:
        if actual_index < 0 or actual_index >= len(self.current_annotations or []):
            continue
        ann = self.current_annotations[actual_index]
        if self._preview_annotation_is_explicitly_approved(ann, approved_names=approved_names):
            any_selected_approved = True
        else:
            if self._preview_annotation_can_be_approved_for_export(ann):
                any_selected_unapproved = True
        if self._preview_annotation_has_auto_plate(ann):
            any_selected_auto = True
    any_auto_in_run = self._get_preview_any_auto_in_run()
    can_move_to_stage = bool(
        can_edit
        and getattr(self, "current_input_dir", None) is not None
        and not self._is_manual_plate_stage_input(self.current_input_dir)
    )
    if refresh_summary:
        self._refresh_preview_list_summary()

    try:
        self.preview_prev_btn.configure(state=(tk.NORMAL if can_go_prev else tk.DISABLED))
        self.preview_next_btn.configure(state=(tk.NORMAL if can_go_next else tk.DISABLED))
        self.preview_fit_btn.configure(state=(tk.NORMAL if (has_image and not scope_selection_mode_active) else tk.DISABLED))
        self.preview_draw_btn.configure(state=(tk.NORMAL if (can_edit and not scope_selection_mode_active) else tk.DISABLED))
        self.preview_draw_btn.configure(text=("Anuluj rysowanie (D)" if self._preview_draw_mode else "Rysuj ramkę 4 pkt (D)"))
        if hasattr(self, "preview_fullscreen_btn"):
            self.preview_fullscreen_btn.configure(
                state=(tk.NORMAL if (has_image and not scope_selection_mode_active) else tk.DISABLED),
                text=("Wyjdź z pełnego ekranu (Enter)" if self._preview_fullscreen_active else "Pełny ekran (Enter)")
            )
        if hasattr(self, "preview_apply_filter_btn"):
            self.preview_apply_filter_btn.configure(
                state=(tk.NORMAL if (display_total > 0 and not scope_selection_mode_active) else tk.DISABLED)
            )
        if hasattr(self, "preview_move_stage_btn"):
            self.preview_move_stage_btn.configure(state=(tk.NORMAL if (can_move_to_stage and not scope_selection_mode_active) else tk.DISABLED))
        if hasattr(self, "preview_delete_image_btn"):
            self.preview_delete_image_btn.configure(state=(tk.NORMAL if (can_edit and not scope_selection_mode_active) else tk.DISABLED))
        self.preview_save_btn.configure(state=(tk.NORMAL if (can_edit and has_dirty and not scope_selection_mode_active) else tk.DISABLED))
    except Exception:
        pass

    try:
        self._refresh_preview_list_context_menu_state(
            has_visible_rows=bool(display_total > 0),
            has_group_selection=bool(has_group_selection),
            can_rename_single=bool(can_rename_single),
            any_selected_unapproved=bool(any_selected_unapproved),
            any_selected_approved=bool(any_selected_approved),
            can_clear_auto=bool(can_edit and any_auto_in_run),
            scope_selection_mode_active=bool(scope_selection_mode_active),
        )
    except Exception:
        pass

    try:
        self.preview_fullscreen_hint_var.set("")
        self._set_inline_label_state(self.preview_fullscreen_hint_lbl, tone="muted", emphasis=False)
    except Exception:
        pass

    scope_refresh_callback = getattr(self, "_plate_auto_scope_modal_refresh_callback", None)
    if callable(scope_refresh_callback):
        try:
            scope_refresh_callback()
        except Exception:
            pass

    self._sync_preview_canvas_cursor()


def _preview_draw_corner_label(point_idx: int) -> str:
    labels = {
        1: "1 rog",
        2: "2 rog",
        3: "3 rog",
        4: "4 rog",
    }
    return labels.get(int(point_idx), f"{int(point_idx)} rog")


def _preview_campaign_reuse_manual_note(self, ann, *, editable: bool) -> str:
    if not self._preview_annotation_is_reused_from_previous_manual(ann):
        return ""

    summary = dict(getattr(self, "_campaign_reuse_manual_summary", {}) or {})
    source_label = str(summary.get("source_label") or "wcześniejszej iteracji")
    if editable:
        return f" To zdjęcie oznaczone jako {self._campaign_reuse_manual_badge()} pochodzi z ręcznej anotacji z {source_label}."
    return (
        f" To zdjęcie oznaczone jako {self._campaign_reuse_manual_badge()} pochodzi z ręcznej anotacji z {source_label}; "
        "nowy run autoanotacji nadpisze poprzednie ręczne oznaczenia."
    )


def _update_preview_edit_status(self, *args, **kwargs):
    return z2_workflow_methods._update_preview_edit_status(self, *args, **kwargs)


def _load_current_preview_selection(
    self,
    reset_view: bool = True,
    selection_changed: bool = True,
    refresh_summary: bool = True,
    fast_fullscreen: bool | None = None,
):
    if fast_fullscreen is None:
        fast_fullscreen = bool(getattr(self, "_preview_fullscreen_active", False))
    else:
        fast_fullscreen = bool(fast_fullscreen and getattr(self, "_preview_fullscreen_active", False))

    ann = self._get_preview_annotation()
    if ann is None:
        try:
            self.preview_canvas.clear_image()
        except Exception:
            pass
        self._update_preview_edit_status()
        self._update_preview_toolbar_state(refresh_summary=refresh_summary)
        return

    if selection_changed:
        try:
            self.preview_canvas.grab_release()
        except Exception:
            pass
        self._preview_drag_state = None
        self._preview_pending_vertex_hit = None
        self._preview_draw_mode = False
        self._preview_draw_points = []
        self._preview_delete_mode = False
        self._preview_delete_candidate_idx = None
        self._preview_polygon_focus_restore_state = None
        self._preview_focus_target = None
        self._push_preview_debug_event(
            "select",
            f"idx={self.current_preview_index if self.current_preview_index is not None else '-'} file={getattr(ann, 'filename', '-')}"
        )

    self._render_preview_image(
        ann,
        reset_view=reset_view,
        refresh_summary=refresh_summary,
        fast_fullscreen=fast_fullscreen,
    )
    if fast_fullscreen:
        try:
            self._place_preview_legend_overlay(refresh=True)
        except Exception:
            pass
        try:
            self._place_preview_overlay_dock()
        except Exception:
            pass
    self._update_preview_edit_status(
        refresh_toolbar=not fast_fullscreen,
        refresh_debug=not fast_fullscreen,
    )
    if not fast_fullscreen:
        self._update_preview_toolbar_state(refresh_summary=refresh_summary)


def _schedule_preview_selection_render_after_restore(self) -> None:
    if not self.current_annotations or self.current_preview_index is None:
        return

    def _rerender() -> None:
        if not self.current_annotations or self.current_preview_index is None:
            return
        try:
            self._select_preview_index(int(self.current_preview_index), reset_view=False)
        except Exception:
            pass

    try:
        self.frame.after_idle(_rerender)
    except Exception:
        _rerender()


def _render_preview_image(
    self,
    ann,
    reset_view: bool = True,
    refresh_summary: bool = True,
    fast_fullscreen: bool = False,
):
    img_path = self._resolve_preview_image_path(ann)

    if img_path is None or not img_path.exists():
        self.preview_canvas.clear_image()
        self.preview_canvas.create_text(20, 20, text="Plik nie istnieje na dysku!", fill="red", anchor="nw")
        self._update_preview_toolbar_state(refresh_summary=refresh_summary)
        self._refresh_preview_legend_backdrop()
        self._place_preview_overlay_dock()
        self._place_preview_campaign_gate_overlay()
        return

    try:
        try:
            stat = img_path.stat()
            cache_key = (
                str(img_path.resolve()),
                int(getattr(stat, "st_mtime_ns", 0) or 0),
                int(getattr(stat, "st_size", 0) or 0),
            )
        except Exception:
            cache_key = (str(img_path), 0, 0)
        image_cache = getattr(self, "_preview_render_image_cache", None)
        if not isinstance(image_cache, dict):
            image_cache = {}
            self._preview_render_image_cache = image_cache
        preview_image = image_cache.get(cache_key)
        if preview_image is None:
            load_started_at = time.perf_counter()
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Nie można załadować obrazu do podglądu.")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(img_rgb)
            image_cache[cache_key] = preview_image
            load_elapsed_ms = max(0.0, (time.perf_counter() - load_started_at) * 1000.0)
            if load_elapsed_ms >= 120.0:
                try:
                    logger.info(
                        "[Z2 PERF] preview_image_load total=%.0fms cache=%s file=%s",
                        load_elapsed_ms,
                        len(image_cache),
                        Path(img_path).name,
                    )
                except Exception:
                    pass
            while len(image_cache) > PREVIEW_IMAGE_CACHE_LIMIT:
                try:
                    image_cache.pop(next(iter(image_cache)))
                except Exception:
                    break
        try:
            image_width = max(1, int(getattr(preview_image, "width", 1) or 1))
            image_height = max(1, int(getattr(preview_image, "height", 1) or 1))
            if (
                int(getattr(ann, "width", 0) or 0) != image_width
                or int(getattr(ann, "height", 0) or 0) != image_height
            ):
                ann.width = image_width
                ann.height = image_height
        except Exception:
            pass
        render_interaction_fast = bool(fast_fullscreen)
        if reset_view:
            if hasattr(self.preview_canvas, "set_image_fit_to_view"):
                if not render_interaction_fast:
                    try:
                        render_interaction_fast = self._preview_user_interaction_quiet_remaining_ms(padding_ms=0) > 0
                    except Exception:
                        render_interaction_fast = False
                try:
                    self.preview_canvas.set_image_fit_to_view(
                        preview_image,
                        interaction_fast=bool(render_interaction_fast),
                    )
                except TypeError:
                    self.preview_canvas.set_image_fit_to_view(preview_image)
            else:
                self.preview_canvas.set_image(preview_image)
                self.preview_canvas.fit_to_view()
            self._preview_force_fit_after_resize = True
            self._schedule_preview_layout_restore_after_resize()
        else:
            if hasattr(self.preview_canvas, "set_image_preserve_view"):
                self.preview_canvas.set_image_preserve_view(preview_image)
            else:
                self.preview_canvas.set_image(preview_image)
        if fast_fullscreen:
            self._preview_metrics_overlay_render_key = None
            try:
                self._update_preview_canvas_metrics_overlay(force_render=True)
            except Exception:
                pass
        else:
            self._refresh_preview_legend_backdrop()
            self._update_preview_canvas_metrics_overlay()
            if not render_interaction_fast:
                try:
                    self.preview_canvas.after_idle(
                        lambda: self._update_preview_canvas_metrics_overlay(force_render=True)
                    )
                except Exception:
                    pass
            self._place_preview_overlay_dock()
            self._place_preview_campaign_gate_overlay()
    except Exception as e:
        logger.error(f"Błąd rysowania podglądu YOLO: {e}")
        self.preview_canvas.clear_image()
        self.preview_canvas.create_text(20, 20, text=f"Błąd podglądu: {e}", fill="red", anchor="nw")
        self._refresh_preview_legend_backdrop()
        self._update_preview_canvas_metrics_overlay()
        self._place_preview_overlay_dock()
        self._place_preview_campaign_gate_overlay()


def _refresh_preview_canvas(self, rerender_image: bool = False, *, refresh_chrome: bool = True):
    drag_active = bool(getattr(self, "_preview_drag_state", None)) or bool(
        getattr(self, "_preview_super_correction_drag_state", None)
    )
    try:
        if self.preview_canvas.original_image is not None:
            if rerender_image:
                self.preview_canvas._update_display()
            else:
                self.preview_canvas.refresh_overlay_only(
                    skip_info=drag_active
                )
    except Exception:
        pass

    if refresh_chrome and not drag_active:
        self._refresh_preview_legend_backdrop()
        self._update_preview_canvas_metrics_overlay()
        self._place_preview_overlay_dock()
        self._place_preview_campaign_gate_overlay()


def _cancel_preview_post_interaction_refresh(self) -> None:
    pending = getattr(self, "_preview_post_interaction_refresh_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_post_interaction_refresh_after_id = None


def _refresh_preview_canvas_light(self) -> None:
    try:
        if getattr(self.preview_canvas, "original_image", None) is None:
            return
        self._preview_light_overlay_refresh = True
        self.preview_canvas.refresh_overlay_only(skip_info=True)
    except Exception:
        pass
    finally:
        self._preview_light_overlay_refresh = False


def _schedule_preview_post_interaction_refresh(self, delay_ms: int = 360) -> None:
    self._cancel_preview_post_interaction_refresh()

    def _flush() -> None:
        self._preview_post_interaction_refresh_after_id = None
        if getattr(self, "_preview_drag_state", None) is not None:
            return
        if getattr(self, "_preview_super_correction_drag_state", None) is not None:
            return
        try:
            quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=260)
        except Exception:
            quiet_remaining = 0
        if quiet_remaining > 0:
            self._schedule_preview_post_interaction_refresh(delay_ms=quiet_remaining)
            return

        started_at = time.perf_counter()
        self._refresh_preview_canvas()
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 120.0:
            logger.info("[Z2 PERF] post_interaction_refresh total=%.1fms", elapsed_ms)

    try:
        self._preview_post_interaction_refresh_after_id = self.frame.after(
            max(0, int(delay_ms)),
            _flush,
        )
    except Exception:
        self._preview_post_interaction_refresh_after_id = None


def _refresh_preview_canvas_interactive(self, delay_ms: int = 360) -> None:
    self._refresh_preview_canvas_light()
    self._schedule_preview_post_interaction_refresh(delay_ms=delay_ms)


def _refresh_preview_legend_backdrop(self):
    if not bool(getattr(self, "_preview_controls_legend_visible", True)):
        self._hide_preview_controls_legend_overlay()
        return
    try:
        self._place_preview_legend_overlay(refresh=True)
    except Exception:
        pass


def _cancel_preview_drag_refresh(self):
    pending = getattr(self, "_preview_drag_refresh_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_drag_refresh_after_id = None


def _schedule_preview_drag_refresh(self, delay_ms: int = 24):
    if getattr(self, "_preview_drag_refresh_after_id", None):
        return

    def _flush():
        self._preview_drag_refresh_after_id = None
        if getattr(self, "_preview_drag_state", None) is None:
            return
        self._refresh_preview_canvas_light()

    try:
        self._preview_drag_refresh_after_id = self.frame.after(max(0, int(delay_ms)), _flush)
    except Exception:
        self._preview_drag_refresh_after_id = None
        self._refresh_preview_canvas()


def _cancel_preview_selection_render(self) -> None:
    pending = getattr(self, "_preview_select_render_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_select_render_after_id = None
    self._preview_pending_select_render = None


def _schedule_preview_selection_render(
    self,
    idx: int,
    previous_idx: int | None,
    *,
    reset_view: bool = True,
    delay_ms: int = 55,
) -> None:
    self._preview_pending_select_render = {
        "idx": int(idx),
        "previous_idx": None if previous_idx is None else int(previous_idx),
        "reset_view": bool(reset_view),
    }

    pending = getattr(self, "_preview_select_render_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
        self._preview_select_render_after_id = None

    def _flush_selection_render() -> None:
        payload = getattr(self, "_preview_pending_select_render", None)
        self._preview_select_render_after_id = None
        self._preview_pending_select_render = None
        if not isinstance(payload, dict):
            return
        target_idx = int(payload.get("idx", -1))
        if target_idx < 0 or target_idx >= len(self.current_annotations or []):
            return
        if self.current_preview_index is None or int(self.current_preview_index) != target_idx:
            return

        started = time.perf_counter()
        previous = payload.get("previous_idx")
        self._load_current_preview_selection(
            reset_view=bool(payload.get("reset_view", True)),
            selection_changed=(previous is None or target_idx != int(previous)),
            refresh_summary=False,
        )
        self._schedule_preview_resume_persist(include_preview_approved=False)
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        if elapsed_ms >= 20.0:
            logger.debug(
                "[AnnotationTab][PERF] preview_select_render: "
                f"{elapsed_ms:.1f} ms | idx={target_idx} total={len(self.current_annotations or [])}"
            )
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass
        self._refresh_plate_auto_scope_modal_selection_state()

    try:
        self._preview_select_render_after_id = self.frame.after(
            max(0, int(delay_ms)),
            _flush_selection_render,
        )
    except Exception:
        self._preview_select_render_after_id = None
        _flush_selection_render()


def _draw_annotation_preview_overlay(self, canvas: ZoomableCanvas):
    ann = self._get_preview_annotation()
    if ann is None or canvas.original_image is None:
        self._preview_super_correction_badge_bbox = None
        self._preview_super_correction_handle_bbox = None
        self._preview_fullscreen_toggle_bbox = None
        return

    self._preview_super_correction_badge_bbox = None
    self._preview_super_correction_handle_bbox = None
    self._preview_fullscreen_toggle_bbox = None
    self._preview_bottom_hint_bbox = None
    self._preview_bottom_hint_move_bbox = None
    self._preview_bottom_hint_collapse_bbox = None
    self._preview_bottom_hint_restore_bbox = None
    drag_active = isinstance(getattr(self, "_preview_drag_state", None), dict)
    light_overlay = (
        drag_active
        or bool(getattr(self, "_preview_light_overlay_refresh", False))
        or bool(getattr(canvas, "_interaction_fast_rendering", False))
    )
    vehicle_color = "#2ecc71"
    plate_color = "#e74c3c"
    approved_plate_color = "#2fbf71"
    active_color = "#f1c40f"
    delete_color = "#ff4d4d"
    handle_fill = "#ffffff"
    handle_outline = "#111111"
    label_fill = "#f8f8f8"
    label_bg = "#111111"

    vehicle_detections = self._get_vehicle_detections(ann)
    selected_vehicle_idx = self._get_selected_vehicle_index_for_ann(ann)
    plate_detections = self._get_plate_detections(ann)
    selected_plate_idx = self._get_selected_plate_index_for_ann(ann)
    try:
        image_approved = bool(self._preview_annotation_is_explicitly_approved(ann))
    except Exception:
        image_approved = False
    delete_candidate_idx = (
        int(self._preview_delete_candidate_idx)
        if self._preview_delete_mode and self._preview_delete_candidate_idx is not None
        else None
    )

    try:
        legend_theme = self._get_preview_legend_theme()
        canvas_width = max(1.0, float(canvas.winfo_width() or 1.0))
        canvas_height = max(1.0, float(canvas.winfo_height() or 1.0))

        toggle_size = 24.0
        toggle_pad = 12.0
        toggle_gap = 8.0
        toggle_x2 = max(toggle_pad + toggle_size, canvas_width - toggle_pad)
        toggle_x1 = toggle_x2 - toggle_size
        toggle_y1 = toggle_pad
        toggle_y2 = toggle_y1 + toggle_size
        toggle_fill = str(legend_theme.get("panel_fill", "#1f2933"))
        toggle_outline = str(legend_theme.get("shell_outline", "#2fbf71"))
        toggle_icon = str(legend_theme.get("entry_text", "#f8fafc"))

        canvas.create_rectangle(
            toggle_x1,
            toggle_y1,
            toggle_x2,
            toggle_y2,
            outline=toggle_outline,
            fill=toggle_fill,
            width=1,
            tags=("preview_overlay", "preview_fullscreen_toggle"),
        )
        inner_pad = 5.0
        inner_x1 = toggle_x1 + inner_pad
        inner_y1 = toggle_y1 + inner_pad
        inner_x2 = toggle_x2 - inner_pad
        inner_y2 = toggle_y2 - inner_pad
        corner_len = 5.0
        if bool(getattr(self, "_preview_fullscreen_active", False)):
            canvas.create_line(inner_x1 + corner_len, inner_y1, inner_x1, inner_y1, inner_x1, inner_y1 + corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x2 - corner_len, inner_y1, inner_x2, inner_y1, inner_x2, inner_y1 + corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x1 + corner_len, inner_y2, inner_x1, inner_y2, inner_x1, inner_y2 - corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x2 - corner_len, inner_y2, inner_x2, inner_y2, inner_x2, inner_y2 - corner_len, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
        else:
            canvas.create_line(inner_x1, inner_y1 + corner_len, inner_x1, inner_y1, inner_x1 + corner_len, inner_y1, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x2, inner_y1 + corner_len, inner_x2, inner_y1, inner_x2 - corner_len, inner_y1, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x1, inner_y2 - corner_len, inner_x1, inner_y2, inner_x1 + corner_len, inner_y2, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
            canvas.create_line(inner_x2, inner_y2 - corner_len, inner_x2, inner_y2, inner_x2 - corner_len, inner_y2, fill=toggle_icon, width=1.8, capstyle=tk.ROUND, tags=("preview_overlay", "preview_fullscreen_toggle"))
        self._preview_fullscreen_toggle_bbox = (
            float(toggle_x1),
            float(toggle_y1),
            float(toggle_x2),
            float(toggle_y2),
        )
        try:
            canvas.tag_raise("preview_fullscreen_toggle")
        except Exception:
            pass
        if light_overlay:
            raise StopIteration
        if not bool(getattr(self, "_preview_super_correction_badge_visible", True)):
            raise StopIteration

        overlay_state_text = (
            "Superkorekta: WŁĄCZONA"
            if bool(getattr(self, "_preview_super_correction_active", False))
            else "Superkorekta: wyłączona"
        )
        is_super_active = bool(getattr(self, "_preview_super_correction_active", False))
        overlay_state_fill = str(legend_theme.get("panel_fill", "#1f2933"))
        overlay_state_outline = str(legend_theme.get("shell_outline", "#2fbf71")) if is_super_active else str(legend_theme.get("panel_outline", "#4b5563"))
        overlay_state_text_fill = str(legend_theme.get("entry_text", "#f8fafc"))
        grip_fill = str(legend_theme.get("entry_fill", "#2f3a46"))
        grip_dot_fill = str(legend_theme.get("section_muted", "#d7e1ec"))
        lamp_fill = str(legend_theme.get("badge_plate_outline", "#2fbf71")) if is_super_active else str(legend_theme.get("panel_outline", "#4b5563"))
        state_anchor_y = float(getattr(self, "_preview_super_correction_badge_offset_y", 14.0) or 14.0)
        font_obj = self._get_preview_legend_font(8, "normal")
        pad_x = 8.0
        pad_y = 4.0
        handle_w = 16.0
        lamp_d = 8.0
        gap = 6.0
        text_w = float(font_obj.measure(overlay_state_text))
        badge_h = 22.0
        badge_w = lamp_d + gap + text_w + gap + handle_w + (pad_x * 2.0)
        default_x = max(14.0, toggle_x1 - toggle_gap - badge_w)
        state_anchor_x = float(getattr(self, "_preview_super_correction_badge_offset_x", default_x) or default_x)
        state_anchor_x = min(max(0.0, state_anchor_x), max(0.0, canvas_width - badge_w))
        max_badge_x_before_toggle = max(0.0, toggle_x1 - toggle_gap - badge_w)
        state_anchor_x = min(state_anchor_x, max_badge_x_before_toggle)
        state_anchor_y = min(max(0.0, state_anchor_y), max(0.0, canvas_height - badge_h))
        self._preview_super_correction_badge_offset_x = state_anchor_x
        self._preview_super_correction_badge_offset_y = state_anchor_y
        state_bg_id = canvas.create_rectangle(
            state_anchor_x,
            state_anchor_y,
            state_anchor_x + badge_w,
            state_anchor_y + badge_h,
            outline=overlay_state_outline,
            fill=overlay_state_fill,
            width=1,
            tags=("preview_overlay", "preview_super_correction_badge"),
        )
        lamp_x1 = state_anchor_x + pad_x
        lamp_y1 = state_anchor_y + ((badge_h - lamp_d) / 2.0)
        canvas.create_oval(
            lamp_x1,
            lamp_y1,
            lamp_x1 + lamp_d,
            lamp_y1 + lamp_d,
            outline="",
            fill=lamp_fill,
            tags=("preview_overlay", "preview_super_correction_badge"),
        )
        text_x = lamp_x1 + lamp_d + gap
        handle_x2 = state_anchor_x + badge_w - 4.0
        handle_x1 = handle_x2 - handle_w
        handle_y1 = state_anchor_y + 3.0
        handle_y2 = state_anchor_y + badge_h - 3.0
        state_text_id = canvas.create_text(
            text_x,
            state_anchor_y + (badge_h / 2.0),
            text=overlay_state_text,
            fill=overlay_state_text_fill,
            anchor="w",
            font=font_obj,
            tags=("preview_overlay", "preview_super_correction_badge"),
        )
        canvas.create_rectangle(
            handle_x1,
            handle_y1,
            handle_x2,
            handle_y2,
            outline="",
            fill=grip_fill,
            tags=("preview_overlay", "preview_super_correction_badge", "preview_super_correction_handle"),
        )
        dot_cx = (handle_x1 + handle_x2) / 2.0
        for row_y in (state_anchor_y + 8.0, state_anchor_y + 11.0, state_anchor_y + 14.0):
            for col_dx in (-2.5, 2.5):
                canvas.create_oval(
                    dot_cx + col_dx - 1.0,
                    row_y - 1.0,
                    dot_cx + col_dx + 1.0,
                    row_y + 1.0,
                    fill=grip_dot_fill,
                    outline="",
                    tags=("preview_overlay", "preview_super_correction_badge", "preview_super_correction_handle"),
                )
        self._preview_super_correction_badge_bbox = (
            float(state_anchor_x),
            float(state_anchor_y),
            float(state_anchor_x + badge_w),
            float(state_anchor_y + badge_h),
        )
        self._preview_super_correction_handle_bbox = (
            float(handle_x1),
            float(handle_y1),
            float(handle_x2),
            float(handle_y2),
        )
    except Exception:
        self._preview_super_correction_badge_bbox = None
        self._preview_super_correction_handle_bbox = None
        pass

    if not light_overlay:
        for vehicle_idx, det in enumerate(vehicle_detections):
            x1, y1, x2, y2 = det.bbox
            cx1, cy1 = canvas.image_to_canvas_coords(x1, y1)
            cx2, cy2 = canvas.image_to_canvas_coords(x2, y2)
            is_selected_vehicle = selected_vehicle_idx == vehicle_idx
            vehicle_outline = active_color if is_selected_vehicle else vehicle_color
            vehicle_width = 3 if is_selected_vehicle else 2
            vehicle_dash = None if is_selected_vehicle else (6, 4)
            canvas.create_rectangle(
                cx1,
                cy1,
                cx2,
                cy2,
                outline=vehicle_outline,
                width=vehicle_width,
                dash=vehicle_dash,
                tags=("preview_overlay",)
            )
            canvas.create_text(
                cx1 + 6,
                max(10, cy1 - 8),
                text=f"Vehicle {vehicle_idx + 1}/{len(vehicle_detections)}",
                fill=vehicle_outline,
                anchor="sw",
                font=("Segoe UI", 9, "bold"),
                tags=("preview_overlay",)
            )

    for plate_idx, det in enumerate(plate_detections):
        if drag_active and selected_plate_idx is not None and plate_idx != selected_plate_idx:
            continue
        polygon = self._detection_polygon(det)
        points = []
        for px, py in polygon:
            cx, cy = canvas.image_to_canvas_coords(px, py)
            points.extend([cx, cy])

        is_selected = selected_plate_idx == plate_idx
        is_delete_candidate = delete_candidate_idx == plate_idx
        if is_delete_candidate:
            canvas.create_polygon(
                points,
                outline="",
                fill=delete_color,
                stipple="gray25",
                width=0,
                tags=("preview_overlay",)
            )
        if is_delete_candidate:
            outline = delete_color
        elif image_approved:
            outline = approved_plate_color
        else:
            outline = active_color if is_selected else plate_color
        width = 2 if (is_selected or is_delete_candidate) else 1
        dash = None if (is_selected or is_delete_candidate) else (5, 3)
        canvas.create_polygon(
            points,
            outline=outline,
            fill="",
            width=width,
            dash=dash,
            tags=("preview_overlay", "preview_plate_outline")
        )

        if not light_overlay:
            min_x = min(points[0::2]) if points else 0
            min_y = min(points[1::2]) if points else 0
            fit_score = self._get_plate_detection_fit_score(det, ann)
            label_text = f"Det {float(getattr(det, 'confidence', 0.0) or 0.0):.2f}"
            if fit_score is not None:
                label_text += f" | Fit {fit_score:.2f}"
            label_id = canvas.create_text(
                min_x + 6,
                max(10, min_y - 7),
                text=label_text,
                fill=label_fill,
                anchor="sw",
                font=("Segoe UI", 9, "bold"),
                tags=("preview_overlay",)
            )
            label_bbox = canvas.bbox(label_id) or (
                min_x,
                max(0, min_y - 22),
                min_x + 132,
                max(18, min_y - 2),
            )
            rect_id = canvas.create_rectangle(
                min_x,
                max(0, min_y - 22),
                max(min_x + 132, float(label_bbox[2]) + 8.0),
                max(18, min_y - 2),
                outline="",
                fill=label_bg,
                tags=("preview_overlay",)
            )
            canvas.tag_lower(rect_id, label_id)

        if is_selected and not is_delete_candidate:
            zoom_level = max(0.01, float(getattr(canvas, "zoom_level", 1.0) or 1.0))
            radius = 5.0 if zoom_level <= 2.0 else min(8.0, 5.0 + ((zoom_level - 2.0) * 1.0))
            for vertex_idx, (px, py) in enumerate(polygon):
                cx, cy = canvas.image_to_canvas_coords(px, py)
                canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    outline=handle_outline,
                    fill=handle_fill,
                    width=1,
                    tags=("preview_overlay",)
                )
                if not light_overlay:
                    canvas.create_text(
                        cx,
                        cy - 12,
                        text=str(vertex_idx + 1),
                        fill=active_color,
                        font=("Segoe UI", 8, "bold"),
                        tags=("preview_overlay",)
                    )

    if self._preview_draw_mode and self._preview_draw_points:
        draw_points = []
        for idx, (px, py) in enumerate(self._preview_draw_points):
            cx, cy = canvas.image_to_canvas_coords(px, py)
            draw_points.extend([cx, cy])
            canvas.create_oval(
                cx - 5,
                cy - 5,
                cx + 5,
                cy + 5,
                outline="#111111",
                fill=active_color,
                width=1,
                tags=("preview_overlay",)
            )
            canvas.create_text(
                cx + 10,
                cy - 10,
                text=str(idx + 1),
                fill=active_color,
                font=("Segoe UI", 8, "bold"),
                anchor="sw",
                tags=("preview_overlay",)
            )

        if len(draw_points) >= 4:
            canvas.create_polygon(
                draw_points,
                outline=active_color,
                fill="",
                width=1,
                dash=(4, 2),
                tags=("preview_overlay",)
            )
        elif len(draw_points) >= 2:
            canvas.create_line(
                draw_points,
                fill=active_color,
                width=1,
                dash=(4, 2),
                tags=("preview_overlay",)
            )

    _draw_preview_plate_combo_overlay(
        self,
        canvas,
        plate_detections,
        selected_plate_idx,
        image_approved=bool(image_approved),
    )

    if not light_overlay:
        self._draw_preview_bottom_hint(canvas)


def _draw_preview_plate_combo_overlay(
    self,
    canvas: ZoomableCanvas,
    plate_detections: list,
    selected_plate_idx: int | None,
    *,
    image_approved: bool = False,
) -> None:
    try:
        canvas_width = max(1.0, float(canvas.winfo_width() or 1.0))
        canvas_height = max(1.0, float(canvas.winfo_height() or 1.0))
        if canvas_width < 180.0 or canvas_height < 90.0:
            return

        total = max(0, int(len(plate_detections or [])))
        if total > 0:
            try:
                current = int(selected_plate_idx if selected_plate_idx is not None else 0) + 1
            except Exception:
                current = 1
            current = max(1, min(int(current), int(total)))
            title_text = "TABLICA"
            value_text = f"{current} / {total}"
        else:
            title_text = "BRAK RAMKI"
            value_text = "0 / 0"

        theme = self._get_preview_legend_theme()
        panel_fill = str(theme.get("panel_fill", "#101820"))
        entry_fill = str(theme.get("entry_fill", "#172432"))
        muted = str(theme.get("section_muted", "#9fb0bd"))
        accent = str(theme.get("shell_outline", "#2fbf71"))
        warning = "#f1c40f"
        error = "#ff5b5b"
        is_ok = bool(image_approved) and total > 0
        value_fill = accent if is_ok else (warning if total > 0 else error)
        outline = value_fill
        status_text = "OK" if is_ok else "NOK"
        status_fill = "#21a765" if is_ok else "#d64545"
        status_outline = "#8ff0b8" if is_ok else "#ff9a9a"
        try:
            panel_is_light = bool(self._legend_color_is_light(panel_fill))
        except Exception:
            panel_is_light = False
        shadow_fill = blend_hex_colors(panel_fill, "#000000", 0.12 if panel_is_light else 0.42)
        font_cache = getattr(self, "_preview_plate_combo_font_cache", None)
        if not isinstance(font_cache, dict):
            font_cache = {}
            self._preview_plate_combo_font_cache = font_cache

        def _combo_font(size: int, weight: str, family: str):
            key = (str(family), int(size), str(weight))
            font_obj = font_cache.get(key)
            if font_obj is None:
                font_obj = tkfont.Font(self.frame, family=str(family), size=int(size), weight=str(weight))
                font_cache[key] = font_obj
            return font_obj

        label_font = _combo_font(8, "bold", "Segoe UI")
        value_font = _combo_font(24, "bold", "Bahnschrift SemiBold")
        status_font = _combo_font(18, "bold", "Bahnschrift SemiBold")
        title_w = float(label_font.measure(title_text))
        value_w = float(value_font.measure(value_text))
        status_w = float(status_font.measure(status_text))
        title_h = max(12.0, float(label_font.metrics("linespace") or 12))
        value_h = max(28.0, float(value_font.metrics("linespace") or 28))
        pad_x = 10.0
        pad_y = 5.0
        gap = 8.0
        title_box_w = max(72.0, title_w + (pad_x * 2.0))
        value_box_w = max(88.0, value_w + (pad_x * 2.0))
        status_box_w = max(70.0, status_w + (pad_x * 2.0))
        combo_w = title_box_w + gap + value_box_w + gap + status_box_w
        combo_h = max(34.0, value_h + (pad_y * 2.0))
        x1 = (canvas_width - combo_w) / 2.0
        y1 = 12.0
        x1 = max(12.0, min(x1, canvas_width - combo_w - 12.0))
        y1 = max(10.0, min(y1, canvas_height - combo_h - 10.0))
        x2 = x1 + combo_w
        y2 = y1 + combo_h
        title_x2 = x1 + title_box_w
        value_x1 = title_x2 + gap
        value_x2 = value_x1 + value_box_w
        status_x1 = value_x2 + gap

        shadow_offset = 2.0
        canvas.create_rectangle(
            x1 + shadow_offset,
            y1 + shadow_offset,
            x2 + shadow_offset,
            y2 + shadow_offset,
            outline="",
            fill=shadow_fill,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_rectangle(
            x1,
            y1,
            title_x2,
            y2,
            outline=outline,
            fill=entry_fill,
            width=1,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_rectangle(
            value_x1,
            y1,
            value_x2,
            y2,
            outline=outline,
            fill=panel_fill,
            width=2,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_rectangle(
            status_x1,
            y1,
            x2,
            y2,
            outline=status_outline,
            fill=status_fill,
            width=2,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_text(
            x1 + (title_box_w / 2.0),
            y1 + (combo_h / 2.0),
            text=title_text,
            fill=muted,
            anchor="center",
            font=label_font,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_text(
            value_x1 + (value_box_w / 2.0),
            y1 + (combo_h / 2.0) - 1.0,
            text=value_text,
            fill=value_fill,
            anchor="center",
            font=value_font,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.create_text(
            status_x1 + (status_box_w / 2.0),
            y1 + (combo_h / 2.0) - 1.0,
            text=status_text,
            fill="#f7fff9",
            anchor="center",
            font=status_font,
            tags=("preview_overlay", "preview_plate_combo_overlay"),
        )
        canvas.tag_raise("preview_plate_combo_overlay")
    except Exception:
        pass


def _build_preview_canvas_metrics_rows(self, *args, **kwargs):
    return z2_canvas_overlays._build_preview_canvas_metrics_rows(self, *args, **kwargs)


def _toggle_preview_metrics_overlay(self, *args, **kwargs):
    return z2_canvas_overlays._toggle_preview_metrics_overlay(self, *args, **kwargs)


def _on_preview_metrics_overlay_press(self, *args, **kwargs):
    return z2_canvas_overlays._on_preview_metrics_overlay_press(self, *args, **kwargs)


def _on_preview_metrics_overlay_drag(self, *args, **kwargs):
    return z2_canvas_overlays._on_preview_metrics_overlay_drag(self, *args, **kwargs)


def _on_preview_metrics_overlay_release(self, *args, **kwargs):
    return z2_canvas_overlays._on_preview_metrics_overlay_release(self, *args, **kwargs)


def _clamp_preview_metrics_overlay_offsets(self, *args, **kwargs):
    return z2_canvas_overlays._clamp_preview_metrics_overlay_offsets(self, *args, **kwargs)


def _render_preview_metrics_grab_handle(self, *args, **kwargs):
    return z2_canvas_overlays._render_preview_metrics_grab_handle(self, *args, **kwargs)


def _render_preview_metrics_table(self, *args, **kwargs):
    return z2_canvas_overlays._render_preview_metrics_table(self, *args, **kwargs)


def _update_preview_canvas_metrics_overlay(self, *args, **kwargs):
    return z2_canvas_overlays._update_preview_canvas_metrics_overlay(self, *args, **kwargs)


def _get_preview_bottom_hint_text(self) -> str:
    ann = self._get_preview_annotation()
    if ann is None:
        return ""

    if self._preview_fullscreen_active and not self._preview_is_editable():
        route = ""
        try:
            route = str(self._get_workflow_route() or "").strip().lower()
        except Exception:
            route = ""
        if route == "manual":
            return (
                "Ten podgląd jest jeszcze tylko informacyjny. Najpierw wyjdź z pełnego ekranu klawiszem Enter. "
                f"Po lewej stronie użyj {self._get_step2_start_action_reference()}, aby przygotować annotations.xml."
            )

    plates = self._get_plate_detections(ann)
    vehicles = self._get_vehicle_detections(ann)
    selected_vehicle_idx = self._get_selected_vehicle_index_for_ann(ann)
    vehicle_suffix = ""
    if vehicles:
        vehicle_no = 1 if selected_vehicle_idx is None else (int(selected_vehicle_idx) + 1)
        vehicle_suffix = f" Pojazd {vehicle_no}/{len(vehicles)}."

    if self._preview_draw_mode:
        clicked_points = len(self._preview_draw_points)
        next_idx = clicked_points + 1
        next_corner = self._preview_draw_corner_label(next_idx)
        base = f"Rysowanie ramki | kliknij {next_corner} ({next_idx}/4) | D anuluj"
    elif self._preview_delete_mode:
        if self._preview_delete_candidate_idx is not None:
            base = "Usuwanie | PPM usuwa | S anuluj"
        else:
            base = "Usuwanie | kliknij ramkę | PPM usuwa | S anuluj"
    elif not plates:
        base = "Brak ramki tablicy | D uzbrój rysowanie | kliknij 1. wierzchołek"
    else:
        selected_idx = self._get_selected_plate_index_for_ann(ann)
        plate_no = 0 if selected_idx is None else (int(selected_idx) + 1)
        nav_hint = (
            "Q/E tablice"
            if bool(getattr(self, "_preview_super_correction_active", False))
            else "Q/E zdjęcia"
        )
        super_hint = (
            "Y wyłącz"
            if bool(getattr(self, "_preview_super_correction_active", False))
            else "Y super korekta"
        )
        base = (
            f"Tablica {plate_no}/{len(plates)} | W+LPM róg | A lokalnie | {nav_hint} | "
            f"Spacja OK/NOK | R kadr | R+LPM zoom | R+PPM cofnij | F dopasuj | "
            f"{super_hint} | D nowa ramka | S usuń ramkę | Del obraz | Ctrl+Z/Y historia | Ctrl+S zapis"
        )

    if vehicle_suffix:
        return f"{base} | {vehicle_suffix.strip()}"
    return base


def _preview_hint_point_in_bbox(x: float, y: float, bbox) -> bool:
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except Exception:
        return False
    return x1 <= float(x) <= x2 and y1 <= float(y) <= y2


def _preview_bottom_hint_widget_bbox(canvas: ZoomableCanvas, widget) -> tuple[float, float, float, float] | None:
    try:
        if widget is None or not str(widget.winfo_manager()) or not bool(widget.winfo_ismapped()):
            return None
        canvas_root_x = float(canvas.winfo_rootx() or 0)
        canvas_root_y = float(canvas.winfo_rooty() or 0)
        x1 = float(widget.winfo_rootx() or 0) - canvas_root_x
        y1 = float(widget.winfo_rooty() or 0) - canvas_root_y
        width = float(widget.winfo_width() or widget.winfo_reqwidth() or 0)
        height = float(widget.winfo_height() or widget.winfo_reqheight() or 0)
        if width <= 0 or height <= 0:
            return None
        return (x1, y1, x1 + width, y1 + height)
    except Exception:
        return None


def _preview_bottom_hint_overlaps(
    x: float,
    y: float,
    width: float,
    height: float,
    blocker: tuple[float, float, float, float],
    *,
    padding: float = 8.0,
) -> bool:
    try:
        x1, y1, x2, y2 = [float(value) for value in blocker]
    except Exception:
        return False
    return (
        x < x2 + padding
        and x + width > x1 - padding
        and y < y2 + padding
        and y + height > y1 - padding
    )


def _preview_bottom_hint_is_clear(
    x: float,
    y: float,
    width: float,
    height: float,
    blockers: list[tuple[float, float, float, float]],
) -> bool:
    return not any(
        _preview_bottom_hint_overlaps(x, y, width, height, blocker)
        for blocker in blockers
    )


def _preview_bottom_hint_clamp(
    x: float,
    y: float,
    canvas_width: float,
    canvas_height: float,
    box_width: float,
    box_height: float,
    *,
    margin: float,
) -> tuple[float, float]:
    max_x = max(margin, float(canvas_width) - float(box_width) - margin)
    max_y = max(margin, float(canvas_height) - float(box_height) - margin)
    return (
        min(max(margin, float(x)), max_x),
        min(max(margin, float(y)), max_y),
    )


def _preview_bottom_hint_safe_position(
    self,
    x: float,
    y: float,
    canvas_width: float,
    canvas_height: float,
    box_width: float,
    box_height: float,
    *,
    margin: float = 14.0,
) -> tuple[float, float]:
    x, y = _preview_bottom_hint_clamp(
        x,
        y,
        canvas_width,
        canvas_height,
        box_width,
        box_height,
        margin=margin,
    )

    canvas = getattr(self, "preview_canvas", None)
    blockers: list[tuple[float, float, float, float]] = []
    if canvas is not None:
        for widget_name in ("preview_overlay_dock",):
            blocker = _preview_bottom_hint_widget_bbox(canvas, getattr(self, widget_name, None))
            if blocker is not None:
                blockers.append(blocker)

    if blockers and not _preview_bottom_hint_is_clear(x, y, box_width, box_height, blockers):
        candidates = [(x, y)]
        for bx1, by1, bx2, by2 in blockers:
            candidates.extend([
                (bx1 - float(box_width) - margin, y),
                (x, by2 + margin),
                (x, by1 - float(box_height) - margin),
                (margin, y),
                ((float(canvas_width) - float(box_width)) / 2.0, margin),
                (
                    (float(canvas_width) - float(box_width)) / 2.0,
                    float(canvas_height) - float(box_height) - margin,
                ),
            ])
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = _preview_bottom_hint_clamp(
                candidate_x,
                candidate_y,
                canvas_width,
                canvas_height,
                box_width,
                box_height,
                margin=margin,
            )
            if _preview_bottom_hint_is_clear(candidate_x, candidate_y, box_width, box_height, blockers):
                x, y = candidate_x, candidate_y
                break

    return float(x), float(y)


def _cycle_preview_bottom_hint_position(self) -> None:
    order = ("bottom", "right", "top", "left")
    current = str(getattr(self, "_preview_bottom_hint_position", "bottom") or "bottom")
    try:
        next_position = order[(order.index(current) + 1) % len(order)]
    except ValueError:
        next_position = "bottom"
    self._preview_bottom_hint_position = next_position
    self._preview_bottom_hint_collapsed = False
    self._refresh_preview_canvas(refresh_chrome=False)


def _toggle_preview_bottom_hint_collapsed(self, collapsed: bool | None = None) -> None:
    if collapsed is None:
        collapsed = not bool(getattr(self, "_preview_bottom_hint_collapsed", False))
    self._preview_bottom_hint_collapsed = bool(collapsed)
    self._preview_bottom_hint_drag_state = None
    self._refresh_preview_canvas(refresh_chrome=False)


def _shift_preview_bottom_hint_bbox(bbox, dx: float, dy: float):
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return bbox
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (x1 + float(dx), y1 + float(dy), x2 + float(dx), y2 + float(dy))
    except Exception:
        return bbox


def _begin_preview_bottom_hint_drag(self, canvas_x: float, canvas_y: float, event=None) -> bool:
    bbox = getattr(self, "_preview_bottom_hint_bbox", None)
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except Exception:
        return False
    self._preview_bottom_hint_drag_state = {
        "press_canvas_x": float(canvas_x),
        "press_canvas_y": float(canvas_y),
        "start_x": float(x1),
        "start_y": float(y1),
        "width": max(1.0, float(x2) - float(x1)),
        "height": max(1.0, float(y2) - float(y1)),
        "moved": False,
    }
    self._preview_bottom_hint_collapsed = False
    try:
        self.preview_canvas.configure(cursor="fleur")
    except Exception:
        pass
    return True


def _drag_preview_bottom_hint(self, event=None) -> bool:
    drag_state = getattr(self, "_preview_bottom_hint_drag_state", None)
    if not isinstance(drag_state, dict):
        return False
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        self._preview_bottom_hint_drag_state = None
        return False
    try:
        canvas_x = float(getattr(event, "canvas_x", getattr(event, "x", 0.0)) or 0.0)
        canvas_y = float(getattr(event, "canvas_y", getattr(event, "y", 0.0)) or 0.0)
        canvas_width = float(canvas.winfo_width() or 0)
        canvas_height = float(canvas.winfo_height() or 0)
        width = max(1.0, float(drag_state.get("width", 1.0) or 1.0))
        height = max(1.0, float(drag_state.get("height", 1.0) or 1.0))
        next_x = float(drag_state.get("start_x", 0.0) or 0.0) + (
            canvas_x - float(drag_state.get("press_canvas_x", canvas_x) or canvas_x)
        )
        next_y = float(drag_state.get("start_y", 0.0) or 0.0) + (
            canvas_y - float(drag_state.get("press_canvas_y", canvas_y) or canvas_y)
        )
        next_x, next_y = _preview_bottom_hint_safe_position(
            self,
            next_x,
            next_y,
            canvas_width,
            canvas_height,
            width,
            height,
        )
        current_bbox = getattr(self, "_preview_bottom_hint_bbox", None)
        if isinstance(current_bbox, tuple) and len(current_bbox) == 4:
            current_x = float(current_bbox[0])
            current_y = float(current_bbox[1])
        else:
            current_x = float(drag_state.get("start_x", 0.0) or 0.0)
            current_y = float(drag_state.get("start_y", 0.0) or 0.0)
        dx = float(next_x) - current_x
        dy = float(next_y) - current_y
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            canvas.move("preview_bottom_hint", dx, dy)
            for attr_name in (
                "_preview_bottom_hint_bbox",
                "_preview_bottom_hint_move_bbox",
                "_preview_bottom_hint_collapse_bbox",
                "_preview_bottom_hint_restore_bbox",
            ):
                setattr(
                    self,
                    attr_name,
                    _shift_preview_bottom_hint_bbox(getattr(self, attr_name, None), dx, dy),
                )
            drag_state["moved"] = True
        self._preview_bottom_hint_manual_position = (float(next_x), float(next_y))
        return True
    except Exception:
        return True


def _end_preview_bottom_hint_drag(self, event=None) -> bool:
    drag_state = getattr(self, "_preview_bottom_hint_drag_state", None)
    if not isinstance(drag_state, dict):
        return False
    self._preview_bottom_hint_drag_state = None
    try:
        self._sync_preview_canvas_cursor()
    except Exception:
        pass
    return True


def _handle_preview_bottom_hint_click(self, canvas_x: float, canvas_y: float, event=None) -> bool:
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        return False
    if _preview_hint_point_in_bbox(canvas_x, canvas_y, getattr(self, "_preview_bottom_hint_restore_bbox", None)):
        _toggle_preview_bottom_hint_collapsed(self, False)
        return True
    if _preview_hint_point_in_bbox(canvas_x, canvas_y, getattr(self, "_preview_bottom_hint_collapse_bbox", None)):
        _toggle_preview_bottom_hint_collapsed(self, True)
        return True
    if _preview_hint_point_in_bbox(canvas_x, canvas_y, getattr(self, "_preview_bottom_hint_move_bbox", None)):
        return _begin_preview_bottom_hint_drag(self, canvas_x, canvas_y, event)
    return False


def _preview_bottom_hint_origin(
    self,
    canvas_width: float,
    canvas_height: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float]:
    margin = 14.0
    manual_position = getattr(self, "_preview_bottom_hint_manual_position", None)
    if isinstance(manual_position, tuple) and len(manual_position) == 2:
        try:
            manual_x = float(manual_position[0])
            manual_y = float(manual_position[1])
            return _preview_bottom_hint_safe_position(
                self,
                manual_x,
                manual_y,
                canvas_width,
                canvas_height,
                box_width,
                box_height,
                margin=margin,
            )
        except Exception:
            self._preview_bottom_hint_manual_position = None

    position = str(getattr(self, "_preview_bottom_hint_position", "bottom") or "bottom")
    if position == "top":
        x = (float(canvas_width) - float(box_width)) / 2.0
        y = 54.0
    elif position == "left":
        x = margin
        y = max(54.0, (float(canvas_height) - float(box_height)) / 2.0)
    elif position == "right":
        x = float(canvas_width) - float(box_width) - margin
        y = max(54.0, (float(canvas_height) - float(box_height)) / 2.0)
    else:
        x = (float(canvas_width) - float(box_width)) / 2.0
        y = float(canvas_height) - float(box_height) - margin
    x, y = _preview_bottom_hint_clamp(
        x,
        y,
        canvas_width,
        canvas_height,
        box_width,
        box_height,
        margin=margin,
    )

    canvas = getattr(self, "preview_canvas", None)
    blockers: list[tuple[float, float, float, float]] = []
    if canvas is not None:
        for widget_name in ("preview_overlay_dock",):
            blocker = _preview_bottom_hint_widget_bbox(canvas, getattr(self, widget_name, None))
            if blocker is not None:
                blockers.append(blocker)

    if blockers and not _preview_bottom_hint_is_clear(x, y, box_width, box_height, blockers):
        candidates = [(x, y)]
        for bx1, by1, bx2, by2 in blockers:
            candidates.extend([
                (bx1 - float(box_width) - margin, y),
                (x, by2 + margin),
                (x, by1 - float(box_height) - margin),
                (margin, y),
                ((float(canvas_width) - float(box_width)) / 2.0, margin),
                (
                    (float(canvas_width) - float(box_width)) / 2.0,
                    float(canvas_height) - float(box_height) - margin,
                ),
            ])
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = _preview_bottom_hint_clamp(
                candidate_x,
                candidate_y,
                canvas_width,
                canvas_height,
                box_width,
                box_height,
                margin=margin,
            )
            if _preview_bottom_hint_is_clear(candidate_x, candidate_y, box_width, box_height, blockers):
                x, y = candidate_x, candidate_y
                break
    return float(x), float(y)


def _draw_preview_bottom_hint(self, canvas: ZoomableCanvas):
    if not bool(getattr(self, "_preview_fullscreen_active", False)):
        return
    try:
        main_pane = getattr(self, "main_pane", None)
        pane_ids = set(str(pane) for pane in (main_pane.panes() if main_pane is not None else ()))
        for frame in (getattr(self, "main_left_frame", None), getattr(self, "main_right_frame", None)):
            if frame is not None and str(frame) in pane_ids:
                return
    except Exception:
        pass

    hint_text = str(self._get_preview_bottom_hint_text() or "").strip()
    if not hint_text:
        return

    try:
        canvas_width = int(canvas.winfo_width() or 0)
        canvas_height = int(canvas.winfo_height() or 0)
    except Exception:
        canvas_width = 0
        canvas_height = 0
    if canvas_width <= 120 or canvas_height <= 80:
        return

    try:
        theme = self._get_preview_legend_theme()
    except Exception:
        theme = {}
    panel_fill = str(theme.get("panel_fill", "#101419"))
    outline = "#d6a73a" if self._manual_xml_template_enabled() else str(theme.get("shell_outline", "#56f29d"))
    chip_fill = str(theme.get("entry_fill", "#1b242d"))
    text_fill = str(theme.get("entry_text", "#f4f7fb"))
    muted_fill = str(theme.get("section_muted", "#c7d0db"))
    accent_fill = str(theme.get("badge_plate_outline", "#56f29d"))
    title_font = self._get_preview_legend_font(8, "bold")
    chip_font = self._get_preview_legend_font(8, "normal")
    control_font = self._get_preview_legend_font(7, "bold")

    collapsed = bool(getattr(self, "_preview_bottom_hint_collapsed", False))
    if collapsed:
        pill_w = 62.0
        pill_h = 26.0
        x, y = _preview_bottom_hint_origin(self, canvas_width, canvas_height, pill_w, pill_h)
        canvas.create_rectangle(
            x,
            y,
            x + pill_w,
            y + pill_h,
            fill=panel_fill,
            outline=outline,
            width=1,
            tags=("preview_overlay", "preview_bottom_hint"),
        )
        canvas.create_text(
            x + (pill_w / 2.0),
            y + (pill_h / 2.0),
            text="AS Z2",
            fill=text_fill,
            anchor="center",
            font=title_font,
            tags=("preview_overlay", "preview_bottom_hint"),
        )
        self._preview_bottom_hint_restore_bbox = (float(x), float(y), float(x + pill_w), float(y + pill_h))
        return

    raw_parts = [part.strip() for part in re.split(r"\s*\|\s*", hint_text) if part.strip()]
    if not raw_parts:
        return
    primary = raw_parts[0]
    parts = raw_parts[1:15]
    max_box_w = max(250.0, min(float(canvas_width) - 28.0, 560.0))
    min_box_w = min(max_box_w, 330.0)
    control_w = 23.0
    title_h = 24.0
    gap = 5.0
    chip_h = 22.0
    chip_pad_x = 8.0
    inner_pad = 10.0
    content_w = max(160.0, max_box_w - (inner_pad * 2.0))

    rows: list[list[tuple[str, float]]] = []
    current_row: list[tuple[str, float]] = []
    current_w = 0.0
    for part in parts:
        chip_w = min(content_w, max(46.0, float(chip_font.measure(part)) + (chip_pad_x * 2.0)))
        projected = chip_w if not current_row else current_w + gap + chip_w
        if current_row and projected > content_w:
            rows.append(current_row)
            current_row = []
            current_w = 0.0
        current_row.append((part, chip_w))
        current_w = chip_w if current_w <= 0.0 else current_w + gap + chip_w
    if current_row:
        rows.append(current_row)
    rows = rows[:2]

    title_w = float(title_font.measure(primary)) + 96.0
    row_w = 0.0
    for row in rows:
        row_w = max(row_w, sum(width for _text, width in row) + (gap * max(0, len(row) - 1)))
    box_w = min(max_box_w, max(min_box_w, title_w, row_w + (inner_pad * 2.0)))
    box_h = inner_pad + title_h + (len(rows) * chip_h) + (max(0, len(rows) - 1) * gap) + inner_pad
    x, y = _preview_bottom_hint_origin(self, canvas_width, canvas_height, box_w, box_h)

    canvas.create_rectangle(
        x,
        y,
        x + box_w,
        y + box_h,
        fill=panel_fill,
        outline=outline,
        width=1,
        tags=("preview_overlay", "preview_bottom_hint"),
    )
    title_y = y + inner_pad
    canvas.create_text(
        x + inner_pad,
        title_y + 2.0,
        text=primary,
        fill=text_fill,
        anchor="nw",
        font=title_font,
        tags=("preview_overlay", "preview_bottom_hint"),
    )

    move_x = x + box_w - inner_pad - (control_w * 2.0) - 4.0
    close_x = x + box_w - inner_pad - control_w
    control_y = title_y
    for label, bx in (("↔", move_x), ("×", close_x)):
        canvas.create_rectangle(
            bx,
            control_y,
            bx + control_w,
            control_y + 19.0,
            fill=chip_fill,
            outline=accent_fill if label == "↔" else muted_fill,
            width=1,
            tags=("preview_overlay", "preview_bottom_hint"),
        )
        canvas.create_text(
            bx + (control_w / 2.0),
            control_y + 9.5,
            text=label,
            fill=text_fill,
            anchor="center",
            font=control_font,
            tags=("preview_overlay", "preview_bottom_hint"),
        )
    canvas.create_rectangle(
        move_x,
        control_y,
        move_x + control_w,
        control_y + 19.0,
        fill=chip_fill,
        outline=accent_fill,
        width=1,
        tags=("preview_overlay", "preview_bottom_hint"),
    )
    dot_radius = 1.45
    for dot_x in (move_x + 8.2, move_x + control_w - 8.2):
        for dot_y in (control_y + 5.3, control_y + 9.5, control_y + 13.7):
            canvas.create_oval(
                dot_x - dot_radius,
                dot_y - dot_radius,
                dot_x + dot_radius,
                dot_y + dot_radius,
                fill=accent_fill,
                outline=accent_fill,
                width=1,
                tags=("preview_overlay", "preview_bottom_hint"),
            )
    self._preview_bottom_hint_move_bbox = (
        float(move_x),
        float(control_y),
        float(move_x + control_w),
        float(control_y + 19.0),
    )
    self._preview_bottom_hint_collapse_bbox = (
        float(close_x),
        float(control_y),
        float(close_x + control_w),
        float(control_y + 19.0),
    )

    chip_y = y + inner_pad + title_h
    for row in rows:
        row_total_w = sum(width for _text, width in row) + (gap * max(0, len(row) - 1))
        chip_x = x + inner_pad + max(0.0, (box_w - (inner_pad * 2.0) - row_total_w) / 2.0)
        for part, chip_w in row:
            canvas.create_rectangle(
                chip_x,
                chip_y,
                chip_x + chip_w,
                chip_y + chip_h,
                fill=chip_fill,
                outline=muted_fill,
                width=1,
                tags=("preview_overlay", "preview_bottom_hint"),
            )
            canvas.create_text(
                chip_x + (chip_w / 2.0),
                chip_y + (chip_h / 2.0),
                text=part,
                fill=text_fill,
                anchor="center",
                font=chip_font,
                tags=("preview_overlay", "preview_bottom_hint"),
            )
            chip_x += chip_w + gap
        chip_y += chip_h + gap

    self._preview_bottom_hint_bbox = (float(x), float(y), float(x + box_w), float(y + box_h))


def _clamp_preview_point(self, x: float, y: float) -> tuple[float, float]:
    if self.preview_canvas.original_image is None:
        return float(x), float(y)
    width = max(1.0, float(self.preview_canvas.original_image.width) - 1.0)
    height = max(1.0, float(self.preview_canvas.original_image.height) - 1.0)
    return (
        min(max(0.0, float(x)), width),
        min(max(0.0, float(y)), height),
    )


def _find_preview_vertex_hit(self, canvas_x: float, canvas_y: float):
    ann = self._get_preview_annotation()
    if ann is None:
        return None

    plate_detections = self._get_plate_detections(ann)
    if not plate_detections:
        return None

    handle_radius = self._get_preview_vertex_hit_radius()
    handle_radius_sq = float(handle_radius) * float(handle_radius)
    selected_idx = self._get_selected_plate_index_for_ann(ann)

    def _nearest_vertex_for_plate(plate_idx: int):
        polygon = self._detection_polygon(plate_detections[int(plate_idx)])
        best_vertex_idx = None
        best_dist_sq = None
        for vertex_idx, (px, py) in enumerate(polygon):
            point_x, point_y = self.preview_canvas.image_to_canvas_coords(px, py)
            dx = float(canvas_x) - float(point_x)
            dy = float(canvas_y) - float(point_y)
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq <= handle_radius_sq and (best_dist_sq is None or dist_sq < best_dist_sq):
                best_vertex_idx = int(vertex_idx)
                best_dist_sq = float(dist_sq)
        if best_vertex_idx is None:
            return None
        return int(plate_idx), int(best_vertex_idx), float(best_dist_sq or 0.0)

    if (
        self._preview_modifier_active()
        and selected_idx is not None
        and 0 <= int(selected_idx) < len(plate_detections)
    ):
        selected_hit = _nearest_vertex_for_plate(int(selected_idx))
        if selected_hit is not None:
            return int(selected_hit[0]), int(selected_hit[1])

    ordered_indices = list(range(len(plate_detections)))
    if selected_idx is not None and selected_idx in ordered_indices:
        ordered_indices.remove(selected_idx)
        ordered_indices.insert(0, selected_idx)

    best_hit = None
    best_dist_sq = None
    for plate_idx in ordered_indices:
        candidate = _nearest_vertex_for_plate(int(plate_idx))
        if candidate is None:
            continue
        candidate_plate_idx, candidate_vertex_idx, candidate_dist_sq = candidate
        if best_dist_sq is None or float(candidate_dist_sq) < float(best_dist_sq):
            best_hit = (candidate_plate_idx, candidate_vertex_idx)
            best_dist_sq = float(candidate_dist_sq)
    return best_hit


def _get_nearest_preview_vertex(self, plate_idx: int, canvas_x: float, canvas_y: float):
    ann = self._get_preview_annotation()
    if ann is None:
        return None

    plate_detections = self._get_plate_detections(ann)
    if plate_idx < 0 or plate_idx >= len(plate_detections):
        return None

    polygon = self._detection_polygon(plate_detections[plate_idx])
    best_vertex_idx = None
    best_dist = None
    for vertex_idx, (px, py) in enumerate(polygon):
        point_x, point_y = self.preview_canvas.image_to_canvas_coords(px, py)
        dist = math.hypot(float(canvas_x) - point_x, float(canvas_y) - point_y)
        if best_dist is None or dist < best_dist:
            best_vertex_idx = vertex_idx
            best_dist = dist

    if best_vertex_idx is None:
        return None
    return plate_idx, int(best_vertex_idx), float(best_dist if best_dist is not None else 0.0)


def _find_preview_polygon_hit(self, canvas_x: float, canvas_y: float):
    ann = self._get_preview_annotation()
    if ann is None or self.preview_canvas.original_image is None:
        return None

    img_x, img_y = self.preview_canvas.canvas_to_image_coords(canvas_x, canvas_y, clamp=False)
    plate_detections = self._get_plate_detections(ann)
    if not plate_detections:
        return None

    threshold_img = max(2.0, 8.0 / max(0.01, float(self.preview_canvas.zoom_level)))
    best_idx = None
    best_dist = None

    for plate_idx, det in enumerate(plate_detections):
        polygon = np.array(self._detection_polygon(det), dtype=np.float32)
        try:
            dist = cv2.pointPolygonTest(polygon, (float(img_x), float(img_y)), True)
        except Exception:
            continue
        if dist >= -threshold_img and (best_dist is None or dist > best_dist):
            best_idx = plate_idx
            best_dist = dist

    return best_idx


def _find_preview_vehicle_hit(self, canvas_x: float, canvas_y: float):
    ann = self._get_preview_annotation()
    if ann is None or self.preview_canvas.original_image is None:
        return None

    img_x, img_y = self.preview_canvas.canvas_to_image_coords(canvas_x, canvas_y, clamp=False)
    vehicle_detections = self._get_vehicle_detections(ann)
    if not vehicle_detections:
        return None

    zoom_level = max(0.01, float(getattr(self.preview_canvas, "zoom_level", 1.0) or 1.0))
    threshold_img = max(3.0, 8.0 / zoom_level)
    best_idx = None
    best_area = None

    for vehicle_idx, det in enumerate(vehicle_detections):
        x1, y1, x2, y2 = [float(v) for v in det.bbox[:4]]
        if (
            (x1 - threshold_img) <= float(img_x) <= (x2 + threshold_img)
            and (y1 - threshold_img) <= float(img_y) <= (y2 + threshold_img)
        ):
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if best_area is None or area < best_area:
                best_idx = vehicle_idx
                best_area = area

    return best_idx


def _get_global_nearest_preview_vertex(self, canvas_x: float, canvas_y: float):
    ann = self._get_preview_annotation()
    if ann is None:
        return None

    plate_detections = self._get_plate_detections(ann)
    best_candidate = None
    best_dist = None
    for plate_idx in range(len(plate_detections)):
        candidate = self._get_nearest_preview_vertex(plate_idx, canvas_x, canvas_y)
        if candidate is None:
            continue

        _plate_idx, _vertex_idx, dist = candidate
        if best_dist is None or dist < best_dist:
            best_candidate = candidate
            best_dist = dist

    return best_candidate


def _get_preview_vertex_canvas_position(self, plate_idx: int, vertex_idx: int):
    ann = self._get_preview_annotation()
    if ann is None:
        return None

    plate_detections = self._get_plate_detections(ann)
    if plate_idx < 0 or plate_idx >= len(plate_detections):
        return None

    polygon = self._detection_polygon(plate_detections[plate_idx])
    if vertex_idx < 0 or vertex_idx >= len(polygon):
        return None

    px, py = polygon[vertex_idx]
    return self.preview_canvas.image_to_canvas_coords(px, py)


def _describe_preview_hit_debug(self, canvas_x: float, canvas_y: float, event=None) -> str:
    if self.preview_canvas.original_image is None:
        return ""

    img_x, img_y = self.preview_canvas.canvas_to_image_coords(canvas_x, canvas_y, clamp=False)
    handle_radius = self._get_preview_vertex_hit_radius()

    parts = [
        f"img=({float(img_x):.1f},{float(img_y):.1f})",
        f"hit<={handle_radius:.1f}",
    ]
    if event is not None:
        norm_source = str(getattr(event, "norm_source", "raw") or "raw")
        raw_x = getattr(event, "raw_x", None)
        raw_y = getattr(event, "raw_y", None)
        pointer_local_x = getattr(event, "pointer_local_x", None)
        pointer_local_y = getattr(event, "pointer_local_y", None)
        event_local_x = getattr(event, "event_local_x", None)
        event_local_y = getattr(event, "event_local_y", None)
        canvas_event_x = getattr(event, "canvas_x", None)
        canvas_event_y = getattr(event, "canvas_y", None)
        canvas_offset_x = getattr(event, "canvas_offset_x", None)
        canvas_offset_y = getattr(event, "canvas_offset_y", None)
        parts.append(f"src={norm_source}")
        if raw_x is not None and raw_y is not None:
            parts.append(f"raw=({float(raw_x):.1f},{float(raw_y):.1f})")
        if event_local_x is not None and event_local_y is not None:
            parts.append(f"event_local=({float(event_local_x):.1f},{float(event_local_y):.1f})")
        if pointer_local_x is not None and pointer_local_y is not None:
            parts.append(f"pointer_local=({float(pointer_local_x):.1f},{float(pointer_local_y):.1f})")
        if canvas_event_x is not None and canvas_event_y is not None:
            parts.append(f"canvas_event=({float(canvas_event_x):.1f},{float(canvas_event_y):.1f})")
        if canvas_offset_x is not None and canvas_offset_y is not None:
            parts.append(f"canvas_offset=({float(canvas_offset_x):.1f},{float(canvas_offset_y):.1f})")

    ann = self._get_preview_annotation()
    selected_idx = self._get_selected_plate_index_for_ann(ann) if ann is not None else None
    selected_candidate = None
    if selected_idx is not None:
        selected_candidate = self._get_nearest_preview_vertex(int(selected_idx), canvas_x, canvas_y)
        if selected_candidate is not None:
            plate_idx, vertex_idx, dist = selected_candidate
            point = self._get_preview_vertex_canvas_position(int(plate_idx), int(vertex_idx))
            if point is not None:
                point_x, point_y = point
                parts.append(
                    f"sel={self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} "
                    f"dist={float(dist):.1f} pt=({float(point_x):.1f},{float(point_y):.1f})"
                )

    global_candidate = self._get_global_nearest_preview_vertex(canvas_x, canvas_y)
    if global_candidate is not None:
        plate_idx, vertex_idx, dist = global_candidate
        is_same_as_selected = (
            selected_candidate is not None
            and int(selected_candidate[0]) == int(plate_idx)
            and int(selected_candidate[1]) == int(vertex_idx)
        )
        if not is_same_as_selected:
            point = self._get_preview_vertex_canvas_position(int(plate_idx), int(vertex_idx))
            if point is not None:
                point_x, point_y = point
                parts.append(
                    f"global={self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} "
                    f"dist={float(dist):.1f} pt=({float(point_x):.1f},{float(point_y):.1f})"
                )

    return " ".join(parts)


def _describe_selected_polygon_vertices_debug(self) -> str:
    ann = self._get_preview_annotation()
    if ann is None:
        return "verts=-"

    selected_idx = self._get_selected_plate_index_for_ann(ann)
    if selected_idx is None:
        return "verts=-"

    plate_detections = self._get_plate_detections(ann)
    if selected_idx < 0 or selected_idx >= len(plate_detections):
        return "verts=-"

    polygon = self._detection_polygon(plate_detections[selected_idx])
    parts = []
    for vertex_idx, (img_x, img_y) in enumerate(polygon):
        canvas_point = self._get_preview_vertex_canvas_position(int(selected_idx), int(vertex_idx))
        if canvas_point is None:
            continue
        canvas_x, canvas_y = canvas_point
        parts.append(
            f"v{int(vertex_idx) + 1}=img({float(img_x):.1f},{float(img_y):.1f})/canvas({float(canvas_x):.1f},{float(canvas_y):.1f})"
        )

    if not parts:
        return "verts=-"
    return f"verts[p{int(selected_idx) + 1}]: " + " ".join(parts)


def _get_preview_vertex_hit_radius(self) -> float:
    zoom_level = max(0.01, float(getattr(self.preview_canvas, "zoom_level", 1.0) or 1.0))
    if self._preview_modifier_active():
        # Przy duzym zoomie klik jest "blisko" rogu w obrazie, ale daleko w pikselach canvasa.
        # Skalujemy hitbox z zoomem, zamiast trzymac sztywny promien ekranowy.
        return max(12.0, min(30.0, 6.0 * zoom_level))
    return 8.0


def _resolve_preview_drag_target(self, canvas_x: float, canvas_y: float):
    vertex_hit = self._find_preview_vertex_hit(canvas_x, canvas_y)
    if vertex_hit is not None:
        plate_idx, vertex_idx = vertex_hit
        return int(plate_idx), int(vertex_idx), "handle"
    return None


def _finish_preview_vertex_drag(self, mark_dirty: bool = True):
    drag_state = self._preview_drag_state
    ann = self._get_preview_annotation()
    sequence_active = bool(self._preview_modifier_active())
    if not isinstance(drag_state, dict) or ann is None:
        try:
            self.preview_canvas.grab_release()
        except Exception:
            pass
        self._cancel_preview_drag_refresh()
        self._preview_drag_state = None
        self._preview_pending_vertex_hit = None
        self._refresh_preview_canvas()
        self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
        return False

    plate_detections = self._get_plate_detections(ann)
    plate_idx = int(drag_state.get("plate_idx", -1))
    if 0 <= plate_idx < len(plate_detections):
        det = plate_detections[plate_idx]
        current_polygon = self._detection_polygon(det)
        # Przy trzymanym W uzytkownik zwykle poprawia kilka naroznikow pod rzad.
        # Ciezsza normalizacja wraca po zakonczeniu serii, zeby kolejny chwyt
        # nie czekal na prace UI po poprzednim rogu.
        fixed_polygon = current_polygon if sequence_active else PolygonValidator.fix_polygon(current_polygon)
        det.polygon = fixed_polygon
        det.bbox = self._bbox_from_polygon(fixed_polygon)
        det.keypoints = self._keypoints_from_polygon(fixed_polygon)
        det.attributes["manually_edited"] = "true"
        det.attributes["manual_source"] = "preview"
        if mark_dirty:
            # Podczas seryjnej korekty rogĂłw nie odswiezamy od razu calej listy
            # wynikow, bo to bylo odczuwalne wlasnie przed drugim chwytem.
            self._mark_preview_image_dirty(
                ann,
                refresh_list=False,
                refresh_row=False,
                invalidate_runtime=not sequence_active,
            )
            try:
                self._refresh_preview_list_row_for_actual_index(
                    self.current_preview_index,
                    refresh_summary=False,
                )
            except Exception:
                pass

    if not sequence_active:
        try:
            self.preview_canvas.grab_release()
        except Exception:
            pass
    self._cancel_preview_drag_refresh()
    self._preview_drag_state = None
    self._preview_pending_vertex_hit = None
    if sequence_active:
        self._cancel_preview_post_interaction_refresh()
        self._refresh_preview_canvas_light()
    else:
        self._refresh_preview_canvas_interactive(delay_ms=420)
        self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
    if mark_dirty:
        # Przytrzymane W oznacza sesje szybkiej korekty wielu rogĂłw.
        # Nie zapisujemy wtedy po kazdym puszczeniu myszy, bo to wcinalo
        # sie w chwyt kolejnego punktu. Zapis wraca po puszczeniu W.
        if not sequence_active:
            self._schedule_preview_autosave(delay_ms=6500)
    if not sequence_active:
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass
    return True


def _persist_preview_structural_change(self, success_message: str, fallback_message: str | None = None):
    try:
        self.preview_canvas.update_idletasks()
    except Exception:
        pass
    try:
        self.frame.update_idletasks()
    except Exception:
        pass

    dirty_before_save = len(self._preview_dirty_images or set())
    save_started = time.perf_counter()
    if self._save_preview_edits():
        elapsed_ms = max(0.0, (time.perf_counter() - save_started) * 1000.0)
        if elapsed_ms >= 100.0:
            logger.debug(
                "[AnnotationTab][PERF] preview_structural_change_save: "
                f"{elapsed_ms:.1f} ms | dirty={dirty_before_save}"
            )
        self._update_preview_edit_status(success_message)
        return True

    elapsed_ms = max(0.0, (time.perf_counter() - save_started) * 1000.0)
    if elapsed_ms >= 100.0:
        logger.debug(
            "[AnnotationTab][PERF] preview_structural_change_save: "
            f"{elapsed_ms:.1f} ms | failed=1 dirty={dirty_before_save}"
        )
    self._update_preview_edit_status(
        fallback_message
        or "Zmiana została wprowadzona w podglądzie, ale nie udało się od razu zapisać annotations.xml. Użyj Ctrl+S."
    )
    return False


def _delete_preview_polygon(self, plate_idx: int, autosave: bool = True):
    ann = self._get_preview_annotation()
    if ann is None:
        return False

    plate_detections = self._get_plate_detections(ann)
    if plate_idx < 0 or plate_idx >= len(plate_detections):
        return False

    had_plate_before = bool(self._get_plate_detections(ann))
    self._push_preview_history_snapshot(ann, lightweight_plate_edit=True)
    target_detection = plate_detections[int(plate_idx)]
    try:
        ann.detections.remove(target_detection)
    except ValueError:
        return False

    remaining_plates = self._get_plate_detections(ann)
    if remaining_plates:
        self._set_selected_plate_index_for_ann(ann, min(int(plate_idx), len(remaining_plates) - 1))
        ann.status = AnnotationStatus.SUCCESS
    else:
        self._set_selected_plate_index_for_ann(ann, None)
        ann.status = AnnotationStatus.NO_PLATE

    ann.status_message = "Polygon tablicy usuniety recznie."
    self._preview_drag_state = None
    self._preview_pending_vertex_hit = None
    self._preview_delete_mode = False
    self._preview_delete_candidate_idx = None
    self._mark_preview_image_dirty(ann, refresh_list=False, refresh_row=False)
    try:
        self._refresh_preview_list_row_for_actual_index(
            self.current_preview_index,
            refresh_summary=False,
            lightweight=True,
        )
    except Exception:
        pass
    self._refresh_preview_canvas_interactive(delay_ms=220)
    self._push_preview_debug_event("delete", f"p{int(plate_idx) + 1}")

    if autosave:
        self._schedule_preview_autosave(
            delay_ms=4500,
            status_message="Zapisano usunięcie ramki tablicy do annotations.xml.",
        )
        self._update_preview_edit_status("Usunięto ramkę tablicy. Zapis nastąpi za chwilę.")
        return True

    self._update_preview_edit_status("Usunięto ramkę tablicy. Użyj Ctrl+S, aby zapisać zmianę do annotations.xml.")
    return True


def _commit_new_preview_polygon(self):
    ann = self._get_preview_annotation()
    if ann is None or len(self._preview_draw_points) != 4:
        return

    points = [self._clamp_preview_point(x, y) for x, y in self._preview_draw_points[:4]]
    fixed_points = PolygonValidator.fix_polygon(points)
    if not PolygonValidator.is_valid_quad(fixed_points):
        self._preview_draw_points = []
        self._update_preview_edit_status("Nowa ramka jest zbyt mała albo nieprawidłowa. Spróbuj ponownie.")
        self._refresh_preview_canvas()
        return

    self._push_preview_history_snapshot(ann, lightweight_plate_edit=True)
    had_plate_before = bool(self._get_plate_detections(ann))
    new_det = Detection(
        label="plate",
        confidence=1.0,
        bbox=self._bbox_from_polygon(fixed_points),
        keypoints=self._keypoints_from_polygon(fixed_points),
        polygon=fixed_points,
    )
    new_det.attributes["manually_edited"] = "true"
    new_det.attributes["manual_source"] = "preview"
    ann.detections.append(new_det)
    try:
        count_cache = getattr(self, "_current_preview_plate_count_cache", None)
        if isinstance(count_cache, dict):
            if not had_plate_before:
                count_cache["images_with_plates"] = max(
                    0,
                    int(count_cache.get("images_with_plates", 0) or 0) + 1,
                )
            count_cache["total_plates"] = max(
                0,
                int(count_cache.get("total_plates", 0) or 0) + 1,
            )
            self._current_preview_plate_count_cache = count_cache
    except Exception:
        pass
    ann.status = AnnotationStatus.SUCCESS
    ann.status_message = "Dodano ręcznie ramkę tablicy."

    plates = self._get_plate_detections(ann)
    self._set_selected_plate_index_for_ann(ann, len(plates) - 1)
    # Domknięcie ramki ma być natychmiast widoczne na canvasie.
    # Pelny refresh listy zostawiamy na etap zapisu annotations.xml.
    self._mark_preview_image_dirty(
        ann,
        refresh_list=False,
        refresh_row=False,
        invalidate_runtime=False,
    )
    self._preview_draw_mode = False
    self._preview_draw_points = []
    self._refresh_preview_canvas(refresh_chrome=False)
    self._schedule_preview_post_interaction_refresh(delay_ms=900)
    self._push_preview_debug_event("add", f"p{len(plates)}")
    self._update_preview_edit_status(
        "Dodano nową ramkę tablicy. Możesz od razu poprawić rogi; zapis nastąpi za chwilę.",
        refresh_toolbar=False,
        refresh_debug=False,
    )

    def _refresh_commit_chrome() -> None:
        try:
            self._refresh_preview_list_row_for_actual_index(
                self.current_preview_index,
                refresh_summary=False,
            )
        except Exception:
            pass
        try:
            self._update_preview_toolbar_state(refresh_summary=False)
        except Exception:
            pass

    try:
        self.frame.after(180, _refresh_commit_chrome)
    except Exception:
        _refresh_commit_chrome()
    if self._is_free_mode_session_context():
        try:
            self._sync_main_pane_right_panel_visibility()
            self._refresh_free_mode_manual_right_panel(reuse_existing_tables=True)
        except Exception:
            pass
        try:
            self._refresh_free_mode_manual_review_export_controls()
        except Exception:
            pass
    self._schedule_preview_autosave(
        delay_ms=6500,
        status_message="Zapisano nową ramkę tablicy do annotations.xml.",
    )


def _refresh_free_mode_manual_review_export_controls(self) -> None:
    if not self._is_free_mode_session_context():
        return
    try:
        if self._get_workflow_route() != "manual":
            return
        if str(self._coerce_free_mode_screen() or "").strip().lower() != "manual_review":
            return
    except Exception:
        return

    try:
        approved_ready = bool(self._get_plate_dataset_export_approval_state().get("ok"))
    except Exception:
        approved_ready = False
    try:
        annotation_ready = bool(self._get_plate_annotation_package_export_state().get("ok"))
    except Exception:
        annotation_ready = False
    export_enabled = bool(not self.is_processing and (approved_ready or annotation_ready))
    back_enabled = bool(not self.is_processing)
    extract_enabled = bool(not self.is_processing and approved_ready)

    try:
        self.workflow_back_btn.configure(
            state=(tk.NORMAL if back_enabled else tk.DISABLED),
            text="Wstecz",
            width=10,
        )
    except Exception:
        pass
    try:
        self.workflow_next_btn.configure(
            state=tk.DISABLED,
            text="",
        )
        self.workflow_next_btn.grid_remove()
    except Exception:
        pass
    try:
        self.manual_stage_use_btn.configure(
            text="Wyodrębnij tablice",
            command=self._open_step3_from_z2_annotation_source,
            state=(tk.NORMAL if extract_enabled else tk.DISABLED),
        )
    except Exception:
        pass
    try:
        self.manual_stage_add_btn.configure(
            text="Otwórz eksport",
            command=self._start_z2_export_choice_flow,
            state=(tk.NORMAL if export_enabled else tk.DISABLED),
        )
    except Exception:
        pass
    try:
        self.export_plate_dataset_btn.configure(
            text="EKSPORT",
            command=self._start_z2_export_choice_flow,
            state=(tk.NORMAL if export_enabled else tk.DISABLED),
        )
    except Exception:
        pass


def _get_current_annotation_xml_path(self) -> Path | None:
    candidates = []

    if getattr(self, "current_annotation_xml_path", None):
        try:
            current_xml_path = Path(self.current_annotation_xml_path)
            safe_run_dir = self._resolve_safe_annotation_run_dir(current_xml_path.parent)
            if safe_run_dir is not None and current_xml_path.name.lower() == "annotations.xml":
                candidates.append(safe_run_dir / "annotations.xml")
        except Exception:
            pass

    for run_candidate in (
        getattr(self, "current_annotation_run_dir", None),
        getattr(self, "last_staging_run_dir", None),
        str(self.plate_dataset_run_var.get() or "").strip(),
    ):
        safe_run_dir = self._resolve_safe_annotation_run_dir(run_candidate)
        if safe_run_dir is not None:
            candidates.append(safe_run_dir / "annotations.xml")

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists() or candidate.parent.exists():
                return candidate
        except Exception:
            continue

    return candidates[0] if candidates else None


def _ensure_preview_edits_saved(self, action_label: str) -> bool:
    self._flush_preview_approved_persist()
    if not self._preview_dirty_images:
        return True

    if self._save_preview_edits():
        return True

    messagebox.showerror(
        "Błąd zapisu poprawek",
        f"Nie udało się zapisać zmian przed operacją: {action_label}."
    )
    return False


def _cancel_preview_autosave(self):
    pending = getattr(self, "_preview_autosave_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._preview_autosave_after_id = None


def _schedule_preview_autosave(
    self,
    delay_ms: int = 900,
    *,
    status_message: str | None = None,
    refresh_workflow: bool = False,
    refresh_export_sources: bool = False,
):
    try:
        requested_delay_ms = int(delay_ms)
    except Exception:
        requested_delay_ms = 900
    requested_delay_ms = max(requested_delay_ms, PREVIEW_CORRECTION_AUTOSAVE_DELAY_MS)
    if bool(getattr(self, "_preview_super_correction_active", False)):
        # In super correction the user often moves from corner to corner and
        # plate to plate very quickly.  Saving the whole CVAT XML for a 9k-image
        # run in the middle of that loop blocks Tk and looks like a corner-drag
        # lag.  Keep the safety autosave, but only after a real quiet window.
        requested_delay_ms = max(requested_delay_ms, SUPER_CORRECTION_AUTOSAVE_DELAY_MS)
        try:
            logger.info(
                "[Z2 AUTOSAVE] defer super-correction delay=%sms dirty=%s",
                requested_delay_ms,
                len(getattr(self, "_preview_dirty_images", set()) or set()),
            )
        except Exception:
            pass
    self._cancel_preview_autosave()

    def _flush_autosave() -> None:
        self._preview_autosave_after_id = None
        quiet_remaining = 0
        try:
            quiet_remaining = self._preview_user_interaction_quiet_remaining_ms(padding_ms=350)
        except Exception:
            quiet_remaining = 0
        if quiet_remaining > 0:
            self._schedule_preview_autosave(
                delay_ms=quiet_remaining,
                status_message=status_message,
                refresh_workflow=bool(refresh_workflow),
                refresh_export_sources=bool(refresh_export_sources),
            )
            return
        self._save_preview_edits(
            interactive=False,
            status_message=status_message or "Zapisano korekte polygonu do annotations.xml.",
            refresh_list=False,
            refresh_workflow=bool(refresh_workflow),
            refresh_export_sources=bool(refresh_export_sources),
        )

    try:
        self._preview_autosave_after_id = self.frame.after(
            int(requested_delay_ms),
            _flush_autosave,
        )
    except Exception:
        self._preview_autosave_after_id = None


def _defer_preview_autosave_for_navigation(self, delay_ms: int = 4500):
    if not getattr(self, "_preview_autosave_after_id", None):
        return
    if not self._preview_dirty_images:
        return
    delay_ms = max(int(delay_ms), PREVIEW_CORRECTION_AUTOSAVE_DELAY_MS)
    if bool(getattr(self, "_preview_super_correction_active", False)):
        delay_ms = max(int(delay_ms), SUPER_CORRECTION_AUTOSAVE_DELAY_MS)
    self._schedule_preview_autosave(delay_ms=max(2500, int(delay_ms)))


def _refresh_preview_list_row_for_actual_index(
    self,
    actual_index: int | None,
    *,
    refresh_summary: bool = True,
    lightweight: bool = False,
) -> None:
    if actual_index is None or not self.current_annotations:
        return
    try:
        safe_actual_index = int(actual_index)
    except Exception:
        return
    if safe_actual_index < 0 or safe_actual_index >= len(self.current_annotations):
        return

    display_index = self._get_preview_display_index(safe_actual_index)
    if display_index is None:
        return
    try:
        ann = self.current_annotations[safe_actual_index]
    except Exception:
        return
    try:
        cache = getattr(self, "_preview_list_render_state_cache", None)
        if isinstance(cache, dict):
            cache.pop(id(ann), None)
    except Exception:
        pass

    label_text = self._preview_list_item_text(
        ann,
        display_index=int(display_index),
        total_count=len(getattr(self, "_preview_list_display_indices", []) or self.current_annotations or []),
        lightweight=bool(lightweight),
    )
    try:
        self.preview_listbox.delete(display_index)
        self.preview_listbox.insert(display_index, label_text)
        item_color = self._preview_list_item_color(ann)
        self.preview_listbox.itemconfig(display_index, foreground=item_color)
    except Exception:
        return

    try:
        if self.current_preview_index is not None and int(self.current_preview_index) == safe_actual_index:
            self._clear_listbox_selection_fast(self.preview_listbox)
            self.preview_listbox.selection_set(display_index)
            self.preview_listbox.activate(display_index)
            self.preview_listbox.see(display_index)
    except Exception:
        pass
    _refresh_preview_group_selection_anchor_style(self)

    if refresh_summary:
        try:
            self._update_preview_toolbar_state(refresh_summary=True)
        except Exception:
            pass


def _remove_image_from_stage_manifest(self, image_path: Path):
    if image_path is None or not self._is_manual_plate_stage_input(image_path.parent):
        return

    manifest_path = self._get_manual_plate_stage_dir() / "stage_manifest.json"
    manifest = self.dataset_creator._load_stage_manifest(manifest_path)
    entries = manifest.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}

    stage_key = self.dataset_creator._path_key(image_path)
    filtered_entries = {}
    for entry_key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage_name") or "").strip()
        if entry_key == stage_key or stage_name == image_path.name:
            continue
        filtered_entries[entry_key] = entry

    stage_images = get_image_files(self._get_manual_plate_stage_images_dir())
    manifest["entries"] = filtered_entries
    manifest["pending_images"] = len(stage_images)
    manifest["stage_images_total"] = len(stage_images)
    manifest["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        self.dataset_creator._save_stage_manifest(manifest_path, manifest)
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć manifestu stage po usunięciu obrazu: {e}")


def _save_preview_edits(self, *args, **kwargs):
    return z2_workflow_methods._save_preview_edits(self, *args, **kwargs)


def _delete_current_preview_image_hard(self, event=None):
    if event is not None and not self._preview_shortcuts_enabled(event, allow_when_fullscreen=True):
        return None
    if not self._ensure_preview_is_editable_for_action():
        self._update_preview_edit_status(
            "Usuwanie obrazu bedzie dostepne dopiero po przygotowaniu annotations.xml."
        )
        return "break" if event is not None else False

    ann = self._get_preview_annotation()
    if ann is None or self.current_input_dir is None:
        return "break" if event is not None else False

    image_path = Path(self.current_input_dir) / str(getattr(ann, "filename", "") or "")
    delete_index = int(self.current_preview_index or 0)
    backup_annotations = list(self.current_annotations)
    backup_dirty = set(self._preview_dirty_images)
    backup_selected_plate = dict(self._preview_selected_plate_by_image)
    backup_selected_vehicle = dict(self._preview_selected_vehicle_by_image)

    self.current_annotations = [item for idx, item in enumerate(self.current_annotations) if idx != delete_index]
    self._preview_dirty_images.discard(str(getattr(ann, "filename", "") or ""))
    self._preview_selected_plate_by_image.pop(str(getattr(ann, "filename", "") or ""), None)
    self._preview_selected_vehicle_by_image.pop(str(getattr(ann, "filename", "") or ""), None)

    if not self._save_preview_edits(interactive=False, status_message="Usunieto obraz z annotations.xml."):
        self.current_annotations = backup_annotations
        self._preview_dirty_images = backup_dirty
        self._preview_selected_plate_by_image = backup_selected_plate
        self._preview_selected_vehicle_by_image = backup_selected_vehicle
        self._refresh_preview_list(preserve_selection=True, render_current=True)
        if interactive:
            messagebox.showerror("Błąd usuwania", "Nie udało się usunąć wpisu obrazu z annotations.xml.")
        return "break" if event is not None else False

    try:
        if image_path.exists():
            image_path.unlink()
    except Exception as e:
        self.current_annotations = backup_annotations
        self._preview_dirty_images = backup_dirty
        self._preview_selected_plate_by_image = backup_selected_plate
        self._preview_selected_vehicle_by_image = backup_selected_vehicle
        self._preview_dirty_images.add(str(getattr(ann, "filename", "") or ""))
        self._save_preview_edits(interactive=False)
        self._refresh_preview_list(preserve_selection=True, render_current=True)
        messagebox.showerror("Błąd usuwania", f"Nie udało się usunąć pliku z dysku:\n{image_path}\n\n{e}")
        return "break" if event is not None else False

    self._remove_image_from_stage_manifest(image_path)
    self._refresh_manual_plate_stage_ui()

    if self.current_annotations:
        new_index = min(delete_index, len(self.current_annotations) - 1)
        self.current_preview_index = new_index
        self._refresh_preview_list(preserve_selection=True, render_current=True)
        self._select_preview_index(new_index)
    else:
        self._clear_preview_editor_state(clear_dirty=True)
        self.preview_listbox.delete(0, tk.END)
        self._update_preview_edit_status("Usunieto ostatnie zdjecie z aktywnego runu.")

    self._refresh_plate_dataset_export_sources()
    self._refresh_step2_action_states()
    self._refresh_free_mode_workflow_ui()
    self._update_preview_edit_status("Usunieto zdjecie z dysku i z annotations.xml.")
    return "break" if event is not None else True


def _move_current_preview_image_to_stage(self):
    ann = self._get_preview_annotation()
    if ann is None or self.current_input_dir is None:
        return False

    image_path = Path(self.current_input_dir) / str(getattr(ann, "filename", "") or "")
    if not image_path.exists():
        messagebox.showerror("Brak obrazu", f"Plik nie istnieje:\n{image_path}")
        return False

    if self._is_manual_plate_stage_input(image_path.parent):
        messagebox.showinfo("Stage", "To zdjecie jest juz w stage.")
        return False

    stage_dir = self._get_manual_plate_stage_dir()
    temp_dir = stage_dir / "_incoming_preview" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_copy = temp_dir / image_path.name

    try:
        shutil.copy2(image_path, temp_copy)
        ok, msg, _stats = self.dataset_creator.add_images_to_stage(temp_dir, stage_dir)
    except Exception as e:
        ok = False
        msg = str(e)
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    if not ok:
        messagebox.showerror("Błąd stage", f"Nie udało się przenieść obrazu do stage:\n{msg}")
        return False

    delete_ok = self._delete_current_preview_image_hard()
    if not delete_ok:
        messagebox.showwarning(
            "Czesciowe przeniesienie",
            "Obraz zostal skopiowany do stage, ale nie udalo sie usunac go z biezacego runu."
        )
        return False

    self._refresh_manual_plate_stage_ui()
    self._update_preview_edit_status("Przeniesiono obraz do stage i usunieto go z aktywnego runu.")
    return True


def _on_preview_canvas_right_click(self, event):
    canvas = getattr(self, "preview_canvas", None)
    if canvas is None:
        return None

    if not bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        try:
            canvas.focus_set()
        except Exception:
            pass

    ann = self._get_preview_annotation()
    if ann is None or canvas.original_image is None:
        return "break"

    canvas_x = float(canvas.canvasx(getattr(event, "x", 0.0)))
    canvas_y = float(canvas.canvasy(getattr(event, "y", 0.0)))

    if bool(getattr(self, "_preview_focus_zoom_modifier_down", False)):
        self._preview_focus_zoom_modifier_consumed = True
        if self._restore_preview_focus_zoom_view():
            self._update_preview_edit_status(
                "Przywrócono kadr sprzed płynnego zoomu.",
                refresh_toolbar=False,
                refresh_debug=False,
            )
        else:
            self._update_preview_edit_status(
                "Brak wcześniejszego kadru do przywrócenia dla R + PPM.",
                refresh_toolbar=False,
                refresh_debug=False,
            )
        return "break"

    if self._preview_draw_mode:
        self._preview_draw_mode = False
        self._preview_draw_points = []
        self._refresh_preview_canvas()
        self._push_preview_debug_event("draw-cancel", "PPM anulowal nowy polygon")
        self._update_preview_edit_status("Anulowano rysowanie nowego polygonu.")
        return "break"

    if not self._preview_delete_mode:
        self._update_preview_edit_status(
            "Usuwanie polygonu jest dwuetapowe: naciśnij S, kliknij wewnątrz polygonu, a potem PPM usunie zaznaczoną tablicę."
        )
        return "break"

    target_idx = self._preview_delete_candidate_idx
    if target_idx is None:
        self._update_preview_edit_status(
            "Tryb usuwania jest aktywny. Kliknij najpierw wewnątrz polygonu, aby zaznaczyć go na czerwono."
        )
        return "break"

    self._set_selected_plate_index_for_ann(ann, int(target_idx))
    self._push_preview_debug_event("delete-request", f"p{int(target_idx) + 1} PPM")
    self._delete_preview_polygon(int(target_idx), autosave=True)
    return "break"


def on_zoomable_canvas_press(self, canvas: ZoomableCanvas, event):
    press_started = time.perf_counter()
    phase_at = press_started
    phase_ms: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal phase_at
        now = time.perf_counter()
        phase_ms[name] = max(0.0, (now - phase_at) * 1000.0)
        phase_at = now

    if canvas is not self.preview_canvas:
        return False
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass

    ann = self._get_preview_annotation()
    if ann is None or canvas.original_image is None:
        return False

    canvas_x = float(getattr(event, "canvas_x", getattr(event, "x", 0.0)))
    canvas_y = float(getattr(event, "canvas_y", getattr(event, "y", 0.0)))
    mark_phase("coords")

    if _handle_preview_bottom_hint_click(self, canvas_x, canvas_y, event):
        return True

    if not bool(getattr(self, "_preview_corner_drag_modifier_down", False)):
        try:
            canvas.focus_set()
        except Exception:
            pass
    mark_phase("focus")

    if self._is_preview_fullscreen_toggle_hit(canvas_x, canvas_y):
        self._toggle_preview_fullscreen()
        return True

    if self._is_preview_super_correction_handle_hit(canvas_x, canvas_y):
        bbox = getattr(self, "_preview_super_correction_badge_bbox", None)
        if isinstance(bbox, tuple) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:
                x1, y1, x2, y2 = (0.0, 0.0, 0.0, 0.0)
            self._preview_super_correction_drag_state = {
                "press_canvas_x": float(canvas_x),
                "press_canvas_y": float(canvas_y),
                "start_x": float(x1),
                "start_y": float(y1),
                "width": max(1.0, float(x2 - x1)),
                "height": max(1.0, float(y2 - y1)),
            }
            self._sync_preview_canvas_cursor()
            return True
    if self._is_preview_super_correction_badge_hit(canvas, canvas_x, canvas_y):
        return bool(self._toggle_preview_super_correction())

    if self._preview_draw_mode:
        if not canvas.point_is_inside_image(canvas_x, canvas_y):
            return True

        point = self._clamp_preview_point(*canvas.canvas_to_image_coords(canvas_x, canvas_y, clamp=True))
        self._preview_draw_points.append(point)
        if len(self._preview_draw_points) >= 4:
            self._commit_new_preview_polygon()
        else:
            self._refresh_preview_canvas_interactive(delay_ms=360)
            self._update_preview_edit_status()
        return True

    if self._preview_delete_mode:
        polygon_hit = self._find_preview_polygon_hit(canvas_x, canvas_y)
        if polygon_hit is not None:
            self._preview_delete_candidate_idx = int(polygon_hit)
            self._preview_pending_vertex_hit = None
            self._set_selected_plate_index_for_ann(ann, int(polygon_hit))
            self._push_preview_debug_event(
                "delete-select",
                (
                    f"p{int(polygon_hit) + 1} canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                    f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
                )
            )
            self._refresh_preview_canvas_interactive(delay_ms=360)
            self._update_preview_edit_status(
                f"Polygon {int(polygon_hit) + 1}/{len(self._get_plate_detections(ann))} jest zaznaczony do usunięcia. Kliknij PPM, aby go usunąć."
            )
            return True

        self._preview_delete_candidate_idx = None
        self._push_preview_debug_event(
            "delete-miss",
            (
                f"canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
            )
        )
        self._refresh_preview_canvas_interactive(delay_ms=360)
        self._update_preview_edit_status(
            "Tryb usuwania aktywny: kliknij wewnątrz polygonu, aby zaznaczyć tablicę do usunięcia."
        )
        return True

    if bool(getattr(self, "_preview_focus_zoom_modifier_down", False)):
        self._preview_focus_zoom_modifier_consumed = True
        if not canvas.point_is_inside_image(canvas_x, canvas_y):
            return True
        try:
            current_view_state = canvas.get_view_state()
            if isinstance(current_view_state, dict):
                history = list(getattr(self, "_preview_focus_zoom_history", []) or [])
                history.append(current_view_state)
                history = history[-4:]
                self._preview_focus_zoom_history = history
                self._preview_focus_zoom_restore_state = history[0] if history else None
            target_zoom = float(self._get_preview_focus_zoom_target(canvas))
            canvas._animate_zoom_to(canvas_x, canvas_y, target_zoom, duration_ms=170)
            self._preview_focus_zoom_click_stage = min(
                2,
                int(getattr(self, "_preview_focus_zoom_click_stage", 0) or 0) + 1,
            )
        except Exception:
            pass
        return True

    if self._preview_modifier_active():
        resolve_started = time.perf_counter()
        drag_target = self._resolve_preview_drag_target(canvas_x, canvas_y)
        mark_phase("resolve_target")
        if drag_target is not None:
            plate_idx, vertex_idx, target_mode = drag_target
            previously_selected_plate = self._get_selected_plate_index_for_ann(ann)
            self._set_selected_plate_index_for_ann(ann, plate_idx)
            mark_phase("select_plate")
            self._begin_preview_vertex_drag(plate_idx, vertex_idx, canvas_x, canvas_y)
            mark_phase("begin_drag")
            if bool(getattr(self, "_preview_debug_enabled", False)):
                self._push_preview_debug_event(
                    "press-target",
                    (
                        f"{self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} "
                        f"mode={target_mode} canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                        f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
                    )
                )
            else:
                self._push_preview_debug_event(
                    "press-target",
                    f"{self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} mode={target_mode}"
                )
            if previously_selected_plate != int(plate_idx):
                self._refresh_preview_canvas_light()
                mark_phase("refresh_light")
            elapsed_ms = max(0.0, (time.perf_counter() - press_started) * 1000.0)
            _log_preview_super_perf(
                self,
                "press.delegate_z2",
                elapsed_ms,
                threshold_ms=0.0 if _preview_super_perf_active(self) else 50.0,
                hit=1,
                mode=target_mode,
                plate=plate_idx,
                vertex=vertex_idx,
                coords=f"{phase_ms.get('coords', 0.0):.1f}",
                focus=f"{phase_ms.get('focus', 0.0):.1f}",
                resolve=f"{phase_ms.get('resolve_target', 0.0):.1f}",
                select=f"{phase_ms.get('select_plate', 0.0):.1f}",
                begin=f"{phase_ms.get('begin_drag', 0.0):.1f}",
                refresh=f"{phase_ms.get('refresh_light', 0.0):.1f}",
                resolve_total=f"{(time.perf_counter() - resolve_started) * 1000.0:.1f}",
            )
            return True
        elapsed_ms = max(0.0, (time.perf_counter() - press_started) * 1000.0)
        _log_preview_super_perf(
            self,
            "press.delegate_z2",
            elapsed_ms,
            threshold_ms=0.0 if _preview_super_perf_active(self) else 50.0,
            hit=0,
            coords=f"{phase_ms.get('coords', 0.0):.1f}",
            focus=f"{phase_ms.get('focus', 0.0):.1f}",
            resolve=f"{phase_ms.get('resolve_target', 0.0):.1f}",
        )

    vertex_hit = self._find_preview_vertex_hit(canvas_x, canvas_y)
    if vertex_hit is not None:
        plate_idx, vertex_idx = vertex_hit
        self._preview_pending_vertex_hit = None
        self._set_selected_plate_index_for_ann(ann, plate_idx)
        self._push_preview_debug_event(
            "press-handle",
            (
                f"bez W {self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} "
                f"canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
            )
        )
        self._refresh_preview_canvas_interactive(delay_ms=360)
        self._update_preview_edit_status(
            "Aby przesuwać rogi tablicy, przytrzymaj W i przeciągaj uchwyt myszą.",
            refresh_toolbar=False,
            refresh_debug=False,
        )
        return True

    polygon_hit = self._find_preview_polygon_hit(canvas_x, canvas_y)
    if polygon_hit is not None:
        self._preview_pending_vertex_hit = None
        self._set_selected_plate_index_for_ann(ann, polygon_hit)
        self._push_preview_debug_event(
            "press-polygon",
            (
                f"p{int(polygon_hit) + 1} canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
            )
        )
        self._refresh_preview_canvas_interactive(delay_ms=360)
        self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
        return True

    vehicle_hit = self._find_preview_vehicle_hit(canvas_x, canvas_y)
    if vehicle_hit is not None:
        self._preview_pending_vertex_hit = None
        self._set_selected_vehicle_index_for_ann(ann, vehicle_hit)
        self._push_preview_debug_event(
            "press-vehicle",
            (
                f"v{int(vehicle_hit) + 1} canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
            )
        )
        self._refresh_preview_canvas_interactive(delay_ms=360)
        self._update_preview_edit_status(
            f"Wybrano pojazd {int(vehicle_hit) + 1}/{len(self._get_vehicle_detections(ann))}. "
            "Wybrany pojazd może pomóc przy ręcznym dodawaniu tablicy w jego obszarze.",
            refresh_toolbar=False,
            refresh_debug=False,
        )
        return False

    self._preview_pending_vertex_hit = None
    if self._preview_corner_drag_modifier_down:
        self._push_preview_debug_event(
            "press-miss",
            (
                f"W=on canvas=({float(canvas_x):.1f},{float(canvas_y):.1f}) "
                f"{self._describe_preview_hit_debug(canvas_x, canvas_y, event)}"
            )
        )
        self._update_preview_edit_status(refresh_toolbar=False, refresh_debug=False)
        return True

    return False


def on_zoomable_canvas_drag(self, canvas: ZoomableCanvas, event):
    if canvas is not self.preview_canvas:
        return False
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass

    canvas_x = float(getattr(event, "canvas_x", getattr(event, "x", 0.0)))
    canvas_y = float(getattr(event, "canvas_y", getattr(event, "y", 0.0)))
    if _drag_preview_bottom_hint(self, event):
        return True
    super_drag = getattr(self, "_preview_super_correction_drag_state", None)
    if isinstance(super_drag, dict):
        badge_w = max(1.0, float(super_drag.get("width", 1.0)))
        badge_h = max(1.0, float(super_drag.get("height", 1.0)))
        canvas_w = max(1.0, float(canvas.winfo_width() or 1.0))
        canvas_h = max(1.0, float(canvas.winfo_height() or 1.0))
        next_x = float(super_drag.get("start_x", 0.0)) + (float(canvas_x) - float(super_drag.get("press_canvas_x", canvas_x)))
        next_y = float(super_drag.get("start_y", 0.0)) + (float(canvas_y) - float(super_drag.get("press_canvas_y", canvas_y)))
        self._preview_super_correction_badge_offset_x = min(max(0.0, next_x), max(0.0, canvas_w - badge_w))
        self._preview_super_correction_badge_offset_y = min(max(0.0, next_y), max(0.0, canvas_h - badge_h))
        self._refresh_preview_canvas()
        return True
    drag_state = self._preview_drag_state
    ann = self._get_preview_annotation()
    if isinstance(drag_state, dict) and (ann is None or canvas.original_image is None):
        return self._finish_preview_vertex_drag(mark_dirty=False)

    if not isinstance(drag_state, dict) or ann is None or canvas.original_image is None:
        pending_hit = self._preview_pending_vertex_hit
        if (
            isinstance(pending_hit, dict)
            and ann is not None
            and canvas.original_image is not None
            and self._preview_modifier_active()
        ):
            plate_idx = int(pending_hit.get("plate_idx", -1))
            vertex_idx = int(pending_hit.get("vertex_idx", -1))
            anchor_canvas_x = float(pending_hit.get("anchor_canvas_x", canvas_x))
            anchor_canvas_y = float(pending_hit.get("anchor_canvas_y", canvas_y))
            if plate_idx >= 0 and vertex_idx >= 0:
                self._set_selected_plate_index_for_ann(ann, plate_idx)
                self._begin_preview_vertex_drag(plate_idx, vertex_idx, anchor_canvas_x, anchor_canvas_y)
                self._push_preview_debug_event(
                    "drag-begin",
                    f"{self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} anchor=({anchor_canvas_x:.1f},{anchor_canvas_y:.1f})"
                )
                drag_state = self._preview_drag_state
            else:
                self._preview_pending_vertex_hit = None
                if self._preview_corner_drag_modifier_down:
                    return True
                return False
        else:
            if self._preview_corner_drag_modifier_down:
                return True
            return False

    if not isinstance(drag_state, dict):
        if self._preview_corner_drag_modifier_down:
            return True
        return False

    plate_detections = self._get_plate_detections(ann)
    plate_idx = int(drag_state.get("plate_idx", -1))
    vertex_idx = int(drag_state.get("vertex_idx", -1))
    if plate_idx < 0 or plate_idx >= len(plate_detections):
        return self._finish_preview_vertex_drag(mark_dirty=False)

    det = plate_detections[plate_idx]
    polygon = self._detection_polygon(det)
    if vertex_idx < 0 or vertex_idx >= len(polygon):
        return self._finish_preview_vertex_drag(mark_dirty=False)

    press_canvas_x = float(drag_state.get("press_canvas_x", canvas_x))
    press_canvas_y = float(drag_state.get("press_canvas_y", canvas_y))
    if not bool(drag_state.get("drag_started", False)):
        motion_canvas_dist = math.hypot(float(canvas_x) - press_canvas_x, float(canvas_y) - press_canvas_y)
        # Nizszy prog sprawia, ze chwyt zaczyna dzialac praktycznie od razu.
        if motion_canvas_dist < 1.5:
            self._push_preview_debug_event("drag-threshold", f"d={motion_canvas_dist:.2f}", refresh_only=True)
            return True
        drag_state["drag_started"] = True
        image_key = str(self._get_preview_history_image_key(ann) or "").strip()
        if not image_key or not self._preview_modifier_active():
            self._push_preview_history_snapshot(ann)
        elif str(getattr(self, "_preview_drag_session_history_image_key", "") or "") != image_key:
            self._push_preview_history_snapshot(ann, lightweight_plate_edit=True)
            self._preview_drag_session_history_image_key = image_key
        self._push_preview_debug_event(
            "drag-start",
            f"{self._format_preview_debug_vertex_ref(plate_idx, vertex_idx)} d={motion_canvas_dist:.2f}"
        )

    cursor_x, cursor_y = canvas.canvas_to_image_coords(canvas_x, canvas_y, clamp=True)
    start_cursor_x = float(drag_state.get("start_cursor_x", cursor_x))
    start_cursor_y = float(drag_state.get("start_cursor_y", cursor_y))
    start_vertex_x = float(drag_state.get("start_vertex_x", polygon[vertex_idx][0]))
    start_vertex_y = float(drag_state.get("start_vertex_y", polygon[vertex_idx][1]))

    polygon[vertex_idx] = self._clamp_preview_point(
        start_vertex_x + (float(cursor_x) - start_cursor_x),
        start_vertex_y + (float(cursor_y) - start_cursor_y),
    )
    det.polygon = polygon
    # Bbox i keypointy przeliczamy dopiero po puszczeniu myszy. W trakcie
    # dragu liczy się płynny podgląd narożnika, nie pełna normalizacja detekcji.
    det.attributes["manually_edited"] = "true"
    det.attributes["manual_source"] = "preview"
    ann.status = AnnotationStatus.SUCCESS
    ann.status_message = "Poligon tablicy poprawiony recznie."
    self._preview_last_modifier_press_at = time.monotonic()
    drag_state["was_moved"] = True

    # Zapis wykonujemy dopiero po zakonczeniu przeciagania, zeby nie mielic XML-a
    # przy kazdym ruchu kursora i nie powodowac lagow podczas edycji.
    self._schedule_preview_drag_refresh(delay_ms=24)
    now = time.monotonic()
    if (now - float(getattr(self, "_preview_debug_last_drag_update_at", 0.0) or 0.0)) >= 0.2:
        self._preview_debug_last_drag_update_at = now
        self._push_preview_debug_event("drag-move", refresh_only=True)
    return True


def on_zoomable_canvas_release(self, canvas: ZoomableCanvas, event):
    if canvas is not self.preview_canvas:
        return False
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass

    if _end_preview_bottom_hint_drag(self, event):
        return True

    if isinstance(getattr(self, "_preview_super_correction_drag_state", None), dict):
        self._preview_super_correction_drag_state = None
        self._sync_preview_canvas_cursor()
        return True

    if self._preview_drag_state is None:
        if self._preview_pending_vertex_hit is not None:
            self._preview_pending_vertex_hit = None
            self._refresh_preview_canvas()
            self._update_preview_edit_status()
            self._push_preview_debug_event("release", "pending cleared without drag")
            return True
        return False

    self._push_preview_debug_event(
        "release",
        f"moved={int(bool(self._preview_drag_state.get('was_moved', False)))} "
        f"{self._format_preview_debug_vertex_ref(self._preview_drag_state.get('plate_idx'), self._preview_drag_state.get('vertex_idx'))}"
    )
    return self._finish_preview_vertex_drag(
        mark_dirty=bool(self._preview_drag_state.get("was_moved", False))
    )


def _preview_auto_scope_modal_blocks_list(self) -> bool:
    modal_open = bool(getattr(self, "_plate_auto_scope_modal_open", False))
    selection_mode = bool(getattr(self, "_plate_auto_scope_selection_mode_active", False))
    if not modal_open and not selection_mode:
        return False

    dialog = getattr(self, "_plate_auto_scope_active_dialog", None)
    dialog_alive = False
    try:
        dialog_alive = bool(dialog is not None and dialog.winfo_exists())
    except Exception:
        dialog_alive = False
    if dialog_alive:
        return bool(modal_open and not selection_mode)

    # Defensive cleanup: stale modal flags must not block normal list selection.
    try:
        self._plate_auto_scope_modal_open = False
        self._plate_auto_scope_active_dialog = None
        self._plate_auto_scope_selection_mode_active = False
    except Exception:
        pass
    try:
        self._set_plate_auto_scope_selection_mode(False)
    except Exception:
        pass
    try:
        self._update_preview_toolbar_state(refresh_summary=False)
    except Exception:
        pass
    return False


def _preview_listbox_color_to_hex(listbox, color: str, fallback: str) -> str:
    raw = str(color or "").strip()
    if raw.startswith("#") and len(raw.lstrip("#")) in (3, 6):
        return raw
    try:
        red, green, blue = listbox.winfo_rgb(raw)
        return "#{:02x}{:02x}{:02x}".format(red // 256, green // 256, blue // 256)
    except Exception:
        return str(fallback or "#000000")


def _get_preview_group_anchor_select_bg(self) -> str:
    listbox = getattr(self, "preview_listbox", None)
    try:
        raw_select_bg = str(listbox.cget("selectbackground") or "#2f6f9f") if listbox is not None else "#2f6f9f"
    except Exception:
        raw_select_bg = "#2f6f9f"
    select_bg = _preview_listbox_color_to_hex(listbox, raw_select_bg, "#2f6f9f") if listbox is not None else "#2f6f9f"
    try:
        palette = getattr(getattr(self, "app", None), "palette", {}) or {}
        raw_accent = palette.get("warning", palette.get("accent", "#d39b35"))
    except Exception:
        raw_accent = "#d39b35"
    accent = _preview_listbox_color_to_hex(listbox, raw_accent, "#d39b35") if listbox is not None else "#d39b35"
    return blend_hex_colors(select_bg, accent, 0.32)


def _reset_preview_group_selection_anchor_style(self) -> None:
    listbox = getattr(self, "preview_listbox", None)
    previous_index = getattr(self, "_preview_group_selection_anchor_styled_index", None)
    self._preview_group_selection_anchor_styled_index = None
    if listbox is None or previous_index is None:
        return
    try:
        previous_index = int(previous_index)
        size = int(listbox.size() or 0)
    except Exception:
        return
    if previous_index < 0 or previous_index >= size:
        return
    try:
        select_bg = str(listbox.cget("selectbackground") or "#2f6f9f")
        select_fg = str(listbox.cget("selectforeground") or "#ffffff")
        listbox.itemconfig(previous_index, selectbackground=select_bg, selectforeground=select_fg)
    except Exception:
        pass


def _set_preview_group_selection_anchor(self, display_index: int | None) -> None:
    if display_index is None:
        self._preview_group_selection_anchor_display_index = None
        _reset_preview_group_selection_anchor_style(self)
        return
    try:
        self._preview_group_selection_anchor_display_index = int(display_index)
    except Exception:
        self._preview_group_selection_anchor_display_index = None
        _reset_preview_group_selection_anchor_style(self)
        return
    _refresh_preview_group_selection_anchor_style(self)


def _refresh_preview_group_selection_anchor_style(self) -> None:
    listbox = getattr(self, "preview_listbox", None)
    if listbox is None:
        return
    try:
        size = int(listbox.size() or 0)
        selected = {int(idx) for idx in (listbox.curselection() or ())}
    except Exception:
        return
    if len(selected) <= 1:
        self._preview_group_selection_anchor_display_index = None
        _reset_preview_group_selection_anchor_style(self)
        return
    try:
        anchor_index = int(getattr(self, "_preview_group_selection_anchor_display_index", -1))
    except Exception:
        anchor_index = -1
    if anchor_index not in selected:
        try:
            anchor_index = int(listbox.index(tk.ANCHOR))
        except Exception:
            anchor_index = min(selected)
    if anchor_index not in selected or anchor_index < 0 or anchor_index >= size:
        anchor_index = min(selected)
    previous_index = getattr(self, "_preview_group_selection_anchor_styled_index", None)
    if previous_index is not None:
        try:
            previous_matches = int(previous_index) == int(anchor_index)
        except Exception:
            previous_matches = False
        if not previous_matches:
            _reset_preview_group_selection_anchor_style(self)
    try:
        select_fg = str(listbox.cget("selectforeground") or "#ffffff")
        listbox.itemconfig(
            int(anchor_index),
            selectbackground=_get_preview_group_anchor_select_bg(self),
            selectforeground=select_fg,
        )
        self._preview_group_selection_anchor_display_index = int(anchor_index)
        self._preview_group_selection_anchor_styled_index = int(anchor_index)
    except Exception:
        pass


def _on_preview_list_mouse_primary(self, event):
    if _preview_auto_scope_modal_blocks_list(self):
        return "break"
    try:
        self._on_preview_canvas_focus_out()
    except Exception:
        pass
    try:
        self.preview_listbox.focus_set()
    except Exception:
        pass
    try:
        modifier_state = int(getattr(event, "state", 0) or 0)
    except Exception:
        modifier_state = 0

    try:
        target_index = int(self.preview_listbox.nearest(getattr(event, "y", 0)))
    except Exception:
        return None

    try:
        size = int(self.preview_listbox.size() or 0)
    except Exception:
        size = 0
    if size <= 0 or target_index < 0 or target_index >= size:
        return "break"

    shift_pressed = bool(modifier_state & 0x0001)
    control_pressed = bool(modifier_state & 0x0004)
    preserve_preview = False

    try:
        if shift_pressed:
            self._suppress_preview_reload_on_list_select = True
            try:
                anchor_index = int(self.preview_listbox.index(tk.ANCHOR))
            except Exception:
                anchor_index = -1
            if anchor_index < 0 or anchor_index >= size:
                try:
                    anchor_index = int(self.preview_listbox.index(tk.ACTIVE))
                except Exception:
                    anchor_index = target_index
            anchor_index = max(0, min(anchor_index, size - 1))
            start_index = min(anchor_index, target_index)
            end_index = max(anchor_index, target_index)
            if not control_pressed:
                self._clear_listbox_selection_fast(self.preview_listbox)
            self.preview_listbox.selection_set(start_index, end_index)
            self.preview_listbox.selection_anchor(anchor_index)
            self.preview_listbox.activate(anchor_index)
            _set_preview_group_selection_anchor(self, anchor_index)
            preserve_preview = True
        elif control_pressed:
            self._suppress_preview_reload_on_list_select = True
            if self.preview_listbox.selection_includes(target_index):
                self.preview_listbox.selection_clear(target_index)
            else:
                self.preview_listbox.selection_set(target_index)
            self.preview_listbox.selection_anchor(target_index)
            _refresh_preview_group_selection_anchor_style(self)
            preserve_preview = True
        else:
            self._suppress_preview_reload_on_list_select = False
            _set_preview_group_selection_anchor(self, None)
            self._clear_listbox_selection_fast(self.preview_listbox)
            self.preview_listbox.selection_set(target_index)
            self.preview_listbox.selection_anchor(target_index)
            self.preview_listbox.activate(target_index)
            self.preview_listbox.see(target_index)
    except Exception:
        return "break"

    if preserve_preview:
        self._suppress_preview_reload_on_list_select = True
        _refresh_preview_group_selection_anchor_style(self)
        self._refresh_plate_auto_scope_modal_selection_state()
        return "break"

    self._on_preview_select(None, defer_render=True)
    return "break"


def _refresh_preview_list_context_menu_state(
    self,
    *,
    has_visible_rows: bool,
    has_group_selection: bool,
    can_rename_single: bool,
    any_selected_unapproved: bool,
    any_selected_approved: bool,
    can_clear_auto: bool,
    scope_selection_mode_active: bool,
) -> None:
    menu = getattr(self, "preview_list_context_menu", None)
    if menu is None:
        return

    def _state(enabled: bool) -> str:
        return tk.NORMAL if enabled else tk.DISABLED

    try:
        menu.entryconfigure("Zaznacz obrazy po filtrze", state=_state(has_visible_rows and not scope_selection_mode_active))
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Oznacz zaznaczone jako OK",
            state=_state(has_group_selection and any_selected_unapproved and not scope_selection_mode_active),
        )
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Cofnij OK",
            state=_state(has_group_selection and any_selected_approved and not scope_selection_mode_active),
        )
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Zmień nazwę pliku...",
            state=_state(can_rename_single and not scope_selection_mode_active),
        )
    except Exception:
        pass
    try:
        menu.entryconfigure(
            "Wyczyść auto w całym runie",
            state=_state(can_clear_auto and not scope_selection_mode_active),
        )
    except Exception:
        pass


def _show_preview_list_context_menu(self, x_root: int, y_root: int) -> str:
    menu = getattr(self, "preview_list_context_menu", None)
    if menu is None:
        return "break"

    try:
        self._update_preview_toolbar_state(refresh_summary=False)
    except Exception:
        pass

    try:
        menu.tk_popup(int(x_root), int(y_root))
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass
    return "break"


def _open_preview_list_context_menu_from_keyboard(self, _event=None):
    listbox = getattr(self, "preview_listbox", None)
    if listbox is None:
        return "break"

    try:
        selected = list(listbox.curselection() or ())
    except Exception:
        selected = []
    if selected:
        target_index = int(selected[-1])
    else:
        try:
            target_index = int(listbox.index(tk.ACTIVE))
        except Exception:
            target_index = 0
    try:
        bbox = listbox.bbox(target_index)
    except Exception:
        bbox = None
    if bbox:
        x_root = int(listbox.winfo_rootx() + bbox[0] + min(40, max(16, bbox[2] // 2)))
        y_root = int(listbox.winfo_rooty() + bbox[1] + bbox[3])
    else:
        x_root = int(listbox.winfo_rootx() + 24)
        y_root = int(listbox.winfo_rooty() + 24)
    return self._show_preview_list_context_menu(x_root, y_root)


def _on_preview_list_mouse_secondary(self, event):
    try:
        self._on_preview_canvas_focus_out()
    except Exception:
        pass
    try:
        self.preview_listbox.focus_set()
    except Exception:
        pass

    try:
        target_index = int(self.preview_listbox.nearest(getattr(event, "y", 0)))
        size = int(self.preview_listbox.size() or 0)
    except Exception:
        return "break"

    if size > 0 and 0 <= target_index < size:
        try:
            if not self.preview_listbox.selection_includes(target_index):
                _set_preview_group_selection_anchor(self, None)
                self._clear_listbox_selection_fast(self.preview_listbox)
                self.preview_listbox.selection_set(target_index)
                self.preview_listbox.selection_anchor(target_index)
                self.preview_listbox.activate(target_index)
                self.preview_listbox.see(target_index)
                self._on_preview_select(None, defer_render=False)
            else:
                self.preview_listbox.activate(target_index)
        except Exception:
            pass

    x_root = int(getattr(event, "x_root", 0) or 0)
    y_root = int(getattr(event, "y_root", 0) or 0)
    return self._show_preview_list_context_menu(x_root, y_root)


def _on_preview_select(self, event, *, defer_render: bool = True):
    select_started = time.perf_counter()
    try:
        self._mark_preview_user_interaction(quiet_ms=1400)
    except Exception:
        pass
    if getattr(self, "_suppress_preview_reload_on_list_select", False):
        self._suppress_preview_reload_on_list_select = False
        _refresh_preview_group_selection_anchor_style(self)
        self._refresh_plate_auto_scope_modal_selection_state()
        return
    if _preview_auto_scope_modal_blocks_list(self):
        return "break"
    if getattr(self, "_plate_auto_scope_selection_mode_active", False):
        self._refresh_plate_auto_scope_modal_selection_state()
        return
    self._defer_preview_autosave_for_navigation(delay_ms=4500)
    sel = self.preview_listbox.curselection()
    _refresh_preview_group_selection_anchor_style(self)
    if not sel or not self.current_annotations:
        self._update_preview_toolbar_state(refresh_summary=False)
        self._refresh_plate_auto_scope_modal_selection_state()
        return

    try:
        active_display_index = int(self.preview_listbox.index(tk.ACTIVE))
    except Exception:
        active_display_index = int(sel[-1])
    if active_display_index not in sel:
        active_display_index = int(sel[-1])

    idx = self._get_preview_actual_index_from_display(active_display_index)
    if idx is None:
        self._update_preview_toolbar_state(refresh_summary=False)
        self._refresh_plate_auto_scope_modal_selection_state()
        return
    previous_idx = self.current_preview_index
    if previous_idx is not None and int(idx) == int(previous_idx):
        self._refresh_plate_auto_scope_modal_selection_state()
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass
        return
    self.current_preview_index = idx
    self._preview_session_restore_index = idx
    self._preview_session_restore_filename = str(getattr(self.current_annotations[idx], "filename", "") or "")
    if defer_render:
        self._schedule_preview_selection_render(
            idx,
            previous_idx,
            reset_view=True,
        )
        self._refresh_plate_auto_scope_modal_selection_state()
        return
    self._cancel_preview_selection_render()
    self._load_current_preview_selection(
        reset_view=True,
        selection_changed=(idx != previous_idx),
        refresh_summary=False,
    )
    self._schedule_preview_resume_persist(include_preview_approved=False)
    elapsed_ms = max(0.0, (time.perf_counter() - select_started) * 1000.0)
    if elapsed_ms >= 20.0:
        logger.debug(
            "[AnnotationTab][PERF] preview_select: "
            f"{elapsed_ms:.1f} ms | idx={idx} total={len(self.current_annotations or [])}"
        )
    try:
        self.preview_canvas.focus_set()
    except Exception:
        pass
    self._refresh_plate_auto_scope_modal_selection_state()
