#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted Z2 workflow/state methods for AnnotationTab.

This module intentionally keeps methods as plain functions receiving ``self``.
The owning class delegates to them, which physically reduces tab_annotation.py
without changing the state model or the public method names used by callbacks.
"""

import copy
import csv
import datetime
import json
import logging
import math
import os
import queue
import re
import shutil
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from ..annotators.runtime_factory import (
    create_combined_plate_annotator,
    create_plate_annotator,
    create_vehicle_annotator,
)
from ..campaign_manager import CAMPAIGN
from ..config import AVAILABLE_DETECT_MODELS, CONFIG, SESSION, YOLO_AVAILABLE, logger
from ..data_models import AnnotationReport, AnnotationStatus, Detection, ImageAnnotation
from ..exporters import CVATExporter, ReportGenerator
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..rectification.polygon_validator import PolygonValidator
from ..training import DatasetCreator
from .dataset_display import build_dataset_display_ref
from .run_display import build_run_display_ref
from ..utils import cleanup_gpu_memory, count_images_in_directory, format_duration, get_image_files, get_image_size
from ..validators import format_yolo_model_identity, validate_model_file
from .canvas_progress_overlay import CanvasProgressOverlay
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
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
from .z2_flow_models import Z2CopyPayload, Z2LeftPanelCopyContext
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
from .z2_shared_ui import (
    apply_z2_workflow_cta_ui as dispatch_apply_z2_workflow_cta_ui,
    apply_z2_workflow_left_layout as dispatch_apply_z2_workflow_left_layout,
    build_z2_workflow_base_context as dispatch_build_z2_workflow_base_context,
    campaign_gate_id_for_edge,
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas
from .z2_annotation_process import (
    _refresh_step2_action_states,
    _start_annotation,
    _approve_annotation_stage,
    _process_thread,
    _finish,
    _switch_annotation_input_dir,
    _go_to_next_workflow_step,
)
from .z2_panel_workflow import (
    _refresh_free_mode_workflow_ui,
    get_campaign_step2_view_model,
    _refresh_manual_review_followup_ui,
    apply_theme,
    _finalize_successful_annotation_run_ui,
    _apply_workflow_step_widget_style,
    _render_compact_info_table,
    _refresh_free_mode_manual_right_panel,
    _refresh_manual_plate_stage_ui,
    _show_auto_annotation_success_dialog,
    _apply_z2_left_panel_copy_payload,
    _refresh_workflow_button_styles,
    _refresh_z2_miniflow_progress,
    _apply_main_pane_layout,
)
from .z2_preview_workflow import (
    _set_selected_preview_images_approved,
    _rename_selected_preview_image_file,
    _refresh_preview_workspace_visibility,
    _populate_preview_list_async,
    _refresh_preview_list_legend_theme,
    _refresh_preview_list_summary,
    _parse_cvat_preview_annotations,
    _open_preview_metric_filter_modal,
    _get_preview_image_file_metadata,
    _update_preview_edit_status,
    _save_preview_edits,
    _clear_selected_preview_auto_plates,
)
from .z2_campaign_runtime import (
    _build_campaign_char_effective_source,
    _get_campaign_auto_annotation_bootstrap,
    _collect_campaign_auto_annotation_sources,
    _schedule_deferred_campaign_route_cleanup,
    get_campaign_step2_source_state,
    _get_campaign_step3_preview_source_context,
    _build_campaign_plate_approved_entries_from_run,
    reset_campaign_iteration_route_state,
    _sync_campaign_iteration_artifact_registry,
    _build_campaign_plate_approved_preview_bundle,
    _build_campaign_plate_approved_export_source,
    _prepare_approved_step3_source_from_z2_run,
    _promote_campaign_char_repair_ok_to_approved_pool_before_return,
    _reset_campaign_runtime_state,
    _build_campaign_z2_gate_overlay_state,
    _apply_campaign_plate_auto_model_choice,
)
from .z2_restore_workflow import (
    _restore_preview_from_annotation_run,
    _apply_campaign_project_snapshot,
    _apply_annotation_run_restore_payload,
    _ensure_free_mode_input_workspace_preview,
    _restore_preview_from_session_run,
    _apply_free_mode_session_snapshot,
    _prepare_campaign_source_preview_payload,
    _apply_campaign_source_preview_payload,
    _prepare_annotation_run_restore_payload,
)
from .z2_export_workflow import (
    _start_plate_dataset_export,
    _prompt_z2_export_choice,
    _start_plate_annotation_package_export,
    _prompt_plate_annotation_package_export_options,
    _refresh_plate_dataset_export_sources,
)

NAV_BUTTON_WIDTH = 18
YOLO = None


def _prompt_campaign_plate_auto_model_choice(self, *, force_campaign: bool = False) -> bool:
    if self._is_free_mode_session_context() and not force_campaign:
        return True

    try:
        self._campaign_plate_auto_model_modal_start_requested = False
    except Exception:
        pass

    try:
        from ..campaign_manager import CAMPAIGN
        if not CAMPAIGN.get_active_project_name() and not force_campaign:
            return True
    except Exception:
        if not force_campaign:
            return True

    project_model_path = self._get_campaign_project_plate_model_path()
    palette = getattr(self.app, "palette", {})
    selected_picker_scope = {"value": "project"}
    selected_modal_model_path: dict[str, Path | None] = {"value": None}

    def _schedule_auto_cta_refresh_after_modal() -> None:
        try:
            self._campaign_plate_auto_model_modal_start_requested = False
        except Exception:
            pass
        try:
            self._set_plate_auto_scope_selection_mode(False)
        except Exception:
            pass
        try:
            self._set_plate_auto_scope_modal_ui_lock(False)
        except Exception:
            pass
        try:
            self._plate_auto_scope_modal_open = False
            self._plate_auto_scope_active_dialog = None
            self._plate_auto_scope_modal_refresh_callback = None
            self._plate_auto_scope_modal_selection_refresh_callback = None
        except Exception:
            pass

        def _refresh() -> None:
            try:
                self._refresh_step2_action_states(lightweight=False)
            except TypeError:
                try:
                    self._refresh_step2_action_states()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                self._refresh_free_mode_workflow_ui()
            except Exception:
                pass

        try:
            self.frame.after_idle(_refresh)
        except Exception:
            _refresh()

    def _prompt_picker_scope_for_model_file() -> str | None:
        choice: dict[str, str | None] = {"value": None}
        panel_bg = palette.get("panel", "#252526")
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")

        scope_dialog = tk.Toplevel(self.frame)
        try:
            scope_dialog.withdraw()
        except Exception:
            pass
        scope_dialog.title("Katalog startowy modelu")
        scope_dialog.transient(self.frame)
        try:
            scope_dialog.grab_set()
        except Exception:
            pass
        try:
            scope_dialog.configure(bg=panel_bg)
        except Exception:
            pass

        body = tk.Frame(scope_dialog, bg=panel_bg, padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            body,
            text="Od którego katalogu zacząć wybór pliku .pt?",
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            body,
            text=(
                "To tylko skrót do okna wyboru modelu. "
                "Sam wybór katalogu startowego nie zmienia aktywnego modelu."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=440,
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

        buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
        buttons.pack(fill=tk.X)

        def finish(value: str | None) -> None:
            choice["value"] = value
            try:
                scope_dialog.destroy()
            except Exception:
                pass

        ttk.Button(buttons, text="Modele projektu", command=lambda: finish("project")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Główny katalog modeli", command=lambda: finish("free")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Anuluj", command=lambda: finish(None)).pack(side=tk.RIGHT)

        try:
            self._fit_borderless_dialog(scope_dialog, parent=self.frame, min_width=520, min_height=190)
            scope_dialog.update_idletasks()
            scope_dialog.deiconify()
            scope_dialog.lift()
        except Exception:
            try:
                scope_dialog.update_idletasks()
                scope_dialog.deiconify()
                scope_dialog.lift()
            except Exception:
                pass
        scope_dialog.bind("<Escape>", lambda _event: finish(None))
        scope_dialog.wait_window()
        return choice.get("value")

    def _show_choice_modal() -> str:
        result = {"choice": ""}

        dialog = tk.Toplevel(self.frame)
        try:
            dialog.withdraw()
        except Exception:
            pass
        dialog_styler = getattr(self.app, "style_dialog_window", None)
        if callable(dialog_styler):
            try:
                dialog_styler(dialog, title="Wybór modelu tablic", geometry="760x460", parent=self.frame)
            except Exception:
                try:
                    dialog.title("Wybór modelu tablic")
                    dialog.resizable(False, False)
                except Exception:
                    pass
        else:
            try:
                dialog.title("Wybór modelu tablic")
                dialog.resizable(False, False)
            except Exception:
                pass

        panel_bg = palette.get("panel", "#252526")
        field_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        success = palette.get("success", "#2ecc71")
        warning = palette.get("warning", "#f39c12")

        try:
            dialog.configure(
                bg=panel_bg,
                highlightbackground=panel_bg,
                highlightcolor=panel_bg,
            )
        except Exception:
            pass

        body_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
        if callable(body_surface_builder):
            try:
                body_surface = body_surface_builder(dialog, tone="info")
            except Exception:
                body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
                body_surface.pack(fill=tk.BOTH, expand=True)
        else:
            body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            body_surface.pack(fill=tk.BOTH, expand=True)

        body = tk.Frame(body_surface, bg=panel_bg, bd=0, highlightthickness=0)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        title_lbl = tk.Label(
            body,
            text="Wybierz model tablic do autoanotacji Z2",
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
        )
        title_lbl.pack(anchor=tk.W, fill=tk.X)

        tk.Label(
            body,
            text=(
                "Głównym źródłem MT jest model projektu: zwykle pochodzi z wcześniejszego treningu w tej kampanii "
                "albo został zaimportowany w E1 jako zasób startowy. Jeśli nie masz jeszcze dobrego MT w projekcie, "
                "możesz świadomie wskazać sprawdzony model z głównego katalogu modeli. Model jednorazowy służy tylko "
                "do bieżącej autoanotacji i nie zmienia zasobów projektu."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        ).pack(anchor=tk.W, fill=tk.X, pady=(6, 14))

        runtime_meta = {}
        try:
            runtime_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
        except Exception:
            runtime_meta = {}
        runtime_model_path = None
        runtime_path_text = str(runtime_meta.get("path") or "").strip()
        if runtime_path_text:
            try:
                candidate_runtime_path = Path(runtime_path_text)
                if candidate_runtime_path.exists():
                    runtime_model_path = candidate_runtime_path
            except Exception:
                runtime_model_path = None

        def _show_models_details() -> None:
            project_path = project_model_path if project_model_path is not None and project_model_path.exists() else None
            if project_path is None and runtime_model_path is None:
                messagebox.showwarning(
                    "Brak modeli",
                    "Nie ma modelu, którego parametry można wyświetlić.",
                    parent=dialog,
                )
                return

            details = tk.Toplevel(dialog)
            try:
                details.withdraw()
            except Exception:
                pass
            title = "Szczegóły modeli autoanotacji"
            details_styler = getattr(self.app, "style_dialog_window", None)
            if callable(details_styler):
                try:
                    details_styler(details, title=title, geometry="760x560", parent=dialog)
                except Exception:
                    try:
                        details.title(title)
                        details.resizable(False, False)
                    except Exception:
                        pass
            else:
                try:
                    details.title(title)
                    details.resizable(False, False)
                except Exception:
                    pass
            try:
                details.transient(dialog)
                details.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
            except Exception:
                pass

            details_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
            if callable(details_surface_builder):
                try:
                    details_surface = details_surface_builder(details, tone="info")
                except Exception:
                    details_surface = tk.Frame(details, bg=panel_bg, bd=0, highlightthickness=0)
                    details_surface.pack(fill=tk.BOTH, expand=True)
            else:
                details_surface = tk.Frame(details, bg=panel_bg, bd=0, highlightthickness=0)
                details_surface.pack(fill=tk.BOTH, expand=True)

            details_body = tk.Frame(details_surface, bg=panel_bg, bd=0, highlightthickness=0)
            details_body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
            tk.Label(
                details_body,
                text=title,
                bg=panel_bg,
                fg=fg,
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                justify=tk.LEFT,
            ).pack(anchor=tk.W, fill=tk.X)
            tk.Label(
                details_body,
                text=(
                    "Tu porównujesz trwały model projektu z modelem aktywnym dla bieżącej autoanotacji Z2."
                ),
                bg=panel_bg,
                fg=muted,
                font=("Segoe UI", 9),
                anchor="w",
                justify=tk.LEFT,
                wraplength=690,
            ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

            rendered_model_detail_paths: set[str] = set()

            def _model_detail_key(model_path: Path | None) -> str:
                if model_path is None:
                    return ""
                try:
                    return str(Path(model_path).resolve()).casefold()
                except Exception:
                    return str(model_path).strip().casefold()

            def _add_details_block(label: str, model_path: Path | None, context_text: str) -> None:
                model_key = _model_detail_key(model_path)
                if model_key and model_key in rendered_model_detail_paths:
                    return
                if model_key:
                    rendered_model_detail_paths.add(model_key)
                block = tk.Frame(
                    details_body,
                    bg=field_bg,
                    bd=0,
                    highlightthickness=1,
                    highlightbackground=border,
                    highlightcolor=border,
                    padx=10,
                    pady=8,
                )
                block.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
                tk.Label(
                    block,
                    text=label,
                    bg=field_bg,
                    fg=fg,
                    font=("Segoe UI", 9, "bold"),
                    anchor="w",
                    justify=tk.LEFT,
                ).pack(anchor=tk.W, fill=tk.X)
                tk.Label(
                    block,
                    text=context_text,
                    bg=field_bg,
                    fg=muted,
                    font=("Segoe UI", 8),
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=670,
                ).pack(anchor=tk.W, fill=tk.X, pady=(3, 8))
                if model_path is None:
                    tk.Label(
                        block,
                        text="Brak modelu w tym miejscu.",
                        bg=field_bg,
                        fg=warning,
                        font=("Segoe UI", 8, "bold"),
                        anchor="w",
                        justify=tk.LEFT,
                    ).pack(anchor=tk.W, fill=tk.X)
                    return
                try:
                    quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(model_path)
                except Exception:
                    quality_rows, quality_tone = [], "warning"
                self._render_auto_annotation_model_quality_table(
                    block,
                    quality_rows,
                    quality_tone,
                    bg=panel_bg,
                    fg=fg,
                    muted=muted,
                    border=border,
                    success=success,
                    warning=warning,
                    wraplength=560,
                    label_width=17,
                    model_path=model_path,
                )

            _add_details_block(
                "Model projektu (MT)",
                project_path,
                "Trwały zasób kampanii. Jeśli jest aktywny, Z2 może użyć go bez ponownego wskazywania pliku.",
            )
            _add_details_block(
                "Aktywny model autoanotacji",
                runtime_model_path,
                "Model używany przez bieżące uruchomienie autoanotacji. Może być modelem projektu albo modelem jednorazowym.",
            )

            footer = tk.Frame(details_body, bg=panel_bg, bd=0, highlightthickness=0)
            footer.pack(fill=tk.X, pady=(2, 0), side=tk.BOTTOM)
            ttk.Button(footer, text="Zamknij", command=details.destroy).pack(side=tk.RIGHT)
            try:
                self._fit_borderless_dialog(details, parent=dialog, min_width=760, min_height=520)
                details.update_idletasks()
                details.deiconify()
                details.lift()
                details.focus_force()
            except Exception:
                pass
            try:
                details.grab_set()
            except Exception:
                pass
            details.bind("<Escape>", lambda _e: details.destroy())
            try:
                details.wait_window()
            finally:
                try:
                    if details.winfo_exists():
                        details.grab_release()
                except Exception:
                    pass

        status_card = tk.Frame(
            body,
            bg=field_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=12,
            pady=10,
        )
        status_card.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            status_card,
            text="Status modeli autoanotacji",
            bg=field_bg,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

        def _status_line(label_text: str, value_text: str, *, tone: str = "info") -> None:
            row = tk.Frame(status_card, bg=field_bg, bd=0, highlightthickness=0)
            row.pack(fill=tk.X, pady=(5, 0))
            tk.Label(
                row,
                text=label_text,
                bg=field_bg,
                fg=muted,
                font=("Segoe UI", 8),
                anchor="w",
                justify=tk.LEFT,
                width=28,
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=value_text,
                bg=field_bg,
                fg=(success if tone == "success" else warning if tone == "warning" else fg),
                font=("Segoe UI", 8, "bold" if tone in {"success", "warning"} else "normal"),
                anchor="w",
                justify=tk.LEFT,
                wraplength=410,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        if project_model_path is not None and project_model_path.exists():
            project_identity = self._get_model_identity_caption(project_model_path)
            project_status = f"jest: {project_model_path.name}"
            if project_identity:
                project_status += f" | {project_identity}"
            _status_line("Model projektu (MT)", project_status, tone="success")
        else:
            _status_line("Model projektu (MT)", "brak modelu zapisanego w projekcie", tone="warning")

        if runtime_model_path is not None:
            runtime_identity = str(runtime_meta.get("identity") or "").strip() or self._get_model_identity_caption(runtime_model_path)
            runtime_status = runtime_model_path.name
            if runtime_identity:
                runtime_status += f" | {runtime_identity}"
            runtime_scope = str(runtime_meta.get("scope") or "").strip().lower()
            runtime_source = str(runtime_meta.get("source") or "").strip().lower()
            if runtime_scope == "project" or runtime_source == "project":
                runtime_status += " | źródło: model projektu"
                runtime_tone = "success"
            else:
                runtime_status += " | źródło: bieżąca autoanotacja Z2"
                runtime_tone = "warning"
            _status_line("Aktywny model autoanotacji", runtime_status, tone=runtime_tone)
        else:
            _status_line("Aktywny model autoanotacji", "brak - wybierzesz go w tym modalu", tone="warning")

        details_row = tk.Frame(status_card, bg=field_bg, bd=0, highlightthickness=0)
        details_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            details_row,
            text="Szczegóły modeli",
            command=_show_models_details,
            state=(
                tk.NORMAL
                if (
                    (project_model_path is not None and project_model_path.exists())
                    or runtime_model_path is not None
                )
                else tk.DISABLED
            ),
        ).pack(side=tk.LEFT)

        options_host = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
        options_host.pack(fill=tk.BOTH, expand=True)

        def choose(mode: str, model_path: Path | None = None) -> None:
            result["choice"] = str(mode or "").strip()
            if model_path is not None:
                selected_modal_model_path["value"] = Path(model_path)
            try:
                dialog.destroy()
            except Exception:
                pass

        def add_option(title: str, description: str, button_text: str, mode: str, *, accent: bool = False) -> None:
            card = tk.Frame(
                options_host,
                bg=field_bg,
                bd=0,
                highlightthickness=1,
                highlightbackground=border,
                highlightcolor=border,
                padx=12,
                pady=10,
            )
            card.pack(fill=tk.X, pady=(0, 10))

            text_col = tk.Frame(card, bg=field_bg, bd=0, highlightthickness=0)
            text_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            tk.Label(
                text_col,
                text=title,
                bg=field_bg,
                fg=fg,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                justify=tk.LEFT,
            ).pack(anchor=tk.W, fill=tk.X)

            tk.Label(
                text_col,
                text=description,
                bg=field_bg,
                fg=muted,
                font=("Segoe UI", 9),
                anchor="w",
                justify=tk.LEFT,
                wraplength=390,
            ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

            ttk.Button(
                card,
                text=button_text,
                command=lambda selected_mode=mode: choose(selected_mode),
                style=("Accent.TButton" if accent else "TButton"),
            ).pack(side=tk.RIGHT, padx=(12, 0))

        if project_model_path is not None:
            project_identity = self._get_model_identity_caption(project_model_path)
            project_model_text = f"Aktywny model projektu: {project_model_path.name}"
            if project_identity:
                project_model_text += f" | {project_identity}"
            add_option(
                "Użyj modelu projektu (MT)",
                (
                    f"{project_model_text}\n"
                    "To normalna ścieżka pracy: Z2 użyje MT zapisanego w projekcie, a zasoby kampanii pozostaną spójne."
                ),
                "Użyj modelu projektu",
                "project",
                accent=True,
            )

        extension_note = tk.Frame(options_host, bg=panel_bg, bd=0, highlightthickness=0)
        extension_note.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            extension_note,
            text="Opcje rozszerzone / awaryjne",
            bg=panel_bg,
            fg=warning,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        tk.Label(
            extension_note,
            text=(
                "Użyj ich wtedy, gdy świadomie chcesz odejść od MT projektu: testujesz inny model "
                "albo podmieniasz MT na sprawdzony model wytrenowany poza tą kampanią."
            ),
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            wraplength=540,
        ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

        add_option(
            "Model jednorazowy dla bieżącej autoanotacji",
            (
                "Wskaż plik .pt tylko do tej operacji Z2. To dobry wybór testowy lub awaryjny: "
                "autoanotacja użyje wskazanego modelu, ale MT projektu pozostanie bez zmian."
            ),
            "Wskaż model jednorazowy",
            "custom_temp",
        )
        add_option(
            "Zmień model projektu (MT)",
            (
                "Użyj tej opcji tylko świadomie: wskazany model stanie się głównym MT projektu. "
                "Ma to sens, gdy model został już sensownie wytrenowany poza tą kampanią albo został przygotowany na innej maszynie."
            ),
            "Wskaż i ustaw jako MT",
            "custom_adopt",
        )

        buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
        buttons.pack(fill=tk.X, pady=(12, 0))
        start_model_path = runtime_model_path
        if start_model_path is None and project_model_path is not None:
            try:
                if Path(project_model_path).exists():
                    start_model_path = Path(project_model_path)
            except Exception:
                start_model_path = None
        ttk.Button(
            buttons,
            text="Uruchom autoanotację",
            command=(
                (lambda path=start_model_path: choose("runtime_start", path))
                if start_model_path is not None
                else (lambda: None)
            ),
            style="Accent.TButton",
            state=(tk.NORMAL if start_model_path is not None else tk.DISABLED),
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(buttons, text="Anuluj", command=lambda: choose("cancel")).pack(side=tk.RIGHT)

        try:
            self._fit_borderless_dialog(dialog, parent=self.frame, min_width=760, min_height=600)
            dialog.update_idletasks()
            dialog.deiconify()
            dialog.lift()
        except Exception:
            pass
        dialog.bind("<Escape>", lambda _e: choose("cancel"))
        dialog.wait_window()
        return str(result.get("choice") or "").strip().lower()

    while True:
        choice = _show_choice_modal()
        if not choice or choice == "cancel":
            _schedule_auto_cta_refresh_after_modal()
            return False

        start_requested = bool(choice in {"runtime_start", "project", "custom_temp", "custom_adopt"})

        if choice in {"runtime", "runtime_start"}:
            selected_model = selected_modal_model_path.get("value")
            if selected_model is None and project_model_path is not None:
                selected_model = project_model_path
            if selected_model is None:
                _schedule_auto_cta_refresh_after_modal()
                return False
            if start_requested:
                try:
                    effective_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
                    effective_path = str(effective_meta.get("path") or "").strip()
                    same_effective_model = bool(
                        effective_path
                        and Path(effective_path).exists()
                        and Path(effective_path).resolve() == Path(selected_model).resolve()
                    )
                except Exception:
                    same_effective_model = False
                if same_effective_model:
                    apply_result = True
                else:
                    apply_result = self._apply_campaign_plate_auto_model_choice(
                        Path(selected_model),
                        adopt_to_project=False,
                    )
            else:
                apply_result = self._apply_campaign_plate_auto_model_choice(
                    Path(selected_model),
                    adopt_to_project=False,
                )
        elif choice == "project":
            if project_model_path is None:
                _schedule_auto_cta_refresh_after_modal()
                return False
            selected_model = project_model_path
            apply_result = self._apply_campaign_plate_auto_model_choice(project_model_path, adopt_to_project=False)
        else:
            picker_scope = _prompt_picker_scope_for_model_file()
            if not picker_scope:
                continue
            selected_picker_scope["value"] = str(picker_scope or "project").strip().lower() or "project"
            selected_model = self._pick_campaign_plate_auto_model_file(
                preferred_path=project_model_path,
                picker_scope=str(selected_picker_scope.get("value") or "project"),
            )
            if selected_model is None:
                continue
            apply_result = self._apply_campaign_plate_auto_model_choice(
                selected_model,
                adopt_to_project=bool(choice == "custom_adopt"),
            )

        if apply_result == "retry":
            continue
        if apply_result:
            try:
                self._campaign_plate_auto_model_modal_start_requested = start_requested
            except Exception:
                pass
            if start_requested:
                confirmed_model_for_scope = selected_model
                try:
                    effective_meta = dict(self._get_effective_plate_model_runtime_meta() or {})
                    effective_path = str(effective_meta.get("path") or "").strip()
                    if effective_path and Path(effective_path).exists():
                        confirmed_model_for_scope = Path(effective_path)
                except Exception:
                    confirmed_model_for_scope = selected_model
                try:
                    self._campaign_plate_auto_model_confirmed_for_scope = str(Path(confirmed_model_for_scope).resolve())
                except Exception:
                    self._campaign_plate_auto_model_confirmed_for_scope = str(confirmed_model_for_scope or "")
            if not start_requested:
                continue
        final_result = bool(apply_result)
        if not final_result:
            _schedule_auto_cta_refresh_after_modal()
        return final_result

def _prompt_campaign_return_to_wizard_ok_modal(
    self,
    *,
    approved_images: int,
    approved_plates: int,
    images_with_plates: int,
    required_images: int = 0,
    required_plates: int = 0,
    total_plates: int = 0,
    xml_required: bool = False,
    xml_exists: bool = False,
) -> bool:
    result = {"choice": "stay"}
    try:
        graph_context = dict(getattr(self, "_campaign_graph_entry_context", {}) or {})
    except Exception:
        graph_context = {}
    graph_gate_id = campaign_gate_id_for_edge(
        graph_context.get("graph_edge_key"),
        graph_context.get("graph_gate_id"),
    )
    repair_origin_gate_id = campaign_gate_id_for_edge(
        graph_context.get("repair_origin_edge_key"),
        graph_context.get("repair_origin_gate_id")
        or graph_context.get("source_graph_gate_id"),
    )
    is_t06_context = bool(graph_gate_id == "T05")
    is_t07_repair_context = bool(graph_gate_id == "T04" and repair_origin_gate_id == "T06")
    if is_t06_context:
        dialog_title = "Przekazanie zdjęć [OK] do Z3"
        lead_title = "Przekaż zatwierdzone zdjęcia albo zostań w Z2"
        stay_label = "Zostań w Z2 i oznacz więcej [OK]"
        leave_label = (
            "Przekaż [OK] do Z3 i wróć do grafu"
            if int(approved_images or 0) > 0
            else "Wróć do grafu bez przekazania"
        )
    elif is_t07_repair_context:
        dialog_title = "Przekazanie zdjęć [OK] do puli YOLO"
        lead_title = "Przekaż zatwierdzone zdjęcia albo zostań w Z2"
        stay_label = "Zostań w Z2 i oznacz więcej [OK]"
        leave_label = (
            "Przekaż [OK] do puli YOLO i wróć do grafu"
            if int(approved_images or 0) > 0
            else "Wróć do grafu bez przekazania"
        )
    else:
        dialog_title = "Zatwierdź zdjęcia [OK] przed powrotem"
        lead_title = "Korekta ramek nie oznacza jeszcze statusu [OK]"
        stay_label = "Zostań i oznacz [OK]"
        leave_label = "Wróć do grafu"

    palette = getattr(self.app, "palette", {}) or {}
    dialog = tk.Toplevel(self.frame)
    try:
        dialog.withdraw()
    except Exception:
        pass

    dialog_styler = getattr(self.app, "style_dialog_window", None)
    if callable(dialog_styler):
        try:
            dialog_styler(
                dialog,
                title=dialog_title,
                geometry="680x340",
                parent=self.frame,
            )
        except Exception:
            try:
                dialog.title(dialog_title)
                dialog.resizable(False, False)
            except Exception:
                pass
    else:
        try:
            dialog.title(dialog_title)
            dialog.resizable(False, False)
        except Exception:
            pass

    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")

    try:
        dialog.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
    except Exception:
        pass

    body_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(body_surface_builder):
        try:
            body_surface = body_surface_builder(dialog, tone="warning")
        except Exception:
            body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            body_surface.pack(fill=tk.BOTH, expand=True)
    else:
        body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        body_surface.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(body_surface, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

    tk.Label(
        body,
        text=lead_title,
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)

    tk.Label(
        body,
        text=(
            (
                f"Czeka na przekazanie: {int(approved_images)} obrazów [OK] / "
                f"{int(approved_plates or 0)} tablic."
            )
            if (is_t06_context or is_t07_repair_context)
            else (
                f"Do przekazania: {int(approved_images)} obrazów [OK] / {int(approved_plates or 0)} tablic. "
                f"Warunek: {int(approved_plates or 0)}/{int(required_plates or 0)} tablic [OK]."
            )
        ),
        bg=panel_bg,
        fg=success,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    missing_images = max(0, int(required_images or 0) - int(approved_images or 0))
    missing_plate = max(0, int(required_plates or 0) - int(approved_plates or 0))
    missing_xml = bool(xml_required and not xml_exists)

    if is_t06_context:
        missing_plate = max(0, int(required_plates or 0) - int(approved_plates or 0))
        message_text = (
            "T06 używa Z2 jako pętli pomocniczej. Zdjęcia oznaczone [OK] nie trafią do źródła Z3, "
            "dopóki jawnie ich nie przekażesz.\n\n"
            + (
                f"Do minimalnego progu brakuje jeszcze {missing_plate} tablic [OK]. "
                if missing_plate > 0
                else "Minimalny próg tablic [OK] jest spełniony. "
            )
            + "Możesz zostać w Z2 i oznaczać dalej albo przekazać obecną pulę [OK] do Z3 i wrócić do grafu."
        )
    elif is_t07_repair_context:
        message_text = (
            "To tryb naprawczy T07. Zdjęcia oznaczone [OK] nie zasilą puli YOLO, "
            "dopóki jawnie ich nie przekażesz.\n\n"
            "Możesz zostać w Z2 i oznaczać dalej albo przekazać obecną pulę [OK] do projektu i wrócić do grafu."
        )
    elif missing_images > 0 or missing_plate > 0 or missing_xml:
        message_text = (
            "Ta bramka nie ma jeszcze minimalnej liczby anotacji potrzebnej do przekazania materiału.\n\n"
            f"Minimum: {int(required_plates or 0)} zatwierdzonych tablic na obrazach oznaczonych jako [OK]"
            + (" i utworzony plik XML anotacji." if xml_required else ".")
            + "\n"
            + (
                f"Brakuje: {missing_plate} tablic"
                if missing_images <= 0
                else f"Brakuje: {missing_images} obrazów [OK], {missing_plate} tablic"
            )
            + (" i pliku XML." if missing_xml else ".")
            + "\n\n"
            "Aby zatwierdzić pojedynczy obraz, kliknij go PPM i wybierz „Oznacz jako OK”. "
            "Możesz też zaznaczyć kilka obrazów i użyć opcji grupowej. "
            + ("Najpierw utwórz też plik anotacji tablic XML przyciskiem w lewym panelu. " if missing_xml else "")
        )
    else:
        message_text = (
            "Tylko obrazy oznaczone jako [OK] zasilą zatwierdzoną pulę projektu.\n\n"
            "Aby zatwierdzić pojedynczy obraz, kliknij go PPM i wybierz „Oznacz jako OK”.\n"
            "Możesz też zatwierdzić kilka obrazów naraz: zaznacz grupę na liście, kliknij PPM "
            "i wybierz „Oznacz zaznaczone jako OK”.\n\n"
            "Obrazy bez [OK] pozostaną w zestawie roboczym do dalszej korekty."
        )
    if is_t06_context:
        message_text = message_text.replace("T06", "T05")
    if is_t07_repair_context:
        message_text = message_text.replace("T07", "T06")
    tk.Label(
        body,
        text=message_text,
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    ).pack(anchor=tk.W, fill=tk.X, pady=(10, 0))

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def choose(mode: str) -> None:
        result["choice"] = str(mode or "").strip().lower()
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(
        buttons,
        text=leave_label,
        command=lambda: choose("leave"),
    ).pack(side=tk.RIGHT)

    ttk.Button(
        buttons,
        text=stay_label,
        command=lambda: choose("stay"),
        style="Accent.TButton",
    ).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.update_idletasks()
        self._fit_borderless_dialog(
            dialog,
            parent=getattr(self.app, "root", None) or self.frame,
            min_width=600,
            min_height=280,
        )
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

    dialog.bind("<Escape>", lambda _e: choose("stay"))
    dialog.wait_window()
    return str(result.get("choice") or "").strip().lower() == "leave"

def _confirm_campaign_plate_model_identity_choice(self, model_path: Path, info: dict | None, *, adopt_to_project: bool) -> bool:
    safe_path = Path(model_path)
    info = dict(info or {})
    source_architecture = str(info.get("source_architecture_label") or "").strip()
    source_model_name = str(info.get("source_model_name") or "").strip()
    source_display = source_architecture or source_model_name
    is_checkpoint_like = bool(
        safe_path.name.lower() == "best.pt"
        or safe_path.stem.lower().startswith("epoch")
    )

    scope_text = "po zapisaniu w projekcie" if adopt_to_project else "tylko dla bieżącej autoanotacji Z2"
    result = {"confirmed": False}
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f39c12")

    dialog = tk.Toplevel(self.frame)
    try:
        dialog.withdraw()
    except Exception:
        pass

    dialog_styler = getattr(self.app, "style_dialog_window", None)
    if callable(dialog_styler):
        try:
            dialog_styler(dialog, title="Potwierdź model tablic", geometry="760x560", parent=self.frame)
        except Exception:
            try:
                dialog.title("Potwierdź model tablic")
                dialog.resizable(False, False)
            except Exception:
                pass
    else:
        try:
            dialog.title("Potwierdź model tablic")
            dialog.resizable(False, False)
        except Exception:
            pass

    try:
        dialog.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
    except Exception:
        pass
    try:
        dialog.transient(getattr(self.app, "root", None) or self.frame.winfo_toplevel())
    except Exception:
        pass

    body_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(body_surface_builder):
        try:
            body_surface = body_surface_builder(dialog, tone="info")
        except Exception:
            body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            body_surface.pack(fill=tk.BOTH, expand=True)
    else:
        body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        body_surface.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(body_surface, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

    title_lbl = tk.Label(
        body,
        text="Potwierdź model tablic dla autoanotacji",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    title_lbl.pack(anchor=tk.W, fill=tk.X)

    intro_lines = [
        "Sprawdź parametry modelu przed uruchomieniem autoanotacji. To ten model będzie tworzył ramki tablic w bieżącej autoanotacji.",
        f"Zakres użycia: {scope_text}.",
    ]
    if is_checkpoint_like and source_display:
        intro_lines.append(f"Checkpoint wytrenowano z: {source_display}.")

    tk.Label(
        body,
        text="\n".join(intro_lines),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=650,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

    table_host = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    table_host.pack(anchor=tk.W, fill=tk.X, pady=(0, 12))
    quality_rows, quality_tone = self._build_auto_annotation_model_quality_rows(safe_path)
    self._render_auto_annotation_model_quality_table(
        table_host,
        quality_rows,
        quality_tone,
        bg=field_bg,
        fg=fg,
        muted=muted,
        border=border,
        success=success,
        warning=warning,
        wraplength=500,
        label_width=17,
        model_path=safe_path,
    )

    if str(quality_tone or "").strip().lower() != "success":
        tk.Label(
            body,
            text=(
                "Uwaga: metryki są niskie albo nie udało się ich odczytać. "
                "Możesz użyć tego modelu, ale wynik autoanotacji może wymagać większej korekty ręcznej."
            ),
            bg=panel_bg,
            fg=warning,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=650,
        ).pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(8, 0))

    def choose(value: bool) -> None:
        result["confirmed"] = bool(value)
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(buttons, text="Anuluj", command=lambda: choose(False)).pack(side=tk.RIGHT)
    ttk.Button(
        buttons,
        text="Użyj tego modelu",
        command=lambda: choose(True),
        style=("Accent.TButton" if str(quality_tone or "").strip().lower() == "success" else "TButton"),
    ).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        self._fit_borderless_dialog(dialog, parent=self.frame, min_width=760, min_height=520)
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass
    try:
        dialog.grab_set()
    except Exception:
        pass
    dialog.bind("<Escape>", lambda _e: choose(False))
    try:
        dialog.wait_window()
    finally:
        try:
            if dialog.winfo_exists():
                dialog.grab_release()
        except Exception:
            pass
    return bool(result.get("confirmed"))

def _prompt_z2_text_input(
    self,
    *,
    title: str,
    prompt: str,
    initial_value: str = "",
    width: int = 560,
) -> str | None:
    result = {"value": None}
    palette = getattr(self.app, "palette", {}) or {}

    dialog = tk.Toplevel(self.frame)
    try:
        dialog.withdraw()
    except Exception:
        pass

    dialog_styler = getattr(self.app, "style_dialog_window", None)
    if callable(dialog_styler):
        try:
            dialog_styler(
                dialog,
                title=str(title or "").strip(),
                geometry=f"{int(max(420, width))}x220",
                parent=self.frame,
            )
        except Exception:
            try:
                dialog.title(str(title or "").strip())
                dialog.resizable(False, False)
            except Exception:
                pass
    else:
        try:
            dialog.title(str(title or "").strip())
            dialog.resizable(False, False)
        except Exception:
            pass

    panel_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", palette.get("panel_alt", "#2d2d30"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")

    try:
        dialog.configure(bg=panel_bg, highlightbackground=panel_bg, highlightcolor=panel_bg)
    except Exception:
        pass

    body_surface_builder = getattr(self.app, "_build_themed_dialog_surface", None)
    if callable(body_surface_builder):
        try:
            body_surface = body_surface_builder(dialog, tone="info")
        except Exception:
            body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
            body_surface.pack(fill=tk.BOTH, expand=True)
    else:
        body_surface = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
        body_surface.pack(fill=tk.BOTH, expand=True)

    body = tk.Frame(body_surface, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

    tk.Label(
        body,
        text=str(prompt or "").strip(),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=max(360, int(width) - 100),
    ).pack(anchor=tk.W, fill=tk.X)

    entry = tk.Entry(
        body,
        bd=0,
        highlightthickness=1,
        relief=tk.FLAT,
        bg=field_bg,
        fg=fg,
        insertbackground=fg,
        highlightbackground=border,
        highlightcolor=border,
    )
    entry.pack(fill=tk.X, pady=(12, 0), ipady=6)
    try:
        entry.insert(0, str(initial_value or ""))
        entry.selection_range(0, tk.END)
    except Exception:
        pass

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def _close(value):
        result["value"] = value
        try:
            dialog.destroy()
        except Exception:
            pass

    ttk.Button(buttons, text="Anuluj", command=lambda: _close(None)).pack(side=tk.RIGHT)
    ttk.Button(buttons, text="OK", command=lambda: _close(entry.get())).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.bind("<Return>", lambda _e: _close(entry.get()))
        dialog.bind("<Escape>", lambda _e: _close(None))
    except Exception:
        pass

    try:
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        entry.focus_force()
    except Exception:
        pass

    dialog.wait_window()
    return result.get("value")

def _extract_plate_model_metrics_from_rows(self, rows: list[dict], *, source: str = "") -> dict:
    if not rows:
        return {}

    best_row: dict | None = None
    best_score = -1.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        strict_value = self._model_quality_value(
            row,
            (
                "map50_95",
                "box_map50_95",
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
                "map50_90",
                "box_map50_90",
                "metrics/mAP50-90(B)",
                "metrics/mAP50-90",
                "pose_map50_95",
                "metrics/mAP50-95(P)",
                "pose_map50_90",
                "metrics/mAP50-90(P)",
            ),
        )
        loose_value = self._model_quality_value(
            row,
            (
                "map50",
                "box_map50",
                "metrics/mAP50(B)",
                "metrics/mAP50",
                "pose_map50",
                "metrics/mAP50(P)",
            ),
        )
        score = strict_value if strict_value is not None else loose_value
        if score is None:
            continue
        if score >= best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return {}

    metrics = {
        "map50_95": self._model_quality_value(
            best_row,
            (
                "map50_95",
                "box_map50_95",
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
                "pose_map50_95",
                "metrics/mAP50-95(P)",
            ),
        ),
        "map50_90": self._model_quality_value(
            best_row,
            (
                "map50_90",
                "box_map50_90",
                "metrics/mAP50-90(B)",
                "metrics/mAP50-90",
                "pose_map50_90",
                "metrics/mAP50-90(P)",
            ),
        ),
        "map50": self._model_quality_value(
            best_row,
            (
                "map50",
                "box_map50",
                "metrics/mAP50(B)",
                "metrics/mAP50",
                "pose_map50",
                "metrics/mAP50(P)",
            ),
        ),
        "precision": self._model_quality_value(
            best_row,
            (
                "precision",
                "box_precision",
                "metrics/precision(B)",
                "pose_precision",
                "metrics/precision(P)",
            ),
        ),
        "recall": self._model_quality_value(
            best_row,
            (
                "recall",
                "box_recall",
                "metrics/recall(B)",
                "pose_recall",
                "metrics/recall(P)",
            ),
        ),
        "loss": self._model_quality_value(
            best_row,
            (
                "loss",
                "val/loss",
                "train/loss",
                "val/box_loss",
                "train/box_loss",
                "val/pose_loss",
                "train/pose_loss",
            ),
        ),
        "epoch": self._model_quality_value(best_row, ("epoch", "Epoch")),
        "source": source,
    }
    return {key: value for key, value in metrics.items() if value not in (None, "")}

def _build_auto_annotation_model_quality_rows(self, model_path: Path | str | None) -> tuple[list[tuple[str, str, str]], str]:
    path_text = str(model_path or "").strip()
    if not path_text:
        return [("Status", "nie wybrano pliku .pt", "warning")], "warning"

    try:
        safe_path = Path(path_text)
    except Exception:
        return [("Status", "nieprawidłowa ścieżka pliku .pt", "warning")], "warning"

    if not safe_path.exists():
        return [("Status", "plik .pt nie istnieje", "warning")], "warning"

    identity = self._get_model_identity_caption(safe_path)
    size_text = ""
    created_text = ""
    modified_text = ""
    try:
        model_stat = safe_path.stat()
        size_text = f"{model_stat.st_size / (1024 * 1024):.1f} MB"
        created_raw = getattr(model_stat, "st_birthtime", None)
        if created_raw is None:
            created_raw = getattr(model_stat, "st_ctime", None)
        if created_raw is not None:
            created_text = datetime.datetime.fromtimestamp(float(created_raw)).strftime("%Y-%m-%d %H:%M:%S")
        modified_text = datetime.datetime.fromtimestamp(float(model_stat.st_mtime)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        size_text = ""

    rows: list[tuple[str, str, str]] = [("Plik", safe_path.name, "info")]
    if identity:
        rows.append(("Typ", identity, "info"))
    if size_text:
        rows.append(("Rozmiar", size_text, "info"))
    if created_text:
        rows.append(("Utworzono plik", created_text, "info"))
    if modified_text:
        rows.append(("Modyfikacja pliku", modified_text, "info"))

    metrics = self._get_plate_model_quality_metrics(safe_path)
    if metrics:
        has_metric_numbers = any(
            self._model_quality_float(metrics.get(metric_key)) is not None
            for metric_key in ("map50_95", "map50_90", "map50", "precision", "recall", "loss")
        )
        strict_value = metrics.get("map50_95")
        strict_label = "mAP50-95"
        if strict_value is None and metrics.get("map50_90") is not None:
            strict_value = metrics.get("map50_90")
            strict_label = "mAP50-90"
        if strict_value is not None:
            rows.append((strict_label, self._format_model_quality_number(strict_value), self._auto_model_metric_tone(strict_value, "strict")))
        if metrics.get("map50") is not None:
            rows.append(("mAP50", self._format_model_quality_number(metrics.get("map50")), self._auto_model_metric_tone(metrics.get("map50"), "loose")))
        if metrics.get("precision") is not None:
            rows.append(("Precision", self._format_model_quality_number(metrics.get("precision")), self._auto_model_metric_tone(metrics.get("precision"), "standard")))
        if metrics.get("recall") is not None:
            rows.append(("Recall", self._format_model_quality_number(metrics.get("recall")), self._auto_model_metric_tone(metrics.get("recall"), "standard")))
        if metrics.get("loss") is not None:
            rows.append(("Loss", self._format_model_quality_number(metrics.get("loss")), "info"))
        if metrics.get("epoch") is not None:
            rows.append(("Najlepsza epoka", self._format_model_quality_epoch(metrics.get("epoch")), "info"))

        quality_basis = strict_value if strict_value is not None else metrics.get("map50")
        quality_note = self._describe_auto_model_quality(quality_basis)
        quality_numeric = self._model_quality_float(quality_basis)
        if quality_note:
            rows.append(
                (
                    "Ocena",
                    quality_note,
                    "success" if quality_numeric is not None and quality_numeric >= 0.60 else "warning",
                )
            )
        elif not has_metric_numbers:
            rows.append(
                (
                    "Wyniki treningu",
                    "znaleziono powiązaną sesję treningową, ale bez zapisanych wartości mAP/precision/recall",
                    "warning",
                )
            )

        table_tone = "success" if quality_numeric is not None and quality_numeric >= 0.60 else "warning"
        run_name = str(metrics.get("run_name") or "").strip()
        run_id = str(metrics.get("run_id") or "").strip()
        run_label = run_name or run_id
        if run_label:
            try:
                run_label = build_run_display_ref(
                    {"run_name": run_name, "run_id": run_id},
                    kind_hint="training",
                ).id
            except Exception:
                if run_name and run_id and run_id not in run_name:
                    run_label = f"{run_name} [{run_id}]"
        source = str(metrics.get("source") or "").strip()
        if run_label or source:
            source_line = "Źródło metryk"
            source_bits = []
            if source:
                source_bits.append(source)
            if run_label:
                source_bits.append(self._shorten_model_quality_text(run_label, 68))
            rows.append((source_line, " | ".join(source_bits), "info"))

        dataset_path = str(metrics.get("dataset_path") or "").strip()
        if dataset_path:
            try:
                dataset_display = self._format_workspace_relative_path(dataset_path)
            except Exception:
                dataset_display = dataset_path
            try:
                dataset_ref = build_dataset_display_ref(dataset_path)
                dataset_display = f"{dataset_ref.id} | {dataset_display}"
            except Exception:
                pass
            rows.append(("Dataset", self._shorten_model_quality_text(dataset_display, 86), "info"))
        return rows, (table_tone if has_metric_numbers else "warning")

    rows.append(
        (
            "Wyniki treningu",
            "brak metryk w historii treningu albo w results.csv",
            "warning",
        )
    )
    rows.append(
        (
            "Wskazówka",
            "przed wyborem sprawdź raport treningu, żeby nie startować autoanotacji w ciemno",
            "warning",
        )
    )
    return rows, "warning"
