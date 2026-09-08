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
    refresh_workflow_route_cards as dispatch_refresh_workflow_route_cards,
)
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .zoomable_canvas import ZoomableCanvas

NAV_BUTTON_WIDTH = 18
YOLO = None


def _start_plate_dataset_export(self):
    if not self._ensure_preview_edits_saved("eksport datasetu tablic"):
        return

    campaign_approved_source = {}
    if not self._is_free_mode_session_context():
        try:
            campaign_approved_source = self._build_campaign_plate_approved_export_source()
        except Exception as e:
            logger.debug(f"Nie udało się przygotować ApprovedSet tablic do eksportu: {e}")
            campaign_approved_source = {}

    run_dir_value = str(self.plate_dataset_run_var.get() or "").strip()
    images_dir_value = str(self.plate_dataset_images_var.get() or "").strip()

    export_source_kind = "z2_run_export"
    allowed_export_image_names: set[str] | None = None
    if campaign_approved_source:
        run_dir = Path(campaign_approved_source["run_dir"])
        images_dir = Path(campaign_approved_source["images_dir"])
        xml_path = Path(campaign_approved_source["xml_path"])
        export_source_kind = "campaign_approved_set"
    else:
        if not run_dir_value:
            return messagebox.showerror("Brak runu anotacji", "Wskaz folder runu anotacji Z2 zawierajacy annotations.xml.")
        if not images_dir_value:
            return messagebox.showerror("Brak obrazów", "Wskaż folder obrazów, na których powstał wybrany run anotacji.")

        run_dir = self._resolve_safe_annotation_run_dir(run_dir_value, require_xml=True)
        images_dir = Path(images_dir_value)

        if run_dir is None:
            return messagebox.showerror(
                "Błędny run anotacji",
                "Wybrany folder runu anotacji musi lezec w aktywnym workspace Z2 i zawierac annotations.xml.",
            )
        if not images_dir.exists() or not images_dir.is_dir():
            return messagebox.showerror("Brak obrazów", "Wybrany folder obrazów nie istnieje.")

        xml_path = run_dir / "annotations.xml"

    if export_source_kind == "z2_run_export":
        approval_state = self._get_run_plate_strict_approved_state(run_dir)
        min_dataset_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
        approved_dataset_plates = int(approval_state.get("approved_plates", 0) or 0)
        if (
            self._is_free_mode_session_context()
            and approved_dataset_plates < int(min_dataset_plates)
        ):
            self._warn_plate_dataset_export_requires_ok(approval_state)
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            return
        if not approval_state.get("ok"):
            self._warn_plate_dataset_export_requires_ok(approval_state)
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            return

        allowed_export_image_names = set(approval_state.get("approved_filenames") or set())
        if not allowed_export_image_names:
            self._warn_plate_dataset_export_requires_ok(approval_state)
            self._refresh_step2_action_states()
            self._refresh_free_mode_workflow_ui()
            return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = self._get_plate_dataset_base_dir() / f"Plates_Z2_{run_dir.name}_{timestamp}"
    self.plate_dataset_out_var.set(str(out_dir))
    self.plate_export_progress_var.set(0.0)
    self._dataset_export_completed = False
    self.export_plate_dataset_btn.configure(state=tk.DISABLED)
    self._set_plate_export_status("Rozpoczynam eksport datasetu YOLO Pose...", "success")
    self._set_preview_processing_overlay(
        True,
        title="Trwa eksport datasetu YOLO Pose",
        details="Lista i podgląd pozostają widoczne, ale są zablokowane do końca eksportu.",
        cancel_visible=False,
    )
    self._update_preview_processing_overlay_progress(
        pct=0.0,
        current=0,
        total=0,
        filename="",
        meta_text="0% | przygotowanie eksportu",
    )

    self._update_plate_dataset_ratio_labels()
    train = max(0.0, min(1.0, float(self.plate_train_pct.get()) / 100.0))
    val = max(0.0, min(1.0 - train, float(self.plate_val_pct.get()) / 100.0))
    ratios = {
        "train": train,
        "val": val,
        "test": max(0.0, 1.0 - train - val)
    }

    def worker():
        try:
            logger.info("=" * 50)
            logger.info("START EKSPORTU DATASETU TABLIC (Z2 -> YOLO Pose)")
            logger.info("=" * 50)
            logger.info(f"Run: {run_dir}")
            logger.info(f"Obrazy: {images_dir}")
            logger.info(f"Output: {out_dir}")
            if allowed_export_image_names is not None:
                logger.info(f"Eksport ograniczony do statusu OK: {len(allowed_export_image_names)} obrazów.")
            run_manifest = self._load_annotation_run_manifest(run_dir)
            manual_stage_enabled = bool(
                export_source_kind == "z2_run_export"
                and
                run_manifest.get("manual_xml_template", False)
                and not self._is_free_mode_session_context()
            )
            stage_result = {
                "enabled": manual_stage_enabled,
                "ok": False,
                "message": "",
                "pending_count": 0,
                "stage_images_total": 0,
                "stage_images_dir": "",
            }

            self.dataset_creator.annotations = []
            ok, msg, _ = self.dataset_creator.parse_cvat_xml(
                xml_path,
                allowed_image_names=allowed_export_image_names,
            )
            if not ok:
                def finish_xml_error(message=msg):
                    self._set_preview_processing_overlay(False)
                    messagebox.showerror("Błędny XML", message)
                    self._set_plate_export_status(message, "error")

                self._post_to_ui(finish_xml_error)
                return

            def prog(current, total, image_name):
                pct = (current / total) * 100 if total > 0 else 0
                def update_export_progress():
                    self.plate_export_progress_var.set(pct)
                    self._update_preview_processing_overlay_progress(
                        pct=pct,
                        current=current,
                        total=total,
                        filename=image_name,
                        meta_text=f"{int(round(pct))}% | eksport {current}/{total} obrazów",
                    )
                    self._set_plate_export_status(
                        f"Eksport datasetu: {current}/{total} obrazow... ({image_name})",
                        "success"
                    )

                self._post_to_ui(update_export_progress)

            ok, msg, _ = self.dataset_creator.create_dataset(images_dir, out_dir, ratios, prog)
            if not ok:
                def finish_export_error(message=msg):
                    self._set_preview_processing_overlay(False)
                    messagebox.showerror("Błąd eksportu", message)
                    self._set_plate_export_status(message, "error")

                self._post_to_ui(finish_export_error)
                return

            try:
                self._write_plate_dataset_source_manifest(
                    out_dir,
                    source_kind=export_source_kind,
                    source_run_dir=run_dir,
                    source_xml_path=xml_path,
                    source_images_dir=images_dir,
                )
            except Exception as e:
                logger.debug(f"Nie udalo sie zapisac manifestu zrodla datasetu tablic: {e}")

            if manual_stage_enabled:
                stage_source_image_paths = None
                try:
                    if self._get_campaign_iteration_manifest_image_count() > 0:
                        stage_source_image_paths = self._get_campaign_iteration_manifest_image_paths(images_dir)
                except Exception:
                    stage_source_image_paths = None
                stage_ok, stage_msg, stage_stats = self.dataset_creator.sync_pending_stage(
                    images_dir,
                    self._get_manual_plate_stage_dir(),
                    source_image_paths=stage_source_image_paths,
                )
                stage_result["ok"] = bool(stage_ok)
                stage_result["message"] = str(stage_msg or "").strip()
                if isinstance(stage_stats, dict):
                    stage_result["pending_count"] = int(stage_stats.get("pending_count", 0) or 0)
                    stage_result["stage_images_total"] = int(stage_stats.get("stage_images_total", 0) or 0)
                    stage_result["stage_images_dir"] = str(stage_stats.get("stage_images_dir") or "").strip()
                if stage_ok:
                    logger.info(stage_msg)
                else:
                    logger.warning(stage_msg)

            logger.info(f"[OK] Dataset YOLO Pose gotowy: {out_dir}")

            def finish_success():
                self.plate_export_progress_var.set(100.0)
                self._update_preview_processing_overlay_progress(
                    pct=100.0,
                    current=None,
                    total=None,
                    filename="",
                    meta_text="100% | eksport zakończony",
                )
                self._set_preview_processing_overlay(False)
                self._dataset_export_completed = True
                stage_note = ""
                stage_hint = ""
                if stage_result.get("enabled"):
                    pending_count = int(stage_result.get("pending_count", 0) or 0)
                    stage_total = int(stage_result.get("stage_images_total", pending_count) or pending_count)
                    stage_images_dir = str(stage_result.get("stage_images_dir") or "").strip()
                    stage_rel = self._format_workspace_relative_path(stage_images_dir) if stage_images_dir else ""
                    if stage_result.get("ok"):
                        if pending_count > 0:
                            stage_note = (
                                f" Do stage oczekujacych trafilo {pending_count} nieoznaczonych zdjec. "
                                f"Stage zawiera teraz {stage_total} obrazow: {stage_rel}."
                            )
                            stage_hint = (
                                f" Stage oczekujacych zawiera teraz {stage_total} obrazow. "
                                f"Do kolejnej rundy recznej anotacji wykorzystaj folder {stage_rel}."
                            )
                        else:
                            stage_note = " Wszystkie zdjecia z tego zestawu maja juz oznaczone tablice."
                            stage_hint = stage_note
                    else:
                        stage_note = f" Dataset powstal, ale stage oczekujacych nie zostal zaktualizowany: {stage_result.get('message')}"
                        stage_hint = stage_note
                next_step_text = (
                    f"Dataset gotowy: {self._format_workspace_relative_path(out_dir)}. "
                    "Możesz teraz przejść do [Z4], aby rozpocząć trening modelu tablic."
                )
                self._set_plate_export_status(
                    next_step_text,
                    "success"
                )
                if stage_note:
                    self._set_plate_export_status(next_step_text + stage_note, "success")
                self._refresh_manual_plate_stage_ui()
                training_tab = getattr(getattr(self, "app", None), "tabs", {}).get("training")
                dataset_preloaded = False
                if training_tab is not None and hasattr(training_tab, "dataset_var"):
                    try:
                        training_tab.dataset_var.set(str(out_dir))
                        if hasattr(training_tab, "_update_training_dataset_hint"):
                            training_tab._update_training_dataset_hint()
                        dataset_preloaded = True
                    except Exception:
                        pass
                if dataset_preloaded:
                    self._set_post_annotation_hint(
                        "Dataset tablic jest już gotowy i został podstawiony w [Z4]. Możesz przejść do treningu modelu.",
                        "success"
                    )
                if dataset_preloaded and stage_hint:
                    self._set_post_annotation_hint(
                        "Dataset tablic jest już gotowy i został podstawiony w [Z4]. Możesz przejść do treningu modelu."
                        + stage_hint,
                        "success"
                    )
                self._refresh_free_mode_workflow_ui()
                self._queue_free_mode_session_save()

                if self._is_free_mode_session_context():
                    choice = self._prompt_post_z2_export_completion_action(out_dir)
                    if choice == "training":
                        self._open_step4_training_from_z2_export(out_dir)
                    elif choice == "finish":
                        self._clear_free_mode_route_selection()
                    return

                messagebox.showinfo(
                    "Sukces",
                    (
                        "Dataset YOLO Pose został utworzony poprawnie.\n\n"
                        f"{out_dir}\n\n"
                        "Możesz teraz przejść do [Z4], aby rozpocząć trening modelu tablic."
                        + ("\nŚcieżka datasetu została już podstawiona w Z4." if dataset_preloaded else "")
                    )
                )

                if stage_result.get("enabled") and stage_result.get("ok") and str(stage_result.get("stage_images_dir") or "").strip():
                    pending_count = int(stage_result.get("pending_count", 0) or 0)
                    stage_total = int(stage_result.get("stage_images_total", pending_count) or pending_count)
                    messagebox.showinfo(
                        "Stage oczekujacych",
                        (
                            f"Do stage oczekujacych trafilo {pending_count} nieoznaczonych zdjec z tego zestawu.\n\n"
                            f"Stage zawiera teraz lacznie {stage_total} obrazow.\n"
                            f"{stage_result.get('stage_images_dir')}\n\n"
                            "Ten folder mozesz wykorzystac pozniej jako kolejna pule do recznej anotacji."
                        )
                    )
                elif stage_result.get("enabled") and not stage_result.get("ok") and str(stage_result.get("message") or "").strip():
                    messagebox.showwarning(
                        "Stage oczekujacych",
                        stage_result.get("message"),
                    )
                self._refresh_free_mode_workflow_ui()
                self._queue_free_mode_session_save()

            self._post_to_ui(finish_success)
        except Exception as e:
            logger.error(f"Blad eksportu datasetu tablic: {e}")
            def finish_critical_error(err=str(e)):
                self._set_preview_processing_overlay(False)
                messagebox.showerror("Krytyczny blad", err)
                self._set_plate_export_status(
                    f"Krytyczny blad eksportu: {err}",
                    "error"
                )

            self._post_to_ui(finish_critical_error)
        finally:
            self._post_to_ui(self._refresh_step2_action_states)

    threading.Thread(target=worker, daemon=True).start()

def _prompt_z2_export_choice(self, state: dict) -> str | None:
    result: dict = {}
    dataset_ready = bool(state.get("dataset_ready"))
    annotation_ready = bool(state.get("annotation_ready"))
    if not dataset_ready and not annotation_ready:
        messagebox.showwarning(
            "Eksport Z2",
            (
                "Nie ma jeszcze czego eksportować.\n\n"
                "Dataset YOLO wymaga spełnienia bramki zatwierdzonych tablic [OK]. "
                "Pakiet anotacji wymaga co najmniej jednej zapisanej tablicy w XML."
            ),
            parent=self.frame.winfo_toplevel(),
        )
        return None

    dialog = tk.Toplevel(self.frame)
    try:
        dialog.title("Eksport Z2")
        dialog.transient(getattr(self.app, "root", None) or self.frame.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)
    except Exception:
        pass

    palette = getattr(self.app, "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#4ec9b0")
    warning = palette.get("warning", "#f39c12")
    try:
        dialog.configure(bg=panel_bg)
    except Exception:
        pass

    body = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

    tk.Label(
        body,
        text="Wybierz rodzaj eksportu",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    tk.Label(
        body,
        text=(
            "Dataset YOLO Pose służy do treningu modelu tablic i korzysta ze splitu train/val/test. "
            "Pakiet anotacji XML służy do przenoszenia pracy między kopiami programu i nie używa splitu."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

    quality_info = dict(state.get("quality_info") or {})
    quality_label = str(quality_info.get("label", "SŁABY") or "SŁABY")
    quality_tone = str(quality_info.get("tone", "warning") or "warning").strip().lower()
    quality_color = success if quality_tone == "success" else warning
    stats_frame = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    stats_frame.pack(fill=tk.X, pady=(0, 12))
    stats_rows = (
        ("Wszystkie anotacje", f"{int(state.get('images_with_plates', 0) or 0)} obrazów / {int(state.get('total_plates', 0) or 0)} tablic", success),
        ("Zatwierdzone [OK]", f"{int(state.get('approved_images', 0) or 0)} obrazów / {int(state.get('approved_plates', 0) or 0)} tablic", success if int(state.get("approved_plates", 0) or 0) > 0 else warning),
        (
            "Bramka datasetu YOLO",
            (
                "otwarta"
                if dataset_ready
                else f"zamknięta, brakuje {int(state.get('dataset_missing_approved_plates', 0) or 0)} tablic [OK]"
            ),
            success if dataset_ready else warning,
        ),
        ("Jakość zbioru", quality_label, quality_color),
    )
    for row_idx, (label, value, color) in enumerate(stats_rows):
        tk.Label(
            stats_frame,
            text=label,
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=row_idx, column=0, sticky="w", pady=(0, 2))
        tk.Label(
            stats_frame,
            text=value,
            bg=panel_bg,
            fg=color,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).grid(row=row_idx, column=1, sticky="w", padx=(14, 0), pady=(0, 2))

    choice_var = tk.StringVar(value=("dataset" if dataset_ready else "annotations"))

    def _option(parent, value: str, title: str, description: str, enabled: bool):
        box = tk.Frame(parent, bg=panel_bg, bd=0, highlightthickness=1)
        border = success if enabled else palette.get("border", "#3a3a3a")
        try:
            box.configure(highlightbackground=border, highlightcolor=border)
        except Exception:
            pass
        box.pack(fill=tk.X, pady=(0, 8))
        radio = ttk.Radiobutton(
            box,
            text=title,
            variable=choice_var,
            value=value,
            state=(tk.NORMAL if enabled else tk.DISABLED),
        )
        radio.pack(anchor=tk.W, padx=8, pady=(7, 2))
        tk.Label(
            box,
            text=description,
            bg=panel_bg,
            fg=(muted if enabled else palette.get("disabled_fg", "#777777")),
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        ).pack(anchor=tk.W, fill=tk.X, padx=8, pady=(0, 7))
        return box

    _option(
        body,
        "dataset",
        "Eksport datasetu YOLO Pose ze splitem",
        (
            "Tworzy dataset do treningu modelu tablic. Eksportuje wyłącznie obrazy zatwierdzone [OK], "
            "a poniżej ustawiasz proporcje train/val/test."
            if dataset_ready
            else (
                "Niedostępne: dataset YOLO wymaga bramki [OK]. "
                f"Minimum to {int(state.get('dataset_min_approved_plates', 0) or 0)} zatwierdzonych tablic; "
                f"brakuje {int(state.get('dataset_missing_approved_plates', 0) or 0)}. "
                "Możesz nadal wyeksportować same anotacje XML."
            )
        ),
        dataset_ready,
    )

    split_box = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    split_box.pack(fill=tk.X, pady=(0, 8))
    train_var = tk.DoubleVar(value=float(self.plate_train_pct.get()))
    val_var = tk.DoubleVar(value=float(self.plate_val_pct.get()))
    train_lbl = tk.Label(split_box, bg=panel_bg, fg=fg, font=("Segoe UI", 8), width=9, anchor="w")
    val_lbl = tk.Label(split_box, bg=panel_bg, fg=fg, font=("Segoe UI", 8), width=9, anchor="w")
    test_lbl = tk.Label(split_box, bg=panel_bg, fg=fg, font=("Segoe UI", 8), width=9, anchor="w")

    def _sync_split_labels(_event=None):
        train = max(50.0, min(90.0, float(train_var.get())))
        max_val = max(0.0, 100.0 - train)
        val = max(0.0, min(max_val, float(val_var.get())))
        if abs(val - float(val_var.get())) > 0.01:
            try:
                val_var.set(val)
            except Exception:
                pass
        test = max(0.0, 100.0 - train - val)
        try:
            val_scale.configure(to=max_val)
        except Exception:
            pass
        train_lbl.configure(text=f"Train {train:.0f}%")
        val_lbl.configure(text=f"Val {val:.0f}%")
        test_lbl.configure(text=f"Test {test:.0f}%")

    ttk.Label(split_box, text="Split datasetu YOLO", style="Panel.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
    train_lbl.grid(row=1, column=0, sticky="w", pady=(0, 3))
    train_scale = ttk.Scale(split_box, from_=50, to=90, variable=train_var, command=_sync_split_labels)
    train_scale.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(0, 3))
    val_lbl.grid(row=2, column=0, sticky="w", pady=(0, 3))
    val_scale = ttk.Scale(split_box, from_=0, to=20, variable=val_var, command=_sync_split_labels)
    val_scale.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(0, 3))
    test_lbl.grid(row=3, column=0, sticky="w")
    split_box.columnconfigure(1, weight=1)
    _sync_split_labels()

    annotation_option_box = _option(
        body,
        "annotations",
        "Eksport anotacji XML do współpracy",
        (
            "Tworzy pakiet annotations.xml, opcjonalnie z obrazami. Ten wariant nie wykonuje splitu i nie wymaga bramki datasetu YOLO."
            if annotation_ready
            else "Niedostępne: w XML musi istnieć co najmniej jedna zapisana tablica."
        ),
        annotation_ready,
    )

    def _sync_export_choice_ui(*_args):
        show_split = bool(choice_var.get() == "dataset" and dataset_ready)
        try:
            if show_split:
                if str(split_box.winfo_manager()) != "pack":
                    split_box.pack(fill=tk.X, pady=(0, 8), before=annotation_option_box)
            else:
                split_box.pack_forget()
        except Exception:
            pass

    try:
        choice_var.trace_add("write", _sync_export_choice_ui)
    except Exception:
        pass
    _sync_export_choice_ui()

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(8, 0))

    def close_dialog():
        try:
            dialog.destroy()
        except Exception:
            pass

    def confirm_choice():
        choice = str(choice_var.get() or "").strip()
        if choice == "dataset" and not dataset_ready:
            return
        if choice == "annotations" and not annotation_ready:
            return
        if choice == "dataset":
            try:
                self.plate_train_pct.set(float(train_var.get()))
                self.plate_val_pct.set(float(val_var.get()))
                self._update_plate_dataset_ratio_labels()
            except Exception:
                pass
        result["choice"] = choice
        close_dialog()

    ttk.Button(buttons, text="Anuluj", command=close_dialog).pack(side=tk.RIGHT)
    ttk.Button(buttons, text="Dalej", command=confirm_choice).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.bind("<Escape>", lambda _e: close_dialog())
        self._fit_borderless_dialog(dialog, parent=self.frame, min_width=620, min_height=560)
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

    dialog.wait_window()
    return result.get("choice")

def _start_plate_annotation_package_export(self):
    if not self._is_free_mode_session_context():
        return
    if not self._ensure_preview_edits_saved("eksport anotacji tablic"):
        return

    state = self._get_plate_annotation_package_export_state()
    if not state.get("ok"):
        return messagebox.showwarning(
            "Eksport anotacji tablic",
            str(state.get("message") or "Najpierw przygotuj i zapisz anotacje tablic."),
            parent=self.frame.winfo_toplevel(),
        )

    run_dir = Path(state["run_dir"])
    options = self._prompt_plate_annotation_package_export_options(state)
    if not options:
        return
    include_images = bool(options.get("include_images", True))
    annotation_scope = str(options.get("annotation_scope") or "all_saved").strip() or "all_saved"
    approved_only = annotation_scope == "approved_ok_only"
    scope_label = "tylko zatwierdzone [OK]" if approved_only else "wszystkie zapisane"
    export_root = Path(options.get("export_root") or self._get_plate_annotation_exports_base_dir())
    try:
        annotations = [copy.deepcopy(ann) for ann in self._get_plate_annotation_export_annotations(run_dir)]
    except Exception as e:
        return messagebox.showerror(
            "Eksport anotacji tablic",
            f"Nie udało się odczytać anotacji z annotations.xml:\n{e}",
            parent=self.frame.winfo_toplevel(),
        )

    if approved_only:
        annotations = self._filter_plate_annotation_export_annotations_by_approved(run_dir, annotations)
        approved_export_images, approved_export_plates = self._count_plate_annotations(annotations)
        if approved_export_images <= 0 or approved_export_plates <= 0:
            return messagebox.showwarning(
                "Eksport anotacji tablic",
                (
                    "Wybrano eksport tylko obrazów zatwierdzonych [OK], ale w tym runie nie ma "
                    "zatwierdzonych obrazów z zapisaną anotacją tablicy.\n\n"
                    "Zatwierdź poprawne obrazy na liście wyników prawym przyciskiem myszy, "
                    "a potem uruchom eksport ponownie."
                ),
                parent=self.frame.winfo_toplevel(),
            )

    source_manifest = self._load_annotation_run_manifest(run_dir)
    compatible_images_dir = (
        self._resolve_existing_dir(str(self.plate_dataset_images_var.get() or "").strip())
        or self._resolve_step3_images_dir_from_z2_run(run_dir)
    )

    resolved_images: list[tuple[str, Path, Path]] = []
    if include_images:
        image_roots = self._get_external_run_image_roots(
            run_dir,
            source_manifest,
            compatible_images_dir=compatible_images_dir,
        )
        resolved_images, missing_images = self._resolve_external_run_source_images(annotations, image_roots)
        if missing_images:
            preview_missing = "\n".join(str(name) for name in missing_images[:6])
            extra_missing = max(0, len(missing_images) - 6)
            suffix = f"\n... i jeszcze {extra_missing} plików." if extra_missing else ""
            return messagebox.showerror(
                "Eksport anotacji tablic",
                (
                    "Nie można utworzyć kompletnego pakietu, bo brakuje obrazów powiązanych z annotations.xml.\n\n"
                    f"Brakujące pliki:\n{preview_missing}{suffix}"
                ),
                parent=self.frame.winfo_toplevel(),
            )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = export_root / f"PlateAnnotations_Z2_{run_dir.name}_{timestamp}"
    images_dir = package_dir / "images"
    xml_path = package_dir / "annotations.xml"
    manifest_path = package_dir / "annotation_export_manifest.json"
    readme_path = package_dir / "README.txt"

    try:
        package_dir.mkdir(parents=True, exist_ok=False)
        if include_images:
            images_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return messagebox.showerror(
            "Eksport anotacji tablic",
            f"Nie udało się utworzyć katalogu eksportu:\n{e}",
            parent=self.frame.winfo_toplevel(),
        )

    self.is_processing = True
    self._set_plate_annotation_export_button_state()
    try:
        self._set_preview_processing_overlay(
            True,
            title="Eksport anotacji tablic",
            details=(
                f"Tworzę pakiet annotations.xml ({scope_label}) z obrazami potrzebnymi do importu w E1."
                if include_images
                else f"Tworzę lekki pakiet annotations.xml ({scope_label}). Zdjęcia zostaną wskazane dopiero przy imporcie E1."
            ),
            cancel_visible=False,
        )
        self._update_preview_processing_overlay_progress(
            pct=0.0,
            current=0,
            total=max(1, len(resolved_images)) if include_images else 1,
            filename="",
            meta_text="0% | przygotowanie pakietu",
        )
        try:
            self.frame.update_idletasks()
        except Exception:
            pass

        relative_name_map: dict[str, str] = {}
        used_export_paths: set[str] = set()
        if include_images:
            total_images = max(1, len(resolved_images))
            for idx, (filename, _raw_path, source_image_path) in enumerate(resolved_images, start=1):
                relative_path = self._build_safe_imported_image_relative_path(
                    filename,
                    index=idx - 1,
                    used_paths=used_export_paths,
                )
                relative_name = str(relative_path).replace("\\", "/")
                target_image_path = images_dir / relative_path
                target_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image_path, target_image_path)
                relative_name_map[str(filename)] = relative_name

                if idx == len(resolved_images) or idx % 25 == 0:
                    pct = (idx / total_images) * 100.0
                    self._update_preview_processing_overlay_progress(
                        pct=pct,
                        current=idx,
                        total=len(resolved_images),
                        filename=Path(relative_name).name,
                        meta_text=f"{int(round(pct))}% | kopiowanie obrazów {idx}/{len(resolved_images)}",
                    )
                    try:
                        self.frame.update_idletasks()
                    except Exception:
                        pass

            for ann in annotations:
                original_name = str(getattr(ann, "filename", "") or "").strip()
                mapped_name = relative_name_map.get(original_name)
                if mapped_name:
                    ann.filename = mapped_name

        if not CVATExporter(task_name="Z2 Plate Annotation Export").export(
            annotations,
            xml_path,
            include_confidence=True,
            only_successful=False,
        ):
            raise RuntimeError("Eksporter CVAT nie zapisał pliku annotations.xml.")

        images_with_plates, total_plates = self._count_plate_annotations(annotations)
        strict_approval_state = self._get_run_plate_strict_approved_state(run_dir)
        approved_images = int(strict_approval_state.get("approved_images", 0) or 0)
        approved_plates = int(strict_approval_state.get("approved_plates", 0) or 0)
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        manifest_payload = {
            "export_type": "z2_plate_annotations_package",
            "created_at": now_iso,
            "source_run_dir": str(run_dir.resolve()),
            "source_xml_path": str((run_dir / "annotations.xml").resolve()),
            "source_images_dir": str(Path(compatible_images_dir).resolve()) if compatible_images_dir else "",
            "package_dir": str(package_dir.resolve()),
            "annotations_xml": str(xml_path.resolve()),
            "images_included": bool(include_images),
            "images_dir": str(images_dir.resolve()) if include_images else "",
            "annotation_scope": annotation_scope,
            "annotation_scope_label": scope_label,
            "exported_images": int(images_with_plates),
            "exported_plates": int(total_plates),
            "approved_images_in_source_run": int(approved_images),
            "approved_plates_in_source_run": int(approved_plates),
            "import_hint": "W E1 wybierz import anotacji tablic i wskaż plik annotations.xml z tego katalogu.",
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        readme_path.write_text(
            (
                "Pakiet eksportu anotacji tablic Z2\n"
                "=================================\n\n"
                "Ten katalog służy do przeniesienia anotacji tablic na inną maszynę lub do innej kopii programu.\n"
                "W E1 użyj importu anotacji tablic i wskaż plik annotations.xml znajdujący się w tym katalogu.\n\n"
                f"Zakres anotacji: {scope_label}\n"
                f"Obrazy dołączone do pakietu: {'tak' if include_images else 'nie'}\n"
                f"Obrazy z anotacjami: {images_with_plates}\n"
                f"Tablice: {total_plates}\n"
                f"Status OK w runie źródłowym: {approved_images} obrazów / {approved_plates} tablic\n"
            ),
            encoding="utf-8",
        )

        self._update_preview_processing_overlay_progress(
            pct=100.0,
            current=(len(resolved_images) if include_images else 1),
            total=(len(resolved_images) if include_images else 1),
            filename="annotations.xml",
            meta_text="100% | pakiet gotowy",
        )
        self._set_plate_export_status(
            (
                f"Pakiet anotacji tablic gotowy ({scope_label}): {images_with_plates} obrazów / {total_plates} tablic. "
                "W E1 na drugiej maszynie wskaż plik annotations.xml z tego katalogu."
            ),
            "success",
        )
        message = (
            "Utworzono pakiet anotacji tablic do importu w E1.\n\n"
            f"Zakres anotacji: {scope_label}\n"
            f"Obrazy dołączone do pakietu: {'tak' if include_images else 'nie'}\n"
            f"Obrazy z anotacjami: {images_with_plates}\n"
            f"Tablice: {total_plates}\n"
            f"Status OK w runie źródłowym: {approved_images} obrazów / {approved_plates} tablic\n\n"
            f"Plik do importu: {xml_path}\n"
            f"Katalog pakietu: {package_dir}\n\n"
            "Na drugiej maszynie otwórz E1, wybierz import anotacji tablic i wskaż ten plik annotations.xml. "
            "Jeśli pakiet nie zawiera obrazów, import poprosi o zgodny katalog zdjęć."
        )
        themed_info = getattr(getattr(self, "app", None), "themed_info", None)
        if callable(themed_info):
            themed_info("Eksport anotacji tablic", message, parent=self.frame, tone="success")
        else:
            messagebox.showinfo("Eksport anotacji tablic", message, parent=self.frame.winfo_toplevel())
    except Exception as e:
        try:
            shutil.rmtree(package_dir)
        except Exception:
            pass
        messagebox.showerror(
            "Eksport anotacji tablic",
            f"Nie udało się utworzyć pakietu anotacji:\n{e}",
            parent=self.frame.winfo_toplevel(),
        )
    finally:
        self.is_processing = False
        try:
            self._set_preview_processing_overlay(False)
        except Exception:
            pass
        self._set_plate_annotation_export_button_state()
        self._refresh_step2_action_states()

def _prompt_plate_annotation_package_export_options(self, state: dict) -> dict | None:
    default_root = self._get_plate_annotation_exports_base_dir()
    result: dict = {}

    dialog = tk.Toplevel(self.frame)
    try:
        dialog.title("Eksport anotacji tablic")
        dialog.transient(getattr(self.app, "root", None) or self.frame.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)
    except Exception:
        pass

    palette = getattr(self.app, "palette", {}) or {}
    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#4ec9b0")
    try:
        dialog.configure(bg=panel_bg)
    except Exception:
        pass

    body = tk.Frame(dialog, bg=panel_bg, bd=0, highlightthickness=0)
    body.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

    images_with_plates = int(state.get("images_with_plates", 0) or 0)
    total_plates = int(state.get("total_plates", 0) or 0)
    approved_images = 0
    approved_plates = 0
    try:
        run_dir = Path(state.get("run_dir")) if state.get("run_dir") else None
        strict_state = self._get_run_plate_strict_approved_state(run_dir)
        approved_images = int(strict_state.get("approved_images", 0) or 0)
        approved_plates = int(strict_state.get("approved_plates", 0) or 0)
    except Exception:
        approved_images = 0
        approved_plates = 0
    tk.Label(
        body,
        text="Pakiet anotacji tablic do współpracy",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X)
    tk.Label(
        body,
        text=(
            "Program utworzy katalog z annotations.xml. Obrazy możesz dołączyć do pakietu albo pominąć, "
            "jeśli odbiorca ma już ten sam katalog zdjęć."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=500,
    ).pack(anchor=tk.W, fill=tk.X, pady=(6, 12))

    stats = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    stats.pack(fill=tk.X, pady=(0, 10))
    for row, (label, value) in enumerate(
        (
            ("Obrazy z tablicami", str(images_with_plates)),
            ("Liczba tablic", str(total_plates)),
            ("Zatwierdzone [OK]", f"{approved_images} obrazów / {approved_plates} tablic"),
        )
    ):
        tk.Label(
            stats,
            text=label,
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=row, column=0, sticky="w", pady=(0, 2))
        tk.Label(
            stats,
            text=value,
            bg=panel_bg,
            fg=success,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).grid(row=row, column=1, sticky="w", padx=(14, 0), pady=(0, 2))

    path_var = tk.StringVar(value=str(default_root))
    include_images_var = tk.BooleanVar(value=True)
    export_scope_var = tk.StringVar(value="all_saved")

    scope_frame = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    scope_frame.pack(fill=tk.X, pady=(0, 10))
    tk.Label(
        scope_frame,
        text="Zakres anotacji",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(anchor=tk.W)
    ttk.Radiobutton(
        scope_frame,
        text="Wszystkie zapisane anotacje",
        variable=export_scope_var,
        value="all_saved",
    ).pack(anchor=tk.W, pady=(4, 0))
    ttk.Radiobutton(
        scope_frame,
        text="Tylko obrazy zatwierdzone [OK]",
        variable=export_scope_var,
        value="approved_ok_only",
    ).pack(anchor=tk.W, pady=(2, 0))
    tk.Label(
        scope_frame,
        text=(
            "Tryb [OK] pomija obrazy bez zatwierdzenia na liście wyników. "
            "To dobry wybór, gdy eksport ma być filtrem jakości."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=500,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    path_frame = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    path_frame.pack(fill=tk.X, pady=(0, 8))
    tk.Label(
        path_frame,
        text="Katalog eksportu",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(anchor=tk.W)
    path_row = tk.Frame(path_frame, bg=panel_bg, bd=0, highlightthickness=0)
    path_row.pack(fill=tk.X, pady=(4, 0))
    path_entry = ttk.Entry(path_row, textvariable=path_var, width=56)
    path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def browse_export_dir():
        initialdir = str(path_var.get() or default_root)
        selected = filedialog.askdirectory(
            parent=dialog,
            initialdir=initialdir,
            title="Wybierz katalog, w którym utworzyć pakiet anotacji",
        )
        if selected:
            path_var.set(selected)

    ttk.Button(path_row, text="Wskaż", command=browse_export_dir).pack(side=tk.LEFT, padx=(8, 0))

    ttk.Checkbutton(
        body,
        text="Dołącz obrazy do pakietu",
        variable=include_images_var,
    ).pack(anchor=tk.W, pady=(8, 0))
    tk.Label(
        body,
        text=(
            "Jeśli odznaczysz tę opcję, eksport będzie lekki i przeniesie tylko XML. "
            "Import E1 poprosi wtedy o wskazanie zgodnego katalogu zdjęć."
        ),
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=500,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

    buttons = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def close_dialog():
        try:
            dialog.destroy()
        except Exception:
            pass

    def confirm_export():
        raw_dir = str(path_var.get() or "").strip()
        if not raw_dir:
            messagebox.showwarning(
                "Eksport anotacji tablic",
                "Wskaż katalog eksportu.",
                parent=dialog,
            )
            return
        try:
            export_root = Path(raw_dir).expanduser()
        except Exception:
            messagebox.showwarning(
                "Eksport anotacji tablic",
                "Podana ścieżka katalogu eksportu jest nieprawidłowa.",
                parent=dialog,
            )
            return
        result["export_root"] = export_root
        result["include_images"] = bool(include_images_var.get())
        result["annotation_scope"] = str(export_scope_var.get() or "all_saved").strip() or "all_saved"
        close_dialog()

    ttk.Button(buttons, text="Anuluj", command=close_dialog).pack(side=tk.RIGHT)
    ttk.Button(buttons, text="Eksportuj", command=confirm_export).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        dialog.bind("<Escape>", lambda _e: close_dialog())
        self._fit_borderless_dialog(dialog, parent=self.frame, min_width=560, min_height=430)
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

    dialog.wait_window()
    if not result:
        return None
    return result

def _refresh_plate_dataset_export_sources(self):
    run_dir = None
    run_value = str(self.plate_dataset_run_var.get() or "").strip()
    input_value = str(self.input_dir_var.get() or "").strip()
    current_input_dir = Path(input_value) if input_value else None
    if run_value:
        run_dir = self._resolve_safe_annotation_run_dir(run_value)

    project_active = False
    if run_dir is None:
        try:
            from ..campaign_manager import CAMPAIGN

            project_active = bool(CAMPAIGN.get_active_project_name()) and not self._is_free_mode_session_context()
            if project_active:
                candidate = self._resolve_safe_annotation_run_dir(CAMPAIGN.get_step2_staging_run())
                if candidate is not None and self._annotation_run_matches_input(candidate, current_input_dir):
                    run_dir = candidate

                if run_dir is None:
                    search_roots = []
                    for root_candidate in (CAMPAIGN.get_staging_dir("auto_ann"), CAMPAIGN.get_dir("auto_ann")):
                        if root_candidate is not None:
                            search_roots.append(Path(root_candidate))

                    run_dir = self._find_latest_annotation_run_for_input(current_input_dir, search_roots)
        except Exception:
            project_active = False

    if run_dir is None and getattr(self, "last_staging_run_dir", None) and not project_active:
        candidate = self._resolve_safe_annotation_run_dir(self.last_staging_run_dir)
        if candidate is not None and (
            current_input_dir is None or self._annotation_run_matches_input(candidate, current_input_dir)
        ):
            run_dir = candidate

    if run_dir is None and current_input_dir is None and not self._is_free_mode_session_context():
        run_dir = self._find_latest_annotation_run_dir()

    if run_dir is not None:
        self._load_plate_dataset_context_from_run(run_dir, force_images_update=project_active)
        xml_path = run_dir / "annotations.xml"
        images_text = str(self.plate_dataset_images_var.get() or "").strip()
        if xml_path.exists():
            if images_text and Path(images_text).exists():
                approval_state = self._get_run_plate_strict_approved_state(run_dir)
                approved_images = int(approval_state.get("approved_images", 0) or 0)
                approved_plates = int(approval_state.get("approved_plates", 0) or 0)
                total_images = int(approval_state.get("total_images", 0) or 0)
                total_plates = int(approval_state.get("total_plates", 0) or 0)
                if approval_state.get("ok"):
                    skipped_images = max(0, total_images - approved_images)
                    if self._is_free_mode_session_context():
                        min_dataset_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
                        missing_dataset_plates = max(0, min_dataset_plates - approved_plates)
                        if missing_dataset_plates > 0:
                            self._set_plate_export_status(
                                (
                                    "Eksport anotacji XML jest dostępny bez bramki datasetu.\n\n"
                                    f"Dataset YOLO Pose ze splitem wymaga jeszcze {missing_dataset_plates} tablic [OK]. "
                                    f"Teraz: {approved_images} obrazów OK / {approved_plates} tablic OK."
                                ),
                                "warning",
                            )
                        else:
                            self._set_plate_export_status(
                                (
                                    f"Gotowe: dataset YOLO może objąć {approved_images} obrazów OK / {approved_plates} tablic OK. "
                                    f"Pakiet anotacji XML jest także dostępny. Pozycje bez OK zostaną pominięte w datasecie ({skipped_images} obrazów)."
                                ),
                                "success",
                            )
                    else:
                        self._set_plate_export_status(
                            (
                                f"Gotowe do eksportu: {approved_images} obrazów OK / {approved_plates} tablic OK. "
                                f"Pozycje bez OK zostaną pominięte ({skipped_images} obrazów)."
                            ),
                            "success"
                        )
                else:
                    if self._is_free_mode_session_context() and total_plates > 0:
                        self._set_plate_export_status(
                            (
                                "Eksport anotacji XML jest dostępny, bo XML zawiera zapisane tablice.\n\n"
                                "Dataset YOLO Pose ze splitem jest zablokowany do czasu nadania statusu [OK] "
                                "odpowiedniej liczbie obrazów."
                            ),
                            "warning",
                        )
                    else:
                        self._set_plate_export_status(
                            (
                                "Eksport zablokowany: żadna anotacja nie ma statusu OK.\n\n"
                                "Jak odblokować: zaznacz poprawne obrazy na liście, kliknij PPM i wybierz "
                                "„Oznacz zaznaczone jako OK”. Eksport obejmie wyłącznie pozycje zatwierdzone OK.\n\n"
                                f"OK: {approved_images} obrazów / {approved_plates} tablic; "
                                f"w runie: {total_images} obrazów / {total_plates} tablic."
                            ),
                            "warning"
                        )
            else:
                self._set_plate_export_status(
                    "Wybrano run anotacji Z2, ale trzeba jeszcze wskazac folder zrodlowych obrazow dla tego runu anotacji.",
                    "warning"
                )
        else:
            self._set_plate_export_status(
                "Wybrany folder runu anotacji nie zawiera pliku annotations.xml.",
                "error"
            )
        self._refresh_manual_plate_stage_ui()
        self._refresh_step2_action_states()
        return

    if current_input_dir is not None:
        self.plate_dataset_out_var.set(self._plate_dataset_output_preview())
        self._set_plate_export_status(
            f"Dla aktualnego folderu obrazów nie ma jeszcze runu anotacji Z2. Kliknij {self._get_step2_start_action_reference()}, aby utworzyć nowy run anotacji dla tej puli albo wskaż run anotacji ręcznie.",
            "muted"
        )
        self._refresh_manual_plate_stage_ui()
        self._refresh_step2_action_states()
        return

    self.plate_dataset_out_var.set(self._plate_dataset_output_preview())
    self._set_plate_export_status(
        f"Brak runu anotacji Z2. Najpierw kliknij {self._get_step2_start_action_reference()}, aby utworzyć nowy run autoanotacji Z2, albo wskaż istniejący folder runu anotacji ręcznie.",
        "muted"
    )
    self._refresh_manual_plate_stage_ui()
    self._refresh_step2_action_states()
