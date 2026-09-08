#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 annotation startup helpers extracted from tab_annotation.py."""

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
from . import z2_run_lifecycle
from . import z2_manifest_runtime
from . import z2_status_ui_runtime
from . import z2_layout_ui_runtime
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

def _get_model_path(self, model_type: str) -> Path:
    if model_type == "vehicle":
        if self.vehicle_model_var.get() == "Custom": return Path(self.vehicle_custom_var.get())
        else: return CONFIG.get_base_models_dir("vehicle") / AVAILABLE_DETECT_MODELS[self.vehicle_model_var.get()]["file"]
    else:
        return Path(self.plate_custom_var.get())


def _collect_pending_model_downloads(self, mode_text: str):
    pending = []

    if self._mode_uses_vehicle(mode_text) and self.vehicle_model_var.get() != "Custom":
        model_key = (self.vehicle_model_var.get() or "").strip()
        model_info = AVAILABLE_DETECT_MODELS.get(model_key, {})
        target_path = self._get_model_path("vehicle")

        if model_info and target_path and not (
            Path(target_path).is_file() and Path(target_path).stat().st_size > 0
        ):
            pending.append(
                {
                    "role": "pojazdy",
                    "label": model_info.get("name", model_key or "model pojazdów"),
                    "asset_name": model_info.get("file", Path(target_path).name),
                    "target_path": Path(target_path),
                }
            )

    return pending


def _confirm_and_download_missing_models(self, pending_downloads) -> bool:
    pending_downloads = [item for item in pending_downloads
                         if not (Path(item["target_path"]).is_file()
                                 and Path(item["target_path"]).stat().st_size > 0)]
    if not pending_downloads:
        return True

    details = "\n".join(
        f"- {item['label']} -> {item['target_path']}"
        for item in pending_downloads
    )

    consent = messagebox.askyesno(
        "Pobieranie modelu z sieci",
        (
            "Brakuje lokalnych modeli potrzebnych do uruchomienia Z2.\n\n"
            f"{details}\n\n"
            "Aplikacja może pobrać te pliki z internetu dopiero po Twojej zgodzie.\n"
            "Źródło: oficjalne repozytorium github.com/ultralytics/assets.\n"
            "Okno pobierania pokaże postęp, rozmiar pliku i szybkość transferu.\n\n"
            "Czy chcesz pobrać brakujące modele teraz?"
        ),
        parent=self.frame.winfo_toplevel()
    )

    if not consent:
        logger.warning("Uruchomienie Z2 anulowane: użytkownik nie wyraził zgody na pobranie brakujących modeli.")
        self._set_status_label_state("Anulowano: brak zgody na pobranie modelu", "warning")
        return False

    from .model_download_dialog import download_models_with_splash

    self._set_status_label_state("Pobieranie modelu pojazdów", "info")
    downloaded = download_models_with_splash(self, pending_downloads)
    if not downloaded:
        self._set_status_label_state("Nie pobrano modelu. Anotacja nie została uruchomiona.", "warning")
    return downloaded


def _start_annotation(self, *args, **kwargs):
    return z2_workflow_methods._start_annotation(self, *args, **kwargs)


def _build_manual_annotations_template(
    self,
    in_dir: Path,
    progress_callback,
) -> tuple[list[ImageAnnotation], AnnotationReport]:
    image_paths = get_image_files(in_dir)
    annotations: list[ImageAnnotation] = []
    report = AnnotationReport()
    total = len(image_paths)
    previous_annotations_by_name: dict[str, ImageAnnotation] = {}

    if not self._is_free_mode_session_context():
        try:
            previous_bundle = self._get_campaign_previous_manual_source_bundle()
            previous_xml_path = previous_bundle.get("xml_path")
            if isinstance(previous_xml_path, Path) and previous_xml_path.exists():
                for previous_ann in self._parse_cvat_preview_annotations(previous_xml_path):
                    filename = str(getattr(previous_ann, "filename", "") or "").strip()
                    if filename:
                        previous_annotations_by_name[filename] = copy.deepcopy(previous_ann)
        except Exception as e:
            logger.debug(f"Nie udało się wczytać wcześniejszych ręcznych anotacji do nowego XML Z2: {e}")

    for idx, image_path in enumerate(image_paths, start=1):
        if not self.is_processing:
            raise KeyboardInterrupt("Anulowano")

        previous_ann = previous_annotations_by_name.get(image_path.name)
        if previous_ann is not None:
            ann = copy.deepcopy(previous_ann)
            ann.filename = image_path.name
            ann.status_message = "Przywrócono wcześniejszą ręczną anotację tablic dla tego obrazu."
        else:
            width, height = get_image_size(image_path)
            ann = ImageAnnotation(
                filename=image_path.name,
                width=max(1, int(width)),
                height=max(1, int(height)),
                detections=[],
                status=AnnotationStatus.NO_PLATE,
                status_message="Obraz przygotowany do recznej anotacji tablic.",
            )
        annotations.append(ann)
        report.add_result(ann)

        if callable(progress_callback):
            progress_callback(idx, total, image_path.name, idx)

    return annotations, report


def _build_manual_annotations_template_from_vehicle_seed(
    self,
    in_dir: Path,
    seed_annotations: list[ImageAnnotation] | None,
) -> tuple[list[ImageAnnotation], AnnotationReport]:
    image_paths = get_image_files(in_dir)
    annotations: list[ImageAnnotation] = []
    report = AnnotationReport()
    seed_map = {
        str(getattr(ann, "filename", "") or ""): ann
        for ann in (seed_annotations or [])
        if str(getattr(ann, "filename", "") or "").strip()
    }
    previous_annotations_by_name: dict[str, ImageAnnotation] = {}

    if not self._is_free_mode_session_context():
        try:
            previous_bundle = self._get_campaign_previous_manual_source_bundle()
            previous_xml_path = previous_bundle.get("xml_path")
            if isinstance(previous_xml_path, Path) and previous_xml_path.exists():
                for previous_ann in self._parse_cvat_preview_annotations(previous_xml_path):
                    filename = str(getattr(previous_ann, "filename", "") or "").strip()
                    if filename:
                        previous_annotations_by_name[filename] = copy.deepcopy(previous_ann)
        except Exception as e:
            logger.debug(f"Nie udało się wczytać wcześniejszych ręcznych anotacji do XML Z2 z boxami pojazdów: {e}")

    for image_path in image_paths:
        if not self.is_processing:
            raise KeyboardInterrupt("Anulowano")

        source_ann = seed_map.get(image_path.name)
        previous_ann = previous_annotations_by_name.get(image_path.name)
        if source_ann is not None:
            width = max(1, int(getattr(source_ann, "width", 0) or 0))
            height = max(1, int(getattr(source_ann, "height", 0) or 0))
        else:
            width, height = get_image_size(image_path)
            width = max(1, int(width))
            height = max(1, int(height))

        vehicle_detections = []
        if source_ann is not None:
            for det in getattr(source_ann, "detections", []) or []:
                if str(getattr(det, "label", "") or "").lower() != "vehicle":
                    continue
                vehicle_detections.append(
                    Detection(
                        label="vehicle",
                        confidence=float(getattr(det, "confidence", 1.0) or 1.0),
                        bbox=tuple(float(v) for v in (getattr(det, "bbox", ()) or (0.0, 0.0, 0.0, 0.0))[:4]),
                        attributes=dict(getattr(det, "attributes", {}) or {}),
                    )
                )

        if previous_ann is not None:
            ann = copy.deepcopy(previous_ann)
            ann.filename = image_path.name
            ann.width = width
            ann.height = height
            ann.detections = list(getattr(ann, "detections", []) or []) + vehicle_detections
            ann.status_message = "Przywrócono wcześniejszą ręczną anotację tablic i dodano pomocnicze boxy pojazdów."
            annotations.append(ann)
            report.add_result(ann)
            continue

        if vehicle_detections:
            status = AnnotationStatus.SUCCESS
            status_message = (
                f"Wykryto {len(vehicle_detections)} pojazdów. "
                "D dodaje ręczną tablicę, a boxy pojazdów pomagają zawęzić obszar pracy."
            )
        else:
            status = AnnotationStatus.NO_PLATE
            status_message = (
                "Nie wykryto pojazdu automatycznie. Nadal mozesz recznie dodac tablice polygonem 4 pkt."
            )

        ann = ImageAnnotation(
            filename=image_path.name,
            width=width,
            height=height,
            detections=vehicle_detections,
            status=status,
            status_message=status_message,
        )
        annotations.append(ann)
        report.add_result(ann)

    return annotations, report
