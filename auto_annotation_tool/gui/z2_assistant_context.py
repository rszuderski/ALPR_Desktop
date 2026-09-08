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
from .z2_model_dialogs import (
    _prompt_campaign_plate_auto_model_choice,
    _prompt_campaign_return_to_wizard_ok_modal,
    _confirm_campaign_plate_model_identity_choice,
    _prompt_z2_text_input,
    _extract_plate_model_metrics_from_rows,
    _build_auto_annotation_model_quality_rows,
)
from .z2_import_workflow import (
    _import_external_annotation_run_to_workspace,
)
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


def get_free_mode_assistant_context(self) -> dict:
    try:
        campaign_char_repair = bool(self._is_campaign_char_repair_return_mode())
        campaign_plate_repair = bool(self._is_campaign_plate_step4_repair_return_mode())
    except Exception:
        campaign_char_repair = False
        campaign_plate_repair = False

    if campaign_char_repair or campaign_plate_repair:
        return {
            "location": "[Z2] Tryb naprawczy kampanii",
            "goal": (
                "Wróciłeś do Z2, żeby poprawić albo powiększyć zbiór tablic przed dalszą pracą kampanii. "
                "Najważniejsze jest przygotowanie poprawnych ramek tablic i nadanie zdjęciom statusu [OK]."
            ),
            "current": "Z2 działa jako kontrola jakości tablic. Tylko pozycje [OK] zasilają pulę projektową i przyrost bieżącej iteracji.",
            "workflow": (
                "Jeśli masz dobry model tablic, uruchom autoanotację i potraktuj jej wynik jako punkt startowy.",
                "Jeśli dokładność jest ważniejsza niż czas, popraw ramki ręcznie na canvasie.",
                "Po utworzeniu lub korekcie ramek zaznacz poprawne zdjęcia na liście i nadaj im status [OK].",
                "Najwygodniej rysować i poprawiać ramki w pełnym ekranie, przełączanym klawiszem Enter.",
                "Po zakończeniu wróć do grafu, żeby kontynuować pracę bramki kampanii.",
            ),
            "glossary": (
                "Z2 = zakładka pracy nad tablicami",
                "tryb naprawczy = powrót do karty roboczej, żeby uzupełnić dane",
                "ramka = obrys tablicy na zdjęciu",
                "status [OK] = zdjęcie/anotacja zatwierdzone do dalszych etapów",
                "autoanotacja = model PT tworzy wstępne ramki",
                "PPM = prawy przycisk myszy na liście wyników",
                "LPM = lewy przycisk myszy",
                "Enter = przełączenie canvasa w pełny ekran",
            ),
            "caution": (
                "Nie zatwierdzaj zdjęć bez poprawnych ramek tablic. "
                "Tylko pozycje z [OK] zasilą wspólną pulę projektu."
            ),
            "references": ("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
        }

    try:
        free_mode_screen = str(self._coerce_free_mode_screen() or "").strip().lower()
        free_mode_route = str(self._get_workflow_route() or "").strip().lower()
        free_mode_context = bool(self._is_free_mode_session_context())
    except Exception:
        free_mode_screen = ""
        free_mode_route = ""
        free_mode_context = False

    if free_mode_context and free_mode_route == "auto" and free_mode_screen == "auto_summary":
        return {
            "location": "[Z2] Korekta po autoanotacji",
            "goal": (
                "Sprawdzasz wynik autoanotacji tablic i decydujesz, co dalej zrobić z bieżącym runem Z2."
            ),
            "current": "Autoanotacja jest propozycją. Materiał projektowy powstaje dopiero po ręcznej kontroli i statusie [OK].",
            "workflow": (
                "Po prawej możesz ręcznie poprawić poligony tablic i zatwierdzić poprawne pozycje statusem [OK].",
                "Po samej autoanotacji akcje „Wyodrębnij tablice” i eksport datasetu YOLO Pose pozostają zablokowane, jeśli nie ma pozycji [OK].",
                "Eksport samych anotacji XML jest wyjątkiem: wystarczy, że run ma zapisaną co najmniej jedną tablicę.",
                "„Ponowna autoanotacja” wraca do startu procesu, żeby uruchomić kolejny przebieg, także innym modelem.",
                "„Wyodrębnij tablice” prowadzi do Z3/PZ1 i przygotowuje tablice do dalszej pracy nad znakami.",
                "„Eksport” otwiera wybór: pakiet anotacji XML bez splitu albo dataset tablic YOLO Pose do treningu w Z4.",
                "Tylko dataset YOLO Pose i wyodrębnianie tablic biorą obrazy oznaczone statusem [OK]; niezatwierdzone pozycje są w nich pomijane.",
                "Jeśli wejdziesz w Z3/PZ1, run Z2 nadal zostaje zapisany i możesz wrócić do eksportu później z poziomu Z2.",
            ),
            "glossary": (
                "run = katalog pracy Z2 z anotacjami i metadanymi",
                "polygon = obrys tablicy zapisany w anotacji",
                "crop = wycięty fragment obrazu, tutaj sama tablica dla Z3",
                "YOLO Pose = typ datasetu/modelu, który uczy się obrysu tablicy",
                "status [OK] = pozycja zatwierdzona do dalszego wykorzystania",
            ),
            "caution": (
                "Wyodrębnianie tablic do Z3 i eksport datasetu YOLO Pose to ścieżki treningowe wymagające [OK]. "
                "Eksport anotacji XML służy przeniesieniu zapisanej pracy i nie wymaga statusu [OK]."
            ),
            "references": ("docs/mapa_funkcji_i_kodu.md",),
        }

    if free_mode_context and free_mode_route == "manual" and free_mode_screen == "manual_review":
        return {
            "location": "[Z2] Korekta po anotacji ręcznej",
            "goal": (
                "Sprawdzasz ręczny run anotacji tablic i wybierasz, czy ma zasilić pracę nad znakami, "
                "czy dataset tablic do treningu YOLO Pose."
            ),
            "current": "Ręcznie zapisany XML jest artefaktem roboczym. Do dalszych etapów treningowych trafiają pozycje zatwierdzone jako [OK].",
            "workflow": (
                "Po prawej możesz dalej poprawiać poligony tablic i zatwierdzać poprawne pozycje statusem [OK].",
                "Po samym utworzeniu XML wyodrębnianie tablic i dataset YOLO Pose pozostają zablokowane do czasu nadania statusu [OK].",
                "Eksport samych anotacji XML jest wyjątkiem: wystarczy co najmniej jedna zapisana ramka/poligon tablicy.",
                "„Wyodrębnij tablice” prowadzi do Z3/PZ1 i przygotowuje tablice do anotacji znaków.",
                "„Otwórz eksport” przenosi do kroku Eksport, gdzie wybierasz pakiet anotacji XML albo dataset tablic YOLO Pose do treningu w Z4.",
                "Dataset YOLO Pose bierze tylko obrazy oznaczone statusem [OK]; niezatwierdzone pozycje są pomijane.",
                "Run Z2 pozostaje zapisany, więc eksport możesz wykonać później z historii runów bez utraty pracy.",
            ),
            "glossary": (
                "run = katalog pracy Z2 z ręcznym XML-em i metadanymi",
                "polygon = obrys tablicy zapisany w anotacji",
                "crop = wycięty fragment obrazu, tutaj sama tablica dla Z3",
                "YOLO Pose = typ datasetu/modelu, który uczy się obrysu tablicy",
                "status [OK] = pozycja zatwierdzona do dalszego wykorzystania",
            ),
            "caution": (
                "Wyodrębnianie tablic do Z3 i eksport datasetu YOLO Pose wymagają [OK]. "
                "Eksport anotacji XML służy przeniesieniu zapisanej pracy i nie wymaga statusu [OK]."
            ),
            "references": ("docs/mapa_funkcji_i_kodu.md",),
        }

    return {
        "location": "[Z2] Anotacja tablic",
        "goal": "Ta zakładka służy do przygotowania i kontroli anotacji tablic: możesz utworzyć ręczny XML, uruchomić autoanotację albo poprawić istniejący run.",
        "current": "W kampanii Z2 jest czarną skrzynką kontroli: wejściem jest O/AT albo detekcja, wyjściem są tylko zatwierdzone pozycje [OK].",
        "workflow": (
            "Wybierz tor pracy: autoanotacja, anotacja ręczna albo powrót do istniejącego runu.",
            "Wskaż katalog zdjęć i przejdź dalej dopiero wtedy, gdy chcesz załadować pełny obszar roboczy Z2.",
            "W autoanotacji kliknij start, ustaw model tablic, opcjonalny model pojazdów i confidence w modalu.",
            "W ręcznej pracy utwórz XML, przejrzyj obrazy, dodaj lub popraw poligony i oznacz poprawne pozycje jako OK.",
            "Na końcu przejdź do eksportu: zapisz anotacje lub dataset YOLO, a potem zdecyduj czy wracasz do Z2, czy idziesz do Z4.",
        ),
        "glossary": (
            "run = katalog pracy Z2",
            "XML = anotacje tablic",
            "autoanotacja = model PT tworzy wstępne ramki",
            "O = zbiór obrazów",
            "AT = anotacje tablic",
            "status [OK] = pozycja zatwierdzona do dalszego użycia",
        ),
        "caution": "W autoanotacji najpierw wybierasz katalog obrazów, a model i progi ustawiasz dopiero w modalu startu.",
        "references": ("docs/mapa_funkcji_i_kodu.md",),
    }
