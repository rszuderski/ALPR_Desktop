#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka ZNAKÓW.
Logicznie podzielona na:
1. Wycinanie tablic (surowe)
2. Wykrywanie znaków i Analiza (OCR/YOLO, Laboratorium Filtrów, Turniej z Cache)
3. Integracje i Dataset YOLO
"""

import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import threading
import xml.etree.ElementTree as ET
import json
import cv2
import shutil
import copy
import math
import time
from .z3_metadata_cache import read_preview_metadata, mark_preview_metadata_changed
import textwrap
from types import SimpleNamespace

from ..config import CONFIG, logger, get_yolo_class
from ..campaign_manager import CAMPAIGN
from ..icons import IconManager
from ..character_recognition import PlateGenerator, CharacterDetector, CharacterDetection, DetectionMethod
from ..character_recognition.reading_order import (
    annotate_records_reading_order,
    infer_plate_layout_from_records,
    sort_records_reading_order,
)
from ..ocr import PlateOCR
from ..data_models import ImageAnnotation, Detection
from ..training.dataset_splitter import DatasetSplitter
from ..utils import cleanup_gpu_memory
from ..validators import validate_yolo_dataset
from .help_manager import HELP
from .guided_action_card import GuidedActionCard
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .canvas_progress_overlay import CanvasProgressOverlay
from .z3_slim_progress_bar import SlimProgressBar
from . import z3_init_runtime
from . import z3_navigation_runtime
from . import z3_preview_typing_runtime
from . import z3_preview_metadata_runtime
from . import z3_plate_layout_runtime
from . import z3_character_geometry
from . import z3_preview_editor_runtime
from . import z3_preview_overlay_runtime
from .free_mode_assistant import get_step3_free_mode_assistant_context
from .z3_campaign_flow import (
    auto_progress_campaign_step3_entry,
    clear_step3_campaign_context,
    back_to_substep_1_campaign,
    back_to_substep_2_campaign,
    build_step3_pz3_path_selection_view_model,
    build_step3_pz3_status_panel_view_model,
    enter_campaign_step3_mode,
    finalize_step3_from_existing_outputs,
    get_step3_finish_block_message,
    go_to_substep_2_campaign,
    go_to_substep_3_campaign,
    has_any_step3_export_outputs,
    _mark_t06_pz2_contract,
    mark_step3_work_interrupted_on_app_close,
    open_campaign_step3_entry,
    persist_step3_progress,
    refresh_campaign_step3_navigation_visibility,
    resolve_step3_campaign_action_command,
    restore_campaign_step3_mode,
    return_step3_result_to_wizard,
    return_to_t05_work_after_step3_pz1,
    return_to_wizard_from_step3_pz2,
    return_to_wizard_for_step3_rework,
    set_step3_finish_hint,
    unlock_dataset_subtab,
    unlock_detection_subtab,
)
from .z3_free_mode_flow import (
    back_to_substep_1_free_mode,
    back_to_substep_2_free_mode,
    ensure_step3_free_mode_context,
    go_to_substep_2_free_mode,
    go_to_substep_3_free_mode,
    reset_extract_source_inputs,
    handle_pz3_dataset_source_var_write,
    reset_step3_subtab_flow,
    set_pz3_dataset_source_mode,
    set_pz3_selected_path,
)
from .z3_preview_events import (
    on_preview_canvas_drag,
    on_preview_canvas_keypress,
    on_preview_canvas_keyrelease,
    on_preview_canvas_leave,
    on_preview_canvas_motion,
    on_preview_canvas_mousewheel,
    on_preview_canvas_press,
    on_preview_canvas_release,
    on_preview_list_mouse_primary,
    finalize_preview_char_add_state,
    on_preview_canvas_secondary_press,
    start_preview_character_box_drag,
)
from .z3_preview_ui import (
    apply_preview_sort_bar_style,
    apply_preview_fullscreen_chrome,
    apply_preview_canvas_cursor,
    apply_preview_focus_prompt_style,
    apply_preview_info_stats_style,
    apply_preview_source_actions_style,
    apply_preview_surface_style,
    apply_preview_typing_overlay_style,
    build_preview_legend_sections,
    build_plates_list_legend_counts,
    apply_plate_listbox_row_style,
    characters_to_text,
    characters_to_display_rows,
    characters_to_display_text,
    clamp_preview_image_position,
    clamp_preview_controls_legend_offsets,
    get_preview_image_bottom_reserved,
    draw_preview_badge_stack,
    draw_preview_canvas_info_overlay,
    draw_preview_fixed_text_badge,
    draw_preview_source_legend,
    draw_preview_text_badge,
    estimate_preview_badge_layout,
    estimate_preview_source_legend_height,
    estimate_preview_source_legend_width,
    ensure_detection_preview_loaded,
    fit_preview_text_to_width,
    format_plate_listbox_label,
    format_preview_layout_summary,
    format_preview_record_source_label,
    format_preview_source_counts_line,
    get_preview_badge_component_style,
    focus_preview_canvas,
    get_preview_badge_layout_metrics,
    get_preview_legend_theme,
    get_preview_status_presentation,
    get_preview_box_records,
    get_preview_box_variants,
    get_plate_listbox_source_flags,
    get_plate_listbox_ordinal,
    get_plate_row_foreground,
    get_preview_repair_progress_snapshot,
    get_preview_mode_overlay_default_position,
    get_preview_overlay_dock_theme,
    get_preview_layout_filter_key,
    get_preview_sort_mode_key,
    get_preview_sort_priority,
    get_preview_sort_source_count,
    get_preview_source_badge_layers,
    get_preview_source_component_legend_items,
    get_preview_source_visual_style,
    get_preview_step3_gate_overlay_state,
    get_sorted_preview_plate_ids,
    measure_preview_badge_stack,
    measure_preview_overlay_font_height,
    measure_preview_overlay_text_width,
    measure_preview_text_badge,
    on_preview_select,
    load_preview_data,
    normalize_preview_layout_filter_key,
    normalize_preview_sort_mode_key,
    on_preview_controls_legend_drag,
    open_detection_subtab_with_preview,
    on_preview_controls_legend_leave,
    on_preview_controls_legend_motion,
    on_preview_controls_legend_mousewheel,
    on_preview_canvas_enter,
    on_preview_controls_legend_press,
    on_preview_controls_legend_release,
    place_preview_record_overlay,
    place_preview_overlay_dock,
    place_preview_hint_overlay,
    plan_preview_canvas_info_overlay_layout,
    plate_matches_preview_layout_filter,
    persist_active_preview_characters,
    preview_record_has_symbol,
    refresh_preview_controls_legend,
    refresh_preview_import_focus_ui,
    refresh_preview_layout_override_ui_light,
    refresh_preview_live_metadata_ui,
    refresh_preview_typing_overlay_visibility,
    rebuild_preview_listbox,
    reset_preview_cache,
    reset_preview_view_state,
    log_preview_edit_flow,
    log_preview_latency,
    schedule_preview_latency_paint,
    start_preview_latency_probe,
    draw_preview_fast_render_details,
    update_preview_character_selection_items_fast,
    redraw_preview_character_overlay_only,
    redraw_preview_character_overlays_light,
    draw_preview_plate_status_frame,
    draw_preview_layout_separator,
    update_preview_layout_separator_visual,
    update_preview_character_drag_visual,
    redraw_preview_add_box_overlay_only,
    render_preview_overlay_dock,
    set_preview_fullscreen,
    set_plates_legend_info,
    set_preview_layout_filter_hover,
    select_preview_character_box,
    set_preview_counts_info,
    set_preview_fusion_info,
    set_preview_info,
    set_preview_layout_summary_info,
    set_preview_box_info,
    set_preview_sort_hover,
    set_preview_processing_overlay,
    set_preview_repair_progress_status,
    serialize_character_records,
    sync_preview_intro_wraplength,
    sync_preview_edit_status_visibility,
    set_preview_import_focus,
    toggle_preview_controls_legend,
    toggle_preview_import_focus,
    update_preview_info_label,
    update_preview_box_info_label,
    update_preview_processing_overlay_progress,
    update_preview_controls_legend_cursor,
    update_preview_edit_status,
    update_preview_toolbar_state,
    update_preview_repair_progress_ui,
)
from .z3_preview_records import (
    adapt_reextract_seed_payload_to_current_preview,
    build_character_source_tags,
    build_preview_plate_reextract_match_key,
    build_plate_layout_export_metadata,
    capture_preview_reextract_seed_metadata,
    char_record_to_symbol_and_x,
    character_record_collides_with_manual,
    character_record_overlap_score,
    clone_base_record_with_candidate_bbox,
    collect_historical_preview_reextract_seed_metadata,
    fusion_details_yolo_box_backend_positions,
    get_character_source_tag,
    merge_reextract_seed_metadata_into_preview,
    merge_detected_characters_preserving_manual,
    normalize_character_source_kind,
    normalize_preview_char_bbox,
    normalize_preview_source_path_key,
    normalize_plate_source_bucket,
    normalize_plate_source_origin,
    preview_plate_has_renderable_boxes,
    scale_preview_char_records_to_new_size,
)
from .z3_session_state import (
    clear_project_bound_session_values,
    force_save_all,
    load_local_session,
    prune_legacy_yolo_arch_session_keys,
    save_local_setting,
)
from .z3_shared_ui import (
    build_step3_local_success_message,
    build_step3_pz3_dataset_mode_view_model,
    build_lazy_subtab_placeholder,
    clear_lazy_subtab_placeholder,
    create_step3_widgets,
    ensure_dataset_tab_built,
    ensure_detect_tab_built,
    ensure_step3_subtab_built,
    get_subtab_state,
    hide_pz3_status_summary_section,
    log_message,
    paint_extract_entry_cards_first,
    refresh_pz3_cards_ui,
    refresh_pz3_cvat_card,
    refresh_pz3_dataset_card,
    refresh_pz3_dataset_mode_ui,
    refresh_pz3_dataset_source_card,
    refresh_pz3_status_panel_ui,
    refresh_step3_mode_specific_ui,
    remember_pz3_operation_message,
    set_button_state,
    set_console_text,
    set_subtab_state,
    show_pz3_operation_summary_modal,
    sync_step3_nav_buttons,
)
from .z3_theme_ui import (
    apply_character_annotation_theme,
    apply_gold_export_filter_check_style,
    apply_gold_export_source_check_style,
    apply_gold_export_split_check_style,
    apply_plates_legend_style,
    apply_preview_mode_overlay_style,
    apply_preview_mode_radio_style,
    apply_yolo_option_check_style,
    draw_selection_indicator,
    get_inline_status_widget_snapshot,
    get_readable_text_color,
    mark_inline_status_contrast_boost,
    panel_style_name,
    redraw_preview_mode_eye_icon,
    refresh_preview_mode_overlay_visibility_with_labels,
    refresh_selection_row,
    set_selection_row_hover,
    set_inline_status_label_state,
    set_themed_label_state,
)
from .z3_extraction_tab_ui import (
    append_extract_cut_plan,
    bind_source_path_watchers,
    build_continue_extract_summary_rows,
    build_extraction_tab,
    format_extract_cut_plan,
    format_extract_source_datetime,
    format_extract_source_folder_name,
    format_extract_start_path,
    hide_campaign_detect_splash,
    pick_images_dir,
    pick_xml_file,
    refresh_continue_source_summary,
    refresh_extract_action_state,
    refresh_extract_continue_action_visibility,
    refresh_extract_start_card_style,
    refresh_extract_start_summary,
    refresh_extract_workflow_ui,
    run_extraction,
    schedule_source_binding_refresh,
    set_extraction_status,
    set_extract_start_summary_rows,
    set_source_binding_status,
    set_widget_state,
    show_campaign_extract_below_minimum_modal,
    show_campaign_extract_failure_modal,
    show_campaign_detect_splash,
    show_extract_completed_modal,
    on_source_path_var_changed,
    update_step3_source_path_lock,
)
from .z3_extraction_sources import (
    backfill_preview_expected_texts_from_sources,
    count_xml_plate_cut_targets,
    current_extract_source_payload,
    derive_candidate_root_for_match,
    evaluate_images_dir_for_xml,
    extract_source_plate_tokens_from_filename,
    extract_source_manifest_path,
    find_latest_extract_preview_run_dir,
    find_matching_images_dir_for_xml,
    get_extract_preview_manifest_state,
    get_extract_preview_ready_count,
    get_preferred_source_roots,
    is_extract_preview_ready,
    is_extract_preview_ready_fast,
    is_path_within,
    is_recommended_images_dir,
    normalize_xml_image_relpath,
    plate_cut_reading_order_key,
    preview_matches_current_extract_source,
    prepare_plate_cut_detections_for_source,
    read_xml_image_names,
    refresh_source_binding_status,
    summarize_missing_xml_images,
    write_extract_source_manifest,
    xml_relpath_to_path,
)
from .z3_export_summary import (
    build_step3_export_summary,
    inspect_yolo_dataset_label_objects,
    read_step3_export_summary,
    write_step3_export_summary,
)
from .z3_scroll_ui import (
    bind_scroll_canvas_children,
    canvas_overflows,
    mousewheel_units,
    on_cvat_export_global_mousewheel,
    on_detect_right_global_mousewheel,
    on_plates_listbox_mousewheel,
    redirect_child_mousewheel_to_canvas,
    redirect_pz3_child_mousewheel_to_canvas,
    refresh_detect_right_adaptive_wraps,
    restore_scroll_canvas_focus,
    schedule_detect_right_adaptive_wrap_refresh,
    suppress_selection_hover_during_scroll,
    sync_cvat_export_canvas_width,
    sync_cvat_export_scrollregion,
    sync_detect_right_canvas_width,
    sync_detect_right_scrollregion,
    sync_extract_left_canvas_width,
    sync_extract_left_scrollregion,
    widget_contains_point,
)
from .z3_split_utils import allocate_split_counts, build_split_entries
from .z3_detection_tab_ui import (
    apply_detection_advanced_section_style,
    apply_global_yolo_device_choice,
    apply_detection_param_row_style,
    apply_detection_pipeline_blocks,
    auto_device_label,
    bind_detect_mode_card,
    build_detection_tab,
    build_pz2_detection_guard_counts,
    append_detection_pipeline_builder_block,
    apply_final_truth_count_guard,
    clear_detection_pipeline_builder,
    close_detection_pipeline_builder,
    compile_detection_pipeline_blocks,
    compose_detection_method_status,
    confirm_last_detection_result,
    commit_detection_pipeline_builder,
    format_last_detection_summary_line,
    format_plate_last_detection_line,
    best_text_distance,
    device_to_ocr,
    device_to_ultralytics,
    draw_detection_pipeline_builder_canvas,
    ensure_yolo_model_checkpoint,
    get_detection_active_model_status,
    get_detection_method_key,
    get_detection_pipeline_block_meta,
    get_detection_pipeline_block_style,
    get_detection_pipeline_builder_blocks,
    get_detection_pipeline_blocks,
    get_detection_pipeline_short_label,
    get_saved_detection_pipeline_blocks,
    get_detection_method_status_label,
    get_detection_workflow_text,
    get_hybrid_detection_status_text,
    get_hybrid_rescue_max_chars,
    get_yolo_rescue_enabled,
    get_yolo_runtime_settings,
    get_available_devices,
    get_campaign_detection_yolo_model_path,
    get_campaign_char_model_path,
    get_effective_detection_device_choice,
    get_effective_yolo_model_path,
    get_detection_advanced_toggle_label,
    infer_yolo_arch_from_model_path,
    load_last_detection_summary,
    normalize_selected_device,
    normalize_detection_method_key,
    handle_detect_mode_selection,
    has_configured_yolo_detection_model,
    move_detection_pipeline_builder_selected_block,
    open_detection_pipeline_advanced_modal,
    open_detection_pipeline_builder,
    pick_detection_pipeline_yolo_model,
    on_yolo_option_var_write,
    pick_yolo_model,
    prompt_pz2_detection_guard_options,
    refresh_detection_advanced_sections,
    refresh_detection_active_model_label,
    refresh_detection_refiner_guard_label,
    refresh_detection_pipeline_builder,
    refresh_detection_pipeline_model_row,
    refresh_detection_pipeline_builder_property_panel,
    refresh_detection_review_controls,
    refresh_detection_workflow_info_label,
    refresh_detect_mode_cards,
    refresh_last_detection_status_label,
    refresh_device_options,
    refresh_yolo_model_picker_state,
    remove_detection_pipeline_builder_selected_block,
    unlock_ui_after_testing,
    run_detection_stage,
    run_fast_ocr_test,
    select_detection_pipeline_builder_block,
    set_button_emphasis,
    set_detection_pipeline_builder_blocks,
    set_detection_pipeline_builder_preset,
    set_detection_method_key,
    save_last_detection_summary,
    save_detection_pipeline_blocks,
    show_detection_pipeline_model_details,
    show_last_detection_details,
    set_test_progress_counter,
    set_test_progress_detail,
    set_test_status,
    set_detection_process_log_visibility,
    sync_yolo_model_binding,
    toggle_detection_advanced_panel,
    toggle_detection_process_log,
    undo_last_detection_result,
    update_device_hint,
    update_detection_progress_ui,
    update_yolo_visibility,
    find_best_yolo_rescue_index,
    fit_detection_count_to_truths,
    get_text_mismatch_positions,
    find_best_yolo_box_backend_index,
    apply_yolo_box_backend,
    levenshtein_distance,
    pick_best_true_text,
    repair_ocr_with_yolo_boxes,
    resolve_canonical_detections,
    get_yolo_box_ocr_status_text,
)
from .z3_detection_controls_ui import use_hybrid_yolo_box_backend
from .z3_cvat_tab_ui import (
    build_cvat_tab,
    get_active_preview_context,
    pick_file,
    run_cvat_import,
    refresh_preview_bound_action_states,
    run_cvat_export,
)
from .z3_goldpack_ui import (
    build_gold_export_plate_unique_key,
    collect_gold_export_plate_candidates,
    build_campaign_aware_gold_export_counts,
    build_contextual_gold_export_counts,
    build_merged_gold_export_counts,
    count_exportable_characters_in_data,
    count_exportable_perfect_plates_in_metadata,
    count_statuses_in_metadata_mapping,
    empty_plate_layout_counts,
    empty_perfect_strategy_counts,
    empty_gold_source_counts,
    ensure_plate_source_metadata,
    format_gold_export_split_summary,
    format_perfect_strategy_counts,
    format_plate_layout_counts,
    format_selected_gold_export_source_labels,
    format_selected_gold_export_strategy_labels,
    get_gold_export_split_percentages,
    get_gold_export_meta_candidates,
    get_selected_gold_export_source_buckets,
    get_selected_gold_export_strategy_buckets,
    increment_plate_layout_counts,
    normalize_classification_char_crop_bbox,
    refresh_gold_export_filter_labels,
    refresh_gold_export_scope_label,
    refresh_gold_export_source_labels,
    run_char_classification_export,
    run_pz3_existing_dataset_split,
    run_yolo_gold_export,
    set_gold_export_scope_info,
    set_preview_record_source_info,
    update_gold_export_split_labels,
)
from .z3_ocr_lab import (
    build_live_ocr_sample_pool,
    compose_ocr_ranking_source_text,
    get_best_preset,
    get_current_prep_params,
    get_ocr_demo_dir,
    get_ocr_demo_samples,
    get_ocr_lab_sample_bundle,
    get_ocr_ranking_sample_bundle,
    get_true_texts_from_filename,
    iter_ocr_demo_source_run_dirs,
    load_cached_ocr_demo_samples,
    open_filter_lab,
    open_ocr_ranking_modal,
    open_ocr_summary_modal,
    populate_ocr_ranking_results_host,
    refresh_ocr_ranking_modal_results,
    remember_ocr_ranking_results,
    run_preset_ranking,
    seed_ocr_demo_samples,
    set_ocr_ranking_modal_label,
    set_ocr_ranking_modal_running,
    set_ocr_ranking_modal_source,
    set_ocr_ranking_modal_status,
    set_winner_acc,
    set_winner_name,
    update_ocr_ranking_modal_progress,
    update_winner_label,
)
from .z3_readiness import (
    get_campaign_step3_annotation_readiness,
    get_campaign_step3_training_readiness,
    get_step3_yolo_export_readiness_snapshot,
    inspect_pz3_dataset_source_dir,
)
from .z3_paths import (
    campaign_preview_is_below_min_extracted_plate_count,
    campaign_preview_meets_min_extracted_plate_count,
    find_latest_preview_run_dir,
    get_campaign_expected_step3_preview_plate_count,
    get_preferred_step3_preview_dir,
    get_preview_dir_plate_count,
    get_saved_step3_preview_dir,
    get_step3_char_classification_datasets_root_dir,
    get_step3_chars_root_dir,
    get_step3_datasets_root_dir,
    get_step3_summary_dir,
    is_usable_step3_preview_dir,
    preview_dir_is_campaign_inflated,
    preview_dir_has_plate_entries,
)
from .z3_view_models import (
    Step3ExtractWorkflowViewModel,
    Step3Pz3DatasetModeViewModel,
    Step3Pz3PathSelectionViewModel,
    Step3Pz3StatusPanelViewModel,
)
from .z3_character_extract_delegates import bind_character_extract_delegates

NAV_BUTTON_WIDTH = 18

YOLO = None

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

PREVIEW_BOX_MODE_OPTIONS = [
    ("AUTO", "Auto (wg etapu)"),
    ("FINAL", "Końcowe ramki treningowe"),
    ("YOLO_FILTERED", "YOLO po filtrze sekwencji"),
    ("YOLO_NMS", "YOLO po usunięciu dubli (NMS)"),
    ("YOLO_RAW", "YOLO surowe propozycje"),
]
PREVIEW_BOX_MODE_LABELS = {key: label for key, label in PREVIEW_BOX_MODE_OPTIONS}
PREVIEW_BOX_MODE_BY_LABEL = {label: key for key, label in PREVIEW_BOX_MODE_OPTIONS}
PREVIEW_BOX_MODE_BY_LABEL.update({
    "Auto (wg metody)": "AUTO",
    "Wynik końcowy": "FINAL",
    "YOLO po NMS": "YOLO_NMS",
    "YOLO surowe": "YOLO_RAW",
})

PREVIEW_SORT_OPTIONS = [
    ("DEFAULT", "Domyślne"),
    ("OK", "Po OK"),
    ("1R", "Po 1R"),
    ("2R", "Po 2R"),
    ("M", "Po M"),
    ("YOLO", "Po YOLO"),
    ("OCR", "Po OCR"),
    ("BOXES", "Po boxach"),
]
PREVIEW_SORT_LABELS = {key: label for key, label in PREVIEW_SORT_OPTIONS}
PREVIEW_SORT_COLOR_KEYS = {
    "DEFAULT": "muted",
    "OK": "success",
    "1R": "success",
    "2R": "warning",
    "M": "warning",
    "YOLO": "info",
    "OCR": "success",
    "BOXES": "info",
}
PREVIEW_LAYOUT_FILTER_OPTIONS = []
PREVIEW_LAYOUT_FILTER_LABELS = {key: label for key, label in PREVIEW_LAYOUT_FILTER_OPTIONS}
PREVIEW_LAYOUT_FILTER_COLOR_KEYS = {}

PERFECT_STRATEGY_BUCKETS = [
    ("ocr_exact", "OCR exact"),
    ("yolo_exact", "YOLO exact"),
    ("ocr_yolo_rescue", "OCR + YOLO rescue"),
    ("yolo_box_ocr", "YOLO boxy + OCR"),
    ("other_perfect", "Manual / inne perfect"),
]
PERFECT_STRATEGY_LABELS = {key: label for key, label in PERFECT_STRATEGY_BUCKETS}
GOLD_SOURCE_BUCKETS = [
    ("auto_preview", "Auto z runu"),
    ("local_manual", "Ręczne poprawki lokalne"),
    ("cvat_manual", "Poprawki CVAT"),
]
GOLD_SOURCE_LABELS = {key: label for key, label in GOLD_SOURCE_BUCKETS}
CHAR_CLASS_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DETECTION_METHOD_OPTIONS = [
    ("OCR", "OCR: odczyt znaków"),
    ("YOLO", "YOLO: odczyt i ramki"),
    ("YOLO_BOX", "YOLO boxy: tylko ramki"),
    ("YOLO_SYMBOL", "YOLO znaki: tylko symbole"),
    ("BOTH", "Hybryda: OCR odczyt + YOLO ramki"),
    ("YOLO_OCR", "Hybryda: YOLO boxy + OCR odczyt"),
]
DETECTION_METHOD_LABELS = {key: label for key, label in DETECTION_METHOD_OPTIONS}
DETECTION_METHOD_KEY_BY_LABEL = {label: key for key, label in DETECTION_METHOD_OPTIONS}
DETECTION_METHOD_KEY_BY_LABEL.update({
    "OCR": "OCR",
    "YOLO": "YOLO",
    "YB": "YOLO_BOX",
    "YOLO box": "YOLO_BOX",
    "YOLO boxy": "YOLO_BOX",
    "YOLO_BOX": "YOLO_BOX",
    "YS": "YOLO_SYMBOL",
    "YOLO znak": "YOLO_SYMBOL",
    "YOLO znaki": "YOLO_SYMBOL",
    "YOLO_SYMBOL": "YOLO_SYMBOL",
    "Hybryda OCR/YOLO": "BOTH",
    "OCR + YOLO": "BOTH",
    "OCR+YOLO": "BOTH",
    "Hybryda: OCR odczyt + YOLO ramki": "BOTH",
    "YOLO boxy + OCR": "YOLO_OCR",
    "YOLO + OCR": "YOLO_OCR",
    "YOLO+OCR": "YOLO_OCR",
    "Hybryda: YOLO boxy + OCR odczyt": "YOLO_OCR",
})
DETECTION_METHOD_CARD_META = {
    "OCR": {
        "title": "OCR",
        "desc": "OCR odczytuje cały napis i dzieli go na techniczne segmenty znaków.",
    },
    "YOLO": {
        "title": "YOLO",
        "desc": "Model detekcji YOLO wykrywa ramki znaków i proponuje klasy. Nie jest to wybór modelu do treningu.",
    },
    "YOLO_BOX": {
        "title": "YB",
        "desc": "YOLO wyznacza tylko ramki znakow. Tekst zostaje pusty do OCR, YS albo recznej korekty.",
    },
    "YOLO_SYMBOL": {
        "title": "YS",
        "desc": "YOLO wpisuje symbole w istniejace ramki. Nie tworzy nowych boxow.",
    },
    "BOTH": {
        "title": "OCR + YOLO",
        "desc": "OCR pilnuje tekstu, a model detekcji YOLO dopasowuje pozycje i może przejąć końcowe ramki.",
    },
    "YOLO_OCR": {
        "title": "YOLO boxy + OCR",
        "desc": "YOLO wyznacza boxy znaków, a OCR czyta każdy crop osobno.",
    },
}
DETECTION_PIPELINE_BLOCK_LIBRARY = {
    "ocr_symbol": {
        "badge": "OCR",
        "title": "OCR",
        "subtitle": "czyta napis i segmentuje znaki",
        "desc": "OCR odczytuje cały napis tablicy i tworzy techniczne segmenty znaków bez użycia modelu YOLO.",
        "tone": "ocr",
    },
    "yolo_box": {
        "badge": "YB",
        "title": "YOLO box",
        "subtitle": "lokalizuje ramki znaków",
        "desc": "YOLO wyznacza geometryczne położenie znaku i daje boxy do dalszej pracy.",
        "tone": "yolo_box",
    },
    "yolo_symbol": {
        "badge": "YS",
        "title": "YOLO znak",
        "subtitle": "czyta znak z klasy modelu",
        "desc": "YOLO przypisuje znak z klasy modelu albo pełni rolę rescue w hybrydzie.",
        "tone": "yolo_symbol",
    },
}
DETECTION_PIPELINE_PRESET_META = {
    "OCR": {
        "label": "OCR",
        "desc": "OCR odczytuje napis i sam tworzy segmenty znaków.",
    },
    "YOLO": {
        "label": "YOLO",
        "desc": "Model detekcji YOLO daje box i znak z klas modelu.",
    },
    "YOLO_BOX": {
        "label": "YB",
        "desc": "YOLO tworzy tylko ramki znakow.",
    },
    "YOLO_SYMBOL": {
        "label": "YS",
        "desc": "YOLO wpisuje symbole w istniejace ramki.",
    },
    "BOTH": {
        "label": "OCR + YOLO",
        "desc": "OCR czyta tekst, YOLO ustawia pozycje, a rescue jest opcjonalny.",
    },
    "YOLO_OCR": {
        "label": "YOLO boxy + OCR",
        "desc": "YOLO daje boxy, a OCR czyta cropy znaków.",
    },
}
PREVIEW_MODE_EYE_ICON = "\N{EYE}"
ADVANCED_PANEL_VISIBLE_ICON = "\N{EYE}"
ADVANCED_PANEL_HIDDEN_ICON = "\N{EYE}✕"




class CharacterAnnotationTab:
    def release_gpu_resources_for_training(self) -> None:
        """Zwalnia pamięć GPU używaną chwilowo przez OCR/YOLO w Z3."""
        try:
            cleanup_gpu_memory()
        except Exception as e:
            logger.debug(f"Nie udało się wyczyścić pamięci GPU po Z3: {e}")

    def __init__(self, parent, app):
        z3_init_runtime.__init__(self, parent, app)

    def _mark_startup_ui_ready(self):
        self._startup_ui_ready = True
        try:
            self._schedule_extract_workflow_refresh(delay_ms=320)
        except Exception:
            pass

    def is_startup_ui_ready(self) -> bool:
        return bool(getattr(self, "_startup_ui_ready", False))

    def prepare_lazy_tab_first_paint(self) -> None:
        """Dopina pierwszy, widoczny układ Z3 zanim loader schowa placeholder."""
        suppress_overlay = False
        try:
            suppress_overlay = bool(
                getattr(self.app, "_suppress_characters_lazy_first_paint_overlay_once", False)
            )
            self.app._suppress_characters_lazy_first_paint_overlay_once = False
        except Exception:
            suppress_overlay = False
        if not suppress_overlay:
            self._show_lazy_first_paint_overlay()
        try:
            self._refresh_extract_workflow_ui()
        except Exception:
            pass
        try:
            shell = getattr(self, "extract_workflow_shell", None)
            if shell is not None and not str(shell.winfo_manager()):
                shell.pack(fill=tk.X)
        except Exception:
            pass
        for _ in range(2):
            try:
                self._sync_extract_left_canvas_width()
                self._sync_extract_left_scrollregion()
            except Exception:
                pass
            for widget in (
                getattr(self, "extract_entry_cards_frame", None),
                getattr(self, "extract_workflow_shell_inner", None),
                getattr(self, "extract_left_canvas", None),
                getattr(self, "frame", None),
            ):
                if widget is None:
                    continue
                try:
                    widget.update_idletasks()
                except Exception:
                    pass
        if not suppress_overlay:
            self._schedule_lazy_first_paint_overlay_release()

    def _show_lazy_first_paint_overlay(self) -> None:
        palette = getattr(self.app, "palette", {})
        bg = palette.get("panel", "#252526")
        card_bg = palette.get("panel_alt", "#2d2d30")
        border = palette.get("accent", palette.get("border", "#3c3c3c"))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        accent = palette.get("success", "#2ecc71")

        overlay = getattr(self, "_lazy_first_paint_overlay", None)
        if overlay is None:
            overlay = tk.Frame(
                self.frame,
                bg=bg,
                bd=0,
                highlightthickness=0,
            )
            card = tk.Frame(
                overlay,
                bg=card_bg,
                bd=0,
                highlightthickness=1,
                highlightbackground=border,
                highlightcolor=border,
                padx=18,
                pady=16,
            )
            card.place(relx=0.5, rely=0.42, anchor=tk.CENTER, width=520)
            tk.Label(
                card,
                text="Przygotowuję Z3/PZ1",
                bg=card_bg,
                fg=fg,
                font=("Segoe UI Semibold", 13),
                anchor="w",
            ).pack(fill=tk.X)
            tk.Label(
                card,
                text="Układam kafle wejścia i sprawdzam pierwszy stan widoku. Za chwilę pokażę gotową kartę.",
                bg=card_bg,
                fg=muted,
                font=("Segoe UI", 9),
                justify=tk.LEFT,
                anchor="w",
                wraplength=470,
            ).pack(fill=tk.X, pady=(8, 10))
            progress = ttk.Progressbar(card, mode="indeterminate")
            progress.pack(fill=tk.X)
            status = tk.Label(
                card,
                text="Czekam na pełne dorysowanie kafli...",
                bg=card_bg,
                fg=accent,
                font=("Segoe UI", 9),
                anchor="w",
            )
            status.pack(fill=tk.X, pady=(8, 0))
            overlay._lazy_progress = progress
            self._lazy_first_paint_overlay = overlay

        try:
            overlay.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
            overlay.lift()
            progress = getattr(overlay, "_lazy_progress", None)
            if progress is not None:
                progress.start(12)
        except Exception:
            pass

    def _hide_lazy_first_paint_overlay(self) -> None:
        overlay = getattr(self, "_lazy_first_paint_overlay", None)
        if overlay is None:
            return
        try:
            progress = getattr(overlay, "_lazy_progress", None)
            if progress is not None:
                progress.stop()
        except Exception:
            pass
        try:
            overlay.place_forget()
        except Exception:
            pass

    def _extract_entry_cards_first_paint_ready(self) -> bool:
        shell = getattr(self, "extract_workflow_shell", None)
        cards = getattr(self, "_extract_entry_cards", {}) or {}
        if shell is None or not str(shell.winfo_manager()) or len(cards) < 2:
            return False
        for widgets in cards.values():
            frame = widgets.get("frame") if isinstance(widgets, dict) else None
            title = widgets.get("title") if isinstance(widgets, dict) else None
            desc = widgets.get("desc") if isinstance(widgets, dict) else None
            if frame is None or title is None or desc is None:
                return False
            try:
                width = max(int(frame.winfo_width() or 0), int(frame.winfo_reqwidth() or 0))
                if width < 280:
                    return False
                if not str(title.cget("text") or "").strip():
                    return False
                if not str(desc.cget("text") or "").strip():
                    return False
            except Exception:
                return False
        return True

    def wait_for_lazy_tab_first_paint(self, *, timeout_ms: int = 4500) -> bool:
        if getattr(self, "_lazy_first_paint_overlay", None) is None:
            try:
                self._refresh_extract_workflow_ui()
                shell = getattr(self, "extract_workflow_shell", None)
                if shell is not None and not str(shell.winfo_manager()):
                    shell.pack(fill=tk.X)
                self._sync_extract_left_canvas_width()
                self._sync_extract_left_scrollregion()
                self.frame.update_idletasks()
            except Exception:
                pass
            return True

        deadline = time.monotonic() + max(0.2, float(timeout_ms) / 1000.0)
        while time.monotonic() < deadline:
            try:
                self._refresh_extract_workflow_ui()
                shell = getattr(self, "extract_workflow_shell", None)
                if shell is not None and not str(shell.winfo_manager()):
                    shell.pack(fill=tk.X)
                self._sync_extract_left_canvas_width()
                self._sync_extract_left_scrollregion()
                self.frame.update_idletasks()
            except Exception:
                pass
            if self._extract_entry_cards_first_paint_ready():
                self._hide_lazy_first_paint_overlay()
                return True
            try:
                self.frame.after(60, lambda: None)
                self.frame.update()
            except Exception:
                break
        ready = self._extract_entry_cards_first_paint_ready()
        self._hide_lazy_first_paint_overlay()
        return ready

    def _schedule_lazy_first_paint_overlay_release(self, attempt: int = 0) -> None:
        try:
            self._sync_extract_left_canvas_width()
            self._sync_extract_left_scrollregion()
            self.frame.update_idletasks()
        except Exception:
            pass

        if self._extract_entry_cards_first_paint_ready() or attempt >= 160:
            self._hide_lazy_first_paint_overlay()
            return

        try:
            self.frame.after(75, lambda: self._schedule_lazy_first_paint_overlay_release(attempt + 1))
        except Exception:
            self._hide_lazy_first_paint_overlay()

    # =========================================================
    # Small helpers
    # =========================================================

    _reset_preview_cache = reset_preview_cache
    _reset_preview_view_state = reset_preview_view_state

    def _get_preview_badge_offsets_for_plate(self, plate_id: str):
        if not plate_id:
            return {}
        return self._preview_badge_offsets.setdefault(str(plate_id), {})

    def _get_preview_badge_offset(self, plate_id: str, badge_key: str):
        offsets = self._get_preview_badge_offsets_for_plate(plate_id)
        entry = offsets.get(str(badge_key), {})
        return float(entry.get("dx", 0.0)), float(entry.get("dy", 0.0))

    def _set_preview_badge_offset(self, plate_id: str, badge_key: str, dx: float, dy: float):
        offsets = self._get_preview_badge_offsets_for_plate(plate_id)
        offsets[str(badge_key)] = {"dx": float(dx), "dy": float(dy)}

    def _clamp_preview_badge_offset(self, badge_key: str, dx: float, dy: float):
        runtime = self._preview_badge_runtime.get(str(badge_key), {})
        base_left = float(runtime.get("base_left", 0.0))
        base_top = float(runtime.get("base_top", 0.0))
        badge_width = float(runtime.get("badge_width", 0.0))
        left_limit = float(runtime.get("left_limit", base_left))
        right_limit = float(runtime.get("right_limit", base_left + badge_width))
        top_limit = float(runtime.get("top_limit", 46.0))
        bottom_limit = float(runtime.get("bottom_limit", 0.0) or 0.0)
        badge_height = float(runtime.get("badge_height", 0.0) or 0.0)

        min_dx = left_limit - base_left
        max_dx = (right_limit - badge_width) - base_left
        if max_dx < min_dx:
            max_dx = min_dx

        dx = max(min_dx, min(max_dx, float(dx)))
        dy = max(float(top_limit) - base_top, float(dy))
        if bottom_limit > top_limit and badge_height > 0:
            dy = min(float(bottom_limit) - float(badge_height) - base_top, float(dy))
        return float(dx), float(dy)

    _clamp_preview_image_position = staticmethod(clamp_preview_image_position)
    _get_preview_image_bottom_reserved = get_preview_image_bottom_reserved

    _get_preview_canvas_size = z3_preview_overlay_runtime._get_preview_canvas_size
    _get_preview_mode_overlay_size = z3_preview_overlay_runtime._get_preview_mode_overlay_size
    _clamp_preview_mode_overlay_position = z3_preview_overlay_runtime._clamp_preview_mode_overlay_position

    _get_preview_mode_overlay_default_position = get_preview_mode_overlay_default_position

    _ensure_preview_mode_overlay_position = z3_preview_overlay_runtime._ensure_preview_mode_overlay_position

    _focus_preview_canvas = focus_preview_canvas
    _apply_preview_canvas_cursor = apply_preview_canvas_cursor
    _on_preview_canvas_enter = on_preview_canvas_enter

    def _coerce_event_to_preview_canvas(self, event=None):
        canvas = getattr(self, "preview_canvas", None)
        if event is None or canvas is None:
            return event
        if getattr(event, "widget", None) is canvas:
            return event
        try:
            x = float(getattr(event, "x_root")) - float(canvas.winfo_rootx())
            y = float(getattr(event, "y_root")) - float(canvas.winfo_rooty())
        except Exception:
            return event
        return SimpleNamespace(
            x=x,
            y=y,
            x_root=getattr(event, "x_root", None),
            y_root=getattr(event, "y_root", None),
            state=getattr(event, "state", 0),
            num=getattr(event, "num", None),
            delta=getattr(event, "delta", 0),
            keysym=getattr(event, "keysym", ""),
            char=getattr(event, "char", ""),
            widget=canvas,
        )

    def _unbind_preview_char_drag_session(self):
        bindings = list(getattr(self, "_preview_char_drag_window_bindings", []) or [])
        self._preview_char_drag_window_bindings = []
        for widget, sequence, funcid in bindings:
            try:
                widget.unbind(sequence, funcid)
            except Exception:
                pass
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            try:
                if canvas.grab_current() is canvas:
                    canvas.grab_release()
            except Exception:
                pass

    def _bind_preview_char_drag_session(self):
        self._unbind_preview_char_drag_session()
        # Nie pozwalamy, żeby odłożony zapis metadata wszedł w kolejny chwyt
        # narożnika. Zapis zostanie zaplanowany ponownie po puszczeniu LPM.
        self._cancel_scheduled_preview_metadata_save()
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            try:
                canvas.grab_set()
            except Exception:
                pass
        try:
            host = self.frame.winfo_toplevel()
        except Exception:
            host = None
        if host is None:
            return

        def _drag_motion(event):
            if getattr(event, "widget", None) is getattr(self, "preview_canvas", None):
                return None
            if not isinstance(getattr(self, "_preview_char_drag_state", None), dict):
                self._unbind_preview_char_drag_session()
                return None
            return self._on_preview_canvas_drag(self._coerce_event_to_preview_canvas(event))

        def _drag_release(event):
            if getattr(event, "widget", None) is getattr(self, "preview_canvas", None):
                return None
            if not isinstance(getattr(self, "_preview_char_drag_state", None), dict):
                self._unbind_preview_char_drag_session()
                return None
            return self._on_preview_canvas_release(self._coerce_event_to_preview_canvas(event))

        try:
            motion_id = host.bind("<B1-Motion>", _drag_motion, add="+")
            release_id = host.bind("<ButtonRelease-1>", _drag_release, add="+")
            self._preview_char_drag_window_bindings = [
                (host, "<B1-Motion>", motion_id),
                (host, "<ButtonRelease-1>", release_id),
            ]
        except Exception:
            self._preview_char_drag_window_bindings = []

    def _preview_canvas_to_image_point(self, canvas_x: float, canvas_y: float):
        state = getattr(self, "_preview_render_state", None) or {}
        scale = max(0.001, float(state.get("scale", 0.0) or 0.0))
        image_left = float(state.get("image_left", 0.0))
        image_top = float(state.get("image_top", 0.0))
        orig_w = max(1.0, float(state.get("orig_w", 1.0)))
        orig_h = max(1.0, float(state.get("orig_h", 1.0)))

        img_x = (float(canvas_x) - image_left) / scale
        img_y = (float(canvas_y) - image_top) / scale
        img_x = max(0.0, min(orig_w, img_x))
        img_y = max(0.0, min(orig_h, img_y))
        return float(img_x), float(img_y)

    def _preview_image_to_canvas_point(self, img_x: float, img_y: float):
        state = getattr(self, "_preview_render_state", None) or {}
        scale = max(0.001, float(state.get("scale", 0.0) or 0.0))
        image_left = float(state.get("image_left", 0.0))
        image_top = float(state.get("image_top", 0.0))
        return (
            (float(img_x) * scale) + image_left,
            (float(img_y) * scale) + image_top,
        )

    def _preview_point_inside_image(self, canvas_x: float, canvas_y: float) -> bool:
        state = getattr(self, "_preview_render_state", None) or {}
        return bool(
            float(state.get("image_left", 0.0)) <= float(canvas_x) <= float(state.get("image_right", -1.0))
            and float(state.get("image_top", 0.0)) <= float(canvas_y) <= float(state.get("image_bottom", -1.0))
        )

    def _get_preview_metadata_path(self) -> Path | None:
        preview_dir_raw = str(getattr(self, "preview_dir_var", None).get() if hasattr(self, "preview_dir_var") else "").strip()
        if not preview_dir_raw:
            return None
        return Path(preview_dir_raw) / "metadata.json"

    def _get_preview_active_data(self, create: bool = False):
        pid = str(getattr(self, "_preview_active_pid", "") or "")
        if not pid:
            return None
        if create:
            return self.preview_metadata.setdefault(pid, {})
        return self.preview_metadata.get(pid)

    def _get_preview_active_character_records(self, create: bool = False):
        data = self._get_preview_active_data(create=create)
        if not isinstance(data, dict):
            return []
        chars = data.get("characters")
        if isinstance(chars, list):
            return chars
        if create:
            data["characters"] = []
            return data["characters"]
        return []

    _normalize_preview_source_path_key = staticmethod(normalize_preview_source_path_key)
    _build_preview_plate_reextract_match_key = staticmethod(build_preview_plate_reextract_match_key)
    _capture_preview_reextract_seed_metadata = capture_preview_reextract_seed_metadata
    _preview_plate_has_renderable_boxes = staticmethod(preview_plate_has_renderable_boxes)
    _collect_historical_preview_reextract_seed_metadata = collect_historical_preview_reextract_seed_metadata
    _scale_preview_char_records_to_new_size = scale_preview_char_records_to_new_size
    _adapt_reextract_seed_payload_to_current_preview = adapt_reextract_seed_payload_to_current_preview
    _merge_reextract_seed_metadata_into_preview = merge_reextract_seed_metadata_into_preview

    def _sanitize_preview_char_symbol(self, value) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return ""
        if len(text) == 1 and text in CHAR_CLASS_ALPHABET:
            return text
        return ""

    _is_exportable_character_record = z3_preview_metadata_runtime._is_exportable_character_record

    _normalize_preview_char_bbox = normalize_preview_char_bbox

    def _mark_preview_char_record_manual(self, rec: dict, *, box: bool = True, sign: bool = True):
        if not isinstance(rec, dict):
            return
        if box:
            rec["box_source"] = "manual_box"
            rec["bbox_source"] = "manual"
            rec["geometry_source"] = "manual"
            rec["geometry_method"] = "manual"
        if sign:
            rec["sign_source"] = "manual_sign"
            rec["symbol_source"] = "manual"
            rec["symbol_method"] = "manual"
        if box and sign:
            rec["method"] = "manual"
            rec["source_tag"] = "manual"
            rec["source_kind"] = "local_manual"
        elif box:
            rec["method"] = "manual"
            rec["source_tag"] = "manual"
            rec["source_kind"] = "local_manual"
        elif sign:
            rec["source_kind"] = "local_manual"
        try:
            rec["confidence"] = float(rec.get("confidence", 1.0) or 1.0)
        except Exception:
            rec["confidence"] = 1.0

    _derive_preview_status_from_characters = z3_preview_metadata_runtime._derive_preview_status_from_characters
    _preview_has_reference_text_source = z3_preview_metadata_runtime._preview_has_reference_text_source
    _get_preview_reference_text_values = z3_preview_metadata_runtime._get_preview_reference_text_values
    _get_preview_filename_expected_texts = z3_preview_metadata_runtime._get_preview_filename_expected_texts
    _get_preview_expected_texts = z3_preview_metadata_runtime._get_preview_expected_texts
    _resolve_preview_expected_text_for_crop = z3_preview_metadata_runtime._resolve_preview_expected_text_for_crop
    _derive_preview_status_from_data = z3_preview_metadata_runtime._derive_preview_status_from_data
    _get_preview_live_status = z3_preview_metadata_runtime._get_preview_live_status

    _get_preview_status_presentation = get_preview_status_presentation

    _recalculate_preview_statuses_in_metadata = z3_preview_metadata_runtime._recalculate_preview_statuses_in_metadata
    _persist_preview_metadata = z3_preview_metadata_runtime._persist_preview_metadata
    _flush_scheduled_preview_metadata_save = z3_preview_metadata_runtime._flush_scheduled_preview_metadata_save
    _cancel_scheduled_preview_metadata_save = z3_preview_metadata_runtime._cancel_scheduled_preview_metadata_save
    _schedule_preview_metadata_save = z3_preview_metadata_runtime._schedule_preview_metadata_save
    _schedule_preview_info_refresh = z3_preview_metadata_runtime._schedule_preview_info_refresh

    _refresh_preview_layout_override_ui_light = refresh_preview_layout_override_ui_light

    _clone_preview_plate_data = z3_preview_metadata_runtime._clone_preview_plate_data
    _get_preview_history_stack = z3_preview_metadata_runtime._get_preview_history_stack
    _push_preview_history_snapshot = z3_preview_metadata_runtime._push_preview_history_snapshot
    _trim_preview_history_stack = z3_preview_metadata_runtime._trim_preview_history_stack
    _refresh_preview_listbox_row = z3_preview_metadata_runtime._refresh_preview_listbox_row

    _refresh_preview_live_metadata_ui = refresh_preview_live_metadata_ui

    _restore_preview_plate_history_snapshot = z3_preview_metadata_runtime._restore_preview_plate_history_snapshot
    _undo_preview_edit = z3_preview_metadata_runtime._undo_preview_edit
    _redo_preview_edit = z3_preview_metadata_runtime._redo_preview_edit
    _event_has_control_modifier = staticmethod(z3_preview_metadata_runtime._event_has_control_modifier)
    _event_has_shift_modifier = staticmethod(z3_preview_metadata_runtime._event_has_shift_modifier)
    _pane_has_child = staticmethod(z3_preview_metadata_runtime._pane_has_child)
    _get_current_preview_list_index = z3_preview_metadata_runtime._get_current_preview_list_index
    _clear_listbox_selection_fast = staticmethod(z3_preview_metadata_runtime._clear_listbox_selection_fast)

    _on_preview_list_mouse_primary = on_preview_list_mouse_primary

    _handle_preview_list_arrow_nav = z3_preview_metadata_runtime._handle_preview_list_arrow_nav
    _select_preview_relative = z3_preview_metadata_runtime._select_preview_relative
    _append_preview_status_sentence = staticmethod(z3_preview_typing_runtime._append_preview_status_sentence)
    _clear_preview_char_label_canvas_fields = z3_preview_typing_runtime._clear_preview_char_label_canvas_fields
    _cancel_preview_char_label_interaction = z3_preview_typing_runtime._cancel_preview_char_label_interaction
    _get_preview_typing_state = z3_preview_typing_runtime._get_preview_typing_state
    _format_preview_typing_status = z3_preview_typing_runtime._format_preview_typing_status

    _sync_preview_edit_status_visibility = sync_preview_edit_status_visibility

    _get_preview_typing_overlay_text = z3_preview_typing_runtime._get_preview_typing_overlay_text
    _format_preview_operation_assistant_status = z3_preview_typing_runtime._format_preview_operation_assistant_status
    _preview_status_overlay_available = z3_preview_typing_runtime._preview_status_overlay_available
    _preview_operation_overlay_available = z3_preview_typing_runtime._preview_operation_overlay_available
    _resolve_preview_typing_overlay_anchor = z3_preview_typing_runtime._resolve_preview_typing_overlay_anchor
    _get_preview_typing_overlay_bottom_offset = z3_preview_typing_runtime._get_preview_typing_overlay_bottom_offset
    _get_preview_controls_legend_clearance_y = z3_preview_typing_runtime._get_preview_controls_legend_clearance_y
    _get_preview_typing_overlay_anchor = z3_preview_typing_runtime._get_preview_typing_overlay_anchor
    _configure_preview_typing_overlay_text = z3_preview_typing_runtime._configure_preview_typing_overlay_text

    _refresh_preview_typing_overlay_visibility = refresh_preview_typing_overlay_visibility
    _update_preview_edit_status = update_preview_edit_status

    _get_preview_char_label_canvas_rect = z3_preview_typing_runtime._get_preview_char_label_canvas_rect
    _find_preview_char_label_hit = z3_preview_typing_runtime._find_preview_char_label_hit
    _get_leftmost_preview_char_index = z3_preview_typing_runtime._get_leftmost_preview_char_index
    _resolve_preview_char_label_entry_index = z3_preview_typing_runtime._resolve_preview_char_label_entry_index
    _activate_preview_char_label_input = z3_preview_typing_runtime._activate_preview_char_label_input
    _assign_character_to_active_preview_label = z3_preview_typing_runtime._assign_character_to_active_preview_label

    def _schedule_preview_select_render(self, delay_ms: int = 35):
        """Render only the latest selected preview row after rapid list/key navigation settles."""
        try:
            pending_zoom_after = getattr(self, "_preview_zoom_render_after_id", None)
            if pending_zoom_after:
                self.preview_canvas.after_cancel(pending_zoom_after)
                self._preview_zoom_render_after_id = None
            self._preview_zoom_pending_state = None
        except Exception:
            pass
        try:
            pending_after = getattr(self, "_preview_list_select_after_id", None)
            if pending_after:
                self.frame.after_cancel(pending_after)
        except Exception:
            pass
        try:
            pending_detail_after = getattr(self, "_preview_detail_render_after_id", None)
            if pending_detail_after:
                self.frame.after_cancel(pending_detail_after)
                self._preview_detail_render_after_id = None
        except Exception:
            pass
        try:
            pending_fast_hud_after = getattr(self, "_preview_fast_hud_refresh_after_id", None)
            if pending_fast_hud_after:
                self.frame.after_cancel(pending_fast_hud_after)
                self._preview_fast_hud_refresh_after_id = None
        except Exception:
            pass
        try:
            pending_prefetch_after = getattr(self, "_preview_neighbor_prefetch_after_id", None)
            if pending_prefetch_after:
                self.frame.after_cancel(pending_prefetch_after)
                self._preview_neighbor_prefetch_after_id = None
        except Exception:
            pass

        self._suppress_preview_reload_on_list_select = True
        self._preview_fast_select_render = True
        try:
            self._preview_select_generation = int(getattr(self, "_preview_select_generation", 0) or 0) + 1
        except Exception:
            self._preview_select_generation = 1

        def _render_selected_preview_once():
            try:
                self._preview_list_select_after_id = None
                self._suppress_preview_reload_on_list_select = False
                self._on_preview_select(None)
            except Exception:
                self._suppress_preview_reload_on_list_select = False
            finally:
                self._preview_keyboard_crop_navigation_active = False

        try:
            delay = max(1, int(delay_ms))
        except Exception:
            delay = 35
        try:
            self._preview_list_select_after_id = self.frame.after(delay, _render_selected_preview_once)
        except Exception:
            self._suppress_preview_reload_on_list_select = False
            self._on_preview_select(None)

    def _reset_current_preview_view(self):
        plate_id = str(getattr(self, "_preview_active_pid", "") or "")
        if plate_id:
            self._preview_badge_offsets.pop(plate_id, None)
        self._reset_preview_view_state()
        if getattr(self, "preview_canvas", None) is not None:
            self._on_preview_select(None)

    def _init_preview_vertical_split(self):
        pane = getattr(self, "preview_vertical_split", None)
        if pane is None or self._preview_vertical_split_ready:
            return

        try:
            total_w = int(pane.winfo_width() or 0)
            if total_w < 240:
                self.frame.after(120, self._init_preview_vertical_split)
                return
            total_w = max(760, total_w)
            min_left = 240
            max_left = 340
            target_x = max(min_left, min(int(total_w * 0.28), max_left))
            pane.sash_place(0, target_x, 1)
            self._preview_vertical_split_ready = True
        except Exception:
            pass

    def _apply_preview_badge_selection_style(self):
        canvas = getattr(self, "preview_canvas", None)
        if canvas is None:
            return

        for badge_key, runtime in getattr(self, "_preview_badge_runtime", {}).items():
            is_selected = str(badge_key) == str(getattr(self, "_preview_selected_badge_key", None))
            box_color = str(runtime.get("box_color", "#cccccc"))
            line_color = str(runtime.get("line_color", "#cccccc"))
            box_width = 4 if is_selected else int(runtime.get("box_width", 2))
            line_width = 3 if is_selected else int(runtime.get("line_width", 1))

            try:
                canvas.itemconfigure(runtime.get("box_id"), outline=box_color, width=box_width)
            except Exception:
                pass

            try:
                canvas.itemconfigure(runtime.get("line_id"), fill=line_color, width=line_width)
            except Exception:
                pass

    def _set_preview_selected_badge(self, badge_key: str = None):
        self._preview_selected_badge_key = str(badge_key) if badge_key else None
        self._apply_preview_badge_selection_style()

    def _move_preview_badge_to_offset(self, badge_key: str, dx: float, dy: float):
        runtime = self._preview_badge_runtime.get(str(badge_key))
        canvas = getattr(self, "preview_canvas", None)
        if runtime is None or canvas is None:
            return

        dx, dy = self._clamp_preview_badge_offset(badge_key, dx, dy)

        prev_dx = float(runtime.get("offset_dx", 0.0))
        prev_dy = float(runtime.get("offset_dy", 0.0))
        delta_x = float(dx) - prev_dx
        delta_y = float(dy) - prev_dy

        if abs(delta_x) > 0.001 or abs(delta_y) > 0.001:
            try:
                canvas.move(runtime["tag"], delta_x, delta_y)
            except Exception:
                pass

        runtime["offset_dx"] = float(dx)
        runtime["offset_dy"] = float(dy)

        try:
            line_start_y = float(runtime.get("base_line_y", runtime.get("base_bottom_y", 0.0))) + float(dy)
            box_anchor_y = float(runtime.get("box_anchor_y", runtime.get("box_top_y", 0.0)))
            canvas.coords(
                runtime["line_id"],
                float(runtime["base_center_x"]) + float(dx),
                line_start_y,
                float(runtime["box_center_x"]),
                box_anchor_y,
            )
        except Exception:
            pass

    def _on_preview_canvas_configure(self, event=None):
        if getattr(self, "_preview_render_when_visible", False):
            self._schedule_preview_select_render(delay_ms=35)
            return
        size = (getattr(event, "width", None), getattr(event, "height", None))
        if event is not None and size == getattr(self, "_preview_last_canvas_size", None):
            return
        self._preview_last_canvas_size = size
        if bool(getattr(self, "_pz2_panel_resize_active", False)):
            self._pz2_panel_resize_pending_preview = True
            self._schedule_preview_overlay_relayout(delay_ms=90)
            return
        self._schedule_preview_stabilized_rerender(delay_ms=90)

    def _begin_pz2_panel_resize(self, event=None):
        self._pz2_panel_resize_active = True
        self._pz2_panel_resize_pending_preview = False
        return None

    def _end_pz2_panel_resize(self, event=None):
        was_active = bool(getattr(self, "_pz2_panel_resize_active", False))
        self._pz2_panel_resize_active = False
        if not was_active:
            return None
        try:
            self._schedule_preview_overlay_relayout(delay_ms=35, force_render=True)
        except Exception:
            pass
        try:
            self._schedule_preview_stabilized_rerender(delay_ms=160)
        except Exception:
            pass
        try:
            self._schedule_detect_right_adaptive_wrap_refresh()
        except Exception:
            pass
        return None

    def _schedule_preview_overlay_relayout(
        self,
        delay_ms: int = 45,
        *,
        force_render: bool = False,
        include_legend: bool = True,
        refresh_legend: bool = True,
    ):
        if getattr(self, "_preview_fullscreen_transition_active", False):
            return
        canvas_host = getattr(self, "preview_canvas_host", None)
        if canvas_host is None:
            return

        previous_after_id = getattr(self, "_preview_overlay_relayout_after_id", None)
        if previous_after_id:
            try:
                canvas_host.after_cancel(previous_after_id)
            except Exception:
                pass
            self._preview_overlay_relayout_after_id = None

        def _relayout_after_geometry_settles():
            self._preview_overlay_relayout_after_id = None
            if bool(getattr(self, "_pz2_panel_resize_active", False)):
                self._schedule_preview_overlay_relayout(delay_ms=90, force_render=force_render)
                return
            try:
                self._place_preview_overlay_dock(force_render=force_render)
            except Exception:
                pass
            if bool(include_legend):
                try:
                    self._place_preview_hint_overlay(refresh=bool(refresh_legend))
                except Exception:
                    pass

        try:
            self._preview_overlay_relayout_after_id = canvas_host.after(
                max(1, int(delay_ms)),
                _relayout_after_geometry_settles,
            )
        except Exception:
            self._preview_overlay_relayout_after_id = None

    def _schedule_preview_stabilized_rerender(self, delay_ms: int = 35):
        canvas = getattr(self, "preview_canvas", None)
        if canvas is None:
            return

        previous_after_id = getattr(self, "_preview_stabilized_render_after_id", None)
        if previous_after_id:
            try:
                canvas.after_cancel(previous_after_id)
            except Exception:
                pass
            self._preview_stabilized_render_after_id = None

        def _rerender_after_layout_settles():
            self._preview_stabilized_render_after_id = None
            if getattr(self, "_preview_fullscreen_transition_active", False):
                old_size = getattr(self, "_preview_fullscreen_old_canvas_size", None)
                current_size = (canvas.winfo_width(), canvas.winfo_height())
                started = getattr(self, "_preview_fullscreen_transition_started", 0.0)
                # Native maximize/restore can deliver its first Configure after
                # the timer. Do not redraw the old viewport during that gap.
                if old_size == current_size and time.perf_counter() - started < 0.4:
                    self._schedule_preview_stabilized_rerender(delay_ms=45)
                    return
            self._preview_fullscreen_transition_active = False
            if getattr(self, "preview_canvas", None) is None:
                return
            if not bool(getattr(self, "_preview_active_pid", None)):
                return
            if (
                getattr(self, "_preview_char_drag_state", None) is not None
                or getattr(self, "_preview_char_add_state", None) is not None
            ):
                return
            self._on_preview_select(None)

        try:
            self._preview_stabilized_render_after_id = canvas.after(
                max(1, int(delay_ms)),
                _rerender_after_layout_settles,
            )
        except Exception:
            self._preview_stabilized_render_after_id = None

    _on_preview_mode_eye_press = z3_preview_overlay_runtime._on_preview_mode_eye_press
    _on_preview_mode_eye_drag = z3_preview_overlay_runtime._on_preview_mode_eye_drag
    _on_preview_mode_eye_release = z3_preview_overlay_runtime._on_preview_mode_eye_release
    _get_preview_selected_char_record = z3_preview_editor_runtime._get_preview_selected_char_record
    _is_preview_char_record_selected = z3_preview_editor_runtime._is_preview_char_record_selected
    _preview_char_add_requested = z3_preview_editor_runtime._preview_char_add_requested
    _find_preview_char_record_index = staticmethod(z3_preview_editor_runtime._find_preview_char_record_index)

    _select_preview_character_box = select_preview_character_box

    _select_hovered_preview_char_box = z3_preview_editor_runtime._select_hovered_preview_char_box
    _cycle_preview_character_selection = z3_preview_editor_runtime._cycle_preview_character_selection
    _cycle_preview_character_row = z3_preview_editor_runtime._cycle_preview_character_row
    _get_preview_char_handle_radius = z3_preview_editor_runtime._get_preview_char_handle_radius
    _get_preview_char_move_handle_radius = z3_preview_editor_runtime._get_preview_char_move_handle_radius
    _find_preview_character_box_hit = z3_preview_editor_runtime._find_preview_character_box_hit
    _find_preview_character_handle_hit = z3_preview_editor_runtime._find_preview_character_handle_hit
    _find_preview_character_move_handle_hit = z3_preview_editor_runtime._find_preview_character_move_handle_hit
    _find_preview_character_grip_hit = z3_preview_editor_runtime._find_preview_character_grip_hit
    _get_preview_character_canvas_tag = staticmethod(z3_preview_editor_runtime._get_preview_character_canvas_tag)
    _get_preview_character_record_canvas_tag = staticmethod(z3_preview_editor_runtime._get_preview_character_record_canvas_tag)

    _redraw_preview_add_box_overlay_only = redraw_preview_add_box_overlay_only
    _draw_preview_fast_render_details = draw_preview_fast_render_details

    _clear_preview_character_drag_visual = z3_preview_editor_runtime._clear_preview_character_drag_visual

    _log_preview_edit_flow = log_preview_edit_flow
    _log_preview_latency = log_preview_latency
    _schedule_preview_latency_paint = schedule_preview_latency_paint
    _start_preview_latency_probe = start_preview_latency_probe
    _update_preview_character_selection_items_fast = update_preview_character_selection_items_fast
    _update_preview_character_drag_visual = update_preview_character_drag_visual
    _draw_preview_plate_status_frame = draw_preview_plate_status_frame
    _draw_preview_layout_separator = draw_preview_layout_separator
    _update_preview_layout_separator_visual = update_preview_layout_separator_visual
    _redraw_preview_character_overlay_only = redraw_preview_character_overlay_only
    _redraw_preview_character_overlays_light = redraw_preview_character_overlays_light

    _refresh_preview_character_selection_visual = z3_preview_editor_runtime._refresh_preview_character_selection_visual
    _refresh_preview_editor_toolbar = z3_preview_editor_runtime._refresh_preview_editor_toolbar
    def _ensure_preview_final_box_mode(self, *, render_preview: bool = True):
        if self._get_preview_box_mode_key() == "FINAL":
            return
        try:
            self.preview_box_mode_var.set(PREVIEW_BOX_MODE_LABELS["FINAL"])
        except Exception:
            pass
        if bool(render_preview):
            self._on_preview_box_mode_change()
            return
        try:
            self._save_local_setting("char_preview_box_mode", self._get_preview_box_mode_key())
        except Exception:
            pass
        try:
            self._refresh_preview_mode_overlay_visibility()
        except Exception:
            pass
        self._refresh_preview_editor_toolbar()
        if not self._redraw_preview_character_overlays_light():
            self._update_preview_box_info_label()

    _set_preview_char_editor_modes = z3_preview_editor_runtime._set_preview_char_editor_modes
    _toggle_preview_char_edit_mode = z3_preview_editor_runtime._toggle_preview_char_edit_mode
    _toggle_preview_char_add_mode = z3_preview_editor_runtime._toggle_preview_char_add_mode
    _toggle_preview_char_label_mode = z3_preview_editor_runtime._toggle_preview_char_label_mode
    _on_preview_char_label_shortcut = z3_preview_editor_runtime._on_preview_char_label_shortcut

    _persist_active_preview_characters = persist_active_preview_characters

    _edit_selected_preview_char_symbol = z3_preview_editor_runtime._edit_selected_preview_char_symbol
    _edit_preview_source_filename = z3_preview_editor_runtime._edit_preview_source_filename
    _delete_selected_preview_char_box = z3_preview_editor_runtime._delete_selected_preview_char_box

    _on_preview_canvas_motion = on_preview_canvas_motion
    _on_preview_canvas_leave = on_preview_canvas_leave

    _on_preview_prev_shortcut = z3_preview_editor_runtime._on_preview_prev_shortcut
    _on_preview_next_shortcut = z3_preview_editor_runtime._on_preview_next_shortcut
    _on_preview_fit_shortcut = z3_preview_editor_runtime._on_preview_fit_shortcut
    _on_preview_enter_fullscreen_shortcut = z3_preview_editor_runtime._on_preview_enter_fullscreen_shortcut
    _on_preview_escape_shortcut = z3_preview_editor_runtime._on_preview_escape_shortcut

    _on_preview_canvas_keypress = on_preview_canvas_keypress
    _on_preview_canvas_keyrelease = on_preview_canvas_keyrelease

    _toggle_preview_fullscreen = z3_preview_editor_runtime._toggle_preview_fullscreen

    def _get_preview_legend_font(self, size: int, weight: str = "normal"):
        cache = getattr(self, "_preview_legend_font_cache", None)
        if cache is None:
            cache = {}
            self._preview_legend_font_cache = cache

        key = (int(size), str(weight))
        font_obj = cache.get(key)
        if font_obj is None:
            font_obj = tkfont.Font(self.frame, family="Segoe UI", size=int(size), weight=str(weight))
            cache[key] = font_obj
        return font_obj

    _truncate_preview_filename = staticmethod(z3_preview_editor_runtime._truncate_preview_filename)
    _get_preview_legend_context = z3_preview_editor_runtime._get_preview_legend_context

    def _is_preview_controls_legend_expanded(self) -> bool:
        return bool(getattr(self, "_preview_controls_legend_expanded", False))

    _get_preview_legend_theme = get_preview_legend_theme
    _toggle_preview_controls_legend = toggle_preview_controls_legend

    def _is_preview_controls_legend_grab_hit(self, local_x: float, local_y: float) -> bool:
        bbox = getattr(self, "_preview_controls_legend_grab_bbox", None)
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return False
        return x1 <= float(local_x) <= x2 and y1 <= float(local_y) <= y2

    _clamp_preview_controls_legend_offsets = clamp_preview_controls_legend_offsets
    _update_preview_controls_legend_cursor = update_preview_controls_legend_cursor
    _on_preview_controls_legend_press = on_preview_controls_legend_press
    _on_preview_controls_legend_drag = on_preview_controls_legend_drag
    _on_preview_controls_legend_motion = on_preview_controls_legend_motion
    _on_preview_controls_legend_leave = on_preview_controls_legend_leave
    _on_preview_controls_legend_release = on_preview_controls_legend_release

    def _on_preview_controls_legend_configure(self, event=None):
        if not bool(getattr(self, "_preview_controls_legend_visible", True)):
            return None
        self._schedule_preview_controls_legend_refresh(delay_ms=70)
        return None

    def _schedule_preview_controls_legend_refresh(self, delay_ms: int = 50):
        canvas = getattr(self, "preview_controls_canvas", None)
        if canvas is None:
            return
        previous_after_id = getattr(self, "_preview_controls_legend_configure_after_id", None)
        if previous_after_id:
            try:
                canvas.after_cancel(previous_after_id)
            except Exception:
                pass
            self._preview_controls_legend_configure_after_id = None

        def _refresh_after_geometry_settles():
            self._preview_controls_legend_configure_after_id = None
            if bool(getattr(self, "_pz2_panel_resize_active", False)):
                self._schedule_preview_controls_legend_refresh(delay_ms=90)
                return
            self._refresh_preview_controls_legend()

        try:
            self._preview_controls_legend_configure_after_id = canvas.after(
                max(1, int(delay_ms)),
                _refresh_after_geometry_settles,
            )
        except Exception:
            self._preview_controls_legend_configure_after_id = None

    _on_preview_controls_legend_mousewheel = on_preview_controls_legend_mousewheel
    _build_preview_legend_sections = build_preview_legend_sections
    _refresh_preview_controls_legend = refresh_preview_controls_legend
    _place_preview_hint_overlay = place_preview_hint_overlay

    _schedule_preview_overlay_toggle_job = z3_preview_overlay_runtime._schedule_preview_overlay_toggle_job
    _toggle_preview_overlay_dock = z3_preview_overlay_runtime._toggle_preview_overlay_dock
    _toggle_preview_overlay_dock_tool = z3_preview_overlay_runtime._toggle_preview_overlay_dock_tool

    _get_preview_overlay_dock_theme = get_preview_overlay_dock_theme
    _get_preview_step3_gate_overlay_state = get_preview_step3_gate_overlay_state
    _render_preview_overlay_dock = render_preview_overlay_dock
    _place_preview_overlay_dock = place_preview_overlay_dock
    _apply_preview_fullscreen_chrome = apply_preview_fullscreen_chrome
    _set_preview_fullscreen = set_preview_fullscreen
    _update_preview_toolbar_state = update_preview_toolbar_state

    _extract_preview_badge_key_from_current_item = z3_preview_overlay_runtime._extract_preview_badge_key_from_current_item
    _extract_preview_action_from_current_item = z3_preview_overlay_runtime._extract_preview_action_from_current_item
    _clear_preview_canvas_action = z3_preview_overlay_runtime._clear_preview_canvas_action

    _start_preview_character_box_drag = start_preview_character_box_drag
    _on_preview_canvas_press = on_preview_canvas_press
    _on_preview_canvas_secondary_press = on_preview_canvas_secondary_press
    _on_preview_canvas_drag = on_preview_canvas_drag
    _finalize_preview_char_add_state = finalize_preview_char_add_state
    _on_preview_canvas_release = on_preview_canvas_release
    _on_preview_canvas_mousewheel = on_preview_canvas_mousewheel
    _char_record_to_symbol_and_x = staticmethod(char_record_to_symbol_and_x)

    _get_preview_dir_plate_count = get_preview_dir_plate_count
    _preview_dir_has_plate_entries = preview_dir_has_plate_entries

    def _is_campaign_char_step3_context(self) -> bool:
        try:
            return bool(
                getattr(self, "_step3_linear_mode", False)
                and CAMPAIGN.get_active_project_name()
                and str(CAMPAIGN.get_iteration_target() or "").strip().lower() == "char"
            )
        except Exception:
            return False

    @staticmethod
    def _get_campaign_step3_min_extracted_plate_count() -> int:
        return 10

    _campaign_preview_meets_min_extracted_plate_count = campaign_preview_meets_min_extracted_plate_count
    _campaign_preview_is_below_min_extracted_plate_count = campaign_preview_is_below_min_extracted_plate_count
    _get_campaign_expected_step3_preview_plate_count = get_campaign_expected_step3_preview_plate_count
    _preview_dir_is_campaign_inflated = preview_dir_is_campaign_inflated
    _is_usable_step3_preview_dir = is_usable_step3_preview_dir
    _get_saved_step3_preview_dir = get_saved_step3_preview_dir
    _get_preferred_step3_preview_dir = get_preferred_step3_preview_dir
    _find_latest_preview_run_dir = find_latest_preview_run_dir

    def _load_json_file_safely(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None

    def _find_preview_plate_image_path(self, images_dir: Path, pid: str) -> Path | None:
        pid_str = str(pid or "").strip()
        if not pid_str:
            return None
        for ext in sorted(CONFIG.IMAGE_EXTENSIONS):
            candidate = images_dir / f"{pid_str}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _get_preview_plate_image_size(self, preview_dir: Path | None, pid: str) -> tuple[int, int] | None:
        try:
            if preview_dir is None:
                return None
            images_dir = Path(preview_dir) / "images"
            img_path = self._find_preview_plate_image_path(images_dir, pid)
            if img_path is None:
                return None
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return None
            return int(w), int(h)
        except Exception:
            return None

    _build_live_ocr_sample_pool = build_live_ocr_sample_pool
    _get_ocr_demo_dir = get_ocr_demo_dir
    _load_cached_ocr_demo_samples = load_cached_ocr_demo_samples
    _iter_ocr_demo_source_run_dirs = iter_ocr_demo_source_run_dirs
    _seed_ocr_demo_samples = seed_ocr_demo_samples
    _get_ocr_demo_samples = get_ocr_demo_samples
    _get_ocr_lab_sample_bundle = get_ocr_lab_sample_bundle
    _get_ocr_ranking_sample_bundle = get_ocr_ranking_sample_bundle

    def _restore_preview_context_from_project(self, require_plates: bool = False):
        """
        Przywraca ostatni preview run projektu, jeśli istnieje.
        """
        preview_dir = self._get_preferred_step3_preview_dir(
            require_plates=require_plates,
            allow_fallback=False,
        )
        if not preview_dir:
            preview_dir = self._get_preferred_step3_preview_dir(
                require_plates=require_plates,
                allow_fallback=True,
            )
        if not preview_dir:
            return False

        try:
            current_preview_dir = str(self.preview_dir_var.get() or "").strip()
        except Exception:
            current_preview_dir = ""
        try:
            current_path = Path(current_preview_dir).resolve() if current_preview_dir else None
            target_path = Path(preview_dir).resolve()
            preview_changed = not bool(current_path and current_path == target_path)
        except Exception:
            preview_changed = str(current_preview_dir or "").strip() != str(preview_dir or "").strip()
        try:
            meta_path = Path(preview_dir) / "metadata.json"
            if (
                not preview_changed
                and bool(getattr(self, "preview_metadata", None))
                and getattr(self, "_loaded_meta_path", None) == meta_path
                and bool(getattr(self, "_listbox_pid_by_index", None))
            ):
                return True
        except Exception:
            pass

        try:
            self.preview_dir_var.set(preview_dir)
            if preview_changed:
                self._reset_preview_cache()
            if bool(getattr(self, "_detect_tab_built", False)) and hasattr(self, "plates_listbox"):
                self._load_preview_data(quiet=True)
            return True
        except Exception as e:
            logger.debug(f"Nie udało się przywrócić preview runu projektu: {e}")
            return False

    _ensure_detection_preview_loaded = ensure_detection_preview_loaded
    _open_detection_subtab_with_preview = open_detection_subtab_with_preview

    def _schedule_detection_preview_autoload(self):
        if not hasattr(self, "frame"):
            return
        if not bool(getattr(self, "_detect_tab_built", False)):
            return

        after_id = getattr(self, "_detect_preview_autoload_after_id", None)
        if after_id is not None:
            try:
                self.frame.after_cancel(after_id)
            except Exception:
                pass
            finally:
                self._detect_preview_autoload_after_id = None

        def _run():
            self._detect_preview_autoload_after_id = None
            try:
                self._ensure_detection_preview_loaded()
            except Exception as e:
                logger.debug(f"Nie udało się zaplanować autoload PZ2: {e}")

        try:
            self._detect_preview_autoload_after_id = self.frame.after_idle(_run)
        except Exception:
            self._detect_preview_autoload_after_id = None

    def _metadata_has_detection_results(self, metadata_map) -> bool:
        for _, data in (metadata_map or {}).items():
            if not isinstance(data, dict):
                continue

            status = str(data.get("status", "unknown")).strip().lower()
            if status in ("perfect", "needs_fix"):
                return True

            if isinstance(data.get("characters"), list) and data.get("characters"):
                return True

            for key in ("yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
                if isinstance(data.get(key), list) and data.get(key):
                    return True

            if str(data.get("fusion_strategy", "") or "").strip():
                return True

        return False

    def _preview_dir_has_completed_detection_output(self, preview_dir=None) -> bool:
        preview_dir_raw = str(
            preview_dir
            if preview_dir is not None
            else (self.preview_dir_var.get() or self._get_preferred_step3_preview_dir(allow_fallback=True) or "")
        ).strip()
        if not preview_dir_raw:
            return False

        try:
            meta_path = Path(preview_dir_raw) / "metadata.json"
            if not meta_path.exists():
                return False

            loaded = read_preview_metadata(self, meta_path)
            if not isinstance(loaded, dict):
                return False

            return self._metadata_has_detection_results(loaded)
        except Exception:
            return False

    def _sync_step3_access_from_preview_state(
        self,
        metadata_map=None,
        *,
        mark_current_work: bool = False,
        mark_reason: str = "pz2_manual_ready",
    ):
        stage1_ready = False
        stage2_ready = False
        requires_current_pz2 = False
        in_campaign = bool(getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
        hold_pz2_after_reextract = bool(
            in_campaign and getattr(self, "_campaign_step3_hold_pz2_after_reextract", False)
        )

        try:
            stage1_ready = self.can_restore_step3_substep(2)
        except Exception:
            stage1_ready = False

        if metadata_map is not None:
            stage2_ready = self._metadata_has_detection_results(metadata_map)
        if not stage2_ready:
            stage2_ready = self._preview_dir_has_completed_detection_output()
        if hold_pz2_after_reextract:
            stage2_ready = False
        elif in_campaign:
            try:
                stage2_ready = stage2_ready or bool(self._campaign_step3_pz2_base_ready())
            except Exception:
                pass
            try:
                requires_current_pz2 = bool(self._campaign_step3_requires_current_pz2_for_pz3())
            except Exception:
                requires_current_pz2 = False
            if requires_current_pz2:
                try:
                    current_pz2_ready = bool(self._campaign_step3_pz2_current_contract_ready())
                except Exception:
                    current_pz2_ready = False
                if bool(mark_current_work) and stage2_ready and not current_pz2_ready:
                    try:
                        _mark_t06_pz2_contract(
                            self,
                            reason=str(mark_reason or "pz2_manual_ready"),
                            force=True,
                        )
                        current_pz2_ready = bool(self._campaign_step3_pz2_current_contract_ready())
                    except Exception:
                        current_pz2_ready = False
                stage2_ready = bool(stage2_ready and current_pz2_ready)

        if stage1_ready:
            try:
                self._set_button_state("btn_to_detect", True)
                self._set_subtab_state(self.tab_detect, "normal")
            except Exception:
                pass
            if in_campaign:
                try:
                    CAMPAIGN.set_step3_stage1_done(True)
                except Exception:
                    pass

        if stage2_ready:
            try:
                self._set_button_state("btn_to_dataset", True)
                self._set_subtab_state(self.tab_dataset, "normal")
            except Exception:
                pass
            if in_campaign:
                try:
                    CAMPAIGN.set_step3_stage2_done(True)
                except Exception:
                    pass
        elif hold_pz2_after_reextract:
            try:
                self._set_button_state("btn_to_dataset", False)
                self._set_subtab_state(self.tab_dataset, "disabled")
            except Exception:
                pass
            if in_campaign:
                try:
                    CAMPAIGN.set_step3_stage2_done(False)
                except Exception:
                    pass
        elif in_campaign and requires_current_pz2:
            try:
                self._set_button_state("btn_to_dataset", False)
                self._set_subtab_state(self.tab_dataset, "disabled")
            except Exception:
                pass
            try:
                CAMPAIGN.set_step3_stage2_done(False)
            except Exception:
                pass

        try:
            self._sync_step3_nav_buttons()
        except Exception:
            pass

        try:
            self._update_step3_finish_button_state()
        except Exception:
            pass

    _sort_character_records_by_x = z3_plate_layout_runtime._sort_character_records_by_x
    _get_preview_forced_layout_key = z3_plate_layout_runtime._get_preview_forced_layout_key
    _is_preview_two_row_layout_active = z3_plate_layout_runtime._is_preview_two_row_layout_active
    _is_preview_layout_separator_interactive = z3_plate_layout_runtime._is_preview_layout_separator_interactive
    _should_preview_use_two_row_layers = z3_plate_layout_runtime._should_preview_use_two_row_layers
    _normalize_preview_layout_separator = z3_plate_layout_runtime._normalize_preview_layout_separator
    _build_auto_preview_layout_separator = z3_plate_layout_runtime._build_auto_preview_layout_separator
    _ensure_preview_layout_separator = z3_plate_layout_runtime._ensure_preview_layout_separator
    _get_preview_layout_separator_for_reading = z3_plate_layout_runtime._get_preview_layout_separator_for_reading
    _normalize_preview_manual_layout_override = z3_plate_layout_runtime._normalize_preview_manual_layout_override
    _capture_preview_manual_layout_state = z3_plate_layout_runtime._capture_preview_manual_layout_state
    _restore_preview_manual_layout_state = z3_plate_layout_runtime._restore_preview_manual_layout_state
    _backfill_preview_manual_layout_from_related_runs = z3_plate_layout_runtime._backfill_preview_manual_layout_from_related_runs
    _capture_preview_two_row_layout_state = z3_plate_layout_runtime._capture_preview_two_row_layout_state
    _restore_preview_two_row_layout_state = z3_plate_layout_runtime._restore_preview_two_row_layout_state
    _preview_separator_y_at_x = z3_plate_layout_runtime._preview_separator_y_at_x
    _get_preview_row_for_bbox = z3_plate_layout_runtime._get_preview_row_for_bbox
    _constrain_preview_char_bbox_to_layout_separator = z3_plate_layout_runtime._constrain_preview_char_bbox_to_layout_separator
    _find_preview_layout_separator_handle_hit = z3_plate_layout_runtime._find_preview_layout_separator_handle_hit
    _set_preview_layout_separator_handle_y_from_canvas = z3_plate_layout_runtime._set_preview_layout_separator_handle_y_from_canvas
    _move_preview_layout_separator_from_canvas_delta = z3_plate_layout_runtime._move_preview_layout_separator_from_canvas_delta
    _build_preview_layout_separator_drag_preview = z3_plate_layout_runtime._build_preview_layout_separator_drag_preview
    _clamp_preview_layout_separator_to_existing_rows = z3_plate_layout_runtime._clamp_preview_layout_separator_to_existing_rows
    _apply_preview_layout_separator_constraints_to_chars = z3_plate_layout_runtime._apply_preview_layout_separator_constraints_to_chars
    _preview_layout_separator_conflicts_with_chars = z3_plate_layout_runtime._preview_layout_separator_conflicts_with_chars
    _annotate_preview_character_reading_positions = z3_plate_layout_runtime._annotate_preview_character_reading_positions
    _update_preview_plate_layout_metadata = z3_plate_layout_runtime._update_preview_plate_layout_metadata
    _get_preview_plate_layout_label = z3_plate_layout_runtime._get_preview_plate_layout_label
    _get_preview_plate_layout_dock_text = z3_plate_layout_runtime._get_preview_plate_layout_dock_text
    _get_preview_character_reading_position_label = z3_plate_layout_runtime._get_preview_character_reading_position_label
    _format_preview_layout_semantics = z3_plate_layout_runtime._format_preview_layout_semantics
    _cycle_preview_plate_layout_override = z3_plate_layout_runtime._cycle_preview_plate_layout_override
    _normalize_character_source_tag = z3_plate_layout_runtime._normalize_character_source_tag
    _character_record_uses_yolo_box_backend = staticmethod(z3_plate_layout_runtime._character_record_uses_yolo_box_backend)
    _character_record_yolo_box_backend_confidence = staticmethod(z3_plate_layout_runtime._character_record_yolo_box_backend_confidence)
    _fusion_details_yolo_box_backend_positions = staticmethod(z3_plate_layout_runtime._fusion_details_yolo_box_backend_positions)
    _normalize_character_source_kind = z3_plate_layout_runtime._normalize_character_source_kind
    _normalize_plate_source_bucket = z3_plate_layout_runtime._normalize_plate_source_bucket
    _normalize_plate_source_origin = z3_plate_layout_runtime._normalize_plate_source_origin

    def _empty_gold_source_counts(self):
        return empty_gold_source_counts(GOLD_SOURCE_BUCKETS)

    _get_plate_source_bucket = z3_plate_layout_runtime._get_plate_source_bucket
    _ensure_plate_source_metadata = z3_plate_layout_runtime._ensure_plate_source_metadata
    _build_character_source_tags = z3_plate_layout_runtime._build_character_source_tags
    _serialize_character_records = z3_plate_layout_runtime._serialize_character_records
    _get_character_source_tag = z3_plate_layout_runtime._get_character_source_tag
    _get_character_box_source_tag = z3_plate_layout_runtime._get_character_box_source_tag
    _get_character_sign_source_tag = z3_plate_layout_runtime._get_character_sign_source_tag
    _compose_character_source_tag = z3_plate_layout_runtime._compose_character_source_tag
    _get_character_source_kind = z3_plate_layout_runtime._get_character_source_kind

    _get_plate_listbox_source_flags = get_plate_listbox_source_flags

    _get_plate_listbox_layout_flag = z3_plate_layout_runtime._get_plate_listbox_layout_flag
    _get_plate_layout_count_label = z3_plate_layout_runtime._get_plate_layout_count_label
    _build_plate_layout_export_metadata = z3_plate_layout_runtime._build_plate_layout_export_metadata
    _is_uncertain_plate_layout = z3_plate_layout_runtime._is_uncertain_plate_layout
    _count_uncertain_layout_plate_entries = z3_plate_layout_runtime._count_uncertain_layout_plate_entries
    _confirm_export_with_uncertain_layouts = z3_plate_layout_runtime._confirm_export_with_uncertain_layouts
    _count_character_sources = z3_plate_layout_runtime._count_character_sources

    def _reset_preview_box_mode_selection(self):
        try:
            self.preview_box_mode_var.set(PREVIEW_BOX_MODE_LABELS["AUTO"])
        except Exception:
            pass
        self._on_preview_box_mode_change()

        active_pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
        self._preview_char_selected_index = None
        self._preview_char_hover_index = None
        self._preview_char_hover_label_index = None
        self._preview_char_label_active_index = None
        if active_pid and active_pid in self.preview_metadata:
            active_data = self.preview_metadata.get(active_pid)
            if isinstance(active_data, dict):
                self._refresh_preview_live_metadata_ui(
                    status_message=None,
                    status_tone="muted",
                    render_preview=True,
                )

        self._persist_preview_metadata(success_message=None, refresh_list=False, sync_access=False)
        try:
            self._refresh_preview_listbox_row(active_pid)
        except Exception:
            pass
        try:
            self._schedule_preview_info_refresh(delay_ms=120)
        except Exception:
            pass

    def _is_manual_character_record(self, rec, data=None) -> bool:
        try:
            box_source = str(self._get_character_box_source_tag(rec, data=data) or "").strip().lower()
            sign_source = str(self._get_character_sign_source_tag(rec, data=data) or "").strip().lower()
        except Exception:
            box_source = ""
            sign_source = ""
        if box_source == "manual_box" or sign_source == "manual_sign":
            return True

        try:
            source_kind = str(self._get_character_source_kind(rec, data=data) or "").strip().lower()
        except Exception:
            source_kind = ""
        if source_kind in {"local_manual", "cvat_manual"}:
            return True

        try:
            source_tag = str(self._get_character_source_tag(rec, data=data) or "").strip().lower()
        except Exception:
            source_tag = ""
        if source_tag == "manual":
            return True

        try:
            method_name = str(rec.get("method", "") if isinstance(rec, dict) else getattr(rec, "method", "")).strip().lower()
        except Exception:
            method_name = ""
        return method_name in {"manual", "cvat_manual"}

    _character_record_overlap_score = character_record_overlap_score
    _character_record_collides_with_manual = character_record_collides_with_manual
    _clone_base_record_with_candidate_bbox = clone_base_record_with_candidate_bbox

    def _is_existing_plate_perfect(self, existing_chars, data=None) -> bool:
        ordered_existing = self._sort_character_records_by_x(list(existing_chars or []))
        if not ordered_existing:
            return False
        return str(self._derive_preview_status_from_data(data, ordered_existing) or "").strip().lower() == "perfect"

    def _preserve_existing_perfect_plate_during_detection(
        self,
        existing_chars,
        yolo_detections,
        data=None,
        plate_image=None,
        enable_perfect_refiner: bool = True,
        perfect_refiner_continuity_guard: bool = True,
    ):
        ordered_existing = self._sort_character_records_by_x(list(existing_chars or []))
        if not ordered_existing:
            return [], None

        rebuilt, backend_details = self._apply_yolo_box_backend(
            ordered_existing,
            yolo_detections,
            preserve_base_record_metadata=True,
            protect_manual_records=True,
            base_data=data,
            plate_image=plate_image,
            enable_perfect_refiner=bool(enable_perfect_refiner),
            perfect_refiner_continuity_guard=bool(perfect_refiner_continuity_guard),
        )
        final_chars = rebuilt or ordered_existing

        fusion_strategy = str((data or {}).get("fusion_strategy", "") or "").strip() if isinstance(data, dict) else ""
        fusion_details = (data or {}).get("fusion_details", {}) if isinstance(data, dict) else {}
        serialized = self._serialize_character_records(
            final_chars,
            fusion_strategy=fusion_strategy,
            fusion_details=fusion_details if isinstance(fusion_details, dict) else None,
            data=data,
        )
        return serialized, backend_details

    def _merge_detected_characters_preserving_manual(self, existing_chars, detected_chars, data=None) -> tuple[list[dict], dict]:
        return merge_detected_characters_preserving_manual(
            self,
            existing_chars,
            detected_chars,
            data=data,
        )

    def _get_perfect_strategy_bucket(self, data: dict) -> str:
        if not isinstance(data, dict):
            return "other_perfect"

        raw_strategy = str(data.get("fusion_strategy", "") or "").strip().lower()
        details = data.get("fusion_details", {})
        auto_strategy = ""
        if isinstance(details, dict):
            auto_strategy = str(details.get("auto_strategy", "") or "").strip().lower()
        if raw_strategy == "manual_correction" and auto_strategy:
            raw_strategy = auto_strategy
        if raw_strategy in ("ocr_exact", "ocr_only", "ocr_fallback"):
            return "ocr_exact"
        if raw_strategy in ("yolo_exact", "yolo_only", "yolo_fallback"):
            return "yolo_exact"
        if raw_strategy == "ocr_yolo_rescue":
            return "ocr_yolo_rescue"
        if raw_strategy in (
            "yolo_box_ocr",
            "yolo_box_ocr_exact",
            "yolo_box_ocr_fallback",
            "yolo_box_ocr_no_gt",
            "both_combined",
            "both_combined_no_gt",
        ):
            return "yolo_box_ocr"

        return "other_perfect"

    def _empty_perfect_strategy_counts(self):
        return empty_perfect_strategy_counts(PERFECT_STRATEGY_BUCKETS)

    def _format_perfect_strategy_counts(self, counts: dict) -> str:
        return format_perfect_strategy_counts(counts, PERFECT_STRATEGY_BUCKETS)

    def _empty_plate_layout_counts(self):
        return empty_plate_layout_counts()

    def _increment_plate_layout_counts(self, counts: dict, data: dict | None, *, amount: int = 1):
        increment_plate_layout_counts(self, counts, data, amount=amount)

    def _format_plate_layout_counts(self, counts: dict | None, *, prefix: str = "Układ") -> str:
        return format_plate_layout_counts(counts, prefix=prefix)

    def _get_selected_gold_export_strategy_buckets(self):
        return get_selected_gold_export_strategy_buckets(self)

    def _format_selected_gold_export_strategy_labels(self) -> str:
        return format_selected_gold_export_strategy_labels(self, PERFECT_STRATEGY_BUCKETS)

    def _get_selected_gold_export_source_buckets(self):
        return get_selected_gold_export_source_buckets(self)

    def _format_selected_gold_export_source_labels(self) -> str:
        return format_selected_gold_export_source_labels(self, GOLD_SOURCE_BUCKETS)

    def _count_exportable_characters_in_data(self, data: dict) -> int:
        return count_exportable_characters_in_data(self, data)

    def _count_exportable_perfect_plates_in_metadata(self, metadata_map) -> int:
        return count_exportable_perfect_plates_in_metadata(self, metadata_map)

    def _build_contextual_gold_export_counts(self, *, selected_strategies=None, selected_sources=None):
        return build_contextual_gold_export_counts(
            self,
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )

    def _count_statuses_in_metadata_mapping(self, metadata_map):
        return count_statuses_in_metadata_mapping(self, metadata_map)

    def _build_merged_gold_export_counts(self, *, selected_strategies=None, selected_sources=None):
        return build_merged_gold_export_counts(
            self,
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )

    def _build_campaign_aware_gold_export_counts(self, *, selected_strategies=None, selected_sources=None):
        return build_campaign_aware_gold_export_counts(
            self,
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )

    def _get_step3_yolo_export_readiness_snapshot(self, *, selected_strategies=None, selected_sources=None) -> dict:
        return get_step3_yolo_export_readiness_snapshot(
            self,
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )

    def _get_gold_export_meta_candidates(self) -> list[Path]:
        return get_gold_export_meta_candidates(self)

    _char_record_bbox = z3_character_geometry._char_record_bbox
    _char_record_width = z3_character_geometry._char_record_width
    _char_record_height = z3_character_geometry._char_record_height
    _char_record_center_y = z3_character_geometry._char_record_center_y
    _build_char_record_geometry_stats = z3_character_geometry._build_char_record_geometry_stats
    _char_record_confidence = z3_character_geometry._char_record_confidence
    _clone_character_detection = z3_character_geometry._clone_character_detection

    _levenshtein_distance = staticmethod(levenshtein_distance)
    _best_text_distance = best_text_distance
    _fit_detection_count_to_truths = fit_detection_count_to_truths
    _apply_final_truth_count_guard = apply_final_truth_count_guard
    _pick_best_true_text = pick_best_true_text
    _get_text_mismatch_positions = staticmethod(get_text_mismatch_positions)
    _find_best_yolo_rescue_index = find_best_yolo_rescue_index
    _find_best_yolo_box_backend_index = find_best_yolo_box_backend_index
    _apply_yolo_box_backend = apply_yolo_box_backend
    _repair_ocr_with_yolo_boxes = repair_ocr_with_yolo_boxes

    def _resolve_canonical_detections(
        self,
        method,
        combined_detections,
        ocr_detections,
        yolo_detections,
        true_texts,
        hybrid_rescue_max_chars: int = 999,
        prefer_yolo_box_positions: bool = False,
        yolo_box_backend_detections=None,
        plate_image=None,
    ):
        return resolve_canonical_detections(
            self,
            method,
            combined_detections,
            ocr_detections,
            yolo_detections,
            true_texts,
            hybrid_rescue_max_chars=hybrid_rescue_max_chars,
            prefer_yolo_box_positions=prefer_yolo_box_positions,
            yolo_box_backend_detections=yolo_box_backend_detections,
            plate_image=plate_image,
        )

    def _get_preview_box_mode_key(self) -> str:
        raw_value = str((getattr(self, "preview_box_mode_var", None).get() if hasattr(self, "preview_box_mode_var") else "AUTO") or "AUTO").strip()
        if not raw_value:
            return "AUTO"

        direct_key = raw_value.upper()
        if direct_key in PREVIEW_BOX_MODE_LABELS:
            return direct_key

        return PREVIEW_BOX_MODE_BY_LABEL.get(raw_value, "AUTO")

    def _normalize_detection_method_key(self, raw_value=None) -> str:
        return normalize_detection_method_key(raw_value, DETECTION_METHOD_LABELS, DETECTION_METHOD_KEY_BY_LABEL)

    def _get_detection_method_key(self) -> str:
        return get_detection_method_key(self, DETECTION_METHOD_LABELS, DETECTION_METHOD_KEY_BY_LABEL)

    _get_hybrid_rescue_max_chars = get_hybrid_rescue_max_chars
    _get_yolo_rescue_enabled = get_yolo_rescue_enabled
    _use_hybrid_yolo_box_backend = use_hybrid_yolo_box_backend
    _get_hybrid_detection_status_text = get_hybrid_detection_status_text
    _get_yolo_box_ocr_status_text = staticmethod(get_yolo_box_ocr_status_text)

    def _get_detection_pipeline_blocks(self, method_key: str | None = None) -> list[str]:
        return get_detection_pipeline_blocks(
            self,
            method_key,
            DETECTION_METHOD_LABELS,
            DETECTION_METHOD_KEY_BY_LABEL,
        )

    def _compile_detection_pipeline_blocks(self, blocks=None) -> dict:
        return compile_detection_pipeline_blocks(blocks, DETECTION_METHOD_CARD_META)

    _apply_detection_pipeline_blocks = apply_detection_pipeline_blocks

    def _get_detection_workflow_text(self, method_key: str | None = None) -> str:
        return get_detection_workflow_text(
            self,
            method_key,
            DETECTION_METHOD_LABELS,
            DETECTION_METHOD_KEY_BY_LABEL,
        )

    _refresh_detection_workflow_info_label = refresh_detection_workflow_info_label
    _get_detection_active_model_status = get_detection_active_model_status
    _refresh_detection_active_model_label = refresh_detection_active_model_label
    _refresh_detection_refiner_guard_label = refresh_detection_refiner_guard_label

    def _get_preview_box_mode_label(self, key=None) -> str:
        resolved_key = (key or self._get_preview_box_mode_key() or "AUTO").upper().strip()
        return PREVIEW_BOX_MODE_LABELS.get(resolved_key, PREVIEW_BOX_MODE_LABELS["AUTO"])

    @staticmethod
    def _normalize_preview_sort_mode_key(mode_key: str | None) -> str:
        return normalize_preview_sort_mode_key(mode_key, PREVIEW_SORT_LABELS)

    @staticmethod
    def _normalize_preview_layout_filter_key(filter_key: str | None) -> str:
        return normalize_preview_layout_filter_key(filter_key, PREVIEW_LAYOUT_FILTER_LABELS)

    def _get_preview_sort_mode_key(self) -> str:
        return get_preview_sort_mode_key(self, PREVIEW_SORT_LABELS)

    def _get_preview_layout_filter_key(self) -> str:
        return get_preview_layout_filter_key(self, PREVIEW_LAYOUT_FILTER_LABELS)

    def _plate_matches_preview_layout_filter(self, data: dict | None) -> bool:
        return plate_matches_preview_layout_filter(self, data, PREVIEW_LAYOUT_FILTER_LABELS)

    _get_preview_sort_source_count = get_preview_sort_source_count
    _get_preview_sort_priority = get_preview_sort_priority
    _get_sorted_preview_plate_ids = get_sorted_preview_plate_ids
    _set_preview_sort_hover = set_preview_sort_hover
    _set_preview_layout_filter_hover = set_preview_layout_filter_hover

    def _get_readable_text_color(self, background: str, preferred: str = None) -> str:
        return get_readable_text_color(background, preferred)

    def _apply_preview_sort_bar_style(self):
        apply_preview_sort_bar_style(
            self,
            sort_labels=PREVIEW_SORT_LABELS,
            sort_color_keys=PREVIEW_SORT_COLOR_KEYS,
        )

    def _on_preview_sort_mode_change(self, mode_key: str = None):
        normalized_key = self._normalize_preview_sort_mode_key(
            mode_key or self._get_preview_sort_mode_key()
        )

        try:
            if self.preview_sort_mode_var.get() != normalized_key:
                self.preview_sort_mode_var.set(normalized_key)
        except Exception:
            pass

        try:
            self._save_local_setting("char_preview_sort_mode", normalized_key)
        except Exception:
            pass

        self._apply_preview_sort_bar_style()

        if hasattr(self, "plates_listbox"):
            self._rebuild_preview_listbox(preserve_selection=True)

    def _on_preview_layout_filter_change(self, filter_key: str = None):
        normalized_key = self._normalize_preview_layout_filter_key(
            filter_key or self._get_preview_layout_filter_key()
        )

        try:
            if self.preview_layout_filter_var.get() != normalized_key:
                self.preview_layout_filter_var.set(normalized_key)
        except Exception:
            pass

        try:
            self._save_local_setting("char_preview_layout_filter", normalized_key)
        except Exception:
            pass

        self._apply_preview_sort_bar_style()

        if hasattr(self, "plates_listbox"):
            self._rebuild_preview_listbox(preserve_selection=True)

    _get_preview_box_variants = get_preview_box_variants
    _get_preview_box_records = get_preview_box_records

    def _on_preview_box_mode_var_write(self, *_args):
        self._apply_preview_mode_radio_style()

    def _on_preview_dir_var_write(self, *_args):
        try:
            self._save_local_setting("char_preview_dir", str(self.preview_dir_var.get() or "").strip())
        except Exception:
            pass
        try:
            if (
                getattr(self, "_step3_linear_mode", False)
                and CAMPAIGN.get_active_project_name()
                and not bool(getattr(self, "_persist_step3_preview_dir_clear_in_progress", False))
            ):
                preview_dir_value = str(self.preview_dir_var.get() or "").strip()
                CAMPAIGN.set_step3_preview_dir(preview_dir_value)
                self._sync_campaign_step3_preview_artifact_registry(preview_dir_value)
        except Exception:
            pass

        refresh = getattr(self, "_refresh_preview_bound_action_states", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

        refresh_source = getattr(self, "_refresh_preview_source_panel", None)
        if callable(refresh_source):
            try:
                refresh_source()
            except Exception:
                pass

        refresh_extract = getattr(self, "_refresh_extract_workflow_ui", None)
        if callable(refresh_extract):
            try:
                refresh_extract()
            except Exception:
                pass

        refresh_pz3_source = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh_pz3_source):
            try:
                refresh_pz3_source()
            except Exception:
                pass

    def _set_preview_dir_runtime_value(self, value: str = "", *, persist_registry: bool = True) -> None:
        if persist_registry:
            self.preview_dir_var.set(str(value or "").strip())
            return

        try:
            self._persist_step3_preview_dir_clear_in_progress = True
            self.preview_dir_var.set(str(value or "").strip())
        finally:
            self._persist_step3_preview_dir_clear_in_progress = False

    def _on_yolo_option_var_write(self, *_args):
        on_yolo_option_var_write(self, *_args)

    def _on_gold_export_filter_var_write(self, *_args):
        self._apply_gold_export_filter_check_style()
        self._refresh_gold_export_filter_labels()
        self._refresh_gold_export_source_labels()
        self._refresh_gold_export_scope_label()
        refresh = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _on_gold_export_source_var_write(self, *_args):
        self._apply_gold_export_source_check_style()
        self._refresh_gold_export_source_labels()
        self._refresh_gold_export_filter_labels()
        self._refresh_gold_export_scope_label()
        refresh = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _on_gold_export_split_var_write(self, *_args):
        self._on_gold_export_split_change()

    _draw_selection_indicator = draw_selection_indicator
    _set_selection_row_hover = set_selection_row_hover
    _refresh_selection_row = refresh_selection_row
    _apply_preview_mode_radio_style = apply_preview_mode_radio_style

    def _toggle_preview_mode_overlay(self, event=None):
        self._preview_mode_overlay_expanded = not bool(getattr(self, "_preview_mode_overlay_expanded", False))
        self._refresh_preview_mode_overlay_visibility()
        self._focus_preview_canvas()
        return "break"

    _redraw_preview_mode_eye_icon = redraw_preview_mode_eye_icon

    def _refresh_preview_mode_overlay_visibility(self):
        refresh_preview_mode_overlay_visibility_with_labels(self, PREVIEW_BOX_MODE_LABELS)

    def _apply_preview_mode_overlay_style(self):
        apply_preview_mode_overlay_style(self, PREVIEW_BOX_MODE_LABELS)

    _apply_yolo_option_check_style = apply_yolo_option_check_style
    _apply_gold_export_filter_check_style = apply_gold_export_filter_check_style
    _apply_gold_export_source_check_style = apply_gold_export_source_check_style
    _apply_gold_export_split_check_style = apply_gold_export_split_check_style
    _apply_plates_legend_style = apply_plates_legend_style
    _apply_preview_source_actions_style = apply_preview_source_actions_style

    def _apply_preview_info_stats_style(self):
        apply_preview_info_stats_style(self, SlimProgressBar)

    _apply_preview_typing_overlay_style = apply_preview_typing_overlay_style
    _apply_preview_surface_style = apply_preview_surface_style
    _sync_preview_intro_wraplength = sync_preview_intro_wraplength
    _apply_preview_focus_prompt_style = apply_preview_focus_prompt_style
    _apply_detection_advanced_section_style = apply_detection_advanced_section_style

    def _register_detection_param_row(self, frame, *, labels=None, containers=None, scales=None):
        if frame is None:
            return
        row_entry = {
            "frame": frame,
            "labels": [widget for widget in (labels or []) if widget is not None],
            "containers": [widget for widget in (containers or []) if widget is not None],
            "scales": [widget for widget in (scales or []) if widget is not None],
        }
        rows = getattr(self, "_detection_param_rows", None)
        if not isinstance(rows, list):
            self._detection_param_rows = []
            rows = self._detection_param_rows
        rows.append(row_entry)

    def _register_detection_section_widgets(self, *, labels=None, dividers=None):
        entry = {
            "labels": [widget for widget in (labels or []) if widget is not None],
            "dividers": [widget for widget in (dividers or []) if widget is not None],
        }
        rows = getattr(self, "_detection_section_widgets", None)
        if not isinstance(rows, list):
            self._detection_section_widgets = []
            rows = self._detection_section_widgets
        rows.append(entry)

    _apply_detection_param_row_style = apply_detection_param_row_style
    _has_configured_yolo_detection_model = has_configured_yolo_detection_model

    def _set_detection_method_key(self, method_key: str, *, save: bool = True):
        set_detection_method_key(self, method_key, DETECTION_METHOD_LABELS, save=save)

    def _bind_detect_mode_card(self, widget, mode_key: str):
        bind_detect_mode_card(self, widget, mode_key)

    def _handle_detect_mode_selection(self, mode_key: str):
        handle_detect_mode_selection(self, mode_key, DETECTION_METHOD_LABELS)

    def _refresh_detect_mode_cards(self):
        refresh_detect_mode_cards(self, DETECTION_METHOD_CARD_META)

    def _get_detection_pipeline_block_meta(self, block_key: str) -> dict:
        return get_detection_pipeline_block_meta(block_key, DETECTION_PIPELINE_BLOCK_LIBRARY)

    _get_detection_pipeline_block_style = get_detection_pipeline_block_style
    _get_detection_pipeline_builder_blocks = get_detection_pipeline_builder_blocks
    _get_saved_detection_pipeline_blocks = get_saved_detection_pipeline_blocks
    _save_detection_pipeline_blocks = save_detection_pipeline_blocks
    _set_detection_pipeline_builder_blocks = set_detection_pipeline_builder_blocks
    _select_detection_pipeline_builder_block = select_detection_pipeline_builder_block
    _set_detection_pipeline_builder_preset = set_detection_pipeline_builder_preset
    _append_detection_pipeline_builder_block = append_detection_pipeline_builder_block
    _move_detection_pipeline_builder_selected_block = move_detection_pipeline_builder_selected_block
    _remove_detection_pipeline_builder_selected_block = remove_detection_pipeline_builder_selected_block
    _clear_detection_pipeline_builder = clear_detection_pipeline_builder
    _close_detection_pipeline_builder = close_detection_pipeline_builder
    _commit_detection_pipeline_builder = commit_detection_pipeline_builder
    _refresh_detection_pipeline_builder = refresh_detection_pipeline_builder
    _refresh_detection_pipeline_model_row = refresh_detection_pipeline_model_row
    _pick_detection_pipeline_yolo_model = pick_detection_pipeline_yolo_model
    _show_detection_pipeline_model_details = show_detection_pipeline_model_details
    _open_detection_pipeline_advanced_modal = open_detection_pipeline_advanced_modal
    _draw_detection_pipeline_builder_canvas = draw_detection_pipeline_builder_canvas
    _refresh_detection_pipeline_builder_property_panel = refresh_detection_pipeline_builder_property_panel

    def _open_detection_pipeline_builder(self, initial_method: str | None = None):
        open_detection_pipeline_builder(
            self,
            initial_method=initial_method,
            detection_pipeline_preset_meta=DETECTION_PIPELINE_PRESET_META,
        )

    _refresh_yolo_model_picker_state = refresh_yolo_model_picker_state
    _toggle_detection_advanced_panel = toggle_detection_advanced_panel

    def _get_detection_advanced_toggle_label(self) -> str:
        return get_detection_advanced_toggle_label(self, ADVANCED_PANEL_VISIBLE_ICON, ADVANCED_PANEL_HIDDEN_ICON)

    def _refresh_detection_advanced_sections(self):
        refresh_detection_advanced_sections(self, ADVANCED_PANEL_VISIBLE_ICON, ADVANCED_PANEL_HIDDEN_ICON)

    _get_preview_source_visual_style = get_preview_source_visual_style
    _get_preview_badge_component_style = get_preview_badge_component_style
    _get_preview_source_component_legend_items = get_preview_source_component_legend_items

    _get_preview_source_badge_layers = get_preview_source_badge_layers
    _measure_preview_badge_stack = measure_preview_badge_stack
    _draw_preview_badge_stack = draw_preview_badge_stack
    _draw_preview_text_badge = draw_preview_text_badge
    _fit_preview_text_to_width = fit_preview_text_to_width
    _draw_preview_fixed_text_badge = draw_preview_fixed_text_badge
    _measure_preview_text_badge = measure_preview_text_badge
    _get_preview_badge_layout_metrics = get_preview_badge_layout_metrics
    _estimate_preview_badge_layout = estimate_preview_badge_layout
    _draw_preview_source_legend = draw_preview_source_legend
    _preview_record_has_symbol = preview_record_has_symbol
    _measure_preview_overlay_text_width = measure_preview_overlay_text_width
    _measure_preview_overlay_font_height = measure_preview_overlay_font_height
    _estimate_preview_source_legend_height = estimate_preview_source_legend_height
    _estimate_preview_source_legend_width = estimate_preview_source_legend_width
    _format_preview_source_counts_line = format_preview_source_counts_line

    def _plan_preview_canvas_info_overlay_layout(
        self,
        canvas_width: int,
        source_line: str,
        status_text: str,
        *,
        data: dict | None = None,
        box_chars=None,
    ):
        layout = plan_preview_canvas_info_overlay_layout(
            self,
            canvas_width,
            source_line,
            status_text,
            data=data,
            box_chars=box_chars,
        )
        if not bool(getattr(self, "_preview_fullscreen_active", False)):
            layout["bar_height"] = max(float(layout.get("bar_height", 0.0) or 0.0), 68.0)
        return layout

    _place_preview_record_overlay = place_preview_record_overlay
    _draw_preview_canvas_info_overlay = draw_preview_canvas_info_overlay
    _get_yolo_runtime_settings = get_yolo_runtime_settings
    _characters_to_text = characters_to_text
    _characters_to_display_rows = characters_to_display_rows
    _characters_to_display_text = characters_to_display_text
    _get_plate_listbox_ordinal = get_plate_listbox_ordinal

    def _format_preview_record_source_label(self, data: dict | None = None) -> tuple[str, str]:
        return format_preview_record_source_label(self, data, GOLD_SOURCE_LABELS)

    def _update_preview_record_source_label(self, data: dict | None = None):
        text, tone = self._format_preview_record_source_label(data)
        self._set_preview_record_source_info(text, tone)

    _refresh_preview_import_focus_ui = refresh_preview_import_focus_ui
    _set_preview_import_focus = set_preview_import_focus
    _toggle_preview_import_focus = toggle_preview_import_focus
    _format_plate_listbox_label = format_plate_listbox_label
    _get_plate_row_foreground = get_plate_row_foreground
    _apply_plate_listbox_row_style = apply_plate_listbox_row_style
    _update_preview_info_label = update_preview_info_label
    _update_preview_box_info_label = update_preview_box_info_label

    def _on_preview_box_mode_change(self, event=None):
        try:
            self._save_local_setting("char_preview_box_mode", self._get_preview_box_mode_key())
        except Exception:
            pass
        if self._get_preview_box_mode_key() != "FINAL" and (
            bool(getattr(self, "_preview_char_edit_mode", False))
            or bool(getattr(self, "_preview_char_add_mode", False))
        ):
            self._preview_char_edit_mode = False
            self._preview_char_add_mode = False
            self._preview_char_add_modifier_down = False
            self._preview_char_add_click_armed = False
            self._preview_char_drag_state = None
            self._preview_char_add_state = None
        try:
            self._refresh_preview_mode_overlay_visibility()
        except Exception:
            pass
        self._refresh_preview_editor_toolbar()
        self._on_preview_select(None)

    def _rebuild_preview_listbox(self, preserve_selection: bool = True, schedule_render: bool = True):
        rebuild_preview_listbox(
            self,
            preserve_selection=preserve_selection,
            schedule_render=schedule_render,
        )

    def _clear_project_bound_session_values(self, clear_ui: bool = False):
        clear_project_bound_session_values(self, clear_ui=clear_ui)


    def _apply_preview_metadata_update(
        self,
        new_meta: dict,
        preserve_selection: bool = True,
        render_selection: bool = True,
        *,
        recalculate_statuses: bool = True,
    ):
        mark_preview_metadata_changed(self)
        apply_started = time.perf_counter()
        recalc_ms = 0.0
        order_ms = 0.0
        list_ms = 0.0

        phase_started = time.perf_counter()
        if recalculate_statuses:
            try:
                new_meta = self._backfill_preview_manual_layout_from_related_runs(new_meta)
            except Exception:
                pass
            self.preview_metadata = self._recalculate_preview_statuses_in_metadata(new_meta)
        else:
            self.preview_metadata = new_meta if isinstance(new_meta, dict) else {}
        recalc_ms = (time.perf_counter() - phase_started) * 1000.0

        phase_started = time.perf_counter()
        current_order = [pid for pid in self._preview_base_plate_ids if pid in self.preview_metadata]
        appended = [pid for pid in self.preview_metadata.keys() if pid not in current_order]
        self._preview_base_plate_ids = current_order + appended
        self.preview_plate_ids = list(self._preview_base_plate_ids)
        order_ms = (time.perf_counter() - phase_started) * 1000.0

        phase_started = time.perf_counter()
        self._rebuild_preview_listbox(
            preserve_selection=preserve_selection,
            schedule_render=render_selection,
        )
        list_ms = (time.perf_counter() - phase_started) * 1000.0

        total_ms = (time.perf_counter() - apply_started) * 1000.0
        if total_ms >= 180.0:
            try:
                logger.info(
                    "[PZ2 apply metadata] total=%.1fms recalc=%.1fms order=%.1fms list=%.1fms count=%s recalc_enabled=%s",
                    total_ms,
                    recalc_ms,
                    order_ms,
                    list_ms,
                    len(self.preview_metadata) if isinstance(self.preview_metadata, dict) else 0,
                    bool(recalculate_statuses),
                )
            except Exception:
                pass



    def _set_detection_process_log_visibility(self, visible: bool):
        set_detection_process_log_visibility(self, visible)

    def _toggle_detection_process_log(self):
        toggle_detection_process_log(self)


    def clear_campaign_context(self):
        clear_step3_campaign_context(self, NAV_BUTTON_WIDTH)

    def apply_theme(self):
        apply_character_annotation_theme(self, SlimProgressBar)

    def _sync_detect_right_scrollregion(self, event=None):
        sync_detect_right_scrollregion(self, event)

    def _sync_extract_left_scrollregion(self, event=None):
        sync_extract_left_scrollregion(self, event)

    _sync_extract_left_canvas_width = sync_extract_left_canvas_width
    _sync_detect_right_canvas_width = sync_detect_right_canvas_width
    _refresh_detect_right_adaptive_wraps = refresh_detect_right_adaptive_wraps
    _schedule_detect_right_adaptive_wrap_refresh = schedule_detect_right_adaptive_wrap_refresh
    _sync_cvat_export_scrollregion = sync_cvat_export_scrollregion
    _sync_cvat_export_canvas_width = sync_cvat_export_canvas_width
    _widget_contains_point = staticmethod(widget_contains_point)
    _mousewheel_units = mousewheel_units
    _on_plates_listbox_mousewheel = on_plates_listbox_mousewheel

    def _detect_right_canvas_overflows(self) -> bool:
        return canvas_overflows(getattr(self, "detect_right_canvas", None))

    _on_detect_right_global_mousewheel = on_detect_right_global_mousewheel

    def _cvat_export_canvas_overflows(self) -> bool:
        return canvas_overflows(getattr(self, "cvat_export_canvas", None))

    def _extract_left_canvas_overflows(self) -> bool:
        return canvas_overflows(getattr(self, "extract_left_canvas", None))

    _on_cvat_export_global_mousewheel = on_cvat_export_global_mousewheel
    _suppress_selection_hover_during_scroll = suppress_selection_hover_during_scroll
    _restore_scroll_canvas_focus = staticmethod(restore_scroll_canvas_focus)
    _redirect_child_mousewheel_to_canvas = redirect_child_mousewheel_to_canvas

    _redirect_pz3_child_mousewheel_to_canvas = redirect_pz3_child_mousewheel_to_canvas
    _bind_scroll_canvas_children = bind_scroll_canvas_children
    _panel_style_name = staticmethod(panel_style_name)
    _set_themed_label_state = set_themed_label_state
    _mark_inline_status_contrast_boost = mark_inline_status_contrast_boost
    _set_inline_status_label_state = set_inline_status_label_state
    _set_extraction_status = set_extraction_status
    _refresh_continue_source_summary = refresh_continue_source_summary
    _format_extract_source_datetime = staticmethod(format_extract_source_datetime)
    _format_extract_source_folder_name = staticmethod(format_extract_source_folder_name)
    _build_continue_extract_summary_rows = build_continue_extract_summary_rows
    _set_extract_start_summary_rows = set_extract_start_summary_rows
    _format_extract_start_path = staticmethod(format_extract_start_path)

    def _refresh_extract_start_card_style(self, *, lightweight: bool = False):
        refresh_extract_start_card_style(self, SlimProgressBar, lightweight=lightweight)

    _extract_source_manifest_path = extract_source_manifest_path
    _get_extract_preview_manifest_state = get_extract_preview_manifest_state
    _current_extract_source_payload = current_extract_source_payload
    _preview_matches_current_extract_source = preview_matches_current_extract_source
    _find_latest_extract_preview_run_dir = find_latest_extract_preview_run_dir
    _write_extract_source_manifest = write_extract_source_manifest
    _is_extract_preview_ready = is_extract_preview_ready
    _is_extract_preview_ready_fast = is_extract_preview_ready_fast
    _get_extract_preview_ready_count = get_extract_preview_ready_count
    _refresh_extract_action_state = refresh_extract_action_state
    _refresh_extract_continue_action_visibility = refresh_extract_continue_action_visibility

    def _cancel_continue_extract_flow(self):
        try:
            reset_extract_source_inputs(self)
        except Exception:
            pass
        try:
            self.extract_entry_mode_var.set("")
        except Exception:
            pass
        self._extract_workflow_step = "entry"
        try:
            self._persist_step3_extract_state()
        except Exception:
            pass
        self._refresh_extract_workflow_ui()

    _refresh_extract_start_summary = refresh_extract_start_summary

    _show_campaign_detect_splash = show_campaign_detect_splash
    _hide_campaign_detect_splash = hide_campaign_detect_splash
    _set_source_binding_status = set_source_binding_status
    _set_test_status = set_test_status
    _get_detection_method_status_label = get_detection_method_status_label
    _compose_detection_method_status = compose_detection_method_status
    _get_detection_pipeline_short_label = get_detection_pipeline_short_label
    _load_last_detection_summary = load_last_detection_summary
    _save_last_detection_summary = save_last_detection_summary
    _format_last_detection_summary_line = format_last_detection_summary_line
    _format_plate_last_detection_line = format_plate_last_detection_line
    _refresh_last_detection_status_label = refresh_last_detection_status_label
    _show_last_detection_details = show_last_detection_details
    _set_test_progress_counter = set_test_progress_counter
    _set_test_progress_detail = set_test_progress_detail
    _update_detection_progress_ui = update_detection_progress_ui
    _set_preview_processing_overlay = set_preview_processing_overlay
    _update_preview_processing_overlay_progress = update_preview_processing_overlay_progress
    _set_preview_info = set_preview_info
    _set_preview_counts_info = set_preview_counts_info

    def _format_preview_layout_summary(self, counts: dict | None = None) -> str:
        return format_preview_layout_summary(counts)

    _set_preview_layout_summary_info = set_preview_layout_summary_info
    _build_plates_list_legend_counts = build_plates_list_legend_counts
    _set_plates_legend_info = set_plates_legend_info
    _set_preview_repair_progress_status = set_preview_repair_progress_status
    _allocate_split_counts = staticmethod(allocate_split_counts)
    _build_split_entries = staticmethod(build_split_entries)
    _get_preview_repair_progress_snapshot = get_preview_repair_progress_snapshot
    _update_preview_repair_progress_ui = update_preview_repair_progress_ui
    _set_preview_fusion_info = set_preview_fusion_info
    _set_preview_box_info = set_preview_box_info

    @staticmethod
    def _set_grid_visibility(widget, visible: bool):
        if widget is None:
            return
        try:
            if visible:
                if not str(widget.winfo_manager()):
                    widget.grid()
            elif str(widget.winfo_manager()):
                widget.grid_remove()
        except Exception:
            pass

    def _refresh_step3_mode_specific_ui(self):
        refresh_step3_mode_specific_ui(self)

    def _reset_pz3_runtime_ui(self, collapse_cards: bool = True):
        try:
            if collapse_cards:
                in_campaign = bool(getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
                self._pz3_selected_path = "dataset" if in_campaign else ""
            self._pz3_cvat_expanded = False
        except Exception:
            pass

        try:
            self._set_console_text(getattr(self, "export_console", None), "Oczekuję na akcję...")
        except Exception:
            pass

        try:
            self._set_console_text(getattr(self, "import_console", None), "Oczekuję na plik XML...")
        except Exception:
            pass

        try:
            self._set_step3_finish_hint("")
        except Exception:
            pass

        try:
            refresh_pz3_cards = getattr(self, "_refresh_pz3_cards_ui", None)
            if callable(refresh_pz3_cards):
                refresh_pz3_cards()
        except Exception:
            pass

    def _set_preview_record_source_info(self, text: str, tone: str = "muted"):
        set_preview_record_source_info(self, text, tone=tone)

    def _set_gold_export_scope_info(self, text: str, tone: str = "muted"):
        set_gold_export_scope_info(self, text, tone=tone)

    def _refresh_gold_export_scope_label(self):
        refresh_gold_export_scope_label(self)
        return

        selected_strategies = self._get_selected_gold_export_strategy_buckets()
        selected_sources = self._get_selected_gold_export_source_buckets()
        if not selected_strategies:
            self._set_gold_export_scope_info("Do eksportu gold packa nie wybrano żadnej strategii.", "warning")
            return
        if not selected_sources:
            self._set_gold_export_scope_info("Do eksportu gold packa nie wybrano żadnego źródła.", "warning")
            return

        selected_labels = self._format_selected_gold_export_strategy_labels()
        selected_source_labels = self._format_selected_gold_export_source_labels()
        contextual = self._build_campaign_aware_gold_export_counts(
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )
        selected_count = int(contextual.get("selected_plate_count", 0) or 0)
        selected_chars = int(contextual.get("selected_char_count", 0) or 0)
        layout_summary = self._format_plate_layout_counts(
            contextual.get("selected_layout_counts", {}),
            prefix="układ",
        )

        self._set_gold_export_scope_info(
            f"Do gold packa: strategie={selected_labels} | źródła={selected_source_labels} | "
            f"perfect={selected_count} | znaki={selected_chars} | {layout_summary} | "
            f"{self._format_gold_export_split_summary()}",
            "muted"
        )

    def _format_gold_export_split_summary(self) -> str:
        return format_gold_export_split_summary(self)

    def _get_gold_export_split_percentages(self):
        return get_gold_export_split_percentages(self)

    def _update_gold_export_split_labels(self):
        update_gold_export_split_labels(self)

    def _on_gold_export_split_change(self):
        try:
            self._save_local_setting("char_gold_export_split", bool(self.gold_export_split_var.get()))
            self._save_local_setting("char_gold_export_train_pct", float(self.gold_export_train_pct_var.get()))
            self._save_local_setting("char_gold_export_val_pct", float(self.gold_export_val_pct_var.get()))
        except Exception:
            pass

        self._apply_gold_export_split_check_style()
        self._update_gold_export_split_labels()
        self._refresh_gold_export_scope_label()
        try:
            self._update_preview_repair_progress_ui()
        except Exception:
            pass
        refresh = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _on_pz3_dataset_source_var_write(self, *_args):
        handle_pz3_dataset_source_var_write(self, *_args)

    def _on_pz3_existing_dataset_var_write(self, *_args):
        try:
            self._save_local_setting("char_pz3_existing_dataset", str(self.pz3_existing_dataset_var.get() or "").strip())
        except Exception:
            pass

        refresh = getattr(self, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _refresh_gold_export_filter_labels(self):
        refresh_gold_export_filter_labels(self)

    def _refresh_gold_export_source_labels(self):
        refresh_gold_export_source_labels(self)

    def _on_gold_export_filter_change(self):
        try:
            self._save_local_setting("char_gold_include_ocr_exact", bool(self.gold_include_ocr_exact_var.get()))
            self._save_local_setting("char_gold_include_yolo_exact", bool(self.gold_include_yolo_exact_var.get()))
            self._save_local_setting("char_gold_include_ocr_yolo_rescue", bool(self.gold_include_ocr_yolo_rescue_var.get()))
            self._save_local_setting("char_gold_include_yolo_box_ocr", bool(self.gold_include_yolo_box_ocr_var.get()))
            self._save_local_setting("char_gold_include_other_perfect", bool(self.gold_include_other_perfect_var.get()))
        except Exception:
            pass

        self._apply_gold_export_filter_check_style()
        self._refresh_gold_export_filter_labels()
        self._refresh_gold_export_source_labels()
        self._refresh_gold_export_scope_label()

    def _on_gold_export_source_change(self):
        try:
            self._save_local_setting("char_gold_include_source_auto", bool(self.gold_include_source_auto_var.get()))
            self._save_local_setting("char_gold_include_source_local_manual", bool(self.gold_include_source_local_manual_var.get()))
        except Exception:
            pass

        self._apply_gold_export_source_check_style()
        self._refresh_gold_export_source_labels()
        self._refresh_gold_export_filter_labels()
        self._refresh_gold_export_scope_label()

    def _set_winner_name(self, text: str, tone: str = "neutral"):
        set_winner_name(self, text, tone)

    def _set_winner_acc(self, text: str, tone: str = "muted"):
        set_winner_acc(self, text, tone)

    def _remember_ocr_ranking_results(self, results_table, total_imgs: int, source_desc: str, ranking_mode: str):
        remember_ocr_ranking_results(self, results_table, total_imgs, source_desc, ranking_mode)

    def _compose_ocr_ranking_source_text(
        self,
        source_desc: str = "",
        total_imgs: int = 0,
        ranking_mode: str = "",
        preset_count: int = 0,
    ) -> str:
        return compose_ocr_ranking_source_text(source_desc, total_imgs, ranking_mode, preset_count)

    def _set_ocr_ranking_modal_label(self, attr_name: str, text: str, tone: str = "muted", emphasis: bool = False):
        set_ocr_ranking_modal_label(self, attr_name, text, tone=tone, emphasis=emphasis)

    def _set_ocr_ranking_modal_status(self, text: str, tone: str = "neutral"):
        set_ocr_ranking_modal_status(self, text, tone=tone)

    def _set_ocr_ranking_modal_source(self, text: str, tone: str = "muted"):
        set_ocr_ranking_modal_source(self, text, tone=tone)

    def _set_ocr_ranking_modal_running(self, running: bool):
        set_ocr_ranking_modal_running(self, running)

    def _update_ocr_ranking_modal_progress(
        self,
        processed_presets: int,
        total_presets: int,
        *,
        sample_count: int = 0,
        demo_mode: bool = False,
        finished: bool = False,
    ):
        update_ocr_ranking_modal_progress(
            self,
            processed_presets,
            total_presets,
            sample_count=sample_count,
            demo_mode=demo_mode,
            finished=finished,
        )

    def _populate_ocr_ranking_results_host(self, host, *, panel_alt: str, border: str, fg: str, palette: dict):
        populate_ocr_ranking_results_host(self, host, panel_alt=panel_alt, border=border, fg=fg, palette=palette)

    def _refresh_ocr_ranking_modal_results(self):
        refresh_ocr_ranking_modal_results(self)

    def _open_ocr_ranking_modal(self):
        open_ocr_ranking_modal(self, SlimProgressBar)

    def _open_ocr_summary_modal(self):
        open_ocr_summary_modal(self)

    def _get_campaign_char_model_path(self) -> str:
        return get_campaign_char_model_path(self)

    def _get_campaign_detection_yolo_model_path(self) -> str:
        return get_campaign_detection_yolo_model_path(self)

    def _get_effective_yolo_model_path(self) -> str:
        return get_effective_yolo_model_path(self)

    def _sync_yolo_model_binding(self):
        sync_yolo_model_binding(self)

    def _infer_yolo_arch_from_model_path(self, model_path: str):
        return infer_yolo_arch_from_model_path(model_path)
    # Extraction-source delegates are bound after class creation.

    def _get_char_manual_pool_dir(self) -> Path | None:
        """
        Katalog na ręcznie poprawione zestawy znaków importowane z CVAT.
        """
        campaign_chars_dir = getattr(self, "_campaign_chars_dir", None)
        if not campaign_chars_dir:
            return None

        root = Path(campaign_chars_dir)
        root.mkdir(parents=True, exist_ok=True)

        pool_dir = root / "manual_char_pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        return pool_dir

    def _get_char_merged_pool_dir(self) -> Path | None:
        """
        Katalog na scaloną pulę znaków: gold + manual.
        """
        campaign_datasets_dir = getattr(self, "_campaign_datasets_dir", None)
        if not campaign_datasets_dir:
            return None

        root = Path(campaign_datasets_dir)
        root.mkdir(parents=True, exist_ok=True)

        pool_dir = root / "char_merged_pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        return pool_dir

    def _is_valid_step3_training_dataset_dir(self, dataset_dir: Path | None) -> bool:
        """
        Sprawdza, czy katalog wygląda jak realny dataset treningowy znaków
        gotowy do użycia w etapie 4.
        """
        if dataset_dir is None:
            return False

        try:
            if not dataset_dir.exists() or not dataset_dir.is_dir():
                return False

            data_yaml = dataset_dir / "data.yaml"
            images_dir = dataset_dir / "images"
            labels_dir = dataset_dir / "labels"

            if not data_yaml.exists():
                return False

            if not images_dir.exists() or not images_dir.is_dir():
                return False

            if not labels_dir.exists() or not labels_dir.is_dir():
                return False

            return True
        except Exception:
            return False

    def _get_preferred_step3_training_dataset_dir(self) -> Path | None:
        """
        Zwraca najlepszy dostępny dataset znaków dla finiszu kroku 3.

        Priorytet:
        1. char_merged_pool
        2. najnowszy poprawny dataset z katalogu projektowych datasetów
        """
        merged_dir = self._get_char_merged_pool_dir()
        if self._is_valid_step3_training_dataset_dir(merged_dir):
            return merged_dir

        campaign_datasets_dir = getattr(self, "_campaign_datasets_dir", None)
        if not campaign_datasets_dir:
            return None

        ds_root = Path(campaign_datasets_dir)
        if not ds_root.exists() or not ds_root.is_dir():
            return None

        try:
            candidates = [
                p for p in ds_root.iterdir()
                if p.is_dir() and self._is_valid_step3_training_dataset_dir(p)
            ]
            if not candidates:
                return None

            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
        except Exception:
            return None

    def _inspect_pz3_dataset_source_dir(self, dataset_dir: Path | None) -> dict:
        return inspect_pz3_dataset_source_dir(dataset_dir)

    def _pick_pz3_existing_dataset_dir(self):
        initial = str((self.pz3_existing_dataset_var.get() or "").strip())
        if not initial:
            preferred = self._get_preferred_step3_training_dataset_dir()
            if preferred is not None:
                initial = str(preferred)
        if not initial:
            initial = str(self._get_step3_datasets_root_dir())

        selected = filedialog.askdirectory(
            initialdir=str(initial),
            title="Wybierz gotowy dataset YOLO znaków"
        )
        if selected:
            self.pz3_existing_dataset_var.set(str(selected))
            try:
                self._force_save_all()
            except Exception:
                pass

    def _use_preferred_pz3_dataset_dir(self):
        preferred = self._get_preferred_step3_training_dataset_dir()
        if preferred is None:
            messagebox.showwarning(
                "Brak datasetu",
                "Nie znaleziono jeszcze gotowego datasetu znaków w katalogach projektu."
            )
            return

        self.pz3_existing_dataset_var.set(str(preferred))
        try:
            self._force_save_all()
        except Exception:
            pass

    def _get_project_review_dir(self) -> Path | None:
        """
        Projektowy katalog review dla eksportów CVAT z kroku 3.
        """
        campaign_chars_dir = getattr(self, "_campaign_chars_dir", None)
        if not campaign_chars_dir:
            return None

        root = Path(campaign_chars_dir)
        root.mkdir(parents=True, exist_ok=True)

        review_dir = root / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        return review_dir
    def _store_current_preview_in_manual_char_pool(self, metadata: dict):
        """
        Zapisuje aktualnie poprawiony preview run do manual_char_pool,
        aby mogła później zasilić trening modelu znaków.
        """
        manual_pool_dir = self._get_char_manual_pool_dir()
        if manual_pool_dir is None:
            raise RuntimeError("Brak katalogu manual_char_pool dla aktywnego projektu.")

        preview_dir = Path((self.preview_dir_var.get() or "").strip())
        if not preview_dir.exists() or not preview_dir.is_dir():
            raise RuntimeError("Brak poprawnego katalogu preview do zapisania w manual_char_pool.")

        preview_name = preview_dir.name if preview_dir.name else "manual_import"
        target_dir = manual_pool_dir / preview_name
        target_images_dir = target_dir / "images"

        target_dir.mkdir(parents=True, exist_ok=True)
        target_images_dir.mkdir(parents=True, exist_ok=True)

        # zapis metadata
        target_meta = target_dir / "metadata.json"
        self._atomic_write_json(target_meta, metadata)

        # kopiowanie obrazów źródłowych preview
        source_images_dir = preview_dir / "images"
        if source_images_dir.exists() and source_images_dir.is_dir():
            for img_file in source_images_dir.iterdir():
                if not img_file.is_file():
                    continue
                dst = target_images_dir / img_file.name
                shutil.copy2(img_file, dst)

        return target_dir

    # PZ3 status delegates are bound after class creation.

    def _resolve_guidance_button(self, attr_name: str):
        if not attr_name:
            return None

        candidates = [attr_name]
        if attr_name.endswith("_frame"):
            candidates.append(attr_name[:-6])

        for candidate in candidates:
            widget = getattr(self, candidate, None)
            if isinstance(widget, ttk.Button):
                return widget

        return None

    def _update_step3_finish_button_state(self):
        self._refresh_pz3_status_panel_ui()

    def _update_preview_path_lock(self):
        try:
            self._refresh_preview_source_panel()
        except Exception:
            pass

    _refresh_preview_source_panel = z3_navigation_runtime._refresh_preview_source_panel
    _set_button_state = set_button_state
    _set_subtab_state = set_subtab_state
    _get_subtab_state = get_subtab_state
    _select_subtab = z3_navigation_runtime._select_subtab

    # --- STEP3 NOTEBOOK PERSIST ---
    _on_main_nb_tab_changed = z3_navigation_runtime._on_main_nb_tab_changed
    _campaign_step3_pz2_base_ready = z3_navigation_runtime.campaign_step3_pz2_base_ready
    _campaign_step3_pz2_current_contract_ready = z3_navigation_runtime.campaign_step3_pz2_current_contract_ready
    _campaign_step3_requires_current_pz2_for_pz3 = z3_navigation_runtime.campaign_step3_requires_current_pz2_for_pz3
    _can_open_step3_dataset_from_current_context = z3_navigation_runtime.campaign_step3_can_open_pz3_from_current_context
    get_free_mode_assistant_context = get_step3_free_mode_assistant_context
    _refresh_campaign_step3_navigation_visibility = refresh_campaign_step3_navigation_visibility
    _sync_step3_nav_buttons = sync_step3_nav_buttons
    enter_campaign_step3_mode = enter_campaign_step3_mode
    open_campaign_step3_entry = open_campaign_step3_entry
    _auto_progress_campaign_step3_entry = auto_progress_campaign_step3_entry
    reset_subtab_flow = reset_step3_subtab_flow
    ensure_free_mode_context_ready = ensure_step3_free_mode_context
    unlock_detection_subtab = unlock_detection_subtab
    unlock_dataset_subtab = unlock_dataset_subtab

    def go_to_substep_2(self, *, force: bool = False):
        if self._step3_linear_mode and CAMPAIGN.get_active_project_name():
            go_to_substep_2_campaign(self, force=force)
            return
        if self._step3_linear_mode:
            self.ensure_free_mode_context_ready()

        go_to_substep_2_free_mode(self)

    def go_to_substep_3(self, *, force: bool = False):
        if self._step3_linear_mode and CAMPAIGN.get_active_project_name():
            go_to_substep_3_campaign(self, force=force)
            return
        if self._step3_linear_mode:
            self.ensure_free_mode_context_ready()
        go_to_substep_3_free_mode(self)
        return

    def back_to_substep_1(self):
        """
        Cofnięcie do 1 blokuje 2 i 3 tylko w trybie kampanii.
        """
        if getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name():
            return
        if self._step3_linear_mode:
            self.ensure_free_mode_context_ready()

        back_to_substep_1_free_mode(self)


    def back_to_substep_2(self):
        """
        Powrót z PZ3 do PZ2 zostawia użytkownika w tym samym etapie E3,
        ale otwiera z powrotem aktywny zestaw wykrywania znaków.
        """
        if self._step3_linear_mode and CAMPAIGN.get_active_project_name():
            back_to_substep_2_campaign(self)
            return
        if self._step3_linear_mode:
            self.ensure_free_mode_context_ready()

        back_to_substep_2_free_mode(self)

    _persist_step3_progress = persist_step3_progress
    _return_to_t05_work_after_step3_pz1 = return_to_t05_work_after_step3_pz1
    _return_to_wizard_from_step3_pz2 = return_to_wizard_from_step3_pz2
    restore_campaign_step3_mode = restore_campaign_step3_mode
    can_restore_step3_substep = z3_navigation_runtime.can_restore_step3_substep

    def _atomic_write_json(self, path: Path, data: dict):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            if path.name == "metadata.json":
                f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        if path.name == "metadata.json":
            # A full save/import replaces the source fragments used by autosave.
            self._preview_metadata_file_cache = None

    def _backup_json_before_import(self, path: Path) -> Path | None:
        if path is None or not path.exists() or not path.is_file():
            return None
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_name(f"{path.stem}.before_cvat_import_{timestamp}{path.suffix}")
            shutil.copy2(path, backup_path)
            return backup_path
        except Exception as e:
            logger.debug(f"Nie udało się utworzyć backupu metadata przed importem CVAT: {e}")
            return None

    _load_local_session = load_local_session
    _save_local_setting = save_local_setting
    _prune_legacy_yolo_arch_session_keys = prune_legacy_yolo_arch_session_keys
    _force_save_all = force_save_all

    def _on_app_close(self, event=None):
        event_widget = getattr(event, "widget", None)
        if event is None or str(event_widget) == str(self.app.root):
            self._force_save_all()
            try:
                mark_step3_work_interrupted_on_app_close(self)
            except Exception as exc:
                logger.debug(f"Nie udało się oznaczyć przerwanej pracy Z3 przy zamykaniu: {exc}")

    _update_yolo_visibility = update_yolo_visibility
    _auto_device_label = staticmethod(auto_device_label)
    _get_available_devices = get_available_devices
    _normalize_selected_device = normalize_selected_device
    _get_effective_detection_device_choice = get_effective_detection_device_choice
    _refresh_device_options = refresh_device_options
    _device_to_ultralytics = device_to_ultralytics
    _device_to_ocr = device_to_ocr
    _update_device_hint = update_device_hint
    apply_global_yolo_device_choice = apply_global_yolo_device_choice
    _set_button_emphasis = set_button_emphasis

    # =========================================================
    # Resolvers
    # =========================================================

    _ensure_yolo_model_checkpoint = ensure_yolo_model_checkpoint

    # =========================================================
    # Pickers
    # =========================================================

    _pick_xml_file = pick_xml_file
    _pick_images_dir = pick_images_dir
    _pick_yolo_model = pick_yolo_model

    # =========================================================
    # Logging helper
    # =========================================================

    _log = log_message

    def _build_step3_local_success_message(self, title: str, target_path, next_hint: str) -> str:
        return build_step3_local_success_message(title, target_path, next_hint)

    def _hide_pz3_status_summary_section(self) -> None:
        hide_pz3_status_summary_section(self)

    def _show_pz3_operation_summary_modal(self, title: str, message: str, tone: str = "info") -> None:
        show_pz3_operation_summary_modal(self, title, message, tone=tone)

    def _remember_pz3_operation_message(self, console_widget, message: str, tone: str = "info") -> None:
        remember_pz3_operation_message(self, console_widget, message, tone=tone)

    def _get_true_texts_from_filename(self, filename: str) -> list:
        return get_true_texts_from_filename(filename)

    def _get_current_prep_params(self):
        return get_current_prep_params(self)

    def _get_best_preset(self):
        return get_best_preset(self)

    # Extraction workflow delegates are bound after class creation.

    # =========================================================
    # UI build
    # =========================================================

    _create_widgets = create_step3_widgets
    _clear_lazy_subtab_placeholder = clear_lazy_subtab_placeholder
    _build_lazy_subtab_placeholder = build_lazy_subtab_placeholder
    _ensure_step3_subtab_built = ensure_step3_subtab_built
    _ensure_detect_tab_built = ensure_detect_tab_built
    _ensure_dataset_tab_built = ensure_dataset_tab_built
    _paint_extract_entry_cards_first = paint_extract_entry_cards_first

    # =========================================================
    # TAB 1: Extraction
    # =========================================================

    def _build_extraction_tab(self, parent):
        return build_extraction_tab(self, parent, SlimProgressBar, NAV_BUTTON_WIDTH)

    _show_extract_completed_modal = show_extract_completed_modal
    _show_campaign_extract_below_minimum_modal = show_campaign_extract_below_minimum_modal
    _show_campaign_extract_failure_modal = show_campaign_extract_failure_modal
    _run_extraction = run_extraction

    # =========================================================
    # TAB 2: Detection + Preview
    # =========================================================

    def _build_detection_tab(self, parent):
        return build_detection_tab(
            self,
            parent,
            SlimProgressBar,
            NAV_BUTTON_WIDTH,
            DETECTION_METHOD_CARD_META,
            PREVIEW_BOX_MODE_OPTIONS,
            PREVIEW_LAYOUT_FILTER_OPTIONS,
            PREVIEW_SORT_OPTIONS,
        )

    _build_pz2_detection_guard_counts = build_pz2_detection_guard_counts
    _prompt_pz2_detection_guard_options = prompt_pz2_detection_guard_options
    _run_detection_stage = run_detection_stage
    _undo_last_detection_result = undo_last_detection_result
    _confirm_last_detection_result = confirm_last_detection_result
    _refresh_detection_review_controls = refresh_detection_review_controls
    _update_winner_label = update_winner_label

    # =========================================================
    # Preview load + render
    # =========================================================
    _load_preview_data = load_preview_data


    def _refresh_listbox_rows_from_metadata(self):
        """Pełne odświeżenie listy i preview na podstawie aktualnego self.preview_metadata."""
        if not hasattr(self, "plates_listbox"):
            return

        try:
            current_map = list(getattr(self, "_listbox_pid_by_index", []) or [])
            metadata_keys = set(str(key) for key in getattr(self, "preview_metadata", {}).keys())
            current_keys = set(str(key) for key in current_map)
            active_pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
            list_structure_unchanged = bool(current_map) and metadata_keys == current_keys
        except Exception:
            current_map = []
            active_pid = ""
            list_structure_unchanged = False

        if list_structure_unchanged and active_pid in current_map:
            try:
                self._refresh_preview_listbox_row(active_pid)
            except Exception as e:
                logger.debug(f"Nie udalo sie lekko odswiezyc wiersza listy tablic: {e}")
            try:
                self._schedule_preview_info_refresh(delay_ms=120)
            except Exception:
                pass
            return

        try:
            self._apply_preview_metadata_update(self.preview_metadata, preserve_selection=True)
        except Exception as e:
            logger.debug(f"Nie udało się odświeżyć listy tablic: {e}")

        try:
            if self._listbox_pid_by_index and self.plates_listbox.curselection():
                self.frame.after_idle(lambda: self._on_preview_select(None))
        except Exception as e:
            logger.debug(f"Nie udało się odświeżyć preview po _refresh_listbox_rows_from_metadata: {e}")

    _on_preview_select = on_preview_select

    # =========================================================
    # Fast test UI lock/unlock
    # =========================================================

    def _lock_ui_for_testing(self, owner: str, label: str) -> bool:
        if hasattr(self.app, "try_begin_exclusive_operation"):
            ok, busy_message = self.app.try_begin_exclusive_operation(owner, label)
            if not ok:
                messagebox.showinfo("Proces w toku", busy_message)
                return False
        self._active_test_operation_owner = owner
        self.is_processing = True

        # główne akcje
        for attr_name in (
            "btn_run_detection",
            "btn_rank_presets",
            "btn_ocr_lab",
            "btn_undo_detection_result",
            "btn_confirm_detection_result",
        ):
            widget = getattr(self, attr_name, None)
            if widget is not None:
                try:
                    widget.config(state=tk.DISABLED)
                except Exception:
                    pass

        # na czas detekcji blokujemy też nawigację workflow
        for attr_name in ("btn_to_detect", "btn_to_dataset"):
            widget = getattr(self, attr_name, None)
            if widget is not None:
                try:
                    widget.config(state=tk.DISABLED)
                except Exception:
                    pass

        # lista tablic ma być zablokowana tylko na czas testu
        try:
            self.plates_listbox.config(state=tk.DISABLED)
        except Exception:
            pass

        # zdejmij podświetlenie głównej akcji na czas pracy
        try:
            self._set_button_emphasis("btn_run_detection_frame", False)
        except Exception:
            pass
        return True

    _unlock_ui_after_testing = unlock_ui_after_testing
    # =========================================================
    # FAST OCR TEST (stable)
    # =========================================================

    _run_fast_ocr_test = run_fast_ocr_test

    # =========================================================
    # TAB 3: CVAT + YOLO exports/imports
    # =========================================================

    def _build_cvat_tab(self, parent):
        return build_cvat_tab(
            self,
            parent,
            NAV_BUTTON_WIDTH,
            PERFECT_STRATEGY_LABELS,
            GOLD_SOURCE_LABELS,
        )

    _set_console_text = set_console_text
    _pick_file = pick_file
    _get_active_preview_context = get_active_preview_context
    _refresh_preview_bound_action_states = refresh_preview_bound_action_states
    _run_cvat_export = run_cvat_export

    def _run_yolo_gold_export(self):
        run_yolo_gold_export(
            self,
            char_class_alphabet=CHAR_CLASS_ALPHABET,
            gold_source_buckets=GOLD_SOURCE_BUCKETS,
            gold_source_labels=GOLD_SOURCE_LABELS,
        )

    _run_pz3_existing_dataset_split = run_pz3_existing_dataset_split
    _build_gold_export_plate_unique_key = staticmethod(build_gold_export_plate_unique_key)
    _collect_gold_export_plate_candidates = collect_gold_export_plate_candidates

    _normalize_classification_char_crop_bbox = staticmethod(normalize_classification_char_crop_bbox)

    def _run_char_classification_export(self):
        run_char_classification_export(
            self,
            char_class_alphabet=CHAR_CLASS_ALPHABET,
            gold_source_buckets=GOLD_SOURCE_BUCKETS,
            gold_source_labels=GOLD_SOURCE_LABELS,
        )

    _run_cvat_import = run_cvat_import

    # =========================================================
    # Ranking presetów OCR
    # =========================================================

    _run_preset_ranking = run_preset_ranking
    _open_filter_lab = open_filter_lab

bind_character_extract_delegates(CharacterAnnotationTab)
