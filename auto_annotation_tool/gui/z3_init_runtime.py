#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 CharacterAnnotationTab initialization extracted from tab_character_annotation.py."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

from ..config import CONFIG
from ..campaign_manager import CAMPAIGN
from ..icons import IconManager
from ..training.dataset_splitter import DatasetSplitter
from .z3_detection_pipeline_ui import compile_detection_pipeline_blocks, normalize_detection_pipeline_blocks
from .inertial_scroll import InertialScrollController

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

DETECTION_METHOD_OPTIONS = [
    ("OCR", "OCR: odczyt znaków"),
    ("YOLO", "YOLO: odczyt i ramki"),
    ("BOTH", "Hybryda: OCR odczyt + YOLO ramki"),
    ("YOLO_OCR", "Hybryda: YOLO boxy + OCR odczyt"),
]

DETECTION_METHOD_LABELS = {key: label for key, label in DETECTION_METHOD_OPTIONS}


def _ensure_ocr_presets_dir_with_legacy_copy() -> tuple[Path, Path]:
    primary = CONFIG.get_presets_dir("ocr")
    legacy = CONFIG.get_legacy_ocr_presets_dir()
    primary.mkdir(parents=True, exist_ok=True)

    if legacy.exists() and legacy.is_dir() and legacy.resolve() != primary.resolve():
        for source in legacy.glob("*.json"):
            target = primary / source.name
            if target.exists():
                continue
            try:
                target.write_bytes(source.read_bytes())
            except OSError:
                continue

    return primary, legacy


def __init__(self, parent, app):
    self.parent = parent
    self.app = app
    self.icon_manager = IconManager
    self.frame = ttk.Frame(parent)
    self._inertial_scroll = InertialScrollController(self.frame)
    self._startup_ui_ready = False

    # core state
    self.is_processing = False
    self._step3_linear_mode = False

    # preview/cache state
    self.preview_metadata = {}
    self._preview_metadata_revision = 0
    self.preview_plate_ids = []
    self._preview_base_plate_ids = []
    self._listbox_pid_by_index = []
    self._current_photo = None
    self._reloading_preview = False
    self.preview_box_mode_rows = []
    self.gold_export_filter_rows = []
    self.yolo_option_rows = []
    self._detection_param_rows = []
    self.preview_sort_buttons = {}
    self.preview_layout_filter_buttons = {}
    self._plates_listbox_scroll_units = 0
    self._plates_listbox_scroll_after_id = None
    self._preview_sort_hover_key = None
    self._preview_layout_filter_hover_key = None
    self._preview_active_pid = None
    self._preview_zoom_level = 1.0
    self._preview_zoom_min = 0.72
    self._preview_zoom_max = 2.4
    self._preview_safe_zoom_max = 2.4
    self._preview_zoom_step = 1.16
    self._preview_zoom_target = 1.0
    self._preview_zoom_velocity = 0.0
    self._preview_zoom_anim_after_id = None
    self._preview_zoom_last_ts = None
    self._preview_zoom_last_input_ts = None
    self._preview_zoom_max_tail_s = 1.0
    self._preview_zoom_anchor = None
    self._preview_zoom_frame_ms = 12
    self._preview_zoom_stiffness = 120.0
    self._preview_zoom_damping = 58.0
    self._preview_pan_x = 0.0
    self._preview_pan_y = 0.0
    self._preview_render_state = {}
    self._preview_badge_offsets = {}
    self._preview_badge_runtime = {}
    self._preview_badge_drag_state = None
    self._preview_pan_drag_state = None
    self._preview_selected_badge_key = None
    self._preview_vertical_split_ready = False
    self._preview_mode_overlay_expanded = False
    self._preview_mode_overlay_position = None
    self._preview_mode_eye_drag_state = None
    self._preview_char_edit_mode = False
    self._preview_char_add_mode = False
    self._preview_char_add_modifier_down = False
    self._preview_char_add_click_armed = False
    self._preview_alt_modifier_down = False
    self._preview_char_label_mode = False
    self._preview_char_selected_index = None
    self._preview_char_drag_state = None
    self._preview_char_drag_window_bindings = []
    self._preview_last_char_edit_interaction_ts = 0.0
    self._preview_char_edit_trace_seq = 0
    self._preview_latency_trace_seq = 0
    self._preview_pending_select_latency_probe = None
    self._preview_post_release_latency_probe = None
    self._preview_metadata_save_defer_logged = False
    self._preview_char_add_state = None
    self._preview_char_hover_index = None
    self._preview_char_hover_grip = None
    self._preview_char_hover_label_index = None
    self._preview_char_label_active_index = None
    self._preview_char_record_render_tags = {}
    self._preview_stabilized_render_after_id = None
    self._preview_history_undo = {}
    self._preview_history_redo = {}
    self._preview_history_replaying = False
    self._preview_history_limit = 30
    self._preview_import_focus_plate_ids = []
    self._preview_import_focus_batch_id = ""
    self._preview_import_focus_active = False
    self._preview_status_counts_snapshot = {
        "perfect": 0,
        "needs_fix": 0,
        "unknown": 0,
        "total": 0,
        "strategy_counts": {},
        "strategy_char_counts": {},
    }
    self._campaign_step3_reextract_seed_metadata = {}
    self._preview_fullscreen_active = False
    self._preview_fullscreen_restore_log_visible = False
    self._preview_fullscreen_restore_root_state = False
    self._preview_fullscreen_restore_window_state = "normal"
    self._preview_fullscreen_restore_geometry = ""
    self._preview_legend_font_cache = {}
    self._preview_legend_image_cache = {}
    self._preview_controls_legend_expanded = False
    self._preview_controls_legend_current_width = 0.0
    self._preview_controls_legend_current_height = 0.0
    self._preview_controls_legend_current_bounds = None
    self._preview_controls_legend_render_key = None
    self._preview_controls_legend_offset_x = 10.0
    self._preview_controls_legend_offset_y = 10.0
    self._preview_controls_legend_position_manual = False
    self._preview_controls_legend_last_collapsed_offset = None
    self._preview_controls_legend_grab_bbox = None
    self._preview_controls_legend_drag_state = None
    self._preview_controls_legend_click_state = None
    self._preview_fullscreen_toggle_rect = None
    self._preview_controls_legend_visible = True
    self._preview_operation_assistant_visible = True
    self._preview_overlay_dock_expanded = True
    self._preview_overlay_dock_render_key = None
    self._preview_overlay_dock_tool_rows = {}
    self.preview_edit_status_var = tk.StringVar(value="W PZ2 tutaj poprawisz boxy znaków.")


    # Cache kluczy dla przełączania runów i zmian plików.
    self._loaded_meta_path = None
    self._loaded_meta_mtime = None

    # Stan szybkiego testu OCR niezależny od procesu wycinania.
    self.fast_test_stop = threading.Event()
    self.fast_test_running = False
    self._project_reset_token = 0
    self._detection_log_visible = False
    self._detect_preview_autoload_after_id = None
    self._detect_mode_hover_key = None
    self._detect_mode_cards = {}
    self._detection_advanced_expanded = False
    self._detection_ocr_advanced_expanded = False
    self._detection_yolo_advanced_expanded = False
    self._detection_pipeline_modal = None
    self._detection_pipeline_canvas = None
    self._detection_pipeline_state = {"blocks": [], "selected_index": None}
    self._detection_pipeline_runtime = {}
    self._detection_pipeline_property_body = None
    self._detection_pipeline_status_var = None
    self._detection_pipeline_hint_var = None
    self._detection_pipeline_confirm_btn = None
    self._pz3_cvat_expanded = False
    self._pz3_selected_path = ""
    self._ocr_summary_modal = None
    self._ocr_ranking_modal = None
    self._ocr_ranking_last_results = []
    self._ocr_ranking_last_total = 0
    self._ocr_ranking_last_source_desc = ""
    self._ocr_ranking_last_mode = ""
    self._source_binding_after_id = None
    self._source_binding_sync_in_progress = False
    self._extract_entry_hover_mode = None
    self._extract_workflow_step = "entry"
    self._extract_entry_cards = {}
    self._extract_step_cards = []
    self._extract_last_source_binding_result = {"ok": False}
    self._pending_z2_source = {}
    self._pz3_dataset_splitter = DatasetSplitter()
    self._extract_workflow_refresh_after_id = None
    self._preview_dir_plate_count_cache = {}
    self._detect_tab_built = False
    self._dataset_tab_built = False

    # Presets live in Workspace/8_presets/<module>; old OCR presets are copied lazily.
    self.presets_dir, self.legacy_presets_dir = _ensure_ocr_presets_dir_with_legacy_copy()
    self.presets_search_dirs = CONFIG.get_preset_search_dirs("ocr")

    # local session
    self.session_file = Path.home() / ".auto_annotation_tool" / "char_tab_session.json"
    self.session_file.parent.mkdir(parents=True, exist_ok=True)
    self.local_session = self._load_local_session()

    def get_val(key, default):
        val = self.local_session.get(key, default)
        return val if val != "" else default

    saved_preview_box_mode = str(get_val("char_preview_box_mode", "AUTO") or "AUTO").strip()
    saved_preview_box_mode = PREVIEW_BOX_MODE_LABELS.get(saved_preview_box_mode.upper(), saved_preview_box_mode)
    if saved_preview_box_mode not in PREVIEW_BOX_MODE_BY_LABEL:
        saved_preview_box_mode = PREVIEW_BOX_MODE_LABELS["AUTO"]
    saved_preview_sort_mode = self._normalize_preview_sort_mode_key(
        get_val("char_preview_sort_mode", "DEFAULT")
    )
    saved_preview_layout_filter = self._normalize_preview_layout_filter_key(
        get_val("char_preview_layout_filter", "ALL")
    )
    saved_detection_method = self._normalize_detection_method_key(get_val("char_det_method", "OCR"))
    saved_pipeline_blocks = normalize_detection_pipeline_blocks(
        self.local_session.get("char_detection_pipeline_blocks", [])
    )
    saved_pipeline_compiled = compile_detection_pipeline_blocks(saved_pipeline_blocks)
    if bool(saved_pipeline_compiled.get("valid")):
        self._detection_pipeline_last_blocks = list(saved_pipeline_blocks)
        saved_detection_method = self._normalize_detection_method_key(
            saved_pipeline_compiled.get("method_key") or saved_detection_method
        )
    else:
        self._detection_pipeline_last_blocks = []
    try:
        saved_hybrid_rescue_raw = int(get_val("char_hybrid_rescue_max_chars", 1) or 1)
    except Exception:
        saved_hybrid_rescue_raw = 1
    saved_hybrid_rescue_max_chars = 1 if saved_hybrid_rescue_raw > 0 else 0
    if bool(saved_pipeline_compiled.get("valid")) and saved_detection_method == "BOTH":
        saved_hybrid_rescue_max_chars = 1 if bool(saved_pipeline_compiled.get("rescue_enabled")) else 0
    saved_hybrid_yolo_box_backend = bool(get_val("char_hybrid_yolo_box_backend", True))
    saved_detect_protect_manual = bool(get_val("char_detect_protect_manual", True))
    saved_detect_protect_perfect = bool(get_val("char_detect_protect_perfect", True))
    saved_detect_refine_perfect_yolo = bool(get_val("char_detect_refine_perfect_yolo", True))
    saved_detect_refiner_continuity_guard = bool(get_val("char_detect_refiner_continuity_guard", True))
    saved_gold_export_split = False
    saved_gold_export_train_pct = max(50.0, min(90.0, float(get_val("char_gold_export_train_pct", 80.0) or 80.0)))
    saved_gold_export_val_pct = max(5.0, min(45.0, float(get_val("char_gold_export_val_pct", 10.0) or 10.0)))
    saved_pz3_dataset_source_mode = str(get_val("char_pz3_dataset_source_mode", "perfect") or "perfect").strip().lower()
    if saved_pz3_dataset_source_mode not in {"perfect", "existing"}:
        saved_pz3_dataset_source_mode = "perfect"
    saved_extract_entry_mode = ""

    # vars
    self.extract_entry_mode_var = tk.StringVar(value=saved_extract_entry_mode)
    self.annotation_run_dir_var = tk.StringVar(value=get_val("char_annotation_run_dir", ""))
    self.detection_method_var = tk.StringVar(value=DETECTION_METHOD_LABELS.get(saved_detection_method, "OCR"))
    self.xml_path_var = tk.StringVar(value=get_val("char_xml_path", ""))
    self.images_dir_var = tk.StringVar(value=get_val("char_images_dir", ""))
    self.yolo_model_path_var = tk.StringVar(value=get_val("char_yolo_model", ""))
    initial_yolo_device = str(get_val("char_yolo_device", "auto") or "auto").strip() or "auto"
    try:
        app_device_getter = getattr(self.app, "get_global_yolo_device_choice", None)
        if callable(app_device_getter):
            initial_yolo_device = str(app_device_getter() or initial_yolo_device).strip() or initial_yolo_device
    except Exception:
        pass
    self.yolo_device_var = tk.StringVar(value=initial_yolo_device)
    saved_yolo_conf = float(get_val("char_yolo_conf", 0.25))
    saved_yolo_box_conf = float(get_val("char_yolo_box_conf", saved_yolo_conf))
    saved_yolo_symbol_conf = float(get_val("char_yolo_symbol_conf", saved_yolo_conf))
    self.yolo_box_conf_var = tk.DoubleVar(value=saved_yolo_box_conf)
    self.yolo_symbol_conf_var = tk.DoubleVar(value=saved_yolo_symbol_conf)
    self.yolo_conf_var = tk.DoubleVar(value=min(saved_yolo_box_conf, saved_yolo_symbol_conf))
    self.yolo_iou_var = tk.DoubleVar(value=float(get_val("char_yolo_iou", 0.45)))
    self.yolo_overlap_var = tk.DoubleVar(value=float(get_val("char_yolo_overlap", 0.70)))
    self.yolo_agnostic_nms_var = tk.BooleanVar(value=bool(get_val("char_yolo_agnostic_nms", False)))
    self.yolo_seq_center_y_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_center_y", 0.60)))
    self.yolo_seq_min_h_ratio_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_min_h_ratio", 0.55)))
    self.yolo_seq_max_h_ratio_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_max_h_ratio", 1.80)))
    self.yolo_seq_max_w_ratio_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_max_w_ratio", 2.60)))
    self.yolo_seq_soft_overlap_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_soft_overlap", 0.18)))
    self.yolo_seq_hard_overlap_var = tk.DoubleVar(value=float(get_val("char_yolo_seq_hard_overlap", 0.30)))
    self.hybrid_rescue_max_chars_var = tk.IntVar(value=saved_hybrid_rescue_max_chars)
    self.hybrid_yolo_box_backend_var = tk.BooleanVar(value=saved_hybrid_yolo_box_backend)
    self.detect_protect_manual_var = tk.BooleanVar(value=saved_detect_protect_manual)
    self.detect_protect_perfect_var = tk.BooleanVar(value=saved_detect_protect_perfect)
    self.detect_refine_perfect_yolo_var = tk.BooleanVar(value=saved_detect_refine_perfect_yolo)
    self.detect_refiner_continuity_guard_var = tk.BooleanVar(value=saved_detect_refiner_continuity_guard)
    self.preview_dir_var = tk.StringVar(value=get_val("char_preview_dir", ""))
    self.preview_box_mode_var = tk.StringVar(value=saved_preview_box_mode)
    self.preview_sort_mode_var = tk.StringVar(value=saved_preview_sort_mode)
    self.preview_layout_filter_var = tk.StringVar(value=saved_preview_layout_filter)

    self.ocr_conf_var = tk.DoubleVar(value=float(get_val("char_ocr_conf", 0.25)))
    self.ocr_min_height_ratio_var = tk.DoubleVar(
        value=max(0.20, min(1.00, float(get_val("char_ocr_min_height_ratio", 0.58) or 0.58)))
    )
    self.smart_export_var = tk.BooleanVar(value=get_val("char_smart_export", True))
    self.gold_include_ocr_exact_var = tk.BooleanVar(value=bool(get_val("char_gold_include_ocr_exact", True)))
    self.gold_include_yolo_exact_var = tk.BooleanVar(value=bool(get_val("char_gold_include_yolo_exact", True)))
    self.gold_include_ocr_yolo_rescue_var = tk.BooleanVar(value=bool(get_val("char_gold_include_ocr_yolo_rescue", True)))
    self.gold_include_yolo_box_ocr_var = tk.BooleanVar(value=bool(get_val("char_gold_include_yolo_box_ocr", True)))
    self.gold_include_other_perfect_var = tk.BooleanVar(value=bool(get_val("char_gold_include_other_perfect", True)))
    self.gold_include_source_auto_var = tk.BooleanVar(value=bool(get_val("char_gold_include_source_auto", True)))
    self.gold_include_source_local_manual_var = tk.BooleanVar(value=bool(get_val("char_gold_include_source_local_manual", True)))
    self.gold_include_source_cvat_manual_var = tk.BooleanVar(value=True)
    self.gold_export_split_var = tk.BooleanVar(value=saved_gold_export_split)
    self.gold_export_train_pct_var = tk.DoubleVar(value=saved_gold_export_train_pct)
    self.gold_export_val_pct_var = tk.DoubleVar(value=saved_gold_export_val_pct)
    self.pz3_dataset_source_mode_var = tk.StringVar(value=saved_pz3_dataset_source_mode)
    self.pz3_existing_dataset_var = tk.StringVar(value=get_val("char_pz3_existing_dataset", ""))

    # lab params
    self.prep_angle_var = tk.DoubleVar(value=float(get_val("char_prep_angle", 0.0)))
    self.prep_height_var = tk.IntVar(value=int(get_val("char_prep_height", 80)))
    self.prep_padding_var = tk.IntVar(value=int(get_val("char_prep_padding", 20)))
    self.prep_clip_var = tk.IntVar(value=int(get_val("char_prep_clip", 255)))
    self.prep_denoise_var = tk.IntVar(value=int(get_val("char_prep_denoise", 0)))
    self.prep_clahe_var = tk.DoubleVar(value=float(get_val("char_prep_clahe", 0.0)))
    self.prep_use_bin_var = tk.BooleanVar(value=get_val("char_prep_use_bin", True))
    self.prep_block_var = tk.IntVar(value=int(get_val("char_prep_block", 15)))
    self.prep_c_var = tk.IntVar(value=int(get_val("char_prep_c", 5)))
    self.prep_erode_var = tk.IntVar(value=int(get_val("char_prep_erode", 0)))
    self.do_clahe_var = tk.BooleanVar(value=get_val("char_do_clahe", True))
    self.interpolation_var = tk.StringVar(value=get_val("char_interpolation", "lanczos4"))
    self.preview_dir_var.trace_add("write", self._on_preview_dir_var_write)
    self.preview_box_mode_var.trace_add("write", self._on_preview_box_mode_var_write)
    self.yolo_box_conf_var.trace_add("write", self._on_yolo_option_var_write)
    self.yolo_symbol_conf_var.trace_add("write", self._on_yolo_option_var_write)
    self.yolo_agnostic_nms_var.trace_add("write", self._on_yolo_option_var_write)
    self.hybrid_yolo_box_backend_var.trace_add("write", self._on_yolo_option_var_write)
    for filter_var in (
        self.gold_include_ocr_exact_var,
        self.gold_include_yolo_exact_var,
        self.gold_include_ocr_yolo_rescue_var,
        self.gold_include_yolo_box_ocr_var,
        self.gold_include_other_perfect_var,
    ):
        filter_var.trace_add("write", self._on_gold_export_filter_var_write)
    for source_var in (
        self.gold_include_source_auto_var,
        self.gold_include_source_local_manual_var,
    ):
        source_var.trace_add("write", self._on_gold_export_source_var_write)
    self.gold_export_split_var.trace_add("write", self._on_gold_export_split_var_write)
    self.gold_export_train_pct_var.trace_add("write", self._on_gold_export_split_var_write)
    self.gold_export_val_pct_var.trace_add("write", self._on_gold_export_split_var_write)
    self.pz3_dataset_source_mode_var.trace_add("write", self._on_pz3_dataset_source_var_write)
    self.pz3_existing_dataset_var.trace_add("write", self._on_pz3_existing_dataset_var_write)
    try:
        self.ocr_min_height_ratio_var.trace_add(
            "write",
            lambda *_args: self._save_local_setting(
                "char_ocr_min_height_ratio",
                max(0.20, min(1.00, float(self.ocr_min_height_ratio_var.get()))),
            ),
        )
    except Exception:
        pass

    self._create_widgets()
    self._bind_source_path_watchers()

    self._update_step3_source_path_lock()
    if self._detect_tab_built:
        self._update_preview_path_lock()
    self.reset_subtab_flow()
    if (self.xml_path_var.get() or "").strip() or (self.images_dir_var.get() or "").strip():
        self._schedule_source_binding_refresh(delay_ms=0)

    if not CAMPAIGN.get_active_project_name():
        self._clear_project_bound_session_values(clear_ui=True)
        self.reset_subtab_flow()

    if self._detect_tab_built:
        self._update_yolo_visibility()
    self.app.root.bind("<Destroy>", self._on_app_close, add="+")
    self.frame.after(180, self._mark_startup_ui_ready)
