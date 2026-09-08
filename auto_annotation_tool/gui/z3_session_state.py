#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local session persistence helpers for Z3."""

import json

from ..campaign_manager import CAMPAIGN


def load_local_session(host):
    if host.session_file.exists():
        try:
            with open(host.session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_local_setting(host, key, value):
    host.local_session[key] = value
    try:
        with open(host.session_file, "w", encoding="utf-8") as f:
            json.dump(host.local_session, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def prune_legacy_yolo_arch_session_keys(host):
    legacy_keys = ("char_yolo_size", "char_yolo_version", "char_extract_entry_mode")
    removed = False
    for key in legacy_keys:
        if key in host.local_session:
            host.local_session.pop(key, None)
            removed = True

    if not removed:
        return

    try:
        with open(host.session_file, "w", encoding="utf-8") as f:
            json.dump(host.local_session, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def clear_project_bound_session_values(host, clear_ui: bool = False):
    project_bound_keys = (
        "char_annotation_run_dir",
        "char_xml_path",
        "char_images_dir",
        "char_preview_dir",
        "char_yolo_model",
        "char_pz3_dataset_source_mode",
        "char_pz3_existing_dataset",
    )

    for key in project_bound_keys:
        host.local_session.pop(key, None)

    try:
        with open(host.session_file, "w", encoding="utf-8") as f:
            json.dump(host.local_session, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

    if clear_ui:
        for attr_name in (
            "annotation_run_dir_var",
            "xml_path_var",
            "images_dir_var",
            "yolo_model_path_var",
            "pz3_existing_dataset_var",
        ):
            try:
                getattr(host, attr_name).set("")
            except Exception:
                pass
        try:
            host._set_preview_dir_runtime_value("", persist_registry=False)
        except Exception:
            pass
        try:
            host.pz3_dataset_source_mode_var.set("perfect")
        except Exception:
            pass


def force_save_all(host):
    try:
        host._flush_scheduled_preview_metadata_save()
        always_saved = [
            ("char_yolo_device", host.yolo_device_var),
            ("char_yolo_conf", host.yolo_conf_var),
            ("char_yolo_box_conf", host.yolo_box_conf_var),
            ("char_yolo_symbol_conf", host.yolo_symbol_conf_var),
            ("char_yolo_iou", host.yolo_iou_var),
            ("char_yolo_overlap", host.yolo_overlap_var),
            ("char_yolo_agnostic_nms", host.yolo_agnostic_nms_var),
            ("char_yolo_seq_center_y", host.yolo_seq_center_y_var),
            ("char_yolo_seq_min_h_ratio", host.yolo_seq_min_h_ratio_var),
            ("char_yolo_seq_max_h_ratio", host.yolo_seq_max_h_ratio_var),
            ("char_yolo_seq_max_w_ratio", host.yolo_seq_max_w_ratio_var),
            ("char_yolo_seq_soft_overlap", host.yolo_seq_soft_overlap_var),
            ("char_yolo_seq_hard_overlap", host.yolo_seq_hard_overlap_var),
            ("char_hybrid_rescue_max_chars", host.hybrid_rescue_max_chars_var),
            ("char_hybrid_yolo_box_backend", host.hybrid_yolo_box_backend_var),
            ("char_detect_protect_manual", host.detect_protect_manual_var),
            ("char_detect_protect_perfect", host.detect_protect_perfect_var),
            ("char_detect_refine_perfect_yolo", host.detect_refine_perfect_yolo_var),
            ("char_detect_refiner_continuity_guard", host.detect_refiner_continuity_guard_var),
            ("char_ocr_conf", host.ocr_conf_var),
            ("char_ocr_min_height_ratio", host.ocr_min_height_ratio_var),
            ("char_smart_export", host.smart_export_var),
            ("char_gold_include_ocr_exact", host.gold_include_ocr_exact_var),
            ("char_gold_include_yolo_exact", host.gold_include_yolo_exact_var),
            ("char_gold_include_ocr_yolo_rescue", host.gold_include_ocr_yolo_rescue_var),
            ("char_gold_include_yolo_box_ocr", host.gold_include_yolo_box_ocr_var),
            ("char_gold_include_other_perfect", host.gold_include_other_perfect_var),
            ("char_gold_include_source_auto", host.gold_include_source_auto_var),
            ("char_gold_include_source_local_manual", host.gold_include_source_local_manual_var),
            ("char_gold_include_source_cvat_manual", host.gold_include_source_cvat_manual_var),
            ("char_gold_export_split", host.gold_export_split_var),
            ("char_gold_export_train_pct", host.gold_export_train_pct_var),
            ("char_gold_export_val_pct", host.gold_export_val_pct_var),
            ("char_pz3_dataset_source_mode", host.pz3_dataset_source_mode_var),
            ("char_pz3_existing_dataset", host.pz3_existing_dataset_var),
            ("char_prep_angle", host.prep_angle_var),
            ("char_prep_height", host.prep_height_var),
            ("char_prep_padding", host.prep_padding_var),
            ("char_prep_clip", host.prep_clip_var),
            ("char_prep_denoise", host.prep_denoise_var),
            ("char_prep_clahe", host.prep_clahe_var),
            ("char_prep_use_bin", host.prep_use_bin_var),
            ("char_prep_block", host.prep_block_var),
            ("char_prep_c", host.prep_c_var),
            ("char_prep_erode", host.prep_erode_var),
            ("char_do_clahe", host.do_clahe_var),
            ("char_interpolation", host.interpolation_var),
        ]

        for key, var in always_saved:
            host._save_local_setting(key, var.get())

        host._save_local_setting("char_det_method", host._get_detection_method_key())
        try:
            pipeline_blocks = []
            try:
                pipeline_blocks = list(getattr(host, "_detection_pipeline_last_blocks", []) or [])
            except Exception:
                pipeline_blocks = []
            if not pipeline_blocks:
                try:
                    pipeline_blocks = host._get_saved_detection_pipeline_blocks()
                except Exception:
                    pipeline_blocks = []
            if not pipeline_blocks:
                pipeline_blocks = host._get_detection_pipeline_blocks(host._get_detection_method_key())
            host._save_detection_pipeline_blocks(pipeline_blocks)
        except Exception:
            pass

        host._prune_legacy_yolo_arch_session_keys()

        host._save_local_setting("char_preview_box_mode", host._get_preview_box_mode_key())
        host._save_local_setting("char_preview_sort_mode", host._get_preview_sort_mode_key())
        host._save_local_setting("char_preview_layout_filter", host._get_preview_layout_filter_key())
        try:
            preview_dir_value = str(host.preview_dir_var.get() or "").strip()
            host._save_local_setting("char_preview_dir", preview_dir_value)
            try:
                if (
                    preview_dir_value
                    and getattr(host, "_step3_linear_mode", False)
                    and CAMPAIGN.get_active_project_name()
                ):
                    CAMPAIGN.set_step3_preview_dir(preview_dir_value)
                    sync_registry = getattr(host, "_sync_campaign_step3_preview_artifact_registry", None)
                    if callable(sync_registry):
                        sync_registry(preview_dir_value)
            except Exception:
                pass
        except Exception:
            pass

        if not getattr(host, "_step3_linear_mode", False):
            for key, var in [
                ("char_annotation_run_dir", host.annotation_run_dir_var),
                ("char_xml_path", host.xml_path_var),
                ("char_images_dir", host.images_dir_var),
                ("char_yolo_model", host.yolo_model_path_var),
            ]:
                host._save_local_setting(key, var.get())
        else:
            try:
                host._persist_step3_extract_state()
            except Exception:
                pass

    except Exception:
        pass
