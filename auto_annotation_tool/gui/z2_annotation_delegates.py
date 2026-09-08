"""Delegate bindings for the Z2 annotation tab.

The annotation tab is still the public integration point for older GUI code,
but the implementation lives in focused runtime modules.  Binding delegates
after class creation keeps the public method names stable without keeping
hundreds of one-line wrappers in ``tab_annotation.py``.
"""

from __future__ import annotations

from . import z2_layout_ui_runtime
from . import z2_model_runtime
from . import z2_session_runtime


_SESSION_INSTANCE_METHODS = (
    "_annotation_session_defaults",
    "_annotation_auto_modal_defaults",
    "_apply_annotation_auto_modal_defaults_to_snapshot",
    "_strip_persisted_annotation_miniflow_snapshot",
    "_reset_annotation_auto_modal_state",
    "_is_free_mode_session_context",
    "_is_campaign_step2_context",
    "_is_annotation_tab_selected",
    "reset_preview_selection_to_first_visible_on_tab_entry",
    "_get_campaign_annotation_state_path",
    "_path_is_within_any",
    "_get_annotation_output_base_dir",
    "_get_annotation_run_roots",
    "_coerce_annotation_output_dir",
    "_resolve_safe_annotation_run_dir",
    "_get_annotation_session_text",
    "_get_annotation_session_float",
    "_get_annotation_session_int",
    "_get_annotation_session_bool",
    "_get_annotation_session_list",
    "_load_free_mode_session_snapshot",
    "_collect_free_mode_session_snapshot",
    "_collect_campaign_project_snapshot",
    "_sanitize_free_mode_session_snapshot",
    "capture_free_mode_snapshot_for_project_return",
    "_load_campaign_project_snapshot",
    "_save_campaign_project_snapshot",
    "_save_campaign_project_snapshot_with_options",
    "_cancel_pending_campaign_route_cleanup",
    "_build_campaign_route_cleanup_payload",
    "_schedule_deferred_campaign_route_cleanup",
    "reset_campaign_iteration_route_state",
    "_remember_annotation_input_dir_ready",
    "_annotation_input_dir_ready",
    "prepare_campaign_iteration_transition",
    "_get_campaign_iteration_artifact_bundle",
    "_get_campaign_step2_active_run_entry",
    "_normalize_campaign_step2_bootstrap_manual_template",
    "_get_campaign_auto_annotation_bootstrap",
    "_is_campaign_char_repair_return_mode",
    "_is_campaign_plate_step4_repair_return_mode",
    "get_campaign_step2_bootstrap",
    "get_campaign_step2_source_state",
    "_get_campaign_latest_approved_plate_run_state",
    "get_campaign_step2_view_model",
    "get_campaign_step2_wizard_status",
    "_get_campaign_previous_manual_source_bundle",
    "_get_campaign_reused_manual_annotation_count",
    "_ensure_campaign_manual_package_ready",
    "_ensure_campaign_preview_edit_run",
    "_get_campaign_plate_approved_filenames",
    "_get_campaign_plate_approved_source_keys",
    "_get_campaign_step3_preview_source_context",
    "_get_campaign_char_effective_source_hidden_filenames",
    "_get_campaign_iteration_manifest_image_paths",
    "_get_campaign_iteration_manifest_image_count",
    "_collect_campaign_auto_annotation_sources",
    "_clear_campaign_manual_reuse_context",
    "_apply_campaign_manual_reuse_context",
    "_sync_campaign_pending_batch_summary_from_preview",
    "_on_campaign_reuse_manual_toggle",
    "_get_campaign_manual_reuse_ui_state",
    "_refresh_campaign_manual_reuse_option_ui",
    "_restore_preview_from_annotation_run",
    "_finalize_successful_annotation_run_ui",
    "_apply_campaign_project_snapshot",
    "_bind_free_mode_session_observers",
    "_on_free_mode_session_var_changed",
    "_queue_free_mode_session_save",
    "_mark_preview_user_interaction",
    "_preview_user_interaction_quiet_remaining_ms",
    "_cancel_campaign_char_effective_source_refresh",
    "_campaign_char_effective_source_refresh_needed",
    "_schedule_campaign_char_effective_source_refresh",
    "_cancel_preview_approved_persist",
    "_schedule_preview_approved_persist",
    "_flush_preview_approved_persist",
    "_cancel_preview_approval_followup_refresh",
    "_schedule_preview_approval_followup_refresh",
    "_refresh_campaign_char_effective_source_after_approval",
    "_cancel_preview_resume_persist",
    "_schedule_preview_resume_persist",
    "_invalidate_preview_runtime_caches",
    "flush_free_mode_session_state",
    "_apply_free_mode_session_snapshot",
)


_SESSION_STATIC_METHODS = (
    "_normalize_workflow_route_value",
    "_normalize_manual_entry_mode",
    "_normalize_auto_vehicle_choice",
    "_normalize_workflow_step_value",
    "_normalize_free_mode_screen_value",
    "_path_value_to_text",
    "_dir_has_images",
    "_resolve_annotation_input_images_dir",
    "_annotation_input_cache_key",
    "_normalize_campaign_iteration_target_value",
    "_extract_iteration_label_from_path",
    "_load_cvat_plate_annotated_filenames",
    "_build_campaign_source_image_key",
    "_campaign_reuse_manual_badge",
)


_SESSION_CLASS_METHODS = (
    "_path_value_to_path",
    "_paths_equivalent",
    "_path_is_within",
    "_dedupe_paths",
)


_LAYOUT_INSTANCE_METHODS = (
    "_sync_right_panel_scrollregion",
    "_sync_right_panel_canvas_width",
    "_sync_left_panel_scrollregion",
    "_sync_left_panel_canvas_width",
    "_should_show_free_mode_manual_right_panel",
    "_should_show_right_panel",
    "_get_preview_left_counter_total",
    "_get_preview_left_counter_width_bucket",
    "_get_preview_left_counter_min_width",
    "_sync_preview_left_counter_layout_width",
    "_sync_main_pane_right_panel_visibility",
    "_get_z2_free_status_panel_run_dir",
    "_build_z2_free_export_status_state",
    "_is_z2_free_export_choice_available",
    "_refresh_free_mode_manual_right_panel",
    "_restore_right_panel_content_after_fullscreen",
    "_schedule_right_panel_content_restore_after_fullscreen",
    "_get_main_pane_width_limits",
    "_schedule_main_pane_layout_refresh",
    "_reset_main_pane_left_width_for_z2",
    "_apply_main_pane_layout",
    "_on_main_pane_configure",
    "_on_main_pane_button_press",
    "_on_main_pane_drag_motion",
    "_on_main_pane_drag_release",
    "_sync_approve_hint_wraplength",
    "_sync_workflow_copy_wraplength",
    "_widget_contains_point",
    "_get_preview_pointer_canvas_position",
    "_mousewheel_units",
    "_on_preview_listbox_mousewheel",
    "_panel_canvas_overflows",
    "_scroll_panel_canvas_if_targeted",
    "_on_left_panel_global_mousewheel",
    "_on_right_panel_global_mousewheel",
    "_restore_scroll_canvas_focus",
    "_redirect_child_mousewheel_to_canvas",
    "_bind_scroll_canvas_children",
    "_build_left_section_separator",
    "_build_left_path_row",
    "_bind_workflow_card",
    "_bind_preview_sort_tile",
    "_set_preview_list_sort_mode",
    "_normalize_preview_list_sort_mode",
    "_get_preview_metric_filter_input_thresholds",
    "_get_preview_metric_filter_thresholds",
    "_preview_annotation_passes_metric_filters",
    "_filter_preview_list_entries",
    "_reset_preview_metric_filters",
    "_apply_preview_metric_filters",
    "_open_preview_metric_filter_modal",
    "_refresh_preview_filter_bar_state",
    "_invalidate_preview_list_frozen_order",
    "_get_preview_list_context_key",
    "_store_preview_list_frozen_order",
    "_set_workflow_route_card_hover",
    "_build_z2_action_context",
    "_get_z2_primary_actions",
    "_get_z2_secondary_actions",
    "_activate_z2_secondary_action",
    "_select_workflow_route",
    "_build_workflow_step_card",
    "_register_workflow_step_card",
    "_refresh_bound_label_wraplength",
    "_bind_label_wrap_to_container",
    "_apply_workflow_step_widget_style",
    "_refresh_workflow_button_styles",
    "_refresh_workflow_progress_style",
    "_refresh_workflow_step_cards",
    "_set_widget_packed",
    "_normalize_workflow_start_section_order",
    "_normalize_approve_panel_order",
    "_measure_visible_pack_height",
    "_scroll_left_panel_to_widget",
)


_LAYOUT_STATIC_METHODS = (
    "_widget_is_packed",
)


_MODEL_INSTANCE_METHODS = (
    "_normalize_mode_value",
    "_mode_uses_vehicle",
    "_mode_uses_plate",
    "_vehicle_model_controls_enabled",
    "_plate_model_controls_enabled",
    "_refresh_detection_configuration_ui",
    "_set_progress_counters",
    "_on_mode_change",
    "_update_model_lists",
    "_on_vehicle_model_change",
    "_set_plate_model_controls_state",
    "_get_bound_character_model_path",
    "_collect_character_model_candidates",
    "_refresh_character_model_choices",
    "_get_selected_character_model_path",
    "_apply_character_model_selection",
    "_on_character_model_change",
    "_select_vehicle_custom",
    "_select_plate_custom",
    "_advance_campaign_auto_step_after_plate_model_selection",
    "_get_campaign_project_plate_model_path",
    "_iter_plate_training_history_files",
    "_find_plate_training_run_for_model",
    "_extract_plate_model_metrics_from_rows",
    "_read_results_csv_metrics",
    "_read_results_csv_metrics_for_model",
    "_read_exported_model_metadata_metrics",
    "_build_plate_model_quality_metrics",
    "_get_plate_model_quality_metrics",
    "_build_auto_annotation_model_quality_rows",
    "_format_auto_annotation_model_quality_summary",
    "_render_auto_annotation_model_quality_table",
    "_confirm_campaign_plate_model_identity_choice",
    "_remember_plate_model_runtime_meta",
    "_clear_plate_model_runtime_meta",
    "_collect_plate_model_manifest_fields",
    "_get_effective_plate_model_runtime_meta",
    "_refresh_plate_model_runtime_info_ui",
    "_restore_plate_model_selection_from_active_run",
    "_get_auto_model_picker_initial_dir",
    "_get_campaign_plate_model_picker_initial_dir",
    "_pick_campaign_plate_auto_model_file",
    "_adopt_campaign_plate_model_into_project",
    "_apply_campaign_plate_auto_model_choice",
    "_clear_annotation_run_scope_meta",
    "_remember_annotation_run_scope_meta",
    "_collect_annotation_run_scope_manifest_fields",
    "_get_preview_group_actual_indices",
    "_resolve_preview_scope_paths",
    "_get_preview_auto_scope_protected_filenames",
    "_collect_plate_auto_scope_candidates",
    "_collect_plate_auto_scope_bucket_protected_counts",
    "_collect_plate_auto_scope_bucket_paths",
    "_can_offer_preview_scope_selection_for_auto_run",
    "_set_plate_auto_scope_modal_ui_lock",
    "_focus_plate_auto_scope_modal",
    "_is_plate_auto_scope_modal_blocking_actions",
    "_fit_borderless_dialog",
    "_set_plate_auto_scope_selection_mode",
    "_get_preview_listbox_selected_row_count",
    "_get_preview_listbox_selection_signature",
    "_refresh_plate_auto_scope_modal_selection_state",
    "_prompt_plate_auto_scope_choice",
    "_build_annotation_input_subset_dir",
    "_prompt_campaign_plate_auto_model_choice",
)


_MODEL_STATIC_METHODS = (
    "_get_model_identity_caption",
    "_model_quality_float",
    "_paths_refer_to_same_model",
    "_path_is_inside_dir",
    "_training_history_run_looks_like_plate",
    "_load_training_runs_from_history_file",
    "_format_model_quality_number",
    "_format_model_quality_epoch",
    "_describe_auto_model_quality",
    "_auto_model_metric_tone",
    "_shorten_model_quality_text",
    "_normalize_plate_model_source",
    "_normalize_plate_model_scope",
    "_dedupe_image_paths_by_name",
    "_count_unique_preview_scope_indices",
    "_count_protected_preview_scope_indices",
)


_MODEL_CLASS_METHODS = (
    "_model_quality_value",
)


def bind_annotation_tab_delegates(annotation_tab_cls):
    """Attach extracted runtime methods to ``AnnotationTab``."""

    for method_name in _SESSION_INSTANCE_METHODS:
        setattr(annotation_tab_cls, method_name, getattr(z2_session_runtime, method_name))

    for method_name in _SESSION_STATIC_METHODS:
        setattr(
            annotation_tab_cls,
            method_name,
            staticmethod(getattr(z2_session_runtime, method_name)),
        )

    for method_name in _SESSION_CLASS_METHODS:
        setattr(
            annotation_tab_cls,
            method_name,
            classmethod(getattr(z2_session_runtime, method_name)),
        )

    for method_name in _LAYOUT_INSTANCE_METHODS:
        setattr(annotation_tab_cls, method_name, getattr(z2_layout_ui_runtime, method_name))

    for method_name in _LAYOUT_STATIC_METHODS:
        setattr(
            annotation_tab_cls,
            method_name,
            staticmethod(getattr(z2_layout_ui_runtime, method_name)),
        )

    for method_name in _MODEL_INSTANCE_METHODS:
        setattr(annotation_tab_cls, method_name, getattr(z2_model_runtime, method_name))

    for method_name in _MODEL_STATIC_METHODS:
        setattr(
            annotation_tab_cls,
            method_name,
            staticmethod(getattr(z2_model_runtime, method_name)),
        )

    for method_name in _MODEL_CLASS_METHODS:
        setattr(
            annotation_tab_cls,
            method_name,
            classmethod(getattr(z2_model_runtime, method_name)),
        )

    return annotation_tab_cls
