#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted Z2 workflow/state methods for AnnotationTab.

This module intentionally keeps methods as plain functions receiving ``self``.
The owning class delegates to them, which physically reduces tab_annotation.py
without changing the state model or the public method names used by callbacks.
"""

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
from .z2_assistant_context import (
    get_free_mode_assistant_context,
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
    _promote_campaign_t05_ok_to_approved_pool_before_return,
    _promote_campaign_char_repair_ok_to_approved_pool_before_return,
    _sync_campaign_char_repair_approved_run_to_project_source,
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
