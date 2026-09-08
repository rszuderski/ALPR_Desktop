#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Late help bindings and event hooks for the Z2 main widget tree."""

from __future__ import annotations

import time

from ..campaign_manager import CAMPAIGN
from ..config import logger
from .help_manager import HELP


def bind_annotation_widget_help_and_events(
    host,
    *,
    row_in,
    row_out,
    run_row,
    img_row,
    split_lf,
    row_conf,
) -> None:
    self = host
    binding_started = time.perf_counter()
    binding_phase_started = binding_started

    def _mark_binding_phase(name: str) -> None:
        nonlocal binding_phase_started
        try:
            now = time.perf_counter()
            elapsed_ms = (now - binding_phase_started) * 1000.0
            total_ms = (now - binding_started) * 1000.0
            if elapsed_ms >= 250.0 or total_ms >= 1000.0:
                logger.info(
                    "[Z2 PERF] create_annotation_widgets "
                    f"bindings.{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms"
                )
            binding_phase_started = now
        except Exception:
            pass

    def _bind_help(widget, key: str) -> None:
        if widget is None:
            return
        try:
            HELP.bind_help(widget, key)
        except Exception:
            pass

    # Podpinanie systemu pomocy pod lokalnÄ… konsolÄ™
    HELP.bind_help(self.input_dir_title_lbl, "tab1_input")
    HELP.bind_help(row_in, "tab1_input")
    HELP.bind_help(self.output_dir_title_lbl, "tab1_output")
    HELP.bind_help(row_out, "tab1_output")
    HELP.bind_help(self.run_title_lbl, "tab1_start")
    HELP.bind_help(self.route_badge_lbl, "tab1_start")
    HELP.bind_help(self.route_summary_lbl, "tab1_start")
    HELP.bind_help(self.auto_plate_model_section, "tab1_model_pla")
    HELP.bind_help(self.workflow_plate_path_entry, "tab1_model_pla")
    HELP.bind_help(self.workflow_conf_row, "tab1_conf")
    HELP.bind_help(self.auto_vehicle_choice_title_lbl, "tab1_model_veh")
    HELP.bind_help(self.auto_vehicle_choice_row, "tab1_model_veh")
    HELP.bind_help(self.workflow_vehicle_model_section, "tab1_model_veh")
    HELP.bind_help(self.workflow_vehicle_combo, "tab1_model_veh")
    HELP.bind_help(self.workflow_manual_vehicle_assist_check, "tab1_model_veh")
    HELP.bind_help(self.workflow_input_section, "tab1_input")
    HELP.bind_help(self.workflow_input_entry, "tab1_input")
    HELP.bind_help(self.workflow_nav_row, "tab1_workflow_nav")
    HELP.bind_help(self.workflow_back_btn, "tab1_workflow_nav")
    HELP.bind_help(self.workflow_next_btn, "tab1_workflow_nav")
    HELP.bind_help(self.workflow_start_section, "tab1_start")
    HELP.bind_help(self.workflow_start_title_lbl, "tab1_start")
    HELP.bind_help(self.workflow_start_intro_lbl, "tab1_start")
    HELP.bind_help(self.workflow_start_action_hint_lbl, "tab1_start")
    HELP.bind_help(self.auto_route_radio, "tab1_start")
    HELP.bind_help(self.manual_route_radio, "tab1_start")
    _bind_help(getattr(self, "mode_combo", None), "tab1_mode")
    _bind_help(getattr(self, "vehicle_combo", None), "tab1_model_veh")
    _bind_help(getattr(self, "pla_frame", None), "tab1_model_pla")
    _bind_help(getattr(self, "plate_path_entry", None), "tab1_model_pla")
    HELP.bind_help(self.export_title_lbl, "tab1_dataset_export")
    HELP.bind_help(self.plate_dataset_run_title_lbl, "tab1_dataset_run")
    HELP.bind_help(run_row, "tab1_dataset_run")
    HELP.bind_help(self.plate_dataset_images_title_lbl, "tab1_dataset_images")
    HELP.bind_help(img_row, "tab1_dataset_images")
    HELP.bind_help(self.plate_dataset_out_title_lbl, "tab1_dataset_export")
    HELP.bind_help(self.split_title_lbl, "tab1_dataset_split")
    HELP.bind_help(split_lf, "tab1_dataset_split")
    HELP.bind_help(self.export_plate_dataset_btn, "tab1_dataset_export")
    HELP.bind_help(self.export_plate_annotations_btn, "tab1_dataset_export")
    HELP.bind_help(self.jump_to_export_btn, "tab1_dataset_export")
    HELP.bind_help(self.followup_section, "tab1_dataset_export")
    HELP.bind_help(self.followup_title_lbl, "tab1_dataset_export")
    HELP.bind_help(self.followup_intro_lbl, "tab1_dataset_export")
    HELP.bind_help(self.post_annotation_hint_lbl, "tab1_dataset_export")
    HELP.bind_help(self.manual_stage_section, "tab1_stage")
    HELP.bind_help(self.manual_stage_title_lbl, "tab1_stage")
    HELP.bind_help(self.manual_stage_help_lbl, "tab1_stage")
    HELP.bind_help(self.manual_stage_path_title_lbl, "tab1_stage")
    HELP.bind_help(self.manual_stage_path_entry, "tab1_stage")
    HELP.bind_help(self.manual_stage_status_lbl, "tab1_stage")
    HELP.bind_help(self.manual_stage_use_btn, "tab1_stage_use")
    HELP.bind_help(self.manual_stage_add_btn, "tab1_stage_add")
    HELP.bind_help(self.manual_stage_export_box, "tab1_dataset_export")
    HELP.bind_help(self.manual_stage_export_title_lbl, "tab1_dataset_export")
    HELP.bind_help(self.manual_stage_export_help_lbl, "tab1_dataset_export")
    HELP.bind_help(self.manual_stage_export_btn, "tab1_dataset_export")
    HELP.bind_help(self.export_back_btn, "tab1_dataset_export")
    _bind_help(getattr(self, "plate_browse_btn", None), "tab1_model_pla")
    _bind_help(row_conf, "tab1_conf")
    _bind_help(getattr(self, "device_combo", None), "tab1_device")
    HELP.bind_help(self.start_btn, "tab1_start")
    HELP.bind_help(self.stop_btn, "tab1_stop")
    HELP.bind_help(self.enter_manual_review_btn, "tab1_enter_review")
    HELP.bind_help(self.open_run_dir_btn, "tab1_open_run")
    HELP.bind_help(self.btn_toggle_annotation_log, "tab1_logs")
    HELP.bind_help(self.preview_listbox, "tab1_preview_list")
    HELP.bind_help(self.preview_canvas, "tab1_preview_canvas")
    HELP.bind_help(self.preview_prev_btn, "tab1_preview_nav")
    HELP.bind_help(self.preview_next_btn, "tab1_preview_nav")
    HELP.bind_help(self.preview_fit_btn, "tab1_preview_nav")
    HELP.bind_help(self.preview_fullscreen_btn, "tab1_preview_nav")
    HELP.bind_help(self.preview_draw_btn, "tab1_preview_edit")
    HELP.bind_help(self.preview_save_btn, "tab1_preview_edit")
    HELP.bind_help(self.preview_move_stage_btn, "tab1_preview_stage_actions")
    HELP.bind_help(self.preview_delete_image_btn, "tab1_preview_stage_actions")
    _bind_help(getattr(self, "veh_custom_row", None), "tab1_custom_model")
    HELP.bind_help(self.approve_btn, "tab1_approve")
    HELP.bind_help(self.return_to_campaign_right_btn, "tab1_return_wizard")
    _mark_binding_phase("help")
    # The global mousewheel routing below is enough for both scrollable panels.
    # Recursively binding every child in Z2 used to cost several seconds on
    # first entry, especially after the left panel grew with the graph workflow.
    try:
        self.workflow_entry_shell_inner.bind("<Configure>", self._sync_workflow_copy_wraplength, add="+")
    except Exception:
        pass
    _mark_binding_phase("configure")
    self.frame.after_idle(self._sync_left_panel_canvas_width)
    self.frame.after_idle(self._sync_workflow_copy_wraplength)
    self.frame.after_idle(self._sync_left_panel_scrollregion)
    self.frame.after_idle(self._sync_right_panel_canvas_width)
    self.frame.after_idle(self._sync_right_panel_scrollregion)
    self.frame.after_idle(self._sync_approve_hint_wraplength)
    self.frame.after_idle(lambda: self._schedule_main_pane_layout_refresh(force_defaults=True))
    self.frame.after_idle(self._refresh_preview_filter_bar_state)
    _mark_binding_phase("after_idle")
    self.frame.bind_all("<MouseWheel>", self._on_left_panel_global_mousewheel, add="+")
    self.frame.bind_all("<Button-4>", self._on_left_panel_global_mousewheel, add="+")
    self.frame.bind_all("<Button-5>", self._on_left_panel_global_mousewheel, add="+")
    self.frame.bind_all("<MouseWheel>", self._on_right_panel_global_mousewheel, add="+")
    self.frame.bind_all("<Button-4>", self._on_right_panel_global_mousewheel, add="+")
    self.frame.bind_all("<Button-5>", self._on_right_panel_global_mousewheel, add="+")
    _mark_binding_phase("global_scroll")
    self._sync_main_pane_right_panel_visibility()
    _mark_binding_phase("right_panel_visibility")
    campaign_active = False
    try:
        campaign_active = bool(CAMPAIGN.get_active_project_name()) and not bool(
            getattr(getattr(self, "app", None), "campaign_free_mode", False)
        )
    except Exception:
        campaign_active = False

    if campaign_active:
        try:
            self._campaign_step2_transition_refresh_pending = True
        except Exception:
            pass
        _mark_binding_phase("workflow_ui_skipped_campaign")
    else:
        self._refresh_free_mode_workflow_ui()
        _mark_binding_phase("workflow_ui")
