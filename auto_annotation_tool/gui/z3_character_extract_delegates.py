"""Extraction-source delegates for ``CharacterAnnotationTab``.

These methods keep legacy method names on the tab while the actual work lives
in focused Z3 modules.
"""

from __future__ import annotations

from pathlib import Path

from ..data_models import Detection
from .z3_campaign_flow import (
    build_step3_pz3_path_selection_view_model,
    build_step3_pz3_status_panel_view_model,
    finalize_step3_from_existing_outputs,
    get_step3_finish_block_message,
    has_any_step3_export_outputs,
    resolve_step3_campaign_action_command,
    return_step3_result_to_wizard,
    return_to_t05_work_after_step3_pz1,
    return_to_wizard_for_step3_rework,
    set_step3_finish_hint,
)
from .z3_export_summary import (
    build_step3_export_summary,
    inspect_yolo_dataset_label_objects,
    read_step3_export_summary,
    write_step3_export_summary,
)
from .z3_extraction_sources import (
    backfill_preview_expected_texts_from_sources,
    count_xml_plate_cut_targets,
    derive_candidate_root_for_match,
    evaluate_images_dir_for_xml,
    extract_source_plate_tokens_from_filename,
    find_matching_images_dir_for_xml,
    get_preferred_source_roots,
    is_path_within,
    is_recommended_images_dir,
    normalize_xml_image_relpath,
    plate_cut_reading_order_key,
    prepare_plate_cut_detections_for_source,
    read_xml_image_names,
    refresh_source_binding_status,
    summarize_missing_xml_images,
    xml_relpath_to_path,
)
from .z3_extraction_tab_ui import (
    append_extract_cut_plan,
    bind_source_path_watchers,
    format_extract_cut_plan,
    on_source_path_var_changed,
    schedule_source_binding_refresh,
    set_widget_state,
    update_step3_source_path_lock,
)
from .z3_paths import (
    get_step3_char_classification_datasets_root_dir,
    get_step3_chars_root_dir,
    get_step3_datasets_root_dir,
    get_step3_summary_dir,
)
from .z3_readiness import (
    get_campaign_step3_annotation_readiness,
    get_campaign_step3_training_readiness,
)
from .z3_shared_ui import (
    build_step3_pz3_dataset_mode_view_model,
    refresh_pz3_status_panel_ui,
)
from .z3_theme_ui import get_inline_status_widget_snapshot

from .z3_free_mode_flow import (
    go_to_next_extract_step,
    go_to_previous_extract_step,
    handle_extract_entry_selection,
    set_extract_entry_mode,
)
from .z3_extraction_tab_ui import (
    annotation_run_manifest_path,
    apply_annotation_run_source,
    bind_extract_entry_card,
    coerce_extract_workflow_step,
    get_annotation_run_input_dir,
    get_annotation_tab,
    get_campaign_iteration_artifact_bundle,
    get_extract_workflow_step,
    get_preferred_z2_source_candidate,
    get_registry_preferred_step3_source_candidate,
    load_annotation_run_manifest,
    normalize_extract_entry_mode,
    normalize_extract_workflow_step,
    paths_equivalent,
    persist_step3_extract_state,
    pick_annotation_run_dir,
    refresh_extract_entry_cards,
    refresh_extract_workflow_ui,
    resolve_existing_annotation_run_dir,
    restore_step3_extract_state_from_project,
    schedule_extract_workflow_refresh,
    set_extract_workflow_step,
    set_pending_z2_annotation_source as set_pending_z2_annotation_source_ui,
    sync_campaign_step3_artifact_registry,
    sync_campaign_step3_preview_artifact_registry,
    use_z2_source_on_demand,
)
from .z3_shared_ui import (
    build_step3_extract_workflow_view_model,
    refresh_extract_step_cards,
    refresh_extract_step_nav_buttons,
)


def _set_widget_state(self, widget, state: str):
    set_widget_state(self, widget, state)


def _update_step3_source_path_lock(self):
    update_step3_source_path_lock(self)


def _bind_source_path_watchers(self):
    bind_source_path_watchers(self)


def _on_source_path_var_changed(self, *_args):
    on_source_path_var_changed(self, *_args)


def _schedule_source_binding_refresh(self, delay_ms: int = 150):
    schedule_source_binding_refresh(self, delay_ms)


def _normalize_xml_image_relpath(self, raw_name: str) -> str:
    return normalize_xml_image_relpath(raw_name)


def _xml_relpath_to_path(self, rel_path: str) -> Path:
    return xml_relpath_to_path(rel_path)


def _read_xml_image_names(self, xml_path: Path) -> list[str]:
    return read_xml_image_names(xml_path)


def _count_xml_plate_cut_targets(self, xml_path: Path) -> dict:
    return count_xml_plate_cut_targets(xml_path)


def _extract_source_plate_tokens_from_filename(filename: str) -> list[str]:
    return extract_source_plate_tokens_from_filename(filename)


def _plate_cut_reading_order_key(index_and_detection):
    return plate_cut_reading_order_key(index_and_detection)


def _prepare_plate_cut_detections_for_source(self, image_name: str, plates: list[Detection]) -> list[Detection]:
    return prepare_plate_cut_detections_for_source(self, image_name, plates)


def _backfill_preview_expected_texts_from_sources(self, metadata_map: dict) -> bool:
    return backfill_preview_expected_texts_from_sources(self, metadata_map)


def _format_extract_cut_plan(self, payload: dict | None = None) -> str:
    return format_extract_cut_plan(self, payload)


def _append_extract_cut_plan(self, message: str, payload: dict | None = None) -> str:
    return append_extract_cut_plan(self, message, payload)


def _evaluate_images_dir_for_xml(self, images_dir: Path, xml_image_names: list[str]) -> dict:
    return evaluate_images_dir_for_xml(images_dir, xml_image_names)


def _summarize_missing_xml_images(self, missing: list[str], limit: int = 3) -> str:
    return summarize_missing_xml_images(missing, limit=limit)


def _is_path_within(self, path: Path, root: Path) -> bool:
    return is_path_within(path, root)


def _get_preferred_source_roots(self) -> list[Path]:
    return get_preferred_source_roots()


def _is_recommended_images_dir(self, images_dir: Path) -> bool:
    return is_recommended_images_dir(images_dir)


def _derive_candidate_root_for_match(self, found_file: Path, xml_rel_name: str) -> Path | None:
    return derive_candidate_root_for_match(found_file, xml_rel_name)


def _find_matching_images_dir_for_xml(self, xml_image_names: list[str]) -> dict | None:
    return find_matching_images_dir_for_xml(xml_image_names)


def _refresh_source_binding_status(self, allow_autofind: bool = True) -> dict:
    return refresh_source_binding_status(self, allow_autofind=allow_autofind)


def _count_preview_statuses(self):
    return self._count_statuses_in_metadata_mapping(self.preview_metadata)


def _get_step3_chars_root_dir(self, ensure_exists: bool = False) -> Path:
    return get_step3_chars_root_dir(self, ensure_exists=ensure_exists)


def _get_step3_datasets_root_dir(self, ensure_exists: bool = False) -> Path:
    return get_step3_datasets_root_dir(self, ensure_exists=ensure_exists)


def _get_step3_char_classification_datasets_root_dir(self, ensure_exists: bool = False) -> Path:
    return get_step3_char_classification_datasets_root_dir(self, ensure_exists=ensure_exists)


def _get_step3_summary_dir(self) -> Path:
    return get_step3_summary_dir(self)


def _build_step3_export_summary(
    self,
    gold_dataset_path: str | None = None,
    review_pack_path: str | None = None,
    retry_pack_path: str | None = None,
    note: str = "",
):
    return build_step3_export_summary(
        self,
        gold_dataset_path=gold_dataset_path,
        review_pack_path=review_pack_path,
        retry_pack_path=retry_pack_path,
        note=note,
    )


def _write_step3_export_summary(self, summary: dict) -> Path:
    return write_step3_export_summary(self, summary)


def _read_step3_export_summary(self) -> dict:
    return read_step3_export_summary(self)


def _inspect_yolo_dataset_label_objects(self, dataset_dir: Path | None) -> dict:
    return inspect_yolo_dataset_label_objects(dataset_dir)


def _get_campaign_step3_annotation_readiness(self) -> dict:
    return get_campaign_step3_annotation_readiness(self)


def _return_step3_result_to_wizard(self, summary: dict):
    return return_step3_result_to_wizard(self, summary)


def _finalize_step3_from_existing_outputs(self):
    return finalize_step3_from_existing_outputs(self)


def _get_campaign_step3_training_readiness(self) -> dict:
    return get_campaign_step3_training_readiness(self)


def _get_step3_finish_block_message(self, readiness: dict | None = None) -> str:
    return get_step3_finish_block_message(readiness)


def _set_step3_finish_hint(self, text: str = "", tone: str = "muted"):
    set_step3_finish_hint(self, text=text, tone=tone)


def _has_any_step3_export_outputs(self) -> bool:
    return has_any_step3_export_outputs(self)


def _return_to_wizard_for_step3_rework(self):
    return_to_wizard_for_step3_rework(self)


def _return_to_t05_work_after_step3_pz1(self):
    return_to_t05_work_after_step3_pz1(self)


def _resolve_step3_campaign_action_command(self, command_id: str):
    return resolve_step3_campaign_action_command(self, command_id)


def _get_step3_pz3_path_selection_view_model(self):
    return build_step3_pz3_path_selection_view_model(self)


def _get_step3_pz3_dataset_mode_view_model(self):
    return build_step3_pz3_dataset_mode_view_model(self)


def _get_inline_status_widget_snapshot(
    self,
    widget,
    *,
    fallback_text: str = "",
    fallback_tone: str = "muted",
) -> tuple[str, str]:
    return get_inline_status_widget_snapshot(
        widget,
        fallback_text=fallback_text,
        fallback_tone=fallback_tone,
    )


def _get_step3_pz3_status_panel_view_model(self):
    return build_step3_pz3_status_panel_view_model(self)


def _refresh_pz3_status_panel_ui(self):
    refresh_pz3_status_panel_ui(self)

def _normalize_extract_entry_mode(self, value: str | None = None) -> str:
    return normalize_extract_entry_mode(self, value)

@staticmethod
def _normalize_extract_workflow_step(step: str | None = None) -> str:
    return normalize_extract_workflow_step(step)

def _coerce_extract_workflow_step(self, step: str | None = None, *, lightweight: bool = False) -> str:
    return coerce_extract_workflow_step(self, step, lightweight=lightweight)

def _get_extract_workflow_step(self, *, lightweight: bool = False) -> str:
    return get_extract_workflow_step(self, lightweight=lightweight)

def _set_extract_workflow_step(self, step: str, *, refresh: bool = True):
    set_extract_workflow_step(self, step, refresh=refresh)

def _get_extract_entry_mode(self) -> str:
    return self._normalize_extract_entry_mode()

def _set_extract_entry_mode(self, mode: str, *, persist: bool = True):
    set_extract_entry_mode(self, mode, persist=persist)

def _handle_extract_entry_selection(self, mode: str):
    handle_extract_entry_selection(self, mode)

def _persist_step3_extract_state(self):
    persist_step3_extract_state(self)

def _sync_campaign_step3_artifact_registry(
    self,
    *,
    entry_mode: str = "",
    workflow_step: str = "entry",
    annotation_run_dir: str = "",
    xml_path: str = "",
    images_dir: str = "",
) -> None:
    sync_campaign_step3_artifact_registry(
        self,
        entry_mode=entry_mode,
        workflow_step=workflow_step,
        annotation_run_dir=annotation_run_dir,
        xml_path=xml_path,
        images_dir=images_dir,
    )

def _sync_campaign_step3_preview_artifact_registry(self, preview_dir) -> None:
    sync_campaign_step3_preview_artifact_registry(self, preview_dir)

def _restore_step3_extract_state_from_project(self):
    restore_step3_extract_state_from_project(self)

def _go_to_previous_extract_step(self):
    go_to_previous_extract_step(self)

def _go_to_next_extract_step(self):
    go_to_next_extract_step(self)

def _get_step3_extract_workflow_view_model(self) -> Step3ExtractWorkflowViewModel:
    return build_step3_extract_workflow_view_model(self)

def _refresh_extract_step_nav_buttons(self, vm: Step3ExtractWorkflowViewModel | None = None):
    refresh_extract_step_nav_buttons(self, vm)

def _bind_extract_entry_card(self, card_widget, mode: str):
    bind_extract_entry_card(self, card_widget, mode)

def _refresh_extract_entry_cards(self):
    refresh_extract_entry_cards(self)

def _refresh_extract_step_cards(self, vm: Step3ExtractWorkflowViewModel | None = None):
    refresh_extract_step_cards(self, vm)

@staticmethod
def _paths_equivalent(left, right) -> bool:
    return paths_equivalent(left, right)

def _resolve_existing_annotation_run_dir(self, candidate) -> Path | None:
    return resolve_existing_annotation_run_dir(candidate)

def _annotation_run_manifest_path(self, run_dir: Path) -> Path:
    return annotation_run_manifest_path(run_dir)

def _load_annotation_run_manifest(self, run_dir: Path) -> dict:
    return load_annotation_run_manifest(self, run_dir)

def _get_campaign_iteration_artifact_bundle(self) -> dict:
    return get_campaign_iteration_artifact_bundle()

def _get_registry_preferred_step3_source_candidate(self) -> dict | None:
    return get_registry_preferred_step3_source_candidate(self)

def _get_annotation_tab(self):
    return get_annotation_tab(self)

def _get_annotation_run_input_dir(self, run_dir: Path | None) -> Path | None:
    return get_annotation_run_input_dir(self, run_dir)

def _apply_annotation_run_source(self, run_dir: Path | None, *, refresh_status: bool = True) -> bool:
    return apply_annotation_run_source(self, run_dir, refresh_status=refresh_status)

def set_pending_z2_annotation_source(self, *, xml_path: str = "", images_dir: str = "", run_dir: str = ""):
    set_pending_z2_annotation_source_ui(self, xml_path=xml_path, images_dir=images_dir, run_dir=run_dir)

def _pick_annotation_run_dir(self):
    pick_annotation_run_dir(self)

def _get_preferred_z2_source_candidate(self) -> dict | None:
    return get_preferred_z2_source_candidate(self)

def _use_z2_source_on_demand(self, *, notify_on_failure: bool = True, advance_to_start: bool = False) -> bool:
    return use_z2_source_on_demand(self, notify_on_failure=notify_on_failure, advance_to_start=advance_to_start)

def _refresh_extract_workflow_ui(self):
    refresh_extract_workflow_ui(self)

def _schedule_extract_workflow_refresh(self, delay_ms: int = 0):
    schedule_extract_workflow_refresh(self, delay_ms=delay_ms)



_INSTANCE_METHODS = (
    "_set_widget_state",
    "_update_step3_source_path_lock",
    "_bind_source_path_watchers",
    "_on_source_path_var_changed",
    "_schedule_source_binding_refresh",
    "_normalize_xml_image_relpath",
    "_xml_relpath_to_path",
    "_read_xml_image_names",
    "_count_xml_plate_cut_targets",
    "_prepare_plate_cut_detections_for_source",
    "_backfill_preview_expected_texts_from_sources",
    "_format_extract_cut_plan",
    "_append_extract_cut_plan",
    "_evaluate_images_dir_for_xml",
    "_summarize_missing_xml_images",
    "_is_path_within",
    "_get_preferred_source_roots",
    "_is_recommended_images_dir",
    "_derive_candidate_root_for_match",
    "_find_matching_images_dir_for_xml",
    "_refresh_source_binding_status",
    "_count_preview_statuses",
    "_get_step3_chars_root_dir",
    "_get_step3_datasets_root_dir",
    "_get_step3_char_classification_datasets_root_dir",
    "_get_step3_summary_dir",
    "_build_step3_export_summary",
    "_write_step3_export_summary",
    "_read_step3_export_summary",
    "_inspect_yolo_dataset_label_objects",
    "_get_campaign_step3_annotation_readiness",
    "_return_step3_result_to_wizard",
    "_finalize_step3_from_existing_outputs",
    "_get_campaign_step3_training_readiness",
    "_get_step3_finish_block_message",
    "_set_step3_finish_hint",
    "_has_any_step3_export_outputs",
    "_return_to_t05_work_after_step3_pz1",
    "_return_to_wizard_for_step3_rework",
    "_resolve_step3_campaign_action_command",
    "_get_step3_pz3_path_selection_view_model",
    "_get_step3_pz3_dataset_mode_view_model",
    "_get_inline_status_widget_snapshot",
    "_get_step3_pz3_status_panel_view_model",
    "_refresh_pz3_status_panel_ui",
    "_schedule_extract_workflow_refresh",
    "_refresh_extract_workflow_ui",
    "_use_z2_source_on_demand",
    "_get_preferred_z2_source_candidate",
    "_pick_annotation_run_dir",
    "set_pending_z2_annotation_source",
    "_apply_annotation_run_source",
    "_get_annotation_run_input_dir",
    "_get_annotation_tab",
    "_get_registry_preferred_step3_source_candidate",
    "_get_campaign_iteration_artifact_bundle",
    "_load_annotation_run_manifest",
    "_annotation_run_manifest_path",
    "_resolve_existing_annotation_run_dir",
    "_refresh_extract_step_cards",
    "_refresh_extract_entry_cards",
    "_bind_extract_entry_card",
    "_refresh_extract_step_nav_buttons",
    "_get_step3_extract_workflow_view_model",
    "_go_to_next_extract_step",
    "_go_to_previous_extract_step",
    "_restore_step3_extract_state_from_project",
    "_sync_campaign_step3_preview_artifact_registry",
    "_sync_campaign_step3_artifact_registry",
    "_persist_step3_extract_state",
    "_handle_extract_entry_selection",
    "_set_extract_entry_mode",
    "_get_extract_entry_mode",
    "_set_extract_workflow_step",
    "_get_extract_workflow_step",
    "_coerce_extract_workflow_step",
    "_normalize_extract_entry_mode",
)


_STATIC_METHODS = (
    "_extract_source_plate_tokens_from_filename",
    "_plate_cut_reading_order_key",
    "_paths_equivalent",
    "_normalize_extract_workflow_step",
)


def bind_character_extract_delegates(tab_cls):
    for method_name in _INSTANCE_METHODS:
        setattr(tab_cls, method_name, globals()[method_name])
    for method_name in _STATIC_METHODS:
        setattr(tab_cls, method_name, staticmethod(globals()[method_name]))
    return tab_cls
