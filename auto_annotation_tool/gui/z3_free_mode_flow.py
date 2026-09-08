from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def handle_pz3_dataset_source_var_write(host: "CharacterAnnotationTab", *_args) -> None:
    try:
        raw_mode = str(host.pz3_dataset_source_mode_var.get() or "").strip().lower()
        normalized = "perfect"
        if raw_mode != normalized:
            host.pz3_dataset_source_mode_var.set(normalized)
            return
        host._save_local_setting("char_pz3_dataset_source_mode", normalized)
    except Exception:
        normalized = "perfect"

    refresh = getattr(host, "_refresh_pz3_dataset_mode_ui", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass


def set_pz3_dataset_source_mode(host: "CharacterAnnotationTab", mode: str) -> str:
    normalized = "perfect"

    current_mode = str(host.pz3_dataset_source_mode_var.get() or "").strip().lower()
    if current_mode != normalized:
        try:
            host.pz3_dataset_source_mode_var.set(normalized)
        except Exception:
            pass
    else:
        refresh = getattr(host, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
    return normalized


def set_pz3_selected_path(host: "CharacterAnnotationTab", path_key: str) -> str:
    normalized = str(path_key or "").strip().lower()
    current = str(getattr(host, "_pz3_selected_path", "") or "").strip().lower()
    if normalized not in {"dataset", "cvat"}:
        normalized = ""
    host._pz3_selected_path = "" if current == normalized else normalized
    host._pz3_cvat_expanded = host._pz3_selected_path == "cvat"

    refresh = getattr(host, "_refresh_pz3_cards_ui", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass
    return str(getattr(host, "_pz3_selected_path", "") or "").strip().lower()


def clear_step3_free_mode_campaign_markers(host: "CharacterAnnotationTab") -> None:
    """Usuń wyłącznie lokalne markery wejścia z grafu; nie czyści danych użytkownika."""
    for attr_name, value in (
        ("_campaign_graph_entry_context", {}),
        ("_campaign_force_pz2_entry", False),
        ("_campaign_force_pz3_entry", False),
        ("_campaign_force_detect_entry", False),
        ("_campaign_pz2_sync_loading", False),
        ("_campaign_step3_entry_splash_pinned", False),
        ("_campaign_detect_splash_force_root_surface", False),
        ("_campaign_step3_hold_pz2_after_reextract", False),
        ("_campaign_step3_reextract_seed_metadata", {}),
    ):
        try:
            setattr(host, attr_name, value.copy() if isinstance(value, dict) else value)
        except Exception:
            pass


def ensure_step3_free_mode_context(host: "CharacterAnnotationTab") -> None:
    clear_step3_free_mode_campaign_markers(host)
    host._step3_linear_mode = False
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    for tab_widget in (
        getattr(host, "tab_extract", None),
        getattr(host, "tab_detect", None),
        getattr(host, "tab_dataset", None),
    ):
        if tab_widget is None:
            continue
        try:
            host._set_subtab_state(tab_widget, "normal")
        except Exception:
            pass
    for button_name in (
        "btn_to_detect",
        "btn_to_dataset",
        "btn_run_detection",
        "btn_rank_presets",
        "btn_ocr_lab",
    ):
        try:
            host._set_button_state(button_name, True)
        except Exception:
            pass
    for frame_name in (
        "btn_to_detect_frame",
        "btn_to_dataset_frame",
        "btn_run_detection_frame",
    ):
        try:
            host._set_button_emphasis(frame_name, False)
        except Exception:
            pass
    try:
        host._refresh_step3_mode_specific_ui()
    except Exception:
        pass
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass
    try:
        host._sync_step3_nav_buttons()
    except Exception:
        pass


def reset_step3_subtab_flow(host: "CharacterAnnotationTab") -> None:
    clear_step3_free_mode_campaign_markers(host)
    host._step3_linear_mode = False
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass

    host._set_subtab_state(host.tab_extract, "normal")
    host._set_subtab_state(host.tab_detect, "normal")
    host._set_subtab_state(host.tab_dataset, "normal")

    host._set_button_state("btn_to_detect", True)
    host._set_button_state("btn_to_dataset", True)
    host._set_button_state("btn_run_detection", True)
    host._set_button_state("btn_rank_presets", True)
    host._set_button_state("btn_ocr_lab", True)

    host._set_button_emphasis("btn_to_detect_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass
    host._set_button_emphasis("btn_run_detection_frame", False)
    if (getattr(host, "_startup_ui_ready", None) is False
            and not getattr(host, "_detect_tab_built", False)
            and not getattr(host, "_dataset_tab_built", False)):
        # Initial construction only creates PZ1. Do not load a previous
        # campaign's large preview to paint controls that do not exist yet.
        return

    try:
        host._update_preview_path_lock()
    except Exception:
        pass

    try:
        host._sync_step3_access_from_preview_state()
    except Exception:
        pass

    try:
        host._update_step3_source_path_lock()
    except Exception:
        pass

    try:
        host._update_yolo_visibility()
    except Exception:
        pass

    try:
        host._update_step3_finish_button_state()
    except Exception:
        pass

    try:
        host.test_progress.config(value=0)
        host._set_test_progress_counter()
    except Exception:
        pass

    try:
        host._set_test_status(host._compose_detection_method_status("gotowa do uruchomienia"), "neutral")
    except Exception:
        pass

    try:
        host.fast_test_running = False
        host.fast_test_stop.clear()
    except Exception:
        pass

    try:
        host._reset_pz3_runtime_ui(collapse_cards=True)
    except Exception:
        pass

    try:
        host._refresh_step3_mode_specific_ui()
    except Exception:
        pass

    try:
        schedule_refresh = getattr(host, "_schedule_extract_workflow_refresh", None)
        if callable(schedule_refresh):
            schedule_refresh(delay_ms=40 if not host.is_startup_ui_ready() else 0)
        else:
            host._refresh_extract_workflow_ui()
    except Exception:
        pass


def set_extract_entry_mode(host: "CharacterAnnotationTab", mode: str, *, persist: bool = True) -> None:
    normalized = host._normalize_extract_entry_mode(mode)
    try:
        if host.extract_entry_mode_var.get() != normalized:
            host.extract_entry_mode_var.set(normalized)
    except Exception:
        pass

    host._set_extract_workflow_step("source" if normalized else "entry", refresh=False)
    if persist:
        try:
            host._persist_step3_extract_state()
        except Exception:
            pass
    host._refresh_extract_workflow_ui()


def reset_extract_source_inputs(host: "CharacterAnnotationTab") -> None:
    host._source_binding_sync_in_progress = True
    try:
        host.annotation_run_dir_var.set("")
        host.xml_path_var.set("")
        host.images_dir_var.set("")
        host.preview_dir_var.set("")
    finally:
        host._source_binding_sync_in_progress = False

    host._extract_last_source_binding_result = {"ok": False}

    try:
        host._reset_preview_cache()
    except Exception:
        pass

    try:
        host.ext_progress.config(value=0)
    except Exception:
        pass

    try:
        host._set_extraction_status("Gotowy", "neutral")
    except Exception:
        pass

    try:
        host._set_source_binding_status("", "warning")
    except Exception:
        pass

    try:
        host._force_save_all()
    except Exception:
        pass

    try:
        host._persist_step3_extract_state()
    except Exception:
        pass


def handle_extract_entry_selection(host: "CharacterAnnotationTab", mode: str) -> None:
    selected_mode = host._normalize_extract_entry_mode(mode)
    current_mode = host._get_extract_entry_mode()
    if selected_mode == "continue":
        if current_mode != "continue":
            reset_extract_source_inputs(host)
        set_extract_entry_mode(host, "continue")
        try:
            host._use_z2_source_on_demand(notify_on_failure=False, advance_to_start=False)
        except Exception:
            pass
        try:
            host._set_extract_workflow_step("start")
        except Exception:
            pass
        return

    reset_extract_source_inputs(host)
    set_extract_entry_mode(host, "manual")


def go_to_previous_extract_step(host: "CharacterAnnotationTab") -> None:
    current = host._get_extract_workflow_step()
    if current == "start":
        host._set_extract_workflow_step("source")
    elif current == "source":
        host._set_extract_workflow_step("entry")


def go_to_next_extract_step(host: "CharacterAnnotationTab") -> None:
    current = host._get_extract_workflow_step()
    route = host._get_extract_entry_mode()
    has_preview = bool(str(host.preview_dir_var.get() or "").strip())

    if current == "entry":
        if route in {"continue", "manual"}:
            host._set_extract_workflow_step("source")
        return

    if current != "source":
        return

    validation = host._refresh_source_binding_status(allow_autofind=True)
    if validation.get("ok") or has_preview:
        host._set_extract_workflow_step("start")


def go_to_substep_2_free_mode(host: "CharacterAnnotationTab") -> None:
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass

    opened = False
    opener = getattr(host, "_open_detection_subtab_with_preview", None)
    if callable(opener):
        try:
            opened = bool(opener(force_reload=True))
        except Exception:
            opened = False
    if not opened:
        host._select_subtab(host.tab_detect)
    try:
        host._persist_step3_progress()
    except Exception:
        pass
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass


def go_to_substep_3_free_mode(host: "CharacterAnnotationTab") -> None:
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass

    host._select_subtab(host.tab_dataset)
    try:
        host._persist_step3_progress()
    except Exception:
        pass
    host._set_button_emphasis("btn_run_detection_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass


def back_to_substep_1_free_mode(host: "CharacterAnnotationTab") -> None:
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass

    host._select_subtab(host.tab_extract)
    try:
        host._persist_step3_progress()
    except Exception:
        pass


def back_to_substep_2_free_mode(host: "CharacterAnnotationTab") -> None:
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass

    host._select_subtab(host.tab_detect)
    try:
        host._persist_step3_progress()
    except Exception:
        pass
