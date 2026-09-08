from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING, Any

from ..config import SESSION
from ..project_cache import PROJECT_CACHE
from .z2_flow_models import (
    Z2CopyPayload,
    Z2CtaState,
    Z2FreeModeRuntimeState,
    Z2LayoutState,
    Z2LeftPanelCopyContext,
)

if TYPE_CHECKING:
    from .tab_annotation import AnnotationTab


AUTO_REVIEW_FOLLOWUP_TITLE = "Korekta i decyzja po autoanotacji"
AUTO_REVIEW_FOLLOWUP_TEXT = (
    "Masz gotowy wynik autoanotacji zapisany w runie Z2. Najpierw sprawdź ramki lub poligony tablic, "
    "popraw błędy i zatwierdź poprawne obrazy statusem [OK]. Status [OK] jest wymagany dla "
    "„Wyodrębnij tablice” oraz dla eksportu datasetu YOLO Pose ze splitem, bo obie ścieżki "
    "operują na danych treningowych. Wyjątkiem jest eksport samych anotacji XML: możesz go wykonać "
    "po zapisaniu co najmniej jednej tablicy, nawet bez statusu [OK]. Run Z2 pozostaje zapisany, "
    "więc możesz wrócić do niego z historii runów bez utraty pracy."
)
MANUAL_REVIEW_FOLLOWUP_TITLE = "Korekta i decyzja po anotacji ręcznej"
MANUAL_REVIEW_FOLLOWUP_TEXT = (
    "Masz gotowy ręczny run anotacji tablic zapisany do pliku xml. Jeśli chcesz przejść do pracy "
    "nad znakami, wybierz „Wyodrębnij tablice”. Jeśli chcesz trenować model tablic na tym etapie, "
    "otwórz eksport i wybierz wariant datasetu YOLO Pose ze splitem. Run Z2 pozostaje zapisany, "
    "więc później możesz wrócić do niego z historii runów bez utraty pracy. Status [OK] jest "
    "wymagany dla wyodrębniania tablic i datasetu YOLO Pose. Eksport samych anotacji XML jest "
    "wyjątkiem: wystarczy co najmniej jedna zapisana tablica, nawet bez statusu [OK]."
)


def select_free_mode_route(host: "AnnotationTab", route: str) -> None:
    normalized_route = host._normalize_workflow_route_value(route)
    if not normalized_route:
        return

    host.workflow_route_var.set(normalized_route)
    host.free_mode_screen_var.set("workflow")
    host._manual_review_active = False
    host._manual_review_from_auto = False
    host._manual_review_origin_route = ""
    host._manual_review_export_ready = False
    host._dataset_export_completed = False
    host._last_completed_workflow_route = ""

    if normalized_route == "auto":
        host._clear_active_annotation_run_context(preserve_input_dir=False)
        try:
            host.input_dir_var.set("")
        except Exception:
            pass
        try:
            host.plate_dataset_images_var.set("")
        except Exception:
            pass
        try:
            host.current_input_dir = None
        except Exception:
            pass
        host.manual_xml_template_var.set(False)
        host.auto_vehicle_choice_var.set(host._get_auto_vehicle_choice())
        host._auto_route_settings_pending = True
        default_step = "auto_input"
    else:
        host._clear_active_annotation_run_context(preserve_input_dir=False)
        try:
            host.input_dir_var.set("")
        except Exception:
            pass
        try:
            host.plate_dataset_images_var.set("")
        except Exception:
            pass
        try:
            host.current_input_dir = None
        except Exception:
            pass
        host._auto_route_settings_pending = False
        default_entry_mode = "new"
        host.manual_entry_mode_var.set(default_entry_mode)
        host.manual_xml_template_var.set(default_entry_mode == "new")
        host.manual_history_run_var.set("")
        host.manual_vehicle_assist_var.set(False)
        default_step = "manual_entry"
    host.workflow_step_var.set(default_step)
    host._refresh_left_panel_route_copy()
    host._refresh_detection_configuration_ui()
    if normalized_route == "manual":
        host._refresh_manual_review_history_ui()
    host._refresh_step2_action_states()
    host._refresh_free_mode_workflow_ui()
    host._schedule_left_panel_scroll_to_widget(
        getattr(
            host,
            (
                "workflow_input_section"
                if normalized_route == "auto"
                else "manual_entry_section"
            ),
            None,
        )
        or getattr(host, "actions_section", None),
        delay_ms=0,
    )
    host._queue_free_mode_session_save()


def set_manual_entry_mode(host: "AnnotationTab", mode: str) -> None:
    previous_mode = host._get_manual_entry_mode()
    normalized_mode = host._normalize_manual_entry_mode(mode)
    host.workflow_route_var.set("manual")
    host.free_mode_screen_var.set("workflow")
    host.manual_entry_mode_var.set(normalized_mode)
    host.manual_xml_template_var.set(normalized_mode == "new")
    if normalized_mode in {"continue", "import"} and not host._manual_review_active:
        host.manual_history_run_var.set("")
    elif (
        normalized_mode == "new"
        and previous_mode != "new"
        and not host._manual_review_active
        and not host._manual_review_from_auto
        and not host.is_processing
    ):
        host.input_dir_var.set("")
        try:
            host.plate_dataset_images_var.set("")
        except Exception:
            pass
        try:
            host.current_input_dir = None
        except Exception:
            pass
    host._manual_review_active = False
    host._manual_review_from_auto = False
    host._manual_review_origin_route = ""
    host._manual_review_export_ready = False
    host._dataset_export_completed = False
    if normalized_mode != "new":
        host.manual_vehicle_assist_var.set(False)
    else:
        host.manual_vehicle_assist_var.set(False)
        host._clear_active_annotation_run_context(preserve_input_dir=False)
    host.workflow_step_var.set("manual_entry")
    host._refresh_left_panel_route_copy()
    host._refresh_detection_configuration_ui()
    host._refresh_manual_review_history_ui()
    host._refresh_step2_action_states()
    host._refresh_free_mode_workflow_ui()
    host._schedule_left_panel_scroll_to_widget(
        getattr(host, "manual_entry_section", None) or getattr(host, "actions_section", None),
        delay_ms=0,
    )
    host._queue_free_mode_session_save()


def prepare_free_mode_workflow_runtime(host: "AnnotationTab") -> Z2FreeModeRuntimeState:
    free_mode_screen = host._coerce_free_mode_screen()
    if host.free_mode_screen_var.get() != free_mode_screen:
        host.free_mode_screen_var.set(free_mode_screen)
    return Z2FreeModeRuntimeState(
        free_mode_screen=free_mode_screen,
        available_primary_action_ids=[],
    )


def jump_to_export_section(host: "AnnotationTab") -> None:
    campaign_context = not host._is_free_mode_session_context()
    free_mode_screen = host._coerce_free_mode_screen() if host._is_free_mode_session_context() else ""
    should_save_preview_edits = bool(
        host._manual_review_active
        or (
            not campaign_context
            and free_mode_screen in {"auto_summary", "manual_review"}
            and host._get_preferred_annotation_run_dir(require_xml=True) is not None
        )
    )

    if should_save_preview_edits and not host._ensure_preview_edits_saved("przejscie do splitu i eksportu"):
        return

    if not campaign_context:
        run_dir = host._get_preferred_annotation_run_dir(require_xml=True)
        try:
            approval_state = host._get_run_plate_strict_approved_state(run_dir)
        except Exception:
            approval_state = {}
        if not approval_state.get("ok"):
            host._warn_plate_dataset_export_requires_ok(approval_state)
            return

    host._dataset_export_completed = False
    host._manual_review_export_ready = True
    if not campaign_context:
        host.free_mode_screen_var.set("export")
    host._refresh_plate_dataset_export_sources()
    host._refresh_free_mode_workflow_ui()
    host._scroll_left_panel_to_widget(getattr(host, "export_section", None))


def open_existing_run_for_manual_review(
    host: "AnnotationTab",
    run_dir: Path | None = None,
    *,
    allow_fallback: bool = True,
    from_auto: bool = False,
    show_dialog: bool = True,
    entry_mode: str | None = None,
) -> bool:
    target_run_dir = host._resolve_safe_annotation_run_dir(
        run_dir if run_dir is not None else (
            host._get_preferred_annotation_run_dir(require_xml=True)
            if allow_fallback
            else None
        ),
        require_xml=True,
    )
    if target_run_dir is None:
        messagebox.showwarning(
            "Brak runu anotacji do korekty",
            "Nie znaleziono jeszcze zadnego runu Z2 z annotations.xml do kontynuacji."
        )
        return False

    if not host._restore_preview_from_annotation_run(target_run_dir):
        messagebox.showerror(
            "Błąd podglądu",
            f"Nie udalo sie otworzyc runu anotacji do korekty:\n{target_run_dir}"
        )
        return False

    normalized_entry_mode = (
        "continue"
        if from_auto
        else host._normalize_manual_entry_mode(entry_mode or host._get_manual_entry_mode())
    )
    if normalized_entry_mode not in {"continue", "import"}:
        normalized_entry_mode = "continue"

    host.workflow_route_var.set("manual")
    host.manual_entry_mode_var.set(normalized_entry_mode)
    host.workflow_step_var.set("manual_history" if normalized_entry_mode == "continue" else "manual_entry")
    host.free_mode_screen_var.set("manual_review")
    host.manual_xml_template_var.set(False)
    host.manual_vehicle_assist_var.set(False)
    host._manual_review_active = True
    host._manual_review_from_auto = bool(from_auto)
    host._manual_review_origin_route = ("auto" if from_auto else "manual")
    host._manual_review_export_ready = False
    host._dataset_export_completed = False
    host._last_completed_workflow_route = ("auto" if from_auto else "manual")

    if host.current_input_dir is not None:
        host.input_dir_var.set(str(host.current_input_dir))

    host._refresh_left_panel_route_copy()
    host._refresh_detection_configuration_ui()
    host._refresh_run_output_info()
    host._refresh_step2_action_states()
    host._refresh_free_mode_workflow_ui()
    host._remember_manual_review_run(
        target_run_dir,
        source=("auto" if from_auto else "manual"),
        created_at=str(
            host._load_annotation_run_manifest(target_run_dir).get("generated_at")
            or host._load_annotation_run_manifest(target_run_dir).get("completed_at")
            or ""
        ).strip(),
    )
    host._queue_free_mode_session_save()

    if from_auto:
        host._set_post_annotation_hint(
            "Jestes w torze recznej korekty. Del usuwa zdjecie z dysku i z XML, a poprawki polygonow sa od razu zapisywane.",
            "muted",
        )

    if show_dialog:
        messagebox.showinfo(
            "Korekta ręczna",
            f"Otworzono run anotacji do korekty recznej:\n{target_run_dir}"
        )
    return True


def get_manual_review_history_display_entries(host: "AnnotationTab") -> list[dict[str, Any]]:
    collected_entries = list(host._manual_review_history_entries or [])

    try:
        run_roots = host._dedupe_paths(host._get_annotation_run_roots())
    except Exception:
        run_roots = []

    for root in run_roots:
        try:
            run_dirs = PROJECT_CACHE.list_annotation_run_dirs(root, require_xml=True)
        except Exception:
            continue

        for run_dir in run_dirs:
            safe_run_dir = host._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
            if safe_run_dir is None:
                continue

            try:
                manifest = host._load_annotation_run_manifest(safe_run_dir)
            except Exception:
                manifest = {}

            collected_entries.append(
                {
                    "run_dir": str(safe_run_dir),
                    "created_at": host._resolve_manual_review_history_created_at(
                        {
                            "created_at": (
                                str(manifest.get("last_manual_edit_at") or "").strip()
                                or str(manifest.get("resume_preview_saved_at") or "").strip()
                                or str(manifest.get("completed_at") or "").strip()
                                or str(manifest.get("generated_at") or "").strip()
                            )
                        },
                        safe_run_dir,
                    ),
                    "source": str(manifest.get("annotation_run_type") or "").strip(),
                }
            )

    return host._normalize_manual_review_history_entries(collected_entries)


def refresh_manual_review_history_ui(host: "AnnotationTab") -> None:
    if host._is_free_mode_session_context() and host._get_manual_entry_mode() != "continue":
        return
    values = []
    label_map = {}

    display_entries = get_manual_review_history_display_entries(host)

    for entry in display_entries:
        label = host._format_manual_review_history_label(entry)
        if not label:
            continue
        values.append(label)
        label_map[label] = str(entry.get("run_dir") or "").strip()

    host._manual_review_history_label_map = label_map

    combo = getattr(host, "manual_history_combo", None)
    if combo is not None:
        try:
            combo.configure(values=values)
        except Exception:
            pass

    current_value = str(host.manual_history_run_var.get() or "").strip()
    if current_value not in label_map:
        preferred_value = ""
        for candidate_run in (
            getattr(host, "current_annotation_run_dir", None),
            getattr(host, "last_staging_run_dir", None),
            str(host.plate_dataset_run_var.get() or "").strip(),
        ):
            current_run = host._resolve_safe_annotation_run_dir(candidate_run, require_xml=True)
            if current_run is None:
                continue
            for label, run_dir in label_map.items():
                if run_dir == str(current_run):
                    preferred_value = label
                    break
            if preferred_value:
                break
        if not preferred_value and values and host._get_manual_entry_mode() == "continue":
            preferred_value = values[0]
        host.manual_history_run_var.set(preferred_value)

    hint_text = ""
    run_definition = host._get_annotation_run_definition_text()
    default_storage = host._get_annotation_run_storage_display_path()
    try:
        prefix_lookup = host._get_z2_thematic_title_prefixes()
        host.manual_history_title_lbl.configure(
            text=host._format_z2_thematic_title(
                "Historia runów Z2",
                prefix_lookup.get("manual_history"),
            )
        )
    except Exception:
        pass
    if values and host._manual_review_active and host._get_manual_entry_mode() == "continue":
        hint_text = (
            "Aktywny run jest już otwarty w podglądzie. Na karcie korekty zdecydujesz, czy wyciąć tablice do Z3, "
            "czy przejść do splitu i eksportu datasetu. "
            "Możesz też zaznaczyć inny run i otworzyć go ponownie.\n\n"
            f"{run_definition}\n\n"
            f"Domyślny katalog runów Z2: {default_storage}"
        )
    elif values:
        hint_text = (
            "Wybierz run Z2 z historii i kliknij Dalej, aby otworzyć go do korekty. "
            "Jeśli szukanego runu tu nie ma, wróć o krok i wybierz wskazanie dowolnego runu Z2.\n\n"
            f"{run_definition}\n\n"
            f"Domyślny katalog runów Z2: {default_storage}"
        )
    elif host._get_manual_entry_mode() == "continue":
        hint_text = (
            "Nie znaleziono jeszcze lokalnych runów Z2 do wznowienia. Wróć o krok i utwórz nowy run ręczny "
            "albo wskaż dowolny run Z2.\n\n"
            f"{run_definition}\n\n"
            f"Domyślny katalog runów Z2: {default_storage}"
        )
    elif host._manual_review_active or host._manual_review_from_auto:
        hint_text = "Brak zapisanej historii korekt dla aktualnego runu Z2."
    host._set_inline_label_state(
        getattr(host, "manual_history_hint_lbl", None),
        text=hint_text,
        tone=("muted" if values else "warning" if host._get_manual_entry_mode() == "continue" and hint_text else "muted"),
        emphasis=False,
    )

    try:
        host.manual_history_open_btn.configure(
            state=(tk.NORMAL if bool(str(host.manual_history_run_var.get() or "").strip()) else tk.DISABLED)
        )
    except Exception:
        pass


def remember_manual_review_run(
    host: "AnnotationTab",
    run_dir: Path | str | None,
    *,
    source: str = "manual",
    created_at: str | None = None,
) -> None:
    safe_run_dir = host._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe_run_dir is None:
        return

    timestamp = str(created_at or "").strip() or host._resolve_manual_review_history_created_at({}, safe_run_dir)
    updated_entries = [
        {
            "run_dir": str(safe_run_dir),
            "created_at": timestamp,
            "source": str(source or "").strip(),
        }
    ]

    for entry in host._manual_review_history_entries or []:
        if str(entry.get("run_dir") or "").strip() == str(safe_run_dir):
            continue
        updated_entries.append(entry)

    host._manual_review_history_entries = host._normalize_manual_review_history_entries(updated_entries)
    refresh_manual_review_history_ui(host)
    try:
        safe_run_dir_text = str(safe_run_dir)
        for label, run_dir_text in host._manual_review_history_label_map.items():
            if run_dir_text == safe_run_dir_text:
                host.manual_history_run_var.set(label)
                break
    except Exception:
        pass
    host._queue_free_mode_session_save()
    if host._is_free_mode_session_context():
        try:
            if SESSION:
                SESSION.set("annotation", "manual_review_history", list(host._manual_review_history_entries or []))
                SESSION.set("annotation", "manual_review_active", bool(host._manual_review_active))
                SESSION.set("annotation", "manual_review_from_auto", bool(host._manual_review_from_auto))
                SESSION.set("annotation", "manual_review_export_ready", bool(host._manual_review_export_ready))
                SESSION.set("annotation", "workflow_route", host._normalize_workflow_route_value())
                SESSION.set("annotation", "manual_entry_mode", host._normalize_manual_entry_mode())
                SESSION.set("annotation", "workflow_step", host._get_workflow_step())
                SESSION.set("annotation", "plate_dataset_run", str(safe_run_dir))
                SESSION.set("annotation", "last_preview_run_dir", str(safe_run_dir))
                SESSION.save_session()
            host.flush_free_mode_session_state()
        except Exception:
            pass


def build_z2_layout_state_free_mode(
    host: "AnnotationTab",
    *,
    free_mode_screen: str,
    route: str,
    manual_review_active: bool,
) -> Z2LayoutState:
    show_route_choice = free_mode_screen == "route_choice"
    show_export_followup = free_mode_screen == "export"
    show_auto_followup = free_mode_screen == "auto_summary"
    show_manual_review_followup = bool(
        free_mode_screen == "manual_review" and manual_review_active
    )
    show_stage_export_cta = False
    compact_export_followup = bool(show_export_followup)
    show_workflow_steps = bool(free_mode_screen == "workflow" and route)
    compact_single_route_layout = False
    compact_left_column_layout = False
    show_nav_panel = bool(
        route
        and (
            show_workflow_steps
            or show_auto_followup
            or show_export_followup
            or show_manual_review_followup
        )
    )
    workflow_route_panel = bool(show_workflow_steps and route in {"auto", "manual"})
    current_step = ""
    try:
        current_step = str(host._coerce_workflow_step() or "").strip().lower()
    except Exception:
        current_step = ""
    try:
        has_active_annotation_run = bool(host._get_preferred_annotation_run_dir(require_xml=True) is not None)
    except Exception:
        has_active_annotation_run = False
    has_loaded_input_workspace = False
    try:
        has_loaded_input_workspace = bool(getattr(host, "current_annotations", None))
    except Exception:
        has_loaded_input_workspace = False
    if not has_loaded_input_workspace:
        try:
            has_loaded_input_workspace = bool(str(host.input_dir_var.get() or "").strip())
        except Exception:
            has_loaded_input_workspace = False
    show_free_mode_right_panel = bool(
        route
        and (
            (
                show_workflow_steps
                and current_step in {"auto_start", "manual_start"}
                and (has_active_annotation_run or has_loaded_input_workspace)
            )
            or (
                has_active_annotation_run
                and (
                show_auto_followup
                or show_export_followup
                or show_manual_review_followup
                )
            )
        )
    )
    return Z2LayoutState(
        show_route_choice=show_route_choice,
        show_export_followup=show_export_followup,
        show_auto_followup=show_auto_followup,
        show_manual_review_followup=show_manual_review_followup,
        show_stage_export_cta=show_stage_export_cta,
        compact_export_followup=compact_export_followup,
        show_workflow_steps=show_workflow_steps,
        compact_single_route_layout=compact_single_route_layout,
        compact_left_column_layout=compact_left_column_layout,
        show_campaign_context_header=False,
        show_nav_panel=show_nav_panel,
        show_right_panel=show_free_mode_right_panel,
    )


def build_z2_cta_state_free_mode(
    host: "AnnotationTab",
    *,
    route: str,
    current_step: str,
    free_mode_screen: str,
    show_workflow_steps: bool,
    has_existing_run: bool,
    input_dir_ready: bool,
    auto_setup_pending: bool,
    auto_vehicle_choice: str,
    manual_setup: bool,
    manual_run_already_created: bool,
) -> Z2CtaState:
    def _has_approved_plate_positions() -> bool:
        if not has_existing_run:
            return False
        try:
            return bool(host._get_plate_dataset_export_approval_state().get("ok"))
        except Exception:
            return False

    show_start_controls = bool(
        show_workflow_steps and current_step in {"auto_start", "manual_start"}
    )
    show_nav_controls = bool(show_workflow_steps and current_step and not host.is_processing)

    start_enabled = not host.is_processing and bool(route)
    start_command = host._start_annotation
    start_text = "Wybierz tor"
    if route == "auto":
        if not input_dir_ready:
            start_enabled = not host.is_processing
            start_text = "Wybierz obrazy do autoanotacji" if auto_setup_pending else "Wybierz obrazy"
            start_command = host._select_input_dir
        elif auto_setup_pending:
            start_text = "Start autoanotacji"
        elif auto_vehicle_choice == "skip":
            start_text = "Start autoanotacji tablic"
        else:
            start_text = "Start autoanotacji tablic i pojazdów"
    if route == "auto" and input_dir_ready:
        start_text = "Start autoanotacji"
    elif manual_setup:
        if manual_run_already_created:
            start_enabled = False
            start_text = "Run istnieje"
        elif not input_dir_ready:
            start_enabled = not host.is_processing
            start_text = "Wybierz obrazy"
            start_command = host._select_input_dir
        else:
            start_text = (
                "Utwórz XML + boxy pojazdów"
                if host._manual_vehicle_assist_enabled()
                else "Utwórz XML anotacji"
            )

    back_text = "Wstecz"
    next_text = "Dalej"
    if free_mode_screen == "workflow":
        back_enabled = bool(
            not host.is_processing
            and (
                host._manual_review_export_ready
                or host._manual_review_active
                or bool(route)
            )
        )
        next_enabled = bool(host._dataset_export_completed) or bool(
            show_nav_controls and (not show_start_controls) and host._is_workflow_step_complete(current_step)
        )
        if (
            route == "auto"
            and current_step == "auto_start"
            and has_existing_run
            and host._is_z2_auto_flow_completed(route=route, has_existing_run=has_existing_run)
        ):
            next_enabled = bool(not host.is_processing)
        if (
            route == "manual"
            and current_step == "manual_start"
            and manual_setup
            and bool(getattr(host, "_manual_template_ready_for_review", False))
        ):
            next_enabled = bool(not host.is_processing)
            next_text = "Dalej"
        if host._dataset_export_completed:
            next_text = "Powrot"
        if route == "manual" and current_step == "manual_entry" and not host._dataset_export_completed:
            next_text = "Dalej"
        elif route == "manual" and current_step == "manual_history" and not host._dataset_export_completed:
            next_text = "Dalej" if host._manual_review_active else "Otworz run"
    elif free_mode_screen == "auto_summary":
        try:
            export_state = host._build_z2_free_export_status_state()
            export_choice_ready = bool(export_state.get("dataset_ready") or export_state.get("annotation_ready"))
        except Exception:
            try:
                export_choice_ready = bool(host._is_z2_free_export_choice_available())
            except Exception:
                export_choice_ready = False
        back_enabled = bool(not host.is_processing)
        next_enabled = bool(not host.is_processing and export_choice_ready)
        back_text = "Wstecz"
        next_text = "Otwórz eksport"
    elif free_mode_screen == "manual_review":
        approved_ready = _has_approved_plate_positions()
        if route == "manual":
            back_enabled = bool(not host.is_processing)
            next_enabled = False
            back_text = "Wstecz"
            next_text = ""
        else:
            back_enabled = bool(not host.is_processing and has_existing_run)
            next_enabled = bool(not host.is_processing and approved_ready)
            next_text = "Dalej do Z3/PZ1"
    elif free_mode_screen == "export":
        back_enabled = bool(not host.is_processing)
        next_enabled = False
        next_text = ""
    else:
        back_enabled = False
        next_enabled = False

    return Z2CtaState(
        show_start_controls=show_start_controls,
        show_nav_controls=show_nav_controls,
        start_enabled=start_enabled,
        start_command=start_command,
        start_text=start_text,
        back_enabled=back_enabled,
        back_text=back_text,
        next_enabled=next_enabled,
        next_text=next_text,
        suppress_duplicate_start_cta=False,
    )


def build_z2_left_panel_copy_payload_free_mode(
    host: "AnnotationTab",
    ctx: Z2LeftPanelCopyContext,
    payload: Z2CopyPayload,
) -> Z2CopyPayload:
    route = str(ctx.route or "")
    manual_entry_mode = str(ctx.manual_entry_mode or "")
    current_step = str(ctx.current_step or "")
    vehicle_assist_enabled = bool(ctx.vehicle_assist_enabled)
    auto_vehicle_choice = str(ctx.auto_vehicle_choice or "")
    has_manual_history = bool(ctx.has_manual_history)
    auto_completed = bool(ctx.auto_completed)
    auto_setup_pending = bool(ctx.auto_setup_pending)
    input_dir_ready = bool(
        host._annotation_input_dir_ready(str(host.input_dir_var.get() or "").strip())
    )

    if route == "auto":
        payload["run_title"] = (
            "Katalog obraz\u00f3w wej\u015bciowych"
            if current_step == "auto_input"
            else "Autoanotacja tablic"
        )
        payload["badge_text"] = "Aktywny tor: autoanotacja tablic"
        payload["badge_tone"] = "success"
        payload["route_tone"] = "muted"
        payload["workflow_conf_title"] = "Ustaw pewność detekcji"
        payload["workflow_conf_hint"] = (
            "Ten próg dotyczy bieżącego runu anotacji Z2. Po wyborze modelu tablic możesz od razu go dopasować."
        )
        payload["workflow_vehicle_title"] = "Wskaż model pojazdów (YOLO Box)"
        payload["workflow_vehicle_hint"] = (
            "Ten model jest opcjonalny. Jeśli go pominiesz, run autoanotacji Z2 wykona tylko autoanotację tablic."
        )
        payload["workflow_input_title"] = "Wskaż folder obrazów"
        payload["workflow_input_hint"] = (
            "Wybierz folder obrazów. Model tablic, confidence i opcjonalne boxy pojazdów ustawisz przy starcie autoanotacji w modalu."
        )
        payload["workflow_start_title"] = "Uruchom proces autoanotacji"
        payload["workflow_start_intro"] = (
            "Uruchom model na bieżącym katalogu obrazów. Przy starcie wybierzesz w modalu zakres, model tablic, confidence i ewentualne boxy pojazdów, a po zakończeniu sprawdzisz wynik runu Z2."
        )
        if auto_setup_pending:
            payload["workflow_start_title"] = "Ustawienia autoanotacji"
            payload["route_text"] = (
                "Najpierw przygotujesz ustawienia tego runu autoanotacji. "
                "Model tablic, confidence i opcjonalne boxy pojazdów wybierzesz w modalu startu."
            )
            if str(host.input_dir_var.get() or "").strip():
                payload["action_text"] = (
                    "Folder obrazów jest już wskazany. Kliknij Start autoanotacji, aby otworzyć modal ustawień i potwierdzić bieżący run Z2."
                )
            else:
                payload["action_text"] = (
                    "Najpierw wskaż folder obrazów dla tego runu. Potem kliknij Start autoanotacji, aby otworzyć modal ustawień."
                )
            payload["workflow_start_intro"] = (
                "To jest krok przygotowania startu. Sam wybór modelu i pozostałych parametrów wykonasz w modalu autoanotacji."
            )
        if (not auto_setup_pending) and not bool(str(host.plate_custom_var.get() or "").strip()):
            payload["route_text"] = "Model tablic wybierzesz przy starcie autoanotacji."
            payload["action_text"] = "Kliknij Start, a w modalu wskażesz zakres pracy i model dla bieżącego runu Z2."
            payload["workflow_start_intro"] = (
                "Start otworzy modal ustawień bieżącego runu Z2. Wybierzesz tam zakres obrazów, wymagany model tablic, "
                "confidence oraz opcjonalne wsparcie modelem pojazdów. Po zatwierdzeniu modala program uruchomi autoanotację, "
                "a wynik sprawdzisz i poprawisz na liście oraz podglądzie Z2."
            )
        elif auto_vehicle_choice == "skip":
            payload["route_text"] = "Domyślnie pomijasz boxowanie pojazdów i uruchomisz tylko autoanotację tablic."
            payload["action_text"] = "Jeśli chcesz dodać pojazdy, odznacz pole pomijania. W następnym kroku wybierzesz wtedy model pojazdów."
            payload["auto_choice_hint"] = "Zaznaczone pole oznacza wariant: tylko tablice."
        else:
            payload["route_text"] = "Run anotacji Z2 zostanie wykonany dla tablic i pojazdów."
            payload["action_text"] = "W następnym kroku wybierzesz model pojazdów, potem wskażesz folder z obrazami i uruchomisz autoanotację tablic + pojazdów."
            payload["auto_choice_hint"] = "Odznaczone pole oznacza wariant: pojazdy + tablice."
        if auto_setup_pending:
            payload["workflow_start_title"] = "Ustawienia autoanotacji"
            if input_dir_ready:
                payload["route_text"] = (
                    "Źródło pracy jest już wskazane. Teraz przechodzisz do ustawień bieżącego runu autoanotacji Z2."
                )
                payload["action_text"] = (
                    "Kliknij Start autoanotacji, aby otworzyć modal i wybrać model tablic, confidence oraz opcjonalne boxy pojazdów."
                )
                payload["workflow_start_intro"] = (
                    "Ten krok dotyczy już ustawień runu. Obrazy są gotowe, więc teraz potwierdzisz parametry autoanotacji w modalu startu."
                )
            else:
                payload["route_text"] = (
                    "Najpierw wybierasz źródło pracy dla tego runu Z2, czyli folder obrazów do autoanotacji."
                )
                payload["action_text"] = (
                    "Kliknij przycisk poniżej, aby wskazać folder obrazów. Ustawienia modelu tablic i pozostałych parametrów wybierzesz dopiero po przygotowaniu źródła."
                )
                payload["workflow_start_intro"] = (
                    "To jest jeszcze etap przygotowania źródła pracy. Najpierw wskaż obrazy do autoanotacji, a dopiero potem otworzysz modal ustawień runu."
                )
        if auto_setup_pending:
            payload["workflow_start_title"] = "Ustawienia autoanotacji"
            if input_dir_ready:
                payload["route_text"] = (
                    "Źródło pracy jest już wskazane. Teraz otworzysz modal ustawień autoanotacji dla bieżącego runu Z2."
                )
                payload["action_text"] = (
                    "Kliknij Start autoanotacji, aby wybrać model tablic, confidence i opcjonalne boxy pojazdów."
                )
                payload["workflow_start_intro"] = (
                    "Obrazy są już gotowe. W następnym kroku potwierdzisz ustawienia autoanotacji w modalu startu."
                )
            else:
                payload["route_text"] = (
                    "Najpierw wskaż obrazy źródłowe dla tego runu Z2. Dopiero potem otworzysz modal ustawień autoanotacji."
                )
                payload["action_text"] = (
                    "Kliknij przycisk poniżej, aby wybrać folder obrazów do autoanotacji."
                )
                payload["workflow_start_intro"] = (
                    "To jest krok przygotowania źródła pracy. Ustawienia modelu tablic, confidence i opcjonalnych boxów pojazdów potwierdzisz dopiero po wskazaniu obrazów."
                )
        if auto_completed:
            payload["followup_title"] = AUTO_REVIEW_FOLLOWUP_TITLE
            payload["followup_text"] = AUTO_REVIEW_FOLLOWUP_TEXT
            payload["export_text"] = (
                "Przycisk „Otwórz eksport” przenosi miniflow do kroku Eksport i otwiera dwie ścieżki. "
                "Eksport anotacji XML pakuje zapisane anotacje tablic i nie wymaga statusu [OK]. "
                "Eksport datasetu YOLO Pose ze splitem "
                "korzysta wyłącznie z pozycji oznaczonych statusem [OK] i jest osobną ścieżką "
                "względem wyodrębniania tablic do Z3/PZ1."
            )
            if current_step == "auto_start":
                payload["run_title"] = "Autoanotacja zakończona"
                payload["route_text"] = (
                    "Run autoanotacji Z2 jest zapisany. Kliknij Dalej, aby przejść do kroku Korekta "
                    "i tam sprawdzić wynik, zatwierdzić obrazy [OK] albo otworzyć eksport."
                )
                payload["action_text"] = (
                    "Ten krok nie przenosi już automatycznie do korekty. Dalej świadomie otwiera następną kapsułkę miniflow."
                )
                payload["workflow_start_intro"] = (
                    "Możesz uruchomić autoanotację ponownie z innymi ustawieniami albo przejść dalej do korekty wyniku."
                )

        payload["workflow_input_title"] = "Wska\u017c katalog obraz\u00f3w"
        payload["workflow_start_title"] = "Autoanotacja"
        payload["workflow_vehicle_title"] = "Model pojazdow do wsparcia tablic (YOLO Box)"
        payload["workflow_vehicle_hint"] = (
            "Ten model jest opcjonalny. Sluzy tylko do zawezenia szukania tablic do obszaru pojazdu. "
            "Boxy pojazdow sa pomocnicze i nie trafiaja do finalnego eksportu YOLO."
        )
        if auto_setup_pending:
            if input_dir_ready:
                payload["route_text"] = "Zrodlo pracy jest juz przygotowane dla biezacego runu Z2."
                payload["action_text"] = (
                    "Kliknij Start autoanotacji, aby otworzyc modal i wybrac model tablic, confidence, fit oraz opcjonalne wsparcie pojazdami."
                )
                payload["workflow_start_intro"] = (
                    "To jest wlasciwy etap autoanotacji. Start otworzy modal ustawien, a po zatwierdzeniu ruszy proces na liscie i podgladzie Z2."
                )
            else:
                payload["route_text"] = "Najpierw przygotuj zrodlo pracy dla tego runu Z2."
                payload["action_text"] = "Kliknij przycisk ponizej, aby wskazac katalog zdjec do autoanotacji."
                payload["workflow_start_intro"] = (
                    "Ten etap zaczyna sie od wskazania obrazow. Dopiero po ich wyborze przejdziesz dalej do wlasciwej autoanotacji."
                )
        elif auto_completed and current_step == "auto_start":
            payload["auto_choice_hint"] = ""
        elif auto_vehicle_choice == "skip":
            payload["auto_choice_hint"] = "Zaznaczone pole oznacza wariant: tylko tablice."
        else:
            payload["route_text"] = "Run Z2 wykona autoanotacje tablic ze wsparciem wykrywania pojazdow."
            payload["action_text"] = "W tym wariancie najpierw wykrywany jest pojazd, a model tablic szuka tablic tylko w jego obrebie. Boxy pojazdow pozostaja pomocnicze i nie trafiaja do finalnego YOLO."
            payload["auto_choice_hint"] = "Odznaczone pole oznacza wariant: wsparcie pojazdami dla tablic."

    elif route == "manual":
        payload["run_title"] = (
            "Praca ręczna na runie Z2"
            if bool(ctx.manual_setup) and not bool(ctx.manual_review_active)
            else "Ręczna korekta runu Z2"
        )
        payload["badge_text"] = "Aktywny tor: praca ręczna w Z2"
        payload["badge_tone"] = "warning"
        payload["route_tone"] = "muted"
        payload["followup_title"] = MANUAL_REVIEW_FOLLOWUP_TITLE
        payload["followup_text"] = MANUAL_REVIEW_FOLLOWUP_TEXT
        payload["export_text"] = (
            "Krok Eksport ma dwie ścieżki. Eksport anotacji XML pakuje zapisane anotacje tablic "
            "i nie wymaga statusu [OK]. Eksport datasetu YOLO Pose ze splitem korzysta wyłącznie "
            "z pozycji oznaczonych statusem [OK] i jest osobną ścieżką względem wyodrębniania tablic."
        )
        payload["workflow_input_title"] = "Wskaż katalog obrazów"
        payload["workflow_input_hint"] = "Najpierw wybierz folder obrazów. Ten folder będzie bazą nowego ręcznego runu anotacji Z2."

        if current_step == "manual_entry":
            if bool(ctx.manual_import):
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej wskażesz dowolny run Z2 do korekty. "
                    "Jeśli leży poza workspace, program bezpiecznie skopiuje go do lokalnego importu."
                )
            elif manual_entry_mode == "continue":
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej przejdziesz do historii lokalnych runów Z2 "
                    "i wybierzesz run do wznowienia korekty."
                )
            else:
                payload["manual_hint"] = (
                    "Po kliknięciu Dalej przejdziesz do tworzenia nowego runu ręcznego Z2."
                )
            payload["manual_hint_tone"] = "muted"

        if manual_entry_mode == "continue":
            if current_step == "manual_history":
                payload["workflow_start_title"] = "Otwórz run anotacji do korekty"
                payload["route_text"] = "Na tym etapie wybierasz run Z2 z lokalnej historii i otwierasz go do dalszej pracy ręcznej."
                payload["action_text"] = (
                    "Zaznacz run Z2 z historii i kliknij Dalej, aby otworzyć go w edytorze."
                    if has_manual_history
                    else "Historia jest pusta. Wróć i wybierz nowy run ręczny albo wskaż dowolny run Z2."
                )
                payload["manual_hint"] = (
                    "Podgląd pozostaje wyłączony, dopóki nie otworzysz konkretnego runu Z2 z historii."
                    if has_manual_history
                    else "Jeśli nie masz jeszcze lokalnej historii runów, cofnij się i wybierz inny sposób wejścia."
                )
                payload["manual_hint_tone"] = "muted" if has_manual_history else "warning"
                payload["manual_template_hint"] = (
                    "Run Z2 do kontynuacji to katalog z annotations.xml i zgodnymi obrazami. "
                    f"Domyślny katalog runów Z2: {host._get_annotation_run_storage_display_path()}."
                )
                payload["manual_template_tone"] = "muted"
        elif bool(ctx.manual_import):
            payload["route_text"] = "Wybrano tor wskazania dowolnego runu Z2 do korekty."
            payload["action_text"] = "Kliknij Dalej, aby wskazać run Z2 i ewentualnie zaimportować go do lokalnego workspace."
            payload["manual_hint"] = (
                "Program przyjmie tylko run Z2 z annotations.xml i zgodnymi obrazami. "
                "Jeśli taki run leży poza workspace, zostanie bezpiecznie skopiowany do lokalnego runu import_*."
            )
            payload["manual_hint_tone"] = "muted"
            payload["manual_template_hint"] = (
                f"Domyślny katalog runów Z2: {host._get_annotation_run_storage_display_path()}."
            )
            payload["manual_template_tone"] = "muted"
        else:
            payload["workflow_input_title"] = "Wskaż katalog obrazów"
            payload["workflow_input_hint"] = (
                "Tutaj wybierasz wyłącznie katalog zdjęć wejściowych. Po kliknięciu Dalej program wczyta listę i podgląd Z2."
            )
            payload["workflow_start_title"] = "Utwórz XML anotacji"
            payload["workflow_start_intro"] = (
                "W tym kroku tworzysz nowy run ręcznej anotacji Z2 i powiązany z nim plik XML "
                "ze współrzędnymi ramek tablic dla wybranego katalogu zdjęć. Po udanym utworzeniu XML "
                "kliknij Dalej, aby przejść do kroku Korekta i tam rysować albo poprawiać ramki. "
                "Eksport samych anotacji będzie możliwy po zapisaniu pierwszej ramki; dataset YOLO Pose "
                "i wyodrębnianie tablic wymagają pozycji ze statusem [OK]."
            )
            payload["route_text"] = "Ręcznie oznaczysz tablice w wybranym katalogu zdjęć."
            payload["action_text"] = "Kliknij przycisk poniżej, aby utworzyć XML anotacji dla nowego runu ręcznej pracy."
            payload["manual_hint"] = "Ten tor tworzy nowy run ręcznej anotacji Z2 i nie nadpisuje starszych XML-i."
            payload["manual_hint_tone"] = "muted"
            payload["manual_template_hint"] = "Nowy run ręcznej anotacji Z2 nie nadpisuje starszych XML-i i jest zapisywany w workspace Z2."
            payload["manual_template_tone"] = "muted"
            payload["manual_vehicle_hint"] = (
                "Opcja przydatna głównie przy trudnych zestawach zdjęć: pojazdy pomagają zorientować się, gdzie szukać tablicy. "
                "Te boxy są pomocnicze i nie trafiają do finalnego eksportu YOLO tablic."
            )
            payload["manual_vehicle_tone"] = "muted"
            if vehicle_assist_enabled:
                payload["workflow_conf_title"] = ""
                payload["workflow_conf_hint"] = (
                    "Próg confidence dla wybranego modelu pojazdów. Im wyższy próg, tym mniej pomocniczych boxów trafi do nowego XML."
                )
                payload["workflow_vehicle_title"] = "Model pojazdów do pomocniczego boxowania"
                payload["workflow_vehicle_hint"] = (
                    "Model pojazdów posłuży tylko jako wsparcie przy ręcznym rysowaniu tablic. Boxy pojazdów nie trafiają do finalnego eksportu YOLO."
                )

        if current_step == "manual_entry" and payload.manual_template_hint:
            payload["manual_hint"] = (
                f"{payload.manual_hint}\n\n{payload.manual_template_hint}"
                if payload.manual_hint
                else str(payload.manual_template_hint or "")
            )
            payload["manual_template_hint"] = ""
            payload["manual_template_tone"] = "muted"

        if current_step == "manual_entry" and manual_entry_mode == "new":
            payload["route_text"] = "Wybrano utworzenie nowego runu recznej anotacji Z2 dla tej iteracji."
            payload["action_text"] = "Kliknij Dalej, aby przejsc do wyboru obrazow i przygotowac nowy XML do recznej pracy."
            payload["manual_hint"] = (
                "Nowy run zapisze sie jako osobny katalog w workspace Z2. "
                f"Domyslny katalog runow Z2: {host._get_annotation_run_storage_display_path()}."
            )
            payload["manual_hint_tone"] = "muted"
            payload["manual_template_hint"] = ""
            payload["manual_template_tone"] = "muted"

    return payload
