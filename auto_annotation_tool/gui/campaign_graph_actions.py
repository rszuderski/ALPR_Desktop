"""GUI-side executor for campaign graph actions.

The campaign graph and state machine stay UI-agnostic. This module is the thin
bridge that maps graph action names to existing wizard callbacks while we
gradually retire ad-hoc widget commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from tkinter import messagebox

from ..campaign_manager import CAMPAIGN
from ..config import logger
from ..campaign_iteration_paths import (
    STEP1_ITERATION_PATHS,
    iteration_path_target,
    normalize_iteration_path,
)
from .z2_shared_ui import campaign_visible_gate_id, campaign_gate_id_for_edge


@dataclass(frozen=True)
class CampaignGraphActionResult:
    ok: bool
    action: str
    message: str = ""
    pending: bool = False


def _method_result(value, action, success_message, error_message):
    if isinstance(value, CampaignGraphActionResult):
        return value
    if isinstance(value, Mapping) and "ok" in value:
        return CampaignGraphActionResult(
            bool(value["ok"]), action,
            str(value.get("message") or (success_message if value["ok"] else error_message)),
            pending=bool(value.get("pending", False)),
        )
    return CampaignGraphActionResult(value is not False, action, error_message if value is False else success_message)


_E1_RESOURCE_CONTRACT_PATH_GROUPS = {
    "plate_training": "T01",
    "char_from_images": "T01",
    "char_from_ready_plates": "T02",
}


def _safe_update_status(host: Any, message: str, tone: str = "info") -> None:
    try:
        app = getattr(host, "app", None)
        if app is not None:
            app.update_status(message, tone)
    except Exception:
        pass


def _safe_refresh_wizard(host: Any) -> None:
    try:
        host._refresh_active_project_wizard_only()
        return
    except Exception:
        pass
    try:
        host._refresh_dashboard()
    except Exception:
        pass


def _e1_resource_contract_group(path: str | None) -> str:
    return _E1_RESOURCE_CONTRACT_PATH_GROUPS.get(normalize_iteration_path(path), "")


def _e1_resource_contract_path_label(path: str | None) -> str:
    normalized = normalize_iteration_path(path)
    if normalized in {"plate_training", "char_from_images"}:
        return "T01"
    if normalized == "char_from_ready_plates":
        return "T02"
    return "E1"


def _confirm_e1_resource_contract_rollback(host: Any, previous_path: str, next_path: str) -> bool:
    previous_label = _e1_resource_contract_path_label(previous_path)
    next_label = _e1_resource_contract_path_label(next_path)
    parent = None
    try:
        frame = getattr(host, "frame", None)
        if frame is not None:
            parent = frame.winfo_toplevel()
    except Exception:
        parent = None
    try:
        return bool(
            messagebox.askyesno(
                "Porzucić roboczy wybór?",
                (
                    f"Masz roboczo zmienione zasoby w {previous_label}.\n\n"
                    f"Przejście do {next_label} przywróci aktywny stan zasobów do początku bieżącej iteracji. "
                    "Pliki robocze nie zostaną skasowane, ale nie będą użyte przez wybraną teraz bramkę.\n\n"
                    f"Czy przejść do {next_label} i porzucić szkic {previous_label}?"
                ),
                parent=parent,
            )
        )
    except Exception:
        return False


def _prepare_e1_resource_contract_switch(host: Any, normalized_path: str) -> CampaignGraphActionResult | None:
    next_group = _e1_resource_contract_group(normalized_path)
    if not next_group:
        return None

    try:
        previous_path = normalize_iteration_path(CAMPAIGN.get_explicit_iteration_path())
    except Exception:
        previous_path = ""
    previous_group = _e1_resource_contract_group(previous_path)
    if not previous_group:
        try:
            CAMPAIGN.ensure_e1_resource_contract_baseline()
        except Exception:
            pass
        return None
    if previous_path == normalized_path or previous_group == next_group:
        try:
            CAMPAIGN.ensure_e1_resource_contract_baseline()
        except Exception:
            pass
        return None

    try:
        contract_state = dict(CAMPAIGN.get_e1_resource_contract_state() or {})
    except Exception:
        contract_state = {}
    if not bool(contract_state.get("baseline_ready")):
        try:
            CAMPAIGN.ensure_e1_resource_contract_baseline()
        except Exception:
            pass
        return None
    if not bool(contract_state.get("has_draft")):
        return None

    if not _confirm_e1_resource_contract_rollback(host, previous_path, normalized_path):
        message = "Pozostawiono bieżący szkic zasobów E1 bez zmian."
        _safe_update_status(host, message, "info")
        return CampaignGraphActionResult(False, "set_iteration_path", message)

    try:
        restored = bool(
            CAMPAIGN.restore_e1_resource_contract_baseline(
                from_path=previous_path,
                to_path=normalized_path,
            )
        )
    except Exception as exc:
        logger.error(f"Nie udało się przywrócić bazowego kontraktu E1: {exc}")
        restored = False
    if not restored:
        message = "Nie udało się przywrócić bazowego stanu zasobów E1. Przełączanie bramki przerwano."
        _safe_update_status(host, message, "warning")
        return CampaignGraphActionResult(False, "set_iteration_path", message)

    _safe_update_status(
        host,
        f"Porzucono szkic {_e1_resource_contract_path_label(previous_path)} i przywrócono zasoby do początku iteracji.",
        "info",
    )
    return None


def _execute_set_iteration_path(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    normalized_path = normalize_iteration_path(
        payload.get("path_key")
        or payload.get("iteration_path")
        or payload.get("path")
    )
    target = iteration_path_target(normalized_path)
    try:
        t01_state = dict(CAMPAIGN.get_t01_entry_commit_state() or {})
        t01_committed = bool(t01_state.get("committed"))
    except Exception:
        t01_state = {}
        t01_committed = False
    if t01_committed:
        locked_path = normalize_iteration_path(t01_state.get("path") or "")
        if normalized_path != locked_path:
            message = (
                "Ta iteracja ma juz zatwierdzona sciezke T01. "
                "Zmiana wyboru bedzie dostepna dopiero w kolejnej iteracji albo po osobnym cofnieciu T01."
            )
            _safe_update_status(host, message, "warning")
            return CampaignGraphActionResult(False, "set_iteration_path", message)
    if normalized_path and normalized_path != "char_from_ready_plates":
        try:
            t02_committed = bool(CAMPAIGN.is_t02_at_review_committed_current_iteration())
        except Exception:
            t02_committed = False
        if t02_committed:
            message = (
                "Ta iteracja ma juz zapisana kontrole AT w T02. "
                "Zeby nie rozjechac kontraktu O-AT, biezacy cykl pozostaje na sciezce T02."
            )
            _safe_update_status(host, message, "warning")
            return CampaignGraphActionResult(False, "set_iteration_path", message)
    if not normalized_path or target not in {"plate", "char"}:
        return CampaignGraphActionResult(False, "set_iteration_path", "Nieznana ścieżka E1.")

    contract_switch_result = _prepare_e1_resource_contract_switch(host, normalized_path)
    if contract_switch_result is not None:
        return contract_switch_result

    try:
        previous_target = str(host._get_iteration_target() or "").strip().lower()
    except Exception:
        previous_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()

    if previous_target in {"plate", "char"} and previous_target != target:
        try:
            reason = str(host._get_iteration_target_lock_reason() or "").strip()
        except Exception:
            reason = ""
        if reason:
            _safe_update_status(host, reason, "info")
            return CampaignGraphActionResult(False, "set_iteration_path", reason)

    lightweight_refresh = bool(payload.get("refresh") is False or payload.get("lightweight") is True)

    if previous_target != target and not lightweight_refresh:
        chooser = getattr(host, "_choose_step1_iteration_target", None)
        if callable(chooser):
            chooser(target)
        else:
            CAMPAIGN.set_iteration_target(target)
    elif previous_target != target:
        CAMPAIGN.set_iteration_target(target)
        try:
            target_var = getattr(host, "_wizard_step2_target_var", None)
            if target_var is not None:
                target_var.set(target)
        except Exception:
            pass

    try:
        CAMPAIGN.set_iteration_path(normalized_path)
        if str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved":
            CAMPAIGN.set_current_step(1)
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu set_iteration_path: {exc}")
        return CampaignGraphActionResult(False, "set_iteration_path", "Nie udało się zapisać ścieżki E1.")

    if not lightweight_refresh:
        _safe_refresh_wizard(host)
    path_title = str(STEP1_ITERATION_PATHS.get(normalized_path, {}).get("title") or normalized_path)
    message = f"Wybrano ścieżkę E1: {path_title}."
    _safe_update_status(host, message, "info")
    return CampaignGraphActionResult(True, "set_iteration_path", message)


def _execute_approve_step1(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "approve_step1",
        "_approve_step1_from_wizard",
        success_message="E1 przekazano do zatwierdzenia.",
        error_message="Nie udało się zatwierdzić E1.",
    )


def _execute_approve_step1_ready_plates(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    method = getattr(host, "_approve_step1_with_existing_char_material", None)
    if not callable(method):
        return CampaignGraphActionResult(
            False,
            "approve_step1_ready_plates",
            "Brak akcji zatwierdzania T02.",
        )
    try:
        CAMPAIGN.set_iteration_path("char_from_ready_plates")
    except Exception as exc:
        logger.error(f"Nie udało się ustawić ścieżki T02 przed zatwierdzeniem: {exc}")
        return CampaignGraphActionResult(
            False,
            "approve_step1_ready_plates",
            "Nie udało się wybrać ścieżki T02. Odśwież projekt i spróbuj ponownie.",
        )
    try:
        result = method()
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu approve_step1_ready_plates: {exc}")
        return CampaignGraphActionResult(
            False,
            "approve_step1_ready_plates",
            "Nie udało się zatwierdzić T02.",
        )
    after_step = int(CAMPAIGN.get_current_step() or 1)
    if bool(result) and after_step >= 3:
        try:
            CAMPAIGN.set_iteration_path("char_from_ready_plates")
            if str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved":
                CAMPAIGN.approve_step1()
            if str(CAMPAIGN.get_step2_status() or "").strip().lower() != "approved":
                CAMPAIGN.approve_step2()
            CAMPAIGN.set_current_step(3)
            after_step = 3
        except Exception as exc:
            logger.error(f"Nie udało się znormalizować stanu po zatwierdzeniu T02: {exc}")
            return CampaignGraphActionResult(
                False,
                "approve_step1_ready_plates",
                "T02 została potwierdzona, ale nie udało się przejść do E3. Odśwież projekt i spróbuj ponownie.",
            )
    if not bool(result) or after_step < 3:
        return CampaignGraphActionResult(
            False,
            "approve_step1_ready_plates",
            "T02 nie została zatwierdzona. Pozostajesz w E1; sprawdź źródło tablic i potwierdzenie decyzji.",
        )
    return CampaignGraphActionResult(
        True,
        "approve_step1_ready_plates",
        "T02 zatwierdzona. Potwierdzono źródło tablic i przejście dalej do znaków.",
    )


def _execute_host_method(
    host: Any,
    action: str,
    method_name: str,
    *,
    success_message: str,
    error_message: str,
) -> CampaignGraphActionResult:
    method = getattr(host, method_name, None)
    if not callable(method):
        return CampaignGraphActionResult(False, action, f"Brak akcji: {method_name}.")
    try:
        value = method()
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu {action}: {exc}")
        return CampaignGraphActionResult(False, action, error_message)
    return _method_result(value, action, success_message, error_message)


def _graph_context_from_payload(payload: Mapping[str, Any]) -> dict:
    context = dict(payload.get("context") or payload.get("preferred_source_context") or {})
    for key in (
        "source",
        "graph_edge_key",
        "graph_gate_id",
        "graph_gate_label",
        "graph_transition_title",
        "graph_transition_source",
        "graph_transition_target",
        "graph_path_key",
    ):
        if key not in context and payload.get(key) not in (None, ""):
            context[key] = payload.get(key)
    return context


def _execute_host_method_with_context(
    host: Any,
    action: str,
    method_name: str,
    payload: Mapping[str, Any],
    *,
    success_message: str,
    error_message: str,
) -> CampaignGraphActionResult:
    method = getattr(host, method_name, None)
    if not callable(method):
        return CampaignGraphActionResult(False, action, f"Brak akcji: {method_name}.")
    context = _graph_context_from_payload(payload)
    try:
        kwargs = {"on_complete": payload.get("_on_complete")} if action.startswith("open_z2") else {}
        value = method(context, **kwargs)
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu {action}: {exc}")
        return CampaignGraphActionResult(False, action, error_message)
    return _method_result(value, action, success_message, error_message)


def _execute_open_z2_campaign_context(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    method = getattr(host, "_step_goto_auto_annotation", None)
    if not callable(method):
        return CampaignGraphActionResult(False, "open_z2_campaign_context", "Brak akcji: _step_goto_auto_annotation.")
    context = _graph_context_from_payload(payload)
    try:
        value = method(preferred_source_context=context, on_complete=payload.get("_on_complete"))
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu open_z2_campaign_context: {exc}")
        return CampaignGraphActionResult(False, "open_z2_campaign_context", "Nie udało się otworzyć Z2.")
    return _method_result(value, "open_z2_campaign_context", "Otwieram Z2…", "Nie udało się otworzyć Z2.")


def _execute_open_z2_step2_review(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method_with_context(
        host,
        "open_z2_step2_review",
        "_step_open_z2_from_step2_review",
        _payload,
        success_message="Z2 przekazano do przeglądu E2.",
        error_message="Nie udało się otworzyć Z2 w przeglądzie E2.",
    )


def _execute_open_z2_step3_repair(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method_with_context(
        host,
        "open_z2_step3_repair",
        "_step_open_z2_repair_from_later_stage",
        _payload,
        success_message="Z2 przekazano do trybu uzupełniania tablic.",
        error_message="Nie udało się otworzyć Z2 w trybie uzupełniania tablic.",
    )


def _execute_return_to_z2_review(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    method = getattr(host, "_step_return_to_annotation_review", None)
    if not callable(method):
        return CampaignGraphActionResult(False, "return_to_z2_review", "Brak akcji powrotu do Z2.")
    mark_rework = bool(payload.get("mark_step3_rework", True))
    context = _graph_context_from_payload(payload)
    try:
        value = method(mark_step3_rework=mark_rework, preferred_source_context=context, on_complete=payload.get("_on_complete"))
    except Exception as exc:
        logger.error(f"Nie udało się wykonać akcji grafu return_to_z2_review: {exc}")
        return CampaignGraphActionResult(False, "return_to_z2_review", "Nie udało się wrócić do Z2.")
    return _method_result(value, "return_to_z2_review", "Otwieram Z2…", "Nie udało się wrócić do Z2.")


def _execute_approve_step2(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    context = _graph_context_from_payload(payload)
    gate_id = campaign_gate_id_for_edge(context.get("graph_edge_key"), context.get("graph_gate_id"))
    visible_gate_id = campaign_visible_gate_id(gate_id) or gate_id
    target = str(context.get("graph_transition_target") or "").strip().upper()
    success_message = "Bramka została przekazana do zatwierdzenia."
    if target == "E4T" or gate_id == "T04":
        success_message = f"Bramka {visible_gate_id} zatwierdzona. Odblokowano E4T, czyli trening modelu tablic."
    elif target == "E3" or gate_id == "T03":
        success_message = f"Bramka {visible_gate_id} zatwierdzona. Odblokowano E3, czyli pracę nad znakami."

    return _execute_host_method_with_context(
        host,
        "approve_step2",
        "_approve_step2_from_wizard",
        payload,
        success_message=success_message,
        error_message="Nie udało się zatwierdzić E2.",
    )


def _execute_validate_and_adopt_annotations(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "validate_and_adopt_annotations",
        "_import_project_start_plate_run",
        success_message="Import/adopcja anotacji została przekazana do E1.",
        error_message="Nie udało się uruchomić importu/adopcji anotacji.",
    )


def _execute_continue_z3(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method_with_context(
        host,
        "continue_z3",
        "_step_continue_characters_from_ready_source",
        payload,
        success_message="Z3 przekazano do kontynuacji na gotowym źródle.",
        error_message="Nie udało się kontynuować pracy w Z3.",
    )


def _execute_open_z3(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method_with_context(
        host,
        "open_z3",
        "_step_goto_characters",
        payload,
        success_message="Z3 przekazano do otwarcia.",
        error_message="Nie udało się otworzyć Z3.",
    )


def _execute_open_z3_detect(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method_with_context(
        host,
        "open_z3_detect",
        "_step_goto_characters_detect",
        payload,
        success_message="Z3/PZ2 przekazano do pracy nad znakami.",
        error_message="Nie udało się otworzyć Z3/PZ2.",
    )


def _execute_approve_step3(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "approve_step3",
        "_approve_step3_from_wizard",
        success_message="Bramka T05 zatwierdzona. Odblokowano E4Z, czyli trening modelu znaków.",
        error_message="Nie udało się zatwierdzić E3.",
    )


def _execute_open_z4(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "open_z4",
        "_step_goto_training",
        success_message="Z4/PZ2 przekazano do treningu albo do przygotowania brakującego wariantu.",
        error_message="Nie udało się otworzyć treningu w Z4.",
    )


def _execute_open_z4_training(host: Any, payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    method = getattr(host, "_step_goto_training", None)
    if not callable(method):
        return CampaignGraphActionResult(False, "open_z4", "Brak akcji: _step_goto_training.")
    preferred_subtab = str(payload.get("preferred_subtab") or payload.get("subtab") or "train").strip().lower()
    if preferred_subtab not in {"dataset", "train"}:
        preferred_subtab = "train"
    try:
        method(preferred_subtab=preferred_subtab)
    except Exception as exc:
        logger.error(f"Nie udaĹ‚o siÄ™ wykonaÄ‡ akcji grafu open_z4: {exc}")
        return CampaignGraphActionResult(False, "open_z4", "Nie udaĹ‚o siÄ™ otworzyÄ‡ treningu w Z4.")
    return CampaignGraphActionResult(
        True,
        "open_z4",
        "Z4/PZ2 przekazano do treningu i wyboru wyniku bramki.",
    )


def _execute_open_z4_dataset(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "open_z4_dataset",
        "_step_goto_training_dataset",
        success_message="Z4/PZ1 przekazano do utworzenia wariantu datasetu.",
        error_message="Nie udało się otworzyć Z4/PZ1.",
    )


def _current_step4_without_training_decision_ready() -> bool:
    try:
        decision = dict(CAMPAIGN.get_step4_without_training_decision() or {})
    except Exception:
        decision = {}
    if not bool(decision.get("ready")):
        return False
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    try:
        decision_iteration = int(decision.get("iteration", 0) or 0)
    except Exception:
        decision_iteration = 0
    current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    decision_target = str(decision.get("target", "") or "").strip().lower()
    return bool(
        decision_iteration == current_iteration
        and (not decision_target or not current_target or decision_target == current_target)
    )


def _current_training_stage_label() -> str:
    target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    if target == "plate":
        return "E4T"
    if target == "char":
        return "E4Z"
    return "E4T/E4Z"


def _current_iteration_step4_training_record() -> dict:
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    if current_iteration <= 0:
        return {}
    try:
        synced = dict(CAMPAIGN.sync_step4_training_record_from_history(iteration_num=current_iteration) or {})
        if synced:
            return synced
    except Exception:
        pass
    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=current_iteration) or {})
        record = dict(iteration_state.get("step4_training") or {})
    except Exception:
        record = {}
    if record:
        return record
    try:
        bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(iteration_num=current_iteration) or {})
        record = dict(bundle.get("step4_training") or {})
    except Exception:
        record = {}
    return record


def _current_step4_training_finish_ready(host: Any) -> bool:
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    finish_state = {}
    try:
        app = getattr(host, "app", None)
        training_tab = app.tabs.get("training") if app is not None and getattr(app, "tabs", None) else None
        getter = getattr(training_tab, "get_campaign_step4_finish_state", None)
        if callable(getter):
            finish_state = dict(getter(iteration_target=current_target) or {})
    except Exception:
        finish_state = {}
    if not finish_state:
        try:
            finish_state = dict(CAMPAIGN.get_step4_finish_state() or {})
        except Exception:
            finish_state = {}
    if not bool(finish_state.get("ready")):
        return False
    run_id = str(finish_state.get("run_id", "") or "").strip()
    if not run_id:
        return False
    try:
        finish_iteration = int(finish_state.get("iteration", 0) or 0)
    except Exception:
        finish_iteration = 0
    finish_target = str(finish_state.get("target", "") or "").strip().lower()
    if finish_iteration != current_iteration:
        return False
    if finish_target and current_target and finish_target != current_target:
        return False
    selected_model_path = str(finish_state.get("model_path", "") or "").strip()
    if bool(finish_state.get("selection_confirmed", True)) and selected_model_path:
        try:
            return bool(Path(selected_model_path).exists() and Path(selected_model_path).is_file())
        except Exception:
            return bool(selected_model_path)
    training_record = _current_iteration_step4_training_record()
    if not training_record:
        return False
    record_run_id = str(training_record.get("run_id", "") or "").strip()
    if record_run_id != run_id:
        return False
    record_target = str(training_record.get("target", "") or "").strip().lower()
    if record_target and current_target and record_target != current_target:
        return False
    record_status = str(training_record.get("status", "") or "").strip().lower()
    if record_status != "completed":
        return False
    best_weights = str(training_record.get("best_weights", "") or "").strip()
    if not best_weights:
        return False
    try:
        return bool(Path(best_weights).exists())
    except Exception:
        return False


def _execute_prepare_step4_without_training(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    if not CAMPAIGN.get_active_project_name():
        return CampaignGraphActionResult(False, "prepare_step4_without_training", "Brak aktywnego projektu.")
    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
    except Exception:
        current_step = 0
    stage_label = _current_training_stage_label()
    if current_step != 4:
        return CampaignGraphActionResult(
            False,
            "prepare_step4_without_training",
            f"Zakończenie bez treningu dotyczy wyłącznie bramki T06 w {stage_label}.",
        )
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    try:
        CAMPAIGN.set_step4_without_training_decision(
            True,
            target=target,
            iteration_num=current_iteration,
        )
    except Exception as exc:
        logger.error(f"Nie udało się zapisać decyzji T06 bez treningu: {exc}")
        return CampaignGraphActionResult(
            False,
            "prepare_step4_without_training",
            "Nie udało się zapisać decyzji zakończenia bez treningu.",
        )
    try:
        app = getattr(host, "app", None)
        if app is not None and hasattr(app, "themed_info"):
            app.themed_info(
                "Decyzja T06",
                (
                    f"Wybrano zakończenie {stage_label} bez treningu.\n\n"
                    "To jeszcze nie przenosi projektu do E1. Bramka T06 została przygotowana do zamknięcia; "
                    "formalny skok do kolejnej iteracji wykonasz dopiero polem „ZATWIERDŹ” na bramce T06."
                ),
                parent=getattr(host, "frame", None),
                tone="warning",
            )
    except Exception:
        pass
    _safe_refresh_wizard(host)
    return CampaignGraphActionResult(
        True,
        "prepare_step4_without_training",
        f"Wybrano zakończenie {stage_label} bez treningu. Zatwierdź bramkę T06, aby przejść dalej.",
    )


def _execute_approve_step4(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    if _current_step4_without_training_decision_ready():
        return _execute_host_method(
            host,
            "approve_step4_without_training",
            "_finish_step4_without_training",
            success_message="T06 przekazano do zamknięcia bez treningu.",
            error_message="Nie udało się zamknąć T06 bez treningu.",
        )
    if not _current_step4_training_finish_ready(host):
        _safe_update_status(
            host,
            "T06 wymaga decyzji w bieżącej iteracji: uruchom trening albo wybierz świadome pominięcie treningu.",
            "warning",
        )
        _safe_refresh_wizard(host)
        return CampaignGraphActionResult(
            False,
            "approve_step4",
            "T06 nie ma bieżącego wyniku treningu ani decyzji pominięcia treningu.",
        )
    return _execute_host_method(
        host,
        "approve_step4",
        "_finish_step4_iteration",
        success_message=f"{_current_training_stage_label()} przekazano do zatwierdzenia.",
        error_message=f"Nie udało się zatwierdzić {_current_training_stage_label()}.",
    )


def _execute_approve_step4_without_training(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "approve_step4_without_training",
        "_finish_step4_without_training",
        success_message=f"{_current_training_stage_label()} przekazano do zamknięcia bez treningu.",
        error_message=f"Nie udało się zamknąć {_current_training_stage_label()} bez treningu.",
    )


def _execute_start_next_iteration(host: Any, _payload: Mapping[str, Any]) -> CampaignGraphActionResult:
    return _execute_host_method(
        host,
        "start_next_iteration",
        "_advance_iteration",
        success_message="Przejście do E1 kolejnej iteracji zostało przekazane.",
        error_message="Nie udało się przejść do E1 kolejnej iteracji.",
    )


def _record_graph_action_history(
    action: str,
    payload: Mapping[str, Any],
    result: CampaignGraphActionResult,
    *,
    project_name: str = "",
    iteration_num: int | None = None,
) -> None:
    project_name = project_name or CAMPAIGN.get_active_project_name()
    if not project_name:
        return
    if iteration_num is None:
        iteration_num = CAMPAIGN.get_current_iteration_num()

    context = dict(payload.get("context") or payload.get("preferred_source_context") or {})
    transition_id = str(
        payload.get("graph_edge_key")
        or payload.get("transition_id")
        or context.get("graph_edge_key")
        or ""
    ).strip()
    gate_id = str(
        payload.get("graph_gate_id")
        or payload.get("gate_id")
        or context.get("graph_gate_id")
        or transition_id
        or ""
    ).strip()
    transition_title = str(
        payload.get("graph_transition_title")
        or context.get("graph_transition_title")
        or payload.get("title")
        or ""
    ).strip()
    title = transition_title or str(result.message or action or "Akcja grafu").strip()
    artifacts: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    if str(action or "").strip().startswith("approve_step") and bool(result.ok):
        from ..campaign_history_resources import HistoryResourceReader

        try:
            reader = HistoryResourceReader(CAMPAIGN.get_project_root_dir(project_name), project_name)
            snapshot = reader.snapshot(int(iteration_num))
            artifacts["resource_snapshot"] = snapshot
            for code, label in (
                ("O", "Wybrane obrazy"), ("AT", "Zatwierdzone anotacje tablic"),
                ("AZ", "Zatwierdzone anotacje znaków"), ("DS", "Dataset znaków"),
                ("MT", "Model tablic"), ("MZ", "Model znaków"),
            ):
                if code in snapshot:
                    resources[code] = label
            if action == "approve_step2":
                at = snapshot.get("AT") or {}
                for name, field in (("approved_images", "images"), ("approved_plates", "plates")):
                    if at.get(field) is not None:
                        artifacts[name] = at[field]
        except Exception as exc:
            logger.warning("Nie udało się zapisać liczników historii %s IT%s: %s", project_name, iteration_num, exc)

    try:
        CAMPAIGN.append_project_history_event(
            "graph_action",
            title,
            transition_id=transition_id,
            gate_id=gate_id,
            action=str(action or "").strip(),
            status="ok" if bool(result.ok) else "error",
            resources=resources,
            artifacts=artifacts,
            project_name=project_name,
            iteration_num=iteration_num,
            details={
                "message": str(result.message or "").strip(),
                "source": str(payload.get("source") or context.get("source") or "campaign_graph").strip(),
                "target": str(payload.get("graph_transition_target") or context.get("graph_transition_target") or "").strip(),
                "path": str(payload.get("graph_path_key") or context.get("graph_path_key") or normalize_iteration_path(payload.get("path")) or "").strip(),
            },
        )
    except Exception:
        pass


def execute_campaign_graph_action(
    host: Any,
    action: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> CampaignGraphActionResult:
    normalized = str(action or "").strip()
    action_payload = dict(payload or {})
    before_iteration = CAMPAIGN.get_current_iteration_num()
    before_project = CAMPAIGN.get_active_project_name()
    # Approval can advance the state. Its resource record belongs to the source iteration.
    history_context = {}
    if normalized.startswith("approve_step") or normalized in {"start_next_iteration", "open_z2_campaign_context", "open_z2_step2_review", "open_z2_step3_repair", "return_to_z2_review"}:
        history_context = {
            "project_name": before_project,
            "iteration_num": before_iteration,
        }
    if normalized in {"open_z2_campaign_context", "open_z2_step2_review", "open_z2_step3_repair", "return_to_z2_review"}:
        def _on_complete(value):
            completed = _method_result(value, normalized, "Otworzono Z2.", "Nie udało się otworzyć Z2.")
            _record_graph_action_history(normalized, action_payload, completed, **history_context)
        action_payload["_on_complete"] = _on_complete
    handlers = {
        "set_iteration_path": _execute_set_iteration_path,
        "approve_step1": _execute_approve_step1,
        "approve_step1_ready_plates": _execute_approve_step1_ready_plates,
        "validate_and_adopt_annotations": _execute_validate_and_adopt_annotations,
        "open_z2_campaign_context": _execute_open_z2_campaign_context,
        "open_z2_step2_review": _execute_open_z2_step2_review,
        "open_z2_step3_repair": _execute_open_z2_step3_repair,
        "return_to_z2_review": _execute_return_to_z2_review,
        "approve_step2": _execute_approve_step2,
        "continue_z3": _execute_continue_z3,
        "open_z3": _execute_open_z3,
        "open_z3_detect": _execute_open_z3_detect,
        "approve_step3": _execute_approve_step3,
        "open_z4": _execute_open_z4_training,
        "open_z4_dataset": _execute_open_z4_dataset,
        "prepare_step4_without_training": _execute_prepare_step4_without_training,
        "approve_step4": _execute_approve_step4,
        "approve_step4_without_training": _execute_approve_step4_without_training,
        "start_next_iteration": _execute_start_next_iteration,
    }
    handler = handlers.get(normalized)
    if handler is None:
        message = f"Akcja grafu nie jest jeszcze podłączona: {normalized or '-'}."
        logger.debug(message)
        result = CampaignGraphActionResult(False, normalized, message)
        _record_graph_action_history(normalized, action_payload, result, **history_context)
        return result
    if normalized.startswith("approve_step"):
        from ..campaign_transition_specs import get_transition_specs_for_edge
        from ..campaign_transition_evaluator import is_transition_path_active

        context = _graph_context_from_payload(action_payload)
        edge = str(context.get("graph_edge_key") or action_payload.get("transition_id") or "")
        specs = get_transition_specs_for_edge(edge) if edge else ()
        source_step = int(normalized[len("approve_step")])
        path = normalize_iteration_path(CAMPAIGN.get_iteration_path())
        if (not before_project or int(CAMPAIGN.get_current_step() or 1) != source_step
                or (specs and not is_transition_path_active(specs[0], path))
                or (normalized == "approve_step1" and path not in {"plate_training", "char_from_images"})):
            result = CampaignGraphActionResult(False, normalized,
                "Ta bramka nie jest aktywnym przejściem bieżącej iteracji. Odśwież graf i wybierz bramkę aktualnego etapu.")
            _record_graph_action_history(normalized, action_payload, result, **history_context)
            return result
    result = handler(host, action_payload)
    if result.ok and (normalized.startswith("approve_step") or normalized == "start_next_iteration"):
        expected_steps = {"approve_step1": 2, "approve_step1_ready_plates": 3,
                          "approve_step2": 4 if CAMPAIGN.get_iteration_target() == "plate" else 3,
                          "approve_step3": 4, "approve_step4": 5, "approve_step4_without_training": 5}
        expected = expected_steps.get(normalized, 1)
        after_step = int(CAMPAIGN.get_current_step() or 1)
        next_iteration = int(CAMPAIGN.get_current_iteration_num() or 1) > int(before_iteration or 1)
        same_project = CAMPAIGN.get_active_project_name() == before_project
        if normalized == "start_next_iteration":
            completed = same_project and next_iteration and after_step == 1
        else:
            status_getter = getattr(CAMPAIGN, f"get_step{normalized[len('approve_step')]}_status", None)
            status = str(status_getter() or "") if callable(status_getter) else ""
            completed = same_project and ((next_iteration and after_step == 1 and normalized.startswith("approve_step4")) or
                        (not next_iteration and after_step == expected and status == "approved"))
        if not completed:
            result = CampaignGraphActionResult(False, normalized,
                "Bramka nie została zatwierdzona. Stan etapu pozostał bez zatwierdzenia; sprawdź wymagane zasoby i potwierdzenie decyzji.")
    if not result.pending:
        _record_graph_action_history(normalized, action_payload, result, **history_context)
    return result


SUPPORTED_CAMPAIGN_GRAPH_ACTIONS = frozenset({
    "set_iteration_path",
    "approve_step1",
    "approve_step1_ready_plates",
    "validate_and_adopt_annotations",
    "open_z2_campaign_context",
    "open_z2_step2_review",
    "open_z2_step3_repair",
    "return_to_z2_review",
    "approve_step2",
    "continue_z3",
    "open_z3",
    "open_z3_detect",
    "approve_step3",
    "open_z4",
    "open_z4_dataset",
    "prepare_step4_without_training",
    "approve_step4",
    "approve_step4_without_training",
    "start_next_iteration",
})
