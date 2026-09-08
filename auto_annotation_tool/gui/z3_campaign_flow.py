from __future__ import annotations

import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import logger
from ..project_cache import PROJECT_CACHE
from .z3_metadata_cache import file_signature, path_key
from .campaign_graph_presentation import close_graph_dialogs
from .z3_view_models import (
    Step3CampaignNavigationViewModel,
    Step3EntryFlowViewModel,
    Step3FinishActionViewModel,
    Step3Pz3PathSelectionViewModel,
    Step3Pz3StatusPanelViewModel,
    Step3Pz3StatusRowViewModel,
)

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


CHAR_WORK_GATE_DISPLAY_ID = "T05"
CHAR_WORK_GATE_SESSION_IDS = {CHAR_WORK_GATE_DISPLAY_ID, "T06"}


def _is_t06_z3_work_context(host: "CharacterAnnotationTab", context: dict | None = None) -> bool:
    if not getattr(host, "_step3_linear_mode", False) or not CAMPAIGN.get_active_project_name():
        return False
    ctx = {}
    try:
        ctx.update(dict(getattr(host, "_campaign_graph_entry_context", {}) or {}))
    except Exception:
        pass
    if isinstance(context, dict):
        try:
            ctx.update(dict(context or {}))
        except Exception:
            pass
    gate_id = str(ctx.get("graph_gate_id") or ctx.get("gate_id") or "").strip().upper()
    edge_key = str(ctx.get("graph_edge_key") or ctx.get("edge_key") or "").strip().lower()
    if gate_id in CHAR_WORK_GATE_SESSION_IDS or edge_key == "e3_to_e4":
        return True
    try:
        session = dict((CAMPAIGN.get_iteration_state() or {}).get("t06_work_session") or {})
        return (
            str(session.get("working_gate_id") or "").strip().upper() in CHAR_WORK_GATE_SESSION_IDS
            and str(session.get("work_area") or "").strip().lower() == "z3"
            and bool(session.get("active"))
        )
    except Exception:
        return False


def _mark_t06_z3_work_session(
    host: "CharacterAnnotationTab",
    *,
    state: str = "active",
    substep: int | None = None,
    reason: str = "",
    context: dict | None = None,
    force: bool = False,
) -> None:
    if not force and not _is_t06_z3_work_context(host, context=context):
        return
    normalized_state = str(state or "active").strip().lower()
    now = datetime.now().isoformat(timespec="seconds")
    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
        session = dict(iteration_state.get("t06_work_session") or {})
    except Exception:
        session = {}
    if session and str(session.get("work_area") or "").strip().lower() not in {"", "z3"} and not force:
        # Nie nadpisujemy przerwanej pracy Z2 zaległymi [OK].
        return
    try:
        current_substep = int(substep if substep is not None else CAMPAIGN.get_step3_substep() or 1)
    except Exception:
        current_substep = 1
    active = normalized_state not in {
        "resolved",
        "closed",
        "complete",
        "completed",
        "abandoned",
        "paused",
        "ready_for_pz2",
        "waiting_for_pz2",
        "ready_for_pz3",
        "waiting_for_pz3",
    }
    session.update(
        {
            "active": bool(active),
            "state": normalized_state,
            "working_gate_id": "T05",
            "work_area": "z3",
            "substep": max(1, current_substep),
            "updated_at": now,
        }
    )
    if active:
        session.setdefault("started_at", now)
        session["last_active_at"] = now
    else:
        if normalized_state in {"resolved", "closed", "complete", "completed", "paused",
                                "ready_for_pz2", "waiting_for_pz2", "ready_for_pz3", "waiting_for_pz3"}:
            previous_interrupted_at = str(session.get("interrupted_at", "") or "").strip()
            # Registry upserts merge dictionaries; omission cannot clear an old marker.
            session["interrupted_at"] = ""
            if previous_interrupted_at:
                session.setdefault("resolved_interrupted_at", previous_interrupted_at)
        session["closed_at"] = now
    if reason:
        session["reason"] = str(reason or "").strip()
    try:
        preview_dir = str(host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "").strip()
    except Exception:
        preview_dir = ""
    if preview_dir:
        session["preview_dir"] = preview_dir
    try:
        CAMPAIGN.upsert_iteration_state(updates={"t06_work_session": session})
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać sesji pracy T06/Z3: {exc}")


def _invalidate_step3_campaign_ui_caches() -> None:
    """Drop campaign UI caches after a T05/PZ2/PZ3 contract transition."""
    try:
        CAMPAIGN.invalidate_step3_char_source_state_cache()
    except Exception:
        pass
    try:
        CAMPAIGN.clear_project_iteration_ui_snapshots()
    except Exception:
        pass


def mark_step3_work_interrupted_on_app_close(host: "CharacterAnnotationTab") -> bool:
    """Mark campaign Z3/T05 work as interrupted when the app is closed from Z3."""
    if not getattr(host, "_step3_linear_mode", False) or not CAMPAIGN.get_active_project_name():
        return False

    selected_tab_key = ""
    try:
        selected_tab_key = str(host.app._get_selected_tab_key() or "").strip().lower()
    except Exception:
        selected_tab_key = ""

    selected_substep = None
    try:
        selected_subtab = str(host.main_nb.select())
        if selected_subtab == str(host.tab_dataset):
            selected_substep = 3
        elif selected_subtab == str(host.tab_detect):
            selected_substep = 2
        elif selected_subtab == str(host.tab_extract):
            selected_substep = 1
    except Exception:
        selected_substep = None
    selected_work_substep = bool(selected_tab_key == "characters" and selected_substep in {2, 3})

    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
        session = dict(iteration_state.get("t06_work_session") or {})
    except Exception:
        session = {}

    session_state = str(session.get("state") or "").strip().lower()
    session_gate = str(session.get("working_gate_id") or "").strip().upper()
    session_area = str(session.get("work_area") or "").strip().lower()
    if selected_substep in {2, 3} and not selected_work_substep:
        try:
            selected_work_substep = bool(
                _is_t06_z3_work_context(host)
                or (session_gate in CHAR_WORK_GATE_SESSION_IDS and session_area in {"", "z3"})
            )
        except Exception:
            selected_work_substep = bool(session_gate in CHAR_WORK_GATE_SESSION_IDS and session_area in {"", "z3"})
    if session_state in {"resolved", "closed", "complete", "completed"} and not selected_work_substep:
        return False
    if session_state == "paused" and selected_tab_key != "characters":
        return False

    if session and session_area not in {"", "z3"} and not selected_work_substep:
        return False
    session_active = bool(session.get("active")) or session_state in {"active", "started", "dirty", "interrupted"}
    should_mark = bool(
        selected_work_substep
        or (
            session_gate in CHAR_WORK_GATE_SESSION_IDS
            and session_area == "z3"
            and session_active
        )
    )
    if not should_mark:
        return False

    if selected_substep in {2, 3}:
        substep = selected_substep
    else:
        try:
            substep = int(session.get("substep") or CAMPAIGN.get_step3_substep() or 1)
        except Exception:
            substep = 1

    _mark_t06_z3_work_session(
        host,
        state="interrupted",
        substep=max(1, int(substep or 1)),
        reason="app_closed_from_z3",
        force=True,
    )
    _invalidate_step3_campaign_ui_caches()
    return True


def _mark_t06_contract(
    host: "CharacterAnnotationTab",
    contract_key: str,
    payload: dict | None = None,
) -> None:
    if not getattr(host, "_step3_linear_mode", False) or not CAMPAIGN.get_active_project_name():
        return
    key = str(contract_key or "").strip()
    if not key:
        return
    contract = dict(payload or {})
    contract.setdefault("fulfilled", True)
    contract.setdefault("project", str(CAMPAIGN.get_active_project_name() or "").strip())
    contract.setdefault("iteration", int(CAMPAIGN.get_current_iteration_num() or 1))
    contract.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
    try:
        CAMPAIGN.upsert_iteration_state(updates={"t06_contracts": {key: contract}})
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać kontraktu T06/{key}: {exc}")


def _mark_t06_pz2_contract(
    host: "CharacterAnnotationTab",
    *,
    reason: str = "enter_pz3",
    force: bool = False,
) -> None:
    if not force and not _is_t06_z3_work_context(host):
        return
    try:
        readiness = dict(host._get_campaign_step3_annotation_readiness() or {})
    except Exception:
        readiness = {}
    if not bool(readiness.get("ok")):
        return
    _mark_t06_contract(
        host,
        "pz2_char_boxes",
        {
            "fulfilled": True,
            "product": "char_boxes_on_plates",
            "source": "PZ2",
            "reason": str(reason or "enter_pz3"),
            "exportable_plate_count": int(readiness.get("exportable_plate_count", 0) or 0),
            "exportable_char_count": int(readiness.get("exportable_char_count", 0) or 0),
            "perfect_count": int(readiness.get("perfect_count", 0) or 0),
            "min_exportable_plate_count": int(readiness.get("min_exportable_plate_count", 0) or 0),
            "missing_exportable_plate_count": int(readiness.get("missing_exportable_plate_count", 0) or 0),
            "fulfilled_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _mark_t06_pz3_contract(
    host: "CharacterAnnotationTab",
    summary: dict | None,
    *,
    reason: str = "dataset_exported",
    force: bool = False,
) -> None:
    if not force and not _is_t06_z3_work_context(host):
        return
    data = dict(summary or {})
    dataset_path = str(data.get("gold_dataset_path") or "").strip()
    valid = bool(data.get("gold_dataset_created")) and bool(data.get("gold_dataset_valid", True)) and bool(dataset_path)
    if not valid:
        return
    try:
        source_iteration = int(
            data.get("source_iteration")
            or data.get("created_iteration")
            or data.get("iteration")
            or CAMPAIGN.get_current_iteration_num()
            or 1
        )
    except Exception:
        source_iteration = 1
    _mark_t06_contract(
        host,
        "pz3_char_dataset",
        {
            "fulfilled": True,
            "product": "char_yolo_dataset",
            "source": "PZ3",
            "reason": str(reason or "dataset_exported"),
            "dataset_path": dataset_path,
            "exportable_plate_count": int(data.get("exportable_plate_count", 0) or 0),
            "exportable_char_count": int(data.get("exportable_char_count", 0) or 0),
            "perfect_count": int(data.get("perfect_count", data.get("exportable_plate_count", 0)) or 0),
            "gold_dataset_valid": bool(data.get("gold_dataset_valid", True)),
            "summary_path": str(data.get("_summary_path") or data.get("summary_path") or ""),
            "summary_dir": str(data.get("_summary_dir") or data.get("summary_dir") or ""),
            "iteration": int(source_iteration or 0),
            "source_iteration": int(source_iteration or 0),
            "created_iteration": int(source_iteration or 0),
            "fulfilled_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _read_ready_step3_export_summary(host: "CharacterAnnotationTab") -> dict:
    try:
        summary = dict(host._read_step3_export_summary() or {})
    except Exception:
        summary = {}
    if not summary:
        return {}
    dataset_path = str(summary.get("gold_dataset_path") or "").strip()
    if not (
        bool(summary.get("gold_dataset_created"))
        and bool(summary.get("gold_dataset_valid", True))
        and dataset_path
    ):
        return {}
    try:
        if not Path(dataset_path).exists():
            return {}
    except Exception:
        return {}
    return summary


def mark_step3_dataset_exported_for_campaign(
    host: "CharacterAnnotationTab",
    summary: dict,
    *,
    reason: str = "dataset_exported",
) -> bool:
    """Persist the T06/PZ3 contract as soon as PZ3 creates a valid AZ dataset."""
    gold_ok = bool(summary.get("gold_dataset_created")) and bool(summary.get("gold_dataset_valid", True))
    dataset_path = str(summary.get("gold_dataset_path") or "").strip()
    if not (gold_ok and dataset_path):
        return False
    try:
        if not Path(dataset_path).exists():
            return False
    except Exception:
        return False
    _mark_t06_pz2_contract(host, reason=str(reason or "dataset_exported"), force=True)
    _mark_t06_pz3_contract(host, summary, reason=str(reason or "dataset_exported"), force=True)
    _mark_t06_z3_work_session(
        host,
        state="completed",
        substep=3,
        reason=str(reason or "dataset_exported"),
        force=True,
    )
    try:
        CAMPAIGN.set_current_step(3)
        CAMPAIGN.set_step3_ready()
    except Exception:
        pass
    try:
        CAMPAIGN.invalidate_step3_char_source_state_cache()
    except Exception:
        pass
    return True


def enter_campaign_step3_mode(host: "CharacterAnnotationTab"):
    """
    Start kroku 3 od początku.
    """
    CAMPAIGN.reset_step3_progress()
    host.restore_campaign_step3_mode()
    host._set_button_emphasis("btn_to_detect_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)
    try:
        host._sync_yolo_model_binding()
    except Exception:
        pass


def _set_campaign_step3_hold_pz2_after_reextract(host: "CharacterAnnotationTab", active: bool) -> None:
    try:
        host._campaign_step3_hold_pz2_after_reextract = bool(active)
    except Exception:
        pass


def _campaign_step3_hold_pz2_after_reextract(host: "CharacterAnnotationTab") -> bool:
    try:
        return bool(getattr(host, "_campaign_step3_hold_pz2_after_reextract", False))
    except Exception:
        return False


def _apply_campaign_step3_pz2_hold(host: "CharacterAnnotationTab") -> None:
    if not _campaign_step3_hold_pz2_after_reextract(host):
        return

    try:
        host._set_subtab_state(host.tab_dataset, "disabled")
        host._set_button_state("btn_to_dataset", False)
        host._set_button_emphasis("btn_to_dataset_frame", False)
    except Exception:
        pass
    try:
        CAMPAIGN.set_step3_stage2_done(False)
    except Exception:
        pass


def get_step3_finish_block_message(readiness: dict | None = None) -> str:
    readiness = dict(readiness or {})
    if bool(readiness.get("ok")):
        return ""

    reason = str(readiness.get("reason", "") or "").strip().lower()
    train_images = int(readiness.get("train_images", 0) or 0)
    val_images = int(readiness.get("val_images", 0) or 0)
    test_images = int(readiness.get("test_images", 0) or 0)
    validation_msg = str(readiness.get("validation_message", "") or "").strip()

    if reason == "missing_char_boxes":
        min_exportable_plates = int(readiness.get("min_exportable_plate_count", 10) or 10)
        return str(readiness.get("message") or "").strip() or (
            f"E3 wymaga co najmniej {min_exportable_plates} tablic perfect z poprawnymi boxami znaków i etykietami. "
            "Wróć do PZ2, oznacz znaki na tablicach i ponownie wykonaj eksport w PZ3."
        )

    if reason == "invalid_char_dataset":
        if validation_msg in {"Brak obrazów w images/val", "Brak obrazów w images/val"}:
            return (
                f"Dataset znaków nadal nie ma walidacji: train={train_images}, val={val_images}, test={test_images}. "
                "W PZ3 przebuduj eksport źródłowy, a wariant treningowy/split przygotuj w Z4."
            )
        if validation_msg in {"Brak obrazów w images/train", "Brak obrazów w images/train"}:
            return "Dataset znaków nie ma jeszcze danych treningowych. W PZ3 przebuduj eksport źródłowy, a split przygotuj w Z4."
        return (
            f"Dataset znaków nadal nie jest gotowy do treningu: train={train_images}, val={val_images}, test={test_images}. "
            "W PZ3 popraw eksport źródłowy, a wariant treningowy/split przygotuj w Z4."
        )

    return "Aby zakończyć etap 3, przygotuj źródłowy dataset znaków w PZ3, a wariant treningowy/split przygotuj w Z4."


def set_step3_finish_hint(host: "CharacterAnnotationTab", text: str = "", tone: str = "muted") -> None:
    action_card = getattr(host, "step3_finish_action_card", None)
    if action_card is not None:
        try:
            action_card.set_description(str(text or "").strip(), tone=str(tone or "muted"))
            return
        except Exception:
            pass

    label = getattr(host, "step3_finish_hint_lbl", None)
    if label is None:
        return

    msg = str(text or "").strip()
    try:
        host._set_inline_status_label_state(label, text=msg, tone=tone, emphasis=False)
    except Exception:
        try:
            label.configure(text=msg)
        except Exception:
            pass


def has_any_step3_export_outputs(host: "CharacterAnnotationTab") -> bool:
    try:
        if getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name():
            summary = host._read_step3_export_summary()
            if not isinstance(summary, dict) or not summary:
                return False
            if summary.get("gold_dataset_created") is False or summary.get("gold_dataset_valid") is False:
                return False
            summary_dataset_path = str(summary.get("gold_dataset_path", "") or "").strip()
            if not (summary.get("gold_dataset_created") and summary_dataset_path):
                return False
            try:
                return Path(summary_dataset_path).exists()
            except Exception:
                return False
        return host._get_preferred_step3_training_dataset_dir() is not None
    except Exception:
        return False


def return_to_wizard_for_step3_rework(host: "CharacterAnnotationTab") -> None:
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass

    focus_step = 3
    status_message = "Wracasz do grafu. Praca w Z3 pozostaje w toku."
    status_tone = "info"
    t06_session_recorded = False
    try:
        if CAMPAIGN.get_active_project_name():
            current_step = int(CAMPAIGN.get_current_step() or 3)
            current_status = str(CAMPAIGN.get_step3_status() or "").strip().lower() or "pending"
            readiness = host._get_campaign_step3_training_readiness()
            ready_summary = _read_ready_step3_export_summary(host)
            ready_for_approval = bool(
                (
                    host._has_any_step3_export_outputs()
                    and bool(readiness.get("ok"))
                )
                or bool(ready_summary)
            )

            if current_status == "approved":
                _mark_t06_z3_work_session(
                    host,
                    state="completed",
                    substep=3,
                    reason="return_to_graph_approved",
                    force=True,
                )
                t06_session_recorded = True
                CAMPAIGN.set_current_step(max(4, current_step))
                focus_step = 4
                status_message = f"Wracasz do grafu. {CHAR_WORK_GATE_DISPLAY_ID} jest już zatwierdzona, więc możesz kontynuować kolejny krok."
                status_tone = "info"
            elif ready_for_approval:
                try:
                    _mark_t06_pz2_contract(host, reason="return_ready_to_graph", force=True)
                    summary = dict(ready_summary or host._read_step3_export_summary() or {})
                    _mark_t06_pz3_contract(host, summary, reason="return_ready_to_graph", force=True)
                    if not bool(summary.get("gold_dataset_created")):
                        dataset_path = str(
                            readiness.get("ready_dataset")
                            or readiness.get("dataset_hint")
                            or readiness.get("dataset_path")
                            or ""
                        ).strip()
                        if dataset_path:
                            _mark_t06_contract(
                                host,
                                "pz3_char_dataset",
                                {
                                    "fulfilled": True,
                                    "product": "char_yolo_dataset",
                                    "source": "PZ3",
                                    "reason": "return_ready_to_graph",
                                    "dataset_path": dataset_path,
                                    "exportable_plate_count": int(readiness.get("exportable_plate_count", 0) or 0),
                                    "exportable_char_count": int(readiness.get("exportable_char_count", 0) or 0),
                                    "perfect_count": int(readiness.get("perfect_count", 0) or 0),
                                    "gold_dataset_valid": True,
                                    "fulfilled_at": datetime.now().isoformat(timespec="seconds"),
                                },
                            )
                    _mark_t06_z3_work_session(
                        host,
                        state="completed",
                        substep=3,
                        reason="return_ready_to_graph",
                        force=True,
                    )
                    t06_session_recorded = True
                except Exception as exc:
                    logger.debug(f"Nie udało się domknąć kontraktu T06/PZ3 przy powrocie do grafu: {exc}")
                CAMPAIGN.set_current_step(3)
                CAMPAIGN.set_step3_ready()
                status_message = f"Wracasz do grafu. Dataset znaków jest gotowy do zatwierdzenia na {CHAR_WORK_GATE_DISPLAY_ID}."
                status_tone = "success"
            elif current_status == "needs_rework":
                _mark_t06_z3_work_session(host, state="paused", substep=3, reason="return_to_graph_needs_rework")
                t06_session_recorded = True
                CAMPAIGN.set_current_step(3)
                CAMPAIGN.set_step3_needs_rework()
                status_message = f"Wracasz do grafu w trybie poprawy pracy {CHAR_WORK_GATE_DISPLAY_ID}."
                status_tone = "warning"
            else:
                _mark_t06_z3_work_session(host, state="paused", substep=3, reason="return_to_graph_pending")
                t06_session_recorded = True
                CAMPAIGN.set_current_step(3)
                CAMPAIGN.set_step3_pending()
    except Exception as e:
        logger.debug(f"Nie udało się ustawić stanu powrotu dla kroku 3: {e}")
    finally:
        if not t06_session_recorded:
            try:
                _mark_t06_z3_work_session(host, state="paused", substep=3, reason="return_to_graph")
            except Exception:
                pass

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(step_num=focus_step)
            except Exception:
                pass
            campaign_tab._rebuild_roadmap_ui()
            campaign_tab._refresh_dashboard()
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć wizarda po powrocie z kroku 3: {e}")

    try:
        host.app.open_controlled_tab("campaign")
        host.app.update_campaign_tab_access()
        host.app.update_status(status_message, status_tone)
    except Exception as e:
        logger.debug(f"Nie udało się wrócić do grafu dla kroku 3: {e}")


def _step3_preview_ready_for_pz2(host: "CharacterAnnotationTab") -> bool:
    try:
        return bool(host.can_restore_step3_substep(2))
    except Exception:
        pass

    preview_dir_raw = ""
    try:
        preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        preview_dir_raw = ""
    if not preview_dir_raw:
        try:
            preview_dir_raw = str(host._get_saved_step3_preview_dir(require_plates=True) or "").strip()
        except Exception:
            preview_dir_raw = ""
    if not preview_dir_raw:
        return False
    try:
        return bool(
            host._is_usable_step3_preview_dir(
                preview_dir_raw,
                require_plates=True,
                check_campaign_inflated=False,
            )
        )
    except TypeError:
        try:
            return bool(host._is_usable_step3_preview_dir(preview_dir_raw, require_plates=True))
        except Exception:
            return False
    except Exception:
        return False


def _request_t05_work_modal_on_campaign_graph(
    host: "CharacterAnnotationTab",
    *,
    reason: str = "",
) -> None:
    try:
        campaign_tab = host.app.tabs.get("campaign")
    except Exception:
        campaign_tab = None
    if campaign_tab is None:
        return
    try:
        campaign_tab._pending_gate_action_modal_edge_key = "e3_to_e4"
        campaign_tab._pending_gate_action_modal_reason = str(reason or "step3_return_to_t05_work").strip()
        campaign_tab._pending_gate_action_modal_attempts = 12
        campaign_tab._campaign_graph_selected_edge_key = "e3_to_e4"
        if hasattr(CAMPAIGN, "set_graph_selected_edge_key"):
            CAMPAIGN.set_graph_selected_edge_key("e3_to_e4")
    except Exception:
        pass


def return_to_t05_work_after_step3_pz1(host: "CharacterAnnotationTab") -> None:
    """Finish campaign PZ1 and hand the next choice back to the T05 work modal."""
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass

    ready_for_pz2 = _step3_preview_ready_for_pz2(host)
    try:
        CAMPAIGN.set_current_step(3)
        CAMPAIGN.set_step3_substep(1)
        CAMPAIGN.set_step3_stage1_done(bool(ready_for_pz2))
        CAMPAIGN.set_step3_stage2_done(False)
        CAMPAIGN.set_step3_pending()
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać stanu PZ1 przed powrotem do T05: {exc}")

    try:
        host._campaign_force_pz2_entry = False
        host._campaign_force_pz3_entry = False
        host._campaign_force_detect_entry = False
        host._campaign_graph_entry_context = {}
    except Exception:
        pass

    try:
        _mark_t06_z3_work_session(
            host,
            state="ready_for_pz2" if ready_for_pz2 else "waiting_for_pz2",
            substep=1,
            reason="pz1_ready_return_to_t05_work" if ready_for_pz2 else "pz1_return_to_t05_without_preview",
            force=True,
        )
    except Exception:
        pass
    _invalidate_step3_campaign_ui_caches()

    _request_t05_work_modal_on_campaign_graph(
        host,
        reason="pz1_ready_return_to_t05_work" if ready_for_pz2 else "pz1_return_to_t05_without_preview",
    )

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(step_num=3)
            except Exception:
                pass
            campaign_tab._rebuild_roadmap_ui()
            campaign_tab._refresh_dashboard()
    except Exception as exc:
        logger.debug(f"Nie udało się odświeżyć grafu po PZ1: {exc}")

    try:
        host.app.open_controlled_tab("campaign")
        host.app.update_campaign_tab_access()
        if ready_for_pz2:
            host.app.update_status(
                f"PZ1 jest zatwierdzone. W pracy bramki {CHAR_WORK_GATE_DISPLAY_ID} wybierz kolejny krok: PZ2.",
                "success",
            )
        else:
            host.app.update_status(
                f"PZ1 nie ma jeszcze gotowego zestawu tablic. Wróć do pracy bramki {CHAR_WORK_GATE_DISPLAY_ID}.",
                "warning",
            )
    except Exception as exc:
        logger.debug(f"Nie udało się wrócić do grafu po PZ1: {exc}")


def return_to_wizard_from_step3_pz2(host: "CharacterAnnotationTab") -> None:
    campaign_tab = getattr(host.app, "tabs", {}).get("campaign")
    if campaign_tab is not None:
        close_graph_dialogs(campaign_tab.frame)
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass
    try:
        flush = getattr(host, "_flush_scheduled_preview_metadata_save", None)
        if callable(flush):
            flush()
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać odłożonych zmian PZ2 przed powrotem do grafu: {exc}")
    try:
        host._sync_step3_access_from_preview_state(
            getattr(host, "preview_metadata", None),
            mark_current_work=True,
            mark_reason="return_to_graph_pz2_ready",
        )
    except TypeError:
        try:
            host._sync_step3_access_from_preview_state(getattr(host, "preview_metadata", None))
        except Exception:
            pass
    except Exception:
        pass

    try:
        pz2_ready = bool(host._campaign_step3_pz2_current_contract_ready())
    except Exception:
        pz2_ready = False

    try:
        CAMPAIGN.set_current_step(3)
        if pz2_ready:
            CAMPAIGN.set_step3_stage2_done(True)
        else:
            CAMPAIGN.set_step3_stage2_done(False)
        CAMPAIGN.set_step3_pending()
    except Exception as exc:
        logger.debug(f"Nie udało się ustawić stanu T05/PZ2 przy powrocie do grafu: {exc}")

    try:
        _mark_t06_z3_work_session(
            host,
            state="ready_for_pz3" if pz2_ready else "paused",
            substep=2,
            reason="return_to_graph_pz2_ready" if pz2_ready else "return_to_graph_pz2_pending",
            force=True,
        )
    except Exception:
        pass

    try:
        host._campaign_force_pz2_entry = False
        host._campaign_force_pz3_entry = False
        host._campaign_force_detect_entry = False
        host._campaign_graph_entry_context = {}
    except Exception:
        pass
    _invalidate_step3_campaign_ui_caches()

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(step_num=3)
            except Exception:
                pass
            campaign_tab._rebuild_roadmap_ui()
            campaign_tab._refresh_dashboard()
    except Exception as exc:
        logger.debug(f"Nie udało się odświeżyć grafu po powrocie z PZ2: {exc}")

    try:
        host.app.open_controlled_tab("campaign")
        host.app.update_campaign_tab_access()
        if pz2_ready:
            host.app.update_status(
                f"PZ2 jest zapisane. W pracy bramki {CHAR_WORK_GATE_DISPLAY_ID} wybierz krok 2: utwórz dataset znaków w PZ3.",
                "success",
            )
        else:
            host.app.update_status(
                f"Wracasz do grafu. Praca w PZ2 bramki {CHAR_WORK_GATE_DISPLAY_ID} pozostaje w toku.",
                "info",
            )
    except Exception as exc:
        logger.debug(f"Nie udało się wrócić do grafu z PZ2: {exc}")


def return_step3_result_to_wizard(host: "CharacterAnnotationTab", summary: dict) -> None:
    """
    Single return contract for campaign E3.

    If a valid gold dataset exists, E3 becomes ready for approval.
    Otherwise the wizard stays on E3 in rework mode.
    """
    gold_ok = bool(summary.get("gold_dataset_created")) and bool(summary.get("gold_dataset_valid", True))

    if gold_ok:
        mark_step3_dataset_exported_for_campaign(host, summary, reason="dataset_exported")
        status_msg = (
            "Z3 przygotowało poprawny dataset znaków. "
            f"Wróć do grafu i zatwierdź bramkę {CHAR_WORK_GATE_DISPLAY_ID}, aby odblokować dalszą pracę."
        )
        status_kind = "success"
    else:
        _mark_t06_z3_work_session(host, state="paused", substep=3, reason="dataset_not_ready")
        CAMPAIGN.set_current_step(3)
        CAMPAIGN.set_step3_needs_rework()
        status_msg = (
            "Z3 nie utworzyło datasetu znaków. "
            f"Wracasz do grafu w trybie poprawy pracy {CHAR_WORK_GATE_DISPLAY_ID}."
        )
        status_kind = "warning"

    try:
        host._campaign_force_pz2_entry = False
        host._campaign_force_pz3_entry = False
        host._campaign_force_detect_entry = False
        host._campaign_graph_entry_context = {}
    except Exception:
        pass
    _invalidate_step3_campaign_ui_caches()

    try:
        campaign_tab = host.app.tabs.get("campaign")
        if campaign_tab:
            try:
                campaign_tab.request_wizard_stage_focus(step_num=3)
            except Exception:
                pass
            campaign_tab._rebuild_roadmap_ui()
            campaign_tab._refresh_dashboard()
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć Wizarda po etapie 3: {e}")

    try:
        host.app.open_controlled_tab("campaign")
        host.app.update_campaign_tab_access()
    except Exception as e:
        logger.debug(f"Nie udało się wrócić do Wizarda po etapie 3: {e}")

    try:
        host.app.update_status(status_msg, status_kind)
    except Exception:
        pass


def finalize_step3_from_existing_outputs(host: "CharacterAnnotationTab") -> None:
    """
    Soft-finalize E3 from artifacts that already exist after PZ3 export.

    This function does not build a dataset. It evaluates existing outputs,
    writes export_summary.json and returns the result to the wizard.
    """
    gold_dataset_path = ""
    review_pack_path = ""

    summary_override = host._read_step3_export_summary()
    if (
        getattr(host, "_step3_linear_mode", False)
        and CAMPAIGN.get_active_project_name()
        and isinstance(summary_override, dict)
        and summary_override
        and (
            summary_override.get("gold_dataset_created") is False
            or summary_override.get("gold_dataset_valid") is False
        )
    ):
        host._return_step3_result_to_wizard(summary_override)
        return

    campaign_readiness = None
    try:
        if getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name():
            campaign_readiness = host._get_campaign_step3_training_readiness()
    except Exception:
        campaign_readiness = None

    preferred_dataset_dir = None
    if isinstance(campaign_readiness, dict):
        ready_dataset = str(campaign_readiness.get("ready_dataset", "") or "").strip()
        if bool(campaign_readiness.get("ok")) and ready_dataset:
            preferred_dataset_dir = Path(ready_dataset)
    else:
        preferred_dataset_dir = host._get_preferred_step3_training_dataset_dir()

    if preferred_dataset_dir is not None:
        gold_dataset_path = str(preferred_dataset_dir)

    campaign_chars_dir = getattr(host, "_campaign_chars_dir", None)
    if campaign_chars_dir:
        chars_root = Path(campaign_chars_dir)
        review_dir = chars_root / "review"
        if review_dir.exists():
            review_pack_path = str(review_dir)

    summary = host._build_step3_export_summary(
        gold_dataset_path=gold_dataset_path,
        review_pack_path=review_pack_path,
        retry_pack_path="",
        note=(
            "Finalizacja kroku 3 na podstawie najlepszego dostępnego datasetu "
            "znaków projektu (merged preferowany, fallback do istniejącego datasetu)."
        ),
    )

    if isinstance(campaign_readiness, dict):
        summary["gold_dataset_valid"] = bool(campaign_readiness.get("ok"))
        summary["gold_dataset_validation_message"] = str(campaign_readiness.get("message", "") or "").strip()

    host._write_step3_export_summary(summary)
    host._return_step3_result_to_wizard(summary)


def resolve_step3_campaign_action_command(host: "CharacterAnnotationTab", command_id: str):
    normalized = str(command_id or "").strip().lower()
    if normalized in {"return_to_wizard_rework", "return_to_wizard_step3"}:
        return host._return_to_wizard_for_step3_rework
    if normalized == "finalize_step3":
        return host._finalize_step3_from_existing_outputs
    return None


def _campaign_preview_entry_key(xml_path, images_dir):
    """Reuse an editor only within the same iteration and unchanged sources."""
    try:
        preview_dir = CAMPAIGN.get_step3_preview_dir()
        if not preview_dir or not xml_path or not images_dir:
            return None
        return (
            CAMPAIGN.get_active_project_name(), CAMPAIGN.get_current_iteration_num(),
            file_signature(xml_path), file_signature(images_dir),
            file_signature(Path(preview_dir) / "metadata.json"),
        )
    except OSError:
        return None


def _can_reuse_campaign_preview(host, key):
    return bool(
        key is not None
        and key == getattr(host, "_last_campaign_preview_entry_key", None)
        and path_key(getattr(host, "_loaded_meta_path", None)) == key[-1][0]
        and getattr(host, "_detect_tab_built", False)
        and getattr(host, "preview_metadata", None)
        and getattr(host, "_listbox_pid_by_index", None)
    )


def open_campaign_step3_entry(
    host: "CharacterAnnotationTab",
    preferred_source_context: dict | None = None,
) -> dict:
    entry_started = time.perf_counter()
    entry_phase_started = entry_started
    entry_phase_marks: list[str] = []

    def _mark_entry_phase(label: str, *, threshold_ms: float = 180.0) -> None:
        nonlocal entry_phase_started
        now = time.perf_counter()
        delta_ms = (now - entry_phase_started) * 1000.0
        total_ms = (now - entry_started) * 1000.0
        entry_phase_started = now
        if delta_ms >= threshold_ms:
            entry_phase_marks.append(f"{label}={delta_ms:.0f}ms/{total_ms:.0f}ms")

    if not CAMPAIGN.get_active_project_name() or int(CAMPAIGN.get_current_step() or 0) < 3:
        return {"ok": False, "reason": "campaign_inactive"}

    # Finish edits to the previous loaded run before replacing its UI context.
    flush_preview = getattr(host, "_flush_scheduled_preview_metadata_save", None)
    if callable(flush_preview):
        flush_preview()
    writer = getattr(host, "_preview_autosave_writer", None)
    if writer is not None:
        writer.close()
        host._preview_autosave_writer = None

    raw_dir = CAMPAIGN.get_dir("raw")
    auto_dir = CAMPAIGN.get_dir("auto_ann")
    chars_dir = CAMPAIGN.get_dir("chars")
    datasets_dir = CAMPAIGN.get_dir("datasets")

    if raw_dir is None or auto_dir is None:
        return {"ok": False, "reason": "missing_campaign_dirs"}

    incoming_source_context = preferred_source_context if isinstance(preferred_source_context, dict) else {}

    def _context_points_to_ready_source(context: dict | None) -> bool:
        if not isinstance(context, dict) or not context:
            return False
        try:
            restore_run = (
                context.get("restore_run_dir")
                or context.get("run_dir")
                or context.get("annotation_run_dir")
            )
            xml_value = (
                context.get("xml_path")
                or context.get("source_xml")
                or context.get("source_xml_path")
            )
            images_value = (
                context.get("input_dir")
                or context.get("images_dir")
                or context.get("source_images_dir")
            )
            xml_path = Path(str(xml_value or "").strip()) if str(xml_value or "").strip() else None
            if xml_path is None and str(restore_run or "").strip():
                xml_path = Path(str(restore_run).strip()) / "annotations.xml"
            images_dir = Path(str(images_value or "").strip()) if str(images_value or "").strip() else None
            return bool(
                xml_path is not None
                and xml_path.exists()
                and images_dir is not None
                and images_dir.exists()
                and images_dir.is_dir()
            )
        except Exception:
            return False

    def _registry_points_to_ready_char_source() -> bool:
        try:
            bundle = dict(host._get_campaign_iteration_artifact_bundle() or {})
        except Exception:
            bundle = {}
        char_effective = dict(bundle.get("char_effective_source") or {})
        if not char_effective:
            return False
        try:
            run_dir = Path(str(char_effective.get("run_dir") or "").strip())
            images_dir = Path(str(char_effective.get("images_dir") or "").strip())
            xml_path = Path(str(char_effective.get("xml_path") or "").strip())
            return bool(
                run_dir.exists()
                and run_dir.is_dir()
                and images_dir.exists()
                and images_dir.is_dir()
                and xml_path.exists()
                and xml_path.is_file()
            )
        except Exception:
            return False

    try:
        if (
            str(CAMPAIGN.get_iteration_target() or "").strip().lower() == "char"
            and not _context_points_to_ready_source(incoming_source_context)
            and not _registry_points_to_ready_char_source()
        ):
            annotation_tab = getattr(getattr(host, "app", None), "tabs", {}).get("annotation")
            if annotation_tab is not None:
                flush_approved = getattr(annotation_tab, "_flush_preview_approved_persist", None)
                if callable(flush_approved):
                    flush_approved()
                rebuild_source = getattr(annotation_tab, "_build_campaign_char_effective_source", None)
                if callable(rebuild_source):
                    rebuild_source()
    except Exception as exc:
        logger.debug(f"Nie udało się odświeżyć kanonicznego źródła E3 przed wejściem do Z3: {exc}")
    _mark_entry_phase("source_flush")

    iter_num = CAMPAIGN.get_current_iteration_num()
    try:
        iteration_source_dir = (
            CAMPAIGN.get_iteration_image_source_dir(iter_num)
            or CAMPAIGN.get_iteration_raw_dir(iter_num)
        )
    except Exception:
        iteration_source_dir = None
    legacy_default_folder = Path(raw_dir) / f"Iteracja_{iter_num:03d}"
    default_folder = (
        Path(iteration_source_dir)
        if iteration_source_dir is not None and Path(iteration_source_dir).exists()
        else legacy_default_folder
    )
    folder = default_folder

    source_context = preferred_source_context if isinstance(preferred_source_context, dict) else {}
    if not source_context:
        try:
            source_context = host._get_registry_preferred_step3_source_candidate() or {}
        except Exception:
            source_context = {}
    preferred_run_dir = None
    preferred_xml = ""
    preferred_input_dir = None
    using_preferred_source = False

    try:
        restore_run_dir = (
            source_context.get("restore_run_dir")
            or source_context.get("run_dir")
            or source_context.get("annotation_run_dir")
        )
        if restore_run_dir:
            candidate_run_dir = Path(restore_run_dir)
            candidate_xml = candidate_run_dir / "annotations.xml"
            if candidate_run_dir.exists() and candidate_run_dir.is_dir() and candidate_xml.exists():
                preferred_run_dir = candidate_run_dir
                preferred_xml = str(candidate_xml)
    except Exception:
        preferred_run_dir = None
        preferred_xml = ""

    if not preferred_xml:
        try:
            explicit_xml = (
                source_context.get("xml_path")
                or source_context.get("source_xml")
                or source_context.get("source_xml_path")
            )
            if explicit_xml:
                candidate_xml = Path(explicit_xml)
                if candidate_xml.exists() and candidate_xml.is_file():
                    preferred_xml = str(candidate_xml)
                    if preferred_run_dir is None:
                        preferred_run_dir = candidate_xml.parent
        except Exception:
            preferred_xml = ""

    try:
        input_dir = (
            source_context.get("input_dir")
            or source_context.get("images_dir")
            or source_context.get("source_images_dir")
        )
        if input_dir:
            candidate_input_dir = Path(input_dir)
            if candidate_input_dir.exists() and candidate_input_dir.is_dir():
                preferred_input_dir = candidate_input_dir
    except Exception:
        preferred_input_dir = None

    if preferred_input_dir is not None:
        folder = preferred_input_dir
    elif not folder.exists():
        folder = Path(raw_dir)

    latest_xml = preferred_xml
    if latest_xml:
        using_preferred_source = True
    else:
        try:
            run_dirs = PROJECT_CACHE.list_annotation_run_dirs(Path(auto_dir), require_xml=True)
            xml_files = [Path(run_dir) / "annotations.xml" for run_dir in run_dirs]
        except Exception:
            xml_files = []
        latest_xml = str(max(xml_files, key=lambda p: p.stat().st_mtime)) if xml_files else ""
    _mark_entry_phase("source_resolution")

    entry_preview_key = _campaign_preview_entry_key(latest_xml, folder)
    reuse_loaded_preview = _can_reuse_campaign_preview(host, entry_preview_key)
    if not reuse_loaded_preview:
        host._set_preview_dir_runtime_value("", persist_registry=False)
        try:
            host._cancel_preview_char_label_interaction()
        except Exception:
            pass
        host.preview_metadata = {}
        host.preview_plate_ids = []
        host._loaded_meta_path = None
        host._loaded_meta_mtime = None
        host._preview_active_pid = None
        host._pz3_selected_path = "dataset"
        host._pz3_cvat_expanded = False
        try:
            host._reset_pz3_runtime_ui(collapse_cards=False)
        except Exception:
            pass

        try:
            host.plates_listbox.delete(0, 999999)
        except Exception:
            pass

        try:
            host.preview_canvas.delete("all")
        except Exception:
            pass

        try:
            host.preview_info_lbl.config(text="Oczekuje na nowy zestaw zdjęć...", foreground="#2980b9")
        except Exception:
            pass

        for attr_name in ("annotation_run_dir_var", "images_dir_var", "xml_path_var"):
            try:
                getattr(host, attr_name).set("")
            except Exception:
                pass
    _mark_entry_phase("clear_preview_ui")

    try:
        latest_run_dir = str(Path(latest_xml).parent) if latest_xml else ""
    except Exception:
        latest_run_dir = ""

    try:
        effective_source_context = dict(source_context or {})
    except Exception:
        effective_source_context = {}

    if latest_xml:
        effective_source_context["xml_path"] = str(latest_xml)
    if latest_run_dir:
        effective_source_context["restore_run_dir"] = str(latest_run_dir)
        effective_source_context.setdefault("run_dir", str(latest_run_dir))
    if folder.exists():
        effective_source_context["input_dir"] = str(folder)
        effective_source_context.setdefault("images_dir", str(folder))
    source_context = effective_source_context

    try:
        host._campaign_graph_entry_context = {
            key: str(source_context.get(key) or "").strip()
            for key in (
                "source",
                "graph_edge_key",
                "graph_gate_id",
                "graph_gate_label",
                "graph_transition_title",
                "graph_transition_source",
                "graph_transition_target",
                "graph_path_key",
                "target_substep",
                "graph_target_substep",
                "preferred_substep",
                "force_pz2",
                "force_pz3",
            )
            if str(source_context.get(key) or "").strip()
        }
    except Exception:
        host._campaign_graph_entry_context = {}
    _mark_entry_phase("graph_context")

    host.set_pending_z2_annotation_source(
        xml_path=latest_xml,
        images_dir=(str(folder) if folder.exists() else ""),
        run_dir=latest_run_dir,
    )
    if latest_xml:
        try:
            host._source_binding_sync_in_progress = True
            host.annotation_run_dir_var.set(latest_run_dir)
            host.xml_path_var.set(str(latest_xml))
            host.images_dir_var.set(str(folder) if folder.exists() else "")
        except Exception:
            pass
        finally:
            try:
                host._source_binding_sync_in_progress = False
            except Exception:
                pass
    try:
        # Wejście kampanijne E3 nie może dziedziczyć wyniku walidacji PZ1/PZ2
        # z trybu swobodnego albo poprzedniego źródła. Stary licznik potrafił
        # odrzucić poprawny preview i uruchomić zbędne auto-wycinanie.
        host._extract_last_source_binding_result = {}
    except Exception:
        pass

    try:
        saved_extract_state = CAMPAIGN.get_step3_extract_state() or {}
    except Exception:
        saved_extract_state = {}

    try:
        saved_substep = int(CAMPAIGN.get_step3_substep() or 1)
    except Exception:
        saved_substep = 1

    force_detect_entry = False
    force_dataset_entry = False
    try:
        target_substep_hint = str(
            source_context.get("target_substep")
            or source_context.get("graph_target_substep")
            or source_context.get("preferred_substep")
            or ""
        ).strip().lower()
        force_detect_entry = (
            bool(source_context.get("force_pz2"))
            or target_substep_hint in {"2", "detect", "pz2", "z3_pz2"}
        )
        force_dataset_entry = (
            bool(source_context.get("force_pz3"))
            or target_substep_hint in {"3", "dataset", "pz3", "z3_pz3"}
        ) and not force_detect_entry
    except Exception:
        force_detect_entry = False
        force_dataset_entry = False
    if force_detect_entry:
        saved_substep = 2
        try:
            CAMPAIGN.set_step3_substep(2)
        except Exception:
            pass
    elif force_dataset_entry:
        saved_substep = 3
        try:
            CAMPAIGN.set_step3_substep(3)
        except Exception:
            pass
    try:
        host._campaign_force_detect_entry = bool(force_detect_entry)
        host._campaign_force_pz2_entry = bool(force_detect_entry)
        host._campaign_force_pz3_entry = bool(force_dataset_entry)
    except Exception:
        pass
    _mark_t06_z3_work_session(
        host,
        state="active",
        substep=saved_substep,
        reason="enter_z3",
        context=source_context,
    )
    _mark_entry_phase("bind_source_vars")

    try:
        has_saved_step3_progress = bool(
            saved_substep > 1
            or bool(CAMPAIGN.is_step3_stage1_done())
            or bool(CAMPAIGN.is_step3_stage2_done())
            or str(saved_extract_state.get("entry_mode", "") or "").strip()
            or str(saved_extract_state.get("workflow_step", "entry") or "entry").strip().lower() != "entry"
            or str(saved_extract_state.get("annotation_run_dir", "") or "").strip()
            or str(saved_extract_state.get("xml_path", "") or "").strip()
            or str(saved_extract_state.get("images_dir", "") or "").strip()
        )
    except Exception:
        has_saved_step3_progress = False

    preview_missing_or_unusable = True
    try:
        saved_preview_dir = str(CAMPAIGN.get_step3_preview_dir() or "").strip()
    except Exception:
        saved_preview_dir = ""
    if saved_preview_dir:
        try:
            preview_missing_or_unusable = not bool(
                host._is_usable_step3_preview_dir(
                    saved_preview_dir,
                    require_plates=True,
                    check_campaign_inflated=False,
                )
            )
        except Exception:
            preview_missing_or_unusable = False

    if latest_xml and (not has_saved_step3_progress or preview_missing_or_unusable):
        try:
            CAMPAIGN.set_step3_extract_state(
                entry_mode="continue",
                workflow_step="start",
                annotation_run_dir=latest_run_dir,
                xml_path=latest_xml,
                images_dir=(str(folder) if folder.exists() else ""),
            )
        except Exception as exc:
            logger.debug(f"Nie udało się ustawić domyslnego wejscia do Z3/PZ1: {exc}")
        try:
            host._sync_campaign_step3_artifact_registry(
                entry_mode="continue",
                workflow_step="start",
                annotation_run_dir=latest_run_dir,
                xml_path=latest_xml,
                images_dir=(str(folder) if folder.exists() else ""),
            )
        except Exception:
            pass

    _mark_entry_phase("saved_step3_state")

    host._campaign_chars_dir = str(chars_dir) if chars_dir else None
    host._campaign_datasets_dir = str(datasets_dir) if datasets_dir else None

    registry_bundle = host._get_campaign_iteration_artifact_bundle()
    registry_char_model = dict(registry_bundle.get("char_model") or {})
    try:
        effective_char_model = dict(
            CAMPAIGN.get_effective_project_model(
                "char",
                before_iteration=int(CAMPAIGN.get_current_iteration_num() or 1),
            )
            or {}
        )
    except Exception:
        effective_char_model = {}
    char_model_path = str(effective_char_model.get("path") or "").strip()
    if not char_model_path or not Path(char_model_path).exists():
        char_model_path = str(registry_char_model.get("path") or "").strip()
    if char_model_path and Path(char_model_path).exists():
        host.detection_method_var.set("BOTH")
        host.yolo_model_path_var.set(char_model_path)

        try:
            version, size = host._infer_yolo_arch_from_model_path(char_model_path)
            if version in {"8", "11", "26"}:
                host.yolo_model_version_var.set(version)
            if size in {"n", "s", "m", "l", "x"}:
                host.yolo_model_size_var.set(size)
        except Exception as exc:
            logger.debug(f"Nie udało się odczytać architektury YOLO z nazwy modelu: {exc}")

        try:
            host._sync_yolo_model_binding()
        except Exception as exc:
            logger.debug(f"Nie udało się zsynchronizowac ścieżki modelu YOLO: {exc}")

        try:
            host._update_yolo_visibility()
        except Exception as exc:
            logger.debug(f"Nie udało się odświeżyć widoku YOLO w Zakładce Znaków: {exc}")
    else:
        try:
            host.detection_method_var.set("OCR")
            host.yolo_model_path_var.set("")
            host._update_yolo_visibility()
        except Exception as exc:
            logger.debug(f"Nie udało się ustawić trybu OCR dla braku modelu znaków: {exc}")
        try:
            host._sync_yolo_model_binding()
        except Exception:
            pass

        try:
            host._update_yolo_visibility()
        except Exception:
            pass

    _mark_entry_phase("model_binding")

    try:
        if not force_detect_entry:
            host._restore_preview_context_from_project()
    except Exception as exc:
        logger.debug(f"Nie udało się przywrocic preview projektu: {exc}")
    _mark_entry_phase("restore_preview_context")

    try:
        host._campaign_pz2_preview_loaded_this_entry = reuse_loaded_preview
    except Exception:
        pass

    def _can_continue_forced_pz3_without_reextract(refresh_state: dict | None = None) -> bool:
        if not force_dataset_entry:
            return False
        try:
            if host.can_restore_step3_substep(3):
                return True
        except Exception:
            pass
        try:
            contracts = dict((CAMPAIGN.get_iteration_state() or {}).get("t06_contracts") or {})
            pz2_contract = dict(contracts.get("pz2_char_boxes") or {})
            pz2_ready = bool(pz2_contract.get("fulfilled") or CAMPAIGN.is_step3_stage2_done())
        except Exception:
            pz2_ready = False
        if not pz2_ready:
            return False
        preview_dir_raw = ""
        try:
            preview_dir_raw = str((refresh_state or {}).get("preview_dir") or "").strip()
        except Exception:
            preview_dir_raw = ""
        if not preview_dir_raw:
            try:
                preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
            except Exception:
                preview_dir_raw = ""
        if not preview_dir_raw:
            try:
                preview_dir_raw = str(host._get_saved_step3_preview_dir(require_plates=True) or "").strip()
            except Exception:
                preview_dir_raw = ""
        if not preview_dir_raw:
            return False
        try:
            return bool(
                host._is_usable_step3_preview_dir(
                    preview_dir_raw,
                    require_plates=True,
                    check_campaign_inflated=False,
                )
            )
        except TypeError:
            try:
                return bool(host._is_usable_step3_preview_dir(preview_dir_raw, require_plates=True))
            except Exception:
                return False
        except Exception:
            return False

    def _refresh_forced_pz3_dataset_surface() -> None:
        for method_name in (
            "_refresh_pz3_cards_ui",
            "_refresh_pz3_status_panel_ui",
            "_update_step3_finish_button_state",
        ):
            try:
                method = getattr(host, method_name, None)
                if callable(method):
                    method()
            except Exception as exc:
                logger.debug(f"Nie udało się odświeżyć PZ3 po wejściu T06: {method_name}: {exc}")

    entry_needs_reextract = False
    try:
        entry_refresh_state = get_campaign_step3_source_refresh_state(
            host,
            preferred_source_context=source_context,
        )
        entry_needs_reextract = bool(entry_refresh_state.get("needs_reextract"))
        if force_dataset_entry and entry_needs_reextract:
            try:
                if _can_continue_forced_pz3_without_reextract(entry_refresh_state):
                    logger.info(
                        "[Z3] Pomijam ponowne wycinanie przy wejściu T06->PZ3: reason=%s",
                        str(entry_refresh_state.get("reason") or ""),
                    )
                    try:
                        CAMPAIGN.set_step3_stage2_done(True)
                    except Exception:
                        pass
                    entry_needs_reextract = False
                    entry_refresh_state["needs_reextract"] = False
                    entry_refresh_state["suppressed_for_pz3"] = True
            except Exception:
                pass
    except Exception:
        entry_needs_reextract = False
    _mark_entry_phase("entry_refresh_state")

    if entry_needs_reextract:
        host._step3_linear_mode = True
    else:
        direct_dataset_entry = bool(
            force_dataset_entry and _can_continue_forced_pz3_without_reextract(entry_refresh_state)
        )
        if direct_dataset_entry:
            try:
                CAMPAIGN.set_step3_stage2_done(True)
            except Exception:
                pass
            try:
                host.go_to_substep_3(force=True)
                _refresh_forced_pz3_dataset_surface()
            except Exception as exc:
                logger.debug(f"Nie udalo sie wymusic wejscia T06 do PZ3: {exc}")
        else:
            try:
                host.restore_campaign_step3_mode()
            except Exception as exc:
                logger.debug(f"Nie udało się przywrocic stanu kroku 3: {exc}")
                CAMPAIGN.reset_step3_progress()
                enter_campaign_step3_mode(host)
            if force_detect_entry:
                try:
                    host.go_to_substep_2(force=True)
                except Exception as exc:
                    logger.debug(f"Nie udalo sie wymusic wejscia T06 do PZ2: {exc}")
            elif force_dataset_entry:
                try:
                    if host.can_restore_step3_substep(3) or _can_continue_forced_pz3_without_reextract(entry_refresh_state):
                        try:
                            CAMPAIGN.set_step3_stage2_done(True)
                        except Exception:
                            pass
                        host.go_to_substep_3(force=True)
                        _refresh_forced_pz3_dataset_surface()
                    elif host.can_restore_step3_substep(2):
                        host.go_to_substep_2(force=True)
                except Exception as exc:
                    logger.debug(f"Nie udalo sie wymusic wejscia T06 do PZ3: {exc}")
    _mark_entry_phase("restore_step3_mode")

    if not force_detect_entry and not force_dataset_entry and not entry_needs_reextract:
        try:
            schedule_refresh = getattr(host, "_schedule_extract_workflow_refresh", None)
            if callable(schedule_refresh):
                schedule_refresh(delay_ms=40 if not host.is_startup_ui_ready() else 0)
            else:
                host._refresh_extract_workflow_ui()
        except Exception:
            pass

    if force_detect_entry:
        try:
            host._campaign_force_detect_entry = False
        except Exception:
            pass

    try:
        refresh_pz3_cards = getattr(host, "_refresh_pz3_cards_ui", None)
        if callable(refresh_pz3_cards) and not force_detect_entry:
            refresh_pz3_cards()
    except Exception:
        pass
    _mark_entry_phase("refresh_views")

    if not bool(getattr(host, "_campaign_step3_entry_splash_pinned", False)):
        try:
            host._hide_campaign_detect_splash()
        except Exception:
            pass

    try:
        if (not force_detect_entry and not force_dataset_entry) or entry_needs_reextract:
            host._auto_progress_campaign_step3_entry(preferred_source_context=source_context)
    except Exception as exc:
        logger.debug(f"Nie udało się automatycznie ustawić wejścia kampanii do Z3: {exc}")
    _mark_entry_phase("auto_progress")

    host._last_campaign_preview_entry_key = _campaign_preview_entry_key(latest_xml, folder)

    elapsed_ms = (time.perf_counter() - entry_started) * 1000.0
    if elapsed_ms >= 250.0:
        try:
            logger.info(
                "[Z3][PERF] open_campaign_step3_entry: total=%.1fms preferred=%s detect_entry=%s reextract=%s latest_xml=%s images_dir=%s phases=[%s]",
                elapsed_ms,
                bool(using_preferred_source),
                bool(force_detect_entry),
                bool(entry_needs_reextract),
                bool(latest_xml),
                str(folder) if folder.exists() else "",
                " ".join(entry_phase_marks),
            )
        except Exception:
            pass

    return {
        "ok": True,
        "latest_xml": latest_xml,
        "images_dir": (str(folder) if folder.exists() else ""),
        "using_preferred_source": bool(using_preferred_source),
        "preferred_run_dir": str(preferred_run_dir or ""),
        "char_model_path": char_model_path,
    }


def auto_progress_campaign_step3_entry(
    host: "CharacterAnnotationTab",
    preferred_source_context: dict | None = None,
) -> dict:
    source_context = preferred_source_context if isinstance(preferred_source_context, dict) else {}
    try:
        target_substep_hint = str(
            source_context.get("target_substep")
            or source_context.get("graph_target_substep")
            or source_context.get("preferred_substep")
            or ""
        ).strip().lower()
    except Exception:
        target_substep_hint = ""
    explicit_pz2_entry = bool(
        source_context.get("force_pz2")
        or getattr(host, "_campaign_force_pz2_entry", False)
        or target_substep_hint in {"2", "detect", "pz2", "z3_pz2"}
    )

    plan = get_campaign_step3_entry_flow_view_model(
        host,
        preferred_source_context=preferred_source_context,
    )

    result = {
        "handled": False,
        "mode": str(getattr(plan, "mode", "") or "").strip(),
        "message": str(getattr(plan, "message", "") or "").strip(),
    }
    mode = str(getattr(plan, "mode", "") or "").strip().lower()

    if getattr(plan, "should_hide_splash", False):
        try:
            host._hide_campaign_detect_splash()
        except Exception:
            pass

    if mode == "auto_extract":
        try:
            host._campaign_pz2_sync_loading = True
            host._campaign_detect_splash_force_root_surface = True
            host._set_subtab_state(host.tab_detect, "normal")
            host._set_subtab_state(host.tab_dataset, "disabled")
            host._set_subtab_state(host.tab_extract, "normal")
        except Exception as exc:
            logger.debug(f"Nie udało się przygotować ekranu oczekiwania PZ2 dla kampanii: {exc}")

    splash_already_pinned = bool(
        getattr(host, "_campaign_step3_entry_splash_pinned", False)
        and getattr(host, "_campaign_detect_splash_visible", False)
    )
    if getattr(plan, "should_show_splash", False) and not splash_already_pinned and mode != "auto_extract":
        splash_progress = None
        if (
            bool(getattr(plan, "splash_show_progress", False))
        ):
            splash_progress = 0.0
        try:
            host._show_campaign_detect_splash(
                title=str(getattr(plan, "splash_title", "") or "").strip(),
                body=str(getattr(plan, "splash_body", "") or "").strip(),
                tone=str(getattr(plan, "splash_tone", "info") or "info"),
                progress=splash_progress,
                show_progress=bool(getattr(plan, "splash_show_progress", False)),
                show_return=bool(getattr(plan, "splash_show_return", False)),
            )
        except Exception:
            pass

    if mode == "source_invalid":
        try:
            host._campaign_pz2_sync_loading = False
            host._campaign_detect_splash_force_root_surface = False
        except Exception:
            pass
        return result

    if mode == "auto_extract":
        try:
            if hasattr(host.app, "update_status"):
                host.app.update_status(
                    "Przygotowuję tablice dla Z3. Po wyodrębnieniu wrócisz do pracy bramki T05 i wybierzesz kolejny krok.",
                    "info",
                )
        except Exception:
            pass

        def _auto_start_extraction():
            try:
                if not (getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()):
                    return
                if getattr(host, "is_processing", False):
                    return
                host._run_extraction()
            except Exception as exc:
                logger.debug(f"Nie udało się automatycznie uruchomić PZ1 dla kampanii: {exc}")

        frame = getattr(host, "frame", None)
        if frame is not None:
            frame.after(0, _auto_start_extraction)
        else:
            _auto_start_extraction()

        result["handled"] = True
        return result

    target_substep = int(getattr(plan, "target_substep", 0) or 0)
    if mode == "open_detect" or target_substep == 2:
        if not explicit_pz2_entry:
            try:
                host._set_extraction_status(
                    "PZ1 jest gotowe. PZ2 otwieramy teraz wyłącznie przez modal pracy bramki T05.",
                    "success",
                )
            except Exception:
                pass
            try:
                if hasattr(host.app, "update_status"):
                    host.app.update_status(
                        "PZ1 jest gotowe. W pracy bramki T05 wybierz PZ2 jako następny krok.",
                        "info",
                    )
            except Exception:
                pass
            return result
        try:
            host._set_extraction_status(
                "Wyodrębnione tablice są już gotowe. Otwieram od razu PZ2 do pracy nad znakami.",
                "success",
            )
        except Exception:
            pass
        try:
            if hasattr(host.app, "update_status"):
                host.app.update_status(
                    "Tablice dla Z3 są już przygotowane. Otwieram od razu PZ2.",
                    "info",
                )
        except Exception:
            pass

        def _auto_open_detect():
            try:
                if not (getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()):
                    return
                host.go_to_substep_2(force=True)
            except Exception as exc:
                logger.debug(f"Nie udało się automatycznie otworzyć PZ2 dla kampanii: {exc}")

        frame = getattr(host, "frame", None)
        if frame is not None:
            frame.after(0, _auto_open_detect)
        else:
            _auto_open_detect()

        result["handled"] = True
        return result

    if mode == "restore_existing_substep":
        result["handled"] = True

    return result


def get_campaign_step3_entry_flow_view_model(
    host: "CharacterAnnotationTab",
    preferred_source_context: dict | None = None,
) -> Step3EntryFlowViewModel:
    try:
        if not (getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()):
            return Step3EntryFlowViewModel()
    except Exception:
        return Step3EntryFlowViewModel()

    try:
        saved_substep = int(CAMPAIGN.get_step3_substep() or 1)
    except Exception:
        saved_substep = 1

    refresh_state = get_campaign_step3_source_refresh_state(
        host,
        preferred_source_context=preferred_source_context,
    )

    if bool(refresh_state.get("needs_reextract")):
        prepare_campaign_step3_reextract_from_current_source(
            host,
            preferred_source_context=preferred_source_context,
            background_transition=True,
        )
        try:
            host._set_subtab_state(host.tab_detect, "normal")
            host._set_subtab_state(host.tab_dataset, "disabled")
            host._set_subtab_state(host.tab_extract, "normal")
        except Exception:
            pass

        validation = host._refresh_source_binding_status(allow_autofind=True)
        if not bool(validation.get("ok")):
            msg = str(
                validation.get("message")
                or refresh_state.get("message")
                or "Źródło tablic nie jest jeszcze gotowe do automatycznego wycinania."
            ).strip()
            return Step3EntryFlowViewModel(
                mode="source_invalid",
                message=msg,
                should_show_splash=True,
                splash_title="Nie mogę przygotować tablic dla Z3",
                splash_body=msg,
                splash_tone="error",
                splash_show_progress=False,
                splash_show_return=True,
            )

        return Step3EntryFlowViewModel(
            mode="auto_extract",
            message="Uruchamiam wyodrębnianie tablic do PZ2.",
            target_substep=2,
            should_show_splash=False,
        )

    if saved_substep >= 2 and host.can_restore_step3_substep(saved_substep):
        return Step3EntryFlowViewModel(
            mode="restore_existing_substep",
            message="Przywracam zapisany etap pracy w Z3.",
            target_substep=int(saved_substep),
            should_hide_splash=True,
        )

    if host.can_restore_step3_substep(2):
        return Step3EntryFlowViewModel(
            mode="open_detect",
            message="Otwieram PZ2 na gotowym zestawie wyciętych tablic.",
            target_substep=2,
            should_hide_splash=True,
        )

    return Step3EntryFlowViewModel()


def get_campaign_step3_source_refresh_state(
    host: "CharacterAnnotationTab",
    preferred_source_context: dict | None = None,
) -> dict:
    result = {
        "needs_reextract": False,
        "reason": "",
        "message": "",
        "source_xml": "",
        "source_run_dir": "",
        "preview_dir": "",
    }

    try:
        if not (getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name()):
            return result
    except Exception:
        return result

    source_context = preferred_source_context if isinstance(preferred_source_context, dict) else {}

    source_xml_raw = str(
        source_context.get("xml_path")
        or (host.xml_path_var.get() if hasattr(host, "xml_path_var") else "")
        or ""
    ).strip()
    source_run_raw = str(
        source_context.get("restore_run_dir")
        or source_context.get("run_dir")
        or (host.annotation_run_dir_var.get() if hasattr(host, "annotation_run_dir_var") else "")
        or ""
    ).strip()
    preview_dir_raw = str(
        host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else ""
    ).strip()

    def _is_ready_preview_dir(raw_value: str) -> bool:
        raw_value = str(raw_value or "").strip()
        if not raw_value:
            return False
        try:
            return bool(
                host._is_usable_step3_preview_dir(
                    raw_value,
                    require_plates=True,
                    check_campaign_inflated=False,
                )
            )
        except TypeError:
            try:
                return bool(host._is_usable_step3_preview_dir(raw_value, require_plates=True))
            except Exception:
                return False
        except Exception:
            return False

    def _same_path(left: str, right: str) -> bool:
        left = str(left or "").strip()
        right = str(right or "").strip()
        if not left or not right:
            return False
        try:
            return Path(left).resolve() == Path(right).resolve()
        except Exception:
            return left == right

    def _adopt_preview_dir(raw_value: str) -> str:
        nonlocal preview_dir_raw
        preview_dir_raw = str(raw_value or "").strip()
        result["preview_dir"] = preview_dir_raw
        if not preview_dir_raw:
            return ""
        try:
            current_var = getattr(host, "preview_dir_var", None)
            if current_var is not None and not _same_path(str(current_var.get() or ""), preview_dir_raw):
                current_var.set(preview_dir_raw)
        except Exception:
            pass
        try:
            CAMPAIGN.set_step3_preview_dir(preview_dir_raw)
        except Exception:
            pass
        try:
            sync_registry = getattr(host, "_sync_campaign_step3_preview_artifact_registry", None)
            if callable(sync_registry):
                sync_registry(preview_dir_raw)
        except Exception:
            pass
        return preview_dir_raw

    def _find_latest_current_preview_dir() -> str:
        finder = getattr(host, "_find_latest_extract_preview_run_dir", None)
        if not callable(finder):
            return ""
        try:
            candidate = str(finder(require_plates=True) or "").strip()
        except Exception:
            candidate = ""
        if candidate and _is_ready_preview_dir(candidate):
            return candidate
        return ""

    if not _is_ready_preview_dir(preview_dir_raw):
        try:
            saved_preview_dir = str(host._get_saved_step3_preview_dir(require_plates=True) or "").strip()
        except Exception:
            saved_preview_dir = ""
        if saved_preview_dir and _is_ready_preview_dir(saved_preview_dir):
            _adopt_preview_dir(saved_preview_dir)
        else:
            latest_preview_dir = _find_latest_current_preview_dir()
            if latest_preview_dir:
                _adopt_preview_dir(latest_preview_dir)

    result["source_xml"] = source_xml_raw
    result["source_run_dir"] = source_run_raw
    result["preview_dir"] = preview_dir_raw

    if not source_xml_raw:
        return result

    try:
        source_xml = Path(source_xml_raw)
    except Exception:
        return result

    if not source_xml.exists() or not source_xml.is_file():
        return result

    if not preview_dir_raw:
        result.update(
            needs_reextract=True,
            reason="missing_preview",
            message="Tablice z Z2 są gotowe, ale w Z3 nie ma jeszcze aktualnego zestawu wyciętych tablic. Najpierw uruchom wycinanie w PZ1.",
        )
        return result

    try:
        preview_dir = Path(preview_dir_raw)
    except Exception:
        result.update(
            needs_reextract=True,
            reason="invalid_preview_dir",
            message="Poprzedni zestaw wyciętych tablic nie jest już dostępny. Najpierw uruchom ponowne wycinanie w PZ1.",
        )
        return result

    meta_path = preview_dir / "metadata.json"
    images_dir = preview_dir / "images"
    if not preview_dir.exists() or not preview_dir.is_dir() or not meta_path.exists() or not images_dir.exists():
        result.update(
            needs_reextract=True,
            reason="incomplete_preview",
            message="Poprzedni zestaw wyciętych tablic jest niepełny. Najpierw uruchom ponowne wycinanie w PZ1.",
        )
        return result

    try:
        if hasattr(host, "_preview_matches_current_extract_source") and not host._preview_matches_current_extract_source(preview_dir):
            latest_preview_dir = _find_latest_current_preview_dir()
            if latest_preview_dir:
                _adopt_preview_dir(latest_preview_dir)
                preview_dir = Path(preview_dir_raw)
                meta_path = preview_dir / "metadata.json"
                images_dir = preview_dir / "images"
            else:
                result.update(
                    needs_reextract=True,
                    reason="preview_source_mismatch",
                    message="Poprzednio wycięte tablice pochodzą z innego źródła Z2. Najpierw uruchamiam ponowne wycinanie.",
                )
                return result
    except Exception:
        pass

    preview_has_current_manifest_contract = False
    try:
        manifest_state = dict(host._get_extract_preview_manifest_state(preview_dir) or {})
        preview_has_current_manifest_contract = bool(
            manifest_state.get("manifest_exists")
            and manifest_state.get("source_matches")
            and int(manifest_state.get("plate_count", 0) or 0) > 0
        )
        if preview_has_current_manifest_contract:
            result["preview_contract_ready"] = True
    except Exception:
        preview_has_current_manifest_contract = False

    try:
        saved_state = CAMPAIGN.get_step3_extract_state() or {}
    except Exception:
        saved_state = {}

    saved_xml_raw = str(saved_state.get("xml_path", "") or "").strip()
    saved_run_raw = str(saved_state.get("annotation_run_dir", "") or "").strip()

    def _safe_path_key(raw_value: str) -> str:
        raw_value = str(raw_value or "").strip()
        if not raw_value:
            return ""
        try:
            return str(Path(raw_value).resolve())
        except Exception:
            return raw_value

    source_xml_key = _safe_path_key(source_xml_raw)
    saved_xml_key = _safe_path_key(saved_xml_raw)
    source_run_key = _safe_path_key(source_run_raw)
    saved_run_key = _safe_path_key(saved_run_raw)

    if (
        saved_xml_key
        and source_xml_key
        and saved_xml_key != source_xml_key
        and not preview_has_current_manifest_contract
    ):
        result.update(
            needs_reextract=True,
            reason="source_xml_changed",
            message="Zmienilo się źródło anotacji tablic z Z2. Najpierw uruchom ponowne wycinanie w PZ1.",
        )
        return result

    if (
        saved_run_key
        and source_run_key
        and saved_run_key != source_run_key
        and not preview_has_current_manifest_contract
    ):
        result.update(
            needs_reextract=True,
            reason="source_run_changed",
            message="Zmienil się run tablic z Z2. Najpierw uruchom ponowne wycinanie w PZ1.",
        )
        return result

    try:
        source_mtime = float(source_xml.stat().st_mtime)
        preview_mtime = float(meta_path.stat().st_mtime)
        if source_mtime > (preview_mtime + 0.001) and not preview_has_current_manifest_contract:
            result.update(
                needs_reextract=True,
                reason="source_xml_newer_than_preview",
                message="Anotacje tablic w Z2 zostały rozszerzone po ostatnim wycinaniu. Najpierw uruchom ponowne wycinanie w PZ1.",
            )
            return result
    except Exception:
        pass

    return result


def prepare_campaign_step3_reextract_from_current_source(
    host: "CharacterAnnotationTab",
    preferred_source_context: dict | None = None,
    background_transition: bool = False,
) -> dict:
    refresh_state = get_campaign_step3_source_refresh_state(host, preferred_source_context=preferred_source_context)
    if not bool(refresh_state.get("needs_reextract")):
        return refresh_state

    try:
        host._campaign_step3_reextract_seed_metadata = host._capture_preview_reextract_seed_metadata()
    except Exception:
        host._campaign_step3_reextract_seed_metadata = {}

    try:
        host._set_preview_dir_runtime_value("", persist_registry=False)
    except Exception:
        pass

    try:
        CAMPAIGN.set_step3_preview_dir("")
    except Exception:
        pass

    host.preview_metadata = {}
    host.preview_plate_ids = []
    host._loaded_meta_path = None
    host._loaded_meta_mtime = None

    try:
        host._reset_preview_cache()
    except Exception:
        pass

    try:
        host.plates_listbox.delete(0, tk.END)
    except Exception:
        pass

    try:
        host.preview_canvas.delete("all")
    except Exception:
        pass

    try:
        host.preview_info_lbl.config(
            text="Źródło z Z2 zmienilo się. Najpierw uruchom ponowne wycinanie tablic w PZ1.",
            foreground="#b9770e",
        )
    except Exception:
        pass

    try:
        host._set_extraction_status(
            "Źródło z Z2 zmienilo się. Uruchom ponowne wycinanie tablic.",
            "warning",
        )
    except Exception:
        pass

    try:
        _set_campaign_step3_hold_pz2_after_reextract(host, True)
        CAMPAIGN.set_step3_needs_rework()
        CAMPAIGN.set_step3_stage1_done(False)
        CAMPAIGN.set_step3_stage2_done(False)
        CAMPAIGN.set_step3_substep(1)
    except Exception:
        pass

    try:
        host._extract_workflow_step = "start"
        host._persist_step3_extract_state()
    except Exception:
        pass

    try:
        host._set_subtab_state(host.tab_extract, "normal")
        host._set_subtab_state(host.tab_dataset, "disabled")
        host._set_button_state("btn_to_detect", False)
        host._set_button_state("btn_to_dataset", False)
        host._set_button_emphasis("btn_to_detect_frame", False)
        host._set_button_emphasis("btn_to_dataset_frame", False)
        if background_transition:
            host._campaign_pz2_sync_loading = True
            host._campaign_detect_splash_force_root_surface = True
            host._set_subtab_state(host.tab_detect, "normal")
        else:
            host._set_subtab_state(host.tab_detect, "disabled")
            host._select_subtab(host.tab_extract)
    except Exception:
        pass

    if not background_transition:
        try:
            schedule_refresh = getattr(host, "_schedule_extract_workflow_refresh", None)
            if callable(schedule_refresh):
                schedule_refresh(delay_ms=40 if not host.is_startup_ui_ready() else 0)
            else:
                host._refresh_extract_workflow_ui()
        except Exception:
            pass

    try:
        host._persist_step3_progress()
    except Exception:
        pass

    try:
        host._update_step3_finish_button_state()
    except Exception:
        pass

    return refresh_state


def unlock_detection_subtab(host: "CharacterAnnotationTab"):
    host._set_button_state("btn_to_detect", True)
    host._set_button_emphasis("btn_to_detect_frame", False)
    try:
        schedule_refresh = getattr(host, "_schedule_extract_workflow_refresh", None)
        if callable(schedule_refresh):
            schedule_refresh(delay_ms=0)
        else:
            host._refresh_extract_workflow_ui()
    except Exception:
        pass

    if getattr(host, "_step3_linear_mode", False):
        CAMPAIGN.set_step3_stage1_done(True)
        host._persist_step3_progress()


def unlock_dataset_subtab(host: "CharacterAnnotationTab"):
    _set_campaign_step3_hold_pz2_after_reextract(host, False)
    if getattr(host, "_step3_linear_mode", False):
        try:
            _mark_t06_pz2_contract(host, reason="pz2_detection_ready", force=True)
        except Exception:
            pass
        try:
            dataset_enabled = bool(host._can_open_step3_dataset_from_current_context())
        except Exception:
            dataset_enabled = False
        host._set_button_state("btn_to_dataset", dataset_enabled)
        host._set_subtab_state(host.tab_dataset, "normal" if dataset_enabled else "disabled")
        host._set_button_emphasis("btn_run_detection_frame", False)
        host._set_button_emphasis("btn_to_dataset_frame", dataset_enabled)
        if dataset_enabled:
            CAMPAIGN.set_step3_stage2_done(True)
        else:
            CAMPAIGN.set_step3_stage2_done(False)
        host._persist_step3_progress()
        return

    host._set_button_state("btn_to_dataset", True)
    host._set_button_emphasis("btn_run_detection_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", True)


def _ensure_campaign_pz2_preview_loaded(
    host: "CharacterAnnotationTab",
    *,
    force_reload: bool = False,
) -> bool:
    """
    Deterministycznie odtwarza liste i canvas PZ2 z preview runu projektu.

    Sam zaplanowany after_idle bywal zbyt kruchy po restarcie: karta PZ2 mogla
    zbudowac sie z pusta/stara sciezka, mimo ze projekt mial poprawne
    metadata.json. Ten helper najpierw podstawia zapis projektu, potem buduje PZ2
    i dopiero wtedy wymusza faktyczne wczytanie danych.
    """
    load_start = time.perf_counter()
    restore_ms = build_ms = preview_ms = 0.0
    try:
        phase_start = time.perf_counter()
        saved_preview_dir = str(host._get_saved_step3_preview_dir(require_plates=True) or "").strip()
        current_var = getattr(host, "preview_dir_var", None)
        current_preview_dir = str((current_var.get() if current_var is not None else "") or "").strip()

        def _ready_preview(raw_value: str) -> bool:
            return bool(
                raw_value
                and host._is_usable_step3_preview_dir(
                    raw_value,
                    require_plates=True,
                    check_campaign_inflated=False,
                )
            )

        def _source_current(raw_value: str) -> bool:
            if not _ready_preview(raw_value):
                return False
            matcher = getattr(host, "_preview_matches_current_extract_source", None)
            if not callable(matcher):
                return True
            try:
                return bool(matcher(raw_value))
            except Exception:
                return False

        def _same_preview(left: str, right: str) -> bool:
            left = str(left or "").strip()
            right = str(right or "").strip()
            if not left or not right:
                return False
            try:
                return Path(left).resolve() == Path(right).resolve()
            except Exception:
                return left == right

        def _use_preview(raw_value: str) -> None:
            raw_value = str(raw_value or "").strip()
            if not raw_value:
                return
            if current_var is not None and not _same_preview(str(current_var.get() or ""), raw_value):
                current_var.set(raw_value)
            try:
                CAMPAIGN.set_step3_preview_dir(raw_value)
            except Exception:
                pass
            try:
                sync_registry = getattr(host, "_sync_campaign_step3_preview_artifact_registry", None)
                if callable(sync_registry):
                    sync_registry(raw_value)
            except Exception:
                pass

        if _source_current(current_preview_dir):
            _use_preview(current_preview_dir)
        elif _source_current(saved_preview_dir):
            _use_preview(saved_preview_dir)
        else:
            latest_preview_dir = ""
            finder = getattr(host, "_find_latest_extract_preview_run_dir", None)
            if callable(finder):
                try:
                    latest_preview_dir = str(finder(require_plates=True) or "").strip()
                except Exception:
                    latest_preview_dir = ""
            if _source_current(latest_preview_dir):
                _use_preview(latest_preview_dir)
            elif not _ready_preview(current_preview_dir):
                host._restore_preview_context_from_project(require_plates=True)
        restore_ms = (time.perf_counter() - phase_start) * 1000.0
    except Exception:
        pass

    try:
        phase_start = time.perf_counter()
        if not host._ensure_detect_tab_built():
            return False
        build_ms = (time.perf_counter() - phase_start) * 1000.0
    except Exception:
        return False

    loaded = False
    try:
        phase_start = time.perf_counter()
        loaded = bool(host._ensure_detection_preview_loaded(force_reload=force_reload))
        preview_ms = (time.perf_counter() - phase_start) * 1000.0
    except Exception as exc:
        logger.debug(f"Nie udalo sie bezposrednio wczytac PZ2 z projektu: {exc}")
        loaded = False

    if loaded:
        try:
            host._sync_step3_access_from_preview_state(getattr(host, "preview_metadata", None))
        except Exception:
            pass
        _apply_campaign_step3_pz2_hold(host)
        try:
            if (
                not _campaign_step3_hold_pz2_after_reextract(host)
                and host._preview_dir_has_completed_detection_output()
                and host._can_open_step3_dataset_from_current_context()
            ):
                CAMPAIGN.set_step3_stage2_done(True)
                host._set_subtab_state(host.tab_dataset, "normal")
                host._set_button_state("btn_to_dataset", True)
        except Exception:
            pass

    if not loaded:
        try:
            host._schedule_detection_preview_autoload()
        except Exception:
            pass

    elapsed_ms = (time.perf_counter() - load_start) * 1000.0
    if elapsed_ms >= 250.0:
        try:
            logger.info(
                "[Z3/PZ2 load] force_reload=%s loaded=%s total=%.1fms restore=%.1fms build=%.1fms preview=%.1fms",
                bool(force_reload),
                bool(loaded),
                elapsed_ms,
                restore_ms,
                build_ms,
                preview_ms,
            )
        except Exception:
            pass

    return loaded


def go_to_substep_2_campaign(host: "CharacterAnnotationTab", *, force: bool = False):
    keep_entry_splash = bool(
        getattr(host, "_campaign_step3_entry_splash_pinned", False)
        and getattr(host, "_campaign_detect_splash_visible", False)
    )
    try:
        if not keep_entry_splash:
            host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass
    if not force:
        btn = getattr(host, "btn_to_detect", None)
        if btn is not None and str(btn.cget("state")) != "normal":
            return
        return_to_t05_work_after_step3_pz1(host)
        return
    else:
        try:
            host._set_button_state("btn_to_detect", True)
            host._set_button_emphasis("btn_to_detect_frame", False)
        except Exception:
            pass

    host._set_subtab_state(host.tab_detect, "normal")
    host._set_subtab_state(host.tab_dataset, "disabled")

    CAMPAIGN.set_step3_substep(2)
    _mark_t06_z3_work_session(host, state="active", substep=2, reason="enter_pz2", force=True)
    try:
        after_id = getattr(host, "_detect_preview_autoload_after_id", None)
        if after_id is not None:
            host.frame.after_cancel(after_id)
            host._detect_preview_autoload_after_id = None
    except Exception:
        pass
    try:
        host._campaign_pz2_sync_loading = True
    except Exception:
        pass
    host._select_subtab(host.tab_detect)
    host._set_subtab_state(host.tab_extract, "disabled")
    try:
        notify = getattr(host.app, "notify_free_mode_assistant_context_changed", None)
        if callable(notify):
            notify()
    except Exception:
        pass
    already_loaded_this_entry = bool(getattr(host, "_campaign_pz2_preview_loaded_this_entry", False))
    loaded_this_call = False
    force_reload_preview = True
    try:
        saved_dir = host._get_saved_step3_preview_dir(require_plates=True)
        current_var = getattr(host, "preview_dir_var", None)
        current_dir = str((current_var.get() if current_var is not None else "") or "").strip()
        saved_path = Path(str(saved_dir)).resolve() if saved_dir else None
        current_path = Path(current_dir).resolve() if current_dir else None
        same_preview_dir = bool(saved_path and current_path and saved_path == current_path)
        has_preview_list = bool(getattr(host, "_listbox_pid_by_index", None))
        has_preview_metadata = bool(getattr(host, "preview_metadata", None))
        if same_preview_dir and has_preview_list and has_preview_metadata:
            force_reload_preview = False
    except Exception:
        force_reload_preview = True
    if not already_loaded_this_entry:
        loaded_this_call = bool(_ensure_campaign_pz2_preview_loaded(host, force_reload=force_reload_preview))
    _apply_campaign_step3_pz2_hold(host)
    try:
        host._sync_step3_access_from_preview_state(getattr(host, "preview_metadata", None))
    except Exception:
        pass
    try:
        selected_tab = str(host.main_nb.select())
    except Exception:
        selected_tab = ""
    try:
        logger.info(
            "[Z3/PZ2 campaign entry] loaded=%s already_loaded=%s built=%s selected_detect=%s preview=%s",
            bool(loaded_this_call),
            bool(already_loaded_this_entry),
            bool(getattr(host, "_detect_tab_built", False)),
            bool(selected_tab == str(host.tab_detect)),
            str(host.preview_dir_var.get() or "").strip(),
        )
    except Exception:
        pass
    try:
        host._campaign_pz2_preview_loaded_this_entry = bool(already_loaded_this_entry or loaded_this_call)
    except Exception:
        pass
    host._persist_step3_progress()
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass
    try:
        host._campaign_pz2_sync_loading = False
    except Exception:
        pass
    if keep_entry_splash and not _campaign_step3_hold_pz2_after_reextract(host):
        try:
            host._hide_campaign_detect_splash()
        except Exception:
            pass
    if _campaign_step3_hold_pz2_after_reextract(host):
        def _stabilize_campaign_pz2_after_reextract():
            try:
                if not (
                    getattr(host, "_step3_linear_mode", False)
                    and CAMPAIGN.get_active_project_name()
                    and _campaign_step3_hold_pz2_after_reextract(host)
                ):
                    return
                host._campaign_pz2_sync_loading = True
                host._set_subtab_state(host.tab_detect, "normal")
                host._set_subtab_state(host.tab_dataset, "disabled")
                host._select_subtab(host.tab_detect)
                host._set_subtab_state(host.tab_extract, "disabled")
                _apply_campaign_step3_pz2_hold(host)
                try:
                    host._hide_campaign_detect_splash()
                except Exception:
                    pass
            except Exception as exc:
                logger.debug(f"Nie udało się ustabilizować PZ2 po reekstrakcji: {exc}")
            finally:
                try:
                    host._campaign_pz2_sync_loading = False
                except Exception:
                    pass

        try:
            host.frame.after(180, _stabilize_campaign_pz2_after_reextract)
        except Exception:
            pass


def go_to_substep_3_campaign(host: "CharacterAnnotationTab", *, force: bool = False):
    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass
    if not force:
        _request_t05_work_modal_on_campaign_graph(host, reason="pz2_direct_pz3_blocked_return_to_t05")
        return_to_wizard_from_step3_pz2(host)
        return
    try:
        if not bool(host._can_open_step3_dataset_from_current_context()):
            host._set_button_state("btn_to_dataset", False)
            host._set_subtab_state(host.tab_dataset, "disabled")
            try:
                host.app.update_status(
                    "PZ3 jest dostępne dopiero po domknięciu PZ2 w bieżącej iteracji.",
                    "warning",
                )
            except Exception:
                pass
            return
    except Exception:
        pass
    btn = getattr(host, "btn_to_dataset", None)
    if btn is not None and str(btn.cget("state")) != "normal":
        if force:
            try:
                host._set_button_state("btn_to_dataset", True)
            except Exception:
                pass
        else:
            return

    try:
        host._restore_preview_context_from_project(require_plates=True)
    except Exception:
        pass

    host._set_subtab_state(host.tab_extract, "disabled")
    host._set_subtab_state(host.tab_detect, "disabled")
    host._set_subtab_state(host.tab_dataset, "normal")

    _mark_t06_pz2_contract(host, reason="enter_pz3")
    CAMPAIGN.set_step3_substep(3)
    _mark_t06_z3_work_session(host, state="active", substep=3, reason="enter_pz3", force=True)
    host._select_subtab(host.tab_dataset)
    host._persist_step3_progress()
    host._set_button_emphasis("btn_run_detection_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass


def back_to_substep_1_campaign(host: "CharacterAnnotationTab"):
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass
    host._set_subtab_state(host.tab_extract, "normal")
    host._set_subtab_state(host.tab_detect, "disabled")
    host._set_subtab_state(host.tab_dataset, "disabled")
    host._set_button_state("btn_to_detect", False)
    host._set_button_state("btn_to_dataset", False)

    host._set_button_emphasis("btn_to_detect_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)

    CAMPAIGN.set_step3_substep(1)
    host._select_subtab(host.tab_extract)
    host._persist_step3_progress()


def back_to_substep_2_campaign(host: "CharacterAnnotationTab"):
    try:
        host._cancel_preview_char_label_interaction()
    except Exception:
        pass
    host._set_subtab_state(host.tab_extract, "disabled")
    host._set_subtab_state(host.tab_detect, "normal")
    host._set_subtab_state(host.tab_dataset, "disabled")
    if _campaign_step3_hold_pz2_after_reextract(host):
        host._set_button_state("btn_to_dataset", False)
    else:
        try:
            host._sync_step3_access_from_preview_state(getattr(host, "preview_metadata", None))
        except Exception:
            pass
        host._set_button_state("btn_to_dataset", host.can_restore_step3_substep(3))

    host._set_button_emphasis("btn_to_dataset_frame", False)

    CAMPAIGN.set_step3_substep(2)
    _mark_t06_z3_work_session(host, state="active", substep=2, reason="back_to_pz2", force=True)
    host._select_subtab(host.tab_detect)
    host._persist_step3_progress()


def persist_step3_progress(host: "CharacterAnnotationTab"):
    if not getattr(host, "_step3_linear_mode", False):
        return

    try:
        if CAMPAIGN.get_active_project_name():
            current_step = int(CAMPAIGN.get_current_step() or 0)
            step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
            iteration_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
            iteration_path = str(CAMPAIGN.get_iteration_path() or "").strip().lower()
            if (
                current_step < 3
                and iteration_target == "char"
                and iteration_path == "char_from_images"
                and step2_status != "approved"
            ):
                logger.debug("[Z3] Pomijam zapis postepu E3: T03 czeka jeszcze na jawne zatwierdzenie.")
                return
            if step2_status != "approved":
                CAMPAIGN.approve_step2()
            if current_step < 3:
                CAMPAIGN.set_current_step(3)
    except Exception as exc:
        logger.debug(f"Nie udało się zsynchronizowac stanu kampanii z E3: {exc}")

    current_substep = 1
    try:
        selected = str(host.main_nb.select())
        if selected == str(host.tab_extract):
            current_substep = 1
        elif selected == str(host.tab_detect):
            current_substep = 2
        elif selected == str(host.tab_dataset):
            current_substep = 3
    except Exception:
        current_substep = 1

    CAMPAIGN.set_step3_substep(current_substep)
    preserve_t05_session = False
    try:
        session = dict((CAMPAIGN.get_iteration_state() or {}).get("t06_work_session") or {})
        session_state = str(session.get("state") or "").strip().lower()
        session_gate = str(session.get("working_gate_id") or "").strip().upper()
        session_area = str(session.get("work_area") or "").strip().lower()
        if session_gate in CHAR_WORK_GATE_SESSION_IDS and session_area == "z3":
            if session_state in {"ready_for_pz2", "waiting_for_pz2"} and current_substep <= 1:
                preserve_t05_session = True
            elif (
                session_state in {"resolved", "closed", "complete", "completed"}
                and current_substep >= 3
                and _read_ready_step3_export_summary(host)
            ):
                preserve_t05_session = True
    except Exception:
        preserve_t05_session = False
    if not preserve_t05_session:
        _mark_t06_z3_work_session(host, state="active", substep=current_substep, reason="persist_progress", force=True)

    stage1_done = False
    stage2_done = False

    try:
        btn = getattr(host, "btn_to_detect", None)
        if btn is not None:
            stage1_done = str(btn.cget("state")) == "normal"
    except Exception:
        pass

    try:
        btn = getattr(host, "btn_to_dataset", None)
        if btn is not None:
            stage2_done = str(btn.cget("state")) == "normal"
    except Exception:
        pass
    if _campaign_step3_hold_pz2_after_reextract(host):
        stage2_done = False

    CAMPAIGN.set_step3_stage1_done(stage1_done)
    CAMPAIGN.set_step3_stage2_done(stage2_done)


def restore_campaign_step3_mode(host: "CharacterAnnotationTab"):
    """
    Przywraca zapisany postęp kroku 3 aktywnego projektu.
    """
    host._step3_linear_mode = True

    try:
        host._restore_step3_extract_state_from_project()
    except Exception as exc:
        logger.debug(f"Nie udało się przywrocic stanu wejscia PZ1: {exc}")

    saved_substep = CAMPAIGN.get_step3_substep()
    stage1_done = CAMPAIGN.is_step3_stage1_done()
    stage2_done = CAMPAIGN.is_step3_stage2_done()
    force_detect_entry = bool(getattr(host, "_campaign_force_detect_entry", False))
    force_dataset_entry = bool(getattr(host, "_campaign_force_pz3_entry", False))
    gate_forced_entry = bool(force_detect_entry or force_dataset_entry)
    hold_pz2_after_reextract = _campaign_step3_hold_pz2_after_reextract(host)
    if hold_pz2_after_reextract:
        force_detect_entry = True
        gate_forced_entry = True
        stage2_done = False

    try:
        current_preview_dir = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        current_preview_dir = ""
    try:
        preview_ready = bool(
            current_preview_dir
            and host._is_usable_step3_preview_dir(current_preview_dir, require_plates=True)
        )
    except Exception:
        preview_ready = bool(current_preview_dir)

    if saved_substep > 1 and not preview_ready:
        try:
            host._restore_preview_context_from_project(require_plates=True)
        except Exception:
            pass

    try:
        can_restore_substep_2 = bool(host.can_restore_step3_substep(2))
    except Exception:
        can_restore_substep_2 = False
    try:
        can_restore_substep_3 = bool(host.can_restore_step3_substep(3))
    except Exception:
        can_restore_substep_3 = False
    if can_restore_substep_2:
        stage1_done = True

    try:
        if not hold_pz2_after_reextract and host._preview_dir_has_completed_detection_output():
            stage2_done = True
    except Exception:
        pass
    try:
        if not hold_pz2_after_reextract:
            stage2_done = stage2_done or bool(host._campaign_step3_pz2_base_ready())
            if not stage2_done:
                stage2_done = bool(host._get_campaign_step3_annotation_readiness().get("ok"))
    except Exception:
        pass
    try:
        if not bool(host._can_open_step3_dataset_from_current_context()):
            stage2_done = False
    except Exception:
        pass

    try:
        if stage1_done:
            CAMPAIGN.set_step3_stage1_done(True)
        if hold_pz2_after_reextract:
            CAMPAIGN.set_step3_stage2_done(False)
        elif stage2_done:
            CAMPAIGN.set_step3_stage2_done(True)
    except Exception:
        pass

    try:
        if not gate_forced_entry and not hold_pz2_after_reextract:
            saved_substep = 1
        else:
            if force_detect_entry and saved_substep >= 3 and can_restore_substep_2:
                saved_substep = 2
            if saved_substep < 2 and can_restore_substep_2:
                saved_substep = 2
            if hold_pz2_after_reextract and saved_substep >= 3:
                saved_substep = 2
            if (
                force_dataset_entry
                and not force_detect_entry
                and not hold_pz2_after_reextract
                and saved_substep < 3
                and can_restore_substep_3
            ):
                saved_substep = 3
    except Exception:
        pass
    if not gate_forced_entry and not hold_pz2_after_reextract:
        can_restore_saved_substep = True
    elif saved_substep == 2:
        can_restore_saved_substep = can_restore_substep_2
    elif saved_substep >= 3:
        can_restore_saved_substep = can_restore_substep_3
    else:
        try:
            can_restore_saved_substep = bool(host.can_restore_step3_substep(saved_substep))
        except Exception:
            can_restore_saved_substep = False
    if not can_restore_saved_substep:
        saved_substep = 1
        stage1_done = False
        stage2_done = False

        try:
            CAMPAIGN.reset_step3_progress()
        except Exception:
            pass

    display_stage1_done = bool(stage1_done)
    display_stage2_done = bool(stage2_done)
    if not gate_forced_entry and not hold_pz2_after_reextract:
        display_stage2_done = False

    host._set_button_state("btn_to_detect", display_stage1_done)
    host._set_button_state("btn_to_dataset", display_stage2_done)

    if saved_substep <= 1:
        host._set_subtab_state(host.tab_extract, "normal")
        host._set_subtab_state(host.tab_detect, "disabled")
        host._set_subtab_state(host.tab_dataset, "disabled")
        host._select_subtab(host.tab_extract)
    elif saved_substep == 2:
        host._set_subtab_state(host.tab_extract, "disabled")
        host._set_subtab_state(host.tab_detect, "normal")
        host._set_subtab_state(host.tab_dataset, "disabled")
        host._select_subtab(host.tab_detect)
        try:
            notify = getattr(host.app, "notify_free_mode_assistant_context_changed", None)
            if callable(notify):
                notify()
        except Exception:
            pass
        loaded = _ensure_campaign_pz2_preview_loaded(host, force_reload=False)
        try:
            host._campaign_pz2_preview_loaded_this_entry = bool(loaded)
        except Exception:
            pass
        try:
            if (
                loaded
                and not _campaign_step3_hold_pz2_after_reextract(host)
                and (
                    host._preview_dir_has_completed_detection_output()
                    or bool(host._campaign_step3_pz2_base_ready())
                    or bool(host._get_campaign_step3_annotation_readiness().get("ok"))
                )
                and host._can_open_step3_dataset_from_current_context()
            ):
                stage2_done = True
                CAMPAIGN.set_step3_stage2_done(True)
                host._set_subtab_state(host.tab_dataset, "normal")
                host._set_button_state("btn_to_dataset", True)
        except Exception:
            pass
        _apply_campaign_step3_pz2_hold(host)
    else:
        try:
            host._restore_preview_context_from_project(require_plates=True)
        except Exception:
            pass
        host._set_subtab_state(host.tab_extract, "disabled")
        host._set_subtab_state(host.tab_detect, "disabled")
        host._set_subtab_state(host.tab_dataset, "normal")
        host._select_subtab(host.tab_dataset)

    if gate_forced_entry or hold_pz2_after_reextract:
        display_stage1_done = bool(stage1_done)
        display_stage2_done = bool(stage2_done)

    host._set_button_emphasis("btn_run_detection_frame", False)
    host._set_button_emphasis("btn_to_dataset_frame", False)

    if saved_substep == 2 and not display_stage2_done:
        host._set_button_emphasis("btn_run_detection_frame", True)
    elif saved_substep == 2 and display_stage2_done:
        host._set_button_emphasis("btn_to_dataset_frame", True)
    elif saved_substep == 3:
        host._update_step3_finish_button_state()

    try:
        host._update_preview_path_lock()
    except Exception:
        pass

    try:
        host._update_step3_source_path_lock()
    except Exception:
        pass

    try:
        host._sync_yolo_model_binding()
    except Exception:
        pass

    try:
        host._update_yolo_visibility()
    except Exception:
        pass

    if gate_forced_entry or hold_pz2_after_reextract:
        try:
            persist_step3_progress(host)
        except Exception:
            pass

    try:
        host._update_step3_finish_button_state()
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


def build_step3_campaign_navigation_view_model(
    host: "CharacterAnnotationTab",
) -> Step3CampaignNavigationViewModel:
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False

    splash_visible = bool(getattr(host, "_campaign_detect_splash_visible", False))
    return Step3CampaignNavigationViewModel(
        in_campaign=in_campaign,
        splash_visible=splash_visible,
        show_detect_back_to_extract=not in_campaign,
        show_detect_return_to_graph=in_campaign and (not splash_visible),
        show_detect_to_dataset=not in_campaign,
        show_dataset_back_to_detect=True,
    )


def build_step3_finish_action_view_model(
    host: "CharacterAnnotationTab",
) -> Step3FinishActionViewModel:
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False

    if not in_campaign:
        return Step3FinishActionViewModel()

    readiness = host._get_campaign_step3_training_readiness()
    has_outputs = host._has_any_step3_export_outputs()
    ready_summary = _read_ready_step3_export_summary(host)
    ready_for_approval = bool((has_outputs and bool(readiness.get("ok"))) or ready_summary)

    try:
        current_status = str(CAMPAIGN.get_step3_status() or "").strip().lower() or "pending"
    except Exception:
        current_status = "pending"

    finish_label = "Wróć do grafu bez zatwierdzenia"
    finish_hint = "Powrót nie zatwierdza bramki. Praca w PZ3 pozostaje dostępna do kontynuacji."
    finish_tone = "muted"
    emphasize = False

    if ready_for_approval:
        finish_label = "Przekaż dataset i wróć do grafu"
        finish_hint = (
            "Dataset znaków jest już utworzony. Ten przycisk przekazuje wynik do "
            f"bramki {CHAR_WORK_GATE_DISPLAY_ID} i wraca do grafu, gdzie zatwierdzisz bramkę."
        )
        finish_tone = "success"
        emphasize = True
    elif current_status == "needs_rework":
        finish_hint = host._get_step3_finish_block_message(readiness)
        finish_tone = "warning"
    elif current_status == "approved":
        finish_label = "Wróć do grafu kampanii"
        finish_hint = f"Bramka {CHAR_WORK_GATE_DISPLAY_ID} jest już zatwierdzona. Graf otworzy się na kolejnym kroku."
        finish_tone = "success"
    elif not has_outputs:
        finish_hint = host._get_step3_finish_block_message(readiness) or finish_hint
        finish_tone = "warning"

    return Step3FinishActionViewModel(
        visible=True,
        label=finish_label,
        command_id="return_to_wizard_step3",
        enabled=True,
        emphasize=emphasize,
        hint=finish_hint,
        hint_tone=finish_tone,
        back_to_wizard_enabled=True,
    )


def _safe_positive_int(value, default: int = 0) -> int:
    try:
        number = int(value or default or 0)
    except Exception:
        number = int(default or 0)
    return number if number > 0 else 0


def _safe_iteration_from_payload(payload: dict | None) -> int:
    data = dict(payload or {})
    for key in ("source_iteration", "created_iteration", "produced_iteration", "fulfilled_iteration", "iteration"):
        iteration = _safe_positive_int(data.get(key))
        if iteration > 0:
            return iteration
    return 0


def _safe_path_token(path_like) -> str:
    value = str(path_like or "").strip()
    if not value:
        return ""
    try:
        return str(Path(value).resolve()).lower()
    except Exception:
        return value.replace("\\", "/").lower()


def _dataset_counts_text_from_context(context: dict | None) -> str:
    data = dict(context or {})
    details = []
    plates = _safe_positive_int(data.get("plates"))
    chars = _safe_positive_int(data.get("chars"))
    if plates > 0:
        details.append(f"{plates} tablic")
    if chars > 0:
        details.append(f"{chars} znaków")
    return f" ({', '.join(details)})" if details else ""


def _is_step3_pz3_session_interrupted(session: dict | None) -> bool:
    data = dict(session or {})
    if str(data.get("work_area") or "").strip().lower() not in {"z3", "pz3", "char"}:
        return False
    substep = str(data.get("substep") or data.get("stage") or "").strip().lower()
    if substep not in {"", "3", "pz3", "dataset", "z3_pz3"}:
        return False
    state = str(data.get("state") or "").strip().lower()
    if state in {"completed", "closed", "resolved", "finished"}:
        return False
    return bool(data.get("active")) or state in {"active", "started", "dirty", "interrupted", "paused"}


def _build_step3_pz3_dataset_iteration_context(host: "CharacterAnnotationTab", summary: dict | None) -> dict:
    data = dict(summary or {})
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        current_iteration = 1

    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
    except Exception:
        iteration_state = {}

    contracts = iteration_state.get("t06_contracts")
    contracts = dict(contracts) if isinstance(contracts, dict) else {}
    current_contract = contracts.get("pz3_char_dataset")
    current_contract = dict(current_contract) if isinstance(current_contract, dict) else {}

    dataset_path = str(
        data.get("gold_dataset_path")
        or data.get("dataset_path")
        or current_contract.get("dataset_path")
        or current_contract.get("gold_dataset_path")
        or ""
    ).strip()
    dataset_name = Path(dataset_path).name if dataset_path else ""
    summary_path = str(
        data.get("_summary_path")
        or data.get("summary_path")
        or current_contract.get("summary_path")
        or ""
    ).strip()

    dataset_exists = False
    if dataset_path:
        try:
            dataset_exists = Path(dataset_path).exists()
        except Exception:
            dataset_exists = False

    created_from_summary = (
        bool(data.get("gold_dataset_created"))
        and bool(data.get("gold_dataset_valid", True))
        and bool(dataset_path)
        and bool(dataset_exists)
    )
    created_from_contract = (
        bool(current_contract.get("fulfilled"))
        and bool(current_contract.get("gold_dataset_valid", True))
        and bool(dataset_path)
        and bool(dataset_exists)
    )
    has_dataset = bool(created_from_summary or created_from_contract)

    origin_iteration = _safe_iteration_from_payload(data)
    dataset_token = _safe_path_token(dataset_path)
    summary_token = _safe_path_token(summary_path)
    contract_dataset_token = _safe_path_token(
        current_contract.get("dataset_path") or current_contract.get("gold_dataset_path")
    )
    contract_summary_token = _safe_path_token(current_contract.get("summary_path"))
    contract_matches = bool(
        current_contract
        and (
            (dataset_token and dataset_token == contract_dataset_token)
            or (summary_token and summary_token == contract_summary_token)
        )
    )
    if contract_matches:
        origin_iteration = _safe_iteration_from_payload(current_contract) or origin_iteration

    if has_dataset and origin_iteration <= 0:
        try:
            project_name = CAMPAIGN.get_active_project_name()
            registry = dict(CAMPAIGN.load_artifact_registry(project_name) or {})
            registry_states = registry.get("iteration_state")
            registry_states = dict(registry_states) if isinstance(registry_states, dict) else {}
        except Exception:
            registry_states = {}
        for raw_iteration, raw_state in registry_states.items():
            state = dict(raw_state) if isinstance(raw_state, dict) else {}
            registry_contracts = state.get("t06_contracts")
            registry_contracts = dict(registry_contracts) if isinstance(registry_contracts, dict) else {}
            candidate = registry_contracts.get("pz3_char_dataset")
            candidate = dict(candidate) if isinstance(candidate, dict) else {}
            if not candidate:
                continue
            candidate_dataset_token = _safe_path_token(
                candidate.get("dataset_path") or candidate.get("gold_dataset_path")
            )
            candidate_summary_token = _safe_path_token(candidate.get("summary_path"))
            if not (
                (dataset_token and dataset_token == candidate_dataset_token)
                or (summary_token and summary_token == candidate_summary_token)
            ):
                continue
            origin_iteration = _safe_iteration_from_payload(candidate) or _safe_positive_int(raw_iteration)
            if origin_iteration > 0:
                break

    if has_dataset and origin_iteration <= 0:
        origin_iteration = current_iteration

    plates = (
        _safe_positive_int(data.get("exportable_plate_count"))
        or _safe_positive_int(current_contract.get("exportable_plate_count"))
        or _safe_positive_int(data.get("perfect_count"))
        or _safe_positive_int(current_contract.get("perfect_count"))
    )
    chars = _safe_positive_int(data.get("exportable_char_count")) or _safe_positive_int(
        current_contract.get("exportable_char_count")
    )

    session = iteration_state.get("t06_work_session")
    session = dict(session) if isinstance(session, dict) else {}
    interrupted_pz3 = _is_step3_pz3_session_interrupted(session)

    return {
        "current_iteration": current_iteration,
        "origin_iteration": origin_iteration,
        "is_current_iteration": bool(has_dataset and origin_iteration == current_iteration),
        "has_dataset": has_dataset,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "summary_path": summary_path,
        "plates": plates,
        "chars": chars,
        "interrupted_pz3": interrupted_pz3,
    }


def _describe_step3_export_summary_for_status(
    summary: dict | None,
    *,
    dataset_context: dict | None = None,
) -> tuple[str, str]:
    data = dict(summary or {})
    context = dict(dataset_context or {})
    if bool(context.get("has_dataset")):
        dataset_name = str(context.get("dataset_name") or "").strip() or "dataset znaków"
        details_text = _dataset_counts_text_from_context(context)
        current_iteration = _safe_positive_int(context.get("current_iteration"), 1)
        origin_iteration = _safe_positive_int(context.get("origin_iteration"), current_iteration)
        if bool(context.get("is_current_iteration")):
            if bool(context.get("interrupted_pz3")):
                return (
                    f"Dataset znaków gotowy w bieżącej IT{current_iteration}: {dataset_name}{details_text}. "
                    "Ostatnia sesja PZ3 jest jednak przerwana.",
                    "warning",
                )
            return (
                f"Dataset znaków gotowy w bieżącej iteracji IT{current_iteration}: {dataset_name}{details_text}.",
                "success",
            )
        return (
            f"Dataset znaków dostępny z IT{origin_iteration}: {dataset_name}{details_text}. "
            f"W bieżącej IT{current_iteration} nie ma jeszcze nowego eksportu PZ3.",
            "warning",
        )

    dataset_path = str(data.get("gold_dataset_path") or "").strip()
    dataset_name = Path(dataset_path).name if dataset_path else ""
    dataset_exists = False
    if dataset_path:
        try:
            dataset_exists = Path(dataset_path).exists()
        except Exception:
            dataset_exists = False

    created = bool(data.get("gold_dataset_created")) and bool(data.get("gold_dataset_valid", True))
    if created and dataset_path and dataset_exists:
        plates = int(data.get("exportable_plate_count", 0) or 0)
        chars = int(data.get("exportable_char_count", 0) or 0)
        details = []
        if plates > 0:
            details.append(f"{plates} tablic")
        if chars > 0:
            details.append(f"{chars} znaków")
        details_text = f" ({', '.join(details)})" if details else ""
        return f"Dataset znaków gotowy: {dataset_name}{details_text}.", "success"

    if bool(data.get("gold_dataset_created")) and not dataset_exists:
        return "Poprzedni dataset zapisany w podsumowaniu nie jest już dostępny na dysku.", "warning"

    if bool(data.get("gold_dataset_created")) and not bool(data.get("gold_dataset_valid", True)):
        message = str(data.get("gold_dataset_validation_message") or "").strip()
        if message:
            return f"Dataset wymaga sprawdzenia: {message}", "warning"
        return "Dataset wymaga sprawdzenia przed powrotem do grafu.", "warning"

    return "Dataset znaków nie został jeszcze utworzony w PZ3.", "warning"


def _describe_step3_run_for_status(summary: dict | None, host: "CharacterAnnotationTab") -> tuple[str, str]:
    data = dict(summary or {})
    summary_dir = str(data.get("_summary_dir") or "").strip()
    if summary_dir:
        run_name = Path(summary_dir).name
        if run_name:
            return f"Run: {run_name}. To z niego powstał materiał dla datasetu znaków.", "success"

    try:
        preview_context = dict(host._get_active_preview_context() or {})
    except Exception:
        preview_context = {}
    preview_dir = preview_context.get("preview_dir")
    if preview_dir:
        run_name = Path(str(preview_dir)).name
        return f"Run: {run_name}. To aktualne źródło materiału dla PZ3.", "success"

    return "Run: brak aktywnego wyniku PZ2/PZ3. Najpierw przygotuj anotacje znaków.", "warning"


def _describe_step3_dataset_for_status(
    summary: dict | None,
    *,
    readiness: dict | None,
    dataset_context: dict | None = None,
) -> tuple[str, str]:
    data = dict(summary or {})
    readiness_data = dict(readiness or {})
    context = dict(dataset_context or {})
    if bool(context.get("has_dataset")):
        dataset_name = str(context.get("dataset_name") or "").strip() or "dataset znaków"
        suffix = _dataset_counts_text_from_context(context)
        current_iteration = _safe_positive_int(context.get("current_iteration"), 1)
        origin_iteration = _safe_positive_int(context.get("origin_iteration"), current_iteration)
        if bool(context.get("is_current_iteration")):
            if bool(context.get("interrupted_pz3")):
                return (
                    f"Dataset: {dataset_name}{suffix}. Źródło: eksport PZ3 z bieżącej IT{current_iteration}; "
                    "ostatnia sesja PZ3 jest przerwana.",
                    "warning",
                )
            return (
                f"Dataset: {dataset_name}{suffix}. Źródło: eksport PZ3 z bieżącej IT{current_iteration}.",
                "success",
            )
        return (
            f"Dataset: {dataset_name}{suffix}. Źródło: eksport PZ3 z IT{origin_iteration}; "
            f"w IT{current_iteration} nie wyeksportowano jeszcze nowego datasetu.",
            "warning",
        )

    dataset_path = str(data.get("gold_dataset_path") or "").strip()
    dataset_name = Path(dataset_path).name if dataset_path else ""
    dataset_exists = False
    if dataset_path:
        try:
            dataset_exists = Path(dataset_path).exists()
        except Exception:
            dataset_exists = False

    if bool(data.get("gold_dataset_created")) and bool(data.get("gold_dataset_valid", True)) and dataset_exists:
        plates = int(data.get("exportable_plate_count", 0) or 0)
        chars = int(data.get("exportable_char_count", 0) or 0)
        details = []
        if plates > 0:
            details.append(f"{plates} tablic")
        if chars > 0:
            details.append(f"{chars} znaków")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"Dataset: {dataset_name}{suffix}.", "success"

    if bool(readiness_data.get("ok")):
        return "Dataset: można go utworzyć z gotowego runu PZ2.", "success"

    return "Dataset: jeszcze nie powstał, bo run nie ma wystarczających anotacji znaków.", "warning"


def _describe_step3_export_for_status(
    summary: dict | None,
    *,
    readiness: dict | None,
    dataset_context: dict | None = None,
) -> tuple[str, str]:
    data = dict(summary or {})
    readiness_data = dict(readiness or {})
    context = dict(dataset_context or {})
    if bool(context.get("has_dataset")):
        current_iteration = _safe_positive_int(context.get("current_iteration"), 1)
        origin_iteration = _safe_positive_int(context.get("origin_iteration"), current_iteration)
        if bool(context.get("is_current_iteration")):
            if bool(context.get("interrupted_pz3")):
                return (
                    f"Eksport PZ3: dataset AZ z bieżącej IT{current_iteration} istnieje, "
                    "ale ostatnia sesja PZ3 została przerwana.",
                    "warning",
                )
            return (
                f"Eksport PZ3: wykonany w bieżącej IT{current_iteration}; dataset AZ może domknąć {CHAR_WORK_GATE_DISPLAY_ID}.",
                "success",
            )
        prefix = f"Eksport PZ3 w IT{current_iteration}: "
        if bool(context.get("interrupted_pz3")):
            prefix += "praca przerwana, brak nowego eksportu. "
        else:
            prefix += "brak nowego eksportu. "
        return (
            f"{prefix}Ostatni dostępny dataset pochodzi z IT{origin_iteration}.",
            "warning",
        )

    dataset_path = str(data.get("gold_dataset_path") or "").strip()
    dataset_exists = False
    if dataset_path:
        try:
            dataset_exists = Path(dataset_path).exists()
        except Exception:
            dataset_exists = False

    if bool(data.get("gold_dataset_created")) and bool(data.get("gold_dataset_valid", True)) and dataset_exists:
        return f"Eksport: źródłowy dataset znaków jest zapisany i gotowy do przekazania do {CHAR_WORK_GATE_DISPLAY_ID}.", "success"

    if bool(readiness_data.get("ok")):
        return "Eksport: uruchom tworzenie źródłowego datasetu znaków.", "warning"

    return "Eksport: zablokowany do czasu poprawy anotacji znaków w PZ2.", "warning"


def _describe_step3_readiness_flow_for_status(
    *,
    readiness: dict | None,
    export_summary_tone: str,
    dataset_context: dict | None = None,
) -> tuple[str, str]:
    context = dict(dataset_context or {})
    if bool(context.get("has_dataset")):
        current_iteration = _safe_positive_int(context.get("current_iteration"), 1)
        origin_iteration = _safe_positive_int(context.get("origin_iteration"), current_iteration)
        if bool(context.get("is_current_iteration")):
            if bool(context.get("interrupted_pz3")):
                return (
                    f"Dataset AZ z bieżącej IT{current_iteration} istnieje, ale ostatnia praca PZ3 jest przerwana.",
                    "warning",
                )
            return (
                f"Bramka {CHAR_WORK_GATE_DISPLAY_ID} gotowa do zatwierdzenia dzięki eksportowi AZ z bieżącej IT{current_iteration}.",
                "success",
            )
        if bool(context.get("interrupted_pz3")):
            return (
                f"Praca PZ3 w IT{current_iteration} jest przerwana. Ostatni eksport AZ pochodzi z IT{origin_iteration}.",
                "warning",
            )
        return (
            f"W IT{current_iteration} nie ma nowego eksportu AZ. Dostępny dataset pochodzi z IT{origin_iteration}.",
            "warning",
        )

    if str(export_summary_tone or "").strip().lower() == "success":
        return (
            f"Bramka {CHAR_WORK_GATE_DISPLAY_ID} gotowa do zatwierdzenia.",
            "success",
        )

    data = dict(readiness or {})
    if bool(data.get("ok")):
        return (
            "Run jest gotowy; brakuje utworzenia i zapisania datasetu znaków.",
            "success",
        )

    return (
        "Łańcuch niekompletny: popraw anotacje znaków w PZ2, aby run mógł zasilić dataset.",
        "warning",
    )


def build_step3_pz3_path_selection_view_model(
    host: "CharacterAnnotationTab",
) -> Step3Pz3PathSelectionViewModel:
    selected_path = str(getattr(host, "_pz3_selected_path", "") or "").strip().lower()
    selected_path = selected_path if selected_path in {"dataset", "cvat"} else ""
    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False
    if not selected_path:
        selected_path = "dataset"
    return Step3Pz3PathSelectionViewModel(
        selected_path=selected_path,
        show_dataset_section=True,
        show_cvat_section=(selected_path == "cvat"),
        show_status_section=True,
        dataset_card_selected=True,
        cvat_card_selected=(selected_path == "cvat"),
        dataset_badge_text="DATASET",
        dataset_title_text="Dataset znaków",
        dataset_desc_text="Główna ścieżka PZ3: materiał z PZ2, zakres tablic perfect i utworzenie źródłowego datasetu znaków.",
        cvat_badge_text="OPCJA",
        cvat_title_text="Korekta w CVAT",
        cvat_desc_text="Opcjonalny obieg: wyślij cropy tablic do CVAT, popraw boxy znaków i wczytaj XML z powrotem.",
    )


def build_step3_pz3_status_panel_view_model(
    host: "CharacterAnnotationTab",
) -> Step3Pz3StatusPanelViewModel:
    path_vm = build_step3_pz3_path_selection_view_model(host)
    finish_action = build_step3_finish_action_view_model(host)
    try:
        export_summary = dict(host._read_step3_export_summary() or {})
    except Exception:
        export_summary = {}
    dataset_context = _build_step3_pz3_dataset_iteration_context(host, export_summary)
    summary_export_text, summary_export_tone = _describe_step3_export_summary_for_status(
        export_summary,
        dataset_context=dataset_context,
    )
    try:
        readiness = dict(host._get_campaign_step3_training_readiness() or {})
    except Exception:
        readiness = {}
    run_text, run_tone = _describe_step3_run_for_status(export_summary, host)
    dataset_text, dataset_tone = _describe_step3_dataset_for_status(
        export_summary,
        readiness=readiness,
        dataset_context=dataset_context,
    )
    chain_export_text, chain_export_tone = _describe_step3_export_for_status(
        export_summary,
        readiness=readiness,
        dataset_context=dataset_context,
    )
    readiness_text, readiness_tone = _describe_step3_readiness_flow_for_status(
        readiness=readiness,
        export_summary_tone=summary_export_tone,
        dataset_context=dataset_context,
    )
    export_text, export_tone = host._get_inline_status_widget_snapshot(
        getattr(host, "export_console", None),
        fallback_text=summary_export_text,
        fallback_tone=summary_export_tone,
    )
    export_text_key = str(export_text or "").strip().lower()
    if (
        summary_export_tone == "success"
        or export_text_key in {"", "-", "—", "oczekuję na akcję...", "oczekuje na akcje..."}
    ):
        export_text = summary_export_text
        export_tone = summary_export_tone
    if str(export_tone or "").strip().lower() not in {"danger", "error"} and "błąd" not in str(export_text or "").lower():
        export_text = chain_export_text
        export_tone = chain_export_tone
    import_text, import_tone = host._get_inline_status_widget_snapshot(
        getattr(host, "import_console", None),
        fallback_text="Korekta CVAT nie była używana w tej ścieżce.",
        fallback_tone="muted",
    )
    if str(import_text or "").strip() in {"", "-", "—"}:
        import_text = "Korekta CVAT nie była używana w tej ścieżce."
        import_tone = "muted"

    return Step3Pz3StatusPanelViewModel(
        title="Status PZ3",
        show_section=bool(path_vm.show_status_section),
        run_row=Step3Pz3StatusRowViewModel(
            label="Run",
            text=run_text,
            tone=run_tone,
        ),
        dataset_row=Step3Pz3StatusRowViewModel(
            label="Dataset",
            text=dataset_text,
            tone=dataset_tone,
        ),
        export_row=Step3Pz3StatusRowViewModel(
            label="Eksport",
            text=export_text,
            tone=export_tone,
        ),
        readiness_row=Step3Pz3StatusRowViewModel(
            label="Gotowość",
            text=readiness_text,
            tone=readiness_tone,
        ),
        import_row=Step3Pz3StatusRowViewModel(
            label="Import",
            text=import_text,
            tone=import_tone,
        ),
        finish_action=finish_action,
    )


def refresh_campaign_step3_navigation_visibility(host: "CharacterAnnotationTab"):
    view_model = build_step3_campaign_navigation_view_model(host)

    back_to_extract = getattr(host, "btn_back_to_extract", None)
    if back_to_extract is not None:
        try:
            if not bool(view_model.show_detect_back_to_extract):
                if str(back_to_extract.winfo_manager()):
                    back_to_extract.grid_remove()
            elif not str(back_to_extract.winfo_manager()):
                back_to_extract.grid(row=0, column=0, sticky="w")
        except Exception:
            pass

    return_to_graph_frame = getattr(host, "btn_return_to_graph_pz2_frame", None)
    if return_to_graph_frame is not None:
        try:
            if not bool(view_model.show_detect_return_to_graph):
                if str(return_to_graph_frame.winfo_manager()):
                    return_to_graph_frame.grid_remove()
            elif not str(return_to_graph_frame.winfo_manager()):
                return_to_graph_frame.grid(row=0, column=0, sticky="sw")
        except Exception:
            pass

    to_dataset_frame = getattr(host, "btn_to_dataset_frame", None)
    if to_dataset_frame is not None:
        try:
            if not bool(view_model.show_detect_to_dataset):
                if str(to_dataset_frame.winfo_manager()):
                    to_dataset_frame.grid_remove()
            elif not str(to_dataset_frame.winfo_manager()):
                to_dataset_frame.grid(row=0, column=2, sticky="e", padx=(10, 0))
        except Exception:
            pass

    back_to_detect = getattr(host, "btn_back_to_detect", None)
    back_to_detect_frame = getattr(host, "pz3_back_to_detect_frame", None)
    if back_to_detect is not None and back_to_detect_frame is not None:
        try:
            if not bool(view_model.show_dataset_back_to_detect):
                if str(back_to_detect_frame.winfo_manager()):
                    back_to_detect_frame.pack_forget()
            elif not str(back_to_detect_frame.winfo_manager()):
                back_to_detect_frame.pack(side=tk.LEFT, fill=tk.Y)
                if not str(back_to_detect.winfo_manager()):
                    back_to_detect.pack(side=tk.BOTTOM)
        except Exception:
            pass


def clear_step3_campaign_context(host, nav_button_width: int = 18):
    self = host
    NAV_BUTTON_WIDTH = nav_button_width
    """
    Czyści projektowy kontekst UI po wyjściu z projektu.
    Oprócz pól wejściowych czyści też preview, metadata, log testów
    i local_session powiązany z projektem.
    """
    self._project_reset_token += 1
    self.is_processing = False
    self._reloading_preview = False

    if hasattr(self, "_campaign_chars_dir"):
        self._campaign_chars_dir = None

    if hasattr(self, "_campaign_datasets_dir"):
        self._campaign_datasets_dir = None

    try:
        self._campaign_graph_entry_context = {}
    except Exception:
        pass
    try:
        self._campaign_force_pz2_entry = False
        self._campaign_force_pz3_entry = False
        self._campaign_force_detect_entry = False
    except Exception:
        pass

    try:
        self._clear_project_bound_session_values(clear_ui=False)
    except Exception:
        pass

    try:
        self.annotation_run_dir_var.set("")
    except Exception:
        pass

    try:
        self.extract_entry_mode_var.set("")
    except Exception:
        pass

    try:
        self.xml_path_var.set("")
    except Exception:
        pass

    try:
        self.images_dir_var.set("")
    except Exception:
        pass

    try:
        self._set_preview_dir_runtime_value("", persist_registry=False)
    except Exception:
        pass

    try:
        self.yolo_model_path_var.set("")
    except Exception:
        pass
    try:
        self.pz3_dataset_source_mode_var.set("perfect")
    except Exception:
        pass
    try:
        self.pz3_existing_dataset_var.set("")
    except Exception:
        pass

    self._pending_z2_source = {}
    self._extract_workflow_step = "entry"
    self._extract_last_source_binding_result = {"ok": False}
    try:
        self._set_source_binding_status("", "warning")
    except Exception:
        pass

    self._reset_preview_cache()
    self.preview_metadata = {}
    self.preview_plate_ids = []
    self._preview_base_plate_ids = []
    self._listbox_pid_by_index = []

    try:
        self.plates_listbox.delete(0, tk.END)
    except Exception:
        pass

    try:
        self.preview_canvas.delete("all")
    except Exception:
        pass

    try:
        self._set_preview_info("Brak wczytanych danych", "muted")
        self._set_preview_counts_info(0, 0, 0)
    except Exception:
        pass

    try:
        self._set_preview_box_info("Źródło końcowych ramek: brak wczytanych danych", "muted")
    except Exception:
        pass

    try:
        self.test_log_text.configure(state=tk.NORMAL)
        self.test_log_text.delete("1.0", tk.END)
        self.test_log_text.configure(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self._set_detection_process_log_visibility(False)
    except Exception:
        pass

    try:
        self.fast_test_running = False
        self.fast_test_stop.set()
    except Exception:
        pass

    try:
        self.test_progress.config(value=0)
        self._set_test_progress_counter()
    except Exception:
        pass

    try:
        self._set_test_status(self._compose_detection_method_status("gotowa do uruchomienia"), "neutral")
    except Exception:
        pass

    try:
        self.ext_log.configure(state=tk.NORMAL)
        self.ext_log.delete("1.0", tk.END)
        self.ext_log.configure(state=tk.NORMAL)
    except Exception:
        pass

    try:
        self.ext_progress.config(value=0)
    except Exception:
        pass

    try:
        self._set_extraction_status("Gotowy", "neutral")
    except Exception:
        pass

    try:
        self._refresh_extract_action_state()
    except Exception:
        pass

    try:
        self.btn_ext_stop.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self.import_cvat_xml_var.set("")
    except Exception:
        pass

    try:
        self._set_console_text(self.export_console, "Oczekuję na akcję...")
    except Exception:
        pass

    try:
        self._set_console_text(self.import_console, "Oczekuję na plik XML...")
    except Exception:
        pass

    try:
        self._set_winner_name("BRAK DANYCH Z TURNIEJU", "neutral")
    except Exception:
        pass

    try:
        self._set_winner_acc("Skuteczność detekcji OCR: 0.0%", "error")
    except Exception:
        pass

    try:
        if hasattr(self, "btn_finish_step3"):
            self.btn_finish_step3.config(
                text="Zakończ etap 3",
                state=tk.DISABLED,
                width=NAV_BUTTON_WIDTH,
            )
    except Exception:
        pass

    try:
        if hasattr(self, "btn_back_to_wizard_step3"):
            self.btn_back_to_wizard_step3.config(state=tk.DISABLED)
    except Exception:
        pass

    try:
        self._set_button_emphasis("btn_finish_step3_frame", False)
    except Exception:
        pass

    try:
        self.reset_subtab_flow()
    except Exception as e:
        logger.debug(f"Nie udało się zresetować liniowego flow kroku 3: {e}")

    try:
        self._refresh_extract_workflow_ui()
    except Exception:
        pass

    try:
        self._refresh_step3_mode_specific_ui()
    except Exception:
        pass
