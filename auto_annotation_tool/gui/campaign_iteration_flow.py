#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign iteration completion and training-step flow extracted from tab_campaign.py."""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from datetime import datetime
from time import perf_counter
import xml.etree.ElementTree as ET
import shutil
import threading

from ..config import CONFIG, logger, PIL_AVAILABLE, Image, ImageTk, ImageDraw, ImageFont
from ..campaign_manager import CAMPAIGN
from ..campaign_ingest_planner import CHAR_ALPHABET, CampaignIngestPlanner
from ..validators import validate_model_file, format_yolo_model_identity
from ..icons import IconManager
from ..project_cache import PROJECT_CACHE
from . import campaign_dashboard_cache
from . import campaign_ui_helpers
from . import campaign_project_browser
from . import campaign_model_status
from . import campaign_step1_assets
from . import campaign_step1_ingest
from . import campaign_stage_ui
from . import campaign_stage_logic
from . import campaign_navigation
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
from .campaign_models import WizardStageStatus


def _campaign_training_stage_label(target: str | None = None) -> str:
    normalized = str(target or CAMPAIGN.get_iteration_target() or "").strip().lower()
    if normalized == "plate":
        return "E4T"
    if normalized == "char":
        return "E4Z"
    return "E4T/E4Z"


def _step_goto_training_dataset(self):
    if not CAMPAIGN.get_active_project_name() or CAMPAIGN.get_current_step() < 4:
        return

    try:
        iteration_target = self._get_iteration_target()
        if iteration_target not in {"plate", "char"}:
            iteration_target = "char"

        tab_train = self.app.tabs.get("training")
        if not tab_train:
            logger.error("Nie znaleziono zakładki TrainingTab w app.tabs.")
            return

        target_label = "tablic" if iteration_target == "plate" else "znaków"
        stage_label = _campaign_training_stage_label(iteration_target)
        try:
            self._show_project_loading_overlay(
                title="Przygotowuję Z4/PZ1",
                body=(
                    f"Odtwarzam kontekst {stage_label} dla toru {target_label} "
                    "i przygotowuję panel datasetu."
                ),
                tone="info",
                progress=None,
            )
            self.frame.update_idletasks()
            self.frame.update()
        except Exception:
            pass
        try:
            self.app.update_status("Przygotowuję Z4/PZ1 i odtwarzam kontekst datasetu.", "info")
        except Exception:
            pass

        result = tab_train.open_campaign_step4_entry(
            iteration_target=iteration_target,
            preferred_subtab="dataset",
        )
    except Exception as e:
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
        logger.error(f"Błąd otwierania PZ1 dla Z4: {e}")
        return

    if not result.get("ok"):
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
        warn_msg = str(result.get("message") or "").strip()
        if warn_msg:
            try:
                self.app.update_status(warn_msg, "warning")
            except Exception:
                pass
            try:
                messagebox.showwarning("Z4/PZ1 jeszcze zablokowane", warn_msg, parent=self.frame)
            except Exception:
                pass
        return

    try:
        self.app.update_status(
            "Przejście do Z4 otworzyło PZ1, aby utworzyć wariant datasetu tej iteracji.",
            "info",
        )
    except Exception:
        pass

    try:
        self._show_project_loading_overlay(
            title="Przygotowuję Z4/PZ1",
            body="Panel tworzenia wariantu datasetu jest gotowy. Przełączam widok na zakładkę treningu.",
            tone="success",
            progress=86.0,
        )
        self.frame.update_idletasks()
    except Exception:
        pass
    self.app.open_controlled_tab("training")
    try:
        self.frame.after(180, self._hide_project_loading_overlay)
    except Exception:
        pass
    return


def _show_step4_completion_next_action_modal(self, *, completed_with_training: bool = True) -> bool:
    mode = self._ask_iteration_advance_mode(completed_with_training=completed_with_training)
    if str(mode or "").strip().lower() not in {"reuse_input", "new_input"}:
        try:
            self.app.update_status(
                f"Pozostajesz w {_campaign_training_stage_label()}. Iteracja nie została przeniesiona do kolejnego cyklu.",
                "info",
            )
        except Exception:
            pass
        try:
            self.request_wizard_stage_focus(step_num=4)
            self._refresh_active_project_wizard_only()
        except Exception:
            pass
        return False
    self._start_iteration_advance(mode)
    return True


def _start_iteration_advance(self, mode: str | None) -> None:
    mode = str(mode or "").strip().lower()
    if mode not in {"reuse_input", "new_input"}:
        return

    existing_worker = getattr(self, "_iteration_advance_thread", None)
    if existing_worker is not None:
        try:
            if existing_worker.is_alive():
                try:
                    self.app.update_status(
                        "Trwa już przygotowywanie nowej iteracji. Zaczekaj na zakończenie bieżącej operacji.",
                        "info",
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Bezpiecznik na stary, już zakończony worker, który mógł zostać
        # w stanie UI po wcześniejszym przejściu dashboardu.
        self._iteration_advance_thread = None
        self._iteration_advance_result = None
        self._iteration_advance_mode = ""
        try:
            pending_poll = getattr(self, "_iteration_advance_poll_after_id", None)
            if pending_poll:
                self.frame.after_cancel(pending_poll)
        except Exception:
            pass
        self._iteration_advance_poll_after_id = None
        self._set_iteration_advance_busy(False)

    try:
        if mode == "reuse_input":
            self.app.update_status(
                "Przygotowuję nową iterację z tej samej puli zdjęć. To może chwilę potrwać przy dużym katalogu zdjęć.",
                "info",
            )
        else:
            self.app.update_status(
                "Rozpoczynam nową iterację projektu.",
                "info",
            )
    except Exception:
        pass
    try:
        self.frame.update_idletasks()
    except Exception:
        pass

    self._iteration_advance_mode = mode
    self._iteration_advance_result = None
    self._set_iteration_advance_busy(True)
    try:
        self._show_project_loading_overlay(
            title="Przygotowuję kolejną iterację",
            body="Zapisuję stan przejścia i przygotowuję obrazy dla E1.",
            eyebrow="BRAMKA T06", tone="info", progress=None,
        )
    except Exception:
        pass

    worker = threading.Thread(
        target=self._run_iteration_advance_worker,
        args=(self._iteration_advance_mode,),
        daemon=True,
    )
    self._iteration_advance_thread = worker
    worker.start()
    self._schedule_iteration_advance_poll()


def _finish_step4_iteration(self):
    if not CAMPAIGN.get_active_project_name():
        return

    training_tab = self.app.tabs.get("training") if getattr(self.app, "tabs", None) else None
    try:
        if training_tab is not None and hasattr(training_tab, "_step4_has_active_operation"):
            if bool(training_tab._step4_has_active_operation()):
                self.app.themed_info(
                    "Poczekaj na zakończenie operacji Z4",
                    "W Z4 nadal trwa aktywna operacja. Zaczekaj na jej zakończenie, a potem domknij iterację.",
                    parent=self.frame,
                    tone="warning",
                )
                return
    except Exception:
        pass

    try:
        if training_tab is not None and hasattr(training_tab, "_finish_campaign_step4"):
            handled_by_training = bool(training_tab._finish_campaign_step4())
            try:
                step_after_training = int(CAMPAIGN.get_current_step() or 0)
            except Exception:
                step_after_training = 0
            if handled_by_training or step_after_training >= 5:
                return
    except Exception as e:
        logger.debug(f"Nie udało się domknąć {_campaign_training_stage_label()} przez Z4, przechodzę na fallback: {e}")

    finish_state = {}
    try:
        current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        getter = getattr(training_tab, "get_campaign_step4_finish_state", None) if training_tab is not None else None
        finish_state = (
            dict(getter(iteration_target=current_target) or {})
            if callable(getter)
            else dict(CAMPAIGN.get_step4_finish_state() or {})
        )
    except Exception:
        try:
            current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
            finish_state = dict(CAMPAIGN.get_step4_finish_state() or {})
        except Exception:
            current_target = ""
            finish_state = {}
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    try:
        finish_iteration = int(finish_state.get("iteration", 0) or 0)
    except Exception:
        finish_iteration = 0
    finish_target = str(finish_state.get("target", "") or "").strip().lower()
    run_id = str(finish_state.get("run_id", "") or "").strip()
    selected_model_path = str(finish_state.get("model_path", "") or "").strip()
    try:
        selected_model_ready = bool(
            selected_model_path and Path(selected_model_path).exists() and Path(selected_model_path).is_file()
        )
    except Exception:
        selected_model_ready = bool(selected_model_path)
    training_record = {}
    try:
        iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=current_iteration) or {})
        training_record = dict(iteration_state.get("step4_training") or {})
    except Exception:
        training_record = {}
    if not training_record:
        try:
            bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(iteration_num=current_iteration) or {})
            training_record = dict(bundle.get("step4_training") or {})
        except Exception:
            training_record = {}
    record_run_id = str(training_record.get("run_id", "") or "").strip()
    record_target = str(training_record.get("target", "") or "").strip().lower()
    record_status = str(training_record.get("status", "") or "").strip().lower()
    best_weights = str(training_record.get("best_weights", "") or "").strip()
    try:
        best_exists = bool(best_weights and Path(best_weights).exists())
    except Exception:
        best_exists = bool(best_weights)
    record_ready = bool(
        record_run_id
        and record_run_id == run_id
        and (not record_target or not current_target or record_target == current_target)
        and record_status == "completed"
        and best_exists
    )
    finish_ready = bool(
        finish_state.get("ready")
        and run_id
        and finish_iteration == current_iteration
        and (not finish_target or not current_target or finish_target == current_target)
        and (record_ready or selected_model_ready)
    )
    if not finish_ready:
        try:
            self.app.themed_info(
                "T07 wymaga decyzji",
                (
                    "Bramka T07 nie ma bieżącego wyniku treningu dla tej iteracji.\n\n"
                    "Uruchom trening w Z4 albo wybierz w pracy bramki świadome pominięcie treningu."
                ),
                parent=self.frame,
                tone="warning",
            )
        except Exception:
            pass
        try:
            self.app.update_status(
                "T07 wymaga bieżącego wyniku treningu albo jawnej decyzji pominięcia treningu.",
                "warning",
            )
        except Exception:
            pass
        try:
            self.request_wizard_stage_focus(step_num=4)
            self._refresh_active_project_wizard_only()
        except Exception:
            pass
        return

    mode = self._ask_iteration_advance_mode(completed_with_training=True)
    if str(mode or "").strip().lower() not in {"reuse_input", "new_input"}:
        try:
            self.app.update_status(
                f"Pozostajesz w {_campaign_training_stage_label(current_target)}. Iteracja nie została domknięta, więc badge zatwierdzający pozostaje dostępny.",
                "info",
            )
        except Exception:
            pass
        try:
            self.request_wizard_stage_focus(step_num=4)
            self._refresh_active_project_wizard_only()
        except Exception:
            pass
        return

    try:
        CAMPAIGN.set_current_step(5)
        CAMPAIGN.set_step4_finish_state(False)
        try:
            CAMPAIGN.set_step4_without_training_decision(False)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Nie udało się domknąć iteracji w {_campaign_training_stage_label(current_target)}: {e}")
        return

    try:
        self.app.update_status("Iteracja została domknięta. Przygotowuję przejście do E1.", "info")
    except Exception:
        pass
    self._start_iteration_advance(mode)


def _finish_step4_without_training(self):
    if not CAMPAIGN.get_active_project_name():
        return

    training_tab = self.app.tabs.get("training") if getattr(self.app, "tabs", None) else None
    try:
        if training_tab is not None and hasattr(training_tab, "_step4_has_active_operation"):
            if bool(training_tab._step4_has_active_operation()):
                self.app.themed_info(
                    "Poczekaj na zakończenie operacji Z4",
                    "W Z4 nadal trwa aktywna operacja. Zaczekaj na jej zakończenie, a potem zdecyduj, czy chcesz pominąć trening.",
                    parent=self.frame,
                    tone="warning",
                )
                return
    except Exception:
        pass

    try:
        decision = dict(CAMPAIGN.get_step4_without_training_decision() or {})
    except Exception:
        decision = {}
    try:
        decision_iteration = int(decision.get("iteration", 0) or 0)
    except Exception:
        decision_iteration = 0
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0
    decision_target = str(decision.get("target", "") or "").strip().lower()
    current_target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
    graph_decision_ready = bool(
        decision.get("ready")
        and decision_iteration == current_iteration
        and (not decision_target or decision_target == current_target)
    )
    if graph_decision_ready:
        mode = self._ask_iteration_advance_mode(completed_with_training=False)
        if str(mode or "").strip().lower() not in {"reuse_input", "new_input"}:
            try:
                self.app.update_status(
                    f"Pozostajesz w {_campaign_training_stage_label(current_target)}. Iteracja bez treningu nie została domknięta.",
                    "info",
                )
            except Exception:
                pass
            try:
                self.request_wizard_stage_focus(step_num=4)
                self._refresh_active_project_wizard_only()
            except Exception:
                pass
            return

        try:
            CAMPAIGN.set_current_step(5)
            CAMPAIGN.set_step4_finish_state(False)
            try:
                CAMPAIGN.set_step4_without_training_decision(False)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Nie udało się domknąć iteracji bez treningu: {e}")
            return

        try:
            self.app.update_status("Iteracja została domknięta bez treningu. Przygotowuję przejście do E1.", "info")
        except Exception:
            pass
        self._start_iteration_advance(mode)
        return

    should_finish = self.app.themed_confirm(
        "Zamknąć iterację bez treningu?",
        "Ta iteracja zostanie formalnie domknięta bez uruchamiania treningu.\n\n"
        "Wyniki przygotowania datasetu i adnotacji pozostaną w projekcie, ale dopiero kolejna iteracja "
        "będzie mogła wystartować z poziomu E1.\n\n"
        "Czy chcesz zamknąć iterację bez treningu?",
        parent=self.frame,
        confirm_label="Zamknij iterację",
        tone="warning",
    )
    if not should_finish:
        return

    mode = self._ask_iteration_advance_mode(completed_with_training=False)
    if str(mode or "").strip().lower() not in {"reuse_input", "new_input"}:
        try:
            self.app.update_status(
                f"Pozostajesz w {_campaign_training_stage_label(current_target)}. Iteracja bez treningu nie została domknięta.",
                "info",
            )
        except Exception:
            pass
        try:
            self.request_wizard_stage_focus(step_num=4)
            self._refresh_active_project_wizard_only()
        except Exception:
            pass
        return

    try:
        CAMPAIGN.set_current_step(5)
        CAMPAIGN.set_step4_finish_state(False)
        try:
            CAMPAIGN.set_step4_without_training_decision(False)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Nie udało się domknąć iteracji bez treningu: {e}")
        return

    try:
        self.app.update_status("Iteracja została domknięta bez treningu. Przygotowuję przejście do E1.", "info")
    except Exception:
        pass
    self._start_iteration_advance(mode)


def _set_iteration_advance_busy(self, busy: bool) -> None:
    state_cache = getattr(self, "_iteration_advance_ui_cache", None)
    if busy and not isinstance(state_cache, dict):
        state_cache = {}
        try:
            state_cache["btn_complete_project"] = {
                "text": self.btn_complete_project.cget("text"),
                "state": self.btn_complete_project.cget("state"),
            }
        except Exception:
            pass
        for attr_name in ("btn_open_proj", "btn_del_proj", "btn_exit_project"):
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            try:
                state_cache[attr_name] = {"state": widget.cget("state")}
            except Exception:
                pass
        self._iteration_advance_ui_cache = state_cache

    try:
        if busy:
            self.btn_complete_project.config(
                text="Przygotowywanie iteracji...",
                state="disabled",
            )
        elif isinstance(state_cache, dict) and "btn_complete_project" in state_cache:
            self.btn_complete_project.config(**dict(state_cache["btn_complete_project"]))
    except Exception:
        pass

    for attr_name in ("btn_open_proj", "btn_del_proj", "btn_exit_project"):
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        try:
            if busy:
                widget.config(state="disabled")
            elif isinstance(state_cache, dict) and attr_name in state_cache:
                widget.config(**dict(state_cache[attr_name]))
        except Exception:
            pass

    try:
        self.frame.configure(cursor=("watch" if busy else ""))
    except Exception:
        pass
    try:
        self.app.root.configure(cursor=("watch" if busy else ""))
    except Exception:
        pass
    if not busy:
        self._iteration_advance_ui_cache = None


def _run_iteration_advance_worker(self, mode: str) -> None:
    started = perf_counter()
    try:
        result = CAMPAIGN.advance_to_next_iteration(start_mode=mode)
    except Exception as exc:
        logger.exception("Błąd podczas tworzenia nowej iteracji")
        result = {
            "ok": False,
            "reason": "exception",
            "error": str(exc),
        }
    self._iteration_advance_result = dict(result or {})
    self._iteration_advance_result["prepare_ms"] = (perf_counter() - started) * 1000.0


def _schedule_iteration_advance_poll(self) -> None:
    pending = getattr(self, "_iteration_advance_poll_after_id", None)
    if pending:
        try:
            self.frame.after_cancel(pending)
        except Exception:
            pass
    self._iteration_advance_poll_after_id = self.frame.after(80, self._poll_iteration_advance_worker)


def _poll_iteration_advance_worker(self) -> None:
    self._iteration_advance_poll_after_id = None
    worker = getattr(self, "_iteration_advance_thread", None)
    if worker is not None and worker.is_alive():
        self._schedule_iteration_advance_poll()
        return

    mode = str(getattr(self, "_iteration_advance_mode", "") or "").strip().lower()
    result = dict(getattr(self, "_iteration_advance_result", {}) or {})
    self._iteration_advance_thread = None
    self._iteration_advance_result = None
    self._iteration_advance_mode = ""
    self._set_iteration_advance_busy(False)
    self._finish_iteration_advance(mode, result)


def _finish_iteration_advance(self, mode: str, result: dict) -> None:
    started = perf_counter()
    if not result.get("ok"):
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
        reason = str(result.get("reason") or "").strip().lower()
        if reason == "step4_not_finished":
            stage_label = _campaign_training_stage_label()
            self.app.themed_info(
                f"Najpierw domknij {stage_label}",
                (
                    f"Nie możesz rozpocząć kolejnej iteracji, dopóki {stage_label} tej iteracji "
                    "nie zostanie zakończony.\n\n"
                    "Najpierw wróć do Z4, uruchom trening albo domknij etap tylko wtedy, "
                    "gdy kampania oznaczy trening jako gotowy."
                ),
                parent=self.frame,
                tone="warning",
            )
        elif reason == "project_not_active":
            self.app.themed_info(
                "Projekt nie jest aktywny",
                "Wznów projekt, zanim rozpoczniesz kolejną iterację.",
                parent=self.frame,
                tone="warning",
            )
        elif reason == "missing_remaining_images":
            self.app.themed_info(
                "Brak kolejnego zestawu zdjęć",
                (
                    "Nie ma już kolejnych zdjęć do pobrania z tej samej puli projektu.\n"
                    "System pomija tu obrazy już zatwierdzone w projekcie.\n\n"
                    "Aby kontynuować, rozpocznij kolejną iterację od nowego zestawu zdjęć "
                    "albo zakończ projekt."
                ),
                parent=self.frame,
                tone="warning",
            )
        elif reason in {"copy_failed", "manifest_save_failed"}:
            self.app.themed_info(
                "Nie udało się przygotować iteracji",
                (
                    "Nie udało się zapisać manifestu kolejnego katalogu zdjęć.\n\n"
                    "Spróbuj ponownie albo rozpocznij iterację od innego katalogu zdjęć."
                ),
                parent=self.frame,
                tone="warning",
            )
        else:
            self.app.themed_info(
                "Nie udało się rozpocząć iteracji",
                "Nowa iteracja nie została utworzona. Sprawdź stan projektu i spróbuj ponownie.",
                parent=self.frame,
                tone="warning",
            )
        return

    try:
        annotation_tab = getattr(self.app, "tabs", {}).get("annotation")
        try:
            next_input_dir = CAMPAIGN.get_iteration_image_source_dir()
        except Exception:
            next_input_dir = CAMPAIGN.get_iteration_raw_dir()
        if annotation_tab is not None and hasattr(annotation_tab, "prepare_campaign_iteration_transition"):
            annotation_tab.prepare_campaign_iteration_transition(
                input_dir=(Path(next_input_dir) if next_input_dir else None),
                refresh_ui=False,
            )
    except Exception as e:
        logger.debug(f"Nie udało się przygotować Z2 do nowej iteracji: {e}")

    # The existing graph host can render the new iteration once. Rebuilding it
    # also scheduled several full scans of the same pool after idle and resize.
    previous_lightweight = bool(getattr(self, "_project_open_lightweight_refresh", False))
    try:
        self._project_open_lightweight_refresh = True
        self._clear_dashboard_perf_cache()
        self._refresh_dashboard()
    finally:
        self._project_open_lightweight_refresh = previous_lightweight

    iter_num = int(result.get("next_iteration", CAMPAIGN.get_current_iteration_num()) or CAMPAIGN.get_current_iteration_num())
    effective_mode = str(result.get("effective_mode", "new_input") or "new_input").strip().lower()

    try:
        if mode == "reuse_input" and effective_mode != "reuse_input":
            self.app.update_status(
                (
                    f"Rozpoczęto iterację {iter_num:03d}, ale nie udało się przenieść "
                    "poprzedniego zestawu zdjęć. Iteracja startuje od E1."
                ),
                "warning"
            )
        elif effective_mode == "reuse_input":
            manifest_images = int(result.get("manifest_images", result.get("copied_images", 0)) or 0)
            source_kind = str(result.get("source_kind") or "").strip().lower()
            if source_kind == "stage":
                self.app.update_status(
                    (
                        f"Rozpoczęto iterację {iter_num:03d}. E1 korzysta z manifestu {manifest_images} zdjęć "
                        "ze źródła zatwierdzonych tablic. Wybierz tor iteracji i zatwierdź etap. "
                        "Aktywne modele projektu pozostały zachowane."
                    ),
                    "info"
                )
            else:
                self.app.update_status(
                    (
                        f"Rozpoczęto iterację {iter_num:03d} na manifeście kolejnych zdjęć z tej samej puli "
                        f"({manifest_images} obrazów). Zdjęcia nie są kopiowane między iteracjami. "
                        "Wybierz tor iteracji i zatwierdź etap. "
                        "Aktywne modele projektu pozostały zachowane."
                    ),
                    "info"
                )
        else:
            restored_dir = str(result.get("restored_master_pool_dir") or result.get("previous_image_source") or "").strip()
            if restored_dir:
                restored_name = Path(restored_dir).name or restored_dir
                self.app.update_status(
                    (
                        f"Rozpoczęto iterację {iter_num:03d}. W E1 odtworzono poprzedni katalog zdjęć "
                        f"({restored_name}); możesz go zatwierdzić, zmienić albo wyczyścić. "
                        "Aktywne modele projektu pozostały zachowane."
                    ),
                    "info"
                )
            else:
                self.app.update_status(
                    f"Rozpoczęto iterację {iter_num:03d}. Wskaż nowy zestaw zdjęć w E1. Aktywne modele projektu pozostały zachowane.",
                    "info"
                )
    except Exception:
        pass

    try:
        self.step1_panel_expanded = True
        self.request_wizard_stage_focus(step_num=1)
    except Exception:
        pass

    try:
        self.app.open_controlled_tab("campaign")
    except Exception:
        pass

    if bool(result.get("needs_new_image_source")):
        message = (
            "Nie udało się odtworzyć katalogu zdjęć z poprzedniej iteracji.\n\n"
            "W E1 wskaż katalog zdjęć wejściowych, wybierz tor iteracji i dopiero wtedy "
            "zatwierdź etap. Jeśli poprzedni katalog nadal istnieje, możesz wskazać go ponownie."
        )
        try:
            self.app.themed_info(
                "Wybierz nowy katalog zdjęć",
                message,
                parent=self.frame,
                tone="warning",
            )
        except Exception:
            try:
                messagebox.showwarning("Wybierz nowy katalog zdjęć", message, parent=self.frame)
            except Exception:
                pass
    logger.info(
        "[T06 PERF] iteration=%s mode=%s prepare=%.0fms finish_ui=%.0fms",
        iter_num, mode, float(result.get("prepare_ms", 0.0)), (perf_counter() - started) * 1000.0,
    )


def _toggle_project_completion(self):
    active_project = CAMPAIGN.get_active_project_name()
    if not active_project:
        return

    project_status = CAMPAIGN.get_project_status()
    if project_status in {"completed", "paused"}:
        status_label = "odłożony" if project_status == "paused" else "zakończony"
        if self.app.themed_confirm(
            "Wznowienie projektu",
            f"Czy wznowić {status_label} projekt '{active_project}'?\n\n"
            "Po wznowieniu znowu będzie można rozpocząć nową iterację.",
            parent=self.frame,
            confirm_label="Wznów",
            tone="info"
        ):
            CAMPAIGN.reopen_project()
            self._refresh_dashboard()
            self.app.update_campaign_tab_access()
            try:
                self.app.update_status(
                    f"Projekt '{active_project}' został wznowiony. Możesz wrócić do workflow albo rozpocząć kolejną iterację.",
                    "info"
                )
            except Exception:
                pass
        return

    if CAMPAIGN.get_current_step() < 5:
        return

    if self.app.themed_confirm(
        "Opuszczenie projektu",
        f"Czy opuścić aktywny projekt '{active_project}'?\n\n"
        "Projekt pozostanie zapisany i będzie można wrócić do niego później.",
        parent=self.frame,
        confirm_label="Opuść projekt",
        tone="info"
    ):
        CAMPAIGN.pause_project()
        self._refresh_dashboard()
        self.app.update_campaign_tab_access()
        try:
            self.app.update_status(
                f"Projekt '{active_project}' został odłożony. Możesz wrócić do niego później.",
                "info"
            )
        except Exception:
            pass
