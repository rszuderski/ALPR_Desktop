#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign cross-tab navigation helpers extracted from tab_campaign.py."""

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
from ..campaign_iteration_paths import normalize_iteration_path
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
from .help_manager import HELP
from .run_display import build_run_display_ref
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
from .campaign_models import WizardStageStatus
from .campaign_t02_source import resolve_t02_review_source


def _schedule_z2_right_panel_refresh(tab_ann, *delays_ms: int, lightweight: bool = False) -> None:
    for raw_delay in delays_ms or (350,):
        try:
            delay = max(1, int(raw_delay or 1))
        except Exception:
            delay = 350

        def _refresh(tab=tab_ann) -> None:
            if not lightweight:
                try:
                    tab._refresh_step2_action_states(lightweight=False)
                except Exception:
                    pass
            try:
                tab._sync_right_panel_scrollregion()
            except Exception:
                pass

        try:
            tab_ann.frame.after(delay, _refresh)
        except Exception:
            pass


def _step_open_z2_from_step2_review(self, preferred_source_context=None, *, on_complete=None):
    return self._step_return_to_annotation_review(
        mark_step3_rework=False, preferred_source_context=preferred_source_context, on_complete=on_complete,
    )


def _step_open_z2_repair_from_later_stage(self, preferred_source_context=None, *, on_complete=None):
    return self._step_return_to_annotation_review(
        mark_step3_rework=True, preferred_source_context=preferred_source_context, on_complete=on_complete,
    )


def _step_return_to_annotation_review(
    self, mark_step3_rework=True, preferred_source_context=None, *, on_complete=None,
):
    def _opened(result):
        if result.get("ok") and self._get_iteration_target() == "char" and mark_step3_rework:
            CAMPAIGN.set_current_step(3)
            CAMPAIGN.set_step3_needs_rework()
        if callable(on_complete):
            on_complete(result)

    return self._step_goto_auto_annotation(
        force_annotation_tab=True, open_existing_run=True,
        preferred_source_context=preferred_source_context, on_complete=_opened,
    )


def _step_goto_auto_annotation(
    self,
    force_annotation_tab: bool = False,
    entry_strategy: str | None = None,
    open_existing_run: bool = True,
    preferred_source_context=None,
    on_complete=None,
    _retry_count: int = 0,
):
    def _reject(message, reason="entry_blocked"):
        self.app.update_status(message, "warning")
        return {"ok": False, "message": message, "reason": reason}

    entry_identity = (CAMPAIGN.get_active_project_name(), CAMPAIGN.get_current_iteration_num())

    def _retry_open():
        if entry_identity != (CAMPAIGN.get_active_project_name(), CAMPAIGN.get_current_iteration_num()):
            result = _reject("Anulowano wejście do Z2: zmienił się projekt lub iteracja.")
        else:
            result = _step_goto_auto_annotation(
                self, force_annotation_tab=force_annotation_tab, entry_strategy=entry_strategy,
                open_existing_run=open_existing_run, preferred_source_context=preferred_source_context,
                on_complete=on_complete, _retry_count=_retry_count + 1,
            )
        if callable(on_complete) and not result.get("pending"):
            on_complete(result)

    source_context = dict(preferred_source_context or {})
    t02_at_review = bool(
        str(source_context.get("z2_work_mode") or "").strip().lower() == "t02_at_review"
        or (
            str(source_context.get("graph_edge_key") or "").strip() == "e1_to_e3"
            and str(source_context.get("graph_gate_id") or "").strip().upper() == "T02"
        )
    )
    if not CAMPAIGN.get_active_project_name() or (CAMPAIGN.get_current_step() < 2 and not t02_at_review):
        return _reject("Najpierw wybierz projekt i zatwierdź bramkę wejścia do E2.")
    if t02_at_review:
        t02_source = resolve_t02_review_source(CAMPAIGN, source_context)
        if not t02_source.get("ok"):
            return _reject(t02_source["message"], t02_source["reason"])
        source_context.update(t02_source.get("context") or {})
        try:
            CAMPAIGN.set_iteration_path("char_from_ready_plates")
            if hasattr(CAMPAIGN, "set_graph_selected_edge_key"):
                CAMPAIGN.set_graph_selected_edge_key("e1_to_e3")
        except Exception:
            pass
        try:
            plate_source = dict(CAMPAIGN.get_project_start_plate_source() or {})
        except Exception:
            plate_source = {}
        source_run = str(plate_source.get("source_run_path") or "").strip()
        source_xml = str(plate_source.get("source_xml_path") or "").strip()
        source_input = str(plate_source.get("source_input_path") or "").strip()
        if source_run:
            source_context.setdefault("restore_run_dir", source_run)
        elif source_xml:
            try:
                source_context.setdefault("restore_run_dir", str(Path(source_xml).parent))
            except Exception:
                pass
        if source_input:
            source_context.setdefault("input_dir", source_input)
        source_context.setdefault("input_source", "t02_at_review")
        source_context.setdefault("graph_edge_key", "e1_to_e3")
        source_context.setdefault("graph_gate_id", "T02")
        source_context.setdefault("graph_visible_gate_id", "T02")
        source_context.setdefault("graph_display_gate_id", "T02")
        source_context.setdefault("graph_transition_source", "E1")
        source_context.setdefault("graph_transition_target", "E3")
        source_context.setdefault("graph_path_key", "char_from_ready_plates")
    if str(CAMPAIGN.get_step1_status() or "").strip().lower() != "approved" and not t02_at_review:
        try:
            self.step1_panel_expanded = True
            self.request_wizard_stage_focus(step_num=1)
            self._refresh_active_project_wizard_only()
        except Exception:
            pass
        try:
            self.app.update_status(
                "Najpierw zatwierdź E1: wskaż katalog zdjęć i wybierz tor iteracji. Dopiero wtedy STEP2-P1 będzie aktywne.",
                "warning",
            )
        except Exception:
            pass
        return _reject("Najpierw zatwierdź T01: wskaż obrazy i wybierz tor iteracji.")

    iteration_target = self._get_iteration_target()
    if iteration_target not in {"plate", "char"}:
        try:
            self.app.update_status(
                "Najpierw wybierz w E1 tor iteracji: tablice albo znaki.",
                "warning"
            )
        except Exception:
            pass
        return _reject("Najpierw wybierz w E1 tor iteracji: tablice albo znaki.")
    try:
        current_iteration_path = normalize_iteration_path(CAMPAIGN.get_iteration_path())
    except Exception:
        current_iteration_path = ""
    if (
        iteration_target == "char"
        and current_iteration_path == "char_from_ready_plates"
        and int(CAMPAIGN.get_current_step() or 0) < 3
        and not t02_at_review
    ):
        try:
            self.app.update_status(
                "Ten tor korzysta z istniejącego źródła tablic. Otwórz kontrolę AT przez T02 albo zatwierdź T02, aby przejść do znaków.",
                "info",
            )
        except Exception:
            pass
        try:
            self.request_wizard_stage_focus(step_num=2)
            self._refresh_dashboard()
        except Exception:
            pass
        return _reject("Kontrolę istniejących tablic otwórz przez pole Praca bramki T02.")

    nav_started = perf_counter()
    nav_phase_started = nav_started
    nav_phases: list[str] = []
    nav_overlay_visible = False
    splash_title = "Ładuję kontrolę AT w Z2" if t02_at_review else "Ładuję pracę Z2"
    splash_body = (
        "Wczytuję import anotacji tablic do kontroli. Lista i podgląd pojawią się po nałożeniu runu."
        if t02_at_review
        else "Przygotowuję kontekst pracy bramki i listę obrazów."
    )

    def _mark_nav_phase(name: str) -> None:
        nonlocal nav_phase_started
        try:
            now = perf_counter()
            elapsed_ms = (now - nav_phase_started) * 1000.0
            if elapsed_ms >= 80.0:
                nav_phases.append(f"{name}={elapsed_ms:.0f}ms")
            nav_phase_started = now
        except Exception:
            pass

    def _log_nav_preopen(reason: str) -> None:
        try:
            total_ms = (perf_counter() - nav_started) * 1000.0
            if total_ms < 350.0 and not nav_phases:
                return
            phases = ", ".join(nav_phases) if nav_phases else "ok"
            logger.info(
                "[Z2 NAV PERF] prepare_graph_to_z2 total="
                f"{total_ms:.0f}ms target={iteration_target} "
                f"force={int(bool(force_annotation_tab))} "
                f"t02_review={int(bool(t02_at_review))} "
                f"reason={reason} phases=[{phases}]"
            )
        except Exception:
            pass

    def _show_nav_overlay(progress: float | None, phase: str) -> None:
        nonlocal nav_overlay_visible
        show_overlay = getattr(self, "_show_project_loading_overlay", None)
        if not callable(show_overlay):
            return
        try:
            body = splash_body
            phase_text = str(phase or "").strip()
            if phase_text:
                body = f"{splash_body}\n{phase_text}"
            show_overlay(
                title=splash_title,
                body=body,
                eyebrow="PRZEJŚCIE DO Z2",
                tone="info",
                progress=progress,
            )
            nav_overlay_visible = True
            try:
                self.frame.update_idletasks()
                self.frame.update()
            except Exception:
                try:
                    root = getattr(self.app, "root", None)
                    if root is not None:
                        root.update_idletasks()
                        root.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _hide_nav_overlay() -> None:
        nonlocal nav_overlay_visible
        if not nav_overlay_visible:
            return
        try:
            hide_overlay = getattr(self, "_hide_project_loading_overlay", None)
            if callable(hide_overlay):
                hide_overlay()
        except Exception:
            pass
        nav_overlay_visible = False

    _show_nav_overlay(6.0, "Sprawdzam zasoby wejściowe i ostatni stan pracy.")

    raw_dir = CAMPAIGN.get_dir("raw")
    auto_out = CAMPAIGN.get_staging_dir("auto_ann")
    if auto_out is not None:
        Path(auto_out).mkdir(parents=True, exist_ok=True)
    _mark_nav_phase("dirs")

    if raw_dir is None or auto_out is None:
        _hide_nav_overlay()
        _log_nav_preopen("missing_dirs")
        return _reject("Nie znaleziono katalogów roboczych projektu. Otwórz projekt ponownie.")

    _show_nav_overlay(18.0, "Odtwarzam katalog obrazów bieżącej iteracji.")
    iter_num = CAMPAIGN.get_current_iteration_num()
    try:
        input_dir = CAMPAIGN.get_iteration_image_source_dir(iter_num) or Path(raw_dir)
    except Exception:
        folder = Path(raw_dir) / f"Iteracja_{iter_num:03d}"
        input_dir = folder if folder.exists() else raw_dir
    v_mod = CAMPAIGN.get_global_model("vehicle")
    p_mod = CAMPAIGN.get_global_model("plate")
    _show_nav_overlay(28.0, "Sprawdzam model tablic i źródła poprzedniej pracy.")
    plate_source_state = {} if t02_at_review else self._get_annotation_step2_source_state("plate")
    plate_model_ready = bool(plate_source_state.get("plate_model_ready"))
    if plate_model_ready and (not p_mod or not Path(p_mod).exists()):
        try:
            p_mod = str((plate_source_state.get("bootstrap") or {}).get("plate_model_path") or p_mod or "").strip()
        except Exception:
            p_mod = str(p_mod or "").strip()
    char_source_state = {}
    char_has_existing_source = False
    if iteration_target == "char" and not force_annotation_tab and not t02_at_review:
        _show_nav_overlay(36.0, "Sprawdzam źródło tablic dla toru znaków.")
        char_source_state = self._get_char_route_source_state()
        char_has_existing_source = bool(char_source_state.get("has_source"))
    _mark_nav_phase("source_state")

    if iteration_target == "char" and not force_annotation_tab and not t02_at_review:
        # STEP2-P1 is an entry into Z2, not an implicit approval of E2.
        # A ready plate source only enables the wizard badge; the user can still
        # enter Z2 to add more plate annotations before closing the stage.
        try:
            ready_source = self._get_char_route_ready_source()
        except Exception:
            ready_source = {}
        if ready_source:
            try:
                self.app.update_status(
                    "E2 ma już źródło tablic dla toru znaków. Otwieram Z2, jeśli chcesz dopisać kolejne anotacje; przejście do E3 wymaga jawnego zatwierdzenia etapu.",
                    "info",
                )
            except Exception:
                pass

    tab_ann = self.app.tabs.get("annotation")
    if tab_ann is None or not callable(getattr(tab_ann, "open_campaign_step2_entry", None)):
        try:
            _show_nav_overlay(46.0, "Ładuję moduł Z2 i przygotowuję okno pracy.")
            loader = getattr(self.app, "_ensure_tab_loaded", None)
            if callable(loader):
                tab_ann = loader("annotation", select=False)
            _mark_nav_phase("ensure_tab_loaded")
        except Exception as exc:
            logger.error(f"Nie udało się dociągnąć zakładki Z2 przed wejściem z grafu: {exc}")
            tab_ann = None
    if tab_ann is None or not callable(getattr(tab_ann, "open_campaign_step2_entry", None)):
        if bool(getattr(self.app, "_lazy_tab_load_in_progress", False)) and _retry_count < 20:
            try:
                _hide_nav_overlay()
                _log_nav_preopen("lazy_retry")
                self.frame.after(
                    250,
                    _retry_open,
                )
                self.app.update_status("Kończę ładowanie Z2 i ponowię wejście do pracy bramki.", "info")
            except Exception:
                return _reject("Nie udało się zaplanować ponownego wejścia do Z2. Otwórz projekt ponownie.")
            return {"ok": True, "pending": True, "message": "Trwa ładowanie Z2."}
        try:
            _hide_nav_overlay()
            _log_nav_preopen("tab_unavailable")
            self.app.update_status("Nie udało się przygotować karty Z2 dla pracy tej bramki.", "warning")
        except Exception:
            pass
        return _reject("Nie udało się przygotować karty Z2 dla pracy tej bramki.")

    if t02_at_review:
        try:
            review_root = Path(auto_out) / f"t02_review_iter_{int(iter_num):03d}"
            t02_source = resolve_t02_review_source(
                CAMPAIGN, source_context,
                build_approved_source=lambda: tab_ann._build_campaign_plate_approved_export_source(export_root=review_root),
            )
        except Exception as exc:
            logger.exception("Nie udało się przygotować źródła T02: %s", exc)
            t02_source = {"ok": False, "message": "Nie udało się przygotować zatwierdzonych tablic do kontroli w Z2."}
        if not t02_source.get("ok"):
            _hide_nav_overlay()
            return _reject(t02_source["message"])
        source_context.update(t02_source["context"])

    defer_preview_load = bool(force_annotation_tab or iteration_target == "plate")
    splash_token = 0
    _show_nav_overlay(62.0, "Przygotowuję bezpieczne przełączenie widoku.")
    _mark_nav_phase("z2_splash_prepare")

    try:
        self.app.campaign_free_mode = False
        self.app.set_campaign_mode(True)
    except Exception:
        pass

    def _finish_open() -> None:
        def _complete(ok, message):
            self.app.update_status(message, "info" if ok else "warning")
            if callable(on_complete):
                on_complete({"ok": ok, "message": message})

        if entry_identity != (CAMPAIGN.get_active_project_name(), CAMPAIGN.get_current_iteration_num()):
            _hide_nav_overlay()
            _complete(False, "Anulowano wejście do Z2: zmienił się projekt lub iteracja.")
            return
        finish_started = perf_counter()
        entry_elapsed_ms = 0.0
        switch_elapsed_ms = 0.0
        try:
            _show_nav_overlay(74.0, "Otwieram właściwy kontekst Z2. Lista i podgląd zostaną doładowane po przełączeniu.")
            entry_started = perf_counter()
            result = tab_ann.open_campaign_step2_entry(
                iteration_target=iteration_target,
                entry_strategy=entry_strategy,
                restore_preview=bool(not t02_at_review and not force_annotation_tab and not char_has_existing_source),
                open_existing_run=open_existing_run,
                defer_preview_load=defer_preview_load,
                source_context=dict(source_context or {}),
            )
            entry_elapsed_ms = (perf_counter() - entry_started) * 1000.0
        except Exception as e:
            logger.error(f"Nie udało się otworzyc punktu startowego Z2: {e}")
            _hide_nav_overlay()
            _log_nav_preopen("entry_exception")
            try:
                tab_ann._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass
            _complete(False, "Nie udało się otworzyć Z2. Szczegóły błędu zapisano w logu.")
            return

        if not result.get("ok"):
            _hide_nav_overlay()
            _log_nav_preopen("entry_not_ok")
            try:
                tab_ann._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass
            try:
                reason = str(result.get("reason") or "nieznany powód").strip()
                self.app.update_status(
                    f"Nie udało się otworzyć Z2 z STEP2-P1: {reason}.",
                    "warning",
                )
            except Exception:
                pass
            _complete(False, str(result.get("message") or f"Nie udało się otworzyć Z2: {result.get('reason', 'brak kontekstu')}."))
            return

        try:
            tab_ann.frame.update_idletasks()
        except Exception:
            pass

        try:
            _show_nav_overlay(88.0, "Przełączam widok na Z2.")
            switch_started = perf_counter()
            self.app.open_controlled_tab("annotation")
            selected_tab = getattr(self.app, "_get_selected_tab_key", None)
            if callable(selected_tab) and selected_tab() != "annotation":
                raise RuntimeError("Z2 was not selected after controlled navigation")
            try:
                self.app.root.update_idletasks()
            except Exception:
                pass
            switch_elapsed_ms = (perf_counter() - switch_started) * 1000.0
            _hide_nav_overlay()
            _mark_nav_phase("switch_to_z2")
            _log_nav_preopen("z2_visible")
        except Exception as e:
            logger.error(f"Nie udało się przelaczyc na Z2 po przygotowaniu wejscia: {e}")
            _hide_nav_overlay()
            _log_nav_preopen("switch_exception")
            _complete(False, "Nie udało się przełączyć widoku na Z2. Szczegóły błędu zapisano w logu.")
            return

        _complete(True, "Otworzono kontrolę AT w Z2." if t02_at_review else "Otworzono Z2 w kontekście bramki.")

        try:
            input_dir_local = Path(result.get("input_dir") or ".")
            auto_out_local = Path(result.get("auto_out") or ".")
        except Exception:
            input_dir_local = Path(".")
            auto_out_local = Path(".")
        manual_template = bool(result.get("manual_template"))
        plate_bootstrap_model = str(result.get("plate_model_path") or "").strip()
        input_source = str(result.get("input_source") or "raw").strip()
        opened_existing_run = bool(result.get("opened_existing_run"))
        manual_prepare_deferred_to_user = bool(result.get("manual_prepare_deferred_to_user"))

        try:
            if iteration_target == "plate":
                if opened_existing_run:
                    run_name = ""
                    try:
                        run_name = build_run_display_ref(
                            {"run_dir": result.get("restore_run_dir") or ""},
                            kind_hint="annotation",
                        ).id
                    except Exception:
                        run_name = ""
                    self.app.update_status(
                        (
                            f"Otworzono Z2 bezpośrednio w korekcie runu {run_name}."
                            if run_name
                            else "Otworzono Z2 bezpośrednio w aktywnej korekcie wykrytego runu."
                        ),
                        "info"
                    )
                    try:
                        if bool(result.get("deferred_existing_run_restore")):
                            scheduled = tab_ann._schedule_deferred_campaign_run_restore(
                                Path(str(result.get("restore_run_dir") or "").strip()),
                                status_message="Otworzono Z2. Wczytuję aktywny run i listę obrazów tego katalogu...",
                                splash_token=splash_token,
                            )
                            if not scheduled:
                                tab_ann._campaign_deferred_run_restore_in_progress = False
                                tab_ann._campaign_deferred_run_restore_payload_applied = False
                                tab_ann._refresh_step2_action_states()
                                tab_ann._hide_campaign_step2_splash(token=splash_token)
                            else:
                                _schedule_z2_right_panel_refresh(tab_ann, 600, 1600, 3200)
                        else:
                            tab_ann._hide_campaign_step2_splash(token=splash_token)
                    except Exception:
                        try:
                            tab_ann._campaign_deferred_run_restore_in_progress = False
                            tab_ann._campaign_deferred_run_restore_payload_applied = False
                            tab_ann._refresh_step2_action_states()
                        except Exception:
                            pass
                        try:
                            tab_ann._hide_campaign_step2_splash(token=splash_token)
                        except Exception:
                            pass
                    return
                extra_hint = ""
                if not manual_template and plate_bootstrap_model:
                    extra_hint += " Aktywny model tablic projektu został podstawiony automatycznie."
                if input_source == "stage_previous_iteration":
                    extra_hint += " Jako wejście ustawiono stage z poprzedniej iteracji."
                elif input_source == "manual_source_run":
                    extra_hint += " Przywrócono ostatnie ręczne anotacje tablic dla tego zestawu zdjęć."
                elif input_source == "reused_manual_source_run":
                    extra_hint += " Przywrócono ręczne anotacje z poprzedniej iteracji dla tego samego katalogu zdjęć."
                elif input_source == "latest_approved_run":
                    extra_hint += " Przywrócono też ostatni zatwierdzony run anotacji tablic projektu."
                elif input_source == "reused_training_source_run":
                    extra_hint += " Przywrócono ręczne anotacje z runu anotacji Z2, który zasilił trening w poprzedniej iteracji."
                elif input_source == "reused_iteration_run":
                    extra_hint += " Przywrócono zatwierdzony run anotacji Z2 z poprzedniej iteracji dla tego samego katalogu zdjęć."
                if input_source == "project_imported_manual_source":
                    extra_hint += " Wykorzystano run tablic podpiety na starcie projektu."
                elif input_source == "project_imported_images":
                    extra_hint += " Jako wejście ustawiono obrazy wskazane przy starcie projektu."
                if manual_prepare_deferred_to_user:
                    workflow_hint = (
                        "Z2 czeka na Twoją decyzję: przygotuj roboczy XML ręcznie albo uruchom autoanotację "
                        "dopiero po świadomym wyborze akcji w Z2."
                    )
                elif manual_template:
                    workflow_hint = (
                        "Tryb ręczny utworzy annotations.xml, a nowe polygony zapiszą się z etykietą 'plate'."
                    )
                else:
                    workflow_hint = (
                        "Możesz uruchomić autoanotację tablic aktywnym modelem projektu i ręcznie poprawiać wynik."
                    )
                self.app.update_status(
                    f"Ustawiono Z2 dla toru tablic: IN={Path(input_dir_local).name} | OUT={Path(auto_out_local).name}. "
                    + workflow_hint
                    + extra_hint,
                    "info"
                )
            else:
                if opened_existing_run:
                    run_name = ""
                    try:
                        run_name = build_run_display_ref(
                            {"run_dir": result.get("restore_run_dir") or ""},
                            kind_hint="annotation",
                        ).id
                    except Exception:
                        run_name = ""
                    self.app.update_status(
                        (
                            f"Otworzono Z2 bezpośrednio w korekcie runu {run_name} dla toru znaków."
                            if run_name
                            else "Otworzono Z2 bezpośrednio w korekcie istniejących tablic dla toru znaków."
                        ),
                        "info"
                    )
                else:
                    if manual_template:
                        message = (
                            f"Auto-ustawiono Z2 dla toru znaków: IN={Path(input_dir_local).name} | OUT={Path(auto_out_local).name}. "
                            "Projekt nie ma jeszcze modelu tablic, więc startujesz ręcznie: utwórz XML, oznacz tablice i zatwierdź poprawne zdjęcia."
                        )
                    elif plate_bootstrap_model:
                        message = (
                            f"Auto-ustawiono Z2 dla toru znaków: IN={Path(input_dir_local).name} | OUT={Path(auto_out_local).name}. "
                            "Model tablic aktywnego projektu został podstawiony automatycznie. Przygotuj tablice w Z2, a po zatwierdzeniu przejdziesz do Z3."
                        )
                    else:
                        message = (
                            f"Auto-ustawiono Z2 dla toru znaków: IN={Path(input_dir_local).name} | OUT={Path(auto_out_local).name}. "
                            "Przygotuj źródło tablic w Z2 ręcznie albo wskaż model dopiero przy starcie autoanotacji."
                        )
                    self.app.update_status(message, "info")
        except Exception:
            pass
        if bool(result.get("reused_loaded_run")):
            try:
                tab_ann._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass
            if bool(result.get("repopulate_preview_list")):
                try:
                    tab_ann._populate_preview_list_async(
                        preserve_selection=True,
                        render_current=False,
                        batch_size=220,
                        lightweight_summary=True,
                    )
                except Exception:
                    pass
            try:
                _schedule_z2_right_panel_refresh(tab_ann, 120, lightweight=True)
            except Exception:
                pass
            return
        if bool(result.get("deferred_existing_run_restore")):
            try:
                scheduled = tab_ann._schedule_deferred_campaign_run_restore(
                    Path(str(result.get("restore_run_dir") or "").strip()),
                    status_message="Otworzono Z2. Wczytuję aktywny run i listę obrazów tego katalogu...",
                    splash_token=splash_token,
                )
                if not scheduled:
                    tab_ann._campaign_deferred_run_restore_in_progress = False
                    tab_ann._campaign_deferred_run_restore_payload_applied = False
                    tab_ann._refresh_step2_action_states()
                    tab_ann._hide_campaign_step2_splash(token=splash_token)
                else:
                    _schedule_z2_right_panel_refresh(tab_ann, 600, 1600, 3200)
            except Exception as e:
                logger.debug(f"Nie udało się odroczyć przywrócenia runu Z2 po otwarciu zakładki: {e}")
                try:
                    tab_ann._campaign_deferred_run_restore_in_progress = False
                    tab_ann._campaign_deferred_run_restore_payload_applied = False
                    tab_ann._refresh_step2_action_states()
                except Exception:
                    pass
                try:
                    tab_ann._hide_campaign_step2_splash(token=splash_token)
                except Exception:
                    pass
        if bool(result.get("deferred_preview_load")):
            try:
                deferred_input_dir = Path(str(result.get("deferred_preview_input_dir") or "").strip())
            except Exception:
                deferred_input_dir = None
            if deferred_input_dir is not None:
                try:
                    tab_ann._schedule_deferred_campaign_source_preview_load(
                        deferred_input_dir,
                        status_message="Otworzono Z2. Wczytuję listę obrazów tego katalogu...",
                        splash_token=splash_token,
                    )
                except Exception as e:
                    logger.debug(f"Nie udało się odroczyć wczytania obrazów Z2 po otwarciu zakładki: {e}")
                    try:
                        tab_ann._hide_campaign_step2_splash(token=splash_token)
                    except Exception:
                        pass
            else:
                try:
                    if not bool(result.get("deferred_existing_run_restore")):
                        pass
                except Exception:
                    pass
        elif not bool(result.get("deferred_existing_run_restore")):
            try:
                tab_ann._hide_campaign_step2_splash(token=splash_token)
            except Exception:
                pass
        total_elapsed_ms = (perf_counter() - finish_started) * 1000.0
        if total_elapsed_ms >= 500.0:
            try:
                logger.info(
                    "[Z2 PERF] graph_to_z2 total="
                    f"{total_elapsed_ms:.0f}ms entry={entry_elapsed_ms:.0f}ms "
                    f"tab_switch={switch_elapsed_ms:.0f}ms target={iteration_target} "
                    f"force={int(bool(force_annotation_tab))} "
                    f"deferred_preview={int(bool(result.get('deferred_preview_load')))} "
                    f"deferred_run={int(bool(result.get('deferred_existing_run_restore')))}"
                )
            except Exception:
                pass

    try:
        self.frame.after(25, _finish_open)
    except Exception:
        _finish_open()
    return {"ok": True, "pending": True, "message": "Otwieram Z2…"}


def _return_to_step1_for_char_source_rework(self, *, clear_target: bool = False) -> None:
    try:
        CAMPAIGN.reset_step3()
        CAMPAIGN.reset_step2()
        CAMPAIGN.reset_step1()
        CAMPAIGN.set_current_step(1)
        if clear_target:
            CAMPAIGN.set_iteration_target("")
    except Exception as e:
        logger.debug(f"Nie udało się cofnąć kampanii do E1 po braku minimum tablic: {e}")

    self.current_ingest_plan = {}
    self.step1_panel_expanded = True
    try:
        self.request_wizard_stage_focus(step_num=1)
    except Exception:
        pass
    try:
        self._refresh_dashboard()
    except Exception:
        pass
    try:
        self.app.open_controlled_tab("campaign")
    except Exception:
        pass
    try:
        if clear_target:
            self.app.update_status(
                "Wrócono do E1. Wybierz tor iteracji i katalog zdjęć przed ponownym zatwierdzeniem.",
                "warning",
            )
        else:
            self.app.update_status(
                "Wrócono do E1. Wybierz większy katalog zdjęć albo ponownie zatwierdź E1 po uzupełnieniu źródła.",
                "warning",
            )
    except Exception:
        pass


def _show_step2_char_minimum_not_met_dialog(
    self,
    *,
    source_images: int,
    source_plates: int,
    min_plates: int,
    run_name: str = "",
) -> str:
    source_images = max(0, int(source_images or 0))
    source_plates = max(0, int(source_plates or 0))
    min_plates = max(1, int(min_plates or self.STEP3_CHAR_MIN_PLATES))
    missing_plates = max(0, min_plates - source_plates)
    run_display_name = ""
    if str(run_name or "").strip():
        try:
            run_display_name = build_run_display_ref({"run_name": run_name}, kind_hint="annotation").id
        except Exception:
            run_display_name = str(run_name or "").strip()
    run_line = f"Źródło: {run_display_name}\n" if run_display_name else ""
    message = (
        f"{run_line}"
        "E2 nie ma jeszcze minimum do przejścia w tor znaków.\n\n"
        f"Zatwierdzone obrazy z tablicami: {source_images}\n"
        f"Gotowe tablice: {source_plates}\n"
        f"Minimum dla E3: {min_plates} tablic\n"
        f"Brakuje: {missing_plates} tablic\n\n"
        "Możesz dalej oznaczać tablice w Z2, wrócić do E1 po większy katalog zdjęć "
        "albo wrócić do E1 i zmienić tor iteracji."
    )
    buttons = ["Zmień tor", "Wróć do E1", "Oznacz dalej w Z2"]
    try:
        choice = self.app.themed_message_dialog(
            "Za mało tablic dla toru znaków",
            message,
            parent=self.frame,
            buttons=buttons,
            default_button="Oznacz dalej w Z2",
            tone="warning",
            wraplength=560,
        )
    except Exception:
        try:
            messagebox.showwarning("Za mało tablic dla toru znaków", message, parent=self.frame)
        except Exception:
            pass
        choice = "Oznacz dalej w Z2"
    return str(choice or "Oznacz dalej w Z2").strip()


def _finish_step2_char_and_focus_step3(
    self,
    preferred_source_context: dict | None = None,
    *,
    approve_step2: bool = False,
) -> bool:
    if not CAMPAIGN.get_active_project_name() or CAMPAIGN.get_current_step() < 2:
        return False

    source_state = self._get_char_route_source_state()
    if source_state.get("needs_more_tables"):
        source_images = int(source_state.get("images_with_plates", 0) or 0)
        source_plates = int(source_state.get("total_plates", 0) or 0)
        min_plates = int(getattr(self, "STEP3_CHAR_MIN_PLATES", 10) or 10)
        missing_plates = max(0, min_plates - source_plates)
        run_name = str(source_state.get("run_name", "") or "").strip()
        run_display_name = ""
        if run_name:
            try:
                run_display_name = build_run_display_ref({"run_name": run_name}, kind_hint="annotation").id
            except Exception:
                run_display_name = run_name
        try:
            self.app.update_status(
                (
                    f"Run {run_display_name or run_name}: zatwierdzonych obrazów {source_images}, zapisanych tablic {source_plates}. "
                    f"Minimum do wejścia do znaków: {min_plates} tablic; brakuje {missing_plates}. "
                    "Najpierw przygotuj więcej tablic w Z2, a dopiero potem przejdź do znaków."
                )
                if run_display_name or run_name
                else (
                    f"Zatwierdzonych obrazów: {source_images}. Zapisanych tablic: {source_plates}. "
                    f"Minimum do wejścia do znaków: {min_plates} tablic; brakuje {missing_plates}. "
                    "Najpierw przygotuj więcej tablic w Z2, a dopiero potem przejdź do znaków."
                ),
                "warning",
            )
        except Exception:
            pass
        choice = self._show_step2_char_minimum_not_met_dialog(
            source_images=source_images,
            source_plates=source_plates,
            min_plates=min_plates,
            run_name=run_display_name or run_name,
        )
        if choice == "Wróć do E1":
            self._return_to_step1_for_char_source_rework(clear_target=False)
        elif choice == "Zmień tor":
            self._return_to_step1_for_char_source_rework(clear_target=True)
        else:
            self._step_return_to_annotation_review(mark_step3_rework=False)
        return False

    source_context = preferred_source_context if isinstance(preferred_source_context, dict) else {}
    if not source_context:
        source_context = self._get_char_route_ready_source()
    if not source_context:
        try:
            self.app.update_status(
                "Nie znaleziono gotowych tablic dla tego katalogu zdjęć. Najpierw przygotuj tablice, potem przejdź do pracy nad znakami.",
                "warning",
            )
        except Exception:
            pass
        return False

    ready_run_dir = source_context.get("restore_run_dir")
    ready_run_name = str(
        source_context.get("display_name")
        or source_context.get("run_name")
        or ""
    ).strip()
    try:
        if not ready_run_name and ready_run_dir is not None:
            ready_run_name = build_run_display_ref({"run_dir": ready_run_dir}, kind_hint="annotation").id
        elif ready_run_name:
            ready_run_name = build_run_display_ref({"run_name": ready_run_name}, kind_hint="annotation").id
    except Exception:
        ready_run_name = ""

    try:
        self._suppress_stale_char_iteration_reset_until = perf_counter() + 2.5
    except Exception:
        pass

    try:
        step2_status = str(CAMPAIGN.get_step2_status() or "").strip().lower()
    except Exception:
        step2_status = ""
    if step2_status != "approved":
        if not approve_step2:
            try:
                self.app.update_status(
                    "E2 nie zostało jeszcze zatwierdzone. Wejście do E3 wymaga użycia badge'a „Zatwierdź etap”.",
                    "warning",
                )
            except Exception:
                pass
            return False
        CAMPAIGN.approve_step2()
    if CAMPAIGN.get_current_step() < 3:
        CAMPAIGN.set_current_step(3)

    try:
        self.request_wizard_stage_focus(step_num=3)
    except Exception:
        pass
    self._refresh_dashboard()
    try:
        self.app.open_controlled_tab("campaign")
    except Exception:
        pass
    self.app.update_campaign_tab_access()

    try:
        run_hint = f" Korzystam z runu anotacji {ready_run_name}." if ready_run_name else ""
        self.app.update_status(
            "Znaleziono gotowe ręczne tablice dla tego katalogu zdjęć. "
            "E2 zostało zatwierdzone. Przechodzę do E3 w wizardzie."
            + run_hint,
            "info"
        )
    except Exception:
        pass

    return True


def _step_continue_characters_from_ready_source(self, preferred_source_context: dict | None = None):
    source_context = dict(preferred_source_context) if isinstance(preferred_source_context, dict) else {}
    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
    except Exception:
        current_step = 0
    if current_step < 3:
        if not self._finish_step2_char_and_focus_step3(preferred_source_context):
            return
        source_context = dict(preferred_source_context) if isinstance(preferred_source_context, dict) else {}
        if not source_context:
            source_context = self._get_char_route_ready_source()
    elif not source_context:
        source_context = self._get_char_route_ready_source()
        if not source_context:
            try:
                self.app.update_status(
                    "T06 nie ma gotowego źródła tablic dla Z3. Wróć do Z2 albo sprawdź zasoby bramki.",
                    "warning",
                )
            except Exception:
                pass
            return
    self._step_goto_characters(preferred_source_context=source_context)


def _step_goto_characters_detect(self, preferred_source_context: dict | None = None):
    if not CAMPAIGN.get_active_project_name() or CAMPAIGN.get_current_step() < 3:
        return

    source_context = dict(preferred_source_context or {})
    source_context["target_substep"] = "detect"
    source_context["force_pz2"] = "1"
    self._step_goto_characters(preferred_source_context=source_context)


def _step_goto_characters(self, preferred_source_context: dict | None = None):
    if not CAMPAIGN.get_active_project_name() or CAMPAIGN.get_current_step() < 3:
        return

    source_context = dict(preferred_source_context) if isinstance(preferred_source_context, dict) else {}
    if not source_context and self._get_iteration_target() == "char":
        source_context = self._get_char_route_ready_source()

    def _t06_should_start_from_pz3() -> bool:
        try:
            iteration_state = dict(CAMPAIGN.get_iteration_state() or {})
            contracts = dict(iteration_state.get("t06_contracts") or {})
            pz2_contract = dict(contracts.get("pz2_char_boxes") or {})
            if bool(pz2_contract.get("fulfilled")):
                return True
            session = dict(iteration_state.get("t06_work_session") or {})
            session_substep = str(session.get("substep") or "").strip().lower()
            session_area = str(session.get("work_area") or "").strip().lower()
            if session_area == "z3" and session_substep in {"3", "pz3", "dataset"}:
                return True
        except Exception:
            pass
        try:
            return bool(CAMPAIGN.is_step3_stage2_done())
        except Exception:
            return False

    should_prime_z3_surface = False
    should_prime_dataset_surface = False
    try:
        gate_hint = str(source_context.get("graph_gate_id") or source_context.get("gate_id") or "").strip().upper()
        target_hint = str(
            source_context.get("target_substep")
            or source_context.get("graph_target_substep")
            or source_context.get("preferred_substep")
            or ""
        ).strip().lower()
        explicit_pz2 = bool(source_context.get("force_pz2")) or target_hint in {"2", "detect", "pz2", "z3_pz2"}
        explicit_pz3 = bool(source_context.get("force_pz3")) or target_hint in {"3", "dataset", "pz3", "z3_pz3"}
        if gate_hint == "T06":
            # T06 ma dwa kroki robocze: PZ2 przygotowuje ramki znaków, a PZ3 eksportuje AZ.
            # Jeśli kontrakt PZ2 jest już spełniony, kontynuacja powinna wracać od razu do PZ3.
            if explicit_pz3:
                source_context["target_substep"] = "pz3"
                source_context.pop("force_pz2", None)
            elif explicit_pz2:
                source_context["target_substep"] = "detect"
                source_context["force_pz2"] = "1"
            elif _t06_should_start_from_pz3():
                source_context["target_substep"] = "pz3"
                source_context.pop("force_pz2", None)
            else:
                source_context["target_substep"] = "detect"
                source_context["force_pz2"] = "1"
        target_hint = str(
            source_context.get("target_substep")
            or source_context.get("graph_target_substep")
            or source_context.get("preferred_substep")
            or ""
        ).strip().lower()
        should_prime_z3_surface = (
            bool(source_context.get("force_pz2"))
            or target_hint in {"2", "detect", "pz2", "z3_pz2"}
        )
        should_prime_dataset_surface = (
            not should_prime_z3_surface
            and target_hint in {"3", "dataset", "pz3", "z3_pz3"}
        )
    except Exception:
        should_prime_z3_surface = False
        should_prime_dataset_surface = False

    tab_char = None
    try:
        if should_prime_z3_surface or should_prime_dataset_surface:
            previous_target = str(getattr(self.app, "_controlled_tab_target_key", "") or "").strip()
            self.app._controlled_tab_target_key = "characters"
            try:
                setattr(self.app, "_suppress_characters_lazy_first_paint_overlay_once", True)
                current_tab = self.app.tabs.get("characters")
                frame = getattr(current_tab, "frame", None)
                if frame is not None:
                    try:
                        self.app.notebook.tab(str(frame), state="normal")
                    except Exception:
                        pass
                tab_char = self.app._ensure_tab_loaded("characters", select=False)
            finally:
                try:
                    setattr(self.app, "_suppress_characters_lazy_first_paint_overlay_once", False)
                except Exception:
                    pass
                self.app._controlled_tab_target_key = previous_target
        else:
            self.app.open_controlled_tab("characters")
            tab_char = self.app.tabs.get("characters")
    except Exception as e:
        try:
            setattr(self.app, "_suppress_characters_lazy_first_paint_overlay_once", False)
        except Exception:
            pass
        logger.error(f"Nie udało się przejść do Z3: {e}")
        return

    if tab_char is None:
        tab_char = self.app.tabs.get("characters")
    if not tab_char or not hasattr(tab_char, "open_campaign_step3_entry"):
        logger.error("Nie udało się otworzyć Z3: zakładka znaków nie jest gotowa.")
        return

    try:
        tab_char._step3_linear_mode = True
        if should_prime_dataset_surface:
            tab_char._campaign_pz2_sync_loading = False
            tab_char._campaign_step3_entry_splash_pinned = False
            tab_char._campaign_detect_splash_force_root_surface = False
            tab_char._set_subtab_state(tab_char.tab_extract, "disabled")
            tab_char._set_subtab_state(tab_char.tab_detect, "disabled")
            tab_char._set_subtab_state(tab_char.tab_dataset, "normal")
            tab_char._select_subtab(tab_char.tab_dataset)
            try:
                tab_widget = str(tab_char.frame)
                self.app.notebook.tab(tab_widget, state="normal")
                self.app.notebook.select(tab_widget)
                self.app.update_campaign_tab_access()
            except Exception:
                pass
        if should_prime_z3_surface:
            tab_char._campaign_pz2_sync_loading = True
            tab_char._campaign_step3_entry_splash_pinned = True
            tab_char._campaign_detect_splash_force_root_surface = True
            tab_char._set_subtab_state(tab_char.tab_detect, "normal")
            tab_char._set_subtab_state(tab_char.tab_dataset, "disabled")
        if should_prime_z3_surface:
            tab_char._set_subtab_state(tab_char.tab_extract, "normal")
            tab_char._show_campaign_detect_splash(
                title="Wyodrębniam tablice dla Z3",
                body="Startuję wyodrębnianie tablic z aktualnego źródła T06.",
                tone="info",
                progress=0.0,
                show_progress=True,
                show_return=False,
            )
            try:
                tab_char.frame.update_idletasks()
            except Exception:
                pass
            try:
                tab_widget = str(tab_char.frame)
                self.app.notebook.tab(tab_widget, state="normal")
                self.app.notebook.select(tab_widget)
                self.app.update_campaign_tab_access()
            except Exception:
                pass
            try:
                overlay = getattr(tab_char, "campaign_detect_splash_overlay", None)
                if overlay is not None and overlay.winfo_exists():
                    overlay.lift()
                    tab_char.frame.after_idle(overlay.lift)
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"Nie udało się przygotować wczesnego ekranu Z3/PZ2: {exc}")

    try:
        result = tab_char.open_campaign_step3_entry(preferred_source_context=source_context)
    except Exception as e:
        try:
            tab_char._campaign_pz2_sync_loading = False
            tab_char._campaign_step3_entry_splash_pinned = False
            tab_char._hide_campaign_detect_splash()
        except Exception:
            pass
        logger.error(f"Nie udało się otworzyc punktu startowego Z3: {e}")
        return

    if not result.get("ok"):
        try:
            tab_char._campaign_pz2_sync_loading = False
            tab_char._campaign_step3_entry_splash_pinned = False
            tab_char._hide_campaign_detect_splash()
        except Exception:
            pass
        return

    latest_xml = str(result.get("latest_xml") or "").strip()
    images_dir = str(result.get("images_dir") or "").strip()
    using_preferred_source = bool(result.get("using_preferred_source"))
    preferred_run_dir_raw = str(result.get("preferred_run_dir") or "").strip()
    preferred_run_dir = Path(preferred_run_dir_raw) if preferred_run_dir_raw else None
    preferred_source_name = str(
        source_context.get("display_name")
        or source_context.get("run_name")
        or ""
    ).strip()

    try:
        folder_name = Path(images_dir).name if images_dir else ""
    except Exception:
        folder_name = ""

    try:
        if latest_xml:
            if using_preferred_source and preferred_run_dir is not None:
                source_label = preferred_source_name or preferred_run_dir.name
                self.app.update_status(
                    f"Ustawiono Z3 na gotowe źródło tablic: {source_label}. XML={preferred_run_dir.name}/annotations.xml | IMG={folder_name}.",
                    "info"
                )
            else:
                self.app.update_status(
                    f"Ustawiono świeże źródła dla Zakładki Znaków: XML={Path(latest_xml).parent.name}/annotations.xml | IMG={folder_name}.",
                    "info"
                )
        else:
            self.app.update_status(
                "Nie znaleziono nowego pliku annotations.xml. Upewnij się, ze Autoanotacja zakończyła się sukcesem i etap został zatwierdzony.",
                "warning"
            )
    except Exception:
        pass

    try:
        tab_char._campaign_step3_entry_splash_pinned = False
        if (
            not bool(getattr(tab_char, "_campaign_detect_splash_visible", False))
            and not bool(getattr(tab_char, "is_processing", False))
        ):
            tab_char._campaign_pz2_sync_loading = False
    except Exception:
        pass


def _step_goto_training(self, preferred_subtab: str | None = None):
    if not CAMPAIGN.get_active_project_name() or CAMPAIGN.get_current_step() < 4:
        return

    try:
        iteration_target = self._get_iteration_target()
        if iteration_target not in {"plate", "char"}:
            iteration_target = "char"
        requested_subtab = str(preferred_subtab or "").strip().lower()
        if requested_subtab not in {"dataset", "train"}:
            requested_subtab = ""

        tab_train = self.app.tabs.get("training")
        if not tab_train:
            logger.error("Nie znaleziono zakładki TrainingTab w app.tabs.")
            return

        preferred_subtab = requested_subtab or None
        readiness = None
        try:
            if hasattr(tab_train, "get_campaign_step4_readiness"):
                readiness = tab_train.get_campaign_step4_readiness(iteration_target=iteration_target)
        except Exception as e:
            logger.debug(f"Nie udało się sprawdzic gotowosci wejscia do Z4: {e}")
            readiness = None

        if isinstance(readiness, dict) and not readiness.get("ok", False):
            reason = str(readiness.get("reason") or "").strip().lower()
            if reason == "stale_plate_dataset":
                if requested_subtab == "train":
                    warn_msg = str(readiness.get("message") or "").strip() or "Najpierw przebuduj wariant datasetu dla bieżącej iteracji."
                    try:
                        self.app.update_status(warn_msg, "warning")
                    except Exception:
                        pass
                    try:
                        messagebox.showwarning("Z4 jeszcze zablokowane", warn_msg, parent=self.frame)
                    except Exception:
                        pass
                    return
                preferred_subtab = "dataset"
                warn_msg = str(readiness.get("message") or "").strip()
                try:
                    if warn_msg:
                        self.app.update_status(warn_msg, "warning")
                except Exception:
                    pass
            else:
                can_open_char_dataset_stage = False
                if iteration_target == "char" and reason in {"invalid_char_dataset", "missing_char_dataset"}:
                    try:
                        datasets_dir = CAMPAIGN.get_dir("datasets")
                        if datasets_dir is not None and hasattr(tab_train, "_find_dataset_source_candidates"):
                            for path in tab_train._find_dataset_source_candidates(Path(datasets_dir)):
                                try:
                                    inferred = tab_train._infer_dataset_target(str(path))
                                except Exception:
                                    inferred = "char"
                                if inferred == "char":
                                    can_open_char_dataset_stage = True
                                    break
                    except Exception:
                        can_open_char_dataset_stage = False

                if can_open_char_dataset_stage and requested_subtab != "train":
                    preferred_subtab = "dataset"
                else:
                    warn_msg = str(readiness.get("message") or "").strip() or "Z4 nie jest jeszcze gotowe do otwarcia."
                    try:
                        self.app.update_status(warn_msg, "warning")
                    except Exception:
                        pass
                    try:
                        messagebox.showwarning("Z4 jeszcze zablokowane", warn_msg, parent=self.frame)
                    except Exception:
                        pass
                    return

            if not readiness.get("ok", False) and preferred_subtab != "dataset":
                warn_msg = str(readiness.get("message") or "").strip() or "Z4 nie jest jeszcze gotowe do otwarcia."
                try:
                    self.app.update_status(warn_msg, "warning")
                except Exception:
                    pass
                try:
                    messagebox.showwarning("Z4 jeszcze zablokowane", warn_msg, parent=self.frame)
                except Exception:
                    pass
                return
        elif isinstance(readiness, dict):
            reason = str(readiness.get("reason") or "").strip().lower()
            if requested_subtab == "dataset":
                preferred_subtab = "dataset"
            elif requested_subtab == "train":
                preferred_subtab = "train"
            elif iteration_target == "char" and reason == "source_dataset_ready_for_split":
                preferred_subtab = "dataset"
            elif readiness.get("ok", False):
                ready_dataset = str(readiness.get("ready_dataset") or "").strip()
                ready_train = int(readiness.get("train_images", 0) or 0)
                ready_val = int(readiness.get("val_images", 0) or 0)
                if ready_dataset and ready_train > 0 and ready_val > 0:
                    preferred_subtab = "train"
                elif iteration_target == "plate":
                    preferred_subtab = "dataset"
                    try:
                        self.app.themed_message_dialog(
                            "Najpierw utwórz wariant datasetu",
                            (
                                "Bramka T07 ma już materiał projektu, ale nie ma jeszcze gotowego wariantu "
                                "datasetu tablic z podziałem train / val / test.\n\n"
                                "Otwieram Z4/PZ1. Utworzenie wariantu nie zamyka T07; dopiero trening "
                                "albo świadome zakończenie bez treningu pozwoli wrócić do grafu i zatwierdzić bramkę."
                            ),
                            parent=self.frame,
                            buttons=["OK"],
                            default_button="OK",
                            tone="info",
                            wraplength=560,
                        )
                    except Exception:
                        try:
                            messagebox.showinfo(
                                "Najpierw utwórz wariant datasetu",
                                (
                                    "Bramka T07 ma już materiał projektu, ale nie ma jeszcze gotowego wariantu "
                                    "datasetu tablic z podziałem train / val / test.\n\n"
                                    "Otwieram Z4/PZ1. Utworzenie wariantu nie zamyka T07."
                                ),
                                parent=self.frame,
                            )
                        except Exception:
                            pass

        target_label = "tablic" if iteration_target == "plate" else "znaków"
        stage_label = "E4T" if iteration_target == "plate" else "E4Z"
        try:
            self._show_project_loading_overlay(
                title="Przygotowuję Z4",
                body=(
                    f"Odtwarzam kontekst {stage_label} dla toru {target_label}, sprawdzam dataset "
                    "i przygotowuję zakładkę treningu."
                ),
                tone="info",
                progress=None,
            )
            self.frame.update_idletasks()
            self.frame.update()
        except Exception:
            pass
        try:
            self.app.update_status("Przygotowuję Z4 i odtwarzam kontekst treningu.", "info")
        except Exception:
            pass

        try:
            result = tab_train.open_campaign_step4_entry(
                iteration_target=iteration_target,
                preferred_subtab=preferred_subtab,
            )
        except Exception as e:
            try:
                self._hide_project_loading_overlay()
            except Exception:
                pass
            logger.error(f"Błąd otwierania punktu startowego Z4: {e}")
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
            return

        latest_source_raw = str(result.get("latest_source") or "").strip()
        latest_source = Path(latest_source_raw) if latest_source_raw else None
        dataset_hint = str(result.get("dataset_hint") or "").strip()

        try:
            if iteration_target == "plate":
                if dataset_hint:
                    self.app.update_status(
                        f"Ustawiono tor treningu tablic: gotowy dataset = {Path(dataset_hint).name}, źródła XML z Z2 i model Pose.",
                        "info"
                    )
                else:
                    self.app.update_status(
                        "Przelaczono do Treningu w torze tablic. Zbuduj dataset z XML CVAT i uruchom trening modelu Pose.",
                        "info"
                    )
            elif latest_source is not None:
                self.app.update_status(
                    f"Ustawiono automatycznie Trening: źródło splittera = {latest_source.name}, wynik splitu w katalogu projektu oraz model DETECT dla znaków.",
                    "info"
                )
            else:
                self.app.update_status(
                    "Przelaczono do Treningu w kontekscie projektu, ale nie znaleziono jeszcze datasetu źródłowego w 4_training_datasets.",
                    "warning"
                )
        except Exception:
            pass

        try:
            self._show_project_loading_overlay(
                title="Przygotowuję Z4",
                body=f"Kontekst {stage_label} jest gotowy. Przełączam widok na zakładkę treningu.",
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
    except Exception as e:
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
        logger.error(f"Błąd nawigacji (Krok 4): {e}")
