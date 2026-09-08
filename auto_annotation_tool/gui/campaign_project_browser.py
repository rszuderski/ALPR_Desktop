from __future__ import annotations

"""
Zakładka: Panel kampanii i etapow projektu.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from textwrap import shorten
from collections import Counter
from dataclasses import dataclass
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
from .help_manager import HELP
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z2_view_models import Step2CtaViewModel, Step2ViewModel
from .z3_view_models import Step3ViewModel
def _ensure_project_context_is_switchable(
    self,
    *,
    next_project: str = "",
    dialog_title: str = "Najpierw zakończ aktywną operację",
) -> bool:
    if bool(getattr(self, "_project_switch_in_progress", False)):
        try:
            self.app.update_status(
                "Trwa już przełączanie projektu. Poczekaj na zakończenie poprzedniego ładowania.",
                "info",
            )
        except Exception:
            pass
        return False

    active_before = str(CAMPAIGN.get_active_project_name() or "").strip()
    if not active_before:
        return True

    blocker_message = self._get_project_switch_blocker_message(
        current_project=active_before,
        next_project=str(next_project or "").strip(),
    )
    if not blocker_message:
        return True

    self.app.themed_info(
        dialog_title,
        blocker_message,
        parent=self.frame,
        tone="warning",
    )
    return False

def _reset_campaign_graph_runtime_state(self) -> None:
    """Clear project-specific graph UI state before switching project context."""
    for attr_name, value in (
        ("_campaign_graph_selected_edge_key", ""),
        ("_campaign_graph_attention_seen_keys", set()),
        ("_campaign_graph_attention_scope", None),
        ("_campaign_graph_gate_offsets", {}),
        ("_campaign_graph_node_offsets", {}),
        ("_campaign_graph_gate_zoom_scales", {}),
        ("_campaign_graph_view", {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0}),
    ):
        try:
            setattr(self, attr_name, value)
        except Exception:
            pass

    try:
        state = getattr(self, "_campaign_graph_attention_state", None)
        if isinstance(state, dict):
            state["after_id"] = None
            state["defer_after_id"] = None
            state["running_key"] = ""
            state["canvas"] = None
            state["token"] = int(state.get("token", 0) or 0) + 1
    except Exception:
        pass

    for attr_name in (
        "_campaign_graph_active_hover_item",
        "_campaign_graph_active_drag_item",
        "_campaign_graph_selected_badge_key",
    ):
        try:
            if hasattr(self, attr_name):
                setattr(self, attr_name, "")
        except Exception:
            pass

def _add_new_project(self):
    new_name = self.app.themed_ask_string(
        "Nowy projekt",
        "Podaj unikalną nazwę projektu.",
        parent=self.frame,
        action_label="Utwórz"
    )
    if not new_name:
        return

    if not self._ensure_project_context_is_switchable(
        next_project=str(new_name or "").strip(),
        dialog_title="Najpierw zakończ aktywny projekt",
    ):
        return

    active_before = str(CAMPAIGN.get_active_project_name() or "").strip()

    try:
        annotation_tab = self.app.tabs.get("annotation")
        if annotation_tab is not None and hasattr(annotation_tab, "capture_free_mode_snapshot_for_project_return"):
            annotation_tab.capture_free_mode_snapshot_for_project_return()
    except Exception as e:
        logger.debug(f"Nie udało się zapisać migawki free mode przed utworzeniem projektu: {e}")

    self._project_switch_in_progress = True
    try:
        self._reset_campaign_graph_runtime_state()
        if active_before:
            try:
                self._release_active_project_resources_before_switch(active_before)
            except Exception as e:
                logger.debug(f"Nie udało się zwolnić zasobów projektu '{active_before}' przed utworzeniem nowego: {e}")

        if CAMPAIGN.create_project(new_name):
            self._reset_campaign_graph_runtime_state()
            self.app.campaign_free_mode = False
            self.app.set_campaign_mode(True)
            try:
                CAMPAIGN.set_project_start_mode("fresh")
            except Exception:
                pass
            self._rebuild_wizard_stage_ui()
            self._refresh_dashboard()
            self.app.update_campaign_tab_access()
            self.app.themed_info(
                "Projekt utworzony",
                (
                    f"Projekt '{new_name}' został utworzony.\n\n"
                    "W lewym dolnym rogu aplikacji znajduje się globalny asystent AS. "
                    "Po kliknięciu ikony AS możesz rozwinąć krótkie podpowiedzi dotyczące aktualnej zakładki, "
                    "etapu i celu pracy."
                ),
                parent=self.frame,
                tone="success",
            )
        else:
            self.app.themed_error(
                "Błąd",
                "Projekt o takiej nazwie już istnieje lub nazwa jest nieprawidłowa.",
                parent=self.frame
            )
    finally:
        self._project_switch_in_progress = False

def _get_selected_projects_from_list(self) -> list[str]:
    if self.project_listbox is None:
        return []

    try:
        selected = []
        for raw_idx in self.project_listbox.curselection():
            idx = int(raw_idx)
            if 0 <= idx < len(self._project_name_by_index):
                project_name = str(self._project_name_by_index[idx]).strip()
                if project_name and project_name not in selected:
                    selected.append(project_name)
        return selected
    except Exception:
        return []

def _get_selected_project_from_list(self) -> str:
    selected = self._get_selected_projects_from_list()
    return selected[0] if selected else ""

def _select_project_in_list(self, project_name: str):
    if self.project_listbox is None or not project_name:
        return

    try:
        idx = self._project_name_by_index.index(project_name)
        self.project_listbox.selection_clear(0, tk.END)
        self.project_listbox.selection_set(idx)
        self.project_listbox.activate(idx)
        self.project_listbox.see(idx)
        self.project_listbox.focus_set()
    except Exception:
        pass

def _show_project_context_menu(self, event):
    if self.project_listbox is None or not getattr(self, "project_context_menu", None):
        return "break"

    try:
        idx = int(self.project_listbox.nearest(event.y))
    except Exception:
        return "break"

    if idx < 0 or idx >= len(getattr(self, "_project_name_by_index", [])):
        return "break"

    try:
        selected_now = set(int(value) for value in self.project_listbox.curselection())
        if idx not in selected_now:
            self.project_listbox.selection_clear(0, tk.END)
            self.project_listbox.selection_set(idx)
        self.project_listbox.activate(idx)
        self.project_listbox.see(idx)
        self.project_listbox.focus_set()
        self._on_project_changed()
        selected_projects = self._get_selected_projects_from_list()
        try:
            self.project_context_menu.entryconfig(
                0,
                label="Otwórz projekt" if len(selected_projects) <= 1 else "Otwórz pierwszy projekt"
            )
            self.project_context_menu.entryconfig(
                2,
                label="Usuń zaznaczony projekt" if len(selected_projects) <= 1 else "Usuń zaznaczone projekty"
            )
        except Exception:
            pass
        self.project_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        try:
            self.project_context_menu.grab_release()
        except Exception:
            pass
    return "break"

def _set_project_list_status(self, selected: str = "", project_count: int = 0):
    if self.project_list_status_lbl is None:
        return

    def compact(value: str, width: int = 54) -> str:
        text = str(value or "").replace("\n", " ").strip()
        if not text:
            return ""
        return shorten(text, width=width, placeholder="...")

    def apply_lines(*lines: str):
        labels = list(getattr(self, "project_list_status_labels", []))
        if not labels:
            return
        normalized = [compact(line) for line in lines[:3]]
        while len(normalized) < len(labels):
            normalized.append("")
        for lbl, line in zip(labels, normalized):
            lbl.config(text=line)

    selected_projects = self._get_selected_projects_from_list()
    if len(selected_projects) > 1:
        apply_lines(
            f"Zaznaczone projekty: {len(selected_projects)}",
            f"Pierwszy na liście: {selected_projects[0]}",
            "Dwuklik otwiera pierwszy projekt. PPM usuwa całe zaznaczenie."
        )
        return

    if not selected:
        if project_count > 0:
            apply_lines(
                f"Zapisane projekty: {project_count}",
                "Wybierz projekt z listy.",
                "Dwuklik otwiera zaznaczony projekt."
            )
        else:
            apply_lines(
                "Brak zapisanych projektów.",
                "Utwórz pierwszy projekt.",
                "Lista pojawi się po dodaniu projektu."
            )
        return

    created_at = ""
    try:
        created_raw = CAMPAIGN.get_project_created_at(selected)
        if created_raw:
            formatter = getattr(self.app, "_format_project_created_at", None)
            created_at = formatter(created_raw) if callable(formatter) else str(created_raw).replace("T", " ")
    except Exception:
        created_at = ""

    created_line = f"Utworzono: {created_at}" if created_at else "Utworzono: brak danych"

    if selected == CAMPAIGN.get_active_project_name():
        action_line = "Projekt jest już aktywny."
    else:
        action_line = "Dwuklik otwiera zaznaczony projekt."

    apply_lines(
        f"Projekt: {selected}",
        created_line,
        action_line
    )

def _refresh_projects_list(self):
    if self.project_listbox is None:
        return

    projects = CAMPAIGN.get_all_projects()
    active_project = CAMPAIGN.get_active_project_name()
    previous_selection = self._get_selected_project_from_list()

    self._project_list_refreshing = True
    try:
        self.project_listbox.delete(0, tk.END)
        self._project_name_by_index = []

        if not projects:
            self._set_project_list_status("", 0)
            self.project_listbox.insert(tk.END, "(brak zapisanych projektów)")
            self.project_listbox.itemconfig(0, foreground="#888888")
            self.frame.after_idle(self._sync_left_panel_scrollregion)
            return

        for project_name in projects:
            display_name = f"* {project_name}" if project_name == active_project else project_name
            self.project_listbox.insert(tk.END, display_name)
            idx = len(self._project_name_by_index)
            self._project_name_by_index.append(project_name)
            if project_name == active_project:
                self.project_listbox.itemconfig(idx, foreground="#27ae60")

        target_name = active_project or previous_selection or projects[0]
        self._select_project_in_list(target_name)
    finally:
        self._project_list_refreshing = False

    self._set_project_list_status(self._get_selected_project_from_list(), len(projects))
    self.frame.after_idle(self._sync_left_panel_scrollregion)

def _on_project_changed(self, event=None):
    if self._project_list_refreshing:
        return

    selected_projects = self._get_selected_projects_from_list()
    if not selected_projects or self.project_list_status_lbl is None:
        return

    try:
        self._set_project_list_status(selected_projects[0], len(self._project_name_by_index))
        self.frame.after_idle(self._sync_left_panel_scrollregion)
    except Exception:
        pass

def _open_selected_project(self):
    selected_projects = self._get_selected_projects_from_list()
    selected = selected_projects[0] if selected_projects else ""
    if not selected:
        selected = self._ask_project_from_list(
            title="Otwórz projekt",
            action_label="Otwórz"
        )
    if not selected:
        return

    if not self._ensure_project_context_is_switchable(
        next_project=str(selected or "").strip(),
        dialog_title="Najpierw zakończ aktywną operację",
    ):
        return

    active_before = str(CAMPAIGN.get_active_project_name() or "").strip()

    open_started = perf_counter()
    selected_label = str(selected or "").strip()
    self._project_switch_in_progress = True

    def _show_open_overlay(message: str, *, progress: float | None = None, tone: str = "info") -> None:
        try:
            self._show_project_loading_overlay(
                title=f"Ładuję projekt {selected_label}" if selected_label else "Ładuję projekt",
                body=message,
                tone=tone,
                progress=progress,
            )
            try:
                self.frame.update()
            except Exception:
                try:
                    self.frame.update_idletasks()
                except Exception:
                    pass
        except Exception:
            pass

    _show_open_overlay(
        "Przygotowuję przełączenie projektu i zabezpieczam bieżący kontekst.",
        progress=None,
    )

    try:
        annotation_tab = self.app.tabs.get("annotation")
        if annotation_tab is not None and hasattr(annotation_tab, "capture_free_mode_snapshot_for_project_return"):
            annotation_tab.capture_free_mode_snapshot_for_project_return()
    except Exception as e:
        logger.debug(f"Nie udało się zapisać migawki free mode przed otwarciem projektu: {e}")

    self._cancel_deferred_project_open_tasks()
    if active_before and active_before != str(selected or "").strip():
        try:
            self._release_active_project_resources_before_switch(active_before)
        except Exception as e:
            logger.debug(f"Nie udało się zwolnić zasobów projektu '{active_before}' przed przełączeniem: {e}")
        _show_open_overlay(
            "Zwolniono poprzedni projekt. Odtwarzam stan wybranego projektu.",
            progress=18.0,
        )
    def _log_project_open_part(label: str, started_at: float, *, threshold_ms: float = 250.0) -> None:
        try:
            elapsed_ms = (perf_counter() - float(started_at)) * 1000.0
        except Exception:
            return
        if elapsed_ms >= float(threshold_ms):
            logger.info(f"[PROJECT PERF] {label}: {elapsed_ms:.0f} ms")

    part_started = perf_counter()
    _show_open_overlay(
        "Wczytuję stan projektu, iterację i aktywną bramkę.",
        progress=28.0,
    )
    CAMPAIGN.set_active_project(selected)
    try:
        self._reset_campaign_graph_runtime_state()
    except Exception:
        pass
    _log_project_open_part("set_active_project", part_started)
    self.app.campaign_free_mode = False
    part_started = perf_counter()
    _show_open_overlay(
        "Przełączam aplikację w tryb kampanii.",
        progress=38.0,
    )
    self.app.set_campaign_mode(True)
    _log_project_open_part("set_campaign_mode", part_started)
    self._project_open_lightweight_refresh = True
    part_started = perf_counter()
    _show_open_overlay(
        "Przygotowuję układ grafu i paneli projektu.",
        progress=46.0,
    )
    self._ensure_wizard_stage_ui_ready()
    _log_project_open_part("ensure_wizard_stage_ui_ready", part_started)

    def _finish_project_open_refresh():
        self._project_open_refresh_after_id = None
        refresh_failed = False
        try:
            if str(CAMPAIGN.get_active_project_name() or "").strip() != str(selected or "").strip():
                return
            try:
                _show_open_overlay(
                    "Buduję graf, zasoby i statusy bramek. Przy większym projekcie może to chwilę potrwać.",
                    progress=58.0,
                )
                self._refresh_dashboard()
            except Exception as e:
                refresh_failed = True
                logger.debug(f"Nie udało się odświeżyć dashboardu po otwarciu projektu: {e}")
                _show_open_overlay(
                    "Projekt został wybrany, ale nie udało się odświeżyć widoku grafu. Sprawdź log błędu i spróbuj odświeżyć widok.",
                    progress=100.0,
                    tone="error",
                )
            try:
                self._schedule_project_open_post_refresh(str(selected or "").strip())
            except Exception as e:
                logger.debug(f"Nie udało się zaplanować drugiego odświeżenia po otwarciu projektu: {e}")
            self._log_perf(
                "open_selected_project_refresh",
                open_started,
                threshold_ms=20.0,
                extra=f"project={selected}",
            )
        finally:
            self._project_open_lightweight_refresh = False
            self._project_switch_in_progress = False
            if refresh_failed:
                try:
                    self.frame.after(2600, self._hide_project_loading_overlay)
                except Exception:
                    pass

    try:
        self._project_open_refresh_after_id = self.frame.after(15, _finish_project_open_refresh)
    except Exception:
        try:
            _finish_project_open_refresh()
        finally:
            self._project_open_lightweight_refresh = False
            self._project_switch_in_progress = False

    try:
        self.app.update_status(
            f"Aktywowano projekt: {selected}. Trwa ładowanie kontekstu kampanii.",
            "info"
        )
    except Exception:
        pass

def _schedule_project_open_post_refresh(self, expected_project: str) -> None:
    try:
        pending = getattr(self, "_project_open_post_refresh_after_id", None)
        if pending:
            self.frame.after_cancel(pending)
    except Exception:
        pass

    def _run():
        self._project_open_post_refresh_after_id = None
        if str(CAMPAIGN.get_active_project_name() or "").strip() != str(expected_project or "").strip():
            return
        try:
            self.frame.after_idle(self._sync_right_panel_scrollregion)
        except Exception as e:
            logger.debug(f"Nie udalo sie zsynchronizowac panelu po otwarciu projektu: {e}")
        # Po _refresh_dashboard() graf i panele są już zbudowane. Nie wolno tu
        # stabilizować badge'y pełnym przeliczeniem, bo duże projekty (NEON)
        # płaciły wtedy za kolejną rekonstrukcję bramek, zasobów i zakładek.
        try:
            self.frame.after_idle(self._sync_right_panel_scrollregion)
        except Exception as e:
            logger.debug(f"Nie udało się zsynchronizować panelu po otwarciu projektu: {e}")

        def _late_badge_stabilize() -> None:
            if str(CAMPAIGN.get_active_project_name() or "").strip() != str(expected_project or "").strip():
                return
            try:
                self._sync_right_panel_scrollregion()
            except Exception as exc:
                logger.debug(f"Nie udalo sie wykonac poznego renderu grafu kampanii: {exc}")
            try:
                self._sync_right_panel_scrollregion()
            except Exception as exc:
                logger.debug(f"Nie udało się wykonać późnej synchronizacji panelu wizarda: {exc}")

        try:
            self.frame.after(420, _late_badge_stabilize)
        except Exception:
            pass

    try:
        self._project_open_post_refresh_after_id = self.frame.after(90, _run)
    except Exception:
        self._project_open_post_refresh_after_id = None

def _get_project_switch_blocker_message(self, *, current_project: str = "", next_project: str = "") -> str:
    issues: list[str] = []

    try:
        annotation_tab = self.app.tabs.get("annotation")
    except Exception:
        annotation_tab = None
    if annotation_tab is not None:
        if bool(getattr(annotation_tab, "is_processing", False)):
            issues.append("Z2 nadal wykonuje operację autoanotacji albo przygotowania runu.")
        elif bool(getattr(annotation_tab, "_campaign_project_restore_in_progress", False)):
            issues.append("Z2 nadal odtwarza kontekst projektu.")
        elif bool(getattr(annotation_tab, "_campaign_deferred_run_restore_in_progress", False)):
            issues.append("Z2 nadal doczytuje run kampanii.")
        elif bool(getattr(annotation_tab, "_campaign_step2_transition_in_progress", False)):
            issues.append("Z2 nadal kończy przejście kampanijne.")

    try:
        char_tab = self.app.tabs.get("characters")
    except Exception:
        char_tab = None
    if char_tab is not None:
        if bool(getattr(char_tab, "is_processing", False)):
            issues.append("Z3 nadal wykonuje operację znaków.")
        elif bool(getattr(char_tab, "fast_test_running", False)):
            issues.append("Z3 nadal wykonuje szybki test OCR.")

    try:
        train_tab = self.app.tabs.get("training")
    except Exception:
        train_tab = None
    if train_tab is not None:
        try:
            if bool(train_tab._step4_has_active_operation()):
                op_label = str(train_tab._get_active_step4_operation_label() or "operacja Z4").strip()
                issues.append(f"Z4 nadal wykonuje: {op_label}.")
        except Exception:
            if bool(getattr(train_tab, "is_processing", False)):
                issues.append("Z4 nadal wykonuje operację treningową.")

    if not issues:
        return ""

    current_label = str(current_project or "bieżący projekt").strip()
    next_label = str(next_project or "nowy projekt").strip()
    intro = (
        f"Nie mogę jeszcze przełączyć projektu z '{current_label}' na '{next_label}', "
        "bo poprzedni projekt nadal ma aktywne zadania:"
    )
    return intro + "\n\n- " + "\n- ".join(issues) + "\n\nZatrzymaj albo dokończ te działania i spróbuj ponownie."

def _release_active_project_resources_before_switch(self, current_project: str = "") -> None:
    current_label = str(current_project or "").strip()
    if current_label:
        logger.debug(f"Zwalniam zasoby aktywnego projektu przed przełączeniem: {current_label}")

    try:
        self._hide_project_loading_overlay()
    except Exception:
        pass

    self._clear_project_contexts(restore_free_mode_preview=False)

    try:
        self.frame.update_idletasks()
    except Exception:
        pass

def _delete_project(self):
    if bool(getattr(self, "_project_delete_in_progress", False)):
        try:
            self.app.update_status("Usuwanie projektu już trwa. Poczekaj na zakończenie operacji.", "info")
        except Exception:
            pass
        return

    selected_projects = self._get_selected_projects_from_list()
    if not selected_projects:
        selected = self._ask_project_from_list(
            title="Usuń projekt",
            action_label="Usuń"
        )
        selected_projects = [selected] if selected else []
    if not selected_projects:
        return

    if len(selected_projects) == 1:
        confirm_title = "Usuwanie projektu"
        confirm_message = (
            f"Usunąć projekt '{selected_projects[0]}' wraz z całym katalogiem projektu?"
        )
    else:
        confirm_title = "Usuwanie projektów"
        preview = ", ".join(selected_projects[:4])
        if len(selected_projects) > 4:
            preview += f" +{len(selected_projects) - 4} więcej"
        confirm_message = (
            f"Usunąć {len(selected_projects)} zaznaczone projekty wraz z ich katalogami?\n\n"
            f"{preview}"
        )

    if self.app.themed_confirm(
        confirm_title,
        confirm_message,
        parent=self.frame,
        confirm_label="Usuń",
        tone="warning"
    ):
        active_project = CAMPAIGN.get_active_project_name()
        if active_project and active_project in selected_projects:
            if not self._ensure_project_context_is_switchable(
                next_project="usunięcia aktywnego projektu",
                dialog_title="Najpierw zakończ aktywną operację",
            ):
                return
        projects_to_delete = [
            str(project_name or "").strip()
            for project_name in selected_projects
            if str(project_name or "").strip()
        ]
        if not projects_to_delete:
            return

        self._project_delete_in_progress = True
        self._project_switch_in_progress = True
        try:
            for attr_name in ("btn_open_proj", "btn_del_proj", "btn_exit_project"):
                widget = getattr(self, attr_name, None)
                if widget is not None:
                    widget.config(state="disabled")
        except Exception:
            pass

        if len(projects_to_delete) == 1:
            body = f"Usuwam katalog projektu '{projects_to_delete[0]}'. To może chwilę potrwać przy dużych datasetach."
        else:
            body = f"Usuwam {len(projects_to_delete)} projekty wraz z katalogami. To może chwilę potrwać przy dużych datasetach."
        try:
            self._show_project_loading_overlay(
                title="Usuwam projekt",
                body=body,
                tone="warning",
                progress=None,
            )
        except Exception:
            pass
        try:
            self.app.update_status("Usuwam projekt. Poczekaj na zakończenie operacji.", "info")
        except Exception:
            pass

        def _delete_worker() -> None:
            removed: list[str] = []
            failed: list[str] = []
            for project_name in projects_to_delete:
                try:
                    if CAMPAIGN.delete_project(project_name):
                        removed.append(project_name)
                    else:
                        failed.append(project_name)
                except Exception as exc:
                    failed.append(project_name)
                    logger.error(f"Nie udało się usunąć projektu {project_name}: {exc}")

            def _finish_delete() -> None:
                try:
                    if active_project in removed:
                        self._clear_project_contexts(restore_free_mode_preview=False)
                        self.app.campaign_free_mode = True
                        self.app.set_campaign_mode(False)

                    self._rebuild_wizard_stage_ui()
                    self._refresh_dashboard()
                    self.app.update_campaign_tab_access()

                    if removed:
                        if len(removed) == 1:
                            message = f"Projekt '{removed[0]}' został usunięty."
                        else:
                            message = f"Usunięto {len(removed)} projektów."
                        if failed:
                            message += f"\n\nNie udało się usunąć: {', '.join(failed)}."
                        self.app.themed_info(
                            "Usunięto",
                            message,
                            parent=self.frame,
                            tone=("warning" if failed else "success"),
                        )
                    elif failed:
                        self.app.themed_info(
                            "Nie usunięto projektu",
                            f"Nie udało się usunąć: {', '.join(failed)}.",
                            parent=self.frame,
                            tone="warning",
                        )
                finally:
                    self._project_delete_in_progress = False
                    self._project_switch_in_progress = False
                    try:
                        self._hide_project_loading_overlay()
                    except Exception:
                        pass
                    try:
                        self._refresh_dashboard()
                    except Exception:
                        pass

            try:
                self.frame.after(0, _finish_delete)
            except Exception:
                _finish_delete()

        threading.Thread(target=_delete_worker, daemon=True, name="CampaignProjectDelete").start()
        return

def _clear_project_contexts(
    self,
    *,
    restore_free_mode_preview: bool = True,
    hide_loading_overlay: bool = True,
    lightweight_tab_clear: bool = False,
):
    if hide_loading_overlay:
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
    try:
        self._reset_campaign_graph_runtime_state()
    except Exception:
        pass
    tab_labels = {
        "annotation": "Autoanotacji",
        "characters": "Zakładki Znaków",
        "training": "Treningu",
    }

    for tab_key, label in tab_labels.items():
        try:
            if tab_key in self.app.tabs:
                tab = self.app.tabs[tab_key]
                try:
                    setattr(tab, "_campaign_context_project_name", "")
                    setattr(tab, "_campaign_graph_entry_context", {})
                except Exception:
                    pass
                if lightweight_tab_clear:
                    try:
                        setattr(tab, "_campaign_lightweight_context_detached", True)
                    except Exception:
                        pass
                    logger.debug(f"Lekko odpinam kontekst {label} bez czyszczenia niewidocznego UI.")
                    continue
                clear_context = getattr(tab, "clear_campaign_context", None)
                if clear_context is None:
                    continue
                try:
                    clear_context(restore_free_mode_preview=restore_free_mode_preview)
                except TypeError as exc:
                    if "restore_free_mode_preview" not in str(exc):
                        raise
                    clear_context()
        except Exception as e:
            logger.debug(f"Nie udało się wyczyścić kontekstu {label}: {e}")

def _exit_project_mode(self):
    active = CAMPAIGN.get_active_project_name()
    if not active:
        return

    if not self._ensure_project_context_is_switchable(
        next_project="trybu swobodnego",
        dialog_title="Najpierw zakończ aktywną operację",
    ):
        return

    if self.app.themed_confirm(
        "Wyjście z projektu",
        f"Czy na pewno chcesz opuścić projekt '{active}' i przejść do trybu swobodnego?\n\n"
        "Projekt nie zostanie usunięty.",
        parent=self.frame,
        confirm_label="Wyjdź",
        tone="warning"
    ):
        exit_started = perf_counter()
        phase_started = exit_started

        def _mark_exit_phase(name: str) -> None:
            nonlocal phase_started
            try:
                now = perf_counter()
                elapsed_ms = (now - phase_started) * 1000.0
                total_ms = (now - exit_started) * 1000.0
                if elapsed_ms >= 200.0 or total_ms >= 1000.0:
                    logger.info(
                        "[PROJECT EXIT PERF] "
                        f"{name}={elapsed_ms:.0f}ms total={total_ms:.0f}ms project={active}"
                    )
                phase_started = now
            except Exception:
                pass

        try:
            self._show_project_loading_overlay(
                title="Wychodzę z projektu",
                body="Czyszczę kontekst kampanii bez odtwarzania ciężkich podglądów trybu swobodnego.",
                tone="info",
                progress=None,
            )
        except Exception:
            pass

        # użytkownik ręcznie wymusza tryb swobodny
        previous_lightweight_refresh = bool(getattr(self, "_project_open_lightweight_refresh", False))
        self._project_exit_in_progress = True
        self._project_switch_in_progress = True
        self._project_open_lightweight_refresh = True
        try:
            self._cancel_deferred_project_open_tasks()
        except Exception:
            pass
        _mark_exit_phase("cancel_deferred_open_tasks")

        self.app.campaign_free_mode = True

        # czyścimy aktywny projekt
        CAMPAIGN.clear_active_project()
        _mark_exit_phase("clear_active_project")

        # wyłączamy tryb kampanii
        self.app.set_campaign_mode(False)
        _mark_exit_phase("set_campaign_mode")

        # odświeżamy dashboard
        self._rebuild_wizard_stage_ui()
        _mark_exit_phase("rebuild_wizard")
        self._refresh_dashboard()
        _mark_exit_phase("refresh_dashboard")

        # finalna synchronizacja dostępności zakładek
        self.app.update_campaign_tab_access()
        _mark_exit_phase("tab_access")

        try:
            self.app.open_controlled_tab("campaign")
            self.app.root.update_idletasks()
            _mark_exit_phase("open_campaign_tab")
        except Exception as e:
            logger.debug(f"Nie udało się przełączyć na główne okno po wyjściu z projektu: {e}")

        # czyścimy projektowy kontekst innych zakładek dopiero po przejściu do Z1,
        # żeby użytkownik nie widział chwilowego odtwarzania kafli workflow Z2.
        self._clear_project_contexts(
            restore_free_mode_preview=False,
            hide_loading_overlay=False,
            lightweight_tab_clear=True,
        )
        _mark_exit_phase("clear_contexts_light")

        try:
            self.app.update_status(
                "Opuściłeś aktywny projekt. Aplikacja działa teraz w trybie swobodnym.",
                "info"
            )
        except Exception:
            pass
        try:
            self._hide_project_loading_overlay()
        except Exception:
            pass
        self._project_open_lightweight_refresh = previous_lightweight_refresh
        self._project_exit_in_progress = False
        self._project_switch_in_progress = False
# ======================================================
# STEPS
# ======================================================

def _build_projects_browser(self, parent):
    palette = getattr(self.app, "palette", {})

    browser_lf = ttk.LabelFrame(parent, text=" Zapisane projekty ", padding=10)
    browser_lf.pack(fill=tk.X, pady=(0, 12))
    self.project_browser_frame = browser_lf

    status_panel = tk.Frame(
        browser_lf,
        bg=palette.get("panel", "#252526")
    )
    status_panel.pack(fill=tk.X, pady=(0, 6))
    self.project_list_status_lbl = status_panel
    self.project_list_status_labels = []
    self.project_status_top_row = tk.Frame(
        status_panel,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    self.project_status_top_row.pack(fill=tk.X)

    status_fonts = [
        ("Segoe UI", 10, "bold"),
        ("Segoe UI", 9),
        ("Segoe UI", 9),
    ]

    first_lbl = tk.Label(
        self.project_status_top_row,
        text="",
        justify=tk.LEFT,
        anchor="w",
        height=1,
        fg=palette.get("muted", "#b8b8b8"),
        bg=palette.get("panel", "#252526"),
        font=status_fonts[0]
    )
    first_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.project_list_status_labels.append(first_lbl)

    self.project_add_button_canvas = tk.Canvas(
        status_panel,
        width=1,
        height=38,
        bd=0,
        highlightthickness=0,
        bg=palette.get("panel", "#252526"),
        cursor="hand2",
    )
    self.project_add_button_canvas.pack(fill=tk.X, pady=(4, 2))
    HELP.bind_help(self.project_add_button_canvas, "camp_new_project")
    self._bind_icon_button(self.project_add_button_canvas, role="project_add", command=self._add_new_project)
    self.frame.after_idle(lambda: self._draw_icon_button("project_add"))

    from .project_attachment_dialog import show_project_attachment_dialog
    self.project_attach_button = ttk.Button(
        status_panel, text="Podłącz istniejący projekt…",
        command=lambda: show_project_attachment_dialog(self),
    )
    self.project_attach_button.pack(fill=tk.X, pady=(2, 8))

    for font_spec in status_fonts[1:]:
        lbl = tk.Label(
            status_panel,
            text="",
            justify=tk.LEFT,
            anchor="w",
            height=1,
            fg=palette.get("muted", "#b8b8b8"),
            bg=palette.get("panel", "#252526"),
            font=font_spec
        )
        lbl.pack(fill=tk.X)
        self.project_list_status_labels.append(lbl)

    list_host = tk.Frame(
        browser_lf,
        bg=palette.get("field", "#1a1a1a"),
        bd=0,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        highlightcolor=palette.get("panel_border", palette.get("border", "#3c3c3c"))
    )
    list_host.pack(fill=tk.X, expand=False)
    self.project_list_host = list_host

    scroll = WebSlimScrollbar(list_host, orient=tk.VERTICAL)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.project_list_scrollbar = scroll
    try:
        green = self._get_campaign_green_accent()
        scroll.configure_style(
            track_color=palette.get("field", "#1a1a1a"),
            thumb_color=green,
            thumb_hover_color=blend_hex_colors(green, "#ffffff", 0.18),
        )
    except Exception:
        pass

    self.project_listbox = tk.Listbox(
        list_host,
        exportselection=False,
        selectmode=tk.EXTENDED,
        height=5,
        width=1,
        font=("Segoe UI", 10),
        bg=palette.get("field", "#1a1a1a"),
        fg=palette.get("fg", "#f3f3f3"),
        selectbackground=self._get_project_list_selection_bg(),
        selectforeground=self._get_project_list_selection_fg(),
        activestyle="none",
        bd=0,
        highlightthickness=1,
        highlightbackground=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        highlightcolor=palette.get("accent", "#2980b9"),
        takefocus=1,
        yscrollcommand=scroll.set
    )
    self.project_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self.project_listbox.bind("<<ListboxSelect>>", self._on_project_changed)
    self.project_listbox.bind("<Double-Button-1>", lambda _e: self._open_selected_project())
    self.project_listbox.bind("<Button-1>", lambda _e: self.project_listbox.focus_set(), add="+")
    self.project_listbox.bind("<Button-3>", self._show_project_context_menu)
    self.project_listbox.bind("<MouseWheel>", lambda e: self._on_listbox_mousewheel(e, self.project_listbox))
    self.project_listbox.bind("<Button-4>", lambda e: self._on_listbox_mousewheel(e, self.project_listbox))
    self.project_listbox.bind("<Button-5>", lambda e: self._on_listbox_mousewheel(e, self.project_listbox))
    scroll.config(command=self.project_listbox.yview)

    self.project_context_menu = tk.Menu(self.frame, tearoff=0)
    repair_text = campaign_ui_helpers._repair_polish_text
    self.project_context_menu.add_command(label=repair_text("OtwĂłrz projekt"), command=self._open_selected_project)
    self.project_context_menu.add_separator()
    self.project_context_menu.add_command(label=repair_text("UsuĹ„ zaznaczone projekty"), command=self._delete_project)

    self.project_browser_footer = None

    HELP.bind_help(browser_lf, "camp_open_project")
    HELP.bind_help(status_panel, "camp_open_project")
    for lbl in self.project_list_status_labels:
        HELP.bind_help(lbl, "camp_open_project")
    HELP.bind_help(self.project_listbox, "camp_open_project")
