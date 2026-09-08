#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign graph assistant context and focus helpers."""

from ..campaign_manager import CAMPAIGN
from .campaign_models import WizardStageStatus
from .campaign_ui_helpers import _repair_polish_text


def _repair_assistant_context(value):
    if isinstance(value, str):
        return _repair_polish_text(value)
    if isinstance(value, tuple):
        return tuple(_repair_assistant_context(item) for item in value)
    if isinstance(value, list):
        return [_repair_assistant_context(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_assistant_context(item) for key, item in value.items()}
    return value


def _build_campaign_graph_assistant_context(self, status: WizardStageStatus | None = None) -> dict:
    try:
        step_num = int(getattr(status, "step_num", None) or CAMPAIGN.get_current_step() or 1)
    except Exception:
        step_num = 1
    step_num = min(max(step_num, 1), 4)
    stage_title = str(getattr(status, "title", "") or f"Etap E{step_num}").strip()
    state = str(getattr(status, "state", "") or "").strip().lower()
    if state in {"ready", "completed", "approved"}:
        state_text = "bramka mo\u017ce by\u0107 gotowa do zatwierdzenia, ale decyzj\u0119 podejmuj na aktywnej kraw\u0119dzi grafu."
    elif state in {"needs_attention", "blocked"}:
        state_text = "aktywny etap wymaga uzupe\u0142nienia zasob\u00f3w albo wykonania akcji roboczej."
    else:
        state_text = "pracuj na jednej wybranej bramce i nie mieszaj zasob\u00f3w mi\u0119dzy \u015bcie\u017ckami."

    step3_workflow = ()
    step3_glossary = ()
    step3_caution = ""
    if step_num == 3:
        step3_workflow = (
            "Dla T05 praca ma dwa kroki: PZ2 buduje bazę znaków na wyodrębnionych tablicach, a PZ3 eksportuje źródłowy dataset znaków.",
            "Próg w PZ2 mówi tylko, czy baza znaków jest sensowna. Bramkę T05 domyka dopiero artefakt AZ utworzony w PZ3.",
        )
        step3_glossary = (
            "baza PZ2 = tablice perfect i ramki znaków przygotowane do eksportu",
            "AZ = źródłowy dataset znaków utworzony w PZ3",
        )
        step3_caution = "W E3 nie traktuj progu PZ2 jako pełnego otwarcia T05: to tylko pierwszy składnik pracy."

    return _repair_assistant_context({
        "location": f"[Z1] Mapa przej\u015b\u0107 kampanii / E{step_num}",
        "goal": (
            "Z1 działa jak mapa przejść kampanii. Węzły E1, E2, E3, E4T i E4Z są etapami, "
            "a bramki przy kraw\u0119dziach opisuj\u0105 konkretne przej\u015bcia mi\u0119dzy etapami."
        ),
        "workflow": (
            "Najpierw wybierz jedn\u0105 bramk\u0119 elektrod\u0105 przy jej etykiecie. Dopiero wybrana bramka jest aktywn\u0105 \u015bcie\u017ck\u0105 pracy.",
            "Pole Zasoby otwiera wymagane wej\u015bcia dla tej jednej \u015bcie\u017cki: obrazy, modele albo anotacje.",
            "Pole Praca prowadzi do właściwej karty roboczej, np. Z2, Z3 albo Z4. Po wyjściu wracasz do mapy przejść.",
            "Pole Zatwierd\u017a zamyka przej\u015bcie dopiero wtedy, gdy bramka jest otwarta i warunki s\u0105 spe\u0142nione.",
            "Jeśli wybierasz ścieżkę startową iteracji, porównuj T01 i T02 jako alternatywy.",
            *step3_workflow,
        ),
        "current": (
            f"Aktualny punkt odniesienia: {stage_title}. "
            f"Stan roboczy: {state_text}"
        ),
        "glossary": (
            "graf = widok ca\u0142ej kampanii jako mapy przej\u015b\u0107",
            "w\u0119ze\u0142 = etap projektu, np. E1, E2, E3, E4T albo E4Z",
            "E4T = trening modelu tablic; E4Z = trening modelu znaków",
            "kraw\u0119d\u017a = mo\u017cliwe przej\u015bcie mi\u0119dzy etapami",
            "bramka = mały panel na krawędzi z polami Bramka, Zasoby, Praca i Zatwierdź",
            "elektroda = prze\u0142\u0105cznik wyboru bramki; bez niej pola bramki pozostaj\u0105 pasywne",
            "zasoby = dane wymagane przez wybran\u0105 \u015bcie\u017ck\u0119, np. katalog zdj\u0119\u0107, model albo anotacje",
            "akcje = operacje dostępne w polu Praca",
            "zatwierd\u017a = formalne zamkni\u0119cie przej\u015bcia i przesuni\u0119cie kampanii dalej",
            "T01 = E1 -> E2, praca od obrazów i anotacji tablic",
            "T02 = E1 -> E3, skrót do pracy nad znakami na dostępnych tablicach",
            "T03 = E2 -> E3, przekazanie zatwierdzonych tablic do znaków",
            "T04 = E2 -> E4T, dataset i trening modelu tablic",
            "T05 = E3 -> E4Z, dataset i trening modelu znaków",
            "T06 = E4T/E4Z -> E1, domknięcie iteracji",
            "kontrakt zasobu = odpowiedź, czy zasób jest spełniony, do kontroli albo brakujący",
            "przyrost iteracji = zatwierdzony materiał wytworzony w bieżącym cyklu",
            *step3_glossary,
        ),
        "caution": (
            "Je\u015bli kilka bramek wychodzi z tego samego etapu, najpierw wybierz elektrod\u0105 t\u0119, kt\u00f3r\u0105 realnie chcesz prowadzi\u0107. "
            "Nie mieszaj zasob\u00f3w mi\u0119dzy r\u00f3wnoleg\u0142ymi \u015bcie\u017ckami."
            + (f" {step3_caution}" if step3_caution else "")
        ),
        "references": ("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
    })


def get_free_mode_assistant_context(self) -> dict:
    active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
    if not active_project:
        return _repair_assistant_context({
            "location": "[Z1] Projekty i mapa przej\u015b\u0107",
            "goal": "Wybierz istniej\u0105cy projekt albo utw\u00f3rz nowy. Po otwarciu projektu zobaczysz graf przej\u015b\u0107 E1, E2, E3 oraz E4T/E4Z.",
            "current": "Projekt kampanii przechowuje iteracje, wybrane ścieżki, zasoby, wyniki bramek i ślad pracy.",
            "workflow": (
                "Otw\u00f3rz projekt z listy albo rozpocznij nowy projekt.",
                "Po otwarciu projektu AS opisuje aktywną bramkę, wymagane zasoby, przyrost iteracji i dalszą pracę.",
            ),
            "glossary": (
                "Z1 = panel projektu i mapa przej\u015b\u0107 kampanii",
                "kampania = projekt prowadzony etapami",
                "graf = widok przej\u015b\u0107 mi\u0119dzy etapami E1, E2, E3, E4T i E4Z",
                "bramka = panel przy krawędzi grafu, który zbiera zasoby, pracę i zatwierdzenie",
                "ślad projektu = zapis kolejnych decyzji i wyników iteracji",
            ),
            "caution": "AS jest pasywn\u0105 podpowiedzi\u0105. Nie wykonuje akcji i nie zmienia stanu projektu.",
            "references": ("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
        })

    status = self._get_wizard_assistant_stage_status()
    return _build_campaign_graph_assistant_context(self, status)


def _wizard_stage_key_from_step_num(step_num: int | None) -> str:
    try:
        normalized = int(step_num or 1)
    except Exception:
        normalized = 1
    normalized = min(max(normalized, 1), 4)
    return f"step{normalized}"

def request_wizard_stage_focus(self, step_num: int | None = None, *, stage_key: str | None = None) -> None:
    key = str(stage_key or "").strip().lower()
    if not key:
        key = self._wizard_stage_key_from_step_num(step_num)
    if key not in {"step1", "step2", "step3", "step4"}:
        return
    self._wizard_focus_stage_request = key

def _set_wizard_assistant_stage_context(self, stage_key: str) -> None:
    key = str(stage_key or "").strip().lower()
    if key not in {"step1", "step2", "step3", "step4"}:
        return
    if str(getattr(self, "_wizard_assistant_stage_key", "") or "").strip().lower() == key:
        return
    self._wizard_assistant_stage_key = key
    try:
        notify = getattr(self.app, "notify_free_mode_assistant_context_changed", None)
        if callable(notify):
            notify()
    except Exception:
        pass

def _get_wizard_assistant_stage_status(self) -> WizardStageStatus | None:
    statuses: list[WizardStageStatus] = []
    for status in list(getattr(self, "_wizard_header_metro_statuses", []) or []):
        if isinstance(status, WizardStageStatus) and bool(getattr(status, "visible", True)):
            statuses.append(status)

    if not statuses:
        for card in dict(getattr(self, "wizard_stage_cards", {}) or {}).values():
            if not isinstance(card, dict):
                continue
            status = card.get("status")
            if isinstance(status, WizardStageStatus) and bool(getattr(status, "visible", True)):
                statuses.append(status)

    if not statuses:
        return None

    requested_key = str(getattr(self, "_wizard_assistant_stage_key", "") or "").strip().lower()
    if requested_key:
        for status in statuses:
            if str(getattr(status, "key", "") or "").strip().lower() == requested_key:
                return status

    for status in statuses:
        if bool(getattr(status, "is_current", False)):
            return status

    for wanted_state in ("needs_attention", "ready", "in_progress"):
        for status in statuses:
            if str(getattr(status, "state", "") or "").strip().lower() == wanted_state:
                return status

    return statuses[0]

def _wizard_state_assistant_label(state: str) -> str:
    state_key = str(state or "").strip().lower()
    labels = {
        "locked": "zablokowana",
        "in_progress": "w toku",
        "needs_attention": "wymaga uwagi",
        "ready": "gotowa do decyzji",
        "done": "zakończona",
        "skipped": "pominięta",
    }
    return labels.get(state_key, state_key or "nieznany")

def _is_step3_z2_repair_status(status: WizardStageStatus) -> bool:
    if str(getattr(status, "key", "") or "").strip().lower() != "step3":
        return False
    haystack = " ".join(
        str(value or "")
        for value in (
            getattr(status, "title", ""),
            getattr(status, "summary", ""),
            getattr(status, "details", ""),
            getattr(status, "primary_label", ""),
            getattr(status, "secondary_label", ""),
        )
    ).casefold()
    return bool("z2" in haystack and ("więcej tablic" in haystack or "popraw tablice" in haystack or "przygotuj więcej tablic" in haystack))

def _build_wizard_stage_assistant_context(self, status: WizardStageStatus) -> dict:
    return _build_campaign_graph_assistant_context(self, status)

def _cancel_pending_wizard_stage_focus(self) -> None:
    pending = getattr(self, "_wizard_focus_after_id", None)
    if not pending:
        return
    try:
        self.frame.after_cancel(pending)
    except Exception:
        pass
    self._wizard_focus_after_id = None

def _schedule_pending_wizard_stage_focus(self, *, attempts_left: int = 4) -> None:
    self._cancel_pending_wizard_stage_focus()
    if not str(getattr(self, "_wizard_focus_stage_request", "") or "").strip():
        return

    def _run():
        self._wizard_focus_after_id = None
        applied = self._apply_pending_wizard_stage_focus()
        if applied:
            self._wizard_focus_stage_request = ""
            return
        if attempts_left > 1:
            try:
                self._wizard_focus_after_id = self.frame.after(
                    60,
                    lambda: self._schedule_pending_wizard_stage_focus(attempts_left=attempts_left - 1),
                )
            except Exception:
                self._wizard_focus_after_id = None

    try:
        self._wizard_focus_after_id = self.frame.after_idle(_run)
    except Exception:
        self._wizard_focus_after_id = None

def _apply_pending_wizard_stage_focus(self) -> bool:
    stage_key = str(getattr(self, "_wizard_focus_stage_request", "") or "").strip().lower()
    if stage_key not in {"step1", "step2", "step3", "step4"}:
        return False
    self._set_wizard_assistant_stage_context(stage_key)
    try:
        refresh_graph = getattr(self, "_refresh_wizard_transition_graph", None)
        if callable(refresh_graph):
            refresh_graph()
    except Exception:
        pass
    return True

