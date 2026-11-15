# -*- coding: utf-8 -*-
"""Operator guidance for copying a previously exported ALPR package to a phone.

Read-only presentation. No Android approval, import, model mutation, or export.
A historic Desktop PASS counts only when the library has re-hashed current bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .z4_mobile_artifact_validation_history import SCHEMA, SCOPE, history_key


@dataclass(frozen=True)
class TransferGuidance:
    label: str
    explanation: str
    next_step: str
    copy_ready: bool


_BLOCKING = frozenset({
    "MANIFEST_ERROR", "READ_ERROR", "REGISTERED_SHA_MISMATCH",
    "REGISTERED_SHA_UNREADABLE", "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED",
    "RESEARCH_SHA_MISMATCH", "RESEARCH_ONLY_NOT_PACKAGED",
})


def transfer_guidance(row, *, desktop_validated=False, validation_failed=False,
                      validation_running=False, history_entry=None) -> TransferGuidance:
    """Status is for *manual copying only*, never Android-import authorization.

    History MATCH is provided by the read-only SHA comparison performed on scan.
    Never infer a pass from a file name, extension, mtime or export event alone.
    """
    if row is None:
        return TransferGuidance("Wybierz paczkę", "Nie wybrano pliku.",
                                "Zaznacz plik .alprmodel w tabeli.", False)
    kind = str(getattr(row, "kind", ""))
    state = str(getattr(row, "state", ""))
    name = str(getattr(row, "path", "") or "")
    error = str(getattr(row, "error", "") or "").strip()
    if kind not in {"MODEL", "PACKAGE"} or Path(name).suffix.lower() != ".alprmodel":
        return TransferGuidance(
            "Nie jest paczką na telefon", "Wybrany plik nie jest obsługiwaną paczką .alprmodel.",
            "Wybierz gotową paczkę .alprmodel z biblioteki.", False)
    if state in _BLOCKING or error or state not in {
        "DISCOVERED_NOT_VERIFIED", "REGISTERED_SHA_MATCH_NOT_VERIFIED",
    }:
        return TransferGuidance(
            "Nie kopiuj — problem z paczką", "Plik ma błąd odczytu, metadanych lub zgodności SHA.",
            "Sprawdź przyczynę w szczegółach i wybierz poprawny pakiet.", False)
    if validation_running:
        return TransferGuidance(
            "Trwa kontrola na PC", "Program sprawdza zawartość paczki na komputerze.",
            "Poczekaj na wynik kontroli.", False)
    if validation_failed:
        return TransferGuidance(
            "Nie kopiuj — kontrola niezaliczona", "Walidator Desktop zgłosił problem.",
            "Przeczytaj przyczynę i nie używaj paczki przed wyjaśnieniem błędu.", False)

    observation = history_entry if isinstance(history_entry, dict) else {}
    record = observation.get("record") or {}
    comparison = str(observation.get("comparison") or "")
    if comparison == "CHANGED":
        return TransferGuidance(
            "Nie kopiuj — plik zmieniony", "Zawartość różni się od sprawdzonej wcześniej wersji.",
            "Uruchom pełną kontrolę ponownie; stary wynik jest nieważny.", False)
    if comparison == "UNREADABLE":
        return TransferGuidance(
            "Nie kopiuj — brak kontroli SHA", "Nie można potwierdzić tożsamości pliku.",
            "Sprawdź dostęp do pliku i powtórz kontrolę.", False)
    if not desktop_validated and record.get("outcome") == "FAILED":
        return TransferGuidance(
            "Nie kopiuj — kontrola niezaliczona", "Poprzednia kontrola techniczna zakończyła się błędem.",
            "Sprawdź przyczynę i ponów pełną kontrolę.", False)

    valid_history = (
        comparison == "MATCH" and record.get("outcome") == "PASSED"
        and record.get("schema") == SCHEMA
        and record.get("scope") == SCOPE
        and record.get("kind") == kind
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or "")))
        and history_key(record.get("path", "")) == history_key(name)
        and record.get("android_verified") is False
        and record.get("experiment_verified") is False
    )
    if desktop_validated or valid_history:
        if kind == "PACKAGE":
            return TransferGuidance(
                "Gotowa do skopiowania", "Kompletny pakiet .alprmodel przeszedł kontrolę techniczną PC. "
                "To nie potwierdza jeszcze możliwości importu ani działania na Androidzie.",
                "Kliknij «Pokaż wybrany plik w folderze». Eksplorator zaznaczy paczkę; "
                "skopiuj ją na telefon, a następnie sprawdź import i działanie w ALPR Mobile.", True)
        return TransferGuidance(
            "Model PC OK — niepełny ALPR", "Pojedynczy model przeszedł kontrolę PC, "
            "ale do kompletnego ALPR potrzebna jest para MT+MZ.",
            "Przygotuj lub wybierz pełny pakiet MT+MZ do przeniesienia na telefon.", False)

    return TransferGuidance(
        "Najpierw kontrola PC", "Program znalazł paczkę, ale nie potwierdził jeszcze pełnej "
        "poprawności jej modeli i precyzji.",
        "Zaznacz pakiet i kliknij «Sprawdź paczkę (pełna kontrola)» pod listą. "
        "Jeśli kontrola będzie zaliczona, będzie można skopiować plik na telefon.", False)
