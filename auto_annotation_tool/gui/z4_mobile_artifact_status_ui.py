# -*- coding: utf-8 -*-
"""Read-only Polish explanations of library states; NOT new approval states.

The registry's machine codes, saved evidence and export decisions are immutable.
A Desktop PASS describes only the checked bytes at the moment of validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StatusGuidance:
    label: str
    meaning: str
    next_step: str


FOUND = StatusGuidance(
    "Nie sprawdzono paczki",
    "Program znalazł plik i odczytał jego opis, ale nie sprawdził jeszcze kompletności paczki, "
    "sum kontrolnych jej zawartości ani rzeczywistej precyzji modeli. Taka kontrola jest potrzebna, "
    "aby przed testami na telefonie wykryć uszkodzoną lub nieprawidłowo opisaną paczkę.",
    "W zakładce «Pakiety i modele» zaznacz paczkę na liście, a następnie na dole okna kliknij "
    "«Sprawdź paczkę (pełna kontrola)». Program tylko odczyta plik; nie rozpocznie testu na telefonie.",
)
INCOMPLETE = StatusGuidance(
    "Niepełny opis paczki",
    "Program odczytał część informacji o modelach, ale nie potwierdził całego opisu pakietu. "
    "Nie wiadomo jeszcze, czy paczka nadaje się do następnego etapu.",
    "W zakładce «Pakiety i modele» zaznacz paczkę, przeczytaj przyczynę w szczegółach i "
    "na dole okna kliknij «Sprawdź paczkę (pełna kontrola)». Jeśli kontrola nie przejdzie, "
    "nie używaj tej paczki w eksperymencie.",
)
UNREADABLE = StatusGuidance(
    "Problem z paczką",
    "Program nie zdołał poprawnie odczytać pliku lub jego wymaganego opisu.",
    "Sprawdź przyczynę w szczegółach. Nie używaj paczki, dopóki problem nie zostanie usunięty.",
)
HISTORY_MATCH = StatusGuidance(
    "Zgodny z historią eksportu",
    "Suma kontrolna pliku zgadza się z zapisem wcześniejszego eksportu, ale nie oznacza to "
    "jeszcze sprawdzenia kompletności i precyzji modeli.",
    "Przed użyciem w eksperymencie zaznacz paczkę w zakładce «Pakiety i modele» i "
    "na dole okna kliknij «Sprawdź paczkę (pełna kontrola)». To nie uruchamia testu na telefonie.",
)
HISTORY_MISMATCH = StatusGuidance(
    "Niezgodny z historią",
    "Obecny plik różni się od wersji zapisanej wcześniej w historii eksportu.",
    "Nie używaj paczki jako wcześniejszego wyniku. Sprawdź pochodzenie pliku i jego SHA-256.",
)
HISTORY_UNREADABLE = StatusGuidance(
    "Nie można sprawdzić pliku",
    "Nie udało się porównać pliku z zapisem wcześniejszego eksportu.",
    "Sprawdź dostęp do pliku. Nie zakładaj, że jego zawartość jest zgodna.",
)
RESEARCH = StatusGuidance(
    "Plik badawczy — nie paczka",
    "To pomocniczy plik TFLite, a nie gotowa paczka .alprmodel do importu.",
    "Wybierz kompletny pakiet .alprmodel. Tego pliku nie oznaczaj jako sprawdzonego na telefonie.",
)
RESEARCH_CHANGED = StatusGuidance(
    "Plik badawczy niezgodny",
    "Suma kontrolna lub rozmiar pliku badawczego odbiega od oczekiwanej wersji.",
    "Nie używaj go jako zweryfikowanego wariantu eksperymentalnego.",
)
UNKNOWN = StatusGuidance(
    "Stan wymaga wyjaśnienia",
    "Biblioteka napotkała status, którego nie potrafi jednoznacznie opisać.",
    "Sprawdź kod stanu i szczegóły pliku. Nie traktuj go jako zatwierdzonego.",
)
DESKTOP_PASS = StatusGuidance(
    "Kontrola na PC zaliczona",
    "Walidator Desktop potwierdził archiwum, zgodność manifestów, plików i deklarowanej precyzji grafów w chwili kontroli. To nie jest test na telefonie.",
    "Sprawdź zgodność modeli z planem eksperymentu, a następnie wykonaj osobny test w ALPR Mobile.",
)
DESKTOP_FAIL = StatusGuidance(
    "Kontrola na PC niezaliczona",
    "Walidator Desktop wykrył problem. Paczka nie przeszła pełnej kontroli technicznej.",
    "Odczytaj przyczynę poniżej. Nie wykorzystuj tej paczki do testu, dopóki błąd nie zostanie wyjaśniony.",
)
STATES = {
    "DISCOVERED_NOT_VERIFIED": FOUND,
    "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED": INCOMPLETE,
    "MANIFEST_ERROR": UNREADABLE,
    "READ_ERROR": UNREADABLE,
    "REGISTERED_SHA_MATCH_NOT_VERIFIED": HISTORY_MATCH,
    "REGISTERED_SHA_MISMATCH": HISTORY_MISMATCH,
    "REGISTERED_SHA_UNREADABLE": HISTORY_UNREADABLE,
    "RESEARCH_ONLY_NOT_PACKAGED": RESEARCH,
    "RESEARCH_SHA_MISMATCH": RESEARCH_CHANGED,
}
# A local graph PASS cannot erase a historical provenance conflict.
PROVENANCE_BLOCKERS = frozenset({
    "MANIFEST_ERROR", "READ_ERROR", "REGISTERED_SHA_MISMATCH",
    "REGISTERED_SHA_UNREADABLE", "RESEARCH_SHA_MISMATCH",
    "RESEARCH_ONLY_NOT_PACKAGED",
})


def artifact_status_guidance(row, *, desktop_validated=False, validation_failed=False) -> StatusGuidance:
    state = str(getattr(row, "state", "") or "")
    base = STATES.get(state, UNKNOWN)
    if state in PROVENANCE_BLOCKERS:
        return base
    if validation_failed:
        return DESKTOP_FAIL
    if desktop_validated:
        return DESKTOP_PASS
    return base


def artifact_status_label(row, *, desktop_validated=False, validation_failed=False,
                          validation_running=False) -> str:
    if validation_running:
        return "Trwa kontrola na PC"
    return artifact_status_guidance(
        row, desktop_validated=desktop_validated, validation_failed=validation_failed
    ).label


def validation_result_text(*, ok: bool, path: str, sha256: str = "", problem: str = "") -> str:
    filename = Path(path).name
    if ok:
        return ("KONTROLA TECHNICZNA NA KOMPUTERZE: ZALICZONA\n\n"
                f"Plik: {filename}\nSHA-256 pliku podczas kontroli: {sha256}\n\n"
                "Sprawdzono strukturę paczki, sumy plików, manifesty, role MT/MZ "
                "i zgodność deklarowanej precyzji z grafami.\n"
                "Nie potwierdzono jeszcze uruchomienia w ALPR Mobile ani jakości rozpoznawania.\n\n"
                "NASTĘPNY KROK: porównaj modele źródłowe z planem eksperymentu, "
                "a następnie sprawdź tę parę na telefonie. "
                "Przed ponownym użyciem program musi zweryfikować plik ponownie.")
    return ("KONTROLA TECHNICZNA NA KOMPUTERZE: NIEZALICZONA\n\n"
            f"Plik: {filename}\nPrzyczyna: {str(problem or 'brak szczegółów')[:800]}\n\n"
            "NASTĘPNY KROK: ustal przyczynę niezgodności lub wybierz inny pakiet. "
            "Nie używaj tej paczki w eksperymencie bez rozwiązania problemu.")
