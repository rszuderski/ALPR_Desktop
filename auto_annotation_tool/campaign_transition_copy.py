"""User-facing copy generated from campaign transition specs.

The graph should explain the active transition, not rephrase old wizard panels.
This module keeps that language close to the state-machine contract.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .campaign_transition_resource_report import TransitionResourceReport
from .campaign_transition_specs import CampaignTransitionSpec


TRANSITION_SUMMARY_OVERRIDES = {
    "e1_to_e2_prepare_plate_annotations": (
        "Wspólna bramka wejścia do E2/Z2. W torze tablic potrzebuje nowej puli obrazów, "
        "a w torze znaków może pracować także na istniejącym źródle tablic, jeśli obrazy "
        "zostały już wykorzystane, ale materiał tablic nadal pozwala budować anotacje znaków."
    ),
    "e1_to_e3_char_from_ready_plates": (
        "Skrót dla sytuacji, w której masz już zgodne i zatwierdzone anotacje "
        "tablic. Nie wracasz wtedy do przygotowania tablic w E2, tylko "
        "przechodzisz do E3 i przygotowania datasetu znaków."
    ),
    "e2_to_e3_chars": (
        "Ta bramka zamyka przygotowanie tablic i prowadzi do E3. Użyj jej, "
        "gdy masz dość zatwierdzonych tablic, żeby wyodrębnić je i rozpocząć "
        "budowę datasetu znaków."
    ),
    "e2_to_e4_plates": (
        "Ta bramka prowadzi z E2 bezpośrednio do E4T, czyli treningu modelu tablic. "
        "Wybierz ją, gdy zatwierdzone anotacje tablic mają posłużyć do "
        "datasetu YOLO Pose."
    ),
    "e3_to_e4_chars": (
        "Ta bramka zamyka przygotowanie źródłowego datasetu znaków. Warunkiem jest gotowy "
        "dataset znaków YOLO Detect utworzony przez pracę „Przygotuj dataset znaków w Z3”. "
        "Po zatwierdzeniu T05 przechodzisz do E4Z/T06, gdzie z tego źródła przygotujesz "
        "wariant datasetu train/val/test albo świadomie zakończysz iterację bez treningu."
    ),
    "e4_to_e1_next_iteration": (
        "Ta bramka kończy iterację. Po treningu albo świadomej decyzji bez "
        "treningu wracasz z E4T/E4Z do E1, gdzie wybierasz cel i zasoby następnej iteracji."
    ),
}


def transition_display_name(specs: Sequence[CampaignTransitionSpec] | Iterable[CampaignTransitionSpec]) -> str:
    spec_list = tuple(specs or ())
    if not spec_list:
        return "Bramka grafu"
    labels = []
    for spec in spec_list:
        badge_id = str(getattr(spec, "badge_id", "") or "").strip()
        badge_label = str(getattr(spec, "badge_label", "") or "").strip()
        title = str(getattr(spec, "title", "") or "").strip()
        if badge_id and badge_label:
            labels.append(f"{badge_id} - {badge_label}")
        elif title:
            labels.append(title)
    return ", ".join(labels) if labels else "Bramka grafu"


def build_transition_action_copy(
    specs: Sequence[CampaignTransitionSpec] | Iterable[CampaignTransitionSpec],
    report: TransitionResourceReport | None = None,
) -> str:
    spec_list = tuple(specs or ())
    if not spec_list:
        return (
            "Ta bramka nie ma jeszcze pełnego opisu. Opcje pracy widoczne poniżej "
            "są podłączone bezpośrednio do grafu kampanii."
        )

    spec = spec_list[0]
    route = str(getattr(spec, "route_label", "") or "").strip()
    title = transition_display_name(spec_list)
    spec_key = str(getattr(spec, "key", "") or "").strip()
    summary = TRANSITION_SUMMARY_OVERRIDES.get(
        spec_key,
        str(getattr(spec, "summary", "") or "").strip(),
    )
    source = str(getattr(spec, "source", "") or "").strip().upper()

    lines = [f"{title} ({route})." if route else f"{title}."]
    if summary:
        lines.append(summary)
    if report is not None:
        lines.append(report.status_text())

    if source == "E1":
        lines.append(
            "Ścieżkę E1 wybierasz elektrodą przy bramce. Po wyborze pracujesz "
            "przede wszystkim w polu Zasoby, a przejście zamykasz polem Zatwierdź, "
            "gdy wymagania bramki są spełnione."
        )
    else:
        lines.append(
            "Opcje pracy poniżej otwierają właściwe karty robocze. Samo zamknięcie "
            "przejścia wykonujesz polem Zatwierdź na bramce, gdy spełnisz warunki."
        )
    return "\n\n".join(lines)


def build_transition_resource_copy(
    specs: Sequence[CampaignTransitionSpec] | Iterable[CampaignTransitionSpec],
    report: TransitionResourceReport,
) -> str:
    spec_list = tuple(specs or ())
    title = transition_display_name(spec_list)
    spec_key = str(getattr(spec_list[0], "key", "") or "").strip() if spec_list else ""
    status = report.status_text()
    if spec_key == "e3_to_e4_chars":
        if report.required_ready:
            return (
                f"{title}: dataset znaków YOLO Detect jest gotowy. Możesz przejść do pracy bramki "
                "albo zamknąć T06 polem Zatwierdź."
            )
        return (
            f"{title}: {status} Brakuje datasetu znaków. W modalu Praca T06 wybierz "
            "„Przygotuj dataset znaków w Z3”. Jeśli najpierw trzeba dopisać lub poprawić anotacje tablic, "
            "użyj „Uzupełnij anotacje tablic”."
        )
    if report.required_ready:
        return (
            f"{title}: wymagane zasoby są gotowe. Możesz przejść do pracy bramki "
            "albo zatwierdzić przejście, jeśli pole Zatwierdź jest aktywne."
        )
    return (
        f"{title}: {status} Uzupełnij brakujące zasoby, a graf odblokuje dalsze "
        "pracę na tej ścieżce."
    )
