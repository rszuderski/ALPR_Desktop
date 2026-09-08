"""Declarative transition contracts for the campaign workflow graph.

This module is the first stable layer between the visual graph and the old
wizard callbacks. A transition spec describes a single possible edge-level
decision: what it means, which resources it cares about, which actions it can
offer, and which approval action closes the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from .campaign_resource_catalog import campaign_resource_label
from .campaign_transition_graph import TRAINING_STAGE_CHARS, TRAINING_STAGE_PLATES


@dataclass(frozen=True)
class TransitionResourceSpec:
    key: str
    label: str
    requirement: str = "optional"
    note: str = ""


@dataclass(frozen=True)
class TransitionActionSpec:
    key: str
    label: str
    graph_action: str
    payload: Mapping[str, object] = field(default_factory=dict)
    tone: str = "info"
    opens_resources_after: bool = False
    iteration_scope: str = "all"


@dataclass(frozen=True)
class CampaignTransitionSpec:
    key: str
    edge_key: str
    source: str
    target: str
    title: str
    summary: str
    route_label: str
    badge_id: str = ""
    badge_label: str = ""
    kind: str = "standard"
    path_key: str = ""
    tone: str = "info"
    resources: Tuple[TransitionResourceSpec, ...] = ()
    actions: Tuple[TransitionActionSpec, ...] = ()
    approve_action: str = ""


TRANSITION_SPECS: Tuple[CampaignTransitionSpec, ...] = (
    CampaignTransitionSpec(
        key="e1_to_e2_prepare_plate_annotations",
        edge_key="e1_to_e2",
        source="E1",
        target="E2",
        title="Bramka wyboru źródła anotacji tablic",
        summary=(
            "To wspólna bramka wejścia do E2/Z2. Dla toru tablic wymaga nowej puli obrazów, "
            "a dla toru znaków może wykorzystać obrazy albo istniejące źródło tablic, jeśli "
            "daje jeszcze sensowną pracę nad anotacjami znaków."
        ),
        route_label="E1 -> E2",
        badge_id="T01",
        badge_label="Trenuj model tablic/znaków od podstaw",
        path_key="",
        tone="success",
        resources=(
            TransitionResourceSpec("images", campaign_resource_label("images"), "required"),
            TransitionResourceSpec("plate_model", campaign_resource_label("plate_model"), "optional"),
            TransitionResourceSpec("plate_run", campaign_resource_label("plate_run"), "optional"),
            TransitionResourceSpec("char_model", campaign_resource_label("char_model"), "optional"),
            TransitionResourceSpec("char_run", campaign_resource_label("char_run"), "optional"),
        ),
        actions=(
            TransitionActionSpec(
                key="select_plate_training",
                label="Przygotuj AT dla modelu tablic",
                graph_action="set_iteration_path",
                payload={"path_key": "plate_training", "refresh": False, "lightweight": True},
                tone="success",
            ),
            TransitionActionSpec(
                key="select_char_from_images",
                label="Przygotuj AT dla modelu znaków",
                graph_action="set_iteration_path",
                payload={"path_key": "char_from_images", "refresh": False, "lightweight": True},
                tone="info",
            ),
        ),
        approve_action="approve_step1",
    ),
    CampaignTransitionSpec(
        key="e1_to_e3_char_from_ready_plates",
        edge_key="e1_to_e3",
        source="E1",
        target="E3",
        title="Bramka pracy nad znakami na istniejącym zbiorze wyodrębnionych tablic",
        summary=(
            "Skrót, gdy projekt ma już źródło tablic: AT z importu lub poprzedniej iteracji "
            "albo materiał, z którego można przygotować zbiór wyodrębnionych tablic. "
            "Ta bramka nie wykonuje pracy; tylko potwierdza źródło i pozwala przejść dalej bez ponownego oznaczania tablic w Z2."
        ),
        route_label="E1 -> E3",
        badge_id="T02",
        badge_label="Trenuj model znaków na podstawie już istniejących anotacji tablic",
        kind="shortcut",
        path_key="char_from_ready_plates",
        tone="warning",
        resources=(
            TransitionResourceSpec("plate_run", campaign_resource_label("plate_run"), "required"),
            TransitionResourceSpec("images", campaign_resource_label("images"), "optional"),
            TransitionResourceSpec("char_model", campaign_resource_label("char_model"), "optional"),
            TransitionResourceSpec("char_run", campaign_resource_label("char_run"), "optional"),
        ),
        actions=(
            TransitionActionSpec(
                key="select_char_from_ready_plates",
                label="Wybierz przejście przez istniejące źródło tablic",
                graph_action="set_iteration_path",
                payload={"path_key": "char_from_ready_plates"},
                tone="warning",
            ),
            TransitionActionSpec(
                key="review_imported_plate_annotations",
                label="Kontroluj AT w Z2",
                graph_action="open_z2_campaign_context",
                payload={
                    "context": {
                        "z2_work_mode": "t02_at_review",
                        "restore_project_start_plate_source": "1",
                    }
                },
                tone="info",
            ),
        ),
        approve_action="approve_step1_ready_plates",
    ),
    CampaignTransitionSpec(
        key="e2_to_e3_chars",
        edge_key="e2_to_e3",
        source="E2",
        target="E3",
        title="Przekaż zatwierdzone anotacje tablic do pracy nad znakami",
        summary="Po przygotowaniu tablic przejdź do Z3: najpierw wyodrębnij tablice, potem przygotuj anotacje i dataset znaków.",
        route_label="E2 -> E3",
        badge_id="T03",
        badge_label="Przekazanie tablic do pracy nad znakami",
        path_key="char_from_images",
        resources=(TransitionResourceSpec("approved_plates", campaign_resource_label("approved_plates"), "required"),),
        actions=(
            TransitionActionSpec(
                "open_z2_first",
                "Przygotuj tablice w Z2",
                "open_z2_campaign_context",
                iteration_scope="first",
            ),
            TransitionActionSpec(
                "open_z2_later",
                "Dodaj jeszcze tablice w Z2",
                "open_z2_campaign_context",
                iteration_scope="later",
            ),
        ),
        approve_action="approve_step2",
    ),
    CampaignTransitionSpec(
        key="e2_to_e4_plates",
        edge_key="e2_to_e4",
        source="E2",
        target=TRAINING_STAGE_PLATES,
        title="Przygotuj dataset i trening modelu tablic",
        summary="Po E2 przejdź do Z4: z zatwierdzonych anotacji tablic przygotujesz wariant datasetu, a następnie trening modelu tablic.",
        route_label="E2 -> E4T",
        badge_id="T04",
        badge_label="Dataset i trening modelu tablic",
        kind="shortcut",
        path_key="plate_training",
        resources=(TransitionResourceSpec("approved_plates", campaign_resource_label("approved_plates"), "required"),),
        actions=(
            TransitionActionSpec(
                "open_z2_first",
                "Przygotuj tablice w Z2",
                "open_z2_campaign_context",
                iteration_scope="first",
            ),
            TransitionActionSpec(
                "open_z2_later",
                "Dodaj jeszcze tablice w Z2",
                "open_z2_campaign_context",
                iteration_scope="later",
            ),
        ),
        approve_action="approve_step2",
    ),
    CampaignTransitionSpec(
        key="e3_to_e4_chars",
        edge_key="e3_to_e4",
        source="E3",
        target=TRAINING_STAGE_CHARS,
        title="Bramka datasetu znaków do treningu",
        summary=(
            "Zamknij pracę nad znakami w Z3: przygotuj dataset znaków i dopiero wtedy przejdź do E4Z, "
            "czyli treningu modelu znaków."
        ),
        route_label="E3 -> E4Z",
        badge_id="T05",
        badge_label="Dataset znaków",
        path_key="",
        resources=(TransitionResourceSpec("char_dataset", campaign_resource_label("char_dataset"), "required"),),
        actions=(
            TransitionActionSpec("continue_z3", "Przygotuj dataset znaków w Z3", "continue_z3"),
            TransitionActionSpec("repair_plates", "Uzupełnij anotacje tablic", "open_z2_step3_repair", tone="warning"),
        ),
        approve_action="approve_step3",
    ),
    CampaignTransitionSpec(
        key="e4_to_e1_next_iteration",
        edge_key="e4_to_e1",
        source="E4",
        target="E1",
        title="Zamknij iterację po treningu",
        summary="Po treningu albo świadomym pominięciu treningu przejdź z E4T/E4Z do E1 kolejnej iteracji.",
        route_label="E4T/E4Z -> E1",
        badge_id="T06",
        badge_label="Zamknij iterację",
        resources=(TransitionResourceSpec("training_result", campaign_resource_label("training_result"), "required"),),
        actions=(
            TransitionActionSpec("open_z4_dataset", "Utwórz / przebuduj wariant datasetu", "open_z4_dataset"),
            TransitionActionSpec("open_z4", "Trenuj model", "open_z4"),
            TransitionActionSpec("repair_plates", "Uzupełnij anotacje tablic", "open_z2_step3_repair", tone="warning"),
            TransitionActionSpec(
                "approve_without_training",
                "Wybierz zakończenie bez treningu",
                "prepare_step4_without_training",
                tone="warning",
            ),
        ),
        approve_action="approve_step4",
    ),
)

EDGE_KEY_ALIASES = {
    "e1_to_e2": ("e1_to_e2",),
    "e1_to_e2_plate_training": ("e1_to_e2",),
    "e1_to_e2_char_from_images": ("e1_to_e2",),
    "e4t_to_e1": ("e4_to_e1",),
    "e4z_to_e1": ("e4_to_e1",),
}


def get_transition_specs_for_edge(edge_key: str | None) -> Tuple[CampaignTransitionSpec, ...]:
    normalized = str(edge_key or "").strip()
    edge_keys = EDGE_KEY_ALIASES.get(normalized, (normalized,))
    return tuple(spec for spec in TRANSITION_SPECS if spec.edge_key in edge_keys)


def get_transition_spec(key: str | None) -> CampaignTransitionSpec | None:
    normalized = str(key or "").strip()
    for spec in TRANSITION_SPECS:
        if spec.key == normalized:
            return spec
    return None


def get_transition_specs_for_path(path_key: str | None) -> Tuple[CampaignTransitionSpec, ...]:
    normalized = str(path_key or "").strip()
    return tuple(spec for spec in TRANSITION_SPECS if spec.path_key == normalized)


def transition_action_visible_for_iteration(
    action: TransitionActionSpec,
    iteration_num: int | None = None,
) -> bool:
    scope = str(getattr(action, "iteration_scope", "all") or "all").strip().lower()
    if scope in {"", "all", "any"}:
        return True
    try:
        iteration = int(iteration_num or 1)
    except Exception:
        iteration = 1
    if scope in {"first", "initial", "it1"}:
        return iteration <= 1
    if scope in {"later", "next", "k>1", "gt1", "after_first"}:
        return iteration > 1
    return True


def get_transition_action_specs_for_edge(
    edge_key: str | None,
    iteration_num: int | None = None,
) -> Tuple[TransitionActionSpec, ...]:
    actions: list[TransitionActionSpec] = []
    for spec in get_transition_specs_for_edge(edge_key):
        actions.extend(
            action for action in spec.actions
            if transition_action_visible_for_iteration(action, iteration_num)
        )
    return tuple(actions)
