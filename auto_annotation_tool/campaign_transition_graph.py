"""Static campaign transition map for the new wizard graph.

The graph is intentionally small: stages are abstract nodes, and transition
edges carry gate semantics. UI decides how to evaluate and render every gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

CAMPAIGN_GRAPH_STAGE_ORDER: Tuple[str, ...] = ("E1", "E2", "E3", "E4T", "E4Z")
TRAINING_STAGE_PLATES = "E4T"
TRAINING_STAGE_CHARS = "E4Z"


def campaign_stage_step(stage_key: str | None) -> int:
    """Return the physical wizard step for a semantic graph node."""

    normalized = str(stage_key or "").strip().upper()
    if normalized.startswith("E4"):
        return 4
    if normalized.startswith("E"):
        digits = []
        for char in normalized[1:]:
            if char.isdigit():
                digits.append(char)
            else:
                break
        try:
            return int("".join(digits) or 0)
        except Exception:
            return 0
    return 0


def campaign_stage_status_key(stage_key: str | None) -> str:
    """Map semantic graph nodes to the legacy campaign status key."""

    step = campaign_stage_step(stage_key)
    return f"E{step}" if step > 0 else str(stage_key or "").strip().upper()


@dataclass(frozen=True)
class CampaignTransitionNode:
    key: str
    title: str
    short_title: str
    description: str = ""


@dataclass(frozen=True)
class CampaignTransitionEdge:
    key: str
    source: str
    target: str
    title: str
    summary: str
    kind: str = "standard"
    paths: Tuple[str, ...] = ()
    resources_label: str = "Zasoby"
    actions_label: str = "Akcje"
    approve_label: str = "Zatwierdź"


@dataclass(frozen=True)
class CampaignTransitionGraph:
    nodes: Mapping[str, CampaignTransitionNode]
    edges: Tuple[CampaignTransitionEdge, ...]

    def get_node(self, key: str | None) -> CampaignTransitionNode | None:
        return self.nodes.get(str(key or "").strip())

    def get_edge(self, key: str | None) -> CampaignTransitionEdge | None:
        edge_key = str(key or "").strip()
        for edge in self.edges:
            if edge.key == edge_key:
                return edge
        return None


CAMPAIGN_TRANSITION_GRAPH = CampaignTransitionGraph(
    nodes={
        "E1": CampaignTransitionNode(
            key="E1",
            title="E1. Wejście",
            short_title="E1",
            description="Wybór celu iteracji i zasobów wejściowych.",
        ),
        "E2": CampaignTransitionNode(
            key="E2",
            title="E2. Tablice",
            short_title="E2",
            description="Anotacja i korekta tablic.",
        ),
        "E3": CampaignTransitionNode(
            key="E3",
            title="E3. Znaki",
            short_title="E3",
            description="Wyodrębnione tablice i anotacja znaków.",
        ),
        "E4T": CampaignTransitionNode(
            key="E4T",
            title="E4T. Trening modelu tablic",
            short_title="E4T",
            description="Dataset, split, augmentacja i trening modelu tablic.",
        ),
        "E4Z": CampaignTransitionNode(
            key="E4Z",
            title="E4Z. Trening modelu znaków",
            short_title="E4Z",
            description="Dataset, split, augmentacja i trening modelu znaków.",
        ),
    },
    edges=(
        CampaignTransitionEdge(
            key="e1_to_e2",
            source="E1",
            target="E2",
            title="Przygotuj anotacje tablic z obrazów",
            summary=(
                "Wspólna bramka przygotowania anotacji tablic na obrazach z zasobów. "
                "Po E2 zdecydujesz, czy anotacje zasilą model tablic, czy tor znaków."
            ),
            kind="standard",
            paths=("plate_training", "char_from_images"),
        ),
        CampaignTransitionEdge(
            key="e2_to_e3",
            source="E2",
            target="E3",
            title="Przekaż tablice do pracy nad znakami",
            summary="Zamknięcie E2 i przekazanie zatwierdzonych anotacji tablic do wyodrębniania tablic oraz budowy datasetu znaków.",
            kind="standard",
            paths=("char_from_images",),
        ),
        CampaignTransitionEdge(
            key="e3_to_e4",
            source="E3",
            target=TRAINING_STAGE_CHARS,
            title="Trening modelu znaków",
            summary="Przejście do przygotowania wariantu treningowego i treningu modelu znaków.",
            kind="standard",
            paths=("char_from_images", "char_from_ready_plates"),
        ),
        CampaignTransitionEdge(
            key="e4t_to_e1",
            source=TRAINING_STAGE_PLATES,
            target="E1",
            title="Kolejna iteracja po modelu tablic",
            summary="Domknięcie treningu modelu tablic i powrót do E1 kolejnej iteracji.",
            kind="standard",
            paths=("plate_training",),
        ),
        CampaignTransitionEdge(
            key="e4z_to_e1",
            source=TRAINING_STAGE_CHARS,
            target="E1",
            title="Kolejna iteracja po modelu znaków",
            summary="Domknięcie treningu modelu znaków i powrót do E1 kolejnej iteracji.",
            kind="standard",
            paths=("char_from_images", "char_from_ready_plates"),
        ),
        CampaignTransitionEdge(
            key="e1_to_e3",
            source="E1",
            target="E3",
            title="Znaki: istniejący zbiór tablic",
            summary="Skrót, gdy istniejące anotacje albo wyodrębnione tablice pozwalają pominąć E2.",
            kind="shortcut",
            paths=("char_from_ready_plates",),
        ),
        CampaignTransitionEdge(
            key="e2_to_e4",
            source="E2",
            target=TRAINING_STAGE_PLATES,
            title="Dataset i trening modelu tablic",
            summary="Skrót toru tablic: po E2 przechodzimy do przygotowania datasetu i treningu modelu tablic.",
            kind="shortcut",
            paths=("plate_training",),
        ),
    ),
)
