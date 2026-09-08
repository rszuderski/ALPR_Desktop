"""Compatibility helpers for campaign iteration paths.

This module intentionally keeps only the small E1 path contract. The previous
canvas/state-machine graph was removed so the new campaign graph can be designed
from a clean model instead of reusing the experimental renderer.
"""

from __future__ import annotations

from typing import Dict, Tuple

STEP1_ITERATION_PATHS: Dict[str, dict] = {
    "plate_training": {
        "target": "plate",
        "title": "Model tablic",
        "short_title": "Tablice",
        "summary": "Zdjęcia -> E2/Z2 anotacje tablic -> Z4/E4T trening modelu tablic",
        "requires": ("images",),
        "stage_sequence": ("E1", "E2/Z2", "Z4/E4T"),
        "edge_sequence": ("e1_to_e2", "e2_to_e4", "e4t_to_e1"),
        "scope": "campaign",
    },
    "char_from_ready_plates": {
        "target": "char",
        "title": "Model znaków z istniejących tablic",
        "short_title": "Znaki: istniejący zbiór tablic",
        "summary": "Istniejące anotacje tablic -> E3/Z3 dataset znaków -> Z4/E4Z trening modelu znaków",
        "requires": ("plate_run",),
        "stage_sequence": ("E1", "E3/Z3", "Z4/E4Z"),
        "edge_sequence": ("e1_to_e3", "e3_to_e4", "e4z_to_e1"),
        "scope": "campaign",
    },
    "char_from_images": {
        "target": "char",
        "title": "Model znaków: brakujące anotacje tablic",
        "short_title": "Model znaków: brakujące AT",
        "summary": "Obrazy z zasobów -> E2/Z2 brakujące anotacje tablic -> E3/Z3 dataset znaków -> Z4/E4Z trening modelu znaków",
        "requires": ("images",),
        "stage_sequence": ("E1", "E2/Z2", "E3/Z3", "Z4/E4Z"),
        "edge_sequence": ("e1_to_e2", "e2_to_e3", "e3_to_e4", "e4z_to_e1"),
        "scope": "campaign",
    },
}
STEP1_PATH_ORDER: Tuple[str, ...] = ("plate_training", "char_from_ready_plates", "char_from_images")


def normalize_iteration_path(path: str | None) -> str:
    value = str(path or "").strip().lower()
    aliases = {
        "plate": "plate_training",
        "plates": "plate_training",
        "tablice": "plate_training",
        "char_ready": "char_from_ready_plates",
        "ready_plates": "char_from_ready_plates",
        "znaki_gotowe": "char_from_ready_plates",
        "char_images": "char_from_images",
        "new_images": "char_from_images",
        "znaki_zdjecia": "char_from_images",
    }
    value = aliases.get(value, value)
    return value if value in STEP1_ITERATION_PATHS else ""


def iteration_path_target(path: str | None) -> str:
    normalized = normalize_iteration_path(path)
    if not normalized:
        return ""
    return str(STEP1_ITERATION_PATHS[normalized].get("target") or "").strip().lower()


def default_iteration_path_for_target(target: str | None) -> str:
    value = str(target or "").strip().lower()
    if value == "plate":
        return "plate_training"
    if value == "char":
        return "char_from_images"
    return ""


def get_iteration_path_definition(path: str | None) -> dict:
    normalized = normalize_iteration_path(path)
    if not normalized:
        return {}
    return dict(STEP1_ITERATION_PATHS.get(normalized) or {})
