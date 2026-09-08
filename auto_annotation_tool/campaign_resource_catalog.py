"""Shared campaign resource vocabulary.

The campaign graph should describe resources with one stable language. Older
wizard code still uses historical keys such as ``plate_run`` or
``approved_plates``; this catalog maps them to the user-facing O/MT/MZ/AT/AZ
vocabulary without forcing a risky rewrite of every caller at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class CampaignResourceDefinition:
    key: str
    code: str
    label: str
    short_label: str
    description: str
    counted: bool = False
    aliases: Tuple[str, ...] = ()


RESOURCE_DEFINITIONS: Mapping[str, CampaignResourceDefinition] = {
    "images": CampaignResourceDefinition(
        key="images",
        code="O",
        label="O - obrazy",
        short_label="O",
        description="Katalog obrazów wejściowych dla bieżącej iteracji.",
        counted=True,
    ),
    "plate_model": CampaignResourceDefinition(
        key="plate_model",
        code="MT",
        label="MT - model tablic",
        short_label="MT",
        description="Model detekcji tablic używany jako wsparcie autoanotacji albo baza dalszego treningu.",
    ),
    "char_model": CampaignResourceDefinition(
        key="char_model",
        code="MZ",
        label="MZ - model znaków",
        short_label="MZ",
        description="Model znaków używany jako wsparcie autoanotacji znaków albo baza dalszego treningu.",
    ),
    "plate_run": CampaignResourceDefinition(
        key="plate_run",
        code="AT",
        label="AT - anotacje tablic",
        short_label="AT",
        description="Anotacje tablic zgodne z obrazami projektu; mogą pochodzić z importu albo wcześniejszej pracy.",
        counted=True,
        aliases=("approved_plates",),
    ),
    "char_run": CampaignResourceDefinition(
        key="char_run",
        code="AZ",
        label="AZ - anotacje znaków",
        short_label="AZ",
        description="Anotacje znaków zgodne z wyodrębnionymi tablicami.",
        counted=True,
    ),
    "char_dataset": CampaignResourceDefinition(
        key="char_dataset",
        code="DZ",
        label="Dataset znaków YOLO",
        short_label="DZ",
        description=(
            "Dataset znaków YOLO Detect tworzony przez pracę „Przygotuj dataset znaków w Z3”. "
            "Jeśli brakuje materiału wejściowego, użyj pracy „Uzupełnij tablice w Z2”."
        ),
    ),
    "training_result": CampaignResourceDefinition(
        key="training_result",
        code="TR",
        label="Wynik treningu lub decyzja bez treningu",
        short_label="TR",
        description="Rezultat E4T/E4Z albo świadoma decyzja zakończenia iteracji bez treningu.",
    ),
}


_ALIASES: dict[str, str] = {}
for definition in RESOURCE_DEFINITIONS.values():
    _ALIASES[definition.key] = definition.key
    for alias in definition.aliases:
        _ALIASES[str(alias).strip()] = definition.key


RESOURCE_ORDER: Tuple[str, ...] = ("images", "plate_run", "char_run", "plate_model", "char_model")


def normalize_campaign_resource_key(key: str | None) -> str:
    normalized = str(key or "").strip()
    return _ALIASES.get(normalized, normalized)


def get_campaign_resource_definition(key: str | None) -> CampaignResourceDefinition | None:
    return RESOURCE_DEFINITIONS.get(normalize_campaign_resource_key(key))


def campaign_resource_label(key: str | None, fallback: str = "") -> str:
    definition = get_campaign_resource_definition(key)
    if definition is not None:
        return definition.label
    return str(fallback or key or "")


def campaign_resource_short_label(key: str | None, fallback: str = "") -> str:
    definition = get_campaign_resource_definition(key)
    if definition is not None:
        return definition.short_label
    return str(fallback or key or "")


def campaign_resource_description(key: str | None, fallback: str = "") -> str:
    definition = get_campaign_resource_definition(key)
    if definition is not None:
        return definition.description
    return str(fallback or "")


def campaign_resource_counted(key: str | None) -> bool:
    definition = get_campaign_resource_definition(key)
    return bool(definition and definition.counted)


def ordered_campaign_resource_keys(extra_keys: tuple[str, ...] | list[str] | None = None) -> Tuple[str, ...]:
    result: list[str] = list(RESOURCE_ORDER)
    for key in extra_keys or ():
        normalized = normalize_campaign_resource_key(key)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)
