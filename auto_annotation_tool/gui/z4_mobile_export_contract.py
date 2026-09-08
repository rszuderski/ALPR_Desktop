"""User-facing names for the two existing mobile export contracts."""
from dataclasses import dataclass
from collections.abc import Iterable

from ..exporters.mobile_model_exporter import MOBILE_MODEL_SCHEMA, MOBILE_ALPR_PACKAGE_SCHEMA


@dataclass(frozen=True)
class MobileExportSelection:
    valid: bool
    label: str
    message: str
    schema: str = ""

    @property
    def is_package(self) -> bool:
        return self.schema == MOBILE_ALPR_PACKAGE_SCHEMA

    @property
    def export_label(self) -> str:
        return "Eksportuj pakiet ALPR (.alprmodel)" if self.is_package else "Eksportuj model mobilny (.alprmodel)"

    @property
    def window_title(self) -> str:
        return "Eksport kompletnego pakietu ALPR" if self.is_package else "Eksport modelu mobilnego"

    @property
    def save_title(self) -> str:
        return "Zapisz pakiet ALPR (.alprmodel)" if self.is_package else "Zapisz model mobilny (.alprmodel)"

    @property
    def success_title(self) -> str:
        return "Pakiet ALPR gotowy" if self.is_package else "Model mobilny gotowy"

    @property
    def progress_message(self) -> str:
        return "Eksportuję kompletny pakiet ALPR." if self.is_package else "Eksportuję model mobilny."


def describe_mobile_export_selection(markers: Iterable[str]) -> MobileExportSelection:
    selected = tuple(str(marker).strip().upper() for marker in markers)
    roles = set(selected)
    if not selected:
        return MobileExportSelection(False, "brak wyboru",
            "Wybierz pojedynczy model MP, MT lub MZ albo kompletny pakiet ALPR: MT+MZ lub MP+MT+MZ.")
    if len(roles) != len(selected) or not roles <= {"MP", "MT", "MZ"}:
        return MobileExportSelection(False, "niezgodny wybór",
            "Wybierz najwyżej jeden model każdej roli: MP, MT, MZ.")
    if len(selected) == 1:
        marker = selected[0]
        return MobileExportSelection(True, f"model mobilny {marker}",
            f"1 model wybrany: {marker} → model mobilny {MOBILE_MODEL_SCHEMA}.", MOBILE_MODEL_SCHEMA)
    if {"MT", "MZ"} <= roles:
        names = "MP+MT+MZ" if "MP" in roles else "MT+MZ"
        return MobileExportSelection(True, f"pakiet ALPR {names}",
            f"{names} → kompletny pakiet ALPR {MOBILE_ALPR_PACKAGE_SCHEMA}.", MOBILE_ALPR_PACKAGE_SCHEMA)
    missing = "MZ" if "MT" in roles else "MT"
    names = "+".join(marker for marker in ("MP", "MT", "MZ") if marker in roles)
    return MobileExportSelection(False, f"dodaj model {missing}",
        f"Wybrano {names}. Kompletny pakiet ALPR wymaga MT i MZ. "
        f"Dodaj model {missing} albo wybierz jeden model do eksportu mobilnego.")
