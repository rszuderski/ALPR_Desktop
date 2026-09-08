"""Contract for imported/adopted plate annotations in campaign E1/Z2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlateAnnotationImportContract:
    matched_images: int = 0
    matched_plates: int = 0
    adoptable_images: int = 0
    approved_overlap: int = 0
    skipped_entries: int = 0
    target: str = ""

    @property
    def remaining_for_z2(self) -> int:
        return max(0, int(self.adoptable_images or 0) - int(self.matched_images or 0))

    @property
    def has_compatible_annotations(self) -> bool:
        return int(self.matched_images or 0) > 0 and int(self.matched_plates or 0) > 0

    def summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("Kompatybilne zdjęcia do importu", str(int(self.matched_images or 0))),
            ("Tablice w kompatybilnych anotacjach", str(int(self.matched_plates or 0))),
            ("Po adopcji zostanie w E2/Z2", f"{self.remaining_for_z2} zdjęć do dopracowania"),
            ("Już wcześniej zatwierdzone [OK]", str(int(self.approved_overlap or 0))),
            ("Pominięte / niepełne wpisy", str(int(self.skipped_entries or 0))),
        ]

    def adoption_z2_notice(self) -> str:
        return (
            f"Zgodne anotacje obejmą {int(self.matched_images or 0)} zdjęć i "
            f"{int(self.matched_plates or 0)} tablic. Po adopcji te zdjęcia trafią "
            "od razu do puli [OK] i nie będą ponownie pokazywane w E2/Z2. "
            f"W Z2 zostanie {self.remaining_for_z2} zdjęć bez zaadoptowanych anotacji."
        )

    def draft_z2_notice(self) -> str:
        return (
            "Import roboczy podepnie anotacje do E1/E2, ale nie nada im statusu [OK]. "
            "W Z2 użytkownik musi je sprawdzić i jawnie zatwierdzić przed eksportem "
            "datasetu albo wyodrębnianiem tablic. Nie jest to zasób otwierający T03."
        )

    def route_rows(self) -> list[tuple[str, str]]:
        target = str(self.target or "").strip().lower()
        if target == "plate":
            return [
                ("Adopcja [OK]", self.adoption_z2_notice()),
                ("Import roboczy", self.draft_z2_notice()),
            ]
        if target == "char":
            return [
                (
                    "Adopcja [OK]",
                    self.adoption_z2_notice()
                    + " Zaadoptowana pula może być od razu źródłem do wyodrębniania tablic dla E3/Z3.",
                ),
                ("Import roboczy", self.draft_z2_notice()),
            ]
        return [
            (
                "Jeśli wybierzesz tor tablic",
                "Adopcja [OK] zasili materiał do treningu/eksportu tablic. Import roboczy wymaga kontroli w Z2.",
            ),
            (
                "Jeśli wybierzesz tor znaków",
                "Adopcja [OK] może dać źródło do wyodrębniania tablic dla E3. Import roboczy wymaga kontroli w Z2.",
            ),
        ]


def build_plate_annotation_import_contract(
    compatibility: Mapping[str, object] | None,
    origin_summary: Mapping[str, object] | None = None,
    *,
    target: str = "",
) -> PlateAnnotationImportContract:
    data = dict(compatibility or {})
    origin = dict(origin_summary or {})
    matched_images = int(data.get("adoptable_matched", data.get("matched", 0)) or 0)
    matched_plates = int(data.get("adoptable_matched_plate_count", data.get("matched_plate_count", 0)) or 0)
    if matched_plates <= 0 and matched_images > 0:
        matched_plates = int(origin.get("scope_plates", 0) or 0)
    skipped_entries = int(data.get("missing", 0) or 0) + int(data.get("incomplete", 0) or 0)
    return PlateAnnotationImportContract(
        matched_images=matched_images,
        matched_plates=matched_plates,
        adoptable_images=int(data.get("adoptable_image_count", 0) or 0),
        approved_overlap=int(data.get("approved_overlap", 0) or 0),
        skipped_entries=skipped_entries,
        target=str(target or "").strip().lower(),
    )
