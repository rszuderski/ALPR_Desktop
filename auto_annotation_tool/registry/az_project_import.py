#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bezpieczna promocja/import AZ między projektami w jednym registry.

AZ006A nie kopiuje payloadów i nie tworzy nowych rewizji. Import jest
dopuszczalny wyłącznie dla logicznych cropów należących już do projektu
docelowego. Istniejąca inna AZ projektu docelowego jest konfliktem i nie jest
nadpisywana.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Iterable

from .az_registry import AZRegistry


IMPORT_PENDING_REVIEW = "imported_pending_review"


@dataclass(frozen=True)
class AZProjectImportItem:
    crop_id: str
    az_revision_id: str
    source_effective_status: str
    status: str
    target_az_revision_id: str | None
    origin_project_id: str | None
    origin_iteration: int | None
    source_kind: str
    trust_state: str
    approved: bool
    excluded: bool
    character_count: int


@dataclass(frozen=True)
class AZProjectImportPlan:
    source_project_id: str
    target_project_id: str
    items: tuple[AZProjectImportItem, ...]

    @property
    def importable_count(self) -> int:
        return sum(item.status == "importable" for item in self.items)

    @property
    def already_bound_count(self) -> int:
        return sum(item.status == "already_bound" for item in self.items)

    @property
    def conflict_count(self) -> int:
        return sum(item.status == "target_conflict" for item in self.items)

    @property
    def target_missing_crop_count(self) -> int:
        return sum(item.status == "target_missing_crop" for item in self.items)

    @property
    def invalid_source_count(self) -> int:
        return sum(item.status == "source_missing_crop_membership" for item in self.items)


@dataclass(frozen=True)
class AZProjectImportResult:
    source_project_id: str
    target_project_id: str
    imported: int
    already_bound: int
    conflicts: int
    target_missing_crop: int
    invalid_source: int
    skipped_race: int
    effective_status: str


@dataclass(frozen=True)
class AZProjectImportSourceCandidate:
    source_project_id: str
    target_project_id: str
    source_display_name: str
    source_folder_name: str
    source_campaign_key: str
    source_az_count: int
    importable_count: int
    already_bound_count: int
    conflict_count: int
    target_missing_crop_count: int
    invalid_source_count: int

    @property
    def can_import(self) -> bool:
        return int(self.importable_count) > 0

    @property
    def compatible_count(self) -> int:
        return int(self.importable_count) + int(self.already_bound_count)


def analyze_project_az_import(
    registry: AZRegistry,
    *,
    source_project_id: str,
    target_project_id: str,
    crop_ids: Iterable[str] | None = None,
) -> AZProjectImportPlan:
    """Zbuduj bezpieczny plan promocji AZ source -> target."""
    registry.initialize()
    source_project_id = _required_text("source_project_id", source_project_id)
    target_project_id = _required_text("target_project_id", target_project_id)
    if source_project_id == target_project_id:
        raise ValueError("Projekt źródłowy i docelowy muszą być różne.")

    selected = _normalize_crop_ids(crop_ids)

    with registry.database.read_connection() as connection:
        _require_project(connection, source_project_id)
        _require_project(connection, target_project_id)

        source_rows = connection.execute(
            """
            SELECT
                pca.crop_id,
                pca.az_revision_id,
                pca.effective_status,
                ar.payload_json,
                ar.origin_project_id,
                ar.origin_iteration,
                ar.source_kind,
                ar.trust_state,
                CASE WHEN spcm.crop_id IS NULL THEN 0 ELSE 1 END AS source_member
            FROM project_crop_az pca
            JOIN az_revisions ar
              ON ar.az_revision_id = pca.az_revision_id
            LEFT JOIN project_crop_members spcm
              ON spcm.project_id = pca.project_id
             AND spcm.crop_id = pca.crop_id
            WHERE pca.project_id = ?
            ORDER BY pca.crop_id
            """,
            (source_project_id,),
        ).fetchall()

        target_members = {
            str(row["crop_id"])
            for row in connection.execute(
                """
                SELECT crop_id
                FROM project_crop_members
                WHERE project_id = ?
                """,
                (target_project_id,),
            ).fetchall()
        }

        target_bindings = {
            str(row["crop_id"]): str(row["az_revision_id"])
            for row in connection.execute(
                """
                SELECT crop_id, az_revision_id
                FROM project_crop_az
                WHERE project_id = ?
                """,
                (target_project_id,),
            ).fetchall()
        }

    items: list[AZProjectImportItem] = []
    for row in source_rows:
        crop_id = str(row["crop_id"])
        if selected is not None and crop_id not in selected:
            continue

        target_revision = target_bindings.get(crop_id)
        if not bool(row["source_member"]):
            status = "source_missing_crop_membership"
        elif crop_id not in target_members:
            status = "target_missing_crop"
        elif target_revision == str(row["az_revision_id"]):
            status = "already_bound"
        elif target_revision:
            status = "target_conflict"
        else:
            status = "importable"

        payload = json.loads(str(row["payload_json"]))
        gold = payload.get("gold_state")
        gold = gold if isinstance(gold, dict) else {}
        chars = payload.get("characters")
        chars = chars if isinstance(chars, list) else []

        items.append(
            AZProjectImportItem(
                crop_id=crop_id,
                az_revision_id=str(row["az_revision_id"]),
                source_effective_status=str(row["effective_status"] or ""),
                status=status,
                target_az_revision_id=target_revision,
                origin_project_id=_optional_text(row["origin_project_id"]),
                origin_iteration=(
                    int(row["origin_iteration"])
                    if row["origin_iteration"] is not None
                    else None
                ),
                source_kind=str(row["source_kind"] or ""),
                trust_state=str(row["trust_state"] or ""),
                approved=bool(gold.get("approved", False)),
                excluded=bool(gold.get("excluded", False)),
                character_count=len(chars),
            )
        )

    return AZProjectImportPlan(
        source_project_id=source_project_id,
        target_project_id=target_project_id,
        items=tuple(items),
    )


def list_project_az_import_sources(
    registry: AZRegistry,
    *,
    target_project_id: str,
) -> tuple[AZProjectImportSourceCandidate, ...]:
    # Read-only discovery of projects that already own AZ.
    registry.initialize()
    target_project_id = _required_text(
        "target_project_id",
        target_project_id,
    )

    with registry.database.read_connection() as connection:
        _require_project(connection, target_project_id)
        rows = connection.execute(
            """
            SELECT
                p.project_id,
                p.display_name,
                p.folder_name,
                p.campaign_key,
                COUNT(pca.crop_id) AS az_count
            FROM projects p
            JOIN project_crop_az pca
              ON pca.project_id = p.project_id
            WHERE p.project_id <> ?
            GROUP BY
                p.project_id,
                p.display_name,
                p.folder_name,
                p.campaign_key
            HAVING COUNT(pca.crop_id) > 0
            ORDER BY
                COALESCE(p.display_name, p.folder_name, p.project_id),
                p.project_id
            """,
            (target_project_id,),
        ).fetchall()

    candidates: list[AZProjectImportSourceCandidate] = []
    for row in rows:
        source_project_id = str(row["project_id"])
        plan = analyze_project_az_import(
            registry,
            source_project_id=source_project_id,
            target_project_id=target_project_id,
        )
        candidates.append(
            AZProjectImportSourceCandidate(
                source_project_id=source_project_id,
                target_project_id=target_project_id,
                source_display_name=(
                    str(row["display_name"] or "").strip()
                    or str(row["folder_name"] or "").strip()
                    or source_project_id
                ),
                source_folder_name=str(row["folder_name"] or "").strip(),
                source_campaign_key=str(row["campaign_key"] or "").strip(),
                source_az_count=int(row["az_count"] or 0),
                importable_count=plan.importable_count,
                already_bound_count=plan.already_bound_count,
                conflict_count=plan.conflict_count,
                target_missing_crop_count=plan.target_missing_crop_count,
                invalid_source_count=plan.invalid_source_count,
            )
        )

    candidates.sort(
        key=lambda item: (
            -int(item.importable_count),
            -int(item.already_bound_count),
            int(item.conflict_count),
            int(item.target_missing_crop_count),
            item.source_display_name.casefold(),
            item.source_project_id,
        )
    )
    return tuple(candidates)


def import_project_az_bindings(
    registry: AZRegistry,
    *,
    source_project_id: str,
    target_project_id: str,
    crop_ids: Iterable[str] | None = None,
    effective_status: str = IMPORT_PENDING_REVIEW,
    updated_at: str | None = None,
) -> AZProjectImportResult:
    """Przypnij bezpiecznie zgodne rewizje AZ do projektu docelowego.

    Nie tworzy nowych az_revisions. Nie nadpisuje istniejącej innej AZ targetu.
    Każdy importowany binding domyślnie trafia do kontroli.
    """
    effective_status = _required_text("effective_status", effective_status)
    plan = analyze_project_az_import(
        registry,
        source_project_id=source_project_id,
        target_project_id=target_project_id,
        crop_ids=crop_ids,
    )
    timestamp = str(updated_at or "").strip() or _utc_now_iso()

    imported = 0
    skipped_race = 0

    with registry.database.transaction() as connection:
        # Rewalidacja w transakcji: plan mógł się zdezaktualizować.
        for item in plan.items:
            if item.status != "importable":
                continue

            member = connection.execute(
                """
                SELECT 1
                FROM project_crop_members
                WHERE project_id = ? AND crop_id = ?
                """,
                (plan.target_project_id, item.crop_id),
            ).fetchone()
            if member is None:
                skipped_race += 1
                continue

            current = connection.execute(
                """
                SELECT az_revision_id
                FROM project_crop_az
                WHERE project_id = ? AND crop_id = ?
                """,
                (plan.target_project_id, item.crop_id),
            ).fetchone()
            if current is not None:
                skipped_race += 1
                continue

            revision = connection.execute(
                """
                SELECT crop_id
                FROM az_revisions
                WHERE az_revision_id = ?
                """,
                (item.az_revision_id,),
            ).fetchone()
            if revision is None or str(revision["crop_id"]) != item.crop_id:
                skipped_race += 1
                continue

            connection.execute(
                """
                INSERT INTO project_crop_az (
                    project_id,
                    crop_id,
                    az_revision_id,
                    effective_status,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    plan.target_project_id,
                    item.crop_id,
                    item.az_revision_id,
                    effective_status,
                    timestamp,
                ),
            )
            imported += 1

    return AZProjectImportResult(
        source_project_id=plan.source_project_id,
        target_project_id=plan.target_project_id,
        imported=imported,
        already_bound=plan.already_bound_count,
        conflicts=plan.conflict_count,
        target_missing_crop=plan.target_missing_crop_count,
        invalid_source=plan.invalid_source_count,
        skipped_race=skipped_race,
        effective_status=effective_status,
    )


def _normalize_crop_ids(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    result = {
        str(value or "").strip()
        for value in values
        if str(value or "").strip()
    }
    return result


def _require_project(connection, project_id: str) -> None:
    row = connection.execute(
        "SELECT 1 FROM projects WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Nieznany project_id: {project_id}")


def _required_text(name: str, value: Any) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} nie może być puste.")
    return result


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
