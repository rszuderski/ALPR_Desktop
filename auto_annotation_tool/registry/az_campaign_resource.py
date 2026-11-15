#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agregacja AZ projektu do statusu zasobu kampanii.

AZ007A: warstwa tylko do odczytu. Nie tworzy projektu, cropów ani bindingów.
Czyta istniejące project_crop_members/project_crop_az i zwraca mały,
UI-neutralny snapshot stanu AZ aktywnego projektu.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .az_registry import AZRegistry


_READY_STATUSES = {
    "approved",
    "ok",
    "ready",
    "perfect",
}
_PENDING_STATUSES = {
    "imported_pending_review",
    "needs_fix",
    "pending_review",
    "review_required",
    "candidate",
    "unknown",
}
_EXCLUDED_STATUSES = {
    "excluded",
}


@dataclass(frozen=True)
class AZCampaignResourceState:
    project_id: str
    project_exists: bool
    crop_count: int
    az_count: int
    usable_count: int
    ready_count: int
    pending_review_count: int
    excluded_count: int
    other_count: int
    latest_updated_at: str
    coverage_status: str

    @property
    def missing_count(self) -> int:
        return max(0, int(self.crop_count) - int(self.az_count))

    @property
    def reviewed_count(self) -> int:
        return int(self.ready_count) + int(self.excluded_count)

    @property
    def review_required(self) -> bool:
        return bool(
            int(self.pending_review_count) > 0
            or int(self.other_count) > 0
        )

    @property
    def review_complete(self) -> bool:
        return bool(
            int(self.crop_count) > 0
            and self.missing_count == 0
            and not self.review_required
            and self.reviewed_count == int(self.crop_count)
        )

    @property
    def contract_ready(self) -> bool:
        # AZ może być kontraktowo gotowe dopiero po zakończonym review
        # całego targetowego zbioru cropów. Świadome excluded liczy się jako
        # zakończona decyzja, ale projekt musi mieć co najmniej jedną gotową AZ.
        return bool(
            self.review_complete
            and int(self.ready_count) > 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_count"] = self.missing_count
        payload["reviewed_count"] = self.reviewed_count
        payload["review_required"] = self.review_required
        payload["review_complete"] = self.review_complete
        payload["contract_ready"] = self.contract_ready
        return payload


def summarize_project_az_resource(
    registry: AZRegistry,
    *,
    project_id: str,
) -> AZCampaignResourceState:
    """Zwróć stan AZ projektu bez modyfikowania registry."""
    normalized_project_id = str(project_id or "").strip()
    if not normalized_project_id:
        raise ValueError("project_id nie może być puste.")

    registry.initialize()
    with registry.database.read_connection() as connection:
        project = connection.execute(
            """
            SELECT project_id
            FROM projects
            WHERE project_id = ?
            """,
            (normalized_project_id,),
        ).fetchone()

        if project is None:
            return AZCampaignResourceState(
                project_id=normalized_project_id,
                project_exists=False,
                crop_count=0,
                az_count=0,
                usable_count=0,
                ready_count=0,
                pending_review_count=0,
                excluded_count=0,
                other_count=0,
                latest_updated_at="",
                coverage_status="no_project",
            )

        crop_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM project_crop_members
                WHERE project_id = ?
                """,
                (normalized_project_id,),
            ).fetchone()[0]
        )

        rows = connection.execute(
            """
            SELECT
                pca.effective_status,
                pca.updated_at
            FROM project_crop_az pca
            JOIN project_crop_members pcm
              ON pcm.project_id = pca.project_id
             AND pcm.crop_id = pca.crop_id
            WHERE pca.project_id = ?
            ORDER BY pca.updated_at DESC, pca.crop_id
            """,
            (normalized_project_id,),
        ).fetchall()

    ready = pending = excluded = other = 0
    latest_updated_at = ""
    for row in rows:
        status = str(row["effective_status"] or "").strip().lower()
        updated_at = str(row["updated_at"] or "").strip()
        if updated_at and (not latest_updated_at or updated_at > latest_updated_at):
            latest_updated_at = updated_at

        if status in _READY_STATUSES:
            ready += 1
        elif status in _PENDING_STATUSES:
            pending += 1
        elif status in _EXCLUDED_STATUSES:
            excluded += 1
        else:
            other += 1

    az_count = len(rows)
    usable_count = max(0, az_count - excluded)

    if crop_count <= 0:
        coverage_status = "no_crops"
    elif az_count <= 0:
        coverage_status = "missing"
    elif az_count < crop_count:
        coverage_status = "partial"
    else:
        coverage_status = "full"

    return AZCampaignResourceState(
        project_id=normalized_project_id,
        project_exists=True,
        crop_count=crop_count,
        az_count=az_count,
        usable_count=usable_count,
        ready_count=ready,
        pending_review_count=pending,
        excluded_count=excluded,
        other_count=other,
        latest_updated_at=latest_updated_at,
        coverage_status=coverage_status,
    )
