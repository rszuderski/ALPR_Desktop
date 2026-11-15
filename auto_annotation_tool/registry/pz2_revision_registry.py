#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kontrolowany zapis trwałej rewizji AZ z jednego rekordu PZ2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .az_registry import AZRegistry
from .az_revision_store import AZRevisionStore, AZRevisionWriteResult
from .pz2_az_adapter import derive_pz2_revision_context, pz2_metadata_to_az_payload


@dataclass(frozen=True)
class PZ2RevisionSaveResult:
    revision: AZRevisionWriteResult
    source_kind: str
    trust_state: str
    source_status: str
    effective_status: str


def save_pz2_revision(
    registry: AZRegistry,
    data: Mapping[str, Any],
    *,
    project_id: str | None = None,
    iteration_num: int | None = None,
) -> PZ2RevisionSaveResult:
    """Zapisz semantyczny stan PZ2 jako rewizję AZ.

    Wymiary cropa są brane z plate_crops, czyli z kontraktu PZ1/AZ003,
    a nie z efemerycznego stanu renderera GUI.
    """
    if not isinstance(data, Mapping):
        raise TypeError("PZ2 metadata row musi być mapowaniem.")

    crop_id = str(data.get("crop_id") or "").strip()
    if not crop_id:
        raise ValueError("Brak crop_id. Rekord musi pochodzić z AZ003C.")

    registry.initialize()
    with registry.database.read_connection() as connection:
        crop = connection.execute(
            """
            SELECT crop_id, identity_sha256, width, height
            FROM plate_crops
            WHERE crop_id = ?
            """,
            (crop_id,),
        ).fetchone()

    if crop is None:
        raise ValueError(f"Nieznany crop_id: {crop_id}")

    width = int(crop["width"] or 0)
    height = int(crop["height"] or 0)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Crop {crop_id} nie ma poprawnych wymiarów w plate_crops."
        )

    metadata_identity = str(
        data.get("crop_identity_sha256") or ""
    ).strip().lower()
    registry_identity = str(crop["identity_sha256"] or "").strip().lower()
    if metadata_identity != registry_identity:
        raise ValueError(
            "crop_identity_sha256 metadata PZ2 nie odpowiada plate_crops."
        )

    payload = pz2_metadata_to_az_payload(
        data,
        image_width=width,
        image_height=height,
    )
    context = derive_pz2_revision_context(data)

    normalized_project_id = str(project_id or "").strip() or None
    normalized_iteration = (
        int(iteration_num)
        if iteration_num is not None
        else None
    )

    store = AZRevisionStore(registry.database)
    revision = store.save_revision(
        crop_id=crop_id,
        payload=payload,
        source_kind=context.source_kind,
        trust_state=context.trust_state,
        source_status=context.source_status,
        origin_project_id=normalized_project_id,
        origin_iteration=normalized_iteration,
        bind_project_id=normalized_project_id,
        effective_status=(
            context.effective_status
            if normalized_project_id
            else None
        ),
    )

    return PZ2RevisionSaveResult(
        revision=revision,
        source_kind=context.source_kind,
        trust_state=context.trust_state,
        source_status=context.source_status,
        effective_status=context.effective_status,
    )
