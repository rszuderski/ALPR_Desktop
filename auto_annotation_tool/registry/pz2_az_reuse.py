#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bezpieczne automatyczne reuse trwałych AZ w metadata PZ2.

Warstwa nie zna Tkintera ani CAMPAIGN. Dostaje gotowy AZRegistry,
metadata map oraz opcjonalny project_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .az_registry import AZRegistry
from .az_revision_store import AZRevisionStore
from .pz2_az_adapter import az_revision_to_pz2_metadata


@dataclass(frozen=True)
class PZ2AZReuseSummary:
    scanned: int
    resolved: int
    applied: int
    already_current: int
    protected: int
    missing_revision: int
    invalid_crop: int
    errors: int


def pz2_az_reuse_protection_reason(
    data: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> str:
    """Zwróć powód ochrony istniejącego stanu PZ2 albo pusty tekst."""
    if not isinstance(data, Mapping):
        return "invalid_metadata"

    revision_id = str(revision.get("az_revision_id") or "").strip()
    reuse = data.get("az_reuse")
    reuse = reuse if isinstance(reuse, Mapping) else {}
    current_reuse_id = str(reuse.get("az_revision_id") or "").strip()
    fusion_strategy = str(data.get("fusion_strategy") or "").strip().lower()

    if current_reuse_id and current_reuse_id == revision_id:
        effective_status = str(
            revision.get("effective_status") or ""
        ).strip().lower()

        if effective_status != "imported_pending_review":
            return "already_current"

        same_review = data.get("review_state")
        same_review = same_review if isinstance(same_review, Mapping) else {}
        same_review_status = str(
            same_review.get("status") or ""
        ).strip().lower()
        same_review_source = str(
            same_review.get("source") or ""
        ).strip().lower()

        # Praca człowieka zawsze wygrywa nad automatyczną rematerializacją.
        if bool(same_review.get("human_edited")):
            return "human_review_edit"
        if (
            same_review_status == "approved"
            or same_review.get("approved_at")
            or same_review.get("approved_reference")
        ):
            return "human_review_approved"
        if same_review_source == "manual_editor":
            return "manual_review"

        same_gold = data.get("gold_state")
        same_gold = same_gold if isinstance(same_gold, Mapping) else {}
        if bool(same_gold.get("excluded", False)):
            return "local_excluded"
        if bool(same_gold.get("approved", False)):
            return "local_gold"

        pending_materialization_current = bool(
            str(data.get("status") or "").strip().lower() == "needs_fix"
            and not bool(same_gold.get("candidate", False))
            and not bool(same_gold.get("approved", False))
            and not bool(same_gold.get("excluded", False))
            and bool(reuse.get("requires_review", False))
            and str(reuse.get("effective_status") or "").strip().lower()
            == "imported_pending_review"
            and same_review_status == "in_progress"
            and same_review_source == "az_project_import"
        )
        if pending_materialization_current:
            return "already_current"

        # Ta sama rewizja, ale stary/niepełny stan TARGET: rematerializuj.
        return ""

    review = data.get("review_state")
    review = review if isinstance(review, Mapping) else {}
    review_status = str(review.get("status") or "").strip().lower()
    review_source = str(review.get("source") or "").strip().lower()

    # Stan pochodzący wyłącznie z automatycznego RAW nie jest jeszcze pracą
    # człowieka i może zostać zastąpiony lepszym trwałym AZ.
    automatic_raw_review = bool(
        review
        and review_status == "in_progress"
        and review_source == "raw_detection"
        and not bool(review.get("human_edited"))
        and not review.get("approved_at")
        and not review.get("approved_reference")
        and not review.get("reopened_at")
    )

    if review and not automatic_raw_review:
        if bool(review.get("human_edited")):
            return "human_review_edit"
        if review_status == "approved" or review.get("approved_at") or review.get("approved_reference"):
            return "human_review_approved"
        if review_source == "manual_editor":
            return "manual_review"
        if review_status and fusion_strategy != "az_reuse":
            return "review_in_progress"

    if fusion_strategy in {"manual_correction", "manual", "cvat_import"}:
        return "manual_fusion"

    source_info = data.get("source_info")
    source_info = source_info if isinstance(source_info, Mapping) else {}
    source_bucket = str(source_info.get("bucket") or "").strip().lower()
    source_origin = str(source_info.get("origin") or "").strip().lower()
    last_modified_by = str(source_info.get("last_modified_by") or "").strip().lower()

    if source_bucket in {"local_manual", "cvat_manual"}:
        return "manual_source"
    if source_origin in {"preview_editor", "cvat_import"}:
        return "manual_source"
    if last_modified_by == "human":
        return "human_source"

    # Jeśli rekord już pochodzi z AZ, jego semantyczny GOLD/N może zostać
    # zaktualizowany nowszą rewizją AZ. Lokalne GOLD/N bez az_reuse chronimy.
    if fusion_strategy != "az_reuse":
        gold = data.get("gold_state")
        gold = gold if isinstance(gold, Mapping) else {}
        if bool(gold.get("excluded")):
            return "local_excluded"
        if bool(gold.get("approved")) or bool(gold.get("candidate")):
            return "local_gold"

    return ""


def apply_reusable_az_to_metadata(
    registry: AZRegistry,
    metadata: MutableMapping[str, Any],
    *,
    project_id: str | None = None,
) -> PZ2AZReuseSummary:
    """Nałóż właściwe istniejące AZ na bezpieczne rekordy PZ2.

    Free mode: najnowsza rewizja cropa.
    Campaign: wyłącznie jawny project_crop_az dla project_id.
    """
    if not isinstance(metadata, MutableMapping):
        raise TypeError("metadata musi być mutowalnym mapowaniem.")

    registry.initialize()
    store = AZRevisionStore(registry.database)

    scanned = resolved = applied = already_current = protected = 0
    missing_revision = invalid_crop = errors = 0

    for plate_id, data in list(metadata.items()):
        if not isinstance(data, Mapping):
            continue
        scanned += 1

        crop_id = str(data.get("crop_id") or "").strip()
        identity = str(data.get("crop_identity_sha256") or "").strip().lower()
        if not crop_id or not identity:
            invalid_crop += 1
            continue

        try:
            revision = store.resolve_reusable_az(
                crop_id=crop_id,
                project_id=project_id,
            )
        except Exception:
            errors += 1
            continue

        if not revision:
            missing_revision += 1
            continue
        resolved += 1

        reason = pz2_az_reuse_protection_reason(data, revision)
        if reason == "already_current":
            already_current += 1
            continue
        if reason:
            protected += 1
            continue

        try:
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
                invalid_crop += 1
                continue
            if str(crop["identity_sha256"] or "").strip().lower() != identity:
                invalid_crop += 1
                continue

            width = int(crop["width"] or 0)
            height = int(crop["height"] or 0)
            if width <= 0 or height <= 0:
                invalid_crop += 1
                continue

            metadata[str(plate_id)] = az_revision_to_pz2_metadata(
                data,
                revision,
                image_width=width,
                image_height=height,
            )
            applied += 1
        except Exception:
            errors += 1

    return PZ2AZReuseSummary(
        scanned=scanned,
        resolved=resolved,
        applied=applied,
        already_current=already_current,
        protected=protected,
        missing_revision=missing_revision,
        invalid_crop=invalid_crop,
        errors=errors,
    )
