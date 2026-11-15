#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wersjonowany magazyn anotacji znaków (AZ) dla logicznych cropów.

AZ004A: czysty backend SQLite. Moduł nie zna Tkintera ani metadata.json PZ2.
Adapter metadata PZ2 -> canonical AZ payload zostanie podpięty w AZ004B.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
import uuid
from typing import Any, Mapping

from .database import RegistryDatabase


AZ_PAYLOAD_SCHEMA = "alpr.az_revision.v1"


@dataclass(frozen=True)
class AZRevisionWriteResult:
    az_revision_id: str
    crop_id: str
    payload_sha256: str
    created: bool
    parent_revision_id: str | None
    project_id: str | None
    effective_status: str | None


def canonicalize_az_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Zbuduj stabilny semantyczny payload AZ.

    Payload nie zawiera transient GUI state ani danych transportowych.
    Bbox w canonical AZ jest znormalizowany do [0, 1].
    """
    if not isinstance(payload, Mapping):
        raise TypeError("AZ payload musi być mapowaniem.")

    crop_identity_sha256 = str(
        payload.get("crop_identity_sha256") or ""
    ).strip().lower()
    if len(crop_identity_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in crop_identity_sha256
    ):
        raise ValueError("crop_identity_sha256 musi być pełnym SHA-256 hex.")

    raw_chars = payload.get("characters")
    if raw_chars is None:
        raw_chars = []
    if not isinstance(raw_chars, list):
        raise ValueError("characters musi być listą.")

    characters = [
        _canonicalize_character(item, index)
        for index, item in enumerate(raw_chars)
    ]
    characters.sort(
        key=lambda item: (
            int(item["row"]),
            round((item["bbox"][0] + item["bbox"][2]) / 2.0, 8),
            round((item["bbox"][1] + item["bbox"][3]) / 2.0, 8),
            item["character"],
        )
    )

    layout_raw = payload.get("layout")
    layout_raw = layout_raw if isinstance(layout_raw, Mapping) else {}
    layout = {
        "kind": str(layout_raw.get("kind") or "").strip(),
        "confirmed": bool(layout_raw.get("confirmed", False)),
    }

    gold_raw = payload.get("gold_state")
    gold_raw = gold_raw if isinstance(gold_raw, Mapping) else {}
    gold_state = {
        "approved": bool(gold_raw.get("approved", False)),
        "excluded": bool(gold_raw.get("excluded", False)),
        "candidate": bool(gold_raw.get("candidate", False)),
    }

    status = str(payload.get("status") or "unknown").strip().lower() or "unknown"
    expected_text = str(payload.get("expected_text") or "").strip().upper()

    result = {
        "schema": AZ_PAYLOAD_SCHEMA,
        "crop_identity_sha256": crop_identity_sha256,
        "characters": characters,
        "layout": layout,
        "gold_state": gold_state,
        "status": status,
    }
    if expected_text:
        result["expected_text"] = expected_text
    return result


def canonical_az_payload_json(payload: Mapping[str, Any]) -> str:
    canonical = canonicalize_az_payload(payload)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_az_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = canonical_az_payload_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AZRevisionStore:
    """Trwałe rewizje AZ i project binding."""

    def __init__(self, database: RegistryDatabase) -> None:
        self.database = database

    def initialize(self) -> int:
        return self.database.initialize()

    def save_revision(
        self,
        *,
        crop_id: str,
        payload: Mapping[str, Any],
        source_kind: str,
        trust_state: str,
        source_status: str | None = None,
        origin_project_id: str | None = None,
        origin_iteration: int | None = None,
        parent_revision_id: str | None = None,
        bind_project_id: str | None = None,
        effective_status: str | None = None,
        created_at: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> AZRevisionWriteResult:
        # Repairs can recheck preconditions and write under one caller-owned
        # transaction. Ordinary callers retain initialization and commit rules.
        if connection is None:
            self.initialize()
        elif not connection.in_transaction:
            raise ValueError("connection wymaga aktywnej transakcji wywołującego.")

        crop_id = _required_text("crop_id", crop_id)
        source_kind = _required_text("source_kind", source_kind)
        trust_state = _required_text("trust_state", trust_state)
        source_status = _optional_text(source_status)
        origin_project_id = _optional_text(origin_project_id)
        bind_project_id = _optional_text(bind_project_id)
        effective_status = _optional_text(effective_status)
        parent_revision_id = _optional_text(parent_revision_id)

        if origin_iteration is not None:
            origin_iteration = _positive_int("origin_iteration", origin_iteration)

        if bind_project_id and not effective_status:
            raise ValueError("bind_project_id wymaga effective_status.")

        canonical = canonicalize_az_payload(payload)
        payload_json = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload_sha256 = hashlib.sha256(
            payload_json.encode("utf-8")
        ).hexdigest()
        timestamp = str(created_at or "").strip() or _utc_now_iso()

        with (self.database.transaction() if connection is None else nullcontext(connection)) as connection:
            crop_row = connection.execute(
                """
                SELECT crop_id, identity_sha256
                FROM plate_crops
                WHERE crop_id = ?
                """,
                (crop_id,),
            ).fetchone()
            if crop_row is None:
                raise ValueError(f"Nieznany crop_id: {crop_id}")

            if str(crop_row["identity_sha256"]).lower() != str(
                canonical["crop_identity_sha256"]
            ).lower():
                raise ValueError(
                    "crop_identity_sha256 payloadu nie odpowiada plate_crops."
                )

            if origin_project_id:
                _require_project(connection, origin_project_id)
            if bind_project_id:
                _require_project(connection, bind_project_id)

            if parent_revision_id:
                _require_parent_for_crop(
                    connection,
                    crop_id=crop_id,
                    parent_revision_id=parent_revision_id,
                )
            elif bind_project_id:
                current = connection.execute(
                    """
                    SELECT az_revision_id
                    FROM project_crop_az
                    WHERE project_id = ? AND crop_id = ?
                    """,
                    (bind_project_id, crop_id),
                ).fetchone()
                if current is not None:
                    parent_revision_id = str(current["az_revision_id"])

            existing = connection.execute(
                """
                SELECT az_revision_id, parent_revision_id
                FROM az_revisions
                WHERE crop_id = ? AND payload_sha256 = ?
                LIMIT 1
                """,
                (crop_id, payload_sha256),
            ).fetchone()

            created = False
            if existing is not None:
                az_revision_id = str(existing["az_revision_id"])
                stored_parent = _optional_text(existing["parent_revision_id"])
                parent_revision_id = stored_parent
            else:
                az_revision_id = f"AZR-{uuid.uuid4().hex.upper()}"
                connection.execute(
                    """
                    INSERT INTO az_revisions (
                        az_revision_id,
                        crop_id,
                        parent_revision_id,
                        payload_sha256,
                        payload_json,
                        source_kind,
                        source_status,
                        trust_state,
                        origin_project_id,
                        origin_iteration,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        az_revision_id,
                        crop_id,
                        parent_revision_id,
                        payload_sha256,
                        payload_json,
                        source_kind,
                        source_status,
                        trust_state,
                        origin_project_id,
                        origin_iteration,
                        timestamp,
                    ),
                )
                created = True

            if bind_project_id:
                connection.execute(
                    """
                    INSERT INTO project_crop_az (
                        project_id,
                        crop_id,
                        az_revision_id,
                        effective_status,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, crop_id) DO UPDATE SET
                        az_revision_id = excluded.az_revision_id,
                        effective_status = excluded.effective_status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        bind_project_id,
                        crop_id,
                        az_revision_id,
                        effective_status,
                        timestamp,
                    ),
                )

        return AZRevisionWriteResult(
            az_revision_id=az_revision_id,
            crop_id=crop_id,
            payload_sha256=payload_sha256,
            created=created,
            parent_revision_id=parent_revision_id,
            project_id=bind_project_id,
            effective_status=effective_status if bind_project_id else None,
        )

    def get_revision(self, az_revision_id: str) -> dict[str, Any] | None:
        self.initialize()
        revision_id = _required_text("az_revision_id", az_revision_id)
        with self.database.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM az_revisions WHERE az_revision_id = ?",
                (revision_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(str(result["payload_json"]))
        return result

    def get_project_az(
        self,
        *,
        project_id: str,
        crop_id: str,
    ) -> dict[str, Any] | None:
        self.initialize()
        project_id = _required_text("project_id", project_id)
        crop_id = _required_text("crop_id", crop_id)
        with self.database.read_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    pca.project_id,
                    pca.crop_id,
                    pca.az_revision_id,
                    pca.effective_status,
                    pca.updated_at,
                    ar.parent_revision_id,
                    ar.payload_sha256,
                    ar.payload_json,
                    ar.source_kind,
                    ar.source_status,
                    ar.trust_state,
                    ar.origin_project_id,
                    ar.origin_iteration,
                    ar.created_at
                FROM project_crop_az pca
                JOIN az_revisions ar
                  ON ar.az_revision_id = pca.az_revision_id
                WHERE pca.project_id = ? AND pca.crop_id = ?
                """,
                (project_id, crop_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(str(result["payload_json"]))
        return result

    def get_latest_crop_az(
        self,
        *,
        crop_id: str,
    ) -> dict[str, Any] | None:
        """Zwróć najnowszą znaną rewizję AZ dla cropa.

        To jest resolver trybu swobodnego: AZ jest zasobem globalnym,
        bez project binding. Pochodzenie rewizji pozostaje zachowane w
        origin_project_id/origin_iteration i nie wpływa na wybór.
        """
        self.initialize()
        crop_id = _required_text("crop_id", crop_id)
        with self.database.read_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    ar.az_revision_id,
                    ar.crop_id,
                    ar.parent_revision_id,
                    ar.payload_sha256,
                    ar.payload_json,
                    ar.source_kind,
                    ar.source_status,
                    ar.trust_state,
                    ar.origin_project_id,
                    ar.origin_iteration,
                    ar.created_at
                FROM az_revisions ar
                WHERE ar.crop_id = ?
                ORDER BY ar.created_at DESC, ar.rowid DESC
                LIMIT 1
                """,
                (crop_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(str(result["payload_json"]))
        result["project_id"] = None
        result["effective_status"] = None
        return result

    def resolve_reusable_az(
        self,
        *,
        crop_id: str,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Wybierz AZ do automatycznego reuse w PZ2.

        Kampania jest ściśle project-scoped: używamy wyłącznie jawnego
        project_crop_az i nie wykonujemy cichego fallbacku do rewizji
        globalnej/innego projektu. Tryb swobodny używa najnowszej rewizji
        znanej dla logicznego cropa.
        """
        crop_id = _required_text("crop_id", crop_id)
        normalized_project_id = _optional_text(project_id)
        if normalized_project_id:
            return self.get_project_az(
                project_id=normalized_project_id,
                crop_id=crop_id,
            )
        return self.get_latest_crop_az(crop_id=crop_id)


def _canonicalize_character(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"characters[{index}] musi być mapowaniem.")

    character = str(
        value.get("character")
        or value.get("text")
        or value.get("char")
        or ""
    ).strip().upper()
    if not character:
        raise ValueError(f"characters[{index}].character jest puste.")

    bbox = value.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError(f"characters[{index}].bbox musi mieć 4 wartości.")
    try:
        coords = [round(float(v), 8) for v in bbox[:4]]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"characters[{index}].bbox zawiera nieprawidłowe liczby."
        ) from exc
    if not all(math.isfinite(v) for v in coords):
        raise ValueError(f"characters[{index}].bbox zawiera NaN/Inf.")
    x1, y1, x2, y2 = coords
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            f"characters[{index}].bbox musi być znormalizowany do [0,1]."
        )

    row_raw = value.get("row", 0)
    try:
        row = int(row_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"characters[{index}].row musi być int.") from exc
    if row < 0:
        raise ValueError(f"characters[{index}].row nie może być ujemny.")

    confidence_raw = value.get("confidence", 1.0)
    try:
        confidence = round(float(confidence_raw), 8)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"characters[{index}].confidence musi być liczbą."
        ) from exc
    if not math.isfinite(confidence):
        raise ValueError(
            f"characters[{index}].confidence zawiera NaN/Inf."
        )

    return {
        "character": character,
        "bbox": coords,
        "row": row,
        "method": str(value.get("method") or "").strip(),
        "source_kind": str(value.get("source_kind") or "").strip(),
        "confidence": confidence,
    }


def _require_project(connection: sqlite3.Connection, project_id: str) -> None:
    row = connection.execute(
        "SELECT 1 FROM projects WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Nieznany project_id: {project_id}")


def _require_parent_for_crop(
    connection: sqlite3.Connection,
    *,
    crop_id: str,
    parent_revision_id: str,
) -> None:
    row = connection.execute(
        """
        SELECT crop_id
        FROM az_revisions
        WHERE az_revision_id = ?
        """,
        (parent_revision_id,),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"Nieznany parent_revision_id: {parent_revision_id}"
        )
    if str(row["crop_id"]) != crop_id:
        raise ValueError("parent_revision_id należy do innego cropa.")


def _required_text(name: str, value: Any) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} nie może być puste.")
    return result


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} musi być dodatnim int.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} musi być dodatnim int.") from exc
    if result <= 0:
        raise ValueError(f"{name} musi być > 0.")
    return result


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
