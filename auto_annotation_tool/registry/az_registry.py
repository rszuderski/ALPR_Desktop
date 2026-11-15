"""Rejestracja logicznych cropów i ich fizycznych artefaktów w SQLite.

AZ003A: czysta warstwa backendowa. Ten moduł nie zna Tkintera, CAMPAIGN ani
konkretnego workflow PZ1. Integracja z końcem runu PZ1 nastąpi w AZ003B.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3
import uuid
from typing import Any

from .database import RegistryDatabase
from .crop_identity import CROP_IDENTITY_SCHEMA


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLATE_CROP_KIND = "plate_crop"


@dataclass(frozen=True)
class CropRegistrationResult:
    crop_id: str
    artifact_id: str
    source_image_id: str
    crop_created: bool
    artifact_created: bool
    project_attached: bool
    iteration_attached: bool


class AZRegistry:
    """Wysokopoziomowa warstwa zapisu crop identity do registry v6."""

    def __init__(
        self,
        database: RegistryDatabase,
        *,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self.database = database
        self.workspace_dir = (
            Path(workspace_dir).resolve()
            if workspace_dir is not None
            else None
        )

    @classmethod
    def for_workspace(cls, workspace_dir: str | Path) -> "AZRegistry":
        workspace = Path(workspace_dir)
        return cls(
            RegistryDatabase(
                workspace / "_registry" / "alpr_registry.sqlite3"
            ),
            workspace_dir=workspace,
        )

    def initialize(self) -> int:
        return self.database.initialize()

    def register_crop_artifact(
        self,
        *,
        crop_identity_sha256: str,
        identity_mode: str,
        source_file_sha256: str,
        artifact_path: str | Path,
        artifact_sha256: str,
        size_bytes: int | None,
        width: int | None,
        height: int | None,
        source_annotation_id: str | None = None,
        source_geometry_hash: str | None = None,
        crop_contract_sha256: str | None = None,
        identity_schema: str = CROP_IDENTITY_SCHEMA,
        project_id: str | None = None,
        iteration_num: int | None = None,
        source_mode: str | None = None,
        source_plate_key: str | None = None,
        source_at_ref: str | None = None,
        created_at: str | None = None,
    ) -> CropRegistrationResult:
        """Zarejestruj fizyczny crop i jego logiczną tożsamość.

        Operacja jest atomowa i idempotentna dla:
        - tej samej logical identity,
        - tego samego fizycznego artifact_path + SHA + kind.

        Ten sam logical crop może mieć wiele fizycznych artefaktów.
        """
        self.initialize()

        crop_identity_sha256 = _require_sha256(
            "crop_identity_sha256",
            crop_identity_sha256,
        )
        source_file_sha256 = _require_sha256(
            "source_file_sha256",
            source_file_sha256,
        )
        artifact_sha256 = _require_sha256(
            "artifact_sha256",
            artifact_sha256,
        )
        identity_schema = _require_text(
            "identity_schema",
            identity_schema,
        )
        identity_mode = _require_text(
            "identity_mode",
            identity_mode,
        )

        if crop_contract_sha256:
            crop_contract_sha256 = _require_sha256(
                "crop_contract_sha256",
                crop_contract_sha256,
            )
        if size_bytes is not None:
            size_bytes = _nonnegative_int("size_bytes", size_bytes)
        if width is not None:
            width = _positive_int("width", width)
        if height is not None:
            height = _positive_int("height", height)

        project_id = _optional_text(project_id)
        source_annotation_id = _optional_text(source_annotation_id)
        source_geometry_hash = _optional_text(source_geometry_hash)
        source_mode = _optional_text(source_mode)
        source_plate_key = _optional_text(source_plate_key)
        source_at_ref = _optional_text(source_at_ref)

        if iteration_num is not None:
            iteration_num = _positive_int(
                "iteration_num",
                iteration_num,
            )
            if not project_id:
                raise ValueError(
                    "iteration_num wymaga project_id"
                )

        artifact = Path(artifact_path)
        relative_path, external_path = self._artifact_location(artifact)
        timestamp = str(created_at or "").strip() or _utc_now_iso()

        with self.database.transaction() as connection:
            source_image_id = self._get_or_create_source_image(
                connection,
                source_file_sha256=source_file_sha256,
                created_at=timestamp,
            )

            crop_id, crop_created = self._get_or_create_crop(
                connection,
                identity_schema=identity_schema,
                crop_identity_sha256=crop_identity_sha256,
                identity_mode=identity_mode,
                source_image_id=source_image_id,
                source_annotation_id=source_annotation_id,
                source_geometry_hash=source_geometry_hash,
                crop_contract_sha256=crop_contract_sha256,
                width=width,
                height=height,
                created_at=timestamp,
            )

            artifact_id, artifact_created = self._get_or_create_artifact(
                connection,
                source_image_id=source_image_id,
                relative_path=relative_path,
                external_path=external_path,
                sha256=artifact_sha256,
                size_bytes=size_bytes,
                width=width,
                height=height,
                created_at=timestamp,
            )

            self._link_crop_artifact(
                connection,
                crop_id=crop_id,
                artifact_id=artifact_id,
                created_at=timestamp,
            )

            project_attached = False
            iteration_attached = False

            if project_id:
                self._attach_project(
                    connection,
                    project_id=project_id,
                    crop_id=crop_id,
                    iteration_num=iteration_num,
                    source_mode=source_mode,
                    timestamp=timestamp,
                )
                project_attached = True

            if project_id and iteration_num is not None:
                self._attach_iteration(
                    connection,
                    project_id=project_id,
                    iteration_num=iteration_num,
                    crop_id=crop_id,
                    artifact_id=artifact_id,
                    source_image_id=source_image_id,
                    source_plate_key=source_plate_key,
                    source_at_ref=source_at_ref,
                    timestamp=timestamp,
                )
                iteration_attached = True

        return CropRegistrationResult(
            crop_id=crop_id,
            artifact_id=artifact_id,
            source_image_id=source_image_id,
            crop_created=crop_created,
            artifact_created=artifact_created,
            project_attached=project_attached,
            iteration_attached=iteration_attached,
        )

    def find_crop_by_identity(
        self,
        identity_sha256: str,
        *,
        identity_schema: str = CROP_IDENTITY_SCHEMA,
    ) -> dict[str, Any] | None:
        self.initialize()
        digest = _require_sha256(
            "identity_sha256",
            identity_sha256,
        )
        identity_schema = _require_text(
            "identity_schema",
            identity_schema,
        )
        with self.database.read_connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM plate_crops
                WHERE identity_schema = ?
                  AND identity_sha256 = ?
                LIMIT 1
                """,
                (identity_schema, digest),
            ).fetchone()
        return dict(row) if row is not None else None

    def _artifact_location(
        self,
        path: Path,
    ) -> tuple[str | None, str | None]:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()

        if self.workspace_dir is not None:
            try:
                relative = resolved.relative_to(
                    self.workspace_dir
                ).as_posix()
                return relative, None
            except ValueError:
                pass

        return None, str(resolved)

    @staticmethod
    def _get_or_create_source_image(
        connection: sqlite3.Connection,
        *,
        source_file_sha256: str,
        created_at: str,
    ) -> str:
        row = connection.execute(
            """
            SELECT source_image_id
            FROM source_images
            WHERE canonical_sha256 = ?
            LIMIT 1
            """,
            (source_file_sha256,),
        ).fetchone()
        if row is not None:
            return str(row["source_image_id"])

        source_image_id = source_image_id_from_sha256(
            source_file_sha256
        )
        connection.execute(
            """
            INSERT INTO source_images (
                source_image_id,
                canonical_sha256,
                origin_status,
                created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                source_image_id,
                source_file_sha256,
                "known",
                created_at,
            ),
        )
        return source_image_id

    @staticmethod
    def _get_or_create_crop(
        connection: sqlite3.Connection,
        *,
        identity_schema: str,
        crop_identity_sha256: str,
        identity_mode: str,
        source_image_id: str,
        source_annotation_id: str | None,
        source_geometry_hash: str | None,
        crop_contract_sha256: str | None,
        width: int | None,
        height: int | None,
        created_at: str,
    ) -> tuple[str, bool]:
        row = connection.execute(
            """
            SELECT
                crop_id,
                identity_mode,
                source_image_id,
                source_annotation_id,
                source_geometry_hash,
                crop_contract_sha256,
                width,
                height
            FROM plate_crops
            WHERE identity_schema = ?
              AND identity_sha256 = ?
            LIMIT 1
            """,
            (
                identity_schema,
                crop_identity_sha256,
            ),
        ).fetchone()

        if row is not None:
            AZRegistry._validate_existing_crop(
                row,
                identity_mode=identity_mode,
                source_image_id=source_image_id,
                source_annotation_id=source_annotation_id,
                source_geometry_hash=source_geometry_hash,
                crop_contract_sha256=crop_contract_sha256,
                width=width,
                height=height,
            )
            return str(row["crop_id"]), False

        crop_id = f"CROP-{uuid.uuid4().hex.upper()}"
        connection.execute(
            """
            INSERT INTO plate_crops (
                crop_id,
                identity_sha256,
                identity_mode,
                identity_schema,
                source_image_id,
                source_annotation_id,
                source_geometry_hash,
                crop_contract_sha256,
                width,
                height,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                crop_id,
                crop_identity_sha256,
                identity_mode,
                identity_schema,
                source_image_id,
                source_annotation_id,
                source_geometry_hash,
                crop_contract_sha256,
                width,
                height,
                created_at,
            ),
        )
        return crop_id, True

    @staticmethod
    def _validate_existing_crop(
        row: sqlite3.Row,
        *,
        identity_mode: str,
        source_image_id: str,
        source_annotation_id: str | None,
        source_geometry_hash: str | None,
        crop_contract_sha256: str | None,
        width: int | None,
        height: int | None,
    ) -> None:
        checks = {
            "identity_mode": identity_mode,
            "source_image_id": source_image_id,
            "source_annotation_id": source_annotation_id,
            "source_geometry_hash": source_geometry_hash,
            "crop_contract_sha256": crop_contract_sha256,
            "width": width,
            "height": height,
        }
        for key, incoming in checks.items():
            stored = row[key]
            if stored is None or incoming is None:
                continue
            if str(stored) != str(incoming):
                raise RuntimeError(
                    "Niezgodny rekord istniejącego crop identity: "
                    f"{key}: registry={stored!r}, incoming={incoming!r}"
                )

    @staticmethod
    def _get_or_create_artifact(
        connection: sqlite3.Connection,
        *,
        source_image_id: str,
        relative_path: str | None,
        external_path: str | None,
        sha256: str,
        size_bytes: int | None,
        width: int | None,
        height: int | None,
        created_at: str,
    ) -> tuple[str, bool]:
        if relative_path is not None:
            row = connection.execute(
                """
                SELECT artifact_id, source_image_id
                FROM image_artifacts
                WHERE kind = ?
                  AND relative_path = ?
                  AND sha256 = ?
                LIMIT 1
                """,
                (
                    PLATE_CROP_KIND,
                    relative_path,
                    sha256,
                ),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT artifact_id, source_image_id
                FROM image_artifacts
                WHERE kind = ?
                  AND external_path = ?
                  AND sha256 = ?
                LIMIT 1
                """,
                (
                    PLATE_CROP_KIND,
                    external_path,
                    sha256,
                ),
            ).fetchone()

        if row is not None:
            if str(row["source_image_id"]) != source_image_id:
                raise RuntimeError(
                    "Istniejący artifact path+SHA ma inne source_image_id."
                )
            return str(row["artifact_id"]), False

        artifact_id = f"ART-CROP-{uuid.uuid4().hex.upper()}"
        connection.execute(
            """
            INSERT INTO image_artifacts (
                artifact_id,
                source_image_id,
                relative_path,
                external_path,
                sha256,
                size_bytes,
                kind,
                derived_from_artifact_id,
                width,
                height,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                artifact_id,
                source_image_id,
                relative_path,
                external_path,
                sha256,
                size_bytes,
                PLATE_CROP_KIND,
                width,
                height,
                created_at,
            ),
        )
        return artifact_id, True

    @staticmethod
    def _link_crop_artifact(
        connection: sqlite3.Connection,
        *,
        crop_id: str,
        artifact_id: str,
        created_at: str,
    ) -> None:
        existing_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crop_artifacts
                WHERE crop_id = ?
                """,
                (crop_id,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO crop_artifacts (
                crop_id,
                artifact_id,
                is_primary,
                first_seen_at,
                last_seen_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(crop_id, artifact_id) DO UPDATE SET
                last_seen_at = excluded.last_seen_at
            """,
            (
                crop_id,
                artifact_id,
                1 if existing_count == 0 else 0,
                created_at,
                created_at,
            ),
        )

    @staticmethod
    def _attach_project(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        crop_id: str,
        iteration_num: int | None,
        source_mode: str | None,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO project_crop_members (
                project_id,
                crop_id,
                first_seen_iteration,
                last_seen_iteration,
                first_seen_at,
                last_seen_at,
                source_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, crop_id) DO UPDATE SET
                last_seen_iteration = COALESCE(
                    excluded.last_seen_iteration,
                    project_crop_members.last_seen_iteration
                ),
                last_seen_at = excluded.last_seen_at,
                source_mode = COALESCE(
                    excluded.source_mode,
                    project_crop_members.source_mode
                )
            """,
            (
                project_id,
                crop_id,
                iteration_num,
                iteration_num,
                timestamp,
                timestamp,
                source_mode,
            ),
        )

    @staticmethod
    def _attach_iteration(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        iteration_num: int,
        crop_id: str,
        artifact_id: str,
        source_image_id: str,
        source_plate_key: str | None,
        source_at_ref: str | None,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO iteration_crop_members (
                project_id,
                iteration_num,
                crop_id,
                artifact_id,
                source_image_id,
                source_plate_key,
                source_at_ref,
                first_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, iteration_num, crop_id) DO UPDATE SET
                artifact_id = excluded.artifact_id,
                source_image_id = excluded.source_image_id,
                source_plate_key = COALESCE(
                    excluded.source_plate_key,
                    iteration_crop_members.source_plate_key
                ),
                source_at_ref = COALESCE(
                    excluded.source_at_ref,
                    iteration_crop_members.source_at_ref
                )
            """,
            (
                project_id,
                iteration_num,
                crop_id,
                artifact_id,
                source_image_id,
                source_plate_key,
                source_at_ref,
                timestamp,
            ),
        )



def project_id_from_folder_name(folder_name: str) -> str:
    """Stabilny project_id zgodny z historycznym registry eksperymentów."""
    normalized = str(folder_name or "").strip().replace("\\", "/").casefold()
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()
    return f"PRJ-{digest[:20]}"


def ensure_campaign_project(
    registry: AZRegistry,
    *,
    project_name: str,
    folder_name: str,
) -> str:
    """Zapewnij rekord projektu kampanii w SQLite i zwróć stabilny project_id."""
    display_name = str(project_name or "").strip()
    folder = str(folder_name or "").strip()
    if not display_name:
        raise ValueError("project_name nie może być puste")
    if not folder:
        raise ValueError("folder_name nie może być puste")

    project_id = project_id_from_folder_name(folder)
    if not project_id:
        raise ValueError("Nie udało się wyliczyć project_id")

    registry.initialize()
    now = _utc_now_iso()
    with registry.database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO projects (
                project_id,
                campaign_key,
                folder_name,
                display_name,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                campaign_key = excluded.campaign_key,
                folder_name = excluded.folder_name,
                display_name = excluded.display_name,
                created_at = COALESCE(projects.created_at, excluded.created_at),
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                display_name,
                folder,
                display_name,
                now,
                now,
            ),
        )
    return project_id

def source_image_id_from_sha256(sha256: str) -> str:
    """Id zgodny z historycznym registry eksperymentów."""
    digest = _require_sha256("sha256", sha256)
    return f"SRC-SHA256-{digest.upper()}"


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_contract_sha256(crop_contract: dict[str, Any]) -> str:
    from .crop_identity import canonical_json_bytes

    if not isinstance(crop_contract, dict):
        raise TypeError("crop_contract musi być dict")
    return hashlib.sha256(
        canonical_json_bytes(crop_contract)
    ).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(name: str, value: Any) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name} musi być pełnym SHA-256 hex")
    return digest


def _require_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} nie może być puste")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} musi być dodatnim int")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} musi być dodatnim int") from exc
    if number <= 0:
        raise ValueError(f"{name} musi być > 0")
    return number


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} musi być nieujemnym int")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} musi być nieujemnym int") from exc
    if number < 0:
        raise ValueError(f"{name} musi być >= 0")
    return number
