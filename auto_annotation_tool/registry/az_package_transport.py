#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Przenośny pakiet cropów + AZ — model i read-only analyzer.

AZ009A:
- bez GUI,
- bez zapisu do SQLite,
- bez modyfikacji aktywnego PZ2,
- walidacja manifestu, crop identity, artefaktów i AZ,
- analiza kompatybilności z projektem docelowym.

Warstwa APPLY/attach pojawi się dopiero w AZ009B.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from .az_registry import AZRegistry, crop_contract_sha256
from .az_revision_store import (
    canonicalize_az_payload,
    compute_az_payload_sha256,
)
from .crop_identity import (
    CROP_IDENTITY_SCHEMA,
    compute_crop_identity_sha256,
)


AZ_PACKAGE_SCHEMA = "alpr.az_package.v1"


class AZPackageError(ValueError):
    """Nieprawidłowy kontrakt przenośnego pakietu AZ."""


@dataclass(frozen=True)
class AZPackageInvalidItem:
    index: int
    entry_id: str
    reason: str


@dataclass(frozen=True)
class AZPackageAZ:
    payload: dict[str, Any]
    payload_sha256: str
    source_kind: str
    source_status: str | None
    trust_state: str
    origin_project_id: str | None
    origin_iteration: int | None
    source_az_revision_id: str | None
    created_at: str | None


@dataclass(frozen=True)
class AZPackageCrop:
    index: int
    entry_id: str
    crop_identity: dict[str, Any]
    crop_identity_sha256: str
    crop_contract_sha256: str
    source_file_sha256: str
    artifact_relpath: str
    artifact_path: Path
    artifact_sha256: str
    size_bytes: int
    width: int
    height: int
    az: AZPackageAZ | None


@dataclass(frozen=True)
class AZPackage:
    root: Path
    package_id: str
    created_at: str
    source: dict[str, Any]
    crops: tuple[AZPackageCrop, ...]
    invalid_items: tuple[AZPackageInvalidItem, ...]


@dataclass(frozen=True)
class AZPackageAnalysisItem:
    entry_id: str
    crop_identity_sha256: str
    status: str
    reason: str
    registry_crop_id: str | None
    needs_new_crop: bool
    needs_project_attach: bool
    artifact_already_linked: bool
    package_has_az: bool
    target_has_az: bool
    az_relation: str


@dataclass(frozen=True)
class AZPackageAnalysisPlan:
    package_id: str
    target_project_id: str
    items: tuple[AZPackageAnalysisItem, ...]
    invalid_items: tuple[AZPackageInvalidItem, ...]

    @property
    def new_count(self) -> int:
        return sum(item.status == "new" for item in self.items)

    @property
    def already_present_count(self) -> int:
        return sum(item.status == "already_present" for item in self.items)

    @property
    def conflict_count(self) -> int:
        return sum(item.status == "conflict" for item in self.items)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_items)

    @property
    def with_az_count(self) -> int:
        return sum(item.package_has_az for item in self.items)

    @property
    def without_az_count(self) -> int:
        return sum(not item.package_has_az for item in self.items)


def load_az_package(package_root: str | Path) -> AZPackage:
    """Wczytaj i zweryfikuj katalogowy pakiet AZ.

    AZ009A świadomie zaczyna od katalogu. ZIP/archive transport będzie można
    dołożyć nad tym samym kontraktem bez zmiany semantyki manifestu.
    """
    root = Path(package_root)
    if not root.exists() or not root.is_dir():
        raise AZPackageError(f"Pakiet AZ nie jest katalogiem: {root}")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AZPackageError(f"Brak manifest.json: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise AZPackageError(f"Nie można odczytać manifest.json: {exc}") from exc

    if not isinstance(payload, dict):
        raise AZPackageError("manifest.json musi zawierać obiekt JSON.")
    if str(payload.get("schema") or "").strip() != AZ_PACKAGE_SCHEMA:
        raise AZPackageError(
            f"Nieobsługiwany schema: {payload.get('schema')!r}; "
            f"oczekiwano {AZ_PACKAGE_SCHEMA!r}."
        )

    package_id = _required_text("package_id", payload.get("package_id"))
    created_at = _required_text("created_at", payload.get("created_at"))
    source = payload.get("source")
    source = dict(source) if isinstance(source, Mapping) else {}

    raw_crops = payload.get("crops")
    if not isinstance(raw_crops, list):
        raise AZPackageError("crops musi być listą.")

    crops: list[AZPackageCrop] = []
    invalid: list[AZPackageInvalidItem] = []
    seen_identities: set[str] = set()

    for index, raw in enumerate(raw_crops):
        entry_id = _entry_id(raw, index)
        try:
            item = _parse_crop_entry(
                root=root,
                raw=raw,
                index=index,
                entry_id=entry_id,
            )
            if item.crop_identity_sha256 in seen_identities:
                raise AZPackageError(
                    "Pakiet zawiera drugi rekord tego samego "
                    f"crop_identity_sha256: {item.crop_identity_sha256}"
                )
            seen_identities.add(item.crop_identity_sha256)
            crops.append(item)
        except Exception as exc:
            invalid.append(
                AZPackageInvalidItem(
                    index=index,
                    entry_id=entry_id,
                    reason=str(exc),
                )
            )

    return AZPackage(
        root=root,
        package_id=package_id,
        created_at=created_at,
        source=source,
        crops=tuple(crops),
        invalid_items=tuple(invalid),
    )


def analyze_az_package(
    registry: AZRegistry,
    package: AZPackage,
    *,
    target_project_id: str,
) -> AZPackageAnalysisPlan:
    """Porównaj pakiet z targetem bez żadnego zapisu.

    Funkcja otwiera SQLite w trybie `mode=ro`; nie uruchamia migracji i nie
    tworzy rekordów.
    """
    target_project_id = _required_text(
        "target_project_id",
        target_project_id,
    )
    db_path = Path(registry.database.db_path)
    if not db_path.is_file():
        raise AZPackageError(f"Brak registry SQLite: {db_path}")

    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        project = connection.execute(
            "SELECT 1 FROM projects WHERE project_id = ?",
            (target_project_id,),
        ).fetchone()
        if project is None:
            raise AZPackageError(
                f"Nieznany target_project_id: {target_project_id}"
            )

        items = [
            _analyze_crop(
                connection,
                crop,
                target_project_id=target_project_id,
            )
            for crop in package.crops
        ]
    finally:
        connection.close()

    return AZPackageAnalysisPlan(
        package_id=package.package_id,
        target_project_id=target_project_id,
        items=tuple(items),
        invalid_items=package.invalid_items,
    )


def _parse_crop_entry(
    *,
    root: Path,
    raw: Any,
    index: int,
    entry_id: str,
) -> AZPackageCrop:
    if not isinstance(raw, Mapping):
        raise AZPackageError("crop entry musi być obiektem JSON.")

    identity = raw.get("crop_identity")
    if not isinstance(identity, Mapping):
        raise AZPackageError("brak crop_identity.")
    identity = dict(identity)

    identity_schema = str(identity.get("schema") or "").strip()
    if identity_schema != CROP_IDENTITY_SCHEMA:
        raise AZPackageError(
            f"crop_identity.schema={identity_schema!r}; "
            f"oczekiwano {CROP_IDENTITY_SCHEMA!r}."
        )

    declared_identity_sha = _sha256(
        "crop_identity_sha256",
        raw.get("crop_identity_sha256"),
    )
    computed_identity_sha = compute_crop_identity_sha256(identity)
    if declared_identity_sha != computed_identity_sha:
        raise AZPackageError(
            "crop_identity_sha256 nie odpowiada canonical crop_identity."
        )

    crop_contract = identity.get("crop_contract")
    if not isinstance(crop_contract, dict):
        raise AZPackageError("crop_identity.crop_contract musi być obiektem.")
    declared_contract_sha = _sha256(
        "crop_contract_sha256",
        raw.get("crop_contract_sha256"),
    )
    computed_contract_sha = crop_contract_sha256(crop_contract)
    if declared_contract_sha != computed_contract_sha:
        raise AZPackageError(
            "crop_contract_sha256 nie odpowiada crop_identity.crop_contract."
        )

    source_file_sha = _sha256(
        "source_file_sha256",
        raw.get("source_file_sha256"),
    )
    artifact = raw.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AZPackageError("brak artifact.")

    relpath = _safe_relpath(artifact.get("path"))
    artifact_path = root.joinpath(*PurePosixPath(relpath).parts)
    try:
        artifact_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AZPackageError(
            "artifact.path wychodzi poza katalog pakietu."
        ) from exc
    if not artifact_path.is_file():
        raise AZPackageError(f"Brak pliku artefaktu: {relpath}")

    width = _positive_int("artifact.width", artifact.get("width"))
    height = _positive_int("artifact.height", artifact.get("height"))
    size_bytes = _nonnegative_int(
        "artifact.size_bytes",
        artifact.get("size_bytes"),
    )
    declared_artifact_sha = _sha256(
        "artifact.sha256",
        artifact.get("sha256"),
    )

    actual_size = int(artifact_path.stat().st_size)
    if actual_size != size_bytes:
        raise AZPackageError(
            f"artifact.size_bytes={size_bytes}, rzeczywisty={actual_size}."
        )
    actual_sha = _file_sha256(artifact_path)
    if actual_sha != declared_artifact_sha:
        raise AZPackageError("artifact.sha256 nie odpowiada plikowi.")

    contract_w = _positive_int(
        "crop_contract.output_width",
        crop_contract.get("output_width"),
    )
    contract_h = _positive_int(
        "crop_contract.output_height",
        crop_contract.get("output_height"),
    )
    if width != contract_w or height != contract_h:
        raise AZPackageError(
            "Wymiary artifact nie odpowiadają crop_contract."
        )

    az = _parse_az(raw.get("az"), declared_identity_sha)

    return AZPackageCrop(
        index=index,
        entry_id=entry_id,
        crop_identity=identity,
        crop_identity_sha256=declared_identity_sha,
        crop_contract_sha256=declared_contract_sha,
        source_file_sha256=source_file_sha,
        artifact_relpath=relpath,
        artifact_path=artifact_path,
        artifact_sha256=declared_artifact_sha,
        size_bytes=size_bytes,
        width=width,
        height=height,
        az=az,
    )


def _parse_az(raw: Any, identity_sha: str) -> AZPackageAZ | None:
    if raw in (None, {}):
        return None
    if not isinstance(raw, Mapping):
        raise AZPackageError("az musi być obiektem albo null.")

    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        raise AZPackageError("az.payload musi być obiektem.")

    canonical = canonicalize_az_payload(payload)
    if canonical["crop_identity_sha256"] != identity_sha:
        raise AZPackageError(
            "az.payload.crop_identity_sha256 nie odpowiada crop entry."
        )

    declared_sha = _sha256(
        "az.payload_sha256",
        raw.get("payload_sha256"),
    )
    computed_sha = compute_az_payload_sha256(canonical)
    if declared_sha != computed_sha:
        raise AZPackageError(
            "az.payload_sha256 nie odpowiada canonical AZ payload."
        )

    origin_iteration = raw.get("origin_iteration")
    if origin_iteration is not None:
        origin_iteration = _positive_int(
            "az.origin_iteration",
            origin_iteration,
        )

    return AZPackageAZ(
        payload=canonical,
        payload_sha256=declared_sha,
        source_kind=_required_text(
            "az.source_kind",
            raw.get("source_kind"),
        ),
        source_status=_optional_text(raw.get("source_status")),
        trust_state=_required_text(
            "az.trust_state",
            raw.get("trust_state"),
        ),
        origin_project_id=_optional_text(raw.get("origin_project_id")),
        origin_iteration=origin_iteration,
        source_az_revision_id=_optional_text(
            raw.get("source_az_revision_id")
        ),
        created_at=_optional_text(raw.get("created_at")),
    )


def _analyze_crop(
    connection: sqlite3.Connection,
    crop: AZPackageCrop,
    *,
    target_project_id: str,
) -> AZPackageAnalysisItem:
    row = connection.execute(
        """
        SELECT
            pc.crop_id,
            pc.identity_mode,
            pc.source_image_id,
            pc.source_annotation_id,
            pc.source_geometry_hash,
            pc.crop_contract_sha256,
            pc.width,
            pc.height,
            si.canonical_sha256 AS source_file_sha256
        FROM plate_crops pc
        JOIN source_images si
          ON si.source_image_id = pc.source_image_id
        WHERE pc.identity_schema = ?
          AND pc.identity_sha256 = ?
        LIMIT 1
        """,
        (
            CROP_IDENTITY_SCHEMA,
            crop.crop_identity_sha256,
        ),
    ).fetchone()

    if row is None:
        return AZPackageAnalysisItem(
            entry_id=crop.entry_id,
            crop_identity_sha256=crop.crop_identity_sha256,
            status="new",
            reason="Nowy logical crop; wymaga rejestracji i attach do targetu.",
            registry_crop_id=None,
            needs_new_crop=True,
            needs_project_attach=True,
            artifact_already_linked=False,
            package_has_az=crop.az is not None,
            target_has_az=False,
            az_relation="package_only" if crop.az is not None else "none",
        )

    conflicts = _registry_identity_conflicts(row, crop)
    crop_id = str(row["crop_id"])
    if conflicts:
        return AZPackageAnalysisItem(
            entry_id=crop.entry_id,
            crop_identity_sha256=crop.crop_identity_sha256,
            status="conflict",
            reason="; ".join(conflicts),
            registry_crop_id=crop_id,
            needs_new_crop=False,
            needs_project_attach=False,
            artifact_already_linked=False,
            package_has_az=crop.az is not None,
            target_has_az=False,
            az_relation="identity_conflict",
        )

    artifact_link = connection.execute(
        """
        SELECT 1
        FROM crop_artifacts ca
        JOIN image_artifacts ia
          ON ia.artifact_id = ca.artifact_id
        WHERE ca.crop_id = ?
          AND ia.sha256 = ?
        LIMIT 1
        """,
        (crop_id, crop.artifact_sha256),
    ).fetchone()
    artifact_already_linked = artifact_link is not None

    member = connection.execute(
        """
        SELECT 1
        FROM project_crop_members
        WHERE project_id = ? AND crop_id = ?
        """,
        (target_project_id, crop_id),
    ).fetchone()

    target_az = connection.execute(
        """
        SELECT
            pca.az_revision_id,
            ar.payload_sha256
        FROM project_crop_az pca
        JOIN az_revisions ar
          ON ar.az_revision_id = pca.az_revision_id
        WHERE pca.project_id = ?
          AND pca.crop_id = ?
        """,
        (target_project_id, crop_id),
    ).fetchone()

    target_has_az = target_az is not None
    az_relation = "none"
    if crop.az is not None and target_az is None:
        az_relation = "package_only"
    elif crop.az is None and target_az is not None:
        az_relation = "target_only"
    elif crop.az is not None and target_az is not None:
        if str(target_az["payload_sha256"]) == crop.az.payload_sha256:
            az_relation = "same"
        else:
            az_relation = "different_target_preserved"

    if member is not None:
        return AZPackageAnalysisItem(
            entry_id=crop.entry_id,
            crop_identity_sha256=crop.crop_identity_sha256,
            status="already_present",
            reason=(
                "Crop już należy do targetu; append go nie dubluje. "
                "Lokalne AZ targetu ma pierwszeństwo."
            ),
            registry_crop_id=crop_id,
            needs_new_crop=False,
            needs_project_attach=False,
            artifact_already_linked=artifact_already_linked,
            package_has_az=crop.az is not None,
            target_has_az=target_has_az,
            az_relation=az_relation,
        )

    return AZPackageAnalysisItem(
        entry_id=crop.entry_id,
        crop_identity_sha256=crop.crop_identity_sha256,
        status="new",
        reason=(
            "Logical crop istnieje globalnie, ale nie należy jeszcze do targetu."
        ),
        registry_crop_id=crop_id,
        needs_new_crop=False,
        needs_project_attach=True,
        artifact_already_linked=artifact_already_linked,
        package_has_az=crop.az is not None,
        target_has_az=target_has_az,
        az_relation=az_relation,
    )


def _registry_identity_conflicts(
    row: sqlite3.Row,
    crop: AZPackageCrop,
) -> list[str]:
    identity = crop.crop_identity
    expected = {
        "identity_mode": str(identity.get("identity_mode") or "").strip(),
        "source_annotation_id": _optional_text(
            identity.get("source_annotation_id")
        ),
        "source_geometry_hash": _optional_text(
            identity.get("source_geometry_hash")
        ),
        "crop_contract_sha256": crop.crop_contract_sha256,
        "width": crop.width,
        "height": crop.height,
        "source_file_sha256": crop.source_file_sha256,
    }

    conflicts = []
    for key, wanted in expected.items():
        stored = row[key]
        if stored is None or wanted is None:
            continue
        if str(stored).lower() != str(wanted).lower():
            conflicts.append(
                f"{key}: registry={stored!r}, package={wanted!r}"
            )
    return conflicts


def _entry_id(raw: Any, index: int) -> str:
    if isinstance(raw, Mapping):
        value = str(raw.get("entry_id") or "").strip()
        if value:
            return value
    return f"crop[{index}]"


def _safe_relpath(value: Any) -> str:
    text = _required_text("artifact.path", value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts:
        raise AZPackageError("artifact.path musi być ścieżką względną.")
    if ":" in path.parts[0]:
        raise AZPackageError(
            "artifact.path nie może zawierać prefiksu dysku/URI."
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AZPackageError(
            "artifact.path zawiera niedozwolony segment."
        )
    return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(name: str, value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise AZPackageError(f"{name} musi być pełnym SHA-256 hex.")
    return text


def _required_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise AZPackageError(f"{name} nie może być puste.")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise AZPackageError(f"{name} musi być dodatnim int.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AZPackageError(f"{name} musi być dodatnim int.") from exc
    if number <= 0:
        raise AZPackageError(f"{name} musi być > 0.")
    return number


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise AZPackageError(f"{name} musi być nieujemnym int.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AZPackageError(f"{name} musi być nieujemnym int.") from exc
    if number < 0:
        raise AZPackageError(f"{name} musi być >= 0.")
    return number

@dataclass(frozen=True)
class AZPackageAppendItemResult:
    entry_id: str
    crop_identity_sha256: str
    crop_id: str
    artifact_id: str
    artifact_path: str
    az_revision_id: str | None
    az_bound: bool


@dataclass(frozen=True)
class AZPackageAppendResult:
    package_id: str
    target_project_id: str
    iteration_num: int
    destination_dir: str
    planned_new: int
    appended: int
    already_present: int
    conflicts: int
    invalid: int
    az_bound: int
    without_az: int
    new_crops: int
    reused_crops: int
    new_artifacts: int
    reused_artifacts: int
    items: tuple[AZPackageAppendItemResult, ...]


def append_az_package(
    registry: AZRegistry,
    package: AZPackage,
    *,
    target_project_id: str,
    iteration_num: int,
    target_artifact_dir: str | Path,
    source_mode: str = "az_package_append",
    effective_status: str = "imported_pending_review",
) -> AZPackageAppendResult:
    """Dopnij nowe cropy z pakietu do istniejącego targetu.

    To jest APPEND, nigdy replacement:
    - already_present = no-op,
    - conflict/invalid = pominięte bez nadpisania,
    - wszystkie zaakceptowane new zapisują się w jednej transakcji SQLite,
    - metadata.json aktywnego PZ2 nie jest tutaj modyfikowany (AZ009C).
    """
    registry.initialize()
    target_project_id = _required_text(
        "target_project_id",
        target_project_id,
    )
    iteration_num = _positive_int("iteration_num", iteration_num)
    source_mode = _required_text("source_mode", source_mode)
    effective_status = _required_text(
        "effective_status",
        effective_status,
    )

    plan = analyze_az_package(
        registry,
        package,
        target_project_id=target_project_id,
    )
    plan_by_identity = {
        item.crop_identity_sha256: item
        for item in plan.items
    }
    accepted = [
        crop
        for crop in package.crops
        if plan_by_identity[crop.crop_identity_sha256].status == "new"
    ]

    destination = Path(target_artifact_dir)
    materialized: dict[str, Path] = {}
    created_paths: list[Path] = []

    appended_items: list[AZPackageAppendItemResult] = []
    new_crops = 0
    reused_crops = 0
    new_artifacts = 0
    reused_artifacts = 0
    az_bound_count = 0
    without_az = 0

    try:
        if accepted:
            destination.mkdir(parents=True, exist_ok=True)
            for crop in accepted:
                final_path, created = _materialize_package_artifact(
                    crop,
                    destination,
                )
                materialized[crop.crop_identity_sha256] = final_path
                if created:
                    created_paths.append(final_path)

        timestamp = datetime.now(timezone.utc).isoformat()

        if accepted:
            with registry.database.transaction() as connection:
                project = connection.execute(
                    "SELECT 1 FROM projects WHERE project_id = ?",
                    (target_project_id,),
                ).fetchone()
                if project is None:
                    raise AZPackageError(
                        f"Nieznany target_project_id: {target_project_id}"
                    )

                for crop in accepted:
                    existing = connection.execute(
                        """
                        SELECT crop_id
                        FROM plate_crops
                        WHERE identity_schema = ?
                          AND identity_sha256 = ?
                        LIMIT 1
                        """,
                        (
                            CROP_IDENTITY_SCHEMA,
                            crop.crop_identity_sha256,
                        ),
                    ).fetchone()

                    if existing is not None:
                        already_member = connection.execute(
                            """
                            SELECT 1
                            FROM project_crop_members
                            WHERE project_id = ? AND crop_id = ?
                            """,
                            (
                                target_project_id,
                                str(existing["crop_id"]),
                            ),
                        ).fetchone()
                        if already_member is not None:
                            continue

                    source_image_id = registry._get_or_create_source_image(
                        connection,
                        source_file_sha256=crop.source_file_sha256,
                        created_at=timestamp,
                    )

                    identity = crop.crop_identity
                    crop_id, crop_created = registry._get_or_create_crop(
                        connection,
                        identity_schema=CROP_IDENTITY_SCHEMA,
                        crop_identity_sha256=crop.crop_identity_sha256,
                        identity_mode=str(
                            identity.get("identity_mode") or ""
                        ).strip(),
                        source_image_id=source_image_id,
                        source_annotation_id=_optional_text(
                            identity.get("source_annotation_id")
                        ),
                        source_geometry_hash=_optional_text(
                            identity.get("source_geometry_hash")
                        ),
                        crop_contract_sha256=crop.crop_contract_sha256,
                        width=crop.width,
                        height=crop.height,
                        created_at=timestamp,
                    )
                    if crop_created:
                        new_crops += 1
                    else:
                        reused_crops += 1

                    artifact_path = materialized[
                        crop.crop_identity_sha256
                    ]
                    relative_path, external_path = registry._artifact_location(
                        artifact_path
                    )
                    artifact_id, artifact_created = (
                        registry._get_or_create_artifact(
                            connection,
                            source_image_id=source_image_id,
                            relative_path=relative_path,
                            external_path=external_path,
                            sha256=crop.artifact_sha256,
                            size_bytes=crop.size_bytes,
                            width=crop.width,
                            height=crop.height,
                            created_at=timestamp,
                        )
                    )
                    if artifact_created:
                        new_artifacts += 1
                    else:
                        reused_artifacts += 1

                    registry._link_crop_artifact(
                        connection,
                        crop_id=crop_id,
                        artifact_id=artifact_id,
                        created_at=timestamp,
                    )
                    registry._attach_project(
                        connection,
                        project_id=target_project_id,
                        crop_id=crop_id,
                        iteration_num=iteration_num,
                        source_mode=source_mode,
                        timestamp=timestamp,
                    )
                    registry._attach_iteration(
                        connection,
                        project_id=target_project_id,
                        iteration_num=iteration_num,
                        crop_id=crop_id,
                        artifact_id=artifact_id,
                        source_image_id=source_image_id,
                        source_plate_key=crop.entry_id,
                        source_at_ref=f"azpkg:{package.package_id}",
                        timestamp=timestamp,
                    )

                    az_revision_id = None
                    az_bound = False
                    if crop.az is not None:
                        az_revision_id, az_bound = _bind_transported_az(
                            connection,
                            package=package,
                            crop=crop,
                            crop_id=crop_id,
                            target_project_id=target_project_id,
                            effective_status=effective_status,
                            timestamp=timestamp,
                        )
                        if az_bound:
                            az_bound_count += 1
                    else:
                        without_az += 1

                    appended_items.append(
                        AZPackageAppendItemResult(
                            entry_id=crop.entry_id,
                            crop_identity_sha256=(
                                crop.crop_identity_sha256
                            ),
                            crop_id=crop_id,
                            artifact_id=artifact_id,
                            artifact_path=str(artifact_path),
                            az_revision_id=az_revision_id,
                            az_bound=az_bound,
                        )
                    )

        used_paths = {
            Path(item.artifact_path).resolve()
            for item in appended_items
        }
        for path in list(created_paths):
            try:
                if path.resolve() not in used_paths and path.exists():
                    path.unlink()
            except Exception:
                pass

        return AZPackageAppendResult(
            package_id=package.package_id,
            target_project_id=target_project_id,
            iteration_num=iteration_num,
            destination_dir=str(destination),
            planned_new=plan.new_count,
            appended=len(appended_items),
            already_present=(
                plan.already_present_count
                + max(0, plan.new_count - len(appended_items))
            ),
            conflicts=plan.conflict_count,
            invalid=plan.invalid_count,
            az_bound=az_bound_count,
            without_az=without_az,
            new_crops=new_crops,
            reused_crops=reused_crops,
            new_artifacts=new_artifacts,
            reused_artifacts=reused_artifacts,
            items=tuple(appended_items),
        )

    except Exception:
        for path in reversed(created_paths):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        raise


def _materialize_package_artifact(
    crop: AZPackageCrop,
    destination: Path,
) -> tuple[Path, bool]:
    suffix = crop.artifact_path.suffix.lower() or ".bin"
    final_path = destination / (
        f"crop_{crop.crop_identity_sha256}{suffix}"
    )

    if final_path.exists():
        if not final_path.is_file():
            raise AZPackageError(
                f"Docelowa ścieżka nie jest plikiem: {final_path}"
            )
        if _file_sha256(final_path) != crop.artifact_sha256:
            raise AZPackageError(
                "Kolizja pliku targetu: ten sam identity name ma inne SHA."
            )
        return final_path, False

    temp_path = final_path.with_name(
        final_path.name + f".tmp-{uuid.uuid4().hex}"
    )
    try:
        shutil.copy2(crop.artifact_path, temp_path)
        if _file_sha256(temp_path) != crop.artifact_sha256:
            raise AZPackageError(
                "SHA skopiowanego artefaktu różni się od pakietu."
            )
        temp_path.replace(final_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
    return final_path, True


def _bind_transported_az(
    connection: sqlite3.Connection,
    *,
    package: AZPackage,
    crop: AZPackageCrop,
    crop_id: str,
    target_project_id: str,
    effective_status: str,
    timestamp: str,
) -> tuple[str | None, bool]:
    if crop.az is None:
        return None, False

    current = connection.execute(
        """
        SELECT az_revision_id
        FROM project_crop_az
        WHERE project_id = ? AND crop_id = ?
        """,
        (target_project_id, crop_id),
    ).fetchone()
    if current is not None:
        return str(current["az_revision_id"]), False

    existing_revision = connection.execute(
        """
        SELECT az_revision_id
        FROM az_revisions
        WHERE crop_id = ? AND payload_sha256 = ?
        LIMIT 1
        """,
        (crop_id, crop.az.payload_sha256),
    ).fetchone()

    if existing_revision is not None:
        az_revision_id = str(existing_revision["az_revision_id"])
    else:
        origin_project_id = _ensure_external_origin_project(
            connection,
            package=package,
            origin_project_id=crop.az.origin_project_id,
            timestamp=timestamp,
        )
        az_revision_id = f"AZR-{uuid.uuid4().hex.upper()}"
        payload_json = json.dumps(
            crop.az.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
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
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                az_revision_id,
                crop_id,
                crop.az.payload_sha256,
                payload_json,
                "package_import:"
                + str(crop.az.source_kind or "unknown").strip(),
                crop.az.source_status,
                "external_pending_review",
                origin_project_id,
                crop.az.origin_iteration,
                timestamp,
            ),
        )

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
            target_project_id,
            crop_id,
            az_revision_id,
            effective_status,
            timestamp,
        ),
    )
    return az_revision_id, True


def _ensure_external_origin_project(
    connection: sqlite3.Connection,
    *,
    package: AZPackage,
    origin_project_id: str | None,
    timestamp: str,
) -> str | None:
    origin_project_id = _optional_text(origin_project_id)
    if not origin_project_id:
        return None

    row = connection.execute(
        "SELECT 1 FROM projects WHERE project_id = ?",
        (origin_project_id,),
    ).fetchone()
    if row is not None:
        return origin_project_id

    source = package.source if isinstance(package.source, dict) else {}
    declared_project_id = str(
        source.get("project_id") or ""
    ).strip()
    if declared_project_id == origin_project_id:
        display_name = (
            str(source.get("project_name") or "").strip()
            or str(source.get("display_name") or "").strip()
            or origin_project_id
        )
        folder_name = _optional_text(source.get("folder_name"))
        campaign_key = _optional_text(source.get("campaign_key"))
    else:
        display_name = origin_project_id
        folder_name = None
        campaign_key = None

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
        """,
        (
            origin_project_id,
            campaign_key,
            folder_name,
            display_name,
            timestamp,
            timestamp,
        ),
    )
    return origin_project_id
