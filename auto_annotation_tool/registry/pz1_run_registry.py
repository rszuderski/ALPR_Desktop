"""Adapter zakończonego runu PZ1 do logicznego rejestru cropów AZ.

AZ003B: moduł nie zna Tkintera ani CAMPAIGN. Otrzymuje gotowy katalog
``preview_dir`` z ``metadata.json`` i ``images/``. Wszystkie rekordy są
najpierw walidowane, a następnie rejestrowane w jednej transakcji SQLite.
Dopiero po udanym COMMIT metadata.json jest atomowo wzbogacane o ID registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .az_registry import AZRegistry, crop_contract_sha256, file_sha256
from .crop_identity import (
    CROP_IDENTITY_SCHEMA,
    PZ1_OUTPUT_WIDTH,
    PZ1_SINGLE_ROW_HEIGHT,
    PZ1_TWO_ROW_HEIGHT,
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


class PZ1RunRegistryError(RuntimeError):
    """Run PZ1 nie spełnia kontraktu wymaganego do rejestracji cropów."""


@dataclass(frozen=True)
class PZ1RunRegistrationReport:
    preview_dir: str
    total: int
    registered: int
    new_crops: int
    reused_crops: int
    new_artifacts: int
    reused_artifacts: int
    project_memberships: int
    iteration_memberships: int
    metadata_updated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "preview_dir": self.preview_dir,
            "total": self.total,
            "registered": self.registered,
            "new_crops": self.new_crops,
            "reused_crops": self.reused_crops,
            "new_artifacts": self.new_artifacts,
            "reused_artifacts": self.reused_artifacts,
            "project_memberships": self.project_memberships,
            "iteration_memberships": self.iteration_memberships,
            "metadata_updated": self.metadata_updated,
        }


@dataclass(frozen=True)
class _PreparedCrop:
    plate_key: str
    image_path: Path
    source_file_sha256: str
    source_image_identity: str
    source_annotation_id: str
    source_geometry_hash: str
    crop_identity_payload: dict[str, Any]
    crop_identity_sha256: str
    crop_contract_sha256: str
    artifact_sha256: str
    size_bytes: int
    width: int
    height: int


def register_pz1_preview_run(
    registry: AZRegistry,
    preview_dir: str | Path,
    *,
    interpolation: str = "lanczos4",
    project_id: str | None = None,
    iteration_num: int | None = None,
    source_mode: str = "pz1",
    source_at_ref: str | None = None,
) -> PZ1RunRegistrationReport:
    """Zarejestruj cały zakończony run PZ1 i wzbogac metadata.json.

    Wszystkie cropy są walidowane przed rozpoczęciem transakcji. Jeśli choć
    jeden rekord jest niepełny albo jego plik nie istnieje, baza nie jest
    modyfikowana i metadata.json pozostaje bez zmian.
    """
    preview = Path(preview_dir)
    metadata_path = preview / "metadata.json"
    images_dir = preview / "images"

    if not metadata_path.exists() or not metadata_path.is_file():
        raise PZ1RunRegistryError(f"Brak metadata.json: {metadata_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise PZ1RunRegistryError(f"Brak katalogu images/: {images_dir}")

    metadata = _load_metadata(metadata_path)
    prepared = _prepare_all(
        metadata,
        images_dir=images_dir,
        interpolation=interpolation,
    )

    if project_id is None and iteration_num is not None:
        raise PZ1RunRegistryError("iteration_num wymaga project_id")

    registry.initialize()
    enriched = {str(k): dict(v) for k, v in metadata.items()}

    new_crops = 0
    reused_crops = 0
    new_artifacts = 0
    reused_artifacts = 0
    project_memberships = 0
    iteration_memberships = 0

    # Jedna transakcja dla całego runu PZ1.
    with registry.database.transaction() as connection:
        for item in prepared:
            source_image_id = registry._get_or_create_source_image(
                connection,
                source_file_sha256=item.source_file_sha256,
                created_at=_timestamp_from_file(item.image_path),
            )

            crop_id, crop_created = registry._get_or_create_crop(
                connection,
                identity_schema=CROP_IDENTITY_SCHEMA,
                crop_identity_sha256=item.crop_identity_sha256,
                identity_mode=str(item.crop_identity_payload["identity_mode"]),
                source_image_id=source_image_id,
                source_annotation_id=item.source_annotation_id,
                source_geometry_hash=item.source_geometry_hash,
                crop_contract_sha256=item.crop_contract_sha256,
                width=item.width,
                height=item.height,
                created_at=_timestamp_from_file(item.image_path),
            )

            relative_path, external_path = registry._artifact_location(item.image_path)
            artifact_id, artifact_created = registry._get_or_create_artifact(
                connection,
                source_image_id=source_image_id,
                relative_path=relative_path,
                external_path=external_path,
                sha256=item.artifact_sha256,
                size_bytes=item.size_bytes,
                width=item.width,
                height=item.height,
                created_at=_timestamp_from_file(item.image_path),
            )

            registry._link_crop_artifact(
                connection,
                crop_id=crop_id,
                artifact_id=artifact_id,
                created_at=_timestamp_from_file(item.image_path),
            )

            if project_id:
                registry._attach_project(
                    connection,
                    project_id=project_id,
                    crop_id=crop_id,
                    iteration_num=iteration_num,
                    source_mode=source_mode,
                    timestamp=_timestamp_from_file(item.image_path),
                )
                project_memberships += 1

            if project_id and iteration_num is not None:
                registry._attach_iteration(
                    connection,
                    project_id=project_id,
                    iteration_num=int(iteration_num),
                    crop_id=crop_id,
                    artifact_id=artifact_id,
                    source_image_id=source_image_id,
                    source_plate_key=item.plate_key,
                    source_at_ref=source_at_ref,
                    timestamp=_timestamp_from_file(item.image_path),
                )
                iteration_memberships += 1

            if crop_created:
                new_crops += 1
            else:
                reused_crops += 1
            if artifact_created:
                new_artifacts += 1
            else:
                reused_artifacts += 1

            row = enriched[item.plate_key]
            row["crop_id"] = crop_id
            row["crop_identity_sha256"] = item.crop_identity_sha256
            row["crop_identity_schema"] = CROP_IDENTITY_SCHEMA
            row["crop_identity_mode"] = str(
                item.crop_identity_payload["identity_mode"]
            )
            row["crop_contract_sha256"] = item.crop_contract_sha256
            row["artifact_id"] = artifact_id
            row["registry_source_image_id"] = source_image_id

    metadata_updated = enriched != metadata
    if metadata_updated:
        _atomic_write_json(metadata_path, enriched)

    return PZ1RunRegistrationReport(
        preview_dir=str(preview.resolve()),
        total=len(prepared),
        registered=len(prepared),
        new_crops=new_crops,
        reused_crops=reused_crops,
        new_artifacts=new_artifacts,
        reused_artifacts=reused_artifacts,
        project_memberships=project_memberships,
        iteration_memberships=iteration_memberships,
        metadata_updated=metadata_updated,
    )


def _prepare_all(
    metadata: dict[str, dict[str, Any]],
    *,
    images_dir: Path,
    interpolation: str,
) -> list[_PreparedCrop]:
    prepared: list[_PreparedCrop] = []
    for plate_key in sorted(metadata):
        raw = metadata[plate_key]
        if not isinstance(raw, dict):
            raise PZ1RunRegistryError(
                f"metadata[{plate_key!r}] nie jest obiektem JSON"
            )

        image_path = images_dir / f"{plate_key}.jpg"
        if not image_path.exists() or not image_path.is_file():
            raise PZ1RunRegistryError(
                f"Brak cropa dla {plate_key}: {image_path}"
            )

        source_file_sha256 = _required_text(
            raw,
            "source_file_sha256",
            plate_key,
        ).lower()
        source_image_identity = str(raw.get("source_image_id") or "").strip()
        if not source_image_identity:
            source_image_identity = f"img-sha256-{source_file_sha256}"

        source_annotation_id = _required_text(
            raw,
            "source_annotation_id",
            plate_key,
        )
        source_geometry_hash = _required_text(
            raw,
            "source_geometry_hash",
            plate_key,
        ).lower()

        is_square = bool(raw.get("is_square"))
        output_height = (
            PZ1_TWO_ROW_HEIGHT
            if is_square
            else PZ1_SINGLE_ROW_HEIGHT
        )

        identity_payload = build_pz1_crop_identity(
            source_image_id=source_image_identity,
            source_annotation_id=source_annotation_id,
            source_geometry_hash=source_geometry_hash,
            rectify=True,
            do_deskew=False,
            enhance_contrast=False,
            interpolation=interpolation,
            output_width=PZ1_OUTPUT_WIDTH,
            output_height=output_height,
        )
        identity_sha = compute_crop_identity_sha256(identity_payload)
        contract_sha = crop_contract_sha256(identity_payload["crop_contract"])

        prepared.append(
            _PreparedCrop(
                plate_key=str(plate_key),
                image_path=image_path,
                source_file_sha256=source_file_sha256,
                source_image_identity=source_image_identity,
                source_annotation_id=source_annotation_id,
                source_geometry_hash=source_geometry_hash,
                crop_identity_payload=identity_payload,
                crop_identity_sha256=identity_sha,
                crop_contract_sha256=contract_sha,
                artifact_sha256=file_sha256(image_path),
                size_bytes=int(image_path.stat().st_size),
                width=PZ1_OUTPUT_WIDTH,
                height=output_height,
            )
        )

    return prepared


def _required_text(
    row: dict[str, Any],
    key: str,
    plate_key: str,
) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise PZ1RunRegistryError(
            f"Crop {plate_key} nie ma wymaganego pola {key}"
        )
    return value


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PZ1RunRegistryError(
            f"Nie można odczytać metadata.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PZ1RunRegistryError("metadata.json musi zawierać obiekt JSON")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except Exception:
            pass


def _timestamp_from_file(path: Path) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
