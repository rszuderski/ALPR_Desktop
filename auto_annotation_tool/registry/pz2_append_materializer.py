#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AZ009C — materializacja appendu registry do aktywnego metadata PZ2.

AZ009B zapisuje crop/artifact/project/iteration/AZ.
AZ009C rozszerza TEN SAM aktywny run PZ2 o rekordy już zapisane w registry.

Kontrakt:
- append, nigdy replacement,
- istniejące rekordy metadata nie są modyfikowane,
- nowe cropy muszą już być w images/ aktywnego runu,
- imported_pending_review nie dziedziczy OK/GOLD,
- zapis metadata jest atomowy i poprzedzony backupem,
- operacja jest idempotentna,
- recovery-safe: po restarcie potrafi odtworzyć kandydatów z
  iteration_crop_members.source_at_ref = azpkg:<package_id>.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

from .az_package_transport import (
    AZPackage,
    AZPackageAppendResult,
)
from .az_registry import AZRegistry
from .az_revision_store import AZRevisionStore
from .pz2_az_adapter import az_revision_to_pz2_metadata


class PZ2AppendMaterializationError(RuntimeError):
    """Append registry nie może zostać bezpiecznie włączony do aktywnego PZ2."""


@dataclass(frozen=True)
class PZ2AppendMaterializationItem:
    plate_id: str
    crop_id: str
    crop_identity_sha256: str
    artifact_id: str
    az_applied: bool
    requires_review: bool


@dataclass(frozen=True)
class PZ2AppendMaterializationResult:
    preview_dir: str
    metadata_path: str
    backup_path: str | None
    package_id: str
    target_project_id: str
    iteration_num: int
    before_count: int
    candidates: int
    added: int
    already_materialized: int
    with_az: int
    without_az: int
    after_count: int
    items: tuple[PZ2AppendMaterializationItem, ...]
    verified_metadata: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    metadata_signature: tuple[int, int] | None = None


def materialize_az_append_to_preview(
    registry: AZRegistry,
    package: AZPackage,
    *,
    preview_dir: str | Path,
    target_project_id: str,
    iteration_num: int,
    append_result: AZPackageAppendResult | None = None,
) -> PZ2AppendMaterializationResult:
    """Dopisz append package do istniejącego metadata.json PZ2.

    Normalny flow może przekazać ``append_result`` z AZ009B.
    Po restarcie ``append_result`` może być None — kandydaci są wtedy
    odzyskiwani z registry po source_at_ref pakietu.
    """
    preview = Path(preview_dir)
    metadata_path = preview / "metadata.json"
    images_dir = preview / "images"

    if not preview.is_dir():
        raise PZ2AppendMaterializationError(
            f"Brak aktywnego preview runu: {preview}"
        )
    if not metadata_path.is_file():
        raise PZ2AppendMaterializationError(
            f"Brak metadata.json: {metadata_path}"
        )
    if not images_dir.is_dir():
        raise PZ2AppendMaterializationError(
            f"Brak images/: {images_dir}"
        )

    target_project_id = _required_text(
        "target_project_id",
        target_project_id,
    )
    iteration_num = _positive_int(
        "iteration_num",
        iteration_num,
    )

    if append_result is not None:
        if append_result.package_id != package.package_id:
            raise PZ2AppendMaterializationError(
                "append_result pochodzi z innego package_id."
            )
        if append_result.target_project_id != target_project_id:
            raise PZ2AppendMaterializationError(
                "append_result pochodzi z innego projektu docelowego."
            )
        if int(append_result.iteration_num) != iteration_num:
            raise PZ2AppendMaterializationError(
                "append_result pochodzi z innej iteracji."
            )

    metadata = _load_metadata(metadata_path)
    before_count = len(metadata)

    candidates = _collect_package_iteration_candidates(
        registry,
        package=package,
        target_project_id=target_project_id,
        iteration_num=iteration_num,
    )

    for _crop, _row, artifact_path in candidates:
        try:
            artifact_path.resolve().relative_to(images_dir.resolve())
        except ValueError as exc:
            raise PZ2AppendMaterializationError(
                "AZ009C wymaga, aby AZ009B materializował cropy bezpośrednio "
                "do images/ aktywnego runu PZ2."
            ) from exc

    # W zwykłym flow każdy realnie appended item AZ009B musi dać się odzyskać
    # z registry. To wykrywa częściowy/niezgodny zapis zanim dotkniemy metadata.
    if append_result is not None:
        registry_crop_ids = {
            str(row["crop_id"])
            for _crop, row, _artifact_path in candidates
        }
        missing = [
            item.crop_id
            for item in append_result.items
            if item.crop_id not in registry_crop_ids
        ]
        if missing:
            raise PZ2AppendMaterializationError(
                "Nie można odtworzyć z registry wszystkich cropów appendu: "
                + ", ".join(missing[:5])
            )

    existing_by_crop: dict[str, tuple[str, dict[str, Any]]] = {}
    existing_by_identity: dict[str, tuple[str, dict[str, Any]]] = {}
    for plate_id, row in metadata.items():
        if not isinstance(row, dict):
            continue
        crop_id = str(row.get("crop_id") or "").strip()
        identity = str(
            row.get("crop_identity_sha256") or ""
        ).strip().lower()
        if crop_id:
            existing_by_crop[crop_id] = (str(plate_id), row)
        if identity:
            existing_by_identity[identity] = (str(plate_id), row)

    # Append only inserts new rows; it never edits the nested old rows.
    # Keep those as the post-check reference without cloning detector payloads.
    updated = dict(metadata)
    added_items: list[PZ2AppendMaterializationItem] = []
    already_materialized = 0
    with_az = 0
    without_az = 0

    store = AZRevisionStore(registry.database)
    now_iso = datetime.now(timezone.utc).isoformat()

    for package_crop, registry_row, artifact_path in candidates:
        crop_id = str(registry_row["crop_id"])
        identity_sha = str(
            registry_row["identity_sha256"] or ""
        ).strip().lower()
        artifact_id = str(registry_row["artifact_id"])

        existing_crop = existing_by_crop.get(crop_id)
        existing_identity = existing_by_identity.get(identity_sha)

        if existing_crop or existing_identity:
            existing = existing_crop or existing_identity
            assert existing is not None
            _existing_plate_id, existing_row = existing

            existing_crop_id = str(
                existing_row.get("crop_id") or ""
            ).strip()
            existing_identity_sha = str(
                existing_row.get("crop_identity_sha256") or ""
            ).strip().lower()

            if existing_crop_id and existing_crop_id != crop_id:
                raise PZ2AppendMaterializationError(
                    "Istniejący metadata row ma tę samą identity, "
                    "ale inne crop_id."
                )
            if (
                existing_identity_sha
                and existing_identity_sha != identity_sha
            ):
                raise PZ2AppendMaterializationError(
                    "Istniejący metadata row ma ten sam crop_id, "
                    "ale inną identity."
                )
            already_materialized += 1
            continue

        plate_id = artifact_path.stem
        if not plate_id:
            raise PZ2AppendMaterializationError(
                f"Nie można wyznaczyć plate_id z {artifact_path}"
            )
        if plate_id in updated:
            raise PZ2AppendMaterializationError(
                f"plate_id {plate_id} jest już zajęty przez inny rekord."
            )

        base = _build_base_metadata_row(
            package=package,
            package_crop=package_crop,
            registry_row=registry_row,
            preview=preview,
            target_project_id=target_project_id,
            iteration_num=iteration_num,
            now_iso=now_iso,
        )

        revision = store.get_project_az(
            project_id=target_project_id,
            crop_id=crop_id,
        )
        az_applied = revision is not None
        if revision is not None:
            base = az_revision_to_pz2_metadata(
                base,
                revision,
                image_width=int(registry_row["width"]),
                image_height=int(registry_row["height"]),
            )
            with_az += 1
        else:
            without_az += 1

        provenance = base.get("append_provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            base["append_provenance"] = provenance
        provenance["materialized_at"] = now_iso

        updated[plate_id] = base
        existing_by_crop[crop_id] = (plate_id, base)
        existing_by_identity[identity_sha] = (plate_id, base)

        review_state = base.get("review_state")
        review_state = (
            review_state if isinstance(review_state, dict) else {}
        )
        requires_review = (
            str(review_state.get("status") or "").strip().lower()
            != "approved"
        )

        added_items.append(
            PZ2AppendMaterializationItem(
                plate_id=plate_id,
                crop_id=crop_id,
                crop_identity_sha256=identity_sha,
                artifact_id=artifact_id,
                az_applied=az_applied,
                requires_review=requires_review,
            )
        )

    if not added_items:
        return PZ2AppendMaterializationResult(
            preview_dir=str(preview),
            metadata_path=str(metadata_path),
            backup_path=None,
            package_id=package.package_id,
            target_project_id=target_project_id,
            iteration_num=iteration_num,
            before_count=before_count,
            candidates=len(candidates),
            added=0,
            already_materialized=already_materialized,
            with_az=0,
            without_az=0,
            after_count=before_count,
            items=(),
            verified_metadata=metadata,
            metadata_signature=_metadata_signature(metadata_path),
        )

    backup_path = _backup_metadata(metadata_path)
    _atomic_write_json(metadata_path, updated)

    # Post-check: stare rekordy muszą pozostać logicznie IDENTYCZNE.
    written = _load_metadata(metadata_path)
    for plate_id, old_row in metadata.items():
        if plate_id not in written or written[plate_id] != old_row:
            shutil.copy2(backup_path, metadata_path)
            raise PZ2AppendMaterializationError(
                "Post-check wykrył zmianę istniejącego rekordu PZ2. "
                "Przywrócono backup metadata.json."
            )

    if len(written) != before_count + len(added_items):
        shutil.copy2(backup_path, metadata_path)
        raise PZ2AppendMaterializationError(
            "Post-check liczności metadata nie przeszedł. "
            "Przywrócono backup metadata.json."
        )

    return PZ2AppendMaterializationResult(
        preview_dir=str(preview),
        metadata_path=str(metadata_path),
        backup_path=str(backup_path),
        package_id=package.package_id,
        target_project_id=target_project_id,
        iteration_num=iteration_num,
        before_count=before_count,
        candidates=len(candidates),
        added=len(added_items),
        already_materialized=already_materialized,
        with_az=with_az,
        without_az=without_az,
        after_count=len(written),
        items=tuple(added_items),
        verified_metadata=written,
        metadata_signature=_metadata_signature(metadata_path),
    )


def _collect_package_iteration_candidates(
    registry: AZRegistry,
    *,
    package: AZPackage,
    target_project_id: str,
    iteration_num: int,
):
    registry.initialize()
    source_at_ref = f"azpkg:{package.package_id}"
    package_by_identity = {
        crop.crop_identity_sha256: crop
        for crop in package.crops
    }
    if not package_by_identity:
        return []

    placeholders = ",".join(
        "?" for _ in package_by_identity
    )
    params = [
        target_project_id,  # project_crop_members.project_id
        target_project_id,  # iteration_crop_members.project_id
        iteration_num,
        source_at_ref,
        *package_by_identity.keys(),
    ]

    with registry.database.read_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                pc.crop_id,
                pc.identity_sha256,
                pc.identity_mode,
                pc.identity_schema,
                pc.source_image_id,
                pc.source_annotation_id,
                pc.source_geometry_hash,
                pc.crop_contract_sha256,
                pc.width,
                pc.height,
                si.canonical_sha256 AS source_file_sha256,
                ia.artifact_id,
                ia.sha256 AS artifact_sha256,
                ia.relative_path,
                ia.external_path,
                pcm.source_mode,
                icm.source_plate_key,
                icm.source_at_ref
            FROM plate_crops pc
            JOIN source_images si
              ON si.source_image_id = pc.source_image_id
            JOIN project_crop_members pcm
              ON pcm.crop_id = pc.crop_id
             AND pcm.project_id = ?
            JOIN iteration_crop_members icm
              ON icm.crop_id = pc.crop_id
             AND icm.project_id = ?
             AND icm.iteration_num = ?
             AND icm.source_at_ref = ?
            JOIN image_artifacts ia
              ON ia.artifact_id = icm.artifact_id
            WHERE pc.identity_sha256 IN ({placeholders})
            ORDER BY pc.identity_sha256
            """,
            params,
        ).fetchall()

    result = []
    for row in rows:
        identity_sha = str(
            row["identity_sha256"] or ""
        ).strip().lower()
        package_crop = package_by_identity.get(identity_sha)
        if package_crop is None:
            continue

        if identity_sha != package_crop.crop_identity_sha256:
            raise PZ2AppendMaterializationError(
                "Registry identity nie odpowiada pakietowi."
            )

        artifact_path = _registry_artifact_path(
            registry,
            row,
        )
        if not artifact_path.is_file():
            raise PZ2AppendMaterializationError(
                f"Brak artefaktu appendu: {artifact_path}"
            )

        result.append((package_crop, row, artifact_path))

    return result


def _registry_artifact_path(
    registry: AZRegistry,
    row,
) -> Path:
    relative = str(row["relative_path"] or "").strip()
    external = str(row["external_path"] or "").strip()

    if relative:
        if registry.workspace_dir is None:
            raise PZ2AppendMaterializationError(
                "Registry ma relative_path, ale brak workspace_dir."
            )
        return Path(registry.workspace_dir) / Path(relative)
    if external:
        return Path(external)
    raise PZ2AppendMaterializationError(
        "Registry artifact nie ma ścieżki."
    )


def _build_base_metadata_row(
    *,
    package: AZPackage,
    package_crop,
    registry_row,
    preview: Path,
    target_project_id: str,
    iteration_num: int,
    now_iso: str,
) -> dict[str, Any]:
    identity = package_crop.crop_identity
    crop_contract = identity.get("crop_contract")
    crop_contract = (
        crop_contract if isinstance(crop_contract, dict) else {}
    )

    width = int(registry_row["width"])
    height = int(registry_row["height"])
    if width <= 0 or height <= 0:
        raise PZ2AppendMaterializationError(
            "Registry crop nie ma poprawnych wymiarów."
        )

    output_height = int(
        crop_contract.get("output_height") or height
    )

    source_image_identity = str(
        identity.get("source_image_id") or ""
    ).strip()
    source_annotation_id = str(
        identity.get("source_annotation_id") or ""
    ).strip()
    source_geometry_hash = str(
        identity.get("source_geometry_hash") or ""
    ).strip().lower()

    if not (
        source_image_identity
        and source_annotation_id
        and source_geometry_hash
    ):
        raise PZ2AppendMaterializationError(
            "Pakiet nie ma kompletnego lineage PZ1."
        )

    return {
        "characters": [],
        "status": "needs_fix",
        "plate_image_width": width,
        "plate_image_height": height,
        "is_square": bool(output_height >= 128),
        "plate_layout": (
            "two_row" if output_height >= 128 else "single_row"
        ),
        "layout_row_count": 2 if output_height >= 128 else 1,
        "source_file_sha256": package_crop.source_file_sha256,
        "source_image_id": source_image_identity,
        "source_annotation_id": source_annotation_id,
        "source_geometry_hash": source_geometry_hash,
        "crop_id": str(registry_row["crop_id"]),
        "crop_identity_sha256": package_crop.crop_identity_sha256,
        "crop_identity_schema": str(
            registry_row["identity_schema"] or ""
        ),
        "crop_identity_mode": str(
            registry_row["identity_mode"] or ""
        ),
        "crop_contract_sha256": str(
            registry_row["crop_contract_sha256"] or ""
        ),
        "artifact_id": str(registry_row["artifact_id"]),
        "registry_source_image_id": str(
            registry_row["source_image_id"] or ""
        ),
        "gold_state": {
            "approved": False,
            "excluded": False,
            "candidate": False,
        },
        "review_state": {
            "schema": "alpr.pz2.review.v1",
            "status": "in_progress",
            "source": "az_package_append",
            "human_edited": False,
            "approved_at": None,
        },
        "source_info": {
            "bucket": "auto_preview",
            "origin": "pz2_detect",
            "run_id": preview.name,
            "review_manifest_id": "",
            "import_batch_id": package.package_id,
            "last_modified_at": now_iso,
            "last_modified_by": "az_package_append",
        },
        "append_provenance": {
            "schema": "alpr.az_append.v1",
            "package_id": package.package_id,
            "entry_id": package_crop.entry_id,
            "target_project_id": target_project_id,
            "iteration_num": int(iteration_num),
            "source_at_ref": str(
                registry_row["source_at_ref"] or ""
            ),
            "source_plate_key": str(
                registry_row["source_plate_key"] or ""
            ),
            "artifact_id": str(registry_row["artifact_id"]),
            "registry_source_image_id": str(
                registry_row["source_image_id"] or ""
            ),
            "materialized_at": now_iso,
        },
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except Exception as exc:
        raise PZ2AppendMaterializationError(
            f"Nie można odczytać metadata.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PZ2AppendMaterializationError(
            "metadata.json musi zawierać obiekt JSON."
        )
    return payload


def _backup_metadata(path: Path) -> Path:
    backup_dir = path.parent / "_az_append_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = (
        backup_dir
        / f"metadata_before_append_{stamp}.json"
    )
    shutil.copy2(path, backup_path)
    return backup_path


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temp = path.with_name(
        f".{path.name}.az009c.{uuid.uuid4().hex}.tmp"
    )
    try:
        temp.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except Exception:
            pass


def _metadata_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _required_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise PZ2AppendMaterializationError(
            f"{name} nie może być puste."
        )
    return text


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise PZ2AppendMaterializationError(
            f"{name} musi być dodatnim int."
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PZ2AppendMaterializationError(
            f"{name} musi być dodatnim int."
        ) from exc
    if number <= 0:
        raise PZ2AppendMaterializationError(
            f"{name} musi być > 0."
        )
    return number
