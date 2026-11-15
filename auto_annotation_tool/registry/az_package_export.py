#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AZ009E1 — eksport aktywnego PZ2 do przenośnego pakietu cropów + AZ.

Format wyjściowy jest dokładnie tym samym kontraktem `alpr.az_package.v1`,
który AZ009A/B potrafią zweryfikować i dopiąć do innego projektu.

Źródłem prawdy eksportu jest:
- aktywne metadata.json PZ2 — lineage i bieżąca semantyka AZ,
- registry v6 — crop/project/artifact identity i opcjonalny revision provenance.

Eksporter nie mutuje PZ2 ani registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Mapping, Sequence
from PIL import Image

from .az_package_transport import AZ_PACKAGE_SCHEMA
from .az_registry import AZRegistry, crop_contract_sha256
from .az_revision_store import compute_az_payload_sha256
from .crop_identity import (
    CROP_IDENTITY_MODE_LINEAGE,
    CROP_IDENTITY_SCHEMA,
    PZ1_CROP_CONTRACT_VERSION,
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)
from .pz2_az_adapter import (
    derive_pz2_revision_context,
    pz2_metadata_to_az_payload,
)


class AZPackageExportError(RuntimeError):
    """Aktywnego PZ2 nie można wyeksportować bez utraty kontraktu identity."""


@dataclass(frozen=True)
class AZPackageExportItem:
    plate_id: str
    crop_id: str
    crop_identity_sha256: str
    artifact_sha256: str
    payload_sha256: str
    source_az_revision_id: str | None
    output_relpath: str


@dataclass(frozen=True)
class AZPackageExportResult:
    output_dir: str
    manifest_path: str
    package_id: str
    project_id: str | None
    iteration_num: int | None
    exported: int
    requested: int
    with_bound_revision: int
    items: tuple[AZPackageExportItem, ...]
    source_mode: str = "campaign"


def export_pz2_az_package(
    registry: AZRegistry,
    *,
    preview_dir: str | Path,
    project_id: str | None = None,
    iteration_num: int | None = None,
    output_dir: str | Path,
    plate_ids: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AZPackageExportResult:
    """Eksportuj wybrane albo wszystkie rekordy aktywnego PZ2.

    `plate_ids=None` oznacza wszystkie rekordy metadata posiadające crop_id.
    Jeśli `plate_ids` podano, każdy wskazany rekord musi istnieć i być
    eksportowalny; nie ma cichego pomijania.
    Brak project_id i iteration_num oznacza tryb swobodny. Opcjonalne metadata
    pozwala wyeksportować bieżący stan edytora bez zapisywania pliku źródłowego.
    """
    # Export must not create or migrate the source registry.
    if not registry.database.db_path.is_file():
        raise AZPackageExportError("Brak istniejącego AZ registry.")
    preview = Path(preview_dir)
    metadata_path = preview / "metadata.json"
    images_dir = preview / "images"

    if not metadata_path.is_file():
        raise AZPackageExportError(f"Brak metadata.json: {metadata_path}")
    if not images_dir.is_dir():
        raise AZPackageExportError(f"Brak images/: {images_dir}")

    if project_id is None:
        if iteration_num is not None:
            raise AZPackageExportError("Tryb swobodny nie ma iteracji projektu.")
    else:
        project_id = _required_text("project_id", project_id)
        iteration_num = _positive_int("iteration_num", iteration_num)

    metadata = _load_metadata(metadata_path) if metadata is None else metadata
    if not isinstance(metadata, Mapping):
        raise AZPackageExportError("Metadata PZ2 musi być obiektem JSON.")
    selected_ids = _resolve_plate_ids(metadata, plate_ids)
    if not selected_ids:
        raise AZPackageExportError("Brak rekordów PZ2 do eksportu.")

    project_info = _load_project_info(registry, project_id) if project_id else None
    prepared = [
        _prepare_export_item(
            registry,
            preview=preview,
            images_dir=images_dir,
            project_id=project_id,
            iteration_num=iteration_num,
            plate_id=plate_id,
            row=metadata[plate_id],
        )
        for plate_id in selected_ids
    ]

    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise AZPackageExportError(
                f"Ścieżka docelowa nie jest katalogiem: {destination}"
            )
        if any(destination.iterdir()):
            raise AZPackageExportError(
                "Katalog docelowy pakietu musi być nowy albo pusty."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = destination.with_name(
        f".{destination.name}.azpkg-{uuid.uuid4().hex}.tmp"
    )
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    crops_dir = temp_dir / "crops"

    package_id = f"AZPKG-{uuid.uuid4().hex.upper()}"
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        crops_dir.mkdir(parents=True, exist_ok=False)
        manifest_crops = []
        result_items = []
        bound_count = 0

        for item in prepared:
            suffix = item["source_artifact"].suffix.lower() or ".bin"
            output_name = (
                f"crop_{item['crop_identity_sha256']}{suffix}"
            )
            output_path = crops_dir / output_name
            shutil.copy2(item["source_artifact"], output_path)

            copied_sha = _file_sha256(output_path)
            if copied_sha != item["artifact_sha256"]:
                raise AZPackageExportError(
                    f"SHA kopii eksportowej nie zgadza się dla {item['plate_id']}."
                )

            relpath = f"crops/{output_name}"
            az_payload = item["az_payload"]
            az_meta = {
                "payload": az_payload,
                "payload_sha256": item["payload_sha256"],
                "source_kind": item["source_kind"],
                "source_status": item["source_status"],
                "trust_state": item["trust_state"],
                "origin_project_id": item["origin_project_id"],
                "origin_iteration": item["origin_iteration"],
                "source_az_revision_id": item["source_az_revision_id"],
                "created_at": item["az_created_at"],
            }

            manifest_crops.append({
                "entry_id": item["plate_id"],
                "crop_identity": item["crop_identity"],
                "crop_identity_sha256": item["crop_identity_sha256"],
                "crop_contract_sha256": item["crop_contract_sha256"],
                "source_file_sha256": item["source_file_sha256"],
                "artifact": {
                    "path": relpath,
                    "sha256": item["artifact_sha256"],
                    "size_bytes": item["size_bytes"],
                    "width": item["width"],
                    "height": item["height"],
                },
                "az": az_meta,
            })

            if item["source_az_revision_id"]:
                bound_count += 1
            result_items.append(
                AZPackageExportItem(
                    plate_id=item["plate_id"],
                    crop_id=item["crop_id"],
                    crop_identity_sha256=item["crop_identity_sha256"],
                    artifact_sha256=item["artifact_sha256"],
                    payload_sha256=item["payload_sha256"],
                    source_az_revision_id=item["source_az_revision_id"],
                    output_relpath=relpath,
                )
            )

        manifest = {
            "schema": AZ_PACKAGE_SCHEMA,
            "package_id": package_id,
            "created_at": created_at,
            "source": ({
                "project_id": project_id,
                "project_name": project_info["display_name"],
                "folder_name": project_info["folder_name"],
                "campaign_key": project_info["campaign_key"],
                "iteration": iteration_num,
                "preview_run": preview.name,
                "scope": (
                    "selected"
                    if plate_ids is not None
                    else "active_pz2_all"
                ),
            } if project_info is not None else {
                "mode": "free_mode",
                "preview_run": preview.name,
                "scope": "selected" if plate_ids is not None else "active_pz2_all",
            }),
            "crops": manifest_crops,
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        # Własny format musi przejść własny importer jeszcze przed publikacją.
        from .az_package_transport import load_az_package
        verified = load_az_package(temp_dir)
        if verified.invalid_items:
            reasons = "; ".join(
                item.reason for item in verified.invalid_items[:3]
            )
            raise AZPackageExportError(
                f"Wewnętrzna walidacja pakietu nie przeszła: {reasons}"
            )
        if len(verified.crops) != len(prepared):
            raise AZPackageExportError(
                "Wewnętrzna walidacja pakietu zwróciła inną liczbę cropów."
            )

        if destination.exists():
            # Dozwolony był tylko pusty katalog — usuń go tuż przed atomowym rename.
            destination.rmdir()
        os.replace(temp_dir, destination)

    except Exception:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception:
            pass
        raise

    return AZPackageExportResult(
        output_dir=str(destination),
        manifest_path=str(destination / "manifest.json"),
        package_id=package_id,
        project_id=project_id,
        iteration_num=iteration_num,
        exported=len(prepared),
        requested=len(selected_ids),
        with_bound_revision=bound_count,
        items=tuple(result_items),
        source_mode="campaign" if project_id else "free_mode",
    )


def _prepare_export_item(
    registry: AZRegistry,
    *,
    preview: Path,
    images_dir: Path,
    project_id: str | None,
    iteration_num: int | None,
    plate_id: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise AZPackageExportError(
            f"metadata[{plate_id!r}] nie jest obiektem."
        )

    crop_id = _required_text(
        f"metadata[{plate_id}].crop_id",
        row.get("crop_id"),
    )
    identity_sha = _sha256(
        f"metadata[{plate_id}].crop_identity_sha256",
        row.get("crop_identity_sha256"),
    )
    artifact_id = _required_text(
        f"metadata[{plate_id}].artifact_id",
        row.get("artifact_id"),
    )

    registry_row = _load_registry_export_row(
        registry,
        project_id=project_id,
        crop_id=crop_id,
        artifact_id=artifact_id,
    )

    if str(registry_row["identity_sha256"]).lower() != identity_sha:
        raise AZPackageExportError(
            f"Identity metadata/registry nie zgadza się dla {plate_id}."
        )

    source_file_sha256 = _sha256(
        "source_file_sha256",
        registry_row["source_file_sha256"],
    )
    metadata_source_sha = str(
        row.get("source_file_sha256") or ""
    ).strip().lower()
    if metadata_source_sha and metadata_source_sha != source_file_sha256:
        raise AZPackageExportError(
            f"source_file_sha256 metadata/registry nie zgadza się dla {plate_id}."
        )

    width = _positive_int("width", registry_row["width"])
    height = _positive_int("height", registry_row["height"])

    source_artifact = _find_preview_image(images_dir, plate_id)
    if source_artifact is None:
        source_artifact = _registry_artifact_path(
            registry,
            registry_row,
        )
    if not source_artifact.is_file():
        raise AZPackageExportError(
            f"Brak fizycznego cropa dla {plate_id}: {source_artifact}"
        )

    actual_sha = _file_sha256(source_artifact)
    artifact_sha = _sha256(
        "artifact_sha256",
        registry_row["artifact_sha256"],
    )
    if actual_sha != artifact_sha:
        raise AZPackageExportError(
            f"SHA fizycznego cropa nie odpowiada registry dla {plate_id}."
        )

    size_bytes = int(source_artifact.stat().st_size)
    stored_size = registry_row["artifact_size_bytes"]
    if stored_size is not None and int(stored_size) != size_bytes:
        raise AZPackageExportError(
            f"Rozmiar fizycznego cropa nie odpowiada registry dla {plate_id}."
        )

    try:
        with Image.open(source_artifact) as image:
            actual_dimensions = image.size
            image.verify()
    except Exception as exc:
        raise AZPackageExportError(f"Nieprawidłowy obraz cropa {plate_id}: {exc}") from exc
    if actual_dimensions != (width, height):
        raise AZPackageExportError(f"Wymiary cropa nie odpowiadają registry dla {plate_id}.")
    for key, expected in (("plate_image_width", width), ("plate_image_height", height)):
        if row.get(key) is not None and _positive_int(key, row[key]) != expected:
            raise AZPackageExportError(f"Wymiary metadata/registry nie zgadzają się dla {plate_id}.")

    crop_identity = _reconstruct_verified_identity(
        row=row,
        registry_row=registry_row,
        identity_sha256=identity_sha,
        source_file_sha256=source_file_sha256,
        width=width,
        height=height,
    )
    contract_sha = crop_contract_sha256(
        crop_identity["crop_contract"]
    )
    contract = crop_identity["crop_contract"]
    if (contract.get("output_width"), contract.get("output_height")) != (width, height):
        raise AZPackageExportError(f"Wymiary crop contract/registry nie zgadzają się dla {plate_id}.")
    for key in ("source_annotation_id", "source_geometry_hash"):
        stored = registry_row[key]
        if stored and str(crop_identity.get(key) or "") != str(stored):
            raise AZPackageExportError(f"Lineage identity/registry nie zgadza się dla {plate_id}: {key}.")
        if row.get(key) and str(row[key]) != str(crop_identity.get(key) or ""):
            raise AZPackageExportError(f"Lineage metadata/identity nie zgadza się dla {plate_id}: {key}.")
    if row.get("source_image_id") and row["source_image_id"] != crop_identity.get("source_image_id"):
        raise AZPackageExportError(f"source_image_id metadata/identity nie zgadza się dla {plate_id}.")
    stored_contract_sha = str(
        registry_row["crop_contract_sha256"] or ""
    ).strip().lower()
    if stored_contract_sha and contract_sha != stored_contract_sha:
        raise AZPackageExportError(
            f"crop_contract SHA nie odpowiada registry dla {plate_id}."
        )

    try:
        az_payload = pz2_metadata_to_az_payload(
            row,
            image_width=width,
            image_height=height,
        )
    except Exception as exc:
        raise AZPackageExportError(
            f"Nie można zbudować AZ z PZ2 dla {plate_id}: {exc}"
        ) from exc

    payload_sha = compute_az_payload_sha256(az_payload)
    context = derive_pz2_revision_context(row)

    bound = _load_matching_bound_revision(
        registry,
        project_id=project_id,
        crop_id=crop_id,
        payload_sha256=payload_sha,
    ) if project_id is not None else None
    if bound is not None:
        source_kind = str(bound["source_kind"] or context.source_kind)
        source_status = (
            str(bound["source_status"] or "").strip()
            or context.source_status
        )
        trust_state = str(
            bound["trust_state"] or context.trust_state
        )
        origin_project_id = (
            str(bound["origin_project_id"] or "").strip()
            or project_id
        )
        origin_iteration = (
            int(bound["origin_iteration"])
            if bound["origin_iteration"] is not None
            else iteration_num
        )
        source_az_revision_id = str(
            bound["az_revision_id"]
        )
        az_created_at = str(bound["created_at"] or "").strip()
    else:
        source_kind = context.source_kind
        source_status = context.source_status
        trust_state = context.trust_state
        origin_project_id = project_id
        origin_iteration = iteration_num
        source_az_revision_id = None
        az_created_at = datetime.now(timezone.utc).isoformat()

    return {
        "plate_id": plate_id,
        "crop_id": crop_id,
        "crop_identity": crop_identity,
        "crop_identity_sha256": identity_sha,
        "crop_contract_sha256": contract_sha,
        "source_file_sha256": source_file_sha256,
        "source_artifact": source_artifact,
        "artifact_sha256": artifact_sha,
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
        "az_payload": az_payload,
        "payload_sha256": payload_sha,
        "source_kind": source_kind,
        "source_status": source_status,
        "trust_state": trust_state,
        "origin_project_id": origin_project_id,
        "origin_iteration": origin_iteration,
        "source_az_revision_id": source_az_revision_id,
        "az_created_at": az_created_at,
    }


def _reconstruct_verified_identity(
    *,
    row: Mapping[str, Any],
    registry_row,
    identity_sha256: str,
    source_file_sha256: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    identity_schema = str(
        registry_row["identity_schema"] or ""
    ).strip()
    identity_mode = str(
        registry_row["identity_mode"] or ""
    ).strip()

    if identity_schema != CROP_IDENTITY_SCHEMA:
        raise AZPackageExportError(
            f"Nieobsługiwany crop identity schema: {identity_schema!r}."
        )
    if identity_mode != CROP_IDENTITY_MODE_LINEAGE:
        raise AZPackageExportError(
            "AZ009E1 eksportuje obecnie tylko lineage_v1. "
            f"Otrzymano: {identity_mode!r}."
        )

    embedded = row.get("crop_identity_payload")
    if isinstance(embedded, Mapping):
        candidate = dict(embedded)
        if compute_crop_identity_sha256(candidate) == identity_sha256:
            return candidate

    source_image_identity = str(
        row.get("source_image_id") or ""
    ).strip()
    if not source_image_identity:
        source_image_identity = f"img-sha256-{source_file_sha256}"

    source_annotation_id = _required_text(
        "source_annotation_id",
        row.get("source_annotation_id")
        or registry_row["source_annotation_id"],
    )
    source_geometry_hash = _required_text(
        "source_geometry_hash",
        row.get("source_geometry_hash")
        or registry_row["source_geometry_hash"],
    ).lower()

    interpolation_values = _unique([
        row.get("crop_interpolation"),
        row.get("interpolation"),
        row.get("source_crop_interpolation"),
        "lanczos4",
        "cubic",
        "linear",
        "nearest",
    ])
    rectify_values = _bool_candidates(row, "rectify", True)
    deskew_values = _bool_candidates(row, "do_deskew", False)
    contrast_values = _bool_candidates(
        row,
        "enhance_contrast",
        False,
    )
    contract_versions = _unique([
        row.get("crop_contract_version"),
        PZ1_CROP_CONTRACT_VERSION,
    ])

    stored_contract_sha = str(
        registry_row["crop_contract_sha256"] or ""
    ).strip().lower()

    matches = []
    for interpolation in interpolation_values:
        for rectify in rectify_values:
            for do_deskew in deskew_values:
                for enhance_contrast in contrast_values:
                    for contract_version in contract_versions:
                        try:
                            candidate = build_pz1_crop_identity(
                                source_image_id=source_image_identity,
                                source_annotation_id=source_annotation_id,
                                source_geometry_hash=source_geometry_hash,
                                rectify=rectify,
                                do_deskew=do_deskew,
                                enhance_contrast=enhance_contrast,
                                interpolation=str(interpolation),
                                output_width=width,
                                output_height=height,
                                contract_version=str(contract_version),
                            )
                        except Exception:
                            continue
                        if (
                            compute_crop_identity_sha256(candidate)
                            != identity_sha256
                        ):
                            continue
                        if stored_contract_sha:
                            if (
                                crop_contract_sha256(
                                    candidate["crop_contract"]
                                )
                                != stored_contract_sha
                            ):
                                continue
                        matches.append(candidate)

    if not matches:
        raise AZPackageExportError(
            "Nie udało się odtworzyć crop_identity zgodnego z SHA. "
            "Eksport przerwany zamiast zgadywania kontraktu PZ1."
        )

    # Canonical identity SHA praktycznie eliminuje semantyczną wieloznaczność.
    return matches[0]


def _load_registry_export_row(
    registry: AZRegistry,
    *,
    project_id: str | None,
    crop_id: str,
    artifact_id: str,
):
    with registry.database.read_connection() as connection:
        row = connection.execute(
            """
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
                ia.size_bytes AS artifact_size_bytes,
                ia.relative_path,
                ia.external_path
            FROM plate_crops pc
            JOIN source_images si
              ON si.source_image_id = pc.source_image_id
            JOIN crop_artifacts ca
              ON ca.crop_id = pc.crop_id
            JOIN image_artifacts ia
              ON ia.artifact_id = ca.artifact_id
            WHERE pc.crop_id = ?
              AND ia.artifact_id = ?
              AND (? IS NULL OR EXISTS (
                  SELECT 1 FROM project_crop_members pcm
                  WHERE pcm.crop_id = pc.crop_id AND pcm.project_id = ?
              ))
            """,
            (crop_id, artifact_id, project_id, project_id),
        ).fetchone()
    if row is None:
        raise AZPackageExportError(
            f"Crop {crop_id} / artifact {artifact_id} nie istnieje w registry"
            + (" lub nie należy do projektu." if project_id else ".")
        )
    return row


def _load_matching_bound_revision(
    registry: AZRegistry,
    *,
    project_id: str,
    crop_id: str,
    payload_sha256: str,
):
    with registry.database.read_connection() as connection:
        return connection.execute(
            """
            SELECT
                pca.az_revision_id,
                pca.effective_status,
                ar.payload_sha256,
                ar.source_kind,
                ar.source_status,
                ar.trust_state,
                ar.origin_project_id,
                ar.origin_iteration,
                ar.created_at
            FROM project_crop_az pca
            JOIN az_revisions ar
              ON ar.az_revision_id = pca.az_revision_id
            WHERE pca.project_id = ?
              AND pca.crop_id = ?
              AND ar.payload_sha256 = ?
            """,
            (project_id, crop_id, payload_sha256),
        ).fetchone()


def _load_project_info(
    registry: AZRegistry,
    project_id: str,
) -> dict[str, str | None]:
    with registry.database.read_connection() as connection:
        row = connection.execute(
            """
            SELECT project_id, display_name, folder_name, campaign_key
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
    if row is None:
        raise AZPackageExportError(
            f"Nieznany project_id: {project_id}"
        )
    return {
        "display_name": (
            str(row["display_name"] or "").strip()
            or project_id
        ),
        "folder_name": str(
            row["folder_name"] or ""
        ).strip() or None,
        "campaign_key": str(
            row["campaign_key"] or ""
        ).strip() or None,
    }


def _registry_artifact_path(
    registry: AZRegistry,
    row,
) -> Path:
    relative = str(row["relative_path"] or "").strip()
    external = str(row["external_path"] or "").strip()
    if relative:
        if registry.workspace_dir is None:
            raise AZPackageExportError(
                "Registry ma relative_path, ale brak workspace_dir."
            )
        return Path(registry.workspace_dir) / Path(relative)
    if external:
        return Path(external)
    raise AZPackageExportError(
        "Artifact registry nie ma ścieżki."
    )


def _find_preview_image(
    images_dir: Path,
    plate_id: str,
) -> Path | None:
    for candidate in sorted(images_dir.glob(f"{plate_id}.*")):
        if candidate.is_file():
            return candidate
    return None


def _resolve_plate_ids(
    metadata: Mapping[str, Any],
    plate_ids: Sequence[str] | None,
) -> list[str]:
    if plate_ids is None:
        result = [
            str(pid)
            for pid, row in metadata.items()
            if isinstance(row, Mapping)
            and str(row.get("crop_id") or "").strip()
        ]
        return result

    result = []
    seen = set()
    for value in plate_ids:
        pid = str(value or "").strip()
        if not pid or pid in seen:
            continue
        if pid not in metadata:
            raise AZPackageExportError(
                f"Nieznany plate_id w aktywnym PZ2: {pid}"
            )
        seen.add(pid)
        result.append(pid)
    return result


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except Exception as exc:
        raise AZPackageExportError(
            f"Nie można odczytać metadata.json: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AZPackageExportError(
            "metadata.json musi zawierać obiekt JSON."
        )
    return payload


def _bool_candidates(
    row: Mapping[str, Any],
    key: str,
    default: bool,
) -> list[bool]:
    if key in row:
        return [bool(row.get(key))]
    return [default, not default]


def _unique(values) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(name: str, value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        ch not in "0123456789abcdef"
        for ch in text
    ):
        raise AZPackageExportError(
            f"{name} musi być pełnym SHA-256 hex."
        )
    return text


def _required_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise AZPackageExportError(
            f"{name} nie może być puste."
        )
    return text


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise AZPackageExportError(
            f"{name} musi być dodatnim int."
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AZPackageExportError(
            f"{name} musi być dodatnim int."
        ) from exc
    if number <= 0:
        raise AZPackageExportError(
            f"{name} musi być > 0."
        )
    return number
