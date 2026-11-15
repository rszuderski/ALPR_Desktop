#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observed checkpoint-to-mobile conversion events for Desktop Z4.

Only explicitly fresh MobileExportRequest jobs generate conversion events. Reuse
of a ready .alprmodel NEVER does. SUCCEEDED requires evidence from the newly
written, exporter-validated final archive (including nested child sidecars).
This module never converts models or creates any .alprmodel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from .z4_mobile_artifact_registry import MANIFEST_CAP, append_event, sha256_file

_ROLE_TO_MARKER = {"plate": "MT", "character": "MZ", "vehicle": "MP"}
_RUNTIME_ALIAS = {"litert": "tflite", "tflite": "tflite", "onnx": "onnx", "ncnn": "ncnn"}


def _variant_spec(request: Any) -> list[dict]:
    formats = tuple(getattr(request, "formats", ()) or ())
    by_format = getattr(request, "format_quantizations", None) or {}
    general = tuple(getattr(request, "quantizations", ()) or ())
    result: list[dict] = []
    for fmt in formats:
        runtime = _RUNTIME_ALIAS.get(str(fmt).lower())
        if runtime is None:
            raise ValueError(f"Nieznany format konwersji: {fmt}")
        for precision in by_format.get(fmt, general or ("fp32",)):
            prec = str(precision).lower()
            if prec not in {"fp32", "int8"}:
                raise ValueError(f"Nieznana precyzja konwersji: {prec}")
            entry = {"runtime": runtime, "precision": prec}
            if entry not in result:
                result.append(entry)
    if not result:
        raise ValueError("Świeży eksport nie ma wariantów wyjściowych.")
    return result


def _fresh_sources(request: Any) -> tuple[bool, list[tuple[str, Any]]]:
    """Bundle jobs carry *_request; *_package means verified reuse, not conversion."""
    if hasattr(request, "plate_request") and hasattr(request, "character_request"):
        items = []
        for role in ("vehicle", "plate", "character"):
            sub = getattr(request, role + "_request", None)
            existing = getattr(request, role + "_package", None)
            if sub is not None and existing is not None:
                raise ValueError(f"{_ROLE_TO_MARKER[role]}: niejednoznaczny eksport i reuse.")
            if sub is not None:
                if str(getattr(sub, "role", role)) != role:
                    raise ValueError(f"{_ROLE_TO_MARKER[role]}: rola żądania niezgodna.")
                items.append((role, sub))
        return True, items
    role = str(getattr(request, "role", ""))
    if role not in _ROLE_TO_MARKER:
        raise ValueError("Świeży eksport: brak jednoznacznej roli modelu.")
    return False, [(role, request)]


def build_conversion_plan(request: Any, *, parent_operation_id: str) -> list[dict]:
    """Evidence pinned before calling the exporter, not inferred from file dates."""
    is_bundle, sources = _fresh_sources(request)
    result = []
    for role, source in sources:
        candidate = Path(getattr(source, "checkpoint", ""))
        if candidate.is_symlink():
            raise ValueError(f"{_ROLE_TO_MARKER[role]}: dowiązanie checkpointu nie jest dozwolone.")
        checkpoint = candidate.resolve()
        if not checkpoint.is_file():
            raise ValueError(f"{_ROLE_TO_MARKER[role]}: brak zwykłego checkpointu do konwersji.")
        result.append({
            "operation_id": parent_operation_id + "_" + _ROLE_TO_MARKER[role],
            "parent_export_operation_id": parent_operation_id,
            "role": _ROLE_TO_MARKER[role],
            "source_role": role,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "requested_variants": _variant_spec(source),
            "artifact_scope": "NESTED_CHILD_OF_COMPLETE_PACKAGE" if is_bundle else "SINGLE_MODEL_PACKAGE",
        })
    return result


def record_conversion_starts(root: Path, plan: list[dict], *, destination: Path) -> None:
    for slot in plan:
        append_event(
            root, operation_id=slot["operation_id"], operation="MODEL_CONVERSION",
            outcome="STARTED", artifact_path=str(destination),
            detail=f"{slot['role']}: rozpoczęto nowy eksport z checkpointu; wynik niepotwierdzony.",
            conversion=slot,
        )


def _sha_archive_member(archive: zipfile.ZipFile, path: str) -> str:
    digest = hashlib.sha256()
    info = archive.getinfo(path)
    if info.file_size > 512 * 1024 * 1024:
        raise ValueError("Model potomny przekracza limit bezpieczeństwa.")
    with archive.open(info) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(archive: zipfile.ZipFile, name: str) -> dict:
    hits = [item for item in archive.infolist() if item.filename == name]
    if len(hits) != 1 or not 2 <= hits[0].file_size <= MANIFEST_CAP:
        raise ValueError(f"Brak jednoznacznego manifestu: {name}.")
    obj = json.loads(archive.read(hits[0]).decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("Manifest potomny nie jest obiektem.")
    return obj


def attest_conversion(slot: dict, artifact_path: Path) -> dict:
    """Observe a proven child inside the final archive; don't approve Android/F4."""
    target = Path(artifact_path)
    checkpoint = Path(slot["checkpoint_path"])
    if not target.is_file() or target.suffix.lower() != ".alprmodel":
        raise ValueError("Nie istnieje poprawny wynikowy .alprmodel.")
    if not checkpoint.is_file() or sha256_file(checkpoint) != slot["checkpoint_sha256"]:
        raise ValueError("Checkpoint zmienił SHA po rozpoczęciu konwersji.")
    with zipfile.ZipFile(target, "r") as archive:
        root = _manifest(archive, "manifest.json")
        nested = slot["artifact_scope"] == "NESTED_CHILD_OF_COMPLETE_PACKAGE"
        nested_sha = ""
        if nested:
            if root.get("schema") != "alpr.package.v1":
                raise ValueError("Wynik nie jest kompletnym pakietem ALPR.")
            entry = (root.get("models") or {}).get(slot["source_role"])
            if not isinstance(entry, dict):
                raise ValueError(f"Brak modelu {slot['role']} w wyniku.")
            prefix = "models/" + slot["source_role"] + "/"
            sidecar = prefix + "manifest.json"
            child = prefix + "model.alprmodel"
            if entry.get("manifest_file") != sidecar or entry.get("package_file") != child:
                raise ValueError("Nieoczekiwane ścieżki modeli potomnych.")
            hashes = entry.get("sha256") or {}
            if not isinstance(hashes, dict):
                raise ValueError("Brak SHA plików modelu potomnego.")
            if hashes.get(sidecar) != _sha_archive_member(archive, sidecar):
                raise ValueError("SHA manifestu potomnego niezgodne.")
            nested_sha = _sha_archive_member(archive, child)
            if hashes.get(child) != nested_sha:
                raise ValueError("SHA paczki potomnej niezgodne.")
            model = _manifest(archive, sidecar)
        else:
            if root.get("schema") != "alpr.model.v1":
                raise ValueError("Wynik nie jest pojedynczym modelem ALPR.")
            model = root
        if model.get("role") != slot["source_role"] or model.get("schema") != "alpr.model.v1":
            raise ValueError("Rola/schemat modelu w wynikowej paczce niezgodne.")
        source = model.get("source")
        actual_checkpoint_sha = str(source.get("checkpoint_sha256") or "").lower() if isinstance(source, dict) else ""
        if actual_checkpoint_sha != slot["checkpoint_sha256"]:
            raise ValueError("Wynikowy model nie pochodzi z przypiętego checkpointu SHA.")
        variants = model.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("Wynikowy model nie zawiera wariantów.")
        actual_variants = [{"runtime": str(v.get("runtime") or "").lower(),
                            "precision": str(v.get("precision") or "").lower()}
                           for v in variants if isinstance(v, dict)]
        if not all(expected in actual_variants for expected in slot["requested_variants"]):
            raise ValueError("Wynikowe warianty nie odpowiadają zleceniu konwersji.")
    return {
        "actual_variants": actual_variants,
        "model_id": str(model.get("model_id") or ""),
        "embedded_package_sha256": nested_sha,
        "verification": "EXPORTER_COMPLETED_PLUS_SHA_AND_FINAL_ARCHIVE_MANIFEST",
        "android_approved": False,
        "final_test_verified": False,
    }


def record_conversion_results(root: Path, plan: list[dict], *, artifact_path: Path) -> list[str]:
    """Never report SUCCEEDED without output SHA/role/variants evidence."""
    warnings = []
    for slot in plan:
        try:
            attestation = attest_conversion(slot, artifact_path)
            append_event(
                root, operation_id=slot["operation_id"], operation="MODEL_CONVERSION",
                outcome="SUCCEEDED", artifact_path=str(artifact_path),
                detail=f"{slot['role']}: model z checkpointu odnaleziony w wyniku eksportu; bez zatwierdzenia Android.",
                conversion={**slot, **attestation},
            )
        except Exception as exc:
            reason = f"{slot['role']}: nie można potwierdzić konwersji: {type(exc).__name__}: {exc}"
            warnings.append(reason)
            try:
                append_event(
                    root, operation_id=slot["operation_id"], operation="MODEL_CONVERSION",
                    outcome="UNCONFIRMED", artifact_path=str(artifact_path),
                    detail=reason, conversion=slot,
                )
            except Exception as journal_error:
                warnings.append(f"{slot['role']}: rejestr zdarzeń: {journal_error}")
    return warnings


def record_conversion_unconfirmed(root: Path, plan: list[dict], *, reason: str) -> list[str]:
    """Exporter failure does not prove every individual child conversion failed."""
    warnings = []
    for slot in plan:
        try:
            append_event(
                root, operation_id=slot["operation_id"], operation="MODEL_CONVERSION",
                outcome="UNCONFIRMED", detail=f"{slot['role']}: zakończenie niepotwierdzone; {reason}"[:750],
                conversion=slot,
            )
        except Exception as exc:
            warnings.append(f"{slot['role']}: nie można zapisać statusu UNCONFIRMED: {exc}")
    return warnings
