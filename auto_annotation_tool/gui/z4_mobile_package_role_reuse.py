#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed reuse of an already-exported MZ .alprmodel.

A discovery label is NEVER proof of graph precision, checkpoint identity, or
Android compatibility. No conversions, package writes or approvals occur here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable
import zipfile


class ReusedMzPackageError(ValueError):
    """Invalid or unverified candidate for reuse as MZ."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_mz_package(
    package_path: Path, selected_checkpoint: Path, *,
    expected_image_size: int, validate_package: Callable,
    expected_precision: str | None = None,
) -> dict:
    """Verify full graph+ZIP, MZ role and live source-checkpoint SHA before reuse.

    Caller MUST pass MobileModelExporter.validate_package. The returned package
    fingerprint must be checked again before taking a snapshot for the bundle.
    An explicit expected_precision makes F4 bundle selection fail closed: one
    TFLite variant, never an implicit FP32 fallback for requested INT8.
    Legacy callers can omit the value for inspection only; Z4 always pins it.
    """
    requested = None if expected_precision is None else str(expected_precision or "").strip().lower()
    if requested is not None and requested not in {"int8", "fp32"}:
        raise ReusedMzPackageError("MZ: wybierz jawnie precyzję INT8 lub FP32.")
    raw = str(package_path or "").strip()
    if not raw or Path(raw).suffix.lower() != ".alprmodel":
        raise ReusedMzPackageError("MZ: wymagany gotowy .alprmodel; badawczy .tflite nie jest pakietem.")
    package = Path(raw).resolve()
    checkpoint = Path(selected_checkpoint).resolve()
    if not package.is_file() or Path(raw).is_symlink():
        raise ReusedMzPackageError("MZ: brak gotowego pliku lub dowiązanie symboliczne.")
    if not checkpoint.is_file():
        raise ReusedMzPackageError("MZ: brak wybranego best.pt.")
    if not callable(validate_package):
        raise ReusedMzPackageError("MZ: brak pełnego walidatora; ponowne użycie zablokowane.")
    try:
        image_size = int(expected_image_size)
    except (ValueError, TypeError) as exc:
        raise ReusedMzPackageError("MZ: nie można potwierdzić imgsz.") from exc
    if image_size < 128 or image_size > 2048:
        raise ReusedMzPackageError("MZ: imgsz poza zakresem 128–2048.")
    package_sha = _sha256(package)
    checkpoint_sha = _sha256(checkpoint)
    try:
        # Authoritative exporter checks ZIP entries, nested hashes and TFLite graph.
        validate_package(package, require_precision=True)
        with zipfile.ZipFile(package, "r") as archive:
            matches = [part for part in archive.infolist() if part.filename == "manifest.json"]
            if len(matches) != 1 or matches[0].file_size > 2 * 1024 * 1024 or matches[0].file_size < 2:
                raise ReusedMzPackageError("MZ: manifest.json brakujący, powielony lub zbyt duży.")
            manifest = json.loads(archive.read(matches[0]).decode("utf-8-sig"))
    except ReusedMzPackageError:
        raise
    except Exception as exc:
        raise ReusedMzPackageError(f"MZ: pełna walidacja paczki nie przeszła: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "alpr.model.v1":
        raise ReusedMzPackageError("MZ: wymagany pojedynczy pakiet alpr.model.v1.")
    if manifest.get("role") != "character" or manifest.get("task") != "detect":
        raise ReusedMzPackageError("MZ: pakiet musi mieć role=character oraz task=detect.")
    source = manifest.get("source")
    source_sha = str(source.get("checkpoint_sha256") or "").lower() if isinstance(source, dict) else ""
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha) or source_sha != checkpoint_sha:
        raise ReusedMzPackageError("MZ: SHA checkpointu nie zgadza się z wybranym modelem znaków.")
    input_spec = manifest.get("input")
    if not isinstance(input_spec, dict) or (input_spec.get("width"), input_spec.get("height")) != (image_size, image_size):
        raise ReusedMzPackageError("MZ: imgsz istniejącego pakietu nie odpowiada wybranemu checkpointowi.")
    labels = manifest.get("labels")
    output = manifest.get("output")
    if (not isinstance(labels, list) or not labels or not isinstance(output, dict)
            or int(output.get("class_count") or 0) != len(labels)):
        raise ReusedMzPackageError("MZ: niekompletna lista klas znaków lub niespójne wyjście.")
    try:
        confidence = float(output["confidence_threshold"])
        iou = float(output["iou_threshold"])
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (confidence, iou)):
            raise ValueError("threshold outside [0,1]")
    except (ValueError, TypeError, KeyError) as exc:
        raise ReusedMzPackageError("MZ: nieprawidłowe progi confidence/IoU.") from exc
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ReusedMzPackageError("MZ: brak wariantów mobilnych.")
    if requested is not None:
        if (len(variants) != 1 or not isinstance(variants[0], dict)
                or variants[0].get("runtime") != "tflite"
                or str(variants[0].get("precision") or "").lower() != requested):
            raise ReusedMzPackageError(
                f"MZ: dla F4 wymagany dokładnie jeden wariant TFLite {requested.upper()}; "
                "nie wolno automatycznie wybierać innej precyzji ani paczki mieszanej."
            )
        expected_input = "INT8" if requested == "int8" else "FLOAT32"
        if str(input_spec.get("data_type") or "").upper() != expected_input:
            raise ReusedMzPackageError(
                f"MZ: precyzja {requested.upper()} wymaga wejścia {expected_input} zgodnego z grafem."
            )
    precisions = []
    for variant in variants:
        if not isinstance(variant, dict) or variant.get("runtime") != "tflite":
            raise ReusedMzPackageError("MZ: wybór biblioteki wymaga wyłącznie wariantów TFLite.")
        precision = str(variant.get("precision") or "").lower()
        if precision not in {"fp32", "int8"}:
            raise ReusedMzPackageError("MZ: nieobsługiwana precyzja wariantu.")
        if precision == "int8" and (not isinstance(variant.get("graph_precision"), dict)
                                    or not variant["graph_precision"]):
            raise ReusedMzPackageError("MZ: brak dowodu analizy grafu INT8.")
        precisions.append(precision)
    if len(set(precisions)) == 1 and precisions[0] == "int8":
        if str(input_spec.get("data_type") or "").upper() != "INT8":
            raise ReusedMzPackageError("MZ: paczka INT8 nie deklaruje wejścia signed INT8.")
    calibration_status = "NOT_APPLICABLE_FP32"
    if "int8" in precisions:
        evidence = manifest.get("calibration_evidence")
        if not isinstance(evidence, dict) or not evidence.get("file"):
            raise ReusedMzPackageError("MZ: brak dowodu kalibracji INT8.")
        mode = str(evidence.get("mode") or "").lower()
        if mode not in {"strict", "exploratory"}:
            raise ReusedMzPackageError("MZ: nieznany tryb kalibracji INT8.")
        calibration_status = "EXPLORATORY_ONLY" if mode == "exploratory" else "READY"
        if mode == "strict" and evidence.get("status") != "READY":
            raise ReusedMzPackageError("MZ: brak statusu READY ścisłej kalibracji.")
        if calibration_status == "EXPLORATORY_ONLY":
            research = manifest.get("research_calibration") or {}
            if evidence.get("status") != "EXPLORATORY_ONLY" or research.get("final_test_status") != "NOT_VERIFIED":
                raise ReusedMzPackageError("MZ: niekompletne ograniczenia badawcze kalibracji.")
    if _sha256(package) != package_sha or _sha256(checkpoint) != checkpoint_sha:
        raise ReusedMzPackageError("MZ: paczka lub checkpoint zmienił się podczas walidacji SHA.")
    return {
        "path": str(package), "package_sha256": package_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model_id": str(manifest.get("model_id") or ""),
        "precisions": list(dict.fromkeys(precisions)),
        "precision": requested if requested is not None else (precisions[0] if len(set(precisions)) == 1 else "mixed"),
        "runtime": "tflite", "imgsz": image_size,
        "confidence_threshold": confidence, "iou_threshold": iou,
        "calibration_status": calibration_status,
        "execution_precision_android": "UNVERIFIED",
    }
