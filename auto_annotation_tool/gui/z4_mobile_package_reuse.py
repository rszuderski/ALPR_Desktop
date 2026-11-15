#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict, precision-pinned reuse of an existing MT TFLite .alprmodel.

Important: This helper NEVER re-exports, rewrites or approves MZ. The full
static graph and nested package check is delegated to MobileModelExporter.
The original package SHA is pinned again before long MZ conversion.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable
import zipfile


class ReusedMtPackageError(ValueError):
    """Fail-closed reusable-package validation failure."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_mt_package(
    package_path: Path,
    selected_checkpoint: Path,
    *,
    expected_image_size: int,
    validate_package: Callable,
    expected_precision: str = "int8",
) -> dict:
    """Verify one TFLite MT variant pinned to an explicitly requested precision.

    ``validate_package`` MUST be MobileModelExporter.validate_package, which
    checks ZIP entries, embedded weights SHA, calibration and actual TFLite graph.
    This helper does not trust declared precision or unchecked ZIP metadata.
    """
    expected = str(expected_precision or '').strip().lower()
    if expected not in {'int8', 'fp32'}:
        raise ReusedMtPackageError('MT: precyzja gotowego pakietu musi być jawnie INT8 albo FP32.')
    package = Path(package_path).resolve()
    checkpoint = Path(selected_checkpoint).resolve()
    if not str(package_path).strip() or package.suffix.lower() != '.alprmodel':
        raise ReusedMtPackageError('MT: wskaż gotowy plik .alprmodel, nie checkpoint .pt.')
    if not package.is_file():
        raise ReusedMtPackageError(f'MT: brak wskazanego pakietu: {package}')
    if not checkpoint.is_file():
        raise ReusedMtPackageError(f'MT: brak wybranego checkpointu: {checkpoint}')
    if not callable(validate_package):
        raise ReusedMtPackageError('MT: brak pełnej walidacji eksportera; użycie gotowego MT zablokowane.')
    try:
        width = int(expected_image_size)
    except (ValueError, TypeError) as exc:
        raise ReusedMtPackageError('MT: nie można potwierdzić imgsz wybranego modelu.') from exc
    if width < 128 or width > 2048:
        raise ReusedMtPackageError('MT: nieprawidłowe imgsz wybranego modelu.')

    source_package_sha = file_sha256(package)
    checkpoint_sha = file_sha256(checkpoint)
    try:
        # This verifies the actual TFLite graph rather than trusting a manifest label.
        validate_package(package, require_precision=True)
        with zipfile.ZipFile(package, 'r') as archive:
            members = [part for part in archive.infolist() if part.filename == 'manifest.json']
            if len(members) != 1 or members[0].is_dir() or not 2 <= members[0].file_size <= 2 * 1024 * 1024:
                raise ReusedMtPackageError('MT: wymagany jeden poprawny manifest.json (maks. 2 MiB).')
            manifest = json.loads(archive.read(members[0]).decode('utf-8-sig'))
    except ReusedMtPackageError:
        raise
    except Exception as exc:
        raise ReusedMtPackageError(f'MT: pełna walidacja istniejącego pakietu nie przeszła: {exc}') from exc

    if not isinstance(manifest, dict) or manifest.get('schema') != 'alpr.model.v1':
        raise ReusedMtPackageError('MT: wymagany pojedynczy pakiet alpr.model.v1.')
    if manifest.get('role') != 'plate' or manifest.get('task') != 'pose':
        raise ReusedMtPackageError('MT: pakiet musi zawierać model tablic Pose (role=plate).')
    source = manifest.get('source') or {}
    stored_checkpoint_sha = str(source.get('checkpoint_sha256') or '').lower()
    if not re.fullmatch('[0-9a-f]{64}', stored_checkpoint_sha) or stored_checkpoint_sha != checkpoint_sha:
        raise ReusedMtPackageError('MT: SHA checkpointu wewnątrz pakietu nie zgadza się z wybranym MT-DAY.')
    input_spec = manifest.get('input') or {}
    if (int(input_spec.get('width') or 0), int(input_spec.get('height') or 0)) != (width, width):
        raise ReusedMtPackageError(f'MT: imgsz pakietu różni się od wybranego modelu ({width}x{width}).')
    labels = list(manifest.get('labels') or [])
    output = manifest.get('output') or {}
    try:
        confidence = float(output['confidence_threshold'])
        iou = float(output['iou_threshold'])
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (confidence, iou)):
            raise ValueError('threshold outside [0,1]')
    except (KeyError, TypeError, ValueError) as exc:
        raise ReusedMtPackageError('MT: nieprawidłowe progi conf/IoU w gotowym pakiecie.') from exc
    if len(labels) != 1 or int(output.get('class_count') or 0) != 1 or int(output.get('keypoint_count') or 0) < 4:
        raise ReusedMtPackageError('MT: wymagany jednoklasowy model Pose z co najmniej 4 narożnikami.')
    variants = list(manifest.get('variants') or [])
    if len(variants) != 1:
        raise ReusedMtPackageError('MT: pakiet musi zawierać dokładnie jeden wariant (bez FP32 fallback).')
    variant = variants[0]
    if not isinstance(variant, dict) or variant.get('runtime') != 'tflite' or variant.get('precision') != expected:
        raise ReusedMtPackageError(f'MT: wymagany wyłącznie wariant tflite/{expected}; bez domyślnej zamiany precyzji.')
    input_type = str(input_spec.get('data_type') or '').upper()
    if expected == 'int8':
        if input_type != 'INT8':
            raise ReusedMtPackageError('MT: istniejący pakiet nie ma wejścia signed INT8.')
        if not isinstance(variant.get('graph_precision'), dict) or not variant['graph_precision']:
            raise ReusedMtPackageError('MT: brak dowodu analizy grafu INT8 w wariancie.')
        calibration = manifest.get('calibration_evidence') or {}
        if not isinstance(calibration, dict) or not calibration.get('file'):
            raise ReusedMtPackageError('MT: brak zatwierdzonego lub badawczego dowodu kalibracji.')
        status = 'EXPLORATORY_ONLY' if calibration.get('mode') == 'exploratory' else 'READY'
        if status == 'EXPLORATORY_ONLY':
            research = manifest.get('research_calibration') or {}
            if calibration.get('status') != status or research.get('final_test_status') != 'NOT_VERIFIED':
                raise ReusedMtPackageError('MT: niepełne oznaczenie ograniczeń kalibracji badawczej.')
    else:
        if input_type != 'FLOAT32':
            raise ReusedMtPackageError('MT: FP32 musi mieć wejście FLOAT32 zgodne z rzeczywistym grafem.')
        # FP32 does not require INT8 calibration evidence; graph truth belongs
        # to the mandatory MobileModelExporter.validate_package(..., require_precision=True).
        status = 'NOT_APPLICABLE_FP32'
    if file_sha256(package) != source_package_sha or file_sha256(checkpoint) != checkpoint_sha:
        raise ReusedMtPackageError('MT: pakiet lub checkpoint zmienił się podczas kontroli SHA.')
    return {
        'path': str(package),
        'package_sha256': source_package_sha,
        'checkpoint_sha256': checkpoint_sha,
        'model_id': str(manifest.get('model_id') or ''),
        'variant_id': str(variant.get('id') or ''),
        'precision': expected,
        'format': 'tflite',
        'imgsz': width,
        'confidence_threshold': confidence,
        'iou_threshold': iou,
        'calibration_status': status,
        'execution_precision_android': 'UNVERIFIED',
    }
