#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4 guard: compose an ALPR bundle only from verified, existing mobile packages.

This guard does not approve models for Android, alter calibration status,
convert a checkpoint, write a package or mutate the experiment history.
The existing Z4 role-specific TFLite validators remain authoritative.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


class F4NoConversionError(ValueError):
    """An F4 no-conversion operation is not safe to proceed."""


def require_f4_selection(*, has_vehicle: bool, use_mt: bool, use_mz: bool,
                         mt_precision: str, mz_precision: str) -> str:
    """Reject any choice that could dispatch a checkpoint conversion."""
    if has_vehicle:
        raise F4NoConversionError(
            'Wybór gotowych modeli: wybierz zestaw MT+MZ bez MP; brak gotowego MP w tej ścieżce.'
        )
    if not use_mt:
        raise F4NoConversionError(
            'Wybór gotowych modeli: MT wymaga opcji Użyj gotowego MT i wyboru .alprmodel z biblioteki.'
        )
    if not use_mz:
        raise F4NoConversionError(
            'Wybór gotowych modeli: MZ wymaga opcji Użyj gotowego MZ i wyboru .alprmodel z biblioteki. '
            'Surowy MZ-DAY CLEAN .tflite nie jest zatwierdzoną paczką.'
        )
    mt = str(mt_precision or '').strip().lower()
    mz = str(mz_precision or '').strip().lower()
    if mt not in {'int8', 'fp32'} or mz not in {'int8', 'fp32'}:
        raise F4NoConversionError('Wybór gotowych modeli: wybierz jawnie INT8 albo FP32 w obu modelach.')
    if mt != mz:
        raise F4NoConversionError(
            f'Wybór gotowych modeli: para ma różne precyzje MT={mt.upper()} i MZ={mz.upper()}. '
            'Do porównania wybierz dwie paczki tej samej precyzji.'
        )
    return mt


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_f4_no_conversion_request(request, *, require_fresh_destination: bool = False) -> dict:
    """Recheck a composed request before export, including pinned on-disk SHA.

    Must be called *after* both existing-package full-graph validators. Do not
    interpret this lightweight guard as independent graph/mobile approval.
    """
    if any(getattr(request, key, None) is not None for key in
           ('plate_request', 'character_request', 'vehicle_request', 'vehicle_package')):
        raise F4NoConversionError(
            'Wybór gotowych modeli: żądanie zawiera nowy eksport checkpointu albo MP; operacja zablokowana.'
        )
    metadata = getattr(request, 'metadata', None)
    if not isinstance(metadata, dict):
        raise F4NoConversionError('Wybór gotowych modeli: brak metadanych zatwierdzonego źródła MT/MZ.')
    output_raw = str(getattr(request, 'destination', '') or '').strip()
    if not output_raw or Path(output_raw).suffix.lower() != '.alprmodel':
        raise F4NoConversionError('Wybór gotowych modeli: wynik musi być plikiem .alprmodel.')
    selected_output = Path(output_raw)
    # The preview/preflight uses the suggested filename, which can already exist.
    # Enforce no-clobber ONLY after the operator chooses the real output path.
    # Check the symlink itself too: Path.exists() is false for broken links.
    if require_fresh_destination and (selected_output.exists() or selected_output.is_symlink()):
        raise F4NoConversionError(
            'Wybór gotowych modeli: docelowy plik już istnieje. Wybierz NOWĄ nazwę; '
            'nadpisanie wcześniejszego pakietu ALPR jest zablokowane.'
        )
    output = selected_output.resolve()

    precisions = {}
    checked = {}
    for marker, field, proof_field in (
        ('MT', 'plate_package', 'reused_verified_mt'),
        ('MZ', 'character_package', 'reused_verified_mz'),
    ):
        raw = str(getattr(request, field, '') or '').strip()
        if not raw or Path(raw).suffix.lower() != '.alprmodel':
            raise F4NoConversionError(f'Wybór gotowych modeli: {marker} wymaga gotowej paczki .alprmodel.')
        source = Path(raw)
        if source.is_symlink() or not source.is_file():
            raise F4NoConversionError(f'Wybór gotowych modeli: {marker} nie jest zwykłym, istniejącym plikiem.')
        source = source.resolve()
        if output == source:
            raise F4NoConversionError(
                f'Wybór gotowych modeli: plik docelowy jest źródłem {marker}; nadpisanie modelu zablokowane.'
            )
        proof = metadata.get(proof_field)
        if not isinstance(proof, dict):
            raise F4NoConversionError(f'Wybór gotowych modeli: brak pełnej weryfikacji źródła {marker}.')
        proof_path = str(proof.get('path') or '').strip()
        if not proof_path or Path(proof_path).resolve() != source:
            raise F4NoConversionError(f'Wybór gotowych modeli: {marker} wskazuje inną paczkę niż zweryfikowana.')
        expected_sha = str(proof.get('package_sha256') or '').strip().lower()
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
            raise F4NoConversionError(f'Wybór gotowych modeli: brak zweryfikowanego SHA pakietu {marker}.')
        if _file_digest(source) != expected_sha:
            raise F4NoConversionError(f'Wybór gotowych modeli: zmieniono paczkę {marker} od ostatniej kontroli SHA.')
        checked[marker] = {'path': str(source), 'sha256': expected_sha}
        precisions[marker] = str(proof.get('precision') or '').strip().lower()
    precise = require_f4_selection(
        has_vehicle=False, use_mt=True, use_mz=True,
        mt_precision=precisions['MT'], mz_precision=precisions['MZ'],
    )
    checked['precision'] = precise
    checked['android_execution'] = 'NOT_VERIFIED'
    return checked
