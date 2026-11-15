#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4 inventory -> explicit pair selection, without converting or approving models.

This is a MANIFEST-ONLY binding gate. The existing MobileModelExporter and
role-specific validators remain the authority for TFLite graph verification.
It never exports, packages, mutates source artifacts or authorizes Android F4.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

from .z4_mobile_f4_inventory import _manifest_candidate

SHA_RE = re.compile(r'[0-9a-f]{64}\Z')


class F4PairBindingError(ValueError):
    """The F4 matrix choice is stale, ambiguous or does not match selected checkpoints."""


def _require_file(value, role: str, suffix: str) -> Path:
    raw = str(value or '').strip()
    if not raw or Path(raw).suffix.lower() != suffix:
        raise F4PairBindingError(f'{role}: wybierz plik {suffix}.')
    original = Path(raw)
    if original.is_symlink() or not original.is_file():
        raise F4PairBindingError(f'{role}: plik nie istnieje lub jest dowiązaniem symbolicznym.')
    return original.resolve()


def validate_f4_pair_choice(
    report: dict,
    *,
    precision: str,
    package_paths: dict,
    checkpoint_paths: dict,
    expected_image_size: dict,
    read_archive: Callable,
    sha256_file: Callable,
) -> dict:
    """Re-read *both* selected packages before binding, fail closed on any drift.

    `report` must have come from build_f4_inventory for the current two
    checkpoint candidates. This helper reads real bytes but not neural graphs.
    Returned SHA pins are diagnostic at selection time; the later preflight
    must revalidate graph and source checkpoint regardless of these results.
    """
    if not isinstance(report, dict) or report.get('schema') != 'alpr.desktop.f4_candidate_inventory.v1':
        raise F4PairBindingError('Wybór gotowych modeli: brak wiarygodnego wyniku macierzy; ponów skan.')
    if report.get('scan_truncated'):
        raise F4PairBindingError('Wybór gotowych modeli: skan był niekompletny — nie wolno wybierać pary automatycznie.')
    fmt = str(precision or '').strip().lower()
    if fmt not in {'fp32', 'int8'}:
        raise F4PairBindingError('Wybór gotowych modeli: wybierz FP32 albo INT8.')
    pair = (report.get('pairs') or {}).get(fmt) or {}
    if pair.get('state') != 'MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION':
        raise F4PairBindingError(f'Nie znaleziono obu modeli {fmt.upper()} pasujących do wybranych treningów.')
    if not callable(read_archive) or not callable(sha256_file):
        raise F4PairBindingError('Wybór gotowych modeli: brak funkcji odczytu manifestów lub SHA-256.')

    by_role = {}
    chosen = []
    for role in ('MT','MZ'):
        path = _require_file((package_paths or {}).get(role), role, '.alprmodel')
        selected = {str(Path(str(item.get('path') or '')).resolve())
                    for item in ((pair.get('candidates') or {}).get(role) or [])
                    if isinstance(item, dict) and item.get('path')}
        if str(path) not in selected:
            raise F4PairBindingError(f'{role}: paczka nie pochodzi z aktualnego przeglądu modeli.')
        if path in chosen:
            raise F4PairBindingError('Wybór gotowych modeli: MT i MZ muszą być dwoma odrębnymi paczkami.')
        chosen.append(path)

        checkpoint = _require_file((checkpoint_paths or {}).get(role), role+' checkpoint', '.pt')
        recorded = str((report.get('selected_checkpoint_sha256') or {}).get(role) or '').lower()
        if not SHA_RE.fullmatch(recorded):
            raise F4PairBindingError(f'{role}: macierz nie ma SHA-256 wybranego checkpointu.')
        actual_checkpoint_sha = str(sha256_file(checkpoint)).lower()
        if actual_checkpoint_sha != recorded:
            raise F4PairBindingError(f'{role}: checkpoint zmienił się od skanowania macierzy (SHA).')
        try:
            expected_size = int((expected_image_size or {})[role])
        except (KeyError, TypeError, ValueError) as exc:
            raise F4PairBindingError(f'{role}: brak rozmiaru wejścia wybranego modelu.') from exc
        if expected_size < 128 or expected_size > 2048:
            raise F4PairBindingError(f'{role}: nieprawidłowe imgsz.')

        before_sha = str(sha256_file(path)).lower()
        if not SHA_RE.fullmatch(before_sha):
            raise F4PairBindingError(f'{role}: nie udało się obliczyć SHA paczki.')
        row = read_archive(path)
        okay, reason, details = _manifest_candidate(row, role, fmt, recorded)
        if not okay:
            raise F4PairBindingError(f'{role}: paczka nie przechodzi ponownej kontroli manifestu ({reason}).')
        if str(Path(str(getattr(row, 'path', '') or '')).resolve()) != str(path):
            raise F4PairBindingError(f'{role}: odczyt manifestu pochodzi z innego pliku.')
        if int(details['image_size']) != expected_size:
            raise F4PairBindingError(
                f'{role}: imgsz paczki {details["image_size"]} != wybrany model {expected_size}.'
            )
        # Detect modifications between SHA and manifest inspection.
        after_sha = str(sha256_file(path)).lower()
        if before_sha != after_sha:
            raise F4PairBindingError(f'{role}: paczka zmieniła się podczas kontroli SHA.')
        by_role[role] = {
            'path': str(path), 'package_sha256_at_selection': after_sha,
            'checkpoint_sha256': recorded, 'image_size': expected_size,
            'precision': fmt, 'authority': 'MANIFEST_RECHECK_ONLY_NOT_GRAPH_VALIDATED',
        }
    return {
        'precision': fmt,
        'roles': by_role,
        'desktop_graph_verified': False,
        'android_import_approved': False,
        'export_started': False,
    }
