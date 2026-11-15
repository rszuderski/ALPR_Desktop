#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded, read-only discovery of INT8 manifests in *managed report locations*.

V3: never descend through unrelated image/training/diagnostic trees.  The old
Workspace-wide walk could exhaust its directory/file budget before reaching an
experiment's ``raporty_do_pracy``.  Only report, calibration and shallow project
containers are searched; ordinary files do NOT consume the JSON candidate
limit.  Discovery is a prefilter; the GUI MUST run the authoritative manifest
validator against the live checkpoint and settings before binding anything.

The resulting uniqueness is within the documented managed-report search scope,
not a claim of globally unique files anywhere on disk.  A truncated search is
never safe for automatic binding.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

SCHEMAS = frozenset({
    'alpr.int8_calibration.exploratory.v1',
    'alpr.int8_calibration.exploratory.v2',
    'alpr.int8_calibration.v1',
})
SKIP_DIRS = frozenset({
    '.git', '.venv', 'venv', 'env', '__pycache__', '.pytest_cache',
    'node_modules', 'images', 'labels', 'weights', 'crops', 'crop',
    'train', 'val', 'test', 'mobile_packages', 'mobile_export_diagnostics',
    '_registry', '.tox', 'site-packages', 'models',
    '3_cropped_characters', '4_training_datasets', '5_training_runs',
    '6_models', '1_images', '2_annotations',
})
REPORT_TOKENS = ('raport', 'report', 'calibr', 'kalibr', 'manifest',
                 'int8', 'wynik', 'result', 'audyt', 'audit')
PROJECT_COLLECTIONS = frozenset({
    '9_projects', 'projects', 'projekty', 'experiments',
    'eksperymenty', 'badania', 'research', 'output', 'exports',
    'eksporty', 'artifacts',
})
MAX_REPORT_NESTING = 7
MAX_JSON_BYTES = 16 * 1024 * 1024


def _json_candidate(name: str, directory: Path) -> bool:
    lower = name.casefold()
    if not lower.endswith('.json'):
        return False
    return ('calib' in lower or 'kalibr' in lower or 'int8' in lower or
            (lower == 'manifest.json' and
             any('calib' in part.casefold() or 'kalibr' in part.casefold() or
                 'int8' in part.casefold() for part in directory.parts)))


def _matches_metadata(document, *, role: str, checkpoint_sha256: str,
                      image_size: int, confidence_threshold: float,
                      iou_threshold: float, mode: str):
    if not isinstance(document, dict):
        return None
    schema = document.get('schema')
    status = document.get('status')
    expected_status = 'EXPLORATORY_ONLY' if mode == 'exploratory' else 'READY'
    if schema not in SCHEMAS or status != expected_status:
        return None
    if mode == 'exploratory' and '.exploratory.' not in schema:
        return None
    if mode == 'strict' and '.exploratory.' in schema:
        return None
    if document.get('split') != 'train' or document.get('role') != role:
        return None
    checkpoint = document.get('checkpoint')
    if not isinstance(checkpoint, dict) or str(checkpoint.get('sha256') or '').lower() != checkpoint_sha256.lower():
        return None
    prep = document.get('preprocessing')
    if not isinstance(prep, dict):
        return None
    try:
        if (int(prep.get('width')), int(prep.get('height'))) != (image_size, image_size):
            return None
        config = document.get('config')
        if not isinstance(config, dict):
            return None
        conf = float(config['confidence_threshold'])
        iou = float(config['iou_threshold'])
        if not (math.isfinite(conf) and math.isfinite(iou)):
            return None
        if abs(conf - confidence_threshold) > 1e-7 or abs(iou - iou_threshold) > 1e-7:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    fingerprint = str(document.get('input_fingerprint') or '').lower()
    if not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
        return None
    samples = document.get('samples')
    if not isinstance(samples, list) or not samples:
        return None
    return {
        'schema': schema,
        'status': status,
        'fingerprint': fingerprint,
        'sample_count': len(samples),
        'role': role,
    }


def _is_report_dir(name: str) -> bool:
    token = name.casefold()
    return any(part in token for part in REPORT_TOKENS)


def _safe_to_descend(*, directory_name: str, parent: Path,
                     search_root: Path, max_report_nesting: int = MAX_REPORT_NESTING) -> bool:
    """Do not walk datasets and diagnostic trees merely to find JSON filenames.

    Paths are relative to the managed root, e.g. ``Workspace`` or ``output``.
    Search direct files in top-level experiments and project collections, and
    recurse into report/calibration trees.  Descend into project identifiers only
    when their immediate parent is an explicitly named project collection.
    """
    name = directory_name.casefold()
    if name in SKIP_DIRS or name.startswith('.'):
        return False
    child = parent / directory_name
    if child.is_symlink():
        return False
    try:
        relative_parent = parent.relative_to(search_root)
    except ValueError:
        return False
    parent_parts = tuple(part.casefold() for part in relative_parent.parts)
    child_depth = len(parent_parts) + 1
    if any(_is_report_dir(part) for part in parent_parts) or _is_report_dir(search_root.name):
        # A report root is bounded but can contain nested versions / subreports.
        return child_depth <= max_report_nesting
    if _is_report_dir(directory_name):
        return child_depth <= max_report_nesting
    if child_depth == 1:
        # Scan direct files and report children in Workspace/<experiment>.
        return True
    if name in PROJECT_COLLECTIONS and child_depth <= 3:
        return True
    if parent_parts and parent_parts[-1] in PROJECT_COLLECTIONS and child_depth <= 4:
        return True
    return False


def discover_matching_calibrations(
    repo_root: Path, *, role: str, checkpoint_sha256: str,
    image_size: int, confidence_threshold: float, iou_threshold: float,
    mode: str, max_directories: int = 3500,
    max_files: int = 80000, max_matches: int = 32,
) -> dict:
    """Find prefiltered manifests in managed report/calibration roots.

    ``truncated=True`` means uniqueness has NOT been established, even if a
    candidate is present.  ``files_scanned`` counts only candidate JSON files,
    not unrelated images, plots or logs.  Full validation is a GUI responsibility.
    """
    root = Path(repo_root).resolve()
    if role not in {'plate', 'character', 'vehicle'}:
        raise ValueError('Nieobsługiwana rola manifestu.')
    if mode not in {'strict', 'exploratory'}:
        raise ValueError('Nieobsługiwany tryb manifestu.')
    if not re.fullmatch('[0-9a-fA-F]{64}', str(checkpoint_sha256)):
        raise ValueError('Nieprawidłowy SHA-256 checkpointu.')
    if image_size <= 0 or not all(math.isfinite(v) and 0.0 <= v <= 1.0
                                  for v in (confidence_threshold, iou_threshold)):
        raise ValueError('Nieprawidłowe parametry wejścia manifestu.')
    if max_directories < 1 or max_files < 1 or max_matches < 1:
        raise ValueError('Limity wyszukiwania muszą być dodatnie.')
    results = []
    scanned_dirs = 0
    scanned_files = 0
    truncated = False
    limit_reason = ''
    visited = set()
    search_roots = [root / 'Workspace', root / 'output', root / 'calibration', root / 'calibrations']
    for search_root in search_roots:
        if not search_root.is_dir() or search_root.is_symlink():
            continue

        def _on_walk_error(err):
            nonlocal truncated, limit_reason
            truncated = True
            limit_reason = 'unreadable_directory'

        for current, directories, filenames in os.walk(search_root, followlinks=False, onerror=_on_walk_error):
            if truncated:
                break
            current_path = Path(current)
            try:
                key = str(current_path.resolve()).casefold()
            except OSError:
                truncated = True
                limit_reason = 'unreadable_directory'
                break
            if key in visited:
                directories[:] = []
                continue
            visited.add(key)
            scanned_dirs += 1
            if scanned_dirs > max_directories:
                truncated = True
                limit_reason = 'directory_limit'
                break
            directories[:] = sorted(
                (name for name in directories
                 if _safe_to_descend(directory_name=name, parent=current_path,
                                     search_root=search_root)),
                key=lambda name: (not _is_report_dir(name), name.casefold()),
            )
            for file_name in sorted(filenames):
                if not _json_candidate(file_name, current_path):
                    continue
                scanned_files += 1
                if scanned_files > max_files:
                    truncated = True
                    limit_reason = 'candidate_file_limit'
                    break
                path = current_path / file_name
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
                        continue
                    with path.open('r', encoding='utf-8-sig') as stream:
                        document = json.load(stream)
                except (OSError, UnicodeError, ValueError, TypeError):
                    continue
                match = _matches_metadata(
                    document, role=role, checkpoint_sha256=checkpoint_sha256,
                    image_size=image_size, confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold, mode=mode,
                )
                if match:
                    results.append({'path': str(path.resolve()), **match})
                    if len(results) >= max_matches:
                        truncated = True
                        limit_reason = 'match_limit'
                        break
            if truncated:
                break
        if truncated:
            break
    return {
        'matches': sorted(results, key=lambda item: item['path'].casefold()),
        'truncated': truncated,
        'directories_scanned': scanned_dirs,
        'files_scanned': scanned_files,
        'limit_reason': limit_reason,
        'search_scope': 'MANAGED_REPORT_AND_CALIBRATION_ROOTS',
        'authority': 'PREFILTER_ONLY_VALIDATE_BEFORE_BIND',
    }
