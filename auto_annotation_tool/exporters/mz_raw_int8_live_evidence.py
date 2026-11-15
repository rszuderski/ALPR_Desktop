#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB026: read-only live TRAIN calibration + checkpoint class-order audit.

This bridges the already existing managed-report manifest prefilter and the
MZ-DAY CLEAN research readiness validator.  It DOES NOT convert, create a
mobile .alprmodel, approve Android import, or inspect final_test samples.

Automatic binding is allowed only if the managed-report scan completed and
found exactly one metadata-compatible calibration manifest.  The authoritative
live validator is invoked separately; the index is never used as approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

from .mz_raw_int8_package_readiness import assess_research_mz_int8, SOURCES

SCHEMA = 'alpr.f4.mz_day_clean_live_train_evidence_audit.v1'
CHECKPOINT_SHA256 = SOURCES['checkpoint'][1]
EXPECTED_ALPHABET = frozenset('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


class LiveEvidenceAuditError(ValueError):
    """Do not use incomplete search results as proof of model readiness."""


def ordered_label_digest(names: list[str]) -> str:
    """Stable digest of the *ordered* 36-class sequence, not a set hash."""
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_checkpoint_class_order(audit: dict) -> dict:
    """Confirm exactly 0-9 + A-Z from the pinned best.pt; no Android claim."""
    payload = dict(audit.get('class_order') or {})
    names = payload.get('ordered_labels')
    if payload.get('status') != 'CHECKPOINT_VERIFIED' or not isinstance(names, list):
        return {'status': 'NOT_VERIFIED'}
    if (len(names) != 36 or any(type(value) is not str or len(value) != 1 for value in names)
            or set(names) != EXPECTED_ALPHABET):
        raise LiveEvidenceAuditError('CHECKPOINT_ALPHABET_NOT_EXACT_36_ASCII_ALNUM')
    return {
        'status': 'CHECKPOINT_VERIFIED',
        'count': 36,
        'ordered_labels': list(names),
        'order_sha256': ordered_label_digest(names),
        'android_class_order_parity': 'UNVERIFIED',
    }


def _scan_calibration(root: Path, scanner: Callable | None) -> dict:
    if scanner is None:
        from ..gui.z4_mobile_manifest_index import discover_matching_calibrations
        scanner = discover_matching_calibrations
    result = scanner(
        root,
        role='character',
        checkpoint_sha256=CHECKPOINT_SHA256,
        image_size=320,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        mode='exploratory',
    )
    if not isinstance(result, dict):
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_INVALID_RESULT')
    if result.get('authority') != 'PREFILTER_ONLY_VALIDATE_BEFORE_BIND':
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_AUTHORITY_MISMATCH')
    if result.get('search_scope') != 'MANAGED_REPORT_AND_CALIBRATION_ROOTS':
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_SCOPE_MISMATCH')
    matches = result.get('matches')
    if not isinstance(matches, list):
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_INVALID_MATCHES')
    if len(matches) > 32:
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_TOO_MANY_MATCHES')
    # Do not resolve or accept symlink redirects here; the live readiness gate
    # checks every path component separately before reading bytes.
    clean_paths: list[str] = []
    for item in matches:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_INVALID_PATH')
        match_path = Path(item['path'])
        if not match_path.is_absolute():
            raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_PATH_NOT_ABSOLUTE')
        clean_paths.append(str(match_path))
    if len(set(clean_paths)) != len(clean_paths):
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_DUPLICATED_PATH')
    truncated = result.get('truncated') is True
    if result.get('truncated') not in (True, False):
        raise LiveEvidenceAuditError('CALIBRATION_PREFILTER_MISSING_COMPLETENESS')
    return {
        'paths': clean_paths,
        'truncated': truncated,
        'scope': result['search_scope'],
        'directories_scanned': int(result.get('directories_scanned') or 0),
        'files_scanned': int(result.get('files_scanned') or 0),
        'limit_reason': str(result.get('limit_reason') or ''),
    }


def inspect_mz_day_live_evidence(
    repo_root: Path,
    *,
    calibration_manifest: Path | None = None,
    scanner: Callable | None = None,
    assessor: Callable | None = None,
) -> dict[str, Any]:
    """Use a UNIQUE complete prefilter, then live recheck checkpoint and manifest.

    Explicit calibration_manifest bypasses only the *search*, never validation.
    It is useful when more than one legitimate candidate is found.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise LiveEvidenceAuditError(f'REPOSITORY_DIRECTORY_MISSING: {root}')
    if assessor is None:
        assessor = assess_research_mz_int8

    scan = _scan_calibration(root, scanner)
    explicit = calibration_manifest is not None
    chosen: Path | None = None
    if explicit:
        chosen = Path(calibration_manifest)
        method = 'USER_EXPLICIT_LIVE_VALIDATION_REQUIRED'
    elif scan['truncated']:
        method = 'INCOMPLETE_SCAN_NO_AUTOMATIC_BINDING'
    elif len(scan['paths']) > 1:
        method = 'AMBIGUOUS_MULTIPLE_MATCHES_NO_AUTOMATIC_BINDING'
    elif len(scan['paths']) == 1:
        chosen = Path(scan['paths'][0])
        method = 'UNIQUE_PREFILTER_LIVE_VALIDATION_REQUIRED'
    else:
        method = 'NONE_FOUND_IN_MANAGED_SCOPE'

    # The live readiness module recalculates the TFLite static graph, checks the
    # pinned source hashes and calls the authoritative exploratory calibration
    # validator against train/checkpoint/size/conf/IoU.  Even full success here
    # NEVER sets an Android approval flag.
    audit = assessor(root, calibration_manifest=chosen, verify_checkpoint_labels=True)
    if not isinstance(audit, dict):
        raise LiveEvidenceAuditError('LIVE_ASSESSMENT_NOT_OBJECT')
    if audit.get('schema') != 'alpr.f4.mz_raw_int8_packaging_readiness.v1':
        raise LiveEvidenceAuditError('LIVE_ASSESSMENT_SCHEMA_MISMATCH')
    for name, expected in (
        ('model_sha256', SOURCES['clean_raw_tflite'][1]),
        ('checkpoint_sha256', CHECKPOINT_SHA256),
        ('research_artifact_integrity', 'PASS_PINNED_SHA_AND_LIVE_GRAPH_EVIDENCE'),
        ('status', 'BLOCKED_NOT_ANDROID_DEPLOYABLE'),
        ('alprmodel_export_approved', False),
        ('android_import_approved', False),
        ('final_test', 'NOT_VERIFIED'),
        ('conversion_performed', False),
        ('packaging_performed', False),
    ):
        if type(audit.get(name)) is not type(expected) or audit[name] != expected:
            raise LiveEvidenceAuditError(f'LIVE_ASSESSMENT_UNSAFE_FIELD: {name}')
    class_order = validate_checkpoint_class_order(audit)
    blockers = list(audit.get('blockers') or [])
    if not all(isinstance(value, str) for value in blockers):
        raise LiveEvidenceAuditError('LIVE_ASSESSMENT_BLOCKERS_INVALID')
    for required in (
        'ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED',
        'ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT',
        'FINAL_TEST_NOT_VERIFIED',
    ):
        if required not in blockers:
            raise LiveEvidenceAuditError(f'LIVE_ASSESSMENT_MISSING_HARD_BLOCKER: {required}')
    if chosen is None and 'LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED' not in blockers:
        raise LiveEvidenceAuditError('CALIBRATION_BLOCKER_MISSING_WITHOUT_LIVE_VALIDATION')
    cal = dict(audit.get('calibration') or {})
    if chosen is not None:
        if cal.get('status') != 'EXPLORATORY_ONLY_REVALIDATED' or not re.fullmatch('[a-f0-9]{64}', str(cal.get('sha256') or '')):
            raise LiveEvidenceAuditError('CALIBRATION_WAS_NOT_LIVE_REVALIDATED')
        if 'LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED' in blockers:
            raise LiveEvidenceAuditError('STALE_CALIBRATION_BLOCKER_AFTER_VALIDATION')

    if chosen is None and scan['truncated']:
        blockers.append('CALIBRATION_DISCOVERY_INCOMPLETE')
    elif chosen is None and len(scan['paths']) > 1:
        blockers.append('CALIBRATION_MANIFEST_AMBIGUOUS_SELECT_EXPLICITLY')

    return {
        'schema': SCHEMA,
        'status': 'RESEARCH_EVIDENCE_ONLY_NOT_DEPLOYABLE',
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'model_sha256': SOURCES['clean_raw_tflite'][1],
        'manifest_discovery': {
            **scan,
            'selection': method,
            'selected_manifest': str(chosen) if chosen is not None else '',
            'unique_complete_auto_binding': chosen is not None and not explicit,
            'selected_by_user': explicit,
        },
        'class_order': class_order,
        'calibration': cal,
        'remaining_blockers': list(dict.fromkeys(blockers)),
        'android_decoder': 'NOT_APPROVED',
        'android_import_approved': False,
        'alprmodel_export_approved': False,
        'final_test': 'NOT_VERIFIED',
        'conversion_performed': False,
        'packaging_performed': False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Read-only MZ-DAY INT8 class and calibration evidence check')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assess', action='store_true', required=True)
    parser.add_argument('--calibration-manifest', type=Path, default=None,
                        help='Use explicit manifest if scan is ambiguous; must pass live validation')
    args = parser.parse_args(argv)
    try:
        result = inspect_mz_day_live_evidence(args.repo, calibration_manifest=args.calibration_manifest)
    except Exception as exc:
        print(f'LIVE_EVIDENCE_AUDIT_BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
