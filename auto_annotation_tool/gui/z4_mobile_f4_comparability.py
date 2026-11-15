#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4 cross-precision comparison of manifest metadata only (NO graph or phone checks).

Compare FP32 and INT8 packages from the SAME selected checkpoints for each
role. Different postprocessing layouts/decoders are flagged for runtime-parity
review, never silently treated as equivalent. Historical missing fields are
inconclusive, not PASS. All inputs are in-memory inventory rows (read-only).
"""
from __future__ import annotations

import math
from pathlib import Path

ROLES = ('MT', 'MZ')
REQUIRED_OUTPUT = ('class_count', 'confidence_threshold', 'iou_threshold')
POSE_FIELDS = ('keypoint_count', 'keypoint_dimensions')
SEMANTIC_FIELDS = ('nms_required', 'nms_in_graph', 'normalized_coordinates',
                   'has_objectness', 'end2end_output')
RUNTIME_FIELDS = ('decoder', 'output_format', 'tensor_layout', 'box_format')
INPUT_FIELDS = ('width', 'height', 'channels', 'layout', 'color')


def _same(lhs, rhs, key: str) -> bool:
    if key in {'confidence_threshold', 'iou_threshold'}:
        try:
            a, b = float(lhs), float(rhs)
            return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-7
        except (ValueError, TypeError, OverflowError):
            return False
    return lhs == rhs and type(lhs) is type(rhs)


def _model_info(manifest: dict) -> dict:
    variants = manifest.get('variants')
    variant = variants[0] if isinstance(variants, list) and len(variants) == 1 and isinstance(variants[0], dict) else {}
    inp = manifest.get('input') if isinstance(manifest.get('input'), dict) else {}
    out = manifest.get('output') if isinstance(manifest.get('output'), dict) else {}
    # A single variant may explicitly override the default specifications.
    effective_in = dict(inp)
    effective_out = dict(out)
    if isinstance(variant.get('input'), dict):
        effective_in.update(variant['input'])
    if isinstance(variant.get('output'), dict):
        effective_out.update(variant['output'])
    return {'input': effective_in, 'output': effective_out,
            'labels': manifest.get('labels'), 'source': manifest.get('source')}


def _compare_role(role: str, fp: dict, iq: dict, pin: str) -> dict:
    """Conservative scientific matching, NOT a runtime or Android approval."""
    blockers = []
    unknown = []
    runtime_review = []
    a, b = _model_info(fp), _model_info(iq)
    for field in ('role', 'task'):
        if not _same(fp.get(field), iq.get(field), field):
            blockers.append(f'{field}_differs')
    source_fp = a['source'] if isinstance(a['source'], dict) else {}
    source_iq = b['source'] if isinstance(b['source'], dict) else {}
    for source in (source_fp, source_iq):
        if str(source.get('checkpoint_sha256') or '').lower() != pin:
            blockers.append('checkpoint_sha_not_pinned')
            break
    if not isinstance(a['labels'], list) or not isinstance(b['labels'], list):
        unknown.append('labels_missing')
    elif a['labels'] != b['labels']:
        blockers.append('ordered_labels_differ')
    for key in INPUT_FIELDS:
        av, bv = a['input'].get(key), b['input'].get(key)
        if av is None or bv is None:
            unknown.append('input_'+key+'_missing')
        elif not _same(av, bv, key):
            blockers.append('input_'+key+'_differs')
    for key in REQUIRED_OUTPUT + (POSE_FIELDS if role == 'MT' else ()) + SEMANTIC_FIELDS:
        av, bv = a['output'].get(key), b['output'].get(key)
        if av is None or bv is None:
            unknown.append('output_'+key+'_missing')
        elif not _same(av, bv, key):
            blockers.append('output_'+key+'_differs')
    for key in RUNTIME_FIELDS:
        av, bv = a['output'].get(key), b['output'].get(key)
        if av is None or bv is None:
            unknown.append('runtime_'+key+'_missing')
        elif not _same(av, bv, key):
            runtime_review.append('runtime_'+key+'_differs')
    if blockers:
        state = 'BLOCKED_METADATA_DIFFERENCE'
    elif unknown:
        state = 'INCONCLUSIVE_METADATA_MISSING'
    elif runtime_review:
        # A decoder/output layout change may require a different Android
        # postprocessor. Manifest equality of the other fields is NOT runtime
        # parity and must not contribute to a clean cross-precision MATCH.
        state = 'RUNTIME_PARITY_REVIEW_REQUIRED'
    else:
        state = 'MANIFEST_SEMANTICS_MATCH_RUNTIME_NOT_VERIFIED'
    return {'role': role, 'state': state,
            'paths': {'fp32': str(fp.get('_path') or ''), 'int8': str(iq.get('_path') or '')},
            'blockers': sorted(set(blockers)), 'unknown': sorted(set(unknown)),
            'decoder_review': sorted(set(runtime_review)),
            'label_count': len(a['labels']) if isinstance(a['labels'], list) else None,
            'checkpoint_sha256': pin, 'graph_checked': False, 'phone_checked': False}


def compare_f4_cross_precision(rows, inventory: dict, *, max_combinations_per_role: int = 64) -> dict:
    """Compare candidates from exact selected SHA-pinned FP32 and INT8 inventory.

    Never select one of multiple candidate pairs automatically. Never touch
    disk: the caller's `list_managed_packages` is the sole source of data.
    """
    result = {'schema': 'alpr.desktop.f4_cross_precision_manifest_audit.v1',
              'status': 'INCOMPLETE', 'roles': {}, 'graph_checked': False,
              'android_approved': False, 'f4_experiment_performed': False,
              'authority': 'MANIFEST_ONLY_REQUIRES_RUNTIME_AND_PHONE_COMPARISON'}
    if not isinstance(inventory, dict) or inventory.get('scan_truncated'):
        result['status'] = 'BLOCKED_INCOMPLETE_SCAN'
        return result
    pairs = inventory.get('pairs') or {}
    if any((pairs.get(p) or {}).get('state') != 'MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION'
           for p in ('fp32','int8')):
        result['status'] = 'BLOCKED_NO_BOTH_PRECISIONS'
        return result
    pins = inventory.get('selected_checkpoint_sha256') or {}
    indexed = {}
    for row in rows or ():
        path = str(getattr(row, 'path', '') or '')
        if path and getattr(row, 'kind', '') == 'MODEL' and isinstance(getattr(row, 'manifest', None), dict):
            indexed[path] = row.manifest
    all_present = True
    any_ambiguous = False
    any_blocked = False
    any_incomplete = False
    any_runtime_review = False
    for role in ROLES:
        candidates_fp = (pairs.get('fp32',{}).get('candidates') or {}).get(role) or []
        candidates_iq = (pairs.get('int8',{}).get('candidates') or {}).get(role) or []
        combinations = len(candidates_fp) * len(candidates_iq)
        report = {'total_combinations': combinations, 'checked_pairs': [],
                  'role_state': 'BLOCKED_NO_ROLE_PAIR'}
        result['roles'][role] = report
        if combinations == 0:
            all_present = False
            continue
        if combinations > max_combinations_per_role:
            report['role_state'] = 'INCOMPLETE_TOO_MANY_CANDIDATES'
            any_incomplete = True
            continue
        pin = str(pins.get(role) or '').lower()
        for item_fp in candidates_fp:
            for item_iq in candidates_iq:
                path_fp = str(item_fp.get('path') or '')
                path_iq = str(item_iq.get('path') or '')
                mf = indexed.get(path_fp)
                mi = indexed.get(path_iq)
                if mf is None or mi is None:
                    report['checked_pairs'].append({
                        'role': role, 'state': 'INCONCLUSIVE_MANIFEST_UNAVAILABLE',
                        'paths': {'fp32': path_fp, 'int8': path_iq},
                        'blockers': [], 'unknown': ['manifest_unavailable'],
                        'decoder_review': [], 'graph_checked': False, 'phone_checked': False})
                    continue
                report['checked_pairs'].append(_compare_role(role,
                    dict(mf, _path=path_fp), dict(mi, _path=path_iq), pin))
        matching = [p for p in report['checked_pairs'] if p['state'] == 'MANIFEST_SEMANTICS_MATCH_RUNTIME_NOT_VERIFIED']
        parity_review = [p for p in report['checked_pairs'] if p['state'] == 'RUNTIME_PARITY_REVIEW_REQUIRED']
        incomplete = any(p['state'].startswith('INCONCLUSIVE') for p in report['checked_pairs'])
        # A single clean-looking pair does not establish uniqueness when there
        # are other candidate pairs with different decoder/output contracts.
        possible_pairs = len(matching) + len(parity_review)
        if possible_pairs > 1:
            report['role_state'] = 'AMBIGUOUS_SELECT_EXPLICITLY'
            any_ambiguous = True
        elif incomplete:
            report['role_state'] = 'INCONCLUSIVE_METADATA'
            any_incomplete = True
        elif len(matching) == 1:
            report['role_state'] = 'MANIFEST_MATCH_RUNTIME_NOT_VERIFIED'
        elif len(parity_review) == 1:
            report['role_state'] = 'RUNTIME_PARITY_REVIEW_REQUIRED'
            any_runtime_review = True
        else:
            report['role_state'] = 'BLOCKED_METADATA_MISMATCH'
            any_blocked = True
    if not all_present:
        result['status'] = 'BLOCKED_NO_BOTH_PRECISIONS'
    elif any_ambiguous:
        result['status'] = 'AMBIGUOUS_SELECT_EXPLICITLY'
    elif any_incomplete:
        result['status'] = 'INCONCLUSIVE_METADATA'
    elif any_blocked:
        result['status'] = 'BLOCKED_METADATA_MISMATCH'
    elif any_runtime_review:
        result['status'] = 'RUNTIME_PARITY_REVIEW_REQUIRED'
    else:
        result['status'] = 'MANIFEST_CROSS_PRECISION_MATCH_RUNTIME_NOT_VERIFIED'
    return result


def describe_f4_cross_precision(report: dict) -> str:
    lines = [
        'Czy gotowe paczki FP32 i INT8 pochodzą z tych samych treningów?',
        'Ta kontrola porównuje wyłącznie informacje w paczkach. Nie sprawdza jeszcze działania modeli ani dokładności odczytu na telefonie.',
        f'Kod wyniku (dla diagnostyki): {report.get("status") or "INCONCLUSIVE"}.',
    ]
    if report.get('status') == 'RUNTIME_PARITY_REVIEW_REQUIRED':
        lines.append('ROŻNE DEKODERY / FORMATY WYJŚĆ: wymagany jawny test parytetu w ALPR Mobile. '
                     'Zgodność manifestów nie została potwierdzona jako porównywalność wykonania.')
    for role in ROLES:
        item = (report.get('roles') or {}).get(role) or {}
        if not item:
            continue
        lines.append(f'{role}: {item.get("role_state")}; kombinacji: {item.get("total_combinations", 0)}')
        for pair in (item.get('checked_pairs') or [])[:6]:
            paths = pair.get('paths') or {}
            lines.append('  '+str(pair.get('state'))+' | '+
                         Path(str(paths.get('fp32') or '-')).name+' ↔ '+
                         Path(str(paths.get('int8') or '-')).name)
            if pair.get('blockers'):
                lines.append('    Niezgodności: '+', '.join(pair['blockers']))
            if pair.get('unknown'):
                lines.append('    Niepotwierdzone: '+', '.join(pair['unknown']))
            if pair.get('decoder_review'):
                lines.append('    Wymaga testu dekodowania: '+', '.join(pair['decoder_review']))
        if len(item.get('checked_pairs') or []) > 6:
            lines.append('    Pozostałe kombinacje pominięto w podglądzie; wynik analizy obejmuje je wszystkie.')
    lines.append('Następny krok: sprawdź pełną zawartość wybranych paczek, a potem działanie w ALPR Mobile.')
    return '\n'.join(lines)
