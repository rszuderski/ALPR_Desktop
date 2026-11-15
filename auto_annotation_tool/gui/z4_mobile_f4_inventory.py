#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only F4 FP32/INT8 *candidate* inventory pinned to selected checkpoint SHAs.

A matching manifest is never proof of a valid TFLite graph, Android import,
model accuracy, or successful F4/final_test. Never convert or approve here.
"""
from __future__ import annotations

import re
from pathlib import Path

SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ROLE_SPEC = {"MT": ("plate", "pose"), "MZ": ("character", "detect")}
PRECISIONS = ("fp32", "int8")


def _manifest_candidate(row, role: str, precision: str, checkpoint_sha: str) -> tuple[bool, str, dict]:
    """Assess bounded manifest metadata, NEVER the deployed network graph."""
    role_name, task = ROLE_SPEC[role]
    if getattr(row, 'kind', None) != 'MODEL' or getattr(row, 'role', None) != role:
        return False, "wrong_kind_or_role", {}
    if getattr(row, 'schema', None) != 'alpr.model.v1':
        return False, "wrong_schema", {}
    if getattr(row, 'error', '') or str(getattr(row, 'state', '')).upper() not in {
        'DISCOVERED_NOT_VERIFIED', 'REGISTERED_SHA_MATCH_NOT_VERIFIED',
    }:
        return False, "artifact_state_not_trusted", {}
    manifest = getattr(row, 'manifest', None)
    if not isinstance(manifest, dict) or manifest.get('role') != role_name or manifest.get('task') != task:
        return False, "role_task_conflict", {}
    source = manifest.get('source')
    observed = str(source.get('checkpoint_sha256') or '').strip().lower() if isinstance(source, dict) else ''
    if not SHA_RE.fullmatch(observed) or not checkpoint_sha or observed != checkpoint_sha:
        return False, "checkpoint_sha_mismatch", {}
    variants = manifest.get('variants')
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        return False, "not_single_variant", {}
    variant = variants[0]
    if variant.get('runtime') != 'tflite' or str(variant.get('precision') or '').lower() != precision:
        return False, "wrong_runtime_or_precision", {}
    input_spec = manifest.get('input')
    if not isinstance(input_spec, dict) or (
        input_spec.get('data_type') != ('INT8' if precision == 'int8' else 'FLOAT32')
    ):
        return False, "input_dtype_mismatch", {}
    try:
        width = int(input_spec.get('width'))
        height = int(input_spec.get('height'))
    except (TypeError, ValueError):
        return False, "bad_image_size", {}
    if width < 128 or width > 2048 or width != height:
        return False, "bad_image_size", {}
    output = manifest.get('output')
    labels = manifest.get('labels')
    if not isinstance(output, dict) or not isinstance(labels, list) or not labels:
        return False, "missing_output_or_labels", {}
    try:
        if int(output.get('class_count') or 0) != len(labels):
            return False, "class_count_mismatch", {}
        if role == 'MT' and (len(labels) != 1 or int(output.get('keypoint_count') or 0) < 4):
            return False, "pose_contract_mismatch", {}
    except (ValueError, TypeError):
        return False, "invalid_output", {}
    calibration = 'NOT_APPLICABLE_FP32'
    if precision == 'int8':
        if not isinstance(variant.get('graph_precision'), dict) or not variant['graph_precision']:
            return False, "missing_static_graph_evidence", {}
        evidence = manifest.get('calibration_evidence')
        if not isinstance(evidence, dict) or not evidence.get('file'):
            return False, "missing_calibration_evidence", {}
        mode = str(evidence.get('mode') or '').lower()
        if mode == 'exploratory':
            if (evidence.get('status') != 'EXPLORATORY_ONLY' or
                    (manifest.get('research_calibration') or {}).get('final_test_status') != 'NOT_VERIFIED'):
                return False, "inconsistent_exploratory_evidence", {}
            calibration = 'EXPLORATORY_ONLY'
        elif mode == 'strict' and evidence.get('status') == 'READY':
            calibration = 'READY_IN_MANIFEST_NOT_GRAPH_VERIFIED'
        else:
            return False, "invalid_calibration_evidence", {}
    return True, '', {
        'path': str(getattr(row, 'path', '')),
        'name': str(getattr(row, 'name', '')),
        'model_id': str(manifest.get('model_id') or ''),
        'checkpoint_sha256': observed,
        'image_size': width,
        'calibration': calibration,
        'authority': 'MANIFEST_PREFILTER_ONLY',
    }


def build_f4_inventory(rows, *, checkpoint_sha_by_role: dict, scan_truncated: bool = False) -> dict:
    """Report possible matching MT/MZ pairs for two *selected* best.pt files.

    Never claim availability/approval from ZIP names or from raw research TFLite.
    A truncated scan makes negative findings inconclusive.
    """
    pins = {}
    for role in ROLE_SPEC:
        value = str((checkpoint_sha_by_role or {}).get(role) or '').strip().lower()
        pins[role] = value if SHA_RE.fullmatch(value) else ''
    rows = list(rows or [])
    research_mz = sorted({str(getattr(row, 'path', '')) for row in rows if
                          getattr(row, 'kind', '') == 'RESEARCH_TFLITE' and getattr(row, 'role', '') == 'MZ'
                          and getattr(row, 'state', '') == 'RESEARCH_ONLY_NOT_PACKAGED'})
    pairs = {}
    for precision in PRECISIONS:
        chosen = {}
        excluded = {}
        for role in ROLE_SPEC:
            candidates = []
            rejections = {}
            for row in rows:
                if getattr(row, 'kind', '') != 'MODEL' or getattr(row, 'role', '') != role:
                    continue
                okay, reason, payload = _manifest_candidate(row, role, precision, pins[role])
                if okay:
                    candidates.append(payload)
                else:
                    rejections[reason] = rejections.get(reason, 0) + 1
            chosen[role] = sorted(candidates, key=lambda item: item['path'].casefold())
            excluded[role] = rejections
        missing = [role for role in ROLE_SPEC if not chosen[role]]
        if not all(pins.values()):
            state = 'BLOCKED_NO_SELECTED_CHECKPOINT_SHA'
        elif scan_truncated:
            state = 'SCAN_INCOMPLETE_NO_READINESS_CLAIM'
        elif missing:
            state = 'BLOCKED_NO_MATCHING_PACKAGE'
        else:
            state = 'MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION'
        pairs[precision] = {
            'state': state,
            'candidates': chosen,
            'rejection_counts': excluded,
            'missing_roles': missing,
            'pair_count_manifest_only': len(chosen['MT']) * len(chosen['MZ']),
        }
    return {
        'schema': 'alpr.desktop.f4_candidate_inventory.v1',
        'selected_checkpoint_sha256': pins,
        'scan_truncated': bool(scan_truncated),
        'pairs': pairs,
        'research_mz_tflite': research_mz,
        'desktop_full_graph_verified': False,
        'android_import_approved': False,
        'f4_experiment_performed': False,
    }


def describe_f4_inventory(report: dict) -> str:
    """Human-readable, honest Polish readiness summary, safe for a GUI text box."""
    lines = ['Które modele mobilne już mamy dla wybranych treningów?',
             'Co sprawdzamy: opisy paczek i powiązania z treningami. Sama lista nie oznacza jeszcze poprawnego działania na telefonie.', '']
    pins = report.get('selected_checkpoint_sha256') or {}
    for role in ROLE_SPEC:
        pin = str(pins.get(role) or '')
        lines.append(f'{role} checkpoint SHA-256: {pin if pin else "BRAK — nie można potwierdzić dopasowania"}')
    if report.get('scan_truncated'):
        lines.extend(['', 'UWAGA: skan katalogu był niekompletny; brak paczki nie jest potwierdzony.'])
    for precision in PRECISIONS:
        view = (report.get('pairs') or {}).get(precision) or {}
        lines.extend(['', f'--- {precision.upper()} ---'])
        for role in ROLE_SPEC:
            entries = (view.get('candidates') or {}).get(role) or []
            lines.append(f'{role}: {len(entries)} kandydatów o zgodnym checkpoint SHA (manifest).')
            for item in entries[:8]:
                lines.append(f'  • {Path(item["path"]).name} | imgsz {item["image_size"]} | kalibracja: {item["calibration"]}')
            if len(entries) > 8:
                lines.append(f'  ...i {len(entries) - 8} innych.')
        state = view.get('state')
        if state == 'MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION':
            lines.append('Znaleziono pasujące paczki dla obu modeli. Przed użyciem aplikacja sprawdzi ich zawartość i pochodzenie.')
        elif state == 'SCAN_INCOMPLETE_NO_READINESS_CLAIM':
            lines.append('Nie udało się sprawdzić wszystkich paczek. Ponów przegląd, zanim wybierzesz modele.')
        elif state == 'BLOCKED_NO_SELECTED_CHECKPOINT_SHA':
            lines.append('Nie można potwierdzić, że paczki pochodzą z wybranych treningów. Sprawdź wybór modeli w Centrum.')
        else:
            missing = ', '.join(view.get('missing_roles') or [])
            lines.append(f'Brakuje gotowego modelu pasującego do wybranego treningu: {missing or "MT/MZ"}. Można go przygotować w oknie eksportu.')
    research = report.get('research_mz_tflite') or []
    if research:
        lines.extend(['', f'Wykryto {len(research)} badawczy plik MZ TFLite (nie .alprmodel).',
                      'Nie zaliczono go do wariantu INT8; nie ma zatwierdzenia eksportu/importu.'])
    lines.extend(['', 'Co dalej: przed pomiarami na telefonie sprawdzimy zawartość każdej wybranej paczki, a potem uruchomimy próbę w ALPR Mobile.',
                  'Zestawienie nie modyfikuje modeli ani historii eksperymentów.'])
    return '\n'.join(lines)
