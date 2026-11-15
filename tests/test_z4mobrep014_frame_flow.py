"""Z4MOBREP014 — Android frame_gate alias and reproducible cached flow totals."""
from __future__ import annotations
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


def _load():
    name = 'z4_mobile_report_cards_ux'
    path = Path(__file__).resolve().parents[1] / 'auto_annotation_tool' / 'gui' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = _load()
SHA = 'e' * 64


def report(raw=None, sha=SHA):
    return SimpleNamespace(raw=raw or {}, quality={'available': False}, latency={},
                           source_archive_sha256=sha)


def bundle(rows=(), total=None, *, ok=True, sha=SHA, payload=None):
    rows = tuple(rows)
    return SimpleNamespace(frame_flow_rows=rows, frame_flow_total=len(rows) if total is None else total,
                           trace_total=7, report_payload=payload or {}, source_archive_sha256=sha,
                           validation=SimpleNamespace(ok=ok, checked_hashes=41, warnings=[], errors=[]))


def test_android_column_alias_from_csv_buckets():
    rows = [
        {'elapsed_ms': '1000', 'frames_received': '100', 'frames_processed': '3',
         'frames_skipped_frame_gate': '97', 'frames_skipped_camera_transform': '0'},
        {'elapsed_ms': '2000', 'frames_received': '140', 'frames_processed': '4',
         'frames_skipped_frame_gate': '136', 'frames_skipped_camera_transform': '0'},
    ]
    data = m.frame_counts(report(), bundle(rows))
    assert data == {'received': 240, 'processed': 7, 'gate': 233, 'transform': 0}
    assert m.cards_for_report(report(), bundle(rows))['frames'].subtitle.startswith('233 pominięte')


def test_real_last_row_shape_is_counted_not_used_alone():
    # 21 buckets, including a real-world-shaped final row (1, 1, 1).
    rows = [{'frames_received': '12', 'frames_processed': '0', 'frames_skipped_frame_gate': '12'} for _ in range(19)]
    rows.append({'frames_received': '11', 'frames_processed': '6', 'frames_skipped_frame_gate': '4'})
    rows.append({'frames_received': '1', 'frames_processed': '1', 'frames_skipped_frame_gate': '1'})
    assert len(rows) == 21
    # Deliberately this example does not balance perfectly by a single bucket;
    # the complete counts are still summed separately, not inferred by subtraction.
    result = m.frame_counts(report(), bundle(rows))
    assert result['received'] == 240 and result['processed'] == 7
    assert result['gate'] == 233


def test_legacy_alias_still_works():
    x = {'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7,
                                    'frames_skipped_gate': 233}}}
    assert m.frame_counts(report(x))['gate'] == 233


def test_two_aliases_same_number_allowed():
    x = {'summary': {'frame_flow': {'frames_skipped_frame_gate': 233, 'frames_skipped_gate': 233}}}
    assert m.frame_counts(report(x))['gate'] == 233


def test_two_aliases_inconsistent_not_guessed():
    x = {'summary': {'frame_flow': {'frames_skipped_frame_gate': 233, 'frames_skipped_gate': 244}}}
    assert m.frame_counts(report(x))['gate'] is None


def test_zero_gate_is_not_unknown():
    r = report({'summary': {'frame_flow': {'frames_skipped_frame_gate': 0}}})
    assert m.frame_counts(r)['gate'] == 0
    assert '0 pominięte' in m.cards_for_report(r)['frames'].subtitle


def test_store_snapshot_roundtrip_all_four_counters():
    b = bundle(payload={'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7,
                        'frames_skipped_frame_gate': 233, 'frames_skipped_camera_transform': 0}}})
    snap = m.verified_frame_snapshot(report(), b)
    assert snap['schema'] == 'alpr.desktop.frame_flow_totals.v1'
    assert snap['source_archive_sha256'] == SHA
    assert snap['frames_skipped_frame_gate'] == 233
    stored_raw = json.loads(json.dumps({'_desktop_frame_flow_totals_v1': snap}))
    restored = m.frame_counts(report(stored_raw))
    assert restored == {'received': 240, 'processed': 7, 'gate': 233, 'transform': 0}
    assert m.cards_for_report(report(stored_raw))['integrity'].value == 'Zapisany raport'


def test_snapshot_refuses_wrong_archive_sha():
    b = bundle(payload={'summary': {'frame_flow': {'frames_received': 240, 'frames_skipped_frame_gate': 233}}})
    snap = m.verified_frame_snapshot(report(sha='f' * 64), b)
    assert snap == {}
    valid = m.verified_frame_snapshot(report(), b)
    assert m.frame_counts(report({'_desktop_frame_flow_totals_v1': valid}, sha='f' * 64))['gate'] is None


def test_snapshot_refuses_failed_validation():
    b = bundle(ok=False, payload={'summary': {'frame_flow': {'frames_received': 240, 'frames_skipped_frame_gate': 233}}})
    assert m.verified_frame_snapshot(report(), b) == {}


def test_snapshot_refuses_invalid_totals():
    b = bundle(payload={'summary': {'frame_flow': {'frames_received': 5, 'frames_processed': 7,
                                                 'frames_skipped_frame_gate': 11}}})
    assert m.verified_frame_snapshot(report(), b) == {}


def test_no_totals_when_only_truncated_preview():
    b = bundle([{'frames_received': '20', 'frames_skipped_frame_gate': '18'}], total=21)
    snap = m.verified_frame_snapshot(report(), b)
    assert 'frames_received' not in snap
    assert 'frames_skipped_frame_gate' not in snap
    # trace_total is a valid count of analyzed frames when original frame-flow rows aren't complete
    assert snap.get('frames_processed') == 7


def test_no_source_side_effects(tmp_path):
    path = tmp_path / 'untouched.alprsession'
    path.write_bytes(b'opaque-original')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    b = bundle(payload={'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7,
                                                  'frames_skipped_frame_gate': 233}}})
    m.verified_frame_snapshot(report(), b)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_browser_call_site_has_persistence_integration():
    p = Path(__file__).resolve().parents[1] / 'auto_annotation_tool' / 'gui' / 'z4_mobile_report_browser.py'
    text = p.read_text(encoding='utf-8')
    assert 'verified_frame_snapshot(report, bundle)' in text
    assert '_compact_mobile_report_for_store(bundle.report, bundle=bundle)' in text
