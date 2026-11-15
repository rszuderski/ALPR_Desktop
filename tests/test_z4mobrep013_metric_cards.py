"""Z4MOBREP013: metric card semantics, no fabricated counts, no IO writes."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        'z4_mobile_report_cards_ux',
        Path(__file__).resolve().parents[1] / 'auto_annotation_tool' / 'gui' / 'z4_mobile_report_cards_ux.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _load()


def report(raw=None, quality=None, latency=None, **extras):
    return SimpleNamespace(raw=raw or {}, quality=quality or {}, latency=latency or {},
                           source_archive_sha256='d' * 64,
                           device={'android_version': '12', 'android_sdk': '31'}, **extras)


def bundle(rows=(), total=None, **kwargs):
    return SimpleNamespace(frame_flow_rows=tuple(rows), frame_flow_total=(len(rows) if total is None else total),
                           trace_total=kwargs.pop('trace_total', 0),
                           report_payload=kwargs.pop('report_payload', {}),
                           validation=kwargs.pop('validation', None), **kwargs)


def test_summary_real_counts():
    r = report({'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7, 'frames_skipped_gate': 233}}},
               latency={'pipeline': {'p95_ms': 2568.2, 'count': 7}})
    flow = m.frame_counts(r)
    assert flow['received'] == 240 and flow['processed'] == 7 and flow['gate'] == 233
    card = m.cards_for_report(r)['frames']
    assert card.value == '7 z 240'
    assert '233 pominięte' in card.subtitle
    assert 'nie są utratą danych' in card.explanation
    assert m.cards_for_report(r)['latency'].subtitle.endswith('n=7')


def test_complete_frame_flow_buckets():
    b = bundle([{'elapsed_ms': '1000', 'frames_received': '100', 'frames_processed': '3', 'frames_skipped_gate': '97'},
                {'elapsed_ms': '2000', 'frames_received': '140', 'frames_processed': '4', 'frames_skipped_gate': '136'}])
    values = m.frame_counts(report(), b)
    assert values['received'] == 240 and values['processed'] == 7 and values['gate'] == 233


def test_no_count_from_incomplete_preview():
    b = bundle([{'frames_received': '100', 'frames_processed': '3', 'frames_skipped_gate': '97'}], total=50)
    values = m.frame_counts(report(), b)
    assert values['received'] is None and values['gate'] is None
    assert m.cards_for_report(report(), b)['frames'].value.startswith('—')


def test_aggregate_precedes_bucket_preview():
    r = report({'summary': {'counters': {'frames_received': 250, 'frames_processed': 8}}})
    b = bundle([{'frames_received': '100', 'frames_processed': '3', 'frames_skipped_gate': '97'}], total=2)
    values = m.frame_counts(r, b)
    assert (values['received'], values['processed'], values['gate']) == (250, 8, None)


def test_report_payload_precedes_compacted_preview():
    r = report({'summary': {'frames_received': 250, 'frames_processed': 10}})
    b = bundle(report_payload={'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7}}})
    assert m.frame_counts(r, b)['received'] == 240


def test_unknown_no_fabricated_gate():
    r = report({'summary': {'processed_frames': 7}})
    cards = m.cards_for_report(r)
    assert cards['frames'].value == '7 klatek'
    assert 'Brak danych' in cards['frames'].subtitle or 'brak danych' in cards['frames'].subtitle
    assert m.frame_counts(r)['gate'] is None


def test_zero_measured():
    r = report({'summary': {'frame_flow': {'frames_received': 0, 'frames_processed': 0, 'frames_skipped_gate': 0}}})
    assert m.cards_for_report(r)['frames'].value == '0 z 0'


@pytest.mark.parametrize('bad', ['NaN', 'inf', -1, True, 'x', 1.3, None])
def test_bad_count_not_reported(bad):
    assert m._count(bad) is None


def test_inconsistent_receipt_not_used_as_ratio():
    r = report({'summary': {'frames_received': 3, 'frames_processed': 7}})
    assert m.frame_counts(r)['received'] is None
    assert m.cards_for_report(r)['frames'].value == '7 klatek'


def test_report_summary_metric_rows():
    r = report({'summary': {'frame_flow': {'frames_received': 240, 'frames_processed': 7, 'frames_skipped_gate': 233}}},
               latency={'pipeline': {'p95_ms': 2568.2, 'count': 7}})
    rows = dict((k, value) for k, value, note in m.summary_measure_rows(r))
    assert rows['Klatki odebrane'] == '240'
    assert rows['Pominięte przez frame_gate'] == '233'
    assert rows['Opóźnienie E2E p95'] == '2568,2 ms'


def test_quality_distinguishes_desktop():
    r = report(quality={'available': False})
    card = m.cards_for_report(r, review={'state': 'in_progress', 'detail': 'Rewizja 60'})['quality']
    assert card.value == 'Android: Brak GT'
    assert card.subtitle == 'Desktop: w toku'
    assert 'nie jest liczbą ukończonych ocen' in card.explanation


def test_desktop_human_quality_not_phone_claim():
    r = report(quality={'available': True, 'quality_source': 'human_review'})
    assert m.cards_for_report(r)['quality'].value == 'Android: Nieustalone'


def test_stored_report_not_automatically_validated():
    card = m.cards_for_report(report())['integrity']
    assert card.value == 'Zapisany raport'
    assert 'Bez ponownej walidacji' in card.subtitle


def test_validated_and_failed_bundle():
    good = bundle(validation=SimpleNamespace(ok=True, checked_hashes=41, warnings=['recovered'], errors=[]))
    bad = bundle(validation=SimpleNamespace(ok=False, checked_hashes=9, warnings=[], errors=['sha mismatch']))
    assert m.cards_for_report(report(), good)['integrity'].value == 'SHA zweryfikowane'
    assert '41 plików' in m.cards_for_report(report(), good)['integrity'].subtitle
    assert m.cards_for_report(report(), bad)['integrity'].value == 'Błąd archiwum'


def test_saved_without_latency_count():
    card = m.cards_for_report(report(latency={'pipeline': {'p95_ms': 2568.2}}))['latency']
    assert card.value == '2568,2 ms'
    assert 'n nieznane' in card.subtitle


def test_missing_p95_not_zero():
    assert m.cards_for_report(report())['latency'].value == '—'


def test_device_explanation_is_not_json():
    assert m.device_details(report()) == 'Android 12 · API 31'
    assert m.device_details(report()._replace() if hasattr(report(), '_replace') else SimpleNamespace(device={})) .startswith('Szczegóły')


def test_bucket_with_total_row_not_double_counted():
    b = bundle([{'frames_received': '10', 'kind': 'bucket'}, {'frames_received': '10', 'kind': 'total'}])
    assert m.frame_counts(report(), b)['received'] is None


def test_no_io_in_metrics(tmp_path):
    before = set(tmp_path.rglob('*'))
    m.cards_for_report(report({'summary': {'processed_frames': 5}}))
    assert set(tmp_path.rglob('*')) == before
