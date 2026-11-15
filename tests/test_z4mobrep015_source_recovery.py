"""Z4MOBREP015: SHA-pinned source restoration and no cross-session data leaks."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui.mobile_report_source_recovery import (
    restore_source, stored_source, relinked_report, crop_entry_for_row,
)
from auto_annotation_tool.ranking import MobileBenchmarkReport, MobilePackageExperimentStore


def report(*, sha='a'*64, report_id='r1', session='session-1', source=''):
    return MobileBenchmarkReport(
        report_id=report_id, package_id='pkg', variant_id='fp32',
        measured_at='2026-10-17T19:00:00Z', source_archive_sha256=sha,
        source_path=source, experiment_index={'experiment_session_id': session, 'source_path': source},
        raw={'desktop_ingest': {'source_path': source, 'source_archive_sha256': sha}},
        quality={'available': False}, latency={'mt': {'count': 8, 'p95_ms': 869.5}},
    )


def bundle(r=None, *, sha='a'*64, ok=True, session='session-1'):
    r = r or report(sha=sha)
    return SimpleNamespace(
        source_archive_sha256=sha, validation=SimpleNamespace(ok=ok), report=r,
        collection_session={'session_id': session}, experiment_session={'experiment_session_id': session},
        trace_rows=({'total_ms': '4'},), entries=(),
    )


def call(tmp_path, r=None, b=None):
    path = tmp_path/'model.alprsession'
    path.write_bytes(b'test-file')
    return restore_source(r or report(), path, reader=lambda _: (b or bundle(),))


def test_matching_archive_has_access_to_detail(tmp_path):
    value = call(tmp_path)
    assert value.ok and len(value.bundle.trace_rows) == 1


def test_no_source_hash_rejects_even_matching_content(tmp_path):
    value = call(tmp_path, r=report(sha=''))
    assert value.status == 'no_fingerprint'


def test_missing_source_never_claims_missing_measurement(tmp_path):
    value = restore_source(report(), tmp_path/'not-here.alprsession', reader=lambda _: ())
    assert value.status == 'missing' and not value.ok


def test_different_file_digest_rejected(tmp_path):
    value = call(tmp_path, b=bundle(sha='b'*64))
    assert value.status == 'mismatch' and 'SHA' in value.detail


def test_invalid_manifest_rejected(tmp_path):
    value = call(tmp_path, b=bundle(ok=False))
    assert value.status == 'mismatch' and 'integralności' in value.detail


def test_report_id_mismatch_rejected(tmp_path):
    value = call(tmp_path, b=bundle(r=report(report_id='other')))
    assert value.status == 'mismatch'


def test_session_id_mismatch_rejected(tmp_path):
    value = call(tmp_path, b=bundle(session='another'))
    assert value.status == 'mismatch' and 'sesji' in value.detail


def test_variant_mismatch_rejected(tmp_path):
    value = call(tmp_path, b=bundle(r=replace(report(), variant_id='int8')))
    assert value.status == 'mismatch'


def test_measurement_date_mismatch_rejected(tmp_path):
    value = call(tmp_path, b=bundle(r=replace(report(), measured_at='2026-10-18T20:00:00Z')))
    assert value.status == 'mismatch'


def test_bad_archive_error_is_displayed_without_value(tmp_path):
    value = restore_source(report(), tmp_path/'bad.alprsession', reader=lambda _: ())
    assert value.status == 'missing'
    path = tmp_path/'bad.alprsession'; path.write_bytes(b'corrupted')
    value = restore_source(report(), path, reader=lambda _: (_ for _ in ()).throw(ValueError('bad archive')))
    assert value.status == 'invalid' and value.bundle is None


def test_no_mutation_of_original_file_on_verified_read(tmp_path):
    path = tmp_path/'source.alprsession';path.write_bytes(b'original research evidence')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    value = restore_source(report(), path, reader=lambda _: (bundle(),))
    assert value.ok and hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_manual_relink_preserves_report_research_measurements(tmp_path):
    r = report(source='C:/old/session.alprsession')
    relinked = relinked_report(r,'C:/new/session.alprsession')
    assert r.source_path == 'C:/old/session.alprsession'
    assert relinked.source_archive_sha256 == r.source_archive_sha256
    assert relinked.quality == r.quality and relinked.latency == r.latency
    assert relinked.raw['desktop_ingest']['source_archive_sha256'] == r.source_archive_sha256
    assert relinked.source_path == 'C:/new/session.alprsession'
    store = MobilePackageExperimentStore(tmp_path/'records')
    store.add_report(relinked)
    restored = MobilePackageExperimentStore(tmp_path/'records').reports[0]
    assert restored.source_path == relinked.source_path
    assert restored.source_archive_sha256 == r.source_archive_sha256
    assert restored.quality == r.quality


def test_fallback_locator_must_be_exact_recorded_path():
    r = report(source='')
    assert stored_source(r) == ''
    r = replace(r, raw={'desktop_ingest': {'source_path': 'D:/backup/one.alprsession'}})
    assert stored_source(r) == 'D:/backup/one.alprsession'


def test_crop_resolver_requires_existing_zip_entry():
    entries = {'samples/crops/a1.jpg', 'samples/crops/a2.png'}
    assert crop_entry_for_row({'capture_id': 'a1'}, entries) == 'samples/crops/a1.jpg'
    assert crop_entry_for_row({'capture_id': 'a2', 'path': 'samples/crops/a2.png'}, entries) == 'samples/crops/a2.png'
    assert crop_entry_for_row({'capture_id': 'a3'}, entries) == ''
    assert crop_entry_for_row({'capture_id': '../../a1'}, entries) == ''
    assert crop_entry_for_row({'capture_id': 'a1', 'crop': 'samples/evidence/a1.jpg'}, entries) == 'samples/crops/a1.jpg'


def test_browser_restoration_is_nonblocking_and_does_not_reimport():
    p = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
    text = p.read_text('utf-8')
    assert 'threading.Thread(target=worker, daemon=True, name="alpr-source-verify").start()' in text
    assert 'self._matching_cached_bundle(report)' in text
    assert 'self.bundle_by_report_id[report_id] = result.bundle' in text
    assert 'self._populate_crops(report, bundle)' in text
    assert 'read_mobile_sample_image(self.current_bundle, entry' in text
