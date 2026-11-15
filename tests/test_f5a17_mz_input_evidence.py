"""F5A/17: original Android delivery; all writes use private temporary copies."""
from pathlib import Path
from _private_test_data import private_test_path
import copy
import hashlib
import json
import shutil
import zipfile

import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_mz_input_evidence as mz

PRODUCER = private_test_path('f5a16_mz_scope_0a92f7c3')
GOOD = ['mz-backend-threw', 'mz-byte-budget', 'mz-float32-nchw', 'mz-float32-nhwc',
        'mz-int8-nchw', 'mz-int8-nhwc', 'mz-interrupted', 'mz-legacy-no-calls',
        'mz-memory-missing', 'mz-negative-source', 'mz-not-run', 'mz-off',
        'mz-oversize-missing', 'mz-payload-limit', 'mz-slow-writer', 'mz-uint8-nchw',
        'mz-uint8-nhwc', 'mz-write-missing', 'mz-written-r0', 'mz-written-r1', 'mz-written-r2-two-roi']
BAD = ['negative-duplicate-index', 'negative-duplicate-zip', 'negative-payload-missing',
       'negative-shape', 'negative-source-stamp', 'negative-tensor-sha', 'negative-unknown-invocation']


def original(name):
    path = PRODUCER / 'fixtures' / (name + '.alprsession')
    assert path.is_file(), 'BLOCKED: original Android fixture missing'
    inventory = json.loads((PRODUCER / 'fixture_inventory.json').read_text(encoding='utf-8'))
    expected = inventory[path.name]
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    return path


def private_session(tmp_path, name):
    root = tmp_path / name
    root.mkdir(parents=True)
    source = root / 'source.alprsession'
    shutil.copyfile(original(name), source)
    return MobileReviewSession.open(source, sidecar_path=root / 'review.json')


@pytest.mark.parametrize('name', GOOD)
def test_original_positive_native_import(name, request):
    path = original(name)
    bundle = read_mobile_report_bundle(path)
    assert bundle.validation.ok, bundle.validation.errors
    result = bundle.mz_input_evidence
    with zipfile.ZipFile(path) as archive:
        session = json.loads(archive.read('session.json'))
        if name in {'mz-off', 'mz-legacy-no-calls'}:
            assert result['status'] == 'ABSENT' and not result['available']
        else:
            assert result['available']
            assert result['status'] == ('VALID' if session['mz_input_evidence']['complete'] else 'PARTIAL')
            assert result['summary'] == session['mz_input_evidence']
            attempts = {r['attempt_id']: r for r in map(json.loads, archive.read('samples/attempts.jsonl').splitlines())}
            for row in result['rows']:
                for k in mz.STAMP[:-1]:
                    assert type(row['source_stamp'][k]) is int
                    assert row['source_stamp'][k] == attempts[row['attempt_id']][k]
                if row['tensor']['write_status'] == 'WRITTEN':
                    tensor = row['tensor']
                    data = archive.read(tensor['entry'])
                    assert len(data) == tensor['byte_size']
                    assert hashlib.sha256(data).hexdigest() == tensor['sha256']
            assert not any('binarne artefakty modeli' in w for w in bundle.validation.warnings)
    assert bundle.to_dict()['mz_input_evidence'] == result
    request.node.user_properties.append(('fixture', dict(name=name, status=result['status'],
        sha256=bundle.source_archive_sha256, rows=len(result.get('rows', [])),
        dependencies=len(result.get('dependencies', {})))))


@pytest.mark.parametrize('name', BAD)
def test_original_negative_native_import(name, request):
    bundle = read_mobile_report_bundle(original(name))
    assert not bundle.validation.ok
    assert bundle.mz_input_evidence['status'] == 'INVALID'
    request.node.user_properties.append(('rejection', dict(name=name, reasons=bundle.validation.errors)))


def mutate(tmp_path, change, name='mz-written-r0'):
    with zipfile.ZipFile(original(name)) as archive:
        payloads = {n: archive.read(n) for n in archive.namelist()}
    rows = list(map(json.loads, payloads[mz.INDEX].splitlines()))
    change(rows, payloads)
    payloads[mz.INDEX] = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()
    manifest = json.loads(payloads['manifest.json'])
    manifest['entry_sha256'] = {n: hashlib.sha256(data).hexdigest() for n, data in payloads.items() if n != 'manifest.json'}
    payloads['manifest.json'] = json.dumps(manifest).encode()
    path = tmp_path / 'mutation.alprsession'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for n, data in payloads.items():
            archive.writestr(n, data)
    return path


@pytest.mark.parametrize('field,value', [('source_timestamp_nanos', 9007199254740994.0),
    ('source_sequence', True), ('scene_generation', 2**63), ('source_timestamp_nanos', 9007199254740995)])
def test_stamp_is_exact_int64(tmp_path, field, value):
    path = mutate(tmp_path, lambda rows, _: rows[0]['source_stamp'].__setitem__(field, value))
    assert read_mobile_report_bundle(path).mz_input_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('field,value', [('dtype', 'FLOAT64'), ('layout', 'HWC'), ('byte_order', 'NATIVE'),
    ('original_position', -1), ('range_start', 13), ('shape', [1, 320, 320, True]),
    ('normalization_scale', float('inf')), ('quantization_zero_point', 0.5)])
def test_tensor_contract_rejects_contradictions(tmp_path, field, value):
    path = mutate(tmp_path, lambda rows, _: rows[0]['tensor'].__setitem__(field, value))
    assert read_mobile_report_bundle(path).mz_input_evidence['status'] == 'INVALID'


def test_no_reconstructed_transform_or_context_as_tensor(tmp_path):
    path = mutate(tmp_path, lambda rows, _: rows[0]['homography'].__setitem__('H', [1] * 9))
    assert read_mobile_report_bundle(path).mz_input_evidence['status'] == 'INVALID'
    path = mutate(tmp_path, lambda rows, _: rows[0]['source_image'].__setitem__('exact_tensor_input', True))
    assert read_mobile_report_bundle(path).mz_input_evidence['status'] == 'INVALID'


def test_recorded_position_and_two_roi_identity():
    result = read_mobile_report_bundle(original('mz-written-r0')).mz_input_evidence
    assert result['rows'][0]['tensor']['original_position'] == 13
    assert result['rows'][0]['tensor']['range_start'] == 0
    assert result['rows'][0]['source_stamp']['source_timestamp_nanos'] == 9007199254740994
    result = read_mobile_report_bundle(original('mz-written-r2-two-roi')).mz_input_evidence
    assert len({r['mt_invocation_id'] for r in result['rows']}) == 2
    assert len({(r['attempt_id'], r['call_index']) for r in result['rows']}) == 2


def test_read_only_diagnostic_purpose_and_no_operator_decision(tmp_path):
    from auto_annotation_tool.ranking.mobile_experiment_identity import from_bundle, from_review_provenance
    s = private_session(tmp_path, 'mz-memory-missing')
    before = copy.deepcopy(s.review.to_dict())
    summary = s.statistics()
    text = mz.diagnostic_text(s.mz_input_evidence)
    assert 'PARTIAL' in text and '.bin' in text and 'JPEG' in text and 'RECTIFIER_TRANSFORM_NOT_EXPOSED' in text
    assert 'SYNTHETIC_DIAGNOSTIC' in text
    assert s.review.to_dict() == before
    assert not s.review.scene_annotations and not s.review.invocation_annotations
    assert summary['mz_input_evidence']['status'] == 'PARTIAL'
    assert from_bundle(s.bundle)['experiment'] == 'SYNTHETIC_DIAGNOSTIC'
    assert from_review_provenance(s.review.provenance, verified=True)['experiment'] == 'SYNTHETIC_DIAGNOSTIC'


@pytest.mark.parametrize('question', ['F4', 'F5A'])
def test_diagnostic_source_excluded_from_official_analysis(tmp_path, question):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    s = private_session(tmp_path, 'mz-written-r0')
    inp = make_input(s.bundle.report, source_path=s.path, results_root=tmp_path / '03_wyniki', review_root=tmp_path / 'reviews')
    result = analyze([inp], question=question).sessions[0]
    assert result.source_status == 'VERIFIED', result.source_reason
    assert result.metadata['experiment'] == 'SYNTHETIC_DIAGNOSTIC'
    assert result.metadata['registry_verified'] is False
    assert all(m.status == 'NOT_COMPARABLE' and m.value is None for m in result.metrics.values())


def test_diagnostic_export_cannot_join_official_cohort(tmp_path):
    from dataclasses import replace
    from auto_annotation_tool.gui.mobile_import_comparison import record_from_import
    from auto_annotation_tool.gui.mobile_comparison_core import validate_selection
    s = private_session(tmp_path, 'mz-written-r0')
    diagnostic = record_from_import(s.bundle.report, s.bundle, include_review=False)[0]
    assert diagnostic.measurement_purpose == 'SYNTHETIC_DIAGNOSTIC'
    official = replace(diagnostic, session_id='official', measurement_purpose='')
    with pytest.raises(ValueError, match='osobnej kohorty'):
        validate_selection([diagnostic, official])
    assert any('SYNTHETIC_DIAGNOSTIC' in w for w in validate_selection([diagnostic]))


def test_export_cannot_silently_drop_declared_evidence(tmp_path):
    s = private_session(tmp_path, 'mz-written-r0')
    with pytest.raises(ValueError, match='descriptor missing'):
        mz.load_export(tmp_path / 'empty', {'files': []}, s.bundle.collection_session)


def test_diagnostic_missing_report_date_is_not_import_time(monkeypatch):
    from auto_annotation_tool.ranking import mobile_package_experiments as package
    monkeypatch.setattr(package, '_utc_now_iso', lambda: '2030-01-01T00:00:00Z')
    first = read_mobile_report_bundle(original('mz-written-r0'))
    monkeypatch.setattr(package, '_utc_now_iso', lambda: '2031-01-01T00:00:00Z')
    second = read_mobile_report_bundle(original('mz-written-r0'))
    assert first.report.measured_at == second.report.measured_at == ''
    assert first.mz_input_evidence == second.mz_input_evidence


def test_corrupted_zip_payload_crc_is_invalid(tmp_path):
    path = mutate(tmp_path, lambda rows, payloads: None)
    with zipfile.ZipFile(path) as archive:
        payloads = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
        for n, data in payloads.items():
            archive.writestr(n, data)
    with zipfile.ZipFile(path) as archive:
        info = next(i for i in archive.infolist() if i.filename.endswith('.bin'))
    offset = info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra)
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)
    result = read_mobile_report_bundle(path)
    assert not result.validation.ok and result.mz_input_evidence['status'] == 'INVALID'
    assert any('CRC' in reason for reason in result.validation.errors)


@pytest.mark.parametrize('name,status', [('mz-written-r0', 'VALID'), ('mz-memory-missing', 'PARTIAL'), ('mz-off', 'ABSENT')])
def test_T19_full_native_roundtrip(tmp_path, monkeypatch, name, status, request):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, InvalidExport
    s = private_session(tmp_path, name)
    s.set_review_mode('assisted')
    before = json.loads(json.dumps(s.review.to_dict()))
    restarted = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert restarted.review.to_dict() == before
    assert restarted.mz_input_evidence == s.mz_input_evidence
    monkeypatch.setattr(native, 'source_session', lambda _: restarted)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, s = next(generator)
    try:
        out = tmp_path / '03_wyniki'
        out.mkdir()
        errors = []
        monkeypatch.setattr(native.module.messagebox, 'showerror', lambda *a, **kw: errors.append(a))
        monkeypatch.setattr(native.module.filedialog, 'askdirectory', lambda **kw: str(out))
        monkeypatch.setattr(native.module.messagebox, 'askyesno', lambda *a, **kw: True)
        h.export()
        native.pump(root, lambda: not h._saving, seconds=90)
        assert not errors, errors
        manifests = list(out.rglob('export_manifest.json'))
        assert len(manifests) == 1, h.status_var.get()
        folder = manifests[0].parent
        loaded = load_session_export(folder)
        assert loaded.mz_input_evidence['status'] == status
        assert loaded.mz_input_evidence.get('rows', []) == s.mz_input_evidence.get('rows', [])
        assert loaded.mz_input_evidence.get('dependencies', {}) == s.mz_input_evidence.get('dependencies', {})
        reopened = MobileReviewSession.open(s.path, sidecar_path=folder / 'review.json')
        assert reopened.review.to_dict() == before
        assert reopened.mz_input_evidence == s.mz_input_evidence
        assert not reopened.review.scene_annotations and not reopened.review.invocation_annotations
        assert hashlib.sha256(s.path.read_bytes()).hexdigest() == s.bundle.source_archive_sha256
        assert h.conditions_view.cget('state') == 'disabled'
        assert 'MZ input evidence: ' + status in h.conditions_view.get('1.0', 'end')
        manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
        for row in manifest['files']:
            assert hashlib.sha256((folder / row['path']).read_bytes()).hexdigest() == row['sha256']
        request.node.user_properties.append(('roundtrip', dict(fixture=name, status=status,
            source_sha256=s.bundle.source_archive_sha256, manifest_sha256=hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
            dependencies=len(manifest['files']), mz_dependencies=len(s.mz_input_evidence.get('dependencies', {})),
            full_native_export=True, source_int64=[r['source_stamp']['source_timestamp_nanos'] for r in loaded.mz_input_evidence.get('rows', [])])))
        if status != 'ABSENT':
            assert loaded.measurement_purpose == 'SYNTHETIC_DIAGNOSTIC'
            assert loaded.label.startswith('SYNTHETIC_DIAGNOSTIC')
            dep = next(iter(s.mz_input_evidence['tensor_payloads']))
            payload = folder / mz.NAMESPACE / dep
            data = payload.read_bytes()
            payload.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with pytest.raises(InvalidExport):
                load_session_export(folder)
            payload.write_bytes(data)
            # Removing a required dependency even from the OUTER manifest must fail.
            original_manifest = manifests[0].read_bytes()
            manifest['files'] = [r for r in manifest['files'] if r['path'] != mz.NAMESPACE + '/' + dep]
            manifests[0].write_text(json.dumps(manifest), encoding='utf-8')
            with pytest.raises(InvalidExport):
                load_session_export(folder)
            manifests[0].write_bytes(original_manifest)
    finally:
        try:
            next(generator)
        except StopIteration:
            pass
