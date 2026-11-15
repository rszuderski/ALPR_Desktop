"""F5A/19: DEVICE purpose FORMAT tests, never Galaxy A12 measurements.

Original F16/F18 files are read-only. Positive DEVICE archives are explicitly
labelled SYNTHETIC_FORMAT_FIXTURE and DEVICE_RUN=NOT_RUN in their provenance.
"""
from pathlib import Path
from _private_test_data import private_test_path
from dataclasses import replace
import copy
import hashlib
import json
import shutil
import zipfile

import pytest

from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_mz_input_evidence as mz

PROOF = private_test_path('F5A_19_DESKTOP_DEVICE_DOWODY_20261106')
F16 = private_test_path('f5a16_mz_scope_0a92f7c3')
F18 = private_test_path('f5a18_mz_purpose_c7e14b90')
FORMATS = json.loads((PROOF / 'format_fixture_inventory.json').read_text(encoding='utf-8'))
ORIGINALS = {r['name']: r for r in json.loads((F18 / 'fixture_inventory.json').read_text(encoding='utf-8'))}
EXPECTED_F18 = {r['name']: ('INVALID' if r['expected'] == 'REJECT' else r['expected'])
    for r in json.loads((F18 / 'desktop_read/result.json').read_text(encoding='utf-8'))['rows'] if r['population'] == 'new_f18'}
DEVICE = 'DEVICE_DIAGNOSTIC'
SYNTHETIC = 'SYNTHETIC_DIAGNOSTIC'


def checked(path, expected):
    assert path.is_file(), 'BLOCKED: original fixture unavailable'
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    return path


def format_path(name='mz-written-r0'):
    name = 'format-device-' + name + '.alprsession'
    return checked(PROOF / 'format_fixtures' / name, FORMATS[name])


def original_path(name):
    inventory = json.loads((F16 / 'fixture_inventory.json').read_text(encoding='utf-8'))
    return checked(F16 / 'fixtures' / (name + '.alprsession'), inventory[name + '.alprsession'])


def session(tmp_path, source):
    target = tmp_path / source.stem / 'source.alprsession'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return MobileReviewSession.open(target, sidecar_path=target.parent / 'review.json')


@pytest.mark.parametrize('name', sorted(ORIGINALS))
def test_original_f18_delivery_native_reader(name, request):
    path = checked(F18 / 'fixtures' / name, ORIGINALS[name])
    bundle = read_mobile_report_bundle(path)
    expected = EXPECTED_F18[name]
    if name == 'negative-purpose-device_diagnostic.alprsession':
        # M12 rejected this synthetic COPY only because DEVICE was unsupported.
        # Its DEVICE series/flags are already correct for the newly approved enum.
        expected = 'VALID'
        with zipfile.ZipFile(path) as archive:
            report = json.loads(archive.read('report.json'))
            assert report['synthetic'] is True  # never an authentic device run
            assert report['research_collection']['series_id'] == 'DEVICE-DIAGNOSTIC-MZ-CONTRACT-TEST'
        assert bundle.mz_input_evidence['measurement_purpose'] == DEVICE
    assert bundle.mz_input_evidence['status'] == expected, bundle.validation.errors
    assert bundle.validation.ok == (expected != 'INVALID')
    request.node.user_properties.append(('original_f18', dict(name=name, status=expected,
        f18_previous_status=EXPECTED_F18[name], sha256=ORIGINALS[name]['sha256'], DEVICE_RUN='NOT_RUN')))


@pytest.mark.parametrize('name', sorted(FORMATS))
def test_device_format_fixtures_native_reader(name, request):
    value = FORMATS[name]
    path = checked(PROOF / 'format_fixtures' / name, value)
    bundle = read_mobile_report_bundle(path)
    assert bundle.mz_input_evidence['status'] == value['expected'], bundle.validation.errors
    assert bundle.validation.ok == (value['expected'] != 'INVALID')
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('session.json'))
        assert metadata['test_fixture_provenance']['classification'] == 'SYNTHETIC_FORMAT_FIXTURE'
        assert metadata['test_fixture_provenance']['device_run'] == 'NOT_RUN'
        assert metadata['measurement_purpose'] == DEVICE
        assert json.loads(archive.read('report.json'))['synthetic'] is True
        for entry, digest in value['unchanged_entries'].items():
            assert hashlib.sha256(archive.read(entry)).hexdigest() == digest
        if bundle.validation.ok:
            evidence = bundle.mz_input_evidence
            assert evidence['measurement_purpose'] == DEVICE
            for row in evidence['rows']:
                assert all(type(row['source_stamp'][key]) is int for key in mz.STAMP[:-1])
    request.node.user_properties.append(('format_fixture', dict(name=name, status=value['expected'],
        sha256=value['sha256'], origin='SYNTHETIC_FORMAT_FIXTURE', DEVICE_RUN='NOT_RUN')))


def mutation(tmp_path, edit, source=None):
    with zipfile.ZipFile(source or format_path()) as archive:
        payloads = {n: archive.read(n) for n in archive.namelist()}
    session_doc = json.loads(payloads['session.json'])
    report = json.loads(payloads['report.json'])
    descriptor = json.loads(payloads['samples/schema.json'])
    rows = list(map(json.loads, payloads[mz.INDEX].splitlines()))
    edit(session_doc, report, descriptor, rows, payloads)
    for name, doc in (('session.json', session_doc), ('report.json', report), ('samples/schema.json', descriptor)):
        payloads[name] = json.dumps(doc).encode()
    payloads[mz.INDEX] = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()
    manifest = json.loads(payloads['manifest.json'])
    manifest['entry_sha256'] = {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items() if n != 'manifest.json'}
    payloads['manifest.json'] = json.dumps(manifest).encode()
    path = tmp_path / 'synthetic-format-negative.alprsession'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for n, b in payloads.items():
            archive.writestr(n, b)
    return path


PURPOSE_PATHS = ['session', 'summary', 'report', 'report_summary']
BAD_PURPOSES = ['DELETE', None, True, 1, [], {}, '', 'UNKNOWN', SYNTHETIC]


@pytest.mark.parametrize('where', PURPOSE_PATHS)
@pytest.mark.parametrize('value', BAD_PURPOSES)
def test_device_rejects_purpose_types_and_conflicts(tmp_path, where, value):
    def edit(s, r, d, rows, payloads):
        owner, key = {'session': (s, 'measurement_purpose'), 'summary': (s['mz_input_evidence'], 'purpose'),
            'report': (r['research_collection'], 'measurement_purpose'),
            'report_summary': (r['research_collection']['mz_input_evidence'], 'purpose')}[where]
        if value == 'DELETE':
            owner.pop(key)
        else:
            owner[key] = value
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert not result.validation.ok and result.mz_input_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('where', ['session', 'report', 'capability'])
@pytest.mark.parametrize('value', ['DELETE', None, False, 1, 'true', [], {}])
def test_device_requires_true_boolean_diagnostic_flags(tmp_path, where, value):
    def edit(s, r, d, rows, payloads):
        owner, key = {'session': (s, 'mz_input_evidence_enabled'),
            'report': (r['research_collection'], 'mz_input_evidence_enabled'),
            'capability': (d['capabilities'], mz.CAPABILITY)}[where]
        if value == 'DELETE':
            owner.pop(key)
        else:
            owner[key] = value
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert not result.validation.ok and result.mz_input_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('where', ['session', 'report', 'both'])
@pytest.mark.parametrize('value', ['DELETE', None, False, 1, [], {}, '', 'SYNTHETIC-FORMAT',
    'F5A-ROI-A12-01', 'F5A-ROI-A12-01-PILOT', 'DEVICE-DIAGNOSTIC-MZ-', 'DEVICE-DIAGNOSTIC-MZ-X '])
def test_device_series_types_prefix_and_consistency(tmp_path, where, value):
    def edit(s, r, d, rows, payloads):
        owners = [s] if where == 'session' else [r['research_collection']] if where == 'report' else [s, r['research_collection']]
        for owner in owners:
            if value == 'DELETE':
                owner.pop('series_id')
            else:
                owner['series_id'] = value
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert not result.validation.ok and result.mz_input_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('value', [True, 9007199254740994.0, 2**63, 9007199254740995])
def test_device_source_stamp_exactness(tmp_path, value):
    def edit(s, r, d, rows, payloads):
        rows[0]['source_stamp']['source_timestamp_nanos'] = value
    result = read_mobile_report_bundle(mutation(tmp_path, edit))
    assert result.mz_input_evidence['status'] == 'INVALID'


@pytest.mark.parametrize('name,variant', [('mz-written-r0', 'R0'), ('mz-written-r1', 'R1'), ('mz-written-r2-two-roi', 'R2')])
def test_device_identity_conditions_and_read_only_view(tmp_path, name, variant):
    from auto_annotation_tool.ranking.mobile_experiment_identity import from_bundle, from_review_provenance, is_f5a
    from auto_annotation_tool.ranking.mobile_scientific_review import session_conditions, session_conditions_text
    s = session(tmp_path, format_path(name))
    before = copy.deepcopy(s.review.to_dict())
    identity = from_bundle(s.bundle)
    assert identity['experiment'] == identity['measurement_purpose'] == DEVICE
    assert identity['family'] == 'DIAGNOSTIC' and identity['status'] == 'DIAGNOSTIC_ONLY'
    assert identity['variant'] == variant and identity['experiment_type'] == 'device_diagnostic'
    assert not identity['is_pilot'] and not is_f5a(s.bundle)
    assert from_review_provenance(s.review.provenance, verified=True)['experiment'] == DEVICE
    conditions = session_conditions(s)
    assert conditions['measurement_purpose'] == DEVICE and conditions['variant'] == variant
    assert DEVICE in session_conditions_text(conditions)
    assert DEVICE in mz.diagnostic_text(s.mz_input_evidence)
    assert s.review.to_dict() == before and not s.review.scene_annotations and not s.review.invocation_annotations


@pytest.mark.parametrize('question', ['F4', 'F5A'])
def test_device_not_official_analysis(tmp_path, question):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    s = session(tmp_path, format_path())
    inp = make_input(s.bundle.report, source_path=s.path, results_root=tmp_path / '03_wyniki', review_root=tmp_path / 'reviews')
    result = analyze([inp], question=question).sessions[0]
    assert result.source_status == 'VERIFIED', result.source_reason
    assert result.metadata['experiment'] == DEVICE and not result.metadata['is_pilot']
    assert not result.metadata['registry_verified']
    assert all(m.status == 'NOT_COMPARABLE' and m.value is None for m in result.metrics.values())


@pytest.mark.parametrize('other', [SYNTHETIC, 'F4', 'F5A', 'F5A-PILOT'])
def test_native_comparison_rejects_mixed_device_cohort(tmp_path, other):
    from auto_annotation_tool.gui.mobile_import_comparison import record_from_import
    from auto_annotation_tool.gui.mobile_comparison_core import validate_selection
    from auto_annotation_tool.gui.mobile_comparison_scope import validate_comparison
    s = session(tmp_path, format_path())
    record = record_from_import(s.bundle.report, s.bundle, include_review=False)[0]
    assert record.measurement_purpose == DEVICE
    # Negative unit-test metadata only; no new source/attempt/MT IDs are minted.
    second = replace(record, measurement_purpose=SYNTHETIC if other == SYNTHETIC else '')
    for validate in (validate_selection, validate_comparison):
        with pytest.raises(ValueError, match='NOT_COMPARABLE'):
            validate([record, second])
    assert any(DEVICE in w for w in validate_selection([record]))


def test_native_analysis_device_and_synthetic_are_not_comparable(tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    a = session(tmp_path, format_path())
    b = session(tmp_path, original_path('mz-written-r1'))
    inputs = [make_input(s.bundle.report, source_path=s.path, results_root=tmp_path / '03_wyniki', review_root=tmp_path / 'reviews') for s in (a, b)]
    result = analyze(inputs, question='DIAGNOSTIC')
    assert len(result.sessions) == 2 and not result.duplicates
    assert {s.metadata['measurement_purpose'] for s in result.sessions} == {DEVICE, SYNTHETIC}
    for s in result.sessions:
        assert s.source_status == 'VERIFIED', s.source_reason
        assert all(m.status == 'NOT_COMPARABLE' and m.value is None
                   and 'DIAGNOSTIC_PURPOSE_COHORT_MISMATCH' in m.missing_reasons for m in s.metrics.values())


@pytest.mark.parametrize('name,purpose,status', [('device-written', DEVICE, 'VALID'),
    ('device-missing', DEVICE, 'PARTIAL'), ('synthetic-written', SYNTHETIC, 'VALID'), ('legacy-absent', None, 'ABSENT')])
def test_T19_full_native_roundtrip(tmp_path, monkeypatch, name, purpose, status, request):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, InvalidExport
    source = {'device-written': lambda: format_path(), 'device-missing': lambda: format_path('mz-memory-missing'),
              'synthetic-written': lambda: original_path('mz-written-r1'), 'legacy-absent': lambda: original_path('mz-off')}[name]()
    s = session(tmp_path, source)
    s.set_review_mode('assisted')
    before = json.loads(json.dumps(s.review.to_dict()))
    s = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert s.review.to_dict() == before
    monkeypatch.setattr(native, 'source_session', lambda _: s)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, h, ed, s = next(generator)
    try:
        h.window.title('F5A19 SYNTHETIC FORMAT FIXTURE - DEVICE_RUN=NOT_RUN')
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
        exported = load_session_export(folder)
        assert exported.mz_input_evidence['status'] == status
        assert exported.mz_input_evidence == s.mz_input_evidence
        if purpose:
            assert exported.measurement_purpose == purpose and exported.label.startswith(purpose)
            science = json.loads(next(folder.glob('ocena_naukowa_*.json')).read_text(encoding='utf-8'))
            assert science['measurement_purpose'] == science['scientific_state'] == purpose
        reopened = MobileReviewSession.open(s.path, sidecar_path=folder / 'review.json')
        assert reopened.review.to_dict() == before and reopened.mz_input_evidence == s.mz_input_evidence
        assert not reopened.review.scene_annotations and not reopened.review.invocation_annotations
        assert h.conditions_view.cget('state') == 'disabled'
        assert 'MZ input evidence: ' + status in h.conditions_view.get('1.0', 'end')
        if purpose:
            assert purpose in h.conditions_view.get('1.0', 'end')
        mf = json.loads(manifests[0].read_text(encoding='utf-8'))
        for dep in mf['files']:
            assert hashlib.sha256((folder / dep['path']).read_bytes()).hexdigest() == dep['sha256']
        for row in exported.mz_input_evidence.get('rows', []):
            assert all(type(row['source_stamp'][k]) is int for k in mz.STAMP[:-1])
            if row['tensor']['write_status'] == 'WRITTEN':
                tensor = row['tensor']
                data = (folder / mz.NAMESPACE / tensor['entry']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == tensor['sha256']
        assert hashlib.sha256(s.path.read_bytes()).hexdigest() == s.bundle.source_archive_sha256
        request.node.user_properties.append(('roundtrip', dict(case=name, purpose=purpose, status=status,
            source_sha256=s.bundle.source_archive_sha256, manifest_sha256=hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
            dependencies=len(mf['files']), origin='SYNTHETIC_FORMAT_FIXTURE' if purpose == DEVICE else 'ORIGINAL_ANDROID_SYNTHETIC_FIXTURE',
            DEVICE_RUN='NOT_RUN', full_native_export=True,
            source_int64=[r['source_stamp']['source_timestamp_nanos'] for r in exported.mz_input_evidence.get('rows', [])])))
        if purpose == DEVICE:
            # Resigning the outer export manifest cannot relabel DEVICE as SYNTHETIC.
            descriptor_path = folder / mz.EXPORT_FILE
            original_descriptor = descriptor_path.read_bytes()
            original_manifest = manifests[0].read_bytes()
            descriptor = json.loads(original_descriptor)
            descriptor['measurement_purpose'] = SYNTHETIC
            descriptor_path.write_text(json.dumps(descriptor), encoding='utf-8')
            for dep in mf['files']:
                if dep['path'] == mz.EXPORT_FILE:
                    dep.update(bytes=descriptor_path.stat().st_size, sha256=hashlib.sha256(descriptor_path.read_bytes()).hexdigest())
            manifests[0].write_text(json.dumps(mf), encoding='utf-8')
            with pytest.raises(InvalidExport, match='Export status conflict'):
                load_session_export(folder)
            descriptor_path.write_bytes(original_descriptor)
            manifests[0].write_bytes(original_manifest)
    finally:
        try:
            next(generator)
        except StopIteration:
            pass
